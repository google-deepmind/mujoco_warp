# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Backward-only qpos VJPs of the smooth pipeline (kinematics, com_pos, com_vel, rne)."""

import numpy as np
import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import collision_adjoint
from mujoco_warp._src import math
from mujoco_warp._src import smooth
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec10
from mujoco_warp._src.types import vec10f
from mujoco_warp._src.warp_util import event_scope

# adjoint module: backward stays on so AD leaves differentiate through cross-module @wp.funcs
wp.set_module_options({"enable_backward": True})

_SV = wp.spatial_vector
_FREE = int(types.JointType.FREE.value)
_BALL = int(types.JointType.BALL.value)
_SPRING = int(types.DisableBit.SPRING.value)
_ACTUATION = int(types.DisableBit.ACTUATION.value)
_CLAMPCTRL = int(types.DisableBit.CLAMPCTRL.value)
_GRAVITY = int(types.DisableBit.GRAVITY.value)
_GAIN_AFFINE = int(types.GainType.AFFINE.value)
_BIAS_AFFINE = int(types.BiasType.AFFINE.value)


# ----------------------------------------------------------------------------
# Per-depth tree-reduction reverses over m.body_tree: fast equivalents of the O(nbody^2)
# ancestry-walk kernels below. All enable_backward=False (manual VJP), all out-of-place across
# depths: a body reads only an already-finalized parent/child depth, so writes never race.
# ----------------------------------------------------------------------------
# root->leaves ancestor accumulation; launch once per m.body_tree depth in forward order
@wp.kernel(enable_backward=False)
def _anc_acc_sv(
  # Model:
  body_parentid: wp.array[int],
  # In:
  body_tree_level: wp.array[int],
  # Out:
  val_io_out: wp.array2d[wp.spatial_vector],  # init = local; root->leaves: val[b] += val[parent(b)]
):
  w, nodeid = wp.tid()
  b = body_tree_level[nodeid]
  if b != 0:
    val_io_out[w, b] = val_io_out[w, b] + val_io_out[w, body_parentid[b]]


# leaves->root subtree sum; launch per depth reversed; mirrors smooth._subtree_com_acc
@wp.kernel(enable_backward=False)
def _subtree_acc_sv(
  # Model:
  body_parentid: wp.array[int],
  # In:
  body_tree_level: wp.array[int],
  # Out:
  val_io_out: wp.array2d[wp.spatial_vector],  # init = local; leaves->root: val[parent] += val[b]
):
  w, nodeid = wp.tid()
  b = body_tree_level[nodeid]
  if b != 0:
    wp.atomic_add(val_io_out, w, body_parentid[b], val_io_out[w, b])


# io_out += src (merges a contact subtree-COM seed into the bias's)
@wp.kernel(enable_backward=False)
def _acc_vec3(src: wp.array2d[wp.vec3], io_out: wp.array2d[wp.vec3]):
  w, i = wp.tid()
  io_out[w, i] = io_out[w, i] + src[w, i]


# CV3 augmented-seed subtree sum; launch per depth reversed (children finalized first)
@wp.kernel(enable_backward=False)
def _comvel_W_acc(
  # Model:
  body_parentid: wp.array[int],
  # In:
  body_tree_level: wp.array[int],
  H_in: wp.array2d[wp.spatial_vector],
  # Out:
  W_io_out: wp.array2d[wp.spatial_vector],  # init = adj_cvel (A); leaves->root: W[parent] += W[b] + H[b]
):
  w, nodeid = wp.tid()
  b = body_tree_level[nodeid]
  if b != 0:
    wp.atomic_add(W_io_out, w, body_parentid[b], W_io_out[w, b] + H_in[w, b])


# ----------------------------------------------------------------------------
# The shared reduced reverse.
# ----------------------------------------------------------------------------
@event_scope
def smooth_force_backward(
  m: Model,
  d: Data,
  lam: wp.array2d,
  flg_acc: bool = True,
  res_cdof_extra: wp.array2d = None,
  res_subtree_extra: wp.array2d = None,
):
  """Computes the reduced smooth-force qpos VJP over converged d.

  res_qpos (nq) = d(lam^T qfrc_bias)/dqpos. adj_qvel is discarded (the qvel channel is owned by
  derivative.deriv_smooth_vel); optional extras merge the contact residual's cdof / subtree-COM
  cotangents so the shared reverse runs once.
  """
  nworld = d.qvel.shape[0]
  nv = m.nv
  nq = d.qpos.shape[1]
  nbody = m.nbody

  # ---- rne reverse (K1, K2', K3, K4a', K4b) ----
  # TODO(etaoxing): scratch preallocation for graph capture
  adj_cdof = wp.zeros((nworld, nv), dtype=_SV)
  adj_cdof_dot = wp.zeros((nworld, nv), dtype=_SV)
  adj_qvel = wp.zeros((nworld, nv), dtype=float)  # discarded (deriv_smooth_vel owns qvel)
  adj_qacc = wp.zeros((nworld, nv), dtype=float)  # discarded here (rne_backward exposes it for the qacc gate)
  adj_force = wp.zeros((nworld, nbody), dtype=_SV)

  # K1: lam -> adj_cdof (tau projection) + adj_force (body force seed)
  wp.launch(_rne_qfrcbias_cdof_vjp, dim=(nworld, nv), inputs=[m.dof_bodyid, d.cfrc_int, lam], outputs=[adj_cdof])
  wp.launch(
    _rne_qfrcbias_force_vjp, dim=(nworld, nbody), inputs=[m.body_dofnum, m.body_dofadr, d.cdof, lam], outputs=[adj_force]
  )

  # K2' (per-depth): adj_force -> adj_f (ancestor accumulation, root->leaves)
  adj_f = wp.clone(adj_force)
  for body_tree in m.body_tree:
    wp.launch(_anc_acc_sv, dim=(nworld, body_tree.size), inputs=[m.body_parentid, body_tree], outputs=[adj_f])

  # K3: adj_f -> adj_{cinert,cacc,cvel} via source-AD of the local inertial-force leaf
  cfrc_local = wp.zeros((nworld, nbody), dtype=_SV)
  adj_cinert = wp.zeros((nworld, nbody), dtype=vec10f)
  adj_cacc = wp.zeros((nworld, nbody), dtype=_SV)
  adj_cvel = wp.zeros((nworld, nbody), dtype=_SV)
  cfrc_inputs = [d.cinert, d.cvel, d.cacc]
  wp.launch(_rne_cfrc_recompute, dim=(nworld, nbody), inputs=cfrc_inputs, outputs=[cfrc_local])
  wp.launch(
    _rne_cfrc_recompute,
    dim=(nworld, nbody),
    inputs=cfrc_inputs,
    outputs=[cfrc_local],
    adj_inputs=[adj_cinert, adj_cvel, adj_cacc],
    adj_outputs=[adj_f],
    adjoint=True,
  )

  # K4a' (per-depth): adj_cacc -> subtree_adj_cacc (subtree sum, leaves->root)
  subtree_adj_cacc = wp.clone(adj_cacc)
  for body_tree in reversed(m.body_tree):
    wp.launch(_subtree_acc_sv, dim=(nworld, body_tree.size), inputs=[m.body_parentid, body_tree], outputs=[subtree_adj_cacc])

  # K4b: subtree_adj_cacc -> adj_{qvel,qacc,cdof_dot, cdof +=} (transpose the cacc dof sweep)
  wp.launch(
    _rne_cacc_dof_vjp,
    dim=(nworld, nv),
    inputs=[m.dof_bodyid, d.qvel, d.qacc, d.cdof, d.cdof_dot, subtree_adj_cacc, flg_acc],
    outputs=[adj_qvel, adj_qacc, adj_cdof_dot, adj_cdof],
  )

  # ---- com_vel reverse (CV1, CV2, CV3', CV4): Coriolis cdof path; adj_qvel discarded ----
  h = wp.zeros((nworld, nv), dtype=_SV)
  kk = wp.zeros((nworld, nv), dtype=_SV)
  Hbody = wp.zeros((nworld, nbody), dtype=_SV)
  cv_adj_qvel = wp.zeros((nworld, nv), dtype=float)  # discarded
  cv_adj_cdof = wp.zeros((nworld, nv), dtype=_SV)
  wp.launch(
    _comvel_vjp_local,
    dim=(nworld, nbody),
    inputs=[m.body_parentid, m.body_jntnum, m.body_jntadr, m.jnt_type, m.jnt_dofadr, d.qvel, d.cdof, d.cvel, adj_cdof_dot],
    outputs=[h, kk, Hbody],
  )
  wp.launch(
    _comvel_vjp_samebody,
    dim=(nworld, nbody),
    inputs=[m.body_jntnum, m.body_jntadr, m.jnt_type, m.jnt_dofadr, d.qvel, d.cdof, h, kk],
    outputs=[cv_adj_qvel, cv_adj_cdof],
  )
  W = wp.clone(adj_cvel)  # CV3' (per-depth): W_B = sum_subtree(B) adj_cvel + sum_strict_subtree(B) H
  for body_tree in reversed(m.body_tree):
    wp.launch(_comvel_W_acc, dim=(nworld, body_tree.size), inputs=[m.body_parentid, body_tree, Hbody], outputs=[W])
  wp.launch(_comvel_scatter_W, dim=(nworld, nv), inputs=[m.dof_bodyid, d.qvel, d.cdof, W], outputs=[cv_adj_qvel, cv_adj_cdof])

  # total cdof cotangent = rne-proper + com_vel (+ optional contact res_cdof seed -> one shared
  # reverse)
  total_cdof = wp.zeros((nworld, nv), dtype=_SV)
  wp.launch(_add_spatial, dim=(nworld, nv), inputs=[adj_cdof, cv_adj_cdof], outputs=[total_cdof])
  if res_cdof_extra is not None:
    wp.launch(_add_spatial, dim=(nworld, nv), inputs=[total_cdof, res_cdof_extra], outputs=[total_cdof])

  # ---- kinematic reverse: adj_{cinert, total_cdof} -> qpos ----
  res_dof = wp.zeros((nworld, nv), dtype=float)
  adj_subtree = cinert_qpos_vjp(m, d, adj_cinert, res_dof)  # _cinert leaf + pose->dof; subtree seed
  if res_subtree_extra is not None:  # merge the contact subtree-COM seed into the bias's
    wp.launch(_acc_vec3, dim=(nworld, nbody), inputs=[res_subtree_extra], outputs=[adj_subtree])
  wp.launch(
    collision_adjoint._cdof_qpos_vjp,
    dim=(nworld, nv),
    inputs=[nv, m.jnt_type, m.jnt_dofadr, m.dof_jntid, m.dof_parentid, d.cdof, total_cdof],
    outputs=[res_dof],
  )
  ceff = wp.empty((nworld, nbody), dtype=wp.vec3)
  wp.launch(
    collision_adjoint._build_ceff,
    dim=(nworld, nbody),
    inputs=[nv, m.body_rootid, m.dof_bodyid, d.cdof, total_cdof, adj_subtree],
    outputs=[ceff],
  )
  wp.launch(
    collision_adjoint._subtree_com_qpos_vjp,
    dim=(nworld, nv),
    inputs=[
      nbody,
      m.body_parentid,
      m.body_rootid,
      m.body_mass,
      m.body_subtreemass,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.subtree_com,
      d.cdof,
      ceff,
    ],
    outputs=[res_dof],
  )
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  wp.launch(
    collision_adjoint._dof_to_qpos,
    dim=(nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof],
    outputs=[res_qpos],
  )
  return res_qpos


@event_scope
def mass_matrix_qpos_vjp(m: Model, d: Data, y: wp.array2d, w: wp.array2d, scratch: Data = None):
  """Computes the mass-matrix-derivative VJP: res_qpos = d_q[ y^T M(q) w ] with y, w fixed.

  d is never mutated. scratch (a BackwardContext.scratch: off-tape non-grad Data of d's shape) is
  reused when given, refreshing only the kinematic inputs (qpos, plus mocap) because
  kinematics/com_pos/rne overwrite every other field before reading it; otherwise a fresh full
  clone is allocated (the per-call path). Reuse is graph-capture-safe (the T sequential
  per-backward calls share it on one stream) and drops ~165 array-copies/call to ~1 (the qpos
  copy), which the capture bakes into the graph.
  """
  if scratch is None:
    s = adjoint_util._clone_nograd(d)  # non-grad scratch; s.qpos = d.qpos
  else:
    s = scratch  # reuse the preallocated scratch; refresh only what kinematics reads from d
    wp.copy(s.qpos, d.qpos)
    if m.nmocap > 0:
      wp.copy(s.mocap_pos, d.mocap_pos)
      wp.copy(s.mocap_quat, d.mocap_quat)
  s.qvel.zero_()
  s.qacc = w
  smooth.kinematics(m, s)  # fresh xipos/ximat/cdof at d.qpos (the cloned kinematics are a step stale)
  smooth.com_pos(m, s)  # fresh subtree_com/cinert/cdof at d.qpos
  s.cvel.zero_()  # qvel = 0 -> no cvel / cdof_dot (skip com_vel); mass-only has no Coriolis
  s.cdof_dot.zero_()
  s.cacc.zero_()  # root cacc = 0 -> no gravity injected (mass-only)
  smooth._rne_cacc_forward(m, s, flg_acc=True)  # cacc = sum cdof*w  (the cdof_dot*qvel term is 0)
  smooth._rne_cfrc(m, s)  # cfrc_int_local = cinert*cacc  (cvel = 0 -> no gyroscopic)
  smooth._rne_cfrc_backward(m, s)  # subtree-accumulate cfrc_int up the tree
  return smooth_force_backward(m, s, y, flg_acc=True)  # = d_q[ y^T M(q) w ]


# ----------------------------------------------------------------------------
# Joint-spring leaf: the complete spring law of passive._spring_damper_dof_passive (all types
# incl. FREE/BALL quaternion springs) as a source-AD leaf. Reads qpos directly, so its adjoint
# lands in raw-coordinate space (no manual tangent lift); the viscous -damper*qvel term has zero
# qpos derivative and is omitted. An additive res_qpos term, not part of the kinematic reverse.
# ----------------------------------------------------------------------------
# exact reconstruction of the spring branch of passive._spring_damper_dof_passive (all types)
@wp.kernel(enable_backward=True)
def _spring_qfrc_recompute(
  # Model:
  opt_disableflags: int,
  qpos_spring: wp.array2d[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_stiffness: wp.array2d[float],
  jnt_stiffnesspoly: wp.array2d[wp.vec2],
  # Data in:
  qpos_in: wp.array2d[float],
  # Data out:
  qfrc_spring_out: wp.array2d[float],
):
  w, jntid = wp.tid()
  jnttype = jnt_type[jntid]
  dofid = jnt_dofadr[jntid]
  stiffness = jnt_stiffness[w % jnt_stiffness.shape[0], jntid]
  spoly = jnt_stiffnesspoly[w % jnt_stiffnesspoly.shape[0], jntid]
  has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and (opt_disableflags & _SPRING) == 0
  if not has_stiffness:
    return
  qposid = jnt_qposadr[jntid]
  sid = w % qpos_spring.shape[0]
  if jnttype == _FREE:
    difx = qpos_in[w, qposid + 0] - qpos_spring[sid, qposid + 0]
    dify = qpos_in[w, qposid + 1] - qpos_spring[sid, qposid + 1]
    difz = qpos_in[w, qposid + 2] - qpos_spring[sid, qposid + 2]
    r = wp.length(wp.vec3(difx, dify, difz))
    k = util_misc._poly_force(stiffness, spoly, r, 0)
    qfrc_spring_out[w, dofid + 0] = -k * difx
    qfrc_spring_out[w, dofid + 1] = -k * dify
    qfrc_spring_out[w, dofid + 2] = -k * difz
    rot = wp.normalize(wp.quat(qpos_in[w, qposid + 3], qpos_in[w, qposid + 4], qpos_in[w, qposid + 5], qpos_in[w, qposid + 6]))
    ref = wp.quat(
      qpos_spring[sid, qposid + 3], qpos_spring[sid, qposid + 4], qpos_spring[sid, qposid + 5], qpos_spring[sid, qposid + 6]
    )
    dif = math.quat_sub(rot, ref)
    k_rot = util_misc._poly_force(stiffness, spoly, wp.length(dif), 0)
    qfrc_spring_out[w, dofid + 3] = -k_rot * dif[0]
    qfrc_spring_out[w, dofid + 4] = -k_rot * dif[1]
    qfrc_spring_out[w, dofid + 5] = -k_rot * dif[2]
  elif jnttype == _BALL:
    rot = wp.normalize(wp.quat(qpos_in[w, qposid + 0], qpos_in[w, qposid + 1], qpos_in[w, qposid + 2], qpos_in[w, qposid + 3]))
    ref = wp.quat(
      qpos_spring[sid, qposid + 0], qpos_spring[sid, qposid + 1], qpos_spring[sid, qposid + 2], qpos_spring[sid, qposid + 3]
    )
    dif = math.quat_sub(rot, ref)
    k = util_misc._poly_force(stiffness, spoly, wp.length(dif), 0)
    qfrc_spring_out[w, dofid + 0] = -k * dif[0]
    qfrc_spring_out[w, dofid + 1] = -k * dif[1]
    qfrc_spring_out[w, dofid + 2] = -k * dif[2]
  else:  # SLIDE / HINGE
    fdif = qpos_in[w, qposid] - qpos_spring[sid, qposid]
    qfrc_spring_out[w, dofid] = -fdif * util_misc._poly_force(stiffness, spoly, fdif, 0)


@event_scope
def spring_qpos_vjp(m: Model, d: Data, lam: wp.array2d):
  """Computes the spring qpos VJP for all joint types; d is read-only.

  res_qpos = d(lam^T(-qfrc_spring))/dqpos.
  """
  nworld = d.qpos.shape[0]
  nv = m.nv
  nq = d.qpos.shape[1]
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  qfrc_spring = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
  sp_in = [
    m.opt.disableflags,
    m.qpos_spring,
    m.jnt_type,
    m.jnt_qposadr,
    m.jnt_dofadr,
    m.jnt_stiffness,
    m.jnt_stiffnesspoly,
    d.qpos,
  ]
  wp.launch(_spring_qfrc_recompute, dim=(nworld, m.njnt), inputs=sp_in, outputs=[qfrc_spring])
  wp.launch(adjoint_util._neg_cols, dim=(nworld, nv), inputs=[lam], outputs=[qfrc_spring.grad])  # seed -lam
  wp.launch(
    _spring_qfrc_recompute,
    dim=(nworld, m.njnt),
    inputs=sp_in,
    outputs=[qfrc_spring],
    adj_inputs=[None, None, None, None, None, None, None, res_qpos],  # qpos adjoint -> res_qpos
    adj_outputs=[qfrc_spring.grad],
    adjoint=True,
  )
  return res_qpos


# ----------------------------------------------------------------------------
# Actuator leaf: reverse of qfrc_actuator for AFFINE gain/bias on a scalar-JOINT transmission
# (seed adj_tau = -lam): res_dof[j] += -ml_a*dfdl_a*B_aj under the frozen saturation mask. The
# moment B is constant in qpos for JOINT (adj_B = 0); the velocity channel is owned by
# deriv_smooth_vel. Unsupported topologies are gated off by assert_smooth_supported, not FD.
# ----------------------------------------------------------------------------
@event_scope
def actuator_qpos_vjp(m: Model, d: Data, lam: wp.array2d):
  """Computes the affine JOINT-actuator qpos VJP over frozen d.

  res_qpos = d(lam^T(-qfrc_actuator))/dqpos.
  """
  nworld = d.qpos.shape[0]
  nv = m.nv
  nq = d.qpos.shape[1]
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  if m.nu == 0 or (int(m.opt.disableflags) & _ACTUATION) != 0:  # match forward: actuation suppressed
    return res_qpos
  res_dof = wp.zeros((nworld, nv), dtype=float)
  wp.launch(
    _actuator_qpos_vjp,
    dim=(nworld, m.nu),
    inputs=[
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_ctrllimited,
      m.actuator_forcelimited,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_ctrlrange,
      m.actuator_forcerange,
      d.ctrl,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.actuator_force,
      lam,
      int(m.opt.disableflags) & _CLAMPCTRL,
    ],
    outputs=[res_dof],
  )
  wp.launch(
    collision_adjoint._dof_to_qpos,
    dim=(nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof],
    outputs=[res_qpos],
  )
  return res_qpos


# capability-check result per Model, keyed by id(m); the value holds a strong ref to m so a
# GC-reused id() cannot alias a stale entry (in-place structural mutation of m stays undetected).
# adjoint.create_backward_context primes it at setup so the .numpy() reads never run inside a
# graph-capture region.
_SUPPORTED_CACHE = {}


def assert_smooth_supported(m: Model):
  """Raises NotImplementedError for enabled features missing an analytic VJP; no FD fallback."""
  key = id(m)
  if _SUPPORTED_CACHE.get(key) is not None:
    return
  bad = []
  if int(m.ntendon) != 0:
    bad.append("tendons (spring/bias/armature/transmission)")
  opt = m.opt
  if np.any(opt.density.numpy() != 0.0) or np.any(opt.viscosity.numpy() != 0.0) or np.any(opt.wind.numpy() != 0.0):
    bad.append("fluid forces (opt.density/viscosity/wind)")
  if np.any(m.body_gravcomp.numpy() != 0.0) and np.any(m.jnt_actgravcomp.numpy() != 0):
    bad.append("gravcomp routed to an actuator (jnt_actgravcomp; force-limit clamp); passive-bucket gravcomp is supported")
  if int(m.nu) > 0:
    trntype = m.actuator_trntype.numpy()
    gaintype = m.actuator_gaintype.numpy()
    biastype = m.actuator_biastype.numpy()
    jnt_type = m.jnt_type.numpy()
    if np.any(trntype != int(types.TrnType.JOINT.value)):
      bad.append("non-JOINT actuator transmission (tendon/site/body/slider-crank/jointinparent)")
    ok_gain = (gaintype == int(types.GainType.FIXED.value)) | (gaintype == int(types.GainType.AFFINE.value))
    ok_bias = (biastype == int(types.BiasType.NONE.value)) | (biastype == int(types.BiasType.AFFINE.value))
    if not ok_gain.all() or not ok_bias.all():
      bad.append("non-affine actuator gain/bias (muscle/DC-motor/user)")
    jnt_ids = m.actuator_trnid.numpy()[:, 0][trntype == int(types.TrnType.JOINT.value)]
    if jnt_ids.size and np.any((jnt_type[jnt_ids] == _FREE) | (jnt_type[jnt_ids] == _BALL)):
      bad.append("actuator on a FREE/BALL joint (quaternion-dependent transmission length)")
    # activation (na>0) is supported for the dqpos when gain is FIXED: dfdl = biasprm[1] and
    # ctrl_act (=act) is unused (the common filtered/integrated position/velocity servo). A
    # stateful AFFINE-gain actuator would need the gain term gainprm[1]*act (ctrl_act=act, not
    # clamped ctrl); not implemented.
    stateful = m.actuator_dyntype.numpy() != int(types.DynType.NONE.value)
    if np.any(stateful & (gaintype == int(types.GainType.AFFINE.value))):
      bad.append("stateful (na>0) AFFINE-gain actuator (ctrl_act=act not implemented)")
  if bad:
    raise NotImplementedError(
      "smooth_adjoint analytic reverse does not support: "
      + "; ".join(bad)
      + ". Implement the leaf/topology VJP (the FD oracle lives in adjoint_test_util). "
      "There is no silent FD fallback."
    )
  _SUPPORTED_CACHE[key] = (m, True)


# ----------------------------------------------------------------------------
# Gravcomp leaf: a COM-Jacobian contraction with a constant per-body force, so qpos enters only
# via the kinematic intermediates jac_dof reads (cdof, subtree_com, xipos). Source-AD the exact
# forward leaf, then route the adjoints through the same kinematic reverse the rne bias uses.
# Seed -lam masked by (gravity_enabled & ~actgravcomp); the actuator-bucket case is asserted off.
# ----------------------------------------------------------------------------
# exact reconstruction of passive._gravity_force (reuses support.jac_dof)
@wp.kernel(enable_backward=True)
def _gravity_force_recompute(
  # Model:
  opt_gravity: wp.array[wp.vec3],
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_gravcomp: wp.array2d[float],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # Data out:
  qfrc_gravcomp_out: wp.array2d[float],
):
  worldid, bodyid, dofid = wp.tid()
  bodyid += 1  # skip world body
  gravcomp = body_gravcomp[worldid % body_gravcomp.shape[0], bodyid]
  gravity = opt_gravity[worldid % opt_gravity.shape[0]]
  if gravcomp:
    force = -gravity * body_mass[worldid % body_mass.shape[0], bodyid] * gravcomp
    pos = xipos_in[worldid, bodyid]
    jac, _ = support.jac_dof(
      body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, pos, bodyid, dofid, worldid
    )
    wp.atomic_add(qfrc_gravcomp_out[worldid], dofid, wp.dot(jac, force))


@wp.kernel(enable_backward=False)
def _gravcomp_seed(
  # Model:
  jnt_actgravcomp: wp.array[int],
  dof_jntid: wp.array[int],
  # In:
  lam: wp.array2d[float],
  gravity_enabled: int,
  # Out:
  grad_out: wp.array2d[float],  # qfrc_gravcomp.grad = -lam masked by (gravity_enabled & not actgravcomp)
):
  w, i = wp.tid()
  # -lam: gravcomp enters the residual as -qfrc_gravcomp (passive bucket), so seeding -lam on
  # qfrc_gravcomp drops -(dqfrc_gravcomp/dqpos)lam onto res_qpos via the kinematic reverse.
  if gravity_enabled != 0 and jnt_actgravcomp[dof_jntid[i]] == 0:
    grad_out[w, i] = -lam[w, i]
  else:
    grad_out[w, i] = 0.0


# has-gravcomp result per Model, keyed by id(m); the value holds a strong ref to m so a GC-reused
# id() cannot alias a stale entry (in-place structural mutation of m stays undetected).
# create_backward_context primes it (like assert_smooth_supported) so the .numpy() read never
# runs inside a graph-capture region.
_HAS_GRAVCOMP_CACHE = {}


def has_gravcomp(m: Model) -> bool:
  """Returns whether any body has passive-bucket gravcomp (result cached per Model).

  The .numpy() host read runs once, eager; prime it at setup (create_backward_context), never
  inside a graph-capture region.
  """
  key = id(m)
  cached = _HAS_GRAVCOMP_CACHE.get(key)
  if cached is not None:
    return cached[1]
  has = bool(np.any(m.body_gravcomp.numpy() != 0.0))
  _HAS_GRAVCOMP_CACHE[key] = (m, has)
  return has


@event_scope
def gravcomp_qpos_vjp(m: Model, d: Data, lam: wp.array2d):
  """Computes the gravcomp qpos VJP (passive bucket); d is read-only.

  res_qpos = d(lam^T(-qfrc_gravcomp))/dqpos.
  """
  nworld = d.qpos.shape[0]
  nv = m.nv
  nq = d.qpos.shape[1]
  nbody = m.nbody
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  if not has_gravcomp(m):
    return res_qpos

  # 1. recompute qfrc_gravcomp on differentiable kinematic intermediates; seed -lam (masked) ->
  #    adjoints
  xipos = wp.clone(d.xipos)
  xipos.requires_grad = True
  subtree = wp.clone(d.subtree_com)
  subtree.requires_grad = True
  cdof = wp.clone(d.cdof)
  cdof.requires_grad = True
  qfrc_gc = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
  gc_in = [
    m.opt.gravity,
    m.body_parentid,
    m.body_rootid,
    m.body_mass,
    m.body_gravcomp,
    m.dof_bodyid,
    m.body_isdofancestor,
    xipos,
    subtree,
    cdof,
  ]
  wp.launch(_gravity_force_recompute, dim=(nworld, nbody - 1, nv), inputs=gc_in, outputs=[qfrc_gc])
  gravity_enabled = 1 if (int(m.opt.disableflags) & _GRAVITY) == 0 else 0
  wp.launch(
    _gravcomp_seed, dim=(nworld, nv), inputs=[m.jnt_actgravcomp, m.dof_jntid, lam, gravity_enabled], outputs=[qfrc_gc.grad]
  )
  adj_xipos = wp.zeros((nworld, nbody), dtype=wp.vec3)
  adj_subtree = wp.zeros((nworld, nbody), dtype=wp.vec3)
  adj_cdof = wp.zeros((nworld, nv), dtype=_SV)
  wp.launch(
    _gravity_force_recompute,
    dim=(nworld, nbody - 1, nv),
    inputs=gc_in,
    outputs=[qfrc_gc],
    adj_inputs=[None, None, None, None, None, None, None, adj_xipos, adj_subtree, adj_cdof],
    adj_outputs=[qfrc_gc.grad],
    adjoint=True,
  )

  # 2. route the kinematic-intermediate adjoints to qpos, the exact composition rne_qpos_vjp
  #    uses: cdof screw-commutator + (cdof moving-COM folded into the subtree seed via
  #    _build_ceff) + xipos FK.
  res_dof = wp.zeros((nworld, nv), dtype=float)
  wp.launch(
    collision_adjoint._cdof_qpos_vjp,
    dim=(nworld, nv),
    inputs=[nv, m.jnt_type, m.jnt_dofadr, m.dof_jntid, m.dof_parentid, d.cdof, adj_cdof],
    outputs=[res_dof],
  )
  ceff = wp.empty((nworld, nbody), dtype=wp.vec3)  # adj_subtree + the cdof moving-COM fold
  wp.launch(
    collision_adjoint._build_ceff,
    dim=(nworld, nbody),
    inputs=[nv, m.body_rootid, m.dof_bodyid, d.cdof, adj_cdof, adj_subtree],
    outputs=[ceff],
  )
  wp.launch(
    collision_adjoint._subtree_com_qpos_vjp,
    dim=(nworld, nv),
    inputs=[
      nbody,
      m.body_parentid,
      m.body_rootid,
      m.body_mass,
      m.body_subtreemass,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.subtree_com,
      d.cdof,
      ceff,
    ],
    outputs=[res_dof],
  )
  adj_ximat = wp.zeros((nworld, nbody), dtype=wp.mat33)  # gravcomp force is orientation-independent
  wp.launch(
    _cinert_pose_dof_vjp,
    dim=(nworld, nv),
    inputs=[
      nbody,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.ximat,
      d.subtree_com,
      d.cdof,
      adj_xipos,
      adj_ximat,
    ],
    outputs=[res_dof],
  )
  wp.launch(
    collision_adjoint._dof_to_qpos,
    dim=(nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof],
    outputs=[res_qpos],
  )
  return res_qpos


# ============================================================================================
# AD-RNE: analytic backward of smooth.rne, the production dqpos path in step_backward.
# FD-of-rne is an explicitly selected validation oracle in adjoint_test_util. smooth.rne is
# enable_backward=False, so the reverse is reconstructed here: source-AD the loop-free/
# alias-free nonlinear _cfrc leaf (exact), manual transposed VJPs for the two linear tree
# reductions (their forwards have dynamic loops + in-place aliasing, so a naive reverse is wrong).
# ============================================================================================

_SV_ZERO = wp.constant(wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


# K1: d(lam^T qfrc_bias)/dcdof_i = lam_i * F_body(i); += into adj_cdof_out (K4b adds qacc)
@wp.kernel(enable_backward=False)
def _rne_qfrcbias_cdof_vjp(
  # Model:
  dof_bodyid: wp.array[int],
  # Data in:
  cfrc_int_in: wp.array2d[wp.spatial_vector],  # accumulated F = d.cfrc_int_in
  # In:
  lam: wp.array2d[float],  # seed adj_qfrc_bias
  # Out:
  adj_cdof_out: wp.array2d[wp.spatial_vector],  # (+=): the tau_i = cdof_i*F_body(i) part
):
  w, i = wp.tid()
  b = dof_bodyid[i]
  adj_cdof_out[w, i] = adj_cdof_out[w, i] + lam[w, i] * cfrc_int_in[w, b]


# K1: d(lam^T qfrc_bias)/dF_b = sum_{i: body(i)=b} lam_i * cdof_i (one thread per body)
@wp.kernel(enable_backward=False)
def _rne_qfrcbias_force_vjp(
  # Model:
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  lam: wp.array2d[float],  # seed adj_qfrc_bias
  # Out:
  adj_force_out: wp.array2d[wp.spatial_vector],  # adj on the accumulated body force F_b
):
  w, b = wp.tid()
  dofadr = body_dofadr[b]
  dofnum = body_dofnum[b]
  acc = _SV_ZERO
  for k in range(dofnum):
    i = dofadr + k
    acc = acc + lam[w, i] * cdof_in[w, i]
  adj_force_out[w, b] = acc


# K2: transpose of the child->parent sum: adj_f_out[b] = sum of adj_F over ancestors-or-self
@wp.kernel(enable_backward=False)
def _rne_cfrc_tree_vjp(
  # Model:
  body_parentid: wp.array[int],
  # In:
  adj_force: wp.array2d[wp.spatial_vector],  # adj on accumulated F_b (from K1)
  # Out:
  adj_f_out: wp.array2d[wp.spatial_vector],  # adj on the local force f_b (pre-accumulation)
):
  w, b = wp.tid()
  acc = _SV_ZERO
  p = b
  while p > 0:
    acc = acc + adj_force[w, p]
    p = body_parentid[p]
  adj_f_out[w, b] = acc


# K3: exact reconstruction of smooth._cfrc's local inertial force (source-AD leaf)
@wp.kernel
def _rne_cfrc_recompute(
  # Data in:
  cinert_in: wp.array2d[vec10],
  cvel_in: wp.array2d[wp.spatial_vector],
  cacc_in: wp.array2d[wp.spatial_vector],
  # Out:
  cfrc_local_out: wp.array2d[wp.spatial_vector],  # f_b = I*a + v x* (I*v)
):
  w, b = wp.tid()
  frc = _SV_ZERO
  if b != 0:  # world body has no inertial force (mirrors smooth._cfrc's bodyid==0 branch)
    ci = cinert_in[w, b]
    cv = cvel_in[w, b]
    frc = math.inert_vec(ci, cacc_in[w, b]) + math.motion_cross_force(cv, math.inert_vec(ci, cv))
  cfrc_local_out[w, b] = frc


# K4a: subtree-sum of adj_cacc per body; O(nbody*depth) oracle form of _subtree_acc_sv
@wp.kernel(enable_backward=False)
def _rne_cacc_subtree_sum(
  # Model:
  nbody: int,
  body_parentid: wp.array[int],
  # In:
  adj_cacc: wp.array2d[wp.spatial_vector],
  # Out:
  subtree_adj_cacc_out: wp.array2d[wp.spatial_vector],  # sum_{b in subtree(target)} adj_cacc[b]
):
  w, target = wp.tid()
  acc = _SV_ZERO
  for b in range(nbody):
    p = b  # is target an ancestor-or-self of body b?
    while p > 0:
      if p == target:
        acc = acc + adj_cacc[w, b]
        break
      p = body_parentid[p]
  subtree_adj_cacc_out[w, target] = acc


# K4b: transpose of the dof contributions cacc_b += cdof_dot_i*qvel_i + cdof_i*qacc_i
@wp.kernel(enable_backward=False)
def _rne_cacc_dof_vjp(
  # Model:
  dof_bodyid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  # In:
  subtree_adj_cacc: wp.array2d[wp.spatial_vector],  # sum adj_cacc over body(j)'s subtree (from K4a)
  flg_acc: bool,
  # Out:
  adj_qvel_out: wp.array2d[float],
  adj_qacc_out: wp.array2d[float],
  adj_cdof_dot_out: wp.array2d[wp.spatial_vector],
  adj_cdof_out: wp.array2d[wp.spatial_vector],  # (+=): the cdof_i*qacc_i term (K1 added the tau term)
):
  w, i = wp.tid()
  a = subtree_adj_cacc[w, dof_bodyid[i]]
  adj_qvel_out[w, i] = wp.dot(a, cdof_dot_in[w, i])
  adj_cdof_dot_out[w, i] = qvel_in[w, i] * a
  if flg_acc:
    adj_qacc_out[w, i] = wp.dot(a, cdof_in[w, i])
    adj_cdof_out[w, i] = adj_cdof_out[w, i] + qacc_in[w, i] * a


def rne_backward(m: Model, d: Data, lam: wp.array2d, flg_acc: bool = True):
  """Computes the analytic VJP of smooth.rne.

  Maps the cotangent lam on qfrc_bias to adjoints of the RNE inputs. Returns dict {qvel, qacc,
  cdof, cdof_dot, cinert, cvel}: direct input adjoints plus the intermediate seeds for the
  kinematic reverse; d holds converged intermediates (read-only).
  """
  nworld = d.qvel.shape[0]
  nv = m.nv
  nbody = m.nbody
  SV = wp.spatial_vector

  adj_cdof = wp.zeros((nworld, nv), dtype=SV)
  adj_cdof_dot = wp.zeros((nworld, nv), dtype=SV)
  adj_qvel = wp.zeros((nworld, nv), dtype=float)
  adj_qacc = wp.zeros((nworld, nv), dtype=float)
  adj_force = wp.zeros((nworld, nbody), dtype=SV)
  adj_f = wp.zeros((nworld, nbody), dtype=SV)

  # K1: lam -> {adj_cdof (tau part), adj_force}
  wp.launch(_rne_qfrcbias_cdof_vjp, dim=(nworld, nv), inputs=[m.dof_bodyid, d.cfrc_int, lam], outputs=[adj_cdof])
  wp.launch(
    _rne_qfrcbias_force_vjp, dim=(nworld, nbody), inputs=[m.body_dofnum, m.body_dofadr, d.cdof, lam], outputs=[adj_force]
  )

  # K2: adj_force -> adj_f (transpose the child->parent accumulation)
  wp.launch(_rne_cfrc_tree_vjp, dim=(nworld, nbody), inputs=[m.body_parentid, adj_force], outputs=[adj_f])

  # K3: adj_f -> {adj_cinert, adj_cacc, adj_cvel} via autodiff of the reconstructed local leaf
  cfrc_local = wp.zeros((nworld, nbody), dtype=SV)
  adj_cinert = wp.zeros((nworld, nbody), dtype=vec10f)
  adj_cacc = wp.zeros((nworld, nbody), dtype=SV)
  adj_cvel = wp.zeros((nworld, nbody), dtype=SV)
  cfrc_inputs = [d.cinert, d.cvel, d.cacc]
  wp.launch(_rne_cfrc_recompute, dim=(nworld, nbody), inputs=cfrc_inputs, outputs=[cfrc_local])
  wp.launch(
    _rne_cfrc_recompute,
    dim=(nworld, nbody),
    inputs=cfrc_inputs,
    outputs=[cfrc_local],
    adj_inputs=[adj_cinert, adj_cvel, adj_cacc],
    adj_outputs=[adj_f],
    adjoint=True,
  )

  # K4: adj_cacc -> {adj_qvel, adj_qacc, adj_cdof_dot, adj_cdof +=} (transpose the cacc forward
  # sweep)
  subtree_adj_cacc = wp.zeros((nworld, nbody), dtype=SV)
  wp.launch(_rne_cacc_subtree_sum, dim=(nworld, nbody), inputs=[nbody, m.body_parentid, adj_cacc], outputs=[subtree_adj_cacc])
  wp.launch(
    _rne_cacc_dof_vjp,
    dim=(nworld, nv),
    inputs=[m.dof_bodyid, d.qvel, d.qacc, d.cdof, d.cdof_dot, subtree_adj_cacc, flg_acc],
    outputs=[adj_qvel, adj_qacc, adj_cdof_dot, adj_cdof],
  )

  return {
    "qvel": adj_qvel,
    "qacc": adj_qacc,
    "cdof": adj_cdof,
    "cdof_dot": adj_cdof_dot,
    "cinert": adj_cinert,
    "cvel": adj_cvel,
  }


# --------------------------------------------------------------------------------------------
# com_vel reverse: bound-free manual VJP of smooth.com_vel (the Coriolis path); maps the seeds
# adj_cvel / adj_cdof_dot to adj_qvel / adj_cdof in stages CV1-CV4 (dynamic bounds, no
# _CV_MAX_* truncation). Wiring constraint: step_backward already uses deriv_rne_vel for the
# smooth adj_qvel; wire exactly one of the two RNE-qvel paths or the Coriolis term double-counts.
# --------------------------------------------------------------------------------------------


# same-body scatter of a running prefix cotangent t: nubar_j += S_j*t, Sbar_j += nu_j*t
@wp.func
def _cv_scatter(
  # Data in:
  qvel_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  w: int,
  j: int,
  t: wp.spatial_vector,
  # Out:
  adj_qvel_out: wp.array2d[float],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
):
  adj_qvel_out[w, j] = adj_qvel_out[w, j] + wp.dot(cdof_in[w, j], t)
  adj_cdof_out[w, j] = adj_cdof_out[w, j] + qvel_in[w, j] * t


# CV1: replay a body's joints from the parent-cvel_in snapshot: h, k per dof, H per body
@wp.kernel(enable_backward=False)
def _comvel_vjp_local(
  # Model:
  body_parentid: wp.array[int],
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_cdof_dot: wp.array2d[wp.spatial_vector],  # G seed
  # Out:
  h_out: wp.array2d[wp.spatial_vector],  # per dof: cotangent on the snapshot velocity
  k_out: wp.array2d[wp.spatial_vector],  # per dof: direct cotangent on cdof_in
  H_out: wp.array2d[wp.spatial_vector],  # per body: sum h on this body
):
  w, b = wp.tid()
  zero = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  if b == 0:  # world body: no joints / snapshot
    H_out[w, 0] = zero
    return
  Hb = zero
  u = cvel_in[w, body_parentid[b]]  # parent snapshot (cvel_in[0]=0 for a tree root)
  jntadr = body_jntadr[b]
  jntnum = body_jntnum[b]
  for jj in range(jntnum):
    jt = jnt_type[jntadr + jj]
    d = jnt_dofadr[jntadr + jj]
    if jt == _FREE:  # add translations -> snapshot rotations (post-translation) -> add rotations
      u = (
        u
        + cdof_in[w, d + 0] * qvel_in[w, d + 0]
        + cdof_in[w, d + 1] * qvel_in[w, d + 1]
        + cdof_in[w, d + 2] * qvel_in[w, d + 2]
      )
      g3 = adj_cdof_dot[w, d + 3]
      g4 = adj_cdof_dot[w, d + 4]
      g5 = adj_cdof_dot[w, d + 5]
      h3 = math.motion_cross_force(cdof_in[w, d + 3], g3)
      h4 = math.motion_cross_force(cdof_in[w, d + 4], g4)
      h5 = math.motion_cross_force(cdof_in[w, d + 5], g5)
      h_out[w, d + 0] = zero
      h_out[w, d + 1] = zero
      h_out[w, d + 2] = zero
      h_out[w, d + 3] = h3
      h_out[w, d + 4] = h4
      h_out[w, d + 5] = h5
      k_out[w, d + 0] = zero
      k_out[w, d + 1] = zero
      k_out[w, d + 2] = zero
      k_out[w, d + 3] = (-1.0) * math.motion_cross_force(u, g3)
      k_out[w, d + 4] = (-1.0) * math.motion_cross_force(u, g4)
      k_out[w, d + 5] = (-1.0) * math.motion_cross_force(u, g5)
      u = (
        u
        + cdof_in[w, d + 3] * qvel_in[w, d + 3]
        + cdof_in[w, d + 4] * qvel_in[w, d + 4]
        + cdof_in[w, d + 5] * qvel_in[w, d + 5]
      )
      Hb = Hb + h3 + h4 + h5
    elif jt == _BALL:  # snapshot all 3 axes from the pre-ball cvel_in, then add
      g0 = adj_cdof_dot[w, d + 0]
      g1 = adj_cdof_dot[w, d + 1]
      g2 = adj_cdof_dot[w, d + 2]
      h0 = math.motion_cross_force(cdof_in[w, d + 0], g0)
      h1 = math.motion_cross_force(cdof_in[w, d + 1], g1)
      h2 = math.motion_cross_force(cdof_in[w, d + 2], g2)
      h_out[w, d + 0] = h0
      h_out[w, d + 1] = h1
      h_out[w, d + 2] = h2
      k_out[w, d + 0] = (-1.0) * math.motion_cross_force(u, g0)
      k_out[w, d + 1] = (-1.0) * math.motion_cross_force(u, g1)
      k_out[w, d + 2] = (-1.0) * math.motion_cross_force(u, g2)
      u = (
        u
        + cdof_in[w, d + 0] * qvel_in[w, d + 0]
        + cdof_in[w, d + 1] * qvel_in[w, d + 1]
        + cdof_in[w, d + 2] * qvel_in[w, d + 2]
      )
      Hb = Hb + h0 + h1 + h2
    else:  # HINGE / SLIDE: snapshot from the pre-joint cvel_in, then add
      g0 = adj_cdof_dot[w, d]
      h0 = math.motion_cross_force(cdof_in[w, d], g0)
      h_out[w, d] = h0
      k_out[w, d] = (-1.0) * math.motion_cross_force(u, g0)
      u = u + cdof_in[w, d] * qvel_in[w, d]
      Hb = Hb + h0
  H_out[w, b] = Hb


# CV2: same-body reverse suffix scan; scatter precedes T+=h so no axis sees its own snapshot
@wp.kernel(enable_backward=False)
def _comvel_vjp_samebody(
  # Model:
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  h_in: wp.array2d[wp.spatial_vector],
  k_in: wp.array2d[wp.spatial_vector],
  # Out:
  adj_qvel_out: wp.array2d[float],  # (+=): same-body T_j part
  adj_cdof_out: wp.array2d[wp.spatial_vector],  # (+=): same-body T_j part + direct k_j
):
  w, b = wp.tid()
  t = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  jntadr = body_jntadr[b]
  jntnum = body_jntnum[b]
  for jr in range(jntnum):
    jj = jntnum - 1 - jr  # reverse joint order
    jt = jnt_type[jntadr + jj]
    d = jnt_dofadr[jntadr + jj]
    if jt == _FREE:  # reverse: scatter rot adds, T += rot h, k rot, scatter trans adds (now see rot h)
      _cv_scatter(qvel_in, cdof_in, w, d + 5, t, adj_qvel_out, adj_cdof_out)
      _cv_scatter(qvel_in, cdof_in, w, d + 4, t, adj_qvel_out, adj_cdof_out)
      _cv_scatter(qvel_in, cdof_in, w, d + 3, t, adj_qvel_out, adj_cdof_out)
      t = t + h_in[w, d + 5] + h_in[w, d + 4] + h_in[w, d + 3]
      adj_cdof_out[w, d + 5] = adj_cdof_out[w, d + 5] + k_in[w, d + 5]
      adj_cdof_out[w, d + 4] = adj_cdof_out[w, d + 4] + k_in[w, d + 4]
      adj_cdof_out[w, d + 3] = adj_cdof_out[w, d + 3] + k_in[w, d + 3]
      _cv_scatter(qvel_in, cdof_in, w, d + 2, t, adj_qvel_out, adj_cdof_out)
      _cv_scatter(qvel_in, cdof_in, w, d + 1, t, adj_qvel_out, adj_cdof_out)
      _cv_scatter(qvel_in, cdof_in, w, d + 0, t, adj_qvel_out, adj_cdof_out)
    elif jt == _BALL:  # scatter all 3 with the same T, then T += the 3 same-ball h
      _cv_scatter(qvel_in, cdof_in, w, d + 2, t, adj_qvel_out, adj_cdof_out)
      _cv_scatter(qvel_in, cdof_in, w, d + 1, t, adj_qvel_out, adj_cdof_out)
      _cv_scatter(qvel_in, cdof_in, w, d + 0, t, adj_qvel_out, adj_cdof_out)
      t = t + h_in[w, d + 2] + h_in[w, d + 1] + h_in[w, d + 0]
      adj_cdof_out[w, d + 0] = adj_cdof_out[w, d + 0] + k_in[w, d + 0]
      adj_cdof_out[w, d + 1] = adj_cdof_out[w, d + 1] + k_in[w, d + 1]
      adj_cdof_out[w, d + 2] = adj_cdof_out[w, d + 2] + k_in[w, d + 2]
    else:  # HINGE / SLIDE
      _cv_scatter(qvel_in, cdof_in, w, d, t, adj_qvel_out, adj_cdof_out)
      t = t + h_in[w, d]
      adj_cdof_out[w, d] = adj_cdof_out[w, d] + k_in[w, d]


# CV3: W_B = sum_{subtree(B)} adj_cvel + sum_{strict_subtree(B)} H; oracle of _comvel_W_acc
@wp.kernel(enable_backward=False)
def _comvel_subtree_W(
  # Model:
  nbody: int,
  body_parentid: wp.array[int],
  # In:
  adj_cvel: wp.array2d[wp.spatial_vector],  # A seed (per body); not mutated
  H_in: wp.array2d[wp.spatial_vector],
  # Out:
  W_out: wp.array2d[wp.spatial_vector],
):
  w, target = wp.tid()
  acc = adj_cvel[w, target]  # B itself (subtree includes self for the A term)
  for x in range(1, nbody):
    if x != target:
      p = body_parentid[x]  # is target a strict ancestor of x?
      while p > 0:
        if p == target:
          acc = acc + adj_cvel[w, x] + H_in[w, x]
          break
        p = body_parentid[p]
  W_out[w, target] = acc


# CV4: global cvel scatter += W_body(i): nubar_i += S_i*W, Sbar_i += nu_i*W
@wp.kernel(enable_backward=False)
def _comvel_scatter_W(
  # Model:
  dof_bodyid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  W_in: wp.array2d[wp.spatial_vector],
  # Out:
  adj_qvel_out: wp.array2d[float],  # (+=): W_B part
  adj_cdof_out: wp.array2d[wp.spatial_vector],  # (+=): W_B part
):
  w, i = wp.tid()
  wv = W_in[w, dof_bodyid[i]]
  adj_qvel_out[w, i] = adj_qvel_out[w, i] + wp.dot(cdof_in[w, i], wv)
  adj_cdof_out[w, i] = adj_cdof_out[w, i] + qvel_in[w, i] * wv


def comvel_backward(m: Model, d: Data, adj_cvel: wp.array2d, adj_cdof_dot: wp.array2d):
  """Computes the bound-free manual VJP of smooth.com_vel.

  Maps adj_cvel / adj_cdof_dot to {qvel, cdof} adjoints.
  """
  nworld = d.qvel.shape[0]
  nv = m.nv
  nbody = m.nbody
  SV = wp.spatial_vector
  h = wp.zeros((nworld, nv), dtype=SV)
  kk = wp.zeros((nworld, nv), dtype=SV)
  Hbody = wp.zeros((nworld, nbody), dtype=SV)
  W = wp.zeros((nworld, nbody), dtype=SV)
  adj_qvel = wp.zeros((nworld, nv), dtype=float)
  adj_cdof = wp.zeros((nworld, nv), dtype=SV)
  # CV1: local snapshot + motion_cross VJP -> h, k per dof; H per body.
  wp.launch(
    _comvel_vjp_local,
    dim=(nworld, nbody),
    inputs=[m.body_parentid, m.body_jntnum, m.body_jntadr, m.jnt_type, m.jnt_dofadr, d.qvel, d.cdof, d.cvel, adj_cdof_dot],
    outputs=[h, kk, Hbody],
  )
  # CV2: same-body prefix reverse (event scan) -> adj_qvel/adj_cdof (the T_j part + direct k_j).
  wp.launch(
    _comvel_vjp_samebody,
    dim=(nworld, nbody),
    inputs=[m.body_jntnum, m.body_jntadr, m.jnt_type, m.jnt_dofadr, d.qvel, d.cdof, h, kk],
    outputs=[adj_qvel, adj_cdof],
  )
  # CV3: W_B = sum_subtree(B) adj_cvel + sum_strict_subtree(B) H  (the augmented-seed subtree sum).
  wp.launch(_comvel_subtree_W, dim=(nworld, nbody), inputs=[nbody, m.body_parentid, adj_cvel, Hbody], outputs=[W])
  # CV4: global cvel scatter -> += W_B contributions.
  wp.launch(_comvel_scatter_W, dim=(nworld, nv), inputs=[m.dof_bodyid, d.qvel, d.cdof, W], outputs=[adj_qvel, adj_cdof])
  return {"qvel": adj_qvel, "cdof": adj_cdof}


# --------------------------------------------------------------------------------------------
# cinert reverse: the last kinematic dqpos leaf. cinert depends on qpos via ximat / xipos /
# subtree_com[root]: source-AD the body-local _cinert leaf, chain xipos/ximat -> qpos via
# support.jac_dof, route adj_subtree_com through _subtree_com_qpos_vjp. cvel/cdof_dot have no
# direct qpos path (it flows through the total adj_cdof via _cdof_qpos_vjp).
# --------------------------------------------------------------------------------------------


# exact reconstruction of smooth._cinert (body-local, no loop) for a manual adjoint launch
@wp.kernel
def _cinert_recompute(
  # Model:
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  # Data out:
  cinert_out: wp.array2d[vec10],
):
  w, b = wp.tid()
  mat = ximat_in[w, b]
  inert = body_inertia[w % body_inertia.shape[0], b]
  mass = body_mass[w % body_mass.shape[0], b]
  dif = xipos_in[w, b] - subtree_com_in[w, body_rootid[b]]
  # build the vec10 via a single vec10f(...) ctor: warp 1.14's reverse double-counts in-place
  # component writes (adjoint exactly 2x on the rotation block); forward smooth._cinert keeps
  # the component-write form (enable_backward=False, value-only).
  i0 = inert[0]
  i1 = inert[1]
  i2 = inert[2]
  d0 = dif[0]
  d1 = dif[1]
  d2 = dif[2]
  r0 = i0 * mat[0, 0] * mat[0, 0] + i1 * mat[0, 1] * mat[0, 1] + i2 * mat[0, 2] * mat[0, 2] + mass * (d1 * d1 + d2 * d2)
  r1 = i0 * mat[1, 0] * mat[1, 0] + i1 * mat[1, 1] * mat[1, 1] + i2 * mat[1, 2] * mat[1, 2] + mass * (d0 * d0 + d2 * d2)
  r2 = i0 * mat[2, 0] * mat[2, 0] + i1 * mat[2, 1] * mat[2, 1] + i2 * mat[2, 2] * mat[2, 2] + mass * (d0 * d0 + d1 * d1)
  r3 = i0 * mat[0, 0] * mat[1, 0] + i1 * mat[0, 1] * mat[1, 1] + i2 * mat[0, 2] * mat[1, 2] - mass * d0 * d1
  r4 = i0 * mat[0, 0] * mat[2, 0] + i1 * mat[0, 1] * mat[2, 1] + i2 * mat[0, 2] * mat[2, 2] - mass * d0 * d2
  r5 = i0 * mat[1, 0] * mat[2, 0] + i1 * mat[1, 1] * mat[2, 1] + i2 * mat[1, 2] * mat[2, 2] - mass * d1 * d2
  cinert_out[w, b] = vec10f(r0, r1, r2, r3, r4, r5, mass * d0, mass * d1, mass * d2, mass)


# chain adj_{xipos_in, ximat_in} to a per-dof tangent gradient via support.jac_dof
@wp.kernel(enable_backward=False)
def _cinert_pose_dof_vjp(
  # Model:
  nbody: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_xipos: wp.array2d[wp.vec3],
  adj_ximat: wp.array2d[wp.mat33],
  # Out:
  res_dof_out: wp.array2d[float],  # (+=): per-dof tangent gradient
):
  w, k = wp.tid()
  acc = float(0.0)
  for b in range(1, nbody):
    if body_isdofancestor[b, k] == 0:  # dof k does not move body b -> jac = 0
      continue
    jacp, jacr = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      xipos_in[w, b],
      b,
      k,
      w,
    )
    xm = ximat_in[w, b]
    rgm = adj_ximat[w, b]
    tau = (
      wp.cross(wp.vec3(xm[0, 0], xm[1, 0], xm[2, 0]), wp.vec3(rgm[0, 0], rgm[1, 0], rgm[2, 0]))
      + wp.cross(wp.vec3(xm[0, 1], xm[1, 1], xm[2, 1]), wp.vec3(rgm[0, 1], rgm[1, 1], rgm[2, 1]))
      + wp.cross(wp.vec3(xm[0, 2], xm[1, 2], xm[2, 2]), wp.vec3(rgm[0, 2], rgm[1, 2], rgm[2, 2]))
    )
    acc += wp.dot(jacp, adj_xipos[w, b]) + wp.dot(jacr, tau)
  res_dof_out[w, k] += acc


@event_scope
def cinert_qpos_vjp(m: Model, d: Data, adj_cinert: wp.array2d, res_dof: wp.array2d):
  """Computes the VJP of smooth._cinert.

  Pose terms += into res_dof; returns the subtree_com cotangent seed.
  """
  nworld = d.qpos.shape[0]
  nv = m.nv
  nbody = m.nbody
  cinert_rec = wp.zeros((nworld, nbody), dtype=vec10f)
  adj_xipos = wp.zeros((nworld, nbody), dtype=wp.vec3)
  adj_ximat = wp.zeros((nworld, nbody), dtype=wp.mat33)
  adj_subtree = wp.zeros((nworld, nbody), dtype=wp.vec3)
  cin = [m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com]
  wp.launch(_cinert_recompute, dim=(nworld, nbody), inputs=cin, outputs=[cinert_rec])
  # throwaway mass/inertia adjoint buffers: with adj_input=None Warp would auto-route their
  # adjoint into m.body_{mass,inertia}.grad, clobbering the sys-id gradient inertia_param_vjp
  # accumulated this step. Capture and discard so only the qpos (xipos/ximat) channel is touched.
  junk_mass = wp.zeros_like(m.body_mass)
  junk_inertia = wp.zeros_like(m.body_inertia)
  wp.launch(
    _cinert_recompute,
    dim=(nworld, nbody),
    inputs=cin,
    outputs=[cinert_rec],
    adj_inputs=[None, junk_mass, junk_inertia, adj_xipos, adj_ximat, adj_subtree],
    adj_outputs=[adj_cinert],
    adjoint=True,
  )
  wp.launch(
    _cinert_pose_dof_vjp,
    dim=(nworld, nv),
    inputs=[
      nbody,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.ximat,
      d.subtree_com,
      d.cdof,
      adj_xipos,
      adj_ximat,
    ],
    outputs=[res_dof],
  )
  return adj_subtree


# exact reconstruction of smooth._compute_body_inertial_frames for a manual adjoint launch
@wp.kernel
def _inertial_frames_recompute(
  # Model:
  body_ipos: wp.array2d[wp.vec3],
  body_iquat: wp.array2d[wp.quat],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  # Data out:
  xipos_out: wp.array2d[wp.vec3],
  ximat_out: wp.array2d[wp.mat33],
):
  w, b = wp.tid()
  xp = xpos_in[w, b]
  xq = xquat_in[w, b]
  xipos_out[w, b] = xp + math.rot_vec_quat(body_ipos[w % body_ipos.shape[0], b], xq)
  ximat_out[w, b] = math.quat_to_mat(math.mul_quat(xq, body_iquat[w % body_iquat.shape[0], b]))


@event_scope
def inertia_param_vjp(m: Model, d: Data, lam: wp.array2d):
  """Accumulates the inertial sys-id VJP -(d(lam^T qfrc_bias)/dtheta) into m.<param>.grad.

  Caveat: subtree_com is frozen, so body_ipos captures only the local parallel-axis path;
  mass/inertia/iquat are exact. No-op unless a param is requires_grad.
  """
  nworld = d.qpos.shape[0]
  nbody = m.nbody
  want_mass = m.body_mass.requires_grad
  want_inertia = m.body_inertia.requires_grad
  want_ipos = m.body_ipos.requires_grad
  want_iquat = m.body_iquat.requires_grad
  want_pose = want_ipos or want_iquat
  # adj_cinert = d(lam^Tqfrc_bias)/dcinert (captures M*qacc + Coriolis + gravity, qacc frozen).
  adj = rne_backward(m, d, lam, flg_acc=True)
  adj_cinert = adj["cinert"]
  # source-AD the cinert leaf: params (mass/inertia) as direct inputs; xipos/ximat kept for the
  # pose chain.
  cinert_rec = wp.zeros((nworld, nbody), dtype=vec10f, requires_grad=True)
  cin = [m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com]
  wp.launch(_cinert_recompute, dim=(nworld, nbody), inputs=cin, outputs=[cinert_rec])
  adj_mass = wp.zeros_like(m.body_mass) if want_mass else None
  adj_inertia = wp.zeros_like(m.body_inertia) if want_inertia else None
  adj_xipos = wp.zeros((nworld, nbody), dtype=wp.vec3) if want_pose else None
  adj_ximat = wp.zeros((nworld, nbody), dtype=wp.mat33) if want_pose else None
  wp.launch(
    _cinert_recompute,
    dim=(nworld, nbody),
    inputs=cin,
    outputs=[cinert_rec],
    adj_inputs=[None, adj_mass, adj_inertia, adj_xipos, adj_ximat, None],
    adj_outputs=[adj_cinert],
    adjoint=True,
  )
  # body_ipos/body_iquat: continue the reverse from adj_{xipos,ximat} through the inertial-frame
  # leaf.
  if want_pose:
    fout_xipos = wp.zeros((nworld, nbody), dtype=wp.vec3)
    fout_ximat = wp.zeros((nworld, nbody), dtype=wp.mat33)
    fin = [m.body_ipos, m.body_iquat, d.xpos, d.xquat]
    wp.launch(_inertial_frames_recompute, dim=(nworld, nbody), inputs=fin, outputs=[fout_xipos, fout_ximat])
    adj_ipos = wp.zeros_like(m.body_ipos) if want_ipos else None
    adj_iquat = wp.zeros_like(m.body_iquat) if want_iquat else None
    wp.launch(
      _inertial_frames_recompute,
      dim=(nworld, nbody),
      inputs=fin,
      outputs=[fout_xipos, fout_ximat],
      adj_inputs=[adj_ipos, adj_iquat, None, None],
      adj_outputs=[adj_xipos, adj_ximat],
      adjoint=True,
    )
    if adj_ipos is not None:
      wp.launch(adjoint_util._accum_neg_vec3, dim=adj_ipos.shape, inputs=[adj_ipos], outputs=[m.body_ipos.grad])
    if adj_iquat is not None:
      wp.launch(adjoint_util._accum_neg_quat, dim=adj_iquat.shape, inputs=[adj_iquat], outputs=[m.body_iquat.grad])
  # IFT minus into the shared (trajectory-accumulated) model param grads.
  if adj_mass is not None:
    wp.launch(adjoint_util._accum_neg, dim=adj_mass.shape, inputs=[adj_mass], outputs=[m.body_mass.grad])
  if adj_inertia is not None:
    wp.launch(adjoint_util._accum_neg_vec3, dim=adj_inertia.shape, inputs=[adj_inertia], outputs=[m.body_inertia.grad])


@wp.kernel(enable_backward=False)
def _add_spatial(a: wp.array2d[wp.spatial_vector], b: wp.array2d[wp.spatial_vector], out: wp.array2d[wp.spatial_vector]):
  w, i = wp.tid()
  out[w, i] = a[w, i] + b[w, i]


def rne_qpos_vjp(m: Model, d: Data, lam: wp.array2d, flg_acc: bool = True):
  """Computes the full RNE-bias dqpos: res_qpos = d(lam^T qfrc_bias)/dqpos.

  Reads converged d only.
  """
  nworld = d.qpos.shape[0]
  nv = m.nv
  nq = d.qpos.shape[1]
  nbody = m.nbody
  adj = rne_backward(m, d, lam, flg_acc)
  cv = comvel_backward(m, d, adj["cvel"], adj["cdof_dot"])
  total_cdof = wp.zeros((nworld, nv), dtype=wp.spatial_vector)
  wp.launch(_add_spatial, dim=(nworld, nv), inputs=[adj["cdof"], cv["cdof"]], outputs=[total_cdof])

  res_dof = wp.zeros((nworld, nv), dtype=float)
  adj_subtree = cinert_qpos_vjp(m, d, adj["cinert"], res_dof)  # xipos/ximat -> res_dof; subtree seed out
  wp.launch(
    collision_adjoint._cdof_qpos_vjp,
    dim=(nworld, nv),
    inputs=[nv, m.jnt_type, m.jnt_dofadr, m.dof_jntid, m.dof_parentid, d.cdof, total_cdof],
    outputs=[res_dof],
  )
  ceff = wp.empty((nworld, nbody), dtype=wp.vec3)  # cinert subtree seed + cdof moving-COM fold
  wp.launch(
    collision_adjoint._build_ceff,
    dim=(nworld, nbody),
    inputs=[nv, m.body_rootid, m.dof_bodyid, d.cdof, total_cdof, adj_subtree],
    outputs=[ceff],
  )
  wp.launch(
    collision_adjoint._subtree_com_qpos_vjp,
    dim=(nworld, nv),
    inputs=[
      nbody,
      m.body_parentid,
      m.body_rootid,
      m.body_mass,
      m.body_subtreemass,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.subtree_com,
      d.cdof,
      ceff,
    ],
    outputs=[res_dof],
  )
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  wp.launch(
    collision_adjoint._dof_to_qpos,
    dim=(nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof],
    outputs=[res_qpos],
  )
  return res_qpos


# affine joint-actuator local-dqpos leaf (launched by actuator_qpos_vjp above): manual VJP,
# res_dof_out[j] += -ml_a*dfdl_a*moment[a,j]; saturated forcerange zeroes dfdl (mirrors forward);
# velocity coeffs are owned by the velocity stage
@wp.kernel(enable_backward=False)
def _actuator_qpos_vjp(
  # Model:
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_ctrllimited: wp.array[bool],
  actuator_forcelimited: wp.array[bool],
  actuator_gainprm: wp.array2d[vec10f],
  actuator_biasprm: wp.array2d[vec10f],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  actuator_forcerange: wp.array2d[wp.vec2],
  # Data in:
  ctrl_in: wp.array2d[float],
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  dsbl_clampctrl: int,
  # Out:
  res_dof_out: wp.array2d[float],  # (+=): per-dof tangent gradient; lifted by _dof_to_qpos
):
  w, actid = wp.tid()
  dfdl = float(0.0)
  if actuator_gaintype[actid] == _GAIN_AFFINE:
    ctrl = ctrl_in[w % ctrl_in.shape[0], actid]
    if actuator_ctrllimited[actid] and dsbl_clampctrl == 0:
      cr = actuator_ctrlrange[w % actuator_ctrlrange.shape[0], actid]
      ctrl = wp.clamp(ctrl, cr[0], cr[1])
    dfdl += actuator_gainprm[w % actuator_gainprm.shape[0], actid][1] * ctrl
  if actuator_biastype[actid] == _BIAS_AFFINE:
    dfdl += actuator_biasprm[w % actuator_biasprm.shape[0], actid][1]
  if dfdl == 0.0:
    return
  if actuator_forcelimited[actid]:  # saturated force -> the affine slope is clamped off (matches forward)
    fr = actuator_forcerange[w % actuator_forcerange.shape[0], actid]
    f = actuator_force_in[w, actid]
    if f <= fr[0] or f >= fr[1]:
      return
  rownnz = moment_rownnz_in[w, actid]
  rowadr = moment_rowadr_in[w, actid]
  ml = float(0.0)
  for i in range(rownnz):
    sid = rowadr + i
    ml += actuator_moment_in[w, sid] * lam[w, moment_colind_in[w, sid]]
  c = -ml * dfdl  # minus: r_smooth includes -qfrc_actuator
  for i in range(rownnz):
    sid = rowadr + i
    wp.atomic_add(res_dof_out[w], moment_colind_in[w, sid], c * actuator_moment_in[w, sid])
