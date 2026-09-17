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
"""VJPs for smooth forces, RNE dynamics, and mass-matrix position dependence."""

import dataclasses

import numpy as np
import warp as wp

from mujoco_warp._src import adjoint_recompute
from mujoco_warp._src import adjoint_util
from mujoco_warp._src import math
from mujoco_warp._src import smooth
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import support
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import vec10
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": True})


def prepare_backward(d: Data):
  """Allocates gradients for smooth intermediates reused by the analytic reverse."""
  arrays = (
    d.xipos,
    d.ximat,
    d.subtree_com,
    d.cinert,
    d.cvel,
    d.cacc,
    d.cdof,
    d.cdof_dot,
    d.qfrc_spring,
    d.qfrc_gravcomp,
  )
  for array in arrays:
    array.requires_grad = True
  return arrays


# Results hold a strong Model reference so a recycled id cannot select stale metadata.
_SUPPORTED_CACHE = {}


def assert_smooth_supported(m: Model):
  """Raises for enabled features without an analytic smooth VJP."""
  key = id(m)
  if key in _SUPPORTED_CACHE:
    return

  bad = []
  if m.ntendon:
    bad.append("tendons (spring/bias/armature/transmission)")
  opt = m.opt
  if np.any(opt.density.numpy() != 0.0) or np.any(opt.viscosity.numpy() != 0.0) or np.any(opt.wind.numpy() != 0.0):
    bad.append("fluid forces (opt.density/viscosity/wind)")
  if m.flg_adhesion:
    bad.append("passive adhesion (m.flg_adhesion: qpos-dependent qfrc_adhesion has no VJP leaf)")
  if np.any(m.body_gravcomp.numpy() != 0.0) and np.any(m.jnt_actgravcomp.numpy() != 0):
    bad.append("gravcomp routed to an actuator (jnt_actgravcomp; force-limit clamp); passive-bucket gravcomp is supported")
  if m.nu:
    trntype = m.actuator_trntype.numpy()
    gaintype = m.actuator_gaintype.numpy()
    biastype = m.actuator_biastype.numpy()
    jnt_type = m.jnt_type.numpy()
    if np.any(trntype != TrnType.JOINT):
      bad.append("non-JOINT actuator transmission (tendon/site/body/slider-crank/jointinparent)")
    if (
      not ((gaintype == GainType.FIXED) | (gaintype == GainType.AFFINE)).all()
      or not ((biastype == BiasType.NONE) | (biastype == BiasType.AFFINE)).all()
    ):
      bad.append("non-affine actuator gain/bias (muscle/DC-motor/user)")
    jnt_ids = m.actuator_trnid.numpy()[:, 0][trntype == TrnType.JOINT]
    if jnt_ids.size and np.any((jnt_type[jnt_ids] == JointType.FREE) | (jnt_type[jnt_ids] == JointType.BALL)):
      bad.append("actuator on a FREE/BALL joint (quaternion-dependent transmission length)")
    if np.any((m.actuator_dyntype.numpy() != DynType.NONE) & (gaintype == GainType.AFFINE)):
      bad.append("stateful (na>0) AFFINE-gain actuator (ctrl_act=act not implemented)")
  if bad:
    raise NotImplementedError(
      "smooth_adjoint analytic reverse does not support: "
      + "; ".join(bad)
      + ". Implement the leaf/topology VJP (the FD oracle lives in adjoint_test_util). "
      "There is no silent FD fallback."
    )
  _SUPPORTED_CACHE[key] = (m, True)


@event_scope
def spring_qpos_vjp(m: Model, d: Data, lam: wp.array2d, bc: BackwardContext | None = None):
  """Returns d(lam.T @ -qfrc_spring)/dqpos."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  smooth_res_qpos = bc.smooth_res_qpos
  qfrc_spring = d.qfrc_spring
  smooth_res_qpos.zero_()
  qfrc_spring.grad.zero_()
  wp.launch(adjoint_util._neg_cols, dim=(d.nworld, m.nv), inputs=[lam], outputs=[qfrc_spring.grad])
  wp.launch(
    adjoint_recompute._spring_qfrc_recompute,
    dim=(d.nworld, m.njnt),
    inputs=[
      m.opt.disableflags,
      m.qpos_spring,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      d.qpos,
    ],
    outputs=[qfrc_spring],
    adj_inputs=[None, None, None, None, None, None, None, smooth_res_qpos],
    adj_outputs=[qfrc_spring.grad],
    adjoint=True,
  )
  return smooth_res_qpos


@wp.func
def _actuator_moment_dot(
  # Data in:
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  worldid: int,
  rowadr: int,
  rownnz: int,
) -> float:
  value = float(0.0)
  for i in range(rownnz):
    elemid = rowadr + i
    value += actuator_moment_in[worldid, elemid] * lam[worldid, moment_colind_in[worldid, elemid]]
  return value


@wp.kernel(enable_backward=False)
def _actuator_qpos_vjp(
  # Model:
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
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
  res_dof_out: wp.array2d[float],
):
  w, actid = wp.tid()
  dfdl = float(0.0)
  if actuator_gaintype[actid] == GainType.AFFINE:
    ctrl = ctrl_in[w % ctrl_in.shape[0], actid]
    if actuator_ctrllimited[actid] and dsbl_clampctrl == 0:
      ctrlrange = actuator_ctrlrange[w % actuator_ctrlrange.shape[0], actid]
      ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])
    dfdl += actuator_gainprm[w % actuator_gainprm.shape[0], actid][1] * ctrl
  if actuator_biastype[actid] == BiasType.AFFINE:
    dfdl += actuator_biasprm[w % actuator_biasprm.shape[0], actid][1]
  if dfdl == 0.0:
    return

  if actuator_forcelimited[actid]:
    forcerange = actuator_forcerange[w % actuator_forcerange.shape[0], actid]
    force = actuator_force_in[w, actid]
    if force <= forcerange[0] or force >= forcerange[1]:
      return

  rownnz = moment_rownnz_in[w, actid]
  rowadr = moment_rowadr_in[w, actid]
  moment_lam = _actuator_moment_dot(moment_colind_in, actuator_moment_in, lam, w, rowadr, rownnz)
  scale = -moment_lam * dfdl
  for i in range(rownnz):
    sparseid = rowadr + i
    wp.atomic_add(res_dof_out[w], moment_colind_in[w, sparseid], scale * actuator_moment_in[w, sparseid])


@event_scope
def actuator_qpos_vjp(m: Model, d: Data, lam: wp.array2d, bc: BackwardContext | None = None):
  """Returns the affine joint-actuator position VJP."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  smooth_res_qpos = bc.smooth_res_qpos
  smooth_res_qpos.zero_()
  if not m.nu or int(m.opt.disableflags) & DisableBit.ACTUATION:
    return smooth_res_qpos

  smooth_res_dof = bc.smooth_res_dof
  smooth_res_dof.zero_()
  wp.launch(
    _actuator_qpos_vjp,
    dim=(d.nworld, m.nu),
    inputs=[
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_forcelimited,
      m.actuator_forcerange,
      m.actuator_ctrllimited,
      m.actuator_ctrlrange,
      d.ctrl,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.actuator_force,
      lam,
      int(m.opt.disableflags) & DisableBit.CLAMPCTRL,
    ],
    outputs=[smooth_res_dof],
  )
  wp.launch(
    smooth_kinematics_adjoint._dof_to_qpos,
    dim=(d.nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, smooth_res_dof],
    outputs=[smooth_res_qpos],
  )
  return smooth_res_qpos


@wp.kernel(enable_backward=False)
def _gravcomp_seed(
  # Model:
  jnt_actgravcomp: wp.array[int],
  dof_jntid: wp.array[int],
  # In:
  lam: wp.array2d[float],
  gravity_enabled: int,
  # Out:
  grad_out: wp.array2d[float],
):
  w, i = wp.tid()
  if gravity_enabled and not jnt_actgravcomp[dof_jntid[i]]:
    grad_out[w, i] = -lam[w, i]
  else:
    grad_out[w, i] = 0.0


@event_scope
def gravcomp_qpos_vjp(m: Model, d: Data, lam: wp.array2d, bc: BackwardContext | None = None):
  """Returns d(lam.T @ -qfrc_gravcomp)/dqpos."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  d.xipos.grad.zero_()
  d.ximat.grad.zero_()
  d.subtree_com.grad.zero_()
  d.cdof.grad.zero_()
  d.qfrc_gravcomp.grad.zero_()
  bc.res_body_mass.zero_()
  bc.smooth_res_qpos.zero_()

  gravity_enabled = int(not (int(m.opt.disableflags) & DisableBit.GRAVITY))
  wp.launch(
    _gravcomp_seed,
    dim=(d.nworld, m.nv),
    inputs=[m.jnt_actgravcomp, m.dof_jntid, lam, gravity_enabled],
    outputs=[d.qfrc_gravcomp.grad],
  )
  wp.launch(
    adjoint_recompute._gravity_force_recompute,
    dim=(d.nworld, m.nbody - 1, m.nv),
    inputs=[
      m.opt.gravity,
      m.body_parentid,
      m.body_rootid,
      m.body_mass,
      m.body_gravcomp,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.subtree_com,
      d.cdof,
    ],
    outputs=[d.qfrc_gravcomp],
    adj_inputs=[None, None, None, bc.res_body_mass, None, None, None, d.xipos.grad, d.subtree_com.grad, d.cdof.grad],
    adj_outputs=[d.qfrc_gravcomp.grad],
    adjoint=True,
  )
  return _kinematic_qpos_vjp(m, d, d.cdof.grad, bc)


@event_scope
def gravcomp_mass_vjp(m: Model, d: Data, lam: wp.array2d, res_body_mass: wp.array2d):
  """Accumulates d(lam.T @ -qfrc_gravcomp)/d(body_mass)."""
  d.qfrc_gravcomp.grad.zero_()
  gravity_enabled = int(not (int(m.opt.disableflags) & DisableBit.GRAVITY))
  wp.launch(
    _gravcomp_seed,
    dim=(d.nworld, m.nv),
    inputs=[m.jnt_actgravcomp, m.dof_jntid, lam, gravity_enabled],
    outputs=[d.qfrc_gravcomp.grad],
  )
  wp.launch(
    adjoint_recompute._gravity_force_recompute,
    dim=(d.nworld, m.nbody - 1, m.nv),
    inputs=[
      m.opt.gravity,
      m.body_parentid,
      m.body_rootid,
      m.body_mass,
      m.body_gravcomp,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.subtree_com,
      d.cdof,
    ],
    outputs=[d.qfrc_gravcomp],
    adj_inputs=[None, None, None, res_body_mass, None, None, None, None, None, None],
    adj_outputs=[d.qfrc_gravcomp.grad],
    adjoint=True,
  )


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


@wp.kernel(enable_backward=False)
def _acc_spatial(src: wp.array2d[wp.spatial_vector], io_out: wp.array2d[wp.spatial_vector]):
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


def linearize(m: Model, d: Data, d_out: Data, bc: BackwardContext) -> Data:
  """Replays smooth dynamics at the step input state."""
  view = dataclasses.replace(
    d_out,
    qpos=d.qpos,
    qvel=d.qvel,
    ctrl=d.ctrl,
    act=d.act,
    cacc=bc.scratch.cacc,
    cfrc_int=bc.scratch.cfrc_int,
    qfrc_bias=bc.scratch.qfrc_bias,
  )
  smooth.rne(m, view, flg_acc=True)
  return view


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
  bc: BackwardContext | None = None,
):
  """Returns d(lam.T @ qfrc_bias)/dqpos at the converged state."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  nworld, nv = d.nworld, m.nv
  rne_backward(m, d, lam, flg_acc, bc=bc)
  comvel_backward(m, d, d.cvel.grad, d.cdof_dot.grad, bc=bc)
  if res_cdof_extra is not None:
    wp.launch(_acc_spatial, dim=(nworld, nv), inputs=[res_cdof_extra], outputs=[d.cdof.grad])

  d.xipos.grad.zero_()
  d.ximat.grad.zero_()
  d.subtree_com.grad.zero_()
  _cinert_vjp(m, d, d.cinert.grad, bc)
  if res_subtree_extra is not None:
    wp.launch(_acc_vec3, dim=(nworld, m.nbody), inputs=[res_subtree_extra], outputs=[d.subtree_com.grad])
  return _kinematic_qpos_vjp(m, d, d.cdof.grad, bc)


@event_scope
def mass_matrix_qpos_vjp(m: Model, d: Data, y: wp.array2d, w: wp.array2d, bc: BackwardContext | None = None):
  """Returns d(y.T @ M(q) @ w)/dqpos with y and w fixed."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  scratch = bc.scratch
  wp.copy(scratch.qpos, d.qpos)
  if m.nmocap:
    wp.copy(scratch.mocap_pos, d.mocap_pos)
    wp.copy(scratch.mocap_quat, d.mocap_quat)
  scratch.qvel.zero_()
  scratch.qacc = w
  smooth.kinematics(m, scratch)
  smooth.com_pos(m, scratch)
  scratch.cvel.zero_()
  scratch.cdof_dot.zero_()
  scratch.cacc.zero_()
  smooth._rne_cacc_forward(m, scratch, flg_acc=True)
  smooth._rne_cfrc(m, scratch)
  smooth._rne_cfrc_backward(m, scratch)
  return smooth_force_backward(m, scratch, y, bc=bc)


# ============================================================================================
# AD-RNE: analytic backward of smooth.rne, the production dqpos path in step_backward.
# FD-of-rne is an explicitly selected validation oracle in adjoint_test_util. smooth.rne is
# enable_backward=False, so the reverse is reconstructed here: source-AD the loop-free/
# alias-free nonlinear _cfrc leaf (exact), manual transposed VJPs for the two linear tree
# reductions (their forwards have dynamic loops + in-place aliasing, so a naive reverse is wrong).
# ============================================================================================


# d(lam.T @ qfrc_bias)/dF_b = sum(lam_i * cdof_i) for the body's DOFs.
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
  acc = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for k in range(dofnum):
    i = dofadr + k
    acc = acc + lam[w, i] * cdof_in[w, i]
  adj_force_out[w, b] = acc


@wp.kernel(enable_backward=False)
def _rne_dof_vjp(
  # Model:
  dof_bodyid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  cfrc_int_in: wp.array2d[wp.spatial_vector],
  # In:
  lam: wp.array2d[float],
  subtree_adj_cacc: wp.array2d[wp.spatial_vector],  # sum adj_cacc over body(j)'s subtree (from K4a)
  flg_acc: bool,
  # Out:
  adj_qvel_out: wp.array2d[float],
  adj_qacc_out: wp.array2d[float],
  adj_cdof_dot_out: wp.array2d[wp.spatial_vector],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
):
  w, i = wp.tid()
  bodyid = dof_bodyid[i]
  a = subtree_adj_cacc[w, bodyid]
  adj_qvel_out[w, i] = wp.dot(a, cdof_dot_in[w, i])
  adj_cdof_dot_out[w, i] = qvel_in[w, i] * a
  adj_cdof = lam[w, i] * cfrc_int_in[w, bodyid]
  if flg_acc:
    adj_qacc_out[w, i] = wp.dot(a, cdof_in[w, i])
    adj_cdof = adj_cdof + qacc_in[w, i] * a
  else:
    adj_qacc_out[w, i] = 0.0
  adj_cdof_out[w, i] = adj_cdof


def rne_backward(
  m: Model,
  d: Data,
  lam: wp.array2d,
  flg_acc: bool = True,
  bc: BackwardContext | None = None,
):
  """Reverses RNE and returns views of its input cotangents."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  nworld, nv, nbody = d.nworld, m.nv, m.nbody
  d.cinert.grad.zero_()
  d.cvel.grad.zero_()
  d.cacc.grad.zero_()
  wp.launch(
    _rne_qfrcbias_force_vjp,
    dim=(nworld, nbody),
    inputs=[m.body_dofnum, m.body_dofadr, d.cdof, lam],
    outputs=[bc.smooth_force],
  )

  wp.copy(bc.smooth_tree, bc.smooth_force)
  for level in m.body_tree:
    wp.launch(_anc_acc_sv, dim=(nworld, level.size), inputs=[m.body_parentid, level], outputs=[bc.smooth_tree])

  wp.launch(
    adjoint_recompute._rne_cfrc_recompute,
    dim=(nworld, nbody),
    inputs=[d.cinert, d.cvel, d.cacc],
    outputs=[bc.smooth_cfrc],
    adj_inputs=[d.cinert.grad, d.cvel.grad, d.cacc.grad],
    adj_outputs=[bc.smooth_tree],
    adjoint=True,
  )

  wp.copy(bc.smooth_tree, d.cacc.grad)
  for level in reversed(m.body_tree):
    wp.launch(_subtree_acc_sv, dim=(nworld, level.size), inputs=[m.body_parentid, level], outputs=[bc.smooth_tree])
  wp.launch(
    _rne_dof_vjp,
    dim=(nworld, nv),
    inputs=[m.dof_bodyid, d.qvel, d.qacc, d.cdof, d.cdof_dot, d.cfrc_int, lam, bc.smooth_tree, flg_acc],
    outputs=[bc.smooth_adj_qvel, bc.smooth_adj_qacc, d.cdof_dot.grad, d.cdof.grad],
  )
  return {
    "qvel": bc.smooth_adj_qvel,
    "qacc": bc.smooth_adj_qacc,
    "cdof": d.cdof.grad,
    "cdof_dot": d.cdof_dot.grad,
    "cinert": d.cinert.grad,
    "cvel": d.cvel.grad,
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


@wp.func
def _cv_scatter3(
  # Data in:
  qvel_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  w: int,
  dofid: int,
  t: wp.spatial_vector,
  # Out:
  adj_qvel_out: wp.array2d[float],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
):
  for axis in range(3):
    _cv_scatter(qvel_in, cdof_in, w, dofid + 2 - axis, t, adj_qvel_out, adj_cdof_out)


@wp.func
def _comvel_rot3_vjp(
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_cdof_dot: wp.array2d[wp.spatial_vector],
  w: int,
  dofid: int,
  velocity: wp.spatial_vector,
  # Out:
  h_out: wp.array2d[wp.spatial_vector],
  k_out: wp.array2d[wp.spatial_vector],
) -> wp.spatial_vector:
  hsum = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for axis in range(3):
    i = dofid + axis
    g = adj_cdof_dot[w, i]
    h = math.motion_cross_force(cdof_in[w, i], g)
    h_out[w, i] = h
    k_out[w, i] = (-1.0) * math.motion_cross_force(velocity, g)
    hsum += h
  return hsum


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
    if jt == JointType.FREE:
      for axis in range(3):
        i = d + axis
        u += cdof_in[w, i] * qvel_in[w, i]
        h_out[w, i] = zero
        k_out[w, i] = zero
      Hb += _comvel_rot3_vjp(cdof_in, adj_cdof_dot, w, d + 3, u, h_out, k_out)
      for axis in range(3):
        i = d + 3 + axis
        u += cdof_in[w, i] * qvel_in[w, i]
    elif jt == JointType.BALL:
      Hb += _comvel_rot3_vjp(cdof_in, adj_cdof_dot, w, d, u, h_out, k_out)
      for axis in range(3):
        i = d + axis
        u += cdof_in[w, i] * qvel_in[w, i]
    else:
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
  W_in: wp.array2d[wp.spatial_vector],
  # Out:
  adj_qvel_out: wp.array2d[float],  # (+=): same-body T_j part
  adj_cdof_out: wp.array2d[wp.spatial_vector],  # (+=): same-body T_j part + direct k_j
):
  w, b = wp.tid()
  t = W_in[w, b]
  jntadr = body_jntadr[b]
  jntnum = body_jntnum[b]
  for jr in range(jntnum):
    jj = jntnum - 1 - jr  # reverse joint order
    jt = jnt_type[jntadr + jj]
    d = jnt_dofadr[jntadr + jj]
    if jt == JointType.FREE:
      _cv_scatter3(qvel_in, cdof_in, w, d + 3, t, adj_qvel_out, adj_cdof_out)
      for axis in range(3):
        i = d + 5 - axis
        t += h_in[w, i]
        adj_cdof_out[w, i] += k_in[w, i]
      _cv_scatter3(qvel_in, cdof_in, w, d, t, adj_qvel_out, adj_cdof_out)
    elif jt == JointType.BALL:
      _cv_scatter3(qvel_in, cdof_in, w, d, t, adj_qvel_out, adj_cdof_out)
      for axis in range(3):
        i = d + 2 - axis
        t += h_in[w, i]
        adj_cdof_out[w, i] += k_in[w, i]
    else:
      _cv_scatter(qvel_in, cdof_in, w, d, t, adj_qvel_out, adj_cdof_out)
      t = t + h_in[w, d]
      adj_cdof_out[w, d] = adj_cdof_out[w, d] + k_in[w, d]


def comvel_backward(
  m: Model,
  d: Data,
  adj_cvel: wp.array2d,
  adj_cdof_dot: wp.array2d,
  bc: BackwardContext | None = None,
):
  """Accumulates the com-velocity reverse into d.cdof.grad."""
  standalone = bc is None
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  nworld, nbody = d.nworld, m.nbody
  adj_cdof = bc.smooth_cv_adj_cdof if standalone else d.cdof.grad
  if standalone:
    adj_cdof.zero_()
  bc.smooth_cv_adj_qvel.zero_()
  wp.launch(
    _comvel_vjp_local,
    dim=(nworld, nbody),
    inputs=[m.body_parentid, m.body_jntnum, m.body_jntadr, m.jnt_type, m.jnt_dofadr, d.qvel, d.cdof, d.cvel, adj_cdof_dot],
    outputs=[bc.smooth_h, bc.smooth_k, bc.smooth_H],
  )
  wp.copy(bc.smooth_W, adj_cvel)
  for level in reversed(m.body_tree):
    wp.launch(_comvel_W_acc, dim=(nworld, level.size), inputs=[m.body_parentid, level, bc.smooth_H], outputs=[bc.smooth_W])
  wp.launch(
    _comvel_vjp_samebody,
    dim=(nworld, nbody),
    inputs=[m.body_jntnum, m.body_jntadr, m.jnt_type, m.jnt_dofadr, d.qvel, d.cdof, bc.smooth_h, bc.smooth_k, bc.smooth_W],
    outputs=[bc.smooth_cv_adj_qvel, adj_cdof],
  )
  return {"qvel": bc.smooth_cv_adj_qvel, "cdof": adj_cdof}


# --------------------------------------------------------------------------------------------
# cinert reverse: the last kinematic dqpos leaf. cinert depends on qpos via ximat / xipos /
# subtree_com[root]: source-AD the body-local _cinert leaf, chain xipos/ximat -> qpos via
# support.jac_dof, route adj_subtree_com through _subtree_com_qpos_vjp. cvel/cdof_dot have no
# direct qpos path (it flows through the total adj_cdof via _cdof_qpos_vjp).
# --------------------------------------------------------------------------------------------


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
    tau = adjoint_util._adj_rotation(ximat_in[w, b], adj_ximat[w, b])
    acc += wp.dot(jacp, adj_xipos[w, b]) + wp.dot(jacr, tau)
  res_dof_out[w, k] += acc


def _cinert_vjp(m: Model, d: Data, adj_cinert: wp.array2d, bc: BackwardContext):
  # Explicit sinks prevent Warp from falling back to the model parameters' own .grad arrays.
  bc.res_body_mass.zero_()
  bc.res_body_inertia.zero_()
  wp.launch(
    adjoint_recompute._cinert_recompute,
    dim=(d.nworld, m.nbody),
    inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com],
    outputs=[d.cinert],
    adj_inputs=[None, bc.res_body_mass, bc.res_body_inertia, d.xipos.grad, d.ximat.grad, d.subtree_com.grad],
    adj_outputs=[adj_cinert],
    adjoint=True,
  )


def _kinematic_qpos_vjp(m: Model, d: Data, adj_cdof: wp.array2d, bc: BackwardContext):
  nworld, nv = d.nworld, m.nv
  smooth_res_dof, smooth_res_qpos = bc.smooth_res_dof, bc.smooth_res_qpos
  smooth_res_dof.zero_()
  smooth_res_qpos.zero_()
  wp.launch(
    _cinert_pose_dof_vjp,
    dim=(nworld, nv),
    inputs=[
      m.nbody,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.ximat,
      d.subtree_com,
      d.cdof,
      d.xipos.grad,
      d.ximat.grad,
    ],
    outputs=[smooth_res_dof],
  )
  smooth_kinematics_adjoint.kinematics_qpos_backward(
    m,
    d,
    d.qpos,
    adj_cdof,
    d.subtree_com.grad,
    smooth_res_dof,
    bc.smooth_ceff,
    smooth_res_qpos,
  )
  return smooth_res_qpos


# Compatibility exports.
rne_qpos_vjp = smooth_force_backward
