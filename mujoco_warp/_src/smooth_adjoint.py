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
"""Reduced backward-only smooth-force replay (MJPLAN_ADRNE §0/§9/§10/§15).

This is the production smooth-position VJP for ``adjoint.step_backward``: ONE shared, backward-only
kinematic reverse of the reduced forward

    kinematics -> com_pos -> com_vel -> rne(flg_acc=True)

seeded with the IFT cotangent ``lambda`` on ``qfrc_bias`` (and later contact ``res_cdof`` /
``res_subtree_com`` / pose seeds), producing ``qpos`` (and eventually model-parameter) cotangents.  It
deliberately excludes CRB, factorization, the solver, and the rest of ``fwd_position`` / ``fwd_velocity``:
``rne(flg_acc=True)`` already evaluates the rigid-body contraction ``M_body(q)*qacc + b_RNE(q,qvel)``, so
no packed-M / CRB reverse is needed for the ordinary smooth-residual qpos VJP (CRB is a separate later
reverse, only for a genuine packed-M / grad-H cotangent).

DESIGN (the pivot away from the per-effect AD-RNE/com_vel/spring/actuator stacks in ``adjoint.py``):

  * Primal is REUSED, not shadowed: the caller runs the forward ``smooth.*`` on a non-grad scratch
    (``enable_backward=False`` -> bit-identical), and this module only issues the REVERSE.  Each
    body-local NONLINEAR leaf (``_cfrc``, ``_cinert``) is re-run inside its own backward-enabled kernel so
    Warp source-AD's it (the reverse calls the SAME ``math.*`` @wp.func leaves as the forward; Warp
    generates a called func's adjoint in the calling backward-enabled module even though ``smooth.py`` is
    ``enable_backward=False``).
  * Destructive TREE reductions are reversed PER TREE DEPTH over ``m.body_tree`` (out-of-place / manual
    transpose), NOT with the O(nbody^2) ancestry walks of the ``adjoint.py`` oracle and NOT with any
    ``_CV_MAX_DEPTH`` bound.  A body's joint/DOF loops stay inside ``enable_backward=False`` manual
    kernels, where dynamic bounds are safe (Warp never has to replay them); MuJoCo's ``body_dofnum <= 6``
    keeps them tiny.
  * No nested ``wp.Tape``; explicit ``adjoint=True`` launches with explicit cotangent arrays; preallocated
    reusable scratch -> the fixed per-model launch sequence is graph-capturable.

OWNERSHIP / no double-count (MJPLAN_ADRNE §0.2, §15.10): the qvel channel stays with
``derivative.deriv_smooth_vel`` (``step_backward`` §4).  This replay requests qpos + model cotangents
ONLY; the adj_qvel it computes internally (Coriolis + the cdof_dot snapshot terms) is DISCARDED so the
Coriolis qvel derivative is not counted twice.

STATUS: bring-up (``smooth_adjoint_test``, all FD-gated vs float64 MuJoCo-C + A/B vs the ``adjoint.py``
manual oracles over the §14 scene matrix incl. six-joint body, depth>32, zero-joint, multiworld):
  * ``smooth_force_backward`` -- rne(flg_acc=True) reverse + per-depth tree reductions; matches
    ``rne_qpos_vjp`` and float64 ``mj_rne``.
  * contact-seed merge (``res_cdof_extra`` / ``res_subtree_extra``) -- one shared kinematic reverse for
    bias+contact, matches ``collision_adjoint.contact_qpos_vjp``.
  * ``spring_qpos_vjp`` -- complete FREE/BALL/SLIDE/HINGE source-AD spring leaf; matches float64 FD.
Warp 1.14 reverse compliance: every backward-enabled (source-AD) kernel here is LOOP-FREE (no static six-
slot unroll needed); every kernel with a body-joint loop is ``enable_backward=False`` (manual VJP, dynamic
bound safe).  NOT yet wired into ``step_backward``; carries NO finite-difference fallback (FD is
validation-only).  TODO before default-on: actuator staged reverse (transmission VJPs); tendon/gravcomp/
fluid/flex/activation leaves + capability assertions; scratch PREALLOCATION for graph capture (the whole
backward, incl. step_backward, currently allocates per-call); CUDA + capture + end-to-end-step/BPTT gates.
"""

import numpy as np  # host-only: the one-time, cached capability assertion (assert_smooth_supported)
import warp as wp

from mujoco_warp._src import adjoint as _adjoint
from mujoco_warp._src import collision_adjoint as _collision_adjoint
from mujoco_warp._src import math as _math
from mujoco_warp._src import smooth as _smooth
from mujoco_warp._src import support as _support
from mujoco_warp._src import types as _types
from mujoco_warp._src import util_misc as _util_misc
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec10f

_SV = wp.spatial_vector
_FREE = int(_types.JointType.FREE.value)
_BALL = int(_types.JointType.BALL.value)
_SPRING = int(_types.DisableBit.SPRING.value)
_ACTUATION = int(_types.DisableBit.ACTUATION.value)
_CLAMPCTRL = int(_types.DisableBit.CLAMPCTRL.value)
_GRAVITY = int(_types.DisableBit.GRAVITY.value)
_GAIN_AFFINE = int(_types.GainType.AFFINE.value)
_BIAS_AFFINE = int(_types.BiasType.AFFINE.value)


# ----------------------------------------------------------------------------
# Per-depth tree-reduction reverses over m.body_tree (replace the adjoint.py O(nbody^2) ancestry-walk
# oracles _rne_cfrc_tree_vjp / _rne_cacc_subtree_sum / _comvel_subtree_W). All enable_backward=False
# (manual VJP -> not replayed), all out-of-place across depths: a body reads only an already-finalized
# parent/child depth, so writes never race. Mirrors the forward smooth._subtree_com_acc schedule.
# ----------------------------------------------------------------------------
@wp.kernel(enable_backward=False)
def _anc_acc_sv(
  body_parentid: wp.array[int],
  body_tree_: wp.array[int],
  val_io: wp.array2d[wp.spatial_vector],  # init = local; root->leaves: val[b] += val[parent(b)]
):
  """ANCESTOR accumulation (root->leaves), the transpose of the child->parent force sum
  F_b = f_b + sum_children F_c (so f_d's adjoint is sum over ancestors-or-self of adj_F). One launch per
  depth in FORWARD order; the parent sits one depth shallower and is already finalized."""
  w, nodeid = wp.tid()
  b = body_tree_[nodeid]
  if b != 0:
    val_io[w, b] = val_io[w, b] + val_io[w, body_parentid[b]]


@wp.kernel(enable_backward=False)
def _subtree_acc_sv(
  body_parentid: wp.array[int],
  body_tree_: wp.array[int],
  val_io: wp.array2d[wp.spatial_vector],  # init = local; leaves->root: val[parent] += val[b]
):
  """SUBTREE sum (leaves->root): after all depths, val[B] = sum_{d in subtree(B)} local[d]. One launch per
  depth in REVERSED order; mirrors smooth._subtree_com_acc exactly."""
  w, nodeid = wp.tid()
  b = body_tree_[nodeid]
  if b != 0:
    wp.atomic_add(val_io, w, body_parentid[b], val_io[w, b])


@wp.kernel(enable_backward=False)
def _acc_vec3(src: wp.array2d[wp.vec3], io: wp.array2d[wp.vec3]):
  """io += src (merge a contact subtree-COM seed into the bias's, so one COM-Jacobian pass carries both)."""
  w, i = wp.tid()
  io[w, i] = io[w, i] + src[w, i]


@wp.kernel(enable_backward=False)
def _comvel_W_acc(
  body_parentid: wp.array[int],
  body_tree_: wp.array[int],
  H_in: wp.array2d[wp.spatial_vector],
  W_io: wp.array2d[wp.spatial_vector],  # init = adj_cvel (A); leaves->root: W[parent] += W[b] + H[b]
):
  """com_vel CV3 augmented-seed subtree sum via the §10.1A.7 reverse-depth recurrence
  W_b = A_b + sum_{c child}(W_c + H_c)  ==>  W_B = sum_{subtree(B)} A + sum_{strict_subtree(B)} H.
  One launch per depth in REVERSED order (children finalized before the parent reads them)."""
  w, nodeid = wp.tid()
  b = body_tree_[nodeid]
  if b != 0:
    wp.atomic_add(W_io, w, body_parentid[b], W_io[w, b] + H_in[w, b])


# ----------------------------------------------------------------------------
# The shared reduced reverse.
# ----------------------------------------------------------------------------
def smooth_force_backward(m: Model, d: Data, lam: wp.array2d, flg_acc: bool = True,
                          res_cdof_extra: wp.array2d = None, res_subtree_extra: wp.array2d = None):
  """Reduced smooth-force qpos VJP: given the IFT cotangent ``lam`` on ``qfrc_bias`` (= M_body*qacc +
  b_RNE), return ``res_qpos`` (nq) = ``(d(lam^T qfrc_bias)/dqpos)``, the rigid-body smooth-residual qpos
  column.  ``d`` must hold the converged forward intermediates (kinematics -> com_pos -> com_vel ->
  rne(flg_acc) on a non-grad scratch); read-only.  The qvel channel is intentionally DISCARDED (owned by
  derivative.deriv_smooth_vel).

  CONTACT-SEED MERGE (MJPLAN_ADRNE §10): the contact residual reverse's ``res_cdof`` (per-dof) and
  ``res_subtree_com`` (per-body) cotangents are the SAME ``cdof``/``subtree_com`` channels the rne bias
  feeds.  Passing them as ``res_cdof_extra`` / ``res_subtree_extra`` adds them into the bias's seeds BEFORE
  the (linear) cdof screw-commutator + COM-Jacobian reverse, so that destructive kinematic reverse runs
  ONCE for bias+contact instead of twice.  By linearity this equals the separate paths summed.  The
  contact NARROWPHASE-pose ∂qpos is orthogonal (geometry, no overlap with the bias) and stays in
  ``collision_adjoint.contact_qpos_vjp`` -- in production it is added with its cdof/subtree seeds zeroed.

  Reverse spine (one shared kinematic reverse; per-depth tree reductions over m.body_tree):
    RNE      lam -> adj_cdof(tau) + adj_force ; [per-depth] adj_force -> adj_f ; source-AD _cfrc leaf ->
             adj_{cinert,cacc,cvel} ; [per-depth] adj_cacc -> subtree ; dof reverse -> adj_{cdof_dot,cdof,...}
    com_vel  CV1 local snapshot + cross VJP ; CV2 same-body event scan ; [per-depth] CV3 W ; CV4 scatter
             -> adj_cdof (Coriolis cdof path; adj_qvel discarded)
    kinem.   source-AD _cinert leaf -> adj_{xipos,ximat,subtree_com} -> dof (jac_dof) ; total adj_cdof
             screw-commutator -> dof ; subtree-COM Jacobian -> dof ; FREE/BALL quaternion lift -> qpos
  """
  nworld = d.qvel.shape[0]
  nv = m.nv
  nq = d.qpos.shape[1]
  nbody = m.nbody

  # ---- RNE reverse (K1, K2', K3, K4a', K4b) ----
  adj_cdof = wp.zeros((nworld, nv), dtype=_SV)
  adj_cdof_dot = wp.zeros((nworld, nv), dtype=_SV)
  adj_qvel = wp.zeros((nworld, nv), dtype=float)  # DISCARDED (deriv_smooth_vel owns qvel)
  adj_qacc = wp.zeros((nworld, nv), dtype=float)  # DISCARDED here (rne_backward exposes it for the qacc gate)
  adj_force = wp.zeros((nworld, nbody), dtype=_SV)

  # K1: lam -> adj_cdof (tau projection) + adj_force (body force seed)
  wp.launch(_rne_qfrcbias_cdof_vjp, dim=(nworld, nv),
            inputs=[m.dof_bodyid, d.cfrc_int, lam], outputs=[adj_cdof])
  wp.launch(_rne_qfrcbias_force_vjp, dim=(nworld, nbody),
            inputs=[m.body_dofadr, m.body_dofnum, d.cdof, lam], outputs=[adj_force])

  # K2' (per-depth): adj_force -> adj_f (ancestor accumulation, root->leaves)
  adj_f = wp.clone(adj_force)
  for body_tree in m.body_tree:
    wp.launch(_anc_acc_sv, dim=(nworld, body_tree.size),
              inputs=[m.body_parentid, body_tree], outputs=[adj_f])

  # K3: adj_f -> adj_{cinert,cacc,cvel} via source-AD of the local inertial-force leaf
  cfrc_local = wp.zeros((nworld, nbody), dtype=_SV)
  adj_cinert = wp.zeros((nworld, nbody), dtype=vec10f)
  adj_cacc = wp.zeros((nworld, nbody), dtype=_SV)
  adj_cvel = wp.zeros((nworld, nbody), dtype=_SV)
  cfrc_inputs = [d.cinert, d.cacc, d.cvel]
  wp.launch(_rne_cfrc_recompute, dim=(nworld, nbody), inputs=cfrc_inputs, outputs=[cfrc_local])
  wp.launch(_rne_cfrc_recompute, dim=(nworld, nbody), inputs=cfrc_inputs, outputs=[cfrc_local],
            adj_inputs=[adj_cinert, adj_cacc, adj_cvel], adj_outputs=[adj_f], adjoint=True)

  # K4a' (per-depth): adj_cacc -> subtree_adj_cacc (subtree sum, leaves->root)
  subtree_adj_cacc = wp.clone(adj_cacc)
  for body_tree in reversed(m.body_tree):
    wp.launch(_subtree_acc_sv, dim=(nworld, body_tree.size),
              inputs=[m.body_parentid, body_tree], outputs=[subtree_adj_cacc])

  # K4b: subtree_adj_cacc -> adj_{qvel,qacc,cdof_dot, cdof +=} (transpose the cacc dof sweep)
  wp.launch(_rne_cacc_dof_vjp, dim=(nworld, nv),
            inputs=[m.dof_bodyid, d.cdof, d.cdof_dot, d.qvel, d.qacc, subtree_adj_cacc, flg_acc],
            outputs=[adj_qvel, adj_qacc, adj_cdof_dot, adj_cdof])

  # ---- com_vel reverse (CV1, CV2, CV3', CV4): Coriolis cdof path; adj_qvel discarded ----
  h = wp.zeros((nworld, nv), dtype=_SV)
  kk = wp.zeros((nworld, nv), dtype=_SV)
  Hbody = wp.zeros((nworld, nbody), dtype=_SV)
  cv_adj_qvel = wp.zeros((nworld, nv), dtype=float)  # DISCARDED
  cv_adj_cdof = wp.zeros((nworld, nv), dtype=_SV)
  wp.launch(_comvel_vjp_local, dim=(nworld, nbody),
            inputs=[m.body_parentid, m.body_jntadr, m.body_jntnum, m.jnt_type, m.jnt_dofadr,
                    d.cvel, d.cdof, d.qvel, adj_cdof_dot], outputs=[h, kk, Hbody])
  wp.launch(_comvel_vjp_samebody, dim=(nworld, nbody),
            inputs=[m.body_jntadr, m.body_jntnum, m.jnt_type, m.jnt_dofadr, d.cdof, d.qvel, h, kk],
            outputs=[cv_adj_qvel, cv_adj_cdof])
  W = wp.clone(adj_cvel)  # CV3' (per-depth): W_B = sum_subtree(B) adj_cvel + sum_strict_subtree(B) H
  for body_tree in reversed(m.body_tree):
    wp.launch(_comvel_W_acc, dim=(nworld, body_tree.size),
              inputs=[m.body_parentid, body_tree, Hbody], outputs=[W])
  wp.launch(_comvel_scatter_W, dim=(nworld, nv),
            inputs=[m.dof_bodyid, d.cdof, d.qvel, W], outputs=[cv_adj_qvel, cv_adj_cdof])

  # total cdof cotangent = rne-proper + com_vel (+ optional contact res_cdof seed -> one shared reverse)
  total_cdof = wp.zeros((nworld, nv), dtype=_SV)
  wp.launch(_add_spatial, dim=(nworld, nv), inputs=[adj_cdof, cv_adj_cdof], outputs=[total_cdof])
  if res_cdof_extra is not None:
    wp.launch(_add_spatial, dim=(nworld, nv), inputs=[total_cdof, res_cdof_extra], outputs=[total_cdof])

  # ---- kinematic reverse: adj_{cinert, total_cdof} -> qpos ----
  res_dof = wp.zeros((nworld, nv), dtype=float)
  adj_subtree = cinert_qpos_vjp(m, d, adj_cinert, res_dof)  # _cinert leaf + pose->dof; subtree seed
  if res_subtree_extra is not None:  # merge the contact subtree-COM seed into the bias's
    wp.launch(_acc_vec3, dim=(nworld, nbody), inputs=[res_subtree_extra], outputs=[adj_subtree])
  wp.launch(_collision_adjoint._cdof_qpos_vjp, dim=(nworld, nv),
            inputs=[m.dof_parentid, m.dof_jntid, m.jnt_type, m.jnt_dofadr, d.cdof, total_cdof, nv],
            outputs=[res_dof])
  ceff = wp.empty((nworld, nbody), dtype=wp.vec3)
  wp.launch(_collision_adjoint._build_ceff, dim=(nworld, nbody),
            inputs=[m.body_rootid, m.dof_bodyid, d.cdof, total_cdof, adj_subtree, nv], outputs=[ceff])
  wp.launch(_collision_adjoint._subtree_com_qpos_vjp, dim=(nworld, nv),
            inputs=[m.body_parentid, m.body_rootid, m.dof_bodyid, m.body_isdofancestor,
                    m.body_mass, m.body_subtreemass, d.subtree_com, d.cdof, d.xipos, ceff, nbody],
            outputs=[res_dof])
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  wp.launch(_collision_adjoint._dof_to_qpos, dim=(nworld, m.njnt),
            inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof], outputs=[res_qpos])
  return res_qpos


def mass_matrix_qpos_vjp(m: Model, d: Data, y: wp.array2d, w: wp.array2d):
  """MASS-MATRIX-DERIVATIVE VJP: res_qpos (nworld, nq) = ∂_q[ yᵀ M(q) w ], holding y and w fixed.

  This is the state-direct term the implicit-integrator (implicitfast / eulerdamp) backward needs:
  advance_backward maps the solver root (ā_s = M Q⁻¹ ā_u) but must ALSO differentiate the integration
  operator itself.  For Q = M(q) + dt·D that direct term is ∂_q[ yᵀ M(q)(a_s − a_u) ] (+ a qvel term for
  state-dependent D); this primitive is the ∂_q[ yᵀ M(q) w ] piece with w = a_s − a_u.

  Mechanism (REUSE the RNE-bias reverse, no hand-derived ∂M/∂q): recompute a MASS-ONLY RNE bias on a
  NON-GRAD scratch at the step linearization d.qpos -- qvel = 0 (no Coriolis), qacc = w, ZERO root
  acceleration (no gravity) -- so qfrc_bias = M(q)·w exactly, then reverse it with smooth_force_backward
  seeded by lam = y.  Kinematics is re-run on the scratch: the cloned cinert/cdof sit at the PREVIOUS
  step's config, but d.qpos is THIS step's linearization.  A dedicated scratch (not d/d_out) leaves the
  live forward intermediates intact for the §3 contact residual VJP that runs afterward."""
  s = _adjoint._clone_for_fd(d)  # non-grad scratch; s.qpos = d.qpos
  s.qvel.zero_()
  s.qacc = w
  _smooth.kinematics(m, s)  # fresh xipos/ximat/cdof at d.qpos (the cloned kinematics are a step stale)
  _smooth.com_pos(m, s)     # fresh subtree_com/cinert/cdof at d.qpos
  s.cvel.zero_()            # qvel = 0 -> no cvel / cdof_dot (skip com_vel); mass-only has no Coriolis
  s.cdof_dot.zero_()
  s.cacc.zero_()            # root cacc = 0 -> NO gravity injected (mass-only)
  _smooth._rne_cacc_forward(m, s, flg_acc=True)  # cacc = Σ cdof·w  (the cdof_dot·qvel term is 0)
  _smooth._rne_cfrc(m, s)                         # cfrc_int_local = cinert·cacc  (cvel = 0 -> no gyroscopic)
  _smooth._rne_cfrc_backward(m, s)                # subtree-accumulate cfrc_int up the tree
  return smooth_force_backward(m, s, y, flg_acc=True)  # = ∂_q[ yᵀ M(q) w ]


# ----------------------------------------------------------------------------
# Joint-SPRING leaf (MJPLAN_ADRNE §0.2): the COMPLETE per-joint spring law from
# passive._spring_damper_dof_passive, for ALL joint types incl. FREE/BALL quaternion springs -- a safe
# source-AD leaf (loop-free per joint; reuses the forward math.quat_sub + util_misc._poly_force funcs, so
# Warp gets the sign and the quaternion derivative). Reads qpos DIRECTLY (the raw quaternion coords), so
# its adjoint lands on qpos in raw-coordinate space -- the same space _dof_to_qpos's lift and the
# _advance_state integrator adjoint produce -- with NO manual tangent lift needed. The viscous
# -damper*qvel term has zero qpos derivative and is omitted. Springs are a direct qpos function (NOT part
# of the kinematic reverse), so this is an additive res_qpos term the IFT boundary sums with the rne
# bias. Replaces adjoint._residual_spring_local (the HINGE/SLIDE-only prototype/oracle).
# ----------------------------------------------------------------------------
@wp.kernel(module="unique", enable_backward=True)
def _spring_qfrc_recompute(
  opt_disableflags: int,
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_stiffness: wp.array2d[float],
  jnt_stiffnesspoly: wp.array2d[wp.vec2],
  qpos_spring: wp.array2d[float],
  qpos_in: wp.array2d[float],  # [grad: ∂qpos] linearization point; adjoint routed to res_qpos
  qfrc_spring_out: wp.array2d[float],
):
  """EXACT reconstruction of the SPRING branch of passive._spring_damper_dof_passive (FREE/BALL/SLIDE/
  HINGE), source-AD'd wrt qpos. Seeding qfrc_spring_out's adjoint with -λ drops (∂qfrc_spring/∂qpos)ᵀ(-λ)
  = (∂(λᵀ(-qfrc_spring))/∂qpos) -- the spring part of the smooth residual's qpos VJP -- straight onto
  res_qpos. Mirrors the forward statement-for-statement; component writes are distinct array slots (no
  in-place vec-component update -> AD-safe)."""
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
    k = _util_misc._poly_force(stiffness, spoly, r, 0)
    qfrc_spring_out[w, dofid + 0] = -k * difx
    qfrc_spring_out[w, dofid + 1] = -k * dify
    qfrc_spring_out[w, dofid + 2] = -k * difz
    rot = wp.normalize(wp.quat(qpos_in[w, qposid + 3], qpos_in[w, qposid + 4],
                               qpos_in[w, qposid + 5], qpos_in[w, qposid + 6]))
    ref = wp.quat(qpos_spring[sid, qposid + 3], qpos_spring[sid, qposid + 4],
                  qpos_spring[sid, qposid + 5], qpos_spring[sid, qposid + 6])
    dif = _math.quat_sub(rot, ref)
    k_rot = _util_misc._poly_force(stiffness, spoly, wp.length(dif), 0)
    qfrc_spring_out[w, dofid + 3] = -k_rot * dif[0]
    qfrc_spring_out[w, dofid + 4] = -k_rot * dif[1]
    qfrc_spring_out[w, dofid + 5] = -k_rot * dif[2]
  elif jnttype == _BALL:
    rot = wp.normalize(wp.quat(qpos_in[w, qposid + 0], qpos_in[w, qposid + 1],
                               qpos_in[w, qposid + 2], qpos_in[w, qposid + 3]))
    ref = wp.quat(qpos_spring[sid, qposid + 0], qpos_spring[sid, qposid + 1],
                  qpos_spring[sid, qposid + 2], qpos_spring[sid, qposid + 3])
    dif = _math.quat_sub(rot, ref)
    k = _util_misc._poly_force(stiffness, spoly, wp.length(dif), 0)
    qfrc_spring_out[w, dofid + 0] = -k * dif[0]
    qfrc_spring_out[w, dofid + 1] = -k * dif[1]
    qfrc_spring_out[w, dofid + 2] = -k * dif[2]
  else:  # SLIDE / HINGE
    fdif = qpos_in[w, qposid] - qpos_spring[sid, qposid]
    qfrc_spring_out[w, dofid] = -fdif * _util_misc._poly_force(stiffness, spoly, fdif, 0)


def spring_qpos_vjp(m: Model, d: Data, lam: wp.array2d):
  """Joint-spring contribution to the smooth-residual qpos VJP: res_qpos (nq) =
  ``(∂(λᵀ(-qfrc_spring))/∂qpos)``, all joint types (FREE/BALL/SLIDE/HINGE). Source-AD of the complete
  forward spring law, seeded ``-λ`` on qfrc_spring. ``d.qpos`` is the linearization point; never mutated.
  Additive with smooth_force_backward at the IFT boundary."""
  nworld = d.qpos.shape[0]
  nv = m.nv
  nq = d.qpos.shape[1]
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  qfrc_spring = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
  sp_in = [m.opt.disableflags, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_stiffness,
           m.jnt_stiffnesspoly, m.qpos_spring, d.qpos]
  wp.launch(_spring_qfrc_recompute, dim=(nworld, m.njnt), inputs=sp_in, outputs=[qfrc_spring])
  wp.launch(_adjoint._neg_cols, dim=(nworld, nv), inputs=[lam], outputs=[qfrc_spring.grad])  # seed -λ
  wp.launch(_spring_qfrc_recompute, dim=(nworld, m.njnt), inputs=sp_in, outputs=[qfrc_spring],
            adj_inputs=[None, None, None, None, None, None, None, res_qpos],  # qpos adjoint -> res_qpos
            adj_outputs=[qfrc_spring.grad], adjoint=True)
  return res_qpos


# ----------------------------------------------------------------------------
# Actuator leaf (MJPLAN_ADRNE §0.2): the STAGED reusable reverse of qfrc_actuator for the SUPPORTED subset
# -- AFFINE gain/bias on a JOINT transmission to a scalar (slide/hinge) joint. With
#     u_a = Σ_i B_ai qvel_i,   τ_i = Σ_a B_ai F_a,   F_a = Φ_a(ℓ_a, u_a, ctrl, act, params),
# the qpos VJP (seed adj_τ = -λ, the -qfrc_actuator residual term) is, stage by stage:
#   (1) frozen force/ctrl clamps -> the saturation mask;
#   (2) manual transpose of τ = BᵀF: adj_F_a = Σ_i B_ai·(-λ_i) = -ml_a, and adj_B_ai += (-λ_i)F_a
#       (ZERO for a JOINT transmission -- moment is constant in qpos);
#   (3) source-AD of Φ_a's LENGTH derivative dfdl_a = ∂F_a/∂ℓ_a (affine closed form = the source-AD
#       result: [gain AFFINE]·gainprm[1]·ctrl_act + [bias AFFINE]·biasprm[1]); the u_a (velocity) channel
#       is discarded (owned by deriv_smooth_vel);
#   (4) JOINT transmission VJP: ∂ℓ_a/∂(tangent δθ_j) = B_aj -> res_dof[j] += -ml_a·dfdl_a·B_aj.
# Reuses adjoint._actuator_qpos_vjp (the kernel that fuses 1-4) + the _dof_to_qpos lift. COMPLETION (gated
# off by assert_smooth_supported, NOT FD): BALL-joint length is quaternion-dependent (stage-4 VJP differs);
# JOINTINPARENT makes the moment qpos-dependent (stage-2 adj_B nonzero); MUSCLE/DC-motor Φ need the source-
# AD nonlinear leaf; tendon/SITE/BODY/slider-crank need their topology VJP; activation (na>0) needs the act
# reverse. DisableBit.ACTUATION suppresses the whole term (matching the forward).
# ----------------------------------------------------------------------------
def actuator_qpos_vjp(m: Model, d: Data, lam: wp.array2d):
  """Affine JOINT-transmission actuator contribution to the smooth-residual qpos VJP: res_qpos (nq) =
  ``(∂(λᵀ(-qfrc_actuator))/∂qpos)``. ``d`` holds the frozen forward intermediates (actuator_moment/force at
  the linearization, ctrl); never mutated. Gated to the supported subset by assert_smooth_supported."""
  nworld = d.qpos.shape[0]
  nv = m.nv
  nq = d.qpos.shape[1]
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  if m.nu == 0 or (int(m.opt.disableflags) & _ACTUATION) != 0:  # forward suppresses actuation -> so do we
    return res_qpos
  res_dof = wp.zeros((nworld, nv), dtype=float)
  wp.launch(_actuator_qpos_vjp, dim=(nworld, m.nu),
            inputs=[m.actuator_gaintype, m.actuator_biastype, m.actuator_ctrllimited, m.actuator_forcelimited,
                    m.actuator_gainprm, m.actuator_biasprm, m.actuator_ctrlrange, m.actuator_forcerange,
                    d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment, d.actuator_force,
                    d.ctrl, lam, int(m.opt.disableflags) & _CLAMPCTRL],
            outputs=[res_dof])
  wp.launch(_collision_adjoint._dof_to_qpos, dim=(nworld, m.njnt),
            inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof], outputs=[res_qpos])
  return res_qpos


# Cache of the capability-check result per Model (host metadata is static; the per-element .numpy() reads
# run ONCE here, eager, at first call -- a setup-time assertion like io.is_sparse, NEVER per-step or inside
# a graph-capture region).
_SUPPORTED_CACHE = {}


def assert_smooth_supported(m: Model):
  """Capability ASSERTION for the analytic smooth-force reverse (MJPLAN_ADRNE §0.2/§15.7): raise
  NotImplementedError naming any ENABLED smooth feature whose analytic leaf/topology VJP is not yet
  implemented. NO finite-difference fallback -- FD is an explicit test-only path (the caller's flag), never
  an automatic substitute. The supported set is: rigid-body RNE bias; FREE/BALL/SLIDE/HINGE joint springs;
  AFFINE/FIXED-gain + AFFINE/NONE-bias actuators on a JOINT transmission to a slide/hinge joint; viscous
  damping + armature (no qpos dep). Result cached per Model (one-time host read)."""
  key = id(m)
  if _SUPPORTED_CACHE.get(key):
    return
  bad = []
  if int(m.ntendon) != 0:
    bad.append("tendons (spring/bias/armature/transmission)")
  opt = m.opt
  if np.any(opt.density.numpy() != 0.0) or np.any(opt.viscosity.numpy() != 0.0) or np.any(opt.wind.numpy() != 0.0):
    bad.append("fluid forces (opt.density/viscosity/wind)")
  if np.any(m.body_gravcomp.numpy() != 0.0) and np.any(m.jnt_actgravcomp.numpy() != 0):
    bad.append("gravcomp routed to an actuator (jnt_actgravcomp; force-limit clamp) -- passive-bucket gravcomp is supported")
  if int(m.nu) > 0:
    trntype = m.actuator_trntype.numpy()
    gaintype = m.actuator_gaintype.numpy()
    biastype = m.actuator_biastype.numpy()
    jnt_type = m.jnt_type.numpy()
    if np.any(trntype != int(_types.TrnType.JOINT.value)):
      bad.append("non-JOINT actuator transmission (tendon/site/body/slider-crank/jointinparent)")
    ok_gain = (gaintype == int(_types.GainType.FIXED.value)) | (gaintype == int(_types.GainType.AFFINE.value))
    ok_bias = (biastype == int(_types.BiasType.NONE.value)) | (biastype == int(_types.BiasType.AFFINE.value))
    if not ok_gain.all() or not ok_bias.all():
      bad.append("non-affine actuator gain/bias (muscle/DC-motor/user)")
    jnt_ids = m.actuator_trnid.numpy()[:, 0][trntype == int(_types.TrnType.JOINT.value)]
    if jnt_ids.size and np.any((jnt_type[jnt_ids] == _FREE) | (jnt_type[jnt_ids] == _BALL)):
      bad.append("actuator on a FREE/BALL joint (quaternion-dependent transmission length)")
    # Activation (na>0) is SUPPORTED for the ∂qpos when gain is FIXED: dfdl = biasprm[1] and ctrl_act (=act)
    # is unused (the common filtered/integrated position/velocity servo). A STATEFUL AFFINE-gain actuator
    # would need the gain term gainprm[1]·act (ctrl_act=act, not clamped ctrl) -- not implemented.
    stateful = m.actuator_dyntype.numpy() != int(_types.DynType.NONE.value)
    if np.any(stateful & (gaintype == int(_types.GainType.AFFINE.value))):
      bad.append("stateful (na>0) AFFINE-gain actuator (ctrl_act=act not implemented)")
  if bad:
    raise NotImplementedError(
      "smooth_adjoint analytic reverse does not support: " + "; ".join(bad)
      + ". Implement the leaf/topology VJP, or run the explicit FD path for testing "
      "(adjoint._USE_ANALYTIC_RNE_QPOS=False). No silent FD fallback (MJPLAN_ADRNE §0)."
    )
  _SUPPORTED_CACHE[key] = True


# ----------------------------------------------------------------------------
# Gravcomp leaf (MJPLAN_ADRNE §0.2 completion). qfrc_gravcomp[dof] = Σ_b jac_dof(xipos_b)·(-g·m_b·gravcomp_b)
# (passive._gravity_force): a COM-Jacobian contraction with a CONSTANT per-body force, so its qpos
# dependence flows ONLY through the kinematic intermediates jac_dof reads -- cdof, subtree_com, and the
# COM point xipos. Source-AD the exact forward leaf (reuse support.jac_dof) to get adj_{cdof, subtree_com,
# xipos}, then route each through the SAME kinematic reverse the rne bias uses (cdof screw-commutator, COM
# Jacobian, xipos pose VJP). gravcomp enters the residual as -qfrc_gravcomp (added to qfrc_passive when
# gravity_enabled and not jnt_actgravcomp -> the passive bucket; the actuator-bucket case is asserted off
# because its force-limit clamp would change the slope). Seed -λ masked by (gravity_enabled & ~actgravcomp).
# ----------------------------------------------------------------------------
@wp.kernel(module="unique", enable_backward=True)
def _gravity_force_recompute(
  opt_gravity: wp.array[wp.vec3],
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_gravcomp: wp.array2d[float],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  xipos_in: wp.array2d[wp.vec3],  # [grad] body COM (the jac_dof point)
  subtree_com_in: wp.array2d[wp.vec3],  # [grad]
  cdof_in: wp.array2d[wp.spatial_vector],  # [grad]
  qfrc_gravcomp_out: wp.array2d[float],
):
  """EXACT reconstruction of passive._gravity_force (reuses support.jac_dof), source-AD'd wrt the kinematic
  intermediates xipos/subtree_com/cdof. force = -g·m·gravcomp is constant (not differentiated)."""
  worldid, bodyid, dofid = wp.tid()
  bodyid += 1  # skip world body
  gravcomp = body_gravcomp[worldid % body_gravcomp.shape[0], bodyid]
  gravity = opt_gravity[worldid % opt_gravity.shape[0]]
  if gravcomp:
    force = -gravity * body_mass[worldid % body_mass.shape[0], bodyid] * gravcomp
    pos = xipos_in[worldid, bodyid]
    jac, _ = _support.jac_dof(body_parentid, body_rootid, dof_bodyid, body_isdofancestor,
                              subtree_com_in, cdof_in, pos, bodyid, dofid, worldid)
    wp.atomic_add(qfrc_gravcomp_out[worldid], dofid, wp.dot(jac, force))


@wp.kernel(enable_backward=False)
def _gravcomp_seed(
  lam: wp.array2d[float],
  dof_jntid: wp.array[int],
  jnt_actgravcomp: wp.array[int],
  gravity_enabled: int,
  grad_out: wp.array2d[float],  # qfrc_gravcomp.grad = -λ masked by (gravity_enabled & not actgravcomp)
):
  w, i = wp.tid()
  # -λ: gravcomp enters the residual r as -qfrc_gravcomp (added to qfrc_passive), so res_qpos =
  # (∂r/∂qpos)λ gets -(∂qfrc_gravcomp/∂qpos)λ. With the faithful kinematic reverse (cdof screw + _build_ceff
  # moving-COM fold + xipos FK), seeding -λ on qfrc_gravcomp yields exactly that. FD-gated (cos→+1 with the
  # full _build_ceff composition; without the fold the magnitude was ~1.88x off).
  if gravity_enabled != 0 and jnt_actgravcomp[dof_jntid[i]] == 0:
    grad_out[w, i] = -lam[w, i]
  else:
    grad_out[w, i] = 0.0


_HAS_GRAVCOMP_CACHE = {}


def gravcomp_qpos_vjp(m: Model, d: Data, lam: wp.array2d):
  """Gravity-compensation contribution to the smooth-residual qpos VJP: res_qpos (nq) =
  ``(∂(λᵀ(-qfrc_gravcomp))/∂qpos)`` (passive bucket). ``d`` holds the converged forward intermediates;
  never mutated. Returns zeros (cached one-time host check) when the model has no body gravcomp."""
  nworld = d.qpos.shape[0]
  nv = m.nv
  nq = d.qpos.shape[1]
  nbody = m.nbody
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  key = id(m)
  has = _HAS_GRAVCOMP_CACHE.get(key)
  if has is None:
    has = bool(np.any(m.body_gravcomp.numpy() != 0.0))
    _HAS_GRAVCOMP_CACHE[key] = has
  if not has:
    return res_qpos

  # 1. recompute qfrc_gravcomp on differentiable kinematic intermediates; seed -λ (masked) -> adjoints
  xipos = wp.clone(d.xipos)
  xipos.requires_grad = True
  subtree = wp.clone(d.subtree_com)
  subtree.requires_grad = True
  cdof = wp.clone(d.cdof)
  cdof.requires_grad = True
  qfrc_gc = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
  gc_in = [m.opt.gravity, m.body_parentid, m.body_rootid, m.body_mass, m.body_gravcomp,
           m.dof_bodyid, m.body_isdofancestor, xipos, subtree, cdof]
  wp.launch(_gravity_force_recompute, dim=(nworld, nbody - 1, nv), inputs=gc_in, outputs=[qfrc_gc])
  gravity_enabled = 1 if (int(m.opt.disableflags) & _GRAVITY) == 0 else 0
  wp.launch(_gravcomp_seed, dim=(nworld, nv),
            inputs=[lam, m.dof_jntid, m.jnt_actgravcomp, gravity_enabled], outputs=[qfrc_gc.grad])
  adj_xipos = wp.zeros((nworld, nbody), dtype=wp.vec3)
  adj_subtree = wp.zeros((nworld, nbody), dtype=wp.vec3)
  adj_cdof = wp.zeros((nworld, nv), dtype=_SV)
  wp.launch(_gravity_force_recompute, dim=(nworld, nbody - 1, nv), inputs=gc_in, outputs=[qfrc_gc],
            adj_inputs=[None, None, None, None, None, None, None, adj_xipos, adj_subtree, adj_cdof],
            adj_outputs=[qfrc_gc.grad], adjoint=True)

  # 2. route the kinematic-intermediate adjoints to qpos -- the EXACT composition rne_qpos_vjp uses:
  #    cdof screw-commutator + (cdof moving-COM folded into the subtree seed via _build_ceff) + xipos FK.
  res_dof = wp.zeros((nworld, nv), dtype=float)
  wp.launch(_collision_adjoint._cdof_qpos_vjp, dim=(nworld, nv),
            inputs=[m.dof_parentid, m.dof_jntid, m.jnt_type, m.jnt_dofadr, d.cdof, adj_cdof, nv],
            outputs=[res_dof])
  ceff = wp.empty((nworld, nbody), dtype=wp.vec3)  # adj_subtree + the cdof moving-COM fold
  wp.launch(_collision_adjoint._build_ceff, dim=(nworld, nbody),
            inputs=[m.body_rootid, m.dof_bodyid, d.cdof, adj_cdof, adj_subtree, nv], outputs=[ceff])
  wp.launch(_collision_adjoint._subtree_com_qpos_vjp, dim=(nworld, nv),
            inputs=[m.body_parentid, m.body_rootid, m.dof_bodyid, m.body_isdofancestor,
                    m.body_mass, m.body_subtreemass, d.subtree_com, d.cdof, d.xipos, ceff, nbody],
            outputs=[res_dof])
  adj_ximat = wp.zeros((nworld, nbody), dtype=wp.mat33)  # gravcomp force is orientation-independent
  wp.launch(_cinert_pose_dof_vjp, dim=(nworld, nv),
            inputs=[m.body_parentid, m.body_rootid, m.dof_bodyid, m.body_isdofancestor,
                    d.subtree_com, d.cdof, d.xipos, d.ximat, adj_xipos, adj_ximat, nbody],
            outputs=[res_dof])
  wp.launch(_collision_adjoint._dof_to_qpos, dim=(nworld, m.njnt),
            inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof], outputs=[res_qpos])
  return res_qpos


# ============================================================================================
# AD-RNE: analytic backward of smooth.rne (MJPLAN_ADRNE §9-10), now wired into step_backward as the
# production ∂qpos path. The RNE-PROPER reverse produces adj_qvel / adj_qacc and the intermediate seeds
# adj_{cdof, cdof_dot, cinert, cvel}; the kinematic reverse below chains those seeds to qpos. FD-of-rne is
# retained only as an explicitly selected validation oracle in adjoint.py.
#
# smooth.rne is compiled enable_backward=False (smooth.py:41), so tape.backward cannot reverse it
# and we do NOT want to flip that global (it would regen every forward kernel + risk forward byte-
# identity). Instead we RECONSTRUCT the reverse in THIS backward-enabled module (§9.1):
#   - AUTO-DIFF the body-local NONLINEAR leaf (_cfrc: f = I·a + v ×* (I·v)) by re-calling the SAME
#     math.inert_vec / math.motion_cross_force @wp.func leaves from a backward-enabled kernel
#     (_rne_cfrc_recompute) -- Warp generates THEIR adjoint in this module's codegen. No hand-
#     derived inertia math; and because the leaf is loop-free + alias-free its Warp reverse is
#     EXACT (no stale-intermediate bias to compound over the BPTT horizon).
#   - MANUAL VJP the two LINEAR tree reductions (_cacc_branch forward sweep, _cfrc_backward child->
#     parent sum). Their forward kernels have dynamic loops (Warp does NOT replay them in reverse ->
#     stale intermediates) AND in-place same-array aliasing (cfrc_int is both in and out; cacc reads
#     its own output) -- the read-then-overwrite footgun -- so a naive reverse would be WRONG. Both
#     are LINEAR, so the VJP is just the transposed sum, written directly (enable_backward=False ->
#     the dynamic ancestry walks are a manual VJP, no tape replay). §9.2-alt unique-writer scheduling
#     (one thread per target body/dof, O(nbody·depth) -- the production per-depth form is step 4).
#
# Reverse order mirrors smooth.rne with seed adj_qfrc_bias = λ (§10.1):
#   λ --_qfrc_bias^T--> {adj_cdof, adj_F}                          (K1; F = d.cfrc_int, accumulated)
#     --_cfrc_backward^T--> adj_f  (adj_f[d] = Σ_{b ancestor-or-self of d} adj_F[b])   (K2)
#     --_cfrc^T (autodiff)--> {adj_cinert, adj_cacc, adj_cvel}                          (K3)
#     --_cacc_branch^T--> {adj_qvel, adj_qacc, adj_cdof_dot, adj_cdof +=}               (K4)
# Reads the converged d.{cdof,cdof_dot,cinert,cvel,cfrc_int,qvel,qacc} as the linearization point.
# ============================================================================================

_SV_ZERO = wp.constant(wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


@wp.kernel(enable_backward=False)
def _rne_qfrcbias_cdof_vjp(
  dof_bodyid: wp.array[int],
  cfrc_int: wp.array2d[wp.spatial_vector],  # accumulated F = d.cfrc_int
  lam: wp.array2d[float],  # seed adj_qfrc_bias
  adj_cdof: wp.array2d[wp.spatial_vector],  # OUT (+=): the τ_i = cdof_i·F_body(i) part
):
  """∂(λᵀqfrc_bias)/∂cdof_i = λ_i · F_body(i). One thread per dof (unique writer). Accumulates into
  adj_cdof, which _rne_cacc_dof_vjp (K4) also adds the cacc qacc-term into."""
  w, i = wp.tid()
  b = dof_bodyid[i]
  adj_cdof[w, i] = adj_cdof[w, i] + lam[w, i] * cfrc_int[w, b]


@wp.kernel(enable_backward=False)
def _rne_qfrcbias_force_vjp(
  body_dofadr: wp.array[int],
  body_dofnum: wp.array[int],
  cdof: wp.array2d[wp.spatial_vector],
  lam: wp.array2d[float],  # seed adj_qfrc_bias
  adj_force: wp.array2d[wp.spatial_vector],  # OUT: adj on the accumulated body force F_b
):
  """∂(λᵀqfrc_bias)/∂F_b = Σ_{i: body(i)=b} λ_i · cdof_i (a body's dofs are contiguous). One thread
  per body (unique writer); dofnum<=6 dynamic loop is fine (manual VJP, no tape replay)."""
  w, b = wp.tid()
  dofadr = body_dofadr[b]
  dofnum = body_dofnum[b]
  acc = _SV_ZERO
  for k in range(dofnum):
    i = dofadr + k
    acc = acc + lam[w, i] * cdof[w, i]
  adj_force[w, b] = acc


@wp.kernel(enable_backward=False)
def _rne_cfrc_tree_vjp(
  body_parentid: wp.array[int],
  adj_force: wp.array2d[wp.spatial_vector],  # adj on accumulated F_b (from K1)
  adj_f: wp.array2d[wp.spatial_vector],  # OUT: adj on the LOCAL force f_d (pre-accumulation)
):
  """Transpose of the forward leaf->root sum F_b = f_b + Σ_children F_c. Since
  F_b = Σ_{d ∈ subtree(b)} f_d, the reverse is adj_f[d] = Σ_{b: d ∈ subtree(b)} adj_F[b]
  = Σ_{b ancestor-or-self of d} adj_F[b]. One thread per body d walks its ancestry to the root
  (unique writer; the dynamic walk is a manual VJP). Body 0 (world) has no dofs so adj_F[0]=0."""
  w, d = wp.tid()
  acc = _SV_ZERO
  b = d
  while b > 0:
    acc = acc + adj_force[w, b]
    b = body_parentid[b]
  adj_f[w, d] = acc


@wp.kernel
def _rne_cfrc_recompute(
  cinert: wp.array2d[vec10f],
  cacc: wp.array2d[wp.spatial_vector],
  cvel: wp.array2d[wp.spatial_vector],
  cfrc_local: wp.array2d[wp.spatial_vector],  # OUT: f_b = I·a + v ×* (I·v)
):
  """EXACT reconstruction of smooth._cfrc's local inertial force (the body-local NONLINEAR leaf),
  reusing the SAME math.inert_vec / math.motion_cross_force @wp.func leaves. Backward-enabled
  (module default) so a manual adjoint launch auto-diffs it: seed cfrc_local's cotangent with adj_f,
  read adj_{cinert,cacc,cvel}. Loop-free + alias-free -> Warp reverse is exact."""
  w, b = wp.tid()
  frc = _SV_ZERO
  if b != 0:  # world body has no inertial force (mirrors smooth._cfrc's bodyid==0 branch)
    ci = cinert[w, b]
    cv = cvel[w, b]
    frc = _math.inert_vec(ci, cacc[w, b]) + _math.motion_cross_force(cv, _math.inert_vec(ci, cv))
  cfrc_local[w, b] = frc


@wp.kernel(enable_backward=False)
def _rne_cacc_subtree_sum(
  body_parentid: wp.array[int],
  adj_cacc: wp.array2d[wp.spatial_vector],
  nbody: int,
  subtree_adj_cacc: wp.array2d[wp.spatial_vector],  # OUT: Σ_{d ∈ subtree(b)} adj_cacc[d]
):
  """Subtree-sum of adj_cacc. The forward sweep cacc_b = cacc_parent + Σ(dof terms) makes
  cacc_b = Σ_{j: body(j) ancestor-or-self of b} (dof terms), so ∂cacc_b/∂(dof at body B) is nonzero
  iff B is an ancestor-or-self of b. The dof-term reverse (K4) therefore needs, per body B, the sum
  of adj_cacc over B's subtree. One thread per body B (unique writer); O(nbody·depth) correctness-
  oracle form (the production per-depth leaf->root accumulation is step 4). World body B=0 is unused
  (real dofs have body>=1) and resolves to 0."""
  w, target = wp.tid()
  acc = _SV_ZERO
  for d in range(nbody):
    p = d  # is `target` an ancestor-or-self of body d?
    while p > 0:
      if p == target:
        acc = acc + adj_cacc[w, d]
        break
      p = body_parentid[p]
  subtree_adj_cacc[w, target] = acc


@wp.kernel(enable_backward=False)
def _rne_cacc_dof_vjp(
  dof_bodyid: wp.array[int],
  cdof: wp.array2d[wp.spatial_vector],
  cdof_dot: wp.array2d[wp.spatial_vector],
  qvel: wp.array2d[float],
  qacc: wp.array2d[float],
  subtree_adj_cacc: wp.array2d[wp.spatial_vector],  # Σ adj_cacc over body(j)'s subtree (from K4a)
  flg_acc: bool,
  adj_qvel: wp.array2d[float],  # OUT
  adj_qacc: wp.array2d[float],  # OUT
  adj_cdof_dot: wp.array2d[wp.spatial_vector],  # OUT
  adj_cdof: wp.array2d[wp.spatial_vector],  # OUT (+=): the cdof_i·qacc_i term (K1 added the τ term)
):
  """Transpose of the forward sweep's dof contributions cacc_b += cdof_dot_i·qvel_i + cdof_i·qacc_i.
  With a = Σ_{b ∈ subtree(body(i))} adj_cacc[b] (K4a): adj_qvel_i = a·cdof_dot_i, adj_cdof_dot_i =
  qvel_i·a, and (flg_acc) adj_qacc_i = a·cdof_i, adj_cdof_i += qacc_i·a. One thread per dof (unique
  writer)."""
  w, i = wp.tid()
  a = subtree_adj_cacc[w, dof_bodyid[i]]
  adj_qvel[w, i] = wp.dot(a, cdof_dot[w, i])
  adj_cdof_dot[w, i] = qvel[w, i] * a
  if flg_acc:
    adj_qacc[w, i] = wp.dot(a, cdof[w, i])
    adj_cdof[w, i] = adj_cdof[w, i] + qacc[w, i] * a


def rne_backward(m: Model, d: Data, lam: wp.array2d, flg_acc: bool = True):
  """Analytic VJP of smooth.rne: given the cotangent ``lam`` on qfrc_bias, return the adjoints of the
  RNE inputs. ``d`` must hold the converged forward intermediates (run kinematics -> com_pos ->
  com_vel -> rne first). Returns a dict of wp.arrays:
    qvel, qacc            -- direct input adjoints (the §14.3 MuJoCo-C gate targets)
    cdof, cdof_dot, cinert, cvel -- intermediate seeds (chained to qpos by the kinematic reverse,
                                    steps 5-7)
  Backward-only; reads d as the linearization point and never mutates it."""
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

  # K1: λ -> {adj_cdof (τ part), adj_force}
  wp.launch(_rne_qfrcbias_cdof_vjp, dim=(nworld, nv),
            inputs=[m.dof_bodyid, d.cfrc_int, lam], outputs=[adj_cdof])
  wp.launch(_rne_qfrcbias_force_vjp, dim=(nworld, nbody),
            inputs=[m.body_dofadr, m.body_dofnum, d.cdof, lam], outputs=[adj_force])

  # K2: adj_force -> adj_f (transpose the child->parent accumulation)
  wp.launch(_rne_cfrc_tree_vjp, dim=(nworld, nbody),
            inputs=[m.body_parentid, adj_force], outputs=[adj_f])

  # K3: adj_f -> {adj_cinert, adj_cacc, adj_cvel} via AUTODIFF of the reconstructed local leaf
  cfrc_local = wp.zeros((nworld, nbody), dtype=SV)
  adj_cinert = wp.zeros((nworld, nbody), dtype=vec10f)
  adj_cacc = wp.zeros((nworld, nbody), dtype=SV)
  adj_cvel = wp.zeros((nworld, nbody), dtype=SV)
  cfrc_inputs = [d.cinert, d.cacc, d.cvel]
  wp.launch(_rne_cfrc_recompute, dim=(nworld, nbody), inputs=cfrc_inputs, outputs=[cfrc_local])
  wp.launch(_rne_cfrc_recompute, dim=(nworld, nbody), inputs=cfrc_inputs, outputs=[cfrc_local],
            adj_inputs=[adj_cinert, adj_cacc, adj_cvel], adj_outputs=[adj_f], adjoint=True)

  # K4: adj_cacc -> {adj_qvel, adj_qacc, adj_cdof_dot, adj_cdof +=} (transpose the cacc forward sweep)
  subtree_adj_cacc = wp.zeros((nworld, nbody), dtype=SV)
  wp.launch(_rne_cacc_subtree_sum, dim=(nworld, nbody),
            inputs=[m.body_parentid, adj_cacc, nbody], outputs=[subtree_adj_cacc])
  wp.launch(_rne_cacc_dof_vjp, dim=(nworld, nv),
            inputs=[m.dof_bodyid, d.cdof, d.cdof_dot, d.qvel, d.qacc, subtree_adj_cacc, flg_acc],
            outputs=[adj_qvel, adj_qacc, adj_cdof_dot, adj_cdof])

  return {"qvel": adj_qvel, "qacc": adj_qacc, "cdof": adj_cdof,
          "cdof_dot": adj_cdof_dot, "cinert": adj_cinert, "cvel": adj_cvel}


# --------------------------------------------------------------------------------------------
# com_vel reverse (MJPLAN_ADRNE §10.1A): BOUND-FREE manual VJP of smooth.com_vel -- the Coriolis path
# qvel -> cvel/cdof_dot. rne_backward leaves the seeds adj_cvel (per body) / adj_cdof_dot (per dof);
# this maps them to com_vel's contributions to adj_qvel and adj_cdof. Unlike the rne-proper LINEAR
# reductions, cdof_dot_i = motion_cross(U_i, cdof_i) is BILINEAR (U_i = the cvel snapshot just before
# dof i). §10.1A is a fully MANUAL transpose: every body/joint/dof/ancestry loop runs in an
# enable_backward=False kernel, so the bounds may be DYNAMIC -- NO _CV_MAX_* truncation (the old
# auto-diff-reconstruction needed static bounds; this supersedes it). The motion_cross VJP reuses the
# loop-free force-cross leaf (§10.1A.2):
#     h_i = (∂X/∂U)ᵀ G_i = motion_cross_force(cdof_i, G_i)    (cotangent on the snapshot velocity)
#     k_i = (∂X/∂cdof)ᵀ G_i = -motion_cross_force(U_i, G_i)   (direct cotangent on cdof_i)
# with G_i = adj_cdof_dot_i. Four stages (§10.1A.6): CV1 body-local snapshot + cross VJP (h, k per dof;
# H_b = Σ h on the body); CV2 same-body prefix reverse (event scan §10.1A.5: walk the body's joints in
# REVERSE, running T accumulates the suffix h's = T_j, scatter S_j·T_j / ν_j·T_j and add direct k_j);
# CV3 W_B = Σ_{subtree(B)} adj_cvel + Σ_{strict_subtree(B)} H (the augmented-seed subtree sum,
# §10.1A.4 unique-writer form); CV4 global scatter += W_B. Result: ν̄_j += S_j·(W_B+T_j),
# S̄_j += ν_j·(W_B+T_j) + k_j. CV1 reads the STORED parent cvel (no ancestry recompute -> no depth
# bound); depth is handled by CV3's dynamic subtree walk. Mirrors smooth._comvel_branch's event order:
# FREE rows 0-2 translations (cdof_dot=0 -> h=k=0), 3-5 rotations (snapshot = post-translation cvel);
# BALL: all 3 from the pre-ball cvel. The fused reverse-depth scan (§10.1A.7) is the O(nbody+nv)
# production optimization; this four-stage form is the correctness-first transpose. NOTE for wiring
# (§10.1A): step_backward already uses deriv_rne_vel for the smooth adj_qvel -- pick ONE of the two
# RNE-qvel paths when wiring this in, or the Coriolis term double-counts.
# --------------------------------------------------------------------------------------------


@wp.func
def _cv_scatter(w: int, j: int, t: wp.spatial_vector,
                cdof: wp.array2d[wp.spatial_vector], qvel: wp.array2d[float],
                adj_qvel: wp.array2d[float], adj_cdof: wp.array2d[wp.spatial_vector]):
  """Same-body scatter of a running prefix cotangent t (§10.1A.5): ν̄_j += S_j·t, S̄_j += ν_j·t.
  One body owns all its dofs -> the writes are unique (no atomics)."""
  adj_qvel[w, j] = adj_qvel[w, j] + wp.dot(cdof[w, j], t)
  adj_cdof[w, j] = adj_cdof[w, j] + qvel[w, j] * t


@wp.kernel(enable_backward=False)
def _comvel_vjp_local(
  body_parentid: wp.array[int],
  body_jntadr: wp.array[int],
  body_jntnum: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  cvel: wp.array2d[wp.spatial_vector],  # stored forward cvel (linearization point)
  cdof: wp.array2d[wp.spatial_vector],
  qvel: wp.array2d[float],
  adj_cdof_dot: wp.array2d[wp.spatial_vector],  # G seed
  h_out: wp.array2d[wp.spatial_vector],  # OUT per dof: cotangent on the snapshot velocity
  k_out: wp.array2d[wp.spatial_vector],  # OUT per dof: direct cotangent on cdof
  H_out: wp.array2d[wp.spatial_vector],  # OUT per body: Σ h on this body
):
  """CV1 (§10.1A.6): replay the body's joints forward from the STORED parent cvel snapshot (no ancestry
  recompute -> no depth bound), computing per nonconstant cdof_dot row h_i = mcf(S_i, G_i),
  k_i = -mcf(U_i, G_i). Uses jnt_dofadr per joint (never infer the start from a running index)."""
  w, b = wp.tid()
  zero = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  if b == 0:  # world body: no joints / snapshot
    H_out[w, 0] = zero
    return
  Hb = zero
  u = cvel[w, body_parentid[b]]  # parent snapshot (cvel[0]=0 for a tree root)
  jntadr = body_jntadr[b]
  jntnum = body_jntnum[b]
  for jj in range(jntnum):
    jt = jnt_type[jntadr + jj]
    d = jnt_dofadr[jntadr + jj]
    if jt == _FREE:  # ADD translations -> SNAPSHOT rotations (post-translation) -> ADD rotations
      u = u + cdof[w, d + 0] * qvel[w, d + 0] + cdof[w, d + 1] * qvel[w, d + 1] + cdof[w, d + 2] * qvel[w, d + 2]
      g3 = adj_cdof_dot[w, d + 3]
      g4 = adj_cdof_dot[w, d + 4]
      g5 = adj_cdof_dot[w, d + 5]
      h3 = _math.motion_cross_force(cdof[w, d + 3], g3)
      h4 = _math.motion_cross_force(cdof[w, d + 4], g4)
      h5 = _math.motion_cross_force(cdof[w, d + 5], g5)
      h_out[w, d + 0] = zero
      h_out[w, d + 1] = zero
      h_out[w, d + 2] = zero
      h_out[w, d + 3] = h3
      h_out[w, d + 4] = h4
      h_out[w, d + 5] = h5
      k_out[w, d + 0] = zero
      k_out[w, d + 1] = zero
      k_out[w, d + 2] = zero
      k_out[w, d + 3] = (-1.0) * _math.motion_cross_force(u, g3)
      k_out[w, d + 4] = (-1.0) * _math.motion_cross_force(u, g4)
      k_out[w, d + 5] = (-1.0) * _math.motion_cross_force(u, g5)
      u = u + cdof[w, d + 3] * qvel[w, d + 3] + cdof[w, d + 4] * qvel[w, d + 4] + cdof[w, d + 5] * qvel[w, d + 5]
      Hb = Hb + h3 + h4 + h5
    elif jt == _BALL:  # SNAPSHOT all 3 axes from the pre-ball cvel, then ADD
      g0 = adj_cdof_dot[w, d + 0]
      g1 = adj_cdof_dot[w, d + 1]
      g2 = adj_cdof_dot[w, d + 2]
      h0 = _math.motion_cross_force(cdof[w, d + 0], g0)
      h1 = _math.motion_cross_force(cdof[w, d + 1], g1)
      h2 = _math.motion_cross_force(cdof[w, d + 2], g2)
      h_out[w, d + 0] = h0
      h_out[w, d + 1] = h1
      h_out[w, d + 2] = h2
      k_out[w, d + 0] = (-1.0) * _math.motion_cross_force(u, g0)
      k_out[w, d + 1] = (-1.0) * _math.motion_cross_force(u, g1)
      k_out[w, d + 2] = (-1.0) * _math.motion_cross_force(u, g2)
      u = u + cdof[w, d + 0] * qvel[w, d + 0] + cdof[w, d + 1] * qvel[w, d + 1] + cdof[w, d + 2] * qvel[w, d + 2]
      Hb = Hb + h0 + h1 + h2
    else:  # HINGE / SLIDE: SNAPSHOT from the pre-joint cvel, then ADD
      g0 = adj_cdof_dot[w, d]
      h0 = _math.motion_cross_force(cdof[w, d], g0)
      h_out[w, d] = h0
      k_out[w, d] = (-1.0) * _math.motion_cross_force(u, g0)
      u = u + cdof[w, d] * qvel[w, d]
      Hb = Hb + h0
  H_out[w, b] = Hb


@wp.kernel(enable_backward=False)
def _comvel_vjp_samebody(
  body_jntadr: wp.array[int],
  body_jntnum: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  cdof: wp.array2d[wp.spatial_vector],
  qvel: wp.array2d[float],
  h_in: wp.array2d[wp.spatial_vector],
  k_in: wp.array2d[wp.spatial_vector],
  adj_qvel: wp.array2d[float],  # OUT (+=): same-body T_j part
  adj_cdof: wp.array2d[wp.spatial_vector],  # OUT (+=): same-body T_j part + direct k_j
):
  """CV2 (§10.1A.5): same-body event scan. Walk the body's joints in REVERSE; the running T is the
  suffix-of-same-body h's (= T_j when scattering dof j). The scatter precedes T += h for that joint's
  group, so no axis receives its own same-joint snapshot cotangent (FREE rotational / BALL self-block).
  Body-local writes are unique (no atomics)."""
  w, b = wp.tid()
  t = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  jntadr = body_jntadr[b]
  jntnum = body_jntnum[b]
  for jr in range(jntnum):
    jj = jntnum - 1 - jr  # reverse joint order
    jt = jnt_type[jntadr + jj]
    d = jnt_dofadr[jntadr + jj]
    if jt == _FREE:  # reverse: scatter rot ADDs, T += rot h, k rot, scatter trans ADDs (now see rot h)
      _cv_scatter(w, d + 5, t, cdof, qvel, adj_qvel, adj_cdof)
      _cv_scatter(w, d + 4, t, cdof, qvel, adj_qvel, adj_cdof)
      _cv_scatter(w, d + 3, t, cdof, qvel, adj_qvel, adj_cdof)
      t = t + h_in[w, d + 5] + h_in[w, d + 4] + h_in[w, d + 3]
      adj_cdof[w, d + 5] = adj_cdof[w, d + 5] + k_in[w, d + 5]
      adj_cdof[w, d + 4] = adj_cdof[w, d + 4] + k_in[w, d + 4]
      adj_cdof[w, d + 3] = adj_cdof[w, d + 3] + k_in[w, d + 3]
      _cv_scatter(w, d + 2, t, cdof, qvel, adj_qvel, adj_cdof)
      _cv_scatter(w, d + 1, t, cdof, qvel, adj_qvel, adj_cdof)
      _cv_scatter(w, d + 0, t, cdof, qvel, adj_qvel, adj_cdof)
    elif jt == _BALL:  # scatter all 3 with the same T, then T += the 3 same-ball h
      _cv_scatter(w, d + 2, t, cdof, qvel, adj_qvel, adj_cdof)
      _cv_scatter(w, d + 1, t, cdof, qvel, adj_qvel, adj_cdof)
      _cv_scatter(w, d + 0, t, cdof, qvel, adj_qvel, adj_cdof)
      t = t + h_in[w, d + 2] + h_in[w, d + 1] + h_in[w, d + 0]
      adj_cdof[w, d + 0] = adj_cdof[w, d + 0] + k_in[w, d + 0]
      adj_cdof[w, d + 1] = adj_cdof[w, d + 1] + k_in[w, d + 1]
      adj_cdof[w, d + 2] = adj_cdof[w, d + 2] + k_in[w, d + 2]
    else:  # HINGE / SLIDE
      _cv_scatter(w, d, t, cdof, qvel, adj_qvel, adj_cdof)
      t = t + h_in[w, d]
      adj_cdof[w, d] = adj_cdof[w, d] + k_in[w, d]


@wp.kernel(enable_backward=False)
def _comvel_subtree_W(
  body_parentid: wp.array[int],
  adj_cvel: wp.array2d[wp.spatial_vector],  # A seed (per body); NOT mutated
  H_in: wp.array2d[wp.spatial_vector],
  nbody: int,
  W_out: wp.array2d[wp.spatial_vector],
):
  """CV3 (§10.1A.4): W_B = Σ_{x∈subtree(B)} adj_cvel_x + Σ_{x∈strict_subtree(B)} H_x. One thread per
  body (unique writer); dynamic ancestry test (enable_backward=False, like _rne_cacc_subtree_sum), so
  no depth bound. H_B is excluded (B's own snapshots start from V_p(B), carried by B's parent's W)."""
  w, target = wp.tid()
  acc = adj_cvel[w, target]  # B itself (subtree includes self for the A term)
  for x in range(1, nbody):
    if x != target:
      p = body_parentid[x]  # is `target` a STRICT ancestor of x?
      while p > 0:
        if p == target:
          acc = acc + adj_cvel[w, x] + H_in[w, x]
          break
        p = body_parentid[p]
  W_out[w, target] = acc


@wp.kernel(enable_backward=False)
def _comvel_scatter_W(
  dof_bodyid: wp.array[int],
  cdof: wp.array2d[wp.spatial_vector],
  qvel: wp.array2d[float],
  W_in: wp.array2d[wp.spatial_vector],
  adj_qvel: wp.array2d[float],  # OUT (+=): W_B part
  adj_cdof: wp.array2d[wp.spatial_vector],  # OUT (+=): W_B part
):
  """CV4 (§10.1A.6): global cvel scatter += W_body(i): ν̄_i += S_i·W, S̄_i += ν_i·W. One thread per dof
  (unique writer); accumulates onto CV2's same-body contributions."""
  w, i = wp.tid()
  wv = W_in[w, dof_bodyid[i]]
  adj_qvel[w, i] = adj_qvel[w, i] + wp.dot(cdof[w, i], wv)
  adj_cdof[w, i] = adj_cdof[w, i] + qvel[w, i] * wv


def comvel_backward(m: Model, d: Data, adj_cvel: wp.array2d, adj_cdof_dot: wp.array2d):
  """Bound-free manual VJP of smooth.com_vel (MJPLAN_ADRNE §10.1A): given cotangents adj_cvel (per
  body) and adj_cdof_dot (per dof), return com_vel's contributions to {adj_qvel, adj_cdof} (the
  Coriolis path). All loops are enable_backward=False -> dynamic bounds, NO _CV_MAX_* truncation. Reads
  d.{cvel, cdof, qvel} as the linearization point; never mutates d or the adj_cvel seed."""
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
  wp.launch(_comvel_vjp_local, dim=(nworld, nbody),
            inputs=[m.body_parentid, m.body_jntadr, m.body_jntnum, m.jnt_type, m.jnt_dofadr,
                    d.cvel, d.cdof, d.qvel, adj_cdof_dot], outputs=[h, kk, Hbody])
  # CV2: same-body prefix reverse (event scan) -> adj_qvel/adj_cdof (the T_j part + direct k_j).
  wp.launch(_comvel_vjp_samebody, dim=(nworld, nbody),
            inputs=[m.body_jntadr, m.body_jntnum, m.jnt_type, m.jnt_dofadr, d.cdof, d.qvel, h, kk],
            outputs=[adj_qvel, adj_cdof])
  # CV3: W_B = Σ_subtree(B) adj_cvel + Σ_strict_subtree(B) H  (the augmented-seed subtree sum).
  wp.launch(_comvel_subtree_W, dim=(nworld, nbody),
            inputs=[m.body_parentid, adj_cvel, Hbody, nbody], outputs=[W])
  # CV4: global cvel scatter -> += W_B contributions.
  wp.launch(_comvel_scatter_W, dim=(nworld, nv),
            inputs=[m.dof_bodyid, d.cdof, d.qvel, W], outputs=[adj_qvel, adj_cdof])
  return {"qvel": adj_qvel, "cdof": adj_cdof}


# --------------------------------------------------------------------------------------------
# cinert reverse (MJPLAN_ADRNE steps 5-7): the last NEW kinematic ∂qpos leaf for the full RNE bias
# qpos column. cinert_b (the body's spatial inertia in the c-frame, smooth._cinert) depends on qpos
# through ximat_b (inertial-frame orientation), xipos_b (inertial position), and subtree_com[root].
# AUTO-DIFF the body-local _cinert leaf (loop-free, alias-free -> exact) to get adj_{xipos, ximat,
# subtree_com}, then chain xipos/ximat -> qpos via the analytic body Jacobian support.jac_dof (the
# _geom_pose_dof_vjp pattern: jacp·adj_xipos + jacr·τ), and route adj_subtree_com through the DONE
# collision_adjoint._subtree_com_qpos_vjp. body_mass/body_inertia adjoints are exposed by the same
# leaf -> inertial sys-id falls out later (step 8). NOTE: cvel/cdof_dot have NO direct qpos path -- it
# flows through cdof (handled by collision_adjoint._cdof_qpos_vjp on the TOTAL adj_cdof), so cinert is
# the only kinematic leaf this stage adds.
# --------------------------------------------------------------------------------------------


@wp.kernel
def _cinert_recompute(
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  xipos: wp.array2d[wp.vec3],
  ximat: wp.array2d[wp.mat33],
  subtree_com: wp.array2d[wp.vec3],
  cinert_out: wp.array2d[vec10f],
):
  """EXACT reconstruction of smooth._cinert (body-local, no loop) for a manual adjoint launch ->
  adj_{xipos, ximat, subtree_com} (+ body_mass/inertia for sys-id). Static (literal) component indices
  are autodiff-safe; mirrors the forward arithmetic statement-for-statement."""
  w, b = wp.tid()
  mat = ximat[w, b]
  inert = body_inertia[w % body_inertia.shape[0], b]
  mass = body_mass[w % body_mass.shape[0], b]
  dif = xipos[w, b] - subtree_com[w, body_rootid[b]]
  # Build the vec10 as SCALARS + a single vec10f(...) constructor -- NOT `res = vec10f(); res[i] = ...;
  # res[i] = res[i] + ...`. Warp 1.14's reverse double-counts the in-place vec-component update pattern
  # (verified 2× via _scratch/debug/probe_cinert_rot.py). The forward smooth._cinert keeps the
  # component-write form (enable_backward=False -> value-only). res_rot[i,j] = Σ_k mat[i,k]·inert[k]·
  # mat[j,k] (explicit, not mat@diag@matᵀ); minus the parallel-axis mass·[dif]×[dif]× ; first moment
  # mass·dif; mass.
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


@wp.kernel(enable_backward=False)
def _cinert_pose_dof_vjp(
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  subtree_com: wp.array2d[wp.vec3],
  cdof: wp.array2d[wp.spatial_vector],
  xipos: wp.array2d[wp.vec3],
  ximat: wp.array2d[wp.mat33],
  adj_xipos: wp.array2d[wp.vec3],
  adj_ximat: wp.array2d[wp.mat33],
  nbody: int,
  res_dof: wp.array2d[float],  # OUT (+=): per-dof TANGENT gradient
):
  """Chain adj_{xipos, ximat} (per body) to the per-dof TANGENT gradient via support.jac_dof (the
  _geom_pose_dof_vjp pattern, body-indexed): ∂c/∂(dof k) = Σ_b jacp·adj_xipos_b + jacr·τ_b, where
  τ_b = Σ_col ximat_b[:,col] × adj_ximat_b[:,col] reduces the orientation adjoint to a world axis."""
  w, k = wp.tid()
  acc = float(0.0)
  for b in range(1, nbody):
    if body_isdofancestor[b, k] == 0:  # dof k does not move body b -> jac = 0
      continue
    jacp, jacr = _support.jac_dof(
      body_parentid, body_rootid, dof_bodyid, body_isdofancestor,
      subtree_com, cdof, xipos[w, b], b, k, w,
    )
    xm = ximat[w, b]
    rgm = adj_ximat[w, b]
    tau = (
      wp.cross(wp.vec3(xm[0, 0], xm[1, 0], xm[2, 0]), wp.vec3(rgm[0, 0], rgm[1, 0], rgm[2, 0]))
      + wp.cross(wp.vec3(xm[0, 1], xm[1, 1], xm[2, 1]), wp.vec3(rgm[0, 1], rgm[1, 1], rgm[2, 1]))
      + wp.cross(wp.vec3(xm[0, 2], xm[1, 2], xm[2, 2]), wp.vec3(rgm[0, 2], rgm[1, 2], rgm[2, 2]))
    )
    acc += wp.dot(jacp, adj_xipos[w, b]) + wp.dot(jacr, tau)
  res_dof[w, k] += acc


def cinert_qpos_vjp(m: Model, d: Data, adj_cinert: wp.array2d, res_dof: wp.array2d):
  """VJP of smooth._cinert: given the cotangent adj_cinert (per body, vec10), accumulate the xipos/
  ximat -> qpos contribution into res_dof (per-dof TANGENT), and RETURN the subtree_com cotangent
  adj_subtree_com (per body) for the caller to route through collision_adjoint._subtree_com_qpos_vjp
  (combined with the cdof moving-COM term). Reads d as the linearization point; never mutates it."""
  nworld = d.qpos.shape[0]
  nv = m.nv
  nbody = m.nbody
  cinert_rec = wp.zeros((nworld, nbody), dtype=vec10f)
  adj_xipos = wp.zeros((nworld, nbody), dtype=wp.vec3)
  adj_ximat = wp.zeros((nworld, nbody), dtype=wp.mat33)
  adj_subtree = wp.zeros((nworld, nbody), dtype=wp.vec3)
  cin = [m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com]
  wp.launch(_cinert_recompute, dim=(nworld, nbody), inputs=cin, outputs=[cinert_rec])
  wp.launch(_cinert_recompute, dim=(nworld, nbody), inputs=cin, outputs=[cinert_rec],
            adj_inputs=[None, None, None, adj_xipos, adj_ximat, adj_subtree],
            adj_outputs=[adj_cinert], adjoint=True)
  wp.launch(_cinert_pose_dof_vjp, dim=(nworld, nv),
            inputs=[m.body_parentid, m.body_rootid, m.dof_bodyid, m.body_isdofancestor,
                    d.subtree_com, d.cdof, d.xipos, d.ximat, adj_xipos, adj_ximat, nbody],
            outputs=[res_dof])
  return adj_subtree


def inertia_param_vjp(m: Model, d: Data, lam: wp.array2d):
  """Inertial system-id (body_mass / body_inertia): accumulate -(∂(λᵀqfrc_bias)/∂{body_mass,body_inertia})
  into m.body_mass.grad / m.body_inertia.grad (the IFT minus, matching adjoint.smooth_param_backward's
  armature/damping sign). body_mass/body_inertia enter the dynamics ONLY through cinert (the c-frame spatial
  inertia: the M·qacc CRB contraction, the RNE Coriolis term, AND gravity -- via the cacc-world seed -- all
  flow through it), so adj_cinert from the rne-proper reverse (rne_backward, flg_acc=True -- the irreducible
  dynamic-depth tree transpose, a manual VJP with NO unroll/DOF dependency) routed through the SOURCE-AD
  cinert leaf (_cinert_recompute: loop-free + alias-free -> exact, body_mass/body_inertia are direct inputs)
  yields the inertia derivative with NO hand-written inertia VJP -- exactly the autodiff path. ``d`` holds the
  converged smooth intermediates (kinematics->com_pos->com_vel->rne(flg_acc=True) at the linearization);
  read-only. No-op unless body_mass / body_inertia are requires_grad. FD-gated vs float64 mj_rne."""
  nworld = d.qpos.shape[0]
  nbody = m.nbody
  # adj_cinert = ∂(λᵀqfrc_bias)/∂cinert (captures M·qacc + Coriolis + gravity, qacc frozen).
  adj = rne_backward(m, d, lam, flg_acc=True)
  adj_cinert = adj["cinert"]
  # SOURCE-AD the cinert leaf wrt body_mass/body_inertia (the autodiff inertia derivative; no hand VJP).
  cinert_rec = wp.zeros((nworld, nbody), dtype=vec10f, requires_grad=True)
  cin = [m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com]
  wp.launch(_cinert_recompute, dim=(nworld, nbody), inputs=cin, outputs=[cinert_rec])
  adj_mass = wp.zeros_like(m.body_mass) if m.body_mass.requires_grad else None
  adj_inertia = wp.zeros_like(m.body_inertia) if m.body_inertia.requires_grad else None
  wp.launch(_cinert_recompute, dim=(nworld, nbody), inputs=cin, outputs=[cinert_rec],
            adj_inputs=[None, adj_mass, adj_inertia, None, None, None],
            adj_outputs=[adj_cinert], adjoint=True)
  # IFT minus into the shared (trajectory-accumulated) model param grads.
  if adj_mass is not None:
    wp.launch(_adjoint._accum_neg, dim=adj_mass.shape, inputs=[adj_mass], outputs=[m.body_mass.grad])
  if adj_inertia is not None:
    wp.launch(_adjoint._accum_neg_vec3, dim=adj_inertia.shape, inputs=[adj_inertia], outputs=[m.body_inertia.grad])


@wp.kernel(enable_backward=False)
def _add_spatial(a: wp.array2d[wp.spatial_vector], b: wp.array2d[wp.spatial_vector],
                 out: wp.array2d[wp.spatial_vector]):
  w, i = wp.tid()
  out[w, i] = a[w, i] + b[w, i]


def rne_qpos_vjp(m: Model, d: Data, lam: wp.array2d, flg_acc: bool = True):
  """Full RNE-bias ∂qpos: given the cotangent ``lam`` on qfrc_bias, return res_qpos =
  ∂(λᵀqfrc_bias)/∂qpos (nq-indexed). Composes the RNE-proper reverse (rne_backward) + the Coriolis
  com_vel reverse (comvel_backward) with the kinematic ∂qpos VJPs: cinert (this module) + the DONE
  collision_adjoint cdof screw-commutator (_cdof_qpos_vjp), moving-COM fold (_build_ceff), mass-
  weighted subtree-COM Jacobian (_subtree_com_qpos_vjp), and the free/ball quaternion lift
  (_dof_to_qpos). cvel/cdof_dot have no direct qpos path -> their qpos dependence is the TOTAL adj_cdof
  (rne-proper + com_vel) routed through _cdof_qpos_vjp. Reads converged d; never mutates it. This is
  the analytic replacement for FD-of-rne's ∂qpos used by step_backward's production path."""
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
  wp.launch(_collision_adjoint._cdof_qpos_vjp, dim=(nworld, nv),
            inputs=[m.dof_parentid, m.dof_jntid, m.jnt_type, m.jnt_dofadr, d.cdof, total_cdof, nv],
            outputs=[res_dof])
  ceff = wp.empty((nworld, nbody), dtype=wp.vec3)  # cinert subtree seed + cdof moving-COM fold
  wp.launch(_collision_adjoint._build_ceff, dim=(nworld, nbody),
            inputs=[m.body_rootid, m.dof_bodyid, d.cdof, total_cdof, adj_subtree, nv], outputs=[ceff])
  wp.launch(_collision_adjoint._subtree_com_qpos_vjp, dim=(nworld, nv),
            inputs=[m.body_parentid, m.body_rootid, m.dof_bodyid, m.body_isdofancestor,
                    m.body_mass, m.body_subtreemass, d.subtree_com, d.cdof, d.xipos, ceff, nbody],
            outputs=[res_dof])
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  wp.launch(_collision_adjoint._dof_to_qpos, dim=(nworld, m.njnt),
            inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof], outputs=[res_qpos])
  return res_qpos


# ----------------------------------------------------------------------------
# 5b. ORACLES (NOT the production smooth-force boundary). LOCAL smooth-force ∂qpos: joint SPRINGS
#     (slide/hinge ONLY) + AFFINE joint ACTUATORS. Per MJPLAN_ADRNE §0/§15 the production smooth ∂qpos is
#     the reduced backward-only kinematic replay (smooth_adjoint.py: kinematics->com_pos->com_vel->
#     rne(flg_acc=True) reversed once, source-AD leaves + manual tree transposes). These two kernels are
#     kept ONLY as independently FD-gated A/B oracles for that replay; they are UNWIRED, INCOMPLETE
#     (HINGE/SLIDE springs only; affine joint-transmission actuators only), and carry NO runtime FD
#     fallback. Springs: a reconstruct+autodiff leaf (Warp gets the sign). Actuators: a manual VJP over the
#     frozen moment (the affine LENGTH coefficient only; the orthogonal velocity coefficient gainprm[2]/
#     biasprm[2] is the implicit-actuator channel owned by deriv_smooth_vel / step_backward §4).
# ----------------------------------------------------------------------------
@wp.kernel(module="unique", enable_backward=True)
def _residual_spring_local(
  opt_disableflags: int,
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_stiffness: wp.array2d[float],
  jnt_stiffnesspoly: wp.array2d[wp.vec2],
  qpos_spring: wp.array2d[float],
  qpos_in: wp.array2d[float],  # [grad: ∂qpos] linearization point (d.qpos); its adjoint is routed to res_qpos
  r_out: wp.array2d[float],
):
  """Slide/hinge joint-SPRING contribution to ``r_smooth = ... - qfrc_passive``:
  ``r[dof] = -qfrc_spring[dof] = fdif·_poly_force(stiffness, spoly, fdif, 0)``, ``fdif = qpos[qadr] -
  qpos_spring[qadr]`` (the SLIDE/HINGE branch of ``passive._spring_damper_dof_passive``, reusing the
  forward ``@wp.func`` so linear AND polynomial springs come out and Warp gets the sign). AD'd w.r.t.
  ``qpos_in`` with ``adj_r = +λ`` drops ``(∂r/∂qpos)ᵀλ`` into res_qpos (nq-indexed; SLIDE/HINGE have
  qpos coord == dof so no quaternion lift). FREE/BALL quaternion springs are out of scope (the analytic-
  qpos gate excludes free/ball stiffness); the viscous ``-damper·qvel`` has no ∂qpos so it is omitted."""
  w, jntid = wp.tid()
  jt = jnt_type[jntid]
  if jt == _FREE or jt == _BALL:
    return
  stiffness = jnt_stiffness[w % jnt_stiffness.shape[0], jntid]
  spoly = jnt_stiffnesspoly[w % jnt_stiffnesspoly.shape[0], jntid]
  has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and (opt_disableflags & _SPRING) == 0
  if not has_stiffness:
    return
  qadr = jnt_qposadr[jntid]
  dofid = jnt_dofadr[jntid]
  fdif = qpos_in[w, qadr] - qpos_spring[w % qpos_spring.shape[0], qadr]
  r_out[w, dofid] = fdif * _util_misc._poly_force(stiffness, spoly, fdif, 0)


@wp.kernel(enable_backward=False)
def _actuator_qpos_vjp(
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_ctrllimited: wp.array[bool],
  actuator_forcelimited: wp.array[bool],
  actuator_gainprm: wp.array2d[vec10f],
  actuator_biasprm: wp.array2d[vec10f],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  actuator_forcerange: wp.array2d[wp.vec2],
  moment_rownnz: wp.array2d[int],
  moment_rowadr: wp.array2d[int],
  moment_colind: wp.array2d[int],
  actuator_moment: wp.array2d[float],  # frozen joint-transmission moment (∂length/∂qvel; CONST in qpos)
  actuator_force: wp.array2d[float],  # frozen (clamped) force -- the forcerange-saturation gate
  ctrl_in: wp.array2d[float],
  lam: wp.array2d[float],
  dsbl_clampctrl: int,
  res_dof: wp.array2d[float],  # OUT: += -(∂qfrc_actuator/∂δθ)ᵀλ (per-dof TANGENT; lifted by _dof_to_qpos)
):
  """LOCAL ∂qpos of AFFINE joint actuators (manual VJP). ``qfrc_actuator[dof] = Σ_a moment[a,dof]·force_a``
  with ``force_a = gain·ctrl_act + bias`` affine in ``length_a = moment[a]·qpos`` (JOINT transmission ->
  moment CONST in qpos). So ``∂(λᵀ(-qfrc_actuator))/∂δθ_j = -Σ_a ml_a·dfdl_a·moment[a,j]``, where
  ``ml_a = moment[a]·λ`` (= ``_smooth_ctrl_vjp``'s mtl) and ``dfdl_a = ∂force_a/∂length_a =
  [gain AFFINE]·gainprm[1]·ctrl_act + [bias AFFINE]·biasprm[1]``, ZEROED when the force is forcerange-
  saturated (mirrors ``derivative._qderiv_actuator_passive_vel``'s gate exactly -- the velocity-channel
  analogue). ``ctrl_act`` = clamped ctrl (na==0 gated -> stateless). The velocity coeffs gainprm[2]/
  biasprm[2] are the ORTHOGONAL implicit-actuator channel (step_backward §4 ∂qvel) -> no double-count.
  ``enable_backward=False``: the moment-row reduction is a manual VJP, so its dynamic loop is safe."""
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
    f = actuator_force[w, actid]
    if f <= fr[0] or f >= fr[1]:
      return
  rownnz = moment_rownnz[w, actid]
  rowadr = moment_rowadr[w, actid]
  ml = float(0.0)
  for i in range(rownnz):
    sid = rowadr + i
    ml += actuator_moment[w, sid] * lam[w, moment_colind[w, sid]]
  c = -ml * dfdl  # MINUS: r_smooth includes -qfrc_actuator
  for i in range(rownnz):
    sid = rowadr + i
    wp.atomic_add(res_dof[w], moment_colind[w, sid], c * actuator_moment[w, sid])


def smooth_local_qpos_vjp(m: Model, d: Data, lam: wp.array2d):
  """ORACLE (NOT wired into step_backward; see the §5b banner + MJPLAN_ADRNE §0/§15). LOCAL smooth-force
  ∂qpos (HINGE/SLIDE joint SPRINGS + AFFINE joint-transmission ACTUATORS): res_qpos (nq) =
  ``(∂(λᵀ(-qfrc_spring - qfrc_actuator))/∂qpos)``. Composed with ``rne_qpos_vjp`` (the rigid-body bias) it
  is an FD-gated A/B reference for the production reduced kinematic replay. ``d`` is the linearization
  scratch with the smooth forces recomputed (transmission/passive/actuation at the input qpos/qvel/ctrl);
  never mutated. INCOMPLETE: no FREE/BALL springs, no tendon/SITE/BODY/slider-crank or JOINTINPARENT
  transmission, no muscle/DC-motor/user force law -- those are completion requirements of the replay."""
  nworld = d.qpos.shape[0]
  nv = m.nv
  nq = d.qpos.shape[1]
  res_qpos = wp.zeros((nworld, nq), dtype=float)

  # joint springs (slide/hinge): autodiff leaf, qpos input-adjoint routed into res_qpos (accumulate)
  r_sp = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
  sp_in = [m.opt.disableflags, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_stiffness,
           m.jnt_stiffnesspoly, m.qpos_spring, d.qpos]
  wp.launch(_residual_spring_local, dim=(nworld, m.njnt), inputs=sp_in, outputs=[r_sp])
  wp.launch(_adjoint._copy_cols, dim=(nworld, nv), inputs=[lam], outputs=[r_sp.grad])  # seed adj_r = +λ
  wp.launch(_residual_spring_local, dim=(nworld, m.njnt), inputs=sp_in, outputs=[r_sp],
            adj_inputs=[None, None, None, None, None, None, None, res_qpos],  # qpos adjoint -> res_qpos
            adj_outputs=[r_sp.grad], adjoint=True)

  # affine joint actuators: manual VJP -> per-dof TANGENT res_dof -> lift to res_qpos (accumulate).
  # DisableBit.ACTUATION suppresses the forward actuator force, so the VJP must be suppressed too.
  if m.nu > 0 and (int(m.opt.disableflags) & _ACTUATION) == 0:
    res_dof = wp.zeros((nworld, nv), dtype=float)
    wp.launch(_actuator_qpos_vjp, dim=(nworld, m.nu),
              inputs=[m.actuator_gaintype, m.actuator_biastype, m.actuator_ctrllimited,
                      m.actuator_forcelimited, m.actuator_gainprm, m.actuator_biasprm, m.actuator_ctrlrange,
                      m.actuator_forcerange, d.moment_rownnz, d.moment_rowadr, d.moment_colind,
                      d.actuator_moment, d.actuator_force, d.ctrl, lam, int(m.opt.disableflags) & _CLAMPCTRL],
              outputs=[res_dof])
    wp.launch(_collision_adjoint._dof_to_qpos, dim=(nworld, m.njnt),
              inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof], outputs=[res_qpos])
  return res_qpos
