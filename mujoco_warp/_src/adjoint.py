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
"""Analytic (IFT) backward for step(): integrator, solver, and residual VJPs."""

import dataclasses

import warp as wp

from mujoco_warp._src import collision_adjoint
from mujoco_warp._src import constraint
from mujoco_warp._src import constraint_adjoint
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import smooth
from mujoco_warp._src import smooth_adjoint
from mujoco_warp._src import solver
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5
from mujoco_warp._src.types import vec10f

# --- Safe sqrt for the contact adjoint (active whenever this adjoint module is imported) --------
# Swaps wp.sqrt -> safe_sqrt: forward byte-identical, custom grad 0 at x<=0. Needed because
# wp.sqrt's reverse is 0.5/sqrt(x) -> inf at 0 and Warp differentiates BOTH wp.where arms, so any
# sqrt(0) in an untaken branch still yields 0*inf = NaN. Import-time GLOBAL monkeypatch.
_wp_sqrt = wp.sqrt


@wp.func
def safe_sqrt(x: float):
  return _wp_sqrt(x)


@wp.func_grad(safe_sqrt)
def _adj_safe_sqrt_f32(x: wp.float32, adj_ret: wp.float32):
  if x > wp.float32(0.0):
    wp.adjoint[x] += adj_ret / (wp.float32(2.0) * _wp_sqrt(x))


@wp.func
def safe_sqrt(x: wp.float64):  # concrete overload (same name) so float64 sqrt still works
  return _wp_sqrt(x)


@wp.func_grad(safe_sqrt)
def _adj_safe_sqrt_f64(x: wp.float64, adj_ret: wp.float64):
  if x > wp.float64(0.0):
    wp.adjoint[x] += adj_ret / (wp.float64(2.0) * _wp_sqrt(x))


wp.sqrt = safe_sqrt

# NOTE: adjoint.py kernels are differentiable -- do NOT set enable_backward=False here.
# DEDUP & adjoint scope: the residual reuses the forward's OWN @wp.func physics (_contact_kbimp,
# _eval_elliptic_middle) as one source of truth. constraint.py/solver.py enable_backward=False
# affects only THEIR kernels; a @wp.func's adjoint codegens in the CALLING (this) module -- correct.

_FREE = int(types.JointType.FREE.value)
_BALL = int(types.JointType.BALL.value)
_SATISFIED = int(types.ConstraintState.SATISFIED.value)
_LINEARNEG = int(types.ConstraintState.LINEARNEG.value)  # saturated friction, force = +frictionloss
_LINEARPOS = int(types.ConstraintState.LINEARPOS.value)  # saturated friction, force = -frictionloss
_FRICTION_DOF = int(types.ConstraintType.FRICTION_DOF.value)
_EQUALITY = int(types.ConstraintType.EQUALITY.value)
_LIMIT_JOINT = int(types.ConstraintType.LIMIT_JOINT.value)
_ELLIPTIC = int(types.ConeType.ELLIPTIC.value)
_PYRAMIDAL = int(types.ConeType.PYRAMIDAL.value)
_EULER = int(types.IntegratorType.EULER.value)
_IMPLICITFAST = int(types.IntegratorType.IMPLICITFAST.value)
_EULERDAMP = int(types.DisableBit.EULERDAMP.value)
_DAMPER = int(types.DisableBit.DAMPER.value)

# Static unroll bound for the dense NON-CONTACT constraint residual (_residual_constraint, nv<=16).
# Warp does NOT replay dynamic loops in the backward: a nonlinear reduction in a dynamic loop reads
# stale (~0) intermediates in the adjoint and can blow up (1/T at T~0 -> ~1e18); static unrolled
# bounds avoid that. step_backward RAISES for nv>_MAX_NV on the dense path (no silent truncation).
_MAX_NV = 16

# Production default True: the contact residual VJP always takes the SPARSE contract-first
# branch (nv-general, reads no efc.J). False enables structural routing that mirrors the
# forward's jacobian storage -- the dense monolithic kernel for small dense models
# (nv <= _MAX_NV, not m.is_sparse, no geom_friction sys-id) -- a candidate small-scene fast
# path, A/B-pinned by test_sparse_vs_dense_oracle_match; benchmark before flipping the default.
_FORCE_SPARSE_CONTACT = True


# ----------------------------------------------------------------------------
# 1. Integrator (semi-implicit Euler, out-of-place). Backward-enabled: we launch its adjoint.
#    Matches forward._advance for the no-damp case (the bounce uses Euler/implicitfast no-damp).
# ----------------------------------------------------------------------------
@wp.kernel
def _advance_state(
  # Model:
  opt_timestep: wp.array[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # Data out:
  qpos_out: wp.array2d[float],
  qvel_out: wp.array2d[float],
):
  worldid, jntid = wp.tid()
  dt = opt_timestep[worldid % opt_timestep.shape[0]]
  jt = jnt_type[jntid]
  qadr = jnt_qposadr[jntid]
  dadr = jnt_dofadr[jntid]

  # semi-implicit: update qvel, then integrate qpos with it, via the shared forward funcs.
  # next_velocity returns the value (held as a local) so qpos uses it without reading qvel_out back.
  qvel_lin = wp.vec3(0.0, 0.0, 0.0)
  qvel_ang = wp.vec3(0.0, 0.0, 0.0)
  if jt == _FREE:
    vlx = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 0, 1.0)
    vly = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 1, 1.0)
    vlz = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 2, 1.0)
    vax = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 3, 1.0)
    vay = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 4, 1.0)
    vaz = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 5, 1.0)
    qvel_out[worldid, dadr + 0] = vlx
    qvel_out[worldid, dadr + 1] = vly
    qvel_out[worldid, dadr + 2] = vlz
    qvel_out[worldid, dadr + 3] = vax
    qvel_out[worldid, dadr + 4] = vay
    qvel_out[worldid, dadr + 5] = vaz
    qvel_lin = wp.vec3(vlx, vly, vlz)
    qvel_ang = wp.vec3(vax, vay, vaz)
  elif jt == _BALL:
    vx = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 0, 1.0)
    vy = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 1, 1.0)
    vz = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 2, 1.0)
    qvel_out[worldid, dadr + 0] = vx
    qvel_out[worldid, dadr + 1] = vy
    qvel_out[worldid, dadr + 2] = vz
    qvel_ang = wp.vec3(vx, vy, vz)
  else:  # HINGE / SLIDE
    v = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr, 1.0)
    qvel_out[worldid, dadr] = v
    qvel_lin = wp.vec3(v, 0.0, 0.0)

  forward.next_position(qpos_in, jt, qadr, dt, worldid, qvel_lin, qvel_ang, qpos_out)


# ----------------------------------------------------------------------------
# 2. IFT helper.
# ----------------------------------------------------------------------------
@wp.kernel
def _load_rhs(
  # Model:
  nv: int,
  # In:
  adj_qacc: wp.array2d[float],
  # Out:
  grad_out: wp.array2d[float],
):
  """Write adj_qacc into ctx.grad[:, :nv] (zero the padding) as the RHS of H lam = adj_qacc."""
  worldid, i = wp.tid()
  if i < nv:
    grad_out[worldid, i] = adj_qacc[worldid, i]
  else:
    grad_out[worldid, i] = 0.0


# ----------------------------------------------------------------------------
# 3. Contact residual VJP leaves (the elliptic/pyramidal cone force law) live in
#    constraint_adjoint.py; adjoint.contact_residual_backward orchestrates them.
# ----------------------------------------------------------------------------


@wp.kernel
def _copy_cols(src: wp.array2d[float], dst_out: wp.array2d[float]):
  """dst_out[w,i] = src[w,i] over dst_out's columns (seed r.grad = lam from ctx.Mgrad[:, :nv])."""
  w, i = wp.tid()
  dst_out[w, i] = src[w, i]


@wp.kernel
def _sub_cols(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float]):
  """Write out = a - b: integrator-direct adjoint minus residual-VJP scatter (dr/dtheta)^T lam."""
  w, i = wp.tid()
  out[w, i] = a[w, i] - b[w, i]


@wp.kernel
def _accum_cols(a: wp.array2d[float], out: wp.array2d[float]):
  """Compute out += a (accumulate the RNE-bias dqpos into the buffer _sub_cols subtracts)."""
  w, i = wp.tid()
  out[w, i] = out[w, i] + a[w, i]


# ----------------------------------------------------------------------------
# 4. Smooth residual r_smooth = M*qacc - qfrc_smooth. The VJP is linear in r = r_smooth +
#    r_contact, so it shares the IFT lam. dqvel/dctrl are analytic here; dqpos lives in
#    smooth_qpos_backward (stage 5).
# ----------------------------------------------------------------------------
@wp.kernel
def _smooth_qvel_vjp(
  # Model:
  opt_timestep: wp.array[float],
  qD_fullm_i: wp.array[int],  # D-structure (full square) row index of entry e
  qD_fullm_j: wp.array[int],  # D-structure column index of entry e
  # Data in:
  qLU_in: wp.array2d[float],  # assembled so (M_D - qLU_in)/dt = dqfrc_smooth/dqvel (deriv_smooth_vel + rne, no subtract)
  # In:
  M_D: wp.array2d[float],  # mass matrix in D-structure (M mapped via mapM2D)
  lam: wp.array2d[float],  # IFT multiplier lam (cols 0:nv valid)
  adj_qvel: wp.array2d[float],  # += G^T lam  (G = dqfrc_smooth/dqvel = (M_D - qLU_in)/dt)
):
  """adj_qvel += G^T lam, G = dqfrc_smooth/dqvel = (M_D - qLU_in)/dt (full-square D-structure)."""
  w, e = wp.tid()
  dt = opt_timestep[w % opt_timestep.shape[0]]
  g = (M_D[w, e] - qLU_in[w, e]) / dt
  wp.atomic_add(adj_qvel[w], qD_fullm_j[e], g * lam[w, qD_fullm_i[e]])


@wp.kernel
def _smooth_ctrl_vjp(
  # Model:
  actuator_gainprm: wp.array2d[vec10f],  # gain = prm[0] for FIXED gaintype (motor/position)
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],  # sparse actuator_moment_in (frozen)
  # In:
  lam: wp.array2d[float],
  # Out:
  adj_ctrl_out: wp.array2d[float],  # = (dqfrc_actuator/dctrl)^T lam = gain*(moment^T lam)
):
  """adj_ctrl_out = gain*(moment^T lam) for FIXED gaintype (AFFINE-gain not yet handled)."""
  w, actid = wp.tid()
  rownnz = moment_rownnz_in[w, actid]
  rowadr = moment_rowadr_in[w, actid]
  mtl = float(0.0)
  for i in range(rownnz):
    sparseid = rowadr + i
    mtl += actuator_moment_in[w, sparseid] * lam[w, moment_colind_in[w, sparseid]]
  gain = actuator_gainprm[w % actuator_gainprm.shape[0], actid][0]
  adj_ctrl_out[w, actid] = gain * mtl


@wp.kernel(module="unique", enable_backward=True)
def _residual_smooth_local(
  # Model:
  dof_armature: wp.array2d[float],  # [grad target] reflected rotor inertia (added to the M diagonal)
  dof_damping: wp.array2d[float],  # [grad target] viscous joint damping
  # Data in:
  qvel_in: wp.array2d[float],  # frozen input velocity (the EULER passive-force linearization point)
  qacc_in: wp.array2d[float],  # frozen converged acceleration
  # Out:
  r_out: wp.array2d[float],
):
  """r[i] = armature[i]*qacc[i] + damping[i]*qvel[i]: the armature/damping terms of r_smooth."""
  w, i = wp.tid()
  arm = dof_armature[w % dof_armature.shape[0], i]
  dmp = dof_damping[w % dof_damping.shape[0], i]
  r_out[w, i] = arm * qacc_in[w, i] + dmp * qvel_in[w, i]


@wp.kernel(module="unique", enable_backward=True)
def _residual_constraint(
  # Model:
  nv: int,
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  jnt_solref: wp.array2d[wp.vec2],
  jnt_solimp: wp.array2d[vec5],
  dof_solref: wp.array2d[wp.vec2],
  dof_solimp: wp.array2d[vec5],
  dof_frictionloss: wp.array2d[float],  # [grad target] joint Coulomb friction
  eq_solref: wp.array2d[wp.vec2],
  eq_solimp: wp.array2d[vec5],
  # Data in:
  nefc_in: wp.array[int],
  qvel_in: wp.array2d[float],  # [grad: dqvel state] input velocity
  qacc_in: wp.array2d[float],  # frozen converged accel
  efc_type_in: wp.array2d[int],  # ConstraintType per row
  efc_id_in: wp.array2d[int],  # source id (dofid / eqid / jntid)
  efc_J_in: wp.array3d[float],  # frozen constraint Jacobian (dense)
  efc_pos_in: wp.array2d[float],  # frozen efc position (= pos_aref + margin)
  efc_margin_in: wp.array2d[float],  # frozen efc margin
  efc_D_in: wp.array2d[float],  # frozen constraint mass
  efc_state_in: wp.array2d[int],  # frozen active set
  # In:
  dpos_in: wp.array2d[float],  # [grad: TANGENT dqpos] dof-space perturbation seed (zeros); lifted via _dof_to_qpos
  # Out:
  r_out: wp.array2d[float],
):
  """NON-CONTACT constraint residual r += -J^T f (equality/limit/friction) over frozen efc rows.

  Must mirror solver._eval_constraint; piecewise-linear force keeps the dynamic row loop AD-safe.
  """
  w = wp.tid()
  dt = opt_timestep[w % opt_timestep.shape[0]]
  for row in range(nefc_in[w]):  # dynamic loop over this world's efc rows (PIECEWISE-LINEAR -> AD-safe)
    ty = efc_type_in[w, row]
    if ty != _EQUALITY and ty != _LIMIT_JOINT and ty != _FRICTION_DOF:
      continue  # contact rows -> _contact_gather/phi/scatter; LIMIT_TENDON / FRICTION_TENDON -> TODO
    st = efc_state_in[w, row]
    if st == _SATISFIED:
      continue
    cid = efc_id_in[w, row]  # source id: dofid (FRICTION_DOF) / eqid (EQUALITY) / jntid (LIMIT_JOINT)
    if ty == _FRICTION_DOF and st == _LINEARNEG:  # saturated friction: force = +frictionloss
      f = dof_frictionloss[w % dof_frictionloss.shape[0], cid]
    elif ty == _FRICTION_DOF and st == _LINEARPOS:  # saturated friction: force = -frictionloss
      f = -dof_frictionloss[w % dof_frictionloss.shape[0], cid]
    else:  # QUADRATIC: equality (bilateral) / active limit / stuck friction -> force = -D*jaref
      pos_aref0 = efc_pos_in[w, row] - efc_margin_in[w, row]  # frozen signed violation (impedance ref)
      if ty == _FRICTION_DOF:  # per-type solref/solimp -> (k, b, imp) at the frozen pos
        sr = dof_solref[w % dof_solref.shape[0], cid]
        si = dof_solimp[w % dof_solimp.shape[0], cid]
        kbi = constraint._contact_kbimp(opt_disableflags, dt, sr, si, pos_aref0)
      elif ty == _EQUALITY:
        sr = eq_solref[w % eq_solref.shape[0], cid]
        si = eq_solimp[w % eq_solimp.shape[0], cid]
        kbi = constraint._contact_kbimp(opt_disableflags, dt, sr, si, pos_aref0)
      else:  # _LIMIT_JOINT (slide/hinge: scalar J; ball: the 3-dof angular -axis J)
        sr = jnt_solref[w % jnt_solref.shape[0], cid]
        si = jnt_solimp[w % jnt_solimp.shape[0], cid]
        kbi = constraint._contact_kbimp(opt_disableflags, dt, sr, si, pos_aref0)
      Jqa = float(0.0)
      Jqv = float(0.0)
      pos_d = float(0.0)
      for i in range(_MAX_NV):
        if i < nv:
          jj = efc_J_in[w, row, i]
          Jqa += jj * qacc_in[w, i]
          Jqv += jj * qvel_in[w, i]
          pos_d += jj * dpos_in[w, i]  # dpos_aref/d(delta_theta) = efc.J (TANGENT; lifted to qpos by _dof_to_qpos)
      jaref = Jqa + kbi[0] * kbi[2] * (pos_aref0 + pos_d) + kbi[1] * Jqv
      f = -efc_D_in[w, row] * jaref
    for i in range(_MAX_NV):
      if i < nv:
        r_out[w, i] += -f * efc_J_in[w, row, i]


# NON-CONTACT constraint residual VJP -- SPARSE/CSR + nv-general sibling of the sparse contact
# path (gather -> anchored row leaf -> J^T scatter). REUSES ctx.Jaref and efc.vel as the FROZEN
# value anchors (only Z = J*lam is freshly reduced); dense-vs-CSR follows m.is_sparse, and the
# routing vs the dense kernel is structural (no global on/off flag).
def _residual_constraint_sparse(
  m: Model, d_out: Data, ctx_Jaref: wp.array, lam: wp.array, res_qvel: wp.array, res_dof: wp.array
):
  """SPARSE/CSR nv-general non-contact VJP: gather Z = J*lam -> anchored row leaf -> J^T scatter.

  Param adjoints accumulate in-place into m.<param>.grad (the IFT minus is applied here).
  """
  nworld = d_out.qpos.shape[0]
  njmax = d_out.efc.type.shape[1]  # row capacity (njmax); efc.J row dim is njmax_pad >= njmax
  nv = m.nv
  Z = wp.zeros((nworld, njmax), dtype=float, requires_grad=True)
  invw = wp.zeros((nworld, njmax), dtype=float)
  gather = constraint_adjoint._constraint_gather(m.is_sparse)
  wp.launch(
    gather,
    dim=(nworld, njmax),
    inputs=[
      nv,
      m.jnt_dofadr,
      m.dof_invweight0,
      d_out.nefc,
      d_out.efc.type,
      d_out.efc.id,
      d_out.efc.J_rownnz,
      d_out.efc.J_rowadr,
      d_out.efc.J_colind,
      d_out.efc.J,
      d_out.efc.state,
      lam,
    ],
    outputs=[Z, invw],
  )
  for _arr in (d_out.efc.pos, d_out.efc.vel):
    _arr.requires_grad = True  # so the leaf adjoint accumulates Pbar / Vbar into the res buffers
  phi = wp.zeros((nworld, njmax), dtype=float, requires_grad=True)
  res_pos = wp.zeros((nworld, njmax), dtype=float)  # Pbar per row
  res_vel = wp.zeros((nworld, njmax), dtype=float)  # Vbar per row
  # (input, input-adjoint) pairs -- res_pos/res_vel are Pbar/Vbar, Z.grad is Zbar; per-class *_rb
  # param buffers exist only when the param requires_grad and feed the post-launch IFT-minus.
  res_dof_frictionloss = wp.zeros_like(m.dof_frictionloss) if m.dof_frictionloss.requires_grad else None
  res_dof_solref = wp.zeros_like(m.dof_solref) if m.dof_solref.requires_grad else None
  res_dof_solimp = wp.zeros_like(m.dof_solimp) if m.dof_solimp.requires_grad else None
  res_eq_solref = wp.zeros_like(m.eq_solref) if m.eq_solref.requires_grad else None
  res_eq_solimp = wp.zeros_like(m.eq_solimp) if m.eq_solimp.requires_grad else None
  res_jnt_solref = wp.zeros_like(m.jnt_solref) if m.jnt_solref.requires_grad else None
  res_jnt_solimp = wp.zeros_like(m.jnt_solimp) if m.jnt_solimp.requires_grad else None
  leaf = [
    (m.opt.timestep, None),
    (m.opt.disableflags, None),
    (m.jnt_solref, res_jnt_solref),
    (m.jnt_solimp, res_jnt_solimp),
    (m.dof_solref, res_dof_solref),
    (m.dof_solimp, res_dof_solimp),
    (m.dof_frictionloss, res_dof_frictionloss),
    (m.eq_solref, res_eq_solref),
    (m.eq_solimp, res_eq_solimp),
    (d_out.nefc, None),
    (d_out.efc.type, None),
    (d_out.efc.id, None),
    (d_out.efc.pos, res_pos),  # dphi/defc.pos = Pbar
    (d_out.efc.margin, None),
    (d_out.efc.D, None),
    (d_out.efc.vel, res_vel),  # dphi/defc.vel = Vbar
    (d_out.efc.aref, None),
    (d_out.efc.force, None),
    (d_out.efc.state, None),
    (Z, Z.grad),  # Zbar (consumed only by the dJ/dq topology reverse, steps 4-8)
    (invw, None),
    (ctx_Jaref, None),
  ]
  leaf_in = [a for a, _ in leaf]
  wp.launch(constraint_adjoint._constraint_row_phi, dim=(nworld, njmax), inputs=leaf_in, outputs=[phi])
  phi.grad.fill_(1.0)  # seed adj_phi = +1 (phi already folds in lam; inactive rows returned early -> no-op reverse)
  wp.launch(
    constraint_adjoint._constraint_row_phi,
    dim=(nworld, njmax),
    inputs=leaf_in,
    outputs=[phi],
    adj_inputs=[g for _, g in leaf],
    adj_outputs=[phi.grad],
    adjoint=True,
  )
  scatter = constraint_adjoint._constraint_scatter(m.is_sparse)
  wp.launch(
    scatter,
    dim=(nworld, njmax),
    inputs=[
      nv,
      d_out.nefc,
      d_out.efc.type,
      d_out.efc.J_rownnz,
      d_out.efc.J_rowadr,
      d_out.efc.J_colind,
      d_out.efc.J,
      d_out.efc.state,
      res_pos,
      res_vel,
    ],
    outputs=[res_qvel, res_dof],
  )
  if res_dof_frictionloss is not None:  # IFT minus into the shared param grads
    wp.launch(smooth_adjoint._accum_neg, dim=res_dof_frictionloss.shape, inputs=[res_dof_frictionloss], outputs=[m.dof_frictionloss.grad])
  for _pg, _rb in ((m.dof_solref, res_dof_solref), (m.eq_solref, res_eq_solref), (m.jnt_solref, res_jnt_solref)):
    if _rb is not None:
      wp.launch(smooth_adjoint._accum_neg_vec2, dim=_rb.shape, inputs=[_rb], outputs=[_pg.grad])
  for _pg, _rb in ((m.dof_solimp, res_dof_solimp), (m.eq_solimp, res_eq_solimp), (m.jnt_solimp, res_jnt_solimp)):
    if _rb is not None:
      wp.launch(smooth_adjoint._accum_neg_vec5, dim=_rb.shape, inputs=[_rb], outputs=[_pg.grad])


# ----------------------------------------------------------------------------
# 5. Smooth dqpos: the analytic reduced-RNE replay in smooth_adjoint. The FD-of-rne oracle
#    lives in adjoint_test_util (test-only).
# ----------------------------------------------------------------------------
def _smooth_linearization(m: Model, d: Data, d_out: Data) -> Data:
  """Shallow view of d_out at the linearization (d.qpos/d.qvel/d_out.qacc); never mutates d_out."""
  s = dataclasses.replace(
    d_out,
    qpos=d.qpos,
    qvel=d.qvel,
    ctrl=d.ctrl,
    act=d.act,
    cacc=wp.empty_like(d_out.cacc),
    cfrc_int=wp.empty_like(d_out.cfrc_int),
    qfrc_bias=wp.empty_like(d_out.qfrc_bias),
  )
  smooth.rne(m, s, flg_acc=True)
  smooth.tendon_bias(m, s, s.qfrc_bias)
  return s


def _model_has_unsupported_noncontact_rows(m: Model) -> bool:
  """Host-side sync-free predicate: does the model carry a non-contact row class without a VJP?"""
  unsupported_equality = m.eq_connect_adr.size > 0 or m.eq_wld_adr.size > 0
  return unsupported_equality or int(m.ntendon) > 0 or int(m.nflex) > 0


def _assert_dense_unroll(which: str):
  """Guard _MAX_NV <= wp.config.max_unroll: a dynamic loop's reverse is stale (wrong gradient)."""
  assert _MAX_NV <= wp.config.max_unroll, (
    f"adjoint: the dense {which} backward needs _MAX_NV={_MAX_NV} <= wp.config.max_unroll="
    f"{wp.config.max_unroll}; otherwise range(_MAX_NV) stays a DYNAMIC loop and its reverse is stale "
    f"(silent wrong gradient). Raise wp.config.max_unroll to >= {_MAX_NV} (or use the sparse path)."
  )


# ----------------------------------------------------------------------------
# The analytic step backward (registered with forward.py).
# ----------------------------------------------------------------------------
def _assert_step_supported(m: Model):
  """Gate for step_backward / forward_backward_ift: host-side Model metadata only (sync-free)."""
  nv = m.nv
  # The smooth and CONTACT residuals are general over articulations (all 4 joint types; per-dof
  # contact J via support.jac_dof). The narrowphase dcpos/dqpos replay covers only the geom pairs
  # supported by collision_adjoint.
  if m.nflex != 0:
    raise NotImplementedError("adjoint.step_backward does not support flex contacts")
  if m.opt.cone != _ELLIPTIC and m.opt.cone != _PYRAMIDAL:
    raise NotImplementedError("adjoint.step_backward supports only elliptic/pyramidal cones")
  # NON-contact constraint rows: SUPPORTED classes (dof-friction, slide/hinge limit, JOINT
  # equality, BALL limit) run at ANY nv (sparse/CSR for nv>_MAX_NV, dense otherwise). UNSUPPORTED
  # classes (CONNECT/WELD equality, tendon rows, flex) have no landed VJP -> RAISE structurally,
  # never a silent wrong gradient; the predicate is fixed model structure (sync-free).
  if _model_has_unsupported_noncontact_rows(m):
    raise NotImplementedError(
      "adjoint.step_backward: the model carries a non-contact constraint row class without a landed VJP "
      "(connect/weld equality, tendon, or flex). Supported: dof-friction, slide/hinge limit, JOINT equality, "
      "ball limit."
    )
  # condim is mirrored generically (1/3/4/6 -- the valid MuJoCo set; 2/5 cannot be loaded), so no
  # nmaxcondim gate is needed: rotational rows (dimid>=3) are handled in both cone branches.


@wp.kernel(module="unique", enable_backward=True)
def _dampingpoly_Qv_leaf(
  # Model:
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  # Data in:
  qvel_in: wp.array2d[float],
  # In:
  timestep: wp.array[float],
  a_u: wp.array2d[float],
  # Out:
  out: wp.array2d[float],
):
  """out[i] = dt*D_eff(v_i)*a_u[i], D_eff = d(poly damping force)/dv (a_u and y_int held fixed)."""
  w, i = wp.tid()
  dt = timestep[w % timestep.shape[0]]
  damping = dof_damping[w % dof_damping.shape[0], i]
  dpoly = dof_dampingpoly[w % dof_dampingpoly.shape[0], i]
  out[w, i] = dt * util_misc._poly_force_deriv(damping, dpoly, qvel_in[w, i], 1) * a_u[w, i]


def advance_backward(m: Model, d: Data, d_out: Data):
  """Integrator (advance) adjoint (stage 1) -- the VJP of forward.{euler,implicit,...}.

  Replays the backward-enabled _advance_state with adjoint=True (remapping the implicitfast /
  eulerdamp integration solves); returns (adj_qpos, adj_qvel, adj_qacc). adj_qacc seeds the IFT.
  """
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  # --- 1. integrator adjoint: adj(qpos',qvel') -> adj_qacc + integrator-direct adj(qpos,qvel) ---
  adj_qpos = wp.zeros((nworld, nq), dtype=float)
  adj_qvel = wp.zeros((nworld, nv), dtype=float)
  adj_qacc = wp.zeros((nworld, nv), dtype=float)

  # IMPLICITFAST advances with a_int = Q^-1 (M*a), Q = M - dt*d(qfrc_smooth)/d(qvel) -- not the
  # raw solver root. Rebuild a_int here (the quaternion-position adjoint needs it) and later map
  # adj(a_int) through the transpose solve; omitting the map makes the backward behave like
  # explicit Euler (unstable multiplier on a low-inertia damped hinge).
  implicit_deriv_flags = types.DisableBit.ACTUATION | types.DisableBit.SPRING | types.DisableBit.DAMPER
  implicitfast_deriv = int(m.opt.integrator) == _IMPLICITFAST and bool(~(m.opt.disableflags | ~implicit_deriv_flags))
  # EULERDAMP analog: forward.euler advances at a_u = (M+dt*D)^-1 M a_s, so reconstruct a_u and
  # REPLAY THE INTEGRATOR AT a_u (not the solver root a_s) -- FREE/BALL quaternion integration is
  # nonlinear in qvel', so replaying at a_s is WRONG for quaternions. The transpose remap
  # adj(a_s)=M(M+dt*D)^-1 adj(a_u) is applied after the replay (cached factor).
  eulerdamp = int(m.opt.integrator) == _EULER and (int(m.opt.disableflags) & (_EULERDAMP | _DAMPER)) == 0
  qacc_advance = d_out.qacc
  qLD_int = None
  qLDiagInv_int = None
  qLD_s = None
  qLDiagInv_s = None
  if implicitfast_deriv:
    # d_out owns the forward linearization intermediates, but its qvel has since been
    # overwritten by integration.  Alias the input velocity exactly as smooth_vel_backward does
    # before rebuilding Q.
    saved_qvel = d_out.qvel
    d_out.qvel = d.qvel
    Q_int = wp.empty(d_out.M.shape, dtype=float)
    derivative.deriv_smooth_vel(m, d_out, Q_int)
    d_out.qvel = saved_qvel
    qLD_int = wp.empty_like(d_out.qLD)
    qLDiagInv_int = wp.empty((nworld, nv), dtype=float)
    qacc_advance = wp.empty((nworld, nv), dtype=float)
    smooth.factor_solve_i(m, d_out, Q_int, qLD_int, qLDiagInv_int, qacc_advance, d_out.efc.Ma)
  if eulerdamp:
    # Reconstruct a_u = (M+dt*D)^-1 (M a_s) BEFORE the replay; CACHE the factor for the
    # post-replay transpose remap. The damping deriv is taken at the INPUT velocity d.qvel
    # (d_out.qvel was overwritten by integration); dt*D is added to M's diagonal exactly as
    # forward.euler.
    damp_deriv = wp.empty((nworld, nv), dtype=float)
    wp.launch(
      forward._compute_damping_deriv, dim=(nworld, nv), inputs=[m.dof_damping, m.dof_dampingpoly, d.qvel], outputs=[damp_deriv]
    )
    MOD = wp.clone(d_out.M)  # M + dt*D, in M's CSR layout
    wp.launch(
      forward._euler_damp_qfrc, dim=(nworld, nv), inputs=[m.opt.timestep, m.M_rownnz, m.M_rowadr, damp_deriv], outputs=[MOD]
    )
    Ma_s = wp.empty((nworld, nv), dtype=float)
    support.mul_m(m, d_out, Ma_s, d_out.qacc)
    qLD_s = wp.empty_like(d_out.qLD)
    qLDiagInv_s = wp.empty((nworld, nv), dtype=float)
    qacc_advance = wp.empty((nworld, nv), dtype=float)
    smooth.factor_solve_i(m, d_out, MOD, qLD_s, qLDiagInv_s, qacc_advance, Ma_s)  # a_u = (M+dt*D)^-1 M a_s

  qpos_s = wp.empty_like(d.qpos)
  qvel_s = wp.empty_like(d.qvel)
  int_inputs = [m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, d.qvel, qacc_advance]
  wp.launch(_advance_state, dim=(nworld, m.njnt), inputs=int_inputs, outputs=[qpos_s, qvel_s])
  wp.launch(
    _advance_state,
    dim=(nworld, m.njnt),
    inputs=int_inputs,
    outputs=[qpos_s, qvel_s],
    adj_inputs=[None, None, None, None, adj_qpos, adj_qvel, adj_qacc],
    adj_outputs=[d_out.qpos.grad, d_out.qvel.grad],
    adjoint=True,
  )

  if implicitfast_deriv:
    # ROOT remap (Q symmetric here): adj(a) = M Q^-1 adj(a_int), reusing the factorization built
    # above. The DIRECT-state term y^T[(dM) a_s - (dQ) a_int] is handled right below (d_q via the
    # RNE-bias mass VJP, d_v via the dampingpoly leaf).
    y_int = wp.empty((nworld, nv), dtype=float)
    smooth.solve_LD(m, d_out, qLD_int, qLDiagInv_int, y_int, adj_qacc)
    adj_a = wp.empty((nworld, nv), dtype=float)
    support.mul_m(m, d_out, adj_a, y_int)
    adj_qacc = adj_a
    # STATE-DIRECT term the root remap omits: a_int = Q(q)^-1 M(q) a_s depends on qpos through
    # M(q)/Q(q), so adj_qpos += d_q[ y^T M(q)(a_s - a_u) ] (y = y_int held fixed). Distinct
    # cotangent from the stage-2 IFT lam, so no double-count with the smooth residual term.
    w_dir = wp.empty((nworld, nv), dtype=float)
    wp.launch(_sub_cols, dim=(nworld, nv), inputs=[d_out.qacc, qacc_advance], outputs=[w_dir])  # a_s - a_u
    q_dir = smooth_adjoint.mass_matrix_qpos_vjp(m, d, y_int, w_dir)
    wp.launch(_accum_cols, dim=(nworld, nq), inputs=[q_dir], outputs=[adj_qpos])  # adj_qpos += d_q[y^T M w]
    # Stage 4 -- d_v direct term for STATE-DEPENDENT (dampingpoly) damping:
    # adj_qvel += -d_v[ y_int^T dt*D(v) a_u ] (a_u, y_int held fixed). Zero for LINEAR damping;
    # gated on DAMPER enabled to match the forward Q (which drops the damping block when disabled).
    if (int(m.opt.disableflags) & _DAMPER) == 0:
      dp_out = wp.empty((nworld, nv), dtype=float)
      dp_qv_adj = wp.zeros((nworld, nv), dtype=float)
      dp_ins = [m.dof_damping, m.dof_dampingpoly, d.qvel, m.opt.timestep, qacc_advance]
      wp.launch(_dampingpoly_Qv_leaf, dim=(nworld, nv), inputs=dp_ins, outputs=[dp_out])
      wp.launch(
        _dampingpoly_Qv_leaf,
        dim=(nworld, nv),
        inputs=dp_ins,
        outputs=[dp_out],
        adj_inputs=[None, None, dp_qv_adj, None, None],
        adj_outputs=[y_int],
        adjoint=True,
      )
      wp.launch(_sub_cols, dim=(nworld, nv), inputs=[adj_qvel, dp_qv_adj], outputs=[adj_qvel])  # -= d_v[...]
    # Stage 5 (TODO -- capability gate for GENERAL implicitfast): the mass VJP + dampingpoly leaf
    # cover only rigid-body dM/dq and joint dD/dv; FLUID / TENDON damping / GainType.AFFINE Q
    # terms are OMITTED, so the gradient is SILENTLY INCOMPLETE for such models until a
    # correct-or-RAISE gate lands. Safe subset: joint transmission + FIXED gain, no tendon/fluid.

  # --- 1b. EULERDAMP post-replay transpose remap: the replay ran at a_u, so its adjoint is
  # adj(a_u); map back to the solver root adj(a_s) = M (M+dt*D)^-1 adj(a_u) (both symmetric),
  # REUSING the cached factor. No double-count with the smooth-qvel VJP: the damping force lives
  # in qfrc_smooth -> H/lam; this is the separate integrated-accel remap.
  if eulerdamp:
    y_damp = wp.empty((nworld, nv), dtype=float)
    smooth.solve_LD(m, d_out, qLD_s, qLDiagInv_s, y_damp, adj_qacc)  # (M+dt*D)^-1 adj(a_u), cached factor
    adj_a = wp.empty((nworld, nv), dtype=float)
    support.mul_m(m, d_out, adj_a, y_damp)  # adj(a_s) = M y
    adj_qacc = adj_a
    # KNOWN GAP (low-value, no failing test): eulerdamp does NOT yet add the state-DIRECT terms
    # adj_qpos += d_q[y_damp^T M(a_s-a_u)] / adj_qvel -= d_v[y_damp^T dt*D(v) a_u]; closing it is
    # a mirror of the implicitfast block with y_int -> y_damp, needed only if a config-dep-M or
    # dampingpoly EULER case ever needs FD-exact gradients.
  return adj_qpos, adj_qvel, adj_qacc


def solve_backward(m: Model, d_out: Data, adj_qacc: wp.array):
  """The IFT (stage 2): solve H lam = adj_qacc, reusing the forward solver's assembly + factor.

  H is built at the converged qacc (active set matches; the Newton loop is never backpropagated).
  Returns the SolverContext; lam = ctx.Mgrad and ctx owns it, so keep ctx alive while lam is used.
  """
  nworld = d_out.qpos.shape[0]
  nv = m.nv
  nv_pad = m.nv_pad
  # --- 2. IFT: solve H lam = adj_qacc, reusing the solver's assembly + Cholesky factor ---
  ctx = solver._create_solver_context(m, d_out)
  solver.init_context(m, d_out, ctx, grad=True)  # assembles + factors ctx.h; active set = forward's
  wp.launch(_load_rhs, dim=(nworld, nv_pad), inputs=[nv, adj_qacc], outputs=[ctx.grad])
  ctx.done.zero_()
  solver._cholesky_factorize_solve(m, d_out, ctx)  # ctx.Mgrad[:, :nv] = lam
  return ctx


def _contact_residual_vjp_sparse(
  m: Model,
  d: Data,
  d_out: Data,
  lam: wp.array,
  res_qvel: wp.array,
  res_cdof: wp.array,
  res_subtree_com: wp.array,
  res_efc_pos: wp.array,
  res_contact_pos: wp.array,
  res_contact_frame: wp.array,
  efc_pos_ref: wp.array,
):
  """SPARSE contract-first contact seeds (nv-general; reads no efc.J, so any jacobian storage)."""
  # SPARSE contract-first (nv-general, no _MAX_NV): gather V/A/Z over the symmetric-difference
  # ancestor walk -> loop-free source-AD cone leaf phi=-Z*F(V,A,xi) (seed adj_phi=+1) -> manual
  # sparse scatter.
  ncon_max = d_out.contact.pos.shape[0]
  Vc = wp.zeros(ncon_max, dtype=wp.spatial_vector, requires_grad=True)
  Ac = wp.zeros(ncon_max, dtype=wp.spatial_vector, requires_grad=True)
  Zc = wp.zeros(ncon_max, dtype=wp.spatial_vector, requires_grad=True)
  phi = wp.zeros(ncon_max, dtype=float, requires_grad=True)
  adjV = wp.zeros(ncon_max, dtype=wp.spatial_vector)
  adjA = wp.zeros(ncon_max, dtype=wp.spatial_vector)
  adjZ = wp.zeros(ncon_max, dtype=wp.spatial_vector)
  walk_in = [m.body_rootid, m.body_weldid, m.body_dofnum, m.body_dofadr, m.dof_parentid, m.geom_bodyid]
  state_in = [
    d.qvel,
    d_out.qacc,
    d_out.subtree_com,
    d_out.cdof,
    d_out.contact.pos,
    d_out.contact.geom,
    d_out.contact.efc_address,
    d_out.contact.worldid,
    d_out.efc.state,
    d_out.nacon,
  ]
  wp.launch(constraint_adjoint._contact_gather, dim=ncon_max, inputs=walk_in + state_in + [lam], outputs=[Vc, Ac, Zc])
  want_fric = m.geom_friction.requires_grad  # geom_friction sys-id (the contact-PARAM gradient)
  for _arr in (d_out.contact.frame, d_out.efc.pos):
    _arr.requires_grad = True  # so the leaf adjoint accumulates res_contact_frame / res_efc_pos
  res_contact_friction = None
  if want_fric:
    d_out.contact.friction.requires_grad = True  # so the leaf exposes dphi/dcontact.friction (mu -> cone)
    res_contact_friction = wp.zeros(ncon_max, dtype=vec5)
  # (input, input-adjoint) pairs; res_contact_friction is None unless want_fric (set above).
  phi_pairs = [
    (m.opt.timestep, None),
    (m.opt.disableflags, None),
    (m.opt.impratio_invsqrt, None),
    (d_out.contact.frame, res_contact_frame),  # dr/dcontact_frame
    (d_out.contact.friction, res_contact_friction),  # dr/dcontact.friction -> chained to geom_friction below
    (d_out.contact.solref, None),
    (d_out.contact.solreffriction, None),
    (d_out.contact.solimp, None),
    (d_out.contact.dim, None),
    (d_out.contact.efc_address, None),
    (d_out.contact.worldid, None),
    (d_out.efc.pos, res_efc_pos),  # dr/defc_pos (penetration)
    (d_out.efc.margin, None),
    (d_out.efc.D, None),
    (d_out.efc.state, None),
    (d_out.nacon, None),
    (Vc, adjV),  # Vbar
    (Ac, adjA),  # Abar
    (Zc, adjZ),  # Zbar
    (efc_pos_ref, None),
  ]
  phi_in = [a for a, _ in phi_pairs]
  contact_phi_kernel = constraint_adjoint._contact_phi(int(m.opt.cone))  # cone-specialized (cached)
  wp.launch(contact_phi_kernel, dim=ncon_max, inputs=phi_in, outputs=[phi])
  wp.launch(constraint_adjoint._fill_ones, dim=ncon_max, inputs=[phi.grad])  # seed adj_phi = +1 (phi already folds in lam)
  wp.launch(
    contact_phi_kernel,
    dim=ncon_max,
    inputs=phi_in,
    outputs=[phi],
    adj_inputs=[g for _, g in phi_pairs],
    adj_outputs=[phi.grad],
    adjoint=True,
  )
  wp.launch(
    constraint_adjoint._contact_scatter,
    dim=ncon_max,
    inputs=walk_in + state_in + [lam, adjV, adjA, adjZ, res_qvel, res_cdof, res_subtree_com],
    outputs=[res_contact_pos],
  )
  if want_fric:  # CONTACT-PARAM sys-id: chain dphi/dcontact.friction -> m.geom_friction.grad (IFT minus)
    wp.launch(
      constraint_adjoint._contact_friction_geom_vjp,
      dim=ncon_max,
      inputs=[
        m.geom_priority,
        m.geom_friction,
        d_out.contact.geom,
        d_out.contact.efc_address,
        d_out.contact.worldid,
        d_out.efc.state,
        d_out.nacon,
        res_contact_friction,
      ],
      outputs=[m.geom_friction.grad],
    )


def _contact_residual_vjp_dense(
  m: Model,
  d: Data,
  d_out: Data,
  lam: wp.array,
  res_qvel: wp.array,
  res_cdof: wp.array,
  res_subtree_com: wp.array,
  res_efc_pos: wp.array,
  res_contact_pos: wp.array,
  res_contact_frame: wp.array,
  efc_pos_ref: wp.array,
):
  """DENSE contact seeds: one source-AD Warp reverse of the whole residual (nv <= _MAX_NV)."""
  nworld = d.qpos.shape[0]
  nv = m.nv
  _assert_dense_unroll("contact-residual oracle")
  residual_contact_kernel = constraint_adjoint._residual_contact(int(m.opt.cone))  # cone-specialized kernel (cached)
  r = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
  for _arr in (d_out.cdof, d_out.subtree_com, d_out.efc.pos, d_out.contact.pos, d_out.contact.frame):
    _arr.requires_grad = True  # so the manual adjoint launch accumulates their input-adjoints
  # (input, input-adjoint) pairs: one +lam adjoint launch exposes every kinematic-intermediate
  # input-adjoint (dr/dx^T lam) into its res buffer; the IFT minus is applied at write-back.
  rin_pairs = [
    (nv, None),
    (m.opt.timestep, None),
    (m.opt.disableflags, None),
    (m.opt.impratio_invsqrt, None),
    (m.body_parentid, None),
    (m.body_rootid, None),
    (m.dof_bodyid, None),
    (m.geom_bodyid, None),
    (m.body_isdofancestor, None),
    (d.qpos, None),  # unused in-kernel; res_qpos accumulates the stage-3c narrowphase dqpos below
    (d.qvel, res_qvel),
    (d_out.qacc, None),
    (d_out.subtree_com, res_subtree_com),  # dr/dsubtree_com (articulated)
    (d_out.cdof, res_cdof),  # dr/dcdof (articulated)
    (d_out.contact.pos, res_contact_pos),  # dr/dcontact_pos (moment arm)
    (d_out.contact.frame, res_contact_frame),  # dr/dcontact_frame (normal/tangent rows)
    (d_out.contact.friction, None),
    (d_out.contact.solref, None),
    (d_out.contact.solreffriction, None),
    (d_out.contact.solimp, None),
    (d_out.contact.dim, None),
    (d_out.contact.geom, None),
    (d_out.contact.efc_address, None),
    (d_out.contact.worldid, None),
    (d_out.efc.pos, res_efc_pos),  # dr/defc_pos (penetration)
    (d_out.efc.margin, None),
    (d_out.efc.D, None),
    (d_out.efc.state, None),
    (d_out.nacon, None),
    (efc_pos_ref, None),  # frozen efc_pos reference (no adjoint) for the D-recovery imp0/pos0
  ]
  rin = [a for a, _ in rin_pairs]
  wp.launch(residual_contact_kernel, dim=nworld, inputs=rin, outputs=[r])
  wp.launch(_copy_cols, dim=(nworld, nv), inputs=[lam], outputs=[r.grad])  # seed adj_r = lam
  wp.launch(
    residual_contact_kernel,
    dim=nworld,
    inputs=rin,
    outputs=[r],
    adj_inputs=[g for _, g in rin_pairs],
    adj_outputs=[r.grad],
    adjoint=True,
  )


def contact_residual_backward(m: Model, d: Data, d_out: Data, lam: wp.array, res_qpos: wp.array, res_qvel: wp.array):
  """Contact residual VJP (stages 3 and 3c): the dqpos/dqvel channels of -J^T f_contact.

  Stage 3 routes on jacobian storage (dense kernel for nv <= _MAX_NV, sparse contract-first
  otherwise); stage 3c lifts the geometry seeds into res_qpos. cdof is frozen (dcdof/dqpos omitted).
  """
  nworld = d.qpos.shape[0]
  nv = m.nv
  res_cdof = wp.zeros((nworld, nv), dtype=wp.spatial_vector)
  res_subtree_com = wp.zeros((nworld, m.nbody), dtype=wp.vec3)
  res_efc_pos = wp.zeros_like(d_out.efc.pos)
  res_contact_pos = wp.zeros_like(d_out.contact.pos)
  res_contact_frame = wp.zeros_like(d_out.contact.frame)
  efc_pos_ref = wp.clone(d_out.efc.pos)  # frozen D-recovery reference (not differentiated)
  efc_pos_ref.requires_grad = False

  # Route sparse for: forced (tests), sparse-jacobian forwards, nv beyond the dense unroll cap,
  # and geom_friction sys-id (the contact-param VJP chain lives only in the sparse branch).
  if _FORCE_SPARSE_CONTACT or m.is_sparse or nv > _MAX_NV or m.geom_friction.requires_grad:
    _contact_residual_vjp_sparse(
      m, d, d_out, lam, res_qvel, res_cdof, res_subtree_com, res_efc_pos, res_contact_pos, res_contact_frame, efc_pos_ref
    )
  else:
    _contact_residual_vjp_dense(
      m, d, d_out, lam, res_qvel, res_cdof, res_subtree_com, res_efc_pos, res_contact_pos, res_contact_frame, efc_pos_ref
    )

  # --- stage 3c: contact dqpos. collision_adjoint replays each frozen contact's narrowphase
  # geometry and chains geom-pose -> qpos (support.jac_dof), accumulating into res_qpos (which
  # _sub_cols subtracts). nacon=0 / unsupported geom pairs -> no-op.
  collision_adjoint.contact_qpos_vjp(
    m, d_out, d.qpos, res_contact_pos, res_contact_frame, res_efc_pos, res_subtree_com, res_cdof, res_qpos
  )


def smooth_vel_backward(m: Model, d: Data, d_out: Data, lam: wp.array, adj_qvel: wp.array):
  """Smooth residual dqvel + dctrl (stage 4).

  Contracts the analytic G = dqfrc_smooth/dqvel with lam into adj_qvel and runs the actuation
  leaf into d.ctrl.grad; the dqpos channel lives in smooth_qpos_backward.
  """
  nworld = d.qpos.shape[0]
  # --- 4. smooth residual: dqvel (analytic) + dctrl (actuation leaf), same lam ---
  # G = dqfrc_smooth/dqvel = (M_D - qLU)/dt. deriv_rne_vel MUST use flg_subtract=False: the bias
  # term must ADD (qfrc_smooth subtracts qfrc_bias); the forward `implicit` True convention has
  # the wrong sign here. d_out.qvel was overwritten by integration -> alias to d.qvel, then restore.
  saved_qvel = d_out.qvel
  d_out.qvel = d.qvel
  qH_M = wp.empty(d_out.M.shape, dtype=float)
  derivative.deriv_smooth_vel(m, d_out, qH_M)  # M - dt*d(qfrc_passive+qfrc_actuator)/dqvel (M-structure)
  qLU = wp.empty((nworld, m.nD), dtype=float)
  M_D = wp.empty((nworld, m.nD), dtype=float)
  wp.launch(forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, qH_M], outputs=[qLU])
  derivative.deriv_rne_vel(m, d_out, qLU, flg_subtract=False)  # += dt*dqfrc_bias/dqvel; (M_D-qLU)/dt = dqfrc_smooth/dqvel
  wp.launch(forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, d_out.M], outputs=[M_D])
  d_out.qvel = saved_qvel
  # Assemble the dqvel VJP by contracting the sparse (D-structure) velocity Jacobian G against
  # lam: adj_qvel += G^T lam, scattered over the nD nonzeros (mujoco_warp has no transpose-apply
  # for this layout).
  wp.launch(
    _smooth_qvel_vjp,
    dim=(nworld, m.nD),
    inputs=[m.opt.timestep, m.qD_fullm_i, m.qD_fullm_j, qLU, M_D, lam],
    outputs=[adj_qvel],  # adj_qvel += G^T lam
  )
  if m.nu > 0 and d.ctrl.requires_grad:
    wp.launch(
      _smooth_ctrl_vjp,
      dim=(nworld, m.nu),
      inputs=[m.actuator_gainprm, d_out.moment_rownnz, d_out.moment_rowadr, d_out.moment_colind, d_out.actuator_moment, lam],
      outputs=[d.ctrl.grad],
    )


def smooth_param_backward(m: Model, d: Data, d_out: Data, lam: wp.array):
  """Smooth-PARAM sys-id with the same IFT lam (stage 4b).

  Two requires_grad-gated leaves: armature/damping via the LOCAL smooth residual, and the
  inertial params (mass/inertia/ipos/iquat) via smooth_adjoint.inertia_param_vjp (cinert route).
  """
  nworld = d.qpos.shape[0]
  nv = m.nv
  # --- 4b. smooth-PARAM sys-id (armature, viscous damping): AD the LOCAL residual
  # r_local = dof_armature*qacc + dof_damping*qvel with the SAME lam, seeding adj_r = -lam so
  # -(dr/dtheta)^T lam lands in m.<param>.grad. A model param is ONE leaf shared across the
  # trajectory: its grad ACCUMULATES every step/world; the caller zeros it once per trajectory.
  if m.dof_armature.requires_grad or m.dof_damping.requires_grad:
    r_s = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
    _rs_inputs = [m.dof_armature, m.dof_damping, d.qvel, d_out.qacc]
    wp.launch(_residual_smooth_local, dim=(nworld, nv), inputs=_rs_inputs, outputs=[r_s])
    wp.launch(smooth_adjoint._neg_cols, dim=(nworld, nv), inputs=[lam], outputs=[r_s.grad])  # seed adj_r = -lam
    wp.launch(
      _residual_smooth_local,
      dim=(nworld, nv),
      inputs=_rs_inputs,
      outputs=[r_s],
      adj_inputs=[
        m.dof_armature.grad if m.dof_armature.requires_grad else None,
        m.dof_damping.grad if m.dof_damping.requires_grad else None,
        None,
        None,
      ],
      adj_outputs=[r_s.grad],
      adjoint=True,
    )

  # --- 4b (inertial). body_{mass,inertia,ipos,iquat} sys-id: all four enter the smooth residual
  # ONLY through cinert, so adj_cinert from rne_backward routed through the source-AD cinert /
  # inertial-frame leaves yields -(dr/dtheta)^T lam with no hand-written VJP. Needs the kinematic
  # intermediates AT the linearization -> recompute on a non-grad shallow view; never mutate d_out.
  if m.body_mass.requires_grad or m.body_inertia.requires_grad or m.body_ipos.requires_grad or m.body_iquat.requires_grad:
    s = _smooth_linearization(m, d, d_out)  # shallow view + rne(flg_acc=True); replaces the per-step deep clone
    smooth_adjoint.inertia_param_vjp(m, s, lam)  # -(dr/dtheta)^T lam -> m.body_{mass,inertia,ipos,iquat}.grad


def noncontact_constraint_backward(
  m: Model, d: Data, d_out: Data, lam: wp.array, res_qpos: wp.array, res_qvel: wp.array, ctx_Jaref: wp.array = None
):
  """Non-contact constraint residual VJP + param sys-id (stage 4c).

  Tangent dqpos -> res_qpos (via _dof_to_qpos), dqvel -> res_qvel, param grads in-place. Routes
  to the sparse/CSR path (nv > _MAX_NV or CSR efc.J; needs ctx_Jaref) or the dense _MAX_NV kernel.
  """
  nworld = d.qpos.shape[0]
  nv = m.nv
  res_dof = wp.zeros((nworld, nv), dtype=float)  # per-dof TANGENT dqpos (lifted to res_qpos below; shared)
  # Route to the SPARSE/CSR path when nv>_MAX_NV (dense kernel's range(_MAX_NV) unroll cap) OR
  # when efc.J is stored CSR (m.is_sparse -- the dense kernel indexes efc.J[w,row,i] densely and
  # CANNOT read CSR). Otherwise the dense _MAX_NV kernel (nv<=_MAX_NV, dense efc.J) -- the proven
  # path / A-B oracle.
  if nv > _MAX_NV or m.is_sparse:
    # --- SPARSE/CSR nv-general path: gather Z = J*lam -> loop-free anchored leaf phi = -Z*f ->
    # J^T scatter into res_qvel + res_dof; param sys-id accumulates into m.<param>.grad internally.
    # Anchored on the FROZEN ctx.Jaref (only Z is re-reduced); slide/hinge + dof-friction are
    # exact here (dJ/dq=0).
    assert ctx_Jaref is not None, "sparse non-contact constraint VJP (nv>_MAX_NV) requires ctx.Jaref"
    _residual_constraint_sparse(m, d_out, ctx_Jaref, lam, res_qvel, res_dof)
  else:
    # --- dense _MAX_NV static-unroll path (guard its unroll invariant). ONE forward + ONE +lam
    # adjoint launch yields every input-adjoint as +(dr/d*)^T lam into its own res buffer; the IFT
    # MINUS is applied at WRITE-BACK. PARAMS (dof_frictionloss + per-class solref) fall out of the
    # same launch (auto-diffed through _contact_kbimp; no per-param kernel).
    _assert_dense_unroll("_residual_constraint")
    r_c = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
    dpos = wp.zeros((nworld, nv), dtype=float, requires_grad=True)  # TANGENT seed (zeros); dr/d(delta_theta) -> res_dof
    # (input, input-adjoint) pairs; adjoint None for non-differentiated inputs, else a res buffer.
    # Pairing keeps inputs/adj_inputs aligned under reorder -- no index bookkeeping.
    # res_qvel/res_dof are state-grad accumulators; *_grad buffers exist only under requires_grad.
    res_dof_frictionloss = wp.zeros_like(m.dof_frictionloss) if m.dof_frictionloss.requires_grad else None
    res_dof_solref = wp.zeros_like(m.dof_solref) if m.dof_solref.requires_grad else None
    res_eq_solref = wp.zeros_like(m.eq_solref) if m.eq_solref.requires_grad else None
    res_jnt_solref = wp.zeros_like(m.jnt_solref) if m.jnt_solref.requires_grad else None
    rc = [
      (nv, None),
      (m.opt.timestep, None),
      (m.opt.disableflags, None),
      (m.jnt_solref, res_jnt_solref),
      (m.jnt_solimp, None),
      (m.dof_solref, res_dof_solref),
      (m.dof_solimp, None),
      (m.dof_frictionloss, res_dof_frictionloss),
      (m.eq_solref, res_eq_solref),
      (m.eq_solimp, None),
      (d_out.nefc, None),
      (d.qvel, res_qvel),  # dqvel state-grad -> res_qvel (accumulates with contact; subtracted by _sub_cols)
      (d_out.qacc, None),
      (d_out.efc.type, None),
      (d_out.efc.id, None),
      (d_out.efc.J, None),
      (d_out.efc.pos, None),
      (d_out.efc.margin, None),
      (d_out.efc.D, None),
      (d_out.efc.state, None),
      (dpos, res_dof),  # TANGENT dqpos seed -> res_dof (lifted to res_qpos by _dof_to_qpos below)
    ]
    rc_inputs = [a for a, _ in rc]
    wp.launch(_residual_constraint, dim=nworld, inputs=rc_inputs, outputs=[r_c])
    wp.launch(_copy_cols, dim=(nworld, nv), inputs=[lam], outputs=[r_c.grad])  # seed adj_r = +lam
    wp.launch(
      _residual_constraint,
      dim=nworld,
      inputs=rc_inputs,
      outputs=[r_c],
      adj_inputs=[g for _, g in rc],
      adj_outputs=[r_c.grad],
      adjoint=True,
    )
    if res_dof_frictionloss is not None:  # -(dr/dfl)^T lam into the shared param grad
      wp.launch(smooth_adjoint._accum_neg, dim=res_dof_frictionloss.shape, inputs=[res_dof_frictionloss], outputs=[m.dof_frictionloss.grad])
    for _pg, _rb in ((m.dof_solref, res_dof_solref), (m.eq_solref, res_eq_solref), (m.jnt_solref, res_jnt_solref)):
      if _rb is not None:
        wp.launch(smooth_adjoint._accum_neg_vec2, dim=_rb.shape, inputs=[_rb], outputs=[_pg.grad])
  # lift the per-dof TANGENT dqpos to qpos (quaternion-aware: free/BALL via 2*(q outer [0,g]));
  # accumulates into res_qpos alongside the contact dqpos, then both flow through _sub_cols.
  # Shared by both paths.
  wp.launch(
    collision_adjoint._dof_to_qpos,
    dim=(nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof],
    outputs=[res_qpos],
  )


def smooth_qpos_backward(m: Model, d: Data, d_out: Data, lam: wp.array, res_qpos: wp.array, adj_qpos: wp.array):
  """Smooth-force dqpos (stage 5): the analytic reduced replay, accumulated into res_qpos.

  adj_qpos is written only by the drop-in FD oracle (adjoint_test_util.fd_smooth_qpos_backward);
  the analytic path leaves it untouched.
  """
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  # Analytic reduced replay: each term returns d(lam^T r_smooth_term)/dqpos, accumulated into
  # res_qpos (the _sub_cols IFT minus). NO FD fallback: assert_smooth_supported RAISES on any
  # enabled smooth feature without an analytic leaf; FD is the explicit test-only path below.
  # Local import breaks the smooth_adjoint<->adjoint cycle.
  smooth_adjoint.assert_smooth_supported(m)
  # CRITICAL: evaluate at the linearization (d.qpos, d.qvel, d_out.qacc) with rne(flg_acc=True).
  # The live d_out has the flg_acc=False bias (missing the M*qacc term) and the INTEGRATED next
  # state; skipping the _smooth_linearization view drops d(M*qacc)/dqpos.
  s = _smooth_linearization(m, d, d_out)
  res_q = smooth_adjoint.smooth_force_backward(m, s, lam, flg_acc=True)  # rigid-body RNE bias
  wp.launch(_accum_cols, dim=(nworld, nq), inputs=[res_q], outputs=[res_qpos])
  res_sp = smooth_adjoint.spring_qpos_vjp(m, s, lam)  # joint springs (all joint types)
  wp.launch(_accum_cols, dim=(nworld, nq), inputs=[res_sp], outputs=[res_qpos])
  res_ac = smooth_adjoint.actuator_qpos_vjp(m, s, lam)  # affine joint-transmission actuators
  wp.launch(_accum_cols, dim=(nworld, nq), inputs=[res_ac], outputs=[res_qpos])
  res_gc = smooth_adjoint.gravcomp_qpos_vjp(m, s, lam)  # gravity compensation (passive bucket)
  wp.launch(_accum_cols, dim=(nworld, nq), inputs=[res_gc], outputs=[res_qpos])


def _write_input_adjoints(m: Model, d: Data, adj_qpos: wp.array, adj_qvel: wp.array, res_qpos: wp.array, res_qvel: wp.array):
  """Write d.{qpos,qvel}.grad = integrator-direct adj - residual-VJP (the IFT minus)."""
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  # --- write input adjoints (each d == datas[t] is the d_in of exactly one step) ---
  wp.launch(_sub_cols, dim=(nworld, nq), inputs=[adj_qpos, res_qpos], outputs=[d.qpos.grad])
  wp.launch(_sub_cols, dim=(nworld, nv), inputs=[adj_qvel, res_qvel], outputs=[d.qvel.grad])
  # The IFT VJP is linear in r = r_smooth + r_contact, so every residual term shares the SAME lam
  # and SUMS into the writes above. No double-count with H = M + J^T G_s J: M in H is dr/dqacc,
  # while d(M*qacc)/dtheta in r_smooth is w.r.t. theta.


def forward_backward_ift(m: Model, d: Data, d_out: Data, adj_qpos: wp.array, adj_qvel: wp.array, adj_qacc: wp.array):
  """Adjoint (VJP) of forward() via the implicit function theorem, not the pipeline reversed.

  Solves H lam = adj_qacc at the converged qacc -- the Newton loop is never backpropagated -- then
  the commuting term helpers accumulate adj_theta = -(dr/dtheta)^T lam into the input/param grads.
  """
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  ctx = solve_backward(m, d_out, adj_qacc)  # solver.solve_backward: H lam = adj_qacc; ctx owns lam -- keep alive
  lam = ctx.Mgrad
  # residual-VJP cross-term seeds (the qpos / qvel GRID COLUMNS), shared across the contact /
  # constraint / smooth terms and summed into d.{qpos,qvel}.grad at write-back. (The contact
  # term's five contact-geometry seeds are local to its helper.)
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  res_qvel = wp.zeros((nworld, nv), dtype=float)
  # ORDER below = the former monolith's launch order, kept for byte-identical FP. The five term
  # helpers only ACCUMULATE (none reads another's output) so they COMMUTE; only the endpoints are
  # load-bearing: solve_backward FIRST (produces lam), _write_input_adjoints LAST (reads the sums).
  contact_residual_backward(m, d, d_out, lam, res_qpos, res_qvel)  # contact term -> stage 3 dqvel + stage 3c dqpos
  smooth_vel_backward(m, d, d_out, lam, adj_qvel)  # smooth term -> stage 4 qvel + ctrl columns
  smooth_param_backward(m, d, d_out, lam)  # smooth term -> stage 4b model-param column (armature/damping sys-id)
  noncontact_constraint_backward(m, d, d_out, lam, res_qpos, res_qvel, ctx.Jaref)  # constraint term -> stage 4c qpos + qvel
  smooth_qpos_backward(m, d, d_out, lam, res_qpos, adj_qpos)  # smooth term -> stage 5 qpos column
  _write_input_adjoints(m, d, adj_qpos, adj_qvel, res_qpos, res_qvel)  # grad = integrator-direct - residual


def step_backward(m: Model, d: Data, d_out: Data):
  """Analytic backward of step(): reads d_out.{qpos,qvel}.grad, writes d.{qpos,qvel}.grad.

  Registered via forward.register_backward_hook as one tape.record_func per step (distinct in/out
  buffers chain BPTT); composes advance_backward + forward_backward_ift, all graph-capture-ready.
  """
  _assert_step_supported(m)
  adj_qpos, adj_qvel, adj_qacc = advance_backward(m, d, d_out)
  forward_backward_ift(m, d, d_out, adj_qpos, adj_qvel, adj_qacc)


forward.register_backward_hook(step_backward)


# ============================================================================================
# Analytic position backward for forward.fwd_kinematics (differentiable observations).
# No AD of the kinematics tree (Warp dynamic-loop backward staleness): adj_qpos = sum_site J^T*adj
# via the analytic site Jacobian (support.jac_dof) on a fresh non-grad scratch, built in
# dof/tangent space and lifted to qpos by _dof_to_qpos (quaternion-aware; correct for nq != nv).
# ============================================================================================


@wp.kernel(enable_backward=False)
def _site_jac_vjp(
  # Model:
  nsite: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_site: wp.array2d[wp.vec3],
  # Out:
  adj_dof_out: wp.array2d[float],
):
  """adj_dof_out[w, i] += sum_site (dsite_xpos/d(dof i))^T adj_site[w, site] (dof/tangent space)."""
  w, i = wp.tid()
  acc = float(0.0)
  for s in range(nsite):
    jacp, jacr = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      site_xpos_in[w, s],
      site_bodyid[s],
      i,
      w,
    )
    acc += wp.dot(jacp, adj_site[w, s])
  adj_dof_out[w, i] += acc


def fwd_kinematics_backward(m: Model, d: Data):
  """Analytic position VJP for forward.fwd_kinematics: adj(site_xpos) -> adj(qpos).

  Scatters J^T*adj through the analytic site Jacobian on a fresh non-grad scratch clone (no
  kinematics-tree AD). Registered as forward's position backward hook.
  """
  if m.nsite == 0 or d.qpos.grad is None or d.site_xpos.grad is None:
    return
  fd = smooth_adjoint._clone_for_fd(d)  # value-only scratch; refresh kinematics state at qpos
  smooth.kinematics(m, fd)
  smooth.com_pos(m, fd)
  adj_dof = wp.zeros((d.nworld, m.nv), dtype=float)  # per-dof TANGENT VJP, lifted to qpos below
  wp.launch(
    _site_jac_vjp,
    dim=(d.nworld, m.nv),
    inputs=[
      m.nsite,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      m.site_bodyid,
      m.body_isdofancestor,
      fd.site_xpos,
      fd.subtree_com,
      fd.cdof,
      d.site_xpos.grad,
    ],
    outputs=[adj_dof],
  )
  # lift the per-dof tangent gradient to qpos (1:1 for slide/hinge/free-translation;
  # 2*(q outer [0,g]) quaternion lift for free/ball rotation) -- the nq!=nv map, same kernel the
  # contact dqpos path uses.
  wp.launch(
    collision_adjoint._dof_to_qpos,
    dim=(d.nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, adj_dof],
    outputs=[d.qpos.grad],
  )


forward.register_position_backward_hook(fwd_kinematics_backward)
