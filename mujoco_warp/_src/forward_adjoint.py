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
"""VJPs for time integration, smooth dynamics, and forward kinematics."""

import dataclasses

import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import math
from mujoco_warp._src import model_adjoint
from mujoco_warp._src import smooth
from mujoco_warp._src import smooth_adjoint
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import support
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import mat66
from mujoco_warp._src.types import vec6
from mujoco_warp._src.types import vec10
from mujoco_warp._src.warp_util import event_scope

# Adjoint reductions use unordered atomics; deterministic backward execution is deferred.
wp.set_module_options({"enable_backward": True})


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
  jnttype = jnt_type[jntid]
  qadr = jnt_qposadr[jntid]
  dadr = jnt_dofadr[jntid]
  qvel_lin = wp.vec3(0.0, 0.0, 0.0)
  qvel_ang = wp.vec3(0.0, 0.0, 0.0)

  if jnttype == JointType.FREE:
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
  elif jnttype == JointType.BALL:
    vx = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 0, 1.0)
    vy = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 1, 1.0)
    vz = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 2, 1.0)
    qvel_out[worldid, dadr + 0] = vx
    qvel_out[worldid, dadr + 1] = vy
    qvel_out[worldid, dadr + 2] = vz
    qvel_ang = wp.vec3(vx, vy, vz)
  else:
    v = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr, 1.0)
    qvel_out[worldid, dadr] = v
    qvel_lin = wp.vec3(v, 0.0, 0.0)

  forward.next_position(qpos_in, jnttype, qadr, dt, worldid, qvel_lin, qvel_ang, qpos_out)


@wp.kernel(enable_backward=False)
def _smooth_qvel_vjp(
  # Model:
  opt_timestep: wp.array[float],
  qD_fullm_i: wp.array[int],
  qD_fullm_j: wp.array[int],
  # Data in:
  qLU_in: wp.array2d[float],
  # In:
  M_D: wp.array2d[float],
  lam: wp.array2d[float],
  # Out:
  adj_qvel_out: wp.array2d[float],
):
  worldid, elemid = wp.tid()
  dt = opt_timestep[worldid % opt_timestep.shape[0]]
  i = qD_fullm_i[elemid]
  j = qD_fullm_j[elemid]
  wp.atomic_add(
    adj_qvel_out[worldid], j, (M_D[worldid, elemid] - qLU_in[worldid, elemid]) / dt * lam[worldid, i]
  )  # kernel_analyzer: ignore[atomic]


@wp.kernel(enable_backward=False)
def _smooth_ctrl_vjp(
  # Model:
  opt_timestep: wp.array[float],
  jnt_actfrclimited: wp.array[bool],
  jnt_actfrcrange: wp.array2d[wp.vec2],
  dof_jntid: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_gainprm: wp.array2d[vec10],
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  # Data in:
  ctrl_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  actuator_velocity_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  qfrc_actuator_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  qacc_advance: wp.array2d[float],
  y_remap: wp.array2d[float],
  implicitfast: bool,
  dsbl_clampctrl: int,
  # Out:
  adj_ctrl_out: wp.array2d[float],
):
  worldid, actuatorid = wp.tid()
  if actuator_ctrllimited[actuatorid] and not dsbl_clampctrl:
    bounds = actuator_ctrlrange[worldid % actuator_ctrlrange.shape[0], actuatorid]
    ctrl = ctrl_in[worldid, actuatorid]
    # Keep the inward derivative at bounds so projected controls can move off them.
    if ctrl < bounds[0] or ctrl > bounds[1]:
      return
  if actuator_forcelimited[actuatorid]:
    bounds = actuator_forcerange[worldid % actuator_forcerange.shape[0], actuatorid]
    force = actuator_force_in[worldid, actuatorid]
    if force <= bounds[0] or force >= bounds[1]:
      return
  rowadr = moment_rowadr_in[worldid, actuatorid]
  rownnz = moment_rownnz_in[worldid, actuatorid]
  gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], actuatorid]
  gain = gainprm[0]
  if actuator_gaintype[actuatorid] == GainType.AFFINE:
    gain += gainprm[1] * actuator_length_in[worldid, actuatorid] + gainprm[2] * actuator_velocity_in[worldid, actuatorid]
    if implicitfast and gainprm[2] != 0.0:
      # The implicitfast matrix uses clamped control; saturated controls returned above.
      moment_qacc, moment_y = float(0.0), float(0.0)
      for i in range(rownnz):
        elemid = rowadr + i
        dofid = moment_colind_in[worldid, elemid]
        moment = actuator_moment_in[worldid, elemid]
        moment_qacc += moment * qacc_advance[worldid, dofid]
        moment_y += moment * y_remap[worldid, dofid]
      adj_ctrl_out[worldid, actuatorid] += opt_timestep[worldid % opt_timestep.shape[0]] * gainprm[2] * moment_qacc * moment_y
  value, _ = smooth_adjoint._actuator_moment_dots(
    jnt_actfrclimited,
    jnt_actfrcrange,
    dof_jntid,
    moment_colind_in,
    actuator_moment_in,
    qfrc_actuator_in,
    lam,
    worldid,
    rowadr,
    rownnz,
  )
  adj_ctrl_out[worldid, actuatorid] += gain * value


@wp.kernel(enable_backward=False)
def _actuator_limit_qvel_vjp(
  # Model:
  jnt_actfrclimited: wp.array[bool],
  jnt_actfrcrange: wp.array2d[wp.vec2],
  dof_jntid: wp.array[int],
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
  qfrc_actuator_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  dsbl_clampctrl: int,
  # Out:
  adj_qvel_out: wp.array2d[float],
):
  """Corrects actuator velocity derivatives for joint-force limits."""
  w, actid = wp.tid()
  if actuator_forcelimited[actid]:
    bounds = actuator_forcerange[w % actuator_forcerange.shape[0], actid]
    force = actuator_force_in[w, actid]
    if force <= bounds[0] or force >= bounds[1]:
      return  # deriv_smooth_vel already excludes this actuator.
  dfdv = float(0.0)
  if actuator_gaintype[actid] == GainType.AFFINE:
    gain = actuator_gainprm[w % actuator_gainprm.shape[0], actid][2]
    ctrl = ctrl_in[w, actid]
    if actuator_ctrllimited[actid] and not dsbl_clampctrl:
      bounds = actuator_ctrlrange[w % actuator_ctrlrange.shape[0], actid]
      ctrl = wp.clamp(ctrl, bounds[0], bounds[1])
    dfdv += gain * ctrl
  if actuator_biastype[actid] == BiasType.AFFINE:
    dfdv += actuator_biasprm[w % actuator_biasprm.shape[0], actid][2]
  if dfdv == 0.0:
    return
  rowadr = moment_rowadr_in[w, actid]
  rownnz = moment_rownnz_in[w, actid]
  _, clamped_moment_lam = smooth_adjoint._actuator_moment_dots(
    jnt_actfrclimited,
    jnt_actfrcrange,
    dof_jntid,
    moment_colind_in,
    actuator_moment_in,
    qfrc_actuator_in,
    lam,
    w,
    rowadr,
    rownnz,
  )
  correction = -dfdv * clamped_moment_lam
  if correction == 0.0:
    return
  for i in range(rownnz):
    elemid = rowadr + i
    wp.atomic_add(
      adj_qvel_out[w], moment_colind_in[w, elemid], correction * actuator_moment_in[w, elemid]
    )  # kernel_analyzer: ignore[atomic]


@wp.kernel(enable_backward=False)
def _xfrc_applied_vjp(
  # Model:
  nv: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  lam: wp.array2d[float],
  # Out:
  adj_xfrc_applied_out: wp.array2d[wp.spatial_vector],
):
  worldid, bodyid = wp.tid()
  force = wp.vec3(0.0)
  torque = wp.vec3(0.0)
  for dofid in range(nv):
    jacp, jacr = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      xipos_in[worldid, bodyid],
      bodyid,
      dofid,
      worldid,
    )
    force += lam[worldid, dofid] * jacp
    torque += lam[worldid, dofid] * jacr
  # xfrc_applied stores force followed by torque, unlike motion-space spatial vectors.
  adj_xfrc_applied_out[worldid, bodyid] += wp.spatial_vector(force, torque)


@wp.kernel(enable_backward=True)
def _dampingpoly_Qv_leaf(
  # Model:
  opt_timestep: wp.array[float],
  jnt_actuatorid: wp.array[int],
  dof_jntid: wp.array[int],
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # Out:
  out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  dt = opt_timestep[worldid % opt_timestep.shape[0]]
  damping, poly = util_misc.dof_damping_total(
    jnt_actuatorid,
    dof_jntid,
    dof_damping,
    dof_dampingpoly,
    actuator_trntype,
    actuator_trnid,
    actuator_damping,
    actuator_dampingpoly,
    actuator_gear,
    worldid,
    dofid,
  )
  out[worldid, dofid] = dt * util_misc._poly_force_deriv(damping, poly, qvel_in[worldid, dofid], 1) * qacc_in[worldid, dofid]


@wp.kernel(enable_backward=False)
def _implicit_free_body_backward(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  body_dofadr: wp.array[int],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  body_freeadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  xpos_in: wp.array2d[wp.vec3],
  xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  crb_in: wp.array2d[vec10],
  M_in: wp.array2d[float],
  # In:
  damp_deriv: wp.array2d[float],
  qacc_advance: wp.array2d[float],
  adj_qacc: wp.array2d[float],
  # Out:
  y_out: wp.array2d[float],
  adj_qvel_out: wp.array2d[float],
  adj_dof_out: wp.array2d[float],
):
  """Transpose the free-body solve and contract its gyroscopic state derivatives."""
  worldid, freeid = wp.tid()
  bodyid = body_freeadr[freeid]
  dofadr = body_dofadr[bodyid]
  dt = opt_timestep[worldid % opt_timestep.shape[0]]
  R = xmat_in[worldid, bodyid]
  crb = crb_in[worldid, bodyid]
  mass = crb[9]
  Iw = wp.mat33(crb[0], crb[3], crb[4], crb[3], crb[1], crb[5], crb[4], crb[5], crb[2])
  s = subtree_com_in[worldid, bodyid] - xpos_in[worldid, bodyid]
  velocity = wp.vec3(qvel_in[worldid, dofadr + 3], qvel_in[worldid, dofadr + 4], qvel_in[worldid, dofadr + 5])
  lin, rot = math.free_bias_vel_blocks(mass, R, Iw, s, velocity)

  # Match forward._implicit_free_body. Fluid forces and sleep are rejected before backward.
  A = mat66(0.0)
  for r in range(6):
    row = dofadr + r
    start = M_rowadr[row]
    for k in range(M_rownnz[row]):
      c = M_colind[start + k] - dofadr
      value = M_in[worldid, start + k]
      A[r, c] = value
      A[c, r] = value
  if not (opt_disableflags & DisableBit.DAMPER):
    for r in range(6):
      A[r, r] += dt * damp_deriv[worldid, dofadr + r]
  for r in range(3):
    for c in range(3):
      A[r, 3 + c] += -dt * mass * lin[r, c]
      A[3 + r, 3 + c] += dt * rot[r, c]

  factor, pivot, ok = math.lu_factor_6x6(wp.transpose(A))
  if not ok:
    return
  seed = vec6(0.0)
  for r in range(6):
    seed[r] = adj_qacc[worldid, dofadr + r]
  y = math.lu_solve_6x6(factor, pivot, seed)
  for r in range(6):
    y_out[worldid, dofadr + r] = y[r]

  y_lin = wp.vec3(y[0], y[1], y[2])
  y_rot = wp.vec3(y[3], y[4], y[5])
  acceleration = wp.vec3(
    qacc_advance[worldid, dofadr + 3], qacc_advance[worldid, dofadr + 4], qacc_advance[worldid, dofadr + 5]
  )
  # Bias is quadratic in velocity, so d(B(v)*a)/dv = B(a).
  lin_acc, rot_acc = math.free_bias_vel_blocks(mass, R, Iw, s, acceleration)
  adj_velocity = dt * (mass * (wp.transpose(lin_acc) @ y_lin) - wp.transpose(rot_acc) @ y_rot)
  # R, Iw and s rotate together for a rigid subtree. Only the linear block changes
  # in world coordinates; the angular block is invariant in the body's frame.
  adj_rotation = wp.transpose(R) @ wp.cross(y_lin, -dt * mass * (lin @ acceleration))
  for r in range(3):
    adj_qvel_out[worldid, dofadr + 3 + r] += adj_velocity[r]
    adj_dof_out[worldid, dofadr + 3 + r] += adj_rotation[r]


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
  worldid, dofid = wp.tid()
  value = float(0.0)
  for siteid in range(nsite):
    jacp, _ = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      site_xpos_in[worldid, siteid],
      site_bodyid[siteid],
      dofid,
      worldid,
    )
    value += wp.dot(jacp, adj_site[worldid, siteid])
  adj_dof_out[worldid, dofid] += value


@wp.kernel(enable_backward=False)
def _body_xpos_vjp(
  # Model:
  nbody: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_xpos: wp.array2d[wp.vec3],
  # Out:
  adj_dof_out: wp.array2d[float],
):
  w, k = wp.tid()
  value = float(0.0)
  for body in range(1, nbody):
    jacp, _ = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      xpos_in[w, body],
      body,
      k,
      w,
    )
    value += wp.dot(jacp, adj_xpos[w, body])
  adj_dof_out[w, k] += value


@wp.kernel(enable_backward=False)
def _body_xquat_vjp(
  # Model:
  nbody: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_xquat: wp.array2d[wp.quat],
  # Out:
  adj_dof_out: wp.array2d[float],
):
  w, k = wp.tid()
  value = float(0.0)
  for body in range(1, nbody):
    _, jacr = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      xpos_in[w, body],
      body,
      k,
      w,
    )
    # World-frame angular velocity multiplies the wxyz orientation on the left.
    dq = 0.5 * math.mul_quat(wp.quat(0.0, jacr[0], jacr[1], jacr[2]), xquat_in[w, body])
    value += wp.dot(dq, adj_xquat[w, body])
  adj_dof_out[w, k] += value


@event_scope
def advance_backward(m: Model, d: Data, d_out: Data, bc: BackwardContext):
  """Maps integrator cotangents, deferring mass-position dependence to smooth_qpos_backward."""
  nworld, nv = d.nworld, m.nv
  adj_qpos, adj_qvel, adj_qacc = bc.adj_qpos, bc.adj_qvel, bc.adj_qacc

  # Match forward.implicit's IMPLICITFAST branch and forward.euler.
  flags = DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER
  implicitfast = m.opt.integrator == IntegratorType.IMPLICITFAST and (m.opt.disableflags & flags) != flags
  eulerdamp = m.opt.integrator == IntegratorType.EULER and not (m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER))
  qacc_advance = d_out.qacc

  if implicitfast or eulerdamp:
    wp.launch(
      forward._compute_damping_deriv,
      dim=(nworld, nv),
      inputs=[
        m.jnt_actuatorid,
        m.dof_jntid,
        m.dof_damping,
        m.dof_dampingpoly,
        m.actuator_trntype,
        m.actuator_trnid,
        m.actuator_damping,
        m.actuator_dampingpoly,
        m.actuator_gear,
        d.qvel,
      ],
      outputs=[bc.damp_deriv],
    )

  if implicitfast:
    linearization = dataclasses.replace(d_out, qvel=d.qvel, ctrl=d.ctrl)
    derivative.deriv_smooth_vel(m, linearization, bc.qDeriv)
    wp.launch(
      forward._implicit_free_body_reset_m,
      dim=(nworld, m.body_freeadr.size),
      inputs=[m.body_dofadr, m.M_rownnz, m.M_rowadr, m.body_freeadr, d_out.M],
      outputs=[bc.qDeriv],
    )
    qLD, qLDiagInv, qacc_advance = bc.qLD, bc.qLDiagInv, bc.qacc_advance
    smooth.factor_solve_i(m, d_out, bc.qDeriv, qLD, qLDiagInv, qacc_advance, d_out.efc.Ma)
    forward.implicit_free_body(m, linearization, d_out.efc.Ma, qacc_advance, solve=True)
  elif eulerdamp:
    wp.copy(bc.qDeriv, d_out.M)
    wp.launch(
      forward._euler_damp_qfrc,
      dim=(nworld, nv),
      inputs=[m.opt.timestep, m.M_rownnz, m.M_rowadr, bc.damp_deriv],
      outputs=[bc.qDeriv],
    )
    support.mul_m(m, d_out, bc.Ma, d_out.qacc)
    qLD, qLDiagInv, qacc_advance = bc.qLD, bc.qLDiagInv, bc.qacc_advance
    smooth.factor_solve_i(m, d_out, bc.qDeriv, qLD, qLDiagInv, qacc_advance, bc.Ma)

  pairs = [
    (m.opt.timestep, None),
    (m.jnt_type, None),
    (m.jnt_qposadr, None),
    (m.jnt_dofadr, None),
    (d.qpos, adj_qpos),
    (d.qvel, adj_qvel),
    (qacc_advance, adj_qacc),
  ]
  adjoint_util._launch_vjp(
    _advance_state,
    (nworld, m.njnt),
    pairs,
    [d_out.qpos, d_out.qvel],
    [d_out.qpos.grad, d_out.qvel.grad],
  )

  if implicitfast or eulerdamp:
    smooth.solve_LD(m, d_out, qLD, qLDiagInv, bc.y_remap, adj_qacc)
    if implicitfast:
      wp.launch(
        _implicit_free_body_backward,
        dim=(nworld, m.body_freeadr.size),
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          m.body_dofadr,
          m.M_rownnz,
          m.M_rowadr,
          m.M_colind,
          m.body_freeadr,
          d.qvel,
          d_out.xpos,
          d_out.xmat,
          d_out.subtree_com,
          d_out.crb,
          d_out.M,
          bc.damp_deriv,
          qacc_advance,
          adj_qacc,
        ],
        outputs=[bc.y_remap, adj_qvel, bc.adj_dof],
      )
      wp.launch(
        smooth_kinematics_adjoint._dof_to_qpos,
        dim=(nworld, m.njnt),
        inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, bc.adj_dof],
        outputs=[adj_qpos],
      )
    support.mul_m(m, d_out, bc.adj_qacc_root, bc.y_remap)

    # For a_int = Q^-1 M a, the direct mass VJP is y.T @ dM @ (a - a_int).
    # Position and parameter reverses consume this for both implicitfast and Euler damping.
    wp.launch(adjoint_util._sub_cols, dim=(nworld, nv), inputs=[d_out.qacc, qacc_advance], outputs=[bc.w_dir])

    if not (m.opt.disableflags & DisableBit.DAMPER):
      # Q = M + dt*D, where D = -d(force_damper)/dv. Its VJP affects velocity and,
      # when optimized, damping itself. Use a negative seed because qacc = Q^-1 Ma.
      wp.launch(adjoint_util._neg_cols, dim=(nworld, nv), inputs=[bc.y_remap], outputs=[bc.smooth_r.grad])
      pairs = [
        (m.opt.timestep, None),
        (m.jnt_actuatorid, None),
        (m.dof_jntid, None),
        (m.dof_damping, m.dof_damping.grad if m.dof_damping.requires_grad else None),
        (m.dof_dampingpoly, None),
        (m.actuator_trntype, None),
        (m.actuator_trnid, None),
        (m.actuator_damping, None),
        (m.actuator_dampingpoly, None),
        (m.actuator_gear, None),
        (d.qvel, bc.adj_qvel_damp),
        (qacc_advance, None),
      ]
      adjoint_util._launch_vjp(_dampingpoly_Qv_leaf, (nworld, nv), pairs, [bc.dampingpoly_Qv], [bc.smooth_r.grad])
      wp.launch(adjoint_util._accum_cols, dim=(nworld, nv), inputs=[bc.adj_qvel_damp], outputs=[adj_qvel])

    adj_qacc = bc.adj_qacc_root

  wp.launch(adjoint_util._accum_cols, dim=(nworld, nv), inputs=[d_out.qacc.grad], outputs=[adj_qacc])
  return adj_qpos, adj_qvel, adj_qacc


@event_scope
def smooth_qvel_backward(m: Model, d: Data, d_out: Data, lam: wp.array, adj_qvel: wp.array, bc: BackwardContext):
  """Accumulates non-bias force VJPs for velocity, controls, and applied forces."""
  nworld = d.nworld
  linearization = dataclasses.replace(d_out, qvel=d.qvel, ctrl=d.ctrl)
  derivative.deriv_smooth_vel(m, linearization, bc.qDeriv)
  wp.launch(forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, bc.qDeriv], outputs=[bc.qLU])
  # RNE qvel VJP is handled elsewhere via smooth_qpos_backward, avoiding the full Jacobian.
  # derivative.deriv_rne_vel(m, linearization, bc.qLU)
  wp.launch(forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, d_out.M], outputs=[bc.M_D])
  wp.launch(
    _smooth_qvel_vjp,
    dim=(nworld, m.nD),
    inputs=[m.opt.timestep, m.qD_fullm_i, m.qD_fullm_j, bc.qLU, bc.M_D, lam],
    outputs=[adj_qvel],
  )
  if not (m.opt.disableflags & DisableBit.ACTUATION):
    # deriv_smooth_vel includes control clamping but ignores joint-force limits.
    wp.launch(
      _actuator_limit_qvel_vjp,
      dim=(nworld, m.nu),
      inputs=[
        m.jnt_actfrclimited,
        m.jnt_actfrcrange,
        m.dof_jntid,
        m.actuator_gaintype,
        m.actuator_biastype,
        m.actuator_gainprm,
        m.actuator_biasprm,
        m.actuator_forcelimited,
        m.actuator_forcerange,
        m.actuator_ctrllimited,
        m.actuator_ctrlrange,
        d.ctrl,
        d_out.moment_rownnz,
        d_out.moment_rowadr,
        d_out.moment_colind,
        d_out.actuator_moment,
        d_out.actuator_force,
        d_out.qfrc_actuator,
        lam,
        int(m.opt.disableflags) & DisableBit.CLAMPCTRL,
      ],
      outputs=[adj_qvel],
    )
    if d.ctrl.requires_grad:
      wp.launch(
        _smooth_ctrl_vjp,
        dim=(nworld, m.nu),
        inputs=[
          m.opt.timestep,
          m.jnt_actfrclimited,
          m.jnt_actfrcrange,
          m.dof_jntid,
          m.actuator_gaintype,
          m.actuator_gainprm,
          m.actuator_forcelimited,
          m.actuator_forcerange,
          m.actuator_ctrllimited,
          m.actuator_ctrlrange,
          d.ctrl,
          d_out.actuator_length,
          d_out.moment_rownnz,
          d_out.moment_rowadr,
          d_out.moment_colind,
          d_out.actuator_moment,
          d_out.actuator_velocity,
          d_out.actuator_force,
          d_out.qfrc_actuator,
          lam,
          bc.qacc_advance,
          bc.y_remap,
          m.opt.integrator == IntegratorType.IMPLICITFAST,
          int(m.opt.disableflags) & DisableBit.CLAMPCTRL,
        ],
        outputs=[d.ctrl.grad],
      )
  if d.qfrc_applied.requires_grad:
    # fwd_acceleration adds qfrc_applied directly to qfrc_smooth.
    wp.launch(adjoint_util._accum_cols, dim=(nworld, m.nv), inputs=[lam], outputs=[d.qfrc_applied.grad])
  if d.xfrc_applied.requires_grad:
    wp.launch(
      _xfrc_applied_vjp,
      dim=(nworld, m.nbody),
      inputs=[
        m.nv,
        m.body_parentid,
        m.body_rootid,
        m.dof_bodyid,
        m.body_isdofancestor,
        d_out.xipos,
        d_out.subtree_com,
        d_out.cdof,
        lam,
      ],
      outputs=[d.xfrc_applied.grad],
    )


@event_scope
def smooth_qpos_backward(m: Model, d: Data, d_out: Data, lam: wp.array, res_qpos: wp.array, bc: BackwardContext):
  """Shares the smooth position reverse with inertial parameters, including direct mass terms."""
  smooth_adjoint.assert_smooth_supported(m)
  flags = DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER
  with_mass = (m.opt.integrator == IntegratorType.IMPLICITFAST and (m.opt.disableflags & flags) != flags) or (
    m.opt.integrator == IntegratorType.EULER and not (m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER))
  )
  # Bound cooperative RNE scratch to eight 128-element spatial-vector tiles.
  if d.qpos.device.is_cuda and m.nbody <= 128:
    linearization = dataclasses.replace(d_out, qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl, act=d.act)
    residual = smooth_adjoint.smooth_force_channels_backward(
      m, linearization, lam, bc.y_remap if with_mass else None, bc.w_dir if with_mass else None, bc, xfrc_applied=d.xfrc_applied
    )
  else:
    if with_mass:
      direct = smooth_adjoint.mass_matrix_qpos_vjp(m, d, bc.y_remap, bc.w_dir, bc=bc)
      wp.launch(adjoint_util._sub_cols, dim=res_qpos.shape, inputs=[res_qpos, direct], outputs=[res_qpos])
      model_adjoint.inertia_param_vjp(m, bc.scratch, bc, scale=1.0)
    linearization = smooth_adjoint.linearize(m, d, d_out, bc)
    residual = smooth_adjoint.smooth_force_backward(m, linearization, lam, bc=bc, xfrc_applied=d.xfrc_applied)
  wp.launch(adjoint_util._accum_cols, dim=residual.shape, inputs=[residual], outputs=[res_qpos])
  # Consume the inertia cotangents before gravcomp reuses the workspace.
  model_adjoint.inertia_param_vjp(m, linearization, bc)
  for vjp in (
    smooth_adjoint.spring_qpos_vjp,
    smooth_adjoint.actuator_qpos_vjp,
    smooth_adjoint.gravcomp_qpos_vjp,
  ):
    residual = vjp(m, linearization, lam, bc=bc)
    wp.launch(adjoint_util._accum_cols, dim=residual.shape, inputs=[residual], outputs=[res_qpos])
  # gravcomp_qpos_vjp also computed the mass residual; apply the IFT minus sign.
  if m.body_mass.requires_grad:
    wp.launch(
      adjoint_util._sub_cols,
      dim=m.body_mass.shape,
      inputs=[m.body_mass.grad, bc.res_body_mass],
      outputs=[m.body_mass.grad],
    )


@event_scope
def fwd_kinematics_backward(m: Model, d: Data, bc: BackwardContext):
  """Maps site and body-pose cotangents to generalized position coordinates."""
  wp.copy(bc.scratch.qpos, d.qpos)
  if m.nmocap:
    wp.copy(bc.scratch.mocap_pos, d.mocap_pos)
    wp.copy(bc.scratch.mocap_quat, d.mocap_quat)
  smooth.kinematics(m, bc.scratch)
  smooth.com_pos(m, bc.scratch)
  bc.adj_dof.zero_()
  if d.site_xpos.grad is not None:
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
        bc.scratch.site_xpos,
        bc.scratch.subtree_com,
        bc.scratch.cdof,
        d.site_xpos.grad,
      ],
      outputs=[bc.adj_dof],
    )
  body_inputs = [m.nbody, m.body_parentid, m.body_rootid, m.dof_bodyid, m.body_isdofancestor]
  if d.xpos.grad is not None:
    wp.launch(
      _body_xpos_vjp,
      dim=(d.nworld, m.nv),
      inputs=[*body_inputs, bc.scratch.xpos, bc.scratch.subtree_com, bc.scratch.cdof, d.xpos.grad],
      outputs=[bc.adj_dof],
    )
  if d.xquat.grad is not None:
    wp.launch(
      _body_xquat_vjp,
      dim=(d.nworld, m.nv),
      inputs=[*body_inputs, bc.scratch.xpos, bc.scratch.xquat, bc.scratch.subtree_com, bc.scratch.cdof, d.xquat.grad],
      outputs=[bc.adj_dof],
    )
  wp.launch(
    smooth_kinematics_adjoint._dof_to_qpos,
    dim=(d.nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, bc.adj_dof],
    outputs=[d.qpos.grad],
  )
