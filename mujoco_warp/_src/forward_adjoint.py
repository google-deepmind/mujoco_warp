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
from mujoco_warp._src.warp_util import cache_kernel
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
  actuator_actadr: wp.array[int],
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
  if actuator_actadr[actuatorid] >= 0:
    return  # Activation dynamics route force cotangents through _activation_vjp.
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


@wp.func
def _activation_objective(
  # In:
  control: float,
  activation: float,
  timestep: float,
  dyntype: int,
  dynprm: vec10,
  actrange: wp.vec2,
  actlimited: bool,
  actearly: bool,
  ctrlrange: wp.vec2,
  ctrllimited: bool,
  flags: int,
  force_seed: float,
  state_seed: float,
  rate_seed: float,
):
  if ctrllimited and not (flags & DisableBit.CLAMPCTRL):
    # Match the stateless path's inward derivative at either control bound.
    if control < ctrlrange[0]:
      control = ctrlrange[0]
    elif control > ctrlrange[1]:
      control = ctrlrange[1]
  rate = float(0.0)
  if not (flags & DisableBit.ACTUATION):
    rate = forward._scalar_act_dot(dyntype, dynprm, control, activation)
  next_activation = support.next_act(timestep, dyntype, dynprm, actrange, activation, rate, 1.0, actlimited)
  force_activation = next_activation if actearly else activation
  return force_seed * force_activation + state_seed * next_activation + rate_seed * rate


@wp.kernel(enable_backward=False)
def _activation_vjp(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  jnt_actfrclimited: wp.array[bool],
  jnt_actfrcrange: wp.array2d[wp.vec2],
  dof_jntid: wp.array[int],
  actuator_dyntype: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_dynprm: wp.array2d[vec10],
  actuator_gainprm: wp.array2d[vec10],
  actuator_actlimited: wp.array[bool],
  actuator_actrange: wp.array2d[wp.vec2],
  actuator_actearly: wp.array[bool],
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  qfrc_actuator_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  adj_act: wp.array2d[float],
  adj_act_dot: wp.array2d[float],
  # Out:
  adj_act_out: wp.array2d[float],
  adj_ctrl_out: wp.array2d[float],
):
  w, uid = wp.tid()
  adr = actuator_actadr[uid]
  if adr < 0:
    return
  force_seed = float(0.0)
  force_active = not (opt_disableflags & DisableBit.ACTUATION)
  if actuator_forcelimited[uid]:
    bounds = actuator_forcerange[w % actuator_forcerange.shape[0], uid]
    force = actuator_force_in[w, uid]
    force_active = force_active and force > bounds[0] and force < bounds[1]
  if force_active:
    moment_seed, _ = smooth_adjoint._actuator_moment_dots(
      jnt_actfrclimited,
      jnt_actfrcrange,
      dof_jntid,
      moment_colind_in,
      actuator_moment_in,
      qfrc_actuator_in,
      lam,
      w,
      moment_rowadr_in[w, uid],
      moment_rownnz_in[w, uid],
    )
    force_seed = actuator_gainprm[w % actuator_gainprm.shape[0], uid][0] * moment_seed
  # Force limits suppress force derivatives, but never the activation update itself.
  gradients = wp.grad(_activation_objective)(
    ctrl_in[w, uid],
    act_in[w, adr],
    opt_timestep[w % opt_timestep.shape[0]],
    actuator_dyntype[uid],
    actuator_dynprm[w % actuator_dynprm.shape[0], uid],
    actuator_actrange[w % actuator_actrange.shape[0], uid],
    actuator_actlimited[uid],
    actuator_actearly[uid],
    actuator_ctrlrange[w % actuator_ctrlrange.shape[0], uid],
    actuator_ctrllimited[uid],
    opt_disableflags,
    force_seed,
    adj_act[w, adr],
    adj_act_dot[w, adr],
  )
  if adj_ctrl_out:
    adj_ctrl_out[w, uid] += gradients[0]
  if adj_act_out:
    adj_act_out[w, adr] += gradients[1]


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


@wp.func
def _body_wrench(
  # In:
  rotation: wp.quat,
  adj_xpos: wp.array2d[wp.vec3],
  adj_xquat: wp.array2d[wp.quat],
  worldid: int,
  bodyid: int,
) -> wp.spatial_vector:
  """Converts Cartesian position/orientation cotangents to a wrench about the body origin."""
  force, torque = wp.vec3(), wp.vec3()
  if adj_xpos:
    force = adj_xpos[worldid, bodyid]
  if adj_xquat:
    quat_seed = math.mul_quat(adj_xquat[worldid, bodyid], math.quat_inv(rotation))
    torque = 0.5 * wp.vec3(quat_seed[1], quat_seed[2], quat_seed[3])
  return wp.spatial_vector(torque, force)


@cache_kernel
def _kinematics_vjp(nbody: int, shared: bool):
  """Projects joint and mocap subtree wrenches.

  The shared path reuses body cotangents across GPU threads for performance. The non-shared
  path avoids cooperative-thread overhead on CPU and handles GPU trees larger than 512 bodies.
  """
  tile_size = max(32, 1 << (nbody - 1).bit_length())

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    njnt: int,
    nsite: int,
    jnt_type: wp.array[int],
    jnt_qposadr: wp.array[int],
    jnt_bodyid: wp.array[int],
    site_bodyid: wp.array[int],
    site_pos: wp.array2d[wp.vec3],
    body_subtreenum: wp.array[int],
    mocap_bodyid: wp.array[int],
    # Data in:
    qpos_in: wp.array2d[float],
    mocap_quat_in: wp.array2d[wp.quat],
    xpos_in: wp.array2d[wp.vec3],
    xquat_in: wp.array2d[wp.quat],
    xanchor_in: wp.array2d[wp.vec3],
    xaxis_in: wp.array2d[wp.vec3],
    # In:
    ninput: int,
    adj_site_xpos: wp.array2d[wp.vec3],
    adj_xpos: wp.array2d[wp.vec3],
    adj_xquat: wp.array2d[wp.quat],
    # Out:
    adj_qpos_out: wp.array2d[float],
    adj_mocap_pos_out: wp.array2d[wp.vec3],
    adj_mocap_quat_out: wp.array2d[wp.quat],
  ):
    worldid, lane = wp.tid()
    site_worldid = worldid % site_pos.shape[0]
    if wp.static(shared):
      position, wrench = wp.vec3(), wp.spatial_vector()
      if lane < wp.static(nbody):
        position = xpos_in[worldid, lane]
        rotation = xquat_in[worldid, lane]
        wrench = _body_wrench(rotation, adj_xpos, adj_xquat, worldid, lane)
        if adj_site_xpos:
          for siteid in range(nsite):
            if site_bodyid[siteid] == lane:
              seed = adj_site_xpos[worldid, siteid]
              offset = math.rot_vec_quat(site_pos[site_worldid, siteid], rotation)
              wrench += wp.spatial_vector(wp.cross(offset, seed), seed)
      # Publish body-local cotangents once for all joint and mocap projections.
      positions = wp.tile_reshape(wp.tile(position, preserve_type=True), shape=(wp.static(tile_size),))
      wrenches = wp.tile_reshape(wp.tile(wrench, preserve_type=True), shape=(wp.static(tile_size),))

    for inputid in range(lane, ninput, wp.block_dim()):
      if inputid < njnt:
        bodyid = jnt_bodyid[inputid]
        origin = xanchor_in[worldid, inputid]
      else:
        bodyid = mocap_bodyid[inputid - njnt]
        origin = xpos_in[worldid, bodyid]
      subtree_end = bodyid + body_subtreenum[bodyid]
      force, torque = wp.vec3(), wp.vec3()
      for child in range(bodyid, subtree_end):
        if wp.static(shared):
          position = positions[child]
          wrench = wrenches[child]
        else:
          position = xpos_in[worldid, child]
          wrench = _body_wrench(xquat_in[worldid, child], adj_xpos, adj_xquat, worldid, child)
        child_force = wp.spatial_bottom(wrench)
        force += child_force
        torque += wp.spatial_top(wrench) + wp.cross(position - origin, child_force)
      if wp.static(not shared):
        # Read sites once per input, rather than once per descendant body.
        if adj_site_xpos:
          for siteid in range(nsite):
            child = site_bodyid[siteid]
            if child >= bodyid and child < subtree_end:
              seed = adj_site_xpos[worldid, siteid]
              offset = math.rot_vec_quat(site_pos[site_worldid, siteid], xquat_in[worldid, child])
              force += seed
              torque += wp.cross(xpos_in[worldid, child] - origin + offset, seed)
      if inputid < njnt:
        qposadr = jnt_qposadr[inputid]
        jnt_type_ = jnt_type[inputid]
        if jnt_type_ == JointType.FREE or jnt_type_ == JointType.BALL:
          if jnt_type_ == JointType.FREE:
            for i in range(3):
              adj_qpos_out[worldid, qposadr + i] = force[i]
            qposadr += 3
          torque_local = math.rot_vec_quat(torque, math.quat_inv(xquat_in[worldid, bodyid]))
          # Lift the joint-local angular cotangent into quaternion coordinates.
          qpos_quat = wp.quat(
            qpos_in[worldid, qposadr],
            qpos_in[worldid, qposadr + 1],
            qpos_in[worldid, qposadr + 2],
            qpos_in[worldid, qposadr + 3],
          )
          quat_grad = 2.0 * math.quat_mul_axis(qpos_quat, torque_local)
          for i in range(4):
            adj_qpos_out[worldid, qposadr + i] = quat_grad[i]
        elif jnt_type_ == JointType.SLIDE:
          adj_qpos_out[worldid, qposadr] = wp.dot(xaxis_in[worldid, inputid], force)
        else:
          adj_qpos_out[worldid, qposadr] = wp.dot(xaxis_in[worldid, inputid], torque)
      else:
        mocapid = inputid - njnt
        adj_mocap_pos_out[worldid, mocapid] = force
        adj_mocap_quat_out[worldid, mocapid] = smooth_kinematics_adjoint._mocap_quat_vjp(
          mocap_quat_in[worldid, mocapid], torque
        )

  return kernel


@event_scope
def advance_backward(m: Model, d_out: Data, d_bwd: Data, bc: BackwardContext):
  """Maps integrator cotangents, deferring mass-position dependence to smooth_qpos_backward."""
  Ma = d_out.efc.Ma
  nworld, nv = d_bwd.nworld, m.nv
  adj_qpos, adj_qvel, adj_qacc = bc.adj_qpos, bc.adj_qvel, bc.adj_qacc

  # Match forward.implicit's IMPLICITFAST branch and forward.euler.
  flags = DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER
  implicitfast = m.opt.integrator == IntegratorType.IMPLICITFAST and (m.opt.disableflags & flags) != flags
  eulerdamp = m.opt.integrator == IntegratorType.EULER and not (m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER))
  qacc_advance = d_bwd.qacc

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
        d_bwd.qvel,
      ],
      outputs=[bc.damp_deriv],
    )

  if implicitfast:
    derivative.deriv_smooth_vel(m, d_bwd, bc.qDeriv)
    wp.launch(
      forward._implicit_free_body_reset_m,
      dim=(nworld, m.body_freeadr.size),
      inputs=[m.body_dofadr, m.M_rownnz, m.M_rowadr, m.body_freeadr, d_bwd.M],
      outputs=[bc.qDeriv],
    )
    qLD, qLDiagInv, qacc_advance = bc.qLD, bc.qLDiagInv, bc.qacc_advance
    smooth.factor_solve_i(m, d_bwd, bc.qDeriv, qLD, qLDiagInv, qacc_advance, Ma)
    forward.implicit_free_body(m, d_bwd, Ma, qacc_advance, solve=True)
  elif eulerdamp:
    wp.copy(bc.qDeriv, d_bwd.M)
    wp.launch(
      forward._euler_damp_qfrc,
      dim=(nworld, nv),
      inputs=[m.opt.timestep, m.M_rownnz, m.M_rowadr, bc.damp_deriv],
      outputs=[bc.qDeriv],
    )
    support.mul_m(m, d_bwd, bc.Ma, d_bwd.qacc)
    qLD, qLDiagInv, qacc_advance = bc.qLD, bc.qLDiagInv, bc.qacc_advance
    smooth.factor_solve_i(m, d_bwd, bc.qDeriv, qLD, qLDiagInv, qacc_advance, bc.Ma)

  pairs = [
    (m.opt.timestep, None),
    (m.jnt_type, None),
    (m.jnt_qposadr, None),
    (m.jnt_dofadr, None),
    (d_bwd.qpos, adj_qpos),
    (d_bwd.qvel, adj_qvel),
    (qacc_advance, adj_qacc),
  ]
  adjoint_util._launch_vjp(
    _advance_state,
    (nworld, m.njnt),
    pairs,
    # next_position writes through its output array during the reverse replay.
    [bc.scratch.qpos, bc.scratch.qvel],
    [d_out.qpos.grad, d_out.qvel.grad],
  )

  if implicitfast or eulerdamp:
    smooth.solve_LD(m, d_bwd, qLD, qLDiagInv, bc.y_remap, adj_qacc)
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
          d_bwd.qvel,
          d_bwd.xpos,
          d_bwd.xmat,
          d_bwd.subtree_com,
          d_bwd.crb,
          d_bwd.M,
          bc.damp_deriv,
          qacc_advance,
          adj_qacc,
        ],
        outputs=[bc.y_remap, adj_qvel, bc.adj_dof],
      )
      model_adjoint.implicitfast_inertia_vjp(m, d_bwd, qacc_advance, bc.y_remap)
      wp.launch(
        smooth_kinematics_adjoint._dof_to_qpos,
        dim=(nworld, m.njnt),
        inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d_bwd.qpos, bc.adj_dof],
        outputs=[adj_qpos],
      )
    support.mul_m(m, d_bwd, bc.adj_qacc_root, bc.y_remap)

    # For a_int = Q^-1 M a, the direct mass VJP is y.T @ dM @ (a - a_int).
    # Position and parameter reverses consume this for both implicitfast and Euler damping.
    wp.launch(adjoint_util._sub_cols, dim=(nworld, nv), inputs=[d_bwd.qacc, qacc_advance], outputs=[bc.w_dir])

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
        (d_bwd.qvel, bc.adj_qvel_damp),
        (qacc_advance, None),
      ]
      adjoint_util._launch_vjp(_dampingpoly_Qv_leaf, (nworld, nv), pairs, [bc.dampingpoly_Qv], [bc.smooth_r.grad])
      wp.launch(adjoint_util._accum_cols, dim=(nworld, nv), inputs=[bc.adj_qvel_damp], outputs=[adj_qvel])

    adj_qacc = bc.adj_qacc_root

  wp.launch(adjoint_util._accum_cols, dim=(nworld, nv), inputs=[d_out.qacc.grad], outputs=[adj_qacc])
  return adj_qpos, adj_qvel, adj_qacc


@event_scope
def smooth_qvel_backward(m: Model, d: Data, d_bwd: Data, lam: wp.array, adj_qvel: wp.array, bc: BackwardContext):
  """Accumulates non-bias force VJPs for velocity, controls, and applied forces."""
  nworld = d.nworld
  derivative.deriv_smooth_vel(m, d_bwd, bc.qDeriv)
  wp.launch(forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, bc.qDeriv], outputs=[bc.qLU])
  # RNE qvel VJP is handled elsewhere via smooth_qpos_backward, avoiding the full Jacobian.
  # derivative.deriv_rne_vel(m, d_bwd, bc.qLU)
  wp.launch(forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, d_bwd.M], outputs=[bc.M_D])
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
        d_bwd.moment_rownnz,
        d_bwd.moment_rowadr,
        d_bwd.moment_colind,
        d_bwd.actuator_moment,
        d_bwd.actuator_force,
        d_bwd.qfrc_actuator,
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
          m.actuator_actadr,
          m.actuator_gainprm,
          m.actuator_forcelimited,
          m.actuator_forcerange,
          m.actuator_ctrllimited,
          m.actuator_ctrlrange,
          d.ctrl,
          d_bwd.actuator_length,
          d_bwd.moment_rownnz,
          d_bwd.moment_rowadr,
          d_bwd.moment_colind,
          d_bwd.actuator_moment,
          d_bwd.actuator_velocity,
          d_bwd.actuator_force,
          d_bwd.qfrc_actuator,
          lam,
          bc.qacc_advance,
          bc.y_remap,
          m.opt.integrator == IntegratorType.IMPLICITFAST,
          int(m.opt.disableflags) & DisableBit.CLAMPCTRL,
        ],
        outputs=[d.ctrl.grad],
      )
  if m.na:
    wp.launch(
      _activation_vjp,
      dim=(nworld, m.nactuator),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.jnt_actfrclimited,
        m.jnt_actfrcrange,
        m.dof_jntid,
        m.actuator_dyntype,
        m.actuator_actadr,
        m.actuator_dynprm,
        m.actuator_gainprm,
        m.actuator_actlimited,
        m.actuator_actrange,
        m.actuator_actearly,
        m.actuator_forcelimited,
        m.actuator_forcerange,
        m.actuator_ctrllimited,
        m.actuator_ctrlrange,
        d.act,
        d.ctrl,
        d_bwd.moment_rownnz,
        d_bwd.moment_rowadr,
        d_bwd.moment_colind,
        d_bwd.actuator_moment,
        d_bwd.actuator_force,
        d_bwd.qfrc_actuator,
        lam,
        d_bwd.act.grad,
        d_bwd.act_dot.grad,
      ],
      outputs=[d.act.grad, d.ctrl.grad],
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
        d_bwd.xipos,
        d_bwd.subtree_com,
        d_bwd.cdof,
        lam,
      ],
      outputs=[d.xfrc_applied.grad],
    )


@event_scope
def smooth_qpos_backward(m: Model, d: Data, d_bwd: Data, lam: wp.array, res_qpos: wp.array, bc: BackwardContext):
  """Shares the smooth position reverse with inertial parameters, including direct mass terms."""
  mocap_grad = d.mocap_pos.grad is not None or d.mocap_quat.grad is not None
  flags = DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER
  with_mass = (m.opt.integrator == IntegratorType.IMPLICITFAST and (m.opt.disableflags & flags) != flags) or (
    m.opt.integrator == IntegratorType.EULER and not (m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER))
  )
  # Bound cooperative RNE scratch to eight 128-element spatial-vector tiles.
  if d.qpos.device.is_cuda and m.nbody <= 128:
    residual = smooth_adjoint.smooth_force_channels_backward(
      m, d_bwd, lam, bc.y_remap if with_mass else None, bc.w_dir if with_mass else None, bc, xfrc_applied=d.xfrc_applied
    )
  else:
    if with_mass:
      direct = smooth_adjoint.mass_matrix_qpos_vjp(m, d, bc.y_remap, bc.w_dir, bc=bc)
      if mocap_grad:
        smooth_kinematics_adjoint.kinematics_mocap_backward(m, bc.scratch, bc, scale=1.0)
      wp.launch(adjoint_util._sub_cols, dim=res_qpos.shape, inputs=[res_qpos, direct], outputs=[res_qpos])
      model_adjoint.inertia_param_vjp(m, bc.scratch, bc, scale=1.0)
    smooth.rne(m, d_bwd, flg_acc=True)
    residual = smooth_adjoint.smooth_force_backward(m, d_bwd, lam, bc=bc, xfrc_applied=d.xfrc_applied)
  wp.launch(adjoint_util._accum_cols, dim=residual.shape, inputs=[residual], outputs=[res_qpos])
  if mocap_grad:
    smooth_kinematics_adjoint.kinematics_mocap_backward(m, d_bwd, bc)
  # Consume the inertia cotangents before gravcomp reuses the workspace.
  model_adjoint.inertia_param_vjp(m, d_bwd, bc)
  for vjp in (
    smooth_adjoint.spring_qpos_vjp,
    smooth_adjoint.actuator_qpos_vjp,
    smooth_adjoint.gravcomp_qpos_vjp,
  ):
    residual = vjp(m, d_bwd, lam, bc=bc)
    wp.launch(adjoint_util._accum_cols, dim=residual.shape, inputs=[residual], outputs=[res_qpos])
  # Gravcomp leaves mass and COM cotangents in the same workspace.
  if mocap_grad:
    smooth_kinematics_adjoint.kinematics_mocap_backward(m, d_bwd, bc)
  model_adjoint.inertia_param_vjp(m, d_bwd, bc)


@event_scope
def kinematics_backward(m: Model, d_out: Data, d_bwd: Data, bc: BackwardContext):
  """Maps kinematic output gradients to qpos and mocap inputs."""
  mocap_grad = d_bwd.mocap_pos.grad is not None or d_bwd.mocap_quat.grad is not None
  ninput = m.njnt + (m.nmocap if mocap_grad else 0)
  # Bound shared storage; the same wrench contraction also handles CPU and large trees.
  shared = d_bwd.qpos.device.is_cuda and m.nbody <= 512
  block_dim = max(32, 1 << (m.nbody - 1).bit_length()) if shared else 128
  wp.launch_tiled(
    _kinematics_vjp(m.nbody, shared),
    dim=d_bwd.nworld,
    block_dim=block_dim,
    inputs=[
      m.njnt,
      m.nsite,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_bodyid,
      m.site_bodyid,
      m.site_pos,
      m.body_subtreenum,
      m.mocap_bodyid,
      d_bwd.qpos,
      d_bwd.mocap_quat,
      d_bwd.xpos,
      d_bwd.xquat,
      d_bwd.xanchor,
      d_bwd.xaxis,
      ninput,
      d_out.site_xpos.grad,
      d_out.xpos.grad,
      d_out.xquat.grad,
    ],
    outputs=[bc.adj_qpos, bc.adj_mocap_pos, bc.adj_mocap_quat],
  )
