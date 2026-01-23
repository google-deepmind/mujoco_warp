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

import warp as wp

from mujoco_warp._src import support
from mujoco_warp._src import util_misc
from mujoco_warp._src.passive import geom_semiaxes
from mujoco_warp._src.support import next_act
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import mat66
from mujoco_warp._src.types import vec10f
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _qderiv_actuator_passive_vel(
  # Model:
  opt_timestep: wp.array[float],
  actuator_dyntype: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_actnum: wp.array[int],
  actuator_forcelimited: wp.array[bool],
  actuator_actlimited: wp.array[bool],
  actuator_dynprm: wp.array2d[vec10f],
  actuator_gainprm: wp.array2d[vec10f],
  actuator_biasprm: wp.array2d[vec10f],
  actuator_actearly: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_actrange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  # Out:
  vel_out: wp.array2d[float],
):
  worldid, actid = wp.tid()

  actuator_gainprm_id = worldid % actuator_gainprm.shape[0]
  actuator_biasprm_id = worldid % actuator_biasprm.shape[0]

  if actuator_gaintype[actid] == GainType.AFFINE:
    gain = actuator_gainprm[actuator_gainprm_id, actid][2]
  else:
    gain = 0.0

  if actuator_biastype[actid] == BiasType.AFFINE:
    bias = actuator_biasprm[actuator_biasprm_id, actid][2]
  elif actuator_biastype[actid] == BiasType.DCMOTOR:
    dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], actid]
    te = dynprm[0]
    if te <= 0.0:
      gainprm = actuator_gainprm[actuator_gainprm_id, actid]
      R = gainprm[0]
      K = gainprm[1]

      slots = util_misc.dcmotor_slots(dynprm, gainprm)
      slot_Ta = slots[2]

      if slot_Ta >= 0:
        adr = actuator_actadr[actid] + slot_Ta
        T = act_in[worldid, adr]
        alpha = gainprm[2]
        T0 = gainprm[3]
        Ta = dynprm[4]
        R *= 1.0 + alpha * (T + Ta - T0)

      bias = -K * K / wp.max(MJ_MINVAL, R)
    else:
      bias = 0.0
  else:
    bias = 0.0

  if bias == 0.0 and gain == 0.0:
    vel_out[worldid, actid] = 0.0
    return

  # skip if force is clamped by forcerange
  if actuator_forcelimited[actid]:
    force = actuator_force_in[worldid, actid]
    forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]
    if force <= forcerange[0] or force >= forcerange[1]:
      vel_out[worldid, actid] = 0.0
      return

  vel = float(bias)
  if actuator_dyntype[actid] != DynType.NONE:
    if gain != 0.0:
      act_adr = actuator_actadr[actid] + actuator_actnum[actid] - 1

      # use next activation if actearly is set (matching forward pass)
      if actuator_actearly[actid]:
        act = next_act(
          opt_timestep[worldid % opt_timestep.shape[0]],
          actuator_dyntype[actid],
          actuator_dynprm[worldid % actuator_dynprm.shape[0], actid],
          actuator_actrange[worldid % actuator_actrange.shape[0], actid],
          act_in[worldid, act_adr],
          act_dot_in[worldid, act_adr],
          1.0,
          actuator_actlimited[actid],
        )
      else:
        act = act_in[worldid, act_adr]

      vel += gain * act
  else:
    if gain != 0.0:
      vel += gain * ctrl_in[worldid, actid]

  vel_out[worldid, actid] = vel


@wp.func
def _nonzero_mask(x: float) -> float:
  """Returns 1.0 for non-zero input, 0.0 otherwise."""
  if x != 0.0:
    return 1.0
  return 0.0


@wp.kernel
def _qderiv_actuator_passive_actuation_dense(
  # Model:
  nu: int,
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  # In:
  vel_in: wp.array2d[float],
  qMi: wp.array[int],
  qMj: wp.array[int],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  worldid, elemid = wp.tid()

  dofiid = qMi[elemid]
  dofjid = qMj[elemid]
  qderiv_contrib = float(0.0)
  for actid in range(nu):
    vel = vel_in[worldid, actid]
    if vel == 0.0:
      continue

    # TODO(team): restructure sparse version for better parallelism?
    moment_i = float(0.0)
    moment_j = float(0.0)

    rownnz = moment_rownnz_in[worldid, actid]
    rowadr = moment_rowadr_in[worldid, actid]
    for i in range(rownnz):
      sparseid = rowadr + i
      colind = moment_colind_in[worldid, sparseid]
      if colind == dofiid:
        moment_i = actuator_moment_in[worldid, sparseid]
      if colind == dofjid:
        moment_j = actuator_moment_in[worldid, sparseid]
      if moment_i != 0.0 and moment_j != 0.0:
        break

    if moment_i == 0 and moment_j == 0:
      continue

    qderiv_contrib += moment_i * moment_j * vel

  qDeriv_out[worldid, dofiid, dofjid] = qderiv_contrib
  if dofiid != dofjid:
    qDeriv_out[worldid, dofjid, dofiid] = qderiv_contrib


@wp.kernel
def _qderiv_actuator_passive_actuation_sparse(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  # In:
  vel_in: wp.array2d[float],
  qMj: wp.array[int],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  worldid, actid = wp.tid()

  vel = vel_in[worldid, actid]
  if vel == 0.0:
    return

  rownnz = moment_rownnz_in[worldid, actid]
  rowadr = moment_rowadr_in[worldid, actid]

  for i in range(rownnz):
    rowadri = rowadr + i
    moment_i = actuator_moment_in[worldid, rowadri]
    if moment_i == 0.0:
      continue
    dofi = moment_colind_in[worldid, rowadri]

    for j in range(i + 1):
      rowadrj = rowadr + j
      moment_j = actuator_moment_in[worldid, rowadrj]
      if moment_j == 0.0:
        continue
      dofj = moment_colind_in[worldid, rowadrj]

      contrib = moment_i * moment_j * vel

      # Search the corresponding elemid
      # TODO: This could be precalculated for improved performance
      row = dofi
      col = dofj
      row_startk = M_rowadr[row] - 1
      row_nnz = M_rownnz[row]
      for k in range(row_nnz):
        row_startk += 1
        if qMj[row_startk] == col:
          wp.atomic_add(qDeriv_out[worldid, 0], row_startk, contrib)
          break


@wp.kernel
def _qderiv_actuator_passive(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  is_sparse: bool,
  # Data in:
  qvel_in: wp.array2d[float],
  qM_in: wp.array3d[float],
  # In:
  qMi: wp.array[int],
  qMj: wp.array[int],
  qDeriv_in: wp.array3d[float],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  worldid, elemid = wp.tid()

  dofiid = qMi[elemid]
  dofjid = qMj[elemid]

  if is_sparse:
    qderiv = qDeriv_in[worldid, 0, elemid]
  else:
    qderiv = qDeriv_in[worldid, dofiid, dofjid]

  if not (opt_disableflags & DisableBit.DAMPER) and dofiid == dofjid:
    damping = dof_damping[worldid % dof_damping.shape[0], dofiid]
    dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofiid]
    v = qvel_in[worldid, dofiid]
    qderiv -= util_misc._poly_force_deriv(damping, dpoly, v, 1)

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  if is_sparse:
    qDeriv_out[worldid, 0, elemid] = qM_in[worldid, 0, elemid] - qderiv
  else:
    qM = qM_in[worldid, dofiid, dofjid] - qderiv
    qDeriv_out[worldid, dofiid, dofjid] = qM
    if dofiid != dofjid:
      qDeriv_out[worldid, dofjid, dofiid] = qM


# TODO(team): improve performance with tile operations?
@wp.kernel
def _qderiv_tendon_damping(
  # Model:
  ntendon: int,
  opt_timestep: wp.array[float],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  is_sparse: bool,
  # Data in:
  ten_J_in: wp.array2d[float],
  ten_velocity_in: wp.array2d[float],
  # In:
  qMi: wp.array[int],
  qMj: wp.array[int],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  worldid, elemid = wp.tid()
  dofiid = qMi[elemid]
  dofjid = qMj[elemid]

  qderiv = float(0.0)
  tendon_damping_id = worldid % tendon_damping.shape[0]
  for tenid in range(ntendon):
    damping = tendon_damping[tendon_damping_id, tenid]
    dpoly = tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]
    if damping == 0.0 and dpoly[0] == 0.0 and dpoly[1] == 0.0:
      continue

    rownnz = ten_J_rownnz[tenid]
    rowadr = ten_J_rowadr[tenid]
    Ji = float(0.0)
    Jj = float(0.0)
    for k in range(rownnz):
      if Ji != 0.0 and Jj != 0.0:
        break
      sparseid = rowadr + k
      colind = ten_J_colind[sparseid]
      if colind == dofiid:
        Ji = ten_J_in[worldid, sparseid]
      if colind == dofjid:
        Jj = ten_J_in[worldid, sparseid]

    v = ten_velocity_in[worldid, tenid]
    qderiv -= Ji * Jj * util_misc._poly_force_deriv(damping, dpoly, v, 1)

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  if is_sparse:
    qDeriv_out[worldid, 0, elemid] -= qderiv
  else:
    qDeriv_out[worldid, dofiid, dofjid] -= qderiv
    if dofiid != dofjid:
      qDeriv_out[worldid, dofjid, dofiid] -= qderiv


@wp.func
def _ellipsoid_max_moment_deriv(size: wp.vec3, dir: int) -> float:  # kernel_analyzer: ignore
  d0 = size[dir]
  d1 = size[(dir + 1) % 3]
  d2 = size[(dir + 2) % 3]
  d_max = wp.max(d1, d2)
  return wp.static(8.0 / 15.0 * wp.pi) * d0 * d_max * d_max * d_max * d_max


@wp.func
def _add_to_quadrant(B: mat66, D: wp.mat33, col_quad: int, row_quad: int) -> mat66:
  """Add 3x3 matrix D^T to a quadrant of 6x6 matrix B.

  Args:
    B: The 6x6 matrix to modify
    D: The 3x3 matrix to add (will be transposed)
    col_quad: Column quadrant (0 = cols 0-2, 1 = cols 3-5)
    row_quad: Row quadrant (0 = rows 0-2, 1 = rows 3-5)

  This matches MuJoCo's addToQuadrant(B, D, col_quad, row_quad) convention,
  which adds D^T to the specified quadrant due to column-major storage.
  """
  r = 3 * row_quad
  c = 3 * col_quad
  for i in range(3):
    for j in range(3):
      B[r + i, c + j] += D[j, i]  # Note: D[j,i] transposes D
  return B


@wp.func
def _cross_deriv(a: wp.vec3, b: wp.vec3) -> tuple[wp.mat33, wp.mat33]:  # kernel_analyzer: ignore
  """Returns derivatives of cross product a x b w.r.t. both inputs.

  Returns:
    (deriv_a, deriv_b): Derivatives w.r.t. a and b respectively
  """
  deriv_a = wp.mat33(0.0, b[2], -b[1], -b[2], 0.0, b[0], b[1], -b[0], 0.0)
  deriv_b = wp.mat33(0.0, -a[2], a[1], a[2], 0.0, -a[0], -a[1], a[0], 0.0)
  return deriv_a, deriv_b


@wp.func
def _added_mass_forces_deriv(  # kernel_analyzer: ignore
  local_vels: wp.spatial_vector, fluid_density: float, virtual_mass: wp.vec3, virtual_inertia: wp.vec3
) -> mat66:
  lin_vel = wp.spatial_bottom(local_vels)
  ang_vel = wp.spatial_top(local_vels)

  virtual_lin_mom = fluid_density * wp.cw_mul(virtual_mass, lin_vel)
  virtual_ang_mom = fluid_density * wp.cw_mul(virtual_inertia, ang_vel)

  B = mat66(0.0)

  Da, Db = _cross_deriv(virtual_ang_mom, ang_vel)
  B = _add_to_quadrant(B, Db, 0, 0)
  for i in range(3):
    for j in range(3):
      Da[i, j] *= fluid_density * virtual_inertia[j]
  B = _add_to_quadrant(B, Da, 0, 0)

  Da, Db = _cross_deriv(virtual_lin_mom, lin_vel)
  B = _add_to_quadrant(B, Db, 0, 1)
  for i in range(3):
    for j in range(3):
      Da[i, j] *= fluid_density * virtual_mass[j]
  B = _add_to_quadrant(B, Da, 0, 1)

  Da, Db = _cross_deriv(virtual_lin_mom, ang_vel)
  B = _add_to_quadrant(B, Db, 1, 0)
  for i in range(3):
    for j in range(3):
      Da[i, j] *= fluid_density * virtual_mass[j]
  B = _add_to_quadrant(B, Da, 1, 1)

  return B


@wp.func
def _viscous_torque_deriv(  # kernel_analyzer: ignore
  lvel: wp.spatial_vector,
  fluid_density: float,
  fluid_viscosity: float,
  size: wp.vec3,
  slender_drag_coef: float,
  ang_drag_coef: float,
) -> wp.mat33:
  d_max = wp.max(wp.max(size[0], size[1]), size[2])
  d_min = wp.min(wp.min(size[0], size[1]), size[2])
  d_mid = size[0] + size[1] + size[2] - d_max - d_min

  eq_sphere_D = wp.static(2.0 / 3.0) * (size[0] + size[1] + size[2])
  lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D

  I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * d_max * d_max * d_max * d_max
  II = wp.vec3(_ellipsoid_max_moment_deriv(size, 0), _ellipsoid_max_moment_deriv(size, 1), _ellipsoid_max_moment_deriv(size, 2))

  ang = wp.spatial_top(lvel)
  x = ang[0]
  y = ang[1]
  z = ang[2]

  mom_coef = wp.vec3(
    ang_drag_coef * II[0] + slender_drag_coef * (I_max - II[0]),
    ang_drag_coef * II[1] + slender_drag_coef * (I_max - II[1]),
    ang_drag_coef * II[2] + slender_drag_coef * (I_max - II[2]),
  )

  mom_visc = wp.cw_mul(ang, mom_coef)
  norm = wp.length(mom_visc)
  density = fluid_density / wp.max(MJ_MINVAL, norm)

  mom_sq = -density * wp.cw_mul(wp.cw_mul(ang, mom_coef), mom_coef)

  lin_coef = fluid_viscosity * lin_visc_torq_coef
  diag_val = x * mom_sq[0] + y * mom_sq[1] + z * mom_sq[2] - lin_coef

  D = wp.mat33(
    diag_val + mom_sq[0] * x,
    mom_sq[1] * x,
    mom_sq[2] * x,
    mom_sq[0] * y,
    diag_val + mom_sq[1] * y,
    mom_sq[2] * y,
    mom_sq[0] * z,
    mom_sq[1] * z,
    diag_val + mom_sq[2] * z,
  )

  return D


@wp.func
def _viscous_drag_deriv(  # kernel_analyzer: ignore
  lvel: wp.spatial_vector,
  fluid_density: float,
  fluid_viscosity: float,
  size: wp.vec3,
  blunt_drag_coef: float,
  slender_drag_coef: float,
) -> wp.mat33:
  d_max = wp.max(wp.max(size[0], size[1]), size[2])
  d_min = wp.min(wp.min(size[0], size[1]), size[2])
  d_mid = size[0] + size[1] + size[2] - d_max - d_min

  eq_sphere_D = wp.static(2.0 / 3.0) * (size[0] + size[1] + size[2])
  A_max = wp.pi * d_max * d_mid

  a = (size[1] * size[2]) * (size[1] * size[2])
  b = (size[2] * size[0]) * (size[2] * size[0])
  c = (size[0] * size[1]) * (size[0] * size[1])
  aa = a * a
  bb = b * b
  cc = c * c

  lin = wp.spatial_bottom(lvel)
  x = lin[0]
  y = lin[1]
  z = lin[2]
  xx = x * x
  yy = y * y
  zz = z * z
  xy = x * y
  yz = y * z
  xz = x * z

  proj_denom = aa * xx + bb * yy + cc * zz
  proj_num = a * xx + b * yy + c * zz
  dA_coef = wp.pi / wp.max(MJ_MINVAL, wp.sqrt(proj_num * proj_num * proj_num * proj_denom))

  A_proj = wp.pi * wp.sqrt(proj_denom / wp.max(MJ_MINVAL, proj_num))

  norm = wp.sqrt(xx + yy + zz)
  inv_norm = 1.0 / wp.max(MJ_MINVAL, norm)

  lin_coef = fluid_viscosity * wp.static(3.0 * wp.pi) * eq_sphere_D
  quad_coef = fluid_density * (A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj))
  Aproj_coef = fluid_density * norm * (blunt_drag_coef - slender_drag_coef)

  dAproj_dv = wp.vec3(
    Aproj_coef * dA_coef * a * x * (b * yy * (a - b) + c * zz * (a - c)),
    Aproj_coef * dA_coef * b * y * (a * xx * (b - a) + c * zz * (b - c)),
    Aproj_coef * dA_coef * c * z * (a * xx * (c - a) + b * yy * (c - b)),
  )

  inner = xx + yy + zz
  D = wp.mat33(xx + inner, xy, xz, xy, yy + inner, yz, xz, yz, zz + inner)

  D *= -quad_coef * inv_norm

  for i in range(3):
    D[0, i] -= x * dAproj_dv[i]
    D[1, i] -= y * dAproj_dv[i]
    D[2, i] -= z * dAproj_dv[i]

  D[0, 0] -= lin_coef
  D[1, 1] -= lin_coef
  D[2, 2] -= lin_coef

  return D


@wp.func
def _kutta_lift_deriv(lvel: wp.spatial_vector, fluid_density: float, size: wp.vec3, kutta_lift_coef: float) -> wp.mat33:  # kernel_analyzer: ignore
  a = (size[1] * size[2]) * (size[1] * size[2])
  b = (size[2] * size[0]) * (size[2] * size[0])
  c = (size[0] * size[1]) * (size[0] * size[1])
  aa = a * a
  bb = b * b
  cc = c * c

  lin = wp.spatial_bottom(lvel)
  x = lin[0]
  y = lin[1]
  z = lin[2]
  xx = x * x
  yy = y * y
  zz = z * z
  xy = x * y
  yz = y * z
  xz = x * z

  proj_denom = aa * xx + bb * yy + cc * zz
  proj_num = a * xx + b * yy + c * zz
  norm2 = xx + yy + zz
  df_denom = wp.pi * kutta_lift_coef * fluid_density / wp.max(MJ_MINVAL, wp.sqrt(proj_denom * proj_num * norm2))

  dfx_coef = yy * (a - b) + zz * (a - c)
  dfy_coef = xx * (b - a) + zz * (b - c)
  dfz_coef = xx * (c - a) + yy * (c - b)
  proj_term = proj_num / wp.max(MJ_MINVAL, proj_denom)
  cos_term = proj_num / wp.max(MJ_MINVAL, norm2)

  D = wp.mat33(0.0, b - a, c - a, a - b, 0.0, c - b, a - c, b - c, 0.0)
  D *= wp.static(2.0) * proj_num

  inner_term = (
    wp.cw_mul(wp.vec3(aa, bb, cc), wp.vec3(proj_term, proj_term, proj_term))
    - wp.vec3(a, b, c)
    + wp.vec3(cos_term, cos_term, cos_term)
  )

  for i in range(3):
    D[0, i] += inner_term[i] * dfx_coef
    D[1, i] += inner_term[i] * dfy_coef
    D[2, i] += inner_term[i] * dfz_coef

  D[0, 0] *= xx
  D[0, 1] *= xy
  D[0, 2] *= xz
  D[1, 0] *= xy
  D[1, 1] *= yy
  D[1, 2] *= yz
  D[2, 0] *= xz
  D[2, 1] *= yz
  D[2, 2] *= zz

  D[0, 0] -= dfx_coef * proj_num
  D[1, 1] -= dfy_coef * proj_num
  D[2, 2] -= dfz_coef * proj_num

  return D * df_denom


@wp.func
def _magnus_force_deriv(lvel: wp.spatial_vector, fluid_density: float, size: wp.vec3, magnus_lift_coef: float) -> mat66:  # kernel_analyzer: ignore
  volume = wp.static(4.0 / 3.0 * wp.pi) * size[0] * size[1] * size[2]
  magnus_coef = magnus_lift_coef * fluid_density * volume

  lin_vel = wp.spatial_bottom(lvel)
  ang_vel = wp.spatial_top(lvel)

  lin_vel_scaled = lin_vel * magnus_coef
  ang_vel_scaled = ang_vel * magnus_coef

  D_ang, _ = _cross_deriv(ang_vel_scaled, lin_vel_scaled)
  _, D_lin = _cross_deriv(ang_vel_scaled, lin_vel_scaled)

  B = mat66(0.0)
  B = _add_to_quadrant(B, D_ang, 1, 0)
  B = _add_to_quadrant(B, D_lin, 1, 1)

  return B


@wp.func
def _ellipsoid_fluid_qderiv_contrib(
  # Model:
  opt_density: float,
  opt_viscosity: float,
  opt_wind: wp.vec3,
  opt_integrator: int,
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_geomnum: wp.array(dtype=int),
  body_geomadr: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_fluid: wp.array2d(dtype=float),
  dof_bodyid: wp.array(dtype=int),
  # Data in:
  xipos: wp.vec3,
  ximat: wp.mat33,
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  cvel: wp.spatial_vector,
  # In:
  bodyid: int,
  dofiid: int,
  dofjid: int,
  worldid: int,
) -> float:
  """Compute qDeriv contribution for a single DOF pair from ellipsoid fluid forces."""
  wind = opt_wind
  density = opt_density
  viscosity = opt_viscosity

  rotT = wp.transpose(ximat)
  ang_global = wp.spatial_top(cvel)
  lin_global = wp.spatial_bottom(cvel)
  subtree_root = subtree_com_in[worldid, body_rootid[bodyid]]
  lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)

  qderiv_contrib = float(0.0)

  start = body_geomadr[bodyid]
  count = body_geomnum[bodyid]

  for i in range(count):
    geomid = start + i
    coef = geom_fluid[geomid, 0]
    if coef <= 0.0:
      continue

    size = geom_size[worldid % geom_size.shape[0], geomid]
    semiaxes = geom_semiaxes(size, geom_type[geomid])
    geom_rot = geom_xmat_in[worldid, geomid]
    geom_rotT = wp.transpose(geom_rot)
    geom_pos = geom_xpos_in[worldid, geomid]

    lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)

    l_ang = geom_rotT @ ang_global
    l_lin = geom_rotT @ lin_point

    if wind[0] != 0.0 or wind[1] != 0.0 or wind[2] != 0.0:
      l_lin -= geom_rotT @ wind

    lvel = wp.spatial_vector(l_ang, l_lin)

    B = mat66(0.0)

    magnus_coef = geom_fluid[geomid, 5]
    kutta_coef = geom_fluid[geomid, 4]
    blunt_drag_coef = geom_fluid[geomid, 1]
    slender_drag_coef = geom_fluid[geomid, 2]
    ang_drag_coef = geom_fluid[geomid, 3]

    if density > 0.0:
      virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])
      virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])
      B += _added_mass_forces_deriv(lvel, density, virtual_mass, virtual_inertia)

      if magnus_coef != 0.0:
        B += _magnus_force_deriv(lvel, density, semiaxes, magnus_coef)

      if kutta_coef != 0.0:
        D_kutta = _kutta_lift_deriv(lvel, density, semiaxes, kutta_coef)
        B = _add_to_quadrant(B, D_kutta, 1, 1)

    if viscosity > 0.0:
      D_drag = _viscous_drag_deriv(lvel, density, viscosity, semiaxes, blunt_drag_coef, slender_drag_coef)
      B = _add_to_quadrant(B, D_drag, 1, 1)

      D_torque = _viscous_torque_deriv(lvel, density, viscosity, semiaxes, slender_drag_coef, ang_drag_coef)
      B = _add_to_quadrant(B, D_torque, 0, 0)

    B = B * coef

    # Symmetrize B for implicitfast integrator (matches MuJoCo behavior)
    # Note: MuJoCo only symmetrizes for IMPLICITFAST, not IMPLICIT
    if opt_integrator == 2:  # mjINT_IMPLICITFAST
      for row in range(6):
        for col in range(row + 1, 6):
          avg = (B[row, col] + B[col, row]) * 0.5
          B[row, col] = avg
          B[col, row] = avg

    # Compute Jacobian at geometry position in local frame
    jacp_i, jacr_i = support.jac(
      body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in,
      geom_pos, bodyid, dofiid, worldid
    )
    jacp_i_local = geom_rotT @ jacp_i
    jacr_i_local = geom_rotT @ jacr_i

    jacp_j, jacr_j = support.jac(
      body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in,
      geom_pos, bodyid, dofjid, worldid
    )
    jacp_j_local = geom_rotT @ jacp_j
    jacr_j_local = geom_rotT @ jacr_j

    # B matrix is in (angular, linear) order:
    # B[0:3, 0:3] = angular-angular, B[0:3, 3:6] = angular-linear
    # B[3:6, 0:3] = linear-angular, B[3:6, 3:6] = linear-linear
    # Jacobian vectors: jacr = angular, jacp = linear
    # We compute J^T * B * J where J = [jacr; jacp] (angular first, linear second)
    cdof_i_local = wp.spatial_vector(jacr_i_local, jacp_i_local)
    cdof_j_local = wp.spatial_vector(jacr_j_local, jacp_j_local)

    for k in range(6):
      for j in range(6):
        if k < 3:
          cdof_i_k = wp.spatial_top(cdof_i_local)[k]
        else:
          cdof_i_k = wp.spatial_bottom(cdof_i_local)[k - 3]
        if j < 3:
          cdof_j_j = wp.spatial_top(cdof_j_local)[j]
        else:
          cdof_j_j = wp.spatial_bottom(cdof_j_local)[j - 3]
        qderiv_contrib += cdof_i_k * B[k, j] * cdof_j_j

  return qderiv_contrib


@wp.kernel
def _deriv_ellipsoid_fluid_dense(
  # Model:
  nv: int,
  opt_density: wp.array(dtype=float),
  opt_viscosity: wp.array(dtype=float),
  opt_wind: wp.array(dtype=wp.vec3),
  opt_timestep: wp.array(dtype=float),
  opt_integrator: int,
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_geomnum: wp.array(dtype=int),
  body_geomadr: wp.array(dtype=int),
  body_fluid_ellipsoid: wp.array(dtype=bool),
  geom_type: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_fluid: wp.array2d(dtype=float),
  dof_bodyid: wp.array(dtype=int),
  # Data in:
  xipos_in: wp.array2d(dtype=wp.vec3),
  ximat_in: wp.array2d(dtype=wp.mat33),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  """Dense version: iterates over all nv x nv DOF pairs."""
  worldid, dofiid, dofjid = wp.tid()

  # Only process lower triangle (i >= j) to avoid duplicate work
  if dofiid < dofjid:
    return

  bodyid_i = dof_bodyid[dofiid]
  bodyid_j = dof_bodyid[dofjid]

  if bodyid_i == 0 or not body_fluid_ellipsoid[bodyid_i]:
    return

  if bodyid_i != bodyid_j:
    return

  bodyid = bodyid_i
  density = opt_density[worldid % opt_density.shape[0]]
  viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]
  wind = opt_wind[worldid % opt_wind.shape[0]]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  xipos = xipos_in[worldid, bodyid]
  ximat = ximat_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]

  qderiv_contrib = _ellipsoid_fluid_qderiv_contrib(
    density, viscosity, wind, opt_integrator,
    body_parentid, body_rootid, body_geomnum, body_geomadr,
    geom_type, geom_size, geom_fluid, dof_bodyid,
    xipos, ximat, geom_xpos_in, geom_xmat_in, subtree_com_in, cdof_in, cvel,
    bodyid, dofiid, dofjid, worldid
  )

  qderiv_contrib *= timestep

  wp.atomic_sub(qDeriv_out, worldid, dofiid, dofjid, qderiv_contrib)
  if dofiid != dofjid:
    wp.atomic_sub(qDeriv_out, worldid, dofjid, dofiid, qderiv_contrib)


@wp.kernel
def _deriv_ellipsoid_fluid_sparse(
  # Model:
  opt_density: wp.array(dtype=float),
  opt_viscosity: wp.array(dtype=float),
  opt_wind: wp.array(dtype=wp.vec3),
  opt_timestep: wp.array(dtype=float),
  opt_integrator: int,
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_geomnum: wp.array(dtype=int),
  body_geomadr: wp.array(dtype=int),
  body_fluid_ellipsoid: wp.array(dtype=bool),
  geom_type: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_fluid: wp.array2d(dtype=float),
  dof_bodyid: wp.array(dtype=int),
  # Data in:
  xipos_in: wp.array2d(dtype=wp.vec3),
  ximat_in: wp.array2d(dtype=wp.mat33),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  qMi: wp.array(dtype=int),
  qMj: wp.array(dtype=int),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  """Sparse version: iterates over sparse matrix elements."""
  worldid, elemid = wp.tid()
  dofiid = qMi[elemid]
  dofjid = qMj[elemid]

  bodyid_i = dof_bodyid[dofiid]
  bodyid_j = dof_bodyid[dofjid]

  if bodyid_i == 0 or not body_fluid_ellipsoid[bodyid_i]:
    return

  if bodyid_i != bodyid_j:
    return

  bodyid = bodyid_i
  density = opt_density[worldid % opt_density.shape[0]]
  viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]
  wind = opt_wind[worldid % opt_wind.shape[0]]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  xipos = xipos_in[worldid, bodyid]
  ximat = ximat_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]

  qderiv_contrib = _ellipsoid_fluid_qderiv_contrib(
    density, viscosity, wind, opt_integrator,
    body_parentid, body_rootid, body_geomnum, body_geomadr,
    geom_type, geom_size, geom_fluid, dof_bodyid,
    xipos, ximat, geom_xpos_in, geom_xmat_in, subtree_com_in, cdof_in, cvel,
    bodyid, dofiid, dofjid, worldid
  )

  qderiv_contrib *= timestep

  wp.atomic_sub(qDeriv_out, worldid, 0, elemid, qderiv_contrib)


@event_scope
def deriv_smooth_vel(m: Model, d: Data, out: wp.array2d[float]):
  """Analytical derivative of smooth forces w.r.t. velocities.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    out: qM - dt * qDeriv (derivatives of smooth forces w.r.t velocities).
  """
  qMi = m.qM_fullm_i
  qMj = m.qM_fullm_j

  # TODO(team): implicit requires different sparsity structure

  if ~(m.opt.disableflags & (DisableBit.ACTUATION | DisableBit.DAMPER)):
    # TODO(team): only clear elements not set by _qderiv_actuator_passive
    out.zero_()
    if m.nu > 0 and not (m.opt.disableflags & DisableBit.ACTUATION):
      vel = wp.empty((d.nworld, m.nu), dtype=float)
      wp.launch(
        _qderiv_actuator_passive_vel,
        dim=(d.nworld, m.nu),
        inputs=[
          m.opt.timestep,
          m.actuator_dyntype,
          m.actuator_gaintype,
          m.actuator_biastype,
          m.actuator_actadr,
          m.actuator_actnum,
          m.actuator_forcelimited,
          m.actuator_actlimited,
          m.actuator_dynprm,
          m.actuator_gainprm,
          m.actuator_biasprm,
          m.actuator_actearly,
          m.actuator_forcerange,
          m.actuator_actrange,
          d.act,
          d.ctrl,
          d.act_dot,
          d.actuator_force,
        ],
        outputs=[vel],
      )
      if m.is_sparse:
        wp.launch(
          _qderiv_actuator_passive_actuation_sparse,
          dim=(d.nworld, m.nu),
          inputs=[m.M_rownnz, m.M_rowadr, d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment, vel, qMj],
          outputs=[out],
        )
      else:
        wp.launch(
          _qderiv_actuator_passive_actuation_dense,
          dim=(d.nworld, qMi.size),
          inputs=[m.nu, d.moment_rownnz, d.moment_rowadr, d.moment_colind, d.actuator_moment, vel, qMi, qMj],
          outputs=[out],
        )
    wp.launch(
      _qderiv_actuator_passive,
      dim=(d.nworld, qMi.size),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.dof_damping,
        m.dof_dampingpoly,
        m.is_sparse,
        d.qvel,
        d.qM,
        qMi,
        qMj,
        out,
      ],
      outputs=[out],
    )
  else:
    # TODO(team): directly utilize qM for these settings
    wp.copy(out, d.qM)

  if not (m.opt.disableflags & DisableBit.DAMPER):
    wp.launch(
      _qderiv_tendon_damping,
      dim=(d.nworld, qMi.size),
      inputs=[
        m.ntendon,
        m.opt.timestep,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_damping,
        m.tendon_dampingpoly,
        m.is_sparse,
        d.ten_J,
        d.ten_velocity,
        qMi,
        qMj,
      ],
      outputs=[out],
    )

  if m.opt.has_fluid and not m.opt.disableflags & DisableBit.DAMPER:
    if m.opt.is_sparse:
      wp.launch(
        _deriv_ellipsoid_fluid_sparse,
        dim=(d.nworld, qMi.size),
        inputs=[
          m.opt.density,
          m.opt.viscosity,
          m.opt.wind,
          m.opt.timestep,
          m.opt.integrator,
          m.body_parentid,
          m.body_rootid,
          m.body_geomnum,
          m.body_geomadr,
          m.body_fluid_ellipsoid,
          m.geom_type,
          m.geom_size,
          m.geom_fluid,
          m.dof_bodyid,
          d.xipos,
          d.ximat,
          d.geom_xpos,
          d.geom_xmat,
          d.subtree_com,
          d.cvel,
          d.cdof,
          qMi,
          qMj,
        ],
        outputs=[out],
      )
    else:
      # Dense mode: iterate over all nv x nv DOF pairs
      wp.launch(
        _deriv_ellipsoid_fluid_dense,
        dim=(d.nworld, m.nv, m.nv),
        inputs=[
          m.nv,
          m.opt.density,
          m.opt.viscosity,
          m.opt.wind,
          m.opt.timestep,
          m.opt.integrator,
          m.body_parentid,
          m.body_rootid,
          m.body_geomnum,
          m.body_geomadr,
          m.body_fluid_ellipsoid,
          m.geom_type,
          m.geom_size,
          m.geom_fluid,
          m.dof_bodyid,
          d.xipos,
          d.ximat,
          d.geom_xpos,
          d.geom_xmat,
          d.subtree_com,
          d.cvel,
          d.cdof,
        ],
        outputs=[out],
      )

  # TODO(team): rne derivative
