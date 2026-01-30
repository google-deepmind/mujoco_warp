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
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import TileSet
from mujoco_warp._src.types import vec10f
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _qderiv_actuator_passive_vel(
  # Model:
  actuator_dyntype: wp.array(dtype=int),
  actuator_gaintype: wp.array(dtype=int),
  actuator_biastype: wp.array(dtype=int),
  actuator_actadr: wp.array(dtype=int),
  actuator_actnum: wp.array(dtype=int),
  actuator_gainprm: wp.array2d(dtype=vec10f),
  actuator_biasprm: wp.array2d(dtype=vec10f),
  # Data in:
  act_in: wp.array2d(dtype=float),
  ctrl_in: wp.array2d(dtype=float),
  # Out:
  vel_out: wp.array2d(dtype=float),
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
  else:
    bias = 0.0

  if bias == 0.0 and gain == 0.0:
    vel_out[worldid, actid] = 0.0
    return

  vel = float(bias)
  if actuator_dyntype[actid] != DynType.NONE:
    if gain != 0.0:
      act_first = actuator_actadr[actid]
      act_last = act_first + actuator_actnum[actid] - 1
      vel += gain * act_in[worldid, act_last]
  else:
    if gain != 0.0:
      vel += gain * ctrl_in[worldid, actid]

  vel_out[worldid, actid] = vel


@cache_kernel
def _qderiv_actuator_passive_actuation_dense(tile: TileSet, nu: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    actuator_moment_in: wp.array3d(dtype=float),
    # In:
    vel_in: wp.array3d(dtype=float),
    adr: wp.array(dtype=int),
    # Out:
    qDeriv_out: wp.array3d(dtype=float),
  ):
    worldid, nodeid = wp.tid()
    TILE_SIZE = wp.static(tile.size)
    NU = wp.static(nu)

    dofid = adr[nodeid]
    vel_tile = wp.tile_load(vel_in[worldid], shape=(NU, 1), bounds_check=False)
    moment_tile = wp.tile_load(actuator_moment_in[worldid], shape=(NU, TILE_SIZE), offset=(0, dofid), bounds_check=False)
    moment_weighted = wp.tile_map(wp.mul, wp.tile_broadcast(vel_tile, shape=(NU, TILE_SIZE)), moment_tile)
    qderiv_tile = wp.tile_matmul(wp.tile_transpose(moment_tile), moment_weighted)
    wp.tile_store(qDeriv_out[worldid], qderiv_tile, offset=(dofid, dofid), bounds_check=False)

  return kernel


@wp.kernel
def _qderiv_actuator_passive_actuation_sparse(
  # Model:
  nu: int,
  # Data in:
  actuator_moment_in: wp.array3d(dtype=float),
  # In:
  vel_in: wp.array2d(dtype=float),
  qMi: wp.array(dtype=int),
  qMj: wp.array(dtype=int),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  worldid, elemid = wp.tid()

  dofiid = qMi[elemid]
  dofjid = qMj[elemid]
  qderiv_contrib = float(0.0)
  for actid in range(nu):
    vel = vel_in[worldid, actid]
    if vel == 0.0:
      continue

    moment_i = actuator_moment_in[worldid, actid, dofiid]
    moment_j = actuator_moment_in[worldid, actid, dofjid]

    qderiv_contrib += moment_i * moment_j * vel

  qDeriv_out[worldid, 0, elemid] = qderiv_contrib


@wp.kernel
def _qderiv_actuator_passive(
  # Model:
  opt_timestep: wp.array(dtype=float),
  opt_disableflags: int,
  opt_is_sparse: bool,
  dof_damping: wp.array2d(dtype=float),
  # Data in:
  qM_in: wp.array3d(dtype=float),
  # In:
  qMi: wp.array(dtype=int),
  qMj: wp.array(dtype=int),
  qDeriv_in: wp.array3d(dtype=float),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  worldid, elemid = wp.tid()

  dofiid = qMi[elemid]
  dofjid = qMj[elemid]

  if opt_is_sparse:
    qderiv = qDeriv_in[worldid, 0, elemid]
  else:
    qderiv = qDeriv_in[worldid, dofiid, dofjid]

  if not opt_disableflags & DisableBit.DAMPER and dofiid == dofjid:
    qderiv -= dof_damping[worldid % dof_damping.shape[0], dofiid]

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  if opt_is_sparse:
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
  opt_timestep: wp.array(dtype=float),
  opt_is_sparse: bool,
  tendon_damping: wp.array2d(dtype=float),
  # Data in:
  ten_J_in: wp.array3d(dtype=float),
  # In:
  qMi: wp.array(dtype=int),
  qMj: wp.array(dtype=int),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  worldid, elemid = wp.tid()
  dofiid = qMi[elemid]
  dofjid = qMj[elemid]

  qderiv = float(0.0)
  tendon_damping_id = worldid % tendon_damping.shape[0]
  for tenid in range(ntendon):
    qderiv -= ten_J_in[worldid, tenid, dofiid] * ten_J_in[worldid, tenid, dofjid] * tendon_damping[tendon_damping_id, tenid]

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  if opt_is_sparse:
    qDeriv_out[worldid, 0, elemid] -= qderiv
  else:
    qDeriv_out[worldid, dofiid, dofjid] -= qderiv
    if dofiid != dofjid:
      qDeriv_out[worldid, dofjid, dofiid] -= qderiv


@wp.func
def _ellipsoid_max_moment(size: wp.vec3, dir: int) -> float:
  d0 = size[dir]
  d1 = size[(dir + 1) % 3]
  d2 = size[(dir + 2) % 3]
  max_d1_d2 = wp.max(d1, d2)
  return 8.0 / 15.0 * wp.PI * d0 * wp.pow(max_d1_d2, 4.0)


@wp.func
def _add_to_quadrant(B: wp.spatial_matrix, D: wp.mat33, col_quad: int, row_quad: int) -> wp.spatial_matrix:
  r = 3 * row_quad
  c = 3 * col_quad
  for i in range(3):
    for j in range(3):
      B[r + i, c + j] += D[i, j]
  return B


@wp.func
def _derivative_cross(vec_a: wp.vec3, vec_b: wp.vec3):
  Da = wp.skew(vec_b)
  Db = -wp.skew(vec_a)
  return Da, Db


@wp.func
def _fluid_added_mass_forces(
  # In:
  B: wp.spatial_matrix,
  lvel: wp.spatial_vector,
  fluid_density: float,
  virtual_mass: wp.vec3,
  virtual_inertia: wp.vec3,
) -> wp.spatial_matrix:
  lin_vel = lvel[3:6]
  ang_vel = lvel[0:3]

  virtual_lin_mom = fluid_density * wp.cw_mul(virtual_mass, lin_vel)
  virtual_ang_mom = fluid_density * wp.cw_mul(virtual_inertia, ang_vel)

  Da, Db = _derivative_cross(virtual_ang_mom, ang_vel)
  scale = fluid_density * virtual_inertia
  Da_scaled = wp.transpose(
    wp.mat33(
      Da[0] * scale[0],
      Da[1] * scale[1],
      Da[2] * scale[2],
    )
  )
  B = _add_to_quadrant(B, Da_scaled, 0, 0)
  B = _add_to_quadrant(B, Db, 0, 0)

  Da, Db = _derivative_cross(virtual_lin_mom, lin_vel)
  scale = fluid_density * virtual_mass
  Da_scale = wp.transpose(
    wp.mat33(
      Da[0] * scale[0],
      Da[1] * scale[1],
      Da[2] * scale[2],
    )
  )
  B = _add_to_quadrant(B, Da_scale, 0, 1)
  B = _add_to_quadrant(B, Db, 0, 1)

  Da, Db = _derivative_cross(virtual_lin_mom, ang_vel)
  scale = fluid_density * virtual_mass
  Da_scale = wp.transpose(
    wp.mat33(
      Da[0] * scale[0],
      Da[1] * scale[1],
      Da[2] * scale[2],
    )
  )
  B = _add_to_quadrant(B, Da_scale, 1, 1)
  B = _add_to_quadrant(B, Db, 1, 0)

  return B


@wp.func
def _fluid_viscous_torque(
  # In:
  B: wp.spatial_matrix,
  lvel: wp.spatial_vector,
  fluid_density: float,
  fluid_viscosity: float,
  size: wp.vec3,
  slender_drag_coef: float,
  ang_drag_coef: float,
) -> wp.spatial_matrix:
  d_max = wp.max(wp.max(size[0], size[1]), size[2])
  d_min = wp.min(wp.min(size[0], size[1]), size[2])
  d_mid = size[0] + size[1] + size[2] - d_max - d_min

  eq_sphere_D = 2.0 / 3.0 * (size[0] + size[1] + size[2])
  lin_visc_torq_coef = wp.PI * wp.pow(eq_sphere_D, 3.0)

  I_max = 8.0 / 15.0 * wp.PI * d_mid * wp.pow(d_max, 4.0)
  II = wp.vec3(_ellipsoid_max_moment(size, 0), _ellipsoid_max_moment(size, 1), _ellipsoid_max_moment(size, 2))

  ang_vel = lvel[0:3]

  mom_coef = ang_drag_coef * II + slender_drag_coef * (wp.vec3(I_max) - II)

  mom_visc = wp.cw_mul(ang_vel, mom_coef)
  density = fluid_density / wp.max(MJ_MINVAL, wp.length(mom_visc))

  mom_sq = -density * wp.cw_mul(ang_vel, wp.cw_mul(mom_coef, mom_coef))
  lin_coef = fluid_viscosity * lin_visc_torq_coef

  sum_val = wp.dot(ang_vel, mom_sq) - lin_coef
  D = wp.diag(wp.vec3(sum_val, sum_val, sum_val))
  D += wp.outer(mom_sq, ang_vel)

  B = _add_to_quadrant(B, D, 0, 0)
  return B


@wp.func
def _fluid_viscous_drag(
  # In:
  B: wp.spatial_matrix,
  lvel: wp.spatial_vector,
  fluid_density: float,
  fluid_viscosity: float,
  size: wp.vec3,
  blunt_drag_coef: float,
  slender_drag_coef: float,
) -> wp.spatial_matrix:
  d_max = wp.max(wp.max(size[0], size[1]), size[2])
  d_min = wp.min(wp.min(size[0], size[1]), size[2])
  d_mid = size[0] + size[1] + size[2] - d_max - d_min

  eq_sphere_D = 2.0 / 3.0 * (size[0] + size[1] + size[2])
  A_max = wp.PI * d_max * d_mid

  a = (size[1] * size[2]) * (size[1] * size[2])
  b = (size[2] * size[0]) * (size[2] * size[0])
  c = (size[0] * size[1]) * (size[0] * size[1])
  aa = a * a
  bb = b * b
  cc = c * c

  x = lvel[3]
  y = lvel[4]
  z = lvel[5]
  xx = x * x
  yy = y * y
  zz = z * z

  proj_denom = aa * xx + bb * yy + cc * zz
  proj_num = a * xx + b * yy + c * zz

  dA_coef = wp.PI / wp.max(MJ_MINVAL, wp.sqrt(proj_num * proj_num * proj_num * proj_denom))
  A_proj = wp.PI * wp.sqrt(proj_denom / wp.max(MJ_MINVAL, proj_num))

  norm = wp.sqrt(xx + yy + zz)
  inv_norm = 1.0 / wp.max(MJ_MINVAL, norm)

  lin_coef = fluid_viscosity * 3.0 * wp.PI * eq_sphere_D
  quad_coef = fluid_density * (A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj))
  Aproj_coef = fluid_density * norm * (blunt_drag_coef - slender_drag_coef)

  dAproj_dv = wp.vec3(
    Aproj_coef * dA_coef * a * x * (b * yy * (a - b) + c * zz * (a - c)),
    Aproj_coef * dA_coef * b * y * (a * xx * (b - a) + c * zz * (b - c)),
    Aproj_coef * dA_coef * c * z * (a * xx * (c - a) + b * yy * (c - b)),
  )

  inner_sum = xx + yy + zz
  # D = diag(inner_sum) + outer(v, v)
  D = wp.diag(wp.vec3(inner_sum, inner_sum, inner_sum)) + wp.outer(wp.vec3(x, y, z), wp.vec3(x, y, z))

  scale = -quad_coef * inv_norm
  D *= scale

  D += wp.outer(dAproj_dv, -wp.vec3(x, y, z))
  D -= wp.diag(wp.vec3(lin_coef, lin_coef, lin_coef))

  B = _add_to_quadrant(B, D, 1, 1)
  return B


@wp.func
def _fluid_kutta_lift(
  # In:
  B: wp.spatial_matrix,
  lvel: wp.spatial_vector,
  fluid_density: float,
  size: wp.vec3,
  kutta_lift_coef: float,
) -> wp.spatial_matrix:
  a = wp.pow(size[1] * size[2], 2.0)
  b = wp.pow(size[2] * size[0], 2.0)
  c = wp.pow(size[0] * size[1], 2.0)
  aa = a * a
  bb = b * b
  cc = c * c

  x = lvel[3]
  y = lvel[4]
  z = lvel[5]
  xx = x * x
  yy = y * y
  zz = z * z

  proj_denom = aa * xx + bb * yy + cc * zz
  proj_num = a * xx + b * yy + c * zz
  norm2 = xx + yy + zz

  df_denom = wp.PI * kutta_lift_coef * fluid_density / wp.max(MJ_MINVAL, wp.sqrt(proj_denom * proj_num * norm2))

  dfx_coef = yy * (a - b) + zz * (a - c)
  dfy_coef = xx * (b - a) + zz * (b - c)
  dfz_coef = xx * (c - a) + yy * (c - b)
  proj_term = proj_num / wp.max(MJ_MINVAL, proj_denom)
  cos_term = proj_num / wp.max(MJ_MINVAL, norm2)

  skew_vec = wp.vec3(c - b, a - c, b - a)
  D = wp.skew(skew_vec) * (2.0 * proj_num)

  inner_term = wp.vec3(aa * proj_term - a + cos_term, bb * proj_term - b + cos_term, cc * proj_term - c + cos_term)

  df_coef = wp.vec3(dfx_coef, dfy_coef, dfz_coef)
  D += wp.outer(inner_term, df_coef)

  # Element-wise multiply by outer(v, v)
  v = wp.vec3(x, y, z)
  D = wp.cw_mul(D, wp.outer(v, v))

  # Subtract diagonal terms: df_coef * proj_num
  D -= wp.diag(df_coef * proj_num)

  D *= df_denom

  B = _add_to_quadrant(B, D, 1, 1)
  return B


@wp.func
def _fluid_magnus_force(
  # In:
  B: wp.spatial_matrix,
  lvel: wp.spatial_vector,
  fluid_density: float,
  size: wp.vec3,
  magnus_lift_coef: float,
) -> wp.spatial_matrix:
  volume = 4.0 / 3.0 * wp.PI * size[0] * size[1] * size[2]
  magnus_coef = magnus_lift_coef * fluid_density * volume

  lin_vel = lvel[3:]
  ang_vel = lvel[:3]
  lin_vel *= magnus_coef
  ang_vel *= magnus_coef

  D_ang, D_lin = _derivative_cross(ang_vel, lin_vel)

  B = _add_to_quadrant(B, D_ang, 1, 0)
  B = _add_to_quadrant(B, D_lin, 1, 1)
  return B


@wp.func
def _fluid_inertia_box_forces(
  # In:
  B: wp.spatial_matrix,
  lvel: wp.spatial_vector,
  fluid_density: float,
  fluid_viscosity: float,
  mass: float,
  inertia: wp.vec3,
) -> wp.spatial_matrix:
  # Equivalent inertia box
  box = wp.vec3(
    wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) / mass * 6.0),
    wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) / mass * 6.0),
    wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) / mass * 6.0),
  )

  # Viscous force and torque
  if fluid_viscosity > 0.0:
    diam = (box[0] + box[1] + box[2]) / 3.0

    # Rotational viscosity
    visc_rot = -wp.PI * diam * diam * diam * fluid_viscosity
    B[0, 0] += visc_rot
    B[1, 1] += visc_rot
    B[2, 2] += visc_rot

    # Translational viscosity
    visc_lin = -3.0 * wp.PI * diam * fluid_viscosity
    B[3, 3] += visc_lin
    B[4, 4] += visc_lin
    B[5, 5] += visc_lin

  # Lift and drag force and torque
  if fluid_density > 0.0:
    term0 = box[1] * box[1] * box[1] * box[1] + box[2] * box[2] * box[2] * box[2]
    term1 = box[0] * box[0] * box[0] * box[0] + box[2] * box[2] * box[2] * box[2]
    term2 = box[0] * box[0] * box[0] * box[0] + box[1] * box[1] * box[1] * box[1]

    B[0, 0] -= fluid_density * box[0] * term0 * wp.abs(lvel[0]) / 32.0
    B[1, 1] -= fluid_density * box[1] * term1 * wp.abs(lvel[1]) / 32.0
    B[2, 2] -= fluid_density * box[2] * term2 * wp.abs(lvel[2]) / 32.0

    B[3, 3] -= fluid_density * box[1] * box[2] * wp.abs(lvel[3])
    B[4, 4] -= fluid_density * box[0] * box[2] * wp.abs(lvel[4])
    B[5, 5] -= fluid_density * box[0] * box[1] * wp.abs(lvel[5])

  return B


@wp.func
def _get_geom_semiaxes(type: int, size: wp.vec3) -> wp.vec3:
  if type == GeomType.SPHERE:
    return wp.vec3(size[0], size[0], size[0])
  elif type == GeomType.CAPSULE:
    return wp.vec3(size[0], size[0], size[0] + size[1])
  elif type == GeomType.ELLIPSOID:
    return size
  elif type == GeomType.CYLINDER:
    return wp.vec3(size[0], size[0], size[1])
  elif type == GeomType.BOX:
    return size
  else:
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _is_ancestor(body_parentid: wp.array(dtype=int), ancestor: int, child: int) -> bool:
  if child == ancestor:
    return True

  curr = child
  while curr != 0:
    curr = body_parentid[curr]
    if curr == ancestor:
      return True

  return False


@wp.func
def _get_jac_column_local(
  # Model:
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  dof_bodyid: wp.array(dtype=int),
  # Data in:
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  point_global: wp.vec3,
  bodyid: int,
  dofid: int,
  worldid: int,
  b_imat: wp.mat33,
) -> wp.spatial_vector:
  jacp, jacr = support.jac(
    body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, point_global, bodyid, dofid, worldid
  )
  jacp_loc = wp.transpose(b_imat) @ jacp
  jacr_loc = wp.transpose(b_imat) @ jacr
  return wp.spatial_vector(jacr_loc, jacp_loc)


@wp.kernel
def _qderiv_fluid(
  # Model:
  nbody: int,
  opt_timestep: wp.array(dtype=float),
  opt_density: wp.array(dtype=float),
  opt_viscosity: wp.array(dtype=float),
  opt_wind: wp.array(dtype=wp.vec3),
  opt_integrator: int,
  opt_is_sparse: bool,
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_geomnum: wp.array(dtype=int),
  body_geomadr: wp.array(dtype=int),
  body_mass: wp.array2d(dtype=float),
  body_inertia: wp.array2d(dtype=wp.vec3),
  dof_bodyid: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_fluid: wp.array2d(dtype=float),
  body_fluid_ellipsoid: wp.array(dtype=bool),
  # Data in:
  xipos_in: wp.array2d(dtype=wp.vec3),
  ximat_in: wp.array2d(dtype=wp.mat33),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  qMi: wp.array(dtype=int),
  qMj: wp.array(dtype=int),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  worldid, elemid = wp.tid()
  dofi = qMi[elemid]
  dofj = qMj[elemid]

  bodyi = dof_bodyid[dofi]
  bodyj = dof_bodyid[dofj]

  start_body = wp.max(bodyi, bodyj)

  density = opt_density[worldid]
  viscosity = opt_viscosity[worldid]
  wind = opt_wind[worldid]
  dt = opt_timestep[worldid]
  is_implicit = opt_integrator == IntegratorType.IMPLICITFAST

  accum = float(0.0)

  for b in range(start_body, nbody):
    if not _is_ancestor(body_parentid, start_body, b) or body_mass[worldid, b] < MJ_MINVAL:
      continue

    if body_fluid_ellipsoid[b]:
      start_geom = body_geomadr[b]
      num_geom = body_geomnum[b]

      for g in range(start_geom, start_geom + num_geom):
        # geom_interaction_coef is at index 0
        if geom_fluid[g, 0] == 0.0:
          continue

        subtree_root = subtree_com_in[worldid, body_rootid[b]]
        xipos_b = xipos_in[worldid, b]
        g_pos = geom_xpos_in[worldid, g]

        vel_body = cvel_in[worldid, b]
        w_body = wp.vec3(vel_body[0], vel_body[1], vel_body[2])
        v_body_lin = wp.vec3(vel_body[3], vel_body[4], vel_body[5])

        lin_com = v_body_lin - wp.cross(xipos_b - subtree_root, w_body)
        v_geom_lin = lin_com + wp.cross(w_body, g_pos - xipos_b)

        g_mat = geom_xmat_in[worldid, g]
        w_local = wp.transpose(g_mat) @ w_body
        v_local_linear = wp.transpose(g_mat) @ v_geom_lin
        wind_local = wp.transpose(g_mat) @ wind

        lvel = wp.spatial_vector(w_local, v_local_linear - wind_local)

        # Initialize B as zero spatial matrix
        B = wp.spatial_matrix(0.0)

        # Handle potentially batched geom_size (safe broadcasting)
        geom_size_id = worldid % geom_size.shape[0]
        semiaxes = _get_geom_semiaxes(geom_type[g], geom_size[geom_size_id, g])

        # Unpack params
        blunt = geom_fluid[g, 1]
        slender = geom_fluid[g, 2]
        ang_drag = geom_fluid[g, 3]
        kutta = geom_fluid[g, 4]
        magnus = geom_fluid[g, 5]
        vm = wp.vec3(geom_fluid[g, 6], geom_fluid[g, 7], geom_fluid[g, 8])
        vi = wp.vec3(geom_fluid[g, 9], geom_fluid[g, 10], geom_fluid[g, 11])

        B = _fluid_magnus_force(B, lvel, density, semiaxes, magnus)
        B = _fluid_kutta_lift(B, lvel, density, semiaxes, kutta)
        B = _fluid_viscous_drag(B, lvel, density, viscosity, semiaxes, blunt, slender)
        B = _fluid_viscous_torque(B, lvel, density, viscosity, semiaxes, slender, ang_drag)
        B = _fluid_added_mass_forces(B, lvel, density, vm, vi)
        if is_implicit:
          B = 0.5 * (B + wp.transpose(B))
        # Get Jacobian columns
        J_i = _get_jac_column_local(
          body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, g_pos, b, dofi, worldid, g_mat
        )
        J_j = _get_jac_column_local(
          body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, g_pos, b, dofj, worldid, g_mat
        )

        accum += wp.dot(J_i, B @ J_j)
    else:
      # Inertia box approximation
      mass = body_mass[worldid, b]
      inertia = body_inertia[worldid, b]

      subtree_root = subtree_com_in[worldid, body_rootid[b]]
      b_ipos = xipos_in[worldid, b]

      # Body velocity in local frame
      vel_subtree = cvel_in[worldid, b]
      v_subtree_ang = wp.vec3(vel_subtree[0], vel_subtree[1], vel_subtree[2])
      v_subtree_lin = wp.vec3(vel_subtree[3], vel_subtree[4], vel_subtree[5])

      lin_com = v_subtree_lin - wp.cross(b_ipos - subtree_root, v_subtree_ang)

      b_imat = ximat_in[worldid, b]
      v_local_ang = wp.transpose(b_imat) @ v_subtree_ang
      v_local_lin = wp.transpose(b_imat) @ lin_com

      # Wind in local frame
      wind_local = wp.transpose(b_imat) @ wind

      lvel = wp.spatial_vector(v_local_ang, v_local_lin - wind_local)
      B = wp.spatial_matrix(0.0)
      B = _fluid_inertia_box_forces(B, lvel, density, viscosity, mass, inertia)

      if is_implicit:
        B = 0.5 * (B + wp.transpose(B))

      # Jacobian at body frame
      J_i = _get_jac_column_local(
        body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, b_ipos, b, dofi, worldid, b_imat
      )
      J_j = _get_jac_column_local(
        body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, b_ipos, b, dofj, worldid, b_imat
      )

      accum += wp.dot(J_i, B @ J_j)

  val = accum * dt

  if opt_is_sparse:
    qDeriv_out[worldid, 0, elemid] -= val
  else:
    qDeriv_out[worldid, dofi, dofj] -= val
    if dofi != dofj:
      qDeriv_out[worldid, dofj, dofi] -= val


@event_scope
def deriv_smooth_vel(m: Model, d: Data, out: wp.array2d(dtype=float)):
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
    if m.nu > 0 and not m.opt.disableflags & DisableBit.ACTUATION:
      vel = wp.empty((d.nworld, m.nu), dtype=float)
      wp.launch(
        _qderiv_actuator_passive_vel,
        dim=(d.nworld, m.nu),
        inputs=[
          m.actuator_dyntype,
          m.actuator_gaintype,
          m.actuator_biastype,
          m.actuator_actadr,
          m.actuator_actnum,
          m.actuator_gainprm,
          m.actuator_biasprm,
          d.act,
          d.ctrl,
        ],
        outputs=[vel],
      )
      if m.opt.is_sparse:
        wp.launch(
          _qderiv_actuator_passive_actuation_sparse,
          dim=(d.nworld, qMi.size),
          inputs=[m.nu, d.actuator_moment, vel, qMi, qMj],
          outputs=[out],
        )
      else:
        vel_3d = vel.reshape(vel.shape + (1,))
        for tile in m.qM_tiles:
          wp.launch_tiled(
            _qderiv_actuator_passive_actuation_dense(tile, m.nu),
            dim=(d.nworld, tile.adr.size),
            inputs=[d.actuator_moment, vel_3d, tile.adr],
            outputs=[out],
            block_dim=m.block_dim.qderiv_actuator_dense,
          )
    wp.launch(
      _qderiv_actuator_passive,
      dim=(d.nworld, qMi.size),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.opt.is_sparse,
        m.dof_damping,
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

  if not m.opt.disableflags & DisableBit.DAMPER:
    wp.launch(
      _qderiv_tendon_damping,
      dim=(d.nworld, qMi.size),
      inputs=[m.ntendon, m.opt.timestep, m.opt.is_sparse, m.tendon_damping, d.ten_J, qMi, qMj],
      outputs=[out],
    )
  if m.opt.has_fluid:
    wp.launch(
      _qderiv_fluid,
      dim=(d.nworld, m.qM_fullm_i.size),
      inputs=[
        m.nbody,
        m.opt.timestep,
        m.opt.density,
        m.opt.viscosity,
        m.opt.wind,
        m.opt.integrator,
        m.opt.is_sparse,
        m.body_parentid,
        m.body_rootid,
        m.body_geomnum,
        m.body_geomadr,
        m.body_mass,
        m.body_inertia,
        m.dof_bodyid,
        m.geom_type,
        m.geom_size,
        m.geom_fluid,
        m.body_fluid_ellipsoid,
        d.xipos,
        d.ximat,
        d.geom_xpos,
        d.geom_xmat,
        d.subtree_com,
        d.cdof,
        d.cvel,
        m.qM_fullm_i,
        m.qM_fullm_j,
      ],
      outputs=[out],
    )

  # TODO(team): rne derivative
