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

from typing import Tuple

import warp as wp

from mujoco_warp._src import util_misc
from mujoco_warp._src.passive import _ellipsoid_max_moment
from mujoco_warp._src.passive import geom_semiaxes
from mujoco_warp._src.support import next_act
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import Model
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
def _skew_neg(v: wp.vec3) -> wp.mat33:
  """Skew-symmetric matrix: d/db cross(a, b) = [[0, a3, -a2], [-a3, 0, a1], [a2, -a1, 0]]."""
  # fmt: off
  return wp.mat33(
    0.0,   v[2], -v[1],
   -v[2],  0.0,   v[0],
    v[1], -v[0],  0.0,
  )
  # fmt: on


@wp.func
def _skew_pos(v: wp.vec3) -> wp.mat33:
  """Skew matrix: d/da cross(a, b) = [[0, -b3, b2], [b3, 0, -b1], [-b2, b1, 0]]."""
  # fmt: off
  return wp.mat33(
     0.0, -v[2],  v[1],
     v[2],  0.0, -v[0],
    -v[1],  v[0],  0.0,
  )
  # fmt: on


@wp.func
def _scale_cols(M: wp.mat33, s: wp.vec3) -> wp.mat33:
  """Scale column j of M by s[j]."""
  # fmt: off
  return wp.mat33(
    M[0, 0] * s[0], M[0, 1] * s[1], M[0, 2] * s[2],
    M[1, 0] * s[0], M[1, 1] * s[1], M[1, 2] * s[2],
    M[2, 0] * s[0], M[2, 1] * s[1], M[2, 2] * s[2],
  )
  # fmt: on


@wp.func
def _symmetrize33(A: wp.mat33, B: wp.mat33) -> Tuple[wp.mat33, wp.mat33]:
  """Symmetrize off-diagonal quadrants: avg = 0.5*(A + B^T), return (avg, avg^T)."""
  S = wp.mat33(0.0)
  for i in range(3):
    for j in range(3):
      S[i, j] = 0.5 * (A[i, j] + B[j, i])
  return S, wp.transpose(S)


@wp.func
def _deriv_ellipsoid_fluid(
  # Model:
  opt_integrator: int,
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_fluid: wp.array2d[float],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  bodyid: int,
  rootid: int,
  geomadr: int,
  geomnum: int,
  cdof_i: wp.spatial_vector,
  cdof_j: wp.spatial_vector,
  wind: wp.vec3,
  density: float,
  viscosity: float,
) -> float:
  """Compute one body's ellipsoid fluid derivative contribution for a DOF pair.

  Returns the scalar J_i^T @ B @ J_j contribution accumulated across geoms.
  """
  is_implicitfast = opt_integrator == IntegratorType.IMPLICITFAST

  # Body kinematics
  xipos = xipos_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]
  ang_global = wp.spatial_top(cvel)
  lin_global = wp.spatial_bottom(cvel)
  subtree_root = subtree_com_in[worldid, rootid]
  lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)

  qderiv_contrib = float(0.0)

  for g in range(geomnum):
    geomid = geomadr + g
    coef = geom_fluid[geomid, 0]
    if coef <= 0.0:
      continue

    size = geom_size[worldid % geom_size.shape[0], geomid]
    semiaxes = geom_semiaxes(size, geom_type[geomid])
    geom_rot = geom_xmat_in[worldid, geomid]
    geom_rotT = wp.transpose(geom_rot)
    geom_pos = geom_xpos_in[worldid, geomid]

    # compute local velocity
    lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)
    l_ang = geom_rotT @ ang_global
    l_lin = geom_rotT @ lin_point

    if wind[0] != 0.0 or wind[1] != 0.0 or wind[2] != 0.0:
      l_lin -= geom_rotT @ wind

    ang_vel = l_ang
    lin_vel = l_lin

    # read fluid coefficients
    blunt_drag_coef = geom_fluid[geomid, 1]
    slender_drag_coef = geom_fluid[geomid, 2]
    ang_drag_coef = geom_fluid[geomid, 3]
    kutta_lift_coef = geom_fluid[geomid, 4]
    magnus_lift_coef = geom_fluid[geomid, 5]
    virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])
    virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])

    # ===== Build 6x6 B matrix as four 3x3 quadrants =====
    # B = [[B00, B01], [B10, B11]] where rows are [ang; lin], cols are [ang; lin]
    B00 = wp.mat33(0.0)  # torque wrt ang_vel
    B01 = wp.mat33(0.0)  # torque wrt lin_vel
    B10 = wp.mat33(0.0)  # force wrt ang_vel
    B11 = wp.mat33(0.0)  # force wrt lin_vel

    if density > 0.0:
      # --- added mass forces ---
      virtual_lin_mom = density * wp.cw_mul(virtual_mass, lin_vel)
      virtual_ang_mom = density * wp.cw_mul(virtual_inertia, ang_vel)

      # torque += cross(virtual_ang_mom, ang_vel) -> B00
      B00 += _skew_pos(virtual_ang_mom)
      B00 += _scale_cols(_skew_neg(ang_vel), density * virtual_inertia)

      # torque += cross(virtual_lin_mom, lin_vel) -> B01
      B01 += _skew_pos(virtual_lin_mom)
      B01 += _scale_cols(_skew_neg(lin_vel), density * virtual_mass)

      # force += cross(virtual_lin_mom, ang_vel) -> B10
      B10 += _skew_pos(virtual_lin_mom)
      # Da = d/d(vlm via lin) = _skew_neg(ang), scaled by density*vm -> B11
      B11 += _scale_cols(_skew_neg(ang_vel), density * virtual_mass)

    # --- Magnus force: force += magnus_coef * cross(ang_vel, lin_vel) ---
    volume = wp.static(4.0 / 3.0 * wp.pi) * semiaxes[0] * semiaxes[1] * semiaxes[2]
    magnus_coef = magnus_lift_coef * density * volume
    B10 += _skew_neg(lin_vel) * magnus_coef
    ang_vel_scaled = ang_vel * magnus_coef
    B11 += _skew_pos(ang_vel_scaled)

    # --- Kutta lift (3x3 -> B11) ---
    a = (semiaxes[1] * semiaxes[2]) * (semiaxes[1] * semiaxes[2])
    b = (semiaxes[2] * semiaxes[0]) * (semiaxes[2] * semiaxes[0])
    c = (semiaxes[0] * semiaxes[1]) * (semiaxes[0] * semiaxes[1])
    aa = a * a
    bb = b * b
    cc = c * c

    x = lin_vel[0]
    y = lin_vel[1]
    z = lin_vel[2]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    xz = x * z

    proj_denom = aa * xx + bb * yy + cc * zz
    proj_num = a * xx + b * yy + c * zz
    norm2 = xx + yy + zz
    df_denom = wp.pi * kutta_lift_coef * density / wp.max(MJ_MINVAL, wp.sqrt(proj_denom * proj_num * norm2))

    dfx_coef = yy * (a - b) + zz * (a - c)
    dfy_coef = xx * (b - a) + zz * (b - c)
    dfz_coef = xx * (c - a) + yy * (c - b)
    proj_term = proj_num / wp.max(MJ_MINVAL, proj_denom)
    cos_term = proj_num / wp.max(MJ_MINVAL, norm2)

    D = wp.mat33(
      0.0,
      b - a,
      c - a,
      a - b,
      0.0,
      c - b,
      a - c,
      b - c,
      0.0,
    )
    D *= 2.0 * proj_num

    inner_term = wp.vec3(
      aa * proj_term - a + cos_term,
      bb * proj_term - b + cos_term,
      cc * proj_term - c + cos_term,
    )

    for i_k in range(3):
      df_coef_i = wp.where(i_k == 0, dfx_coef, wp.where(i_k == 1, dfy_coef, dfz_coef))
      for j_k in range(3):
        D[i_k, j_k] += inner_term[j_k] * df_coef_i

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

    D *= df_denom
    B11 += D

    # --- viscous drag (3x3 -> B11) ---
    d_max = wp.max(wp.max(semiaxes[0], semiaxes[1]), semiaxes[2])
    d_min = wp.min(wp.min(semiaxes[0], semiaxes[1]), semiaxes[2])
    d_mid = semiaxes[0] + semiaxes[1] + semiaxes[2] - d_max - d_min
    eq_sphere_D = wp.static(2.0 / 3.0) * (semiaxes[0] + semiaxes[1] + semiaxes[2])
    A_max = wp.pi * d_max * d_mid

    A_proj = wp.pi * wp.sqrt(proj_denom / wp.max(MJ_MINVAL, proj_num))

    norm = wp.sqrt(xx + yy + zz)
    inv_norm = 1.0 / wp.max(MJ_MINVAL, norm)

    lin_coef = viscosity * wp.static(3.0 * wp.pi) * eq_sphere_D
    quad_coef = density * (A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj))
    Aproj_coef = density * norm * (blunt_drag_coef - slender_drag_coef)
    dA_coef = wp.pi / wp.max(MJ_MINVAL, wp.sqrt(proj_num * proj_num * proj_num * proj_denom))

    dAproj_dv = wp.vec3(
      Aproj_coef * dA_coef * a * x * (b * yy * (a - b) + c * zz * (a - c)),
      Aproj_coef * dA_coef * b * y * (a * xx * (b - a) + c * zz * (b - c)),
      Aproj_coef * dA_coef * c * z * (a * xx * (c - a) + b * yy * (c - b)),
    )

    inner = xx + yy + zz
    D = wp.mat33(
      xx + inner,
      xy,
      xz,
      xy,
      yy + inner,
      yz,
      xz,
      yz,
      zz + inner,
    )
    D *= -quad_coef * inv_norm

    for i_d in range(3):
      vi = wp.where(i_d == 0, x, wp.where(i_d == 1, y, z))
      for j_d in range(3):
        D[i_d, j_d] -= vi * dAproj_dv[j_d]

    D[0, 0] -= lin_coef
    D[1, 1] -= lin_coef
    D[2, 2] -= lin_coef

    B11 += D

    # --- viscous torque (3x3 -> B00) ---
    lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D
    I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * d_max * d_max * d_max * d_max
    II = wp.vec3(
      _ellipsoid_max_moment(semiaxes, 0),
      _ellipsoid_max_moment(semiaxes, 1),
      _ellipsoid_max_moment(semiaxes, 2),
    )

    ax = ang_vel[0]
    ay = ang_vel[1]
    az = ang_vel[2]

    mom_coef = wp.vec3(
      ang_drag_coef * II[0] + slender_drag_coef * (I_max - II[0]),
      ang_drag_coef * II[1] + slender_drag_coef * (I_max - II[1]),
      ang_drag_coef * II[2] + slender_drag_coef * (I_max - II[2]),
    )

    mom_visc = wp.cw_mul(ang_vel, mom_coef)
    norm_mom = wp.length(mom_visc)
    density_scaled = density / wp.max(MJ_MINVAL, norm_mom)

    mom_sq = -density_scaled * wp.cw_mul(wp.cw_mul(ang_vel, mom_coef), mom_coef)

    torq_lin_coef = viscosity * lin_visc_torq_coef
    diag_val = ax * mom_sq[0] + ay * mom_sq[1] + az * mom_sq[2] - torq_lin_coef

    D = wp.mat33(
      diag_val + mom_sq[0] * ax,
      mom_sq[1] * ax,
      mom_sq[2] * ax,
      mom_sq[0] * ay,
      diag_val + mom_sq[1] * ay,
      mom_sq[2] * ay,
      mom_sq[0] * az,
      mom_sq[1] * az,
      diag_val + mom_sq[2] * az,
    )
    B00 += D

    # symmetrize for implicitfast
    if is_implicitfast:
      B00 = 0.5 * (B00 + wp.transpose(B00))
      B11 = 0.5 * (B11 + wp.transpose(B11))
      B01, B10 = _symmetrize33(B01, B10)

    # --- Jacobian transformation: J_i^T @ B @ J_j ---
    offset = geom_pos - subtree_root

    # local-frame Jacobian for dof i
    cdof_ang_i = wp.vec3(cdof_i[0], cdof_i[1], cdof_i[2])
    cdof_lin_i = wp.vec3(cdof_i[3], cdof_i[4], cdof_i[5])
    jac_p_i = cdof_lin_i + wp.cross(cdof_ang_i, offset)
    la_i = geom_rotT @ cdof_ang_i
    ll_i = geom_rotT @ jac_p_i

    # local-frame Jacobian for dof j
    cdof_ang_j = wp.vec3(cdof_j[0], cdof_j[1], cdof_j[2])
    cdof_lin_j = wp.vec3(cdof_j[3], cdof_j[4], cdof_j[5])
    jac_p_j = cdof_lin_j + wp.cross(cdof_ang_j, offset)
    la_j = geom_rotT @ cdof_ang_j
    ll_j = geom_rotT @ jac_p_j

    # B @ J_j = [B00 @ la_j + B01 @ ll_j; B10 @ la_j + B11 @ ll_j]
    Bj_ang = B00 @ la_j + B01 @ ll_j
    Bj_lin = B10 @ la_j + B11 @ ll_j

    # J_i^T @ (B @ J_j) = la_i . Bj_ang + ll_i . Bj_lin
    qderiv_contrib += wp.dot(la_i, Bj_ang) + wp.dot(ll_i, Bj_lin)

  return qderiv_contrib


@wp.kernel
def _qderiv_ellipsoid_fluid(
  # Model:
  opt_timestep: wp.array[float],
  opt_wind: wp.array[wp.vec3],
  opt_density: wp.array[float],
  opt_viscosity: wp.array[float],
  opt_integrator: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_geomnum: wp.array[int],
  body_geomadr: wp.array[int],
  dof_bodyid: wp.array[int],
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_fluid: wp.array2d[float],
  is_sparse: bool,
  body_fluid_ellipsoid_adr: wp.array[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  qMi: wp.array[int],
  qMj: wp.array[int],
  # Out:
  qDeriv_out: wp.array3d[float],
):
  """Compute ellipsoid fluid force derivative contribution to qDeriv.

  Parallelized over (world, fluid_body, elem). For each fluid body and DOF
  pair, computes the 6x6 derivative matrix B in local geom frame via
  _deriv_ellipsoid_fluid and accumulates J_i^T @ B @ J_j into qDeriv.
  """
  worldid, fluid_idx, elemid = wp.tid()

  bodyid = body_fluid_ellipsoid_adr[fluid_idx]

  dofiid = qMi[elemid]
  dofjid = qMj[elemid]

  # dofiid is the "deeper" DOF (qMi >= qMj in tree ordering).
  # Any body that has dofiid in its chain also has dofjid.
  bodyid_i = dof_bodyid[dofiid]

  if bodyid_i == 0:
    return

  # Walk from bodyid up to root; check if bodyid_i is an ancestor.
  ancestor = bodyid
  is_in_chain = int(0)
  while ancestor > 0 and is_in_chain == 0:
    if ancestor == bodyid_i:
      is_in_chain = 1
    ancestor = body_parentid[ancestor]
  if is_in_chain == 0:
    return

  wind = opt_wind[worldid % opt_wind.shape[0]]
  density = opt_density[worldid % opt_density.shape[0]]
  viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  if density <= 0.0 and viscosity <= 0.0:
    return

  cdof_i = cdof_in[worldid, dofiid]
  cdof_j = cdof_in[worldid, dofjid]

  contrib = _deriv_ellipsoid_fluid(
    opt_integrator,
    geom_type,
    geom_size,
    geom_fluid,
    xipos_in,
    geom_xpos_in,
    geom_xmat_in,
    subtree_com_in,
    cvel_in,
    worldid,
    bodyid,
    body_rootid[bodyid],
    body_geomadr[bodyid],
    body_geomnum[bodyid],
    cdof_i,
    cdof_j,
    wind,
    density,
    viscosity,
  )

  contrib *= timestep

  if is_sparse:
    wp.atomic_add(qDeriv_out[worldid, 0], elemid, -contrib)
  else:
    wp.atomic_add(qDeriv_out[worldid, dofiid], dofjid, -contrib)
    if dofiid != dofjid:
      wp.atomic_add(qDeriv_out[worldid, dofjid], dofiid, -contrib)


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

  if m.has_fluid:
    wp.launch(
      _qderiv_ellipsoid_fluid,
      dim=(d.nworld, m.body_fluid_ellipsoid_adr.size, qMi.size),
      inputs=[
        m.opt.timestep,
        m.opt.wind,
        m.opt.density,
        m.opt.viscosity,
        m.opt.integrator,
        m.body_parentid,
        m.body_rootid,
        m.body_geomnum,
        m.body_geomadr,
        m.dof_bodyid,
        m.geom_type,
        m.geom_size,
        m.geom_fluid,
        m.is_sparse,
        m.body_fluid_ellipsoid_adr,
        d.xipos,
        d.geom_xpos,
        d.geom_xmat,
        d.subtree_com,
        d.cdof,
        d.cvel,
        qMi,
        qMj,
      ],
      outputs=[out],
    )

  # TODO(team): rne derivative
