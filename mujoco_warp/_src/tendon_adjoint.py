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
"""VJPs for fixed/spatial tendons, wrapping geometry, and muscle actuation."""

from typing import Tuple

import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import math
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import support
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import WrapType
from mujoco_warp._src.types import vec10
from mujoco_warp._src.warp_util import event_scope

# Adjoint reductions use unordered atomics; deterministic backward execution is deferred.
wp.set_module_options({"enable_backward": True})


@wp.func
def _wrap_inside_phi(A: float, B: float, G: float) -> float:
  """Solves asin(A*sin(phi)) + asin(B*sin(phi)) - 2*phi + G = 0, or -1.0 on fallback."""
  maxiter = int(20)
  tolerance = float(1.0e-6)
  phi = float(wp.static(0.5 * wp.pi))
  z = float(1.0)
  asin_Az = wp.asin(A)
  asin_Bz = wp.asin(B)
  f = asin_Az + asin_Bz - wp.pi + G
  if f > 0.0:
    return -1.0

  sq_A = A * A
  sq_B = B * B
  iter = int(0)
  while (iter < maxiter) and (wp.abs(f) > tolerance):
    sq_z = z * z
    df = (
      wp.cos(phi) * (A / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_A)) + B / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_B)))
      - 2.0
    )
    if df > -MJ_MINVAL:
      return -1.0
    phi1 = phi - math.safe_div(f, df)
    if phi1 > phi:
      return -1.0
    phi = phi1
    z = wp.sin(phi)
    asin_Az = wp.asin(A * z)
    asin_Bz = wp.asin(B * z)
    f = asin_Az + asin_Bz - 2.0 * phi + G
    if f > tolerance:
      return -1.0
    iter += 1

  if iter >= maxiter:
    return -1.0
  return phi


@wp.func_grad(_wrap_inside_phi)
def _adj_wrap_inside_phi(A: float, B: float, G: float, adj_ret: float):
  if adj_ret == 0.0:
    return
  phi = _wrap_inside_phi(A, B, G)
  if phi < 0.0:
    return
  z = wp.sin(phi)
  sq_z = z * z
  inv_A = 1.0 / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * A * A))
  inv_B = 1.0 / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * B * B))
  df = wp.cos(phi) * (A * inv_A + B * inv_B) - 2.0
  if df < -MJ_MINVAL:
    scale = -adj_ret / df
    wp.adjoint[A] += scale * z * inv_A
    wp.adjoint[B] += scale * z * inv_B
    wp.adjoint[G] += scale


@wp.func
def _wrap_inside_ssa(end: wp.vec4, radius: float) -> Tuple[float, wp.vec2, wp.vec2]:
  """Loop-free differentiable 2D inside wrap using the implicit root rule."""
  end0 = wp.vec2(end[0], end[1])
  end1 = wp.vec2(end[2], end[3])
  len0 = wp.norm_l2(end0)
  len1 = wp.norm_l2(end1)
  dif = end1 - end0
  dd = wp.dot(dif, dif)

  if (len0 <= radius) or (len1 <= radius) or (radius < MJ_MINVAL) or (len0 < MJ_MINVAL) or (len1 < MJ_MINVAL):
    return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)

  if dd > MJ_MINVAL:
    a = -wp.dot(dif, end0) / dd
    if (a > 0.0) and (a < 1.0):
      tmp = end0 + a * dif
      if wp.norm_l2(tmp) <= radius:
        return -1.0, wp.vec2(MJ_MAXVAL), wp.vec2(MJ_MAXVAL)

  pnt_def, _ = math.normalize_with_norm(0.5 * (end0 + end1))
  pnt_def = pnt_def * radius

  A = math.safe_div(radius, len0)
  B = math.safe_div(radius, len1)
  cosG = math.safe_div(len0 * len0 + len1 * len1 - dd, 2.0 * len0 * len1)
  if cosG < -1.0 + MJ_MINVAL:
    return -1.0, pnt_def, pnt_def
  elif cosG > 1.0 - MJ_MINVAL:
    return 0.0, pnt_def, pnt_def

  G = wp.acos(cosG)
  phi = _wrap_inside_phi(A, B, G)
  if phi < 0.0:
    return 0.0, pnt_def, pnt_def

  z = wp.sin(phi)
  if end[0] * end[3] - end[1] * end[2] > 0.0:
    vec = end0
    ang = phi - wp.asin(A * z)
  else:
    vec = end1
    ang = phi - wp.asin(B * z)

  vec_n, _ = math.normalize_with_norm(vec)
  pnt = wp.vec2(
    radius * (wp.cos(ang) * vec_n[0] - wp.sin(ang) * vec_n[1]),
    radius * (wp.sin(ang) * vec_n[0] + wp.cos(ang) * vec_n[1]),
  )
  return 0.0, pnt, pnt


@wp.func
def _wrap_ssa(
  x0: wp.vec3, x1: wp.vec3, pos: wp.vec3, mat: wp.mat33, radius: float, geomtype: int, side: wp.vec3
) -> Tuple[float, wp.vec3, wp.vec3]:
  """SSA-pure sphere and cylinder tendon wrap matching util_misc.wrap."""
  if geomtype != WrapType.SPHERE and geomtype != WrapType.CYLINDER:
    return MJ_MAXVAL, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)

  matT = wp.transpose(mat)
  p0 = matT @ (x0 - pos)
  p1 = matT @ (x1 - pos)
  if (wp.norm_l2(p0) < MJ_MINVAL) or (wp.norm_l2(p1) < MJ_MINVAL):
    return -1.0, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)

  if geomtype == WrapType.SPHERE:
    axis0, _ = math.normalize_with_norm(p0)
    normal, nrm = math.normalize_with_norm(wp.cross(p0, p1))
    if nrm < MJ_MINVAL:
      axis0_abs = wp.abs(axis0)
      if (axis0_abs[1] > axis0_abs[0]) and (axis0_abs[1] > axis0_abs[2]):
        axis1_init = wp.vec3(1.0, 0.0, 1.0)
      elif (axis0_abs[2] > axis0_abs[0]) and (axis0_abs[2] > axis0_abs[1]):
        axis1_init = wp.vec3(1.0, 1.0, 0.0)
      else:
        axis1_init = wp.vec3(0.0, 1.0, 1.0)
      normal, _ = math.normalize_with_norm(wp.cross(axis0, axis1_init))
    axis1, _ = math.normalize_with_norm(wp.cross(normal, axis0))
  else:
    axis0 = wp.vec3(1.0, 0.0, 0.0)
    axis1 = wp.vec3(0.0, 1.0, 0.0)

  end = wp.vec4(wp.dot(p0, axis0), wp.dot(p0, axis1), wp.dot(p1, axis0), wp.dot(p1, axis1))
  valid_side = wp.norm_l2(side) < MJ_MAXVAL
  sidepnt = wp.vec3(0.0)
  if valid_side:
    sidepnt = matT @ (side - pos)
    sidepnt_proj, sidenrm = math.normalize_with_norm(wp.vec2(wp.dot(sidepnt, axis0), wp.dot(sidepnt, axis1)))
    if sidenrm < MJ_MINVAL:
      sidepnt_proj = wp.vec2(1.0, 0.0)
    sidepnt_proj = sidepnt_proj * radius
  else:
    sidepnt_proj = wp.vec2(MJ_MAXVAL)

  if valid_side and wp.norm_l2(sidepnt) < radius:
    wlen_2d, pnt0, pnt1 = _wrap_inside_ssa(end, radius)
  else:
    wlen_2d, pnt0, pnt1 = util_misc.wrap_circle(end, sidepnt_proj, radius)

  if wlen_2d < 0.0:
    return -1.0, wp.vec3(MJ_MAXVAL), wp.vec3(MJ_MAXVAL)

  res0_2d = axis0 * pnt0[0] + axis1 * pnt0[1]
  res1_2d = axis0 * pnt1[0] + axis1 * pnt1[1]
  if geomtype == WrapType.CYLINDER:
    L0 = wp.sqrt((p0[0] - res0_2d[0]) * (p0[0] - res0_2d[0]) + (p0[1] - res0_2d[1]) * (p0[1] - res0_2d[1]))
    L1 = wp.sqrt((p1[0] - res1_2d[0]) * (p1[0] - res1_2d[0]) + (p1[1] - res1_2d[1]) * (p1[1] - res1_2d[1]))
    denom = L0 + wlen_2d + L1
    z0 = p0[2] + (p1[2] - p0[2]) * math.safe_div(L0, denom)
    z1 = p0[2] + (p1[2] - p0[2]) * math.safe_div(L0 + wlen_2d, denom)
    res0_out = wp.vec3(res0_2d[0], res0_2d[1], z0)
    res1_out = wp.vec3(res1_2d[0], res1_2d[1], z1)
    dz = z1 - z0
    wlen_out = wp.sqrt(wlen_2d * wlen_2d + dz * dz)
  else:
    res0_out = res0_2d
    res1_out = res1_2d
    wlen_out = wlen_2d

  wpnt0 = mat @ res0_out + pos
  wpnt1 = mat @ res1_out + pos
  return wlen_out, wpnt0, wpnt1


@wp.func
def _wrap_objective(
  # In:
  x0: wp.vec3,
  x1: wp.vec3,
  pos: wp.vec3,
  mat: wp.mat33,
  radius: float,
  geomtype: int,
  side: wp.vec3,
  adj_wlen: float,
  adj_wpnt0: wp.vec3,
  adj_wpnt1: wp.vec3,
):
  wlen, wpnt0, wpnt1 = _wrap_ssa(x0, x1, pos, mat, radius, geomtype, side)
  if wlen < 0.0:
    return 0.0
  return adj_wlen * wlen + wp.dot(adj_wpnt0, wpnt0) + wp.dot(adj_wpnt1, wpnt1)


@wp.func
def _jac_chain_vjp(
  # Model:
  body_parentid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  ten_J_colind: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_ten_J: wp.array2d[float],
  offset: wp.vec3,
  vec: wp.vec3,
  bodyid: int,
  rootid: int,
  rowadr: int,
  rownnz: int,
  scale: float,
  worldid: int,
  # Out:
  adj_subtree_com_out: wp.array2d[wp.vec3],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
) -> Tuple[wp.vec3, wp.vec3]:
  """Reverses _accumulate_jac_chain, returning (adj_pnt, adj_vec) and scattering cdof/com."""
  ptr = rownnz - 1
  bid = bodyid
  adj_vec = wp.vec3(0.0)
  sum_ang = wp.vec3(0.0)
  while bid > 0:
    bdofadr = body_dofadr[bid]
    bdofnum = body_dofnum[bid]
    for k_rev in range(bdofnum):
      dof = bdofadr + bdofnum - 1 - k_rev
      while ptr >= 0:
        sparseid = rowadr + ptr
        if ten_J_colind[sparseid] <= dof:
          break
        ptr -= 1
      if ptr >= 0 and ten_J_colind[sparseid] == dof:
        w_dof = adj_ten_J[worldid, sparseid] * scale
        if w_dof != 0.0:
          cdof = cdof_in[worldid, dof]
          cdof_ang = wp.spatial_top(cdof)
          cdof_lin = wp.spatial_bottom(cdof)
          jacp = cdof_lin + wp.cross(cdof_ang, offset)
          adj_vec += w_dof * jacp
          sum_ang += w_dof * cdof_ang
          wp.atomic_add(
            adj_cdof_out,
            worldid,
            dof,
            wp.spatial_vector(w_dof * wp.cross(offset, vec), w_dof * vec),
          )  # kernel_analyzer: ignore[atomic]
    bid = body_parentid[bid]
  adj_offset = wp.cross(vec, sum_ang)
  if not (adj_offset == wp.vec3(0.0)):
    wp.atomic_add(adj_subtree_com_out, worldid, rootid, -adj_offset)  # kernel_analyzer: ignore[atomic]
  return adj_offset, adj_vec


@wp.func
def _segment_dif_vjp(vec: wp.vec3, length: float, adj_len: float, adj_vec: wp.vec3) -> wp.vec3:
  """Returns d/d(pnt1 - pnt0) of adj_len * length + dot(adj_vec, vec)."""
  if length < MJ_MINVAL:
    return wp.vec3(0.0)
  return adj_len * vec + (adj_vec - vec * wp.dot(vec, adj_vec)) / length


@wp.kernel(enable_backward=False)
def _joint_tendon_vjp(
  # Model:
  jnt_qposadr: wp.array[int],
  wrap_objid: wp.array[int],
  wrap_prm: wp.array[float],
  tendon_jnt_adr: wp.array[int],
  wrap_jnt_adr: wp.array[int],
  # In:
  adj_ten_length: wp.array2d[float],
  # Out:
  adj_qpos_out: wp.array2d[float],
):
  worldid, wrapid = wp.tid()
  tenid = tendon_jnt_adr[wrapid]
  seed = adj_ten_length[worldid, tenid]
  if seed == 0.0:
    return
  wrapjntid = wrap_jnt_adr[wrapid]
  qadr = jnt_qposadr[wrap_objid[wrapjntid]]
  wp.atomic_add(adj_qpos_out[worldid], qadr, wrap_prm[wrapjntid] * seed)  # kernel_analyzer: ignore[atomic]


@wp.kernel(enable_backward=False)
def _spatial_site_tendon_vjp(
  # Model:
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  site_bodyid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  wrap_objid: wp.array[int],
  tendon_site_pair_adr: wp.array[int],
  wrap_site_pair_adr: wp.array[int],
  wrap_pulley_scale: wp.array[float],
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_ten_length: wp.array2d[float],
  adj_ten_J: wp.array2d[float],
  # Out:
  res_site_xpos_out: wp.array2d[wp.vec3],
  adj_subtree_com_out: wp.array2d[wp.vec3],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
):
  worldid, elementid = wp.tid()
  site_pair_adr = wrap_site_pair_adr[elementid]
  tenid = tendon_site_pair_adr[elementid]
  pulley_scale = wrap_pulley_scale[site_pair_adr]

  id0 = wrap_objid[site_pair_adr + 0]
  id1 = wrap_objid[site_pair_adr + 1]
  pnt0 = site_xpos_in[worldid, id0]
  pnt1 = site_xpos_in[worldid, id1]
  vec, length = math.normalize_with_norm(pnt1 - pnt0)
  if length < MJ_MINVAL:
    vec = wp.vec3(1.0, 0.0, 0.0)

  adj_pnt0 = wp.vec3(0.0)
  adj_pnt1 = wp.vec3(0.0)
  adj_vec = wp.vec3(0.0)

  body0 = site_bodyid[id0]
  body1 = site_bodyid[id1]
  if body0 != body1:
    rownnz = ten_J_rownnz[tenid]
    rowadr = ten_J_rowadr[tenid]
    root0 = body_rootid[body0]
    root1 = body_rootid[body1]
    offset0 = pnt0 - subtree_com_in[worldid, root0]
    offset1 = pnt1 - subtree_com_in[worldid, root1]
    dp0, dv0 = _jac_chain_vjp(
      body_parentid,
      body_dofnum,
      body_dofadr,
      ten_J_colind,
      cdof_in,
      adj_ten_J,
      offset0,
      vec,
      body0,
      root0,
      rowadr,
      rownnz,
      -pulley_scale,
      worldid,
      adj_subtree_com_out,
      adj_cdof_out,
    )
    dp1, dv1 = _jac_chain_vjp(
      body_parentid,
      body_dofnum,
      body_dofadr,
      ten_J_colind,
      cdof_in,
      adj_ten_J,
      offset1,
      vec,
      body1,
      root1,
      rowadr,
      rownnz,
      pulley_scale,
      worldid,
      adj_subtree_com_out,
      adj_cdof_out,
    )
    adj_pnt0 += dp0
    adj_pnt1 += dp1
    adj_vec = dv0 + dv1

  adj_dif = _segment_dif_vjp(vec, length, pulley_scale * adj_ten_length[worldid, tenid], adj_vec)
  adj_pnt0 -= adj_dif
  adj_pnt1 += adj_dif
  if not (adj_pnt0 == wp.vec3(0.0)):
    wp.atomic_add(res_site_xpos_out, worldid, id0, adj_pnt0)  # kernel_analyzer: ignore[atomic]
  if not (adj_pnt1 == wp.vec3(0.0)):
    wp.atomic_add(res_site_xpos_out, worldid, id1, adj_pnt1)  # kernel_analyzer: ignore[atomic]


@wp.kernel(enable_backward=False)
def _spatial_geom_tendon_vjp(
  # Model:
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  site_bodyid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  wrap_prm: wp.array[float],
  tendon_geom_adr: wp.array[int],
  wrap_geom_adr: wp.array[int],
  wrap_pulley_scale: wp.array[float],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  site_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_ten_length: wp.array2d[float],
  adj_ten_J: wp.array2d[float],
  # Out:
  res_geom_xpos_out: wp.array2d[wp.vec3],
  res_geom_xmat_out: wp.array2d[wp.mat33],
  res_site_xpos_out: wp.array2d[wp.vec3],
  adj_subtree_com_out: wp.array2d[wp.vec3],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
):
  worldid, elementid = wp.tid()
  wrap_adr = wrap_geom_adr[elementid]
  tenid = tendon_geom_adr[elementid]
  pulley_scale = wrap_pulley_scale[wrap_adr]

  id_site0 = wrap_objid[wrap_adr - 1]
  id_geom = wrap_objid[wrap_adr + 0]
  id_site1 = wrap_objid[wrap_adr + 1]

  site_pnt0 = site_xpos_in[worldid, id_site0]
  site_pnt1 = site_xpos_in[worldid, id_site1]
  geom_xpos = geom_xpos_in[worldid, id_geom]
  geom_xmat = geom_xmat_in[worldid, id_geom]
  geomsize = geom_size[worldid % geom_size.shape[0], id_geom][0]
  geomtype = wrap_type[wrap_adr]

  body_site0 = site_bodyid[id_site0]
  body_geom = geom_bodyid[id_geom]
  body_site1 = site_bodyid[id_site1]

  sideid = int(wp.round(wrap_prm[wrap_adr]))
  if sideid >= 0:
    side = site_xpos_in[worldid, sideid]
  else:
    side = wp.vec3(MJ_MAXVAL)

  length_geomgeom, geom_pnt0, geom_pnt1 = _wrap_ssa(site_pnt0, site_pnt1, geom_xpos, geom_xmat, geomsize, geomtype, side)
  rownnz = ten_J_rownnz[tenid]
  rowadr = ten_J_rowadr[tenid]
  adj_len_scaled = pulley_scale * adj_ten_length[worldid, tenid]

  if length_geomgeom >= 0.0:
    vec_sg, len_sg = math.normalize_with_norm(geom_pnt0 - site_pnt0)
    vec_gs, len_gs = math.normalize_with_norm(site_pnt1 - geom_pnt1)
    vec_sg_use = wp.vec3(1.0, 0.0, 0.0) if len_sg < MJ_MINVAL else vec_sg
    vec_gs_use = wp.vec3(1.0, 0.0, 0.0) if len_gs < MJ_MINVAL else vec_gs

    adj_site0 = wp.vec3(0.0)
    adj_site1 = wp.vec3(0.0)
    adj_geom0 = wp.vec3(0.0)
    adj_geom1 = wp.vec3(0.0)
    adj_vec_sg = wp.vec3(0.0)
    adj_vec_gs = wp.vec3(0.0)

    if body_site0 != body_geom:
      root_s0 = body_rootid[body_site0]
      root_g = body_rootid[body_geom]
      dps0, dv0 = _jac_chain_vjp(
        body_parentid,
        body_dofnum,
        body_dofadr,
        ten_J_colind,
        cdof_in,
        adj_ten_J,
        site_pnt0 - subtree_com_in[worldid, root_s0],
        vec_sg_use,
        body_site0,
        root_s0,
        rowadr,
        rownnz,
        -pulley_scale,
        worldid,
        adj_subtree_com_out,
        adj_cdof_out,
      )
      dpg0, dv1 = _jac_chain_vjp(
        body_parentid,
        body_dofnum,
        body_dofadr,
        ten_J_colind,
        cdof_in,
        adj_ten_J,
        geom_pnt0 - subtree_com_in[worldid, root_g],
        vec_sg_use,
        body_geom,
        root_g,
        rowadr,
        rownnz,
        pulley_scale,
        worldid,
        adj_subtree_com_out,
        adj_cdof_out,
      )
      adj_site0 += dps0
      adj_geom0 += dpg0
      adj_vec_sg = dv0 + dv1

    if body_geom != body_site1:
      root_g = body_rootid[body_geom]
      root_s1 = body_rootid[body_site1]
      dpg1, dv2 = _jac_chain_vjp(
        body_parentid,
        body_dofnum,
        body_dofadr,
        ten_J_colind,
        cdof_in,
        adj_ten_J,
        geom_pnt1 - subtree_com_in[worldid, root_g],
        vec_gs_use,
        body_geom,
        root_g,
        rowadr,
        rownnz,
        -pulley_scale,
        worldid,
        adj_subtree_com_out,
        adj_cdof_out,
      )
      dps1, dv3 = _jac_chain_vjp(
        body_parentid,
        body_dofnum,
        body_dofadr,
        ten_J_colind,
        cdof_in,
        adj_ten_J,
        site_pnt1 - subtree_com_in[worldid, root_s1],
        vec_gs_use,
        body_site1,
        root_s1,
        rowadr,
        rownnz,
        pulley_scale,
        worldid,
        adj_subtree_com_out,
        adj_cdof_out,
      )
      adj_geom1 += dpg1
      adj_site1 += dps1
      adj_vec_gs = dv2 + dv3

    adj_dif_sg = _segment_dif_vjp(vec_sg_use, len_sg, adj_len_scaled, adj_vec_sg)
    adj_dif_gs = _segment_dif_vjp(vec_gs_use, len_gs, adj_len_scaled, adj_vec_gs)
    adj_site0 -= adj_dif_sg
    adj_geom0 += adj_dif_sg
    adj_geom1 -= adj_dif_gs
    adj_site1 += adj_dif_gs

    wrap_grads = wp.grad(_wrap_objective)(
      site_pnt0,
      site_pnt1,
      geom_xpos,
      geom_xmat,
      geomsize,
      geomtype,
      side,
      adj_len_scaled,
      adj_geom0,
      adj_geom1,
    )
    adj_site0 += wrap_grads[0]
    adj_site1 += wrap_grads[1]
    adj_gpos = wrap_grads[2]
    adj_gmat = wrap_grads[3]
    adj_side = wrap_grads[6]

    if not (adj_site0 == wp.vec3(0.0)):
      wp.atomic_add(res_site_xpos_out, worldid, id_site0, adj_site0)  # kernel_analyzer: ignore[atomic]
    if not (adj_site1 == wp.vec3(0.0)):
      wp.atomic_add(res_site_xpos_out, worldid, id_site1, adj_site1)  # kernel_analyzer: ignore[atomic]
    if sideid >= 0 and not (adj_side == wp.vec3(0.0)):
      wp.atomic_add(res_site_xpos_out, worldid, sideid, adj_side)  # kernel_analyzer: ignore[atomic]
    if not (adj_gpos == wp.vec3(0.0)):
      wp.atomic_add(res_geom_xpos_out, worldid, id_geom, adj_gpos)  # kernel_analyzer: ignore[atomic]
    if not (adj_gmat == wp.mat33(0.0)):
      wp.atomic_add(res_geom_xmat_out, worldid, id_geom, adj_gmat)  # kernel_analyzer: ignore[atomic]
  else:
    vec_ss, len_ss = math.normalize_with_norm(site_pnt1 - site_pnt0)
    vec_ss_use = wp.vec3(1.0, 0.0, 0.0) if len_ss < MJ_MINVAL else vec_ss
    adj_site0 = wp.vec3(0.0)
    adj_site1 = wp.vec3(0.0)
    adj_vec_ss = wp.vec3(0.0)
    if body_site0 != body_site1:
      root_s0 = body_rootid[body_site0]
      root_s1 = body_rootid[body_site1]
      dps0, dv0 = _jac_chain_vjp(
        body_parentid,
        body_dofnum,
        body_dofadr,
        ten_J_colind,
        cdof_in,
        adj_ten_J,
        site_pnt0 - subtree_com_in[worldid, root_s0],
        vec_ss_use,
        body_site0,
        root_s0,
        rowadr,
        rownnz,
        -pulley_scale,
        worldid,
        adj_subtree_com_out,
        adj_cdof_out,
      )
      dps1, dv1 = _jac_chain_vjp(
        body_parentid,
        body_dofnum,
        body_dofadr,
        ten_J_colind,
        cdof_in,
        adj_ten_J,
        site_pnt1 - subtree_com_in[worldid, root_s1],
        vec_ss_use,
        body_site1,
        root_s1,
        rowadr,
        rownnz,
        pulley_scale,
        worldid,
        adj_subtree_com_out,
        adj_cdof_out,
      )
      adj_site0 += dps0
      adj_site1 += dps1
      adj_vec_ss = dv0 + dv1

    adj_dif_ss = _segment_dif_vjp(vec_ss_use, len_ss, adj_len_scaled, adj_vec_ss)
    adj_site0 -= adj_dif_ss
    adj_site1 += adj_dif_ss
    if not (adj_site0 == wp.vec3(0.0)):
      wp.atomic_add(res_site_xpos_out, worldid, id_site0, adj_site0)  # kernel_analyzer: ignore[atomic]
    if not (adj_site1 == wp.vec3(0.0)):
      wp.atomic_add(res_site_xpos_out, worldid, id_site1, adj_site1)  # kernel_analyzer: ignore[atomic]


@wp.kernel(enable_backward=False)
def _tendon_pose_dof_vjp(
  # Model:
  ngeom: int,
  nsite: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  site_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  has_geom: bool,
  res_geom_xpos: wp.array2d[wp.vec3],
  res_geom_xmat: wp.array2d[wp.mat33],
  res_site_xpos: wp.array2d[wp.vec3],
  # Out:
  res_dof_out: wp.array2d[float],
):
  w, k = wp.tid()
  acc = float(0.0)
  for s in range(nsite):
    rsp = res_site_xpos[w, s]
    if rsp == wp.vec3(0.0):
      continue
    body = site_bodyid[s]
    if body_isdofancestor[body, k] == 0:
      continue
    jacp, _ = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      site_xpos_in[w, s],
      body,
      k,
      w,
    )
    acc += wp.dot(jacp, rsp)

  if has_geom:
    for g in range(ngeom):
      rgp = res_geom_xpos[w, g]
      rgm = res_geom_xmat[w, g]
      if rgp == wp.vec3(0.0) and rgm == wp.mat33(0.0):
        continue
      body = geom_bodyid[g]
      if body_isdofancestor[body, k] == 0:
        continue
      jacp, jacr = support.jac_dof(
        body_parentid,
        body_rootid,
        dof_bodyid,
        body_isdofancestor,
        subtree_com_in,
        cdof_in,
        geom_xpos_in[w, g],
        body,
        k,
        w,
      )
      tau = adjoint_util._adj_rotation(geom_xmat_in[w, g], rgm)
      acc += wp.dot(jacp, rgp) + wp.dot(jacr, tau)

  res_dof_out[w, k] += acc


def tendon_kinematics_vjp(
  m: Model,
  d: Data,
  qpos_in: wp.array2d[float],
  adj_ten_length: wp.array2d[float],
  adj_ten_J: wp.array2d[float],
  out_qpos: wp.array2d[float],
  bc: BackwardContext,
):
  """Accumulates d(adj_ten_length.T @ ten_length + adj_ten_J.T @ ten_J)/dqpos into out_qpos."""
  bc.res_site_xpos.zero_()
  bc.res_geom_xpos.zero_()
  bc.res_geom_xmat.zero_()
  bc.adj_subtree_com.zero_()
  bc.adj_cdof.zero_()
  bc.smooth_res_dof.zero_()

  if m.wrap_jnt_adr.size and adj_ten_length is not None:
    wp.launch(
      _joint_tendon_vjp,
      dim=(d.nworld, m.wrap_jnt_adr.size),
      inputs=[
        m.jnt_qposadr,
        m.wrap_objid,
        m.wrap_prm,
        m.tendon_jnt_adr,
        m.wrap_jnt_adr,
        adj_ten_length,
      ],
      outputs=[out_qpos],
    )

  spatial_site = m.wrap_site_pair_adr.size > 0
  spatial_geom = m.wrap_geom_adr.size > 0
  if not spatial_site and not spatial_geom:
    return

  len_seed = adj_ten_length if adj_ten_length is not None else bc.adj_ten_length
  J_seed = adj_ten_J if adj_ten_J is not None else bc.adj_ten_J
  if adj_ten_length is None:
    len_seed.zero_()
  if adj_ten_J is None:
    J_seed.zero_()

  if spatial_site:
    wp.launch(
      _spatial_site_tendon_vjp,
      dim=(d.nworld, m.wrap_site_pair_adr.size),
      inputs=[
        m.body_parentid,
        m.body_rootid,
        m.body_dofnum,
        m.body_dofadr,
        m.site_bodyid,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.wrap_objid,
        m.tendon_site_pair_adr,
        m.wrap_site_pair_adr,
        m.wrap_pulley_scale,
        d.site_xpos,
        d.subtree_com,
        d.cdof,
        len_seed,
        J_seed,
      ],
      outputs=[bc.res_site_xpos, bc.adj_subtree_com, bc.adj_cdof],
    )

  if spatial_geom:
    wp.launch(
      _spatial_geom_tendon_vjp,
      dim=(d.nworld, m.wrap_geom_adr.size),
      inputs=[
        m.body_parentid,
        m.body_rootid,
        m.body_dofnum,
        m.body_dofadr,
        m.geom_bodyid,
        m.geom_size,
        m.site_bodyid,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.wrap_type,
        m.wrap_objid,
        m.wrap_prm,
        m.tendon_geom_adr,
        m.wrap_geom_adr,
        m.wrap_pulley_scale,
        d.geom_xpos,
        d.geom_xmat,
        d.site_xpos,
        d.subtree_com,
        d.cdof,
        len_seed,
        J_seed,
      ],
      outputs=[bc.res_geom_xpos, bc.res_geom_xmat, bc.res_site_xpos, bc.adj_subtree_com, bc.adj_cdof],
    )

  if m.nv:
    wp.launch(
      _tendon_pose_dof_vjp,
      dim=(d.nworld, m.nv),
      inputs=[
        m.ngeom,
        m.nsite,
        m.body_parentid,
        m.body_rootid,
        m.dof_bodyid,
        m.geom_bodyid,
        m.site_bodyid,
        m.body_isdofancestor,
        d.geom_xpos,
        d.geom_xmat,
        d.site_xpos,
        d.subtree_com,
        d.cdof,
        spatial_geom,
        bc.res_geom_xpos,
        bc.res_geom_xmat,
        bc.res_site_xpos,
      ],
      outputs=[bc.smooth_res_dof],
    )
    smooth_kinematics_adjoint.kinematics_qpos_backward(
      m,
      d,
      qpos_in,
      bc.adj_cdof,
      bc.adj_subtree_com,
      bc.smooth_res_dof,
      bc.smooth_ceff,
      out_qpos,
    )


# ---------------------------------------------------------------------------
# Shared actuator gain/bias/force helpers & IMPLICITFAST bilinear forms
# ---------------------------------------------------------------------------


@wp.func
def _actuator_ctrl_act(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  actuator_dyntype: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_dynprm: wp.array2d[vec10],
  actuator_actlimited: wp.array[bool],
  actuator_actrange: wp.array2d[wp.vec2],
  actuator_actearly: wp.array[bool],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  # In:
  w: int,
  uid: int,
) -> float:
  """Returns the effective scalar excitation ctrl_act used by _actuator_force."""
  dyntype = actuator_dyntype[uid]
  if dyntype == DynType.NONE:
    ctrl_act = ctrl_in[w % ctrl_in.shape[0], uid]
    if actuator_ctrllimited[uid] and (opt_disableflags & DisableBit.CLAMPCTRL) == 0:
      ctrlrange = actuator_ctrlrange[w % actuator_ctrlrange.shape[0], uid]
      ctrl_act = wp.clamp(ctrl_act, ctrlrange[0], ctrlrange[1])
    return ctrl_act
  adr = actuator_actadr[uid]
  if actuator_actearly[uid]:
    return support.next_act(
      opt_timestep[w % opt_timestep.shape[0]],
      dyntype,
      actuator_dynprm[w % actuator_dynprm.shape[0], uid],
      actuator_actrange[w % actuator_actrange.shape[0], uid],
      act_in[w, adr],
      act_dot_in[w, adr],
      1.0,
      actuator_actlimited[uid],
    )
  return act_in[w, adr]


@wp.func
def _actuator_gain_and_derivs(
  # In:
  gaintype: int,
  biastype: int,
  gainprm: vec10,
  biasprm: vec10,
  acc0: float,
  lengthrange: wp.vec2,
  length: float,
  velocity: float,
  ctrl_act: float,
) -> Tuple[float, float, float, float]:
  """Returns (gain, dg_dvel, dfdl, dfdv) for FIXED, AFFINE, and MUSCLE actuators."""
  gain = float(0.0)
  dg_dl = float(0.0)
  dg_dv = float(0.0)
  if gaintype == GainType.FIXED:
    gain = gainprm[0]
  elif gaintype == GainType.AFFINE:
    gain = gainprm[0] + gainprm[1] * length + gainprm[2] * velocity
    dg_dl = gainprm[1]
    dg_dv = gainprm[2]
  elif gaintype == GainType.MUSCLE:
    gain = util_misc.muscle_gain(length, velocity, lengthrange, acc0, gainprm)
    dg_dl = util_misc.muscle_gain_len_deriv(length, velocity, lengthrange, acc0, gainprm)
    dg_dv = util_misc.muscle_gain_vel_deriv(length, velocity, lengthrange, acc0, gainprm)

  db_dl = float(0.0)
  db_dv = float(0.0)
  if biastype == BiasType.AFFINE:
    db_dl = biasprm[1]
    db_dv = biasprm[2]
  elif biastype == BiasType.MUSCLE:
    db_dl = util_misc.muscle_bias_len_deriv(length, lengthrange, acc0, biasprm)

  return gain, dg_dv, ctrl_act * dg_dl + db_dl, ctrl_act * dg_dv + db_dv


@wp.func
def _muscle_gain_vel_deriv_grads(
  # In:
  length: float,
  velocity: float,
  lengthrange: wp.vec2,
  acc0: float,
  prm: vec10,
) -> Tuple[float, float]:
  """Returns (d/dlength, d/dvelocity) of util_misc.muscle_gain_vel_deriv."""
  range_ = wp.vec2(prm[0], prm[1])
  force = prm[2]
  scale = prm[3]
  lmin = prm[4]
  lmax = prm[5]
  vmax = prm[6]
  fvmax = prm[8]

  if force < 0.0:
    force = scale / wp.max(MJ_MINVAL, acc0)

  L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])
  inv_L0 = 1.0 / wp.max(MJ_MINVAL, L0)
  inv_L0_vmax = 1.0 / wp.max(MJ_MINVAL, L0 * vmax)
  L = range_[0] + (length - lengthrange[0]) * inv_L0
  V = velocity * inv_L0_vmax

  FL = util_misc.muscle_gain_length(L, lmin, lmax)
  dFL = util_misc.muscle_gain_length_deriv(L, lmin, lmax)

  y = fvmax - 1.0
  dFV = float(0.0)
  d2FV = float(0.0)
  if V <= -1.0:
    dFV = 0.0
    d2FV = 0.0
  elif V <= 0.0:
    dFV = 2.0 * V + 2.0
    d2FV = 2.0
  elif V <= y:
    inv_y = 1.0 / wp.max(MJ_MINVAL, y)
    dFV = (-2.0 * V + 2.0 * y) * inv_y
    d2FV = -2.0 * inv_y

  d_dlen = -force * dFL * inv_L0 * dFV * inv_L0_vmax
  d_dvel = -force * FL * d2FV * inv_L0_vmax * inv_L0_vmax
  return d_dlen, d_dvel


@wp.func
def _poly_damp_deriv_objective(v: float, damping: float, dpoly: wp.vec2):
  return util_misc._poly_force_deriv(damping, dpoly, v, 1)


@wp.kernel
def _enable_leaf_adjoints(
  # In:
  x0: wp.vec3,
  mat: wp.mat33,
  dpoly: wp.vec2,
  # Out:
  out: wp.array[float],
):
  """Ensures transitive reverse codegen for @wp.func objectives used via wp.grad."""
  i = wp.tid()
  out[i] = _wrap_objective(x0, x0, x0, mat, 1.0, 0, x0, 1.0, x0, x0) + _poly_damp_deriv_objective(0.0, 1.0, dpoly)


@wp.func
def _moment_in_tree_bilinear(
  # Model:
  M_elemid: wp.array2d[int],
  # Data in:
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  # In:
  y_remap: wp.array2d[float],
  qacc_advance: wp.array2d[float],
  w: int,
  rowadr: int,
  rownnz: int,
) -> float:
  """Contracts y_remap and qacc_advance over in-tree (M_elemid >= 0) moment pairs."""
  acc = float(0.0)
  for i in range(rownnz):
    ei = rowadr + i
    di = moment_colind_in[w, ei]
    mi = actuator_moment_in[w, ei]
    if mi == 0.0:
      continue
    yi = y_remap[w, di]
    ai = qacc_advance[w, di]
    for j in range(i + 1):
      ej = rowadr + j
      dj = moment_colind_in[w, ej]
      mj = actuator_moment_in[w, ej]
      if mj == 0.0:
        continue
      if M_elemid[wp.max(di, dj), wp.min(di, dj)] >= 0:
        pair = yi * qacc_advance[w, dj] if i == j else (yi * qacc_advance[w, dj] + y_remap[w, dj] * ai)
        acc += mi * mj * pair
  return acc


@wp.func
def _ten_J_in_tree_bilinear(
  # Model:
  ten_J_colind: wp.array[int],
  M_elemid: wp.array2d[int],
  # Data in:
  ten_J_in: wp.array2d[float],
  # In:
  y_remap: wp.array2d[float],
  qacc_advance: wp.array2d[float],
  w: int,
  rowadr: int,
  rownnz: int,
) -> float:
  """Contracts y_remap and qacc_advance over in-tree (M_elemid >= 0) tendon Jacobian pairs."""
  acc = float(0.0)
  for i in range(rownnz):
    ei = rowadr + i
    di = ten_J_colind[ei]
    Ji = ten_J_in[w, ei]
    if Ji == 0.0:
      continue
    yi = y_remap[w, di]
    ai = qacc_advance[w, di]
    for j in range(i + 1):
      ej = rowadr + j
      dj = ten_J_colind[ej]
      Jj = ten_J_in[w, ej]
      if Jj == 0.0:
        continue
      if M_elemid[wp.max(di, dj), wp.min(di, dj)] >= 0:
        pair = yi * qacc_advance[w, dj] if i == j else (yi * qacc_advance[w, dj] + y_remap[w, dj] * ai)
        acc += Ji * Jj * pair
  return acc


# ---------------------------------------------------------------------------
# Velocity-space VJPs: off-tree (M_elemid < 0) couplings & IMPLICITFAST d2/dv2
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _actuator_offtree_and_implicit_qvel_vjp(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  actuator_dyntype: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_dynprm: wp.array2d[vec10],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  actuator_actlimited: wp.array[bool],
  actuator_actrange: wp.array2d[wp.vec2],
  actuator_actearly: wp.array[bool],
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  actuator_acc0: wp.array2d[float],
  actuator_lengthrange: wp.array2d[wp.vec2],
  M_elemid: wp.array2d[int],
  # Data in:
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  actuator_velocity_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  qacc_advance: wp.array2d[float],
  y_remap: wp.array2d[float],
  implicitfast: bool,
  # Out:
  adj_qvel_out: wp.array2d[float],
):
  w, uid = wp.tid()
  if actuator_forcelimited[uid]:
    forcerange = actuator_forcerange[w % actuator_forcerange.shape[0], uid]
    force = actuator_force_in[w, uid]
    if force <= forcerange[0] or force >= forcerange[1]:
      return

  ctrl_act = _actuator_ctrl_act(
    opt_timestep,
    opt_disableflags,
    actuator_dyntype,
    actuator_actadr,
    actuator_dynprm,
    actuator_actlimited,
    actuator_actrange,
    actuator_actearly,
    actuator_ctrllimited,
    actuator_ctrlrange,
    act_in,
    ctrl_in,
    act_dot_in,
    w,
    uid,
  )
  gaintype = actuator_gaintype[uid]
  gainprm = actuator_gainprm[w % actuator_gainprm.shape[0], uid]
  biasprm = actuator_biasprm[w % actuator_biasprm.shape[0], uid]
  acc0 = actuator_acc0[w % actuator_acc0.shape[0], uid]
  lengthrange = actuator_lengthrange[w % actuator_lengthrange.shape[0], uid]
  length = actuator_length_in[w, uid]
  velocity = actuator_velocity_in[w, uid]
  _, _, _, dfdv = _actuator_gain_and_derivs(
    gaintype, actuator_biastype[uid], gainprm, biasprm, acc0, lengthrange, length, velocity, ctrl_act
  )

  rownnz = moment_rownnz_in[w, uid]
  rowadr = moment_rowadr_in[w, uid]

  if dfdv != 0.0 and rownnz > 1:
    for i in range(rownnz):
      ei = rowadr + i
      di = moment_colind_in[w, ei]
      mi = actuator_moment_in[w, ei]
      if mi == 0.0:
        continue
      seed_i = dfdv * mi * lam[w, di]
      for j in range(rownnz):
        if i == j:
          continue
        ej = rowadr + j
        dj = moment_colind_in[w, ej]
        if M_elemid[wp.max(di, dj), wp.min(di, dj)] < 0:
          wp.atomic_add(adj_qvel_out[w], dj, seed_i * actuator_moment_in[w, ej])  # kernel_analyzer: ignore[atomic]

  if implicitfast and gaintype == GainType.MUSCLE and ctrl_act != 0.0:
    _, d2g_dv2 = _muscle_gain_vel_deriv_grads(length, velocity, lengthrange, acc0, gainprm)
    if d2g_dv2 != 0.0:
      B_u = _moment_in_tree_bilinear(M_elemid, moment_colind_in, actuator_moment_in, y_remap, qacc_advance, w, rowadr, rownnz)
      scale = opt_timestep[w % opt_timestep.shape[0]] * ctrl_act * d2g_dv2 * B_u
      if scale != 0.0:
        for j in range(rownnz):
          ej = rowadr + j
          wp.atomic_add(
            adj_qvel_out[w], moment_colind_in[w, ej], scale * actuator_moment_in[w, ej]
          )  # kernel_analyzer: ignore[atomic]


@wp.kernel(enable_backward=False)
def _tendon_damper_offtree_and_implicit_qvel_vjp(
  # Model:
  opt_timestep: wp.array[float],
  tendon_actuatorid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  M_elemid: wp.array2d[int],
  # Data in:
  ten_J_in: wp.array2d[float],
  ten_velocity_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  qacc_advance: wp.array2d[float],
  y_remap: wp.array2d[float],
  implicitfast: bool,
  # Out:
  adj_qvel_out: wp.array2d[float],
):
  w, tenid = wp.tid()
  damping, dpoly = util_misc.tendon_damping_total(
    tendon_actuatorid,
    tendon_damping,
    tendon_dampingpoly,
    actuator_trntype,
    actuator_trnid,
    actuator_damping,
    actuator_dampingpoly,
    actuator_gear,
    w,
    tenid,
  )
  if damping == 0.0 and dpoly[0] == 0.0 and dpoly[1] == 0.0:
    return

  rownnz = ten_J_rownnz[tenid]
  rowadr = ten_J_rowadr[tenid]
  v = ten_velocity_in[w, tenid]
  damp_deriv = util_misc._poly_force_deriv(damping, dpoly, v, 1)

  if damp_deriv != 0.0 and rownnz > 1:
    for i in range(rownnz):
      ei = rowadr + i
      di = ten_J_colind[ei]
      Ji = ten_J_in[w, ei]
      if Ji == 0.0:
        continue
      seed_i = -damp_deriv * Ji * lam[w, di]
      for j in range(rownnz):
        if i == j:
          continue
        ej = rowadr + j
        dj = ten_J_colind[ej]
        if M_elemid[wp.max(di, dj), wp.min(di, dj)] < 0:
          wp.atomic_add(adj_qvel_out[w], dj, seed_i * ten_J_in[w, ej])  # kernel_analyzer: ignore[atomic]

  if implicitfast and (dpoly[0] != 0.0 or dpoly[1] != 0.0):
    dp_grads = wp.grad(_poly_damp_deriv_objective)(v, damping, dpoly)
    d2p_dv2 = dp_grads[0]
    if d2p_dv2 != 0.0:
      B_ten = _ten_J_in_tree_bilinear(ten_J_colind, M_elemid, ten_J_in, y_remap, qacc_advance, w, rowadr, rownnz)
      scale = -opt_timestep[w % opt_timestep.shape[0]] * d2p_dv2 * B_ten
      if scale != 0.0:
        for j in range(rownnz):
          ej = rowadr + j
          wp.atomic_add(adj_qvel_out[w], ten_J_colind[ej], scale * ten_J_in[w, ej])  # kernel_analyzer: ignore[atomic]


def tendon_muscle_qvel_backward(
  m: Model, d_bwd: Data, lam: wp.array2d[float], adj_qvel: wp.array2d[float], bc: BackwardContext
):
  """Adds off-tree velocity couplings and IMPLICITFAST nonlinear damping/muscle qvel VJPs."""
  flags = DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER
  implicitfast = m.opt.integrator == IntegratorType.IMPLICITFAST and (m.opt.disableflags & flags) != flags
  if m.nu and not (m.opt.disableflags & DisableBit.ACTUATION):
    wp.launch(
      _actuator_offtree_and_implicit_qvel_vjp,
      dim=(d_bwd.nworld, m.nu),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.actuator_dyntype,
        m.actuator_gaintype,
        m.actuator_biastype,
        m.actuator_actadr,
        m.actuator_dynprm,
        m.actuator_gainprm,
        m.actuator_biasprm,
        m.actuator_actlimited,
        m.actuator_actrange,
        m.actuator_actearly,
        m.actuator_forcelimited,
        m.actuator_forcerange,
        m.actuator_ctrllimited,
        m.actuator_ctrlrange,
        m.actuator_acc0,
        m.actuator_lengthrange,
        m.M_elemid,
        d_bwd.act,
        d_bwd.ctrl,
        d_bwd.act_dot,
        d_bwd.actuator_length,
        d_bwd.moment_rownnz,
        d_bwd.moment_rowadr,
        d_bwd.moment_colind,
        d_bwd.actuator_moment,
        d_bwd.actuator_velocity,
        d_bwd.actuator_force,
        lam,
        bc.qacc_advance,
        bc.y_remap,
        implicitfast,
      ],
      outputs=[adj_qvel],
    )
  if m.ntendon and not (m.opt.disableflags & DisableBit.DAMPER):
    wp.launch(
      _tendon_damper_offtree_and_implicit_qvel_vjp,
      dim=(d_bwd.nworld, m.ntendon),
      inputs=[
        m.opt.timestep,
        m.tendon_actuatorid,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_damping,
        m.tendon_dampingpoly,
        m.actuator_trntype,
        m.actuator_trnid,
        m.actuator_damping,
        m.actuator_dampingpoly,
        m.actuator_gear,
        m.M_elemid,
        d_bwd.ten_J,
        d_bwd.ten_velocity,
        lam,
        bc.qacc_advance,
        bc.y_remap,
        implicitfast,
      ],
      outputs=[adj_qvel],
    )


# ---------------------------------------------------------------------------
# Position-space tendon passive & tendon-actuator residual VJPs
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _tendon_passive_residual_seed(
  # Model:
  opt_timestep: wp.array[float],
  tendon_actuatorid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_stiffness: wp.array2d[float],
  tendon_stiffnesspoly: wp.array2d[wp.vec2],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  tendon_lengthspring: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  M_elemid: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  ten_J_in: wp.array2d[float],
  ten_length_in: wp.array2d[float],
  ten_velocity_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  qacc_advance: wp.array2d[float],
  y_remap: wp.array2d[float],
  dsbl_spring: bool,
  dsbl_damper: bool,
  implicitfast: bool,
  # Out:
  adj_ten_length_out: wp.array2d[float],
  adj_ten_J_out: wp.array2d[float],
):
  w, tenid = wp.tid()
  stiffness = tendon_stiffness[w % tendon_stiffness.shape[0], tenid]
  spoly = tendon_stiffnesspoly[w % tendon_stiffnesspoly.shape[0], tenid]
  damping, dpoly = util_misc.tendon_damping_total(
    tendon_actuatorid,
    tendon_damping,
    tendon_dampingpoly,
    actuator_trntype,
    actuator_trnid,
    actuator_damping,
    actuator_dampingpoly,
    actuator_gear,
    w,
    tenid,
  )
  has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and not dsbl_spring
  has_damping = (damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0) and not dsbl_damper
  if not has_stiffness and not has_damping:
    return

  rownnz = ten_J_rownnz[tenid]
  rowadr = ten_J_rowadr[tenid]
  J_lam = float(0.0)
  for k in range(rownnz):
    sparseid = rowadr + k
    J_lam += ten_J_in[w, sparseid] * lam[w, ten_J_colind[sparseid]]

  frc_total = float(0.0)
  vel_seed = float(0.0)

  if has_stiffness:
    length = ten_length_in[w, tenid]
    lengthspring = tendon_lengthspring[w % tendon_lengthspring.shape[0], tenid]
    lower = lengthspring[0]
    upper = lengthspring[1]
    if length > upper:
      x = length - upper
      frc_total += -x * util_misc._poly_force(stiffness, spoly, x, 0)
      dfrc_dlen = -util_misc._poly_force_deriv(stiffness, spoly, x, 0)
      adj_ten_length_out[w, tenid] -= J_lam * dfrc_dlen
    elif length < lower:
      x = length - lower
      frc_total += -x * util_misc._poly_force(stiffness, spoly, x, 0)
      dfrc_dlen = -util_misc._poly_force_deriv(stiffness, spoly, x, 0)
      adj_ten_length_out[w, tenid] -= J_lam * dfrc_dlen

  damp_deriv = float(0.0)
  if has_damping:
    v = ten_velocity_in[w, tenid]
    frc_total += -v * util_misc._poly_force(damping, dpoly, v, 1)
    damp_deriv = util_misc._poly_force_deriv(damping, dpoly, v, 1)
    vel_seed += -J_lam * damp_deriv
    if implicitfast and (dpoly[0] != 0.0 or dpoly[1] != 0.0):
      dp_grads = wp.grad(_poly_damp_deriv_objective)(v, damping, dpoly)
      d2p_dv2 = dp_grads[0]
      if d2p_dv2 != 0.0:
        B_ten = _ten_J_in_tree_bilinear(ten_J_colind, M_elemid, ten_J_in, y_remap, qacc_advance, w, rowadr, rownnz)
        vel_seed += -opt_timestep[w % opt_timestep.shape[0]] * d2p_dv2 * B_ten

  dt = opt_timestep[w % opt_timestep.shape[0]]
  for i in range(rownnz):
    ei = rowadr + i
    di = ten_J_colind[ei]
    g_J = frc_total * lam[w, di] + vel_seed * qvel_in[w, di]
    if has_damping and implicitfast and damp_deriv != 0.0:
      yi = y_remap[w, di]
      ai = qacc_advance[w, di]
      q_sum = float(0.0)
      for j in range(rownnz):
        ej = rowadr + j
        dj = ten_J_colind[ej]
        if M_elemid[wp.max(di, dj), wp.min(di, dj)] >= 0:
          q_sum += ten_J_in[w, ej] * (yi * qacc_advance[w, dj] + y_remap[w, dj] * ai)
      g_J += -dt * damp_deriv * q_sum
    if g_J != 0.0:
      adj_ten_J_out[w, ei] -= g_J


@wp.kernel(enable_backward=False)
def _tendon_actuator_residual_seed(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  jnt_actfrclimited: wp.array[bool],
  jnt_actfrcrange: wp.array2d[wp.vec2],
  dof_jntid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  actuator_trntype: wp.array[int],
  actuator_dyntype: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_dynprm: wp.array2d[vec10],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  actuator_actlimited: wp.array[bool],
  actuator_actrange: wp.array2d[wp.vec2],
  actuator_actearly: wp.array[bool],
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  actuator_acc0: wp.array2d[float],
  actuator_lengthrange: wp.array2d[wp.vec2],
  M_elemid: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  ten_J_in: wp.array2d[float],
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
  # Out:
  adj_ten_length_out: wp.array2d[float],
  adj_ten_J_out: wp.array2d[float],
):
  w, uid = wp.tid()
  if actuator_trntype[uid] != TrnType.TENDON:
    return

  tenid = actuator_trnid[uid][0]
  gear0 = actuator_gear[w % actuator_gear.shape[0], uid][0]
  force = actuator_force_in[w, uid]
  unclamped_force = bool(True)
  if actuator_forcelimited[uid]:
    forcerange = actuator_forcerange[w % actuator_forcerange.shape[0], uid]
    unclamped_force = force > forcerange[0] and force < forcerange[1]

  rownnz_ten = ten_J_rownnz[tenid]
  rowadr_ten = ten_J_rowadr[tenid]
  model_w = w % jnt_actfrcrange.shape[0]

  moment_lam = float(0.0)
  for k in range(rownnz_ten):
    dofid = ten_J_colind[rowadr_ten + k]
    jntid = dof_jntid[dofid]
    clamped = bool(False)
    if jnt_actfrclimited[jntid]:
      bounds = jnt_actfrcrange[model_w, jntid]
      qf = qfrc_actuator_in[w, dofid]
      clamped = qf <= bounds[0] or qf >= bounds[1]
    if not clamped:
      moment_lam += gear0 * ten_J_in[w, rowadr_ten + k] * lam[w, dofid]

  g_act_len = float(0.0)
  g_act_vel = float(0.0)
  dfdv = float(0.0)
  dt = opt_timestep[w % opt_timestep.shape[0]]

  if unclamped_force:
    ctrl_act = _actuator_ctrl_act(
      opt_timestep,
      opt_disableflags,
      actuator_dyntype,
      actuator_actadr,
      actuator_dynprm,
      actuator_actlimited,
      actuator_actrange,
      actuator_actearly,
      actuator_ctrllimited,
      actuator_ctrlrange,
      act_in,
      ctrl_in,
      act_dot_in,
      w,
      uid,
    )
    gaintype = actuator_gaintype[uid]
    gainprm = actuator_gainprm[w % actuator_gainprm.shape[0], uid]
    biasprm = actuator_biasprm[w % actuator_biasprm.shape[0], uid]
    acc0 = actuator_acc0[w % actuator_acc0.shape[0], uid]
    lengthrange = actuator_lengthrange[w % actuator_lengthrange.shape[0], uid]
    length = actuator_length_in[w, uid]
    velocity = actuator_velocity_in[w, uid]
    _, _, dfdl, dfdv = _actuator_gain_and_derivs(
      gaintype, actuator_biastype[uid], gainprm, biasprm, acc0, lengthrange, length, velocity, ctrl_act
    )
    g_act_len += moment_lam * dfdl
    g_act_vel += moment_lam * dfdv

    if implicitfast and gaintype == GainType.MUSCLE and ctrl_act != 0.0:
      mg_dlen, mg_dvel = _muscle_gain_vel_deriv_grads(length, velocity, lengthrange, acc0, gainprm)
      B_u = _moment_in_tree_bilinear(
        M_elemid,
        moment_colind_in,
        actuator_moment_in,
        y_remap,
        qacc_advance,
        w,
        moment_rowadr_in[w, uid],
        moment_rownnz_in[w, uid],
      )
      scale_imp = dt * ctrl_act * B_u
      g_act_len += scale_imp * mg_dlen
      g_act_vel += scale_imp * mg_dvel

  if g_act_len != 0.0:
    wp.atomic_add(adj_ten_length_out[w], tenid, -g_act_len * gear0)  # kernel_analyzer: ignore[atomic]

  for i in range(rownnz_ten):
    ei = rowadr_ten + i
    di = ten_J_colind[ei]
    jntid = dof_jntid[di]
    clamped = bool(False)
    if jnt_actfrclimited[jntid]:
      bounds = jnt_actfrcrange[model_w, jntid]
      qf = qfrc_actuator_in[w, di]
      clamped = qf <= bounds[0] or qf >= bounds[1]
    lam_eff = 0.0 if clamped else lam[w, di]
    g_J = (gear0 * force) * lam_eff + (g_act_vel * gear0) * qvel_in[w, di]
    if unclamped_force and implicitfast and dfdv != 0.0:
      yi = y_remap[w, di]
      ai = qacc_advance[w, di]
      q_sum = float(0.0)
      for j in range(rownnz_ten):
        ej = rowadr_ten + j
        dj = ten_J_colind[ej]
        if M_elemid[wp.max(di, dj), wp.min(di, dj)] >= 0:
          q_sum += ten_J_in[w, ej] * (yi * qacc_advance[w, dj] + y_remap[w, dj] * ai)
      g_J += dt * gear0 * gear0 * dfdv * q_sum
    if g_J != 0.0:
      wp.atomic_add(adj_ten_J_out[w], ei, -g_J)  # kernel_analyzer: ignore[atomic]


@event_scope
def tendon_qpos_vjp(m: Model, d: Data, lam: wp.array2d[float], bc: BackwardContext | None = None):
  """Returns the residual position VJP -d(lam.T @ qfrc_tendon_and_tendon_act)/dqpos."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  smooth_res_qpos = bc.smooth_res_qpos
  smooth_res_qpos.zero_()
  if not m.ntendon:
    return smooth_res_qpos

  bc.adj_ten_length.zero_()
  bc.adj_ten_J.zero_()
  flags = DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER
  implicitfast = m.opt.integrator == IntegratorType.IMPLICITFAST and (m.opt.disableflags & flags) != flags
  dsbl_spring = bool(m.opt.disableflags & DisableBit.SPRING)
  dsbl_damper = bool(m.opt.disableflags & DisableBit.DAMPER)

  if not (dsbl_spring and dsbl_damper):
    wp.launch(
      _tendon_passive_residual_seed,
      dim=(d.nworld, m.ntendon),
      inputs=[
        m.opt.timestep,
        m.tendon_actuatorid,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_stiffness,
        m.tendon_stiffnesspoly,
        m.tendon_damping,
        m.tendon_dampingpoly,
        m.tendon_lengthspring,
        m.actuator_trntype,
        m.actuator_trnid,
        m.actuator_damping,
        m.actuator_dampingpoly,
        m.actuator_gear,
        m.M_elemid,
        d.qvel,
        d.ten_J,
        d.ten_length,
        d.ten_velocity,
        lam,
        bc.qacc_advance,
        bc.y_remap,
        dsbl_spring,
        dsbl_damper,
        implicitfast,
      ],
      outputs=[bc.adj_ten_length, bc.adj_ten_J],
    )

  if m.nu and not (m.opt.disableflags & DisableBit.ACTUATION):
    wp.launch(
      _tendon_actuator_residual_seed,
      dim=(d.nworld, m.nu),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.jnt_actfrclimited,
        m.jnt_actfrcrange,
        m.dof_jntid,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.actuator_trntype,
        m.actuator_dyntype,
        m.actuator_gaintype,
        m.actuator_biastype,
        m.actuator_actadr,
        m.actuator_trnid,
        m.actuator_dynprm,
        m.actuator_gainprm,
        m.actuator_biasprm,
        m.actuator_actlimited,
        m.actuator_actrange,
        m.actuator_actearly,
        m.actuator_forcelimited,
        m.actuator_forcerange,
        m.actuator_ctrllimited,
        m.actuator_ctrlrange,
        m.actuator_gear,
        m.actuator_acc0,
        m.actuator_lengthrange,
        m.M_elemid,
        d.qvel,
        d.act,
        d.ctrl,
        d.act_dot,
        d.ten_J,
        d.actuator_length,
        d.moment_rownnz,
        d.moment_rowadr,
        d.moment_colind,
        d.actuator_moment,
        d.actuator_velocity,
        d.actuator_force,
        d.qfrc_actuator,
        lam,
        bc.qacc_advance,
        bc.y_remap,
        implicitfast,
      ],
      outputs=[bc.adj_ten_length, bc.adj_ten_J],
    )

  tendon_kinematics_vjp(m, d, d.qpos, bc.adj_ten_length, bc.adj_ten_J, smooth_res_qpos, bc)
  return smooth_res_qpos


@event_scope
def tendon_kinematics_backward(m: Model, d_out: Data, d_bwd: Data, bc: BackwardContext):
  """Accumulates forward-kinematics cotangents from ten_length.grad and ten_J.grad."""
  if not m.ntendon:
    return
  len_grad = d_out.ten_length.grad if d_out.ten_length.requires_grad else None
  J_grad = d_out.ten_J.grad if d_out.ten_J.requires_grad else None
  if len_grad is None and J_grad is None:
    return
  tendon_kinematics_vjp(m, d_bwd, d_bwd.qpos, len_grad, J_grad, bc.adj_qpos, bc)
  if (d_bwd.mocap_pos.grad is not None or d_bwd.mocap_quat.grad is not None) and (
    m.wrap_site_pair_adr.size > 0 or m.wrap_geom_adr.size > 0
  ):
    smooth_kinematics_adjoint.kinematics_mocap_backward(m, d_bwd, bc, tendon=True, scale=1.0)
