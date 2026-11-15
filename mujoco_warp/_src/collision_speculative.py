# Copyright 2026 The Newton Developers
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

"""Speculative continuous collision detection for fast-moving geoms."""

from typing import Tuple

import warp as wp

from mujoco_warp._src.collision_convex import ccd_grid_size
from mujoco_warp._src.collision_core import CollisionContext
from mujoco_warp._src.collision_core import Geom
from mujoco_warp._src.collision_core import allocate_multiccd_scratch
from mujoco_warp._src.collision_core import contact_margin
from mujoco_warp._src.collision_core import contact_material_params
from mujoco_warp._src.collision_core import make_gjk_geom
from mujoco_warp._src.collision_core import multiccd_scratch_dims
from mujoco_warp._src.collision_core import set_mesh_poly
from mujoco_warp._src.collision_core import write_contact_entry
from mujoco_warp._src.collision_gjk import GJKResult
from mujoco_warp._src.collision_gjk import gjk
from mujoco_warp._src.collision_gjk import gjk_simplex_features
from mujoco_warp._src.collision_gjk import multicontact_from_features
from mujoco_warp._src.collision_gjk import support
from mujoco_warp._src.collision_gjk import supports_multiccd
from mujoco_warp._src.math import axis_angle_to_quat
from mujoco_warp._src.math import make_frame
from mujoco_warp._src.math import quat_to_mat
from mujoco_warp._src.math import rot_vec_quat
from mujoco_warp._src.math import safe_div
from mujoco_warp._src.math import upper_tri_index
from mujoco_warp._src.ray import ray_geom
from mujoco_warp._src.ray import ray_mesh
from mujoco_warp._src.ray import ray_sphere
from mujoco_warp._src.types import MJ_MAXIMP
from mujoco_warp._src.types import MJ_MINIMP
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import ContactType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import mat43
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Maximum conservative advancement iterations per geom pair (matches MuJoCo C)
_MAX_CA_ITERATIONS = 16

# Indices into CollisionContext.spec_state
_SPEC_NFAST = 0
_SPEC_NCCD = 1
_SPEC_ANY_HIT = 2
_SPEC_NACON_START = 3
_SPEC_NACON_DISCRETE = 4

# Per-pair state values in CollisionContext.spec_pair_state
_PAIR_PENDING = 0
_PAIR_DISCRETE_ACTIVE = 1
_PAIR_SPEC_HIT = 2


def prepare_speculative_context(
  m: Model,
  d: Data,
  ctx: CollisionContext,
  convex_pairs: list[tuple[GeomType, GeomType]] | None = None,
):
  """Pre-allocate speculative collision scratch arrays on ctx outside graph capture."""
  ctx.geom_dx = wp.empty((d.nworld, m.ngeom), dtype=wp.vec3)
  ctx.geom_dw = wp.empty((d.nworld, m.ngeom), dtype=wp.vec3)
  ctx.geom_sweep = wp.empty((d.nworld, m.ngeom), dtype=float)

  table_size = 1 << max(1, (2 * d.naconmax - 1).bit_length())
  ctx.spec_pair_table = wp.full(table_size, -1, dtype=int)
  ctx.spec_fast_cid = wp.empty(d.naconmax, dtype=int)
  ctx.spec_pair_state = wp.empty(d.naconmax, dtype=int)
  ctx.spec_state = wp.zeros(5, dtype=int)

  # Ensure multicontact scratch arrays cover both discrete convex and speculative pair types
  npolygonmax, nmeshdegmax = multiccd_scratch_dims(m, convex_pairs, speculative=True)
  allocate_multiccd_scratch(ctx, d.naccdmax, npolygonmax, nmeshdegmax)


@wp.func
def _geom_supported(geomtype: int) -> bool:
  """Return True if geomtype is supported by speculative collision detection."""
  return geomtype != GeomType.HFIELD and geomtype != GeomType.SDF and geomtype != GeomType.FLEX


@wp.func
def pair_min_size(type1: int, type2: int, aabb_size1: wp.vec3, aabb_size2: wp.vec3) -> float:
  """Return minimum AABB half-thickness across a geom pair."""
  s1 = wp.min(aabb_size1[0], wp.min(aabb_size1[1], aabb_size1[2]))
  s2 = wp.min(aabb_size2[0], wp.min(aabb_size2[1], aabb_size2[2]))
  if type1 == GeomType.PLANE:
    return s2
  if type2 == GeomType.PLANE:
    return s1
  return wp.min(s1, s2)


@wp.func
def _shift_pair_to_local(obj1: Geom, obj2: Geom, type1: int, type2: int) -> Tuple[Geom, Geom, wp.vec3]:
  """Shift a geom pair into pair-local coordinates for float32 translation invariance."""
  origin = wp.where(type1 == GeomType.PLANE, obj2.pos, obj1.pos)
  obj1.pos = obj1.pos - origin
  obj2.pos = obj2.pos - origin
  if type1 == GeomType.PLANE:
    n1 = obj1.rot[:, 2]
    obj1.pos = wp.dot(n1, obj1.pos) * n1
  elif type2 == GeomType.PLANE:
    n2 = obj2.rot[:, 2]
    obj2.pos = wp.dot(n2, obj2.pos) * n2
  return obj1, obj2, origin


@wp.func
def _advance_geom_and_witness(obj: Geom, x: wp.vec3, dx: wp.vec3, dw: wp.vec3, step: float) -> Tuple[Geom, wp.vec3]:
  """Advance a rigid geom pose by step * (dx, dw) and transport witness point x with it."""
  local = wp.transpose(obj.rot) @ (x - obj.pos)
  obj.pos = obj.pos + step * dx
  wnorm = wp.norm_l2(dw)
  if wnorm * step >= MJ_MINVAL:
    dq = axis_angle_to_quat(dw / wnorm, wnorm * step)
    obj.rot = quat_to_mat(dq) @ obj.rot
  return obj, obj.pos + obj.rot @ local


@wp.func
def _eval_plane_dist(
  plane_obj: Geom, other_obj: Geom, other_type: int, other_r: float
) -> Tuple[float, wp.vec3, wp.vec3, wp.vec3, int, int]:
  """Evaluate analytical distance, witnesses (plane, other), normal, and support indices."""
  n = plane_obj.rot[:, 2]
  sp = support(other_obj, other_type, -n)
  x_other = sp.point - other_r * n
  dist = wp.dot(n, x_other - plane_obj.pos)
  x_plane = x_other - dist * n
  return dist, x_plane, x_other, n, sp.vertex_index, sp.cached_index


@wp.func
def _eval_geom_pair_dist(
  # In:
  obj1: Geom,
  obj2: Geom,
  type1: int,
  type2: int,
  r1: float,
  r2: float,
  distmax: float,
  x1_init: wp.vec3,
  x2_init: wp.vec3,
  normal_init: wp.vec3,
  ccd_tolerance: float,
  ccd_iterations: int,
) -> Tuple[float, wp.vec3, wp.vec3, wp.vec3, GJKResult]:
  """Evaluate distance, witnesses (x1, x2), normal (g1 to g2), and GJK simplex at current poses."""
  # Shrink local copies to sphere point / capsule line segment without mutating caller structs
  obj1_s = obj1
  obj2_s = obj2
  obj1_s.margin = 0.0
  obj2_s.margin = 0.0
  if type1 == GeomType.SPHERE or type1 == GeomType.CAPSULE:
    obj1_s.size = wp.vec3(0.0, obj1.size[1], obj1.size[2])
  if type2 == GeomType.SPHERE or type2 == GeomType.CAPSULE:
    obj2_s.size = wp.vec3(0.0, obj2.size[1], obj2.size[2])

  # Handle planes analytically (GJK does not support infinite planes)
  if type1 == GeomType.PLANE or type2 == GeomType.PLANE:
    res = GJKResult()
    res.dim = 1
    dist = float(0.0)
    x1 = wp.vec3()
    x2 = wp.vec3()
    normal = wp.vec3()
    if type1 == GeomType.PLANE:
      dist, x1, x2, normal, vert_idx, cache_idx = _eval_plane_dist(obj1, obj2_s, type2, r2)
      res.simplex_index1 = wp.vec4i(0, 0, 0, 0)
      res.simplex_index2 = wp.vec4i(vert_idx, 0, 0, 0)
      res.index1 = -1
      res.index2 = cache_idx
    else:
      dist, x2, x1, n_plane, vert_idx, cache_idx = _eval_plane_dist(obj2, obj1_s, type1, r1)
      normal = -n_plane
      res.simplex_index1 = wp.vec4i(vert_idx, 0, 0, 0)
      res.simplex_index2 = wp.vec4i(0, 0, 0, 0)
      res.index1 = cache_idx
      res.index2 = -1
    res.dist = dist
    res.x1 = x1
    res.x2 = x2
    return dist, x1, x2, normal, res

  cutoff = distmax + r1 + r2
  # Shrunk spheres/capsules, boxes, and meshes have finite vertices; ellipsoids/cylinders are smooth
  is_discrete = (type1 != GeomType.ELLIPSOID and type1 != GeomType.CYLINDER) and (
    type2 != GeomType.ELLIPSOID and type2 != GeomType.CYLINDER
  )
  res = gjk(ccd_tolerance, ccd_iterations, obj1_s, obj2_s, x1_init, x2_init, type1, type2, cutoff, is_discrete)

  normal = normal_init
  x1 = res.x1
  x2 = res.x2
  if res.dist <= cutoff:
    # In float32, res.dist == 0 (overlapping 4-simplex) leaves x2 - x1 as rounding noise;
    # keep normal_init when non-separated or below MJ_MINVAL.
    if res.dist > MJ_MINVAL:
      diff = x2 - x1
      dnorm = wp.norm_l2(diff)
      if dnorm > MJ_MINVAL:
        normal = diff / dnorm
    x1 = x1 + r1 * normal
    x2 = x2 - r2 * normal

  return res.dist - (r1 + r2), x1, x2, normal, res


@wp.func
def _contact_dist(
  # In:
  margin: float,
  t: float,
  closing_dist: float,
  solref: wp.vec2,
  solimp: vec5,
  dt: float,
  is_discrete: bool,
  disable_refsafe: bool,
) -> float:
  """Compute effective speculative contact distance to arrest (1 - t) of normal closing velocity."""
  if (solref[0] > 0.0) != (solref[1] > 0.0):
    solref = wp.vec2(0.02, 1.0)

  imp = wp.clamp(solimp[1], MJ_MINIMP, MJ_MAXIMP)
  delta = wp.max(1.0e-4, solimp[2])
  if dt > 0.0:
    k0 = float(0.0)
    b0 = float(0.0)
    if solref[0] > 0.0:
      if not disable_refsafe and not is_discrete:
        tau_ref = wp.max(2.0 * dt, solref[0])
      else:
        tau_ref = wp.max(MJ_MINVAL, solref[0])
      d_ratio = wp.max(MJ_MINVAL, solref[1])
      k0 = (dt * dt) / wp.max(MJ_MINVAL, tau_ref * tau_ref * d_ratio * d_ratio)
      b0 = 2.0 * dt / tau_ref
    else:
      k0 = -solref[0] * dt * dt
      b0 = -solref[1] * dt

    if is_discrete:
      if not disable_refsafe and solref[0] > 0.0 and k0 > imp:
        excess = k0 / imp
        fmax = MJ_MAXIMP * (1.0 - imp) / wp.max(MJ_MINVAL, imp * (1.0 - MJ_MAXIMP))
        k0 = imp
        b0 = wp.min(b0 / wp.sqrt(excess), imp * wp.max(0.0, fmax - 2.0))
      if k0 > MJ_MINVAL:
        rem = 1.0 - t - t * (b0 + k0)
        delta = (rem / k0) * closing_dist
    elif k0 > MJ_MINVAL:
      rem = 1.0 - t - b0
      delta = (rem / k0) * closing_dist

  return margin - delta


@wp.func
def _pair_hash(worldid: int, g1: int, g2: int, mask: int) -> int:
  """Hash a (worldid, geom1, geom2) key into [0, mask]."""
  h = (
    (wp.uint32(worldid) * wp.uint32(0x9E3779B1))
    ^ (wp.uint32(g1) * wp.uint32(0x85EBCA77))
    ^ (wp.uint32(g2) * wp.uint32(0xC2B2AE3D))
  )
  h ^= h >> wp.uint32(16)
  return int(h & wp.uint32(mask))


@wp.func
def _lookup_collisionid(
  # In:
  worldid: int,
  geoms: wp.vec2i,
  collision_pair_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
  spec_pair_table_in: wp.array[int],
) -> int:
  """Look up fast broadphase pair index for (worldid, geoms), or -1 if absent."""
  table_size = spec_pair_table_in.shape[0]
  mask = table_size - 1
  slot = _pair_hash(worldid, geoms[0], geoms[1], mask)
  for _ in range(table_size):
    cid = spec_pair_table_in[slot]
    if cid < 0:
      return -1
    pair = collision_pair_in[cid]
    if collision_worldid_in[cid] == worldid and pair[0] == geoms[0] and pair[1] == geoms[1]:
      return cid
    slot = (slot + 1) & mask
  return -1


@wp.func
def _is_geom_pair_within_margin(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xquat_in: wp.array2d[wp.quat],
  # In:
  g1: int,
  g2: int,
  margin: float,
  ccd_tolerance: float,
  ccd_iterations: int,
  worldid: int,
) -> bool:
  """Return True if geoms g1 and g2 are within margin + ccd_tolerance at t = 0."""
  type1 = geom_type[g1]
  type2 = geom_type[g2]
  if type1 == GeomType.PLANE and type2 == GeomType.PLANE:
    return False

  obj1 = make_gjk_geom(
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_vertnum,
    mesh_graphadr,
    mesh_vert,
    mesh_graph,
    geom_xpos_in,
    geom_xquat_in,
    type1,
    g1,
    worldid,
  )
  obj2 = make_gjk_geom(
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_vertnum,
    mesh_graphadr,
    mesh_vert,
    mesh_graph,
    geom_xpos_in,
    geom_xquat_in,
    type2,
    g2,
    worldid,
  )
  obj1, obj2, _origin = _shift_pair_to_local(obj1, obj2, type1, type2)

  r1 = wp.where(type1 == GeomType.SPHERE or type1 == GeomType.CAPSULE, obj1.size[0], 0.0)
  r2 = wp.where(type2 == GeomType.SPHERE or type2 == GeomType.CAPSULE, obj2.size[0], 0.0)
  tol = margin + ccd_tolerance
  dist, _x1, _x2, _normal, _res = _eval_geom_pair_dist(
    obj1, obj2, type1, type2, r1, r2, tol, obj1.pos, obj2.pos, wp.vec3(0.0, 0.0, 1.0), ccd_tolerance, ccd_iterations
  )
  return dist <= tol


@cache_kernel
def _is_ray_occluded(has_mesh: bool, has_plane: bool):
  @wp.func
  def func(
    # Model:
    ngeom: int,
    nmeshface: int,
    body_weldid: wp.array[int],
    geom_type: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rbound: wp.array2d[float],
    geom_margin: wp.array2d[float],
    mesh_vertadr: wp.array[int],
    mesh_vertnum: wp.array[int],
    mesh_faceadr: wp.array[int],
    mesh_graphadr: wp.array[int],
    mesh_vert: wp.array[wp.vec3],
    mesh_face: wp.array[wp.vec3i],
    mesh_graph: wp.array[int],
    pair_margin: wp.array2d[float],
    nxn_pairid: wp.array[wp.vec2i],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xquat_in: wp.array2d[wp.quat],
    # In:
    g1: int,
    g2: int,
    a: wp.vec3,
    b: wp.vec3,
    ccd_tolerance: float,
    ccd_iterations: int,
    worldid: int,
  ) -> bool:
    """Return True if segment [a, b] swept by geom g1 (paired with g2) is occluded."""
    ab = b - a
    len_ab = wp.norm_l2(ab)
    if len_ab < MJ_MINVAL:
      return False

    # Shrink segment slightly to avoid self-hits on geoms touching endpoints from outside
    eps = 1.0e-5
    seg_dir = ab / len_ab
    pnt = a + eps * ab
    max_ray_dist = (1.0 - 2.0 * eps) * len_ab

    mid = a + 0.5 * ab
    half_len = 0.5 * len_ab

    weld1 = body_weldid[geom_bodyid[g1]]
    weld2 = body_weldid[geom_bodyid[g2]]

    rbound_id = worldid % geom_rbound.shape[0]
    size_id = worldid % geom_size.shape[0]
    dataid_id = worldid % geom_dataid.shape[0]

    for g3 in range(ngeom):
      g3_type = geom_type[g3]
      if g3 == g1 or g3 == g2 or not _geom_supported(g3_type):
        continue

      # g3 must not share a weld with g1 (or a moving weld with g2)
      weld3 = body_weldid[geom_bodyid[g3]]
      if weld3 == weld1 or (weld3 == weld2 and weld2 != 0):
        continue

      # Filter out non-colliding pairs via L1-cached Model lookup before per-world pose loads
      if g1 < g3:
        pid = nxn_pairid[upper_tri_index(ngeom, g1, g3)][0]
      else:
        pid = nxn_pairid[upper_tri_index(ngeom, g3, g1)][0]
      if pid < -1:
        continue

      dataid3 = int(-1)
      if wp.static(has_mesh):
        if g3_type == GeomType.MESH:
          dataid3 = geom_dataid[dataid_id, g3]
          if dataid3 < 0:
            continue

      rbound3 = geom_rbound[rbound_id, g3]
      gpos = geom_xpos_in[worldid, g3]
      if rbound3 > 0.0:
        diff = gpos - mid
        max_r = rbound3 + half_len
        if wp.dot(diff, diff) > max_r * max_r:
          continue

      size3 = geom_size[size_id, g3]
      hit = bool(False)
      if g3_type == GeomType.SPHERE:
        ray_dist, _ray_n = ray_sphere(gpos, size3[0] * size3[0], pnt, seg_dir)
        hit = (ray_dist >= 0.0) and (ray_dist <= max_ray_dist)
      elif wp.static(has_plane) and g3_type == GeomType.PLANE:
        n = rot_vec_quat(wp.vec3(0.0, 0.0, 1.0), geom_xquat_in[worldid, g3])
        da = wp.dot(a - gpos, n)
        db = da + wp.dot(ab, n)
        hit = (da * db <= 0.0) and (wp.abs(da) > eps or wp.abs(db) > eps)
      else:
        gmat = quat_to_mat(geom_xquat_in[worldid, g3])
        if wp.static(has_mesh) and g3_type == GeomType.MESH:
          ray_dist, _ray_n = ray_mesh(
            nmeshface,
            mesh_vertadr,
            mesh_faceadr,
            mesh_vert,
            mesh_face,
            dataid3,
            gpos,
            gmat,
            size3,
            pnt,
            seg_dir,
          )
        else:
          ray_dist, _ray_n = ray_geom(gpos, gmat, size3, pnt, seg_dir, g3_type)
        hit = (ray_dist >= 0.0) and (ray_dist <= max_ray_dist)

      if hit:
        margin3 = contact_margin(geom_margin, pair_margin, wp.vec2i(g1, g3), pid, worldid)
        if not _is_geom_pair_within_margin(
          geom_type,
          geom_dataid,
          geom_size,
          mesh_vertadr,
          mesh_vertnum,
          mesh_graphadr,
          mesh_vert,
          mesh_graph,
          geom_xpos_in,
          geom_xquat_in,
          g1,
          g3,
          margin3,
          ccd_tolerance,
          ccd_iterations,
          worldid,
        ):
          return True

    return False

  return func


@wp.kernel
def _geom_step_motion_kernel(
  # Model:
  opt_timestep: wp.array[float],
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  body_geomnum: wp.array[int],
  body_geomadr: wp.array[int],
  geom_type: wp.array[int],
  geom_rbound: wp.array2d[float],
  # Data in:
  qvel_in: wp.array2d[float],
  geom_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # Out:
  geom_dx_out: wp.array2d[wp.vec3],
  geom_dw_out: wp.array2d[wp.vec3],
  geom_sweep_out: wp.array2d[float],
):
  """Compute step displacement (dx, dw) and surface sweep bound per body across its geoms."""
  worldid, bodyid = wp.tid()

  geom_num = body_geomnum[bodyid]
  if geom_num == 0:
    return
  geom_start = body_geomadr[bodyid]

  weldid = body_weldid[bodyid]
  if weldid == 0:
    for i in range(geom_num):
      geomid = geom_start + i
      geom_dx_out[worldid, geomid] = wp.vec3(0.0)
      geom_dw_out[worldid, geomid] = wp.vec3(0.0)
      geom_sweep_out[worldid, geomid] = 0.0
    return

  # d.cvel is not yet computed during fwd_position/collision, so accumulate cdof * qvel directly
  cvel_ang = wp.vec3(0.0)
  cvel_lin = wp.vec3(0.0)
  b = weldid
  while b > 0:
    bda = body_dofadr[b]
    dofnum = body_dofnum[b]
    for j in range(dofnum):
      qv = qvel_in[worldid, bda + j]
      if qv != 0.0:
        cd = cdof_in[worldid, bda + j]
        cvel_ang += wp.spatial_top(cd) * qv
        cvel_lin += wp.spatial_bottom(cd) * qv
    b = body_parentid[b]

  subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]
  dt = opt_timestep[worldid % opt_timestep.shape[0]]
  rbound_id = worldid % geom_rbound.shape[0]
  dw_body = cvel_ang * dt
  dw_norm = wp.norm_l2(dw_body)

  for i in range(geom_num):
    geomid = geom_start + i
    gtype = geom_type[geomid]
    if gtype == GeomType.PLANE or not _geom_supported(gtype):
      geom_dx_out[worldid, geomid] = wp.vec3(0.0)
      geom_dw_out[worldid, geomid] = wp.vec3(0.0)
      geom_sweep_out[worldid, geomid] = 0.0
      continue

    # Transform spatial velocity from subtree_com to geom_xpos
    offset = geom_xpos_in[worldid, geomid] - subtree_com
    dx = (cvel_lin + wp.cross(cvel_ang, offset)) * dt
    if gtype == GeomType.SPHERE:
      dw = wp.vec3(0.0)
      sweep = wp.norm_l2(dx)
    else:
      dw = dw_body
      sweep = wp.norm_l2(dx) + dw_norm * geom_rbound[rbound_id, geomid]

    geom_dx_out[worldid, geomid] = dx
    geom_dw_out[worldid, geomid] = dw
    geom_sweep_out[worldid, geomid] = sweep


@wp.kernel
def _record_nacon_start(
  # Data in:
  naconmax_in: int,
  nacon_in: wp.array[int],
  # Out:
  spec_state_out: wp.array[int],
):
  """Snapshot starting contact count before an incremental discrete narrowphase pass."""
  spec_state_out[_SPEC_NACON_START] = wp.min(nacon_in[0], naconmax_in)


@wp.kernel
def _init_speculative_pass(
  # Data in:
  naconmax_in: int,
  nacon_in: wp.array[int],
  # In:
  reset_start_in: int,
  # Out:
  spec_state_out: wp.array[int],
):
  """Reset per-pass counters and snapshot discrete contact count."""
  spec_state_out[_SPEC_NFAST] = 0
  spec_state_out[_SPEC_NCCD] = 0
  spec_state_out[_SPEC_ANY_HIT] = 0
  spec_state_out[_SPEC_NACON_DISCRETE] = wp.min(nacon_in[0], naconmax_in)
  if reset_start_in != 0:
    spec_state_out[_SPEC_NACON_START] = 0


@wp.kernel
def _collect_fast_pairs(
  # Model:
  geom_type: wp.array[int],
  geom_aabb: wp.array3d[wp.vec3],
  # Data in:
  naconmax_in: int,
  ncollision_in: wp.array[int],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_pairid_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
  geom_sweep_in: wp.array2d[float],
  # Out:
  spec_pair_table_out: wp.array[int],
  spec_fast_cid_out: wp.array[int],
  spec_pair_state_out: wp.array[int],
  spec_state_out: wp.array[int],
):
  """Compact broadphase pairs whose step sweep reaches min half-thickness and index them."""
  cid = wp.tid()
  if cid >= wp.min(ncollision_in[0], naconmax_in):
    return

  if collision_pairid_in[cid][0] < -1:
    return

  geoms = collision_pair_in[cid]
  g1 = geoms[0]
  g2 = geoms[1]
  type1 = geom_type[g1]
  type2 = geom_type[g2]
  if not _geom_supported(type1) or not _geom_supported(type2):
    return
  if type1 == GeomType.PLANE and type2 == GeomType.PLANE:
    return

  worldid = collision_worldid_in[cid]
  s1 = geom_sweep_in[worldid, g1]
  s2 = geom_sweep_in[worldid, g2]
  if s1 + s2 < MJ_MINVAL:
    return

  aabb_id = worldid % geom_aabb.shape[0]
  min_size = pair_min_size(type1, type2, geom_aabb[aabb_id, g1, 1], geom_aabb[aabb_id, g2, 1])
  if s1 < min_size and s2 < min_size:
    return

  fidx = wp.atomic_add(spec_state_out, _SPEC_NFAST, 1)  # kernel_analyzer: ignore[atomic]
  if fidx < naconmax_in:
    spec_fast_cid_out[fidx] = cid
  spec_pair_state_out[cid] = _PAIR_PENDING

  table_size = spec_pair_table_out.shape[0]
  mask = table_size - 1
  slot = _pair_hash(worldid, g1, g2, mask)
  for _ in range(table_size):
    if wp.atomic_cas(spec_pair_table_out, slot, -1, cid) == -1:  # kernel_analyzer: ignore[atomic]
      break
    slot = (slot + 1) & mask


@wp.kernel
def _mark_discrete_active(
  # Data in:
  contact_dist_in: wp.array[float],
  contact_includemargin_in: wp.array[float],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
  spec_pair_table_in: wp.array[int],
  spec_state_in: wp.array[int],
  # Out:
  spec_pair_state_out: wp.array[int],
):
  """Mark fast broadphase pairs that already have an active discrete contact (dist < margin)."""
  if spec_state_in[_SPEC_NFAST] == 0:
    return
  conid = spec_state_in[_SPEC_NACON_START] + wp.tid()
  if conid >= spec_state_in[_SPEC_NACON_DISCRETE]:
    return
  if (contact_type_in[conid] & ContactType.CONSTRAINT) and (contact_dist_in[conid] < contact_includemargin_in[conid]):
    cid = _lookup_collisionid(
      contact_worldid_in[conid],
      contact_geom_in[conid],
      collision_pair_in,
      collision_worldid_in,
      spec_pair_table_in,
    )
    if cid >= 0:
      spec_pair_state_out[cid] = _PAIR_DISCRETE_ACTIVE


@cache_kernel
def _speculative_narrowphase(
  use_multiccd: bool,
  has_plane: bool,
  has_mesh: bool,
  is_discrete_integrator: bool,
  disable_refsafe: bool,
  warn_overflow: int,
):
  is_ray_occluded = _is_ray_occluded(has_mesh, has_plane)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    ngeom: int,
    nmeshface: int,
    opt_timestep: wp.array[float],
    opt_ccd_tolerance: wp.array[float],
    opt_ccd_iterations: int,
    body_weldid: wp.array[int],
    geom_type: wp.array[int],
    geom_condim: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_priority: wp.array[int],
    geom_solmix: wp.array2d[float],
    geom_solref: wp.array2d[wp.vec2],
    geom_solimp: wp.array2d[vec5],
    geom_size: wp.array2d[wp.vec3],
    geom_rbound: wp.array2d[float],
    geom_friction: wp.array2d[wp.vec3],
    geom_margin: wp.array2d[float],
    geom_adhesion: wp.array2d[float],
    mesh_vertadr: wp.array[int],
    mesh_vertnum: wp.array[int],
    mesh_faceadr: wp.array[int],
    mesh_graphadr: wp.array[int],
    mesh_vert: wp.array[wp.vec3],
    mesh_face: wp.array[wp.vec3i],
    mesh_graph: wp.array[int],
    mesh_polynum: wp.array[int],
    mesh_polyadr: wp.array[int],
    mesh_polynormal: wp.array[wp.vec3],
    mesh_polyvertadr: wp.array[int],
    mesh_polyvertnum: wp.array[int],
    mesh_polyvert: wp.array[int],
    mesh_polymapadr: wp.array[int],
    mesh_polymapnum: wp.array[int],
    mesh_polymap: wp.array[int],
    pair_dim: wp.array[int],
    pair_solref: wp.array2d[wp.vec2],
    pair_solreffriction: wp.array2d[wp.vec2],
    pair_solimp: wp.array2d[vec5],
    pair_margin: wp.array2d[float],
    pair_adhesion: wp.array2d[float],
    pair_friction: wp.array2d[vec5],
    nxn_pairid: wp.array[wp.vec2i],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xquat_in: wp.array2d[wp.quat],
    naconmax_in: int,
    naccdmax_in: int,
    # In:
    grid_stride_in: int,
    collision_pair_in: wp.array[wp.vec2i],
    collision_pairid_in: wp.array[wp.vec2i],
    collision_worldid_in: wp.array[int],
    spec_fast_cid_in: wp.array[int],
    spec_pair_state_in: wp.array[int],
    geom_dx_in: wp.array2d[wp.vec3],
    geom_dw_in: wp.array2d[wp.vec3],
    geom_sweep_in: wp.array2d[float],
    multiccd_polygon_in: wp.array2d[wp.vec3],
    multiccd_clipped_in: wp.array2d[wp.vec3],
    multiccd_pnormal_in: wp.array2d[wp.vec3],
    multiccd_pdist_in: wp.array2d[float],
    multiccd_idx1_in: wp.array2d[int],
    multiccd_idx2_in: wp.array2d[int],
    multiccd_n1_in: wp.array2d[wp.vec3],
    multiccd_n2_in: wp.array2d[wp.vec3],
    multiccd_edge_in: wp.array2d[wp.vec3],
    multiccd_endvert_in: wp.array2d[wp.vec3],
    multiccd_face1_in: wp.array2d[wp.vec3],
    multiccd_face2_in: wp.array2d[wp.vec3],
    spec_state_in: wp.array[int],
    # Data out:
    contact_dist_out: wp.array[float],
    contact_pos_out: wp.array[wp.vec3],
    contact_frame_out: wp.array[wp.mat33],
    contact_includemargin_out: wp.array[float],
    contact_friction_out: wp.array[vec5],
    contact_solref_out: wp.array[wp.vec2],
    contact_solreffriction_out: wp.array[wp.vec2],
    contact_solimp_out: wp.array[vec5],
    contact_dim_out: wp.array[int],
    contact_geom_out: wp.array[wp.vec2i],
    contact_efc_address_out: wp.array2d[int],
    contact_worldid_out: wp.array[int],
    contact_type_out: wp.array[int],
    contact_geomcollisionid_out: wp.array[int],
    contact_adhesion_out: wp.array[float],
    nacon_out: wp.array[int],
    overflow_out: wp.array[int],
    # Out:
    spec_pair_state_out: wp.array[int],
    spec_state_out: wp.array[int],
  ):
    """Grid-stride over compacted fast pairs to evaluate speculative CCD and emit contacts."""
    nfast = wp.min(spec_state_in[_SPEC_NFAST], naconmax_in)
    if nfast == 0:
      return

    for fidx in range(wp.tid(), nfast, grid_stride_in):
      cid = spec_fast_cid_in[fidx]
      if spec_pair_state_in[cid] != _PAIR_PENDING:
        continue

      geoms = collision_pair_in[cid]
      pairid = collision_pairid_in[cid][0]
      worldid = collision_worldid_in[cid]
      g1 = geoms[0]
      g2 = geoms[1]
      type1 = geom_type[g1]
      type2 = geom_type[g2]

      s1 = geom_sweep_in[worldid, g1]
      s2 = geom_sweep_in[worldid, g2]
      sweep = s1 + s2
      dx1 = wp.vec3(0.0)
      dw1 = wp.vec3(0.0)
      dx2 = wp.vec3(0.0)
      dw2 = wp.vec3(0.0)
      rot_sweep = float(0.0)
      if s1 > 0.0:
        dx1 = geom_dx_in[worldid, g1]
        dw1 = geom_dw_in[worldid, g1]
        rot_sweep += s1 - wp.norm_l2(dx1)
      if s2 > 0.0:
        dx2 = geom_dx_in[worldid, g2]
        dw2 = geom_dw_in[worldid, g2]
        rot_sweep += s2 - wp.norm_l2(dx2)
      dx = dx1 - dx2

      margin = contact_margin(geom_margin, pair_margin, geoms, pairid, worldid)

      obj1 = make_gjk_geom(
        geom_dataid,
        geom_size,
        mesh_vertadr,
        mesh_vertnum,
        mesh_graphadr,
        mesh_vert,
        mesh_graph,
        geom_xpos_in,
        geom_xquat_in,
        type1,
        g1,
        worldid,
      )
      obj2 = make_gjk_geom(
        geom_dataid,
        geom_size,
        mesh_vertadr,
        mesh_vertnum,
        mesh_graphadr,
        mesh_vert,
        mesh_graph,
        geom_xpos_in,
        geom_xquat_in,
        type2,
        g2,
        worldid,
      )
      pos1_0 = obj1.pos
      rot1_0 = obj1.rot
      pos2_0 = obj2.pos
      rot2_0 = obj2.rot

      # Shift into pair-local coordinates for float32 precision at large world offsets
      obj1, obj2, origin = _shift_pair_to_local(obj1, obj2, type1, type2)

      r1 = wp.where(type1 == GeomType.SPHERE or type1 == GeomType.CAPSULE, obj1.size[0], 0.0)
      r2 = wp.where(type2 == GeomType.SPHERE or type2 == GeomType.CAPSULE, obj2.size[0], 0.0)
      distmax = sweep + margin
      ccd_tol = opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]]

      t = float(0.0)
      dist0 = float(0.0)
      dist = float(0.0)
      x1 = obj1.pos
      x2 = obj2.pos
      normal = wp.vec3(0.0, 0.0, 1.0)
      res = GJKResult()
      target = float(0.0)
      rel_tol = float(0.0)

      # Directed conservative advancement (iter 0 evaluates initial clearance at t = 0)
      ca_hit = bool(True)
      ca_iter = int(0)
      while ca_iter <= _MAX_CA_ITERATIONS:
        dist, x1, x2, normal, res = _eval_geom_pair_dist(
          obj1, obj2, type1, type2, r1, r2, distmax, x1, x2, normal, ccd_tol, opt_ccd_iterations
        )
        obj1.index = res.index1
        obj2.index = res.index2

        if ca_iter == 0:
          if dist <= margin or dist - margin > sweep:
            ca_hit = False
            break
          dist0 = dist
          target = 0.02 * (dist0 - margin) + margin
          rel_tol = wp.max(ccd_tol, 0.5 * (target - margin))
        elif dist - margin > (1.0 - t) * sweep:
          ca_hit = False
          break

        clearance = dist - target
        if clearance <= rel_tol or ca_iter == _MAX_CA_ITERATIONS:
          break
        ca_iter += 1

        dir_sweep = wp.dot(normal, dx) + rot_sweep
        if dir_sweep <= MJ_MINVAL:
          ca_hit = False
          break

        step = clearance / dir_sweep
        t += step
        if t >= 1.0:
          ca_hit = False
          break

        # Transport un-inflated core witness points into advanced geom frames
        x1 = x1 - r1 * normal
        if s1 > 0.0:
          obj1, x1 = _advance_geom_and_witness(obj1, x1, dx1, dw1, step)
        x2 = x2 + r2 * normal
        if s2 > 0.0:
          obj2, x2 = _advance_geom_and_witness(obj2, x2, dx2, dw2, step)

      if not ca_hit:
        continue

      # Full-step displacements of witness points: dp = dx + dw x (x - pos)
      dp1 = dx1 + wp.cross(dw1, x1 - obj1.pos) if s1 > 0.0 else wp.vec3(0.0)
      dp2 = dx2 + wp.cross(dw2, x2 - obj2.pos) if s2 > 0.0 else wp.vec3(0.0)
      vn = wp.dot(normal, dp1 - dp2)
      closing_dist = wp.max(vn, safe_div(dist0 - dist, t))

      # Linear extrapolation fallback if the iteration cap was reached
      if dist - target > rel_tol:
        if vn <= MJ_MINVAL or dist - margin > (1.0 - t) * vn:
          continue
        t += (dist - target) / vn

      if closing_dist <= 0.0:
        continue

      # Plane geoms always use multicontact; other pairs follow MULTICCD flag
      nx = int(1)
      x1_all = mat43()
      x2_all = mat43()
      x1_all[0] = x1
      x2_all[0] = x2

      if wp.static(use_multiccd or has_plane):
        if supports_multiccd(type1, type2):
          do_multiccd = True if wp.static(use_multiccd) else ((type1 == GeomType.PLANE) or (type2 == GeomType.PLANE))
          if do_multiccd and multiccd_polygon_in.shape[1] > 0 and multiccd_n1_in.shape[1] > 0:
            nccd_slot = wp.atomic_add(spec_state_out, _SPEC_NCCD, 1)  # kernel_analyzer: ignore[atomic]
            if nccd_slot < naccdmax_in:
              if wp.static(has_mesh):
                if type1 == GeomType.MESH:
                  obj1 = set_mesh_poly(
                    geom_dataid,
                    mesh_polynum,
                    mesh_polyadr,
                    mesh_polynormal,
                    mesh_polyvertadr,
                    mesh_polyvertnum,
                    mesh_polyvert,
                    mesh_polymapadr,
                    mesh_polymapnum,
                    mesh_polymap,
                    obj1,
                    g1,
                    worldid,
                  )
                if type2 == GeomType.MESH:
                  obj2 = set_mesh_poly(
                    geom_dataid,
                    mesh_polynum,
                    mesh_polyadr,
                    mesh_polynormal,
                    mesh_polyvertadr,
                    mesh_polyvertnum,
                    mesh_polyvert,
                    mesh_polymapadr,
                    mesh_polymapnum,
                    mesh_polymap,
                    obj2,
                    g2,
                    worldid,
                  )
              pad = 2.0 * wp.max(0.0, dist) + wp.max(0.0, 1.0 - t) * closing_dist + ccd_tol
              obj1.margin = pad
              obj2.margin = pad
              x1_pad = x1 + 0.5 * pad * normal
              x2_pad = x2 - 0.5 * pad * normal
              nface1, feat1 = gjk_simplex_features(res, 0)
              nface2, feat2 = gjk_simplex_features(res, 1)
              nx, x1_all, x2_all, _dists = multicontact_from_features(
                multiccd_polygon_in[nccd_slot],
                multiccd_clipped_in[nccd_slot],
                multiccd_pnormal_in[nccd_slot],
                multiccd_pdist_in[nccd_slot],
                multiccd_idx1_in[nccd_slot],
                multiccd_idx2_in[nccd_slot],
                multiccd_n1_in[nccd_slot],
                multiccd_n2_in[nccd_slot],
                multiccd_edge_in[nccd_slot],
                multiccd_endvert_in[nccd_slot],
                multiccd_face1_in[nccd_slot],
                multiccd_face2_in[nccd_slot],
                nface1,
                feat1,
                nface2,
                feat2,
                x1_pad,
                x2_pad,
                dist - pad,
                obj1,
                obj2,
                type1,
                type2,
              )
              for i in range(nx):
                x1_all[i] = x1_all[i] - 0.5 * pad * normal
                x2_all[i] = x2_all[i] + 0.5 * pad * normal
            else:
              if wp.static(bool(warn_overflow & OverflowType.CCD)):
                wp.printf(
                  "Warning: Speculative CCD overflow (naccdmax=%d). Increase nccdmax or naccdmax.\n"
                  "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.CCD (or = 0 for all)\n",
                  naccdmax_in,
                )
              wp.atomic_or(overflow_out, worldid, OverflowType.CCD)

      condim, friction, solref, solreffriction, solimp, adhesion = contact_material_params(
        geom_condim,
        geom_priority,
        geom_solmix,
        geom_solref,
        geom_solimp,
        geom_friction,
        geom_adhesion,
        pair_dim,
        pair_solref,
        pair_solreffriction,
        pair_solimp,
        pair_adhesion,
        pair_friction,
        geoms,
        pairid,
        worldid,
      )
      dt = opt_timestep[worldid % opt_timestep.shape[0]]
      cdist = _contact_dist(
        margin,
        t,
        closing_dist,
        solref,
        solimp,
        dt,
        wp.static(is_discrete_integrator),
        wp.static(disable_refsafe),
      )

      w1 = safe_div(s1, sweep)
      w2 = safe_div(s2, sweep)
      g_fast = wp.where(s1 >= s2, g1, g2)
      g_slow = wp.where(s1 >= s2, g2, g1)
      frame = make_frame(normal)
      nwritten = int(0)
      any_wrote = bool(False)

      for i in range(nx):
        x1_loc = x1_all[i]
        x2_loc = x2_all[i]
        x1_t = x1_loc + origin
        x2_t = x2_loc + origin
        x1_0 = pos1_0 + rot1_0 @ (wp.transpose(obj1.rot) @ (x1_loc - obj1.pos)) if s1 > 0.0 else x1_t
        x2_0 = pos2_0 + rot2_0 @ (wp.transpose(obj2.rot) @ (x2_loc - obj2.pos)) if s2 > 0.0 else x2_t

        occluded = bool(False)
        seg = int(0)
        # Use while instead of for range(3) so Warp does not unroll is_ray_occluded 3x
        while seg < 3:
          cur_seg = seg
          seg += 1
          if cur_seg == 0:
            if s1 <= 0.0:
              continue
            ga = g1
            gb = g2
            pa = x1_0
            pb = x1_t
          elif cur_seg == 1:
            if s2 <= 0.0:
              continue
            ga = g2
            gb = g1
            pa = x2_0
            pb = x2_t
          else:
            ga = g_fast
            gb = g_slow
            pa = x1_t
            pb = x2_t

          if is_ray_occluded(
            ngeom,
            nmeshface,
            body_weldid,
            geom_type,
            geom_bodyid,
            geom_dataid,
            geom_size,
            geom_rbound,
            geom_margin,
            mesh_vertadr,
            mesh_vertnum,
            mesh_faceadr,
            mesh_graphadr,
            mesh_vert,
            mesh_face,
            mesh_graph,
            pair_margin,
            nxn_pairid,
            geom_xpos_in,
            geom_xquat_in,
            ga,
            gb,
            pa,
            pb,
            ccd_tol,
            opt_ccd_iterations,
            worldid,
          ):
            occluded = True
            break

        if occluded:
          continue

        wrote = write_contact_entry(
          naconmax_in,
          nwritten,
          cdist,
          w1 * x1_0 + w2 * x2_0,
          frame,
          margin,
          condim,
          friction,
          solref,
          solreffriction,
          solimp,
          adhesion,
          geoms,
          worldid,
          ContactType.CONSTRAINT | ContactType.SPECULATIVE,
          contact_dist_out,
          contact_pos_out,
          contact_frame_out,
          contact_includemargin_out,
          contact_friction_out,
          contact_solref_out,
          contact_solreffriction_out,
          contact_solimp_out,
          contact_dim_out,
          contact_geom_out,
          contact_efc_address_out,
          contact_worldid_out,
          contact_type_out,
          contact_geomcollisionid_out,
          contact_adhesion_out,
          nacon_out,
        )
        nwritten += 1
        if wrote:
          any_wrote = True

      if any_wrote:
        spec_pair_state_out[cid] = _PAIR_SPEC_HIT
        spec_state_out[_SPEC_ANY_HIT] = 1

  return kernel


@wp.kernel
def _clear_replaced_gap_contacts(
  # Data in:
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
  spec_pair_table_in: wp.array[int],
  spec_pair_state_in: wp.array[int],
  spec_state_in: wp.array[int],
  # Data out:
  contact_type_out: wp.array[int],
):
  """Clear CONSTRAINT bit on in-gap discrete contacts superseded by a speculative hit."""
  if spec_state_in[_SPEC_ANY_HIT] == 0:
    return
  conid = spec_state_in[_SPEC_NACON_START] + wp.tid()
  if conid >= spec_state_in[_SPEC_NACON_DISCRETE]:
    return
  ctype = contact_type_in[conid]
  if not (ctype & ContactType.CONSTRAINT):
    return
  cid = _lookup_collisionid(
    contact_worldid_in[conid],
    contact_geom_in[conid],
    collision_pair_in,
    collision_worldid_in,
    spec_pair_table_in,
  )
  if cid >= 0 and spec_pair_state_in[cid] == _PAIR_SPEC_HIT:
    contact_type_out[conid] = ctype & (~ContactType.CONSTRAINT)


@wp.kernel
def _clear_spec_pair_table(
  # In:
  spec_state_in: wp.array[int],
  # Out:
  spec_pair_table_out: wp.array[int],
):
  """Reset open-addressed fast pair table entries to -1 when fast pairs were inserted."""
  if spec_state_in[_SPEC_NFAST] == 0:
    return
  spec_pair_table_out[wp.tid()] = -1


@event_scope
def geom_step_motion(m: Model, d: Data, ctx: CollisionContext):
  """Precompute per-geom step displacement (dx, dw) and surface sweep bound."""
  wp.launch(
    _geom_step_motion_kernel,
    dim=(d.nworld, m.nbody),
    inputs=[
      m.opt.timestep,
      m.body_parentid,
      m.body_rootid,
      m.body_weldid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_geomnum,
      m.body_geomadr,
      m.geom_type,
      m.geom_rbound,
      d.qvel,
      d.geom_xpos,
      d.subtree_com,
      d.cdof,
    ],
    outputs=[ctx.geom_dx, ctx.geom_dw, ctx.geom_sweep],
  )


def speculative_begin_pass(d: Data, ctx: CollisionContext):
  """Snapshot starting contact count before an incremental discrete narrowphase pass."""
  wp.launch(
    _record_nacon_start,
    dim=1,
    inputs=[d.naconmax, d.nacon],
    outputs=[ctx.spec_state],
  )


@event_scope
def speculative_narrowphase(m: Model, d: Data, ctx: CollisionContext, incremental: bool = False):
  """Compact fast pairs, run speculative CCD and ray occlusion, and replace in-gap contacts."""
  wp.launch(
    _init_speculative_pass,
    dim=1,
    inputs=[d.naconmax, d.nacon, int(not incremental)],
    outputs=[ctx.spec_state],
  )
  wp.launch(
    _collect_fast_pairs,
    dim=d.naconmax,
    inputs=[
      m.geom_type,
      m.geom_aabb,
      d.naconmax,
      d.ncollision,
      ctx.collision_pair,
      ctx.collision_pairid,
      ctx.collision_worldid,
      ctx.geom_sweep,
    ],
    outputs=[
      ctx.spec_pair_table,
      ctx.spec_fast_cid,
      ctx.spec_pair_state,
      ctx.spec_state,
    ],
  )
  wp.launch(
    _mark_discrete_active,
    dim=d.naconmax,
    inputs=[
      d.contact.dist,
      d.contact.includemargin,
      d.contact.geom,
      d.contact.worldid,
      d.contact.type,
      ctx.collision_pair,
      ctx.collision_worldid,
      ctx.spec_pair_table,
      ctx.spec_state,
    ],
    outputs=[ctx.spec_pair_state],
  )

  use_multiccd = (m.opt.disableflags & DisableBit.MULTICCD) == 0
  has_plane = bool(m.has_plane_geom)
  has_mesh = bool(m.nmesh > 0)
  is_discrete_integrator = m.opt.integrator == IntegratorType.DISCRETE
  disable_refsafe = bool(m.opt.disableflags & DisableBit.REFSAFE)
  spec_k = _speculative_narrowphase(
    use_multiccd,
    has_plane,
    has_mesh,
    is_discrete_integrator,
    disable_refsafe,
    int(m.opt.warn_overflow),
  )
  spec_grid = ccd_grid_size(spec_k, d.naconmax, d.ncollision.device)
  wp.launch(
    spec_k,
    dim=spec_grid,
    inputs=[
      m.ngeom,
      m.nmeshface,
      m.opt.timestep,
      m.opt.ccd_tolerance,
      m.opt.ccd_iterations,
      m.body_weldid,
      m.geom_type,
      m.geom_condim,
      m.geom_bodyid,
      m.geom_dataid,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_size,
      m.geom_rbound,
      m.geom_friction,
      m.geom_margin,
      m.geom_adhesion,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_faceadr,
      m.mesh_graphadr,
      m.mesh_vert,
      m.mesh_face,
      m.mesh_graph,
      m.mesh_polynum,
      m.mesh_polyadr,
      m.mesh_polynormal,
      m.mesh_polyvertadr,
      m.mesh_polyvertnum,
      m.mesh_polyvert,
      m.mesh_polymapadr,
      m.mesh_polymapnum,
      m.mesh_polymap,
      m.pair_dim,
      m.pair_solref,
      m.pair_solreffriction,
      m.pair_solimp,
      m.pair_margin,
      m.pair_adhesion,
      m.pair_friction,
      m.nxn_pairid,
      d.geom_xpos,
      d.geom_xquat,
      d.naconmax,
      d.naccdmax,
      spec_grid,
      ctx.collision_pair,
      ctx.collision_pairid,
      ctx.collision_worldid,
      ctx.spec_fast_cid,
      ctx.spec_pair_state,
      ctx.geom_dx,
      ctx.geom_dw,
      ctx.geom_sweep,
      ctx.multiccd_polygon,
      ctx.multiccd_clipped,
      ctx.multiccd_pnormal,
      ctx.multiccd_pdist,
      ctx.multiccd_idx1,
      ctx.multiccd_idx2,
      ctx.multiccd_n1,
      ctx.multiccd_n2,
      ctx.multiccd_edge,
      ctx.multiccd_endvert,
      ctx.multiccd_face1,
      ctx.multiccd_face2,
      ctx.spec_state,
    ],
    outputs=[
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.includemargin,
      d.contact.friction,
      d.contact.solref,
      d.contact.solreffriction,
      d.contact.solimp,
      d.contact.dim,
      d.contact.geom,
      d.contact.efc_address,
      d.contact.worldid,
      d.contact.type,
      d.contact.geomcollisionid,
      d.contact.adhesion,
      d.nacon,
      d.overflow,
      ctx.spec_pair_state,
      ctx.spec_state,
    ],
  )

  wp.launch(
    _clear_replaced_gap_contacts,
    dim=d.naconmax,
    inputs=[
      d.contact.geom,
      d.contact.worldid,
      d.contact.type,
      ctx.collision_pair,
      ctx.collision_worldid,
      ctx.spec_pair_table,
      ctx.spec_pair_state,
      ctx.spec_state,
    ],
    outputs=[d.contact.type],
  )
  wp.launch(
    _clear_spec_pair_table,
    dim=ctx.spec_pair_table.shape[0],
    inputs=[ctx.spec_state],
    outputs=[ctx.spec_pair_table],
  )
