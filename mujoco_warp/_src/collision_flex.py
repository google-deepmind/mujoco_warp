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
"""Flex collision detection (geom vs flex triangles)."""

import warp as wp

from mujoco_warp._src import collision_primitive_core
from mujoco_warp._src.math import make_frame
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import MJ_MINMU
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def _flex_element_aabb_filter(
  # In:
  box1_min: wp.vec3,
  box1_max: wp.vec3,
  box2_min: wp.vec3,
  box2_max: wp.vec3,
) -> bool:
  """Return True if the two AABBs do NOT intersect (discard pair)."""
  if box1_max[0] < box2_min[0] or box1_min[0] > box2_max[0]:
    return True
  if box1_max[1] < box2_min[1] or box1_min[1] > box2_max[1]:
    return True
  if box1_max[2] < box2_min[2] or box1_min[2] > box2_max[2]:
    return True
  return False


@wp.kernel
def _flex_broadphase_bounds(
  # Model:
  flex_margin: wp.array[float],
  flex_gap: wp.array[float],
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_radius: wp.array[float],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # Data out:
  flex_aabb_min_out: wp.array2d[wp.vec3],
  flex_aabb_max_out: wp.array2d[wp.vec3],
):
  worldid, flexid = wp.tid()

  start = flex_vertadr[flexid]
  num = flex_vertnum[flexid]
  if num == 0:
    return

  min_bound = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
  max_bound = wp.vec3(-MJ_MAXVAL, -MJ_MAXVAL, -MJ_MAXVAL)

  for i in range(num):
    pos = flexvert_xpos_in[worldid, start + i]
    min_bound = wp.min(min_bound, pos)
    max_bound = wp.max(max_bound, pos)

  margin = flex_margin[flexid] + flex_gap[flexid]
  bound = flex_radius[flexid] + margin
  inflate = wp.vec3(bound, bound, bound)

  flex_aabb_min_out[worldid, flexid] = min_bound - inflate
  flex_aabb_max_out[worldid, flexid] = max_bound + inflate


@wp.func
def _flex_triangle_geom_broadphase(
  # Model:
  ngeom: int,
  geom_type: wp.array[int],
  geom_aabb: wp.array3d[wp.vec3],
  geom_margin: wp.array2d[float],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  naconmax_in: int,
  # In:
  worldid: int,
  element_or_shell_id: int,
  flexid: int,
  geomid: int,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  tri_margin: float,
  flex_aabb_min_val: wp.vec3,
  flex_aabb_max_val: wp.vec3,
  # Data out:
  ncollision_out: wp.array[int],
  # Out:
  collision_pair_out: wp.array[wp.vec2i],
  collision_worldid_out: wp.array[int],
):
  gtype = geom_type[geomid]
  if (
    gtype != int(GeomType.SPHERE)
    and gtype != int(GeomType.CAPSULE)
    and gtype != int(GeomType.BOX)
    and gtype != int(GeomType.CYLINDER)
  ):
    return

  geom_margin_val = geom_margin[worldid % geom_margin.shape[0], geomid]
  margin = geom_margin_val + tri_margin

  aabb_id = worldid % geom_aabb.shape[0]
  geom_center_local = geom_aabb[aabb_id, geomid, 0]
  geom_half_size_local = geom_aabb[aabb_id, geomid, 1]

  geom_pos = geom_xpos_in[worldid, geomid]
  geom_rot = geom_xmat_in[worldid, geomid]

  # Stage 1 Filter: Coarse flex AABB vs Geom world AABB check
  # Transform center to global frame
  geom_center_global = geom_rot @ geom_center_local + geom_pos

  # Project local half-size onto world axes using absolute rotation matrix entries
  geom_half_size_global = wp.vec3(
    wp.abs(geom_rot[0, 0]) * geom_half_size_local[0]
    + wp.abs(geom_rot[0, 1]) * geom_half_size_local[1]
    + wp.abs(geom_rot[0, 2]) * geom_half_size_local[2],
    wp.abs(geom_rot[1, 0]) * geom_half_size_local[0]
    + wp.abs(geom_rot[1, 1]) * geom_half_size_local[1]
    + wp.abs(geom_rot[1, 2]) * geom_half_size_local[2],
    wp.abs(geom_rot[2, 0]) * geom_half_size_local[0]
    + wp.abs(geom_rot[2, 1]) * geom_half_size_local[1]
    + wp.abs(geom_rot[2, 2]) * geom_half_size_local[2],
  )

  inflate = wp.vec3(margin, margin, margin)
  geom_box_min = geom_center_global - geom_half_size_global - inflate
  geom_box_max = geom_center_global + geom_half_size_global + inflate

  if _flex_element_aabb_filter(geom_box_min, geom_box_max, flex_aabb_min_val, flex_aabb_max_val):
    return

  # Stage 2 Filter: Tight Element AABB check in Geom Local frame
  # Transform triangle vertices into the local frame of the geom (transposing geom_rot)
  geom_rot_T = wp.transpose(geom_rot)
  t1_local = geom_rot_T @ (t1 - geom_pos)
  t2_local = geom_rot_T @ (t2 - geom_pos)
  t3_local = geom_rot_T @ (t3 - geom_pos)

  # Local Element AABB (inflated by tri_radius)
  elem_min_local = wp.min(t1_local, wp.min(t2_local, t3_local)) - wp.vec3(tri_radius, tri_radius, tri_radius)
  elem_max_local = wp.max(t1_local, wp.max(t2_local, t3_local)) + wp.vec3(tri_radius, tri_radius, tri_radius)

  # Tight Local Geom AABB (inflated by geom_margin + tri_margin)
  geom_box_min_local = geom_center_local - geom_half_size_local - inflate
  geom_box_max_local = geom_center_local + geom_half_size_local + inflate

  if _flex_element_aabb_filter(geom_box_min_local, geom_box_max_local, elem_min_local, elem_max_local):
    return

  # Stage 3 Filter: Triangle Face Normal Projection (SAT)
  u = t2 - t1
  v = t3 - t1
  normal_unnorm = wp.cross(u, v)
  len_norm = wp.length(normal_unnorm)
  if len_norm > 1e-6:
    n_tri = normal_unnorm / len_norm

    # Projection distance of geom center from triangle plane
    dist_plane = wp.dot(geom_pos - t1, n_tri)

    # Compute max extent of geom along triangle normal n_tri
    r_extent = 0.0
    if gtype == int(GeomType.SPHERE):
      r_extent = geom_half_size_local[0]
    elif gtype == int(GeomType.BOX):
      axis_x = wp.vec3(geom_rot[0, 0], geom_rot[1, 0], geom_rot[2, 0])
      axis_y = wp.vec3(geom_rot[0, 1], geom_rot[1, 1], geom_rot[2, 1])
      axis_z = wp.vec3(geom_rot[0, 2], geom_rot[1, 2], geom_rot[2, 2])
      r_extent = (
        wp.abs(wp.dot(axis_x, n_tri)) * geom_half_size_local[0]
        + wp.abs(wp.dot(axis_y, n_tri)) * geom_half_size_local[1]
        + wp.abs(wp.dot(axis_z, n_tri)) * geom_half_size_local[2]
      )
    else:  # CAPSULE or CYLINDER
      axis_z = wp.vec3(geom_rot[0, 2], geom_rot[1, 2], geom_rot[2, 2])
      r_extent = geom_half_size_local[0] + wp.abs(wp.dot(axis_z, n_tri)) * geom_half_size_local[2]

    if wp.abs(dist_plane) > r_extent + tri_radius + margin:
      return

  # Append Candidate to Context
  idx = wp.atomic_add(ncollision_out, 0, 1)
  if idx < naconmax_in:
    collision_pair_out[idx] = wp.vec2i(element_or_shell_id, geomid)
    collision_worldid_out[idx] = worldid


@wp.kernel
def _flex_broadphase_dim2(
  # Model:
  ngeom: int,
  nflex: int,
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_aabb: wp.array3d[wp.vec3],
  geom_rbound: wp.array2d[float],
  geom_margin: wp.array2d[float],
  flex_margin: wp.array[float],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_elem: wp.array[int],
  flex_radius: wp.array[float],
  flexelem_geom_pair_filtered: wp.array[wp.vec2i],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  naconmax_in: int,
  flex_aabb_min_in: wp.array2d[wp.vec3],
  flex_aabb_max_in: wp.array2d[wp.vec3],
  # Data out:
  ncollision_out: wp.array[int],
  # Out:
  collision_pair_out: wp.array[wp.vec2i],
  collision_worldid_out: wp.array[int],
):
  worldid, pairid = wp.tid()

  pair = flexelem_geom_pair_filtered[pairid]
  elemid = pair[0]
  geomid = pair[1]

  flexid = int(-1)
  for i in range(nflex):
    if flex_dim[i] != 2:
      continue
    elem_adr = flex_elemadr[i]
    elem_num = flex_elemnum[i]
    if elemid >= elem_adr and elemid < elem_adr + elem_num:
      flexid = i
      break

  if flexid < 0:
    return

  vert_adr = flex_vertadr[flexid]
  tri_radius = flex_radius[flexid]
  tri_margin = flex_margin[flexid]

  elem_data_idx = flex_elemdataadr[flexid] + (elemid - flex_elemadr[flexid]) * 3
  v0_local = flex_elem[elem_data_idx]
  v1_local = flex_elem[elem_data_idx + 1]
  v2_local = flex_elem[elem_data_idx + 2]

  t1 = flexvert_xpos_in[worldid, vert_adr + v0_local]
  t2 = flexvert_xpos_in[worldid, vert_adr + v1_local]
  t3 = flexvert_xpos_in[worldid, vert_adr + v2_local]

  _flex_triangle_geom_broadphase(
    ngeom,
    geom_type,
    geom_aabb,
    geom_margin,
    geom_xpos_in,
    geom_xmat_in,
    naconmax_in,
    worldid,
    elemid,
    flexid,
    geomid,
    t1,
    t2,
    t3,
    tri_radius,
    tri_margin,
    flex_aabb_min_in[worldid, flexid],
    flex_aabb_max_in[worldid, flexid],
    ncollision_out,
    collision_pair_out,
    collision_worldid_out,
  )


@wp.kernel
def _flex_broadphase_dim3(
  # Model:
  ngeom: int,
  nflex: int,
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_aabb: wp.array3d[wp.vec3],
  geom_rbound: wp.array2d[float],
  geom_margin: wp.array2d[float],
  flex_margin: wp.array[float],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_shellnum: wp.array[int],
  flex_shelldataadr: wp.array[int],
  flex_shell: wp.array[int],
  flex_radius: wp.array[float],
  flexshell_geom_pair_filtered: wp.array[wp.vec2i],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  naconmax_in: int,
  flex_aabb_min_in: wp.array2d[wp.vec3],
  flex_aabb_max_in: wp.array2d[wp.vec3],
  # Data out:
  ncollision_out: wp.array[int],
  # Out:
  collision_pair_out: wp.array[wp.vec2i],
  collision_worldid_out: wp.array[int],
):
  worldid, pairid = wp.tid()

  pair = flexshell_geom_pair_filtered[pairid]
  shellid = pair[0]
  geomid = pair[1]

  flexid = int(-1)
  shell_offset = int(0)
  for i in range(nflex):
    if flex_dim[i] != 3:
      continue
    shell_num = flex_shellnum[i]
    if shellid >= shell_offset and shellid < shell_offset + shell_num:
      flexid = i
      break
    shell_offset += shell_num

  if flexid < 0:
    return

  vert_adr = flex_vertadr[flexid]
  tri_radius = flex_radius[flexid]
  tri_margin = flex_margin[flexid]

  shell_adr = flex_shelldataadr[flexid]
  local_shellid = shellid - shell_offset
  shell_data_idx = shell_adr + local_shellid * 3

  v0_local = flex_shell[shell_data_idx]
  v1_local = flex_shell[shell_data_idx + 1]
  v2_local = flex_shell[shell_data_idx + 2]

  t1 = flexvert_xpos_in[worldid, vert_adr + v0_local]
  t2 = flexvert_xpos_in[worldid, vert_adr + v1_local]
  t3 = flexvert_xpos_in[worldid, vert_adr + v2_local]

  _flex_triangle_geom_broadphase(
    ngeom,
    geom_type,
    geom_aabb,
    geom_margin,
    geom_xpos_in,
    geom_xmat_in,
    naconmax_in,
    worldid,
    shellid,
    flexid,
    geomid,
    t1,
    t2,
    t3,
    tri_radius,
    tri_margin,
    flex_aabb_min_in[worldid, flexid],
    flex_aabb_max_in[worldid, flexid],
    ncollision_out,
    collision_pair_out,
    collision_worldid_out,
  )


@wp.kernel
def _flex_broadphase_plane(
  # Model:
  ngeom: int,
  geom_type: wp.array[int],
  geom_margin: wp.array2d[float],
  flex_margin: wp.array[float],
  flex_vertadr: wp.array[int],
  flex_radius: wp.array[float],
  flex_vertflexid: wp.array[int],
  flexvert_geom_pair_filtered: wp.array[wp.vec2i],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  naconmax_in: int,
  flex_aabb_min_in: wp.array2d[wp.vec3],
  flex_aabb_max_in: wp.array2d[wp.vec3],
  # Data out:
  ncollision_out: wp.array[int],
  # Out:
  collision_pair_out: wp.array[wp.vec2i],
  collision_worldid_out: wp.array[int],
):
  worldid, pairid = wp.tid()

  pair = flexvert_geom_pair_filtered[pairid]
  vertid = pair[0]
  geomid = pair[1]

  flexid = flex_vertflexid[vertid]
  radius = flex_radius[flexid]
  flex_margin_val = flex_margin[flexid]

  vert = flexvert_xpos_in[worldid, vertid]

  flex_aabb_min = flex_aabb_min_in[worldid, flexid]
  flex_aabb_max = flex_aabb_max_in[worldid, flexid]

  gtype = geom_type[geomid]
  if gtype != int(GeomType.PLANE):
    return

  margin = geom_margin[worldid % geom_margin.shape[0], geomid] + flex_margin_val
  geom_pos = geom_xpos_in[worldid, geomid]
  geom_rot = geom_xmat_in[worldid, geomid]
  plane_normal = wp.vec3(geom_rot[0, 2], geom_rot[1, 2], geom_rot[2, 2])

  # Stage 1 filter: Bounding box of flex vs plane
  flex_center = 0.5 * (flex_aabb_min + flex_aabb_max)
  flex_half_size = 0.5 * (flex_aabb_max - flex_aabb_min)

  proj_half = (
    wp.abs(flex_half_size[0] * plane_normal[0])
    + wp.abs(flex_half_size[1] * plane_normal[1])
    + wp.abs(flex_half_size[2] * plane_normal[2])
  )

  diff_center = flex_center - geom_pos
  dist_center = wp.dot(diff_center, plane_normal)
  if dist_center - proj_half > margin:
    return

  diff = vert - geom_pos
  signed_dist = wp.dot(diff, plane_normal)
  dist = signed_dist - radius

  if dist < margin:
    # Append Candidate to Context
    idx = wp.atomic_add(ncollision_out, 0, 1)
    if idx < naconmax_in:
      collision_pair_out[idx] = wp.vec2i(vertid, geomid)
      collision_worldid_out[idx] = worldid


# TODO(team): generalize into a shared contact parameter mixing function
#   (mj_contactParam) that works for both geom-geom and geom-flex contacts.
@wp.func
def _mix_flex_contact_params(
  # In:
  a_condim: int,
  a_priority: int,
  a_solmix: float,
  a_solref: wp.vec2,
  a_solimp: vec5,
  a_friction: wp.vec3,
  a_gap: float,
  b_condim: int,
  b_priority: int,
  b_solmix: float,
  b_solref: wp.vec2,
  b_solimp: vec5,
  b_friction: wp.vec3,
  b_gap: float,
):
  """Mix contact parameters between geom and flex, matching mj_contactParam."""
  gap = a_gap + b_gap

  if a_priority > b_priority:
    condim = a_condim
    solref = a_solref
    solimp = a_solimp
    fri = a_friction
  elif a_priority < b_priority:
    condim = b_condim
    solref = b_solref
    solimp = b_solimp
    fri = b_friction
  else:
    # same priority
    condim = wp.max(a_condim, b_condim)

    # compute solver mix factor
    if a_solmix >= MJ_MINVAL and b_solmix >= MJ_MINVAL:
      mix = a_solmix / (a_solmix + b_solmix)
    elif a_solmix < MJ_MINVAL and b_solmix < MJ_MINVAL:
      mix = 0.5
    elif a_solmix < MJ_MINVAL:
      mix = 0.0
    else:
      mix = 1.0

    # solref: mix if both standard, min if either direct
    if a_solref[0] > 0.0 and b_solref[0] > 0.0:
      solref = wp.vec2(
        mix * a_solref[0] + (1.0 - mix) * b_solref[0],
        mix * a_solref[1] + (1.0 - mix) * b_solref[1],
      )
    else:
      solref = wp.vec2(
        wp.min(a_solref[0], b_solref[0]),
        wp.min(a_solref[1], b_solref[1]),
      )

    # solimp: mix
    solimp = vec5(
      mix * a_solimp[0] + (1.0 - mix) * b_solimp[0],
      mix * a_solimp[1] + (1.0 - mix) * b_solimp[1],
      mix * a_solimp[2] + (1.0 - mix) * b_solimp[2],
      mix * a_solimp[3] + (1.0 - mix) * b_solimp[3],
      mix * a_solimp[4] + (1.0 - mix) * b_solimp[4],
    )

    # friction: max
    fri = wp.vec3(
      wp.max(a_friction[0], b_friction[0]),
      wp.max(a_friction[1], b_friction[1]),
      wp.max(a_friction[2], b_friction[2]),
    )

  # unpack 5D friction with MJ_MINMU floor
  friction = vec5(
    wp.max(MJ_MINMU, fri[0]),
    wp.max(MJ_MINMU, fri[0]),
    wp.max(MJ_MINMU, fri[1]),
    wp.max(MJ_MINMU, fri[2]),
    wp.max(MJ_MINMU, fri[2]),
  )

  return condim, gap, solref, solimp, friction


@wp.func
def _write_flex_contact(
  # Data in:
  naconmax_in: int,
  # In:
  dist: float,
  pos: wp.vec3,
  frame: wp.mat33,
  margin: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solimp: vec5,
  geom: int,
  flexid: int,
  vertid: int,
  worldid: int,
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
  contact_flex_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  if dist >= margin or dist >= MJ_MAXVAL:
    return

  id_ = wp.atomic_add(nacon_out, 0, 1)
  if id_ >= naconmax_in:
    return

  contact_dist_out[id_] = dist
  contact_pos_out[id_] = pos
  contact_frame_out[id_] = frame
  contact_includemargin_out[id_] = margin
  contact_friction_out[id_] = friction
  contact_solref_out[id_] = solref
  contact_solreffriction_out[id_] = wp.vec2(0.0, 0.0)
  contact_solimp_out[id_] = solimp
  contact_dim_out[id_] = condim
  contact_geom_out[id_] = wp.vec2i(geom, -1)
  contact_flex_out[id_] = wp.vec2i(-1, flexid)
  contact_vert_out[id_] = wp.vec2i(-1, vertid)
  contact_worldid_out[id_] = worldid
  contact_type_out[id_] = 1
  contact_geomcollisionid_out[id_] = 0


@wp.func
def _collide_geom_triangle(
  # Data in:
  naconmax_in: int,
  # In:
  gtype: int,
  pos: wp.vec3,
  rot: wp.mat33,
  size_val: wp.vec3,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  margin: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2,
  solimp: vec5,
  geomid: int,
  flexid: int,
  vertex_id: int,
  worldid: int,
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
  contact_flex_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  if gtype == int(GeomType.SPHERE):
    sphere_radius = size_val[0]
    dist, contact_pos, nrm = collision_primitive_core.sphere_triangle(pos, sphere_radius, t1, t2, t3, tri_radius)
    if dist < margin:
      _write_flex_contact(
        naconmax_in,
        dist,
        contact_pos,
        make_frame(nrm),
        margin,
        condim,
        friction,
        solref,
        solimp,
        geomid,
        flexid,
        vertex_id,
        worldid,
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
        contact_flex_out,
        contact_vert_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
      )
    return

  # Capsule, box, cylinder all return up to 2 contacts - compute then share writing code
  dists = wp.vec2(collision_primitive_core.MJ_MAXVAL, collision_primitive_core.MJ_MAXVAL)
  poss = collision_primitive_core.mat23f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  nrms = collision_primitive_core.mat23f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  if gtype == int(GeomType.CAPSULE):
    cap_radius = size_val[0]
    cap_half_len = size_val[1]
    cap_axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    dists, poss, nrms = collision_primitive_core.capsule_triangle(
      pos, cap_axis, cap_radius, cap_half_len, t1, t2, t3, tri_radius
    )
  elif gtype == int(GeomType.BOX):
    dists, poss, nrms = collision_primitive_core.box_triangle(pos, rot, size_val, t1, t2, t3, tri_radius)
  elif gtype == int(GeomType.CYLINDER):
    cyl_radius = size_val[0]
    cyl_half_height = size_val[1]
    cyl_axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
    dists, poss, nrms = collision_primitive_core.cylinder_triangle(
      pos, cyl_axis, cyl_radius, cyl_half_height, t1, t2, t3, tri_radius
    )

  # Write up to 2 contacts (shared code for capsule/box/cylinder)
  if dists[0] < margin:
    p1 = wp.vec3(poss[0, 0], poss[0, 1], poss[0, 2])
    n1 = wp.vec3(nrms[0, 0], nrms[0, 1], nrms[0, 2])
    _write_flex_contact(
      naconmax_in,
      dists[0],
      p1,
      make_frame(n1),
      margin,
      condim,
      friction,
      solref,
      solimp,
      geomid,
      flexid,
      vertex_id,
      worldid,
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
      contact_flex_out,
      contact_vert_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      nacon_out,
    )
  if dists[1] < margin:
    p2 = wp.vec3(poss[1, 0], poss[1, 1], poss[1, 2])
    n2 = wp.vec3(nrms[1, 0], nrms[1, 1], nrms[1, 2])
    _write_flex_contact(
      naconmax_in,
      dists[1],
      p2,
      make_frame(n2),
      margin,
      condim,
      friction,
      solref,
      solimp,
      geomid,
      flexid,
      vertex_id,
      worldid,
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
      contact_flex_out,
      contact_vert_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      nacon_out,
    )


@wp.kernel
def _flex_plane_narrowphase(
  # Model:
  ngeom: int,
  nflexvert: int,
  geom_type: wp.array[int],
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  flex_condim: wp.array[int],
  flex_priority: wp.array[int],
  flex_solmix: wp.array[float],
  flex_solref: wp.array[wp.vec2],
  flex_solimp: wp.array[vec5],
  flex_friction: wp.array[wp.vec3],
  flex_margin: wp.array[float],
  flex_gap: wp.array[float],
  flex_vertadr: wp.array[int],
  flex_radius: wp.array[float],
  flex_vertflexid: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  nworld_in: int,
  naconmax_in: int,
  ncollision_in: wp.array[int],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
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
  contact_flex_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  collisionid = wp.tid()
  if collisionid >= ncollision_in[0]:
    return

  pair = collision_pair_in[collisionid]
  geomid = pair[1]

  gtype = geom_type[geomid]
  if gtype != int(GeomType.PLANE):
    return

  vertid = pair[0]
  worldid = collision_worldid_in[collisionid]

  flexid = flex_vertflexid[vertid]
  radius = flex_radius[flexid]
  flex_margin_val = flex_margin[flexid]
  # Convert global vertid to local vertex index within this flex
  local_vertid = vertid - flex_vertadr[flexid]

  vert = flexvert_xpos_in[worldid, vertid]

  plane_pos = geom_xpos_in[worldid, geomid]
  plane_rot = geom_xmat_in[worldid, geomid]
  plane_normal = wp.vec3(plane_rot[0, 2], plane_rot[1, 2], plane_rot[2, 2])

  margin = geom_margin[worldid % geom_margin.shape[0], geomid] + flex_margin_val

  diff = vert - plane_pos
  signed_dist = wp.dot(diff, plane_normal)
  dist = signed_dist - radius

  if dist < margin:
    condim, gap, solref, solimp, friction = _mix_flex_contact_params(
      geom_condim[geomid],
      geom_priority[geomid],
      geom_solmix[worldid % geom_solmix.shape[0], geomid],
      geom_solref[worldid % geom_solref.shape[0], geomid],
      geom_solimp[worldid % geom_solimp.shape[0], geomid],
      geom_friction[worldid % geom_friction.shape[0], geomid],
      geom_gap[worldid % geom_gap.shape[0], geomid],
      flex_condim[flexid],
      flex_priority[flexid],
      flex_solmix[flexid],
      flex_solref[flexid],
      flex_solimp[flexid],
      flex_friction[flexid],
      flex_gap[flexid],
    )

    contact_pos = vert - plane_normal * (dist * 0.5 + radius)
    _write_flex_contact(
      naconmax_in,
      dist,
      contact_pos,
      make_frame(plane_normal),
      margin - gap,
      condim,
      friction,
      solref,
      solimp,
      geomid,
      flexid,
      local_vertid,
      worldid,
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
      contact_flex_out,
      contact_vert_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      nacon_out,
    )


@wp.kernel
def _flex_narrowphase_dim2(
  # Model:
  ngeom: int,
  nflex: int,
  geom_type: wp.array[int],
  geom_contype: wp.array[int],
  geom_conaffinity: wp.array[int],
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_size: wp.array2d[wp.vec3],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  flex_contype: wp.array[int],
  flex_conaffinity: wp.array[int],
  flex_condim: wp.array[int],
  flex_priority: wp.array[int],
  flex_solmix: wp.array[float],
  flex_solref: wp.array[wp.vec2],
  flex_solimp: wp.array[vec5],
  flex_friction: wp.array[wp.vec3],
  flex_margin: wp.array[float],
  flex_gap: wp.array[float],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_elem: wp.array[int],
  flex_radius: wp.array[float],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  nworld_in: int,
  naconmax_in: int,
  ncollision_in: wp.array[int],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
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
  contact_flex_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  collisionid = wp.tid()
  if collisionid >= ncollision_in[0]:
    return

  pair = collision_pair_in[collisionid]
  geomid = pair[1]

  gtype = geom_type[geomid]
  if (
    gtype != int(GeomType.SPHERE)
    and gtype != int(GeomType.CAPSULE)
    and gtype != int(GeomType.BOX)
    and gtype != int(GeomType.CYLINDER)
  ):
    return

  elemid = pair[0]
  worldid = collision_worldid_in[collisionid]

  flexid = int(-1)
  for i in range(nflex):
    if flex_dim[i] != 2:
      continue
    elem_adr = flex_elemadr[i]
    elem_num = flex_elemnum[i]
    if elemid >= elem_adr and elemid < elem_adr + elem_num:
      flexid = i
      break

  if flexid < 0:
    return

  vert_adr = flex_vertadr[flexid]
  tri_radius = flex_radius[flexid]
  tri_margin = flex_margin[flexid]

  elem_data_idx = flex_elemdataadr[flexid] + (elemid - flex_elemadr[flexid]) * 3
  v0_local = flex_elem[elem_data_idx]
  v1_local = flex_elem[elem_data_idx + 1]
  v2_local = flex_elem[elem_data_idx + 2]

  t1 = flexvert_xpos_in[worldid, vert_adr + v0_local]
  t2 = flexvert_xpos_in[worldid, vert_adr + v1_local]
  t3 = flexvert_xpos_in[worldid, vert_adr + v2_local]

  geom_margin_val = geom_margin[worldid % geom_margin.shape[0], geomid]
  margin = geom_margin_val + tri_margin

  geom_pos = geom_xpos_in[worldid, geomid]
  geom_rot = geom_xmat_in[worldid, geomid]
  geom_size_val = geom_size[worldid % geom_size.shape[0], geomid]

  condim, gap, solref, solimp, friction = _mix_flex_contact_params(
    geom_condim[geomid],
    geom_priority[geomid],
    geom_solmix[worldid % geom_solmix.shape[0], geomid],
    geom_solref[worldid % geom_solref.shape[0], geomid],
    geom_solimp[worldid % geom_solimp.shape[0], geomid],
    geom_friction[worldid % geom_friction.shape[0], geomid],
    geom_gap[worldid % geom_gap.shape[0], geomid],
    flex_condim[flexid],
    flex_priority[flexid],
    flex_solmix[flexid],
    flex_solref[flexid],
    flex_solimp[flexid],
    flex_friction[flexid],
    flex_gap[flexid],
  )

  _collide_geom_triangle(
    naconmax_in,
    gtype,
    geom_pos,
    geom_rot,
    geom_size_val,
    t1,
    t2,
    t3,
    tri_radius,
    margin,
    condim,
    friction,
    solref,
    solimp,
    geomid,
    flexid,
    v0_local,
    worldid,
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
    contact_flex_out,
    contact_vert_out,
    contact_worldid_out,
    contact_type_out,
    contact_geomcollisionid_out,
    nacon_out,
  )


@wp.kernel
def _flex_narrowphase_dim3(
  # Model:
  ngeom: int,
  nflex: int,
  geom_type: wp.array[int],
  geom_contype: wp.array[int],
  geom_conaffinity: wp.array[int],
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_size: wp.array2d[wp.vec3],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  flex_contype: wp.array[int],
  flex_conaffinity: wp.array[int],
  flex_condim: wp.array[int],
  flex_priority: wp.array[int],
  flex_solmix: wp.array[float],
  flex_solref: wp.array[wp.vec2],
  flex_solimp: wp.array[vec5],
  flex_friction: wp.array[wp.vec3],
  flex_margin: wp.array[float],
  flex_gap: wp.array[float],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_shellnum: wp.array[int],
  flex_shelldataadr: wp.array[int],
  flex_shell: wp.array[int],
  flex_radius: wp.array[float],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  nworld_in: int,
  naconmax_in: int,
  ncollision_in: wp.array[int],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
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
  contact_flex_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  collisionid = wp.tid()
  if collisionid >= ncollision_in[0]:
    return

  pair = collision_pair_in[collisionid]
  geomid = pair[1]

  gtype = geom_type[geomid]
  if (
    gtype != int(GeomType.SPHERE)
    and gtype != int(GeomType.CAPSULE)
    and gtype != int(GeomType.BOX)
    and gtype != int(GeomType.CYLINDER)
  ):
    return

  shellid = pair[0]
  worldid = collision_worldid_in[collisionid]

  flexid = int(-1)
  shell_offset = int(0)
  for i in range(nflex):
    if flex_dim[i] != 3:
      continue
    shell_num = flex_shellnum[i]
    if shellid >= shell_offset and shellid < shell_offset + shell_num:
      flexid = i
      break
    shell_offset += shell_num

  if flexid < 0:
    return

  vert_adr = flex_vertadr[flexid]
  tri_radius = flex_radius[flexid]
  tri_margin = flex_margin[flexid]

  shell_adr = flex_shelldataadr[flexid]
  local_shellid = shellid - shell_offset
  shell_data_idx = shell_adr + local_shellid * 3

  v0_local = flex_shell[shell_data_idx]
  v1_local = flex_shell[shell_data_idx + 1]
  v2_local = flex_shell[shell_data_idx + 2]

  t1 = flexvert_xpos_in[worldid, vert_adr + v0_local]
  t2 = flexvert_xpos_in[worldid, vert_adr + v1_local]
  t3 = flexvert_xpos_in[worldid, vert_adr + v2_local]

  geom_margin_val = geom_margin[worldid % geom_margin.shape[0], geomid]
  margin = geom_margin_val + tri_margin

  geom_pos = geom_xpos_in[worldid, geomid]
  geom_rot = geom_xmat_in[worldid, geomid]
  geom_size_val = geom_size[worldid % geom_size.shape[0], geomid]

  condim, gap, solref, solimp, friction = _mix_flex_contact_params(
    geom_condim[geomid],
    geom_priority[geomid],
    geom_solmix[worldid % geom_solmix.shape[0], geomid],
    geom_solref[worldid % geom_solref.shape[0], geomid],
    geom_solimp[worldid % geom_solimp.shape[0], geomid],
    geom_friction[worldid % geom_friction.shape[0], geomid],
    geom_gap[worldid % geom_gap.shape[0], geomid],
    flex_condim[flexid],
    flex_priority[flexid],
    flex_solmix[flexid],
    flex_solref[flexid],
    flex_solimp[flexid],
    flex_friction[flexid],
    flex_gap[flexid],
  )

  _collide_geom_triangle(
    naconmax_in,
    gtype,
    geom_pos,
    geom_rot,
    geom_size_val,
    t1,
    t2,
    t3,
    tri_radius,
    margin,
    condim,
    friction,
    solref,
    solimp,
    geomid,
    flexid,
    v0_local,
    worldid,
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
    contact_flex_out,
    contact_vert_out,
    contact_worldid_out,
    contact_type_out,
    contact_geomcollisionid_out,
    nacon_out,
  )


@wp.kernel
def _flex_ncollision(
  # In:
  ncollision_dim2_in: wp.array[int],
  ncollision_dim3_in: wp.array[int],
  ncollision_plane_in: wp.array[int],
  # Data out:
  ncollision_out: wp.array[int],
):
  # return maximum number of collisions over all broadphases
  ncollision_out[0] = wp.max(
    ncollision_out[0],
    wp.max(
      ncollision_dim2_in[0],
      wp.max(ncollision_dim3_in[0], ncollision_plane_in[0]),
    ),
  )


@event_scope
def flex_broadphase(m: Model, d: Data):
  """Precompute dynamic flex object bounding boxes."""
  wp.launch(
    _flex_broadphase_bounds,
    dim=(d.nworld, m.nflex),
    inputs=[
      m.flex_margin,
      m.flex_gap,
      m.flex_vertadr,
      m.flex_vertnum,
      m.flex_radius,
      d.flexvert_xpos,
    ],
    outputs=[
      d.flex_aabb_min,
      d.flex_aabb_max,
    ],
  )


@event_scope
def flex_collision(m: Model, d: Data, ctx):
  """Runs collision detection between geoms and flex elements."""
  if m.nflex == 0:
    return

  ncollision_dim2 = wp.zeros(1, dtype=int)
  ncollision_dim3 = wp.zeros(1, dtype=int)
  ncollision_plane = wp.zeros(1, dtype=int)

  # Update dynamic flex object bounding boxes
  flex_broadphase(m, d)

  # 2D Flex Element Collisions
  if m.flexelem_geom_pair_filtered.shape[0] > 0:
    wp.launch(
      _flex_broadphase_dim2,
      dim=(d.nworld, m.flexelem_geom_pair_filtered.shape[0]),
      inputs=[
        m.ngeom,
        m.nflex,
        m.geom_type,
        m.geom_size,
        m.geom_aabb,
        m.geom_rbound,
        m.geom_margin,
        m.flex_margin,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_elemadr,
        m.flex_elemnum,
        m.flex_elemdataadr,
        m.flex_elem,
        m.flex_radius,
        m.flexelem_geom_pair_filtered,
        d.geom_xpos,
        d.geom_xmat,
        d.flexvert_xpos,
        d.naconmax,
        d.flex_aabb_min,
        d.flex_aabb_max,
      ],
      outputs=[
        ncollision_dim2,
        ctx.collision_pair,
        ctx.collision_worldid,
      ],
    )
    wp.launch(
      _flex_narrowphase_dim2,
      dim=d.naconmax,
      inputs=[
        m.ngeom,
        m.nflex,
        m.geom_type,
        m.geom_contype,
        m.geom_conaffinity,
        m.geom_condim,
        m.geom_priority,
        m.geom_solmix,
        m.geom_solref,
        m.geom_solimp,
        m.geom_size,
        m.geom_friction,
        m.geom_margin,
        m.geom_gap,
        m.flex_contype,
        m.flex_conaffinity,
        m.flex_condim,
        m.flex_priority,
        m.flex_solmix,
        m.flex_solref,
        m.flex_solimp,
        m.flex_friction,
        m.flex_margin,
        m.flex_gap,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_elemadr,
        m.flex_elemnum,
        m.flex_elemdataadr,
        m.flex_elem,
        m.flex_radius,
        d.geom_xpos,
        d.geom_xmat,
        d.flexvert_xpos,
        d.nworld,
        d.naconmax,
        ncollision_dim2,
        ctx.collision_pair,
        ctx.collision_worldid,
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
        d.contact.flex,
        d.contact.vert,
        d.contact.worldid,
        d.contact.type,
        d.contact.geomcollisionid,
        d.nacon,
      ],
    )

  # 3D Flex Element Collisions
  if m.flexshell_geom_pair_filtered.shape[0] > 0:
    wp.launch(
      _flex_broadphase_dim3,
      dim=(d.nworld, m.flexshell_geom_pair_filtered.shape[0]),
      inputs=[
        m.ngeom,
        m.nflex,
        m.geom_type,
        m.geom_size,
        m.geom_aabb,
        m.geom_rbound,
        m.geom_margin,
        m.flex_margin,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_shellnum,
        m.flex_shelldataadr,
        m.flex_shell,
        m.flex_radius,
        m.flexshell_geom_pair_filtered,
        d.geom_xpos,
        d.geom_xmat,
        d.flexvert_xpos,
        d.naconmax,
        d.flex_aabb_min,
        d.flex_aabb_max,
      ],
      outputs=[
        ncollision_dim3,
        ctx.collision_pair,
        ctx.collision_worldid,
      ],
    )
    wp.launch(
      _flex_narrowphase_dim3,
      dim=d.naconmax,
      inputs=[
        m.ngeom,
        m.nflex,
        m.geom_type,
        m.geom_contype,
        m.geom_conaffinity,
        m.geom_condim,
        m.geom_priority,
        m.geom_solmix,
        m.geom_solref,
        m.geom_solimp,
        m.geom_size,
        m.geom_friction,
        m.geom_margin,
        m.geom_gap,
        m.flex_contype,
        m.flex_conaffinity,
        m.flex_condim,
        m.flex_priority,
        m.flex_solmix,
        m.flex_solref,
        m.flex_solimp,
        m.flex_friction,
        m.flex_margin,
        m.flex_gap,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_shellnum,
        m.flex_shelldataadr,
        m.flex_shell,
        m.flex_radius,
        d.geom_xpos,
        d.geom_xmat,
        d.flexvert_xpos,
        d.nworld,
        d.naconmax,
        ncollision_dim3,
        ctx.collision_pair,
        ctx.collision_worldid,
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
        d.contact.flex,
        d.contact.vert,
        d.contact.worldid,
        d.contact.type,
        d.contact.geomcollisionid,
        d.nacon,
      ],
    )

  # Plane Vertex Collisions
  if m.flexvert_geom_pair_filtered.shape[0] > 0:
    wp.launch(
      _flex_broadphase_plane,
      dim=(d.nworld, m.flexvert_geom_pair_filtered.shape[0]),
      inputs=[
        m.ngeom,
        m.geom_type,
        m.geom_margin,
        m.flex_margin,
        m.flex_vertadr,
        m.flex_radius,
        m.flex_vertflexid,
        m.flexvert_geom_pair_filtered,
        d.geom_xpos,
        d.geom_xmat,
        d.flexvert_xpos,
        d.naconmax,
        d.flex_aabb_min,
        d.flex_aabb_max,
      ],
      outputs=[
        ncollision_plane,
        ctx.collision_pair,
        ctx.collision_worldid,
      ],
    )
    wp.launch(
      _flex_plane_narrowphase,
      dim=d.naconmax,
      inputs=[
        m.ngeom,
        m.nflexvert,
        m.geom_type,
        m.geom_condim,
        m.geom_priority,
        m.geom_solmix,
        m.geom_solref,
        m.geom_solimp,
        m.geom_friction,
        m.geom_margin,
        m.geom_gap,
        m.flex_condim,
        m.flex_priority,
        m.flex_solmix,
        m.flex_solref,
        m.flex_solimp,
        m.flex_friction,
        m.flex_margin,
        m.flex_gap,
        m.flex_vertadr,
        m.flex_radius,
        m.flex_vertflexid,
        d.geom_xpos,
        d.geom_xmat,
        d.flexvert_xpos,
        d.nworld,
        d.naconmax,
        ncollision_plane,
        ctx.collision_pair,
        ctx.collision_worldid,
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
        d.contact.flex,
        d.contact.vert,
        d.contact.worldid,
        d.contact.type,
        d.contact.geomcollisionid,
        d.nacon,
      ],
    )

  wp.launch(
    _flex_ncollision,
    dim=1,
    inputs=[ncollision_dim2, ncollision_dim3, ncollision_plane],
    outputs=[d.ncollision],
  )
