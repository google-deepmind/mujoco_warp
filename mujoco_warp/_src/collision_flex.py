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

from typing import Tuple

import warp as wp

from mujoco_warp._src import collision_primitive_core
from mujoco_warp._src.collision_core import Geom
from mujoco_warp._src.collision_gjk import ccd
from mujoco_warp._src.math import make_frame
from mujoco_warp._src.types import MJ_MAX_EPAFACES
from mujoco_warp._src.types import MJ_MAX_EPAHORIZON
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import MJ_MINMU
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})


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
def _write_candidate_contact(
  # In:
  max_candidates: int,
  dist: float,
  pos: wp.vec3,
  nrm: wp.vec3,
  geom: int,
  flexid: int,
  elemid: int,
  vertid: int,
  worldid: int,
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  cand_type_out: wp.array[int],
  cand_geomcollisionid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  if dist >= MJ_MAXVAL:
    return

  candid = wp.atomic_add(ncand_out, 0, 1)
  if candid >= max_candidates:
    return

  cand_dist_out[candid] = dist
  cand_pos_out[candid] = pos
  cand_nrm_out[candid] = nrm
  if geom >= 0:
    cand_geom_out[candid] = wp.vec2i(geom, -1)
    cand_flex_out[candid] = wp.vec2i(-1, flexid)
    cand_elem_out[candid] = wp.vec2i(-1, elemid)
    cand_vert_out[candid] = wp.vec2i(-1, vertid)
  elif geom == -2:
    cand_geom_out[candid] = wp.vec2i(-1, -1)
    cand_flex_out[candid] = wp.vec2i(flexid, flexid)
    cand_elem_out[candid] = wp.vec2i(elemid, vertid)
    cand_vert_out[candid] = wp.vec2i(-1, -1)
  else:
    cand_geom_out[candid] = wp.vec2i(-1, -1)
    cand_flex_out[candid] = wp.vec2i(flexid, flexid)
    cand_elem_out[candid] = wp.vec2i(-1, elemid)
    cand_vert_out[candid] = wp.vec2i(vertid, -1)
  cand_worldid_out[candid] = worldid
  cand_type_out[candid] = 1
  cand_geomcollisionid_out[candid] = 0


@wp.func
def _collide_geom_triangle_detect(
  # In:
  max_candidates: int,
  gtype: int,
  pos: wp.vec3,
  rot: wp.mat33,
  size_val: wp.vec3,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  margin: float,
  geomid: int,
  flexid: int,
  elemid: int,
  vertex_id: int,
  worldid: int,
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  cand_type_out: wp.array[int],
  cand_geomcollisionid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  if gtype == int(GeomType.SPHERE):
    sphere_radius = size_val[0]
    dist, contact_pos, nrm = collision_primitive_core.sphere_triangle(pos, sphere_radius, t1, t2, t3, tri_radius)
    if dist < margin:
      _write_candidate_contact(
        max_candidates,
        dist,
        contact_pos,
        nrm,
        geomid,
        flexid,
        elemid,
        vertex_id,
        worldid,
        cand_dist_out,
        cand_pos_out,
        cand_nrm_out,
        cand_geom_out,
        cand_flex_out,
        cand_elem_out,
        cand_vert_out,
        cand_worldid_out,
        cand_type_out,
        cand_geomcollisionid_out,
        ncand_out,
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
    _write_candidate_contact(
      max_candidates,
      dists[0],
      p1,
      n1,
      geomid,
      flexid,
      elemid,
      vertex_id,
      worldid,
      cand_dist_out,
      cand_pos_out,
      cand_nrm_out,
      cand_geom_out,
      cand_flex_out,
      cand_elem_out,
      cand_vert_out,
      cand_worldid_out,
      cand_type_out,
      cand_geomcollisionid_out,
      ncand_out,
    )
  if dists[1] < margin:
    p2 = wp.vec3(poss[1, 0], poss[1, 1], poss[1, 2])
    n2 = wp.vec3(nrms[1, 0], nrms[1, 1], nrms[1, 2])
    _write_candidate_contact(
      max_candidates,
      dists[1],
      p2,
      n2,
      geomid,
      flexid,
      elemid,
      vertex_id,
      worldid,
      cand_dist_out,
      cand_pos_out,
      cand_nrm_out,
      cand_geom_out,
      cand_flex_out,
      cand_elem_out,
      cand_vert_out,
      cand_worldid_out,
      cand_type_out,
      cand_geomcollisionid_out,
      ncand_out,
    )


@wp.kernel
def _flex_plane_narrowphase_detect(
  # Model:
  ngeom: int,
  nflexvert: int,
  geom_type: wp.array[int],
  geom_margin: wp.array2d[float],
  flex_margin: wp.array[float],
  flex_vertadr: wp.array[int],
  flex_radius: wp.array[float],
  flex_vertflexid: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  nworld_in: int,
  # In:
  max_candidates: int,
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  cand_type_out: wp.array[int],
  cand_geomcollisionid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  worldid, vertid = wp.tid()

  flexid = flex_vertflexid[vertid]
  radius = flex_radius[flexid]
  flex_margin_val = flex_margin[flexid]
  # Convert global vertid to local vertex index within this flex
  local_vertid = vertid - flex_vertadr[flexid]

  vert = flexvert_xpos_in[worldid, vertid]

  # TODO: Add a broadphase
  for geomid in range(ngeom):
    gtype = geom_type[geomid]
    if gtype != int(GeomType.PLANE):
      continue

    plane_pos = geom_xpos_in[worldid, geomid]
    plane_rot = geom_xmat_in[worldid, geomid]
    plane_normal = wp.vec3(plane_rot[0, 2], plane_rot[1, 2], plane_rot[2, 2])

    margin = geom_margin[worldid % geom_margin.shape[0], geomid] + flex_margin_val

    diff = vert - plane_pos
    signed_dist = wp.dot(diff, plane_normal)
    dist = signed_dist - radius

    if dist < margin:
      contact_pos = vert - plane_normal * (dist * 0.5 + radius)
      _write_candidate_contact(
        max_candidates,
        dist,
        contact_pos,
        plane_normal,
        geomid,
        flexid,
        -1,
        local_vertid,
        worldid,
        cand_dist_out,
        cand_pos_out,
        cand_nrm_out,
        cand_geom_out,
        cand_flex_out,
        cand_elem_out,
        cand_vert_out,
        cand_worldid_out,
        cand_type_out,
        cand_geomcollisionid_out,
        ncand_out,
      )


@wp.func
def _sphere_tetrahedron(
  # In:
  sphere_pos: wp.vec3,
  sphere_radius: float,
  t0: wp.vec3,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
) -> Tuple[float, wp.vec3, wp.vec3]:
  d0, p0, n0 = collision_primitive_core.sphere_triangle(sphere_pos, sphere_radius, t0, t1, t2, tri_radius)
  d1, p1, n1 = collision_primitive_core.sphere_triangle(sphere_pos, sphere_radius, t0, t2, t3, tri_radius)
  d2, p2, n2 = collision_primitive_core.sphere_triangle(sphere_pos, sphere_radius, t0, t3, t1, tri_radius)
  d3, p3, n3 = collision_primitive_core.sphere_triangle(sphere_pos, sphere_radius, t1, t3, t2, tri_radius)

  min_d = d0
  min_p = p0
  min_n = n0

  if d1 < min_d:
    min_d = d1
    min_p = p1
    min_n = n1
  if d2 < min_d:
    min_d = d2
    min_p = p2
    min_n = n2
  if d3 < min_d:
    min_d = d3
    min_p = p3
    min_n = n3

  return min_d, min_p, min_n


@wp.func
def _plane_vertex(
  # In:
  pos_v: wp.vec3,
  rad: float,
  t0: wp.vec3,
  t1: wp.vec3,
  t2: wp.vec3,
) -> Tuple[bool, float, wp.vec3, wp.vec3]:
  e1 = t1 - t0
  e2 = t2 - t0
  ev = pos_v - t0

  nrm = wp.normalize(wp.cross(e1, e2))
  dst = wp.dot(ev, nrm)
  if dst <= -2.0 * rad:
    return False, 0.0, wp.vec3(0.0), wp.vec3(0.0)

  dist = -dst - 2.0 * rad
  nrm_out = -nrm
  contact_pos = pos_v - nrm * (0.5 * dst)
  return True, dist, contact_pos, nrm_out


@wp.kernel(module="unique", enable_backward=False)
def _flex_internal_collisions_detect(
  # Model:
  nflex: int,
  flex_margin: wp.array[float],
  flex_internal: wp.array[int],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_evpairadr: wp.array[int],
  flex_evpairnum: wp.array[int],
  flex_elem: wp.array[int],
  flex_evpair: wp.array[wp.vec2i],
  flex_radius: wp.array[float],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  max_candidates: int,
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  cand_type_out: wp.array[int],
  cand_geomcollisionid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  worldid, pair_idx = wp.tid()

  flexid = int(-1)
  for i in range(nflex):
    start = flex_evpairadr[i]
    num = flex_evpairnum[i]
    if pair_idx >= start and pair_idx < start + num:
      flexid = i
      break

  if flexid < 0 or flex_internal[flexid] == 0:
    return

  ev = flex_evpair[pair_idx]
  e = ev[0]
  v = ev[1]

  dim = flex_dim[flexid]
  radius = flex_radius[flexid]
  margin = flex_margin[flexid]
  vert_adr = flex_vertadr[flexid]

  sphere_pos = flexvert_xpos_in[worldid, vert_adr + v]

  elem_data_idx = flex_elemdataadr[flexid] + e * (dim + 1)
  v0_local = flex_elem[elem_data_idx]
  p0 = flexvert_xpos_in[worldid, vert_adr + v0_local]

  dist = float(MJ_MAXVAL)
  contact_pos = wp.vec3(0.0)
  nrm = wp.vec3(0.0)

  if dim == 1:
    v1_local = flex_elem[elem_data_idx + 1]
    p1 = flexvert_xpos_in[worldid, vert_adr + v1_local]
    capsule_pos = 0.5 * (p0 + p1)
    capsule_axis = wp.normalize(p1 - p0)
    capsule_half_len = 0.5 * wp.length(p1 - p0)
    dist, contact_pos, nrm = collision_primitive_core.sphere_capsule(
      sphere_pos, radius, capsule_pos, capsule_axis, radius, capsule_half_len
    )
  elif dim == 2:
    v1_local = flex_elem[elem_data_idx + 1]
    v2_local = flex_elem[elem_data_idx + 2]
    p1 = flexvert_xpos_in[worldid, vert_adr + v1_local]
    p2 = flexvert_xpos_in[worldid, vert_adr + v2_local]
    dist, contact_pos, nrm = collision_primitive_core.sphere_triangle(sphere_pos, radius, p0, p1, p2, radius)
  elif dim == 3:
    v1_local = flex_elem[elem_data_idx + 1]
    v2_local = flex_elem[elem_data_idx + 2]
    v3_local = flex_elem[elem_data_idx + 3]
    p1 = flexvert_xpos_in[worldid, vert_adr + v1_local]
    p2 = flexvert_xpos_in[worldid, vert_adr + v2_local]
    p3 = flexvert_xpos_in[worldid, vert_adr + v3_local]
    dist, contact_pos, nrm = _sphere_tetrahedron(sphere_pos, radius, p0, p1, p2, p3, radius)

  if dist < margin:
    _write_candidate_contact(
      max_candidates,
      dist,
      contact_pos,
      nrm,
      -1,
      flexid,
      e,
      v,
      worldid,
      cand_dist_out,
      cand_pos_out,
      cand_nrm_out,
      cand_geom_out,
      cand_flex_out,
      cand_elem_out,
      cand_vert_out,
      cand_worldid_out,
      cand_type_out,
      cand_geomcollisionid_out,
      ncand_out,
    )


@wp.kernel(module="unique", enable_backward=False)
def _flex_tet_internal_collisions_detect(
  # Model:
  nflex: int,
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_elem: wp.array[int],
  flex_radius: wp.array[float],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  max_candidates: int,
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  cand_type_out: wp.array[int],
  cand_geomcollisionid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  worldid, elemid = wp.tid()

  flexid = int(-1)
  for i in range(nflex):
    if flex_dim[i] != 3:
      continue
    elem_adr = flex_elemadr[i]
    elem_num = flex_elemnum[i]
    if elemid >= elem_adr and elemid < elem_adr + elem_num:
      flexid = i
      break

  if flexid < 0:
    return

  radius = flex_radius[flexid]
  vert_adr = flex_vertadr[flexid]

  local_elemid = elemid - flex_elemadr[flexid]
  elem_data_idx = flex_elemdataadr[flexid] + local_elemid * 4

  v0 = flex_elem[elem_data_idx]
  v1 = flex_elem[elem_data_idx + 1]
  v2 = flex_elem[elem_data_idx + 2]
  v3 = flex_elem[elem_data_idx + 3]

  p0 = flexvert_xpos_in[worldid, vert_adr + v0]
  p1 = flexvert_xpos_in[worldid, vert_adr + v1]
  p2 = flexvert_xpos_in[worldid, vert_adr + v2]
  p3 = flexvert_xpos_in[worldid, vert_adr + v3]

  # Test face (0,1,2) vs Vertex 3
  ok0, dist0, pos0, nrm0 = _plane_vertex(p3, radius, p0, p1, p2)
  if ok0:
    _write_candidate_contact(
      max_candidates,
      dist0,
      pos0,
      nrm0,
      -1,
      flexid,
      local_elemid,
      v3,
      worldid,
      cand_dist_out,
      cand_pos_out,
      cand_nrm_out,
      cand_geom_out,
      cand_flex_out,
      cand_elem_out,
      cand_vert_out,
      cand_worldid_out,
      cand_type_out,
      cand_geomcollisionid_out,
      ncand_out,
    )

  # Test face (0,2,3) vs Vertex 1
  ok1, dist1, pos1, nrm1 = _plane_vertex(p1, radius, p0, p2, p3)
  if ok1:
    _write_candidate_contact(
      max_candidates,
      dist1,
      pos1,
      nrm1,
      -1,
      flexid,
      local_elemid,
      v1,
      worldid,
      cand_dist_out,
      cand_pos_out,
      cand_nrm_out,
      cand_geom_out,
      cand_flex_out,
      cand_elem_out,
      cand_vert_out,
      cand_worldid_out,
      cand_type_out,
      cand_geomcollisionid_out,
      ncand_out,
    )

  # Test face (0,3,1) vs Vertex 2
  ok2, dist2, pos2, nrm2 = _plane_vertex(p2, radius, p0, p3, p1)
  if ok2:
    _write_candidate_contact(
      max_candidates,
      dist2,
      pos2,
      nrm2,
      -1,
      flexid,
      local_elemid,
      v2,
      worldid,
      cand_dist_out,
      cand_pos_out,
      cand_nrm_out,
      cand_geom_out,
      cand_flex_out,
      cand_elem_out,
      cand_vert_out,
      cand_worldid_out,
      cand_type_out,
      cand_geomcollisionid_out,
      ncand_out,
    )

  # Test face (1,3,2) vs Vertex 0
  ok3, dist3, pos3, nrm3 = _plane_vertex(p0, radius, p1, p3, p2)
  if ok3:
    _write_candidate_contact(
      max_candidates,
      dist3,
      pos3,
      nrm3,
      -1,
      flexid,
      local_elemid,
      v0,
      worldid,
      cand_dist_out,
      cand_pos_out,
      cand_nrm_out,
      cand_geom_out,
      cand_flex_out,
      cand_elem_out,
      cand_vert_out,
      cand_worldid_out,
      cand_type_out,
      cand_geomcollisionid_out,
      ncand_out,
    )


@wp.func
def _exclude_self_collision(
  # Model:
  flex_vertbodyid: wp.array[int],
  # In:
  v1: wp.vec4i,
  n1: int,
  v2: wp.vec4i,
  n2: int,
  vert_adr: int,
) -> bool:
  for i in range(n1):
    idx1 = v1[i]
    if idx1 >= 0:
      b1 = flex_vertbodyid[vert_adr + idx1]
      for j in range(n2):
        idx2 = v2[j]
        if idx1 == idx2:
          return True
        if idx2 >= 0 and b1 >= 0:
          b2 = flex_vertbodyid[vert_adr + idx2]
          if b1 == b2:
            return True
  return False


@wp.func
def _get_element_vertices(
  # Model:
  flex_elem: wp.array[int],
  # In:
  dim: int,
  elem_data_idx: int,
) -> wp.vec4i:
  v0 = flex_elem[elem_data_idx]
  v1 = flex_elem[elem_data_idx + 1]
  v2 = int(-1)
  v3 = int(-1)
  if dim >= 2:
    v2 = flex_elem[elem_data_idx + 2]
  if dim >= 3:
    v3 = flex_elem[elem_data_idx + 3]
  return wp.vec4i(v0, v1, v2, v3)


@wp.func
def _elements_overlap(
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  dim: int,
  radius: float,
  v1_indices: wp.vec4i,
  v2_indices: wp.vec4i,
  vert_adr: int,
  worldid: int,
) -> bool:
  p1_0 = flexvert_xpos_in[worldid, vert_adr + v1_indices[0]]
  p1_1 = flexvert_xpos_in[worldid, vert_adr + v1_indices[1]]

  min1 = wp.min(p1_0, p1_1)
  max1 = wp.max(p1_0, p1_1)

  if dim >= 2:
    p1_2 = flexvert_xpos_in[worldid, vert_adr + v1_indices[2]]
    min1 = wp.min(min1, p1_2)
    max1 = wp.max(max1, p1_2)
  if dim >= 3:
    p1_3 = flexvert_xpos_in[worldid, vert_adr + v1_indices[3]]
    min1 = wp.min(min1, p1_3)
    max1 = wp.max(max1, p1_3)

  p2_0 = flexvert_xpos_in[worldid, vert_adr + v2_indices[0]]
  p2_1 = flexvert_xpos_in[worldid, vert_adr + v2_indices[1]]

  min2 = wp.min(p2_0, p2_1)
  max2 = wp.max(p2_0, p2_1)

  if dim >= 2:
    p2_2 = flexvert_xpos_in[worldid, vert_adr + v2_indices[2]]
    min2 = wp.min(min2, p2_2)
    max2 = wp.max(max2, p2_2)
  if dim >= 3:
    p2_3 = flexvert_xpos_in[worldid, vert_adr + v2_indices[3]]
    min2 = wp.min(min2, p2_3)
    max2 = wp.max(max2, p2_3)

  rbound = 2.0 * radius

  if min1[0] - rbound > max2[0] or max1[0] + rbound < min2[0]:
    return False
  if min1[1] - rbound > max2[1] or max1[1] + rbound < min2[1]:
    return False
  if min1[2] - rbound > max2[2] or max1[2] + rbound < min2[2]:
    return False

  return True


@wp.kernel(module="unique", enable_backward=False)
def _flex_active_element_collisions_detect(
  # Model:
  nflex: int,
  opt_ccd_tolerance: wp.array[float],
  flex_selfcollide: wp.array[int],
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_radius: wp.array[float],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  max_candidates: int,
  gjk_iterations: int,
  epa_iterations: int,
  n_total_elems: int,
  # Out:
  workspace_verts_out: wp.array[wp.vec3],
  epa_vert_out: wp.array2d[wp.vec3],
  epa_vert_index_out: wp.array2d[int],
  epa_face_out: wp.array2d[int],
  epa_pr_out: wp.array2d[wp.vec3],
  epa_norm2_out: wp.array2d[float],
  epa_horizon_out: wp.array2d[int],
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  cand_type_out: wp.array[int],
  cand_geomcollisionid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  worldid, elem1_global = wp.tid()

  flexid = int(-1)
  for i in range(nflex):
    if flex_selfcollide[i] == 0:
      continue
    elem_adr = flex_elemadr[i]
    elem_num = flex_elemnum[i]
    if elem1_global >= elem_adr and elem1_global < elem_adr + elem_num:
      flexid = i
      break

  if flexid < 0:
    return

  radius = flex_radius[flexid]
  dim = flex_dim[flexid]
  vert_adr = flex_vertadr[flexid]
  elem_adr = flex_elemadr[flexid]
  elem_num = flex_elemnum[flexid]

  e1 = elem1_global - elem_adr
  elem_data_idx1 = flex_elemdataadr[flexid] + e1 * (dim + 1)

  v1_indices = _get_element_vertices(flex_elem, dim, elem_data_idx1)

  unique_thread_id = worldid * n_total_elems + elem1_global

  offset1 = unique_thread_id * 8
  for idx in range(dim + 1):
    workspace_verts_out[offset1 + idx] = flexvert_xpos_in[worldid, vert_adr + v1_indices[idx]]

  for e2 in range(e1 + 1, elem_num):
    elem_data_idx2 = flex_elemdataadr[flexid] + e2 * (dim + 1)
    v2_indices = _get_element_vertices(flex_elem, dim, elem_data_idx2)

    if _exclude_self_collision(flex_vertbodyid, v1_indices, dim + 1, v2_indices, dim + 1, vert_adr):
      continue

    overlap = _elements_overlap(flexvert_xpos_in, dim, radius, v1_indices, v2_indices, vert_adr, worldid)
    if not overlap:
      continue

    if dim == 1:
      p0 = workspace_verts_out[offset1]
      p1 = workspace_verts_out[offset1 + 1]
      cap1_pos = 0.5 * (p0 + p1)
      cap1_axis = wp.normalize(p1 - p0)
      cap1_half_len = 0.5 * wp.length(p1 - p0)

      p2_0 = flexvert_xpos_in[worldid, vert_adr + v2_indices[0]]
      p2_1 = flexvert_xpos_in[worldid, vert_adr + v2_indices[1]]
      cap2_pos = 0.5 * (p2_0 + p2_1)
      cap2_axis = wp.normalize(p2_1 - p2_0)
      cap2_half_len = 0.5 * wp.length(p2_1 - p2_0)

      margin = 0.0

      contact_dist, contact_pos, contact_normal = collision_primitive_core.capsule_capsule(
        cap1_pos, cap1_axis, radius, cap1_half_len, cap2_pos, cap2_axis, radius, cap2_half_len, margin
      )

      for c in range(2):
        d_val = contact_dist[c]
        if d_val < 0.0:
          _write_candidate_contact(
            max_candidates,
            d_val,
            contact_pos[c],
            contact_normal[c],
            -2,
            flexid,
            e1,
            e2,
            worldid,
            cand_dist_out,
            cand_pos_out,
            cand_nrm_out,
            cand_geom_out,
            cand_flex_out,
            cand_elem_out,
            cand_vert_out,
            cand_worldid_out,
            cand_type_out,
            cand_geomcollisionid_out,
            ncand_out,
          )
    else:
      offset2 = unique_thread_id * 8 + 4
      for idx in range(dim + 1):
        workspace_verts_out[offset2 + idx] = flexvert_xpos_in[worldid, vert_adr + v2_indices[idx]]

      geom1 = Geom()
      geom1.pos = wp.vec3(0.0)
      geom1.rot = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
      geom1.size = wp.vec3(0.0)
      geom1.margin = 2.0 * radius
      geom1.vert = workspace_verts_out
      geom1.vertadr = offset1
      geom1.vertnum = dim + 1
      geom1.graphadr = -1
      geom1.index = -1

      geom2 = Geom()
      geom2.pos = wp.vec3(0.0)
      geom2.rot = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
      geom2.size = wp.vec3(0.0)
      geom2.margin = 2.0 * radius
      geom2.vert = workspace_verts_out
      geom2.vertadr = offset2
      geom2.vertnum = dim + 1
      geom2.graphadr = -1
      geom2.index = -1

      center1 = wp.vec3(0.0)
      for idx in range(dim + 1):
        center1 += workspace_verts_out[offset1 + idx]
      center1 = center1 / float(dim + 1)

      center2 = wp.vec3(0.0)
      for idx in range(dim + 1):
        center2 += workspace_verts_out[offset2 + idx]
      center2 = center2 / float(dim + 1)

      tol = opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]]

      dist, ncontact, w1, w2, _ = ccd(
        tol,
        2.0 * radius,
        gjk_iterations,
        epa_iterations,
        geom1,
        geom2,
        int(GeomType.MESH),
        int(GeomType.MESH),
        center1,
        center2,
        epa_vert_out[unique_thread_id],
        epa_vert_index_out[unique_thread_id],
        epa_face_out[unique_thread_id],
        epa_pr_out[unique_thread_id],
        epa_norm2_out[unique_thread_id],
        epa_horizon_out[unique_thread_id],
      )

      phys_dist = dist
      if ncontact > 0 and phys_dist < 0.0:
        pos = 0.5 * (w1 + w2)
        nrm = wp.normalize(w1 - w2)
        _write_candidate_contact(
          max_candidates,
          phys_dist,
          pos,
          nrm,
          -2,
          flexid,
          e1,
          e2,
          worldid,
          cand_dist_out,
          cand_pos_out,
          cand_nrm_out,
          cand_geom_out,
          cand_flex_out,
          cand_elem_out,
          cand_vert_out,
          cand_worldid_out,
          cand_type_out,
          cand_geomcollisionid_out,
          ncand_out,
        )


@wp.kernel
def _flex_narrowphase_dim2_detect(
  # Model:
  ngeom: int,
  nflex: int,
  geom_type: wp.array[int],
  geom_contype: wp.array[int],
  geom_conaffinity: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  flex_contype: wp.array[int],
  flex_conaffinity: wp.array[int],
  flex_margin: wp.array[float],
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
  # In:
  max_candidates: int,
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  cand_type_out: wp.array[int],
  cand_geomcollisionid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  worldid, elemid = wp.tid()

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

  local_elemid = elemid - flex_elemadr[flexid]
  elem_data_idx = flex_elemdataadr[flexid] + local_elemid * 3
  v0_local = flex_elem[elem_data_idx]
  v1_local = flex_elem[elem_data_idx + 1]
  v2_local = flex_elem[elem_data_idx + 2]

  t1 = flexvert_xpos_in[worldid, vert_adr + v0_local]
  t2 = flexvert_xpos_in[worldid, vert_adr + v1_local]
  t3 = flexvert_xpos_in[worldid, vert_adr + v2_local]

  # TODO: Add a broadphase
  for geomid in range(ngeom):
    gtype = geom_type[geomid]
    if (
      gtype != int(GeomType.SPHERE)
      and gtype != int(GeomType.CAPSULE)
      and gtype != int(GeomType.BOX)
      and gtype != int(GeomType.CYLINDER)
    ):
      continue

    g_contype = geom_contype[geomid]
    g_conaffinity = geom_conaffinity[geomid]
    f_contype = flex_contype[flexid]
    f_conaffinity = flex_conaffinity[flexid]
    if not ((g_contype & f_conaffinity) or (f_contype & g_conaffinity)):
      continue

    geom_margin_val = geom_margin[worldid % geom_margin.shape[0], geomid]
    margin = geom_margin_val + tri_margin

    geom_pos = geom_xpos_in[worldid, geomid]
    geom_rot = geom_xmat_in[worldid, geomid]
    geom_size_val = geom_size[worldid % geom_size.shape[0], geomid]

    _collide_geom_triangle_detect(
      max_candidates,
      gtype,
      geom_pos,
      geom_rot,
      geom_size_val,
      t1,
      t2,
      t3,
      tri_radius,
      margin,
      geomid,
      flexid,
      local_elemid,
      -1,
      worldid,
      cand_dist_out,
      cand_pos_out,
      cand_nrm_out,
      cand_geom_out,
      cand_flex_out,
      cand_elem_out,
      cand_vert_out,
      cand_worldid_out,
      cand_type_out,
      cand_geomcollisionid_out,
      ncand_out,
    )


@wp.kernel
def _flex_narrowphase_dim3_detect(
  # Model:
  ngeom: int,
  nflex: int,
  geom_type: wp.array[int],
  geom_contype: wp.array[int],
  geom_conaffinity: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  flex_contype: wp.array[int],
  flex_conaffinity: wp.array[int],
  flex_margin: wp.array[float],
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
  # In:
  max_candidates: int,
  # Out:
  cand_dist_out: wp.array[float],
  cand_pos_out: wp.array[wp.vec3],
  cand_nrm_out: wp.array[wp.vec3],
  cand_geom_out: wp.array[wp.vec2i],
  cand_flex_out: wp.array[wp.vec2i],
  cand_elem_out: wp.array[wp.vec2i],
  cand_vert_out: wp.array[wp.vec2i],
  cand_worldid_out: wp.array[int],
  cand_type_out: wp.array[int],
  cand_geomcollisionid_out: wp.array[int],
  ncand_out: wp.array[int],
):
  worldid, shellid = wp.tid()

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

  # TODO: Add a broadphase
  for geomid in range(ngeom):
    gtype = geom_type[geomid]
    if (
      gtype != int(GeomType.SPHERE)
      and gtype != int(GeomType.CAPSULE)
      and gtype != int(GeomType.BOX)
      and gtype != int(GeomType.CYLINDER)
    ):
      continue

    g_contype = geom_contype[geomid]
    g_conaffinity = geom_conaffinity[geomid]
    f_contype = flex_contype[flexid]
    f_conaffinity = flex_conaffinity[flexid]
    if not ((g_contype & f_conaffinity) or (f_contype & g_conaffinity)):
      continue

    geom_margin_val = geom_margin[worldid % geom_margin.shape[0], geomid]
    margin = geom_margin_val + tri_margin

    geom_pos = geom_xpos_in[worldid, geomid]
    geom_rot = geom_xmat_in[worldid, geomid]
    geom_size_val = geom_size[worldid % geom_size.shape[0], geomid]

    _collide_geom_triangle_detect(
      max_candidates,
      gtype,
      geom_pos,
      geom_rot,
      geom_size_val,
      t1,
      t2,
      t3,
      tri_radius,
      margin,
      geomid,
      flexid,
      local_shellid,
      -1,
      worldid,
      cand_dist_out,
      cand_pos_out,
      cand_nrm_out,
      cand_geom_out,
      cand_flex_out,
      cand_elem_out,
      cand_vert_out,
      cand_worldid_out,
      cand_type_out,
      cand_geomcollisionid_out,
      ncand_out,
    )


@wp.kernel
def _filter_flex_candidates(
  # In:
  max_candidates: int,
  ncand: wp.array[int],
  epsilon: float,
  cand_dist: wp.array[float],
  cand_pos: wp.array[wp.vec3],
  cand_geom: wp.array[wp.vec2i],
  cand_flex: wp.array[wp.vec2i],
  cand_worldid: wp.array[int],
  # Out:
  cand_active_out: wp.array[int],
):
  i = wp.tid()
  limit = ncand[0]
  if i >= limit:
    return

  geom_i = cand_geom[i][0]
  flex_i = cand_flex[i][1]
  world_i = cand_worldid[i]
  pos_i = cand_pos[i]
  dist_i = cand_dist[i]

  keep = int(1)
  for j in range(max_candidates):
    if j >= limit:
      break
    if j == i:
      continue
    geom_j = cand_geom[j][0]
    if (geom_i >= 0 and geom_j >= 0 and geom_j == geom_i) or (geom_i < 0 and geom_j < 0):
      flex_j = cand_flex[j][1]
      if flex_j == flex_i:
        world_j = cand_worldid[j]
        if world_j == world_i:
          pos_j = cand_pos[j]
          dist_j = cand_dist[j]

          diff = pos_i - pos_j
          if wp.dot(diff, diff) < epsilon * epsilon:
            if dist_j < dist_i:
              keep = 0
            elif dist_j == dist_i and j < i:
              keep = 0

  cand_active_out[i] = keep


@wp.kernel
def _write_filtered_contacts(
  # Model:
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
  flex_dim: wp.array[int],
  # Data in:
  naconmax_in: int,
  # In:
  ncand: wp.array[int],
  cand_dist: wp.array[float],
  cand_pos: wp.array[wp.vec3],
  cand_nrm: wp.array[wp.vec3],
  cand_geom: wp.array[wp.vec2i],
  cand_flex: wp.array[wp.vec2i],
  cand_elem: wp.array[wp.vec2i],
  cand_vert: wp.array[wp.vec2i],
  cand_worldid: wp.array[int],
  cand_type: wp.array[int],
  cand_geomcollisionid: wp.array[int],
  cand_active: wp.array[int],
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
  contact_elem_out: wp.array[wp.vec2i],
  contact_vert_out: wp.array[wp.vec2i],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  nacon_out: wp.array[int],
):
  i = wp.tid()
  if i >= ncand[0]:
    return

  if cand_active[i] == 0:
    return

  geomid = cand_geom[i][0]
  worldid = cand_worldid[i]

  condim = int(0)
  margin = float(0.0)
  gap = float(0.0)
  solref = wp.vec2(0.0, 0.0)
  solimp = vec5(0.0, 0.0, 0.0, 0.0, 0.0)
  friction = vec5(0.0, 0.0, 0.0, 0.0, 0.0)

  if geomid >= 0:
    flexid = cand_flex[i][1]
    geom_margin_val = geom_margin[worldid % geom_margin.shape[0], geomid]
    tri_margin = flex_margin[flexid]
    margin = geom_margin_val + tri_margin

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
  else:
    flex1 = cand_flex[i][0]
    flex2 = cand_flex[i][1]
    margin = 0.0
    gap = 0.0

    mixed_condim, _, solref, solimp, friction = _mix_flex_contact_params(
      flex_condim[flex1],
      flex_priority[flex1],
      flex_solmix[flex1],
      flex_solref[flex1],
      flex_solimp[flex1],
      flex_friction[flex1],
      0.0,
      flex_condim[flex2],
      flex_priority[flex2],
      flex_solmix[flex2],
      flex_solref[flex2],
      flex_solimp[flex2],
      flex_friction[flex2],
      0.0,
    )

    if cand_vert[i][0] >= 0 and flex_dim[flex1] == 3:
      condim = 1
    else:
      condim = mixed_condim

  id_ = wp.atomic_add(nacon_out, 0, 1)
  if id_ >= naconmax_in:
    return

  contact_dist_out[id_] = cand_dist[i]
  contact_pos_out[id_] = cand_pos[i]
  contact_frame_out[id_] = make_frame(cand_nrm[i])
  if geomid >= 0 and geom_type[geomid] == int(GeomType.PLANE):
    contact_includemargin_out[id_] = margin - gap
  else:
    contact_includemargin_out[id_] = margin
  contact_friction_out[id_] = friction
  contact_solref_out[id_] = solref
  contact_solreffriction_out[id_] = wp.vec2(0.0, 0.0)
  contact_solimp_out[id_] = solimp
  contact_dim_out[id_] = condim
  contact_geom_out[id_] = cand_geom[i]
  contact_flex_out[id_] = cand_flex[i]
  contact_elem_out[id_] = cand_elem[i]
  contact_vert_out[id_] = cand_vert[i]
  contact_worldid_out[id_] = cand_worldid[i]
  contact_type_out[id_] = cand_type[i]
  contact_geomcollisionid_out[id_] = cand_geomcollisionid[i]


@event_scope
def flex_narrowphase(m: Model, d: Data):
  """Runs collision detection between geoms and flex elements."""
  if m.nflex == 0:
    return

  cand_dist = wp.empty(d.naconmax, dtype=float)
  cand_pos = wp.empty(d.naconmax, dtype=wp.vec3)
  cand_nrm = wp.empty(d.naconmax, dtype=wp.vec3)
  cand_geom = wp.empty(d.naconmax, dtype=wp.vec2i)
  cand_flex = wp.empty(d.naconmax, dtype=wp.vec2i)
  cand_elem = wp.empty(d.naconmax, dtype=wp.vec2i)
  cand_vert = wp.empty(d.naconmax, dtype=wp.vec2i)
  cand_worldid = wp.empty(d.naconmax, dtype=int)
  cand_type = wp.empty(d.naconmax, dtype=int)
  cand_geomcollisionid = wp.empty(d.naconmax, dtype=int)

  ncand = wp.zeros(1, dtype=int)

  wp.launch(
    _flex_narrowphase_dim2_detect,
    dim=(d.nworld, m.nflexelem),
    inputs=[
      m.ngeom,
      m.nflex,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_size,
      m.geom_margin,
      m.flex_contype,
      m.flex_conaffinity,
      m.flex_margin,
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
    ],
    outputs=[
      cand_dist,
      cand_pos,
      cand_nrm,
      cand_geom,
      cand_flex,
      cand_elem,
      cand_vert,
      cand_worldid,
      cand_type,
      cand_geomcollisionid,
      ncand,
    ],
  )

  wp.launch(
    _flex_narrowphase_dim3_detect,
    dim=(d.nworld, m.nflexshelldata // 3),
    inputs=[
      m.ngeom,
      m.nflex,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_size,
      m.geom_margin,
      m.flex_contype,
      m.flex_conaffinity,
      m.flex_margin,
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
    ],
    outputs=[
      cand_dist,
      cand_pos,
      cand_nrm,
      cand_geom,
      cand_flex,
      cand_elem,
      cand_vert,
      cand_worldid,
      cand_type,
      cand_geomcollisionid,
      ncand,
    ],
  )

  wp.launch(
    _flex_plane_narrowphase_detect,
    dim=(d.nworld, m.nflexvert),
    inputs=[
      m.ngeom,
      m.nflexvert,
      m.geom_type,
      m.geom_margin,
      m.flex_margin,
      m.flex_vertadr,
      m.flex_radius,
      m.flex_vertflexid,
      d.geom_xpos,
      d.geom_xmat,
      d.flexvert_xpos,
      d.nworld,
      d.naconmax,
    ],
    outputs=[
      cand_dist,
      cand_pos,
      cand_nrm,
      cand_geom,
      cand_flex,
      cand_elem,
      cand_vert,
      cand_worldid,
      cand_type,
      cand_geomcollisionid,
      ncand,
    ],
  )

  if m.nflexevpair > 0:
    wp.launch(
      _flex_internal_collisions_detect,
      dim=(d.nworld, m.nflexevpair),
      inputs=[
        m.nflex,
        m.flex_margin,
        m.flex_internal,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_elemdataadr,
        m.flex_evpairadr,
        m.flex_evpairnum,
        m.flex_elem,
        m.flex_evpair,
        m.flex_radius,
        d.flexvert_xpos,
        d.naconmax,
      ],
      outputs=[
        cand_dist,
        cand_pos,
        cand_nrm,
        cand_geom,
        cand_flex,
        cand_elem,
        cand_vert,
        cand_worldid,
        cand_type,
        cand_geomcollisionid,
        ncand,
      ],
    )

  if m.nflexelem > 0:
    wp.launch(
      _flex_tet_internal_collisions_detect,
      dim=(d.nworld, m.nflexelem),
      inputs=[
        m.nflex,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_elemadr,
        m.flex_elemnum,
        m.flex_elemdataadr,
        m.flex_elem,
        m.flex_radius,
        d.flexvert_xpos,
        d.naconmax,
      ],
      outputs=[
        cand_dist,
        cand_pos,
        cand_nrm,
        cand_geom,
        cand_flex,
        cand_elem,
        cand_vert,
        cand_worldid,
        cand_type,
        cand_geomcollisionid,
        ncand,
      ],
    )

  selfcollide_enabled = m.has_flex_selfcollide

  if selfcollide_enabled and m.nflexelem > 0:
    workspace_verts = wp.empty(d.nworld * m.nflexelem * 8, dtype=wp.vec3)

    epa_iterations = m.opt.ccd_iterations
    if m.max_flex_dim > 1:
      epa_vert = wp.empty(shape=(d.nworld * m.nflexelem, 10 + 2 * epa_iterations), dtype=wp.vec3)
      epa_vert_index = wp.empty(shape=(d.nworld * m.nflexelem, 10 + 2 * epa_iterations), dtype=int)
      epa_face = wp.empty(shape=(d.nworld * m.nflexelem, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=int)
      epa_pr = wp.empty(shape=(d.nworld * m.nflexelem, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=wp.vec3)
      epa_norm2 = wp.empty(shape=(d.nworld * m.nflexelem, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=float)
      epa_horizon = wp.empty(shape=(d.nworld * m.nflexelem, MJ_MAX_EPAHORIZON), dtype=int)
    else:
      epa_vert = wp.empty(shape=(1, 1), dtype=wp.vec3)
      epa_vert_index = wp.empty(shape=(1, 1), dtype=int)
      epa_face = wp.empty(shape=(1, 1), dtype=int)
      epa_pr = wp.empty(shape=(1, 1), dtype=wp.vec3)
      epa_norm2 = wp.empty(shape=(1, 1), dtype=float)
      epa_horizon = wp.empty(shape=(1, 1), dtype=int)

    if selfcollide_enabled:
      wp.launch(
        _flex_active_element_collisions_detect,
        dim=(d.nworld, m.nflexelem),
        inputs=[
          m.nflex,
          m.opt.ccd_tolerance,
          m.flex_selfcollide,
          m.flex_dim,
          m.flex_vertadr,
          m.flex_elemadr,
          m.flex_elemnum,
          m.flex_elemdataadr,
          m.flex_vertbodyid,
          m.flex_elem,
          m.flex_radius,
          d.flexvert_xpos,
          d.naconmax,
          m.opt.ccd_iterations,
          epa_iterations,
          m.nflexelem,
        ],
        outputs=[
          workspace_verts,
          epa_vert,
          epa_vert_index,
          epa_face,
          epa_pr,
          epa_norm2,
          epa_horizon,
          cand_dist,
          cand_pos,
          cand_nrm,
          cand_geom,
          cand_flex,
          cand_elem,
          cand_vert,
          cand_worldid,
          cand_type,
          cand_geomcollisionid,
          ncand,
        ],
      )

  # Filter duplicate contacts (e.g. from shared vertices or edges)
  cand_active = wp.empty(d.naconmax, dtype=int)
  wp.launch(
    _filter_flex_candidates,
    dim=d.naconmax,
    inputs=[
      d.naconmax,
      ncand,
      1e-3,  # epsilon
      cand_dist,
      cand_pos,
      cand_geom,
      cand_flex,
      cand_worldid,
    ],
    outputs=[
      cand_active,
    ],
  )

  # Copy filtered contacts to the main d.contact array, computing contact parameters on-the-fly
  wp.launch(
    _write_filtered_contacts,
    dim=d.naconmax,
    inputs=[
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
      m.flex_dim,
      d.naconmax,
      ncand,
      cand_dist,
      cand_pos,
      cand_nrm,
      cand_geom,
      cand_flex,
      cand_elem,
      cand_vert,
      cand_worldid,
      cand_type,
      cand_geomcollisionid,
      cand_active,
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
      d.contact.elem,
      d.contact.vert,
      d.contact.worldid,
      d.contact.type,
      d.contact.geomcollisionid,
      d.nacon,
    ],
  )
