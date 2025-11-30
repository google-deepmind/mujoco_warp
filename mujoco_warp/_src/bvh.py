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

from .render_context import RenderContext
from .types import Data
from .types import GeomType
from .types import Model

wp.set_module_options({"enable_backward": False})


@wp.func
def _compute_box_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  min_bound = wp.vec3(wp.inf, wp.inf, wp.inf)
  max_bound = wp.vec3(-wp.inf, -wp.inf, -wp.inf)

  for i in range(2):
    for j in range(2):
      for k in range(2):
        local_corner = wp.vec3(
          size[0] * (2.0 * float(i) - 1.0),
          size[1] * (2.0 * float(j) - 1.0),
          size[2] * (2.0 * float(k) - 1.0),
        )
        world_corner = pos + rot @ local_corner
        min_bound = wp.min(min_bound, world_corner)
        max_bound = wp.max(max_bound, world_corner)

  return min_bound, max_bound


@wp.func
def _compute_sphere_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  radius = size[0]
  return pos - wp.vec3(radius, radius, radius), pos + wp.vec3(radius, radius, radius)


@wp.func
def _compute_capsule_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  radius = size[0]
  half_length = size[1]
  local_end1 = wp.vec3(0.0, 0.0, -half_length)
  local_end2 = wp.vec3(0.0, 0.0, half_length)
  world_end1 = pos + rot @ local_end1
  world_end2 = pos + rot @ local_end2

  seg_min = wp.min(world_end1, world_end2)
  seg_max = wp.max(world_end1, world_end2)

  inflate = wp.vec3(radius, radius, radius)
  return seg_min - inflate, seg_max + inflate


@wp.func
def _compute_plane_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  # If plane size is non-positive, treat as infinite plane and use a large default extent
  size_scale = wp.max(size[0], size[1]) * 2.0
  if size[0] <= 0.0 or size[1] <= 0.0:
    size_scale = 1000.0
  min_bound = wp.vec3(wp.inf, wp.inf, wp.inf)
  max_bound = wp.vec3(-wp.inf, -wp.inf, -wp.inf)

  for i in range(2):
    for j in range(2):
      local_corner = wp.vec3(
        size_scale * (2.0 * float(i) - 1.0),
        size_scale * (2.0 * float(j) - 1.0),
        0.0,
      )
      world_corner = pos + rot @ local_corner
      min_bound = wp.min(min_bound, world_corner)
      max_bound = wp.max(max_bound, world_corner)

  min_bound = min_bound - wp.vec3(0.1, 0.1, 0.1)
  max_bound = max_bound + wp.vec3(0.1, 0.1, 0.1)

  return min_bound, max_bound


@wp.func
def _compute_ellipsoid_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  # Half-extent along each world axis equals the norm of the corresponding row of rot*diag(size)
  row0 = wp.vec3(rot[0, 0] * size[0], rot[0, 1] * size[1], rot[0, 2] * size[2])
  row1 = wp.vec3(rot[1, 0] * size[0], rot[1, 1] * size[1], rot[1, 2] * size[2])
  row2 = wp.vec3(rot[2, 0] * size[0], rot[2, 1] * size[1], rot[2, 2] * size[2])
  extent = wp.vec3(wp.length(row0), wp.length(row1), wp.length(row2))
  return pos - extent, pos + extent


@wp.func
def _compute_cylinder_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  radius = size[0]
  half_height = size[1]

  axis = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])
  axis_abs = wp.vec3(wp.abs(axis[0]), wp.abs(axis[1]), wp.abs(axis[2]))

  basis_x = wp.vec3(rot[0, 0], rot[1, 0], rot[2, 0])
  basis_y = wp.vec3(rot[0, 1], rot[1, 1], rot[2, 1])

  radial_x = radius * wp.sqrt(basis_x[0] * basis_x[0] + basis_y[0] * basis_y[0])
  radial_y = radius * wp.sqrt(basis_x[1] * basis_x[1] + basis_y[1] * basis_y[1])
  radial_z = radius * wp.sqrt(basis_x[2] * basis_x[2] + basis_y[2] * basis_y[2])

  extent = wp.vec3(
    radial_x + half_height * axis_abs[0],
    radial_y + half_height * axis_abs[1],
    radial_z + half_height * axis_abs[2],
  )

  return pos - extent, pos + extent


@wp.kernel
def _compute_bvh_bounds(
  bvh_ngeom: int,
  nworld: int,
  enabled_geom_ids: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_pos: wp.array2d(dtype=wp.vec3),
  geom_rot: wp.array2d(dtype=wp.mat33),
  mesh_bounds_size: wp.array(dtype=wp.vec3),
  hfield_bounds_size: wp.array(dtype=wp.vec3),
  lowers: wp.array(dtype=wp.vec3),
  uppers: wp.array(dtype=wp.vec3),
  groups: wp.array(dtype=int),
):
  tid = wp.tid()
  world_id = tid // bvh_ngeom
  bvh_geom_local = tid % bvh_ngeom

  if bvh_geom_local >= bvh_ngeom or world_id >= nworld:
    return

  geom_id = enabled_geom_ids[bvh_geom_local]

  pos = geom_pos[world_id, geom_id]
  rot = geom_rot[world_id, geom_id]
  size = geom_size[world_id, geom_id]
  type = geom_type[geom_id]

  if type == GeomType.SPHERE:
    lower, upper = _compute_sphere_bounds(pos, rot, size)
  elif type == GeomType.CAPSULE:
    lower, upper = _compute_capsule_bounds(pos, rot, size)
  elif type == GeomType.PLANE:
    lower, upper = _compute_plane_bounds(pos, rot, size)
  elif type == GeomType.MESH:
    size = mesh_bounds_size[geom_dataid[geom_id]]
    lower, upper = _compute_box_bounds(pos, rot, size)
  elif type == GeomType.ELLIPSOID:
    lower, upper = _compute_ellipsoid_bounds(pos, rot, size)
  elif type == GeomType.CYLINDER:
    lower, upper = _compute_cylinder_bounds(pos, rot, size)
  elif type == GeomType.BOX:
    lower, upper = _compute_box_bounds(pos, rot, size)
  elif type == GeomType.HFIELD:
    size = hfield_bounds_size[geom_dataid[geom_id]]
    lower, upper = _compute_box_bounds(pos, rot, size)

  lowers[world_id * bvh_ngeom + bvh_geom_local] = lower
  uppers[world_id * bvh_ngeom + bvh_geom_local] = upper
  groups[world_id * bvh_ngeom + bvh_geom_local] = world_id


@wp.kernel
def compute_bvh_group_roots(
  bvh_id: wp.uint64,
  group_roots: wp.array(dtype=int),
):
  tid = wp.tid()
  root = wp.bvh_get_group_root(bvh_id, tid)
  group_roots[tid] = root


def build_warp_bvh(m: Model, d: Data, rc: RenderContext):
  """Build a Warp BVH for all geometries in all worlds."""
  wp.launch(
    kernel=_compute_bvh_bounds,
    dim=d.nworld * rc.bvh_ngeom,
    inputs=[
      rc.bvh_ngeom,
      d.nworld,
      rc.enabled_geom_ids,
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      d.geom_xpos,
      d.geom_xmat,
      rc.mesh_bounds_size,
      rc.hfield_bounds_size,
      rc.lowers,
      rc.uppers,
      rc.groups,
    ],
  )

  bvh = wp.Bvh(rc.lowers, rc.uppers, groups=rc.groups)

  # BVH handle must be stored to avoid garbage collection
  rc.bvh = bvh
  rc.bvh_id = bvh.id

  wp.launch(
    kernel=compute_bvh_group_roots,
    dim=d.nworld,
    inputs=[bvh.id, rc.group_roots],
  )


def refit_warp_bvh(m: Model, d: Data, rc: RenderContext):
  wp.launch(
    kernel=_compute_bvh_bounds,
    dim=d.nworld * rc.bvh_ngeom,
    inputs=[
      rc.bvh_ngeom,
      d.nworld,
      rc.enabled_geom_ids,
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      d.geom_xpos,
      d.geom_xmat,
      rc.mesh_bounds_size,
      rc.hfield_bounds_size,
      rc.lowers,
      rc.uppers,
      rc.groups,
    ],
  )

  rc.bvh.refit()


@wp.kernel
def accumulate_flex_vertex_normals(
  vert_xpos: wp.array2d(dtype=wp.vec3),
  flex_elem: wp.array(dtype=int),
  vert_norm: wp.array2d(dtype=wp.vec3),
  flex_elem_count: int,
):
  """Accumulate per-vertex normals by summing adjacent face normals."""
  worldid, elemid = wp.tid()

  if worldid >= vert_xpos.shape[0] or elemid >= flex_elem_count:
    return

  base = elemid * 3
  i0 = flex_elem[base + 0]
  i1 = flex_elem[base + 1]
  i2 = flex_elem[base + 2]

  v0 = vert_xpos[worldid, i0]
  v1 = vert_xpos[worldid, i1]
  v2 = vert_xpos[worldid, i2]

  face_nrm = wp.cross(v1 - v0, v2 - v0)
  face_nrm = wp.normalize(face_nrm)
  vert_norm[worldid, i0] += face_nrm
  vert_norm[worldid, i1] += face_nrm
  vert_norm[worldid, i2] += face_nrm


@wp.kernel
def normalize_vertex_normals(
  vert_norm: wp.array2d(dtype=wp.vec3),
):
  """Normalize accumulated vertex normals."""
  worldid, vertid = wp.tid()
  vert_norm[worldid, vertid] = wp.normalize(vert_norm[worldid, vertid])


def refit_flex_bvh(m: Model, d: Data, rc: RenderContext):
  
  flexvert_norm = wp.zeros(d.flexvert_xpos.shape, dtype=wp.vec3)

  wp.launch(
    kernel=accumulate_flex_vertex_normals,
    dim=(d.nworld, d.flexvert_xpos.shape[1]),
    inputs=[
      d.flexvert_xpos,
      rc.flex_elem,
      flexvert_norm,
      d.flexvert_xpos.shape[1],
    ],
  )

  wp.launch(
    kernel=normalize_vertex_normals,
    dim=(d.nworld, d.flexvert_xpos.shape[1]),
    inputs=[flexvert_norm],
  )

  @wp.kernel
  def _update_flex_points(
    vert_xpos: wp.array2d(dtype=wp.vec3),
    flex_elem: wp.array(dtype=int),
    vert_norm: wp.array2d(dtype=wp.vec3),
    elem_adr: int,
    face_offset: int,
    radius: float,
    nfaces: int,
    face_points: wp.array(dtype=wp.vec3),
  ):
    worldid, elemid = wp.tid()

    base = elem_adr + (elemid * 3)
    i0 = flex_elem[base + 0]
    i1 = flex_elem[base + 1]
    i2 = flex_elem[base + 2]

    v0 = vert_xpos[worldid, i0]
    v1 = vert_xpos[worldid, i1]
    v2 = vert_xpos[worldid, i2]

    if wp.static(rc.flex_render_smooth):
      n0 = vert_norm[worldid, i0]
      n1 = vert_norm[worldid, i1]
      n2 = vert_norm[worldid, i2]
    else:
      face_nrm = wp.cross(v1 - v0, v2 - v0)
      face_nrm = wp.normalize(face_nrm)
      n0 = face_nrm
      n1 = face_nrm
      n2 = face_nrm

    p0_pos = v0 + radius * n0
    p1_pos = v1 + radius * n1
    p2_pos = v2 + radius * n2

    p0_neg = v0 - radius * n0
    p1_neg = v1 - radius * n1
    p2_neg = v2 - radius * n2

    world_face_offset = worldid * nfaces
    base0 = (world_face_offset * 3) + ((face_offset + (2 * elemid)) * 3)
    face_points[base0 + 0] = p0_pos
    face_points[base0 + 1] = p1_pos
    face_points[base0 + 2] = p2_pos

    base1 = (world_face_offset * 3) + ((face_offset + (2 * elemid + 1)) * 3)
    face_points[base1 + 0] = p0_neg
    face_points[base1 + 1] = p2_neg
    face_points[base1 + 2] = p1_neg
  
  @wp.kernel
  def _update_flex_2d_shell_points(
    vert_xpos: wp.array2d(dtype=wp.vec3),
    flex_shell: wp.array(dtype=int),
    vert_norm: wp.array2d(dtype=wp.vec3),
    shell_adr: int,
    face_offset: int,
    radius: float,
    nfaces: int,
    face_points: wp.array(dtype=wp.vec3),
  ):
    worldid, shellid = wp.tid()

    base = shell_adr + (2 * shellid)
    i0 = flex_shell[base + 0]
    i1 = flex_shell[base + 1]

    v0 = vert_xpos[worldid, i0]
    v1 = vert_xpos[worldid, i1]
    
    v01 = v1 - v0
    nrm = wp.cross(v01, vert_norm[worldid, i1])
    
    if radius < 0.0:
      nrm = -nrm

    nrm = wp.normalize(nrm)

    world_face_offset = worldid * nfaces
    base0 = (world_face_offset * 3) + ((face_offset + (2 * shellid)) * 3)
    face_points[base0 + 0] = vert_xpos[worldid, i0] + vert_norm[worldid, i0] * (radius * 1.0)
    face_points[base0 + 1] = vert_xpos[worldid, i1] + vert_norm[worldid, i1] * (radius * -1.0)
    face_points[base0 + 2] = vert_xpos[worldid, i1] + vert_norm[worldid, i1] * (radius * 1.0)

    base1 = (world_face_offset * 3) + ((face_offset + (2 * shellid + 1)) * 3)
    neg_radius = -radius
    face_points[base1 + 0] = vert_xpos[worldid, i1] + vert_norm[worldid, i1] * (neg_radius * 1.0)
    face_points[base1 + 1] = vert_xpos[worldid, i0] + vert_norm[worldid, i0] * (neg_radius * -1.0)
    face_points[base1 + 2] = vert_xpos[worldid, i0] + vert_norm[worldid, i0] * (neg_radius * 1.0)

  @wp.kernel
  def update_flex_3d_shell_points(
    vert_xpos: wp.array2d(dtype=wp.vec3),
    flex_shell: wp.array(dtype=int),
    shell_adr: int,
    face_offset: int,
    radius: float,
    nfaces: int,
    face_points: wp.array(dtype=wp.vec3),
  ):
    worldid, shellid = wp.tid()

    base = shell_adr + (shellid * 3)
    i0 = flex_shell[base + 0]
    i1 = flex_shell[base + 1]
    i2 = flex_shell[base + 2]
    
    v0 = vert_xpos[worldid, i0]
    v1 = vert_xpos[worldid, i1]
    v2 = vert_xpos[worldid, i2]

    world_face_offset = worldid * nfaces
    base = (world_face_offset * 3) + ((face_offset + shellid) * 3)

    face_points[base + 0] = v0
    face_points[base + 1] = v1
    face_points[base + 2] = v2


  for f in range(m.nflex):
    dim = int(rc.flex_dim[f])
    nelem = int(rc.flex_elemnum[f])
    elem_adr = int(rc.flex_elemdataadr[f])
    elem_count = int(rc.flex_elemnum[f])
    nshell = int(rc.flex_shellnum[f])
    shell_adr = int(rc.flex_shelldataadr[f])
    face_offset = int(rc.flex_faceadr[f])
    
    if dim == 2:
      wp.launch(
        kernel=_update_flex_points,
        dim=(d.nworld, nelem),
        inputs=[
          d.flexvert_xpos,
          rc.flex_elem,
          flexvert_norm,
          elem_adr,
          face_offset,
          rc.flex_radius[f],
          rc.flex_nfaces,
          rc.flex_face_points,
        ],
      )
      wp.launch(
        kernel=_update_flex_2d_shell_points,
        dim=(d.nworld, nshell),
        inputs=[
          d.flexvert_xpos,
          rc.flex_shell,
          flexvert_norm,
          shell_adr,
          face_offset + (2 * elem_count),
          rc.flex_radius[f],
          rc.flex_nfaces,
          rc.flex_face_points,
        ],
      )
    else:
      wp.launch(
        kernel=update_flex_3d_shell_points,
        dim=(d.nworld, nshell),
        inputs=[
          d.flexvert_xpos,
          rc.flex_shell,
          shell_adr,
          face_offset,
          rc.flex_radius[f],
          rc.flex_nfaces,
          rc.flex_face_points,
        ],
      )

  rc.flex_registry[rc.flex_bvh_id].refit()

