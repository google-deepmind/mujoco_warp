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
def compute_flex_points(
  vert_xpos: wp.array2d(dtype=wp.vec3),
  flex_elem: wp.array(dtype=int),
  vert_norm: wp.array2d(dtype=wp.vec3),
  face_points: wp.array(dtype=wp.vec3),
  elem_count: int,
  radius: float,
  num_face_vertices: int,
):
  """Update top/bottom cloth faces for all worlds.

  The initial flex mesh is built in `_make_flex_mesh` (see `render_context.py`)
  using `_make_face_2d_elements`, which lays out per-world vertices as:

    world_vertex_offset = world_id * num_face_vertices
    local_face_top      = 2 * elem_id
    local_face_bottom   = 2 * elem_id + 1

  i.e. 6 vertices (2 triangles) per element, with a stride of
  `num_face_vertices` per world. This kernel must mirror that layout exactly
  so that the Warp mesh indices remain valid when we refit the geometry
  each frame.
  """
  worldid, elemid = wp.tid()

  if worldid >= vert_xpos.shape[0] or elemid >= elem_count:
    return

  # Element-local vertex indices (3 verts per triangle element).
  base = elemid * 3
  i0 = flex_elem[base + 0]
  i1 = flex_elem[base + 1]
  i2 = flex_elem[base + 2]

  # Vertex positions for this element in the given world.
  v0 = vert_xpos[worldid, i0]
  v1 = vert_xpos[worldid, i1]
  v2 = vert_xpos[worldid, i2]

  # Per-element normal (used for thickness); we also store it as a simple
  # per-vertex normal for use when building side faces.
  nrm = wp.cross(v1 - v0, v2 - v0)
  nrm_len = wp.length(nrm)
  if nrm_len < 1.0e-8:
    nrm = wp.vec3(0.0, 0.0, 1.0)
  else:
    nrm = nrm / nrm_len
  offset = nrm * radius

  # Store the same normal for all three vertices of the element. This is an
  # approximation (shared across adjacent elements), but matches how the
  # initial mesh was constructed and is sufficient for side extrusion.
  vert_norm[worldid, i0] = nrm
  vert_norm[worldid, i1] = nrm
  vert_norm[worldid, i2] = nrm

  # Top/bottom faces: match `_make_face_2d_elements` indexing exactly.
  # Two faces (top & bottom) => 6 vertices per element.
  world_vertex_offset = worldid * num_face_vertices

  # First face (top): i0, i1, i2
  face_local_top = 2 * elemid
  base0 = world_vertex_offset + face_local_top * 3

  face_points[base0 + 0] = v0 + offset
  face_points[base0 + 1] = v1 + offset
  face_points[base0 + 2] = v2 + offset

  # Second face (bottom): i0, i2, i1 (opposite winding)
  face_local_bottom = 2 * elemid + 1
  base1 = world_vertex_offset + face_local_bottom * 3

  face_points[base1 + 0] = v0 - offset
  face_points[base1 + 1] = v2 - offset
  face_points[base1 + 2] = v1 - offset

@wp.kernel
def compute_flex_shell_points(
  vert_xpos: wp.array2d(dtype=wp.vec3),
  vert_norm: wp.array2d(dtype=wp.vec3),
  face_points: wp.array(dtype=wp.vec3),
  shell_pairs: wp.array(dtype=int),
  shell_count: int,
  radius: float,
  face_offset: int,
  num_face_vertices: int,
):
  worldid, shellid = wp.tid()

  i0 = shell_pairs[2 * shellid + 0]
  i1 = shell_pairs[2 * shellid + 1]

  # Per-world vertex offset and local face indices for this shell fragment.
  world_vertex_offset = worldid * num_face_vertices
  face_local0 = face_offset + (2 * shellid)
  face_local1 = face_offset + (2 * shellid + 1)

  # ---- First side: (i0, i1) with +radius ----
  base0 = world_vertex_offset + face_local0 * 3
  # k = 0, ind = i0, sign = +1
  pos = vert_xpos[worldid, i0]
  nrm = vert_norm[worldid, i0]
  p = pos + nrm * (radius * 1.0)
  face_points[base0 + 0] = p
  # k = 1, ind = i1, sign = -1
  pos = vert_xpos[worldid, i1]
  nrm = vert_norm[worldid, i1]
  p = pos + nrm * (radius * -1.0)
  face_points[base0 + 1] = p
  # k = 2, ind = i1, sign = +1
  pos = vert_xpos[worldid, i1]
  nrm = vert_norm[worldid, i1]
  p = pos + nrm * (radius * 1.0)
  face_points[base0 + 2] = p

  # ---- Second side: (i1, i0) with -radius ----
  base1 = world_vertex_offset + face_local1 * 3
  neg_radius = -radius
  # k = 0, ind = i1, sign = +1
  pos = vert_xpos[worldid, i1]
  nrm = vert_norm[worldid, i1]
  p = pos + nrm * (neg_radius * 1.0)
  face_points[base1 + 0] = p
  # k = 1, ind = i0, sign = -1
  pos = vert_xpos[worldid, i0]
  nrm = vert_norm[worldid, i0]
  p = pos + nrm * (neg_radius * -1.0)
  face_points[base1 + 1] = p
  # k = 2, ind = i0, sign = +1
  pos = vert_xpos[worldid, i0]
  nrm = vert_norm[worldid, i0]
  p = pos + nrm * (neg_radius * 1.0)
  face_points[base1 + 2] = p


@wp.kernel
def compute_flex_3d_shell_faces(
  vert_xpos: wp.array2d(dtype=wp.vec3),
  face_points: wp.array(dtype=wp.vec3),
  shell_tris: wp.array(dtype=int),
  shell_count: int,
  face_offset: int,
  num_face_vertices: int,
):
  """Update 3D flex shell triangle faces for all worlds.

  Layout matches `_make_faces_3d_shells` in `render_context.py`:

    world_vertex_offset = world_id * num_face_vertices
    local_face         = shell_id
    base               = world_vertex_offset + local_face * 3

  Each shell fragment contributes a single triangle.
  """
  worldid, shellid = wp.tid()

  if worldid >= vert_xpos.shape[0] or shellid >= shell_count:
    return

  base_idx = shellid * 3
  i0 = shell_tris[base_idx + 0]
  i1 = shell_tris[base_idx + 1]
  i2 = shell_tris[base_idx + 2]

  nvert = vert_xpos.shape[1]
  if i0 < 0 or i0 >= nvert or i1 < 0 or i1 >= nvert or i2 < 0 or i2 >= nvert:
    return

  world_vertex_offset = worldid * num_face_vertices
  face_local = face_offset + shellid
  base = world_vertex_offset + face_local * 3

  v0 = vert_xpos[worldid, i0]
  v1 = vert_xpos[worldid, i1]
  v2 = vert_xpos[worldid, i2]

  face_points[base + 0] = v0
  face_points[base + 1] = v1
  face_points[base + 2] = v2


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


def refit_flex_bvh(m: Model, d: Data, rc: RenderContext):

  # Refit faces for 2D and 3D flex simultaneously, using the same vertex data.
  # Layout must match `_make_flex_mesh` in `render_context.py`.

  # Total faces used when building the mesh.
  nfaces = 2 * rc.flex_elem_count + 2 * rc.flex_shell_pairs_count + rc.flex_shell_tris_count
  if nfaces == 0:
    return
  num_face_vertices = nfaces * 3

  flexvert_norm = wp.zeros(d.flexvert_xpos.shape, dtype=wp.vec3)

  # 2D top/bottom faces.
  if rc.flex_elem_count > 0:
    wp.launch(
      kernel=compute_flex_points,
      dim=(d.nworld, rc.flex_elem_count),
      inputs=[
        d.flexvert_xpos,
        rc.flex_elem,
        flexvert_norm,
        rc.flex_face_points,
        rc.flex_elem_count,
        rc.flex_radius,
        num_face_vertices,
      ],
    )

  # 2D side faces from shell edge pairs.
  if rc.flex_shell_pairs_count > 0:
    wp.launch(
      kernel=compute_flex_shell_points,
      dim=(d.nworld, rc.flex_shell_pairs_count),
      inputs=[
        d.flexvert_xpos,
        flexvert_norm,
        rc.flex_face_points,
        rc.flex_shell_pairs,
        rc.flex_shell_pairs_count,
        rc.flex_radius,
        rc.flex_side_face_offset,
        num_face_vertices,
      ],
    )

  # 3D shell surface faces.
  if rc.flex_shell_tris_count > 0:
    wp.launch(
      kernel=compute_flex_3d_shell_faces,
      dim=(d.nworld, rc.flex_shell_tris_count),
      inputs=[
        d.flexvert_xpos,
        rc.flex_face_points,
        rc.flex_shell_tris,
        rc.flex_shell_tris_count,
        rc.flex_tris_face_offset,
        num_face_vertices,
      ],
    )

  rc.flex_registry[rc.flex_bvh_id].refit()

