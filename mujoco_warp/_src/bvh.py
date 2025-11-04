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

from .types import GeomType
from .types import Model
from .types import Data
from .render_context import RenderContext

wp.set_module_options({"enable_backward": False})


@wp.func
def compute_box_bounds(
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
def compute_sphere_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  radius = size[0]
  return pos - wp.vec3(radius, radius, radius), pos + wp.vec3(radius, radius, radius)


@wp.func
def compute_capsule_bounds(
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
def compute_plane_bounds(
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
def compute_ellipsoid_bounds(
  pos: wp.vec3,
  rot: wp.mat33,
  size: wp.vec3,
) -> Tuple[wp.vec3, wp.vec3]:
  # TODO: Implement ellipsoid bounds computation
  size_scale = 1.0
  return pos - wp.vec3(size_scale, size_scale, size_scale), pos + wp.vec3(size_scale, size_scale, size_scale)


@wp.kernel
def compute_bvh_bounds(
  bvh_ngeom: int,
  nworld: int,
  enabled_geom_ids: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_pos: wp.array2d(dtype=wp.vec3),
  geom_rot: wp.array2d(dtype=wp.mat33),
  mesh_bounds_size: wp.array(dtype=wp.vec3),
  lowers: wp.array(dtype=wp.vec3),
  uppers: wp.array(dtype=wp.vec3),
  groups: wp.array(dtype=wp.int32),
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
    lower, upper = compute_sphere_bounds(pos, rot, size)
  elif type == GeomType.CAPSULE:
    lower, upper = compute_capsule_bounds(pos, rot, size)
  elif type == GeomType.PLANE:
    lower, upper = compute_plane_bounds(pos, rot, size)
  elif type == GeomType.MESH:
    size = mesh_bounds_size[geom_dataid[geom_id]]
    lower, upper = compute_box_bounds(pos, rot, size)
  elif type == GeomType.ELLIPSOID:
    lower, upper = compute_ellipsoid_bounds(pos, rot, size)
  elif type == GeomType.BOX:
    lower, upper = compute_box_bounds(pos, rot, size)

  lowers[world_id * bvh_ngeom + bvh_geom_local] = lower
  uppers[world_id * bvh_ngeom + bvh_geom_local] = upper
  groups[world_id * bvh_ngeom + bvh_geom_local] = world_id


@wp.kernel
def compute_bvh_group_roots(
  bvh_id: wp.uint64,
  group_roots: wp.array(dtype=wp.int32),
):
  tid = wp.tid()
  root = wp.bvh_get_group_root(bvh_id, tid)
  group_roots[tid] = root


def build_warp_bvh(m: Model, d: Data, rc: RenderContext):
  """Build a Warp BVH for all geometries in all worlds."""

  wp.launch(
    kernel=compute_bvh_bounds,
    dim=(rc.nworld * rc.bvh_ngeom),
    inputs=[
      rc.bvh_ngeom,
      rc.nworld,
      rc.enabled_geom_ids,
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      d.geom_xpos,
      d.geom_xmat,
      rc.mesh_bounds_size,
      rc.lowers,
      rc.uppers,
      rc.groups,
    ],
  )

  bvh = wp.Bvh(
    rc.lowers,
    rc.uppers,
    groups=rc.groups,
  )

  # BVH handle must be stored to avoid garbage collection
  rc.bvh = bvh
  rc.bvh_id = bvh.id

  wp.launch(
    kernel=compute_bvh_group_roots,
    dim=rc.nworld,
    inputs=[bvh.id, rc.group_roots],
  )


def refit_warp_bvh(m: Model, d: Data, rc: RenderContext):
  wp.launch(
    kernel=compute_bvh_bounds,
    dim=(rc.nworld * rc.bvh_ngeom),
    inputs=[
      rc.bvh_ngeom,
      rc.nworld,
      rc.enabled_geom_ids,
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      d.geom_xpos,
      d.geom_xmat,
      rc.mesh_bounds_size,
      rc.lowers,
      rc.uppers,
      rc.groups,
    ],
  )

  rc.bvh.refit()
