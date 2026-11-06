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
"""OptiX ray casting for the batch renderer.

The renderer's megakernel runs unchanged as an OptiX raygen program; only its
ray-cast function is swapped for one that traverses an OptiX scene:

- each mesh and heightfield is one triangle acceleration structure, and analytic
  geoms are instances of one unit-cube custom primitive, scaled to the geom's
  bounds, whose intersection program reuses `ray_primitive`;
- one instance acceleration structure holds an instance per (world, geom),
  rewritten on the GPU each step;
- worlds are packed into a grid of cells `3 * extent` apart, and rays are clipped
  to their world's cell, so a single traversal never sees another world;
- planes are unbounded and are intersected analytically after the traversal.

Bounded geometry must stay within `extent` of the origin. Flexes and splats are not supported.

warp_optix is imported only when an OptiX scene is created, so this backend has no
import-time cost when unused.
"""

import functools
from typing import Tuple

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src.ray import geom_color
from mujoco_warp._src.ray import make_ray_primitive
from mujoco_warp._src.ray import ray_plane
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import OptixScene

wp.set_module_options({"enable_backward": False})

_MESH_TYPES = (int(GeomType.MESH), int(GeomType.SDF), int(GeomType.HFIELD))
_TRIANGLE_SBT_OFFSET = 0
_PRIMITIVE_SBT_OFFSET = 1
# Unit-cube bounds of the custom primitive, padded so rays grazing a geom's bounds still test it.
_PRIMITIVE_BOUND = 1.001


@functools.cache
def _get_wo():
  try:
    import warp_optix as wo  # noqa: PLC0415
  except ImportError as e:
    raise ImportError(
      "backend='optix' requires warp_optix (https://github.com/NVIDIA/warp-optix), installed with mujoco-warp[optix]"
    ) from e
  return wo


@wp.func
def _world_offset(worldid: int, grid: int, spacing: float) -> wp.vec3:
  center = float(grid - 1) * 0.5
  cell = wp.vec3(float(worldid % grid), float((worldid // grid) % grid), float(worldid // (grid * grid)))
  return (cell - wp.vec3(center)) * spacing


def create_scene(mjm: mujoco.MjModel, mjd: mujoco.MjData, rc, nworld: int, extent: float | None = None) -> OptixScene:
  """Build OptiX acceleration structures for every world of a render context."""
  wo = _get_wo()
  if rc.bvh_nflexgeom or rc.splat_count:
    raise NotImplementedError("the OptiX backend does not support flexes or splats yet")

  extent = float(extent or max(4.0 * mjm.stat.extent, 1.0))
  grid = max(1, int(np.ceil(nworld ** (1.0 / 3.0))))
  if nworld * mjm.ngeom >= 1 << 24:
    raise ValueError("OptiX instance ids hold 24 bits: nworld * ngeom must be below 2**24")

  accels = []

  def triangle_traversables(bvh_ids: wp.array, registry: dict) -> wp.array:
    ids = []
    for bvh_id in bvh_ids.numpy():
      mesh = registry.get(int(bvh_id))
      if mesh is None:
        ids.append(0)
        continue
      accels.append(wo.Accel([wo.triangles(mesh.points, mesh.indices)], compact=True))
      ids.append(accels[-1].id)
    return wp.array(ids, dtype=wp.uint64)

  mesh_traversable = triangle_traversables(rc.mesh_bvh_id, rc.mesh_registry)
  hfield_traversable = triangle_traversables(rc.hfield_bvh_id, rc.hfield_registry)

  # Bounded analytic geoms share one unit-cube custom primitive; write_instances scales each
  # instance to the geom's bounds, so per-world geom sizes need no rebuild.
  enabled_types = mjm.geom_type[rc.enabled_geom_ids.numpy()]
  primitive_traversable = 0
  if any(t not in _MESH_TYPES and t != GeomType.PLANE for t in enabled_types):
    bounds = wp.array([[-_PRIMITIVE_BOUND] * 3 + [_PRIMITIVE_BOUND] * 3], dtype=float)
    accels.append(wo.Accel([wo.aabbs(bounds=bounds)], compact=True))
    primitive_traversable = accels[-1].id

  spacing = 3.0 * extent

  # Writes one instance per (world, enabled geom) at the geom's current pose.
  @wp.kernel(module="unique", enable_backward=False)
  def write_instances(
    # Model:
    geom_type: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_matid: wp.array2d[int],
    geom_aabb: wp.array3d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    mat_rgba: wp.array2d[wp.vec4],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    # In:
    enabled_geom_ids: wp.array[int],
    mesh_traversable: wp.array[wp.uint64],
    hfield_traversable: wp.array[wp.uint64],
    primitive_traversable: wp.uint64,
    # Out:
    instances_out: wp.array2d[wo.Instance],
  ):
    worldid, i = wp.tid()
    gi = enabled_geom_ids[i]
    gtype = geom_type[gi]
    linear = geom_xmat_in[worldid, gi]
    traversable = primitive_traversable
    sbt_offset = _PRIMITIVE_SBT_OFFSET
    if gtype == GeomType.HFIELD:
      traversable = hfield_traversable[geom_dataid[worldid % geom_dataid.shape[0], gi]]
      sbt_offset = _TRIANGLE_SBT_OFFSET
    elif gtype == GeomType.MESH or gtype == GeomType.SDF:
      traversable = mesh_traversable[geom_dataid[worldid % geom_dataid.shape[0], gi]]
      sbt_offset = _TRIANGLE_SBT_OFFSET
    else:
      linear = linear @ wp.diag(geom_aabb[worldid % geom_aabb.shape[0], gi, 1])

    # Fully transparent geoms are invisible to rays, as in the BVH renderer. Planes are
    # intersected analytically by cast_ray and have no acceleration structure.
    rgba, _ = geom_color(geom_matid, geom_rgba, mat_rgba, worldid, gi)
    mask = wp.where(rgba[3] == 0.0 or gtype == GeomType.PLANE, wp.uint32(0), wp.uint32(0xFF))

    instances_out[worldid, i] = wo.make_instance(
      linear,
      geom_xpos_in[worldid, gi] + _world_offset(worldid, wp.static(grid), wp.static(spacing)),
      traversable,
      wp.uint32(worldid * geom_type.shape[0] + gi),
      wp.uint32(sbt_offset),
      mask,
      wp.uint32(wp.static(int(wo.InstanceFlags.DISABLE_ANYHIT))),
    )

  scene = OptixScene(
    accels=accels,
    mesh_traversable=mesh_traversable,
    hfield_traversable=hfield_traversable,
    primitive_traversable=primitive_traversable,
    instances=wp.empty((nworld, rc.bvh_ngeom), dtype=wo.Instance),
    ias=None,
    extent=extent,
    planes=tuple(int(gi) for gi in rc.enabled_geom_ids.numpy() if mjm.geom_type[gi] == GeomType.PLANE),
    grid=grid,
    write_instances=write_instances,
  )
  _update_instances(
    scene,
    rc,
    wp.array(mjm.geom_type, dtype=int),
    wp.array(mjm.geom_dataid[None], dtype=int),
    wp.array(mjm.geom_matid[None], dtype=int),
    wp.array(mjm.geom_aabb.reshape(1, -1, 2, 3), dtype=wp.vec3),
    wp.array(mjm.geom_rgba[None], dtype=wp.vec4),
    wp.array(mjm.mat_rgba[None], dtype=wp.vec4),
    wp.array(np.tile(mjd.geom_xpos[None], (nworld, 1, 1)), dtype=wp.vec3),
    wp.array(np.tile(mjd.geom_xmat.reshape(1, -1, 3, 3), (nworld, 1, 1, 1)), dtype=wp.mat33),
  )
  return scene


def _update_instances(
  scene: OptixScene,
  rc,
  geom_type: wp.array,
  geom_dataid: wp.array,
  geom_matid: wp.array,
  geom_aabb: wp.array,
  geom_rgba: wp.array,
  mat_rgba: wp.array,
  geom_xpos: wp.array,
  geom_xmat: wp.array,
):
  """Write every instance at its geom's current pose, then build or refit the IAS."""
  wo = _get_wo()
  wp.launch(
    scene.write_instances,
    dim=scene.instances.shape,
    inputs=[
      geom_type,
      geom_dataid,
      geom_matid,
      geom_aabb,
      geom_rgba,
      mat_rgba,
      geom_xpos,
      geom_xmat,
      rc.enabled_geom_ids,
      scene.mesh_traversable,
      scene.hfield_traversable,
      wp.uint64(scene.primitive_traversable),
    ],
    outputs=[scene.instances],
  )
  if scene.ias is None:
    scene.ias = wo.Accel([wo.instances(scene.instances)], flags=wo.BuildFlags.ALLOW_UPDATE | wo.BuildFlags.PREFER_FAST_TRACE)
  else:
    scene.ias.refit()
  rc.bvh_id = scene.ias.id


def refit(m: Model, d: Data, rc):
  """Move every instance to its geom's current pose and refit the scene in place."""
  _update_instances(
    rc.optix, rc, m.geom_type, m.geom_dataid, m.geom_matid, m.geom_aabb, m.geom_rgba, m.mat_rgba, d.geom_xpos, d.geom_xmat
  )


def make_cast_ray(geom_ray_types: Tuple[int], scene: OptixScene, first_hit: bool = False) -> wp.Function:
  """Build an OptiX ray-cast func with the signature and results of the BVH `cast_ray`."""
  wo = _get_wo()
  grid, extent = scene.grid, scene.extent
  spacing = 3.0 * extent
  planes = scene.planes
  has_triangles = any(t in geom_ray_types for t in _MESH_TYPES)
  has_hfield = int(GeomType.HFIELD) in geom_ray_types
  has_mesh = int(GeomType.MESH) in geom_ray_types or int(GeomType.SDF) in geom_ray_types
  cull_flags = int(wo.RayFlags.CULL_BACK_FACING_TRIANGLES)

  @wp.func
  def cast_ray(
    # Model:
    geom_type: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_matid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    flex_vertadr: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_radius: wp.array[float],
    mat_rgba: wp.array2d[wp.vec4],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    # In:
    bvh_id: wp.uint64,
    group_root: int,
    worldid: int,
    bvh_ngeom: int,
    flex_bvh_ngeom: int,
    enabled_geom_ids: wp.array[int],
    mesh_bvh_id: wp.array[wp.uint64],
    hfield_bvh_id: wp.array[wp.uint64],
    flex_rgba: wp.array[wp.vec4],
    flex_geom_flexid: wp.array[int],
    flex_geom_edgeid: wp.array[int],
    flex_bvh_id: wp.array[wp.uint64],
    flex_group_root: wp.array2d[int],
    ray_origin_world: wp.vec3,
    ray_dir_world: wp.vec3,
    max_dist: float,
    cull_backfaces: bool,
  ) -> Tuple[int, float, wp.vec3, float, float, int, int]:
    # Clip the ray where it leaves its world's cell; an origin outside the cell gets an empty range.
    tmax = max_dist
    for axis in range(3):
      if ray_dir_world[axis] != 0.0:
        bound = wp.where(ray_dir_world[axis] > 0.0, wp.static(extent), -wp.static(extent))
        tmax = wp.min(tmax, (bound - ray_origin_world[axis]) / ray_dir_world[axis])
    tmax = wp.max(0.0, tmax)

    origin = ray_origin_world + _world_offset(worldid, wp.static(grid), wp.static(spacing))
    geom_id = int(-1)
    dist = max_dist
    normal = wp.vec3(0.0)
    u = float(0.0)
    v = float(0.0)
    face = int(-1)
    mesh_id = int(-1)

    if wp.static(first_hit):
      # Shadow rays only test for an occluder, so they stop at the first hit and skip surface
      # data. Mesh shadows are two-sided, like the BVH any-hit cast.
      if wo.accel_query_ray_anyhit(bvh_id, origin, ray_dir_world, tmax, 0.0, 0xFF):
        return 0, 0.0, normal, u, v, face, mesh_id
    else:
      ray_flags = wp.uint32(0)
      if cull_backfaces:
        ray_flags = wp.uint32(wp.static(cull_flags))
      query = wo.accel_query_ray(bvh_id, origin, ray_dir_world, tmax, 0.0, 0xFF, ray_flags)
      if query.result:
        geom_id = int(query.instance_id) - worldid * geom_type.shape[0]
        dist = query.t
        # Custom primitives report their normal as three attributes; the query holds the first two.
        is_triangle = False
        if wp.static(has_triangles):
          gtype = geom_type[geom_id]
          is_triangle = gtype == GeomType.MESH or gtype == GeomType.SDF or gtype == GeomType.HFIELD
          if is_triangle:
            face = int(query.primitive_index)
            dataid = geom_dataid[worldid % geom_dataid.shape[0], geom_id]
            mesh = wp.uint64(0)
            if wp.static(has_hfield):
              if gtype == GeomType.HFIELD:
                mesh = hfield_bvh_id[dataid]
            if wp.static(has_mesh):
              if gtype != GeomType.HFIELD:
                mesh = mesh_bvh_id[dataid]
                mesh_id = dataid
            normal = wp.normalize(geom_xmat_in[worldid, geom_id] @ wp.mesh_eval_face_normal(mesh, face))
            u = 1.0 - query.u - query.v
            v = query.u
        if not is_triangle:
          normal = wp.vec3(query.u, query.v, wp.optix_uint32_as_float(wp.optix_hit_object_get_attribute(2)))

    # Planes are unbounded, so they cannot live in a world cell; intersect them directly.
    # ray_plane only reports front-face hits, so planes need no separate backface cull.
    for i in range(wp.static(len(planes))):
      gi = wp.static(planes[i])
      rgba, _ = geom_color(geom_matid, geom_rgba, mat_rgba, worldid, gi)
      if rgba[3] == 0.0:
        continue
      d, n = ray_plane(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
      if d >= 0.0 and d < dist:
        if wp.static(first_hit):
          return gi, d, n, 0.0, 0.0, -1, -1
        geom_id, dist, normal, u, v, face, mesh_id = gi, d, n, 0.0, 0.0, -1, -1

    return geom_id, dist, normal, u, v, face, mesh_id

  return cast_ray


def create_pipeline(scene: OptixScene, megakernel: wp.Kernel, geom_ray_types: Tuple[int], cull_backfaces: bool):
  """Compile the render megakernel into an OptiX pipeline over `scene`."""
  wo = _get_wo()
  ray_primitive = make_ray_primitive(geom_ray_types)
  grid, spacing = scene.grid, 3.0 * scene.extent

  # Binds the megakernel's arguments of the same names.
  @wp.kernel(module="unique", enable_backward=False)
  def intersect_primitive(
    # Model:
    geom_type: wp.array[int],
    geom_size: wp.array2d[wp.vec3],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
  ):
    ngeom = geom_type.shape[0]
    inst_id = int(wp.optix_get_instance_id())
    worldid = inst_id // ngeom
    gi = inst_id - worldid * ngeom
    pnt = wp.optix_get_world_ray_origin() - _world_offset(worldid, wp.static(grid), wp.static(spacing))
    vec = wp.optix_get_world_ray_direction()
    d, n = ray_primitive(
      geom_xpos_in[worldid, gi], geom_xmat_in[worldid, gi], geom_size[worldid % geom_size.shape[0], gi], pnt, vec, geom_type[gi]
    )
    if d < wp.optix_get_ray_tmin():
      return
    if wp.static(cull_backfaces):
      if wp.dot(vec, n) > 0.0:
        return
    wp.optix_report_intersection(
      d, wp.uint32(0), wp.optix_float_as_uint32(n[0]), wp.optix_float_as_uint32(n[1]), wp.optix_float_as_uint32(n[2])
    )

  return wo.Pipeline(
    megakernel,
    hit_groups=[wo.HitGroup(), wo.HitGroup(intersection=intersect_primitive)],
    attribute_words=3,
    graph_flags=wo.TraversableGraphFlags.ALLOW_SINGLE_LEVEL_INSTANCING,
  )


def launch(rc, dim: Tuple[int, int], inputs: list):
  """Launch the compiled render pipeline like `wp.launch(megakernel, dim, inputs)`."""
  _get_wo().launch(rc.optix.pipeline, dim, inputs)
