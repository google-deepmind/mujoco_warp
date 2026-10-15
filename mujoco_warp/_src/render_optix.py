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

- each mesh and heightfield is one triangle GAS, and each analytic geom one
  custom-primitive GAS whose intersection program reuses `ray_primitive`;
- one IAS holds an instance per (world, geom), rewritten on the GPU each step;
- worlds are packed into a grid of cells `3 * extent` apart, and rays are clipped
  to their world's cell, so a single traversal never sees another world;
- planes are unbounded and are intersected analytically after the traversal.

Bounded geometry must stay within `extent` of the origin. Flexes and splats are not supported.
"""

import dataclasses
from typing import Any, Tuple

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src.ray import geom_color
from mujoco_warp._src.ray import make_ray_primitive
from mujoco_warp._src.ray import ray_plane
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model

try:
  import warp_optix as wo
except ImportError:  # optional: warp_optix from https://github.com/nvtw/otk-pyoptix
  wo = None

wp.set_module_options({"enable_backward": False})

_MESH_TYPES = (int(GeomType.MESH), int(GeomType.SDF), int(GeomType.HFIELD))
_TRIANGLE_SBT_OFFSET = 0
_PRIMITIVE_SBT_OFFSET = 1


@dataclasses.dataclass
class OptixScene:
  """OptiX acceleration structures of all worlds and the compiled render pipeline.

  Attributes:
    ctx: OptiX device context
    gas: per-geometry acceleration structures, kept alive for the IAS
    mesh_traversable: triangle GAS handle per mesh
    hfield_traversable: triangle GAS handle per heightfield
    primitive_traversable: custom-primitive GAS handle per geom (0 for meshes)
    instances: OptiX instance records (nworld, nenabled geom)
    ias: instance acceleration structure over all worlds
    extent: half-size of the region each world's bounded geometry and rays occupy
    planes: enabled plane geom ids, which cast_ray intersects without the IAS
    grid: worlds per axis of the cubic world grid
    write_instances: kernel writing `instances` from the current geom poses
    pipeline: render pipeline, compiled with the render context
  """

  ctx: Any
  gas: list
  mesh_traversable: wp.array
  hfield_traversable: wp.array
  primitive_traversable: wp.array
  instances: wp.array2d
  ias: Any
  extent: float
  planes: tuple
  grid: int
  write_instances: wp.Kernel
  pipeline: Any = None


@wp.func
def _world_offset(worldid: int, grid: int, spacing: float) -> wp.vec3:
  center = float(grid - 1) * 0.5
  cell = wp.vec3(float(worldid % grid), float((worldid // grid) % grid), float(worldid // (grid * grid)))
  return (cell - wp.vec3(center)) * spacing


def create_scene(mjm: mujoco.MjModel, mjd: mujoco.MjData, rc, nworld: int, extent: float | None = None) -> OptixScene:
  """Build OptiX acceleration structures for every world of a render context."""
  if wo is None:
    raise ImportError("backend='optix' requires warp_optix (https://github.com/nvtw/otk-pyoptix) and warp-lang>=1.17")
  if rc.bvh_nflexgeom or rc.splat_count:
    raise NotImplementedError("the OptiX backend does not support flexes or splats yet")

  extent = float(extent or max(4.0 * mjm.stat.extent, 1.0))
  grid = max(1, int(np.ceil(nworld ** (1.0 / 3.0))))
  if nworld * mjm.ngeom >= 1 << 24:
    raise ValueError("OptiX instance ids hold 24 bits: nworld * ngeom must be below 2**24")

  ctx = wo.create_context()
  gas = []

  def triangle_traversables(bvh_ids: wp.array, registry: dict) -> wp.array:
    handles = []
    for bvh_id in bvh_ids.numpy():
      mesh = registry.get(int(bvh_id))
      if mesh is None:
        handles.append(0)
        continue
      gas.append(wo.build_accel(ctx, [wo.triangles(mesh.points, mesh.indices)], compact=True))
      handles.append(gas[-1].handle)
    return wp.array(handles, dtype=wp.uint64)

  mesh_traversable = triangle_traversables(rc.mesh_bvh_id, rc.mesh_registry)
  hfield_traversable = triangle_traversables(rc.hfield_bvh_id, rc.hfield_registry)

  # Bounded analytic geoms are custom primitives in their object frame.
  primitive_handles = np.zeros(mjm.ngeom, dtype=np.uint64)
  for gi in rc.enabled_geom_ids.numpy():
    gtype, size = mjm.geom_type[gi], mjm.geom_size[gi]
    if gtype in _MESH_TYPES or gtype == GeomType.PLANE:
      continue
    if gtype == GeomType.SPHERE:
      half = np.full(3, size[0])
    elif gtype == GeomType.CAPSULE:
      half = np.array([size[0], size[0], size[1] + size[0]])
    elif gtype == GeomType.CYLINDER:
      half = np.array([size[0], size[0], size[1]])
    else:  # ellipsoid, box
      half = size
    bounds = wp.array(np.concatenate([-half, half])[None] * 1.001, dtype=float)
    gas.append(wo.build_accel(ctx, [wo.aabbs(bounds)], compact=True))
    primitive_handles[gi] = gas[-1].handle
  primitive_traversable = wp.array(primitive_handles, dtype=wp.uint64)

  spacing = 3.0 * extent

  # Writes one instance per (world, enabled geom) at the geom's current pose.
  @wp.kernel(module="unique", enable_backward=False)
  def write_instances(
    # Model:
    geom_type: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_matid: wp.array2d[int],
    geom_rgba: wp.array2d[wp.vec4],
    mat_rgba: wp.array2d[wp.vec4],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    # In:
    enabled_geom_ids: wp.array[int],
    mesh_traversable: wp.array[wp.uint64],
    hfield_traversable: wp.array[wp.uint64],
    primitive_traversable: wp.array[wp.uint64],
    # Out:
    instances_out: wp.array2d[wo.Instance],
  ):
    worldid, i = wp.tid()
    gi = enabled_geom_ids[i]
    gtype = geom_type[gi]
    traversable = primitive_traversable[gi]
    sbt_offset = _PRIMITIVE_SBT_OFFSET
    if gtype == GeomType.HFIELD:
      traversable = hfield_traversable[geom_dataid[worldid % geom_dataid.shape[0], gi]]
      sbt_offset = _TRIANGLE_SBT_OFFSET
    elif gtype == GeomType.MESH or gtype == GeomType.SDF:
      traversable = mesh_traversable[geom_dataid[worldid % geom_dataid.shape[0], gi]]
      sbt_offset = _TRIANGLE_SBT_OFFSET

    # Fully transparent geoms are invisible to rays, as in the BVH renderer. Planes are
    # intersected analytically by cast_ray and have no acceleration structure.
    rgba, _ = geom_color(geom_matid, geom_rgba, mat_rgba, worldid, gi)
    mask = wp.where(rgba[3] == 0.0 or gtype == GeomType.PLANE, wp.uint32(0), wp.uint32(0xFF))

    instances_out[worldid, i] = wo.make_instance(
      geom_xmat_in[worldid, gi],
      geom_xpos_in[worldid, gi] + _world_offset(worldid, wp.static(grid), wp.static(spacing)),
      traversable,
      wp.uint32(worldid * geom_type.shape[0] + gi),
      wp.uint32(sbt_offset),
      mask,
      wp.uint32(wo.constants.OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT),
    )

  instances = wp.empty((nworld, rc.bvh_ngeom), dtype=wo.Instance)
  wp.launch(
    write_instances,
    dim=instances.shape,
    inputs=[
      wp.array(mjm.geom_type, dtype=int),
      wp.array(mjm.geom_dataid[None], dtype=int),
      wp.array(mjm.geom_matid[None], dtype=int),
      wp.array(mjm.geom_rgba[None], dtype=wp.vec4),
      wp.array(mjm.mat_rgba[None], dtype=wp.vec4),
      wp.array(np.tile(mjd.geom_xpos[None], (nworld, 1, 1)), dtype=wp.vec3),
      wp.array(np.tile(mjd.geom_xmat.reshape(1, -1, 3, 3), (nworld, 1, 1, 1)), dtype=wp.mat33),
      rc.enabled_geom_ids,
      mesh_traversable,
      hfield_traversable,
      primitive_traversable,
    ],
    outputs=[instances],
  )
  ias = wo.build_accel(ctx, [wo.instances(instances)], flags=wo.constants.OPTIX_BUILD_FLAG_ALLOW_UPDATE)
  # OptiX tracks the setup builds until it sees them complete. Refit once and wait here so that
  # a later refit inside CUDA graph capture has nothing outstanding to wait on.
  ias = wo.update_accel(ctx, ias)
  wp.synchronize_device()
  rc.bvh_id = ias.handle

  return OptixScene(
    ctx=ctx,
    gas=gas,
    mesh_traversable=mesh_traversable,
    hfield_traversable=hfield_traversable,
    primitive_traversable=primitive_traversable,
    instances=instances,
    ias=ias,
    extent=extent,
    planes=tuple(int(gi) for gi in rc.enabled_geom_ids.numpy() if mjm.geom_type[gi] == GeomType.PLANE),
    grid=grid,
    write_instances=write_instances,
  )


def refit(m: Model, d: Data, rc):
  """Move every instance to its geom's current pose and refit the scene in place."""
  scene = rc.optix
  wp.launch(
    scene.write_instances,
    dim=scene.instances.shape,
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_matid,
      m.geom_rgba,
      m.mat_rgba,
      d.geom_xpos,
      d.geom_xmat,
      rc.enabled_geom_ids,
      scene.mesh_traversable,
      scene.hfield_traversable,
      scene.primitive_traversable,
    ],
    outputs=[scene.instances],
  )
  scene.ias = wo.update_accel(scene.ctx, scene.ias)
  rc.bvh_id = scene.ias.handle


def make_cast_ray(geom_ray_types: Tuple[int], scene: OptixScene, first_hit: bool = False) -> wp.Function:
  """Build an OptiX ray-cast func with the signature and results of the BVH `cast_ray`."""
  grid, extent = scene.grid, scene.extent
  spacing = 3.0 * extent
  planes = scene.planes
  # Shadow rays stop at the first hit; mesh shadows are two-sided, like the BVH any-hit cast.
  flags = (
    wo.constants.OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | wo.constants.OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT
    if first_hit
    else wo.constants.OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES
  )

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
    # Clip the ray where it leaves its world's cell.
    tmax = max_dist
    for axis in range(3):
      if ray_dir_world[axis] != 0.0:
        bound = wp.where(ray_dir_world[axis] > 0.0, wp.static(extent), -wp.static(extent))
        tmax = wp.min(tmax, (bound - ray_origin_world[axis]) / ray_dir_world[axis])

    ray_flags = wp.uint32(wp.static(flags))
    if wp.static(not first_hit):
      if not cull_backfaces:
        ray_flags = wp.uint32(0)
    origin = ray_origin_world + _world_offset(worldid, wp.static(grid), wp.static(spacing))
    hit = wo.closest_hit(bvh_id, origin, ray_dir_world, 0.0, tmax, wp.uint32(0xFF), ray_flags)
    geom_id = int(-1)
    dist = max_dist
    normal = wp.vec3(0.0)
    u = float(0.0)
    v = float(0.0)
    face = int(-1)
    mesh_id = int(-1)
    if hit.t != wo.MISS:
      geom_id = int(hit.instance_id) - worldid * geom_type.shape[0]
      dist = hit.t
      gtype = geom_type[geom_id]
      if gtype == GeomType.MESH or gtype == GeomType.SDF or gtype == GeomType.HFIELD:
        face = int(hit.primitive_index)
        dataid = geom_dataid[worldid % geom_dataid.shape[0], geom_id]
        mesh_id = dataid
        if gtype == GeomType.HFIELD:
          mesh = hfield_bvh_id[dataid]
          mesh_id = -1
        else:
          mesh = mesh_bvh_id[dataid]
        normal = wp.normalize(geom_xmat_in[worldid, geom_id] @ wp.mesh_eval_face_normal(mesh, face))
        bary = wp.optix_hit_object_get_triangle_barycentrics()
        u = 1.0 - bary[0] - bary[1]
        v = bary[0]
      else:
        normal = wp.vec3(
          wp.uint32_to_float(wp.optix_hit_object_get_attribute_0()),
          wp.uint32_to_float(wp.optix_hit_object_get_attribute_1()),
          wp.uint32_to_float(wp.optix_hit_object_get_attribute_2()),
        )

    # Planes are unbounded, so they cannot live in a world cell; intersect them directly.
    for i in range(wp.static(len(planes))):
      gi = wp.static(planes[i])
      rgba, _ = geom_color(geom_matid, geom_rgba, mat_rgba, worldid, gi)
      d, n = ray_plane(
        geom_xpos_in[worldid, gi],
        geom_xmat_in[worldid, gi],
        geom_size[worldid % geom_size.shape[0], gi],
        ray_origin_world,
        ray_dir_world,
      )
      culled = cull_backfaces and wp.dot(ray_dir_world, n) > 0.0
      if rgba[3] > 0.0 and d >= 0.0 and d < dist and not culled:
        geom_id, dist, normal, u, v, face, mesh_id = gi, d, n, 0.0, 0.0, -1, -1

    return geom_id, dist, normal, u, v, face, mesh_id

  return cast_ray


def create_pipeline(scene: OptixScene, megakernel: wp.Kernel, geom_ray_types: Tuple[int], cull_backfaces: bool):
  """Compile the render megakernel into an OptiX pipeline over `scene`."""
  ray_primitive = make_ray_primitive(geom_ray_types)
  grid, spacing = scene.grid, 3.0 * scene.extent

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
    worldid = int(wp.optix_get_instance_id()) // ngeom
    gi = int(wp.optix_get_instance_id()) - worldid * ngeom
    pnt = wp.optix_get_world_ray_origin() - _world_offset(worldid, wp.static(grid), wp.static(spacing))
    vec = wp.optix_get_world_ray_direction()
    d, n = ray_primitive(
      geom_type[gi], geom_xpos_in[worldid, gi], geom_xmat_in[worldid, gi], geom_size[worldid % geom_size.shape[0], gi], pnt, vec
    )
    if wp.static(cull_backfaces):
      if wp.dot(vec, n) > 0.0:
        return
    if d >= wp.optix_get_ray_tmin():
      wp.optix_report_intersection(
        d, wp.uint32(0), wp.float_to_uint32(n[0]), wp.float_to_uint32(n[1]), wp.float_to_uint32(n[2])
      )

  return wo.create_pipeline(
    scene.ctx,
    megakernel,
    hit_groups=[wo.HitGroup(), wo.HitGroup(intersection=intersect_primitive)],
    attribute_words=3,
    graph_flags=wo.constants.OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
  )


def launch(rc, dim: Tuple[int, int], inputs: list):
  """Launch the compiled render pipeline like `wp.launch(megakernel, dim, inputs)`."""
  wo.launch(rc.optix.pipeline, dim, inputs)
