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

from typing import Optional, Tuple

import warp as wp

from mujoco_warp._src.ray import _ray_eliminate
from mujoco_warp._src.ray import ray_geom
from mujoco_warp._src.ray import ray_mesh_with_bvh
from mujoco_warp._src.render_context import RenderContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec6

wp.set_module_options({"enable_backward": False})


@wp.func
def _ray_geom_mesh_bvh(
  # Model:
  body_weldid: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_matid: wp.array2d(dtype=int),
  geom_group: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_rgba: wp.array2d(dtype=wp.vec4),
  mat_rgba: wp.array2d(dtype=wp.vec4),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  worldid: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: int,
  geomid: int,
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  min_dist: float,
) -> Tuple[float, wp.vec3]:
  if not _ray_eliminate(
    body_weldid,
    geom_bodyid,
    geom_matid[worldid % geom_matid.shape[0]],
    geom_group,
    geom_rgba[worldid % geom_rgba.shape[0]],
    mat_rgba[worldid % mat_rgba.shape[0]],
    geomid,
    geomgroup,
    flg_static,
    bodyexclude,
  ):
    pos = geom_xpos_in[worldid, geomid]
    mat = geom_xmat_in[worldid, geomid]
    gtype = geom_type[geomid]

    if gtype == GeomType.MESH or gtype == GeomType.HFIELD:
      bvh_ids = mesh_bvh_id if gtype == GeomType.MESH else hfield_bvh_id
      t, n, u, v, f, geom_mesh_id = ray_mesh_with_bvh(
        bvh_ids,
        geom_dataid[geomid],
        pos,
        mat,
        pnt,
        vec,
        min_dist,
      )
      if t >= 0.0 and t < min_dist:
        return t, n
    else:
      return ray_geom(
        pos,
        mat,
        geom_size[worldid % geom_size.shape[0], geomid],
        pnt,
        vec,
        gtype,
      )

  return -1.0, wp.vec3()


@wp.kernel
def _ray_bvh(
  # Model:
  ngeom: int,
  body_weldid: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_matid: wp.array2d(dtype=int),
  geom_group: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_rgba: wp.array2d(dtype=wp.vec4),
  mat_rgba: wp.array2d(dtype=wp.vec4),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  pnt: wp.array2d(dtype=wp.vec3),
  vec: wp.array2d(dtype=wp.vec3),
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: wp.array(dtype=int),
  bvh_id: wp.uint64,
  group_root: wp.array(dtype=int),
  enabled_geom_ids: wp.array(dtype=int),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  # Out:
  dist_out: wp.array2d(dtype=float),
  geomid_out: wp.array2d(dtype=int),
  normal_out: wp.array2d(dtype=wp.vec3),
):
  worldid, rayid = wp.tid()

  ray_origin = pnt[worldid, rayid]
  ray_dir = vec[worldid, rayid]
  body_exclude = bodyexclude[rayid]

  min_dist = float(wp.inf)
  min_geomid = int(-1)
  min_normal = wp.vec3()

  query = wp.bvh_query_ray(bvh_id, ray_origin, ray_dir, group_root[worldid])
  bounds_nr = int(0)

  while wp.bvh_query_next(query, bounds_nr, min_dist):
    bvh_local = bounds_nr - (worldid * ngeom)
    geomid = enabled_geom_ids[bvh_local]

    dist, normal = _ray_geom_mesh_bvh(
      body_weldid,
      geom_type,
      geom_bodyid,
      geom_dataid,
      geom_matid,
      geom_group,
      geom_size,
      geom_rgba,
      mat_rgba,
      geom_xpos_in,
      geom_xmat_in,
      worldid,
      pnt[worldid, rayid],
      vec[worldid, rayid],
      geomgroup,
      flg_static,
      body_exclude,
      geomid,
      mesh_bvh_id,
      hfield_bvh_id,
      min_dist,
    )

    if dist >= 0.0 and dist < min_dist:
      min_dist = dist
      min_geomid = geomid
      min_normal = normal

  if wp.isinf(min_dist):
    dist_out[worldid, rayid] = -1.0
  else:
    dist_out[worldid, rayid] = min_dist
  geomid_out[worldid, rayid] = min_geomid
  normal_out[worldid, rayid] = min_normal


def ray_bvh(
  m: Model,
  d: Data,
  rc: RenderContext,
  pnt: wp.array2d(dtype=wp.vec3),
  vec: wp.array2d(dtype=wp.vec3),
  geomgroup: Optional[vec6] = None,
  flg_static: bool = True,
  bodyexclude: int = -1,
) -> Tuple[wp.array, wp.array, wp.array]:
  """Returns the distance at which rays intersect with primitive geoms.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    rc: The render context from containing BVH context information (device).
    pnt: Ray origin points.
    vec: Ray directions.
    geomgroup: Group inclusion/exclusion mask. If all are wp.inf, ignore.
    flg_static: If True, allows rays to intersect with static geoms.
    bodyexclude: Ignore geoms on specified body id (-1 to disable).

  Returns:
    Distances from ray origins to geom surfaces, IDs of intersected geoms (-1 if none),
    and normals at intersection points.
  """
  assert pnt.shape[0] == 1
  assert pnt.shape[0] == vec.shape[0]

  if geomgroup is None:
    geomgroup = vec6(-1, -1, -1, -1, -1, -1)

  ray_bodyexclude = wp.empty(1, dtype=int)
  ray_bodyexclude.fill_(bodyexclude)
  ray_dist = wp.empty((d.nworld, 1), dtype=float)
  ray_geomid = wp.empty((d.nworld, 1), dtype=int)
  ray_normal = wp.empty((d.nworld, 1), dtype=wp.vec3)

  rays_bvh(m, d, rc, pnt, vec, geomgroup, flg_static, ray_bodyexclude, ray_dist, ray_geomid, ray_normal)

  return ray_dist, ray_geomid, ray_normal


def rays_bvh(
  m: Model,
  d: Data,
  rc: RenderContext,
  pnt: wp.array2d(dtype=wp.vec3),
  vec: wp.array2d(dtype=wp.vec3),
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: wp.array(dtype=int),
  dist: wp.array2d(dtype=float),
  geomid: wp.array2d(dtype=int),
  normal: wp.array2d(dtype=wp.vec3),
):
  """BVH accelerated ray intersection for multiple worlds and multiple rays.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    rc: The render context from containing BVH context information (device).
    pnt: Ray origin points, shape (nworld, nray).
    vec: Ray directions, shape (nworld, nray).
    geomgroup: Group inclusion/exclusion mask. Set all elements to -1 to ignore.
    flg_static: If True, allows rays to intersect with static geoms.
    bodyexclude: Per-ray body exclusion array of shape (nray,). Geoms on the
      specified body ids are ignored (-1 to disable for that ray).
    dist: Output array for distances from ray origins to geom surfaces, shape
      (nworld, nray). -1 indicates no intersection.
    geomid: Output array for IDs of intersected geoms, shape (nworld, nray). -1
      indicates no intersection.
    normal: Output array for normals at intersection points, shape (nworld, nray).
  """
  wp.launch(
    _ray_bvh,
    dim=(d.nworld, pnt.shape[1]),
    inputs=[
      m.ngeom,
      m.body_weldid,
      m.geom_type,
      m.geom_bodyid,
      m.geom_dataid,
      m.geom_matid,
      m.geom_group,
      m.geom_size,
      m.geom_rgba,
      m.mat_rgba,
      d.geom_xpos,
      d.geom_xmat,
      pnt,
      vec,
      geomgroup,
      flg_static,
      bodyexclude,
      rc.bvh_id,
      rc.group_root,
      rc.enabled_geom_ids,
      rc.mesh_bvh_id,
      rc.hfield_bvh_id,
    ],
    outputs=[dist, geomid, normal],
  )
