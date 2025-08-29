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

from typing import Any

import warp as wp

from .collision_convex import convex_narrowphase
from .collision_gjk import gjk
from .collision_hfield import hfield_prism_vertex
from .collision_hfield import hfield_subgrid
from .collision_hfield import hfield_triangle_prism
from .collision_primitive import geom
from .collision_primitive import primitive_narrowphase
from .collision_sdf import sdf_narrowphase
from .math import upper_tri_index
from .types import MJ_MAXVAL
from .types import BroadphaseFilter
from .types import BroadphaseType
from .types import Data
from .types import DisableBit
from .types import GeomType
from .types import Model
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _zero_collision_arrays(
  # Data in:
  nworld_in: int,
  # In:
  hfield_geom_pair_in: int,
  # Data out:
  ncon_out: wp.array(dtype=int),
  ncon_hfield_out: wp.array(dtype=int),  # kernel_analyzer: ignore
  collision_hftri_index_out: wp.array(dtype=int),
  ncollision_out: wp.array(dtype=int),
):
  tid = wp.tid()

  if tid == 0:
    # Zero the single collision counter
    ncollision_out[0] = 0
    ncon_out[0] = 0

  if tid < hfield_geom_pair_in * nworld_in:
    ncon_hfield_out[tid] = 0

  # Zero collision pair indices
  collision_hftri_index_out[tid] = 0


@wp.func
def _plane_filter(
  size1: float, size2: float, margin1: float, margin2: float, xpos1: wp.vec3, xpos2: wp.vec3, xmat1: wp.mat33, xmat2: wp.mat33
) -> bool:
  if size1 == 0.0:
    # geom1 is a plane
    dist = wp.dot(xpos2 - xpos1, wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2]))
    return dist <= size2 + wp.max(margin1, margin2)
  elif size2 == 0.0:
    # geom2 is a plane
    dist = wp.dot(xpos1 - xpos2, wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2]))
    return dist <= size1 + wp.max(margin1, margin2)

  return True


@wp.func
def _sphere_filter(size1: float, size2: float, margin1: float, margin2: float, xpos1: wp.vec3, xpos2: wp.vec3) -> bool:
  bound = size1 + size2 + wp.max(margin1, margin2)
  dif = xpos2 - xpos1
  dist_sq = wp.dot(dif, dif)
  return dist_sq <= bound * bound


# TODO(team): improve performance by precomputing bounding box
@wp.func
def _aabb_filter(
  # In:
  center1: wp.vec3,
  center2: wp.vec3,
  size1: wp.vec3,
  size2: wp.vec3,
  margin1: float,
  margin2: float,
  xpos1: wp.vec3,
  xpos2: wp.vec3,
  xmat1: wp.mat33,
  xmat2: wp.mat33,
) -> bool:
  """Axis aligned boxes collision.

  references: see Ericson, Real-time Collision Detection section 4.2.
              filterBox: filter contact based on global AABBs.
  """
  center1 = xmat1 @ center1 + xpos1
  center2 = xmat2 @ center2 + xpos2

  margin = wp.max(margin1, margin2)

  max_x1 = -MJ_MAXVAL
  max_y1 = -MJ_MAXVAL
  max_z1 = -MJ_MAXVAL
  min_x1 = MJ_MAXVAL
  min_y1 = MJ_MAXVAL
  min_z1 = MJ_MAXVAL

  max_x2 = -MJ_MAXVAL
  max_y2 = -MJ_MAXVAL
  max_z2 = -MJ_MAXVAL
  min_x2 = MJ_MAXVAL
  min_y2 = MJ_MAXVAL
  min_z2 = MJ_MAXVAL

  sign = wp.vec2(-1.0, 1.0)

  for i in range(2):
    for j in range(2):
      for k in range(2):
        corner1 = wp.vec3(sign[i] * size1[0], sign[j] * size1[1], sign[k] * size1[2])
        pos1 = xmat1 @ corner1

        corner2 = wp.vec3(sign[i] * size2[0], sign[j] * size2[1], sign[k] * size2[2])
        pos2 = xmat2 @ corner2

        if pos1[0] > max_x1:
          max_x1 = pos1[0]

        if pos1[1] > max_y1:
          max_y1 = pos1[1]

        if pos1[2] > max_z1:
          max_z1 = pos1[2]

        if pos1[0] < min_x1:
          min_x1 = pos1[0]

        if pos1[1] < min_y1:
          min_y1 = pos1[1]

        if pos1[2] < min_z1:
          min_z1 = pos1[2]

        if pos2[0] > max_x2:
          max_x2 = pos2[0]

        if pos2[1] > max_y2:
          max_y2 = pos2[1]

        if pos2[2] > max_z2:
          max_z2 = pos2[2]

        if pos2[0] < min_x2:
          min_x2 = pos2[0]

        if pos2[1] < min_y2:
          min_y2 = pos2[1]

        if pos2[2] < min_z2:
          min_z2 = pos2[2]

  if center1[0] + max_x1 + margin < center2[0] + min_x2:
    return False
  if center1[1] + max_y1 + margin < center2[1] + min_y2:
    return False
  if center1[2] + max_z1 + margin < center2[2] + min_z2:
    return False
  if center2[0] + max_x2 + margin < center1[0] + min_x1:
    return False
  if center2[1] + max_y2 + margin < center1[1] + min_y1:
    return False
  if center2[2] + max_z2 + margin < center1[2] + min_z1:
    return False

  return True


mat23 = wp.types.matrix(shape=(2, 3), dtype=float)
mat63 = wp.types.matrix(shape=(6, 3), dtype=float)


# TODO(team): improve performance by precomputing bounding box
@wp.func
def _obb_filter(
  # In:
  center1: wp.vec3,
  center2: wp.vec3,
  size1: wp.vec3,
  size2: wp.vec3,
  margin1: float,
  margin2: float,
  xpos1: wp.vec3,
  xpos2: wp.vec3,
  xmat1: wp.mat33,
  xmat2: wp.mat33,
) -> bool:
  """Oriented bounding boxes collision (see Gottschalk et al.), see mj_collideOBB."""
  margin = wp.max(margin1, margin2)

  xcenter = mat23()
  normal = mat63()
  proj = wp.vec2()
  radius = wp.vec2()

  # compute centers in local coordinates
  xcenter[0] = xmat1 @ center1 + xpos1
  xcenter[1] = xmat2 @ center2 + xpos2

  # compute normals in global coordinates
  normal[0] = wp.vec3(xmat1[0, 0], xmat1[1, 0], xmat1[2, 0])
  normal[1] = wp.vec3(xmat1[0, 1], xmat1[1, 1], xmat1[2, 1])
  normal[2] = wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2])
  normal[3] = wp.vec3(xmat2[0, 0], xmat2[1, 0], xmat2[2, 0])
  normal[4] = wp.vec3(xmat2[0, 1], xmat2[1, 1], xmat2[2, 1])
  normal[5] = wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2])

  # check intersections
  for j in range(2):
    for k in range(3):
      for i in range(2):
        proj[i] = wp.dot(xcenter[i], normal[3 * j + k])
        if i == 0:
          size = size1
        else:
          size = size2

        # fmt: off
        radius[i] = (
            wp.abs(size[0] * wp.dot(normal[3 * i + 0], normal[3 * j + k]))
          + wp.abs(size[1] * wp.dot(normal[3 * i + 1], normal[3 * j + k]))
          + wp.abs(size[2] * wp.dot(normal[3 * i + 2], normal[3 * j + k]))
        )
        # fmt: on
      if radius[0] + radius[1] + margin < wp.abs(proj[1] - proj[0]):
        return False

  return True


def _broadphase_filter(opt_broadphase_filter: int):
  @wp.func
  def func(
    # Model:
    geom_aabb: wp.array2d(dtype=wp.vec3),
    geom_rbound: wp.array2d(dtype=float),
    geom_margin: wp.array2d(dtype=float),
    # Data in:
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    # In:
    geom1: int,
    geom2: int,
    worldid: int,
  ) -> bool:
    # 1: plane
    # 2: sphere
    # 4: aabb
    # 8: obb

    center1 = geom_aabb[geom1, 0]
    center2 = geom_aabb[geom2, 0]
    size1 = geom_aabb[geom1, 1]
    size2 = geom_aabb[geom2, 1]
    rbound1 = geom_rbound[worldid, geom1]
    rbound2 = geom_rbound[worldid, geom2]
    margin1 = geom_margin[worldid, geom1]
    margin2 = geom_margin[worldid, geom2]
    xpos1 = geom_xpos_in[worldid, geom1]
    xpos2 = geom_xpos_in[worldid, geom2]
    xmat1 = geom_xmat_in[worldid, geom1]
    xmat2 = geom_xmat_in[worldid, geom2]

    if rbound1 == 0.0 or rbound2 == 0.0:
      if wp.static(opt_broadphase_filter & int(BroadphaseFilter.PLANE.value)):
        return _plane_filter(rbound1, rbound2, margin1, margin2, xpos1, xpos2, xmat1, xmat2)
    else:
      if wp.static(opt_broadphase_filter & int(BroadphaseFilter.SPHERE.value)):
        if not _sphere_filter(rbound1, rbound2, margin1, margin2, xpos1, xpos2):
          return False
      if wp.static(opt_broadphase_filter & int(BroadphaseFilter.AABB.value)):
        if not _aabb_filter(center1, center2, size1, size2, margin1, margin2, xpos1, xpos2, xmat1, xmat2):
          return False
      if wp.static(opt_broadphase_filter & int(BroadphaseFilter.OBB.value)):
        if not _obb_filter(center1, center2, size1, size2, margin1, margin2, xpos1, xpos2, xmat1, xmat2):
          return False

    return True

  return func


@wp.func
def _add_geom_pair(
  # Model:
  geom_type: wp.array(dtype=int),
  nxn_pairid: wp.array(dtype=int),
  # Data in:
  nconmax_in: int,
  # In:
  geom1: int,
  geom2: int,
  worldid: int,
  nxnid: int,
  # Data out:
  collision_pair_out: wp.array(dtype=wp.vec2i),
  collision_hftri_index_out: wp.array(dtype=int),
  collision_pairid_out: wp.array(dtype=int),
  collision_worldid_out: wp.array(dtype=int),
  ncollision_out: wp.array(dtype=int),
):
  pairid = wp.atomic_add(ncollision_out, 0, 1)

  if pairid >= nconmax_in:
    return

  type1 = geom_type[geom1]
  type2 = geom_type[geom2]

  if type1 > type2:
    pair = wp.vec2i(geom2, geom1)
  else:
    pair = wp.vec2i(geom1, geom2)

  collision_pair_out[pairid] = pair
  collision_pairid_out[pairid] = nxn_pairid[nxnid]
  collision_worldid_out[pairid] = worldid

  # Writing -1 to collision_hftri_index_out[pairid] signals
  # hfield_midphase to generate a collision pair for every
  # potentially colliding triangle
  if type1 == int(GeomType.HFIELD.value) or type2 == int(GeomType.HFIELD.value):
    collision_hftri_index_out[pairid] = -1


@wp.func
def _binary_search(values: wp.array(dtype=Any), value: Any, lower: int, upper: int) -> int:
  while lower < upper:
    mid = (lower + upper) >> 1
    if values[mid] > value:
      upper = mid
    else:
      lower = mid + 1

  return upper


@wp.kernel
def _sap_project(
  # Model:
  geom_rbound: wp.array2d(dtype=float),
  geom_margin: wp.array2d(dtype=float),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  # In:
  direction_in: wp.vec3,
  # Data out:
  sap_projection_lower_out: wp.array2d(dtype=float),  # kernel_analyzer: ignore
  sap_projection_upper_out: wp.array2d(dtype=float),
  sap_sort_index_out: wp.array2d(dtype=int),  # kernel_analyzer: ignore
):
  worldid, geomid = wp.tid()

  xpos = geom_xpos_in[worldid, geomid]
  rbound = geom_rbound[worldid, geomid]

  if rbound == 0.0:
    # geom is a plane
    rbound = MJ_MAXVAL

  radius = rbound + geom_margin[worldid, geomid]
  center = wp.dot(direction_in, xpos)

  sap_sort_index_out[worldid, geomid] = geomid
  if not wp.isnan(center):
    sap_projection_lower_out[worldid, geomid] = center - radius
    sap_projection_upper_out[worldid, geomid] = center + radius
  else:
    sap_projection_lower_out[worldid, geomid] = MJ_MAXVAL
    sap_projection_upper_out[worldid, geomid] = MJ_MAXVAL


@wp.kernel
def _sap_range(
  # Model:
  ngeom: int,
  # Data in:
  sap_projection_lower_in: wp.array2d(dtype=float),  # kernel_analyzer: ignore
  sap_projection_upper_in: wp.array2d(dtype=float),
  sap_sort_index_in: wp.array2d(dtype=int),  # kernel_analyzer: ignore
  # Data out:
  sap_range_out: wp.array2d(dtype=int),
):
  worldid, geomid = wp.tid()

  # current bounding geom
  idx = sap_sort_index_in[worldid, geomid]

  upper = sap_projection_upper_in[worldid, idx]

  limit = _binary_search(sap_projection_lower_in[worldid], upper, geomid + 1, ngeom)
  limit = wp.min(ngeom - 1, limit)

  # range of geoms for the sweep and prune process
  sap_range_out[worldid, geomid] = limit - geomid


@cache_kernel
def _sap_broadphase(broadphase_filter):
  @nested_kernel
  def kernel(
    # Model:
    ngeom: int,
    geom_type: wp.array(dtype=int),
    geom_aabb: wp.array2d(dtype=wp.vec3),
    geom_rbound: wp.array2d(dtype=float),
    geom_margin: wp.array2d(dtype=float),
    nxn_pairid: wp.array(dtype=int),
    # Data in:
    nworld_in: int,
    nconmax_in: int,
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    sap_sort_index_in: wp.array2d(dtype=int),  # kernel_analyzer: ignore
    sap_cumulative_sum_in: wp.array(dtype=int),  # kernel_analyzer: ignore
    # In:
    nsweep_in: int,
    # Data out:
    collision_pair_out: wp.array(dtype=wp.vec2i),
    collision_hftri_index_out: wp.array(dtype=int),
    collision_pairid_out: wp.array(dtype=int),
    collision_worldid_out: wp.array(dtype=int),
    ncollision_out: wp.array(dtype=int),
  ):
    worldgeomid = wp.tid()

    nworldgeom = nworld_in * ngeom
    nworkpackages = sap_cumulative_sum_in[nworldgeom - 1]

    while worldgeomid < nworkpackages:
      # binary search to find current and next geom pair indices
      i = _binary_search(sap_cumulative_sum_in, worldgeomid, 0, nworldgeom)
      j = i + worldgeomid + 1

      if i > 0:
        j -= sap_cumulative_sum_in[i - 1]

      worldid = i // ngeom
      i = i % ngeom
      j = j % ngeom

      # get geom indices and swap if necessary
      geom1 = sap_sort_index_in[worldid, i]
      geom2 = sap_sort_index_in[worldid, j]

      # find linear index of (geom1, geom2) in upper triangular nxn_pairid
      if geom2 < geom1:
        idx = upper_tri_index(ngeom, geom2, geom1)
      else:
        idx = upper_tri_index(ngeom, geom1, geom2)

      worldgeomid += nsweep_in
      if nxn_pairid[idx] < -1:
        continue

      if broadphase_filter(geom_aabb, geom_rbound, geom_margin, geom_xpos_in, geom_xmat_in, geom1, geom2, worldid):
        _add_geom_pair(
          geom_type,
          nxn_pairid,
          nconmax_in,
          geom1,
          geom2,
          worldid,
          idx,
          collision_pair_out,
          collision_hftri_index_out,
          collision_pairid_out,
          collision_worldid_out,
          ncollision_out,
        )

  return kernel


def _segmented_sort(tile_size: int):
  @wp.kernel
  def segmented_sort(
    # Data in:
    sap_projection_lower_in: wp.array2d(dtype=float),  # kernel_analyzer: ignore
    sap_sort_index_in: wp.array2d(dtype=int),  # kernel_analyzer: ignore
  ):
    worldid = wp.tid()

    # Load input into shared memory
    keys = wp.tile_load(sap_projection_lower_in[worldid], shape=tile_size, storage="shared")
    values = wp.tile_load(sap_sort_index_in[worldid], shape=tile_size, storage="shared")

    # Perform in-place sorting
    wp.tile_sort(keys, values)

    # Store sorted shared memory into output arrays
    wp.tile_store(sap_projection_lower_in[worldid], keys)
    wp.tile_store(sap_sort_index_in[worldid], values)

  return segmented_sort


@event_scope
def sap_broadphase(m: Model, d: Data):
  """Runs broadphase collision detection using a sweep-and-prune (SAP) algorithm.

  This method is more efficient than the N-squared approach for large numbers of
  objects. It works by projecting the bounding spheres of all geoms onto a
  single axis and sorting them. It then sweeps along the axis, only checking
  for overlaps between geoms whose projections are close to each other.

  For each potentially colliding pair identified by the sweep, a more precise
  bounding sphere check is performed. If this check passes, the pair is added
  to the collision arrays in `d` for the narrowphase stage.

  Two sorting strategies are supported, controlled by `m.opt.broadphase`:
  - `SAP_TILE`: Uses a tile-based sort.
  - `SAP_SEGMENTED`: Uses a segmented sort.
  """

  nworldgeom = d.nworld * m.ngeom

  # TODO(team): direction

  # random fixed direction
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)

  wp.launch(
    kernel=_sap_project,
    dim=(d.nworld, m.ngeom),
    inputs=[
      m.geom_rbound,
      m.geom_margin,
      d.geom_xpos,
      direction,
    ],
    outputs=[
      d.sap_projection_lower.reshape((-1, m.ngeom)),
      d.sap_projection_upper,
      d.sap_sort_index.reshape((-1, m.ngeom)),
    ],
  )

  if m.opt.broadphase == int(BroadphaseType.SAP_TILE):
    wp.launch_tiled(
      kernel=_segmented_sort(m.ngeom),
      dim=(d.nworld),
      inputs=[d.sap_projection_lower.reshape((-1, m.ngeom)), d.sap_sort_index.reshape((-1, m.ngeom))],
      block_dim=m.block_dim.segmented_sort,
    )
  else:
    wp.utils.segmented_sort_pairs(
      d.sap_projection_lower.reshape((-1, m.ngeom)),
      d.sap_sort_index.reshape((-1, m.ngeom)),
      nworldgeom,
      d.sap_segment_index.reshape(-1),
    )

  wp.launch(
    kernel=_sap_range,
    dim=(d.nworld, m.ngeom),
    inputs=[
      m.ngeom,
      d.sap_projection_lower.reshape((-1, m.ngeom)),
      d.sap_projection_upper,
      d.sap_sort_index.reshape((-1, m.ngeom)),
    ],
    outputs=[
      d.sap_range,
    ],
  )

  # scan is used for load balancing among the threads
  wp.utils.array_scan(d.sap_range.reshape(-1), d.sap_cumulative_sum.reshape(-1), True)

  # estimate number of overlap checks
  # assumes each geom has 5 other geoms (batched over all worlds)
  nsweep = 5 * nworldgeom
  broadphase_filter = _broadphase_filter(m.opt.broadphase_filter)
  wp.launch(
    kernel=_sap_broadphase(broadphase_filter),
    dim=nsweep,
    inputs=[
      m.ngeom,
      m.geom_type,
      m.geom_aabb,
      m.geom_rbound,
      m.geom_margin,
      m.nxn_pairid,
      d.nworld,
      d.nconmax,
      d.geom_xpos,
      d.geom_xmat,
      d.sap_sort_index.reshape((-1, m.ngeom)),
      d.sap_cumulative_sum.reshape(-1),
      nsweep,
    ],
    outputs=[
      d.collision_pair,
      d.collision_hftri_index,
      d.collision_pairid,
      d.collision_worldid,
      d.ncollision,
    ],
  )


@cache_kernel
def _nxn_broadphase(broadphase_filter):
  @nested_kernel
  def kernel(
    # Model:
    geom_type: wp.array(dtype=int),
    geom_aabb: wp.array2d(dtype=wp.vec3),
    geom_rbound: wp.array2d(dtype=float),
    geom_margin: wp.array2d(dtype=float),
    nxn_geom_pair: wp.array(dtype=wp.vec2i),
    nxn_pairid: wp.array(dtype=int),
    # Data in:
    nconmax_in: int,
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    # Data out:
    collision_pair_out: wp.array(dtype=wp.vec2i),
    collision_hftri_index_out: wp.array(dtype=int),
    collision_pairid_out: wp.array(dtype=int),
    collision_worldid_out: wp.array(dtype=int),
    ncollision_out: wp.array(dtype=int),
  ):
    worldid, elementid = wp.tid()

    geom = nxn_geom_pair[elementid]
    geom1 = geom[0]
    geom2 = geom[1]

    if broadphase_filter(geom_aabb, geom_rbound, geom_margin, geom_xpos_in, geom_xmat_in, geom1, geom2, worldid):
      _add_geom_pair(
        geom_type,
        nxn_pairid,
        nconmax_in,
        geom1,
        geom2,
        worldid,
        elementid,
        collision_pair_out,
        collision_hftri_index_out,
        collision_pairid_out,
        collision_worldid_out,
        ncollision_out,
      )

  return kernel


@event_scope
def nxn_broadphase(m: Model, d: Data):
  """Runs broadphase collision detection using a brute-force N-squared approach.

  This function iterates through a pre-filtered list of all possible geometry pairs and
  performs a quick bounding sphere check to identify potential collisions.

  For each pair that passes the sphere check, it populates the collision arrays in `d`
  (`d.collision_pair`, `d.collision_pairid`, etc.), which are then consumed by the
  narrowphase.

  The initial list of pairs is filtered at model creation time to exclude pairs based on
  `contype`/`conaffinity`, parent-child relationships, and explicit `<exclude>` tags.
  """

  broadphase_filter = _broadphase_filter(m.opt.broadphase_filter)
  wp.launch(
    _nxn_broadphase(broadphase_filter),
    dim=(d.nworld, m.nxn_geom_pair_filtered.shape[0]),
    inputs=[
      m.geom_type,
      m.geom_aabb,
      m.geom_rbound,
      m.geom_margin,
      m.nxn_geom_pair_filtered,
      m.nxn_pairid_filtered,
      d.nconmax,
      d.geom_xpos,
      d.geom_xmat,
    ],
    outputs=[
      d.collision_pair,
      d.collision_hftri_index,
      d.collision_pairid,
      d.collision_worldid,
      d.ncollision,
    ],
  )


@wp.kernel
def _hfield_midphase(
  # Model:
  opt_ccd_tolerance: wp.array(dtype=float),
  opt_gjk_iterations: int,
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_aabb: wp.array2d(dtype=wp.vec3),
  geom_rbound: wp.array2d(dtype=float),
  geom_margin: wp.array2d(dtype=float),
  hfield_adr: wp.array(dtype=int),
  hfield_nrow: wp.array(dtype=int),
  hfield_ncol: wp.array(dtype=int),
  hfield_size: wp.array(dtype=wp.vec4),
  hfield_data: wp.array(dtype=float),
  mesh_vertadr: wp.array(dtype=int),
  mesh_vertnum: wp.array(dtype=int),
  mesh_vert: wp.array(dtype=wp.vec3),
  mesh_graphadr: wp.array(dtype=int),
  mesh_graph: wp.array(dtype=int),
  mesh_polynum: wp.array(dtype=int),
  mesh_polyadr: wp.array(dtype=int),
  mesh_polynormal: wp.array(dtype=wp.vec3),
  mesh_polyvertadr: wp.array(dtype=int),
  mesh_polyvertnum: wp.array(dtype=int),
  mesh_polyvert: wp.array(dtype=int),
  mesh_polymapadr: wp.array(dtype=int),
  mesh_polymapnum: wp.array(dtype=int),
  mesh_polymap: wp.array(dtype=int),
  # Data in:
  nconmax_in: int,
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  collision_pair_in: wp.array(dtype=wp.vec2i),
  collision_hftri_index_in: wp.array(dtype=int),
  collision_pairid_in: wp.array(dtype=int),
  collision_worldid_in: wp.array(dtype=int),
  # Data out:
  collision_pair_out: wp.array(dtype=wp.vec2i),
  collision_hftri_index_out: wp.array(dtype=int),
  collision_pairid_out: wp.array(dtype=int),
  collision_worldid_out: wp.array(dtype=int),
  ncollision_out: wp.array(dtype=int),
):
  """Midphase collision detection for heightfield triangles with other geoms.

  This kernel processes collision pairs where one geom is a heightfield (identified by
  collision_hftri_index_in[pairid] == -1) and expands them into multiple collision pairs,
  one for each potentially colliding triangle. Height field triangular prisms are filtered
  by the GJK routine.

  Args:
    opt_ccd_tolerance: convex collision solver tolerance
    opt_gjk_iterations: number of iterations for GJK routine
    geom_type: geom type
    geom_dataid: geom data id
    geom_size: geom-specific size parameters
    geom_rbound: geom bounding sphere radius
    geom_margin: geom margin
    hfield_adr: start address in hfield_data
    hfield_nrow: height field number of rows
    hfield_ncol: height field number of columns
    hfield_size: height field size
    hfield_data: elevation data
    mesh_vertadr: first vertex address
    mesh_vertnum: number of vertices
    mesh_vert: vertex positions for all meshes
    mesh_graphadr: graph data address; -1: no graph
    mesh_graph: convex graph data
    mesh_polynum: number of polygons per mesh
    mesh_polyadr: first polygon address per mesh
    mesh_polynormal: all polygon normals
    mesh_polyvertadr: polygon vertex start address
    mesh_polyvertnum: number of vertices per polygon
    mesh_polyvert: all polygon vertices
    mesh_polymapadr: first polygon address per vertex
    mesh_polymapnum: number of polygons per vertex
    mesh_polymap: vertex to polygon map
    nconmax_in: maximum number of contacts
    geom_xpos_in: geom position
    geom_xmat_in: geom orientation
    collision_pair_in: collision pair
    collision_hftri_index_in: triangle indices, -1 for height field pair
    collision_pairid_in: collision pair id from broadphase
    collision_worldid_in: collision world id from broadphase
    collision_pair_out: collision pair from midphase
    collision_hftri_index_out: triangle indices from midphase
    collision_pairid_out: collision pair id from midphase
    collision_worldid_out: collision world id from midphase
    ncollision_out: number of collisions from broadphase and midphase
  """
  pairid = wp.tid()

  # only process pairs that are marked for height field collision (-1)
  if collision_hftri_index_in[pairid] != -1:
    return

  # collision pair info
  worldid = collision_worldid_in[pairid]
  pair_id = collision_pairid_in[pairid]

  pair = collision_pair_in[pairid]
  g1 = pair[0]
  g2 = pair[1]

  hfieldid = g1
  geomid = g2

  # SHOULD NOT OCCUR: if the first geom is not a heightfield, swap
  if geom_type[g1] != int(GeomType.HFIELD.value):
    hfieldid = g2
    geomid = g1

  # height field info
  hfdataid = geom_dataid[hfieldid]
  size1 = hfield_size[hfdataid]
  pos1 = geom_xpos_in[worldid, hfieldid]
  mat1 = geom_xmat_in[worldid, hfieldid]
  mat1T = wp.transpose(mat1)

  # geom info
  pos2 = geom_xpos_in[worldid, geomid]
  pos = mat1T @ (pos2 - pos1)
  r2 = geom_rbound[worldid, geomid]

  # TODO(team): margin?
  margin = wp.max(geom_margin[worldid, hfieldid], geom_margin[worldid, geomid])

  # box-sphere test: horizontal plane
  for i in range(2):
    if (size1[i] < pos[i] - r2 - margin) or (-size1[i] > pos[i] + r2 + margin):
      return

  # box-sphere test: vertical direction
  if size1[2] < pos[2] - r2 - margin:  # up
    return

  if -size1[3] > pos[2] + r2 + margin:  # down
    return

  mat2 = geom_xmat_in[worldid, geomid]
  mat = mat1T @ mat2

  # aabb for geom in height field frame
  xmax = -MJ_MAXVAL
  ymax = -MJ_MAXVAL
  zmax = -MJ_MAXVAL
  xmin = MJ_MAXVAL
  ymin = MJ_MAXVAL
  zmin = MJ_MAXVAL

  center2 = geom_aabb[geomid, 0]
  size2 = geom_aabb[geomid, 1]

  pos += mat1T @ center2

  sign = wp.vec2(-1.0, 1.0)

  for i in range(2):
    for j in range(2):
      for k in range(2):
        corner_local = wp.vec3(sign[i] * size2[0], sign[j] * size2[1], sign[k] * size2[2])
        corner_hf = mat @ corner_local

        if corner_hf[0] > xmax:
          xmax = corner_hf[0]
        if corner_hf[1] > ymax:
          ymax = corner_hf[1]
        if corner_hf[2] > zmax:
          zmax = corner_hf[2]
        if corner_hf[0] < xmin:
          xmin = corner_hf[0]
        if corner_hf[1] < ymin:
          ymin = corner_hf[1]
        if corner_hf[2] < zmin:
          zmin = corner_hf[2]

  xmax += pos[0]
  xmin += pos[0]
  ymax += pos[1]
  ymin += pos[1]
  zmax += pos[2]
  zmin += pos[2]

  # box-box test
  if (
    (xmin - margin > size1[0])
    or (xmax + margin < -size1[0])
    or (ymin - margin > size1[1])
    or (ymax + margin < -size1[1])
    or (zmin - margin > size1[2])
    or (zmax + margin < -size1[3])
  ):
    return

  # height field subgrid
  nrow = hfield_nrow[hfieldid]
  ncol = hfield_ncol[hfieldid]
  size = hfield_size[hfieldid]
  cmin, rmin, cmax, rmax = hfield_subgrid(nrow, ncol, size, xmax, xmin, ymax, ymin)

  # GJK setup
  geom1 = geom(
    geom_type,
    geom_dataid,
    geom_size,
    hfield_adr,
    hfield_nrow,
    hfield_ncol,
    hfield_size,
    hfield_data,
    mesh_vertadr,
    mesh_vertnum,
    mesh_vert,
    mesh_graphadr,
    mesh_graph,
    mesh_polynum,
    mesh_polyadr,
    mesh_polynormal,
    mesh_polyvertadr,
    mesh_polyvertnum,
    mesh_polyvert,
    mesh_polymapadr,
    mesh_polymapnum,
    mesh_polymap,
    geom_xpos_in,
    geom_xmat_in,
    worldid,
    hfieldid,
    -1,  # overwrite height field prism in loop below
  )

  geom2 = geom(
    geom_type,
    geom_dataid,
    geom_size,
    hfield_adr,
    hfield_nrow,
    hfield_ncol,
    hfield_size,
    hfield_data,
    mesh_vertadr,
    mesh_vertnum,
    mesh_vert,
    mesh_graphadr,
    mesh_graph,
    mesh_polynum,
    mesh_polyadr,
    mesh_polynormal,
    mesh_polyvertadr,
    mesh_polyvertnum,
    mesh_polyvert,
    mesh_polymapadr,
    mesh_polymapnum,
    mesh_polymap,
    geom_xpos_in,
    geom_xmat_in,
    worldid,
    geomid,
    -1,
  )

  x_1 = geom1.pos
  x_2 = geom2.pos
  geomtype1 = geom_type[hfieldid]
  geomtype2 = geom_type[geomid]

  ccd_tolerance = opt_ccd_tolerance[worldid]

  # loop over subgrid triangles
  firstprism = bool(True)
  for r in range(rmin, rmax):
    for c in range(cmin, cmax):
      # add both triangles from this cell
      for i in range(2):
        # height field prism
        hftri_index = 2 * (r * (ncol - 1) + c) + i
        geom1.hfprism = hfield_triangle_prism(
          geom_dataid, hfield_adr, hfield_nrow, hfield_ncol, hfield_size, hfield_data, g1, hftri_index
        )

        prism_pos = wp.vec3(0.0, 0.0, 0.0)
        for i in range(6):
          prism_pos += hfield_prism_vertex(geom1.hfprism, i)
        prism_pos = geom1.rot @ (prism_pos / 6.0)

        result = gjk(ccd_tolerance, opt_gjk_iterations, geom1, geom2, x_1 + prism_pos, x_2, geomtype1, geomtype2, 0.0, False)

        if result.dim != 0:
          if firstprism:
            new_pairid = pairid
            firstprism = False
          else:  # create a new pair
            new_pairid = wp.atomic_add(ncollision_out, 0, 1)

          if new_pairid >= nconmax_in:
            return

          collision_pair_out[new_pairid] = pair
          collision_hftri_index_out[new_pairid] = hftri_index
          collision_pairid_out[new_pairid] = pair_id
          collision_worldid_out[new_pairid] = worldid


def _narrowphase(m, d):
  # Process heightfield collisions
  if m.nhfield > 0:
    wp.launch(
      kernel=_hfield_midphase,
      dim=d.nconmax,
      inputs=[
        m.opt.ccd_tolerance,
        m.opt.gjk_iterations,
        m.geom_type,
        m.geom_dataid,
        m.geom_size,
        m.geom_aabb,
        m.geom_rbound,
        m.geom_margin,
        m.hfield_adr,
        m.hfield_nrow,
        m.hfield_ncol,
        m.hfield_size,
        m.hfield_data,
        m.mesh_vertadr,
        m.mesh_vertnum,
        m.mesh_vert,
        m.mesh_graphadr,
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
        d.nconmax,
        d.geom_xpos,
        d.geom_xmat,
        d.collision_pair,
        d.collision_hftri_index,
        d.collision_pairid,
        d.collision_worldid,
      ],
      outputs=[d.collision_pair, d.collision_hftri_index, d.collision_pairid, d.collision_worldid, d.ncollision],
    )
  # TODO(team): we should reject far-away contacts in the narrowphase instead of constraint
  #             partitioning because we can move some pressure of the atomics
  convex_narrowphase(m, d)
  primitive_narrowphase(m, d)

  if m.has_sdf_geom:
    sdf_narrowphase(m, d)


@event_scope
def collision(m: Model, d: Data):
  """Runs the full collision detection pipeline.

  This function orchestrates the broadphase and narrowphase collision detection stages. It
  first identifies potential collision pairs using a broadphase algorithm (either N-squared
  or Sweep-and-Prune, based on `m.opt.broadphase`). Then, for each potential pair, it
  performs narrowphase collision detection to compute detailed contact information like
  distance, position, and frame.

  The results are used to populate the `d.contact` array, and the total number of contacts
  is stored in `d.ncon`.  If `d.ncon` is larger than `d.nconmax` then an overflow has
  occurred and the remaining contacts will be skipped.  If this happens, raise the `nconmax`
  parameter in `io.make_data` or `io.put_data`.

  This function will do nothing except zero out arrays if collision detection is disabled
  via `m.opt.disableflags` or if `d.nconmax` is 0.
  """

  # zero collision-related arrays
  wp.launch(
    _zero_collision_arrays,
    dim=d.nconmax,
    inputs=[
      d.nworld,
      d.ncon_hfield.shape[1],
      d.ncon,
      d.ncon_hfield.reshape(-1),
      d.collision_hftri_index,
      d.ncollision,
    ],
  )

  if d.nconmax == 0 or m.opt.disableflags & (DisableBit.CONSTRAINT | DisableBit.CONTACT):
    return

  if m.opt.broadphase == int(BroadphaseType.NXN):
    nxn_broadphase(m, d)
  else:
    sap_broadphase(m, d)

  if m.opt.graph_conditional:
    wp.capture_if(condition=d.ncollision, on_true=_narrowphase, m=m, d=d)
  else:
    _narrowphase(m, d)
