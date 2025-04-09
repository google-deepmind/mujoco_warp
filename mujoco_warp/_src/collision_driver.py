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

import warp as wp

from .collision_box import box_box_narrowphase
from .collision_convex import gjk_narrowphase
from .collision_primitive import primitive_narrowphase
from .types import MJ_MINVAL
from .types import Data
from .types import DisableBit
from .types import Model
from .types import vec5
from .warp_util import event_scope


@wp.func
def _geom_filter(m: Model, geom1: int, geom2: int, filterparent: bool) -> bool:
  bodyid1 = m.geom_bodyid[geom1]
  bodyid2 = m.geom_bodyid[geom2]
  contype1 = m.geom_contype[geom1]
  contype2 = m.geom_contype[geom2]
  conaffinity1 = m.geom_conaffinity[geom1]
  conaffinity2 = m.geom_conaffinity[geom2]
  weldid1 = m.body_weldid[bodyid1]
  weldid2 = m.body_weldid[bodyid2]
  weld_parentid1 = m.body_weldid[m.body_parentid[weldid1]]
  weld_parentid2 = m.body_weldid[m.body_parentid[weldid2]]

  self_collision = weldid1 == weldid2
  parent_child_collision = (
    filterparent
    and (weldid1 != 0)
    and (weldid2 != 0)
    and ((weldid1 == weld_parentid2) or (weldid2 == weld_parentid1))
  )
  mask = (contype1 & conaffinity2) or (contype2 & conaffinity1)

  return mask and (not self_collision) and (not parent_child_collision)


@wp.func
def _add_geom_pair(m: Model, d: Data, geom1: int, geom2: int, worldid: int):
  pairid = wp.atomic_add(d.ncollision, 0, 1)

  if pairid >= d.nconmax:
    return

  type1 = m.geom_type[geom1]
  type2 = m.geom_type[geom2]

  if type1 > type2:
    pair = wp.vec2i(geom2, geom1)
  else:
    pair = wp.vec2i(geom1, geom2)

  d.collision_pair[pairid] = pair
  d.collision_worldid[pairid] = worldid


@wp.kernel
def broadphase_project_spheres_onto_sweep_direction_kernel(
  m: Model,
  d: Data,
  direction: wp.vec3,
):
  worldid, i = wp.tid()

  c = d.geom_xpos[worldid, i]
  r = m.geom_rbound[i]
  if r == 0.0:
    # current geom is a plane
    r = 1000000000.0
  sphere_radius = r + m.geom_margin[i]

  center = wp.dot(direction, c)
  f = center - sphere_radius

  # Store results in the data arrays
  d.sap_projection_lower[worldid, i] = f
  d.sap_projection_upper[worldid, i] = center + sphere_radius
  d.sap_sort_index[worldid, i] = i


# Define constants for plane types
PLANE_ZERO_OFFSET = -1.0
PLANE_NEGATIVE_OFFSET = -2.0
PLANE_POSITIVE_OFFSET = -3.0


@wp.func
def encode_plane(normal: wp.vec3, point_on_plane: wp.vec3, margin: float) -> wp.vec4:
  normal = wp.normalize(normal)
  plane_offset = -wp.dot(normal, point_on_plane + normal * margin)

  # Scale factor for the normal
  scale = wp.abs(plane_offset)

  # Handle special cases
  if wp.abs(plane_offset) < 1e-6:
    return wp.vec4(normal.x, normal.y, normal.z, PLANE_ZERO_OFFSET)
  elif plane_offset < 0.0:
    return wp.vec4(
      scale * normal.x, scale * normal.y, scale * normal.z, PLANE_NEGATIVE_OFFSET
    )
  else:
    return wp.vec4(
      scale * normal.x, scale * normal.y, scale * normal.z, PLANE_POSITIVE_OFFSET
    )


@wp.func
def decode_plane(encoded: wp.vec4) -> wp.vec4:
  magnitude = wp.length(encoded)
  normal = wp.normalize(xyz(encoded))

  if encoded.w == PLANE_ZERO_OFFSET:
    return wp.vec4(normal.x, normal.y, normal.z, 0.0)
  elif encoded.w == PLANE_NEGATIVE_OFFSET:
    return wp.vec4(normal.x, normal.y, normal.z, -magnitude)
  else:
    return wp.vec4(normal.x, normal.y, normal.z, magnitude)


@wp.kernel
def reorder_bounding_spheres_kernel(
  m: Model,
  d: Data,
):
  worldid, i = wp.tid()

  # Get the index from the data indexer
  mapped = d.sap_sort_index[worldid, i]

  # Get the bounding volume
  c = d.geom_xpos[worldid, mapped]
  r = m.geom_rbound[mapped]
  margin = m.geom_margin[mapped]

  # Reorder the box into the sorted array
  if r == 0.0:
    # store the plane equation
    xmat = d.geom_xmat[worldid, mapped]
    plane_normal = wp.vec3(xmat[0, 2], xmat[1, 2], xmat[2, 2])
    d.sap_geom_sort[worldid, i] = encode_plane(
      plane_normal, c, margin
    )  # negative w component is used to disginguish planes from spheres
  else:
    d.sap_geom_sort[worldid, i] = wp.vec4(c.x, c.y, c.z, r + margin)


@wp.func
def xyz(v: wp.vec4) -> wp.vec3:
  return wp.vec3(v.x, v.y, v.z)


@wp.func
def signed_distance_point_plane(point: wp.vec3, plane: wp.vec4) -> float:
  return wp.dot(point, xyz(plane)) + plane.w


@wp.func
def overlap(
  world_id: int,
  a: int,
  b: int,
  spheres_or_planes: wp.array(dtype=wp.vec4, ndim=2),
) -> bool:
  # Extract centers and sizes
  s_a = spheres_or_planes[world_id, a]
  s_b = spheres_or_planes[world_id, b]

  if s_a.w < 0.0 and s_b.w < 0.0:
    # both are planes
    return False
  elif s_a.w < 0.0 or s_b.w < 0.0:
    if s_b.w < 0.0:  # swap if required such that s_a is always a plane
      tmp = s_a
      s_a = s_b
      s_b = tmp
    s_a = decode_plane(s_a)
    dist = signed_distance_point_plane(xyz(s_b), s_a)
    return dist <= s_b.w
  else:
    # geoms are spheres
    delta = xyz(s_a) - xyz(s_b)
    dist_sq = wp.dot(delta, delta)
    radius_sum = s_a.w + s_b.w
    return dist_sq <= radius_sum * radius_sum


@wp.func
def find_first_greater_than(
  worldid: int,
  starts: wp.array(dtype=wp.float32, ndim=2),
  value: wp.float32,
  low: int,
  high: int,
) -> int:
  while low < high:
    mid = (low + high) >> 1
    if starts[worldid, mid] > value:
      high = mid
    else:
      low = mid + 1
  return low


@wp.kernel
def sap_broadphase_prepare_kernel(
  m: Model,
  d: Data,
):
  worldid, i = wp.tid()  # Get the thread ID

  # Get the index of the current bounding box
  idx1 = d.sap_sort_index[worldid, i]

  end = d.sap_projection_upper[worldid, idx1]
  limit = find_first_greater_than(worldid, d.sap_projection_lower, end, i + 1, m.ngeom)
  limit = wp.min(m.ngeom - 1, limit)

  # Calculate the range of boxes for the sweep and prune process
  count = limit - i

  # Store the cumulative sum for the current box
  d.sap_range[worldid, i] = count


@wp.func
def find_right_most_index_int(
  starts: wp.array(dtype=wp.int32, ndim=1), value: wp.int32, low: int, high: int
) -> int:
  while low < high:
    mid = (low + high) >> 1
    if starts[mid] > value:
      high = mid
    else:
      low = mid + 1
  return high


@wp.func
def find_indices(
  id: int, cumulative_sum: wp.array(dtype=wp.int32, ndim=1), length: int
) -> wp.vec2i:
  # Perform binary search to find the right most index
  i = find_right_most_index_int(cumulative_sum, id, 0, length)

  # Get the baseId, and compute the offset and j
  if i > 0:
    base_id = cumulative_sum[i - 1]
  else:
    base_id = 0
  offset = id - base_id
  j = i + offset + 1

  return wp.vec2i(i, j)


@wp.kernel
def sap_broadphase_kernel(m: Model, d: Data, num_threads: int, filter_parent: bool):
  threadId = wp.tid()  # Get thread ID
  if d.sap_cumulative_sum.shape[0] > 0:
    total_num_work_packages = d.sap_cumulative_sum[d.sap_cumulative_sum.shape[0] - 1]
  else:
    total_num_work_packages = 0

  while threadId < total_num_work_packages:
    # Get indices for current and next box pair
    ij = find_indices(threadId, d.sap_cumulative_sum, d.sap_cumulative_sum.shape[0])
    i = ij.x
    j = ij.y

    worldid = i // m.ngeom
    i = i % m.ngeom
    j = j % m.ngeom

    # geom index
    idx1 = d.sap_sort_index[worldid, i]
    idx2 = d.sap_sort_index[worldid, j]

    if not _geom_filter(m, idx1, idx2, filter_parent):
      threadId += num_threads
      continue

    # Check if the boxes overlap
    if overlap(worldid, i, j, d.sap_geom_sort):
      _add_geom_pair(m, d, idx1, idx2, worldid)

    threadId += num_threads


@wp.kernel
def get_contact_solver_params_kernel(
  m: Model,
  d: Data,
):
  tid = wp.tid()

  n_contact_pts = d.ncon[0]
  if tid >= n_contact_pts:
    return

  geoms = d.contact.geom[tid]
  g1 = geoms.x
  g2 = geoms.y

  margin = wp.max(m.geom_margin[g1], m.geom_margin[g2])
  gap = wp.max(m.geom_gap[g1], m.geom_gap[g2])
  solmix1 = m.geom_solmix[g1]
  solmix2 = m.geom_solmix[g2]
  mix = solmix1 / (solmix1 + solmix2)
  mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
  mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
  mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)

  p1 = m.geom_priority[g1]
  p2 = m.geom_priority[g2]
  mix = wp.where(p1 == p2, mix, wp.where(p1 > p2, 1.0, 0.0))

  condim1 = m.geom_condim[g1]
  condim2 = m.geom_condim[g2]
  condim = wp.where(
    p1 == p2, wp.max(condim1, condim2), wp.where(p1 > p2, condim1, condim2)
  )
  d.contact.dim[tid] = condim

  if m.geom_solref[g1].x > 0.0 and m.geom_solref[g2].x > 0.0:
    d.contact.solref[tid] = mix * m.geom_solref[g1] + (1.0 - mix) * m.geom_solref[g2]
  else:
    d.contact.solref[tid] = wp.min(m.geom_solref[g1], m.geom_solref[g2])
  d.contact.includemargin[tid] = margin - gap
  friction_ = wp.max(m.geom_friction[g1], m.geom_friction[g2])
  friction5 = vec5(friction_[0], friction_[0], friction_[1], friction_[2], friction_[2])
  d.contact.friction[tid] = friction5
  d.contact.solimp[tid] = mix * m.geom_solimp[g1] + (1.0 - mix) * m.geom_solimp[g2]


def sap_broadphase(m: Model, d: Data):
  """Broadphase collision detection via sweep-and-prune."""

  # Use random fixed direction vector for now
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)

  wp.launch(
    kernel=broadphase_project_spheres_onto_sweep_direction_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d, direction],
    device=m.device,
  )

  tile_sort_available = False
  segmented_sort_available = hasattr(wp.utils, "segmented_sort_pairs")

  if tile_sort_available:
    segmented_sort_kernel = create_segmented_sort_kernel(m.ngeom)
    wp.launch_tiled(
      kernel=segmented_sort_kernel,
      dim=(d.nworld),
      inputs=[m, d],
      block_dim=128,
      device=m.device,
    )
    print("tile sort available")
  elif segmented_sort_available:
    wp.utils.segmented_sort_pairs(
      d.sap_projection_lower,
      d.sap_sort_index,
      m.ngeom * d.nworld,
      d.sap_segment_index,
    )
  else:
    # Sort each world's segment separately
    for world_id in range(d.nworld):
      start_idx = world_id * m.ngeom

      # Create temporary arrays for sorting
      temp_box_projections_lower = wp.zeros(
        m.ngeom * 2, dtype=d.sap_projection_lower.dtype, device=m.device
      )
      temp_box_sorting_indexer = wp.zeros(
        m.ngeom * 2, dtype=d.sap_sort_index.dtype, device=m.device
      )

      # Copy data to temporary arrays
      wp.copy(
        temp_box_projections_lower,
        d.sap_projection_lower,
        0,
        start_idx,
        m.ngeom,
      )
      wp.copy(
        temp_box_sorting_indexer,
        d.sap_sort_index,
        0,
        start_idx,
        m.ngeom,
      )

      # Sort the temporary arrays
      wp.utils.radix_sort_pairs(
        temp_box_projections_lower, temp_box_sorting_indexer, m.ngeom
      )

      # Copy sorted data back
      wp.copy(
        d.sap_projection_lower,
        temp_box_projections_lower,
        start_idx,
        0,
        m.ngeom,
      )
      wp.copy(
        d.sap_sort_index,
        temp_box_sorting_indexer,
        start_idx,
        0,
        m.ngeom,
      )

  wp.launch(
    kernel=reorder_bounding_spheres_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d],
    device=m.device,
  )

  wp.launch(
    kernel=sap_broadphase_prepare_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d],
    device=m.device,
  )

  # The scan (scan = cumulative sum, either inclusive or exclusive depending on the last argument) is used for load balancing among the threads
  wp.utils.array_scan(d.sap_range.reshape(-1), d.sap_cumulative_sum, True)

  # Estimate how many overlap checks need to be done - assumes each box has to be compared to 5 other boxes (and batched over all worlds)
  num_sweep_threads = 5 * d.nworld * m.ngeom
  filter_parent = not m.opt.disableflags & DisableBit.FILTERPARENT.value
  wp.launch(
    kernel=sap_broadphase_kernel,
    dim=num_sweep_threads,
    inputs=[m, d, num_sweep_threads, filter_parent],
    device=m.device,
  )

  return d


def nxn_broadphase(m: Model, d: Data):
  """Broadphase collision detective via brute-force search."""
  filterparent = not (m.opt.disableflags & DisableBit.FILTERPARENT.value)

  @wp.kernel
  def _nxn_broadphase(m: Model, d: Data):
    worldid, elementid = wp.tid()
    geom1 = (
      m.ngeom
      - 2
      - int(
        wp.sqrt(float(-8 * elementid + 4 * m.ngeom * (m.ngeom - 1) - 7)) / 2.0 - 0.5
      )
    )
    geom2 = (
      elementid
      + geom1
      + 1
      - m.ngeom * (m.ngeom - 1) // 2
      + (m.ngeom - geom1) * ((m.ngeom - geom1) - 1) // 2
    )

    margin1 = m.geom_margin[geom1]
    margin2 = m.geom_margin[geom2]
    pos1 = d.geom_xpos[worldid, geom1]
    pos2 = d.geom_xpos[worldid, geom2]
    size1 = m.geom_rbound[geom1]
    size2 = m.geom_rbound[geom2]

    bound = size1 + size2 + wp.max(margin1, margin2)
    dif = pos2 - pos1

    if size1 != 0.0 and size2 != 0.0:
      # neither geom is a plane
      dist_sq = wp.dot(dif, dif)
      bounds_filter = dist_sq <= bound * bound
    elif size1 == 0.0:
      # geom1 is a plane
      xmat1 = d.geom_xmat[worldid, geom1]
      dist = wp.dot(dif, wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2]))
      bounds_filter = dist <= bound
    else:
      # geom2 is a plane
      xmat2 = d.geom_xmat[worldid, geom2]
      dist = wp.dot(-dif, wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2]))
      bounds_filter = dist <= bound

    geom_filter = _geom_filter(m, geom1, geom2, filterparent)

    if bounds_filter and geom_filter:
      _add_geom_pair(m, d, geom1, geom2, worldid)

  wp.launch(
    _nxn_broadphase,
    dim=(d.nworld, m.ngeom * (m.ngeom - 1) // 2),
    inputs=[m, d],
    device=m.device,
  )


def get_contact_solver_params(m: Model, d: Data):
  wp.launch(
    get_contact_solver_params_kernel,
    dim=[d.nconmax],
    inputs=[m, d],
    device=m.device,
  )

  # TODO(team): do we need condim sorting, deepest penetrating contact here?


@event_scope
def collision(m: Model, d: Data):
  """Collision detection."""

  # AD: based on engine_collision_driver.py in Eric's warp fork/mjx-collisions-dev
  # which is further based on the CUDA code here:
  # https://github.com/btaba/mujoco/blob/warp-collisions/mjx/mujoco/mjx/_src/cuda/engine_collision_driver.cu.cc#L458-L583

  d.ncollision.zero_()
  d.ncon.zero_()

  if d.nconmax == 0:
    return

  dsbl_flgs = m.opt.disableflags
  if (dsbl_flgs & DisableBit.CONSTRAINT) | (dsbl_flgs & DisableBit.CONTACT):
    return

  # TODO(team): determine ngeom to switch from n^2 to sap
  if m.ngeom <= 100:
    nxn_broadphase(m, d)
  else:
    sap_broadphase(m, d)

  # TODO(team): we should reject far-away contacts in the narrowphase instead of constraint
  #             partitioning because we can move some pressure of the atomics
  gjk_narrowphase(m, d)
  primitive_narrowphase(m, d)
  box_box_narrowphase(m, d)

  get_contact_solver_params(m, d)
