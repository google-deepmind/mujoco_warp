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

from .collision_box import box_box_narrowphase
from .collision_convex import gjk_narrowphase
from .collision_primitive import primitive_narrowphase
from .types import MJ_MAXVAL
from .types import MJ_MINVAL
from .types import Data
from .types import DisableBit
from .types import Model
from .types import vec5
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


@wp.func
def _sphere_filter(m: Model, d: Data, geom1: int, geom2: int, worldid: int) -> bool:
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
    return dist_sq <= bound * bound
  elif size1 == 0.0:
    # geom1 is a plane
    xmat1 = d.geom_xmat[worldid, geom1]
    dist = wp.dot(dif, wp.vec3(xmat1[0, 2], xmat1[1, 2], xmat1[2, 2]))
    return dist <= bound
  else:
    # geom2 is a plane
    xmat2 = d.geom_xmat[worldid, geom2]
    dist = wp.dot(-dif, wp.vec3(xmat2[0, 2], xmat2[1, 2], xmat2[2, 2]))
    return dist <= bound


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


@wp.func
def _binary_search(
  values: wp.array(dtype=Any, ndim=1),
  value: Any,
  lower: int,
  upper: int,
) -> int:
  while lower < upper:
    mid = (lower + upper) >> 1
    if values[mid] > value:
      upper = mid
    else:
      lower = mid + 1

  return upper


@wp.kernel
def _sap_project(m: Model, d: Data, direction: wp.vec3):
  worldid, geomid = wp.tid()

  xpos = d.geom_xpos[worldid, geomid]
  rbound = m.geom_rbound[geomid]

  if rbound == 0.0:
    # geom is a plane
    rbound = MJ_MAXVAL

  radius = rbound + m.geom_margin[geomid]
  center = wp.dot(direction, xpos)

  d.sap_projection_lower[worldid, geomid] = center - radius
  d.sap_projection_upper[worldid, geomid] = center + radius
  d.sap_sort_index[worldid, geomid] = geomid


@wp.kernel
def _sap_range(m: Model, d: Data):
  worldid, geomid = wp.tid()

  # current bounding geom
  idx = d.sap_sort_index[worldid, geomid]

  upper = d.sap_projection_upper[worldid, idx]

  limit = _binary_search(d.sap_projection_lower[worldid], upper, geomid + 1, m.ngeom)
  limit = wp.min(m.ngeom - 1, limit)

  # range of geoms for the sweep and prune process
  d.sap_range[worldid, geomid] = limit - geomid


@wp.kernel
def _sap_broadphase(m: Model, d: Data, nsweep: int, filterparent: bool):
  worldgeomid = wp.tid()

  nworldgeom = d.nworld * m.ngeom
  nworkpackages = d.sap_cumulative_sum[nworldgeom - 1]

  while worldgeomid < nworkpackages:
    # binary search to find current and next geom pair indices
    i = _binary_search(d.sap_cumulative_sum, worldgeomid, 0, nworldgeom)
    j = i + worldgeomid + 1

    if i > 0:
      j -= d.sap_cumulative_sum[i - 1]

    worldid = i // m.ngeom
    i = i % m.ngeom
    j = j % m.ngeom

    # geom indices
    geom1 = d.sap_sort_index[worldid, i]
    geom2 = d.sap_sort_index[worldid, j]

    sphere_filter = _sphere_filter(m, d, geom1, geom2, worldid)
    geom_filter = _geom_filter(m, geom1, geom2, filterparent)

    if sphere_filter and geom_filter:
      _add_geom_pair(m, d, geom1, geom2, worldid)

    worldgeomid += nsweep


def sap_broadphase(m: Model, d: Data):
  """Broadphase collision detection via sweep-and-prune."""

  nworldgeom = d.nworld * m.ngeom

  # TODO(team): direction

  # random fixed direction
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)

  wp.launch(
    kernel=_sap_project,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d, direction],
  )

  # TODO(team): tile sort

  wp.utils.segmented_sort_pairs(
    d.sap_projection_lower,
    d.sap_sort_index,
    nworldgeom,
    d.sap_segment_index,
  )

  wp.launch(
    kernel=_sap_range,
    dim=(d.nworld, m.ngeom),
    inputs=[m, d],
  )

  # scan is used for load balancing among the threads
  wp.utils.array_scan(d.sap_range.reshape(-1), d.sap_cumulative_sum, True)

  # estimate number of overlap checks - assumes each geom has 5 other geoms (batched over all worlds)
  nsweep = 5 * nworldgeom
  filterparent = not m.opt.disableflags & DisableBit.FILTERPARENT.value
  wp.launch(
    kernel=_sap_broadphase,
    dim=nsweep,
    inputs=[m, d, nsweep, filterparent],
  )


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

    sphere_filter = _sphere_filter(m, d, geom1, geom2, worldid)
    geom_filter = _geom_filter(m, geom1, geom2, filterparent)

    if sphere_filter and geom_filter:
      _add_geom_pair(m, d, geom1, geom2, worldid)

  wp.launch(
    _nxn_broadphase, dim=(d.nworld, m.ngeom * (m.ngeom - 1) // 2), inputs=[m, d]
  )


def contact_params(m: Model, d: Data):
  @wp.kernel
  def _contact_params(
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
    friction5 = vec5(
      friction_[0], friction_[0], friction_[1], friction_[2], friction_[2]
    )
    d.contact.friction[tid] = friction5
    d.contact.solimp[tid] = mix * m.geom_solimp[g1] + (1.0 - mix) * m.geom_solimp[g2]

  wp.launch(_contact_params, dim=[d.nconmax], inputs=[m, d])


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

  contact_params(m, d)
