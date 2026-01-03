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

from .collision_gjk import ccd
from .collision_gjk import multicontact
from .collision_primitive import Geom
from .collision_primitive import contact_params
from .collision_primitive import geom_collision_pair
from .collision_primitive import write_contact
from .math import make_frame
from .math import upper_trid_index
from .types import MJ_MAX_EPAFACES
from .types import MJ_MAX_EPAHORIZON
from .types import MJ_MAXCONPAIR
from .types import MJ_MAXVAL
from .types import Data
from .types import GeomType
from .types import Model
from .types import mat43
from .types import mat63
from .types import vec5
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import nested_kernel

# TODO(team): improve compile time to enable backward pass
wp.set_module_options({"enable_backward": False})

vec_maxconpair = wp.types.vector(length=MJ_MAXCONPAIR, dtype=float)
mat_maxconpair = wp.types.matrix(shape=(MJ_MAXCONPAIR, 3), dtype=float)


def _arena_layout(epa_iterations: int, nmaxpolygon: int = 0, nmaxmeshdeg: int = 0):
  """Compute arena array sizes and offsets for a single collision.

  Returns dict with array names as keys and (offset, dim) tuples.
  All sizes are in number of 4-byte elements (int/float are both 4 bytes).

  Memory layout is optimized so multicontact arrays can overlap with EPA arrays
  that are not used by multicontact (epa_vert, epa_pr, epa_norm2, epa_index,
  epa_map, epa_horizon). The multicontact inputs are placed first:
  epa_vert1, epa_vert2, epa_vert_index1, epa_vert_index2, epa_face.
  """
  epa_vert_dim = 5 + epa_iterations
  epa_face_dim = 6 + MJ_MAX_EPAFACES * epa_iterations

  layout = {}
  offset = 0

  # === EPA arrays used by multicontact (placed first, not overwritten) ===
  # epa_vert1: vertices in EPA polytope in geom 1 space
  layout["epa_vert1"] = (offset, epa_vert_dim)
  offset += epa_vert_dim * 3
  # epa_vert2: vertices in EPA polytope in geom 2 space
  layout["epa_vert2"] = (offset, epa_vert_dim)
  offset += epa_vert_dim * 3
  # epa_vert_index1: vertex indices for geom 1
  layout["epa_vert_index1"] = (offset, epa_vert_dim)
  offset += epa_vert_dim
  # epa_vert_index2: vertex indices for geom 2
  layout["epa_vert_index2"] = (offset, epa_vert_dim)
  offset += epa_vert_dim
  # epa_face: faces of polytope (vec3i)
  layout["epa_face"] = (offset, epa_face_dim)
  offset += epa_face_dim * 3

  # Record offset where multicontact arrays can begin (after epa_face)
  multiccd_base_offset = offset

  # === EPA arrays NOT used by multicontact (can be overwritten) ===
  # epa_vert: vertices in Minkowski space
  layout["epa_vert"] = (offset, epa_vert_dim)
  offset += epa_vert_dim * 3
  # epa_pr: projection of origin on polytope faces
  layout["epa_pr"] = (offset, epa_face_dim)
  offset += epa_face_dim * 3
  # epa_norm2: epa_pr * epa_pr
  layout["epa_norm2"] = (offset, epa_face_dim)
  offset += epa_face_dim
  # epa_index: index of face in polytope map
  layout["epa_index"] = (offset, epa_face_dim)
  offset += epa_face_dim
  # epa_map: status of faces in polytope
  layout["epa_map"] = (offset, epa_face_dim)
  offset += epa_face_dim
  # epa_horizon: index pair of edges on horizon
  layout["epa_horizon"] = (offset, 2 * MJ_MAX_EPAHORIZON)
  offset += 2 * MJ_MAX_EPAHORIZON

  # Total EPA memory per collision
  epa_total = offset

  # === MultiCCD arrays (start at multiccd_base_offset, overlapping unused EPA) ===
  # These overwrite epa_vert, epa_pr, epa_norm2, epa_index, epa_map, epa_horizon
  mccd_offset = multiccd_base_offset

  layout["multiccd_polygon"] = (mccd_offset, 2 * nmaxpolygon)
  mccd_offset += 2 * nmaxpolygon * 3
  layout["multiccd_clipped"] = (mccd_offset, 2 * nmaxpolygon)
  mccd_offset += 2 * nmaxpolygon * 3
  layout["multiccd_pnormal"] = (mccd_offset, nmaxpolygon)
  mccd_offset += nmaxpolygon * 3
  layout["multiccd_pdist"] = (mccd_offset, nmaxpolygon)
  mccd_offset += nmaxpolygon
  layout["multiccd_idx1"] = (mccd_offset, nmaxmeshdeg)
  mccd_offset += nmaxmeshdeg
  layout["multiccd_idx2"] = (mccd_offset, nmaxmeshdeg)
  mccd_offset += nmaxmeshdeg
  layout["multiccd_n1"] = (mccd_offset, nmaxmeshdeg)
  mccd_offset += nmaxmeshdeg * 3
  layout["multiccd_n2"] = (mccd_offset, nmaxmeshdeg)
  mccd_offset += nmaxmeshdeg * 3
  layout["multiccd_endvert"] = (mccd_offset, nmaxmeshdeg)
  mccd_offset += nmaxmeshdeg * 3
  layout["multiccd_face1"] = (mccd_offset, nmaxpolygon)
  mccd_offset += nmaxpolygon * 3
  layout["multiccd_face2"] = (mccd_offset, nmaxpolygon)
  mccd_offset += nmaxpolygon * 3

  # Total is max of EPA-only and EPA+multicontact (with overlap)
  # Since multicontact overlaps with unused EPA, total = max(epa_total, mccd_offset)
  layout["_total"] = max(epa_total, mccd_offset)
  layout["_epa_vert_dim"] = epa_vert_dim
  layout["_epa_face_dim"] = epa_face_dim
  layout["_nmaxpolygon"] = nmaxpolygon
  layout["_nmaxmeshdeg"] = nmaxmeshdeg
  return layout


_CONVEX_COLLISION_PAIRS = [
  (GeomType.HFIELD, GeomType.SPHERE),
  (GeomType.HFIELD, GeomType.CAPSULE),
  (GeomType.HFIELD, GeomType.ELLIPSOID),
  (GeomType.HFIELD, GeomType.CYLINDER),
  (GeomType.HFIELD, GeomType.BOX),
  (GeomType.HFIELD, GeomType.MESH),
  (GeomType.SPHERE, GeomType.ELLIPSOID),
  (GeomType.SPHERE, GeomType.MESH),
  (GeomType.CAPSULE, GeomType.ELLIPSOID),
  (GeomType.CAPSULE, GeomType.CYLINDER),
  (GeomType.CAPSULE, GeomType.MESH),
  (GeomType.ELLIPSOID, GeomType.ELLIPSOID),
  (GeomType.ELLIPSOID, GeomType.CYLINDER),
  (GeomType.ELLIPSOID, GeomType.BOX),
  (GeomType.ELLIPSOID, GeomType.MESH),
  (GeomType.CYLINDER, GeomType.CYLINDER),
  (GeomType.CYLINDER, GeomType.BOX),
  (GeomType.CYLINDER, GeomType.MESH),
  (GeomType.BOX, GeomType.MESH),
  (GeomType.MESH, GeomType.MESH),
]


def _check_convex_collision_pairs():
  prev_idx = -1
  for pair in _CONVEX_COLLISION_PAIRS:
    idx = upper_trid_index(len(GeomType), pair[0].value, pair[1].value)
    if pair[1] < pair[0] or idx <= prev_idx:
      return False
    prev_idx = idx
  return True


assert _check_convex_collision_pairs(), "_CONVEX_COLLISION_PAIRS is in invalid order."


@wp.func
def _hfield_filter(
  # Model:
  geom_dataid: wp.array(dtype=int),
  geom_aabb: wp.array3d(dtype=wp.vec3),
  geom_rbound: wp.array2d(dtype=float),
  geom_margin: wp.array2d(dtype=float),
  hfield_size: wp.array(dtype=wp.vec4),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  worldid: int,
  g1: int,
  g2: int,
) -> Tuple[bool, float, float, float, float, float, float]:
  """Filter for height field collisions.

  See MuJoCo mjc_ConvexHField.
  """
  # height field info
  hfdataid = geom_dataid[g1]
  size1 = hfield_size[hfdataid]

  # geom info
  rbound_id = worldid % geom_rbound.shape[0]
  margin_id = worldid % geom_margin.shape[0]

  pos1 = geom_xpos_in[worldid, g1]
  mat1 = geom_xmat_in[worldid, g1]
  mat1T = wp.transpose(mat1)
  pos2 = geom_xpos_in[worldid, g2]
  pos = mat1T @ (pos2 - pos1)
  r2 = geom_rbound[rbound_id, g2]

  # TODO(team): margin?
  margin = wp.max(geom_margin[margin_id, g1], geom_margin[margin_id, g2])

  # box-sphere test: horizontal plane
  for i in range(2):
    if (size1[i] < pos[i] - r2 - margin) or (-size1[i] > pos[i] + r2 + margin):
      return True, wp.inf, wp.inf, wp.inf, wp.inf, wp.inf, wp.inf

  # box-sphere test: vertical direction
  if size1[2] < pos[2] - r2 - margin:  # up
    return True, wp.inf, wp.inf, wp.inf, wp.inf, wp.inf, wp.inf

  if -size1[3] > pos[2] + r2 + margin:  # down
    return True, wp.inf, wp.inf, wp.inf, wp.inf, wp.inf, wp.inf

  mat2 = geom_xmat_in[worldid, g2]
  mat = mat1T @ mat2

  # aabb for geom in height field frame
  xmax = -MJ_MAXVAL
  ymax = -MJ_MAXVAL
  zmax = -MJ_MAXVAL
  xmin = MJ_MAXVAL
  ymin = MJ_MAXVAL
  zmin = MJ_MAXVAL

  aabb_id = worldid % geom_aabb.shape[0]
  center2 = geom_aabb[aabb_id, g2, 0]
  size2 = geom_aabb[aabb_id, g2, 1]

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
    return True, wp.inf, wp.inf, wp.inf, wp.inf, wp.inf, wp.inf
  else:
    return False, xmin, xmax, ymin, ymax, zmin, zmax


@cache_kernel
def ccd_hfield_kernel_builder(
  geomtype1: int,
  geomtype2: int,
  gjk_iterations: int,
  epa_iterations: int,
  geomgeomid: int,
):
  """Kernel builder for heightfield CCD collisions (no multiccd args)."""
  # Compute arena layout at build time - these become compile-time constants
  layout = _arena_layout(epa_iterations)
  per_collision_size = layout["_total"]
  epa_vert_dim = layout["_epa_vert_dim"]
  epa_face_dim = layout["_epa_face_dim"]

  # Extract offsets (will be embedded via wp.static)
  epa_vert_offset = layout["epa_vert"][0]
  epa_vert1_offset = layout["epa_vert1"][0]
  epa_vert2_offset = layout["epa_vert2"][0]
  epa_vert_index1_offset = layout["epa_vert_index1"][0]
  epa_vert_index2_offset = layout["epa_vert_index2"][0]
  epa_face_offset = layout["epa_face"][0]
  epa_pr_offset = layout["epa_pr"][0]
  epa_norm2_offset = layout["epa_norm2"][0]
  epa_index_offset = layout["epa_index"][0]
  epa_map_offset = layout["epa_map"][0]
  epa_horizon_offset = layout["epa_horizon"][0]

  # runs convex collision on a set of geom pairs to recover contact info
  @nested_kernel(module="unique", enable_backward=False)
  def ccd_hfield_kernel(
    # Model:
    opt_ccd_tolerance: wp.array(dtype=float),
    geom_type: wp.array(dtype=int),
    geom_condim: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_aabb: wp.array3d(dtype=wp.vec3),
    geom_rbound: wp.array2d(dtype=float),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
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
    pair_dim: wp.array(dtype=int),
    pair_solref: wp.array2d(dtype=wp.vec2),
    pair_solreffriction: wp.array2d(dtype=wp.vec2),
    pair_solimp: wp.array2d(dtype=vec5),
    pair_margin: wp.array2d(dtype=float),
    pair_gap: wp.array2d(dtype=float),
    pair_friction: wp.array2d(dtype=vec5),
    # Data in:
    naconmax_in: int,
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    collision_pair_in: wp.array(dtype=wp.vec2i),
    collision_pairid_in: wp.array(dtype=wp.vec2i),
    collision_worldid_in: wp.array(dtype=int),
    ncollision_in: wp.array(dtype=int),
    naccdmax_in: int,
    nccd_in: wp.array(dtype=int),
    arena_in: wp.array(dtype=int),
    # Data out:
    nacon_out: wp.array(dtype=int),
    contact_dist_out: wp.array(dtype=float),
    contact_pos_out: wp.array(dtype=wp.vec3),
    contact_frame_out: wp.array(dtype=wp.mat33),
    contact_includemargin_out: wp.array(dtype=float),
    contact_friction_out: wp.array(dtype=vec5),
    contact_solref_out: wp.array(dtype=wp.vec2),
    contact_solreffriction_out: wp.array(dtype=wp.vec2),
    contact_solimp_out: wp.array(dtype=vec5),
    contact_dim_out: wp.array(dtype=int),
    contact_geom_out: wp.array(dtype=wp.vec2i),
    contact_worldid_out: wp.array(dtype=int),
    contact_type_out: wp.array(dtype=int),
    contact_geomcollisionid_out: wp.array(dtype=int),
  ):
    tid = wp.tid()
    if tid >= ncollision_in[0]:
      return

    geoms = collision_pair_in[tid]
    g1 = geoms[0]
    g2 = geoms[1]

    if geom_type[g1] != geomtype1 or geom_type[g2] != geomtype2:
      return

    worldid = collision_worldid_in[tid]

    # height field filter
    no_hf_collision, xmin, xmax, ymin, ymax, zmin, zmax = _hfield_filter(
      geom_dataid, geom_aabb, geom_rbound, geom_margin, hfield_size, geom_xpos_in, geom_xmat_in, worldid, g1, g2
    )
    if no_hf_collision:
      return

    # Allocate arena slot using atomic add with static geom pair index
    arenaid = wp.atomic_add(nccd_in, wp.static(geomgeomid), 1)
    if arenaid >= naccdmax_in:
      wp.printf("CCD arena overflow - please increase naccdmax to %u\n", arenaid)
      return

    _, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
      geom_condim,
      geom_priority,
      geom_solmix,
      geom_solref,
      geom_solimp,
      geom_friction,
      geom_margin,
      geom_gap,
      pair_dim,
      pair_solref,
      pair_solreffriction,
      pair_solimp,
      pair_margin,
      pair_gap,
      pair_friction,
      collision_pair_in,
      collision_pairid_in,
      tid,
      worldid,
    )

    geom1, geom2 = geom_collision_pair(
      geom_type,
      geom_dataid,
      geom_size,
      mesh_vertadr,
      mesh_vertnum,
      mesh_graphadr,
      mesh_vert,
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
      geoms,
      worldid,
    )

    # see MuJoCo mjc_ConvexHField
    geom1_dataid = geom_dataid[g1]

    # height field subgrid
    nrow = hfield_nrow[geom1_dataid]
    ncol = hfield_ncol[geom1_dataid]
    size = hfield_size[geom1_dataid]

    # subgrid
    x_scale = 0.5 * float(ncol - 1) / size[0]
    y_scale = 0.5 * float(nrow - 1) / size[1]
    cmin = wp.max(0, int(wp.floor((xmin + size[0]) * x_scale)))
    cmax = wp.min(ncol - 1, int(wp.ceil((xmax + size[0]) * x_scale)))
    rmin = wp.max(0, int(wp.floor((ymin + size[1]) * y_scale)))
    rmax = wp.min(nrow - 1, int(wp.ceil((ymax + size[1]) * y_scale)))

    dx = (2.0 * size[0]) / float(ncol - 1)
    dy = (2.0 * size[1]) / float(nrow - 1)
    dr = wp.vec2i(1, 0)

    prism = mat63()

    # set zbottom value using base size
    prism[0, 2] = -size[3]
    prism[1, 2] = -size[3]
    prism[2, 2] = -size[3]

    adr = hfield_adr[geom1_dataid]

    hfield_contact_dist = vec_maxconpair()
    hfield_contact_pos = mat_maxconpair()
    hfield_contact_normal = mat_maxconpair()
    min_dist = float(wp.inf)
    min_normal = wp.vec3(wp.inf, wp.inf, wp.inf)
    min_pos = wp.vec3(wp.inf, wp.inf, wp.inf)
    min_id = int(-1)

    # TODO(team): height field margin?
    geom1.margin = margin
    geom2.margin = margin

    # Construct EPA arrays from arena memory
    # multicontact inputs first (epa_vert1, epa_vert2, epa_vert_index1, epa_vert_index2, epa_face)
    base_offset = arenaid * wp.static(per_collision_size)

    epa_vert1 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_vert1_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(epa_vert_dim),),
    )
    epa_vert2 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_vert2_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(epa_vert_dim),),
    )
    epa_vert_index1 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_vert_index1_offset)) * 4),
      dtype=int,
      shape=(wp.static(epa_vert_dim),),
    )
    epa_vert_index2 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_vert_index2_offset)) * 4),
      dtype=int,
      shape=(wp.static(epa_vert_dim),),
    )
    epa_face = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_face_offset)) * 4),
      dtype=wp.vec3i,
      shape=(wp.static(epa_face_dim),),
    )
    # Remaining EPA arrays (not used by multicontact)
    epa_vert = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_vert_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(epa_vert_dim),),
    )
    epa_pr = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_pr_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(epa_face_dim),),
    )
    epa_norm2 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_norm2_offset)) * 4),
      dtype=float,
      shape=(wp.static(epa_face_dim),),
    )
    epa_index = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_index_offset)) * 4),
      dtype=int,
      shape=(wp.static(epa_face_dim),),
    )
    epa_map = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_map_offset)) * 4),
      dtype=int,
      shape=(wp.static(epa_face_dim),),
    )
    epa_horizon = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_horizon_offset)) * 4),
      dtype=int,
      shape=(wp.static(2 * MJ_MAX_EPAHORIZON),),
    )

    collision_pairid = collision_pairid_in[tid]

    # process all prisms in subgrid
    count = int(0)
    for r in range(rmin, rmax):
      nvert = int(0)
      for c in range(cmin, cmax + 1):
        # add both triangles from this cell
        for i in range(2):
          if count >= MJ_MAXCONPAIR:
            wp.printf(
              "height field collision overflow, number of collisions >= %u - please adjust resolution: \n decrease the number of hfield rows/cols or modify size of colliding geom\n",
              MJ_MAXCONPAIR,
            )
            continue

          # add vert
          x = dx * float(c) - size[0]
          y = dy * float(r + dr[i]) - size[1]
          z = hfield_data[adr + (r + dr[i]) * ncol + c] * size[2] + margin

          prism[0] = prism[1]
          prism[1] = prism[2]
          prism[3] = prism[4]
          prism[4] = prism[5]

          prism[2, 0] = x
          prism[5, 0] = x
          prism[2, 1] = y
          prism[5, 1] = y
          prism[5, 2] = z

          nvert += 1

          if nvert <= 2:
            continue

          # prism height test
          if prism[3, 2] < zmin and prism[4, 2] < zmin and prism[5, 2] < zmin:
            continue

          geom1.hfprism = prism

          # prism center
          x1 = geom1.pos
          x1_ = wp.vec3(0.0, 0.0, 0.0)
          for i in range(6):
            x1_ += prism[i]
          x1 += geom1.rot @ (x1_ / 6.0)

          dist, ncontact, w1, w2, idx = ccd(
            opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]],
            0.0,
            gjk_iterations,
            epa_iterations,
            geom1,
            geom2,
            geomtype1,
            geomtype2,
            x1,
            geom2.pos,
            epa_vert,
            epa_vert1,
            epa_vert2,
            epa_vert_index1,
            epa_vert_index2,
            epa_face,
            epa_pr,
            epa_norm2,
            epa_index,
            epa_map,
            epa_horizon,
          )

          if ncontact == 0:
            continue

          # cache contact information
          hfield_contact_dist[count] = dist

          pos = 0.5 * (w1 + w2)
          hfield_contact_pos[count, 0] = pos[0]
          hfield_contact_pos[count, 1] = pos[1]
          hfield_contact_pos[count, 2] = pos[2]

          frame = make_frame(w1 - w2)
          normal = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
          hfield_contact_normal[count, 0] = normal[0]
          hfield_contact_normal[count, 1] = normal[1]
          hfield_contact_normal[count, 2] = normal[2]

          # contact with minimum distance
          if dist < min_dist:
            min_dist = dist
            min_normal = normal
            min_pos = pos
            min_id = count

          count += 1

    # contact 0: minimum distance
    write_contact(
      naconmax_in,
      0,
      min_dist,
      min_pos,
      make_frame(min_normal),
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
      collision_pairid,
      worldid,
      contact_dist_out,
      contact_pos_out,
      contact_frame_out,
      contact_includemargin_out,
      contact_friction_out,
      contact_solref_out,
      contact_solreffriction_out,
      contact_solimp_out,
      contact_dim_out,
      contact_geom_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      nacon_out,
    )

    # TODO(team): routine for select subset of contacts
    # TODO(team): if use_multiccd?
    if wp.static(True):
      MIN_DIST_TO_NEXT_CONTACT = 1.0e-3

      # contact 1: furthest from minimum distance contact
      id1 = int(-1)
      dist1 = float(-wp.inf)
      for i in range(count):
        if i == min_id:
          continue

        hf_pos = wp.vec3(hfield_contact_pos[i, 0], hfield_contact_pos[i, 1], hfield_contact_pos[i, 2])
        dist = wp.norm_l2(hf_pos - min_pos)

        if dist > dist1:
          id1 = i
          dist1 = dist

      if id1 == -1 or (0.0 < dist1 and dist1 < MIN_DIST_TO_NEXT_CONTACT):
        return

      pos1 = wp.vec3(hfield_contact_pos[id1, 0], hfield_contact_pos[id1, 1], hfield_contact_pos[id1, 2])
      normal1 = wp.vec3(hfield_contact_normal[id1, 0], hfield_contact_normal[id1, 1], hfield_contact_normal[id1, 2])

      write_contact(
        naconmax_in,
        1,
        hfield_contact_dist[id1],
        pos1,
        make_frame(normal1),
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        geoms,
        collision_pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
      )

      # contact 2: point furthest from min_pos - pos1 line
      dist_min1 = wp.cross(min_normal, min_pos - pos1)

      id2 = int(-1)
      dist_12 = float(-wp.inf)
      for i in range(count):
        if i == min_id or i == id1:
          continue

        hf_pos = wp.vec3(hfield_contact_pos[i, 0], hfield_contact_pos[i, 1], hfield_contact_pos[i, 2])
        dist = wp.abs(wp.dot(hf_pos - min_pos, dist_min1))

        if dist > dist_12:
          id2 = i
          dist_12 = dist

      if id2 == -1 or (0.0 < dist_12 and dist_12 < MIN_DIST_TO_NEXT_CONTACT):
        return

      pos2 = wp.vec3(hfield_contact_pos[id2, 0], hfield_contact_pos[id2, 1], hfield_contact_pos[id2, 2])
      normal2 = wp.vec3(hfield_contact_normal[id2, 0], hfield_contact_normal[id2, 1], hfield_contact_normal[id2, 2])

      write_contact(
        naconmax_in,
        2,
        hfield_contact_dist[id2],
        pos2,
        make_frame(normal2),
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        geoms,
        collision_pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
      )

      # contact 3: point furthest from other triangle edge
      vec_min2 = wp.cross(min_normal, min_pos - pos2)
      vec_12 = wp.cross(min_normal, pos1 - pos2)

      id3 = int(-1)
      dist3 = float(-wp.inf)
      for i in range(count):
        if i == min_id or i == id1 or i == id2:
          continue

        hf_pos = wp.vec3(hfield_contact_pos[i, 0], hfield_contact_pos[i, 1], hfield_contact_pos[i, 2])
        dist = wp.abs(wp.dot(hf_pos - min_pos, vec_min2)) + wp.abs(wp.dot(pos1 - hf_pos, vec_12))

        if dist > dist3:
          id3 = i
          dist3 = dist

      if id3 == -1 or (0.0 < dist3 and dist3 < MIN_DIST_TO_NEXT_CONTACT):
        return

      pos3 = wp.vec3(hfield_contact_pos[id3, 0], hfield_contact_pos[id3, 1], hfield_contact_pos[id3, 2])
      normal3 = wp.vec3(hfield_contact_normal[id3, 0], hfield_contact_normal[id3, 1], hfield_contact_normal[id3, 2])

      write_contact(
        naconmax_in,
        3,
        hfield_contact_dist[id3],
        pos3,
        make_frame(normal3),
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        geoms,
        collision_pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
      )

  return ccd_hfield_kernel


@cache_kernel
def ccd_kernel_builder(
  geomtype1: int,
  geomtype2: int,
  gjk_iterations: int,
  epa_iterations: int,
  geomgeomid: int,
  use_multiccd: bool,
  nmaxpolygon: int = 0,
  nmaxmeshdeg: int = 0,
):
  """Kernel builder for non-heightfield CCD collisions (no hfield args)."""
  # Compute arena layout at build time - these become compile-time constants
  layout = _arena_layout(epa_iterations, nmaxpolygon, nmaxmeshdeg)
  per_collision_size = layout["_total"]
  epa_vert_dim = layout["_epa_vert_dim"]
  epa_face_dim = layout["_epa_face_dim"]

  # Extract EPA offsets (will be embedded via wp.static)
  epa_vert_offset = layout["epa_vert"][0]
  epa_vert1_offset = layout["epa_vert1"][0]
  epa_vert2_offset = layout["epa_vert2"][0]
  epa_vert_index1_offset = layout["epa_vert_index1"][0]
  epa_vert_index2_offset = layout["epa_vert_index2"][0]
  epa_face_offset = layout["epa_face"][0]
  epa_pr_offset = layout["epa_pr"][0]
  epa_norm2_offset = layout["epa_norm2"][0]
  epa_index_offset = layout["epa_index"][0]
  epa_map_offset = layout["epa_map"][0]
  epa_horizon_offset = layout["epa_horizon"][0]

  # Extract multicontact offsets
  multiccd_polygon_offset = layout["multiccd_polygon"][0]
  multiccd_polygon_dim = layout["multiccd_polygon"][1]
  multiccd_clipped_offset = layout["multiccd_clipped"][0]
  multiccd_pnormal_offset = layout["multiccd_pnormal"][0]
  multiccd_pnormal_dim = layout["multiccd_pnormal"][1]
  multiccd_pdist_offset = layout["multiccd_pdist"][0]
  multiccd_idx1_offset = layout["multiccd_idx1"][0]
  multiccd_idx1_dim = layout["multiccd_idx1"][1]
  multiccd_idx2_offset = layout["multiccd_idx2"][0]
  multiccd_n1_offset = layout["multiccd_n1"][0]
  multiccd_n2_offset = layout["multiccd_n2"][0]
  multiccd_endvert_offset = layout["multiccd_endvert"][0]
  multiccd_face1_offset = layout["multiccd_face1"][0]
  multiccd_face1_dim = layout["multiccd_face1"][1]
  multiccd_face2_offset = layout["multiccd_face2"][0]

  @wp.func
  def eval_ccd_write_contact(
    # Model:
    opt_ccd_tolerance: wp.array(dtype=float),
    geom_type: wp.array(dtype=int),
    # Data in:
    naconmax_in: int,
    # In:
    epa_vert_in: wp.array(dtype=wp.vec3),
    epa_vert1_in: wp.array(dtype=wp.vec3),
    epa_vert2_in: wp.array(dtype=wp.vec3),
    epa_vert_index1_in: wp.array(dtype=int),
    epa_vert_index2_in: wp.array(dtype=int),
    epa_face_in: wp.array(dtype=wp.vec3i),
    epa_pr_in: wp.array(dtype=wp.vec3),
    epa_norm2_in: wp.array(dtype=float),
    epa_index_in: wp.array(dtype=int),
    epa_map_in: wp.array(dtype=int),
    epa_horizon_in: wp.array(dtype=int),
    multiccd_polygon_in: wp.array(dtype=wp.vec3),
    multiccd_clipped_in: wp.array(dtype=wp.vec3),
    multiccd_pnormal_in: wp.array(dtype=wp.vec3),
    multiccd_pdist_in: wp.array(dtype=float),
    multiccd_idx1_in: wp.array(dtype=int),
    multiccd_idx2_in: wp.array(dtype=int),
    multiccd_n1_in: wp.array(dtype=wp.vec3),
    multiccd_n2_in: wp.array(dtype=wp.vec3),
    multiccd_endvert_in: wp.array(dtype=wp.vec3),
    multiccd_face1_in: wp.array(dtype=wp.vec3),
    multiccd_face2_in: wp.array(dtype=wp.vec3),
    geom1: Geom,
    geom2: Geom,
    geoms: wp.vec2i,
    worldid: int,
    tid: int,
    margin: float,
    gap: float,
    condim: int,
    friction: vec5,
    solref: wp.vec2,
    solreffriction: wp.vec2,
    solimp: vec5,
    x1: wp.vec3,
    x2: wp.vec3,
    count: int,
    pairid: wp.vec2i,
    # Data out:
    contact_dist_out: wp.array(dtype=float),
    contact_pos_out: wp.array(dtype=wp.vec3),
    contact_frame_out: wp.array(dtype=wp.mat33),
    contact_includemargin_out: wp.array(dtype=float),
    contact_friction_out: wp.array(dtype=vec5),
    contact_solref_out: wp.array(dtype=wp.vec2),
    contact_solreffriction_out: wp.array(dtype=wp.vec2),
    contact_solimp_out: wp.array(dtype=vec5),
    contact_dim_out: wp.array(dtype=int),
    contact_geom_out: wp.array(dtype=wp.vec2i),
    contact_worldid_out: wp.array(dtype=int),
    contact_type_out: wp.array(dtype=int),
    contact_geomcollisionid_out: wp.array(dtype=int),
    nacon_out: wp.array(dtype=int),
  ) -> int:
    points = mat43()
    witness1 = mat43()
    witness2 = mat43()
    geom1.margin = margin
    geom2.margin = margin
    if pairid[1] >= 0:
      # if collision sensor, set large cutoff to work with various sensor cutoff values
      cutoff = 1.0e32
    else:
      cutoff = 0.0
    dist, ncontact, w1, w2, idx = ccd(
      opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]],
      cutoff,
      gjk_iterations,
      epa_iterations,
      geom1,
      geom2,
      geomtype1,
      geomtype2,
      x1,
      x2,
      epa_vert_in,
      epa_vert1_in,
      epa_vert2_in,
      epa_vert_index1_in,
      epa_vert_index2_in,
      epa_face_in,
      epa_pr_in,
      epa_norm2_in,
      epa_index_in,
      epa_map_in,
      epa_horizon_in,
    )

    if dist >= 0.0 and pairid[1] == -1:
      return 0

    witness1[0] = w1
    witness2[0] = w2

    if wp.static(use_multiccd):
      if (
        geom1.margin == 0.0
        and geom2.margin == 0.0
        and (geomtype1 == GeomType.BOX or (geomtype1 == GeomType.MESH and geom1.mesh_polyadr > -1))
        and (geomtype2 == GeomType.BOX or (geomtype2 == GeomType.MESH and geom2.mesh_polyadr > -1))
      ):
        ncontact, witness1, witness2 = multicontact(
          multiccd_polygon_in,
          multiccd_clipped_in,
          multiccd_pnormal_in,
          multiccd_pdist_in,
          multiccd_idx1_in,
          multiccd_idx2_in,
          multiccd_n1_in,
          multiccd_n2_in,
          multiccd_endvert_in,
          multiccd_face1_in,
          multiccd_face2_in,
          epa_vert1_in,
          epa_vert2_in,
          epa_vert_index1_in,
          epa_vert_index2_in,
          epa_face_in[idx],
          w1,
          w2,
          geom1,
          geom2,
          geomtype1,
          geomtype2,
        )

    for i in range(ncontact):
      points[i] = 0.5 * (witness1[i] + witness2[i])
    normal = witness1[0] - witness2[0]
    frame = make_frame(normal)

    # flip if collision sensor
    if pairid[1] >= 0:
      frame *= -1.0
      geoms = wp.vec2i(geoms[1], geoms[0])

    for i in range(ncontact):
      write_contact(
        naconmax_in,
        i,
        dist,
        points[i],
        frame,
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        geoms,
        pairid,
        worldid,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_worldid_out,
        contact_type_out,
        contact_geomcollisionid_out,
        nacon_out,
      )
      if count + (i + 1) >= MJ_MAXCONPAIR:
        return i + 1

    return ncontact

  # runs convex collision on a set of geom pairs to recover contact info (non-heightfield)
  @nested_kernel(module="unique", enable_backward=False)
  def ccd_kernel(
    # Model:
    opt_ccd_tolerance: wp.array(dtype=float),
    geom_type: wp.array(dtype=int),
    geom_condim: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
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
    pair_dim: wp.array(dtype=int),
    pair_solref: wp.array2d(dtype=wp.vec2),
    pair_solreffriction: wp.array2d(dtype=wp.vec2),
    pair_solimp: wp.array2d(dtype=vec5),
    pair_margin: wp.array2d(dtype=float),
    pair_gap: wp.array2d(dtype=float),
    pair_friction: wp.array2d(dtype=vec5),
    # Data in:
    naconmax_in: int,
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    collision_pair_in: wp.array(dtype=wp.vec2i),
    collision_pairid_in: wp.array(dtype=wp.vec2i),
    collision_worldid_in: wp.array(dtype=int),
    ncollision_in: wp.array(dtype=int),
    naccdmax_in: int,
    nccd_in: wp.array(dtype=int),
    arena_in: wp.array(dtype=int),
    # Data out:
    nacon_out: wp.array(dtype=int),
    contact_dist_out: wp.array(dtype=float),
    contact_pos_out: wp.array(dtype=wp.vec3),
    contact_frame_out: wp.array(dtype=wp.mat33),
    contact_includemargin_out: wp.array(dtype=float),
    contact_friction_out: wp.array(dtype=vec5),
    contact_solref_out: wp.array(dtype=wp.vec2),
    contact_solreffriction_out: wp.array(dtype=wp.vec2),
    contact_solimp_out: wp.array(dtype=vec5),
    contact_dim_out: wp.array(dtype=int),
    contact_geom_out: wp.array(dtype=wp.vec2i),
    contact_worldid_out: wp.array(dtype=int),
    contact_type_out: wp.array(dtype=int),
    contact_geomcollisionid_out: wp.array(dtype=int),
  ):
    tid = wp.tid()
    if tid >= ncollision_in[0]:
      return

    geoms = collision_pair_in[tid]
    g1 = geoms[0]
    g2 = geoms[1]

    if geom_type[g1] != geomtype1 or geom_type[g2] != geomtype2:
      return

    # Allocate arena slot using atomic add with static kernel index
    arenaid = wp.atomic_add(nccd_in, wp.static(geomgeomid), 1)
    if arenaid >= naccdmax_in:
      wp.printf("CCD arena overflow - please increase naccdmax to %u\n", arenaid)
      return

    worldid = collision_worldid_in[tid]

    _, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
      geom_condim,
      geom_priority,
      geom_solmix,
      geom_solref,
      geom_solimp,
      geom_friction,
      geom_margin,
      geom_gap,
      pair_dim,
      pair_solref,
      pair_solreffriction,
      pair_solimp,
      pair_margin,
      pair_gap,
      pair_friction,
      collision_pair_in,
      collision_pairid_in,
      tid,
      worldid,
    )

    geom1, geom2 = geom_collision_pair(
      geom_type,
      geom_dataid,
      geom_size,
      mesh_vertadr,
      mesh_vertnum,
      mesh_graphadr,
      mesh_vert,
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
      geoms,
      worldid,
    )

    # Construct EPA arrays from arena using arenaid-based offset
    # multicontact inputs first (epa_vert1, epa_vert2, epa_vert_index1, epa_vert_index2, epa_face)
    base_offset = arenaid * wp.static(per_collision_size)

    epa_vert1 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_vert1_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(epa_vert_dim),),
    )
    epa_vert2 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_vert2_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(epa_vert_dim),),
    )
    epa_vert_index1 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_vert_index1_offset)) * 4),
      dtype=int,
      shape=(wp.static(epa_vert_dim),),
    )
    epa_vert_index2 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_vert_index2_offset)) * 4),
      dtype=int,
      shape=(wp.static(epa_vert_dim),),
    )
    epa_face = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_face_offset)) * 4),
      dtype=wp.vec3i,
      shape=(wp.static(epa_face_dim),),
    )
    # Remaining EPA arrays (not used by multicontact - can be overwritten)
    epa_vert = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_vert_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(epa_vert_dim),),
    )
    epa_pr = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_pr_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(epa_face_dim),),
    )
    epa_norm2 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_norm2_offset)) * 4),
      dtype=float,
      shape=(wp.static(epa_face_dim),),
    )
    epa_index = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_index_offset)) * 4),
      dtype=int,
      shape=(wp.static(epa_face_dim),),
    )
    epa_map = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_map_offset)) * 4),
      dtype=int,
      shape=(wp.static(epa_face_dim),),
    )
    epa_horizon = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(epa_horizon_offset)) * 4),
      dtype=int,
      shape=(wp.static(2 * MJ_MAX_EPAHORIZON),),
    )

    # Construct multicontact arrays from arena
    multiccd_polygon = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_polygon_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(multiccd_polygon_dim),),
    )
    multiccd_clipped = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_clipped_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(multiccd_polygon_dim),),
    )
    multiccd_pnormal = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_pnormal_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(multiccd_pnormal_dim),),
    )
    multiccd_pdist = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_pdist_offset)) * 4),
      dtype=float,
      shape=(wp.static(multiccd_pnormal_dim),),
    )
    multiccd_idx1 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_idx1_offset)) * 4),
      dtype=int,
      shape=(wp.static(multiccd_idx1_dim),),
    )
    multiccd_idx2 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_idx2_offset)) * 4),
      dtype=int,
      shape=(wp.static(multiccd_idx1_dim),),
    )
    multiccd_n1 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_n1_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(multiccd_idx1_dim),),
    )
    multiccd_n2 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_n2_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(multiccd_idx1_dim),),
    )
    multiccd_endvert = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_endvert_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(multiccd_idx1_dim),),
    )
    multiccd_face1 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_face1_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(multiccd_face1_dim),),
    )
    multiccd_face2 = wp.array(
      ptr=arena_in.ptr + wp.uint64((base_offset + wp.static(multiccd_face2_offset)) * 4),
      dtype=wp.vec3,
      shape=(wp.static(multiccd_face1_dim),),
    )

    eval_ccd_write_contact(
      opt_ccd_tolerance,
      geom_type,
      naconmax_in,
      epa_vert,
      epa_vert1,
      epa_vert2,
      epa_vert_index1,
      epa_vert_index2,
      epa_face,
      epa_pr,
      epa_norm2,
      epa_index,
      epa_map,
      epa_horizon,
      multiccd_polygon,
      multiccd_clipped,
      multiccd_pnormal,
      multiccd_pdist,
      multiccd_idx1,
      multiccd_idx2,
      multiccd_n1,
      multiccd_n2,
      multiccd_endvert,
      multiccd_face1,
      multiccd_face2,
      geom1,
      geom2,
      geoms,
      worldid,
      tid,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geom1.pos,
      geom2.pos,
      0,
      collision_pairid_in[tid],
      contact_dist_out,
      contact_pos_out,
      contact_frame_out,
      contact_includemargin_out,
      contact_friction_out,
      contact_solref_out,
      contact_solreffriction_out,
      contact_solimp_out,
      contact_dim_out,
      contact_geom_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      nacon_out,
    )

  return ccd_kernel


# Heightfield collision pairs handled by ccd_hfield_kernel_builder
_HFIELD_COLLISION_PAIRS = [
  (GeomType.HFIELD, GeomType.SPHERE),
  (GeomType.HFIELD, GeomType.CAPSULE),
  (GeomType.HFIELD, GeomType.ELLIPSOID),
  (GeomType.HFIELD, GeomType.CYLINDER),
  (GeomType.HFIELD, GeomType.BOX),
  (GeomType.HFIELD, GeomType.MESH),
]

# Non-heightfield collision pairs handled by ccd_kernel_builder
_NON_HFIELD_COLLISION_PAIRS = [
  (GeomType.SPHERE, GeomType.ELLIPSOID),
  (GeomType.SPHERE, GeomType.MESH),
  (GeomType.CAPSULE, GeomType.ELLIPSOID),
  (GeomType.CAPSULE, GeomType.CYLINDER),
  (GeomType.CAPSULE, GeomType.MESH),
  (GeomType.ELLIPSOID, GeomType.ELLIPSOID),
  (GeomType.ELLIPSOID, GeomType.CYLINDER),
  (GeomType.ELLIPSOID, GeomType.BOX),
  (GeomType.ELLIPSOID, GeomType.MESH),
  (GeomType.CYLINDER, GeomType.CYLINDER),
  (GeomType.CYLINDER, GeomType.BOX),
  (GeomType.CYLINDER, GeomType.MESH),
  (GeomType.BOX, GeomType.MESH),
  (GeomType.MESH, GeomType.MESH),
]


@event_scope
def convex_narrowphase(m: Model, d: Data):
  """Runs narrowphase collision detection for convex geom pairs.

  This function handles collision detection for pairs of convex geometries that were
  identified during the broadphase. It uses the Gilbert-Johnson-Keerthi (GJK) algorithm to
  determine the distance between shapes and the Expanding Polytope Algorithm (EPA) to find
  the penetration depth and contact normal for colliding pairs.

  The convex geom types handled by this function are `SPHERE`, `CAPSULE`, `ELLIPSOID`, `CYLINDER`,
  `BOX`, `MESH`, `HFIELD`.

  To optimize performance, this function dynamically builds and launches a specialized
  kernel for each type of convex collision pair present in the model, avoiding unnecessary
  computations for non-existent pair types.
  """

  def _pair_count(p1: int, p2: int) -> int:
    idx = upper_trid_index(len(GeomType), p1, p2)
    return m.geom_pair_type_count[idx], idx

  # no convex collisions, early return
  if not any(_pair_count(g[0].value, g[1].value)[0] for g in _CONVEX_COLLISION_PAIRS):
    return

  epa_iterations = m.opt.ccd_iterations

  # set to true to enable multiccd
  use_multiccd = False
  nmaxpolygon = m.nmaxpolygon if use_multiccd else 0
  nmaxmeshdeg = m.nmaxmeshdeg if use_multiccd else 0

  # Reset CCD arena counter
  d.nccd.zero_()

  # EPA and multiccd arrays are now constructed from d.arena inside kernels

  # Contact outputs
  contact_outputs = [
    d.nacon,
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
    d.contact.worldid,
    d.contact.type,
    d.contact.geomcollisionid,
  ]

  # Launch heightfield collision kernels (no multiccd args)
  for geom_pair in _HFIELD_COLLISION_PAIRS:
    g1 = geom_pair[0].value
    g2 = geom_pair[1].value
    count, idx = _pair_count(g1, g2)
    if count:
      wp.launch(
        ccd_hfield_kernel_builder(g1, g2, m.opt.ccd_iterations, epa_iterations, idx),
        dim=d.naconmax,
        inputs=[
          m.opt.ccd_tolerance,
          m.geom_type,
          m.geom_condim,
          m.geom_dataid,
          m.geom_priority,
          m.geom_solmix,
          m.geom_solref,
          m.geom_solimp,
          m.geom_size,
          m.geom_aabb,
          m.geom_rbound,
          m.geom_friction,
          m.geom_margin,
          m.geom_gap,
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
          m.pair_dim,
          m.pair_solref,
          m.pair_solreffriction,
          m.pair_solimp,
          m.pair_margin,
          m.pair_gap,
          m.pair_friction,
          d.naconmax,
          d.geom_xpos,
          d.geom_xmat,
          d.collision_pair,
          d.collision_pairid,
          d.collision_worldid,
          d.ncollision,
          d.naccdmax,
          d.nccd,
          d.arena,
        ],
        outputs=contact_outputs,
      )

  # Non-heightfield kernels now use arena memory for EPA/multiccd arrays
  for geom_pair in _NON_HFIELD_COLLISION_PAIRS:
    g1 = geom_pair[0].value
    g2 = geom_pair[1].value
    count, idx = _pair_count(g1, g2)
    if count:
      wp.launch(
        ccd_kernel_builder(g1, g2, m.opt.ccd_iterations, epa_iterations, idx, use_multiccd, nmaxpolygon, nmaxmeshdeg),
        dim=d.naconmax,
        inputs=[
          m.opt.ccd_tolerance,
          m.geom_type,
          m.geom_condim,
          m.geom_dataid,
          m.geom_priority,
          m.geom_solmix,
          m.geom_solref,
          m.geom_solimp,
          m.geom_size,
          m.geom_friction,
          m.geom_margin,
          m.geom_gap,
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
          m.pair_dim,
          m.pair_solref,
          m.pair_solreffriction,
          m.pair_solimp,
          m.pair_margin,
          m.pair_gap,
          m.pair_friction,
          d.naconmax,
          d.geom_xpos,
          d.geom_xmat,
          d.collision_pair,
          d.collision_pairid,
          d.collision_worldid,
          d.ncollision,
          d.naccdmax,
          d.nccd,
          d.arena,
        ],
        outputs=contact_outputs,
      )
