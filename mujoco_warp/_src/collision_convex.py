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

from collections.abc import Callable
from typing import Tuple

import warp as wp

from .collision_gjk import ccd
from .collision_gjk_legacy import epa_legacy
from .collision_gjk_legacy import gjk_legacy
from .collision_gjk_legacy import multicontact_legacy
from .collision_primitive import Geom
from .collision_primitive import geom
from .collision_primitive import write_contact
from .math import make_frame
from .math import safe_div
from .types import MJ_MAXCONPAIR
from .types import MJ_MAXVAL
from .types import MJ_MINMU
from .types import MJ_MINVAL
from .types import GeomType
from .types import vec5
from .warp_util import cache_kernel
from .warp_util import kernel as nested_kernel

# TODO(team): improve compile time to enable backward pass
wp.set_module_options({"enable_backward": False})

MULTI_CONTACT_COUNT = 8
mat3c = wp.types.matrix(shape=(MULTI_CONTACT_COUNT, 3), dtype=float)
mat63 = wp.types.matrix(shape=(6, 3), dtype=float)


@wp.func
def contact_params(
  # Model:
  geom_condim: wp.array(dtype=int),
  geom_priority: wp.array(dtype=int),
  geom_solmix: wp.array2d(dtype=float),
  geom_solref: wp.array2d(dtype=wp.vec2),
  geom_solimp: wp.array2d(dtype=vec5),
  geom_friction: wp.array2d(dtype=wp.vec3),
  geom_margin: wp.array2d(dtype=float),
  geom_gap: wp.array2d(dtype=float),
  pair_dim: wp.array(dtype=int),
  pair_solref: wp.array2d(dtype=wp.vec2),
  pair_solreffriction: wp.array2d(dtype=wp.vec2),
  pair_solimp: wp.array2d(dtype=vec5),
  pair_margin: wp.array2d(dtype=float),
  pair_gap: wp.array2d(dtype=float),
  pair_friction: wp.array2d(dtype=vec5),
  # In:
  g1: int,
  g2: int,
  pairid: int,
  worldid: int,
):
  if pairid > -1:
    margin = pair_margin[worldid, pairid]
    gap = pair_gap[worldid, pairid]
    condim = pair_dim[pairid]
    friction = pair_friction[worldid, pairid]
    solref = pair_solref[worldid, pairid]
    solreffriction = pair_solreffriction[worldid, pairid]
    solimp = pair_solimp[worldid, pairid]
  else:
    solmix_id = worldid % geom_solmix.shape[0]
    friction_id = worldid % geom_friction.shape[0]
    solref_id = worldid % geom_solref.shape[0]
    solimp_id = worldid % geom_solimp.shape[0]
    margin_id = worldid % geom_margin.shape[0]
    gap_id = worldid % geom_gap.shape[0]

    solmix1 = geom_solmix[solmix_id, g1]
    solmix2 = geom_solmix[solmix_id, g2]

    condim1 = geom_condim[g1]
    condim2 = geom_condim[g2]

    # priority
    p1 = geom_priority[g1]
    p2 = geom_priority[g2]

    if p1 > p2:
      mix = 1.0
      condim = condim1
      max_geom_friction = geom_friction[friction_id, g1]
    elif p2 > p1:
      mix = 0.0
      condim = condim2
      max_geom_friction = geom_friction[friction_id, g2]
    else:
      mix = safe_div(solmix1, solmix1 + solmix2)
      mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
      mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
      mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)
      condim = wp.max(condim1, condim2)
      max_geom_friction = wp.max(geom_friction[friction_id, g1], geom_friction[friction_id, g2])

    friction = vec5(
      wp.max(MJ_MINMU, max_geom_friction[0]),
      wp.max(MJ_MINMU, max_geom_friction[0]),
      wp.max(MJ_MINMU, max_geom_friction[1]),
      wp.max(MJ_MINMU, max_geom_friction[2]),
      wp.max(MJ_MINMU, max_geom_friction[2]),
    )

    if geom_solref[solref_id, g1][0] > 0.0 and geom_solref[solref_id, g2][0] > 0.0:
      solref = mix * geom_solref[solref_id, g1] + (1.0 - mix) * geom_solref[solref_id, g2]
    else:
      solref = wp.min(geom_solref[solref_id, g1], geom_solref[solref_id, g2])

    solreffriction = wp.vec2(0.0, 0.0)
    solimp = mix * geom_solimp[solimp_id, g1] + (1.0 - mix) * geom_solimp[solimp_id, g2]
    # geom priority is ignored
    margin = wp.max(geom_margin[margin_id, g1], geom_margin[margin_id, g2])
    gap = wp.max(geom_gap[gap_id, g1], geom_gap[gap_id, g2])

  return margin, gap, condim, friction, solref, solreffriction, solimp


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
  xpos_id = worldid % geom_xpos_in.shape[0]
  xmat_id = worldid % geom_xmat_in.shape[0]
  rbound_id = worldid % geom_rbound.shape[0]
  margin_id = worldid % geom_margin.shape[0]

  pos1 = geom_xpos_in[xpos_id, g1]
  mat1 = geom_xmat_in[xmat_id, g1]
  mat1T = wp.transpose(mat1)
  pos2 = geom_xpos_in[xpos_id, g2]
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
def convex_kernel_builder(
  legacy_gjk: bool,
  geomtype1: int,
  geomtype2: int,
  ccd_iterations: int,
  epa_exact_neg_distance: bool,
  depth_extension: float,
  is_hfield: bool,
  primitive_func: Callable | None,
):
  @wp.func
  def ccd_wrapper(
    # Model:
    opt_ccd_tolerance: wp.array(dtype=float),
    geom_type: wp.array(dtype=int),
    # Data in:
    naconmax_in: int,
    # In:
    epa_vert_in: wp.array2d(dtype=wp.vec3),
    epa_vert1_in: wp.array2d(dtype=wp.vec3),
    epa_vert2_in: wp.array2d(dtype=wp.vec3),
    epa_vert_index1_in: wp.array2d(dtype=int),
    epa_vert_index2_in: wp.array2d(dtype=int),
    epa_face_in: wp.array2d(dtype=wp.vec3i),
    epa_pr_in: wp.array2d(dtype=wp.vec3),
    epa_norm2_in: wp.array2d(dtype=float),
    epa_index_in: wp.array2d(dtype=int),
    epa_map_in: wp.array2d(dtype=int),
    epa_horizon_in: wp.array2d(dtype=int),
    multiccd_polygon_in: wp.array2d(dtype=wp.vec3),
    multiccd_clipped_in: wp.array2d(dtype=wp.vec3),
    multiccd_pnormal_in: wp.array2d(dtype=wp.vec3),
    multiccd_pdist_in: wp.array2d(dtype=float),
    multiccd_idx1_in: wp.array2d(dtype=int),
    multiccd_idx2_in: wp.array2d(dtype=int),
    multiccd_n1_in: wp.array2d(dtype=wp.vec3),
    multiccd_n2_in: wp.array2d(dtype=wp.vec3),
    multiccd_endvert_in: wp.array2d(dtype=wp.vec3),
    multiccd_face1_in: wp.array2d(dtype=wp.vec3),
    multiccd_face2_in: wp.array2d(dtype=wp.vec3),
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
    # TODO(kbayes): remove legacy GJK once multicontact can be enabled
    if wp.static(legacy_gjk):
      simplex, normal = gjk_legacy(
        ccd_iterations,
        geom1,
        geom2,
        geomtype1,
        geomtype2,
      )

      depth, normal = epa_legacy(
        ccd_iterations, geom1, geom2, geomtype1, geomtype2, depth_extension, epa_exact_neg_distance, simplex, normal
      )
      dist = -depth

      if dist >= 0.0 or depth < -depth_extension:
        return 0
      sphere = GeomType.SPHERE
      ellipsoid = GeomType.ELLIPSOID
      g1 = geoms[0]
      g2 = geoms[1]
      if geom_type[g1] == sphere or geom_type[g1] == ellipsoid or geom_type[g2] == sphere or geom_type[g2] == ellipsoid:
        ncontact, points = multicontact_legacy(geom1, geom2, geomtype1, geomtype2, depth_extension, depth, normal, 1, 2, 1.0e-5)
      else:
        ncontact, points = multicontact_legacy(geom1, geom2, geomtype1, geomtype2, depth_extension, depth, normal, 4, 8, 1.0e-1)
      frame = make_frame(normal)
    else:
      points = mat3c()
      geom1.margin = margin
      geom2.margin = margin
      if pairid[1] >= 0:
        # if collision sensor, set large cutoff to work with various sensor cutoff values
        cutoff = 1.0e32
      else:
        cutoff = 0.0
      dist, ncontact, witness1, witness2 = ccd(
        False,  # ignored for box-box, multiccd always on
        opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]],
        cutoff,
        ccd_iterations,
        geom1,
        geom2,
        geomtype1,
        geomtype2,
        x1,
        x2,
        epa_vert_in[tid],
        epa_vert1_in[tid],
        epa_vert2_in[tid],
        epa_vert_index1_in[tid],
        epa_vert_index2_in[tid],
        epa_face_in[tid],
        epa_pr_in[tid],
        epa_norm2_in[tid],
        epa_index_in[tid],
        epa_map_in[tid],
        epa_horizon_in[tid],
        multiccd_polygon_in[tid],
        multiccd_clipped_in[tid],
        multiccd_pnormal_in[tid],
        multiccd_pdist_in[tid],
        multiccd_idx1_in[tid],
        multiccd_idx2_in[tid],
        multiccd_n1_in[tid],
        multiccd_n2_in[tid],
        multiccd_endvert_in[tid],
        multiccd_face1_in[tid],
        multiccd_face2_in[tid],
      )

      if dist >= 0.0 and pairid[1] == -1:
        return 0

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

  # runs convex collision on a set of geom pairs to recover contact info
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
    # In:
    epa_vert_in: wp.array2d(dtype=wp.vec3),
    epa_vert1_in: wp.array2d(dtype=wp.vec3),
    epa_vert2_in: wp.array2d(dtype=wp.vec3),
    epa_vert_index1_in: wp.array2d(dtype=int),
    epa_vert_index2_in: wp.array2d(dtype=int),
    epa_face_in: wp.array2d(dtype=wp.vec3i),
    epa_pr_in: wp.array2d(dtype=wp.vec3),
    epa_norm2_in: wp.array2d(dtype=float),
    epa_index_in: wp.array2d(dtype=int),
    epa_map_in: wp.array2d(dtype=int),
    epa_horizon_in: wp.array2d(dtype=int),
    multiccd_polygon_in: wp.array2d(dtype=wp.vec3),
    multiccd_clipped_in: wp.array2d(dtype=wp.vec3),
    multiccd_pnormal_in: wp.array2d(dtype=wp.vec3),
    multiccd_pdist_in: wp.array2d(dtype=float),
    multiccd_idx1_in: wp.array2d(dtype=int),
    multiccd_idx2_in: wp.array2d(dtype=int),
    multiccd_n1_in: wp.array2d(dtype=wp.vec3),
    multiccd_n2_in: wp.array2d(dtype=wp.vec3),
    multiccd_endvert_in: wp.array2d(dtype=wp.vec3),
    multiccd_face1_in: wp.array2d(dtype=wp.vec3),
    multiccd_face2_in: wp.array2d(dtype=wp.vec3),
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
    if wp.static(is_hfield):
      no_hf_collision, xmin, xmax, ymin, ymax, zmin, _ = _hfield_filter(
        geom_dataid, geom_aabb, geom_rbound, geom_margin, hfield_size, geom_xpos_in, geom_xmat_in, worldid, g1, g2
      )
      if no_hf_collision:
        return

    margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
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
      g1,
      g2,
      collision_pairid_in[tid],
      worldid,
    )

    geom_size_id = worldid % geom_size.shape[0]
    geom_xpos_id = worldid % geom_xpos_in.shape[0]
    geom_xmat_id = worldid % geom_xmat_in.shape[0]

    geom1_dataid = geom_dataid[g1]
    geom1 = geom(
      geomtype1,
      geom1_dataid,
      geom_size[geom_size_id, g1],
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
      geom_xpos_in[geom_xpos_id, g1],
      geom_xmat_in[geom_xmat_id, g1],
    )

    geom2_dataid = geom_dataid[g2]
    geom2 = geom(
      geomtype2,
      geom2_dataid,
      geom_size[geom_size_id, g2],
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
      geom_xpos_in[geom_xpos_id, g2],
      geom_xmat_in[geom_xmat_id, g2],
    )

    if wp.static(primitive_func != None):
      wp.static(primitive_func)(
        naconmax_in,
        geom1,
        geom2,
        worldid,
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        geoms,
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
        nacon_out,
      )
      return

    # see MuJoCo mjc_ConvexHField
    if wp.static(is_hfield):
      # height field subgrid
      nrow = hfield_nrow[g1]
      ncol = hfield_ncol[g1]
      size = hfield_size[g1]

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

      # process all prisms in subgrid
      count = int(0)
      for r in range(rmin, rmax):
        nvert = int(0)
        for c in range(cmin, cmax + 1):
          # add both triangles from this cell
          for i in range(2):
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
            if wp.static(not legacy_gjk):
              x1_ = wp.vec3(0.0, 0.0, 0.0)
              for i in range(6):
                x1_ += prism[i]
              x1 += geom1.rot @ (x1_ / 6.0)

            ncontact = ccd_wrapper(
              opt_ccd_tolerance,
              geom_type,
              naconmax_in,
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
              x1,
              geom2.pos,
              count,
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
            count += ncontact
            if count >= MJ_MAXCONPAIR:
              return
    else:
      ccd_wrapper(
        opt_ccd_tolerance,
        geom_type,
        naconmax_in,
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


@event_scope
def convex_narrowphase(m: Model, d: Data):
  """Runs narrowphase collision detection for convex geom pairs.

  This function handles collision detection for pairs of convex geometries that were
  identified during the broadphase. It uses the Gilbert-Johnson-Keerthi (GJK) algorithm to
  determine the distance between shapes and the Expanding Polytope Algorithm (EPA) to find
  the penetration depth and contact normal for colliding pairs.

  The convex geom types handled by this function are SPHERE, CAPSULE, ELLIPSOID, CYLINDER,
  BOX, MESH, HFIELD.

  To optimize performance, this function dynamically builds and launches a specialized
  kernel for each type of convex collision pair present in the model, avoiding unnecessary
  computations for non-existent pair types.
  """
  if not any(m.geom_pair_type_count[upper_trid_index(len(GeomType), g[0].value, g[1].value)] for g in _CONVEX_COLLISION_PAIRS):
    return

  # epa_vert: vertices in EPA polytope in Minkowski space
  epa_vert = wp.empty(shape=(d.naconmax, 5 + m.opt.ccd_iterations), dtype=wp.vec3)
  # epa_vert1: vertices in EPA polytope in geom 1 space
  epa_vert1 = wp.empty(shape=(d.naconmax, 5 + m.opt.ccd_iterations), dtype=wp.vec3)
  # epa_vert2: vertices in EPA polytope in geom 2 space
  epa_vert2 = wp.empty(shape=(d.naconmax, 5 + m.opt.ccd_iterations), dtype=wp.vec3)
  # epa_vert_index1: vertex indices in EPA polytope for geom 1
  epa_vert_index1 = wp.empty(shape=(d.naconmax, 5 + m.opt.ccd_iterations), dtype=int)
  # epa_vert_index2: vertex indices in EPA polytope for geom 2  (naconmax, 5 + CCDiter)
  epa_vert_index2 = wp.empty(shape=(d.naconmax, 5 + m.opt.ccd_iterations), dtype=int)
  # epa_face: faces of polytope represented by three indices
  epa_face = wp.empty(shape=(d.naconmax, 6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations), dtype=wp.vec3i)
  # epa_pr: projection of origin on polytope faces
  epa_pr = wp.empty(shape=(d.naconmax, 6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations), dtype=wp.vec3)
  # epa_norm2: epa_pr * epa_pr
  epa_norm2 = wp.empty(shape=(d.naconmax, 6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations), dtype=float)
  # epa_index: index of face in polytope map
  epa_index = wp.empty(shape=(d.naconmax, 6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations), dtype=int)
  # epa_map: status of faces in polytope
  epa_map = wp.empty(shape=(d.naconmax, 6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations), dtype=int)
  # epa_horizon: index pair (i j) of edges on horizon
  epa_horizon = wp.empty(shape=(d.naconmax, 2 * MJ_MAX_EPAHORIZON), dtype=int)
  # multiccd_polygon: clipped contact surface
  multiccd_polygon = wp.empty(shape=(d.naconmax, 2 * m.nmaxpolygon), dtype=wp.vec3)
  # multiccd_clipped: clipped contact surface (intermediate)
  multiccd_clipped = wp.empty(shape=(d.naconmax, 2 * m.nmaxpolygon), dtype=wp.vec3)
  # multiccd_pnormal: plane normal of clipping polygon
  multiccd_pnormal = wp.empty(shape=(d.naconmax, m.nmaxpolygon), dtype=wp.vec3)
  # multiccd_pdist: plane distance of clipping polygon
  multiccd_pdist = wp.empty(shape=(d.naconmax, m.nmaxpolygon), dtype=float)
  # multiccd_idx1: list of normal index candidates for Geom 1
  multiccd_idx1 = wp.empty(shape=(d.naconmax, m.nmaxmeshdeg), dtype=int)
  # multiccd_idx2: list of normal index candidates for Geom 2
  multiccd_idx2 = wp.empty(shape=(d.naconmax, m.nmaxmeshdeg), dtype=int)
  # multiccd_n1: list of normal candidates for Geom 1
  multiccd_n1 = wp.empty(shape=(d.naconmax, m.nmaxmeshdeg), dtype=wp.vec3)
  # multiccd_n2: list of normal candidates for Geom 1
  multiccd_n2 = wp.empty(shape=(d.naconmax, m.nmaxmeshdeg), dtype=wp.vec3)
  # multiccd_endvert: list of edge vertices candidates
  multiccd_endvert = wp.empty(shape=(d.naconmax, m.nmaxmeshdeg), dtype=wp.vec3)
  # multiccd_face1: contact face
  multiccd_face1 = wp.empty(shape=(d.naconmax, m.nmaxpolygon), dtype=wp.vec3)
  # multiccd_face2: contact face
  multiccd_face2 = wp.empty(shape=(d.naconmax, m.nmaxpolygon), dtype=wp.vec3)

  for geom_pair in _CONVEX_COLLISION_PAIRS:
    g1 = geom_pair[0].value
    g2 = geom_pair[1].value
    if m.geom_pair_type_count[upper_trid_index(len(GeomType), g1, g2)]:
      wp.launch(
        ccd_kernel_builder(m.opt.legacy_gjk, g1, g2, m.opt.ccd_iterations, True, 1e9, g1 == GeomType.HFIELD),
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
        ],
        outputs=[
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
        ],
      )
