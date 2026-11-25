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

"""Core collision types and utilities shared across collision modules."""

import dataclasses
from typing import Any, Tuple

import warp as wp

from mujoco_warp._src.math import quat_to_mat
from mujoco_warp._src.math import safe_div
from mujoco_warp._src.math import upper_trid_index
from mujoco_warp._src.types import MJ_MINMU
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import ContactType
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import mat63
from mujoco_warp._src.types import vec5

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


def speculative_enabled(m: Model) -> bool:
  """Return True when speculative contacts are enabled on the model."""
  return bool(m.opt.enableflags & EnableBit.SPECULATIVE) and not bool(m.opt.disableflags & DisableBit.NATIVECCD)


@wp.struct
class Geom:
  """Geom properties for pairwise collision detection.

  Bundles a geometry's pose, size, surface normal, and mesh topology data into
  a single struct that can be passed to Warp collision kernels.
  """

  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3
  size: wp.vec3
  margin: float
  polyvert: mat63
  vertadr: int
  vertnum: int
  vert: wp.array[wp.vec3]
  graphadr: int
  graph: wp.array[int]
  mesh_polynum: int
  mesh_polyadr: int
  mesh_polynormal: wp.array[wp.vec3]
  mesh_polyvertadr: wp.array[int]
  mesh_polyvertnum: wp.array[int]
  mesh_polyvert: wp.array[int]
  mesh_polymapadr: wp.array[int]
  mesh_polymapnum: wp.array[int]
  mesh_polymap: wp.array[int]
  index: int


@wp.func
def make_gjk_geom(
  # Model:
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xquat_in: wp.array2d[wp.quat],
  # In:
  geomtype: int,
  geomid: int,
  worldid: int,
) -> Geom:
  """Build a Geom struct populated with pose, size, and mesh GJK support fields."""
  geom = Geom()
  geom.pos = geom_xpos_in[worldid, geomid]
  geom.rot = quat_to_mat(geom_xquat_in[worldid, geomid])
  # z-axis of the rotation matrix, used as the surface normal for plane collisions
  geom.normal = geom.rot[:, 2]
  geom.size = geom_size[worldid % geom_size.shape[0], geomid]
  geom.margin = 0.0
  geom.index = -1

  if geomtype == GeomType.MESH:
    dataid = geom_dataid[worldid % geom_dataid.shape[0], geomid]
    geom.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)
    geom.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)
    geom.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)
    geom.vert = mesh_vert
    geom.graph = mesh_graph

  return geom


@wp.func
def set_mesh_poly(
  # Model:
  geom_dataid: wp.array2d[int],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  mesh_polymapadr: wp.array[int],
  mesh_polymapnum: wp.array[int],
  mesh_polymap: wp.array[int],
  # In:
  geom: Geom,
  geomid: int,
  worldid: int,
) -> Geom:
  """Attach mesh polygon topology fields to a Geom struct for multicontact clipping."""
  dataid = geom_dataid[worldid % geom_dataid.shape[0], geomid]
  geom.mesh_polynum = wp.where(dataid >= 0, mesh_polynum[dataid], -1)
  geom.mesh_polyadr = wp.where(dataid >= 0, mesh_polyadr[dataid], -1)
  geom.mesh_polynormal = mesh_polynormal
  geom.mesh_polyvertadr = mesh_polyvertadr
  geom.mesh_polyvertnum = mesh_polyvertnum
  geom.mesh_polyvert = mesh_polyvert
  geom.mesh_polymapadr = mesh_polymapadr
  geom.mesh_polymapnum = mesh_polymapnum
  geom.mesh_polymap = mesh_polymap
  return geom


@wp.func
def geom_collision_pair_from_types(
  # Model:
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  mesh_polymapadr: wp.array[int],
  mesh_polymapnum: wp.array[int],
  mesh_polymap: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xquat_in: wp.array2d[wp.quat],
  # In:
  geom_type1: int,
  geom_type2: int,
  geoms: wp.vec2i,
  worldid: int,
) -> Tuple[Geom, Geom]:
  g1 = geoms[0]
  g2 = geoms[1]

  geom1 = make_gjk_geom(
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_vertnum,
    mesh_graphadr,
    mesh_vert,
    mesh_graph,
    geom_xpos_in,
    geom_xquat_in,
    geom_type1,
    g1,
    worldid,
  )
  geom2 = make_gjk_geom(
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_vertnum,
    mesh_graphadr,
    mesh_vert,
    mesh_graph,
    geom_xpos_in,
    geom_xquat_in,
    geom_type2,
    g2,
    worldid,
  )

  if geom_type1 == GeomType.MESH:
    geom1 = set_mesh_poly(
      geom_dataid,
      mesh_polynum,
      mesh_polyadr,
      mesh_polynormal,
      mesh_polyvertadr,
      mesh_polyvertnum,
      mesh_polyvert,
      mesh_polymapadr,
      mesh_polymapnum,
      mesh_polymap,
      geom1,
      g1,
      worldid,
    )

  if geom_type2 == GeomType.MESH:
    geom2 = set_mesh_poly(
      geom_dataid,
      mesh_polynum,
      mesh_polyadr,
      mesh_polynormal,
      mesh_polyvertadr,
      mesh_polyvertnum,
      mesh_polyvert,
      mesh_polymapadr,
      mesh_polymapnum,
      mesh_polymap,
      geom2,
      g2,
      worldid,
    )

  return geom1, geom2


@wp.func
def geom_collision_pair(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  mesh_polymapadr: wp.array[int],
  mesh_polymapnum: wp.array[int],
  mesh_polymap: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xquat_in: wp.array2d[wp.quat],
  # In:
  geoms: wp.vec2i,
  worldid: int,
) -> Tuple[Geom, Geom]:
  return geom_collision_pair_from_types(
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
    geom_xquat_in,
    geom_type[geoms[0]],
    geom_type[geoms[1]],
    geoms,
    worldid,
  )


@wp.func
def write_contact_entry(
  # Data in:
  naconmax_in: int,
  # In:
  id_: int,
  dist_in: float,
  pos_in: wp.vec3,
  frame_in: wp.mat33,
  margin_in: float,
  condim_in: int,
  friction_in: vec5,
  solref_in: wp.vec2,
  solreffriction_in: wp.vec2,
  solimp_in: vec5,
  adhesion_in: float,
  geoms_in: wp.vec2i,
  worldid_in: int,
  ctype_in: int,
  # Data out:
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_efc_address_out: wp.array2d[int],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  contact_adhesion_out: wp.array[float],
  nacon_out: wp.array[int],
) -> bool:
  """Atomically allocate and write a contact entry. Returns True if within naconmax."""
  cid = wp.atomic_add(nacon_out, 0, 1)  # kernel_analyzer: ignore[atomic]
  if cid < naconmax_in:
    contact_dist_out[cid] = dist_in
    contact_pos_out[cid] = pos_in
    contact_frame_out[cid] = frame_in
    contact_geom_out[cid] = geoms_in
    contact_worldid_out[cid] = worldid_in
    contact_includemargin_out[cid] = margin_in
    contact_dim_out[cid] = condim_in
    contact_friction_out[cid] = friction_in
    contact_solref_out[cid] = solref_in
    contact_solreffriction_out[cid] = solreffriction_in
    contact_solimp_out[cid] = solimp_in
    contact_adhesion_out[cid] = adhesion_in
    contact_type_out[cid] = ctype_in
    contact_geomcollisionid_out[cid] = id_
    for i in range(contact_efc_address_out.shape[1]):
      contact_efc_address_out[cid, i] = -1
    return True
  return False


@wp.func
def write_contact(
  # Data in:
  naconmax_in: int,
  # In:
  id_: int,
  dist_in: float,
  pos_in: wp.vec3,
  frame_in: wp.mat33,
  margin_in: float,
  gap_in: float,
  condim_in: int,
  friction_in: vec5,
  solref_in: wp.vec2,
  solreffriction_in: wp.vec2,
  solimp_in: vec5,
  adhesion_in: float,
  geoms_in: wp.vec2i,
  pairid_in: wp.vec2i,
  worldid_in: int,
  # Data out:
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_efc_address_out: wp.array2d[int],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  contact_adhesion_out: wp.array[float],
  nacon_out: wp.array[int],
) -> int:
  """Atomically write a detected contact into the contact output arrays.

  Returns 1 if the contact is active (dist < margin or in-gap adhesive), 0 otherwise.
  """
  active = (dist_in < margin_in) or ((adhesion_in != 0.0) and (dist_in < margin_in + gap_in))
  detected = dist_in < margin_in + gap_in
  condim = wp.where((adhesion_in != 0.0) and (dist_in >= margin_in), 1, condim_in)

  # skip contact and no collision sensor
  if (pairid_in[0] == -2 or not detected) and pairid_in[1] == -1:
    return 0

  contact_type = 0

  if pairid_in[0] >= -1 and detected:
    contact_type |= ContactType.CONSTRAINT

  if pairid_in[1] >= 0:
    contact_type |= ContactType.SENSOR

  wrote = write_contact_entry(
    naconmax_in,
    id_,
    dist_in,
    pos_in,
    frame_in,
    margin_in,
    condim,
    friction_in,
    solref_in,
    solreffriction_in,
    solimp_in,
    adhesion_in,
    geoms_in,
    worldid_in,
    contact_type,
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
    contact_efc_address_out,
    contact_worldid_out,
    contact_type_out,
    contact_geomcollisionid_out,
    contact_adhesion_out,
    nacon_out,
  )
  if wrote:
    return int(active)
  return 0


@wp.func
def contact_margin(
  # Model:
  geom_margin: wp.array2d[float],
  pair_margin: wp.array2d[float],
  # In:
  geoms: wp.vec2i,
  pairid: int,
  worldid: int,
) -> float:
  """Resolve effective contact margin for a geom pair."""
  if pairid > -1:
    return pair_margin[worldid % pair_margin.shape[0], pairid]
  margin_id = worldid % geom_margin.shape[0]
  return geom_margin[margin_id, geoms[0]] + geom_margin[margin_id, geoms[1]]


@wp.func
def contact_margin_gap(
  # Model:
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  pair_margin: wp.array2d[float],
  pair_gap: wp.array2d[float],
  # In:
  geoms: wp.vec2i,
  pairid: int,
  worldid: int,
) -> Tuple[float, float]:
  margin = contact_margin(geom_margin, pair_margin, geoms, pairid, worldid)
  if pairid > -1:
    gap = pair_gap[worldid % pair_gap.shape[0], pairid]
  else:
    gap_id = worldid % geom_gap.shape[0]
    gap = geom_gap[gap_id, geoms[0]] + geom_gap[gap_id, geoms[1]]

  return margin, gap


@wp.func
def contact_material_params(
  # Model:
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_friction: wp.array2d[wp.vec3],
  geom_adhesion: wp.array2d[float],
  pair_dim: wp.array[int],
  pair_solref: wp.array2d[wp.vec2],
  pair_solreffriction: wp.array2d[wp.vec2],
  pair_solimp: wp.array2d[vec5],
  pair_adhesion: wp.array2d[float],
  pair_friction: wp.array2d[vec5],
  # In:
  geoms: wp.vec2i,
  pairid: int,
  worldid: int,
):
  if pairid > -1:
    condim = pair_dim[pairid]
    solref = pair_solref[worldid % pair_solref.shape[0], pairid]
    solreffriction = pair_solreffriction[worldid % pair_solreffriction.shape[0], pairid]
    solimp = pair_solimp[worldid % pair_solimp.shape[0], pairid]
    friction = pair_friction[worldid % pair_friction.shape[0], pairid]
    adhesion = pair_adhesion[worldid % pair_adhesion.shape[0], pairid]
  else:
    g1 = geoms[0]
    g2 = geoms[1]
    solref_id = worldid % geom_solref.shape[0]
    solimp_id = worldid % geom_solimp.shape[0]
    friction_id = worldid % geom_friction.shape[0]
    adhesion_id = worldid % geom_adhesion.shape[0]

    p1 = geom_priority[g1]
    p2 = geom_priority[g2]
    if p1 > p2:
      condim = geom_condim[g1]
      solref = geom_solref[solref_id, g1]
      solimp = geom_solimp[solimp_id, g1]
      max_geom_friction = geom_friction[friction_id, g1]
      adhesion = geom_adhesion[adhesion_id, g1]
    elif p2 > p1:
      condim = geom_condim[g2]
      solref = geom_solref[solref_id, g2]
      solimp = geom_solimp[solimp_id, g2]
      max_geom_friction = geom_friction[friction_id, g2]
      adhesion = geom_adhesion[adhesion_id, g2]
    else:
      solmix_id = worldid % geom_solmix.shape[0]
      solmix1 = geom_solmix[solmix_id, g1]
      solmix2 = geom_solmix[solmix_id, g2]
      mix = safe_div(solmix1, solmix1 + solmix2)
      mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
      mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
      mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)

      sref1 = geom_solref[solref_id, g1]
      sref2 = geom_solref[solref_id, g2]
      if sref1[0] > 0.0 and sref2[0] > 0.0:
        solref = mix * sref1 + (1.0 - mix) * sref2
      else:
        solref = wp.min(sref1, sref2)
      solimp = mix * geom_solimp[solimp_id, g1] + (1.0 - mix) * geom_solimp[solimp_id, g2]

      condim = wp.max(geom_condim[g1], geom_condim[g2])
      max_geom_friction = wp.max(geom_friction[friction_id, g1], geom_friction[friction_id, g2])
      adhesion = geom_adhesion[adhesion_id, g1] + geom_adhesion[adhesion_id, g2]

    friction = vec5(
      max_geom_friction[0],
      max_geom_friction[0],
      max_geom_friction[1],
      max_geom_friction[2],
      max_geom_friction[2],
    )
    solreffriction = wp.vec2(0.0, 0.0)

  friction = vec5(
    wp.max(MJ_MINMU, friction[0]),
    wp.max(MJ_MINMU, friction[1]),
    wp.max(MJ_MINMU, friction[2]),
    wp.max(MJ_MINMU, friction[3]),
    wp.max(MJ_MINMU, friction[4]),
  )

  return condim, friction, solref, solreffriction, solimp, adhesion


# TODO(team): early return if collision sensor but no contact
@wp.func
def contact_params(
  # Model:
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  geom_adhesion: wp.array2d[float],
  pair_dim: wp.array[int],
  pair_solref: wp.array2d[wp.vec2],
  pair_solreffriction: wp.array2d[wp.vec2],
  pair_solimp: wp.array2d[vec5],
  pair_margin: wp.array2d[float],
  pair_gap: wp.array2d[float],
  pair_adhesion: wp.array2d[float],
  pair_friction: wp.array2d[vec5],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_pairid_in: wp.array[wp.vec2i],
  cid: int,
  worldid: int,
):
  """Resolve contact parameters for a collision pair.

  Uses explicit pair overrides when available, otherwise mixes geom-level
  properties by priority and solmix weights.
  """
  geoms = collision_pair_in[cid]
  pairid = collision_pairid_in[cid][0]

  margin, gap = contact_margin_gap(geom_margin, geom_gap, pair_margin, pair_gap, geoms, pairid, worldid)
  condim, friction, solref, solreffriction, solimp, adhesion = contact_material_params(
    geom_condim,
    geom_priority,
    geom_solmix,
    geom_solref,
    geom_solimp,
    geom_friction,
    geom_adhesion,
    pair_dim,
    pair_solref,
    pair_solreffriction,
    pair_solimp,
    pair_adhesion,
    pair_friction,
    geoms,
    pairid,
    worldid,
  )

  return geoms, margin, gap, condim, friction, solref, solreffriction, solimp, adhesion


@dataclasses.dataclass
class CollisionContext:
  """Collision driver intermediate arrays.

  Attributes:
    collision_pair: collision pairs from broadphase (naconmax, 2)
    collision_pairid: ids from broadphase           (naconmax, 2)
    collision_worldid: collision world ids          (naconmax,)
    ncollision_inc: incremental pass pair count     (1,)
    woke: wake flag for incremental collision pass  (1,)
    awake_prev: body awake snapshot before wake     (nworld, nbody)
    nsdf: SDF collision candidate count             (1,)
    sdf_collisionid: SDF candidate collision ids    (naconmax,)
    nccd: CCD collider count per geom pair type     (ngeom_types * (ngeom_types + 1) // 2,)
    epa_vert: vertices in EPA polytope              (naccdmax, 10 + 2 * epa_iterations)
    epa_vert_index: vertex indices in EPA polytope  (naccdmax, 10 + 2 * epa_iterations)
    epa_face: polytope faces (three indices)        (naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations)
    epa_pr: projection of origin on polytope faces  (naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations)
    epa_norm2: epa_pr * epa_pr                      (naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations)
    epa_horizon: index pair (i, j) of horizon edges (naccdmax, MJ_MAX_EPAHORIZON)
    multiccd_polygon: clipped contact surface       (naccdmax, 2 * npolygonmax)
    multiccd_clipped: clipped contact surface (tmp) (naccdmax, 2 * npolygonmax)
    multiccd_pnormal: clipping polygon plane normal (naccdmax, npolygonmax)
    multiccd_pdist: clipping polygon plane distance (naccdmax, npolygonmax)
    multiccd_idx1: normal index candidates (Geom 1) (naccdmax, nmeshdegmax)
    multiccd_idx2: normal index candidates (Geom 2) (naccdmax, nmeshdegmax)
    multiccd_n1: normal candidates (Geom 1)         (naccdmax, nmeshdegmax)
    multiccd_n2: normal candidates (Geom 2)         (naccdmax, nmeshdegmax)
    multiccd_edge: edge direction candidates        (naccdmax, nmeshdegmax)
    multiccd_endvert: edge vertices candidates      (naccdmax, nmeshdegmax)
    multiccd_face1: contact face                    (naccdmax, npolygonmax)
    multiccd_face2: contact face                    (naccdmax, npolygonmax)
    geom_dx: per-geom step linear displacement      (nworld, ngeom)
    geom_dw: per-geom step angular displacement     (nworld, ngeom)
    geom_sweep: per-geom step surface sweep bound   (nworld, ngeom)
    spec_pair_table: open-addressed pair hash table (table_size,)
    spec_fast_cid: compacted fast broadphase pairs  (naconmax,)
    spec_pair_state: per-pair speculative state     (naconmax,)
    spec_state: speculative pass counters           (5,)
  """

  collision_pair: wp.array
  collision_pairid: wp.array
  collision_worldid: wp.array
  ncollision_inc: wp.array = None
  woke: wp.array = None
  awake_prev: wp.array = None
  nsdf: wp.array = None
  sdf_collisionid: wp.array = None
  nccd: wp.array = None
  epa_vert: wp.array = None
  epa_vert_index: wp.array = None
  epa_face: wp.array = None
  epa_pr: wp.array = None
  epa_norm2: wp.array = None
  epa_horizon: wp.array = None
  multiccd_polygon: wp.array = None
  multiccd_clipped: wp.array = None
  multiccd_pnormal: wp.array = None
  multiccd_pdist: wp.array = None
  multiccd_idx1: wp.array = None
  multiccd_idx2: wp.array = None
  multiccd_n1: wp.array = None
  multiccd_n2: wp.array = None
  multiccd_edge: wp.array = None
  multiccd_endvert: wp.array = None
  multiccd_face1: wp.array = None
  multiccd_face2: wp.array = None
  geom_dx: wp.array = None
  geom_dw: wp.array = None
  geom_sweep: wp.array = None
  spec_pair_table: wp.array = None
  spec_fast_cid: wp.array = None
  spec_pair_state: wp.array = None
  spec_state: wp.array = None


@wp.func
def sap_binary_search(values: wp.array[Any], value: Any, lower: int, upper: int) -> int:
  """Binary search for the first element > value in sorted array."""
  while lower < upper:
    mid = (lower + upper) >> 1
    if values[mid] > value:
      upper = mid
    else:
      lower = mid + 1
  return upper


@wp.kernel
def sap_range(
  # In:
  n: int,
  lower_in: wp.array2d[float],
  upper_in: wp.array2d[float],
  sort_index_in: wp.array2d[int],
  # Out:
  range_out: wp.array2d[int],
):
  """Compute the sweep range for each sorted element."""
  worldid, sortedid = wp.tid()

  idx = sort_index_in[worldid, sortedid]
  upper = upper_in[worldid, idx]

  limit = sap_binary_search(lower_in[worldid], upper, sortedid + 1, n)
  limit = wp.min(n - 1, limit)

  range_out[worldid, sortedid] = limit - sortedid


def multiccd_scratch_dims(
  m: Model,
  convex_pairs: list[tuple[GeomType, GeomType]] | None = None,
  speculative: bool = False,
) -> Tuple[int, int]:
  """Compute (npolygonmax, nmeshdegmax) for discrete convex or speculative multicontact."""
  use_multiccd = (m.opt.disableflags & DisableBit.MULTICCD) == 0
  geom_dims = {
    GeomType.PLANE: (1, 1),
    GeomType.CAPSULE: (2, 2),
    GeomType.CYLINDER: (16, 3),
    GeomType.BOX: (4, 3),
    GeomType.MESH: (max(m.npolygonmax, 1), max(m.nmeshdegmax, 3)),
  }
  convex_set = {(min(a, b), max(a, b)) for a, b in convex_pairs} if convex_pairs is not None else set()
  types_list = tuple(geom_dims)
  ngeom_types = len(GeomType)
  npolygonmax = 0
  nmeshdegmax = 0
  for idx1, g1 in enumerate(types_list):
    for g2 in types_list[idx1:]:
      if (g1 == GeomType.PLANE and g2 == GeomType.PLANE) or (g1 == GeomType.CAPSULE and g2 == GeomType.CAPSULE):
        continue
      if m.geom_pair_type_count[upper_trid_index(ngeom_types, g1.value, g2.value)] == 0:
        continue
      has_plane = (g1 == GeomType.PLANE) or (g2 == GeomType.PLANE)
      need_conv = (g1, g2) in convex_set and ((g1 == GeomType.BOX and g2 == GeomType.BOX) or use_multiccd)
      need_spec = speculative and (has_plane or use_multiccd)
      if need_conv or need_spec:
        p1, d1 = geom_dims[g1]
        p2, d2 = geom_dims[g2]
        npolygonmax = max(npolygonmax, p1, p2)
        nmeshdegmax = max(nmeshdegmax, d1, d2)
  return npolygonmax, nmeshdegmax


def allocate_multiccd_scratch(ctx: CollisionContext, naccdmax: int, npolygonmax: int, nmeshdegmax: int):
  """Allocate or grow multicontact scratch arrays on ctx to cover the requested dimensions."""
  existing_ccd = ctx.multiccd_polygon.shape[0] if ctx.multiccd_polygon is not None else -1
  existing_poly = ctx.multiccd_polygon.shape[1] // 2 if ctx.multiccd_polygon is not None else -1
  existing_deg = ctx.multiccd_n1.shape[1] if ctx.multiccd_n1 is not None else -1
  if (
    ctx.multiccd_polygon is not None
    and naccdmax <= existing_ccd
    and npolygonmax <= existing_poly
    and nmeshdegmax <= existing_deg
  ):
    return

  naccdmax = max(naccdmax, max(existing_ccd, 0))
  npolygonmax = max(npolygonmax, max(existing_poly, 0))
  nmeshdegmax = max(nmeshdegmax, max(existing_deg, 0))
  ctx.multiccd_polygon = wp.empty(shape=(naccdmax, 2 * npolygonmax), dtype=wp.vec3)
  ctx.multiccd_clipped = wp.empty(shape=(naccdmax, 2 * npolygonmax), dtype=wp.vec3)
  ctx.multiccd_pnormal = wp.empty(shape=(naccdmax, npolygonmax), dtype=wp.vec3)
  ctx.multiccd_pdist = wp.empty(shape=(naccdmax, npolygonmax), dtype=float)
  ctx.multiccd_idx1 = wp.empty(shape=(naccdmax, nmeshdegmax), dtype=int)
  ctx.multiccd_idx2 = wp.empty(shape=(naccdmax, nmeshdegmax), dtype=int)
  ctx.multiccd_n1 = wp.empty(shape=(naccdmax, nmeshdegmax), dtype=wp.vec3)
  ctx.multiccd_n2 = wp.empty(shape=(naccdmax, nmeshdegmax), dtype=wp.vec3)
  ctx.multiccd_edge = wp.empty(shape=(naccdmax, nmeshdegmax), dtype=wp.vec3)
  ctx.multiccd_endvert = wp.empty(shape=(naccdmax, nmeshdegmax), dtype=wp.vec3)
  ctx.multiccd_face1 = wp.empty(shape=(naccdmax, npolygonmax), dtype=wp.vec3)
  ctx.multiccd_face2 = wp.empty(shape=(naccdmax, npolygonmax), dtype=wp.vec3)


def create_collision_context(naconmax: int) -> CollisionContext:
  """Create a CollisionContext with allocated arrays."""
  return CollisionContext(
    collision_pair=wp.empty(naconmax, dtype=wp.vec2i),
    collision_pairid=wp.empty(naconmax, dtype=wp.vec2i),
    collision_worldid=wp.empty(naconmax, dtype=int),
    ncollision_inc=wp.empty(1, dtype=int),
  )
