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

from .collision_primitive import Geom
from .collision_primitive import _geom
from .collision_primitive import contact_params
from .collision_primitive import write_contact
from .math import gjk_normalize
from .math import make_frame
from .math import orthonormal
from .support import all_same
from .support import any_different
from .types import MJ_MINVAL
from .types import Data
from .types import GeomType
from .types import Model
from .types import vec5

from .convex_collision_newton import *

wp.clear_kernel_cache()


# TODO(team): improve compile time to enable backward pass
wp.config.enable_backward = False


_CONVEX_COLLISION_FUNC = {
  (GeomType.SPHERE.value, GeomType.ELLIPSOID.value),
  (GeomType.SPHERE.value, GeomType.MESH.value),
  (GeomType.CAPSULE.value, GeomType.CYLINDER.value),
  (GeomType.CAPSULE.value, GeomType.ELLIPSOID.value),
  (GeomType.CAPSULE.value, GeomType.MESH.value),
  (GeomType.ELLIPSOID.value, GeomType.ELLIPSOID.value),
  (GeomType.ELLIPSOID.value, GeomType.CYLINDER.value),
  (GeomType.ELLIPSOID.value, GeomType.BOX.value),
  (GeomType.ELLIPSOID.value, GeomType.MESH.value),
  (GeomType.CYLINDER.value, GeomType.CYLINDER.value),
  (GeomType.CYLINDER.value, GeomType.BOX.value),
  (GeomType.CYLINDER.value, GeomType.MESH.value),
  (GeomType.BOX.value, GeomType.MESH.value),
  (GeomType.MESH.value, GeomType.MESH.value),
}


class geoMap(wp.types.vector(11, dtype=wp.int32)):
  pass


def build_geo_map():
  # Start with identity map
  map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  map[GeomType.SPHERE.value] = GEO_SPHERE
  map[GeomType.BOX.value] = GEO_BOX
  map[GeomType.CAPSULE.value] = GEO_CAPSULE
  map[GeomType.CYLINDER.value] = GEO_CYLINDER
  map[GeomType.MESH.value] = GEO_CONVEX
  map[GeomType.ELLIPSOID.value] = GEO_ELLIPSOID
  return map


# Map from mujoco_warp to newton geo types
geo_map_host = build_geo_map()
geo_map = wp.constant(geoMap(geo_map_host))


def _gjk_epa_pipeline(
  geomtype1: int,
  geomtype2: int,
  gjk_iterations: int,
  epa_iterations: int,
  epa_exact_neg_distance: bool,
  depth_extension: float,
):
  _gjk = get_gjk(geo_map_host[geomtype1], geo_map_host[geomtype2], gjk_iterations)
  _epa = get_epa(geo_map_host[geomtype1], geo_map_host[geomtype2], epa_iterations, epa_exact_neg_distance, depth_extension)
  _multiple_contacts = get_multiple_contacts(geo_map_host[geomtype1], geo_map_host[geomtype2], depth_extension)

  # runs GJK and EPA on a set of sparse geom pairs per env
  @wp.kernel
  def gjk_epa_sparse(
    # Model:
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
    pair_dim: wp.array(dtype=int),
    pair_solref: wp.array2d(dtype=wp.vec2),
    pair_solreffriction: wp.array2d(dtype=wp.vec2),
    pair_solimp: wp.array2d(dtype=vec5),
    pair_margin: wp.array2d(dtype=float),
    pair_gap: wp.array2d(dtype=float),
    pair_friction: wp.array2d(dtype=vec5),
    # Data in:
    nconmax_in: int,
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    collision_pair_in: wp.array(dtype=wp.vec2i),
    collision_pairid_in: wp.array(dtype=int),
    collision_worldid_in: wp.array(dtype=int),
    ncollision_in: wp.array(dtype=int),
    # Data out:
    ncon_out: wp.array(dtype=int),
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
  ):
    tid = wp.tid()
    if tid >= ncollision_in[0]:
      return

    worldid = collision_worldid_in[tid]
    geoms, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
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

    g1 = geoms[0]
    g2 = geoms[1]

    if geom_type[g1] != geomtype1 or geom_type[g2] != geomtype2:
      return

    geom1 = _geom(
      geom_type,
      geom_dataid,
      geom_size[worldid],
      mesh_vertadr,
      mesh_vertnum,
      mesh_vert,
      geom_xpos_in,
      geom_xmat_in,
      worldid,
      g1,
    )

    geom2 = _geom(
      geom_type,
      geom_dataid,
      geom_size[worldid],
      mesh_vertadr,
      mesh_vertnum,
      mesh_vert,
      geom_xpos_in,
      geom_xmat_in,
      worldid,
      g2,
    )

    margin = wp.max(geom_margin[worldid, g1], geom_margin[worldid, g2])

    simplex, normal = _gjk(mesh_vert, geom1, geom2)

    # TODO(btaba): get depth from GJK, conditionally run EPA.
    depth, normal = _epa(mesh_vert, geom1, geom2, simplex, normal)
    dist = -depth

    if (dist - margin) >= 0.0 or depth != depth:
      return

    # TODO(btaba): split get_multiple_contacts into a separate kernel.
    # TODO(team): multiccd enablebit
    count, points = _multiple_contacts(mesh_vert, geom1, geom2, depth, normal)

    frame = make_frame(normal)
    for i in range(count):
      write_contact(
        nconmax_in,
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
        worldid,
        ncon_out,
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
      )

  return gjk_epa_sparse


_collision_kernels = {}


def gjk_narrowphase(m: Model, d: Data):
  if len(_collision_kernels) == 0:
    for types in _CONVEX_COLLISION_FUNC:
      t1 = types[0]
      t2 = types[1]
      _collision_kernels[(t1, t2)] = _gjk_epa_pipeline(
        t1,
        t2,
        m.opt.gjk_iterations,
        m.opt.epa_iterations,
        m.opt.epa_exact_neg_distance,
        m.opt.depth_extension,
      )

  for collision_kernel in _collision_kernels.values():
    wp.launch(
      collision_kernel,
      dim=d.nconmax,
      inputs=[
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
        m.pair_dim,
        m.pair_solref,
        m.pair_solreffriction,
        m.pair_solimp,
        m.pair_margin,
        m.pair_gap,
        m.pair_friction,
        d.nconmax,
        d.geom_xpos,
        d.geom_xmat,
        d.collision_pair,
        d.collision_pairid,
        d.collision_worldid,
        d.ncollision,
      ],
      outputs=[
        d.ncon,
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
