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

from .collision_hfield import hfield_triangle_prism
from .collision_primitive_core import *
from .math import upper_trid_index
from .types import MJ_MINMU
from .types import MJ_MINVAL
from .types import Data
from .types import GeomType
from .types import Model
from .types import vec5
from .warp_util import event_scope

wp.set_module_options({"enable_backward": False})


class vec8f(wp.types.vector(length=8, dtype=wp.float32)):
  pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=wp.float32)):
  pass


class mat83f(wp.types.matrix(shape=(8, 3), dtype=wp.float32)):
  pass


@wp.struct
class Geom:
  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3
  size: wp.vec3
  hfprism: wp.mat33
  vertadr: int
  vertnum: int
  vert: wp.array(dtype=wp.vec3)
  graphadr: int
  graph: wp.array(dtype=int)
  mesh_polynum: int
  mesh_polyadr: int
  mesh_polynormal: wp.array(dtype=wp.vec3)
  mesh_polyvertadr: wp.array(dtype=int)
  mesh_polyvertnum: wp.array(dtype=int)
  mesh_polyvert: wp.array(dtype=int)
  mesh_polymapadr: wp.array(dtype=int)
  mesh_polymapnum: wp.array(dtype=int)
  mesh_polymap: wp.array(dtype=int)
  index: int


@wp.func
def _geom(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
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
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  worldid: int,
  gid: int,
  hftri_index: int,
) -> Geom:
  geom = Geom()
  geom.pos = geom_xpos_in[worldid, gid]
  rot = geom_xmat_in[worldid, gid]
  geom.rot = rot
  geom.size = geom_size[worldid, gid]
  geom.normal = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])  # plane
  dataid = geom_dataid[gid]

  # If geom is MESH, get mesh verts
  if dataid >= 0 and geom_type[gid] == int(GeomType.MESH.value):
    geom.vertadr = mesh_vertadr[dataid]
    geom.vertnum = mesh_vertnum[dataid]
    geom.graphadr = mesh_graphadr[dataid]
    geom.mesh_polynum = mesh_polynum[dataid]
    geom.mesh_polyadr = mesh_polyadr[dataid]
  else:
    geom.vertadr = -1
    geom.vertnum = -1
    geom.graphadr = -1
    geom.mesh_polynum = -1
    geom.mesh_polyadr = -1

  if geom_type[gid] == int(GeomType.MESH.value):
    geom.vert = mesh_vert
    geom.graph = mesh_graph
    geom.mesh_polynormal = mesh_polynormal
    geom.mesh_polyvertadr = mesh_polyvertadr
    geom.mesh_polyvertnum = mesh_polyvertnum
    geom.mesh_polyvert = mesh_polyvert
    geom.mesh_polymapadr = mesh_polymapadr
    geom.mesh_polymapnum = mesh_polymapnum
    geom.mesh_polymap = mesh_polymap

  # If geom is HFIELD triangle, compute triangle prism verts
  if geom_type[gid] == int(GeomType.HFIELD.value):
    geom.hfprism = hfield_triangle_prism(
      geom_dataid, hfield_adr, hfield_nrow, hfield_ncol, hfield_size, hfield_data, gid, hftri_index
    )

  geom.index = -1
  return geom


@wp.func
def write_contact(
  # Data in:
  nconmax_in: int,
  # In:
  dist_in: float,
  pos_in: wp.vec3,
  frame_in: wp.mat33,
  margin_in: float,
  gap_in: float,
  condim_in: int,
  friction_in: vec5,
  solref_in: wp.vec2f,
  solreffriction_in: wp.vec2f,
  solimp_in: vec5,
  geoms_in: wp.vec2i,
  worldid_in: int,
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
  active = (dist_in - margin_in) < 0
  if active:
    cid = wp.atomic_add(ncon_out, 0, 1)
    if cid < nconmax_in:
      contact_dist_out[cid] = dist_in
      contact_pos_out[cid] = pos_in
      contact_frame_out[cid] = frame_in
      contact_geom_out[cid] = geoms_in
      contact_worldid_out[cid] = worldid_in
      includemargin = margin_in - gap_in
      contact_includemargin_out[cid] = includemargin
      contact_dim_out[cid] = condim_in
      contact_friction_out[cid] = friction_in
      contact_solref_out[cid] = solref_in
      contact_solreffriction_out[cid] = solreffriction_in
      contact_solimp_out[cid] = solimp_in


@wp.struct
class WriteContactArgs:
  # Data in:
  nconmax_in: int
  # In:
  margin_in: float
  gap_in: float
  condim_in: int
  friction_in: vec5
  solref_in: wp.vec2f
  solreffriction_in: wp.vec2f
  solimp_in: vec5
  geoms_in: wp.vec2i
  worldid_in: int
  # Data out:
  ncon_out: wp.array(dtype=int)
  contact_dist_out: wp.array(dtype=float)
  contact_pos_out: wp.array(dtype=wp.vec3)
  contact_frame_out: wp.array(dtype=wp.mat33)
  contact_includemargin_out: wp.array(dtype=float)
  contact_friction_out: wp.array(dtype=vec5)
  contact_solref_out: wp.array(dtype=wp.vec2)
  contact_solreffriction_out: wp.array(dtype=wp.vec2)
  contact_solimp_out: wp.array(dtype=vec5)
  contact_dim_out: wp.array(dtype=int)
  contact_geom_out: wp.array(dtype=wp.vec2i)
  contact_worldid_out: wp.array(dtype=int)


@wp.func
def contact_writer(contact: ContactPoint, args: WriteContactArgs):
  write_contact(
    args.nconmax_in,
    contact.dist,
    contact.pos,
    extract_frame(contact),
    args.margin_in,
    args.gap_in,
    args.condim_in,
    args.friction_in,
    args.solref_in,
    args.solreffriction_in,
    args.solimp_in,
    args.geoms_in,
    args.worldid_in,
    args.ncon_out,
    args.contact_dist_out,
    args.contact_pos_out,
    args.contact_frame_out,
    args.contact_includemargin_out,
    args.contact_friction_out,
    args.contact_solref_out,
    args.contact_solreffriction_out,
    args.contact_solimp_out,
    args.contact_dim_out,
    args.contact_geom_out,
    args.contact_worldid_out,
  )


@wp.func
def plane_sphere_wrapper(
  # In:
  plane: Geom,
  sphere: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates one contact between a plane and a sphere."""
  plane_normal = wp.vec3(plane.rot[0, 2], plane.rot[1, 2], plane.rot[2, 2])
  return wp.static(get_plane_sphere(contact_writer))(
    plane_normal,
    plane.pos,
    sphere.pos,
    sphere.size[0],
    write_contact_args,
  )


@wp.func
def sphere_sphere_wrapper(
  # In:
  sphere1: Geom,
  sphere2: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates one contact between two spheres."""
  return wp.static(get_sphere_sphere(contact_writer))(
    sphere1.pos,
    sphere1.size[0],
    sphere2.pos,
    sphere2.size[0],
    write_contact_args,
  )


@wp.func
def sphere_capsule_wrapper(
  # In:
  sphere: Geom,
  cap: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates one contact between a sphere and a capsule."""
  cap_axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  return wp.static(get_sphere_capsule(contact_writer))(
    sphere.pos,
    sphere.size[0],
    sphere.rot,
    cap.pos,
    cap_axis,
    cap.size[0],
    cap.size[1],
    cap.rot,
    write_contact_args,
  )


@wp.func
def capsule_capsule_wrapper(
  # In:
  cap1: Geom,
  cap2: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates one contact between two capsules."""
  cap1_axis = wp.vec3(cap1.rot[0, 2], cap1.rot[1, 2], cap1.rot[2, 2])
  cap2_axis = wp.vec3(cap2.rot[0, 2], cap2.rot[1, 2], cap2.rot[2, 2])
  return wp.static(get_capsule_capsule(contact_writer))(
    cap1.pos,
    cap1_axis,
    cap1.size[0],
    cap1.size[1],
    cap2.pos,
    cap2_axis,
    cap2.size[0],
    cap2.size[1],
    write_contact_args,
  )


@wp.func
def plane_capsule_wrapper(
  # In:
  plane: Geom,
  cap: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates two contacts between a capsule and a plane."""
  plane_normal = wp.vec3(plane.rot[0, 2], plane.rot[1, 2], plane.rot[2, 2])
  cap_axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  return wp.static(get_plane_capsule(contact_writer))(
    plane_normal,
    plane.pos,
    cap.pos,
    cap_axis,
    cap.size[0],
    cap.size[1],
    write_contact_args,
  )


@wp.func
def plane_ellipsoid_wrapper(
  # In:
  plane: Geom,
  ellipsoid: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates one contact between a plane and an ellipsoid."""
  plane_normal = wp.vec3(plane.rot[0, 2], plane.rot[1, 2], plane.rot[2, 2])
  return wp.static(get_plane_ellipsoid(contact_writer))(
    plane_normal,
    plane.pos,
    ellipsoid.pos,
    ellipsoid.rot,
    ellipsoid.size,
    write_contact_args,
  )


@wp.func
def plane_box_wrapper(
  # In:
  plane: Geom,
  box: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates contacts between a plane and a box."""
  plane_normal = wp.vec3(plane.rot[0, 2], plane.rot[1, 2], plane.rot[2, 2])
  return wp.static(get_plane_box(contact_writer))(
    plane_normal,
    plane.pos,
    box.pos,
    box.rot,
    box.size,
    margin,
    write_contact_args,
  )


@wp.func
def plane_convex_wrapper(
  # In:
  plane: Geom,
  convex: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates contacts between a plane and a convex object."""
  plane_normal = wp.vec3(plane.rot[0, 2], plane.rot[1, 2], plane.rot[2, 2])
  return wp.static(get_plane_convex(contact_writer))(
    plane_normal,
    plane.pos,
    convex.pos,
    convex.rot,
    convex.vert,
    convex.vertadr,
    convex.vertnum,
    convex.graph,
    convex.graphadr,
    write_contact_args,
  )


@wp.func
def sphere_cylinder_wrapper(
  # In:
  sphere: Geom,
  cylinder: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates one contact between a sphere and a cylinder."""
  cylinder_axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2], cylinder.rot[2, 2])
  return wp.static(get_sphere_cylinder(contact_writer))(
    sphere.pos,
    sphere.size[0],
    sphere.rot,
    cylinder.pos,
    cylinder_axis,
    cylinder.size[0],
    cylinder.size[1],
    cylinder.rot,
    write_contact_args,
  )


@wp.func
def plane_cylinder_wrapper(
  # In:
  plane: Geom,
  cylinder: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates contacts between a cylinder and a plane."""
  plane_normal = wp.vec3(plane.rot[0, 2], plane.rot[1, 2], plane.rot[2, 2])
  cylinder_axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2], cylinder.rot[2, 2])
  return wp.static(get_plane_cylinder(contact_writer))(
    plane_normal,
    plane.pos,
    cylinder.pos,
    cylinder_axis,
    cylinder.size[0],
    cylinder.size[1],
    write_contact_args,
  )


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
  # Data in:
  collision_pair_in: wp.array(dtype=wp.vec2i),
  collision_pairid_in: wp.array(dtype=int),
  # In:
  cid: int,
  worldid: int,
):
  geoms = collision_pair_in[cid]
  pairid = collision_pairid_in[cid]

  if pairid > -1:
    margin = pair_margin[worldid, pairid]
    gap = pair_gap[worldid, pairid]
    condim = pair_dim[pairid]
    friction = pair_friction[worldid, pairid]
    solref = pair_solref[worldid, pairid]
    solreffriction = pair_solreffriction[worldid, pairid]
    solimp = pair_solimp[worldid, pairid]
  else:
    g1 = geoms[0]
    g2 = geoms[1]

    p1 = geom_priority[g1]
    p2 = geom_priority[g2]

    solmix1 = geom_solmix[worldid, g1]
    solmix2 = geom_solmix[worldid, g2]

    mix = solmix1 / (solmix1 + solmix2)
    mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
    mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
    mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)
    mix = wp.where(p1 == p2, mix, wp.where(p1 > p2, 1.0, 0.0))

    margin = wp.max(geom_margin[worldid, g1], geom_margin[worldid, g2])
    gap = wp.max(geom_gap[worldid, g1], geom_gap[worldid, g2])

    condim1 = geom_condim[g1]
    condim2 = geom_condim[g2]
    condim = wp.where(p1 == p2, wp.max(condim1, condim2), wp.where(p1 > p2, condim1, condim2))

    max_geom_friction = wp.max(geom_friction[worldid, g1], geom_friction[worldid, g2])
    friction = vec5(
      wp.max(MJ_MINMU, max_geom_friction[0]),
      wp.max(MJ_MINMU, max_geom_friction[0]),
      wp.max(MJ_MINMU, max_geom_friction[1]),
      wp.max(MJ_MINMU, max_geom_friction[2]),
      wp.max(MJ_MINMU, max_geom_friction[2]),
    )

    if geom_solref[worldid, g1].x > 0.0 and geom_solref[worldid, g2].x > 0.0:
      solref = mix * geom_solref[worldid, g1] + (1.0 - mix) * geom_solref[worldid, g2]
    else:
      solref = wp.min(geom_solref[worldid, g1], geom_solref[worldid, g2])

    solreffriction = wp.vec2(0.0, 0.0)

    solimp = mix * geom_solimp[worldid, g1] + (1.0 - mix) * geom_solimp[worldid, g2]

  return geoms, margin, gap, condim, friction, solref, solreffriction, solimp


@wp.func
def sphere_box_wrapper(
  # In:
  sphere: Geom,
  box: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates one contact between a sphere and a box."""
  return wp.static(get_sphere_box(contact_writer))(
    sphere.pos,
    sphere.size[0],
    box.pos,
    box.rot,
    box.size,
    margin,
    write_contact_args,
  )


@wp.func
def capsule_box_wrapper(
  # In:
  cap: Geom,
  box: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates contacts between a capsule and a box."""
  cap_axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  return wp.static(get_capsule_box(contact_writer))(
    cap.pos,
    cap_axis,
    cap.size[0],
    cap.size[1],
    box.pos,
    box.rot,
    box.size,
    margin,
    write_contact_args,
  )


@wp.func
def box_box_wrapper(
  # In:
  box1: Geom,
  box2: Geom,
  margin: float,
  write_contact_args: WriteContactArgs,
) -> int:
  """Calculates contacts between two boxes."""
  return wp.static(get_box_box(contact_writer))(
    box1.pos,
    box1.rot,
    box1.size,
    box2.pos,
    box2.rot,
    box2.size,
    margin,
    write_contact_args,
  )


_PRIMITIVE_COLLISIONS = {
  (GeomType.PLANE.value, GeomType.SPHERE.value): plane_sphere_wrapper,
  (GeomType.PLANE.value, GeomType.CAPSULE.value): plane_capsule_wrapper,
  (GeomType.PLANE.value, GeomType.ELLIPSOID.value): plane_ellipsoid_wrapper,
  (GeomType.PLANE.value, GeomType.CYLINDER.value): plane_cylinder_wrapper,
  (GeomType.PLANE.value, GeomType.BOX.value): plane_box_wrapper,
  (GeomType.PLANE.value, GeomType.MESH.value): plane_convex_wrapper,
  (GeomType.SPHERE.value, GeomType.SPHERE.value): sphere_sphere_wrapper,
  (GeomType.SPHERE.value, GeomType.CAPSULE.value): sphere_capsule_wrapper,
  (GeomType.SPHERE.value, GeomType.CYLINDER.value): sphere_cylinder_wrapper,
  (GeomType.SPHERE.value, GeomType.BOX.value): sphere_box_wrapper,
  (GeomType.CAPSULE.value, GeomType.CAPSULE.value): capsule_capsule_wrapper,
  (GeomType.CAPSULE.value, GeomType.BOX.value): capsule_box_wrapper,
  (GeomType.BOX.value, GeomType.BOX.value): box_box_wrapper,
}


# TODO(team): _check_collisions shared utility
def _check_primitive_collisions():
  prev_idx = -1
  for types in _PRIMITIVE_COLLISIONS.keys():
    idx = upper_trid_index(len(GeomType), types[0], types[1])
    if types[1] < types[0] or idx <= prev_idx:
      return False
    prev_idx = idx
  return True


assert _check_primitive_collisions(), "_PRIMITIVE_COLLISIONS is in invalid order"

_primitive_collisions_types = []
_primitive_collisions_func = []


def _primitive_narrowphase_builder(m: Model):
  for types, func in _PRIMITIVE_COLLISIONS.items():
    idx = upper_trid_index(len(GeomType), types[0], types[1])
    if m.geom_pair_type_count[idx] and types not in _primitive_collisions_types:
      _primitive_collisions_types.append(types)
      _primitive_collisions_func.append(func)

  @wp.kernel
  def _primitive_narrowphase(
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
    nconmax_in: int,
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    collision_pair_in: wp.array(dtype=wp.vec2i),
    collision_hftri_index_in: wp.array(dtype=int),
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

    geoms = collision_pair_in[tid]
    g1 = geoms[0]
    g2 = geoms[1]

    type1 = geom_type[g1]
    type2 = geom_type[g2]

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

    hftri_index = collision_hftri_index_in[tid]

    geom1 = _geom(
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
      g1,
      hftri_index,
    )

    geom2 = _geom(
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
      g2,
      hftri_index,
    )

    write_contact_args = WriteContactArgs()
    write_contact_args.nconmax_in = nconmax_in
    write_contact_args.margin_in = margin
    write_contact_args.gap_in = gap
    write_contact_args.condim_in = condim
    write_contact_args.friction_in = friction
    write_contact_args.solref_in = solref
    write_contact_args.solreffriction_in = solreffriction
    write_contact_args.solimp_in = solimp
    write_contact_args.geoms_in = geoms
    write_contact_args.worldid_in = worldid
    write_contact_args.ncon_out = ncon_out
    write_contact_args.contact_dist_out = contact_dist_out
    write_contact_args.contact_pos_out = contact_pos_out
    write_contact_args.contact_frame_out = contact_frame_out
    write_contact_args.contact_includemargin_out = contact_includemargin_out
    write_contact_args.contact_friction_out = contact_friction_out
    write_contact_args.contact_solref_out = contact_solref_out
    write_contact_args.contact_solreffriction_out = contact_solreffriction_out
    write_contact_args.contact_solimp_out = contact_solimp_out
    write_contact_args.contact_dim_out = contact_dim_out
    write_contact_args.contact_geom_out = contact_geom_out
    write_contact_args.contact_worldid_out = contact_worldid_out

    for i in range(wp.static(len(_primitive_collisions_func))):
      collision_type1 = wp.static(_primitive_collisions_types[i][0])
      collision_type2 = wp.static(_primitive_collisions_types[i][1])

      if collision_type1 == type1 and collision_type2 == type2:
        wp.static(_primitive_collisions_func[i])(
          geom1,
          geom2,
          margin,
          write_contact_args,
        )

  return _primitive_narrowphase


@event_scope
def primitive_narrowphase(m: Model, d: Data):
  """Runs collision detection on primitive geom pairs discovered during broadphase.

  This function processes collision pairs involving primitive shapes that were
  identified during the broadphase stage. It computes detailed contact information
  such as distance, position, and frame, and populates the `d.contact` array.

  The primitive geom types handled are PLANE, SPHERE, CAPSULE, CYLINDER, BOX.

  It also handles collisions between planes and convex hulls.

  To improve performance, it dynamically builds and launches a kernel tailored to
  the specific primitive collision types present in the model, avoiding
  unnecessary checks for non-existent collision pairs.
  """
  # we need to figure out how to keep the overhead of this small - not launching anything
  # for pair types without collisions, as well as updating the launch dimensions.
  wp.launch(
    _primitive_narrowphase_builder(m),
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
      d.nconmax,
      d.geom_xpos,
      d.geom_xmat,
      d.collision_pair,
      d.collision_hftri_index,
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
