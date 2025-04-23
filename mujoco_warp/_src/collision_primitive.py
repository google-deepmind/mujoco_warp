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

from .math import closest_segment_point
from .math import closest_segment_to_segment_points
from .math import make_frame
from .math import normalize_with_norm
from .types import MJ_MINVAL
from .types import Data
from .types import GeomType
from .types import Model
from .types import vec5

wp.set_module_options({"enable_backward": False})

wp.set_module_options({"enable_backward": False})


@wp.struct
class Geom:
  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3
  size: wp.vec3
  vertadr: int
  vertnum: int


@wp.func
def _geom(
  gid: int,
  m: Model,
  geom_xpos: wp.array(dtype=wp.vec3),
  geom_xmat: wp.array(dtype=wp.mat33),
) -> Geom:
  geom = Geom()
  geom.pos = geom_xpos[gid]
  rot = geom_xmat[gid]
  geom.rot = rot
  geom.size = m.geom_size[gid]
  geom.normal = wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])  # plane
  dataid = m.geom_dataid[gid]
  if dataid >= 0:
    geom.vertadr = m.mesh_vertadr[dataid]
    geom.vertnum = m.mesh_vertnum[dataid]
  else:
    geom.vertadr = -1
    geom.vertnum = -1

  return geom


@wp.func
def write_contact(
  d: Data,
  dist: float,
  pos: wp.vec3,
  frame: wp.mat33,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
  worldid: int,
):
  active = (dist - margin) < 0
  if active:
    cid = wp.atomic_add(d.ncon, 0, 1)
    if cid < d.nconmax:
      d.contact.dist[cid] = dist
      d.contact.pos[cid] = pos
      d.contact.frame[cid] = frame
      d.contact.geom[cid] = geoms
      d.contact.worldid[cid] = worldid
      d.contact.includemargin[cid] = margin - gap
      d.contact.dim[cid] = condim
      d.contact.friction[cid] = friction
      d.contact.solref[cid] = solref
      d.contact.solreffriction[cid] = solreffriction
      d.contact.solimp[cid] = solimp


@wp.func
def _plane_sphere(
  plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float
):
  dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@wp.func
def plane_sphere(
  plane: Geom,
  sphere: Geom,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
):
  dist, pos = _plane_sphere(plane.normal, plane.pos, sphere.pos, sphere.size[0])

  write_contact(
    d,
    dist,
    pos,
    make_frame(plane.normal),
    margin,
    gap,
    condim,
    friction,
    solref,
    solreffriction,
    solimp,
    geoms,
    worldid,
  )


@wp.func
def _sphere_sphere(
  pos1: wp.vec3,
  radius1: float,
  pos2: wp.vec3,
  radius2: float,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
):
  dir = pos2 - pos1
  dist = wp.length(dir)
  if dist == 0.0:
    n = wp.vec3(1.0, 0.0, 0.0)
  else:
    n = dir / dist
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)

  write_contact(
    d,
    dist,
    pos,
    make_frame(n),
    margin,
    gap,
    condim,
    friction,
    solref,
    solreffriction,
    solimp,
    geoms,
    worldid,
  )


@wp.func
def _sphere_sphere_ext(
  pos1: wp.vec3,
  radius1: float,
  pos2: wp.vec3,
  radius2: float,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
  mat1: wp.mat33,
  mat2: wp.mat33,
):
  dir = pos2 - pos1
  dist = wp.length(dir)
  if dist == 0.0:
    # Use cross product of z axes like MuJoCo
    axis1 = wp.vec3(mat1[0, 2], mat1[1, 2], mat1[2, 2])
    axis2 = wp.vec3(mat2[0, 2], mat2[1, 2], mat2[2, 2])
    n = wp.cross(axis1, axis2)
    n = wp.normalize(n)
  else:
    n = dir / dist
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)

  write_contact(
    d,
    dist,
    pos,
    make_frame(n),
    margin,
    gap,
    condim,
    friction,
    solref,
    solreffriction,
    solimp,
    geoms,
    worldid,
  )


@wp.func
def sphere_sphere(
  sphere1: Geom,
  sphere2: Geom,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
):
  _sphere_sphere(
    sphere1.pos,
    sphere1.size[0],
    sphere2.pos,
    sphere2.size[0],
    worldid,
    d,
    margin,
    gap,
    condim,
    friction,
    solref,
    solreffriction,
    solimp,
    geoms,
  )


@wp.func
def sphere_capsule(
  sphere: Geom,
  cap: Geom,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
):
  """Calculates one contact between a sphere and a capsule."""
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  length = cap.size[1]
  segment = axis * length

  # Find closest point on capsule centerline to sphere center
  pt = closest_segment_point(cap.pos - segment, cap.pos + segment, sphere.pos)

  # Treat as sphere-sphere collision between sphere and closest point
  _sphere_sphere(
    sphere.pos,
    sphere.size[0],
    pt,
    cap.size[0],
    worldid,
    d,
    margin,
    gap,
    condim,
    friction,
    solref,
    solreffriction,
    solimp,
    geoms,
  )


@wp.func
def capsule_capsule(
  cap1: Geom,
  cap2: Geom,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
):
  axis1 = wp.vec3(cap1.rot[0, 2], cap1.rot[1, 2], cap1.rot[2, 2])
  axis2 = wp.vec3(cap2.rot[0, 2], cap2.rot[1, 2], cap2.rot[2, 2])
  length1 = cap1.size[1]
  length2 = cap2.size[1]
  seg1 = axis1 * length1
  seg2 = axis2 * length2

  pt1, pt2 = closest_segment_to_segment_points(
    cap1.pos - seg1,
    cap1.pos + seg1,
    cap2.pos - seg2,
    cap2.pos + seg2,
  )

  _sphere_sphere(
    pt1,
    cap1.size[0],
    pt2,
    cap2.size[0],
    worldid,
    d,
    margin,
    gap,
    condim,
    friction,
    solref,
    solreffriction,
    solimp,
    geoms,
  )


@wp.func
def plane_capsule(
  plane: Geom,
  cap: Geom,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
):
  """Calculates two contacts between a capsule and a plane."""
  n = plane.normal
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  # align contact frames with capsule axis
  b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))

  if b_norm < 0.5:
    if -0.5 < n[1] and n[1] < 0.5:
      b = wp.vec3(0.0, 1.0, 0.0)
    else:
      b = wp.vec3(0.0, 0.0, 1.0)

  c = wp.cross(n, b)
  frame = wp.mat33(n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2])
  segment = axis * cap.size[1]

  dist1, pos1 = _plane_sphere(n, plane.pos, cap.pos + segment, cap.size[0])
  write_contact(
    d,
    dist1,
    pos1,
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
  )

  dist2, pos2 = _plane_sphere(n, plane.pos, cap.pos - segment, cap.size[0])
  write_contact(
    d,
    dist2,
    pos2,
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
  )


@wp.func
def plane_box(
  plane: Geom,
  box: Geom,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
):
  count = int(0)
  corner = wp.vec3()
  dist = wp.dot(box.pos - plane.pos, plane.normal)

  # test all corners, pick bottom 4
  for i in range(8):
    # get corner in local coordinates
    corner.x = wp.where(i & 1, box.size.x, -box.size.x)
    corner.y = wp.where(i & 2, box.size.y, -box.size.y)
    corner.z = wp.where(i & 4, box.size.z, -box.size.z)

    # get corner in global coordinates relative to box center
    corner = box.rot * corner

    # compute distance to plane, skip if too far or pointing up
    ldist = wp.dot(plane.normal, corner)
    if dist + ldist > margin or ldist > 0:
      continue

    cdist = dist + ldist
    frame = make_frame(plane.normal)
    pos = corner + box.pos + (plane.normal * cdist / -2.0)
    write_contact(
      d,
      cdist,
      pos,
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
    )
    count += 1
    if count >= 4:
      break


@wp.func
def sphere_cylinder(
  sphere: Geom,
  cylinder: Geom,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
):
  axis = wp.vec3(
    cylinder.rot[0, 2],
    cylinder.rot[1, 2],
    cylinder.rot[2, 2],
  )

  vec = sphere.pos - cylinder.pos
  x = wp.dot(vec, axis)

  a_proj = axis * x
  p_proj = vec - a_proj
  p_proj_sqr = wp.dot(p_proj, p_proj)

  collide_side = wp.abs(x) < cylinder.size[1]
  collide_cap = p_proj_sqr < (cylinder.size[0] * cylinder.size[0])

  if collide_side and collide_cap:
    dist_cap = cylinder.size[1] - wp.abs(x)
    dist_radius = cylinder.size[0] - wp.sqrt(p_proj_sqr)

    if dist_cap < dist_radius:
      collide_side = False
    else:
      collide_cap = False

  # Side collision
  if collide_side:
    pos_target = cylinder.pos + a_proj
    _sphere_sphere_ext(
      sphere.pos,
      sphere.size[0],
      pos_target,
      cylinder.size[0],
      worldid,
      d,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
      sphere.rot,
      cylinder.rot,
    )
    return

  # Cap collision
  if collide_cap:
    if x > 0.0:
      # top cap
      pos_cap = cylinder.pos + axis * cylinder.size[1]
      plane_normal = axis
    else:
      # bottom cap
      pos_cap = cylinder.pos - axis * cylinder.size[1]
      plane_normal = -axis

    dist, pos_contact = _plane_sphere(plane_normal, pos_cap, sphere.pos, sphere.size[0])
    plane_normal = -plane_normal  # Flip normal after position calculation

    write_contact(
      d,
      dist,
      pos_contact,
      make_frame(plane_normal),
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
      worldid,
    )

    return

  # Corner collision
  inv_len = 1.0 / wp.sqrt(p_proj_sqr)
  p_proj = p_proj * (cylinder.size[0] * inv_len)

  cap_offset = axis * (wp.sign(x) * cylinder.size[1])
  pos_corner = cylinder.pos + cap_offset + p_proj

  _sphere_sphere_ext(
    sphere.pos,
    sphere.size[0],
    pos_corner,
    0.0,
    worldid,
    d,
    margin,
    gap,
    condim,
    friction,
    solref,
    solreffriction,
    solimp,
    geoms,
    sphere.rot,
    cylinder.rot,
  )


@wp.func
def plane_cylinder(
  plane: Geom,
  cylinder: Geom,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
):
  """Calculates contacts between a cylinder and a plane."""
  # Extract plane normal and cylinder axis
  n = plane.normal
  axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2], cylinder.rot[2, 2])

  # Project, make sure axis points toward plane
  prjaxis = wp.dot(n, axis)
  if prjaxis > 0:
    axis = -axis
    prjaxis = -prjaxis

  # Compute normal distance from plane to cylinder center
  dist0 = wp.dot(cylinder.pos - plane.pos, n)

  # Remove component of -normal along cylinder axis
  vec = axis * prjaxis - n
  len_sqr = wp.dot(vec, vec)

  # If vector is nondegenerate, normalize and scale by radius
  # Otherwise use cylinder's x-axis scaled by radius
  vec = wp.where(
    len_sqr >= 1e-12,
    vec * (cylinder.size[0] / wp.sqrt(len_sqr)),
    wp.vec3(cylinder.rot[0, 0], cylinder.rot[1, 0], cylinder.rot[2, 0])
    * cylinder.size[0],
  )

  # Project scaled vector on normal
  prjvec = wp.dot(vec, n)

  # Scale cylinder axis by half-length
  axis = axis * cylinder.size[1]
  prjaxis = prjaxis * cylinder.size[1]

  frame = make_frame(n)

  # First contact point (end cap closer to plane)
  dist1 = dist0 + prjaxis + prjvec
  if dist1 <= margin:
    pos1 = cylinder.pos + vec + axis - n * (dist1 * 0.5)
    write_contact(
      d,
      dist1,
      pos1,
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
    )
  else:
    # If nearest point is above margin, no contacts
    return

  # Second contact point (end cap farther from plane)
  dist2 = dist0 - prjaxis + prjvec
  if dist2 <= margin:
    pos2 = cylinder.pos + vec - axis - n * (dist2 * 0.5)
    write_contact(
      d,
      dist2,
      pos2,
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
    )

  # Try triangle contact points on side closer to plane
  prjvec1 = -prjvec * 0.5
  dist3 = dist0 + prjaxis + prjvec1
  if dist3 <= margin:
    # Compute sideways vector scaled by radius*sqrt(3)/2
    vec1 = wp.cross(vec, axis)
    vec1 = wp.normalize(vec1) * (cylinder.size[0] * wp.sqrt(3.0) * 0.5)

    # Add contact point A - adjust to closest side
    pos3 = cylinder.pos + vec1 + axis - vec * 0.5 - n * (dist3 * 0.5)
    write_contact(
      d,
      dist3,
      pos3,
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
    )

    # Add contact point B - adjust to closest side
    pos4 = cylinder.pos - vec1 + axis - vec * 0.5 - n * (dist3 * 0.5)
    write_contact(
      d,
      dist3,
      pos4,
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
    )


@wp.func
def contact_params(m: Model, d: Data, cid: int):
  geoms = d.collision_pair[cid]
  pairid = d.collision_pairid[cid]

  if pairid > -1:
    margin = m.pair_margin[pairid]
    gap = m.pair_gap[pairid]
    condim = m.pair_dim[pairid]
    friction = m.pair_friction[pairid]
    solref = m.pair_solref[pairid]
    solreffriction = m.pair_solreffriction[pairid]
    solimp = m.pair_solimp[pairid]
  else:
    g1 = geoms[0]
    g2 = geoms[1]

    p1 = m.geom_priority[g1]
    p2 = m.geom_priority[g2]

    solmix1 = m.geom_solmix[g1]
    solmix2 = m.geom_solmix[g2]

    mix = solmix1 / (solmix1 + solmix2)
    mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
    mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
    mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)
    mix = wp.where(p1 == p2, mix, wp.where(p1 > p2, 1.0, 0.0))

    margin = wp.max(m.geom_margin[g1], m.geom_margin[g2])
    gap = wp.max(m.geom_gap[g1], m.geom_gap[g2])

    condim1 = m.geom_condim[g1]
    condim2 = m.geom_condim[g2]
    condim = wp.where(
      p1 == p2, wp.max(condim1, condim2), wp.where(p1 > p2, condim1, condim2)
    )

    geom_friction = wp.max(m.geom_friction[g1], m.geom_friction[g2])
    friction = vec5(
      geom_friction[0],
      geom_friction[0],
      geom_friction[1],
      geom_friction[2],
      geom_friction[2],
    )

    if m.geom_solref[g1].x > 0.0 and m.geom_solref[g2].x > 0.0:
      solref = mix * m.geom_solref[g1] + (1.0 - mix) * m.geom_solref[g2]
    else:
      solref = wp.min(m.geom_solref[g1], m.geom_solref[g2])

    solreffriction = wp.vec2(0.0, 0.0)

    solimp = mix * m.geom_solimp[g1] + (1.0 - mix) * m.geom_solimp[g2]

  return geoms, margin, gap, condim, friction, solref, solreffriction, solimp


@wp.func
def sphere_box(
  sphere: Geom,
  box: Geom,
  worldid: int,
  d: Data,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
):
  center = wp.transpose(box.rot) @ (sphere.pos - box.pos)

  clamped = wp.max(-box.size, wp.min(box.size, center))
  clamped_dir, dist = normalize_with_norm(clamped - center)

  if dist - sphere.size[0] > margin:
    return

  # sphere center inside box
  if dist <= MJ_MINVAL:
    closest = 2.0 * (box.size[0] + box.size[1] + box.size[2])
    k = wp.int32(0)
    for i in range(6):
      face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box.size[i / 2] - center[i / 2])
      if closest > face_dist:
        closest = face_dist
        k = i

    nearest = wp.vec3(0.0)
    nearest[k / 2] = wp.where(k % 2, -1.0, 1.0)
    pos = center + nearest * (sphere.size[0] - closest) / 2.0
    contact_normal = box.rot @ nearest
    contact_dist = -closest - sphere.size[0]

  else:
    deepest = center + clamped_dir * sphere.size[0]
    pos = 0.5 * (clamped + deepest)
    contact_normal = box.rot @ clamped_dir
    contact_dist = dist - sphere.size[0]

  contact_pos = box.pos + box.rot @ pos
  write_contact(
    d,
    contact_dist,
    contact_pos,
    make_frame(contact_normal),
    margin,
    gap,
    condim,
    friction,
    solref,
    solreffriction,
    solimp,
    geoms,
    worldid,
  )


@wp.kernel
def _primitive_narrowphase(
  m: Model,
  d: Data,
):
  tid = wp.tid()

  if tid >= d.ncollision[0]:
    return

  geoms, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
    m, d, tid
  )
  g1 = geoms[0]
  g2 = geoms[1]

  worldid = d.collision_worldid[tid]

  geom1 = _geom(g1, m, d.geom_xpos[worldid], d.geom_xmat[worldid])
  geom2 = _geom(g2, m, d.geom_xpos[worldid], d.geom_xmat[worldid])

  type1 = m.geom_type[g1]
  type2 = m.geom_type[g2]

  # TODO(team): static loop unrolling to remove unnecessary branching
  if type1 == int(GeomType.PLANE.value) and type2 == int(GeomType.SPHERE.value):
    plane_sphere(
      geom1,
      geom2,
      worldid,
      d,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
    )
  elif type1 == int(GeomType.SPHERE.value) and type2 == int(GeomType.SPHERE.value):
    sphere_sphere(
      geom1,
      geom2,
      worldid,
      d,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
    )
  elif type1 == int(GeomType.PLANE.value) and type2 == int(GeomType.CAPSULE.value):
    plane_capsule(
      geom1,
      geom2,
      worldid,
      d,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
    )
  elif type1 == int(GeomType.PLANE.value) and type2 == int(GeomType.BOX.value):
    plane_box(
      geom1,
      geom2,
      worldid,
      d,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
    )
  elif type1 == int(GeomType.CAPSULE.value) and type2 == int(GeomType.CAPSULE.value):
    capsule_capsule(
      geom1,
      geom2,
      worldid,
      d,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
    )
  elif type1 == int(GeomType.SPHERE.value) and type2 == int(GeomType.CAPSULE.value):
    sphere_capsule(
      geom1,
      geom2,
      worldid,
      d,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
    )
  elif type1 == int(GeomType.SPHERE.value) and type2 == int(GeomType.CYLINDER.value):
    sphere_cylinder(
      geom1,
      geom2,
      worldid,
      d,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
    )
  elif type1 == int(GeomType.SPHERE.value) and type2 == int(GeomType.BOX.value):
    sphere_box(
      geom1,
      geom2,
      worldid,
      d,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
    )
  elif type1 == int(GeomType.PLANE.value) and type2 == int(GeomType.CYLINDER.value):
    plane_cylinder(
      geom1,
      geom2,
      worldid,
      d,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
    )


def primitive_narrowphase(m: Model, d: Data):
  # we need to figure out how to keep the overhead of this small - not launching anything
  # for pair types without collisions, as well as updating the launch dimensions.
  wp.launch(_primitive_narrowphase, dim=d.nconmax, inputs=[m, d])
