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

from .collision_hfield import HFPrism, get_hfield_triangle_prism
from .math import (closest_segment_point, closest_segment_to_segment_points,
                   make_frame, normalize_with_norm)
from .types import MJ_MINVAL, Data, GeomType, Model, vec5

wp.set_module_options({"enable_backward": False})


@wp.struct
class Geom:
  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3
  size: wp.vec3
  vertadr: int
  vertnum: int
  prism: HFPrism # Triangle prism vertices


@wp.func
def _geom(
  gid: int,
  m: Model,
  geom_xpos: wp.array(dtype=wp.vec3),
  geom_xmat: wp.array(dtype=wp.mat33),
  index: int = -1,
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

  # If geom is HFIELD, store 6 vertices of the tri prism
  if m.geom_type[gid] == int(GeomType.HFIELD.value):
    geom.prism = get_hfield_triangle_prism(m, gid, index)

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
def _sphere_box(
  sphere_pos: wp.vec3,
  sphere_size: float,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
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
  center = wp.transpose(box_rot) @ (sphere_pos - box_pos)

  clamped = wp.max(-box_size, wp.min(box_size, center))
  clamped_dir, dist = normalize_with_norm(clamped - center)

  if dist - sphere_size > margin:
    return

  # sphere center inside box
  if dist <= MJ_MINVAL:
    closest = 2.0 * (box_size[0] + box_size[1] + box_size[2])
    k = wp.int32(0)
    for i in range(6):
      face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i / 2] - center[i / 2])
      if closest > face_dist:
        closest = face_dist
        k = i

    nearest = wp.vec3(0.0)
    nearest[k / 2] = wp.where(k % 2, -1.0, 1.0)
    pos = center + nearest * (sphere_size - closest) / 2.0
    contact_normal = box_rot @ nearest
    contact_dist = -closest - sphere_size

  else:
    deepest = center + clamped_dir * sphere_size
    pos = 0.5 * (clamped + deepest)
    contact_normal = box_rot @ clamped_dir
    contact_dist = dist - sphere_size

  contact_pos = box_pos + box_rot @ pos
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
  _sphere_box(
    sphere.pos,
    sphere.size[0],
    box.pos,
    box.rot,
    box.size,
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
def capsule_box(
  cap: Geom,
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
  """Calculates contacts between a capsule and a box."""
  # Based on the mjc implementation
  pos = wp.transpose(box.rot) @ (cap.pos - box.pos)
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  halfaxis = axis * cap.size[1]  # halfaxis is the capsule direction
  axisdir = (
    wp.int32(axis[0] > 0.0) + 2 * wp.int32(axis[1] > 0.0) + 4 * wp.int32(axis[2] > 0.0)
  )

  bestdistmax = margin + 2.0 * (
    cap.size[0] + cap.size[1] + box.size[0] + box.size[1] + box.size[2]
  )

  # keep track of closest point
  bestdist = wp.float32(bestdistmax)
  bestsegmentpos = wp.float32(-12)

  # cltype: encoded collision configuration
  # cltype / 3 == 0 : lower corner is closest to the capsule
  #            == 2 : upper corner is closest to the capsule
  #            == 1 : middle of the edge is closest to the capsule
  # cltype % 3 == 0 : lower corner is closest to the box
  #            == 2 : upper corner is closest to the box
  #            == 1 : middle of the capsule is closest to the box
  cltype = wp.int32(-4)

  # clface: index of the closest face of the box to the capsule
  # -1: no face is closest (edge or corner is closest)
  # 0, 1, 2: index of the axis perpendicular to the closest face
  clface = wp.int32(-12)

  # first: consider cases where a face of the box is closest
  for i in range(-1, 2, 2):
    axisTip = pos + wp.float32(i) * halfaxis
    boxPoint = wp.vec3(axisTip)

    n_out = wp.int32(0)
    ax_out = wp.int32(-1)

    for j in range(3):
      if boxPoint[j] < -box.size[j]:
        n_out += 1
        ax_out = j
        boxPoint[j] = -box.size[j]
      elif boxPoint[j] > box.size[j]:
        n_out += 1
        ax_out = j
        boxPoint[j] = box.size[j]

    if n_out > 1:
      continue

    dist = wp.length_sq(boxPoint - axisTip)

    if dist < bestdist:
      bestdist = dist
      bestsegmentpos = wp.float32(i)
      cltype = -2 + i
      clface = ax_out

  # second: consider cases where an edge of the box is closest
  clcorner = wp.int32(-123)  # which corner is the closest
  cledge = wp.int32(-123)  # which axis
  bestboxpos = wp.float32(0.0)

  for i in range(8):
    for j in range(3):
      if i & (1 << j) != 0:
        continue

      c2 = wp.int32(-123)

      # box_pt is the starting point (corner) on the box
      box_pt = wp.cw_mul(
        wp.vec3(
          wp.where(i & 1, 1.0, -1.0),
          wp.where(i & 2, 1.0, -1.0),
          wp.where(i & 4, 1.0, -1.0),
        ),
        box.size,
      )
      box_pt[j] = 0.0

      # find closest point between capsule and the edge
      dif = box_pt - pos

      u = -box.size[j] * dif[j]
      v = wp.dot(halfaxis, dif)
      ma = box.size[j] * box.size[j]
      mb = -box.size[j] * halfaxis[j]
      mc = cap.size[1] * cap.size[1]
      det = ma * mc - mb * mb
      if wp.abs(det) < MJ_MINVAL:
        continue

      idet = 1.0 / det
      # sX : X=1 means middle of segment. X=0 or 2 one or the other end

      x1 = wp.float32((mc * u - mb * v) * idet)
      x2 = wp.float32((ma * v - mb * u) * idet)

      s1 = wp.int32(1)
      s2 = wp.int32(1)

      if x1 > 1:
        x1 = 1.0
        s1 = 2
        x2 = (v - mb) / mc
      elif x1 < -1:
        x1 = -1.0
        s1 = 0
        x2 = (v + mb) / mc

      x2_over = x2 > 1.0
      if x2_over or x2 < -1.0:
        if x2_over:
          x2 = 1.0
          s2 = 2
          x1 = (u - mb) / ma
        else:
          x2 = -1.0
          s2 = 0
          x1 = (u + mb) / ma

        if x1 > 1:
          x1 = 1.0
          s1 = 2
        elif x1 < -1:
          x1 = -1.0
          s1 = 0

      dif -= halfaxis * x2
      dif[j] += box.size[j] * x1

      # encode relative positions of the closest points
      ct = s1 * 3 + s2

      dif_sq = wp.length_sq(dif)
      if dif_sq < bestdist - MJ_MINVAL:
        bestdist = dif_sq
        bestsegmentpos = x2
        bestboxpos = x1
        # ct<6 means closest point on box is at lower end or middle of edge
        c2 = ct / 6

        clcorner = i + (1 << j) * c2  # index of closest box corner
        cledge = j  # axis index of closest box edge
        cltype = ct  # encoded collision configuration

  best = wp.float32(0.0)
  l = wp.float32(0.0)

  p = wp.vec2(pos.x, pos.y)
  dd = wp.vec2(halfaxis.x, halfaxis.y)
  s = wp.vec2(box.size.x, box.size.y)
  secondpos = wp.float32(-4.0)

  l = wp.length_sq(dd)

  uu = dd.x * s.y
  vv = dd.y * s.x
  w_neg = dd.x * p.y - dd.y * p.x < 0

  best = wp.float32(-1.0)

  ee1 = uu - vv
  ee2 = uu + vv

  if wp.abs(ee1) > best:
    best = wp.abs(ee1)
    c1 = wp.where((ee1 < 0) == w_neg, 0, 3)

  if wp.abs(ee2) > best:
    best = wp.abs(ee2)
    c1 = wp.where((ee2 > 0) == w_neg, 1, 2)

  if cltype == -4:  # invalid type
    return

  if cltype >= 0 and cltype / 3 != 1:  # closest to a corner of the box
    c1 = axisdir ^ clcorner
    # Calculate relative orientation between capsule and corner
    # There are two possible configurations:
    # 1. Capsule axis points toward/away from corner
    # 2. Capsule axis aligns with a face or edge
    if c1 != 0 and c1 != 7:  # create second contact point
      if c1 == 1 or c1 == 2 or c1 == 4:
        mul = 1
      else:
        mul = -1
        c1 = 7 - c1

      # "de" and "dp" distance from first closest point on the capsule to both ends of it
      # mul is a direction along the capsule's axis

      if c1 == 1:
        ax = 0
        ax1 = 1
        ax2 = 2
      elif c1 == 2:
        ax = 1
        ax1 = 2
        ax2 = 0
      elif c1 == 4:
        ax = 2
        ax1 = 0
        ax2 = 1

      if axis[ax] * axis[ax] > 0.5:  # second point along the edge of the box
        m = 2.0 * box.size[ax] / wp.abs(halfaxis[ax])
        secondpos = min(1.0 - wp.float32(mul) * bestsegmentpos, m)
      else:  # second point along a face of the box
        # check for overshoot again
        m = 2.0 * min(
          box.size[ax1] / wp.abs(halfaxis[ax1]), box.size[ax2] / wp.abs(halfaxis[ax2])
        )
        secondpos = -min(1.0 + wp.float32(mul) * bestsegmentpos, m)
      secondpos *= wp.float32(mul)

  elif cltype >= 0 and cltype / 3 == 1:  # we are on box's edge
    # Calculate relative orientation between capsule and edge
    # Two possible configurations:
    # - T configuration: c1 = 2^n (no additional contacts)
    # - X configuration: c1 != 2^n (potential additional contacts)
    c1 = axisdir ^ clcorner
    c1 &= 7 - (1 << cledge)  # mask out edge axis to determine configuration

    if c1 == 1 or c1 == 2 or c1 == 4:  # create second contact point
      if cledge == 0:
        ax1 = 1
        ax2 = 2
      if cledge == 1:
        ax1 = 2
        ax2 = 0
      if cledge == 2:
        ax1 = 0
        ax2 = 1
      ax = cledge

      # Then it finds with which face the capsule has a lower angle and switches the axis names
      if wp.abs(axis[ax1]) > wp.abs(axis[ax2]):
        ax1 = ax2
      ax2 = 3 - ax - ax1

      # mul determines direction along capsule axis for second contact point
      if c1 & (1 << ax2):
        mul = 1
        secondpos = 1.0 - bestsegmentpos
      else:
        mul = -1
        secondpos = 1.0 + bestsegmentpos

      # now we have to find out whether we point towards the opposite side or towards one of the
      # sides and also find the farthest point along the capsule that is above the box

      e1 = 2.0 * box.size[ax2] / wp.abs(halfaxis[ax2])
      secondpos = min(e1, secondpos)

      if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):
        e2 = 1.0 - bestboxpos
      else:
        e2 = 1.0 + bestboxpos

      e1 = box.size[ax] * e2 / wp.abs(halfaxis[ax])

      secondpos = min(e1, secondpos)
      secondpos *= wp.float32(mul)

  elif cltype < 0:
    # similarly we handle the case when one capsule's end is closest to a face of the box
    # and find where is the other end pointing to and clamping to the farthest point
    # of the capsule that's above the box
    # if the closest point is inside the box there's no need for a second point

    if clface != -1:  # create second contact point
      mul = wp.where(cltype == -3, 1, -1)
      secondpos = 2.0

      tmp1 = pos - halfaxis * wp.float32(mul)

      for i in range(3):
        if i != clface:
          ha_r = wp.float32(mul) / halfaxis[i]
          e1 = (box.size[i] - tmp1[i]) * ha_r
          if 0 < e1 and e1 < secondpos:
            secondpos = e1

          e1 = (-box.size[i] - tmp1[i]) * ha_r
          if 0 < e1 and e1 < secondpos:
            secondpos = e1

      secondpos *= wp.float32(mul)

  # create sphere in original orientation at first contact point
  s1_pos_l = pos + halfaxis * bestsegmentpos
  s1_pos_g = box.rot @ s1_pos_l + box.pos

  # collide with sphere
  _sphere_box(
    s1_pos_g,
    cap.size[0],
    box.pos,
    box.rot,
    box.size,
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

  if secondpos > -3:  # secondpos was modified
    s2_pos_l = pos + halfaxis * (secondpos + bestsegmentpos)
    s2_pos_g = box.rot @ s2_pos_l + box.pos
    _sphere_box(
      s2_pos_g,
      cap.size[0],
      box.pos,
      box.rot,
      box.size,
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
  elif type1 == int(GeomType.CAPSULE.value) and type2 == int(GeomType.BOX.value):
    capsule_box(
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
