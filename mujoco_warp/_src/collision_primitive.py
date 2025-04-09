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
from .types import Data
from .types import GeomType
from .types import Model

_MINVAL = 1e-15
_TINY_VAL = 1e-6


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
  geoms: wp.vec2i,
  worldid: int,
):
  active = (dist - margin) < 0
  if active:
    index = wp.atomic_add(d.ncon, 0, 1)
    if index < d.nconmax:
      d.contact.dist[index] = dist
      d.contact.pos[index] = pos
      d.contact.frame[index] = frame
      d.contact.geom[index] = geoms
      d.contact.worldid[index] = worldid


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
  geom_indices: wp.vec2i,
):
  dist, pos = _plane_sphere(plane.normal, plane.pos, sphere.pos, sphere.size[0])

  write_contact(d, dist, pos, make_frame(plane.normal), margin, geom_indices, worldid)


@wp.func
def _sphere_sphere(
  pos1: wp.vec3,
  radius1: float,
  pos2: wp.vec3,
  radius2: float,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  dir = pos2 - pos1
  dist = wp.length(dir)
  if dist == 0.0:
    n = wp.vec3(1.0, 0.0, 0.0)
  else:
    n = dir / dist
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)

  write_contact(d, dist, pos, make_frame(n), margin, geom_indices, worldid)


@wp.func
def sphere_sphere(
  sphere1: Geom,
  sphere2: Geom,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  _sphere_sphere(
    sphere1.pos,
    sphere1.size[0],
    sphere2.pos,
    sphere2.size[0],
    worldid,
    d,
    margin,
    geom_indices,
  )


@wp.func
def sphere_capsule(
  sphere: Geom,
  cap: Geom,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  """Calculates one contact between a sphere and a capsule."""
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  length = cap.size[1]
  segment = axis * length

  # Find closest point on capsule centerline to sphere center
  pt = closest_segment_point(cap.pos - segment, cap.pos + segment, sphere.pos)

  # Treat as sphere-sphere collision between sphere and closest point
  _sphere_sphere(
    sphere.pos, sphere.size[0], pt, cap.size[0], worldid, d, margin, geom_indices
  )


@wp.func
def capsule_capsule(
  cap1: Geom,
  cap2: Geom,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
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

  _sphere_sphere(pt1, cap1.size[0], pt2, cap2.size[0], worldid, d, margin, geom_indices)


@wp.func
def plane_capsule(
  plane: Geom,
  cap: Geom,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
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
  write_contact(d, dist1, pos1, frame, margin, geom_indices, worldid)

  dist2, pos2 = _plane_sphere(n, plane.pos, cap.pos - segment, cap.size[0])
  write_contact(d, dist2, pos2, frame, margin, geom_indices, worldid)


@wp.func
def plane_box(
  plane: Geom,
  box: Geom,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
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
    write_contact(d, cdist, pos, frame, margin, geom_indices, worldid)
    count += 1
    if count >= 4:
      break


@wp.func
def plane_cylinder(
  plane: Geom,
  cylinder: Geom,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
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
    write_contact(d, dist1, pos1, frame, margin, geom_indices, worldid)
  else:
    # If nearest point is above margin, no contacts
    return

  # Second contact point (end cap farther from plane)
  dist2 = dist0 - prjaxis + prjvec
  if dist2 <= margin:
    pos2 = cylinder.pos + vec - axis - n * (dist2 * 0.5)
    write_contact(d, dist2, pos2, frame, margin, geom_indices, worldid)

  # Try triangle contact points on side closer to plane
  prjvec1 = -prjvec * 0.5
  dist3 = dist0 + prjaxis + prjvec1
  if dist3 <= margin:
    # Compute sideways vector scaled by radius*sqrt(3)/2
    vec1 = wp.cross(vec, axis)
    vec1 = wp.normalize(vec1) * (cylinder.size[0] * wp.sqrt(3.0) * 0.5)

    # Add contact point A - adjust to closest side
    pos3 = cylinder.pos + vec1 + axis - vec * 0.5 - n * (dist3 * 0.5)
    write_contact(d, dist3, pos3, frame, margin, geom_indices, worldid)

    # Add contact point B - adjust to closest side
    pos4 = cylinder.pos - vec1 + axis - vec * 0.5 - n * (dist3 * 0.5)
    write_contact(d, dist3, pos4, frame, margin, geom_indices, worldid)


@wp.func
def _sphere_box_raw(
  sphere_pos: wp.vec3,
  sphere_size: float,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
  margin: float,
  geom_indices: wp.vec2i,
  worldid: int,
  d: Data,
):
  center = wp.transpose(box_rot) @ (sphere_pos - box_pos)

  clamped = wp.max(-box_size, wp.min(box_size, center))
  clamped_dir, dist = normalize_with_norm(clamped - center)

  if dist - sphere_size > margin:
    return

  # sphere center inside box
  if dist <= _TINY_VAL:
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
    geom_indices,
    worldid,
  )


@wp.func
def sphere_box(
  sphere: Geom,
  box: Geom,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  _sphere_box_raw(
    sphere.pos,
    sphere.size[0],
    box.pos,
    box.rot,
    box.size,
    margin,
    geom_indices,
    worldid,
    d,
  )


@wp.func
def capsule_box(
  cap: Geom,
  box: Geom,
  worldid: int,
  d: Data,
  margin: float,
  geom_indices: wp.vec2i,
):
  """Calculates contacts between a capsule and a box."""
  # Ported from the mjc implementation
  pos = wp.transpose(box.rot) @ (cap.pos - box.pos)
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  halfaxis = axis * cap.size[1]
  axisdir = (
    wp.int32(axis[0] > 0.0) + 2 * wp.int32(axis[1] > 0.0) + 4 * wp.int32(axis[2] > 0.0)
  )

  bestdistmax = margin + 2.0 * (
    cap.size[0] + cap.size[1] + box.size[0] + box.size[1] + box.size[2]
  )

  # keep track of closest point
  bestdist = wp.float32(bestdistmax)
  bestsegmentpos = wp.float32(-12)
  cltype = wp.int32(-4)
  clface = wp.int32(-12)

  # check if face closest
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

  # check edge edge

  clcorner = wp.int32(-123)  # which corner is the closest
  cledge = wp.int32(-123)  # which axis
  bestboxpos = wp.float32(0.0)

  for i in range(8):
    for j in range(3):
      if i & (1 << j) != 0:
        continue

      # c1<6 means that closest point on the box is at the lower end or in the middle of the edge

      c2 = wp.int32(-123)
      # trick to get a corner
      box_pt = wp.cw_mul(
        wp.vec3(
          wp.where(i & 1, 1.0, -1.0),
          wp.where(i & 2, 1.0, -1.0),
          wp.where(i & 4, 1.0, -1.0),
        ),
        box.size,
      )
      box_pt[j] = 0.0

      # box_pt is the starting point on the box
      # box_dir is the direction along the "j"-th axis
      # pos is the capsule's center
      # halfaxis is the capsule direction

      # find closest point between capsule and the edge
      dif = box_pt - pos

      u = -box.size[j] * dif[j]
      v = wp.dot(halfaxis, dif)
      ma = box.size[j] * box.size[j]
      mb = -box.size[j] * halfaxis[j]
      mc = cap.size[1] * cap.size[1]
      det = ma * mc - mb * mb
      if wp.abs(det) < _MINVAL:
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

      ct = s1 * 3 + s2

      dif_sq = wp.length_sq(dif)
      if dif_sq < bestdist - _MINVAL:
        bestdist = dif_sq
        bestsegmentpos = x2
        bestboxpos = x1
        # ct<6 means that closest point on the box is at the lower end or in the middle of the edge
        c2 = ct / 6
        # cltype /3 == 0 means the lower corner is closest to the capsule
        # cltype /3 == 2 means the upper corner is closest to the capsule
        # cltype /3 == 1 means the middle of the edge is closest to the capsule
        # cltype %3 == 0 means the lower corner is closest to the box
        # cltype %3 == 2 means the upper corner is closest to the box
        # cltype %3 == 1 means the middle of the capsule is closest to the box
        # note that edges include corners

        clcorner = i + (1 << j) * c2  # which corner is the closest
        cledge = j  # which axis
        cltype = ct  # save clamped info

  best = wp.float32(0.0)
  l = wp.float32(0.0)

  p = wp.vec2(pos.x, pos.y)
  dd = wp.vec2(halfaxis.x, halfaxis.y)
  s = wp.vec2(box.size.x, box.size.y)
  secondpos = wp.float32(-4.0)

  l = wp.length_sq(dd)

  uu = dd.x * s.y
  vv = dd.y * s.x
  # w = dd.x * p.y - dd.y * p.x
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
    # hack to find the relative orientation of capsule and corner
    # there are 2 cases:
    #    1: pointing to or away from the corner
    #    2: oriented along a face or an edge

    # case 1: no chance of additional contact
    if c1 != 0 and c1 != 7:
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

      if axis[ax]*axis[ax] > 0.5:  # second point along the edge of the box
        m = 2.0 * box.size[ax] / wp.abs(halfaxis[ax])
        secondpos = min(1.0 - wp.float32(mul) * bestsegmentpos, m)
      else:  # second point along a face of the box
        # check for overshoot again
        m = 2.0 * min(
          box.size[ax1] / wp.abs(halfaxis[ax1]),
          box.size[ax2] / wp.abs(halfaxis[ax2])
        )
        secondpos = -min(1.0 + wp.float32(mul) * bestsegmentpos, m)
      secondpos *= wp.float32(mul)

  elif cltype >= 0 and cltype / 3 == 1:  # we are on box's edge
    # hacks to find the relative orientation of capsule and edge
    # there are 2 cases:
    #    c1= 2^n: edge and capsule are oriented in a T configuration (no more contacts
    #    c1!=2^n: oriented in a cross X configuration
    c1 = axisdir ^ clcorner  # same trick

    c1 &= 7 - (1 << cledge)  # even more hacks

    if c1 == 1 or c1 == 2 or c1 == 4:
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

      # keep track of the axis orientation (mul will tell us which direction along the capsule to
      # find the second point) you can notice all other references to the axis "halfaxis" are with
      # absolute value

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

      if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):  # that is insane
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

    if clface != -1:
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

  # skip:
  # create sphere in original orientation at first contact point
  s1_pos_l = pos + halfaxis * bestsegmentpos
  s1_pos_g = box.rot @ s1_pos_l + box.pos

  # collide with sphere
  _sphere_box_raw(
    s1_pos_g, cap.size[0], box.pos, box.rot, box.size, margin, geom_indices, worldid, d
  )

  if secondpos > -3:  # secondpos was modified
    s2_pos_l = pos + halfaxis * (secondpos + bestsegmentpos)
    s2_pos_g = box.rot @ s2_pos_l + box.pos
    _sphere_box_raw(
      s2_pos_g, cap.size[0], box.pos, box.rot, box.size, margin, geom_indices, worldid, d
    )


@wp.kernel
def _primitive_narrowphase(
  m: Model,
  d: Data,
):
  tid = wp.tid()

  if tid >= d.ncollision[0]:
    return

  geoms = d.collision_pair[tid]
  worldid = d.collision_worldid[tid]

  g1 = geoms[0]
  g2 = geoms[1]
  type1 = m.geom_type[g1]
  type2 = m.geom_type[g2]

  geom1 = _geom(g1, m, d.geom_xpos[worldid], d.geom_xmat[worldid])
  geom2 = _geom(g2, m, d.geom_xpos[worldid], d.geom_xmat[worldid])

  margin = wp.max(m.geom_margin[g1], m.geom_margin[g2])

  # TODO(team): static loop unrolling to remove unnecessary branching
  if type1 == int(GeomType.PLANE.value) and type2 == int(GeomType.SPHERE.value):
    plane_sphere(geom1, geom2, worldid, d, margin, geoms)
  elif type1 == int(GeomType.SPHERE.value) and type2 == int(GeomType.SPHERE.value):
    sphere_sphere(geom1, geom2, worldid, d, margin, geoms)
  elif type1 == int(GeomType.PLANE.value) and type2 == int(GeomType.CAPSULE.value):
    plane_capsule(geom1, geom2, worldid, d, margin, geoms)
  elif type1 == int(GeomType.PLANE.value) and type2 == int(GeomType.BOX.value):
    plane_box(geom1, geom2, worldid, d, margin, geoms)
  elif type1 == int(GeomType.CAPSULE.value) and type2 == int(GeomType.CAPSULE.value):
    capsule_capsule(geom1, geom2, worldid, d, margin, geoms)
  elif type1 == int(GeomType.SPHERE.value) and type2 == int(GeomType.CAPSULE.value):
    sphere_capsule(geom1, geom2, worldid, d, margin, geoms)
  elif type1 == int(GeomType.SPHERE.value) and type2 == int(GeomType.BOX.value):
    sphere_box(geom1, geom2, worldid, d, margin, geoms)
  elif type1 == int(GeomType.PLANE.value) and type2 == int(GeomType.CYLINDER.value):
    plane_cylinder(geom1, geom2, worldid, d, margin, geoms)
  elif type1 == int(GeomType.CAPSULE.value) and type2 == int(GeomType.BOX.value):
    capsule_box(geom1, geom2, worldid, d, margin, geoms)


def primitive_narrowphase(m: Model, d: Data):
  # we need to figure out how to keep the overhead of this small - not launching anything
  # for pair types without collisions, as well as updating the launch dimensions.
  wp.launch(_primitive_narrowphase, dim=d.nconmax, inputs=[m, d])
