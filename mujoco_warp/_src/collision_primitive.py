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

from .math import closest_segment_to_segment_points
from .math import make_frame
from .math import normalize_with_norm
from .types import Data
from .types import GeomType
from .types import Model


@wp.struct
class Geom:
  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3
  size: wp.vec3
  # TODO(team): mesh fields: vertadr, vertnum


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


def primitive_narrowphase(m: Model, d: Data):
  # we need to figure out how to keep the overhead of this small - not launching anything
  # for pair types without collisions, as well as updating the launch dimensions.
  wp.launch(_primitive_narrowphase, dim=d.nconmax, inputs=[m, d], device=m.device)
