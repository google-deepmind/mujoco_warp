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

import mujoco
import warp as wp

from .types import Data
from .types import GeomType
from .types import MJ_MINVAL
from .types import Model

from .types import MJ_MINVAL



@wp.struct
class DistanceWithId:
  dist: wp.float32
  geom_id: wp.int32


@wp.func
def _ray_quad(a: float, b: float, c: float) -> wp.vec2:
  """Returns two solutions for quadratic: a*x^2 + 2*b*x + c = 0."""
  det = b * b - a * c
  det_2 = wp.sqrt(det)

  x0 = (-b - det_2) / a
  x1 = (-b + det_2) / a
  x0 = wp.where((det < MJ_MINVAL) or (x0 < 0.0), wp.inf, x0)
  x1 = wp.where((det < MJ_MINVAL) or (x1 < 0.0), wp.inf, x1)

  return wp.vec2(x0, x1)


@wp.func
def _ray_plane(
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
  geom_id: int,
) -> DistanceWithId:
  """Returns the distance at which a ray intersects with a plane."""
  x = -pnt[2] / vec[2]

  valid = vec[2] <= -MJ_MINVAL  # z-vec pointing towards front face
  valid = valid and x >= 0.0
  # only within rendered rectangle
  p_x = pnt[0] + x * vec[0]
  p_y = pnt[1] + x * vec[1]
  valid = valid and ((size[0] <= 0.0) or (wp.abs(p_x) <= size[0]))
  valid = valid and ((size[1] <= 0.0) or (wp.abs(p_y) <= size[1]))

  return_id = wp.where(valid, geom_id, -1)
  return DistanceWithId(wp.where(valid, x, wp.inf), return_id)


@wp.func
def _ray_sphere(
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
  geom_id: int,
) -> DistanceWithId:
  """Returns the distance at which a ray intersects with a sphere."""
  a = wp.dot(vec, vec)
  b = wp.dot(vec, pnt)
  c = wp.dot(pnt, pnt) - size[0] * size[0]

  solutions = _ray_quad(a, b, c)
  x0 = solutions[0]
  x1 = solutions[1]
  x = wp.where(wp.isinf(x0), x1, x0)

  return_id = wp.where(wp.isinf(x), -1, geom_id)
  return DistanceWithId(x, return_id)


@wp.func
def _ray_capsule(
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
  geom_id: int,
) -> DistanceWithId:
  """Returns the distance at which a ray intersects with a capsule."""

  # cylinder round side: (x*lvec+lpnt)'*(x*lvec+lpnt) = size[0]*size[0]
  # For a capsule, we only care about the x,y components when checking cylinder intersection
  # since the z component is handled separately with the caps
  a = wp.dot(wp.vec2(vec[0], vec[1]), wp.vec2(vec[0], vec[1]))
  b = wp.dot(wp.vec2(pnt[0], pnt[1]), wp.vec2(vec[0], vec[1]))
  c = wp.dot(wp.vec2(pnt[0], pnt[1]), wp.vec2(pnt[0], pnt[1])) - size[0] * size[0]

  # solve a*x^2 + 2*b*x + c = 0
  solutions = _ray_quad(a, b, c)
  x0 = solutions[0]
  x1 = solutions[1]
  x = wp.where(wp.isinf(x0), x1, x0)

  # make sure round solution is between flat sides
  x = wp.where(wp.abs(pnt[2] + x * vec[2]) <= size[1], x, wp.inf)

  # top cap
  dif = pnt - wp.vec3(0.0, 0.0, size[1])
  solutions = _ray_quad(
    wp.dot(vec, vec), wp.dot(vec, dif), wp.dot(dif, dif) - size[0] * size[0]
  )
  x0 = solutions[0]
  x1 = solutions[1]
  # accept only top half of sphere
  x = wp.where((pnt[2] + x0 * vec[2] >= size[1]) and (x0 < x), x0, x)
  x = wp.where((pnt[2] + x1 * vec[2] >= size[1]) and (x1 < x), x1, x)

  # bottom cap
  dif = pnt + wp.vec3(0.0, 0.0, size[1])
  solutions = _ray_quad(
    wp.dot(vec, vec), wp.dot(vec, dif), wp.dot(dif, dif) - size[0] * size[0]
  )
  x0 = solutions[0]
  x1 = solutions[1]
  # accept only bottom half of sphere
  x = wp.where((pnt[2] + x0 * vec[2] <= -size[1]) and (x0 < x), x0, x)
  x = wp.where((pnt[2] + x1 * vec[2] <= -size[1]) and (x1 < x), x1, x)

  return_id = wp.where(wp.isinf(x), -1, geom_id)
  return DistanceWithId(x, return_id)


@wp.func
def _ray_ellipsoid(
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
  geom_id: int,
) -> DistanceWithId:
  """Returns the distance at which a ray intersects with an ellipsoid."""

  # invert size^2
  s = wp.vec3(
    1.0 / (size[0] * size[0]), 1.0 / (size[1] * size[1]), 1.0 / (size[2] * size[2])
  )

  # (x*lvec+lpnt)' * diag(1/size^2) * (x*lvec+lpnt) = 1
  svec = wp.vec3(s[0] * vec[0], s[1] * vec[1], s[2] * vec[2])
  a = wp.dot(svec, vec)
  b = wp.dot(svec, pnt)
  c = wp.dot(wp.vec3(s[0] * pnt[0], s[1] * pnt[1], s[2] * pnt[2]), pnt) - 1.0

  # solve a*x^2 + 2*b*x + c = 0
  solutions = _ray_quad(a, b, c)
  x0 = solutions[0]
  x1 = solutions[1]
  x = wp.where(wp.isinf(x0), x1, x0)

  return_id = wp.where(wp.isinf(x), -1, geom_id)
  return DistanceWithId(x, return_id)


@wp.func
def _ray_box(
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
  geom_id: int,
) -> DistanceWithId:
  """Returns the distance at which a ray intersects with a box."""

  # Initialize with infinity
  min_x = wp.inf

  # Check all 6 faces of the box
  # X faces (i=0)
  if vec[0] != 0.0:
    # +X face
    x_pos = (size[0] - pnt[0]) / vec[0]
    p1 = pnt[1] + x_pos * vec[1]
    p2 = pnt[2] + x_pos * vec[2]
    if x_pos >= 0.0 and wp.abs(p1) <= size[1] and wp.abs(p2) <= size[2]:
      min_x = wp.min(min_x, x_pos)

    # -X face
    x_neg = (-size[0] - pnt[0]) / vec[0]
    p1 = pnt[1] + x_neg * vec[1]
    p2 = pnt[2] + x_neg * vec[2]
    if x_neg >= 0.0 and wp.abs(p1) <= size[1] and wp.abs(p2) <= size[2]:
      min_x = wp.min(min_x, x_neg)

  # Y faces (i=1)
  if vec[1] != 0.0:
    # +Y face
    y_pos = (size[1] - pnt[1]) / vec[1]
    p0 = pnt[0] + y_pos * vec[0]
    p2 = pnt[2] + y_pos * vec[2]
    if y_pos >= 0.0 and wp.abs(p0) <= size[0] and wp.abs(p2) <= size[2]:
      min_x = wp.min(min_x, y_pos)

    # -Y face
    y_neg = (-size[1] - pnt[1]) / vec[1]
    p0 = pnt[0] + y_neg * vec[0]
    p2 = pnt[2] + y_neg * vec[2]
    if y_neg >= 0.0 and wp.abs(p0) <= size[0] and wp.abs(p2) <= size[2]:
      min_x = wp.min(min_x, y_neg)

  # Z faces (i=2)
  if vec[2] != 0.0:
    # +Z face
    z_pos = (size[2] - pnt[2]) / vec[2]
    p0 = pnt[0] + z_pos * vec[0]
    p1 = pnt[1] + z_pos * vec[1]
    if z_pos >= 0.0 and wp.abs(p0) <= size[0] and wp.abs(p1) <= size[1]:
      min_x = wp.min(min_x, z_pos)

    # -Z face
    z_neg = (-size[2] - pnt[2]) / vec[2]
    p0 = pnt[0] + z_neg * vec[0]
    p1 = pnt[1] + z_neg * vec[1]
    if z_neg >= 0.0 and wp.abs(p0) <= size[0] and wp.abs(p1) <= size[1]:
      min_x = wp.min(min_x, z_neg)

  return_id = wp.where(wp.isinf(min_x), -1, geom_id)
  return DistanceWithId(min_x, return_id)


@wp.struct
class Triangle:
  """A struct representing a triangle with 3 vertices."""

  v0: wp.vec3
  v1: wp.vec3
  v2: wp.vec3


@wp.struct
class Basis:
  """A struct representing a basis with 2 vectors."""

  b0: wp.vec3
  b1: wp.vec3

@wp.func
def _ray_triangle(
  triangle: Triangle,
  pnt: wp.vec3,
  vec: wp.vec3,
  basis: Basis,
) -> wp.float32:
  """Returns the distance at which a ray intersects with a triangle."""
  # dif = v[i] - lpnt
  dif0 = triangle.v0 - pnt
  dif1 = triangle.v1 - pnt
  dif2 = triangle.v2 - pnt

  # project difference vectors in normal plane
  planar_0_0 = wp.dot(dif0, basis.b0)
  planar_0_1 = wp.dot(dif0, basis.b1)
  planar_1_0 = wp.dot(dif1, basis.b0)
  planar_1_1 = wp.dot(dif1, basis.b1)
  planar_2_0 = wp.dot(dif2, basis.b0)
  planar_2_1 = wp.dot(dif2, basis.b1)

  # reject if on the same side of any coordinate axis
  if ((planar_0_0 > 0.0 and planar_1_0 > 0.0 and planar_2_0 > 0.0) or
      (planar_0_0 < 0.0 and planar_1_0 < 0.0 and planar_2_0 < 0.0) or
      (planar_0_1 > 0.0 and planar_1_1 > 0.0 and planar_2_1 > 0.0) or
      (planar_0_1 < 0.0 and planar_1_1 < 0.0 and planar_2_1 < 0.0)):
    return wp.float32(wp.inf)

  # determine if origin is inside planar projection of triangle
  # A = (p0-p2, p1-p2), b = -p2, solve A*t = b
  A00 = planar_0_0 - planar_2_0
  A10 = planar_1_0 - planar_2_0
  A01 = planar_0_1 - planar_2_1
  A11 = planar_1_1 - planar_2_1

  b0 = -planar_2_0
  b1 = -planar_2_1

  det = A00 * A11 - A10 * A01
  if wp.abs(det) < MJ_MINVAL:
    return wp.float32(wp.inf)

  t0 = (A11 * b0 - A10 * b1) / det
  t1 = (-A01 * b0 + A00 * b1) / det

  # check if outside
  if t0 < 0.0 or t1 < 0.0 or t0 + t1 > 1.0:
    return wp.float32(wp.inf)

  # intersect ray with plane of triangle
  dif0 = triangle.v0 - triangle.v2  # v0-v2
  dif1 = triangle.v1 - triangle.v2  # v1-v2
  dif2 = pnt - triangle.v2          # lpnt-v2
  nrm = wp.cross(dif0, dif1)        # normal to triangle plane
  denom = wp.dot(vec, nrm)
  if wp.abs(denom) < MJ_MINVAL:    
    return wp.float32(wp.inf)

  dist = -wp.dot(dif2, nrm) / denom
  return wp.where(dist >= 0.0, dist, wp.float32(wp.inf))


@wp.func
def _ray_mesh(
  m: Model,
  geom_id: int,
  unused_size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> DistanceWithId:
  """Returns the distance and geom_id for ray mesh intersections."""
  data_id = m.geom_dataid[geom_id]

  # Create basis vectors for the ray
  basis = Basis()

  # Compute orthogonal basis vectors
  if wp.abs(vec[0]) < wp.abs(vec[1]):
    if wp.abs(vec[0]) < wp.abs(vec[2]):
      basis.b0 = wp.vec3(0.0, vec[2], -vec[1])
    else:
      basis.b0 = wp.vec3(vec[1], -vec[0], 0.0)
  else:
    if wp.abs(vec[1]) < wp.abs(vec[2]):
      basis.b0 = wp.vec3(-vec[2], 0.0, vec[0])
    else:
      basis.b0 = wp.vec3(vec[1], -vec[0], 0.0)

  # Normalize first basis vector
  basis.b0 = wp.normalize(basis.b0)

  # Compute second basis vector as cross product
  basis.b1 = wp.cross(vec, basis.b0)
  basis.b1 = wp.normalize(basis.b1)

  min_dist = wp.float32(wp.inf)
  hit_found = int(0)

  # Get mesh vertex data range
  vert_start = m.mesh_vertadr[data_id]
  # vert_end = wp.where(
  #   data_id + 1 < m.mesh_vertadr.shape[0], m.mesh_vertadr[data_id + 1], m.nmeshvert
  # )


  # Get mesh face and vertex data
  face_start = m.mesh_faceadr[data_id]
  face_end = wp.where(
    data_id + 1 < m.mesh_faceadr.shape[0], m.mesh_faceadr[data_id + 1], m.nmeshface
  )

  # Iterate through all faces
  for i in range(face_start, face_end):
    # Get vertices for this face
    v0_idx = m.mesh_face[i, 0]
    v1_idx = m.mesh_face[i, 1]
    v2_idx = m.mesh_face[i, 2]

    # Create triangle struct
    triangle = Triangle()
    triangle.v0 = m.mesh_vert[vert_start + v0_idx]
    triangle.v1 = m.mesh_vert[vert_start + v1_idx]
    triangle.v2 = m.mesh_vert[vert_start + v2_idx]

    # Calculate intersection
    dist = _ray_triangle(triangle, pnt, vec, basis)
    if dist < min_dist:
      min_dist = dist
      hit_found = 1

  # Return the geom_id if we found a hit, otherwise -1
  return_id = wp.where(hit_found == 1, geom_id, -1)

  print(min_dist)
  
  return DistanceWithId(min_dist, return_id)


@wp.func
def _ray_geom(
  m: Model,
  d: Data,
  geom_id: int,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> DistanceWithId:
  type = m.geom_type[geom_id]
  size = m.geom_size[geom_id]

  # TODO(team): static loop unrolling to remove unnecessary branching
  if type == int(GeomType.PLANE.value):
    return _ray_plane(size, pnt, vec, geom_id)
  elif type == int(GeomType.SPHERE.value):
    return _ray_sphere(size, pnt, vec, geom_id)
  elif type == int(GeomType.CAPSULE.value):
    return _ray_capsule(size, pnt, vec, geom_id)
  elif type == int(GeomType.ELLIPSOID.value):
    return _ray_ellipsoid(size, pnt, vec, geom_id)
  elif type == int(GeomType.BOX.value):
    return _ray_box(size, pnt, vec, geom_id)
  elif type == int(GeomType.MESH.value):
    return _ray_mesh(m, geom_id, size, pnt, vec)
  return DistanceWithId(wp.inf, -1)


@wp.struct
class RayIntersection:
  dist: wp.float32
  geom_id: wp.int32


@wp.func
def _ray_all_geom(
  worldid: int,
  m: Model,
  d: Data,
  pnt: wp.vec3,
  vec: wp.vec3,
  num_threads: int,
  tid: int,
) -> RayIntersection:
  ngeom = m.ngeom

  min_val = wp.float32(wp.inf)
  min_idx = wp.int32(-1)

  upper = ((ngeom + num_threads - 1) // num_threads) * num_threads
  for geom_id in range(tid, upper, num_threads):
    if geom_id < ngeom:
      # Get ray in local coordinates
      pos = d.geom_xpos[worldid, geom_id]
      rot = d.geom_xmat[worldid, geom_id]
      local_pnt = wp.transpose(rot) @ (pnt - pos)
      local_vec = wp.transpose(rot) @ vec

      # Calculate intersection distance
      result = _ray_geom(m, d, geom_id, local_pnt, local_vec)
      cur_dist = result.dist
    else:
      cur_dist = wp.float32(wp.inf)

    t = wp.tile(cur_dist)
    local_min_idx = wp.tile_argmin(t)
    local_min_val = t[local_min_idx[0]]  # wp.tile_min(t)

    if local_min_val < min_val:
      min_val = local_min_val
      min_idx = local_min_idx[0]

  min_val = wp.where(min_val == wp.inf, wp.float32(-1.0), min_val)

  return RayIntersection(min_val, min_idx)


# One thread block/tile per ray query
@wp.kernel
def _ray_all_geom_kernel(
  m: Model,
  d: Data,
  pnt: wp.array(dtype=wp.vec3),
  vec: wp.array(dtype=wp.vec3),
  dist: wp.array(dtype=float, ndim=2),
  closest_hit_geom_id: wp.array(dtype=int, ndim=2),
  num_threads: int,
):
  worldid, rayid, tid = wp.tid()
  intersection = _ray_all_geom(
    worldid,
    m,
    d,
    pnt[rayid],
    vec[rayid],
    num_threads,
    tid,
  )

  # Write intersection results to output arrays
  dist[worldid, rayid] = intersection.dist
  closest_hit_geom_id[worldid, rayid] = intersection.geom_id


def ray_geom(
  m: Model,
  d: Data,
  pnt: wp.array(dtype=wp.vec3),
  vec: wp.array(dtype=wp.vec3),
) -> tuple[wp.array, wp.array]:
  """Returns the distance at which rays intersect with primitive geoms.

  Args:
      m: MuJoCo model
      d: MuJoCo data
      pnt: ray origin points
      vec: ray directions

  Returns:
      dist: distances from ray origins to geom surfaces
      geom_id: IDs of intersected geoms (-1 if none)
  """
  nrays = pnt.shape[0]
  dist = wp.zeros((d.nworld, nrays), dtype=float)
  closest_hit_geom_id = wp.zeros((d.nworld, nrays), dtype=int)
  num_threads = 64
  wp.launch_tiled(
    _ray_all_geom_kernel,
    dim=(d.nworld, nrays),
    inputs=[m, d, pnt, vec, dist, closest_hit_geom_id, num_threads],
    block_dim=num_threads,
  )
  return dist, closest_hit_geom_id
