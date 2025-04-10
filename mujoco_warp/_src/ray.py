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
from .types import array2df
from .types import array3df
from .types import vec5
from .warp_util import event_scope
from .warp_util import kernel

from .types import array2df
from .types import array3df
from .types import vec5
from .warp_util import event_scope
from .warp_util import kernel



@wp.func
def _ray_quad(
    a: float, b: float, c: float
) -> wp.vec2:
  """Returns two solutions for quadratic: a*x^2 + 2*b*x + c = 0."""
  det = b * b - a * c
  det_2 = wp.sqrt(det)

  x0 = (-b - det_2) / a
  x1 = (-b + det_2) / a
  x0 = wp.where((det < MJ_MINVAL) | (x0 < 0.0), wp.inf, x0)
  x1 = wp.where((det < MJ_MINVAL) | (x1 < 0.0), wp.inf, x1)

  return wp.vec2(x0, x1)


@wp.func
def _ray_plane(
    size: wp.vec3,
    pnt: wp.vec3,
    vec: wp.vec3,
) -> float:
  """Returns the distance at which a ray intersects with a plane."""
  x = -pnt[2] / vec[2]

  valid = vec[2] <= -MJ_MINVAL  # z-vec pointing towards front face
  valid = valid and x >= 0.0
  # only within rendered rectangle
  p_x = pnt[0] + x * vec[0]
  p_y = pnt[1] + x * vec[1]
  valid = valid and ((size[0] <= 0.0) or (wp.abs(p_x) <= size[0]))
  valid = valid and ((size[1] <= 0.0) or (wp.abs(p_y) <= size[1]))

  return wp.where(valid, x, wp.inf)


@wp.func
def _ray_sphere(
    size: wp.vec3,
    pnt: wp.vec3,
    vec: wp.vec3,
) -> float:
  """Returns the distance at which a ray intersects with a sphere."""
  a = wp.dot(vec, vec)
  b = wp.dot(vec, pnt)
  c = wp.dot(pnt, pnt) - size[0] * size[0]
  
  solutions = _ray_quad(a, b, c)
  x0 = solutions[0]
  x1 = solutions[1]
  x = wp.where(wp.isinf(x0), x1, x0)

  return x


@wp.func
def _ray_capsule(
    size: wp.vec3,
    pnt: wp.vec3,
    vec: wp.vec3,
) -> float:
  """Returns the distance at which a ray intersects with a capsule."""

  # cylinder round side: (x*lvec+lpnt)'*(x*lvec+lpnt) = size[0]*size[0]
  a = wp.dot(vec[0:2], vec[0:2])
  b = wp.dot(vec[0:2], pnt[0:2])
  c = wp.dot(pnt[0:2], pnt[0:2]) - size[0] * size[0]

  # solve a*x^2 + 2*b*x + c = 0
  solutions = _ray_quad(a, b, c)
  x0 = solutions[0]
  x1 = solutions[1]
  x = wp.where(wp.isinf(x0), x1, x0)

  # make sure round solution is between flat sides
  x = wp.where(wp.abs(pnt[2] + x * vec[2]) <= size[1], x, wp.inf)

  # top cap
  dif = pnt - wp.vec3(0.0, 0.0, size[1])
  solutions = _ray_quad(wp.dot(vec, vec), wp.dot(vec, dif), wp.dot(dif, dif) - size[0] * size[0])
  x0 = solutions[0]
  x1 = solutions[1]
  # accept only top half of sphere
  x = wp.where((pnt[2] + x0 * vec[2] >= size[1]) & (x0 < x), x0, x)
  x = wp.where((pnt[2] + x1 * vec[2] >= size[1]) & (x1 < x), x1, x)

  # bottom cap
  dif = pnt + wp.vec3(0.0, 0.0, size[1])
  solutions = _ray_quad(wp.dot(vec, vec), wp.dot(vec, dif), wp.dot(dif, dif) - size[0] * size[0])
  x0 = solutions[0]
  x1 = solutions[1]
  # accept only bottom half of sphere
  x = wp.where((pnt[2] + x0 * vec[2] <= -size[1]) & (x0 < x), x0, x)
  x = wp.where((pnt[2] + x1 * vec[2] <= -size[1]) & (x1 < x), x1, x)

  return x

@wp.func
def _ray_ellipsoid(
    size: wp.vec3,
    pnt: wp.vec3,
    vec: wp.vec3,
) -> float:
  """Returns the distance at which a ray intersects with an ellipsoid."""

  # invert size^2
  s = wp.vec3(1.0 / (size[0] * size[0]), 1.0 / (size[1] * size[1]), 1.0 / (size[2] * size[2]))

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

  return x


@wp.func
def _ray_box(
    size: wp.vec3,
    pnt: wp.vec3,
    vec: wp.vec3,
) -> float:
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

  return min_x

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
  # project difference vectors in ray normal plane
  # Unrolled loops with scalar variables for planar
  planar_0_0 = wp.dot(triangle.v0 - pnt, basis.b0)
  planar_0_1 = wp.dot(triangle.v0 - pnt, basis.b1)
  planar_1_0 = wp.dot(triangle.v1 - pnt, basis.b0)
  planar_1_1 = wp.dot(triangle.v1 - pnt, basis.b1)
  planar_2_0 = wp.dot(triangle.v2 - pnt, basis.b0)
  planar_2_1 = wp.dot(triangle.v2 - pnt, basis.b1)

  # determine if origin is inside planar projection of triangle
  # A = (p0-p2, p1-p2), b = -p2, solve A*t = b
  A00 = planar_0_0 - planar_2_0
  A01 = planar_0_1 - planar_2_1
  A10 = planar_1_0 - planar_2_0
  A11 = planar_1_1 - planar_2_1
  
  b0 = -planar_2_0
  b1 = -planar_2_1
  
  det = A00 * A11 - A10 * A01

  t0 = (A11 * b0 - A10 * b1) / det
  t1 = (-A01 * b0 + A00 * b1) / det
  valid = (t0 >= 0.0) and (t1 >= 0.0) and (t0 + t1 <= 1.0)

  # intersect ray with plane of triangle
  nrm = wp.cross(triangle.v0 - triangle.v2, triangle.v1 - triangle.v2)
  dist = wp.dot(triangle.v2 - pnt, nrm) / wp.dot(vec, nrm)
  valid = valid and (dist >= 0.0)
  dist = wp.where(valid, dist, wp.float32(float('inf')))

  return dist


@wp.func
def _ray_mesh(
    m: Model,
    geom_id: int,
    unused_size: wp.vec3,
    pnt: wp.vec3,
    vec: wp.vec3,
) -> wp.float32:
  """Returns the distance for ray mesh intersections."""
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
  
  # Get mesh face and vertex data
  face_start = m.mesh_faceadr[data_id]
  face_end = wp.select(data_id + 1 < m.nmeshface, m.mesh_faceadr[data_id + 1], m.nmeshface)
  
  vert_start = m.mesh_vertadr[data_id]
  vert_end = wp.select(data_id + 1 < m.nmeshvert, m.mesh_vertadr[data_id + 1], m.nmeshvert)
  
  # Iterate through all faces
  # TODO(team): make parallel
  for i in range(face_start, face_end):
    # Get vertices for this face
    v0_idx = m.mesh_face[i * 3]
    v1_idx = m.mesh_face[i * 3 + 1]
    v2_idx = m.mesh_face[i * 3 + 2]
    
    # Create triangle struct
    triangle = Triangle()
    triangle.v0 = m.mesh_vert[vert_start + v0_idx]
    triangle.v1 = m.mesh_vert[vert_start + v1_idx]
    triangle.v2 = m.mesh_vert[vert_start + v2_idx]
    
    # Calculate intersection
    dist = _ray_triangle(triangle, pnt, vec, basis)
    min_dist = wp.min(min_dist, dist)
  
  return min_dist



@wp.func
def _ray_geom(
    m: Model,
    d: Data,
    geom_id: int, 
    pnt: wp.vec3,
    vec: wp.vec3,
) -> float:
  type = m.geom_type[geom_id]
  size = m.geom_size[geom_id]

  # TODO(team): static loop unrolling to remove unnecessary branching
  if type == int(GeomType.PLANE.value):
    return _ray_plane(size, pnt, vec)
  elif type == int(GeomType.SPHERE.value):
    return _ray_sphere(size, pnt, vec)
  elif type == int(GeomType.CAPSULE.value):
    return _ray_capsule(size, pnt, vec)
  elif type == int(GeomType.ELLIPSOID.value):
    return _ray_ellipsoid(size, pnt, vec)
  elif type == int(GeomType.BOX.value):
    return _ray_box(size, pnt, vec)
  elif type == int(GeomType.MESH.value):
    return _ray_mesh(size, pnt, vec)
  return wp.inf


# One thread block/tile per ray query
@wp.kernel
def _ray_geom_kernel(
    m: Model,
    d: Data,
    pnt: wp.vec3,
    vec: wp.vec3,
    dist: wp.array(dtype=float),
    closest_hit_geom_id: wp.array(dtype=int),
    num_threads: int
):

  ngeom = m.ngeom
  tid = wp.tid()
  
  min_val = wp.float32(wp.inf)
  min_idx = wp.int32(-1)

  for geom_id in range(tid, ngeom, num_threads):
    # Transform ray into geom frame
    pos = d.geom_xpos[geom_id]
    rot = d.geom_xmat[geom_id]

    # Get ray in local coordinates
    local_pnt = wp.transform_point(wp.inverse(rot), pnt - pos)
    local_vec = wp.transform_vector(wp.inverse(rot), vec)

    # Calculate intersection distance
    cur_dist = _ray_geom(m, d, geom_id, local_pnt, local_vec)

    t = wp.tile_load(cur_dist, shape=num_threads)
    local_min_val, local_min_idx = wp.tile_argmin(t)

    if local_min_val < min_val:
      min_val = local_min_val
      min_idx = local_min_idx

  dist[0] = min_val
  closest_hit_geom_id[0] = min_idx


# TODO(team): Add batching support to process multiple rays at once
def ray_geom(m: Model, d: Data, pnt: wp.vec3, vec: wp.vec3) -> tuple[float, int]:
  """Returns the distance at which a ray intersects with a primitive geom.

  Args:
    m: MuJoCo model
    d: MuJoCo data
    geom_id: ID of geom to test
    pnt: ray origin point
    vec: ray direction

  Returns:
    dist: distance from ray origin to geom surface
  """
  dist = wp.zeros(1, dtype=float)
  closest_hit_geom_id = wp.zeros(1, dtype=int)
  wp.launch_tiled(_ray_geom_kernel, dim=1, inputs=[m, d, pnt, vec, dist, closest_hit_geom_id])
  return dist[0], closest_hit_geom_id[0]
