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

from .types import MJ_MINVAL
from .types import Data
from .types import GeomType
from .types import Model
from .types import vec6


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
  # In:
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
  # In:
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
  # In:
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
  geom_id: int,
) -> DistanceWithId:
  """Returns the distance at which a ray intersects with a capsule."""

  # cylinder round side: (x*lvec+lpnt)'*(x*lvec+lpnt) = size[0]*size[0]
  # For a capsule, we only care about the x,y components when checking cylinder intersection
  # since the z component is handled separately with the caps
  vec_2d = wp.vec2(vec[0], vec[1])
  pnt_2d = wp.vec2(pnt[0], pnt[1])
  a = wp.dot(vec_2d, vec_2d)
  b = wp.dot(pnt_2d, vec_2d)
  c = wp.dot(pnt_2d, pnt_2d) - size[0] * size[0]

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
  x = wp.where((pnt[2] + x0 * vec[2] >= size[1]) and (x0 < x), x0, x)
  x = wp.where((pnt[2] + x1 * vec[2] >= size[1]) and (x1 < x), x1, x)

  # bottom cap
  dif = pnt + wp.vec3(0.0, 0.0, size[1])
  solutions = _ray_quad(wp.dot(vec, vec), wp.dot(vec, dif), wp.dot(dif, dif) - size[0] * size[0])
  x0 = solutions[0]
  x1 = solutions[1]
  # accept only bottom half of sphere
  x = wp.where((pnt[2] + x0 * vec[2] <= -size[1]) and (x0 < x), x0, x)
  x = wp.where((pnt[2] + x1 * vec[2] <= -size[1]) and (x1 < x), x1, x)

  return_id = wp.where(wp.isinf(x), -1, geom_id)
  return DistanceWithId(x, return_id)


@wp.func
def _ray_ellipsoid(
  # In:
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
  geom_id: int,
) -> DistanceWithId:
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

  return_id = wp.where(wp.isinf(x), -1, geom_id)
  return DistanceWithId(x, return_id)


@wp.func
def _ray_box(
  # In:
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
  geom_id: int,
) -> DistanceWithId:
  """Returns the distance at which a ray intersects with a box."""

  # Initialize with infinity
  min_x = wp.inf

  # Check all 6 faces of the box (2 faces per axis)
  for i in range(wp.static(3)):
    if vec[i] != 0.0:
      # Get indices for the other two dimensions
      j = (i + 1) % 3
      k = (i + 2) % 3

      for t in range(wp.static(2)):
        s = wp.where(t == 0, size[i], -size[i])
        sol = (s - pnt[i]) / vec[i]

        pj = pnt[j] + sol * vec[j]
        pk = pnt[k] + sol * vec[k]
        if sol >= 0.0 and wp.abs(pj) <= size[j] and wp.abs(pk) <= size[k]:
          min_x = wp.min(min_x, sol)

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
  # In:
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
  if (
    (planar_0_0 > 0.0 and planar_1_0 > 0.0 and planar_2_0 > 0.0)
    or (planar_0_0 < 0.0 and planar_1_0 < 0.0 and planar_2_0 < 0.0)
    or (planar_0_1 > 0.0 and planar_1_1 > 0.0 and planar_2_1 > 0.0)
    or (planar_0_1 < 0.0 and planar_1_1 < 0.0 and planar_2_1 < 0.0)
  ):
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
  dif2 = pnt - triangle.v2  # lpnt-v2
  nrm = wp.cross(dif0, dif1)  # normal to triangle plane
  denom = wp.dot(vec, nrm)
  if wp.abs(denom) < MJ_MINVAL:
    return wp.float32(wp.inf)

  dist = -wp.dot(dif2, nrm) / denom
  return wp.where(dist >= 0.0, dist, wp.float32(wp.inf))


@wp.func
def _ray_mesh(
  # Model:
  nmeshface: int,
  mesh_vertadr: wp.array(dtype=int),
  mesh_vertnum: wp.array(dtype=int),
  mesh_vert: wp.array(dtype=wp.vec3),
  mesh_faceadr: wp.array(dtype=int),
  mesh_face: wp.array(dtype=wp.vec3i),
  # In:
  geom_id: int,
  data_id: int,
  unused_size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> DistanceWithId:
  """Returns the distance and geom_id for ray mesh intersections."""

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
  vert_start = mesh_vertadr[data_id]

  # Get mesh face and vertex data
  face_start = mesh_faceadr[data_id]
  face_end = wp.where(data_id + 1 < mesh_faceadr.shape[0], mesh_faceadr[data_id + 1], nmeshface)

  # Iterate through all faces
  for i in range(face_start, face_end):
    # Get vertices for this face
    v_idx = mesh_face[i]

    # Create triangle struct
    triangle = Triangle()
    triangle.v0 = mesh_vert[vert_start + v_idx.x]
    triangle.v1 = mesh_vert[vert_start + v_idx.y]
    triangle.v2 = mesh_vert[vert_start + v_idx.z]

    # Calculate intersection
    dist = _ray_triangle(triangle, pnt, vec, basis)
    if dist < min_dist:
      min_dist = dist
      hit_found = 1

  # Return the geom_id if we found a hit, otherwise -1
  return_id = wp.where(hit_found == 1, geom_id, -1)

  return DistanceWithId(min_dist, return_id)


@wp.func
def _ray_map(
  # In:
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> any:
  """Maps ray to local geom frame coordinates.

  Args:
      pos: Position of geom frame
      mat: Orientation of geom frame
      pnt: Starting point of ray in world coordinates
      vec: Direction of ray in world coordinates

  Returns:
      Tuple of (local_pnt, local_vec) mapped to geom frame
  """
  # Calculate difference vector
  dif = pnt - pos

  # Transform point and vector to local coordinates using transpose(mat)
  local_pnt = wp.transpose(mat) @ dif
  local_vec = wp.transpose(mat) @ vec

  return local_pnt, local_vec


@wp.func
def _ray_geom(
  # In:
  pos: wp.vec3,  # Position of geom frame
  mat: wp.mat33,  # Orientation of geom frame
  size: wp.vec3,  # Size parameters of geom
  pnt: wp.vec3,  # Starting point of ray in world coordinates
  vec: wp.vec3,  # Direction of ray in world coordinates
  geomtype: int,  # Type of geometry
) -> float:
  """Returns distance along ray to intersection with geom, or infinity if no intersection.

  Matches MuJoCo's mju_rayGeom API for use with touch sensors.
  Maps inputs to local coordinates before intersection testing.
  """
  # Map ray to local coordinates
  local_pnt, local_vec = _ray_map(pos, mat, pnt, vec)

  # Create DistanceWithId struct to reuse existing ray intersection functions
  result = DistanceWithId(wp.inf, -1)

  # Call appropriate intersection function based on geom type
  if geomtype == int(GeomType.PLANE.value):
    result = _ray_plane(size, local_pnt, local_vec, 0)
  elif geomtype == int(GeomType.SPHERE.value):
    result = _ray_sphere(size, local_pnt, local_vec, 0)
  elif geomtype == int(GeomType.CAPSULE.value):
    result = _ray_capsule(size, local_pnt, local_vec, 0)
  elif geomtype == int(GeomType.ELLIPSOID.value):
    result = _ray_ellipsoid(size, local_pnt, local_vec, 0)
  elif geomtype == int(GeomType.BOX.value):
    result = _ray_box(size, local_pnt, local_vec, 0)

  return result.dist


@wp.func
def _ray_geom_with_mesh(
  # Model:
  nmeshface: int,
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  mesh_vertadr: wp.array(dtype=int),
  mesh_vertnum: wp.array(dtype=int),
  mesh_vert: wp.array(dtype=wp.vec3),
  mesh_faceadr: wp.array(dtype=int),
  mesh_face: wp.array(dtype=wp.vec3i),
  # In:
  geom_id: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  worldid: int,
) -> DistanceWithId:
  type = geom_type[geom_id]
  size = geom_size[worldid, geom_id]

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
    data_id = geom_dataid[geom_id]
    return _ray_mesh(
      nmeshface,
      mesh_vertadr,
      mesh_vertnum,
      mesh_vert,
      mesh_faceadr,
      mesh_face,
      geom_id,
      data_id,
      size,
      pnt,
      vec,
    )
  return DistanceWithId(wp.inf, -1)


@wp.struct
class RayIntersection:
  dist: wp.float32
  geom_id: wp.int32


snippet = """
#if defined(__CUDA_ARCH__)
    return blockDim.x;
#else    
    return 1;
#endif
    """


@wp.func_native(snippet)
def get_block_dim_x() -> int: ...


@wp.func
def _ray_all_geom(
  # Model:
  ngeom: int,
  nmeshface: int,
  body_weldid: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_group: wp.array(dtype=int),
  geom_matid: wp.array2d(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_rgba: wp.array2d(dtype=wp.vec4),
  mesh_vertadr: wp.array(dtype=int),
  mesh_vertnum: wp.array(dtype=int),
  mesh_vert: wp.array(dtype=wp.vec3),
  mesh_faceadr: wp.array(dtype=int),
  mesh_face: wp.array(dtype=wp.vec3i),
  mat_rgba: wp.array2d(dtype=wp.vec4),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  worldid: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  geomgroup: vec6,
  has_geomgroup: bool,
  flg_static: bool,
  bodyexclude: int,
  tid: int,
) -> RayIntersection:
  num_threads = get_block_dim_x()

  min_val = wp.float32(wp.inf)
  min_idx = wp.int32(-1)

  upper = ((ngeom + num_threads - 1) // num_threads) * num_threads
  for geom_id in range(tid, upper, num_threads):
    if geom_id < ngeom:
      # Apply all filters combined into a single boolean
      body_id = geom_bodyid[geom_id]

      # Start with True and apply each filter condition
      geom_filter = True
      # Body exclusion filter
      geom_filter = geom_filter and (body_id != bodyexclude)

      # Static geom filter
      geom_filter = geom_filter and (flg_static or body_weldid[body_id] != 0)

      # Geom group filter
      if has_geomgroup:
        group = geom_group[geom_id]
        # Clip group index to [0, 5] (mjNGROUP-1)
        group = wp.max(0, wp.min(5, group))
        geom_filter = geom_filter and (geomgroup[group] != 0)

      # RGBA filter
      matid = geom_matid[worldid, geom_id]
      geom_alpha = geom_rgba[worldid, geom_id][3]
      mat_alpha = wp.float32(0.0)
      if matid != -1:
        mat_alpha = mat_rgba[worldid, matid][3]

      # Geom is visible if either:
      # 1. No material and non-zero geom alpha, or
      # 2. Has material and non-zero material alpha
      geom_visible = (matid == -1 and geom_alpha != 0.0) or (matid != -1 and mat_alpha != 0.0)
      geom_filter = geom_filter and geom_visible

      if not geom_filter:
        cur_dist = wp.float32(wp.inf)
      else:
        # Get ray in local coordinates
        pos = geom_xpos_in[worldid, geom_id]
        rot = geom_xmat_in[worldid, geom_id]
        local_pnt = wp.transpose(rot) @ (pnt - pos)
        local_vec = wp.transpose(rot) @ vec

        # Calculate intersection distance
        result = _ray_geom_with_mesh(
          nmeshface,
          geom_type,
          geom_dataid,
          geom_size,
          mesh_vertadr,
          mesh_vertnum,
          mesh_vert,
          mesh_faceadr,
          mesh_face,
          geom_id,
          local_pnt,
          local_vec,
          worldid,
        )
        cur_dist = result.dist
    else:
      cur_dist = wp.float32(wp.inf)

    t = wp.tile(cur_dist)
    local_min_idx = wp.tile_argmin(t)
    local_min_val = t[local_min_idx[0]]

    id_tile = wp.tile(geom_id)

    if local_min_val < min_val:
      min_val = local_min_val
      min_idx = id_tile[local_min_idx[0]]

  min_val = wp.where(min_val == wp.inf, wp.float32(-1.0), min_val)

  return RayIntersection(min_val, min_idx)


# One thread block/tile per ray query
@wp.kernel
def _ray_all_geom_kernel(
  # Model:
  ngeom: int,
  nmeshface: int,
  body_weldid: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_group: wp.array(dtype=int),
  geom_matid: wp.array2d(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_rgba: wp.array2d(dtype=wp.vec4),
  mesh_vertadr: wp.array(dtype=int),
  mesh_vertnum: wp.array(dtype=int),
  mesh_vert: wp.array(dtype=wp.vec3),
  mesh_faceadr: wp.array(dtype=int),
  mesh_face: wp.array(dtype=wp.vec3i),
  mat_rgba: wp.array2d(dtype=wp.vec4),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  pnt: wp.array(dtype=wp.vec3),
  vec: wp.array(dtype=wp.vec3),
  geomgroup: vec6,
  has_geomgroup: bool,
  flg_static: bool,
  bodyexclude: int,
  # Out:
  dist_out: wp.array(dtype=float, ndim=2),
  closest_hit_geom_id_out: wp.array(dtype=int, ndim=2),
):
  worldid, rayid, tid = wp.tid()
  intersection = _ray_all_geom(
    ngeom,
    nmeshface,
    body_weldid,
    geom_type,
    geom_bodyid,
    geom_dataid,
    geom_group,
    geom_matid,
    geom_size,
    geom_rgba,
    mesh_vertadr,
    mesh_vertnum,
    mesh_vert,
    mesh_faceadr,
    mesh_face,
    mat_rgba,
    geom_xpos_in,
    geom_xmat_in,
    worldid,
    pnt[rayid],
    vec[rayid],
    geomgroup,
    has_geomgroup,
    flg_static,
    bodyexclude,
    tid,
  )

  # Write intersection results to output arrays
  dist_out[worldid, rayid] = intersection.dist
  closest_hit_geom_id_out[worldid, rayid] = intersection.geom_id


def ray(
  m: Model,
  d: Data,
  pnt: wp.array(dtype=wp.vec3),
  vec: wp.array(dtype=wp.vec3),
  geomgroup: vec6 = None,
  flg_static: bool = True,
  bodyexclude: int = -1,
) -> tuple[wp.array, wp.array]:
  """Returns the distance at which rays intersect with primitive geoms.

  Args:
      m: MuJoCo model
      d: MuJoCo data
      pnt: ray origin points
      vec: ray directions
      geomgroup: group inclusion/exclusion mask (6,), or all zeros to ignore
      flg_static: if True, allows rays to intersect with static geoms
      bodyexclude: ignore geoms on specified body id (-1 to disable)

  Returns:
      dist: distances from ray origins to geom surfaces
      geom_id: IDs of intersected geoms (-1 if none)
  """
  nrays = pnt.shape[0]
  dist = wp.zeros((d.nworld, nrays), dtype=float)
  closest_hit_geom_id = wp.zeros((d.nworld, nrays), dtype=int)
  num_threads = 64

  # Create default geomgroup if None is provided
  has_geomgroup = geomgroup is not None
  if geomgroup is None:
    geomgroup = vec6(0, 0, 0, 0, 0, 0)

  wp.launch_tiled(
    _ray_all_geom_kernel,
    dim=(d.nworld, nrays),
    inputs=[
      m.ngeom,
      m.nmeshface,
      m.body_weldid,
      m.geom_type,
      m.geom_bodyid,
      m.geom_dataid,
      m.geom_group,
      m.geom_matid,
      m.geom_size,
      m.geom_rgba,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_vert,
      m.mesh_faceadr,
      m.mesh_face,
      m.mat_rgba,
      d.geom_xpos,
      d.geom_xmat,
      pnt,
      vec,
      geomgroup,
      has_geomgroup,
      flg_static,
      bodyexclude,
      dist,
      closest_hit_geom_id,
    ],
    block_dim=num_threads,
  )
  return dist, closest_hit_geom_id
