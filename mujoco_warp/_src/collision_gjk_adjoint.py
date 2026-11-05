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
"""Feature recovery and differentiable replay for GJK/EPA collision adjoints."""

from typing import Tuple

import warp as wp

from mujoco_warp._src import collision_gjk
from mujoco_warp._src.collision_gjk import FLOAT_MAX
from mujoco_warp._src.collision_gjk import MINVAL
from mujoco_warp._src.collision_gjk import Polytope
from mujoco_warp._src.collision_gjk import _discrete_geoms
from mujoco_warp._src.collision_gjk import _epa
from mujoco_warp._src.collision_gjk import _inflate
from mujoco_warp._src.collision_gjk import _linear_combine
from mujoco_warp._src.collision_gjk import _polytope2
from mujoco_warp._src.collision_gjk import _polytope3
from mujoco_warp._src.collision_gjk import _polytope4
from mujoco_warp._src.collision_gjk import _project_origin_plane
from mujoco_warp._src.collision_gjk import _subdistance
from mujoco_warp._src.collision_gjk import _tri_affine_coord
from mujoco_warp._src.collision_gjk import gjk
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import mat43

wp.set_module_options({"enable_backward": True})


# Internal probe results.
_TAG_NONE = wp.constant(0)
_TAG_GJK_FEATURE = wp.constant(1)
_TAG_GJK_SIDE = wp.constant(2)
_TAG_EPA = wp.constant(3)


@wp.func
def _mat33_rows(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.mat33:
  return wp.mat33(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])


@wp.func
def _mat43_rows(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> mat43:
  return mat43(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], 0.0, 0.0, 0.0)


@wp.func
def _ccd_probe(
  # In:
  tolerance: float,
  gjk_iterations: int,
  epa_iterations: int,
  geom1: collision_gjk.Geom,
  geom2: collision_gjk.Geom,
  geomtype: int,
  x_1: wp.vec3,
  x_2: wp.vec3,
  vert: wp.array[wp.vec3],
  vert_index: wp.array[int],
  face: wp.array[int],
  face_pr: wp.array[wp.vec3],
  face_norm2: wp.array[float],
  horizon: wp.array[int],
  worldid: int,
  # Data out:
  overflow_out: wp.array[int],
):
  original_margin = geom2.margin
  original_size = geom2.size
  full_margin = float(0.0)
  radius = float(0.0)
  is_discrete = _discrete_geoms(GeomType.HFIELD.value, geomtype) and (geom1.margin == 0.0 and geom2.margin == 0.0)

  if geomtype == GeomType.SPHERE or geomtype == GeomType.CAPSULE:
    radius = geom2.size[0]
    full_margin = radius + 0.5 * geom2.margin
    geom2.margin = 0.0
    geom2.size = wp.vec3(0.0, geom2.size[1], geom2.size[2])

  if radius > 0.0:
    result = gjk(tolerance, gjk_iterations, geom1, geom2, x_1, x_2, GeomType.HFIELD.value, geomtype, full_margin, is_discrete)
    geom1.index = result.index1
    geom2.index = result.index2

    if result.dist > tolerance:
      if result.dist == FLOAT_MAX:
        return result.dist, 1, result.x1, result.x2, _TAG_NONE, result, -1
      is_side = bool(False)
      vertex = result.simplex_index1[0]
      for i in range(result.dim):
        if result.simplex_index1[i] != vertex:
          is_side = True
      dist, x1, x2 = _inflate(result, geom1, geom2, GeomType.HFIELD.value, geomtype, 0.0, full_margin)
      tag = wp.where(is_side, _TAG_GJK_SIDE, _TAG_GJK_FEATURE)
      return dist, 1, x1, x2, tag, result, -1

    geom2.margin = original_margin
    geom2.size = original_size

  result = gjk(tolerance, gjk_iterations, geom1, geom2, x_1, x_2, GeomType.HFIELD.value, geomtype, 0.0, is_discrete)
  geom1.index = result.index1
  geom2.index = result.index2

  if result.dist > tolerance or result.dim < 2 or result.separated:
    return result.dist, 1, result.x1, result.x2, _TAG_NONE, result, -1

  # Preserve the EPA face index needed by the replay.
  pt = Polytope()
  pt.nface = 0
  pt.nvert = 0
  pt.nhorizon = 0
  pt.vert = vert
  pt.vert_index = vert_index
  pt.face = face
  pt.face_pr = face_pr
  pt.face_norm2 = face_norm2
  pt.horizon = horizon

  if result.dim == 2:
    pt, new_result = _polytope2(
      pt,
      result.simplex,
      result.simplex1,
      result.simplex2,
      result.simplex_index1,
      result.simplex_index2,
      geom1,
      geom2,
      GeomType.HFIELD.value,
      geomtype,
    )
    if pt.status == -1:
      result.simplex = new_result.simplex
      result.simplex1 = new_result.simplex1
      result.simplex2 = new_result.simplex2
      result.simplex_index1 = new_result.simplex_index1
      result.simplex_index2 = new_result.simplex_index2
      result.dim = 3
  elif result.dim == 4:
    pt, new_result = _polytope4(
      pt,
      result.simplex,
      result.simplex1,
      result.simplex2,
      result.simplex_index1,
      result.simplex_index2,
    )
    if pt.status == -1:
      result.simplex = new_result.simplex
      result.simplex1 = new_result.simplex1
      result.simplex2 = new_result.simplex2
      result.simplex_index1 = new_result.simplex_index1
      result.simplex_index2 = new_result.simplex_index2
      result.dim = 3

  if result.dim == 3:
    pt = _polytope3(
      pt,
      result.dist,
      result.simplex,
      result.simplex1,
      result.simplex2,
      result.simplex_index1,
      result.simplex_index2,
      geom1,
      geom2,
      GeomType.HFIELD.value,
      geomtype,
    )

  # Boundary witnesses have no differentiable EPA face.
  if pt.status:
    return result.dist, 1, result.x1, result.x2, _TAG_NONE, result, -1

  is_discrete2 = _discrete_geoms(GeomType.HFIELD.value, geomtype) and (geom1.margin == 0.0 and geom2.margin == 0.0)
  dist, x1, x2, idx = _epa(
    tolerance,
    epa_iterations,
    pt,
    geom1,
    geom2,
    GeomType.HFIELD.value,
    geomtype,
    is_discrete2,
    False,
    worldid,
    overflow_out,
  )
  if idx == -1:
    return FLOAT_MAX, 0, wp.vec3(), wp.vec3(), _TAG_NONE, result, -1
  return dist, 1, x1, x2, _TAG_EPA, result, idx


@wp.func
def _ellipsoid_normal(point: wp.vec3, size: wp.vec3, margin: float) -> wp.vec3:
  """Recover the normal of an ellipsoid inflated by half the contact margin."""
  if margin == 0.0:
    return wp.cw_div(wp.cw_div(point, size), size)
  size_sq = wp.cw_mul(size, size)
  # The closest ellipsoid point has coordinates size^2 * point / (size^2 + shift).
  # For an inflated support point, shift = margin/2 * length(size * unit_normal).
  bounds = 0.5 * margin * size
  lower = wp.min(bounds[0], wp.min(bounds[1], bounds[2]))
  upper = wp.max(bounds[0], wp.max(bounds[1], bounds[2]))
  shift = 0.5 * (lower + upper)
  for _ in range(12):
    denom = size_sq + wp.vec3(shift)
    scaled = wp.cw_div(wp.cw_mul(size, point), denom)
    error = wp.dot(scaled, scaled) - 1.0
    if wp.abs(error) < 1.0e-6:
      break
    if error > 0.0:
      lower = shift
    else:
      upper = shift
    slope = -2.0 * wp.dot(scaled, wp.cw_div(scaled, denom))
    candidate = shift - error / slope
    shift = candidate if lower < candidate and candidate < upper else 0.5 * (lower + upper)
  return wp.cw_div(point, size_sq + wp.vec3(shift))


@wp.func
def _recover_direction(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  # In:
  geomtype: int,
  point: wp.vec3,
  vertex: int,
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  margin: float,
  vertadr: int,
  reference: wp.vec3,
) -> Tuple[bool, wp.vec3]:
  """Recovers a support direction consistent with a frozen support point."""
  direction = wp.vec3(0.0, 0.0, 0.0)
  if geomtype == GeomType.SPHERE:
    radius = size[0] + 0.5 * margin
    if radius < 1.0e-12:
      return False, direction
    return True, (point - pos) / radius

  if geomtype == GeomType.CAPSULE:
    radius = size[0] + 0.5 * margin
    if radius < 1.0e-12:
      return False, direction
    local = wp.transpose(mat) @ (point - pos)
    local -= wp.vec3(0.0, 0.0, wp.sign(local[2]) * size[1])
    return True, mat @ (local / radius)

  if geomtype == GeomType.ELLIPSOID or geomtype == GeomType.CYLINDER:
    local = wp.transpose(mat) @ (point - pos)
    if geomtype == GeomType.ELLIPSOID:
      scaled = _ellipsoid_normal(local, size, margin)
    else:
      xy = wp.sqrt(local[0] * local[0] + local[1] * local[1])
      if margin != 0.0:
        radial = float(0.0)
        if xy > MINVAL:
          radial = (xy - size[0]) / xy
        cap = wp.where(vertex == 0, size[1], -size[1])
        scaled = wp.vec3(local[0] * radial, local[1] * radial, local[2] - cap) / (0.5 * margin)
      else:
        # Without inflation, the point determines radial direction and cap, but not
        # their ratio. Use the converged EPA normal to recover that ratio.
        local_ref = wp.transpose(mat) @ reference
        radial = wp.sqrt(local_ref[0] * local_ref[0] + local_ref[1] * local_ref[1])
        scaled = wp.vec3(local[0] * radial, local[1] * radial, wp.sign(local[2]) * xy * wp.abs(local_ref[2]))
        if xy < MINVAL:
          scaled = wp.vec3(0.0, 0.0, wp.sign(local[2]))
    length = wp.norm_l2(scaled)
    if length < 1.0e-12:
      return False, direction
    return True, mat @ (scaled / length)

  if vertex < 0:
    return False, direction
  if margin == 0.0:
    return True, direction

  local = wp.vec3(0.0, 0.0, 0.0)
  if geomtype == GeomType.BOX:
    local = wp.cw_mul(
      wp.vec3(
        wp.where(vertex & 1, 1.0, -1.0),
        wp.where(vertex & 2, 1.0, -1.0),
        wp.where(vertex & 4, 1.0, -1.0),
      ),
      size,
    )
  else:
    local = mesh_vert[vertadr + vertex]
  return True, (point - (mat @ local + pos)) / (0.5 * margin)


@wp.func
def _support_frozen(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  # In:
  geomtype: int,
  vertex: int,
  direction: wp.vec3,
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  margin: float,
  vertadr: int,
) -> wp.vec3:
  if geomtype == GeomType.SPHERE:
    return pos + (size[0] + 0.5 * margin) * direction
  if geomtype == GeomType.CAPSULE:
    axis = wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
    sign = wp.sign(wp.dot(axis, direction))
    return pos + (size[0] + 0.5 * margin) * direction + sign * size[1] * axis
  if geomtype == GeomType.ELLIPSOID:
    local_e = wp.transpose(mat) @ direction
    tmp_e = wp.cw_mul(local_e, size)
    nrm_e = wp.norm_l2(tmp_e)
    res_e = wp.vec3(0.0, 0.0, 0.0)
    if nrm_e > 0.0:
      res_e = wp.cw_mul(tmp_e / nrm_e, size)
    return mat @ res_e + pos + 0.5 * margin * direction
  if geomtype == GeomType.CYLINDER:
    local_c = wp.transpose(mat) @ direction
    res_c = wp.vec3(0.0, 0.0, 0.0)
    dxy = wp.sqrt(local_c[0] * local_c[0] + local_c[1] * local_c[1])
    if dxy > MINVAL:
      scl = size[0] / dxy
      res_c = wp.vec3(local_c[0] * scl, local_c[1] * scl, 0.0)
    cap = wp.where(local_c[2] >= 0.0, size[1], -size[1])
    if vertex >= 0:
      cap = wp.where(vertex == 0, size[1], -size[1])
    res_c = wp.vec3(res_c[0], res_c[1], cap)
    return mat @ res_c + pos + 0.5 * margin * direction
  if geomtype == GeomType.BOX:
    vbox = wp.cw_mul(
      wp.vec3(
        wp.where(vertex & 1, 1.0, -1.0),
        wp.where(vertex & 2, 1.0, -1.0),
        wp.where(vertex & 4, 1.0, -1.0),
      ),
      size,
    )
    return mat @ vbox + pos + 0.5 * margin * direction
  vmesh = mesh_vert[vertadr + vertex]
  return mat @ vmesh + pos + 0.5 * margin * direction


@wp.func
def _tri_project_top(top: wp.mat33, point: wp.vec3) -> wp.vec3:
  """Projects a point onto the top triangle using the forward branch rules."""
  n = wp.vec3(0.0, 0.0, 1.0)
  a = top[0]
  b = top[1]
  c = top[2]
  coords = _tri_affine_coord(a, b, c, point)
  if coords[0] > 0.0 and coords[1] > 0.0 and coords[2] > 0.0:
    return coords[0] * a + coords[1] * b + coords[2] * c
  p = c
  p = wp.where(coords[1] > 0.0, b, p)
  p = wp.where(coords[0] > 0.0, a, p)
  return point - wp.dot(point - p, n) * n


@wp.func
def _replay_frozen(
  # In:
  witness: wp.mat33,
  negative: bool,
  pos1: wp.vec3,
  mat1: wp.mat33,
  pos2: wp.vec3,
  mat2: wp.mat33,
) -> Tuple[float, wp.vec3, wp.vec3]:
  x1 = mat1 @ witness[0] + pos1
  x2 = mat2 @ witness[1] + pos2
  dvec = x1 - x2
  s = wp.where(negative, -1.0, 1.0)
  dist = s * wp.sqrt(wp.dot(dvec, dvec))
  return dist, x1, x2


@wp.func
def _endpoint_sign(
  # In:
  signs: int,
  index: int,
) -> float:
  code = (signs >> (2 * index)) & 3
  return wp.where(code == 1, 1.0, wp.where(code == 2, -1.0, 0.0))


@wp.func
def _replay_gjk_feature(
  # In:
  dim: int,
  signs: int,
  prism: wp.mat33,
  pos: wp.vec3,
  axis: wp.vec3,
  radius: float,
  half_length: float,
  margin: float,
) -> Tuple[int, float, wp.vec3, wp.vec3, wp.vec3]:
  zero = wp.vec3(0.0, 0.0, 0.0)
  geom_a = pos + _endpoint_sign(signs, 0) * half_length * axis
  geom_b = pos + _endpoint_sign(signs, 1) * half_length * axis
  geom_c = pos + _endpoint_sign(signs, 2) * half_length * axis
  simplex = _mat43_rows(prism[0] - geom_a, prism[1] - geom_b, prism[2] - geom_c)
  weights = _subdistance(dim, simplex)
  delta = _linear_combine(dim, weights, simplex)
  point1 = _linear_combine(dim, weights, _mat43_rows(prism[0], prism[1], prism[2]))
  point2 = point1 - delta
  distance = wp.sqrt(wp.dot(delta, delta))
  if distance < 1.0e-12:
    return 0, 0.0, zero, zero, zero
  full_margin = radius + 0.5 * margin
  point2 = point2 - full_margin * (-delta / distance)
  dist = distance - full_margin
  # The inflated witnesses coincide at contact onset. Their subtraction loses
  # the normal in float32 and introduces a spurious 1 / abs(dist) in its VJP.
  # The uninflated separation has length approximately radius and the same direction.
  normal = wp.where(dist <= 0.0, -delta, delta)
  return 1, dist, point1, point2, normal


@wp.func
def _replay_gjk_side(
  # In:
  top: wp.mat33,
  direction: wp.vec3,
  pos: wp.vec3,
  axis: wp.vec3,
  radius: float,
  half_length: float,
  margin: float,
) -> Tuple[int, float, wp.vec3, wp.vec3]:
  n = wp.vec3(0.0, 0.0, 1.0)
  sign = wp.sign(wp.dot(axis, direction))
  x2 = pos + sign * half_length * axis - (radius + 0.5 * margin) * n
  x1 = _tri_project_top(top, x2)
  dvec = x1 - x2
  return 1, -wp.sqrt(wp.dot(dvec, dvec)), x1, x2


@wp.func
def _replay_epa(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  # In:
  hfield_correction: bool,
  geomtype: int,
  features: wp.vec4i,
  prism: wp.mat33,
  directions: wp.mat33,
  reference: wp.vec3,
  top: wp.mat33,
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  margin: float,
  vertadr: int,
) -> Tuple[int, float, wp.vec3, wp.vec3]:
  zero = wp.vec3(0.0, 0.0, 0.0)
  geom_a = _support_frozen(mesh_vert, geomtype, features[0], directions[0], pos, mat, size, margin, vertadr)
  geom_b = _support_frozen(mesh_vert, geomtype, features[1], directions[1], pos, mat, size, margin, vertadr)
  geom_c = _support_frozen(mesh_vert, geomtype, features[2], directions[2], pos, mat, size, margin, vertadr)
  delta_a = prism[0] - geom_a
  delta_b = prism[1] - geom_b
  delta_c = prism[2] - geom_c
  projection, degenerate = _project_origin_plane(delta_c, delta_b, delta_a)
  if degenerate:
    return 0, 0.0, zero, zero
  if wp.dot(projection, reference) < 0.0:
    projection = -projection
  dist = -wp.sqrt(wp.dot(projection, projection))
  weights = _tri_affine_coord(delta_a, delta_b, delta_c, projection)
  x1 = weights[0] * prism[0] + weights[1] * prism[1] + weights[2] * prism[2]
  x2 = weights[0] * geom_a + weights[1] * geom_b + weights[2] * geom_c

  if hfield_correction:
    n = wp.vec3(0.0, 0.0, 1.0)
    x2c = wp.vec3(0.0, 0.0, 0.0)
    if geomtype == GeomType.SPHERE or geomtype == GeomType.CAPSULE:
      axis = wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
      half_length = wp.where(geomtype == GeomType.CAPSULE, size[1], 0.0)
      sign = wp.sign(wp.dot(axis, x2))
      x2c = pos + sign * half_length * axis - (0.5 * margin + size[0]) * n
    else:
      x2n = wp.norm_l2(x2)
      if x2n < 1.0e-12:
        return 0, 0.0, zero, zero
      x2c = _support_frozen(mesh_vert, geomtype, features[3], x2 / x2n, pos, mat, size, margin, vertadr)
    x1 = _tri_project_top(top, x2c)
    dvec = x1 - x2c
    return 1, -wp.sqrt(wp.dot(dvec, dvec)), x1, x2c

  return 1, dist, x1, x2
