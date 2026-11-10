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
from mujoco_warp._src.collision_gjk import MINVAL
from mujoco_warp._src.collision_gjk import _linear_combine
from mujoco_warp._src.collision_gjk import _project_origin_plane
from mujoco_warp._src.collision_gjk import _tri_affine_coord
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import mat43

wp.set_module_options({"enable_backward": True})


# Internal probe results.
_TAG_NONE = wp.constant(0)
_TAG_GJK_FEATURE = wp.constant(1)
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
  cutoff: float,
  gjk_iterations: int,
  epa_iterations: int,
  geom1: collision_gjk.Geom,
  geom2: collision_gjk.Geom,
  geomtype1: int,
  geomtype2: int,
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
  """Recover the forward simplex or EPA face without recording the iterative search."""
  index1, index2 = geom1.index, geom2.index
  needs_epa, dist, ncontact, x1, x2, result, geom1, geom2 = collision_gjk.gjk_phase(
    tolerance, cutoff, gjk_iterations, geom1, geom2, geomtype1, geomtype2, x_1, x_2
  )
  if not needs_epa:
    # The forward helper only retains separated simplexes for sphere/capsule inflation.
    if result.dim == 0 and 0.0 < dist and dist < collision_gjk.FLOAT_MAX:
      geom1.index = index1
      geom2.index = index2
      result = collision_gjk.gjk(
        tolerance,
        gjk_iterations,
        geom1,
        geom2,
        x_1,
        x_2,
        geomtype1,
        geomtype2,
        cutoff,
        collision_gjk._discrete_geoms(geomtype1, geomtype2) and geom1.margin == 0.0 and geom2.margin == 0.0,
      )
    tag = wp.where(result.dim > 0, _TAG_GJK_FEATURE, _TAG_NONE)
    return dist, ncontact, x1, x2, tag, result, -1
  dist, ncontact, x1, x2, idx = collision_gjk.epa_phase(
    tolerance,
    epa_iterations,
    result,
    geom1,
    geom2,
    geomtype1,
    geomtype2,
    vert,
    vert_index,
    face,
    face_pr,
    face_norm2,
    horizon,
    False,
    worldid,
    overflow_out,
  )
  tag = wp.where(idx >= 0, _TAG_EPA, _TAG_NONE)
  return dist, ncontact, x1, x2, tag, result, idx


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
    axis = wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
    sign = wp.sign(wp.dot(point - pos, axis))
    return True, (point - pos - sign * size[1] * axis) / radius

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
    if vertex >= 0:
      # Lateral support can select the center instead of either capsule endpoint.
      sign = wp.where(vertex == 0, 1.0, wp.where(vertex == 1, -1.0, 0.0))
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
def _feature_weights(dim: int, simplex: mat43) -> wp.vec4:
  """Compute closest-point weights on the retained simplex without selecting a new feature."""
  if dim == 1:
    return wp.vec4(1.0, 0.0, 0.0, 0.0)
  edge = simplex[1] - simplex[0]
  if dim == 2:
    denom = wp.dot(edge, edge)
    if denom == 0.0:
      return wp.vec4(1.0, 0.0, 0.0, 0.0)
    weight = -wp.dot(simplex[0], edge) / denom
    return wp.vec4(1.0 - weight, weight, 0.0, 0.0)
  normal = wp.cross(edge, simplex[2] - simplex[0])
  a = wp.dot(normal, wp.cross(simplex[1], simplex[2]))
  b = wp.dot(normal, wp.cross(simplex[2], simplex[0]))
  c = wp.dot(normal, wp.cross(simplex[0], simplex[1]))
  denom = a + b + c
  if denom == 0.0:
    return wp.vec4(1.0, 0.0, 0.0, 0.0)
  return wp.vec4(a / denom, b / denom, c / denom, 0.0)


@wp.func
def _replay_gjk_feature(
  # In:
  dim: int,
  signs: int,
  points1: wp.mat33,
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
  simplex = _mat43_rows(points1[0] - geom_a, points1[1] - geom_b, points1[2] - geom_c)
  weights = _feature_weights(dim, simplex)
  delta = _linear_combine(dim, weights, simplex)
  point1 = _linear_combine(dim, weights, _mat43_rows(points1[0], points1[1], points1[2]))
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
  normal = -delta
  return 1, dist, point1, point2, normal


@wp.func
def _replay_gjk(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  # In:
  dim: int,
  geomtype: int,
  features: wp.vec4i,
  points1: wp.mat33,
  directions: wp.mat33,
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  margin: float,
  vertadr: int,
) -> Tuple[int, float, wp.vec3, wp.vec3, wp.vec3]:
  """Replay a separated GJK simplex with its support features fixed."""
  geom_a = _support_frozen(mesh_vert, geomtype, features[0], directions[0], pos, mat, size, margin, vertadr)
  geom_b = _support_frozen(mesh_vert, geomtype, features[1], directions[1], pos, mat, size, margin, vertadr)
  geom_c = _support_frozen(mesh_vert, geomtype, features[2], directions[2], pos, mat, size, margin, vertadr)
  simplex = _mat43_rows(points1[0] - geom_a, points1[1] - geom_b, points1[2] - geom_c)
  weights = _feature_weights(dim, simplex)
  delta = _linear_combine(dim, weights, simplex)
  distance = wp.length(delta)
  if distance < 1.0e-12:
    return 0, 0.0, wp.vec3(), wp.vec3(), wp.vec3()
  x1 = _linear_combine(dim, weights, _mat43_rows(points1[0], points1[1], points1[2]))
  x2 = _linear_combine(dim, weights, _mat43_rows(geom_a, geom_b, geom_c))
  return 1, distance, x1, x2, -delta


@wp.func
def _replay_epa(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  # In:
  geomtype: int,
  features: wp.vec4i,
  points1: wp.mat33,
  directions: wp.mat33,
  reference: wp.vec3,
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  margin: float,
  vertadr: int,
) -> Tuple[int, float, wp.vec3, wp.vec3]:
  """Replay one EPA face in the first shape's frame, with its vertices fixed."""
  zero = wp.vec3(0.0, 0.0, 0.0)
  geom_a = _support_frozen(mesh_vert, geomtype, features[0], directions[0], pos, mat, size, margin, vertadr)
  geom_b = _support_frozen(mesh_vert, geomtype, features[1], directions[1], pos, mat, size, margin, vertadr)
  geom_c = _support_frozen(mesh_vert, geomtype, features[2], directions[2], pos, mat, size, margin, vertadr)
  delta_a = points1[0] - geom_a
  delta_b = points1[1] - geom_b
  delta_c = points1[2] - geom_c
  projection, degenerate = _project_origin_plane(delta_c, delta_b, delta_a)
  if degenerate:
    return 0, 0.0, zero, zero
  if wp.dot(projection, reference) < 0.0:
    projection = -projection
  dist = -wp.sqrt(wp.dot(projection, projection))
  weights = _tri_affine_coord(delta_a, delta_b, delta_c, projection)
  x1 = weights[0] * points1[0] + weights[1] * points1[1] + weights[2] * points1[2]
  x2 = weights[0] * geom_a + weights[1] * geom_b + weights[2] * geom_c
  return 1, dist, x1, x2
