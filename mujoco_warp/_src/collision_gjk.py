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

from .collision_convex import gjk_support_geom
from .collision_primitive import Geom
from .types import MJ_MINVAL

# TODO(team): improve compile time to enable backward pass
wp.config.enable_backward = False

FLOAT_MIN = -1e30
FLOAT_MAX = 1e30
MJ_MINVAL2 = MJ_MINVAL * MJ_MINVAL

mat43 = wp.types.matrix(shape=(4, 3), dtype=float)


@wp.struct
class GJKResult:
  dist: float
  x1: wp.vec3
  x2: wp.vec3
  dim: int
  simplex: mat43
  simplex1: mat43
  simplex2: mat43


@wp.func
def _linear_combine(n: int, coefs: wp.vec4, mat: mat43):
  v = wp.vec3(0.0)
  if n == 1:
    v = coefs[0] * mat[0]
  elif n == 2:
    v = coefs[0] * mat[0] + coefs[1] * mat[1]
  elif n == 3:
    v = coefs[0] * mat[0] + coefs[1] * mat[1] + coefs[2] * mat[2]
  else:
    v = coefs[0] * mat[0] + coefs[1] * mat[1] + coefs[2] * mat[2] + coefs[3] * mat[3]
  return v


@wp.func
def _almost_equal(v1: wp.vec3, v2: wp.vec3):
  return wp.abs(v1[0] - v2[0]) < MJ_MINVAL and wp.abs(v1[1] - v2[1]) < MJ_MINVAL and wp.abs(v1[2] - v2[2]) < MJ_MINVAL


@wp.func
def _subdistance(n: int, simplex: mat43):
  if n == 4:
    return _S3D(simplex[0], simplex[1], simplex[2], simplex[3])
  if n == 3:
    coordinates3 = _S2D(simplex[0], simplex[1], simplex[2])
    return wp.vec4(coordinates3[0], coordinates3[1], coordinates3[2], 0.0)
  if n == 2:
    coordinates2 = _S1D(simplex[0], simplex[1])
    return wp.vec4(coordinates2[0], coordinates2[1], 0.0, 0.0)
  return wp.vec4(1.0, 0.0, 0.0, 0.0)


@wp.func
def _det3(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3):
  return wp.dot(v1, wp.cross(v2, v3))


@wp.func
def _same_sign(a: float, b: float):
  if a > 0 and b > 0:
    return 1
  if a < 0 and b < 0:
    return -1
  return 0


@wp.func
def _project_origin_line(v1: wp.vec3, v2: wp.vec3):
  diff = v2 - v1
  scl = -(wp.dot(v2, diff) / wp.dot(diff, diff))
  return v2 + scl * diff


@wp.func
def _project_origin_plane(v1: wp.vec3, v2: wp.vec3, v3: wp.vec3):
  z = wp.vec3(0.0)
  diff21 = v2 - v1
  diff31 = v3 - v1
  diff32 = v3 - v2

  # n = (v1 - v2) x (v3 - v2)
  n = wp.cross(diff32, diff21)
  nv = wp.dot(n, v2)
  nn = wp.dot(n, n)
  if nn == 0:
    return z, 1
  if nv != 0 and nn > MJ_MINVAL:
    v = (nv / nn) * n
    return v, 0

  # n = (v2 - v1) x (v3 - v1)
  n = wp.cross(diff21, diff31)
  nv = wp.dot(n, v1)
  nn = wp.dot(n, n)
  if nn == 0:
    return z, 1
  if nv != 0 and nn > MJ_MINVAL:
    v = (nv / nn) * n
    return v, 0

  # n = (v1 - v3) x (v2 - v3)
  n = wp.cross(diff31, diff32)
  nv = wp.dot(n, v3)
  nn = wp.dot(n, n)
  v = (nv / nn) * n
  return v, 0


@wp.func
def _S3D(s1: wp.vec3, s2: wp.vec3, s3: wp.vec3, s4: wp.vec3):
  #  [[ s1_x, s2_x, s3_x, s4_x ],
  #   [ s1_y, s2_y, s3_y, s4_y ],
  #   [ s1_z, s2_z, s3_z, s4_z ],
  #   [ 1,    1,    1,    1    ]]
  # we want to solve M*lambda = P, where P = [p_x, p_y, p_z, 1] with [p_x, p_y, p_z] is the
  # origin projected onto the simplex

  # compute cofactors to find det(M)
  C41 = -_det3(s2, s3, s4)
  C42 = _det3(s1, s3, s4)
  C43 = -_det3(s1, s2, s4)
  C44 = _det3(s1, s2, s3)

  # NOTE: m_det = 6*SignVol(simplex) with C4i corresponding to the volume of the 3-simplex
  # with vertices {s1, s2, s3, 0} - si
  m_det = C41 + C42 + C43 + C44

  comp1 = _same_sign(m_det, C41)
  comp2 = _same_sign(m_det, C42)
  comp3 = _same_sign(m_det, C43)
  comp4 = _same_sign(m_det, C44)

  # if all signs are the same then the origin is inside the simplex
  if comp1 and comp2 and comp3 and comp4:
    return wp.vec4(C41 / m_det, C42 / m_det, C43 / m_det, C44 / m_det)

  # find the smallest distance, and use the corresponding barycentric coordinates
  coordinates = wp.vec4(0.0, 0.0, 0.0, 0.0)
  dmin = FLOAT_MAX

  if not comp1:
    subcoord = _S2D(s2, s3, s4)
    x = subcoord[0] * s2 + subcoord[1] * s3 + subcoord[2] * s4
    d = wp.dot(x, x)
    coordinates[0] = 0.0
    coordinates[1] = subcoord[0]
    coordinates[2] = subcoord[1]
    coordinates[3] = subcoord[2]
    dmin = d

  if not comp2:
    subcoord = _S2D(s1, s3, s4)
    x = subcoord[0] * s1 + subcoord[1] * s3 + subcoord[2] * s4
    d = wp.dot(x, x)
    if d < dmin:
      coordinates[0] = subcoord[0]
      coordinates[1] = 0.0
      coordinates[2] = subcoord[1]
      coordinates[3] = subcoord[2]
      dmin = d

  if not comp3:
    subcoord = _S2D(s1, s2, s4)
    x = subcoord[0] * s1 + subcoord[1] * s2 + subcoord[2] * s4
    d = wp.dot(x, x)
    if d < dmin:
      coordinates[0] = subcoord[0]
      coordinates[1] = subcoord[1]
      coordinates[2] = 0.0
      coordinates[3] = subcoord[2]
      dmin = d

  if not comp4:
    subcoord = _S2D(s1, s2, s3)
    x = subcoord[0] * s1 + subcoord[1] * s2 + subcoord[2] * s3
    d = wp.dot(x, x)
    if d < dmin:
      coordinates[0] = subcoord[0]
      coordinates[1] = subcoord[1]
      coordinates[2] = subcoord[2]
      coordinates[3] = 0.0
  return coordinates


@wp.func
def _S2D(s1: wp.vec3, s2: wp.vec3, s3: wp.vec3):
  # project origin onto affine hull of the simplex
  p_o, ret = _project_origin_plane(s1, s2, s3)
  if ret:
    v = _S1D(s1, s2)
    return wp.vec3(v[0], v[1], 0.0)

  # Below are the minors M_i4 of the matrix M given by
  # [[ s1_x, s2_x, s3_x, s4_x ],
  #  [ s1_y, s2_y, s3_y, s4_y ],
  #  [ s1_z, s2_z, s3_z, s4_z ],
  #  [ 1,    1,    1,    1    ]]
  M_14 = s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1]
  M_24 = s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2] - s1[2] * s2[0]
  M_34 = s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1] - s1[1] * s2[0]

  # exclude the axis with the largest projection of the simplex using the computed minors
  M_max = 0.0
  s1_2D = wp.vec2(0.0)
  s2_2D = wp.vec2(0.0)
  s3_2D = wp.vec2(0.0)
  p_o_2D = wp.vec2(0.0)

  mu1 = wp.abs(M_14)
  mu2 = wp.abs(M_24)
  mu3 = wp.abs(M_34)

  if mu1 >= mu2 and mu1 >= mu3:
    M_max = M_14
    s1_2D[0] = s1[1]
    s1_2D[1] = s1[2]

    s2_2D[0] = s2[1]
    s2_2D[1] = s2[2]

    s3_2D[0] = s3[1]
    s3_2D[1] = s3[2]

    p_o_2D[0] = p_o[1]
    p_o_2D[1] = p_o[2]
  elif mu2 >= mu3:
    M_max = M_24
    s1_2D[0] = s1[0]
    s1_2D[1] = s1[2]

    s2_2D[0] = s2[0]
    s2_2D[1] = s2[2]

    s3_2D[0] = s3[0]
    s3_2D[1] = s3[2]

    p_o_2D[0] = p_o[0]
    p_o_2D[1] = p_o[2]
  else:
    M_max = M_34
    s1_2D[0] = s1[0]
    s1_2D[1] = s1[1]

    s2_2D[0] = s2[0]
    s2_2D[1] = s2[1]

    s3_2D[0] = s3[0]
    s3_2D[1] = s3[1]

    p_o_2D[0] = p_o[0]
    p_o_2D[1] = p_o[1]

  # compute the cofactors C3i of the following matrix:
  # [[ s1_2D[0] - p_o_2D[0], s2_2D[0] - p_o_2D[0], s3_2D[0] - p_o_2D[0] ],
  #  [ s1_2D[1] - p_o_2D[1], s2_2D[1] - p_o_2D[1], s3_2D[1] - p_o_2D[1] ],
  #  [ 1,                    1,                    1                    ]]

  # C31 corresponds to the signed area of 2-simplex: (p_o_2D, s2_2D, s3_2D)
  C31 = (
    p_o_2D[0] * s2_2D[1]
    + p_o_2D[1] * s3_2D[0]
    + s2_2D[0] * s3_2D[1]
    - p_o_2D[0] * s3_2D[1]
    - p_o_2D[1] * s2_2D[0]
    - s3_2D[0] * s2_2D[1]
  )

  # C32 corresponds to the signed area of 2-simplex: (_po_2D, s1_2D, s3_2D)
  C32 = (
    p_o_2D[0] * s3_2D[1]
    + p_o_2D[1] * s1_2D[0]
    + s3_2D[0] * s1_2D[1]
    - p_o_2D[0] * s1_2D[1]
    - p_o_2D[1] * s3_2D[0]
    - s1_2D[0] * s3_2D[1]
  )

  # C33 corresponds to the signed area of 2-simplex: (p_o_2D, s1_2D, s2_2D)
  C33 = (
    p_o_2D[0] * s1_2D[1]
    + p_o_2D[1] * s2_2D[0]
    + s1_2D[0] * s2_2D[1]
    - p_o_2D[0] * s2_2D[1]
    - p_o_2D[1] * s1_2D[0]
    - s2_2D[0] * s1_2D[1]
  )

  comp1 = _same_sign(M_max, C31)
  comp2 = _same_sign(M_max, C32)
  comp3 = _same_sign(M_max, C33)

  # all the same sign, p_o is inside the 2-simplex
  if comp1 and comp2 and comp3:
    return wp.vec3(C31 / M_max, C32 / M_max, C33 / M_max)

  # find the smallest distance, and use the corresponding barycentric coordinates
  dmin = FLOAT_MAX
  coordinates = wp.vec3(0.0, 0.0, 0.0)

  if not comp1:
    subcoord = _S1D(s2, s3)
    x = subcoord[0] * s2 + subcoord[1] * s3
    d = wp.dot(x, x)
    coordinates[0] = 0.0
    coordinates[1] = subcoord[0]
    coordinates[2] = subcoord[1]
    dmin = d

  if not comp2:
    subcoord = _S1D(s1, s3)
    x = subcoord[0] * s1 + subcoord[1] * s2
    d = wp.dot(x, x)
    if d < dmin:
      coordinates[0] = subcoord[0]
      coordinates[1] = 0.0
      coordinates[2] = subcoord[1]
      dmin = d

  if not comp3:
    subcoord = _S1D(s1, s2)
    x = subcoord[0] * s1 + subcoord[1] * s2
    d = wp.dot(x, x)
    if d < dmin:
      coordinates[0] = subcoord[0]
      coordinates[1] = subcoord[1]
      coordinates[2] = 0.0
  return coordinates


@wp.func
def _S1D(s1: wp.vec3, s2: wp.vec3):
  # find projection of origin onto the 1-simplex:
  p_o = _project_origin_line(s1, s2)

  # find the axis with the largest projection "shadow" of the simplex
  mu_max = 0.0
  index = 0
  for i in range(3):
    mu = s1[i] - s2[i]
    if wp.abs(mu) >= wp.abs(mu_max):
      mu_max = mu
      index = i

  C1 = p_o[index] - s2[index]
  C2 = s1[index] - p_o[index]

  # inside the simplex
  if _same_sign(mu_max, C1) and _same_sign(mu_max, C2):
    return wp.vec2(C1 / mu_max, C2 / mu_max)
  return wp.vec2(0.0, 1.0)


@wp.func
def gjk(
  tolerance: float, gjk_iterations: int, geom1: Geom, geom2: Geom, x1_0: wp.vec3, x2_0: wp.vec3, geomtype1: int, geomtype2: int
):
  simplex = mat43()
  simplex1 = mat43()
  simplex2 = mat43()
  n = int(0)
  coordinates = wp.vec4()  # barycentric coordinates
  epsilon = 0.5 * tolerance * tolerance

  # set initial guess
  x_k = x1_0 - x2_0

  for k in range(gjk_iterations):
    xnorm = wp.norm_l2(x_k)
    if xnorm < MJ_MINVAL:
      break
    dir = -(x_k / xnorm)

    # compute the kth support point
    _, s1_k = gjk_support_geom(geom1, geomtype1, dir)
    _, s2_k = gjk_support_geom(geom2, geomtype2, -dir)
    simplex1[n] = s1_k
    simplex2[n] = s2_k
    simplex[n] = s1_k - s2_k

    # stopping criteria using the Frank-Wolfe duality gap given by
    #  |f(x_k) - f(x_min)|^2 <= < grad f(x_k), (x_k - s_k) >
    if wp.dot(x_k, x_k - simplex[n]) < epsilon:
      break

    # run the distance subalgorithm to compute the barycentric coordinates
    # of the closest point to the origin in the simplex
    coordinates = _subdistance(n + 1, simplex)

    # remove vertices from the simplex no longer needed
    n = int(0)
    for i in range(4):
      if coordinates[i] == 0:
        continue

      simplex[n] = simplex[i]
      simplex1[n] = simplex1[i]
      simplex2[n] = simplex2[i]
      coordinates[n] = coordinates[i]
      n += int(1)

    # SHOULD NOT OCCUR
    if n < 1:
      break

    # get the next iteration of x_k
    x_next = _linear_combine(n, coordinates, simplex)

    # x_k has converged to minimum
    if _almost_equal(x_next, x_k):
      break

    # copy next iteration into x_k
    x_k = x_next

    # we have a tetrahedron containing the origin so return early
    if n == 4:
      break

  result = GJKResult()

  # compute the approximate witness points
  result.x1 = _linear_combine(n, coordinates, simplex1)
  result.x2 = _linear_combine(n, coordinates, simplex2)
  result.dist = wp.norm_l2(x_k)

  result.dim = n
  result.simplex1 = simplex1
  result.simplex2 = simplex2
  result.simplex = simplex
  return result
