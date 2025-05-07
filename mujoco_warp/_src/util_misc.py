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

"""Miscellaneous utilities."""

from typing import Tuple

import warp as wp

from . import math
from .types import MJ_MINVAL
from .types import WrapType


@wp.func
def is_intersect(p1: wp.vec2, p2: wp.vec2, p3: wp.vec2, p4: wp.vec2) -> bool:
  # compute determinant, check
  det = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])

  if wp.abs(det) < MJ_MINVAL:
    return False

  # compute intersection point on each line
  a = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / det
  b = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / det

  if a >= 0 and a <= 1.0 and b >= 0.0 and b <= 1.0:
    return True
  else:
    return False


@wp.func
def length_circle(p0: wp.vec2, p1: wp.vec2, ind: int, radius: float) -> float:
  # compute angle between 0 and pi
  p0n, _ = math.normalize_with_norm(p0)
  p1n, _ = math.normalize_with_norm(p1)

  angle = wp.acos(wp.dot(p0n, p1n))

  # flip if necessary
  cross = p0[1] * p1[0] - p0[0] * p1[1]
  if (cross > 0.0 and ind != 0) or (cross < 0.0 and ind == 0):
    angle = 2.0 * wp.pi - angle

  return radius * angle


@wp.func
def wrap_circle(end: wp.vec4, side: wp.vec2, radius: float):
  # TODO(team): return type
  valid_side = wp.norm_l2(side) < wp.inf

  end0 = wp.vec2(end[0], end[1])
  end1 = wp.vec2(end[2], end[3])

  sqlen0 = wp.dot(end0, end0)
  sqlen1 = wp.dot(end1, end1)
  sqrad = radius * radius

  # either point inside circle or circle too small: no wrap
  if (sqlen0 < sqrad) or (sqlen1 < sqrad) or (radius < MJ_MINVAL):
    return -1.0, wp.vec2(wp.inf), wp.vec2(wp.inf)

  # points too close: no wrap
  dif = end1 - end0
  dd = wp.dot(dif, dif)
  if dd < MJ_MINVAL:
    return -1.0, wp.vec2(wp.inf), wp.vec2(wp.inf)

  # find nearest point on line segment to origin: a * dif + d0
  a = -wp.dot(dif, end0) / dd
  a = wp.clamp(a, 0.0, 1.0)

  # check for intersection and side
  tmp = a * dif + end0
  if (wp.dot(tmp, tmp) > sqrad) and (not valid_side or wp.dot(side, tmp) >= 0.0):
    return -1.0, wp.vec2(wp.inf), wp.vec2(wp.inf)

  sqrt0 = wp.sqrt(sqlen0 - sqrad)
  sqrt1 = wp.sqrt(sqlen1 - sqrad)

  # construct the two solutions, compute goodness
  sol00 = wp.vec2(
    (end[0] * sqrad + radius * end[1] * sqrt0) / sqlen0,
    (end[1] * sqrad - radius * end[0] * sqrt0) / sqlen0,
  )
  sol01 = wp.vec2(
    (end[2] * sqrad - radius * end[3] * sqrt1) / sqlen1,
    (end[3] * sqrad + radius * end[2] * sqrt1) / sqlen1,
  )

  sol10 = wp.vec2(
    (end[0] * sqrad - radius * end[1] * sqrt0) / sqlen0,
    (end[1] * sqrad + radius * end[0] * sqrt0) / sqlen0,
  )
  sol11 = wp.vec2(
    (end[2] * sqrad + radius * end[3] * sqrt1) / sqlen1,
    (end[3] * sqrad - radius * end[2] * sqrt1) / sqlen1,
  )

  # goodness: close to sd, or shorter path
  if valid_side:
    tmp0, _ = math.normalize_with_norm(sol00 + sol01)
    good0 = wp.dot(tmp0, side)
    tmp1, _ = math.normalize_with_norm(sol10 + sol11)
    good1 = wp.dot(tmp1, side)
  else:
    tmp0 = sol00 - sol01
    good0 = -wp.dot(tmp0, tmp0)
    tmp1 = sol10 - sol11
    good1 = -wp.dot(tmp1, tmp1)

  # penalize for intersection
  if is_intersect(end0, sol00, end1, sol01):
    good0 = -10000.0
  if is_intersect(end0, sol10, end1, sol11):
    good1 = -10000.0

  # select the better solution
  if good0 > good1:
    pnt0 = sol00
    pnt1 = sol01
    ind = 0
  else:
    pnt0 = sol10
    pnt1 = sol11
    ind = 1

  # check for intersection
  if is_intersect(end0, pnt0, end1, pnt1):
    return -1.0, wp.vec2(wp.inf), wp.vec2(wp.inf)

  # return curve length
  return length_circle(pnt0, pnt1, ind, radius), pnt0, pnt1


@wp.func
def wrap_inside(
  end: wp.vec4,
  radius: float,
  maxiter: int = 20,
  zinit: float = 1.0 - 1.0e-7,
  tolerance: float = 1.0e-6,
):
  # TODO(team): return type
  end0 = wp.vec2(end[0], end[1])
  end1 = wp.vec2(end[2], end[3])

  # constants
  len0 = wp.norm_l2(end0)
  len1 = wp.norm_l2(end1)
  dif = end1 - end0
  dd = wp.dot(dif, dif)

  # either point inside circle or circle too small: no wrap
  if (
    (len0 <= radius)
    or (len1 <= radius)
    or (radius < MJ_MINVAL)
    or (len0 < MJ_MINVAL)
    or (len1 < MJ_MINVAL)
  ):
    return -1.0, wp.vec2(wp.inf), wp.vec2(wp.inf)

  # segment-circle intersection: no wrap
  if dd > MJ_MINVAL:
    # find nearest point on line segment to origin: d0 + a * dif
    a = -wp.dot(dif, end0) / dd

    # in segment
    if (a > 0.0) and (a < 1.0):
      tmp = end0 + a * dif
      if wp.norm_l2(tmp) <= radius:
        return -1.0, wp.vec2(wp.inf), wp.vec2(wp.inf)

  # prepare default in case of numerical failure: average
  pnt = 0.5 * (end0 + end1)
  pnt, _ = math.normalize_with_norm(pnt)
  pnt *= radius

  # compute function parameters: asin(A * z) + asin(B * z) - 2 * asin(z) + G = 0
  A = radius / len0
  B = radius / len1
  sq_A = A * A
  sq_B = B * B
  cosG = (len0 * len0 + len1 * len1 - dd) / (2.0 * len0 * len1)
  if cosG < -1.0 + MJ_MINVAL:
    return -1.0, pnt, pnt
  elif cosG > 1.0 - MJ_MINVAL:
    return 0.0, pnt, pnt
  G = wp.acos(cosG)

  # init
  z = zinit
  f = wp.asin(A * z) + wp.asin(B * z) - 2.0 * wp.asin(z) + G

  # make sure init is not on the other side
  if f > 0.0:
    return 0.0, pnt, pnt

  # Newton method
  iter = int(0)

  while (iter < maxiter) and (wp.abs(f) > tolerance):
    # derivative
    sq_z = z * z
    df = (
      A / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_A))
      + B / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z * sq_B))
      - 2.0 / wp.max(MJ_MINVAL, wp.sqrt(1.0 - sq_z))
    )

    # check sign; SHOULD NOT OCCUR
    if df > -MJ_MINVAL:
      return 0.0, pnt, pnt

    # new point
    z1 = z - f / df

    # make sure we are moving to the left; SHOULD NOT OCCUR
    if z1 > z:
      return 0.0, pnt, pnt

    # update solution
    z = z1
    f = wp.asin(A * z) + wp.asin(B * z) - 2.0 * wp.asin(z) + G

    # exit if positive: SHOULD NOT OCCUR
    if f > tolerance:
      return 0.0, pnt, pnt

    iter += 1

  # check convergence
  if iter >= maxiter:
    return 0.0, pnt, pnt

  # finalize: rotation by ang from vec = a or b, depending on cross(a, b) sign
  if end[0] * end[3] - end[1] * end[2] > 0.0:
    vec = end0
    ang = wp.asin(z) - wp.asin(A * z)
  else:
    vec = end1
    ang = wp.asin(z) - wp.asin(B * z)

  vec, _ = math.normalize_with_norm(vec)
  pnt = wp.vec2(
    radius * (wp.cos(ang) * vec[0] - wp.sin(ang) * vec[1]),
    radius * (wp.sin(ang) * vec[0] + wp.cos(ang) * vec[1]),
  )

  return 0.0, pnt, pnt


@wp.func
def wrap(
  x0: wp.vec3,
  x1: wp.vec3,
  xpos: wp.vec3,
  xmat: wp.mat33,
  radius: float,
  geomtype: int,
  side: wp.vec3,
) -> Tuple[float, wp.vec3, wp.vec3]:
  # TODO(team): check object type; SHOULD NOT OCCUR

  # map sites to wrap object's local frame
  xmatT = wp.transpose(xmat)
  p0 = xmatT @ (x0 - xpos)
  p1 = xmatT @ (x1 - xpos)

  # too close to origin: return
  if (wp.norm_l2(p0) < MJ_MINVAL) or (wp.norm_l2(p1) < MJ_MINVAL):
    return -1.0, wp.vec3(wp.inf), wp.vec3(wp.inf)

  # construct 2D frame for circle wrap
  if geomtype == int(WrapType.SPHERE.value):
    # 1st axis = p0
    axis0, _ = math.normalize_with_norm(p0)

    # normal to p0-0-p1 plane = cross(p0, p1)
    normal = wp.cross(p0, p1)
    normal, nrm = math.normalize_with_norm(normal)

    # if (p0, p1) parallel: different normal
    if nrm < MJ_MINVAL:
      # find max component of axis0
      axis0_abs = wp.abs(axis0)
      i = int(0)
      if (axis0_abs[1] > axis0_abs[0]) and (axis0_abs[1] > axis0_abs[2]):
        i = 1
      if (axis0_abs[2] > axis0_abs[0]) and (axis0_abs[2] > axis0_abs[1]):
        i = 2

      # init second axis: 0 at i; 1 elsewhere
      axis1 = wp.vec3(1.0)
      axis1[i] = 0.0

      # recompute normal
      normal = wp.cross(axis0, axis1)
      normal, _ = math.normalize_with_norm(normal)

    # 2nd axis = cross(normal, p0)
    axis1 = wp.cross(normal, axis0)
    axis1, _ = math.normalize_with_norm(axis1)
  else:  # WrapType.CYLINDER
    # 1st axis = x
    axis0 = wp.vec3(1.0, 0.0, 0.0)

    # 2nd axis = y
    axis1 = wp.vec3(0.0, 1.0, 0.0)

  # project points in 2D frame: p => end
  end = wp.vec4(
    wp.dot(p0, axis0),
    wp.dot(p0, axis1),
    wp.dot(p1, axis0),
    wp.dot(p1, axis1),
  )

  # handle sidesite
  valid_side = wp.norm_l2(side) < wp.inf

  if valid_side:
    # side point: apply same projection as x0, x1
    sidepnt = xmatT @ (side - xpos)

    # side point: project and rescale
    sidepnt_proj = wp.vec2(
      wp.dot(sidepnt, axis0),
      wp.dot(sidepnt, axis1),
    )

    sidepnt_proj, _ = math.normalize_with_norm(sidepnt_proj)
    sidepnt_proj *= radius
  else:
    sidepnt_proj = wp.vec2(wp.inf)

  # apply inside wrap
  if valid_side and wp.norm_l2(sidepnt) < radius:
    wlen, pnt0, pnt1 = wrap_inside(end, radius)
  else:  # apply circle wrap
    wlen, pnt0, pnt1 = wrap_circle(end, sidepnt_proj, radius)

  # no wrap: return
  if wlen < 0.0:
    return -1.0, wp.vec3(wp.inf), wp.vec3(wp.inf)

  # reconstruct 3D points in local frame: res
  res0 = axis0 * pnt0[0] + axis1 * pnt0[1]
  res1 = axis0 * pnt1[0] + axis1 * pnt1[1]

  # cylinder: correct along z
  if geomtype == int(WrapType.CYLINDER.value):
    # set vertical coordinates
    L0 = wp.sqrt(
      (p0[0] - res0[0]) * (p0[0] - res0[0]) + (p0[1] - res0[1]) * (p0[1] - res0[1])
    )
    L1 = wp.sqrt(
      (p1[0] - res1[0]) * (p1[0] - res1[0]) + (p1[1] - res1[1]) * (p1[1] - res1[1])
    )
    res0[2] = p0[2] + (p1[2] - p0[2]) * L0 / (L0 + wlen + L1)
    res1[2] = p0[2] + (p1[2] - p0[2]) * (L0 + wlen) / (L0 + wlen + L1)

    # correct wlen for height
    height = wp.abs(res1[2] - res0[2])
    wlen = wp.sqrt(wlen * wlen + height * height)

  # map back to global frame: wpnt
  wpnt0 = xmat @ res0 + xpos
  wpnt1 = xmat @ res1 + xpos

  return wlen, wpnt0, wpnt1
