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

from .collision_primitive import Geom
from .collision_primitive import _geom
from .collision_primitive import contact_params
from .collision_primitive import write_contact
from .math import gjk_normalize
from .math import make_frame
from .math import orthonormal
from .support import all_same
from .support import any_different
from .types import MJ_MINVAL
from .types import Data
from .types import GeomType
from .types import Model
from .types import vec5

# TODO(team): improve compile time to enable backward pass
wp.config.enable_backward = False

FLOAT_MIN = -1e30
FLOAT_MAX = 1e30
EPS_BEST_COUNT = 12
MULTI_CONTACT_COUNT = 4
MULTI_POLYGON_COUNT = 8
MULTI_TILT_ANGLE = 1.0

matc3 = wp.types.matrix(shape=(EPS_BEST_COUNT, 3), dtype=float)
vecc3 = wp.types.vector(EPS_BEST_COUNT * 3, dtype=float)

# Matrix definition for the `tris` scratch space which is used to store the
# triangles of the polytope. Note that the first dimension is 2, as we need
# to store the previous and current polytope. But since Warp doesn't support
# 3D matrices yet, we use 2 * 3 * EPS_BEST_COUNT as the first dimension.
TRIS_DIM = 3 * EPS_BEST_COUNT
mat2c3 = wp.types.matrix(shape=(2 * TRIS_DIM, 3), dtype=float)
mat3p = wp.types.matrix(shape=(MULTI_POLYGON_COUNT, 3), dtype=float)
mat3c = wp.types.matrix(shape=(MULTI_CONTACT_COUNT, 3), dtype=float)
mat43 = wp.types.matrix(shape=(4, 3), dtype=float)

vec6 = wp.types.vector(6, dtype=int)
VECI1 = vec6(0, 0, 0, 1, 1, 2)
VECI2 = vec6(1, 2, 3, 2, 3, 3)


@wp.func
def _gjk_support_geom(geom: Geom, geomtype: int, dir: wp.vec3, verts: wp.array(dtype=wp.vec3)):
  local_dir = wp.transpose(geom.rot) @ dir
  if geomtype == int(GeomType.SPHERE.value):
    support_pt = geom.pos + geom.size[0] * dir
  elif geomtype == int(GeomType.BOX.value):
    res = wp.cw_mul(wp.sign(local_dir), geom.size)
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.CAPSULE.value):
    res = local_dir * geom.size[0]
    # add cylinder contribution
    res[2] += wp.sign(local_dir[2]) * geom.size[1]
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.ELLIPSOID.value):
    res = wp.cw_mul(local_dir, geom.size)
    res = wp.normalize(res)
    # transform to ellipsoid
    res = wp.cw_mul(res, geom.size)
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.CYLINDER.value):
    res = wp.vec3(0.0, 0.0, 0.0)
    # set result in XY plane: support on circle
    d = wp.sqrt(wp.dot(local_dir, local_dir))
    if d > MJ_MINVAL:
      scl = geom.size[0] / d
      res[0] = local_dir[0] * scl
      res[1] = local_dir[1] * scl
    # set result in Z direction
    res[2] = wp.sign(local_dir[2]) * geom.size[1]
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.MESH.value):
    max_dist = float(FLOAT_MIN)
    # exhaustive search over all vertices
    # TODO(team): consider hill-climb over graph data
    for i in range(geom.vertnum):
      vert = verts[geom.vertadr + i]
      dist = wp.dot(vert, local_dir)
      if dist > max_dist:
        max_dist = dist
        support_pt = vert
    support_pt = geom.rot @ support_pt + geom.pos

  return wp.dot(support_pt, dir), support_pt


@wp.func
def _gjk_support(
  # In:
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
  dir: wp.vec3,
  verts: wp.array(dtype=wp.vec3),
):
  # Returns the distance between support points on two geoms, and the support point.
  # Negative distance means objects are not intersecting along direction `dir`.
  # Positive distance means objects are intersecting along the given direction `dir`.

  dist1, s1 = _gjk_support_geom(geom1, geomtype1, dir, verts)
  dist2, s2 = _gjk_support_geom(geom2, geomtype2, -dir, verts)

  support_pt = s1 - s2
  return dist1 + dist2, support_pt


_CONVEX_COLLISION_FUNC = {
  (GeomType.SPHERE.value, GeomType.ELLIPSOID.value),
  (GeomType.SPHERE.value, GeomType.MESH.value),
  (GeomType.CAPSULE.value, GeomType.CYLINDER.value),
  (GeomType.CAPSULE.value, GeomType.ELLIPSOID.value),
  (GeomType.CAPSULE.value, GeomType.MESH.value),
  (GeomType.ELLIPSOID.value, GeomType.ELLIPSOID.value),
  (GeomType.ELLIPSOID.value, GeomType.CYLINDER.value),
  (GeomType.ELLIPSOID.value, GeomType.BOX.value),
  (GeomType.ELLIPSOID.value, GeomType.MESH.value),
  (GeomType.CYLINDER.value, GeomType.CYLINDER.value),
  (GeomType.CYLINDER.value, GeomType.BOX.value),
  (GeomType.CYLINDER.value, GeomType.MESH.value),
  (GeomType.BOX.value, GeomType.MESH.value),
  (GeomType.MESH.value, GeomType.MESH.value),
}


@wp.func
def _expand_polytope(count: int, prev_count: int, dists: vecc3, tris: mat2c3, p: matc3):
  # expand polytope greedily
  for j in range(count):
    best = int(0)
    dd = dists[0]
    for i in range(1, 3 * prev_count):
      if dists[i] < dd:
        dd = dists[i]
        best = i

    dists[best] = float(wp.static(2 * FLOAT_MAX))

    parent_index = best // 3
    child_index = best % 3

    # fill in the new triangle at the next index
    tris[TRIS_DIM + j * 3 + 0] = tris[parent_index * 3 + child_index]
    tris[TRIS_DIM + j * 3 + 1] = tris[parent_index * 3 + ((child_index + 1) % 3)]
    tris[TRIS_DIM + j * 3 + 2] = p[parent_index]

  for r in range(wp.static(EPS_BEST_COUNT * 3)):
    # swap triangles
    swap = tris[TRIS_DIM + r]
    tris[TRIS_DIM + r] = tris[r]
    tris[r] = swap

  return dists, tris


def _gjk_epa_pipeline(
  geomtype1: int,
  geomtype2: int,
  gjk_iterations: int,
  epa_iterations: int,
  epa_exact_neg_distance: bool,
  depth_extension: float,
):
  # determines if two objects intersect, returns simplex and normal
  @wp.func
  def _gjk(
    # Model:
    mesh_vert: wp.array(dtype=wp.vec3),
    # In:
    geom1: Geom,
    geom2: Geom,
  ):
    dir = wp.vec3(0.0, 0.0, 1.0)
    dir_n = -dir
    depth = float(FLOAT_MAX)

    dist_max, simplex0 = _gjk_support(geom1, geom2, geomtype1, geomtype2, dir, mesh_vert)
    dist_min, simplex1 = _gjk_support(geom1, geom2, geomtype1, geomtype2, dir_n, mesh_vert)

    if dist_max < dist_min:
      depth = dist_max
      normal = dir
    else:
      depth = dist_min
      normal = dir_n

    sd = simplex0 - simplex1
    dir = orthonormal(sd)

    dist_max, simplex3 = _gjk_support(geom1, geom2, geomtype1, geomtype2, dir, mesh_vert)

    # Initialize a 2-simplex with simplex[2]==simplex[1]. This ensures the
    # correct winding order for face normals defined below. Face 0 and face 3
    # are degenerate, and face 1 and 2 have opposing normals.
    simplex = mat43()
    simplex[0] = simplex0
    simplex[1] = simplex1
    simplex[2] = simplex[1]
    simplex[3] = simplex3

    if dist_max < depth:
      depth = dist_max
      normal = dir
    if dist_min < depth:
      depth = dist_min
      normal = dir_n

    plane = mat43()
    for _ in range(gjk_iterations):
      # winding orders: plane[0] ccw, plane[1] cw, plane[2] ccw, plane[3] cw
      plane[0] = wp.cross(simplex[3] - simplex[2], simplex[1] - simplex[2])
      plane[1] = wp.cross(simplex[3] - simplex[0], simplex[2] - simplex[0])
      plane[2] = wp.cross(simplex[3] - simplex[1], simplex[0] - simplex[1])
      plane[3] = wp.cross(simplex[2] - simplex[0], simplex[1] - simplex[0])

      # Compute distance of each face halfspace to the origin. If dplane<0, then the
      # origin is outside the halfspace. If dplane>0 then the origin is inside
      # the halfspace defined by the face plane.

      dplane = wp.vec4(float(FLOAT_MAX))

      plane0, p0 = gjk_normalize(plane[0])
      plane1, p1 = gjk_normalize(plane[1])
      plane2, p2 = gjk_normalize(plane[2])
      plane3, p3 = gjk_normalize(plane[3])

      plane[0] = plane0
      plane[1] = plane1
      plane[2] = plane2
      plane[3] = plane3

      if p0:
        dplane[0] = wp.dot(plane[0], simplex[2])

      if p1:
        dplane[1] = wp.dot(plane[1], simplex[0])

      if p2:
        dplane[2] = wp.dot(plane[2], simplex[1])

      if p3:
        dplane[3] = wp.dot(plane[3], simplex[0])

      # pick plane normal with minimum distance to the origin
      i1 = wp.where(dplane[0] < dplane[1], 0, 1)
      i2 = wp.where(dplane[2] < dplane[3], 2, 3)
      index = wp.where(dplane[i1] < dplane[i2], i1, i2)

      if dplane[index] > 0.0:
        # origin is inside the simplex, objects are intersecting
        break

      # add new support point to the simplex
      dist, simplex_i = _gjk_support(geom1, geom2, geomtype1, geomtype2, plane[index], mesh_vert)
      simplex[index] = simplex_i

      if dist < depth:
        depth = dist
        normal = plane[index]

      # preserve winding order of the simplex faces
      index1 = (index + 1) & 3
      index2 = (index + 2) & 3
      swap = simplex[index1]
      simplex[index1] = simplex[index2]
      simplex[index2] = swap

      if dist < 0.0:
        break  # objects are likely non-intersecting

    return simplex, normal

  # compute contact normal and depth
  @wp.func
  def _epa(
    # Model:
    mesh_vert: wp.array(dtype=wp.vec3),
    # In:
    geom1: Geom,
    geom2: Geom,
    simplex: mat43,
    normal: wp.vec3,
  ):
    # get the support, if depth < 0: objects do not intersect
    depth, _ = _gjk_support(geom1, geom2, geomtype1, geomtype2, normal, mesh_vert)

    if depth < -depth_extension:
      # Objects are not intersecting, and we do not obtain the closest points as
      # specified by depth_extension.
      return wp.nan, wp.vec3(wp.nan, wp.nan, wp.nan)

    if wp.static(epa_exact_neg_distance):
      # Check closest points to all edges of the simplex, rather than just the
      # face normals. This gives the exact depth/normal for the non-intersecting
      # case.
      for i in range(6):
        i1 = VECI1[i]
        i2 = VECI2[i]

        si1 = simplex[i1]
        si2 = simplex[i2]

        if si1[0] != si2[0] or si1[1] != si2[1] or si1[2] != si2[2]:
          v = si1 - si2
          alpha = wp.dot(si1, v) / wp.dot(v, v)

          # p0 is the closest segment point to the origin
          p0 = wp.clamp(alpha, 0.0, 1.0) * v - si1
          p0, pf = gjk_normalize(p0)

          if pf:
            depth2, _ = _gjk_support(geom1, geom2, geomtype1, geomtype2, p0, mesh_vert)

            if depth2 < depth:
              depth = depth2
              normal = p0

    # supporting points for each triangle
    p = matc3()

    # distance to the origin for candidate triangles
    dists = vecc3()

    tris = mat2c3()
    tris[0] = simplex[2]
    tris[1] = simplex[1]
    tris[2] = simplex[3]

    tris[3] = simplex[0]
    tris[4] = simplex[2]
    tris[5] = simplex[3]

    tris[6] = simplex[1]
    tris[7] = simplex[0]
    tris[8] = simplex[3]

    tris[9] = simplex[0]
    tris[10] = simplex[1]
    tris[11] = simplex[2]

    # Calculate the total number of iterations to avoid nested loop
    # This is a hack to reduce compile time
    count = int(4)
    it = int(0)
    for _ in range(wp.static(epa_iterations)):
      it += count
      count = wp.min(count * 3, EPS_BEST_COUNT)

    count = int(4)
    i = int(0)
    for _ in range(it):
      # Loop through all triangles, and obtain distances to the origin for each
      # new triangle candidate.
      ti = 3 * i
      n = wp.cross(tris[ti + 2] - tris[ti + 0], tris[ti + 1] - tris[ti + 0])

      n, nf = gjk_normalize(n)
      if not nf:
        for j in range(3):
          dists[i * 3 + j] = wp.static(float(2 * FLOAT_MAX))
        continue

      dist, pi = _gjk_support(geom1, geom2, geomtype1, geomtype2, n, mesh_vert)
      p[i] = pi

      if dist < depth:
        depth = dist
        normal = n

      # iterate over edges and get distance using support point
      for j in range(3):
        if wp.static(epa_exact_neg_distance):
          # obtain closest point between new triangle edge and origin
          tqj = tris[ti + j]

          if (p[i, 0] != tqj[0]) or (p[i, 1] != tqj[1]) or (p[i, 2] != tqj[2]):
            v = p[i] - tris[ti + j]
            alpha = wp.dot(p[i], v) / wp.dot(v, v)
            p0 = wp.clamp(alpha, 0.0, 1.0) * v - p[i]
            p0, pf = gjk_normalize(p0)

            if pf:
              dist2, v = _gjk_support(geom1, geom2, geomtype1, geomtype2, p0, mesh_vert)

              if dist2 < depth:
                depth = dist2
                normal = p0

        plane = wp.cross(p[i] - tris[ti + j], tris[ti + ((j + 1) % 3)] - tris[ti + j])
        plane, pf = gjk_normalize(plane)

        if pf:
          dd = wp.dot(plane, tris[ti + j])
        else:
          dd = float(FLOAT_MAX)

        if (dd < 0 and depth >= 0) or (
          tris[ti + ((j + 2) % 3)][0] == p[i][0]
          and tris[ti + ((j + 2) % 3)][1] == p[i][1]
          and tris[ti + ((j + 2) % 3)][2] == p[i][2]
        ):
          dists[i * 3 + j] = float(FLOAT_MAX)
        else:
          dists[i * 3 + j] = dd

      if i == count - 1:
        prev_count = count
        count = wp.min(count * 3, EPS_BEST_COUNT)
        dists, tris = _expand_polytope(count, prev_count, dists, tris, p)
        i = int(0)
      else:
        i += 1

    return depth, normal

  @wp.func
  def _multiple_contacts(
    # Model:
    mesh_vert: wp.array(dtype=wp.vec3),
    # In:
    geom1: Geom,
    geom2: Geom,
    depth: float,
    normal: wp.vec3,
  ):
    # Calculates multiple contact points given the normal from EPA.
    #  1. Calculates the polygon on each shape by tiling the normal
    #     "MULTI_TILT_ANGLE" degrees in the orthogonal component of the normal.
    #     The "MULTI_TILT_ANGLE" can be changed to depend on the depth of the
    #     contact, in a future version.
    #  2. The normal is tilted "MULTI_POLYGON_COUNT" times in the directions evenly
    #    spaced in the orthogonal component of the normal.
    #    (works well for >= 6, default is 8).
    #  3. The intersection between these two polygons is calculated in 2D space
    #    (complement to the normal). If they intersect, extreme points in both
    #    directions are found. This can be modified to the extremes in the
    #    direction of eigenvectors of the variance of points of each polygon. If
    #    they do not intersect, the closest points of both polygons are found.
    if depth < -depth_extension:
      return 0, mat3c()

    dir = orthonormal(normal)
    dir2 = wp.cross(normal, dir)

    angle = wp.static(MULTI_TILT_ANGLE * wp.pi / 180.0)
    c = wp.static(wp.cos(angle))
    s = wp.static(wp.sin(angle))
    tc = wp.static(1.0 - c)

    v1 = mat3p()
    v2 = mat3p()

    contact_points = mat3c()

    # Obtain points on the polygon determined by the support and tilt angle,
    # in the basis of the contact frame.
    v1count = int(0)
    v2count = int(0)
    angle_ratio = wp.static(2.0 * wp.pi / float(MULTI_POLYGON_COUNT))

    for i in range(wp.static(MULTI_POLYGON_COUNT)):
      angle = angle_ratio * float(i)
      axis = wp.cos(angle) * dir + wp.sin(angle) * dir2

      # Axis-angle rotation matrix. See
      # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
      mat0 = c + axis[0] * axis[0] * tc
      mat5 = c + axis[1] * axis[1] * tc
      mat10 = c + axis[2] * axis[2] * tc
      t1 = axis[0] * axis[1] * tc
      t2 = axis[2] * s
      mat4 = t1 + t2
      mat1 = t1 - t2
      t1 = axis[0] * axis[2] * tc
      t2 = axis[1] * s
      mat8 = t1 - t2
      mat2 = t1 + t2
      t1 = axis[1] * axis[2] * tc
      t2 = axis[0] * s
      mat9 = t1 + t2
      mat6 = t1 - t2

      n = wp.vec3(
        mat0 * normal[0] + mat1 * normal[1] + mat2 * normal[2],
        mat4 * normal[0] + mat5 * normal[1] + mat6 * normal[2],
        mat8 * normal[0] + mat9 * normal[1] + mat10 * normal[2],
      )

      _, p = _gjk_support_geom(geom1, geomtype1, n, mesh_vert)
      v1[v1count] = wp.vec3(wp.dot(p, dir), wp.dot(p, dir2), wp.dot(p, normal))

      if i == 0:
        v1count += 1
      elif any_different(v1[v1count], v1[v1count - 1]):
        v1count += 1

      n = -n
      _, p = _gjk_support_geom(geom2, geomtype2, n, mesh_vert)
      v2[v2count] = wp.vec3(wp.dot(p, dir), wp.dot(p, dir2), wp.dot(p, normal))

      if i == 0:
        v2count += 1
      elif any_different(v2[v2count], v2[v2count - 1]):
        v2count += 1

    # remove duplicate vertices on the array boundary
    if v1count > 1 and all_same(v1[v1count - 1], v1[0]):
      v1count -= 1

    if v2count > 1 and all_same(v2[v2count - 1], v2[0]):
      v2count -= 1

    # find an intersecting polygon between v1 and v2 in the 2D plane
    out = mat43()
    candCount = int(0)

    if v2count > 1:
      for i in range(v1count):
        m1a = v1[i]
        is_in = bool(True)

        # check if point m1a is inside the v2 polygon on the 2D plane
        for j in range(v2count):
          j2 = (j + 1) % v2count

          # Checks that orientation of the triangle (v2[j], v2[j2], m1a) is
          # counter-clockwise. If so, point m1a is inside the v2 polygon.
          is_in = is_in and ((v2[j2][0] - v2[j][0]) * (m1a[1] - v2[j][1]) - (v2[j2][1] - v2[j][1]) * (m1a[0] - v2[j][0]) >= 0.0)

          if not is_in:
            break

        if is_in:
          if not candCount or m1a[0] < out[0, 0]:
            out[0] = m1a
          if not candCount or m1a[0] > out[1, 0]:
            out[1] = m1a
          if not candCount or m1a[1] < out[2, 1]:
            out[2] = m1a
          if not candCount or m1a[1] > out[3, 1]:
            out[3] = m1a
          candCount += 1

    if v1count > 1:
      for i in range(v2count):
        m1a = v2[i]
        is_in = bool(True)

        for j in range(v1count):
          j2 = (j + 1) % v1count
          is_in = is_in and (v1[j2][0] - v1[j][0]) * (m1a[1] - v1[j][1]) - (v1[j2][1] - v1[j][1]) * (m1a[0] - v1[j][0]) >= 0.0
          if not is_in:
            break

        if is_in:
          if not candCount or m1a[0] < out[0, 0]:
            out[0] = m1a
          if not candCount or m1a[0] > out[1, 0]:
            out[1] = m1a
          if not candCount or m1a[1] < out[2, 1]:
            out[2] = m1a
          if not candCount or m1a[1] > out[3, 1]:
            out[3] = m1a
          candCount += 1

    if v1count > 1 and v2count > 1:
      # Check all edge pairs, and store line segment intersections if they are
      # on the edge of the boundary.
      for i in range(v1count):
        for j in range(v2count):
          m1a = v1[i]
          m1b = v1[(i + 1) % v1count]
          m2a = v2[j]
          m2b = v2[(j + 1) % v2count]

          det = (m2a[1] - m2b[1]) * (m1b[0] - m1a[0]) - (m1a[1] - m1b[1]) * (m2b[0] - m2a[0])

          if wp.abs(det) > 1e-12:
            a11 = (m2a[1] - m2b[1]) / det
            a12 = (m2b[0] - m2a[0]) / det
            a21 = (m1a[1] - m1b[1]) / det
            a22 = (m1b[0] - m1a[0]) / det
            b1 = m2a[0] - m1a[0]
            b2 = m2a[1] - m1a[1]

            alpha = a11 * b1 + a12 * b2
            beta = a21 * b1 + a22 * b2
            if alpha >= 0.0 and alpha <= 1.0 and beta >= 0.0 and beta <= 1.0:
              m0 = wp.vec3(
                m1a[0] + alpha * (m1b[0] - m1a[0]),
                m1a[1] + alpha * (m1b[1] - m1a[1]),
                (m1a[2] + alpha * (m1b[2] - m1a[2]) + m2a[2] + beta * (m2b[2] - m2a[2])) * 0.5,
              )
              if not candCount or m0[0] < out[0, 0]:
                out[0] = m0
              if not candCount or m0[0] > out[1, 0]:
                out[1] = m0
              if not candCount or m0[1] < out[2, 1]:
                out[2] = m0
              if not candCount or m0[1] > out[3, 1]:
                out[3] = m0
              candCount += 1

    var_rx = wp.vec3(0.0)
    contact_count = int(0)
    if candCount > 0:
      # Polygon intersection was found.
      # TODO(btaba): replace the above routine with the manifold point routine
      # from MJX. Deduplicate the points properly.
      last_pt = wp.vec3(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX)

      for k in range(wp.static(MULTI_CONTACT_COUNT)):
        pt = out[k, 0] * dir + out[k, 1] * dir2 + out[k, 2] * normal

        # skip contact points that are too close
        if wp.length(pt - last_pt) <= 1e-6:
          continue

        contact_points[contact_count] = pt
        last_pt = pt
        contact_count += 1

    else:
      # Polygon intersection was not found. Loop through all vertex pairs and
      # calculate an approximate contact point.
      minDist = float(0.0)
      for i in range(v1count):
        for j in range(v2count):
          # Find the closest vertex pair. Calculate a contact point var_rx as the
          # midpoint between the closest vertex pair.
          m1 = v1[i]
          m2 = v2[j]
          dd = (m1[0] - m2[0]) * (m1[0] - m2[0]) + (m1[1] - m2[1]) * (m1[1] - m2[1])

          if i != 0 and j != 0 or dd < minDist:
            minDist = dd
            var_rx = ((m1[0] + m2[0]) * dir + (m1[1] + m2[1]) * dir2 + (m1[2] + m2[2]) * normal) * 0.5

          # Check for a closer point between a point on v2 and an edge on v1.
          m1b = v1[(i + 1) % v1count]
          m2b = v2[(j + 1) % v2count]

          if v1count > 1:
            dd = (m1b[0] - m1[0]) * (m1b[0] - m1[0]) + (m1b[1] - m1[1]) * (m1b[1] - m1[1])
            t = ((m2[1] - m1[1]) * (m1b[0] - m1[0]) - (m2[0] - m1[0]) * (m1b[1] - m1[1])) / dd
            dx = m2[0] + (m1b[1] - m1[1]) * t
            dy = m2[1] - (m1b[0] - m1[0]) * t
            dist = (dx - m2[0]) * (dx - m2[0]) + (dy - m2[1]) * (dy - m2[1])

            if (
              (dist < minDist)
              and ((dx - m1[0]) * (m1b[0] - m1[0]) + (dy - m1[1]) * (m1b[1] - m1[1]) >= 0)
              and ((dx - m1b[0]) * (m1[0] - m1b[0]) + (dy - m1b[1]) * (m1[1] - m1b[1]) >= 0)
            ):
              alpha = wp.sqrt(((dx - m1[0]) * (dx - m1[0]) + (dy - m1[1]) * (dy - m1[1])) / dd)
              minDist = dist
              w = ((1.0 - alpha) * m1 + alpha * m1b + m2) * 0.5
              var_rx = w[0] * dir + w[1] * dir2 + w[2] * normal

          # check for a closer point between a point on v1 and an edge on v2
          if v2count > 1:
            dd = (m2b[0] - m2[0]) * (m2b[0] - m2[0]) + (m2b[1] - m2[1]) * (m2b[1] - m2[1])
            t = ((m1[1] - m2[1]) * (m2b[0] - m2[0]) - (m1[0] - m2[0]) * (m2b[1] - m2[1])) / dd
            dx = m1[0] + (m2b[1] - m2[1]) * t
            dy = m1[1] - (m2b[0] - m2[0]) * t
            dist = (dx - m1[0]) * (dx - m1[0]) + (dy - m1[1]) * (dy - m1[1])

            if (
              dist < minDist
              and (dx - m2[0]) * (m2b[0] - m2[0]) + (dy - m2[1]) * (m2b[1] - m2[1]) >= 0
              and (dx - m2b[0]) * (m2[0] - m2b[0]) + (dy - m2b[1]) * (m2[1] - m2b[1]) >= 0
            ):
              alpha = wp.sqrt(((dx - m2[0]) * (dx - m2[0]) + (dy - m2[1]) * (dy - m2[1])) / dd)
              minDist = dist
              w = (m1 + (1.0 - alpha) * m2 + alpha * m2b) * 0.5
              var_rx = w[0] * dir + w[1] * dir2 + w[2] * normal

      for k in range(wp.static(MULTI_CONTACT_COUNT)):
        contact_points[k] = var_rx

      contact_count = 1

    return contact_count, contact_points

  # runs GJK and EPA on a set of sparse geom pairs per env
  @wp.kernel
  def gjk_epa_sparse(
    # Model:
    geom_type: wp.array(dtype=int),
    geom_condim: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
    mesh_vertadr: wp.array(dtype=int),
    mesh_vertnum: wp.array(dtype=int),
    mesh_vert: wp.array(dtype=wp.vec3),
    pair_dim: wp.array(dtype=int),
    pair_solref: wp.array2d(dtype=wp.vec2),
    pair_solreffriction: wp.array2d(dtype=wp.vec2),
    pair_solimp: wp.array2d(dtype=vec5),
    pair_margin: wp.array2d(dtype=float),
    pair_gap: wp.array2d(dtype=float),
    pair_friction: wp.array2d(dtype=vec5),
    # Data in:
    nconmax_in: int,
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xmat_in: wp.array2d(dtype=wp.mat33),
    collision_pair_in: wp.array(dtype=wp.vec2i),
    collision_pairid_in: wp.array(dtype=int),
    collision_worldid_in: wp.array(dtype=int),
    ncollision_in: wp.array(dtype=int),
    # Data out:
    ncon_out: wp.array(dtype=int),
    contact_dist_out: wp.array(dtype=float),
    contact_pos_out: wp.array(dtype=wp.vec3),
    contact_frame_out: wp.array(dtype=wp.mat33),
    contact_includemargin_out: wp.array(dtype=float),
    contact_friction_out: wp.array(dtype=vec5),
    contact_solref_out: wp.array(dtype=wp.vec2),
    contact_solreffriction_out: wp.array(dtype=wp.vec2),
    contact_solimp_out: wp.array(dtype=vec5),
    contact_dim_out: wp.array(dtype=int),
    contact_geom_out: wp.array(dtype=wp.vec2i),
    contact_worldid_out: wp.array(dtype=int),
  ):
    tid = wp.tid()
    if tid >= ncollision_in[0]:
      return

    worldid = collision_worldid_in[tid]
    geoms, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
      geom_condim,
      geom_priority,
      geom_solmix,
      geom_solref,
      geom_solimp,
      geom_friction,
      geom_margin,
      geom_gap,
      pair_dim,
      pair_solref,
      pair_solreffriction,
      pair_solimp,
      pair_margin,
      pair_gap,
      pair_friction,
      collision_pair_in,
      collision_pairid_in,
      tid,
      worldid,
    )

    g1 = geoms[0]
    g2 = geoms[1]

    if geom_type[g1] != geomtype1 or geom_type[g2] != geomtype2:
      return

    geom1 = _geom(
      geom_type,
      geom_dataid,
      geom_size[worldid],
      mesh_vertadr,
      mesh_vertnum,
      mesh_vert,
      geom_xpos_in,
      geom_xmat_in,
      worldid,
      g1,
    )

    geom2 = _geom(
      geom_type,
      geom_dataid,
      geom_size[worldid],
      mesh_vertadr,
      mesh_vertnum,
      mesh_vert,
      geom_xpos_in,
      geom_xmat_in,
      worldid,
      g2,
    )

    margin = wp.max(geom_margin[worldid, g1], geom_margin[worldid, g2])

    simplex, normal = _gjk(mesh_vert, geom1, geom2)

    # TODO(btaba): get depth from GJK, conditionally run EPA.
    depth, normal = _epa(mesh_vert, geom1, geom2, simplex, normal)
    dist = -depth

    if (dist - margin) >= 0.0 or depth != depth:
      return

    # TODO(btaba): split get_multiple_contacts into a separate kernel.
    # TODO(team): multiccd enablebit
    count, points = _multiple_contacts(mesh_vert, geom1, geom2, depth, normal)

    frame = make_frame(normal)
    for i in range(count):
      write_contact(
        nconmax_in,
        dist,
        points[i],
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
        ncon_out,
        contact_dist_out,
        contact_pos_out,
        contact_frame_out,
        contact_includemargin_out,
        contact_friction_out,
        contact_solref_out,
        contact_solreffriction_out,
        contact_solimp_out,
        contact_dim_out,
        contact_geom_out,
        contact_worldid_out,
      )

  return gjk_epa_sparse


_collision_kernels = {}


def gjk_narrowphase(m: Model, d: Data):
  if len(_collision_kernels) == 0:
    for types in _CONVEX_COLLISION_FUNC:
      t1 = types[0]
      t2 = types[1]
      _collision_kernels[(t1, t2)] = _gjk_epa_pipeline(
        t1,
        t2,
        m.opt.gjk_iterations,
        m.opt.epa_iterations,
        m.opt.epa_exact_neg_distance,
        m.opt.depth_extension,
      )

  for collision_kernel in _collision_kernels.values():
    wp.launch(
      collision_kernel,
      dim=d.nconmax,
      inputs=[
        m.geom_type,
        m.geom_condim,
        m.geom_dataid,
        m.geom_priority,
        m.geom_solmix,
        m.geom_solref,
        m.geom_solimp,
        m.geom_size,
        m.geom_friction,
        m.geom_margin,
        m.geom_gap,
        m.mesh_vertadr,
        m.mesh_vertnum,
        m.mesh_vert,
        m.pair_dim,
        m.pair_solref,
        m.pair_solreffriction,
        m.pair_solimp,
        m.pair_margin,
        m.pair_gap,
        m.pair_friction,
        d.nconmax,
        d.geom_xpos,
        d.geom_xmat,
        d.collision_pair,
        d.collision_pairid,
        d.collision_worldid,
        d.ncollision,
      ],
      outputs=[
        d.ncon,
        d.contact.dist,
        d.contact.pos,
        d.contact.frame,
        d.contact.includemargin,
        d.contact.friction,
        d.contact.solref,
        d.contact.solreffriction,
        d.contact.solimp,
        d.contact.dim,
        d.contact.geom,
        d.contact.worldid,
      ],
    )
