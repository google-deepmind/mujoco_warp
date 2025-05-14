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


import math
from typing import Any

import warp as wp

from .collision_primitive import contact_params
from .collision_primitive import write_contact
from .math import make_frame
from .types import Data
from .types import GeomType
from .types import Model
from .types import vec5

BOX_BOX_BLOCK_DIM = 32


_HUGE_VAL = 1e6
_TINY_VAL = 1e-6


class vec16b(wp.types.vector(length=16, dtype=wp.int8)):
  pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=wp.float32)):
  pass


class mat83f(wp.types.matrix(shape=(8, 3), dtype=wp.float32)):
  pass


class mat16_3f(wp.types.matrix(shape=(16, 3), dtype=wp.float32)):
  pass


Box = mat83f


@wp.func
def _argmin(a: Any) -> wp.int32:
  amin = wp.int32(0)
  vmin = wp.float32(a[0])
  for i in range(1, len(a)):
    if a[i] < vmin:
      amin = i
      vmin = a[i]
  return amin


@wp.func
def box_normals(i: int) -> wp.vec3:
  direction = wp.where(i < 3, -1.0, 1.0)
  mod = i % 3
  if mod == 0:
    return wp.vec3(0.0, direction, 0.0)
  if mod == 1:
    return wp.vec3(0.0, 0.0, direction)
  return wp.vec3(-direction, 0.0, 0.0)


@wp.func
def box(R: wp.mat33, t: wp.vec3, size: wp.vec3) -> Box:
  """Get a transformed box"""
  x = size[0]
  y = size[1]
  z = size[2]
  m = Box()
  for i in range(8):
    ix = wp.where(i & 4, x, -x)
    iy = wp.where(i & 2, y, -y)
    iz = wp.where(i & 1, z, -z)
    m[i] = R @ wp.vec3(ix, iy, iz) + t
  return m


@wp.func
def box_face_verts(box: Box, idx: int) -> mat43f:
  """Get the quad corresponding to a box face"""
  if idx == 0:
    verts = wp.vec4i(0, 4, 5, 1)
  if idx == 1:
    verts = wp.vec4i(0, 2, 6, 4)
  if idx == 2:
    verts = wp.vec4i(6, 7, 5, 4)
  if idx == 3:
    verts = wp.vec4i(2, 3, 7, 6)
  if idx == 4:
    verts = wp.vec4i(1, 5, 7, 3)
  if idx == 5:
    verts = wp.vec4i(0, 1, 3, 2)

  m = mat43f()
  for i in range(4):
    m[i] = box[verts[i]]
  return m


@wp.func
def get_box_axis(axis_idx: int, R: wp.mat33):
  """Get the axis at index axis_idx.
  R: rotation matrix from a to b
  Axes 0-12 are face normals of boxes a & b
  Axes 12-21 are edge cross products."""
  if axis_idx < 6:  # a faces
    axis = R @ wp.vec3(box_normals(axis_idx))
    is_degenerate = False
  elif axis_idx < 12:  # b faces
    axis = wp.vec3(box_normals(axis_idx - 6))
    is_degenerate = False
  else:  # edges cross products
    assert axis_idx < 21
    edges = axis_idx - 12
    axis_a, axis_b = edges / 3, edges % 3
    edge_a = wp.transpose(R)[axis_a]
    if axis_b == 0:
      axis = wp.vec3(0.0, -edge_a[2], edge_a[1])
    elif axis_b == 1:
      axis = wp.vec3(edge_a[2], 0.0, -edge_a[0])
    else:
      axis = wp.vec3(-edge_a[1], edge_a[0], 0.0)
    is_degenerate = wp.length_sq(axis) < _TINY_VAL
  return wp.normalize(axis), is_degenerate


@wp.func
def get_box_axis_support(axis: wp.vec3, degenerate_axis: bool, a: Box, b: Box):
  """Get the overlap (or separating distance if negative) along `axis`, and the sign."""
  axis_d = wp.vec3d(axis)
  support_a_max, support_b_max = wp.float32(-_HUGE_VAL), wp.float32(-_HUGE_VAL)
  support_a_min, support_b_min = wp.float32(_HUGE_VAL), wp.float32(_HUGE_VAL)
  for i in range(8):
    vert_a = wp.vec3d(a[i])
    vert_b = wp.vec3d(b[i])
    proj_a = wp.float32(wp.dot(vert_a, axis_d))
    proj_b = wp.float32(wp.dot(vert_b, axis_d))
    support_a_max = wp.max(support_a_max, proj_a)
    support_b_max = wp.max(support_b_max, proj_b)
    support_a_min = wp.min(support_a_min, proj_a)
    support_b_min = wp.min(support_b_min, proj_b)
  dist1 = support_a_max - support_b_min
  dist2 = support_b_max - support_a_min
  dist = wp.where(degenerate_axis, _HUGE_VAL, wp.min(dist1, dist2))
  sign = wp.where(dist1 > dist2, -1, 1)
  return dist, sign


@wp.struct
class AxisSupport:
  best_dist: wp.float32
  best_sign: wp.int8
  best_idx: wp.int8


@wp.func
def reduce_axis_support(a: AxisSupport, b: AxisSupport):
  return wp.where(a.best_dist > b.best_dist, b, a)


@wp.func
def face_axis_alignment(a: wp.vec3, R: wp.mat33) -> wp.int32:
  """Find the box faces most aligned with the axis `a`"""
  max_dot = wp.float32(0.0)
  max_idx = wp.int32(0)
  for i in range(6):
    d = wp.dot(R @ box_normals(i), a)
    if d > max_dot:
      max_dot = d
      max_idx = i
  return max_idx


@wp.kernel(enable_backward=False)
def _box_box(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_condim: wp.array(dtype=int),
  geom_priority: wp.array(dtype=int),
  geom_solmix: wp.array2d(dtype=float),
  geom_solref: wp.array2d(dtype=wp.vec2),
  geom_solimp: wp.array2d(dtype=vec5),
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_friction: wp.array2d(dtype=wp.vec3),
  geom_margin: wp.array2d(dtype=float),
  geom_gap: wp.array2d(dtype=float),
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
  # In:
  num_kernels_in: int,
  # Data out:
  ncon_out: wp.array(dtype=int),
  contact_dist_out: wp.array(dtype=float),
  contact_pos_out: wp.array(dtype=wp.vec3),
  contact_frame_out: wp.array(dtype=wp.mat33),
  contact_includemargin_out: wp.array(dtype=float),
  contact_dim_out: wp.array(dtype=int),
  contact_friction_out: wp.array(dtype=vec5),
  contact_solref_out: wp.array(dtype=wp.vec2),
  contact_solreffriction_out: wp.array(dtype=wp.vec2),
  contact_solimp_out: wp.array(dtype=vec5),
  contact_geom_out: wp.array(dtype=wp.vec2i),
  contact_worldid_out: wp.array(dtype=int),
):
  """Calculates contacts between pairs of boxes."""
  tid, axis_idx = wp.tid()

  for bp_idx in range(tid, min(ncollision_in[0], nconmax_in), num_kernels_in):
    geoms = collision_pair_in[bp_idx]

    ga, gb = geoms[0], geoms[1]

    if geom_type[ga] != int(GeomType.BOX.value) or geom_type[gb] != int(GeomType.BOX.value):
      continue

    worldid = collision_worldid_in[bp_idx]

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

    # transformations
    a_pos, b_pos = geom_xpos_in[worldid, ga], geom_xpos_in[worldid, gb]
    a_mat, b_mat = geom_xmat_in[worldid, ga], geom_xmat_in[worldid, gb]
    b_mat_inv = wp.transpose(b_mat)
    trans_atob = b_mat_inv @ (a_pos - b_pos)
    rot_atob = b_mat_inv @ a_mat

    a_size = geom_size[worldid, ga]
    b_size = geom_size[worldid, gb]
    a = box(rot_atob, trans_atob, a_size)
    b = box(wp.identity(3, wp.float32), wp.vec3(0.0), b_size)

    # box-box implementation

    # Inlined def collision_axis_tiled( a: Box, b: Box, R: wp.mat33, axis_idx: wp.int32,):
    # Finds the axis of minimum separation.
    # a: Box a vertices, in frame b
    # b: Box b vertices, in frame b
    # R: rotation matrix from a to b
    # Returns:
    #   best_axis: vec3
    #   best_sign: int32
    #   best_idx: int32
    R = rot_atob

    # launch tiled with block_dim=21
    if axis_idx > 20:
      continue

    axis, degenerate_axis = get_box_axis(axis_idx, R)
    axis_dist, axis_sign = get_box_axis_support(axis, degenerate_axis, a, b)

    supports = wp.tile(AxisSupport(axis_dist, wp.int8(axis_sign), wp.int8(axis_idx)))

    face_supports = wp.tile_view(supports, offset=(0,), shape=(12,))
    edge_supports = wp.tile_view(supports, offset=(12,), shape=(9,))

    face_supports_red = wp.tile_reduce(reduce_axis_support, face_supports)
    edge_supports_red = wp.tile_reduce(reduce_axis_support, edge_supports)

    face = face_supports_red[0]
    edge = edge_supports_red[0]

    if axis_idx > 0:  # single thread
      continue

    # choose the best separating axis
    face_axis, _ = get_box_axis(wp.int32(face.best_idx), R)
    best_axis = wp.vec3(face_axis)
    best_sign = wp.int32(face.best_sign)
    best_idx = wp.int32(face.best_idx)
    best_dist = wp.float32(face.best_dist)

    if edge.best_dist < face.best_dist:
      edge_axis, _ = get_box_axis(wp.int32(edge.best_idx), R)
      if wp.abs(wp.dot(face_axis, edge_axis)) < 0.99:
        best_axis = edge_axis
        best_sign = wp.int32(edge.best_sign)
        best_idx = wp.int32(edge.best_idx)
        best_dist = wp.float32(edge.best_dist)
    # end inlined collision_axis_tiled

    # if axis_idx != 0:
    #   continue
    if best_dist < 0:
      continue

    # get the (reference) face most aligned with the separating axis
    a_max = face_axis_alignment(best_axis, rot_atob)
    b_max = face_axis_alignment(best_axis, wp.identity(3, wp.float32))

    sep_axis = wp.float32(best_sign) * best_axis

    if best_sign > 0:
      b_min = (b_max + 3) % 6
      dist, pos = _create_contact_manifold(
        box_face_verts(a, a_max),
        rot_atob @ box_normals(a_max),
        box_face_verts(b, b_min),
        box_normals(b_min),
      )
    else:
      a_min = (a_max + 3) % 6
      dist, pos = _create_contact_manifold(
        box_face_verts(b, b_max),
        box_normals(b_max),
        box_face_verts(a, a_min),
        rot_atob @ box_normals(a_min),
      )

    # For edge contacts, we use the clipped face point, mainly for performance
    # reasons. For small penetration, the clipped face point is roughly the edge
    # contact point.
    if best_idx > 11:  # is_edge_contact
      idx = _argmin(dist)
      dist = wp.vec4f(dist[idx], 1.0, 1.0, 1.0)
      for i in range(4):
        pos[i] = pos[idx]

    margin = wp.max(geom_margin[worldid, ga], geom_margin[worldid, gb])
    for i in range(4):
      pos_glob = b_mat @ pos[i] + b_pos
      n_glob = b_mat @ sep_axis

      write_contact(
        nconmax_in,
        dist[i],
        pos_glob,
        make_frame(n_glob),
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


@wp.func
def _closest_segment_point_plane(a: wp.vec3, b: wp.vec3, p0: wp.vec3, plane_normal: wp.vec3) -> wp.vec3:
  """Gets the closest point between a line segment and a plane.

  Args:
    a: first line segment point
    b: second line segment point
    p0: point on plane
    plane_normal: plane normal

  Returns:
    closest point between the line segment and the plane
  """
  # Parametrize a line segment as S(t) = a + t * (b - a), plug it into the plane
  # equation dot(n, S(t)) - d = 0, then solve for t to get the line-plane
  # intersection. We then clip t to be in [0, 1] to be on the line segment.
  n = plane_normal
  d = wp.dot(p0, n)  # shortest distance from origin to plane
  denom = wp.dot(n, (b - a))
  t = (d - wp.dot(n, a)) / (denom + wp.where(denom == 0.0, _TINY_VAL, 0.0))
  t = wp.clamp(t, 0.0, 1.0)
  segment_point = a + t * (b - a)

  return segment_point


@wp.func
def _project_poly_onto_plane(poly: Any, poly_n: wp.vec3, plane_n: wp.vec3, plane_pt: wp.vec3):
  """Projects poly1 onto the poly2 plane along poly2's normal."""
  d = wp.dot(plane_pt, plane_n)
  denom = wp.dot(poly_n, plane_n)
  qn_scaled = poly_n / (denom + wp.where(denom == 0.0, _TINY_VAL, 0.0))

  for i in range(len(poly)):
    poly[i] = poly[i] + (d - wp.dot(poly[i], plane_n)) * qn_scaled
  return poly


@wp.func
def _clip_edge_to_quad(subject_poly: mat43f, clipping_poly: mat43f, clipping_normal: wp.vec3):
  p0 = mat43f()
  p1 = mat43f()
  mask = wp.vec4b()
  for edge_idx in range(4):
    subject_p0 = subject_poly[(edge_idx + 3) % 4]
    subject_p1 = subject_poly[edge_idx]

    any_both_in_front = wp.int32(0)
    clipped0_dist_max = wp.float32(-_HUGE_VAL)
    clipped1_dist_max = wp.float32(-_HUGE_VAL)
    clipped_p0_distmax = wp.vec3(0.0)
    clipped_p1_distmax = wp.vec3(0.0)

    for clipping_edge_idx in range(4):
      clipping_p0 = clipping_poly[(clipping_edge_idx + 3) % 4]
      clipping_p1 = clipping_poly[clipping_edge_idx]
      edge_normal = wp.cross(clipping_p1 - clipping_p0, clipping_normal)

      p0_in_front = wp.dot(subject_p0 - clipping_p0, edge_normal) > _TINY_VAL
      p1_in_front = wp.dot(subject_p1 - clipping_p0, edge_normal) > _TINY_VAL
      candidate_clipped_p = _closest_segment_point_plane(subject_p0, subject_p1, clipping_p1, edge_normal)
      clipped_p0 = wp.where(p0_in_front, candidate_clipped_p, subject_p0)
      clipped_p1 = wp.where(p1_in_front, candidate_clipped_p, subject_p1)
      clipped_dist_p0 = wp.dot(clipped_p0 - subject_p0, subject_p1 - subject_p0)
      clipped_dist_p1 = wp.dot(clipped_p1 - subject_p1, subject_p0 - subject_p1)
      any_both_in_front |= wp.int32(p0_in_front and p1_in_front)

      if clipped_dist_p0 > clipped0_dist_max:
        clipped0_dist_max = clipped_dist_p0
        clipped_p0_distmax = clipped_p0

      if clipped_dist_p1 > clipped1_dist_max:
        clipped1_dist_max = clipped_dist_p1
        clipped_p1_distmax = clipped_p1
    new_p0 = wp.where(any_both_in_front, subject_p0, clipped_p0_distmax)
    new_p1 = wp.where(any_both_in_front, subject_p1, clipped_p1_distmax)

    mask_val = wp.int8(
      wp.where(
        wp.dot(subject_p0 - subject_p1, new_p0 - new_p1) < 0,
        0,
        wp.int32(not any_both_in_front),
      )
    )

    p0[edge_idx] = new_p0
    p1[edge_idx] = new_p1
    mask[edge_idx] = mask_val
  return p0, p1, mask


@wp.func
def _clip_quad(subject_quad: mat43f, subject_normal: wp.vec3, clipping_quad: mat43f, clipping_normal: wp.vec3):
  """Clips a subject quad against a clipping quad.
  Serial implementation.
  """

  subject_clipped_p0, subject_clipped_p1, subject_mask = _clip_edge_to_quad(subject_quad, clipping_quad, clipping_normal)
  clipping_proj = _project_poly_onto_plane(clipping_quad, clipping_normal, subject_normal, subject_quad[0])
  clipping_clipped_p0, clipping_clipped_p1, clipping_mask = _clip_edge_to_quad(clipping_proj, subject_quad, subject_normal)

  clipped = mat16_3f()
  mask = vec16b()
  for i in range(4):
    clipped[i] = subject_clipped_p0[i]
    clipped[i + 4] = clipping_clipped_p0[i]
    clipped[i + 8] = subject_clipped_p1[i]
    clipped[i + 12] = clipping_clipped_p1[i]
    mask[i] = subject_mask[i]
    mask[i + 4] = clipping_mask[i]
    mask[i + 8] = subject_mask[i]
    mask[i + 8 + 4] = clipping_mask[i]

  return clipped, mask


# TODO(ca): tiling variant
@wp.func
def _manifold_points(poly: Any, mask: Any, clipping_norm: wp.vec3) -> wp.vec4b:
  """Chooses four points on the polygon with approximately maximal area. Return the indices"""
  n = len(poly)

  a_idx = wp.int32(0)
  a_mask = wp.int8(mask[0])
  for i in range(n):
    if mask[i] >= a_mask:
      a_idx = i
      a_mask = mask[i]
  a = poly[a_idx]

  b_idx = wp.int32(0)
  b_dist = wp.float32(-_HUGE_VAL)
  for i in range(n):
    dist = wp.length_sq(poly[i] - a) + wp.where(mask[i], 0.0, -_HUGE_VAL)
    if dist >= b_dist:
      b_idx = i
      b_dist = dist
  b = poly[b_idx]

  ab = wp.cross(clipping_norm, a - b)

  c_idx = wp.int32(0)
  c_dist = wp.float32(-_HUGE_VAL)
  for i in range(n):
    ap = a - poly[i]
    dist = wp.abs(wp.dot(ap, ab)) + wp.where(mask[i], 0.0, -_HUGE_VAL)
    if dist >= c_dist:
      c_idx = i
      c_dist = dist
  c = poly[c_idx]

  ac = wp.cross(clipping_norm, a - c)
  bc = wp.cross(clipping_norm, b - c)

  d_idx = wp.int32(0)
  d_dist = wp.float32(-2.0 * _HUGE_VAL)
  for i in range(n):
    ap = a - poly[i]
    dist_ap = wp.abs(wp.dot(ap, ac)) + wp.where(mask[i], 0.0, -_HUGE_VAL)
    bp = b - poly[i]
    dist_bp = wp.abs(wp.dot(bp, bc)) + wp.where(mask[i], 0.0, -_HUGE_VAL)
    if dist_ap + dist_bp >= d_dist:
      d_idx = i
      d_dist = dist_ap + dist_bp
  d = poly[d_idx]
  return wp.vec4b(wp.int8(a_idx), wp.int8(b_idx), wp.int8(c_idx), wp.int8(d_idx))


@wp.func
def _create_contact_manifold(clipping_quad: mat43f, clipping_normal: wp.vec3, subject_quad: mat43f, subject_normal: wp.vec3):
  # Clip the subject (incident) face onto the clipping (reference) face.
  # The incident points are clipped points on the subject polygon.
  incident, mask = _clip_quad(subject_quad, subject_normal, clipping_quad, clipping_normal)

  clipping_normal_neg = -clipping_normal
  d = wp.dot(clipping_quad[0], clipping_normal_neg) + _TINY_VAL

  for i in range(16):
    if wp.dot(incident[i], clipping_normal_neg) < d:
      mask[i] = wp.int8(0)

  ref = _project_poly_onto_plane(incident, clipping_normal, clipping_normal, clipping_quad[0])

  # Choose four contact points.
  best = _manifold_points(ref, mask, clipping_normal)
  contact_pts = mat43f()
  dist = wp.vec4f()

  for i in range(4):
    idx = wp.int32(best[i])
    contact_pt = ref[idx]
    contact_pts[i] = contact_pt
    penetration_dir = incident[idx] - contact_pt
    penetration = wp.dot(penetration_dir, clipping_normal)
    dist[i] = wp.where(mask[idx], penetration, 1.0)

  return dist, contact_pts


def box_box_narrowphase(
  m: Model,
  d: Data,
):
  """Calculates contacts between pairs of boxes."""
  kernel_ratio = 16
  nthread = math.ceil(d.nconmax / kernel_ratio)  # parallel threads excluding tile dim
  wp.launch_tiled(
    kernel=_box_box,
    dim=nthread,
    inputs=[
      m.geom_type,
      m.geom_condim,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_size,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
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
      nthread,
    ],
    outputs=[
      d.ncon,
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.includemargin,
      d.contact.dim,
      d.contact.friction,
      d.contact.solref,
      d.contact.solreffriction,
      d.contact.solimp,
      d.contact.geom,
      d.contact.worldid,
    ],
    block_dim=BOX_BOX_BLOCK_DIM,
  )
