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
"""Adjoints for convex collision detection.

A non-differentiable pass recovers the active collision features from the retained contacts. A
small differentiable replay then maps contact cotangents to geometry poses. The forward pass does
not need to retain any additional state. Heightfield contacts are currently integrated; the
feature-recovery and replay helpers also provide scaffolding for general GJK/EPA adjoints.
"""

import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import collision_gjk
from mujoco_warp._src import collision_gjk_adjoint
from mujoco_warp._src.collision_convex import _hfield_filter
from mujoco_warp._src.collision_core import geom_collision_pair
from mujoco_warp._src.collision_gjk import _get_face_verts
from mujoco_warp._src.collision_gjk import _tri_affine_coord
from mujoco_warp._src.collision_gjk import support
from mujoco_warp._src.math import make_frame
from mujoco_warp._src.math import upper_trid_index
from mujoco_warp._src.types import MJ_MAX_EPAFACES
from mujoco_warp._src.types import MJ_MAX_EPAHORIZON
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import CollisionConvexBackwardContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import mat63
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": True})


_JUNK_DIST = 0.5 * MJ_MAXVAL


# Packed feature case: tag, GJK dimension, capsule endpoint signs, and distance sign.
_CASE_DEAD = wp.constant(0)
_CASE_FROZEN = wp.constant(1)
_CASE_GJK_FEATURE = wp.constant(2)
_CASE_GJK_SIDE = wp.constant(3)
_CASE_EPA = wp.constant(4)
_CASE_EPA_HFIELD = wp.constant(5)
_CASE_TAG_MASK = wp.constant(7)
_CASE_DIM_SHIFT = wp.constant(3)
_CASE_SIGN_SHIFT = wp.constant(5)
_CASE_NEG_BIT = wp.constant(1 << 11)


def create_collision_convex_backward_context(m: Model, d: Data, nprobe: int | None = None) -> CollisionConvexBackwardContext:
  """Creates reusable convex collision feature-recovery and replay scratch."""
  naconmax = d.contact.pos.shape[0]
  gjk_iterations = int(m.opt.ccd_iterations)
  epa_iterations = int(m.opt.ccd_iterations)
  if nprobe is None:
    nprobe = min(naconmax, max(4 * d.naccdmax, 1))
  return CollisionConvexBackwardContext(
    case=wp.zeros(naconmax, dtype=int),
    prism=wp.zeros(naconmax, dtype=wp.mat33),
    top=wp.zeros(naconmax, dtype=wp.mat33),
    feature=wp.zeros(naconmax, dtype=wp.vec4i),
    direction=wp.zeros(naconmax, dtype=wp.mat33),
    reference=wp.zeros(naconmax, dtype=wp.vec3),
    epa_vert=wp.empty(shape=(nprobe, 10 + 2 * epa_iterations), dtype=wp.vec3),
    epa_vert_index=wp.empty(shape=(nprobe, 10 + 2 * epa_iterations), dtype=int),
    epa_face=wp.empty(shape=(nprobe, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=int),
    epa_pr=wp.empty(shape=(nprobe, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=wp.vec3),
    epa_norm2=wp.empty(shape=(nprobe, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=float),
    epa_horizon=wp.empty(shape=(nprobe, MJ_MAX_EPAHORIZON), dtype=int),
    probe_count=wp.zeros(1, dtype=int),
    overflow=wp.zeros(d.nworld, dtype=int),
    gjk_iterations=gjk_iterations,
    epa_iterations=epa_iterations,
  )


# Isolate iterative collision code from replay adjoint code generation.
@wp.kernel(module="unique", enable_backward=False)
def _hfield_freeze(
  # Model:
  opt_ccd_tolerance: wp.array[float],
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  geom_rbound: wp.array2d[float],
  geom_margin: wp.array2d[float],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  mesh_polymapadr: wp.array[int],
  mesh_polymapnum: wp.array[int],
  mesh_polymap: wp.array[int],
  hfield_size: wp.array[wp.vec4],
  hfield_nrow: wp.array[int],
  hfield_ncol: wp.array[int],
  hfield_adr: wp.array[int],
  hfield_data: wp.array[float],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  contact_dist_in: wp.array[float],
  contact_pos_in: wp.array[wp.vec3],
  contact_frame_in: wp.array[wp.mat33],
  contact_includemargin_in: wp.array[float],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  epa_vert_in: wp.array2d[wp.vec3],
  epa_vert_index_in: wp.array2d[int],
  epa_face_in: wp.array2d[int],
  epa_pr_in: wp.array2d[wp.vec3],
  epa_norm2_in: wp.array2d[float],
  epa_horizon_in: wp.array2d[int],
  probe_count_in: wp.array[int],
  gjk_iterations: int,
  epa_iterations: int,
  # Data out:
  overflow_out: wp.array[int],
  # Out:
  case_out: wp.array[int],
  prism_out: wp.array[wp.mat33],
  top_out: wp.array[wp.mat33],
  feature_out: wp.array[wp.vec4i],
  direction_out: wp.array[wp.mat33],
  reference_out: wp.array[wp.vec3],
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  case_out[cid] = _CASE_DEAD
  geoms = contact_geom_in[cid]
  g1 = geoms[0]
  g2 = geoms[1]
  if g1 < 0 or g2 < 0:
    return
  if geom_type[g1] != GeomType.HFIELD:
    return
  contact_dist = contact_dist_in[cid]
  if wp.abs(contact_dist) >= _JUNK_DIST:
    return

  worldid = contact_worldid_in[cid]
  contact_pos = contact_pos_in[cid]
  contact_frame = contact_frame_in[cid]
  normal = wp.vec3(contact_frame[0, 0], contact_frame[0, 1], contact_frame[0, 2])
  half_dist = 0.5 * wp.abs(contact_dist)
  witness1 = contact_pos + half_dist * normal
  witness2 = contact_pos - half_dist * normal
  geom1_pos = geom_xpos_in[worldid, g1]
  geom1_mat = geom_xmat_in[worldid, g1]
  geom2_pos = geom_xpos_in[worldid, g2]
  geom2_mat = geom_xmat_in[worldid, g2]
  local1 = wp.transpose(geom1_mat) @ (witness1 - geom1_pos)
  local2 = wp.transpose(geom2_mat) @ (witness2 - geom2_pos)
  prism_out[cid] = collision_gjk_adjoint._mat33_rows(local1, local2, wp.vec3(0.0, 0.0, 0.0))
  case_out[cid] = int(_CASE_FROZEN) | wp.where(contact_dist < 0.0, _CASE_NEG_BIT, 0)

  geomtype2 = geom_type[g2]
  supported = (
    geomtype2 == GeomType.SPHERE
    or geomtype2 == GeomType.CAPSULE
    or geomtype2 == GeomType.ELLIPSOID
    or geomtype2 == GeomType.CYLINDER
    or geomtype2 == GeomType.BOX
    or geomtype2 == GeomType.MESH
  )
  if not supported:
    return
  margin = contact_includemargin_in[cid]

  # Reproduce the forward height field subgrid bounds.
  no_hf_collision, xmin, xmax, ymin, ymax, zmin, zmax = _hfield_filter(
    geom_type,
    geom_dataid,
    geom_size,
    geom_rbound,
    geom_margin,
    mesh_vertadr,
    mesh_vertnum,
    mesh_graphadr,
    mesh_vert,
    mesh_graph,
    hfield_size,
    geom_xpos_in,
    geom_xmat_in,
    worldid,
    g1,
    g2,
  )
  if no_hf_collision:
    return

  # Each contact uses its own scratch row for feature recovery.
  rowid = wp.atomic_add(probe_count_in, 0, 1)  # kernel_analyzer: ignore[atomic]
  if rowid >= epa_vert_in.shape[0]:
    return  # Keep the frozen-witness replay when scratch is exhausted.

  geom1, geom2 = geom_collision_pair(
    geom_type,
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_vertnum,
    mesh_graphadr,
    mesh_vert,
    mesh_graph,
    mesh_polynum,
    mesh_polyadr,
    mesh_polynormal,
    mesh_polyvertadr,
    mesh_polyvertnum,
    mesh_polyvert,
    mesh_polymapadr,
    mesh_polymapnum,
    mesh_polymap,
    geom_xpos_in,
    geom_xmat_in,
    geoms,
    worldid,
  )

  hfield_mat_t = wp.transpose(geom1_mat)
  geom2.pos = hfield_mat_t @ (geom2.pos - geom1_pos)
  geom2.rot = hfield_mat_t @ geom2.rot
  geom1.pos = wp.vec3(0.0, 0.0, 0.0)
  geom1.rot = wp.identity(n=3, dtype=float)

  hfield_id = geom_dataid[worldid % geom_dataid.shape[0], g1]
  nrow = hfield_nrow[hfield_id]
  ncol = hfield_ncol[hfield_id]
  size = hfield_size[hfield_id]

  x_scale = 0.5 * float(ncol - 1) / size[0]
  y_scale = 0.5 * float(nrow - 1) / size[1]
  cmin = wp.max(0, int(wp.floor((xmin + size[0]) * x_scale)))
  cmax = wp.min(ncol - 1, int(wp.ceil((xmax + size[0]) * x_scale)))
  rmin = wp.max(0, int(wp.floor((ymin + size[1]) * y_scale)))
  rmax = wp.min(nrow - 1, int(wp.ceil((ymax + size[1]) * y_scale)))

  dx = (2.0 * size[0]) / float(ncol - 1)
  dy = (2.0 * size[1]) / float(nrow - 1)
  dr = wp.vec2i(1, 0)

  hfield_prism = mat63()
  hfield_prism[0, 2] = -size[3]
  hfield_prism[1, 2] = -size[3]
  hfield_prism[2, 2] = -size[3]

  adr = hfield_adr[hfield_id]
  geom2.margin = margin

  tolerance = opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]]
  geom2_axis = wp.vec3(geom2.rot[0, 2], geom2.rot[1, 2], geom2.rot[2, 2])

  epa_vert = epa_vert_in[rowid]
  epa_vert_index = epa_vert_index_in[rowid]
  epa_face = epa_face_in[rowid]
  epa_pr = epa_pr_in[rowid]
  epa_norm2 = epa_norm2_in[rowid]
  epa_horizon = epa_horizon_in[rowid]

  best_error = float(MJ_MAXVAL)
  winning_prism = mat63()

  # Match the retained contact directly. Forward filters by margin + gap before reducing
  # the manifold, so reconstructing its slot from an unfiltered candidate list is incorrect.
  for r in range(rmin, rmax):
    for init_i in range(2):
      x = dx * float(cmin) - size[0]
      y = dy * float(r + dr[init_i]) - size[1]
      z = hfield_data[adr + (r + dr[init_i]) * ncol + cmin] * size[2] + margin
      hfield_prism[0] = hfield_prism[1]
      hfield_prism[1] = hfield_prism[2]
      hfield_prism[3] = hfield_prism[4]
      hfield_prism[4] = hfield_prism[5]
      hfield_prism[2, 0] = x
      hfield_prism[5, 0] = x
      hfield_prism[2, 1] = y
      hfield_prism[5, 1] = y
      hfield_prism[5, 2] = z

    for c in range(cmin + 1, cmax + 1):
      for i in range(2):
        x = dx * float(c) - size[0]
        y = dy * float(r + dr[i]) - size[1]
        z = hfield_data[adr + (r + dr[i]) * ncol + c] * size[2] + margin
        hfield_prism[0] = hfield_prism[1]
        hfield_prism[1] = hfield_prism[2]
        hfield_prism[3] = hfield_prism[4]
        hfield_prism[4] = hfield_prism[5]
        hfield_prism[2, 0] = x
        hfield_prism[5, 0] = x
        hfield_prism[2, 1] = y
        hfield_prism[5, 1] = y
        hfield_prism[5, 2] = z

        if hfield_prism[3, 2] < zmin and hfield_prism[4, 2] < zmin and hfield_prism[5, 2] < zmin:
          continue

        geom1.polyvert = hfield_prism
        center = (
          hfield_prism[0] + hfield_prism[1] + hfield_prism[2] + hfield_prism[3] + hfield_prism[4] + hfield_prism[5]
        ) * wp.static(1.0 / 6.0)

        dist, ncontact, w1, w2, _multiccd_idx = collision_gjk.ccd(
          tolerance,
          0.0,
          gjk_iterations,
          epa_iterations,
          geom1,
          geom2,
          GeomType.HFIELD.value,
          geomtype2,
          center,
          geom2.pos,
          epa_vert,
          epa_vert_index,
          epa_face,
          epa_pr,
          epa_norm2,
          epa_horizon,
          False,
          worldid,
          overflow_out,
        )
        if ncontact == 0 or wp.abs(dist) >= _JUNK_DIST:
          continue
        pos_local = 0.5 * (w1 + w2)
        pos = geom1_mat @ pos_local + geom1_pos
        error = wp.length_sq(pos - contact_pos) + (dist - contact_dist) * (dist - contact_dist)
        if error < best_error:
          best_error = error
          winning_prism = hfield_prism

  if best_error == MJ_MAXVAL:
    return

  geom1.polyvert = winning_prism
  center = (
    winning_prism[0] + winning_prism[1] + winning_prism[2] + winning_prism[3] + winning_prism[4] + winning_prism[5]
  ) * wp.static(1.0 / 6.0)
  _probe_dist, ncontact, _probe_point1, _probe_point2, tag, result, face_index = collision_gjk_adjoint._ccd_probe(
    tolerance,
    gjk_iterations,
    epa_iterations,
    geom1,
    geom2,
    GeomType.HFIELD.value,
    geomtype2,
    center,
    geom2.pos,
    epa_vert,
    epa_vert_index,
    epa_face,
    epa_pr,
    epa_norm2,
    epa_horizon,
    worldid,
    overflow_out,
  )
  if ncontact == 0:
    return

  prism_top = collision_gjk_adjoint._mat33_rows(winning_prism[3], winning_prism[4], winning_prism[5])

  if geomtype2 == GeomType.SPHERE or geomtype2 == GeomType.CAPSULE:
    if tag == collision_gjk_adjoint._TAG_GJK_FEATURE:
      dim = result.dim
      if dim < 1 or dim > 3:
        return
      # Forward's heightfield inflation corrects witnesses spanning prism features.
      is_side = bool(False)
      for i in range(dim):
        if result.simplex_index1[i] != result.simplex_index1[0]:
          is_side = True
      if is_side:
        top_out[cid] = prism_top
        reference_out[cid] = result.x2
        case_out[cid] = _CASE_GJK_SIDE
        return
      signs = int(0)
      point_a = wp.vec3(0.0, 0.0, 0.0)
      point_b = wp.vec3(0.0, 0.0, 0.0)
      point_c = wp.vec3(0.0, 0.0, 0.0)
      for i in range(3):
        if i < dim:
          if i == 0:
            point_a = result.simplex1[0]
          elif i == 1:
            point_b = result.simplex1[1]
          else:
            point_c = result.simplex1[2]
          height = wp.dot(result.simplex2[i] - geom2.pos, geom2_axis)
          code = wp.where(height > 0.0, 1, wp.where(height < 0.0, 2, 0))
          signs |= code << (2 * i)
      prism_out[cid] = collision_gjk_adjoint._mat33_rows(point_a, point_b, point_c)
      top_out[cid] = prism_top
      case_out[cid] = int(_CASE_GJK_FEATURE) | (dim << _CASE_DIM_SHIFT) | (signs << _CASE_SIGN_SHIFT)
      return

  if tag != collision_gjk_adjoint._TAG_EPA or face_index < 0:
    return

  face_vertices = _get_face_verts(epa_face[face_index])
  marker0 = epa_vert_index[2 * face_vertices[0]]
  marker1 = epa_vert_index[2 * face_vertices[1]]
  marker2 = epa_vert_index[2 * face_vertices[2]]
  mixed = marker0 != marker1 or marker0 != marker2

  epa_point_a = wp.vec3(0.0, 0.0, 0.0)
  epa_point_b = wp.vec3(0.0, 0.0, 0.0)
  epa_point_c = wp.vec3(0.0, 0.0, 0.0)
  direction_a = wp.vec3(0.0, 0.0, 0.0)
  direction_b = wp.vec3(0.0, 0.0, 0.0)
  direction_c = wp.vec3(0.0, 0.0, 0.0)
  features = wp.vec4i(-1, -1, -1, -1)
  geom2_size = geom_size[worldid % geom_size.shape[0], g2]
  for k in range(3):
    vertex = face_vertices[k]
    point1 = epa_vert[2 * vertex]
    point2 = epa_vert[2 * vertex + 1]
    feature = epa_vert_index[2 * vertex + 1]
    ok, support_direction = collision_gjk_adjoint._recover_direction(
      mesh_vert, geomtype2, point2, feature, geom2.pos, geom2.rot, geom2_size, margin, geom2.vertadr, -epa_pr[face_index]
    )
    if not ok:
      return
    if k == 0:
      epa_point_a = point1
      direction_a = support_direction
    elif k == 1:
      epa_point_b = point1
      direction_b = support_direction
    else:
      epa_point_c = point1
      direction_c = support_direction
    features[k] = feature

  if mixed and (geomtype2 == GeomType.BOX or geomtype2 == GeomType.MESH):
    delta_a = epa_point_a - epa_vert[2 * face_vertices[0] + 1]
    delta_b = epa_point_b - epa_vert[2 * face_vertices[1] + 1]
    delta_c = epa_point_c - epa_vert[2 * face_vertices[2] + 1]
    weights = _tri_affine_coord(delta_a, delta_b, delta_c, epa_pr[face_index])
    point2 = (
      weights[0] * epa_vert[2 * face_vertices[0] + 1]
      + weights[1] * epa_vert[2 * face_vertices[1] + 1]
      + weights[2] * epa_vert[2 * face_vertices[2] + 1]
    )
    length = wp.norm_l2(point2)
    if length < 1.0e-12:
      return
    features[3] = support(geom2, geomtype2, point2 / length).vertex_index
    if features[3] < 0:
      return

  prism_out[cid] = collision_gjk_adjoint._mat33_rows(epa_point_a, epa_point_b, epa_point_c)
  direction_out[cid] = collision_gjk_adjoint._mat33_rows(direction_a, direction_b, direction_c)
  feature_out[cid] = features
  reference_out[cid] = epa_pr[face_index]
  top_out[cid] = prism_top
  case_out[cid] = wp.where(mixed, int(_CASE_EPA_HFIELD), int(_CASE_EPA))


@wp.func
def _hfield_project_top(top: wp.mat33, point: wp.vec3) -> wp.vec3:
  """Match the prism-top correction in collision_gjk._epa_witness and _inflate."""
  n = wp.vec3(0.0, 0.0, 1.0)
  a, b, c = top[0], top[1], top[2]
  coords = _tri_affine_coord(a, b, c, point)
  if coords[0] > 0.0 and coords[1] > 0.0 and coords[2] > 0.0:
    return coords[0] * a + coords[1] * b + coords[2] * c
  p = wp.where(coords[0] > 0.0, a, wp.where(coords[1] > 0.0, b, c))
  return point - wp.dot(point - p, n) * n


@wp.func
def _hfield_replay_side(
  top: wp.mat33, direction: wp.vec3, pos: wp.vec3, axis: wp.vec3, radius: float, half_length: float, margin: float
) -> tuple[int, float, wp.vec3, wp.vec3]:
  sign = wp.sign(wp.dot(axis, direction))
  x2 = pos + sign * half_length * axis - (radius + 0.5 * margin) * wp.vec3(0.0, 0.0, 1.0)
  x1 = _hfield_project_top(top, x2)
  dvec = x1 - x2
  return 1, -wp.sqrt(wp.dot(dvec, dvec)), x1, x2


@wp.func
def _hfield_correct_witness(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  # In:
  geomtype: int,
  vertex: int,
  top: wp.mat33,
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  margin: float,
  vertadr: int,
  point: wp.vec3,
) -> tuple[int, float, wp.vec3, wp.vec3]:
  """Replay the heightfield-specific EPA witness correction."""
  x2 = wp.vec3()
  if geomtype == GeomType.SPHERE or geomtype == GeomType.CAPSULE:
    axis = wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])
    half_length = wp.where(geomtype == GeomType.CAPSULE, size[1], 0.0)
    sign = wp.sign(wp.dot(axis, point))
    x2 = pos + sign * half_length * axis - (0.5 * margin + size[0]) * wp.vec3(0.0, 0.0, 1.0)
  else:
    length = wp.norm_l2(point)
    if length < 1.0e-12:
      return 0, 0.0, wp.vec3(), wp.vec3()
    x2 = collision_gjk_adjoint._support_frozen(mesh_vert, geomtype, vertex, point / length, pos, mat, size, margin, vertadr)
  x1 = _hfield_project_top(top, x2)
  dvec = x1 - x2
  return 1, -wp.sqrt(wp.dot(dvec, dvec)), x1, x2


@wp.kernel(enable_backward=True)
def _hfield_replay(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  contact_includemargin_in: wp.array[float],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  case_in: wp.array[int],
  prism_in: wp.array[wp.mat33],
  top_in: wp.array[wp.mat33],
  feature_in: wp.array[wp.vec4i],
  direction_in: wp.array[wp.mat33],
  reference_in: wp.array[wp.vec3],
  # Out:
  cpos_out: wp.array[wp.vec3],
  dist_out: wp.array[float],
  frame_out: wp.array[wp.mat33],
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  contact_case = case_in[cid]
  tag = contact_case & _CASE_TAG_MASK
  if tag == _CASE_DEAD:
    return

  geoms = contact_geom_in[cid]
  g1 = geoms[0]
  g2 = geoms[1]
  worldid = contact_worldid_in[cid]
  geom1_pos = geom_xpos_in[worldid, g1]
  geom1_mat = geom_xmat_in[worldid, g1]
  geom2_pos = geom_xpos_in[worldid, g2]
  geom2_mat = geom_xmat_in[worldid, g2]

  if tag == _CASE_FROZEN:
    dist0, x1w, x2w = collision_gjk_adjoint._replay_frozen(
      prism_in[cid], (contact_case & _CASE_NEG_BIT) != 0, geom1_pos, geom1_mat, geom2_pos, geom2_mat
    )
    dist_out[cid] = dist0
    cpos_out[cid] = 0.5 * (x1w + x2w)
    frame_out[cid] = make_frame(x1w - x2w)
    return

  hfield_mat_t = wp.transpose(geom1_mat)
  local_pos = hfield_mat_t @ (geom2_pos - geom1_pos)
  local_mat = hfield_mat_t @ geom2_mat
  margin = contact_includemargin_in[cid]
  geom2_size = geom_size[worldid % geom_size.shape[0], g2]
  geomtype2 = geom_type[g2]

  ok = int(0)
  dist = float(0.0)
  x1 = wp.vec3(0.0, 0.0, 0.0)
  x2 = wp.vec3(0.0, 0.0, 0.0)

  if tag == _CASE_GJK_FEATURE:
    axis_f = wp.vec3(local_mat[0, 2], local_mat[1, 2], local_mat[2, 2])
    half_f = wp.where(geomtype2 == GeomType.CAPSULE, geom2_size[1], 0.0)
    ok, dist, x1, x2, normal_f = collision_gjk_adjoint._replay_gjk_feature(
      (contact_case >> _CASE_DIM_SHIFT) & 3,
      contact_case >> _CASE_SIGN_SHIFT,
      prism_in[cid],
      local_pos,
      axis_f,
      geom2_size[0],
      half_f,
      margin,
    )
    if ok != 0:
      cpos_out[cid] = geom1_mat @ (0.5 * (x1 + x2)) + geom1_pos
      dist_out[cid] = dist
      frame_out[cid] = make_frame(geom1_mat @ normal_f)
    return
  elif tag == _CASE_GJK_SIDE:
    axis_s = wp.vec3(local_mat[0, 2], local_mat[1, 2], local_mat[2, 2])
    half_s = wp.where(geomtype2 == GeomType.CAPSULE, geom2_size[1], 0.0)
    ok, dist, x1, x2 = _hfield_replay_side(top_in[cid], reference_in[cid], local_pos, axis_s, geom2_size[0], half_s, margin)
  elif tag == _CASE_EPA or tag == _CASE_EPA_HFIELD:
    dataid = geom_dataid[worldid % geom_dataid.shape[0], g2]
    vertadr = int(0)
    if geomtype2 == GeomType.MESH and dataid >= 0:
      vertadr = mesh_vertadr[dataid]
    ok, dist, x1, x2 = collision_gjk_adjoint._replay_epa(
      mesh_vert,
      geomtype2,
      feature_in[cid],
      prism_in[cid],
      direction_in[cid],
      reference_in[cid],
      local_pos,
      local_mat,
      geom2_size,
      margin,
      vertadr,
    )
    if ok != 0 and tag == _CASE_EPA_HFIELD:
      ok, dist, x1, x2 = _hfield_correct_witness(
        mesh_vert, geomtype2, feature_in[cid][3], top_in[cid], local_pos, local_mat, geom2_size, margin, vertadr, x2
      )

  if ok == 0:
    return

  cpos_out[cid] = geom1_mat @ (0.5 * (x1 + x2)) + geom1_pos
  dist_out[cid] = dist
  frame_out[cid] = make_frame(geom1_mat @ (x1 - x2))


_CONVEX_TYPES = (
  GeomType.SPHERE,
  GeomType.CAPSULE,
  GeomType.ELLIPSOID,
  GeomType.CYLINDER,
  GeomType.BOX,
  GeomType.MESH,
)


@event_scope
def hfield_contact_pose_vjp(
  m: Model,
  d_out: Data,
  ctx: CollisionConvexBackwardContext,
  res_contact_pos: wp.array,
  res_contact_frame: wp.array,
  adj_contact_dist: wp.array,
  res_geom_xpos: wp.array,
  res_geom_xmat: wp.array,
):
  """Accumulates height field contact cotangents into geometry pose cotangents."""
  ngeomtype = len(GeomType)
  pair_counts = m.geom_pair_type_count
  if not any(pair_counts[upper_trid_index(ngeomtype, GeomType.HFIELD.value, t.value)] for t in _CONVEX_TYPES):
    return

  naconmax = d_out.contact.pos.shape[0]
  ctx.probe_count.zero_()
  ctx.overflow.zero_()

  wp.launch(
    _hfield_freeze,
    dim=naconmax,
    inputs=[
      m.opt.ccd_tolerance,
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      m.geom_rbound,
      m.geom_margin,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_graphadr,
      m.mesh_vert,
      m.mesh_graph,
      m.mesh_polynum,
      m.mesh_polyadr,
      m.mesh_polynormal,
      m.mesh_polyvertadr,
      m.mesh_polyvertnum,
      m.mesh_polyvert,
      m.mesh_polymapadr,
      m.mesh_polymapnum,
      m.mesh_polymap,
      m.hfield_size,
      m.hfield_nrow,
      m.hfield_ncol,
      m.hfield_adr,
      m.hfield_data,
      d_out.geom_xpos,
      d_out.geom_xmat,
      d_out.contact.dist,
      d_out.contact.pos,
      d_out.contact.frame,
      d_out.contact.includemargin,
      d_out.contact.geom,
      d_out.contact.worldid,
      d_out.nacon,
      ctx.epa_vert,
      ctx.epa_vert_index,
      ctx.epa_face,
      ctx.epa_pr,
      ctx.epa_norm2,
      ctx.epa_horizon,
      ctx.probe_count,
      ctx.gjk_iterations,
      ctx.epa_iterations,
    ],
    outputs=[ctx.overflow, ctx.case, ctx.prism, ctx.top, ctx.feature, ctx.direction, ctx.reference],
  )

  pairs = [
    (m.geom_type, None),
    (m.geom_dataid, None),
    (m.geom_size, None),
    (m.mesh_vertadr, None),
    (m.mesh_vert, None),
    (d_out.geom_xpos, res_geom_xpos),
    (d_out.geom_xmat, res_geom_xmat),
    (d_out.contact.includemargin, None),
    (d_out.contact.geom, None),
    (d_out.contact.worldid, None),
    (d_out.nacon, None),
    (ctx.case, None),
    (ctx.prism, None),
    (ctx.top, None),
    (ctx.feature, None),
    (ctx.direction, None),
    (ctx.reference, None),
  ]
  adjoint_util._launch_vjp(
    _hfield_replay,
    naconmax,
    pairs,
    [d_out.contact.pos, d_out.contact.dist, d_out.contact.frame],
    [res_contact_pos, adj_contact_dist, res_contact_frame],
  )
