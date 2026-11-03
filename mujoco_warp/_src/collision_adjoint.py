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
"""VJPs for collision geometry and contact position dependence."""

import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import collision_primitive_core
from mujoco_warp._src import math
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import support
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.warp_util import event_scope

# adjoint module: backward stays on so AD leaves differentiate through cross-module @wp.funcs
wp.set_module_options({"enable_backward": True})


@wp.func
def _capsule_box_coordinates(
  # In:
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_half_length: float,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
  feature: wp.vec3i,
) -> wp.vec2:
  """Recomputes capsule and box-edge coordinates with the closest feature fixed."""
  cltype, clcorner, cledge = feature[0], feature[1], feature[2]
  if cltype < 0:
    return wp.vec2(wp.where(cltype == -3, -1.0, 1.0), 0.0)

  s1 = cltype // 3
  s2 = cltype % 3

  boxmatT = wp.transpose(box_rot)
  pos = boxmatT @ (capsule_pos - box_pos)
  halfaxis = (boxmatT @ capsule_axis) * capsule_half_length
  bx = wp.where(cledge == 0, 0.0, wp.where((clcorner & 1) != 0, box_size[0], -box_size[0]))
  by = wp.where(cledge == 1, 0.0, wp.where((clcorner & 2) != 0, box_size[1], -box_size[1]))
  bz = wp.where(cledge == 2, 0.0, wp.where((clcorner & 4) != 0, box_size[2], -box_size[2]))
  dif = wp.vec3(bx, by, bz) - pos
  sj, dj, hj = box_size[cledge], dif[cledge], halfaxis[cledge]
  u = -sj * dj
  v = wp.dot(halfaxis, dif)
  ma = sj * sj
  mb = -sj * hj
  mc = capsule_half_length * capsule_half_length
  primary = wp.where(s2 == 0, -1.0, 1.0)
  boxcoord = wp.where(s1 == 0, -1.0, 1.0)
  if s2 == 1:
    if s1 == 1:
      idet = math.safe_div(1.0, ma * mc - mb * mb)
      return wp.vec2((ma * v - mb * u) * idet, (mc * u - mb * v) * idet)
    primary = math.safe_div(v - mb * boxcoord, mc)
  if s1 == 1:
    boxcoord = math.safe_div(u - mb * primary, ma)
  return wp.vec2(primary, boxcoord)


@wp.func
def _capsule_box_from_witness(
  # In:
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
  slot: int,
  feature: wp.vec3i,
) -> tuple[float, wp.vec3, wp.vec3]:
  """Replays one capsule-box contact with the closest feature fixed."""
  coords = _capsule_box_coordinates(capsule_pos, capsule_axis, capsule_half_length, box_pos, box_rot, box_size, feature)
  segment = coords[0]
  if slot == 1:
    boxmatT = wp.transpose(box_rot)
    pos = boxmatT @ (capsule_pos - box_pos)
    axis = boxmatT @ capsule_axis
    segment += collision_primitive_core._capsule_box_secondary(
      pos, axis, axis * capsule_half_length, box_size, coords[0], coords[1], feature
    )
  return collision_primitive_core._capsule_box_contact(
    capsule_pos, capsule_axis, capsule_radius, capsule_half_length, box_pos, box_rot, box_size, segment
  )


@wp.func
def _box_box_from_witness(
  # In:
  box1_pos: wp.vec3,
  box1_rot: wp.mat33,
  box1_size: wp.vec3,
  box2_pos: wp.vec3,
  box2_rot: wp.mat33,
  box2_size: wp.vec3,
  feature: wp.vec3i,
  candidate: int,
) -> tuple[float, wp.vec3, wp.vec3]:
  """Replays one box contact with its separating axis and clipping boundaries fixed."""
  code = feature[0]
  rot = wp.transpose(box1_rot) @ box2_rot
  pos21 = wp.transpose(box1_rot) @ (box2_pos - box1_pos)
  if code >= 6:
    i, j = (code - 6) // 3, (code - 6) % 3
    d1 = wp.vec3(float(i == 0), float(i == 1), float(i == 2))
    d2 = wp.vec3(rot[0, j], rot[1, j], rot[2, j])
    axis = float(candidate) * wp.normalize(wp.cross(d1, d2))
    c1 = wp.vec3(
      wp.where(i == 0, 0.0, wp.where((feature[1] & 1) != 0, box1_size[0], -box1_size[0])),
      wp.where(i == 1, 0.0, wp.where((feature[1] & 2) != 0, box1_size[1], -box1_size[1])),
      wp.where(i == 2, 0.0, wp.where((feature[1] & 4) != 0, box1_size[2], -box1_size[2])),
    )
    corner2 = wp.vec3(
      wp.where(j == 0, 0.0, wp.where((feature[2] & 1) != 0, box2_size[0], -box2_size[0])),
      wp.where(j == 1, 0.0, wp.where((feature[2] & 2) != 0, box2_size[1], -box2_size[1])),
      wp.where(j == 2, 0.0, wp.where((feature[2] & 4) != 0, box2_size[2], -box2_size[2])),
    )
    c2 = rot @ corner2 + pos21
    e = c2 - c1
    b = d2[i]
    denom = 1.0 - b * b
    d1e, d2e = e[i], wp.dot(d2, e)
    s = float(0.0)
    if denom >= MJ_MINVAL:
      s = (d1e - b * d2e) / denom
    s = wp.clamp(s, -box1_size[i], box1_size[i])
    t = wp.clamp(b * s - d2e, -box2_size[j], box2_size[j])
    s = wp.clamp(d1e + b * t, -box1_size[i], box1_size[i])
    w1, w2 = c1 + s * d1, c2 + t * d2
    return wp.dot(w2 - w1, axis), box1_rot @ (0.5 * (w1 + w2)) + box1_pos, box1_rot @ axis

  ref1 = code < 3
  a = wp.where(ref1, code, code - 3)
  ax, ay = (a + 1) % 3, (a + 2) % 3
  sizeref = wp.where(ref1, box1_size, box2_size)
  sizeinc = wp.where(ref1, box2_size, box1_size)
  posref = wp.where(ref1, box1_pos, box2_pos)
  matref = wp.where(ref1, box1_rot, box2_rot)
  posoi = wp.where(ref1, pos21, wp.transpose(box2_rot) @ (box1_pos - box2_pos))
  rinc = wp.where(ref1, rot, wp.transpose(rot))
  sgn = float(feature[1])
  binc = feature[2] // 2
  tinc = float(2 * (feature[2] % 2) - 1)
  bu, bv = (binc + 1) % 3, (binc + 2) % 3
  cx = wp.vec3(
    posoi[ax] + tinc * sizeinc[binc] * rinc[ax, binc],
    posoi[ay] + tinc * sizeinc[binc] * rinc[ay, binc],
    sgn * (posoi[a] + tinc * sizeinc[binc] * rinc[a, binc]) - sizeref[a],
  )
  du = wp.vec3(sizeinc[bu] * rinc[ax, bu], sizeinc[bu] * rinc[ay, bu], sizeinc[bu] * rinc[a, bu] * sgn)
  dv = wp.vec3(sizeinc[bv] * rinc[ax, bv], sizeinc[bv] * rinc[ay, bv], sizeinc[bv] * rinc[a, bv] * sgn)

  if (candidate & 15) == 0:  # Intersection of two reference boundaries.
    x = wp.where((candidate & 16) != 0, sizeref[ax], -sizeref[ax])
    y = wp.where((candidate & 64) != 0, sizeref[ay], -sizeref[ay])
    det = du[0] * dv[1] - du[1] * dv[0]
    u = ((x - cx[0]) * dv[1] - (y - cx[1]) * dv[0]) / det
    v = (du[0] * (y - cx[1]) - du[1] * (x - cx[0])) / det
    point = wp.vec3(x, y, cx[2] + u * du[2] + v * dv[2])
  else:
    corner = int(0)
    for k in range(4):
      if (candidate & (1 << k)) != 0:
        corner = k
    if (candidate & 240) == 0:  # Original incident corner.
      for k in range(4):
        if candidate == ((1 << k) | (1 << ((k + 3) % 4))):
          corner = k
    su = wp.where(corner == 0 or corner == 3, 1.0, -1.0)
    sv = wp.where(corner < 2, 1.0, -1.0)
    p = cx + su * du + sv * dv
    point = p
    if (candidate & 240) != 0:  # Incident edge intersecting a reference boundary.
      next_corner = (corner + 1) % 4
      qu = wp.where(next_corner == 0 or next_corner == 3, 1.0, -1.0)
      qv = wp.where(next_corner < 2, 1.0, -1.0)
      q = cx + qu * du + qv * dv
      coord = wp.where((candidate & 48) != 0, 0, 1)
      sign = wp.where((candidate & (16 | 64)) != 0, 1.0, -1.0)
      limit = wp.where(coord == 0, sizeref[ax], sizeref[ay])
      dp, dq = sign * p[coord] - limit, sign * q[coord] - limit
      t = dp / (dp - dq)
      point = wp.vec3(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2]))

  height = sgn * (sizeref[a] + 0.5 * point[2])
  posc = wp.vec3(
    wp.where(a == 0, height, wp.where(ax == 0, point[0], point[1])),
    wp.where(a == 1, height, wp.where(ax == 1, point[0], point[1])),
    wp.where(a == 2, height, wp.where(ax == 2, point[0], point[1])),
  )
  normal = wp.where(ref1, sgn, -sgn) * wp.vec3(matref[0, a], matref[1, a], matref[2, a])
  return point[2], matref @ posc + posref, normal


# Recover discrete features from the primal contacts before differentiating their geometry.
@wp.kernel(enable_backward=False)
def _narrowphase_witness(
  # Model:
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
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
  # Out:
  contact_point_out: wp.array[wp.vec3],
  contact_feature_out: wp.array[wp.vec4i],
):
  """Writes the feature buffer consumed by each contact's replay branch."""
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  geoms = contact_geom_in[cid]
  g0, g1 = geoms[0], geoms[1]
  if g0 < 0 or g1 < 0:
    return
  type0, type1 = geom_type[g0], geom_type[g1]
  w = contact_worldid_in[cid]

  if type0 == GeomType.PLANE and type1 == GeomType.MESH:
    # Undo the contact midpoint offset to recover the selected vertex in mesh coordinates.
    normal = contact_frame_in[cid][0]
    vertex = contact_pos_in[cid] + 0.5 * contact_dist_in[cid] * normal
    contact_point_out[cid] = wp.transpose(geom_xmat_in[w, g1]) @ (vertex - geom_xpos_in[w, g1])
    return

  if type1 != GeomType.BOX or (type0 != GeomType.CAPSULE and type0 != GeomType.BOX):
    return
  gw = w % geom_size.shape[0]
  p0, p1 = geom_xpos_in[w, g0], geom_xpos_in[w, g1]
  m0, m1 = geom_xmat_in[w, g0], geom_xmat_in[w, g1]
  s0, s1 = geom_size[gw, g0], geom_size[gw, g1]
  if type0 == GeomType.CAPSULE:
    axis = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    _, feature = collision_primitive_core.capsule_box_witness(p0, axis, s0[1], p1, m1, s1)
    contact_feature_out[cid] = wp.vec4i(feature[0], feature[1], feature[2], 0)
  else:
    margin = wp.max(contact_includemargin_in[cid], contact_dist_in[cid])
    bb_count, bb_dist, bb_pos, bb_normal, candidates, feature = collision_primitive_core.box_box_witness(
      p0, m0, s0, p1, m1, s1, margin
    )
    # The forward margin includes gap contacts; filtering them can shift the contact slots.
    contact_pos = contact_pos_in[cid]
    slot = int(0)
    best = float(wp.inf)
    for i in range(bb_count):
      error = wp.length_sq(bb_pos[i] - contact_pos)
      if error < best:
        best = error
        slot = i
    contact_feature_out[cid] = wp.vec4i(feature[0], feature[1], feature[2], candidates[slot])


# Replay all thirteen primitive pairs with the primal contact slots and features fixed.
@wp.kernel(enable_backward=True)
def _narrowphase_recompute(
  # Model:
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  contact_geomcollisionid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  contact_point: wp.array[wp.vec3],  # selected mesh vertex in local coordinates
  contact_feature: wp.array[wp.vec4i],  # capsule/box feature and box clipping candidate
  # Out:
  cpos_out: wp.array[wp.vec3],
  dist_out: wp.array[float],
  frame_out: wp.array[wp.mat33],
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  geoms = contact_geom_in[cid]
  if geoms[0] < 0 or geoms[1] < 0:
    return
  w = contact_worldid_in[cid]
  slot = contact_geomcollisionid_in[cid]
  gw = w % geom_size.shape[0]
  g0 = geoms[0]
  g1 = geoms[1]
  p0 = geom_xpos_in[w, g0]
  m0 = geom_xmat_in[w, g0]
  s0 = geom_size[gw, g0]
  p1 = geom_xpos_in[w, g1]
  m1 = geom_xmat_in[w, g1]
  s1 = geom_size[gw, g1]
  type0 = geom_type[g0]
  type1 = geom_type[g1]
  if type0 == GeomType.PLANE and type1 == GeomType.SPHERE:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    dist, pos = collision_primitive_core.plane_sphere(n0, p0, p1, s1[0])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n0)
  elif type0 == GeomType.SPHERE and type1 == GeomType.SPHERE:
    dist, pos, n = collision_primitive_core.sphere_sphere(p0, s0[0], p1, s1[0])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.SPHERE and type1 == GeomType.BOX:
    dist, pos, n = collision_primitive_core.sphere_box(p0, s0[0], p1, m1, s1)
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.PLANE and type1 == GeomType.CAPSULE:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # capsule local z-axis
    dvec, pmat, frame = collision_primitive_core.plane_capsule(n0, p0, p1, axis, s1[0], s1[1])  # two caps, shared frame
    if slot == 0:
      dist_out[cid] = dvec[0]
      cpos_out[cid] = wp.vec3(pmat[0, 0], pmat[0, 1], pmat[0, 2])
    else:
      dist_out[cid] = dvec[1]
      cpos_out[cid] = wp.vec3(pmat[1, 0], pmat[1, 1], pmat[1, 2])
    frame_out[cid] = frame
  elif type0 == GeomType.SPHERE and type1 == GeomType.CAPSULE:
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # capsule local z-axis
    dist, pos, n = collision_primitive_core.sphere_capsule(p0, s0[0], p1, axis, s1[0], s1[1])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.SPHERE and type1 == GeomType.CYLINDER:
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # cylinder local z-axis
    dist, pos, n = collision_primitive_core.sphere_cylinder(p0, s0[0], p1, axis, s1[0], s1[1])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.PLANE and type1 == GeomType.ELLIPSOID:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    dist, pos, n = collision_primitive_core.plane_ellipsoid(n0, p0, p1, m1, s1)  # returns normal = plane normal
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.CAPSULE and type1 == GeomType.CAPSULE:
    axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2]) * s0[1]
    axis1 = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2]) * s1[1]
    a, b = p0 - axis0, p0 + axis0
    c, e = p1 - axis1, p1 + axis1
    _, weights0, weights1 = collision_primitive_core._segment_weights(a, b, c, e)
    weights = weights0 if slot == 0 else weights1
    # Replay one slot directly to avoid differentiating the forward's output component writes.
    dist, pos, n = collision_primitive_core.sphere_sphere(a + weights[0] * (b - a), s0[0], c + weights[1] * (e - c), s1[0])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif type0 == GeomType.PLANE and type1 == GeomType.CYLINDER:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])
    cyl_dist, cyl_pos, cyl_n = collision_primitive_core.plane_cylinder(
      n0, p0, p1, axis, s1[0], s1[1]
    )  # 4 contacts, shared normal
    for i in range(4):  # static unroll; runtime-compare select (no runtime indexing of the adjoint)
      if i == slot:
        dist_out[cid] = cyl_dist[i]
        cpos_out[cid] = wp.vec3(cyl_pos[i, 0], cyl_pos[i, 1], cyl_pos[i, 2])
    frame_out[cid] = math.make_frame(cyl_n)
  elif type0 == GeomType.PLANE and type1 == GeomType.BOX:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    box_dist, box_pos, box_n = collision_primitive_core.plane_box(
      n0, p0, p1, m1, s1
    )  # 8 corners (slot = corner id), shared normal
    for i in range(8):  # static unroll; runtime-compare select (no runtime indexing of the adjoint)
      if i == slot:
        dist_out[cid] = box_dist[i]
        cpos_out[cid] = wp.vec3(box_pos[i, 0], box_pos[i, 1], box_pos[i, 2])
    frame_out[cid] = math.make_frame(box_n)
  elif type0 == GeomType.CAPSULE and type1 == GeomType.BOX:
    axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    feature = contact_feature[cid]
    cb_dist, cb_pos, cb_n = _capsule_box_from_witness(
      p0, axis0, s0[0], s0[1], p1, m1, s1, slot, wp.vec3i(feature[0], feature[1], feature[2])
    )
    dist_out[cid] = cb_dist
    cpos_out[cid] = cb_pos
    frame_out[cid] = math.make_frame(cb_n)
  elif type0 == GeomType.BOX and type1 == GeomType.BOX:
    feature = contact_feature[cid]
    bb_dist, bb_pos, bb_n = _box_box_from_witness(
      p0, m0, s0, p1, m1, s1, wp.vec3i(feature[0], feature[1], feature[2]), feature[3]
    )
    dist_out[cid] = bb_dist
    cpos_out[cid] = bb_pos
    frame_out[cid] = math.make_frame(bb_n)
  elif type0 == GeomType.PLANE and type1 == GeomType.MESH:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    vertex = p1 + m1 @ contact_point[cid]
    dist, pos = collision_primitive_core.plane_sphere(n0, p0, vertex, 0.0)
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n0)


# gather dr/defc_pos (efc-row indexed) to the per-contact normal-row distance adjoint
@wp.kernel(enable_backward=False)
def _gather_efc_to_contact(
  # Data in:
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  res_efc_pos: wp.array2d[float],  # dr/defc_pos * lam (per world, per efc row)
  # Out:
  adj_dist_out: wp.array[float],  # per-contact seed for dist_out.grad (defc_pos/ddist = 1)
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  e0 = contact_efc_address_in[cid, 0]
  if e0 < 0:
    return
  adj_dist_out[cid] = res_efc_pos[contact_worldid_in[cid], e0]


# chain narrowphase geom-pose adjoints to the per-dof tangent gradient via support.jac_dof
@wp.kernel(enable_backward=False)
def _geom_pose_dof_vjp(
  # Model:
  ngeom: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  res_geom_xpos: wp.array2d[wp.vec3],  # adj(geom_xpos) from the narrowphase backward
  res_geom_xmat: wp.array2d[wp.mat33],  # adj(geom_xmat)
  # Out:
  res_dof_out: wp.array2d[float],  # per-dof tangent gradient d(contact)/d(dof k)
):
  w, k = wp.tid()
  acc = float(0.0)
  for g in range(ngeom):
    rgp = res_geom_xpos[w, g]
    rgm = res_geom_xmat[w, g]
    body = geom_bodyid[g]
    jacp, jacr = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      geom_xpos_in[w, g],
      body,
      k,
      w,
    )
    tau = adjoint_util._adj_rotation(geom_xmat_in[w, g], rgm)
    acc += wp.dot(jacp, rgp) + wp.dot(jacr, tau)
  res_dof_out[w, k] += acc


@event_scope
def contact_qpos_vjp(
  m: Model,
  d_out: Data,
  qpos_in: wp.array2d[float],
  res_contact_pos: wp.array,
  res_contact_frame: wp.array,
  res_efc_pos: wp.array2d[float],
  res_subtree_com: wp.array2d[wp.vec3],
  res_cdof: wp.array2d[wp.spatial_vector],
  res_qpos: wp.array2d[float],
  bc: BackwardContext | None = None,
):
  """Accumulates the contact residual's dqpos into res_qpos from the exposed input-adjoints.

  Primitive narrowphase replay yields adj(geom poses); analytic Jacobian VJPs
  chain these into one per-dof tangent buffer that _dof_to_qpos lifts.
  Box-box requires the primitive collider (nativeccd disabled).

  Args:
    m: The model.
    d_out: The step-output data holding the frozen contacts and kinematics.
    qpos_in: Input/linearization qpos (d.qpos), not the integrated d_out.qpos.
    res_contact_pos: dr/dcontact_pos * lam, per-contact vec3.
    res_contact_frame: dr/dcontact_frame * lam, per-contact mat33.
    res_efc_pos: dr/defc_pos * lam, (nworld, njmax).
    res_subtree_com: dr/dsubtree_com * lam, (nworld, nbody).
    res_cdof: dr/dcdof * lam, (nworld, nv).
    res_qpos: Accumulated output; the caller's writeback applies the IFT minus.
    bc: Reusable backward workspace.
  """
  box_pair = math.upper_trid_index(len(GeomType), GeomType.BOX.value, GeomType.BOX.value)
  if m.geom_pair_type_count[box_pair] and not (m.opt.disableflags & DisableBit.NATIVECCD):
    raise NotImplementedError("box-box contact gradients require nativeccd disabled; convex box-box replay is not supported")
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d_out)
  nworld = d_out.qpos.shape[0]
  nv = m.nv
  nconmax = d_out.contact.pos.shape[0]
  contact_point, contact_feature = bc.contact_point, bc.contact_feature
  contact_adj_dist = bc.contact_adj_dist
  res_geom_xpos, res_geom_xmat = bc.res_geom_xpos, bc.res_geom_xmat
  contact_res_dof, contact_ceff = bc.contact_res_dof, bc.contact_ceff
  # _narrowphase_witness overwrites every feature consumed by _narrowphase_recompute.
  contact_adj_dist.zero_()
  res_geom_xpos.zero_()
  res_geom_xmat.zero_()
  contact_res_dof.zero_()
  contact_ceff.zero_()

  wp.launch(
    _narrowphase_witness,
    dim=nconmax,
    inputs=[
      m.geom_type,
      m.geom_size,
      d_out.geom_xpos,
      d_out.geom_xmat,
      d_out.contact.dist,
      d_out.contact.pos,
      d_out.contact.frame,
      d_out.contact.includemargin,
      d_out.contact.geom,
      d_out.contact.worldid,
      d_out.nacon,
    ],
    outputs=[contact_point, contact_feature],
  )
  pairs = [
    (m.geom_type, None),
    (m.geom_size, None),
    (d_out.geom_xpos, res_geom_xpos),
    (d_out.geom_xmat, res_geom_xmat),
    (d_out.contact.geom, None),
    (d_out.contact.worldid, None),
    (d_out.contact.geomcollisionid, None),
    (d_out.nacon, None),
    (contact_point, None),
    (contact_feature, None),
  ]
  wp.launch(
    _gather_efc_to_contact,
    dim=nconmax,
    inputs=[d_out.contact.efc_address, d_out.contact.worldid, d_out.nacon, res_efc_pos],
    outputs=[contact_adj_dist],
  )
  adjoint_util._launch_vjp(
    _narrowphase_recompute,
    nconmax,
    pairs,
    [d_out.contact.pos, d_out.contact.dist, d_out.contact.frame],
    [res_contact_pos, contact_adj_dist, res_contact_frame],
  )
  wp.launch(
    _geom_pose_dof_vjp,
    dim=(nworld, nv),
    inputs=[
      m.ngeom,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      m.geom_bodyid,
      m.body_isdofancestor,
      d_out.geom_xpos,
      d_out.geom_xmat,
      d_out.subtree_com,
      d_out.cdof,
      res_geom_xpos,
      res_geom_xmat,
    ],
    outputs=[contact_res_dof],
  )
  smooth_kinematics_adjoint.kinematics_qpos_backward(
    m,
    d_out,
    qpos_in,
    res_cdof,
    res_subtree_com,
    contact_res_dof,
    contact_ceff,
    res_qpos,
  )
