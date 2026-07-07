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
"""Contact dqpos VJP for the analytic IFT backward (adjoint.py)."""

from typing import Tuple

import warp as wp

from mujoco_warp._src import collision_primitive_core
from mujoco_warp._src import math
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.warp_util import event_scope

# Adjoint module: backward stays ON so AD leaves differentiate through cross-module @wp.funcs.
wp.set_module_options({"enable_backward": True})

_FREE = int(types.JointType.FREE.value)
_BALL = int(types.JointType.BALL.value)
_PLANE = int(types.GeomType.PLANE.value)
_SPHERE = int(types.GeomType.SPHERE.value)
_CAPSULE = int(types.GeomType.CAPSULE.value)
_ELLIPSOID = int(types.GeomType.ELLIPSOID.value)
_CYLINDER = int(types.GeomType.CYLINDER.value)
_BOX = int(types.GeomType.BOX.value)


@wp.func
def _capsule_box_segpos(
  # In:
  pos_c: wp.vec3,
  axis_c: wp.vec3,
  half_len: float,
  pos_b: wp.vec3,
  mat_b: wp.mat33,
  size_b: wp.vec3,
) -> Tuple[wp.vec2, wp.vec3i]:
  """Forward-only closest-feature re-run: returns segment params (t1, t2) and frozen feature state.

  Mirrors collision_primitive_core.capsule_box (keep in sync); not AD-safe, backward-disabled only.
  """
  boxmatT = wp.transpose(mat_b)
  pos = boxmatT @ (pos_c - pos_b)
  axis = boxmatT @ axis_c
  halfaxis = axis * half_len
  axisdir = wp.int32(halfaxis[0] > 0.0) + 2 * wp.int32(halfaxis[1] > 0.0) + 4 * wp.int32(halfaxis[2] > 0.0)

  bestdist = wp.float32(1.0e32)
  bestsegmentpos = wp.float32(-12)
  cltype = wp.int32(-4)
  clface = wp.int32(-12)

  # first: cases where a face of the box is closest to a capsule tip
  for i in range(-1, 2, 2):
    axisTip = pos + wp.float32(i) * halfaxis
    boxPoint = wp.vec3(axisTip)
    n_out = wp.int32(0)
    ax_out = wp.int32(-1)
    for j in range(3):
      if boxPoint[j] < -size_b[j]:
        n_out += 1
        ax_out = j
        boxPoint[j] = -size_b[j]
      elif boxPoint[j] > size_b[j]:
        n_out += 1
        ax_out = j
        boxPoint[j] = size_b[j]
    if n_out > 1:
      continue
    dist = wp.length_sq(boxPoint - axisTip)
    if dist < bestdist:
      bestdist = dist
      bestsegmentpos = wp.float32(i)
      cltype = -2 + i
      clface = ax_out

  # second: cases where an edge of the box is closest
  clcorner = wp.int32(-123)
  cledge = wp.int32(-123)
  bestboxpos = wp.float32(0.0)
  for i in range(8):
    for j in range(3):
      if i & (1 << j) != 0:
        continue
      c2 = wp.int32(-123)
      box_pt = wp.cw_mul(
        wp.vec3(
          wp.where(i & 1, 1.0, -1.0),
          wp.where(i & 2, 1.0, -1.0),
          wp.where(i & 4, 1.0, -1.0),
        ),
        size_b,
      )
      box_pt[j] = 0.0
      dif = box_pt - pos
      u = -size_b[j] * dif[j]
      v = wp.dot(halfaxis, dif)
      ma = size_b[j] * size_b[j]
      mb = -size_b[j] * halfaxis[j]
      mc = half_len * half_len
      det = ma * mc - mb * mb
      if wp.abs(det) < MJ_MINVAL:
        continue
      idet = 1.0 / det
      x1 = wp.float32((mc * u - mb * v) * idet)
      x2 = wp.float32((ma * v - mb * u) * idet)
      s1 = wp.int32(1)
      s2 = wp.int32(1)
      if x1 > 1:
        x1 = 1.0
        s1 = 2
        x2 = math.safe_div(v - mb, mc)
      elif x1 < -1:
        x1 = -1.0
        s1 = 0
        x2 = math.safe_div(v + mb, mc)
      x2_over = x2 > 1.0
      if x2_over or x2 < -1.0:
        if x2_over:
          x2 = 1.0
          s2 = 2
          x1 = math.safe_div(u - mb, ma)
        else:
          x2 = -1.0
          s2 = 0
          x1 = math.safe_div(u + mb, ma)
        if x1 > 1:
          x1 = 1.0
          s1 = 2
        elif x1 < -1:
          x1 = -1.0
          s1 = 0
      dif -= halfaxis * x2
      dif[j] += size_b[j] * x1
      ct = s1 * 3 + s2
      dif_sq = wp.length_sq(dif)
      if dif_sq < bestdist - MJ_MINVAL:
        bestdist = dif_sq
        bestsegmentpos = x2
        bestboxpos = x1
        c2 = ct // 6
        clcorner = i + (1 << j) * c2
        cledge = j
        cltype = ct

  if cltype == -4:  # no valid configuration -> no contact was created by the forward
    return wp.vec2(0.0, 0.0), wp.vec3i(-4, 0, 0)

  secondpos = wp.float32(-4.0)
  if cltype >= 0 and cltype // 3 != 1:  # closest to a corner of the box
    c1 = axisdir ^ clcorner
    if c1 != 0 and c1 != 7:
      mul = wp.int32(1)
      if not (c1 == 1 or c1 == 2 or c1 == 4):
        mul = -1
        c1 = 7 - c1
      ax = wp.int32(0)
      ax1 = wp.int32(1)
      ax2 = wp.int32(2)
      if c1 == 2:
        ax = 1
        ax1 = 2
        ax2 = 0
      elif c1 == 4:
        ax = 2
        ax1 = 0
        ax2 = 1
      if axis[ax] * axis[ax] > 0.5:  # second point along the edge of the box
        m = 2.0 * math.safe_div(size_b[ax], wp.abs(halfaxis[ax]))
        secondpos = min(1.0 - wp.float32(mul) * bestsegmentpos, m)
      else:  # second point along a face of the box
        m = 2.0 * min(
          math.safe_div(size_b[ax1], wp.abs(halfaxis[ax1])),
          math.safe_div(size_b[ax2], wp.abs(halfaxis[ax2])),
        )
        secondpos = -min(1.0 + wp.float32(mul) * bestsegmentpos, m)
      secondpos *= wp.float32(mul)
  elif cltype >= 0 and cltype // 3 == 1:  # closest to a box edge
    c1 = axisdir ^ clcorner
    c1 &= 7 - (1 << cledge)
    if c1 == 1 or c1 == 2 or c1 == 4:
      ax1 = wp.int32(1)
      ax2 = wp.int32(2)
      if cledge == 1:
        ax1 = 2
        ax2 = 0
      if cledge == 2:
        ax1 = 0
        ax2 = 1
      ax = cledge
      if wp.abs(axis[ax1]) > wp.abs(axis[ax2]):
        ax1 = ax2
      ax2 = 3 - ax - ax1
      mul = wp.int32(1)
      if c1 & (1 << ax2):
        secondpos = 1.0 - bestsegmentpos
      else:
        mul = -1
        secondpos = 1.0 + bestsegmentpos
      e1 = 2.0 * math.safe_div(size_b[ax2], wp.abs(halfaxis[ax2]))
      secondpos = min(e1, secondpos)
      if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):
        e2 = 1.0 - bestboxpos
      else:
        e2 = 1.0 + bestboxpos
      e1 = size_b[ax] * math.safe_div(e2, wp.abs(halfaxis[ax]))
      secondpos = min(e1, secondpos)
      secondpos *= wp.float32(mul)
  else:  # a capsule tip is closest to a box face
    if clface != -1:
      mul = wp.where(cltype == -3, 1, -1)
      secondpos = wp.float32(2.0)
      tmp1 = pos - halfaxis * wp.float32(mul)
      for i in range(3):
        if i != clface:
          ha_r = math.safe_div(wp.float32(mul), halfaxis[i])
          e1 = (size_b[i] - tmp1[i]) * ha_r
          if 0 < e1 and e1 < secondpos:
            secondpos = e1
          e1 = (-size_b[i] - tmp1[i]) * ha_r
          if 0 < e1 and e1 < secondpos:
            secondpos = e1
      secondpos *= wp.float32(mul)

  t2 = bestsegmentpos
  if secondpos > -3.0:
    t2 = bestsegmentpos + secondpos
  return wp.vec2(bestsegmentpos, t2), wp.vec3i(cltype, wp.max(clcorner, 0), wp.max(cledge, 0))


@wp.func
def _capsule_box_t0(
  # In:
  pos_bf: wp.vec3,  # capsule center in the box frame
  halfaxis_bf: wp.vec3,  # capsule half-axis in the box frame
  half_len: float,
  size_b: wp.vec3,
  cltype: int,  # frozen discrete feature state (from _capsule_box_segpos)
  clcorner: int,
  cledge: int,
) -> float:
  """Differentiably re-derive segment param t1 with the branch/clamp feature state frozen."""
  if cltype < 0:  # capsule TIP closest to a face: t is the constant tip
    return wp.where(cltype == -3, -1.0, 1.0)
  s1 = cltype // 3
  s2 = cltype - 3 * s1
  if s2 == 0:  # segment end clamped low
    return -1.0
  if s2 == 2:  # segment end clamped high
    return 1.0
  # rebuild the frozen edge's quantities (forward's edge loop body) via compare-selects
  bx = wp.where(cledge == 0, 0.0, wp.where((clcorner & 1) != 0, size_b[0], -size_b[0]))
  by = wp.where(cledge == 1, 0.0, wp.where((clcorner & 2) != 0, size_b[1], -size_b[1]))
  bz = wp.where(cledge == 2, 0.0, wp.where((clcorner & 4) != 0, size_b[2], -size_b[2]))
  dif = wp.vec3(bx, by, bz) - pos_bf
  sj = wp.where(cledge == 0, size_b[0], wp.where(cledge == 1, size_b[1], size_b[2]))
  dj = wp.where(cledge == 0, dif[0], wp.where(cledge == 1, dif[1], dif[2]))
  hj = wp.where(cledge == 0, halfaxis_bf[0], wp.where(cledge == 1, halfaxis_bf[1], halfaxis_bf[2]))
  u = -sj * dj
  v = wp.dot(halfaxis_bf, dif)
  ma = sj * sj
  mb = -sj * hj
  mc = half_len * half_len
  if s1 == 1:  # interior-interior: the 2x2 solve
    return math.safe_div(ma * v - mb * u, ma * mc - mb * mb)
  if s1 == 2:  # box-edge param clamped high
    return math.safe_div(v - mb, mc)
  return math.safe_div(v + mb, mc)  # s1 == 0: box-edge param clamped low


@wp.kernel(enable_backward=False)
def _capsule_box_freeze(
  # Model:
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # Out:
  tseg_out: wp.array[wp.vec2],
  feat_out: wp.array[wp.vec3i],
):
  """Store per capsule-box contact the frozen sphere-reduction segment params and feature state."""
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  geoms = contact_geom_in[cid]
  if geoms[0] < 0 or geoms[1] < 0:
    return
  g0 = geoms[0]
  g1 = geoms[1]
  if geom_type[g0] != _CAPSULE or geom_type[g1] != _BOX:
    return
  w = contact_worldid_in[cid]
  gw = w % geom_size.shape[0]
  m0 = geom_xmat_in[w, g0]
  axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
  ts, ft = _capsule_box_segpos(
    geom_xpos_in[w, g0], axis0, geom_size[gw, g0][1], geom_xpos_in[w, g1], geom_xmat_in[w, g1], geom_size[gw, g1]
  )
  tseg_out[cid] = ts
  feat_out[cid] = ft


# coverage of the dispatch chain in _narrowphase_recompute: 11 of the 13 _PRIMITIVE_COLLISIONS pairs
# are AD-safe and dispatched -- plane-{sphere,capsule,ellipsoid,cylinder,box}, sphere-{sphere,box,
# capsule,cylinder}, capsule-capsule, capsule-box. the remaining 2 pairs are not AD-safe and fall
# through -> dqpos stays silently 0 for that pair: box-box, plane-convex/mesh.
# capsule-box: _capsule_box_freeze stores the frozen segment/feature; the backward diffs sphere_box
# at the frozen t -- first-order exact for the minimizing slot-0 contact (envelope theorem).
# capsule-capsule: the parallel slot-1 assignment is margin-dependent (gated to the non-parallel,
# crossed slot-0 regime). the FD oracle in collision_adjoint_test.py gates every dispatched pair.
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
  capsule_box_tseg: wp.array[wp.vec2],  # frozen capsule-box segment params (_capsule_box_freeze)
  capsule_box_feat: wp.array[wp.vec3i],  # frozen capsule-box discrete feature state
  # Out:
  cpos_out: wp.array[wp.vec3],
  dist_out: wp.array[float],
  frame_out: wp.array[wp.mat33],
):
  """Frozen-witness narrowphase replay, backward-enabled so Warp auto-diffs adj(geom poses).

  Reuses the collision_primitive_core leaves (keep in sync); a Geom struct arg zeroes the adjoint.
  """
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
  t0 = geom_type[g0]
  t1 = geom_type[g1]
  if t0 == _PLANE and t1 == _SPHERE:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    dist, pos = collision_primitive_core.plane_sphere(n0, p0, p1, s1[0])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n0)
  elif t0 == _SPHERE and t1 == _SPHERE:
    dist, pos, n = collision_primitive_core.sphere_sphere(p0, s0[0], p1, s1[0])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif t0 == _SPHERE and t1 == _BOX:
    dist, pos, n = collision_primitive_core.sphere_box(p0, s0[0], p1, m1, s1)
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif t0 == _PLANE and t1 == _CAPSULE:
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
  elif t0 == _SPHERE and t1 == _CAPSULE:
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # capsule local z-axis
    dist, pos, n = collision_primitive_core.sphere_capsule(p0, s0[0], p1, axis, s1[0], s1[1])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif t0 == _SPHERE and t1 == _CYLINDER:
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # cylinder local z-axis
    dist, pos, n = collision_primitive_core.sphere_cylinder(p0, s0[0], p1, axis, s1[0], s1[1])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif t0 == _PLANE and t1 == _ELLIPSOID:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    dist, pos, n = collision_primitive_core.plane_ellipsoid(n0, p0, p1, m1, s1)  # returns normal = plane normal
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = math.make_frame(n)
  elif t0 == _CAPSULE and t1 == _CAPSULE:
    # unique local names per multi-contact branch: Warp codegen scopes locals to the whole
    # function, so reusing one name with a different vec/mat type across branches is a
    # type-conflict error.
    axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    axis1 = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])
    # margin only gates write_contact's slot assignment in the forward; pass a large value so the
    # (frozen) active slot is always populated. non-parallel (crossed) axes -> slot 0; the
    # parallel slot-1 assignment is margin-dependent (see the coverage comment above).
    cc_dist, cc_pos, cc_nrm = collision_primitive_core.capsule_capsule(p0, axis0, s0[0], s0[1], p1, axis1, s1[0], s1[1], 1.0e6)
    for i in range(2):  # static unroll; runtime-compare select (no runtime indexing of the adjoint)
      if i == slot:
        dist_out[cid] = cc_dist[i]
        cpos_out[cid] = wp.vec3(cc_pos[i, 0], cc_pos[i, 1], cc_pos[i, 2])
        frame_out[cid] = math.make_frame(wp.vec3(cc_nrm[i, 0], cc_nrm[i, 1], cc_nrm[i, 2]))  # per-slot normal
  elif t0 == _PLANE and t1 == _CYLINDER:
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
  elif t0 == _PLANE and t1 == _BOX:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    box_dist, box_pos, box_n = collision_primitive_core.plane_box(
      n0, p0, p1, m1, s1
    )  # 8 corners (slot = corner id), shared normal
    for i in range(8):  # static unroll; runtime-compare select (no runtime indexing of the adjoint)
      if i == slot:
        dist_out[cid] = box_dist[i]
        cpos_out[cid] = wp.vec3(box_pos[i, 0], box_pos[i, 1], box_pos[i, 2])
    frame_out[cid] = math.make_frame(box_n)
  elif t0 == _CAPSULE and t1 == _BOX:
    # frozen-feature witness: capsule-box reduces to sphere_box at 1-2 capsule-segment points; the
    # feature search is not AD-safe, so _capsule_box_freeze froze the discrete state and t1 is
    # re-derived differentiably for that feature (_capsule_box_t0). the slot-1 spread point rides
    # t1 with its offset frozen (same approximation class as capsule_capsule's parallel slot-1).
    axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    ts = capsule_box_tseg[cid]
    ft = capsule_box_feat[cid]
    bmT = wp.transpose(m1)
    pos_bf = bmT @ (p0 - p1)
    halfaxis_bf = (bmT @ axis0) * s0[1]
    t0d = _capsule_box_t0(pos_bf, halfaxis_bf, s0[1], s1, ft[0], ft[1], ft[2])
    tk = wp.where(slot == 0, t0d, t0d + (ts[1] - ts[0]))
    cb_center = p0 + axis0 * (s0[1] * tk)
    cb_dist, cb_pos, cb_n = collision_primitive_core.sphere_box(cb_center, s0[0], p1, m1, s1)
    dist_out[cid] = cb_dist
    cpos_out[cid] = cb_pos
    frame_out[cid] = math.make_frame(cb_n)


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
  """Gather dr/defc_pos (efc-row indexed) to the per-contact normal-row distance adjoint."""
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  e0 = contact_efc_address_in[cid, 0]
  if e0 < 0:
    return
  adj_dist_out[cid] = res_efc_pos[contact_worldid_in[cid], e0]


@wp.kernel(enable_backward=False)
def _cdof_qpos_vjp(
  # Model:
  nv: int,
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  dof_jntid: wp.array[int],
  dof_parentid: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  res_cdof: wp.array2d[wp.spatial_vector],  # dr/dcdof * lam (per dof)
  # Out:
  res_dof_out: wp.array2d[float],  # OUT: += per-dof tangent gradient dc/d(dof k)
):
  """Fixed-reference part of dr/dcdof -> qpos: the screw-axis commutator, as a manual VJP.

  The moving-COM reference term is NOT here -- _build_ceff folds it into the subtree-COM seed.
  """
  w, k = wp.tid()
  cdof_k = cdof_in[w, k]
  jk = dof_jntid[k]
  acc = float(0.0)
  for i in range(nv):
    ji = dof_jntid[i]
    jti = jnt_type[ji]
    ofs = jnt_dofadr[ji]
    if jti == _FREE and (i - ofs) < 3:
      continue  # free-translation target row: d(stored cdof_i)/dq = 0
    chi = bool(False)
    p = dof_parentid[i]  # (a) transitive dof_parentid ancestry: is k a predecessor of i?
    while p >= 0:
      if p == k:
        chi = True
        break
      p = dof_parentid[p]
    if (not chi) and ji == jk:  # (b)/(c) full same-BALL / same-FREE-rotation angular block
      if jti == _BALL:
        chi = True
      elif jti == _FREE and (k - ofs) >= 3:
        chi = True
    if chi:
      acc += wp.dot(res_cdof[w, i], math.motion_cross(cdof_k, cdof_in[w, i]))
  res_dof_out[w, k] += acc


@wp.kernel(enable_backward=False)
def _build_ceff(
  # Model:
  nv: int,
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  res_cdof: wp.array2d[wp.spatial_vector],
  res_subtree_com: wp.array2d[wp.vec3],
  # Out:
  ceff_out: wp.array2d[wp.vec3],  # OUT: effective subtree-COM cotangent
):
  """Build the effective subtree-COM seed: res_subtree_com + the moving-COM part of dr/dcdof."""
  w, b = wp.tid()
  c = res_subtree_com[w, b]
  if body_rootid[b] == b and b > 0:  # tree root -> accumulate the rotational rows' moving-COM cotangent
    for i in range(nv):
      if body_rootid[dof_bodyid[i]] == b:
        ri = res_cdof[w, i]
        c += wp.cross(wp.spatial_bottom(ri), wp.spatial_top(cdof_in[w, i]))
  ceff_out[w, b] = c


@wp.kernel(enable_backward=False)
def _subtree_com_qpos_vjp(
  # Model:
  nbody: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_subtreemass: wp.array2d[float],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  res_subtree_com: wp.array2d[wp.vec3],  # dr/dsubtree_com * lam
  # Out:
  res_dof_out: wp.array2d[float],  # OUT: += per-dof tangent gradient dc/d(dof k)
):
  """Chain dr/dsubtree_com to qpos via the mass-weighted subtree-COM Jacobian (mj_jacSubtreeCom)."""
  w, k = wp.tid()
  wm = w % body_mass.shape[0]
  acc = float(0.0)
  for c in range(1, nbody):
    if body_isdofancestor[c, k] == 0:  # dof k does not move body c -> jacp = 0
      continue
    r = body_rootid[c]
    jp, _jr = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      xipos_in[w, c],
      c,
      k,
      w,
    )
    acc += (body_mass[wm, c] / body_subtreemass[w % body_subtreemass.shape[0], r]) * wp.dot(jp, res_subtree_com[w, r])
  res_dof_out[w, k] += acc


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
  res_dof_out: wp.array2d[float],  # OUT: per-dof tangent gradient d(contact)/d(dof k)
):
  """Chain narrowphase geom-pose adjoints to the per-dof tangent gradient via support.jac_dof."""
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
    xm = geom_xmat_in[w, g]
    tau = (
      wp.cross(wp.vec3(xm[0, 0], xm[1, 0], xm[2, 0]), wp.vec3(rgm[0, 0], rgm[1, 0], rgm[2, 0]))
      + wp.cross(wp.vec3(xm[0, 1], xm[1, 1], xm[2, 1]), wp.vec3(rgm[0, 1], rgm[1, 1], rgm[2, 1]))
      + wp.cross(wp.vec3(xm[0, 2], xm[1, 2], xm[2, 2]), wp.vec3(rgm[0, 2], rgm[1, 2], rgm[2, 2]))
    )
    acc += wp.dot(jacp, rgp) + wp.dot(jacr, tau)
  res_dof_out[w, k] += acc


@wp.kernel(enable_backward=False)
def _dof_to_qpos(
  # Model:
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  res_dof: wp.array2d[float],  # per-dof tangent gradient d(contact)/d(dof)
  # Out:
  res_qpos_out: wp.array2d[float],  # += dof -> qpos (1:1 for slide/hinge/free-translation; quaternion lift for free/ball)
):
  """Map the per-dof tangent gradient res_dof to res_qpos_out (nq-indexed; 1:1 except quaternions).

  Free/ball angular gradient g lifts to raw quaternion coords as dc/dq = 2*quat_mul(q, [0, g]).
  """
  w, j = wp.tid()
  jt = jnt_type[j]
  qadr = jnt_qposadr[j]
  dadr = jnt_dofadr[j]
  if jt == _FREE or jt == _BALL:
    if jt == _FREE:  # translation dofs are 1:1; the quaternion starts at qadr+3, angular dofs at dadr+3
      res_qpos_out[w, qadr + 0] += res_dof[w, dadr + 0]
      res_qpos_out[w, qadr + 1] += res_dof[w, dadr + 1]
      res_qpos_out[w, qadr + 2] += res_dof[w, dadr + 2]
      qadr_q = qadr + 3
      dadr_a = dadr + 3
    else:  # BALL: 4 qpos (quaternion), 3 dofs, both at the joint adr
      qadr_q = qadr
      dadr_a = dadr
    q = wp.quat(qpos_in[w, qadr_q + 0], qpos_in[w, qadr_q + 1], qpos_in[w, qadr_q + 2], qpos_in[w, qadr_q + 3])
    g = wp.vec3(res_dof[w, dadr_a + 0], res_dof[w, dadr_a + 1], res_dof[w, dadr_a + 2])
    dq = 2.0 * math.quat_mul_axis(q, g)  # 2 * quat_mul(q, [0, g]): lift tangent angular -> raw quaternion gradient
    res_qpos_out[w, qadr_q + 0] += dq[0]
    res_qpos_out[w, qadr_q + 1] += dq[1]
    res_qpos_out[w, qadr_q + 2] += dq[2]
    res_qpos_out[w, qadr_q + 3] += dq[3]
  else:  # HINGE / SLIDE (nq == nv, qpos coord == dof)
    res_qpos_out[w, qadr] += res_dof[w, dadr]


@event_scope
def contact_qpos_vjp(
  m: Model,
  d_out: Data,
  qpos_in: wp.array2d[float],  # input/linearization qpos (d.qpos), not integrated d_out.qpos
  res_contact_pos: wp.array,  # dr/dcontact_pos * lam (per-contact vec3)
  res_contact_frame: wp.array,  # dr/dcontact_frame * lam (per-contact mat33)
  res_efc_pos: wp.array2d[float],  # dr/defc_pos * lam (nworld, njmax)
  res_subtree_com: wp.array2d[wp.vec3],  # dr/dsubtree_com * lam (nworld, nbody)
  res_cdof: wp.array2d[wp.spatial_vector],  # dr/dcdof * lam (nworld, nv)
  res_qpos: wp.array2d[float],  # OUT: += -(dr_contact/dqpos)^T lam-worth of the contraction (sign per _sub_write)
):
  """Accumulate the contact residual's dqpos into ``res_qpos`` from the exposed input-adjoints.

  A frozen-witness narrowphase replay yields adj(geom poses); analytic Jacobian VJPs then chain
  everything into one per-dof tangent buffer that _dof_to_qpos lifts. Sign: _sub_write subtracts.
  """
  nworld = d_out.qpos.shape[0]
  nv = m.nv
  ncon_max = d_out.contact.pos.shape[0]
  for _arr in (d_out.geom_xpos, d_out.geom_xmat):
    _arr.requires_grad = True  # so the narrowphase backward accumulates their input-adjoints

  cpos_o = wp.zeros(ncon_max, dtype=wp.vec3, requires_grad=True)
  dist_o = wp.zeros(ncon_max, dtype=float, requires_grad=True)
  frame_o = wp.zeros(ncon_max, dtype=wp.mat33, requires_grad=True)
  # frozen capsule-box segment params + feature state (forward-only; see _capsule_box_freeze)
  cb_tseg = wp.zeros(ncon_max, dtype=wp.vec2)
  cb_feat = wp.zeros(ncon_max, dtype=wp.vec3i)
  wp.launch(
    _capsule_box_freeze,
    dim=ncon_max,
    inputs=[m.geom_type, m.geom_size, d_out.geom_xpos, d_out.geom_xmat, d_out.contact.geom, d_out.contact.worldid, d_out.nacon],
    outputs=[cb_tseg, cb_feat],
  )
  res_geom_xpos = wp.zeros_like(d_out.geom_xpos)
  res_geom_xmat = wp.zeros_like(d_out.geom_xmat)
  # (input, input-adjoint) pairs; only geom poses carry adjoints (Warp auto-diffs narrowphase).
  np_pairs = [
    (m.geom_type, None),
    (m.geom_size, None),
    (d_out.geom_xpos, res_geom_xpos),  # geom_xpos_in
    (d_out.geom_xmat, res_geom_xmat),  # geom_xmat_in
    (d_out.contact.geom, None),
    (d_out.contact.worldid, None),
    (d_out.contact.geomcollisionid, None),
    (d_out.nacon, None),
    (cb_tseg, None),
    (cb_feat, None),
  ]
  np_in = [a for a, _ in np_pairs]
  wp.launch(_narrowphase_recompute, dim=ncon_max, inputs=np_in, outputs=[cpos_o, dist_o, frame_o])
  # seed the geometry adjoints: contact_pos/frame are per-contact direct; efc_pos is
  # efc-row-indexed -> gather to per-contact (defc_pos/ddist = 1).
  wp.copy(cpos_o.grad, res_contact_pos)
  wp.copy(frame_o.grad, res_contact_frame)
  wp.launch(
    _gather_efc_to_contact,
    dim=ncon_max,
    inputs=[d_out.contact.efc_address, d_out.contact.worldid, d_out.nacon, res_efc_pos],
    outputs=[dist_o.grad],
  )
  wp.launch(
    _narrowphase_recompute,
    dim=ncon_max,
    inputs=np_in,
    outputs=[cpos_o, dist_o, frame_o],
    adj_inputs=[g for _, g in np_pairs],
    adj_outputs=[cpos_o.grad, dist_o.grad, frame_o.grad],
    adjoint=True,
  )
  # all dqpos terms accumulate into one per-dof tangent buffer res_dof, lifted once to qpos by
  # _dof_to_qpos: (1) geom-pose adjoints via jac_dof; (2) the fixed-COM screw commutator
  # (_cdof_qpos_vjp); (3) the COM-Jacobian VJP (_subtree_com_qpos_vjp) seeded by _build_ceff, which
  # folds the moving-COM reference part of dr/dcdof so one COM pass carries both.
  res_dof = wp.zeros((nworld, nv), dtype=float)
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
    outputs=[res_dof],
  )
  wp.launch(
    _cdof_qpos_vjp,
    dim=(nworld, nv),
    inputs=[nv, m.jnt_type, m.jnt_dofadr, m.dof_jntid, m.dof_parentid, d_out.cdof, res_cdof],
    outputs=[res_dof],
  )
  ceff = wp.empty_like(res_subtree_com)
  wp.launch(
    _build_ceff,
    dim=(nworld, m.nbody),
    inputs=[nv, m.body_rootid, m.dof_bodyid, d_out.cdof, res_cdof, res_subtree_com],
    outputs=[ceff],
  )
  wp.launch(
    _subtree_com_qpos_vjp,
    dim=(nworld, nv),
    inputs=[
      m.nbody,
      m.body_parentid,
      m.body_rootid,
      m.body_mass,
      m.body_subtreemass,
      m.dof_bodyid,
      m.body_isdofancestor,
      d_out.xipos,
      d_out.subtree_com,
      d_out.cdof,
      ceff,
    ],
    outputs=[res_dof],
  )
  wp.launch(
    _dof_to_qpos,
    dim=(nworld, m.njnt),
    inputs=[
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      qpos_in,
      res_dof,
    ],
    outputs=[res_qpos],
  )
