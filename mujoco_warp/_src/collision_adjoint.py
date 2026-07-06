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
"""Contact ``∂qpos`` VJP for the analytic IFT backward (``adjoint.py``).

``adjoint.py``'s residual-VJP exposes Warp's input-adjoints on the contact residual's frozen kinematic
intermediates -- ``∂r/∂{contact_pos, contact_frame, efc_pos, subtree_com, cdof}·λ`` (the elliptic cone
curvature is auto-diffed since they are kernel inputs). This module turns those into ``∂r/∂qpos`` WITHOUT
AD-ing the kinematics tree (the Warp dynamic-loop bug zone), by mirroring the FORWARD narrowphase:

  1. ``_narrowphase_recompute`` -- per frozen contact, recompute (dist, pos, frame) by calling the FORWARD
     pure geom funcs (``collision_primitive_core.{plane_sphere, sphere_sphere, sphere_box, plane_capsule,
     sphere_capsule, sphere_cylinder, plane_ellipsoid, capsule_capsule, plane_cylinder, plane_box}`` -- the
     same leaves the forward ``*_wrapper``s use). Backward-enabled, so Warp AUTO-DIFFS the geometry through
     them, giving ``adj(geom_xpos, geom_xmat)`` seeded by the residual's exposed adjoints. We read RAW geom
     world poses, NOT the forward's ``Geom`` wp.struct: a struct returned from a @wp.func zeroes Warp
     reverse-mode (1.14.0; see _scratch/debug/wp_struct_grad.py). The active set / contact slot / feature
     choices are FROZEN (read from the converged contact); only the geometry is differentiated.
  2. ``_geom_pose_dof_vjp`` -- chain ``adj(geom_xpos, geom_xmat)`` to the per-DOF TANGENT gradient via the
     ANALYTIC body Jacobian ``support.jac_dof`` (general over articulations; the ``adjoint._site_jac_vjp``
     pattern + a rotational term), then ``_dof_to_qpos`` maps dof -> qpos: 1:1 for slide/hinge/free-
     translation, and the QUATERNION LIFT ``∂c/∂q = 2 q ⊗ [0, g]`` for every free/ball joint's angular dofs
     (the nq!=nv fix that matches ``_advance_state``'s quat_integrate adjoint; general over articulations --
     jac_dof already chains the tangent gradient to ancestor dofs, so each joint lifts locally).
  3. ``_subtree_com_qpos_vjp`` -- the moving-com moment-arm term (matters for a SPINNING free body); the
     FREE-body com identity (articulated com-Jacobian is the follow-up).
  4. ``_gather_efc_to_contact`` -- map ``∂r/∂efc_pos`` (efc-row indexed) to the per-contact dist adjoint.

COVERAGE: 11 of the 13 ``_PRIMITIVE_COLLISIONS`` pairs are AD-safe and dispatched here -- plane-sphere,
sphere-sphere, sphere-box, plane-capsule, sphere-capsule, sphere-cylinder, plane-ellipsoid (single contact)
+ capsule-capsule, plane-cylinder, plane-box (multi-contact, slot-indexed) + capsule-box (FROZEN-SEGMENT
witness: ``_capsule_box_freeze`` re-runs the forward's range(8)xrange(3) closest-feature search forward-only
and stores the 1-2 segment parameters; the backward differentiates ``sphere_box`` at the frozen t -- exact
to first order for the minimizing slot-0 contact by the envelope theorem, frozen-t approximate for the
slot-1 spread point). The remaining 2 are NOT AD-safe (data-dependent feature searches / loops over a
runtime-variable contact set) and fall through -> ∂qpos stays 0 for that pair: box-box, plane-convex/mesh.
capsule-capsule is gated on the non-parallel (crossed, slot-0) regime; its parallel slot-1 assignment is
margin-dependent -> documented follow-up. The FD oracle in collision_adjoint_test.py covers + gates every
dispatched pair. Gated only by which pairs ``_narrowphase_recompute`` dispatches -- a frozen-active-set,
frozen-feature, differentiable-geometry boundary.
"""

from typing import Tuple

import warp as wp

from mujoco_warp._src import collision_primitive_core as _cpc
from mujoco_warp._src import math as _math
from mujoco_warp._src import support as _support
from mujoco_warp._src import types as _types
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model

_FREE = int(_types.JointType.FREE.value)
_BALL = int(_types.JointType.BALL.value)
_PLANE = int(_types.GeomType.PLANE.value)
_SPHERE = int(_types.GeomType.SPHERE.value)
_CAPSULE = int(_types.GeomType.CAPSULE.value)
_ELLIPSOID = int(_types.GeomType.ELLIPSOID.value)
_CYLINDER = int(_types.GeomType.CYLINDER.value)
_BOX = int(_types.GeomType.BOX.value)


@wp.func
def _capsule_box_segpos(
  pos_c: wp.vec3,
  axis_c: wp.vec3,
  half_len: float,
  pos_b: wp.vec3,
  mat_b: wp.mat33,
  size_b: wp.vec3,
) -> Tuple[wp.vec2, wp.vec3i]:
  """FROZEN-FEATURE extractor for capsule<->box: re-run ``collision_primitive_core.capsule_box``'s
  closest-feature search and return ``(t1, t2)`` -- the (up to two) SEGMENT PARAMETERS t in [-1, 1] along
  the capsule axis at which the forward reduces the pair to ``sphere_box`` (its tail: sphere at
  ``pos + halfaxis * t``; t2 == t1 when no second contact) -- plus the DISCRETE feature state
  ``(cltype, clcorner, cledge)`` that lets ``_capsule_box_t0`` re-derive t1 as a CLOSED-FORM
  DIFFERENTIABLE function of the poses. This is a VERBATIM transcription of the search half of
  ``capsule_box`` -- the data-dependent argmin and runtime vector indexing that make the core NOT AD-safe
  are fine here because this func is only called from the ``enable_backward=False``
  ``_capsule_box_freeze`` kernel."""
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
      if wp.abs(det) < _cpc.MJ_MINVAL:
        continue
      idet = 1.0 / det
      x1 = wp.float32((mc * u - mb * v) * idet)
      x2 = wp.float32((ma * v - mb * u) * idet)
      s1 = wp.int32(1)
      s2 = wp.int32(1)
      if x1 > 1:
        x1 = 1.0
        s1 = 2
        x2 = _cpc.safe_div(v - mb, mc)
      elif x1 < -1:
        x1 = -1.0
        s1 = 0
        x2 = _cpc.safe_div(v + mb, mc)
      x2_over = x2 > 1.0
      if x2_over or x2 < -1.0:
        if x2_over:
          x2 = 1.0
          s2 = 2
          x1 = _cpc.safe_div(u - mb, ma)
        else:
          x2 = -1.0
          s2 = 0
          x1 = _cpc.safe_div(u + mb, ma)
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
      if dif_sq < bestdist - _cpc.MJ_MINVAL:
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
        m = 2.0 * _cpc.safe_div(size_b[ax], wp.abs(halfaxis[ax]))
        secondpos = min(1.0 - wp.float32(mul) * bestsegmentpos, m)
      else:  # second point along a face of the box
        m = 2.0 * min(
          _cpc.safe_div(size_b[ax1], wp.abs(halfaxis[ax1])),
          _cpc.safe_div(size_b[ax2], wp.abs(halfaxis[ax2])),
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
      e1 = 2.0 * _cpc.safe_div(size_b[ax2], wp.abs(halfaxis[ax2]))
      secondpos = min(e1, secondpos)
      if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):
        e2 = 1.0 - bestboxpos
      else:
        e2 = 1.0 + bestboxpos
      e1 = size_b[ax] * _cpc.safe_div(e2, wp.abs(halfaxis[ax]))
      secondpos = min(e1, secondpos)
      secondpos *= wp.float32(mul)
  else:  # a capsule tip is closest to a box face
    if clface != -1:
      mul = wp.where(cltype == -3, 1, -1)
      secondpos = wp.float32(2.0)
      tmp1 = pos - halfaxis * wp.float32(mul)
      for i in range(3):
        if i != clface:
          ha_r = _cpc.safe_div(wp.float32(mul), halfaxis[i])
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
  pos_bf: wp.vec3,  # capsule center in the BOX frame [grad]
  halfaxis_bf: wp.vec3,  # capsule half-axis in the BOX frame [grad]
  half_len: float,
  size_b: wp.vec3,
  cltype: int,  # frozen discrete feature state (from _capsule_box_segpos)
  clcorner: int,
  cledge: int,
) -> float:
  """DIFFERENTIABLE re-derivation of the primary segment parameter t1 for the FROZEN discrete feature:
  reproduces exactly the closed forms the forward search used, with the branch/clamp state frozen --
  so d(t1)/d(poses) flows and the adjoint matches FD of the forward's actual computation graph (a fully
  frozen t is only first-order exact for INTERIOR minimizers; a box-EDGE contact has t interior with
  pose-dependent drift that FD sees). All feature selects are static compare-selects on frozen ints."""
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
    return _cpc.safe_div(ma * v - mb * u, ma * mc - mb * mb)
  if s1 == 2:  # box-edge param clamped high
    return _cpc.safe_div(v - mb, mc)
  return _cpc.safe_div(v + mb, mc)  # s1 == 0: box-edge param clamped low


@wp.kernel(enable_backward=False)
def _capsule_box_freeze(
  geom_type: wp.array[int],
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  contact_geom: wp.array(dtype=wp.vec2i),
  contact_worldid: wp.array[int],
  nacon: wp.array[int],
  tseg_out: wp.array(dtype=wp.vec2),
  feat_out: wp.array(dtype=wp.vec3i),
):
  """Per frozen capsule<->box contact, extract the sphere-reduction SEGMENT PARAMETERS + discrete feature
  state (see ``_capsule_box_segpos``) so ``_narrowphase_recompute`` can differentiate ``sphere_box`` at a
  re-derived t. Forward-only: the feature search stays out of the backward kernel."""
  cid = wp.tid()
  if cid >= nacon[0]:
    return
  geoms = contact_geom[cid]
  if geoms[0] < 0 or geoms[1] < 0:
    return
  g0 = geoms[0]
  g1 = geoms[1]
  if geom_type[g0] != _CAPSULE or geom_type[g1] != _BOX:
    return
  w = contact_worldid[cid]
  gw = w % geom_size.shape[0]
  m0 = geom_xmat_in[w, g0]
  axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
  ts, ft = _capsule_box_segpos(
    geom_xpos_in[w, g0], axis0, geom_size[gw, g0][1], geom_xpos_in[w, g1], geom_xmat_in[w, g1], geom_size[gw, g1]
  )
  tseg_out[cid] = ts
  feat_out[cid] = ft


@wp.kernel(enable_backward=True)
def _narrowphase_recompute(
  geom_type: wp.array[int],
  geom_size: wp.array2d(dtype=wp.vec3),
  geom_xpos_in: wp.array2d[wp.vec3],  # [grad]
  geom_xmat_in: wp.array2d[wp.mat33],  # [grad]
  contact_geom: wp.array(dtype=wp.vec2i),
  contact_geomcollisionid: wp.array[int],
  contact_worldid: wp.array[int],
  nacon: wp.array[int],
  capsule_box_tseg: wp.array(dtype=wp.vec2),  # frozen capsule-box segment params (_capsule_box_freeze)
  capsule_box_feat: wp.array(dtype=wp.vec3i),  # frozen capsule-box discrete feature state
  cpos_out: wp.array(dtype=wp.vec3),
  dist_out: wp.array[float],
  frame_out: wp.array(dtype=wp.mat33),
):
  """Per-frozen-contact narrowphase REPLAY: per geom pair, call the FORWARD pure geom func from
  collision_primitive_core (the same leaf the forward ``*_wrapper`` uses). Backward-enabled, so Warp
  AUTO-DIFFS the geometry through it -> adj(geom poses). RAW geom poses (NOT the ``Geom`` wp.struct, which
  zeroes Warp reverse-mode). normal/axis = rot[:,2]. slot = contact.geomcollisionid = the FORWARD wrapper's
  write_contact loop index, i.e. the local sub-contact / output row index (single-contact 0; plane-capsule
  caps 0/1; capsule-capsule 0/1; plane-cylinder 0..3; plane-box CORNER id 0..7 -- the wrapper passes id_=i
  over range(8) so the slot is the literal corner, not a bottom-4 remap). Multi-contact rows are selected
  with a static-unrolled loop + runtime compare (NEVER runtime-index a vector/array in a backward kernel).
  geom0=geoms[0], geom1=geoms[1] (forward order). AD-safe primitives only; capsule-box differentiates
  sphere_box at its FROZEN segment parameter (capsule_box_tseg, from _capsule_box_freeze); box-box /
  plane-convex / mesh fall through -> ∂qpos stays 0 for that pair (data-dependent feature search)."""
  cid = wp.tid()
  if cid >= nacon[0]:
    return
  geoms = contact_geom[cid]
  if geoms[0] < 0 or geoms[1] < 0:
    return
  w = contact_worldid[cid]
  slot = contact_geomcollisionid[cid]
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
    dist, pos = _cpc.plane_sphere(n0, p0, p1, s1[0])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = _math.make_frame(n0)
  elif t0 == _SPHERE and t1 == _SPHERE:
    dist, pos, n = _cpc.sphere_sphere(p0, s0[0], p1, s1[0])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = _math.make_frame(n)
  elif t0 == _SPHERE and t1 == _BOX:
    dist, pos, n = _cpc.sphere_box(p0, s0[0], p1, m1, s1)
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = _math.make_frame(n)
  elif t0 == _PLANE and t1 == _CAPSULE:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # capsule local z-axis
    dvec, pmat, frame = _cpc.plane_capsule(n0, p0, p1, axis, s1[0], s1[1])  # two caps, shared frame
    if slot == 0:
      dist_out[cid] = dvec[0]
      cpos_out[cid] = wp.vec3(pmat[0, 0], pmat[0, 1], pmat[0, 2])
    else:
      dist_out[cid] = dvec[1]
      cpos_out[cid] = wp.vec3(pmat[1, 0], pmat[1, 1], pmat[1, 2])
    frame_out[cid] = frame
  elif t0 == _SPHERE and t1 == _CAPSULE:
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # capsule local z-axis
    dist, pos, n = _cpc.sphere_capsule(p0, s0[0], p1, axis, s1[0], s1[1])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = _math.make_frame(n)
  elif t0 == _SPHERE and t1 == _CYLINDER:
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])  # cylinder local z-axis
    dist, pos, n = _cpc.sphere_cylinder(p0, s0[0], p1, axis, s1[0], s1[1])
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = _math.make_frame(n)
  elif t0 == _PLANE and t1 == _ELLIPSOID:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    dist, pos, n = _cpc.plane_ellipsoid(n0, p0, p1, m1, s1)  # returns normal = plane normal
    dist_out[cid] = dist
    cpos_out[cid] = pos
    frame_out[cid] = _math.make_frame(n)
  elif t0 == _CAPSULE and t1 == _CAPSULE:
    # Unique local names per multi-contact branch: Warp codegen scopes locals to the whole function, so
    # reusing one name with a different vec/mat type across branches is a type-conflict error.
    axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    axis1 = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])
    # margin only gates write_contact's slot assignment in the FORWARD; pass a large value so the
    # (frozen) active slot is always populated. Non-parallel (crossed) axes -> slot 0; the parallel
    # slot-1 assignment is margin-dependent -> the documented follow-up (see module docstring).
    cc_dist, cc_pos, cc_nrm = _cpc.capsule_capsule(p0, axis0, s0[0], s0[1], p1, axis1, s1[0], s1[1], 1.0e6)
    for i in range(2):  # static unroll; runtime-compare select (no runtime indexing of the adjoint)
      if i == slot:
        dist_out[cid] = cc_dist[i]
        cpos_out[cid] = wp.vec3(cc_pos[i, 0], cc_pos[i, 1], cc_pos[i, 2])
        frame_out[cid] = _math.make_frame(wp.vec3(cc_nrm[i, 0], cc_nrm[i, 1], cc_nrm[i, 2]))  # per-slot normal
  elif t0 == _PLANE and t1 == _CYLINDER:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    axis = wp.vec3(m1[0, 2], m1[1, 2], m1[2, 2])
    cyl_dist, cyl_pos, cyl_n = _cpc.plane_cylinder(n0, p0, p1, axis, s1[0], s1[1])  # 4 contacts, shared normal
    for i in range(4):  # static unroll; runtime-compare select (no runtime indexing of the adjoint)
      if i == slot:
        dist_out[cid] = cyl_dist[i]
        cpos_out[cid] = wp.vec3(cyl_pos[i, 0], cyl_pos[i, 1], cyl_pos[i, 2])
    frame_out[cid] = _math.make_frame(cyl_n)
  elif t0 == _PLANE and t1 == _BOX:
    n0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    box_dist, box_pos, box_n = _cpc.plane_box(n0, p0, p1, m1, s1)  # 8 corners (slot = corner id), shared normal
    for i in range(8):  # static unroll; runtime-compare select (no runtime indexing of the adjoint)
      if i == slot:
        dist_out[cid] = box_dist[i]
        cpos_out[cid] = wp.vec3(box_pos[i, 0], box_pos[i, 1], box_pos[i, 2])
    frame_out[cid] = _math.make_frame(box_n)
  elif t0 == _CAPSULE and t1 == _BOX:
    # FROZEN-FEATURE WITNESS: the forward's capsule_box tail reduces the pair to sphere_box at 1-2 points
    # along the capsule segment; the search that PICKS the feature is a data-dependent hunt (not AD-safe),
    # so _capsule_box_freeze re-ran it forward-only and froze the DISCRETE state (cltype/clcorner/cledge +
    # clamp branch). Here the primary segment parameter t1 is RE-DERIVED as a closed-form differentiable
    # function of the poses for that frozen feature (_capsule_box_t0) -- the adjoint then matches FD of
    # the forward's actual computation graph (interior-edge contacts have pose-dependent t1 that a fully
    # frozen t misses). The slot-1 spread point rides t1 with its OFFSET frozen (the spread heuristic's
    # own pose-derivative is dropped -- same approximation class as capsule_capsule's parallel slot-1).
    axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
    ts = capsule_box_tseg[cid]
    ft = capsule_box_feat[cid]
    bmT = wp.transpose(m1)
    pos_bf = bmT @ (p0 - p1)
    halfaxis_bf = (bmT @ axis0) * s0[1]
    t0d = _capsule_box_t0(pos_bf, halfaxis_bf, s0[1], s1, ft[0], ft[1], ft[2])
    tk = wp.where(slot == 0, t0d, t0d + (ts[1] - ts[0]))
    cb_center = p0 + axis0 * (s0[1] * tk)
    cb_dist, cb_pos, cb_n = _cpc.sphere_box(cb_center, s0[0], p1, m1, s1)
    dist_out[cid] = cb_dist
    cpos_out[cid] = cb_pos
    frame_out[cid] = _math.make_frame(cb_n)


@wp.kernel(enable_backward=False)
def _gather_efc_to_contact(
  contact_efc_address: wp.array2d[int],
  contact_worldid: wp.array[int],
  nacon: wp.array[int],
  res_efc_pos: wp.array2d[float],  # ∂r/∂efc_pos · λ (per world, per efc row)
  adj_dist_out: wp.array[float],  # per-contact seed for dist_out.grad (∂efc_pos/∂dist = 1)
):
  """Gather ∂r/∂efc_pos (efc-row indexed) to the per-contact normal-row distance adjoint. efc_pos =
  contact.dist - includemargin, so ∂efc_pos/∂dist = 1 (the margin offset is qpos-independent)."""
  cid = wp.tid()
  if cid >= nacon[0]:
    return
  e0 = contact_efc_address[cid, 0]
  if e0 < 0:
    return
  adj_dist_out[cid] = res_efc_pos[contact_worldid[cid], e0]


@wp.kernel(enable_backward=False)
def _cdof_qpos_vjp(
  dof_parentid: wp.array[int],
  dof_jntid: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  cdof_in: wp.array2d[wp.spatial_vector],
  res_cdof: wp.array2d[wp.spatial_vector],  # ∂r/∂cdof · λ (per dof)
  nv: int,
  res_dof: wp.array2d[float],  # OUT: += per-dof TANGENT gradient ∂c/∂(dof k)
):
  """Fixed-reference part of ∂r/∂cdof → qpos: the screw-axis commutator

      ∂cdof_i/∂q_k |_fixed-COM = χ(k,i) · motion_cross(cdof_k, cdof_i),

  with χ the EXACT stored-cdof transport mask (MJPLAN_ADRNE §3): (a) k is a transitive dof_parentid
  ancestor of i (covers inter-body chains AND the free-joint translation→rotation predecessors); PLUS the
  FULL same-quaternion angular block -- (b) i a BALL row and k any of that ball's 3 angular dofs, (c) i a
  FREE-rotation row and k any of that free joint's 3 rotation dofs. The same-quaternion block is the full
  3×3 (NOT triangular): a free/ball basis derivative ∂a_i/∂q_k = a_k×a_i for every (k,i) in the triplet,
  which dof_parentid (k<i only) misses. FREE-translation target rows are skipped (∂(stored cdof)/∂q = 0
  there). The MOVING-COM reference term (0, a_i × ∂C/∂q) is NOT here -- it is folded into the subtree-COM
  seed (_build_ceff) so the validated mass-weighted COM Jacobian carries it. The two together = the honest
  ∂(stored cdof)/∂q (MJPLAN_ADRNE §2/§6; FD-verified vs mj_comPos central diff, cos=1.0 hinge + free+ball).
  enable_backward=False -> the dynamic loops are a manual VJP (no tape replay)."""
  w, k = wp.tid()
  cdof_k = cdof_in[w, k]
  jk = dof_jntid[k]
  acc = float(0.0)
  for i in range(nv):
    ji = dof_jntid[i]
    jti = jnt_type[ji]
    ofs = jnt_dofadr[ji]
    if jti == _FREE and (i - ofs) < 3:
      continue  # FREE-translation target row: ∂(stored cdof_i)/∂q = 0
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
      acc += wp.dot(res_cdof[w, i], _math.motion_cross(cdof_k, cdof_in[w, i]))
  res_dof[w, k] += acc


@wp.kernel(enable_backward=False)
def _build_ceff(
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  cdof_in: wp.array2d[wp.spatial_vector],
  res_cdof: wp.array2d[wp.spatial_vector],
  res_subtree_com: wp.array2d[wp.vec3],
  nv: int,
  ceff_out: wp.array2d[wp.vec3],  # OUT: effective subtree-COM cotangent
):
  """Effective subtree-COM seed (MJPLAN_ADRNE §7): fold the moving-COM part of ∂r/∂cdof into the COM seed so
  ONE pass of the validated subtree-COM Jacobian VJP carries both. Using G_lin,i·(a_i×∂C) = ∂C·(G_lin,i×a_i),

      ceff[r] = res_subtree_com[r] + Σ_{i: root(dof_bodyid[i]) = r} G_lin,i × a_i,

  with G_lin,i = spatial_bottom(res_cdof_i), a_i = spatial_top(cdof_i) (the row's angular axis; zero for
  slide / free-translation, so only rotational rows contribute)."""
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
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  body_mass: wp.array2d[float],
  body_subtreemass: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  xipos_in: wp.array2d[wp.vec3],
  res_subtree_com: wp.array2d[wp.vec3],  # ∂r/∂subtree_com · λ
  nbody: int,
  res_dof: wp.array2d[float],  # OUT: += per-dof TANGENT gradient ∂c/∂(dof k)
):
  """∂r/∂subtree_com chained to qpos via the mass-weighted subtree-com Jacobian (mj_jacSubtreeCom):

      ∂subtree_com[b]/∂q_k = (1/Msub[b]) Σ_{c∈subtree(b)} m_c · jacp(xipos_c, c, k)

  contracted with res_subtree_com and accumulated per dof:
  res_dof[k] += Σ_c (m_c/Msub[root[c]]) · jacp(xipos_c, c, k) · res_subtree_com[root[c]]. The contact
  Jacobian's moment arm is offset = contact_pos - subtree_com[root], so a moving com shifts J even when the
  contact point is handled separately. General over articulations; reproduces the single-free-body identity
  (own-com translation 1:1, rotation 0) so free-body scenes are unchanged. The residual only reads
  subtree_com at TREE-ROOT indices (jac_dof's offset uses subtree_com[body_rootid]), so res_subtree_com is
  nonzero only there -> summing over bodies c by their root[c] covers it. MUST PAIR with _cdof_qpos_vjp."""
  w, k = wp.tid()
  wm = w % body_mass.shape[0]
  acc = float(0.0)
  for c in range(1, nbody):
    if body_isdofancestor[c, k] == 0:  # dof k does not move body c -> jacp = 0
      continue
    r = body_rootid[c]
    jp, _jr = _support.jac_dof(
      body_parentid, body_rootid, dof_bodyid, body_isdofancestor,
      subtree_com_in, cdof_in, xipos_in[w, c], c, k, w,
    )
    acc += (body_mass[wm, c] / body_subtreemass[wm, r]) * wp.dot(jp, res_subtree_com[w, r])
  res_dof[w, k] += acc


@wp.kernel(enable_backward=False)
def _geom_pose_dof_vjp(
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  geom_bodyid: wp.array[int],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  res_geom_xpos: wp.array2d[wp.vec3],  # adj(geom_xpos) from the narrowphase backward
  res_geom_xmat: wp.array2d[wp.mat33],  # adj(geom_xmat)
  ngeom: int,
  res_dof: wp.array2d[float],  # OUT: per-dof TANGENT gradient ∂(contact)/∂(dof k)
):
  """Chain the narrowphase's geom-pose adjoints to the per-DOF TANGENT gradient via support.jac_dof (the
  analytic body Jacobian -- general over articulations, no kinematics-tree AD; the adjoint._site_jac_vjp
  pattern + a rotational term). For dof k: ∂c/∂(dof k) = Σ_geom jacp·adj(geom_xpos) + jacr·τ, where
  τ = Σ_col geom_xmat[:,col] × adj(geom_xmat)[:,col] reduces the orientation adjoint to a WORLD axis
  (∂geom_xmat/∂q_k = [jacr_k]× geom_xmat), so jacr_k·τ = ∂c/∂(rotation about dof k's axis). The dof->qpos
  map -- INCLUDING the free/ball quaternion lift for the angular dofs -- is ``_dof_to_qpos`` (this kernel
  stays in dof/tangent space, which is what the FD surgical oracle validates directly)."""
  w, k = wp.tid()
  acc = float(0.0)
  for g in range(ngeom):
    rgp = res_geom_xpos[w, g]
    rgm = res_geom_xmat[w, g]
    body = geom_bodyid[g]
    jacp, jacr = _support.jac_dof(
      body_parentid, body_rootid, dof_bodyid, body_isdofancestor,
      subtree_com_in, cdof_in, geom_xpos_in[w, g], body, k, w,
    )
    xm = geom_xmat_in[w, g]
    tau = (
      wp.cross(wp.vec3(xm[0, 0], xm[1, 0], xm[2, 0]), wp.vec3(rgm[0, 0], rgm[1, 0], rgm[2, 0]))
      + wp.cross(wp.vec3(xm[0, 1], xm[1, 1], xm[2, 1]), wp.vec3(rgm[0, 1], rgm[1, 1], rgm[2, 1]))
      + wp.cross(wp.vec3(xm[0, 2], xm[1, 2], xm[2, 2]), wp.vec3(rgm[0, 2], rgm[1, 2], rgm[2, 2]))
    )
    acc += wp.dot(jacp, rgp) + wp.dot(jacr, tau)
  res_dof[w, k] += acc


@wp.kernel(enable_backward=False)
def _dof_to_qpos(
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  qpos_in: wp.array2d[float],
  res_dof: wp.array2d[float],  # per-dof TANGENT gradient ∂(contact)/∂(dof)
  res_qpos: wp.array2d[float],  # += dof -> qpos (1:1 for slide/hinge/free-translation; quaternion LIFT for free/ball rotation)
):
  """Map the per-dof tangent gradient ``res_dof`` to ``res_qpos`` (nq-indexed). slide/hinge and the
  free-joint TRANSLATION dofs are 1:1 (qpos coord == dof). FREE/BALL ROTATION needs the QUATERNION LIFT,
  the fix for the nq!=nv mismatch: the per-dof angular gradient g = ∂c/∂ω (3-vec, the joint's 3 angular
  res_dof entries in the BODY frame, since the free/ball cdof angular axes ARE the body frame that
  quat_integrate's right-multiply uses) lifts to the 4 raw-quaternion-coordinate gradients via

      ∂c/∂q = 2 · q ⊗ [0, g]            (= 2 * math.quat_mul_axis(q, g))

  the pseudo-inverse of the forward kinematic map δq = ½ q⊗[0,θ] (|q|=1; (Lᵀ)⁺ = 4L). This matches the
  basis ``_advance_state``'s ``quat_integrate`` adjoint produces for adj(qpos quaternion) -- WITHOUT it the
  3-vec angular gradient was stuffed into the (qw,qx,qy) slots and combined with the integrator's
  4-component quaternion gradient, the rolling-contact bias. General over articulations: jac_dof already
  put the correct tangent gradient on EVERY chain dof (ancestors included), so this just lifts each
  free/ball joint's own quaternion locally."""
  w, j = wp.tid()
  jt = jnt_type[j]
  qadr = jnt_qposadr[j]
  dadr = jnt_dofadr[j]
  if jt == _FREE or jt == _BALL:
    if jt == _FREE:  # translation dofs are 1:1; the quaternion starts at qadr+3, angular dofs at dadr+3
      res_qpos[w, qadr + 0] += res_dof[w, dadr + 0]
      res_qpos[w, qadr + 1] += res_dof[w, dadr + 1]
      res_qpos[w, qadr + 2] += res_dof[w, dadr + 2]
      qadr_q = qadr + 3
      dadr_a = dadr + 3
    else:  # BALL: 4 qpos (quaternion), 3 dofs, both at the joint adr
      qadr_q = qadr
      dadr_a = dadr
    q = wp.quat(qpos_in[w, qadr_q + 0], qpos_in[w, qadr_q + 1], qpos_in[w, qadr_q + 2], qpos_in[w, qadr_q + 3])
    g = wp.vec3(res_dof[w, dadr_a + 0], res_dof[w, dadr_a + 1], res_dof[w, dadr_a + 2])
    dq = 2.0 * _math.quat_mul_axis(q, g)  # 2 q ⊗ [0, g]: lift tangent angular -> raw quaternion gradient
    res_qpos[w, qadr_q + 0] += dq[0]
    res_qpos[w, qadr_q + 1] += dq[1]
    res_qpos[w, qadr_q + 2] += dq[2]
    res_qpos[w, qadr_q + 3] += dq[3]
  else:  # HINGE / SLIDE (nq == nv, qpos coord == dof)
    res_qpos[w, qadr] += res_dof[w, dadr]


def contact_qpos_vjp(
  m: Model,
  d_out: Data,
  qpos_in: wp.array2d[float],  # input/linearization qpos (d.qpos), not integrated d_out.qpos
  res_contact_pos: wp.array,  # ∂r/∂contact_pos · λ (per-contact vec3)
  res_contact_frame: wp.array,  # ∂r/∂contact_frame · λ (per-contact mat33)
  res_efc_pos: wp.array2d[float],  # ∂r/∂efc_pos · λ (nworld, njmax)
  res_subtree_com: wp.array2d[wp.vec3],  # ∂r/∂subtree_com · λ (nworld, nbody)
  res_cdof: wp.array2d[wp.spatial_vector],  # ∂r/∂cdof · λ (nworld, nv)
  res_qpos: wp.array2d[float],  # OUT: += -(∂r_contact/∂qpos)ᵀλ-worth of the contraction (sign per _sub_write)
):
  """Accumulate the contact residual's ∂qpos into ``res_qpos`` from the exposed input-adjoints: replay the
  narrowphase geometry (auto-diff the forward pure funcs) -> adj(geom poses) -> qpos via jac_dof, PLUS the
  articulated contact-Jacobian terms ∂cdof/∂q (screw commutator) and ∂subtree_com/∂q (mass-weighted com
  Jacobian) -- both chained into the same per-dof TANGENT buffer res_dof, then lifted once by _dof_to_qpos
  (free/ball quaternion). ``adjoint.step_backward`` calls this, then ``_sub_write`` subtracts res_qpos
  -> the -(∂r/∂qpos)ᵀλ contribution. nacon=0 / unsupported geom pairs -> no-op."""
  nworld = d_out.qpos.shape[0]
  nv = m.nv
  ncon_max = d_out.contact.pos.shape[0]
  for _arr in (d_out.geom_xpos, d_out.geom_xmat):
    _arr.requires_grad = True  # so the narrowphase backward accumulates their input-adjoints

  cpos_o = wp.zeros(ncon_max, dtype=wp.vec3, requires_grad=True)
  dist_o = wp.zeros(ncon_max, dtype=float, requires_grad=True)
  frame_o = wp.zeros(ncon_max, dtype=wp.mat33, requires_grad=True)
  # frozen capsule-box segment parameters + feature state (forward-only search; see _capsule_box_freeze)
  cb_tseg = wp.zeros(ncon_max, dtype=wp.vec2)
  cb_feat = wp.zeros(ncon_max, dtype=wp.vec3i)
  wp.launch(_capsule_box_freeze, dim=ncon_max,
            inputs=[m.geom_type, m.geom_size, d_out.geom_xpos, d_out.geom_xmat, d_out.contact.geom,
                    d_out.contact.worldid, d_out.nacon],
            outputs=[cb_tseg, cb_feat])
  np_in = [m.geom_type, m.geom_size, d_out.geom_xpos, d_out.geom_xmat, d_out.contact.geom,
           d_out.contact.geomcollisionid, d_out.contact.worldid, d_out.nacon, cb_tseg, cb_feat]
  wp.launch(_narrowphase_recompute, dim=ncon_max, inputs=np_in, outputs=[cpos_o, dist_o, frame_o])
  # seed the geometry adjoints: contact_pos/frame are per-contact direct; efc_pos is efc-row-indexed
  # -> gather to per-contact (∂efc_pos/∂dist = 1).
  wp.copy(cpos_o.grad, res_contact_pos)
  wp.copy(frame_o.grad, res_contact_frame)
  wp.launch(_gather_efc_to_contact, dim=ncon_max,
            inputs=[d_out.contact.efc_address, d_out.contact.worldid, d_out.nacon, res_efc_pos],
            outputs=[dist_o.grad])
  res_geom_xpos = wp.zeros_like(d_out.geom_xpos)
  res_geom_xmat = wp.zeros_like(d_out.geom_xmat)
  adj_np = [None] * len(np_in)
  adj_np[2] = res_geom_xpos  # geom_xpos_in
  adj_np[3] = res_geom_xmat  # geom_xmat_in
  wp.launch(_narrowphase_recompute, dim=ncon_max, inputs=np_in, outputs=[cpos_o, dist_o, frame_o],
            adj_inputs=adj_np, adj_outputs=[cpos_o.grad, dist_o.grad, frame_o.grad], adjoint=True)
  # All contact-Jacobian ∂qpos terms accumulate into ONE per-dof TANGENT buffer res_dof, lifted once to qpos
  # by _dof_to_qpos (free/ball quaternion). The articulated contact-Jacobian-rotation terms are the honest
  # ∂(stored cdof)/∂q + ∂(subtree_com)/∂q chain (MJPLAN_ADRNE §2/§7): (1) narrowphase geom-pose adjoints via
  # jac_dof; (2) the fixed-COM screw commutator (_cdof_qpos_vjp); (3) the mass-weighted COM Jacobian VJP
  # (_subtree_com_qpos_vjp) seeded with the EFFECTIVE COM cotangent (_build_ceff), which folds in the
  # moving-COM reference part of ∂r/∂cdof so one COM-Jacobian pass carries both.
  res_dof = wp.zeros((nworld, nv), dtype=float)
  wp.launch(_geom_pose_dof_vjp, dim=(nworld, nv),
            inputs=[m.body_parentid, m.body_rootid, m.dof_bodyid,
                    m.body_isdofancestor, m.geom_bodyid, d_out.subtree_com, d_out.cdof,
                    d_out.geom_xpos, d_out.geom_xmat, res_geom_xpos, res_geom_xmat, m.ngeom],
            outputs=[res_dof])
  wp.launch(_cdof_qpos_vjp, dim=(nworld, nv),
            inputs=[m.dof_parentid, m.dof_jntid, m.jnt_type, m.jnt_dofadr, d_out.cdof, res_cdof, nv],
            outputs=[res_dof])
  ceff = wp.empty_like(res_subtree_com)
  wp.launch(_build_ceff, dim=(nworld, m.nbody),
            inputs=[m.body_rootid, m.dof_bodyid, d_out.cdof, res_cdof, res_subtree_com, nv],
            outputs=[ceff])
  wp.launch(_subtree_com_qpos_vjp, dim=(nworld, nv),
            inputs=[m.body_parentid, m.body_rootid, m.dof_bodyid, m.body_isdofancestor,
                    m.body_mass, m.body_subtreemass, d_out.subtree_com, d_out.cdof, d_out.xipos,
                    ceff, m.nbody],
            outputs=[res_dof])
  wp.launch(_dof_to_qpos, dim=(nworld, m.njnt),
            inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, qpos_in, res_dof],
            outputs=[res_qpos])
