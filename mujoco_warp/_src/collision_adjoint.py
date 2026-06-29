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

COVERAGE: 10 of the 13 ``_PRIMITIVE_COLLISIONS`` pairs are AD-safe and dispatched here -- plane-sphere,
sphere-sphere, sphere-box, plane-capsule, sphere-capsule, sphere-cylinder, plane-ellipsoid (single contact)
+ capsule-capsule, plane-cylinder, plane-box (multi-contact, slot-indexed). The remaining 3 are NOT AD-safe
(data-dependent feature searches / loops over a runtime-variable contact set) and fall through ->
∂qpos stays 0 for that pair: capsule-box (range(8)xrange(3) closest-feature search with runtime
cltype/clface indices), box-box, plane-convex/mesh. capsule-capsule is gated on the non-parallel (crossed,
slot-0) regime; its parallel slot-1 assignment is margin-dependent -> documented follow-up. The FD oracle in
collision_adjoint_test.py covers + gates every dispatched pair. Gated only by which pairs
``_narrowphase_recompute`` dispatches -- a frozen-active-set, frozen-feature, differentiable-geometry boundary.
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
  geom0=geoms[0], geom1=geoms[1] (forward order). AD-safe primitives only; capsule-box / box-box /
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
def _subtree_com_qpos_vjp(
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  body_jntadr: wp.array[int],
  body_jntnum: wp.array[int],
  body_rootid: wp.array[int],
  res_subtree_com: wp.array2d[wp.vec3],  # ∂r/∂subtree_com · λ
  res_qpos: wp.array2d[float],  # += res_subtree_com[b] · ∂subtree_com[b]/∂qpos
):
  """∂r/∂subtree_com chained to qpos. The contact Jacobian's moment arm is offset = contact_pos -
  subtree_com[root], so a moving com shifts J even when the contact point is handled separately -- this
  is the term that matters for a SPINNING free body (λ's rotational rows make res_subtree_com nonzero).
  FREE-body case: a leaf body that is its own root (com offset 0) has subtree_com[b] = the free-joint
  translation qpos[qadr:qadr+3] -> ∂/∂qpos = identity on the 3 translation coords. The articulated
  com-Jacobian (mass-weighted Σ jac_dof at body coms, for HINGE/SLIDE chains) is the follow-up."""
  w, b = wp.tid()
  if b == 0:
    return
  if body_rootid[b] != b:
    return  # non-root subtree -> articulated com-Jacobian (follow-up)
  ja = body_jntadr[b]
  if body_jntnum[b] != 1 or jnt_type[ja] != _FREE:
    return  # not a single free joint -> articulated (follow-up)
  qadr = jnt_qposadr[ja]
  rsc = res_subtree_com[w, b]
  wp.atomic_add(res_qpos[w], qadr + 0, rsc[0])
  wp.atomic_add(res_qpos[w], qadr + 1, rsc[1])
  wp.atomic_add(res_qpos[w], qadr + 2, rsc[2])


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
  res_contact_pos: wp.array,  # ∂r/∂contact_pos · λ (per-contact vec3)
  res_contact_frame: wp.array,  # ∂r/∂contact_frame · λ (per-contact mat33)
  res_efc_pos: wp.array2d[float],  # ∂r/∂efc_pos · λ (nworld, njmax)
  res_subtree_com: wp.array2d[wp.vec3],  # ∂r/∂subtree_com · λ (nworld, nbody)
  res_qpos: wp.array2d[float],  # OUT: += -(∂r_contact/∂qpos)ᵀλ-worth of the contraction (sign per _sub_write)
):
  """Accumulate the contact residual's ∂qpos into ``res_qpos`` from the exposed input-adjoints: replay the
  narrowphase geometry (auto-diff the forward pure funcs) -> adj(geom poses) -> qpos via jac_dof, plus the
  free-body subtree_com term. ``adjoint.step_backward`` calls this, then ``_sub_write`` subtracts res_qpos
  -> the -(∂r/∂qpos)ᵀλ contribution. nacon=0 / unsupported geom pairs -> no-op."""
  nworld = d_out.qpos.shape[0]
  nv = m.nv
  ncon_max = d_out.contact.pos.shape[0]
  for _arr in (d_out.geom_xpos, d_out.geom_xmat):
    _arr.requires_grad = True  # so the narrowphase backward accumulates their input-adjoints

  cpos_o = wp.zeros(ncon_max, dtype=wp.vec3, requires_grad=True)
  dist_o = wp.zeros(ncon_max, dtype=float, requires_grad=True)
  frame_o = wp.zeros(ncon_max, dtype=wp.mat33, requires_grad=True)
  np_in = [m.geom_type, m.geom_size, d_out.geom_xpos, d_out.geom_xmat, d_out.contact.geom,
           d_out.contact.geomcollisionid, d_out.contact.worldid, d_out.nacon]
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
  # geom-pose adjoints -> per-dof TANGENT gradient (jac_dof), then dof -> qpos with the free/ball
  # quaternion lift (_dof_to_qpos): the nq!=nv quaternion-tangent fix for rotation-dependent contacts.
  res_dof = wp.zeros((nworld, nv), dtype=float)
  wp.launch(_geom_pose_dof_vjp, dim=(nworld, nv),
            inputs=[m.body_parentid, m.body_rootid, m.dof_bodyid,
                    m.body_isdofancestor, m.geom_bodyid, d_out.subtree_com, d_out.cdof,
                    d_out.geom_xpos, d_out.geom_xmat, res_geom_xpos, res_geom_xmat, m.ngeom],
            outputs=[res_dof])
  wp.launch(_dof_to_qpos, dim=(nworld, m.njnt),
            inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d_out.qpos, res_dof],
            outputs=[res_qpos])
  # ∂r/∂subtree_com chained to qpos (the moving-com moment-arm term; matters for spinning free bodies).
  wp.launch(_subtree_com_qpos_vjp, dim=(nworld, m.nbody),
            inputs=[m.jnt_type, m.jnt_qposadr, m.body_jntadr, m.body_jntnum, m.body_rootid, res_subtree_com],
            outputs=[res_qpos])
