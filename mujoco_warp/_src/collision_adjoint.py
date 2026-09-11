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
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.warp_util import event_scope

# adjoint module: backward stays on so AD leaves differentiate through cross-module @wp.funcs
wp.set_module_options({"enable_backward": True})


# Store the primal capsule-box witness needed by the differentiable replay.
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
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  geoms = contact_geom_in[cid]
  if geoms[0] < 0 or geoms[1] < 0:
    return
  g0 = geoms[0]
  g1 = geoms[1]
  if geom_type[g0] != GeomType.CAPSULE or geom_type[g1] != GeomType.BOX:
    return
  w = contact_worldid_in[cid]
  gw = w % geom_size.shape[0]
  m0 = geom_xmat_in[w, g0]
  axis0 = wp.vec3(m0[0, 2], m0[1, 2], m0[2, 2])
  ts, ft = collision_primitive_core.capsule_box_witness(
    geom_xpos_in[w, g0], axis0, geom_size[gw, g0][1], geom_xpos_in[w, g1], geom_xmat_in[w, g1], geom_size[gw, g1]
  )
  tseg_out[cid] = ts
  feat_out[cid] = ft


# Frozen-witness narrowphase replay using collision_primitive_core's differentiable leaves.
# Eleven primitive pairs are dispatched:
# plane-{sphere,capsule,ellipsoid,cylinder,box}, sphere-{sphere,box,capsule,cylinder},
# capsule-capsule, and capsule-box. Box-box and plane-convex/mesh are not AD-safe. Capsule-box
# freezes the minimizing feature; capsule-capsule supports the non-parallel slot-0 regime.
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
    cb_dist, cb_pos, cb_n = collision_primitive_core.capsule_box_from_witness(
      p0, axis0, s0[0], s0[1], p1, m1, s1, slot, capsule_box_tseg[cid], capsule_box_feat[cid]
    )
    dist_out[cid] = cb_dist
    cpos_out[cid] = cb_pos
    frame_out[cid] = math.make_frame(cb_n)


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

  A frozen-witness narrowphase replay yields adj(geom poses); analytic Jacobian VJPs then chain
  everything into one per-dof tangent buffer that _dof_to_qpos lifts.

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
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d_out)
  nworld = d_out.qpos.shape[0]
  nv = m.nv
  nconmax = d_out.contact.pos.shape[0]
  contact_segment, contact_feature = bc.contact_segment, bc.contact_feature
  contact_adj_dist = bc.contact_adj_dist
  res_geom_xpos, res_geom_xmat = bc.res_geom_xpos, bc.res_geom_xmat
  contact_res_dof, contact_ceff = bc.contact_res_dof, bc.contact_ceff
  contact_segment.zero_()
  contact_feature.zero_()
  contact_adj_dist.zero_()
  res_geom_xpos.zero_()
  res_geom_xmat.zero_()
  contact_res_dof.zero_()
  contact_ceff.zero_()

  wp.launch(
    _capsule_box_freeze,
    dim=nconmax,
    inputs=[m.geom_type, m.geom_size, d_out.geom_xpos, d_out.geom_xmat, d_out.contact.geom, d_out.contact.worldid, d_out.nacon],
    outputs=[contact_segment, contact_feature],
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
    (contact_segment, None),
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
