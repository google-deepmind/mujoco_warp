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
"""VJPs for smooth forces, RNE dynamics, and mass-matrix position dependence."""

import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import math
from mujoco_warp._src import smooth
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec10
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

# Adjoint reductions use unordered atomics; deterministic backward execution is deferred.
wp.set_module_options({"enable_backward": True})


@wp.func
def _spring_contract(stiffness: float, poly: wp.vec2, displacement: float, seed: float):
  force = -displacement * util_misc._poly_force(stiffness, poly, displacement, 0)
  return seed * force


@wp.func
def _spring_contract(stiffness: float, poly: wp.vec2, displacement: wp.vec3, seed: wp.vec3):
  k = util_misc._poly_force(stiffness, poly, wp.length(displacement), 0)
  return wp.dot(seed, -k * displacement)


@wp.func
def _spring_contract(stiffness: float, poly: wp.vec2, rotation: wp.quat, reference: wp.quat, seed: wp.vec3):
  displacement = math.quat_sub(wp.normalize(rotation), reference)
  return _spring_contract(stiffness, poly, displacement, seed)


@wp.kernel(enable_backward=False)
def _spring_qpos_vjp(
  # Model:
  opt_disableflags: int,
  qpos_spring: wp.array2d[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_stiffness: wp.array2d[float],
  jnt_stiffnesspoly: wp.array2d[wp.vec2],
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  # Out:
  qpos_spring_grad_out: wp.array2d[float],
  jnt_stiffness_grad_out: wp.array2d[float],
  jnt_stiffnesspoly_grad_out: wp.array2d[wp.vec2],
  res_qpos_out: wp.array2d[float],
):
  worldid, jntid = wp.tid()
  jnttype = jnt_type[jntid]
  qposid = jnt_qposadr[jntid]
  dofid = jnt_dofadr[jntid]
  stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], jntid]
  poly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], jntid]
  nq = 7 if jnttype == JointType.FREE else (4 if jnttype == JointType.BALL else 1)
  # Zero coefficients still have derivatives when fitting springs.
  has_stiffness = stiffness != 0.0 or poly[0] != 0.0 or poly[1] != 0.0
  if (opt_disableflags & DisableBit.SPRING) or (
    not has_stiffness and not jnt_stiffness_grad_out and not jnt_stiffnesspoly_grad_out
  ):
    for i in range(nq):
      res_qpos_out[worldid, qposid + i] = 0.0
    return

  spring_worldid = worldid % qpos_spring.shape[0]
  # Model gradients use the force sign; qpos returns the negative-force residual VJP.
  if jnttype == JointType.HINGE or jnttype == JointType.SLIDE:
    scalar_grad = wp.grad(_spring_contract)(
      stiffness, poly, qpos_in[worldid, qposid] - qpos_spring[spring_worldid, qposid], lam[worldid, dofid]
    )
    stiffness_grad, poly_grad = scalar_grad[0], scalar_grad[1]
    res_qpos_out[worldid, qposid] = -scalar_grad[2]
    if qpos_spring_grad_out:
      wp.atomic_sub(qpos_spring_grad_out, spring_worldid, qposid, scalar_grad[2])  # kernel_analyzer: ignore[atomic]
  else:
    stiffness_grad, poly_grad = float(0.0), wp.vec2()
    if jnttype == JointType.FREE:
      pos = wp.vec3(qpos_in[worldid, qposid], qpos_in[worldid, qposid + 1], qpos_in[worldid, qposid + 2])
      ref_pos = wp.vec3(
        qpos_spring[spring_worldid, qposid], qpos_spring[spring_worldid, qposid + 1], qpos_spring[spring_worldid, qposid + 2]
      )
      seed = wp.vec3(lam[worldid, dofid], lam[worldid, dofid + 1], lam[worldid, dofid + 2])
      linear_grad = wp.grad(_spring_contract)(stiffness, poly, pos - ref_pos, seed)
      stiffness_grad += linear_grad[0]
      poly_grad += linear_grad[1]
      pos_grad = linear_grad[2]
      for i in range(3):
        res_qpos_out[worldid, qposid + i] = -pos_grad[i]
        if qpos_spring_grad_out:
          wp.atomic_sub(qpos_spring_grad_out, spring_worldid, qposid + i, pos_grad[i])  # kernel_analyzer: ignore[atomic]
      qposid += 3
      dofid += 3
    rot = wp.quat(
      qpos_in[worldid, qposid], qpos_in[worldid, qposid + 1], qpos_in[worldid, qposid + 2], qpos_in[worldid, qposid + 3]
    )
    ref_rot = wp.quat(
      qpos_spring[spring_worldid, qposid],
      qpos_spring[spring_worldid, qposid + 1],
      qpos_spring[spring_worldid, qposid + 2],
      qpos_spring[spring_worldid, qposid + 3],
    )
    seed = wp.vec3(lam[worldid, dofid], lam[worldid, dofid + 1], lam[worldid, dofid + 2])
    rotation_grad = wp.grad(_spring_contract)(stiffness, poly, rot, ref_rot, seed)
    stiffness_grad += rotation_grad[0]
    poly_grad += rotation_grad[1]
    rot_grad, ref_rot_grad = rotation_grad[2], rotation_grad[3]
    for i in range(4):
      res_qpos_out[worldid, qposid + i] = -rot_grad[i]
      if qpos_spring_grad_out:
        wp.atomic_add(qpos_spring_grad_out, spring_worldid, qposid + i, ref_rot_grad[i])  # kernel_analyzer: ignore[atomic]
  if jnt_stiffness_grad_out:
    wp.atomic_add(
      jnt_stiffness_grad_out, worldid % jnt_stiffness.shape[0], jntid, stiffness_grad
    )  # kernel_analyzer: ignore[atomic]
  if jnt_stiffnesspoly_grad_out:
    wp.atomic_add(
      jnt_stiffnesspoly_grad_out, worldid % jnt_stiffnesspoly.shape[0], jntid, poly_grad
    )  # kernel_analyzer: ignore[atomic]


@event_scope
def spring_qpos_vjp(m: Model, d: Data, lam: wp.array2d, bc: BackwardContext | None = None):
  """Returns d(lam.T @ -qfrc_spring)/dqpos and accumulates spring parameter gradients."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  wp.launch(
    _spring_qpos_vjp,
    dim=(d.nworld, m.njnt),
    inputs=[
      m.opt.disableflags,
      m.qpos_spring,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      d.qpos,
      lam,
    ],
    outputs=[m.qpos_spring.grad, m.jnt_stiffness.grad, m.jnt_stiffnesspoly.grad, bc.smooth_res_qpos],
  )
  return bc.smooth_res_qpos


@wp.func
def _actuator_moment_dots(
  # Model:
  jnt_actfrclimited: wp.array[bool],
  jnt_actfrcrange: wp.array2d[wp.vec2],
  dof_jntid: wp.array[int],
  # Data in:
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  qfrc_actuator_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  worldid: int,
  rowadr: int,
  rownnz: int,
) -> tuple[float, float]:
  """Contracts a moment row separately with unclamped and clamped joint-force cotangents."""
  unclamped, clamped = float(0.0), float(0.0)
  model_worldid = worldid % jnt_actfrcrange.shape[0]
  for i in range(rownnz):
    elemid = rowadr + i
    dofid = moment_colind_in[worldid, elemid]
    jntid = dof_jntid[dofid]
    is_clamped = bool(False)
    if jnt_actfrclimited[jntid]:
      bounds = jnt_actfrcrange[model_worldid, jntid]
      force = qfrc_actuator_in[worldid, dofid]
      is_clamped = force <= bounds[0] or force >= bounds[1]
    value = actuator_moment_in[worldid, elemid] * lam[worldid, dofid]
    if is_clamped:
      clamped += value
    else:
      unclamped += value
  return unclamped, clamped


@wp.kernel(enable_backward=False)
def _actuator_qpos_vjp(
  # Model:
  jnt_actfrclimited: wp.array[bool],
  jnt_actfrcrange: wp.array2d[wp.vec2],
  dof_jntid: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  # Data in:
  ctrl_in: wp.array2d[float],
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  qfrc_actuator_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  dsbl_clampctrl: int,
  # Out:
  res_dof_out: wp.array2d[float],
):
  w, actid = wp.tid()
  dfdl = float(0.0)
  if actuator_gaintype[actid] == GainType.AFFINE:
    ctrl = ctrl_in[w % ctrl_in.shape[0], actid]
    if actuator_ctrllimited[actid] and dsbl_clampctrl == 0:
      ctrlrange = actuator_ctrlrange[w % actuator_ctrlrange.shape[0], actid]
      ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])
    dfdl += actuator_gainprm[w % actuator_gainprm.shape[0], actid][1] * ctrl
  if actuator_biastype[actid] == BiasType.AFFINE:
    dfdl += actuator_biasprm[w % actuator_biasprm.shape[0], actid][1]
  if dfdl == 0.0:
    return

  if actuator_forcelimited[actid]:
    forcerange = actuator_forcerange[w % actuator_forcerange.shape[0], actid]
    force = actuator_force_in[w, actid]
    if force <= forcerange[0] or force >= forcerange[1]:
      return

  rowadr = moment_rowadr_in[w, actid]
  rownnz = moment_rownnz_in[w, actid]
  moment_lam, _ = _actuator_moment_dots(
    jnt_actfrclimited,
    jnt_actfrcrange,
    dof_jntid,
    moment_colind_in,
    actuator_moment_in,
    qfrc_actuator_in,
    lam,
    w,
    rowadr,
    rownnz,
  )
  scale = -moment_lam * dfdl
  if scale == 0.0:
    return
  for i in range(rownnz):
    elemid = rowadr + i
    wp.atomic_add(
      res_dof_out[w], moment_colind_in[w, elemid], scale * actuator_moment_in[w, elemid]
    )  # kernel_analyzer: ignore[atomic]


@event_scope
def actuator_qpos_vjp(m: Model, d: Data, lam: wp.array2d, bc: BackwardContext | None = None):
  """Returns the affine joint-actuator position VJP."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  smooth_res_qpos = bc.smooth_res_qpos
  smooth_res_qpos.zero_()
  if not m.nu or int(m.opt.disableflags) & DisableBit.ACTUATION:
    return smooth_res_qpos

  smooth_res_dof = bc.smooth_res_dof
  smooth_res_dof.zero_()
  wp.launch(
    _actuator_qpos_vjp,
    dim=(d.nworld, m.nu),
    inputs=[
      m.jnt_actfrclimited,
      m.jnt_actfrcrange,
      m.dof_jntid,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_forcelimited,
      m.actuator_forcerange,
      m.actuator_ctrllimited,
      m.actuator_ctrlrange,
      d.ctrl,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.actuator_force,
      d.qfrc_actuator,
      lam,
      int(m.opt.disableflags) & DisableBit.CLAMPCTRL,
    ],
    outputs=[smooth_res_dof],
  )
  wp.launch(
    smooth_kinematics_adjoint._dof_to_qpos,
    dim=(d.nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, smooth_res_dof],
    outputs=[smooth_res_qpos],
  )
  return smooth_res_qpos


# The body's own H is added by the same-body reverse; only descendants contribute here.
@wp.kernel(enable_backward=False)
def _comvel_W_sum(
  # Model:
  body_subtreenum: wp.array[int],
  # In:
  adj_cvel: wp.array2d[wp.spatial_vector],
  H_in: wp.array2d[wp.spatial_vector],
  # Out:
  W_out: wp.array2d[wp.spatial_vector],
):
  w, b = wp.tid()
  acc = adj_cvel[w, b]
  for child in range(b + 1, b + body_subtreenum[b]):
    acc += adj_cvel[w, child] + H_in[w, child]
  W_out[w, b] = acc


@wp.kernel(enable_backward=False)
def _xfrc_kinematics_vjp(
  # Model:
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_subtreenum: wp.array[int],
  # Data in:
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  lam: wp.array2d[float],
  # Out:
  adj_xipos_out: wp.array2d[wp.vec3],
  adj_subtree_com_out: wp.array2d[wp.vec3],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
):
  worldid, dofid = wp.tid()
  rotational_cdof = wp.spatial_top(cdof_in[worldid, dofid])
  dofbodyid = dof_bodyid[dofid]
  rootid = body_rootid[dofbodyid]
  subtree_com = subtree_com_in[worldid, rootid]
  # The residual subtracts the generalized force produced by support._apply_ft.
  seed = -lam[worldid, dofid]
  adj_rot = wp.vec3()
  adj_lin = wp.vec3()
  adj_com = wp.vec3()
  for bodyid in range(dofbodyid, dofbodyid + body_subtreenum[dofbodyid]):
    wrench = xfrc_applied_in[worldid, bodyid]
    if wrench == wp.spatial_vector():
      continue
    force = wp.spatial_top(wrench)
    torque = wp.spatial_bottom(wrench)
    offset = xipos_in[worldid, bodyid] - subtree_com
    adj_rot += seed * (torque + wp.cross(offset, force))
    adj_lin += seed * force
    adj_offset = seed * wp.cross(force, rotational_cdof)
    wp.atomic_add(adj_xipos_out, worldid, bodyid, adj_offset)  # kernel_analyzer: ignore[atomic]
    adj_com -= adj_offset
  wp.atomic_add(adj_subtree_com_out, worldid, rootid, adj_com)  # kernel_analyzer: ignore[atomic]
  adj_cdof_out[worldid, dofid] += wp.spatial_vector(adj_rot, adj_lin)


def xfrc_kinematics_backward(m: Model, d: Data, lam: wp.array2d, xfrc_applied: wp.array2d, bc: BackwardContext):
  """Accumulates the kinematic VJP of -lam.T @ J(q).T @ xfrc_applied."""
  wp.launch(
    _xfrc_kinematics_vjp,
    dim=(d.nworld, m.nv),
    inputs=[m.body_rootid, m.dof_bodyid, m.body_subtreenum, xfrc_applied, d.xipos, d.subtree_com, d.cdof, lam],
    outputs=[bc.adj_xipos, bc.adj_subtree_com, bc.adj_cdof],
  )


@wp.func
def _rne_force_contract(ci: vec10, cv: wp.spatial_vector, ca: wp.spatial_vector, seed: wp.spatial_vector):
  return wp.dot(smooth.rne_body_force(ci, cv, ca), seed)


@wp.func
def _rne_contract(
  # In:
  ci: vec10,
  cv: wp.spatial_vector,
  ca: wp.spatial_vector,
  mass_ca: wp.spatial_vector,
  seed: wp.spatial_vector,
  mass_seed: wp.spatial_vector,
):
  return wp.dot(smooth.rne_body_force(ci, cv, ca), seed) + wp.dot(math.inert_vec(ci, mass_ca), mass_seed)


@wp.func
def _cinert_contract(mass: float, inertia: wp.vec3, orientation: wp.quat, offset: wp.vec3, seed: vec10):
  return wp.dot(smooth.inert_com(mass, inertia, math.quat_to_mat(orientation), offset), seed)


@wp.func
def _rne_body_inputs(
  # Model:
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  # In:
  lam: wp.array2d[float],
  mass_y: wp.array2d[float],
  mass_w: wp.array2d[float],
  with_mass: bool,
  worldid: int,
  bodyid: int,
):
  """Sums DOF contributions to body acceleration and force cotangents, including the mass term."""
  ca, seed = wp.spatial_vector(), wp.spatial_vector()
  mass_ca, mass_seed = wp.spatial_vector(), wp.spatial_vector()
  start = body_dofadr[bodyid]
  for i in range(start, start + body_dofnum[bodyid]):
    S = cdof_in[worldid, i]
    ca += cdof_dot_in[worldid, i] * qvel_in[worldid, i] + S * qacc_in[worldid, i]
    seed += lam[worldid, i] * S
    if with_mass:
      mass_ca += mass_w[worldid, i] * S
      mass_seed -= mass_y[worldid, i] * S
  return ca, seed, mass_ca, mass_seed


@cache_kernel
def _rne_vjp(nbody: int, with_mass: bool, mass_grad: bool, inertia_grad: bool):
  tile_size = adjoint_util._body_tile_size(nbody, smooth.WORLD_SUM_BLOCK)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    opt_gravity: wp.array[wp.vec3],
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_mass: wp.array2d[float],
    body_inertia: wp.array2d[wp.vec3],
    body_gravcomp: wp.array2d[float],
    dof_bodyid: wp.array[int],
    body_subtreenum: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    qacc_in: wp.array2d[float],
    xipos_in: wp.array2d[wp.vec3],
    xiquat_in: wp.array2d[wp.quat],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    cinert_in: wp.array2d[vec10],
    cvel_in: wp.array2d[wp.spatial_vector],
    cdof_dot_in: wp.array2d[wp.spatial_vector],
    # In:
    gravity_enabled: bool,
    lam: wp.array2d[float],
    mass_y: wp.array2d[float],
    mass_w: wp.array2d[float],
    # Out:
    adj_qvel_out: wp.array2d[float],
    adj_cdof_out: wp.array2d[wp.spatial_vector],
    adj_cdof_dot_out: wp.array2d[wp.spatial_vector],
    adj_cvel_out: wp.array2d[wp.spatial_vector],
    adj_cinert_out: wp.array2d[vec10],
    adj_xipos_out: wp.array2d[wp.vec3],
    adj_xiquat_out: wp.array2d[wp.quat],
    adj_subtree_com_out: wp.array2d[wp.vec3],
    res_body_mass_out: wp.array2d[float],
    res_body_inertia_out: wp.array2d[wp.vec3],
  ):
    w, lane = wp.tid()
    ca_local = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")
    seed_local = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")
    force_tile = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")
    ca_bar_tile = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")
    if wp.static(with_mass):
      mass_ca_local = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")
      mass_seed_local = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")
      mass_force_tile = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")
      mass_ca_bar_tile = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")

    for body_start in range(0, wp.static(nbody), wp.static(tile_size)):
      # Explicit tile shapes also work with the CPU backend's single lane.
      for start in range(0, wp.static(tile_size), wp.block_dim()):
        body = body_start + start + lane
        local_ca, local_seed = wp.spatial_vector(), wp.spatial_vector()
        local_mass_ca, local_mass_seed = wp.spatial_vector(), wp.spatial_vector()
        if body > 0 and body < wp.static(nbody):
          local_ca, local_seed, local_mass_ca, local_mass_seed = _rne_body_inputs(
            body_dofnum,
            body_dofadr,
            qvel_in,
            qacc_in,
            cdof_in,
            cdof_dot_in,
            lam,
            mass_y,
            mass_w,
            wp.static(with_mass),
            w,
            body,
          )
        wp.tile_assign(ca_local, wp.tile(local_ca, preserve_type=True), offset=(start,))
        wp.tile_assign(seed_local, wp.tile(local_seed, preserve_type=True), offset=(start,))
        if wp.static(with_mass):
          wp.tile_assign(mass_ca_local, wp.tile(local_mass_ca, preserve_type=True), offset=(start,))
          wp.tile_assign(mass_seed_local, wp.tile(local_mass_seed, preserve_type=True), offset=(start,))

      for start in range(0, wp.static(tile_size), wp.block_dim()):
        body = body_start + start + lane
        force, mass_force = wp.spatial_vector(), wp.spatial_vector()
        ca_bar, mass_ca_bar = wp.spatial_vector(), wp.spatial_vector()
        if body > 0 and body < wp.static(nbody):
          ca, seed = wp.spatial_vector(), wp.spatial_vector()
          mass_ca, mass_seed = wp.spatial_vector(), wp.spatial_vector()
          b = body
          while b > 0:
            if b >= body_start:
              ca += ca_local[b - body_start]
              seed += seed_local[b - body_start]
              if wp.static(with_mass):
                mass_ca += mass_ca_local[b - body_start]
                mass_seed += mass_seed_local[b - body_start]
            else:
              local_ca, local_seed, local_mass_ca, local_mass_seed = _rne_body_inputs(
                body_dofnum,
                body_dofadr,
                qvel_in,
                qacc_in,
                cdof_in,
                cdof_dot_in,
                lam,
                mass_y,
                mass_w,
                wp.static(with_mass),
                w,
                b,
              )
              ca += local_ca
              seed += local_seed
              if wp.static(with_mass):
                mass_ca += local_mass_ca
                mass_seed += local_mass_seed
            b = body_parentid[b]
          if gravity_enabled:
            gravity = opt_gravity[w % opt_gravity.shape[0]]
            if body_gravcomp:
              gravity *= 1.0 - body_gravcomp[w % body_gravcomp.shape[0], body]
            ca -= wp.spatial_vector(0.0, 0.0, 0.0, gravity[0], gravity[1], gravity[2])
          ci = cinert_in[w, body]
          cv = cvel_in[w, body]
          if wp.static(with_mass):
            ci_bar, cv_bar, ca_bar, mass_ca_bar, force, mass_force = wp.grad(_rne_contract)(
              ci, cv, ca, mass_ca, seed, mass_seed
            )
          else:
            ci_bar, cv_bar, ca_bar, force = wp.grad(_rne_force_contract)(ci, cv, ca, seed)
          adj_cinert_out[w, body] = ci_bar
          adj_cvel_out[w, body] = cv_bar
          root = body_rootid[body]
          mass_bar, inertia_bar, orientation_bar, offset_bar, _ = wp.grad(_cinert_contract)(
            body_mass[w % body_mass.shape[0], body],
            body_inertia[w % body_inertia.shape[0], body],
            xiquat_in[w, body],
            xipos_in[w, body] - subtree_com_in[w, root],
            ci_bar,
          )
          adj_xipos_out[w, body] = offset_bar
          adj_xiquat_out[w, body] = orientation_bar
          wp.atomic_sub(adj_subtree_com_out, w, root, offset_bar)  # kernel_analyzer: ignore[atomic]
          if wp.static(mass_grad):
            wp.atomic_add(res_body_mass_out, w % body_mass.shape[0], body, mass_bar)  # kernel_analyzer: ignore[atomic]
          if wp.static(inertia_grad):
            wp.atomic_add(res_body_inertia_out, w % body_inertia.shape[0], body, inertia_bar)  # kernel_analyzer: ignore[atomic]
        elif body == 0:
          adj_cinert_out[w, 0] = vec10()
          adj_cvel_out[w, 0] = wp.spatial_vector()
          adj_xipos_out[w, 0] = wp.vec3()
          adj_xiquat_out[w, 0] = wp.quat()
        wp.tile_assign(force_tile, wp.tile(force, preserve_type=True), offset=(start,))
        wp.tile_assign(ca_bar_tile, wp.tile(ca_bar, preserve_type=True), offset=(start,))
        if wp.static(with_mass):
          wp.tile_assign(mass_force_tile, wp.tile(mass_force, preserve_type=True), offset=(start,))
          wp.tile_assign(mass_ca_bar_tile, wp.tile(mass_ca_bar, preserve_type=True), offset=(start,))

      for i in range(lane, nv, wp.block_dim()):
        body = dof_bodyid[i]
        acc_force, acc_ca_bar = wp.spatial_vector(), wp.spatial_vector()
        acc_mass_force, acc_mass_ca_bar = wp.spatial_vector(), wp.spatial_vector()
        end = wp.min(body + body_subtreenum[body], body_start + wp.static(tile_size))
        for b in range(wp.max(body, body_start), end):
          acc_force += force_tile[b - body_start]
          acc_ca_bar += ca_bar_tile[b - body_start]
          if wp.static(with_mass):
            acc_mass_force += mass_force_tile[b - body_start]
            acc_mass_ca_bar += mass_ca_bar_tile[b - body_start]
        qvel_bar = wp.dot(acc_ca_bar, cdof_dot_in[w, i])
        cdof_dot_bar = qvel_in[w, i] * acc_ca_bar
        S_bar = lam[w, i] * acc_force + qacc_in[w, i] * acc_ca_bar
        if wp.static(with_mass):
          S_bar += -mass_y[w, i] * acc_mass_force + mass_w[w, i] * acc_mass_ca_bar
        if body_start == 0:
          adj_qvel_out[w, i] = qvel_bar
          adj_cdof_dot_out[w, i] = cdof_dot_bar
          adj_cdof_out[w, i] = S_bar
        else:
          adj_qvel_out[w, i] += qvel_bar
          adj_cdof_dot_out[w, i] += cdof_dot_bar
          adj_cdof_out[w, i] += S_bar

  return kernel


@event_scope
def rne_backward(
  m: Model,
  d: Data,
  lam: wp.array2d,
  mass_y: wp.array2d | None,
  mass_w: wp.array2d | None,
  bc: BackwardContext,
  xfrc_applied: wp.array2d | None = None,
  body_gravcomp: wp.array2d | None = None,
):
  """Returns the RNE qpos VJP with optional -mass_y.T @ M @ mass_w; leaves qvel cotangents in bc."""
  bc.adj_subtree_com.zero_()
  mass_grad = m.body_mass.requires_grad
  inertia_grad = m.body_inertia.requires_grad
  if mass_grad:
    bc.res_body_mass.zero_()
  if inertia_grad:
    bc.res_body_inertia.zero_()
  block_dim = adjoint_util._body_tile_size(m.nbody, smooth.WORLD_SUM_BLOCK)
  wp.launch_tiled(
    _rne_vjp(m.nbody, mass_y is not None, mass_grad, inertia_grad),
    dim=d.nworld,
    block_dim=block_dim,
    inputs=[
      m.nv,
      m.opt.gravity,
      m.body_parentid,
      m.body_rootid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_mass,
      m.body_inertia,
      body_gravcomp,
      m.dof_bodyid,
      m.body_subtreenum,
      d.qvel,
      d.qacc,
      d.xipos,
      d.xiquat,
      d.subtree_com,
      d.cdof,
      d.cinert,
      d.cvel,
      d.cdof_dot,
      not bool(m.opt.disableflags & DisableBit.GRAVITY),
      lam,
      mass_y,
      mass_w,
    ],
    outputs=[
      bc.smooth_adj_qvel,
      bc.adj_cdof,
      bc.adj_cdof_dot,
      bc.adj_cvel,
      bc.adj_cinert,
      bc.adj_xipos,
      bc.adj_xiquat,
      bc.adj_subtree_com,
      bc.res_body_mass if mass_grad else None,
      bc.res_body_inertia if inertia_grad else None,
    ],
  )
  comvel_backward(m, d, bc.adj_cvel, bc.adj_cdof_dot, bc=bc)
  if xfrc_applied is not None:
    xfrc_kinematics_backward(m, d, lam, xfrc_applied, bc)
  bc.smooth_res_qpos.zero_()
  smooth_kinematics_adjoint.kinematics_qpos_vjp(
    m,
    d,
    d.qpos,
    bc.adj_xipos,
    bc.adj_xiquat,
    bc.adj_cdof,
    bc.adj_subtree_com,
    bc.smooth_ceff,
    bc.smooth_res_dof,
    bc.smooth_res_qpos,
  )
  return bc.smooth_res_qpos


# Reverse com_vel into qvel and cdof without materializing its Jacobian.


@wp.func
def _cv_scatter(
  # Data in:
  qvel_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  w: int,
  j: int,
  t: wp.spatial_vector,
  # Out:
  adj_qvel_out: wp.array2d[float],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
):
  """Scatters a running velocity cotangent t to dof j's qvel and cdof cotangents."""
  adj_qvel_out[w, j] = adj_qvel_out[w, j] + wp.dot(cdof_in[w, j], t)
  adj_cdof_out[w, j] = adj_cdof_out[w, j] + qvel_in[w, j] * t


@wp.func
def _cv_scatter3(
  # Data in:
  qvel_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  w: int,
  dofid: int,
  t: wp.spatial_vector,
  # Out:
  adj_qvel_out: wp.array2d[float],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
):
  for axis in range(3):
    _cv_scatter(qvel_in, cdof_in, w, dofid + 2 - axis, t, adj_qvel_out, adj_cdof_out)


@wp.func
def _comvel_rot3_vjp(
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_cdof_dot: wp.array2d[wp.spatial_vector],
  w: int,
  dofid: int,
  velocity: wp.spatial_vector,
  # Out:
  h_out: wp.array2d[wp.spatial_vector],
  k_out: wp.array2d[wp.spatial_vector],
) -> wp.spatial_vector:
  hsum = wp.spatial_vector()
  for axis in range(3):
    i = dofid + axis
    g = adj_cdof_dot[w, i]
    h = math.motion_cross_force(cdof_in[w, i], g)
    h_out[w, i] = h
    k_out[w, i] = -math.motion_cross_force(velocity, g)
    hsum += h
  return hsum


@wp.kernel(enable_backward=False)
def _comvel_vjp_local(
  # Model:
  body_parentid: wp.array[int],
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_cdof_dot: wp.array2d[wp.spatial_vector],
  # Out:
  h_out: wp.array2d[wp.spatial_vector],
  k_out: wp.array2d[wp.spatial_vector],
  H_out: wp.array2d[wp.spatial_vector],
):
  """Computes per-dof pre-joint velocity (h) and cdof (k) cotangents, and per-body sums of h (H)."""
  w, b = wp.tid()
  if b == 0:  # The world body has no joints.
    H_out[w, 0] = wp.spatial_vector()
    return
  Hb = wp.spatial_vector()
  u = cvel_in[w, body_parentid[b]]  # Velocity before the body's first joint; zero for a tree root.
  jntadr = body_jntadr[b]
  for j in range(body_jntnum[b]):
    jnttype = jnt_type[jntadr + j]
    dofadr = jnt_dofadr[jntadr + j]
    if jnttype == JointType.FREE:
      for axis in range(3):
        i = dofadr + axis
        u += cdof_in[w, i] * qvel_in[w, i]
        h_out[w, i] = wp.spatial_vector()
        k_out[w, i] = wp.spatial_vector()
      Hb += _comvel_rot3_vjp(cdof_in, adj_cdof_dot, w, dofadr + 3, u, h_out, k_out)
      for axis in range(3):
        i = dofadr + 3 + axis
        u += cdof_in[w, i] * qvel_in[w, i]
    elif jnttype == JointType.BALL:
      Hb += _comvel_rot3_vjp(cdof_in, adj_cdof_dot, w, dofadr, u, h_out, k_out)
      for axis in range(3):
        i = dofadr + axis
        u += cdof_in[w, i] * qvel_in[w, i]
    else:
      g0 = adj_cdof_dot[w, dofadr]
      h0 = math.motion_cross_force(cdof_in[w, dofadr], g0)
      h_out[w, dofadr] = h0
      k_out[w, dofadr] = -math.motion_cross_force(u, g0)
      u += cdof_in[w, dofadr] * qvel_in[w, dofadr]
      Hb += h0
  H_out[w, b] = Hb


@wp.kernel(enable_backward=False)
def _comvel_vjp_samebody(
  # Model:
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  h_in: wp.array2d[wp.spatial_vector],
  k_in: wp.array2d[wp.spatial_vector],
  W_in: wp.array2d[wp.spatial_vector],
  # Out:
  adj_qvel_out: wp.array2d[float],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
):
  """Reverse-scans each body's joints, scattering each dof before adding its own h."""
  w, b = wp.tid()
  t = W_in[w, b]
  jntadr = body_jntadr[b]
  jntnum = body_jntnum[b]
  for j in range(jntnum):
    jntid = jntadr + jntnum - 1 - j  # Reverse joint order.
    jnttype = jnt_type[jntid]
    dofadr = jnt_dofadr[jntid]
    if jnttype == JointType.FREE:
      _cv_scatter3(qvel_in, cdof_in, w, dofadr + 3, t, adj_qvel_out, adj_cdof_out)
      for axis in range(3):
        i = dofadr + 5 - axis
        t += h_in[w, i]
        adj_cdof_out[w, i] += k_in[w, i]
      _cv_scatter3(qvel_in, cdof_in, w, dofadr, t, adj_qvel_out, adj_cdof_out)
    elif jnttype == JointType.BALL:
      _cv_scatter3(qvel_in, cdof_in, w, dofadr, t, adj_qvel_out, adj_cdof_out)
      for axis in range(3):
        i = dofadr + 2 - axis
        t += h_in[w, i]
        adj_cdof_out[w, i] += k_in[w, i]
    else:
      _cv_scatter(qvel_in, cdof_in, w, dofadr, t, adj_qvel_out, adj_cdof_out)
      t += h_in[w, dofadr]
      adj_cdof_out[w, dofadr] = adj_cdof_out[w, dofadr] + k_in[w, dofadr]


def comvel_backward(
  m: Model,
  d: Data,
  adj_cvel: wp.array2d,
  adj_cdof_dot: wp.array2d,
  bc: BackwardContext | None = None,
):
  """Accumulates the com-velocity reverse in the backward workspace."""
  standalone = bc is None
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  nworld, nbody = d.nworld, m.nbody
  adj_cdof = bc.smooth_cv_adj_cdof if standalone else bc.adj_cdof
  if standalone:
    adj_cdof.zero_()
  bc.smooth_cv_adj_qvel.zero_()
  wp.launch(
    _comvel_vjp_local,
    dim=(nworld, nbody),
    inputs=[m.body_parentid, m.body_jntnum, m.body_jntadr, m.jnt_type, m.jnt_dofadr, d.qvel, d.cdof, d.cvel, adj_cdof_dot],
    outputs=[bc.smooth_h, bc.smooth_k, bc.smooth_H],
  )
  wp.launch(
    _comvel_W_sum,
    dim=(nworld, nbody),
    inputs=[m.body_subtreenum, adj_cvel, bc.smooth_H],
    outputs=[bc.smooth_W],
  )
  wp.launch(
    _comvel_vjp_samebody,
    dim=(nworld, nbody),
    inputs=[m.body_jntnum, m.body_jntadr, m.jnt_type, m.jnt_dofadr, d.qvel, d.cdof, bc.smooth_h, bc.smooth_k, bc.smooth_W],
    outputs=[bc.smooth_cv_adj_qvel, adj_cdof],
  )
  return {"qvel": bc.smooth_cv_adj_qvel, "cdof": adj_cdof}
