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

import dataclasses
import weakref

import numpy as np
import warp as wp

from mujoco_warp._src import adjoint_recompute
from mujoco_warp._src import adjoint_util
from mujoco_warp._src import math
from mujoco_warp._src import smooth
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import support
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import vec10
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

# Adjoint reductions use unordered atomics; deterministic backward execution is deferred.
wp.set_module_options({"enable_backward": True})


# Cache by identity without retaining models or their device buffers.
_SUPPORTED_CACHE: dict[int, tuple[weakref.ReferenceType[Model], frozenset[str]]] = {}


def _model_support(m: Model) -> frozenset[str]:
  """Validates smooth features and caches the model's fixed parameter limitations."""
  if m.has_fluid:
    raise NotImplementedError("smooth_adjoint does not support fluid-capable models")
  key = id(m)
  cached = _SUPPORTED_CACHE.get(key)
  if cached is not None and cached[0]() is m:
    return cached[1]

  bad = []
  unsupported_parameters = set()
  if (
    m.nu != m.nactuator
    or np.any(m.actuator_ctrlnum.numpy() != 1)
    or not np.array_equal(m.actuator_ctrladr.numpy(), np.arange(m.nactuator))
  ):
    bad.append("non-identity actuator/control mapping (requires one control per actuator in actuator order)")
  if m.ntendon:
    bad.append("tendons (spring/bias/armature/transmission)")
  if m.nplastic:
    bad.append("plastic joint state")
  if m.flg_adhesion:
    bad.append("passive adhesion (m.flg_adhesion: qpos-dependent qfrc_adhesion has no VJP leaf)")
  if np.any(m.body_gravcomp.numpy() != 0.0) and np.any(m.jnt_actgravcomp.numpy() != 0):
    bad.append("gravcomp routed to an actuator (jnt_actgravcomp; force-limit clamp); passive-bucket gravcomp is supported")
  if m.body_freeadr.size:
    dofadr = m.body_dofadr.numpy()[m.body_freeadr.numpy()]
    rowadr, rownnz, colind = m.M_rowadr.numpy(), m.M_rownnz.numpy(), m.M_colind.numpy()
    for adr in dofadr:
      for row in range(adr + 3, adr + 6):
        columns = colind[rowadr[row] : rowadr[row] + rownnz[row]]
        if not np.isin(np.arange(adr, adr + 3), columns).all():
          unsupported_parameters.add("body_ipos")
        if not np.isin(np.arange(adr + 3, row + 1), columns).all():
          unsupported_parameters.add("body_iquat")
  if m.nu:
    trntype = m.actuator_trntype.numpy()
    gaintype = m.actuator_gaintype.numpy()
    biastype = m.actuator_biastype.numpy()
    dyntype = m.actuator_dyntype.numpy()
    stateful = m.actuator_actadr.numpy() >= 0
    jnt_type = m.jnt_type.numpy()
    if np.any(trntype != TrnType.JOINT):
      bad.append("non-JOINT actuator transmission (tendon/site/body/slider-crank/jointinparent)")
    if (
      not ((gaintype == GainType.FIXED) | (gaintype == GainType.AFFINE)).all()
      or not ((biastype == BiasType.NONE) | (biastype == BiasType.AFFINE)).all()
    ):
      bad.append("non-affine actuator gain/bias (muscle/DC-motor/user)")
    jnt_ids = m.actuator_trnid.numpy()[:, 0][trntype == TrnType.JOINT]
    if jnt_ids.size and np.any((jnt_type[jnt_ids] == JointType.FREE) | (jnt_type[jnt_ids] == JointType.BALL)):
      bad.append("actuator on a FREE/BALL joint (quaternion-dependent transmission length)")
    if np.any(stateful & (gaintype == GainType.AFFINE)):
      bad.append("stateful (na>0) AFFINE-gain actuator (ctrl_act=act not implemented)")
    scalar_dynamics = np.isin(dyntype, (DynType.INTEGRATOR, DynType.FILTER, DynType.FILTEREXACT, DynType.MUSCLE))
    if np.any(stateful & (~scalar_dynamics | (m.actuator_actnum.numpy() != 1))):
      bad.append("non-scalar or user/DC-motor activation dynamics")
    if m.nhistory and np.any(m.actuator_history.numpy()[..., 0] > 0):
      bad.append("actuator control history/delay")
  if bad:
    raise NotImplementedError(
      "smooth_adjoint analytic reverse does not support: "
      + "; ".join(bad)
      + ". Implement the leaf/topology VJP (the FD oracle lives in adjoint_test_util). "
      "There is no silent FD fallback."
    )
  limitations = frozenset(unsupported_parameters)
  _SUPPORTED_CACHE[key] = (weakref.ref(m, lambda _: _SUPPORTED_CACHE.pop(key, None)), limitations)
  return limitations


def assert_smooth_supported(m: Model):
  """Rejects unsupported model capabilities and requested parameter gradients."""
  # Gradient requests can change; cached parameter-layout limitations do not.
  unsupported = sorted(name for name in _model_support(m) if getattr(m, name).requires_grad)
  if unsupported:
    raise NotImplementedError(
      "smooth_adjoint analytic reverse does not support: "
      + "; ".join(f"{name} gradients with a compiled free-body mass layout missing couplings" for name in unsupported)
    )


@event_scope
def spring_qpos_vjp(m: Model, d: Data, lam: wp.array2d, bc: BackwardContext | None = None):
  """Returns d(lam.T @ -qfrc_spring)/dqpos."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  smooth_res_qpos = bc.smooth_res_qpos
  qfrc_spring = d.qfrc_spring
  smooth_res_qpos.zero_()
  spring_params = (m.qpos_spring, m.jnt_stiffness, m.jnt_stiffnesspoly)
  # Model gradients accumulate directly with the force sign; only qpos returns a residual VJP.
  wp.copy(bc.adj_qfrc_spring, lam)
  wp.launch(
    adjoint_recompute._spring_qfrc_recompute,
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
      any(param.requires_grad for param in spring_params),
    ],
    outputs=[qfrc_spring],
    adj_inputs=[
      None,
      m.qpos_spring.grad,
      None,
      None,
      None,
      m.jnt_stiffness.grad,
      m.jnt_stiffnesspoly.grad,
      smooth_res_qpos,
      None,
    ],
    adj_outputs=[bc.adj_qfrc_spring],
    adjoint=True,
  )
  wp.launch(adjoint_util._neg_cols, dim=smooth_res_qpos.shape, inputs=[smooth_res_qpos], outputs=[smooth_res_qpos])
  return smooth_res_qpos


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


@wp.kernel(enable_backward=False)
def _gravcomp_seed(
  # Model:
  jnt_actgravcomp: wp.array[int],
  dof_jntid: wp.array[int],
  # In:
  lam: wp.array2d[float],
  gravity_enabled: int,
  # Out:
  grad_out: wp.array2d[float],
):
  w, i = wp.tid()
  if gravity_enabled and not jnt_actgravcomp[dof_jntid[i]]:
    grad_out[w, i] = -lam[w, i]
  else:
    grad_out[w, i] = 0.0


@event_scope
def gravcomp_qpos_vjp(m: Model, d: Data, lam: wp.array2d, bc: BackwardContext | None = None):
  """Returns d(lam.T @ -qfrc_gravcomp)/dqpos.

  Leaves body-mass and inertial-frame cotangents for parameter accumulation.
  """
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  bc.adj_xipos.zero_()
  bc.adj_ximat.zero_()
  bc.adj_subtree_com.zero_()
  bc.adj_cdof.zero_()
  bc.adj_qfrc_gravcomp.zero_()
  bc.res_body_mass.zero_()
  bc.res_body_inertia.zero_()
  bc.smooth_res_qpos.zero_()

  passive_disabled = (m.opt.disableflags & (DisableBit.SPRING | DisableBit.DAMPER)) == (DisableBit.SPRING | DisableBit.DAMPER)
  gravity_enabled = int(not (m.opt.disableflags & DisableBit.GRAVITY) and not passive_disabled)
  wp.launch(
    _gravcomp_seed,
    dim=(d.nworld, m.nv),
    inputs=[m.jnt_actgravcomp, m.dof_jntid, lam, gravity_enabled],
    outputs=[bc.adj_qfrc_gravcomp],
  )
  wp.launch(
    adjoint_recompute._gravity_force_recompute,
    dim=(d.nworld, m.nbody - 1, m.nv),
    inputs=[
      m.opt.gravity,
      m.body_parentid,
      m.body_rootid,
      m.body_mass,
      m.body_gravcomp,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.subtree_com,
      d.cdof,
    ],
    outputs=[d.qfrc_gravcomp],
    adj_inputs=[None, None, None, bc.res_body_mass, None, None, None, bc.adj_xipos, bc.adj_subtree_com, bc.adj_cdof],
    adj_outputs=[bc.adj_qfrc_gravcomp],
    adjoint=True,
  )
  return _kinematic_qpos_vjp(m, d, bc.adj_cdof, bc)


# Transposes of the linear RNE tree reductions. Each output reads the original seeds,
# so an entire ancestor/subtree sum needs one launch and no inter-depth synchronization.
@wp.kernel(enable_backward=False)
def _ancestor_sum(
  # Model:
  body_parentid: wp.array[int],
  # In:
  val_in: wp.array2d[wp.spatial_vector],
  # Out:
  sum_out: wp.array2d[wp.spatial_vector],
):
  w, b = wp.tid()
  acc = val_in[w, b]
  parent = b
  while parent != 0:
    parent = body_parentid[parent]
    acc += val_in[w, parent]
  sum_out[w, b] = acc


@wp.kernel(enable_backward=False)
def _subtree_sum(
  # Model:
  body_subtreenum: wp.array[int],
  # In:
  val_in: wp.array2d[wp.spatial_vector],
  # Out:
  sum_out: wp.array2d[wp.spatial_vector],
):
  w, b = wp.tid()
  acc = val_in[w, b]
  for child in range(b + 1, b + body_subtreenum[b]):
    acc += val_in[w, child]
  sum_out[w, b] = acc


# io_out += src (merges a contact subtree-COM seed into the bias's)
@wp.kernel(enable_backward=False)
def _acc_vec3(src: wp.array2d[wp.vec3], io_out: wp.array2d[wp.vec3]):
  w, i = wp.tid()
  io_out[w, i] = io_out[w, i] + src[w, i]


@wp.kernel(enable_backward=False)
def _acc_spatial(src: wp.array2d[wp.spatial_vector], io_out: wp.array2d[wp.spatial_vector]):
  w, i = wp.tid()
  io_out[w, i] = io_out[w, i] + src[w, i]


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


def linearize(m: Model, d: Data, d_out: Data, bc: BackwardContext) -> Data:
  """Replays smooth dynamics at the step input state."""
  view = dataclasses.replace(
    d_out,
    qpos=d.qpos,
    qvel=d.qvel,
    ctrl=d.ctrl,
    act=d.act,
    cacc=bc.scratch.cacc,
    cfrc_int=bc.scratch.cfrc_int,
    qfrc_bias=bc.scratch.qfrc_bias,
  )
  smooth.rne(m, view, flg_acc=True)
  return view


# ----------------------------------------------------------------------------
# The shared reduced reverse.
# ----------------------------------------------------------------------------
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


@event_scope
def smooth_force_backward(
  m: Model,
  d: Data,
  lam: wp.array2d,
  flg_acc: bool = True,
  res_cdof_extra: wp.array2d = None,
  res_subtree_extra: wp.array2d = None,
  bc: BackwardContext | None = None,
  xfrc_applied: wp.array2d | None = None,
):
  """Returns the position VJP and leaves RNE/com_vel velocity cotangents in ``bc``."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  nworld, nv = d.nworld, m.nv
  rne_backward(m, d, lam, flg_acc, bc=bc)
  comvel_backward(m, d, bc.adj_cvel, bc.adj_cdof_dot, bc=bc)
  if res_cdof_extra is not None:
    wp.launch(_acc_spatial, dim=(nworld, nv), inputs=[res_cdof_extra], outputs=[bc.adj_cdof])

  bc.adj_xipos.zero_()
  bc.adj_ximat.zero_()
  bc.adj_subtree_com.zero_()
  _cinert_vjp(m, d, bc.adj_cinert, bc)
  if res_subtree_extra is not None:
    wp.launch(_acc_vec3, dim=(nworld, m.nbody), inputs=[res_subtree_extra], outputs=[bc.adj_subtree_com])
  if xfrc_applied is not None:
    xfrc_kinematics_backward(m, d, lam, xfrc_applied, bc)
  return _kinematic_qpos_vjp(m, d, bc.adj_cdof, bc)


@wp.func
def _rne_force_objective(ci: vec10, cv: wp.spatial_vector, ca: wp.spatial_vector, seed: wp.spatial_vector):
  return wp.dot(smooth.rne_body_force(ci, cv, ca), seed)


@wp.func
def _rne_channels_objective(
  # In:
  ci: vec10,
  cv: wp.spatial_vector,
  ca: wp.spatial_vector,
  mass_ca: wp.spatial_vector,
  seed: wp.spatial_vector,
  mass_seed: wp.spatial_vector,
):
  return wp.dot(smooth.rne_body_force(ci, cv, ca), seed) + wp.dot(math.inert_vec(ci, mass_ca), mass_seed)


@cache_kernel
def _rne_channels(nbody: int, with_mass: bool):
  tile_size = max(32, 1 << (nbody - 1).bit_length())

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    opt_gravity: wp.array[wp.vec3],
    body_parentid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    dof_bodyid: wp.array[int],
    body_subtreenum: wp.array[int],
    # Data in:
    qvel_in: wp.array2d[float],
    qacc_in: wp.array2d[float],
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
    adj_qacc_out: wp.array2d[float],
    adj_cdof_out: wp.array2d[wp.spatial_vector],
    adj_cdof_dot_out: wp.array2d[wp.spatial_vector],
    adj_cvel_out: wp.array2d[wp.spatial_vector],
    adj_cinert_out: wp.array2d[vec10],
  ):
    w, lane = wp.tid()
    local_ca = wp.spatial_vector()
    local_seed = wp.spatial_vector()
    local_mass_ca = wp.spatial_vector()
    local_mass_seed = wp.spatial_vector()
    if lane > 0 and lane < wp.static(nbody):
      start = body_dofadr[lane]
      for i in range(start, start + body_dofnum[lane]):
        S = cdof_in[w, i]
        local_ca += cdof_dot_in[w, i] * qvel_in[w, i] + S * qacc_in[w, i]
        local_seed += lam[w, i] * S
        if wp.static(with_mass):
          local_mass_ca += mass_w[w, i] * S
          # Final input writeback subtracts the residual cotangent.
          local_mass_seed -= mass_y[w, i] * S
    ca_local = wp.tile_reshape(wp.tile(local_ca, preserve_type=True), shape=(wp.static(tile_size),))
    seed_local = wp.tile_reshape(wp.tile(local_seed, preserve_type=True), shape=(wp.static(tile_size),))
    if wp.static(with_mass):
      mass_ca_local = wp.tile_reshape(wp.tile(local_mass_ca, preserve_type=True), shape=(wp.static(tile_size),))
      mass_seed_local = wp.tile_reshape(wp.tile(local_mass_seed, preserve_type=True), shape=(wp.static(tile_size),))

    force = wp.spatial_vector()
    mass_force = wp.spatial_vector()
    ca_bar = wp.spatial_vector()
    mass_ca_bar = wp.spatial_vector()
    if lane > 0 and lane < wp.static(nbody):
      ca = wp.spatial_vector()
      seed = wp.spatial_vector()
      mass_ca = wp.spatial_vector()
      mass_seed = wp.spatial_vector()
      b = lane
      while b > 0:
        ca += ca_local[b]
        seed += seed_local[b]
        if wp.static(with_mass):
          mass_ca += mass_ca_local[b]
          mass_seed += mass_seed_local[b]
        b = body_parentid[b]
      if gravity_enabled:
        gravity = opt_gravity[w % opt_gravity.shape[0]]
        ca -= wp.spatial_vector(0.0, 0.0, 0.0, gravity[0], gravity[1], gravity[2])
      ci = cinert_in[w, lane]
      cv = cvel_in[w, lane]
      # Seed derivatives also recover the primal forces for the DOF transpose.
      if wp.static(with_mass):
        ci_bar, cv_bar, ca_bar, mass_ca_bar, force, mass_force = wp.grad(_rne_channels_objective)(
          ci, cv, ca, mass_ca, seed, mass_seed
        )
      else:
        ci_bar, cv_bar, ca_bar, force = wp.grad(_rne_force_objective)(ci, cv, ca, seed)
      adj_cinert_out[w, lane] = ci_bar
      adj_cvel_out[w, lane] = cv_bar
    elif lane == 0:
      adj_cinert_out[w, 0] = vec10()
      adj_cvel_out[w, 0] = wp.spatial_vector()
    force_tile = wp.tile_reshape(wp.tile(force, preserve_type=True), shape=(wp.static(tile_size),))
    ca_bar_tile = wp.tile_reshape(wp.tile(ca_bar, preserve_type=True), shape=(wp.static(tile_size),))
    if wp.static(with_mass):
      mass_force_tile = wp.tile_reshape(wp.tile(mass_force, preserve_type=True), shape=(wp.static(tile_size),))
      mass_ca_bar_tile = wp.tile_reshape(wp.tile(mass_ca_bar, preserve_type=True), shape=(wp.static(tile_size),))

    for i in range(lane, nv, wp.block_dim()):
      body = dof_bodyid[i]
      acc_force = wp.spatial_vector()
      acc_ca_bar = wp.spatial_vector()
      acc_mass_force = wp.spatial_vector()
      acc_mass_ca_bar = wp.spatial_vector()
      for b in range(body, body + body_subtreenum[body]):
        acc_force += force_tile[b]
        acc_ca_bar += ca_bar_tile[b]
        if wp.static(with_mass):
          acc_mass_force += mass_force_tile[b]
          acc_mass_ca_bar += mass_ca_bar_tile[b]
      adj_qvel_out[w, i] = wp.dot(acc_ca_bar, cdof_dot_in[w, i])
      adj_qacc_out[w, i] = wp.dot(acc_ca_bar, cdof_in[w, i])
      adj_cdof_dot_out[w, i] = qvel_in[w, i] * acc_ca_bar
      S_bar = lam[w, i] * acc_force + qacc_in[w, i] * acc_ca_bar
      if wp.static(with_mass):
        S_bar += -mass_y[w, i] * acc_mass_force + mass_w[w, i] * acc_mass_ca_bar
      adj_cdof_out[w, i] = S_bar

  return kernel


@event_scope
def smooth_force_channels_backward(
  m: Model,
  d: Data,
  lam: wp.array2d,
  mass_y: wp.array2d | None,
  mass_w: wp.array2d | None,
  bc: BackwardContext,
  xfrc_applied: wp.array2d | None = None,
):
  """Combines force and optional direct-mass cotangents before their kinematic reverse.

  The mass channel contributes with a negative sign because the caller subtracts this residual
  VJP. All intermediates are recomputed from the frozen input kinematics; no forward tape grows.
  """
  block_dim = max(32, 1 << (m.nbody - 1).bit_length())
  wp.launch_tiled(
    _rne_channels(m.nbody, mass_y is not None),
    dim=d.nworld,
    block_dim=block_dim,
    inputs=[
      m.nv,
      m.opt.gravity,
      m.body_parentid,
      m.body_dofnum,
      m.body_dofadr,
      m.dof_bodyid,
      m.body_subtreenum,
      d.qvel,
      d.qacc,
      d.cdof,
      d.cinert,
      d.cvel,
      d.cdof_dot,
      not bool(m.opt.disableflags & DisableBit.GRAVITY),
      lam,
      mass_y,
      mass_w,
    ],
    outputs=[bc.smooth_adj_qvel, bc.smooth_adj_qacc, bc.adj_cdof, bc.adj_cdof_dot, bc.adj_cvel, bc.adj_cinert],
  )
  comvel_backward(m, d, bc.adj_cvel, bc.adj_cdof_dot, bc=bc)
  bc.adj_xipos.zero_()
  bc.adj_ximat.zero_()
  bc.adj_subtree_com.zero_()
  _cinert_vjp(m, d, bc.adj_cinert, bc)
  if xfrc_applied is not None:
    xfrc_kinematics_backward(m, d, lam, xfrc_applied, bc)
  return _kinematic_qpos_vjp(m, d, bc.adj_cdof, bc)


@event_scope
def mass_matrix_qpos_vjp(m: Model, d: Data, y: wp.array2d, w: wp.array2d, bc: BackwardContext | None = None):
  """Returns d(y.T @ M(q) @ w)/dqpos with y and w fixed."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  scratch = bc.scratch
  wp.copy(scratch.qpos, d.qpos)
  if m.nmocap:
    wp.copy(scratch.mocap_pos, d.mocap_pos)
    wp.copy(scratch.mocap_quat, d.mocap_quat)
  scratch.qvel.zero_()
  scratch.qacc = w
  smooth.kinematics(m, scratch)
  smooth.com_pos(m, scratch)
  scratch.cvel.zero_()
  scratch.cdof_dot.zero_()
  scratch.cacc.zero_()
  smooth._rne_cacc_forward(m, scratch, flg_acc=True)
  smooth._rne_cfrc(m, scratch)
  return smooth_force_backward(m, scratch, y, bc=bc)


# ============================================================================================
# AD-RNE: analytic backward of smooth.rne, the production dqpos path in step_backward.
# FD-of-rne is an explicitly selected validation oracle in adjoint_test_util. smooth.rne is
# enable_backward=False, so the reverse is reconstructed here: source-AD the loop-free/
# alias-free nonlinear _cfrc leaf (exact), manual transposed VJPs for the two linear tree
# reductions (their forwards have dynamic loops + in-place aliasing, so a naive reverse is wrong).
# ============================================================================================


# d(lam.T @ qfrc_bias)/dF_b = sum(lam_i * cdof_i) for the body's DOFs.
@wp.kernel(enable_backward=False)
def _rne_qfrcbias_force_vjp(
  # Model:
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  lam: wp.array2d[float],  # seed adj_qfrc_bias
  # Out:
  adj_force_out: wp.array2d[wp.spatial_vector],  # adj on the accumulated body force F_b
):
  w, b = wp.tid()
  dofadr = body_dofadr[b]
  dofnum = body_dofnum[b]
  acc = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for k in range(dofnum):
    i = dofadr + k
    acc = acc + lam[w, i] * cdof_in[w, i]
  adj_force_out[w, b] = acc


@wp.kernel(enable_backward=False)
def _rne_dof_vjp(
  # Model:
  dof_bodyid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  cdof_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  cfrc_int_in: wp.array2d[wp.spatial_vector],
  # In:
  lam: wp.array2d[float],
  subtree_adj_cacc: wp.array2d[wp.spatial_vector],  # sum adj_cacc over body(j)'s subtree (from K4a)
  flg_acc: bool,
  # Out:
  adj_qvel_out: wp.array2d[float],
  adj_qacc_out: wp.array2d[float],
  adj_cdof_dot_out: wp.array2d[wp.spatial_vector],
  adj_cdof_out: wp.array2d[wp.spatial_vector],
):
  w, i = wp.tid()
  bodyid = dof_bodyid[i]
  a = subtree_adj_cacc[w, bodyid]
  adj_qvel_out[w, i] = wp.dot(a, cdof_dot_in[w, i])
  adj_cdof_dot_out[w, i] = qvel_in[w, i] * a
  adj_cdof = lam[w, i] * cfrc_int_in[w, bodyid]
  if flg_acc:
    adj_qacc_out[w, i] = wp.dot(a, cdof_in[w, i])
    adj_cdof = adj_cdof + qacc_in[w, i] * a
  else:
    adj_qacc_out[w, i] = 0.0
  adj_cdof_out[w, i] = adj_cdof


def rne_backward(
  m: Model,
  d: Data,
  lam: wp.array2d,
  flg_acc: bool = True,
  bc: BackwardContext | None = None,
):
  """Reverses RNE and returns views of its input cotangents."""
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)
  nworld, nv, nbody = d.nworld, m.nv, m.nbody
  bc.adj_cinert.zero_()
  bc.adj_cvel.zero_()
  bc.adj_cacc.zero_()
  wp.launch(
    _rne_qfrcbias_force_vjp,
    dim=(nworld, nbody),
    inputs=[m.body_dofnum, m.body_dofadr, d.cdof, lam],
    outputs=[bc.smooth_force],
  )

  wp.launch(
    _ancestor_sum,
    dim=(nworld, nbody),
    inputs=[m.body_parentid, bc.smooth_force],
    outputs=[bc.smooth_tree],
  )

  wp.launch(
    adjoint_recompute._rne_cfrc_recompute,
    dim=(nworld, nbody),
    inputs=[d.cinert, d.cvel, d.cacc],
    outputs=[bc.smooth_cfrc],
    adj_inputs=[bc.adj_cinert, bc.adj_cvel, bc.adj_cacc],
    adj_outputs=[bc.smooth_tree],
    adjoint=True,
  )

  wp.launch(
    _subtree_sum,
    dim=(nworld, nbody),
    inputs=[m.body_subtreenum, bc.adj_cacc],
    outputs=[bc.smooth_tree],
  )
  wp.launch(
    _rne_dof_vjp,
    dim=(nworld, nv),
    inputs=[m.dof_bodyid, d.qvel, d.qacc, d.cdof, d.cdof_dot, d.cfrc_int, lam, bc.smooth_tree, flg_acc],
    outputs=[bc.smooth_adj_qvel, bc.smooth_adj_qacc, bc.adj_cdof_dot, bc.adj_cdof],
  )
  return {
    "qvel": bc.smooth_adj_qvel,
    "qacc": bc.smooth_adj_qacc,
    "cdof": bc.adj_cdof,
    "cdof_dot": bc.adj_cdof_dot,
    "cinert": bc.adj_cinert,
    "cvel": bc.adj_cvel,
  }


# --------------------------------------------------------------------------------------------
# com_vel reverse: bound-free manual VJP of smooth.com_vel (the RNE qvel path); maps the seeds
# adj_cvel / adj_cdof_dot to adj_qvel / adj_cdof in stages CV1-CV4 (dynamic bounds, no
# _CV_MAX_* truncation). step_backward adds the RNE and com_vel velocity cotangents to the
# residual VJP, reusing this reverse instead of constructing the RNE velocity Jacobian.
# --------------------------------------------------------------------------------------------


# same-body scatter of a running prefix cotangent t: nubar_j += S_j*t, Sbar_j += nu_j*t
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
  hsum = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for axis in range(3):
    i = dofid + axis
    g = adj_cdof_dot[w, i]
    h = math.motion_cross_force(cdof_in[w, i], g)
    h_out[w, i] = h
    k_out[w, i] = (-1.0) * math.motion_cross_force(velocity, g)
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
  adj_cdof_dot: wp.array2d[wp.spatial_vector],  # G seed
  # Out:
  h_out: wp.array2d[wp.spatial_vector],  # per dof: cotangent on the snapshot velocity
  k_out: wp.array2d[wp.spatial_vector],  # per dof: direct cotangent on cdof_in
  H_out: wp.array2d[wp.spatial_vector],  # per body: sum h on this body
):
  w, b = wp.tid()
  zero = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  if b == 0:  # world body: no joints / snapshot
    H_out[w, 0] = zero
    return
  Hb = zero
  u = cvel_in[w, body_parentid[b]]  # parent snapshot (cvel_in[0]=0 for a tree root)
  jntadr = body_jntadr[b]
  jntnum = body_jntnum[b]
  for jj in range(jntnum):
    jt = jnt_type[jntadr + jj]
    d = jnt_dofadr[jntadr + jj]
    if jt == JointType.FREE:
      for axis in range(3):
        i = d + axis
        u += cdof_in[w, i] * qvel_in[w, i]
        h_out[w, i] = zero
        k_out[w, i] = zero
      Hb += _comvel_rot3_vjp(cdof_in, adj_cdof_dot, w, d + 3, u, h_out, k_out)
      for axis in range(3):
        i = d + 3 + axis
        u += cdof_in[w, i] * qvel_in[w, i]
    elif jt == JointType.BALL:
      Hb += _comvel_rot3_vjp(cdof_in, adj_cdof_dot, w, d, u, h_out, k_out)
      for axis in range(3):
        i = d + axis
        u += cdof_in[w, i] * qvel_in[w, i]
    else:
      g0 = adj_cdof_dot[w, d]
      h0 = math.motion_cross_force(cdof_in[w, d], g0)
      h_out[w, d] = h0
      k_out[w, d] = (-1.0) * math.motion_cross_force(u, g0)
      u = u + cdof_in[w, d] * qvel_in[w, d]
      Hb = Hb + h0
  H_out[w, b] = Hb


# CV2: same-body reverse suffix scan; scatter precedes T+=h so no axis sees its own snapshot
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
  adj_qvel_out: wp.array2d[float],  # (+=): same-body T_j part
  adj_cdof_out: wp.array2d[wp.spatial_vector],  # (+=): same-body T_j part + direct k_j
):
  w, b = wp.tid()
  t = W_in[w, b]
  jntadr = body_jntadr[b]
  jntnum = body_jntnum[b]
  for jr in range(jntnum):
    jj = jntnum - 1 - jr  # reverse joint order
    jt = jnt_type[jntadr + jj]
    d = jnt_dofadr[jntadr + jj]
    if jt == JointType.FREE:
      _cv_scatter3(qvel_in, cdof_in, w, d + 3, t, adj_qvel_out, adj_cdof_out)
      for axis in range(3):
        i = d + 5 - axis
        t += h_in[w, i]
        adj_cdof_out[w, i] += k_in[w, i]
      _cv_scatter3(qvel_in, cdof_in, w, d, t, adj_qvel_out, adj_cdof_out)
    elif jt == JointType.BALL:
      _cv_scatter3(qvel_in, cdof_in, w, d, t, adj_qvel_out, adj_cdof_out)
      for axis in range(3):
        i = d + 2 - axis
        t += h_in[w, i]
        adj_cdof_out[w, i] += k_in[w, i]
    else:
      _cv_scatter(qvel_in, cdof_in, w, d, t, adj_qvel_out, adj_cdof_out)
      t = t + h_in[w, d]
      adj_cdof_out[w, d] = adj_cdof_out[w, d] + k_in[w, d]


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


# --------------------------------------------------------------------------------------------
# cinert reverse: the last kinematic dqpos leaf. cinert depends on qpos via ximat / xipos /
# subtree_com[root]: source-AD the body-local _cinert leaf, chain xipos/ximat -> qpos via
# support.jac_dof, route adj_subtree_com through _subtree_com_qpos_vjp. cvel/cdof_dot have no
# direct qpos path (it flows through the total adj_cdof via _cdof_qpos_vjp).
# --------------------------------------------------------------------------------------------


# chain adj_{xipos_in, ximat_in} to a per-dof tangent gradient via support.jac_dof
@wp.kernel(module="unique", enable_backward=False)
def _cinert_pose_dof_vjp(
  # Model:
  nbody: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  lanes: int,
  adj_xipos: wp.array2d[wp.vec3],
  adj_ximat: wp.array2d[wp.mat33],
  # Out:
  res_dof_out: wp.array2d[float],  # (+=): per-dof tangent gradient
):
  w, k, lane = wp.tid()
  acc = float(0.0)
  for b in range(1 + lane, nbody, lanes):
    if body_isdofancestor[b, k] == 0:  # dof k does not move body b -> jac = 0
      continue
    jacp, jacr = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      xipos_in[w, b],
      b,
      k,
      w,
    )
    tau = adjoint_util._adj_rotation(ximat_in[w, b], adj_ximat[w, b])
    acc += wp.dot(jacp, adj_xipos[w, b]) + wp.dot(jacr, tau)
  wp.atomic_add(res_dof_out, w, k, acc)  # kernel_analyzer: ignore[atomic]


def _cinert_vjp(m: Model, d: Data, adj_cinert: wp.array2d, bc: BackwardContext):
  # Explicit sinks prevent Warp from falling back to the model parameters' own .grad arrays.
  bc.res_body_mass.zero_()
  bc.res_body_inertia.zero_()
  wp.launch(
    adjoint_recompute._cinert_recompute,
    dim=(d.nworld, m.nbody),
    inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com],
    outputs=[d.cinert],
    adj_inputs=[None, bc.res_body_mass, bc.res_body_inertia, bc.adj_xipos, bc.adj_ximat, bc.adj_subtree_com],
    adj_outputs=[adj_cinert],
    adjoint=True,
  )


def _kinematic_qpos_vjp(m: Model, d: Data, adj_cdof: wp.array2d, bc: BackwardContext):
  nworld, nv = d.nworld, m.nv
  smooth_res_dof, smooth_res_qpos = bc.smooth_res_dof, bc.smooth_res_qpos
  smooth_res_dof.zero_()
  smooth_res_qpos.zero_()
  lanes = 8 if d.qpos.device.is_cuda else 1
  wp.launch(
    _cinert_pose_dof_vjp,
    dim=(nworld, nv, lanes),
    inputs=[
      m.nbody,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.ximat,
      d.subtree_com,
      d.cdof,
      lanes,
      bc.adj_xipos,
      bc.adj_ximat,
    ],
    outputs=[smooth_res_dof],
    block_dim=128,
  )
  smooth_kinematics_adjoint.kinematics_qpos_backward(
    m,
    d,
    d.qpos,
    adj_cdof,
    bc.adj_subtree_com,
    smooth_res_dof,
    bc.smooth_ceff,
    smooth_res_qpos,
  )
  return smooth_res_qpos
