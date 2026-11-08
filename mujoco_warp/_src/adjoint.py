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
"""Analytic backward for step() via hybrid analytic differentiability.

This modules uses implicit differentiation, auto-differentiation, and custom analytic
derivatives to compute analytic gradients. For the stationarity residual r(qacc, theta) = 0,
reverse mode solves H * lam = adj_qacc and accumulates adj_theta = -(dr/dtheta)^T * lam.
"""

import contextlib
import contextvars
import dataclasses
import weakref
from typing import Any

import numpy as np
import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import collision_adjoint
from mujoco_warp._src import collision_convex_adjoint
from mujoco_warp._src import constraint_adjoint
from mujoco_warp._src import forward
from mujoco_warp._src import forward_adjoint
from mujoco_warp._src import io
from mujoco_warp._src import math
from mujoco_warp._src import model_adjoint
from mujoco_warp._src import smooth
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import solver
from mujoco_warp._src.types import TILE_SIZE_JTDAJ_DENSE
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import SolverType
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope
from mujoco_warp._src.warp_util import is_array_spec

# These kernels differentiate through @wp.funcs defined in other modules.
wp.set_module_options({"enable_backward": True})

_ACTIVE_BACKWARD_CONTEXT: contextvars.ContextVar[BackwardContext | None] = contextvars.ContextVar(
  "mjw_active_backward_context",
  default=None,
)


@contextlib.contextmanager
def backward_context(bc: BackwardContext):
  """Uses ``bc`` for backward calls in this block."""
  token = _ACTIVE_BACKWARD_CONTEXT.set(bc)
  try:
    yield bc
  finally:
    _ACTIVE_BACKWARD_CONTEXT.reset(token)


def _reset_backward_context(bc: BackwardContext):
  """Zeros a backward context and its array gradients."""
  for field in dataclasses.fields(bc):
    value = getattr(bc, field.name)
    if isinstance(value, wp.array):
      value.zero_()
      if value.grad is not None:
        value.grad.zero_()


# Cache by identity without retaining models or their device buffers.
_SUPPORTED_CACHE: dict[int, tuple[weakref.ReferenceType[Model], frozenset[str]]] = {}


def _model_support(m: Model) -> tuple[tuple[str, ...], frozenset[str]]:
  """Inspects fixed smooth features and caches supported models' parameter limitations."""
  key = id(m)
  cached = _SUPPORTED_CACHE.get(key)
  if cached is not None and cached[0]() is m:
    return (), cached[1]

  reasons = []
  unsupported_parameters = set()
  if (
    m.nu != m.nactuator
    or np.any(m.actuator_ctrlnum.numpy() != 1)
    or not np.array_equal(m.actuator_ctrladr.numpy(), np.arange(m.nactuator))
  ):
    reasons.append("non-identity actuator/control mapping (requires one control per actuator in actuator order)")
  if m.ntendon:
    reasons.append("tendons (spring/bias/armature/transmission)")
  if m.nplastic:
    reasons.append("plastic joint state")
  if m.has_fluid:
    reasons.append("fluid-capable models")
  if m.flg_adhesion:
    reasons.append("passive adhesion (m.flg_adhesion: qpos-dependent qfrc_adhesion has no VJP leaf)")
  if np.any(m.body_gravcomp.numpy() != 0.0) and np.any(m.jnt_actgravcomp.numpy() != 0):
    reasons.append("gravcomp routed to an actuator (jnt_actgravcomp; force-limit clamp); passive-bucket gravcomp is supported")
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
      reasons.append("non-JOINT actuator transmission (tendon/site/body/slider-crank/jointinparent)")
    if (
      not ((gaintype == GainType.FIXED) | (gaintype == GainType.AFFINE)).all()
      or not ((biastype == BiasType.NONE) | (biastype == BiasType.AFFINE)).all()
    ):
      reasons.append("non-affine actuator gain/bias (muscle/DC-motor/user)")
    jnt_ids = m.actuator_trnid.numpy()[:, 0][trntype == TrnType.JOINT]
    if jnt_ids.size and np.any((jnt_type[jnt_ids] == JointType.FREE) | (jnt_type[jnt_ids] == JointType.BALL)):
      reasons.append("actuator on a FREE/BALL joint (quaternion-dependent transmission length)")
    if np.any(stateful & (gaintype == GainType.AFFINE)):
      reasons.append("stateful (na>0) AFFINE-gain actuator (ctrl_act=act not implemented)")
    scalar_dynamics = np.isin(dyntype, (DynType.INTEGRATOR, DynType.FILTER, DynType.FILTEREXACT, DynType.MUSCLE))
    if np.any(stateful & (~scalar_dynamics | (m.actuator_actnum.numpy() != 1))):
      reasons.append("non-scalar or user/DC-motor activation dynamics")
    if m.nhistory and np.any(m.actuator_history.numpy()[..., 0] > 0):
      reasons.append("actuator control history/delay")
  limitations = frozenset(unsupported_parameters)
  if not reasons:
    _SUPPORTED_CACHE[key] = (weakref.ref(m, lambda _: _SUPPORTED_CACHE.pop(key, None)), limitations)
  return tuple(reasons), limitations


def _assert_model_supported(m: Model):
  """Rejects unsupported model capabilities and requested parameter gradients."""
  reasons, limitations = _model_support(m)
  if reasons:
    raise NotImplementedError(
      "smooth_adjoint analytic reverse does not support: "
      + "; ".join(reasons)
      + ". Implement the leaf/topology VJP (the FD oracle lives in adjoint_test_util). "
      "There is no silent FD fallback."
    )
  # Gradient requests can change; cached parameter-layout limitations do not.
  unsupported = sorted(name for name in limitations if getattr(m, name).requires_grad)
  if unsupported:
    raise NotImplementedError(
      "smooth_adjoint analytic reverse does not support: "
      + "; ".join(f"{name} gradients with a compiled free-body mass layout missing couplings" for name in unsupported)
    )


def create_backward_context(m: Model, d: Data) -> BackwardContext:
  """Preallocates the reusable backward workspace for ``(m, d)``."""
  # Cache fixed model metadata before CUDA capture.
  _model_support(m)
  sizes = {field.name: getattr(m, field.name) for field in dataclasses.fields(Model) if field.type is int}
  sizes.update(nworld=d.nworld, njmax=d.efc.type.shape[1], naconmax=d.contact.pos.shape[0], qld_total=d.qLD.shape[1])
  arrays = {
    field.name: io._create_array(None, field.type, sizes)
    for field in dataclasses.fields(BackwardContext)
    if is_array_spec(field.type)
  }
  arrays.update(
    res_dof_frictionloss=wp.zeros_like(m.dof_frictionloss),
    res_dof_solref=wp.zeros_like(m.dof_solref),
    res_dof_solimp=wp.zeros_like(m.dof_solimp),
    res_eq_solref=wp.zeros_like(m.eq_solref),
    res_eq_solimp=wp.zeros_like(m.eq_solimp),
    res_jnt_solref=wp.zeros_like(m.jnt_solref),
    res_jnt_solimp=wp.zeros_like(m.jnt_solimp),
    res_body_mass=wp.zeros_like(m.body_mass),
    res_body_inertia=wp.zeros_like(m.body_inertia),
    res_body_ipos=wp.zeros_like(m.body_ipos),
    res_body_iquat=wp.zeros_like(m.body_iquat),
  )
  bc = BackwardContext(
    solver_ctx=solver._create_solver_context(m, d),
    scratch=adjoint_util._clone_nograd(d),
    collision_convex_ctx=collision_convex_adjoint.create_collision_convex_backward_context(m, d),
    **arrays,
  )
  for name in ("contact_V", "contact_A", "contact_Z", "contact_phi", "efc_Z", "efc_phi", "smooth_r"):
    getattr(bc, name).requires_grad = True
  return bc


def _assert_step_supported(m: Model):
  """Rejects configurations without a complete analytic backward."""
  if m.opt.solver != SolverType.NEWTON:
    raise NotImplementedError("adjoint.step_backward supports only the Newton solver")
  if m.opt.iterations <= 0:
    raise NotImplementedError("adjoint.step_backward requires solver iterations > 0")
  if m.opt.noslip_iterations > 0:
    raise NotImplementedError("adjoint.step_backward does not support the noslip solver")
  if m.opt.integrator not in (IntegratorType.EULER, IntegratorType.IMPLICITFAST):
    raise NotImplementedError("adjoint.step_backward supports only Euler and implicitfast")
  if m.nflex:
    raise NotImplementedError("adjoint.step_backward does not support flex contacts")
  if m.opt.cone not in (ConeType.ELLIPTIC, ConeType.PYRAMIDAL):
    raise NotImplementedError("adjoint.step_backward supports only elliptic and pyramidal cones")
  if m.flg_adhesion:
    raise NotImplementedError("adjoint.step_backward does not support adhesion")
  if m.flg_surfacevel:
    raise NotImplementedError("adjoint.step_backward does not support surface velocity")
  if not m.opt.run_collision_detection:
    raise NotImplementedError("adjoint.step_backward does not support caller-supplied contacts (run_collision_detection=False)")
  if m.opt.enableflags & EnableBit.SLEEP and not (m.opt.disableflags & DisableBit.ISLAND):
    raise NotImplementedError("adjoint.step_backward does not support sleep")
  if m.eq_connect_adr.size or m.eq_wld_adr.size or m.ntendon or m.nflex:
    raise NotImplementedError("adjoint.step_backward does not support connect/weld equality, tendon, or flex constraint rows")
  callbacks = []
  if not (m.opt.disableflags & DisableBit.ACTUATION):
    callbacks.append("control")
    if m.nactuator:
      callbacks.extend(("act_dyn", "act_gain", "act_bias"))
  if (m.opt.disableflags & (DisableBit.SPRING | DisableBit.DAMPER)) != (DisableBit.SPRING | DisableBit.DAMPER):
    callbacks.append("passive")
  for name in callbacks:
    if getattr(m.callback, name):
      raise NotImplementedError(f"adjoint.step_backward does not support the {name} callback")


@wp.kernel(enable_backward=False)
def _load_rhs(nv: int, adj_qacc: wp.array2d[float], grad_out: wp.array2d[float]):
  worldid, i = wp.tid()
  grad_out[worldid, i] = adj_qacc[worldid, i] if i < nv else 0.0


@cache_kernel
def _solve_adjoint_cholesky(nv: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(h_in: wp.array3d[float], rhs_in: wp.array2d[float], solution_out: wp.array2d[float]):
    w = wp.tid()
    h = wp.tile_load(h_in[w], shape=(nv, nv))
    wp.tile_cholesky_inplace(h, fill_mode="upper")
    rhs = wp.tile_load(rhs_in[w], shape=nv)
    solution = wp.tile_cholesky_solve(h, rhs, fill_mode="upper")
    wp.tile_store(solution_out[w], solution)

  return kernel


@event_scope
def solve_backward(m: Model, d_bwd: Data, adj_qacc: wp.array, bc: BackwardContext):
  """Solves ``H * lam = adj_qacc`` at the converged acceleration."""
  solver_ctx = bc.solver_ctx
  if d_bwd.njmax == 0 or m.nv == 0:
    if m.nv:
      smooth.solve_m(m, d_bwd, solver_ctx.search, adj_qacc)
    return solver_ctx
  solver.init_context(m, d_bwd, solver_ctx, grad=True)
  # Only the Hessian is needed; replace the initializer's gradient with the adjoint RHS.
  wp.launch(_load_rhs, dim=(d_bwd.nworld, m.nv_pad), inputs=[m.nv, adj_qacc], outputs=[solver_ctx.grad])
  if m.nv <= solver._BLOCK_CHOLESKY_DIM:
    wp.launch_tiled(
      _solve_adjoint_cholesky(m.nv),
      dim=d_bwd.nworld,
      inputs=[solver_ctx.h, solver_ctx.grad],
      outputs=[solver_ctx.search],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
  else:
    wp.launch(
      solver._padding_h,
      dim=(d_bwd.nworld, m.nv_pad - m.nv),
      inputs=[m.nv, solver_ctx.done],
      outputs=[solver_ctx.h],
    )
    rhs = solver_ctx.grad.reshape((d_bwd.nworld, m.nv_pad, 1))
    # This helper loads the entire RHS before storing its solution, so reuse the padded buffer.
    wp.launch_tiled(
      solver._cholesky_factorize_solve_blocked(TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
      dim=d_bwd.nworld,
      inputs=[solver_ctx.h, rhs],
      outputs=[solver_ctx.hfactor, rhs],
      block_dim=m.block_dim.update_gradient_cholesky_blocked,
    )
    wp.launch(_load_rhs, dim=(d_bwd.nworld, m.nv), inputs=[m.nv, solver_ctx.grad], outputs=[solver_ctx.search])
  return solver_ctx


@event_scope
def _residual_constraint_sparse(
  m: Model,
  d: Data,
  d_bwd: Data,
  ctx_Jaref: wp.array,
  lam: wp.array,
  res_qvel: wp.array,
  res_dof: wp.array,
  bc: BackwardContext,
):
  """Contracts all active non-contact rows and accumulates their residual VJP."""
  nworld, nv = d_bwd.nworld, m.nv
  njmax = d_bwd.efc.type.shape[1]
  efc_Z, efc_invw = bc.efc_Z, bc.efc_invw
  gather = constraint_adjoint._constraint_gather(m.is_sparse)
  wp.launch(
    gather,
    dim=(nworld, njmax),
    inputs=[
      nv,
      m.jnt_dofadr,
      m.dof_invweight0,
      m.eq_obj1id,
      m.eq_obj2id,
      d_bwd.nefc,
      d_bwd.efc.type,
      d_bwd.efc.id,
      d_bwd.efc.J_rownnz,
      d_bwd.efc.J_rowadr,
      d_bwd.efc.J_colind,
      d_bwd.efc.J,
      d_bwd.efc.state,
      lam,
    ],
    outputs=[efc_Z, efc_invw],
  )

  efc_phi = bc.efc_phi
  res_efc_pos, res_efc_vel = bc.adj_efc_pos, bc.adj_efc_vel
  constraint_params = model_adjoint._constraint_param_pairs(m, bc)
  pairs = [
    (m.opt.timestep, None),
    (m.opt.disableflags, None),
    *((param, residual if param.requires_grad else None) for param, residual in constraint_params),
    (d_bwd.nefc, None),
    (d_bwd.efc.type, None),
    (d_bwd.efc.id, None),
    (d_bwd.efc.pos, res_efc_pos),
    (d_bwd.efc.margin, None),
    (d_bwd.efc.D, None),
    (d_bwd.efc.vel, res_efc_vel),
    (d_bwd.efc.aref, None),
    (d_bwd.efc.force, None),
    (d_bwd.efc.state, None),
    (efc_Z, efc_Z.grad),
    (efc_invw, None),
    (ctx_Jaref, None),
  ]
  efc_phi.grad.fill_(1.0)
  adjoint_util._launch_vjp(constraint_adjoint._constraint_row_phi, (nworld, njmax), pairs, [efc_phi], [efc_phi.grad])

  scatter = constraint_adjoint._constraint_scatter(m.is_sparse)
  wp.launch(
    scatter,
    dim=(nworld, njmax),
    inputs=[
      nv,
      m.qpos0,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.eq_obj2id,
      m.eq_data,
      d_bwd.nefc,
      d.qpos,
      d.qvel,
      d_bwd.qacc,
      d_bwd.efc.type,
      d_bwd.efc.id,
      d_bwd.efc.J_rownnz,
      d_bwd.efc.J_rowadr,
      d_bwd.efc.J_colind,
      d_bwd.efc.J,
      d_bwd.efc.D,
      d_bwd.efc.state,
      res_efc_pos,
      res_efc_vel,
      lam,
      efc_Z,
      efc_Z.grad,
    ],
    outputs=[res_qvel, res_dof],
  )

  model_adjoint.constraint_params_backward(m, bc)


@event_scope
def noncontact_constraint_backward(
  m: Model,
  d: Data,
  d_bwd: Data,
  lam: wp.array,
  res_qpos: wp.array,
  res_qvel: wp.array,
  ctx_Jaref: wp.array,
  bc: BackwardContext,
):
  """Accumulates non-contact residuals in input position and velocity space."""
  _residual_constraint_sparse(m, d, d_bwd, ctx_Jaref, lam, res_qvel, bc.res_dof, bc)
  wp.launch(
    smooth_kinematics_adjoint._dof_to_qpos,
    dim=(d.nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, bc.res_dof],
    outputs=[res_qpos],
  )


def _residual_contact(
  m: Model,
  d: Data,
  d_bwd: Data,
  lam: wp.array,
  res_qvel: wp.array,
  bc: BackwardContext,
):
  """Contracts contact rows before differentiating the loop-free cone leaf."""
  nconmax = d_bwd.contact.pos.shape[0]
  contact_V, contact_A, contact_Z = bc.contact_V, bc.contact_A, bc.contact_Z
  contact_phi = bc.contact_phi
  res_cdof, res_subtree_com = bc.adj_cdof, bc.adj_subtree_com
  res_efc_pos = bc.adj_efc_pos
  res_contact_pos, res_contact_frame = bc.adj_contact_pos, bc.adj_contact_frame
  walk = [m.body_rootid, m.body_weldid, m.body_dofnum, m.body_dofadr, m.dof_parentid, m.geom_bodyid]
  state = [
    d.qvel,
    d_bwd.qacc,
    d_bwd.subtree_com,
    d_bwd.cdof,
    d_bwd.contact.pos,
    d_bwd.contact.geom,
    d_bwd.contact.efc_address,
    d_bwd.contact.worldid,
    d_bwd.nacon,
  ]
  wp.launch(
    constraint_adjoint._contact_gather,
    dim=nconmax,
    inputs=walk + state + [lam],
    outputs=[contact_V, contact_A, contact_Z],
  )

  geom_friction_grad = m.geom_friction.requires_grad
  pair_friction_grad = m.pair_friction.requires_grad
  res_contact_friction = bc.adj_contact_friction

  pairs = [
    (m.opt.timestep, None),
    (m.opt.disableflags, None),
    (m.opt.impratio_invsqrt, None),
    (m.body_invweight0, None),
    (m.geom_bodyid, None),
    (d_bwd.contact.frame, res_contact_frame),
    (d_bwd.contact.friction, res_contact_friction),
    (d_bwd.contact.solref, None),
    (d_bwd.contact.solreffriction, None),
    (d_bwd.contact.solimp, None),
    (d_bwd.contact.dim, None),
    (d_bwd.contact.geom, None),
    (d_bwd.contact.efc_address, None),
    (d_bwd.contact.worldid, None),
    (d_bwd.efc.pos, res_efc_pos),
    (d_bwd.efc.margin, None),
    (d_bwd.efc.state, None),
    (d_bwd.nacon, None),
    (contact_V, contact_V.grad),
    (contact_A, contact_A.grad),
    (contact_Z, contact_Z.grad),
  ]
  contact_phi.grad.fill_(1.0)
  kernel = constraint_adjoint._contact_phi(int(m.opt.cone))
  adjoint_util._launch_vjp(kernel, nconmax, pairs, [contact_phi], [contact_phi.grad])
  wp.launch(
    constraint_adjoint._contact_scatter(geom_friction_grad, pair_friction_grad),
    dim=nconmax,
    inputs=[
      m.ngeom,
      *walk,
      m.geom_priority,
      m.geom_friction,
      m.pair_friction,
      m.nxn_pairid,
      *state,
      lam,
      contact_V.grad,
      contact_A.grad,
      contact_Z.grad,
      res_contact_friction,
    ],
    outputs=[res_qvel, res_cdof, res_subtree_com, res_contact_pos, m.geom_friction.grad, m.pair_friction.grad],
  )


@event_scope
def contact_constraint_backward(
  m: Model,
  d: Data,
  d_bwd: Data,
  lam: wp.array,
  res_qpos: wp.array,
  res_qvel: wp.array,
  bc: BackwardContext,
):
  """Accumulates the contact residual VJP in input position and velocity space."""
  _residual_contact(m, d, d_bwd, lam, res_qvel, bc)
  collision_adjoint.contact_qpos_vjp(
    m,
    d_bwd,
    d.qpos,
    bc.adj_contact_pos,
    bc.adj_contact_frame,
    bc.adj_efc_pos,
    bc.adj_subtree_com,
    bc.adj_cdof,
    res_qpos,
    bc,
  )


@event_scope
def smooth_params_backward(m: Model, d: Data, d_bwd: Data, lam: wp.array, bc: BackwardContext):
  """Accumulates smooth Model parameter gradients."""
  if m.dof_armature.requires_grad or m.dof_damping.requires_grad:
    pairs = [
      (m.opt.disableflags, None),
      (m.dof_armature, m.dof_armature.grad if m.dof_armature.requires_grad else None),
      (m.dof_damping, m.dof_damping.grad if m.dof_damping.requires_grad else None),
      # State VJPs were already consumed; explicit sinks avoid Warp's .grad fallback.
      (d.qvel, bc.smooth_adj_qvel),
      (d_bwd.qacc, bc.smooth_adj_qacc),
      (lam, None),
      (bc.y_remap, None),
      (bc.w_dir, None),
    ]
    bc.smooth_r.grad.fill_(-1.0)
    adjoint_util._launch_vjp(
      model_adjoint._smooth_param_contraction,
      (d.nworld, m.nv),
      pairs,
      [bc.smooth_r],
      [bc.smooth_r.grad],
    )


@wp.kernel(enable_backward=False)
def _accum_input_adjoint(direct: wp.array2d[float], residual: wp.array2d[float], adj_out: wp.array2d[float]):
  w, i = wp.tid()
  adj_out[w, i] += direct[w, i] - residual[w, i]


def _write_input_adjoints(d: Data, bc: BackwardContext):
  """Combines direct state adjoints with implicit residual VJPs."""
  wp.launch(_accum_input_adjoint, dim=d.qpos.shape, inputs=[bc.adj_qpos, bc.res_qpos], outputs=[d.qpos.grad])
  wp.launch(_accum_input_adjoint, dim=d.qvel.shape, inputs=[bc.adj_qvel, bc.res_qvel], outputs=[d.qvel.grad])


@event_scope
def forward_backward_ift(m: Model, d: Data, d_bwd: Data, adj_qacc: wp.array, bc: BackwardContext):
  """Differentiates forward dynamics at the converged acceleration."""
  solver_ctx = solve_backward(m, d_bwd, adj_qacc, bc)
  lam = solver_ctx.search
  if d_bwd.njmax and m.nv:
    contact_constraint_backward(m, d, d_bwd, lam, bc.res_qpos, bc.res_qvel, bc)
    noncontact_constraint_backward(m, d, d_bwd, lam, bc.res_qpos, bc.res_qvel, solver_ctx.Jaref, bc)
  forward_adjoint.smooth_qvel_backward(m, d, d_bwd, lam, bc.adj_qvel, bc)
  forward_adjoint.smooth_qpos_backward(m, d, d_bwd, lam, bc.res_qpos, bc)
  # The position reverse also computes the RNE velocity cotangents. Consume them
  # with the residual sign before parameter VJPs reuse the workspace.
  for cotangent in (bc.smooth_adj_qvel, bc.smooth_cv_adj_qvel):
    wp.launch(adjoint_util._accum_cols, dim=(d.nworld, m.nv), inputs=[cotangent], outputs=[bc.res_qvel])
  smooth_params_backward(m, d, d_bwd, lam, bc)


_STEP_IN_FIELDS = (
  "qpos",
  "qvel",
  "act",
)
_STEP_OUT_FIELDS = (
  "qpos",
  "qvel",
  "qacc",
  "act",
  "act_dot",
)
# Supported differentiable mjSTATE_USER inputs, selected separately from state gradients.
_STEP_USER_FIELDS = (
  "ctrl",
  "qfrc_applied",
  "xfrc_applied",
)
_STEP_MODEL_FIELDS = (
  "body_mass",
  "body_inertia",
  "body_ipos",
  "body_iquat",
  "dof_armature",
  "dof_damping",
  "jnt_stiffness",
  "jnt_stiffnesspoly",
  "qpos_spring",
  "jnt_solref",
  "jnt_solimp",
  "dof_solref",
  "dof_solimp",
  "dof_frictionloss",
  "eq_solref",
  "eq_solimp",
  "geom_friction",
  "pair_friction",
)
_KINEMATICS_IN_FIELDS = ("qpos",)
_KINEMATICS_OUT_FIELDS = ("site_xpos", "xpos", "xquat")


def _data_arrays(d: Data, fields: tuple[str, ...], *, force_grad: bool = True) -> list[wp.array]:
  """Returns nonempty Data arrays that require gradients, optionally enabling them first."""
  arrays = [getattr(d, name) for name in fields if getattr(d, name).size]
  if force_grad:
    for array in arrays:
      array.requires_grad = True
  return [array for array in arrays if array.requires_grad]


def _model_arrays(m: Model) -> list[wp.array]:
  """Returns opted-in model parameters, rejecting incomplete step derivatives."""
  parameters = {
    prefix + name: value
    for prefix, obj in (("", m), ("opt.", m.opt))
    for name, value in vars(obj).items()
    if isinstance(value, wp.array) and value.size and value.requires_grad
  }
  unsupported = parameters.keys() - set(_STEP_MODEL_FIELDS)
  if unsupported:
    raise NotImplementedError("native step parameter gradients are not implemented for: " + ", ".join(sorted(unsupported)))
  return list(parameters.values())


def step_supported(m: Model, d: Data, d_out: Data | None = None) -> tuple[str, ...]:
  """Returns unsupported-step reasons using the diff API's callback signature."""
  try:
    _assert_step_supported(m)
    _assert_model_supported(m)
    if d.naconmax:
      collision_adjoint.assert_collision_supported(m)
  except NotImplementedError as error:
    return (str(error),)
  return ()


def step_backward_arrays(d: Data, d_out: Data, m: Model | None = None, *, user_fields: tuple[str, ...] = ("ctrl",)):
  """Prepares state gradients and selected user-input gradients for the legacy tape adapter.

  Explicitly enabled user-input gradients are also tracked. The diff API will own this selection.
  """
  for name in ("nworld", "naconmax", "njmax"):
    assert getattr(d, name) == getattr(d_out, name), f"step() differentiation expected matching {name}"
  requested = set(user_fields)
  unsupported = requested - set(_STEP_USER_FIELDS)
  if unsupported:
    raise NotImplementedError("native step user-input gradients are not implemented for: " + ", ".join(sorted(unsupported)))
  parameters = _model_arrays(m) if m is not None else []
  arrays = [
    *_data_arrays(d, _STEP_IN_FIELDS),
    *_data_arrays(d_out, _STEP_OUT_FIELDS),
    *_data_arrays(d_out, _KINEMATICS_OUT_FIELDS, force_grad=False),
  ]
  for name in _STEP_USER_FIELDS:
    if name in requested or getattr(d, name).requires_grad or getattr(d_out, name).requires_grad:
      arrays.extend(_data_arrays(d, (name,)))
      arrays.extend(_data_arrays(d_out, (name,)))
  arrays.extend(parameters)
  return list(dict.fromkeys(array for array in arrays if array.size))


@wp.kernel(enable_backward=False)
def _accum_copied_input(adj_in: wp.array2d[Any], adj_out: wp.array2d[Any]):
  w, i = wp.tid()
  adj_out[w, i] += adj_in[w, i]


@wp.kernel(enable_backward=False)
def _accum_kinematics_adjoints(
  # In:
  adj_qpos: wp.array2d[float],
  # Out:
  adj_qpos_out: wp.array2d[float],
  adj_site_xpos_out: wp.array2d[wp.vec3],
  adj_xpos_out: wp.array2d[wp.vec3],
  adj_xquat_out: wp.array2d[wp.quat],
):
  w, i = wp.tid()
  if adj_qpos_out and i < adj_qpos_out.shape[1]:
    adj_qpos_out[w, i] += adj_qpos[w, i]
  if adj_site_xpos_out and i < adj_site_xpos_out.shape[1]:
    adj_site_xpos_out[w, i] = wp.vec3()
  if adj_xpos_out and i < adj_xpos_out.shape[1]:
    adj_xpos_out[w, i] = wp.vec3()
  if adj_xquat_out and i < adj_xquat_out.shape[1]:
    adj_xquat_out[w, i] = wp.quat(0.0, 0.0, 0.0, 0.0)


def _write_kinematics_adjoints(d: Data, bc: BackwardContext):
  """Accumulates FK input gradients and consumes its overwritten output seeds."""
  grads = [
    d.qpos.grad,
    d.site_xpos.grad,
    d.xpos.grad,
    d.xquat.grad,
  ]
  wp.launch(
    _accum_kinematics_adjoints,
    dim=(d.nworld, max((grad.shape[1] for grad in grads if grad is not None), default=0)),
    inputs=[bc.adj_qpos],
    outputs=grads,
  )


def _clear_kinematics_adjoints(d: Data):
  """Clears kinematic output gradients after step backward."""
  grads = [getattr(d, name).grad for name in _KINEMATICS_OUT_FIELDS]
  wp.launch(
    _accum_kinematics_adjoints,
    dim=(d.nworld, max((grad.shape[1] for grad in grads if grad is not None), default=0)),
    inputs=[None],
    outputs=[None, *grads],
  )


def step_backward_scratch(m: Model, d: Data, d_out: Data, bc: BackwardContext) -> Data:
  """Prepares backward scratch from forward inputs and retained intermediates."""
  kinematics_grad = bool(_data_arrays(d_out, _KINEMATICS_OUT_FIELDS, force_grad=False))
  d_kin = bc.scratch if kinematics_grad else d_out
  # Borrow only read-only state and forward caches; all other fields remain backward scratch.
  d_bwd = dataclasses.replace(
    bc.scratch,
    nworld=d_out.nworld,
    naconmax=d_out.naconmax,
    njmax=d_out.njmax,
    ne=d_out.ne,
    nf=d_out.nf,
    nefc=d_out.nefc,
    nacon=d_out.nacon,
    qpos=d.qpos,
    qvel=d.qvel,
    ctrl=d.ctrl,
    mocap_pos=d.mocap_pos,
    mocap_quat=d.mocap_quat,
    act=d_out.act,
    act_dot=d_out.act_dot,
    qacc=d_out.qacc,
    xpos=d_kin.xpos,
    xquat=d_kin.xquat,
    xmat=d_kin.xmat,
    xipos=d_kin.xipos,
    ximat=d_kin.ximat,
    geom_xpos=d_kin.geom_xpos,
    geom_xmat=d_kin.geom_xmat,
    subtree_com=d_kin.subtree_com,
    cdof=d_kin.cdof,
    cinert=d_kin.cinert,
    crb=d_out.crb,
    M=d_out.M,
    qLD=d_out.qLD,
    qLDiagInv=d_out.qLDiagInv,
    tree_awake=d_out.tree_awake,
    cvel=d_out.cvel,
    cdof_dot=d_out.cdof_dot,
    cfrc_ext=d_out.cfrc_ext,
    actuator_length=d_out.actuator_length,
    actuator_velocity=d_out.actuator_velocity,
    actuator_force=d_out.actuator_force,
    moment_rownnz=d_out.moment_rownnz,
    moment_rowadr=d_out.moment_rowadr,
    moment_colind=d_out.moment_colind,
    actuator_moment=d_out.actuator_moment,
    qfrc_actuator=d_out.qfrc_actuator,
    qfrc_smooth=d_out.qfrc_smooth,
    dof_frictionadr=d_out.dof_frictionadr,
    dof_cdof=d_out.dof_cdof,
    cdof_dof=d_out.cdof_dof,
    contact=dataclasses.replace(
      bc.scratch.contact,
      dist=d_out.contact.dist,
      pos=d_out.contact.pos,
      frame=d_out.contact.frame,
      includemargin=d_out.contact.includemargin,
      friction=d_out.contact.friction,
      solref=d_out.contact.solref,
      solreffriction=d_out.contact.solreffriction,
      solimp=d_out.contact.solimp,
      dim=d_out.contact.dim,
      geom=d_out.contact.geom,
      efc_address=d_out.contact.efc_address,
      worldid=d_out.contact.worldid,
      geomcollisionid=d_out.contact.geomcollisionid,
    ),
    efc=dataclasses.replace(
      bc.scratch.efc,
      type=d_out.efc.type,
      id=d_out.efc.id,
      J_rownnz=d_out.efc.J_rownnz,
      J_rowadr=d_out.efc.J_rowadr,
      J_colind=d_out.efc.J_colind,
      J=d_out.efc.J,
      pos=d_out.efc.pos,
      margin=d_out.efc.margin,
      D=d_out.efc.D,
      vel=d_out.efc.vel,
      aref=d_out.efc.aref,
      frictionloss=d_out.efc.frictionloss,
      jtdaj_adr=d_out.efc.jtdaj_adr,
      jtdaj_nrow=d_out.efc.jtdaj_nrow,
      jtdaj_nblock=d_out.efc.jtdaj_nblock,
    ),
  )
  if kinematics_grad:
    # A later fwd_kinematics call may recompute d_out using d_out.qpos.
    # Step backward needs kinematics and COM values computed from d.qpos.
    smooth.kinematics(m, d_bwd)
    smooth.com_pos(m, d_bwd)
  return d_bwd


@event_scope
def step_backward(m: Model, d: Data, d_out: Data, bc: BackwardContext | None = None):
  """Maps ``d_out`` state gradients to ``d`` with an analytic backward."""
  reasons = step_supported(m, d, d_out)
  if reasons:
    raise NotImplementedError(", ".join(reasons))
  # TODO: Remove this compatibility preparation when the diff API registers the rule (#1727).
  step_backward_arrays(d, d_out, m, user_fields=())
  bc = bc or _ACTIVE_BACKWARD_CONTEXT.get() or create_backward_context(m, d)
  _reset_backward_context(bc)
  d_bwd = step_backward_scratch(m, d, d_out, bc)
  kinematics_grad = bool(_data_arrays(d_out, _KINEMATICS_OUT_FIELDS, force_grad=False))
  if kinematics_grad:
    forward_adjoint.kinematics_backward(m, d_out, d_bwd, bc)
  _, _, adj_qacc = forward_adjoint.advance_backward(m, d_out, d_bwd, bc)
  forward_backward_ift(m, d, d_bwd, adj_qacc, bc)
  _write_input_adjoints(d, bc)
  for name in _STEP_USER_FIELDS:
    src, dst = getattr(d, name), getattr(d_out, name)
    # Shared gradient storage already contains downstream uses of the held input.
    if src.grad is not None and dst.grad is not None and src.grad.ptr != dst.grad.ptr:
      wp.launch(_accum_copied_input, dim=src.shape, inputs=[dst.grad], outputs=[src.grad])
  if kinematics_grad:
    _clear_kinematics_adjoints(d_out)


def fwd_kinematics_arrays(d: Data):
  """Returns differentiated fwd_kinematics inputs and outputs."""
  return _data_arrays(d, _KINEMATICS_IN_FIELDS + _KINEMATICS_OUT_FIELDS)


def fwd_kinematics_backward_scratch(d: Data, bc: BackwardContext) -> Data:
  """Borrows FK inputs while retaining private storage for recomputed poses."""
  return dataclasses.replace(bc.scratch, nworld=d.nworld, qpos=d.qpos, mocap_pos=d.mocap_pos, mocap_quat=d.mocap_quat)


@event_scope
def fwd_kinematics_backward(m: Model, d: Data, bc: BackwardContext | None = None):
  """Maps Cartesian body/site cotangents to generalized position coordinates."""
  kinematics_grad = bool(_data_arrays(d, _KINEMATICS_OUT_FIELDS, force_grad=False))
  if d.qpos.grad is None or not kinematics_grad:
    return
  bc = bc or _ACTIVE_BACKWARD_CONTEXT.get() or create_backward_context(m, d)
  d_bwd = fwd_kinematics_backward_scratch(d, bc)
  smooth.kinematics1(m, d_bwd)
  forward_adjoint.kinematics_backward(m, d, d_bwd, bc)
  _write_kinematics_adjoints(d, bc)


def enable_grad() -> None:
  """Installs numerical derivative rules and analytic backward hooks.

  Replaces ``wp.sqrt`` with ``adjoint_util.safe_sqrt``, whose float32 and float64 adjoints
  contribute zero at zero, and registers custom quaternion gradients. Also enables forward
  recording and the step and forward-kinematics backward hooks.

  Call before building kernels that need these rules. The numerical rules apply process-wide,
  independently of tapes and backward-hook registration.

  Kinematic output gradients support site_xpos, xpos, and xquat. Step computes these
  before integration; call fwd_kinematics on the output to refresh them using the updated qpos.

  Backward reductions are nondeterministic, regardless of forward determinism options.
  """
  forward.ENABLE_GRAD = True
  wp.sqrt = adjoint_util.safe_sqrt
  wp.func_grad(math.quat_integrate)(adjoint_util._adj_quat_integrate)
  wp.func_grad(math.quat_to_vel)(adjoint_util._adj_quat_to_vel)
  forward.register_step_backward(step_backward, step_backward_arrays, step_supported)
  forward.register_fwd_kinematics_backward(fwd_kinematics_backward, fwd_kinematics_arrays)
