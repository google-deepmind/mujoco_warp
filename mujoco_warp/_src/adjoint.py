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

import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import collision_adjoint
from mujoco_warp._src import constraint_adjoint
from mujoco_warp._src import forward
from mujoco_warp._src import forward_adjoint
from mujoco_warp._src import io
from mujoco_warp._src import math
from mujoco_warp._src import model_adjoint
from mujoco_warp._src import smooth_adjoint
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import solver
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import SolverType
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


def _prepare_data(d: Data):
  """Allocates gradients for Data fields consumed by a backward context."""
  arrays = [
    d.qpos,
    d.qvel,
    d.qacc,
    d.efc.pos,
    d.efc.vel,
    d.contact.pos,
    d.contact.frame,
    d.contact.friction,
  ]
  arrays = [array for array in arrays if array is not None]
  for array in arrays:
    array.requires_grad = True
  return [*arrays, *smooth_adjoint.prepare_backward(d)]


def _reset_backward_context(bc: BackwardContext):
  """Zeros a backward context and its array gradients."""
  for field in dataclasses.fields(bc):
    value = getattr(bc, field.name)
    if isinstance(value, wp.array):
      value.zero_()
      if value.grad is not None:
        value.grad.zero_()


def create_backward_context(m: Model, d: Data) -> BackwardContext:
  """Preallocates the reusable backward workspace for ``(m, d)``."""
  smooth_adjoint.assert_smooth_supported(m)
  _prepare_data(d)
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
  bc = BackwardContext(solver_ctx=solver._create_solver_context(m, d), scratch=adjoint_util._clone_nograd(d), **arrays)
  for name in ("contact_V", "contact_A", "contact_Z", "contact_phi", "efc_Z", "efc_phi", "smooth_r"):
    getattr(bc, name).requires_grad = True
  smooth_adjoint.prepare_backward(bc.scratch)
  return bc


def _assert_step_supported(m: Model):
  """Rejects configurations without a complete analytic backward."""
  if m.opt.solver != SolverType.NEWTON:
    raise NotImplementedError("adjoint.step_backward supports only the Newton solver")
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
  if m.opt.enableflags & EnableBit.SLEEP and not m.opt.disableflags & DisableBit.ISLAND:
    raise NotImplementedError("adjoint.step_backward does not support sleep")
  if m.eq_connect_adr.size or m.eq_wld_adr.size or m.ntendon or m.nflex:
    raise NotImplementedError("adjoint.step_backward does not support connect/weld equality, tendon, or flex constraint rows")


@wp.kernel(enable_backward=False)
def _load_rhs(nv: int, adj_qacc: wp.array2d[float], grad_out: wp.array2d[float]):
  worldid, i = wp.tid()
  grad_out[worldid, i] = -adj_qacc[worldid, i] if i < nv else 0.0


@event_scope
def solve_backward(m: Model, d_out: Data, adj_qacc: wp.array, bc: BackwardContext):
  """Solves ``H * lam = adj_qacc`` at the converged acceleration."""
  solver_ctx = bc.solver_ctx
  solver.init_context(m, d_out, solver_ctx, grad=True)
  wp.launch(_load_rhs, dim=(d_out.nworld, m.nv_pad), inputs=[m.nv, adj_qacc], outputs=[solver_ctx.grad])
  solver._cholesky_factorize_solve(m, d_out, solver_ctx)
  return solver_ctx


@event_scope
def _residual_constraint_sparse(
  m: Model,
  d_out: Data,
  ctx_Jaref: wp.array,
  lam: wp.array,
  res_qvel: wp.array,
  res_dof: wp.array,
  bc: BackwardContext,
):
  """Contracts all active non-contact rows and accumulates their residual VJP."""
  nworld, nv = d_out.nworld, m.nv
  njmax = d_out.efc.type.shape[1]
  efc_Z, efc_invw = bc.efc_Z, bc.efc_invw
  gather = constraint_adjoint._constraint_gather(m.is_sparse)
  wp.launch(
    gather,
    dim=(nworld, njmax),
    inputs=[
      nv,
      m.jnt_dofadr,
      m.dof_invweight0,
      d_out.nefc,
      d_out.efc.type,
      d_out.efc.id,
      d_out.efc.J_rownnz,
      d_out.efc.J_rowadr,
      d_out.efc.J_colind,
      d_out.efc.J,
      d_out.efc.state,
      lam,
    ],
    outputs=[efc_Z, efc_invw],
  )

  efc_phi = bc.efc_phi
  res_efc_pos, res_efc_vel = d_out.efc.pos.grad, d_out.efc.vel.grad
  constraint_params = model_adjoint._constraint_param_pairs(m, bc)
  pairs = [
    (m.opt.timestep, None),
    (m.opt.disableflags, None),
    *((param, residual if param.requires_grad else None) for param, residual in constraint_params),
    (d_out.nefc, None),
    (d_out.efc.type, None),
    (d_out.efc.id, None),
    (d_out.efc.pos, res_efc_pos),
    (d_out.efc.margin, None),
    (d_out.efc.D, None),
    (d_out.efc.vel, res_efc_vel),
    (d_out.efc.aref, None),
    (d_out.efc.force, None),
    (d_out.efc.state, None),
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
      d_out.nefc,
      d_out.efc.type,
      d_out.efc.J_rownnz,
      d_out.efc.J_rowadr,
      d_out.efc.J_colind,
      d_out.efc.J,
      d_out.efc.state,
      res_efc_pos,
      res_efc_vel,
    ],
    outputs=[res_qvel, res_dof],
  )

  model_adjoint.constraint_params_backward(m, bc)


@event_scope
def noncontact_constraint_backward(
  m: Model,
  d: Data,
  d_out: Data,
  lam: wp.array,
  res_qpos: wp.array,
  res_qvel: wp.array,
  ctx_Jaref: wp.array,
  bc: BackwardContext,
):
  """Accumulates non-contact residuals in input position and velocity space."""
  _residual_constraint_sparse(m, d_out, ctx_Jaref, lam, res_qvel, bc.res_dof, bc)
  wp.launch(
    smooth_kinematics_adjoint._dof_to_qpos,
    dim=(d.nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, bc.res_dof],
    outputs=[res_qpos],
  )


def _residual_contact(
  m: Model,
  d: Data,
  d_out: Data,
  lam: wp.array,
  res_qvel: wp.array,
  bc: BackwardContext,
):
  """Contracts contact rows before differentiating the loop-free cone leaf."""
  nconmax = d_out.contact.pos.shape[0]
  contact_V, contact_A, contact_Z = bc.contact_V, bc.contact_A, bc.contact_Z
  contact_phi = bc.contact_phi
  res_cdof, res_subtree_com = d_out.cdof.grad, d_out.subtree_com.grad
  res_efc_pos = d_out.efc.pos.grad
  res_contact_pos, res_contact_frame = d_out.contact.pos.grad, d_out.contact.frame.grad
  efc_pos_ref = bc.efc_pos_ref
  wp.copy(efc_pos_ref, d_out.efc.pos)
  efc_pos_ref.requires_grad = False

  walk = [m.body_rootid, m.body_weldid, m.body_dofnum, m.body_dofadr, m.dof_parentid, m.geom_bodyid]
  state = [
    d.qvel,
    d_out.qacc,
    d_out.subtree_com,
    d_out.cdof,
    d_out.contact.pos,
    d_out.contact.geom,
    d_out.contact.efc_address,
    d_out.contact.worldid,
    d_out.efc.state,
    d_out.nacon,
  ]
  wp.launch(
    constraint_adjoint._contact_gather,
    dim=nconmax,
    inputs=walk + state + [lam],
    outputs=[contact_V, contact_A, contact_Z],
  )

  geom_friction_grad = m.geom_friction.requires_grad
  res_contact_friction = d_out.contact.friction.grad if geom_friction_grad else None

  pairs = [
    (m.opt.timestep, None),
    (m.opt.disableflags, None),
    (m.opt.impratio_invsqrt, None),
    (d_out.contact.frame, res_contact_frame),
    (d_out.contact.friction, res_contact_friction),
    (d_out.contact.solref, None),
    (d_out.contact.solreffriction, None),
    (d_out.contact.solimp, None),
    (d_out.contact.dim, None),
    (d_out.contact.efc_address, None),
    (d_out.contact.worldid, None),
    (d_out.efc.pos, res_efc_pos),
    (d_out.efc.margin, None),
    (d_out.efc.D, None),
    (d_out.efc.state, None),
    (d_out.nacon, None),
    (contact_V, contact_V.grad),
    (contact_A, contact_A.grad),
    (contact_Z, contact_Z.grad),
    (efc_pos_ref, None),
  ]
  contact_phi.grad.fill_(1.0)
  kernel = constraint_adjoint._contact_phi(int(m.opt.cone))
  adjoint_util._launch_vjp(kernel, nconmax, pairs, [contact_phi], [contact_phi.grad])
  wp.launch(
    constraint_adjoint._contact_scatter(geom_friction_grad),
    dim=nconmax,
    inputs=[
      *walk,
      m.geom_priority,
      m.geom_friction,
      *state,
      lam,
      contact_V.grad,
      contact_A.grad,
      contact_Z.grad,
      res_contact_friction,
    ],
    outputs=[res_qvel, res_cdof, res_subtree_com, res_contact_pos, m.geom_friction.grad],
  )


@event_scope
def contact_constraint_backward(
  m: Model,
  d: Data,
  d_out: Data,
  lam: wp.array,
  res_qpos: wp.array,
  res_qvel: wp.array,
  bc: BackwardContext,
):
  """Accumulates the contact residual VJP in input position and velocity space."""
  _residual_contact(m, d, d_out, lam, res_qvel, bc)
  collision_adjoint.contact_qpos_vjp(
    m,
    d_out,
    d.qpos,
    d_out.contact.pos.grad,
    d_out.contact.frame.grad,
    d_out.efc.pos.grad,
    d_out.subtree_com.grad,
    d_out.cdof.grad,
    res_qpos,
    bc,
  )


@event_scope
def smooth_params_backward(m: Model, d: Data, d_out: Data, lam: wp.array, bc: BackwardContext):
  """Accumulates smooth Model parameter gradients."""
  if m.dof_armature.requires_grad or m.dof_damping.requires_grad:
    pairs = [
      (m.dof_armature, m.dof_armature.grad if m.dof_armature.requires_grad else None),
      (m.dof_damping, m.dof_damping.grad if m.dof_damping.requires_grad else None),
      (d.qvel, None),
      (d_out.qacc, None),
    ]
    wp.launch(adjoint_util._neg_cols, dim=(d.nworld, m.nv), inputs=[lam], outputs=[bc.smooth_r.grad])
    adjoint_util._launch_vjp(
      model_adjoint._residual_smooth_local,
      (d.nworld, m.nv),
      pairs,
      [bc.smooth_r],
      [bc.smooth_r.grad],
    )

  if any(param.requires_grad for param in (m.body_mass, m.body_inertia, m.body_ipos, m.body_iquat)):
    model_adjoint.inertia_param_vjp(m, smooth_adjoint.linearize(m, d, d_out, bc), lam, bc=bc)


def _write_input_adjoints(d: Data, bc: BackwardContext):
  """Combines direct state adjoints with implicit residual VJPs."""
  wp.launch(adjoint_util._sub_cols, dim=d.qpos.shape, inputs=[bc.adj_qpos, bc.res_qpos], outputs=[d.qpos.grad])
  wp.launch(adjoint_util._sub_cols, dim=d.qvel.shape, inputs=[bc.adj_qvel, bc.res_qvel], outputs=[d.qvel.grad])


@event_scope
def forward_backward_ift(m: Model, d: Data, d_out: Data, adj_qacc: wp.array, bc: BackwardContext):
  """Differentiates forward dynamics at the converged acceleration."""
  solver_ctx = solve_backward(m, d_out, adj_qacc, bc)
  lam = solver_ctx.search
  contact_constraint_backward(m, d, d_out, lam, bc.res_qpos, bc.res_qvel, bc)
  noncontact_constraint_backward(m, d, d_out, lam, bc.res_qpos, bc.res_qvel, solver_ctx.Jaref, bc)
  smooth_qvel_backward(m, d, d_out, lam, bc.adj_qvel, bc)
  smooth_qpos_backward(m, d, d_out, lam, bc.res_qpos, bc)
  smooth_params_backward(m, d, d_out, lam, bc)
  _write_input_adjoints(d, bc)


def step_backward_arrays(d: Data, d_out: Data):
  """Allocates and returns arrays used by the analytic step backward."""
  inputs = [d.qpos, d.qvel, d.ctrl]
  if d.xfrc_applied.requires_grad:
    inputs.append(d.xfrc_applied)
  for array in inputs:
    array.requires_grad = True
  return [*inputs, *_prepare_data(d_out)]


@event_scope
def step_backward(m: Model, d: Data, d_out: Data, bc: BackwardContext | None = None):
  """Maps ``d_out`` state gradients to ``d`` with an analytic backward."""
  _assert_step_supported(m)
  step_backward_arrays(d, d_out)
  bc = bc or _ACTIVE_BACKWARD_CONTEXT.get() or create_backward_context(m, d)
  _reset_backward_context(bc)
  _, _, adj_qacc = advance_backward(m, d, d_out, bc)
  forward_backward_ift(m, d, d_out, adj_qacc, bc)


def fwd_kinematics_arrays(d: Data):
  """Returns differentiated fwd_kinematics inputs and outputs."""
  return [array for array in (d.qpos, d.site_xpos, d.xpos, d.xquat) if array is not None and array.grad is not None]


@event_scope
def fwd_kinematics_backward(m: Model, d: Data, bc: BackwardContext | None = None):
  """Maps site-position cotangents to generalized position coordinates."""
  if not m.nsite or d.qpos.grad is None or d.site_xpos.grad is None:
    return
  bc = bc or _ACTIVE_BACKWARD_CONTEXT.get() or create_backward_context(m, d)
  forward_adjoint.fwd_kinematics_backward(m, d, bc)


def enable_grad() -> None:
  """Enables the analytic backward hooks for step and forward kinematics."""
  forward.ENABLE_GRAD = True
  wp.sqrt = adjoint_util.safe_sqrt
  wp.func_grad(math.quat_integrate)(adjoint_util._adj_quat_integrate)
  wp.func_grad(math.quat_to_vel)(adjoint_util._adj_quat_to_vel)
  forward.register_step_backward(step_backward, step_backward_arrays)
  forward.register_fwd_kinematics_backward(fwd_kinematics_backward, fwd_kinematics_arrays)


# Compatibility exports.
_create_backward_context = create_backward_context
advance_backward = forward_adjoint.advance_backward
smooth_qvel_backward = forward_adjoint.smooth_qvel_backward
smooth_qpos_backward = forward_adjoint.smooth_qpos_backward

# Legacy test constants; the production contact path is topology-based for every ``nv``.
_MAX_NV = 16
_FORCE_SPARSE_CONTACT = True
