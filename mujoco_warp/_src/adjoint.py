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
from mujoco_warp._src import collision_convex_adjoint
from mujoco_warp._src import constraint_adjoint
from mujoco_warp._src import forward
from mujoco_warp._src import forward_adjoint
from mujoco_warp._src import io
from mujoco_warp._src import math
from mujoco_warp._src import model_adjoint
from mujoco_warp._src import smooth
from mujoco_warp._src import smooth_adjoint
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import solver
from mujoco_warp._src.types import TILE_SIZE_JTDAJ_DENSE
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import SolverType
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
  if any(
    m.geom_pair_type_count[math.upper_trid_index(len(GeomType), GeomType.HFIELD.value, geom_type.value)]
    for geom_type in collision_convex_adjoint._CONVEX_TYPES
  ):
    bc.collision_convex_ctx = collision_convex_adjoint.create_collision_convex_backward_context(m, d)
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
  if m.opt.enableflags & EnableBit.SLEEP and not (m.opt.disableflags & DisableBit.ISLAND):
    raise NotImplementedError("adjoint.step_backward does not support sleep")
  if m.eq_connect_adr.size or m.eq_wld_adr.size or m.ntendon or m.nflex:
    raise NotImplementedError("adjoint.step_backward does not support connect/weld equality, tendon, or flex constraint rows")
  smooth_adjoint.assert_smooth_supported(m)


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
def solve_backward(m: Model, d_out: Data, adj_qacc: wp.array, bc: BackwardContext):
  """Solves ``H * lam = adj_qacc`` at the converged acceleration."""
  solver_ctx = bc.solver_ctx
  if d_out.njmax == 0 or m.nv == 0:
    if m.nv:
      smooth.solve_m(m, d_out, solver_ctx.search, adj_qacc)
    return solver_ctx
  solver.init_context(m, d_out, solver_ctx, grad=True)
  wp.launch(_load_rhs, dim=(d_out.nworld, m.nv_pad), inputs=[m.nv, adj_qacc], outputs=[solver_ctx.grad])
  if m.nv <= solver._BLOCK_CHOLESKY_DIM:
    wp.launch_tiled(
      _solve_adjoint_cholesky(m.nv),
      dim=d_out.nworld,
      inputs=[solver_ctx.h, solver_ctx.grad],
      outputs=[solver_ctx.search],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
  else:
    wp.launch(
      solver._padding_h,
      dim=(d_out.nworld, m.nv_pad - m.nv),
      inputs=[m.nv, solver_ctx.done],
      outputs=[solver_ctx.h],
    )
    rhs = solver_ctx.grad.reshape((d_out.nworld, m.nv_pad, 1))
    # This helper loads the entire RHS before storing its solution, so reuse the padded buffer.
    wp.launch_tiled(
      solver._cholesky_factorize_solve_blocked(TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
      dim=d_out.nworld,
      inputs=[solver_ctx.h, rhs],
      outputs=[solver_ctx.hfactor, rhs],
      block_dim=m.block_dim.update_gradient_cholesky_blocked,
    )
    wp.launch(_load_rhs, dim=(d_out.nworld, m.nv), inputs=[m.nv, solver_ctx.grad], outputs=[solver_ctx.search])
  return solver_ctx


@event_scope
def _residual_constraint_sparse(
  m: Model,
  d: Data,
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
      m.eq_obj1id,
      m.eq_obj2id,
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
      m.qpos0,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.eq_obj2id,
      m.eq_data,
      d_out.nefc,
      d.qpos,
      d.qvel,
      d_out.qacc,
      d_out.efc.type,
      d_out.efc.id,
      d_out.efc.J_rownnz,
      d_out.efc.J_rowadr,
      d_out.efc.J_colind,
      d_out.efc.J,
      d_out.efc.D,
      d_out.efc.state,
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
  d_out: Data,
  lam: wp.array,
  res_qpos: wp.array,
  res_qvel: wp.array,
  ctx_Jaref: wp.array,
  bc: BackwardContext,
):
  """Accumulates non-contact residuals in input position and velocity space."""
  _residual_constraint_sparse(m, d, d_out, ctx_Jaref, lam, res_qvel, bc.res_dof, bc)
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
    d_out.nacon,
  ]
  wp.launch(
    constraint_adjoint._contact_gather,
    dim=nconmax,
    inputs=walk + state + [lam],
    outputs=[contact_V, contact_A, contact_Z],
  )

  geom_friction_grad = m.geom_friction.requires_grad
  pair_friction_grad = m.pair_friction.requires_grad
  res_contact_friction = d_out.contact.friction.grad if geom_friction_grad or pair_friction_grad else None

  pairs = [
    (m.opt.timestep, None),
    (m.opt.disableflags, None),
    (m.opt.impratio_invsqrt, None),
    (m.body_invweight0, None),
    (m.geom_bodyid, None),
    (d_out.contact.frame, res_contact_frame),
    (d_out.contact.friction, res_contact_friction),
    (d_out.contact.solref, None),
    (d_out.contact.solreffriction, None),
    (d_out.contact.solimp, None),
    (d_out.contact.dim, None),
    (d_out.contact.geom, None),
    (d_out.contact.efc_address, None),
    (d_out.contact.worldid, None),
    (d_out.efc.pos, res_efc_pos),
    (d_out.efc.margin, None),
    (d_out.efc.state, None),
    (d_out.nacon, None),
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
      (m.opt.disableflags, None),
      (m.dof_armature, m.dof_armature.grad if m.dof_armature.requires_grad else None),
      (m.dof_damping, m.dof_damping.grad if m.dof_damping.requires_grad else None),
      # State VJPs were already consumed; explicit sinks avoid Warp's .grad fallback.
      (d.qvel, bc.smooth_adj_qvel),
      (d_out.qacc, bc.smooth_adj_qacc),
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
def forward_backward_ift(m: Model, d: Data, d_out: Data, adj_qacc: wp.array, bc: BackwardContext):
  """Differentiates forward dynamics at the converged acceleration."""
  solver_ctx = solve_backward(m, d_out, adj_qacc, bc)
  lam = solver_ctx.search
  if d_out.njmax and m.nv:
    contact_constraint_backward(m, d, d_out, lam, bc.res_qpos, bc.res_qvel, bc)
    noncontact_constraint_backward(m, d, d_out, lam, bc.res_qpos, bc.res_qvel, solver_ctx.Jaref, bc)
  forward_adjoint.smooth_qvel_backward(m, d, d_out, lam, bc.adj_qvel, bc)
  forward_adjoint.smooth_qpos_backward(m, d, d_out, lam, bc.res_qpos, bc)
  # The position reverse also computes the RNE velocity cotangents. Consume them
  # with the residual sign before parameter VJPs reuse the workspace.
  for cotangent in (bc.smooth_adj_qvel, bc.smooth_cv_adj_qvel):
    wp.launch(adjoint_util._accum_cols, dim=(d.nworld, m.nv), inputs=[cotangent], outputs=[bc.res_qvel])
  smooth_params_backward(m, d, d_out, lam, bc)
  _write_input_adjoints(d, bc)


def step_backward_arrays(d: Data, d_out: Data):
  """Allocates and returns arrays used by the analytic step backward."""
  inputs = [d.qpos, d.qvel, d.ctrl]
  for array in (d.qfrc_applied, d.xfrc_applied):
    if array.requires_grad:
      inputs.append(array)
  for array in inputs:
    array.requires_grad = True
  return [array for array in (*inputs, *_prepare_data(d_out)) if array.size]


@event_scope
def step_backward(m: Model, d: Data, d_out: Data, bc: BackwardContext | None = None):
  """Maps ``d_out`` state gradients to ``d`` with an analytic backward."""
  _assert_step_supported(m)
  step_backward_arrays(d, d_out)
  bc = bc or _ACTIVE_BACKWARD_CONTEXT.get() or create_backward_context(m, d)
  _reset_backward_context(bc)
  _, _, adj_qacc = forward_adjoint.advance_backward(m, d, d_out, bc)
  forward_backward_ift(m, d, d_out, adj_qacc, bc)


def fwd_kinematics_arrays(d: Data):
  """Returns differentiated fwd_kinematics inputs and outputs."""
  return [
    array for array in (d.qpos, d.site_xpos, d.xpos, d.xquat) if array is not None and array.size and array.grad is not None
  ]


@event_scope
def fwd_kinematics_backward(m: Model, d: Data, bc: BackwardContext | None = None):
  """Maps site and body-pose cotangents to generalized position coordinates."""
  if d.qpos.grad is None or not any(
    array is not None and array.size and array.grad is not None for array in (d.site_xpos, d.xpos, d.xquat)
  ):
    return
  bc = bc or _ACTIVE_BACKWARD_CONTEXT.get() or create_backward_context(m, d)
  forward_adjoint.fwd_kinematics_backward(m, d, bc)


def enable_grad() -> None:
  """Installs numerical derivative rules and analytic backward hooks.

  Replaces ``wp.sqrt`` with ``adjoint_util.safe_sqrt``, whose float32 and float64 adjoints
  contribute zero at zero, and registers custom quaternion gradients. Also enables forward
  recording and the step and forward-kinematics backward hooks.

  Call before building kernels that need these rules. The numerical rules apply process-wide,
  independently of tapes and backward-hook registration.

  Backward reductions are nondeterministic, regardless of forward determinism options.
  """
  forward.ENABLE_GRAD = True
  wp.sqrt = adjoint_util.safe_sqrt
  wp.func_grad(math.quat_integrate)(adjoint_util._adj_quat_integrate)
  wp.func_grad(math.quat_to_vel)(adjoint_util._adj_quat_to_vel)
  forward.register_step_backward(step_backward, step_backward_arrays)
  forward.register_fwd_kinematics_backward(fwd_kinematics_backward, fwd_kinematics_arrays)
