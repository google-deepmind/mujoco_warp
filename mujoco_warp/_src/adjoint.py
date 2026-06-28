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
"""Analytic (IFT) backward for ``mujoco_warp.step`` -- see ../../MJPLAN.md.

adjoint.py is Warp-native (no torch). It registers an analytic step backward with forward.py
(``register_backward_hook``); when ``step(m, d, d_out)`` runs under a ``wp.Tape``, that backward
is injected as one ``tape.record_func`` (d_out.grad -> d.grad), so ``tape.backward(loss)`` flows
gradients to ``d.{qpos,qvel,ctrl}``. Distinct in/out buffers per step let it chain BPTT.

THE CRUX (Implicit Function Theorem). The solver returns ``qacc`` as the root of the stationarity
residual ``r(qacc, theta) = M*qacc - qfrc_smooth - Jᵀ*efc_force(J*qacc - aref) = 0`` (theta = the
step inputs), NOT a closed form. So we differentiate the *equation*, not the iteration: with
``H = dr/dqacc = M + Jᵀ G_s J``, reverse mode is one linear solve ``H λ = adj_qacc`` then
``adj_theta = -(dr/dtheta)ᵀ λ``. We never backprop through the solver's Newton loop. ``H`` comes from
the forward SOLVER's own ``_update_gradient`` (JTDAJ + JTCJ) via ``init_context(grad=True)`` on a
``SolverContext``, run at the converged ``qacc`` so the active set ``efc.state`` matches by
construction (NOT ``inverse.py``/``InverseContext``, which has no H buffers -- see MJPLAN §5.8).

``step_backward`` composes three pieces, all Warp (graph-capture-ready, no nested tape / no host):

  1. integrator adjoint -- launch the *adjoint* of a backward-enabled ``_advance_state`` kernel
     (``wp.launch(..., adjoint=True)``). Maps adj(qpos',qvel') -> adj_qacc + integrator-direct
     adj(qpos,qvel). Warp source-to-source generates the adjoint, including ``math.quat_integrate``.

  2. IFT -- ``init_context(grad=True)`` on ``d_out`` assembles + factors ``H`` at the converged
     ``qacc``, then ``_cholesky_factorize_solve`` solves ``H λ = adj_qacc`` reusing the factor.

  3. residual-VJP -- propagate ``λ`` by reverse-mode AD of the contact residual
     ``r(qpos,qvel) = -Jᵀ·f(J·qacc - aref)``, launched ``wp.launch(_residual_contact, adjoint=True)``
     (no nested tape; same call style as piece 1) seeded with ``adj_r = λ``, giving
     ``adj_theta = -(dr/dtheta)ᵀ λ`` for ALL contact terms in ONE pass: ``∂J/∂qpos`` (the moment arm),
     the ``aref`` penetration/dissipation, the ``∂D``, and the elliptic cone curvature ``∂f/∂Jaref``.
     Replaces the per-curvature hand scatter (the retired ``_gs_cone`` is archived in MJPLAN §5.9).

SPATIAL FORMULATION. Everything is spatial algebra in the body's motion basis ``cdof`` (the same
``(angular, linear)`` axes the forward builds, ``smooth.py``/``support._compute_jacp``). A contact
row's spatial Jacobian is its line-of-action ``W`` (``wp.spatial_vector``): a translational row is a
unit force ``d`` at the contact point -> ``W = (rvec x d, d)`` (moment, force); a rotational row
(torsion/rolling) is a pure couple about ``d`` -> ``W = (d, 0)``. With the body spatial velocity /
accel ``V = Σ_i qvel_i cdof_i`` / ``A = Σ_i qacc_i cdof_i``, the row's ``J·qvel = wp.dot(V, W)`` and
``J·qacc = wp.dot(A, W)``; the residual is the net wrench ``Wtot = Σ_rows force·W`` projected back,
``r_i = -wp.dot(cdof_i, Wtot)`` (= -Jᵀf; cf. ``support._apply_ft``). Reading ``cdof`` keeps the
rotational rows EXACT for any body orientation (a spinning free body), unlike a world-axis
approximation. The moment arm ``rvec = contact_pos(qpos) - com`` tracks qpos via the sphere-vs-flat
midpoint narrowphase derivative ``∂cpos/∂qpos = I - 1/2 n nᵀ``; ``cdof`` is frozen, so the only
omitted contact term is ``∂cdof/∂qpos`` (the orientation derivative).

WARP DYNAMIC-LOOP GOTCHA (why the per-contact work is in @wp.func's). Warp does NOT replay dynamic
loops in the backward pass, so an intermediate accumulated in a dynamic loop directly in a @wp.kernel
is STALE (~0) when the adjoint runs -- invisible when it feeds a LINEAR op, catastrophic when it feeds
a NONLINEAR one (``T = sqrt(Σ ...)`` accumulated in a kernel loop made the backward recompute T≈0, and
the elliptic cone's ``1/T`` adjoint blew the gradient up ~1e18). Migrating the loop body into a
@wp.func with a STATIC (unrolled) bound fixes it. So the dof reduction (``_spatial_vel``) and the
per-contact rows (``_{elliptic,pyramidal}_wrench``) live in @wp.func's with static loop bounds; the
kernel only loops over contacts (summing wrenches -- a plain ``+=``, which IS correct in a dynamic
loop) and projects ``Wtot`` through cdof to ``r``.

Scope/status (one free-joint body, dofs 0-5; sphere-vs-flat contacts -- plane / box-face):
  * cones: elliptic AND pyramidal (``wp.static`` selects the cone-specific wrench @wp.func).
  * condim: 1, 3, 4, 6 (the valid MuJoCo set). dimid<3 = translational rows (normal + 2 tangents),
    dimid>=3 = rotational rows (3 = torsion about the normal, 4/5 = rolling about the tangents).
  * FD-verified vs ``mjd_transitionFD`` / direct FD on box/sphere-on-plane and bounce steps
    (QUADRATIC and CONE), including a spinning + sliding body for the torsion/rolling rows.
TODO for generality: build ``J`` for general articulations (any joint, multiple moving bodies, the
per-dof J_geom1 - J_geom0 difference), the ``∂cdof/∂qpos`` orientation term, the smooth qpos/qvel and
ctrl paths, and non-contact constraints (equality/limits); §5.6/§5.9.
"""

import warp as wp

from mujoco_warp._src import constraint as _constraint
from mujoco_warp._src import forward as _forward
from mujoco_warp._src import math as _math
from mujoco_warp._src import solver as _solver
from mujoco_warp._src import types as _types
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import cache_kernel

# NOTE: adjoint.py kernels are differentiable -- do NOT set enable_backward=False here, so Warp
# codegens the adjoint of _advance_state / _residual_contact.
#
# DEDUP & Warp adjoint scope (verified 2026-06-27). The residual reuses the forward's OWN physics
# as shared @wp.func's -- ``constraint._contact_kbimp`` (k, b, impedance from solref/solimp) and
# ``solver._eval_elliptic_cone`` (the elliptic cone middle-zone force) -- so the residual is a single
# source of truth with the forward (no drift if those formulas change). ``constraint.py`` / ``solver.py``
# set ``wp.set_module_options({"enable_backward": False})``, but that disables backward ONLY for *their
# kernels*: Warp generates a @wp.func's adjoint from the CALLING kernel's module (this one, which is
# backward-enabled), so differentiating the residual THROUGH those funcs is correct. Verified: the
# bounce gate gradient is unchanged to ~1e-6 and the forward FD is bit-identical (Warp inlines @wp.func,
# so factoring these out of _efc_row / _eval_constraint did not perturb the forward at all).

_FREE = int(_types.JointType.FREE.value)
_BALL = int(_types.JointType.BALL.value)
_SATISFIED = int(_types.ConstraintState.SATISFIED.value)
_CONE = int(_types.ConstraintState.CONE.value)
_MINIMP = float(_types.MJ_MINIMP)
_MAXIMP = float(_types.MJ_MAXIMP)
_MINVAL = float(_types.MJ_MINVAL)
_REFSAFE = int(_types.DisableBit.REFSAFE.value)
_ELLIPTIC = int(_types.ConeType.ELLIPTIC.value)
_PYRAMIDAL = int(_types.ConeType.PYRAMIDAL.value)

# Static loop bounds (the residual is gated to one free joint, nv=6, valid condim {1,3,4,6}). Used as
# compile-time literals so Warp UNROLLS the loops inside the @wp.func's and stores each iteration's
# intermediates for the backward (dynamic loops are not replayed -- see module docstring).
_FREE_NV = 6  # free-joint dof count
_MAXCONDIM = 6  # max valid MuJoCo condim; elliptic friction rows = dimid 1..condim-1
_MAX_PYRAMID_EDGES = 10  # 2*(_MAXCONDIM - 1) pyramidal edges at condim 6


# ----------------------------------------------------------------------------
# 1. Integrator (semi-implicit Euler, out-of-place). Backward-enabled: we launch its adjoint.
#    Matches forward._advance for the no-damp case (the bounce uses Euler/implicitfast no-damp).
# ----------------------------------------------------------------------------
@wp.kernel
def _advance_state(
  opt_timestep: wp.array[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  qpos_out: wp.array2d[float],
  qvel_out: wp.array2d[float],
):
  worldid, jntid = wp.tid()
  dt = opt_timestep[worldid % opt_timestep.shape[0]]
  jt = jnt_type[jntid]
  qadr = jnt_qposadr[jntid]
  dadr = jnt_dofadr[jntid]

  # NOTE: index qpos_in[worldid, k] directly -- do NOT take a row view (`row = qpos_in[worldid]`
  # then `row[k]`): the row view's adjoint does not accumulate back into qpos_in.grad in Warp.
  if jt == _FREE:
    vlx = qvel_in[worldid, dadr + 0] + dt * qacc_in[worldid, dadr + 0]
    vly = qvel_in[worldid, dadr + 1] + dt * qacc_in[worldid, dadr + 1]
    vlz = qvel_in[worldid, dadr + 2] + dt * qacc_in[worldid, dadr + 2]
    vax = qvel_in[worldid, dadr + 3] + dt * qacc_in[worldid, dadr + 3]
    vay = qvel_in[worldid, dadr + 4] + dt * qacc_in[worldid, dadr + 4]
    vaz = qvel_in[worldid, dadr + 5] + dt * qacc_in[worldid, dadr + 5]
    qvel_out[worldid, dadr + 0] = vlx
    qvel_out[worldid, dadr + 1] = vly
    qvel_out[worldid, dadr + 2] = vlz
    qvel_out[worldid, dadr + 3] = vax
    qvel_out[worldid, dadr + 4] = vay
    qvel_out[worldid, dadr + 5] = vaz
    qpos_out[worldid, qadr + 0] = qpos_in[worldid, qadr + 0] + dt * vlx
    qpos_out[worldid, qadr + 1] = qpos_in[worldid, qadr + 1] + dt * vly
    qpos_out[worldid, qadr + 2] = qpos_in[worldid, qadr + 2] + dt * vlz
    q = wp.quat(
      qpos_in[worldid, qadr + 3], qpos_in[worldid, qadr + 4], qpos_in[worldid, qadr + 5], qpos_in[worldid, qadr + 6]
    )
    qn = _math.quat_integrate(q, wp.vec3(vax, vay, vaz), dt)
    qpos_out[worldid, qadr + 3] = qn[0]
    qpos_out[worldid, qadr + 4] = qn[1]
    qpos_out[worldid, qadr + 5] = qn[2]
    qpos_out[worldid, qadr + 6] = qn[3]
  elif jt == _BALL:
    vx = qvel_in[worldid, dadr + 0] + dt * qacc_in[worldid, dadr + 0]
    vy = qvel_in[worldid, dadr + 1] + dt * qacc_in[worldid, dadr + 1]
    vz = qvel_in[worldid, dadr + 2] + dt * qacc_in[worldid, dadr + 2]
    qvel_out[worldid, dadr + 0] = vx
    qvel_out[worldid, dadr + 1] = vy
    qvel_out[worldid, dadr + 2] = vz
    q = wp.quat(
      qpos_in[worldid, qadr + 0], qpos_in[worldid, qadr + 1], qpos_in[worldid, qadr + 2], qpos_in[worldid, qadr + 3]
    )
    qn = _math.quat_integrate(q, wp.vec3(vx, vy, vz), dt)
    qpos_out[worldid, qadr + 0] = qn[0]
    qpos_out[worldid, qadr + 1] = qn[1]
    qpos_out[worldid, qadr + 2] = qn[2]
    qpos_out[worldid, qadr + 3] = qn[3]
  else:  # HINGE / SLIDE
    v = qvel_in[worldid, dadr] + dt * qacc_in[worldid, dadr]
    qvel_out[worldid, dadr] = v
    qpos_out[worldid, qadr] = qpos_in[worldid, qadr] + dt * v


# ----------------------------------------------------------------------------
# 2. IFT helper.
# ----------------------------------------------------------------------------
@wp.kernel
def _load_rhs(adj_qacc: wp.array2d[float], nv: int, grad_out: wp.array2d[float]):
  """Write adj_qacc into ctx.grad[:, :nv] (zero the padding) as the RHS of H λ = adj_qacc."""
  worldid, i = wp.tid()
  if i < nv:
    grad_out[worldid, i] = adj_qacc[worldid, i]
  else:
    grad_out[worldid, i] = 0.0


# ----------------------------------------------------------------------------
# 3. Contact residual r_contact(qpos,qvel) = -Jᵀ·f(J·qacc - aref), seeded adj_r = λ -> -(dr/dtheta)ᵀλ in
#    one AD pass. This is the CONTACT term of the stationarity residual; the SMOOTH term
#    r_smooth = M·qacc - qfrc_smooth is a planned SEPARATE sibling kernel `_residual_smooth` -- the IFT VJP
#    is linear in r = r_smooth + r_contact, so the two AD passes share the same λ and their input adjoints
#    sum (see step_backward's closing note + MJPLAN §5.4/§5.9). Omitted now: zero for the gravity-only free
#    body. Scope: one free-joint body (dofs 0-5), elliptic/pyramidal cones, condim {1,3,4,6}.
# ----------------------------------------------------------------------------
# Per-row physics shared with the forward: (k, b, impedance) from constraint._contact_kbimp and the
# elliptic cone force from solver._eval_elliptic_cone (single source of truth).  Only _contact_D
# (differentiable D recovered from the frozen converged D) is backward-only.
@wp.func
def _contact_D(D_base: float, imp_base: float, imp: float) -> float:
  """Recover fixed invweight from converged D, then evaluate D at differentiable imp."""
  invweight = (1.0 / wp.max(D_base, _MINVAL)) * imp_base / wp.max(1.0 - imp_base, _MINVAL)
  return 1.0 / wp.max(invweight * (1.0 - imp) / imp, _MINVAL)


@wp.func
def _contact_dof_coefficient(
  geom: wp.vec2i,
  dofid: int,
  geom_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
) -> float:
  """Coefficient of one DOF in a rigid-geom contact's J = J_geom1 - J_geom0.

  Shared-ancestor DOFs cancel, geom-0-only DOFs contribute -1, and
  geom-1-only DOFs contribute +1.  Flex contacts use negative geom IDs and
  require a different Jacobian construction, so they have no coefficient here.
  """
  if geom[0] < 0 or geom[1] < 0:
    return 0.0
  body0 = geom_bodyid[geom[0]]
  body1 = geom_bodyid[geom[1]]
  return float(body_isdofancestor[body1, dofid] - body_isdofancestor[body0, dofid])


@wp.func
def _frame_axis(fm: wp.mat33, idx: int) -> wp.vec3:
  """Contact-frame axis: 0 = normal, 1/2 = tangents (rows of contact.frame)."""
  if idx == 1:
    return wp.vec3(fm[1, 0], fm[1, 1], fm[1, 2])
  if idx == 2:
    return wp.vec3(fm[2, 0], fm[2, 1], fm[2, 2])
  return wp.vec3(fm[0, 0], fm[0, 1], fm[0, 2])


@wp.func
def _friction(fri: vec5, idx: int) -> float:
  """Friction coefficient by runtime row index via STATIC vec lookups (a runtime vector index has a
  corrupt Warp reverse-mode adjoint; it is only exercised in enable_backward=False forward kernels)."""
  if idx == 0:
    return fri[0]
  if idx == 1:
    return fri[1]
  if idx == 2:
    return fri[2]
  if idx == 3:
    return fri[3]
  return fri[4]


@wp.func
def _row_jaref(Jqa: float, Jqv: float, k: float, b: float, imp: float, pos_aref: float) -> float:
  """Jaref = J·qacc - aref, with aref = -k·imp·pos_aref - b·(J·qvel) (constraint._efc_row).

  Elliptic friction rows pass pos_aref=0 (no penetration reference); the normal row and every
  pyramidal edge pass pos_aref = pos (the shared normal penetration)."""
  return Jqa - (-k * imp * pos_aref - b * Jqv)


@wp.func
def _spatial_vel(
  w: int,
  cdof_in: wp.array2d[wp.spatial_vector],
  gen_in: wp.array2d[float],
) -> wp.spatial_vector:
  """Body spatial velocity (or accel) ``Σ_i gen_i · cdof_i`` = (angular, linear) in the com motion
  basis. STATIC loop inside this @wp.func so Warp unrolls + replays it for the backward."""
  v = wp.spatial_vector()
  for i in range(_FREE_NV):
    v += gen_in[w, i] * cdof_in[w, i]
  return v


@wp.func
def _row_jac(is_rot: bool, side: float, dirv: wp.vec3, rvec: wp.vec3) -> wp.spatial_vector:
  """Spatial Jacobian (line of action) of one contact row, (angular, linear).  A translational row is
  a unit force ``dirv`` at the contact point (moment rvec x dirv); a rotational row is a pure couple
  about ``dirv``.  ``side`` = the J_geom1 - J_geom0 sign (the forward builds J that way)."""
  if is_rot:
    return wp.spatial_vector(side * dirv, wp.vec3(0.0))
  return wp.spatial_vector(side * wp.cross(rvec, dirv), side * dirv)


@wp.func
def _elliptic_wrench(
  V: wp.spatial_vector,
  A: wp.spatial_vector,
  rvec: wp.vec3,
  side: float,
  fm: wp.mat33,
  fric: vec5,
  k: float,
  b: float,
  b_t: float,
  mu: float,
  imp: float,
  imp0: float,
  D0: float,
  pos: float,
  st: int,
  condim: int,
  w: int,
  efc_D_in: wp.array2d[float],
  efc_address_in: wp.array2d[int],
  cid: int,
) -> wp.spatial_vector:
  """Net body wrench of one elliptic contact: the normal row + (condim>1) the elliptic middle-zone
  cone law coupling all friction rows through ``T = sqrt(Σ_j (Jaref_j friction_j)^2)`` (j = dimid
  1..condim-1), or the bottom-zone / frictionless ``-D_row·Jaref_row``."""
  n = _frame_axis(fm, 0)
  W0 = _row_jac(False, side, n, rvec)  # normal row
  Jaref0 = _row_jaref(wp.dot(A, W0), wp.dot(V, W0), k, b, imp, pos)

  wrench = wp.spatial_vector()
  if (condim > 1) and (st == _CONE):  # middle zone: cone-coupled forces, shared T
    N = Jaref0 * mu
    TT = float(0.0)
    for j in range(1, _MAXCONDIM):  # STATIC bound + guard so Warp unrolls (dynamic loop -> stale T)
      if j < condim:
        Wj = _row_jac(j >= 3, side, _frame_axis(fm, j % 3), rvec)
        uj = _row_jaref(wp.dot(A, Wj), wp.dot(V, Wj), 0.0, b_t, 0.0, 0.0) * _friction(fric, j - 1)
        TT += uj * uj
    T = wp.sqrt(wp.max(TT, _MINVAL * _MINVAL))
    wrench += _solver._eval_elliptic_cone(N, T, D0, mu, 0.0, True)[0] * W0
    for j in range(1, _MAXCONDIM):
      if j < condim:
        frij = _friction(fric, j - 1)
        Wj = _row_jac(j >= 3, side, _frame_axis(fm, j % 3), rvec)
        Jarefj = _row_jaref(wp.dot(A, Wj), wp.dot(V, Wj), 0.0, b_t, 0.0, 0.0)
        # ufrictionj = Jaref_j * friction_j^2 (the elliptic cone tangent force).
        wrench += _solver._eval_elliptic_cone(N, T, D0, mu, Jarefj * frij * frij, False)[0] * Wj
  else:  # bottom zone / frictionless: each row force = -D_row * Jaref_row
    wrench += (-D0 * Jaref0) * W0
    for j in range(1, _MAXCONDIM):
      if j < condim:
        Wj = _row_jac(j >= 3, side, _frame_axis(fm, j % 3), rvec)
        Jarefj = _row_jaref(wp.dot(A, Wj), wp.dot(V, Wj), 0.0, b_t, 0.0, 0.0)
        Dj = _contact_D(efc_D_in[w, efc_address_in[cid, j]], imp0, imp)
        wrench += (-Dj * Jarefj) * Wj
  return wrench


@wp.func
def _pyramidal_wrench(
  V: wp.spatial_vector,
  A: wp.spatial_vector,
  rvec: wp.vec3,
  side: float,
  fm: wp.mat33,
  fric: vec5,
  k: float,
  b: float,
  imp: float,
  imp0: float,
  pos: float,
  condim: int,
  w: int,
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  efc_address_in: wp.array2d[int],
  cid: int,
) -> wp.spatial_vector:
  """Net body wrench of one pyramidal contact: ndim = 2*(condim-1) friction-pyramid edges (1 if
  condim==1).  Each edge is the normal-linear row + (±friction_k)*(k-th friction direction), with
  force = -D*(J_edge.qacc - aref), aref at the shared normal pos (constraint._efc_contact_jac)."""
  n = _frame_axis(fm, 0)
  W0 = _row_jac(False, side, n, rvec)  # normal-linear part, shared by every edge
  ndim = int(1)
  if condim > 1:
    ndim = 2 * (condim - 1)

  wrench = wp.spatial_vector()
  for e in range(_MAX_PYRAMID_EDGES):  # STATIC bound + guard so Warp unrolls (dynamic loop -> stale grads)
    if e < ndim:
      ea = efc_address_in[cid, e]
      if ea >= 0 and efc_state_in[w, ea] != _SATISFIED:
        We = W0
        if condim > 1:
          # dimid2 selects the friction direction: 1/2 = tangents (linear), 3 = torsion,
          # 4/5 = rolling (rotational); the two edges per direction take +/- the friction.
          dimid2 = e / 2 + 1
          fs = _friction(fric, dimid2 - 1) * (1.0 - 2.0 * float(e % 2))
          We += fs * _row_jac(dimid2 >= 3, side, _frame_axis(fm, dimid2 % 3), rvec)
        force = -_contact_D(efc_D_in[w, ea], imp0, imp) * _row_jaref(wp.dot(A, We), wp.dot(V, We), k, b, imp, pos)
        wrench += force * We
  return wrench


@cache_kernel
def _residual_contact(cone_type: int):
  """Per-step contact residual r = -Jᵀ f(qpos,qvel,qacc) for the IFT backward, cone-specialized.

  The kernel computes each contact's shared linearization (moment arm, body spatial vel/accel,
  position/impedance/D), delegates the per-row wrench to the cone-specific @wp.func (where the loops
  live so Warp replays them in the backward), sums the wrenches (a plain ``+=`` -- correct in a
  dynamic loop), and projects the net wrench through cdof to ``r``.  ``wp.static(IS_ELLIPTIC)``
  compiles only the relevant cone path (mirrors constraint.py's cone-specialized contact kernels)."""
  IS_ELLIPTIC = cone_type == _ELLIPTIC

  @wp.kernel(module="unique", enable_backward=True)
  def kernel(
    qpos_in: wp.array2d[float],  # [grad]
    qvel_in: wp.array2d[float],  # [grad]
    qacc_in: wp.array2d[float],  # frozen
    cdof_in: wp.array2d[wp.spatial_vector],  # frozen motion-dof axes (the contact Jacobian basis)
    com0_in: wp.array2d[float],  # frozen linearization qpos (read [:3] = com)
    efc_D_in: wp.array2d[float],  # frozen
    efc_state_in: wp.array2d[int],  # frozen active set
    efc_pos_in: wp.array2d[float],  # frozen row position (normal: contact dist)
    efc_margin_in: wp.array2d[float],  # frozen include margin
    contact_pos_in: wp.array(dtype=wp.vec3),
    contact_frame_in: wp.array(dtype=wp.mat33),  # per-contact frame; rows = normal, tangent1, tangent2
    contact_friction_in: wp.array(dtype=vec5),
    contact_solref_in: wp.array(dtype=wp.vec2),
    contact_solreffriction_in: wp.array(dtype=wp.vec2),
    contact_solimp_in: wp.array(dtype=vec5),
    contact_dim_in: wp.array[int],
    contact_geom_in: wp.array(dtype=wp.vec2i),
    geom_bodyid_in: wp.array[int],
    body_isdofancestor_in: wp.array2d[int],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    nacon_in: wp.array[int],
    opt_timestep: wp.array[float],
    opt_impratio_invsqrt: wp.array[float],
    opt_disableflags: int,
    r_out: wp.array2d[float],  # out: contact residual (pre-zeroed; free-joint dofs 0-5)
  ):
    w = wp.tid()
    com = wp.vec3(qpos_in[w, 0], qpos_in[w, 1], qpos_in[w, 2])
    com0 = wp.vec3(com0_in[w, 0], com0_in[w, 1], com0_in[w, 2])
    dt = opt_timestep[w % opt_timestep.shape[0]]
    imp_isq = opt_impratio_invsqrt[w % opt_impratio_invsqrt.shape[0]]

    wtot = wp.spatial_vector()  # net body wrench from all contacts in this world
    for cid in range(nacon_in[0]):
      if contact_worldid_in[cid] != w:
        continue
      e0 = contact_efc_address_in[cid, 0]
      if e0 < 0:
        continue
      st = efc_state_in[w, e0]
      if st == _SATISFIED:
        continue
      # Production J = jac(geom1) - jac(geom0); for one free body all six dofs share ancestry, so the
      # per-dof coefficient collapses to this single sign (step_backward gates the supported shape).
      side = _contact_dof_coefficient(contact_geom_in[cid], 0, geom_bodyid_in, body_isdofancestor_in)
      if side == 0.0:
        continue

      condim = contact_dim_in[cid]
      fm = contact_frame_in[cid]
      fric = contact_friction_in[cid]
      n = _frame_axis(fm, 0)

      # Contact point tracks qpos: sphere-vs-flat midpoint => dcpos/dcom = I - 1/2 n nᵀ.
      dcom = com - com0
      cpos_eff = contact_pos_in[cid] + (dcom - 0.5 * wp.dot(n, dcom) * n)
      rvec = cpos_eff - com  # moment arm contact_pos - com (com = subtree_com for the free body)
      V = _spatial_vel(w, cdof_in, qvel_in)
      A = _spatial_vel(w, cdof_in, qacc_in)

      # Position, impedance, D, k, b: mirror constraint._efc_row (pos_imp = dist - margin = pos0).
      solref = contact_solref_in[cid]
      solimp = contact_solimp_in[cid]
      pos0 = efc_pos_in[w, e0] - efc_margin_in[w, e0]
      pos = pos0 + side * wp.dot(n, dcom)
      imp0 = _constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos0)[2]
      kbimp_n = _constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos)
      k = kbimp_n[0]
      b = kbimp_n[1]
      imp = kbimp_n[2]
      D0 = _contact_D(efc_D_in[w, e0], imp0, imp)

      if wp.static(IS_ELLIPTIC):
        # Friction rows use solreffriction (if set) for b, and pos_aref = 0.
        ref_t = solref
        solreffriction = contact_solreffriction_in[cid]
        if solreffriction[0] != 0.0 or solreffriction[1] != 0.0:
          ref_t = solreffriction
        b_t = _constraint._contact_kbimp(opt_disableflags, dt, ref_t, solimp, pos)[1]
        mu = fric[0] * imp_isq
        wtot += _elliptic_wrench(
          V, A, rvec, side, fm, fric, k, b, b_t, mu, imp, imp0, D0, pos, st, condim,
          w, efc_D_in, contact_efc_address_in, cid,
        )
      else:
        wtot += _pyramidal_wrench(
          V, A, rvec, side, fm, fric, k, b, imp, imp0, pos, condim,
          w, efc_D_in, efc_state_in, contact_efc_address_in, cid,
        )

    # r = -Jᵀ force: project the net body wrench back through cdof to the free-joint dofs.
    for i in range(_FREE_NV):
      r_out[w, i] = -wp.dot(cdof_in[w, i], wtot)

  return kernel


@wp.kernel
def _copy_cols(src: wp.array2d[float], dst: wp.array2d[float]):
  """dst[w,i] = src[w,i] over dst's columns (seed r.grad = λ from ctx.Mgrad[:, :nv])."""
  w, i = wp.tid()
  dst[w, i] = src[w, i]


@wp.kernel
def _sub_write(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float]):
  """out = a - b: integrator-direct adjoint minus the residual-VJP scatter (dr/dtheta)ᵀλ."""
  w, i = wp.tid()
  out[w, i] = a[w, i] - b[w, i]


# ----------------------------------------------------------------------------
# The analytic step backward (registered with forward.py).
# ----------------------------------------------------------------------------
def step_backward(m: Model, d: Data, d_out: Data):
  """Reads d_out.{qpos,qvel}.grad (upstream), writes d.{qpos,qvel}.grad. Uses d as the input
  state and d_out for the step's intermediates (converged qacc, efc.*, contact.*, cdof)."""
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  nv_pad = m.nv_pad
  # The residual below specializes the general per-DOF contact difference to
  # one free joint, for which all six DOFs have the same ancestry coefficient.
  # Shape alone is insufficient (other joint combinations can also total
  # nq=7,nv=6), so include the topology and contact-family assumptions.
  if m.njnt != 1 or nq != 7 or nv != 6:
    raise NotImplementedError(
      "adjoint.step_backward currently supports exactly one free joint (nq=7, nv=6); "
      "general contact requires per-dof J_geom1 - J_geom0 contributions"
    )
  if m.nflex != 0:
    raise NotImplementedError("adjoint.step_backward does not support flex contacts")
  if m.opt.cone != _ELLIPTIC and m.opt.cone != _PYRAMIDAL:
    raise NotImplementedError("adjoint.step_backward supports only elliptic/pyramidal cones")
  # condim is mirrored generically (1/3/4/6 -- the valid MuJoCo set; 2/5 cannot be loaded), so no
  # nmaxcondim gate is needed: rotational rows (dimid>=3) are handled in both cone branches.
  residual_contact_kernel = _residual_contact(int(m.opt.cone))  # cone-specialized kernel (cached)

  # --- 1. integrator adjoint: adj(qpos',qvel') -> adj_qacc + integrator-direct adj(qpos,qvel) ---
  adj_qpos = wp.zeros((nworld, nq), dtype=float)
  adj_qvel = wp.zeros((nworld, nv), dtype=float)
  adj_qacc = wp.zeros((nworld, nv), dtype=float)
  qpos_s = wp.empty_like(d.qpos)
  qvel_s = wp.empty_like(d.qvel)
  int_inputs = [m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, d.qvel, d_out.qacc]
  wp.launch(_advance_state, dim=(nworld, m.njnt), inputs=int_inputs, outputs=[qpos_s, qvel_s])
  wp.launch(
    _advance_state,
    dim=(nworld, m.njnt),
    inputs=int_inputs,
    outputs=[qpos_s, qvel_s],
    adj_inputs=[None, None, None, None, adj_qpos, adj_qvel, adj_qacc],
    adj_outputs=[d_out.qpos.grad, d_out.qvel.grad],
    adjoint=True,
  )

  # --- 2. IFT: solve H λ = adj_qacc, reusing the solver's assembly + Cholesky at converged qacc ---
  ctx = _solver._create_solver_context(m, d_out)
  _solver.init_context(m, d_out, ctx, grad=True)  # assembles + factors ctx.h; active set = forward's
  wp.launch(_load_rhs, dim=(nworld, nv_pad), inputs=[adj_qacc, nv], outputs=[ctx.grad])
  ctx.done.zero_()
  _solver._cholesky_factorize_solve(m, d_out, ctx)  # ctx.Mgrad[:, :nv] = λ
  lam = ctx.Mgrad

  # --- 3. residual-VJP: adj_theta = integrator-direct - (dr/dtheta)ᵀλ via AD of the contact residual ---
  com0 = wp.clone(d.qpos)  # frozen linearization reference (the kernel reads [:3])
  r = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  res_qvel = wp.zeros((nworld, nv), dtype=float)
  rin = [
    d.qpos,
    d.qvel,
    d_out.qacc,
    d_out.cdof,
    com0,
    d_out.efc.D,
    d_out.efc.state,
    d_out.efc.pos,
    d_out.efc.margin,
    d_out.contact.pos,
    d_out.contact.frame,
    d_out.contact.friction,
    d_out.contact.solref,
    d_out.contact.solreffriction,
    d_out.contact.solimp,
    d_out.contact.dim,
    d_out.contact.geom,
    m.geom_bodyid,
    m.body_isdofancestor,
    d_out.contact.efc_address,
    d_out.contact.worldid,
    d_out.nacon,
    m.opt.timestep,
    m.opt.impratio_invsqrt,
    m.opt.disableflags,
  ]
  wp.launch(residual_contact_kernel, dim=nworld, inputs=rin, outputs=[r])
  wp.launch(_copy_cols, dim=(nworld, nv), inputs=[lam], outputs=[r.grad])  # seed adj_r = λ
  wp.launch(
    residual_contact_kernel,
    dim=nworld,
    inputs=rin,
    outputs=[r],
    adj_inputs=[res_qpos, res_qvel] + [None] * 23,
    adj_outputs=[r.grad],
    adjoint=True,
  )

  # --- write input adjoints (each d == datas[t] is the d_in of exactly one step) ---
  wp.launch(_sub_write, dim=(nworld, nq), inputs=[adj_qpos, res_qpos], outputs=[d.qpos.grad])
  wp.launch(_sub_write, dim=(nworld, nv), inputs=[adj_qvel, res_qvel], outputs=[d.qvel.grad])
  # SMOOTH residual term r_smooth = M·qacc - qfrc_smooth (-> ctrl/actuator grads, sys-id of
  # mass/inertia/damping, ∂M/∂qpos, ∂qfrc_bias/∂qpos): a planned SEPARATE kernel `_residual_smooth`. The
  # IFT VJP is linear in r = r_smooth + r_contact, so AD it with the SAME λ and SUM its adj_{qpos,qvel,...}
  # into the writes above (λ already uses the full H = M + JᵀG_sJ; no double-count -- M in H is ∂r/∂qacc,
  # ∂(M·qacc)/∂θ in r_smooth is w.r.t. θ). Omitted now: qpos/qvel-independent for the gravity-only free
  # body. See MJPLAN §5.4 / §5.9.


_forward.register_backward_hook(step_backward)
