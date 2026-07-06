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

import dataclasses
from typing import Tuple

import warp as wp

from mujoco_warp._src import collision_adjoint as _collision_adjoint
from mujoco_warp._src import constraint as _constraint
from mujoco_warp._src import constraint_adjoint as _constraint_adjoint
from mujoco_warp._src import derivative as _derivative
from mujoco_warp._src import forward as _forward
from mujoco_warp._src import forward_next as _forward_next
from mujoco_warp._src import math as _math
from mujoco_warp._src import passive as _passive
from mujoco_warp._src import smooth as _smooth
from mujoco_warp._src import solver as _solver
from mujoco_warp._src import support as _support
from mujoco_warp._src import types as _types
from mujoco_warp._src import util_misc as _util_misc
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec10f
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import cache_kernel

# --- Safe sqrt for the contact adjoint (active whenever this adjoint module is imported) ----------
# wp.sqrt's reverse is 0.5/sqrt(x) -> inf at x=0. Warp reverse-mode differentiates BOTH arms of a
# wp.where, so `wp.where(cond, ...wp.sqrt(0)..., safe)` yields 0*inf = NaN in the adjoint even when the
# FORWARD picks the safe arm -- e.g. collision_primitive_core.plane_cylinder's degenerate len_sqr=0 (a
# cylinder face resting flat on a plane, axis || plane normal), which made the cylinder<->plane contact
# gradient all-NaN while the forward + FD were finite. We swap wp.sqrt -> safe_sqrt: forward is
# byte-identical (returns wp.sqrt(x)), but its custom grad is 0 at x<=0 (a 0-at-0 sqrt grad is always the
# desired value; inf never is). @wp.func_grad rejects generic (Any) args, so we register one concrete
# overload per float dtype. Global + import-time: it applies wherever the differentiated kernels compile
# (adjoint is imported before any tape.backward) and leaves the forward untouched. Fixes any
# sqrt(0)-in-a-where-branch degeneracy, not just cylinder-plane.
_wp_sqrt = wp.sqrt


@wp.func
def safe_sqrt(x: wp.float32):
  return _wp_sqrt(x)


@wp.func_grad(safe_sqrt)
def _adj_safe_sqrt_f32(x: wp.float32, adj_ret: wp.float32):
  if x > wp.float32(0.0):
    wp.adjoint[x] += adj_ret / (wp.float32(2.0) * _wp_sqrt(x))


@wp.func
def safe_sqrt(x: wp.float64):  # concrete overload (same name) so float64 sqrt still works
  return _wp_sqrt(x)


@wp.func_grad(safe_sqrt)
def _adj_safe_sqrt_f64(x: wp.float64, adj_ret: wp.float64):
  if x > wp.float64(0.0):
    wp.adjoint[x] += adj_ret / (wp.float64(2.0) * _wp_sqrt(x))


wp.sqrt = safe_sqrt

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
_QUADRATIC = int(_types.ConstraintState.QUADRATIC.value)
_LINEARNEG = int(_types.ConstraintState.LINEARNEG.value)  # saturated friction, force = +frictionloss
_LINEARPOS = int(_types.ConstraintState.LINEARPOS.value)  # saturated friction, force = -frictionloss
_FRICTION_DOF = int(_types.ConstraintType.FRICTION_DOF.value)
_EQUALITY = int(_types.ConstraintType.EQUALITY.value)
_LIMIT_JOINT = int(_types.ConstraintType.LIMIT_JOINT.value)
_MINIMP = float(_types.MJ_MINIMP)
_MAXIMP = float(_types.MJ_MAXIMP)
_MINVAL = float(_types.MJ_MINVAL)
_REFSAFE = int(_types.DisableBit.REFSAFE.value)
_ELLIPTIC = int(_types.ConeType.ELLIPTIC.value)
_PYRAMIDAL = int(_types.ConeType.PYRAMIDAL.value)
_EULER = int(_types.IntegratorType.EULER.value)
_IMPLICITFAST = int(_types.IntegratorType.IMPLICITFAST.value)
_EULERDAMP = int(_types.DisableBit.EULERDAMP.value)
_DAMPER = int(_types.DisableBit.DAMPER.value)

# Static loop bounds (the residual is gated to one free joint, nv=6, valid condim {1,3,4,6}). Used as
# compile-time literals so Warp UNROLLS the loops inside the @wp.func's and stores each iteration's
# intermediates for the backward (dynamic loops are not replayed -- see module docstring).
_FREE_NV = 6  # free-joint dof count
_MAXCONDIM = 6  # max valid MuJoCo condim; elliptic friction rows = dimid 1..condim-1
_MAX_PYRAMID_EDGES = 10  # 2*(_MAXCONDIM - 1) pyramidal edges at condim 6
_MAX_NV = 16  # static unroll bound for the per-dof NON-CONTACT constraint residual reductions (_residual_constraint,
# nv<=16). The CONTACT residual no longer uses it (the sparse contract-first gather/phi/scatter below, S4 done);
# _residual_constraint's sparse+CSR rework is the next stage -- step_backward RAISES for nv>_MAX_NV with active-able
# non-contact rows rather than silently truncating (no FD fallback).

# Production default: step_backward §5 uses the ANALYTIC reduced smooth-force replay (smooth_adjoint.py,
# MJPLAN_ADRNE §0/§10): rigid-body RNE bias + joint springs (all joint types) + AFFINE joint actuators. It
# RAISES (assert_smooth_supported) on any enabled smooth feature without an analytic leaf -- NO silent FD
# fallback. Setting this False explicitly selects FD-of-rne as a TEST ORACLE only; production never does so
# automatically. The analytic-vs-FD A/B and float64 end-to-end gates live in smooth_adjoint_test / adjoint_test.
_USE_ANALYTIC_RNE_QPOS = True

# When True (default), the CONTACT residual VJP uses the SPARSE contract-first path (MJPLAN_ARTICULATION S4):
# a manual gather of the three contact-point spatial motions V/A/Z over only the symmetric-difference of the
# two geoms' ancestor-dof chains, a loop-free source-AD cone leaf φ = -Z·F(V,A,ξ) (seed adj_φ=+1), and a
# manual scatter back to the SAME five contact seeds + res_qvel. nv-GENERAL (no _MAX_NV; cost O(chain+condim)
# per contact). When False, step_backward uses the legacy dense `_residual_contact` monolithic autodiff kernel
# (capped at nv<=16) -- retained as an EXACT A/B oracle for nv<=16 scenes (adjoint_test A/B), never production.
_USE_SPARSE_CONTACT = True


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

  # semi-implicit: update qvel, then integrate qpos with it, via the shared forward_next funcs.
  # next_velocity returns the value (held as a local) so qpos uses it without reading qvel_out back.
  qvel_lin = wp.vec3(0.0, 0.0, 0.0)
  qvel_ang = wp.vec3(0.0, 0.0, 0.0)
  if jt == _FREE:
    vlx = _forward_next.next_velocity(worldid, dadr + 0, opt_timestep, qvel_in, qacc_in, 1.0)
    vly = _forward_next.next_velocity(worldid, dadr + 1, opt_timestep, qvel_in, qacc_in, 1.0)
    vlz = _forward_next.next_velocity(worldid, dadr + 2, opt_timestep, qvel_in, qacc_in, 1.0)
    vax = _forward_next.next_velocity(worldid, dadr + 3, opt_timestep, qvel_in, qacc_in, 1.0)
    vay = _forward_next.next_velocity(worldid, dadr + 4, opt_timestep, qvel_in, qacc_in, 1.0)
    vaz = _forward_next.next_velocity(worldid, dadr + 5, opt_timestep, qvel_in, qacc_in, 1.0)
    qvel_out[worldid, dadr + 0] = vlx
    qvel_out[worldid, dadr + 1] = vly
    qvel_out[worldid, dadr + 2] = vlz
    qvel_out[worldid, dadr + 3] = vax
    qvel_out[worldid, dadr + 4] = vay
    qvel_out[worldid, dadr + 5] = vaz
    qvel_lin = wp.vec3(vlx, vly, vlz)
    qvel_ang = wp.vec3(vax, vay, vaz)
  elif jt == _BALL:
    vx = _forward_next.next_velocity(worldid, dadr + 0, opt_timestep, qvel_in, qacc_in, 1.0)
    vy = _forward_next.next_velocity(worldid, dadr + 1, opt_timestep, qvel_in, qacc_in, 1.0)
    vz = _forward_next.next_velocity(worldid, dadr + 2, opt_timestep, qvel_in, qacc_in, 1.0)
    qvel_out[worldid, dadr + 0] = vx
    qvel_out[worldid, dadr + 1] = vy
    qvel_out[worldid, dadr + 2] = vz
    qvel_ang = wp.vec3(vx, vy, vz)
  else:  # HINGE / SLIDE
    v = _forward_next.next_velocity(worldid, dadr, opt_timestep, qvel_in, qacc_in, 1.0)
    qvel_out[worldid, dadr] = v
    qvel_lin = wp.vec3(v, 0.0, 0.0)

  _forward_next.next_position(jt, qadr, dt, qpos_in, worldid, qvel_lin, qvel_ang, qpos_out)


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
# 3. Contact residual VJP leaves (the elliptic/pyramidal cone force law) live in
#    constraint_adjoint.py -- contact rows are constraint (efc) rows, alongside the non-contact
#    equality/limit/friction kernels there. adjoint.contact_residual_backward orchestrates them.
# ----------------------------------------------------------------------------


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


@wp.kernel
def _accum_cols(a: wp.array2d[float], out: wp.array2d[float]):
  """out += a (accumulate the analytic RNE-bias ∂qpos into the residual buffer that _sub_write subtracts)."""
  w, i = wp.tid()
  out[w, i] = out[w, i] + a[w, i]


# ----------------------------------------------------------------------------
# 4. Smooth residual r_smooth = M·qacc - qfrc_smooth, contracted with the SAME IFT λ (the VJP is linear
#    in r = r_smooth + r_contact, so the two share λ and their input-adjoints sum). ∂qvel / ∂ctrl are
#    analytic here; the analytic reduced-RNE ∂qpos replay is composed in §5.
#    adj_θ_smooth = -(∂r_smooth/∂θ)ᵀλ.
# ----------------------------------------------------------------------------
@wp.kernel
def _smooth_qvel_vjp(
  qD_fullm_i: wp.array[int],  # D-structure (full square) row index of entry e
  qD_fullm_j: wp.array[int],  # D-structure column index of entry e
  M_D: wp.array2d[float],  # mass matrix in D-structure (M mapped via mapM2D)
  qLU: wp.array2d[float],  # assembled so (M_D - qLU)/dt = ∂qfrc_smooth/∂qvel (deriv_smooth_vel + rne, no subtract)
  lam: wp.array2d[float],  # IFT multiplier λ (cols 0:nv valid)
  opt_timestep: wp.array[float],
  adj_qvel: wp.array2d[float],  # += Gᵀλ  (G = ∂qfrc_smooth/∂qvel = (M_D - qLU)/dt)
):
  """Add adj_qvel_smooth = -(∂r_smooth/∂qvel)ᵀλ = +Gᵀλ to adj_qvel, with G = ∂qfrc_smooth/∂qvel =
  (M_D - qLU)/dt. The D-structure (qD_fullm) stores the full asymmetric square, so the transpose
  (Gᵀλ)[col] += G[row,col]·λ[row] is exact even though the Coriolis block is non-symmetric."""
  w, e = wp.tid()
  dt = opt_timestep[w % opt_timestep.shape[0]]
  g = (M_D[w, e] - qLU[w, e]) / dt
  wp.atomic_add(adj_qvel[w], qD_fullm_j[e], g * lam[w, qD_fullm_i[e]])


@wp.kernel
def _smooth_ctrl_vjp(
  moment_rownnz: wp.array2d[int],
  moment_rowadr: wp.array2d[int],
  moment_colind: wp.array2d[int],
  actuator_moment: wp.array2d[float],  # sparse actuator_moment (frozen)
  actuator_gainprm: wp.array2d[vec10f],  # gain = prm[0] for FIXED gaintype (motor/position)
  lam: wp.array2d[float],
  adj_ctrl: wp.array2d[float],  # = (∂qfrc_actuator/∂ctrl)ᵀλ = gain·(momentᵀλ)
):
  """∂ctrl actuation leaf. qfrc_actuator = momentᵀ·force, force = gain·ctrl (FIXED gaintype), so
  adj_ctrl_smooth = -(∂r_smooth/∂ctrl)ᵀλ = +(∂qfrc_actuator/∂ctrl)ᵀλ = gain·(momentᵀλ). AFFINE-gain
  actuators (gain depends on length/vel) are not yet handled (S1 = motors)."""
  w, actid = wp.tid()
  rownnz = moment_rownnz[w, actid]
  rowadr = moment_rowadr[w, actid]
  mtl = float(0.0)
  for i in range(rownnz):
    sparseid = rowadr + i
    mtl += actuator_moment[w, sparseid] * lam[w, moment_colind[w, sparseid]]
  gain = actuator_gainprm[w % actuator_gainprm.shape[0], actid][0]
  adj_ctrl[w, actid] = gain * mtl


@wp.kernel
def _neg_cols(src: wp.array2d[float], dst: wp.array2d[float]):
  """dst[w,i] = -src[w,i] over dst's columns (seed adj_r = -λ for the smooth-param residual, so its
  adjoint launch drops -(∂r/∂θ)ᵀλ straight into m.<param>.grad)."""
  w, i = wp.tid()
  dst[w, i] = -src[w, i]


@wp.kernel
def _accum_neg(res: wp.array2d[float], grad: wp.array2d[float]):
  """grad -= res over a per-(world,dof) FLOAT model param: the IFT minus (residual seeded with +λ), summed
  across steps/worlds. Generic over the param (NOT per-param), e.g. dof_frictionloss."""
  w, i = wp.tid()
  grad[w, i] -= res[w, i]


@wp.kernel
def _accum_neg_vec2(res: wp.array2d[wp.vec2], grad: wp.array2d[wp.vec2]):
  """grad -= res over a per-constraint wp.vec2 model param (e.g. *_solref); the vec2 IFT minus-accumulate."""
  w, i = wp.tid()
  grad[w, i] -= res[w, i]


@wp.kernel
def _accum_neg_vec5(res: wp.array2d[vec5], grad: wp.array2d[vec5]):
  """grad -= res over a per-constraint vec5 model param (e.g. *_solimp); the vec5 IFT minus-accumulate."""
  w, i = wp.tid()
  grad[w, i] -= res[w, i]


@wp.kernel
def _accum_neg_vec3(res: wp.array2d[wp.vec3], grad: wp.array2d[wp.vec3]):
  """grad -= res over a per-body wp.vec3 model param (e.g. body_inertia, body_ipos); the vec3 IFT minus-accumulate."""
  w, i = wp.tid()
  grad[w, i] -= res[w, i]


@wp.kernel
def _accum_neg_quat(res: wp.array2d[wp.quat], grad: wp.array2d[wp.quat]):
  """grad -= res over a per-body wp.quat model param (body_iquat); the quat IFT minus-accumulate."""
  w, i = wp.tid()
  grad[w, i] -= res[w, i]


@wp.kernel(module="unique", enable_backward=True)
def _residual_smooth_local(
  qvel_in: wp.array2d[float],  # frozen input velocity (the EULER passive-force linearization point)
  qacc_in: wp.array2d[float],  # frozen converged acceleration
  dof_armature_in: wp.array2d[float],  # [grad target] reflected rotor inertia (added to the M diagonal)
  dof_damping_in: wp.array2d[float],  # [grad target] viscous joint damping
  r_out: wp.array2d[float],
):
  """LOCAL smooth residual ``r[i] = dof_armature[i]·qacc[i] + dof_damping[i]·qvel[i]`` -- exactly the terms
  of ``r_smooth = M·qacc - qfrc_smooth`` that depend on armature/damping (armature adds to the M diagonal,
  so ∂(M·qacc)[i]/∂armature[i] = qacc[i]; the viscous passive force is -damping·qvel (passive.py), so
  -qfrc_passive contributes +damping·qvel ⇒ ∂r[i]/∂damping[i] = qvel[i]). The RNE rigid-body M·qacc,
  springs, and actuator do NOT depend on these two params, so they are excluded -- no RNE/CRB tree.
  AD'd with the same IFT λ: the param adjoints adj_θ = -(∂r/∂θ)ᵀλ fall out of one launch
  (§5.11 near-free rider). Gains/springs are the next terms here (Stage 2);
  mass/inertia (the RNE tree) are the AD-of-rne follow-up."""
  w, i = wp.tid()
  arm = dof_armature_in[w % dof_armature_in.shape[0], i]
  dmp = dof_damping_in[w % dof_damping_in.shape[0], i]
  r_out[w, i] = arm * qacc_in[w, i] + dmp * qvel_in[w, i]


@wp.kernel(module="unique", enable_backward=True)
def _residual_constraint(
  dpos_in: wp.array2d[float],  # [grad: TANGENT ∂qpos] dof-space perturbation seed (zeros); lifted via _dof_to_qpos
  qvel_in: wp.array2d[float],  # [grad: ∂qvel state] input velocity
  qacc_in: wp.array2d[float],  # frozen converged accel
  efc_J_in: wp.array3d[float],  # frozen constraint Jacobian (dense)
  efc_D_in: wp.array2d[float],  # frozen constraint mass
  efc_pos_in: wp.array2d[float],  # frozen efc position (= pos_aref + margin)
  efc_margin_in: wp.array2d[float],  # frozen efc margin
  efc_state_in: wp.array2d[int],  # frozen active set
  efc_type_in: wp.array2d[int],  # ConstraintType per row
  efc_id_in: wp.array2d[int],  # source id (dofid / eqid / jntid)
  dof_frictionloss_in: wp.array2d[float],  # [grad target] joint Coulomb friction
  dof_solref_in: wp.array2d[wp.vec2],
  dof_solimp_in: wp.array2d[vec5],
  eq_solref_in: wp.array2d[wp.vec2],
  eq_solimp_in: wp.array2d[vec5],
  jnt_solref_in: wp.array2d[wp.vec2],
  jnt_solimp_in: wp.array2d[vec5],
  nefc_in: wp.array[int],
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  nv: int,
  r_out: wp.array2d[float],
):
  """NON-CONTACT constraint residual ``r += -Jᵀf`` over the frozen efc rows -- equality, joint limit
  (slide/hinge/BALL), dof Coulomb friction -- mirroring the forward force law (``solver._eval_constraint``)
  gated by the FROZEN ``efc.state``: SATISFIED -> 0; QUADRATIC -> ``-D·jaref`` (equality / active limit /
  stuck friction); LINEARNEG/LINEARPOS -> ``±frictionloss`` (saturated friction). The QUADRATIC ``jaref`` is
  ``J·qacc + k·imp·pos_aref + b·(J·qvel)``, with the constraint violation tracked in TANGENT space:
  ``pos_aref = pos0 + J·δθ`` where ``δθ = dpos_in`` is the per-dof perturbation (J = frozen ``efc.J``;
  impedance ``k,imp,b`` frozen @ ``pos0``). So the STIFFNESS ∂(tangent) (``k·imp·J``), velocity coupling
  ``b·J``, and ``J`` all fall out -- and ``∂r/∂δθ`` (``res_dof``) is lifted to ``∂r/∂qpos`` OUTSIDE the
  kernel by ``collision_adjoint._dof_to_qpos`` (1:1 for slide/hinge/free-translation, ``2·q⊗[0,g]`` for the
  free/BALL quaternion), so this is correct for EVERY joint type incl. the nq!=nv quaternion joints. The
  force is piecewise-LINEAR (no nonlinear reduction, unlike the contact cone) -> AD-safe in the dynamic row
  loop. AD'd with the SAME IFT λ: ``dof_frictionloss``'s adjoint (Coulomb sys-id) + the per-class
  solref/solimp adjoints + the rows' tangent ∂qpos/∂qvel state-grad fall out of one launch. (frictionloss
  read from the MODEL param; the dof_frictionloss->efc copy is unrecorded, like geom_friction->contact.)
  FRICTION_TENDON / LIMIT_TENDON extend the type filter."""
  w = wp.tid()
  dt = opt_timestep[w % opt_timestep.shape[0]]
  for row in range(nefc_in[w]):  # dynamic loop over this world's efc rows (PIECEWISE-LINEAR -> AD-safe)
    ty = efc_type_in[w, row]
    if ty != _EQUALITY and ty != _LIMIT_JOINT and ty != _FRICTION_DOF:
      continue  # contact rows -> _residual_contact; LIMIT_TENDON / FRICTION_TENDON -> TODO
    st = efc_state_in[w, row]
    if st == _SATISFIED:
      continue
    cid = efc_id_in[w, row]  # source id: dofid (FRICTION_DOF) / eqid (EQUALITY) / jntid (LIMIT_JOINT)
    if ty == _FRICTION_DOF and st == _LINEARNEG:  # saturated friction: force = +frictionloss
      f = dof_frictionloss_in[w % dof_frictionloss_in.shape[0], cid]
    elif ty == _FRICTION_DOF and st == _LINEARPOS:  # saturated friction: force = -frictionloss
      f = -dof_frictionloss_in[w % dof_frictionloss_in.shape[0], cid]
    else:  # QUADRATIC: equality (bilateral) / active limit / stuck friction -> force = -D·jaref
      pos_aref0 = efc_pos_in[w, row] - efc_margin_in[w, row]  # frozen signed violation (impedance ref)
      if ty == _FRICTION_DOF:  # per-type solref/solimp -> (k, b, imp) at the frozen pos
        sid = w % dof_solref_in.shape[0]
        kbi = _constraint._contact_kbimp(opt_disableflags, dt, dof_solref_in[sid, cid], dof_solimp_in[sid, cid], pos_aref0)
      elif ty == _EQUALITY:
        sid = w % eq_solref_in.shape[0]
        kbi = _constraint._contact_kbimp(opt_disableflags, dt, eq_solref_in[sid, cid], eq_solimp_in[sid, cid], pos_aref0)
      else:  # _LIMIT_JOINT (slide/hinge: scalar J; ball: the 3-dof angular -axis J)
        sid = w % jnt_solref_in.shape[0]
        kbi = _constraint._contact_kbimp(opt_disableflags, dt, jnt_solref_in[sid, cid], jnt_solimp_in[sid, cid], pos_aref0)
      Jqa = float(0.0)
      Jqv = float(0.0)
      pos_d = float(0.0)
      for i in range(_MAX_NV):
        if i < nv:
          jj = efc_J_in[w, row, i]
          Jqa += jj * qacc_in[w, i]
          Jqv += jj * qvel_in[w, i]
          pos_d += jj * dpos_in[w, i]  # ∂pos_aref/∂δθ = efc.J (TANGENT; lifted to qpos by _dof_to_qpos)
      jaref = Jqa + kbi[0] * kbi[2] * (pos_aref0 + pos_d) + kbi[1] * Jqv
      f = -efc_D_in[w, row] * jaref
    for i in range(_MAX_NV):
      if i < nv:
        r_out[w, i] += -f * efc_J_in[w, row, i]


# NON-CONTACT constraint residual VJP -- SPARSE/CSR + nv-general (MJPLAN_CSR.md), the sibling of the S4
# contact _residual; orchestrates the three constraint_adjoint kernels (gather -> loop-free anchored row leaf
# -> Jᵀ scatter) as one piece of step_backward. REUSES ctx.Jaref (= J·qacc - aref) and efc.vel (= J·qvel)
# as the FROZEN value anchors -- only Z = J·λ needs a fresh reduction. The dense/CSR iterator is the MODEL's
# m.is_sparse; the legacy/new routing (vs the dense _residual_constraint) is the structural
# _model_has_unsupported_noncontact_rows predicate in step_backward -- NO global on/off flag.
def _residual_constraint_sparse(m: Model, d_out: Data, ctx_Jaref: wp.array, lam: wp.array,
                                res_qvel: wp.array, res_dof: wp.array):
  """Run the SPARSE/CSR non-contact constraint residual VJP (gather Z + true-invweight -> loop-free anchored
  leaf φ=-Z·f -> Jᵀ scatter), accumulating ``+∂φ/∂qvel`` into ``res_qvel`` (ALL rows) and the TANGENT
  ``+∂φ/∂qpos`` into ``res_dof`` (POSITION-BEARING rows only). The caller lifts res_dof -> res_qpos
  (_dof_to_qpos) and applies the IFT minus (_sub_write: grad = direct - res). PARAM sys-id (dof_frictionloss
  + per-class solref/solimp) accumulates ``-(∂r/∂θ)ᵀλ`` into ``m.<param>.grad`` in-place (requires_grad-gated,
  IFT minus via _accum_neg*). nv-general; ``m.is_sparse`` selects the dense/CSR iterator."""
  nworld = d_out.qpos.shape[0]
  njmax = d_out.efc.type.shape[1]  # row capacity (njmax); efc.J row dim is njmax_pad >= njmax
  nv = m.nv
  Z = wp.zeros((nworld, njmax), dtype=float, requires_grad=True)
  invw = wp.zeros((nworld, njmax), dtype=float)
  gather = _constraint_adjoint._constraint_gather(m.is_sparse)
  wp.launch(
    gather,
    dim=(nworld, njmax),
    inputs=[m.jnt_dofadr, m.dof_invweight0, lam, d_out.efc.J_rownnz, d_out.efc.J_rowadr, d_out.efc.J_colind,
            d_out.efc.J, d_out.efc.state, d_out.efc.type, d_out.efc.id, d_out.nefc, nv],
    outputs=[Z, invw],
  )
  for _arr in (d_out.efc.pos, d_out.efc.vel):
    _arr.requires_grad = True  # so the leaf adjoint accumulates P̄ / V̄ into the res buffers
  phi = wp.zeros((nworld, njmax), dtype=float, requires_grad=True)
  res_pos = wp.zeros((nworld, njmax), dtype=float)  # P̄ per row
  res_vel = wp.zeros((nworld, njmax), dtype=float)  # V̄ per row
  leaf_in = [d_out.efc.pos, d_out.efc.vel, Z, invw, d_out.efc.margin, d_out.efc.aref, d_out.efc.D,
             d_out.efc.force, ctx_Jaref, d_out.efc.state, d_out.efc.type, d_out.efc.id,
             m.dof_solref, m.dof_solimp, m.dof_frictionloss, m.eq_solref, m.eq_solimp, m.jnt_solref,
             m.jnt_solimp, d_out.nefc, m.opt.timestep, m.opt.disableflags]
  wp.launch(_constraint_adjoint._constraint_row_phi, dim=(nworld, njmax), inputs=leaf_in, outputs=[phi])
  phi.grad.fill_(1.0)  # seed adj_φ = +1 (φ already folds in λ; inactive rows returned early -> no-op reverse)
  adj_leaf = [None] * len(leaf_in)
  adj_leaf[0] = res_pos  # ∂φ/∂efc.pos = P̄
  adj_leaf[1] = res_vel  # ∂φ/∂efc.vel = V̄
  adj_leaf[2] = Z.grad  # Z̄ (computed; consumed only by the ∂J/∂q topology reverse, steps 4-8)
  # PARAM sys-id: expose per-class frictionloss / solref / solimp input-adjoints into res buffers.
  fl_rb = None
  if m.dof_frictionloss.requires_grad:
    fl_rb = wp.zeros_like(m.dof_frictionloss)
    adj_leaf[14] = fl_rb
  solref_rbs = []
  for _slot, _arr in ((12, m.dof_solref), (15, m.eq_solref), (17, m.jnt_solref)):
    if _arr.requires_grad:
      _rb = wp.zeros_like(_arr)
      adj_leaf[_slot] = _rb
      solref_rbs.append((_arr.grad, _rb))
  solimp_rbs = []
  for _slot, _arr in ((13, m.dof_solimp), (16, m.eq_solimp), (18, m.jnt_solimp)):
    if _arr.requires_grad:
      _rb = wp.zeros_like(_arr)
      adj_leaf[_slot] = _rb
      solimp_rbs.append((_arr.grad, _rb))
  wp.launch(_constraint_adjoint._constraint_row_phi, dim=(nworld, njmax), inputs=leaf_in, outputs=[phi],
            adj_inputs=adj_leaf, adj_outputs=[phi.grad], adjoint=True)
  scatter = _constraint_adjoint._constraint_scatter(m.is_sparse)
  wp.launch(
    scatter,
    dim=(nworld, njmax),
    inputs=[d_out.efc.J_rownnz, d_out.efc.J_rowadr, d_out.efc.J_colind, d_out.efc.J, d_out.efc.state,
            d_out.efc.type, d_out.nefc, nv, res_pos, res_vel],
    outputs=[res_qvel, res_dof],
  )
  if fl_rb is not None:  # IFT minus into the shared param grads
    wp.launch(_accum_neg, dim=fl_rb.shape, inputs=[fl_rb], outputs=[m.dof_frictionloss.grad])
  for _pg, _rb in solref_rbs:
    wp.launch(_accum_neg_vec2, dim=_rb.shape, inputs=[_rb], outputs=[_pg])
  for _pg, _rb in solimp_rbs:
    wp.launch(_accum_neg_vec5, dim=_rb.shape, inputs=[_rb], outputs=[_pg])


# ----------------------------------------------------------------------------
# 5. Smooth ∂qpos. Production uses the analytic reduced-RNE replay in smooth_adjoint. The central FD-of-rne
#    implementation remains behind _USE_ANALYTIC_RNE_QPOS=False as an explicit validation oracle only.
# ----------------------------------------------------------------------------
def _clone_for_fd(d: Data) -> Data:
  """Deep-clone a Data's wp.arrays with gradients off for analytic replay or the explicit FD oracle.

  Nested dataclasses are recursed and scalars/None are shared. Neither path may mutate d_out: it is tracked by
  the tape, so clobbering it corrupts the cross-step BPTT gradient through in-place overwrite.
  """

  def cl(o):
    if isinstance(o, wp.array):
      c = wp.clone(o)
      c.requires_grad = False
      return c
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
      return dataclasses.replace(o, **{f.name: cl(getattr(o, f.name)) for f in dataclasses.fields(o)})
    return o

  return cl(d)


def _recompute_smooth_forces(m: Model, d: Data):
  """Recompute qfrc_bias (= M·qacc + Coriolis, via rne flg_acc using the FROZEN d.qacc), qfrc_passive, and
  qfrc_actuator from d.qpos/qvel/ctrl -- the smooth-force subset of the forward (no collision/constraint/
  factor/crb; crb is unneeded since rne yields M·qacc directly)."""
  _smooth.kinematics(m, d)
  _smooth.com_pos(m, d)
  _smooth.tendon(m, d)
  _smooth.transmission(m, d)
  _smooth.com_vel(m, d)
  _passive.passive(m, d)
  _smooth.rne(m, d, flg_acc=True)
  _smooth.tendon_bias(m, d, d.qfrc_bias)
  _forward.fwd_actuation(m, d)


def _smooth_linearization(m: Model, d: Data, d_out: Data) -> Data:
  """A SHALLOW view of d_out at the step's linearization point (d.qpos, d.qvel, d_out.qacc) for the smooth
  VJPs -- the cheap replacement for the per-step deep clone + full smooth recompute (_clone_for_fd +
  _recompute_smooth_forces: ~100 array copies + ~30 launches -> 3 small allocs + one rne).

  Every smooth intermediate in the live d_out (kinematics/xipos/ximat/subtree_com/cinert/cdof/cvel/cdof_dot/
  tendon/actuator_moment/qfrc_passive/qfrc_actuator) was computed by the forward AT THE INPUT STATE and is
  untouched by integration (which only overwrites qpos/qvel/act) -- recomputing them is bit-identical waste.
  Exactly two things differ at the linearization: (1) qpos/qvel/ctrl/act must read the step's INPUTS -> alias
  d's arrays (no copies); (2) d_out's cacc/cfrc_int/qfrc_bias are the WRONG decomposition for the smooth VJP
  either way -- the forward's rne ran flg_acc=False (gravity-seeded cacc, no M*qacc term), and when the model
  has acceleration sensors, sensor_acc's rne_postconstraint then overwrites them with POST-CONSTRAINT values
  (constraint/contact forces folded in) -- so those three fields get FRESH scratch and one rne(flg_acc=True)
  (+ tendon_bias), reusing the shared kinematic intermediates. The view shares everything else with d_out;
  the VJPs read it as a frozen linearization and d_out is never mutated."""
  s = dataclasses.replace(
    d_out,
    qpos=d.qpos,
    qvel=d.qvel,
    ctrl=d.ctrl,
    act=d.act,
    cacc=wp.empty_like(d_out.cacc),
    cfrc_int=wp.empty_like(d_out.cfrc_int),
    qfrc_bias=wp.empty_like(d_out.qfrc_bias),
  )
  _smooth.rne(m, s, flg_acc=True)
  _smooth.tendon_bias(m, s, s.qfrc_bias)
  return s


@wp.kernel
def _perturb_col(base: wp.array2d[float], col: int, delta: float, out: wp.array2d[float]):
  """out = base with column `col` shifted by delta (one qpos coordinate, raw -- kinematics renormalizes
  quaternions, so the raw perturbation is self-consistent with the forward and the integrator adjoint)."""
  w, j = wp.tid()
  v = base[w, j]
  if j == col:
    v += delta
  out[w, j] = v


@wp.kernel
def _rsmooth_value(
  qfrc_bias: wp.array2d[float], qfrc_passive: wp.array2d[float], qfrc_actuator: wp.array2d[float],
  r_out: wp.array2d[float],
):
  """r_smooth = M·qacc - qfrc_smooth = rne(flg_acc).qfrc_bias - qfrc_passive - qfrc_actuator."""
  w, i = wp.tid()
  r_out[w, i] = qfrc_bias[w, i] - qfrc_passive[w, i] - qfrc_actuator[w, i]


@wp.kernel
def _fd_qpos_contract(
  r_plus: wp.array2d[float], r_minus: wp.array2d[float], lam: wp.array2d[float], nv: int, two_eps: float,
  col: int, adj_qpos: wp.array2d[float],
):
  """adj_qpos[:,col] += -(∂r_smooth/∂qpos_col)ᵀλ via central FD: -Σ_i (r_plus_i - r_minus_i)/(2eps)·λ_i."""
  w = wp.tid()
  s = float(0.0)
  for i in range(nv):
    s += (r_plus[w, i] - r_minus[w, i]) * lam[w, i]
  adj_qpos[w, col] += -s / two_eps


def _model_has_unsupported_noncontact_rows(m: Model) -> bool:
  """STRUCTURAL (sync-free, never-stale) predicate (MJPLAN_CSR §8): does the model carry any non-contact
  constraint row class whose VJP has NOT landed? SUPPORTED (NOT flagged) -- they just run via
  ``_residual_constraint_sparse`` (nv-general) / the dense ``_residual_constraint`` (nv<=_MAX_NV): dof-friction,
  slide/hinge limit (∂J/∂q=0 -> the reference-anchored leaf is analytically complete), JOINT equality
  (``eq_jnt_adr``; the coupling J captured by the frozen efc.J), and BALL limit (``jnt_limited_ball_adr``; the
  -axis angular J + the ``_dof_to_qpos`` quaternion lift). Their STATE grad (∂qvel via V̄ + ∂qpos via P̄ through
  the FROZEN efc.J) reuses the SAME frozen-J anchoring the dense ``_residual_constraint`` always used -- so
  landing them here is NOT a regression: FD-exact when ∂J/∂q=0 (identity-coupling joint equality; a settled
  ball whose rotation axis is ~constant -- both verified cos=1.0 vs float64 MuJoCo-C, incl. nv>_MAX_NV
  sparse/CSR) and a small residual bias otherwise (non-identity-polynomial joint-eq / large-axis-motion ball),
  whose ∂J/∂q topology reverse is the tracked MJPLAN_CSR steps 4-5 follow-up. UNSUPPORTED (flagged -> ``step_backward`` RAISES at
  EVERY nv until its topology reverse lands, MJPLAN_CSR steps 6-8): CONNECT / WELD equality (``eq_connect_adr``
  / ``eq_wld_adr`` -- 3-/6-row block leaves, UNION walk + Jdotv, NOT the scalar leaf), any tendon row
  (``m.ntendon`` -- tendon equality/limit/friction; conservative catch-all until step 7), and flex
  (``m.nflex`` -- covers flex contacts AND flex equalities). All are FIXED model structure (host ``.size`` /
  scalar reads) -> no ``.numpy()`` device read and no ``id(m)`` memoization (both went stale/flaky). Used ONLY
  to RAISE, never to alter a gradient."""
  unsupported_equality = m.eq_connect_adr.size > 0 or m.eq_wld_adr.size > 0
  return unsupported_equality or int(m.ntendon) > 0 or int(m.nflex) > 0


def _assert_dense_unroll(which: str):
  """The dense `for i in range(_MAX_NV)` reductions/scatter are replayed in reverse ONLY if Warp UNROLLS the
  range; a range left DYNAMIC (when `_MAX_NV > wp.config.max_unroll`) is NOT replayed in backward ->
  stale/wrong adjoint ([[warp-dynamic-loop-backward]]). The defaults sit exactly at the boundary
  (`wp.config.max_unroll == _MAX_NV == 16`, so the dense path is correct out of the box); this guards against
  a lowered `max_unroll` silently corrupting the dense-path gradient. (nv>_MAX_NV uses the sparse path.)"""
  assert _MAX_NV <= wp.config.max_unroll, (
    f"adjoint: the dense {which} backward needs _MAX_NV={_MAX_NV} <= wp.config.max_unroll="
    f"{wp.config.max_unroll}; otherwise range(_MAX_NV) stays a DYNAMIC loop and its reverse is stale "
    f"(silent wrong gradient). Raise wp.config.max_unroll to >= {_MAX_NV} (or use the sparse path)."
  )


# ----------------------------------------------------------------------------
# The analytic step backward (registered with forward.py).
# ----------------------------------------------------------------------------
def _assert_step_supported(m: Model):
  """Capability gate for step_backward / forward_backward_ift. Host-side Model metadata only (cone, nflex), so
  it stays sync-free / graph-capturable: NO `.numpy()` on device arrays (e.g. nacon is per-world batched --
  a host read both syncs AND only sees world 0)."""
  nv = m.nv
  # Both the smooth residual (_residual_smooth) and the CONTACT residual are now general over articulations
  # (the 4 MuJoCo joint types are handled by _advance_state; the contact J is built per-dof via
  # support.jac_dof -> body-vs-world AND body-vs-body, S2). The contact narrowphase ∂cpos/∂qpos is frozen
  # for now (S2: qvel-exact, contact ∂qpos = 0; S3 adds it via jac_dot_dof + the geometry-pair midpoint
  # derivative -- box-box/dominos stay partial until that lands).
  if m.nflex != 0:
    raise NotImplementedError("adjoint.step_backward does not support flex contacts")
  if m.opt.cone != _ELLIPTIC and m.opt.cone != _PYRAMIDAL:
    raise NotImplementedError("adjoint.step_backward supports only elliptic/pyramidal cones")
  # NON-contact constraint residual VJP. SUPPORTED classes run at ANY nv (the SPARSE/CSR nv-general
  # `_residual_constraint_sparse` for nv>_MAX_NV, the dense _MAX_NV kernel for nv<=_MAX_NV): dof-friction,
  # slide/hinge limit, JOINT equality, and BALL limit -- their STATE grad is FD-exact via the frozen-J
  # reference-anchored leaf (MJPLAN_CSR steps 1-5). UNSUPPORTED classes have no landed VJP (their block/topology
  # reverse is MJPLAN_CSR steps 6-8): CONNECT/WELD equality (need the 3-/6-row block leaf + UNION walk + Jdotv),
  # any tendon row (equality/limit/friction), flex -> RAISE structurally at EVERY nv, never a silent wrong
  # gradient. The predicate is FIXED model structure (sync-free, never stale).
  if _model_has_unsupported_noncontact_rows(m):
    raise NotImplementedError(
      "adjoint.step_backward: the model carries a non-contact constraint row class without a landed VJP "
      "(connect/weld equality, tendon, or flex). Supported: dof-friction, slide/hinge limit, JOINT equality, "
      "ball limit. The remaining block/topology reverses are MJPLAN_CSR steps 6-8."
    )
  # The dense CONTACT oracle is itself _MAX_NV-capped; nv>_MAX_NV needs the sparse contact path.
  if nv > _MAX_NV and not _USE_SPARSE_CONTACT:
    raise NotImplementedError(
      f"adjoint.step_backward: nv={nv} > _MAX_NV={_MAX_NV} requires the sparse contact path "
      f"(_USE_SPARSE_CONTACT); the dense `_residual_contact` oracle is capped at _MAX_NV."
    )
  # condim is mirrored generically (1/3/4/6 -- the valid MuJoCo set; 2/5 cannot be loaded), so no
  # nmaxcondim gate is needed: rotational rows (dimid>=3) are handled in both cone branches.


@wp.kernel(module="unique", enable_backward=True)
def _dampingpoly_Qv_leaf(timestep: wp.array[float], dof_damping: wp.array2d[float],
                         dof_dampingpoly: wp.array2d[wp.vec2], qvel: wp.array2d[float],
                         a_u: wp.array2d[float], out: wp.array2d[float]):
  """out[i] = dt · D_eff(v_i) · a_u[i], D_eff = ∂(poly damping force)/∂v = the diagonal of Q's damping block
  (Q = M + dt·D). Source-AD wrt qvel seeded with y_int gives ∂_v[ y_intᵀ dt·D(v) a_u ] -- the Stage-4
  implicit-integrator direct term for state-dependent (dampingpoly) damping. a_u and y_int are held fixed."""
  w, i = wp.tid()
  dt = timestep[w % timestep.shape[0]]
  damping = dof_damping[w % dof_damping.shape[0], i]
  dpoly = dof_dampingpoly[w % dof_dampingpoly.shape[0], i]
  out[w, i] = dt * _util_misc._poly_force_deriv(damping, dpoly, qvel[w, i], 1) * a_u[w, i]


def advance_backward(m: Model, d: Data, d_out: Data):
  """§1 -- integrator (advance) adjoint, the VJP of forward.{euler,implicit,...}. Maps adj(qpos',qvel')
  (= d_out.{qpos,qvel}.grad) -> adj_qacc + the integrator-direct adj(qpos,qvel). Backward-enabled
  _advance_state launched with adjoint=True (Warp source-to-source, incl. quat_integrate); remaps the
  implicitfast/eulerdamp integration solves. Returns (adj_qpos, adj_qvel, adj_qacc) -- adj_qacc seeds the IFT."""
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  # --- 1. integrator adjoint: adj(qpos',qvel') -> adj_qacc + integrator-direct adj(qpos,qvel) ---
  adj_qpos = wp.zeros((nworld, nq), dtype=float)
  adj_qvel = wp.zeros((nworld, nv), dtype=float)
  adj_qacc = wp.zeros((nworld, nv), dtype=float)

  # IMPLICITFAST does not advance with the solver root d_out.qacc directly.  forward.implicit rebuilds
  # Q = M - dt*d(qfrc_smooth)/d(qvel), solves a_int = Q^-1 (M*a), and passes a_int to _advance.  Rebuild
  # that exact value here both because the quaternion-position adjoint depends on the angular velocity and
  # because adj(a_int) must later be mapped through the transpose solve.  Missing this map makes the
  # backward behave like explicit Euler; on a low-inertia damped hinge its unstable multiplier can be ~8x
  # even though the byte-identical forward is strongly damped.
  implicit_deriv_flags = _types.DisableBit.ACTUATION | _types.DisableBit.SPRING | _types.DisableBit.DAMPER
  implicitfast_deriv = int(m.opt.integrator) == _IMPLICITFAST and bool(
    ~(m.opt.disableflags | ~implicit_deriv_flags)
  )
  # EULERDAMP is the Euler analog: forward.euler integrated a_u = (M+dt*D)^-1 M a_s before _advance_state,
  # so (like implicitfast) we reconstruct a_u and REPLAY THE INTEGRATOR AT IT below -- not at the raw solver
  # root a_s -- because FREE/BALL quaternion integration is nonlinear in qvel'=qvel+dt*a_u. Scalar joints are
  # affine in accel so the old replay-at-a_s + late-remap was correct for them, but WRONG for quaternions.
  # The transpose remap adj(a_s)=M(M+dt*D)^-1 adj(a_u) is KEPT, just moved after the replay (cached factor).
  eulerdamp = int(m.opt.integrator) == _EULER and (int(m.opt.disableflags) & (_EULERDAMP | _DAMPER)) == 0
  qacc_advance = d_out.qacc
  qLD_int = None
  qLDiagInv_int = None
  qLD_s = None
  qLDiagInv_s = None
  if implicitfast_deriv:
    # d_out owns the forward linearization intermediates, but its qvel has since been overwritten by
    # integration.  Alias the input velocity exactly as smooth_vel_backward does before rebuilding Q.
    saved_qvel = d_out.qvel
    d_out.qvel = d.qvel
    Q_int = wp.empty(d_out.M.shape, dtype=float)
    _derivative.deriv_smooth_vel(m, d_out, Q_int)
    d_out.qvel = saved_qvel
    qLD_int = wp.empty_like(d_out.qLD)
    qLDiagInv_int = wp.empty((nworld, nv), dtype=float)
    qacc_advance = wp.empty((nworld, nv), dtype=float)
    _smooth.factor_solve_i(m, d_out, Q_int, qLD_int, qLDiagInv_int, qacc_advance, d_out.efc.Ma)
  if eulerdamp:
    # Reconstruct a_u = (M+dt*D)^-1 (M a_s) BEFORE the replay and CACHE the factor for the post-replay
    # transpose remap. _compute_damping_deriv at the INPUT velocity d.qvel (d_out.qvel was overwritten by
    # integration); _euler_damp_qfrc adds dt*D to M's diagonal EXACTLY as forward.euler. M a_s via mul_m
    # (d_out.qacc is the solver root a_s) -> integrator-agnostic, no reliance on efc.Ma.
    damp_deriv = wp.empty((nworld, nv), dtype=float)
    wp.launch(_forward._compute_damping_deriv, dim=(nworld, nv),
              inputs=[m.dof_damping, m.dof_dampingpoly, d.qvel], outputs=[damp_deriv])
    MOD = wp.clone(d_out.M)  # M + dt*D, in M's CSR layout
    wp.launch(_forward._euler_damp_qfrc, dim=(nworld, nv),
              inputs=[m.opt.timestep, m.M_rownnz, m.M_rowadr, damp_deriv], outputs=[MOD])
    Ma_s = wp.empty((nworld, nv), dtype=float)
    _support.mul_m(m, d_out, Ma_s, d_out.qacc)
    qLD_s = wp.empty_like(d_out.qLD)
    qLDiagInv_s = wp.empty((nworld, nv), dtype=float)
    qacc_advance = wp.empty((nworld, nv), dtype=float)
    _smooth.factor_solve_i(m, d_out, MOD, qLD_s, qLDiagInv_s, qacc_advance, Ma_s)  # a_u = (M+dt*D)^-1 M a_s

  qpos_s = wp.empty_like(d.qpos)
  qvel_s = wp.empty_like(d.qvel)
  int_inputs = [m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, d.qvel, qacc_advance]
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

  if implicitfast_deriv:
    # a_int = Q^-1 M a, Q symmetric in IMPLICITFAST's factor_solve_i path, hence the ROOT remap
    #   adj(a) = M^T Q^-T adj(a_int) = M Q^-1 adj(a_int)   (reusing the factorization built above).
    # The DIRECT-state term y^T[(dM) a_s - (dQ) a_int] (from M(q)/Q(q,v) inside a_int) is handled separately
    # right below: ∂_q via the RNE-bias mass VJP (Stage 2), ∂_v via the dampingpoly leaf (Stage 4).
    y_int = wp.empty((nworld, nv), dtype=float)
    _smooth.solve_LD(m, d_out, qLD_int, qLDiagInv_int, y_int, adj_qacc)
    adj_a = wp.empty((nworld, nv), dtype=float)
    _support.mul_m(m, d_out, adj_a, y_int)
    adj_qacc = adj_a
    # STATE-DIRECT term (the piece the root remap omits): a_int = Q(q)^-1 M(q) a_s also depends on qpos
    # THROUGH M(q)/Q(q), so with a_s, a_u=a_int, y=y_int held fixed the integrator adjoint gains
    #   adj_qpos += ∂_q[ yᵀ M(q)(a_s − a_u) ]   (Q = M + dt·D -> ∂_q Q = ∂_q M for state-independent D).
    # Reuses the RNE-bias mass VJP; distinct cotangent (y_int) from the §2 IFT λ, so no double-count with
    # the smooth residual (∂M/∂q)a_s·λ. (Config-dependent-M only; ~0 for the constant-M hinge/paddle.)
    from mujoco_warp._src import smooth_adjoint as _smooth_adjoint

    w_dir = wp.empty((nworld, nv), dtype=float)
    wp.launch(_sub_write, dim=(nworld, nv), inputs=[d_out.qacc, qacc_advance], outputs=[w_dir])  # a_s − a_u
    q_dir = _smooth_adjoint.mass_matrix_qpos_vjp(m, d, y_int, w_dir)
    wp.launch(_accum_cols, dim=(nworld, nq), inputs=[q_dir], outputs=[adj_qpos])  # adj_qpos += ∂_q[yᵀ M w]
    # Stage 4 -- ∂_v direct term for STATE-DEPENDENT (dampingpoly) damping: Q = M + dt·D(v) depends on qvel,
    # so a_u does too. adj_qvel += −∂_v[ y_intᵀ dt·D(v) a_u ]. Source-AD dt·D(v)·a_u seeded with y_int (a_u,
    # y_int held fixed) gives +∂_v[...]; subtract it. Zero for LINEAR damping (D const -> a no-op). Gated on
    # DAMPER enabled so it matches the forward Q (which drops the damping block when DAMPER is disabled).
    if (int(m.opt.disableflags) & _DAMPER) == 0:
      dp_out = wp.empty((nworld, nv), dtype=float)
      dp_qv_adj = wp.zeros((nworld, nv), dtype=float)
      dp_ins = [m.opt.timestep, m.dof_damping, m.dof_dampingpoly, d.qvel, qacc_advance]
      wp.launch(_dampingpoly_Qv_leaf, dim=(nworld, nv), inputs=dp_ins, outputs=[dp_out])
      wp.launch(_dampingpoly_Qv_leaf, dim=(nworld, nv), inputs=dp_ins, outputs=[dp_out],
                adj_inputs=[None, None, None, dp_qv_adj, None], adj_outputs=[y_int], adjoint=True)
      wp.launch(_sub_write, dim=(nworld, nv), inputs=[adj_qvel, dp_qv_adj], outputs=[adj_qvel])  # −= ∂_v[...]
    # Stage 5 (TODO -- capability gate for GENERAL implicitfast): the mass VJP + dampingpoly leaf cover the
    # rigid-body ∂M/∂q and joint-damping ∂D/∂v ONLY. Q can also carry state-dependent velocity derivatives
    # from FLUID, TENDON damping, and GainType.AFFINE actuators (∂Q/∂qpos, ∂Q/∂qvel, ∂Q/∂{ctrl,act}) that
    # these terms OMIT -> the gradient would be SILENTLY INCOMPLETE for such models. Gate to correct-or-RAISE
    # (the file's policy): safe subset = JOINT transmission (hinge/slide) + GainType.FIXED + Bias{NONE,AFFINE},
    # no tendon/fluid. Actuator type arrays are device-side, so this needs a host capability bool (put_model)
    # or a one-time cached eager .numpy() check. Until it lands, this path is correct only for that subset
    # (all landed implicitfast tests are inside it).

  # --- 1b. EULERDAMP post-replay transpose remap. _advance_state was replayed at a_u (reconstructed above),
  # so its adjoint produced adj(a_u); map back to the solver root: adj(a_s) = M (M+dt*D)^-1 adj(a_u) (M and
  # M+dt*D symmetric), REUSING the cached factor (solve_LD, no re-factor). No double-count with the smooth-
  # qvel VJP: the damping force is in qfrc_smooth -> r -> H/λ; this is the SEPARATE integrated-accel remap.
  # (Same transpose the old late-remap did; the reorder only moved a_u's reconstruction ahead of the replay
  # so FREE/BALL quaternion integration linearizes at a_u -- see the eulerdamp comment at the top of §1.)
  if eulerdamp:
    y_damp = wp.empty((nworld, nv), dtype=float)
    _smooth.solve_LD(m, d_out, qLD_s, qLDiagInv_s, y_damp, adj_qacc)  # (M+dt*D)^-1 adj(a_u), cached factor
    adj_a = wp.empty((nworld, nv), dtype=float)
    _support.mul_m(m, d_out, adj_a, y_damp)  # adj(a_s) = M y
    adj_qacc = adj_a
    # KNOWN GAP (low-value, no failing test): unlike the implicitfast block above, eulerdamp does NOT yet add
    # the state-DIRECT terms  adj_qpos += ∂_q[y_dampᵀ M(a_s−a_u)]  (Stage 2) and  adj_qvel −= ∂_v[y_dampᵀ
    # dt·D(v) a_u]  (Stage 4, dampingpoly). Q = M + dt·D has the SAME structure as implicitfast, so closing it
    # is a mirror of that block with y=y_int -> y_damp. The effect is small for config-dependent M under Euler
    # (the eulerdamp-arm test passes without it, exactly as the implicitfast arm did pre-Stage-2); add it if a
    # config-dep-M / dampingpoly EULER case ever needs FD-exact gradients.
  return adj_qpos, adj_qvel, adj_qacc


def solve_backward(m: Model, d_out: Data, adj_qacc: wp.array):
  """§2 -- the IFT (VJP of solver.solve): solve H λ = adj_qacc reusing the forward solver's own assembly +
  Cholesky factor at the converged qacc (active set matches by construction). Returns the SolverContext;
  the caller reads λ = ctx.Mgrad and must keep ctx alive while λ is in use (ctx owns the buffer)."""
  nworld = d_out.qpos.shape[0]
  nv = m.nv
  nv_pad = m.nv_pad
  # --- 2. IFT: solve H λ = adj_qacc, reusing the solver's assembly + Cholesky at converged qacc ---
  ctx = _solver._create_solver_context(m, d_out)
  _solver.init_context(m, d_out, ctx, grad=True)  # assembles + factors ctx.h; active set = forward's
  wp.launch(_load_rhs, dim=(nworld, nv_pad), inputs=[adj_qacc, nv], outputs=[ctx.grad])
  ctx.done.zero_()
  _solver._cholesky_factorize_solve(m, d_out, ctx)  # ctx.Mgrad[:, :nv] = λ
  return ctx


def contact_residual_backward(m: Model, d: Data, d_out: Data, lam: wp.array,
                              res_qpos: wp.array, res_qvel: wp.array):
  """§3 + §3c -- contact residual-VJP. §3 produces the five contact geometry seeds (∂r/∂{cdof,subtree_com,
  efc_pos,contact_pos,contact_frame}·λ) + res_qvel via the SPARSE contract-first path (default, nv-general)
  or the DENSE _MAX_NV oracle (A/B); §3c lifts the narrowphase ∂qpos into res_qpos via collision_adjoint.
  res_qpos/res_qvel are the cross-term seeds shared with the constraint/smooth terms; the five geometry
  seeds are local (§3 produces, §3c consumes). nacon=0 (smooth scenes) -> every contact kernel is a no-op."""
  nworld = d.qpos.shape[0]
  nv = m.nv
  # --- 3. residual-VJP: adj_theta = integrator-direct - (dr/dtheta)ᵀλ ---
  res_cdof = wp.zeros((nworld, nv), dtype=wp.spatial_vector)
  res_subtree_com = wp.zeros((nworld, m.nbody), dtype=wp.vec3)
  res_efc_pos = wp.zeros_like(d_out.efc.pos)
  res_contact_pos = wp.zeros_like(d_out.contact.pos)
  res_contact_frame = wp.zeros_like(d_out.contact.frame)
  efc_pos_ref = wp.clone(d_out.efc.pos)  # frozen D-recovery reference (separate from the differentiated efc_pos)
  efc_pos_ref.requires_grad = False

  if _USE_SPARSE_CONTACT:
    # SPARSE contract-first (nv-general, no _MAX_NV): gather V/A/Z over the symmetric-difference ancestor
    # walk -> loop-free source-AD cone leaf φ=-Z·F(V,A,ξ) (seed adj_φ=+1) -> manual sparse scatter.
    ncon_max = d_out.contact.pos.shape[0]
    Vc = wp.zeros(ncon_max, dtype=wp.spatial_vector, requires_grad=True)
    Ac = wp.zeros(ncon_max, dtype=wp.spatial_vector, requires_grad=True)
    Zc = wp.zeros(ncon_max, dtype=wp.spatial_vector, requires_grad=True)
    phi = wp.zeros(ncon_max, dtype=float, requires_grad=True)
    adjV = wp.zeros(ncon_max, dtype=wp.spatial_vector)
    adjA = wp.zeros(ncon_max, dtype=wp.spatial_vector)
    adjZ = wp.zeros(ncon_max, dtype=wp.spatial_vector)
    walk_in = [m.body_rootid, m.body_weldid, m.body_dofnum, m.body_dofadr, m.dof_parentid, m.geom_bodyid]
    state_in = [d.qvel, d_out.qacc, lam, d_out.subtree_com, d_out.cdof, d_out.contact.pos, d_out.contact.geom,
                d_out.contact.efc_address, d_out.contact.worldid, d_out.efc.state, d_out.nacon]
    wp.launch(_constraint_adjoint._contact_gather, dim=ncon_max, inputs=walk_in + state_in, outputs=[Vc, Ac, Zc])
    want_fric = m.geom_friction.requires_grad  # geom_friction sys-id (the contact-PARAM gradient)
    for _arr in (d_out.contact.frame, d_out.efc.pos):
      _arr.requires_grad = True  # so the leaf adjoint accumulates res_contact_frame / res_efc_pos
    res_contact_friction = None
    if want_fric:
      d_out.contact.friction.requires_grad = True  # so the leaf exposes ∂φ/∂contact.friction (mu -> cone)
      res_contact_friction = wp.zeros(ncon_max, dtype=vec5)
    phi_in = [Vc, Ac, Zc, d_out.contact.frame, d_out.efc.pos, d_out.efc.margin, d_out.efc.D, d_out.efc.state,
              d_out.contact.friction, d_out.contact.solref, d_out.contact.solreffriction, d_out.contact.solimp,
              d_out.contact.dim, d_out.contact.efc_address, d_out.contact.worldid, d_out.nacon,
              m.opt.timestep, m.opt.impratio_invsqrt, m.opt.disableflags, efc_pos_ref]
    contact_phi_kernel = _constraint_adjoint._contact_phi(int(m.opt.cone))  # cone-specialized (cached)
    wp.launch(contact_phi_kernel, dim=ncon_max, inputs=phi_in, outputs=[phi])
    wp.launch(_constraint_adjoint._fill_ones, dim=ncon_max, inputs=[phi.grad])  # seed adj_φ = +1 (φ already folds in λ)
    adj_phi_in = [None] * len(phi_in)
    adj_phi_in[0] = adjV  # V̄
    adj_phi_in[1] = adjA  # Ā
    adj_phi_in[2] = adjZ  # Z̄
    adj_phi_in[3] = res_contact_frame  # ∂r/∂contact_frame
    adj_phi_in[4] = res_efc_pos  # ∂r/∂efc_pos (penetration)
    if want_fric:
      adj_phi_in[8] = res_contact_friction  # ∂r/∂contact.friction -> chained to geom_friction below
    wp.launch(contact_phi_kernel, dim=ncon_max, inputs=phi_in, outputs=[phi],
              adj_inputs=adj_phi_in, adj_outputs=[phi.grad], adjoint=True)
    wp.launch(_constraint_adjoint._contact_scatter, dim=ncon_max, inputs=walk_in + state_in + [adjV, adjA, adjZ],
              outputs=[res_qvel, res_cdof, res_subtree_com, res_contact_pos])
    if want_fric:  # CONTACT-PARAM sys-id: chain ∂φ/∂contact.friction -> m.geom_friction.grad (IFT minus)
      wp.launch(_constraint_adjoint._contact_friction_geom_vjp, dim=ncon_max,
                inputs=[m.geom_priority, m.geom_friction, d_out.contact.geom, d_out.contact.worldid,
                        d_out.contact.efc_address, d_out.efc.state, d_out.nacon, res_contact_friction],
                outputs=[m.geom_friction.grad])
  else:
    # DENSE oracle (legacy _MAX_NV monolithic autodiff; nv<=16 only). Same five seeds + res_qvel via Warp
    # reverse of the whole residual in ONE pass. Retained as the EXACT A/B reference for the sparse path.
    _assert_dense_unroll("contact-residual oracle")
    residual_contact_kernel = _constraint_adjoint._residual_contact(int(m.opt.cone))  # cone-specialized kernel (cached)
    r = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
    for _arr in (d_out.cdof, d_out.subtree_com, d_out.efc.pos, d_out.contact.pos, d_out.contact.frame):
      _arr.requires_grad = True  # so the manual adjoint launch accumulates their input-adjoints
    rin = [
      d.qpos,  # 0 (unused in-kernel; res_qpos accumulates the S3 narrowphase ∂qpos below)
      d.qvel,  # 1
      d_out.qacc,  # 2
      d_out.cdof,  # 3
      d_out.subtree_com,  # 4
      d_out.efc.D,  # 5
      d_out.efc.state,  # 6
      d_out.efc.pos,  # 7
      d_out.efc.margin,  # 8
      d_out.contact.pos,  # 9
      d_out.contact.frame,  # 10
      d_out.contact.friction,
      d_out.contact.solref,
      d_out.contact.solreffriction,
      d_out.contact.solimp,
      d_out.contact.dim,
      d_out.contact.geom,
      m.geom_bodyid,
      m.body_isdofancestor,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      d_out.contact.efc_address,
      d_out.contact.worldid,
      d_out.nacon,
      m.opt.timestep,
      m.opt.impratio_invsqrt,
      m.opt.disableflags,
      nv,
      efc_pos_ref,  # frozen efc_pos reference (no adjoint) for the D-recovery imp0/pos0
    ]
    wp.launch(residual_contact_kernel, dim=nworld, inputs=rin, outputs=[r])
    wp.launch(_copy_cols, dim=(nworld, nv), inputs=[lam], outputs=[r.grad])  # seed adj_r = λ
    adj_rin = [None] * len(rin)  # expose the kinematic-intermediate input-adjoints (∂r/∂· · λ)
    adj_rin[1] = res_qvel
    adj_rin[3] = res_cdof  # ∂r/∂cdof (articulated; S3b)
    adj_rin[4] = res_subtree_com  # ∂r/∂subtree_com (articulated; S3b)
    adj_rin[7] = res_efc_pos  # ∂r/∂efc_pos (penetration)
    adj_rin[9] = res_contact_pos  # ∂r/∂contact_pos (moment arm)
    adj_rin[10] = res_contact_frame  # ∂r/∂contact_frame (normal/tangent rows)
    wp.launch(
      residual_contact_kernel,
      dim=nworld,
      inputs=rin,
      outputs=[r],
      adj_inputs=adj_rin,
      adj_outputs=[r.grad],
      adjoint=True,
    )

  # --- 3c. S3 contact ∂qpos (general): collision_adjoint replays each frozen contact's narrowphase geometry
  # (auto-diffing the forward pure funcs), chains geom-pose -> qpos via support.jac_dof, and adds the
  # free-body subtree_com term. Accumulates into res_qpos (which _sub_write subtracts). nacon=0 / unsupported
  # geom pairs -> no-op. (See collision_adjoint.py; FD-gated by collision_adjoint_test.py.)
  _collision_adjoint.contact_qpos_vjp(
    m, d_out, d.qpos, res_contact_pos, res_contact_frame, res_efc_pos, res_subtree_com, res_cdof, res_qpos
  )


def smooth_vel_backward(m: Model, d: Data, d_out: Data, lam: wp.array, adj_qvel: wp.array):
  """§4 -- smooth residual ∂qvel (the analytic G = ∂qfrc_smooth/∂qvel contracted with λ -> adj_qvel) + ∂ctrl
  (the actuation leaf -> d.ctrl.grad). Same IFT λ. The ∂qvel / ∂ctrl channels of the fwd_velocity /
  fwd_actuation smooth forces (the ∂qpos channel is smooth_qpos_backward)."""
  nworld = d.qpos.shape[0]
  # --- 4. smooth residual: ∂qvel (analytic) + ∂ctrl (actuation leaf), same λ; sum into adj_qvel/ctrl ---
  # ∂qvel: G = ∂qfrc_smooth/∂qvel = (M_D - qLU)/dt, with qLU assembled from deriv_smooth_vel (the
  # passive+actuator part, M - dt·∂(qfrc_passive+qfrc_actuator)/∂qvel, M-structure) -> map_m2d ->
  # deriv_rne_vel(flg_subtract=FALSE, the Coriolis ∂qfrc_bias/∂qvel). flg_subtract is FALSE -- NOT the
  # forward `implicit` integrator's True: qfrc_smooth = qfrc_passive + qfrc_actuator - qfrc_bias, so the
  # bias term must ADD so that (M_D - qLU)/dt = pa - ∂qfrc_bias/∂qvel = ∂qfrc_smooth/∂qvel. (The True
  # convention gives pa + bias -- right sign for Coriolis ONLY by accident; FD-verified exact for both
  # Coriolis and damping in _scratch/probe_qderiv_accuracy.py.) Evaluate at the step's linearization
  # (d.qpos, d.qvel): d_out carries the linearization intermediates (M, cvel, cdof, cinert) but d_out.qvel
  # was overwritten to the integrated next velocity, so alias d_out.qvel -> d.qvel for the (read-only)
  # deriv_* calls, then restore.
  saved_qvel = d_out.qvel
  d_out.qvel = d.qvel
  qH_M = wp.empty(d_out.M.shape, dtype=float)
  _derivative.deriv_smooth_vel(m, d_out, qH_M)  # M - dt·∂(qfrc_passive+qfrc_actuator)/∂qvel (M-structure)
  qLU = wp.empty((nworld, m.nD), dtype=float)
  M_D = wp.empty((nworld, m.nD), dtype=float)
  wp.launch(_forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, qH_M], outputs=[qLU])
  _derivative.deriv_rne_vel(m, d_out, qLU, flg_subtract=False)  # += dt·∂qfrc_bias/∂qvel -> (M_D-qLU)/dt = ∂qfrc_smooth/∂qvel
  wp.launch(_forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, d_out.M], outputs=[M_D])
  d_out.qvel = saved_qvel
  # Assemble the ∂qvel VJP by contracting the sparse (D-structure) velocity Jacobian G against λ:
  # adj_qvel += Gᵀλ, scattered over the nD nonzeros (mujoco_warp has no transpose-apply for this layout).
  wp.launch(
    _smooth_qvel_vjp,
    dim=(nworld, m.nD),
    inputs=[m.qD_fullm_i, m.qD_fullm_j, M_D, qLU, lam, m.opt.timestep],
    outputs=[adj_qvel],  # adj_qvel += Gᵀλ
  )
  if m.nu > 0 and d.ctrl.requires_grad:
    wp.launch(
      _smooth_ctrl_vjp,
      dim=(nworld, m.nu),
      inputs=[d_out.moment_rownnz, d_out.moment_rowadr, d_out.moment_colind, d_out.actuator_moment,
              m.actuator_gainprm, lam],
      outputs=[d.ctrl.grad],
    )


def smooth_param_backward(m: Model, d: Data, d_out: Data, lam: wp.array):
  """§4b -- smooth-PARAM sys-id with the same IFT λ. Two leaves, each requires_grad-gated (no-op otherwise):
  (1) armature / viscous damping -- the LOCAL smooth residual (no RNE tree, AD-clean); (2) body_mass /
  body_inertia -- the INERTIAL params, which enter the smooth residual ONLY through cinert, so adj_cinert
  (the rne-bias reverse) routed through the SOURCE-AD cinert leaf gives the inertia derivative with no hand-
  written VJP (smooth_adjoint.inertia_param_vjp). The single smooth-param sys-id entry point."""
  nworld = d.qpos.shape[0]
  nv = m.nv
  # --- 4b. smooth-PARAM sys-id (armature, viscous damping): AD the LOCAL smooth residual
  # r_local = dof_armature⊙qacc + dof_damping⊙qvel with the SAME IFT λ. Marking the Model arrays
  # requires_grad and seeding adj_r = -λ drops adj_θ = -(∂r/∂θ)ᵀλ straight into m.<param>.grad -- one
  # launch, no per-param hand VJP (§5.11 near-free rider). A model param is one leaf shared across the
  # trajectory, so its grad ACCUMULATES (the adjoint launch += into the persistent m.<param>.grad each
  # step/world; caller zeros it once per trajectory). No RNE tree here -> AD-clean (mass/inertia are the
  # AD-of-rne follow-up; gains/springs the next terms of this residual).
  if m.dof_armature.requires_grad or m.dof_damping.requires_grad:
    r_s = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
    _rs_inputs = [d.qvel, d_out.qacc, m.dof_armature, m.dof_damping]
    wp.launch(_residual_smooth_local, dim=(nworld, nv), inputs=_rs_inputs, outputs=[r_s])
    wp.launch(_neg_cols, dim=(nworld, nv), inputs=[lam], outputs=[r_s.grad])  # seed adj_r = -λ
    wp.launch(
      _residual_smooth_local,
      dim=(nworld, nv),
      inputs=_rs_inputs,
      outputs=[r_s],
      adj_inputs=[None, None,  # qvel/qacc frozen
                  m.dof_armature.grad if m.dof_armature.requires_grad else None,
                  m.dof_damping.grad if m.dof_damping.requires_grad else None],
      adj_outputs=[r_s.grad],
      adjoint=True,
    )

  # --- 4b (inertial). body_mass / body_inertia / body_ipos / body_iquat sys-id. All four enter the smooth
  # residual ONLY through cinert (the c-frame spatial inertia: the M·qacc CRB contraction AND the RNE Coriolis/
  # gravity bias both flow through it -- gravity via the cacc-world seed). So adj_cinert from the rne-proper
  # reverse (rne_backward; the irreducible dynamic-depth tree transpose, a manual enable_backward=False VJP ->
  # no unroll/DOF-scaling dependency) routed through the SOURCE-AD cinert leaf (_cinert_recompute, loop-free ->
  # exact) yields -(∂r/∂θ)ᵀλ with NO hand-written inertia VJP: mass/inertia are DIRECT cinert inputs, while
  # ipos/iquat continue the reverse from adj_{xipos,ximat} through the inertial-frame leaf (_inertial_frames_
  # recompute). Unlike armature/damping (the LOCAL residual above), this needs the smooth KINEMATIC
  # intermediates at the linearization (cinert/cdof/cvel/cfrc_int from rne flg_acc=True), and the live d_out
  # carries the flg_acc=False bias + post-integration qpos/qvel -- so recompute the reduced smooth pipeline on
  # a NON-GRAD scratch (only when requested; never mutate d_out). Same IFT-minus sign as the local params.
  if (m.body_mass.requires_grad or m.body_inertia.requires_grad
      or m.body_ipos.requires_grad or m.body_iquat.requires_grad):
    from mujoco_warp._src import smooth_adjoint as _smooth_adjoint  # local import breaks the import cycle

    s = _smooth_linearization(m, d, d_out)  # shallow view + rne(flg_acc=True); replaces the per-step deep clone
    _smooth_adjoint.inertia_param_vjp(m, s, lam)  # -(∂r/∂θ)ᵀλ -> m.body_{mass,inertia,ipos,iquat}.grad


def noncontact_constraint_backward(m: Model, d: Data, d_out: Data, lam: wp.array,
                                   res_qpos: wp.array, res_qvel: wp.array, ctx_Jaref: wp.array = None):
  """§4c -- NON-CONTACT constraint residual (slide/hinge limit + dof Coulomb friction) + param sys-id.
  Tangent ∂qpos -> res_qpos via _dof_to_qpos; ∂qvel -> res_qvel; params accumulated in-place. No-op when no
  active non-contact row. TWO residual paths, same reference-anchored math + IFT-minus at write-back:
  nv>_MAX_NV takes the SPARSE/CSR nv-general `_residual_constraint_sparse` (needs the ctx.Jaref anchor);
  nv<=_MAX_NV takes the dense _MAX_NV `_residual_constraint` (the proven path / A-B oracle). Unsupported row
  classes (equality/ball/tendon/flex) never reach here -- `_assert_step_supported` raised in step_backward."""
  nworld = d.qpos.shape[0]
  nv = m.nv
  res_dof = wp.zeros((nworld, nv), dtype=float)  # per-dof TANGENT ∂qpos (lifted to res_qpos below; shared)
  # Route to the SPARSE/CSR path when nv>_MAX_NV (dense kernel's range(_MAX_NV) unroll cap) OR when efc.J is
  # stored CSR (m.is_sparse -- the dense kernel indexes efc.J[w,row,i] densely and CANNOT read CSR). Otherwise
  # the dense _MAX_NV kernel (nv<=_MAX_NV, dense efc.J) -- the proven path / A-B oracle.
  if nv > _MAX_NV or m.is_sparse:
    # --- SPARSE/CSR nv-general path (MJPLAN_CSR): gather Z=Jλ -> reference-anchored loop-free leaf φ=-Z·f
    # -> Jᵀ scatter, accumulating +∂φ/∂qvel into res_qvel (ALL rows) and the TANGENT +∂φ/∂qpos into res_dof
    # (position-bearing rows). PARAM sys-id (frictionloss / per-class solref+solimp) accumulates -(∂r/∂θ)ᵀλ
    # into m.<param>.grad internally. Reuses the FROZEN ctx.Jaref (= J·qacc - aref) as the value anchor
    # (only Z=Jλ is re-reduced). slide/hinge + dof-friction are exact here (∂J/∂q=0).
    assert ctx_Jaref is not None, "sparse non-contact constraint VJP (nv>_MAX_NV) requires ctx.Jaref"
    _residual_constraint_sparse(m, d_out, ctx_Jaref, lam, res_qvel, res_dof)
  else:
    # --- dense _MAX_NV static-unroll path (guard its unroll invariant). ONE forward + ONE +λ adjoint launch
    # yields every input-adjoint as +(∂r/∂·)ᵀλ into its own res buffer; the IFT MINUS is applied at
    # WRITE-BACK. PARAMS (dof_frictionloss + per-class solref) fall out of the same launch (auto-diffed
    # through _contact_kbimp; no per-param kernel).
    _assert_dense_unroll("_residual_constraint")
    r_c = wp.zeros((nworld, nv), dtype=float, requires_grad=True)
    dpos = wp.zeros((nworld, nv), dtype=float, requires_grad=True)  # TANGENT seed (zeros); ∂r/∂δθ -> res_dof
    _rc_inputs = [dpos, d.qvel, d_out.qacc, d_out.efc.J, d_out.efc.D, d_out.efc.pos, d_out.efc.margin,
                  d_out.efc.state, d_out.efc.type, d_out.efc.id, m.dof_frictionloss, m.dof_solref, m.dof_solimp,
                  m.eq_solref, m.eq_solimp, m.jnt_solref, m.jnt_solimp, d_out.nefc, m.opt.timestep,
                  m.opt.disableflags, nv]
    wp.launch(_residual_constraint, dim=nworld, inputs=_rc_inputs, outputs=[r_c])
    wp.launch(_copy_cols, dim=(nworld, nv), inputs=[lam], outputs=[r_c.grad])  # seed adj_r = +λ
    _adj_rc = [None] * len(_rc_inputs)
    _adj_rc[0] = res_dof  # tangent ∂qpos (lifted to res_qpos by _dof_to_qpos below)
    _adj_rc[1] = res_qvel  # ∂qvel state-grad -> res_qvel (accumulates with contact; subtracted by _sub_write)
    _res_fl = None  # PARAM sys-id: expose input-adjoints (dof_frictionloss float; per-class *_solref vec2)
    if m.dof_frictionloss.requires_grad:
      _res_fl = wp.zeros_like(m.dof_frictionloss)
      _adj_rc[10] = _res_fl
    _res_solref = []  # (param_grad, res_buffer) for the vec2 solref params
    for _slot, _arr in ((11, m.dof_solref), (13, m.eq_solref), (15, m.jnt_solref)):
      if _arr.requires_grad:
        _rb = wp.zeros_like(_arr)
        _adj_rc[_slot] = _rb
        _res_solref.append((_arr.grad, _rb))
    wp.launch(_residual_constraint, dim=nworld, inputs=_rc_inputs, outputs=[r_c],
              adj_inputs=_adj_rc, adj_outputs=[r_c.grad], adjoint=True)
    if _res_fl is not None:  # -(∂r/∂fl)ᵀλ into the shared param grad
      wp.launch(_accum_neg, dim=_res_fl.shape, inputs=[_res_fl], outputs=[m.dof_frictionloss.grad])
    for _pg, _rb in _res_solref:
      wp.launch(_accum_neg_vec2, dim=_rb.shape, inputs=[_rb], outputs=[_pg])
  # lift the per-dof TANGENT ∂qpos to qpos (quaternion-aware: free/BALL via 2·q⊗[0,g]); accumulates into
  # res_qpos alongside the contact ∂qpos, then both flow through _sub_write. Shared by both paths.
  wp.launch(_collision_adjoint._dof_to_qpos, dim=(nworld, m.njnt),
            inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, res_dof], outputs=[res_qpos])


def smooth_qpos_backward(m: Model, d: Data, d_out: Data, lam: wp.array,
                         res_qpos: wp.array, adj_qpos: wp.array):
  """§5 -- smooth-force ∂qpos. ANALYTIC reduced replay (smooth_adjoint, accumulated into res_qpos) when
  _USE_ANALYTIC_RNE_QPOS (production default), else explicit FD-of-rne test oracle. See MJPLAN.md §5.4.1."""
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  # --- 5. smooth ∂qpos. ANALYTIC reduced replay (smooth_adjoint) when _USE_ANALYTIC_RNE_QPOS, else FD-of-rne.
  if _USE_ANALYTIC_RNE_QPOS:
    # Analytic smooth-residual ∂qpos via the reduced backward-only replay (MJPLAN_ADRNE §0/§10): rigid-body
    # RNE bias (smooth_force_backward) + joint springs (spring_qpos_vjp) + AFFINE joint actuators
    # (actuator_qpos_vjp). Each returns (∂(λᵀr_smooth_term)/∂qpos); accumulate into res_qpos (subtracted by
    # _sub_write = the IFT minus -(∂r/∂qpos)ᵀλ). NO finite-difference fallback: assert_smooth_supported
    # RAISES on any enabled smooth feature without an analytic leaf (tendon/gravcomp/fluid/muscle/...);
    # FD is the explicit test-only path below (flag False), never an automatic substitute (§0). Local import
    # breaks the smooth_adjoint<->adjoint cycle (smooth_adjoint reuses this module's reverse kernels).
    from mujoco_warp._src import smooth_adjoint as _smooth_adjoint

    _smooth_adjoint.assert_smooth_supported(m)
    # CRITICAL: evaluate the smooth VJPs at the SAME linearization FD-of-rne uses -- (d.qpos, d.qvel,
    # d_out.qacc) with rne(flg_acc=True). The live d_out has qfrc_bias = Coriolis only (forward solves for
    # qacc -> flg_acc=False, so d_out.cacc/cfrc_int miss the M·qacc term) AND its qpos/qvel are the
    # INTEGRATED next state, not the linearization point. Skipping this drops ∂(M·qacc)/∂qpos (verified:
    # BPTT rel 0.14 vs FD-of-rne -> 0 after this recompute). _smooth_linearization provides exactly that as
    # a shallow d_out view (input-state aliases + fresh rne flg_acc=True), replacing the former deep clone.
    s = _smooth_linearization(m, d, d_out)
    res_q = _smooth_adjoint.smooth_force_backward(m, s, lam, flg_acc=True)  # rigid-body RNE bias
    wp.launch(_accum_cols, dim=(nworld, nq), inputs=[res_q], outputs=[res_qpos])
    res_sp = _smooth_adjoint.spring_qpos_vjp(m, s, lam)  # joint springs (all joint types)
    wp.launch(_accum_cols, dim=(nworld, nq), inputs=[res_sp], outputs=[res_qpos])
    res_ac = _smooth_adjoint.actuator_qpos_vjp(m, s, lam)  # affine joint-transmission actuators
    wp.launch(_accum_cols, dim=(nworld, nq), inputs=[res_ac], outputs=[res_qpos])
    res_gc = _smooth_adjoint.gravcomp_qpos_vjp(m, s, lam)  # gravity compensation (passive bucket)
    wp.launch(_accum_cols, dim=(nworld, nq), inputs=[res_gc], outputs=[res_qpos])
  else:
    # FD-of-rne TEST ORACLE (the full-r_smooth reference). Re-run the smooth force sub-pipeline at qpos ± eps
    # (qvel/qacc/ctrl frozen at the linearization), central-difference r_smooth = qfrc_bias - qfrc_passive
    # - qfrc_actuator, contract with λ: adj_qpos += -(∂r_smooth/∂qpos)ᵀλ. Covers bias + passive + actuator
    # + tendon. NOT capture-safe (host eps loop over nq). Runs on a SEPARATE non-grad scratch clone --
    # never mutate d_out (grad-tracked; clobbering corrupts the cross-step BPTT gradient).
    eps = 1.0e-4
    fd = _clone_for_fd(d_out)
    wp.copy(fd.qvel, d.qvel)
    if m.nu > 0:
      wp.copy(fd.ctrl, d.ctrl)
    r_plus = wp.empty((nworld, nv), dtype=float)
    r_minus = wp.empty((nworld, nv), dtype=float)
    for col in range(nq):
      wp.launch(_perturb_col, dim=(nworld, nq), inputs=[d.qpos, col, eps], outputs=[fd.qpos])
      _recompute_smooth_forces(m, fd)
      wp.launch(_rsmooth_value, dim=(nworld, nv),
                inputs=[fd.qfrc_bias, fd.qfrc_passive, fd.qfrc_actuator], outputs=[r_plus])
      wp.launch(_perturb_col, dim=(nworld, nq), inputs=[d.qpos, col, -eps], outputs=[fd.qpos])
      _recompute_smooth_forces(m, fd)
      wp.launch(_rsmooth_value, dim=(nworld, nv),
                inputs=[fd.qfrc_bias, fd.qfrc_passive, fd.qfrc_actuator], outputs=[r_minus])
      wp.launch(_fd_qpos_contract, dim=nworld,
                inputs=[r_plus, r_minus, lam, nv, 2.0 * eps, col], outputs=[adj_qpos])


def _write_input_adjoints(m: Model, d: Data, adj_qpos: wp.array, adj_qvel: wp.array,
                          res_qpos: wp.array, res_qvel: wp.array):
  """Write the step's input adjoints d.{qpos,qvel}.grad = integrator-direct adj − residual-VJP (the IFT
  minus). Each d == datas[t] is the d_in of exactly one step."""
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  # --- write input adjoints (each d == datas[t] is the d_in of exactly one step) ---
  wp.launch(_sub_write, dim=(nworld, nq), inputs=[adj_qpos, res_qpos], outputs=[d.qpos.grad])
  wp.launch(_sub_write, dim=(nworld, nv), inputs=[adj_qvel, res_qvel], outputs=[d.qvel.grad])
  # SMOOTH residual term r_smooth = M·qacc - qfrc_smooth (-> ctrl/actuator grads, sys-id of
  # mass/inertia/damping, ∂M/∂qpos, ∂qfrc_bias/∂qpos): a planned SEPARATE kernel `_residual_smooth`. The
  # IFT VJP is linear in r = r_smooth + r_contact, so AD it with the SAME λ and SUM its adj_{qpos,qvel,...}
  # into the writes above (λ already uses the full H = M + JᵀG_sJ; no double-count -- M in H is ∂r/∂qacc,
  # ∂(M·qacc)/∂θ in r_smooth is w.r.t. θ). Omitted now: qpos/qvel-independent for gravity-only free body.


def forward_backward_ift(m: Model, d: Data, d_out: Data, adj_qpos: wp.array, adj_qvel: wp.array,
                              adj_qacc: wp.array):
  """Adjoint (VJP) of forward() -- the dynamics -- via the IMPLICIT FUNCTION THEOREM.

  forward() is a feed-forward pipeline (position -> velocity -> actuation -> acceleration -> solve), but its
  adjoint is NOT that pipeline reversed. The constraint solver makes qacc an implicit function of the inputs,
  so we differentiate the equation qacc satisfies rather than the iteration that finds it -- hence the
  `_ift` suffix (Implicit Function Theorem), and hence no per-stage fwd_<stage>_backward (the stage map below
  have gone).

  In:  adj_qacc -- the cotangent on qacc from the integrator adjoint; plus the integrator-direct adj_qpos /
       adj_qvel accumulators (all from advance_backward).
  Out: accumulates d.{qpos, qvel, ctrl}.grad and the model-parameter grads (system identification).

  THE IFT IN TWO STEPS. The forward solver returns qacc as the root of the dynamics residual
      r(qacc, θ) = M·qacc − qfrc_smooth(θ) − Jᵀ f(J·qacc − aref) = 0      (θ = qpos, qvel, ctrl, params)
  so the reverse pass is:
      (1)  solve   H λ = adj_qacc        -- solve_backward; H = M + Jᵀ G_s J, reused from the forward solver
      (2)  set     adj_θ = −(∂r/∂θ)ᵀ λ   -- the residual-VJP, spread over the term helpers below
  The single cotangent λ feeds every term, so the helpers are not independently usable (all read λ), and
  solve_backward must run first.

  HOW THE RESIDUAL-VJP IS ORGANIZED (a force-term × input-channel grid). r is a SUM of force terms, so ∂r/∂θ
  factors by TERM; each term reaches the inputs through several forward stages, i.e. the input CHANNELS
  (qpos / qvel / ctrl / params). One autodiff launch per term produces all of that term's channel-seeds at
  once, so the helpers are named by term, not by forward stage:

      smooth      (M·qacc − qfrc_smooth)
          qpos : smooth_qpos_backward     (§5  -- analytic reduced-RNE replay; FD oracle only when requested)
          qvel : smooth_vel_backward      (§4  -- analytic G = ∂qfrc_smooth/∂qvel: Coriolis + damping)
          ctrl : smooth_vel_backward      (§4  -- ∂qfrc_actuator/∂ctrl)
          param: smooth_param_backward    (§4b -- armature / damping system-id)
      contact     (−Jᵀ f_contact)
          qpos : contact_residual_backward       (§3c -- narrowphase ∂cpos/∂qpos)
          qvel : contact_residual_backward       (§3  -- the J·qvel reference, res_qvel)
      non-contact (−Jᵀ f_noncontact: equality / joint-limit / dof-friction)
          qpos : noncontact_constraint_backward  (§4c -- _dof_to_qpos lift)
          qvel : noncontact_constraint_backward  (§4c -- res_qvel)
          param: noncontact_constraint_backward  (§4c -- solref / frictionloss system-id)

  WHERE forward()'S STAGES WENT (they are the input channels above, not helpers of their own):
      solver.solve     -> solve_backward.  The only clean one-to-one (the IFT solve for λ).
      fwd_acceleration -> nothing.  It only SUMS forces (qfrc_smooth = qfrc_passive − qfrc_bias +
                          qfrc_actuator + qfrc_applied) then unconstrained-solves for qacc_smooth. The
                          adjoint of a sum is just the ± sign each smooth term uses when seeding λ, and
                          qacc_smooth is never read by the backward (the IFT uses the converged qacc and H).
      fwd_position     -> the qpos channel, split across contact_residual_backward, noncontact_constraint_
                          backward, and smooth_qpos_backward (fwd_position emits M, J, aref, and contact
                          geometry -- each consumed by a different term).
      fwd_velocity     -> the qvel channel: smooth_vel_backward, plus the res_qvel seeds from the contact
                          and non-contact terms.
      fwd_actuation    -> smooth_vel_backward's ∂qfrc_actuator/∂ctrl leaf (its qpos-dependence, the
                          transmission moment, is handled in smooth_qpos_backward).
  Not differentiated: sleep and collision DETECTION (discrete -- frozen like the active set), and sensors /
  energy (off the qacc residual; a loss on observations uses the separate fwd_kinematics hook).

  NOTES. Internal to step_backward (not a registered hook). The term helpers only accumulate into commutative
  buffers and never read one another's output, so their order is free -- only solve_backward (first) and
  _write_input_adjoints (last) are load-bearing. The launch order here matches the former monolithic
  step_backward exactly, so gradients are byte-identical. See MJPLAN.md §5 / §5.4.1."""
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  ctx = solve_backward(m, d_out, adj_qacc)  # solver.solve_backward: H λ = adj_qacc; ctx owns λ -- keep alive
  lam = ctx.Mgrad
  # residual-VJP cross-term seeds (the qpos / qvel GRID COLUMNS), shared across the contact / constraint /
  # smooth terms and summed into d.{qpos,qvel}.grad at write-back. (The contact term's five contact-geometry
  # seeds are local to its helper.)
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  res_qvel = wp.zeros((nworld, nv), dtype=float)
  # ORDER below = the former monolith's exact launch order (§3 → §4 → §4b → §4c → §5 → write-back), kept for
  # byte-identical FP. The five term helpers only ACCUMULATE into res_qpos/res_qvel/adj_qvel/adj_qpos (all
  # commutative) and none reads another's output, so they COMMUTE -- any permutation gives the same gradient.
  # Only the two endpoints are load-bearing: solve_backward FIRST (produces the shared λ) and
  # _write_input_adjoints LAST (reads the accumulated sums). (Hence the smooth term being "split" -- §4/§4b
  # before the constraint term, §5 after -- is cosmetic, not a dependency.)
  contact_residual_backward(m, d, d_out, lam, res_qpos, res_qvel)  # contact term -> §3 ∂qvel + §3c ∂qpos
  smooth_vel_backward(m, d, d_out, lam, adj_qvel)  # smooth term -> §4 qvel + ctrl columns
  smooth_param_backward(m, d, d_out, lam)  # smooth term -> §4b model-param column (armature/damping sys-id)
  noncontact_constraint_backward(m, d, d_out, lam, res_qpos, res_qvel, ctx.Jaref)  # constraint term -> §4c qpos + qvel
  smooth_qpos_backward(m, d, d_out, lam, res_qpos, adj_qpos)  # smooth term -> §5 qpos column
  _write_input_adjoints(m, d, adj_qpos, adj_qvel, res_qpos, res_qvel)  # grad = integrator-direct − residual


def step_backward(m: Model, d: Data, d_out: Data):
  """Reads d_out.{qpos,qvel}.grad (upstream), writes d.{qpos,qvel}.grad. Uses d as the input state and
  d_out for the step's intermediates (converged qacc, efc.*, contact.*, cdof).

  step = forward + integrator advance, so the backward composes advance_backward (the integrator VJP ->
  adj_qacc) then forward_backward_ift (the IFT + residual-VJP). See MJPLAN.md §5 / §5.4.1 for the
  decomposition rationale (the IFT restructure -- why the dynamics adjoint does NOT mirror forward()'s
  stages)."""
  _assert_step_supported(m)
  adj_qpos, adj_qvel, adj_qacc = advance_backward(m, d, d_out)
  forward_backward_ift(m, d, d_out, adj_qpos, adj_qvel, adj_qacc)


_forward.register_backward_hook(step_backward)


# ============================================================================================
# Analytic position backward for forward.fwd_kinematics (differentiable observations).
#
# fwd_kinematics maps qpos -> {site_xpos, xpos, xquat, ...} via the (smooth) kinematics tree. We do
# NOT AD that tree (dynamic-loop bug); instead the VJP is the analytic point Jacobian
# J = ∂x_site/∂q (support.jac_dof: jacp = cdof_lin + cdof_ang x (point - subtree_com[root])), built
# from cdof/subtree_com recomputed fresh at qpos on a non-grad scratch clone. adj_qpos = Σ_site Jᵀ·adj.
# This is the SHAC-ready differentiable-observation primitive; Stage 1 covers site_xpos. The per-site VJP is
# built in DOF/tangent space (nv) and lifted to qpos (nq) by _collision_adjoint._dof_to_qpos -- 1:1 for
# slide/hinge/free-translation, the 2·q⊗[0,g] quaternion lift for free/ball rotation -- so it is correct for
# free/ball-base bodies (nq!=nv), not just fixed-base hinge/slide chains.
# ============================================================================================


@wp.kernel(enable_backward=False)
def _site_jac_vjp(
  # Model:
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  site_bodyid: wp.array[int],
  # Data (scratch, fresh at qpos):
  subtree_com: wp.array2d[wp.vec3],
  cdof: wp.array2d[wp.spatial_vector],
  site_xpos: wp.array2d[wp.vec3],
  # adjoint of site_xpos (= d.site_xpos.grad):
  adj_site: wp.array2d[wp.vec3],
  nsite: int,
  # Out (accumulate), DOF/tangent-space -- lifted to qpos by _dof_to_qpos (quaternion-aware for free/ball):
  adj_dof: wp.array2d[float],
):
  """adj_dof[w, i] += Σ_site (∂site_xpos/∂(dof i))ᵀ · adj_site[w, site], where i is a DOF index and jacp is
  the dof-i site Jacobian column (support.jac_dof). This stays in dof/tangent space (nv); the dof->qpos map
  -- 1:1 for slide/hinge/free-translation, the 2·q⊗[0,g] quaternion lift for free/ball rotation -- is done
  by _dof_to_qpos in fwd_kinematics_backward (nq!=nv). Forward kernel (enable_backward off): the site loop
  is dynamic but runs in FORWARD mode (manual VJP), so it is safe."""
  w, i = wp.tid()
  acc = float(0.0)
  for s in range(nsite):
    jacp, jacr = _support.jac_dof(
      body_parentid, body_rootid, dof_bodyid, body_isdofancestor,
      subtree_com, cdof, site_xpos[w, s], site_bodyid[s], i, w,
    )
    acc += wp.dot(jacp, adj_site[w, s])
  adj_dof[w, i] += acc


def fwd_kinematics_backward(m: Model, d: Data):
  """Analytic position VJP for forward.fwd_kinematics: adj(site_xpos) -> adj(qpos) via the analytic
  site Jacobian. Recompute kinematics+com_pos on a non-grad scratch clone at d.qpos (fresh
  site_xpos/cdof/subtree_com -- no staleness, no kinematics-tree AD), then scatter Jᵀ·adj into
  d.qpos.grad. Registered as forward's position backward hook (consulted by fwd_kinematics under a
  tape; step pauses the tape so its internal fwd_kinematics never double-records)."""
  if m.nsite == 0 or d.qpos.grad is None or d.site_xpos.grad is None:
    return
  fd = _clone_for_fd(d)  # value-only scratch; refresh kinematics state at qpos
  _smooth.kinematics(m, fd)
  _smooth.com_pos(m, fd)
  adj_dof = wp.zeros((d.nworld, m.nv), dtype=float)  # per-dof TANGENT VJP, lifted to qpos below
  wp.launch(
    _site_jac_vjp,
    dim=(d.nworld, m.nv),
    inputs=[
      m.body_parentid, m.body_rootid, m.dof_bodyid, m.body_isdofancestor, m.site_bodyid,
      fd.subtree_com, fd.cdof, fd.site_xpos, d.site_xpos.grad, m.nsite,
    ],
    outputs=[adj_dof],
  )
  # lift the per-dof tangent gradient to qpos (1:1 for slide/hinge/free-translation; 2·q⊗[0,g] quaternion
  # lift for free/ball rotation) -- the nq!=nv map, same kernel the contact ∂qpos path uses.
  wp.launch(
    _collision_adjoint._dof_to_qpos,
    dim=(d.nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, adj_dof],
    outputs=[d.qpos.grad],
  )


_forward.register_position_backward_hook(fwd_kinematics_backward)
