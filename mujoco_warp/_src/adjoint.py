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
     ``r(qpos,qvel) = -Jᵀ·f(J·qacc - aref)``, launched ``wp.launch(_resid_contact, adjoint=True)``
     (no nested tape; same call style as piece 1) seeded with ``adj_r = λ``, giving
     ``adj_theta = -(dr/dtheta)ᵀ λ`` for ALL contact terms in ONE pass: ``∂J/∂qpos`` (the moment arm
     AND the elliptic cone curvature, both AD'd), the ``aref`` penetration/dissipation, and ``∂D``.
     Replaces the per-curvature hand scatter (the retired ``_gs_cone`` is archived in MJPLAN §5.9).
     The differentiable ``J`` is built for the free-joint body as ``J = [frame, rvec x frame]`` with
     ``rvec = contact_pos(qpos) - com``, the contact point tracking qpos by the sphere-vs-flat
     midpoint narrowphase derivative ``∂cpos/∂qpos = I - 1/2 n nᵀ``. ``M``/``qfrc_smooth`` are
     qpos-independent for the gravity-only sphere, so only the contact term carries ∂/∂(qpos,qvel).

Scope/status (bounce target): a single free-joint body (dofs 0-5), elliptic, sphere-vs-flat contacts
(plane / box-face). The contact residual-VJP is FD-verified per-entry vs ``mjd_transitionFD`` on
box_on_plane / bounce steps (QUADRATIC and CONE) on the dominant entries. KNOWN small gap: the
rotational (ωy/ωz) rows use world-aligned free-joint axes, so they carry a ~few-percent error once
the body spins (the ``∂cdof/∂qpos`` orientation term); negligible for the isotropic-sphere scalar
loss. TODO for generality: build ``J`` from ``d.cdof`` (any joint, exact rotational rows), non-contact
constraints (equality/limits), and a general narrowphase ``∂cpos/∂qpos`` (non-flat colliders); §5.6/§5.9.
"""

import warp as wp

from mujoco_warp._src import forward as _forward
from mujoco_warp._src import math as _math
from mujoco_warp._src import solver as _solver
from mujoco_warp._src import types as _types
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5

# NOTE: adjoint.py kernels are differentiable -- do NOT set enable_backward=False here, so Warp
# codegens the adjoint of _advance_state / _resid_contact.

_FREE = int(_types.JointType.FREE.value)
_BALL = int(_types.JointType.BALL.value)
_SATISFIED = int(_types.ConstraintState.SATISFIED.value)
_CONE = int(_types.ConstraintState.CONE.value)
_MINIMP = float(_types.MJ_MINIMP)
_MAXIMP = float(_types.MJ_MAXIMP)
_MINVAL = float(_types.MJ_MINVAL)


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
# 3. Contact residual r(qpos,qvel) = -Jᵀ·f(J·qacc - aref). Backward-enabled: we launch its adjoint
#    with adj_r = λ to get -(dr/dtheta)ᵀλ in one pass (replaces the hand scatter; module docstring +
#    MJPLAN §5.9). v1 scope: a single free-joint body (dofs 0-5), elliptic, sphere-vs-flat contacts.
# ----------------------------------------------------------------------------
@wp.kernel
def _resid_contact(
  qpos_in: wp.array2d[float],  # [grad]
  qvel_in: wp.array2d[float],  # [grad]
  qacc_in: wp.array2d[float],  # frozen
  com0_in: wp.array2d[float],  # frozen linearization qpos (read [:3] = com)
  efc_D_in: wp.array2d[float],  # frozen
  efc_state_in: wp.array2d[int],  # frozen active set
  efc_pos_in: wp.array2d[float],  # frozen normal penetration
  contact_pos_in: wp.array(dtype=wp.vec3),
  contact_frame_in: wp.array(dtype=wp.mat33),  # per-contact frame; rows = normal, tangent1, tangent2
  contact_friction_in: wp.array(dtype=vec5),
  contact_solref_in: wp.array(dtype=wp.vec2),
  contact_solimp_in: wp.array(dtype=vec5),
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  opt_timestep: wp.array[float],
  opt_impratio_invsqrt: wp.array[float],
  r_out: wp.array2d[float],  # out: contact residual (pre-zeroed; free-joint dofs 0-5)
):
  w = wp.tid()
  com = wp.vec3(qpos_in[w, 0], qpos_in[w, 1], qpos_in[w, 2])
  com0 = wp.vec3(com0_in[w, 0], com0_in[w, 1], com0_in[w, 2])
  vlin = wp.vec3(qvel_in[w, 0], qvel_in[w, 1], qvel_in[w, 2])
  vang = wp.vec3(qvel_in[w, 3], qvel_in[w, 4], qvel_in[w, 5])
  alin = wp.vec3(qacc_in[w, 0], qacc_in[w, 1], qacc_in[w, 2])
  aang = wp.vec3(qacc_in[w, 3], qacc_in[w, 4], qacc_in[w, 5])
  dt = opt_timestep[w % opt_timestep.shape[0]]
  imp_isq = opt_impratio_invsqrt[w % opt_impratio_invsqrt.shape[0]]

  for cid in range(nacon_in[0]):
    if contact_worldid_in[cid] != w:
      continue
    e0 = contact_efc_address_in[cid, 0]
    if e0 < 0:
      continue
    st = efc_state_in[w, e0]
    if st == _SATISFIED:
      continue
    has_fric = contact_dim_in[cid] > 1
    e1 = e0
    e2 = e0
    if has_fric:
      e1 = contact_efc_address_in[cid, 1]
      e2 = contact_efc_address_in[cid, 2]
      if e1 < 0 or e2 < 0:
        has_fric = False

    fm = contact_frame_in[cid]
    f0 = wp.vec3(fm[0, 0], fm[0, 1], fm[0, 2])
    f1 = wp.vec3(fm[1, 0], fm[1, 1], fm[1, 2])
    f2 = wp.vec3(fm[2, 0], fm[2, 1], fm[2, 2])
    # contact point tracks qpos: sphere-vs-flat midpoint => dcpos/dcom = I - 1/2 n nᵀ
    dcom = com - com0
    cpos_eff = contact_pos_in[cid] + (dcom - 0.5 * wp.dot(f0, dcom) * f0)
    rvec = cpos_eff - com  # moment arm contact_pos - com (com = subtree_com for the free sphere)
    Jr0 = wp.cross(rvec, f0)
    Jr1 = wp.cross(rvec, f1)
    Jr2 = wp.cross(rvec, f2)
    Jqa0 = wp.dot(f0, alin) + wp.dot(Jr0, aang)
    Jqv0 = wp.dot(f0, vlin) + wp.dot(Jr0, vang)
    Jqa1 = wp.dot(f1, alin) + wp.dot(Jr1, aang)
    Jqv1 = wp.dot(f1, vlin) + wp.dot(Jr1, vang)
    Jqa2 = wp.dot(f2, alin) + wp.dot(Jr2, aang)
    Jqv2 = wp.dot(f2, vlin) + wp.dot(Jr2, vang)

    # k, b, k·imp  (mirror constraint._efc_row; saturated impedance imp = dmax)
    solref = contact_solref_in[cid]
    solimp = contact_solimp_in[cid]
    timeconst = wp.max(solref[0], 2.0 * dt)
    dampratio = solref[1]
    dmax = wp.clamp(solimp[1], _MINIMP, _MAXIMP)
    dmax_sq = dmax * dmax
    k = wp.where(solref[0] <= 0.0, -solref[0] / dmax_sq, 1.0 / (dmax_sq * timeconst * timeconst * dampratio * dampratio))
    b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, 2.0 / (dmax * timeconst))
    kimp_n = k * dmax

    pen = efc_pos_in[w, e0] + wp.dot(f0, com - com0)
    Jaref0 = Jqa0 - (-kimp_n * pen - b * Jqv0)
    Jaref1 = Jqa1 - (-b * Jqv1)
    Jaref2 = Jqa2 - (-b * Jqv2)

    D0 = efc_D_in[w, e0]
    force0 = float(0.0)
    force1 = float(0.0)
    force2 = float(0.0)
    if has_fric and st == _CONE:  # elliptic cone (sliding): mu-scaled, normal<->tangent coupling
      fri = contact_friction_in[cid]
      mu = fri[0] * imp_isq
      Nn = Jaref0 * mu
      u1 = Jaref1 * fri[0]
      u2 = Jaref2 * fri[1]
      Tn = wp.sqrt(wp.max(u1 * u1 + u2 * u2, _MINVAL))  # sqrt(max ..): wp.sqrt(0) has an inf adjoint
      dm = D0 / (mu * mu * (1.0 + mu * mu))
      fn = -dm * (Nn - mu * Tn) * mu
      force0 = fn
      force1 = -(fn / Tn) * (mu * u1)  # ufrictionj = mu * Jaref_j * fri_j
      force2 = -(fn / Tn) * (mu * u2)
    else:  # QUADRATIC (normal + sticking friction)
      force0 = -D0 * Jaref0
      if has_fric:
        force1 = -efc_D_in[w, e1] * Jaref1
        force2 = -efc_D_in[w, e2] * Jaref2

    # r += -Jᵀ force  (atomic_add: AD-safe accumulation across contacts; r pre-zeroed)
    rl = -(f0 * force0 + f1 * force1 + f2 * force2)
    rr = -(Jr0 * force0 + Jr1 * force1 + Jr2 * force2)
    wp.atomic_add(r_out, w, 0, rl[0])
    wp.atomic_add(r_out, w, 1, rl[1])
    wp.atomic_add(r_out, w, 2, rl[2])
    wp.atomic_add(r_out, w, 3, rr[0])
    wp.atomic_add(r_out, w, 4, rr[1])
    wp.atomic_add(r_out, w, 5, rr[2])


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
  state and d_out for the step's intermediates (converged qacc, efc.*, contact.*)."""
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  nv_pad = m.nv_pad

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
    com0,
    d_out.efc.D,
    d_out.efc.state,
    d_out.efc.pos,
    d_out.contact.pos,
    d_out.contact.frame,
    d_out.contact.friction,
    d_out.contact.solref,
    d_out.contact.solimp,
    d_out.contact.dim,
    d_out.contact.efc_address,
    d_out.contact.worldid,
    d_out.nacon,
    m.opt.timestep,
    m.opt.impratio_invsqrt,
  ]
  wp.launch(_resid_contact, dim=nworld, inputs=rin, outputs=[r])
  wp.launch(_copy_cols, dim=(nworld, nv), inputs=[lam], outputs=[r.grad])  # seed adj_r = λ
  wp.launch(
    _resid_contact,
    dim=nworld,
    inputs=rin,
    outputs=[r],
    adj_inputs=[res_qpos, res_qvel, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    adj_outputs=[r.grad],
    adjoint=True,
  )

  # --- write input adjoints (each d == datas[t] is the d_in of exactly one step) ---
  wp.launch(_sub_write, dim=(nworld, nq), inputs=[adj_qpos, res_qpos], outputs=[d.qpos.grad])
  wp.launch(_sub_write, dim=(nworld, nv), inputs=[adj_qvel, res_qvel], outputs=[d.qvel.grad])
  # ctrl / qfrc_smooth-velocity / ∂cdof,∂M/∂qpos paths: TODO (not needed for the gravity-only bounce)


_forward.register_backward_hook(step_backward)
