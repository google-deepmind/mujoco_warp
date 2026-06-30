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
"""Constraint residual VJP kernels for the analytic IFT backward (``adjoint.py``) -- the VJP of
``constraint.py``'s efc rows. BOTH the NON-CONTACT rows (equality / joint-limit / dof-friction) and the
CONTACT rows (the elliptic/pyramidal cone force law) live here; ``adjoint.py`` keeps only the orchestration
(``noncontact_constraint_backward`` / ``contact_residual_backward``).

NON-CONTACT (the gather/leaf/scatter kernels just below) -- the SPARSE/CSR + nv-general replacement of the
legacy dense ``adjoint._residual_constraint`` (MJPLAN_CSR.md). ``adjoint._residual_constraint_sparse``
orchestrates these three kernels (gather -> loop-free leaf -> scatter) as one piece of ``step_backward``.

CONTACT (the cone force law in the bottom section, moved from adjoint.py) -- the dense ``_residual_contact``
``_MAX_NV`` oracle and the sparse ``_contact_gather`` / ``_contact_phi`` / ``_contact_scatter`` path.

  1. ``_constraint_gather`` (enable_backward=False) -- per active non-contact row, reduce ``Z_e = Σ_i
     J_ei·λ_i`` over the row's J support (mirrors ``solver._solve_init_jaref_kernel``'s CSR/dense iterator)
     and gather the TRUE topology invweight (§5.2). ``A_e`` and ``V_e`` are NOT re-reduced -- the orchestrator
     reuses ``ctx.Jaref`` (= J·qacc - aref) and ``efc.vel`` (= J·qvel) as the FROZEN value anchors.
  2. ``_constraint_row_phi`` (enable_backward=True, LOOP-FREE) -- the reference-anchored row leaf
     ``φ_e = -Z·f``: ``f`` value-anchored to the stored ``efc.force``; ``D`` differentiated via the TRUE
     topology invweight (so ``∂D/∂solimp`` is exact, NOT canceled by an ``efc.D`` recovery). Reuses
     ``constraint._contact_kbimp`` auto-diffed THROUGH (Warp generates the @wp.func adjoint in this
     backward-enabled module, even though constraint.py is enable_backward=False).
  3. ``_constraint_scatter`` (enable_backward=False) -- re-walk the J support: ``res_qvel_i += J_ei·V̄_e``
     for ALL rows; ``res_dof_i += J_ei·P̄_e`` for POSITION-BEARING rows ONLY (friction ``P≡0`` is not a
     function of qpos -> no qpos route, else a spurious qpos-stiffness; MJPLAN_CSR must-fix #1).

``is_sparse``-specialized where they touch ``efc.J`` (dense ``J[w,row,i]`` vs CSR ``J[w,0,rowadr+k]``);
nv-general (no ``_MAX_NV``). ``enable_backward=False`` on gather/scatter -> their dynamic CSR loop is a SAFE
manual VJP (Warp does not replay dynamic loops in reverse, but these run forward-only); only the loop-free
leaf is AD'd. The dense/CSR branch is the MODEL's ``m.is_sparse``; the legacy/new routing is the structural
``_model_has_unsupported_noncontact_rows`` predicate in ``step_backward`` -- there is NO global on/off flag.
"""

from typing import Tuple

import warp as wp

from mujoco_warp._src import constraint as _constraint
from mujoco_warp._src import solver as _solver
from mujoco_warp._src import support as _support
from mujoco_warp._src import types as _types
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import cache_kernel

_SATISFIED = int(_types.ConstraintState.SATISFIED.value)
_LINEARNEG = int(_types.ConstraintState.LINEARNEG.value)  # saturated friction, force = +frictionloss
_LINEARPOS = int(_types.ConstraintState.LINEARPOS.value)  # saturated friction, force = -frictionloss
_FRICTION_DOF = int(_types.ConstraintType.FRICTION_DOF.value)
_EQUALITY = int(_types.ConstraintType.EQUALITY.value)
_LIMIT_JOINT = int(_types.ConstraintType.LIMIT_JOINT.value)
_MINVAL = float(_types.MJ_MINVAL)
# Contact cone-residual constants (used by the CONTACT section below, moved from adjoint.py):
_CONE = int(_types.ConstraintState.CONE.value)
_ELLIPTIC = int(_types.ConeType.ELLIPTIC.value)
_MAXCONDIM = 6  # max valid MuJoCo condim; elliptic friction rows = dimid 1..condim-1
_MAX_PYRAMID_EDGES = 10  # 2*(_MAXCONDIM - 1) pyramidal edges at condim 6
_MAX_NV = 16  # static unroll bound for the dense `_residual_contact` oracle (the sparse path is nv-general)


@cache_kernel
def _constraint_gather(is_sparse: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    jnt_dofadr: wp.array[int],
    dof_invweight0: wp.array2d[float],
    # Data in (frozen):
    lam_in: wp.array2d[float],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_state_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    nefc_in: wp.array[int],
    nv: int,
    # Out (per row):
    Z_out: wp.array2d[float],
    invw_out: wp.array2d[float],
  ):
    """GATHER (manual): per active non-contact row, reduce ``Z_e = Σ_i J_ei·λ_i`` over the row's J support
    (mirrors solver._solve_init_jaref_kernel's CSR/dense iterator) and gather the TRUE topology invweight
    (MJPLAN_CSR §5.2): dof-friction ``dof_invweight0[dofid]``, slide/hinge/ball limit ``dof_invweight0
    [jnt_dofadr[jntid]]``. ``invw=0`` signals the frozen-D fallback (equality / other -- production-gated and
    not solimp-sys-id-tested, so the efc.D-frozen branch is acceptable there)."""
    w, row = wp.tid()
    Z_out[w, row] = 0.0
    invw_out[w, row] = 0.0
    if row >= nefc_in[w]:
      return
    ty = efc_type_in[w, row]
    if ty != _EQUALITY and ty != _LIMIT_JOINT and ty != _FRICTION_DOF:
      return  # contact rows -> _contact_*; tendon limit/friction -> TODO (structural gate raises)
    if efc_state_in[w, row] == _SATISFIED:
      return
    Z = float(0.0)
    if wp.static(is_sparse):
      rownnz = efc_J_rownnz_in[w, row]
      rowadr = efc_J_rowadr_in[w, row]
      for k in range(rownnz):
        sid = rowadr + k
        Z += efc_J_in[w, 0, sid] * lam_in[w, efc_J_colind_in[w, 0, sid]]
    else:
      for i in range(nv):
        Z += efc_J_in[w, row, i] * lam_in[w, i]
    Z_out[w, row] = Z
    cid = efc_id_in[w, row]
    if ty == _FRICTION_DOF:
      invw_out[w, row] = dof_invweight0[w % dof_invweight0.shape[0], cid]
    elif ty == _LIMIT_JOINT:
      invw_out[w, row] = dof_invweight0[w % dof_invweight0.shape[0], jnt_dofadr[cid]]
    # EQUALITY / other -> invw=0 (leaf uses the frozen-D fallback; their solimp grad is gated, not exact)

  return kernel


@wp.kernel(module="unique", enable_backward=True)
def _constraint_row_phi(
  efc_pos_in: wp.array2d[float],  # [grad] penetration source (P = efc.pos - margin) -> P̄ (res_efc_pos)
  efc_vel_in: wp.array2d[float],  # [grad] J·qvel -> V̄ (res_efc_vel)
  Z_in: wp.array2d[float],  # [grad] = J·λ (gather) -> Z̄ (= -f; the ∂J/∂q topology seed, unused for G1)
  invw_in: wp.array2d[float],  # frozen TRUE topology invweight (0 -> frozen-D fallback)
  efc_margin_in: wp.array2d[float],  # frozen
  efc_aref_in: wp.array2d[float],  # frozen (= -g_base; cancels g0 so jaref base == ctx.Jaref)
  efc_D_in: wp.array2d[float],  # frozen efc.D_ref (D base + frozen-D fallback)
  efc_force_in: wp.array2d[float],  # frozen efc.force_ref (force VALUE anchor)
  ctx_Jaref_in: wp.array2d[float],  # frozen J·qacc - aref (jaref VALUE anchor)
  efc_state_in: wp.array2d[int],  # frozen active set (NEVER re-derived from a perturbed jaref)
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  dof_solref_in: wp.array2d[wp.vec2],
  dof_solimp_in: wp.array2d[vec5],
  dof_frictionloss_in: wp.array2d[float],
  eq_solref_in: wp.array2d[wp.vec2],
  eq_solimp_in: wp.array2d[vec5],
  jnt_solref_in: wp.array2d[wp.vec2],
  jnt_solimp_in: wp.array2d[vec5],
  nefc_in: wp.array[int],
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  phi_out: wp.array2d[float],
):
  """LOOP-FREE reference-anchored row leaf: ``φ_e = -Z·f``. The force is VALUE-anchored to the stored
  ``efc.force``: ``f = efc.force_ref + (F_state - stopgrad(F_state0))``, so the base value byte-equals
  ``efc.force_ref`` (because ``efc.aref = -g_base`` cancels ``g`` in jaref, and the true-invweight ``D_live``
  base equals ``efc.D_ref``) while only the intended derivatives flow. QUADRATIC: ``F_state = -D·jaref`` with
  ``jaref = ctx.Jaref + efc.aref + g``, ``g = k·imp(P)·P + b·V`` (differentiable P/V via constraint._contact_kbimp),
  ``D = 1/max(invw·(1-imp)/imp, MJ_MINVAL)`` (true-invweight -> ∂imp/∂P, ∂D/∂P, ∂D/∂solimp all flow; clamp
  branch frozen by wp.max). LINEARNEG/POS (saturated friction): ``f = ±frictionloss``. Loop-free -> AD-safe.
  ``efc.state`` is read FROZEN and branched on (never recomputed from a perturbed jaref, which would be
  non-smooth)."""
  w, row = wp.tid()
  phi_out[w, row] = 0.0  # assign output before any early return (Warp footgun)
  if row >= nefc_in[w]:
    return
  ty = efc_type_in[w, row]
  if ty != _EQUALITY and ty != _LIMIT_JOINT and ty != _FRICTION_DOF:
    return
  st = efc_state_in[w, row]
  if st == _SATISFIED:
    return
  cid = efc_id_in[w, row]
  Z = Z_in[w, row]
  if ty == _FRICTION_DOF and st == _LINEARNEG:  # saturated friction: force = +frictionloss
    phi_out[w, row] = -Z * dof_frictionloss_in[w % dof_frictionloss_in.shape[0], cid]
    return
  if ty == _FRICTION_DOF and st == _LINEARPOS:  # saturated friction: force = -frictionloss
    phi_out[w, row] = -Z * (-dof_frictionloss_in[w % dof_frictionloss_in.shape[0], cid])
    return
  # QUADRATIC (equality / active limit / stuck friction): f = -D·jaref, value-anchored to efc.force.
  dt = opt_timestep[w % opt_timestep.shape[0]]
  P = efc_pos_in[w, row] - efc_margin_in[w, row]
  V = efc_vel_in[w, row]
  if ty == _FRICTION_DOF:
    sid = w % dof_solref_in.shape[0]
    kbi = _constraint._contact_kbimp(opt_disableflags, dt, dof_solref_in[sid, cid], dof_solimp_in[sid, cid], P)
  elif ty == _EQUALITY:
    sid = w % eq_solref_in.shape[0]
    kbi = _constraint._contact_kbimp(opt_disableflags, dt, eq_solref_in[sid, cid], eq_solimp_in[sid, cid], P)
  else:  # _LIMIT_JOINT (slide/hinge scalar J; ball 3-dof -axis J)
    sid = w % jnt_solref_in.shape[0]
    kbi = _constraint._contact_kbimp(opt_disableflags, dt, jnt_solref_in[sid, cid], jnt_solimp_in[sid, cid], P)
  k = kbi[0]
  b = kbi[1]
  imp = kbi[2]
  g = k * imp * P + b * V
  Jaref0 = ctx_Jaref_in[w, row]
  jaref = Jaref0 + efc_aref_in[w, row] + g  # value == Jaref0 at base (efc.aref = -g_base, same _contact_kbimp)
  D_ref = efc_D_in[w, row]
  invw = invw_in[w, row]
  if invw > 0.0:
    D = 1.0 / wp.max(invw * (1.0 - imp) / imp, _MINVAL)  # true-invweight D_live (base == D_ref by construction)
  else:
    D = D_ref  # frozen-D fallback (equality / other; matches the legacy dense kernel)
  F_state = -D * jaref
  F_state0 = -D_ref * Jaref0  # frozen base (== F_state at base)
  f = efc_force_in[w, row] + (F_state - F_state0)  # VALUE anchor: f base == efc.force_ref byte-exact
  phi_out[w, row] = -Z * f


@cache_kernel
def _constraint_scatter(is_sparse: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in (frozen):
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_state_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    nefc_in: wp.array[int],
    nv: int,
    # Leaf adjoints (per row):
    adjP_in: wp.array2d[float],  # P̄ (∂φ/∂efc.pos)
    adjV_in: wp.array2d[float],  # V̄ (∂φ/∂efc.vel)
    # Out (accumulate):
    res_qvel: wp.array2d[float],
    res_dof: wp.array2d[float],
  ):
    """SCATTER (manual): re-walk the row's J support (∂efc.vel/∂qvel = ∂P/∂qpos_tangent = J for the
    supported rows). ``res_qvel_i += J_ei·V̄_e`` for ALL rows; ``res_dof_i += J_ei·P̄_e`` for POSITION-BEARING
    rows ONLY -- NOT FRICTION_DOF (its P≡0 is not a function of qpos; routing P̄ would invent a spurious
    qpos-stiffness, MJPLAN_CSR must-fix #1). Atomics: res_qvel/res_dof are shared across rows that touch a dof."""
    w, row = wp.tid()
    if row >= nefc_in[w]:
      return
    ty = efc_type_in[w, row]
    if ty != _EQUALITY and ty != _LIMIT_JOINT and ty != _FRICTION_DOF:
      return
    if efc_state_in[w, row] == _SATISFIED:
      return
    Vb = adjV_in[w, row]
    Pb = adjP_in[w, row]
    route_p = ty == _LIMIT_JOINT or ty == _EQUALITY  # position-bearing; FRICTION_DOF excluded (must-fix #1)
    if wp.static(is_sparse):
      rownnz = efc_J_rownnz_in[w, row]
      rowadr = efc_J_rowadr_in[w, row]
      for kk in range(rownnz):
        sid = rowadr + kk
        i = efc_J_colind_in[w, 0, sid]
        jj = efc_J_in[w, 0, sid]
        wp.atomic_add(res_qvel[w], i, jj * Vb)
        if route_p:
          wp.atomic_add(res_dof[w], i, jj * Pb)
    else:
      for i in range(nv):
        jj = efc_J_in[w, row, i]
        if jj != 0.0:
          wp.atomic_add(res_qvel[w], i, jj * Vb)
          if route_p:
            wp.atomic_add(res_dof[w], i, jj * Pb)

  return kernel


# ============================================================================================
# CONTACT constraint residual VJP -- the elliptic/pyramidal cone force law (moved from adjoint.py).
# The CONTACT-row counterpart of the non-contact gather/leaf/scatter above; both are the VJP of
# constraint.py's efc rows. Orchestrated by adjoint.contact_residual_backward (which stays in
# adjoint.py): the dense `_residual_contact` oracle + the sparse `_contact_gather/phi/scatter`.
# ============================================================================================
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
def _jac_dif(
  body0: int,
  body1: int,
  dofid: int,
  point: wp.vec3,
  w: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
) -> Tuple[wp.vec3, wp.vec3]:
  """Per-dof contact-Jacobian difference J_i = jac_dof(body1, i) - jac_dof(body0, i) -> (jacp_dif,
  jacr_dif), each via that body's OWN subtree_com[body_rootid] (jac_dof returns 0 for non-ancestor dofs).
  Mirrors the forward constraint._efc_contact_jac_sparse exactly -> handles body-vs-world AND body-vs-body.
  cdof/subtree_com frozen and `point` frozen (its qpos-tracking = the S3 narrowphase)."""
  jp1, jr1 = _support.jac_dof(body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, point, body1, dofid, w)
  jp0, jr0 = _support.jac_dof(body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, point, body0, dofid, w)
  return jp1 - jp0, jr1 - jr0


@wp.func
def _row_coef(jpd: wp.vec3, jrd: wp.vec3, fm: wp.mat33, dimid: int) -> float:
  """Contact-Jacobian coefficient of one dof for row ``dimid``: frame-axis . jac_dif. dimid<3 =
  translational (jacp_dif; axes normal/tangent1/tangent2), dimid>=3 = rotational (jacr_dif; axis dimid-3)."""
  if dimid < 3:
    return wp.dot(_frame_axis(fm, dimid), jpd)
  return wp.dot(_frame_axis(fm, dimid - 3), jrd)


@wp.func
def _row_Jqv_Jqa(
  body0: int,
  body1: int,
  point: wp.vec3,
  fm: wp.mat33,
  dimid: int,
  nv: int,
  w: int,
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
) -> Tuple[float, float]:
  """Row ``dimid``'s (J.qvel, J.qacc) = Σ_i coef_i·{qvel_i, qacc_i}. STATIC loop (range _MAX_NV + runtime
  nv guard, like the condim loops) so Warp unrolls + replays it -- the reduction feeds the nonlinear cone T."""
  Jqv = float(0.0)
  Jqa = float(0.0)
  for i in range(_MAX_NV):
    if i < nv:
      jpd, jrd = _jac_dif(body0, body1, i, point, w, body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in)
      c = _row_coef(jpd, jrd, fm, dimid)
      Jqv += c * qvel_in[w, i]
      Jqa += c * qacc_in[w, i]
  return Jqv, Jqa


@wp.func
def _edge_coef(jpd: wp.vec3, jrd: wp.vec3, fm: wp.mat33, fric: vec5, e: int, condim: int) -> float:
  """Pyramidal edge ``e``'s coefficient of one dof: normal row + (condim>1) (±friction_k)·(k-th friction
  direction). dimid2 = e/2 + 1 selects the friction dir; the two edges per dir take +/- the friction."""
  c = _row_coef(jpd, jrd, fm, 0)
  if condim > 1:
    dimid2 = e / 2 + 1
    fs = _friction(fric, dimid2 - 1) * (1.0 - 2.0 * float(e % 2))
    c += fs * _row_coef(jpd, jrd, fm, dimid2)
  return c


@wp.func
def _edge_Jqv_Jqa(
  body0: int,
  body1: int,
  point: wp.vec3,
  fm: wp.mat33,
  fric: vec5,
  e: int,
  condim: int,
  nv: int,
  w: int,
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
) -> Tuple[float, float]:
  """Pyramidal edge ``e``'s (J_edge.qvel, J_edge.qacc), J_edge = normal ± friction·tangent. STATIC dof loop."""
  Jqv = float(0.0)
  Jqa = float(0.0)
  for i in range(_MAX_NV):
    if i < nv:
      jpd, jrd = _jac_dif(body0, body1, i, point, w, body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in)
      c = _edge_coef(jpd, jrd, fm, fric, e, condim)
      Jqv += c * qvel_in[w, i]
      Jqa += c * qacc_in[w, i]
  return Jqv, Jqa


@wp.func
def _elliptic_TN(
  body0: int,
  body1: int,
  point: wp.vec3,
  fm: wp.mat33,
  fric: vec5,
  k: float,
  b: float,
  b_t: float,
  mu: float,
  imp: float,
  pos: float,
  condim: int,
  nv: int,
  w: int,
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
) -> Tuple[float, float]:
  """Elliptic middle-zone cone coupling: N = mu·Jaref_normal, T = sqrt(Σ_{j=1..condim-1} (Jaref_j·fric_j)^2).
  STATIC row loop (range _MAXCONDIM + condim guard); the √Σ is nonlinear, so the accumulation must unroll."""
  Jqv0, Jqa0 = _row_Jqv_Jqa(body0, body1, point, fm, 0, nv, w, qvel_in, qacc_in, body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in)
  N = _row_jaref(Jqa0, Jqv0, k, b, imp, pos) * mu
  TT = float(0.0)
  for j in range(1, _MAXCONDIM):
    if j < condim:
      Jqvj, Jqaj = _row_Jqv_Jqa(body0, body1, point, fm, j, nv, w, qvel_in, qacc_in, body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in)
      uj = _row_jaref(Jqaj, Jqvj, 0.0, b_t, 0.0, 0.0) * _friction(fric, j - 1)
      TT += uj * uj
  T = wp.sqrt(wp.max(TT, _MINVAL * _MINVAL))
  return T, N


@cache_kernel
def _residual_contact(cone_type: int):
  """Per-step contact residual r = -Jᵀ f(qvel, qacc) for the IFT backward, cone-specialized and GENERAL
  over articulations. Each contact's per-dof Jacobian is built from ``support.jac_dof``
  (``J_i = jac_dof(body1,i) - jac_dof(body0,i)``, each body via its OWN ``subtree_com`` -> body-vs-world
  AND body-vs-body); the per-row force law (elliptic cone T-coupling / pyramidal edges, shared with the
  forward) acts on each row's ``J·qvel``/``J·qacc`` scalars, and ``r_i`` accumulates
  ``-Σ_rows force_row·coef_row,i``. The contact point + penetration are FROZEN (S2: qvel-exact, contact
  ∂qpos = 0; the narrowphase ∂cpos/∂qpos is S3). Per-dof/row reductions are STATIC (range
  ``_MAX_NV``/``_MAXCONDIM`` + runtime guards) so Warp unrolls + replays them -- the nonlinear cone ``T``
  must not be accumulated in the dynamic contact loop (the dynamic-loop staleness gotcha); the contact
  loop stays dynamic with a plain linear ``+=`` into ``r``."""
  IS_ELLIPTIC = cone_type == _ELLIPTIC

  @wp.kernel(module="unique", enable_backward=True)
  def kernel(
    qpos_in: wp.array2d[float],  # unused in-kernel (narrowphase ∂qpos is the S3 VJP); kept for the res_qpos slot
    qvel_in: wp.array2d[float],  # [grad]
    qacc_in: wp.array2d[float],  # frozen
    cdof_in: wp.array2d[wp.spatial_vector],  # frozen motion-dof axes (the contact Jacobian basis)
    subtree_com_in: wp.array2d[wp.vec3],  # frozen per-body com (jac_dof moment-arm reference)
    efc_D_in: wp.array2d[float],  # frozen
    efc_state_in: wp.array2d[int],  # frozen active set
    efc_pos_in: wp.array2d[float],  # frozen row position (normal: contact dist)
    efc_margin_in: wp.array2d[float],  # frozen include margin
    contact_pos_in: wp.array(dtype=wp.vec3),  # frozen contact point
    contact_frame_in: wp.array(dtype=wp.mat33),  # per-contact frame; rows = normal, tangent1, tangent2
    contact_friction_in: wp.array(dtype=vec5),
    contact_solref_in: wp.array(dtype=wp.vec2),
    contact_solreffriction_in: wp.array(dtype=wp.vec2),
    contact_solimp_in: wp.array(dtype=vec5),
    contact_dim_in: wp.array[int],
    contact_geom_in: wp.array(dtype=wp.vec2i),
    geom_bodyid_in: wp.array[int],
    body_isdofancestor_in: wp.array2d[int],
    body_parentid_in: wp.array[int],
    body_rootid_in: wp.array[int],
    dof_bodyid_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    nacon_in: wp.array[int],
    opt_timestep: wp.array[float],
    opt_impratio_invsqrt: wp.array[float],
    opt_disableflags: int,
    nv: int,
    efc_pos_ref_in: wp.array2d[float],  # FROZEN efc_pos (no adjoint): the D-recovery reference (pos0/imp0),
    # kept SEPARATE from the differentiated efc_pos_in so _contact_D's invweight stays frozen -- else AD
    # differentiates imp0==imp and ∂D/∂pos is wrong in the unsaturated-solimp regime (CONE hides it; the
    # settled/QUADRATIC deep-penetration regime exposes it).
    r_out: wp.array2d[float],  # out: contact residual (pre-zeroed)
  ):
    w = wp.tid()
    dt = opt_timestep[w % opt_timestep.shape[0]]
    imp_isq = opt_impratio_invsqrt[w % opt_impratio_invsqrt.shape[0]]

    for cid in range(nacon_in[0]):  # dynamic contact loop (linear += into r)
      if contact_worldid_in[cid] != w:
        continue
      e0 = contact_efc_address_in[cid, 0]
      if e0 < 0:
        continue
      st = efc_state_in[w, e0]
      if st == _SATISFIED:
        continue
      geom = contact_geom_in[cid]
      if geom[0] < 0 or geom[1] < 0:  # flex (negative geom ids) -- unsupported
        continue
      body0 = geom_bodyid_in[geom[0]]
      body1 = geom_bodyid_in[geom[1]]
      condim = contact_dim_in[cid]
      fm = contact_frame_in[cid]
      fric = contact_friction_in[cid]
      solref = contact_solref_in[cid]
      solimp = contact_solimp_in[cid]
      pos0 = efc_pos_ref_in[w, e0] - efc_margin_in[w, e0]  # FROZEN penetration (D-recovery reference)
      imp0 = _constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos0)[2]  # frozen-pos imp (D ref)

      # The contact point + penetration are FROZEN here (point = contact_pos_in, pos = pos0); their
      # qpos-tracking (∂cpos/∂qpos, ∂n/∂qpos, ∂pos/∂qpos) is the GENERAL S3 narrowphase VJP
      # (_narrowphase_recompute + _geom_pose_qpos_vjp in step_backward), which auto-diffs the forward
      # narrowphase pure funcs per geom pair and chains to qpos via jac_dof. (Replaced the old
      # single-free-body sphere-vs-flat in-kernel stopgap.)
      point = contact_pos_in[cid]
      pos = efc_pos_in[w, e0] - efc_margin_in[w, e0]  # DIFFERENTIABLE penetration (-> res_efc_pos)

      kbimp = _constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos)
      k = kbimp[0]
      b = kbimp[1]
      imp = kbimp[2]
      D0 = _contact_D(efc_D_in[w, e0], imp0, imp)

      if wp.static(IS_ELLIPTIC):
        ref_t = solref
        solreffriction = contact_solreffriction_in[cid]
        if solreffriction[0] != 0.0 or solreffriction[1] != 0.0:
          ref_t = solreffriction
        b_t = _constraint._contact_kbimp(opt_disableflags, dt, ref_t, solimp, pos)[1]
        mu = fric[0] * imp_isq

        if (condim > 1) and (st == _CONE):  # middle zone: cone-coupled forces, shared T
          T, N = _elliptic_TN(body0, body1, point, fm, fric, k, b, b_t, mu, imp, pos, condim, nv, w,
                              qvel_in, qacc_in, body_parentid_in, body_rootid_in, dof_bodyid_in,
                              body_isdofancestor_in, subtree_com_in, cdof_in)
          fn = _solver._eval_elliptic_cone(N, T, D0, mu, 0.0, True)[0]  # normal-row force
          for i in range(_MAX_NV):
            if i < nv:
              jpd, jrd = _jac_dif(body0, body1, i, point, w, body_parentid_in, body_rootid_in, dof_bodyid_in, body_isdofancestor_in, subtree_com_in, cdof_in)
              r_out[w, i] += -fn * _row_coef(jpd, jrd, fm, 0)
          for j in range(1, _MAXCONDIM):
            if j < condim:
              Jqvj, Jqaj = _row_Jqv_Jqa(body0, body1, point, fm, j, nv, w, qvel_in, qacc_in, body_parentid_in, body_rootid_in, dof_bodyid_in, body_isdofancestor_in, subtree_com_in, cdof_in)
              frij = _friction(fric, j - 1)
              fj = _solver._eval_elliptic_cone(N, T, D0, mu, _row_jaref(Jqaj, Jqvj, 0.0, b_t, 0.0, 0.0) * frij * frij, False)[0]
              for i in range(_MAX_NV):
                if i < nv:
                  jpd, jrd = _jac_dif(body0, body1, i, point, w, body_parentid_in, body_rootid_in, dof_bodyid_in, body_isdofancestor_in, subtree_com_in, cdof_in)
                  r_out[w, i] += -fj * _row_coef(jpd, jrd, fm, j)
        else:  # bottom zone / frictionless: each row force = -D_row * Jaref_row
          Jqv0, Jqa0 = _row_Jqv_Jqa(body0, body1, point, fm, 0, nv, w, qvel_in, qacc_in, body_parentid_in, body_rootid_in, dof_bodyid_in, body_isdofancestor_in, subtree_com_in, cdof_in)
          f0 = -D0 * _row_jaref(Jqa0, Jqv0, k, b, imp, pos)
          for i in range(_MAX_NV):
            if i < nv:
              jpd, jrd = _jac_dif(body0, body1, i, point, w, body_parentid_in, body_rootid_in, dof_bodyid_in, body_isdofancestor_in, subtree_com_in, cdof_in)
              r_out[w, i] += -f0 * _row_coef(jpd, jrd, fm, 0)
          for j in range(1, _MAXCONDIM):
            if j < condim:
              Jqvj, Jqaj = _row_Jqv_Jqa(body0, body1, point, fm, j, nv, w, qvel_in, qacc_in, body_parentid_in, body_rootid_in, dof_bodyid_in, body_isdofancestor_in, subtree_com_in, cdof_in)
              Dj = _contact_D(efc_D_in[w, contact_efc_address_in[cid, j]], imp0, imp)
              fj = -Dj * _row_jaref(Jqaj, Jqvj, 0.0, b_t, 0.0, 0.0)
              for i in range(_MAX_NV):
                if i < nv:
                  jpd, jrd = _jac_dif(body0, body1, i, point, w, body_parentid_in, body_rootid_in, dof_bodyid_in, body_isdofancestor_in, subtree_com_in, cdof_in)
                  r_out[w, i] += -fj * _row_coef(jpd, jrd, fm, j)
      else:  # PYRAMIDAL: ndim = 2*(condim-1) edges (1 if condim==1); each edge an independent force
        ndim = int(1)
        if condim > 1:
          ndim = 2 * (condim - 1)
        for e in range(_MAX_PYRAMID_EDGES):
          if e < ndim:
            ea = contact_efc_address_in[cid, e]
            if ea >= 0 and efc_state_in[w, ea] != _SATISFIED:
              Jqve, Jqae = _edge_Jqv_Jqa(body0, body1, point, fm, fric, e, condim, nv, w, qvel_in, qacc_in, body_parentid_in, body_rootid_in, dof_bodyid_in, body_isdofancestor_in, subtree_com_in, cdof_in)
              fe = -_contact_D(efc_D_in[w, ea], imp0, imp) * _row_jaref(Jqae, Jqve, k, b, imp, pos)
              for i in range(_MAX_NV):
                if i < nv:
                  jpd, jrd = _jac_dif(body0, body1, i, point, w, body_parentid_in, body_rootid_in, dof_bodyid_in, body_isdofancestor_in, subtree_com_in, cdof_in)
                  r_out[w, i] += -fe * _edge_coef(jpd, jrd, fm, fric, e, condim)

  return kernel


# ----------------------------------------------------------------------------
# 3b. SPARSE contract-first contact residual VJP (MJPLAN_ARTICULATION S4; the nv-general successor to the
#     dense `_residual_contact` above). Three pieces (mirror the smooth_adjoint pivot: manual sparse
#     reductions + a loop-free source-AD nonlinear leaf):
#       gather:  V = Σ_i h_i qvel_i, A = Σ_i h_i qacc_i, Z = Σ_i h_i λ_i  -- contact-point spatial motions,
#                h_i = side_i · jac_dof(body_i, i, p) (spatial: top=cdof_ang, bottom=cdof_lin+cdof_ang×(p-com)).
#       leaf:    φ_c = λᵀr_c = -Z · F(V,A,ξ), the SAME cone force law F as the forward, but on the contact-
#                frame projections of V/A (loop-free in nv). Including Z = Jλ makes the leaf's Z̄ = -F carry
#                the direct -Jᵀf projection path automatically (Codex). Source-AD (adj_φ=+1) -> V̄,Ā,Z̄ +
#                contact_frame / efc_pos cotangents.
#       scatter: walk the same chain; G_i = side_i(qvel_i V̄ + qacc_i Ā + λ_i Z̄) is the cotangent on the raw
#                jac column -> res_qvel (h_i·V̄), res_cdof, res_subtree_com, res_contact_pos (the moment-arm /
#                screw derivatives of h_i wrt cdof, com, p). The SAME five seeds the dense kernel produced.
# The gather/scatter walk only the SYMMETRIC DIFFERENCE of the two geoms' ancestor-dof chains (shared
# ancestors cancel in J = J1 - J0), mirroring constraint._efc_contact_jac_sparse -> O(chain) per contact, no
# _MAX_NV. enable_backward=False on gather/scatter -> their dynamic while-loop is a SAFE manual VJP (Warp
# does not replay dynamic loops in reverse, but these run forward only). Only the loop-free leaf is AD'd.
# ----------------------------------------------------------------------------
@wp.func
def _proj_row_spatial(Vsp: wp.spatial_vector, fm: wp.mat33, dimid: int) -> float:
  """Project a contact-point spatial motion (spatial_top=angular, spatial_bottom=linear) onto contact-frame
  row ``dimid`` -- the SUMMED form of ``_row_coef``: Σ_i coef_{dimid,i}·x_i = frame_axis·(Σ_i jac_dif_i·x_i).
  Translational rows (0/1/2) use the linear part, rotational rows (3/4/5) the angular part."""
  if dimid < 3:
    return wp.dot(_frame_axis(fm, dimid), wp.spatial_bottom(Vsp))
  return wp.dot(_frame_axis(fm, dimid - 3), wp.spatial_top(Vsp))


@wp.func
def _proj_edge_spatial(Vsp: wp.spatial_vector, fm: wp.mat33, fric: vec5, e: int, condim: int) -> float:
  """Pyramidal edge ``e``'s projection of a contact-point spatial motion -- the SUMMED ``_edge_coef``:
  normal row + (condim>1) (±friction_k)·(k-th friction direction)."""
  c = _proj_row_spatial(Vsp, fm, 0)
  if condim > 1:
    dimid2 = e / 2 + 1
    fs = _friction(fric, dimid2 - 1) * (1.0 - 2.0 * float(e % 2))
    c += fs * _proj_row_spatial(Vsp, fm, dimid2)
  return c


@wp.kernel(enable_backward=False)
def _contact_gather(
  # Model (sparse ancestor walk):
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  dof_parentid: wp.array[int],
  geom_bodyid: wp.array[int],
  # Data in (frozen):
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  lam_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  contact_pos_in: wp.array(dtype=wp.vec3),
  contact_geom_in: wp.array(dtype=wp.vec2i),
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_state_in: wp.array2d[int],
  nacon_in: wp.array[int],
  # Out (per contact):
  V_out: wp.array(dtype=wp.spatial_vector),
  A_out: wp.array(dtype=wp.spatial_vector),
  Z_out: wp.array(dtype=wp.spatial_vector),
):
  """GATHER (manual, sparse): assemble each contact's three contact-point spatial motions V/A/Z. Walks
  the SYMMETRIC DIFFERENCE of the two geoms' ancestor-dof chains (side=+1 geom1, -1 geom0); shared
  ancestors cancel (J=J1-J0) so the walk breaks when the two chain pointers meet."""
  cid = wp.tid()
  z = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
  V_out[cid] = z
  A_out[cid] = z
  Z_out[cid] = z
  if cid >= nacon_in[0]:
    return
  w = contact_worldid_in[cid]
  e0 = contact_efc_address_in[cid, 0]
  if e0 < 0:
    return
  if efc_state_in[w, e0] == _SATISFIED:
    return
  geom = contact_geom_in[cid]
  if geom[0] < 0 or geom[1] < 0:  # flex (negative geom ids) -- unsupported
    return
  b0 = body_weldid[geom_bodyid[geom[0]]]
  b1 = body_weldid[geom_bodyid[geom[1]]]
  p = contact_pos_in[cid]
  V = z
  A = z
  Z = z
  d0 = body_dofadr[b0] + body_dofnum[b0] - 1
  d1 = body_dofadr[b1] + body_dofnum[b1] - 1
  while d0 >= 0 or d1 >= 0:
    if d0 == d1:  # reached the common ancestor chain -> all remaining dofs cancel
      break
    i = int(0)
    side = float(0.0)
    bb = int(0)
    if d1 > d0:
      i = d1
      side = 1.0
      bb = b1
      d1 = dof_parentid[d1]
    else:
      i = d0
      side = -1.0
      bb = b0
      d0 = dof_parentid[d0]
    cdof = cdof_in[w, i]
    a = wp.spatial_top(cdof)
    lin = wp.spatial_bottom(cdof)
    off = p - subtree_com_in[w, body_rootid[bb]]
    jacp = lin + wp.cross(a, off)
    h = wp.spatial_vector(side * a, side * jacp)  # top=angular, bottom=linear (cdof convention)
    V += h * qvel_in[w, i]
    A += h * qacc_in[w, i]
    Z += h * lam_in[w, i]
  V_out[cid] = V
  A_out[cid] = A
  Z_out[cid] = Z


@cache_kernel
def _contact_phi(cone_type: int):
  """Loop-free source-AD cone leaf (one thread per contact). φ_c = λᵀr_c = -Σ_rows force_row · (Z projected
  on row) = -Z·F(V,A,ξ). Reuses the forward force law (constraint._contact_kbimp, solver._eval_elliptic_cone,
  _contact_D, _row_jaref) on the contact-frame projections of V/A (no per-dof loop -- only the condim/edge
  static bounds). AD'd wrt V/A/Z (-> V̄/Ā/Z̄), contact_frame (-> res_contact_frame), efc_pos (-> res_efc_pos).
  Cone-specialized + cached, exactly like the dense `_residual_contact`."""
  IS_ELLIPTIC = cone_type == _ELLIPTIC

  @wp.kernel(module="unique", enable_backward=True)
  def kernel(
    V_in: wp.array(dtype=wp.spatial_vector),  # [grad]
    A_in: wp.array(dtype=wp.spatial_vector),  # [grad] (qacc frozen, but A feeds the qacc->cdof scatter path)
    Z_in: wp.array(dtype=wp.spatial_vector),  # [grad] (= Jλ; Z̄ = -F carries the direct -Jᵀf projection)
    contact_frame_in: wp.array(dtype=wp.mat33),  # [grad] -> res_contact_frame
    efc_pos_in: wp.array2d[float],  # [grad] -> res_efc_pos (penetration)
    efc_margin_in: wp.array2d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    contact_friction_in: wp.array(dtype=vec5),
    contact_solref_in: wp.array(dtype=wp.vec2),
    contact_solreffriction_in: wp.array(dtype=wp.vec2),
    contact_solimp_in: wp.array(dtype=vec5),
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    nacon_in: wp.array[int],
    opt_timestep: wp.array[float],
    opt_impratio_invsqrt: wp.array[float],
    opt_disableflags: int,
    efc_pos_ref_in: wp.array2d[float],  # FROZEN efc_pos (no adjoint): the D-recovery reference (pos0/imp0)
    phi_out: wp.array(dtype=float),
  ):
    cid = wp.tid()
    phi_out[cid] = 0.0
    if cid >= nacon_in[0]:
      return
    w = contact_worldid_in[cid]
    e0 = contact_efc_address_in[cid, 0]
    if e0 < 0:
      return
    st = efc_state_in[w, e0]
    if st == _SATISFIED:
      return
    dt = opt_timestep[w % opt_timestep.shape[0]]
    imp_isq = opt_impratio_invsqrt[w % opt_impratio_invsqrt.shape[0]]
    condim = contact_dim_in[cid]
    fm = contact_frame_in[cid]
    fric = contact_friction_in[cid]
    solref = contact_solref_in[cid]
    solimp = contact_solimp_in[cid]
    pos0 = efc_pos_ref_in[w, e0] - efc_margin_in[w, e0]  # FROZEN penetration (D-recovery reference)
    imp0 = _constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos0)[2]
    pos = efc_pos_in[w, e0] - efc_margin_in[w, e0]  # DIFFERENTIABLE penetration (-> res_efc_pos)
    kbimp = _constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos)
    k = kbimp[0]
    b = kbimp[1]
    imp = kbimp[2]
    D0 = _contact_D(efc_D_in[w, e0], imp0, imp)
    V = V_in[cid]
    A = A_in[cid]
    Z = Z_in[cid]
    phi = float(0.0)

    if wp.static(IS_ELLIPTIC):
      ref_t = solref
      solreffriction = contact_solreffriction_in[cid]
      if solreffriction[0] != 0.0 or solreffriction[1] != 0.0:
        ref_t = solreffriction
      b_t = _constraint._contact_kbimp(opt_disableflags, dt, ref_t, solimp, pos)[1]
      mu = fric[0] * imp_isq

      if (condim > 1) and (st == _CONE):  # middle zone: cone-coupled forces, shared N/T
        N = _row_jaref(_proj_row_spatial(A, fm, 0), _proj_row_spatial(V, fm, 0), k, b, imp, pos) * mu
        TT = float(0.0)
        for j in range(1, _MAXCONDIM):
          if j < condim:
            uj = _row_jaref(_proj_row_spatial(A, fm, j), _proj_row_spatial(V, fm, j), 0.0, b_t, 0.0, 0.0) * _friction(fric, j - 1)
            TT += uj * uj
        T = wp.sqrt(wp.max(TT, _MINVAL * _MINVAL))
        fn = _solver._eval_elliptic_cone(N, T, D0, mu, 0.0, True)[0]
        phi += -fn * _proj_row_spatial(Z, fm, 0)
        for j in range(1, _MAXCONDIM):
          if j < condim:
            frij = _friction(fric, j - 1)
            uj = _row_jaref(_proj_row_spatial(A, fm, j), _proj_row_spatial(V, fm, j), 0.0, b_t, 0.0, 0.0)
            fj = _solver._eval_elliptic_cone(N, T, D0, mu, uj * frij * frij, False)[0]
            phi += -fj * _proj_row_spatial(Z, fm, j)
      else:  # bottom zone / frictionless: each row force = -D_row · Jaref_row
        f0 = -D0 * _row_jaref(_proj_row_spatial(A, fm, 0), _proj_row_spatial(V, fm, 0), k, b, imp, pos)
        phi += -f0 * _proj_row_spatial(Z, fm, 0)
        for j in range(1, _MAXCONDIM):
          if j < condim:
            Dj = _contact_D(efc_D_in[w, contact_efc_address_in[cid, j]], imp0, imp)
            fj = -Dj * _row_jaref(_proj_row_spatial(A, fm, j), _proj_row_spatial(V, fm, j), 0.0, b_t, 0.0, 0.0)
            phi += -fj * _proj_row_spatial(Z, fm, j)
    else:  # PYRAMIDAL: ndim = 2*(condim-1) edges (1 if condim==1); each edge an independent force
      ndim = int(1)
      if condim > 1:
        ndim = 2 * (condim - 1)
      for e in range(_MAX_PYRAMID_EDGES):
        if e < ndim:
          ea = contact_efc_address_in[cid, e]
          if ea >= 0 and efc_state_in[w, ea] != _SATISFIED:
            Jqve = _proj_edge_spatial(V, fm, fric, e, condim)
            Jqae = _proj_edge_spatial(A, fm, fric, e, condim)
            fe = -_contact_D(efc_D_in[w, ea], imp0, imp) * _row_jaref(Jqae, Jqve, k, b, imp, pos)
            phi += -fe * _proj_edge_spatial(Z, fm, fric, e, condim)

    phi_out[cid] = phi

  return kernel


@wp.kernel
def _fill_ones(out: wp.array(dtype=float)):
  """Seed adj_φ = +1 (per contact). φ = λᵀr_c already folds in the IFT seed λ, so adj_φ is a plain 1."""
  out[wp.tid()] = 1.0


@wp.kernel(enable_backward=False)
def _contact_scatter(
  # Model (sparse ancestor walk):
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  dof_parentid: wp.array[int],
  geom_bodyid: wp.array[int],
  # Data in (frozen):
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  lam_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  contact_pos_in: wp.array(dtype=wp.vec3),
  contact_geom_in: wp.array(dtype=wp.vec2i),
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_state_in: wp.array2d[int],
  nacon_in: wp.array[int],
  # Leaf adjoints (per contact):
  adjV_in: wp.array(dtype=wp.spatial_vector),
  adjA_in: wp.array(dtype=wp.spatial_vector),
  adjZ_in: wp.array(dtype=wp.spatial_vector),
  # Out (accumulate):
  res_qvel: wp.array2d[float],
  res_cdof: wp.array2d[wp.spatial_vector],
  res_subtree_com: wp.array2d[wp.vec3],
  res_contact_pos: wp.array(dtype=wp.vec3),
):
  """SCATTER (manual, sparse): walk the same symmetric-difference chain. For each supported dof i,
  G_i = side_i(qvel_i·V̄ + qacc_i·Ā + λ_i·Z̄) is the cotangent on the RAW jac column h0_i = jac_dof(body,i,p)
  (= (a, l+a×off)). res_qvel += h_i·V̄; res_cdof += (ang: G_a + off×G_l, lin: G_l); res_contact_pos +=
  G_l×a; res_subtree_com += a×G_l. qacc is the implicit root (no adj_qacc), but its Ā term still feeds
  cdof/com (J·qacc depends on cdof(qpos)). Atomics: res_qvel/cdof/subtree_com are shared across contacts;
  res_contact_pos[cid] is this thread's unique writer."""
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  w = contact_worldid_in[cid]
  e0 = contact_efc_address_in[cid, 0]
  if e0 < 0:
    return
  if efc_state_in[w, e0] == _SATISFIED:
    return
  geom = contact_geom_in[cid]
  if geom[0] < 0 or geom[1] < 0:
    return
  b0 = body_weldid[geom_bodyid[geom[0]]]
  b1 = body_weldid[geom_bodyid[geom[1]]]
  p = contact_pos_in[cid]
  Vb = adjV_in[cid]
  Ab = adjA_in[cid]
  Zb = adjZ_in[cid]
  cpos_acc = wp.vec3(0.0)
  d0 = body_dofadr[b0] + body_dofnum[b0] - 1
  d1 = body_dofadr[b1] + body_dofnum[b1] - 1
  while d0 >= 0 or d1 >= 0:
    if d0 == d1:
      break
    i = int(0)
    side = float(0.0)
    bb = int(0)
    if d1 > d0:
      i = d1
      side = 1.0
      bb = b1
      d1 = dof_parentid[d1]
    else:
      i = d0
      side = -1.0
      bb = b0
      d0 = dof_parentid[d0]
    cdof = cdof_in[w, i]
    a = wp.spatial_top(cdof)
    lin = wp.spatial_bottom(cdof)
    off = p - subtree_com_in[w, body_rootid[bb]]
    jacp = lin + wp.cross(a, off)
    h = wp.spatial_vector(side * a, side * jacp)
    wp.atomic_add(res_qvel[w], i, wp.dot(h, Vb))  # ∂φ/∂qvel_i = h_i·V̄
    G = side * (qvel_in[w, i] * Vb + qacc_in[w, i] * Ab + lam_in[w, i] * Zb)  # cotangent on raw jac column
    Ga = wp.spatial_top(G)
    Gl = wp.spatial_bottom(G)
    wp.atomic_add(res_cdof[w], i, wp.spatial_vector(Ga + wp.cross(off, Gl), Gl))
    cpos_acc += wp.cross(Gl, a)  # ∂(off=p-com)/∂p
    wp.atomic_add(res_subtree_com[w], body_rootid[bb], wp.cross(a, Gl))  # ∂off/∂com = -∂off/∂p
  res_contact_pos[cid] = cpos_acc


