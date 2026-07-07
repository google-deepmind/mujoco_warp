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
"""Constraint residual VJP kernels (non-contact and contact efc rows) for the IFT backward."""

from typing import Tuple

import warp as wp

from mujoco_warp._src import constraint
from mujoco_warp._src import solver
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import cache_kernel

_SATISFIED = int(types.ConstraintState.SATISFIED.value)
_LINEARNEG = int(types.ConstraintState.LINEARNEG.value)  # saturated friction, force = +frictionloss
_LINEARPOS = int(types.ConstraintState.LINEARPOS.value)  # saturated friction, force = -frictionloss
_FRICTION_DOF = int(types.ConstraintType.FRICTION_DOF.value)
_EQUALITY = int(types.ConstraintType.EQUALITY.value)
_LIMIT_JOINT = int(types.ConstraintType.LIMIT_JOINT.value)
_MINVAL = float(types.MJ_MINVAL)
# Contact cone-residual constants (used by the CONTACT section below):
_CONE = int(types.ConstraintState.CONE.value)
_ELLIPTIC = int(types.ConeType.ELLIPTIC.value)
_MAXCONDIM = 6  # max valid MuJoCo condim; elliptic friction rows = dimid 1..condim-1
_MAX_PYRAMID_EDGES = 10  # 2*(_MAXCONDIM - 1) pyramidal edges at condim 6
_MAX_NV = 16  # static unroll bound of the dense `_residual_contact` path (sparse is nv-general)


# Non-contact constraint residual VJP (equality / joint-limit / dof-friction rows), orchestrated by
# adjoint._residual_constraint_sparse; CONTACT rows live in the bottom section of this file.
#   gather (manual): Z_e = sum_i J_ei*lam_i + TRUE topology invweight (A/V stay frozen anchors).
#   leaf _constraint_row_phi (loop-free, only AD'd piece): phi_e = -Z*f, f anchored to efc.force.
#   scatter (manual): res_qvel += J*Vbar (all rows); res_dof += J*Pbar (position-bearing only).
# Routing: dense/CSR per m.is_sparse; legacy/new per _model_has_unsupported_noncontact_rows only.
@cache_kernel
def _constraint_gather(is_sparse: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    jnt_dofadr: wp.array[int],
    dof_invweight0: wp.array2d[float],
    # Data in:
    nefc_in: wp.array[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_state_in: wp.array2d[int],
    # In:
    lam_in: wp.array2d[float],
    # Out:
    Z_out: wp.array2d[float],
    invw_out: wp.array2d[float],
  ):
    """GATHER (manual): per active non-contact row, reduce Z = J*lam and the topology invweight.

    Mirrors solver._solve_init_jaref_kernel's CSR/dense J iterator (keep in sync).
    """
    w, row = wp.tid()
    Z_out[w, row] = 0.0
    invw_out[w, row] = 0.0
    if row >= nefc_in[w]:
      return
    ty = efc_type_in[w, row]
    if ty != _EQUALITY and ty != _LIMIT_JOINT and ty != _FRICTION_DOF:
      return  # contact rows -> _contact_*; TODO(team): tendon limit/friction (structural gate raises)
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
    # EQUALITY / other -> invw=0 (leaf uses the frozen-D fallback; solimp grad gated, not exact)

  return kernel


@wp.kernel(module="unique", enable_backward=True)
def _constraint_row_phi(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  jnt_solref: wp.array2d[wp.vec2],
  jnt_solimp: wp.array2d[vec5],
  dof_solref: wp.array2d[wp.vec2],
  dof_solimp: wp.array2d[vec5],
  dof_frictionloss: wp.array2d[float],
  eq_solref: wp.array2d[wp.vec2],
  eq_solimp: wp.array2d[vec5],
  # Data in:
  nefc_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_pos_in: wp.array2d[float],  # [grad] penetration source (P = efc.pos - margin) -> Pbar (res_efc_pos)
  efc_margin_in: wp.array2d[float],  # frozen
  efc_D_in: wp.array2d[float],  # frozen efc.D_ref (D base + frozen-D fallback)
  efc_vel_in: wp.array2d[float],  # [grad] J*qvel -> Vbar (res_efc_vel)
  efc_aref_in: wp.array2d[float],  # frozen (= -g_base; cancels g0 so jaref base == ctx.Jaref)
  efc_force_in: wp.array2d[float],  # frozen efc.force_ref (force VALUE anchor)
  efc_state_in: wp.array2d[int],  # frozen active set (NEVER re-derived from a perturbed jaref)
  # In:
  Z_in: wp.array2d[float],  # [grad] = J*lam (gather) -> Zbar (= -f; the dJ/dq topology seed, unused for G1)
  invw_in: wp.array2d[float],  # frozen TRUE topology invweight (0 -> frozen-D fallback)
  ctx_Jaref_in: wp.array2d[float],  # frozen J*qacc - aref (jaref VALUE anchor)
  # Out:
  phi_out: wp.array2d[float],
):
  """LOOP-FREE row leaf: phi_e = -Z*f, f VALUE-anchored to the stored efc.force."""
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
    phi_out[w, row] = -Z * dof_frictionloss[w % dof_frictionloss.shape[0], cid]
    return
  if ty == _FRICTION_DOF and st == _LINEARPOS:  # saturated friction: force = -frictionloss
    phi_out[w, row] = -Z * (-dof_frictionloss[w % dof_frictionloss.shape[0], cid])
    return
  # QUADRATIC (equality / active limit / stuck friction): f = -D*jaref, value-anchored to efc.force.
  dt = opt_timestep[w % opt_timestep.shape[0]]
  P = efc_pos_in[w, row] - efc_margin_in[w, row]
  V = efc_vel_in[w, row]
  if ty == _FRICTION_DOF:
    sr = dof_solref[w % dof_solref.shape[0], cid]
    si = dof_solimp[w % dof_solimp.shape[0], cid]
    kbi = constraint._contact_kbimp(opt_disableflags, dt, sr, si, P)
  elif ty == _EQUALITY:
    sr = eq_solref[w % eq_solref.shape[0], cid]
    si = eq_solimp[w % eq_solimp.shape[0], cid]
    kbi = constraint._contact_kbimp(opt_disableflags, dt, sr, si, P)
  else:  # _LIMIT_JOINT (slide/hinge scalar J; ball 3-dof -axis J)
    sr = jnt_solref[w % jnt_solref.shape[0], cid]
    si = jnt_solimp[w % jnt_solimp.shape[0], cid]
    kbi = constraint._contact_kbimp(opt_disableflags, dt, sr, si, P)
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
    # Model:
    nv: int,
    # Data in:
    nefc_in: wp.array[int],
    efc_type_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_state_in: wp.array2d[int],
    # In:
    adjP_in: wp.array2d[float],  # Pbar (dphi/defc.pos)
    adjV_in: wp.array2d[float],  # Vbar (dphi/defc.vel)
    res_qvel: wp.array2d[float],
    res_dof: wp.array2d[float],
  ):
    """SCATTER (manual): re-walk the row's J support; res_qvel += J*Vbar for all active rows.

    res_dof += J*Pbar only for position-bearing rows -- a friction row's P is identically zero.
    """
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
    route_p = ty == _LIMIT_JOINT or ty == _EQUALITY  # position-bearing; FRICTION_DOF excluded (its P == 0, no qpos route)
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
# CONTACT constraint residual VJP -- the elliptic/pyramidal cone force law. Orchestrated by
# adjoint.contact_residual_backward, routing on the forward's jacobian storage: dense
# `_residual_contact` (one monolithic source-AD pass, nv<=_MAX_NV static unroll) when
# not m.is_sparse and nv<=_MAX_NV, else the SPARSE contract-first path (gather V/A/Z over the
# symmetric-difference ancestor-dof walk, source-AD cone leaf phi=-Z*F(V,A,xi), manual scatter;
# nv-general, reads no efc.J). test_sparse_vs_dense_oracle_match pins the two to ~1e-5.
# ============================================================================================
# Per-row physics shared with the forward: (k, b, impedance) from constraint._contact_kbimp and
# the elliptic cone force from solver._eval_elliptic_middle (single source of truth). Only
# _contact_D (differentiable D recovered from the frozen converged D) is backward-only.
@wp.func
def _contact_D(D_base: float, imp_base: float, imp: float) -> float:
  """Recover fixed invweight from converged D, then evaluate D at differentiable imp."""
  invweight = (1.0 / wp.max(D_base, _MINVAL)) * imp_base / wp.max(1.0 - imp_base, _MINVAL)
  return 1.0 / wp.max(invweight * (1.0 - imp) / imp, _MINVAL)


@wp.func
def _contact_dof_coefficient(
  # Model:
  geom_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # In:
  geom: wp.vec2i,
  dofid: int,
) -> float:
  """Coefficient of one dof in a rigid-geom contact's J = J_geom1 - J_geom0 (+1/-1/0; flex -> 0)."""
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
  """Friction coef by row index; STATIC lookups (a runtime vec index corrupts Warp's adjoint)."""
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
  """Jaref = J*qacc - aref, aref = -k*imp*pos_aref - b*(J*qvel) (mirrors constraint._efc_row)."""
  return Jqa - (-k * imp * pos_aref - b * Jqv)


@wp.func
def _jac_dif(
  # Model:
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  body0: int,
  body1: int,
  dofid: int,
  point: wp.vec3,
  w: int,
) -> Tuple[wp.vec3, wp.vec3]:
  """Per-dof contact-Jacobian difference: (jacp_dif, jacr_dif) = jac_dof(body1) - jac_dof(body0).

  Mirrors constraint._efc_contact_jac_sparse: each body via its OWN subtree_com[body_rootid].
  """
  jp1, jr1 = support.jac_dof(
    body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, point, body1, dofid, w
  )
  jp0, jr0 = support.jac_dof(
    body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, point, body0, dofid, w
  )
  return jp1 - jp0, jr1 - jr0


@wp.func
def _row_coef(jpd: wp.vec3, jrd: wp.vec3, fm: wp.mat33, dimid: int) -> float:
  """One dof's coefficient for contact row dimid: frame-axis dot jacp_dif (dimid<3) or jacr_dif."""
  if dimid < 3:
    return wp.dot(_frame_axis(fm, dimid), jpd)
  return wp.dot(_frame_axis(fm, dimid - 3), jrd)


@wp.func
def _row_Jqv_Jqa(
  # Model:
  nv: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  body0: int,
  body1: int,
  point: wp.vec3,
  fm: wp.mat33,
  dimid: int,
  w: int,
) -> Tuple[float, float]:
  """Row dimid's (J.qvel, J.qacc) = sum_i coef_i*(qvel_i, qacc_i); STATIC dof loop."""
  Jqv = float(0.0)
  Jqa = float(0.0)
  for i in range(_MAX_NV):
    if i < nv:
      jpd, jrd = _jac_dif(
        body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, body0, body1, i, point, w
      )
      c = _row_coef(jpd, jrd, fm, dimid)
      Jqv += c * qvel_in[w, i]
      Jqa += c * qacc_in[w, i]
  return Jqv, Jqa


@wp.func
def _edge_coef(jpd: wp.vec3, jrd: wp.vec3, fm: wp.mat33, fric: vec5, e: int, condim: int) -> float:
  """Pyramidal edge e's dof coefficient: normal coef +/- friction * (friction-dir e/2+1 coef)."""
  c = _row_coef(jpd, jrd, fm, 0)
  if condim > 1:
    dimid2 = e / 2 + 1
    fs = _friction(fric, dimid2 - 1) * (1.0 - 2.0 * float(e % 2))
    c += fs * _row_coef(jpd, jrd, fm, dimid2)
  return c


@wp.func
def _edge_Jqv_Jqa(
  # Model:
  nv: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  body0: int,
  body1: int,
  point: wp.vec3,
  fm: wp.mat33,
  fric: vec5,
  e: int,
  condim: int,
  w: int,
) -> Tuple[float, float]:
  """Pyramidal edge e's (J_edge.qvel, J_edge.qacc); STATIC dof loop."""
  Jqv = float(0.0)
  Jqa = float(0.0)
  for i in range(_MAX_NV):
    if i < nv:
      jpd, jrd = _jac_dif(
        body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in, cdof_in, body0, body1, i, point, w
      )
      c = _edge_coef(jpd, jrd, fm, fric, e, condim)
      Jqv += c * qvel_in[w, i]
      Jqa += c * qacc_in[w, i]
  return Jqv, Jqa


@wp.func
def _elliptic_TN(
  # Model:
  nv: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
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
  w: int,
) -> Tuple[float, float]:
  """Elliptic middle-zone coupling: (T, N) = (sqrt(sum_j (Jaref_j*fric_j)^2), mu*Jaref_normal).

  STATIC row loop: the nonlinear sqrt-of-sum accumulation must unroll for Warp's replay.
  """
  Jqv0, Jqa0 = _row_Jqv_Jqa(
    nv,
    body_parentid,
    body_rootid,
    dof_bodyid,
    body_isdofancestor,
    qvel_in,
    qacc_in,
    subtree_com_in,
    cdof_in,
    body0,
    body1,
    point,
    fm,
    0,
    w,
  )
  N = _row_jaref(Jqa0, Jqv0, k, b, imp, pos) * mu
  TT = float(0.0)
  for j in range(1, _MAXCONDIM):
    if j < condim:
      Jqvj, Jqaj = _row_Jqv_Jqa(
        nv,
        body_parentid,
        body_rootid,
        dof_bodyid,
        body_isdofancestor,
        qvel_in,
        qacc_in,
        subtree_com_in,
        cdof_in,
        body0,
        body1,
        point,
        fm,
        j,
        w,
      )
      uj = _row_jaref(Jqaj, Jqvj, 0.0, b_t, 0.0, 0.0) * _friction(fric, j - 1)
      TT += uj * uj
  T = wp.sqrt(wp.max(TT, _MINVAL * _MINVAL))
  return T, N


@cache_kernel
def _residual_contact(cone_type: int):
  """Cone-specialized dense per-step contact residual r = -J^T f(qvel, qacc) for the IFT backward.

  Per-dof/row reductions are STATIC (Warp replay); only the linear += over contacts stays dynamic.
  """
  IS_ELLIPTIC = cone_type == _ELLIPTIC

  @wp.kernel(module="unique", enable_backward=True)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    opt_impratio_invsqrt: wp.array[float],
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    dof_bodyid: wp.array[int],
    geom_bodyid: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    # Data in:
    qpos_in: wp.array2d[float],  # unused in-kernel (d/dqpos is the narrowphase VJP); kept for the res_qpos slot
    qvel_in: wp.array2d[float],  # [grad]
    qacc_in: wp.array2d[float],  # frozen
    subtree_com_in: wp.array2d[wp.vec3],  # frozen per-body com (jac_dof moment-arm reference)
    cdof_in: wp.array2d[wp.spatial_vector],  # frozen motion-dof axes (the contact Jacobian basis)
    contact_pos_in: wp.array[wp.vec3],  # frozen contact point
    contact_frame_in: wp.array[wp.mat33],  # per-contact frame; rows = normal, tangent1, tangent2
    contact_friction_in: wp.array[vec5],
    contact_solref_in: wp.array[wp.vec2],
    contact_solreffriction_in: wp.array[wp.vec2],
    contact_solimp_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_geom_in: wp.array[wp.vec2i],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    efc_pos_in: wp.array2d[float],  # frozen row position (normal: contact dist)
    efc_margin_in: wp.array2d[float],  # frozen include margin
    efc_D_in: wp.array2d[float],  # frozen
    efc_state_in: wp.array2d[int],  # frozen active set
    nacon_in: wp.array[int],
    # In:
    efc_pos_ref_in: wp.array2d[float],  # FROZEN efc_pos (no adjoint): the D-recovery reference (pos0/imp0)
    # kept SEPARATE from differentiated efc_pos_in so _contact_D's invweight stays frozen -- else AD
    # differentiates imp0 == imp and dD/dpos is wrong in the unsaturated-solimp regime.
    # Out:
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
      body0 = geom_bodyid[geom[0]]
      body1 = geom_bodyid[geom[1]]
      condim = contact_dim_in[cid]
      fm = contact_frame_in[cid]
      fric = contact_friction_in[cid]
      solref = contact_solref_in[cid]
      solimp = contact_solimp_in[cid]
      pos0 = efc_pos_ref_in[w, e0] - efc_margin_in[w, e0]  # FROZEN penetration (D-recovery reference)
      imp0 = constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos0)[2]  # frozen-pos imp (D ref)

      # The contact point + penetration are FROZEN here (point = contact_pos_in, pos = pos0); their
      # qpos-tracking is the general narrowphase VJP (_narrowphase_recompute + _geom_pose_qpos_vjp
      # in step_backward), which auto-diffs the forward narrowphase pure funcs per geom pair and
      # chains to qpos via jac_dof.
      point = contact_pos_in[cid]
      pos = efc_pos_in[w, e0] - efc_margin_in[w, e0]  # DIFFERENTIABLE penetration (-> res_efc_pos)

      kbimp = constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos)
      k = kbimp[0]
      b = kbimp[1]
      imp = kbimp[2]
      D0 = _contact_D(efc_D_in[w, e0], imp0, imp)

      if wp.static(IS_ELLIPTIC):
        ref_t = solref
        solreffriction = contact_solreffriction_in[cid]
        if solreffriction[0] != 0.0 or solreffriction[1] != 0.0:
          ref_t = solreffriction
        b_t = constraint._contact_kbimp(opt_disableflags, dt, ref_t, solimp, pos)[1]
        mu = fric[0] * imp_isq

        if (condim > 1) and (st == _CONE):  # middle zone: cone-coupled forces, shared T
          T, N = _elliptic_TN(
            nv,
            body_parentid,
            body_rootid,
            dof_bodyid,
            body_isdofancestor,
            qvel_in,
            qacc_in,
            subtree_com_in,
            cdof_in,
            body0,
            body1,
            point,
            fm,
            fric,
            k,
            b,
            b_t,
            mu,
            imp,
            pos,
            condim,
            w,
          )
          fn = solver._eval_elliptic_middle(N, T, D0, mu, 0.0, True)[0]  # normal-row force
          for i in range(_MAX_NV):
            if i < nv:
              jpd, jrd = _jac_dif(
                body_parentid,
                body_rootid,
                dof_bodyid,
                body_isdofancestor,
                subtree_com_in,
                cdof_in,
                body0,
                body1,
                i,
                point,
                w,
              )
              r_out[w, i] += -fn * _row_coef(jpd, jrd, fm, 0)
          for j in range(1, _MAXCONDIM):
            if j < condim:
              Jqvj, Jqaj = _row_Jqv_Jqa(
                nv,
                body_parentid,
                body_rootid,
                dof_bodyid,
                body_isdofancestor,
                qvel_in,
                qacc_in,
                subtree_com_in,
                cdof_in,
                body0,
                body1,
                point,
                fm,
                j,
                w,
              )
              frij = _friction(fric, j - 1)
              fj = solver._eval_elliptic_middle(N, T, D0, mu, _row_jaref(Jqaj, Jqvj, 0.0, b_t, 0.0, 0.0) * frij * frij, False)[
                0
              ]
              for i in range(_MAX_NV):
                if i < nv:
                  jpd, jrd = _jac_dif(
                    body_parentid,
                    body_rootid,
                    dof_bodyid,
                    body_isdofancestor,
                    subtree_com_in,
                    cdof_in,
                    body0,
                    body1,
                    i,
                    point,
                    w,
                  )
                  r_out[w, i] += -fj * _row_coef(jpd, jrd, fm, j)
        else:  # bottom zone / frictionless: each row force = -D_row * Jaref_row
          Jqv0, Jqa0 = _row_Jqv_Jqa(
            nv,
            body_parentid,
            body_rootid,
            dof_bodyid,
            body_isdofancestor,
            qvel_in,
            qacc_in,
            subtree_com_in,
            cdof_in,
            body0,
            body1,
            point,
            fm,
            0,
            w,
          )
          f0 = -D0 * _row_jaref(Jqa0, Jqv0, k, b, imp, pos)
          for i in range(_MAX_NV):
            if i < nv:
              jpd, jrd = _jac_dif(
                body_parentid,
                body_rootid,
                dof_bodyid,
                body_isdofancestor,
                subtree_com_in,
                cdof_in,
                body0,
                body1,
                i,
                point,
                w,
              )
              r_out[w, i] += -f0 * _row_coef(jpd, jrd, fm, 0)
          for j in range(1, _MAXCONDIM):
            if j < condim:
              Jqvj, Jqaj = _row_Jqv_Jqa(
                nv,
                body_parentid,
                body_rootid,
                dof_bodyid,
                body_isdofancestor,
                qvel_in,
                qacc_in,
                subtree_com_in,
                cdof_in,
                body0,
                body1,
                point,
                fm,
                j,
                w,
              )
              Dj = _contact_D(efc_D_in[w, contact_efc_address_in[cid, j]], imp0, imp)
              fj = -Dj * _row_jaref(Jqaj, Jqvj, 0.0, b_t, 0.0, 0.0)
              for i in range(_MAX_NV):
                if i < nv:
                  jpd, jrd = _jac_dif(
                    body_parentid,
                    body_rootid,
                    dof_bodyid,
                    body_isdofancestor,
                    subtree_com_in,
                    cdof_in,
                    body0,
                    body1,
                    i,
                    point,
                    w,
                  )
                  r_out[w, i] += -fj * _row_coef(jpd, jrd, fm, j)
      else:  # PYRAMIDAL: ndim = 2*(condim-1) edges (1 if condim==1); each edge an independent force
        ndim = int(1)
        if condim > 1:
          ndim = 2 * (condim - 1)
        for e in range(_MAX_PYRAMID_EDGES):
          if e < ndim:
            ea = contact_efc_address_in[cid, e]
            if ea >= 0 and efc_state_in[w, ea] != _SATISFIED:
              Jqve, Jqae = _edge_Jqv_Jqa(
                nv,
                body_parentid,
                body_rootid,
                dof_bodyid,
                body_isdofancestor,
                qvel_in,
                qacc_in,
                subtree_com_in,
                cdof_in,
                body0,
                body1,
                point,
                fm,
                fric,
                e,
                condim,
                w,
              )
              fe = -_contact_D(efc_D_in[w, ea], imp0, imp) * _row_jaref(Jqae, Jqve, k, b, imp, pos)
              for i in range(_MAX_NV):
                if i < nv:
                  jpd, jrd = _jac_dif(
                    body_parentid,
                    body_rootid,
                    dof_bodyid,
                    body_isdofancestor,
                    subtree_com_in,
                    cdof_in,
                    body0,
                    body1,
                    i,
                    point,
                    w,
                  )
                  r_out[w, i] += -fe * _edge_coef(jpd, jrd, fm, fric, e, condim)

  return kernel


# SPARSE contract-first contact residual VJP: the nv-general successor to the dense
# `_residual_contact`. gather assembles per-contact spatial motions V/A/Z; the loop-free source-AD
# leaf phi_c = -Z*F(V,A,xi) reuses the forward cone law (Zbar = -F carries the direct -J^T f
# path); scatter routes the leaf adjoints. Symmetric-difference walk; forward-only manual VJP.


@wp.func
def _proj_row_spatial(Vsp: wp.spatial_vector, fm: wp.mat33, dimid: int) -> float:
  """Project a contact-point spatial motion onto contact-frame row dimid (the SUMMED _row_coef)."""
  if dimid < 3:
    return wp.dot(_frame_axis(fm, dimid), wp.spatial_bottom(Vsp))
  return wp.dot(_frame_axis(fm, dimid - 3), wp.spatial_top(Vsp))


@wp.func
def _proj_edge_spatial(Vsp: wp.spatial_vector, fm: wp.mat33, fric: vec5, e: int, condim: int) -> float:
  """Pyramidal edge e's projection of a contact-point spatial motion (the SUMMED _edge_coef)."""
  c = _proj_row_spatial(Vsp, fm, 0)
  if condim > 1:
    dimid2 = e / 2 + 1
    fs = _friction(fric, dimid2 - 1) * (1.0 - 2.0 * float(e % 2))
    c += fs * _proj_row_spatial(Vsp, fm, dimid2)
  return c


@wp.kernel(enable_backward=False)
def _contact_gather(
  # Model:
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  dof_parentid: wp.array[int],
  geom_bodyid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  contact_pos_in: wp.array[wp.vec3],
  contact_geom_in: wp.array[wp.vec2i],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_state_in: wp.array2d[int],
  nacon_in: wp.array[int],
  # In:
  lam_in: wp.array2d[float],
  # Out:
  V_out: wp.array[wp.spatial_vector],
  A_out: wp.array[wp.spatial_vector],
  Z_out: wp.array[wp.spatial_vector],
):
  """GATHER (manual, sparse): per-contact spatial motions V/A/Z (symmetric-difference dof walk)."""
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
  """Loop-free source-AD cone leaf (one thread per contact): phi_c = lam^T r_c = -Z*F(V,A,xi)."""
  IS_ELLIPTIC = cone_type == _ELLIPTIC

  @wp.kernel(module="unique", enable_backward=True)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    contact_frame_in: wp.array[wp.mat33],  # [grad] -> res_contact_frame
    contact_friction_in: wp.array[vec5],
    contact_solref_in: wp.array[wp.vec2],
    contact_solreffriction_in: wp.array[wp.vec2],
    contact_solimp_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    efc_pos_in: wp.array2d[float],  # [grad] -> res_efc_pos (penetration)
    efc_margin_in: wp.array2d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    nacon_in: wp.array[int],
    # In:
    V_in: wp.array[wp.spatial_vector],  # [grad]
    A_in: wp.array[wp.spatial_vector],  # [grad] (qacc frozen, but A feeds the qacc->cdof scatter path)
    Z_in: wp.array[wp.spatial_vector],  # [grad] (= Jlam; Zbar = -F carries the direct -J^Tf projection)
    efc_pos_ref_in: wp.array2d[float],  # FROZEN efc_pos (no adjoint): the D-recovery reference (pos0/imp0)
    # Out:
    phi_out: wp.array[float],
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
    imp0 = constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos0)[2]
    pos = efc_pos_in[w, e0] - efc_margin_in[w, e0]  # DIFFERENTIABLE penetration (-> res_efc_pos)
    kbimp = constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos)
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
      b_t = constraint._contact_kbimp(opt_disableflags, dt, ref_t, solimp, pos)[1]
      mu = fric[0] * imp_isq

      if (condim > 1) and (st == _CONE):  # middle zone: cone-coupled forces, shared N/T
        N = _row_jaref(_proj_row_spatial(A, fm, 0), _proj_row_spatial(V, fm, 0), k, b, imp, pos) * mu
        TT = float(0.0)
        for j in range(1, _MAXCONDIM):
          if j < condim:
            uj = _row_jaref(_proj_row_spatial(A, fm, j), _proj_row_spatial(V, fm, j), 0.0, b_t, 0.0, 0.0) * _friction(
              fric, j - 1
            )
            TT += uj * uj
        T = wp.sqrt(wp.max(TT, _MINVAL * _MINVAL))
        fn = solver._eval_elliptic_middle(N, T, D0, mu, 0.0, True)[0]
        phi += -fn * _proj_row_spatial(Z, fm, 0)
        for j in range(1, _MAXCONDIM):
          if j < condim:
            frij = _friction(fric, j - 1)
            uj = _row_jaref(_proj_row_spatial(A, fm, j), _proj_row_spatial(V, fm, j), 0.0, b_t, 0.0, 0.0)
            fj = solver._eval_elliptic_middle(N, T, D0, mu, uj * frij * frij, False)[0]
            phi += -fj * _proj_row_spatial(Z, fm, j)
      else:  # bottom zone / frictionless: each row force = -D_row * Jaref_row
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
def _fill_ones(out: wp.array[float]):
  """Seed adj_phi = +1 per contact (phi = lam^T r_c already folds in the IFT seed lam)."""
  out[wp.tid()] = 1.0


@wp.kernel(enable_backward=False)
def _contact_scatter(
  # Model:
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  dof_parentid: wp.array[int],
  geom_bodyid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  contact_pos_in: wp.array[wp.vec3],
  contact_geom_in: wp.array[wp.vec2i],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_state_in: wp.array2d[int],
  nacon_in: wp.array[int],
  # In:
  lam_in: wp.array2d[float],
  adjV_in: wp.array[wp.spatial_vector],
  adjA_in: wp.array[wp.spatial_vector],
  adjZ_in: wp.array[wp.spatial_vector],
  res_qvel: wp.array2d[float],
  res_cdof: wp.array2d[wp.spatial_vector],
  res_subtree_com: wp.array2d[wp.vec3],
  # Out:
  res_contact_pos_out: wp.array[wp.vec3],
):
  """SCATTER (manual, sparse): route leaf adjoints over the gather's symmetric-difference walk.

  qacc is the implicit root (no adj_qacc), but its Abar term still feeds cdof/com via cdof(qpos).
  """
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
    wp.atomic_add(res_qvel[w], i, wp.dot(h, Vb))  # dphi/dqvel_i = h_i*Vbar
    G = side * (qvel_in[w, i] * Vb + qacc_in[w, i] * Ab + lam_in[w, i] * Zb)  # cotangent on raw jac column
    Ga = wp.spatial_top(G)
    Gl = wp.spatial_bottom(G)
    wp.atomic_add(res_cdof[w], i, wp.spatial_vector(Ga + wp.cross(off, Gl), Gl))
    cpos_acc += wp.cross(Gl, a)  # d(off=p-com)/dp
    wp.atomic_add(res_subtree_com[w], body_rootid[bb], wp.cross(a, Gl))  # doff/dcom = -doff/dp
  res_contact_pos_out[cid] = cpos_acc


@wp.kernel(enable_backward=False)
def _contact_friction_geom_vjp(
  # Model:
  geom_priority: wp.array[int],
  geom_friction: wp.array2d[wp.vec3],
  # Data in:
  contact_geom_in: wp.array[wp.vec2i],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_state_in: wp.array2d[int],
  nacon_in: wp.array[int],
  # In:
  adj_friction_in: wp.array[vec5],  # dphi/dcontact.friction (the leaf's input-adjoint)
  geom_friction_grad: wp.array2d[wp.vec3],
):
  """CONTACT-PARAM sys-id: chain dphi/dcontact.friction to dphi/dgeom_friction (IFT minus).

  Mirrors collision_core.contact_params' priority/max routing; explicit-pair grads stay 0.
  """
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  w = contact_worldid_in[cid]
  e0 = contact_efc_address_in[cid, 0]
  if e0 < 0 or efc_state_in[w, e0] == _SATISFIED:
    return
  geom = contact_geom_in[cid]
  g1 = geom[0]
  g2 = geom[1]
  if g1 < 0 or g2 < 0:
    return  # flex (negative geom ids) -- no geom_friction
  acf = adj_friction_in[cid]
  adj_mgf = wp.vec3(acf[0] + acf[1], acf[2], acf[3] + acf[4])  # de-duplicate the vec5 -> vec3 layout
  fid = w % geom_friction.shape[0]
  p1 = geom_priority[g1]
  p2 = geom_priority[g2]
  gf1 = geom_friction[fid, g1]
  gf2 = geom_friction[fid, g2]
  ag1 = wp.vec3(0.0, 0.0, 0.0)  # enable_backward=False -> component writes are safe (no adjoint of THIS kernel)
  ag2 = wp.vec3(0.0, 0.0, 0.0)
  for c in range(3):  # static-unrolled; priority is per-geom, the equal-priority max is per-component
    win1 = (p1 > p2) or ((p1 == p2) and (gf1[c] >= gf2[c]))  # wp.max routes to g1 (the >= arg) at ties
    if win1:
      ag1[c] = adj_mgf[c]
    else:
      ag2[c] = adj_mgf[c]
  wp.atomic_add(geom_friction_grad, fid, g1, -ag1)  # -= (IFT minus)
  wp.atomic_add(geom_friction_grad, fid, g2, -ag2)
