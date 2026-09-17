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

import warp as wp

from mujoco_warp._src import constraint
from mujoco_warp._src import model_adjoint
from mujoco_warp._src import solver
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import ConstraintState
from mujoco_warp._src.types import ConstraintType
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import cache_kernel

# adjoint module: backward stays on so AD leaves differentiate through cross-module @wp.funcs
wp.set_module_options({"enable_backward": True})

_MAXCONDIM = 6  # max valid MuJoCo condim; elliptic friction rows = dimid 1..condim-1
_MAX_PYRAMID_EDGES = 10  # 2*(_MAXCONDIM - 1) pyramidal edges at condim 6


# Non-contact constraint residual VJP (equality / joint-limit / dof-friction rows), orchestrated by
# adjoint._residual_constraint_sparse; contact rows live in the bottom section of this file.
#   gather (manual): Z_e = sum_i J_ei*lam_i + true topology invweight (A/V stay frozen anchors).
#   leaf _constraint_row_phi (loop-free, only AD'd piece): phi_e = -Z*f, f anchored to efc.force.
#   scatter (manual): res_qvel += J*Vbar (all rows); res_dof += J*Pbar (position-bearing only).
# Routing: dense or CSR according to m.is_sparse.


# gather (manual): per active non-contact row, reduce Z = J*lam and the topology invweight;
# mirrors solver._solve_init_jaref_kernel's CSR/dense J iterator (keep in sync)
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
    w, row = wp.tid()
    Z_out[w, row] = 0.0
    invw_out[w, row] = 0.0
    if row >= nefc_in[w]:
      return
    ty = efc_type_in[w, row]
    if ty != ConstraintType.EQUALITY and ty != ConstraintType.LIMIT_JOINT and ty != ConstraintType.FRICTION_DOF:
      return  # contact rows -> _contact_*; TODO(etaoxing): tendon limit/friction (structural gate raises)
    if efc_state_in[w, row] == ConstraintState.SATISFIED:
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
    if ty == ConstraintType.FRICTION_DOF:
      invw_out[w, row] = dof_invweight0[w % dof_invweight0.shape[0], cid]
    elif ty == ConstraintType.LIMIT_JOINT:
      invw_out[w, row] = dof_invweight0[w % dof_invweight0.shape[0], jnt_dofadr[cid]]
    # EQUALITY / other -> invw=0 (leaf uses the frozen-D fallback; solimp grad gated, not exact)

  return kernel


# loop-free row leaf: phi_e = -Z*f, f value-anchored to the stored efc.force
@wp.kernel(enable_backward=True)
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
  efc_pos_in: wp.array2d[float],
  efc_margin_in: wp.array2d[float],
  efc_D_in: wp.array2d[float],
  efc_vel_in: wp.array2d[float],
  efc_aref_in: wp.array2d[float],
  efc_force_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  # In:
  Z_in: wp.array2d[float],  # = J*lam (gather) -> Zbar (= -f; the dJ/dq topology seed, unused for G1)
  invw_in: wp.array2d[float],  # frozen true topology invweight (0 -> frozen-D fallback)
  ctx_Jaref_in: wp.array2d[float],  # frozen J*qacc - aref (jaref value anchor)
  # Out:
  phi_out: wp.array2d[float],
):
  w, row = wp.tid()
  phi_out[w, row] = 0.0  # assign the output before any early return
  if row >= nefc_in[w]:
    return
  ty = efc_type_in[w, row]
  if ty != ConstraintType.EQUALITY and ty != ConstraintType.LIMIT_JOINT and ty != ConstraintType.FRICTION_DOF:
    return
  st = efc_state_in[w, row]
  if st == ConstraintState.SATISFIED:
    return
  cid = efc_id_in[w, row]
  Z = Z_in[w, row]
  if ty == ConstraintType.FRICTION_DOF and st == ConstraintState.LINEARNEG:  # saturated friction: force = +frictionloss
    phi_out[w, row] = -Z * dof_frictionloss[w % dof_frictionloss.shape[0], cid]
    return
  if ty == ConstraintType.FRICTION_DOF and st == ConstraintState.LINEARPOS:  # saturated friction: force = -frictionloss
    phi_out[w, row] = -Z * (-dof_frictionloss[w % dof_frictionloss.shape[0], cid])
    return
  # QUADRATIC (equality / active limit / stuck friction): f = -D*jaref, value-anchored to efc.force.
  dt = opt_timestep[w % opt_timestep.shape[0]]
  P = efc_pos_in[w, row] - efc_margin_in[w, row]
  V = efc_vel_in[w, row]
  if ty == ConstraintType.FRICTION_DOF:
    sr = dof_solref[w % dof_solref.shape[0], cid]
    si = dof_solimp[w % dof_solimp.shape[0], cid]
    kbi = constraint._contact_kbimp(opt_disableflags, dt, sr, si, P)
  elif ty == ConstraintType.EQUALITY:
    sr = eq_solref[w % eq_solref.shape[0], cid]
    si = eq_solimp[w % eq_solimp.shape[0], cid]
    kbi = constraint._contact_kbimp(opt_disableflags, dt, sr, si, P)
  else:  # LIMIT_JOINT (slide/hinge scalar J; ball 3-dof -axis J)
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
    D = constraint._efc_D(invw, imp)  # true-invweight D_live (base == D_ref by construction)
  else:
    D = D_ref  # frozen-D fallback (equality / other; matches the legacy dense kernel)
  F_state = -D * jaref
  F_state0 = -D_ref * Jaref0  # frozen base (== F_state at base)
  f = efc_force_in[w, row] + (F_state - F_state0)  # value anchor: f base == efc.force_ref byte-exact
  phi_out[w, row] = -Z * f


# scatter (manual): re-walk the row's J support; res_qvel_out += J*Vbar for all active rows,
# res_dof_out += J*Pbar only for position-bearing rows (a friction row's P is identically zero)
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
    # Out:
    res_qvel_out: wp.array2d[float],
    res_dof_out: wp.array2d[float],
  ):
    w, row = wp.tid()
    if row >= nefc_in[w]:
      return
    ty = efc_type_in[w, row]
    if ty != ConstraintType.EQUALITY and ty != ConstraintType.LIMIT_JOINT and ty != ConstraintType.FRICTION_DOF:
      return
    if efc_state_in[w, row] == ConstraintState.SATISFIED:
      return
    Vb = adjV_in[w, row]
    Pb = adjP_in[w, row]
    # position-bearing rows; FRICTION_DOF excluded (its P == 0, no qpos route)
    route_p = ty == ConstraintType.LIMIT_JOINT or ty == ConstraintType.EQUALITY
    if wp.static(is_sparse):
      rownnz = efc_J_rownnz_in[w, row]
      rowadr = efc_J_rowadr_in[w, row]
      for kk in range(rownnz):
        sid = rowadr + kk
        i = efc_J_colind_in[w, 0, sid]
        jj = efc_J_in[w, 0, sid]
        wp.atomic_add(res_qvel_out[w], i, jj * Vb)
        if route_p:
          wp.atomic_add(res_dof_out[w], i, jj * Pb)
    else:
      for i in range(nv):
        jj = efc_J_in[w, row, i]
        if jj != 0.0:
          wp.atomic_add(res_qvel_out[w], i, jj * Vb)
          if route_p:
            wp.atomic_add(res_dof_out[w], i, jj * Pb)

  return kernel


# Contact residual VJP: gather spatial motion per contact, differentiate the cone
# force leaf, then scatter its cotangents over the same ancestor-dof walk.


@wp.func
def _contact_D(D0: float, imp0: float, imp: float) -> float:
  invweight = imp0 / (wp.max(D0, MJ_MINVAL) * wp.max(1.0 - imp0, MJ_MINVAL))
  return constraint._efc_D(invweight, imp)


@wp.func
def _frame_axis(frame: wp.mat33, i: int) -> wp.vec3:
  if i == 1:
    return wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])
  if i == 2:
    return wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])
  return wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])


@wp.func
def _friction(friction: vec5, i: int) -> float:
  # Static branches avoid a corrupt Warp adjoint from a runtime vector index.
  if i == 0:
    return friction[0]
  if i == 1:
    return friction[1]
  if i == 2:
    return friction[2]
  if i == 3:
    return friction[3]
  return friction[4]


@wp.func
def _row_jaref(Jqa: float, Jqv: float, k: float, b: float, imp: float, pos: float) -> float:
  return Jqa + k * imp * pos + b * Jqv


@wp.func
def _proj_row_spatial(Vsp: wp.spatial_vector, fm: wp.mat33, dimid: int) -> float:
  if dimid < 3:
    return wp.dot(_frame_axis(fm, dimid), wp.spatial_bottom(Vsp))
  return wp.dot(_frame_axis(fm, dimid - 3), wp.spatial_top(Vsp))


# pyramidal edge e's projection of a contact-point spatial motion (the summed _edge_coef)
@wp.func
def _proj_edge_spatial(Vsp: wp.spatial_vector, fm: wp.mat33, fric: vec5, e: int, condim: int) -> float:
  c = _proj_row_spatial(Vsp, fm, 0)
  if condim > 1:
    dimid2 = e / 2 + 1
    fs = _friction(fric, dimid2 - 1) * (1.0 - 2.0 * float(e % 2))
    c += fs * _proj_row_spatial(Vsp, fm, dimid2)
  return c


# Topology gather: per-contact spatial motions V/A/Z from a symmetric-difference dof walk.
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
  if efc_state_in[w, e0] == ConstraintState.SATISFIED:
    return
  geom = contact_geom_in[cid]
  if geom[0] < 0 or geom[1] < 0:  # flex (negative geom ids): unsupported
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


# loop-free source-AD cone leaf (one thread per contact): phi_c = lam^T r_c = -Z*F(V,A,xi)
@cache_kernel
def _contact_phi(cone_type: int):
  IS_ELLIPTIC = cone_type == ConeType.ELLIPTIC

  @wp.kernel(module="unique", enable_backward=True)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    contact_frame_in: wp.array[wp.mat33],
    contact_friction_in: wp.array[vec5],
    contact_solref_in: wp.array[wp.vec2],
    contact_solreffriction_in: wp.array[wp.vec2],
    contact_solimp_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    efc_pos_in: wp.array2d[float],
    efc_margin_in: wp.array2d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    nacon_in: wp.array[int],
    # In:
    V_in: wp.array[wp.spatial_vector],
    A_in: wp.array[wp.spatial_vector],  # qacc frozen, but A feeds the qacc->cdof scatter path
    Z_in: wp.array[wp.spatial_vector],  # = Jlam; Zbar = -F carries the direct -J^Tf projection
    efc_pos_ref_in: wp.array2d[float],  # frozen efc_pos (no adjoint): the D-recovery reference (pos0/imp0)
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
    if st == ConstraintState.SATISFIED:
      return
    dt = opt_timestep[w % opt_timestep.shape[0]]
    imp_isq = opt_impratio_invsqrt[w % opt_impratio_invsqrt.shape[0]]
    condim = contact_dim_in[cid]
    fm = contact_frame_in[cid]
    fric = contact_friction_in[cid]
    solref = contact_solref_in[cid]
    solimp = contact_solimp_in[cid]
    pos0 = efc_pos_ref_in[w, e0] - efc_margin_in[w, e0]  # frozen penetration (D-recovery reference)
    imp0 = constraint._contact_kbimp(opt_disableflags, dt, solref, solimp, pos0)[2]
    pos = efc_pos_in[w, e0] - efc_margin_in[w, e0]  # differentiable penetration (-> res_efc_pos)
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

      if (condim > 1) and (st == ConstraintState.CONE):  # middle zone: cone-coupled forces, shared N/T
        N = _row_jaref(_proj_row_spatial(A, fm, 0), _proj_row_spatial(V, fm, 0), k, b, imp, pos) * mu
        TT = float(0.0)
        for j in range(1, _MAXCONDIM):
          if j < condim:
            uj = _row_jaref(_proj_row_spatial(A, fm, j), _proj_row_spatial(V, fm, j), 0.0, b_t, 0.0, 0.0) * _friction(
              fric, j - 1
            )
            TT += uj * uj
        T = wp.sqrt(wp.max(TT, MJ_MINVAL * MJ_MINVAL))
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
    else:  # pyramidal: ndim = 2*(condim-1) edges (1 if condim==1); each edge an independent force
      ndim = int(1)
      if condim > 1:
        ndim = 2 * (condim - 1)
      for e in range(_MAX_PYRAMID_EDGES):
        if e < ndim:
          ea = contact_efc_address_in[cid, e]
          if ea >= 0 and efc_state_in[w, ea] != ConstraintState.SATISFIED:
            Jqve = _proj_edge_spatial(V, fm, fric, e, condim)
            Jqae = _proj_edge_spatial(A, fm, fric, e, condim)
            fe = -_contact_D(efc_D_in[w, ea], imp0, imp) * _row_jaref(Jqae, Jqve, k, b, imp, pos)
            phi += -fe * _proj_edge_spatial(Z, fm, fric, e, condim)

    phi_out[cid] = phi

  return kernel


# Topology scatter: route leaf adjoints over the gather's symmetric-difference walk. qacc is the
# implicit root, but its Abar term still feeds cdof/com through cdof(qpos).
@cache_kernel
def _contact_scatter(geom_friction_grad: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    dof_parentid: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_priority: wp.array[int],
    geom_friction: wp.array2d[wp.vec3],
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
    res_contact_friction_in: wp.array[vec5],
    # Out:
    res_qvel_out: wp.array2d[float],
    res_cdof_out: wp.array2d[wp.spatial_vector],
    res_subtree_com_out: wp.array2d[wp.vec3],
    res_contact_pos_out: wp.array[wp.vec3],
    geom_friction_grad_out: wp.array2d[wp.vec3],
  ):
    cid = wp.tid()
    if cid >= nacon_in[0]:
      return
    w = contact_worldid_in[cid]
    e0 = contact_efc_address_in[cid, 0]
    if e0 < 0:
      return
    if efc_state_in[w, e0] == ConstraintState.SATISFIED:
      return
    geom = contact_geom_in[cid]
    if geom[0] < 0 or geom[1] < 0:
      return
    if wp.static(geom_friction_grad):
      model_adjoint.accumulate_geom_friction(
        geom_priority,
        geom_friction,
        geom,
        w,
        res_contact_friction_in[cid],
        geom_friction_grad_out,
      )
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
      wp.atomic_add(res_qvel_out[w], i, wp.dot(h, Vb))  # dphi/dqvel_i = h_i*Vbar
      G = side * (qvel_in[w, i] * Vb + qacc_in[w, i] * Ab + lam_in[w, i] * Zb)  # cotangent on raw jac column
      Ga = wp.spatial_top(G)
      Gl = wp.spatial_bottom(G)
      wp.atomic_add(res_cdof_out[w], i, wp.spatial_vector(Ga + wp.cross(off, Gl), Gl))
      cpos_acc += wp.cross(Gl, a)  # d(off=p-com)/dp
      wp.atomic_add(res_subtree_com_out[w], body_rootid[bb], wp.cross(a, Gl))  # doff/dcom = -doff/dp
    res_contact_pos_out[cid] += cpos_acc

  return kernel
