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
from mujoco_warp._src import math
from mujoco_warp._src import model_adjoint
from mujoco_warp._src import solver
from mujoco_warp._src.types import MJ_MINMU
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import ConstraintState
from mujoco_warp._src.types import ConstraintType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import vec5
from mujoco_warp._src.types import vec12
from mujoco_warp._src.warp_util import cache_kernel

# adjoint module: backward stays on so AD leaves differentiate through cross-module @wp.funcs
# Adjoint reductions use unordered atomics; deterministic backward execution is deferred.
wp.set_module_options({"enable_backward": True})

_MAXCONDIM = 6  # max valid MuJoCo condim; elliptic friction rows = dimid 1..condim-1
_MAX_PYRAMID_EDGES = 10  # 2*(_MAXCONDIM - 1) pyramidal edges at condim 6


# Non-contact residuals gather J*lambda, differentiate a scalar row contraction, and scatter
# its cotangents in one kernel. Constraint states and solver quantities remain frozen.


@wp.func
def _constraint_contract(
  # In:
  pos: float,
  velocity: float,
  Z: float,
  solref: wp.vec2,
  solimp: vec5,
  frictionloss: float,
  invweight: float,
  timestep: float,
  disableflags: int,
  constraint_type: int,
  state: int,
  aref: float,
  force: float,
  D_ref: float,
  Jaref_ref: float,
):
  """Returns phi = -Z*f for an active non-contact row, anchored to its stored force."""
  if constraint_type == ConstraintType.FRICTION_DOF:
    if state == ConstraintState.LINEARNEG:
      return -Z * frictionloss
    if state == ConstraintState.LINEARPOS:
      return Z * frictionloss
  kbi = constraint._contact_kbimp(disableflags, timestep, solref, solimp, pos, False)
  k = 0.0 if constraint_type == ConstraintType.FRICTION_DOF else kbi[0]
  b, imp = kbi[1], kbi[2]
  g = k * imp * pos + b * velocity
  jaref = Jaref_ref + aref + g
  D = constraint._efc_D(invweight, imp)
  F_state = -D * jaref
  F_state0 = -D_ref * Jaref_ref
  f = force + (F_state - F_state0)
  return -Z * f


@wp.func
def _joint_jacobian_contract(x: float, data: vec12, jacobian_bar: float):
  _, slope = constraint._joint_polynomial(data, x)
  return -jacobian_bar * slope


@wp.func
def _ball_jacobian_contract(delta: wp.vec3, quat: wp.quat, axis_bar: wp.vec3):
  rotated = math.quat_integrate(quat, delta, 1.0)
  axis, _ = constraint._ball_axis_angle(rotated)
  return -wp.dot(axis, axis_bar)


@cache_kernel
def _constraint_residual(is_sparse: bool, param_grad: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    qpos0: wp.array2d[float],
    jnt_type: wp.array[int],
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_solref: wp.array2d[wp.vec2],
    jnt_solimp: wp.array2d[vec5],
    dof_solref: wp.array2d[wp.vec2],
    dof_solimp: wp.array2d[vec5],
    dof_frictionloss: wp.array2d[float],
    dof_invweight0: wp.array2d[float],
    eq_obj1id: wp.array[int],
    eq_obj2id: wp.array[int],
    eq_solref: wp.array2d[wp.vec2],
    eq_solimp: wp.array2d[vec5],
    eq_data: wp.array2d[vec12],
    # Data in:
    nefc_in: wp.array[int],
    qpos_in: wp.array2d[float],
    qvel_in: wp.array2d[float],
    qacc_in: wp.array2d[float],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_pos_in: wp.array2d[float],
    efc_margin_in: wp.array2d[float],
    efc_D_in: wp.array2d[float],
    efc_vel_in: wp.array2d[float],
    efc_aref_in: wp.array2d[float],
    efc_force_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    lam_in: wp.array2d[float],
    ctx_Jaref_in: wp.array2d[float],
    # Out:
    res_qvel_out: wp.array2d[float],
    res_dof_out: wp.array2d[float],
    res_jnt_solref_out: wp.array2d[wp.vec2],
    res_jnt_solimp_out: wp.array2d[vec5],
    res_dof_solref_out: wp.array2d[wp.vec2],
    res_dof_solimp_out: wp.array2d[vec5],
    res_dof_frictionloss_out: wp.array2d[float],
    res_eq_solref_out: wp.array2d[wp.vec2],
    res_eq_solimp_out: wp.array2d[vec5],
  ):
    w, row = wp.tid()
    if row >= nefc_in[w]:
      return
    ty = efc_type_in[w, row]
    if ty != ConstraintType.EQUALITY and ty != ConstraintType.LIMIT_JOINT and ty != ConstraintType.FRICTION_DOF:
      return
    st = efc_state_in[w, row]
    if st == ConstraintState.SATISFIED:
      return

    # Gather the same dense or CSR row in its original order.
    count, start = nv, int(0)
    if wp.static(is_sparse):
      count = efc_J_rownnz_in[w, row]
      start = efc_J_rowadr_in[w, row]
    Z = float(0.0)
    for k in range(count):
      i, jj = k, float(0.0)
      if wp.static(is_sparse):
        sid = start + k
        i = efc_J_colind_in[w, 0, sid]
        jj = efc_J_in[w, 0, sid]
      else:
        jj = efc_J_in[w, row, i]
      Z += jj * lam_in[w, i]

    cid = efc_id_in[w, row]
    model_worldid = w % dof_invweight0.shape[0]
    invweight = float(0.0)
    sr, si, frictionloss = wp.vec2(), vec5(), float(0.0)
    if ty == ConstraintType.FRICTION_DOF:
      invweight = dof_invweight0[model_worldid, cid]
      sr = dof_solref[w % dof_solref.shape[0], cid]
      si = dof_solimp[w % dof_solimp.shape[0], cid]
      frictionloss = dof_frictionloss[w % dof_frictionloss.shape[0], cid]
    elif ty == ConstraintType.LIMIT_JOINT:
      invweight = dof_invweight0[model_worldid, jnt_dofadr[cid]]
      sr = jnt_solref[w % jnt_solref.shape[0], cid]
      si = jnt_solimp[w % jnt_solimp.shape[0], cid]
    else:
      invweight = dof_invweight0[model_worldid, jnt_dofadr[eq_obj1id[cid]]]
      joint2 = eq_obj2id[cid]
      if joint2 >= 0:
        invweight += dof_invweight0[model_worldid, jnt_dofadr[joint2]]
      sr = eq_solref[w % eq_solref.shape[0], cid]
      si = eq_solimp[w % eq_solimp.shape[0], cid]

    derivatives = wp.grad(_constraint_contract)(
      efc_pos_in[w, row] - efc_margin_in[w, row],
      efc_vel_in[w, row],
      Z,
      sr,
      si,
      frictionloss,
      invweight,
      opt_timestep[w % opt_timestep.shape[0]],
      opt_disableflags,
      ty,
      st,
      efc_aref_in[w, row],
      efc_force_in[w, row],
      efc_D_in[w, row],
      ctx_Jaref_in[w, row],
    )
    Pb, Vb, Zb = derivatives[0], derivatives[1], derivatives[2]
    if wp.static(param_grad):
      if ty == ConstraintType.FRICTION_DOF:
        if res_dof_solref_out:
          wp.atomic_add(res_dof_solref_out, w % dof_solref.shape[0], cid, derivatives[3])  # kernel_analyzer: ignore[atomic]
        if res_dof_solimp_out:
          wp.atomic_add(res_dof_solimp_out, w % dof_solimp.shape[0], cid, derivatives[4])  # kernel_analyzer: ignore[atomic]
        if res_dof_frictionloss_out:
          wp.atomic_add(
            res_dof_frictionloss_out, w % dof_frictionloss.shape[0], cid, derivatives[5]
          )  # kernel_analyzer: ignore[atomic]
      elif ty == ConstraintType.LIMIT_JOINT:
        if res_jnt_solref_out:
          wp.atomic_add(res_jnt_solref_out, w % jnt_solref.shape[0], cid, derivatives[3])  # kernel_analyzer: ignore[atomic]
        if res_jnt_solimp_out:
          wp.atomic_add(res_jnt_solimp_out, w % jnt_solimp.shape[0], cid, derivatives[4])  # kernel_analyzer: ignore[atomic]
      else:
        if res_eq_solref_out:
          wp.atomic_add(res_eq_solref_out, w % eq_solref.shape[0], cid, derivatives[3])  # kernel_analyzer: ignore[atomic]
        if res_eq_solimp_out:
          wp.atomic_add(res_eq_solimp_out, w % eq_solimp.shape[0], cid, derivatives[4])  # kernel_analyzer: ignore[atomic]

    # Friction rows have no position dependence. Nonlinear joint Jacobians add a separate term.
    route_p = ty == ConstraintType.LIMIT_JOINT or ty == ConstraintType.EQUALITY
    for k in range(count):
      i, jj = k, float(0.0)
      if wp.static(is_sparse):
        sid = start + k
        i = efc_J_colind_in[w, 0, sid]
        jj = efc_J_in[w, 0, sid]
      else:
        jj = efc_J_in[w, row, i]
      if jj != 0.0:
        wp.atomic_add(res_qvel_out, w, i, jj * Vb)  # kernel_analyzer: ignore[atomic]
        if route_p:
          wp.atomic_add(res_dof_out, w, i, jj * Pb)  # kernel_analyzer: ignore[atomic]
    if not route_p:
      return
    jointid = cid
    if ty == ConstraintType.EQUALITY:
      jointid = eq_obj2id[cid]
      if jointid < 0:
        return
    elif jnt_type[jointid] != JointType.BALL:
      return
    qadr, dadr = jnt_qposadr[jointid], jnt_dofadr[jointid]
    acceleration_bar = efc_D_in[w, row] * Z
    if ty == ConstraintType.EQUALITY:
      jacobian_bar = Zb * lam_in[w, dadr] + Vb * qvel_in[w, dadr] + acceleration_bar * qacc_in[w, dadr]
      x = qpos_in[w, qadr] - qpos0[w % qpos0.shape[0], qadr]
      jacobian_grad = wp.grad(_joint_jacobian_contract)(x, eq_data[w % eq_data.shape[0], cid], jacobian_bar)[0]
      wp.atomic_add(res_dof_out, w, dadr, jacobian_grad)  # kernel_analyzer: ignore[atomic]
    else:
      axis_bar = wp.vec3()
      for i in range(3):
        dof = dadr + i
        axis_bar[i] = Zb * lam_in[w, dof] + Vb * qvel_in[w, dof] + acceleration_bar * qacc_in[w, dof]
      quat = wp.quat(qpos_in[w, qadr], qpos_in[w, qadr + 1], qpos_in[w, qadr + 2], qpos_in[w, qadr + 3])
      rotation_grad = wp.grad(_ball_jacobian_contract)(wp.vec3(), quat, axis_bar)[0]
      for i in range(3):
        wp.atomic_add(res_dof_out, w, dadr + i, rotation_grad[i])  # kernel_analyzer: ignore[atomic]

  return kernel


# Contact residual VJP: gather spatial motion per contact, differentiate the cone
# force leaf, then scatter its cotangents over the same ancestor-dof walk.


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


@wp.func
def _contact_contract(
  # In:
  fm: wp.mat33,
  fric: vec5,
  V: wp.spatial_vector,
  A: wp.spatial_vector,
  Z: wp.spatial_vector,
  pos: float,
  dt: float,
  imp_isq: float,
  invweight: float,
  solref: wp.vec2,
  solreffriction: wp.vec2,
  solimp: vec5,
  disableflags: int,
  condim: int,
  st: int,
  active_edges: int,
  is_elliptic: bool,
):
  """Returns phi = lambda.T @ r_contact = -(J @ lambda).T @ f_contact."""
  if is_elliptic:
    if st == ConstraintState.SATISFIED:
      return 0.0
  kbimp = constraint._contact_kbimp(disableflags, dt, solref, solimp, pos, False)
  k = kbimp[0]
  b = kbimp[1]
  imp = kbimp[2]
  D0 = constraint._efc_D(invweight, imp)
  phi = float(0.0)

  if is_elliptic:
    ref_t = solref
    if solreffriction[0] != 0.0 or solreffriction[1] != 0.0:
      ref_t = solreffriction
    b_t = constraint._contact_kbimp(disableflags, dt, ref_t, solimp, pos, False)[1]
    mu = fric[0] * imp_isq

    if (condim > 1) and (st == ConstraintState.CONE):  # middle zone: cone-coupled forces, shared N/T
      N = _row_jaref(_proj_row_spatial(A, fm, 0), _proj_row_spatial(V, fm, 0), k, b, imp, pos) * mu
      TT = float(0.0)
      for j in range(1, _MAXCONDIM):
        if j < condim:
          uj = _row_jaref(_proj_row_spatial(A, fm, j), _proj_row_spatial(V, fm, j), 0.0, b_t, 0.0, 0.0) * _friction(fric, j - 1)
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
          scale = imp_isq * imp_isq
          if j > 1:
            friction = _friction(fric, j - 1)
            scale *= fric[0] * fric[0] / (friction * friction)
          Dj = D0 / scale
          fj = -Dj * _row_jaref(_proj_row_spatial(A, fm, j), _proj_row_spatial(V, fm, j), 0.0, b_t, 0.0, 0.0)
          phi += -fj * _proj_row_spatial(Z, fm, j)
  else:  # pyramidal: ndim = 2*(condim-1) edges (1 if condim==1); each edge an independent force
    ndim = int(1)
    if condim > 1:
      ndim = 2 * (condim - 1)
      friction = fric[0]
      invweight = invweight + friction * friction * invweight
      invweight = invweight * 2.0 * friction * friction * imp_isq * imp_isq
    D = constraint._efc_D(invweight, imp)
    for e in range(_MAX_PYRAMID_EDGES):
      if e < ndim:
        if (active_edges & (1 << e)) != 0:
          Jqve = _proj_edge_spatial(V, fm, fric, e, condim)
          Jqae = _proj_edge_spatial(A, fm, fric, e, condim)
          fe = -D * _row_jaref(Jqae, Jqve, k, b, imp, pos)
          phi += -fe * _proj_edge_spatial(Z, fm, fric, e, condim)

  return phi


@wp.func
def _contact_material_vjp(
  # Model:
  ngeom: int,
  geom_priority: wp.array[int],
  geom_friction: wp.array2d[wp.vec3],
  pair_friction: wp.array2d[vec5],
  nxn_pairid: wp.array[wp.vec2i],
  # In:
  geom: wp.vec2i,
  w: int,
  friction_bar: vec5,
  geom_friction_grad: bool,
  pair_friction_grad: bool,
  # Out:
  geom_friction_grad_out: wp.array2d[wp.vec3],
  pair_friction_grad_out: wp.array2d[vec5],
):
  """Maps contact friction cotangents to geom or explicit-pair parameter gradients."""
  if geom_friction_grad or pair_friction_grad:
    # Recover the same explicit-pair selection used by the broadphase.
    pairid = nxn_pairid[math.upper_tri_index(ngeom, wp.min(geom[0], geom[1]), wp.max(geom[0], geom[1]))][0]
    if pairid >= 0:
      if pair_friction_grad:
        pair_worldid = w % pair_friction.shape[0]
        friction = pair_friction[pair_worldid, pairid]
        grad = vec5(0.0)
        for i in range(5):
          if friction[i] > MJ_MINMU:
            grad[i] = -friction_bar[i]
        wp.atomic_add(pair_friction_grad_out, pair_worldid, pairid, grad)  # kernel_analyzer: ignore[atomic]
    elif geom_friction_grad:
      grad1, grad2 = model_adjoint.geom_friction_vjp(geom_priority, geom_friction, geom, w, friction_bar)
      model_worldid = w % geom_friction.shape[0]
      wp.atomic_add(geom_friction_grad_out, model_worldid, geom[0], grad1)  # kernel_analyzer: ignore[atomic]
      wp.atomic_add(geom_friction_grad_out, model_worldid, geom[1], grad2)  # kernel_analyzer: ignore[atomic]


@cache_kernel
def _contact_residual(cone_type: int, geom_friction_grad: bool, pair_friction_grad: bool):
  """Gathers, differentiates and scatters one contact with scalar generated adjoints."""
  IS_ELLIPTIC = cone_type == ConeType.ELLIPTIC

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    ngeom: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    opt_impratio_invsqrt: wp.array[float],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_invweight0: wp.array2d[wp.vec2],
    dof_parentid: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_priority: wp.array[int],
    geom_friction: wp.array2d[wp.vec3],
    pair_friction: wp.array2d[vec5],
    nxn_pairid: wp.array[wp.vec2i],
    # Data in:
    qvel_in: wp.array2d[float],
    qacc_in: wp.array2d[float],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array[wp.mat33],
    contact_friction_in: wp.array[vec5],
    contact_solref_in: wp.array[wp.vec2],
    contact_solreffriction_in: wp.array[wp.vec2],
    contact_solimp_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_geom_in: wp.array[wp.vec2i],
    contact_efc_address_in: wp.array2d[int],
    contact_worldid_in: wp.array[int],
    efc_pos_in: wp.array2d[float],
    efc_margin_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    nacon_in: wp.array[int],
    # In:
    lam_in: wp.array2d[float],
    # Out:
    res_qvel_out: wp.array2d[float],
    res_cdof_out: wp.array2d[wp.spatial_vector],
    res_subtree_com_out: wp.array2d[wp.vec3],
    res_contact_pos_out: wp.array[wp.vec3],
    res_contact_frame_out: wp.array[wp.mat33],
    res_efc_pos_out: wp.array2d[float],
    geom_friction_grad_out: wp.array2d[wp.vec3],
    pair_friction_grad_out: wp.array2d[vec5],
  ):
    cid = wp.tid()
    if cid >= nacon_in[0]:
      return
    w = contact_worldid_in[cid]
    e0 = contact_efc_address_in[cid, 0]
    if e0 < 0:
      return
    geom = contact_geom_in[cid]
    if geom[0] < 0 or geom[1] < 0:
      return
    st = efc_state_in[w, e0]
    condim = contact_dim_in[cid]
    active_edges = int(0)
    if wp.static(IS_ELLIPTIC):
      if st == ConstraintState.SATISFIED:
        return
    else:
      ndim = 1 if condim == 1 else 2 * (condim - 1)
      for edge in range(_MAX_PYRAMID_EDGES):
        if edge < ndim:
          row = contact_efc_address_in[cid, edge]
          if row >= 0 and efc_state_in[w, row] != ConstraintState.SATISFIED:
            active_edges = active_edges | (1 << edge)
      if active_edges == 0:
        return

    # Gather spatial velocity (V), acceleration (A), and cotangent (Z) at the contact point.
    b0 = body_weldid[geom_bodyid[geom[0]]]
    b1 = body_weldid[geom_bodyid[geom[1]]]
    p = contact_pos_in[cid]
    V = wp.spatial_vector()
    A = wp.spatial_vector()
    Z = wp.spatial_vector()
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

    # Differentiate phi with Warp's generated adjoint.
    dt = opt_timestep[w % opt_timestep.shape[0]]
    imp_isq = opt_impratio_invsqrt[w % opt_impratio_invsqrt.shape[0]]
    model_worldid = w % body_invweight0.shape[0]
    invweight = (
      body_invweight0[model_worldid, geom_bodyid[geom[0]]][0] + body_invweight0[model_worldid, geom_bodyid[geom[1]]][0]
    )
    derivatives = wp.grad(_contact_contract)(
      contact_frame_in[cid],
      contact_friction_in[cid],
      V,
      A,
      Z,
      efc_pos_in[w, e0] - efc_margin_in[w, e0],
      dt,
      imp_isq,
      invweight,
      contact_solref_in[cid],
      contact_solreffriction_in[cid],
      contact_solimp_in[cid],
      opt_disableflags,
      condim,
      st,
      active_edges,
      wp.static(IS_ELLIPTIC),
    )
    # Only the first six inputs have cotangents consumed by this residual path.
    frame_bar, friction_bar = derivatives[0], derivatives[1]
    Vb, Ab, Zb = derivatives[2], derivatives[3], derivatives[4]
    pos_bar = derivatives[5]
    res_contact_frame_out[cid] += frame_bar
    wp.atomic_add(res_efc_pos_out, w, e0, pos_bar)  # kernel_analyzer: ignore[atomic]
    if wp.static(geom_friction_grad or pair_friction_grad):
      _contact_material_vjp(
        ngeom,
        geom_priority,
        geom_friction,
        pair_friction,
        nxn_pairid,
        geom,
        w,
        friction_bar,
        wp.static(geom_friction_grad),
        wp.static(pair_friction_grad),
        geom_friction_grad_out,
        pair_friction_grad_out,
      )

    # Reload contact geometry to keep register lifetimes short, then scatter over the same DOFs.
    b0 = body_weldid[geom_bodyid[geom[0]]]
    b1 = body_weldid[geom_bodyid[geom[1]]]
    p = contact_pos_in[cid]
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
      # dphi/dqvel_i = h_i*Vbar
      wp.atomic_add(res_qvel_out[w], i, wp.dot(h, Vb))  # kernel_analyzer: ignore[atomic]
      G = side * (qvel_in[w, i] * Vb + qacc_in[w, i] * Ab + lam_in[w, i] * Zb)  # cotangent on raw jac column
      Ga = wp.spatial_top(G)
      Gl = wp.spatial_bottom(G)
      wp.atomic_add(res_cdof_out[w], i, wp.spatial_vector(Ga + wp.cross(off, Gl), Gl))  # kernel_analyzer: ignore[atomic]
      cpos_acc += wp.cross(Gl, a)  # d(off=p-com)/dp
      # doff/dcom = -doff/dp
      wp.atomic_add(res_subtree_com_out[w], body_rootid[bb], wp.cross(a, Gl))  # kernel_analyzer: ignore[atomic]
    res_contact_pos_out[cid] += cpos_acc

  return kernel
