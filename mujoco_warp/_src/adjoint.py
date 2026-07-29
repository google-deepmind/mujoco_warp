"""custom adjoint definitions for MuJoCo Warp autodifferentiation.

This module centralizes all ``@wp.func_grad`` registrations, the
implicit differentiation adjoint for the constraint solver, and the
smooth constraint adjoint for friction gradient signal.

Import this module via ``grad.py`` dont import it directly
"""

import os
import warnings

import warp as wp

from mujoco_warp._src import math
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_factorize_solve_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_func
from mujoco_warp._src.collision_smooth import compute_k_imp
from mujoco_warp._src.warp_util import cache_kernel

# Custom adjoint kernels read, accumulate, and consume cotangent buffers in
# place. The forward-kernel analyzer models arrays as read-only inputs or
# write-only outputs and cannot represent that VJP contract.
# kernel_analyzer: off

# ---------------------------------------------------------------------------
# Phase 3: efc-level gradient kernels for collision chain
# ---------------------------------------------------------------------------


@wp.kernel
def _efc_J_grad_dense_full_kernel(
  nv: int,
  nefc_in: wp.array[int],
  efc_force_in: wp.array2d[float],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  contact_efc_address_in: wp.array2d[int],
  contact_solref_in: wp.array[wp.vec2],
  contact_solreffriction_in: wp.array[wp.vec2],
  contact_solimp_in: wp.array[types.vec5],
  opt_timestep_in: wp.array[float],
  opt_disableflags: int,
  v_in: wp.array2d[float],
  efc_aref_grad_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  efc_J_grad_out: wp.array3d[float],
):
  worldid, efcid, dofid = wp.tid()
  if efcid >= nefc_in[worldid] or dofid >= nv:
    return
  b = float(0.0)
  if efc_type_in[worldid, efcid] >= int(types.ConstraintType.CONTACT_FRICTIONLESS.value):
    conid = efc_id_in[worldid, efcid]
    solref = contact_solref_in[conid]
    if efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC and efcid != contact_efc_address_in[conid, 0]:
      friction_ref = contact_solreffriction_in[conid]
      if friction_ref[0] != 0.0 or friction_ref[1] != 0.0:
        solref = friction_ref
    solimp = contact_solimp_in[conid]
    timestep = opt_timestep_in[worldid % opt_timestep_in.shape[0]]
    dmax = wp.clamp(solimp[1], types.MJ_MINIMP, types.MJ_MAXIMP)
    timeconst = solref[0]
    if not (opt_disableflags & int(types.DisableBit.REFSAFE.value)):
      timeconst = wp.max(timeconst, 2.0 * timestep)
    b = 2.0 / (dmax * timeconst)
    b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)
  efc_J_grad_out[worldid, efcid, dofid] = v_in[worldid, dofid] * efc_force_in[worldid, efcid] - efc_aref_grad_in[
    worldid, efcid
  ] * (qacc_in[worldid, dofid] + b * qvel_in[worldid, dofid])


@wp.kernel
def _efc_J_grad_sparse_full_kernel(
  nv: int,
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_force_in: wp.array2d[float],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  contact_efc_address_in: wp.array2d[int],
  contact_solref_in: wp.array[wp.vec2],
  contact_solreffriction_in: wp.array[wp.vec2],
  contact_solimp_in: wp.array[types.vec5],
  opt_timestep_in: wp.array[float],
  opt_disableflags: int,
  v_in: wp.array2d[float],
  efc_aref_grad_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  efc_J_grad_out: wp.array3d[float],
):
  worldid, efcid, k = wp.tid()
  if efcid >= nefc_in[worldid] or k >= efc_J_rownnz_in[worldid, efcid]:
    return
  sparseid = efc_J_rowadr_in[worldid, efcid] + k
  dofid = efc_J_colind_in[worldid, 0, sparseid]
  if dofid >= nv:
    return
  b = float(0.0)
  if efc_type_in[worldid, efcid] >= int(types.ConstraintType.CONTACT_FRICTIONLESS.value):
    conid = efc_id_in[worldid, efcid]
    solref = contact_solref_in[conid]
    if efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC and efcid != contact_efc_address_in[conid, 0]:
      friction_ref = contact_solreffriction_in[conid]
      if friction_ref[0] != 0.0 or friction_ref[1] != 0.0:
        solref = friction_ref
    solimp = contact_solimp_in[conid]
    timestep = opt_timestep_in[worldid % opt_timestep_in.shape[0]]
    dmax = wp.clamp(solimp[1], types.MJ_MINIMP, types.MJ_MAXIMP)
    timeconst = solref[0]
    if not (opt_disableflags & int(types.DisableBit.REFSAFE.value)):
      timeconst = wp.max(timeconst, 2.0 * timestep)
    b = 2.0 / (dmax * timeconst)
    b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)
  efc_J_grad_out[worldid, 0, sparseid] = v_in[worldid, dofid] * efc_force_in[worldid, efcid] - efc_aref_grad_in[
    worldid, efcid
  ] * (qacc_in[worldid, dofid] + b * qvel_in[worldid, dofid])


@wp.kernel
def _sparse_J_grad_to_dense_kernel(
  nv: int,
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_type_in: wp.array2d[int],
  sparse_grad_inout: wp.array3d[float],
  dense_grad_out: wp.array3d[float],
):
  worldid, efcid, k = wp.tid()
  if efcid >= nefc_in[worldid] or k >= efc_J_rownnz_in[worldid, efcid]:
    return
  sparseid = efc_J_rowadr_in[worldid, efcid] + k
  dofid = efc_J_colind_in[worldid, 0, sparseid]
  if dofid < nv:
    dense_grad_out[worldid, efcid, dofid] = sparse_grad_inout[worldid, 0, sparseid]
    if efc_type_in[worldid, efcid] >= int(types.ConstraintType.CONTACT_FRICTIONLESS.value):
      sparse_grad_inout[worldid, 0, sparseid] = 0.0


@wp.kernel
def _sparse_J_to_dense_kernel(
  nv: int,
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  sparse_J_in: wp.array3d[float],
  dense_J_out: wp.array3d[float],
):
  worldid, efcid, k = wp.tid()
  if efcid >= nefc_in[worldid] or k >= efc_J_rownnz_in[worldid, efcid]:
    return
  sparseid = efc_J_rowadr_in[worldid, efcid] + k
  dofid = efc_J_colind_in[worldid, 0, sparseid]
  if dofid < nv:
    dense_J_out[worldid, efcid, dofid] = sparse_J_in[worldid, 0, sparseid]


@wp.kernel
def _solver_mass_matrix_adjoint(
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  qacc: wp.array2d[float],
  v: wp.array2d[float],
  M_grad: wp.array2d[float],
):
  """VJP -sym(v qacc.T) for the packed symmetric inertia in H qacc = rhs."""
  worldid, row = wp.tid()
  rowadr = M_rowadr[row]
  rownnz = M_rownnz[row]
  for k in range(rownnz):
    madr = rowadr + k
    col = M_colind[madr]
    grad = -v[worldid, row] * qacc[worldid, col]
    if col != row:
      grad -= v[worldid, col] * qacc[worldid, row]
    wp.atomic_add(M_grad, worldid, madr, grad)


@wp.kernel
def _efc_J_to_geometry_dense_kernel(
  nv: int,
  opt_cone: int,
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  geom_bodyid: wp.array[int],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  contact_pos_in: wp.array[wp.vec3],
  contact_frame_in: wp.array[wp.mat33],
  contact_friction_in: wp.array[types.vec5],
  contact_dim_in: wp.array[int],
  contact_geom_in: wp.array[wp.vec2i],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  nacon_in: wp.array[int],
  efc_J_grad_inout: wp.array3d[float],
  subtree_com_grad_out: wp.array2d[wp.vec3],
  cdof_grad_out: wp.array2d[wp.spatial_vector],
  contact_pos_grad_out: wp.array[wp.vec3],
  contact_frame_grad_out: wp.array[wp.mat33],
):
  """Route a dense contact-J cotangent to geometry without the dynamic assembly loop."""
  conid, dimid, dofid = wp.tid()
  if conid >= nacon_in[0] or dofid >= nv or not (contact_type_in[conid] & 1):
    return
  condim = contact_dim_in[conid]
  is_elliptic = opt_cone == int(types.ConeType.ELLIPTIC.value)
  if is_elliptic:
    if dimid >= condim:
      return
  elif (condim == 1 and dimid > 0) or (condim > 1 and dimid >= 2 * (condim - 1)):
    return
  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return
  worldid = contact_worldid_in[conid]
  gJ = efc_J_grad_inout[worldid, efcid, dofid]
  efc_J_grad_inout[worldid, efcid, dofid] = 0.0
  if gJ == 0.0:
    return

  frame = contact_frame_in[conid]
  gp = wp.vec3(0.0)
  gr = wp.vec3(0.0)
  dimid2 = int(0)
  s = float(0.0)
  if is_elliptic:
    if dimid < 3:
      gp = wp.vec3(frame[dimid, 0], frame[dimid, 1], frame[dimid, 2]) * gJ
    else:
      gr = wp.vec3(frame[dimid - 3, 0], frame[dimid - 3, 1], frame[dimid - 3, 2]) * gJ
  else:
    gp = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2]) * gJ
  if not is_elliptic and condim > 1:
    dimid2 = dimid / 2 + 1
    frii = contact_friction_in[conid][dimid2 - 1]
    s = wp.where(dimid % 2 == 0, 1.0, -1.0) * frii * gJ
    if dimid2 < 3:
      gp += wp.vec3(frame[dimid2, 0], frame[dimid2, 1], frame[dimid2, 2]) * s
    else:
      gr += wp.vec3(frame[dimid2 - 3, 0], frame[dimid2 - 3, 1], frame[dimid2 - 3, 2]) * s

  geom = contact_geom_in[conid]
  body1 = body_weldid[geom_bodyid[geom[0]]]
  body2 = body_weldid[geom_bodyid[geom[1]]]
  point = contact_pos_in[conid]
  point_grad = wp.vec3(0.0)
  jacp1 = wp.vec3(0.0)
  jacp2 = wp.vec3(0.0)
  jacr1 = wp.vec3(0.0)
  jacr2 = wp.vec3(0.0)

  if body_isdofancestor[body2, dofid] != 0:
    root2 = body_rootid[body2]
    off2 = point - subtree_com_in[worldid, root2]
    ang2 = wp.spatial_top(cdof_in[worldid, dofid])
    lin2 = wp.spatial_bottom(cdof_in[worldid, dofid])
    jacp2 = lin2 + wp.cross(ang2, off2)
    jacr2 = ang2
    off_grad2 = wp.cross(gp, ang2)
    cdof_grad2 = wp.spatial_vector(wp.cross(off2, gp) + gr, gp)
    wp.atomic_add(cdof_grad_out, worldid, dofid, cdof_grad2)
    wp.atomic_add(subtree_com_grad_out, worldid, root2, -off_grad2)
    point_grad += off_grad2

  if body_isdofancestor[body1, dofid] != 0:
    root1 = body_rootid[body1]
    off1 = point - subtree_com_in[worldid, root1]
    ang1 = wp.spatial_top(cdof_in[worldid, dofid])
    lin1 = wp.spatial_bottom(cdof_in[worldid, dofid])
    jacp1 = lin1 + wp.cross(ang1, off1)
    jacr1 = ang1
    ngp = -gp
    ngr = -gr
    off_grad1 = wp.cross(ngp, ang1)
    cdof_grad1 = wp.spatial_vector(wp.cross(off1, ngp) + ngr, ngp)
    wp.atomic_add(cdof_grad_out, worldid, dofid, cdof_grad1)
    wp.atomic_add(subtree_com_grad_out, worldid, root1, -off_grad1)
    point_grad += off_grad1

  wp.atomic_add(contact_pos_grad_out, conid, point_grad)
  jacp_dif = jacp2 - jacp1
  jacr_dif = jacr2 - jacr1
  frame_grad = wp.mat33(0.0)
  for xyz in range(3):
    if is_elliptic:
      if dimid < 3:
        frame_grad[dimid, xyz] += gJ * jacp_dif[xyz]
      else:
        frame_grad[dimid - 3, xyz] += gJ * jacr_dif[xyz]
    else:
      frame_grad[0, xyz] += gJ * jacp_dif[xyz]
    if not is_elliptic and condim > 1:
      if dimid2 < 3:
        frame_grad[dimid2, xyz] += s * jacp_dif[xyz]
      else:
        frame_grad[dimid2 - 3, xyz] += s * jacr_dif[xyz]
  wp.atomic_add(contact_frame_grad_out, conid, frame_grad)


@wp.kernel
def _limit_joint_state_grad_dense_kernel(
  timestep: wp.array[float],
  opt_disableflags: int,
  nefc_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_pos_in: wp.array2d[float],
  efc_D_grad_in: wp.array2d[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_margin: wp.array2d[float],
  jnt_solref: wp.array2d[wp.vec2],
  jnt_solimp: wp.array2d[types.vec5],
  efc_aref_grad_inout: wp.array2d[float],
  qpos_grad_out: wp.array2d[float],
  qvel_grad_out: wp.array2d[float],
):
  """Route active slide/hinge limit aref/D cotangents to qpos and qvel."""
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid] or efc_type_in[worldid, efcid] != types.ConstraintType.LIMIT_JOINT:
    return
  jntid = efc_id_in[worldid, efcid]
  joint_type = jnt_type[jntid]
  if joint_type != types.JointType.SLIDE and joint_type != types.JointType.HINGE:
    return
  qposadr = jnt_qposadr[jntid]
  dofadr = jnt_dofadr[jntid]
  J = efc_J_in[worldid, efcid, dofadr]
  if J == 0.0:
    return

  margin_id = worldid % jnt_margin.shape[0]
  solref_id = worldid % jnt_solref.shape[0]
  solimp_id = worldid % jnt_solimp.shape[0]
  pos_val = efc_pos_in[worldid, efcid] - jnt_margin[margin_id, jntid]
  solref = jnt_solref[solref_id, jntid]
  solimp = jnt_solimp[solimp_id, jntid]
  dt = timestep[worldid % timestep.shape[0]]
  k_imp = compute_k_imp(opt_disableflags, solref, solimp, pos_val, dt)
  k = k_imp[0]
  imp = k_imp[1]

  width = wp.max(types.MJ_MINVAL, solimp[2])
  mid = wp.clamp(solimp[3], types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, solimp[4])
  imp_x = wp.abs(pos_val) / width
  dimp_dpos = float(0.0)
  if imp_x < 1.0:
    dy_dx = float(0.0)
    if imp_x < mid:
      dy_dx = power * wp.pow(imp_x, power - 1.0) / wp.pow(mid, power - 1.0)
    else:
      dy_dx = power * wp.pow(1.0 - imp_x, power - 1.0) / wp.pow(1.0 - mid, power - 1.0)
    sign_pos = wp.where(pos_val > 0.0, 1.0, wp.where(pos_val < 0.0, -1.0, 0.0))
    dimp_dpos = (solimp[1] - solimp[0]) * dy_dx * sign_pos / width

  daref_dpos = -k * (imp + pos_val * dimp_dpos)
  D = efc_D_in[worldid, efcid]
  dD_dpos = D * (1.0 / wp.max(imp, types.MJ_MINVAL) + 1.0 / wp.max(1.0 - imp, types.MJ_MINVAL)) * dimp_dpos
  aref_grad = efc_aref_grad_inout[worldid, efcid]
  wp.atomic_add(qpos_grad_out, worldid, qposadr, (aref_grad * daref_dpos + efc_D_grad_in[worldid, efcid] * dD_dpos) * J)

  dmax = wp.clamp(solimp[1], types.MJ_MINIMP, types.MJ_MAXIMP)
  timeconst = solref[0]
  if not (opt_disableflags & types.DisableBit.REFSAFE.value):
    timeconst = wp.max(timeconst, 2.0 * dt)
  b = 2.0 / (dmax * timeconst)
  b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)
  wp.atomic_add(qvel_grad_out, worldid, dofadr, -b * aref_grad * J)
  efc_aref_grad_inout[worldid, efcid] = 0.0


@wp.kernel
def _eq_connect_weld_grad_dense_kernel(
  # Model:
  nv: int,
  nsite: int,
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  body_rootid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  site_bodyid: wp.array[int],
  site_quat: wp.array2d[wp.quat],
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_objtype: wp.array[int],
  eq_solref: wp.array2d[wp.vec2],
  eq_solimp: wp.array2d[types.vec5],
  eq_data: wp.array2d[types.vec11],
  # In:
  njmax_in: int,
  nefc_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_pos_in: wp.array2d[float],
  efc_D_in: wp.array2d[float],
  efc_J_in: wp.array3d[float],
  qvel_in: wp.array2d[float],
  xpos_in: wp.array2d[wp.vec3],
  xmat_in: wp.array2d[wp.mat33],
  xquat_in: wp.array2d[wp.quat],
  site_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  efc_D_grad_in: wp.array2d[float],
  # In/out:
  efc_aref_grad_inout: wp.array2d[float],
  efc_J_grad_inout: wp.array3d[float],
  # Out:
  qvel_grad_out: wp.array2d[float],
  xpos_grad_out: wp.array2d[wp.vec3],
  xmat_grad_out: wp.array2d[wp.mat33],
  xquat_grad_out: wp.array2d[wp.quat],
  site_xpos_grad_out: wp.array2d[wp.vec3],
  subtree_com_grad_out: wp.array2d[wp.vec3],
  cdof_grad_out: wp.array2d[wp.spatial_vector],
):
  """Route connect/weld equality aref/D/J cotangents to the fixed-point state.

  Mirrors _equality_connect/_equality_weld: the row group carries
  aref_i = -k*imp*pos_i - b*(J_i . qvel) - Jdotv_i with one impedance from the group residual
  norm, and J assembled from body Jacobians at the anchor points. The pos/impedance VJP is
  emitted into the anchor frames (xpos/xmat/xquat or site_xpos), the -b*vel VJP into qvel, and
  the J-geometry VJP into cdof/subtree_com and the anchor points. The Jdotv term is treated as
  locally constant (exact at qvel = 0). FLEX/FLEXSTRAIN equality rows have no state VJP; their
  cotangents are explicitly zeroed.
  """
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  if efc_type_in[worldid, efcid] != types.ConstraintType.EQUALITY:
    return
  eqid = efc_id_in[worldid, efcid]
  # Only the first row of each constraint's contiguous row group proceeds.
  if efcid > 0 and efc_type_in[worldid, efcid - 1] == types.ConstraintType.EQUALITY and efc_id_in[worldid, efcid - 1] == eqid:
    return
  etype = eq_type[eqid]
  is_connect = etype == types.EqType.CONNECT
  is_weld = etype == types.EqType.WELD
  if not (is_connect or is_weld):
    if etype == types.EqType.FLEX or etype == types.EqType.FLEXSTRAIN:
      i = int(efcid)
      while i < nefc_in[worldid] and i < njmax_in:
        if efc_type_in[worldid, i] != types.ConstraintType.EQUALITY or efc_id_in[worldid, i] != eqid:
          break
        efc_aref_grad_inout[worldid, i] = 0.0
        for dofid in range(nv):
          efc_J_grad_inout[worldid, i, dofid] = 0.0
        i += 1
    return
  nrows = 3
  if is_weld:
    nrows = 6
  if efcid >= njmax_in - nrows:
    # The assembly kernel skipped writing this overflowing row group.
    return

  data = eq_data[worldid % eq_data.shape[0], eqid]
  solref = eq_solref[worldid % eq_solref.shape[0], eqid]
  solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  pos = types.vec6(0.0)
  adj_aref = types.vec6(0.0)
  pos_sq = float(0.0)
  for i in range(nrows):
    p = efc_pos_in[worldid, efcid + i]
    pos[i] = p
    pos_sq += p * p
    adj_aref[i] = efc_aref_grad_inout[worldid, efcid + i]
  pos_imp = wp.sqrt(pos_sq)

  k_imp = compute_k_imp(opt_disableflags, solref, solimp, pos_imp, timestep)
  k = k_imp[0]
  imp = k_imp[1]

  dmin = wp.clamp(solimp[0], types.MJ_MINIMP, types.MJ_MAXIMP)
  dmax = wp.clamp(solimp[1], types.MJ_MINIMP, types.MJ_MAXIMP)
  width = wp.max(types.MJ_MINVAL, solimp[2])
  mid = wp.clamp(solimp[3], types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, solimp[4])
  imp_x = pos_imp / width
  dimp_dx = float(0.0)
  if imp_x < 1.0:
    dy_dx = float(0.0)
    if imp_x < mid:
      dy_dx = power * wp.pow(imp_x, power - 1.0) / wp.pow(mid, power - 1.0)
    else:
      dy_dx = power * wp.pow(1.0 - imp_x, power - 1.0) / wp.pow(1.0 - mid, power - 1.0)
    dimp_dx = (dmax - dmin) * dy_dx / width

  timeconst = solref[0]
  if not (opt_disableflags & int(types.DisableBit.REFSAFE.value)):
    timeconst = wp.max(timeconst, 2.0 * timestep)
  b = 2.0 / (dmax * timeconst)
  b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)

  # Impedance cotangent pooled over the group: aref_i = -k*imp*pos_i and D = imp/(iw*(1-imp)).
  s_imp = float(0.0)
  for i in range(nrows):
    dD_dimp = efc_D_in[worldid, efcid + i] * (1.0 / wp.max(imp, types.MJ_MINVAL) + 1.0 / wp.max(1.0 - imp, types.MJ_MINVAL))
    s_imp += adj_aref[i] * (-k * pos[i]) + efc_D_grad_in[worldid, efcid + i] * dD_dimp

  adj_pos = types.vec6(0.0)
  inv_pos_imp = 1.0 / wp.max(pos_imp, types.MJ_MINVAL)
  for i in range(nrows):
    adj_pos[i] = -k * imp * adj_aref[i] + s_imp * dimp_dx * pos[i] * inv_pos_imp

  obj1id = eq_obj1id[eqid]
  obj2id = eq_obj2id[eqid]
  is_site = nsite > 0 and eq_objtype[eqid] == types.ObjType.SITE
  anchor1 = wp.vec3(data[0], data[1], data[2])
  anchor2 = wp.vec3(data[3], data[4], data[5])
  body1 = obj1id
  body2 = obj2id
  pos1 = wp.vec3(0.0)
  pos2 = wp.vec3(0.0)
  if is_site:
    body1 = site_bodyid[obj1id]
    body2 = site_bodyid[obj2id]
    pos1 = site_xpos_in[worldid, obj1id]
    pos2 = site_xpos_in[worldid, obj2id]
  elif is_connect:
    pos1 = xpos_in[worldid, body1] + xmat_in[worldid, body1] @ anchor1
    pos2 = xpos_in[worldid, body2] + xmat_in[worldid, body2] @ anchor2
  else:
    # Weld anchors are swapped relative to connect (see _equality_weld).
    pos1 = xpos_in[worldid, body1] + xmat_in[worldid, body1] @ anchor2
    pos2 = xpos_in[worldid, body2] + xmat_in[worldid, body2] @ anchor1
  root1 = body_rootid[body1]
  root2 = body_rootid[body2]

  ts = data[10]
  quat = wp.quat(1.0, 0.0, 0.0, 0.0)
  quat1 = wp.quat(1.0, 0.0, 0.0, 0.0)
  sq1 = wp.quat(1.0, 0.0, 0.0, 0.0)
  sq2 = wp.quat(1.0, 0.0, 0.0, 0.0)
  relpose = wp.quat(data[6], data[7], data[8], data[9])
  if is_weld:
    if is_site:
      sqid = worldid % site_quat.shape[0]
      sq1 = site_quat[sqid, obj1id]
      sq2 = site_quat[sqid, obj2id]
      quat = math.mul_quat(xquat_in[worldid, body1], sq1)
      quat1 = math.quat_inv(math.mul_quat(xquat_in[worldid, body2], sq2))
    else:
      quat = math.mul_quat(xquat_in[worldid, body1], relpose)
      quat1 = math.quat_inv(xquat_in[worldid, body2])

  adj_p1 = wp.vec3(0.0)
  adj_p2 = wp.vec3(0.0)
  adj_quat_t = wp.quat(0.0, 0.0, 0.0, 0.0)
  adj_quat1_t = wp.quat(0.0, 0.0, 0.0, 0.0)

  for dofid in range(nv):
    qveld = qvel_in[worldid, dofid]

    # Velocity dissipation: aref_i carries -b * (J_i . qvel).
    acc = float(0.0)
    for i in range(nrows):
      acc += adj_aref[i] * efc_J_in[worldid, efcid + i, dofid]
    if acc != 0.0:
      wp.atomic_add(qvel_grad_out, worldid, dofid, -b * acc)

    # Full J-row cotangent: implicit-solve term plus -b*qvel_dof from vel = J . qvel.
    gp = wp.vec3(
      efc_J_grad_inout[worldid, efcid + 0, dofid] - b * adj_aref[0] * qveld,
      efc_J_grad_inout[worldid, efcid + 1, dofid] - b * adj_aref[1] * qveld,
      efc_J_grad_inout[worldid, efcid + 2, dofid] - b * adj_aref[2] * qveld,
    )
    gr = wp.vec3(0.0)
    if is_weld:
      gr = wp.vec3(
        efc_J_grad_inout[worldid, efcid + 3, dofid] - b * adj_aref[3] * qveld,
        efc_J_grad_inout[worldid, efcid + 4, dofid] - b * adj_aref[4] * qveld,
        efc_J_grad_inout[worldid, efcid + 5, dofid] - b * adj_aref[5] * qveld,
      )
    for i in range(nrows):
      efc_J_grad_inout[worldid, efcid + i, dofid] = 0.0

    ancestor1 = body_isdofancestor[body1, dofid] != 0
    ancestor2 = body_isdofancestor[body2, dofid] != 0
    if not (ancestor1 or ancestor2):
      continue

    ang = wp.spatial_top(cdof_in[worldid, dofid])
    cdof_g = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))

    # Translational rows: J = jacp1 - jacp2 with unit row directions.
    if ancestor1:
      off1 = pos1 - subtree_com_in[worldid, root1]
      cdof_g += wp.spatial_vector(wp.cross(off1, gp), gp)
      og1 = wp.cross(gp, ang)
      wp.atomic_add(subtree_com_grad_out, worldid, root1, -og1)
      adj_p1 += og1
    if ancestor2:
      ngp = -gp
      off2 = pos2 - subtree_com_in[worldid, root2]
      cdof_g += wp.spatial_vector(wp.cross(off2, ngp), ngp)
      og2 = wp.cross(ngp, ang)
      wp.atomic_add(subtree_com_grad_out, worldid, root2, -og2)
      adj_p2 += og2

    if is_weld:
      # Rotational rows: J_rot = 0.5 * axis(quat1 * (0, (jacr1 - jacr2) * ts) * quat).
      jacr1 = wp.vec3(0.0)
      jacr2 = wp.vec3(0.0)
      if ancestor1:
        jacr1 = ang
      if ancestor2:
        jacr2 = ang
      jdr = (jacr1 - jacr2) * ts
      wq = wp.quat(0.0, jdr[0], jdr[1], jdr[2])
      x = math.mul_quat(quat1, wq)
      adj_jq = wp.quat(0.0, 0.5 * gr[0], 0.5 * gr[1], 0.5 * gr[2])
      adj_x = math.mul_quat(adj_jq, math.quat_inv(quat))
      adj_quat_t += math.mul_quat(math.quat_inv(x), adj_jq)
      adj_quat1_t += math.mul_quat(adj_x, math.quat_inv(wq))
      adj_w = math.mul_quat(math.quat_inv(quat1), adj_x)
      adj_jdr = wp.vec3(adj_w[1], adj_w[2], adj_w[3]) * ts
      if ancestor1:
        cdof_g += wp.spatial_vector(adj_jdr, wp.vec3(0.0))
      if ancestor2:
        cdof_g += wp.spatial_vector(-adj_jdr, wp.vec3(0.0))

    wp.atomic_add(cdof_grad_out, worldid, dofid, cdof_g)

  # Residual cotangent: cpos = pos1 - pos2 feeds rows 0..2 and the anchor points.
  a_lin = wp.vec3(adj_pos[0], adj_pos[1], adj_pos[2])
  adj_p1 += a_lin
  adj_p2 -= a_lin
  if is_site:
    wp.atomic_add(site_xpos_grad_out, worldid, obj1id, adj_p1)
    wp.atomic_add(site_xpos_grad_out, worldid, obj2id, adj_p2)
  else:
    a1 = wp.where(is_connect, anchor1, anchor2)
    a2 = wp.where(is_connect, anchor2, anchor1)
    wp.atomic_add(xpos_grad_out, worldid, body1, adj_p1)
    wp.atomic_add(xmat_grad_out, worldid, body1, wp.outer(adj_p1, a1))
    wp.atomic_add(xpos_grad_out, worldid, body2, adj_p2)
    wp.atomic_add(xmat_grad_out, worldid, body2, wp.outer(adj_p2, a2))

  if is_weld:
    # crot = axis(quat1 * quat) * torquescale feeds rows 3..5.
    adj_crotq = wp.quat(0.0, adj_pos[3] * ts, adj_pos[4] * ts, adj_pos[5] * ts)
    adj_quat1_t += math.mul_quat(adj_crotq, math.quat_inv(quat))
    adj_quat_t += math.mul_quat(math.quat_inv(quat1), adj_crotq)
    adj_xq1 = wp.quat(0.0, 0.0, 0.0, 0.0)
    adj_xq2 = wp.quat(0.0, 0.0, 0.0, 0.0)
    if is_site:
      adj_xq1 = math.mul_quat(adj_quat_t, math.quat_inv(sq1))
      adj_xq2 = math.mul_quat(math.quat_inv(adj_quat1_t), math.quat_inv(sq2))
    else:
      adj_xq1 = math.mul_quat(adj_quat_t, math.quat_inv(relpose))
      adj_xq2 = math.quat_inv(adj_quat1_t)
    wp.atomic_add(xquat_grad_out, worldid, body1, adj_xq1)
    wp.atomic_add(xquat_grad_out, worldid, body2, adj_xq2)

  for i in range(nrows):
    efc_aref_grad_inout[worldid, efcid + i] = 0.0


@wp.kernel
def _eq_tendon_grad_dense_kernel(
  # Model:
  nv: int,
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_solref: wp.array2d[wp.vec2],
  eq_solimp: wp.array2d[types.vec5],
  eq_data: wp.array2d[types.vec11],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_length0: wp.array2d[float],
  # In:
  nefc_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_pos_in: wp.array2d[float],
  efc_D_in: wp.array2d[float],
  efc_J_in: wp.array3d[float],
  qvel_in: wp.array2d[float],
  ten_J_in: wp.array2d[float],
  ten_length_in: wp.array2d[float],
  efc_D_grad_in: wp.array2d[float],
  # In/out:
  efc_aref_grad_inout: wp.array2d[float],
  efc_J_grad_inout: wp.array3d[float],
  # Out:
  qvel_grad_out: wp.array2d[float],
  ten_J_grad_out: wp.array2d[float],
  ten_length_grad_out: wp.array2d[float],
):
  """Route tendon equality aref/D/J cotangents to ten_length/ten_J/qvel at the fixed point.

  Mirrors _equality_tendon: pos = (L1 - L1_0) - poly(L2 - L2_0) and J = ten_J_1 - deriv * ten_J_2.
  ten_length/ten_J cotangents flow onward through the taped smooth.tendon backward.
  """
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  if efc_type_in[worldid, efcid] != types.ConstraintType.EQUALITY:
    return
  eqid = efc_id_in[worldid, efcid]
  if eq_type[eqid] != types.EqType.TENDON:
    return

  data = eq_data[worldid % eq_data.shape[0], eqid]
  solref = eq_solref[worldid % eq_solref.shape[0], eqid]
  solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  pos = efc_pos_in[worldid, efcid]
  adj_aref = efc_aref_grad_inout[worldid, efcid]
  adj_D = efc_D_grad_in[worldid, efcid]

  k_imp = compute_k_imp(opt_disableflags, solref, solimp, pos, timestep)
  k = k_imp[0]
  imp = k_imp[1]

  dmin = wp.clamp(solimp[0], types.MJ_MINIMP, types.MJ_MAXIMP)
  dmax = wp.clamp(solimp[1], types.MJ_MINIMP, types.MJ_MAXIMP)
  width = wp.max(types.MJ_MINVAL, solimp[2])
  mid = wp.clamp(solimp[3], types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, solimp[4])
  imp_x = wp.abs(pos) / width
  dimp_dpos = float(0.0)
  if imp_x < 1.0:
    dy_dx = float(0.0)
    if imp_x < mid:
      dy_dx = power * wp.pow(imp_x, power - 1.0) / wp.pow(mid, power - 1.0)
    else:
      dy_dx = power * wp.pow(1.0 - imp_x, power - 1.0) / wp.pow(1.0 - mid, power - 1.0)
    sign_pos = wp.where(pos > 0.0, 1.0, wp.where(pos < 0.0, -1.0, 0.0))
    dimp_dpos = (dmax - dmin) * dy_dx * sign_pos / width

  timeconst = solref[0]
  if not (opt_disableflags & int(types.DisableBit.REFSAFE.value)):
    timeconst = wp.max(timeconst, 2.0 * timestep)
  b = 2.0 / (dmax * timeconst)
  b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)

  dD_dimp = efc_D_in[worldid, efcid] * (1.0 / wp.max(imp, types.MJ_MINVAL) + 1.0 / wp.max(1.0 - imp, types.MJ_MINVAL))
  s_imp = adj_aref * (-k * pos) + adj_D * dD_dimp
  adj_pos = -k * imp * adj_aref + s_imp * dimp_dpos

  obj1id = eq_obj1id[eqid]
  obj2id = eq_obj2id[eqid]
  deriv = float(0.0)
  dif = float(0.0)
  if obj2id > -1:
    dif = ten_length_in[worldid, obj2id] - tendon_length0[worldid % tendon_length0.shape[0], obj2id]
    deriv = data[1] + 2.0 * data[2] * dif + 3.0 * data[3] * dif * dif + 4.0 * data[4] * dif * dif * dif

  # Velocity dissipation over the assembled row.
  if adj_aref != 0.0:
    for dofid in range(nv):
      Jval = efc_J_in[worldid, efcid, dofid]
      if Jval != 0.0:
        wp.atomic_add(qvel_grad_out, worldid, dofid, -b * adj_aref * Jval)

  # J-row cotangent into ten_J (and the polynomial coupling into ten_length via deriv).
  rownnz1 = ten_J_rownnz[obj1id]
  rowadr1 = ten_J_rowadr[obj1id]
  for kk in range(rownnz1):
    sp = rowadr1 + kk
    col = ten_J_colind[sp]
    gJ = efc_J_grad_inout[worldid, efcid, col] - b * adj_aref * qvel_in[worldid, col]
    wp.atomic_add(ten_J_grad_out, worldid, sp, gJ)
  adj_deriv = float(0.0)
  if deriv != 0.0:
    rownnz2 = ten_J_rownnz[obj2id]
    rowadr2 = ten_J_rowadr[obj2id]
    for kk in range(rownnz2):
      sp = rowadr2 + kk
      col = ten_J_colind[sp]
      gJ = efc_J_grad_inout[worldid, efcid, col] - b * adj_aref * qvel_in[worldid, col]
      wp.atomic_add(ten_J_grad_out, worldid, sp, -deriv * gJ)
      adj_deriv -= gJ * ten_J_in[worldid, sp]
  for dofid in range(nv):
    efc_J_grad_inout[worldid, efcid, dofid] = 0.0

  # pos = pos1 - poly(dif): route to tendon lengths.
  wp.atomic_add(ten_length_grad_out, worldid, obj1id, adj_pos)
  if obj2id > -1 and deriv != 0.0:
    dderiv = 2.0 * data[2] + 6.0 * data[3] * dif + 12.0 * data[4] * dif * dif
    wp.atomic_add(ten_length_grad_out, worldid, obj2id, -deriv * adj_pos + adj_deriv * dderiv)

  efc_aref_grad_inout[worldid, efcid] = 0.0


@wp.kernel
def _efc_pos_grad_kernel(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  opt_cone: int,
  # Data in:
  contact_dist_in: wp.array[float],
  contact_includemargin_in: wp.array[float],
  contact_solref_in: wp.array[wp.vec2],
  contact_solimp_in: wp.array[types.vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  efc_aref_grad_in: wp.array2d[float],
  efc_D_in: wp.array2d[float],
  efc_D_grad_in: wp.array2d[float],
  # Out:
  efc_pos_grad_out: wp.array2d[float],
  contact_dist_grad_out: wp.array[float],
):
  """Compute adj_efc_pos from adj_efc_aref.

  From efc_aref = -k * imp * pos - b * vel, d(aref)/d(pos) = -k*imp.
  So adj_efc_pos = adj_efc_aref * (-k * imp).
  We iterate over contacts and their first dimension (normal direction).
  """
  conid, dimid = wp.tid()
  if conid >= nacon_in[0]:
    return
  if not (contact_type_in[conid] & 1):  # ContactType.CONSTRAINT
    return

  condim = contact_dim_in[conid]
  is_elliptic = opt_cone == int(types.ConeType.ELLIPTIC.value)
  if is_elliptic:
    if dimid >= condim:
      return
  else:
    if condim == 1 and dimid > 0:
      return
    if condim > 1 and dimid >= 2 * (condim - 1):
      return
  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return

  worldid = contact_worldid_in[conid]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  solref = contact_solref_in[conid]
  solimp = contact_solimp_in[conid]
  includemargin = contact_includemargin_in[conid]
  pos_val = contact_dist_in[conid] - includemargin

  k_imp = compute_k_imp(opt_disableflags, solref, solimp, pos_val, timestep)

  k = k_imp[0]
  imp = k_imp[1]
  dmin = wp.clamp(solimp[0], types.MJ_MINIMP, types.MJ_MAXIMP)
  dmax = wp.clamp(solimp[1], types.MJ_MINIMP, types.MJ_MAXIMP)
  width = wp.max(types.MJ_MINVAL, solimp[2])
  mid = wp.clamp(solimp[3], types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, solimp[4])
  imp_x = wp.abs(pos_val) / width
  dimp_dpos = float(0.0)
  if imp_x < 1.0:
    dy_dx = float(0.0)
    if imp_x < mid:
      dy_dx = power * wp.pow(imp_x, power - 1.0) / wp.pow(mid, power - 1.0)
    else:
      dy_dx = power * wp.pow(1.0 - imp_x, power - 1.0) / wp.pow(1.0 - mid, power - 1.0)
    pos_sign = wp.where(pos_val < 0.0, -1.0, wp.where(pos_val > 0.0, 1.0, 0.0))
    dimp_dpos = (dmax - dmin) * dy_dx * pos_sign / width

  daref_dpos = -k * (imp + pos_val * dimp_dpos)
  if is_elliptic and dimid > 0:
    daref_dpos = 0.0
  D = efc_D_in[worldid, efcid]
  dD_dpos = D * (1.0 / wp.max(imp, types.MJ_MINVAL) + 1.0 / wp.max(1.0 - imp, types.MJ_MINVAL)) * dimp_dpos

  adj_aref = efc_aref_grad_in[worldid, efcid]
  pos_grad = adj_aref * daref_dpos + efc_D_grad_in[worldid, efcid] * dD_dpos
  if is_elliptic:
    wp.atomic_add(contact_dist_grad_out, conid, pos_grad)
  else:
    efc_pos_grad_out[worldid, efcid] = pos_grad


# kernel_analyzer: on

# ---------------------------------------------------------------------------
# Smooth constraint adjoint: friction Hessian correction kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _smooth_hessian_friction_correction(
  # Model:
  nv: int,
  # Data in:
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  nacon_in: wp.array[int],
  # In:
  friction_viscosity: float,
  friction_scale: float,
  # Out:
  H_out: wp.array3d[float],
):
  """Apply friction smoothing correction to the Hessian.

  For each friction constraint row (dimid > 0):
    - QUADRATIC (active): delta_D = D * (friction_scale - 1.0)  [reduces stiffness]
    - Otherwise (SATISFIED etc): delta_D = friction_viscosity    [adds viscous term]

  Applies delta_D * J_row^T * J_row to H via atomic_add.
  """
  conid, dimid = wp.tid()

  if conid >= nacon_in[0]:
    return

  # Only process constraint contacts
  if not (contact_type_in[conid] & 1):  # ContactType.CONSTRAINT = 1
    return

  # Skip normal direction (dimid=0) — only modify friction rows
  if dimid == 0:
    return

  condim = contact_dim_in[conid]
  if condim == 1:
    return  # frictionless contact, no friction rows
  if dimid >= 2 * (condim - 1):
    return  # beyond valid friction dimensions

  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return

  worldid = contact_worldid_in[conid]

  D = efc_D_in[worldid, efcid]
  state = efc_state_in[worldid, efcid]

  # Compute delta_D: difference between smooth D and what's currently in H
  # QUADRATIC state (value=1): constraint was active, D is in H → reduce it
  # SATISFIED state (value=0): constraint was inactive, 0 in H → add viscous
  delta_D = float(0.0)
  if state == 1:  # QUADRATIC
    delta_D = D * (friction_scale - 1.0)
  else:
    delta_D = friction_viscosity

  if delta_D == 0.0:
    return

  # Apply delta_D * J_row^T * J_row to H
  for i in range(nv):
    Ji = efc_J_in[worldid, efcid, i]
    if Ji == 0.0:
      continue
    for j in range(nv):
      Jj = efc_J_in[worldid, efcid, j]
      if Jj == 0.0:
        continue
      wp.atomic_add(H_out, worldid, i, j, delta_D * Ji * Jj)


# ---------------------------------------------------------------------------
# Smooth constraint adjoint: friction gradient bypass kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _friction_bypass_correction(
  # Model:
  nv: int,
  # Data in:
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  nacon_in: wp.array[int],
  # In:
  v_hessian_in: wp.array2d[float],
  v_free_in: wp.array2d[float],
  bypass_kf: float,
  # Out:
  v_out: wp.array2d[float],
):
  """Friction gradient bypass: restore tangential gradients attenuated by H^{-1}.

  For each friction constraint face (dimid > 0), computes:
    delta = J_fric . (v_free - v_hessian)   [gradient lost to friction attenuation]
    v_out += kf * J_fric^T * delta            [inject it back, scaled by kf]

  v_hessian = H^{-1} * adj_qacc  (attenuated in friction directions)
  v_free    = M^{-1} * adj_qacc  (what gradient would be without constraints)

  This makes the backward pass produce dflex-like friction gradients while
  keeping the forward physics unchanged.
  """
  conid, dimid = wp.tid()

  if conid >= nacon_in[0]:
    return

  # Only process constraint contacts
  if not (contact_type_in[conid] & 1):  # ContactType.CONSTRAINT = 1
    return

  # Skip normal direction (dimid=0) — only bypass friction rows
  if dimid == 0:
    return

  condim = contact_dim_in[conid]
  if condim == 1:
    return  # frictionless contact, no friction rows
  if dimid >= 2 * (condim - 1):
    return  # beyond valid friction dimensions

  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return

  worldid = contact_worldid_in[conid]

  # Compute delta = J_fric . (v_free - v_hessian) for this friction face
  delta = float(0.0)
  for dofid in range(nv):
    J_val = efc_J_in[worldid, efcid, dofid]
    if J_val != 0.0:
      delta += J_val * (v_free_in[worldid, dofid] - v_hessian_in[worldid, dofid])

  # Apply correction: v_out += kf * J_fric^T * delta
  if delta != 0.0:
    scaled_delta = bypass_kf * delta
    for dofid in range(nv):
      J_val = efc_J_in[worldid, efcid, dofid]
      if J_val != 0.0:
        wp.atomic_add(v_out, worldid, dofid, scaled_delta * J_val)


@wp.kernel
def _friction_bypass_correction_normalized(
  # Model:
  nv: int,
  # Data in:
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  nacon_in: wp.array[int],
  # In:
  v_hessian_in: wp.array2d[float],
  v_free_in: wp.array2d[float],
  bypass_kf: float,
  max_ratio: float,
  norm_eps: float,
  # Out:
  v_out: wp.array2d[float],
):
  """Normalized and capped friction bypass correction.

  Projects the free-body delta onto each friction row and injects only a
  bounded fraction of that projected component.

  Compared to _friction_bypass_correction this avoids scaling by ||J_row||^2
  and prevents over-injection when contact rows become poorly conditioned.
  """
  conid, dimid = wp.tid()

  if conid >= nacon_in[0]:
    return

  # Only process constraint contacts
  if not (contact_type_in[conid] & 1):  # ContactType.CONSTRAINT = 1
    return

  # Skip normal direction (dimid=0) - only bypass friction rows
  if dimid == 0:
    return

  condim = contact_dim_in[conid]
  if condim == 1:
    return
  if dimid >= 2 * (condim - 1):
    return

  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return

  worldid = contact_worldid_in[conid]

  delta = float(0.0)
  j_norm2 = float(0.0)
  for dofid in range(nv):
    J_val = efc_J_in[worldid, efcid, dofid]
    if J_val != 0.0:
      delta += J_val * (v_free_in[worldid, dofid] - v_hessian_in[worldid, dofid])
      j_norm2 += J_val * J_val

  if j_norm2 <= norm_eps:
    return

  # Row-normalized projection coefficient.
  base_coeff = delta / j_norm2
  coeff = bypass_kf * base_coeff

  # Bound injected magnitude relative to the projected free-body component.
  max_coeff = wp.abs(base_coeff) * max_ratio
  abs_coeff = wp.abs(coeff)
  if abs_coeff > max_coeff and abs_coeff > 0.0:
    coeff = coeff * (max_coeff / abs_coeff)

  if coeff == 0.0:
    return

  for dofid in range(nv):
    J_val = efc_J_in[worldid, efcid, dofid]
    if J_val != 0.0:
      wp.atomic_add(v_out, worldid, dofid, coeff * J_val)


# Penalty-model adjoint: friction damping kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _penalty_friction_damping(
  # Model:
  nv: int,
  # Data in:
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  nacon_in: wp.array[int],
  # In:
  v_free_in: wp.array2d[float],
  damping_alpha: float,
  # Out:
  v_out: wp.array2d[float],
):
  """Apply penalty-model friction damping to the free-body adjoint.

  For each friction face: v_out -= alpha * J_fric^T * (J_fric . v_free)

  This attenuates v in friction directions by factor (1 - alpha), mimicking
  dflex's penalty friction gradient where d(v_next)/d(v_prev) has eigenvalues
  < 1 in friction-constrained directions.  Provides natural BPTT decay that
  prevents gradient explosion while preserving gradient direction.
  """
  conid, dimid = wp.tid()

  if conid >= nacon_in[0]:
    return

  if not (contact_type_in[conid] & 1):
    return

  # Friction rows only (dimid > 0)
  if dimid == 0:
    return

  condim = contact_dim_in[conid]
  if condim == 1:
    return
  if dimid >= 2 * (condim - 1):
    return

  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return

  worldid = contact_worldid_in[conid]

  # Project v_free onto this friction face
  proj = float(0.0)
  for dofid in range(nv):
    J_val = efc_J_in[worldid, efcid, dofid]
    if J_val != 0.0:
      proj += J_val * v_free_in[worldid, dofid]

  # Subtract friction damping: v_out -= alpha * J^T * proj
  if proj != 0.0:
    scaled = damping_alpha * proj
    for dofid in range(nv):
      J_val = efc_J_in[worldid, efcid, dofid]
      if J_val != 0.0:
        wp.atomic_add(v_out, worldid, dofid, -scaled * J_val)


@wp.func_grad(math.quat_integrate)
def _quat_integrate_grad(q: wp.quat, v: wp.vec3, dt: float, adj_ret: wp.quat):
  """Custom adjoint avoiding gradient singularity at |v|=0."""
  EPS = float(1e-10)
  norm_v = wp.length(v)
  norm_v_sq = norm_v * norm_v
  half_angle = dt * norm_v * 0.5

  # sinc-safe rotation quaternion construction
  if norm_v > EPS:
    s_over_nv = wp.sin(half_angle) / norm_v  # sin(dt|v|/2) / |v|
    c = wp.cos(half_angle)
    # d(s_over_nv)/dv_j = ds_coeff * v_j
    ds_coeff = (c * dt * 0.5 - s_over_nv) / norm_v_sq
  else:
    s_over_nv = dt * 0.5
    c = 1.0
    # Taylor limit: (c*dt/2 - s_over_nv) / |v|^2 -> -dt^3/24
    ds_coeff = -dt * dt * dt / 24.0

  q_rot = wp.quat(
    c,
    s_over_nv * v[0],
    s_over_nv * v[1],
    s_over_nv * v[2],
  )

  # recompute forward intermediates
  q_len = wp.length(q)
  q_inv_len = 1.0 / wp.max(q_len, EPS)
  q_n = wp.quat(
    q[0] * q_inv_len,
    q[1] * q_inv_len,
    q[2] * q_inv_len,
    q[3] * q_inv_len,
  )

  q_res = math.mul_quat(q_n, q_rot)
  res_len = wp.length(q_res)
  res_inv = 1.0 / wp.max(res_len, EPS)

  # result = normalize(q_res)
  # adj_q_res_k = adj_ret_k / |q_res| - q_res_k * dot(adj_ret, q_res) / |q_res|^3
  dot_ar = adj_ret[0] * q_res[0] + adj_ret[1] * q_res[1] + adj_ret[2] * q_res[2] + adj_ret[3] * q_res[3]
  res_inv3 = res_inv * res_inv * res_inv
  adj_qr = wp.quat(
    adj_ret[0] * res_inv - q_res[0] * dot_ar * res_inv3,
    adj_ret[1] * res_inv - q_res[1] * dot_ar * res_inv3,
    adj_ret[2] * res_inv - q_res[2] * dot_ar * res_inv3,
    adj_ret[3] * res_inv - q_res[3] * dot_ar * res_inv3,
  )

  # q_res = mul_quat(q_n, q_rot)
  # adj_q_n  = mul_quat(adj_qr, conj(q_rot))
  # adj_q_rot = mul_quat(conj(q_n), adj_qr)
  q_rot_conj = wp.quat(q_rot[0], -q_rot[1], -q_rot[2], -q_rot[3])
  adj_qn = math.mul_quat(adj_qr, q_rot_conj)

  q_n_conj = wp.quat(q_n[0], -q_n[1], -q_n[2], -q_n[3])
  adj_q_rot = math.mul_quat(q_n_conj, adj_qr)

  # q_rot = (c, s_over_nv * v)
  # d(c)/dv_j = -s_over_nv * dt/2 * v_j
  # d(s_over_nv * v_i)/dv_j = ds_coeff * v_j * v_i + s_over_nv * delta_ij
  sv_dot = adj_q_rot[1] * v[0] + adj_q_rot[2] * v[1] + adj_q_rot[3] * v[2]
  common = -s_over_nv * dt * 0.5 * adj_q_rot[0] + ds_coeff * sv_dot
  adj_v_val = wp.vec3(
    common * v[0] + s_over_nv * adj_q_rot[1],
    common * v[1] + s_over_nv * adj_q_rot[2],
    common * v[2] + s_over_nv * adj_q_rot[3],
  )

  # adj_dt from q_rot dependency on dt
  # d(c)/d(dt)            = -sin(half_angle) * norm_v / 2
  # d(s_over_nv * v_i)/dt = (c / 2) * v_i
  adj_dt_val = adj_q_rot[0] * (-wp.sin(half_angle) * norm_v * 0.5)
  adj_dt_val += sv_dot * c * 0.5

  # q_n = normalize(q)
  # adj_q_k = adj_qn_k / |q| - q_k * dot(adj_qn, q) / |q|^3
  dot_aqn = adj_qn[0] * q[0] + adj_qn[1] * q[1] + adj_qn[2] * q[2] + adj_qn[3] * q[3]
  q_inv_len3 = q_inv_len * q_inv_len * q_inv_len
  adj_q_val = wp.quat(
    adj_qn[0] * q_inv_len - q[0] * dot_aqn * q_inv_len3,
    adj_qn[1] * q_inv_len - q[1] * dot_aqn * q_inv_len3,
    adj_qn[2] * q_inv_len - q[2] * dot_aqn * q_inv_len3,
    adj_qn[3] * q_inv_len - q[3] * dot_aqn * q_inv_len3,
  )

  # accumulate adjoints
  wp.adjoint[q] += adj_q_val
  wp.adjoint[v] += adj_v_val
  wp.adjoint[dt] += adj_dt_val


# ---------------------------------------------------------------------------
# Solver implicit differentiation adjoint
# ---------------------------------------------------------------------------

_BLOCK_CHOLESKY_DIM = 32

_eq_flex_warned = False


def _warn_eq_flex_rows():
  """Warn once per process that flex equality rows carry no state cotangent."""
  global _eq_flex_warned
  if not _eq_flex_warned:
    _eq_flex_warned = True
    warnings.warn(
      "equality FLEX/FLEXSTRAIN constraint rows are not differentiated: the state dependence of their "
      "aref/J/impedance is dropped in the solver backward (their contribution is explicitly zeroed), so "
      "gradients through these rows are incomplete."
    )


@wp.kernel
def _copy_grad_kernel(
  # In:
  src: wp.array2d[float],
  # Out:
  dst_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  dst_out[worldid, dofid] = src[worldid, dofid]


@wp.kernel
def _accumulate_grad_kernel(
  # In:
  src: wp.array2d[float],
  # Out:
  dst_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  dst_out[worldid, dofid] = dst_out[worldid, dofid] + src[worldid, dofid]


@cache_kernel
def _adjoint_cholesky_tile(nv: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    H: wp.array3d[float],
    b: wp.array2d[float],
    # Out:
    out: wp.array2d[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(nv)
    H_tile = wp.tile_load(H[worldid], shape=(TILE_SIZE, TILE_SIZE))
    b_tile = wp.tile_load(b[worldid], shape=(TILE_SIZE,))
    L = wp.tile_cholesky(H_tile)
    x = wp.tile_cholesky_solve(L, b_tile)
    wp.tile_store(out[worldid], x)

  return kernel


@cache_kernel
def _adjoint_cholesky_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    hfactor: wp.array3d[float],
    b: wp.array3d[float],
    nv_runtime: int,
    # Out:
    out: wp.array3d[float],
  ):
    worldid = wp.tid()
    wp.static(create_blocked_cholesky_solve_func(tile_size, matrix_size))(
      hfactor[worldid], b[worldid], nv_runtime, out[worldid]
    )

  return kernel


@cache_kernel
def _adjoint_cholesky_full_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    H: wp.array3d[float],
    b: wp.array3d[float],
    nv_runtime: int,
    hfactor_tmp: wp.array3d[float],
    # Out:
    out: wp.array3d[float],
  ):
    worldid = wp.tid()
    # Fused factorize+solve (upstream replaced the separate factorize func);
    # hfactor_tmp receives the factor as a side effect.
    wp.static(create_blocked_cholesky_factorize_solve_func(tile_size, matrix_size))(
      H[worldid], b[worldid], nv_runtime, hfactor_tmp[worldid], out[worldid]
    )

  return kernel


@wp.kernel
def _padding_h_adjoint(
  # Model:
  nv: int,
  # Out:
  H_out: wp.array3d[float],
):
  worldid, elementid = wp.tid()
  dofid = nv + elementid
  H_out[worldid, dofid, dofid] = 1.0


@wp.kernel
def _symmetrize_upper_kernel(
  # Model:
  nv: int,
  # Out:
  H_out: wp.array3d[float],
):
  """Mirror the upper triangle of H into the lower triangle.

  The Newton solver's dense JTDAJ kernels (full tiled build and the
  incremental active-set update) maintain only the UPPER triangle of the
  Hessian; the lower triangle can hold stale values from earlier solves.
  The adjoint Cholesky consumes the full matrix, so without this the
  factorization sees an asymmetric (possibly indefinite) matrix and
  produces NaNs (observed with contact-rich scenes, e.g. multi-leg ant).
  """
  worldid, elementid = wp.tid()
  # Strict lower-triangle element (row > col) via triangular indexing.
  row = int(0)
  rem = elementid
  size = nv - 1
  while rem >= size and size > 0:
    rem -= size
    size -= 1
    row += 1
  col = row + 1 + rem
  H_out[worldid, col, row] = H_out[worldid, row, col]


def _solve_hessian_system(m: types.Model, d: types.Data, b, out, H=None):
  """Solve H * x = b using stored solver Hessian or a provided H.

  Args:
    m: Model.
    d: Data.
    b: Right-hand side vector (nworld, nv_pad).
    out: Solution vector (nworld, nv_pad).
    H: Optional Hessian override. When provided, always factorizes from
       scratch (ignores stored d.solver_hfactor). Used by smooth adjoint.
  """
  use_stored = H is None
  if use_stored:
    H = d.solver_h

  if H.shape[1] < m.nv or H.shape[2] < m.nv:
    raise RuntimeError(
      f"solver adjoint: the retained constraint Hessian was not populated for this step (shape "
      f"{tuple(H.shape)}, need at least ({d.nworld}, {m.nv}, {m.nv})). The recorded backward cannot run; this "
      "happens when the solve bypassed the gradient-retaining path (e.g. the sleep/island compact solver)."
    )

  # The solver only maintains the upper triangle (see kernel docstring).
  # Skip when a stored factor will be used instead of refactorizing.
  will_factorize = m.nv <= _BLOCK_CHOLESKY_DIM or not (use_stored and d.solver_hfactor.shape[1] > 0)
  if will_factorize and m.nv > 1:
    wp.launch(
      _symmetrize_upper_kernel,
      dim=(d.nworld, m.nv * (m.nv - 1) // 2),
      inputs=[m.nv],
      outputs=[H],
    )

  if m.nv <= _BLOCK_CHOLESKY_DIM:
    wp.launch_tiled(
      _adjoint_cholesky_tile(m.nv),
      dim=d.nworld,
      inputs=[H, b],
      outputs=[out],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
  else:
    # The blocked Cholesky kernels operate on nv_pad-sized tiles, so the
    # right-hand side must be nv_pad wide. The incoming adjoint b is only nv
    # wide (e.g. nv=81, nv_pad=96 for a sparse model), so a direct reshape to
    # (nworld, nv_pad, 1) fails the same-total-size check. Copy b into a padded
    # zero buffer first; the trailing padding rows stay zero (they correspond to
    # the padding DOFs handled by _padding_h_adjoint).
    if b.shape[1] != m.nv_pad:
      nv_b = b.shape[1]
      b_pad = wp.zeros((d.nworld, m.nv_pad), dtype=float)
      wp.launch(_copy_grad_kernel, dim=(d.nworld, nv_b), inputs=[b], outputs=[b_pad])
      b = b_pad
    b_3d = b.reshape((d.nworld, m.nv_pad, 1))
    out_3d = out.reshape((d.nworld, m.nv_pad, 1))

    if use_stored and d.solver_hfactor.shape[1] > 0:
      # Solve-only using stored Cholesky factor (original H only).
      # Pass nv_pad (not nv) as the runtime matrix size: the blocked kernels
      # load tiles at block_size-aligned offsets, so a runtime size that is not
      # a multiple of the tile size produces unaligned tile loads and an illegal
      # memory access (CUDA 719). The forward factorization (_padding_h plus
      # _update_gradient_cholesky_blocked) already pads H to nv_pad with an
      # identity block and runs the solve at nv_pad, so the stored factor is
      # nv_pad-wide and the padding rows resolve to zero.
      wp.launch_tiled(
        _adjoint_cholesky_blocked(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
        dim=d.nworld,
        inputs=[d.solver_hfactor, b_3d, m.nv_pad],
        outputs=[out_3d],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )
    else:
      # Full factorize + solve
      if m.nv_pad > m.nv:
        wp.launch(
          _padding_h_adjoint,
          dim=(d.nworld, m.nv_pad - m.nv),
          inputs=[m.nv],
          outputs=[H],
        )
      hfactor_tmp = wp.zeros((d.nworld, m.nv_pad, m.nv_pad), dtype=float)
      # Pass nv_pad (not nv): see the note above on tile alignment. H is padded
      # to nv_pad with an identity block by _padding_h_adjoint, so factorizing
      # and solving the full nv_pad system is exact for the leading nv rows.
      wp.launch_tiled(
        _adjoint_cholesky_full_blocked(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
        dim=d.nworld,
        inputs=[H, b_3d, m.nv_pad, hfactor_tmp],
        outputs=[out_3d],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )


def solver_implicit_adjoint(
  m: types.Model,
  d: types.Data,
  qacc_array=None,
  qacc_smooth_ref=None,
  qfrc_smooth_ref=None,
  cap=None,
  M_ref=None,
  H_ref=None,
):
  """Implicit differentiation adjoint for constraint solver.

  Called during tape backward. Reads qacc_array.grad (set by downstream
  integrator adjoint), solves H*v = adj_qacc, accumulates into
  qacc_smooth_ref.grad += M*v.

  Args:
    m: Model containing static simulation parameters.
    d: Data containing mutable simulation state.
    qacc_array: The array whose .grad contains the incoming adjoint.
                Defaults to d.qacc when called from diff_forward().
                Integrators pass their local qacc array when it differs
                from d.qacc (e.g. euler with implicit damping).
    qacc_smooth_ref: The record-time qacc_smooth array used by the unconstrained fallback.
    qfrc_smooth_ref: The record-time smooth-force array that receives the direct M*v adjoint.
    cap: Record-time constraint/contact snapshot used to reconstruct the fixed-point VJP.
    M_ref: Record-time mass matrix array; its cotangent receives -v*qacc^T.
    H_ref: Retained post-solve Hessian used by the implicit linear solve.
  """
  nv = m.nv
  if nv == 0:
    return

  if m.opt.enableflags & types.EnableBit.SLEEP:
    # The compact sleep solve never populates the retained top-level solver state the recorded
    # adjoint consumes; reading it would produce garbage gradients or device faults.
    raise RuntimeError(
      "differentiable stepping does not support EnableBit.SLEEP: the sleep/island compact solver does not "
      "populate the retained solver state (d.solver_h) the recorded adjoint consumes. Disable the sleep flag "
      "when computing gradients."
    )

  if qacc_array is None:
    qacc_array = d.qacc

  if qacc_smooth_ref is None:
    qacc_smooth_ref = d.qacc_smooth
  if qfrc_smooth_ref is None:
    qfrc_smooth_ref = d.qfrc_smooth
  if M_ref is None:
    M_ref = d.M

  adj_qacc = qacc_array.grad
  if adj_qacc is None:
    return

  debug_level = os.environ.get("MJW_DEBUG_ADJOINT", "0")
  if debug_level in ("1", "2"):
    import numpy as np

    adj_norm = np.linalg.norm(adj_qacc.numpy())
    print(f"[adjoint] |adj_qacc|={adj_norm:.6e}, njmax={d.njmax}")

  if debug_level == "2" and d.njmax > 0:
    import numpy as np

    efc_state_np = d.efc.state.numpy()
    nefc_np = d.nefc.numpy()
    for w in range(min(d.nworld, 1)):
      ne = nefc_np[w]
      n_quad = int(np.sum(efc_state_np[w, :ne] == 1))
      n_sat = int(np.sum(efc_state_np[w, :ne] == 0))
      H_np = d.solver_h.numpy()[w, :nv, :nv]
      H_diag = np.diag(H_np)
      cond_approx = np.max(H_diag) / max(np.min(H_diag[H_diag > 0]), 1e-30)
      print(
        f"[adjoint:diag] world={w} nefc={ne} QUAD={n_quad} SAT={n_sat}"
        f" H_cond~{cond_approx:.1f}"
        f" H_diag=[{np.min(H_diag):.3e}, {np.max(H_diag):.3e}]"
      )

  if d.njmax == 0:
    # Solver was identity (qacc = qacc_smooth), accumulate adjoint through
    wp.launch(
      _accumulate_grad_kernel,
      dim=(d.nworld, nv),
      inputs=[adj_qacc],
      outputs=[qacc_smooth_ref.grad],
    )
    return

  if m.opt.solver != types.SolverType.NEWTON and cap is None and H_ref is None:
    # The fixed-point backward only needs the Hessian factorization at the converged solution.
    # The CG forward retains it via solver._assemble_hessian_at_solution; without that record
    # there is nothing valid to consume.
    raise RuntimeError(
      "solver adjoint: no retained Hessian for the CG constraint solver. CG gradients require the post-solve "
      "Hessian assembly recorded by the gradient-enabled forward; step with mjw.enable_grad(d) under the tape, "
      "or use the Newton solver."
    )

  if cap is not None and cap.get("eq_flex", False):
    _warn_eq_flex_rows()

  # Solve H * v = adj_qacc. Prefer the record-time captured active-set Hessian (correct contact
  # active set at the fixed point); fall back to the stored d.solver_h otherwise.
  v = wp.zeros((d.nworld, m.nv_pad), dtype=float)
  if cap is not None:
    _solve_hessian_system(m, d, adj_qacc, v, H=cap["H"])
  elif H_ref is not None:
    _solve_hessian_system(m, d, adj_qacc, v, H=H_ref)
  else:
    _solve_hessian_system(m, d, adj_qacc, v)

  # Differentiate the fixed-point equation directly:
  #   (M + J.T D J) qacc = qfrc_smooth + J.T D aref.
  # This avoids a fragile handoff through qacc_smooth.grad and the separate
  # mass-solve callback.
  wp.launch(
    _accumulate_grad_kernel,
    dim=(d.nworld, nv),
    inputs=[v],
    outputs=[qfrc_smooth_ref.grad],
  )
  qacc_values = cap["qacc"] if cap is not None and "qacc" in cap else qacc_array
  if M_ref.grad is not None:
    wp.launch(
      _solver_mass_matrix_adjoint,
      dim=(d.nworld, nv),
      inputs=[m.M_rownnz, m.M_rowadr, m.M_colind, qacc_values, v],
      outputs=[M_ref.grad],
    )
    if cap is not None and cap["cdof_ref"].grad is not None and cap["cinert_ref"].grad is not None:
      from mujoco_warp._src import smooth as smooth_module

      crb_grad = wp.zeros_like(cap["crb_ref"])
      wp.launch(
        smooth_module._M_adjoint,
        dim=(d.nworld, nv),
        inputs=[m.dof_bodyid, m.dof_parentid, m.M_rownnz, m.M_rowadr, cap["cdof_ref"], cap["crb_ref"], M_ref.grad],
        outputs=[cap["cdof_ref"].grad, crb_grad],
      )
      wp.launch(
        smooth_module._crb_to_cinert_adjoint,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_parentid, crb_grad],
        outputs=[cap["cinert_ref"].grad],
      )
      M_ref.grad.zero_()

  # Phase 3: efc-level gradients for the collision chain (use the captured snapshot when
  # available so J grad and the aref->vel->qvel path use the fixed-point active set).
  _efc_level_gradients(m, d, v, qacc_array, cap=cap)


def _efc_level_gradients(m: types.Model, d: types.Data, v, qacc_array, cap=None):
  """Compute efc-level gradients for collision chain (shared by both adjoints).

  When cap (a record-time snapshot) is provided, the contact active set / J / D / pos come from
  the fixed point rather than the cleared backward-time buffers.

  Contact-specific terms currently linearize MuJoCo's default pyramidal cone, where the active
  response is row-separable (K=diag(D)). Elliptic cones require their coupled per-contact K block
  for adj_aref=K*(J*v) and are not covered by these contact kernels.
  """
  if d.njmax > 0:
    efc_J = cap["J_ref"] if cap is not None else d.efc.J
    qacc_values = cap["qacc"] if cap is not None and "qacc" in cap else qacc_array
    efc_aref = cap["aref_ref"] if cap is not None else d.efc.aref
    efc_pos = cap["pos_ref"] if cap is not None else d.efc.pos
    # Populate efc.aref.grad = D * (J . v) on the active set (the contact-velocity gradient
    # path: aref carries the Baumgarte -b*vel term). Uses the captured fixed-point J/D/pos.
    if cap is not None and hasattr(efc_aref, "grad") and efc_aref.grad is not None:
      if cap["is_sparse"]:
        wp.launch(
          _efc_aref_grad_kernel,
          dim=(d.nworld, d.njmax),
          inputs=[
            cap["nefc"],
            cap["J_rownnz"],
            cap["J_rowadr"],
            cap["J_colind"],
            cap["J"],
            cap["D"],
            cap["state"],
            v,
          ],
          outputs=[efc_aref.grad],
        )
      else:
        wp.launch(
          _efc_aref_grad_dense_kernel,
          dim=(d.nworld, d.njmax),
          inputs=[m.nv, cap["nefc"], cap["J"], cap["D"], cap["state"], v],
          outputs=[efc_aref.grad],
        )

    efc_D_grad = wp.zeros((d.nworld, d.njmax_pad), dtype=float)
    if (
      cap is not None
      and m.opt.cone == int(types.ConeType.ELLIPTIC.value)
      and hasattr(efc_aref, "grad")
      and efc_aref.grad is not None
    ):
      cone_J = cap["J"]
      if cap["is_sparse"]:
        cone_J = wp.zeros((d.nworld, d.njmax_pad, m.nv_pad), dtype=float)
        wp.launch(
          _sparse_J_to_dense_kernel,
          dim=(d.nworld, d.njmax, m.nv),
          inputs=[m.nv, cap["nefc"], cap["J_rownnz"], cap["J_rowadr"], cap["J_colind"], cap["J"]],
          outputs=[cone_J],
        )
      wp.launch(
        _efc_elliptic_cone_grad_dense_kernel,
        dim=d.naconmax,
        inputs=[
          m.nv,
          cap["nacon"],
          m.opt.impratio_invsqrt,
          cap["contact_worldid"],
          cap["contact_type"],
          cap["contact_dim"],
          cap["contact_friction"],
          cap["contact_efc_address"],
          cap["state"],
          cone_J,
          cap["D"],
          cap["Jaref"],
          cap["force"],
          v,
        ],
        outputs=[efc_aref.grad, efc_D_grad],
      )

    geometry_J_grad = None
    if (
      cap is not None
      and hasattr(efc_J, "grad")
      and efc_J.grad is not None
      and hasattr(efc_aref, "grad")
      and efc_aref.grad is not None
    ):
      if cap["is_sparse"]:
        wp.launch(
          _efc_J_grad_sparse_full_kernel,
          dim=(d.nworld, d.njmax, m.nv),
          inputs=[
            m.nv,
            cap["nefc"],
            cap["J_rownnz"],
            cap["J_rowadr"],
            cap["J_colind"],
            cap["force"],
            cap["efc_type"],
            cap["efc_id"],
            cap["contact_efc_address"],
            cap["contact_solref"],
            cap["contact_solreffriction"],
            cap["contact_solimp"],
            m.opt.timestep,
            m.opt.disableflags,
            v,
            efc_aref.grad,
            qacc_values,
            cap["qvel"],
          ],
          outputs=[efc_J.grad],
        )
        geometry_J_grad = wp.zeros((d.nworld, d.njmax_pad, m.nv_pad), dtype=float)
        wp.launch(
          _sparse_J_grad_to_dense_kernel,
          dim=(d.nworld, d.njmax, m.nv),
          inputs=[m.nv, cap["nefc"], cap["J_rownnz"], cap["J_rowadr"], cap["J_colind"], cap["efc_type"]],
          outputs=[efc_J.grad, geometry_J_grad],
        )
      else:
        wp.launch(
          _efc_J_grad_dense_full_kernel,
          dim=(d.nworld, d.njmax_pad, m.nv_pad),
          inputs=[
            m.nv,
            cap["nefc"],
            cap["force"],
            cap["efc_type"],
            cap["efc_id"],
            cap["contact_efc_address"],
            cap["contact_solref"],
            cap["contact_solreffriction"],
            cap["contact_solimp"],
            m.opt.timestep,
            m.opt.disableflags,
            v,
            efc_aref.grad,
            qacc_values,
            cap["qvel"],
          ],
          outputs=[efc_J.grad],
        )
        geometry_J_grad = efc_J.grad

    if (
      cap is not None
      and geometry_J_grad is not None
      and hasattr(efc_J, "grad")
      and efc_J.grad is not None
      and cap["subtree_com_ref"].grad is not None
      and cap["cdof_ref"].grad is not None
      and cap["contact_pos_ref"].grad is not None
      and cap["contact_frame_ref"].grad is not None
    ):
      wp.launch(
        _efc_J_to_geometry_dense_kernel,
        dim=(d.naconmax, m.nmaxpyramid, m.nv),
        inputs=[
          m.nv,
          m.opt.cone,
          m.body_rootid,
          m.body_weldid,
          m.body_isdofancestor,
          m.geom_bodyid,
          cap["subtree_com_ref"],
          cap["cdof_ref"],
          cap["contact_pos_ref"],
          cap["contact_frame"],
          cap["contact_friction"],
          cap["contact_dim"],
          cap["contact_geom"],
          cap["contact_efc_address"],
          cap["contact_worldid"],
          cap["contact_type"],
          cap["nacon"],
        ],
        outputs=[
          geometry_J_grad,
          cap["subtree_com_ref"].grad,
          cap["cdof_ref"].grad,
          cap["contact_pos_ref"].grad,
          cap["contact_frame_ref"].grad,
        ],
      )

    if cap is not None and efc_D_grad is not None and hasattr(efc_aref, "grad") and efc_aref.grad is not None:
      if cap["is_sparse"]:
        wp.launch(
          _efc_D_grad_sparse_kernel,
          dim=(d.nworld, d.njmax),
          inputs=[
            cap["nefc"],
            cap["J_rownnz"],
            cap["J_rowadr"],
            cap["J_colind"],
            cap["J"],
            cap["D"],
            cap["state"],
            cap["aref"],
            qacc_values,
            efc_aref.grad,
          ],
          outputs=[efc_D_grad],
        )
      else:
        wp.launch(
          _efc_D_grad_dense_kernel,
          dim=(d.nworld, d.njmax),
          inputs=[m.nv, cap["nefc"], cap["J"], cap["D"], cap["state"], cap["aref"], qacc_values, efc_aref.grad],
          outputs=[efc_D_grad],
        )

    if cap is not None and efc_D_grad is not None and cap["D_ref"].grad is not None:
      wp.launch(
        _replay_D_grad_kernel,
        dim=(d.nworld, d.njmax),
        inputs=[cap["nefc"], cap["efc_type"], cap["efc_id"], m.eq_type, m.jnt_type, efc_D_grad],
        outputs=[cap["D_ref"].grad],
      )

    if cap is not None and efc_D_grad is not None and hasattr(efc_aref, "grad") and efc_aref.grad is not None:
      want_limit = cap["qpos_ref"].grad is not None and cap["qvel_ref"].grad is not None
      want_eq_cw = (
        cap["eq_connect_weld"]
        and geometry_J_grad is not None
        and cap["qvel_ref"].grad is not None
        and cap["xpos_ref"].grad is not None
        and cap["xmat_ref"].grad is not None
        and cap["xquat_ref"].grad is not None
        and cap["site_xpos_ref"].grad is not None
        and cap["subtree_com_ref"].grad is not None
        and cap["cdof_ref"].grad is not None
      )
      want_eq_ten = (
        cap["eq_tendon"]
        and geometry_J_grad is not None
        and cap["qvel_ref"].grad is not None
        and cap["ten_J_ref"].grad is not None
        and cap["ten_length_ref"].grad is not None
      )
      dense_J = cap["J"]
      if cap["is_sparse"] and (want_limit or want_eq_cw or want_eq_ten):
        dense_J = wp.zeros((d.nworld, d.njmax_pad, m.nv_pad), dtype=float)
        wp.launch(
          _sparse_J_to_dense_kernel,
          dim=(d.nworld, d.njmax, m.nv),
          inputs=[m.nv, cap["nefc"], cap["J_rownnz"], cap["J_rowadr"], cap["J_colind"], cap["J"]],
          outputs=[dense_J],
        )
      if want_limit:
        wp.launch(
          _limit_joint_state_grad_dense_kernel,
          dim=(d.nworld, d.njmax),
          inputs=[
            m.opt.timestep,
            m.opt.disableflags,
            cap["nefc"],
            cap["efc_type"],
            cap["efc_id"],
            dense_J,
            cap["D"],
            cap["pos"],
            efc_D_grad,
            m.jnt_type,
            m.jnt_qposadr,
            m.jnt_dofadr,
            m.jnt_margin,
            m.jnt_solref,
            m.jnt_solimp,
          ],
          outputs=[efc_aref.grad, cap["qpos_ref"].grad, cap["qvel_ref"].grad],
        )
      if want_eq_cw:
        wp.launch(
          _eq_connect_weld_grad_dense_kernel,
          dim=(d.nworld, d.njmax),
          inputs=[
            m.nv,
            m.nsite,
            m.opt.timestep,
            m.opt.disableflags,
            m.body_rootid,
            m.body_isdofancestor,
            m.site_bodyid,
            m.site_quat,
            m.eq_type,
            m.eq_obj1id,
            m.eq_obj2id,
            m.eq_objtype,
            m.eq_solref,
            m.eq_solimp,
            m.eq_data,
            d.njmax,
            cap["nefc"],
            cap["efc_type"],
            cap["efc_id"],
            cap["pos"],
            cap["D"],
            dense_J,
            cap["qvel"],
            cap["xpos"],
            cap["xmat"],
            cap["xquat"],
            cap["site_xpos"],
            cap["subtree_com_ref"],
            cap["cdof_ref"],
            efc_D_grad,
          ],
          outputs=[
            efc_aref.grad,
            geometry_J_grad,
            cap["qvel_ref"].grad,
            cap["xpos_ref"].grad,
            cap["xmat_ref"].grad,
            cap["xquat_ref"].grad,
            cap["site_xpos_ref"].grad,
            cap["subtree_com_ref"].grad,
            cap["cdof_ref"].grad,
          ],
        )
      if want_eq_ten:
        wp.launch(
          _eq_tendon_grad_dense_kernel,
          dim=(d.nworld, d.njmax),
          inputs=[
            m.nv,
            m.opt.timestep,
            m.opt.disableflags,
            m.eq_type,
            m.eq_obj1id,
            m.eq_obj2id,
            m.eq_solref,
            m.eq_solimp,
            m.eq_data,
            m.ten_J_rownnz,
            m.ten_J_rowadr,
            m.ten_J_colind,
            m.tendon_length0,
            cap["nefc"],
            cap["efc_type"],
            cap["efc_id"],
            cap["pos"],
            cap["D"],
            dense_J,
            cap["qvel"],
            cap["ten_J"],
            cap["ten_length"],
            efc_D_grad,
          ],
          outputs=[
            efc_aref.grad,
            geometry_J_grad,
            cap["qvel_ref"].grad,
            cap["ten_J_ref"].grad,
            cap["ten_length_ref"].grad,
          ],
        )

    if (
      cap is not None
      and efc_D_grad is not None
      and hasattr(efc_aref, "grad")
      and efc_aref.grad is not None
      and hasattr(efc_pos, "grad")
      and efc_pos.grad is not None
      and cap["contact_dist_ref"].grad is not None
    ):
      wp.launch(
        _efc_pos_grad_kernel,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          m.opt.cone,
          cap["contact_dist"],
          cap["contact_includemargin"],
          cap["contact_solref"],
          cap["contact_solimp"],
          cap["contact_dim"],
          cap["contact_efc_address"],
          cap["contact_worldid"],
          cap["contact_type"],
          cap["nacon"],
          efc_aref.grad,
          cap["D"],
          efc_D_grad,
        ],
        outputs=[efc_pos.grad, cap["contact_dist_ref"].grad],
      )

    # Route the contact velocity-dissipation adjoint directly to qvel with a
    # colind-indexed J^T scatter: adj_qvel = -sum_i b_i adj_aref_i J_i.
    # For QUADRATIC rows adj_aref_i = D_i (J_i . v); for elliptic CONE rows
    # adj_aref is the coupled block product C (J v).
    # The sparse native assembly can mis-route stored row positions, and leaving
    # efc.aref.grad live would also duplicate the position VJP already emitted by
    # _efc_pos_grad_kernel.  Therefore both dense and sparse paths handle every
    # active contact row here and consume efc.aref/efc.vel cotangents.
    qvel_ref = cap.get("qvel_ref") if cap is not None else None
    if (
      cap is not None
      and qvel_ref is not None
      and qvel_ref.grad is not None
      and hasattr(efc_aref, "grad")
      and efc_aref.grad is not None
    ):
      efc_vel = cap["vel_ref"]
      efc_vel_grad = efc_vel.grad if (hasattr(efc_vel, "grad") and efc_vel.grad is not None) else efc_aref.grad
      if cap["is_sparse"]:
        wp.launch(
          _qvel_dissipation_kernel,
          dim=(d.nworld, d.njmax),
          inputs=[
            cap["nefc"],
            cap["J_rownnz"],
            cap["J_rowadr"],
            cap["J_colind"],
            cap["J"],
            cap["D"],
            cap["state"],
            cap["efc_type"],
            cap["efc_id"],
            cap["contact_efc_address"],
            cap["contact_solref"],
            cap["contact_solreffriction"],
            cap["contact_solimp"],
            m.dof_solref,
            m.dof_solimp,
            m.tendon_solref_fri,
            m.tendon_solimp_fri,
            m.opt.timestep,
            m.opt.disableflags,
          ],
          outputs=[qvel_ref.grad, efc_aref.grad, efc_vel_grad],
        )
      else:
        wp.launch(
          _qvel_dissipation_dense_kernel,
          dim=(d.nworld, d.njmax),
          inputs=[
            m.nv,
            cap["nefc"],
            cap["J"],
            cap["D"],
            cap["state"],
            cap["efc_type"],
            cap["efc_id"],
            cap["contact_efc_address"],
            cap["contact_solref"],
            cap["contact_solreffriction"],
            cap["contact_solimp"],
            m.dof_solref,
            m.dof_solimp,
            m.tendon_solref_fri,
            m.tendon_solimp_fri,
            m.opt.timestep,
            m.opt.disableflags,
          ],
          outputs=[qvel_ref.grad, efc_aref.grad, efc_vel_grad],
        )


# ---------------------------------------------------------------------------
# Smooth constraint adjoint: backward-only friction gradient smoothing
# ---------------------------------------------------------------------------


def solver_smooth_adjoint(
  m: types.Model,
  d: types.Data,
  qacc_array=None,
  qacc_smooth_ref=None,
):
  """Smooth constraint adjoint for friction gradient signal.

  Like solver_implicit_adjoint, but builds a modified Hessian H_smooth that
  reduces friction constraint stiffness and adds viscous friction for
  SATISFIED constraints. This provides non-zero gradients through the friction
  cone dead zone while keeping the forward physics unchanged.

  Parameters are read from d.smooth_friction_viscosity and
  d.smooth_friction_scale. Enable via d.smooth_adjoint = 1.

  Args:
    m: Model containing static simulation parameters.
    d: Data containing mutable simulation state.
    qacc_array: The array whose .grad contains the incoming adjoint.
    qacc_smooth_ref: The qacc_smooth array whose .grad receives the
                     accumulated adjoint.
  """
  nv = m.nv
  if nv == 0:
    return

  if qacc_array is None:
    qacc_array = d.qacc

  if qacc_smooth_ref is None:
    qacc_smooth_ref = d.qacc_smooth

  adj_qacc = qacc_array.grad
  if adj_qacc is None:
    return

  debug_level = os.environ.get("MJW_DEBUG_ADJOINT", "0")
  if debug_level in ("1", "2"):
    import numpy as np

    adj_norm = np.linalg.norm(adj_qacc.numpy())
    print(f"[smooth_adjoint] |adj_qacc|={adj_norm:.6e}, njmax={d.njmax}")

  if d.njmax == 0:
    wp.launch(
      _accumulate_grad_kernel,
      dim=(d.nworld, nv),
      inputs=[adj_qacc],
      outputs=[qacc_smooth_ref.grad],
    )
    return

  if m.opt.solver != types.SolverType.NEWTON:
    wp.launch(
      _accumulate_grad_kernel,
      dim=(d.nworld, nv),
      inputs=[adj_qacc],
      outputs=[qacc_smooth_ref.grad],
    )
    return

  # Read smooth adjoint parameters from Data
  free_body = getattr(d, "smooth_free_body_adjoint", False)
  penalty_alpha = getattr(d, "smooth_penalty_damping_alpha", 0.0)
  surrogate = getattr(d, "smooth_friction_surrogate_adjoint", False)
  surrogate_alpha = float(getattr(d, "smooth_friction_surrogate_alpha", 0.0))
  if surrogate_alpha < 0.0:
    surrogate_alpha = 0.0
  elif surrogate_alpha > 1.0:
    surrogate_alpha = 1.0

  if surrogate:
    friction_viscosity = getattr(d, "smooth_friction_viscosity", 10.0)
    friction_scale = getattr(d, "smooth_friction_scale", 0.01)

    H_smooth = wp.clone(d.solver_h)

    if d.naconmax > 0:
      wp.launch(
        _smooth_hessian_friction_correction,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.nv,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          d.nacon,
          friction_viscosity,
          friction_scale,
        ],
        outputs=[H_smooth],
      )

    v_hessian = wp.zeros((d.nworld, m.nv_pad), dtype=float)
    _solve_hessian_system(m, d, adj_qacc, v_hessian, H=H_smooth)

    from mujoco_warp._src.smooth import solve_m

    v_free = wp.zeros((d.nworld, m.nv_pad), dtype=float)
    solve_m(m, d, v_free, adj_qacc)

    v = wp.clone(v_hessian)
    if d.naconmax > 0:
      # Recover only a controlled fraction of the tangential free-body signal.
      # alpha=0 keeps the full bypass, alpha=1 leaves the smooth/Newton result.
      correction_scale = 1.0 - surrogate_alpha
      correction_cap_ratio = 1.0
      correction_norm_eps = 1.0e-8
      wp.launch(
        _friction_bypass_correction_normalized,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.nv,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.efc.J,
          d.nacon,
          v_hessian,
          v_free,
          correction_scale,
          correction_cap_ratio,
          correction_norm_eps,
        ],
        outputs=[v],
      )

  elif free_body or penalty_alpha > 0.0:
    # Free-body base: v = M^{-1} * adj_qacc
    # Eliminates H^{-1} attenuation entirely.
    from mujoco_warp._src.smooth import solve_m

    v = wp.zeros((d.nworld, m.nv_pad), dtype=float)
    solve_m(m, d, v, adj_qacc)

    # Penalty-model friction damping: attenuate v in friction directions
    # by factor (1 - alpha) per face, mimicking dflex's penalty friction
    # d(v_next)/d(v_prev) eigenvalues.  Provides natural BPTT decay.
    if penalty_alpha > 0.0 and d.naconmax > 0:
      v_free = wp.clone(v)  # save unmodified for projection
      wp.launch(
        _penalty_friction_damping,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.nv,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.efc.J,
          d.nacon,
          v_free,
          penalty_alpha,
        ],
        outputs=[v],
      )

  else:
    # Original smooth adjoint: H_smooth with friction correction + optional bypass
    friction_viscosity = getattr(d, "smooth_friction_viscosity", 10.0)
    friction_scale = getattr(d, "smooth_friction_scale", 0.01)
    bypass_kf = getattr(d, "smooth_friction_bypass_kf", 0.0)

    # Build H_smooth = d.solver_h + friction correction
    H_smooth = wp.clone(d.solver_h)

    if d.naconmax > 0:
      wp.launch(
        _smooth_hessian_friction_correction,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.nv,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          d.nacon,
          friction_viscosity,
          friction_scale,
        ],
        outputs=[H_smooth],
      )

    if debug_level == "2":
      import numpy as np

      H_np = H_smooth.numpy()[0, :nv, :nv]
      H_orig = d.solver_h.numpy()[0, :nv, :nv]
      diff = H_np - H_orig
      print(
        f"[smooth_adjoint:diag] H_smooth diag="
        f"[{np.min(np.diag(H_np)):.3e}, {np.max(np.diag(H_np)):.3e}]"
        f" |delta_H|_F={np.linalg.norm(diff):.3e}"
      )

    # Solve H_smooth * v = adj_qacc
    v = wp.zeros((d.nworld, m.nv_pad), dtype=float)
    _solve_hessian_system(m, d, adj_qacc, v, H=H_smooth)

    if debug_level == "2":
      import numpy as np

      v_np = v.numpy()[0, :nv]
      print(f"[smooth_adjoint:diag] |v|={np.linalg.norm(v_np):.6e} v={v_np}")

    # Friction gradient bypass: restore tangential gradients attenuated by H^{-1}
    if bypass_kf > 0.0 and d.naconmax > 0:
      from mujoco_warp._src.smooth import solve_m

      v_free = wp.zeros((d.nworld, m.nv_pad), dtype=float)
      solve_m(m, d, v_free, adj_qacc)

      wp.launch(
        _friction_bypass_correction,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.nv,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.efc.J,
          d.nacon,
          v,
          v_free,
          bypass_kf,
        ],
        outputs=[v],
      )

      if debug_level == "2":
        import numpy as np

        v_bypass = v.numpy()[0, :nv]
        print(f"[smooth_adjoint:diag] bypass kf={bypass_kf} |v_after_bypass|={np.linalg.norm(v_bypass):.6e}")

  # adj_qacc_smooth += M * v
  tmp = wp.zeros((d.nworld, m.nv_pad), dtype=float)
  support.mul_m(m, d, tmp, v)
  wp.launch(
    _accumulate_grad_kernel,
    dim=(d.nworld, nv),
    inputs=[tmp],
    outputs=[qacc_smooth_ref.grad],
  )

  # Phase 3: efc-level gradients for collision chain
  _efc_level_gradients(m, d, v, qacc_array)


def _eq_family_flags(m: types.Model, d: types.Data):
  """Host-cached booleans: model has (connect/weld, tendon, flex/flexstrain) equality rows."""
  flags = getattr(d, "_eq_adjoint_flags", None)
  if flags is None:
    if m.neq > 0:
      import numpy as np

      eqt = m.eq_type.numpy()
      flags = (
        bool(np.any((eqt == types.EqType.CONNECT) | (eqt == types.EqType.WELD))),
        bool(np.any(eqt == types.EqType.TENDON)),
        bool(np.any((eqt == types.EqType.FLEX) | (eqt == types.EqType.FLEXSTRAIN))),
      )
    else:
      flags = (False, False, False)
    d._eq_adjoint_flags = flags
  return flags


def capture_contact_adjoint_state(m: types.Model, d: types.Data):
  """Snapshot constraint inputs before solve; post-solve state/H/force replace them later."""
  if not (d.njmax > 0 and m.opt.solver in (types.SolverType.NEWTON, types.SolverType.CG)):
    return None
  eq_connect_weld, eq_tendon, eq_flex = _eq_family_flags(m, d)
  cap = {
    "is_sparse": bool(m.is_sparse),
    "H": None,
    "nefc": wp.clone(d.nefc, requires_grad=False),
    "J": wp.clone(d.efc.J, requires_grad=False),
    "D": wp.clone(d.efc.D, requires_grad=False),
    "aref": wp.clone(d.efc.aref, requires_grad=False),
    "force": wp.clone(d.efc.force, requires_grad=False),
    "qvel": wp.clone(d.qvel, requires_grad=False),
    "qvel_ref": d.qvel,
    "qpos_ref": d.qpos,
    "J_ref": d.efc.J,
    "D_ref": d.efc.D,
    "aref_ref": d.efc.aref,
    "pos_ref": d.efc.pos,
    "vel_ref": d.efc.vel,
    "contact_dist": wp.clone(d.contact.dist, requires_grad=False),
    "contact_dist_ref": d.contact.dist,
    "contact_pos_ref": d.contact.pos,
    "contact_frame_ref": d.contact.frame,
    "contact_frame": wp.clone(d.contact.frame, requires_grad=False),
    "contact_friction": wp.clone(d.contact.friction, requires_grad=False),
    "contact_geom": wp.clone(d.contact.geom, requires_grad=False),
    "subtree_com_ref": d.subtree_com,
    "cdof_ref": d.cdof,
    "crb_ref": d.crb,
    "cinert_ref": d.cinert,
    "contact_includemargin": wp.clone(d.contact.includemargin, requires_grad=False),
    "contact_solref": wp.clone(d.contact.solref, requires_grad=False),
    "contact_solreffriction": wp.clone(d.contact.solreffriction, requires_grad=False),
    "contact_solimp": wp.clone(d.contact.solimp, requires_grad=False),
    "contact_dim": wp.clone(d.contact.dim, requires_grad=False),
    "contact_efc_address": wp.clone(d.contact.efc_address, requires_grad=False),
    "contact_worldid": wp.clone(d.contact.worldid, requires_grad=False),
    "contact_type": wp.clone(d.contact.type, requires_grad=False),
    "nacon": wp.clone(d.nacon, requires_grad=False),
    "pos": wp.clone(d.efc.pos, requires_grad=False),
    "state": wp.clone(d.efc.state, requires_grad=False),
    "efc_id": wp.clone(d.efc.id, requires_grad=False),
    "efc_type": wp.clone(d.efc.type, requires_grad=False),
    "eq_connect_weld": eq_connect_weld,
    "eq_tendon": eq_tendon,
    "eq_flex": eq_flex,
  }
  if m.is_sparse:
    cap["J_rownnz"] = wp.clone(d.efc.J_rownnz, requires_grad=False)
    cap["J_rowadr"] = wp.clone(d.efc.J_rowadr, requires_grad=False)
    cap["J_colind"] = wp.clone(d.efc.J_colind, requires_grad=False)
  if eq_connect_weld:
    # Anchor-frame values at the fixed point (xquat/site_xpos are shared across substeps, so
    # record-time refs alone would be stale by backward time).
    cap["xpos"] = wp.clone(d.xpos, requires_grad=False)
    cap["xpos_ref"] = d.xpos
    cap["xmat"] = wp.clone(d.xmat, requires_grad=False)
    cap["xmat_ref"] = d.xmat
    cap["xquat"] = wp.clone(d.xquat, requires_grad=False)
    cap["xquat_ref"] = d.xquat
    cap["site_xpos"] = wp.clone(d.site_xpos, requires_grad=False)
    cap["site_xpos_ref"] = d.site_xpos
  if eq_tendon:
    cap["ten_J"] = wp.clone(d.ten_J, requires_grad=False)
    cap["ten_J_ref"] = d.ten_J
    cap["ten_length"] = wp.clone(d.ten_length, requires_grad=False)
    cap["ten_length_ref"] = d.ten_length
  return cap


# These kernels consume retained cotangents in place; see the module note.
# kernel_analyzer: off


@wp.kernel
def _efc_aref_grad_kernel(
  nefc_in: wp.array(dtype=int),
  efc_J_rownnz_in: wp.array2d(dtype=int),
  efc_J_rowadr_in: wp.array2d(dtype=int),
  efc_J_colind_in: wp.array3d(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_state_in: wp.array2d(dtype=int),
  v_in: wp.array2d(dtype=float),
  efc_aref_grad_out: wp.array2d(dtype=float),
):
  """adj_aref[i] = D[i] * (J[i,:] . v) on the quadratic active set."""
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  dd = efc_D_in[worldid, efcid]
  if not (dd > 0.0 and efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value):
    efc_aref_grad_out[worldid, efcid] = 0.0
    return
  rownnz = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]
  jv = float(0.0)
  for k in range(rownnz):
    col = efc_J_colind_in[worldid, 0, rowadr + k]
    jv += efc_J_in[worldid, 0, rowadr + k] * v_in[worldid, col]
  efc_aref_grad_out[worldid, efcid] = dd * jv


@wp.kernel
def _efc_aref_grad_dense_kernel(
  nv: int,
  nefc_in: wp.array(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_state_in: wp.array2d(dtype=int),
  v_in: wp.array2d(dtype=float),
  efc_aref_grad_out: wp.array2d(dtype=float),
):
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  dd = efc_D_in[worldid, efcid]
  if not (dd > 0.0 and efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value):
    efc_aref_grad_out[worldid, efcid] = 0.0
    return
  jv = float(0.0)
  for dofid in range(nv):
    jv += efc_J_in[worldid, efcid, dofid] * v_in[worldid, dofid]
  efc_aref_grad_out[worldid, efcid] = dd * jv


@wp.kernel
def _efc_elliptic_cone_grad_dense_kernel(
  nv: int,
  nacon_in: wp.array[int],
  impratio_invsqrt: wp.array[float],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  contact_dim_in: wp.array[int],
  contact_friction_in: wp.array[types.vec5],
  contact_efc_address_in: wp.array2d[int],
  efc_state_in: wp.array2d[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  solver_Jaref_in: wp.array2d[float],
  efc_force_in: wp.array2d[float],
  v_in: wp.array2d[float],
  efc_aref_grad_out: wp.array2d[float],
  efc_D_grad_out: wp.array2d[float],
):
  conid = wp.tid()
  if conid >= nacon_in[0] or not (contact_type_in[conid] & 1):
    return
  worldid = contact_worldid_in[conid]
  efcid0 = contact_efc_address_in[conid, 0]
  if efcid0 < 0 or efc_state_in[worldid, efcid0] != types.ConstraintState.CONE.value:
    return

  dim = contact_dim_in[conid]
  friction = contact_friction_in[conid]
  mu = friction[0] * impratio_invsqrt[worldid % impratio_invsqrt.shape[0]]
  mu2 = mu * mu
  D0 = efc_D_in[worldid, efcid0]
  dm = math.safe_div(D0, mu2 * (1.0 + mu2))
  if dm == 0.0:
    return

  # Reproduce the fixed-point cone coordinates used to build solver_h.
  n = solver_Jaref_in[worldid, efcid0] * mu
  u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)
  jv = types.vec6(0.0)
  tt = float(0.0)
  for i in range(6):
    if i < dim:
      efcid = contact_efc_address_in[conid, i]
      for dofid in range(nv):
        jv[i] += efc_J_in[worldid, efcid, dofid] * v_in[worldid, dofid]
      if i > 0:
        ui = solver_Jaref_in[worldid, efcid] * friction[i - 1]
        u[i] = ui
        tt += ui * ui

  t = wp.max(wp.sqrt(tt), types.MJ_MINVAL)
  ttt = wp.max(t * t * t, types.MJ_MINVAL)

  # C is exactly the per-contact block used by _update_gradient_JTCJ_*.
  for i in range(6):
    if i < dim:
      grad_i = float(0.0)
      ui = u[i]
      for j in range(6):
        if j < dim:
          uj = u[j]
          hcone = float(0.0)
          if i == 0 and j == 0:
            hcone = 1.0
          elif i == 0:
            hcone = -math.safe_div(mu, t) * uj
          elif j == 0:
            hcone = -math.safe_div(mu, t) * ui
          else:
            hcone = mu * math.safe_div(n, ttt) * ui * uj
            if i == j:
              hcone += mu2 - mu * math.safe_div(n, t)

          fri_i = mu
          fri_j = mu
          if i > 0:
            fri_i = friction[i - 1]
          if j > 0:
            fri_j = friction[j - 1]
          Cij = dm * fri_i * fri_j * hcone
          grad_i += Cij * jv[j]
      efcid = contact_efc_address_in[conid, i]
      efc_aref_grad_out[worldid, efcid] = grad_i

  # The middle-zone force is linear in D0 through dm.
  D0_grad = float(0.0)
  for i in range(6):
    if i < dim:
      efcid = contact_efc_address_in[conid, i]
      D0_grad += jv[i] * efc_force_in[worldid, efcid] / D0
  efc_D_grad_out[worldid, efcid0] = D0_grad


@wp.kernel
def _efc_D_grad_dense_kernel(
  nv: int,
  nefc_in: wp.array(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_state_in: wp.array2d(dtype=int),
  efc_aref_in: wp.array2d(dtype=float),
  qacc_in: wp.array2d(dtype=float),
  efc_aref_grad_in: wp.array2d(dtype=float),
  efc_D_grad_out: wp.array2d(dtype=float),
):
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  dd = efc_D_in[worldid, efcid]
  state = efc_state_in[worldid, efcid]
  if not (dd > 0.0 and state == types.ConstraintState.QUADRATIC.value):
    return
  jqacc = float(0.0)
  for dofid in range(nv):
    jqacc += efc_J_in[worldid, efcid, dofid] * qacc_in[worldid, dofid]
  jv = efc_aref_grad_in[worldid, efcid] / dd
  efc_D_grad_out[worldid, efcid] = jv * (efc_aref_in[worldid, efcid] - jqacc)


@wp.kernel
def _efc_D_grad_sparse_kernel(
  nefc_in: wp.array(dtype=int),
  efc_J_rownnz_in: wp.array2d(dtype=int),
  efc_J_rowadr_in: wp.array2d(dtype=int),
  efc_J_colind_in: wp.array3d(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_state_in: wp.array2d(dtype=int),
  efc_aref_in: wp.array2d(dtype=float),
  qacc_in: wp.array2d(dtype=float),
  efc_aref_grad_in: wp.array2d(dtype=float),
  efc_D_grad_out: wp.array2d(dtype=float),
):
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  dd = efc_D_in[worldid, efcid]
  state = efc_state_in[worldid, efcid]
  if not (dd > 0.0 and state == types.ConstraintState.QUADRATIC.value):
    return
  rownnz = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]
  jqacc = float(0.0)
  for k in range(rownnz):
    col = efc_J_colind_in[worldid, 0, rowadr + k]
    jqacc += efc_J_in[worldid, 0, rowadr + k] * qacc_in[worldid, col]
  jv = efc_aref_grad_in[worldid, efcid] / dd
  efc_D_grad_out[worldid, efcid] = jv * (efc_aref_in[worldid, efcid] - jqacc)


@wp.kernel
def _replay_D_grad_kernel(
  nefc_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  eq_type: wp.array[int],
  jnt_type: wp.array[int],
  efc_D_grad_in: wp.array2d[float],
  efc_D_grad_out: wp.array2d[float],
):
  """Accumulate D cotangents for rows handled by differentiable non-contact replay."""
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return

  constraint_type = efc_type_in[worldid, efcid]
  constraint_id = efc_id_in[worldid, efcid]
  replayed = False
  if constraint_type == types.ConstraintType.EQUALITY:
    replayed = eq_type[constraint_id] == types.EqType.JOINT
  elif constraint_type == types.ConstraintType.LIMIT_TENDON:
    replayed = True
  elif constraint_type == types.ConstraintType.LIMIT_JOINT:
    replayed = jnt_type[constraint_id] == types.JointType.BALL

  if replayed:
    wp.atomic_add(efc_D_grad_out, worldid, efcid, efc_D_grad_in[worldid, efcid])


@wp.kernel
def _qvel_dissipation_dense_kernel(
  nv: int,
  nefc_in: wp.array(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_state_in: wp.array2d(dtype=int),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  contact_solref_in: wp.array(dtype=wp.vec2),
  contact_solreffriction_in: wp.array(dtype=wp.vec2),
  contact_solimp_in: wp.array(dtype=types.vec5),
  dof_solref_in: wp.array2d(dtype=wp.vec2),
  dof_solimp_in: wp.array2d(dtype=types.vec5),
  tendon_solref_fri_in: wp.array2d(dtype=wp.vec2),
  tendon_solimp_fri_in: wp.array2d(dtype=types.vec5),
  opt_timestep_in: wp.array(dtype=float),
  opt_disableflags: int,
  qvel_grad_out: wp.array2d(dtype=float),
  efc_aref_grad_out: wp.array2d(dtype=float),
  efc_vel_grad_out: wp.array2d(dtype=float),
):
  """Dense-J contact/friction velocity VJP, injected before native intermediates can erase it."""
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  etype = efc_type_in[worldid, efcid]
  is_friction_dof = etype == int(types.ConstraintType.FRICTION_DOF.value)
  is_friction_tendon = etype == int(types.ConstraintType.FRICTION_TENDON.value)
  if etype < int(types.ConstraintType.CONTACT_FRICTIONLESS.value) and not (is_friction_dof or is_friction_tendon):
    return
  dd = efc_D_in[worldid, efcid]
  state = efc_state_in[worldid, efcid]
  if not (dd > 0.0 and (state == types.ConstraintState.QUADRATIC.value or state == types.ConstraintState.CONE.value)):
    return
  # For friction rows efc_id is the dof/tendon index; for contacts it is the contact index.
  conid = efc_id_in[worldid, efcid]
  if is_friction_dof:
    solref = dof_solref_in[worldid % dof_solref_in.shape[0], conid]
    solimp = dof_solimp_in[worldid % dof_solimp_in.shape[0], conid]
  elif is_friction_tendon:
    solref = tendon_solref_fri_in[worldid % tendon_solref_fri_in.shape[0], conid]
    solimp = tendon_solimp_fri_in[worldid % tendon_solimp_fri_in.shape[0], conid]
  else:
    solref = contact_solref_in[conid]
    if etype == types.ConstraintType.CONTACT_ELLIPTIC and efcid != contact_efc_address_in[conid, 0]:
      friction_ref = contact_solreffriction_in[conid]
      if friction_ref[0] != 0.0 or friction_ref[1] != 0.0:
        solref = friction_ref
    solimp = contact_solimp_in[conid]
  timestep = opt_timestep_in[worldid % opt_timestep_in.shape[0]]
  dmax = wp.clamp(solimp[1], types.MJ_MINIMP, types.MJ_MAXIMP)
  timeconst = solref[0]
  if not (opt_disableflags & int(types.DisableBit.REFSAFE.value)):
    timeconst = wp.max(timeconst, 2.0 * timestep)
  b = 2.0 / (dmax * timeconst)
  b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)
  factor = -b * efc_aref_grad_out[worldid, efcid]
  for dofid in range(nv):
    wp.atomic_add(qvel_grad_out, worldid, dofid, factor * efc_J_in[worldid, efcid, dofid])
  efc_aref_grad_out[worldid, efcid] = 0.0
  efc_vel_grad_out[worldid, efcid] = 0.0


@wp.kernel
def _qvel_dissipation_kernel(
  nefc_in: wp.array(dtype=int),
  efc_J_rownnz_in: wp.array2d(dtype=int),
  efc_J_rowadr_in: wp.array2d(dtype=int),
  efc_J_colind_in: wp.array3d(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_state_in: wp.array2d(dtype=int),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  contact_solref_in: wp.array(dtype=wp.vec2),
  contact_solreffriction_in: wp.array(dtype=wp.vec2),
  contact_solimp_in: wp.array(dtype=types.vec5),
  dof_solref_in: wp.array2d(dtype=wp.vec2),
  dof_solimp_in: wp.array2d(dtype=types.vec5),
  tendon_solref_fri_in: wp.array2d(dtype=wp.vec2),
  tendon_solimp_fri_in: wp.array2d(dtype=types.vec5),
  opt_timestep_in: wp.array(dtype=float),
  opt_disableflags: int,
  qvel_grad_out: wp.array2d(dtype=float),
  efc_aref_grad_out: wp.array2d(dtype=float),
  efc_vel_grad_out: wp.array2d(dtype=float),
):
  """Inject the contact/friction velocity-dissipation adjoint directly into qvel.grad.

  The constraint solve carries the Baumgarte term
  aref_i = -k*imp*pos_i - b_i*vel_i with vel_i = J_i . qvel. Its reverse-mode contribution is
  adj_qvel = -sum_i b_i adj_aref_i J_i. For independent QUADRATIC rows,
  adj_aref_i = D_i (J_i . v); for elliptic CONE contacts it is the coupled block product C (J v).
  DOF/tendon friction rows carry the same -b*vel term (with pos_aref = 0), sourcing solref/solimp
  from dof_solref/dof_solimp and tendon_solref_fri/tendon_solimp_fri respectively.

  We scatter this J^T product ourselves (atomic_add over the sparse row, indexed by colind),
  rather than route it through efc.vel.grad / efc.aref.grad. The native vel->qvel autodiff of the
  sparse-J assembly indexes the velocity gradient by stored row position instead of efc.J column
  index. We therefore handle every active contact row here and zero its efc.aref.grad /
  efc.vel.grad after the explicit J^T scatter. This also prevents the native aref backward from
  duplicating the position contribution already written by _efc_pos_grad_kernel.
  """
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  etype = efc_type_in[worldid, efcid]
  is_friction_dof = etype == int(types.ConstraintType.FRICTION_DOF.value)
  is_friction_tendon = etype == int(types.ConstraintType.FRICTION_TENDON.value)
  if etype < int(types.ConstraintType.CONTACT_FRICTIONLESS.value) and not (is_friction_dof or is_friction_tendon):
    return
  dd = efc_D_in[worldid, efcid]
  state = efc_state_in[worldid, efcid]
  if not (dd > 0.0 and (state == types.ConstraintState.QUADRATIC.value or state == types.ConstraintState.CONE.value)):
    return
  rownnz = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]
  # For friction rows efc_id is the dof/tendon index; for contacts it is the contact index.
  conid = efc_id_in[worldid, efcid]
  if is_friction_dof:
    solref = dof_solref_in[worldid % dof_solref_in.shape[0], conid]
    solimp = dof_solimp_in[worldid % dof_solimp_in.shape[0], conid]
  elif is_friction_tendon:
    solref = tendon_solref_fri_in[worldid % tendon_solref_fri_in.shape[0], conid]
    solimp = tendon_solimp_fri_in[worldid % tendon_solimp_fri_in.shape[0], conid]
  else:
    solref = contact_solref_in[conid]
    if etype == types.ConstraintType.CONTACT_ELLIPTIC and efcid != contact_efc_address_in[conid, 0]:
      friction_ref = contact_solreffriction_in[conid]
      if friction_ref[0] != 0.0 or friction_ref[1] != 0.0:
        solref = friction_ref
    solimp = contact_solimp_in[conid]
  timestep = opt_timestep_in[worldid % opt_timestep_in.shape[0]]
  dmax = wp.clamp(solimp[1], types.MJ_MINIMP, types.MJ_MAXIMP)
  timeconst = solref[0]
  if not (opt_disableflags & int(types.DisableBit.REFSAFE.value)):
    timeconst = wp.max(timeconst, 2.0 * timestep)
  b = 2.0 / (dmax * timeconst)
  b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)
  factor = -b * efc_aref_grad_out[worldid, efcid]
  for k in range(rownnz):
    col = efc_J_colind_in[worldid, 0, rowadr + k]
    wp.atomic_add(qvel_grad_out, worldid, col, factor * efc_J_in[worldid, 0, rowadr + k])
  # Kill the native velocity propagation for this row so it is not counted twice.
  efc_aref_grad_out[worldid, efcid] = 0.0
  efc_vel_grad_out[worldid, efcid] = 0.0


# kernel_analyzer: on
