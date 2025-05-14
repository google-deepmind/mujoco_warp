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

import warp as wp

from . import math
from . import support
from . import types
from .types import vec5
from .types import vec11
from .warp_util import event_scope

wp.config.enable_backward = False


@wp.func
def _update_efc_row(
  # Model:
  opt_timestep: float,
  # In:
  refsafe: int,
  efcid: int,
  pos_aref: float,
  pos_imp: float,
  invweight: float,
  solref: wp.vec2,
  solimp: vec5,
  margin: float,
  Jqvel: float,
  frictionloss: float,
  id: int,
  # Data out:
  efc_id_out: wp.array(dtype=int),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
):
  # Calculate kbi
  timeconst = solref[0]
  dampratio = solref[1]
  dmin = solimp[0]
  dmax = solimp[1]
  width = solimp[2]
  mid = solimp[3]
  power = solimp[4]

  # TODO(team): wp.static?
  if not refsafe:
    timeconst = wp.max(timeconst, 2.0 * opt_timestep)

  dmin = wp.clamp(dmin, types.MJ_MINIMP, types.MJ_MAXIMP)
  dmax = wp.clamp(dmax, types.MJ_MINIMP, types.MJ_MAXIMP)
  width = wp.max(types.MJ_MINVAL, width)
  mid = wp.clamp(mid, types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, power)

  # See https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
  k = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)
  b = 2.0 / (dmax * timeconst)
  k = wp.where(solref[0] <= 0, -solref[0] / (dmax * dmax), k)
  b = wp.where(solref[1] <= 0, -solref[1] / dmax, b)

  imp_x = wp.abs(pos_imp) / width
  imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
  imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)
  imp_y = wp.where(imp_x < mid, imp_a, imp_b)
  imp = dmin + imp_y * (dmax - dmin)
  imp = wp.clamp(imp, dmin, dmax)
  imp = wp.where(imp_x > 1.0, dmax, imp)

  # Update constraints
  efc_D_out[efcid] = 1.0 / wp.max(invweight * (1.0 - imp) / imp, types.MJ_MINVAL)
  efc_aref_out[efcid] = -k * imp * pos_aref - b * Jqvel
  efc_pos_out[efcid] = pos_aref + margin
  efc_margin_out[efcid] = margin
  efc_frictionloss_out[efcid] = frictionloss
  efc_id_out[efcid] = id


@wp.kernel
def _efc_equality_connect(
  # Model:
  nv: int,
  nsite: int,
  opt_timestep: float,
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_invweight0: wp.array3d(dtype=float),
  dof_bodyid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  eq_obj1id: wp.array(dtype=int),
  eq_obj2id: wp.array(dtype=int),
  eq_objtype: wp.array(dtype=int),
  eq_solref: wp.array2d(dtype=wp.vec2),
  eq_solimp: wp.array2d(dtype=vec5),
  eq_data: wp.array2d(dtype=vec11),
  eq_connect_adr: wp.array(dtype=int),
  # Data in:
  nefc_in: wp.array(dtype=int),
  qvel_in: wp.array2d(dtype=float),
  eq_active_in: wp.array2d(dtype=bool),
  xpos_in: wp.array2d(dtype=wp.vec3),
  xmat_in: wp.array2d(dtype=wp.mat33),
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  refsafe_in: int,
  # Data out:
  ne_connect_out: wp.array(dtype=int),
  efc_worldid_out: wp.array(dtype=int),
  efc_id_out: wp.array(dtype=int),
  efc_J_out: wp.array2d(dtype=float),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
):
  """Calculates constraint rows for connect equality constraints."""

  worldid, i_eq_connect_adr = wp.tid()
  i_eq = eq_connect_adr[i_eq_connect_adr]

  if not eq_active_in[worldid, i_eq]:
    return

  necid = wp.atomic_add(ne_connect_out, 0, 3)
  efcid = nefc_in[0] + necid

  data = eq_data[worldid, i_eq]
  anchor1 = wp.vec3f(data[0], data[1], data[2])
  anchor2 = wp.vec3f(data[3], data[4], data[5])

  obj1id = eq_obj1id[i_eq]
  obj2id = eq_obj2id[i_eq]

  if nsite and eq_objtype[i_eq] == wp.static(types.ObjType.SITE.value):
    # body1id stores the index of site_bodyid.
    body1id = site_bodyid[obj1id]
    body2id = site_bodyid[obj2id]
    pos1 = site_xpos_in[worldid, obj1id]
    pos2 = site_xpos_in[worldid, obj2id]
  else:
    body1id = obj1id
    body2id = obj2id
    pos1 = xpos_in[worldid, body1id] + xmat_in[worldid, body1id] @ anchor1
    pos2 = xpos_in[worldid, body2id] + xmat_in[worldid, body2id] @ anchor2

  # error is difference in global positions
  pos = pos1 - pos2

  # compute Jacobian difference (opposite of contact: 0 - 1)
  Jqvel = wp.vec3f(0.0, 0.0, 0.0)
  for dofid in range(nv):  # TODO: parallelize
    jacp1, _ = support.jac(
      body_parentid,
      body_rootid,
      dof_bodyid,
      subtree_com_in,
      cdof_in,
      pos1,
      body1id,
      dofid,
      worldid,
    )
    jacp2, _ = support.jac(
      body_parentid,
      body_rootid,
      dof_bodyid,
      subtree_com_in,
      cdof_in,
      pos2,
      body2id,
      dofid,
      worldid,
    )
    j1mj2 = jacp1 - jacp2
    efc_J_out[efcid + 0, dofid] = j1mj2[0]
    efc_J_out[efcid + 1, dofid] = j1mj2[1]
    efc_J_out[efcid + 2, dofid] = j1mj2[2]
    Jqvel += j1mj2 * qvel_in[worldid, dofid]

  invweight = body_invweight0[worldid, body1id, 0] + body_invweight0[worldid, body2id, 0]
  pos_imp = wp.length(pos)

  solref = eq_solref[worldid, i_eq]
  solimp = eq_solimp[worldid, i_eq]

  for i in range(3):
    efcidi = efcid + i
    efc_worldid_out[efcidi] = worldid

    _update_efc_row(
      opt_timestep,
      refsafe_in,
      efcidi,
      pos[i],
      pos_imp,
      invweight,
      solref,
      solimp,
      0.0,
      Jqvel[i],
      0.0,
      i_eq,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_aref_out,
      efc_frictionloss_out,
    )


@wp.kernel
def _efc_equality_joint(
  # Model:
  opt_timestep: float,
  qpos0: wp.array2d(dtype=float),
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  dof_invweight0: wp.array2d(dtype=float),
  eq_obj1id: wp.array(dtype=int),
  eq_obj2id: wp.array(dtype=int),
  eq_solref: wp.array2d(dtype=wp.vec2),
  eq_solimp: wp.array2d(dtype=vec5),
  eq_data: wp.array2d(dtype=vec11),
  eq_jnt_adr: wp.array(dtype=int),
  # Data in:
  ne_connect_in: wp.array(dtype=int),
  ne_weld_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  qpos_in: wp.array2d(dtype=float),
  qvel_in: wp.array2d(dtype=float),
  eq_active_in: wp.array2d(dtype=bool),
  # In:
  refsafe_in: int,
  # Data out:
  ne_jnt_out: wp.array(dtype=int),
  efc_worldid_out: wp.array(dtype=int),
  efc_id_out: wp.array(dtype=int),
  efc_J_out: wp.array2d(dtype=float),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
):
  worldid, i_eq_joint_adr = wp.tid()
  i_eq = eq_jnt_adr[i_eq_joint_adr]
  if not eq_active_in[worldid, i_eq]:
    return

  nejid = wp.atomic_add(ne_jnt_out, 0, 1)
  efcid = nefc_in[0] + ne_connect_in[0] + ne_weld_in[0] + nejid
  efc_worldid_out[efcid] = worldid

  jntid_1 = eq_obj1id[i_eq]
  jntid_2 = eq_obj2id[i_eq]
  data = eq_data[worldid, i_eq]
  dofadr1 = jnt_dofadr[jntid_1]
  qposadr1 = jnt_qposadr[jntid_1]
  efc_J_out[efcid, dofadr1] = 1.0

  if jntid_2 > -1:
    # Two joint constraint
    qposadr2 = jnt_qposadr[jntid_2]
    dofadr2 = jnt_dofadr[jntid_2]
    dif = qpos_in[worldid, qposadr2] - qpos0[worldid, qposadr2]

    # Horner's method for polynomials
    rhs = data[0] + dif * (data[1] + dif * (data[2] + dif * (data[3] + dif * data[4])))
    deriv_2 = data[1] + dif * (2.0 * data[2] + dif * (3.0 * data[3] + dif * 4.0 * data[4]))

    pos = qpos_in[worldid, qposadr1] - qpos0[worldid, qposadr1] - rhs
    Jqvel = qvel_in[worldid, dofadr1] - qvel_in[worldid, dofadr2] * deriv_2
    invweight = dof_invweight0[worldid, dofadr1] + dof_invweight0[worldid, dofadr2]

    efc_J_out[efcid, dofadr2] = -deriv_2
  else:
    # Single joint constraint
    pos = qpos_in[worldid, qposadr1] - qpos0[worldid, qposadr1] - data[0]
    Jqvel = qvel_in[worldid, dofadr1]
    invweight = dof_invweight0[worldid, dofadr1]

  # Update constraint parameters
  _update_efc_row(
    opt_timestep,
    refsafe_in,
    efcid,
    pos,
    pos,
    invweight,
    eq_solref[worldid, i_eq],
    eq_solimp[worldid, i_eq],
    0.0,
    Jqvel,
    0.0,
    i_eq,
    efc_id_out,
    efc_pos_out,
    efc_margin_out,
    efc_D_out,
    efc_aref_out,
    efc_frictionloss_out,
  )


@wp.kernel
def _efc_equality_tendon(
  # Model:
  nv: int,
  opt_timestep: float,
  eq_obj1id: wp.array(dtype=int),
  eq_obj2id: wp.array(dtype=int),
  eq_solref: wp.array2d(dtype=wp.vec2),
  eq_solimp: wp.array2d(dtype=vec5),
  eq_data: wp.array2d(dtype=vec11),
  eq_ten_adr: wp.array(dtype=int),
  tendon_length0: wp.array2d(dtype=float),
  tendon_invweight0: wp.array2d(dtype=float),
  # Data in:
  ne_connect_in: wp.array(dtype=int),
  ne_weld_in: wp.array(dtype=int),
  ne_jnt_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  qvel_in: wp.array2d(dtype=float),
  eq_active_in: wp.array2d(dtype=bool),
  ten_length_in: wp.array2d(dtype=float),
  ten_J_in: wp.array3d(dtype=float),
  # In:
  refsafe_in: int,
  # Data out:
  ne_ten_out: wp.array(dtype=int),
  efc_worldid_out: wp.array(dtype=int),
  efc_id_out: wp.array(dtype=int),
  efc_J_out: wp.array2d(dtype=float),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
):
  worldid, tenid = wp.tid()
  eqid = eq_ten_adr[tenid]

  if not eq_active_in[worldid, eqid]:
    return

  netid = wp.atomic_add(ne_ten_out, 0, 1)
  efcid = nefc_in[0] + ne_connect_in[0] + ne_weld_in[0] + ne_jnt_in[0] + netid
  efc_worldid_out[efcid] = worldid

  obj1id = eq_obj1id[eqid]
  obj2id = eq_obj2id[eqid]
  data = eq_data[worldid,eqid]
  solref = eq_solref[worldid, eqid]
  solimp = eq_solimp[worldid, eqid]
  pos1 = ten_length_in[worldid, obj1id] - tendon_length0[worldid, obj1id]
  pos2 = ten_length_in[worldid, obj2id] - tendon_length0[worldid, obj2id]
  jac1 = ten_J_in[worldid, obj1id]
  jac2 = ten_J_in[worldid, obj2id]

  if obj2id > -1:
    invweight = tendon_invweight0[worldid, obj1id] + tendon_invweight0[worldid, obj2id]

    dif = pos2
    dif2 = dif * dif
    dif3 = dif2 * dif
    dif4 = dif3 * dif

    pos = pos1 - (data[0] + data[1] * dif + data[2] * dif2 + data[3] * dif3 + data[4] * dif4)
    deriv = data[1] + 2.0 * data[2] * dif + 3.0 * data[3] * dif2 + 4.0 * data[4] * dif3
  else:
    invweight = tendon_invweight0[worldid, obj1id]
    pos = pos1 - data[0]
    deriv = 0.0

  Jqvel = float(0.0)
  for i in range(nv):
    if deriv != 0.0:
      J = jac1[i] + jac2[i] * -deriv
    else:
      J = jac1[i]
    efc_J_out[efcid, i] = J
    Jqvel += J * qvel_in[worldid, i]

  _update_efc_row(
    opt_timestep,
    refsafe_in,
    efcid,
    pos,
    pos,
    invweight,
    solref,
    solimp,
    0.0,
    Jqvel,
    0.0,
    eqid,
    efc_id_out,
    efc_pos_out,
    efc_margin_out,
    efc_D_out,
    efc_aref_out,
    efc_frictionloss_out,
  )


@wp.kernel
def _efc_friction(
  # Model:
  opt_timestep: float,
  dof_invweight0: wp.array2d(dtype=float),
  dof_frictionloss: wp.array2d(dtype=float),
  dof_solimp: wp.array2d(dtype=vec5),
  dof_solref: wp.array2d(dtype=wp.vec2),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  # In:
  refsafe_in: int,
  # Data out:
  nf_out: wp.array(dtype=int),
  nefc_out: wp.array(dtype=int),
  efc_worldid_out: wp.array(dtype=int),
  efc_id_out: wp.array(dtype=int),
  efc_J_out: wp.array2d(dtype=float),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
):
  # TODO(team): tendon
  worldid, dofid = wp.tid()

  if dof_frictionloss[worldid, dofid] <= 0.0:
    return

  efcid = wp.atomic_add(nefc_out, 0, 1)
  wp.atomic_add(nf_out, 0, 1)
  efc_worldid_out[efcid] = worldid

  efc_J_out[efcid, dofid] = 1.0
  Jqvel = qvel_in[worldid, dofid]

  _update_efc_row(
    opt_timestep,
    refsafe_in,
    efcid,
    0.0,
    0.0,
    dof_invweight0[worldid, dofid],
    dof_solref[worldid, dofid],
    dof_solimp[worldid, dofid],
    0.0,
    Jqvel,
    dof_frictionloss[worldid, dofid],
    dofid,
    efc_id_out,
    efc_pos_out,
    efc_margin_out,
    efc_D_out,
    efc_aref_out,
    efc_frictionloss_out,
  )


@wp.kernel
def _efc_equality_weld(
  # Model:
  nv: int,
  nsite: int,
  opt_timestep: float,
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_invweight0: wp.array3d(dtype=float),
  dof_bodyid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  site_quat: wp.array2d(dtype=wp.quat),
  eq_obj1id: wp.array(dtype=int),
  eq_obj2id: wp.array(dtype=int),
  eq_objtype: wp.array(dtype=int),
  eq_solref: wp.array2d(dtype=wp.vec2),
  eq_solimp: wp.array2d(dtype=vec5),
  eq_data: wp.array2d(dtype=vec11),
  eq_wld_adr: wp.array(dtype=int),
  # Data in:
  ne_connect_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  qvel_in: wp.array2d(dtype=float),
  eq_active_in: wp.array2d(dtype=bool),
  xpos_in: wp.array2d(dtype=wp.vec3),
  xquat_in: wp.array2d(dtype=wp.quat),
  xmat_in: wp.array2d(dtype=wp.mat33),
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  refsafe_in: int,
  # Data out:
  ne_weld_out: wp.array(dtype=int),
  efc_worldid_out: wp.array(dtype=int),
  efc_id_out: wp.array(dtype=int),
  efc_J_out: wp.array2d(dtype=float),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
):
  worldid, i_eq_weld_adr = wp.tid()
  i_eq = eq_wld_adr[i_eq_weld_adr]
  if not eq_active_in[worldid, i_eq]:
    return

  newid = wp.atomic_add(ne_weld_out, 0, 6)
  efcid = nefc_in[0] + ne_connect_in[0] + newid
  for i in range(wp.static(6)):
    efc_worldid_out[efcid + i] = worldid

  is_site = eq_objtype[i_eq] == wp.static(types.ObjType.SITE.value) and nsite > 0

  obj1id = eq_obj1id[i_eq]
  obj2id = eq_obj2id[i_eq]

  data = eq_data[worldid, i_eq]
  anchor1 = wp.vec3(data[0], data[1], data[2])
  anchor2 = wp.vec3(data[3], data[4], data[5])
  relpose = wp.quat(data[6], data[7], data[8], data[9])
  torquescale = data[10]

  if is_site:
    # body1id stores the index of site_bodyid.
    body1id = site_bodyid[obj1id]
    body2id = site_bodyid[obj2id]
    pos1 = site_xpos_in[worldid, obj1id]
    pos2 = site_xpos_in[worldid, obj2id]

    quat = math.mul_quat(xquat_in[worldid, body1id], site_quat[worldid, obj1id])
    quat1 = math.quat_inv(math.mul_quat(xquat_in[worldid, body2id], site_quat[worldid, obj2id]))

  else:
    body1id = obj1id
    body2id = obj2id
    pos1 = xpos_in[worldid, body1id] + xmat_in[worldid, body1id] @ anchor2
    pos2 = xpos_in[worldid, body2id] + xmat_in[worldid, body2id] @ anchor1

    quat = math.mul_quat(xquat_in[worldid, body1id], relpose)
    quat1 = math.quat_inv(xquat_in[worldid, body2id])

  # compute Jacobian difference (opposite of contact: 0 - 1)
  Jqvelp = wp.vec3f(0.0, 0.0, 0.0)
  Jqvelr = wp.vec3f(0.0, 0.0, 0.0)

  for dofid in range(nv):  # TODO: parallelize
    jacp1, jacr1 = support.jac(
      body_parentid,
      body_rootid,
      dof_bodyid,
      subtree_com_in,
      cdof_in,
      pos1,
      body1id,
      dofid,
      worldid,
    )
    jacp2, jacr2 = support.jac(
      body_parentid,
      body_rootid,
      dof_bodyid,
      subtree_com_in,
      cdof_in,
      pos2,
      body2id,
      dofid,
      worldid,
    )

    jacdifp = jacp1 - jacp2
    for i in range(wp.static(3)):
      efc_J_out[efcid + i, dofid] = jacdifp[i]

    jacdifr = (jacr1 - jacr2) * torquescale
    jacdifrq = math.mul_quat(math.quat_mul_axis(quat1, jacdifr), quat)
    jacdifr = 0.5 * wp.vec3(jacdifrq[1], jacdifrq[2], jacdifrq[3])

    for i in range(wp.static(3)):
      efc_J_out[efcid + 3 + i, dofid] = jacdifr[i]

    Jqvelp += jacdifp * qvel_in[worldid, dofid]
    Jqvelr += jacdifr * qvel_in[worldid, dofid]

  # error is difference in global position and orientation
  cpos = pos1 - pos2

  crotq = math.mul_quat(quat1, quat)  # copy axis components
  crot = wp.vec3(crotq[1], crotq[2], crotq[3]) * torquescale

  invweight_t = body_invweight0[worldid, body1id, 0] + body_invweight0[worldid, body2id, 0]

  pos_imp = wp.sqrt(wp.length_sq(cpos) + wp.length_sq(crot))

  solref = eq_solref[worldid, i_eq]
  solimp = eq_solimp[worldid, i_eq]

  for i in range(3):
    _update_efc_row(
      opt_timestep,
      refsafe_in,
      efcid + i,
      cpos[i],
      pos_imp,
      invweight_t,
      solref,
      solimp,
      0.0,
      Jqvelp[i],
      0.0,
      i_eq,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_aref_out,
      efc_frictionloss_out,
    )

  invweight_r = body_invweight0[worldid, body1id, 1] + body_invweight0[worldid, body2id, 1]

  for i in range(3):
    _update_efc_row(
      opt_timestep,
      refsafe_in,
      efcid + 3 + i,
      crot[i],
      pos_imp,
      invweight_r,
      solref,
      solimp,
      0.0,
      Jqvelr[i],
      0.0,
      i_eq,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_aref_out,
      efc_frictionloss_out,
    )


@wp.kernel
def _efc_limit_slide_hinge(
  # Model:
  opt_timestep: float,
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  jnt_solref: wp.array2d(dtype=wp.vec2),
  jnt_solimp: wp.array2d(dtype=vec5),
  jnt_range: wp.array3d(dtype=float),
  jnt_margin: wp.array2d(dtype=float),
  jnt_limited_slide_hinge_adr: wp.array(dtype=int),
  dof_invweight0: wp.array2d(dtype=float),
  # Data in:
  nefc_in: wp.array(dtype=int),
  qpos_in: wp.array2d(dtype=float),
  qvel_in: wp.array2d(dtype=float),
  # In:
  refsafe_in: int,
  # Data out:
  nl_out: wp.array(dtype=int),
  efc_worldid_out: wp.array(dtype=int),
  efc_id_out: wp.array(dtype=int),
  efc_J_out: wp.array2d(dtype=float),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
):
  worldid, jntlimitedid = wp.tid()
  jntid = jnt_limited_slide_hinge_adr[jntlimitedid]
  jntrange = jnt_range[worldid, jntid]

  qpos = qpos_in[worldid, jnt_qposadr[jntid]]
  jntmargin = jnt_margin[worldid, jntid]
  dist_min, dist_max = qpos - jntrange[0], jntrange[1] - qpos
  pos = wp.min(dist_min, dist_max) - jntmargin
  active = pos < 0

  if active:
    lid = wp.atomic_add(nl_out, 0, 1)
    efcid = lid + nefc_in[0]
    efc_worldid_out[efcid] = worldid

    dofadr = jnt_dofadr[jntid]

    J = float(dist_min < dist_max) * 2.0 - 1.0
    efc_J_out[efcid, dofadr] = J
    Jqvel = J * qvel_in[worldid, dofadr]

    _update_efc_row(
      opt_timestep,
      refsafe_in,
      efcid,
      pos,
      pos,
      dof_invweight0[worldid, dofadr],
      jnt_solref[worldid, jntid],
      jnt_solimp[worldid, jntid],
      jntmargin,
      Jqvel,
      0.0,
      dofadr,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_aref_out,
      efc_frictionloss_out,
    )


@wp.kernel
def _efc_limit_ball(
  # Model:
  opt_timestep: float,
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  jnt_solref: wp.array2d(dtype=wp.vec2),
  jnt_solimp: wp.array2d(dtype=vec5),
  jnt_range: wp.array3d(dtype=float),
  jnt_margin: wp.array2d(dtype=float),
  jnt_limited_ball_adr: wp.array(dtype=int),
  dof_invweight0: wp.array2d(dtype=float),
  # Data in:
  nefc_in: wp.array(dtype=int),
  qpos_in: wp.array2d(dtype=float),
  qvel_in: wp.array2d(dtype=float),
  # In:
  refsafe_in: int,
  # Data out:
  nl_out: wp.array(dtype=int),
  efc_worldid_out: wp.array(dtype=int),
  efc_id_out: wp.array(dtype=int),
  efc_J_out: wp.array2d(dtype=float),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
):
  worldid, jntlimitedid = wp.tid()
  jntid = jnt_limited_ball_adr[jntlimitedid]
  qposadr = jnt_qposadr[jntid]

  qpos = qpos_in[worldid]
  jnt_quat = wp.quat(qpos[qposadr + 0], qpos[qposadr + 1], qpos[qposadr + 2], qpos[qposadr + 3])
  axis_angle = math.quat_to_vel(jnt_quat)
  jntrange = jnt_range[worldid, jntid]
  axis, angle = math.normalize_with_norm(axis_angle)
  jntmargin = jnt_margin[worldid, jntid]

  pos = wp.max(jntrange[0], jntrange[1]) - angle - jntmargin
  active = pos < 0

  if active:
    lid = wp.atomic_add(nl_out, 0, 1)
    efcid = lid + nefc_in[0]
    efc_worldid_out[efcid] = worldid

    dofadr = jnt_dofadr[jntid]

    efc_J_out[efcid, dofadr + 0] = -axis[0]
    efc_J_out[efcid, dofadr + 1] = -axis[1]
    efc_J_out[efcid, dofadr + 2] = -axis[2]

    Jqvel = -axis[0] * qvel_in[worldid, dofadr + 0]
    Jqvel -= axis[1] * qvel_in[worldid, dofadr + 1]
    Jqvel -= axis[2] * qvel_in[worldid, dofadr + 2]

    _update_efc_row(
      opt_timestep,
      refsafe_in,
      efcid,
      pos,
      pos,
      dof_invweight0[worldid, dofadr],
      jnt_solref[worldid, jntid],
      jnt_solimp[worldid, jntid],
      jntmargin,
      Jqvel,
      0.0,
      jntid,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_aref_out,
      efc_frictionloss_out,
    )


@wp.kernel
def _efc_limit_tendon(
  # Model:
  nv: int,
  opt_timestep: float,
  jnt_dofadr: wp.array(dtype=int),
  tendon_adr: wp.array(dtype=int),
  tendon_num: wp.array(dtype=int),
  tendon_limited_adr: wp.array(dtype=int),
  tendon_solref_lim: wp.array2d(dtype=wp.vec2),
  tendon_solimp_lim: wp.array2d(dtype=vec5),
  tendon_range: wp.array2d(dtype=wp.vec2),
  tendon_margin: wp.array2d(dtype=float),
  tendon_invweight0: wp.array2d(dtype=float),
  wrap_objid: wp.array(dtype=int),
  wrap_type: wp.array(dtype=int),
  # Data in:
  nefc_in: wp.array(dtype=int),
  qvel_in: wp.array2d(dtype=float),
  ten_length_in: wp.array2d(dtype=float),
  ten_J_in: wp.array3d(dtype=float),
  # In:
  refsafe_in: int,
  # Data out:
  nl_out: wp.array(dtype=int),
  efc_worldid_out: wp.array(dtype=int),
  efc_id_out: wp.array(dtype=int),
  efc_J_out: wp.array2d(dtype=float),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
):
  worldid, tenlimitedid = wp.tid()
  tenid = tendon_limited_adr[tenlimitedid]

  tenrange = tendon_range[worldid, tenid]
  length = ten_length_in[worldid, tenid]
  dist_min, dist_max = length - tenrange[0], tenrange[1] - length
  tenmargin = tendon_margin[worldid, tenid]
  pos = wp.min(dist_min, dist_max) - tenmargin
  active = pos < 0

  if active:
    lid = wp.atomic_add(nl_out, 0, 1)
    efcid = nefc_in[0] + lid
    efc_worldid_out[efcid] = worldid

    Jqvel = float(0.0)
    scl = float(dist_min < dist_max) * 2.0 - 1.0

    adr = tendon_adr[tenid]
    if wrap_type[adr] == wp.static(types.WrapType.JOINT.value):
      ten_num = tendon_num[tenid]
      for i in range(ten_num):
        dofadr = jnt_dofadr[wrap_objid[adr + i]]
        J = scl * ten_J_in[worldid, tenid, dofadr]
        efc_J_out[efcid, dofadr] = J
        Jqvel += J * qvel_in[worldid, dofadr]
    else:
      for i in range(nv):
        J = scl * ten_J_in[worldid, tenid, i]
        efc_J_out[efcid, i] = J
        Jqvel += J * qvel_in[worldid, i]

    _update_efc_row(
      opt_timestep,
      refsafe_in,
      efcid,
      pos,
      pos,
      tendon_invweight0[worldid, tenid],
      tendon_solref_lim[worldid, tenid],
      tendon_solimp_lim[worldid, tenid],
      tenmargin,
      Jqvel,
      0.0,
      tenid,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_aref_out,
      efc_frictionloss_out,
    )


@wp.kernel
def _efc_contact_pyramidal(
  # Model:
  nv: int,
  opt_timestep: float,
  opt_impratio: float,
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_invweight0: wp.array3d(dtype=float),
  dof_bodyid: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  # Data in:
  ncon_in: wp.array(dtype=int),
  qvel_in: wp.array2d(dtype=float),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  refsafe_in: int,
  dist_in: wp.array(dtype=float),
  condim_in: wp.array(dtype=int),
  includemargin_in: wp.array(dtype=float),
  worldid_in: wp.array(dtype=int),
  geom_in: wp.array(dtype=wp.vec2i),
  pos_in: wp.array(dtype=wp.vec3),
  frame_in: wp.array(dtype=wp.mat33),
  friction_in: wp.array(dtype=vec5),
  solref_in: wp.array(dtype=wp.vec2),
  solimp_in: wp.array(dtype=vec5),
  # Data out:
  nefc_out: wp.array(dtype=int),
  efc_worldid_out: wp.array(dtype=int),
  efc_id_out: wp.array(dtype=int),
  efc_J_out: wp.array2d(dtype=float),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
):
  conid, dimid = wp.tid()

  if conid >= ncon_in[0]:
    return

  condim = condim_in[conid]

  if condim == 1 and dimid > 0:
    return
  elif condim > 1 and dimid >= 2 * (condim - 1):
    return

  includemargin = includemargin_in[conid]
  pos = dist_in[conid] - includemargin
  active = pos < 0

  if active:
    efcid = wp.atomic_add(nefc_out, 0, 1)
    worldid = worldid_in[conid]
    efc_worldid_out[efcid] = worldid

    geom = geom_in[conid]
    body1 = geom_bodyid[geom[0]]
    body2 = geom_bodyid[geom[1]]

    con_pos = pos_in[conid]
    frame = frame_in[conid]

    # pyramidal has common invweight across all edges
    invweight = body_invweight0[worldid, body1, 0] + body_invweight0[worldid, body2, 0]

    if condim > 1:
      dimid2 = dimid / 2 + 1

      friction = friction_in[conid]
      fri0 = friction[0]
      frii = friction[dimid2 - 1]
      invweight = invweight + fri0 * fri0 * invweight
      invweight = invweight * 2.0 * fri0 * fri0 / opt_impratio

    Jqvel = float(0.0)
    for i in range(nv):
      J = float(0.0)
      Ji = float(0.0)
      jac1p, jac1r = support.jac(
        body_parentid,
        body_rootid,
        dof_bodyid,
        subtree_com_in,
        cdof_in,
        con_pos,
        body1,
        i,
        worldid,
      )
      jac2p, jac2r = support.jac(
        body_parentid,
        body_rootid,
        dof_bodyid,
        subtree_com_in,
        cdof_in,
        con_pos,
        body2,
        i,
        worldid,
      )
      jacp_dif = jac2p - jac1p
      for xyz in range(3):
        J += frame[0, xyz] * jacp_dif[xyz]

        if condim > 1:
          if dimid2 < 3:
            Ji += frame[dimid2, xyz] * jacp_dif[xyz]
          else:
            Ji += frame[dimid2 - 3, xyz] * (jac2r[xyz] - jac1r[xyz])

      if condim > 1:
        if dimid % 2 == 0:
          J += Ji * frii
        else:
          J -= Ji * frii

      efc_J_out[efcid, i] = J
      Jqvel += J * qvel_in[worldid, i]

    _update_efc_row(
      opt_timestep,
      refsafe_in,
      efcid,
      pos,
      pos,
      invweight,
      solref_in[conid],
      solimp_in[conid],
      includemargin,
      Jqvel,
      0.0,
      conid,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_aref_out,
      efc_frictionloss_out,
    )


@wp.kernel
def _efc_contact_elliptic(
  # Model:
  nv: int,
  opt_timestep: float,
  opt_impratio: float,
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_invweight0: wp.array3d(dtype=float),
  dof_bodyid: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  # Data in:
  ncon_in: wp.array(dtype=int),
  qvel_in: wp.array2d(dtype=float),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  refsafe_in: int,
  dist_in: wp.array(dtype=float),
  condim_in: wp.array(dtype=int),
  includemargin_in: wp.array(dtype=float),
  worldid_in: wp.array(dtype=int),
  geom_in: wp.array(dtype=wp.vec2i),
  pos_in: wp.array(dtype=wp.vec3),
  frame_in: wp.array(dtype=wp.mat33),
  friction_in: wp.array(dtype=vec5),
  solref_in: wp.array(dtype=wp.vec2),
  solreffriction_in: wp.array(dtype=wp.vec2),
  solimp_in: wp.array(dtype=vec5),
  # Data out:
  nefc_out: wp.array(dtype=int),
  efc_worldid_out: wp.array(dtype=int),
  efc_id_out: wp.array(dtype=int),
  efc_J_out: wp.array2d(dtype=float),
  efc_pos_out: wp.array(dtype=float),
  efc_margin_out: wp.array(dtype=float),
  efc_D_out: wp.array(dtype=float),
  efc_aref_out: wp.array(dtype=float),
  efc_frictionloss_out: wp.array(dtype=float),
  # Out:
  efc_address_out: wp.array2d(dtype=int),
):
  conid, dimid = wp.tid()

  if conid >= ncon_in[0]:
    return

  condim = condim_in[conid]

  if dimid > condim - 1:
    return

  includemargin = includemargin_in[conid]
  pos = dist_in[conid] - includemargin
  active = pos < 0.0

  if active:
    efcid = wp.atomic_add(nefc_out, 0, 1)
    worldid = worldid_in[conid]
    efc_worldid_out[efcid] = worldid
    efc_address_out[conid, dimid] = efcid

    geom = geom_in[conid]
    body1 = geom_bodyid[geom[0]]
    body2 = geom_bodyid[geom[1]]

    cpos = pos_in[conid]
    frame = frame_in[conid]

    # TODO(team): parallelize J and Jqvel computation?
    Jqvel = float(0.0)
    for i in range(nv):
      J = float(0.0)
      jac1p, jac1r = support.jac(
        body_parentid,
        body_rootid,
        dof_bodyid,
        subtree_com_in,
        cdof_in,
        cpos,
        body1,
        i,
        worldid,
      )
      jac2p, jac2r = support.jac(
        body_parentid,
        body_rootid,
        dof_bodyid,
        subtree_com_in,
        cdof_in,
        cpos,
        body2,
        i,
        worldid,
      )
      for xyz in range(3):
        if dimid < 3:
          jac_dif = jac2p[xyz] - jac1p[xyz]
          J += frame[dimid, xyz] * jac_dif
        else:
          jac_dif = jac2r[xyz] - jac1r[xyz]
          J += frame[dimid - 3, xyz] * jac_dif

      efc_J_out[efcid, i] = J
      Jqvel += J * qvel_in[worldid, i]

    invweight = body_invweight0[worldid, body1, 0] + body_invweight0[worldid, body2, 0]

    ref = solref_in[conid]
    pos_aref = pos

    if dimid > 0:
      solreffriction = solreffriction_in[conid]

      # non-normal directions use solreffriction (if non-zero)
      if solreffriction[0] or solreffriction[1]:
        ref = solreffriction

      # TODO(team): precompute 1 / impratio
      invweight = invweight / opt_impratio
      friction = friction_in[conid]

      if dimid > 1:
        fri0 = friction[0]
        frii = friction[dimid - 1]
        fri = fri0 * fri0 / (frii * frii)
        invweight *= fri

      pos_aref = 0.0

    _update_efc_row(
      opt_timestep,
      refsafe_in,
      efcid,
      pos_aref,
      pos,
      invweight,
      ref,
      solimp_in[conid],
      includemargin,
      Jqvel,
      0.0,
      conid,
      efc_id_out,
      efc_pos_out,
      efc_margin_out,
      efc_D_out,
      efc_aref_out,
      efc_frictionloss_out,
    )


@wp.kernel
def _num_equality(
  # Data in:
  ne_connect_in: wp.array(dtype=int),
  ne_weld_in: wp.array(dtype=int),
  ne_jnt_in: wp.array(dtype=int),
  ne_ten_in: wp.array(dtype=int),
  # Data out:
  ne_out: wp.array(dtype=int),
  nefc_out: wp.array(dtype=int),
):
  ne = ne_connect_in[0] + ne_weld_in[0] + ne_jnt_in[0] + ne_ten_in[0]
  ne_out[0] = ne
  nefc_out[0] += ne


@wp.kernel
def _update_nefc(
  # Data in:
  nl_in: wp.array(dtype=int),
  # Data out:
  nefc_out: wp.array(dtype=int),
):
  nefc_out[0] += nl_in[0]


@event_scope
def make_constraint(m: types.Model, d: types.Data):
  """Creates constraint jacobians and other supporting data."""

  d.ne.zero_()
  d.ne_connect.zero_()
  d.ne_weld.zero_()
  d.ne_jnt.zero_()
  d.ne_ten.zero_()
  d.nefc.zero_()
  d.nf.zero_()
  d.nl.zero_()

  if not (m.opt.disableflags & types.DisableBit.CONSTRAINT.value):
    d.efc.J.zero_()

    refsafe = m.opt.disableflags & types.DisableBit.REFSAFE

    if not (m.opt.disableflags & types.DisableBit.EQUALITY.value):
      wp.launch(
        _efc_equality_connect,
        dim=(d.nworld, m.eq_connect_adr.size),
        inputs=[
          m.nv,
          m.nsite,
          m.opt.timestep,
          m.body_parentid,
          m.body_rootid,
          m.body_invweight0,
          m.dof_bodyid,
          m.site_bodyid,
          m.eq_obj1id,
          m.eq_obj2id,
          m.eq_objtype,
          m.eq_solref,
          m.eq_solimp,
          m.eq_data,
          m.eq_connect_adr,
          d.nefc,
          d.qvel,
          d.eq_active,
          d.xpos,
          d.xmat,
          d.site_xpos,
          d.subtree_com,
          d.cdof,
          refsafe,
        ],
        outputs=[
          d.ne_connect,
          d.efc.worldid,
          d.efc.id,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.aref,
          d.efc.frictionloss,
        ],
      )
      wp.launch(
        _efc_equality_weld,
        dim=(d.nworld, m.eq_wld_adr.size),
        inputs=[
          m.nv,
          m.nsite,
          m.opt.timestep,
          m.body_parentid,
          m.body_rootid,
          m.body_invweight0,
          m.dof_bodyid,
          m.site_bodyid,
          m.site_quat,
          m.eq_obj1id,
          m.eq_obj2id,
          m.eq_objtype,
          m.eq_solref,
          m.eq_solimp,
          m.eq_data,
          m.eq_wld_adr,
          d.ne_connect,
          d.nefc,
          d.qvel,
          d.eq_active,
          d.xpos,
          d.xquat,
          d.xmat,
          d.site_xpos,
          d.subtree_com,
          d.cdof,
          refsafe,
        ],
        outputs=[
          d.ne_weld,
          d.efc.worldid,
          d.efc.id,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.aref,
          d.efc.frictionloss,
        ],
      )
      wp.launch(
        _efc_equality_joint,
        dim=(d.nworld, m.eq_jnt_adr.size),
        inputs=[
          m.opt.timestep,
          m.qpos0,
          m.jnt_qposadr,
          m.jnt_dofadr,
          m.dof_invweight0,
          m.eq_obj1id,
          m.eq_obj2id,
          m.eq_solref,
          m.eq_solimp,
          m.eq_data,
          m.eq_jnt_adr,
          d.ne_connect,
          d.ne_weld,
          d.nefc,
          d.qpos,
          d.qvel,
          d.eq_active,
          refsafe,
        ],
        outputs=[
          d.ne_jnt,
          d.efc.worldid,
          d.efc.id,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.aref,
          d.efc.frictionloss,
        ],
      )
      wp.launch(
        _efc_equality_tendon,
        dim=(d.nworld, m.eq_ten_adr.size),
        inputs=[
          m.nv,
          m.opt.timestep,
          m.eq_obj1id,
          m.eq_obj2id,
          m.eq_solref,
          m.eq_solimp,
          m.eq_data,
          m.eq_ten_adr,
          m.tendon_length0,
          m.tendon_invweight0,
          d.ne_connect,
          d.ne_weld,
          d.ne_jnt,
          d.nefc,
          d.qvel,
          d.eq_active,
          d.ten_length,
          d.ten_J,
          refsafe,
        ],
        outputs=[
          d.ne_ten,
          d.efc.worldid,
          d.efc.id,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.aref,
          d.efc.frictionloss,
        ],
      )

      wp.launch(
        _num_equality,
        dim=(1,),
        inputs=[
          d.ne_connect,
          d.ne_weld,
          d.ne_jnt,
          d.ne_ten,
        ],
        outputs=[
          d.ne,
          d.nefc,
        ],
      )

    if not (m.opt.disableflags & types.DisableBit.FRICTIONLOSS.value):
      wp.launch(
        _efc_friction,
        dim=(d.nworld, m.nv),
        inputs=[
          m.opt.timestep,
          m.dof_invweight0,
          m.dof_frictionloss,
          m.dof_solimp,
          m.dof_solref,
          d.qvel,
          refsafe,
        ],
        outputs=[
          d.nf,
          d.nefc,
          d.efc.worldid,
          d.efc.id,
          d.efc.J,
          d.efc.pos,
          d.efc.margin,
          d.efc.D,
          d.efc.aref,
          d.efc.frictionloss,
        ],
      )

    # limit
    if not (m.opt.disableflags & types.DisableBit.LIMIT.value):
      limit_ball = m.jnt_limited_ball_adr.size > 0
      if limit_ball:
        wp.launch(
          _efc_limit_ball,
          dim=(d.nworld, m.jnt_limited_ball_adr.size),
          inputs=[
            m.opt.timestep,
            m.jnt_qposadr,
            m.jnt_dofadr,
            m.jnt_solref,
            m.jnt_solimp,
            m.jnt_range,
            m.jnt_margin,
            m.jnt_limited_ball_adr,
            m.dof_invweight0,
            d.nefc,
            d.qpos,
            d.qvel,
            refsafe,
          ],
          outputs=[
            d.nl,
            d.efc.worldid,
            d.efc.id,
            d.efc.J,
            d.efc.pos,
            d.efc.margin,
            d.efc.D,
            d.efc.aref,
            d.efc.frictionloss,
          ],
        )

      limit_slide_hinge = m.jnt_limited_slide_hinge_adr.size > 0
      if limit_slide_hinge:
        wp.launch(
          _efc_limit_slide_hinge,
          dim=(d.nworld, m.jnt_limited_slide_hinge_adr.size),
          inputs=[
            m.opt.timestep,
            m.jnt_qposadr,
            m.jnt_dofadr,
            m.jnt_solref,
            m.jnt_solimp,
            m.jnt_range,
            m.jnt_margin,
            m.jnt_limited_slide_hinge_adr,
            m.dof_invweight0,
            d.nefc,
            d.qpos,
            d.qvel,
            refsafe,
          ],
          outputs=[
            d.nl,
            d.efc.worldid,
            d.efc.id,
            d.efc.J,
            d.efc.pos,
            d.efc.margin,
            d.efc.D,
            d.efc.aref,
            d.efc.frictionloss,
          ],
        )

      limit_tendon = m.tendon_limited_adr.size > 0
      if limit_tendon:
        wp.launch(
          _efc_limit_tendon,
          dim=(d.nworld, m.tendon_limited_adr.size),
          inputs=[
            m.nv,
            m.opt.timestep,
            m.jnt_dofadr,
            m.tendon_adr,
            m.tendon_num,
            m.tendon_limited_adr,
            m.tendon_solref_lim,
            m.tendon_solimp_lim,
            m.tendon_range,
            m.tendon_margin,
            m.tendon_invweight0,
            m.wrap_objid,
            m.wrap_type,
            d.nefc,
            d.qvel,
            d.ten_length,
            d.ten_J,
            refsafe,
          ],
          outputs=[
            d.nl,
            d.efc.worldid,
            d.efc.id,
            d.efc.J,
            d.efc.pos,
            d.efc.margin,
            d.efc.D,
            d.efc.aref,
            d.efc.frictionloss,
          ],
        )

      if limit_ball or limit_slide_hinge or limit_tendon:
        wp.launch(
          _update_nefc,
          dim=(1,),
          inputs=[d.nl],
          outputs=[d.nefc],
        )

    # contact
    if not (m.opt.disableflags & types.DisableBit.CONTACT.value):
      if m.opt.cone == types.ConeType.PYRAMIDAL.value:
        wp.launch(
          _efc_contact_pyramidal,
          dim=(d.nconmax, 2 * (m.condim_max - 1) if m.condim_max > 1 else 1),
          inputs=[
            m.nv,
            m.opt.timestep,
            m.opt.impratio,
            m.body_parentid,
            m.body_rootid,
            m.body_invweight0,
            m.dof_bodyid,
            m.geom_bodyid,
            d.ncon,
            d.qvel,
            d.subtree_com,
            d.cdof,
            refsafe,
            d.contact.dist,
            d.contact.dim,
            d.contact.includemargin,
            d.contact.worldid,
            d.contact.geom,
            d.contact.pos,
            d.contact.frame,
            d.contact.friction,
            d.contact.solref,
            d.contact.solimp,
          ],
          outputs=[
            d.nefc,
            d.efc.worldid,
            d.efc.id,
            d.efc.J,
            d.efc.pos,
            d.efc.margin,
            d.efc.D,
            d.efc.aref,
            d.efc.frictionloss,
          ],
        )
      elif m.opt.cone == types.ConeType.ELLIPTIC.value:
        wp.launch(
          _efc_contact_elliptic,
          dim=(d.nconmax, m.condim_max),
          inputs=[
            m.nv,
            m.opt.timestep,
            m.opt.impratio,
            m.body_parentid,
            m.body_rootid,
            m.body_invweight0,
            m.dof_bodyid,
            m.geom_bodyid,
            d.ncon,
            d.qvel,
            d.subtree_com,
            d.cdof,
            refsafe,
            d.contact.dist,
            d.contact.dim,
            d.contact.includemargin,
            d.contact.worldid,
            d.contact.geom,
            d.contact.pos,
            d.contact.frame,
            d.contact.friction,
            d.contact.solref,
            d.contact.solreffriction,
            d.contact.solimp,
          ],
          outputs=[
            d.nefc,
            d.efc.worldid,
            d.efc.id,
            d.efc.J,
            d.efc.pos,
            d.efc.margin,
            d.efc.D,
            d.efc.aref,
            d.efc.frictionloss,
            d.contact.efc_address,
          ],
        )
