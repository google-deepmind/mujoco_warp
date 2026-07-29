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
"""Differentiable replay of non-contact constraint rows.

The discrete constraint assembly allocates rows and selects active sets.  During
AD, these kernels reuse that fixed row topology and overwrite the continuous
row values with an equivalent differentiable computation.  Gradients therefore
follow a locally constant active set; no derivative is claimed at activation
boundaries.
"""

import warp as wp

from mujoco_warp._src import ad_flags as _ad_flags
from mujoco_warp._src import math
from mujoco_warp._src import types
from mujoco_warp._src.collision_smooth import _smooth_efc_row

# Backward-enabled kernels generate slower forward code, so AD compilation is
# opt-in: off by default, enabled by mjw.enable_ad() / make_diff_data().
wp.set_module_options({"enable_backward": _ad_flags.ad_enabled()})


@wp.kernel
def _equality_joint_to_efc(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  qpos0: wp.array2d[float],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  dof_invweight0: wp.array2d[float],
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_solref: wp.array2d[wp.vec2],
  eq_solimp: wp.array2d[types.vec5],
  eq_data: wp.array2d[types.vec11],
  is_sparse: bool,
  # Data in:
  nefc_in: wp.array[int],
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  # Data out:
  efc_J_out: wp.array3d[float],
  efc_pos_out: wp.array2d[float],
  efc_D_out: wp.array2d[float],
  efc_vel_out: wp.array2d[float],
  efc_aref_out: wp.array2d[float],
):
  """Replay scalar joint-equality rows allocated by make_constraint."""
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid] or efc_type_in[worldid, efcid] != types.ConstraintType.EQUALITY:
    return
  eqid = efc_id_in[worldid, efcid]
  if eq_type[eqid] != types.EqType.JOINT:
    return

  jntid_1 = eq_obj1id[eqid]
  jntid_2 = eq_obj2id[eqid]
  dofadr1 = jnt_dofadr[jntid_1]
  qposadr1 = jnt_qposadr[jntid_1]
  qpos0_id = worldid % qpos0.shape[0]
  invweight_id = worldid % dof_invweight0.shape[0]
  data = eq_data[worldid % eq_data.shape[0], eqid]
  Jqvel = qvel_in[worldid, dofadr1]
  pos = float(0.0)
  invweight = dof_invweight0[invweight_id, dofadr1]
  rowadr = int(0)

  if is_sparse:
    rowadr = efc_J_rowadr_in[worldid, efcid]
    efc_J_out[worldid, 0, rowadr] = 1.0
  else:
    efc_J_out[worldid, efcid, dofadr1] = 1.0

  if jntid_2 > -1:
    dofadr2 = jnt_dofadr[jntid_2]
    qposadr2 = jnt_qposadr[jntid_2]
    dif = qpos_in[worldid, qposadr2] - qpos0[qpos0_id, qposadr2]
    rhs = data[0] + dif * (data[1] + dif * (data[2] + dif * (data[3] + dif * data[4])))
    deriv_2 = data[1] + dif * (2.0 * data[2] + dif * (3.0 * data[3] + dif * 4.0 * data[4]))
    pos = qpos_in[worldid, qposadr1] - qpos0[qpos0_id, qposadr1] - rhs
    J2 = -deriv_2
    Jqvel += J2 * qvel_in[worldid, dofadr2]
    invweight += dof_invweight0[invweight_id, dofadr2]
    if is_sparse:
      efc_J_out[worldid, 0, rowadr + 1] = J2
    else:
      efc_J_out[worldid, efcid, dofadr2] = J2
  else:
    pos = qpos_in[worldid, qposadr1] - qpos0[qpos0_id, qposadr1] - data[0]

  _smooth_efc_row(
    opt_disableflags,
    worldid,
    opt_timestep[worldid % opt_timestep.shape[0]],
    efcid,
    pos,
    pos,
    invweight,
    eq_solref[worldid % eq_solref.shape[0], eqid],
    eq_solimp[worldid % eq_solimp.shape[0], eqid],
    0.0,
    Jqvel,
    efc_pos_out,
    efc_D_out,
    efc_aref_out,
    efc_vel_out,
  )


@wp.kernel
def _limit_tendon_to_efc(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_solref_lim: wp.array2d[wp.vec2],
  tendon_solimp_lim: wp.array2d[types.vec5],
  tendon_range: wp.array2d[wp.vec2],
  tendon_margin: wp.array2d[float],
  tendon_invweight0: wp.array2d[float],
  is_sparse: bool,
  # Data in:
  nefc_in: wp.array[int],
  qvel_in: wp.array2d[float],
  ten_J_in: wp.array2d[float],
  ten_length_in: wp.array2d[float],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  # Data out:
  efc_J_out: wp.array3d[float],
  efc_pos_out: wp.array2d[float],
  efc_D_out: wp.array2d[float],
  efc_vel_out: wp.array2d[float],
  efc_aref_out: wp.array2d[float],
):
  """Replay active tendon-limit rows."""
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid] or efc_type_in[worldid, efcid] != types.ConstraintType.LIMIT_TENDON:
    return

  tenid = efc_id_in[worldid, efcid]
  tenrange = tendon_range[worldid % tendon_range.shape[0], tenid]
  length = ten_length_in[worldid, tenid]
  dist_min = length - tenrange[0]
  dist_max = tenrange[1] - length
  margin = tendon_margin[worldid % tendon_margin.shape[0], tenid]
  pos = wp.min(dist_min, dist_max) - margin
  scl = float(dist_min < dist_max) * 2.0 - 1.0
  rownnz = ten_J_rownnz[tenid]
  ten_rowadr = ten_J_rowadr[tenid]
  efc_rowadr = int(0)
  if is_sparse:
    efc_rowadr = efc_J_rowadr_in[worldid, efcid]
  Jqvel = float(0.0)

  for k in range(rownnz):
    sparseid = ten_rowadr + k
    dofid = ten_J_colind[sparseid]
    J = scl * ten_J_in[worldid, sparseid]
    if is_sparse:
      efc_J_out[worldid, 0, efc_rowadr + k] = J
    else:
      efc_J_out[worldid, efcid, dofid] = J
    Jqvel += J * qvel_in[worldid, dofid]

  _smooth_efc_row(
    opt_disableflags,
    worldid,
    opt_timestep[worldid % opt_timestep.shape[0]],
    efcid,
    pos,
    pos,
    tendon_invweight0[worldid % tendon_invweight0.shape[0], tenid],
    tendon_solref_lim[worldid % tendon_solref_lim.shape[0], tenid],
    tendon_solimp_lim[worldid % tendon_solimp_lim.shape[0], tenid],
    margin,
    Jqvel,
    efc_pos_out,
    efc_D_out,
    efc_aref_out,
    efc_vel_out,
  )


@wp.kernel
def _limit_ball_to_efc(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_solref: wp.array2d[wp.vec2],
  jnt_solimp: wp.array2d[types.vec5],
  jnt_range: wp.array2d[wp.vec2],
  jnt_margin: wp.array2d[float],
  dof_invweight0: wp.array2d[float],
  is_sparse: bool,
  # Data in:
  nefc_in: wp.array[int],
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  # Data out:
  efc_J_out: wp.array3d[float],
  efc_pos_out: wp.array2d[float],
  efc_D_out: wp.array2d[float],
  efc_vel_out: wp.array2d[float],
  efc_aref_out: wp.array2d[float],
):
  """Replay active ball-limit rows, including the quaternion-dependent axis."""
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid] or efc_type_in[worldid, efcid] != types.ConstraintType.LIMIT_JOINT:
    return
  jntid = efc_id_in[worldid, efcid]
  if jnt_type[jntid] != types.JointType.BALL:
    return

  qposadr = jnt_qposadr[jntid]
  dofadr = jnt_dofadr[jntid]
  quat = wp.quat(
    qpos_in[worldid, qposadr],
    qpos_in[worldid, qposadr + 1],
    qpos_in[worldid, qposadr + 2],
    qpos_in[worldid, qposadr + 3],
  )
  axis_angle = math.quat_to_vel(wp.normalize(quat))
  axis, angle = math.normalize_with_norm(axis_angle)
  jntrange = jnt_range[worldid % jnt_range.shape[0], jntid]
  margin = jnt_margin[worldid % jnt_margin.shape[0], jntid]
  pos = wp.max(jntrange[0], jntrange[1]) - angle - margin
  Jqvel = float(0.0)
  rowadr = int(0)
  if is_sparse:
    rowadr = efc_J_rowadr_in[worldid, efcid]

  for k in range(3):
    J = -axis[k]
    if is_sparse:
      efc_J_out[worldid, 0, rowadr + k] = J
    else:
      efc_J_out[worldid, efcid, dofadr + k] = J
    Jqvel += J * qvel_in[worldid, dofadr + k]

  _smooth_efc_row(
    opt_disableflags,
    worldid,
    opt_timestep[worldid % opt_timestep.shape[0]],
    efcid,
    pos,
    pos,
    dof_invweight0[worldid % dof_invweight0.shape[0], dofadr],
    jnt_solref[worldid % jnt_solref.shape[0], jntid],
    jnt_solimp[worldid % jnt_solimp.shape[0], jntid],
    margin,
    Jqvel,
    efc_pos_out,
    efc_D_out,
    efc_aref_out,
    efc_vel_out,
  )


def smooth_noncontact_to_efc(m: types.Model, d: types.Data):
  """Replay supported non-contact rows on a differentiable fixed topology."""
  if d.njmax == 0:
    return

  outputs = [d.efc.J, d.efc.pos, d.efc.D, d.efc.vel, d.efc.aref]

  if m.neq:
    wp.launch(
      _equality_joint_to_efc,
      dim=(d.nworld, d.njmax),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.qpos0,
        m.jnt_qposadr,
        m.jnt_dofadr,
        m.dof_invweight0,
        m.eq_type,
        m.eq_obj1id,
        m.eq_obj2id,
        m.eq_solref,
        m.eq_solimp,
        m.eq_data,
        m.is_sparse,
        d.nefc,
        d.qpos,
        d.qvel,
        d.efc.type,
        d.efc.id,
        d.efc.J_rowadr,
      ],
      outputs=outputs,
    )

  if m.ntendon:
    wp.launch(
      _limit_tendon_to_efc,
      dim=(d.nworld, d.njmax),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_solref_lim,
        m.tendon_solimp_lim,
        m.tendon_range,
        m.tendon_margin,
        m.tendon_invweight0,
        m.is_sparse,
        d.nefc,
        d.qvel,
        d.ten_J,
        d.ten_length,
        d.efc.type,
        d.efc.id,
        d.efc.J_rowadr,
      ],
      outputs=outputs,
    )

  if m.njnt:
    wp.launch(
      _limit_ball_to_efc,
      dim=(d.nworld, d.njmax),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.jnt_type,
        m.jnt_qposadr,
        m.jnt_dofadr,
        m.jnt_solref,
        m.jnt_solimp,
        m.jnt_range,
        m.jnt_margin,
        m.dof_invweight0,
        m.is_sparse,
        d.nefc,
        d.qpos,
        d.qvel,
        d.efc.type,
        d.efc.id,
        d.efc.J_rowadr,
      ],
      outputs=outputs,
    )
