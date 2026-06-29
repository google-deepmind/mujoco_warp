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

from mujoco_warp._src import math
from mujoco_warp._src import util_misc
from mujoco_warp._src.support import next_act
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import vec10f

wp.set_module_options({"enable_backward": False})


@wp.func
def next_position(
  jnttype: int,
  qpos_adr: int,
  timestep: float,
  qpos_in: wp.array2d[float],
  worldid: int,
  # In:
  qvel_lin: wp.vec3,
  qvel_ang: wp.vec3,
  # Data out:
  qpos_out: wp.array2d[float],
):
  """Per-joint semi-implicit position integration from velocity values."""
  if jnttype == JointType.FREE:
    qpos_pos = wp.vec3(qpos_in[worldid, qpos_adr], qpos_in[worldid, qpos_adr + 1], qpos_in[worldid, qpos_adr + 2])
    qpos_new = qpos_pos + timestep * qvel_lin

    qpos_quat = wp.quat(
      qpos_in[worldid, qpos_adr + 3],
      qpos_in[worldid, qpos_adr + 4],
      qpos_in[worldid, qpos_adr + 5],
      qpos_in[worldid, qpos_adr + 6],
    )
    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

    qpos_out[worldid, qpos_adr + 0] = qpos_new[0]
    qpos_out[worldid, qpos_adr + 1] = qpos_new[1]
    qpos_out[worldid, qpos_adr + 2] = qpos_new[2]
    qpos_out[worldid, qpos_adr + 3] = qpos_quat_new[0]
    qpos_out[worldid, qpos_adr + 4] = qpos_quat_new[1]
    qpos_out[worldid, qpos_adr + 5] = qpos_quat_new[2]
    qpos_out[worldid, qpos_adr + 6] = qpos_quat_new[3]

  elif jnttype == JointType.BALL:
    qpos_quat = wp.quat(
      qpos_in[worldid, qpos_adr + 0],
      qpos_in[worldid, qpos_adr + 1],
      qpos_in[worldid, qpos_adr + 2],
      qpos_in[worldid, qpos_adr + 3],
    )
    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

    qpos_out[worldid, qpos_adr + 0] = qpos_quat_new[0]
    qpos_out[worldid, qpos_adr + 1] = qpos_quat_new[1]
    qpos_out[worldid, qpos_adr + 2] = qpos_quat_new[2]
    qpos_out[worldid, qpos_adr + 3] = qpos_quat_new[3]

  else:  # HINGE / SLIDE
    qpos_out[worldid, qpos_adr] = qpos_in[worldid, qpos_adr] + timestep * qvel_lin[0]


@wp.func
def next_velocity(
  worldid: int,
  dofid: int,
  # Model:
  opt_timestep: wp.array[float],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # In:
  qacc_scale_in: float,
) -> float:
  # returns the updated velocity (not written) so the backward integrator can hold it as a local
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  return qvel_in[worldid, dofid] + qacc_scale_in * qacc_in[worldid, dofid] * timestep


@wp.func
def next_activation(
  worldid: int,
  uid: int,
  # Model:
  opt_timestep: wp.array[float],
  actuator_dyntype: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_actnum: wp.array[int],
  actuator_actlimited: wp.array[bool],
  actuator_dynprm: wp.array2d[vec10f],
  actuator_gainprm: wp.array2d[vec10f],
  actuator_biasprm: wp.array2d[vec10f],
  actuator_actrange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_velocity_in: wp.array2d[float],
  # In:
  act_dot_scale: float,
  limit: bool,
  # Data out:
  act_out: wp.array2d[float],
):
  opt_timestep_id = worldid % opt_timestep.shape[0]
  actuator_dynprm_id = worldid % actuator_dynprm.shape[0]
  actuator_actrange_id = worldid % actuator_actrange.shape[0]
  actuator_gainprm_id = worldid % actuator_gainprm.shape[0]
  actuator_biasprm_id = worldid % actuator_biasprm.shape[0]

  actadr = actuator_actadr[uid]
  actnum = actuator_actnum[uid]
  dyntype = actuator_dyntype[uid]

  if dyntype == DynType.DCMOTOR:
    dynprm = actuator_dynprm[actuator_dynprm_id, uid]
    gainprm = actuator_gainprm[actuator_gainprm_id, uid]
    biasprm = actuator_biasprm[actuator_biasprm_id, uid]
    slots = util_misc.dcmotor_slots(dynprm, gainprm)

    for j in range(actadr, actadr + actnum):
      offset = j - actadr
      act = act_in[worldid, j]
      act_dot = act_dot_in[worldid, j]

      if offset == slots[4]:  # current
        R = gainprm[0]
        te = wp.max(MJ_MINVAL, dynprm[0])
        act = act + act_dot * te * (1.0 - wp.exp(-opt_timestep[opt_timestep_id] / te))
      elif offset == slots[3]:  # bristle
        F_C = biasprm[3]
        F_S = biasprm[4]
        v_S = biasprm[5]
        sigma0 = dynprm[5]
        velocity = actuator_velocity_in[worldid, uid]
        g = util_misc.lugre_stribeck(velocity, F_C, F_S, v_S)

        a = -sigma0 * wp.abs(velocity) / wp.max(MJ_MINVAL, g)
        h = opt_timestep[opt_timestep_id]
        exp_ah = wp.exp(a * h)
        int_h = h
        if wp.abs(a) > MJ_MINVAL:
          int_h = (exp_ah - 1.0) / a
        act = exp_ah * act + int_h * velocity
      elif offset == slots[1]:  # integral
        act = act + act_dot * opt_timestep[opt_timestep_id]
        Imax = dynprm[8]
        if Imax > 0.0:
          act = wp.clamp(act, -Imax, Imax)
      else:  # temperature and slew
        act = act + act_dot * opt_timestep[opt_timestep_id]

      act_out[worldid, j] = act
  else:
    for j in range(actadr, actadr + actnum):
      act = next_act(
        opt_timestep[opt_timestep_id],
        dyntype,
        actuator_dynprm[actuator_dynprm_id, uid],
        actuator_actrange[actuator_actrange_id, uid],
        act_in[worldid, j],
        act_dot_in[worldid, j],
        act_dot_scale,
        limit and actuator_actlimited[uid],
      )
      act_out[worldid, j] = act


@wp.func
def next_time(
  worldid: int,
  # Model:
  opt_timestep: wp.array[float],
  # Data in:
  time_in: wp.array[float],
  # Data out:
  time_out: wp.array[float],
):
  time_out[worldid] = time_in[worldid] + opt_timestep[worldid % opt_timestep.shape[0]]
