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

from . import forward
from . import math
from .types import BiasType
from .types import Data
from .types import DisableBit
from .types import DynType
from .types import GainType
from .types import JointType
from .types import Model
from .types import TileSet
from .types import vec10f
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import nested_kernel

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _qderiv_actuator_passive_vel(
  # Model:
  actuator_dyntype: wp.array(dtype=int),
  actuator_gaintype: wp.array(dtype=int),
  actuator_biastype: wp.array(dtype=int),
  actuator_actadr: wp.array(dtype=int),
  actuator_actnum: wp.array(dtype=int),
  actuator_gainprm: wp.array2d(dtype=vec10f),
  actuator_biasprm: wp.array2d(dtype=vec10f),
  # Data in:
  act_in: wp.array2d(dtype=float),
  ctrl_in: wp.array2d(dtype=float),
  # Out:
  vel_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  actuator_gainprm_id = worldid % actuator_gainprm.shape[0]
  actuator_biasprm_id = worldid % actuator_biasprm.shape[0]

  if actuator_gaintype[actid] == GainType.AFFINE:
    gain = actuator_gainprm[actuator_gainprm_id, actid][2]
  else:
    gain = 0.0

  if actuator_biastype[actid] == BiasType.AFFINE:
    bias = actuator_biasprm[actuator_biasprm_id, actid][2]
  else:
    bias = 0.0

  if bias == 0.0 and gain == 0.0:
    vel_out[worldid, actid] = 0.0
    return

  vel = float(bias)
  if actuator_dyntype[actid] != DynType.NONE:
    if gain != 0.0:
      act_first = actuator_actadr[actid]
      act_last = act_first + actuator_actnum[actid] - 1
      vel += gain * act_in[worldid, act_last]
  else:
    if gain != 0.0:
      vel += gain * ctrl_in[worldid, actid]

  vel_out[worldid, actid] = vel


@cache_kernel
def _qderiv_actuator_passive_actuation_dense(tile: TileSet, nu: int):
  @nested_kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    vel_in: wp.array3d(dtype=float),
    actuator_moment_in: wp.array3d(dtype=float),
    # In:
    adr: wp.array(dtype=int),
    # Out:
    qDeriv_out: wp.array3d(dtype=float),
  ):
    worldid, nodeid = wp.tid()
    TILE_SIZE = wp.static(tile.size)
    NU = wp.static(nu)

    dofid = adr[nodeid]
    vel_tile = wp.tile_load(vel_in[worldid], shape=(NU, 1), bounds_check=False)
    moment_tile = wp.tile_load(actuator_moment_in[worldid], shape=(NU, TILE_SIZE), offset=(0, dofid), bounds_check=False)
    moment_weighted = wp.tile_map(wp.mul, wp.tile_broadcast(vel_tile, shape=(NU, TILE_SIZE)), moment_tile)
    qderiv_tile = wp.tile_matmul(wp.tile_transpose(moment_tile), moment_weighted)
    wp.tile_store(qDeriv_out[worldid], qderiv_tile, offset=(dofid, dofid), bounds_check=False)

  return kernel


@wp.kernel
def _qderiv_actuator_passive_actuation_sparse(
  # Model:
  nu: int,
  # Data in:
  actuator_moment_in: wp.array3d(dtype=float),
  # In:
  vel_in: wp.array2d(dtype=float),
  qMi: wp.array(dtype=int),
  qMj: wp.array(dtype=int),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  worldid, elemid = wp.tid()

  dofiid = qMi[elemid]
  dofjid = qMj[elemid]
  qderiv_contrib = float(0.0)
  for actid in range(nu):
    vel = vel_in[worldid, actid]
    if vel == 0.0:
      continue

    moment_i = actuator_moment_in[worldid, actid, dofiid]
    moment_j = actuator_moment_in[worldid, actid, dofjid]

    qderiv_contrib += moment_i * moment_j * vel

  qDeriv_out[worldid, 0, elemid] = qderiv_contrib


@wp.kernel
def _qderiv_actuator_passive(
  # Model:
  opt_timestep: wp.array(dtype=float),
  opt_disableflags: int,
  opt_is_sparse: bool,
  dof_damping: wp.array2d(dtype=float),
  # Data in:
  qM_in: wp.array3d(dtype=float),
  # In:
  qMi: wp.array(dtype=int),
  qMj: wp.array(dtype=int),
  qDeriv_in: wp.array3d(dtype=float),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  worldid, elemid = wp.tid()

  dofiid = qMi[elemid]
  dofjid = qMj[elemid]

  if opt_is_sparse:
    qderiv = qDeriv_in[worldid, 0, elemid]
  else:
    qderiv = qDeriv_in[worldid, dofiid, dofjid]

  if not opt_disableflags & DisableBit.DAMPER and dofiid == dofjid:
    qderiv -= dof_damping[worldid % dof_damping.shape[0], dofiid]

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  if opt_is_sparse:
    qDeriv_out[worldid, 0, elemid] = qM_in[worldid, 0, elemid] - qderiv
  else:
    qM = qM_in[worldid, dofiid, dofjid] - qderiv
    qDeriv_out[worldid, dofiid, dofjid] = qM
    if dofiid != dofjid:
      qDeriv_out[worldid, dofjid, dofiid] = qM


# TODO(team): improve performance with tile operations?
@wp.kernel
def _qderiv_tendon_damping(
  # Model:
  ntendon: int,
  opt_timestep: wp.array(dtype=float),
  opt_is_sparse: bool,
  tendon_damping: wp.array2d(dtype=float),
  # Data in:
  ten_J_in: wp.array3d(dtype=float),
  # In:
  qMi: wp.array(dtype=int),
  qMj: wp.array(dtype=int),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  worldid, elemid = wp.tid()
  dofiid = qMi[elemid]
  dofjid = qMj[elemid]

  qderiv = float(0.0)
  tendon_damping_id = worldid % tendon_damping.shape[0]
  for tenid in range(ntendon):
    qderiv -= ten_J_in[worldid, tenid, dofiid] * ten_J_in[worldid, tenid, dofjid] * tendon_damping[tendon_damping_id, tenid]

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  if opt_is_sparse:
    qDeriv_out[worldid, 0, elemid] -= qderiv
  else:
    qDeriv_out[worldid, dofiid, dofjid] -= qderiv
    if dofiid != dofjid:
      qDeriv_out[worldid, dofjid, dofiid] -= qderiv


@event_scope
def deriv_smooth_vel(m: Model, d: Data, out: wp.array2d(dtype=float)):
  """Analytical derivative of smooth forces w.r.t. velocities.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    out: qM - dt * qDeriv (derivatives of smooth forces w.r.t velocities).
  """
  qMi = m.qM_fullm_i
  qMj = m.qM_fullm_j

  # TODO(team): implicit requires different sparsity structure

  if ~(m.opt.disableflags & (DisableBit.ACTUATION | DisableBit.DAMPER)):
    # TODO(team): only clear elements not set by _qderiv_actuator_passive
    out.zero_()
    if m.nu > 0 and not m.opt.disableflags & DisableBit.ACTUATION:
      vel = wp.empty((d.nworld, m.nu), dtype=float)
      wp.launch(
        _qderiv_actuator_passive_vel,
        dim=(d.nworld, m.nu),
        inputs=[
          m.actuator_dyntype,
          m.actuator_gaintype,
          m.actuator_biastype,
          m.actuator_actadr,
          m.actuator_actnum,
          m.actuator_gainprm,
          m.actuator_biasprm,
          d.act,
          d.ctrl,
        ],
        outputs=[vel],
      )
      if m.opt.is_sparse:
        wp.launch(
          _qderiv_actuator_passive_actuation_sparse,
          dim=(d.nworld, qMi.size),
          inputs=[m.nu, d.actuator_moment, vel, qMi, qMj],
          outputs=[out],
        )
      else:
        vel_3d = vel.reshape(vel.shape + (1,))
        for tile in m.qM_tiles:
          wp.launch_tiled(
            _qderiv_actuator_passive_actuation_dense(tile, m.nu),
            dim=(d.nworld, tile.adr.size),
            inputs=[vel_3d, d.actuator_moment, tile.adr],
            outputs=[out],
            block_dim=m.block_dim.mul_m_dense,
          )
    wp.launch(
      _qderiv_actuator_passive,
      dim=(d.nworld, qMi.size),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.opt.is_sparse,
        m.dof_damping,
        d.qM,
        qMi,
        qMj,
        out,
      ],
      outputs=[out],
    )
  else:
    # TODO(team): directly utilize qM for these settings
    wp.copy(out, d.qM)

  if not m.opt.disableflags & DisableBit.DAMPER:
    wp.launch(
      _qderiv_tendon_damping,
      dim=(d.nworld, qMi.size),
      inputs=[m.ntendon, m.opt.timestep, m.opt.is_sparse, m.tendon_damping, d.ten_J, qMi, qMj],
      outputs=[out],
    )

  # TODO(team): rne derivative


@wp.kernel
def _get_state(
  # Model:
  nq: int,
  nv: int,
  na: int,
  # Data in:
  qpos_in: wp.array2d(dtype=float),
  qvel_in: wp.array2d(dtype=float),
  act_in: wp.array2d(dtype=float),
  # Out:
  state_out: wp.array2d(dtype=float),
):
  # get state = [qpos, qvel, act]
  worldid = wp.tid()
  for i in range(nq):
    state_out[worldid, i] = qpos_in[worldid, i]
    if i < nv:
      state_out[worldid, nq + i] = qvel_in[worldid, i]
  for i in range(na):
    state_out[worldid, nq + nv + i] = act_in[worldid, i]


@wp.kernel
def _set_state(
  # Model:
  nq: int,
  nv: int,
  na: int,
  # In:
  state_in: wp.array2d(dtype=float),
  # Data out:
  qpos_out: wp.array2d(dtype=float),
  qvel_out: wp.array2d(dtype=float),
  act_out: wp.array2d(dtype=float),
):
  # set state = [qpos, qvel, act]
  worldid = wp.tid()
  for i in range(nq):
    qpos_out[worldid, i] = state_in[worldid, i]
    if i < nv:
      qvel_out[worldid, i] = state_in[worldid, nq + i]
  for i in range(na):
    act_out[worldid, i] = state_in[worldid, nq + nv + i]


@wp.kernel
def _state_diff(
  # Model:
  nq: int,
  nv: int,
  na: int,
  njnt: int,
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  # In:
  state1_in: wp.array2d(dtype=float),
  state2_in: wp.array2d(dtype=float),
  inv_h: float,
  # Out:
  ds_out: wp.array2d(dtype=float),
):
  # finite difference two state vectors: ds = (s2 - s1) / h
  worldid = wp.tid()

  # position difference via joint type
  for jntid in range(njnt):
    jnttype = jnt_type[jntid]
    qpos_adr = jnt_qposadr[jntid]
    dof_adr = jnt_dofadr[jntid]

    if jnttype == JointType.FREE:
      # linear position difference
      ds_out[worldid, dof_adr + 0] = (state2_in[worldid, qpos_adr + 0] - state1_in[worldid, qpos_adr + 0]) * inv_h
      ds_out[worldid, dof_adr + 1] = (state2_in[worldid, qpos_adr + 1] - state1_in[worldid, qpos_adr + 1]) * inv_h
      ds_out[worldid, dof_adr + 2] = (state2_in[worldid, qpos_adr + 2] - state1_in[worldid, qpos_adr + 2]) * inv_h
      # quaternion difference
      q1 = wp.quat(
        state1_in[worldid, qpos_adr + 3],
        state1_in[worldid, qpos_adr + 4],
        state1_in[worldid, qpos_adr + 5],
        state1_in[worldid, qpos_adr + 6],
      )
      q2 = wp.quat(
        state2_in[worldid, qpos_adr + 3],
        state2_in[worldid, qpos_adr + 4],
        state2_in[worldid, qpos_adr + 5],
        state2_in[worldid, qpos_adr + 6],
      )
      dq = math.quat_sub(q2, q1)
      ds_out[worldid, dof_adr + 3] = dq[0] * inv_h
      ds_out[worldid, dof_adr + 4] = dq[1] * inv_h
      ds_out[worldid, dof_adr + 5] = dq[2] * inv_h
    elif jnttype == JointType.BALL:
      q1 = wp.quat(
        state1_in[worldid, qpos_adr + 0],
        state1_in[worldid, qpos_adr + 1],
        state1_in[worldid, qpos_adr + 2],
        state1_in[worldid, qpos_adr + 3],
      )
      q2 = wp.quat(
        state2_in[worldid, qpos_adr + 0],
        state2_in[worldid, qpos_adr + 1],
        state2_in[worldid, qpos_adr + 2],
        state2_in[worldid, qpos_adr + 3],
      )
      dq = math.quat_sub(q2, q1)
      ds_out[worldid, dof_adr + 0] = dq[0] * inv_h
      ds_out[worldid, dof_adr + 1] = dq[1] * inv_h
      ds_out[worldid, dof_adr + 2] = dq[2] * inv_h
    else:  # SLIDE, HINGE
      ds_out[worldid, dof_adr] = (state2_in[worldid, qpos_adr] - state1_in[worldid, qpos_adr]) * inv_h

  # velocity and activation difference
  for i in range(nv):
    ds_out[worldid, nv + i] = (state2_in[worldid, nq + i] - state1_in[worldid, nq + i]) * inv_h
  for i in range(na):
    ds_out[worldid, 2 * nv + i] = (state2_in[worldid, nq + nv + i] - state1_in[worldid, nq + nv + i]) * inv_h


@wp.kernel
def _perturb_position(
  # Model:
  nq: int,
  njnt: int,
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  # Data in:
  qpos_in: wp.array2d(dtype=float),
  # In:
  dof_idx: int,
  eps: float,
  # Data out:
  qpos_out: wp.array2d(dtype=float),
):
  worldid = wp.tid()

  # copy qpos_in to qpos_out
  for i in range(nq):
    qpos_out[worldid, i] = qpos_in[worldid, i]

  # find joint for this dof and perturb
  for jntid in range(njnt):
    jnttype = jnt_type[jntid]
    qpos_adr = jnt_qposadr[jntid]
    dof_adr = jnt_dofadr[jntid]

    if jnttype == JointType.FREE:
      if dof_idx >= dof_adr and dof_idx < dof_adr + 3:
        qpos_out[worldid, qpos_adr + (dof_idx - dof_adr)] += eps
      elif dof_idx >= dof_adr + 3 and dof_idx < dof_adr + 6:
        q = wp.quat(
          qpos_in[worldid, qpos_adr + 3],
          qpos_in[worldid, qpos_adr + 4],
          qpos_in[worldid, qpos_adr + 5],
          qpos_in[worldid, qpos_adr + 6],
        )
        local_idx = dof_idx - dof_adr - 3
        if local_idx == 0:
          v = wp.vec3(1.0, 0.0, 0.0)
        elif local_idx == 1:
          v = wp.vec3(0.0, 1.0, 0.0)
        else:
          v = wp.vec3(0.0, 0.0, 1.0)
        q_new = math.quat_integrate(q, v, eps)
        qpos_out[worldid, qpos_adr + 3] = q_new[0]
        qpos_out[worldid, qpos_adr + 4] = q_new[1]
        qpos_out[worldid, qpos_adr + 5] = q_new[2]
        qpos_out[worldid, qpos_adr + 6] = q_new[3]
    elif jnttype == JointType.BALL:
      if dof_idx >= dof_adr and dof_idx < dof_adr + 3:
        q = wp.quat(
          qpos_in[worldid, qpos_adr + 0],
          qpos_in[worldid, qpos_adr + 1],
          qpos_in[worldid, qpos_adr + 2],
          qpos_in[worldid, qpos_adr + 3],
        )
        local_idx = dof_idx - dof_adr
        if local_idx == 0:
          v = wp.vec3(1.0, 0.0, 0.0)
        elif local_idx == 1:
          v = wp.vec3(0.0, 1.0, 0.0)
        else:
          v = wp.vec3(0.0, 0.0, 1.0)
        q_new = math.quat_integrate(q, v, eps)
        qpos_out[worldid, qpos_adr + 0] = q_new[0]
        qpos_out[worldid, qpos_adr + 1] = q_new[1]
        qpos_out[worldid, qpos_adr + 2] = q_new[2]
        qpos_out[worldid, qpos_adr + 3] = q_new[3]
    else:  # SLIDE, HINGE
      if dof_idx == dof_adr:
        qpos_out[worldid, qpos_adr] += eps


@wp.kernel
def _perturb_array(
  # In:
  idx: int,
  eps: float,
  arr_in: wp.array2d(dtype=float),
  # Out:
  arr_out: wp.array2d(dtype=float),
):
  worldid = wp.tid()
  for i in range(arr_in.shape[1]):
    if i == idx:
      arr_out[worldid, i] = arr_in[worldid, i] + eps
    else:
      arr_out[worldid, i] = arr_in[worldid, i]


@wp.kernel
def _diff_vectors(
  # In:
  x1_in: wp.array2d(dtype=float),
  x2_in: wp.array2d(dtype=float),
  inv_h: float,
  n: int,
  # Out:
  dx_out: wp.array2d(dtype=float),
):
  # dx = (x2 - x1) / h
  worldid = wp.tid()
  for i in range(n):
    dx_out[worldid, i] = (x2_in[worldid, i] - x1_in[worldid, i]) * inv_h


@wp.kernel
def _copy_to_jacobian_col(
  # In:
  col_in: wp.array2d(dtype=float),
  col_idx: int,
  nrow: int,
  # Out:
  jac_out: wp.array3d(dtype=float),
):
  worldid = wp.tid()
  for i in range(nrow):
    jac_out[worldid, i, col_idx] = col_in[worldid, i]


@event_scope
def transition_fd(
  m: Model,
  d: Data,
  eps: float,
  centered: bool = False,
  A: wp.array3d(dtype=float) = None,
  B: wp.array3d(dtype=float) = None,
  C: wp.array3d(dtype=float) = None,
  D: wp.array3d(dtype=float) = None,
):
  """Finite differenced transition matrices (control theory notation).

  Computes: d(x_next) = A*dx + B*du, d(sensor) = C*dx + D*du
  where x = [qvel_diff, qvel, act] is the state in tangent space.

  Args:
    m: model
    d: data
    eps: finite difference epsilon
    centered: if True, use centered differences
    A: output state transition matrix (nworld, ndx, ndx) where ndx = 2*nv+na
    B: output control transition matrix (nworld, ndx, nu)
    C: output state observation matrix (nworld, nsensordata, ndx)
    D: output control observation matrix (nworld, nsensordata, nu)
  """
  # TODO(team): add option for scratch memory

  nq, nv, na, nu = m.nq, m.nv, m.na, m.nu
  ns = m.nsensordata
  ndx = 2 * nv + na
  nworld = d.nworld

  # skip sensor computations if not requested
  skip_sensor = C is None and D is None

  # save current state
  state_size = nq + nv + na
  state0 = wp.zeros((nworld, state_size), dtype=float)
  ctrl0 = wp.zeros((nworld, nu), dtype=float) if nu > 0 else None
  wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[state0])
  if nu > 0:
    wp.copy(ctrl0, d.ctrl)

  # baseline step
  forward.step(m, d)

  # save baseline next state and sensors
  next_state = wp.zeros((nworld, state_size), dtype=float)
  wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_state])
  sensor0 = None
  if not skip_sensor:
    sensor0 = wp.zeros((nworld, ns), dtype=float)
    wp.copy(sensor0, d.sensordata)

  # restore state
  wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
  if nu > 0:
    wp.copy(d.ctrl, ctrl0)

  # allocate work arrays
  next_plus = wp.zeros((nworld, state_size), dtype=float)
  next_minus = wp.zeros((nworld, state_size), dtype=float) if centered else None
  ds = wp.zeros((nworld, ndx), dtype=float)
  sensor_plus = wp.zeros((nworld, ns), dtype=float) if not skip_sensor else None
  sensor_minus = wp.zeros((nworld, ns), dtype=float) if not skip_sensor and centered else None
  dsensor = wp.zeros((nworld, ns), dtype=float) if not skip_sensor else None

  inv_eps = 1.0 / eps
  inv_2eps = 1.0 / (2.0 * eps) if centered else inv_eps

  # finite difference controls
  if (B is not None or D is not None) and nu > 0:
    ctrl_temp = wp.zeros((nworld, nu), dtype=float)
    for i in range(nu):
      # nudge forward
      wp.launch(_perturb_array, dim=nworld, inputs=[i, eps, ctrl0], outputs=[ctrl_temp])
      wp.copy(d.ctrl, ctrl_temp)
      forward.step(m, d)
      wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_plus])
      if not skip_sensor:
        wp.copy(sensor_plus, d.sensordata)

      # restore and nudge backward if centered
      wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
      wp.copy(d.ctrl, ctrl0)

      if centered:
        wp.launch(_perturb_array, dim=nworld, inputs=[i, -eps, ctrl0], outputs=[ctrl_temp])
        wp.copy(d.ctrl, ctrl_temp)
        forward.step(m, d)
        wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_minus])
        if not skip_sensor:
          wp.copy(sensor_minus, d.sensordata)
        wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
        wp.copy(d.ctrl, ctrl0)

      # compute derivatives
      if B is not None:
        if centered:
          wp.launch(
            _state_diff,
            dim=nworld,
            inputs=[nq, nv, na, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, next_minus, next_plus, inv_2eps],
            outputs=[ds],
          )
        else:
          wp.launch(
            _state_diff,
            dim=nworld,
            inputs=[nq, nv, na, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, next_state, next_plus, inv_eps],
            outputs=[ds],
          )
        wp.launch(_copy_to_jacobian_col, dim=nworld, inputs=[ds, i, ndx], outputs=[B])

      if D is not None:
        if centered:
          wp.launch(_diff_vectors, dim=nworld, inputs=[sensor_plus, sensor_minus, inv_2eps, ns], outputs=[dsensor])
        else:
          wp.launch(_diff_vectors, dim=nworld, inputs=[sensor0, sensor_plus, inv_eps, ns], outputs=[dsensor])
        wp.launch(_copy_to_jacobian_col, dim=nworld, inputs=[dsensor, i, ns], outputs=[D])

  # finite difference activations
  if (A is not None or C is not None) and na > 0:
    act0 = wp.zeros((nworld, na), dtype=float)
    wp.copy(act0, d.act)
    act_temp = wp.zeros((nworld, na), dtype=float)
    for i in range(na):
      # nudge forward
      wp.launch(_perturb_array, dim=nworld, inputs=[i, eps, act0], outputs=[act_temp])
      wp.copy(d.act, act_temp)
      forward.step(m, d)
      wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_plus])
      if not skip_sensor:
        wp.copy(sensor_plus, d.sensordata)

      # restore and nudge backward if centered
      wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
      if nu > 0:
        wp.copy(d.ctrl, ctrl0)

      if centered:
        wp.launch(_perturb_array, dim=nworld, inputs=[i, -eps, act0], outputs=[act_temp])
        wp.copy(d.act, act_temp)
        forward.step(m, d)
        wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_minus])
        if not skip_sensor:
          wp.copy(sensor_minus, d.sensordata)
        wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
        if nu > 0:
          wp.copy(d.ctrl, ctrl0)

      # compute derivatives
      col_idx = 2 * nv + i
      if A is not None:
        if centered:
          wp.launch(
            _state_diff,
            dim=nworld,
            inputs=[nq, nv, na, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, next_minus, next_plus, inv_2eps],
            outputs=[ds],
          )
        else:
          wp.launch(
            _state_diff,
            dim=nworld,
            inputs=[nq, nv, na, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, next_state, next_plus, inv_eps],
            outputs=[ds],
          )
        wp.launch(_copy_to_jacobian_col, dim=nworld, inputs=[ds, col_idx, ndx], outputs=[A])

      if C is not None:
        if centered:
          wp.launch(_diff_vectors, dim=nworld, inputs=[sensor_minus, sensor_plus, inv_2eps, ns], outputs=[dsensor])
        else:
          wp.launch(_diff_vectors, dim=nworld, inputs=[sensor0, sensor_plus, inv_eps, ns], outputs=[dsensor])
        wp.launch(_copy_to_jacobian_col, dim=nworld, inputs=[dsensor, col_idx, ns], outputs=[C])

  # finite difference velocities
  if A is not None or C is not None:
    qvel0 = wp.zeros((nworld, nv), dtype=float)
    wp.copy(qvel0, d.qvel)
    qvel_temp = wp.zeros((nworld, nv), dtype=float)
    for i in range(nv):
      # nudge forward
      wp.launch(_perturb_array, dim=nworld, inputs=[i, eps, qvel0], outputs=[qvel_temp])
      wp.copy(d.qvel, qvel_temp)
      forward.step(m, d)
      wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_plus])
      if not skip_sensor:
        wp.copy(sensor_plus, d.sensordata)

      # restore and nudge backward if centered
      wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
      if nu > 0:
        wp.copy(d.ctrl, ctrl0)

      if centered:
        wp.launch(_perturb_array, dim=nworld, inputs=[i, -eps, qvel0], outputs=[qvel_temp])
        wp.copy(d.qvel, qvel_temp)
        forward.step(m, d)
        wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_minus])
        if not skip_sensor:
          wp.copy(sensor_minus, d.sensordata)
        wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
        if nu > 0:
          wp.copy(d.ctrl, ctrl0)

      # compute derivatives
      col_idx = nv + i
      if A is not None:
        if centered:
          wp.launch(
            _state_diff,
            dim=nworld,
            inputs=[nq, nv, na, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, next_minus, next_plus, inv_2eps],
            outputs=[ds],
          )
        else:
          wp.launch(
            _state_diff,
            dim=nworld,
            inputs=[nq, nv, na, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, next_state, next_plus, inv_eps],
            outputs=[ds],
          )
        wp.launch(_copy_to_jacobian_col, dim=nworld, inputs=[ds, col_idx, ndx], outputs=[A])

      if C is not None:
        if centered:
          wp.launch(_diff_vectors, dim=nworld, inputs=[sensor_minus, sensor_plus, inv_2eps, ns], outputs=[dsensor])
        else:
          wp.launch(_diff_vectors, dim=nworld, inputs=[sensor0, sensor_plus, inv_eps, ns], outputs=[dsensor])
        wp.launch(_copy_to_jacobian_col, dim=nworld, inputs=[dsensor, col_idx, ns], outputs=[C])

  # finite difference positions
  if A is not None or C is not None:
    qpos_perturbed = wp.zeros((nworld, nq), dtype=float)
    for i in range(nv):
      # nudge position forward
      wp.launch(
        _perturb_position,
        dim=nworld,
        inputs=[nq, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, i, eps],
        outputs=[qpos_perturbed],
      )
      wp.copy(d.qpos, qpos_perturbed)
      forward.step(m, d)
      wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_plus])
      if not skip_sensor:
        wp.copy(sensor_plus, d.sensordata)

      # restore and nudge backward if centered
      wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
      if nu > 0:
        wp.copy(d.ctrl, ctrl0)

      if centered:
        wp.launch(
          _perturb_position,
          dim=nworld,
          inputs=[nq, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, i, -eps],
          outputs=[qpos_perturbed],
        )
        wp.copy(d.qpos, qpos_perturbed)
        forward.step(m, d)
        wp.launch(_get_state, dim=nworld, inputs=[nq, nv, na, d.qpos, d.qvel, d.act], outputs=[next_minus])
        if not skip_sensor:
          wp.copy(sensor_minus, d.sensordata)
        wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
        if nu > 0:
          wp.copy(d.ctrl, ctrl0)

      # compute derivatives
      col_idx = i
      if A is not None:
        if centered:
          wp.launch(
            _state_diff,
            dim=nworld,
            inputs=[nq, nv, na, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, next_minus, next_plus, inv_2eps],
            outputs=[ds],
          )
        else:
          wp.launch(
            _state_diff,
            dim=nworld,
            inputs=[nq, nv, na, m.njnt, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, next_state, next_plus, inv_eps],
            outputs=[ds],
          )
        wp.launch(_copy_to_jacobian_col, dim=nworld, inputs=[ds, col_idx, ndx], outputs=[A])

      if C is not None:
        if centered:
          wp.launch(_diff_vectors, dim=nworld, inputs=[sensor_minus, sensor_plus, inv_2eps, ns], outputs=[dsensor])
        else:
          wp.launch(_diff_vectors, dim=nworld, inputs=[sensor0, sensor_plus, inv_eps, ns], outputs=[dsensor])
        wp.launch(_copy_to_jacobian_col, dim=nworld, inputs=[dsensor, col_idx, ns], outputs=[C])

  # restore final state
  wp.launch(_set_state, dim=nworld, inputs=[nq, nv, na, state0], outputs=[d.qpos, d.qvel, d.act])
  if nu > 0:
    wp.copy(d.ctrl, ctrl0)
