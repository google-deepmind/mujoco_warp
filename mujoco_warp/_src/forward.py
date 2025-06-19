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

from typing import Optional

import warp as wp

from . import collision_driver
from . import constraint
from . import derivative
from . import math
from . import passive
from . import sensor
from . import smooth
from . import solver
from . import util_misc
from .support import xfrc_accumulate
from .types import MJ_MINVAL
from .types import BiasType
from .types import Data
from .types import DisableBit
from .types import DynType
from .types import EnableBit
from .types import GainType
from .types import IntegratorType
from .types import JointType
from .types import Model
from .types import TileSet
from .types import TrnType
from .types import vec10f
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})

# RK4 tableau
_RK4_A = [
  [0.5, 0.0, 0.0],
  [0.0, 0.5, 0.0],
  [0.0, 0.0, 1.0],
]
_RK4_B = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]


@wp.kernel
def _next_position(
  # Model:
  opt_timestep: wp.array(dtype=float),
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  # Data in:
  qpos_in: wp.array2d(dtype=float),
  qvel_in: wp.array2d(dtype=float),
  # In:
  qvel_scale_in: float,
  # Data out:
  qpos_out: wp.array2d(dtype=float),
):
  worldid, jntid = wp.tid()
  timestep = opt_timestep[worldid]

  jnttype = jnt_type[jntid]
  qpos_adr = jnt_qposadr[jntid]
  dof_adr = jnt_dofadr[jntid]
  qpos = qpos_in[worldid]
  qpos_next = qpos_out[worldid]
  qvel = qvel_in[worldid]

  if jnttype == wp.static(JointType.FREE.value):
    qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])
    qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in

    qpos_new = qpos_pos + timestep * qvel_lin

    qpos_quat = wp.quat(
      qpos[qpos_adr + 3],
      qpos[qpos_adr + 4],
      qpos[qpos_adr + 5],
      qpos[qpos_adr + 6],
    )
    qvel_ang = wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5]) * qvel_scale_in

    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

    qpos_next[qpos_adr + 0] = qpos_new[0]
    qpos_next[qpos_adr + 1] = qpos_new[1]
    qpos_next[qpos_adr + 2] = qpos_new[2]
    qpos_next[qpos_adr + 3] = qpos_quat_new[0]
    qpos_next[qpos_adr + 4] = qpos_quat_new[1]
    qpos_next[qpos_adr + 5] = qpos_quat_new[2]
    qpos_next[qpos_adr + 6] = qpos_quat_new[3]

  elif jnttype == wp.static(JointType.BALL.value):
    qpos_quat = wp.quat(
      qpos[qpos_adr + 0],
      qpos[qpos_adr + 1],
      qpos[qpos_adr + 2],
      qpos[qpos_adr + 3],
    )
    qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale_in

    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)

    qpos_next[qpos_adr + 0] = qpos_quat_new[0]
    qpos_next[qpos_adr + 1] = qpos_quat_new[1]
    qpos_next[qpos_adr + 2] = qpos_quat_new[2]
    qpos_next[qpos_adr + 3] = qpos_quat_new[3]

  else:  # if jnt_type in (JointType.HINGE, JointType.SLIDE):
    qpos_next[qpos_adr] = qpos[qpos_adr] + timestep * qvel[dof_adr] * qvel_scale_in


@wp.kernel
def _next_velocity(
  # Model:
  opt_timestep: wp.array(dtype=float),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qacc_in: wp.array2d(dtype=float),
  # In:
  qacc_scale_in: float,
  # Data out:
  qvel_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  timestep = opt_timestep[worldid]
  qvel_out[worldid, dofid] = qvel_in[worldid, dofid] + qacc_scale_in * qacc_in[worldid, dofid] * timestep


@wp.kernel
def _next_activation(
  # Model:
  opt_timestep: wp.array(dtype=float),
  actuator_dyntype: wp.array(dtype=int),
  actuator_actlimited: wp.array(dtype=bool),
  actuator_dynprm: wp.array2d(dtype=vec10f),
  actuator_actrange: wp.array2d(dtype=wp.vec2),
  # Data in:
  act_in: wp.array2d(dtype=float),
  act_dot_in: wp.array2d(dtype=float),
  # In:
  act_dot_scale_in: float,
  limit: bool,
  # Data out:
  act_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()
  timestep = opt_timestep[worldid]

  act = act_in[worldid, actid]
  act_dot = act_dot_in[worldid, actid]

  # advance the actuation
  if actuator_dyntype[actid] == wp.static(DynType.FILTEREXACT.value):
    dyn_prm = actuator_dynprm[worldid, actid]
    tau = wp.max(MJ_MINVAL, dyn_prm[0])
    act += act_dot_scale_in * act_dot * tau * (1.0 - wp.exp(-timestep / tau))
  else:
    act += act_dot_scale_in * act_dot * timestep

  # clamp to actrange
  if limit and actuator_actlimited[actid]:
    actrange = actuator_actrange[worldid, actid]
    act = wp.clamp(act, actrange[0], actrange[1])

  act_out[worldid, actid] = act


@wp.kernel
def _next_time(
  # Model:
  opt_timestep: wp.array(dtype=float),
  # Data in:
  nconmax_in: int,
  njmax_in: int,
  ncon_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  time_in: wp.array(dtype=float),
  ncollision_in: wp.array(dtype=int),
  # Data out:
  time_out: wp.array(dtype=float),
):
  worldid = wp.tid()
  time_out[worldid] = time_in[worldid] + opt_timestep[worldid]

  if worldid == 0:
    if nefc_in[0] > njmax_in:
      wp.printf("nefc overflow - please increase njmax to %u\n", nefc_in[0])

    if ncollision_in[0] > nconmax_in:
      wp.printf("ncollision overflow - please increase nconmax to %u\n", ncollision_in[0])

    if ncon_in[0] > nconmax_in:
      wp.printf("ncon overflow - please increase nconmax to %u\n", ncon_in[0])


def _advance(m: Model, d: Data, qacc: wp.array, qvel: Optional[wp.array] = None):
  """Advance state and time given activation derivatives and acceleration."""

  # TODO(team): can we assume static timesteps?

  # advance activations
  if m.na:
    wp.launch(
      _next_activation,
      dim=(d.nworld, m.na),
      inputs=[
        m.opt.timestep,
        m.actuator_dyntype,
        m.actuator_actlimited,
        m.actuator_dynprm,
        m.actuator_actrange,
        d.act,
        d.act_dot,
        1.0,
        True,
      ],
      outputs=[
        d.act,
      ],
    )

  wp.launch(
    _next_velocity,
    dim=(d.nworld, m.nv),
    inputs=[
      m.opt.timestep,
      d.qvel,
      qacc,
      1.0,
    ],
    outputs=[
      d.qvel,
    ],
  )

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  if qvel is not None:
    qvel_in = qvel
  else:
    qvel_in = d.qvel

  wp.launch(
    _next_position,
    dim=(d.nworld, m.njnt),
    inputs=[
      m.opt.timestep,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      d.qpos,
      qvel_in,
      1.0,
    ],
    outputs=[
      d.qpos,
    ],
  )

  wp.launch(
    _next_time,
    dim=(d.nworld,),
    inputs=[
      m.opt.timestep,
      d.nconmax,
      d.njmax,
      d.ncon,
      d.nefc,
      d.time,
      d.ncollision,
    ],
    outputs=[
      d.time,
    ],
  )


@wp.kernel
def _euler_damp_qfrc_sparse(
  # Model:
  opt_timestep: wp.array(dtype=float),
  dof_Madr: wp.array(dtype=int),
  dof_damping: wp.array2d(dtype=float),
  # Data in:
  qfrc_smooth_in: wp.array2d(dtype=float),
  qfrc_constraint_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_integration_out: wp.array2d(dtype=float),
  qM_integration_out: wp.array3d(dtype=float),
):
  worldid, tid = wp.tid()
  timestep = opt_timestep[worldid]

  adr = dof_Madr[tid]
  qM_integration_out[worldid, 0, adr] += timestep * dof_damping[worldid, tid]
  qfrc_integration_out[worldid, tid] = qfrc_smooth_in[worldid, tid] + qfrc_constraint_in[worldid, tid]


def _euler_sparse(m: Model, d: Data):
  wp.copy(d.qM_integration, d.qM)
  wp.launch(
    _euler_damp_qfrc_sparse,
    dim=(d.nworld, m.nv),
    inputs=[
      m.opt.timestep,
      m.dof_Madr,
      m.dof_damping,
      d.qfrc_smooth,
      d.qfrc_constraint,
    ],
    outputs=[
      d.qfrc_integration,
      d.qM_integration,
    ],
  )
  smooth.factor_solve_i(
    m,
    d,
    d.qM_integration,
    d.qLD_integration,
    d.qLDiagInv_integration,
    d.qacc_integration,
    d.qfrc_integration,
  )


@cache_kernel
def _tile_euler_dense(tile: TileSet):
  @nested_kernel
  def euler_dense(
    # Model:
    dof_damping: wp.array2d(dtype=float),
    opt_timestep: wp.array(dtype=float),
    # Data in:
    qM_in: wp.array3d(dtype=float),
    qfrc_smooth_in: wp.array2d(dtype=float),
    qfrc_constraint_in: wp.array2d(dtype=float),
    # In:
    adr_in: wp.array(dtype=int),
    # Data out:
    qacc_integration_out: wp.array2d(dtype=float),
  ):
    worldid, nodeid = wp.tid()
    timestep = opt_timestep[worldid]
    TILE_SIZE = wp.static(tile.size)

    dofid = adr_in[nodeid]
    M_tile = wp.tile_load(qM_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(dofid, dofid))
    damping_tile = wp.tile_load(dof_damping[worldid], shape=(TILE_SIZE,), offset=(dofid,))
    damping_scaled = damping_tile * timestep
    qm_integration_tile = wp.tile_diag_add(M_tile, damping_scaled)

    qfrc_smooth_tile = wp.tile_load(qfrc_smooth_in[worldid], shape=(TILE_SIZE,), offset=(dofid,))
    qfrc_constraint_tile = wp.tile_load(qfrc_constraint_in[worldid], shape=(TILE_SIZE,), offset=(dofid,))

    qfrc_tile = qfrc_smooth_tile + qfrc_constraint_tile

    L_tile = wp.tile_cholesky(qm_integration_tile)
    qacc_tile = wp.tile_cholesky_solve(L_tile, qfrc_tile)
    wp.tile_store(qacc_integration_out[worldid], qacc_tile, offset=(dofid))

  return euler_dense


@event_scope
def euler(m: Model, d: Data):
  """Euler integrator, semi-implicit in velocity."""

  # integrate damping implicitly
  if not m.opt.disableflags & DisableBit.EULERDAMP.value:
    if m.opt.is_sparse:
      _euler_sparse(m, d)
    else:
      for tile in m.qM_tiles:
        wp.launch_tiled(
          _tile_euler_dense(tile),
          dim=(d.nworld, tile.adr.size),
          inputs=[m.dof_damping, m.opt.timestep, d.qM, d.qfrc_smooth, d.qfrc_constraint, tile.adr],
          outputs=[d.qacc_integration],
          block_dim=m.block_dim.euler_dense,
        )

    _advance(m, d, d.qacc_integration)
  else:
    _advance(m, d, d.qacc)


def _rk_perturb_state(m: Model, d: Data, scale: float):
  # position
  wp.launch(
    _next_position,
    dim=(d.nworld, m.njnt),
    inputs=[m.opt.timestep, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos_t0, d.qvel, scale],
    outputs=[d.qpos],
  )

  # velocity
  wp.launch(
    _next_velocity,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.timestep, d.qvel_t0, d.qacc, scale],
    outputs=[d.qvel],
  )

  # activation
  if m.na:
    wp.launch(
      _next_activation,
      dim=(d.nworld, m.na),
      inputs=[m.opt.timestep, d.act_t0, d.act_dot, scale, False],
      outputs=[d.act],
    )


@wp.kernel
def _rk_accumulate_velocity_acceleration(
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qacc_in: wp.array2d(dtype=float),
  # In:
  scale: float,
  # Data out:
  qvel_out: wp.array2d(dtype=float),
  qacc_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qvel_out[worldid, dofid] += scale * qvel_in[worldid, dofid]
  qacc_out[worldid, dofid] += scale * qacc_in[worldid, dofid]


@wp.kernel
def _rk_accumulate_activation_velocity(
  # Data in:
  act_dot_in: wp.array2d(dtype=float),
  # In:
  scale: float,
  # Data out:
  act_dot_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()
  act_dot_out[worldid, actid] += scale * act_dot_in[worldid, actid]


def _rk_accumulate(m: Model, d: Data, scale: float):
  """Computes one term of 1/6 k_1 + 1/3 k_2 + 1/3 k_3 + 1/6 k_4"""

  wp.launch(
    _rk_accumulate_velocity_acceleration,
    dim=(d.nworld, m.nv),
    inputs=[d.qvel, d.qacc, scale],
    outputs=[d.qvel_rk, d.qacc_rk],
  )

  if m.na:
    wp.launch(
      _rk_accumulate_activation_velocity,
      dim=(d.nworld, m.na),
      inputs=[d.act_dot, scale],
      outputs=[d.act_dot_rk],
    )


@event_scope
def rungekutta4(m: Model, d: Data):
  """Runge-Kutta explicit order 4 integrator."""

  wp.copy(d.qpos_t0, d.qpos)
  wp.copy(d.qvel_t0, d.qvel)

  d.qvel_rk.zero_()
  d.qacc_rk.zero_()
  d.act_dot_rk.zero_()

  if m.na:
    wp.copy(d.act_t0, d.act)

  A, B = _RK4_A, _RK4_B

  _rk_accumulate(m, d, B[0])
  for i in range(3):
    a, b = float(A[i][i]), B[i + 1]
    _rk_perturb_state(m, d, a)
    forward(m, d)
    _rk_accumulate(m, d, b)

  wp.copy(d.qpos, d.qpos_t0)
  wp.copy(d.qvel, d.qvel_t0)
  if m.na:
    wp.copy(d.act, d.act_t0)
    wp.copy(d.act_dot, d.act_dot_rk)
  _advance(m, d, d.qacc_rk, d.qvel_rk)


@event_scope
def implicit(m: Model, d: Data):
  """Integrates fully implicit in velocity."""

  # TODO(team): add sparse version

  # compile-time constants
  passive_enabled = not m.opt.disableflags & DisableBit.PASSIVE.value
  actuation_enabled = (not m.opt.disableflags & DisableBit.ACTUATION.value) and m.actuator_affine_bias_gain

  if passive_enabled or actuation_enabled:
    derivative.deriv_smooth_vel(m, d)
    smooth._factor_solve_i_dense(m, d, d.qM_integration, d.qacc_integration, d.qfrc_integration)
    _advance(m, d, d.qacc_integration)
  else:
    _advance(m, d, d.qacc)


@event_scope
def fwd_position(m: Model, d: Data):
  """Position-dependent computations."""

  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.camlight(m, d)
  smooth.tendon(m, d)
  smooth.crb(m, d)
  smooth.tendon_armature(m, d)
  smooth.factor_m(m, d)
  collision_driver.collision(m, d)
  constraint.make_constraint(m, d)
  smooth.transmission(m, d)


# TODO(team): sparse version
def _actuator_velocity_sparse(m: Model, d: Data):
  NV = m.nv

  @kernel
  def actuator_velocity_sparse(
    # Data in:
    qvel_in: wp.array2d(dtype=float),
    actuator_moment_in: wp.array3d(dtype=float),
    # Data out:
    actuator_velocity_out: wp.array2d(dtype=float),
  ):
    worldid, actid = wp.tid()
    moment_tile = wp.tile_load(actuator_moment_in[worldid, actid], shape=NV)
    qvel_tile = wp.tile_load(qvel_in[worldid], shape=NV)
    moment_qvel_tile = wp.tile_map(wp.mul, moment_tile, qvel_tile)
    actuator_velocity_tile = wp.tile_reduce(wp.add, moment_qvel_tile)
    wp.tile_store(actuator_velocity_out[worldid], actuator_velocity_tile)

  wp.launch_tiled(
    actuator_velocity_sparse,
    dim=(d.nworld, m.nu),
    inputs=[
      d.qvel,
      d.actuator_moment,
    ],
    outputs=[
      d.actuator_velocity,
    ],
    block_dim=m.block_dim.actuator_velocity_sparse,
  )


@cache_kernel
def _tile_actuator_velocity_dense(
  tile_nu: TileSet,
  tile_nv: TileSet,
):
  @nested_kernel
  def actuator_velocity_dense(
    # Data in:
    qvel_in: wp.array3d(dtype=float),
    actuator_moment_in: wp.array3d(dtype=float),
    # In:
    tile_nu_adr: wp.array(dtype=int),
    tile_nv_adr: wp.array(dtype=int),
    # Data out:
    actuator_velocity_out: wp.array3d(dtype=float),
  ):
    worldid, nodeid = wp.tid()

    TILE_NU_SIZE = wp.static(int(tile_nu.size))
    TILE_NV_SIZE = wp.static(int(tile_nv.size))

    offset_nu = tile_nu_adr[nodeid]
    offset_nv = tile_nv_adr[nodeid]

    actuator_moment_tile = wp.tile_load(
      actuator_moment_in[worldid],
      shape=(TILE_NU_SIZE, TILE_NV_SIZE),
      offset=(offset_nu, offset_nv),
    )
    qvel_tile = wp.tile_load(qvel_in[worldid], shape=(TILE_NV_SIZE, 1), offset=(offset_nv, 0))
    velocity_tile = wp.tile_matmul(actuator_moment_tile, qvel_tile)

    wp.tile_store(actuator_velocity_out[worldid], velocity_tile, offset=(offset_nu, 0))

  return actuator_velocity_dense


def _tendon_velocity(m: Model, d: Data):
  NV = m.nv

  @kernel
  def tendon_velocity(
    # Data in:
    qvel_in: wp.array2d(dtype=float),
    ten_J_in: wp.array3d(dtype=float),
    # Data out:
    ten_velocity_out: wp.array2d(dtype=float),
  ):
    worldid, tenid = wp.tid()
    ten_J_tile = wp.tile_load(ten_J_in[worldid, tenid], shape=NV)
    qvel_tile = wp.tile_load(qvel_in[worldid], shape=NV)
    ten_J_qvel_tile = wp.tile_map(wp.mul, ten_J_tile, qvel_tile)
    ten_velocity_tile = wp.tile_reduce(wp.add, ten_J_qvel_tile)
    wp.tile_store(ten_velocity_out[worldid], ten_velocity_tile)

  wp.launch_tiled(
    tendon_velocity,
    dim=(d.nworld, m.ntendon),
    inputs=[
      d.qvel,
      d.ten_J,
    ],
    outputs=[
      d.ten_velocity,
    ],
    block_dim=m.block_dim.tendon_velocity,
  )


@event_scope
def fwd_velocity(m: Model, d: Data):
  """Velocity-dependent computations."""

  if m.opt.is_sparse:
    _actuator_velocity_sparse(m, d)
  else:
    for tile_nu, tile_nv in zip(m.actuator_moment_tiles_nu, m.actuator_moment_tiles_nv):
      # TODO(team): avoid creating invalid tiles
      if tile_nu.size == 0 or tile_nv.size == 0:
        continue

      wp.launch_tiled(
        _tile_actuator_velocity_dense(tile_nu, tile_nv),
        dim=(d.nworld, tile_nu.adr.size, tile_nv.adr.size),
        inputs=[d.qvel.reshape(d.qvel.shape + (1,)), d.actuator_moment, tile_nu.adr, tile_nv.adr],
        outputs=[d.actuator_velocity.reshape(d.actuator_velocity.shape + (1,))],
        block_dim=m.block_dim.actuator_velocity_dense,
      )

  if m.ntendon > 0:
    # TODO(team): sparse version
    _tendon_velocity(m, d)

  smooth.com_vel(m, d)
  passive.passive(m, d)
  smooth.rne(m, d)
  smooth.tendon_bias(m, d, d.qfrc_bias)


@wp.kernel
def _actuator_force(
  # Model:
  na: int,
  actuator_dyntype: wp.array(dtype=int),
  actuator_gaintype: wp.array(dtype=int),
  actuator_biastype: wp.array(dtype=int),
  actuator_actadr: wp.array(dtype=int),
  actuator_actnum: wp.array(dtype=int),
  actuator_ctrllimited: wp.array(dtype=bool),
  actuator_forcelimited: wp.array(dtype=bool),
  actuator_dynprm: wp.array2d(dtype=vec10f),
  actuator_gainprm: wp.array2d(dtype=vec10f),
  actuator_biasprm: wp.array2d(dtype=vec10f),
  actuator_ctrlrange: wp.array2d(dtype=wp.vec2),
  actuator_forcerange: wp.array2d(dtype=wp.vec2),
  actuator_acc0: wp.array(dtype=float),
  actuator_lengthrange: wp.array(dtype=wp.vec2),
  # Data in:
  act_in: wp.array2d(dtype=float),
  ctrl_in: wp.array2d(dtype=float),
  actuator_length_in: wp.array2d(dtype=float),
  actuator_velocity_in: wp.array2d(dtype=float),
  # In:
  dsbl_clampctrl: int,
  # Data out:
  act_dot_out: wp.array2d(dtype=float),
  actuator_force_out: wp.array2d(dtype=float),
):
  worldid, uid = wp.tid()

  ctrl = ctrl_in[worldid, uid]

  if actuator_ctrllimited[uid] and not dsbl_clampctrl:
    ctrlrange = actuator_ctrlrange[worldid, uid]
    ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])

  if na:
    actadr = actuator_actadr[uid]

    if actadr > -1:
      dyntype = actuator_dyntype[uid]
      act_dot = 0.0
      if dyntype == int(DynType.INTEGRATOR.value):
        act_dot = ctrl
      elif dyntype == int(DynType.FILTER.value) or dyntype == int(DynType.FILTEREXACT.value):
        dynprm = actuator_dynprm[worldid, uid]
        act = act_in[worldid, actadr]
        act_dot = (ctrl - act) / wp.max(dynprm[0], MJ_MINVAL)
      elif dyntype == int(DynType.MUSCLE.value):
        dynprm = actuator_dynprm[worldid, uid]
        act = act_in[worldid, actadr]
        act_dot = util_misc.muscle_dynamics(ctrl, act, dynprm)

      act_dot_out[worldid, actadr] = act_dot

  ctrl_act = ctrl
  if na:
    if actuator_actadr[uid] > -1:
      ctrl_act = act_in[worldid, actuator_actadr[uid] + actuator_actnum[uid] - 1]

  # TODO(team): actuator_actearly

  length = actuator_length_in[worldid, uid]
  velocity = actuator_velocity_in[worldid, uid]

  # gain
  gaintype = actuator_gaintype[uid]
  gainprm = actuator_gainprm[worldid, uid]

  gain = 0.0
  if gaintype == int(GainType.FIXED.value):
    gain = gainprm[0]
  elif gaintype == int(GainType.AFFINE.value):
    gain = gainprm[0] + gainprm[1] * length + gainprm[2] * velocity
  elif gaintype == int(GainType.MUSCLE.value):
    acc0 = actuator_acc0[uid]
    lengthrange = actuator_lengthrange[uid]
    gain = util_misc.muscle_gain(length, velocity, lengthrange, acc0, gainprm)

  # bias
  biastype = actuator_biastype[uid]
  biasprm = actuator_biasprm[worldid, uid]

  bias = 0.0  # BiasType.NONE
  if biastype == int(BiasType.AFFINE.value):
    bias = biasprm[0] + biasprm[1] * length + biasprm[2] * velocity
  elif biastype == int(BiasType.MUSCLE.value):
    acc0 = actuator_acc0[uid]
    lengthrange = actuator_lengthrange[uid]
    bias = util_misc.muscle_bias(length, lengthrange, acc0, biasprm)

  force = gain * ctrl_act + bias

  # TODO(team): tendon total force clamping

  if actuator_forcelimited[uid]:
    forcerange = actuator_forcerange[worldid, uid]
    force = wp.clamp(force, forcerange[0], forcerange[1])

  actuator_force_out[worldid, uid] = force


@wp.kernel
def _tendon_actuator_force(
  # Model:
  actuator_trntype: wp.array(dtype=int),
  actuator_trnid: wp.array(dtype=wp.vec2i),
  # Data in:
  actuator_force_in: wp.array2d(dtype=float),
  # Data out:
  ten_actfrc_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  if actuator_trntype[actid] == int(TrnType.TENDON.value):
    tenid = actuator_trnid[actid][0]
    # TODO(team): only compute for tendons with force limits?
    wp.atomic_add(ten_actfrc_out[worldid], tenid, actuator_force_in[worldid, actid])


@wp.kernel
def _tendon_actuator_force_clamp(
  # Model:
  actuator_trntype: wp.array(dtype=int),
  actuator_trnid: wp.array(dtype=wp.vec2i),
  tendon_actfrclimited: wp.array(dtype=bool),
  tendon_actfrcrange: wp.array2d(dtype=wp.vec2),
  # Data in:
  ten_actfrc_in: wp.array2d(dtype=float),
  # Data out:
  actuator_force_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  if actuator_trntype[actid] == int(TrnType.TENDON.value):
    tenid = actuator_trnid[actid][0]
    if tendon_actfrclimited[tenid]:
      ten_actfrc = ten_actfrc_in[worldid, tenid]
      actfrcrange = tendon_actfrcrange[worldid, tenid]

      if ten_actfrc < actfrcrange[0]:
        actuator_force_out[worldid, actid] *= actfrcrange[0] / ten_actfrc
      elif ten_actfrc > actfrcrange[1]:
        actuator_force_out[worldid, actid] *= actfrcrange[1] / ten_actfrc


@wp.kernel
def _qfrc_actuator_sparse(
  # Model:
  nu: int,
  ngravcomp: int,
  jnt_actfrclimited: wp.array(dtype=bool),
  jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
  jnt_actgravcomp: wp.array(dtype=int),
  dof_jntid: wp.array(dtype=int),
  # Data in:
  actuator_moment_in: wp.array3d(dtype=float),
  qfrc_gravcomp_in: wp.array2d(dtype=float),
  actuator_force_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_actuator_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()

  qfrc = float(0.0)
  for uid in range(nu):
    # TODO consider using Tile API or transpose moment for better access pattern
    qfrc += actuator_moment_in[worldid, uid, dofid] * actuator_force_in[worldid, uid]

  jntid = dof_jntid[dofid]

  # actuator-level gravity compensation, skip if added as passive force
  if ngravcomp and jnt_actgravcomp[jntid]:
    qfrc += qfrc_gravcomp_in[worldid, dofid]

  if jnt_actfrclimited[jntid]:
    frcrange = jnt_actfrcrange[worldid, jntid]
    qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])

  qfrc_actuator_out[worldid, dofid] = qfrc


@wp.kernel
def _qfrc_actuator_limited(
  # Model:
  ngravcomp: int,
  jnt_actfrclimited: wp.array(dtype=bool),
  jnt_actfrcrange: wp.array2d(dtype=wp.vec2),
  jnt_actgravcomp: wp.array(dtype=int),
  dof_jntid: wp.array(dtype=int),
  # Data in:
  qfrc_gravcomp_in: wp.array2d(dtype=float),
  qfrc_actuator_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_actuator_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  jntid = dof_jntid[dofid]
  qfrc_dof = qfrc_actuator_in[worldid, dofid]

  # actuator-level gravity compensation, skip if added as a passive force
  if ngravcomp and jnt_actgravcomp[jntid]:
    qfrc_dof += qfrc_gravcomp_in[worldid, dofid]

  if jnt_actfrclimited[jntid]:
    frcrange = jnt_actfrcrange[worldid, jntid]
    qfrc_dof = wp.clamp(qfrc_dof, frcrange[0], frcrange[1])

  qfrc_actuator_out[worldid, dofid] = qfrc_dof


@cache_kernel
def _tile_qfrc_actuator(tile_nu: TileSet, tile_nv: TileSet):
  @nested_kernel
  def qfrc_actuator(
    # Data in:
    actuator_force_in: wp.array3d(dtype=float),
    actuator_moment_in: wp.array3d(dtype=float),
    # In:
    tile_nu_adr: wp.array(dtype=int),
    tile_nv_adr: wp.array(dtype=int),
    # Data out:
    qfrc_actuator_out: wp.array3d(dtype=float),
  ):
    worldid, nodeid = wp.tid()

    TILE_NU_SIZE = wp.static(int(tile_nu.size))
    TILE_NV_SIZE = wp.static(int(tile_nv.size))

    offset_nu = tile_nu_adr[nodeid]
    offset_nv = tile_nv_adr[nodeid]

    actuator_moment_tile = wp.tile_load(
      actuator_moment_in[worldid], shape=(TILE_NU_SIZE, TILE_NV_SIZE), offset=(offset_nu, offset_nv)
    )
    actuator_moment_T_tile = wp.tile_transpose(actuator_moment_tile)

    force_tile = wp.tile_load(actuator_force_in[worldid], shape=(TILE_NU_SIZE, 1), offset=(offset_nu, 0))
    qfrc_tile = wp.tile_matmul(actuator_moment_T_tile, force_tile)
    wp.tile_store(qfrc_actuator_out[worldid], qfrc_tile, offset=(offset_nv, 0))

  return qfrc_actuator


@event_scope
def fwd_actuation(m: Model, d: Data):
  """Actuation-dependent computations."""
  if not m.nu or (m.opt.disableflags & DisableBit.ACTUATION):
    d.act_dot.zero_()
    d.qfrc_actuator.zero_()
    return

  wp.launch(
    _actuator_force,
    dim=(d.nworld, m.nu),
    inputs=[
      m.na,
      m.actuator_dyntype,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_ctrllimited,
      m.actuator_forcelimited,
      m.actuator_dynprm,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_ctrlrange,
      m.actuator_forcerange,
      m.actuator_acc0,
      m.actuator_lengthrange,
      d.act,
      d.ctrl,
      d.actuator_length,
      d.actuator_velocity,
      m.opt.disableflags & DisableBit.CLAMPCTRL,
    ],
    outputs=[d.act_dot, d.actuator_force],
  )

  if m.ntendon:
    d.ten_actfrc.zero_()

    wp.launch(
      _tendon_actuator_force,
      dim=(d.nworld, m.nu),
      inputs=[
        m.actuator_trntype,
        m.actuator_trnid,
        d.actuator_force,
      ],
      outputs=[d.ten_actfrc],
    )

    wp.launch(
      _tendon_actuator_force_clamp,
      dim=(d.nworld, m.nu),
      inputs=[
        m.actuator_trntype,
        m.actuator_trnid,
        m.tendon_actfrclimited,
        m.tendon_actfrcrange,
        d.ten_actfrc,
      ],
      outputs=[d.actuator_force],
    )

  if m.opt.is_sparse:
    wp.launch(
      _qfrc_actuator_sparse,
      dim=(d.nworld, m.nv),
      inputs=[
        m.nu,
        m.ngravcomp,
        m.jnt_actfrclimited,
        m.jnt_actfrcrange,
        m.jnt_actgravcomp,
        m.dof_jntid,
        d.actuator_moment,
        d.qfrc_gravcomp,
        d.actuator_force,
      ],
      outputs=[d.qfrc_actuator],
    )

  else:
    for tile_nu, tile_nv in zip(m.actuator_moment_tiles_nu, m.actuator_moment_tiles_nv):
      if tile_nu.size == 0 or tile_nv.size == 0:
        continue

      wp.launch_tiled(
        _tile_qfrc_actuator(tile_nu, tile_nv),
        dim=(d.nworld, tile_nu.adr.size, tile_nv.adr.size),
        inputs=[
          d.actuator_force.reshape(d.actuator_force.shape + (1,)),
          d.actuator_moment,
          tile_nu.adr,
          tile_nv.adr,
        ],
        outputs=[
          d.qfrc_actuator.reshape(d.qfrc_actuator.shape + (1,)),
        ],
        block_dim=m.block_dim.qfrc_actuator,
      )

    wp.launch(
      _qfrc_actuator_limited,
      dim=(d.nworld, m.nv),
      inputs=[
        m.ngravcomp,
        m.jnt_actfrclimited,
        m.jnt_actfrcrange,
        m.jnt_actgravcomp,
        m.dof_jntid,
        d.qfrc_gravcomp,
        d.qfrc_actuator,
      ],
      outputs=[d.qfrc_actuator],
    )

  # TODO actuator-level gravity compensation, skip if added as passive force


@wp.kernel
def _qfrc_smooth(
  # Data in:
  qfrc_applied_in: wp.array2d(dtype=float),
  qfrc_bias_in: wp.array2d(dtype=float),
  qfrc_passive_in: wp.array2d(dtype=float),
  qfrc_actuator_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_smooth_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qfrc_smooth_out[worldid, dofid] = (
    qfrc_passive_in[worldid, dofid]
    - qfrc_bias_in[worldid, dofid]
    + qfrc_actuator_in[worldid, dofid]
    + qfrc_applied_in[worldid, dofid]
  )


@event_scope
def fwd_acceleration(m: Model, d: Data):
  """Add up all non-constraint forces, compute qacc_smooth."""

  wp.launch(
    _qfrc_smooth,
    dim=(d.nworld, m.nv),
    inputs=[
      d.qfrc_applied,
      d.qfrc_bias,
      d.qfrc_passive,
      d.qfrc_actuator,
    ],
    outputs=[
      d.qfrc_smooth,
    ],
  )
  xfrc_accumulate(m, d, d.qfrc_smooth)

  smooth.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)


@event_scope
def forward(m: Model, d: Data):
  """Forward dynamics."""
  energy = m.opt.enableflags & EnableBit.ENERGY

  fwd_position(m, d)
  sensor.sensor_pos(m, d)

  if energy:
    if m.sensor_e_potential == 0:  # not computed by sensor
      sensor.energy_pos(m, d)
  else:
    d.energy.zero_()

  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  if energy:
    if m.sensor_e_kinetic == 0:  # not computed by sensor
      sensor.energy_vel(m, d)

  fwd_actuation(m, d)
  fwd_acceleration(m, d)
  sensor.sensor_acc(m, d)

  solver.solve(m, d)


@event_scope
def step(m: Model, d: Data):
  """Advance simulation."""
  forward(m, d)

  if m.opt.integrator == IntegratorType.EULER:
    euler(m, d)
  elif m.opt.integrator == IntegratorType.RK4:
    rungekutta4(m, d)
  elif m.opt.integrator == IntegratorType.IMPLICITFAST:
    implicit(m, d)
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")
