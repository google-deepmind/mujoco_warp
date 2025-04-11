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

import mujoco
import warp as wp

from . import collision_driver
from . import constraint
from . import math
from . import passive
from . import sensor
from . import smooth
from . import solver
from .support import xfrc_accumulate
from .types import MJ_MINVAL
from .types import BiasType
from .types import Data
from .types import DisableBit
from .types import DynType
from .types import GainType
from .types import JointType
from .types import Model
from .types import array2df
from .types import array3df
from .warp_util import event_scope
from .warp_util import kernel
from .warp_util import kernel_copy

# RK4 tableau
_RK4_A = [
  [0.5, 0.0, 0.0],
  [0.0, 0.5, 0.0],
  [0.0, 0.0, 1.0],
]
_RK4_B = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]


@wp.func
def _integrate_pos(
  worldId: int,
  jntid: int,
  m: Model,
  qpos_out: array2df,
  qpos_in: array2df,
  qvel_in: array2df,
  qvel_scale: float = 1.0,
):
  jnt_type = m.jnt_type[jntid]
  qpos_adr = m.jnt_qposadr[jntid]
  dof_adr = m.jnt_dofadr[jntid]
  qpos = qpos_in[worldId]
  qpos_o = qpos_out[worldId]
  qvel = qvel_in[worldId]

  if jnt_type == wp.static(JointType.FREE.value):
    qpos_pos = wp.vec3(qpos[qpos_adr], qpos[qpos_adr + 1], qpos[qpos_adr + 2])
    qvel_lin = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale

    qpos_new = qpos_pos + m.opt.timestep * qvel_lin

    qpos_quat = wp.quat(
      qpos[qpos_adr + 3],
      qpos[qpos_adr + 4],
      qpos[qpos_adr + 5],
      qpos[qpos_adr + 6],
    )
    qvel_ang = (
      wp.vec3(qvel[dof_adr + 3], qvel[dof_adr + 4], qvel[dof_adr + 5]) * qvel_scale
    )

    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, m.opt.timestep)

    qpos_o[qpos_adr] = qpos_new[0]
    qpos_o[qpos_adr + 1] = qpos_new[1]
    qpos_o[qpos_adr + 2] = qpos_new[2]
    qpos_o[qpos_adr + 3] = qpos_quat_new[0]
    qpos_o[qpos_adr + 4] = qpos_quat_new[1]
    qpos_o[qpos_adr + 5] = qpos_quat_new[2]
    qpos_o[qpos_adr + 6] = qpos_quat_new[3]

  elif jnt_type == wp.static(JointType.BALL.value):
    qpos_quat = wp.quat(
      qpos[qpos_adr],
      qpos[qpos_adr + 1],
      qpos[qpos_adr + 2],
      qpos[qpos_adr + 3],
    )
    qvel_ang = wp.vec3(qvel[dof_adr], qvel[dof_adr + 1], qvel[dof_adr + 2]) * qvel_scale

    qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, m.opt.timestep)

    qpos_o[qpos_adr] = qpos_quat_new[0]
    qpos_o[qpos_adr + 1] = qpos_quat_new[1]
    qpos_o[qpos_adr + 2] = qpos_quat_new[2]
    qpos_o[qpos_adr + 3] = qpos_quat_new[3]

  else:  # if jnt_type in (JointType.HINGE, JointType.SLIDE):
    qpos_o[qpos_adr] = qpos[qpos_adr] + m.opt.timestep * qvel[dof_adr] * qvel_scale


def _advance(
  m: Model, d: Data, act_dot: wp.array, qacc: wp.array, qvel: Optional[wp.array] = None
):
  """Advance state and time given activation derivatives and acceleration."""

  # TODO(team): can we assume static timesteps?

  @kernel
  def next_activation(
    m: Model,
    d: Data,
    act_dot_in: array2df,
  ):
    worldid, actid = wp.tid()

    # get the high/low range for each actuator state
    limited = m.actuator_actlimited[actid]
    range_low = wp.where(limited, m.actuator_actrange[actid][0], -wp.inf)
    range_high = wp.where(limited, m.actuator_actrange[actid][1], wp.inf)

    # get the actual actuation - skip if -1 (means stateless actuator)
    act_adr = m.actuator_actadr[actid]
    if act_adr == -1:
      return

    acts = d.act[worldid]
    acts_dot = act_dot_in[worldid]

    act = acts[act_adr]
    act_dot = acts_dot[act_adr]

    # check dynType
    dyn_type = m.actuator_dyntype[actid]
    dyn_prm = m.actuator_dynprm[actid][0]

    # advance the actuation
    if dyn_type == wp.static(DynType.FILTEREXACT.value):
      tau = wp.where(dyn_prm < MJ_MINVAL, MJ_MINVAL, dyn_prm)
      act = act + act_dot * tau * (1.0 - wp.exp(-m.opt.timestep / tau))
    else:
      act = act + act_dot * m.opt.timestep

    # apply limits
    wp.clamp(act, range_low, range_high)

    acts[act_adr] = act

  @kernel
  def advance_velocities(m: Model, d: Data, qacc: array2df):
    worldid, tid = wp.tid()
    d.qvel[worldid, tid] = d.qvel[worldid, tid] + qacc[worldid, tid] * m.opt.timestep

  @kernel
  def integrate_joint_positions(m: Model, d: Data, qvel_in: array2df):
    worldid, jntid = wp.tid()
    _integrate_pos(worldid, jntid, m, d.qpos, d.qpos, qvel_in)

  # skip if no stateful actuators.
  if m.na:
    wp.launch(next_activation, dim=(d.nworld, m.nu), inputs=[m, d, act_dot])

  wp.launch(advance_velocities, dim=(d.nworld, m.nv), inputs=[m, d, qacc])

  # advance positions with qvel if given, d.qvel otherwise (semi-implicit)
  if qvel is not None:
    qvel_in = qvel
  else:
    qvel_in = d.qvel

  wp.launch(integrate_joint_positions, dim=(d.nworld, m.njnt), inputs=[m, d, qvel_in])

  d.time = d.time + m.opt.timestep


@event_scope
def euler(m: Model, d: Data):
  """Euler integrator, semi-implicit in velocity."""

  # integrate damping implicitly

  def eulerdamp_sparse(m: Model, d: Data):
    @kernel
    def add_damping_sum_qfrc_kernel_sparse(m: Model, d: Data):
      worldid, tid = wp.tid()

      dof_Madr = m.dof_Madr[tid]
      d.qM_integration[worldid, 0, dof_Madr] += m.opt.timestep * m.dof_damping[tid]

      d.qfrc_integration[worldid, tid] = (
        d.qfrc_smooth[worldid, tid] + d.qfrc_constraint[worldid, tid]
      )

    kernel_copy(d.qM_integration, d.qM)
    wp.launch(add_damping_sum_qfrc_kernel_sparse, dim=(d.nworld, m.nv), inputs=[m, d])
    smooth.factor_solve_i(
      m,
      d,
      d.qM_integration,
      d.qLD_integration,
      d.qLDiagInv_integration,
      d.qacc_integration,
      d.qfrc_integration,
    )

  def eulerdamp_fused_dense(m: Model, d: Data):
    def tile_eulerdamp(adr: int, size: int, tilesize: int):
      @kernel
      def eulerdamp(
        m: Model, d: Data, damping: wp.array(dtype=wp.float32), leveladr: int
      ):
        worldid, nodeid = wp.tid()
        dofid = m.qLD_tile[leveladr + nodeid]
        M_tile = wp.tile_load(
          d.qM[worldid], shape=(tilesize, tilesize), offset=(dofid, dofid)
        )
        damping_tile = wp.tile_load(damping, shape=(tilesize,), offset=(dofid,))
        damping_scaled = damping_tile * m.opt.timestep
        qm_integration_tile = wp.tile_diag_add(M_tile, damping_scaled)

        qfrc_smooth_tile = wp.tile_load(
          d.qfrc_smooth[worldid], shape=(tilesize,), offset=(dofid,)
        )
        qfrc_constraint_tile = wp.tile_load(
          d.qfrc_constraint[worldid], shape=(tilesize,), offset=(dofid,)
        )

        qfrc_tile = qfrc_smooth_tile + qfrc_constraint_tile

        L_tile = wp.tile_cholesky(qm_integration_tile)
        qacc_tile = wp.tile_cholesky_solve(L_tile, qfrc_tile)
        wp.tile_store(d.qacc_integration[worldid], qacc_tile, offset=(dofid))

      wp.launch_tiled(
        eulerdamp, dim=(d.nworld, size), inputs=[m, d, m.dof_damping, adr], block_dim=32
      )

    qLD_tileadr, qLD_tilesize = m.qLD_tileadr.numpy(), m.qLD_tilesize.numpy()

    for i in range(len(qLD_tileadr)):
      beg = qLD_tileadr[i]
      end = m.qLD_tile.shape[0] if i == len(qLD_tileadr) - 1 else qLD_tileadr[i + 1]
      tile_eulerdamp(beg, end - beg, int(qLD_tilesize[i]))

  if not m.opt.disableflags & DisableBit.EULERDAMP.value:
    if m.opt.is_sparse:
      eulerdamp_sparse(m, d)
    else:
      eulerdamp_fused_dense(m, d)

    _advance(m, d, d.act_dot, d.qacc_integration)
  else:
    _advance(m, d, d.act_dot, d.qacc)


@event_scope
def rungekutta4(m: Model, d: Data):
  """Runge-Kutta explicit order 4 integrator."""

  kernel_copy(d.qpos_t0, d.qpos)
  kernel_copy(d.qvel_t0, d.qvel)
  kernel_copy(d.act_t0, d.act)

  A, B = _RK4_A, _RK4_B

  def rk_accumulate(d: Data, b: float):
    """Computes one term of 1/6 k_1 + 1/3 k_2 + 1/3 k_3 + 1/6 k_4"""

    @kernel
    def _qvel_acc(d: Data, b: float):
      worldId, tid = wp.tid()
      d.qvel_rk[worldId, tid] += b * d.qvel[worldId, tid]
      d.qacc_rk[worldId, tid] += b * d.qacc[worldId, tid]

    @kernel
    def _act_dot(d: Data, b: float):
      worldId, tid = wp.tid()
      d.act_dot_rk[worldId, tid] += b * d.act_dot[worldId, tid]

    wp.launch(_qvel_acc, dim=(d.nworld, m.nv), inputs=[d, b])
    wp.launch(_act_dot, dim=(d.nworld, m.na), inputs=[d, b])

  def perturb_state(m: Model, d: Data, a: float):
    @kernel
    def _qpos(m: Model, d: Data):
      """Integrate joint positions"""
      worldId, jntId = wp.tid()
      _integrate_pos(worldId, jntId, m, d.qpos, d.qpos_t0, d.qvel, qvel_scale=a)

    @kernel
    def _act(m: Model, d: Data):
      worldId, tid = wp.tid()
      dact_dot = a * d.act_dot[worldId, tid]
      d.act[worldId, tid] = d.act_t0[worldId, tid] + dact_dot * m.opt.timestep

    @kernel
    def _qvel(m: Model, d: Data):
      worldId, tid = wp.tid()
      dqacc = a * d.qacc[worldId, tid]
      d.qvel[worldId, tid] = d.qvel_t0[worldId, tid] + dqacc * m.opt.timestep

    wp.launch(_qpos, dim=(d.nworld, m.njnt), inputs=[m, d])
    wp.launch(_act, dim=(d.nworld, m.na), inputs=[m, d])
    wp.launch(_qvel, dim=(d.nworld, m.nv), inputs=[m, d])

  rk_accumulate(d, B[0])
  for i in range(3):
    a, b = float(A[i][i]), B[i + 1]
    perturb_state(m, d, a)
    forward(m, d)
    rk_accumulate(d, b)

  kernel_copy(d.qpos, d.qpos_t0)
  kernel_copy(d.qvel, d.qvel_t0)
  kernel_copy(d.act, d.act_t0)
  _advance(m, d, d.act_dot_rk, d.qacc_rk, d.qvel_rk)


@event_scope
def implicit(m: Model, d: Data):
  """Integrates fully implicit in velocity."""

  # optimization comments (AD)
  # I went from small kernels for every step to a relatively big single
  # kernel using tile API because it kept improving performance -
  # 30M to 50M FPS on an A6000.
  #
  # The main benefit is reduced global memory roundtrips, but I assume
  # there is also some benefit to loading data as early as possible.
  #
  # I further tried fusing in the cholesky factor/solve but the high
  # storage requirements led to low occupancy and thus worse performance.
  #
  # The actuator_bias_gain_vel kernel could theoretically be fused in as well,
  # but it's pretty clean straight-line code that loads a lot of data but
  # only stores one array, so I think the benefit of keeping that one on-chip
  # is likely not worth it compared to the compromises we're making with tile API.
  # It would also need a different data layout for the biasprm/gainprm arrays
  # to be tileable.

  # assumptions
  assert not m.opt.is_sparse  # unsupported
  # TODO(team): add sparse version

  # compile-time constants
  passive_enabled = not m.opt.disableflags & DisableBit.PASSIVE.value
  actuation_enabled = (
    not m.opt.disableflags & DisableBit.ACTUATION.value
  ) and m.actuator_affine_bias_gain

  @kernel
  def actuator_bias_gain_vel(m: Model, d: Data):
    worldid, actid = wp.tid()

    bias_vel = 0.0
    gain_vel = 0.0

    actuator_biastype = m.actuator_biastype[actid]
    actuator_gaintype = m.actuator_gaintype[actid]
    actuator_dyntype = m.actuator_dyntype[actid]

    if actuator_biastype == wp.static(BiasType.AFFINE.value):
      bias_vel = m.actuator_biasprm[actid, 2]

    if actuator_gaintype == wp.static(GainType.AFFINE.value):
      gain_vel = m.actuator_gainprm[actid, 2]

    ctrl = d.ctrl[worldid, actid]

    if actuator_dyntype != wp.static(DynType.NONE.value):
      ctrl = d.act[worldid, actid]

    d.act_vel_integration[worldid, actid] = bias_vel + gain_vel * ctrl

  def qderiv_actuator_damping_fused(
    m: Model, d: Data, damping: wp.array(dtype=wp.float32)
  ):
    if actuation_enabled:
      block_dim = 64
    else:
      block_dim = 256

    @wp.func
    def subtract_multiply(x: wp.float32, y: wp.float32):
      return x - y * wp.static(m.opt.timestep)

    def qderiv_actuator_damping_tiled(
      adr: int, size: int, tilesize_nv: int, tilesize_nu: int
    ):
      @kernel
      def qderiv_actuator_fused_kernel(
        m: Model, d: Data, damping: wp.array(dtype=wp.float32), leveladr: int
      ):
        worldid, nodeid = wp.tid()
        offset_nv = m.actuator_moment_offset_nv[leveladr + nodeid]

        # skip tree with no actuators.
        if wp.static(actuation_enabled and tilesize_nu != 0):
          offset_nu = m.actuator_moment_offset_nu[leveladr + nodeid]
          actuator_moment_tile = wp.tile_load(
            d.actuator_moment[worldid],
            shape=(tilesize_nu, tilesize_nv),
            offset=(offset_nu, offset_nv),
          )
          zeros = wp.tile_zeros(shape=(tilesize_nu, tilesize_nu), dtype=wp.float32)
          vel_tile = wp.tile_load(
            d.act_vel_integration[worldid], shape=(tilesize_nu), offset=offset_nu
          )
          diag = wp.tile_diag_add(zeros, vel_tile)
          actuator_moment_T = wp.tile_transpose(actuator_moment_tile)
          amTVel = wp.tile_matmul(actuator_moment_T, diag)
          qderiv_tile = wp.tile_matmul(amTVel, actuator_moment_tile)
        else:
          qderiv_tile = wp.tile_zeros(
            shape=(tilesize_nv, tilesize_nv), dtype=wp.float32
          )

        if wp.static(passive_enabled):
          dof_damping = wp.tile_load(damping, shape=tilesize_nv, offset=offset_nv)
          negative = wp.neg(dof_damping)
          qderiv_tile = wp.tile_diag_add(qderiv_tile, negative)

        # add to qM
        qM_tile = wp.tile_load(
          d.qM[worldid], shape=(tilesize_nv, tilesize_nv), offset=(offset_nv, offset_nv)
        )
        qderiv_tile = wp.tile_map(subtract_multiply, qM_tile, qderiv_tile)
        wp.tile_store(
          d.qM_integration[worldid], qderiv_tile, offset=(offset_nv, offset_nv)
        )

        # sum qfrc
        qfrc_smooth_tile = wp.tile_load(
          d.qfrc_smooth[worldid], shape=tilesize_nv, offset=offset_nv
        )
        qfrc_constraint_tile = wp.tile_load(
          d.qfrc_constraint[worldid], shape=tilesize_nv, offset=offset_nv
        )
        qfrc_combined = wp.add(qfrc_smooth_tile, qfrc_constraint_tile)
        wp.tile_store(d.qfrc_integration[worldid], qfrc_combined, offset=offset_nv)

      wp.launch_tiled(
        qderiv_actuator_fused_kernel,
        dim=(d.nworld, size),
        inputs=[m, d, damping, adr],
        block_dim=block_dim,
      )

    qderiv_tilesize_nv = m.actuator_moment_tilesize_nv.numpy()
    qderiv_tilesize_nu = m.actuator_moment_tilesize_nu.numpy()
    qderiv_tileadr = m.actuator_moment_tileadr.numpy()

    for i in range(len(qderiv_tileadr)):
      beg = qderiv_tileadr[i]
      end = (
        m.qLD_tile.shape[0] if i == len(qderiv_tileadr) - 1 else qderiv_tileadr[i + 1]
      )
      if qderiv_tilesize_nv[i] != 0:
        qderiv_actuator_damping_tiled(
          beg, end - beg, int(qderiv_tilesize_nv[i]), int(qderiv_tilesize_nu[i])
        )

  if passive_enabled or actuation_enabled:
    if actuation_enabled:
      wp.launch(
        actuator_bias_gain_vel,
        dim=(d.nworld, m.nu),
        inputs=[m, d],
      )

    qderiv_actuator_damping_fused(m, d, m.dof_damping)

    smooth._factor_solve_i_dense(
      m, d, d.qM_integration, d.qacc_integration, d.qfrc_integration
    )

    _advance(m, d, d.act_dot, d.qacc_integration)
  else:
    _advance(m, d, d.act_dot, d.qacc)


@event_scope
def fwd_position(m: Model, d: Data):
  """Position-dependent computations."""

  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.camlight(m, d)
  smooth.tendon(m, d)
  smooth.crb(m, d)
  smooth.factor_m(m, d)
  collision_driver.collision(m, d)
  constraint.make_constraint(m, d)
  smooth.transmission(m, d)


@event_scope
def fwd_velocity(m: Model, d: Data):
  """Velocity-dependent computations."""

  if m.opt.is_sparse:
    # TODO(team): sparse version
    NV = m.nv

    @kernel
    def _actuator_velocity(d: Data):
      worldid, actid = wp.tid()
      moment_tile = wp.tile_load(d.actuator_moment[worldid, actid], shape=NV)
      qvel_tile = wp.tile_load(d.qvel[worldid], shape=NV)
      moment_qvel_tile = wp.tile_map(wp.mul, moment_tile, qvel_tile)
      actuator_velocity_tile = wp.tile_reduce(wp.add, moment_qvel_tile)
      wp.tile_store(d.actuator_velocity[worldid], actuator_velocity_tile)

    wp.launch_tiled(_actuator_velocity, dim=(d.nworld, m.nu), inputs=[d], block_dim=32)
  else:

    def actuator_velocity(
      adr: int,
      size: int,
      tilesize_nu: int,
      tilesize_nv: int,
    ):
      @kernel
      def _actuator_velocity(
        m: Model, d: Data, leveladr: int, velocity: array3df, qvel: array3df
      ):
        worldid, nodeid = wp.tid()
        offset_nu = m.actuator_moment_offset_nu[leveladr + nodeid]
        offset_nv = m.actuator_moment_offset_nv[leveladr + nodeid]
        actuator_moment_tile = wp.tile_load(
          d.actuator_moment[worldid],
          shape=(tilesize_nu, tilesize_nv),
          offset=(offset_nu, offset_nv),
        )
        qvel_tile = wp.tile_load(
          qvel[worldid], shape=(tilesize_nv, 1), offset=(offset_nv, 0)
        )
        velocity_tile = wp.tile_matmul(actuator_moment_tile, qvel_tile)

        wp.tile_store(velocity[worldid], velocity_tile, offset=(offset_nu, 0))

      wp.launch_tiled(
        _actuator_velocity,
        dim=(d.nworld, size),
        inputs=[
          m,
          d,
          adr,
          d.actuator_velocity.reshape(d.actuator_velocity.shape + (1,)),
          d.qvel.reshape(d.qvel.shape + (1,)),
        ],
        block_dim=32,
      )

    actuator_moment_tilesize_nu = m.actuator_moment_tilesize_nu.numpy()
    actuator_moment_tilesize_nv = m.actuator_moment_tilesize_nv.numpy()
    actuator_moment_tileadr = m.actuator_moment_tileadr.numpy()

    for i in range(len(actuator_moment_tileadr)):
      beg = actuator_moment_tileadr[i]
      end = (
        m.actuator_moment_tileadr.shape[0]
        if i == len(actuator_moment_tileadr) - 1
        else actuator_moment_tileadr[i + 1]
      )
      if actuator_moment_tilesize_nu[i] != 0 and actuator_moment_tilesize_nv[i] != 0:
        actuator_velocity(
          beg,
          end - beg,
          int(actuator_moment_tilesize_nu[i]),
          int(actuator_moment_tilesize_nv[i]),
        )

  if m.ntendon > 0:
    # TODO(team): sparse version
    NV = m.nv

    @kernel
    def _tendon_velocity(d: Data):
      worldid, tenid = wp.tid()
      ten_J_tile = wp.tile_load(d.ten_J[worldid, tenid], shape=NV)
      qvel_tile = wp.tile_load(d.qvel[worldid], shape=NV)
      ten_J_qvel_tile = wp.tile_map(wp.mul, ten_J_tile, qvel_tile)
      ten_velocity_tile = wp.tile_reduce(wp.add, ten_J_qvel_tile)
      wp.tile_store(d.ten_velocity[worldid], ten_velocity_tile)

    wp.launch_tiled(
      _tendon_velocity, dim=(d.nworld, m.ntendon), inputs=[d], block_dim=32
    )

  smooth.com_vel(m, d)
  passive.passive(m, d)
  smooth.rne(m, d)


@event_scope
def fwd_actuation(m: Model, d: Data):
  """Actuation-dependent computations."""
  if not m.nu or m.opt.disableflags & DisableBit.ACTUATION:
    d.act_dot.zero_()
    d.qfrc_actuator.zero_()
    return

  # TODO support stateful actuators

  @kernel
  def _force(
    m: Model,
    d: Data,
    # outputs
    force: array2df,
  ):
    worldid, uid = wp.tid()

    actuator_length = d.actuator_length[worldid, uid]
    actuator_velocity = d.actuator_velocity[worldid, uid]

    gain = m.actuator_gainprm[uid, 0]
    gain += m.actuator_gainprm[uid, 1] * actuator_length
    gain += m.actuator_gainprm[uid, 2] * actuator_velocity

    bias = m.actuator_biasprm[uid, 0]
    bias += m.actuator_biasprm[uid, 1] * actuator_length
    bias += m.actuator_biasprm[uid, 2] * actuator_velocity

    ctrl = d.ctrl[worldid, uid]
    disable_clampctrl = m.opt.disableflags & wp.static(DisableBit.CLAMPCTRL.value)
    if m.actuator_ctrllimited[uid] and not disable_clampctrl:
      r = m.actuator_ctrlrange[uid]
      ctrl = wp.clamp(ctrl, r[0], r[1])
    f = gain * ctrl + bias
    if m.actuator_forcelimited[uid]:
      r = m.actuator_forcerange[uid]
      f = wp.clamp(f, r[0], r[1])
    force[worldid, uid] = f

  @kernel
  def _qfrc_limited(m: Model, d: Data):
    worldid, dofid = wp.tid()
    jntid = m.dof_jntid[dofid]
    if m.jnt_actfrclimited[jntid]:
      d.qfrc_actuator[worldid, dofid] = wp.clamp(
        d.qfrc_actuator[worldid, dofid],
        m.jnt_actfrcrange[jntid][0],
        m.jnt_actfrcrange[jntid][1],
      )

  if m.opt.is_sparse:
    # TODO(team): sparse version
    @kernel
    def _qfrc(m: Model, moment: array3df, force: array2df, qfrc: array2df):
      worldid, vid = wp.tid()

      s = float(0.0)
      for uid in range(m.nu):
        # TODO consider using Tile API or transpose moment for better access pattern
        s += moment[worldid, uid, vid] * force[worldid, uid]
      jntid = m.dof_jntid[vid]
      if m.jnt_actfrclimited[jntid]:
        r = m.jnt_actfrcrange[jntid]
        s = wp.clamp(s, r[0], r[1])
      qfrc[worldid, vid] = s

  wp.launch(_force, dim=[d.nworld, m.nu], inputs=[m, d], outputs=[d.actuator_force])

  if m.opt.is_sparse:
    # TODO(team): sparse version

    wp.launch(
      _qfrc,
      dim=(d.nworld, m.nv),
      inputs=[m, d.actuator_moment, d.actuator_force],
      outputs=[d.qfrc_actuator],
    )

  else:

    def qfrc_actuator(adr: int, size: int, tilesize_nu: int, tilesize_nv: int):
      @kernel
      def qfrc_actuator_kernel(
        m: Model,
        d: Data,
        leveladr: int,
        qfrc_actuator: array3df,
        actuator_force: array3df,
      ):
        worldid, nodeid = wp.tid()
        offset_nu = m.actuator_moment_offset_nu[leveladr + nodeid]
        offset_nv = m.actuator_moment_offset_nv[leveladr + nodeid]

        actuator_moment_tile = wp.tile_load(
          d.actuator_moment[worldid],
          shape=(tilesize_nu, tilesize_nv),
          offset=(offset_nu, offset_nv),
        )
        actuator_moment_T_tile = wp.tile_transpose(actuator_moment_tile)

        force_tile = wp.tile_load(
          actuator_force[worldid], shape=(tilesize_nu, 1), offset=(offset_nu, 0)
        )
        qfrc_tile = wp.tile_matmul(actuator_moment_T_tile, force_tile)
        wp.tile_store(qfrc_actuator[worldid], qfrc_tile, offset=(offset_nv, 0))

      wp.launch_tiled(
        qfrc_actuator_kernel,
        dim=(d.nworld, size),
        inputs=[
          m,
          d,
          adr,
          d.qfrc_actuator.reshape(d.qfrc_actuator.shape + (1,)),
          d.actuator_force.reshape(d.actuator_force.shape + (1,)),
        ],
        block_dim=32,
      )

    qderiv_tilesize_nu = m.actuator_moment_tilesize_nu.numpy()
    qderiv_tilesize_nv = m.actuator_moment_tilesize_nv.numpy()
    qderiv_tileadr = m.actuator_moment_tileadr.numpy()

    for i in range(len(qderiv_tileadr)):
      beg = qderiv_tileadr[i]
      end = (
        m.qLD_tile.shape[0] if i == len(qderiv_tileadr) - 1 else qderiv_tileadr[i + 1]
      )
      if qderiv_tilesize_nu[i] != 0 and qderiv_tilesize_nv[i] != 0:
        qfrc_actuator(
          beg, end - beg, int(qderiv_tilesize_nu[i]), int(qderiv_tilesize_nv[i])
        )

    wp.launch(_qfrc_limited, dim=(d.nworld, m.nv), inputs=[m, d])

  # TODO actuator-level gravity compensation, skip if added as passive force


@event_scope
def fwd_acceleration(m: Model, d: Data):
  """Add up all non-constraint forces, compute qacc_smooth."""

  @kernel
  def _qfrc_smooth(d: Data):
    worldid, dofid = wp.tid()
    d.qfrc_smooth[worldid, dofid] = (
      d.qfrc_passive[worldid, dofid]
      - d.qfrc_bias[worldid, dofid]
      + d.qfrc_actuator[worldid, dofid]
      + d.qfrc_applied[worldid, dofid]
    )

  wp.launch(_qfrc_smooth, dim=(d.nworld, m.nv), inputs=[d])
  xfrc_accumulate(m, d, d.qfrc_smooth)

  smooth.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)


@event_scope
def forward(m: Model, d: Data):
  """Forward dynamics."""

  fwd_position(m, d)
  sensor.sensor_pos(m, d)
  fwd_velocity(m, d)
  sensor.sensor_vel(m, d)
  fwd_actuation(m, d)
  fwd_acceleration(m, d)
  sensor.sensor_acc(m, d)

  if d.njmax == 0:
    kernel_copy(d.qacc, d.qacc_smooth)
  else:
    solver.solve(m, d)


@event_scope
def step(m: Model, d: Data):
  """Advance simulation."""
  forward(m, d)

  if m.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER:
    euler(m, d)
  elif m.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
    rungekutta4(m, d)
  elif m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
    implicit(m, d)
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")
