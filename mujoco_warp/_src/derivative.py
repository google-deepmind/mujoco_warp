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

from .types import BiasType
from .types import Data
from .types import DisableBit
from .types import DynType
from .types import GainType
from .types import Model
from .types import TileSet
from .types import vec10f
from .warp_util import cache_kernel
from .warp_util import event_scope
from .warp_util import kernel as nested_kernel

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _actuator_bias_gain_vel(
  # Model:
  actuator_dyntype: wp.array(dtype=int),
  actuator_gaintype: wp.array(dtype=int),
  actuator_biastype: wp.array(dtype=int),
  actuator_gainprm: wp.array2d(dtype=vec10f),
  actuator_biasprm: wp.array2d(dtype=vec10f),
  # Data in:
  act_in: wp.array2d(dtype=float),
  ctrl_in: wp.array2d(dtype=float),
  # Data out:
  act_vel_integration_out: wp.array2d(dtype=float),
):
  worldid, actid = wp.tid()

  if actuator_biastype[actid] == wp.static(BiasType.AFFINE.value):
    bias_vel = actuator_biasprm[worldid, actid][2]
  else:
    bias_vel = 0.0

  if actuator_gaintype[actid] == wp.static(GainType.AFFINE.value):
    gain_vel = actuator_gainprm[worldid, actid][2]
  else:
    gain_vel = 0.0

  if actuator_dyntype[actid] != wp.static(DynType.NONE.value):
    ctrl = act_in[worldid, actid]
  else:
    ctrl = ctrl_in[worldid, actid]

  act_vel_integration_out[worldid, actid] = bias_vel + gain_vel * ctrl


@cache_kernel
def _tile_qderiv_actuator_passive(
  tile_nu: TileSet,
  tile_nv: TileSet,
  actuation_enabled: bool,
  passive_enabled: bool,
  flg_forward: bool,
):
  @nested_kernel
  def qderiv_actuator_passive(
    # Model:
    opt_timestep: wp.array(dtype=float),
    dof_damping: wp.array2d(dtype=float),
    # Data in:
    qacc_in: wp.array3d(dtype=float),
    actuator_moment_in: wp.array3d(dtype=float),
    qM_in: wp.array3d(dtype=float),
    qfrc_smooth_in: wp.array3d(dtype=float),
    qfrc_constraint_in: wp.array3d(dtype=float),
    act_vel_integration_in: wp.array2d(dtype=float),
    # In:
    tile_nu_adr: wp.array(dtype=int),
    tile_nv_adr: wp.array(dtype=int),
    # Data out:
    qfrc_integration_out: wp.array3d(dtype=float),
    qM_integration_out: wp.array3d(dtype=float),
  ):
    worldid, nodeid = wp.tid()
    timestep = opt_timestep[worldid]

    TILE_NU_SIZE = wp.static(int(tile_nu.size))
    TILE_NV_SIZE = wp.static(int(tile_nv.size))

    offset_nv = tile_nv_adr[nodeid]

    # skip tree with no actuators.
    if wp.static(actuation_enabled and TILE_NU_SIZE != 0):
      offset_nu = tile_nu_adr[nodeid]
      actuator_moment_tile = wp.tile_load(
        actuator_moment_in[worldid],
        shape=(TILE_NU_SIZE, TILE_NV_SIZE),
        offset=(offset_nu, offset_nv),
      )
      zeros = wp.tile_zeros(shape=(TILE_NU_SIZE, TILE_NU_SIZE), dtype=wp.float32)
      vel_tile = wp.tile_load(act_vel_integration_in[worldid], shape=TILE_NU_SIZE, offset=offset_nu)
      diag = wp.tile_diag_add(zeros, vel_tile)
      actuator_moment_T = wp.tile_transpose(actuator_moment_tile)
      amTVel = wp.tile_matmul(actuator_moment_T, diag)
      qderiv_tile = wp.tile_matmul(amTVel, actuator_moment_tile)
    else:
      qderiv_tile = wp.tile_zeros(shape=(TILE_NV_SIZE, TILE_NV_SIZE), dtype=wp.float32)

    if wp.static(passive_enabled):
      dof_damping_tile = wp.tile_load(dof_damping[worldid], shape=TILE_NV_SIZE, offset=offset_nv)
      negative = wp.neg(dof_damping_tile)
      qderiv_tile = wp.tile_diag_add(qderiv_tile, negative)

    # add to qM
    qM_tile = wp.tile_load(
      qM_in[worldid],
      shape=(TILE_NV_SIZE, TILE_NV_SIZE),
      offset=(offset_nv, offset_nv),
    )
    qderiv_tile = wp.tile_map(wp.sub, qM_tile, qderiv_tile * timestep)
    wp.tile_store(qM_integration_out[worldid], qderiv_tile, offset=(offset_nv, offset_nv))

    if wp.static(flg_forward):
      # sum qfrc
      qfrc_smooth_tile = wp.tile_load(qfrc_smooth_in[worldid], shape=(TILE_NV_SIZE, 1), offset=(offset_nv, 0))
      qfrc_constraint_tile = wp.tile_load(qfrc_constraint_in[worldid], shape=(TILE_NV_SIZE, 1), offset=(offset_nv, 0))
      qfrc_combined = wp.add(qfrc_smooth_tile, qfrc_constraint_tile)
      wp.tile_store(qfrc_integration_out[worldid], qfrc_combined, offset=(offset_nv, 0))
    else:  # inverse
      qacc_tile = wp.tile_load(qacc_in[worldid], shape=(TILE_NV_SIZE, 1), offset=(offset_nv, 0))
      qfrc_inverse = wp.tile_matmul(qderiv_tile, qacc_tile)
      wp.tile_store(qfrc_integration_out[worldid], qfrc_inverse, offset=(offset_nv, 0))

  return qderiv_actuator_passive


@event_scope
def deriv_smooth_vel(m: Model, d: Data, flg_forward: bool = True):
  """Analytical derivative of smooth forces w.r.t velocities.

  Args:
    m (Model): The model containing kinematic and dynamic information (device).
    d (Data): The data object containing the current state and outputs arrays (device).
    flg_forward (bool, optional). If True forward dynamics else inverse dynamics routine.
                                  Default is True.
  """

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
  # The _actuator_bias_gain_vel kernel could theoretically be fused in as well,
  # but it's pretty clean straight-line code that loads a lot of data but
  # only stores one array, so I think the benefit of keeping that one on-chip
  # is likely not worth it compared to the compromises we're making with tile API.
  # It would also need a different data layout for the biasprm/gainprm arrays
  # to be tileable.

  # compile-time constants
  passive_enabled = not m.opt.disableflags & DisableBit.PASSIVE.value
  actuation_enabled = (not m.opt.disableflags & DisableBit.ACTUATION.value) and m.actuator_affine_bias_gain

  if actuation_enabled:
    wp.launch(
      _actuator_bias_gain_vel,
      dim=(d.nworld, m.nu),
      inputs=[
        m.actuator_dyntype,
        m.actuator_gaintype,
        m.actuator_biastype,
        m.actuator_gainprm,
        m.actuator_biasprm,
        d.act,
        d.ctrl,
      ],
      outputs=[
        d.act_vel_integration,
      ],
    )

  if actuation_enabled or passive_enabled:
    # TODO(team): sparse version

    for tile_nu, tile_nv in zip(m.actuator_moment_tiles_nu, m.actuator_moment_tiles_nv):
      wp.launch_tiled(
        _tile_qderiv_actuator_passive(tile_nu, tile_nv, actuation_enabled, passive_enabled, flg_forward),
        dim=(d.nworld, tile_nu.adr.size, tile_nv.adr.size),
        inputs=[
          m.opt.timestep,
          m.dof_damping,
          d.qacc.reshape(d.qacc.shape + (1,)),
          d.actuator_moment,
          d.qM,
          d.qfrc_smooth.reshape(d.qfrc_smooth.shape + (1,)),
          d.qfrc_constraint.reshape(d.qfrc_constraint.shape + (1,)),
          d.act_vel_integration,
          tile_nu.adr,
          tile_nv.adr,
        ],
        outputs=[
          d.qfrc_integration.reshape(d.qfrc_integration.shape + (1,)),
          d.qM_integration,
        ],
        block_dim=m.block_dim.qderiv_actuator_passive_actuation
        if actuation_enabled
        else m.block_dim.qderiv_actuator_passive_no_actuation,
      )

  # TODO(team): rne derivative
