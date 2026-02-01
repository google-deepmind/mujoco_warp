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
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import TileSet
from mujoco_warp._src.types import vec10f
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

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
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    actuator_moment_in: wp.array3d(dtype=float),
    # In:
    vel_in: wp.array3d(dtype=float),
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
def deriv_smooth_vel(m: Model, d: Data, out: wp.array2d(dtype=float), flg_rne: bool = True):
  """Analytical derivative of smooth forces w.r.t. velocities.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    out: qM - dt * qDeriv (derivatives of smooth forces w.r.t velocities).
    flg_rne: Whether to include RNE derivatives.
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
            inputs=[d.actuator_moment, vel_3d, tile.adr],
            outputs=[out],
            block_dim=m.block_dim.qderiv_actuator_dense,
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

  if flg_rne:
    rne_vel(m, d, out)


@wp.kernel
def _derivative_com_vel_root(Dcvel_out: wp.array3d(dtype=wp.spatial_vector)):
  worldid, elementid, k = wp.tid()
  Dcvel_out[worldid, 0, k][elementid] = 0.0


@wp.kernel
def _derivative_com_vel_level(
  # Model:
  nv: int,
  body_parentid: wp.array(dtype=int),
  body_jntnum: wp.array(dtype=int),
  body_jntadr: wp.array(dtype=int),
  body_dofadr: wp.array(dtype=int),
  jnt_type: wp.array(dtype=int),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  body_tree_: wp.array(dtype=int),
  # Data out:
  # Out:
  Dcvel_out: wp.array3d(dtype=wp.spatial_vector),
  Dcdof_dot_out: wp.array3d(dtype=wp.spatial_vector),
):
  worldid, nodeid, k = wp.tid()
  bodyid = body_tree_[nodeid]
  dofid = body_dofadr[bodyid]
  jntid = body_jntadr[bodyid]
  jntnum = body_jntnum[bodyid]
  pid = body_parentid[bodyid]

  # Initialize from parent
  cvel_k = Dcvel_out[worldid, pid, k]

  if jntnum == 0:
    Dcvel_out[worldid, bodyid, k] = cvel_k
    return

  qvel = qvel_in[worldid]
  cdof = cdof_in[worldid]

  for j in range(jntid, jntid + jntnum):
    jnttype = jnt_type[j]

    if jnttype == JointType.FREE:
      # cvel += cdof * qvel
      if k >= dofid and k < dofid + 3:
        cvel_k += cdof[k]
      elif k >= dofid + 3 and k < dofid + 6:
        cvel_k += cdof[k]

      if k < nv:
        Dcdof_dot_out[worldid, dofid + 3, k] = math.motion_cross(cvel_k, cdof[dofid + 3])
        Dcdof_dot_out[worldid, dofid + 4, k] = math.motion_cross(cvel_k, cdof[dofid + 4])
        Dcdof_dot_out[worldid, dofid + 5, k] = math.motion_cross(cvel_k, cdof[dofid + 5])

      dofid += 6

    elif jnttype == JointType.BALL:
      if k < nv:
        Dcdof_dot_out[worldid, dofid + 0, k] = math.motion_cross(cvel_k, cdof[dofid + 0])
        Dcdof_dot_out[worldid, dofid + 1, k] = math.motion_cross(cvel_k, cdof[dofid + 1])
        Dcdof_dot_out[worldid, dofid + 2, k] = math.motion_cross(cvel_k, cdof[dofid + 2])

      if k >= dofid and k < dofid + 3:
        cvel_k += cdof[k]

      dofid += 3
    else:
      if k < nv:
        Dcdof_dot_out[worldid, dofid, k] = math.motion_cross(cvel_k, cdof[dofid])

      if k == dofid:
        cvel_k += cdof[dofid]

      dofid += 1

  Dcvel_out[worldid, bodyid, k] = cvel_k


@wp.func
def _mul_inert_vec(inert: vec10f, vec: wp.spatial_vector) -> wp.spatial_vector:
  mass = inert[0]
  h = wp.vec3(inert[1], inert[2], inert[3])
  # I_3x3 from symmetric values (xx, yy, zz, xy, xz, yz)
  # row 0: xx, xy, xz
  # row 1: xy, yy, yz
  # row 2: xz, yz, zz
  I = wp.mat33(inert[4], inert[7], inert[8], inert[7], inert[5], inert[9], inert[8], inert[9], inert[6])

  ang = wp.spatial_top(vec)
  lin = wp.spatial_bottom(vec)

  res_ang = I * ang + wp.cross(h, lin)
  res_lin = mass * lin - wp.cross(h, ang)

  return wp.spatial_vector(res_ang, res_lin)


@wp.kernel
def _derivative_rne_forward_level(
  # Model:
  nv: int,
  body_parentid: wp.array(dtype=int),
  body_dofnum: wp.array(dtype=int),
  body_dofadr: wp.array(dtype=int),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  cinert_in: wp.array2d(dtype=vec10f),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  cdof_dot_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  body_tree_: wp.array(dtype=int),
  Dcvel_in: wp.array3d(dtype=wp.spatial_vector),
  Dcdof_dot_in: wp.array3d(dtype=wp.spatial_vector),
  # Out:
  Dcacc_out: wp.array3d(dtype=wp.spatial_vector),
  Dcfrcbody_out: wp.array3d(dtype=wp.spatial_vector),
):
  worldid, nodeid, k = wp.tid()
  bodyid = body_tree_[nodeid]
  dofid = body_dofadr[bodyid]
  dofnum = body_dofnum[bodyid]
  pid = body_parentid[bodyid]

  dcacc = Dcacc_out[worldid, pid, k]

  qvel = qvel_in[worldid]

  for j in range(dofid, dofid + dofnum):
    # Term 1: cdof_dot * d(qvel)/dk
    if j == k:
      dcacc += cdof_dot_in[worldid, j]

    # Term 2: Dcdofdot * qvel
    dcdofdot = Dcdof_dot_in[worldid, j, k]
    dcacc += dcdofdot * qvel[j]

  Dcacc_out[worldid, bodyid, k] = dcacc

  # Dcfrcbody calculation
  cinert = cinert_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]
  dcvel = Dcvel_in[worldid, bodyid, k]

  # term1 = cinert * dcacc
  term1 = _mul_inert_vec(cinert, dcacc)

  cinert_cvel = _mul_inert_vec(cinert, cvel)
  cinert_dcvel = _mul_inert_vec(cinert, dcvel)

  term2 = math.motion_cross_force(dcvel, cinert_cvel) + math.motion_cross_force(cvel, cinert_dcvel)

  Dcfrcbody_out[worldid, bodyid, k] = term1 + term2


@wp.kernel
def _derivative_rne_backward_level(
  # Model:
  body_parentid: wp.array(dtype=int),
  # In:
  body_tree_: wp.array(dtype=int),
  # Out:
  Dcfrcbody_out: wp.array3d(dtype=wp.spatial_vector),
):
  worldid, nodeid, k = wp.tid()
  bodyid = body_tree_[nodeid]
  pid = body_parentid[bodyid]

  if pid == 0 and bodyid == 0:
    return  # World body has no parent to add to

  val = Dcfrcbody_out[worldid, bodyid, k]
  wp.atomic_add(Dcfrcbody_out[worldid, pid], k, val)


@wp.kernel
def _derivative_rne_update_sparse(
  # Model:
  dof_bodyid: wp.array(dtype=int),
  # Data in:
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  timestep: wp.array(dtype=float),
  qMi: wp.array(dtype=int),
  qMj: wp.array(dtype=int),
  Dcfrcbody_in: wp.array3d(dtype=wp.spatial_vector),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  worldid, elemid = wp.tid()
  dt = timestep[worldid % timestep.shape[0]]

  i = qMi[elemid]
  j = qMj[elemid]

  # qDeriv[i, j] -= cdof[i] * Dcfrcbody[body(i), j]

  body_i = dof_bodyid[i]
  dcfrc = Dcfrcbody_in[worldid, body_i, j]
  term = wp.dot(cdof_in[worldid, i], dcfrc)

  wp.atomic_add(qDeriv_out[worldid, 0], elemid, -dt * term)


@wp.kernel
def _derivative_rne_update_dense(
  # Model:
  dof_bodyid: wp.array(dtype=int),
  # Data in:
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  timestep: wp.array(dtype=float),
  Dcfrcbody_in: wp.array3d(dtype=wp.spatial_vector),
  # Out:
  qDeriv_out: wp.array3d(dtype=float),
):
  worldid, i, j = wp.tid()
  dt = timestep[worldid % timestep.shape[0]]

  body_i = dof_bodyid[i]
  dcfrc = Dcfrcbody_in[worldid, body_i, j]
  term = wp.dot(cdof_in[worldid, i], dcfrc)

  qDeriv_out[worldid, i, j] -= dt * term


@event_scope
def rne_vel(m: Model, d: Data, out: wp.array2d(dtype=float)):  # out is qDeriv-like
  # Temporary dense allocations
  Dcvel = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)
  Dcdof_dot = wp.zeros((d.nworld, m.nv, m.nv), dtype=wp.spatial_vector)
  Dcacc = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)
  Dcfrcbody = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)

  # Compute Dcvel and Dcdofdot
  wp.launch(
    _derivative_com_vel_root,
    dim=(d.nworld, 1, m.nv),
    inputs=[Dcvel],
    outputs=[],
  )

  for body_tree in m.body_tree:
    wp.launch(
      _derivative_com_vel_level,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[m.nv, m.body_parentid, m.body_jntnum, m.body_jntadr, m.body_dofadr, m.jnt_type, d.qvel, d.cdof, body_tree],
      outputs=[Dcvel, Dcdof_dot],
    )

  # Forward pass (Dcacc, Dcfrcbody)
  for body_tree in m.body_tree:
    wp.launch(
      _derivative_rne_forward_level,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[
        m.nv,
        m.body_parentid,
        m.body_dofnum,
        m.body_dofadr,
        d.qvel,
        d.cinert,
        d.cvel,
        d.cdof_dot,
        body_tree,
        Dcvel,
        Dcdof_dot,
      ],
      outputs=[Dcacc, Dcfrcbody],
    )

  # Backward pass (Accumulate Dcfrcbody)
  for body_tree in reversed(m.body_tree):
    wp.launch(
      _derivative_rne_backward_level,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[m.body_parentid, body_tree],
      outputs=[Dcfrcbody],  # In/Out
    )

  if m.opt.is_sparse:
    wp.launch(
      _derivative_rne_update_sparse,
      dim=(d.nworld, m.qM_fullm_i.size),
      inputs=[m.dof_bodyid, d.cdof, m.opt.timestep, m.qM_fullm_i, m.qM_fullm_j, Dcfrcbody],
      outputs=[out],
    )
  else:
    wp.launch(
      _derivative_rne_update_dense,
      dim=(d.nworld, m.nv, m.nv),
      inputs=[m.dof_bodyid, d.cdof, m.opt.timestep, Dcfrcbody],
      outputs=[out],
    )
