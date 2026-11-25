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

from mujoco_warp._src import constraint
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import fusion
from mujoco_warp._src import sensor
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src import support
from mujoco_warp._src import util_misc
from mujoco_warp._src.support import mul_m
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import Model

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@fusion.stage("nworld", "nv")
def _qfrc_eulerdamp(
  worldid: int,
  dofid: int,
  # Model:
  opt_timestep: wp.array[float],
  jnt_actuatorid: wp.array[int],
  dof_jntid: wp.array[int],
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  actuator_damping: wp.array2d[float],
  actuator_dampingpoly: wp.array2d[wp.vec2],
  actuator_gear: wp.array2d[wp.spatial_vector],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # Out:
  qfrc_out: wp.array2d[float],
):
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  damping, dpoly = util_misc.dof_damping_total(
    jnt_actuatorid,
    dof_jntid,
    dof_damping,
    dof_dampingpoly,
    actuator_trntype,
    actuator_trnid,
    actuator_damping,
    actuator_dampingpoly,
    actuator_gear,
    worldid,
    dofid,
  )
  v = qvel_in[worldid, dofid]

  damp_deriv = util_misc._poly_force_deriv(damping, dpoly, v, 1)
  qfrc_out[worldid, dofid] += timestep * damp_deriv * qacc_in[worldid, dofid]


@fusion.stage("nworld", "nv")
def _mul_qLU(
  worldid: int,
  dofid: int,
  # Model:
  D_rownnz: wp.array[int],
  D_rowadr: wp.array[int],
  D_colind: wp.array[int],
  # Data in:
  qLU_in: wp.array2d[float],
  # In:
  vec: wp.array2d[float],
  # Out:
  res: wp.array2d[float],
):
  rowadr = D_rowadr[dofid]
  val = float(0.0)
  for k in range(D_rownnz[dofid]):
    adr = rowadr + k
    val += qLU_in[worldid, adr] * vec[worldid, D_colind[adr]]
  res[worldid, dofid] = val


@fusion.stage("nworld", "nv")
def _qfrc_inverse(
  worldid: int,
  dofid: int,
  # Data in:
  qfrc_bias_in: wp.array2d[float],
  qfrc_passive_in: wp.array2d[float],
  qfrc_constraint_in: wp.array2d[float],
  # In:
  Ma: wp.array2d[float],
  # Data out:
  qfrc_inverse_out: wp.array2d[float],
):
  qfrc_inverse = qfrc_bias_in[worldid, dofid]
  qfrc_inverse += Ma[worldid, dofid]
  qfrc_inverse -= qfrc_passive_in[worldid, dofid]
  qfrc_inverse -= qfrc_constraint_in[worldid, dofid]

  qfrc_inverse_out[worldid, dofid] = qfrc_inverse


def discrete_acc(m: Model, d: Data, qacc: wp.array2d[float]):
  """Convert discrete-time qacc to continuous-time qacc.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    qacc: Acceleration.
  """
  qfrc = wp.empty((d.nworld, m.nv), dtype=float)

  if m.opt.integrator == IntegratorType.RK4:
    raise NotImplementedError("discrete inverse dynamics is not supported by RK4 integrator")
  elif m.opt.integrator == IntegratorType.EULER:
    if m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER):
      fusion.copy(qacc, d.qacc)
      return

    # TODO(team): qacc = d.qacc if (m.dof_damping == 0.0).all()

    # set qfrc = (d.M + m.opt.timestep * diag(m.dof_damping)) * d.qacc

    # d.M @ d.qacc
    support.mul_m(m, d, qfrc, d.qacc)

    # qfrc += m.opt.timestep * damp_deriv * d.qacc
    fusion.launch(m, d, _qfrc_eulerdamp, qfrc_out=qfrc)
  elif m.opt.integrator == IntegratorType.IMPLICIT:
    # set qLU = M - dt * qDeriv, in scratch: d.qLU holds the forward factorization
    qLU = wp.empty((d.nworld, m.nD), dtype=float)
    forward.implicit_qlu(m, d, qLU)

    # set qfrc = qLU * qacc
    fusion.launch(m, d, _mul_qLU, qLU_in=qLU, vec=d.qacc, res=qfrc)
    forward.implicit_free_body(m, d, d.qacc, qfrc, solve=False)
  elif m.opt.integrator == IntegratorType.IMPLICITFAST:
    qDeriv = wp.empty((d.nworld, m.nC), dtype=float)
    derivative.deriv_smooth_vel(m, d, qDeriv)
    mul_m(m, d, qfrc, d.qacc, M=qDeriv)

    # standalone free bodies: overwrite block rows with the unsymmetric local product,
    # including the bias (gyroscopic) derivative
    forward.implicit_free_body(m, d, d.qacc, qfrc, solve=False)
  elif m.opt.integrator == IntegratorType.DISCRETE:
    # Under discrete, qacc is already the discrete step map: INVDISCRETE is implied.
    fusion.copy(qacc, d.qacc)
    return
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")

  # solve for qacc: qfrc = d.M @ d.qacc
  smooth.solve_m(m, d, qacc, qfrc)


@fusion.stage("nworld", "nv")
def _qfrc_inverse_discrete(
  worldid: int,
  dofid: int,
  # Data in:
  qfrc_bias_in: wp.array2d[float],
  qfrc_passive_in: wp.array2d[float],
  qfrc_constraint_in: wp.array2d[float],
  efm_c_in: wp.array2d[float],
  efm_ca_in: wp.array2d[float],
  # In:
  Ma: wp.array2d[float],
  # Data out:
  qfrc_inverse_out: wp.array2d[float],
):
  qfrc_inverse = qfrc_bias_in[worldid, dofid]
  qfrc_inverse += Ma[worldid, dofid]
  qfrc_inverse -= efm_c_in[worldid, dofid]
  qfrc_inverse -= efm_ca_in[worldid, dofid]
  qfrc_inverse -= qfrc_passive_in[worldid, dofid]
  qfrc_inverse -= qfrc_constraint_in[worldid, dofid]

  qfrc_inverse_out[worldid, dofid] = qfrc_inverse


@fusion.stage("nworld", "nv")
def _zero_qfrc_constraint_nefc(worldid: int, dofid: int, nefc_in: wp.array[int], qfrc_constraint_out: wp.array2d[float]):
  if nefc_in[worldid] == 0:
    qfrc_constraint_out[worldid, dofid] = 0.0


def inv_constraint(m: Model, d: Data):
  """Inverse constraint solver."""
  # no constraints
  if d.njmax == 0:
    fusion.zero(d.qfrc_constraint)
    return

  if m.is_sparse:
    fusion.launch(m, d, _zero_qfrc_constraint_nefc)

  ctx = solver.create_inverse_context(m, d)
  solver.init_context(m, d, ctx, grad=False)


def inverse(m: Model, d: Data):
  """Inverse dynamics."""
  forward.check_discrete(m)
  forward.fwd_position(m, d)
  sensor.sensor_pos(m, d)
  forward.fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  is_discrete = m.opt.integrator == IntegratorType.DISCRETE
  invdiscrete = (m.opt.enableflags & EnableBit.INVDISCRETE) or is_discrete
  if invdiscrete:
    # save discrete-time qacc and compute continuous-time qacc
    qacc_discrete = fusion.clone(d.qacc)
    discrete_acc(m, d, d.qacc)

  if is_discrete:
    derivative.eff_actuation(m, d)
    constraint.regularize_constraint(m, d)

  inv_constraint(m, d)
  smooth.rne(m, d)
  smooth.tendon_bias(m, d, d.qfrc_bias)
  sensor.sensor_acc(m, d)

  if is_discrete:
    derivative.eff_mul_m(m, d, d.qfrc_inverse, d.qacc)
    forward._launch_discrete_free_gyro(m, d, d.qfrc_inverse, d.qacc, True)
    fusion.launch(m, d, _qfrc_inverse_discrete, Ma=d.qfrc_inverse)
  else:
    support.mul_m(m, d, d.qfrc_inverse, d.qacc)
    fusion.launch(m, d, _qfrc_inverse, Ma=d.qfrc_inverse)

  if invdiscrete:
    # restore discrete-time qacc
    fusion.copy(d.qacc, qacc_discrete)
