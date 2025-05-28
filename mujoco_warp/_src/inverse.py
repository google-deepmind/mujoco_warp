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
from . import sensor
from . import smooth
from . import solver
from . import support
from .types import Data
from .types import DisableBit
from .types import EnableBit
from .types import IntegratorType
from .types import Model


@wp.kernel
def _qfrc_eulerdamp(
  # Model:
  opt_timestep: float,
  dof_damping: wp.array2d(dtype=float),
  # Data in:
  qacc_in: wp.array2d(dtype=float),
  # Out:
  qfrc_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qfrc_out[worldid, dofid] += opt_timestep * dof_damping[worldid, dofid] * qacc_in[worldid, dofid]


@wp.kernel
def _qfrc_inverse(
  # Model:
  dof_armature: wp.array2d(dtype=float),
  # Data in:
  qacc_in: wp.array2d(dtype=float),
  qfrc_bias_in: wp.array2d(dtype=float),
  qfrc_passive_in: wp.array2d(dtype=float),
  qfrc_constraint_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_inverse_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()

  qfrc_inverse = 0.0
  qfrc_inverse += qfrc_bias_in[worldid, dofid]
  qfrc_inverse += dof_armature[worldid, dofid] * qacc_in[worldid, dofid]
  qfrc_inverse -= qfrc_passive_in[worldid, dofid]
  qfrc_inverse -= qfrc_constraint_in[worldid, dofid]

  qfrc_inverse_out[worldid, dofid] = qfrc_inverse


def discrete_acc(m: Model, d: Data, qacc: wp.array2d(dtype=float)):
  """Convert discrete-time qacc to continuous-time qacc."""

  if m.opt.integrator == IntegratorType.RK4:
    raise ValueError("discrete inverse dynamics is not supported by RK4 integrator")
  elif m.opt.integrator == IntegratorType.EULER:
    if m.opt.disableflags & DisableBit.EULERDAMP:
      wp.copy(qacc, d.qacc)
      return

    # TODO(team): qacc = d.qacc if (m.dof_damping == 0.0).all()

    # set qfrc = (d.qM + m.opt.timestep * diag(m.dof_damping)) * d.qacc
    skip = wp.zeros(d.nworld, dtype=bool)
    qfrc = wp.empty((d.nworld, m.nv), dtype=float)

    # d.qM @ d.qacc
    support.mul_m(m, d, qfrc, d.qacc, skip)

    # qfrc += m.opt.timestep * m.dof_damping * d.qacc
    wp.launch(
      _qfrc_eulerdamp,
      dim=(d.nworld, m.nv),
      inputs=[m.opt.timestep, m.dof_damping, d.qacc],
      outputs=[qfrc],
    )
  elif m.opt.integrator == IntegratorType.IMPLICITFAST:
    # TODO(team):
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")
  else:
    raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")

  # solve for qacc: qfrc = d.qM @ d.qacc
  smooth.solve_m(m, d, qacc, qfrc)


def inv_constraint(m: Model, d: Data):
  """Inverse constraint solver."""

  # no constraints
  if d.njmax == 0:
    d.qfrc_constraint.zero_()
    return

  # update
  solver.create_context(m, d, grad=False)


def inverse(m: Model, d: Data):
  """Inverse dynamics."""
  forward.fwd_position(m, d)
  sensor.sensor_pos(m, d)
  forward.fwd_velocity(m, d)
  sensor.sensor_vel(m, d)

  invdiscrete = m.opt.enableflags & EnableBit.INVDISCRETE
  if invdiscrete:
    # save discrete-time qacc and compute continuous-time qacc
    wp.copy(d.qacc_discrete, d.qacc)
    discrete_acc(m, d, d.qacc)

  inv_constraint(m, d)
  smooth.rne(m, d, flg_acc=True)
  sensor.sensor_acc(m, d)

  wp.launch(
    _qfrc_inverse,
    dim=(d.nworld, m.nv),
    inputs=[
      m.dof_armature,
      d.qacc,
      d.qfrc_bias,
      d.qfrc_passive,
      d.qfrc_constraint,
    ],
    outputs=[d.qfrc_inverse],
  )

  if invdiscrete:
    # restore discrete-time qacc
    wp.copy(d.qacc, d.qacc_discrete)
