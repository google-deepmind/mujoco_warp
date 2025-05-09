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

"""Tests for miscellaneous utility functions."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from . import util_misc
from .types import MJ_MINVAL
from .types import vec10


def _assert_eq(a, b, name):
  tol = 1e-3  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _muscle_dynamics_millard(ctrl, act, prm):
  """Compute time constant as in Millard et al. (2013) https://doi.org/10.1115/1.4023390."""

  # clamp control
  ctrlclamp = np.clip(ctrl, 0.0, 1.0)

  # clamp activation
  actclamp = np.clip(act, 0.0, 1.0)

  if ctrlclamp > act:
    tau = prm[0] * (0.5 + 1.5 * actclamp)
  else:
    tau = prm[1] / (0.5 + 1.5 * actclamp)

  # filter output
  return (ctrlclamp - act) / np.maximum(MJ_MINVAL, tau)


def _muscle_dynamics(ctrl, act, prm):
  @wp.kernel
  def muscle_dynamics(
    ctrl: float, act: float, prm: vec10, output: wp.array(dtype=float)
  ):
    output[0] = util_misc.muscle_dynamics(ctrl, act, prm)

  output = wp.empty(1, dtype=float)
  wp.launch(
    muscle_dynamics,
    dim=(1,),
    inputs=[
      ctrl,
      act,
      vec10(prm[0], prm[1], prm[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
    outputs=[output],
  )

  return output.numpy()[0]


def _muscle_gain_length(length, lmin, lmax):
  @wp.kernel
  def muscle_gain_length(
    length: float, lmin: float, lmax: float, output: wp.array(dtype=float)
  ):
    output[0] = util_misc.muscle_gain_length(length, lmin, lmax)

  output = wp.empty(1, dtype=float)
  wp.launch(muscle_gain_length, dim=(1,), inputs=[length, lmin, lmax], outputs=[output])

  return output.numpy()[0]


def _muscle_dynamics_timescale(dctrl, tau_act, tau_deact, smooth_width):
  @wp.kernel
  def muscle_gain_length(
    dctrl: float,
    tau_act: float,
    tau_deact: float,
    smooth_width: float,
    output: wp.array(dtype=float),
  ):
    output[0] = util_misc.muscle_dynamics_timescale(
      dctrl, tau_act, tau_deact, smooth_width
    )

  output = wp.empty(1, dtype=float)
  wp.launch(
    muscle_gain_length,
    dim=(1,),
    inputs=[dctrl, tau_act, tau_deact, smooth_width],
    outputs=[output],
  )

  return output.numpy()[0]


class UtilMiscTest(parameterized.TestCase):
  @parameterized.product(
    ctrl=[-0.1, 0.0, 0.4, 0.5, 1.0, 1.1], act=[-0.1, 0.0, 0.4, 0.5, 1.0, 1.1]
  )
  def test_muscle_dynamics_tausmooth0(self, ctrl, act):
    # exact equality if tau_smooth = 0
    prm = np.array([0.01, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    actdot_old = _muscle_dynamics_millard(ctrl, act, prm)
    actdot_new = _muscle_dynamics(ctrl, act, prm)

    _assert_eq(actdot_new, actdot_old, "actdot")

  def test_muscle_dynamics_tausmooth_positive(self):
    # positive tau_smooth
    prm = np.array([0.01, 0.04, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    act = 0.5
    eps = 1.0e-6

    ctrl = 0.4 - eps  # smaller than act by just over 0.5 * tau_smooth
    _assert_eq(
      _muscle_dynamics(ctrl, act, prm),
      _muscle_dynamics_millard(ctrl, act, prm),
      "actdot",
    )

    ctrl = 0.6 + eps  # larger than act by just over 0.5 * tau_smooth
    _assert_eq(
      _muscle_dynamics(ctrl, act, prm),
      _muscle_dynamics_millard(ctrl, act, prm),
      "actdot",
    )

  @parameterized.parameters(0.0, 0.1, 0.2, 1.0, 1.1)
  def test_muscle_dynamics_timescale(self, dctrl):
    # right in the middle should give average of time constants
    tau_smooth = 0.2
    tau_act = 0.2
    tau_deact = 0.3

    lower = _muscle_dynamics_timescale(-dctrl, tau_act, tau_deact, tau_smooth)
    upper = _muscle_dynamics_timescale(dctrl, tau_act, tau_deact, tau_smooth)

    _assert_eq(
      0.5 * (lower + upper), 0.5 * (tau_act + tau_deact), "muscle_dynamics_timescale"
    )

  @parameterized.parameters(
    (0.0, 0.0),
    (0.5, 0.0),
    (0.75, 0.5),
    (1.0, 1.0),
    (1.25, 0.5),
    (1.5, 0.0),
    (2.0, 0.0),
  )
  def test_muscle_gain_length(self, input, output):
    _assert_eq(_muscle_gain_length(input, 0.5, 1.5), output, "length-gain")

  # TODO(team): test util_misc.muscle_gain
  # TODO(team): test util_misc.muscle_bias


if __name__ == "__main__":
  wp.init()
  absltest.main()
