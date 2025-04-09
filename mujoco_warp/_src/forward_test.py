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

"""Tests for forward dynamics functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath

import mujoco_warp as mjwarp

from . import test_util

wp.config.verify_cuda = True

# tolerance for difference between MuJoCo and mjwarp smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class ForwardTest(parameterized.TestCase):
  def test_fwd_velocity(self):
    _, mjd, m, d = test_util.fixture("humanoid/humanoid.xml", kick=True)

    d.actuator_velocity.zero_()
    mjwarp.fwd_velocity(m, d)

    _assert_eq(
      d.actuator_velocity.numpy()[0], mjd.actuator_velocity, "actuator_velocity"
    )
    _assert_eq(d.qfrc_bias.numpy()[0], mjd.qfrc_bias, "qfrc_bias")

  @parameterized.parameters(True, False)
  def test_fwd_actuation(self, is_sparse):
    mjm, mjd, m, d = test_util.fixture("actuation.xml", kick=True)

    mujoco.mj_fwdActuation(mjm, mjd)

    for arr in (d.actuator_force, d.qfrc_actuator):
      arr.zero_()

    mjwarp.fwd_actuation(m, d)

    _assert_eq(d.ctrl.numpy()[0], mjd.ctrl, "ctrl")
    _assert_eq(d.actuator_force.numpy()[0], mjd.actuator_force, "actuator_force")
    _assert_eq(d.qfrc_actuator.numpy()[0], mjd.qfrc_actuator, "qfrc_actuator")

    # TODO(team): test DisableBit.CLAMPCTRL
    # TODO(team): test DisableBit.ACTUATION
    # TODO(team): test actuator gain/bias (e.g. position control)

  def test_fwd_acceleration(self):
    _, mjd, m, d = test_util.fixture("humanoid/humanoid.xml", kick=True)

    for arr in (d.qfrc_smooth, d.qacc_smooth):
      arr.zero_()

    mjwarp.fwd_acceleration(m, d)

    _assert_eq(d.qfrc_smooth.numpy()[0], mjd.qfrc_smooth, "qfrc_smooth")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")

  def test_eulerdamp(self):
    mjm, mjd, _, _ = test_util.fixture("pendula.xml", kick=True)
    self.assertTrue((mjm.dof_damping > 0).any())

    mjd.qvel[:] = 1.0
    mjd.qacc[:] = 1.0
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mjwarp.euler(m, d)
    mujoco.mj_Euler(mjm, mjd)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.act.numpy()[0], mjd.act, "act")

    # also test sparse
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    mjd = mujoco.MjData(mjm)
    mjd.qvel[:] = 1.0
    mjd.qacc[:] = 1.0
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mjwarp.euler(m, d)
    mujoco.mj_Euler(mjm, mjd)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.act.numpy()[0], mjd.act, "act")

  def test_disable_eulerdamp(self):
    mjm, mjd, _, _ = test_util.fixture("pendula.xml", kick=True)
    mjm.opt.disableflags = mjm.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_EULERDAMP

    mujoco.mj_forward(mjm, mjd)
    mjd.qvel[:] = 1.0
    mjd.qacc[:] = 1.0

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mjwarp.euler(m, d)

    np.testing.assert_allclose(d.qvel.numpy()[0], 1 + mjm.opt.timestep)

  def test_rungekutta4(self):
    # slower than other tests because `forward` compilation
    mjm, mjd, m, d = test_util.fixture(
      xml="""
        <mujoco>
          <option integrator="RK4">
            <flag constraint="disable"/>
          </option>
          <worldbody>
            <geom type="plane" size="1 1 .01" pos="0 0 -1"/>
            <body pos="0.15 0 0">
              <joint type="hinge" axis="0 1 0"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
              <body pos="0.1 0 0">
                <joint type="slide" axis="1 0 0" stiffness="200"/>
                <geom type="capsule" size="0.015" fromto="-.1 0 0 .1 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    mjwarp.rungekutta4(m, d)
    mujoco.mj_RungeKutta(mjm, mjd, 4)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.qvel.numpy()[0], mjd.qvel, "qvel")
    _assert_eq(d.act.numpy()[0], mjd.act, "act")
    _assert_eq(d.time, mjd.time, "time")
    _assert_eq(d.xpos.numpy()[0], mjd.xpos, "xpos")


class ImplicitIntegratorTest(parameterized.TestCase):
  @parameterized.parameters(
    0,
    mjwarp.DisableBit.PASSIVE.value,
    mjwarp.DisableBit.ACTUATION.value,
    mjwarp.DisableBit.PASSIVE.value & mjwarp.DisableBit.ACTUATION.value,
  )
  def test_implicit(self, disableFlags):
    mjm, _, _, _ = test_util.fixture("pendula.xml")

    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mjm.opt.disableflags = mjm.opt.disableflags | disableFlags
    mjm.actuator_gainprm[:, 2] = np.random.normal(
      scale=10, size=mjm.actuator_gainprm[:, 2].shape
    )

    # change actuators to velocity/damper to cover all codepaths
    mjm.actuator_gaintype[3] = mujoco.mjtGain.mjGAIN_AFFINE
    mjm.actuator_gaintype[6] = mujoco.mjtGain.mjGAIN_AFFINE
    mjm.actuator_biastype[0:3] = mujoco.mjtBias.mjBIAS_AFFINE
    mjm.actuator_biastype[4:6] = mujoco.mjtBias.mjBIAS_AFFINE
    mjm.actuator_biasprm[0:3, 2] = -1
    mjm.actuator_biasprm[4:6, 2] = -1
    mjm.actuator_ctrlrange[3:7] = 10.0
    mjm.actuator_gear[:] = 1.0

    mjd = mujoco.MjData(mjm)

    mjd.qvel = np.random.uniform(low=-0.01, high=0.01, size=mjd.qvel.shape)
    mjd.ctrl = np.random.normal(scale=10, size=mjd.ctrl.shape)
    mjd.act = np.random.normal(scale=10, size=mjd.act.shape)
    mujoco.mj_forward(mjm, mjd)

    mjd.ctrl = np.random.normal(scale=10, size=mjd.ctrl.shape)
    mjd.act = np.random.normal(scale=10, size=mjd.act.shape)
    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mjwarp.implicit(m, d)
    mujoco.mj_implicit(mjm, mjd)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.act.numpy()[0], mjd.act, "act")


if __name__ == "__main__":
  wp.init()
  absltest.main()
