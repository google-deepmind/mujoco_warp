# Copyright 2026 The Newton Developers
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

"""Tests for actuator and sensor delay."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

from mujoco_warp import test_data
from mujoco_warp._src import delay
from mujoco_warp._src import forward

_TOLERANCE = 1e-8


class ActuatorDelayTest(absltest.TestCase):
  """Test actuator delay, mirroring MuJoCo C tests."""

  def test_actuator_delay(self):
    """Test basic actuator delay with ZOH (default interp).

    delay=0.02, timestep=0.01, nsample=2.
    Ctrl set to 10 should arrive after 2 timesteps.
    """
    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="slide" delay="0.02" nsample="2"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # delay = 0.02, timestep = 0.01 => 2-step delay
    self.assertEqual(mjm.actuator_history[0, 0], 2)

    # set ctrl=10
    d.ctrl.numpy()[:] = 10.0
    wp.copy(d.ctrl, wp.array(np.full((1, 1), 10.0), dtype=float))

    # step 1: delayed ctrl still 0
    forward.step(m, d)
    act_force = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(act_force, 0.0, atol=_TOLERANCE, err_msg="step 0")

    # step 2: still 0
    forward.step(m, d)
    act_force = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(act_force, 0.0, atol=_TOLERANCE, err_msg="step 1")

    # step 3: now ctrl=10 arrives
    forward.step(m, d)
    act_force = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(act_force, 10.0, atol=_TOLERANCE, err_msg="step 2")

  def test_actuator_delay_linear_interp(self):
    """Test actuator delay with linear interpolation.

    delay=0.015, timestep=0.01, nsample=3, interp=linear.
    """
    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="slide" delay="0.015" nsample="3" interp="linear"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertEqual(mjm.actuator_history[0, 0], 3)
    self.assertEqual(mjm.actuator_history[0, 1], 1)  # interp=linear

    # step 0: ctrl=10, expected force=0
    wp.copy(d.ctrl, wp.array(np.full((1, 1), 10.0), dtype=float))
    forward.step(m, d)
    act_force = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(act_force, 0.0, atol=_TOLERANCE, err_msg="step 0")

    # step 1: ctrl=20, expected force=5
    wp.copy(d.ctrl, wp.array(np.full((1, 1), 20.0), dtype=float))
    forward.step(m, d)
    act_force = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(act_force, 5.0, atol=_TOLERANCE, err_msg="step 1")

    # step 2: ctrl=30, expected force=15
    wp.copy(d.ctrl, wp.array(np.full((1, 1), 30.0), dtype=float))
    forward.step(m, d)
    act_force = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(act_force, 15.0, atol=_TOLERANCE, err_msg="step 2")


class SensorDelayTest(absltest.TestCase):
  """Test sensor delay, mirroring MuJoCo C tests."""

  def test_sensor_delay(self):
    """Test basic sensor delay with ZOH.

    delay=0.02, timestep=0.01, nsample=3.
    """
    xml = """
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <sensor>
        <jointpos joint="slide" delay="0.02" nsample="3"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertEqual(mjm.sensor_history[0, 0], 3)
    np.testing.assert_allclose(mjm.sensor_delay[0], 0.02, atol=1e-10)

    # step 0: qpos=10
    wp.copy(d.qpos, wp.array(np.full((1, 1), 10.0), dtype=float))
    forward.step(m, d)
    sdata = d.sensordata.numpy()[0, 0]
    np.testing.assert_allclose(sdata, 0.0, atol=_TOLERANCE, err_msg="step 0")

    # step 1: qpos=20
    wp.copy(d.qpos, wp.array(np.full((1, 1), 20.0), dtype=float))
    forward.step(m, d)
    sdata = d.sensordata.numpy()[0, 0]
    np.testing.assert_allclose(sdata, 0.0, atol=_TOLERANCE, err_msg="step 1")

    # step 2: qpos=30, expecting value from step 0 (delay=2 steps)
    wp.copy(d.qpos, wp.array(np.full((1, 1), 30.0), dtype=float))
    forward.step(m, d)
    sdata = d.sensordata.numpy()[0, 0]
    np.testing.assert_allclose(sdata, 10.0, atol=_TOLERANCE, err_msg="step 2")

    # step 3: qpos=40, expecting value from step 1
    wp.copy(d.qpos, wp.array(np.full((1, 1), 40.0), dtype=float))
    forward.step(m, d)
    sdata = d.sensordata.numpy()[0, 0]
    np.testing.assert_allclose(sdata, 20.0, atol=_TOLERANCE, err_msg="step 3")

  def test_sensor_delay_linear_interp(self):
    """Test sensor delay with linear interpolation.

    delay=0.015, timestep=0.01, nsample=3, interp=linear.
    """
    xml = """
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <sensor>
        <jointpos joint="slide" delay="0.015" nsample="3" interp="linear"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertEqual(mjm.sensor_history[0, 0], 3)
    self.assertEqual(mjm.sensor_history[0, 1], 1)

    # step 0: qpos=10, expected=0
    wp.copy(d.qpos, wp.array(np.full((1, 1), 10.0), dtype=float))
    forward.step(m, d)
    sdata = d.sensordata.numpy()[0, 0]
    np.testing.assert_allclose(sdata, 0.0, atol=_TOLERANCE, err_msg="step 0")

    # step 1: qpos=20, expected=5
    wp.copy(d.qpos, wp.array(np.full((1, 1), 20.0), dtype=float))
    forward.step(m, d)
    sdata = d.sensordata.numpy()[0, 0]
    np.testing.assert_allclose(sdata, 5.0, atol=_TOLERANCE, err_msg="step 1")

    # step 2: qpos=30, expected=15
    wp.copy(d.qpos, wp.array(np.full((1, 1), 30.0), dtype=float))
    forward.step(m, d)
    sdata = d.sensordata.numpy()[0, 0]
    np.testing.assert_allclose(sdata, 15.0, atol=_TOLERANCE, err_msg="step 2")


class MujocoReferenceTest(absltest.TestCase):
  """Test that MuJoCo Warp matches MuJoCo C reference for delays.

  Uses MuJoCo C as ground truth and compares outputs.
  """

  def test_actuator_delay_reference(self):
    """Compare actuator delay against MuJoCo C reference."""
    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="slide" delay="0.02" nsample="2"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # set ctrl=10 in both
    mjd.ctrl[0] = 10.0
    wp.copy(d.ctrl, wp.array(np.full((1, 1), 10.0), dtype=float))

    # step both 4 times and compare
    for i in range(4):
      mujoco.mj_step(mjm, mjd)
      forward.step(m, d)

      mj_force = mjd.actuator_force[0]
      warp_force = d.actuator_force.numpy()[0, 0]
      np.testing.assert_allclose(warp_force, mj_force, atol=_TOLERANCE, err_msg=f"actuator_force mismatch at step {i}")

  def test_sensor_delay_reference(self):
    """Compare sensor delay against MuJoCo C reference."""
    xml = """
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <sensor>
        <jointpos joint="slide" delay="0.02" nsample="3"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    for i in range(5):
      qpos_val = float((i + 1) * 10)
      mjd.qpos[0] = qpos_val
      wp.copy(d.qpos, wp.array(np.full((1, 1), qpos_val), dtype=float))

      mujoco.mj_step(mjm, mjd)
      forward.step(m, d)

      mj_sdata = mjd.sensordata[0]
      warp_sdata = d.sensordata.numpy()[0, 0]
      np.testing.assert_allclose(warp_sdata, mj_sdata, atol=_TOLERANCE, err_msg=f"sensordata mismatch at step {i}")


class PublicAPITest(absltest.TestCase):
  """Test public delay API functions against MuJoCo C reference."""

  def test_read_ctrl(self):
    """Test read_ctrl matches mj_readCtrl."""
    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="slide" delay="0.02" nsample="3" interp="linear"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # step both with ctrl=10, then ctrl=20
    for ctrl_val in [10.0, 20.0, 30.0]:
      mjd.ctrl[0] = ctrl_val
      wp.copy(d.ctrl, wp.array(np.full((1, 1), ctrl_val), dtype=float))
      mujoco.mj_step(mjm, mjd)
      forward.step(m, d)

    # compare read_ctrl at current time
    time_arr = d.time
    warp_result = wp.empty(d.nworld, dtype=float)
    delay.read_ctrl(m, d, 0, time_arr, interp=-1, result=warp_result)
    mj_result = mujoco.mj_readCtrl(mjm, mjd, 0, mjd.time, -1)
    np.testing.assert_allclose(
      warp_result.numpy()[0],
      mj_result,
      atol=_TOLERANCE,
      err_msg="read_ctrl mismatch",
    )

    # compare with explicit interp=0 (ZOH)
    warp_result_zoh = wp.empty(d.nworld, dtype=float)
    delay.read_ctrl(m, d, 0, time_arr, interp=0, result=warp_result_zoh)
    mj_result_zoh = mujoco.mj_readCtrl(mjm, mjd, 0, mjd.time, 0)
    np.testing.assert_allclose(
      warp_result_zoh.numpy()[0],
      mj_result_zoh,
      atol=_TOLERANCE,
      err_msg="read_ctrl ZOH mismatch",
    )

  def test_read_sensor(self):
    """Test read_sensor matches mj_readSensor."""
    xml = """
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <sensor>
        <jointpos joint="slide" delay="0.02" nsample="3" interp="linear"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    for i in range(4):
      qpos_val = float((i + 1) * 10)
      mjd.qpos[0] = qpos_val
      wp.copy(d.qpos, wp.array(np.full((1, 1), qpos_val), dtype=float))
      mujoco.mj_step(mjm, mjd)
      forward.step(m, d)

    # compare read_sensor at current time
    dim = mjm.sensor_dim[0]
    time_arr = d.time
    result = wp.empty((d.nworld, dim), dtype=float)
    delay.read_sensor(m, d, 0, time_arr, interp=-1, result=result)

    mj_result_buf = np.zeros(dim)
    ptr = mujoco.mj_readSensor(mjm, mjd, 0, mjd.time, mj_result_buf, -1)
    mj_val = ptr if ptr is not None else mj_result_buf

    np.testing.assert_allclose(
      result.numpy()[0],
      mj_val,
      atol=_TOLERANCE,
      err_msg="read_sensor mismatch",
    )

  def test_init_ctrl_history(self):
    """Test init_ctrl_history sets buffer correctly."""
    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="slide" delay="0.02" nsample="3"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    nsample = mjm.actuator_history[0, 0]

    # initialize with custom values
    custom_times = np.array([0.1, 0.2, 0.3])
    custom_values = np.array([100.0, 200.0, 300.0])
    times_wp = wp.array(custom_times, dtype=float)
    values_wp = wp.array(custom_values.reshape(1, -1), dtype=float)
    delay.init_ctrl_history(m, d, 0, times_wp, values_wp)

    # also init MuJoCo C side
    mujoco.mj_initCtrlHistory(mjm, mjd, 0, custom_times, custom_values)

    # read at a time in the buffer
    query_time = 0.23  # between samples → ZOH should return value at t=0.2
    time_arr = wp.array([query_time], dtype=float)
    warp_result = wp.empty(d.nworld, dtype=float)
    delay.read_ctrl(m, d, 0, time_arr, interp=0, result=warp_result)
    mj_result = mujoco.mj_readCtrl(mjm, mjd, 0, query_time, 0)
    np.testing.assert_allclose(
      warp_result.numpy()[0],
      mj_result,
      atol=_TOLERANCE,
      err_msg="init_ctrl_history read mismatch",
    )

  def test_init_sensor_history(self):
    """Test init_sensor_history sets buffer correctly."""
    xml = """
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <sensor>
        <jointpos joint="slide" delay="0.02" nsample="3"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    nsample = mjm.sensor_history[0, 0]
    dim = mjm.sensor_dim[0]

    # initialize with custom values
    custom_times = np.array([0.1, 0.2, 0.3])
    custom_values = np.array([100.0, 200.0, 300.0])
    phase = 0.05

    times_wp = wp.array(custom_times, dtype=float)
    values_wp = wp.array(custom_values.reshape(1, -1), dtype=float)
    phase_wp = wp.array([phase], dtype=float)
    delay.init_sensor_history(m, d, 0, times_wp, values_wp, phase=phase_wp)

    # also init MuJoCo C side
    mujoco.mj_initSensorHistory(mjm, mjd, 0, custom_times, custom_values, phase)

    # read at a time in the buffer
    query_time = 0.23
    time_arr = wp.array([query_time], dtype=float)
    result = wp.empty((1, dim), dtype=float)
    delay.read_sensor(m, d, 0, time_arr, interp=0, result=result)

    mj_result_buf = np.zeros(dim)
    ptr = mujoco.mj_readSensor(mjm, mjd, 0, query_time, mj_result_buf, 0)
    mj_val = ptr if ptr is not None else mj_result_buf

    np.testing.assert_allclose(
      result.numpy()[0],
      mj_val,
      atol=_TOLERANCE,
      err_msg="init_sensor_history read mismatch",
    )


if __name__ == "__main__":
  absltest.main()
