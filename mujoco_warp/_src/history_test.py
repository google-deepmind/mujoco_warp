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
from absl.testing import parameterized

from mujoco_warp import CtrlChart
from mujoco_warp import GainType
from mujoco_warp import State
from mujoco_warp import get_data_into
from mujoco_warp import get_state
from mujoco_warp import init_ctrl_history
from mujoco_warp import init_sensor_history
from mujoco_warp import read_ctrl
from mujoco_warp import read_sensor
from mujoco_warp import reset_data
from mujoco_warp import reset_history
from mujoco_warp import set_state
from mujoco_warp import step
from mujoco_warp import test_data
from mujoco_warp._src import util_pkg
from mujoco_warp._src.history import _reset_actuator_history_kernel
from mujoco_warp._src.history import read_ctrl_delayed

_TOLERANCE = 1e-8


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
      step(m, d)

    # compare read_ctrl at current time
    time_arr = d.time
    warp_result = wp.empty(d.nworld, dtype=float)
    read_ctrl(m, d, 0, time_arr, interp=-1, result=warp_result)
    if util_pkg.check_version("mujoco>=3.13.1.dev984848064"):
      mj_result = np.zeros(1)
      mujoco.mj_readCtrl(mjm, mjd, 0, mjd.time, mj_result, -1)
      mj_result = mj_result[0]
    else:
      mj_result = mujoco.mj_readCtrl(mjm, mjd, 0, mjd.time, -1)
    np.testing.assert_allclose(
      warp_result.numpy()[0],
      mj_result,
      atol=_TOLERANCE,
      err_msg="read_ctrl mismatch",
    )

    # compare with explicit interp=0 (ZOH)
    warp_result_zoh = wp.empty(d.nworld, dtype=float)
    read_ctrl(m, d, 0, time_arr, interp=0, result=warp_result_zoh)
    if util_pkg.check_version("mujoco>=3.13.1.dev984848064"):
      mj_result_zoh = np.zeros(1)
      mujoco.mj_readCtrl(mjm, mjd, 0, mjd.time, mj_result_zoh, 0)
      mj_result_zoh = mj_result_zoh[0]
    else:
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
      step(m, d)

    # compare read_sensor at current time
    dim = mjm.sensor_dim[0]
    time_arr = d.time
    result = wp.empty((d.nworld, dim), dtype=float)
    read_sensor(m, d, 0, time_arr, interp=-1, result=result)

    mj_result_buf = np.zeros(dim)
    ptr = mujoco.mj_readSensor(mjm, mjd, 0, mjd.time, mj_result_buf, -1)
    mj_val = ptr if ptr is not None else mj_result_buf

    np.testing.assert_allclose(
      result.numpy()[0],
      mj_val,
      atol=_TOLERANCE,
      err_msg="read_sensor mismatch",
    )

  def test_read_sensor_arbitrary_times(self):
    """Read sensor at arbitrary query times, compare against C."""
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
        <jointpos joint="slide" nsample="5"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    for i, qval in enumerate([1.0, 2.0, 3.0]):
      mjd.qpos[0] = qval
      wp.copy(d.qpos, wp.array(np.full((1, 1), qval), dtype=float))
      mujoco.mj_step(mjm, mjd)
      step(m, d)

    dim = mjm.sensor_dim[0]
    for query_time, expected in [(0.0, 1.0), (0.01, 2.0), (0.02, 3.0)]:
      time_arr = wp.array([query_time], dtype=float)
      result = wp.empty((1, dim), dtype=float)
      read_sensor(m, d, 0, time_arr, interp=0, result=result)

      mj_result_buf = np.zeros(dim)
      ptr = mujoco.mj_readSensor(mjm, mjd, 0, query_time, mj_result_buf, 0)
      mj_val = ptr if ptr is not None else mj_result_buf

      np.testing.assert_allclose(result.numpy()[0], mj_val, atol=_TOLERANCE, err_msg=f"read_sensor mismatch at t={query_time}")

  def test_read_sensor_second_index(self):
    """Test read_sensor for sensor index > 0 (OOB bug regression test)."""
    xml = """
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide1" type="slide" axis="1 0 0"/>
          <joint name="slide2" type="slide" axis="0 1 0"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <sensor>
        <jointpos joint="slide1" delay="0.02" nsample="3"/>
        <jointpos joint="slide2" delay="0.02" nsample="3"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    for i in range(4):
      qpos_val = float((i + 1) * 10)
      mjd.qpos[0] = qpos_val
      mjd.qpos[1] = qpos_val * 2
      wp.copy(d.qpos, wp.array(np.array([[qpos_val, qpos_val * 2]]), dtype=float))
      mujoco.mj_step(mjm, mjd)
      step(m, d)

    # read sensor 1 (index > 0) at current time
    dim = mjm.sensor_dim[1]
    time_arr = d.time
    result = wp.empty((1, dim), dtype=float)
    read_sensor(m, d, 1, time_arr, interp=-1, result=result)

    mj_result_buf = np.zeros(dim)
    ptr = mujoco.mj_readSensor(mjm, mjd, 1, mjd.time, mj_result_buf, -1)
    mj_val = ptr if ptr is not None else mj_result_buf

    np.testing.assert_allclose(
      result.numpy()[0], mj_val, atol=_TOLERANCE, err_msg="read_sensor mismatch for sensor index 1 (OOB regression)"
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

    # initialize with custom values
    custom_times = np.array([0.1, 0.2, 0.3])
    custom_values = np.array([100.0, 200.0, 300.0])
    times_wp = wp.array(custom_times, dtype=float)
    values_wp = wp.array(custom_values.reshape(1, -1), dtype=float)
    init_ctrl_history(m, d, 0, times_wp, values_wp)

    # also init MuJoCo C side
    mujoco.mj_initCtrlHistory(mjm, mjd, 0, custom_times, custom_values)

    # read at a time in the buffer
    query_time = 0.23  # between samples → ZOH should return value at t=0.2
    time_arr = wp.array([query_time], dtype=float)
    warp_result = wp.empty(d.nworld, dtype=float)
    read_ctrl(m, d, 0, time_arr, interp=0, result=warp_result)
    if util_pkg.check_version("mujoco>=3.13.1.dev984848064"):
      mj_result = np.zeros(1)
      mujoco.mj_readCtrl(mjm, mjd, 0, query_time, mj_result, 0)
      mj_result = mj_result[0]
    else:
      mj_result = mujoco.mj_readCtrl(mjm, mjd, 0, query_time, 0)
    np.testing.assert_allclose(
      warp_result.numpy()[0],
      mj_result,
      atol=_TOLERANCE,
      err_msg="init_ctrl_history read mismatch",
    )

  def test_init_ctrl_history_mimo(self):
    """Test init_ctrl_history and read_ctrl with multi-input actuator."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <joint name="hinge" type="hinge"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
        <actuator>
          <dcmotor joint="hinge" motorconst="1.0" resistance="1.0" input="pos vel ff"
                   delay="0.02" nsample="3"/>
        </actuator>
      </mujoco>
      """
    )
    self.assertEqual(int(m.actuator_ctrlnum.numpy()[0]), 3)

    nhistory = 2 + 3 + 3 * 3
    if m.nhistory < nhistory:
      m.nhistory = nhistory
      d.history = wp.zeros((d.nworld, nhistory), dtype=float)

    times = np.array([0.0, 0.01, 0.02])
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    times_wp = wp.array(times, dtype=float)
    values_wp = wp.array(values.reshape(1, 9), dtype=float)
    init_ctrl_history(m, d, 0, times_wp, values_wp)

    # ZOH read at 0.03 (lookup at 0.01 -> [4.0, 5.0, 6.0])
    res = wp.empty((d.nworld, 3), dtype=float)
    res.fill_(wp.inf)
    read_ctrl(m, d, 0, wp.array([0.03], dtype=float), 0, res)
    np.testing.assert_allclose(res.numpy()[0], [4.0, 5.0, 6.0])

    # Linear interpolation at 0.035 (lookup at 0.015 -> [5.5, 6.5, 7.5])
    res.fill_(wp.inf)
    read_ctrl(m, d, 0, wp.array([0.035], dtype=float), 1, res)
    np.testing.assert_allclose(res.numpy()[0], [5.5, 6.5, 7.5])

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

    dim = mjm.sensor_dim[0]

    # initialize with custom values
    custom_times = np.array([0.1, 0.2, 0.3])
    custom_values = np.array([100.0, 200.0, 300.0])
    phase = 0.05

    times_wp = wp.array(custom_times, dtype=float)
    values_wp = wp.array(custom_values.reshape(1, -1), dtype=float)
    phase_wp = wp.array([phase], dtype=float)
    init_sensor_history(m, d, 0, times_wp, values_wp, phase=phase_wp)

    # also init MuJoCo C side
    mujoco.mj_initSensorHistory(mjm, mjd, 0, custom_times, custom_values, phase)

    # read at a time in the buffer
    query_time = 0.23
    time_arr = wp.array([query_time], dtype=float)
    result = wp.empty((1, dim), dtype=float)
    read_sensor(m, d, 0, time_arr, interp=0, result=result)

    mj_result_buf = np.zeros(dim)
    ptr = mujoco.mj_readSensor(mjm, mjd, 0, query_time, mj_result_buf, 0)
    mj_val = ptr if ptr is not None else mj_result_buf

    np.testing.assert_allclose(
      result.numpy()[0],
      mj_val,
      atol=_TOLERANCE,
      err_msg="init_sensor_history read mismatch",
    )

  def test_actuator_history_only(self):
    """Actuator with delay=0 records history but applies ctrl immediately."""
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
        <motor joint="slide" delay="0.0" nsample="3"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertGreater(m.nhistory, 0)

    # ctrl applied immediately when delay=0
    for ctrl_val in [10.0, 20.0, 30.0]:
      wp.copy(d.ctrl, wp.array(np.full((1, 1), ctrl_val), dtype=float))
      step(m, d)
      np.testing.assert_allclose(d.actuator_force.numpy()[0, 0], ctrl_val, atol=_TOLERANCE)

    # history buffer still readable
    result = wp.empty(d.nworld, dtype=float)
    time_arr = wp.array([0.015], dtype=float)
    read_ctrl(m, d, 0, time_arr, interp=0, result=result)
    np.testing.assert_allclose(result.numpy()[0], 20.0, atol=_TOLERANCE)

  def test_sensor_history_only(self):
    """Sensor with delay=0 records history but reports current value."""
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
        <jointpos joint="slide" delay="0.0" nsample="5"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertGreater(m.nhistory, 0)

    wp.copy(d.qpos, wp.array(np.full((1, 1), 10.0), dtype=float))
    step(m, d)
    np.testing.assert_allclose(d.sensordata.numpy()[0, 0], 10.0, atol=_TOLERANCE)

    # history buffer still readable
    time_arr = wp.array([d.time.numpy()[0] - 0.005], dtype=float)
    result = wp.empty((d.nworld, 1), dtype=float)
    read_sensor(m, d, 0, time_arr, interp=0, result=result)
    np.testing.assert_allclose(result.numpy()[0, 0], 10.0, atol=_TOLERANCE)


class MultiWorldDelayTest(parameterized.TestCase):
  """Test delay with nworld > 1 and varying delay values."""

  @parameterized.parameters(1, 2)
  def test_actuator_delay(self, nworld):
    for delay, nsample, nzero in ((0.02, 2, 2), (0.04, 5, 4)):
      xml = f"""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <body>
            <joint name="slide" type="slide"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="slide" delay="{delay}" nsample="{nsample}"/>
        </actuator>
      </mujoco>
      """
      mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

      wp.copy(d.ctrl, wp.array(np.full((nworld, 1), 10.0), dtype=float))

      for i in range(nzero):
        step(m, d)
        for w in range(nworld):
          act_force = d.actuator_force.numpy()[w, 0]
          np.testing.assert_allclose(
            act_force,
            0.0,
            atol=_TOLERANCE,
            err_msg=f"nworld={nworld} delay={delay} world={w} step {i}",
          )

      step(m, d)
      for w in range(nworld):
        act_force = d.actuator_force.numpy()[w, 0]
        np.testing.assert_allclose(
          act_force,
          10.0,
          atol=_TOLERANCE,
          err_msg=f"nworld={nworld} delay={delay} world={w} step {nzero}",
        )

  @parameterized.parameters(1, 2)
  def test_actuator_delay_mimo(self, nworld):
    """Test delayed actuation with multi-input actuator delays all controls."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0"/>
        <worldbody>
          <body>
            <joint name="hinge" type="hinge"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
        <actuator>
          <dcmotor joint="hinge" motorconst="1.0" resistance="1.0" input="pos vel ff"
                   delay="0.02" nsample="2"/>
        </actuator>
      </mujoco>
      """,
      nworld=nworld,
    )
    self.assertEqual(int(m.actuator_ctrlnum.numpy()[0]), 3)

    nhistory = 2 + 2 + 2 * 3
    if m.nhistory < nhistory:
      m.nhistory = nhistory
      d.history = wp.zeros((nworld, nhistory), dtype=float)
      reset_history(m, d)

    ctrl_in = np.zeros((nworld, m.nu), dtype=float)
    ctrl_in[0] = [1.0, 2.0, 5.0]
    if nworld == 2:
      ctrl_in[1] = [2.0, 4.0, 10.0]
    d.ctrl = wp.array(ctrl_in, dtype=float)

    delayed_ctrl = wp.empty((d.nworld, m.nu), dtype=float)

    # Before step 1 (t=0.0):
    delayed_ctrl.fill_(wp.inf)
    read_ctrl_delayed(m, d, delayed_ctrl)
    for w in range(nworld):
      np.testing.assert_allclose(delayed_ctrl.numpy()[w], [0.0, 0.0, 0.0])

    # Step 1: advances t from 0.0 to 0.01
    step(m, d)
    delayed_ctrl.fill_(wp.inf)
    read_ctrl_delayed(m, d, delayed_ctrl)
    for w in range(nworld):
      np.testing.assert_allclose(delayed_ctrl.numpy()[w], [0.0, 0.0, 0.0])

    # Step 2: advances t from 0.01 to 0.02 (delayed inputs arrive at t=0.02)
    step(m, d)
    delayed_ctrl.fill_(wp.inf)
    read_ctrl_delayed(m, d, delayed_ctrl)
    np.testing.assert_allclose(delayed_ctrl.numpy()[0], [1.0, 2.0, 5.0])
    if nworld == 2:
      np.testing.assert_allclose(delayed_ctrl.numpy()[1], [2.0, 4.0, 10.0])
      self.assertFalse(np.allclose(delayed_ctrl.numpy()[0], delayed_ctrl.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_sensor_delay(self, nworld):
    for delay, nsample, expected in (
      (0.02, 3, [0.0, 0.0, 10.0, 20.0]),
      (0.04, 5, [0.0, 0.0, 0.0, 0.0, 10.0]),
    ):
      xml = f"""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0"/>
        <worldbody>
          <body>
            <joint name="slide" type="slide"/>
            <geom size="0.1"/>
          </body>
        </worldbody>
        <sensor>
          <jointpos joint="slide" delay="{delay}" nsample="{nsample}"/>
        </sensor>
      </mujoco>
      """
      mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

      for i in range(len(expected)):
        qpos_val = float((i + 1) * 10)
        wp.copy(d.qpos, wp.array(np.full((nworld, 1), qpos_val), dtype=float))
        step(m, d)
        for w in range(nworld):
          sdata = d.sensordata.numpy()[w, 0]
          np.testing.assert_allclose(
            sdata,
            expected[i],
            atol=_TOLERANCE,
            err_msg=f"nworld={nworld} delay={delay} world={w} step {i}",
          )

  @parameterized.parameters(1, 2)
  def test_delayed_joint_limit_force(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <body>
            <joint name="slide" type="slide" range="0 1"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="slide"/>
        </actuator>
        <sensor>
          <jointlimitfrc joint="slide" delay="0.02" nsample="5"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
    )

    mjds = [mjd]
    qpos = d.qpos.numpy()
    qpos[0, 0] = 1.05
    mjd.qpos[0] = 1.05

    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      qpos[1, 0] = 1.15
      mjd1.qpos[0] = 1.15
      mjds.append(mjd1)

    d.qpos.assign(qpos)

    all_warp_outputs = [[] for _ in range(nworld)]

    for _ in range(6):
      d.sensordata.fill_(wp.inf)
      step(m, d)
      for w in range(nworld):
        mujoco.mj_step(mjm, mjds[w])
        all_warp_outputs[w].append(d.sensordata.numpy()[w, 0])
        np.testing.assert_allclose(
          d.sensordata.numpy()[w],
          mjds[w].sensordata,
          rtol=1e-3,
          atol=1e-3,
        )

    if nworld == 2:
      self.assertFalse(np.allclose(all_warp_outputs[0], all_warp_outputs[1]))

  @parameterized.parameters(1, 2)
  def test_delayed_tendon_limit_force(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <body>
            <joint name="slide" type="slide"/>
            <geom size="0.1" mass="1"/>
            <site name="s1" pos="0 0 0"/>
          </body>
          <site name="s2" pos="0 0 1"/>
        </worldbody>
        <tendon>
          <spatial name="ten" range="0 0.5">
            <site site="s1"/>
            <site site="s2"/>
          </spatial>
        </tendon>
        <sensor>
          <tendonlimitfrc tendon="ten" delay="0.02" nsample="5"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
    )

    mjds = [mjd]
    qpos = d.qpos.numpy()
    qpos[0, 0] = -0.7
    mjd.qpos[0] = -0.7

    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      qpos[1, 0] = -0.9
      mjd1.qpos[0] = -0.9
      mjds.append(mjd1)

    d.qpos.assign(qpos)

    all_warp_outputs = [[] for _ in range(nworld)]

    for _ in range(6):
      d.sensordata.fill_(wp.inf)
      step(m, d)
      for w in range(nworld):
        mujoco.mj_step(mjm, mjds[w])
        all_warp_outputs[w].append(d.sensordata.numpy()[w, 0])
        np.testing.assert_allclose(
          d.sensordata.numpy()[w],
          mjds[w].sensordata,
          rtol=1e-3,
          atol=1e-3,
        )

    if nworld == 2:
      self.assertFalse(np.allclose(all_warp_outputs[0], all_warp_outputs[1]))


class MultiActuatorSensorDelayTest(absltest.TestCase):
  """Test delay with multiple actuators/sensors with different delays."""

  def test_multi_actuator_delay(self):
    """2 actuators with different delays."""
    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body>
          <joint name="slide1" type="slide" axis="1 0 0"/>
          <joint name="slide2" type="slide" axis="0 1 0"/>
          <geom size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="slide1" delay="0.02" nsample="2"/>
        <motor joint="slide2" delay="0.03" nsample="3"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # set both ctrl to 10
    mjd.ctrl[:] = 10.0
    wp.copy(d.ctrl, wp.array(np.full((1, 2), 10.0), dtype=float))

    for i in range(5):
      mujoco.mj_step(mjm, mjd)
      step(m, d)

      mj_force = mjd.actuator_force.copy()
      warp_force = d.actuator_force.numpy()[0]
      np.testing.assert_allclose(warp_force, mj_force, atol=_TOLERANCE, err_msg=f"actuator_force mismatch at step {i}")

  def test_multi_sensor_delay(self):
    """2 sensors with different delays."""
    xml = """
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide1" type="slide" axis="1 0 0"/>
          <joint name="slide2" type="slide" axis="0 1 0"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <sensor>
        <jointpos joint="slide1" delay="0.02" nsample="3"/>
        <jointpos joint="slide2" delay="0.03" nsample="4"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    for i in range(5):
      qpos_val = float((i + 1) * 10)
      mjd.qpos[:] = qpos_val
      wp.copy(d.qpos, wp.array(np.full((1, mjm.nq), qpos_val), dtype=float))

      mujoco.mj_step(mjm, mjd)
      step(m, d)

      mj_sdata = mjd.sensordata.copy()
      warp_sdata = d.sensordata.numpy()[0, : mjm.nsensordata]
      np.testing.assert_allclose(warp_sdata, mj_sdata, atol=_TOLERANCE, err_msg=f"sensordata mismatch at step {i}")


class InterpolationTest(parameterized.TestCase):
  """Test linear and cubic interpolation against MuJoCo C reference."""

  @parameterized.parameters(
    ("linear", 1, 0.015, 3),
    ("cubic", 2, 0.015, 5),
  )
  def test_actuator_interp(self, interp_name, interp_val, delay, nsample):
    xml = f"""
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="slide" delay="{delay}" nsample="{nsample}"
               interp="{interp_name}"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertEqual(mjm.actuator_history[0, 1], interp_val)

    for i in range(6):
      ctrl_val = float((i + 1) * 10)
      mjd.ctrl[0] = ctrl_val
      wp.copy(d.ctrl, wp.array(np.full((1, 1), ctrl_val), dtype=float))
      mujoco.mj_step(mjm, mjd)
      step(m, d)

      mj_force = mjd.actuator_force[0]
      warp_force = d.actuator_force.numpy()[0, 0]
      np.testing.assert_allclose(warp_force, mj_force, atol=_TOLERANCE, err_msg=f"{interp_name} interp mismatch at step {i}")

  @parameterized.parameters(
    ("linear", 1, 0.015, 3),
    ("cubic", 2, 0.015, 5),
  )
  def test_sensor_interp(self, interp_name, interp_val, delay, nsample):
    xml = f"""
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <sensor>
        <jointpos joint="slide" delay="{delay}" nsample="{nsample}"
                  interp="{interp_name}"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertEqual(mjm.sensor_history[0, 1], interp_val)

    for i in range(6):
      qpos_val = float((i + 1) * 10)
      mjd.qpos[0] = qpos_val
      wp.copy(d.qpos, wp.array(np.full((1, 1), qpos_val), dtype=float))
      mujoco.mj_step(mjm, mjd)
      step(m, d)

      mj_sdata = mjd.sensordata[0]
      warp_sdata = d.sensordata.numpy()[0, 0]
      np.testing.assert_allclose(warp_sdata, mj_sdata, atol=_TOLERANCE, err_msg=f"{interp_name} interp mismatch at step {i}")

  def test_read_ctrl_cubic(self):
    """read_ctrl with cubic interpolation, compared to MuJoCo C."""
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
        <motor joint="slide" delay="0.015" nsample="5" interp="cubic"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    for ctrl_val in [10.0, 20.0, 30.0, 40.0, 50.0]:
      mjd.ctrl[0] = ctrl_val
      wp.copy(d.ctrl, wp.array(np.full((1, 1), ctrl_val), dtype=float))
      mujoco.mj_step(mjm, mjd)
      step(m, d)

    # compare read_ctrl with cubic interp at current time
    time_arr = d.time
    warp_result = wp.empty(d.nworld, dtype=float)
    read_ctrl(m, d, 0, time_arr, interp=2, result=warp_result)
    if util_pkg.check_version("mujoco>=3.13.1.dev984848064"):
      mj_result = np.zeros(1)
      mujoco.mj_readCtrl(mjm, mjd, 0, mjd.time, mj_result, 2)
      mj_result = mj_result[0]
    else:
      mj_result = mujoco.mj_readCtrl(mjm, mjd, 0, mjd.time, 2)
    np.testing.assert_allclose(warp_result.numpy()[0], mj_result, atol=_TOLERANCE, err_msg="read_ctrl cubic interp mismatch")


class SensorFeatureTest(absltest.TestCase):
  """Test sensor-specific features: interval and multi-dimensional sensors."""

  def test_sensor_delay_interval(self):
    """Combined delay + interval, matching SensorDelayInterval C test."""
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
        <jointpos joint="slide" delay="0.02" interval="0.03 0" nsample="5"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertEqual(mjm.sensor_history[0, 0], 5)
    np.testing.assert_allclose(mjm.sensor_delay[0], 0.02, atol=1e-10)

    # set position
    mjd.qpos[0] = 5.0
    wp.copy(d.qpos, wp.array(np.full((1, 1), 5.0), dtype=float))

    for i in range(3):
      mujoco.mj_step(mjm, mjd)
      step(m, d)
      np.testing.assert_allclose(d.sensordata.numpy()[0, 0], mjd.sensordata[0], atol=_TOLERANCE, err_msg=f"step {i}")

  def test_sensor_delay_multi_dim(self):
    """3D ballangvel with delay, matching SensorDelayMultiDim C test."""
    xml = """
    <mujoco>
      <option timestep="0.01"/>
      <worldbody>
        <body>
          <joint name="ball" type="ball"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <sensor>
        <ballangvel joint="ball" delay="0.02" nsample="2"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertEqual(mjm.sensor_dim[0], 3)

    # set angular velocity
    mjd.qvel[:] = [1.0, 2.0, 3.0]
    wp.copy(d.qvel, wp.array(np.array([[1.0, 2.0, 3.0]]), dtype=float))

    for _ in range(4):
      mujoco.mj_step(mjm, mjd)
      step(m, d)
      warp_sdata = d.sensordata.numpy()[0, :3]
      np.testing.assert_allclose(warp_sdata, mjd.sensordata[:3], atol=_TOLERANCE)


class MonotonicityCheckTest(absltest.TestCase):
  """Test strict monotonicity check in init functions."""

  def test_init_ctrl_history_monotonicity(self):
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
    _, _, m, d = test_data.fixture(xml=xml)

    values = wp.array(np.full((1, 3), 1.0), dtype=float)

    # non-monotonic and equal times should raise
    for bad in ([0.1, 0.05, 0.2], [0.1, 0.1, 0.2]):
      with self.assertRaises(ValueError):
        init_ctrl_history(m, d, 0, wp.array(bad, dtype=float), values)

    # good times should succeed
    init_ctrl_history(m, d, 0, wp.array([0.1, 0.2, 0.3], dtype=float), values)

  def test_init_sensor_history_monotonicity(self):
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
    _, _, m, d = test_data.fixture(xml=xml)

    values = wp.array(np.full((1, 3), 1.0), dtype=float)
    phase = wp.array([0.0], dtype=float)

    with self.assertRaises(ValueError):
      init_sensor_history(m, d, 0, wp.array([0.3, 0.2, 0.1], dtype=float), values, phase=phase)


class BufferMechanicsTest(parameterized.TestCase):
  """Test buffer mechanics: save/restore, out-of-order insertion."""

  @parameterized.parameters(1, 2)
  def test_state_save_restore_actuator(self, nworld):
    """Save/restore history state gives identical actuator delay results."""
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
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    # step 3 times with ctrl=10
    wp.copy(d.ctrl, wp.array(np.full((nworld, 1), 10.0), dtype=float))
    for _ in range(3):
      step(m, d)

    # save state
    saved_history = wp.empty_like(d.history)
    wp.copy(saved_history, d.history)
    saved_time = d.time.numpy().copy()
    saved_qpos = d.qpos.numpy().copy()
    saved_qvel = d.qvel.numpy().copy()
    saved_qacc_warmstart = d.qacc_warmstart.numpy().copy()

    # step 3 more times with ctrl=20, record final force
    wp.copy(d.ctrl, wp.array(np.full((nworld, 1), 20.0), dtype=float))
    for _ in range(3):
      step(m, d)
    force_after = d.actuator_force.numpy().copy()

    # restore saved state
    wp.copy(d.history, saved_history)
    wp.copy(d.time, wp.array(saved_time, dtype=float))
    wp.copy(d.qpos, wp.array(saved_qpos, dtype=float))
    wp.copy(d.qvel, wp.array(saved_qvel, dtype=float))
    wp.copy(d.qacc_warmstart, wp.array(saved_qacc_warmstart, dtype=float))

    # step 3 more times with same ctrl=20
    wp.copy(d.ctrl, wp.array(np.full((nworld, 1), 20.0), dtype=float))
    for _ in range(3):
      step(m, d)
    force_restored = d.actuator_force.numpy().copy()

    np.testing.assert_allclose(
      force_after,
      force_restored,
      atol=_TOLERANCE,
      err_msg=f"nworld={nworld}: force mismatch after state restore",
    )

  @parameterized.parameters(1, 2)
  def test_state_save_restore_sensor(self, nworld):
    """Save/restore history state gives identical sensor delay results."""
    xml = """
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="slide"/>
      </actuator>
      <sensor>
        <jointpos joint="slide" delay="0.02" nsample="3"/>
      </sensor>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    # step 3 times with ctrl=5
    wp.copy(d.ctrl, wp.array(np.full((nworld, 1), 5.0), dtype=float))
    for _ in range(3):
      step(m, d)

    # save state
    saved_history = wp.empty_like(d.history)
    wp.copy(saved_history, d.history)
    saved_time = d.time.numpy().copy()
    saved_qpos = d.qpos.numpy().copy()
    saved_qvel = d.qvel.numpy().copy()
    saved_qacc_warmstart = d.qacc_warmstart.numpy().copy()

    # step 3 more times with ctrl=10
    wp.copy(d.ctrl, wp.array(np.full((nworld, 1), 10.0), dtype=float))
    for _ in range(3):
      step(m, d)
    sensor_after = d.sensordata.numpy().copy()

    # restore saved state
    wp.copy(d.history, saved_history)
    wp.copy(d.time, wp.array(saved_time, dtype=float))
    wp.copy(d.qpos, wp.array(saved_qpos, dtype=float))
    wp.copy(d.qvel, wp.array(saved_qvel, dtype=float))
    wp.copy(d.qacc_warmstart, wp.array(saved_qacc_warmstart, dtype=float))

    # step 3 more times with same ctrl=10
    wp.copy(d.ctrl, wp.array(np.full((nworld, 1), 10.0), dtype=float))
    for _ in range(3):
      step(m, d)
    sensor_restored = d.sensordata.numpy().copy()

    np.testing.assert_allclose(
      sensor_after,
      sensor_restored,
      atol=_TOLERANCE,
      err_msg=f"nworld={nworld}: sensor mismatch after state restore",
    )

  @parameterized.parameters(1, 2)
  def test_insert_and_read_at_middle_time(self, nworld):
    """Insert values at specific times, then read at a time between them."""
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
        <motor joint="slide" delay="0.02" nsample="5" interp="linear"/>
      </actuator>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    times = wp.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    values = wp.array(np.tile([1.0, 2.0, 3.0, 4.0, 5.0], (nworld, 1)), dtype=float)
    init_ctrl_history(m, d, 0, times, values)

    # read_ctrl reads at (query_time - delay). delay=0.02.
    delay = 0.02
    for read_t, expected in [(0.25, 2.5), (0.35, 3.5)]:
      query_time = wp.array(np.full(nworld, read_t + delay), dtype=float)
      result = wp.empty((nworld,), dtype=float)
      read_ctrl(m, d, 0, query_time, 1, result)
      for w in range(nworld):
        np.testing.assert_allclose(
          result.numpy()[w],
          expected,
          atol=_TOLERANCE,
          err_msg=f"nworld={nworld}, world={w}: read at t={read_t}",
        )

  def test_init_replace_on_collision(self):
    """Initializing with same times replaces values (exact match path)."""
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
    _, _, m, d = test_data.fixture(xml=xml)

    times = wp.array([0.1, 0.2, 0.3], dtype=float)
    delay = 0.02
    query = wp.array([0.2 + delay], dtype=float)
    result = wp.empty((1,), dtype=float)

    # init with values [1, 2, 3], read at t=0.2
    init_ctrl_history(m, d, 0, times, wp.array(np.array([[1.0, 2.0, 3.0]]), dtype=float))
    read_ctrl(m, d, 0, query, 0, result)
    np.testing.assert_allclose(result.numpy()[0], 2.0, atol=_TOLERANCE)

    # re-init with same times but different values [10, 20, 30]
    init_ctrl_history(m, d, 0, times, wp.array(np.array([[10.0, 20.0, 30.0]]), dtype=float))
    read_ctrl(m, d, 0, query, 0, result)
    np.testing.assert_allclose(result.numpy()[0], 20.0, atol=_TOLERANCE)


class StressWrapTest(parameterized.TestCase):
  """Stress test: many steps to exercise circular buffer wrapping."""

  @parameterized.parameters(1, 2)
  def test_long_actuator_delay_wrapping(self, nworld):
    """25 steps with nsample=5 to exercise full circular buffer wrapping.

    Compare each step's actuator_force against MuJoCo C reference.
    Uses nsample=5 (more than minimum) to exercise wrapping with excess slots.
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
        <motor joint="slide" delay="0.02" nsample="5"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    nsteps = 25
    for step_i in range(nsteps):
      ctrl_val = float((step_i + 1) * 3.0)  # varying ctrl

      # C reference
      mjd.ctrl[0] = ctrl_val
      mujoco.mj_step(mjm, mjd)

      # Warp
      wp.copy(d.ctrl, wp.array(np.full((nworld, 1), ctrl_val), dtype=float))
      step(m, d)

      mj_force = mjd.actuator_force[0]
      warp_force = d.actuator_force.numpy()[0, 0]
      np.testing.assert_allclose(
        warp_force,
        mj_force,
        atol=_TOLERANCE,
        err_msg=f"nworld={nworld}, step {step_i}: force {warp_force} vs C {mj_force}",
      )

  @parameterized.parameters(1, 2)
  def test_long_sensor_delay_wrapping(self, nworld):
    """25 steps with nsample=5 to exercise full sensor circular buffer wrapping."""
    xml = """
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <body>
          <joint name="slide" type="slide"/>
          <geom size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="slide"/>
      </actuator>
      <sensor>
        <jointpos joint="slide" delay="0.02" nsample="5"/>
      </sensor>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    nsteps = 25
    for step_i in range(nsteps):
      ctrl_val = float((step_i + 1) * 2.0)

      mjd.ctrl[0] = ctrl_val
      mujoco.mj_step(mjm, mjd)

      wp.copy(d.ctrl, wp.array(np.full((nworld, 1), ctrl_val), dtype=float))
      step(m, d)

      mj_sensor = mjd.sensordata[0]
      warp_sensor = d.sensordata.numpy()[0, 0]
      np.testing.assert_allclose(
        warp_sensor,
        mj_sensor,
        atol=_TOLERANCE,
        err_msg=f"nworld={nworld}, step {step_i}: sensor {warp_sensor} vs C {mj_sensor}",
      )


class ActivationDelayTest(parameterized.TestCase):
  """Test combining activation dynamics with actuator delay."""

  @parameterized.parameters(1, 2)
  def test_filter_with_delay(self, nworld):
    """Actuator with dyntype=filter and delay should match C reference.

    The activation dynamics da/dt = (u-a)/tau are independent of delay,
    but the ctrl signal fed to dynamics is the delayed ctrl.
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
        <general joint="slide" dyntype="filter" dynprm="0.05"
                 gainprm="1" biastype="none"
                 delay="0.02" nsample="3"/>
      </actuator>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    nsteps = 15
    for step_i in range(nsteps):
      ctrl_val = 1.0

      mjd.ctrl[0] = ctrl_val
      mujoco.mj_step(mjm, mjd)

      wp.copy(d.ctrl, wp.array(np.full((nworld, 1), ctrl_val), dtype=float))
      step(m, d)

      # compare activation
      mj_act = mjd.act[0]
      warp_act = d.act.numpy()[0, 0]
      np.testing.assert_allclose(
        warp_act,
        mj_act,
        atol=_TOLERANCE,
        err_msg=f"nworld={nworld}, step {step_i}: act {warp_act} vs C {mj_act}",
      )

      # compare force
      mj_force = mjd.actuator_force[0]
      warp_force = d.actuator_force.numpy()[0, 0]
      np.testing.assert_allclose(
        warp_force,
        mj_force,
        atol=_TOLERANCE,
        err_msg=f"nworld={nworld}, step {step_i}: force {warp_force} vs C {mj_force}",
      )

  @parameterized.parameters(1, 2)
  def test_integrator_with_delay(self, nworld):
    """Actuator with dyntype=integrator and delay should match C reference."""
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
        <general joint="slide" dyntype="integrator"
                 gainprm="1" biastype="none"
                 delay="0.02" nsample="3"/>
      </actuator>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    nsteps = 10
    for step_i in range(nsteps):
      ctrl_val = 5.0

      mjd.ctrl[0] = ctrl_val
      mujoco.mj_step(mjm, mjd)

      wp.copy(d.ctrl, wp.array(np.full((nworld, 1), ctrl_val), dtype=float))
      step(m, d)

      mj_act = mjd.act[0]
      warp_act = d.act.numpy()[0, 0]
      np.testing.assert_allclose(
        warp_act,
        mj_act,
        atol=_TOLERANCE,
        err_msg=f"nworld={nworld}, step {step_i}: act {warp_act} vs C {mj_act}",
      )


class StateParityTest(absltest.TestCase):
  """Test state parity features for get_state, set_state, and get_data_into."""

  def test_get_data_into_history(self):
    """Test get_data_into copies history from device back to host."""
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

    # step both with ctrl
    wp.copy(d.ctrl, wp.array(np.full((1, 1), 100.0), dtype=float))
    step(m, d)

    # check history on device is non-empty
    dev_history = d.history.numpy()[0]
    self.assertTrue((dev_history != 0.0).any())

    # pull data into host
    get_data_into(mjd, mjm, d, world_id=0)

    # assert parity
    np.testing.assert_allclose(mjd.history, dev_history, atol=_TOLERANCE)

  def test_state_history_get_set(self):
    """Test get_state and set_state correctly serialize/deserialize HISTORY."""
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
    mjm, _, m, d = test_data.fixture(xml=xml)

    # Populate history on device
    wp.copy(d.ctrl, wp.array(np.full((1, 1), 50.0), dtype=float))
    step(m, d)

    # Get size of State.PHYSICS
    state_size = mujoco.mj_stateSize(mjm, State.PHYSICS.value)
    state_wp = wp.empty((1, state_size), dtype=float)
    get_state(m, d, state_wp, State.PHYSICS.value)

    # Check non-zero state contents
    state_np = state_wp.numpy()[0]
    self.assertTrue((state_np != 0.0).any())

    # Create new clean data, set_state onto it, and verify history matches
    _, _, _, d2 = test_data.fixture(xml=xml)
    set_state(m, d2, state_wp, State.PHYSICS.value)

    np.testing.assert_allclose(d2.history.numpy(), d.history.numpy(), atol=_TOLERANCE)


class ResetHistoryTest(parameterized.TestCase):
  """Tests for full history and delay reset matching MuJoCo C."""

  @parameterized.parameters(1, 2)
  def test_reset_history_parity_and_reproducibility(self, nworld):
    """Test initial make_data parity, reset_data parity, and trajectory reproducibility."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
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
        <sensor>
          <jointpos joint="slide" delay="0.02" nsample="3"/>
          <jointvel joint="slide" interval="0.05 0" nsample="4"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
    )

    # 1. Initial make_data parity with MuJoCo C
    self.assertGreater(mjm.nhistory, 0)
    for w in range(nworld):
      np.testing.assert_allclose(d.history.numpy()[w], mjd.history, atol=1e-6)

    # 2. Step pass 1 and verify history changes
    d.ctrl.fill_(10.0)
    for _ in range(15):
      step(m, d)
    qpos_pass1 = d.qpos.numpy().copy()
    sensordata_pass1 = d.sensordata.numpy().copy()
    self.assertFalse(np.allclose(d.history.numpy()[0], mjd.history))

    # 3. reset_data parity with mj_resetData
    reset_data(m, d)
    mujoco.mj_resetData(mjm, mjd)
    for w in range(nworld):
      np.testing.assert_allclose(d.history.numpy()[w], mjd.history, atol=1e-6)

    # 4. Step pass 2 and verify trajectory reproducibility
    d.ctrl.fill_(10.0)
    for _ in range(15):
      step(m, d)
    np.testing.assert_allclose(d.qpos.numpy(), qpos_pass1, atol=1e-6)
    np.testing.assert_allclose(d.sensordata.numpy(), sensordata_pass1, atol=1e-6)

  @parameterized.parameters(1, 2)
  def test_selective_reset_data(self, nworld):
    """Test selective reset mask resets targeted worlds while preserving others."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
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
      """,
      nworld=nworld,
    )

    d.ctrl.fill_(5.0)
    for _ in range(10):
      step(m, d)
    hist_before = d.history.numpy().copy()

    # World 0 is reset, world 1 (if present) is not reset
    mask_list = [True] if nworld == 1 else [True, False]
    mask = wp.array(mask_list, dtype=bool)
    reset_data(m, d, reset=mask)

    hist_after = d.history.numpy()
    mujoco.mj_resetData(mjm, mjd)

    # World 0 should be reset to initial state
    np.testing.assert_allclose(hist_after[0], mjd.history, atol=1e-6)
    # Unreset worlds should be preserved
    for w in range(1, nworld):
      np.testing.assert_allclose(hist_after[w], hist_before[w], atol=1e-6)

    # Selective reset with reverse mask
    if nworld == 2:
      for _ in range(10):
        step(m, d)
      hist_dirty = d.history.numpy().copy()
      mask_alt = wp.array([False, True], dtype=bool)
      reset_data(m, d, reset=mask_alt)
      hist_after_alt = d.history.numpy()
      # World 0 preserved dirty, World 1 reset
      np.testing.assert_allclose(hist_after_alt[0], hist_dirty[0], atol=1e-6)
      np.testing.assert_allclose(hist_after_alt[1], mjd.history, atol=1e-6)

  @parameterized.parameters(1, 2)
  def test_interval_sensor_at_t0(self, nworld):
    """Test interval sensor with phase=0 computes at t=0 matching MuJoCo C."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <body>
            <joint name="slide" type="slide"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
        <sensor>
          <jointpos joint="slide" interval="0.05 0" nsample="3"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Move joint position before first step
    d.qpos.fill_(0.42)
    mjd.qpos[:] = 0.42

    # Step at t=0
    step(m, d)
    mujoco.mj_step(mjm, mjd)

    for w in range(nworld):
      np.testing.assert_allclose(d.sensordata.numpy()[w], mjd.sensordata, atol=1e-6)

  @parameterized.parameters(1, 2)
  def test_init_history_none_times_preserves_buffer(self, nworld):
    """Test init_ctrl_history and init_sensor_history with times=None preserve timestamps."""
    _, _, m, d = test_data.fixture(
      xml="""
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
        <sensor>
          <jointpos joint="slide" delay="0.02" nsample="3"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
    )
    orig_history = d.history.numpy().copy()

    # Actuator buffer
    ctrl_vals = wp.array2d(np.tile([10.0, 20.0, 30.0], (nworld, 1)), dtype=float)
    init_ctrl_history(m, d, ctrlid=0, times=None, values=ctrl_vals)

    # Sensor buffer
    sensor_vals = wp.array2d(np.tile([1.0, 2.0, 3.0], (nworld, 1)), dtype=float)
    init_sensor_history(m, d, sensorid=0, times=None, values=sensor_vals)

    act_adr = m.actuator_historyadr.numpy()[0, 0]
    act_n = m.actuator_history.numpy()[0, 0][0]
    act_t = act_adr + 2
    act_v = act_t + act_n

    sens_adr = m.sensor_historyadr.numpy()[0, 0]
    sens_n = m.sensor_history.numpy()[0, 0][0]
    sens_dim = m.sensor_dim.numpy()[0]
    sens_t = sens_adr + 2
    sens_v = sens_t + sens_n

    updated_history = d.history.numpy()
    for w in range(nworld):
      # Timestamps preserved
      np.testing.assert_allclose(updated_history[w, act_t : act_t + act_n], orig_history[w, act_t : act_t + act_n], atol=1e-6)
      np.testing.assert_allclose(
        updated_history[w, sens_t : sens_t + sens_n], orig_history[w, sens_t : sens_t + sens_n], atol=1e-6
      )
      # Values updated
      np.testing.assert_allclose(updated_history[w, act_v : act_v + act_n], [10.0, 20.0, 30.0], atol=1e-6)
      np.testing.assert_allclose(updated_history[w, sens_v : sens_v + sens_n * sens_dim], [1.0, 2.0, 3.0], atol=1e-6)

  def test_init_sensor_history_phase_none_preserves_user_slot(self):
    """Test init_sensor_history with phase=None preserves interval user slot (time_prev)."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <body>
            <joint name="slide" type="slide"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
        <sensor>
          <jointpos joint="slide" interval="0.05 0" nsample="3"/>
        </sensor>
      </mujoco>
      """,
      nworld=1,
    )
    sens_adr = m.sensor_historyadr.numpy()[0, 0]
    user_slot_orig = d.history.numpy()[0, sens_adr]  # -0.05
    self.assertAlmostEqual(user_slot_orig, -0.05)

    vals = wp.array2d(np.array([[1.0, 2.0, 3.0]]), dtype=float)
    init_sensor_history(m, d, sensorid=0, times=None, values=vals, phase=None)

    # User slot should remain -0.05, NOT be overwritten by 0.0
    self.assertAlmostEqual(d.history.numpy()[0, sens_adr], user_slot_orig)

  def test_init_history_shape_and_range_validation(self):
    """Test host-side shape, range, and type validation for init functions."""
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
        <motor joint="slide"/>
      </actuator>
      <sensor>
        <jointpos joint="slide" delay="0.02" nsample="3"/>
        <jointvel joint="slide"/>
      </sensor>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=2)

    good_times = wp.array([0.01, 0.02, 0.03], dtype=float)
    bad_times = wp.array([0.01, 0.02], dtype=float)
    good_ctrl_vals = wp.zeros((2, 3), dtype=float)
    bad_ctrl_vals = wp.zeros((2, 4), dtype=float)
    good_sensor_vals = wp.zeros((2, 3), dtype=float)
    bad_sensor_vals = wp.zeros((2, 4), dtype=float)

    # ctrlid out of range
    with self.assertRaises(ValueError):
      init_ctrl_history(m, d, ctrlid=-1, times=good_times, values=good_ctrl_vals)
    with self.assertRaises(ValueError):
      init_ctrl_history(m, d, ctrlid=2, times=good_times, values=good_ctrl_vals)

    # ctrlid has no history buffer allocated (nsample == 0)
    with self.assertRaises(ValueError):
      init_ctrl_history(m, d, ctrlid=1, times=good_times, values=good_ctrl_vals)

    # ctrl times shape mismatch
    with self.assertRaises(ValueError):
      init_ctrl_history(m, d, ctrlid=0, times=bad_times, values=good_ctrl_vals)

    # ctrl values shape mismatch
    with self.assertRaises(ValueError):
      init_ctrl_history(m, d, ctrlid=0, times=good_times, values=bad_ctrl_vals)

    # sensorid out of range
    with self.assertRaises(ValueError):
      init_sensor_history(m, d, sensorid=-1, times=good_times, values=good_sensor_vals)
    with self.assertRaises(ValueError):
      init_sensor_history(m, d, sensorid=2, times=good_times, values=good_sensor_vals)

    # sensorid has no history buffer allocated (nsample == 0)
    with self.assertRaises(ValueError):
      init_sensor_history(m, d, sensorid=1, times=good_times, values=good_sensor_vals)

    # sensor times shape mismatch
    with self.assertRaises(ValueError):
      init_sensor_history(m, d, sensorid=0, times=bad_times, values=good_sensor_vals)

    # sensor values shape mismatch
    with self.assertRaises(ValueError):
      init_sensor_history(m, d, sensorid=0, times=good_times, values=bad_sensor_vals)

    # sensor phase array shape mismatch
    bad_phase = wp.array([0.0], dtype=float)
    with self.assertRaises(ValueError):
      init_sensor_history(m, d, sensorid=0, times=good_times, values=good_sensor_vals, phase=bad_phase)

    # sensor scalar float phase succeeds
    init_sensor_history(m, d, sensorid=0, times=good_times, values=good_sensor_vals, phase=0.05)
    sens_adr = m.sensor_historyadr.numpy()[0, 0]
    np.testing.assert_allclose(d.history.numpy()[:, sens_adr], [0.05, 0.05])


class BatchedDelayTest(parameterized.TestCase):
  """Tests for batched delay and history fields across worlds."""

  @parameterized.parameters(1, 2)
  def test_batched_delay(self, nworld):
    """Test actuators and sensors with batched delays per world."""
    batch_sizes = {
      "actuator_delay": nworld,
      "actuator_history": nworld,
      "actuator_historyadr": nworld,
      "sensor_delay": nworld,
      "sensor_interval": nworld,
      "sensor_history": nworld,
      "sensor_historyadr": nworld,
    }
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <body>
            <joint name="slide" type="slide"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="slide" delay="0.01" nsample="4"/>
        </actuator>
        <sensor>
          <jointpos joint="slide" delay="0.01" interval="0.02 0" nsample="4"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
      batch_sizes=batch_sizes,
    )

    for field in batch_sizes:
      self.assertEqual(getattr(m, field).shape[0], nworld)

    if nworld == 1:
      wp.copy(m.actuator_delay, wp.array2d([[0.02]], dtype=float))
      wp.copy(m.sensor_delay, wp.array2d([[0.02]], dtype=float))
      d.ctrl.fill_(10.0)
      step(m, d)
      self.assertEqual(d.actuator_force.numpy()[0, 0], 0.0)
    else:
      delays = np.array([[0.0], [0.02]], dtype=np.float32)
      wp.copy(m.actuator_delay, wp.array2d(delays, dtype=float))
      wp.copy(m.sensor_delay, wp.array2d(delays, dtype=float))

      for t in range(5):
        ctrl_val = float(t + 1) * 10.0
        d.ctrl.fill_(ctrl_val)
        step(m, d)

      qpos = d.qpos.numpy()[:, 0]
      sdata = d.sensordata.numpy()[:, 0]
      self.assertGreater(qpos[0], qpos[1])
      self.assertGreater(sdata[0], sdata[1])

  @parameterized.parameters(1, 2)
  def test_batched_delay_reset(self, nworld):
    """Test reset_data parity and selective mask reset with per-world batched delay arrays."""
    batch_sizes = {
      "actuator_delay": nworld,
      "actuator_history": nworld,
      "actuator_historyadr": nworld,
      "sensor_delay": nworld,
      "sensor_interval": nworld,
      "sensor_history": nworld,
      "sensor_historyadr": nworld,
    }
    mjm, mjd, m, d = test_data.fixture(
      xml="""
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
        <sensor>
          <jointpos joint="slide" delay="0.02" nsample="3"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
      batch_sizes=batch_sizes,
    )

    d.ctrl.fill_(10.0)
    for _ in range(10):
      step(m, d)

    # Full reset
    reset_data(m, d)
    mujoco.mj_resetData(mjm, mjd)
    for w in range(nworld):
      np.testing.assert_allclose(d.history.numpy()[w], mjd.history, atol=1e-6)

    # Selective reset with reverse mask
    if nworld == 2:
      for _ in range(10):
        step(m, d)
      hist_dirty = d.history.numpy().copy()
      mask = wp.array([False, True], dtype=bool)
      reset_data(m, d, reset=mask)
      hist_after = d.history.numpy()
      # World 0 preserved dirty, World 1 reset
      np.testing.assert_allclose(hist_after[0], hist_dirty[0], atol=1e-6)
      np.testing.assert_allclose(hist_after[1], mjd.history, atol=1e-6)

  def test_reset_history_standalone(self):
    """Test reset_history public function directly."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
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
        <sensor>
          <jointpos joint="slide" delay="0.02" nsample="3"/>
        </sensor>
      </mujoco>
      """,
      nworld=1,
    )
    d.ctrl.fill_(10.0)
    for _ in range(5):
      step(m, d)
    self.assertFalse(np.allclose(d.history.numpy()[0], mjd.history))

    reset_history(m, d)
    mujoco.mj_resetData(mjm, mjd)
    np.testing.assert_allclose(d.history.numpy()[0], mjd.history, atol=1e-6)

  def test_reset_actuator_history_so3_quat(self):
    """Test SO(3) quaternion actuator history initializes to identity quat."""
    nworld = 2
    nactuator = 1
    nsample = 3
    dim = 4

    opt_timestep = wp.array([0.01], dtype=float)
    actuator_ctrlnum = wp.array([dim], dtype=int)
    actuator_gaintype = wp.array([GainType.SO3], dtype=int)
    actuator_ctrlspec = wp.array([CtrlChart.QUAT], dtype=int)
    actuator_history = wp.array([[(nsample, 0)]], dtype=wp.vec2i)
    actuator_historyadr = wp.array([[0]], dtype=int)
    reset_in = wp.empty(0, dtype=bool)

    nhistory = 2 + nsample + nsample * dim
    history = wp.empty((nworld, nhistory), dtype=float)
    history.fill_(wp.inf)

    wp.launch(
      _reset_actuator_history_kernel,
      dim=(nworld, nactuator),
      inputs=[
        opt_timestep,
        actuator_gaintype,
        actuator_ctrlnum,
        actuator_ctrlspec,
        actuator_history,
        actuator_historyadr,
        reset_in,
      ],
      outputs=[history],
    )

    hist_np = history.numpy()
    values_offset = 2 + nsample
    for w in range(nworld):
      for j in range(nsample):
        w_val = hist_np[w, values_offset + 4 * j]
        xyz = hist_np[w, values_offset + 4 * j + 1 : values_offset + 4 * j + 4]
        np.testing.assert_equal(w_val, 1.0)
        np.testing.assert_allclose(xyz, [0.0, 0.0, 0.0])


if __name__ == "__main__":
  absltest.main()
