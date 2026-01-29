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

"""Tests for warning system."""

import warnings

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp._src import types


class WarningTest(absltest.TestCase):
  def test_warning_arrays_initialized(self):
    """Tests that warning arrays are properly initialized."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <geom type="sphere" size="0.1"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)

    d = mjw.put_data(mjm, mjd)

    # Check shapes
    self.assertEqual(d.warning.shape, (types.NUM_WARNINGS,))
    self.assertEqual(d.warning_info.shape, (types.NUM_WARNINGS, 2))

    # Check initial values are zero
    np.testing.assert_array_equal(d.warning.numpy(), np.zeros(types.NUM_WARNINGS, dtype=np.int32))
    np.testing.assert_array_equal(d.warning_info.numpy(), np.zeros((types.NUM_WARNINGS, 2), dtype=np.int32))

  def test_check_warnings_no_warnings(self):
    """Tests check_warnings returns empty list when no warnings."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <geom type="sphere" size="0.1"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)

    # Run a step - should not trigger any warnings
    mjw.step(m, d)
    wp.synchronize()

    # Check warnings
    result = mjw.get_warnings(d)
    self.assertEqual(result, [])

  def test_nefc_overflow_warning(self):
    """Tests that nefc overflow sets warning flag correctly."""
    # Sphere close to ground with large timestep - contacts quickly
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <geom type="plane" size="5 5 0.1"/>
          <body pos="0 0 0.12">
            <geom type="sphere" size="0.1"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    # No initial contacts
    self.assertEqual(mjd.ncon, 0, "Test setup: should have no initial contacts")

    m = mjw.put_model(mjm)
    m.opt.warning_printf = False  # disable printf in tests
    # Set njmax very low to trigger overflow when sphere hits ground
    d = mjw.put_data(mjm, mjd, njmax=1)

    # Run steps until sphere falls and creates contact (~10 steps at 0.01 timestep)
    for _ in range(20):
      mjw.step(m, d)
    wp.synchronize()

    # Check warning flag is set
    warning_flags = d.warning.numpy()
    self.assertEqual(warning_flags[types.WarningType.NEFC_OVERFLOW], 1)

    # Check warning info contains suggested value (should be 4 for a single contact with friction)
    warning_info = d.warning_info.numpy()
    self.assertGreater(warning_info[types.WarningType.NEFC_OVERFLOW, 0], 1)

    # Check get_warnings returns the message
    result = mjw.get_warnings(d)
    self.assertEqual(len(result), 1)
    self.assertIn("nefc overflow", result[0])

  def test_check_warnings_clears_flags(self):
    """Tests that check_warnings clears flags by default."""
    # Sphere close to ground with large timestep
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <geom type="plane" size="5 5 0.1"/>
          <body pos="0 0 0.12">
            <geom type="sphere" size="0.1"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    m = mjw.put_model(mjm)
    m.opt.warning_printf = False  # disable printf in tests
    d = mjw.put_data(mjm, mjd, njmax=1)

    # Run until contact
    for _ in range(20):
      mjw.step(m, d)
    wp.synchronize()

    # First check_warnings should return warnings and clear
    with warnings.catch_warnings(record=True):
      warnings.simplefilter("always")
      result1 = mjw.check_warnings(d, clear=True)

    # Second call should return empty (flags cleared)
    result2 = mjw.get_warnings(d)

    self.assertGreater(len(result1), 0)
    self.assertEqual(len(result2), 0)

  def test_clear_warnings(self):
    """Tests clear_warnings utility."""
    # Sphere close to ground with large timestep
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <geom type="plane" size="5 5 0.1"/>
          <body pos="0 0 0.12">
            <geom type="sphere" size="0.1"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    m = mjw.put_model(mjm)
    m.opt.warning_printf = False  # disable printf in tests
    d = mjw.put_data(mjm, mjd, njmax=1)

    # Run until contact
    for _ in range(20):
      mjw.step(m, d)
    wp.synchronize()

    # Verify warning is set
    self.assertGreater(len(mjw.get_warnings(d)), 0)

    # Clear warnings
    mjw.clear_warnings(d)

    # Verify cleared
    self.assertEqual(len(mjw.get_warnings(d)), 0)
    np.testing.assert_array_equal(d.warning.numpy(), np.zeros(types.NUM_WARNINGS, dtype=np.int32))

  def test_multi_step_graph_captures_mid_graph_warning(self):
    """Tests that a multi-step graph captures warnings even if they occur mid-graph.

    Captures 20 steps in one graph. The sphere starts close to ground and hits it
    around step 5-10. The warning from that step should be captured and readable
    after the graph completes.
    """
    # Sphere close to ground - will contact around step 5-10
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <geom type="plane" size="5 5 0.1"/>
          <body pos="0 0 0.12">
            <geom type="sphere" size="0.1"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    # No initial contact
    self.assertEqual(mjd.ncon, 0, "Test setup: should have no initial contacts")

    m = mjw.put_model(mjm)
    m.opt.warning_printf = False  # disable printf in tests
    d = mjw.put_data(mjm, mjd, njmax=1)

    # Clear warnings
    mjw.clear_warnings(d)

    # Capture 20 steps as a single graph - warning should occur around step 5-10
    nsteps = 20
    with wp.ScopedCapture() as capture:
      for _ in range(nsteps):
        mjw.step(m, d)
    graph = capture.graph

    # Run graph once - warning happens mid-graph
    wp.capture_launch(graph)
    wp.synchronize()

    # Check that warning was captured from the step where contact happened
    warning_flags = d.warning.numpy()
    self.assertEqual(
      warning_flags[types.WarningType.NEFC_OVERFLOW], 1, f"Expected nefc overflow warning, got flags: {warning_flags}"
    )

    # Verify warning info shows the correct suggested value
    warning_info = d.warning_info.numpy()
    self.assertEqual(
      warning_info[types.WarningType.NEFC_OVERFLOW, 0],
      4,
      f"Expected nefc=4 in warning info, got: {warning_info[types.WarningType.NEFC_OVERFLOW]}",
    )

    # Get warnings - should include the overflow message
    result = mjw.get_warnings(d)
    self.assertEqual(len(result), 1)
    self.assertIn("nefc overflow", result[0])
    self.assertIn("4", result[0])

  def test_single_step_graph_warns_only_when_event_occurs(self):
    """Tests that a single-step graph only reports warning on the step it happens.

    Captures 1 step as a graph. Runs it multiple times. Verifies that:
    1. Before contact: no warnings generated
    2. After contact: warnings generated
    3. After clearing and running more: warnings still generated (overflow persists)
    """
    # Sphere close to ground - will contact around launch 5-10
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option timestep="0.01"/>
        <worldbody>
          <geom type="plane" size="5 5 0.1"/>
          <body pos="0 0 0.12">
            <geom type="sphere" size="0.1"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    m = mjw.put_model(mjm)
    m.opt.warning_printf = False  # disable printf in tests
    d = mjw.put_data(mjm, mjd, njmax=1)

    # Clear warnings
    mjw.clear_warnings(d)

    # Capture single step
    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    graph = capture.graph

    # Run a few launches before contact happens (sphere is falling)
    for _ in range(3):
      wp.capture_launch(graph)
    wp.synchronize()

    # Check: no warnings yet (sphere hasn't hit ground)
    warning_flags_before = d.warning.numpy().copy()
    self.assertEqual(warning_flags_before[types.WarningType.NEFC_OVERFLOW], 0, "Should have no warning before contact")

    # Clear and run more launches until contact happens
    mjw.clear_warnings(d)
    for _ in range(15):
      wp.capture_launch(graph)
    wp.synchronize()

    # Check: warning should now be set (contact occurred)
    warning_flags_after = d.warning.numpy()
    self.assertEqual(warning_flags_after[types.WarningType.NEFC_OVERFLOW], 1, "Should have warning after contact")

    # Clear and run more - warning should still be generated (overflow persists each step)
    mjw.clear_warnings(d)
    for _ in range(5):
      wp.capture_launch(graph)
    wp.synchronize()

    warning_flags_final = d.warning.numpy()
    self.assertEqual(
      warning_flags_final[types.WarningType.NEFC_OVERFLOW], 1, "Warning should still be generated (overflow persists)"
    )

    # Check warning message
    result = mjw.get_warnings(d)
    self.assertEqual(len(result), 1)
    self.assertIn("nefc overflow", result[0])

  def test_warning_printf_option(self):
    """Tests that warning_printf option is available and defaults to True."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <geom type="sphere" size="0.1"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
    """)

    m = mjw.put_model(mjm)

    # Default should be True (printf enabled)
    self.assertTrue(m.opt.warning_printf)


if __name__ == "__main__":
  wp.init()
  absltest.main()
