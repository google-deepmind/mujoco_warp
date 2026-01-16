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
"""Tests for render context creation."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data

_CAMERA_TEST_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam1" pos="0 -3 2" xyaxes="1 0 0 0 0.6 0.8" resolution="64 64" output="rgb"/>
    <camera name="cam2" pos="0 3 2" xyaxes="-1 0 0 0 0.6 0.8" resolution="32 32" output="depth"/>
    <camera name="cam3" pos="3 0 2" xyaxes="0 1 0 -0.6 0 0.8" resolution="16 16" output="rgb depth"/>
    <geom type="plane" size="5 5 0.1"/>
    <geom type="sphere" size="0.5" pos="0 0 1"/>
  </worldbody>
</mujoco>
"""


class RenderContextTest(parameterized.TestCase):
  """Tests for render context creation."""

  @parameterized.parameters(1, 4)
  def test_bvh_creation(self, nworld):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)
    rc = mjw.create_render_context(mjm, m, d, cam_res=(64, 64))

    self.assertIsNotNone(rc)
    self.assertEqual(rc.ncam, mjm.ncam)

    self.assertEqual(rc.lower.shape, (nworld * rc.bvh_ngeom,), "lower")
    self.assertEqual(rc.upper.shape, (nworld * rc.bvh_ngeom,), "upper")
    self.assertEqual(rc.group.shape, (nworld * rc.bvh_ngeom,), "group")
    self.assertEqual(rc.group_root.shape, (nworld,), "group_root")

    self.assertIsNotNone(rc.bvh_id)
    self.assertNotEqual(rc.bvh_id, 0, "bvh_id")

    group_np = rc.group.numpy()
    np.testing.assert_array_equal(group_np, np.repeat(np.arange(nworld), rc.bvh_ngeom), err_msg="render context group values")

  def test_output_buffers(self):
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 24
    rc = mjw.create_render_context(mjm, m, d, cam_res=(width, height), render_rgb=True, render_depth=True)

    expected_total = 3 * width * height

    self.assertEqual(rc.ncam, 3, "ncam")
    self.assertEqual(rc.rgb_data.shape, (d.nworld, expected_total), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (d.nworld, expected_total), "depth_data")

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()
    np.testing.assert_array_equal(rgb_adr, [0, width * height, 2 * width * height], err_msg="rgb_adr")
    np.testing.assert_array_equal(depth_adr, [0, width * height, 2 * width * height], err_msg="depth_adr")

  def test_heterogeneous_camera(self):
    """Tests render context with different resolutions and output."""
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    cam_res = [(64, 64), (32, 32), (16, 16)]
    rc = mjw.create_render_context(mjm, m, d, cam_res=cam_res, render_rgb=True, render_depth=True)

    self.assertEqual(rc.ncam, 3, "ncam")
    np.testing.assert_array_equal(rc.cam_res.numpy(), cam_res, err_msg="cam_res")

    expected_total = 64 * 64 + 32 * 32 + 16 * 16
    self.assertEqual(rc.rgb_data.shape, (d.nworld, expected_total), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (d.nworld, expected_total), "depth_data")

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()
    np.testing.assert_array_equal(rgb_adr, [0, 64 * 64, 64 * 64 + 32 * 32], err_msg="rgb_adr")
    np.testing.assert_array_equal(depth_adr, [0, 64 * 64, 64 * 64 + 32 * 32], err_msg="depth_adr")

    # Test that results are same when reading from mjmodel fields loaded through xml
    rc_xml = mjw.create_render_context(mjm, m, d, render_rgb=True, render_depth=True)
    self.assertEqual(rc.rgb_data.shape, rc_xml.rgb_data.shape, "rgb_data")
    self.assertEqual(rc.depth_data.shape, rc_xml.depth_data.shape, "depth_data")
    np.testing.assert_array_equal(rc.rgb_adr.numpy(), rc_xml.rgb_adr.numpy(), err_msg="rgb_adr")
    np.testing.assert_array_equal(rc.depth_adr.numpy(), rc_xml.depth_adr.numpy(), err_msg="depth_adr")

  def test_cam_active_filtering(self):
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 32

    rc = mjw.create_render_context(mjm, m, d, cam_res=(width, height), cam_active=[True, False, True])

    self.assertEqual(rc.ncam, 2, "ncam")

    expected_total = 2 * width * height
    self.assertEqual(rc.rgb_data.shape, (d.nworld, expected_total), "rgb_data")

  def test_rgb_only_and_depth_only(self):
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 32
    pixels = width * height

    rc = mjw.create_render_context(
      mjm,
      m,
      d,
      cam_res=(width, height),
      render_rgb=[True, False, True],
      render_depth=[False, True, True],
    )

    self.assertEqual(rc.rgb_data.shape, (d.nworld, 2 * pixels), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (d.nworld, 2 * pixels), "depth_data")
    np.testing.assert_array_equal(rc.rgb_adr.numpy(), [0, -1, pixels], "rgb_adr")
    np.testing.assert_array_equal(rc.depth_adr.numpy(), [-1, 0, pixels], "depth_adr")
    np.testing.assert_array_equal(rc.render_rgb.numpy(), [True, False, True], "render_rgb")
    np.testing.assert_array_equal(rc.render_depth.numpy(), [False, True, True], "render_depth")

    # Test that results are same when reading from mjmodel fields loaded through xml
    rc_xml = mjw.create_render_context(mjm, m, d, cam_res=(width, height))
    self.assertEqual(rc.rgb_data.shape, rc_xml.rgb_data.shape, "rgb_data")
    self.assertEqual(rc.depth_data.shape, rc_xml.depth_data.shape, "depth_data")
    np.testing.assert_array_equal(rc.rgb_adr.numpy(), rc_xml.rgb_adr.numpy(), "rgb_adr")
    np.testing.assert_array_equal(rc.depth_adr.numpy(), rc_xml.depth_adr.numpy(), "depth_adr")
    np.testing.assert_array_equal(rc.render_rgb.numpy(), rc_xml.render_rgb.numpy(), "render_rgb")
    np.testing.assert_array_equal(rc.render_depth.numpy(), rc_xml.render_depth.numpy(), "render_depth")


if __name__ == "__main__":
  wp.init()
  absltest.main()
