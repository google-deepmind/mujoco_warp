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
"""Tests for render context functions."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp import test_data

from . import render_context

_CAMERA_TEST_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam1" pos="0 -3 2" xyaxes="1 0 0 0 0.6 0.8"/>
    <camera name="cam2" pos="0 3 2" xyaxes="-1 0 0 0 0.6 0.8"/>
    <camera name="cam3" pos="3 0 2" xyaxes="0 1 0 -0.6 0 0.8"/>
    <geom type="plane" size="5 5 0.1"/>
    <geom type="sphere" size="0.5" pos="0 0 1"/>
  </worldbody>
</mujoco>
"""


class RenderContextTest(parameterized.TestCase):
  """Tests for render context creation using primitives.xml."""

  @parameterized.parameters(1, 4)
  def test_bvh_creation(self, nworld):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)
    rc = render_context.create_render_context(mjm, m, d, cam_res=(64, 64))

    self.assertIsNotNone(rc)
    self.assertEqual(rc.ncam, mjm.ncam)

    self.assertEqual(rc.lower.shape[0], nworld * rc.bvh_ngeom)
    self.assertEqual(rc.upper.shape[0], nworld * rc.bvh_ngeom)
    self.assertEqual(rc.group.shape[0], nworld * rc.bvh_ngeom)
    self.assertEqual(rc.group_root.shape[0], nworld)

    self.assertIsNotNone(rc.bvh_id)
    self.assertNotEqual(rc.bvh_id, 0)

    group_np = rc.group.numpy()
    for i in range(nworld):
      start = i * rc.bvh_ngeom
      end = start + rc.bvh_ngeom
      np.testing.assert_array_equal(group_np[start:end], i)

  def test_output_buffers(self):
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 24
    rc = render_context.create_render_context(mjm, m, d, cam_res=(width, height), render_rgb=True, render_depth=True)

    expected_total = 3 * width * height

    self.assertEqual(rc.ncam, 3)
    self.assertEqual(rc.rgb_data.shape[0], d.nworld)
    self.assertEqual(rc.rgb_data.shape[1], expected_total)
    self.assertEqual(rc.depth_data.shape[0], d.nworld)
    self.assertEqual(rc.depth_data.shape[1], expected_total)

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()
    np.testing.assert_array_equal(rgb_adr, [0, width * height, 2 * width * height])
    np.testing.assert_array_equal(depth_adr, [0, width * height, 2 * width * height])

  def test_different_camera_resolutions(self):
    """Tests render context with different resolutions per camera."""
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    cam_res = [(64, 64), (32, 32), (16, 16)]
    rc = render_context.create_render_context(mjm, m, d, cam_res=cam_res, render_rgb=True, render_depth=True)

    self.assertEqual(rc.ncam, 3)

    cam_res_np = rc.cam_res.numpy()
    np.testing.assert_array_equal(cam_res_np[0], [64, 64])
    np.testing.assert_array_equal(cam_res_np[1], [32, 32])
    np.testing.assert_array_equal(cam_res_np[2], [16, 16])

    expected_total = 64 * 64 + 32 * 32 + 16 * 16
    self.assertEqual(rc.rgb_data.shape[1], expected_total)
    self.assertEqual(rc.depth_data.shape[1], expected_total)

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()
    np.testing.assert_array_equal(rgb_adr, [0, 64 * 64, 64 * 64 + 32 * 32])
    np.testing.assert_array_equal(depth_adr, [0, 64 * 64, 64 * 64 + 32 * 32])

  def test_cam_active_filtering(self):
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 32

    rc = render_context.create_render_context(mjm, m, d, cam_res=(width, height), cam_active=[True, False, True])

    self.assertEqual(rc.ncam, 2)

    expected_total = 2 * width * height
    self.assertEqual(rc.rgb_data.shape[1], expected_total)

  def test_rgb_only_and_depth_only(self):
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 32
    pixels = width * height

    rc = render_context.create_render_context(
      mjm,
      m,
      d,
      cam_res=(width, height),
      render_rgb=[True, False, True],
      render_depth=[False, True, True],
    )

    self.assertEqual(rc.rgb_data.shape[1], 2 * pixels)
    self.assertEqual(rc.depth_data.shape[1], 2 * pixels)

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()

    self.assertEqual(rgb_adr[0], 0)
    self.assertEqual(depth_adr[0], -1)

    self.assertEqual(rgb_adr[1], -1)
    self.assertEqual(depth_adr[1], 0)

    self.assertEqual(rgb_adr[2], pixels)
    self.assertEqual(depth_adr[2], pixels)

    render_rgb = rc.render_rgb.numpy()
    render_depth = rc.render_depth.numpy()
    np.testing.assert_array_equal(render_rgb, [True, False, True])
    np.testing.assert_array_equal(render_depth, [False, True, True])


if __name__ == "__main__":
  wp.init()
  absltest.main()
