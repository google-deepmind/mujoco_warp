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
"""Tests for render functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data

try:
  mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"))
  _HAS_RENDERER = True
except Exception:
  _HAS_RENDERER = False


def _assert_eq(a, b, name):
  tol = 5e-4
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class RenderTest(parameterized.TestCase):
  @parameterized.parameters(2, 512)
  def test_render(self, nworld: int):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)

    rc = mjw.create_render_context(
      mjm,
      nworld=nworld,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )

    mjw.render(m, d, rc)

    rgb = rc.rgb_data.numpy()
    depth = rc.depth_data.numpy()

    self.assertGreater(np.count_nonzero(rgb), 0)
    self.assertGreater(np.count_nonzero(depth), 0)

    self.assertNotEqual(np.unique(rgb).shape[0], 1)
    self.assertNotEqual(np.unique(depth).shape[0], 1)

  def test_render_humanoid(self):
    mjm, mjd, m, d = test_data.fixture("humanoid/humanoid.xml")
    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )
    mjw.render(m, d, rc)
    rgb = rc.rgb_data.numpy()

    self.assertNotEqual(np.unique(rgb).shape[0], 1)

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires CUDA.")
  def test_render_graph_capture(self):
    mjm, mjd, m, d = test_data.fixture("humanoid/humanoid.xml")
    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )

    mjw.render(m, d, rc)
    rgb_np = rc.rgb_data.numpy()

    with wp.ScopedCapture() as capture:
      mjw.render(m, d, rc)

    wp.capture_launch(capture.graph)

    _assert_eq(rgb_np, rc.rgb_data.numpy(), "rgb_data")

  @parameterized.parameters(2, 512)
  def test_render_segmentation(self, nworld: int):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)

    rc = mjw.create_render_context(
      mjm,
      nworld=nworld,
      cam_res=(32, 32),
      render_rgb=False,
      render_depth=False,
      render_seg=True,
    )

    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()

    # Should have at least one geom hit (>= 0) and multiple distinct geom IDs.
    self.assertTrue(np.any(seg >= 0), "Expected at least one geom hit (>= 0)")
    self.assertGreater(np.unique(seg).shape[0], 1)

  def test_get_segmentation(self):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=2)

    rc = mjw.create_render_context(
      mjm,
      nworld=2,
      cam_res=(32, 32),
      render_seg=True,
    )

    mjw.render(m, d, rc)

    seg_out = wp.zeros((2, 32, 32), dtype=wp.int32)
    mjw.get_segmentation(rc, 0, seg_out)

    seg_np = seg_out.numpy()
    self.assertEqual(seg_np.shape, (2, 32, 32))
    self.assertTrue(np.any(seg_np >= 0), "Expected at least one geom hit")
    self.assertGreater(np.unique(seg_np).shape[0], 1)

  def test_render_rgb_and_segmentation(self):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=2)

    rc = mjw.create_render_context(
      mjm,
      nworld=2,
      cam_res=(32, 32),
      render_rgb=True,
      render_seg=True,
    )

    mjw.render(m, d, rc)

    rgb = rc.rgb_data.numpy()
    seg = rc.seg_data.numpy()

    self.assertGreater(np.count_nonzero(rgb), 0)
    self.assertTrue(np.any(seg >= 0))

  def test_segmentation_from_camera_output(self):
    """Segmentation auto-detected from camera output attribute in XML."""
    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1"/>
        <geom type="sphere" size="0.2" pos="0 0 0.5" rgba="1 0 0 1"/>
        <camera name="cam" pos="0 -1 0.5" xyaxes="1 0 0 0 0 1"
                resolution="32 32" output="segmentation"/>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=1)

    rc = mjw.create_render_context(mjm, nworld=1, cam_res=(32, 32))
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()
    self.assertTrue(np.any(seg >= 0), "Expected geom hits from auto-detected seg")
    self.assertGreater(np.unique(seg).shape[0], 1)

  @absltest.skipIf(not _HAS_RENDERER, "MuJoCo rendering requires OpenGL")
  def test_depth_matches_mujoco(self):
    """Depth values should match native MuJoCo (planar depth, not Euclidean)."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=1)
    cam_w, cam_h = 32, 32

    # mjwarp depth
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_rgb=[False],
      render_depth=[True],
    )
    mjw.render(m, d, rc)
    warp_depth = rc.depth_data.numpy()[0]  # flat array for world 0

    # Native MuJoCo depth
    with mujoco.Renderer(mjm, height=cam_h, width=cam_w) as renderer:
      renderer.update_scene(mjd, camera=0)
      renderer.enable_depth_rendering()
      mj_depth = renderer.render().flatten()

    # Compare only pixels that hit geometry (non-zero in both)
    valid = (warp_depth > 0) & (mj_depth > 0)
    np.testing.assert_allclose(
      warp_depth[valid],
      mj_depth[valid],
      atol=1e-2,
      rtol=1e-2,
    )


if __name__ == "__main__":
  wp.init()
  absltest.main()
