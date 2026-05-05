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


def _unpack_rgb(packed):
  r = ((packed >> 16) & 0xFF).astype(np.uint8)
  g = ((packed >> 8) & 0xFF).astype(np.uint8)
  b = (packed & 0xFF).astype(np.uint8)
  return np.stack([r, g, b], axis=-1)


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

    geom_mask = seg[..., 1] == int(mjw.ObjType.GEOM)
    self.assertTrue(np.any(geom_mask), "Expected at least one geom hit")
    self.assertGreater(np.unique(seg[..., 0][geom_mask]).shape[0], 1)

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
    self.assertTrue(np.any(seg[..., 1] == int(mjw.ObjType.GEOM)))

  def test_disable_ambient_lighting(self):
    xml = """
    <mujoco>
      <visual>
        <headlight active="0" ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
      </visual>
      <worldbody>
        <camera name="cam" pos="0 -3 1" xyaxes="1 0 0 0 0.25 1" resolution="32 32" output="rgb"/>
        <geom type="sphere" pos="0 0 0.5" size="0.5" rgba="1 0 0 1"/>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)
    self.assertEqual(mjm.nlight, 0)
    self.assertEqual(m.nlight, 0)

    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      render_seg=True,
    )
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()[0]
    geom_mask = seg[:, 1] == int(mjw.ObjType.GEOM)
    self.assertTrue(np.any(geom_mask), "Expected at least one geom hit")

    rgb = _unpack_rgb(rc.rgb_data.numpy()[0])
    self.assertGreater(np.count_nonzero(rgb[geom_mask]), 0)

    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      render_seg=True,
      use_ambient_lighting=False,
    )
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()[0]
    geom_mask = seg[:, 1] == int(mjw.ObjType.GEOM)
    rgb = _unpack_rgb(rc.rgb_data.numpy()[0])
    np.testing.assert_array_equal(rgb[geom_mask], 0)

  @absltest.skipIf(not _HAS_RENDERER, "MuJoCo rendering requires OpenGL")
  def test_segmentation_matches_mujoco(self):
    """Segmentation should match native MuJoCo's `(object_id, object_type)` output."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=1)
    cam_w, cam_h = 32, 32

    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_seg=[True],
    )
    mjw.render(m, d, rc)

    warp_seg_np = rc.seg_data.numpy()[0].reshape(-1, 2)

    with mujoco.Renderer(mjm, height=cam_h, width=cam_w) as renderer:
      renderer.update_scene(mjd, camera=0)
      renderer.enable_segmentation_rendering()
      mj_seg = renderer.render().reshape(-1, 2)

    np.testing.assert_array_equal(warp_seg_np, mj_seg)

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
