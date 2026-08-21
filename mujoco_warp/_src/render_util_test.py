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
"""Tests for render utility functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import render_util
from mujoco_warp._src import types


def _assert_eq(a, b, name):
  tol = 5e-4
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


# TODO: Add more cameras for testing projection and intrinsics
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


class RenderUtilTest(parameterized.TestCase):
  def test_create_warp_texture(self):
    """Tests that create_warp_texture creates a valid texture."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    texture = render_util.create_warp_texture(mjm, 0)

    self.assertNotEqual(texture.id, wp.uint64(0), "texture id")
    self.assertFalse(np.array_equal(np.array(texture), np.array([0.0, 0.0, 0.0])), "texture")

  def test_compute_ray(self):
    """Tests that compute_ray computes correct rays for both projections."""
    img_w, img_h = 2, 2
    px, py = 1, 1
    fovy = 90.0
    znear = 1.0
    sensorsize = wp.vec2(0.0, 0.0)
    intrinsic = wp.vec4(0.0, 0.0, 0.0, 0.0)

    persp_ray = render_util.compute_ray(
      int(types.ProjectionType.PERSPECTIVE),
      fovy,
      sensorsize,
      intrinsic,
      img_w,
      img_h,
      px,
      py,
      znear,
    )
    ortho_ray = render_util.compute_ray(
      int(types.ProjectionType.ORTHOGRAPHIC),
      fovy,
      sensorsize,
      intrinsic,
      img_w,
      img_h,
      px,
      py,
      znear,
    )

    mag = np.sqrt(0.5**2 + 0.5**2 + 1.0**2)
    expected_persp = np.array([0.5 / mag, -0.5 / mag, -1.0 / mag])
    np.testing.assert_allclose(np.array(persp_ray), expected_persp, atol=1e-5)

    expected_ortho = np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(np.array(ortho_ray), expected_ortho, atol=1e-5)

    self.assertFalse(
      np.allclose(np.array(persp_ray), np.array(ortho_ray)),
      "perspective != orthographic raydir",
    )

  def test_get_segmentation(self):
    """Tests that get_segmentation extracts MuJoCo-style typed IDs."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=2)

    rc = mjw.create_render_context(
      mjm,
      nworld=2,
      cam_res=(32, 32),
      render_seg=True,
    )

    mjw.render(m, d, rc)

    seg_out = wp.zeros((2, 32, 32), dtype=wp.vec2i)
    mjw.get_segmentation(rc, 0, seg_out)

    seg_np = seg_out.numpy()
    self.assertEqual(seg_np.shape, (2, 32, 32, 2))
    self.assertTrue(np.any(seg_np[..., 1] == int(types.ObjType.GEOM)))

    geom_mask = seg_np[..., 1] == int(types.ObjType.GEOM)
    self.assertTrue(np.any(geom_mask), "Expected at least one geom hit")
    self.assertGreater(np.unique(seg_np[..., 0][geom_mask]).shape[0], 1)

    background_mask = seg_np[..., 1] == -1
    np.testing.assert_array_equal(seg_np[..., 0][background_mask], -1)

  def test_get_segmentation_preserves_flex_ids(self):
    """Tests that flex hits keep their real flex ids and type tags."""
    mjm, mjd, m, d = test_data.fixture("flex/multiflex.xml", nworld=1)

    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(64, 64),
      render_seg=True,
    )

    mjw.render(m, d, rc)

    seg_out = wp.zeros((1, 64, 64), dtype=wp.vec2i)
    mjw.get_segmentation(rc, 0, seg_out)
    seg_np = seg_out.numpy()[0]

    flex_mask = seg_np[..., 1] == int(types.ObjType.FLEX)
    self.assertTrue(np.any(flex_mask), "Expected at least one flex hit")
    self.assertTrue(np.all(seg_np[..., 0][flex_mask] >= 0))
    self.assertGreater(np.unique(seg_np[..., 0][flex_mask]).shape[0], 1)

  @parameterized.parameters(1, 4)
  def test_bvh_creation(self, nworld):
    """Test that the BVH is created correctly for single world and multiple worlds."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)
    rc = mjw.create_render_context(mjm, nworld=nworld, cam_res=(64, 64), use_textures=False)

    self.assertIsNotNone(rc)
    self.assertEqual(rc.nrender, mjm.ncam)

    self.assertEqual(rc.lower.shape, (nworld * rc.bvh_ngeom,), "lower")
    self.assertEqual(rc.upper.shape, (nworld * rc.bvh_ngeom,), "upper")
    self.assertEqual(rc.group.shape, (nworld * rc.bvh_ngeom,), "group")
    self.assertEqual(rc.group_root.shape, (nworld,), "group_root")

    self.assertIsNotNone(rc.bvh_id)
    self.assertNotEqual(rc.bvh_id, 0, "bvh_id")

    group_np = rc.group.numpy()
    _assert_eq(group_np, np.repeat(np.arange(nworld), rc.bvh_ngeom), "render context group values")

  def test_output_buffers(self):
    """Test that the output rgb and depth buffers have correct shapes and addresses."""
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 24
    rc = mjw.create_render_context(mjm, cam_res=(width, height), render_rgb=True, render_depth=True)

    expected_total = 3 * width * height

    self.assertEqual(rc.nrender, 3, "nrender")
    self.assertEqual(rc.rgb_data.shape, (1, expected_total), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (1, expected_total), "depth_data")

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()
    _assert_eq(rgb_adr, [0, width * height, 2 * width * height], "rgb_adr")
    _assert_eq(depth_adr, [0, width * height, 2 * width * height], "depth_adr")

  def test_heterogeneous_camera(self):
    """Tests render context with different resolutions and output."""
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    cam_res = [(64, 64), (32, 32), (16, 16)]
    rc = mjw.create_render_context(mjm, cam_res=cam_res, render_rgb=True, render_depth=True)

    self.assertEqual(rc.nrender, 3, "nrender")
    _assert_eq(rc.cam_res.numpy(), cam_res, "cam_res")

    expected_total = 64 * 64 + 32 * 32 + 16 * 16
    self.assertEqual(rc.rgb_data.shape, (1, expected_total), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (1, expected_total), "depth_data")

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()
    _assert_eq(rgb_adr, [0, 64 * 64, 64 * 64 + 32 * 32], "rgb_adr")
    _assert_eq(depth_adr, [0, 64 * 64, 64 * 64 + 32 * 32], "depth_adr")

    # Test that results are same when reading from mjmodel fields loaded through xml
    rc_xml = mjw.create_render_context(mjm, render_rgb=True, render_depth=True)
    self.assertEqual(rc.rgb_data.shape, rc_xml.rgb_data.shape, "rgb_data")
    self.assertEqual(rc.depth_data.shape, rc_xml.depth_data.shape, "depth_data")
    _assert_eq(rc.rgb_adr.numpy(), rc_xml.rgb_adr.numpy(), "rgb_adr")
    _assert_eq(rc.depth_adr.numpy(), rc_xml.depth_adr.numpy(), "depth_adr")

  def test_cam_active_filtering(self):
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 32

    rc = mjw.create_render_context(mjm, cam_res=(width, height), cam_active=[True, False, True])

    self.assertEqual(rc.nrender, 2, "nrender")

    expected_total = 2 * width * height
    self.assertEqual(rc.rgb_data.shape, (1, expected_total), "rgb_data")

  def test_rgb_only_and_depth_only(self):
    """Test that disabling rgb or depth correctly reduces the shape and invalidates the address."""
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 32
    pixels = width * height

    rc = mjw.create_render_context(
      mjm,
      cam_res=(width, height),
      render_rgb=[True, False, True],
      render_depth=[False, True, True],
    )

    self.assertEqual(rc.rgb_data.shape, (1, 2 * pixels), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (1, 2 * pixels), "depth_data")
    _assert_eq(rc.rgb_adr.numpy(), [0, -1, pixels], "rgb_adr")
    _assert_eq(rc.depth_adr.numpy(), [-1, 0, pixels], "depth_adr")
    _assert_eq(rc.render_rgb.numpy(), [True, False, True], "render_rgb")
    _assert_eq(rc.render_depth.numpy(), [False, True, True], "render_depth")

    # Test that results are same when reading from mjmodel fields loaded through xml
    rc_xml = mjw.create_render_context(mjm, cam_res=(width, height))
    self.assertEqual(rc.rgb_data.shape, rc_xml.rgb_data.shape, "rgb_data")
    self.assertEqual(rc.depth_data.shape, rc_xml.depth_data.shape, "depth_data")
    _assert_eq(rc.rgb_adr.numpy(), rc_xml.rgb_adr.numpy(), "rgb_adr")
    _assert_eq(rc.depth_adr.numpy(), rc_xml.depth_adr.numpy(), "depth_adr")
    _assert_eq(rc.render_rgb.numpy(), rc_xml.render_rgb.numpy(), "render_rgb")
    _assert_eq(rc.render_depth.numpy(), rc_xml.render_depth.numpy(), "render_depth")

  def test_segmentation_from_camera_output(self):
    """Segmentation auto-detected from camera output attribute in XML."""
    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="10 10 0.1"/>
        <geom type="sphere" size="0.2" pos="0 0 0.5" rgba="1 0 0 1"/>
        <flexcomp type="grid" count="2 2 1" spacing="0.1 0.1 0.1" pos="-0.1 -0.1 0.7"
                  radius="0.02" name="cloth" dim="2" mass="0.1">
          <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001"
                   selfcollide="none" conaffinity="1" contype="1"/>
          <edge damping="0.01"/>
        </flexcomp>
        <camera name="cam" pos="0 -1 0.5" xyaxes="1 0 0 0 0 1"
                resolution="32 32" output="segmentation"/>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    self.assertEqual(mjm.nflex, 1, "nflex")
    rc = mjw.create_render_context(mjm, nworld=1, cam_res=(32, 32))
    pixels = 32 * 32

    self.assertEqual(rc.seg_data.shape, (1, pixels), "seg_data")
    _assert_eq(rc.seg_adr.numpy(), [0], "seg_adr")
    _assert_eq(rc.render_seg.numpy(), [True], "render_seg")

  def test_render_context_with_textures(self):
    mjm, mjd, m, d = test_data.fixture("mug/mug.xml")
    rc = mjw.create_render_context(mjm, render_rgb=True, render_depth=True, use_textures=True)
    self.assertTrue(rc.use_textures, "use_textures")
    self.assertEqual(rc.textures.shape, (mjm.ntex,), "textures")

  def test_render_context_lighting_flags(self):
    mjm, _, _, _ = test_data.fixture(
      xml="""
      <mujoco>
        <visual>
          <headlight active="0" ambient="0.2 0.3 0.4" diffuse="0.5 0.6 0.7" specular="0.8 0.9 1.0"/>
        </visual>
        <worldbody>
          <light pos="0 0 3" dir="0 0 -1" attenuation="1 0.1 0.05" cutoff="25"/>
          <geom type="sphere" size="0.3"/>
        </worldbody>
      </mujoco>
      """
    )
    rc = mjw.create_render_context(
      mjm,
      cam_res=(32, 32),
      render_rgb=True,
      use_shadows=False,
      use_ambient_lighting=False,
      enable_specular=False,
      enable_emission=False,
      enable_per_light_ambient=False,
    )
    self.assertFalse(rc.use_shadows)
    self.assertFalse(rc.use_ambient_lighting)
    self.assertFalse(rc.enable_specular)
    self.assertFalse(rc.enable_emission)
    self.assertFalse(rc.enable_per_light_ambient)
    self.assertFalse(rc.headlight_active)
    self.assertFalse(rc.light_attenuation_is_default)
    self.assertTrue(rc.has_spot_lights)
    _assert_eq(np.asarray(rc.headlight_ambient), mjm.vis.headlight.ambient, "headlight_ambient")
    _assert_eq(np.asarray(rc.headlight_diffuse), mjm.vis.headlight.diffuse, "headlight_diffuse")
    _assert_eq(np.asarray(rc.headlight_specular), mjm.vis.headlight.specular, "headlight_specular")


if __name__ == "__main__":
  wp.init()
  absltest.main()
