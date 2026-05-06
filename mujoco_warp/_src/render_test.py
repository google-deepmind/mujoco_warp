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

  # Each scene places the camera at the origin fully enclosed by a primitive,
  # with a marker box at +Y (in front of the camera) well outside the
  # enclosure. A correctly backface-culling renderer must drop the far
  # exit-face hit on the enclosure and "see through" to the marker.
  _BACKFACE_CULL_SCENE = """
<mujoco>
  <visual>
    <map znear="0.001" />
  </visual>
  <worldbody>
    <camera name="cam" xyaxes="1 0 0 0 0 1" />
    <geom name="enclosure" type="{geom_type}" size="{size}" />
    <geom name="marker" type="box" size="0.5 0.5 0.5" pos="0 5 0" />
  </worldbody>
</mujoco>
"""

  _BACKFACE_CULL_PRIMITIVES = (
    ("sphere", "sphere", "1"),
    ("ellipsoid", "ellipsoid", "1 1 1"),
    ("capsule", "capsule", "0.5 0.5"),
    ("cylinder", "cylinder", "1 1"),
    ("box", "box", "1 1 1"),
  )

  @parameterized.named_parameters(*_BACKFACE_CULL_PRIMITIVES)
  def test_backface_cull_camera_inside_primitive(self, geom_type: str, size: str):
    """Camera inside a primitive must not render that primitive's back face.

    Mirrors MuJoCo's mesh ray rule (`dot(lvec, n) < 0`) for primitives: an
    exit-face hit (ray going outward through the surface) is dropped, so a ray
    originating inside a primitive sees through it. Without this cull, the
    enclosing geom would fill the frame with its back wall.
    """
    xml = self._BACKFACE_CULL_SCENE.format(geom_type=geom_type, size=size)
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=1)

    cam_w, cam_h = 16, 16
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_rgb=True,
      render_depth=True,
      render_seg=True,
    )
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()[0]
    depth = rc.depth_data.numpy()[0]

    enclosure_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "enclosure")
    marker_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "marker")

    geom_mask = seg[..., 1] == int(mjw.ObjType.GEOM)
    hit_ids = seg[..., 0][geom_mask]

    # The enclosing primitive must never appear in segmentation: every ray
    # originates inside it, so every "hit" against it is an exit-face hit and
    # must be culled.
    self.assertFalse(
      np.any(hit_ids == enclosure_id),
      f"enclosing {geom_type} should be backface-culled but appeared in segmentation",
    )

    # The marker box sits directly in front of the camera and is well outside
    # the enclosure, so at least one ray should reach it.
    self.assertTrue(
      np.any(hit_ids == marker_id),
      f"camera inside {geom_type} should see through to the marker box",
    )

    # Depth on enclosure-only pixels would equal the inner surface distance
    # (~size). Since we cull those, depth where we hit the marker must be
    # consistent with its world position (~5m away), not the small enclosure.
    marker_depth = depth.reshape(cam_h, cam_w)[seg[..., 0].reshape(cam_h, cam_w) == marker_id]
    if marker_depth.size > 0:
      self.assertGreater(float(np.min(marker_depth)), 1.0)

  @absltest.skipIf(not _HAS_RENDERER, "MuJoCo rendering requires OpenGL")
  @parameterized.named_parameters(*_BACKFACE_CULL_PRIMITIVES)
  def test_backface_cull_matches_mujoco(self, geom_type: str, size: str):
    """Backface-cull behavior for primitives must match native MuJoCo."""
    xml = self._BACKFACE_CULL_SCENE.format(geom_type=geom_type, size=size)
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=1)

    cam_w, cam_h = 16, 16
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_seg=[True],
    )
    mjw.render(m, d, rc)
    warp_seg = rc.seg_data.numpy()[0].reshape(-1, 2)

    with mujoco.Renderer(mjm, height=cam_h, width=cam_w) as renderer:
      renderer.update_scene(mjd, camera=0)
      renderer.enable_segmentation_rendering()
      mj_seg = renderer.render().reshape(-1, 2)

    np.testing.assert_array_equal(warp_seg, mj_seg)

  @parameterized.named_parameters(*_BACKFACE_CULL_PRIMITIVES)
  def test_backface_cull_disabled_keeps_enclosure(self, geom_type: str, size: str):
    """When `enable_backface_culling=False`, the enclosure must reappear.

    This is the inverse of test_backface_cull_camera_inside_primitive: it
    confirms the option actually toggles renderer behavior. With cull off,
    rays exit through the enclosing geom's far wall and that geom shows up
    in segmentation.
    """
    xml = self._BACKFACE_CULL_SCENE.format(geom_type=geom_type, size=size)
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=1)

    cam_w, cam_h = 16, 16
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_seg=True,
      enable_backface_culling=False,
    )
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()[0]
    enclosure_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "enclosure")
    geom_mask = seg[..., 1] == int(mjw.ObjType.GEOM)
    hit_ids = seg[..., 0][geom_mask]

    self.assertTrue(
      np.any(hit_ids == enclosure_id),
      f"with cull disabled, enclosing {geom_type} should appear in segmentation",
    )

  # Mesh-encloser scene: a unit-extent tetrahedron centered on the origin, and
  # a marker box at +Y. The mesh path's cull lives in ray_mesh_with_bvh and is
  # gated by the same `enable_backface_culling` option as the primitive paths.
  _BACKFACE_CULL_MESH_SCENE = """
<mujoco>
  <visual>
    <map znear="0.001" />
  </visual>
  <asset>
    <mesh name="tetra" vertex="1 1 1  1 -1 -1  -1 1 -1  -1 -1 1" />
  </asset>
  <worldbody>
    <camera name="cam" xyaxes="1 0 0 0 0 1" />
    <geom name="enclosure" type="mesh" mesh="tetra" />
    <geom name="marker" type="box" size="0.5 0.5 0.5" pos="0 5 0" />
  </worldbody>
</mujoco>
"""

  def test_backface_cull_mesh_camera_inside(self):
    """Camera inside a convex mesh sees through it when the cull is on."""
    mjm, mjd, m, d = test_data.fixture(xml=self._BACKFACE_CULL_MESH_SCENE, nworld=1)

    cam_w, cam_h = 16, 16
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_seg=True,
    )
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()[0]
    enclosure_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "enclosure")
    marker_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "marker")
    geom_mask = seg[..., 1] == int(mjw.ObjType.GEOM)
    hit_ids = seg[..., 0][geom_mask]

    self.assertFalse(
      np.any(hit_ids == enclosure_id),
      "enclosing mesh should be backface-culled but appeared in segmentation",
    )
    self.assertTrue(
      np.any(hit_ids == marker_id),
      "camera inside mesh should see through to the marker box",
    )

  def test_backface_cull_disabled_keeps_mesh_enclosure(self):
    """With `enable_backface_culling=False`, the mesh enclosure must reappear.

    Confirms the option also gates ray_mesh_with_bvh's local-space cull, not
    just the renderer-level trailing cull on primitives.
    """
    mjm, mjd, m, d = test_data.fixture(xml=self._BACKFACE_CULL_MESH_SCENE, nworld=1)

    cam_w, cam_h = 16, 16
    rc = mjw.create_render_context(
      mjm,
      nworld=1,
      cam_res=(cam_w, cam_h),
      render_seg=True,
      enable_backface_culling=False,
    )
    mjw.render(m, d, rc)

    seg = rc.seg_data.numpy()[0]
    enclosure_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "enclosure")
    geom_mask = seg[..., 1] == int(mjw.ObjType.GEOM)
    hit_ids = seg[..., 0][geom_mask]

    self.assertTrue(
      np.any(hit_ids == enclosure_id),
      "with cull disabled, enclosing mesh should appear in segmentation",
    )


if __name__ == "__main__":
  wp.init()
  absltest.main()
