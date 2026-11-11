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
"""Tests that the OptiX backend renders like the BVH backend."""

import importlib.util

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data


@absltest.skipIf(
  importlib.util.find_spec("warp_optix") is None or not wp.get_device().is_cuda,
  "requires warp_optix and a CUDA device",
)
class RenderOptixTest(parameterized.TestCase):
  @parameterized.product(nworld=(1, 2), use_fast_math=(False, True))
  def test_matches_bvh(self, nworld, use_fast_math):
    """OptiX and BVH renders, with camera and shadow rays, agree up to float rounding."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <asset>
          <hfield name="hf" nrow="4" ncol="4" size="1 1 0.2 0.1"
                  elevation="0 0.2 0.4 0 0.1 0.5 0.3 0.1 0.2 0.4 0.6 0.2 0 0.1 0.2 0"/>
          <mesh name="tet" vertex="0 0 0 0.3 0 0 0 0.3 0 0 0 0.3" face="0 2 1 0 1 3 1 2 3 0 3 2"/>
        </asset>
        <worldbody>
          <light pos="0 -1 3" dir="0 0.3 -1" castshadow="true"/>
          <camera pos="0 -2.2 1.2" xyaxes="1 0 0 0 0.5 1"/>
          <geom type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
          <geom type="hfield" hfield="hf" pos="0 0.5 0" rgba="0.4 0.7 0.4 1"/>
          <body pos="-0.4 0 0.3">
            <freejoint/>
            <geom size="0.15" rgba="1 0 0 1"/>
            <geom type="capsule" pos="0.3 0 0" size="0.08 0.1" rgba="0 1 0 1"/>
            <geom type="box" pos="0.6 0 0" size="0.1 0.1 0.1" rgba="0 0 1 1"/>
            <geom type="cylinder" pos="0 -0.3 0" size="0.08 0.1" rgba="1 1 0 1"/>
            <geom type="ellipsoid" pos="0.3 -0.3 0" size="0.12 0.08 0.1" rgba="0 1 1 1"/>
            <geom type="mesh" mesh="tet" pos="0.6 -0.3 0" rgba="1 0 1 1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      # Move world 1's body and enlarge its sphere, which OptiX instances scale to per world.
      qpos = d.qpos.numpy()
      qpos[1, 0] += 0.15
      d.qpos.assign(qpos)
      mjw.kinematics(m, d)
      sphere = 2
      geom_size = np.repeat(m.geom_size.numpy(), 2, axis=0)
      geom_size[1, sphere] *= 1.5
      m.geom_size = wp.array(geom_size, dtype=wp.vec3)

    outs = {}
    for backend in ("bvh", "optix"):
      rc = mjw.create_render_context(
        mjm,
        nworld=nworld,
        cam_res=(48, 48),
        render_rgb=True,
        render_depth=True,
        render_seg=True,
        use_shadows=True,
        use_fast_math=use_fast_math,
        backend=backend,
      )
      mjw.refit_bvh(m, d, rc)
      rc.rgb_data.fill_(wp.uint32(0xFFFFFFFF))
      rc.depth_data.fill_(wp.inf)
      rc.seg_data.fill_(wp.vec2i(-2, -2))
      mjw.render(m, d, rc)
      outs[backend] = (rc.rgb_data.numpy(), rc.depth_data.numpy(), rc.seg_data.numpy())

    (rgb, depth, seg), (rgb_optix, depth_optix, seg_optix) = outs["bvh"], outs["optix"]
    self.assertGreater(np.unique(seg[..., 0]).size, 4, "the scene must show several geoms")
    # OptiX offsets each world's rays into its grid cell and back, which rounds the ray origin by
    # ~1e-6 and grazing hits amplify that.
    np.testing.assert_allclose(depth, depth_optix, atol=1e-3)
    np.testing.assert_array_equal(seg, seg_optix)
    # Compare packed RGBA8 pixels channel by channel.
    np.testing.assert_array_less(np.abs(rgb.view(np.uint8).astype(int) - rgb_optix.view(np.uint8)), 2)
    if nworld == 2:
      self.assertFalse(np.allclose(depth[0], depth[1]))

  @parameterized.product(nworld=(1, 2), groups=((), (0,), (1,), (0, 1)), active=(False, True))
  def test_empty_and_plane_scenes(self, nworld, groups, active):
    """Empty selections and plane-only scenes agree with BVH, including after refits."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <camera pos="0 0 3"/>
          <geom type="plane" group="0" size="2 2 0.1" rgba="1 0 0 1"/>
          <geom type="plane" group="0" size="2 2 0.1" rgba="0 1 0 1"/>
          <geom type="sphere" group="1" pos="0 0 0.3" size="0.2"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      geom_xpos = d.geom_xpos.numpy()
      geom_xpos[1, :, 2] += 0.1
      d.geom_xpos.assign(geom_xpos)
    outs = {}
    for backend in ("bvh", "optix") if groups else ("optix",):
      rc = mjw.create_render_context(
        mjm,
        nworld=nworld,
        cam_res=(8, 8),
        cam_active=[active],
        enabled_geom_groups=list(groups),
        render_rgb=True,
        render_depth=True,
        render_seg=True,
        backend=backend,
      )
      for _ in range(2):
        mjw.refit_bvh(m, d, rc)
        rc.rgb_data.fill_(wp.uint32(0xFFFFFFFF))
        rc.depth_data.fill_(wp.inf)
        rc.seg_data.fill_(wp.vec2i(-2, -2))
        mjw.render(m, d, rc)
      outs[backend] = (rc.rgb_data.numpy(), rc.depth_data.numpy(), rc.seg_data.numpy())
      if backend == "optix" and 0 in groups:
        # Null traversables keep the unbounded plane out of the IAS even with primitives present.
        np.testing.assert_array_equal(rc.optix.instances.numpy()["traversable"][:, :2], 0)
    if groups:
      for expected, actual in zip(outs["bvh"], outs["optix"]):
        np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)
    else:
      np.testing.assert_array_equal(outs["optix"][0], rc.background_color)
      np.testing.assert_array_equal(outs["optix"][1], 0.0)
      np.testing.assert_array_equal(outs["optix"][2], -1)
    if active and groups:
      self.assertTrue(np.isfinite(outs["optix"][1]).all())
      if nworld == 2:
        self.assertFalse(np.allclose(outs["optix"][1][0], outs["optix"][1][1]))

  @parameterized.parameters(1, 2)
  def test_camera_outside_world_cell(self, nworld):
    """Rays enter their own cell without hitting neighboring worlds on the way in."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <camera pos="6 0 0" xyaxes="0 1 0 0 0 1"/>
          <geom type="sphere" size="0.2"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      geom_xpos = d.geom_xpos.numpy()
      geom_xpos[1, 0, 0] = 0.1
      d.geom_xpos.assign(geom_xpos)
    outputs = {}
    for backend in ("bvh", "optix"):
      rc = mjw.create_render_context(mjm, nworld=nworld, cam_res=(32, 32), render_depth=True, render_seg=True, backend=backend)
      mjw.refit_bvh(m, d, rc)
      rc.depth_data.fill_(wp.inf)
      rc.seg_data.fill_(wp.vec2i(-2, -2))
      mjw.render(m, d, rc)
      outputs[backend] = rc.depth_data.numpy(), rc.seg_data.numpy()
    for expected, actual in zip(outputs["bvh"], outputs["optix"]):
      np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)
    self.assertTrue(np.any(outputs["optix"][1][..., 0] == 0))
    if nworld == 2:
      self.assertFalse(np.allclose(outputs["optix"][0][0], outputs["optix"][0][1]))


if __name__ == "__main__":
  absltest.main()
