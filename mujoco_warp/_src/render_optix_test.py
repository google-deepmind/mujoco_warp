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


@absltest.skipIf(importlib.util.find_spec("warp_optix") is None, "requires warp_optix")
class RenderOptixTest(parameterized.TestCase):
  @parameterized.parameters(1, 2)
  def test_matches_bvh(self, nworld):
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
      geom_aabb = np.repeat(m.geom_aabb.numpy(), 2, axis=0)
      geom_aabb[1, sphere, 1] *= 1.5
      m.geom_aabb = wp.array(geom_aabb, dtype=wp.vec3)

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


if __name__ == "__main__":
  absltest.main()
