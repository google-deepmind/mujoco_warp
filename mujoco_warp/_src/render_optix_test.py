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

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import render_optix


def _render(fixture: str, backend: str, nworld: int, steps: int, **kwargs):
  mjm, _, m, d = test_data.fixture(fixture, nworld=nworld, qvel_noise=0.5)
  rc = mjw.create_render_context(
    mjm, nworld=nworld, cam_res=(48, 48), render_rgb=True, render_depth=True, render_seg=True, backend=backend, **kwargs
  )
  for _ in range(steps):
    mjw.step(m, d)
  mjw.refit_bvh(m, d, rc)
  mjw.render(m, d, rc)
  return rc.rgb_data.numpy(), rc.depth_data.numpy(), rc.seg_data.numpy()


@absltest.skipIf(render_optix.wo is None, "requires warp_optix")
class RenderOptixTest(parameterized.TestCase):
  @parameterized.parameters("primitives.xml", "mug/mug.xml")
  def test_matches_bvh(self, fixture):
    """OptiX and BVH renders, with camera and shadow rays, agree up to float rounding."""
    (rgb, depth, seg), (rgb_optix, depth_optix, seg_optix) = (
      _render(fixture, backend, nworld=4, steps=8, use_shadows=True) for backend in ("bvh", "optix")
    )
    self.assertGreater(np.unique(seg).size, 2, "the scene must show several geoms")
    # Compare packed RGBA8 pixels channel by channel.
    np.testing.assert_array_less(np.abs(rgb.view(np.uint8).astype(int) - rgb_optix.view(np.uint8)), 2)
    np.testing.assert_allclose(depth, depth_optix, atol=1e-4)
    np.testing.assert_array_equal(seg, seg_optix)


if __name__ == "__main__":
  absltest.main()
