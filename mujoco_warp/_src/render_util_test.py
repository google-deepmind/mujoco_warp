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

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp import test_data
from mujoco_warp._src import render_util
from mujoco_warp._src import types


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


if __name__ == "__main__":
  wp.init()
  absltest.main()
