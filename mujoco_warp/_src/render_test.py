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
"""Tests for render functions."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp import test_data

from . import render
from . import render_context


class RenderTest(parameterized.TestCase):
  @parameterized.parameters(2, 512)
  def test_render(self, nworld: int):
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)

    rc = render_context.create_render_context(
      mjm,
      m,
      d,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )

    render.render(m, d, rc)

    rgb = rc.rgb_data.numpy()
    depth = rc.depth_data.numpy()

    self.assertGreater(np.count_nonzero(rgb), 0)
    self.assertGreater(np.count_nonzero(depth), 0)

    self.assertNotEqual(np.unique(rgb).shape[0], 1)
    self.assertNotEqual(np.unique(depth).shape[0], 1)

  def test_render_humanoid(self):
    mjm, mjd, m, d = test_data.fixture("humanoid/humanoid.xml")
    rc = render_context.create_render_context(
      mjm,
      m,
      d,
      cam_res=(32, 32),
      render_rgb=True,
      render_depth=True,
    )
    render.render(m, d, rc)
    rgb = rc.rgb_data.numpy().copy()

    self.assertNotEqual(np.unique(rgb).shape[0], 1)


if __name__ == "__main__":
  wp.init()
  absltest.main()
