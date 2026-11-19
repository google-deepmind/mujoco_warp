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

"""Tests for warp_util."""

import numpy as np
import warp as wp
from absl.testing import absltest

from mujoco_warp import test_data
from mujoco_warp._src import smooth
from mujoco_warp._src import warp_util


class CacheKernelTest(absltest.TestCase):
  def test_numpy_scalar(self):
    """Numpy scalars are keyed by value, like the Python scalars they equal."""

    @warp_util.cache_kernel
    def scalar_factory(n, flag):
      return int(n), bool(flag)

    self.assertEqual(scalar_factory(np.int64(4), np.True_), (4, True))
    self.assertEqual(scalar_factory(np.int64(10), np.True_), (10, True))
    self.assertEqual(scalar_factory(np.int64(10), np.False_), (10, False))
    self.assertIs(scalar_factory(4, True), scalar_factory(np.int64(4), np.True_))

  def test_array(self):
    """Arrays are keyed by size."""

    @warp_util.cache_kernel
    def array_factory(a):
      return [a.size]

    self.assertIs(array_factory(np.zeros(3)), array_factory(np.ones(3)))
    self.assertIsNot(array_factory(np.zeros(3)), array_factory(np.zeros(4)))

  def test_transmission_body_moment(self):
    """Models with different condim get their own body transmission kernel."""
    kernels = []
    for condim in (3, 6):
      _, _, m, _ = test_data.fixture(
        xml=f"""
        <mujoco>
          <worldbody>
            <geom type="plane" size="10 10 .001" condim="{condim}"/>
            <body name="body">
              <geom type="sphere" size=".1"/>
              <joint type="slide" axis="0 0 1"/>
            </body>
          </worldbody>
          <actuator>
            <adhesion body="body" gain="1" ctrlrange="0 1"/>
          </actuator>
        </mujoco>
        """
      )
      self.assertEqual(m.nmaxpyramid, 2 * (condim - 1))
      # as launched with opt.deterministic=ATOMICS
      kernels.append(smooth._transmission_body_moment(m.nmaxpyramid, True))

    self.assertIsNot(kernels[0], kernels[1])


if __name__ == "__main__":
  wp.init()
  absltest.main()
