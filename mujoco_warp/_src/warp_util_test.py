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

"""Tests for Warp launch utilities."""

from unittest import mock

import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp._src import warp_util


class WarpUtilTest(parameterized.TestCase):
  def test_efc_threads_per_world_cpu(self):
    """CPU launches use capacity without querying CUDA occupancy."""
    device = wp.get_device("cpu")
    kernel = mock.Mock(spec=wp.Kernel)
    with mock.patch.object(wp, "get_suggested_block_size") as occupancy:
      for capacity in (0, 16, 32, 65, 384):
        self.assertEqual(warp_util.efc_threads_per_world(2048, capacity, device, kernel), capacity)
      occupancy.assert_not_called()

  @parameterized.parameters(0, 1, 16, 31, 32)
  def test_efc_threads_per_world_small_capacity(self, capacity):
    """Small and empty row buffers never launch beyond their capacity."""
    device = wp.get_device()
    kernel = mock.Mock(spec=wp.Kernel)
    with mock.patch.object(wp, "get_suggested_block_size") as occupancy:
      self.assertEqual(warp_util.efc_threads_per_world(2048, capacity, device, kernel), capacity)
      occupancy.assert_not_called()

  @parameterized.parameters(
    (1, 384, 6, 384),
    (512, 385, 6, 384),
    (2048, 384, 6, 384),
    (4096, 384, 6, 288),
    (8192, 384, 6, 128),
    (8192, 384, 1, 32),
    (513, 1024, 1, 416),
    (0, 65, 1, 64),
  )
  def test_efc_threads_per_world_cuda(self, nworld, capacity, num_waves, expected):
    """Kernel occupancy controls whole-warp widths even for uneven batch sizes."""
    device = wp.get_device()
    if not device.is_cuda:
      self.skipTest("CUDA launch sizing")
    kernel = mock.Mock(spec=wp.Kernel)
    with mock.patch.object(warp_util.wp, "get_suggested_block_size", return_value=(640, 340)) as occupancy:
      width = warp_util.efc_threads_per_world(nworld, capacity, device, kernel, num_waves)
      self.assertEqual(width, expected)
      self.assertLessEqual(width, capacity)
      self.assertEqual(width % 32, 0)
      occupancy.assert_called_once_with(kernel, device)


if __name__ == "__main__":
  absltest.main()
