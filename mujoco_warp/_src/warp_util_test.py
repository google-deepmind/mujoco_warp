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

import inspect
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest

from mujoco_warp._src import constraint
from mujoco_warp._src import solver
from mujoco_warp._src import warp_util


class WarpUtilTest(absltest.TestCase):
  def test_efc_threads_per_world(self):
    """Launch widths span capacity to one warp without a separate kernel variant."""
    cpu = SimpleNamespace(is_cuda=False)
    gpu = SimpleNamespace(is_cuda=True, sm_count=128)
    for capacity in (0, 16, 384):
      self.assertEqual(warp_util.efc_threads_per_world(2048, capacity, cpu), capacity)
    widths = [warp_util.efc_threads_per_world(nworld, 384, gpu) for nworld in (1, 128, 256, 512, 2048)]
    self.assertEqual(widths[0], 384)
    self.assertEqual(widths[-1], 32)
    self.assertEqual(widths, sorted(widths, reverse=True))
    self.assertGreater(len(set(widths)), 2)

  def test_iterative_efc_threads_per_world(self):
    """Iterative launches use kernel occupancy while contact launches keep their width."""
    cpu = SimpleNamespace(is_cuda=False)
    gpu = SimpleNamespace(is_cuda=True, sm_count=170)
    kernel = object()
    with mock.patch.object(warp_util.wp, "get_suggested_block_size", return_value=(640, 340)) as occupancy:
      self.assertEqual(warp_util.efc_threads_per_world(2048, 384, cpu, kernel), 384)
      self.assertEqual(warp_util.efc_threads_per_world(2048, 384, gpu), 32)
      occupancy.assert_not_called()
      widths = [warp_util.efc_threads_per_world(nworld, 384, gpu, kernel) for nworld in (2048, 4096, 8192)]
      self.assertEqual(widths, [384, 288, 128])
      self.assertEqual(occupancy.call_args_list, [mock.call(kernel, gpu)] * 3)

  def test_sparse_launch_structure(self):
    """Keep one launch-sizing owner and constraint evaluation out of iterative line search."""
    self.assertEqual(warp_util.efc_threads_per_world.__module__, warp_util.__name__)
    for module in (constraint, solver, warp_util):
      self.assertNotRegex(inspect.getsource(module), r"\b(WORLD_WARP|world_warp|launch_world_warp_enabled)\b")
      self.assertIs(module.efc_threads_per_world, warp_util.efc_threads_per_world)
    for function in (solver._linesearch_iterative_kernel, solver._linesearch_iterative):
      self.assertNotRegex(inspect.getsource(function), r"\b(_eval_constraint|fuse_constraint_update)\b")


if __name__ == "__main__":
  absltest.main()
