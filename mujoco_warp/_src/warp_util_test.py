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

import ast
from pathlib import Path
from types import SimpleNamespace

from absl.testing import absltest

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

  def test_sparse_launch_structure(self):
    """Keep one launch-sizing owner and constraint evaluation out of iterative line search."""
    directory = Path(__file__).parent
    forbidden = {"WORLD_WARP", "world_warp", "launch_world_warp_enabled"}
    owners = []
    for module in ("constraint", "solver", "warp_util"):
      tree = ast.parse((directory / f"{module}.py").read_text())
      for node in ast.walk(tree):
        if isinstance(node, ast.Name):
          self.assertNotIn(node.id, forbidden)
        elif isinstance(node, ast.arg):
          self.assertNotIn(node.arg, forbidden)
        elif isinstance(node, ast.FunctionDef):
          self.assertNotIn(node.name, forbidden)
          if node.name == "efc_threads_per_world":
            owners.append(module)
      if module == "solver":
        for node in tree.body:
          if isinstance(node, ast.FunctionDef) and node.name in ("_linesearch_iterative_kernel", "_linesearch_iterative"):
            names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
            self.assertNotIn("_eval_constraint", names)
            self.assertNotIn("fuse_constraint_update", names)
    self.assertEqual(owners, ["warp_util"])


if __name__ == "__main__":
  absltest.main()
