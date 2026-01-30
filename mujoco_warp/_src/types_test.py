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

"""Tests for the core MuJoCo Warp types."""

import dataclasses

import mujoco
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import Option


class TypesTest(parameterized.TestCase):
  @parameterized.parameters((mujoco.MjOption, Option), (mujoco.MjModel, Model), (mujoco.MjData, Data))
  def test_field_order(self, mj_class, mjw_class):
    """Tests that MJW field order matches MuJoCo, and all warp-only fields are at the end."""
    self.maxDiff = None

    mj_fields = list(mj_class._all_fields)
    mjw_fields = [f.name for f in dataclasses.fields(mjw_class)]

    # _all_fields are missing struct fields
    if mjw_class is Model:
      mj_fields.insert(mj_fields.index("nbuffer") + 1, "opt")
      mj_fields.insert(mj_fields.index("nbuffer") + 2, "vis")
      mj_fields.insert(mj_fields.index("nbuffer") + 3, "stat")
    elif mjw_class is Data:
      # TODO(team): remove this reordering after MjData._all_fields order is fixed
      # there's a bug in _all_fields where solver_niter is in the wrong place
      mj_fields.insert(0, mj_fields.pop(mj_fields.index("solver_niter")))
      # MuJoCo's "warning" field is different from our warp-only warning system
      mj_fields.remove("warning")

    mj_set, mjw_set = set(mj_fields), set(mjw_fields)

    # first, put any union fields
    desired_fields = [f for f in mj_fields if f in mjw_set]
    # then, put any warp-only fields
    desired_fields.extend(f for f in mjw_fields if f not in mj_set)

    actual_fields = [f.name for f in dataclasses.fields(mjw_class)]
    self.assertListEqual(actual_fields, desired_fields)

  @parameterized.parameters(Option, Model, Data)
  def test_docstring_order(self, mjw_class):
    """Tests that docstring attribute order matches class attribute order."""
    self.maxDiff = None

    # curiously, there's no ruff rule for this, so rely on a unit test
    docstring_lines = [l.strip() for l in mjw_class.__doc__.splitlines()]
    attr_lines = docstring_lines[docstring_lines.index("Attributes:") + 1 :]
    attrs = []
    for line in attr_lines:
      if "warp only" in line:
        continue  # skip "warp only" section headers
      if ":" in line:
        attrs.append(line.split(":")[0].strip())

    self.assertListEqual(attrs, [f.name for f in dataclasses.fields(mjw_class)])


if __name__ == "__main__":
  wp.init()
  absltest.main()
