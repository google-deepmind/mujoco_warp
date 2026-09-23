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
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp import test_data
from mujoco_warp._src.collision_driver import collision
from mujoco_warp._src.constraint import make_constraint
from mujoco_warp._src.forward import step
from mujoco_warp._src.io import override_model
from mujoco_warp._src.island import island
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DeterminismType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import Option
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import TileSet


class TypesTest(parameterized.TestCase):
  @parameterized.parameters(1, 3)
  def test_tileset_structural_equality_and_hash(self, count):
    tile_a = TileSet(wp.array(np.arange(count) * 6, dtype=int), 16)
    tile_b = TileSet(wp.array(np.arange(count) * 6, dtype=int), 16)
    tile_c = TileSet(wp.array(np.arange(count) * 6 + 1, dtype=int), 16)

    self.assertEqual(tile_a, tile_b)
    self.assertEqual(hash(tile_a), hash(tile_b))
    self.assertNotEqual(tile_a, tile_c)

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

  def test_overflow_type_flags(self):
    self.assertEqual(int(OverflowType.NONE), 0)
    self.assertEqual(int(OverflowType.NEFC), 1 << 0)
    self.assertEqual(int(OverflowType.NJMAX_NNZ), 1 << 1)
    self.assertEqual(int(OverflowType.BROADPHASE), 1 << 2)
    self.assertEqual(int(OverflowType.NARROWPHASE), 1 << 3)
    self.assertEqual(int(OverflowType.CCD), 1 << 4)
    self.assertEqual(int(OverflowType.HFIELD), 1 << 5)
    self.assertEqual(int(OverflowType.CONTACT_MATCH), 1 << 6)
    self.assertEqual(int(OverflowType.NVMAX), 1 << 7)
    self.assertEqual(int(OverflowType.EPA_HORIZON), 1 << 8)
    self.assertEqual(int(OverflowType.ITERATIONS), 1 << 9)
    self.assertEqual(int(OverflowType.LS_ITERATIONS), 1 << 10)
    self.assertEqual(int(OverflowType.TACTILE), 1 << 11)
    self.assertEqual(int(OverflowType.ALL), (1 << 12) - 1)

  def test_option_warn_overflow(self):
    _, _, m, _ = test_data.fixture(
      xml="""
      <mujoco/>
      """
    )

    # Defaults to ALL
    self.assertEqual(m.opt.warn_overflow, int(OverflowType.ALL))

    # Setting boolean False converts to 0
    m.opt.warn_overflow = False
    self.assertEqual(m.opt.warn_overflow, 0)

    # Setting boolean True converts to OverflowType.ALL
    m.opt.warn_overflow = True
    self.assertEqual(m.opt.warn_overflow, int(OverflowType.ALL))

    # Setting selective bitmask
    m.opt.warn_overflow &= ~OverflowType.NEFC
    self.assertEqual(m.opt.warn_overflow, int(OverflowType.ALL) & ~OverflowType.NEFC)
    self.assertFalse(bool(m.opt.warn_overflow & OverflowType.NEFC))
    self.assertTrue(bool(m.opt.warn_overflow & OverflowType.CCD))

    # Test override_model support
    for override, expected in [
      ({"opt.warn_overflow": 0}, 0),
      ({"opt.warn_overflow": "CCD"}, int(OverflowType.CCD)),
      (["opt.warn_overflow=ALL"], int(OverflowType.ALL)),
      (
        ["opt.warn_overflow=~ITERATIONS|~LS_ITERATIONS"],
        int(OverflowType.ALL) & ~OverflowType.ITERATIONS & ~OverflowType.LS_ITERATIONS,
      ),
    ]:
      override_model(m, override)
      self.assertEqual(m.opt.warn_overflow, expected)

  def test_determinism_type_flags(self):
    self.assertEqual(int(DeterminismType.NONE), 0)
    self.assertEqual(int(DeterminismType.CONTACTS), 1 << 0)
    self.assertEqual(int(DeterminismType.CONSTRAINT), 1 << 1)
    self.assertEqual(int(DeterminismType.ATOMICS), 1 << 2)
    self.assertEqual(int(DeterminismType.ISLANDS), 1 << 3)
    self.assertEqual(int(DeterminismType.ALL), (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3))

  def test_option_deterministic(self):
    _, _, m, _ = test_data.fixture(
      xml="""
      <mujoco/>
      """
    )

    # Defaults to NONE
    self.assertEqual(m.opt.deterministic, int(DeterminismType.NONE))
    self.assertFalse(bool(m.opt.deterministic))

    # Setting boolean True converts to ALL
    m.opt.deterministic = True
    self.assertEqual(m.opt.deterministic, int(DeterminismType.ALL))
    self.assertTrue(bool(m.opt.deterministic & DeterminismType.CONTACTS))
    self.assertTrue(bool(m.opt.deterministic & DeterminismType.CONSTRAINT))
    self.assertTrue(bool(m.opt.deterministic & DeterminismType.ATOMICS))
    self.assertTrue(bool(m.opt.deterministic & DeterminismType.ISLANDS))

    # Setting boolean False converts to NONE
    m.opt.deterministic = False
    self.assertEqual(m.opt.deterministic, int(DeterminismType.NONE))
    self.assertFalse(bool(m.opt.deterministic))

    # Setting individual flags with orthogonality check
    flags = (
      DeterminismType.CONTACTS,
      DeterminismType.CONSTRAINT,
      DeterminismType.ATOMICS,
      DeterminismType.ISLANDS,
    )
    for flag in flags:
      m.opt.deterministic = flag
      self.assertEqual(m.opt.deterministic, int(flag))
      for other in flags:
        self.assertEqual(bool(m.opt.deterministic & other), flag == other)

    # Test override_model support
    for override, expected in [
      ({"opt.deterministic": 0}, int(DeterminismType.NONE)),
      ({"opt.deterministic": True}, int(DeterminismType.ALL)),
      ({"opt.deterministic": "CONTACTS"}, int(DeterminismType.CONTACTS)),
      (
        ["opt.deterministic=CONSTRAINT|ATOMICS|ISLANDS"],
        int(DeterminismType.CONSTRAINT | DeterminismType.ATOMICS | DeterminismType.ISLANDS),
      ),
      (["opt.deterministic=ALL"], int(DeterminismType.ALL)),
      (["opt.deterministic=~CONTACTS"], int(DeterminismType.ALL) & ~DeterminismType.CONTACTS),
    ]:
      override_model(m, override)
      self.assertEqual(m.opt.deterministic, expected)

  @parameterized.parameters(1, 2)
  def test_determinism_pipeline_hooks(self, nworld):
    """Tests that sort hooks and atomic determinism modes execute cleanly across the pipeline."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body pos="0 0 0.05">
            <freejoint/>
            <geom type="sphere" size=".1"/>
          </body>
          <geom type="plane" size="1 1 .1"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1, 2] += 0.5
      d.qpos.assign(qpos)

    # 1. Step with atomics modes
    for flag in (DeterminismType.NONE, DeterminismType.ATOMICS, DeterminismType.ALL):
      m.opt.deterministic = flag
      d.qacc.fill_(wp.inf)
      step(m, d)
      self.assertFalse(np.any(np.isinf(d.qacc.numpy())))

    # 2. Test sort hooks
    with mock.patch("mujoco_warp._src.collision_driver._sort_contacts") as mock_sort_contacts:
      m.opt.deterministic = DeterminismType.CONTACTS
      collision(m, d)
      self.assertTrue(mock_sort_contacts.called)

    with mock.patch("mujoco_warp._src.constraint._sort_constraints") as mock_sort_constraints:
      m.opt.deterministic = DeterminismType.CONSTRAINT
      make_constraint(m, d)
      self.assertTrue(mock_sort_constraints.called)

    with mock.patch("mujoco_warp._src.island._sort_islands") as mock_sort_islands:
      m.opt.deterministic = DeterminismType.ISLANDS
      island(m, d)
      self.assertTrue(mock_sort_islands.called)

    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))


if __name__ == "__main__":
  wp.init()
  absltest.main()
