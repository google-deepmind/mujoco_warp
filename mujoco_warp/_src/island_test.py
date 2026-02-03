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

"""Tests for island discovery."""

import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import island


class IslandEdgeDiscoveryTest(absltest.TestCase):
  """Tests for edge discovery from constraint Jacobian."""

  # TODO(team): add test for additional constraint types to test special cases

  def test_single_constraint_two_trees(self):
    """A single weld constraint between two bodies creates one edge."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="body1">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="body2" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="body1" body2="body2"/>
        </equality>
      </mujoco>
      """
    )

    mjwarp.forward(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(tt[0, 0, 1], 1)
    self.assertEqual(tt[0, 1, 0], 1)

  def test_constraint_within_single_tree_creates_self_edge(self):
    """A constraint within a single tree creates a self-edge."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="body1">
            <joint name="j1" type="slide"/>
            <geom size=".1"/>
            <body name="body2" pos="0 0 0.5">
              <joint name="j2" type="slide"/>
              <geom size=".1"/>
            </body>
          </body>
        </worldbody>
        <equality>
          <joint joint1="j1" joint2="j2"/>
        </equality>
      </mujoco>
      """
    )

    mjwarp.forward(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(tt[0, 0, 0], 1)  # self-edge for tree 0

  def test_three_bodies_chain(self):
    """Three bodies with constraints A-B and B-C should have 2 edges."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="A">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="B" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="C" pos="2 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="A" body2="B"/>
          <weld body1="B" body2="C"/>
        </equality>
      </mujoco>
      """
    )

    mjwarp.forward(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(tt[0, 0, 1], 1)
    self.assertEqual(tt[0, 1, 0], 1)
    self.assertEqual(tt[0, 1, 2], 1)
    self.assertEqual(tt[0, 2, 1], 1)

  def test_deduplication(self):
    """Repeated constraints between same trees should be deduplicated."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="body1">
            <joint name="j1" type="free"/>
            <geom size=".1"/>
          </body>
          <body name="body2" pos="1 0 0">
            <joint name="j2" type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="body1" body2="body2"/>
          <connect body1="body1" body2="body2" anchor="0.5 0 0"/>
        </equality>
      </mujoco>
      """
    )

    mjwarp.forward(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(tt[0, 0, 1], 1)
    self.assertEqual(tt[0, 1, 0], 1)
    self.assertEqual(np.sum(tt[0]), 2)

  def test_no_constraints(self):
    """No constraints should produce no edges."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )

    mjwarp.forward(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(np.sum(tt[0]), 0)

  def test_multi_world_parallel(self):
    """Each world's edges should be computed independently."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="body1">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="body2" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="body1" body2="body2"/>
        </equality>
      </mujoco>
      """,
      nworld=2,
    )

    mjwarp.forward(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(tt[0, 0, 1], 1)
    self.assertEqual(tt[0, 1, 0], 1)
    self.assertEqual(tt[1, 0, 1], 1)
    self.assertEqual(tt[1, 1, 0], 1)

  def test_contact_constraint_edges(self):
    """Contact constraints between geoms should create edges."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="body1" pos="0 0 0.5">
            <joint type="free"/>
            <geom size=".3"/>
          </body>
          <body name="body2" pos="0 0 1.1">
            <joint type="free"/>
            <geom size=".3"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
    )

    mjwarp.forward(m, d)

    nefc = d.nefc.numpy()
    if nefc[0] > 0:
      treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
      island.tree_edges(m, d, treetree)

      tt = treetree.numpy()
      self.assertEqual(tt[0, 0, 1], 1)
      self.assertEqual(tt[0, 1, 0], 1)

  def test_isolated_tree_no_edge(self):
    """A floating body with no constraints should produce no edges."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body pos="0 0 10">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
    )

    mjwarp.forward(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(np.sum(tt[0]), 0)
    self.assertEqual(np.sum(tt[1]), 0)

  def test_mixed_equality_and_contact(self):
    """Both equality and contact constraints should contribute to edges."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="A" pos="0 0 0.5">
            <joint type="free"/>
            <geom size=".2"/>
          </body>
          <body name="B" pos="0 0 1.0">
            <joint type="free"/>
            <geom size=".2"/>
          </body>
          <body name="C" pos="2 0 0.5">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="B" body2="C"/>
        </equality>
      </mujoco>
      """,
      nworld=2,
    )

    mjwarp.forward(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(tt[0, 1, 2], 1)
    self.assertEqual(tt[0, 2, 1], 1)

  def test_worldbody_dofs_ignored(self):
    """Constraints involving worldbody (tree < 0) should not cause spurious edges."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="fixed" pos="0 0 0">
            <geom size=".1"/>
          </body>
          <body name="floating" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="world" body2="floating"/>
        </equality>
      </mujoco>
      """,
      nworld=2,
    )

    mjwarp.forward(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(tt[0, 0, 0], 1)  # self-edge for floating tree

  def test_constraint_touches_three_trees(self):
    """Multiple constraints sharing a body create a star topology."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="A" pos="0 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="B" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="C" pos="2 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="A" body2="B"/>
          <weld body1="A" body2="C"/>
        </equality>
      </mujoco>
      """,
      nworld=2,
    )

    mjwarp.forward(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(tt[0, 0, 1], 1)
    self.assertEqual(tt[0, 1, 0], 1)
    self.assertEqual(tt[0, 0, 2], 1)
    self.assertEqual(tt[0, 2, 0], 1)


if __name__ == "__main__":
  wp.init()
  absltest.main()
