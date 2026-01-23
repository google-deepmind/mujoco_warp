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

import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import island


class IslandEdgeDiscoveryTest(absltest.TestCase):
  """Tests for edge discovery from constraint Jacobian."""

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

    # run forward to populate constraints
    mjwarp.forward(m, d)

    # find edges
    edges, nedge = island.find_tree_edges(m, d)

    # should have exactly 1 edge between tree 0 and tree 1
    self.assertEqual(nedge.numpy()[0], 1)
    edge_np = edges.numpy()
    self.assertEqual(edge_np[0, 0], 0)  # tree 0
    self.assertEqual(edge_np[0, 1], 1)  # tree 1

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
    edges, nedge = island.find_tree_edges(m, d)

    # should have exactly 1 self-edge for tree 0
    self.assertEqual(nedge.numpy()[0], 1)
    edge_np = edges.numpy()
    self.assertEqual(edge_np[0, 0], 0)  # tree 0
    self.assertEqual(edge_np[0, 1], 0)  # tree 0 (self-edge)

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
    edges, nedge = island.find_tree_edges(m, d)

    # should have 2 edges: (0,1) and (1,2)
    n = nedge.numpy()[0]
    self.assertEqual(n, 2)
    edge_np = edges.numpy()[:n]
    edges_set = set(tuple(e) for e in edge_np)
    self.assertIn((0, 1), edges_set)
    self.assertIn((1, 2), edges_set)

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
    edges, nedge = island.find_tree_edges(m, d)

    # should have 1 unique edge (0,1) despite 2 constraints
    self.assertEqual(nedge.numpy()[0], 1)

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
    edges, nedge = island.find_tree_edges(m, d)

    self.assertEqual(nedge.numpy()[0], 0)


if __name__ == "__main__":
  wp.init()
  absltest.main()
