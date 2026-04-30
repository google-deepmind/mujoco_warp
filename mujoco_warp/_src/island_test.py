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

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import island
from mujoco_warp._src import solver
from mujoco_warp._src import types


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

    mjwarp.fwd_position(m, d)

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

    mjwarp.fwd_position(m, d)

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

    mjwarp.fwd_position(m, d)

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

    mjwarp.fwd_position(m, d)

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

    mjwarp.fwd_position(m, d)

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

    mjwarp.fwd_position(m, d)

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

    mjwarp.fwd_position(m, d)

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

    mjwarp.fwd_position(m, d)

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

    mjwarp.fwd_position(m, d)

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

    mjwarp.fwd_position(m, d)

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

    mjwarp.fwd_position(m, d)

    treetree = wp.empty((d.nworld, m.ntree, m.ntree), dtype=int)
    island.tree_edges(m, d, treetree)

    tt = treetree.numpy()
    self.assertEqual(tt[0, 0, 1], 1)
    self.assertEqual(tt[0, 1, 0], 1)
    self.assertEqual(tt[0, 0, 2], 1)
    self.assertEqual(tt[0, 2, 0], 1)


class IslandDiscoveryTest(absltest.TestCase):
  """Tests for full island discovery."""

  def test_two_trees_one_constraint_one_island(self):
    """Two trees connected by one constraint form one island.

    topology:
      [[0, 1],
       [1, 0]]
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag island="disable"/>
        </option>
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

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    mjwarp.fwd_position(m, d)
    island.island(m, d)

    # should have exactly 1 island
    self.assertEqual(d.nisland.numpy()[0], 1)
    # both trees should be in island 0
    tree_island = d.tree_island.numpy()[0]
    self.assertEqual(tree_island[0], tree_island[1])
    self.assertEqual(tree_island[0], 0)

  def test_three_trees_chain_one_island(self):
    """Three trees in a chain form one island.

    topology:
      [[0, 1, 0],
       [1, 0, 1],
       [0, 1, 0]]
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag island="disable"/>
        </option>
        <worldbody>
          <body name="body1">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="body2" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="body3" pos="2 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="body1" body2="body2"/>
          <weld body1="body2" body2="body3"/>
        </equality>
      </mujoco>
      """
    )

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    mjwarp.fwd_position(m, d)
    island.island(m, d)

    # should have exactly 1 island
    self.assertEqual(d.nisland.numpy()[0], 1)
    # all trees should be in the same island
    tree_island = d.tree_island.numpy()[0]
    self.assertEqual(tree_island[0], tree_island[1])
    self.assertEqual(tree_island[1], tree_island[2])

  def test_two_disconnected_pairs_two_islands(self):
    """Two pairs of disconnected trees form two islands.

    topology:
      [[0, 1, 0, 0],
       [1, 0, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]]
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag island="disable"/>
        </option>
        <worldbody>
          <body name="body1">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="body2" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="body3" pos="10 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="body4" pos="11 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="body1" body2="body2"/>
          <weld body1="body3" body2="body4"/>
        </equality>
      </mujoco>
      """
    )

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    mjwarp.fwd_position(m, d)
    island.island(m, d)

    # should have exactly 2 islands
    self.assertEqual(d.nisland.numpy()[0], 2)
    # trees 0,1 should be in one island, trees 2,3 in another
    tree_island = d.tree_island.numpy()[0]
    self.assertEqual(tree_island[0], tree_island[1])
    self.assertEqual(tree_island[2], tree_island[3])
    self.assertNotEqual(tree_island[0], tree_island[2])

  def test_no_constraints_no_islands(self):
    """No constraints means no constrained islands.

    topology:
      [[0]]  (no edges)
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag island="disable"/>
        </option>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    mjwarp.fwd_position(m, d)
    island.island(m, d)

    # should have 0 islands (unconstrained tree is not an island)
    self.assertEqual(d.nisland.numpy()[0], 0)

  def test_multiple_worlds(self):
    """Test island discovery with nworld=2.

    topology:
      [[0, 1],
       [1, 0]]
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag island="disable"/>
        </option>
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

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    mjwarp.fwd_position(m, d)
    island.island(m, d)

    # both worlds should have exactly 1 island
    nisland = d.nisland.numpy()
    self.assertEqual(nisland[0], 1)
    self.assertEqual(nisland[1], 1)

    # both trees in both worlds should be in island 0
    tree_island = d.tree_island.numpy()
    for worldid in range(2):
      self.assertEqual(tree_island[worldid, 0], 0)
      self.assertEqual(tree_island[worldid, 1], 0)

  def test_three_trees_star_hub_at_end(self):
    """Three trees with tree 2 as hub connecting trees 0 and 1.

    topology:
      [[0, 0, 1],
       [0, 0, 1],
       [1, 1, 0]]
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag island="disable"/>
        </option>
        <worldbody>
          <body name="body1">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="body2" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="body3" pos="2 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="body1" body2="body3"/>
          <weld body1="body2" body2="body3"/>
        </equality>
      </mujoco>
      """
    )

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    mjwarp.fwd_position(m, d)
    island.island(m, d)

    # should have exactly 1 island
    self.assertEqual(d.nisland.numpy()[0], 1)
    # all trees should be in the same island
    tree_island = d.tree_island.numpy()[0]
    self.assertEqual(tree_island[0], tree_island[1])
    self.assertEqual(tree_island[1], tree_island[2])


class IslandMappingTest(absltest.TestCase):
  """Tests for island DOF/constraint mapping and gather/scatter."""

  def test_two_body_weld_mapping(self):
    """Two free bodies with a weld: 1 island, all DOFs constrained."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.use_islands": True})
    ctx = solver.create_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nisland = d.nisland.numpy()[0]
    self.assertEqual(nisland, 1)

    # all DOFs should be in island 0
    dof_island = d.dof_island.numpy()[0, : m.nv]
    np.testing.assert_array_equal(dof_island, np.zeros(m.nv, dtype=int))

    # nidof == nv (all DOFs are in islands)
    nidof = ctx.nidof.numpy()[0]
    self.assertEqual(nidof, m.nv)

    # island_nv[0] == nv
    island_nv = ctx.island_nv.numpy()[0]
    self.assertEqual(island_nv[0], m.nv)

  def test_two_disconnected_pairs_mapping(self):
    """Two pairs of welded bodies: 2 islands, each with 12 DOFs."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="a1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b1" pos="5 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="6 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="a1" body2="a2"/>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.use_islands": True})
    ctx = solver.create_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nisland = d.nisland.numpy()[0]
    self.assertEqual(nisland, 2)

    # nidof == nv (all DOFs are in islands)
    nidof = ctx.nidof.numpy()[0]
    self.assertEqual(nidof, m.nv)

    # each island has 12 DOFs (2 free joints = 12 DOFs)
    island_nv = ctx.island_nv.numpy()[0]
    self.assertEqual(island_nv[0], 12)
    self.assertEqual(island_nv[1], 12)

  def test_unconstrained_body_excluded(self):
    """Body with no constraints gets dof_island=-1, is not in nidof."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="constrained1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="constrained2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="unconstrained" pos="5 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="constrained1" body2="constrained2"/>
      </equality>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.use_islands": True})
    ctx = solver.create_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nisland = d.nisland.numpy()[0]
    self.assertEqual(nisland, 1)

    dof_island = d.dof_island.numpy()[0, : m.nv]
    # first 12 DOFs (2 constrained bodies) in island 0
    np.testing.assert_array_equal(dof_island[:12], np.zeros(12, dtype=int))
    # last 6 DOFs (unconstrained body) should be -1
    np.testing.assert_array_equal(dof_island[12:18], -np.ones(6, dtype=int))

    # nidof == 12
    nidof = ctx.nidof.numpy()[0]
    self.assertEqual(nidof, 12)

  def test_map_roundtrip(self):
    """map_dof2idof and map_idof2dof are inverses for island DOFs."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.use_islands": True})
    ctx = solver.create_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nidof = ctx.nidof.numpy()[0]
    map_d2i = ctx.map_dof2idof.numpy()[0, : m.nv]
    map_i2d = ctx.map_idof2dof.numpy()[0, : m.nv]

    # roundtrip: for island DOFs, map_idof2dof[map_dof2idof[d]] == d
    for dof in range(m.nv):
      island_id = d.dof_island.numpy()[0, dof]
      if island_id >= 0:
        idof = map_d2i[dof]
        self.assertEqual(map_i2d[idof], dof)

  def test_efc_map_roundtrip(self):
    """map_efc2iefc and map_iefc2efc are inverses."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.use_islands": True})
    ctx = solver.create_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nefc = d.nefc.numpy()[0]
    map_e2i = ctx.map_efc2iefc.numpy()[0, :nefc]
    map_i2e = ctx.map_iefc2efc.numpy()[0, :nefc]

    # roundtrip: map_iefc2efc[map_efc2iefc[c]] == c
    for c in range(nefc):
      ic = map_e2i[c]
      self.assertEqual(map_i2e[ic], c)

  def test_mujoco_parity_mapping(self):
    """Compare DOF/constraint mapping arrays against MuJoCo C."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="a1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b1" pos="5 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="6 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="a1" body2="a2"/>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>
    """

    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.use_islands": True})
    ctx = solver.create_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nv = mjm.nv
    nisland = mjd.nisland
    nefc = mjd.nefc

    # Compare mapping arrays with MuJoCo C
    np.testing.assert_array_equal(
      ctx.island_nv.numpy()[0, :nisland],
      mjd.island_nv[:nisland],
    )
    np.testing.assert_array_equal(
      ctx.island_nefc.numpy()[0, :nisland],
      mjd.island_nefc[:nisland],
    )
    np.testing.assert_array_equal(
      d.island_dofadr.numpy()[0, :nisland],
      mjd.island_idofadr[:nisland],
    )
    np.testing.assert_array_equal(
      ctx.island_iefcadr.numpy()[0, :nisland],
      mjd.island_iefcadr[:nisland],
    )
    np.testing.assert_array_equal(
      d.dof_island.numpy()[0, :nv],
      mjd.dof_island[:nv],
    )
    np.testing.assert_array_equal(
      ctx.map_dof2idof.numpy()[0, :nv],
      mjd.map_dof2idof[:nv],
    )
    np.testing.assert_array_equal(
      ctx.map_idof2dof.numpy()[0, :nv],
      mjd.map_idof2dof[:nv],
    )
    np.testing.assert_array_equal(
      d.efc.island.numpy()[0, :nefc],
      mjd.efc_island[:nefc],
    )
    np.testing.assert_array_equal(
      ctx.map_efc2iefc.numpy()[0, :nefc],
      mjd.map_efc2iefc[:nefc],
    )
    np.testing.assert_array_equal(
      ctx.map_iefc2efc.numpy()[0, :nefc],
      mjd.map_iefc2efc[:nefc],
    )

  def test_gather_scatter_roundtrip(self):
    """Gather then scatter recovers original DOF arrays."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.use_islands": True})
    ctx = solver.create_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    # Save originals
    qacc_orig = d.qacc.numpy().copy()
    qfrc_constraint_orig = d.qfrc_constraint.numpy().copy()

    # Gather
    island.gather_island_inputs(m, d, ctx)

    # Verify gathered arrays are non-trivially reordered
    iacc = ctx.iacc.numpy()
    nidof = ctx.nidof.numpy()[0]

    # Simulate solver output: copy qacc_smooth into iacc (as solver would)
    # and set ifrc_constraint to some values
    wp.copy(ctx.iacc, ctx.iacc_smooth)
    ctx.ifrc_constraint.zero_()

    # Scatter back
    island.scatter_island_results(m, d, ctx, scatter_Ma=False)

    # After scatter, qacc should equal qacc_smooth at island DOF positions
    qacc_scattered = d.qacc.numpy()[0, : m.nv]
    qacc_smooth = d.qacc_smooth.numpy()[0, : m.nv]
    dof_island = d.dof_island.numpy()[0, : m.nv]

    for dof in range(m.nv):
      if dof_island[dof] >= 0:
        np.testing.assert_allclose(qacc_scattered[dof], qacc_smooth[dof], atol=1e-12)

  def test_gather_efc_ordering(self):
    """Gathered EFC arrays preserve values via island mapping."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.use_islands": True})
    ctx = solver.create_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)
    island.gather_island_inputs(m, d, ctx)

    nefc = d.nefc.numpy()[0]
    efc_D = d.efc.D.numpy()[0]
    iefc_D = ctx.iefc_D.numpy()[0]
    map_i2e = ctx.map_iefc2efc.numpy()[0]

    # iefc_D[ic] == efc_D[map_iefc2efc[ic]]
    for ic in range(nefc):
      c = map_i2e[ic]
      np.testing.assert_allclose(iefc_D[ic], efc_D[c], atol=1e-12)

  def test_island_ne_nf_parity(self):
    """island_ne and island_nf match MuJoCo C values."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="0.15 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>
    """

    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.use_islands": True})
    ctx = solver.create_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nisland = mjd.nisland

    if nisland > 0:
      np.testing.assert_array_equal(
        ctx.island_ne.numpy()[0, :nisland],
        mjd.island_ne[:nisland],
      )


class IslandSolverTest(parameterized.TestCase):
  """Tests for the parallel island solver."""

  @parameterized.parameters(*tuple(types.SolverType))
  def test_single_island_weld(self, solver):
    """Single island: weld constraint between two free bodies."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

    nefc = d_monolithic.nefc.numpy()[0]
    np.testing.assert_allclose(
      d_island.efc.force.numpy()[0, :nefc],
      d_monolithic.efc.force.numpy()[0, :nefc],
      atol=1e-4,
    )

  @parameterized.parameters(*tuple(types.SolverType))
  def test_multi_island_weld(self, solver):
    """Two independent weld pairs form two separate islands."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="a1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b1" pos="10 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="11 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="a1" body2="a2"/>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_three_islands(self, solver):
    """Three independent weld pairs form three separate islands."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="a1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b1" pos="5 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="6 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="c1" pos="10 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="c2" pos="11 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="a1" body2="a2"/>
        <weld body1="b1" body2="b2"/>
        <weld body1="c1" body2="c2"/>
      </equality>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_contact_constraint(self, solver):
    """Contact constraints from ground plane collision."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body name="box1" pos="0 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="1"/>
        </body>
        <body name="box2" pos="5 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="1"/>
        </body>
      </worldbody>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_contact_with_friction(self, solver):
    """Contact constraints with friction (condim=3, pyramidal)."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body name="box" pos="0 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="3" friction="0.5"/>
        </body>
      </worldbody>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_friction_joint(self, solver):
    """Hinge joint with frictionloss generates friction constraints."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="arm1">
          <joint type="hinge" axis="0 0 1" frictionloss="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.5 0 0" size=".05"/>
          <body name="arm2" pos="0.5 0 0">
            <joint type="hinge" axis="0 0 1" frictionloss="0.2"/>
            <geom type="capsule" fromto="0 0 0 0.5 0 0" size=".05"/>
          </body>
        </body>
      </worldbody>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

    nefc = d_monolithic.nefc.numpy()[0]
    np.testing.assert_allclose(
      d_island.efc.force.numpy()[0, :nefc],
      d_monolithic.efc.force.numpy()[0, :nefc],
      atol=1e-4,
    )

  @parameterized.parameters(*tuple(types.SolverType))
  def test_joint_limit(self, solver):
    """Joint with active limits generates inequality constraints."""
    xml = """
    <mujoco>
      <compiler autolimits="true"/>

      <worldbody>
        <body name="arm">
          <joint type="hinge" axis="0 0 1" range="-30 30" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0.5 0 0" size=".05"/>
        </body>
      </worldbody>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_connect_constraint(self, solver):
    """Connect (point) constraint between two bodies."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <connect body1="b1" body2="b2" anchor="0.5 0 0"/>
      </equality>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

    nefc = d_monolithic.nefc.numpy()[0]
    np.testing.assert_allclose(
      d_island.efc.force.numpy()[0, :nefc],
      d_monolithic.efc.force.numpy()[0, :nefc],
      atol=1e-4,
    )

  @parameterized.parameters(*tuple(types.SolverType))
  def test_mixed_constrained_unconstrained(self, solver):
    """Mix of constrained bodies (forming islands) and unconstrained bodies."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="free1" pos="-5 0 1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="free2" pos="5 0 1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="a1" body2="a2"/>
      </equality>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_multi_world(self, solver):
    """Island solver with multiple parallel worlds (nworld=4)."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      nworld=4,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(xml=xml, nworld=4, overrides={"opt.use_islands": True})
    mjwarp.forward(m, d_island)

    for w in range(4):
      np.testing.assert_allclose(d_island.qacc.numpy()[w], d_monolithic.qacc.numpy()[w], atol=1e-4)
      np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[w], d_monolithic.qfrc_constraint.numpy()[w], atol=1e-4)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_warmstart_disabled(self, solver):
    """Island solver with warmstart disabled."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
        "opt.disableflags": types.DisableBit.WARMSTART,
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
        "opt.disableflags": types.DisableBit.WARMSTART,
        "opt.use_islands": True,
      },
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_mujoco_c_parity(self, solver):
    """Island solver qacc should be close to MuJoCo C reference."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>"""

    mjm, mjd, m, d = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )

    # MuJoCo C reference
    mujoco.mj_forward(mjm, mjd)
    qacc_mjc = mjd.qacc.copy()

    # MuJoCo Warp island solver
    mjwarp.forward(m, d)
    qacc_warp = d.qacc.numpy()[0]

    np.testing.assert_allclose(
      qacc_warp,
      qacc_mjc,
      atol=1e-3,
      err_msg="qacc mismatch between island solver and MuJoCo C",
    )

  @parameterized.parameters(*tuple(types.SolverType))
  def test_multi_step(self, solver):
    """Island solver produces stable multi-step simulation."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="0.5 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>"""

    mjm, mjd, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )

    # Run 10 steps with island solver
    for _ in range(10):
      mjwarp.forward(m, d_island)
      mjwarp.step(m, d_island)

    qacc = d_island.qacc.numpy()[0]
    # Check that accelerations are finite and not NaN
    self.assertTrue(np.all(np.isfinite(qacc)), msg=f"Non-finite qacc after 10 steps: {qacc}")

    # Verify reasonable magnitude (shouldn't blow up)
    self.assertLess(np.max(np.abs(qacc)), 1e6, msg=f"qacc magnitude too large: {np.max(np.abs(qacc))}")

  @parameterized.parameters(*tuple(types.SolverType))
  def test_chain_single_island(self, solver):
    """Three bodies in a chain form one island with two welds."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b3" pos="2 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
        <weld body1="b2" body2="b3"/>
      </equality>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

    nefc = d_monolithic.nefc.numpy()[0]
    np.testing.assert_allclose(
      d_island.efc.force.numpy()[0, :nefc],
      d_monolithic.efc.force.numpy()[0, :nefc],
      atol=1e-4,
    )

  @parameterized.parameters(*tuple(types.SolverType))
  def test_asymmetric_islands(self, solver):
    """Islands of different sizes: one small (1 weld) and one large (chain)."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="a1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b1" pos="5 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="6 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b3" pos="7 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b4" pos="8 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="a1" body2="a2"/>
        <weld body1="b1" body2="b2"/>
        <weld body1="b2" body2="b3"/>
        <weld body1="b3" body2="b4"/>
      </equality>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_hinge_chain_with_limits(self, solver):
    """Kinematic chain with hinge joints and active limits."""
    xml = """
    <mujoco>
      <compiler autolimits="true"/>

      <worldbody>
        <body name="link1">
          <joint type="hinge" axis="0 1 0" range="-45 45" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.5 0 0" size=".03"/>
          <body name="link2" pos="0.5 0 0">
            <joint type="hinge" axis="0 1 0" range="-45 45" damping="0.1"/>
            <geom type="capsule" fromto="0 0 0 0.5 0 0" size=".03"/>
          </body>
        </body>
      </worldbody>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_ball_joint_limit(self, solver):
    """Ball joint with active limit generates inequality constraints."""
    xml = """
    <mujoco>
      <compiler autolimits="true"/>

      <worldbody>
        <body>
          <joint type="ball" range="0 30"/>
          <geom type="box" size=".1 .2 .3" pos=".1 .2 .3"/>
        </body>
      </worldbody>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml, overrides={"opt.solver": solver, "opt.iterations": 100, "opt.tolerance": "1e-10", "opt.use_islands": True}
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_contact_elliptic(self, solver):
    """Contact constraints with elliptic friction cone (condim=3)."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body name="box" pos="0 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="3" friction="0.5"/>
        </body>
      </worldbody>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.cone": types.ConeType.ELLIPTIC,
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides={
        "opt.cone": types.ConeType.ELLIPTIC,
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
        "opt.use_islands": True,
      },
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_contact_elliptic_condim4(self, solver):
    """Elliptic friction cones with condim=4 (normal + 2 tangential + spin)."""
    xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 .01"/>
        <body name="box" pos="0 0 0.15">
          <joint type="free"/>
          <geom name="box" type="box" size=".1 .1 .1" condim="4" friction="0.5 0.3 0.01"/>
        </body>
      </worldbody>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.cone": types.ConeType.ELLIPTIC,
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides={
        "opt.cone": types.ConeType.ELLIPTIC,
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
        "opt.use_islands": True,
      },
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.parameters(*tuple(types.SolverType))
  def test_contact_elliptic_multi_island(self, solver):
    """Two separate bodies with elliptic contacts forming separate islands."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body name="box1" pos="0 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="3" friction="0.5"/>
        </body>
        <body name="box2" pos="5 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="3" friction="0.5"/>
        </body>
      </worldbody>
    </mujoco>"""

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides={
        "opt.cone": types.ConeType.ELLIPTIC,
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    mjwarp.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides={
        "opt.cone": types.ConeType.ELLIPTIC,
        "opt.solver": solver,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
        "opt.use_islands": True,
      },
    )
    mjwarp.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.product(
    warmstart=[False, True],
    solver=list(types.SolverType),
    cone=list(types.ConeType),
  )
  def test_islands_equivalent_forward(self, warmstart, solver, cone):
    """Island vs. monolithic forward parity across solver/cone/warmstart combos.

    Mirrors MuJoCo C's IslandsEquivalentForward test: a single forward call
    comparing island and monolithic solvers across all combinations of
    solver type, cone type, and warmstart on a rich model with mixed
    constraint types.
    """
    xml = """
    <mujoco>
      <option iterations="100" tolerance="0" ls_iterations="20"/>

      <default>
        <geom size=".1"/>
      </default>

      <worldbody>
        <body>
          <joint type="slide" axis="0 0 1" range="0 1" limited="true"/>
          <geom/>
        </body>

        <body pos=".25 0 0">
          <joint type="slide" axis="1 0 0"/>
          <geom/>
        </body>

        <body pos="0 0 0.25">
          <joint type="slide" axis="0 0 1"/>
          <geom/>
          <body pos="0 -.15 0">
            <joint name="hinge1" axis="0 1 0"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
            <body pos="-.2 0 0">
              <joint axis="0 1 0"/>
              <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
            </body>
          </body>
        </body>

        <body pos=".5 0 0">
          <joint type="slide" axis="0 0 1" frictionloss="15"/>
          <geom type="box" size=".08 .08 .02" euler="0 10 0"/>
        </body>

        <body pos="-.5 0 0">
          <joint axis="0 1 0" frictionloss=".01"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
        </body>

        <body pos="0 0 .5">
          <joint name="hinge2" axis="0 1 0"/>
          <geom type="box" size=".08 .02 .08"/>
        </body>

        <body pos=".5 0 .1">
          <freejoint/>
          <geom type="box" size=".03 .03 .03" pos="0.01 0.01 0.01"/>
        </body>

        <site name="0" pos="-.45 -.05 .35"/>
        <body pos="-.5 0 .3" name="connect">
          <freejoint/>
          <geom type="box" size=".05 .05 .05"/>
          <site name="1" pos=".05 -.05 .05"/>
        </body>
      </worldbody>

      <equality>
        <joint joint1="hinge1" joint2="hinge2"/>
        <connect body1="connect" body2="world" anchor="-.05 -.05 .05"/>
        <connect site1="0" site2="1"/>
      </equality>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.cone": cone,
      "opt.use_islands": True,
    }
    if not warmstart:
      overrides["opt.disableflags"] = types.DisableBit.WARMSTART

    # Monolithic (islands disabled)
    overrides_monolithic = dict(overrides)
    overrides_monolithic["opt.use_islands"] = False
    # monolithic: use_islands defaults to False (no island solving)
    _, _, m_monolithic, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides_monolithic,
    )

    # step once to populate warmstart, then forward
    mjwarp.step(m_monolithic, d_monolithic)
    mjwarp.forward(m_monolithic, d_monolithic)

    # Island
    _, _, m_island, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    mjwarp.step(m_island, d_island)
    mjwarp.forward(m_island, d_island)

    qacc_island = d_island.qacc.numpy()[0]
    qacc_monolithic = d_monolithic.qacc.numpy()[0]
    scale = 0.5 * (np.linalg.norm(qacc_island) + np.linalg.norm(qacc_monolithic))
    rtol = 1e-3 if solver == types.SolverType.CG else 1e-4
    tol = max(scale * rtol, 1e-8)
    np.testing.assert_allclose(
      qacc_island,
      qacc_monolithic,
      atol=tol,
    )

  @parameterized.parameters(*tuple(types.SolverType))
  def test_islands_equivalent(self, solver):
    """Multi-step island vs. monolithic parity with synchronized state.

    Mirrors MuJoCo C's IslandsEquivalent test: runs multiple forward calls
    comparing island and monolithic solvers while keeping the state
    synchronized, verifying convergence agreement over time.
    """
    xml = """
    <mujoco>
      <option iterations="60" tolerance="0" ls_iterations="60"/>

      <default>
        <geom size=".1"/>
      </default>

      <worldbody>
        <body>
          <joint type="slide" axis="0 0 1" range="0 1" limited="true"/>
          <geom/>
        </body>

        <body pos=".25 0 0">
          <joint type="slide" axis="1 0 0"/>
          <geom/>
        </body>

        <body pos="0 0 0.25">
          <joint type="slide" axis="0 0 1"/>
          <geom/>
          <body pos="0 -.15 0">
            <joint name="hinge1" axis="0 1 0"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
            <body pos="-.2 0 0">
              <joint axis="0 1 0"/>
              <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
            </body>
          </body>
        </body>

        <body pos=".5 0 0">
          <joint type="slide" axis="0 0 1" frictionloss="15"/>
          <geom type="box" size=".08 .08 .02" euler="0 10 0"/>
        </body>

        <body pos="-.5 0 0">
          <joint axis="0 1 0" frictionloss=".01"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
        </body>

        <body pos="0 0 .5">
          <joint name="hinge2" axis="0 1 0"/>
          <geom type="box" size=".08 .02 .08"/>
        </body>

        <body pos=".5 0 .1">
          <freejoint/>
          <geom type="box" size=".03 .03 .03" pos="0.01 0.01 0.01"/>
        </body>

        <site name="0" pos="-.45 -.05 .35"/>
        <body pos="-.5 0 .3" name="connect">
          <freejoint/>
          <geom type="box" size=".05 .05 .05"/>
          <site name="1" pos=".05 -.05 .05"/>
        </body>
      </worldbody>

      <equality>
        <joint joint1="hinge1" joint2="hinge2"/>
        <connect body1="connect" body2="world" anchor="-.05 -.05 .05"/>
        <connect site1="0" site2="1"/>
      </equality>
    </mujoco>"""

    overrides_monolithic = {
      "opt.solver": solver,
      "opt.iterations": 60,
      "opt.tolerance": "0",
    }
    overrides_island = {
      "opt.solver": solver,
      "opt.iterations": 60,
      "opt.tolerance": "0",
      "opt.use_islands": True,
    }

    _, _, m_monolithic, d_monolithic = test_data.fixture(xml=xml, overrides=overrides_monolithic)
    _, _, m_island, d_island = test_data.fixture(xml=xml, overrides=overrides_island)

    nv = m_monolithic.nv
    rtol = 1e-3 if solver == types.SolverType.CG else 1e-4

    # Run 5 synchronized steps
    for step in range(5):
      # Synchronize state: copy monolithic qpos/qvel to island
      d_island.qpos.assign(d_monolithic.qpos)
      d_island.qvel.assign(d_monolithic.qvel)

      mjwarp.forward(m_monolithic, d_monolithic)
      mjwarp.forward(m_island, d_island)

      qacc_island = d_island.qacc.numpy()[0, :nv]
      qacc_monolithic = d_monolithic.qacc.numpy()[0, :nv]
      scale = 0.5 * (np.linalg.norm(qacc_island) + np.linalg.norm(qacc_monolithic))
      tol = max(scale * rtol, 1e-8)
      np.testing.assert_allclose(
        qacc_island,
        qacc_monolithic,
        atol=tol,
        err_msg=f"qacc mismatch at step {step}, solver={solver.name}",
      )

      mjwarp.step(m_monolithic, d_monolithic)


if __name__ == "__main__":
  wp.init()
  absltest.main()
