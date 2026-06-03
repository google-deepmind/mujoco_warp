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
from mujoco_warp._src import io
from mujoco_warp._src import island
from mujoco_warp._src import solver
from mujoco_warp._src import types

# Shared XML models used across multiple island tests.
# Basic weld constraint model.
_WELD_XML = """
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
  def setUp(self):
    super().setUp()
    io.ENABLE_ISLANDS = True

  def tearDown(self):
    io.ENABLE_ISLANDS = False
    super().tearDown()

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
  def setUp(self):
    super().setUp()
    io.ENABLE_ISLANDS = True

  def tearDown(self):
    io.ENABLE_ISLANDS = False
    super().tearDown()

  """Tests for island DOF/constraint mapping and gather/scatter."""

  def test_two_body_weld_mapping(self):
    """Two free bodies with a weld: 1 island, all DOFs constrained."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    ctx = solver.create_island_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nisland = d.nisland.numpy()[0]
    self.assertEqual(nisland, 1)

    # all DOFs should be in island 0
    dof_island = d.dof_island.numpy()[0, : m.nv]
    np.testing.assert_array_equal(dof_island, np.zeros(m.nv, dtype=int))

    # nidof == nv (all DOFs are in islands)
    nidof = d.nidof.numpy()[0]
    self.assertEqual(nidof, m.nv)

    # island_nv[0] == nv
    island_nv = d.island_nv.numpy()[0]
    self.assertEqual(island_nv[0], m.nv)

  def test_two_disconnected_pairs_mapping(self):
    """Two pairs of welded bodies: 2 islands, each with 12 DOFs."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
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
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    ctx = solver.create_island_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nisland = d.nisland.numpy()[0]
    self.assertEqual(nisland, 2)

    # nidof == nv (all DOFs are in islands)
    nidof = d.nidof.numpy()[0]
    self.assertEqual(nidof, m.nv)

    # each island has 12 DOFs (2 free joints = 12 DOFs)
    island_nv = d.island_nv.numpy()[0]
    self.assertEqual(island_nv[0], 12)
    self.assertEqual(island_nv[1], 12)

  def test_unconstrained_body_excluded(self):
    """Body with no constraints gets dof_island=-1, is not in nidof."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
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
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    ctx = solver.create_island_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nisland = d.nisland.numpy()[0]
    self.assertEqual(nisland, 1)

    dof_island = d.dof_island.numpy()[0, : m.nv]
    # first 12 DOFs (2 constrained bodies) in island 0
    np.testing.assert_array_equal(dof_island[:12], np.zeros(12, dtype=int))
    # last 6 DOFs (unconstrained body) should be -1
    np.testing.assert_array_equal(dof_island[12:18], -np.ones(6, dtype=int))

    # nidof == 12
    nidof = d.nidof.numpy()[0]
    self.assertEqual(nidof, 12)

  def test_map_roundtrip(self):
    """map_dof2idof and map_idof2dof are inverses for island DOFs."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    ctx = solver.create_island_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nidof = d.nidof.numpy()[0]
    map_d2i = d.map_dof2idof.numpy()[0, : m.nv]
    map_i2d = d.map_idof2dof.numpy()[0, : m.nv]

    # roundtrip: for island DOFs, map_idof2dof[map_dof2idof[d]] == d
    for dof in range(m.nv):
      island_id = d.dof_island.numpy()[0, dof]
      if island_id >= 0:
        idof = map_d2i[dof]
        self.assertEqual(map_i2d[idof], dof)

  def test_efc_map_roundtrip(self):
    """map_efc2iefc and map_iefc2efc are inverses."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    ctx = solver.create_island_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nefc = d.nefc.numpy()[0]
    map_e2i = d.map_efc2iefc.numpy()[0, :nefc]
    map_i2e = d.map_iefc2efc.numpy()[0, :nefc]

    # roundtrip: map_iefc2efc[map_efc2iefc[c]] == c
    for c in range(nefc):
      ic = map_e2i[c]
      self.assertEqual(map_i2e[ic], c)

  def test_mujoco_parity_mapping(self):
    """Compare DOF/constraint mapping arrays against MuJoCo C."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
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
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    ctx = solver.create_island_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nv = mjm.nv
    nisland = mjd.nisland
    nefc = mjd.nefc

    # Compare mapping arrays with MuJoCo C
    np.testing.assert_array_equal(
      d.island_nv.numpy()[0, :nisland],
      mjd.island_nv[:nisland],
    )
    np.testing.assert_array_equal(
      d.island_nefc.numpy()[0, :nisland],
      mjd.island_nefc[:nisland],
    )
    np.testing.assert_array_equal(
      d.island_dofadr.numpy()[0, :nisland],
      mjd.island_idofadr[:nisland],
    )
    np.testing.assert_array_equal(
      d.island_efcadr.numpy()[0, :nisland],
      mjd.island_iefcadr[:nisland],
    )
    np.testing.assert_array_equal(
      d.dof_island.numpy()[0, :nv],
      mjd.dof_island[:nv],
    )
    np.testing.assert_array_equal(
      d.map_dof2idof.numpy()[0, :nv],
      mjd.map_dof2idof[:nv],
    )
    np.testing.assert_array_equal(
      d.map_idof2dof.numpy()[0, :nv],
      mjd.map_idof2dof[:nv],
    )
    np.testing.assert_array_equal(
      d.efc.island.numpy()[0, :nefc],
      mjd.efc_island[:nefc],
    )
    np.testing.assert_array_equal(
      d.map_efc2iefc.numpy()[0, :nefc],
      mjd.map_efc2iefc[:nefc],
    )
    np.testing.assert_array_equal(
      d.map_iefc2efc.numpy()[0, :nefc],
      mjd.map_iefc2efc[:nefc],
    )

  def test_gather_scatter_roundtrip(self):
    """Gather then scatter recovers original DOF arrays."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    ctx = solver.create_island_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    # Save originals
    qacc_orig = d.qacc.numpy().copy()
    qfrc_constraint_orig = d.qfrc_constraint.numpy().copy()

    # Gather
    island.gather_island_inputs(m, d, ctx)

    # Verify gathered arrays are non-trivially reordered
    iacc = d.iqacc.numpy()
    nidof = d.nidof.numpy()[0]

    # Simulate solver output: copy qacc_smooth into iacc (as solver would)
    # and set ifrc_constraint to some values
    wp.copy(d.iqacc, d.iqacc_smooth)
    d.iqfrc_constraint.zero_()

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
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    ctx = solver.create_island_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)
    island.gather_island_inputs(m, d, ctx)

    nefc = d.nefc.numpy()[0]
    efc_D = d.efc.D.numpy()[0]
    iefc_D = d.efc.iD.numpy()[0]
    map_i2e = d.map_iefc2efc.numpy()[0]

    # iefc_D[ic] == efc_D[map_iefc2efc[ic]]
    for ic in range(nefc):
      c = map_i2e[ic]
      np.testing.assert_allclose(iefc_D[ic], efc_D[c], atol=1e-12)

  def test_island_ne_nf_parity(self):
    """island_ne and island_nf match MuJoCo C values."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    ctx = solver.create_island_solver_context(m, d)
    island.compute_island_mapping(m, d, ctx)

    nisland = mjd.nisland

    if nisland > 0:
      np.testing.assert_array_equal(
        d.island_ne.numpy()[0, :nisland],
        mjd.island_ne[:nisland],
      )


if __name__ == "__main__":
  wp.init()
  absltest.main()
