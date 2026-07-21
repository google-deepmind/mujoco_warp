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

import inspect
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import island
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

_ELLIPTIC_CONTACT_XML = """
<mujoco>
  <option cone="elliptic"/>
  <worldbody>
    <body name="left" pos="-.05 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
    <body name="right" pos=".05 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>
"""


def _site_equality_xml(kind: str) -> str:
  return f"""
  <mujoco>
    <worldbody>
      <body name="left" pos="-1 0 0"><freejoint/><geom size=".1"/><site name="left_site"/></body>
      <body name="right" pos="1 0 0"><freejoint/><geom size=".1"/><site name="right_site"/></body>
    </worldbody>
    <equality><{kind} site1="left_site" site2="right_site"/></equality>
  </mujoco>
  """


def _chain_xml(count: int) -> str:
  bodies = "".join(f'<body name="b{i}" pos="{i * 0.3} 0 0"><freejoint/><geom size=".1"/></body>' for i in range(count))
  welds = "".join(f'<weld body1="b{i}" body2="b{i + 1}"/>' for i in range(count - 1))
  return f"<mujoco><worldbody>{bodies}</worldbody><equality>{welds}</equality></mujoco>"


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

  def test_parent_workspace_uses_minimum_tree_identity_pointers(self):
    """Active parents are compressed pointers to their component's minimum tree."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option><flag island="disable"/></option>
        <worldbody>
          <body name="a"><freejoint/><geom size=".1"/></body>
          <body name="b" pos="1 0 0"><freejoint/><geom size=".1"/></body>
          <body name="c" pos="2 0 0"><freejoint/><geom size=".1"/></body>
          <body name="free" pos="4 0 0"><freejoint/><geom size=".1"/></body>
        </worldbody>
        <equality>
          <weld body1="c" body2="b"/>
          <weld body1="b" body2="a"/>
        </equality>
      </mujoco>
      """,
      nworld=2,
    )
    del mjm, mjd
    mjwarp.fwd_position(m, d)

    island.island(m, d)
    parent = d.island_parent.numpy()
    tree_island = d.tree_island.numpy()

    for world in range(d.nworld):
      active = np.flatnonzero(tree_island[world] >= 0)
      self.assertGreater(active.size, 0)
      component_minimum = int(active.min())
      np.testing.assert_array_equal(parent[world, active], component_minimum)
      self.assertTrue(np.all(parent[world, active] >= 0))
      self.assertTrue(np.all(parent[world, active] <= active))
      self.assertEqual(parent[world, component_minimum], component_minimum)
      inactive = np.flatnonzero(tree_island[world] < 0)
      np.testing.assert_array_equal(parent[world, inactive], inactive)

  def test_parent_workspace_and_canonical_labels(self):
    """DSU storage is linear and labels are ordered by their first active tree."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option><flag island="disable"/></option>
        <worldbody>
          <body name="a"><joint type="free"/><geom size=".1"/></body>
          <body name="b" pos="1 0 0"><joint type="free"/><geom size=".1"/></body>
          <body name="c" pos="5 0 0"><joint type="free"/><geom size=".1"/></body>
          <body name="d" pos="6 0 0"><joint type="free"/><geom size=".1"/></body>
          <body name="free" pos="10 0 0"><joint type="free"/><geom size=".1"/></body>
        </worldbody>
        <equality>
          <weld body1="a" body2="b"/>
          <weld body1="c" body2="d"/>
        </equality>
      </mujoco>
      """,
      nworld=2,
    )
    del mjm, mjd

    self.assertEqual(d.island_parent.shape, (d.nworld, m.ntree))
    self.assertEqual(d.island_parent.dtype, wp.int32)
    self.assertEqual(d.island_parent.capacity, d.nworld * m.ntree * 4)

    mjwarp.fwd_position(m, d)
    island.island(m, d)
    np.testing.assert_array_equal(
      d.tree_island.numpy(),
      np.array([[0, 0, 1, 1, -1], [0, 0, 1, 1, -1]], dtype=np.int32),
    )

  def test_inactive_efc_suffix_is_ignored_per_world(self):
    """Allocated EFC rows beyond each world's active prefix cannot activate trees."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML, nworld=2)
    del mjm, mjd
    mjwarp.fwd_position(m, d)
    nefc = d.nefc.numpy().copy()
    self.assertTrue(np.all(nefc > 0))
    nefc[1] = 0
    wp.copy(d.nefc, wp.array(nefc, dtype=wp.int32, device=d.nefc.device))

    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.array([[0, 0], [-1, -1]], dtype=np.int32))
    np.testing.assert_array_equal(d.island_parent.numpy(), np.array([[0, 0], [0, 1]], dtype=np.int32))

  def test_site_connect_and_weld_lower_to_body_trees(self):
    """Site-based CONNECT and WELD constraints use their owning body trees."""
    for kind in ("connect", "weld"):
      with self.subTest(kind=kind):
        mjm, mjd, m, d = test_data.fixture(xml=_site_equality_xml(kind), nworld=2)
        del mjm, mjd
        mjwarp.fwd_position(m, d)
        self.assertEqual(m.eq_objtype.numpy()[0], types.ObjType.SITE)

        island.island(m, d)

        np.testing.assert_array_equal(d.nisland.numpy(), np.ones(2, dtype=np.int32))
        np.testing.assert_array_equal(d.tree_island.numpy(), np.zeros((2, 2), dtype=np.int32))

  def test_elliptic_contact_uses_direct_geom_incidence(self):
    """Elliptic contact rows connect the trees owning their two geoms."""
    mjm, mjd, m, d = test_data.fixture(xml=_ELLIPTIC_CONTACT_XML, nworld=2)
    del mjm, mjd
    mjwarp.fwd_position(m, d)
    efc_type = d.efc.type.numpy()
    nefc = d.nefc.numpy()
    self.assertTrue(
      all(np.any(efc_type[world, : nefc[world]] == types.ConstraintType.CONTACT_ELLIPTIC) for world in range(d.nworld))
    )

    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.ones(2, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.zeros((2, 2), dtype=np.int32))

  def test_generic_equality_matches_in_explicit_dense_and_sparse_modes(self):
    """Generic Jacobian incidence is exact for both supported storage modes."""
    xml = """
    <mujoco>
      <worldbody>
        <body><joint name="j0" type="hinge"/><geom size=".1"/></body>
        <body pos="1 0 0"><joint name="j1" type="hinge"/><geom size=".1"/></body>
      </worldbody>
      <equality><joint joint1="j0" joint2="j1"/></equality>
    </mujoco>
    """
    for jacobian in (mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE):
      with self.subTest(jacobian=int(jacobian)):
        mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=2, overrides={"opt.jacobian": jacobian})
        del mjm, mjd
        mjwarp.fwd_position(m, d)
        self.assertEqual(m.is_sparse, jacobian == mujoco.mjtJacobian.mjJAC_SPARSE)

        island.island(m, d)

        np.testing.assert_array_equal(d.nisland.numpy(), np.ones(2, dtype=np.int32))
        np.testing.assert_array_equal(d.tree_island.numpy(), np.zeros((2, 2), dtype=np.int32))

  def test_atomic_discovery_drives_downstream_mapping(self):
    """Atomic discovery outputs feed exact production DOF and EFC mappings."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML, nworld=2)
    mjwarp.fwd_position(m, d)
    island.island(m, d)
    island.compute_island_mapping(m, d)

    for world in range(d.nworld):
      np.testing.assert_array_equal(d.dof_island.numpy()[world, : m.nv], mjd.dof_island[: mjm.nv])
      np.testing.assert_array_equal(d.efc.island.numpy()[world, : mjd.nefc], mjd.efc_island[: mjd.nefc])
      np.testing.assert_array_equal(d.island_nv.numpy()[world, : mjd.nisland], mjd.island_nv[: mjd.nisland])
      np.testing.assert_array_equal(d.island_nefc.numpy()[world, : mjd.nisland], mjd.island_nefc[: mjd.nisland])

  def test_direct_dsu_owns_no_hidden_allocation(self):
    """Production discovery reuses its one persistent parent array and outputs."""
    source = inspect.getsource(island.direct_dsu)
    self.assertNotIn("wp.empty", source)
    self.assertNotIn("wp.zeros", source)
    self.assertIn("dim=(d.nworld, m.ntree)", source)
    self.assertIn("dim=(d.nworld, d.njmax)", source)
    self.assertIn("d.island_parent", source)

  def test_repeated_island_reset_is_bitwise_stable(self):
    """Each call overwrites the persistent DSU workspace and output labels."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    del mjm, mjd
    mjwarp.fwd_position(m, d)

    island.island(m, d)
    first_parent = d.island_parent.numpy().copy()
    first_labels = d.tree_island.numpy().copy()
    first_nisland = d.nisland.numpy().copy()

    for _ in range(16):
      d.island_parent.fill_(123)
      d.tree_island.fill_(123)
      d.nisland.fill_(123)
      island.island(m, d)

      np.testing.assert_array_equal(d.island_parent.numpy(), first_parent)
      np.testing.assert_array_equal(d.tree_island.numpy(), first_labels)
      np.testing.assert_array_equal(d.nisland.numpy(), first_nisland)

  def test_high_contention_chain_is_bitwise_stable(self):
    """Concurrent duplicate weld rows converge to one deterministic minimum root."""
    mjm, mjd, m, d = test_data.fixture(xml=_chain_xml(64), nworld=2)
    del mjm, mjd
    mjwarp.fwd_position(m, d)
    expected_parent = np.zeros((d.nworld, m.ntree), dtype=np.int32)
    expected_labels = np.zeros((d.nworld, m.ntree), dtype=np.int32)

    for _ in range(16):
      island.island(m, d)
      np.testing.assert_array_equal(d.island_parent.numpy(), expected_parent)
      np.testing.assert_array_equal(d.tree_island.numpy(), expected_labels)
      np.testing.assert_array_equal(d.nisland.numpy(), np.ones(d.nworld, dtype=np.int32))

  def test_joint_friction_and_limit_rows_activate_their_dof_tree(self):
    """Special one-tree EFC rows use their prescribed DOF-tree incidence."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option><flag island="disable"/></option>
        <worldbody>
          <body><joint type="hinge" limited="true" range="-1 1" frictionloss="1"/><geom size=".1"/></body>
        </worldbody>
      </mujoco>
      """
    )
    del mjm, mjd
    d.qpos.fill_(2.0)
    mjwarp.fwd_position(m, d)

    efc_types = d.efc.type.numpy()[0, : d.nefc.numpy()[0]]
    self.assertIn(types.ConstraintType.FRICTION_DOF, efc_types)
    self.assertIn(types.ConstraintType.LIMIT_JOINT, efc_types)

    island.island(m, d)
    np.testing.assert_array_equal(d.nisland.numpy(), np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.array([[0]], dtype=np.int32))

  def test_generic_equality_jacobian_unions_all_active_trees(self):
    """Joint equality takes the generic Jacobian path rather than body incidence."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option><flag island="disable"/></option>
        <worldbody>
          <body><joint name="j0" type="hinge"/><geom size=".1"/></body>
          <body pos="1 0 0"><joint name="j1" type="hinge"/><geom size=".1"/></body>
        </worldbody>
        <equality><joint joint1="j0" joint2="j1"/></equality>
      </mujoco>
      """
    )
    del mjm, mjd
    mjwarp.fwd_position(m, d)

    self.assertIn(types.ConstraintType.EQUALITY, d.efc.type.numpy()[0, : d.nefc.numpy()[0]])
    island.island(m, d)
    np.testing.assert_array_equal(d.tree_island.numpy(), np.array([[0, 0]], dtype=np.int32))

  def test_warmed_island_does_not_allocate_or_sync_with_host(self):
    """The persistent DSU path is safe to call inside a warmed graph region."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    del mjm, mjd
    mjwarp.fwd_position(m, d)
    island.island(m, d)  # compile and warm the kernel before installing spies

    with (
      mock.patch.object(wp, "empty", side_effect=AssertionError("wp.empty")),
      mock.patch.object(wp, "zeros", side_effect=AssertionError("wp.zeros")),
      mock.patch.object(wp, "synchronize", side_effect=AssertionError("wp.synchronize")),
      mock.patch.object(wp, "synchronize_device", side_effect=AssertionError("wp.synchronize_device")),
      mock.patch.object(wp.array, "numpy", side_effect=AssertionError("array.numpy")),
    ):
      island.island(m, d)

  @absltest.skipIf(not wp.get_device().is_cuda, "CUDA graph capture requires a CUDA device.")
  def test_capture_replay_matches_direct_island_output(self):
    """Graph replay produces the same labels and DSU workspace as a direct launch."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    del mjm, mjd
    mjwarp.fwd_position(m, d)
    island.island(m, d)
    expected_parent = d.island_parent.numpy().copy()
    expected_labels = d.tree_island.numpy().copy()
    expected_nisland = d.nisland.numpy().copy()

    with wp.ScopedCapture() as capture:
      island.island(m, d)
    wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(d.island_parent.numpy(), expected_parent)
    np.testing.assert_array_equal(d.tree_island.numpy(), expected_labels)
    np.testing.assert_array_equal(d.nisland.numpy(), expected_nisland)

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
  """Tests for island DOF/constraint mapping."""

  def test_two_body_weld_mapping(self):
    """Two free bodies with a weld: 1 island, all DOFs constrained."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    island.compute_island_mapping(m, d)

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
    island.compute_island_mapping(m, d)

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
    island.compute_island_mapping(m, d)

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
    island.compute_island_mapping(m, d)

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
    island.compute_island_mapping(m, d)

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
    island.compute_island_mapping(m, d)

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
      d.island_idofadr.numpy()[0, :nisland],
      mjd.island_idofadr[:nisland],
    )
    np.testing.assert_array_equal(
      d.island_dofadr.numpy()[0, :nisland],
      mjd.island_dofadr[:nisland],
    )
    np.testing.assert_array_equal(
      d.island_iefcadr.numpy()[0, :nisland],
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

  def test_dof_mapping_is_canonical_across_worlds(self):
    """Island-local DOFs retain MuJoCo's ascending global-DOF order."""
    mjm, mjd, m, d = test_data.fixture(xml=_chain_xml(22), nworld=256)
    m.opt.disableflags &= ~types.DisableBit.ISLAND

    mjwarp.fwd_position(m, d)
    island.compute_island_mapping(m, d)

    expected_dof2idof = np.tile(mjd.map_dof2idof[: mjm.nv], (d.nworld, 1))
    expected_idof2dof = np.tile(mjd.map_idof2dof[: mjm.nv], (d.nworld, 1))
    np.testing.assert_array_equal(d.map_dof2idof.numpy()[:, : mjm.nv], expected_dof2idof)
    np.testing.assert_array_equal(d.map_idof2dof.numpy()[:, : mjm.nv], expected_idof2dof)

  def test_efc_mapping_is_canonical_across_worlds(self):
    """Island-local constraints retain MuJoCo's category and EFC order."""
    mjm, mjd, m, d = test_data.fixture(xml=_chain_xml(22), nworld=256)
    m.opt.disableflags &= ~types.DisableBit.ISLAND

    mjwarp.fwd_position(m, d)
    island.compute_island_mapping(m, d)

    expected_efc2iefc = np.tile(mjd.map_efc2iefc[: mjd.nefc], (d.nworld, 1))
    expected_iefc2efc = np.tile(mjd.map_iefc2efc[: mjd.nefc], (d.nworld, 1))
    np.testing.assert_array_equal(d.map_efc2iefc.numpy()[:, : mjd.nefc], expected_efc2iefc)
    np.testing.assert_array_equal(d.map_iefc2efc.numpy()[:, : mjd.nefc], expected_iefc2efc)

  def test_island_ne_nf_parity(self):
    """island_ne and island_nf match MuJoCo C values."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    island.compute_island_mapping(m, d)

    nisland = mjd.nisland

    if nisland > 0:
      np.testing.assert_array_equal(
        d.island_ne.numpy()[0, :nisland],
        mjd.island_ne[:nisland],
      )


if __name__ == "__main__":
  wp.init()
  absltest.main()
