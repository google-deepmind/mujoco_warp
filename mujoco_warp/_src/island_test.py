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

from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import island
from mujoco_warp._src import types


def _discover(m, d):
  """Run discovery through a caller-owned workspace and return it for inspection."""
  island_parent = wp.empty((d.nworld, m.ntree), dtype=int)
  island.direct_dsu(m, d, island_parent)
  return island_parent


def _launch_chunking(nworld: int, njmax: int) -> tuple[int, int]:
  """Report the (chunk count, chunk size) `direct_dsu` would launch for this shape.

  The sizing arithmetic is inline in `direct_dsu`, so this drives the real code and reads
  the grid back off the intercepted launch rather than restating the formula here.
  """
  m, d = mock.MagicMock(), mock.MagicMock()
  d.nworld, d.njmax = nworld, njmax
  with mock.patch.object(wp, "launch"), mock.patch.object(wp, "launch_tiled") as launch_tiled:
    island.direct_dsu(m, d, mock.MagicMock())
  call = launch_tiled.call_args
  return call.kwargs["dim"][1], call.kwargs["inputs"][-3]


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

# One flex equality emitting several rows that share an id but not their tree support.
_FLEX_EQUALITY_XML = """
<mujoco>
  <option jacobian="sparse"><flag contact="disable" gravity="disable"/></option>
  <worldbody>
    <flexcomp name="f" type="grid" dim="1" count="3 1 1"
              spacing=".05 .05 .05" radius=".01" mass="1">
      <edge equality="true"/>
      <contact internal="false" selfcollide="none"/>
    </flexcomp>
  </worldbody>
</mujoco>
"""

# A weld between two static bodies: both endpoints resolve to tree -1.
_STATIC_WELD_XML = """
<mujoco>
  <option><flag contact="disable"/></option>
  <worldbody>
    <body name="s1"><geom size=".05"/></body>
    <body name="s2" pos="1 0 0"><geom size=".05"/></body>
    <body name="dyn" pos="2 0 0"><joint type="slide"/><geom size=".05"/></body>
  </worldbody>
  <equality><weld body1="s1" body2="s2"/></equality>
</mujoco>
"""

_ELLIPTIC_CONTACT_XML = """
<mujoco>
  <option cone="elliptic"/>
  <worldbody>
    <body name="left" pos="-.05 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
    <body name="right" pos=".05 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
  </worldbody>
</mujoco>
"""


# Mixed-constraint pile: static-plane contacts, body-body contacts that merge and split
# as the pile settles, a weld pair, and a hinge tree carrying both a limit and friction.
_MIXED_CONTACT_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body name="p0" pos="0 0 .12"><freejoint/><geom type="sphere" size=".1"/></body>
    <body name="p1" pos=".13 0 .34"><freejoint/><geom type="sphere" size=".1"/></body>
    <body name="p2" pos="-.05 .12 .56"><freejoint/><geom type="sphere" size=".1"/></body>
    <body name="p3" pos="2 0 .12"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    <body name="p4" pos="2 .05 .40"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
    <body name="w0" pos="-2 0 .5"><freejoint/><geom type="sphere" size=".1"/></body>
    <body name="w1" pos="-2 .3 .5"><freejoint/><geom type="sphere" size=".1"/></body>
    <body name="h0" pos="4 0 .5">
      <joint type="hinge" axis="0 1 0" limited="true" range="-.3 .3" frictionloss="1"/>
      <geom type="capsule" size=".05" fromto="0 0 0 .4 0 0"/>
    </body>
  </worldbody>
  <equality><weld body1="w0" body2="w1"/></equality>
</mujoco>
"""


def _limit_chain_xml(trees: int) -> str:
  """One limited slide joint per tree.

  A joint limit emits exactly one scalar EFC row, so each tree's activation rests on a
  single row. Dropping any row anywhere in the prefix deactivates exactly one tree and
  moves nisland, which multi-row constraints like weld would mask.
  """
  bodies = "".join(
    f'<body name="b{i}" pos="{5 * i} 0 0">'
    f'<joint type="slide" axis="1 0 0" limited="true" range="-1 1"/><geom size=".1"/></body>'
    for i in range(trees)
  )
  return f'<mujoco><option><flag contact="disable"/></option><worldbody>{bodies}</worldbody></mujoco>'


def _disjoint_contact_pairs_xml(pairs: int) -> str:
  """Overlapping sphere pairs, far enough apart that each pair is its own island."""
  bodies = "".join(
    f'<body name="l{i}" pos="{10 * i} 0 1"><freejoint/><geom type="sphere" size=".1"/></body>'
    f'<body name="r{i}" pos="{10 * i + 0.15} 0 1"><freejoint/><geom type="sphere" size=".1"/></body>'
    for i in range(pairs)
  )
  return f'<mujoco><option gravity="0 0 0"/><worldbody>{bodies}</worldbody></mujoco>'


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


def _chain_xml(count: int, freejoint: bool = True) -> str:
  """Welded chain of `count` trees. Slide joints keep `nv` linear for the large sizes."""
  joint = "<freejoint/>" if freejoint else '<joint type="slide" axis="0 0 1"/>'
  bodies = "".join(f'<body name="b{i}" pos="{i * 0.3} 0 0">{joint}<geom size=".1"/></body>' for i in range(count))
  welds = "".join(f'<weld body1="b{i}" body2="b{i + 1}"/>' for i in range(count - 1))
  option = "" if freejoint else '<option jacobian="sparse"><flag contact="disable"/></option>'
  return f"<mujoco>{option}<worldbody>{bodies}</worldbody><equality>{welds}</equality></mujoco>"


def _one_equality_xml(trees: int) -> str:
  """Many single-DOF trees, exactly one of which is constrained, leaving the rest inactive."""
  bodies = "".join(
    f'<body name="b{i}" pos="{i} 0 0"><joint type="slide" axis="0 0 1"/><geom size=".05"/></body>' for i in range(trees)
  )
  return (
    f'<mujoco><option jacobian="sparse"><flag contact="disable"/></option><worldbody>{bodies}</worldbody>'
    f'<equality><weld body1="b0" body2="b1"/></equality></mujoco>'
  )


def _random_edge_pool_xml(trees: int, edges: int, seed: int = 7) -> str:
  """Sparse fixed pool of weld equalities, plus static ones, for the randomized differential.

  The pool is sparse relative to `trees` so random subsets of it produce fragmented partitions
  rather than collapsing to a single island every trial.
  """
  rng = np.random.default_rng(seed)
  bodies = "".join(
    f'<body name="b{i}" pos="{i} 0 0"><joint type="slide" axis="0 0 1"/><geom size=".05"/></body>' for i in range(trees)
  )
  pairs = set()
  while len(pairs) < edges:
    lo, hi = sorted(rng.integers(0, trees, size=2))
    if lo != hi:
      pairs.add((int(lo), int(hi)))
  equalities = "".join(f'<weld body1="b{lo}" body2="b{hi}"/>' for lo, hi in sorted(pairs))
  equalities += "".join(f'<weld body1="world" body2="b{i}"/>' for i in range(0, trees, max(1, trees // 4)))
  return (
    f'<mujoco><option jacobian="sparse"><flag contact="disable"/></option>'
    f"<worldbody>{bodies}</worldbody><equality>{equalities}</equality></mujoco>"
  )


def _reference_islands(edges, ntree: int):
  """Label connected components by BFS from the lowest unvisited active tree.

  Deliberately independent of the disjoint-set implementation under test: adjacency lists plus a
  stack, matching the reference traversal in MuJoCo's randomized differential test.
  """
  active = np.zeros(ntree, dtype=bool)
  adjacent = [[] for _ in range(ntree)]
  for tree0, tree1 in edges:
    if tree0 >= 0:
      active[tree0] = True
    if tree1 >= 0:
      active[tree1] = True
    if tree0 >= 0 and tree1 >= 0:
      adjacent[tree0].append(tree1)
      adjacent[tree1].append(tree0)

  labels = np.full(ntree, -1, dtype=np.int32)
  nisland = 0
  for start in range(ntree):
    if not active[start] or labels[start] != -1:
      continue
    labels[start] = nisland
    pending = [start]
    while pending:
      tree = pending.pop()
      for neighbor in adjacent[tree]:
        if labels[neighbor] == -1:
          labels[neighbor] = nisland
          pending.append(neighbor)
    nisland += 1
  return labels, nisland


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

    parent = _discover(m, d).numpy()
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

    mjwarp.fwd_position(m, d)
    workspace = _discover(m, d)
    # the disjoint-set workspace stays linear in ntree
    self.assertEqual(workspace.shape, (d.nworld, m.ntree))
    self.assertEqual(workspace.dtype, wp.int32)
    self.assertEqual(workspace.capacity, d.nworld * m.ntree * 4)
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

    parent = _discover(m, d).numpy()

    np.testing.assert_array_equal(d.nisland.numpy(), np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.array([[0, 0], [-1, -1]], dtype=np.int32))
    np.testing.assert_array_equal(parent, np.array([[0, 0], [0, 1]], dtype=np.int32))

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

  def test_dsu_chunking_splits_only_when_the_batch_leaves_the_device_idle(self):
    """Chunk count trades against nworld, so a large batch keeps one block per world."""
    # a batch that already fills the device is left alone: splitting it only launches
    # blocks past the active prefix
    self.assertEqual(_launch_chunking(nworld=2048, njmax=1024), (1, 1024))
    self.assertEqual(_launch_chunking(nworld=4096, njmax=1024), (1, 1024))

    # a single world with a long prefix is split until the device has work
    nchunk, chunk_size = _launch_chunking(nworld=1, njmax=12352)
    self.assertGreater(nchunk, 8)
    self.assertGreaterEqual(chunk_size, 32)

    # every input keeps full coverage of the prefix and at least one chunk
    for nworld in (1, 7, 64, 512, 2048):
      for njmax in (1, 31, 64, 255, 256, 1024, 3136, 12352, 65536):
        nchunk, chunk_size = _launch_chunking(nworld=nworld, njmax=njmax)
        self.assertGreaterEqual(nchunk, 1, f"{nworld=} {njmax=}")
        self.assertGreaterEqual(chunk_size, 1, f"{nworld=} {njmax=}")
        self.assertGreaterEqual(nchunk * chunk_size, njmax, f"{nworld=} {njmax=}")

  def test_flex_equality_rows_are_rescanned_not_quotiented(self):
    """Rows sharing one flex equality id carry different tree incidence, so none may be dropped.

    Mirrors MuJoCo's `ProductionFlexEqualityRescansRows`. The repeated-row shortcut keys on
    `(efc_type, efc_id)`, which is equal across these rows even though their support differs;
    quotienting them would leave the last tree unactivated.
    """
    mjm, mjd, m, d = test_data.fixture(xml=_FLEX_EQUALITY_XML, nworld=2)
    mjwarp.fwd_position(m, d)

    nefc = int(d.nefc.numpy()[0])
    efc_id = d.efc.id.numpy()[0, :nefc]
    # the rows this test exists for: same id, different support
    self.assertEqual(nefc, 2)
    np.testing.assert_array_equal(efc_id, np.zeros(2, dtype=np.int32))
    self.assertEqual(mjm.eq_type[0], types.EqType.FLEX)
    row_trees = []
    for row in range(nefc):
      adr, nnz = d.efc.J_rowadr.numpy()[0, row], d.efc.J_rownnz.numpy()[0, row]
      dofs = d.efc.J_colind.numpy()[0, 0, adr : adr + nnz]
      row_trees.append(sorted(set(int(t) for t in m.dof_treeid.numpy()[dofs])))
    self.assertEqual(row_trees, [[0, 1], [1, 2]])

    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.full(d.nworld, mjd.nisland, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.tile(mjd.tree_island[: mjm.ntree], (d.nworld, 1)))

  def test_duplicate_fixed_support_rows_do_not_change_the_partition(self):
    """Repeated scalar rows with equal `(efc_type, efc_id)` collapse to one representative."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML, nworld=2)
    del mjm, mjd
    mjwarp.fwd_position(m, d)
    island.island(m, d)
    expected_labels = d.tree_island.numpy().copy()
    expected_nisland = d.nisland.numpy().copy()

    # duplicate every active row in place; the H0 partition is unchanged by duplicate incidence
    efc_type, efc_id, nefc = d.efc.type.numpy(), d.efc.id.numpy(), d.nefc.numpy()
    for world in range(d.nworld):
      n = nefc[world]
      self.assertLessEqual(2 * n, d.njmax)
      efc_type[world, n : 2 * n] = efc_type[world, :n]
      efc_id[world, n : 2 * n] = efc_id[world, :n]
      nefc[world] = 2 * n
    wp.copy(d.efc.type, wp.array(efc_type, dtype=wp.int32, device=d.efc.type.device))
    wp.copy(d.efc.id, wp.array(efc_id, dtype=wp.int32, device=d.efc.id.device))
    wp.copy(d.nefc, wp.array(nefc, dtype=wp.int32, device=d.nefc.device))

    island.island(m, d)

    np.testing.assert_array_equal(d.tree_island.numpy(), expected_labels)
    np.testing.assert_array_equal(d.nisland.numpy(), expected_nisland)

  def test_static_static_constraint_creates_no_island(self):
    """A constraint between two static bodies activates nothing, as in MuJoCo.

    Covers the incidence that MuJoCo's `DsuMergeRejectsStaticSelfIncidence` and
    `ReportsConstraintBetweenTwoStaticBodies` guard. `_dsu_union` returns without touching the
    workspace when both endpoints are static, so the partition is empty rather than diagnosed.
    `tree_island` is only compared against its own expected value: with `nisland == 0` MuJoCo
    leaves its copy unwritten, so cross-checking that array here would read undefined memory.
    """
    mjm, mjd, m, d = test_data.fixture(xml=_STATIC_WELD_XML, nworld=2)
    mjwarp.fwd_position(m, d)
    self.assertGreater(int(d.nefc.numpy()[0]), 0)
    np.testing.assert_array_equal(m.body_treeid.numpy()[[1, 2]], np.full(2, -1, dtype=np.int32))

    island.island(m, d)

    self.assertEqual(mjd.nisland, 0)
    np.testing.assert_array_equal(d.nisland.numpy(), np.zeros(d.nworld, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.full((d.nworld, mjm.ntree), -1, dtype=np.int32))

  def test_randomized_incidence_matches_graph_traversal(self):
    """Differential against an independent traversal over randomized incidence.

    Mirrors MuJoCo's `DsuRandomizedDifferentialAgainstGraphTraversal`. Each world carries an
    independent random multiset of equality rows drawn from a sparse fixed edge pool, including
    static endpoints, duplicates and reversed pairs, so one launch covers `nworld` trials.
    """
    ntree, nedge, nworld, rounds = 24, 18, 16, 16
    mjm, mjd, m, d = test_data.fixture(xml=_random_edge_pool_xml(ntree, nedge), nworld=nworld, njmax=1024)
    del mjd
    body_treeid = m.body_treeid.numpy()
    eq_obj1, eq_obj2 = m.eq_obj1id.numpy(), m.eq_obj2id.numpy()
    edge_trees = [(int(body_treeid[eq_obj1[e]]), int(body_treeid[eq_obj2[e]])) for e in range(mjm.neq)]
    self.assertTrue(any(a < 0 or b < 0 for a, b in edge_trees), "pool must contain static endpoints")

    rng = np.random.default_rng(0)
    efc_type, efc_id, nefc = d.efc.type.numpy(), d.efc.id.numpy(), d.nefc.numpy()
    seen_nisland = set()
    for round_ in range(rounds):
      efc_type[:], efc_id[:] = 0, 0
      trials = []
      for world in range(nworld):
        rows = rng.integers(0, mjm.neq, size=int(rng.integers(0, 16)))
        efc_type[world, : rows.size] = int(types.ConstraintType.EQUALITY)
        efc_id[world, : rows.size] = rows
        nefc[world] = rows.size
        trials.append([edge_trees[e] for e in rows])
      wp.copy(d.efc.type, wp.array(efc_type, dtype=wp.int32, device=d.efc.type.device))
      wp.copy(d.efc.id, wp.array(efc_id, dtype=wp.int32, device=d.efc.id.device))
      wp.copy(d.nefc, wp.array(nefc, dtype=wp.int32, device=d.nefc.device))

      island.island(m, d)

      labels, nisland = d.tree_island.numpy(), d.nisland.numpy()
      for world in range(nworld):
        want_labels, want_nisland = _reference_islands(trials[world], ntree)
        seen_nisland.add(want_nisland)
        msg = f"round {round_} world {world}"
        self.assertEqual(int(nisland[world]), want_nisland, msg)
        np.testing.assert_array_equal(labels[world], want_labels, err_msg=msg)

    # a pool that only ever produced one island would not discriminate
    self.assertGreaterEqual(len(seen_nisland), 5)

  def test_deep_connected_chain_compresses_to_one_island(self):
    """A chain spanning every tree collapses to a single island.

    Mirrors MuJoCo's `DsuAssignCompresses4096NodeAdversarialChain` and
    `DsuHandlesLongConnectedBoundaryCase` at a size that keeps the fixture cheap.
    """
    trees = 1024
    mjm, mjd, m, d = test_data.fixture(xml=_chain_xml(trees, freejoint=False), nworld=2, njmax=6 * trees + 64)
    del mjm, mjd
    mjwarp.fwd_position(m, d)

    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.ones(d.nworld, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.zeros((d.nworld, trees), dtype=np.int32))

  def test_one_equality_among_many_trees_leaves_the_rest_inactive(self):
    """Only the two constrained trees join an island; every other tree stays unlabeled.

    Mirrors MuJoCo's `BoundedArenaSupports1024Trees`.
    """
    trees = 1024
    mjm, mjd, m, d = test_data.fixture(xml=_one_equality_xml(trees), nworld=2, njmax=256)
    del mjm, mjd
    mjwarp.fwd_position(m, d)

    island.island(m, d)

    expected = np.full((d.nworld, trees), -1, dtype=np.int32)
    expected[:, 0] = 0
    expected[:, 1] = 0
    np.testing.assert_array_equal(d.nisland.numpy(), np.ones(d.nworld, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), expected)

  def test_repeated_island_reset_is_bitwise_stable(self):
    """Each call overwrites the persistent DSU workspace and output labels."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    del mjm, mjd
    mjwarp.fwd_position(m, d)

    first_parent = _discover(m, d).numpy().copy()
    first_labels = d.tree_island.numpy().copy()
    first_nisland = d.nisland.numpy().copy()

    for _ in range(16):
      d.tree_island.fill_(123)
      d.nisland.fill_(123)

      np.testing.assert_array_equal(_discover(m, d).numpy(), first_parent)
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
      np.testing.assert_array_equal(_discover(m, d).numpy(), expected_parent)
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

  def test_warmed_island_does_not_sync_with_host(self):
    """The DSU path is safe to call inside a warmed graph region.

    `island` constructs its own workspace, so `wp.empty` is expected here. What must not
    happen is a host round trip, which would break graph capture.
    """
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    del mjm, mjd
    mjwarp.fwd_position(m, d)
    island.island(m, d)  # compile and warm the kernel before installing spies

    with (
      mock.patch.object(wp, "synchronize", side_effect=AssertionError("wp.synchronize")),
      mock.patch.object(wp, "synchronize_device", side_effect=AssertionError("wp.synchronize_device")),
      mock.patch.object(wp.array, "numpy", side_effect=AssertionError("array.numpy")),
    ):
      island.island(m, d)

    # the workspace is the only allocation, and discovery itself makes none
    workspace = wp.empty((d.nworld, m.ntree), dtype=int)
    with (
      mock.patch.object(wp, "empty", side_effect=AssertionError("wp.empty")),
      mock.patch.object(wp, "zeros", side_effect=AssertionError("wp.zeros")),
    ):
      island.direct_dsu(m, d, workspace)

  def test_discovery_covers_efc_rows_across_chunk_boundaries(self):
    """Every row of an EFC prefix longer than one chunk still activates its tree."""
    trees = 900
    # the shared fixture defaults njmax to 64, which would truncate the prefix this test
    # exists to cross, so Data is built here with capacity for every row
    mjm = mujoco.MjModel.from_xml_string(_limit_chain_xml(trees))
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = 2.0
    mujoco.mj_forward(mjm, mjd)
    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd, nworld=2, njmax=trees + 64, nconmax=1, nccdmax=1)
    mjwarp.fwd_position(m, d)

    nefc = d.nefc.numpy()
    nchunk, chunk_size = _launch_chunking(d.nworld, d.njmax)
    self.assertGreater(nchunk, 1, "fixture must be split across blocks")
    self.assertGreater(int(nefc.min()), chunk_size, "active prefix must cross a chunk boundary")
    self.assertLessEqual(int(nefc.max()), d.njmax, "prefix must fit njmax or rows are truncated")

    island.island(m, d)
    dsu_labels = d.tree_island.numpy().copy()
    dsu_nisland = d.nisland.numpy().copy()

    # one row activates one tree, so a row dropped anywhere in the prefix moves the count
    np.testing.assert_array_equal(dsu_nisland, np.full(d.nworld, trees, dtype=np.int32))

    # every tree is its own island, so labels must be the identity over active trees
    np.testing.assert_array_equal(dsu_labels, np.tile(np.arange(trees, dtype=np.int32), (d.nworld, 1)))

  def test_labels_are_invariant_to_efc_row_permutation(self):
    """The H0 partition is a function of incidence, not of the order rows arrive in."""
    # Consecutive rows share a constraint type but belong to different islands, so any
    # order-sensitive shortcut merges or drops the wrong pair once the rows move.
    mjm, mjd, m, d = test_data.fixture(xml=_disjoint_contact_pairs_xml(8), nworld=4)
    del mjm, mjd
    mjwarp.fwd_position(m, d)

    island.island(m, d)
    expected_labels = d.tree_island.numpy().copy()
    expected_nisland = d.nisland.numpy().copy()
    np.testing.assert_array_equal(expected_nisland, np.full(d.nworld, 8, dtype=np.int32))

    efc_type = d.efc.type.numpy()
    efc_id = d.efc.id.numpy()
    nefc = d.nefc.numpy()
    rng = np.random.default_rng(0)

    for trial in range(8):
      permuted_type, permuted_id = efc_type.copy(), efc_id.copy()
      for world in range(d.nworld):
        order = rng.permutation(nefc[world])
        permuted_type[world, : nefc[world]] = efc_type[world, order]
        permuted_id[world, : nefc[world]] = efc_id[world, order]
      wp.copy(d.efc.type, wp.array(permuted_type, dtype=wp.int32, device=d.efc.type.device))
      wp.copy(d.efc.id, wp.array(permuted_id, dtype=wp.int32, device=d.efc.id.device))

      island.island(m, d)

      np.testing.assert_array_equal(d.tree_island.numpy(), expected_labels, err_msg=f"trial {trial}")
      np.testing.assert_array_equal(d.nisland.numpy(), expected_nisland, err_msg=f"trial {trial}")

  @absltest.skipIf(not wp.get_device().is_cuda, "CUDA graph capture requires a CUDA device.")
  def test_capture_replay_matches_direct_island_output(self):
    """Graph replay produces the same labels and DSU workspace as a direct launch."""
    mjm, mjd, m, d = test_data.fixture(xml=_WELD_XML)
    del mjm, mjd
    mjwarp.fwd_position(m, d)
    workspace = wp.empty((d.nworld, m.ntree), dtype=int)
    island.direct_dsu(m, d, workspace)
    expected_parent = workspace.numpy().copy()
    expected_labels = d.tree_island.numpy().copy()
    expected_nisland = d.nisland.numpy().copy()

    with wp.ScopedCapture() as capture:
      island.direct_dsu(m, d, workspace)
    wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(workspace.numpy(), expected_parent)
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

  def test_dof_mapping_interleaves_islands_and_unconstrained_trees(self):
    """Disjoint islands separated by unconstrained trees keep MuJoCo's DOF order."""
    bodies = "".join(f'<body name="b{i}" pos="{i} 0 0"><freejoint/><geom size=".1"/></body>' for i in range(6))
    mjm, mjd, m, d = test_data.fixture(
      xml=f"<mujoco><worldbody>{bodies}</worldbody>"
      '<equality><weld body1="b0" body2="b2"/><weld body1="b3" body2="b5"/></equality></mujoco>',
      nworld=64,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND

    mjwarp.fwd_position(m, d)
    island.compute_island_mapping(m, d)

    # b1 and b4 stay outside every island, so the constrained and unconstrained
    # DOF ranges interleave rather than splitting at one boundary.
    self.assertEqual(mjd.nisland, 2)
    self.assertTrue(np.any(mjd.dof_island[: mjm.nv] < 0))

    np.testing.assert_array_equal(d.map_dof2idof.numpy()[:, : mjm.nv], np.tile(mjd.map_dof2idof[: mjm.nv], (d.nworld, 1)))
    np.testing.assert_array_equal(d.map_idof2dof.numpy()[:, : mjm.nv], np.tile(mjd.map_idof2dof[: mjm.nv], (d.nworld, 1)))
    np.testing.assert_array_equal(
      d.island_dofadr.numpy()[:, : mjd.nisland], np.tile(mjd.island_dofadr[: mjd.nisland], (d.nworld, 1))
    )

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
