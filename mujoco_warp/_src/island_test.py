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

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import island
from mujoco_warp._src import types


class IslandDiscoveryTopologyTest(absltest.TestCase):
  """Tests island discovery across fundamental graph topologies."""

  def test_connected_chain_collapses_to_single_island(self):
    """A welded chain of trees compresses to a single island rooted at tree 0."""
    count = 8
    bodies = "".join(f'<body name="b{i}" pos="{i * 0.3} 0 0"><freejoint/><geom size=".1"/></body>' for i in range(count))
    welds = "".join(f'<weld body1="b{i}" body2="b{i + 1}"/>' for i in range(count - 1))

    mjm, mjd, m, d = test_data.fixture(
      xml=f"<mujoco><worldbody>{bodies}</worldbody><equality>{welds}</equality></mujoco>",
      nworld=2,
    )
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.full(d.nworld, mjd.nisland, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.tile(mjd.tree_island[: mjm.ntree], (d.nworld, 1)))

  def test_disconnected_pairs_form_independent_islands(self):
    """Disconnected pairs form separate islands with canonical ascending IDs."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="a1"><freejoint/><geom size=".1"/></body>
          <body name="a2" pos="1 0 0"><freejoint/><geom size=".1"/></body>
          <body name="b1" pos="5 0 0"><freejoint/><geom size=".1"/></body>
          <body name="b2" pos="6 0 0"><freejoint/><geom size=".1"/></body>
        </worldbody>
        <equality>
          <weld body1="a1" body2="a2"/>
          <weld body1="b1" body2="b2"/>
        </equality>
      </mujoco>
      """,
      nworld=2,
    )
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.full(d.nworld, mjd.nisland, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.tile(mjd.tree_island[: mjm.ntree], (d.nworld, 1)))

  def test_unconstrained_trees_remain_unlabeled(self):
    """Unconstrained trees remain labeled -1 and do not create islands."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="c1"><freejoint/><geom size=".1"/></body>
          <body name="c2" pos="1 0 0"><freejoint/><geom size=".1"/></body>
          <body name="free1" pos="3 0 0"><freejoint/><geom size=".1"/></body>
          <body name="free2" pos="4 0 0"><freejoint/><geom size=".1"/></body>
        </worldbody>
        <equality><weld body1="c1" body2="c2"/></equality>
      </mujoco>
      """,
      nworld=2,
    )
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.full(d.nworld, mjd.nisland, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.tile(mjd.tree_island[: mjm.ntree], (d.nworld, 1)))

  def test_worldbody_constraint_activates_single_tree(self):
    """A constraint between worldbody (tree < 0) and a body activates only that tree."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="floating"><freejoint/><geom size=".1"/></body>
        </worldbody>
        <equality><weld body1="world" body2="floating"/></equality>
      </mujoco>
      """,
      nworld=2,
    )
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.full(d.nworld, mjd.nisland, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.tile(mjd.tree_island[: mjm.ntree], (d.nworld, 1)))

  def test_no_constraints_no_islands(self):
    """Free bodies with no constraints produce zero islands."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body><freejoint/><geom size=".1"/></body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
    )
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.zeros(d.nworld, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.full((d.nworld, mjm.ntree), -1, dtype=np.int32))


class IslandDiscoveryConstraintsTest(absltest.TestCase):
  """Tests island discovery across all MuJoCo constraint types."""

  def test_site_connect_and_weld(self):
    """Site-based connect and weld equalities resolve to owning body trees."""
    for kind in ("connect", "weld"):
      with self.subTest(kind=kind):
        mjm, mjd, m, d = test_data.fixture(
          xml=f"""
          <mujoco>
            <worldbody>
              <body name="l" pos="-1 0 0"><freejoint/><geom size=".1"/><site name="s1"/></body>
              <body name="r" pos="1 0 0"><freejoint/><geom size=".1"/><site name="s2"/></body>
            </worldbody>
            <equality><{kind} site1="s1" site2="s2"/></equality>
          </mujoco>
          """,
          nworld=2,
        )
        mjwarp.fwd_position(m, d)

        d.nisland.fill_(-1)
        d.tree_island.fill_(-1)
        island.island(m, d)

        np.testing.assert_array_equal(d.nisland.numpy(), np.full(d.nworld, mjd.nisland, dtype=np.int32))
        np.testing.assert_array_equal(d.tree_island.numpy(), np.tile(mjd.tree_island[: mjm.ntree], (d.nworld, 1)))

  def test_generic_equality_dense_and_sparse(self):
    """Joint equalities take the generic Jacobian scan in both dense and sparse modes."""
    for jac in (mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE):
      with self.subTest(jacobian=int(jac)):
        mjm, mjd, m, d = test_data.fixture(
          xml="""
          <mujoco>
            <worldbody>
              <body><joint name="j0" type="hinge"/><geom size=".1"/></body>
              <body pos="1 0 0"><joint name="j1" type="hinge"/><geom size=".1"/></body>
            </worldbody>
            <equality><joint joint1="j0" joint2="j1"/></equality>
          </mujoco>
          """,
          nworld=2,
          overrides={"opt.jacobian": jac},
        )
        mjwarp.fwd_position(m, d)

        d.nisland.fill_(-1)
        d.tree_island.fill_(-1)
        island.island(m, d)

        np.testing.assert_array_equal(d.nisland.numpy(), np.full(d.nworld, mjd.nisland, dtype=np.int32))
        np.testing.assert_array_equal(d.tree_island.numpy(), np.tile(mjd.tree_island[: mjm.ntree], (d.nworld, 1)))

  def test_flex_equality_rescans_all_rows(self):
    """Flex equality rows sharing an ID carry differing tree incidence and are not quotiented."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option jacobian="sparse"><flag contact="disable" gravity="disable"/></option>
        <worldbody>
          <flexcomp name="f" type="grid" dim="1" count="3 1 1" spacing=".05 .05 .05" radius=".01" mass="1">
            <edge equality="true"/>
            <contact internal="false" selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
    )
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.full(d.nworld, mjd.nisland, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.tile(mjd.tree_island[: mjm.ntree], (d.nworld, 1)))

  def test_joint_friction_and_limits(self):
    """Joint limits and frictionloss activate their owning DOF tree."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body><joint type="hinge" limited="true" range="-1 1" frictionloss="1"/><geom size=".1"/></body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
    )
    d.qpos.fill_(2.0)
    mjd.qpos[:] = 2.0
    mujoco.mj_forward(mjm, mjd)
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.full(d.nworld, mjd.nisland, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.tile(mjd.tree_island[: mjm.ntree], (d.nworld, 1)))

  def test_contacts_pyramidal_and_elliptic(self):
    """Colliding geoms activate both trees under pyramidal and elliptic cones."""
    for cone in ("pyramidal", "elliptic"):
      with self.subTest(cone=cone):
        mjm, mjd, m, d = test_data.fixture(
          xml=f"""
          <mujoco>
            <option cone="{cone}"/>
            <worldbody>
              <body pos="-.05 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
              <body pos=".05 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
            </worldbody>
          </mujoco>
          """,
          nworld=2,
        )
        mjwarp.fwd_position(m, d)

        d.nisland.fill_(-1)
        d.tree_island.fill_(-1)
        island.island(m, d)

        np.testing.assert_array_equal(d.nisland.numpy(), np.full(d.nworld, mjd.nisland, dtype=np.int32))
        np.testing.assert_array_equal(d.tree_island.numpy(), np.tile(mjd.tree_island[: mjm.ntree], (d.nworld, 1)))

  def test_static_static_constraint_creates_no_island(self):
    """Constraints between two static bodies create no island, matching MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option><flag contact="disable"/></option>
        <worldbody>
          <body name="s1"><geom size=".05"/></body>
          <body name="s2" pos="1 0 0"><geom size=".05"/></body>
        </worldbody>
        <equality><weld body1="s1" body2="s2"/></equality>
      </mujoco>
      """,
      nworld=2,
    )
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)

    self.assertEqual(mjd.nisland, 0)
    np.testing.assert_array_equal(d.nisland.numpy(), np.zeros(d.nworld, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy(), np.full((d.nworld, mjm.ntree), -1, dtype=np.int32))


class IslandDiscoveryExecutionTest(absltest.TestCase):
  """Tests multi-world execution, idempotency, and CUDA graph capturability."""

  def test_inactive_world_prefix_ignored(self):
    """A world with nefc=0 leaves all trees inactive even if njmax > 0."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="b1"><freejoint/><geom size=".1"/></body>
          <body name="b2" pos="1 0 0"><freejoint/><geom size=".1"/></body>
        </worldbody>
        <equality><weld body1="b1" body2="b2"/></equality>
      </mujoco>
      """,
      nworld=2,
    )
    mjwarp.fwd_position(m, d)

    nefc = d.nefc.numpy()
    nefc[1] = 0
    wp.copy(d.nefc, wp.array(nefc, dtype=wp.int32, device=d.nefc.device))

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)

    np.testing.assert_array_equal(d.nisland.numpy(), np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy()[0], np.zeros(mjm.ntree, dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy()[1], np.full(mjm.ntree, -1, dtype=np.int32))

  def test_repeated_discovery_is_idempotent(self):
    """Poisoning output arrays before each run reproduces bitwise-identical results."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="b1"><freejoint/><geom size=".1"/></body>
          <body name="b2" pos="1 0 0"><freejoint/><geom size=".1"/></body>
        </worldbody>
        <equality><weld body1="b1" body2="b2"/></equality>
      </mujoco>
      """,
      nworld=2,
    )
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)
    expected_labels = d.tree_island.numpy().copy()
    expected_nisland = d.nisland.numpy().copy()

    for _ in range(3):
      d.tree_island.fill_(-1)
      d.nisland.fill_(-1)
      island.island(m, d)
      np.testing.assert_array_equal(d.tree_island.numpy(), expected_labels)
      np.testing.assert_array_equal(d.nisland.numpy(), expected_nisland)

  @absltest.skipIf(not wp.get_device().is_cuda, "CUDA graph capture requires a CUDA device.")
  def test_cuda_graph_capture_and_replay(self):
    """Direct DSU is capturable in CUDA graphs with zero host synchronizations."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="b1"><freejoint/><geom size=".1"/></body>
          <body name="b2" pos="1 0 0"><freejoint/><geom size=".1"/></body>
        </worldbody>
        <equality><weld body1="b1" body2="b2"/></equality>
      </mujoco>
      """,
      nworld=2,
    )
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    island.island(m, d)
    expected_labels = d.tree_island.numpy().copy()
    expected_nisland = d.nisland.numpy().copy()

    d.tree_island.fill_(-1)
    d.nisland.fill_(-1)
    workspace = wp.empty((d.nworld, m.ntree), dtype=int)
    with wp.ScopedCapture() as capture:
      island.direct_dsu(m, d, workspace)
    wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(d.tree_island.numpy(), expected_labels)
    np.testing.assert_array_equal(d.nisland.numpy(), expected_nisland)


class IslandMappingTest(absltest.TestCase):
  """Tests downstream DOF and constraint mapping parity against MuJoCo C."""

  def test_mapping_parity_with_interleaved_unconstrained_trees(self):
    """Disjoint islands separated by unconstrained trees match MuJoCo mapping parity."""
    bodies = "".join(f'<body name="b{i}" pos="{i} 0 0"><freejoint/><geom size=".1"/></body>' for i in range(6))
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <worldbody>{bodies}</worldbody>
        <equality>
          <weld body1="b0" body2="b2"/>
          <weld body1="b3" body2="b5"/>
        </equality>
      </mujoco>
      """,
      nworld=4,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjwarp.fwd_position(m, d)

    # Poison all destination fields before computing mapping
    d.dof_island.fill_(-1)
    d.efc.island.fill_(-1)
    d.island_nv.fill_(-1)
    d.island_nefc.fill_(-1)
    d.island_idofadr.fill_(-1)
    d.island_dofadr.fill_(-1)
    d.island_iefcadr.fill_(-1)
    d.map_dof2idof.fill_(-1)
    d.map_idof2dof.fill_(-1)
    d.map_efc2iefc.fill_(-1)
    d.map_iefc2efc.fill_(-1)

    island.compute_island_mapping(m, d)

    for name, got, want in (
      ("dof_island", d.dof_island.numpy()[:, : mjm.nv], mjd.dof_island[: mjm.nv]),
      ("map_dof2idof", d.map_dof2idof.numpy()[:, : mjm.nv], mjd.map_dof2idof[: mjm.nv]),
      ("map_idof2dof", d.map_idof2dof.numpy()[:, : mjm.nv], mjd.map_idof2dof[: mjm.nv]),
      ("island_dofadr", d.island_dofadr.numpy()[:, : mjd.nisland], mjd.island_dofadr[: mjd.nisland]),
      ("efc_island", d.efc.island.numpy()[:, : mjd.nefc], mjd.efc_island[: mjd.nefc]),
      ("map_efc2iefc", d.map_efc2iefc.numpy()[:, : mjd.nefc], mjd.map_efc2iefc[: mjd.nefc]),
      ("map_iefc2efc", d.map_iefc2efc.numpy()[:, : mjd.nefc], mjd.map_iefc2efc[: mjd.nefc]),
    ):
      np.testing.assert_array_equal(got, np.tile(want, (d.nworld, 1)), err_msg=name)


if __name__ == "__main__":
  wp.init()
  absltest.main()
