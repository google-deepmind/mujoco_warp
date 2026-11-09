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


class IslandDiscoveryConstraintsTest(parameterized.TestCase):
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
            <contact selfcollide="none"/>
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

  def test_flex_contact_with_zero_row(self):
    """Flex contacts whose Jacobian rows vanish are in the island of their bodies."""
    models = {
      # cloth pinned to a hinged body: pins on the hinge axis touch the plane
      "pinned_cloth": """
        <mujoco>
          <worldbody>
            <geom type="plane" size="1 1 .1"/>
            <body name="flap">
              <joint axis="0 1 0"/>
              <geom size=".01" pos=".3 0 .1"/>
              <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" mass=".1">
                <pin id="3 4 5"/>
                <edge equality="true"/>
              </flexcomp>
            </body>
          </worldbody>
        </mujoco>
      """,
      # triangle with vertices in three trees, hinges orthogonal to the box
      "hinged_triangle": """
        <mujoco>
          <option cone="elliptic"/>
          <worldbody>
            <geom type="box" size=".2 .2 .05" pos="0 0 -.05"/>
            <body name="b0" pos=".1 0 0">
              <joint/>
              <geom size=".01" pos=".02 0 .1"/>
            </body>
            <body name="b1" pos="-.05 .1 0">
              <joint/>
              <geom size=".01" pos=".02 0 .1"/>
            </body>
            <body name="b2" pos="-.05 -.1 0">
              <joint/>
              <geom size=".01" pos=".02 0 .1"/>
            </body>
          </worldbody>
          <deformable>
            <flex name="triangle" dim="2" body="b0 b1 b2" vertex="0 0 0 0 0 0 0 0 0" element="0 1 2" radius=".005">
              <edge damping="1"/>
            </flex>
          </deformable>
        </mujoco>
      """,
    }
    for name, xml in models.items():
      for jac in (mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE):
        with self.subTest(model=name, jacobian=int(jac)):
          mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.jacobian": jac})
          self.assertGreater(mjd.ncon, 0)
          mjwarp.fwd_position(m, d)

          d.nisland.fill_(-1)
          d.tree_island.fill_(-1)
          island.island(m, d)

          # same islands as MuJoCo: one island with every tree
          self.assertEqual(mjd.nisland, 1)
          np.testing.assert_array_equal(d.nisland.numpy(), [mjd.nisland])
          np.testing.assert_array_equal(d.tree_island.numpy()[0], mjd.tree_island[: mjm.ntree])

          # every constraint row, including the vanishing ones, is in the island
          d.efc.island.fill_(-1)
          island.compute_island_mapping(m, d)
          nefc = d.nefc.numpy()[0]
          self.assertGreater(nefc, 0)
          np.testing.assert_array_equal(d.efc.island.numpy()[0, :nefc], np.zeros(nefc, dtype=np.int32))

  @parameterized.parameters(
    """
    <tendon>
      <spatial damping="2">
        <site site="s0"/>
        <site site="s1"/>
      </spatial>
    </tendon>
    """,
    """
    <actuator>
      <position site="s0" refsite="s1" kp="1" kv="1"/>
    </actuator>
    """,
  )
  def test_discrete_metric_couplings(self, coupling_xml):
    """Under discrete integrator, metric tendons and actuators couple trees into one island."""
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option integrator="discrete"/>
        <worldbody>
          <geom type="plane" size="2 2 0.1"/>
          <body name="b0" pos="-0.5 0 0.08">
            <freejoint/>
            <geom size="0.1"/>
            <site name="s0"/>
          </body>
          <body name="b1" pos="0.5 0 0.08">
            <freejoint/>
            <geom size="0.1"/>
            <site name="s1"/>
          </body>
        </worldbody>
        {coupling_xml}
      </mujoco>
      """,
      nworld=2,
    )
    qpos = d.qpos.numpy()
    qpos[1, 2] = 1.0
    qpos[1, 9] = 1.0
    d.qpos.assign(qpos)
    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    d.efc.island.fill_(-1)
    island.island(m, d)
    island.compute_island_mapping(m, d)

    self.assertEqual(mjd.nisland, 1)
    np.testing.assert_array_equal(d.nisland.numpy(), np.array([1, 0], dtype=np.int32))
    tree_island = d.tree_island.numpy()
    np.testing.assert_array_equal(tree_island[0], mjd.tree_island[: mjm.ntree])
    np.testing.assert_array_equal(tree_island[1], np.full(mjm.ntree, -1, dtype=np.int32))
    np.testing.assert_array_equal(d.efc.island.numpy()[0, : mjd.nefc], mjd.efc_island[: mjd.nefc])

    # Under PGS, metric couplings are inactive (matching mj_effCouplings in C)
    mjm.opt.solver = mujoco.mjtSolver.mjSOL_PGS
    m.opt.solver = int(mujoco.mjtSolver.mjSOL_PGS)
    mujoco.mj_forward(mjm, mjd)
    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    d.efc.island.fill_(-1)
    island.island(m, d)
    island.compute_island_mapping(m, d)
    self.assertEqual(mjd.nisland, 2)
    np.testing.assert_array_equal(d.nisland.numpy(), np.array([2, 0], dtype=np.int32))
    np.testing.assert_array_equal(d.tree_island.numpy()[0], mjd.tree_island[: mjm.ntree])

  @parameterized.product(
    mode=("tendon", "slidercrank"),
    nworld=(1, 2),
  )
  def test_discrete_metric_couplings_tree0_asleep_and_tendon_actuator(self, mode, nworld):
    """Awake trees coupled by a tendon or slider-crank actuator merge when tree 0 is asleep."""
    if mode == "tendon":
      mjm, mjd, m, d = test_data.fixture(
        xml="""
        <mujoco>
          <option integrator="discrete">
            <flag sleep="enable"/>
          </option>
          <worldbody>
            <geom type="plane" size="2 2 0.1"/>
            <body name="b0" pos="0 0 1">
              <freejoint/>
              <geom size="0.1"/>
            </body>
            <body name="b1" pos="-0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s1"/>
            </body>
            <body name="b2" pos="0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s2"/>
            </body>
          </worldbody>
          <tendon>
            <spatial name="t12">
              <site site="s1"/>
              <site site="s2"/>
            </spatial>
          </tendon>
          <actuator>
            <position tendon="t12" kp="1" kv="2"/>
          </actuator>
        </mujoco>
        """,
        nworld=nworld,
      )
    else:
      mjm, mjd, m, d = test_data.fixture(
        xml="""
        <mujoco>
          <option integrator="discrete">
            <flag sleep="enable"/>
          </option>
          <worldbody>
            <geom type="plane" size="2 2 0.1"/>
            <body name="b0" pos="0 0 1">
              <freejoint/>
              <geom size="0.1"/>
            </body>
            <body name="b1" pos="-0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s1"/>
            </body>
            <body name="b2" pos="0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s2"/>
            </body>
          </worldbody>
          <actuator>
            <position cranksite="s1" slidersite="s2" cranklength="0.2" kp="1" kv="2"/>
          </actuator>
        </mujoco>
        """,
        nworld=nworld,
      )

    mjwarp.fwd_position(m, d)
    tree_awake = np.zeros((nworld, 3), dtype=np.int32)
    tree_awake[0] = [0, 1, 1]
    if nworld == 2:
      tree_awake[1] = [0, 1, 0]
    d.tree_awake.assign(tree_awake)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    d.efc.island.fill_(-1)
    island.island(m, d)
    island.compute_island_mapping(m, d)

    self.assertEqual(mjd.nisland, 1)
    np.testing.assert_array_equal(d.nisland.numpy()[0], mjd.nisland)
    np.testing.assert_array_equal(d.tree_island.numpy()[0], mjd.tree_island[: mjm.ntree])
    np.testing.assert_array_equal(d.efc.island.numpy()[0, : mjd.nefc], mjd.efc_island[: mjd.nefc])
    if nworld == 2:
      self.assertEqual(int(d.nisland.numpy()[1]), 1)
      np.testing.assert_array_equal(d.tree_island.numpy()[1], np.array([-1, 0, -1], dtype=np.int32))
      self.assertFalse(np.array_equal(d.tree_island.numpy()[0], d.tree_island.numpy()[1]))

  @parameterized.product(
    case=(
      "stiffness_only",
      "stiffnesspoly_only",
      "dampingpoly_only",
      "all_disabled",
      "actuator_damping",
    ),
    nworld=(1, 2),
  )
  def test_eff_tendon_possible_branches(self, case, nworld):
    """Tests eff_tendon_possible stiffness, polynomial/actuator damping, and disable flags."""
    if case == "stiffness_only":
      mjm, mjd, m, d = test_data.fixture(
        xml="""
        <mujoco>
          <option integrator="discrete"/>
          <worldbody>
            <geom type="plane" size="2 2 0.1"/>
            <body name="b0" pos="-0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s0"/>
            </body>
            <body name="b1" pos="0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s1"/>
            </body>
          </worldbody>
          <tendon>
            <spatial stiffness="5">
              <site site="s0"/>
              <site site="s1"/>
            </spatial>
          </tendon>
        </mujoco>
        """,
        nworld=nworld,
      )
      expected_nisland = 1
    elif case == "stiffnesspoly_only":
      mjm, mjd, m, d = test_data.fixture(
        xml="""
        <mujoco>
          <option integrator="discrete"/>
          <worldbody>
            <geom type="plane" size="2 2 0.1"/>
            <body name="b0" pos="-0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s0"/>
            </body>
            <body name="b1" pos="0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s1"/>
            </body>
          </worldbody>
          <tendon>
            <spatial stiffness="0 2 0">
              <site site="s0"/>
              <site site="s1"/>
            </spatial>
          </tendon>
        </mujoco>
        """,
        nworld=nworld,
      )
      expected_nisland = 1
    elif case == "dampingpoly_only":
      mjm, mjd, m, d = test_data.fixture(
        xml="""
        <mujoco>
          <option integrator="discrete"/>
          <worldbody>
            <geom type="plane" size="2 2 0.1"/>
            <body name="b0" pos="-0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s0"/>
            </body>
            <body name="b1" pos="0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s1"/>
            </body>
          </worldbody>
          <tendon>
            <spatial damping="0 2 0">
              <site site="s0"/>
              <site site="s1"/>
            </spatial>
          </tendon>
        </mujoco>
        """,
        nworld=nworld,
      )
      expected_nisland = 1
    elif case == "all_disabled":
      mjm, mjd, m, d = test_data.fixture(
        xml="""
        <mujoco>
          <option integrator="discrete">
            <flag spring="disable" damper="disable"/>
          </option>
          <worldbody>
            <geom type="plane" size="2 2 0.1"/>
            <body name="b0" pos="-0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s0"/>
            </body>
            <body name="b1" pos="0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s1"/>
            </body>
          </worldbody>
          <tendon>
            <spatial stiffness="5" damping="2">
              <site site="s0"/>
              <site site="s1"/>
            </spatial>
          </tendon>
        </mujoco>
        """,
        nworld=nworld,
      )
      expected_nisland = 2
    else:
      mjm, mjd, m, d = test_data.fixture(
        xml="""
        <mujoco>
          <option integrator="discrete"/>
          <worldbody>
            <geom type="plane" size="2 2 0.1"/>
            <body name="b0" pos="-0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s0"/>
            </body>
            <body name="b1" pos="0.5 0 0.08">
              <freejoint/>
              <geom size="0.1"/>
              <site name="s1"/>
            </body>
          </worldbody>
          <tendon>
            <spatial name="t">
              <site site="s0"/>
              <site site="s1"/>
            </spatial>
          </tendon>
          <actuator>
            <motor tendon="t" damping="2"/>
          </actuator>
        </mujoco>
        """,
        nworld=nworld,
      )
      expected_nisland = 1

    mjds = [mjd]
    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      qpos = d.qpos.numpy()
      qpos[1, 2] = 1.0
      qpos[1, 9] = 1.0
      d.qpos.assign(qpos)
      mjd1.qpos[:] = qpos[1]
      mujoco.mj_forward(mjm, mjd1)
      mjds.append(mjd1)

    mjwarp.fwd_position(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    d.efc.island.fill_(-1)
    island.island(m, d)
    island.compute_island_mapping(m, d)

    self.assertEqual(mjd.nisland, expected_nisland)
    for w in range(nworld):
      self.assertEqual(int(d.nisland.numpy()[w]), mjds[w].nisland)
      np.testing.assert_array_equal(d.tree_island.numpy()[w], mjds[w].tree_island[: mjm.ntree])
      np.testing.assert_array_equal(d.efc.island.numpy()[w, : mjds[w].nefc], mjds[w].efc_island[: mjds[w].nefc])
    if nworld == 2:
      self.assertFalse(np.array_equal(d.nisland.numpy()[0], d.nisland.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_flex_and_passive_contact_islands(self, nworld):
    """Tests multi-tree flex and passive flex contact island unioning under discrete integrator."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete">
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="2 2 0.1"/>
          <body name="box" pos="0 0 0.08">
            <freejoint/>
            <geom size="0.1" contype="3" conaffinity="3"/>
          </body>
          <flexcomp name="cloth" count="2 2 1" spacing="0.06 0.06 0.06" pos="-0.03 -0.03 0.17"
                    radius="0.02" mass="0.2">
            <contact selfcollide="none" passive="true" contype="2" conaffinity="2"/>
            <elasticity young="500" poisson="0.3" thickness="0.01" elastic2d="stretch" damping="0.01"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    mjds = [mjd]
    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      qpos = d.qpos.numpy()
      # Lift the box off the floor and away from the cloth in world 1 so nefc == 0
      # while sleep + discrete + stiff flex still unions the 4 cloth vertex trees into 1 island
      qpos[1, 2] = 2.0
      d.qpos.assign(qpos)
      mjd1.qpos[:] = qpos[1]
      mjds.append(mjd1)

    mjwarp.fwd_position(m, d)
    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    d.efc.island.fill_(-1)
    island.island(m, d)
    island.compute_island_mapping(m, d)

    nisland = d.nisland.numpy()
    tree_island = d.tree_island.numpy()
    # World 0: floor EFC + box-cloth passive contact merge all 5 trees into 1 island
    self.assertEqual(int(nisland[0]), 1)
    np.testing.assert_array_equal(tree_island[0], np.zeros(mjm.ntree, dtype=np.int32))
    if nworld == 2:
      # World 1: nefc == 0, box (-1) is unconstrained while 4 cloth trees merge into island 0
      self.assertEqual(int(nisland[1]), 1)
      np.testing.assert_array_equal(tree_island[1], np.array([-1, 0, 0, 0, 0], dtype=np.int32))
      self.assertFalse(np.array_equal(tree_island[0], tree_island[1]))

    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])
      self.assertEqual(int(nisland[w]), mjds[w].nisland)
      np.testing.assert_array_equal(tree_island[w], mjds[w].tree_island[: mjm.ntree])


class IslandDiscoveryExecutionTest(absltest.TestCase):
  """Tests multi-world execution, idempotency, and graph capturability."""

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

    with wp.ScopedCapture() as capture:
      island.island(m, d)

    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    wp.capture_launch(capture.graph)
    expected_labels = d.tree_island.numpy().copy()
    expected_nisland = d.nisland.numpy().copy()

    for _ in range(3):
      d.tree_island.fill_(-1)
      d.nisland.fill_(-1)
      wp.capture_launch(capture.graph)
      np.testing.assert_array_equal(d.tree_island.numpy(), expected_labels)
      np.testing.assert_array_equal(d.nisland.numpy(), expected_nisland)

  def test_graph_capture_and_replay(self):
    """Direct DSU is capturable in graphs with zero host synchronizations."""
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

  def test_island_dsu_module_is_unique(self):
    """Ensure _island_dsu does not trigger shared module recompilation."""
    self.assertNotEqual(island._island_dsu.module, island._finalize_dsu.module)


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
