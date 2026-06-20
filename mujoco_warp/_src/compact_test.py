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

"""Tests for active-DOF compaction (nvmax < nv): bookkeeping and compacted solves."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import island
from mujoco_warp._src import sleep
from mujoco_warp._src import solver
from mujoco_warp._src.sleep import K_AWAKE_VAL

# An actuated 2-hinge arm (tree 0, dofs 0-1) plus two free bodies
# (tree 1, dofs 2-7 and tree 2, dofs 8-13).
_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body>
      <joint name="j0" type="hinge"/>
      <geom type="capsule" fromto="0 0 0 0 0 .3" size=".05"/>
      <body pos="0 0 .3">
        <joint name="j1" type="hinge"/>
        <geom type="capsule" fromto="0 0 0 0 0 .3" size=".05"/>
      </body>
    </body>
    <body pos="1 0 1"><freejoint/><geom name="ball0" type="sphere" size=".1"/></body>
    <body pos="2 0 1"><freejoint/><geom name="ball1" type="sphere" size=".1"/></body>
  </worldbody>
  <actuator>
    <motor joint="j0"/>
    <motor joint="j1"/>
  </actuator>
</mujoco>
"""


# Two free spheres resting (penetrating) on a floor, plus an actuated hinge arm.
# Generates contacts so the constrained solver does real work.
_CONTACT_XML = """
<mujoco>
  <option jacobian="sparse" solver="Newton" iterations="20"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body>
      <joint name="j0" type="hinge"/>
      <geom type="capsule" fromto="0 0 .5 .3 0 .5" size=".05"/>
    </body>
    <body pos="1 0 .08"><freejoint/><geom type="sphere" size=".1"/></body>
    <body pos="2 0 .08"><freejoint/><geom type="sphere" size=".1"/></body>
  </worldbody>
  <actuator><motor joint="j0"/></actuator>
</mujoco>
"""


def _put_compact(xml: str, nvmax: int | None = None, sparse: bool = False):
  """Build (mjm, mjd, m, d) requesting the compact workspace.

  ``nvmax=None`` allocates the compact arrays at full size (nvmax == nv) so an
  all-active compacted solve can be compared directly against the full baseline.
  ``test_data.fixture`` does not expose nvmax, so the solve tests build directly.
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  if sparse:
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjwarp.put_model(mjm)
  d = mjwarp.put_data(mjm, mjd, nvmax=mjm.nv if nvmax is None else nvmax)
  return mjm, mjd, m, d


class NvCompactBookkeepingTest(absltest.TestCase):
  def test_actuated_tree_seeded(self):
    """Only the actuated arm tree is active at construction; maps are compact."""
    _, _, m, d = test_data.fixture(xml=_XML)
    d.tree_awake = wp.array([[1, 0, 0]], dtype=int)
    island.update_active_dofs(m, d)

    self.assertEqual(d.ncdof.numpy()[0], 2)
    # arm dofs 0,1 map to compacted 0,1; the rest are inactive (-1).
    dof_cdof = d.dof_cdof.numpy()[0]
    np.testing.assert_array_equal(dof_cdof[:2], [0, 1])
    self.assertTrue((dof_cdof[2:] == -1).all())
    np.testing.assert_array_equal(d.cdof_dof.numpy()[0][:2], [0, 1])

  def test_applied_force_wakes_tree_per_world(self):
    """qfrc_applied on a free-body DOF wakes its whole tree, independently per world."""
    _, _, m, d = test_data.fixture(xml=_XML, nworld=2)

    # 1. Set all trees asleep initially
    d.tree_awake.fill_(0)
    d.tree_asleep.fill_(0)

    # 2. Apply force to dof 2 (belongs to tree 1) in world 0
    qfrc = d.qfrc_applied.numpy()
    qfrc[0, 2] = 1.0
    d.qfrc_applied = wp.array(qfrc, dtype=float)

    # 3. Run sleep.wake to wake up trees with forces
    sleep.wake(m, d)
    sleep.update_sleep(m, d)

    island.update_active_dofs(m, d)

    # world 0: arm (2) + free body 1 (6) = 8; world 1: arm only = 2
    np.testing.assert_array_equal(d.ncdof.numpy(), [8, 2])
    np.testing.assert_array_equal(d.dof_cdof.numpy()[0][:8], np.arange(8))
    self.assertTrue((d.dof_cdof.numpy()[1][2:] == -1).all())

  def _fake_contact(self, d, g0, g1):
    d.nacon = wp.array([1], dtype=int)
    geom = d.contact.geom.numpy()
    geom[0] = (g0, g1)
    d.contact.geom = wp.array(geom, dtype=wp.vec2i)
    d.contact.worldid = wp.array(np.zeros(d.naconmax, dtype=int), dtype=int)

  def _set_body_moving(self, m, d, geomid, speed=1.0):
    bodyid = m.geom_bodyid.numpy()[geomid]
    cvel = d.cvel.numpy()
    cvel[0, bodyid, 3] = speed  # linear-x component of spatial velocity
    d.cvel = wp.array(cvel, dtype=wp.spatial_vector)

  def test_moving_contact_wakes_both_trees(self):
    """A contact with a moving body activates the trees of both involved geoms."""
    mjm, _, m, d = test_data.fixture(xml=_XML)
    ball0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball0")
    ball1 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball1")

    # 1. Set tree 1 (ball0) awake initially, tree 0 and 2 asleep (pointing to themselves)
    asleep = np.array([[0, K_AWAKE_VAL, 2]], dtype=np.int32)
    d.tree_asleep = wp.array(asleep, dtype=int)
    sleep.update_sleep(m, d)

    self._fake_contact(d, ball0, ball1)

    sleep.wake_collision(m, d)
    sleep.update_sleep(m, d)

    island.update_active_dofs(m, d)

    # both free bodies awake = 6 + 6 = 12 DOFs
    self.assertEqual(d.ncdof.numpy()[0], 12)

  def test_resting_contact_does_not_wake(self):
    """A contact where both bodies are at rest does not wake them."""
    mjm, _, m, d = test_data.fixture(xml=_XML)
    ball0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball0")
    ball1 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball1")

    # 1. Set all trees asleep initially
    asleep = np.array([[0, 1, 2]], dtype=np.int32)
    d.tree_asleep = wp.array(asleep, dtype=int)
    sleep.update_sleep(m, d)

    self._fake_contact(d, ball0, ball1)

    sleep.wake_collision(m, d)
    sleep.update_sleep(m, d)

    island.update_active_dofs(m, d)

    # both stay asleep
    self.assertEqual(d.ncdof.numpy()[0], 0)

  def test_static_geom_does_not_wake(self):
    """A contact with a static (world) geom only wakes the dynamic side."""
    mjm, _, m, d = test_data.fixture(xml=_XML)
    floor = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ball0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball0")

    # 1. Set ball0 (tree 1) awake, tree 0 and 2 asleep
    asleep = np.array([[0, K_AWAKE_VAL, 2]], dtype=np.int32)
    d.tree_asleep = wp.array(asleep, dtype=int)
    sleep.update_sleep(m, d)

    self._fake_contact(d, floor, ball0)

    sleep.wake_collision(m, d)
    sleep.update_sleep(m, d)

    island.update_active_dofs(m, d)

    # only tree 1 (ball0) wakes -> 6 DOFs
    self.assertEqual(d.ncdof.numpy()[0], 6)

  def test_overflow_clamps_and_warns(self):
    """When active DOFs exceed nvmax, ncdof is clamped to nvmax."""
    mjm = mujoco.MjModel.from_xml_string(_XML)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd, nvmax=4)
    d.tree_awake = wp.array([[1, 1, 1]], dtype=int)

    island.update_active_dofs(m, d)

    self.assertEqual(d.ncdof.numpy()[0], 4)


class NvCompactSmoothSolveTest(absltest.TestCase):
  def test_smooth_solve_equivalence_all_active(self):
    """With every tree active and nvmax=nv, compacted qacc_smooth matches baseline."""
    _, _, m, d = _put_compact(_XML, sparse=True)
    self.assertTrue(m.is_sparse)
    mjwarp.forward(m, d)
    baseline = d.qacc_smooth.numpy().copy()

    d.tree_awake = wp.array(np.ones((d.nworld, m.ntree), dtype=int), dtype=int)
    island.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], m.nv)

    solver.smooth_solve_compact(m, d)

    np.testing.assert_allclose(d.qacc_smooth.numpy(), baseline, rtol=1e-4, atol=1e-5)

  def test_smooth_solve_partial_active_freezes_rest(self):
    """Active trees match baseline (M is block-diagonal); inactive DOFs are frozen to 0."""
    _, _, m, d = _put_compact(_XML, sparse=True)
    mjwarp.forward(m, d)
    baseline = d.qacc_smooth.numpy().copy()

    # only the actuated arm tree (dofs 0-1) is active
    d.tree_awake = wp.array([[1, 0, 0]], dtype=int)
    island.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], 2)

    solver.smooth_solve_compact(m, d)

    out = d.qacc_smooth.numpy()
    np.testing.assert_allclose(out[0, :2], baseline[0, :2], rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(out[0, 2:], np.zeros(m.nv - 2))


class NvCompactConstrainedSolveTest(absltest.TestCase):
  def test_constrained_solve_equivalence_all_active(self):
    """With every tree active and nvmax=nv, the compacted Newton solve matches baseline qacc."""
    _, _, m, d = _put_compact(_CONTACT_XML)
    self.assertTrue(m.is_sparse)
    mjwarp.forward(m, d)  # full baseline solve (also builds efc.J, M)
    self.assertGreater(d.nacon.numpy()[0], 0)  # contacts exist
    baseline_qacc = d.qacc.numpy().copy()

    d.tree_awake = wp.array(np.ones((d.nworld, m.ntree), dtype=int), dtype=int)
    island.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], m.nv)

    solver.solve_compact(m, d)

    np.testing.assert_allclose(d.qacc.numpy(), baseline_qacc, rtol=1e-3, atol=1e-4)

  def test_solve_compact_populates_islands(self):
    """When using the compact solver via mjwarp.solve, island mapping fields are updated."""
    mjm = mujoco.MjModel.from_xml_string(_CONTACT_XML)
    mjm.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP  # Newton + sleeping -> m.is_compact
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjwarp.put_model(mjm)
    self.assertTrue(m.is_compact)
    # nv = 13. We set nvmax = 8 to size the compacted block below nv.
    d = mjwarp.put_data(mjm, mjd, nvmax=8)

    # Artificially set tree_island map on device
    d.tree_island = wp.array([[2, 2, 2]], dtype=int)
    d.nisland = wp.array([3], dtype=int)

    # Run solver which triggers solve_compact
    mjwarp.solve(m, d)

    # Assert that d.dof_island and d.efc.island are updated based on the new tree_island
    dof_island = d.dof_island.numpy()[0]
    np.testing.assert_array_equal(dof_island, [2] * m.nv)

    nefc = d.nefc.numpy()[0]
    efc_island = d.efc.island.numpy()[0]
    self.assertGreater(nefc, 0)
    np.testing.assert_array_equal(efc_island[:nefc], [2] * nefc)
    np.testing.assert_array_equal(efc_island[nefc:], [-1] * (d.njmax - nefc))


if __name__ == "__main__":
  wp.clear_kernel_cache()
  absltest.main()
