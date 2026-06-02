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

"""Tests for nv_compact active-DOF bookkeeping."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp
from mujoco_warp._src import nvmax

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


def _setup(nworld=1, nv_max=None, sparse=False):
  mjm = mujoco.MjModel.from_xml_string(_XML)
  if sparse:
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  m = mjwarp.put_model(mjm)
  d = mjwarp.make_data(mjm, nworld=nworld, nv_max=nv_max)
  return mjm, m, d


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


def _setup_contact(nv_max=None):
  mjm = mujoco.MjModel.from_xml_string(_CONTACT_XML)
  m = mjwarp.put_model(mjm)
  d = mjwarp.make_data(mjm, nv_max=nv_max)
  return mjm, m, d


class NvCompactBookkeepingTest(absltest.TestCase):
  def test_actuated_tree_seeded(self):
    """Only the actuated arm tree is active at construction; maps are compact."""
    _, m, d = _setup()
    np.testing.assert_array_equal(d.tree_active.numpy()[0], [True, False, False])

    nvmax.update_active_dofs(m, d)

    self.assertEqual(d.ncdof.numpy()[0], 2)
    # arm dofs 0,1 map to compacted 0,1; the rest are inactive (-1).
    dof_cdof = d.dof_cdof.numpy()[0]
    np.testing.assert_array_equal(dof_cdof[:2], [0, 1])
    self.assertTrue((dof_cdof[2:] == -1).all())
    np.testing.assert_array_equal(d.cdof_dof.numpy()[0][:2], [0, 1])

  def test_applied_force_wakes_tree_per_world(self):
    """qfrc_applied on a free-body DOF wakes its whole tree, independently per world."""
    _, m, d = _setup(nworld=2)
    qfrc = d.qfrc_applied.numpy()
    qfrc[0, 2] = 1.0  # dof 2 belongs to tree 1 (first free body) in world 0
    d.qfrc_applied = wp.array(qfrc, dtype=float)

    nvmax.update_active_dofs(m, d)

    # world 0: arm (2) + free body (6) = 8; world 1: arm only = 2
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
    mjm, m, d = _setup()
    ball0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball0")
    ball1 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball1")

    self._fake_contact(d, ball0, ball1)
    self._set_body_moving(m, d, ball0)  # ball0 is moving into ball1
    nvmax.update_active_dofs(m, d)

    # arm (seeded) + both free bodies = 2 + 6 + 6 = 14
    np.testing.assert_array_equal(d.tree_active.numpy()[0], [True, True, True])
    self.assertEqual(d.ncdof.numpy()[0], 14)

  def test_resting_contact_does_not_wake(self):
    """A contact where both bodies are at rest does not wake them (velocity gate)."""
    mjm, m, d = _setup()
    ball0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball0")
    ball1 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball1")

    self._fake_contact(d, ball0, ball1)  # both at rest (cvel = 0)
    nvmax.update_active_dofs(m, d)

    # only the actuated arm stays active; the resting balls remain asleep
    np.testing.assert_array_equal(d.tree_active.numpy()[0], [True, False, False])
    self.assertEqual(d.ncdof.numpy()[0], 2)

  def test_static_geom_does_not_wake(self):
    """A contact with a static (world) geom only wakes the moving dynamic side."""
    mjm, m, d = _setup()
    floor = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ball0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball0")

    self._fake_contact(d, floor, ball0)
    self._set_body_moving(m, d, ball0)  # ball0 dropping onto the floor
    nvmax.update_active_dofs(m, d)

    # floor is static (tree -1): only ball0's tree wakes, alongside the arm.
    np.testing.assert_array_equal(d.tree_active.numpy()[0], [True, True, False])
    self.assertEqual(d.ncdof.numpy()[0], 8)

  def test_monotonic(self):
    """Once active, a tree stays active after the triggering force is removed."""
    _, m, d = _setup()
    qfrc = d.qfrc_applied.numpy()
    qfrc[0, 8] = 1.0  # wake tree 2
    d.qfrc_applied = wp.array(qfrc, dtype=float)
    nvmax.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], 8)

    d.qfrc_applied = wp.array(np.zeros_like(qfrc), dtype=float)
    nvmax.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], 8)  # still active

  def test_overflow_clamps_and_warns(self):
    """When active DOFs exceed nv_max, ncdof is clamped to nv_max."""
    _, m, d = _setup(nv_max=4)
    ta = d.tree_active.numpy()
    ta[0, :] = True
    d.tree_active = wp.array(ta, dtype=bool)

    nvmax.update_active_dofs(m, d)

    self.assertEqual(d.ncdof.numpy()[0], 4)


class NvCompactSmoothSolveTest(absltest.TestCase):
  def test_smooth_solve_equivalence_all_active(self):
    """With every tree active and nv_max=nv, compacted qacc_smooth matches baseline."""
    _, m, d = _setup(sparse=True)
    self.assertTrue(m.is_sparse)
    mjwarp.forward(m, d)
    baseline = d.qacc_smooth.numpy().copy()

    d.tree_active = wp.array(np.ones((d.nworld, m.ntree), dtype=bool), dtype=bool)
    nvmax.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], m.nv)

    ctx = nvmax.create_nvcompact_context(m, d)
    nvmax.smooth_solve_compact(m, d, ctx)

    np.testing.assert_allclose(d.qacc_smooth.numpy(), baseline, rtol=1e-4, atol=1e-5)

  def test_smooth_solve_partial_active_freezes_rest(self):
    """Active trees match baseline (M is block-diagonal); inactive DOFs are frozen to 0."""
    _, m, d = _setup(sparse=True)
    mjwarp.forward(m, d)
    baseline = d.qacc_smooth.numpy().copy()

    # only the actuated arm tree (dofs 0-1) is active
    nvmax.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], 2)

    ctx = nvmax.create_nvcompact_context(m, d)
    nvmax.smooth_solve_compact(m, d, ctx)

    out = d.qacc_smooth.numpy()
    np.testing.assert_allclose(out[0, :2], baseline[0, :2], rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(out[0, 2:], np.zeros(m.nv - 2))


class NvCompactConstrainedSolveTest(absltest.TestCase):
  def test_constrained_solve_equivalence_all_active(self):
    """With every tree active and nv_max=nv, the compacted Newton solve matches baseline qacc."""
    _, m, d = _setup_contact()
    self.assertTrue(m.is_sparse)
    mjwarp.forward(m, d)  # full baseline solve (also builds efc.J, M)
    self.assertGreater(d.nacon.numpy()[0], 0)  # contacts exist
    baseline_qacc = d.qacc.numpy().copy()

    d.tree_active = wp.array(np.ones((d.nworld, m.ntree), dtype=bool), dtype=bool)
    nvmax.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], m.nv)

    ctx = nvmax.create_nvcompact_context(m, d)
    nvmax.solve_compact(m, d, ctx)

    np.testing.assert_allclose(d.qacc.numpy(), baseline_qacc, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
  wp.clear_kernel_cache()
  absltest.main()
