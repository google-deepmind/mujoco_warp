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

"""Tests for set_const functions."""

import copy

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import util_pkg
from mujoco_warp._src.set_const import set_length_range


def _assert_eq(a, b, name):
  tol = 5e-4
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


_GROUPS = (
  ("fixed", mjwarp.SetConstGroup.FIXED),
  ("kin0", mjwarp.SetConstGroup.KIN0),
  ("inert0", mjwarp.SetConstGroup.INERT0),
)


def _is_batched_model_field(name: str) -> bool:
  ann = mjwarp.Model.__annotations__.get(name)
  return isinstance(ann, wp.array) and ann.shape[0] in ("*", "nworld")


def _setconst_output_fields(*groups: str, pure_only: bool = False) -> frozenset[str]:
  if not groups:
    groups = ("fixed", "kin0", "inert0")
  indices = [mujoco.MjModel._setconst_groups.index(g) for g in groups]
  return frozenset(
    name
    for name, access in mujoco.MjModel._setconst_fields.items()
    if any(access[i] == mujoco.mjSC_OUT if pure_only else (access[i] & mujoco.mjSC_OUT) for i in indices)
    and _is_batched_model_field(name)
  )


class SetConstTest(parameterized.TestCase):
  def test_set_const_qpos0_modification(self):
    """Test set_const recomputes fields after qpos0 modification."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="link1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <site name="s1" pos="0.1 0 0"/>
          <body name="link2" pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom name="g2" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
            <site name="s2" pos="0.4 0 0"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="tendon1">
          <site site="s1"/>
          <site site="s2"/>
        </spatial>
      </tendon>
    </mujoco>
    """
    )

    mjm.qpos0[:] = [0.3, 0.5]
    m.qpos0.numpy()[0, :] = [0.3, 0.5]

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")
    _assert_eq(m.tendon_invweight0.numpy()[0], mjm.tendon_invweight0, "tendon_invweight0")
    _assert_eq(m.tendon_length0.numpy()[0], mjm.tendon_length0, "tendon_length0")

  def test_set_const_body_mass_modification(self):
    """Test set_const recomputes fields after body_mass modification."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="link1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <body name="link2" pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom name="g2" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor1" joint="j1" gear="1"/>
        <motor name="motor2" joint="j2" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    new_mass = 3.0
    mjm.body_mass[1] = new_mass
    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = new_mass
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")
    _assert_eq(m.body_subtreemass.numpy()[0], mjm.body_subtreemass, "body_subtreemass")
    _assert_eq(m.actuator_acc0.numpy()[0], mjm.actuator_acc0, "actuator_acc0")
    _assert_eq(m.body_invweight0.numpy()[0, 1, 0], mjm.body_invweight0[1, 0], "body_invweight0")

  def test_set_const_unbatched_model_multi_world(self):
    """Test set_const with unbatched model and nworld > 1."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml", nworld=4)

    new_mass = 3.0
    mjm.body_mass[1] = new_mass
    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = new_mass
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.body_subtreemass.numpy()[0], mjm.body_subtreemass, "body_subtreemass")
    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")

  def test_set_const_eq_data_connect(self):
    """Test set_const recomputes eq_data for connect constraints."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option gravity="0 0 0">
        <flag contact="disable"/>
      </option>
      <worldbody>
        <body name="b1" pos="1 0 0">
          <joint type="slide" axis="1 0 0" ref="1"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
        <body name="b2" pos="2 0 0">
          <joint type="slide" axis="1 0 0" ref="4"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
      <equality>
        <connect body1="b1" body2="b2" anchor="0.5 0 0"/>
      </equality>
    </mujoco>
    """
    )

    # move the anchor on body1 and clear the derived second anchor
    new_data = mjm.eq_data[0].copy()
    new_data[0:3] = [0.4, 0.1, 0.0]
    new_data[3:6] = 0.0
    mjm.eq_data[0] = new_data
    eq_data = m.eq_data.numpy()
    eq_data[0, 0] = new_data
    wp.copy(m.eq_data, wp.array(eq_data, dtype=m.eq_data.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.eq_data.numpy()[0], mjm.eq_data, "eq_data")

  @parameterized.parameters(1, 2)
  def test_set_const_eq_data_weld(self, nworld):
    """Test set_const recomputes eq_data for marked weld constraints."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="b1" pos="0.1 0 0">
          <joint ref="0.3"/>
          <geom size="0.05"/>
        </body>
        <body name="b2" pos="0.4 0 0" euler="0 0 20">
          <joint axis="0 1 0" ref="-0.2"/>
          <geom size="0.05"/>
        </body>
        <body name="mocap" mocap="true" pos="0 0.5 0"/>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2" anchor="0.1 0.2 0.3"/>
        <weld body1="mocap" body2="b2"/>
      </equality>
    </mujoco>
    """,
      nworld=nworld,
      batch_sizes={"eq_data": nworld},
    )

    # displaced mocap pose in Data must not affect set_const weld pose, and is restored
    mjd.mocap_pos[0] = [5.0, 6.0, 7.0]
    mocap_pos = d.mocap_pos.numpy()
    mocap_pos[:, 0] = [5.0, 6.0, 7.0]
    d.mocap_pos.assign(mocap_pos)

    # case 1: marked weld (data[11] != 0): set_const recomputes relpose and anchor offset
    anchors = np.array([[0.15, 0.1, 0.25], [0.25, -0.05, 0.1]])[:nworld]
    eq_data = m.eq_data.numpy()
    for w in range(nworld):
      eq_data[w, 0, 0:3] = anchors[w]
      eq_data[w, 0, 3:10] = 0.0
      eq_data[w, 1, 3:10] = 0.0
    wp.copy(m.eq_data, wp.array(eq_data, dtype=m.eq_data.dtype))

    mjwarp.set_const(m, d)
    _assert_eq(d.mocap_pos.numpy()[:, 0], mocap_pos[:, 0], "mocap_pos_restored")

    eq_data_np = m.eq_data.numpy()
    for w in range(nworld):
      mjm.eq_data[0, 0:3] = anchors[w]
      mjm.eq_data[0, 3:10] = 0.0
      mjm.eq_data[1, 3:10] = 0.0
      mujoco.mj_setConst(mjm, mjd)
      _assert_eq(eq_data_np[w], mjm.eq_data, f"eq_data_world_{w}")
      self.assertTrue(np.any(eq_data_np[w, 0, 3:10] != 0.0))

    if nworld == 2:
      self.assertFalse(np.allclose(eq_data_np[0, 0, 3:6], eq_data_np[1, 0, 3:6]))

    # case 2: unmarked weld (data[11] == 0): set_const leaves relpose unchanged
    new_data = mjm.eq_data[0].copy()
    new_data[3:10] = [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    new_data[11] = 0.0
    mjm.eq_data[0] = new_data
    eq_data[:, 0] = new_data
    wp.copy(m.eq_data, wp.array(eq_data, dtype=m.eq_data.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    eq_data_np = m.eq_data.numpy()
    for w in range(nworld):
      _assert_eq(eq_data_np[w], mjm.eq_data, f"unmarked_eq_data_world_{w}")
      _assert_eq(eq_data_np[w, 0, 3:10], new_data[3:10], f"unmarked_relpose_world_{w}")

  def test_set_const_eq_data_slider_crank_tracking(self):
    """Test connect anchor recompute on a slider-crank with nonzero joint ref."""
    r, ell = 0.075, 0.096
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option gravity="0 0 0" timestep="0.002">
        <flag contact="disable"/>
      </option>
      <worldbody>
        <body name="crank">
          <joint name="phi" type="hinge" axis="0 0 1" ref="0.35" springref="0.55" stiffness="0.02" damping="0.002"/>
          <geom type="capsule" fromto="0 0 0 0.075 0 0" size="0.008" mass="0.06"/>
          <body name="conrod" pos="0.075 0 0">
            <joint name="psi" type="hinge" axis="0 0 1" ref="-0.2" damping="0.0005"/>
            <geom type="capsule" fromto="0 0 0 -0.1 0 0" size="0.006" mass="0.05"/>
          </body>
        </body>
        <body name="slider" pos="0.005 0 0">
          <joint name="s" type="slide" axis="1 0 0" ref="0.01" springref="0.02" stiffness="0.5" damping="0.01"/>
          <geom type="box" size="0.01 0.008 0.008" mass="0.08"/>
        </body>
      </worldbody>
      <equality>
        <connect body1="conrod" body2="slider" anchor="-0.1 0 0"/>
      </equality>
      <actuator>
        <motor joint="phi" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    def slider_crank_s(phi):
      # closed-form slide position vs crank angle, s(0) = 0 at top dead center
      return r * np.cos(phi) - np.sqrt(ell**2 - (r * np.sin(phi)) ** 2) - (r - ell)

    # re-anchor the conrod pin (effective conrod length 0.100 -> 0.096) and clear
    # the derived anchor; per set_const docs the offsets are recomputed if not set
    new_data = mjm.eq_data[0].copy()
    new_data[0:3] = [-ell, 0.0, 0.0]
    new_data[3:6] = 0.0
    mjm.eq_data[0] = new_data
    eq_data = m.eq_data.numpy()
    eq_data[0, 0] = new_data
    wp.copy(m.eq_data, wp.array(eq_data, dtype=m.eq_data.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.eq_data.numpy()[0], mjm.eq_data, "eq_data")

    # roll out the warp path from qpos0 under a constant crank torque and check
    # the loop against the closed form (joint ref values offset qpos readings)
    qpos0 = mjm.qpos0.copy()
    ctrl = 0.03

    mjwarp.reset_data(m, d)
    d.ctrl.fill_(ctrl)
    err_wp = 0.0
    for _ in range(10):
      mjwarp.step(m, d)
      qpos = d.qpos.numpy()[0]
      err_wp = max(err_wp, abs(qpos[2] - qpos0[2] - slider_crank_s(qpos[0] - qpos0[0])))

    self.assertLess(err_wp, 1.0e-3, f"warp path loop error {err_wp:.2e}")

  def test_set_const_freejoint(self):
    """Test set_const with freejoint (6 DOFs with special averaging)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="floating" pos="0 0 1">
          <freejoint/>
          <geom name="box" type="box" size="0.1 0.2 0.3" mass="2.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    new_mass = 5.0
    mjm.body_mass[1] = new_mass
    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = new_mass
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")
    _assert_eq(m.body_invweight0.numpy()[0, 1], mjm.body_invweight0[1], "body_invweight0")

  def test_set_const_full_freejoint_per_world_com(self):
    """A full free-joint factor handles diagonal and coupled worlds."""
    mjm, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="floating" pos="0 0 1">
          <freejoint/>
          <inertial pos="0 0 0" quat="0 1 0 0" mass="1" diaginertia="0.1 0.2 0.3"/>
        </body>
      </worldbody>
    </mujoco>
    """,
      nworld=2,
    )

    self.assertEqual(mjm.nC, 21)
    self.assertLen(m.M_tiles, 1)
    self.assertEqual(m.M_tiles[0].size, 6)
    self.assertEqual(m.M_tiles[0].elemid.size, 0)

    body_ipos = np.tile(mjm.body_ipos, (2, 1, 1))
    body_ipos[1, 1] = (0.05, 0.0, -0.02)
    m.body_ipos = wp.array(body_ipos, dtype=wp.vec3)
    m.body_invweight0 = wp.array(np.tile(mjm.body_invweight0, (2, 1, 1)), dtype=wp.vec2)
    m.dof_invweight0 = wp.array(np.tile(mjm.dof_invweight0, (2, 1)), dtype=float)

    mjwarp.set_const(m, d, groups=mjwarp.SetConstGroup.KIN0 | mjwarp.SetConstGroup.INERT0)

    dense = np.zeros((2, m.nv, m.nv))
    for worldid in range(2):
      mujoco.mju_sym2dense(dense[worldid], d.M.numpy()[worldid], mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    np.testing.assert_allclose(dense[0], np.diag(np.diag(dense[0])), atol=1e-6)
    self.assertGreater(np.max(np.abs(dense[1] - np.diag(np.diag(dense[1])))), 1e-6)
    self.assertGreaterEqual(m.qLD_block_adr.numpy()[0], 0)

    rhs = wp.ones((2, m.nv), dtype=float)
    result = wp.zeros_like(rhs)
    mjwarp.solve_m(m, d, result, rhs)
    expected = np.stack([np.linalg.solve(dense[worldid], np.ones(m.nv)) for worldid in range(2)])
    np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

  def test_set_const_balljoint(self):
    """Test set_const with ball joint (3 DOFs with averaging)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="arm">
          <joint name="ball" type="ball"/>
          <geom name="box" type="box" size="0.1 0.2 0.3" mass="2.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    new_inertia = np.array([0.1, 0.2, 0.3])
    mjm.body_inertia[1] = new_inertia
    body_inertia_np = m.body_inertia.numpy()
    body_inertia_np[0, 1] = new_inertia
    wp.copy(m.body_inertia, wp.array(body_inertia_np, dtype=m.body_inertia.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")

  def test_set_const_static_body(self):
    """Test set_const with static body (welded to world)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="static_body" pos="1 0 0">
          <geom name="static_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
        </body>
        <body name="dynamic_body">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom name="dynamic_geom" type="sphere" size="0.1" mass="2.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.body_invweight0.numpy()[0, 1], [0.0, 0.0], "body_invweight0")
    self.assertGreater(m.body_invweight0.numpy()[0, 2, 0], 0.0)
    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")

    # nv == 0: body_invweight0, tendon_invweight0, and actuator_acc0 are zeroed
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <site name="s1" pos="0 0 0"/>
        <body name="static_body" pos="1 0 0">
          <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>
          <site name="s2" pos="0.1 0 0"/>
        </body>
      </worldbody>
      <tendon>
        <spatial>
          <site site="s1"/>
          <site site="s2"/>
        </spatial>
      </tendon>
      <actuator>
        <adhesion body="static_body" ctrlrange="0 1"/>
      </actuator>
    </mujoco>
    """
    )
    for f in ("body_invweight0", "tendon_invweight0", "actuator_acc0"):
      getattr(m, f).fill_(wp.inf)
    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)
    for f in ("body_invweight0", "tendon_invweight0", "actuator_acc0"):
      _assert_eq(getattr(m, f).numpy()[0], getattr(mjm, f), f"{f}_nv0")

  def test_set_const_preserves_qpos(self):
    """Test that qpos is restored after set_const."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="mass">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom name="mass_geom" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    # Set qpos to a specific value
    mjd.qpos[0] = 0.5
    mujoco.mj_forward(mjm, mjd)
    d.qpos.numpy()[0, 0] = 0.5

    qpos_before = d.qpos.numpy().copy()
    mjwarp.set_const(m, d)

    _assert_eq(d.qpos.numpy(), qpos_before, "qpos")

  def test_set_fixed_body_subtreemass(self):
    """Test body_subtreemass accumulation for multi-level tree."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="root">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="sphere" size="0.1" mass="1.0"/>
          <body name="child1" pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom name="g2" type="sphere" size="0.1" mass="2.0"/>
            <body name="grandchild1" pos="0.5 0 0">
              <joint name="j3" type="hinge" axis="0 0 1"/>
              <geom name="g3" type="sphere" size="0.1" mass="3.0"/>
            </body>
          </body>
          <body name="child2" pos="0 0.5 0">
            <joint name="j4" type="hinge" axis="0 0 1"/>
            <geom name="g4" type="sphere" size="0.1" mass="4.0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    # Modify body masses and recompute
    mjm.body_mass[1] = 10.0  # root
    mjm.body_mass[2] = 20.0  # child1
    mjm.body_mass[3] = 30.0  # grandchild1
    mjm.body_mass[4] = 40.0  # child2

    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = 10.0
    body_mass_np[0, 2] = 20.0
    body_mass_np[0, 3] = 30.0
    body_mass_np[0, 4] = 40.0
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.body_subtreemass.numpy()[0], mjm.body_subtreemass, "body_subtreemass")

    # Verify: root=10+(20+30)+40=100, child1=20+30=50, grandchild1=30, child2=40
    np.testing.assert_allclose(m.body_subtreemass.numpy()[0, 1], 100.0, rtol=1e-6)
    np.testing.assert_allclose(m.body_subtreemass.numpy()[0, 2], 50.0, rtol=1e-6)
    np.testing.assert_allclose(m.body_subtreemass.numpy()[0, 3], 30.0, rtol=1e-6)
    np.testing.assert_allclose(m.body_subtreemass.numpy()[0, 4], 40.0, rtol=1e-6)

  def test_set_const_camera_light_positions(self):
    """Test camera and light reference position computations."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="body1" pos="1 2 3">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="sphere" size="0.1" mass="1.0"/>
          <camera name="cam1" pos="0.1 0.2 0.3"/>
          <light name="light1" pos="0.4 0.5 0.6" dir="0 0 -1"/>
        </body>
        <body name="body2" pos="4 5 6">
          <joint name="j2" type="hinge" axis="0 0 1"/>
          <geom name="g2" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.cam_pos0.numpy()[0, 0], mjm.cam_pos0[0], "cam_pos0")
    _assert_eq(m.cam_poscom0.numpy()[0, 0], mjm.cam_poscom0[0], "cam_poscom0")
    _assert_eq(m.cam_mat0.numpy()[0, 0].flatten(), mjm.cam_mat0[0], "cam_mat0")
    _assert_eq(m.light_pos0.numpy()[0, 0], mjm.light_pos0[0], "light_pos0")
    _assert_eq(m.light_poscom0.numpy()[0, 0], mjm.light_poscom0[0], "light_poscom0")
    _assert_eq(m.light_dir0.numpy()[0, 0], mjm.light_dir0[0], "light_dir0")

  def test_set_const_idempotent(self):
    """Test calling set_const twice gives same results."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="link1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <body name="link2" pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom name="g2" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor1" joint="j1" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    mjwarp.set_const(m, d)
    # TODO(team): remove version check after mujoco>=3.17.0 is required
    if util_pkg.check_version("mujoco>=3.17.0"):
      output_fields = sorted(_setconst_output_fields())
    else:
      output_fields = ["actuator_acc0", "body_invweight0", "body_subtreemass", "dof_invweight0"]
    saved = {f: getattr(m, f).numpy().copy() for f in output_fields}

    mjwarp.set_const(m, d)
    for f in output_fields:
      _assert_eq(getattr(m, f).numpy(), saved[f], f)

  @parameterized.parameters(
    (mjwarp.SetConstGroup.ALL,),
    (mjwarp.SetConstGroup.FIXED,),
    (mjwarp.SetConstGroup.KIN0,),
    (mjwarp.SetConstGroup.INERT0,),
    (mjwarp.SetConstGroup.KIN0 | mjwarp.SetConstGroup.INERT0,),
  )
  def test_set_const_restore(self, groups):
    """Test set_const functions restore Data fields to correspond to d.qpos."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <site name="site1" pos="0.2 0 1"/>
        <body pos="0 0 1">
          <joint name="joint1" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.04" mass="1.0"/>
          <site name="site2" pos="0 0 0.5"/>
        </body>
      </worldbody>
      <tendon>
        <spatial armature="0.1">
          <site site="site1"/>
          <site site="site2"/>
        </spatial>
      </tendon>
      <actuator>
        <motor joint="joint1" ctrlrange="-10 10" ctrllimited="true"/>
      </actuator>
    </mujoco>
    """
    )

    qpos_custom = np.array([[1.23]], dtype=np.float32)
    wp.copy(d.qpos, wp.array(qpos_custom))
    mjwarp.forward(m, d)

    fields = [name for name, val in vars(d).items() if isinstance(val, wp.array) and not name.startswith("_")]
    saved_states = {f: getattr(d, f).numpy().copy() for f in fields}

    # Execute the target function
    mjwarp.set_const(m, d, groups=groups)

    # Verify matching with initial state
    for f in fields:
      _assert_eq(getattr(d, f).numpy(), saved_states[f], f"{f} after {groups}")

  def test_set_const_full_pipeline(self):
    """Test complete set_const matches MuJoCo for complex model."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="torso" pos="0 0 1">
          <freejoint/>
          <geom name="torso_geom" type="box" size="0.1 0.2 0.3" mass="10.0"/>
          <body name="arm" pos="0.2 0 0">
            <joint name="shoulder" type="ball"/>
            <geom name="arm_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" mass="2.0"/>
            <site name="arm_site" pos="0.15 0 0"/>
            <body name="forearm" pos="0.3 0 0">
              <joint name="elbow" type="hinge" axis="0 1 0"/>
              <geom name="forearm_geom" type="capsule" fromto="0 0 0 0.25 0 0" size="0.04" mass="1.0"/>
              <site name="hand_site" pos="0.25 0 0"/>
            </body>
          </body>
          <body name="leg" pos="0 0 -0.3">
            <joint name="hip" type="hinge" axis="0 1 0"/>
            <geom name="leg_geom" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.06" mass="3.0"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="arm_tendon">
          <site site="arm_site"/>
          <site site="hand_site"/>
        </spatial>
      </tendon>
      <actuator>
        <motor name="elbow_motor" joint="elbow" gear="1"/>
        <motor name="hip_motor" joint="hip" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    mjm.qpos0[7:11] = [0.9, 0.1, 0.1, 0.1]
    mjm.qpos0[11] = 0.5
    mjm.qpos0[12] = 0.3

    qpos0_np = m.qpos0.numpy()
    qpos0_np[0, 7:11] = [0.9, 0.1, 0.1, 0.1]
    qpos0_np[0, 11] = 0.5
    qpos0_np[0, 12] = 0.3
    wp.copy(m.qpos0, wp.array(qpos0_np, dtype=m.qpos0.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    # TODO(team): remove version check after mujoco>=3.17.0 is required
    if util_pkg.check_version("mujoco>=3.17.0"):
      for f in sorted(_setconst_output_fields()):
        mjm_val = getattr(mjm, f)
        _assert_eq(getattr(m, f).numpy()[0].reshape(mjm_val.shape), mjm_val, f)
    else:
      _assert_eq(m.body_subtreemass.numpy()[0], mjm.body_subtreemass, "body_subtreemass")
      _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")
      _assert_eq(m.tendon_invweight0.numpy()[0], mjm.tendon_invweight0, "tendon_invweight0")
      _assert_eq(m.tendon_length0.numpy()[0], mjm.tendon_length0, "tendon_length0")
      _assert_eq(m.actuator_acc0.numpy()[0], mjm.actuator_acc0, "actuator_acc0")
      for i in range(mjm.nbody):
        _assert_eq(m.body_invweight0.numpy()[0, i], mjm.body_invweight0[i], f"body_invweight0[{i}]")
    _assert_eq(m.tendon_lengthspring.numpy()[0], mjm.tendon_lengthspring, "tendon_lengthspring")

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  @parameterized.parameters(
    (mjwarp.SetConstGroup.ALL,),
    (mjwarp.SetConstGroup.FIXED,),
    (mjwarp.SetConstGroup.KIN0,),
    (mjwarp.SetConstGroup.INERT0,),
    (mjwarp.SetConstGroup.KIN0 | mjwarp.SetConstGroup.INERT0,),
  )
  def test_set_const_graph_capture(self, groups):
    """Test that set_const functions are compatible with CUDA graph capture."""
    _, _, m, d = test_data.fixture("humanoid/humanoid.xml", keyframe=0)

    with wp.ScopedCapture() as capture:
      mjwarp.set_const(m, d, groups=groups)

    wp.capture_launch(capture.graph)

  def test_set_const_actuator_acc0_per_world(self):
    """Test actuator_acc0 has 2D shape [nworld, nu] and values match MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="link1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor1" joint="j1" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    acc0_np = m.actuator_acc0.numpy()
    self.assertEqual(acc0_np.ndim, 2)
    self.assertEqual(acc0_np.shape, (1, mjm.nu))
    _assert_eq(acc0_np[0], mjm.actuator_acc0, "actuator_acc0")

  @parameterized.parameters(1, 2)
  def test_set_const_compiled_constants(self, nworld):
    """Test set_const preserves compiled constants as in mj_setConst."""
    mjm, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="slider" type="slide" springref=".25"/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        </body>
      </worldbody>
      <tendon>
        <fixed stiffness="1">
          <joint joint="slider" coef="2"/>
        </fixed>
      </tendon>
      <actuator>
        <position joint="slider" kp="4" dampratio="1"/>
      </actuator>
    </mujoco>
    """,
      nworld=nworld,
      batch_sizes={
        "body_mass": nworld,
        "qpos_spring": nworld,
        "dof_invweight0": nworld,
        "actuator_biasprm": nworld,
        "tendon_lengthspring": nworld,
      },
    )

    # compiled: damping 2 * sqrt(kp * mass), length 2 * springref
    _assert_eq(mjm.actuator_biasprm[0, 2], -4, "compiled biasprm[2]")
    _assert_eq(mjm.tendon_lengthspring[0], [0.5, 0.5], "compiled lengthspring")

    # their inputs change: the values stay
    masses = [4.0, 16.0][:nworld]
    body_mass = m.body_mass.numpy()
    qpos_spring = m.qpos_spring.numpy()
    for w in range(nworld):
      body_mass[w, 1] = masses[w]
      qpos_spring[w, 0] = 0.5
    m.body_mass.assign(body_mass)
    m.qpos_spring.assign(qpos_spring)
    mjwarp.set_const(m, d)
    for w in range(nworld):
      _assert_eq(m.dof_invweight0.numpy()[w, 0], 1 / masses[w], f"dof_invweight0_world_{w}")
      _assert_eq(m.actuator_biasprm.numpy()[w, 0, 2], -4, f"biasprm[2]_world_{w}")
      _assert_eq(m.tendon_lengthspring.numpy()[w, 0], [0.5, 0.5], f"lengthspring_world_{w}")

    # the conventions are not resolved, as in mj_setConst
    biasprm = m.actuator_biasprm.numpy()
    biasprm[:, 0, 2] = 1
    m.actuator_biasprm.assign(biasprm)
    m.tendon_lengthspring.assign(np.full((nworld, 1, 2), -1.0))
    mjwarp.set_const(m, d)
    mjm.actuator_biasprm[0, 2] = 1
    mjm.tendon_lengthspring[0] = -1
    mujoco.mj_setConst(mjm, mujoco.MjData(mjm))
    for w in range(nworld):
      _assert_eq(m.actuator_biasprm.numpy()[w, 0], mjm.actuator_biasprm[0], f"biasprm_world_{w}")
      _assert_eq(m.tendon_lengthspring.numpy()[w, 0], mjm.tendon_lengthspring[0], f"lengthspring_world_{w}")
    _assert_eq(mjm.actuator_biasprm[0, 2], 1, "mj_setConst biasprm[2]")
    _assert_eq(mjm.tendon_lengthspring[0], [-1, -1], "mj_setConst lengthspring")

  def test_set_length_range_joint_limited(self):
    """Test set_length_range for joint-limited actuator matches joint range."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j1" type="hinge" axis="0 0 1" limited="true" range="-90 90"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="j1" gear="2"/>
      </actuator>
    </mujoco>
    """
    )

    set_length_range(m, d)

    lr_np = m.actuator_lengthrange.numpy()
    # range stored in radians: [-pi/2, pi/2], gear=2 => [-pi, pi]
    expected_lo = mjm.jnt_range[0, 0] * 2.0
    expected_hi = mjm.jnt_range[0, 1] * 2.0
    np.testing.assert_allclose(lr_np[0, 0, 0], expected_lo, atol=1e-5)
    np.testing.assert_allclose(lr_np[0, 0, 1], expected_hi, atol=1e-5)

  def test_set_length_range_tendon_limited(self):
    """Test set_length_range for tendon-limited actuator matches tendon range."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <site name="s1" pos="0.1 0 0"/>
          <body pos="0.5 0 0">
            <joint type="hinge" axis="0 0 1"/>
            <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
            <site name="s2" pos="0.4 0 0"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="t1" limited="true" range="0.1 0.5">
          <site site="s1"/>
          <site site="s2"/>
        </spatial>
      </tendon>
      <actuator>
        <motor tendon="t1" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    set_length_range(m, d)

    lr_np = m.actuator_lengthrange.numpy()
    # tendon range is not in degrees, so [0.1, 0.5] stays as-is with gear=1
    expected_lo = mjm.tendon_range[0, 0]
    expected_hi = mjm.tendon_range[0, 1]
    np.testing.assert_allclose(lr_np[0, 0, 0], expected_lo, atol=1e-5)
    np.testing.assert_allclose(lr_np[0, 0, 1], expected_hi, atol=1e-5)

  @parameterized.parameters(1, 2)
  def test_set_const_geom_bound(self, nworld):
    """Test geom bounds follow geom_size and hfield_size as in mj_setConst."""
    mjm, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <asset>
        <hfield name="terrain" nrow="2" ncol="2" size="1 1 .5 .1"/>
      </asset>
      <worldbody>
        <geom type="plane" size="5 5 .1"/>
        <geom type="hfield" hfield="terrain"/>
        <body>
          <freejoint/>
          <geom size=".1"/>
          <geom type="capsule" size=".1 .2"/>
          <geom type="cylinder" size=".1 .3"/>
          <geom type="ellipsoid" size=".1 .2 .3"/>
          <geom type="box" size=".1 .2 .3" pos=".5 0 0"/>
        </body>
      </worldbody>
    </mujoco>
    """,
      nworld=nworld,
      batch_sizes={"geom_size": nworld, "geom_aabb": nworld, "geom_rbound": nworld},
    )

    scales = [1.5, 0.5][:nworld]
    size = m.geom_size.numpy()
    for w in range(nworld):
      size[w, 2:] *= scales[w]
    m.geom_size.assign(size)
    hfield_size = m.hfield_size.numpy() * 2.0
    m.hfield_size.assign(hfield_size)

    mjwarp.set_const(m, d)

    aabb = m.geom_aabb.numpy().reshape(nworld, -1, 6)
    rbound = m.geom_rbound.numpy()
    for w in range(nworld):
      mjm_w = copy.copy(mjm)
      mjm_w.geom_size[2:] *= scales[w]
      mjm_w.hfield_size[:] = hfield_size
      mujoco.mj_setConst(mjm_w, mujoco.MjData(mjm_w))
      _assert_eq(m.geom_size.numpy()[w], mjm_w.geom_size, f"geom_size_world_{w}")
      _assert_eq(aabb[w, 0] / mujoco.mjMAXVAL, mjm_w.geom_aabb[0] / mujoco.mjMAXVAL, f"geom_aabb_plane_world_{w}")
      _assert_eq(aabb[w, 1:], mjm_w.geom_aabb[1:], f"geom_aabb_world_{w}")
      _assert_eq(rbound[w], mjm_w.geom_rbound, f"geom_rbound_world_{w}")

    if nworld == 2:
      self.assertFalse(np.allclose(rbound[0], rbound[1]))

  def test_set_const_ignores_contacts(self):
    """Test actuator_acc0 of an adhesion actuator does not depend on the contacts in d."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <geom type="plane" size="1 1 .1"/>
        <body name="box" pos="0 0 .1">
          <freejoint/>
          <geom type="box" size=".1 .1 .1"/>
        </body>
      </worldbody>
      <actuator>
        <adhesion body="box" ctrlrange="0 1"/>
      </actuator>
    </mujoco>
    """
    )
    nacon = d.nacon.numpy()[0]
    self.assertGreater(nacon, 0)  # the box rests on the plane

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d, groups=mjwarp.SetConstGroup.KIN0 | mjwarp.SetConstGroup.INERT0)

    _assert_eq(m.actuator_acc0.numpy()[0], mjm.actuator_acc0, "actuator_acc0")
    self.assertEqual(d.nacon.numpy()[0], nacon)  # the contacts are restored

  # TODO(team): remove version check after mujoco>=3.17.0 is required
  @absltest.skipIf(not util_pkg.check_version("mujoco>=3.17.0"), "Requires MuJoCo >= 3.17.0")
  @parameterized.parameters(1, 2)
  def test_set_const_groups(self, nworld):
    """Test set_const with FIXED, KIN0, and INERT0 groups recomputes only their group outputs."""
    # Batch all batched setconst fields except cam_quat and light_pos to also test mixed batching
    batch_sizes = {
      name: nworld
      for name in mujoco.MjModel._setconst_fields
      if _is_batched_model_field(name) and name not in ("cam_quat", "light_pos")
    }
    mjm, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <asset>
        <hfield name="terrain" nrow="2" ncol="2" size="1 1 .5 .1"/>
      </asset>
      <worldbody>
        <geom type="hfield" hfield="terrain"/>
        <camera pos="0.1 0.1 1.0" euler="5 10 15"/>
        <light pos="0.0 0.0 2.0" dir="0.1 -0.2 -1"/>
        <body name="body1" pos="1 2 3">
          <joint name="j1" axis="0 1 0"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0"/>
          <geom name="wrap_cyl" type="cylinder" pos="0.25 0.02 0" size="0.08 0.1"/>
          <site name="s1" pos="0.1 0 0"/>
          <site name="sidesite" pos="0.25 0.15 0"/>
          <camera pos="0.1 0.2 0.3" euler="10 20 30" mode="trackcom" target="body2"/>
          <light pos="0.4 0.5 0.6" dir="1 2 -3" mode="targetbodycom" target="body2"/>
          <body name="body2" pos="0.5 0 0">
            <joint name="j2"/>
            <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="2.0"/>
            <site name="s2" pos="0.4 0.2 0"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial armature="0.1">
          <site site="s1"/>
          <site site="s2"/>
        </spatial>
        <spatial name="t_wrap">
          <site site="s1"/>
          <geom geom="wrap_cyl" sidesite="sidesite"/>
          <site site="s2"/>
        </spatial>
      </tendon>
      <equality>
        <connect body1="body1" body2="body2" anchor="0.2 0 0"/>
      </equality>
      <actuator>
        <motor joint="j1"/>
        <motor joint="j2"/>
        <motor tendon="t_wrap"/>
      </actuator>
    </mujoco>
    """,
      nworld=nworld,
      batch_sizes=batch_sizes,
    )

    mjms = []
    for w in range(nworld):
      mjm_w = copy.copy(mjm)
      mjm_w.body_mass[1:] *= 1.5 + w
      mjm_w.geom_size[1:] *= 1.2 - 0.4 * w
      mjm_w.hfield_size[:] *= 1.5
      mjm_w.qpos0[:] = [0.25 - 0.65 * w, -0.35 + 0.55 * w]
      mjm_w.body_quat[1] = [1.0, 0.0, 0.0, 0.0] if w == 0 else [0.96, 0.28, 0.0, 0.0]
      mjm_w.body_pos[2, 0] = 0.5 + 0.2 * w
      mjm_w.cam_pos[:] += 0.05 - 0.13 * w
      mjm_w.light_dir[:] += [0.2 + 0.1 * w, -0.1 * w, 0.0]
      mjm_w.eq_data[0, :3] = [0.25 - 0.1 * w, 0.05, 0.0]
      mjm_w.eq_data[0, 3:6] = 0.0
      mjm_w.dof_armature[:] = [0.1 + 0.2 * w, 0.2 + 0.3 * w]
      mjm_w.tendon_armature[0] = 0.1 + 0.2 * w
      mjms.append(mjm_w)

    m.hfield_size.assign(mjms[0].hfield_size)
    for f in (
      "body_mass",
      "geom_size",
      "qpos0",
      "body_quat",
      "body_pos",
      "cam_pos",
      "light_dir",
      "eq_data",
      "dof_armature",
      "tendon_armature",
    ):
      getattr(m, f).assign(np.stack([getattr(mjm_w, f) for mjm_w in mjms]))

    for mjm_w in mjms:
      mujoco.mj_setConst(mjm_w, mujoco.MjData(mjm_w))

    all_pure_outputs = _setconst_output_fields(pure_only=True)

    # Verify each group's outputs match mj_setConstGroups while other outputs stay inf
    for group_name, group_flag in _GROUPS:
      idx = mujoco.MjModel._setconst_groups.index(group_name)
      self.assertEqual(group_flag, 1 << idx)
      group_outputs = _setconst_output_fields(group_name)
      group_pure_outputs = _setconst_output_fields(group_name, pure_only=True)
      poisoned_other_outputs = {
        f for f in (all_pure_outputs - group_pure_outputs) if not (mujoco.MjModel._setconst_fields[f][idx] & mujoco.mjSC_IN)
      }

      for f in group_pure_outputs | poisoned_other_outputs:
        arr = getattr(m, f)
        arr.assign(np.full(arr.numpy().shape, np.inf, dtype=np.float32))

      mjwarp.set_const(m, d, groups=group_flag)

      for f in sorted(group_outputs):
        m_val = getattr(m, f).numpy()
        for w in range(nworld):
          mjm_val = getattr(mjms[w], f)
          _assert_eq(m_val[w].reshape(mjm_val.shape), mjm_val, f"{group_name}:{f}_w{w}")
        if nworld == 2:
          self.assertFalse(np.allclose(m_val[0], m_val[1]), f"{group_name}:{f} should differ across worlds")

      for f in sorted(poisoned_other_outputs):
        self.assertTrue(np.isinf(getattr(m, f).numpy()).all(), f"{group_name} modified unrelated output {f}")

  # TODO(team): remove version check after mujoco>=3.17.0 is required
  @absltest.skipIf(not util_pkg.check_version("mujoco>=3.17.0"), "Requires MuJoCo >= 3.17.0")
  def test_set_const_group_inputs(self):
    """Test observed set_const group inputs are a subset of MjModel._setconst_fields."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <asset>
        <hfield name="terrain" nrow="2" ncol="2" size="1 1 .5 .1"/>
      </asset>
      <worldbody>
        <geom type="hfield" hfield="terrain"/>
        <camera pos="0.1 0.1 1.0" euler="5 10 15"/>
        <light pos="0.0 0.0 2.0" dir="0.1 -0.2 -1"/>
        <body name="body1" pos="1 2 3">
          <joint name="j1" axis="0 1 0"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0"/>
          <geom name="wrap_cyl" type="cylinder" pos="0.25 0.02 0" size="0.08 0.1"/>
          <site name="s1" pos="0.1 0 0"/>
          <site name="sidesite" pos="0.25 0.15 0"/>
          <camera pos="0.1 0.2 0.3" euler="10 20 30" mode="trackcom" target="body2"/>
          <light pos="0.4 0.5 0.6" dir="1 2 -3" mode="targetbodycom" target="body2"/>
          <body name="body2" pos="0.5 0 0">
            <joint name="j2"/>
            <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="2.0"/>
            <site name="s2" pos="0.4 0.2 0"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial armature="0.1">
          <site site="s1"/>
          <site site="s2"/>
        </spatial>
        <spatial name="t_wrap">
          <site site="s1"/>
          <geom geom="wrap_cyl" sidesite="sidesite"/>
          <site site="s2"/>
        </spatial>
      </tendon>
      <equality>
        <connect body1="body1" body2="body2" anchor="0.2 0 0"/>
      </equality>
      <actuator>
        <motor joint="j1"/>
        <motor joint="j2"/>
        <motor tendon="t_wrap"/>
      </actuator>
    </mujoco>
    """
    )

    mjwarp.set_const(m, d)
    all_outputs = _setconst_output_fields()
    all_pure_outputs = _setconst_output_fields(pure_only=True)
    baseline = {
      f: arr.numpy().copy()
      for f, arr in vars(m).items()
      if f in mujoco.MjModel._setconst_fields
      and isinstance(arr, wp.array)
      and arr.size > 0
      and np.issubdtype(arr.numpy().dtype, np.floating)
    }

    def _observed_groups(field_name: str) -> set[str]:
      orig = baseline[field_name]
      if "quat" in field_name:
        perturbed = orig + np.array([0.0, 0.2, -0.15, 0.1], dtype=np.float32)
        perturbed /= np.linalg.norm(perturbed, axis=-1, keepdims=True)
      else:
        perturbed = orig * np.float32(1.3) + np.float32(0.17)

      observed: set[str] = set()
      for group_name, group_flag in _GROUPS:
        for out in all_outputs | {field_name}:
          getattr(m, out).assign(baseline[out])
        getattr(m, field_name).assign(perturbed)
        group_outputs = _setconst_output_fields(group_name)
        pre = {out: getattr(m, out).numpy().copy() for out in group_outputs}
        mjwarp.set_const(m, d, groups=group_flag)
        if any(not np.allclose(getattr(m, out).numpy(), pre[out], atol=1e-6, rtol=1e-6) for out in group_outputs):
          observed.add(group_name)

      for out in all_outputs | {field_name}:
        getattr(m, out).assign(baseline[out])
      return observed

    observed_by_field: dict[str, set[str]] = {}
    for f in sorted(baseline.keys() - all_pure_outputs):
      observed = _observed_groups(f)
      observed_by_field[f] = observed
      access = mujoco.MjModel._setconst_fields.get(f)
      declared = {
        group_name
        for group_name, _ in _GROUPS
        if access is not None and (access[mujoco.MjModel._setconst_groups.index(group_name)] & mujoco.mjSC_IN)
      }
      self.assertLessEqual(observed, declared, f"Field {f} affected undeclared groups: {observed - declared}")

    # With cylinder tendon wrapping, geom_size affects FIXED, KIN0, and INERT0
    self.assertEqual(observed_by_field["geom_size"], {"fixed", "kin0", "inert0"})

    # Without sphere/cylinder tendon wrapping, geom_size only affects FIXED
    m.wrap_geom_adr = wp.zeros((0,), dtype=int)
    mjwarp.set_const(m, d)
    for out in all_outputs:
      baseline[out] = getattr(m, out).numpy().copy()
    self.assertEqual(_observed_groups("geom_size"), {"fixed"})

  # TODO(team): remove in future version
  def test_set_const_deprecated_aliases(self):
    """Test deprecated set_const_fixed and set_const_0 emit DeprecationWarning."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="body1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" size="0.1" mass="2.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )
    mjm.body_mass[1] = 4.0
    m.body_mass.assign(np.array([[0.0, 4.0]], dtype=np.float32))
    mujoco.mj_setConst(mjm, mjd)

    m.body_subtreemass.fill_(wp.inf)
    with self.assertWarns(DeprecationWarning):
      mjwarp.set_const_fixed(m, d)
    _assert_eq(m.body_subtreemass.numpy()[0], mjm.body_subtreemass, "body_subtreemass")

    m.dof_invweight0.fill_(wp.inf)
    with self.assertWarns(DeprecationWarning):
      mjwarp.set_const_0(m, d)
    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")

  @parameterized.parameters(1, 2)
  def test_set_const_sparse(self, nworld):
    """Test set_const with sparse jacobian and sparse LDL inertia factorization."""
    mjm, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option jacobian="sparse"/>
      <worldbody>
        <body name="root" pos="0 0 1">
          <freejoint/>
          <geom size="0.1" mass="2.0"/>
          <site name="s0" pos="0.1 0 0"/>
          <replicate count="60" offset="0.02 0 0">
            <body pos="0.1 0 0">
              <joint axis="0 1 0"/>
              <geom size="0.1" mass="0.5"/>
            </body>
          </replicate>
          <body name="tip" pos="0 0.2 0">
            <joint name="tip_joint" axis="0 0 1"/>
            <geom size="0.1" mass="0.5"/>
            <site name="s1" pos="0 0.1 0"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial>
          <site site="s0"/>
          <site site="s1"/>
        </spatial>
      </tendon>
      <actuator>
        <motor joint="tip_joint"/>
      </actuator>
    </mujoco>
    """,
      nworld=nworld,
      batch_sizes={
        "body_mass": nworld,
        "body_pos": nworld,
        "body_subtreemass": nworld,
        "dof_invweight0": nworld,
        "body_invweight0": nworld,
        "tendon_length0": nworld,
        "tendon_invweight0": nworld,
        "actuator_acc0": nworld,
      },
    )

    self.assertTrue(m.is_sparse)
    self.assertGreater(d.qLD.shape[1], m.qLD_block_total)

    mjms = []
    for w in range(nworld):
      mjm_w = copy.copy(mjm)
      mjm_w.body_mass[1:] *= 1.5 + w
      mjm_w.body_pos[-1, 0] = 0.2 + 0.3 * w
      mjms.append(mjm_w)

    for f in ("body_mass", "body_pos"):
      getattr(m, f).assign(np.stack([getattr(mjm_w, f) for mjm_w in mjms]))

    for mjm_w in mjms:
      mujoco.mj_setConst(mjm_w, mujoco.MjData(mjm_w))

    output_fields = (
      "body_subtreemass",
      "dof_invweight0",
      "body_invweight0",
      "tendon_length0",
      "tendon_invweight0",
      "actuator_acc0",
    )
    for f in output_fields:
      getattr(m, f).fill_(wp.inf)

    mjwarp.set_const(m, d)

    for f in output_fields:
      m_val = getattr(m, f).numpy()
      for w in range(nworld):
        mjm_val = getattr(mjms[w], f)
        _assert_eq(m_val[w].reshape(mjm_val.shape), mjm_val, f"{f}_w{w}")
      if nworld == 2:
        self.assertFalse(np.allclose(m_val[0], m_val[1]), f"{f} should differ across worlds")


if __name__ == "__main__":
  wp.init()
  absltest.main()
