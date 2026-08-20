# Copyright 2025 The Newton Developers
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

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src.set_const import set_length_range


def _assert_eq(a, b, name):
  tol = 5e-4
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


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

  def test_set_const_eq_data_weld(self):
    """Test set_const recomputes eq_data for weld constraints."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option gravity="0 0 0">
        <flag contact="disable"/>
      </option>
      <worldbody>
        <body name="b1" pos="0.1 0 0">
          <joint type="hinge" axis="0 0 1" ref="0.3"/>
          <geom type="sphere" size="0.05" mass="1"/>
        </body>
        <body name="b2" pos="0.4 0 0" euler="0 0 20">
          <joint type="hinge" axis="0 1 0" ref="-0.2"/>
          <geom type="sphere" size="0.05" mass="1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2" anchor="0.1 0.2 0.3"/>
      </equality>
    </mujoco>
    """
    )

    # case 1: quaternion data cleared: set_const recomputes relpose and anchor offset
    new_data = mjm.eq_data[0].copy()
    new_data[0:3] = [0.15, 0.1, 0.25]
    new_data[3:10] = 0.0
    mjm.eq_data[0] = new_data
    eq_data = m.eq_data.numpy()
    eq_data[0, 0] = new_data
    wp.copy(m.eq_data, wp.array(eq_data, dtype=m.eq_data.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.eq_data.numpy()[0], mjm.eq_data, "eq_data")

    # case 2: user-specified quaternion: set_const only normalizes it
    new_data = mjm.eq_data[0].copy()
    new_data[6:10] = [2.0, 0.0, 2.0, 0.0]
    mjm.eq_data[0] = new_data
    eq_data = m.eq_data.numpy()
    eq_data[0, 0] = new_data
    wp.copy(m.eq_data, wp.array(eq_data, dtype=m.eq_data.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.eq_data.numpy()[0], mjm.eq_data, "eq_data")

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

  @parameterized.named_parameters(
    dict(testcase_name="dense", jacobian="dense"),
    dict(testcase_name="sparse", jacobian="sparse"),
  )
  def test_set_const_meaninertia(self, jacobian):
    """Test meaninertia computation matches MuJoCo after qpos0/mass changes."""
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
    <mujoco>
      <option jacobian="{jacobian}"/>
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
    </mujoco>
    """
    )

    # Test initial value matches
    _assert_eq(m.stat.meaninertia.numpy()[0], mjm.stat.meaninertia, "meaninertia initial")

    # Modify qpos0 and verify meaninertia updates
    new_qpos0 = np.array([0.5, 0.3])
    mjm.qpos0[:] = new_qpos0
    qpos0_np = m.qpos0.numpy()
    qpos0_np[0, :] = new_qpos0
    wp.copy(m.qpos0, wp.array(qpos0_np, dtype=m.qpos0.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.stat.meaninertia.numpy()[0], mjm.stat.meaninertia, "meaninertia after qpos0 change")

    # Modify body mass and verify meaninertia updates
    new_mass = 3.0
    mjm.body_mass[1] = new_mass
    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = new_mass
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.stat.meaninertia.numpy()[0], mjm.stat.meaninertia, "meaninertia after mass change")

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

    mjwarp.set_const_0(m, d)

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
    dof_invweight0_1 = m.dof_invweight0.numpy().copy()
    body_invweight0_1 = m.body_invweight0.numpy().copy()
    body_subtreemass_1 = m.body_subtreemass.numpy().copy()
    actuator_acc0_1 = m.actuator_acc0.numpy().copy()

    mjwarp.set_const(m, d)
    _assert_eq(m.dof_invweight0.numpy(), dof_invweight0_1, "dof_invweight0")
    _assert_eq(m.body_invweight0.numpy(), body_invweight0_1, "body_invweight0")
    _assert_eq(m.body_subtreemass.numpy(), body_subtreemass_1, "body_subtreemass")
    _assert_eq(m.actuator_acc0.numpy(), actuator_acc0_1, "actuator_acc0")

  def test_set_const_spring(self):
    """Test set_const_spring resolves tendon_lengthspring."""
    xml = """
    <mujoco>
      <worldbody>
        <body pos="0 0 1">
          <freejoint/>
          <geom type="box" size="0.1 0.2 0.3" mass="10.0"/>
          <body pos="0.2 0 0">
            <joint type="ball"/>
            <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" mass="2.0"/>
            <site name="arm_site" pos="0.15 0 0"/>
            <body pos="0.3 0 0">
              <joint type="hinge" axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.04" mass="1.0"/>
              <site name="hand_site" pos="0.25 0 0"/>
            </body>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial>
          <site site="arm_site"/>
          <site site="hand_site"/>
        </spatial>
      </tendon>
    </mujoco>
    """

    # Run with default qpos_spring
    mjm_default, mjd_default, m_default, d_default = test_data.fixture(xml=xml)
    mjm_default.tendon_lengthspring[:] = -1.0
    m_default.tendon_lengthspring.assign(
      np.full((m_default.tendon_lengthspring.shape[0], m_default.tendon_lengthspring.shape[1], 2), -1.0)
    )
    mujoco.mj_setConst(mjm_default, mjd_default)
    mjwarp.set_const(m_default, d_default)
    lengthspring_default = m_default.tendon_lengthspring.numpy().copy()

    # Run with modified qpos_spring
    mjm, mjd, m, d = test_data.fixture(xml=xml)
    mjm.qpos_spring[11] = 0.5

    qpos_spring_np = m.qpos_spring.numpy()
    qpos_spring_np[0, 11] = 0.5
    m.qpos_spring.assign(qpos_spring_np)

    mjm.tendon_lengthspring[:] = -1.0
    m.tendon_lengthspring.assign(np.full((m.tendon_lengthspring.shape[0], m.tendon_lengthspring.shape[1], 2), -1.0))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    # Verify matching with MuJoCo
    _assert_eq(m.tendon_lengthspring.numpy()[0], mjm.tendon_lengthspring, "tendon_lengthspring")

    # Verify that modified spring length differs from the default spring length
    self.assertFalse(np.allclose(lengthspring_default, m.tendon_lengthspring.numpy()))

  @parameterized.named_parameters(
    dict(
      testcase_name="set_const",
      func_name="set_const",
      fields=[
        "xpos",
        "xquat",
        "xmat",
        "xipos",
        "ximat",
        "xanchor",
        "xaxis",
        "geom_xpos",
        "geom_xmat",
        "site_xpos",
        "site_xmat",
        "cam_xpos",
        "cam_xmat",
        "light_xpos",
        "light_xdir",
        "flexvert_xpos",
        "flexedge_J",
        "flexedge_length",
        "flexedge_velocity",
        "subtree_com",
        "cinert",
        "cdof",
        "ten_length",
        "ten_J",
        "wrap_obj",
        "wrap_xpos",
        "ten_wrapadr",
        "ten_wrapnum",
        "crb",
        "M",
        "qLD",
        "qLDiagInv",
        "actuator_length",
        "actuator_moment",
        "moment_rownnz",
        "moment_rowadr",
        "moment_colind",
      ],
    ),
    dict(
      testcase_name="set_const_0",
      func_name="set_const_0",
      fields=[
        "xpos",
        "xquat",
        "xmat",
        "xipos",
        "ximat",
        "xanchor",
        "xaxis",
        "geom_xpos",
        "geom_xmat",
        "site_xpos",
        "site_xmat",
        "cam_xpos",
        "cam_xmat",
        "light_xpos",
        "light_xdir",
        "flexvert_xpos",
        "flexedge_J",
        "flexedge_length",
        "flexedge_velocity",
        "subtree_com",
        "cinert",
        "cdof",
        "ten_length",
        "ten_J",
        "wrap_obj",
        "wrap_xpos",
        "ten_wrapadr",
        "ten_wrapnum",
        "crb",
        "M",
        "qLD",
        "qLDiagInv",
        "actuator_length",
        "actuator_moment",
        "moment_rownnz",
        "moment_rowadr",
        "moment_colind",
      ],
    ),
    dict(
      testcase_name="set_const_spring",
      func_name="set_const_spring",
      fields=[
        "xpos",
        "xquat",
        "xmat",
        "xanchor",
        "xaxis",
        "geom_xpos",
        "geom_xmat",
        "site_xpos",
        "site_xmat",
        "subtree_com",
        "cdof",
        "ten_length",
        "ten_J",
        "actuator_length",
        "actuator_moment",
        "moment_rownnz",
        "moment_rowadr",
        "moment_colind",
      ],
    ),
  )
  def test_set_const_restore(self, func_name, fields):
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
        <spatial>
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

    saved_states = {f: getattr(d, f).numpy().copy() for f in fields}

    # Execute the target function
    getattr(mjwarp, func_name)(m, d)

    # Verify matching with initial state
    for f in fields:
      _assert_eq(getattr(d, f).numpy(), saved_states[f], f"{f} after {func_name}")

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

    _assert_eq(m.body_subtreemass.numpy()[0], mjm.body_subtreemass, "body_subtreemass")
    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")
    _assert_eq(m.tendon_invweight0.numpy()[0], mjm.tendon_invweight0, "tendon_invweight0")
    _assert_eq(m.tendon_length0.numpy()[0], mjm.tendon_length0, "tendon_length0")
    _assert_eq(m.actuator_acc0.numpy()[0], mjm.actuator_acc0, "actuator_acc0")
    _assert_eq(m.tendon_lengthspring.numpy()[0], mjm.tendon_lengthspring, "tendon_lengthspring")

    for i in range(mjm.nbody):
      _assert_eq(m.body_invweight0.numpy()[0, i], mjm.body_invweight0[i], f"body_invweight0[{i}]")

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  def test_set_const_graph_capture(self):
    """Test that set_const is compatible with CUDA graph capture."""
    _, _, m, d = test_data.fixture("humanoid/humanoid.xml", keyframe=0)

    with wp.ScopedCapture() as capture:
      mjwarp.set_const(m, d)

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

  def test_set_const_dampratio(self):
    """Test dampratio resolution for position actuator matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <body pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position joint="j1" kp="100" dampratio="1.0"/>
        <position joint="j2" kp="50" dampratio="0.5"/>
      </actuator>
    </mujoco>
    """
    )

    # Set new dampratio values (positive biasprm[2]) to exercise resolution
    new_dampratio = [2.0, 0.8]
    for i in range(mjm.nu):
      mjm.actuator_biasprm[i, 2] = new_dampratio[i]
    mujoco.mj_setConst(mjm, mjd)

    bp = m.actuator_biasprm.numpy()
    for i in range(mjm.nu):
      bp[0, i, 2] = new_dampratio[i]
    wp.copy(m.actuator_biasprm, wp.array(bp, dtype=m.actuator_biasprm.dtype))
    mjwarp.set_const(m, d)

    biasprm_np = m.actuator_biasprm.numpy()
    for i in range(mjm.nu):
      _assert_eq(
        biasprm_np[0, i, 2],
        mjm.actuator_biasprm[i, 2],
        f"actuator_biasprm[{i}][2]",
      )

  def test_set_const_dampratio_explicit_kv(self):
    """Test actuator with explicit negative kv is NOT modified by dampratio."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
        </body>
      </worldbody>
      <actuator>
        <general joint="j1" gainprm="100"
                 biastype="affine" biasprm="0 -100 -10"/>
      </actuator>
    </mujoco>
    """
    )

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    biasprm_np = m.actuator_biasprm.numpy()
    _assert_eq(
      biasprm_np[0, 0, 2],
      mjm.actuator_biasprm[0, 2],
      "actuator_biasprm[0][2]",
    )

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


# TODO(team): test set_const_0 sparse


if __name__ == "__main__":
  wp.init()
  absltest.main()
