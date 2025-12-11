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

"""Tests for io functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp
from mujoco_warp import test_data


def _assert_eq(a, b, name):
  tol = 5e-4
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


# NOTE: modify io_jax_test _IO_TEST_MODELS if changed here.
_IO_TEST_MODELS = (
  "pendula.xml",
  "collision_sdf/tactile.xml",
  "flex/floppy.xml",
  "actuation/tendon_force_limit.xml",
  "hfield/hfield.xml",
)


class IOTest(parameterized.TestCase):
  def test_make_put_data(self):
    """Tests that make_data and put_data are producing the same shapes for all arrays."""
    mjm, _, _, d = test_data.fixture("pendula.xml")
    md = mjwarp.make_data(mjm, nconmax=512, njmax=512)

    # same number of fields
    self.assertEqual(len(d.__dict__), len(md.__dict__))

    # test shapes for all arrays
    for attr, val in md.__dict__.items():
      if isinstance(val, wp.array):
        self.assertEqual(val.shape, getattr(d, attr).shape, f"{attr} shape mismatch")

  def test_get_data_into_m(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body pos="0 0 0" >
            <geom type="box" pos="0 0 0" size=".5 .5 .5" />
            <joint type="hinge" />
          </body>
          <body pos="0 0 0.1">
            <geom type="sphere" size="0.5"/>
            <freejoint/>
          </body>
        </worldbody>
      </mujoco>
    """)

    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    mjd_ref = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd_ref)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mjd.qLD.fill(-123)
    mjd.qM.fill(-123)

    mjwarp.get_data_into(mjd, mjm, d)
    np.testing.assert_allclose(mjd.qLD, mjd_ref.qLD)
    np.testing.assert_allclose(mjd.qM, mjd_ref.qM)

  @parameterized.named_parameters(
    dict(testcase_name="nworld=1", nworld=1, world_id=0),
    dict(testcase_name="nworld=2_world_id=1", nworld=2, world_id=1),
  )
  def test_get_data_into(self, nworld, world_id):
    # keyframe=0: ncon=8, nefc=32
    mjm, mjd, _, d = test_data.fixture("humanoid/humanoid.xml", keyframe=0, nworld=nworld)

    # keyframe=2: ncon=0, nefc=0
    mujoco.mj_resetDataKeyframe(mjm, mjd, 2)

    # check that mujoco._functions._realloc_con_efc allocates for contact and efc
    mjwarp.get_data_into(mjd, mjm, d, world_id=world_id)
    self.assertEqual(mjd.ncon, 8)
    self.assertEqual(mjd.nefc, 32)

    # compare fields
    self.assertEqual(d.solver_niter.numpy()[world_id], mjd.solver_niter[0])
    self.assertEqual(d.nacon.numpy()[0], mjd.ncon * nworld)
    self.assertEqual(d.ne.numpy()[world_id], mjd.ne)
    self.assertEqual(d.nf.numpy()[world_id], mjd.nf)
    self.assertEqual(d.nl.numpy()[world_id], mjd.nl)

    for field in [
      "energy",
      "qpos",
      "qvel",
      "act",
      "qacc_warmstart",
      "ctrl",
      "qfrc_applied",
      "xfrc_applied",
      "eq_active",
      "mocap_pos",
      "mocap_quat",
      "qacc",
      "act_dot",
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
      "subtree_com",
      "cdof",
      "cinert",
      "flexvert_xpos",
      "flexedge_length",
      "flexedge_velocity",
      "actuator_length",
      # TODO(team): actuator_moment mjd sparse2dense
      "crb",
      # TODO(team): qLDiagInv sparse factorization
      "ten_velocity",
      "actuator_velocity",
      "cvel",
      "cdof_dot",
      "qfrc_bias",
      "qfrc_spring",
      "qfrc_damper",
      "qfrc_gravcomp",
      "qfrc_fluid",
      "qfrc_passive",
      "subtree_linvel",
      "subtree_angmom",
      "actuator_force",
      "qfrc_actuator",
      "qfrc_smooth",
      "qacc_smooth",
      "qfrc_constraint",
      "qfrc_inverse",
      # TODO(team): qM
      # TODO(team): qLD
      "cacc",
      "cfrc_int",
      "cfrc_ext",
      "ten_length",
      "ten_J",
      "ten_wrapadr",
      "ten_wrapnum",
      "wrap_obj",
      "wrap_xpos",
      "sensordata",
    ]:
      _assert_eq(
        getattr(d, field).numpy()[world_id].reshape(-1),
        getattr(mjd, field).reshape(-1),
        field,
      )

    # contact
    ncon = int(d.nacon.numpy()[0] / nworld)
    for field in [
      "dist",
      "pos",
      "frame",
      "includemargin",
      "friction",
      "solref",
      "solreffriction",
      "solimp",
      "dim",
      "geom",
      # TODO(team): efc_address
    ]:
      _assert_eq(
        getattr(d.contact, field).numpy()[world_id * ncon : world_id * ncon + ncon].reshape(-1),
        getattr(mjd.contact, field).reshape(-1),
        field,
      )

    # efc
    nefc = d.nefc.numpy()[world_id]
    for field in [
      "type",
      "id",
      "pos",
      "margin",
      "D",
      "vel",
      "aref",
      "frictionloss",
      "state",
      "force",
    ]:
      _assert_eq(
        getattr(d.efc, field).numpy()[world_id, :nefc].reshape(-1),
        getattr(mjd, "efc_" + field).reshape(-1),
        field,
      )

  def test_ellipsoid_fluid_model(self):
    mjm = mujoco.MjModel.from_xml_string(
      """
    <mujoco>
      <option density="1.1" viscosity="0.05"/>
      <worldbody>
        <body>
          <geom type="sphere" size=".15" fluidshape="ellipsoid"/>
          <freejoint/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    m = mjwarp.put_model(mjm)

    np.testing.assert_allclose(m.geom_fluid.numpy(), mjm.geom_fluid)
    self.assertTrue(m.opt.has_fluid)

    body_has = m.body_fluid_ellipsoid.numpy()
    self.assertTrue(body_has[mjm.geom_bodyid[0]])
    self.assertFalse(body_has[0])

  def test_jacobian_auto(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option jacobian="auto"/>
        <worldbody>
          <replicate count="11">
          <body>
            <geom type="sphere" size=".1"/>
            <freejoint/>
            </body>
          </replicate>
        </worldbody>
      </mujoco>
    """)
    mjwarp.put_model(mjm)

  def test_put_data_qLD(self):
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="hinge"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    d = mjwarp.put_data(mjm, mjd)
    self.assertTrue((d.qLD.numpy() == 0.0).all())

    mujoco.mj_forward(mjm, mjd)
    mjd.qM[:] = 0.0
    d = mjwarp.put_data(mjm, mjd)
    self.assertTrue((d.qLD.numpy() == 0.0).all())

    mujoco.mj_forward(mjm, mjd)
    mjd.qLD[:] = 0.0
    d = mjwarp.put_data(mjm, mjd)
    self.assertTrue((d.qLD.numpy() == 0.0).all())

  def test_static_geom_collision_with_put_data(self):
    """Test that static geoms (ground plane) work correctly with put_data."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option timestep="0.02"/>
        <worldbody>
          <geom name="ground" type="plane" pos="0 0 0" size="0 0 1"/>
          <body name="box" pos="0 0 0.6">
            <freejoint/>
            <geom name="box" type="box" size="0.5 0.5 0.5"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd, nconmax=16, njmax=16)

    # let the box fall and settle on the ground
    for _ in range(30):
      mjwarp.step(m, d)

    # check that box is above ground
    # box center should be at z ≈ 0.5 when resting on ground
    box_z = d.xpos.numpy()[0, 1, 2]  # world 0, body 1 (box), z coordinate
    self.assertGreater(box_z, 0.4, msg=f"Box fell through ground plane (z={box_z}, should be > 0.4)")

  def test_noslip_solver(self):
    with self.assertRaises(NotImplementedError):
      test_data.fixture(
        xml="""
      <mujoco>
        <option noslip_iterations="1"/>
      </mujoco>
      """
      )

  @parameterized.parameters(*_IO_TEST_MODELS)
  def test_reset_data(self, xml):
    reset_datafield = [
      "ne",
      "nf",
      "nl",
      "nefc",
      "time",
      "energy",
      "qpos",
      "qvel",
      "act",
      "ctrl",
      "eq_active",
      "qfrc_applied",
      "xfrc_applied",
      "qacc",
      "qacc_warmstart",
      "act_dot",
      "sensordata",
      "mocap_pos",
      "mocap_quat",
      "qM",
    ]

    mjm, mjd, m, d = test_data.fixture(xml)
    naconmax = d.naconmax

    # data fields
    for arr in reset_datafield:
      attr = getattr(d, arr)
      if attr.dtype == float:
        attr.fill_(wp.nan)
      else:
        attr.fill_(-1)

    for arr in d.contact.__dataclass_fields__:
      attr = getattr(d.contact, arr)
      if attr.dtype == float:
        attr.fill_(wp.nan)
      else:
        attr.fill_(-1)

    mujoco.mj_resetData(mjm, mjd)

    # set nacon in order to zero all contact memory
    wp.copy(d.nacon, wp.array([naconmax], dtype=int))
    mjwarp.reset_data(m, d)

    for arr in reset_datafield:
      d_arr = getattr(d, arr).numpy()
      for i in range(d_arr.shape[0]):
        di_arr = d_arr[i]
        if arr == "qM":
          di_arr = di_arr.reshape(-1)[: mjd.qM.size]
        _assert_eq(di_arr, getattr(mjd, arr), arr)

    _assert_eq(d.nacon.numpy(), 0, "nacon")

    for arr in d.contact.__dataclass_fields__:
      _assert_eq(getattr(d.contact, arr).numpy(), 0.0, arr)

  def test_reset_data_world(self):
    """Tests per-world reset."""
    _MJCF = """
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(_MJCF)
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm, nworld=2)

    # nonzero values
    qvel = wp.array(np.array([[1.0], [2.0]]), dtype=float)

    wp.copy(d.qvel, qvel)

    # reset both worlds
    mjwarp.reset_data(m, d)

    _assert_eq(d.qvel.numpy()[0], 0.0, "qvel[0]")
    _assert_eq(d.qvel.numpy()[1], 0.0, "qvel[1]")

    wp.copy(d.qvel, qvel)

    # don't reset second world
    reset10 = wp.array(np.array([True, False]), dtype=bool)
    mjwarp.reset_data(m, d, reset=reset10)

    _assert_eq(d.qvel.numpy()[0], 0.0, "qvel[0]")
    _assert_eq(d.qvel.numpy()[1], 2.0, "qvel[1]")

    wp.copy(d.qvel, qvel)

    # don't reset both worlds
    reset00 = wp.array(np.array([False, False], dtype=bool))
    mjwarp.reset_data(m, d, reset=reset00)

    _assert_eq(d.qvel.numpy()[0], 1.0, "qvel[0]")
    _assert_eq(d.qvel.numpy()[1], 2.0, "qvel[1]")

  def test_sdf(self):
    """Tests that an SDF can be loaded."""
    mjm, mjd, m, d = test_data.fixture("collision_sdf/cow.xml")

    self.assertIsInstance(m.oct_aabb, wp.array)
    self.assertEqual(m.oct_aabb.dtype, wp.vec3)
    self.assertEqual(len(m.oct_aabb.shape), 2)
    if m.oct_aabb.size > 0:
      self.assertEqual(m.oct_aabb.shape[1], 2)

  def test_collision_sensor_box_box(self):
    """Tests for collision sensors that are not implemented."""
    with self.assertRaises(NotImplementedError):
      test_data.fixture(
        xml=f"""
      <mujoco>
        <worldbody>
          <geom name="box1" type="box" size=".1 .1 .1"/>
          <geom name="box2" type="box" size=".1 .1 .1"/>
        </worldbody>
        <sensor>
          <distance geom1="box1" geom2="box2"/>
        </sensor>
      </mujoco>
      """
      )

  def test_implicit_integrator_fluid_model(self):
    """Tests for implicit integrator with fluid model."""
    with self.assertRaises(NotImplementedError):
      test_data.fixture(
        xml="""
        <mujoco>
          <option viscosity="1" density="1" integrator="implicitfast"/>
          <worldbody>
            <body>
              <geom type="sphere" size=".1"/>
              <freejoint/>
            </body>
          </worldbody>
        </mujoco>
        """
      )

  def test_plugin(self):
    with self.assertRaises(NotImplementedError):
      test_data.fixture(
        xml="""
      <mujoco>
        <extension>
          <plugin plugin="mujoco.pid"/>
          <plugin plugin="mujoco.sensor.touch_grid"/>
          <plugin plugin="mujoco.elasticity.cable"/>
        </extension>
        <worldbody>
          <geom type="plane" size="10 10 .001"/>
          <body>
            <joint name="joint" type="slide"/>
            <geom type="sphere" size=".1"/>
            <site name="site"/>
          </body>
          <composite type="cable" curve="s" count="41 1 1" size="1" offset="-.3 0 .6" initial="none">
            <plugin plugin="mujoco.elasticity.cable">
              <config key="twist" value="1e7"/>
              <config key="bend" value="4e6"/>
              <config key="vmax" value="0.05"/>
            </plugin>
            <joint kind="main" damping=".015"/>
            <geom type="capsule" size=".005" rgba=".8 .2 .1 .1" condim="1"/>
          </composite>
        </worldbody>
        <actuator>
          <plugin plugin="mujoco.pid" joint="joint"/>
        </actuator>
        <sensor>
          <plugin plugin="mujoco.sensor.touch_grid" objtype="site" objname="site">
            <config key="size" value="7 7"/>
            <config key="fov" value="45 45"/>
            <config key="gamma" value="0"/>
            <config key="nchannel" value="3"/>
          </plugin>
        </sensor>
      </mujoco>
      """
      )

  def test_ls_parallel(self):
    _, _, m, _ = test_data.fixture(
      xml="""
    <mujoco>
    </mujoco>
    """
    )

    self.assertEqual(m.opt.ls_parallel, False)

    _, _, m, _ = test_data.fixture(
      xml="""
    <mujoco>
      <custom>
        <numeric data="1" name="ls_parallel"/>
      </custom>
    </mujoco>
    """
    )

    self.assertEqual(m.opt.ls_parallel, True)

  def test_contact_sensor_maxmatch(self):
    _, _, m, _ = test_data.fixture(
      xml="""
    <mujoco>
    </mujoco>
    """
    )

    self.assertEqual(m.opt.contact_sensor_maxmatch, 64)

    _, _, m, _ = test_data.fixture(
      xml="""
    <mujoco>
      <custom>
        <numeric data="5" name="contact_sensor_maxmatch"/>
      </custom>
    </mujoco>
    """
    )

    self.assertEqual(m.opt.contact_sensor_maxmatch, 5)

  def test_set_const_qpos0_modification(self):
    """Test set_const recomputes fields after qpos0 modification."""
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="link1" pos="0 0 0">
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
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mjm.qpos0[:] = [0.3, 0.5]
    m.qpos0.numpy()[0, :] = [0.3, 0.5]

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    np.testing.assert_allclose(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, rtol=1e-3)
    np.testing.assert_allclose(m.tendon_invweight0.numpy()[0], mjm.tendon_invweight0, rtol=1e-3)
    np.testing.assert_allclose(m.tendon_length0.numpy()[0], mjm.tendon_length0, rtol=1e-4)

  def test_set_const_body_mass_modification(self):
    """Test set_const recomputes fields after body_mass modification."""
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="link1" pos="0 0 0">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <body name="link2" pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom name="g2" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    new_mass = 3.0
    mjm.body_mass[1] = new_mass
    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = new_mass
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")
    np.testing.assert_allclose(m.body_invweight0.numpy()[0, 1, 0], mjm.body_invweight0[1, 0], rtol=1e-4)

  def test_set_const_freejoint(self):
    """Test set_const with freejoint (6 DOFs with special averaging)."""
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="floating" pos="0 0 1">
          <freejoint/>
          <geom name="box" type="box" size="0.1 0.2 0.3" mass="2.0"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    new_mass = 5.0
    mjm.body_mass[1] = new_mass
    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = new_mass
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    np.testing.assert_allclose(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, rtol=1e-3)
    np.testing.assert_allclose(m.body_invweight0.numpy()[0, 1], mjm.body_invweight0[1], rtol=1e-3)

  def test_set_const_balljoint(self):
    """Test set_const with ball joint (3 DOFs with averaging)."""
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="arm" pos="0 0 0">
          <joint name="ball" type="ball"/>
          <geom name="box" type="box" size="0.1 0.2 0.3" mass="2.0"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    new_inertia = np.array([0.1, 0.2, 0.3])
    mjm.body_inertia[1] = new_inertia
    body_inertia_np = m.body_inertia.numpy()
    body_inertia_np[0, 1] = new_inertia
    wp.copy(m.body_inertia, wp.array(body_inertia_np, dtype=m.body_inertia.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    np.testing.assert_allclose(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, rtol=1e-3)

  def test_set_const_static_body(self):
    """Test set_const with static body (welded to world)."""
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="static_body" pos="1 0 0">
          <geom name="static_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
        </body>
        <body name="dynamic_body" pos="0 0 0">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom name="dynamic_geom" type="sphere" size="0.1" mass="2.0"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    np.testing.assert_allclose(m.body_invweight0.numpy()[0, 1], [0.0, 0.0], atol=1e-6)
    self.assertGreater(m.body_invweight0.numpy()[0, 2, 0], 0.0)
    np.testing.assert_allclose(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, rtol=1e-3)

  def test_set_const_preserves_qpos(self):
    """Test that qpos is restored after set_const."""
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="mass" pos="0 0 0">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom name="mass_geom" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mjd.qpos[0] = 0.5
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    qpos_before = d.qpos.numpy().copy()
    mjwarp.set_const(m, d)

    _assert_eq(d.qpos.numpy(), qpos_before, "qpos")


if __name__ == "__main__":
  wp.init()
  absltest.main()
