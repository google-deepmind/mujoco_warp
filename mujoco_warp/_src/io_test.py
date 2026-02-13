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

import dataclasses

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

# TODO: Add more cameras for testing projection and intrinsics
_CAMERA_TEST_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam1" pos="0 -3 2" xyaxes="1 0 0 0 0.6 0.8" resolution="64 64" output="rgb"/>
    <camera name="cam2" pos="0 3 2" xyaxes="-1 0 0 0 0.6 0.8" resolution="32 32" output="depth"/>
    <camera name="cam3" pos="3 0 2" xyaxes="0 1 0 -0.6 0 0.8" resolution="16 16" output="rgb depth"/>
    <geom type="plane" size="5 5 0.1"/>
    <geom type="sphere" size="0.5" pos="0 0 1"/>
  </worldbody>
</mujoco>
"""


class IOTest(parameterized.TestCase):
  def test_make_put_data(self):
    """Tests that make_data and put_data are producing the same shapes for all arrays."""
    mjm, _, _, d = test_data.fixture("pendula.xml")
    md = mjwarp.make_data(mjm)

    # same number of fields
    self.assertEqual(len(d.__dict__), len(md.__dict__))

    # test shapes for all arrays
    for attr, val in md.__dict__.items():
      if isinstance(val, wp.array):
        self.assertEqual(val.shape, getattr(d, attr).shape, f"{attr} shape mismatch")

  @parameterized.parameters(*_IO_TEST_MODELS)
  def test_put_data_sizes(self, xml):
    EXPECTED_SIZES = {
      "pendula.xml": (48, 64),
      "collision_sdf/tactile.xml": (64, 256),
      "flex/floppy.xml": (256, 512),
      "actuation/tendon_force_limit.xml": (48, 64),
      "actuation/tendon_force_limit.xml": (48, 64),
      "hfield/hfield.xml": (96, 384),
    }
    _, _, _, d = test_data.fixture(xml)
    nconmax_expected, njmax_expected = EXPECTED_SIZES[xml]
    self.assertEqual(d.naconmax, nconmax_expected)
    self.assertEqual(d.njmax, njmax_expected)

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
    d.time.fill_(0.12345)

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
    _assert_eq(d.time.numpy()[world_id], mjd.time, "time")

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

    # actuator_moment
    actuator_moment_dense = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(actuator_moment_dense, mjd.actuator_moment, mjd.moment_rownnz, mjd.moment_rowadr, mjd.moment_colind)
    _assert_eq(
      d.actuator_moment.numpy()[world_id].reshape(-1),
      actuator_moment_dense.reshape(-1),
      "actuator_moment",
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

  @parameterized.parameters(*_IO_TEST_MODELS)
  def test_get_data_into_io_test_models(self, xml):
    """Tests get_data_into for field coverage across diverse model types."""
    mjm, mjd, _, d = test_data.fixture(xml)

    # Create fresh MjData to verify get_data_into populates it correctly
    mjd_result = mujoco.MjData(mjm)

    mjwarp.get_data_into(mjd_result, mjm, d)

    # Compare key fields, including flex/tendon data not covered by humanoid.xml
    for field in [
      "qpos",
      "qvel",
      "qacc",
      "ctrl",
      "act",
      "flexvert_xpos",
      "flexedge_length",
      "flexedge_velocity",
      "ten_length",
      "ten_velocity",
      "actuator_length",
      "actuator_velocity",
      "actuator_force",
      "xpos",
      "xquat",
      "geom_xpos",
    ]:
      if getattr(mjd, field).size > 0:
        _assert_eq(
          getattr(mjd_result, field).reshape(-1),
          getattr(mjd, field).reshape(-1),
          f"{field} (model: {xml})",
        )

    # flexedge_J
    if xml == "flex/floppy.xml":
      from mujoco_warp._src.io import BLEEDING_EDGE_MUJOCO

      if BLEEDING_EDGE_MUJOCO:
        _assert_eq(
          mjd_result.flexedge_J.reshape(-1),
          d.flexedge_J.numpy()[0].reshape(-1),
          "flexedge_J",
        )
      else:
        m = mjwarp.put_model(mjm)
        _assert_eq(
          mjd_result.flexedge_J_rownnz,
          m.flexedge_J_rownnz.numpy(),
          "flexedge_J_rownnz",
        )
        _assert_eq(
          mjd_result.flexedge_J_rowadr,
          m.flexedge_J_rowadr.numpy(),
          "flexedge_J_rowadr",
        )
        _assert_eq(
          mjd_result.flexedge_J_colind,
          m.flexedge_J_colind.numpy().reshape((mjm.nflexedge, mjm.nv)),
          "flexedge_J_colind",
        )
        flexedge_J = np.zeros((mjm.nflexedge, mjm.nv))
        mujoco.mju_sparse2dense(
          flexedge_J,
          d.flexedge_J.numpy().reshape(-1),
          m.flexedge_J_rownnz.numpy(),
          m.flexedge_J_rowadr.numpy(),
          m.flexedge_J_colind.numpy().reshape(-1),
        )
        _assert_eq(
          mjd_result.flexedge_J,
          flexedge_J,
          "flexedge_J",
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
    self.assertTrue(m.has_fluid)

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

  def test_make_data_nccdmax_exceeds_nconmax(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    with self.assertRaises(ValueError, msg="nccdmax.*nconmax"):
      mjwarp.make_data(mjm, nconmax=16, nccdmax=17)

  def test_make_data_naccdmax_exceeds_naconmax(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    with self.assertRaises(ValueError, msg="naccdmax.*naconmax"):
      mjwarp.make_data(mjm, nconmax=16, naconmax=16, naccdmax=17)

  def test_put_data_nccdmax_exceeds_nconmax(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    mjd = mujoco.MjData(mjm)
    with self.assertRaises(ValueError, msg="nccdmax.*nconmax"):
      mjwarp.put_data(mjm, mjd, nconmax=16, nccdmax=17)

  def test_put_data_naccdmax_exceeds_naconmax(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    mjd = mujoco.MjData(mjm)
    with self.assertRaises(ValueError, msg="naccdmax.*naconmax"):
      mjwarp.put_data(mjm, mjd, nconmax=16, naconmax=16, naccdmax=17)

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
    _assert_eq(m.actuator_acc0.numpy(), mjm.actuator_acc0, "actuator_acc0")
    _assert_eq(m.body_invweight0.numpy()[0, 1, 0], mjm.body_invweight0[1, 0], "body_invweight0")

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

  @parameterized.parameters(
    '<worldbody><geom type="sphere" size=".1" condim="3" friction="0 0.1 0.1"/></worldbody>',
    '<worldbody><geom type="sphere" size=".1" condim="4" friction="1 0 0.1"/></worldbody>',
    '<worldbody><geom type="sphere" size=".1" condim="6" friction="1 1 0"/></worldbody>',
    """
      <worldbody>
        <geom name="g1" type="sphere" size=".1"/>
        <geom name="g2" type="sphere" size=".1" pos="0.5 0 0"/>
      </worldbody>
      <contact>
        <pair geom1="g1" geom2="g2" condim="3" friction="0 1 1 1 1"/>
      </contact>
    """,
    """
      <worldbody>
        <geom name="g1" type="sphere" size=".1"/>
        <geom name="g2" type="sphere" size=".1" pos="0.5 0 0"/>
      </worldbody>
      <contact>
        <pair geom1="g1" geom2="g2" condim="4" friction="1 0 0 1 1"/>
      </contact>
    """,
    """
      <worldbody>
        <geom name="g1" type="sphere" size=".1"/>
        <geom name="g2" type="sphere" size=".1" pos="0.5 0 0"/>
      </worldbody>
      <contact>
        <pair geom1="g1" geom2="g2" condim="6" friction="1 1 1 0 0"/>
      </contact>
    """,
  )
  def test_small_friction_warning(self, xml):
    """Tests that a warning is raised for small friction values."""
    with self.assertWarns(UserWarning):
      mjwarp.put_model(mujoco.MjModel.from_xml_string(f"<mujoco>{xml}</mujoco>"))

  @parameterized.product(active=["true", "false"], make_data=[True, False])
  def test_eq_active(self, active, make_data):
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
    <mujoco>
      <worldbody>
        <body name="body1">
          <joint/>
          <geom size=".1"/>
        </body>
        <body name="body2">
          <joint/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="body1" body2="body2" active="{active}"/>
      </equality>
    </mujoco>
    """
    )
    if make_data:
      d = mjwarp.make_data(mjm)

    _assert_eq(d.eq_active.numpy()[0], mjd.eq_active, "eq_active")

  @parameterized.parameters(True, False)
  def test_sleep_state_initial(self, use_make_data):
    """Tests that make_data and put_data initialize all trees awake."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <joint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)

    if use_make_data:
      d = mjwarp.make_data(mjm)
    else:
      d = mjwarp.put_data(mjm, mjd)

    # All trees should be awake (tree_asleep < 0)
    tree_asleep = d.tree_asleep.numpy()
    self.assertTrue((tree_asleep < 0).all(), "tree_asleep should be < 0 (awake)")
    # tree_awake should all be 1
    tree_awake = d.tree_awake.numpy()
    np.testing.assert_array_equal(tree_awake, 1, "tree_awake should be 1")
    # body_awake should all be 1
    body_awake = d.body_awake.numpy()
    np.testing.assert_array_equal(body_awake, 1, "body_awake should be 1")

  def test_sleep_policy_import(self):
    """Tests that tree_sleep_policy matches MuJoCo."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <joint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    m = mjwarp.put_model(mjm)
    np.testing.assert_array_equal(m.tree_sleep_policy.numpy(), mjm.tree_sleep_policy)

  def test_dof_length_import(self):
    """Tests that dof_length matches MuJoCo."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <joint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    m = mjwarp.put_model(mjm)
    np.testing.assert_allclose(m.dof_length.numpy(), mjm.dof_length)

  def test_tree_structure_fields(self):
    """Tests that tree structure fields match between types.Model and mjModel."""
    mjm, _, m, _ = test_data.fixture("pendula.xml")

    # verify fields match MuJoCo
    for field in ["ntree", "tree_dofadr", "tree_dofnum", "tree_bodynum", "body_treeid", "dof_treeid"]:
      m_val = getattr(m, field)
      mjm_val = getattr(mjm, field)
      if isinstance(m_val, wp.array):
        m_val = m_val.numpy()
      np.testing.assert_array_equal(m_val, mjm_val, err_msg=f"mismatch: {field}")

  def test_model_batched_fields(self):
    """Test Model batched fields."""
    nworld = 2
    mjm, _, m, d = test_data.fixture("humanoid/humanoid.xml", keyframe=0, nworld=nworld)

    for f in dataclasses.fields(m):
      # TODO(team): test arrays that are warp only
      if not hasattr(mjm, f.name):
        continue
      if isinstance(f.type, wp.array):
        # get fields
        arr = getattr(m, f.name)
        mj_arr = getattr(mjm, f.name)

        # check that field is not empty
        if 0 in mj_arr.shape + arr.shape:
          continue

        # check for batched field
        if hasattr(arr, "_is_batched") and arr._is_batched:
          assert arr.shape[0] == 1

          # reshape if necessary
          if f.name in ("cam_mat0"):
            mj_arr = mj_arr.reshape((-1, 3, 3))

          # set batched field
          setattr(m, f.name, wp.array(np.tile(mj_arr, (nworld,) + arr.shape[1:]), dtype=f.type.dtype))

    mjwarp.forward(m, d)
    mjwarp.reset_data(m, d)
    mjwarp.forward(m, d)

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

  def test_set_fixed_ngravcomp(self):
    """Test ngravcomp counting with gravcomp bodies."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="body1" gravcomp="1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="sphere" size="0.1" mass="1.0"/>
        </body>
        <body name="body2" pos="1 0 0" gravcomp="0">
          <joint name="j2" type="hinge" axis="0 0 1"/>
          <geom name="g2" type="sphere" size="0.1" mass="1.0"/>
        </body>
        <body name="body3" pos="2 0 0" gravcomp="1">
          <joint name="j3" type="hinge" axis="0 0 1"/>
          <geom name="g3" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    self.assertEqual(m.ngravcomp, mjm.ngravcomp)
    self.assertEqual(m.ngravcomp, 2)  # body1 and body3

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
    _assert_eq(m.actuator_acc0.numpy(), mjm.actuator_acc0, "actuator_acc0")

    for i in range(mjm.nbody):
      _assert_eq(m.body_invweight0.numpy()[0, i], mjm.body_invweight0[i], f"body_invweight0[{i}]")

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  def test_set_const_graph_capture(self):
    """Test that set_const_0 is compatible with CUDA graph capture."""
    _, _, m, d = test_data.fixture("humanoid/humanoid.xml", keyframe=0)

    with wp.ScopedCapture() as capture:
      mjwarp.set_const_0(m, d)
      # TODO(team): set_const_fixed

    wp.capture_launch(capture.graph)

  @parameterized.parameters(1, 4)
  def test_bvh_creation(self, nworld):
    """Test that the BVH is created correctly for single world and multiple worlds."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)
    rc = mjwarp.create_render_context(mjm, nworld=nworld, cam_res=(64, 64), use_textures=False)

    self.assertIsNotNone(rc)
    self.assertEqual(rc.nrender, mjm.ncam)

    self.assertEqual(rc.lower.shape, (nworld * rc.bvh_ngeom,), "lower")
    self.assertEqual(rc.upper.shape, (nworld * rc.bvh_ngeom,), "upper")
    self.assertEqual(rc.group.shape, (nworld * rc.bvh_ngeom,), "group")
    self.assertEqual(rc.group_root.shape, (nworld,), "group_root")

    self.assertIsNotNone(rc.bvh_id)
    self.assertNotEqual(rc.bvh_id, 0, "bvh_id")

    group_np = rc.group.numpy()
    _assert_eq(group_np, np.repeat(np.arange(nworld), rc.bvh_ngeom), "render context group values")

  def test_output_buffers(self):
    """Test that the output rgb and depth buffers have correct shapes and addresses."""
    # TODO: remove after mjwarp depends on mujoco >= 3.4.1 in pyproject.toml
    from mujoco_warp._src.io import BLEEDING_EDGE_MUJOCO

    if not BLEEDING_EDGE_MUJOCO:
      self.skipTest("Skipping test that requires mujoco >= 3.4.1")
      return

    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 24
    rc = mjwarp.create_render_context(mjm, cam_res=(width, height), render_rgb=True, render_depth=True)

    expected_total = 3 * width * height

    self.assertEqual(rc.nrender, 3, "nrender")
    self.assertEqual(rc.rgb_data.shape, (1, expected_total), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (1, expected_total), "depth_data")

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()
    _assert_eq(rgb_adr, [0, width * height, 2 * width * height], "rgb_adr")
    _assert_eq(depth_adr, [0, width * height, 2 * width * height], "depth_adr")

  def test_heterogeneous_camera(self):
    # TODO: remove after mjwarp depends on mujoco >= 3.4.1 in pyproject.toml
    from mujoco_warp._src.io import BLEEDING_EDGE_MUJOCO

    if not BLEEDING_EDGE_MUJOCO:
      self.skipTest("Skipping test that requires mujoco >= 3.4.1")
      return

    """Tests render context with different resolutions and output."""
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    cam_res = [(64, 64), (32, 32), (16, 16)]
    rc = mjwarp.create_render_context(mjm, cam_res=cam_res, render_rgb=True, render_depth=True)

    self.assertEqual(rc.nrender, 3, "nrender")
    _assert_eq(rc.cam_res.numpy(), cam_res, "cam_res")

    expected_total = 64 * 64 + 32 * 32 + 16 * 16
    self.assertEqual(rc.rgb_data.shape, (1, expected_total), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (1, expected_total), "depth_data")

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()
    _assert_eq(rgb_adr, [0, 64 * 64, 64 * 64 + 32 * 32], "rgb_adr")
    _assert_eq(depth_adr, [0, 64 * 64, 64 * 64 + 32 * 32], "depth_adr")

    # Test that results are same when reading from mjmodel fields loaded through xml
    rc_xml = mjwarp.create_render_context(mjm, render_rgb=True, render_depth=True)
    self.assertEqual(rc.rgb_data.shape, rc_xml.rgb_data.shape, "rgb_data")
    self.assertEqual(rc.depth_data.shape, rc_xml.depth_data.shape, "depth_data")
    _assert_eq(rc.rgb_adr.numpy(), rc_xml.rgb_adr.numpy(), "rgb_adr")
    _assert_eq(rc.depth_adr.numpy(), rc_xml.depth_adr.numpy(), "depth_adr")

  def test_cam_active_filtering(self):
    # TODO: remove after mjwarp depends on mujoco >= 3.4.1 in pyproject.toml
    from mujoco_warp._src.io import BLEEDING_EDGE_MUJOCO

    if not BLEEDING_EDGE_MUJOCO:
      self.skipTest("Skipping test that requires mujoco >= 3.4.1")
      return

    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 32

    rc = mjwarp.create_render_context(mjm, cam_res=(width, height), cam_active=[True, False, True])

    self.assertEqual(rc.nrender, 2, "nrender")

    expected_total = 2 * width * height
    self.assertEqual(rc.rgb_data.shape, (1, expected_total), "rgb_data")

  def test_rgb_only_and_depth_only(self):
    """Test that disabling rgb or depth correctly reduces the shape and invalidates the address."""
    # TODO: remove after mjwarp depends on mujoco >= 3.4.1 in pyproject.toml
    from mujoco_warp._src.io import BLEEDING_EDGE_MUJOCO

    if not BLEEDING_EDGE_MUJOCO:
      self.skipTest("Skipping test that requires mujoco >= 3.4.1")
      return

    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 32
    pixels = width * height

    rc = mjwarp.create_render_context(
      mjm,
      cam_res=(width, height),
      render_rgb=[True, False, True],
      render_depth=[False, True, True],
    )

    self.assertEqual(rc.rgb_data.shape, (1, 2 * pixels), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (1, 2 * pixels), "depth_data")
    _assert_eq(rc.rgb_adr.numpy(), [0, -1, pixels], "rgb_adr")
    _assert_eq(rc.depth_adr.numpy(), [-1, 0, pixels], "depth_adr")
    _assert_eq(rc.render_rgb.numpy(), [True, False, True], "render_rgb")
    _assert_eq(rc.render_depth.numpy(), [False, True, True], "render_depth")

    # Test that results are same when reading from mjmodel fields loaded through xml
    rc_xml = mjwarp.create_render_context(mjm, cam_res=(width, height))
    self.assertEqual(rc.rgb_data.shape, rc_xml.rgb_data.shape, "rgb_data")
    self.assertEqual(rc.depth_data.shape, rc_xml.depth_data.shape, "depth_data")
    _assert_eq(rc.rgb_adr.numpy(), rc_xml.rgb_adr.numpy(), "rgb_adr")
    _assert_eq(rc.depth_adr.numpy(), rc_xml.depth_adr.numpy(), "depth_adr")
    _assert_eq(rc.render_rgb.numpy(), rc_xml.render_rgb.numpy(), "render_rgb")
    _assert_eq(rc.render_depth.numpy(), rc_xml.render_depth.numpy(), "render_depth")

  def test_render_context_with_textures(self):
    # TODO: remove after mjwarp depends on warp >= 1.12 in pyproject.toml
    if not hasattr(wp, "Texture2D"):
      self.skipTest("Skipping test that requires warp >= 1.12")
      return

    mjm, mjd, m, d = test_data.fixture("mug/mug.xml")
    rc = mjwarp.create_render_context(mjm, render_rgb=True, render_depth=True, use_textures=True)
    self.assertTrue(rc.use_textures, "use_textures")
    self.assertEqual(rc.textures.shape, (mjm.ntex,), "textures")


if __name__ == "__main__":
  wp.init()
  absltest.main()
