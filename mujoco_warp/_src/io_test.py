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

  def test_get_data_into(self):
    # keyframe=0: ncon=8, nefc=32
    mjm, mjd, _, d = test_data.fixture("humanoid/humanoid.xml", keyframe=0)

    # keyframe=2: ncon=0, nefc=0
    mujoco.mj_resetDataKeyframe(mjm, mjd, 2)

    # check that mujoco._functions._realloc_con_efc allocates for contact and efc
    mjwarp.get_data_into(mjd, mjm, d)
    self.assertEqual(mjd.ncon, 8)
    self.assertEqual(mjd.nefc, 32)

    # compare fields
    self.assertEqual(d.solver_niter.numpy()[0], mjd.solver_niter[0])
    self.assertEqual(d.nacon.numpy()[0], mjd.ncon)
    self.assertEqual(d.ne.numpy()[0], mjd.ne)
    self.assertEqual(d.nf.numpy()[0], mjd.nf)
    self.assertEqual(d.nl.numpy()[0], mjd.nl)

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
      _assert_eq(getattr(d, field).numpy()[0].reshape(-1), getattr(mjd, field).reshape(-1), field)

    # contact
    ncon = d.nacon.numpy()[0]
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
      _assert_eq(getattr(d.contact, field).numpy()[:ncon].reshape(-1), getattr(mjd.contact, field).reshape(-1), field)

    # efc
    nefc = d.nefc.numpy()[0]
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
      _assert_eq(getattr(d.efc, field).numpy()[0, :nefc].reshape(-1), getattr(mjd, "efc_" + field).reshape(-1), field)

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

  @parameterized.parameters(
    ('<distance geom1="plane" geom2="sphere"/>', NotImplementedError),
    ('<distance geom1="sphere" geom2="plane"/>', NotImplementedError),
    ('<distance geom1="hfield" geom2="sphere"/>', ValueError),
    ('<distance geom1="sphere" geom2="hfield"/>', ValueError),
  )
  def test_collision_sensors(self, sensor, err):
    """Tests for collision sensors that are not implemented."""
    with self.assertRaises(err):
      test_data.fixture(
        xml=f"""
      <mujoco>
        <asset>
          <hfield name="hfield" nrow="2" ncol="2" size="1 1 1 1"/>
        </asset>
        <worldbody>
          <geom name="plane" type="plane" size="10 10 .01"/>
          <geom name="hfield" type="hfield" hfield="hfield"/>
          <body>
            <geom name="sphere" type="sphere" size=".1"/>
            <joint type="slide" axis="0 0 1"/>
          </body>
        </worldbody>
        <sensor>
          {sensor}
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


if __name__ == "__main__":
  wp.init()
  absltest.main()
