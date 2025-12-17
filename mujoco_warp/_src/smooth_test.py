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

"""Tests for smooth dynamics functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import ConeType
from mujoco_warp import DisableBit
from mujoco_warp import test_data

# tolerance for difference between MuJoCo and MJWarp smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SmoothTest(parameterized.TestCase):
  def test_mocap_kinematics(self):
    """Tests that mocap bodies and child bodies are correctly updated after mocap_pos changes.

    This is a regression test for a bug where mocap positions were updated after kinematics,
    causing child bodies of mocap bodies to have incorrect positions based on stale mocap data.
    """
    # Create a simple scene with a mocap body that has a child body attached
    xml = """
    <mujoco>
      <worldbody>
        <body name="mocap_parent" mocap="true">
          <geom type="sphere" size="0.1"/>
          <body name="child" pos="1 0 0">
            <geom type="box" size="0.1 0.1 0.1"/>
            <site name="child_site" pos="0.5 0 0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """

    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertEqual(m.nmocap, 1)

    # Find mocap body and child body IDs
    mocap_body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "mocap_parent")
    child_body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "child")
    child_site_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "child_site")

    # Initial kinematics
    mjw.kinematics(m, d)
    initial_mocap_xpos = d.xpos.numpy()[0, mocap_body_id]
    initial_child_xpos = d.xpos.numpy()[0, child_body_id]
    initial_site_xpos = d.site_xpos.numpy()[0, child_site_id].copy()

    # Verify initial positions match MuJoCo
    _assert_eq(d.xpos.numpy()[0], mjd.xpos, "initial xpos")

    # Expected: child should be at mocap_pos + (1, 0, 0) relative offset
    _assert_eq(initial_child_xpos, [1.0, 0.0, 0.0], "initial child xpos")
    _assert_eq(initial_site_xpos, [1.5, 0.0, 0.0], "initial site xpos")

    # Update mocap_pos to a new position
    new_mocap_pos = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    d.mocap_pos.assign(new_mocap_pos)
    mjd.mocap_pos[:] = new_mocap_pos

    # Run kinematics again - this is the critical test
    # Before the fix, this would compute child positions based on the OLD mocap position
    mjw.kinematics(m, d)
    mujoco.mj_kinematics(mjm, mjd)

    # Check positions after update
    updated_mocap_xpos = d.xpos.numpy()[0, mocap_body_id]
    updated_child_xpos = d.xpos.numpy()[0, child_body_id]
    updated_site_xpos = d.site_xpos.numpy()[0, child_site_id]

    # Verify mocap body moved to new position
    _assert_eq(updated_mocap_xpos, new_mocap_pos[0], "Mocap body should be at new position")

    # KEY TEST: Child body should be at new_mocap_pos + (1, 0, 0)
    expected_child_pos = new_mocap_pos[0] + np.array([1.0, 0.0, 0.0])
    _assert_eq(updated_child_xpos, expected_child_pos, "Child body should be offset from NEW mocap position")

    # Site should also be correctly positioned relative to new mocap position
    expected_site_pos = new_mocap_pos[0] + np.array([1.5, 0.0, 0.0])
    _assert_eq(updated_site_xpos, expected_site_pos, "Site should be offset from NEW mocap position")

    # Verify matches MuJoCo reference
    _assert_eq(d.xpos.numpy()[0], mjd.xpos, "updated xpos")
    _assert_eq(d.site_xpos.numpy()[0], mjd.site_xpos, "updated site_xpos")

  @parameterized.parameters(True, False)
  def test_kinematics(self, make_data):
    """Tests kinematics."""
    # TODO(team): improve batched Model field testing (eg, body_pos, body_quat, jnt_axis)
    nworld = 2
    mjm, mjd, m, d = test_data.fixture("pendula.xml", nworld=nworld, keyframe=0)
    if make_data:
      mjd = mujoco.MjData(mjm)
      d = mjw.make_data(mjm, nworld=nworld)

    for arr in (d.xpos, d.xipos, d.xquat, d.xmat, d.ximat, d.xanchor, d.xaxis, d.site_xpos, d.site_xmat):
      arr_view = arr[:, 1:]  # skip world body
      arr_view.fill_(wp.inf)

    qpos = mjm.key_qpos[0]
    mocap_pos = mjm.key_mpos[0]
    mocap_quat = mjm.key_mquat[0]

    mjd.qpos[:] = qpos
    mjd.mocap_pos[:] = mocap_pos.reshape((mjm.nmocap, 3))
    mjd.mocap_quat[:] = mocap_quat.reshape((mjm.nmocap, 4))

    wp.copy(d.qpos, wp.array(np.tile(qpos, (nworld, 1)), shape=(nworld, mjm.nq), dtype=float))
    wp.copy(d.mocap_pos, wp.array(np.tile(mocap_pos, (nworld, 1)), shape=(nworld, mjm.nmocap), dtype=wp.vec3))
    wp.copy(d.mocap_quat, wp.array(np.tile(mocap_quat, (nworld, 1)), shape=(nworld, mjm.nmocap), dtype=wp.quat))

    mujoco.mj_kinematics(mjm, mjd)
    mjw.kinematics(m, d)

    ximat = np.zeros((m.nbody, 9), dtype=np.float64)
    for i in range(m.nbody):
      mujoco.mju_quat2Mat(ximat[i], d.xiquat.numpy()[0, i])

    geom_xmat = np.zeros((m.ngeom, 9), dtype=np.float64)
    for i in range(m.ngeom):
      mujoco.mju_quat2Mat(geom_xmat[i], d.geom_xquat.numpy()[0, i])

    site_xmat = np.zeros((m.nsite, 9), dtype=np.float64)
    for i in range(m.nsite):
      mujoco.mju_quat2Mat(site_xmat[i], d.site_xquat.numpy()[0, i])

    for i in range(nworld):
      _assert_eq(d.xanchor.numpy()[i], mjd.xanchor, "xanchor")
      _assert_eq(d.xaxis.numpy()[i], mjd.xaxis, "xaxis")
      _assert_eq(d.xpos.numpy()[i], mjd.xpos, "xpos")
      _assert_eq(d.xquat.numpy()[i], mjd.xquat, "xquat")
      _assert_eq(d.xipos.numpy()[i], mjd.xipos, "xipos")
      _assert_eq(ximat.numpy()[i], mjd.ximat.reshape((-1, 3, 3)), "ximat")
      _assert_eq(d.geom_xpos.numpy()[i], mjd.geom_xpos, "geom_xpos")
      _assert_eq(geom_xmat.numpy()[i], mjd.geom_xmat.reshape((-1, 3, 3)), "geom_xmat")
      _assert_eq(d.site_xpos.numpy()[i], mjd.site_xpos, "site_xpos")
      _assert_eq(site_xmat.numpy()[i], mjd.site_xmat.reshape((-1, 3, 3)), "site_xmat")
      _assert_eq(d.mocap_pos.numpy()[i], mjd.mocap_pos, "mocap_pos")
      _assert_eq(d.mocap_quat.numpy()[i], mjd.mocap_quat, "mocap_quat")

  def test_com_pos(self):
    """Tests com_pos."""
    _, mjd, m, d = test_data.fixture("pendula.xml")

    for arr in (d.subtree_com, d.cinert, d.cdof):
      arr.zero_()

    mjw.com_pos(m, d)
    _assert_eq(d.subtree_com.numpy()[0], mjd.subtree_com, "subtree_com")
    _assert_eq(d.cinert.numpy()[0], mjd.cinert, "cinert")
    _assert_eq(d.cdof.numpy()[0], mjd.cdof, "cdof")

  def test_camlight(self):
    """Tests camlight."""
    _, mjd, m, d = test_data.fixture("pendula.xml")

    d.cam_xpos.zero_()
    d.cam_xmat.zero_()
    d.light_xpos.zero_()
    d.light_xdir.zero_()

    mjw.camlight(m, d)
    _assert_eq(d.cam_xpos.numpy()[0], mjd.cam_xpos, "cam_xpos")
    _assert_eq(d.cam_xmat.numpy()[0], mjd.cam_xmat.reshape((-1, 3, 3)), "cam_xmat")
    _assert_eq(d.light_xpos.numpy()[0], mjd.light_xpos, "light_xpos")
    _assert_eq(d.light_xdir.numpy()[0], mjd.light_xdir, "light_xdir")

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_crb(self, jacobian):
    """Tests crb."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml", overrides={"opt.jacobian": jacobian})

    d.crb.zero_()

    mjw.crb(m, d)
    _assert_eq(d.crb.numpy()[0], mjd.crb, "crb")

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      _assert_eq(d.qM.numpy()[0, 0], mjd.qM, "qM")
    else:
      qM = np.zeros((mjm.nv, mjm.nv))
      mujoco.mj_fullM(mjm, qM, mjd.qM)
      _assert_eq(d.qM.numpy()[0, : mjm.nv, : mjm.nv], qM, "qM")

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_factor_m(self, jacobian):
    """Tests factor_m."""
    _, mjd, m, d = test_data.fixture("pendula.xml", overrides={"opt.jacobian": jacobian})

    qLD = d.qLD.numpy()[0].copy()
    for arr in (d.qLD, d.qLDiagInv):
      arr.zero_()

    mjw.factor_m(m, d)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      _assert_eq(d.qLD.numpy()[0, 0], mjd.qLD, "qLD (sparse)")
      _assert_eq(d.qLDiagInv.numpy()[0], mjd.qLDiagInv, "qLDiagInv")
    else:
      _assert_eq(d.qLD.numpy()[0], qLD, "qLD (dense)")

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_solve_m(self, jacobian):
    """Tests solve_m."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml", overrides={"opt.jacobian": jacobian})

    qfrc_smooth = np.tile(mjd.qfrc_smooth, (1, 1))
    qacc_smooth = np.zeros(shape=(1, mjm.nv), dtype=float)
    mujoco.mj_solveM(mjm, mjd, qacc_smooth, qfrc_smooth)

    d.qacc_smooth.zero_()

    mjw.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)
    _assert_eq(d.qacc_smooth.numpy()[0], qacc_smooth[0], "qacc_smooth")

  @parameterized.parameters(0, DisableBit.GRAVITY)
  def test_rne(self, gravity):
    """Tests rne."""
    _, mjd, m, d = test_data.fixture("pendula.xml", overrides={"opt.disableflags": DisableBit.CONTACT | gravity})

    d.qfrc_bias.zero_()

    mjw.rne(m, d)
    _assert_eq(d.qfrc_bias.numpy()[0], mjd.qfrc_bias, "qfrc_bias")

  @parameterized.parameters(0, DisableBit.GRAVITY)
  def test_rne_postconstraint(self, gravity):
    """Tests rne_postconstraint."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml", overrides={"opt.disableflags": DisableBit.CONTACT | gravity})

    mjd.xfrc_applied = np.random.uniform(low=-0.01, high=0.01, size=mjd.xfrc_applied.shape)
    d.xfrc_applied = wp.array(np.expand_dims(mjd.xfrc_applied, axis=0), dtype=wp.spatial_vector)

    mujoco.mj_rnePostConstraint(mjm, mjd)

    for arr in (d.cacc, d.cfrc_int, d.cfrc_ext):
      arr.zero_()

    mjw.rne_postconstraint(m, d)

    _assert_eq(d.cacc.numpy()[0], mjd.cacc, "cacc")
    _assert_eq(d.cfrc_int.numpy()[0], mjd.cfrc_int, "cfrc_int")
    _assert_eq(d.cfrc_ext.numpy()[0], mjd.cfrc_ext, "cfrc_ext")

    _EQUALITY = """
      <mujoco>
        <option gravity="1 1 -1">
          <flag contact="disable"/>
        </option>
        <worldbody>
          <site name="siteworld"/>
          <body name="body0">
            <geom type="sphere" size=".1"/>
            <freejoint/>
          </body>
          <body name="body1">
            <geom type="sphere" size=".1"/>
            <site name="site1"/>
            <freejoint/>
          </body>
          <body name="body2">
            <geom type="sphere" size=".1"/>
            <freejoint/>
          </body>
          <body name="body3">
            <geom type="sphere" size=".1"/>
            <site name="site3" quat="0 1 0 0"/>
            <freejoint/>
          </body>
        </worldbody>
        <equality>
          <connect body1="body0" anchor="1 1 1"/>
          <connect site1="siteworld" site2="site1"/>
          <weld body1="body2" relpose="1 1 1 0 1 0 0"/>
          <weld site1="siteworld" site2="site3"/>
        </equality>
        <keyframe>
          <key qpos="0 0 0 1 0 0 0 1 1 1 1 0 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 0"/>
        </keyframe>
      </mujoco>
      """
    mjm, mjd, m, d = test_data.fixture(xml=_EQUALITY, qvel_noise=0.01, ctrl_noise=0.1, keyframe=0)

    mujoco.mj_rnePostConstraint(mjm, mjd)

    d.cfrc_ext.zero_()
    mjw.rne_postconstraint(m, d)

    _assert_eq(d.cfrc_ext.numpy()[0], mjd.cfrc_ext, "cfrc_ext (equality)")

    mjm, mjd, m, d = test_data.fixture("constraints.xml", keyframe=1, overrides={"opt.disableflags": DisableBit.EQUALITY})

    mujoco.mj_rnePostConstraint(mjm, mjd)

    d.cfrc_ext.zero_()

    # clear equality constraint counts
    d.ne_connect.zero_()
    d.ne_weld.zero_()
    d.ne_jnt.zero_()

    mjw.rne_postconstraint(m, d)

    _assert_eq(d.cfrc_ext.numpy()[0], mjd.cfrc_ext, "cfrc_ext (contact)")

  def test_com_vel(self):
    """Tests com_vel."""
    _, mjd, m, d = test_data.fixture("pendula.xml")

    for arr in (d.cvel, d.cdof_dot):
      arr.zero_()

    mjw.com_vel(m, d)
    _assert_eq(d.cvel.numpy()[0], mjd.cvel, "cvel")
    _assert_eq(d.cdof_dot.numpy()[0], mjd.cdof_dot, "cdof_dot")

  @parameterized.parameters("pendula.xml", "actuation/site.xml", "actuation/slidercrank.xml")
  def test_transmission(self, xml):
    """Tests transmission."""
    mjm, mjd, m, d = test_data.fixture(xml)

    for arr in (d.actuator_length, d.actuator_moment):
      arr.zero_()

    actuator_moment = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(
      actuator_moment,
      mjd.actuator_moment,
      mjd.moment_rownnz,
      mjd.moment_rowadr,
      mjd.moment_colind,
    )

    mjw._src.smooth.transmission(m, d)
    _assert_eq(d.actuator_length.numpy()[0], mjd.actuator_length, "actuator_length")
    _assert_eq(d.actuator_moment.numpy()[0], actuator_moment, "actuator_moment")

  @parameterized.product(keyframe=list(range(4)), cone=list(ConeType))
  def test_actuator_adhesion(self, keyframe, cone):
    """Tests adhesion actuator."""
    mjm, mjd, m, d = test_data.fixture("actuation/adhesion.xml", keyframe=keyframe, overrides={"opt.cone": cone})

    d.actuator_length.zero_()
    d.actuator_moment.zero_()
    mjw._src.collision_driver.collision(m, d)  # compute contact.includemargin
    mjw._src.constraint.make_constraint(m, d)  # compute contact.efc_address
    mjw._src.smooth.transmission(m, d)

    actuator_moment = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(actuator_moment, mjd.actuator_moment, mjd.moment_rownnz, mjd.moment_rowadr, mjd.moment_colind)

    _assert_eq(d.actuator_length.numpy()[0], mjd.actuator_length, "actuator_length")
    _assert_eq(d.actuator_moment.numpy()[0], actuator_moment, "acutator_moment")

  def test_subtree_vel(self):
    """Tests subtree_vel."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml")

    for arr in (d.subtree_linvel, d.subtree_angmom):
      arr.zero_()

    mujoco.mj_subtreeVel(mjm, mjd)
    mjw.subtree_vel(m, d)

    _assert_eq(d.subtree_linvel.numpy()[0], mjd.subtree_linvel, "subtree_linvel")
    _assert_eq(d.subtree_angmom.numpy()[0], mjd.subtree_angmom, "subtree_angmom")

  @parameterized.parameters(
    "tendon/fixed.xml",
    "tendon/site.xml",
    "tendon/pulley_site.xml",
    "tendon/fixed_site.xml",
    "tendon/pulley_fixed_site.xml",
    "tendon/site_fixed.xml",
    "tendon/pulley_site_fixed.xml",
    "tendon/wrap.xml",
    "tendon/pulley_wrap.xml",
  )
  def test_tendon(self, xml):
    """Tests tendon."""
    mjm, mjd, m, d = test_data.fixture(xml, keyframe=0)

    for arr in (d.ten_length, d.ten_J, d.actuator_length, d.actuator_moment):
      arr.zero_()

    mjw.tendon(m, d)
    mjw.transmission(m, d)

    _assert_eq(d.ten_length.numpy()[0], mjd.ten_length, "ten_length")
    _assert_eq(d.ten_J.numpy()[0], mjd.ten_J.reshape((mjm.ntendon, mjm.nv)), "ten_J")
    _assert_eq(d.wrap_xpos.numpy()[0], mjd.wrap_xpos, "wrap_xpos")
    _assert_eq(d.wrap_obj.numpy()[0], mjd.wrap_obj, "wrap_obj")
    _assert_eq(d.ten_wrapnum.numpy()[0], mjd.ten_wrapnum, "ten_wrapnum")
    _assert_eq(d.ten_wrapadr.numpy()[0], mjd.ten_wrapadr, "ten_wrapadr")
    _assert_eq(d.actuator_length.numpy()[0], mjd.actuator_length, "actuator_length")
    actuator_moment = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(
      actuator_moment,
      mjd.actuator_moment,
      mjd.moment_rownnz,
      mjd.moment_rowadr,
      mjd.moment_colind,
    )
    _assert_eq(d.actuator_moment.numpy()[0], actuator_moment, "actuator_moment")

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_factor_solve_i(self, jacobian):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size=".1"/>
          <freejoint/>
        </body>
      </worldbody>
    </mujoco>
    """,
      overrides={"opt.jacobian": jacobian},
    )

    qM = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, qM, mjd.qM)

    sparse = jacobian == mujoco.mjtJacobian.mjJAC_SPARSE

    d.qLD.fill_(wp.inf)
    if sparse:
      d.qLDiagInv.fill_(wp.inf)

    res = wp.zeros((1, mjm.nv), dtype=float)
    vec = wp.ones((1, mjm.nv), dtype=float)

    mjw._src.smooth.factor_solve_i(m, d, d.qM, d.qLD, d.qLDiagInv, res, vec)

    _assert_eq(res.numpy()[0], np.linalg.solve(qM, vec.numpy()[0]), "qM \\ 1")

    if sparse:
      _assert_eq(d.qLD.numpy()[0].reshape(-1), mjd.qLD, "qLD")
      _assert_eq(d.qLDiagInv.numpy()[0], mjd.qLDiagInv, "qLDiagInv")
    else:
      qLD = np.linalg.cholesky(qM)
      _assert_eq(d.qLD.numpy()[0], qLD, "qLD")

  def test_tendon_armature(self):
    mjm, mjd, m, d = test_data.fixture("tendon/armature.xml", keyframe=0)

    # qM
    d.qM.zero_()

    mjw._src.smooth.crb(m, d)
    mjw._src.smooth.tendon_armature(m, d)

    qM = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, qM, mjd.qM)
    _assert_eq(d.qM.numpy()[0, : mjm.nv, : mjm.nv], qM, "qM")

    # qfrc_bias
    d.qfrc_bias.zero_()

    mjw._src.smooth.rne(m, d)
    mjw._src.smooth.tendon_bias(m, d, d.qfrc_bias)
    _assert_eq(d.qfrc_bias.numpy()[0], mjd.qfrc_bias, "qfrc_bias")

  def test_flex(self):
    mjm, mjd, m, d = test_data.fixture("flex/floppy.xml")
    d.flexvert_xpos.fill_(wp.inf)
    d.flexedge_length.fill_(wp.inf)
    d.flexedge_velocity.fill_(wp.inf)
    d.flexedge_J.fill_(wp.inf)

    mjw.kinematics(m, d)
    mjw.com_pos(m, d)
    mjw.flex(m, d)
    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_comPos(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)

    flexedge_J = np.zeros((mjm.nflexedge, mjm.nv), dtype=float)
    mujoco.mju_sparse2dense(
      flexedge_J, mjd.flexedge_J.ravel(), mjd.flexedge_J_rownnz, mjd.flexedge_J_rowadr, mjd.flexedge_J_colind.ravel()
    )

    _assert_eq(d.flexvert_xpos.numpy()[0], mjd.flexvert_xpos, "flexvert_xpos")
    _assert_eq(d.flexedge_length.numpy()[0], mjd.flexedge_length, "flexedge_length")
    _assert_eq(d.flexedge_velocity.numpy()[0], mjd.flexedge_velocity, "flexedge_velocity")
    _assert_eq(d.flexedge_J.numpy()[0], flexedge_J, "flexedge_J")


if __name__ == "__main__":
  wp.init()
  absltest.main()
