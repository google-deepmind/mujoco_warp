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

    for i in range(nworld):
      _assert_eq(d.xanchor.numpy()[i], mjd.xanchor, "xanchor")
      _assert_eq(d.xaxis.numpy()[i], mjd.xaxis, "xaxis")
      _assert_eq(d.xpos.numpy()[i], mjd.xpos, "xpos")
      _assert_eq(d.xquat.numpy()[i], mjd.xquat, "xquat")
      _assert_eq(d.xmat.numpy()[i], mjd.xmat.reshape((-1, 3, 3)), "xmat")
      _assert_eq(d.xipos.numpy()[i], mjd.xipos, "xipos")
      _assert_eq(d.ximat.numpy()[i], mjd.ximat.reshape((-1, 3, 3)), "ximat")
      _assert_eq(d.geom_xpos.numpy()[i], mjd.geom_xpos, "geom_xpos")
      _assert_eq(d.geom_xmat.numpy()[i], mjd.geom_xmat.reshape((-1, 3, 3)), "geom_xmat")
      _assert_eq(d.site_xpos.numpy()[i], mjd.site_xpos, "site_xpos")
      _assert_eq(d.site_xmat.numpy()[i], mjd.site_xmat.reshape((-1, 3, 3)), "site_xmat")
      _assert_eq(d.mocap_pos.numpy()[i], mjd.mocap_pos, "mocap_pos")
      _assert_eq(d.mocap_quat.numpy()[i], mjd.mocap_quat, "mocap_quat")

  @parameterized.parameters(5, 6)
  def test_kinematics_many_joints_per_body(self, njoints):
    """A single body can carry up to six non-free joints (3 slide + 3 hinge).

    The unrolled joint loop in _kinematics_branch must process all of them, otherwise
    the trailing joints keep their zero-initialized xaxis and corrupt downstream dynamics.
    This is a regression test for bodies with five or six joints, where the unroll
    previously stopped at four and left xaxis at zero for the fifth and sixth joints.
    """
    slides = """
        <joint name="x" type="slide" axis="1 0 0"/>
        <joint name="y" type="slide" axis="0 1 0"/>
        <joint name="z" type="slide" axis="0 0 1"/>"""
    hinges = """
        <joint name="rx" type="hinge" axis="1 0 0"/>
        <joint name="ry" type="hinge" axis="0 1 0"/>"""
    if njoints == 6:
      hinges += """
        <joint name="rz" type="hinge" axis="0 0 1"/>"""
    xml = f"""
    <mujoco>
      <worldbody>
        <body name="multi" pos="0.1 0.2 0.3">{slides}{hinges}
          <geom type="sphere" size="0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    self.assertEqual(int((mjm.jnt_bodyid == 1).sum()), njoints)

    mjd = mujoco.MjData(mjm)
    # Non-trivial configuration so every joint contributes a distinct transform.
    mjd.qpos[:] = np.array([0.05, -0.1, 0.15, 0.3, -0.2, 0.25][:njoints])
    mujoco.mj_kinematics(mjm, mjd)

    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm)
    wp.copy(d.qpos, wp.array(mjd.qpos.reshape(1, -1), dtype=float))
    mjw.kinematics(m, d)

    _assert_eq(d.xanchor.numpy()[0], mjd.xanchor, "xanchor")
    _assert_eq(d.xaxis.numpy()[0], mjd.xaxis, "xaxis")
    _assert_eq(d.xpos.numpy()[0], mjd.xpos, "xpos")
    _assert_eq(d.xquat.numpy()[0], mjd.xquat, "xquat")

  def test_com_pos(self):
    """Tests com_pos."""
    _, mjd, m, d = test_data.fixture("pendula.xml")

    for arr in (d.subtree_com, d.cinert, d.cdof):
      arr.fill_(wp.inf)

    mjw.com_pos(m, d)
    _assert_eq(d.subtree_com.numpy()[0], mjd.subtree_com, "subtree_com")
    _assert_eq(d.cinert.numpy()[0], mjd.cinert, "cinert")
    _assert_eq(d.cdof.numpy()[0], mjd.cdof, "cdof")

  def test_camlight(self):
    """Tests camlight."""
    _, mjd, m, d = test_data.fixture("pendula.xml")

    d.cam_xpos.fill_(wp.inf)
    d.cam_xmat.fill_(wp.inf)
    d.light_xpos.fill_(wp.inf)
    d.light_xdir.fill_(wp.inf)

    mjw.camlight(m, d)
    _assert_eq(d.cam_xpos.numpy()[0], mjd.cam_xpos, "cam_xpos")
    _assert_eq(d.cam_xmat.numpy()[0], mjd.cam_xmat.reshape((-1, 3, 3)), "cam_xmat")
    _assert_eq(d.light_xpos.numpy()[0], mjd.light_xpos, "light_xpos")
    _assert_eq(d.light_xdir.numpy()[0], mjd.light_xdir, "light_xdir")

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_crb(self, jacobian):
    """Tests crb."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml", overrides={"opt.jacobian": jacobian})

    d.crb.fill_(wp.inf)

    mjw.crb(m, d)
    _assert_eq(d.crb.numpy()[0], mjd.crb, "crb")

    _assert_eq(d.M.numpy()[0], mjd.M, "M")

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_factor_m(self, jacobian):
    """Tests factor_m."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml", overrides={"opt.jacobian": jacobian})

    d.qLDiagInv.fill_(wp.inf)
    d.qLD.fill_(wp.inf)

    mjw.factor_m(m, d)

    # qLD layout is per-block, independent of is_sparse (which selects only the constraint
    # Jacobian/Hessian layout). A packed dense region is not MuJoCo's LDL, so verify via solve;
    # a pure sparse model can compare the LDL factor directly.
    if m.qLD_has_dense:
      # packed dense-block Cholesky: factor layout differs from LDL, so verify it solves correctly
      ref = np.zeros((1, mjm.nv))
      mujoco.mj_solveM(mjm, mjd, ref, np.tile(mjd.qfrc_smooth, (1, 1)))
      res = wp.zeros((1, mjm.nv), dtype=float)
      mjw.solve_m(m, d, res, d.qfrc_smooth)
      _assert_eq(res.numpy()[0], ref[0], "M \\ qfrc_smooth (block-dense)")
    else:
      _assert_eq(d.qLD.numpy()[0], mjd.qLD, "qLD (sparse)")
      _assert_eq(d.qLDiagInv.numpy()[0], mjd.qLDiagInv, "qLDiagInv")

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_solve_m(self, jacobian):
    """Tests solve_m."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml", overrides={"opt.jacobian": jacobian})

    qfrc_smooth = np.tile(mjd.qfrc_smooth, (1, 1))
    qacc_smooth = np.zeros(shape=(1, mjm.nv), dtype=float)
    mujoco.mj_solveM(mjm, mjd, qacc_smooth, qfrc_smooth)

    d.qacc_smooth.fill_(wp.inf)

    mjw.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)
    _assert_eq(d.qacc_smooth.numpy()[0], qacc_smooth[0], "qacc_smooth")

  @parameterized.parameters(0, DisableBit.GRAVITY)
  def test_rne(self, gravity):
    """Tests rne."""
    _, mjd, m, d = test_data.fixture("pendula.xml", overrides={"opt.disableflags": DisableBit.CONTACT | gravity})

    d.qfrc_bias.fill_(wp.inf)

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
      arr.fill_(wp.inf)

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

    mjw.rne_postconstraint(m, d)

    _assert_eq(d.cfrc_ext.numpy()[0], mjd.cfrc_ext, "cfrc_ext (contact)")

  def test_com_vel(self):
    """Tests com_vel."""
    _, mjd, m, d = test_data.fixture("pendula.xml")

    for arr in (d.cvel, d.cdof_dot):
      arr.fill_(wp.inf)

    mjw.com_vel(m, d)
    _assert_eq(d.cvel.numpy()[0], mjd.cvel, "cvel")
    _assert_eq(d.cdof_dot.numpy()[0], mjd.cdof_dot, "cdof_dot")

  @parameterized.parameters("pendula.xml", "actuation/site.xml", "actuation/slidercrank.xml")
  def test_transmission(self, xml):
    """Tests transmission."""
    mjm, mjd, m, d = test_data.fixture(xml)

    for arr in (d.actuator_length, d.actuator_moment):
      arr.fill_(wp.inf)

    mj_actuator_moment = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(
      mj_actuator_moment,
      mjd.actuator_moment,
      mjd.moment_rownnz,
      mjd.moment_rowadr,
      mjd.moment_colind,
    )

    mjw.transmission(m, d)

    actuator_moment = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(
      actuator_moment,
      d.actuator_moment.numpy()[0],
      d.moment_rownnz.numpy()[0],
      d.moment_rowadr.numpy()[0],
      d.moment_colind.numpy()[0],
    )

    _assert_eq(d.actuator_length.numpy()[0], mjd.actuator_length, "actuator_length")
    _assert_eq(actuator_moment, mj_actuator_moment, "actuator_moment")

  @parameterized.product(
    keyframe=list(range(4)), cone=list(ConeType), jacobian=[mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE]
  )
  def test_actuator_adhesion(self, keyframe, cone, jacobian):
    """Tests adhesion actuator."""
    mjm, mjd, m, d = test_data.fixture(
      "actuation/adhesion.xml", keyframe=keyframe, overrides={"opt.cone": cone, "opt.jacobian": jacobian}
    )

    for arr in (d.actuator_length, d.actuator_moment):
      arr.fill_(wp.inf)

    mjw._src.collision_driver.collision(m, d)  # compute contact.includemargin
    mjw._src.constraint.make_constraint(m, d)  # compute contact.efc_address
    mjw._src.smooth.transmission(m, d)

    actuator_moment = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(actuator_moment, mjd.actuator_moment, mjd.moment_rownnz, mjd.moment_rowadr, mjd.moment_colind)

    _assert_eq(d.actuator_length.numpy()[0], mjd.actuator_length, "actuator_length")
    wp_actuator_moment = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(
      wp_actuator_moment,
      d.actuator_moment.numpy()[0],
      d.moment_rownnz.numpy()[0],
      d.moment_rowadr.numpy()[0],
      d.moment_colind.numpy()[0],
    )
    _assert_eq(wp_actuator_moment, actuator_moment, "actuator_moment")

  def test_subtree_vel(self):
    """Tests subtree_vel."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml")

    for arr in (d.subtree_linvel, d.subtree_angmom):
      arr.fill_(wp.inf)

    mujoco.mj_subtreeVel(mjm, mjd)
    mjw.subtree_vel(m, d)

    _assert_eq(d.subtree_linvel.numpy()[0], mjd.subtree_linvel, "subtree_linvel")
    _assert_eq(d.subtree_angmom.numpy()[0], mjd.subtree_angmom, "subtree_angmom")

  @parameterized.product(
    xml=[
      "tendon/fixed.xml",
      "tendon/site.xml",
      "tendon/pulley_site.xml",
      "tendon/fixed_site.xml",
      "tendon/pulley_fixed_site.xml",
      "tendon/site_fixed.xml",
      "tendon/pulley_site_fixed.xml",
      "tendon/wrap.xml",
      "tendon/pulley_wrap.xml",
    ],
    jacobian=(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE),
  )
  def test_tendon(self, xml, jacobian):
    """Tests tendon."""
    mjm, mjd, m, d = test_data.fixture(xml, keyframe=0, overrides={"opt.jacobian": jacobian})

    for arr in (d.ten_length, d.ten_J, d.actuator_length, d.actuator_moment):
      arr.fill_(wp.inf)

    mjw.tendon(m, d)
    mjw.transmission(m, d)

    _assert_eq(d.ten_length.numpy()[0], mjd.ten_length, "ten_length")
    ten_J = np.zeros((mjm.ntendon, mjm.nv))
    mujoco.mju_sparse2dense(ten_J, mjd.ten_J.reshape(-1), mjm.ten_J_rownnz, mjm.ten_J_rowadr, mjm.ten_J_colind.reshape(-1))
    _assert_eq(d.wrap_xpos.numpy()[0], mjd.wrap_xpos, "wrap_xpos")
    _assert_eq(d.wrap_obj.numpy()[0], mjd.wrap_obj, "wrap_obj")
    _assert_eq(d.ten_wrapnum.numpy()[0], mjd.ten_wrapnum, "ten_wrapnum")
    _assert_eq(d.ten_wrapadr.numpy()[0], mjd.ten_wrapadr, "ten_wrapadr")
    _assert_eq(d.actuator_length.numpy()[0], mjd.actuator_length, "actuator_length")
    if mjm.nu:
      mj_actuator_moment = np.zeros((mjm.nu, mjm.nv))
      mujoco.mju_sparse2dense(
        mj_actuator_moment,
        mjd.actuator_moment,
        mjd.moment_rownnz,
        mjd.moment_rowadr,
        mjd.moment_colind,
      )
      wp_actuator_moment = np.zeros((mjm.nu, mjm.nv))
      mujoco.mju_sparse2dense(
        wp_actuator_moment,
        d.actuator_moment.numpy()[0],
        d.moment_rownnz.numpy()[0],
        d.moment_rowadr.numpy()[0],
        d.moment_colind.numpy()[0],
      )
      _assert_eq(wp_actuator_moment, mj_actuator_moment, "actuator_moment")

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
    mujoco.mju_sym2dense(qM, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)

    d.qLD.fill_(wp.inf)
    d.qLDiagInv.fill_(wp.inf)

    res = wp.zeros((1, mjm.nv), dtype=float)
    vec = wp.ones((1, mjm.nv), dtype=float)

    mjw._src.smooth.factor_solve_i(m, d, d.M, d.qLD, d.qLDiagInv, res, vec)

    _assert_eq(res.numpy()[0], np.linalg.solve(qM, vec.numpy()[0]), "M \\ 1")

    # The M \ 1 check above verifies the solve regardless of layout. The raw factor matches MuJoCo's
    # only in specific cases: a pure sparse-LDL model stores the L'DL factor in qLD, while a pure
    # simple (diagonal) model stores nothing in qLD and just 1/diag in qLDiagInv.
    if m.qLD_has_sparse and not m.qLD_has_dense and not m.qLD_has_simple:
      _assert_eq(d.qLD.numpy()[0].reshape(-1), mjd.qLD, "qLD")
      _assert_eq(d.qLDiagInv.numpy()[0], mjd.qLDiagInv, "qLDiagInv")
    elif m.qLD_has_simple and not m.qLD_has_dense and not m.qLD_has_sparse:
      _assert_eq(d.qLDiagInv.numpy()[0], mjd.qLDiagInv, "qLDiagInv")

  def test_factor_solve_mixed_blocks(self):
    """Per-block factor/solve: one oversized block (sparse LDL) plus small blocks (packed dense)."""

    # A 70-dof hinge chain (one block > M_BLOCK_DENSE_MAX -> sparse LDL) and three short hinge
    # chains (6-dof coupled blocks -> packed dense): factor_m/solve_m must run both passes into one
    # qLD. (Coupled blocks, not free joints: a free joint on a sphere has diagonal M and would take
    # the trivial sparse path, so it would not exercise the packed-dense factor.)
    def _hinge_chain(k):
      return (
        "".join(
          '<body pos="0 0 .05"><joint type="hinge" axis="0 1 0"/><geom type="capsule" size=".02 .025"/>' for _ in range(k)
        )
        + "</body>" * k
      )

    n = 70
    chain = (
      "".join('<body pos="0 0 .05"><joint type="hinge" axis="1 0 0"/><geom type="capsule" size=".02 .025"/>' for _ in range(n))
      + "</body>" * n
    )
    small = "".join(f'<body pos="{i} 0 1">{_hinge_chain(6)}</body>' for i in range(3))
    xml = f"<mujoco><worldbody><body pos='0 0 2'>{chain}</body>{small}</worldbody></mujoco>"

    mjm, mjd, m, d = test_data.fixture(xml=xml)
    # genuinely mixed: both factor paths active in one model
    self.assertTrue(m.qLD_has_dense)
    self.assertTrue(m.qLD_has_sparse)

    # M^{-1} @ qfrc_smooth parity against MuJoCo exercises both the packed and the LDL solve.
    ref = np.zeros((1, mjm.nv))
    mujoco.mj_solveM(mjm, mjd, ref, np.tile(mjd.qfrc_smooth, (1, 1)))

    mjw.crb(m, d)
    mjw.factor_m(m, d)
    res = wp.zeros((1, mjm.nv), dtype=float)
    mjw.solve_m(m, d, res, d.qfrc_smooth)
    _assert_eq(res.numpy()[0], ref[0], "M \\ qfrc_smooth (mixed blocks)")

  def test_factor_solve_lu(self):
    """Tests factor_solve_lu with a sparse matrix representing a tree."""
    xml = """
    <mujoco>
      <option>
        <flag constraint="disable"/>
      </option>
      <worldbody>
        <body>
          <joint type="slide" axis="0 0 1"/>
          <geom size=".03"/>
          <body>
            <joint axis="0 1 0"/>
            <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
          </body>
        </body>
        <body pos="0 0.1 0">
          <joint name="slide" type="slide" axis="0 0 1"/>
          <geom size=".03"/>
          <body>
            <joint name="hinge" axis="0 1 0"/>
            <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    nv = m.nv
    nD = m.nD

    # Create a random non-symmetric diagonally dominant matrix
    np.random.seed(42)
    A = np.random.rand(nv, nv)
    A += np.diag(np.sum(np.abs(A), axis=1) + 1.0)

    # Zero out elements not in D-structure to enforce sparsity
    D_rowadr = m.D_rowadr.numpy()
    D_rownnz = m.D_rownnz.numpy()
    D_colind = m.D_colind.numpy()

    A_sparse = np.zeros((nv, nv))
    qLU_np = np.zeros((1, 1, nD), dtype=np.float32)

    for i in range(nv):
      row_start = D_rowadr[i]
      row_nnz = D_rownnz[i]
      for k in range(row_nnz):
        j = D_colind[row_start + k]
        A_sparse[i, j] = A[i, j]
        qLU_np[0, 0, row_start + k] = A[i, j]

    wp.copy(d.qLU, wp.array(qLU_np, dtype=float))

    vec = wp.ones((1, nv), dtype=float)
    res = wp.zeros((1, nv), dtype=float)

    mjw._src.smooth.factor_solve_lu(m, d, d.qLU, res, vec)

    expected_res = np.linalg.solve(A_sparse, vec.numpy()[0])
    _assert_eq(res.numpy()[0], expected_res, "qLU \\ 1 (sparse)")

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_tendon_armature(self, jacobian):
    mjm, mjd, m, d = test_data.fixture("tendon/armature.xml", keyframe=0, overrides={"opt.jacobian": jacobian})

    # M
    d.M.fill_(wp.inf)

    mjw._src.smooth.crb(m, d)
    mjw._src.smooth.tendon_armature(m, d)

    _assert_eq(d.M.numpy()[0], mjd.M, "M")

    # qfrc_bias
    d.qfrc_bias.fill_(wp.inf)

    mjw._src.smooth.rne(m, d)
    mjw._src.smooth.tendon_bias(m, d, d.qfrc_bias)
    _assert_eq(d.qfrc_bias.numpy()[0], mjd.qfrc_bias, "qfrc_bias")

  def test_flex(self):
    mjm, mjd, m, d = test_data.fixture("flex/floppy.xml")
    self.assertTrue(m.is_sparse)

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

    rownnz = mjm.flexedge_J_rownnz
    rowadr = mjm.flexedge_J_rowadr
    colind = mjm.flexedge_J_colind.reshape(-1)

    mj_flexedge_J = np.zeros((mjm.nflexedge, mjm.nv), dtype=float)
    mujoco.mju_sparse2dense(mj_flexedge_J, mjd.flexedge_J.ravel(), rownnz, rowadr, colind)

    _assert_eq(d.flexvert_xpos.numpy()[0], mjd.flexvert_xpos, "flexvert_xpos")
    _assert_eq(d.flexedge_length.numpy()[0], mjd.flexedge_length, "flexedge_length")
    _assert_eq(d.flexedge_velocity.numpy()[0], mjd.flexedge_velocity, "flexedge_velocity")

    flexedge_J = np.zeros((mjm.nflexedge, mjm.nv))
    mujoco.mju_sparse2dense(
      flexedge_J,
      d.flexedge_J.numpy()[0].reshape(-1),
      m.flexedge_J_rownnz.numpy(),
      m.flexedge_J_rowadr.numpy(),
      m.flexedge_J_colind.numpy(),
    )

    _assert_eq(flexedge_J, mj_flexedge_J, "flexedge_J")

  def test_flex_1d_pinned(self):
    """Tests that 1D flex with pinned vertex computes correct Jacobian and velocity.

    This is a regression test for issue #1229 where the _flex_edges kernel
    assumed every vertex body has 3 DOFs, causing garbage reads for pinned
    vertices (body_dofnum=0, body_dofadr=-1) and out-of-plane spinning.
    """
    xml = """
    <mujoco>
      <option gravity="0 0 -10"/>
      <worldbody>
        <body name="rope" pos="0.5 0.5 1.0">
          <geom type="sphere" size="0.02" mass="0.01"/>
          <flexcomp name="line" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                    radius="0.01" dim="1" mass="1">
            <contact contype="0" conaffinity="0"/>
            <edge equality="true" damping="0.01"/>
            <pin id="0"/>
          </flexcomp>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    self.assertEqual(m.nflex, 1)
    self.assertEqual(m.nflexvert, 5)

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

    _assert_eq(d.flexvert_xpos.numpy()[0], mjd.flexvert_xpos, "flexvert_xpos")
    _assert_eq(d.flexedge_length.numpy()[0], mjd.flexedge_length, "flexedge_length")
    _assert_eq(d.flexedge_velocity.numpy()[0], mjd.flexedge_velocity, "flexedge_velocity")

    # Compare dense Jacobians
    rownnz = mjm.flexedge_J_rownnz
    rowadr = mjm.flexedge_J_rowadr
    colind = mjm.flexedge_J_colind.reshape(-1)

    mj_flexedge_J = np.zeros((mjm.nflexedge, mjm.nv), dtype=float)
    mujoco.mju_sparse2dense(mj_flexedge_J, mjd.flexedge_J.ravel(), rownnz, rowadr, colind)

    flexedge_J = np.zeros((mjm.nflexedge, mjm.nv))
    mujoco.mju_sparse2dense(
      flexedge_J,
      d.flexedge_J.numpy()[0].reshape(-1),
      m.flexedge_J_rownnz.numpy(),
      m.flexedge_J_rowadr.numpy(),
      m.flexedge_J_colind.numpy(),
    )

    _assert_eq(flexedge_J, mj_flexedge_J, "flexedge_J")


if __name__ == "__main__":
  wp.init()
  absltest.main()
