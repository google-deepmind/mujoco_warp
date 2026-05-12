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

"""Tests for derivative functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src.util_pkg import check_version

# tolerance for difference between MuJoCo and mjwarp smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class DerivativeTest(parameterized.TestCase):
  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE)
  def test_smooth_vel(self, jacobian):
    """Tests qDeriv."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option>
        <flag gravity="disable"/>
      </option>
      <worldbody>
        <body>
          <geom type="sphere" size=".1"/>
          <joint name="joint0" type="hinge" axis="0 1 0"/>
          <site name="site0" pos="0 0 1"/>
          <body pos="1 0 0">
            <geom type="sphere" size=".1"/>
            <joint name="joint1" type="hinge" axis="0 1 0"/>
            <site name="site1" pos="0 0 1"/>
            <body pos="1 0 0">
              <geom type="sphere" size=".1"/>
              <joint name="joint2" type="hinge" axis="0 1 0"/>
              <site name="site2" pos="0 0 1"/>
            </body>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="tendon0">
          <site site="site0"/>
          <site site="site1"/>
        </spatial>
        <spatial name="tendon1">
          <site site="site0"/>
          <site site="site1"/>
          <site site="site2"/>
        </spatial>
        <spatial name="tendon2">
          <site site="site0"/>
          <site site="site2"/>
        </spatial>
      </tendon>
      <actuator>
        <general joint="joint0" dyntype="filter" gaintype="affine" biastype="affine" dynprm="1 1 1 0 0 0 0 0 0 0" gainprm="1 1 1 0 0 0 0 0 0 0" biasprm="1 1 1 0 0 0 0 0 0 0"/>
        <general joint="joint1" dyntype="filter" gaintype="affine" biastype="affine" dynprm="1 1 1 0 0 0 0 0 0 0" gainprm="1 1 1 0 0 0 0 0 0 0" biasprm="1 1 1 0 0 0 0 0 0 0"/>
        <general tendon="tendon0" dyntype="filter" gaintype="affine" biastype="affine" dynprm="1 1 1 0 0 0 0 0 0 0" gainprm="1 1 1 0 0 0 0 0 0 0" biasprm="1 1 1 0 0 0 0 0 0 0"/>
        <general tendon="tendon1" dyntype="filter" gaintype="affine" biastype="affine" dynprm="1 1 1 0 0 0 0 0 0 0" gainprm="1 1 1 0 0 0 0 0 0 0" biasprm="1 1 1 0 0 0 0 0 0 0"/>
        <general tendon="tendon2" dyntype="filter" gaintype="affine" biastype="affine" dynprm="1 1 1 0 0 0 0 0 0 0" gainprm="1 1 1 0 0 0 0 0 0 0" biasprm="1 1 1 0 0 0 0 0 0 0"/>
      </actuator>
      <keyframe>
        <key qpos="0.5 1 1.5" qvel="1 2 3" act="1 2 3 4 5" ctrl="1 2 3 4 5"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=0,
      overrides={"opt.jacobian": jacobian},
    )

    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mujoco.mj_step(mjm, mjd)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      out_smooth_vel = wp.zeros((1, 1, m.nM), dtype=float)
    else:
      out_smooth_vel = wp.zeros(d.qM.shape, dtype=float)

    # Compute kinematics without factorizing qM to allow direct comparison
    forward.fwd_position(m, d, factorize=False)
    forward.fwd_velocity(m, d)

    # 1. Test with RNE Disabled (Matches MuJoCo's ImplicitFast)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      mjw_out = np.zeros((m.nv, m.nv))
      for elem, (i, j) in enumerate(zip(m.qM_fullm_i.numpy(), m.qM_fullm_j.numpy())):
        mjw_out[i, j] = out_smooth_vel.numpy()[0, 0, elem]
    else:
      mjw_out = out_smooth_vel.numpy()[0, : m.nv, : m.nv]

    # Symmetrize mjw_out (use lower triangle)
    mjw_out = np.tril(mjw_out) + np.tril(mjw_out, -1).T

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    mj_qM = np.zeros((m.nv, m.nv))
    if check_version("mujoco>=3.8.1.dev910242375"):
      mujoco.mju_sym2dense(mj_qM, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    else:
      mujoco.mj_fullM(mjm, mj_qM, mjd.qM)
    mj_out = mj_qM - mjm.opt.timestep * mj_qDeriv

    _assert_eq(mjw_out, mj_out, "qM - dt * qDeriv")

  _TENDON_SERIAL_CHAIN_XML = """
    <mujoco>
      <compiler angle="radian" autolimits="true"/>
      <option integrator="implicitfast"/>
      <default>
        <general biastype="affine"/>
      </default>
      <worldbody>
        <body>
          <inertial mass="1" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
          <joint name="parent_j" axis="0 1 0"/>
          <body pos="0 0.03 0.1">
            <inertial mass="0.01" pos="0 0 0" diaginertia="1e-06 1e-06 1e-06"/>
            <joint name="j_r" axis="1 0 0" armature="0.005" damping="0.1"/>
          </body>
          <body pos="0 -0.03 0.1">
            <inertial mass="0.01" pos="0 0 0" diaginertia="1e-06 1e-06 1e-06"/>
            <joint name="j_l" axis="1 0 0" armature="0.005" damping="0.1"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <fixed name="split">
          <joint joint="j_r" coef="0.5"/>
          <joint joint="j_l" coef="0.5"/>
        </fixed>
      </tendon>
      <actuator>
        <general name="grip" tendon="split" gainprm="80 0 0" biasprm="0 -100 -10"/>
      </actuator>
      <keyframe>
        <key qpos="0 0 0" qvel="0 0 0" ctrl="0"/>
      </keyframe>
    </mujoco>
  """

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE)
  def test_smooth_vel_tendon_serial_chain(self, jacobian):
    """Tests qDeriv for tendon actuator on serial chain."""
    mjm, mjd, m, d = test_data.fixture(
      xml=self._TENDON_SERIAL_CHAIN_XML,
      keyframe=0,
      overrides={"opt.jacobian": jacobian},
    )

    mujoco.mj_step(mjm, mjd)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      out_smooth_vel = wp.zeros((1, 1, m.nM), dtype=float)
    else:
      out_smooth_vel = wp.zeros(d.qM.shape, dtype=float)

    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      mjw_out = np.zeros((m.nv, m.nv))
      for elem, (i, j) in enumerate(zip(m.qM_fullm_i.numpy(), m.qM_fullm_j.numpy())):
        mjw_out[i, j] = out_smooth_vel.numpy()[0, 0, elem]
    else:
      mjw_out = out_smooth_vel.numpy()[0, : m.nv, : m.nv]

    # Final comparison against new ground truth: qM - dt * qDeriv
    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)
    mj_qM = np.zeros((m.nv, m.nv))
    if check_version("mujoco>=3.8.1.dev910242375"):
      mujoco.mju_sym2dense(mj_qM, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    else:
      mujoco.mj_fullM(mjm, mj_qM, mjd.qM)
    mj_out = mj_qM - mjm.opt.timestep * mj_qDeriv

    self.assertFalse(np.any(np.isnan(mjw_out)))
    _assert_eq(mjw_out, mj_out, "qM - dt * qDeriv (tendon serial chain)")

  def test_step_tendon_serial_chain_no_nan(self):
    """Regression: implicitfast + tendon on serial chain must not NaN."""
    mjm, mjd, m, d = test_data.fixture(
      xml=self._TENDON_SERIAL_CHAIN_XML,
      keyframe=0,
    )

    for _ in range(10):
      mjw.step(m, d)

    mjw.get_data_into(mjd, mjm, d)
    self.assertFalse(np.any(np.isnan(mjd.qpos)))
    self.assertFalse(np.any(np.isnan(mjd.qvel)))

  def test_smooth_vel_sparse_tendon_coupled(self):
    """Tests qDeriv kernel with nv > 32 and moment_rownnz > 1.

    Builds a chain of 35 DOFs (forcing sparse path) with a fixed tendon
    coupling two joints, producing an actuator with moment_rownnz=2.
    """
    # Build a chain long enough to force sparse (nv > 32)
    xml = f"""
    <mujoco>
      <option integrator="implicitfast">
        <flag gravity="disable"/>
      </option>
      <worldbody>
        <body pos="0.1 0 0">
          <geom type="sphere" size=".05"/>
          <joint name="j0" type="hinge" axis="0 1 0"/>
          <body pos="0.1 0 0">
            <geom type="sphere" size=".05"/>
            <joint name="j1" type="hinge" axis="0 1 0"/>
            <body pos="0.1 0 0">
              <geom type="sphere" size=".05"/>
              <joint name="j2" type="hinge" axis="0 1 0"/>
              <body pos="0.1 0 0">
                <geom type="sphere" size=".05"/>
                <joint name="j3" type="hinge" axis="0 1 0"/>
                <body pos="0.1 0 0">
                  <geom type="sphere" size=".05"/>
                  <joint name="j4" type="hinge" axis="0 1 0"/>
                  <body pos="0.1 0 0">
                    <geom type="sphere" size=".05"/>
                    <joint name="j5" type="hinge" axis="0 1 0"/>
                    <body pos="0.1 0 0">
                      <geom type="sphere" size=".05"/>
                      <joint name="j6" type="hinge" axis="0 1 0"/>
                      <body pos="0.1 0 0">
                        <geom type="sphere" size=".05"/>
                        <joint name="j7" type="hinge" axis="0 1 0"/>
                        <body pos="0.1 0 0">
                          <geom type="sphere" size=".05"/>
                          <joint name="j8" type="hinge" axis="0 1 0"/>
                          <body pos="0.1 0 0">
                            <geom type="sphere" size=".05"/>
                            <joint name="j9" type="hinge" axis="0 1 0"/>
                            <body pos="0.1 0 0">
                              <geom type="sphere" size=".05"/>
                              <joint name="j10" type="hinge" axis="0 1 0"/>
                              <body pos="0.1 0 0">
                                <geom type="sphere" size=".05"/>
                                <joint name="j11" type="hinge" axis="0 1 0"/>
                                <body pos="0.1 0 0">
                                  <geom type="sphere" size=".05"/>
                                  <joint name="j12" type="hinge" axis="0 1 0"/>
                                  <body pos="0.1 0 0">
                                    <geom type="sphere" size=".05"/>
                                    <joint name="j13" type="hinge" axis="0 1 0"/>
                                    <body pos="0.1 0 0">
                                      <geom type="sphere" size=".05"/>
                                      <joint name="j14" type="hinge" axis="0 1 0"/>
                                      <body pos="0.1 0 0">
                                        <geom type="sphere" size=".05"/>
                                        <joint name="j15" type="hinge" axis="0 1 0"/>
                                        <body pos="0.1 0 0">
                                          <geom type="sphere" size=".05"/>
                                          <joint name="j16" type="hinge" axis="0 1 0"/>
                                          <body pos="0.1 0 0">
                                            <geom type="sphere" size=".05"/>
                                            <joint name="j17" type="hinge" axis="0 1 0"/>
                                            <body pos="0.1 0 0">
                                              <geom type="sphere" size=".05"/>
                                              <joint name="j18" type="hinge" axis="0 1 0"/>
                                              <body pos="0.1 0 0">
                                                <geom type="sphere" size=".05"/>
                                                <joint name="j19" type="hinge" axis="0 1 0"/>
                                                <body pos="0.1 0 0">
                                                  <geom type="sphere" size=".05"/>
                                                  <joint name="j20" type="hinge" axis="0 1 0"/>
                                                  <body pos="0.1 0 0">
                                                    <geom type="sphere" size=".05"/>
                                                    <joint name="j21" type="hinge" axis="0 1 0"/>
                                                    <body pos="0.1 0 0">
                                                      <geom type="sphere" size=".05"/>
                                                      <joint name="j22" type="hinge" axis="0 1 0"/>
                                                      <body pos="0.1 0 0">
                                                        <geom type="sphere" size=".05"/>
                                                        <joint name="j23" type="hinge" axis="0 1 0"/>
                                                        <body pos="0.1 0 0">
                                                          <geom type="sphere" size=".05"/>
                                                          <joint name="j24" type="hinge" axis="0 1 0"/>
                                                          <body pos="0.1 0 0">
                                                            <geom type="sphere" size=".05"/>
                                                            <joint name="j25" type="hinge" axis="0 1 0"/>
                                                            <body pos="0.1 0 0">
                                                              <geom type="sphere" size=".05"/>
                                                              <joint name="j26" type="hinge" axis="0 1 0"/>
                                                              <body pos="0.1 0 0">
                                                                <geom type="sphere" size=".05"/>
                                                                <joint name="j27" type="hinge" axis="0 1 0"/>
                                                                <body pos="0.1 0 0">
                                                                  <geom type="sphere" size=".05"/>
                                                                  <joint name="j28" type="hinge" axis="0 1 0"/>
                                                                  <body pos="0.1 0 0">
                                                                    <geom type="sphere" size=".05"/>
                                                                    <joint name="j29" type="hinge" axis="0 1 0"/>
                                                                    <body pos="0.1 0 0">
                                                                      <geom type="sphere" size=".05"/>
                                                                      <joint name="j30" type="hinge" axis="0 1 0"/>
                                                                      <body pos="0.1 0 0">
                                                                        <geom type="sphere" size=".05"/>
                                                                        <joint name="j31" type="hinge" axis="0 1 0"/>
                                                                        <body pos="0.1 0 0">
                                                                          <geom type="sphere" size=".05"/>
                                                                          <joint name="j32" type="hinge" axis="0 1 0"/>
                                                                          <body pos="0.1 0 0">
                                                                            <geom type="sphere" size=".05"/>
                                                                            <joint name="j33" type="hinge" axis="0 1 0"/>
                                                                            <body pos="0.1 0 0">
                                                                              <geom type="sphere" size=".05"/>
                                                                              <joint name="j34" type="hinge" axis="0 1 0"/>
                                                                            </body>
                                                                          </body>
                                                                        </body>
                                                                      </body>
                                                                    </body>
                                                                  </body>
                                                                </body>
                                                              </body>
                                                            </body>
                                                          </body>
                                                        </body>
                                                      </body>
                                                    </body>
                                                  </body>
                                                </body>
                                              </body>
                                            </body>
                                          </body>
                                        </body>
                                      </body>
                                    </body>
                                  </body>
                                </body>
                              </body>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <tendon>
        <fixed name="coupled">
          <joint joint="j10" coef="1"/>
          <joint joint="j11" coef="0.5"/>
        </fixed>
      </tendon>
      <actuator>
        <general tendon="coupled" gainprm="100" biasprm="0 -100 0"
                 dyntype="none" gaintype="fixed" biastype="affine"/>
        <motor joint="j0" gear="1"/>
        <motor joint="j5" gear="1"/>
        <motor joint="j20" gear="1"/>
      </actuator>
      <keyframe>
        <key qpos="0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1" qvel="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1" ctrl="1 1 1 1"/>
      </keyframe>
    </mujoco>
    """

    mjm, mjd, m, d = test_data.fixture(
      xml=xml,
      keyframe=0,
      overrides={"opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE},
    )

    self.assertTrue(m.is_sparse, "Model should use sparse path (nv > 32)")

    mujoco.mj_step(mjm, mjd)

    out_smooth_vel = wp.zeros((1, 1, m.nM), dtype=float)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    mjw_out = np.zeros((m.nv, m.nv))
    for elem, (i, j) in enumerate(zip(m.qM_fullm_i.numpy(), m.qM_fullm_j.numpy())):
      mjw_out[i, j] = out_smooth_vel.numpy()[0, 0, elem]
      mjw_out[j, i] = out_smooth_vel.numpy()[0, 0, elem]

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    mj_qM = np.zeros((m.nv, m.nv))
    if check_version("mujoco>=3.8.1.dev910242375"):
      mujoco.mju_sym2dense(mj_qM, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    else:
      mujoco.mj_fullM(mjm, mj_qM, mjd.qM)
    mj_out = mj_qM - mjm.opt.timestep * mj_qDeriv

    self.assertFalse(np.any(np.isnan(mjw_out)))
    _assert_eq(mjw_out, mj_out, "qM - dt * qDeriv (sparse tendon coupled)")

  def test_smooth_vel_sparse_free_joint_precedes_actuator(self):
    """Sparse qDeriv uses chain-aware row offsets when indexing qM_fullm.

    A free joint's internal block is stored diagonal-only in the compact
    (M_rownnz, M_rowadr) layout but fully chained (1+2+...+n entries per
    internal dof) in qM_fullm_i / qM_fullm_j. For any actuated dof that
    follows the free joint in qvel order, indexing qM_fullm with the compact
    offsets lands in slots that belong to the free-joint block, the actuator
    contribution to qDeriv is silently dropped, and the implicit step loses
    damping for the actuated dof.
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option integrator="implicitfast">
        <flag gravity="disable"/>
      </option>
      <worldbody>
        <body>
          <joint type="free"/>
          <geom type="sphere" size="0.05" mass="1"/>
        </body>
        <body pos="1 0 0">
          <joint name="hinge0" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.05" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <position joint="hinge0" kp="10" kv="1"/>
      </actuator>
      <keyframe>
        <key qpos="0 0 0 1 0 0 0 0" qvel="0 0 0 0 0 0 1" ctrl="0.1"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=0,
      overrides={"opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE},
    )

    mujoco.mj_step(mjm, mjd)

    out_smooth_vel = wp.zeros((1, 1, m.nM), dtype=float)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    mjw_out = np.zeros((m.nv, m.nv))
    for elem, (i, j) in enumerate(zip(m.qM_fullm_i.numpy(), m.qM_fullm_j.numpy())):
      mjw_out[i, j] = out_smooth_vel.numpy()[0, 0, elem]
      mjw_out[j, i] = out_smooth_vel.numpy()[0, 0, elem]

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)
    mj_qM = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(mjm, mj_qM, mjd.qM)
    mj_out = mj_qM - mjm.opt.timestep * mj_qDeriv

    self.assertFalse(np.any(np.isnan(mjw_out)))
    _assert_eq(mjw_out, mj_out, "qM - dt * qDeriv (sparse, free joint precedes actuated dof)")

  def test_actearly_derivative(self):
    """Implicit derivatives should use next activation when actearly is set."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option timestep="1" integrator="implicitfast"/>
      <worldbody>
        <body>
          <joint name="early" type="slide"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
        <body pos="1 0 0">
          <joint name="late" type="slide"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <general joint="early" dyntype="integrator" gaintype="affine"
                 gainprm="1 0 1" actearly="true"/>
        <general joint="late" dyntype="integrator" gaintype="affine"
                 gainprm="1 0 1" actearly="false"/>
      </actuator>
      <keyframe>
        <key ctrl="1 1" act="0 0"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=0,
    )

    # both should have same act_dot (ctrl = 1 for integrator dynamics)
    _assert_eq(d.act_dot.numpy()[0, 0], d.act_dot.numpy()[0, 1], "act_dot")

    # compute qDeriv using deriv_smooth_vel
    out_smooth_vel = wp.zeros(d.qM.shape, dtype=float)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel)
    mjw_out = out_smooth_vel.numpy()[0, : m.nv, : m.nv]

    # with actearly=true and nonzero act_dot, derivative should differ
    # because actearly uses next activation: act + act_dot*dt
    # for our model: next_act = 0 + 1*1 = 1, current_act = 0
    # derivative adds gain_vel * act to qDeriv diagonal
    # qDeriv = qM - dt * actuator_vel_derivative
    # for independent bodies with mass=1: qM diagonal = 1.0
    # actearly=true: vel = gain_vel * next_act = 1 * 1 = 1, out = 1 - 1*1 = 0
    # actearly=false: vel = gain_vel * current_act = 1 * 0 = 0, out = 1 - 1*0 = 1
    self.assertNotAlmostEqual(
      mjw_out[0, 0],
      mjw_out[1, 1],
      msg="actearly=true should use next activation in derivative",
    )
    _assert_eq(mjw_out[0, 0], 0.0, "actearly=true: qM - dt*gain_vel*next_act = 1 - 1*1 = 0")
    _assert_eq(mjw_out[1, 1], 1.0, "actearly=false: qM - dt*gain_vel*current_act = 1 - 1*0 = 1")

  def test_forcerange_clamped_derivative(self):
    """Implicit integration is more accurate than Euler with active forcerange clamping."""
    xml = """
    <mujoco>
      <option timestep="0.01" integrator="implicitfast"/>
      <worldbody>
        <geom type="plane" size="10 10 0.001"/>
        <body pos="0 0 1">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <position joint="slide" kp="10000" kv="1000" forcerange="-10 10"/>
      </actuator>
    </mujoco>
    """

    dt_small = 5e-4
    dt_large = 5e-2
    duration = 1.0
    nsteps_large = int(duration / dt_large)
    nsubstep = int(dt_large / dt_small)

    # ground truth: Euler with small timestep
    mjm_gt = mujoco.MjModel.from_xml_string(xml)
    mjd_gt = mujoco.MjData(mjm_gt)
    mjm_gt.opt.timestep = dt_small
    mjm_gt.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mujoco.mj_resetData(mjm_gt, mjd_gt)
    mjd_gt.ctrl[0] = 0.5

    # implicitfast at large timestep
    mjm_impl, mjd_impl, m_impl, d_impl = test_data.fixture(xml=xml)
    m_impl.opt.timestep.fill_(dt_large)
    m_impl.opt.integrator = int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    d_impl.ctrl.fill_(0.5)

    # euler at large timestep
    mjm_euler, mjd_euler, m_euler, d_euler = test_data.fixture(xml=xml)
    m_euler.opt.timestep.fill_(dt_large)
    m_euler.opt.integrator = int(mujoco.mjtIntegrator.mjINT_EULER)
    d_euler.ctrl.fill_(0.5)

    error_implicit = 0.0
    error_euler = 0.0

    for _ in range(nsteps_large):
      # ground truth: small steps with Euler
      mujoco.mj_step(mjm_gt, mjd_gt, nsubstep)

      # implicit at large timestep
      mjw.step(m_impl, d_impl)

      # euler at large timestep
      mjw.step(m_euler, d_euler)

      # accumulate errors
      gt_qpos = mjd_gt.qpos[0]
      diff_implicit = gt_qpos - d_impl.qpos.numpy()[0, 0]
      diff_euler = gt_qpos - d_euler.qpos.numpy()[0, 0]
      error_implicit += diff_implicit * diff_implicit
      error_euler += diff_euler * diff_euler

    self.assertLess(
      error_implicit,
      error_euler,
      "implicitfast should be more accurate than Euler at large timestep when forcerange derivatives are correctly handled",
    )

  _RNE_MODELS = {
    "hinge_chain": """
    <mujoco>
      <option>
        <flag gravity="disable"/>
      </option>
      <worldbody>
        <body>
          <geom type="capsule" size=".05" fromto="0 0 0 0.3 0 0"/>
          <joint name="j0" type="hinge" axis="0 1 0"/>
          <body pos="0.3 0 0">
            <geom type="capsule" size=".05" fromto="0 0 0 0.3 0 0"/>
            <joint name="j1" type="hinge" axis="0 1 0"/>
            <body pos="0.3 0 0">
              <geom type="capsule" size=".05" fromto="0 0 0 0.3 0 0"/>
              <joint name="j2" type="hinge" axis="0 0 1"/>
              <body pos="0.3 0 0">
                <geom type="capsule" size=".05" fromto="0 0 0 0.3 0 0"/>
                <joint name="j3" type="hinge" axis="1 0 0"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <keyframe>
        <key qpos="0.5 1.0 -0.3 0.7" qvel="2.0 -1.5 3.0 -0.5"/>
      </keyframe>
    </mujoco>
    """,
    "free_joint": """
    <mujoco>
      <option>
        <flag gravity="disable" contact="disable"/>
      </option>
      <worldbody>
        <body>
          <freejoint/>
          <geom type="box" size=".1 .2 .3" mass="1"/>
        </body>
      </worldbody>
      <keyframe>
        <key qpos="0 0 1 1 0 0 0" qvel="1.0 -0.5 0.3 0.2 -0.8 1.5"/>
      </keyframe>
    </mujoco>
    """,
    "mixed_chain": """
    <mujoco>
      <option>
        <flag constraint="disable"/>
      </option>
      <worldbody>
        <body pos="0.15 0 0">
          <joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
          <body pos="0.1 0 0">
            <joint type="slide" axis="1 0 0"/>
            <geom type="capsule" size="0.015" fromto="-.1 0 0 .1 0 0"/>
            <body pos=".1 0 0">
              <joint type="ball"/>
              <geom type="box" size=".02" fromto="0 0 0 0 .1 0"/>
              <body pos="0 .1 0">
                <joint axis="1 0 0"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0 .1 0"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <keyframe>
        <key qpos="0.974068045 0.09778919028 0.8824329717 0.2947814751 -0.3653561696 0.03050904378 -0.5772862322"
             qvel="5.241365683 1.565431578 3.319814864 -2.229322074 -0.2737814514 -1.358480177"/>
      </keyframe>
    </mujoco>
    """,
    "branched": """
    <mujoco>
      <option/>
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
      <keyframe>
        <key qpos="-0.198162 0 -0.198162 0" qvel="-1.962 0 -1.962 0"/>
      </keyframe>
    </mujoco>
    """,
    "tumbling_thin_object": """
    <mujoco>
      <option density="1.225" viscosity="1.8e-5" wind="0 0 1">
        <flag constraint="disable"/>
      </option>
      <worldbody>
        <body>
          <freejoint/>
          <body>
            <geom type="box" size=".025 .01 0.0001" pos=".025 0 0" euler="20 0 0" mass="1e-4"/>
          </body>
          <body>
            <geom type="box" size=".025 .01 0.0001" pos="-.025 0 0" euler="-19 0 0" mass="1e-4"/>
          </body>
        </body>
      </worldbody>
      <keyframe>
        <key qpos="0.0004946999296 -0.001202060483 -0.0555835832 0.7523249953 0.001535295742 0.01207989364 -0.6586796038"
             qvel="0.007826319845 -0.0166791028 -0.4384455175 0.09592236433 0.1566970362 -15.31785051"/>
      </keyframe>
    </mujoco>
    """,
    "tumbling_ellipsoid": """
    <mujoco>
      <option density="1.225" viscosity="1.8e-5" wind="0 0 1">
        <flag constraint="disable"/>
      </option>
      <worldbody>
        <body>
          <freejoint/>
          <geom type="box" size=".025 .01 0.0001" pos=".025 0 0" euler="20 0 0" mass="1e-4" fluidshape="ellipsoid"/>
          <geom type="box" size=".025 .01 0.0001" pos="-.025 0 0" euler="-19 0 0" mass="1e-4" fluidshape="ellipsoid"/>
        </body>
      </worldbody>
      <keyframe>
        <key qpos="-7.555310792e-05 9.606312749e-05 -0.03860222825 0.3270616399 -0.002154923278 0.005836083002 -0.944982529"
             qvel="-0.0002248200636 3.956744354e-06 -0.1660734509 0.000756193568 0.2852354411 -23.83662532"/>
      </keyframe>
    </mujoco>
    """,
    "pendulum_stiffness": """
    <mujoco>
      <option>
        <flag constraint="disable"/>
      </option>
      <worldbody>
        <body pos="0.15 0 0">
          <joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
          <body pos="0.1 0 0">
            <joint type="slide" axis="1 0 0" stiffness="200"/>
            <geom type="capsule" size="0.015" fromto="-.1 0 0 .1 0 0"/>
            <body pos=".1 0 0">
              <joint type="ball"/>
              <geom type="box" size=".02" fromto="0 0 0 0 .1 0"/>
              <body pos="0 .1 0">
                <joint axis="1 0 0"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0 .1 0"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <keyframe>
        <key qpos="1.046766071 0.02856001812 0.9491354349 0.2279934469 -0.09267758993 -0.1963969926 0.05635068068"
             qvel="6.031731375 0.1635244246 -3.466530779 9.931235553 -9.339705364 16.62432373"/>
      </keyframe>
    </mujoco>
    """,
    "damped_pendulum": """
    <mujoco>
      <default>
        <joint damping=".01"/>
      </default>
      <option>
        <flag constraint="disable"/>
      </option>
      <worldbody>
        <body pos="0.15 0 0">
          <joint name="hinge" axis="0 1 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
          <body pos="0.1 0 0">
            <joint type="slide" axis="1 0 0" stiffness="200"/>
            <geom type="capsule" size="0.015" fromto="-.1 0 0 .1 0 0"/>
            <body pos=".1 0 0">
              <joint type="ball"/>
              <geom type="box" size=".02" fromto="0 0 0 0 .1 0"/>
              <body pos="0 .1 0">
                <joint axis="1 0 0"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0 0 .1"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <keyframe>
        <key qpos="0.9260158353 0.02554858795 0.9692205685 0.07944192314 -0.115133083 -0.2025952704 -0.12351129"
             qvel="6.446399218 0.2195989631 0.230254317 0.2110774142 -9.605579935 -3.906956847"/>
      </keyframe>
    </mujoco>
    """,
  }

  @parameterized.named_parameters(
    [dict(testcase_name=f"{name}_dense", xml_name=name, jacobian=mujoco.mjtJacobian.mjJAC_DENSE) for name in _RNE_MODELS]
    + [dict(testcase_name=f"{name}_sparse", xml_name=name, jacobian=mujoco.mjtJacobian.mjJAC_SPARSE) for name in _RNE_MODELS]
  )
  def test_rne_derivative(self, xml_name, jacobian):
    """Tests RNE derivative matches MuJoCo's qDeriv with mjINT_IMPLICIT."""
    mjm, mjd, m, d = test_data.fixture(
      xml=self._RNE_MODELS[xml_name],
      keyframe=0,
      overrides={"opt.jacobian": jacobian},
    )

    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    mujoco.mj_step(mjm, mjd)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      out_rne = wp.zeros((1, 1, m.nD), dtype=float)
    else:
      out_rne = wp.zeros(d.qM.shape, dtype=float)

    derivative.rne_vel_deriv(m, d, out_rne)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      mjw_rne = np.zeros((m.nv, m.nv))
      for elem, (i, j) in enumerate(zip(m.qD_fullm_i.numpy(), m.qD_fullm_j.numpy())):
        mjw_rne[i, j] = out_rne.numpy()[0, 0, elem]
    else:
      mjw_rne = out_rne.numpy()[0, : m.nv, : m.nv]

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    _assert_eq(mjw_rne, -mjm.opt.timestep * mj_qDeriv, f"RNE {xml_name}")


if __name__ == "__main__":
  wp.init()
  absltest.main()
