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
from mujoco_warp._src import types

# tolerance for difference between MuJoCo and mjwarp smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10
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

    # deriv_smooth_vel outputs in M's CSR layout (nC entries).
    out_smooth_vel = wp.zeros((1, m.nC), dtype=float)

    # Compute kinematics without factorizing M to allow direct comparison
    forward.fwd_position(m, d, factorize=False)
    forward.fwd_velocity(m, d)

    # 1. Test with RNE Disabled (Matches MuJoCo's ImplicitFast)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    mjw_out = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(
      mjw_out, out_smooth_vel.numpy().reshape(-1).astype(np.float64), mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind
    )

    # Symmetrize mjw_out (use lower triangle)
    mjw_out = np.tril(mjw_out) + np.tril(mjw_out, -1).T

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    mj_M = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(mj_M, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    mj_out = mj_M - mjm.opt.timestep * mj_qDeriv

    _assert_eq(mjw_out, mj_out, "M - dt * qDeriv")

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

    # deriv_smooth_vel outputs in M's CSR layout (nC entries).
    out_smooth_vel = wp.zeros((1, m.nC), dtype=float)

    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    mjw_out = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(
      mjw_out, out_smooth_vel.numpy().reshape(-1).astype(np.float64), mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind
    )

    # Final comparison against new ground truth: M - dt * qDeriv
    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)
    mj_M = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(mj_M, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    mj_out = mj_M - mjm.opt.timestep * mj_qDeriv

    self.assertFalse(np.any(np.isnan(mjw_out)))
    _assert_eq(mjw_out, mj_out, "M - dt * qDeriv (tendon serial chain)")

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

    out_smooth_vel = wp.zeros((1, m.nC), dtype=float)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    mjw_out = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(
      mjw_out, out_smooth_vel.numpy().reshape(-1).astype(np.float64), mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind
    )

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    mj_M = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(mj_M, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    mj_out = mj_M - mjm.opt.timestep * mj_qDeriv

    self.assertFalse(np.any(np.isnan(mjw_out)))
    _assert_eq(mjw_out, mj_out, "M - dt * qDeriv (sparse tendon coupled)")

  def test_smooth_vel_sparse_free_joint_precedes_actuator(self):
    """Sparse qDeriv uses chain-aware row offsets when indexing M_fullm.

    A free joint's internal block is stored diagonal-only in the compact
    (M_rownnz, M_rowadr) layout but fully chained (1+2+...+n entries per
    internal dof) in M_fullm_i / M_fullm_j. For any actuated dof that
    follows the free joint in qvel order, indexing M_fullm with the compact
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

    out_smooth_vel = wp.zeros((1, m.nC), dtype=float)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    mjw_out = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(
      mjw_out, out_smooth_vel.numpy().reshape(-1).astype(np.float64), mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind
    )

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)
    mj_M = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(mj_M, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    mj_out = mj_M - mjm.opt.timestep * mj_qDeriv

    self.assertFalse(np.any(np.isnan(mjw_out)))
    _assert_eq(mjw_out, mj_out, "M - dt * qDeriv (sparse, free joint precedes actuated dof)")

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

    # compute qDeriv using deriv_smooth_vel (M's CSR layout, nC entries)
    out_smooth_vel = wp.zeros((1, m.nC), dtype=float)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel)
    mjw_out = np.zeros((m.nv, m.nv))
    qi = m.M_fullm_i.numpy()
    qj = m.M_fullm_j.numpy()
    for elem, (i, j) in enumerate(zip(qi, qj)):
      mjw_out[i, j] = out_smooth_vel.numpy()[0, elem]

    # with actearly=true and nonzero act_dot, derivative should differ
    # because actearly uses next activation: act + act_dot*dt
    # for our model: next_act = 0 + 1*1 = 1, current_act = 0
    # derivative adds gain_vel * act to qDeriv diagonal
    # qDeriv = M - dt * actuator_vel_derivative
    # for independent bodies with mass=1: M diagonal = 1.0
    # actearly=true: vel = gain_vel * next_act = 1 * 1 = 1, out = 1 - 1*1 = 0
    # actearly=false: vel = gain_vel * current_act = 1 * 0 = 0, out = 1 - 1*0 = 1
    self.assertNotAlmostEqual(
      mjw_out[0, 0],
      mjw_out[1, 1],
      msg="actearly=true should use next activation in derivative",
    )
    _assert_eq(mjw_out[0, 0], 0.0, "actearly=true: M - dt*gain_vel*next_act = 1 - 1*1 = 0")
    _assert_eq(mjw_out[1, 1], 1.0, "actearly=false: M - dt*gain_vel*current_act = 1 - 1*0 = 1")

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

    out_rne = wp.zeros((1, m.nD), dtype=float)
    derivative.deriv_rne_vel(m, d, out_rne)

    mjw_rne = np.zeros((m.nv, m.nv))
    for elem, (i, j) in enumerate(zip(m.qD_fullm_i.numpy(), m.qD_fullm_j.numpy())):
      mjw_rne[i, j] = out_rne.numpy()[0, elem]

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    _assert_eq(mjw_rne, -mjm.opt.timestep * mj_qDeriv, f"RNE {xml_name}")

  def test_passive_derivatives(self):
    """Tests derivatives for passive forces (joint and tendon stiffness/damping)."""
    xml = """
    <mujoco>
      <option integrator="implicitfast"/>
      <worldbody>
        <body name="box">
          <joint type="hinge" axis="0 1 0" stiffness="10" damping="1"/>
          <geom type="sphere" size=".1" mass="1"/>
          <site name="site1" pos="0 0 0.1"/>
        </body>
        <site name="site2" pos="0 0 1"/>
      </worldbody>
      <tendon>
        <spatial stiffness="100" damping="10">
          <site site="site1"/>
          <site site="site2"/>
        </spatial>
      </tendon>
      <keyframe>
        <key qpos="0.5" qvel="2.0"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(
      xml=xml,
      keyframe=0,
    )

    mujoco.mj_step(mjm, mjd)

    # deriv_smooth_vel outputs in M's CSR layout (nC entries).
    out_smooth_vel = wp.zeros((1, m.nC), dtype=float)
    forward.fwd_position(m, d, factorize=False)
    forward.fwd_velocity(m, d)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    mjw_out = np.zeros((m.nv, m.nv))
    qi = m.M_fullm_i.numpy()
    qj = m.M_fullm_j.numpy()
    for elem, (i, j) in enumerate(zip(qi, qj)):
      mjw_out[i, j] = out_smooth_vel.numpy()[0, elem]
    mjw_out = np.tril(mjw_out) + np.tril(mjw_out, -1).T

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    mj_M = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(mj_M, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    expected_out = mj_M - mjm.opt.timestep * mj_qDeriv

    self.assertFalse(np.any(np.isnan(mjw_out)))
    _assert_eq(mjw_out, expected_out, "M - dt * qDeriv")

  _DCMOTOR_XML = """
    <mujoco>
      <worldbody>
        <body name="motor1" pos="0 0.1 0">
          <joint name="slide1" type="slide" axis="0 0 1"/>
          <geom size=".03"/>
        </body>
        <body name="motor2" pos="0 0.2 0">
          <joint name="slide2" type="slide" axis="0 0 1"/>
          <geom size=".03"/>
        </body>
        <body name="motor3" pos="0 0.3 0">
          <joint name="slide3" type="slide" axis="0 0 1"/>
          <geom size=".03"/>
        </body>
        <body name="motor4" pos="0 0.4 0">
          <joint name="joint4"/>
          <geom size=".03"/>
        </body>
        <body name="motor5" pos="0 0.5 0">
          <joint name="slide5" type="slide" axis="0 0 1"/>
          <geom size=".03"/>
        </body>
        <body name="motor6" pos="0 0.6 0">
          <joint name="slide6" type="slide" axis="0 0 1"/>
          <geom size=".03"/>
        </body>
        <body name="motor7" pos="0 0.7 0">
          <joint name="slide7" type="slide" axis="0 0 1"/>
          <geom size=".03"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor name="dc_bias" joint="slide1"
                 motorconst="2.0" resistance="0.5" input="voltage"/>
        <dcmotor name="dc_vel" joint="slide2"
                 motorconst="1.0" resistance="1.0"
                 input="vel" controller="0 0 5"/>
        <dcmotor name="dc_pos" joint="slide3"
                 motorconst="1.0" resistance="1.0"
                 input="pos vel" controller="10 0 5"/>
        <dcmotor name="dc_lugre" joint="joint4"
                 motorconst="0.05" resistance="2.0"
                 input="none"
                 lugre="1e4 100 0.005 0.008 0.1"/>
        <dcmotor name="dc_stateful_v" joint="slide5"
                 motorconst="2.0" resistance="0.5"
                 inductance="0.001" input="voltage"/>
        <dcmotor name="dc_stateful_pos" joint="slide6"
                 motorconst="1.0" resistance="1.0"
                 inductance="0.001"
                 input="pos vel" controller="10 0 5"/>
        <dcmotor name="dc_stateful_vel" joint="slide7"
                 motorconst="1.0" resistance="1.0"
                 inductance="0.001"
                 input="vel" controller="0 0 5"/>
      </actuator>
      <keyframe>
        <key qvel="1 2 3 4 5 6 7" ctrl=".1 .2 .3 .4 .5 .6 .7 .8"/>
      </keyframe>
    </mujoco>
  """

  @absltest.skip("TODO(team): Support dcmotor setpoint controller redesign.")
  @parameterized.parameters(
    mujoco.mjtJacobian.mjJAC_DENSE,
    mujoco.mjtJacobian.mjJAC_SPARSE,
  )
  @absltest.skip("TODO(team): Support dcmotor setpoint controller redesign.")
  def test_smooth_vel_dcmotor(self, jacobian):
    """Tests qDeriv parity with MuJoCo C for all DCMotor modes."""
    mjm, mjd, m, d = test_data.fixture(
      xml=self._DCMOTOR_XML,
      keyframe=0,
      overrides={"opt.jacobian": jacobian},
    )

    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mujoco.mj_step(mjm, mjd)

    # deriv_smooth_vel outputs in M's CSR layout (nC entries).
    out = wp.zeros((1, m.nC), dtype=float)

    forward.fwd_position(m, d, factorize=False)
    forward.fwd_velocity(m, d)
    derivative.deriv_smooth_vel(m, d, out)

    mjw_out = np.zeros((m.nv, m.nv))
    qi = m.M_fullm_i.numpy()
    qj = m.M_fullm_j.numpy()
    for elem, (i, j) in enumerate(zip(qi, qj)):
      mjw_out[i, j] = out.numpy()[0, elem]

    mjw_out = np.tril(mjw_out) + np.tril(mjw_out, -1).T

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(
      mj_qDeriv,
      mjd.qDeriv,
      mjm.D_rownnz,
      mjm.D_rowadr,
      mjm.D_colind,
    )

    mj_M = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(mj_M, mjd.M, mjm.M_rownnz, mjm.M_rowadr, mjm.M_colind)
    expected_out = mj_M - mjm.opt.timestep * mj_qDeriv

    _assert_eq(mjw_out, expected_out, "M - dt * qDeriv DCMotor")

  @absltest.skip("TODO(team): Support dcmotor setpoint controller redesign.")
  def test_dcmotor_stateful_analytical(self):
    """Stateful DCMotor derivative matches analytical formula."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <body>
          <joint name="j" type="slide"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor name="dc" joint="j" motorconst="2.0"
                 resistance="0.5" inductance="0.0005"
                 input="pos vel" controller="10 0 5"/>
      </actuator>
    </mujoco>
    """,
    )

    # Set nonzero velocity and ctrl
    mjd.qvel[0] = 1.0
    mjd.ctrl[0] = 0.5
    mjd.ctrl[1] = 0.0
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)

    out = wp.zeros((1, m.nC), dtype=float)
    forward.fwd_position(m, d, factorize=False)
    forward.fwd_velocity(m, d)
    derivative.deriv_smooth_vel(m, d, out)

    # Expected: K*(dVdw - K)*(1 - exp(-h/te))/R = -kd*(1 - exp(-h/te))
    # K=2, R=0.5, te=0.001 (L/R = 0.0005/0.5), h=0.002, kd=5
    te, h, kd = 0.001, 0.002, 5.0
    expected = -kd * (1.0 - np.exp(-h / te))

    # Extract diagonal
    out_diag = out.numpy()[0, 0]
    dt = mjm.opt.timestep
    qDeriv_diag = (1.0 - out_diag) / dt

    np.testing.assert_allclose(
      qDeriv_diag,
      expected,
      atol=1e-4,
      err_msg="stateful DCMotor derivative vs formula",
    )

  @absltest.skip("TODO(team): Support dcmotor setpoint controller redesign.")
  def test_dcmotor_stateful_converges_to_stateless(self):
    """Stateful DCMotor derivative converges to stateless as te->0."""
    xml_stateless = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <body>
          <joint name="j" type="slide"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor name="dc" joint="j" motorconst="1.0"
                 resistance="1.0" input="pos vel"
                 controller="10 0 5"/>
      </actuator>
    </mujoco>
    """

    xml_stateful = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <body>
          <joint name="j" type="slide"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor name="dc" joint="j" motorconst="1.0"
                 resistance="1.0" inductance="1e-8"
                 input="pos vel" controller="10 0 5"/>
      </actuator>
    </mujoco>
    """

    # Stateless
    mjm_sl, mjd_sl, m_sl, d_sl = test_data.fixture(xml=xml_stateless)
    mjd_sl.qvel[0] = 1.0
    mjd_sl.ctrl[0] = 0.5
    mujoco.mj_forward(mjm_sl, mjd_sl)
    d_sl = mjw.put_data(mjm_sl, mjd_sl)
    out_sl = wp.zeros((1, m_sl.nC), dtype=float)
    forward.fwd_position(m_sl, d_sl, factorize=False)
    forward.fwd_velocity(m_sl, d_sl)
    derivative.deriv_smooth_vel(m_sl, d_sl, out_sl)

    # Stateful (te ≈ 0)
    mjm_sf, mjd_sf, m_sf, d_sf = test_data.fixture(xml=xml_stateful)
    mjd_sf.qvel[0] = 1.0
    mjd_sf.ctrl[0] = 0.5
    mujoco.mj_forward(mjm_sf, mjd_sf)
    d_sf = mjw.put_data(mjm_sf, mjd_sf)
    out_sf = wp.zeros((1, m_sf.nC), dtype=float)
    forward.fwd_position(m_sf, d_sf, factorize=False)
    forward.fwd_velocity(m_sf, d_sf)
    derivative.deriv_smooth_vel(m_sf, d_sf, out_sf)

    # Extract diagonals
    val_sl = out_sl.numpy()[0, 0]
    val_sf = out_sf.numpy()[0, 0]

    np.testing.assert_allclose(
      val_sf,
      val_sl,
      atol=1e-4,
      err_msg="stateful should converge to stateless as te->0",
    )

  _FLUID_SCENARIOS = {
    "basic": """
      <mujoco>
        <option integrator="implicitfast" density="1000" viscosity="0.002"/>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom type="ellipsoid" size="0.1 0.15 0.2" pos="0.05 0.05 0.05"
                  euler="10 20 30" fluidshape="ellipsoid"
                  fluidcoef="1.0 1.5 2.0 0.5 0.8"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="1 2 3 0.5 0.8 1.2"/>
        </keyframe>
      </mujoco>
      """,
    "chain": """
      <mujoco>
        <option integrator="implicitfast" density="1000" viscosity="0.002"/>
        <worldbody>
          <body>
            <joint type="hinge" axis="0 1 0"/>
            <geom type="sphere" size="0.05" pos="0.01 0.02 0.03" euler="5 10 15"
                  fluidshape="ellipsoid"/>
            <body pos="0.2 0 0">
              <joint type="hinge" axis="0 1 0"/>
              <geom type="ellipsoid" size="0.1 0.15 0.2" pos="0.02 0.03 0.04"
                    euler="10 15 20" fluidshape="ellipsoid"
                    fluidcoef="1.0 1.5 2.0 0.5 0.8"/>
            </body>
          </body>
        </worldbody>
        <keyframe>
          <key qpos="0.5 1.0" qvel="1.0 2.0"/>
        </keyframe>
      </mujoco>
      """,
    "multi_geom": """
      <mujoco>
        <option integrator="implicitfast" density="1.225" viscosity="1.8e-5"/>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="box" size=".025 .01 0.0001" pos=".025 0 0"
                  euler="20 0 0" mass="1e-4" fluidshape="ellipsoid"/>
            <geom type="box" size=".025 .01 0.0001" pos="-.025 0 0"
                  euler="-19 0 0" mass="1e-4" fluidshape="ellipsoid"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="0.1 -0.2 0.3 0.5 0.8 -0.4"/>
        </keyframe>
      </mujoco>
      """,
    "wind": """
      <mujoco>
        <option integrator="implicitfast" density="1000" viscosity="0.002"
                wind="1 0.5 0"/>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom type="ellipsoid" size="0.1 0.15 0.2" pos="-0.05 0.05 -0.05"
                  euler="-10 20 -30" fluidshape="ellipsoid"
                  fluidcoef="1.0 1.5 2.0 0.5 0.8"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="1 2 3 0.5 0.8 1.2"/>
        </keyframe>
      </mujoco>
      """,
    "density_only": """
      <mujoco>
        <option integrator="implicitfast" density="1000" viscosity="0"/>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom type="ellipsoid" size="0.1 0.15 0.2" pos="0.05 -0.05 0.05"
                  euler="10 -20 30" fluidshape="ellipsoid"
                  fluidcoef="1.0 1.5 2.0 0.5 0.8"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="1 2 3 0.5 0.8 1.2"/>
        </keyframe>
      </mujoco>
      """,
    "viscosity_only": """
      <mujoco>
        <option integrator="implicitfast" density="0" viscosity="0.01"/>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom type="ellipsoid" size="0.1 0.15 0.2" pos="-0.05 -0.05 0.05"
                  euler="-10 -20 30" fluidshape="ellipsoid"
                  fluidcoef="1.0 1.5 2.0 0.5 0.8"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="1 2 3 0.5 0.8 1.2"/>
        </keyframe>
      </mujoco>
      """,
    "capsule": """
      <mujoco>
        <option integrator="implicitfast" density="1000" viscosity="0.002"/>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom type="capsule" size="0.05 0.15" pos="0.03 0.04 0.05"
                  euler="15 25 35" fluidshape="ellipsoid"
                  fluidcoef="1.0 1.5 2.0 0.5 0.8"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="1 2 3 0.5 0.8 1.2"/>
        </keyframe>
      </mujoco>
      """,
    "inertia_box": """
      <mujoco>
        <option integrator="implicitfast" density="1.225" viscosity="1.8e-5"/>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="box" size="0.1 0.15 0.2" pos="0.05 0.05 0.05" euler="10 20 30" mass="1"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="1 2 3 0.5 0.8 1.2"/>
        </keyframe>
      </mujoco>
      """,
  }

  @parameterized.product(
    scenario=list(_FLUID_SCENARIOS.keys()),
    jacobian=[mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE],
  )
  def test_smooth_vel_fluid(self, scenario, jacobian):
    """Tests qDeriv with various ellipsoid fluid force configurations."""
    mjm, mjd, m, d = test_data.fixture(
      xml=self._FLUID_SCENARIOS[scenario],
      keyframe=0,
      overrides={"opt.jacobian": jacobian},
    )
    mujoco.mj_step(mjm, mjd)

    out_smooth_vel = wp.zeros(d.M.shape, dtype=float)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    mjw_out = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(
      mjw_out,
      out_smooth_vel.numpy().reshape(-1).astype(np.float64),
      mjm.M_rownnz,
      mjm.M_rowadr,
      mjm.M_colind,
    )

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    mj_M = np.zeros((m.nv, m.nv))
    mujoco.mju_sym2dense(
      mj_M,
      mjd.M,
      mjm.M_rownnz,
      mjm.M_rowadr,
      mjm.M_colind,
    )
    mj_out = mj_M - mjm.opt.timestep * mj_qDeriv

    has_free_body = bool(m.body_is_free.numpy().any())
    if has_free_body:
      mj_out = 0.5 * (mj_out + mj_out.T)
    name = f"M - dt * qDeriv (fluid {scenario})"
    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      mask = m.M_elemid.numpy() >= 0
      _assert_eq(mjw_out[mask], mj_out[mask], name)
    else:
      _assert_eq(mjw_out, mj_out, name)

  @parameterized.parameters(1, 2)
  def test_discrete_flex_stiffness_and_solve(self, nworld):
    """Tests discrete flex stiffness assembly, velocity shift, and PCG solve."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" integrator="discrete"/>
        <worldbody>
          <flexcomp name="cloth" type="grid" count="6 6 1" spacing="0.05 0.05 0.05"
                    radius=".005" dim="2" mass="0.5" pos="0 0 1" dof="full">
            <contact selfcollide="none" contype="0" conaffinity="0"/>
            <elasticity young="1e3" poisson="0.2" damping="0.1" elastic2d="both" thickness="0.01"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    mjds = [mjd]
    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      qpos = d.qpos.numpy()
      qpos[1, 2] += 0.05
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)
      mjd1.qpos[:] = qpos[1]
      mjds.append(mjd1)

    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])

    d.qacc_smooth.fill_(wp.inf)
    d.efm_K_val.fill_(wp.inf)
    mjw.forward(m, d)

    # Verify efm_K_val matches MuJoCo C exactly
    for w in range(nworld):
      np.testing.assert_allclose(d.efm_K_val.numpy()[w], mjds[w].efm_K_val, atol=1e-5, rtol=1e-5)

    # Verify efm_c matches MuJoCo C with non-zero velocity
    np.random.seed(42)
    qvel = np.random.randn(nworld, mjm.nv).astype(np.float32)
    for w in range(nworld):
      mjds[w].qvel[:] = qvel[w]
      mujoco.mj_forward(mjm, mjds[w])
    d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)
    d.efm_c.fill_(wp.inf)
    mjw.forward(m, d)
    for w in range(nworld):
      np.testing.assert_allclose(d.efm_c.numpy()[w], mjds[w].efm_c, atol=1e-5, rtol=1e-5)
      # Verify qacc_smooth from PCG matches MuJoCo C
      np.testing.assert_allclose(d.qacc_smooth.numpy()[w], mjds[w].qacc_smooth, atol=1e-3, rtol=1e-3)

    if nworld == 2:
      self.assertFalse(np.allclose(d.efm_c.numpy()[0], d.efm_c.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_flex_3d_tetrahedral(self, nworld):
    """Verifies discrete flex stiffness and PCG solve for 3D volumetric flex."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" integrator="discrete"/>
        <worldbody>
          <flexcomp name="box" type="grid" count="3 3 3" spacing="0.05 0.05 0.05"
                    radius=".005" dim="3" mass="0.5" pos="0 0 1" dof="full">
            <contact selfcollide="none" contype="0" conaffinity="0"/>
            <elasticity young="1e3" poisson="0.2" damping="0.1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    mjds = [mjd]
    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      qpos = d.qpos.numpy()
      qpos[1, 2] += 0.05
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)
      mjd1.qpos[:] = qpos[1]
      mjds.append(mjd1)

    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])

    d.qacc_smooth.fill_(wp.inf)
    d.efm_K_val.fill_(wp.inf)
    mjw.forward(m, d)

    self.assertGreater(m.nefmK, 0)
    self.assertEqual(int(m.flex_dim.numpy()[0]), 3)

    # Verify CSR stiffness matrix matches MuJoCo C
    for w in range(nworld):
      np.testing.assert_allclose(d.efm_K_val.numpy()[w], mjds[w].efm_K_val, atol=1e-5, rtol=1e-5)
      # Verify velocity shift c = -h*K*v matches MuJoCo C
      np.testing.assert_allclose(d.efm_c.numpy()[w], mjds[w].efm_c, atol=1e-5, rtol=1e-5)
      # Verify qacc_smooth from PCG matches MuJoCo C
      np.testing.assert_allclose(d.qacc_smooth.numpy()[w], mjds[w].qacc_smooth, atol=1e-3, rtol=1e-3)

    if nworld == 2:
      self.assertFalse(np.allclose(d.efm_K_val.numpy()[0], d.efm_K_val.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_eff_contact_stiffness_and_scale(self, nworld):
    """Verifies passive flex contact stiffness and scale calculation."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" integrator="discrete" timestep="0.002" gravity="0 0 0"/>
        <worldbody>
          <geom type="sphere" size="0.2" pos="0 0 0"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.19" dim="2" mass="0.09">
            <contact passive="true" contype="1" conaffinity="1"/>
            <elasticity young="1e3" poisson="0.3" thickness="1e-3" damping="0" elastic2d="both"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1, 2] += 0.005
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    d.qacc.fill_(wp.inf)
    d.qfrc_spring.fill_(wp.inf)
    mjw.forward(m, d)
    _, _, efm_con_scale, _, _ = derivative.build_efm_contact(m, d)
    scales = efm_con_scale.numpy()
    active_scales = scales[scales > 0]
    self.assertGreater(len(active_scales), 0)
    np.testing.assert_allclose(active_scales, 2.0, atol=1e-5)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qfrc_spring.numpy()[0], d.qfrc_spring.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_eff_contact_force_and_jacobian(self, nworld):
    """Verifies passive flex contact normal Jacobian and repulsive forces."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" integrator="discrete" timestep="0.002" gravity="0 0 0"/>
        <worldbody>
          <geom type="sphere" size="0.2" pos="0 0 0"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.19" dim="2" mass="0.09">
            <contact passive="true" contype="1" conaffinity="1"/>
            <elasticity young="1e3" poisson="0.3" thickness="1e-3" damping="0" elastic2d="both"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1, 2] += 0.005
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    d.qacc.fill_(wp.inf)
    d.qfrc_spring.fill_(wp.inf)
    mjw.forward(m, d)
    for w in range(nworld):
      # Spring force on generalized coordinates should be non-zero and repulsive
      qfrc_spring = d.qfrc_spring.numpy()[w]
      self.assertGreater(np.max(np.abs(qfrc_spring)), 0)

    # Passive contact forces should be positive (repulsive)
    efm_con_dof, efm_con_val, _, efm_con_force, efm_con_nnz = derivative.build_efm_contact(m, d)
    forces = efm_con_force.numpy()
    active_forces = forces[forces > 0]
    self.assertGreater(len(active_forces), 0)

    # For each contact, check that Jacobian row DOFs are valid and NNZ > 0
    con_nnz = efm_con_nnz.numpy()
    con_dof = efm_con_dof.numpy()
    con_val = efm_con_val.numpy()
    for i in range(len(forces)):
      if forces[i] > 0:
        nnz_i = con_nnz[i]
        self.assertIn(nnz_i, (3, 9))
        dofs = con_dof[i, :nnz_i]
        self.assertTrue(np.all(dofs >= 0) and np.all(dofs < m.nv))
        vals = con_val[i, :nnz_i]
        self.assertGreater(np.sum(vals**2), 0)

    if nworld == 2:
      self.assertFalse(np.allclose(d.qfrc_spring.numpy()[0], d.qfrc_spring.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_eff_contact_shift_and_mul_add(self, nworld):
    """Verifies contact velocity shift and PCG curvature operator positive semi-definiteness."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" integrator="discrete" timestep="0.002" gravity="0 0 0"/>
        <worldbody>
          <geom type="sphere" size="0.2" pos="0 0 0"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.19" dim="2" mass="0.09">
            <contact passive="true" contype="1" conaffinity="1"/>
            <elasticity young="1e3" poisson="0.3" thickness="1e-3" damping="0" elastic2d="both"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1, 2] += 0.005
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    d.qacc.fill_(wp.inf)
    mjw.forward(m, d)

    # Velocity shift: non-zero qvel induces shift in efm_c
    qvel = np.ones((nworld, m.nv), dtype=np.float32)
    if nworld == 2:
      qvel[1] = -1.0
    d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)
    d.efm_c.fill_(wp.inf)
    derivative.eff_shift(m, d)
    for w in range(nworld):
      efm_c = d.efm_c.numpy()[w]
      self.assertGreater(np.max(np.abs(efm_c)), 0)

    # Curvature operator: eff_mul_m adds h^2 * k * J_n^T (J_n p)
    np.random.seed(42)
    p_vec = np.random.randn(nworld, m.nv).astype(np.float32)
    wp_p = wp.array(p_vec, dtype=float, device=d.qvel.device)
    res_with_contact = wp.zeros_like(wp_p)
    derivative.eff_mul_m(m, d, res_with_contact, wp_p)

    # Disable passive contact to get metric without contact
    m.has_flex_passive = False
    res_without_contact = wp.zeros_like(wp_p)
    derivative.eff_mul_m(m, d, res_without_contact, wp_p)
    m.has_flex_passive = True

    delta_res = res_with_contact.numpy() - res_without_contact.numpy()
    for w in range(nworld):
      curvature = float(np.dot(p_vec[w], delta_res[w]))
      self.assertGreaterEqual(curvature, -1e-6, "Contact metric must be positive semi-definite")

    if nworld == 2:
      self.assertFalse(np.allclose(delta_res[0], delta_res[1]))

  @parameterized.parameters(
    mujoco.mjtJacobian.mjJAC_DENSE,
    mujoco.mjtJacobian.mjJAC_SPARSE,
  )
  def test_flex_preconditioner_block_allocation(self, jacobian):
    """Verifies flex 3x3 preconditioner block folding for tendon, actuator, efc, and contact."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" integrator="discrete" timestep="0.005"/>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <site name="anchor" pos="0 0 0.2"/>
          <flexcomp name="string" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1" pos="0 0 0.001" radius="0.005" mass="0.05">
            <contact passive="true" contype="1" conaffinity="1"/>
            <elasticity young="1e4" poisson="0.2" thickness="1e-3" elastic2d="stretch"/>
          </flexcomp>
          <body pos="0.1 0 0.002">
            <joint name="j" type="slide" axis="0 0 1" limited="true" range="-0.01 0.01"/>
            <geom type="sphere" size="0.01" mass="0.1"/>
            <site name="s_body" pos="0 0 0"/>
          </body>
        </worldbody>
        <tendon>
          <spatial name="ten" stiffness="50" damping="2">
            <site site="anchor"/>
            <site site="s_body"/>
          </spatial>
        </tendon>
        <actuator>
          <position tendon="ten" kp="20" kv="1"/>
        </actuator>
      </mujoco>
      """,
      overrides={"opt.jacobian": jacobian},
    )

    self.assertEqual(m.nefmdof, 9)
    d.ctrl.fill_(0.1)
    mjw.forward(m, d)
    epL = derivative.eff_prec_fold(m, d)
    res = wp.zeros_like(d.qacc)
    derivative.eff_prec(m, d, res, d.qfrc_smooth, epL=epL)

    self.assertTrue(np.all(np.isfinite(epL.numpy())))
    self.assertTrue(np.all(np.isfinite(res.numpy())))
    self.assertGreater(np.linalg.norm(res.numpy()), 0.0)

  @parameterized.parameters(1, 2)
  def test_eff_joint_tendon_actuator(self, nworld):
    """Tests effective metric build, shift, actuation, mul_m, and solve."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005"/>
        <worldbody>
          <site name="s0" pos="0 0 0"/>
          <body pos="0 0 0.2">
            <joint name="j0" type="slide" axis="0 0 1" stiffness="40" damping="3"/>
            <geom type="sphere" size="0.05" mass="0.5"/>
            <site name="s1" pos="0 0 0"/>
            <body pos="0 0 0.2">
              <joint name="j1" type="slide" axis="0 0 1" stiffness="25" damping="2"/>
              <geom type="sphere" size="0.05" mass="0.4"/>
              <site name="s2" pos="0 0 0"/>
            </body>
          </body>
        </worldbody>
        <tendon>
          <spatial name="t0" stiffness="80" damping="4">
            <site site="s0"/>
            <site site="s2"/>
          </spatial>
        </tendon>
        <actuator>
          <position joint="j0" kp="30" kv="2" delay="0.01" nsample="4"/>
          <dcmotor joint="j1" motorconst="1.5" resistance="0.8"/>
          <muscle tendon="t0" lengthrange="0.1 0.6"/>
        </actuator>
      </mujoco>
      """,
      qpos_noise=0.02,
      qvel_noise=0.2,
      ctrl_noise=0.2,
      nworld=nworld,
    )
    if nworld == 2:
      qvel_np = d.qvel.numpy()
      qvel_np[1] = qvel_np[0] * 1.5 + 0.1
      d.qvel = wp.array(qvel_np, dtype=float)

    d.efm_diag.fill_(wp.inf)
    d.efm_ts.fill_(wp.inf)
    d.efm_as.fill_(wp.inf)
    d.efm_c.fill_(wp.inf)
    d.efm_ca.fill_(wp.inf)
    d.qacc_smooth.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    _assert_eq(d.efm_diag.numpy()[0], mjd.efm_diag, "efm_diag")
    _assert_eq(d.efm_ts.numpy()[0], mjd.efm_ts, "efm_ts")
    _assert_eq(d.efm_as.numpy()[0], mjd.efm_as, "efm_as")
    _assert_eq(d.efm_c.numpy()[0], mjd.efm_c, "efm_c")
    _assert_eq(d.efm_ca.numpy()[0], mjd.efm_ca, "efm_ca")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc_smooth.numpy()[0], d.qacc_smooth.numpy()[1]))

    res_wp = wp.zeros_like(d.qacc_smooth)
    derivative.eff_mul_m(m, d, res_wp, d.qacc_smooth)
    expected_rhs = mjd.qfrc_smooth + mjd.efm_c + mjd.efm_ca
    _assert_eq(res_wp.numpy()[0], expected_rhs, "eff_mul_m")

  @parameterized.product(
    scenario=("basic", "inertia_box"),
    nworld=(1, 2),
  )
  def test_eff_fluid(self, scenario, nworld):
    """Tests discrete integrator fluid drag blocks (efm_fluid, qH, eff_mul_m)."""
    mjm, mjd, m, d = test_data.fixture(
      xml=self._FLUID_SCENARIOS[scenario],
      keyframe=0,
      overrides={"opt.integrator": mujoco.mjtIntegrator.mjINT_DISCRETE},
      nworld=nworld,
    )
    if nworld == 2:
      qvel_np = d.qvel.numpy()
      qvel_np[1] = qvel_np[0] * 1.5 + 0.2
      d.qvel = wp.array(qvel_np, dtype=float)

    d.efm_fluid.fill_(wp.inf)
    d.qH.fill_(wp.inf)
    d.qacc_smooth.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    _assert_eq(d.efm_fluid.numpy()[0], mjd.efm_fluid, f"efm_fluid ({scenario})")
    _assert_eq(d.qH.numpy()[0], mjd.M + mjd.efm_fluid, f"qH ({scenario})")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, f"qacc_smooth ({scenario})")
    if nworld == 2:
      self.assertFalse(np.allclose(d.efm_fluid.numpy()[0], d.efm_fluid.numpy()[1]))

    res_wp = wp.zeros_like(d.qacc_smooth)
    derivative.eff_mul_m(m, d, res_wp, d.qacc_smooth)
    expected_rhs = mjd.qfrc_smooth + mjd.efm_c
    _assert_eq(res_wp.numpy()[0], expected_rhs, f"eff_mul_m ({scenario})")

  @parameterized.parameters(1, 2)
  def test_discrete_muscle_and_actuator_damping(self, nworld):
    """Tests discrete integrator with active muscle actuators and actuator damping/dampingpoly."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005"/>
        <worldbody>
          <site name="s0" pos="0 0 0"/>
          <body pos="0 0 0.2">
            <joint name="j0" type="slide" axis="0 0 1" stiffness="40" damping="3"/>
            <geom type="sphere" size="0.05" mass="0.5"/>
            <body pos="0 0 0.2">
              <joint name="j1" type="slide" axis="0 0 1" stiffness="25" damping="2"/>
              <geom type="sphere" size="0.05" mass="0.4"/>
              <site name="s1" pos="0 0 0"/>
            </body>
          </body>
        </worldbody>
        <tendon>
          <spatial name="t0" stiffness="50" damping="2">
            <site site="s0"/>
            <site site="s1"/>
          </spatial>
        </tendon>
        <actuator>
          <general joint="j0" gear="2.5" damping="1.5" gaintype="fixed" gainprm="10"/>
          <general tendon="t0" gear="1.8" damping="1.2" gaintype="fixed" gainprm="5"/>
          <muscle tendon="t0" lengthrange="0.2 0.6"/>
        </actuator>
        <keyframe>
          <key qpos="0.03 -0.02" qvel="0.4 -0.3" ctrl="0.5 -0.2 0.8" act="0.6"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      overrides={"actuator_dampingpoly": np.array([[0.4, 0.2], [0.3, 0.1], [0.0, 0.0]])},
      nworld=nworld,
    )
    if nworld == 2:
      qvel_np = d.qvel.numpy()
      qvel_np[1] = [0.7, -0.5]
      d.qvel = wp.array(qvel_np, dtype=float)

    d.efm_diag.fill_(wp.inf)
    d.efm_ts.fill_(wp.inf)
    d.efm_as.fill_(wp.inf)
    d.efm_c.fill_(wp.inf)
    d.efm_ca.fill_(wp.inf)
    d.qacc_smooth.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    _assert_eq(d.efm_diag.numpy()[0], mjd.efm_diag, "efm_diag")
    _assert_eq(d.efm_ts.numpy()[0], mjd.efm_ts, "efm_ts")
    self.assertEqual(mjd.nefmA, 1)
    self.assertEqual(mjd.efm_aid[0], 2)
    np.testing.assert_allclose(d.efm_as.numpy()[0, 2], mjd.efm_as[0], rtol=1e-3, atol=1e-4)
    _assert_eq(d.efm_c.numpy()[0], mjd.efm_c, "efm_c")
    _assert_eq(d.efm_ca.numpy()[0], mjd.efm_ca, "efm_ca")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc_smooth.numpy()[0], d.qacc_smooth.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_flex_interp_and_pinned(self, nworld):
    """Tests discrete integrator with pinned non-centered flex."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.002"/>
        <worldbody>
          <flexcomp name="pinned_flex" type="grid" count="3 2 1" spacing="0.05 0.05 0.05"
                    pos="0.3 0 0.3" dim="2" radius="0.01" mass="0.2" dof="full">
            <pin id="0 1"/>
            <contact selfcollide="none"/>
            <elasticity young="2000" damping="0.01" thickness="0.01" elastic2d="both"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      qpos_noise=0.005,
      qvel_noise=0.1,
      nworld=nworld,
    )
    if nworld == 2:
      qvel_np = d.qvel.numpy()
      qvel_np[1] = qvel_np[0] * 1.5 + 1.0
      d.qvel = wp.array(qvel_np, dtype=float)

    d.efm_L.fill_(wp.inf)
    d.efm_c.fill_(wp.inf)
    d.qacc_smooth.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    _assert_eq(d.efm_L.numpy()[0], mjd.efm_L, "efm_L")
    _assert_eq(d.efm_c.numpy()[0], mjd.efm_c, "efm_c")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc_smooth.numpy()[0], d.qacc_smooth.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_flex_passive_contact_rotated(self, nworld):
    """Tests passive flex contact under discrete integrator with rotated body and dynamic geom."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.002"/>
        <worldbody>
          <geom name="floor" type="plane" size="2 2 0.01"/>
          <body name="dyn_obstacle" pos="0.2 0 0.02">
            <joint type="slide" axis="1 0 0"/>
            <geom type="box" size="0.1 0.1 0.02" mass="1.0"/>
          </body>
          <flexcomp name="f0" type="grid" count="2 2 1" spacing="0.06 0.06 0.06"
                    pos="0 0 0.005" euler="25 35 15" dim="2" radius="0.015" mass="0.4" dof="full">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      qvel_noise=0.2,
      nworld=nworld,
    )
    if nworld == 2:
      qvel_np = d.qvel.numpy()
      qvel_np[1] = qvel_np[0] * 1.5 + 0.05
      d.qvel = wp.array(qvel_np, dtype=float)

    d.efm_c.fill_(wp.inf)
    d.qacc_smooth.fill_(wp.inf)
    d.qacc.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    _assert_eq(d.efm_c.numpy()[0], mjd.efm_c, "efm_c")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")
    _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_flex_trilinear_interp_rotated(self, nworld):
    """Tests trilinear interpolated flex (flex_interp=1) with rotated body and cell frames."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" integrator="discrete" timestep="0.002"/>
        <worldbody>
          <flexcomp name="interp_box" type="grid" count="3 3 3" spacing="0.05 0.05 0.05"
                    pos="0.1 -0.1 0.3" euler="25 -30 40" dim="3" radius="0.005" mass="0.5" dof="trilinear">
            <contact selfcollide="none" contype="0" conaffinity="0"/>
            <elasticity young="2e3" poisson="0.25" damping="0.05"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      qpos_noise=0.01,
      qvel_noise=0.1,
      nworld=nworld,
    )
    if nworld == 2:
      qvel_np = d.qvel.numpy()
      qvel_np[1] = qvel_np[0] * 1.5 + 0.05
      d.qvel = wp.array(qvel_np, dtype=float)

    d.efm_K_val.fill_(wp.inf)
    d.efm_c.fill_(wp.inf)
    d.qacc_smooth.fill_(wp.inf)
    d.qacc.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    _assert_eq(d.efm_K_val.numpy()[0], mjd.efm_K_val, "efm_K_val")
    _assert_eq(d.efm_c.numpy()[0], mjd.efm_c, "efm_c")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")
    _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_flex_bend_only_efm0_and_rotated(self, nworld):
    """Tests 2D bending-only flex with efm0_L sparse Cholesky preconditioner and rotated bodies."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" integrator="discrete" timestep="0.002"/>
        <worldbody>
          <body pos="0 0 0.5">
            <joint type="slide" axis="0 0 1" stiffness="20" damping="1"/>
            <geom type="sphere" size="0.02" mass="0.1"/>
          </body>
          <flexcomp name="bend_cloth" type="grid" count="4 4 1" spacing="0.05 0.05 0.05"
                    pos="0 0 0.4" euler="30 20 -15" dim="2" radius="0.005" mass="0.4" dof="full">
            <contact selfcollide="none" contype="0" conaffinity="0"/>
            <elasticity young="1e3" poisson="0.2" damping="0.05" elastic2d="bend" thickness="0.02"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      qpos_noise=0.005,
      qvel_noise=0.1,
      nworld=nworld,
    )
    if nworld == 2:
      qvel_np = d.qvel.numpy()
      qvel_np[1] = qvel_np[0] * 1.5 + 0.05
      d.qvel = wp.array(qvel_np, dtype=float)

    self.assertTrue(m.efm0_active)
    self.assertGreater(m.nefm0dof, 0)

    d.efm_c.fill_(wp.inf)
    d.qacc_smooth.fill_(wp.inf)
    d.qacc.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    _assert_eq(d.efm_c.numpy()[0], mjd.efm_c, "efm_c")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")
    _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

    # Verify _eff_prec_efm0_solve with cross-coordinate (off-axis) couplings in efm0_L.
    n0 = m.nefm0dof
    efm0_dofid = m.efm0_dofid.numpy()
    L_dense = np.eye(n0, dtype=np.float64) * 2.5
    rownnz = []
    rowadr = []
    colind = []
    vals = []
    for i in range(n0):
      rowadr.append(len(colind))
      for c in range(max(0, i - 5), i):
        # Include cross-coordinate couplings where (i - c) % 3 != 0
        val = 0.15 * ((i + 1) / (n0 + 1)) * (-1.0 if (i + c) % 2 else 1.0)
        L_dense[i, c] = val
        colind.append(c)
        vals.append(val)
      colind.append(i)
      vals.append(L_dense[i, i])
      rownnz.append(len(colind) - rowadr[-1])

    m.efm0_L_rownnz = wp.array(rownnz, dtype=int)
    m.efm0_L_rowadr = wp.array(rowadr, dtype=int)
    m.efm0_L_colind = wp.array(colind, dtype=int)
    m.efm0_L = wp.array(vals, dtype=float)

    rng = np.random.default_rng(123)
    vec_np = rng.uniform(-1.0, 1.0, size=(nworld, mjm.nv)).astype(np.float32)
    vec_wp = wp.array(vec_np, dtype=float)
    res_wp = wp.full((nworld, mjm.nv), wp.inf, dtype=float)

    derivative.eff_prec(m, d, res_wp, vec_wp)
    res_np = res_wp.numpy()

    A_efm0 = L_dense.T @ L_dense
    for w in range(nworld):
      expected_efm0 = np.linalg.solve(A_efm0, vec_np[w, efm0_dofid])
      _assert_eq(res_np[w, efm0_dofid], expected_efm0, f"efm0_cross_coord_world_{w}")

  @parameterized.product(
    elastic2d=["bend", "both"],
    nworld=[1, 2],
  )
  def test_discrete_flex_welded_and_rotated_vertex(self, elastic2d, nworld):
    """Tests discrete integrator flex stiffness and shift with rotated frames and welded vertex."""
    mjm, _, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option integrator="discrete" solver="Newton" timestep="0.002"/>
        <worldbody>
          <body name="turned" euler="90 35 20">
            <body name="v0" pos="0 0 0">
              <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
              <joint type="slide" axis="1 0 0"/>
              <joint type="slide" axis="0 1 0"/>
              <joint type="slide" axis="0 0 1"/>
            </body>
            <body name="v1" pos="0.1 0 0">
              <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
              <joint type="slide" axis="1 0 0"/>
              <joint type="slide" axis="0 1 0"/>
              <joint type="slide" axis="0 0 1"/>
            </body>
            <body name="v2" pos="0 0.1 0">
              <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
              <joint type="slide" axis="1 0 0"/>
              <joint type="slide" axis="0 1 0"/>
              <joint type="slide" axis="0 0 1"/>
            </body>
            <body name="carrier" pos="0.1 0.1 0">
              <inertial pos="0 0 0" mass="0.05" diaginertia="5e-5 5e-5 5e-5"/>
              <joint type="slide" axis="1 0 0"/>
              <joint type="slide" axis="0 1 0"/>
              <joint type="slide" axis="0 0 1"/>
              <body name="v3_welded" pos="0 0 0">
                <inertial pos="0 0 0" mass="0.05" diaginertia="5e-5 5e-5 5e-5"/>
              </body>
            </body>
          </body>
        </worldbody>
        <deformable>
          <flex name="patch" dim="2" body="v0 v1 v2 v3_welded" element="0 1 2 1 3 2">
            <contact selfcollide="none" contype="0" conaffinity="0"/>
            <elasticity young="1e4" poisson="0.3" thickness="0.01"
                        elastic2d="{elastic2d}" damping="0.02"/>
          </flex>
        </deformable>
      </mujoco>
      """,
      overrides={"opt.disableflags": mjw.DisableBit.CONTACT | mjw.DisableBit.GRAVITY},
      nworld=nworld,
    )

    mjm_ref, mjd_ref, _, _ = test_data.fixture(
      xml=f"""
      <mujoco>
        <option integrator="discrete" solver="Newton" timestep="0.002"/>
        <worldbody>
          <body name="v0" pos="0 0 0">
            <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
            <joint type="slide" axis="1 0 0"/>
            <joint type="slide" axis="0 1 0"/>
            <joint type="slide" axis="0 0 1"/>
          </body>
          <body name="v1" pos="0.1 0 0">
            <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
            <joint type="slide" axis="1 0 0"/>
            <joint type="slide" axis="0 1 0"/>
            <joint type="slide" axis="0 0 1"/>
          </body>
          <body name="v2" pos="0 0.1 0">
            <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
            <joint type="slide" axis="1 0 0"/>
            <joint type="slide" axis="0 1 0"/>
            <joint type="slide" axis="0 0 1"/>
          </body>
          <body name="v3" pos="0.1 0.1 0">
            <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
            <joint type="slide" axis="1 0 0"/>
            <joint type="slide" axis="0 1 0"/>
            <joint type="slide" axis="0 0 1"/>
          </body>
        </worldbody>
        <deformable>
          <flex name="patch" dim="2" body="v0 v1 v2 v3" element="0 1 2 1 3 2">
            <contact selfcollide="none" contype="0" conaffinity="0"/>
            <elasticity young="1e4" poisson="0.3" thickness="0.01"
                        elastic2d="{elastic2d}" damping="0.02"/>
          </flex>
        </deformable>
      </mujoco>
      """,
      overrides={"opt.disableflags": mjw.DisableBit.CONTACT | mjw.DisableBit.GRAVITY},
    )

    rng = np.random.default_rng(42)
    mjds_ref = [mjd_ref]
    if nworld == 2:
      mjds_ref.append(mujoco.MjData(mjm_ref))

    qpos_worlds = []
    qvel_worlds = []
    for w in range(nworld):
      qpos_w = rng.uniform(-5e-3, 5e-3, size=mjm.nq).astype(np.float32)
      qvel_w = rng.uniform(-0.5, 0.5, size=mjm.nv).astype(np.float32)
      qpos_worlds.append(qpos_w)
      qvel_worlds.append(qvel_w)
      mjds_ref[w].qpos[:] = qpos_w
      mjds_ref[w].qvel[:] = qvel_w
      mujoco.mj_forward(mjm_ref, mjds_ref[w])

    d.qpos.assign(np.stack(qpos_worlds))
    d.qvel.assign(np.stack(qvel_worlds))
    for arr in (d.efm_c, d.qfrc_spring, d.qfrc_damper, d.qacc_smooth, d.qacc):
      arr.fill_(wp.inf)

    mjw.forward(m, d)

    for w in range(nworld):
      if d.efm_K_val.size > 0:
        _assert_eq(d.efm_K_val.numpy()[w], mjds_ref[w].efm_K_val, f"efm_K_val_w{w}")
      _assert_eq(d.efm_c.numpy()[w], mjds_ref[w].efm_c, f"efm_c_w{w}")
      _assert_eq(d.qfrc_spring.numpy()[w], mjds_ref[w].qfrc_spring, f"qfrc_spring_w{w}")
      _assert_eq(d.qfrc_damper.numpy()[w], mjds_ref[w].qfrc_damper, f"qfrc_damper_w{w}")
      _assert_eq(d.qacc_smooth.numpy()[w], mjds_ref[w].qacc_smooth, f"qacc_smooth_w{w}")
      _assert_eq(d.qacc.numpy()[w], mjds_ref[w].qacc, f"qacc_w{w}")

    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_free_gyro_sensor_and_passive_contact(self, nworld):
    """Tests free gyro execution before sensor_acc and rne_postconstraint."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005"/>
        <worldbody>
          <body name="spinner" pos="0 0 1">
            <freejoint/>
            <geom type="box" size="0.1 0.2 0.3" mass="2.0" pos="0.05 -0.03 0.02"/>
            <site name="imu" pos="0.05 0.05 0.05"/>
          </body>
        </worldbody>
        <sensor>
          <accelerometer site="imu"/>
          <frameangacc objtype="body" objname="spinner"/>
        </sensor>
      </mujoco>
      """,
      qvel_noise=2.0,
      nworld=nworld,
    )
    if nworld == 2:
      qvel_np = d.qvel.numpy()
      qvel_np[1] = qvel_np[0] * 1.5 + 0.1
      d.qvel = wp.array(qvel_np, dtype=float)

    d.qacc.fill_(wp.inf)
    d.cacc.fill_(wp.inf)
    d.sensordata.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
    _assert_eq(d.cacc.numpy()[0], mjd.cacc.reshape(-1, 6), "cacc")
    _assert_eq(d.sensordata.numpy()[0], mjd.sensordata, "sensordata")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_sleep(self, nworld):
    """Tests sleep filtering under discrete integrator."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.002" gravity="0 0 0">
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="2 2 0.01"/>
          <body name="awake_b" pos="0 0 0.05">
            <joint type="slide" axis="0 0 1" stiffness="50" damping="2"/>
            <geom type="sphere" size="0.08" mass="1.0"/>
          </body>
          <body name="asleep_b" pos="0.5 0 0.5">
            <joint type="slide" axis="0 0 1" stiffness="50" damping="2"/>
            <geom type="sphere" size="0.05" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Keep awake_b moving while asleep_b is stationary so C MuJoCo puts tree 1 to sleep naturally
    mjd.qvel[0] = -1.0
    for _ in range(mujoco.mjMINAWAKE + 1):
      mujoco.mj_step(mjm, mjd)
    self.assertEqual(int(mjd.tree_awake[1]), 0)

    # Inject a non-zero warmstart value on sleeping DOF 1 to verify it is zeroed
    mjd.qacc_warmstart[1] = 1e6
    d = mjw.put_data(mjm, mjd, nworld=nworld)
    if nworld == 2:
      qvel_np = d.qvel.numpy()
      qvel_np[1, 0] = -1.5
      d.qvel = wp.array(qvel_np, dtype=float)

    d.qacc_smooth.fill_(wp.inf)
    d.qacc.fill_(wp.inf)
    d.qfrc_constraint.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    self.assertEqual(int(d.tree_awake.numpy()[0, 1]), 0)
    self.assertEqual(d.qacc_smooth.numpy()[0, 1], 0.0)
    self.assertEqual(d.qacc.numpy()[0, 1], 0.0)
    self.assertEqual(d.qfrc_constraint.numpy()[0, 1], 0.0)
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")
    _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_disabled_actuation_gravcomp(self, nworld):
    """Tests disabled actuation does not add gravcomp into qfrc_actuator for discrete integrator."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005">
          <flag actuation="disable"/>
        </option>
        <worldbody>
          <body gravcomp="1" pos="0 0 1">
            <joint name="j" type="slide" axis="0 0 1" stiffness="10" damping="2"/>
            <geom type="sphere" size="0.1" mass="1.0"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="j" gear="1"/>
        </actuator>
      </mujoco>
      """,
      qpos_noise=0.1,
      ctrl_noise=1.0,
      nworld=nworld,
    )
    if nworld == 2:
      qpos_np = d.qpos.numpy()
      qpos_np[1, 0] += 0.2
      d.qpos = wp.array(qpos_np, dtype=float)

    d.qfrc_actuator.fill_(wp.inf)
    d.qacc_smooth.fill_(wp.inf)
    d.qacc.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    _assert_eq(d.qfrc_actuator.numpy()[0], mjd.qfrc_actuator, "qfrc_actuator")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")
    _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_passive_contact_eligibility(self, nworld):
    """Tests passive flex contact eligibility with disabled spring, 1D flex, and dynamic bodies."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.002">
          <flag spring="disable"/>
        </option>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <body pos="0 0 0.02">
            <freejoint/>
            <geom type="sphere" size="0.05" mass="1.0"/>
          </body>
          <flexcomp name="f" type="grid" count="2 2 1" spacing="0.1 0.1 0.1" pos="0 0 0.01" radius="0.02" mass="1.0" dim="2">
            <contact passive="true" selfcollide="none"/>
            <elasticity young="100" damping="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      qvel_noise=0.5,
      nworld=nworld,
    )
    # Verify put_data preserves CPU contact classification (exclude == 4 -> PASSIVE)
    expected_types = np.where(
      mjd.contact.exclude[: mjd.ncon] == 4,
      int(types.ContactType.PASSIVE),
      int(types.ContactType.CONSTRAINT),
    )
    np.testing.assert_array_equal(d.contact.type.numpy()[: mjd.ncon], expected_types)

    if nworld == 2:
      qvel_np = d.qvel.numpy()
      qvel_np[1] = qvel_np[0] * 1.5 - 0.2
      d.qvel = wp.array(qvel_np, dtype=float)

    d.qacc.fill_(wp.inf)
    d.nefc.fill_(-1)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    self.assertEqual(int(d.nefc.numpy()[0]), mjd.nefc)
    _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_direct_solve_with_motor(self, nworld):
    """Tests that eff_solve uses direct qH solve when actuators/tendons add no metric coupling."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" iterations="0" timestep="0.005"/>
        <worldbody>
          <body pos="0 0 1">
            <joint name="j" type="slide" axis="0 0 1" stiffness="10" damping="2"/>
            <geom type="sphere" size="0.1" mass="1.0"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="j" gear="1"/>
        </actuator>
      </mujoco>
      """,
      ctrl_noise=2.0,
      nworld=nworld,
    )
    if nworld == 2:
      ctrl_np = d.ctrl.numpy()
      ctrl_np[1, 0] += 5.0
      d.ctrl = wp.array(ctrl_np, dtype=float)

    d.qacc_smooth.fill_(wp.inf)
    d.qacc.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")
    _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))


if __name__ == "__main__":
  wp.init()
  absltest.main()
