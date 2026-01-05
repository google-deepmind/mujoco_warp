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
      <option integrator="implicitfast">
        <flag gravity="disable"/>
      </option>
      <worldbody>
        <body>
          <geom type="sphere" size=".1"/>
          <joint name="joint0" type="hinge" axis="0 1 0"/>
          <site name="site0" pos="0 0 1"/>
        </body>
        <body pos="1 0 0">
          <geom type="sphere" size=".1"/>
          <joint name="joint1" type="hinge" axis="0 1 0"/>
          <site name="site1" pos="0 0 1"/>
        </body>
        <body pos="2 0 0">
          <geom type="sphere" size=".1"/>
          <joint name="joint2" type="hinge" axis="0 1 0"/>
          <site name="site2" pos="0 0 1"/>
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

    mujoco.mj_step(mjm, mjd)  # step w/ implicitfast calls mjd_smooth_vel to compute qDeriv

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

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    mj_qM = np.zeros((m.nv, m.nv))
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
    """Tests qDeriv for tendon actuator on serial chain.

    Verifies that sibling DOF cross-terms from tendon coupling are dropped
    (matching MuJoCo CPU's implicitfast approximation) and that no NaN or
    stale values leak into the result.
    """
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

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    mj_qM = np.zeros((m.nv, m.nv))
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
    mujoco.mj_fullM(mjm, mj_qM, mjd.qM)
    mj_out = mj_qM - mjm.opt.timestep * mj_qDeriv

    self.assertFalse(np.any(np.isnan(mjw_out)))
    _assert_eq(mjw_out, mj_out, "qM - dt * qDeriv (sparse tendon coupled)")

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

  @parameterized.parameters(False, True)
  def test_transition_fd_linear_system(self, centered):
    """Tests A and B matrices match MuJoCo mjd_transitionFD."""
    # simple linear system with 3 slide joints
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j0" type="slide" axis="1 0 0" damping="1" stiffness="2"/>
          <geom size=".1"/>
          <body pos="1 0 0">
            <joint name="j1" type="slide" axis="0 1 0" damping="2" stiffness="3"/>
            <geom size=".1"/>
            <body pos="0 1 0">
              <joint name="j2" type="slide" axis="0 0 1" damping="3" stiffness="4"/>
              <geom size=".1"/>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor joint="j0" ctrlrange="-1 1" ctrllimited="true"/>
        <motor joint="j1" ctrlrange="-1 1" ctrllimited="true"/>
      </actuator>
      <keyframe>
        <key qpos="0.1 0.2 0.3" qvel="0.4 0.5 0.6" ctrl="0.1 -0.1"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=0,
    )

    # larger eps needed for float32 precision
    eps = 1e-3
    ndx = 2 * mjm.nv + mjm.na

    # mujoco reference
    A_mj = np.zeros((ndx, ndx))
    B_mj = np.zeros((ndx, mjm.nu))
    mujoco.mjd_transitionFD(mjm, mjd, eps, centered, A_mj, B_mj, None, None)

    # mujoco warp
    A_mjw = wp.zeros((1, ndx, ndx), dtype=float)
    B_mjw = wp.zeros((1, ndx, mjm.nu), dtype=float)
    mjw.transition_fd(m, d, eps, centered, A_mjw, B_mjw, None, None)

    _assert_eq(A_mjw.numpy()[0], A_mj, "A")
    _assert_eq(B_mjw.numpy()[0], B_mj, "B")

  @parameterized.parameters(False, True)
  def test_transition_fd_sensor_derivatives(self, centered):
    """Tests C and D matrices against MuJoCo mjd_transitionFD."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint" type="slide"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <actuator>
        <general name="actuator" joint="joint" gainprm="3"/>
      </actuator>
      <sensor>
        <jointpos joint="joint"/>
        <jointvel joint="joint"/>
        <actuatorfrc actuator="actuator"/>
      </sensor>
    </mujoco>
    """,
    )

    # larger eps needed for float32 precision
    eps = 1e-3
    nv = mjm.nv
    nu = mjm.nu
    ns = mjm.nsensordata
    ndx = 2 * nv + mjm.na

    # mujoco reference
    C_mj = np.zeros((ns, ndx))
    D_mj = np.zeros((ns, nu))
    mujoco.mjd_transitionFD(mjm, mjd, eps, centered, None, None, C_mj, D_mj)

    # mujoco warp
    C_mjw = wp.zeros((1, ns, ndx), dtype=float)
    D_mjw = wp.zeros((1, ns, nu), dtype=float)
    mjw.transition_fd(m, d, eps, centered, None, None, C_mjw, D_mjw)

    _assert_eq(C_mjw.numpy()[0], C_mj, "C")
    _assert_eq(D_mjw.numpy()[0], D_mj, "D")

  @parameterized.parameters(False, True)
  def test_transition_fd_clamped_ctrl(self, centered):
    """Tests that B matrix is zero when ctrl is at or beyond limits."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint" type="slide"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="joint" ctrlrange="-1 1" ctrllimited="true"/>
      </actuator>
    </mujoco>
    """,
    )

    eps = 1e-3
    nv = mjm.nv
    nu = mjm.nu
    ndx = 2 * nv + mjm.na

    # set ctrl beyond limits
    mjd.ctrl[0] = 2.0
    d.ctrl.fill_(2.0)

    # mujoco reference - B should be zero
    B_mj = np.zeros((ndx, nu))
    mujoco.mjd_transitionFD(mjm, mjd, eps, centered, None, B_mj, None, None)

    # mujoco warp
    B_mjw = wp.zeros((1, ndx, nu), dtype=float)
    mjw.transition_fd(m, d, eps, centered, None, B_mjw, None, None)

    # expect B to be zero since ctrl is beyond limits
    _assert_eq(B_mjw.numpy()[0], B_mj, "B clamped")
    np.testing.assert_allclose(B_mj, 0.0, atol=1e-10)

  def test_transition_fd_no_state_mutation(self):
    """Tests that transition_fd does not mutate state."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j0" type="slide"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="j0"/>
      </actuator>
      <keyframe>
        <key qpos="0.5" qvel="0.3" ctrl="0.1"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=0,
    )

    # save state before
    qpos_before = d.qpos.numpy().copy()
    qvel_before = d.qvel.numpy().copy()
    ctrl_before = d.ctrl.numpy().copy()

    # call transition_fd
    eps = 1e-3
    ndx = 2 * m.nv + m.na
    A = wp.zeros((1, ndx, ndx), dtype=float)
    B = wp.zeros((1, ndx, m.nu), dtype=float)
    mjw.transition_fd(m, d, eps, False, A, B, None, None)

    # check state unchanged
    _assert_eq(d.qpos.numpy(), qpos_before, "qpos")
    _assert_eq(d.qvel.numpy(), qvel_before, "qvel")
    _assert_eq(d.ctrl.numpy(), ctrl_before, "ctrl")

  @parameterized.parameters(False, True)
  def test_transition_fd_free_joint(self, centered):
    """Tests A and B matrices with a free joint (quaternion perturbation)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option>
        <flag gravity="disable"/>
      </option>
      <worldbody>
        <body>
          <freejoint name="free"/>
          <geom size=".1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="free" gear="1 0 0 0 0 0" ctrlrange="-1 1"
               ctrllimited="true"/>
      </actuator>
      <keyframe>
        <key qpos="0 0 0.5 1 0 0 0" qvel="0.1 0.2 0.3 0.01 0.02 0.03"
             ctrl="0.5"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=0,
    )

    eps = 1e-3
    ndx = 2 * mjm.nv + mjm.na

    # mujoco reference
    A_mj = np.zeros((ndx, ndx))
    B_mj = np.zeros((ndx, mjm.nu))
    mujoco.mjd_transitionFD(mjm, mjd, eps, centered, A_mj, B_mj, None, None)

    # mujoco warp
    A_mjw = wp.zeros((1, ndx, ndx), dtype=float)
    B_mjw = wp.zeros((1, ndx, mjm.nu), dtype=float)
    mjw.transition_fd(m, d, eps, centered, A_mjw, B_mjw, None, None)

    _assert_eq(A_mjw.numpy()[0], A_mj, "A free joint")
    _assert_eq(B_mjw.numpy()[0], B_mj, "B free joint")

  @parameterized.parameters(False, True)
  def test_transition_fd_activations(self, centered):
    """Tests A and B matrices with actuator activations (na > 0)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j0" type="slide" damping="1"/>
          <geom size=".1"/>
        </body>
        <body pos="1 0 0">
          <joint name="j1" type="slide" damping="1"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <actuator>
        <general joint="j0" dyntype="integrator" gainprm="100"
                 biastype="affine" biasprm="0 -100 0"/>
        <general joint="j1" dyntype="filter" dynprm="0.5"
                 gainprm="100" biastype="affine" biasprm="0 -100 0"/>
      </actuator>
      <keyframe>
        <key qpos="0.1 0.2" qvel="0.3 0.4" act="0.5 0.6" ctrl="0.1 0.2"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=0,
    )

    self.assertGreater(mjm.na, 0, "Model should have activations")
    eps = 1e-3
    ndx = 2 * mjm.nv + mjm.na

    # mujoco reference
    A_mj = np.zeros((ndx, ndx))
    B_mj = np.zeros((ndx, mjm.nu))
    mujoco.mjd_transitionFD(mjm, mjd, eps, centered, A_mj, B_mj, None, None)

    # mujoco warp
    A_mjw = wp.zeros((1, ndx, ndx), dtype=float)
    B_mjw = wp.zeros((1, ndx, mjm.nu), dtype=float)
    mjw.transition_fd(m, d, eps, centered, A_mjw, B_mjw, None, None)

    _assert_eq(A_mjw.numpy()[0], A_mj, "A activations")
    _assert_eq(B_mjw.numpy()[0], B_mj, "B activations")

  @parameterized.parameters(False, True)
  def test_transition_fd_ctrl_preserved(self, centered):
    """Tests that ctrl values are preserved despite internal clamping."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint" type="slide"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="joint" ctrlrange="-1 1" ctrllimited="true"/>
      </actuator>
    </mujoco>
    """,
    )

    eps = 1e-3
    nv = mjm.nv
    nu = mjm.nu
    ndx = 2 * nv + mjm.na

    # set ctrl beyond limits
    mjd.ctrl[0] = 2.0
    d.ctrl.fill_(2.0)

    # mujoco reference - B should be zero
    B_mj = np.zeros((ndx, nu))
    mujoco.mjd_transitionFD(mjm, mjd, eps, centered, None, B_mj, None, None)

    # mujoco warp
    B_mjw = wp.zeros((1, ndx, nu), dtype=float)
    mjw.transition_fd(m, d, eps, centered, None, B_mjw, None, None)

    # expect B to be zero since ctrl is beyond limits
    _assert_eq(B_mjw.numpy()[0], B_mj, "B beyond limit")
    np.testing.assert_allclose(B_mj, 0.0, atol=1e-10)

    # verify ctrl preserved despite internal clamping during FD
    np.testing.assert_allclose(
      mjd.ctrl[0],
      2.0,
      atol=1e-10,
      err_msg="MuJoCo ctrl should not be modified",
    )
    np.testing.assert_allclose(
      d.ctrl.numpy()[0, 0],
      2.0,
      atol=1e-10,
      err_msg="Warp ctrl should not be modified",
    )

  def test_transition_fd_full_no_mutation(self):
    """Tests state preservation with free joints, activations, time, sensors."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <freejoint name="free"/>
          <geom size=".1" mass="1"/>
        </body>
        <body pos="1 0 0">
          <joint name="slide" type="slide" damping="1"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="free" gear="1 0 0 0 0 0"/>
        <general joint="slide" dyntype="integrator" gainprm="100"
                 biastype="affine" biasprm="0 -100 0"/>
      </actuator>
      <sensor>
        <jointpos joint="slide"/>
        <jointvel joint="slide"/>
      </sensor>
      <keyframe>
        <key qpos="0.1 0.2 0.3 1 0 0 0 0.5"
             qvel="0.01 0.02 0.03 0.04 0.05 0.06 0.1"
             act="0.5" ctrl="0.1 0.2"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=0,
    )

    self.assertGreater(mjm.na, 0, "Model should have activations")
    self.assertGreater(mjm.nsensordata, 0, "Model should have sensors")

    # save state before
    qpos_before = d.qpos.numpy().copy()
    qvel_before = d.qvel.numpy().copy()
    act_before = d.act.numpy().copy()
    ctrl_before = d.ctrl.numpy().copy()
    time_before = d.time.numpy().copy()

    # call transition_fd requesting all matrices
    eps = 1e-3
    nv = m.nv
    ns = m.nsensordata
    ndx = 2 * nv + m.na
    A = wp.zeros((1, ndx, ndx), dtype=float)
    B = wp.zeros((1, ndx, m.nu), dtype=float)
    C = wp.zeros((1, ns, ndx), dtype=float)
    D = wp.zeros((1, ns, m.nu), dtype=float)
    mjw.transition_fd(m, d, eps, False, A, B, C, D)

    # check all state fields unchanged
    _assert_eq(d.qpos.numpy(), qpos_before, "qpos")
    _assert_eq(d.qvel.numpy(), qvel_before, "qvel")
    _assert_eq(d.act.numpy(), act_before, "act")
    _assert_eq(d.ctrl.numpy(), ctrl_before, "ctrl")
    _assert_eq(d.time.numpy(), time_before, "time")


if __name__ == "__main__":
  wp.init()
  absltest.main()
