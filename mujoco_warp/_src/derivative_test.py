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

    from mujoco_warp._src import forward

    # Compute kinematics without factorizing qM to allow direct comparison
    forward.fwd_position(m, d, factorize=False)
    forward.fwd_velocity(m, d)

    # 1. Test with RNE Disabled (Matches MuJoCo's ImplicitFast)
    mjw.deriv_smooth_vel(m, d, out_smooth_vel, flg_rne=False)

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
    mujoco.mj_fullM(mjm, mj_qM, mjd.qM)
    mj_out = mj_qM - mjm.opt.timestep * mj_qDeriv

    _assert_eq(mjw_out, mj_out, "qM - dt * qDeriv")

    # 2. RNE Smoke Test: Verify Coriolis/Centrifugal terms
    # Reuse buffers to check if flg_rne=True changes the result
    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      out_rne = wp.zeros((1, 1, m.nM), dtype=float)
    else:
      out_rne = wp.zeros(d.qM.shape, dtype=float)

    mjw.deriv_smooth_vel(m, d, out_rne, flg_rne=True)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      mjw_rne_out = np.zeros((m.nv, m.nv))
      for elem, (i, j) in enumerate(zip(m.qM_fullm_i.numpy(), m.qM_fullm_j.numpy())):
        mjw_rne_out[i, j] = out_rne.numpy()[0, 0, elem]
    else:
      mjw_rne_out = out_rne.numpy()[0, : m.nv, : m.nv]

    diff_norm = np.linalg.norm(mjw_rne_out - mjw_out)
    if diff_norm < 1e-6:
      qvel_norm = np.linalg.norm(mjd.qvel)
      if qvel_norm > 1e-3:
        raise AssertionError(f"RNE enabled but no derivative change detected! (Diff={diff_norm}, qvel={qvel_norm})")

  def _create_random_model_xml(self):
    # Create a simple chain with Free Joint
    return """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <body pos="0 0 1">
          <joint type="free"/>
          <geom type="capsule" size="0.1 0.5" density="100"/>
        </body>
      </worldbody>
    </mujoco>
    """

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE)
  def test_rne_stress(self, jacobian):
    """Tests deriv_smooth_vel logic (qM generation) with RNE disabled."""
    xml = self._create_random_model_xml()

    mjm, mjd, m, d = test_data.fixture(
      xml=xml,
      keyframe=None,
      overrides={"opt.jacobian": jacobian},
    )

    rng = np.random.RandomState(0)
    mjd.qpos = rng.randn(mjm.nq)
    mjd.qvel = rng.randn(mjm.nv)

    mujoco.mj_normalizeQuat(mjm, mjd.qpos)

    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    mujoco.mj_step(mjm, mjd)

    d.qpos = wp.from_numpy(mjd.qpos.reshape(1, -1), dtype=float, device=d.qpos.device)
    d.qvel = wp.from_numpy(mjd.qvel.reshape(1, -1), dtype=float, device=d.qvel.device)

    from mujoco_warp._src import forward

    forward.fwd_position(m, d)
    forward.fwd_velocity(m, d)

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

    # Disable RNE to match MuJoCo's implicit integrator (which ignores Coriolis terms).
    mjw.deriv_smooth_vel(m, d, out_smooth_vel, flg_rne=False)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      mjw_out = np.zeros((m.nv, m.nv))
      for elem, (i, j) in enumerate(zip(m.qM_fullm_i.numpy(), m.qM_fullm_j.numpy())):
        mjw_out[i, j] = out_smooth_vel.numpy()[0, 0, elem]
    else:
      mjw_out = out_smooth_vel.numpy()[0, : m.nv, : m.nv]

    # Verify RNE contributes non-zero Coriolis/Centrifugal terms.
    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      out_rne = wp.zeros((1, 1, m.nM), dtype=float)
    else:
      out_rne = wp.zeros(d.qM.shape, dtype=float)

    mjw.deriv_smooth_vel(m, d, out_rne, flg_rne=True)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      mjw_rne_out = np.zeros((m.nv, m.nv))
      for elem, (i, j) in enumerate(zip(m.qM_fullm_i.numpy(), m.qM_fullm_j.numpy())):
        mjw_rne_out[i, j] = out_rne.numpy()[0, 0, elem]
    else:
      mjw_rne_out = out_rne.numpy()[0, : m.nv, : m.nv]

    diff_norm = np.linalg.norm(mjw_rne_out - mjw_out)
    if diff_norm < 1e-6:
      qvel_norm = np.linalg.norm(mjd.qvel)
      if qvel_norm > 1e-3:
        raise AssertionError(f"RNE enabled but no derivative change detected! (Diff={diff_norm})")

    # Final comparison against new ground truth: qM - dt * qDeriv
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


if __name__ == "__main__":
  wp.init()
  absltest.main()
