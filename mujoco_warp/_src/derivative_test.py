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


def _assert_eq(a, b, name):
  tol = 1e-4
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
      out_smooth_vel = wp.zeros((1, m.nv, m.nv), dtype=float)

    mjw.deriv_smooth_vel(m, d, out_smooth_vel)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      mjw_out = np.zeros((m.nv, m.nv))
      for elem, (i, j) in enumerate(zip(m.qM_fullm_i.numpy(), m.qM_fullm_j.numpy())):
        mjw_out[i, j] = out_smooth_vel.numpy()[0, 0, elem]
    else:
      mjw_out = out_smooth_vel.numpy()[0]

    mj_qDeriv = np.zeros((mjm.nv, mjm.nv))
    mujoco.mju_sparse2dense(mj_qDeriv, mjd.qDeriv, mjm.D_rownnz, mjm.D_rowadr, mjm.D_colind)

    mj_qM = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(mjm, mj_qM, mjd.qM)
    mj_out = mj_qM - mjm.opt.timestep * mj_qDeriv

    _assert_eq(mjw_out, mj_out, "qM - dt * qDeriv")

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


if __name__ == "__main__":
  wp.init()
  absltest.main()
