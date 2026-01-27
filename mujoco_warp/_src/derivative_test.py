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

  @parameterized.product(
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_rne_vel_effect(self, jacobian):
    """Tests that RNE derivative computes non-zero terms for centrifugal/coriolis effects."""
    # A double pendulum or spinning body should have RNE terms (d(C(q,qdot))/dqdot)
    # The default sphere in previous tests might be too simple, use something with rotation.
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body pos="0 0 0">
          <joint type="hinge" axis="1 0 0"/>
          <geom type="capsule" fromto="0 0 0 0 1 0" size="0.1"/>
          <body pos="0 1 0">
            <joint type="hinge" axis="0 0 1"/>
            <geom type="capsule" fromto="0 0 0 1 0 0" size="0.1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
      """,
      keyframe=0,
      overrides={"opt.jacobian": jacobian},
      qvel_noise=1.0,  # randomize velocity to ensure coriolis terms exist
    )

    # Run step to populate data
    mjw.step(m, d)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      shape = (1, 1, m.nM)
    else:
      shape = (1, m.nv, m.nv)

    # Compute with RNE
    out_rne = wp.zeros(shape, dtype=float)
    mjw.deriv_smooth_vel(m, d, out_rne, flg_rne=True)

    # Compute without RNE
    out_no_rne = wp.zeros(shape, dtype=float)
    mjw.deriv_smooth_vel(m, d, out_no_rne, flg_rne=False)

    # Difference should be non-zero if RNE is working and physics dictates it
    res_rne = out_rne.numpy()
    res_no_rne = out_no_rne.numpy()

    # We expect a difference due to RNE specific terms (e.g. coriolis/centrifugal effects:
    # cinert * cacc + cvel x (cinert * cvel)).
    #
    # Note: `deriv_smooth_vel` (without RNE) only computes damping and actuation derivatives.
    # While the pure mass matrix part is handled separately in implicit integration (M - dt*qDeriv),
    # the RNE contributions must be added directly to qDeriv here.

    diff = np.linalg.norm(res_rne - res_no_rne)
    self.assertGreater(diff, 1e-6, "RNE derivative should contribute non-zero terms for this model")


if __name__ == "__main__":
  wp.init()
  absltest.main()
