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

    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    mujoco.mj_step(mjm, mjd)

    if jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
      out_smooth_vel = wp.zeros((1, 1, m.nM), dtype=float)
    else:
      out_smooth_vel = wp.zeros(d.qM.shape, dtype=float)

    mjw.deriv_smooth_vel(m, d, out_smooth_vel, flg_rne=False)

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

    mj_qM = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(mjm, mj_qM, mjd.qM)

    mj_out = mj_qM

    _assert_eq(mjw_out, mj_out, "qM match (RNE disabled)")

    # Smoke test: Verify RNE contributes non-zero Coriolis/Centrifugal terms.

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


if __name__ == "__main__":
  wp.init()
  absltest.main()
