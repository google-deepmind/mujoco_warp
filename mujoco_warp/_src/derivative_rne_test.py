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

"""Tests for RNE derivative functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data


class RNEDerivativeTest(parameterized.TestCase):

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
      qvel_noise=1.0, # randomize velocity to ensure coriolis terms exist
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

    # We expect some difference due to "cinert * cacc + cvel x (cinert * cvel)" and similar terms
    # derivatives which are captured in RNE but not in simple damping/actuation derivatives (unless
    # they are coincidentally zero)
    # Actually, deriv_smooth_vel without RNE only calculates:
    # - damping
    # - actuation derivatives
    # - qM (mass matrix) part if added? No, deriv_smooth_vel calculates "qDeriv" which is
    # D(qfrc_smooth)/D(qvel).
    # The pure mass matrix part is usually handled separately in implicit integration
    # (M - dt*qDeriv).
    # BUT, rne_vel adds to qDeriv.
    # Let's verify that res_rne is NOT equal to res_no_rne.

    diff = np.linalg.norm(res_rne - res_no_rne)
    self.assertGreater(diff, 1e-6, "RNE derivative should contribute non-zero terms for this model")

    # Double check against finite differences?
    # That might be complex to set up here without modifying the state and re-running inverse
    # dynamics.
    # For now, scoping down means ensuring the function runs and contributes.

if __name__ == "__main__":
  wp.init()
  absltest.main()
