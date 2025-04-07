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

"""Tests for constraint functions."""

import mujoco
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp

from . import test_util

# tolerance for difference between MuJoCo and MJWarp constraint calculations,
# mostly due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class ConstraintTest(parameterized.TestCase):
  def test_contact_frictionless(self):
    """Test frictionless contact (condim=1)."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body pos="0 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" condim="1"/>
          </body>
          <body pos="1 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" condim="1"/>
          </body>
          <body pos="2 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" condim="3"/>
          </body>
        </worldbody>
        <keyframe>
          <key qpos='0 0 0 1 0 0 0
                     0 0 0 1 0 0 0
                     0 0 0 1 0 0 0'/>
        </keyframe>
      </mujoco>
    """)

    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mjwarp.make_constraint(m, d)

    _assert_eq(d.ncon.numpy()[0], 3, "ncon")
    _assert_eq(d.nefc.numpy()[0], 9, "nefc")
    _assert_eq(d.efc.J.numpy()[: mjd.nefc, :].reshape(-1), mjd.efc_J, "efc_J")
    _assert_eq(d.efc.D.numpy()[: mjd.nefc], mjd.efc_D, "efc_D")
    _assert_eq(d.efc.aref.numpy()[: mjd.nefc], mjd.efc_aref, "efc_aref")
    _assert_eq(d.efc.pos.numpy()[: mjd.nefc], mjd.efc_pos, "efc_pos")
    _assert_eq(d.efc.margin.numpy()[: mjd.nefc], mjd.efc_margin, "efc_margin")

  @parameterized.parameters(
    mujoco.mjtCone.mjCONE_PYRAMIDAL,
    mujoco.mjtCone.mjCONE_ELLIPTIC,
  )
  def test_constraints(self, cone):
    """Test constraints."""
    mjm, mjd, _, _ = test_util.fixture("constraints.xml", sparse=False, cone=cone)

    for key in range(3):
      mujoco.mj_resetDataKeyframe(mjm, mjd, key)

      mujoco.mj_forward(mjm, mjd)
      m = mjwarp.put_model(mjm)
      d = mjwarp.put_data(mjm, mjd)
      mjwarp.make_constraint(m, d)

      _assert_eq(d.efc.J.numpy()[: mjd.nefc, :].reshape(-1), mjd.efc_J, "efc_J")
      _assert_eq(d.efc.D.numpy()[: mjd.nefc], mjd.efc_D, "efc_D")
      _assert_eq(d.efc.aref.numpy()[: mjd.nefc], mjd.efc_aref, "efc_aref")
      _assert_eq(d.efc.pos.numpy()[: mjd.nefc], mjd.efc_pos, "efc_pos")
      _assert_eq(d.efc.margin.numpy()[: mjd.nefc], mjd.efc_margin, "efc_margin")


if __name__ == "__main__":
  absltest.main()
