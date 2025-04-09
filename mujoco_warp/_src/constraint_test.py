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
  def test_condim(self):
    """Test condim."""
    xml = """
      <mujoco>
        <worldbody>
          <body pos="0 0 0">
            <joint type="slide"/>
            <geom type="sphere" size=".1" condim="1"/>
          </body>
          <body pos="1 0 0">
            <joint type="slide"/>
            <geom type="sphere" size=".1" condim="1"/>
          </body>
          <body pos="2 0 0">
            <joint type="slide"/>
            <geom type="sphere" size=".1" condim="3"/>
          </body>
          <body pos="3 0 0">
            <joint type="slide"/>
            <geom type="sphere" size=".1" condim="3"/>
          </body>
          <body pos="4 0 0">
            <joint type="slide"/>
            <geom type="sphere" size=".1" condim="4"/>
          </body>
          <body pos="5 0 0">
            <joint type="slide"/>
            <geom type="sphere" size=".1" condim="4"/>
          </body>
          <body pos="6 0 0">
            <joint type="slide"/>
            <geom type="sphere" size=".1" condim="6"/>
          </body>
          <body pos="7 0 0">
            <joint type="slide"/>
            <geom type="sphere" size=".1" condim="6"/>
          </body>
        </worldbody>
        <keyframe>
          <key qpos='0 0 0 0 0 0 0 0'/>
          <key qpos='0 .01 .02 .03 .04 .05 .06 .07'/>
          <key qpos='0 0 1 2 3 4 5 6'/>
          <key qpos='1 2 0 0 3 4 5 6'/>
          <key qpos='1 2 3 4 0 0 5 6'/>
          <key qpos='1 2 3 4 5 6 0 0'/>
        </keyframe>
      </mujoco>
    """

    # TODO(team): test elliptic friction cone

    for keyframe in range(6):
      _, mjd, m, d = test_util.fixture(xml=xml, keyframe=keyframe)
      mjwarp.make_constraint(m, d)

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
    mjm, mjd, _, _ = test_util.fixture("constraints.xml", cone=cone)

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
