# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native parity for elliptic contact regularization at its numerical floor."""

import itertools

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data


class ContactRegularizationTest(parameterized.TestCase):
  @parameterized.parameters(*itertools.product((3, 4, 6), (0.5, 20), (False, True), (False, True)))
  def test_elliptic_regularization_floor(self, condim, impratio, hinged, with_flex):
    # A body whose center of mass lies on its hinge has zero translational inverse weight,
    # although its offset contact point can move. Explicit pairing retains contact.
    joint = '<joint type="hinge" axis="0 1 0"/>' if hinged else "<freejoint/>"
    flex = '<flexcomp name="unused" type="grid" count="2 2 2" spacing=".1 .1 .1" pos="0 0 5" dim="3"/>' if with_flex else ""
    xml = f"""
      <mujoco>
        <option cone="elliptic" impratio="{impratio}"/>
        <worldbody>
          <geom name="ground" type="plane" size="1 1 .1"/>
          <body pos="0 0 .09">
            {joint}
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
            <geom name="ball" type="sphere" pos=".1 0 0" size=".1"/>
          </body>
          {flex}
        </worldbody>
        <contact>
          <pair geom1="ground" geom2="ball" condim="{condim}" friction=".7 .3 .02 .004 .006"/>
        </contact>
      </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)
    self.assertEqual(mjm.body_invweight0[1, 0] == 0, hinged)
    self.assertEqual(mjd.ncon, 1)
    mjw.make_constraint(m, d)
    address = d.contact.efc_address.numpy()[0, :condim]
    self.assertTrue(np.all(address >= 0))
    native_address = mjd.contact[0].efc_address
    expected = mjd.efc_D[native_address : native_address + condim]
    actual = d.efc.D.numpy()[0, address]
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=0)


if __name__ == "__main__":
  absltest.main()
