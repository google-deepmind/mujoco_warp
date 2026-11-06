# Copyright 2026 The Newton Developers
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

"""Contact address initialization regressions."""

import mujoco
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data


class ContactAddressTest(parameterized.TestCase):
  @parameterized.product(nworld=(1, 2), cone=("elliptic", "pyramidal"), jacobian=("dense", "sparse"))
  def test_repeated_mixed_dimension_contacts(self, nworld, cone, jacobian):
    """Sorting must not transfer old rigid rows to unused flex address components."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity=".25 .5 -.75" iterations="30" tolerance="1e-9"><flag warmstart="disable"/></option>
        <worldbody>
          <geom type="plane" size="5 5 .1" condim="1"/>
          <body pos="-1 0 .095">
            <freejoint/>
            <geom type="sphere" size=".1" mass="1" condim="1" priority="1" margin=".01" gap=".04"/>
          </body>
          <body pos="1 0 .095">
            <freejoint/>
            <geom type="sphere" size=".1" mass="1" condim="6" priority="1" friction=".5 .5 .5" margin=".01" gap=".04"/>
          </body>
          <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".1 .1 .1" pos="0 0 .004" dim="2" mass="1">
            <contact condim="3" selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=32,
      njmax=128,
      overrides={"opt.cone": cone, "opt.jacobian": jacobian},
    )
    m.opt.deterministic = mjw.DeterminismType.CONTACTS | mjw.DeterminismType.CONSTRAINT
    qpos = d.qpos.numpy().copy()
    qvel = np.full((nworld, m.nv), 0.03125, dtype=np.float32)
    if nworld == 2:
      qpos[1, 2] += 0.002
      qpos[1, 9] -= 0.001
      qvel[1] = -0.0625
    d.qpos.assign(qpos)
    d.qvel.assign(qvel)
    expected = []
    for world in range(nworld):
      mjd = mujoco.MjData(mjm)
      mjd.qpos[:], mjd.qvel[:] = qpos[world], qvel[world]
      mujoco.mj_forward(mjm, mjd)
      expected.append(mjd.qacc.copy())
    if nworld == 2:
      self.assertFalse(np.array_equal(expected[0], expected[1]))

    # Start from unwritten output; subsequent calls reuse actual previous outputs.
    d.contact.efc_address.fill_(-1)
    for _ in range(3):
      mjw.forward(m, d)
      np.testing.assert_array_equal(d.overflow.numpy(), 0)
      ncon = int(d.nacon.numpy()[0])
      addresses = d.contact.efc_address.numpy().copy()[:ncon]
      dimensions = d.contact.dim.numpy().copy()[:ncon]
      for conid, condim in enumerate(dimensions):
        width = int(condim) if cone == "elliptic" else max(1, 2 * (int(condim) - 1))
        np.testing.assert_array_equal(addresses[conid, width:], -1)
      np.testing.assert_allclose(d.qacc.numpy(), expected, rtol=1e-5, atol=1e-5)

  @parameterized.product(nworld=(1, 2), cone=("elliptic", "pyramidal"), jacobian=("dense", "sparse"))
  def test_standalone_active_to_gap(self, nworld, cone, jacobian):
    """Assembly owns output addresses even when externally supplied contacts become inactive."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .095"><freejoint/><geom size=".1"/></body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=4,
      njmax=16,
      overrides={"opt.cone": cone, "opt.jacobian": jacobian},
    )
    m.opt.deterministic = mjw.DeterminismType.CONTACTS | mjw.DeterminismType.CONSTRAINT
    if nworld == 2:
      qpos = d.qpos.numpy().copy()
      qpos[1, 2] -= 0.002
      d.qpos.assign(qpos)
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertGreater(count, 0)
    self.assertTrue(np.all(d.contact.efc_address.numpy()[:count, 0] >= 0))
    # The external producer updates contact inputs, not the previous assembly's output addresses.
    distance = d.contact.dist.numpy().copy()
    distance[:count] = d.contact.includemargin.numpy()[:count] + 0.01
    d.contact.dist.assign(distance)
    mjw.make_constraint(m, d)
    np.testing.assert_array_equal(d.nefc.numpy(), 0)
    np.testing.assert_array_equal(d.contact.efc_address.numpy()[:count], -1)

  @parameterized.product(nworld=(1, 2), cone=("elliptic", "pyramidal"), deterministic=(0, 3))
  def test_collision_clears_flex_addresses(self, nworld, cone, deterministic):
    """Fresh collision contacts have no associated constraint rows, just as in native MuJoCo."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1" condim="1"/>
          <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".1 .1 .1" pos="0 0 .004" dim="2" mass="1">
            <contact condim="3" selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=16,
      njmax=64,
      overrides={"opt.cone": cone},
    )
    m.opt.deterministic = deterministic
    qvel = np.full((nworld, m.nv), 0.03125, dtype=np.float32)
    if nworld == 2:
      qvel[1] = -0.0625
    d.qvel.assign(qvel)
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertGreater(count, 0)
    self.assertTrue(np.all(d.contact.efc_address.numpy()[:count, 0] >= 0))
    mjw.collision(m, d)
    np.testing.assert_array_equal(d.contact.efc_address.numpy()[:count], -1)
    np.testing.assert_array_equal(d.overflow.numpy(), 0)
    native = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, native)
    mujoco.mj_collision(mjm, native)
    np.testing.assert_array_equal(native.contact.efc_address, -1)


if __name__ == "__main__":
  absltest.main()
