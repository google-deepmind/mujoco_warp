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
"""Tests for flex element collision."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp
from mujoco_warp import test_data


class FlexCollisionTest(absltest.TestCase):
  """Tests for flex element collision detection."""

  def test_sphere_cloth_contact_generated(self):
    """Test that contacts are generated between sphere and cloth."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".001"/>
      <size memory="10M"/>

      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>

        <!-- Ground plane -->
        <geom type="plane" size="5 5 .1" pos="0 0 0"/>

        <!-- Sphere positioned just above the cloth -->
        <body pos="0 0 0.12">
          <freejoint/>
          <geom type="sphere" size=".1" mass="1"/>
        </body>

        <!-- Cloth (dim=2 flex) -->
        <flexcomp type="grid" count="4 4 1" spacing=".2 .2 .1" pos="-.3 -.3 0"
                  radius=".02" name="cloth" dim="2" mass=".5">
          <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001"
                   selfcollide="none" conaffinity="1" contype="1"/>
          <edge damping="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, _, m, d = test_data.fixture(xml=xml)

    self.assertEqual(mjm.nflex, 1)
    self.assertEqual(mjm.flex_dim[0], 2)

    self.assertEqual(m.nflex, 1)
    self.assertGreater(m.flex_elemnum.numpy()[0], 0)

    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)

    nacon = int(d.nacon.numpy()[0])

    # Sphere is just above the cloth, so there should be contacts
    self.assertGreater(nacon, 0, "Expected contacts between sphere and cloth")

  def test_folded_cloth_self_contacts_match_c(self):
    """Folded cloth self-contacts should match between C and Warp."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".002"/>
      <size memory="10M"/>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="5 5 .1" pos="0 0 -1"/>
        <flexcomp type="grid" count="8 4 1" spacing=".05 .05 .1"
                  pos="-.175 -.075 0.5" radius=".005" name="cloth" dim="2"
                  mass=".5">
          <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001"
                   selfcollide="narrow" conaffinity="1" contype="1"/>
          <edge damping="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetData(mjm, mjd)
    mujoco.mj_forward(mjm, mjd)

    # Fold the right half of the cloth over the left half.
    nvert = mjm.flex_vertnum[0]
    vertadr = mjm.flex_vertadr[0]
    verts = mjd.flexvert_xpos[vertadr : vertadr + nvert].copy()
    mid_x = (verts[:, 0].min() + verts[:, 0].max()) / 2.0

    for i in range(nvert):
      bodyid = mjm.flex_vertbodyid[i]
      jntid = mjm.body_jntadr[bodyid]
      qposadr = mjm.jnt_qposadr[jntid]
      if verts[i, 0] > mid_x + 1e-6:
        dx = verts[i, 0] - mid_x
        new_x = mid_x - dx - 0.05
        delta_x = new_x - verts[i, 0]
        mjd.qpos[qposadr + 0] = delta_x
        mjd.qpos[qposadr + 1] = 0.0
        mjd.qpos[qposadr + 2] = -0.002

    mujoco.mj_forward(mjm, mjd)

    # Count C self-contacts.
    c_self = 0
    for c in range(mjd.ncon):
      con = mjd.contact[c]
      if con.flex[0] >= 0 and con.flex[0] == con.flex[1]:
        c_self += 1

    self.assertGreater(c_self, 0, f"C should produce self-contacts")

    # Run Warp with matched state.
    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)
    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)
    wp.synchronize_device()

    nacon = int(d.nacon.numpy()[0])
    if nacon > 0:
      flex_arr = d.contact.flex.numpy()[:nacon]
      warp_self = int(np.sum((flex_arr[:, 0] >= 0) & (flex_arr[:, 0] == flex_arr[:, 1])))
    else:
      warp_self = 0

    self.assertGreater(warp_self, 0, f"Warp should produce self-contacts (nacon={nacon})")
    self.assertGreaterEqual(warp_self, c_self, f"Warp={warp_self} should be >= C={c_self} self-contacts")

  def test_flat_cloth_self_contacts_match_c(self):
    """Flat cloth (no fold) self-contacts should match C MuJoCo."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".002"/>
      <size memory="10M"/>
      <worldbody>
        <flexcomp type="grid" count="10 10 1" spacing=".05 .05 .05"
                  radius=".02" name="cloth" dim="2" mass=".1"
                  pos="0 0 2">
          <contact selfcollide="narrow" condim="3"/>
          <edge damping="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm_c = mujoco.MjModel.from_xml_string(xml)
    mjd_c = mujoco.MjData(mjm_c)
    mujoco.mj_forward(mjm_c, mjd_c)
    c_self = 0
    for c in range(mjd_c.ncon):
      con = mjd_c.contact[c]
      if con.flex[0] >= 0 and con.flex[0] == con.flex[1]:
        c_self += 1

    mjd_clean = mujoco.MjData(mjm_c)
    mujoco.mj_resetData(mjm_c, mjd_clean)
    mujoco.mj_kinematics(mjm_c, mjd_clean)
    mujoco.mj_comPos(mjm_c, mjd_clean)
    m = mjwarp.put_model(mjm_c)
    d = mjwarp.put_data(mjm_c, mjd_clean)
    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)
    wp.synchronize_device()

    nacon = int(d.nacon.numpy()[0])
    if nacon > 0:
      flex_arr = d.contact.flex.numpy()[:nacon]
      warp_self = int(np.sum((flex_arr[:, 0] >= 0) & (flex_arr[:, 0] == flex_arr[:, 1])))
    else:
      warp_self = 0

    self.assertGreaterEqual(warp_self, c_self, f"Warp={warp_self} should be >= C={c_self} self-contacts")

  def test_unsupported_selfcollide_raises(self):
    """Verify unsupported selfcollide modes raise NotImplementedError."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".002"/>
      <size memory="10M"/>
      <worldbody>
        <flexcomp type="grid" count="3 3 1" spacing=".1 .1 .1" pos="0 0 0"
                  radius=".01" name="cloth" dim="2" mass=".5">
          <contact selfcollide="auto" conaffinity="1" contype="1"/>
          <edge damping="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)


if __name__ == "__main__":
  absltest.main()
