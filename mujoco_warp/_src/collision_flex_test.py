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

import numpy as np
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

  def test_sphere_cloth_pruned_by_broadphase(self):
    """Test that far-away geoms are successfully pruned by broadphase (yielding 0 contacts)."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".001"/>
      <size memory="10M"/>

      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>

        <!-- Sphere positioned very far away from the cloth -->
        <body pos="10.0 10.0 10.0">
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
    _, _, m, d = test_data.fixture(xml=xml)

    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertEqual(nacon, 0, "Expected 0 contacts because the sphere is very far away")

  def test_sphere_cloth_exact_bounds(self):
    """Test that the dynamic flex AABB calculation computes the exact expected bounding box."""
    xml = """
    <mujoco>
      <worldbody>
        <!-- Cloth (dim=2 flex) -->
        <flexcomp type="grid" count="4 4 1" spacing=".2 .2 .1" pos="-.3 -.3 0"
                  radius=".02" name="cloth" dim="2" mass=".5">
          <contact selfcollide="none"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml)

    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)

    # Fetch computed AABB bounds
    aabb_min = d.flex_aabb_min.numpy()[0, 0]
    aabb_max = d.flex_aabb_max.numpy()[0, 0]

    # Fetch the actual vertices from flexvert_xpos
    vert_adr = m.flex_vertadr.numpy()[0]
    vert_num = m.flex_vertnum.numpy()[0]
    verts = d.flexvert_xpos.numpy()[0, vert_adr : vert_adr + vert_num]

    # Compute the expected bounds from actual vertices
    v_min = np.min(verts, axis=0)
    v_max = np.max(verts, axis=0)

    # Inflation
    radius = m.flex_radius.numpy()[0]
    margin = m.flex_margin.numpy()[0] + m.flex_gap.numpy()[0]
    inflate = radius + margin

    expected_min = v_min - inflate
    expected_max = v_max + inflate

    np.testing.assert_allclose(aabb_min, expected_min, atol=1e-5)
    np.testing.assert_allclose(aabb_max, expected_max, atol=1e-5)

  def test_plane_cloth_contact_generated(self):
    """Test that contacts are generated between plane and cloth vertices."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".001"/>
      <size memory="10M"/>

      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>

        <!-- Ground plane -->
        <geom type="plane" size="5 5 .1" pos="0 0 0"/>

        <!-- Cloth (dim=2 flex) placed just above the plane -->
        <flexcomp type="grid" count="4 4 1" spacing=".2 .2 .1" pos="-.3 -.3 0.01"
                  radius=".02" name="cloth" dim="2" mass=".5">
          <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001"
                   selfcollide="none" conaffinity="1" contype="1"/>
          <edge damping="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml)

    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected contacts between plane and cloth vertices")

    # Verify that contact geom is the plane
    contact_geom = d.contact.geom.numpy()[:nacon]
    # Check that at least one contact involves geom 0 (the plane)
    plane_contacts = np.sum(contact_geom[:, 0] == 0)
    self.assertGreater(plane_contacts, 0, "Expected at least one contact with the plane")


if __name__ == "__main__":
  absltest.main()
