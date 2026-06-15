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

  def test_sphere_cloth_no_duplicates(self):
    """Test that duplicate/redundant contacts are filtered out."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" tolerance="1e-6" timestep=".001"/>
        <worldbody>
          <!-- Sphere positioned exactly above a vertex shared by multiple elements -->
          <body pos="0 0 0.1">
            <freejoint/>
            <geom type="sphere" size=".1" mass="1"/>
          </body>
          <!-- Cloth (dim=2 flex) -->
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="-.2 -.2 0"
                    radius=".02" dim="2" mass=".5">
            <contact condim="3" selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """
    )

    d.nacon.zero_()
    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0)

    # Retrieve contact positions
    pos = d.contact.pos.numpy()[:nacon]

    # Verify that contact positions are unique (no two contacts are closer than epsilon)
    for i in range(nacon):
      for j in range(i + 1, nacon):
        dist = np.linalg.norm(pos[i] - pos[j])
        self.assertGreater(dist, 1e-3, f"Duplicate contacts found at positions: {pos[i]} and {pos[j]}")

  def test_flex_internal_collision(self):
    """Test that predefined element-vertex internal collisions generate contacts."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" internal="true" margin="0.05"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """
    )

    self.assertGreater(m.nflexevpair, 0)

    # Find a pair
    evpair = m.flex_evpair.numpy()[0]
    e = int(evpair[0])
    v = int(evpair[1])

    # Vertices of element e
    dim = int(m.flex_dim.numpy()[0])
    elem_data_idx = int(m.flex_elemdataadr.numpy()[0]) + e * (dim + 1)
    v_indices = m.flex_elem.numpy()[elem_data_idx : elem_data_idx + dim + 1]

    # Move vertex v close to v0
    v0_global_idx = int(m.flex_vertadr.numpy()[0]) + int(v_indices[0])
    v_global_idx = int(m.flex_vertadr.numpy()[0]) + v

    p0 = d.flexvert_xpos.numpy()[0, v0_global_idx]

    # Set vertex v to be at p0 + small offset in Z (overlapping the element)
    xpos = d.flexvert_xpos.numpy()
    xpos[0, v_global_idx] = p0 + np.array([0.0, 0.0, 0.01])
    d.flexvert_xpos.assign(xpos)

    # Run collision detection
    mjwarp.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected at least one contact from internal self-collision")

    # Verify contact properties
    self.assertEqual(int(d.contact.geom.numpy()[0, 0]), -1)
    self.assertEqual(int(d.contact.geom.numpy()[0, 1]), -1)
    self.assertEqual(int(d.contact.flex.numpy()[0, 0]), 0)
    self.assertEqual(int(d.contact.flex.numpy()[0, 1]), 0)
    self.assertEqual(int(d.contact.dim.numpy()[0]), 3)

  def test_flex_self_collision_1d(self):
    """Test active element self-collisions for 1D ropes (Capsule-Capsule)."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="rope" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="1" mass=".5">
          </flexcomp>
        </worldbody>
      </mujoco>
      """
    )
    m.flex_selfcollide.assign(np.array([4], dtype=np.int32))
    m.nflexevpair = 0

    # Fold rope so vertex 3 is close to vertex 0
    v0_global_idx = int(m.flex_vertadr.numpy()[0])
    v_global_idx = int(m.flex_vertadr.numpy()[0]) + 3
    xpos = d.flexvert_xpos.numpy()
    xpos[0, v_global_idx] = xpos[0, v0_global_idx] + np.array([0.0, 0.0, 0.01])
    d.flexvert_xpos.assign(xpos)

    # Run collision detection
    mjwarp.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected at least one contact from 1D self-collision")

    # Verify contact properties
    found = False
    for idx in range(nacon):
      g0 = int(d.contact.geom.numpy()[idx, 0])
      g1 = int(d.contact.geom.numpy()[idx, 1])
      f0 = int(d.contact.flex.numpy()[idx, 0])
      f1 = int(d.contact.flex.numpy()[idx, 1])
      e0 = int(d.contact.elem.numpy()[idx, 0])
      e1 = int(d.contact.elem.numpy()[idx, 1])

      if g0 == -1 and g1 == -1 and f0 == 0 and f1 == 0:
        if (e0 == 0 and e1 == 2) or (e0 == 2 and e1 == 0):
          found = True
          self.assertGreaterEqual(int(d.contact.dim.numpy()[idx]), 3)
          break

    self.assertTrue(found, "Expected active element self-collision contact between element 0 and 2 not found")

  def test_flex_self_collision_2d(self):
    """Test active element self-collisions for 2D meshes (Triangle-Triangle via GJK/EPA)."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="2" mass=".5">
            <contact selfcollide="auto"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """
    )

    elem_num = m.flex_elemnum.numpy()[0]
    dim = int(m.flex_dim.numpy()[0])
    elem_data_idx = int(m.flex_elemdataadr.numpy()[0])
    elem_verts = m.flex_elem.numpy()[elem_data_idx : elem_data_idx + elem_num * (dim + 1)].reshape(elem_num, dim + 1)

    # Find two elements with disjoint vertices
    e1 = -1
    e2 = -1
    for i in range(elem_num):
      for j in range(i + 1, elem_num):
        if len(set(elem_verts[i]) & set(elem_verts[j])) == 0:
          e1 = i
          e2 = j
          break
      if e1 >= 0:
        break

    self.assertGreaterEqual(e1, 0)
    self.assertGreaterEqual(e2, 0)

    # Calculate center of element 1
    vert_adr = int(m.flex_vertadr.numpy()[0])
    xpos = d.flexvert_xpos.numpy()

    p_center1 = np.zeros(3)
    for v_idx in elem_verts[e1]:
      p_center1 += xpos[0, vert_adr + v_idx]
    p_center1 /= dim + 1

    # Calculate center of element 2
    p_center2 = np.zeros(3)
    for v_idx in elem_verts[e2]:
      p_center2 += xpos[0, vert_adr + v_idx]
    p_center2 /= dim + 1

    # Move element 2 vertices close to element 1 center
    shift = p_center1 - p_center2 + np.array([0.0, 0.0, 0.005])
    for v_idx in elem_verts[e2]:
      xpos[0, vert_adr + v_idx] += shift

    d.flexvert_xpos.assign(xpos)

    # Run collision detection
    mjwarp.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected at least one contact from 2D self-collision")

    # Verify contact properties
    found = False
    for idx in range(nacon):
      g0 = int(d.contact.geom.numpy()[idx, 0])
      g1 = int(d.contact.geom.numpy()[idx, 1])
      f0 = int(d.contact.flex.numpy()[idx, 0])
      f1 = int(d.contact.flex.numpy()[idx, 1])
      elem0 = int(d.contact.elem.numpy()[idx, 0])
      elem1 = int(d.contact.elem.numpy()[idx, 1])

      if g0 == -1 and g1 == -1 and f0 == 0 and f1 == 0:
        if (elem0 == e1 and elem1 == e2) or (elem0 == e2 and elem1 == e1):
          found = True
          self.assertGreaterEqual(int(d.contact.dim.numpy()[idx]), 3)
          break

    self.assertTrue(found, f"Expected active element self-collision contact between element {e1} and {e2} not found")

  def test_flex_self_collision_weld_exclusion(self):
    """Test self-collision exclusions when vertices are welded to the same body."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="rope" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="1" mass=".5">
          </flexcomp>
        </worldbody>
      </mujoco>
      """
    )
    m.flex_selfcollide.assign(np.array([4], dtype=np.int32))
    m.nflexevpair = 0

    # Fold rope
    v0_global_idx = int(m.flex_vertadr.numpy()[0])
    v_global_idx = int(m.flex_vertadr.numpy()[0]) + 3
    xpos = d.flexvert_xpos.numpy()
    xpos[0, v_global_idx] = xpos[0, v0_global_idx] + np.array([0.0, 0.0, 0.01])
    d.flexvert_xpos.assign(xpos)

    # Weld vertex 0 and 3 to same body ID (e.g. 1)
    vertbody = m.flex_vertbodyid.numpy()
    vertbody[v0_global_idx] = 1
    vertbody[v_global_idx] = 1
    m.flex_vertbodyid.assign(vertbody)

    # Run collision detection
    mjwarp.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertEqual(nacon, 0, "Expected 0 contacts due to weld same-body exclusion")

  def test_flex_self_collision_no_adjacent_contacts(self):
    """Test that a flat cloth does not generate any self-collision contacts."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco model="Poncho">
        <option solver="CG" tolerance="1e-6" jacobian="sparse"/>
        <worldbody>
          <flexcomp name="cloth" type="grid" count="10 10 1" spacing="0.05 0.05 0.05"
                    radius="0.01" dim="2" rgba="1 0.5 0.5 1" pos="0 0 2" mass=".1">
            <contact selfcollide="auto"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """
    )

    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertEqual(nacon, 0, f"Expected 0 self-collision contacts on a flat cloth, but got {nacon}")


if __name__ == "__main__":
  absltest.main()
