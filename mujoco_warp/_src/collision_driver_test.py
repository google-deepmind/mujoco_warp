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
"""Tests the collision driver."""

import mujoco
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp

from . import io
from . import test_util


class CollisionTest(parameterized.TestCase):
  """Tests the collision contact functions."""

  _FIXTURES = {
    "box_plane": """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.3" euler="45 0 0">
              <freejoint/>
              <geom size="0.5 0.5 0.5" type="box"/>
            </body>
          </worldbody>
        </mujoco>
      """,
    "box_mesh": """
        <mujoco>
          <asset>
            <mesh name="boxmesh" scale="0.1 0.1 0.1"
                  vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1
                           1 -1 1 -1 1 -1 -1 1 1 -1 -1 1"/>
          </asset>
          <worldbody>
            <geom pos="0 0 -0.1" type="box" size="0.5 0.5 0.1"/>
            <body pos="0 0 .099">
              <joint type="free"/>
              <geom type="mesh" mesh="boxmesh"/>
            </body>
          </worldbody>
        </mujoco>
      """,
    "plane_sphere": """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.2" euler="45 0 0">
              <freejoint/>
              <geom size="0.5" type="sphere"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "plane_capsule": """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.0" euler="30 30 0">
              <freejoint/>
              <geom size="0.05 0.05" type="capsule"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "convex_convex": """
        <mujoco>
          <asset>
            <mesh name="poly"
            vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
            face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
          </asset>
          <worldbody>
            <body pos="0.0 2.0 0.35" euler="0 0 90">
              <freejoint/>
              <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
            </body>
            <body pos="0.0 2.0 2.281" euler="180 0 0">
              <freejoint/>
              <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "capsule_capsule": """
        <mujoco model="two_capsules">
          <worldbody>
            <body>
              <joint type="free"/>
              <geom fromto="0.62235904  0.58846647 0.651046 1.5330081 0.33564585 0.977849"
               size="0.05" type="capsule"/>
            </body>
            <body>
              <joint type="free"/>
              <geom fromto="0.5505271 0.60345304 0.476661 1.3900293 0.30709633 0.932082"
               size="0.05" type="capsule"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "sphere_sphere": """
        <mujoco>
          <worldbody>
            <body>
              <joint type="free"/>
              <geom pos="0 0 0" size="0.2" type="sphere"/>
            </body>
            <body >
              <joint type="free"/>
              <geom pos="0 0.3 0" size="0.11" type="sphere"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "sphere_capsule": """
        <mujoco>
          <worldbody>
            <body>
              <joint type="free"/>
              <geom pos="0 0 0" size="0.2" type="sphere"/>
            </body>
            <body>
              <joint type="free"/>
              <geom fromto="0.3 0 0 0.7 0 0" size="0.1" type="capsule"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "sphere_cylinder_corner": """
        <mujoco>
          <worldbody>
            <body>
              <joint type="slide" axis="1 0 0"/>
              <joint type="slide" axis="0 1 0"/>
              <joint type="slide" axis="0 0 1"/>
              <geom size="0.1" type="sphere" pos=".33 0 0"/>
            </body>
            <body>
              <geom size="0.15 0.2" type="cylinder" euler="30 45 0"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "sphere_cylinder_cap": """
        <mujoco>
          <worldbody>
            <body>
              <joint type="slide" axis="1 0 0"/>
              <joint type="slide" axis="0 1 0"/>
              <joint type="slide" axis="0 0 1"/>
              <geom size="0.1" type="sphere" pos=".26 -.14 .1"/>
            </body>
            <body>
              <geom size="0.15 0.2" type="cylinder" euler="30 45 0"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "sphere_cylinder_side": """
        <mujoco>
          <worldbody>
            <body>
              <joint type="slide" axis="1 0 0"/>
              <joint type="slide" axis="0 1 0"/>
              <joint type="slide" axis="0 0 1"/>
              <geom size="0.1" type="sphere" pos="0 -.26 0"/>
            </body>
            <body>
              <geom size="0.15 0.2" type="cylinder" euler="30 45 0"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "plane_cylinder_1": """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane" euler="3 0 0"/>
            <body pos="0 0 0.1" euler="30 30 0">
              <freejoint/>
              <geom size="0.05 0.1" type="cylinder"/>
            </body>           
          </worldbody>
        </mujoco>
        """,
    "plane_cylinder_2": """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane" euler="3 0 0"/>            
            <body pos="0.2 0 0.04" euler="90 0 0">
              <freejoint/>
              <geom size="0.05 0.1" type="cylinder"/>
            </body>            
          </worldbody>
        </mujoco>
        """,
    "plane_cylinder_3": """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane" euler="3 0 0"/>            
            <body pos="0.5 0 0.1" euler="3 0 0">
              <freejoint/>
              <geom size="0.05 0.1" type="cylinder"/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "sphere_box_shallow": """
        <mujoco>
          <worldbody>
            <geom type="box" pos="0 0 0" size=".5 .5 .5" />
            <body pos="-0.6 -0.6 0.7">
              <geom type="sphere" size="0.5"/>
              <freejoint/>
            </body>
          </worldbody>
        </mujoco>
        """,
    "sphere_box_deep": """
        <mujoco>
          <worldbody>
            <geom type="box" pos="0 0 0" size=".5 .5 .5" />
            <body pos="-0.6 -0.6 0.7">
              <geom type="sphere" size="0.5"/>
              <freejoint/>
            </body>
          </worldbody>
        </mujoco>
        """,
  }

  @parameterized.parameters(_FIXTURES.keys())
  def test_collision(self, fixture):
    """Tests convex collision with different geometries."""
    _, mjd, _, d = test_util.fixture(xml=self._FIXTURES[fixture])

    for i in range(mjd.ncon):
      actual_dist = mjd.contact.dist[i]
      actual_pos = mjd.contact.pos[i]
      actual_frame = mjd.contact.frame[i]
      # This is because Gjk generates more contact
      result = False
      for j in range(d.ncon.numpy()[0]):
        test_dist = d.contact.dist.numpy()[j]
        test_pos = d.contact.pos.numpy()[j, :]
        test_frame = d.contact.frame.numpy()[j].flatten()
        check_dist = np.allclose(actual_dist, test_dist, rtol=5e-2, atol=1.0e-2)
        check_pos = np.allclose(actual_pos, test_pos, rtol=5e-2, atol=1.0e-2)
        check_frame = np.allclose(actual_frame, test_frame, rtol=5e-2, atol=1.0e-2)
        if check_dist and check_pos and check_frame:
          result = True
          break
      np.testing.assert_equal(result, True, f"Contact {i} not found in Gjk results")

  def test_contact_exclude(self):
    """Tests contact exclude."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body name="body1">
            <freejoint/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="body2">
            <freejoint/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="body3">
            <freejoint/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <contact>
          <exclude body1="body1" body2="body2"/>
        </contact>
      </mujoco>
    """)
    geompair, pairid = io.geom_pair(mjm)
    self.assertEqual(geompair.shape[0], 3)
    np.testing.assert_equal(pairid, np.array([-2, -1, -1]))

  def test_contact_pair(self):
    """Tests contact pair."""
    # no pairs
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    _, pairid = io.geom_pair(mjm)
    self.assertTrue((pairid == -1).all())

    # 1 pair
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom name="geom1" type="sphere" size=".1"/>
          </body>
          <body>
            <freejoint/>
            <geom name="geom2" type="sphere" size=".1"/>
          </body>
        </worldbody>
        <contact>
          <pair geom1="geom1" geom2="geom2" margin="2" gap="3" condim="6" friction="5 4 3 2 1" solref="-.25 -.5" solreffriction="2 4" solimp=".1 .2 .3 .4 .5"/>
        </contact>
      </mujoco>
    """)
    _, pairid = io.geom_pair(mjm)
    self.assertTrue((pairid == 0).all())

    # generate contact
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm)
    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)

    self.assertEqual(d.ncon.numpy()[0], 1)
    self.assertEqual(d.contact.includemargin.numpy()[0], -1)
    self.assertEqual(d.contact.dim.numpy()[0], 6)
    np.testing.assert_allclose(d.contact.friction.numpy()[0], np.array([5, 4, 3, 2, 1]))
    np.testing.assert_allclose(d.contact.solref.numpy()[0], np.array([-0.25, -0.5]))
    np.testing.assert_allclose(
      d.contact.solreffriction.numpy()[0], np.array([2.0, 4.0])
    )
    np.testing.assert_allclose(
      d.contact.solimp.numpy()[0], np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    )

    # 1 pair: override contype and conaffinity
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body name="body1">
            <freejoint/>
            <geom name="geom1" type="sphere" size=".1" contype="0" conaffinity="0"/>
          </body>
          <body name="body2">
            <freejoint/>
            <geom name="geom2" type="sphere" size=".1" contype="0" conaffinity="0"/>
          </body>
        </worldbody>
        <contact>
          <pair geom1="geom1" geom2="geom2" margin="2" gap="3" condim="6" friction="5 4 3 2 1" solref="-.25 -.5" solreffriction="2 4" solimp=".1 .2 .3 .4 .5"/>
        </contact>
      </mujoco>
    """)
    _, pairid = io.geom_pair(mjm)
    self.assertTrue((pairid == 0).all())

    # generate contact
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm)
    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)

    self.assertEqual(d.ncon.numpy()[0], 1)
    self.assertEqual(d.contact.includemargin.numpy()[0], -1)
    self.assertEqual(d.contact.dim.numpy()[0], 6)
    np.testing.assert_allclose(d.contact.friction.numpy()[0], np.array([5, 4, 3, 2, 1]))
    np.testing.assert_allclose(d.contact.solref.numpy()[0], np.array([-0.25, -0.5]))
    np.testing.assert_allclose(
      d.contact.solreffriction.numpy()[0], np.array([2.0, 4.0])
    )
    np.testing.assert_allclose(
      d.contact.solimp.numpy()[0], np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    )

    # 1 pair: override exclude
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body name="body1">
            <freejoint/>
            <geom name="geom1" type="sphere" size=".1"/>
          </body>
          <body name="body2">
            <freejoint/>
            <geom name="geom2" type="sphere" size=".1"/>
          </body>
        </worldbody>
        <contact>
          <exclude body1="body1" body2="body2"/>
          <pair geom1="geom1" geom2="geom2" margin="2" gap="3" condim="6" friction="5 4 3 2 1" solref="-.25 -.5" solreffriction="2 4" solimp=".1 .2 .3 .4 .5"/>
        </contact>
      </mujoco>
    """)
    _, pairid = io.geom_pair(mjm)
    self.assertTrue((pairid == 0).all())

    # generate contact
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm)
    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)

    self.assertEqual(d.ncon.numpy()[0], 1)
    self.assertEqual(d.contact.includemargin.numpy()[0], -1)
    self.assertEqual(d.contact.dim.numpy()[0], 6)
    np.testing.assert_allclose(d.contact.friction.numpy()[0], np.array([5, 4, 3, 2, 1]))
    np.testing.assert_allclose(d.contact.solref.numpy()[0], np.array([-0.25, -0.5]))
    np.testing.assert_allclose(
      d.contact.solreffriction.numpy()[0], np.array([2.0, 4.0])
    )
    np.testing.assert_allclose(
      d.contact.solimp.numpy()[0], np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    )

    # 1 pair 1 exclude
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body name="body1">
            <freejoint/>
            <geom name="geom1" type="sphere" size=".1"/>
          </body>
          <body name="body2">
            <freejoint/>
            <geom name="geom2" type="sphere" size=".1"/>
          </body>
          <body name="body3">
            <freejoint/>
            <geom name="geom3" type="sphere" size=".1"/>
          </body>
        </worldbody>
        <contact>
          <exclude body1="body1" body2="body2"/>
          <pair geom1="geom2" geom2="geom3" margin="2" gap="3" condim="6" friction="5 4 3 2 1" solref="-.25 -.5" solreffriction="2 4" solimp=".1 .2 .3 .4 .5"/>
        </contact>
      </mujoco>
    """)
    _, pairid = io.geom_pair(mjm)
    np.testing.assert_equal(pairid, np.array([-2, -1, 0]))

    # generate contact
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm)
    mjwarp.kinematics(m, d)
    mjwarp.collision(m, d)

    self.assertEqual(d.ncon.numpy()[0], 2)
    self.assertEqual(d.contact.includemargin.numpy()[1], -1)
    self.assertEqual(d.contact.dim.numpy()[1], 6)
    np.testing.assert_allclose(d.contact.friction.numpy()[1], np.array([5, 4, 3, 2, 1]))
    np.testing.assert_allclose(d.contact.solref.numpy()[1], np.array([-0.25, -0.5]))
    np.testing.assert_allclose(
      d.contact.solreffriction.numpy()[1], np.array([2.0, 4.0])
    )
    np.testing.assert_allclose(
      d.contact.solimp.numpy()[1], np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    )

    # TODO(team): test sap_broadphase

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, True),
    (False, False),
  )
  def test_collision_disableflags(self, constraint, contact):
    """Tests collision disableflags."""
    mjm, mjd, m, d = test_util.fixture(
      "humanoid/humanoid.xml",
      keyframe=0,
      constraint=constraint,
      contact=contact,
      kick=False,
    )

    mujoco.mj_collision(mjm, mjd)
    mjwarp.collision(m, d)

    self.assertEqual(d.ncon.numpy()[0], mjd.ncon)

  # TODO(team): test contact parameter mixing


if __name__ == "__main__":
  absltest.main()
