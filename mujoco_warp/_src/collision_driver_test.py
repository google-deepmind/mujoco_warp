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

from . import test_util


class CollisionTest(parameterized.TestCase):
  """Tests the collision contact functions."""

  _BOX_PLANE = """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.3" euler="45 0 0">
              <freejoint/>
              <geom size="0.5 0.5 0.5" type="box"/>
            </body>
          </worldbody>
        </mujoco>
      """
  _PLANE_SPHERE = """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.2" euler="45 0 0">
              <freejoint/>
              <geom size="0.5" type="sphere"/>
            </body>
          </worldbody>
        </mujoco>
        """
  _PLANE_CAPSULE = """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.0" euler="30 30 0">
              <freejoint/>
              <geom size="0.05 0.05" type="capsule"/>
            </body>
          </worldbody>
        </mujoco>
        """
  _CONVEX_CONVEX = """
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
        """
  _CAPSULE_CAPSULE = """
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
        """
  _SPHERE_SPHERE = """
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
        """
  _SPHERE_CAPSULE = """
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
        """
  _PLANE_CYLINDER_1 = """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane" euler="3 0 0"/>
            <body pos="0 0 0.1" euler="30 30 0">
              <freejoint/>
              <geom size="0.05 0.1" type="cylinder"/>
            </body>           
          </worldbody>
        </mujoco>
        """
  _PLANE_CYLINDER_2 = """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane" euler="3 0 0"/>            
            <body pos="0.2 0 0.04" euler="90 0 0">
              <freejoint/>
              <geom size="0.05 0.1" type="cylinder"/>
            </body>            
          </worldbody>
        </mujoco>
        """
  _PLANE_CYLINDER_3 = """
        <mujoco>
          <worldbody>
            <geom size="40 40 40" type="plane" euler="3 0 0"/>            
            <body pos="0.5 0 0.1" euler="3 0 0">
              <freejoint/>
              <geom size="0.05 0.1" type="cylinder"/>
            </body>
          </worldbody>
        </mujoco>
        """
  _SPHERE_BOX = """
        <mujoco>
          <worldbody>
            <geom type="box" pos="0 0 0" size=".5 .5 .5" />
            <body pos="0.2 0.2 0.4">
              <geom type="sphere" size="0.5"/>
              <freejoint/>
            </body>
            <body pos="-0.6 -0.6 0.7">
              <geom type="sphere" size="0.5 0.8"/>
              <freejoint/>
            </body>
          </worldbody>
        </mujoco>
        """
  _CAPSULE_BOX = """
        <mujoco>
          <worldbody>
            <geom type="box" pos="0 0 0" size=".5 .4 .9" />
            <body pos="0.4 0.2 0.8" euler="0 -40 0" >
              <geom type="capsule" size="0.5 0.8"/>
              <freejoint/>
            </body>
            <body pos="0 0 1.5" >
              <geom type="capsule" size="0.5 0.8"/>
              <freejoint/>
            </body>
          </worldbody>
        </mujoco>
        """

  @parameterized.parameters(
    (_BOX_PLANE),
    (_PLANE_SPHERE),
    (_PLANE_CAPSULE),
    (_CONVEX_CONVEX),
    (_SPHERE_SPHERE),
    (_SPHERE_CAPSULE),
    (_SPHERE_BOX),
    (_CAPSULE_CAPSULE),
    (_CAPSULE_BOX),
    (_PLANE_CYLINDER_1),
    (_PLANE_CYLINDER_2),
    (_PLANE_CYLINDER_3),
  )
  def test_collision(self, xml_string):
    """Tests convex collision with different geometries."""
    m = mujoco.MjModel.from_xml_string(xml_string)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mjwarp.put_model(m)
    dx = mjwarp.put_data(m, d)
    # Enable Gjk algorithm
    mjwarp.collision(mx, dx)
    mujoco.mj_collision(m, d)
    for i in range(d.ncon):
      actual_dist = d.contact.dist[i]
      actual_pos = d.contact.pos[i]
      actual_frame = d.contact.frame[i]
      # This is because Gjk generates more contact
      result = False
      for j in range(dx.ncon.numpy()[0]):
        test_dist = dx.contact.dist.numpy()[j]
        test_pos = dx.contact.pos.numpy()[j, :]
        test_frame = dx.contact.frame.numpy()[j].flatten()
        check_dist = np.allclose(actual_dist, test_dist, rtol=5e-2, atol=1.0e-2)
        check_pos = np.allclose(actual_pos, test_pos, rtol=5e-2, atol=1.0e-2)
        check_frame = np.allclose(actual_frame, test_frame, rtol=5e-2, atol=1.0e-2)
        if check_dist and check_pos and check_frame:
          result = True
          break
      np.testing.assert_equal(result, True, f"Contact {i} not found in Gjk results")

  @parameterized.parameters(
    (True, True),
    (True, False),
    (False, True),
    (False, False),
  )
  def test_collision_disableflags(self, dsbl_constraint, dsbl_contact):
    """Tests collision disableflags."""
    mjm, mjd, _, _ = test_util.fixture("humanoid/humanoid.xml")

    if dsbl_constraint:
      mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    if dsbl_contact:
      mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mujoco.mj_collision(mjm, mjd)
    mjwarp.collision(m, d)

    self.assertEqual(d.ncon.numpy()[0], mjd.ncon)


if __name__ == "__main__":
  absltest.main()
