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
from . import collision_driver
from . import test_util

import mujoco_warp as mjwarp

# tolerance for difference between MuJoCo and MJWarp calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class PrimitiveTest(parameterized.TestCase):
  """Tests the collision primitive functions."""

  _MJCFS = {
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
  }

  @parameterized.parameters(
    "box_plane",
    "plane_sphere",
    "sphere_sphere",
    "plane_capsule",
    "capsule_capsule",
  )
  def test_primitives(self, name):
    """Tests collision primitive functions."""
    mjm = mujoco.MjModel.from_xml_string(self._MJCFS[name])
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mujoco.mj_collision(mjm, mjd)
    mjwarp.collision(m, d)

    ncon = d.ncon.numpy()[0]
    np.testing.assert_equal(ncon, mjd.ncon)

    for i in range(ncon):
      _assert_eq(d.contact.dist.numpy()[i], mjd.contact.dist[i], "dist")
      _assert_eq(d.contact.pos.numpy()[i], mjd.contact.pos[i], "pos")
      _assert_eq(d.contact.frame.numpy()[i].flatten(), mjd.contact.frame[i], "frame")

  # TODO(team): test primitive_narrowphase

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
    pairs = collision_driver.geom_pair(mjm)[0]
    self.assertEqual(pairs.shape[0], 2)

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
    _, pairid = collision_driver.geom_pair(mjm)
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
    _, pairid = collision_driver.geom_pair(mjm)
    self.assertTrue((pairid == 0).all())

    # generate contact
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm)
    mjwarp.forward(m, d)

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
    _, pairid = collision_driver.geom_pair(mjm)
    self.assertTrue((pairid == 0).all())

    # generate contact
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm)
    mjwarp.forward(m, d)

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
    _, pairid = collision_driver.geom_pair(mjm)
    self.assertTrue((pairid == 0).all())

    # generate contact
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm)
    mjwarp.forward(m, d)

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
    _, pairid = collision_driver.geom_pair(mjm)
    np.testing.assert_equal(pairid, np.array([-1, 0]))

    # generate contact
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm)
    mjwarp.forward(m, d)

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

  # TODO(team): test contact parameter mixing


if __name__ == "__main__":
  absltest.main()
