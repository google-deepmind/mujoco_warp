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

"""Tests for broadphase functions."""

import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp

from . import collision_driver
from . import test_util


class BroadphaseTest(absltest.TestCase):
  def test_sap_broadphase(self):
    """Tests sap_broadphase."""

    _SAP_MODEL = """
      <mujoco>
        <worldbody>
          <geom size="40 40 40" type="plane"/>   <!- (0) intersects with nothing -->
          <body pos="0 0 0.7">
            <freejoint/>
            <geom size="0.5 0.5 0.5" type="box"/> <!- (1) intersects with 2, 6, 7 -->
          </body>
          <body pos="0.1 0 0.7">
            <freejoint/>
            <geom size="0.5 0.5 0.5" type="box"/> <!- (2) intersects with 1, 6, 7 -->
          </body>
          <body pos="1.8 0 0.7">
            <freejoint/>
            <geom size="0.5 0.5 0.5" type="box"/> <!- (3) intersects with 4  -->
          </body>
          <body pos="1.6 0 0.7">
            <freejoint/>
            <geom size="0.5 0.5 0.5" type="box"/> <!- (4) intersects with 3 -->
          </body>
          <body pos="0 0 1.8">
            <freejoint/>
            <geom size="0.5 0.5 0.5" type="box"/> <!- (5) intersects with 7 -->
            <geom size="0.5 0.5 0.5" type="box" pos="0 0 -1"/> <!- (6) intersects with 2, 1, 7 -->
          </body>
          <body pos="0 0.5 1.2">
            <freejoint/>
            <geom size="0.5 0.5 0.5" type="box"/> <!- (7) intersects with 5, 6 -->
          </body>
        </worldbody>
      </mujoco>
    """

    collision_pair = [
      (0, 1),
      (0, 2),
      (0, 3),
      (0, 4),
      (0, 6),
      (1, 2),
      (1, 4),
      (1, 5),
      (1, 6),
      (1, 7),
      (2, 3),
      (2, 4),
      (2, 5),
      (2, 6),
      (2, 7),
      (3, 4),
      (4, 6),
      (5, 7),
      (6, 7),
    ]

    _, _, m, d = test_util.fixture(xml=_SAP_MODEL)

    mjwarp.sap_broadphase(m, d)

    ncollision = d.ncollision.numpy()[0]
    np.testing.assert_equal(ncollision, len(collision_pair), "ncollision")

    for i in range(ncollision):
      pair = d.collision_pair.numpy()[i]
      if pair[0] > pair[1]:
        pair_tuple = (int(pair[1]), int(pair[0]))
      else:
        pair_tuple = (int(pair[0]), int(pair[1]))

      np.testing.assert_equal(
        pair_tuple in collision_pair,
        True,
        f"geom pair {pair_tuple} not found in brute force results",
      )

    # TODO(team): test DisableBit.FILTERPARENT

    # TODO(team): test DisableBit.FILTERPARENT

  def test_nxn_broadphase(self):
    """Tests nxn_broadphase."""

    _NXN_MODEL = """
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="sphere" size="0.1"/>
          </body>
          <body>
            <freejoint/>
            <geom type="sphere" size="0.1"/>
          </body>
          <body>
            <freejoint/>
            <geom type="capsule" size="0.1 0.1"/>
          </body>
          <body>
            <freejoint/>
            <geom type="sphere" size="0.1"/>
          </body>
          <body>
            <freejoint/>
            <!-- self collision -->
            <geom type="sphere" size="0.1"/>
            <geom type="sphere" size="0.1"/>
            <!-- parent-child self collision -->
            <body>
              <geom type="sphere" size="0.1"/>
              <joint type="hinge"/>
            </body>
          </body>
        </worldbody>
        <keyframe>
          <key qpos='0 0 0 1 0 0 0
                    1 0 0 1 0 0 0
                    2 0 0 1 0 0 0
                    3 0 0 1 0 0 0
                    4 0 0 1 0 0 0
                    0'/>
          <key qpos='0 0 0 1 0 0 0
                    .05 0 0 1 0 0 0
                    2 0 0 1 0 0 0
                    3 0 0 1 0 0 0
                    4 0 0 1 0 0 0 
                    0'/>
          <key qpos='0 0 0 1 0 0 0
                    .01 0 0 1 0 0 0
                    .02 0 0 1 0 0 0
                    3 0 0 1 0 0 0
                    4 0 0 1 0 0 0 
                    0'/>
          <key qpos='0 0 0 1 0 0 0
                    1 0 0 1 0 0 0
                    2 0 0 1 0 0 0
                    2 0 0 1 0 0 0
                    4 0 0 1 0 0 0 
                    0'/>
        </keyframe>
      </mujoco>
    """
    # one world and zero collisions
    mjm, _, m, d0 = test_util.fixture(xml=_NXN_MODEL, keyframe=0)
    collision_driver.nxn_broadphase(m, d0)
    np.testing.assert_allclose(d0.ncollision.numpy()[0], 0)

    # one world and one collision
    _, mjd1, _, d1 = test_util.fixture(xml=_NXN_MODEL, keyframe=1)
    collision_driver.nxn_broadphase(m, d1)

    np.testing.assert_allclose(d1.ncollision.numpy()[0], 1)
    np.testing.assert_allclose(d1.collision_pair.numpy()[0][0], 0)
    np.testing.assert_allclose(d1.collision_pair.numpy()[0][1], 1)

    # one world and three collisions
    _, mjd2, _, d2 = test_util.fixture(xml=_NXN_MODEL, keyframe=2)
    collision_driver.nxn_broadphase(m, d2)
    np.testing.assert_allclose(d2.ncollision.numpy()[0], 3)
    np.testing.assert_allclose(d2.collision_pair.numpy()[0][0], 0)
    np.testing.assert_allclose(d2.collision_pair.numpy()[0][1], 1)
    np.testing.assert_allclose(d2.collision_pair.numpy()[1][0], 0)
    np.testing.assert_allclose(d2.collision_pair.numpy()[1][1], 2)
    np.testing.assert_allclose(d2.collision_pair.numpy()[2][0], 1)
    np.testing.assert_allclose(d2.collision_pair.numpy()[2][1], 2)

    # two worlds and four collisions
    d3 = mjwarp.make_data(mjm, nworld=2)
    d3.geom_xpos = wp.array(
      np.vstack(
        [np.expand_dims(mjd1.geom_xpos, axis=0), np.expand_dims(mjd2.geom_xpos, axis=0)]
      ),
      dtype=wp.vec3,
    )

    collision_driver.nxn_broadphase(m, d3)
    np.testing.assert_allclose(d3.ncollision.numpy()[0], 4)
    np.testing.assert_allclose(d3.collision_pair.numpy()[0][0], 0)
    np.testing.assert_allclose(d3.collision_pair.numpy()[0][1], 1)
    np.testing.assert_allclose(d3.collision_pair.numpy()[1][0], 0)
    np.testing.assert_allclose(d3.collision_pair.numpy()[1][1], 1)
    np.testing.assert_allclose(d3.collision_pair.numpy()[2][0], 0)
    np.testing.assert_allclose(d3.collision_pair.numpy()[2][1], 2)
    np.testing.assert_allclose(d3.collision_pair.numpy()[3][0], 1)
    np.testing.assert_allclose(d3.collision_pair.numpy()[3][1], 2)

    # one world and zero collisions: contype and conaffinity incompatibility
    _, _, m4, d4 = test_util.fixture(xml=_NXN_MODEL, keyframe=1)
    m4.geom_contype = wp.array(
      np.array(np.repeat(0, m.geom_type.shape)), dtype=wp.int32
    )
    m4.geom_conaffinity = wp.array(
      np.array(np.repeat(1, m.geom_type.shape)), dtype=wp.int32
    )

    collision_driver.nxn_broadphase(m4, d4)
    np.testing.assert_allclose(d4.ncollision.numpy()[0], 0)

    # one world and one collision: geomtype ordering
    _, _, _, d5 = test_util.fixture(xml=_NXN_MODEL, keyframe=3)
    collision_driver.nxn_broadphase(m, d5)
    np.testing.assert_allclose(d5.ncollision.numpy()[0], 1)
    np.testing.assert_allclose(d5.collision_pair.numpy()[0][0], 3)
    np.testing.assert_allclose(d5.collision_pair.numpy()[0][1], 2)

    # TODO(team): test margin
    # TODO(team): test DisableBit.FILTERPARENT


if __name__ == "__main__":
  wp.init()
  absltest.main()
