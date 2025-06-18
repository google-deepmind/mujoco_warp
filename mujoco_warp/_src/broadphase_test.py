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
from absl.testing import parameterized

import mujoco_warp as mjwarp

from . import collision_driver
from . import test_util
from .types import BroadphaseType


def broadphase_caller(m, d):
  if m.opt.broadphase == int(BroadphaseType.NXN):
    collision_driver.nxn_broadphase(m, d)
  else:
    collision_driver.sap_broadphase(m, d)


class BroadphaseTest(parameterized.TestCase):
  @parameterized.parameters(BroadphaseType.NXN, BroadphaseType.SAP_TILE, BroadphaseType.SAP_SEGMENTED)
  def test_broadphase(self, broadphase):
    """Tests collision broadphase algorithms."""

    _XML = """
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
    mjm, _, m, d0 = test_util.fixture(xml=_XML, keyframe=0)

    m.opt.broadphase = broadphase

    broadphase_caller(m, d0)
    np.testing.assert_allclose(d0.ncollision.numpy()[0], 0)

    # one world and one collision
    _, mjd1, _, d1 = test_util.fixture(xml=_XML, keyframe=1)
    broadphase_caller(m, d1)

    np.testing.assert_allclose(d1.ncollision.numpy()[0], 1)
    np.testing.assert_allclose(d1.collision_pair.numpy()[0][0], 0)
    np.testing.assert_allclose(d1.collision_pair.numpy()[0][1], 1)

    # one world and three collisions
    _, mjd2, _, d2 = test_util.fixture(xml=_XML, keyframe=2)
    broadphase_caller(m, d2)

    ncollision = d2.ncollision.numpy()[0]
    np.testing.assert_allclose(ncollision, 3)

    collision_pairs = [[0, 1], [0, 2], [1, 2]]
    for i in range(ncollision):
      self.assertTrue([d2.collision_pair.numpy()[i][0], d2.collision_pair.numpy()[i][1]] in collision_pairs)

    # two worlds and four collisions
    d3 = mjwarp.make_data(mjm, nworld=2, nconmax=512, njmax=512)
    d3.geom_xpos = wp.array(
      np.vstack([np.expand_dims(mjd1.geom_xpos, axis=0), np.expand_dims(mjd2.geom_xpos, axis=0)]),
      dtype=wp.vec3,
    )
    broadphase_caller(m, d3)

    ncollision = d3.ncollision.numpy()[0]
    np.testing.assert_allclose(ncollision, 4)

    collision_pairs = [[[0, 1]], [[0, 1], [0, 2], [1, 2]]]
    worldids = [0, 1, 1, 1]
    for i in range(ncollision):
      worldid = d3.collision_worldid.numpy()[i]
      self.assertTrue(worldid == worldids[i])
      self.assertTrue([d3.collision_pair.numpy()[i][0], d3.collision_pair.numpy()[i][1]] in collision_pairs[worldid])

    # one world and zero collisions: contype and conaffinity incompatibility
    mjm4, _, m4, d4 = test_util.fixture(xml=_XML, keyframe=1)
    mjm4.geom_contype[:3] = 0
    m4 = mjwarp.put_model(mjm4)

    broadphase_caller(m4, d4)
    np.testing.assert_allclose(d4.ncollision.numpy()[0], 0)

    # one world and one collision: geomtype ordering
    _, _, _, d5 = test_util.fixture(xml=_XML, keyframe=3)
    broadphase_caller(m, d5)
    np.testing.assert_allclose(d5.ncollision.numpy()[0], 1)
    np.testing.assert_allclose(d5.collision_pair.numpy()[0][0], 3)
    np.testing.assert_allclose(d5.collision_pair.numpy()[0][1], 2)

    # TODO(team): test margin
    # TODO(team): test DisableBit.FILTERPARENT


if __name__ == "__main__":
  wp.init()
  absltest.main()
