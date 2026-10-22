# Copyright 2026 Google LLC
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

"""Regression for a tilted box above a flat heightfield selecting an upper vertex."""

import mujoco
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data


class HfieldSupportTest(parameterized.TestCase):
  @parameterized.parameters(0.0, -0.5, 0.25)
  def test_tilted_box_normal(self, shift):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <asset><hfield name="flat" nrow="161" ncol="161" size="2 2 .1 .1"/></asset>
        <worldbody>
          <geom type="hfield" hfield="flat"/>
          <body pos=".11924547865911737 1.652866488033965 .004782060812356274"
                quat=".7446686495903571 .0006327301958846996 .007022425944823349 -.6673970988127098">
            <freejoint/><geom type="box" size=".085 .03 .005" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
    """
    )
    mjd.qpos[0] += shift
    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)
    d.qpos.assign(mjd.qpos[None])
    d.contact.frame.fill_(np.inf)
    mjw.kinematics(m, d)
    mjw.collision(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertGreater(count, 0)
    self.assertGreater(mjd.ncon, 0)
    self.assertTrue(np.all(mjd.contact.frame[: mjd.ncon, 2] >= 0))
    self.assertTrue(np.all(d.contact.frame.numpy()[:count, 0, 2] >= 0))
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=1e-7, rtol=1e-6)

  @parameterized.parameters(False, True)
  def test_box_at_step_side(self, flat):
    heights = np.zeros((13, 13), dtype=np.float32)
    if not flat:
      heights[:4] = 1.0
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <asset><hfield name="step" nrow="13" ncol="13" size=".15 .15 .163 .1"/></asset>
        <worldbody>
          <geom type="hfield" hfield="step"/>
          <body pos="-.01252218357228116 .00528042099085879 .085727085183132615"
                quat=".81019302509732682 .021597807194852645 -.015807462496793441 -.58555180892678416">
            <freejoint/><geom type="box" size=".085 .03 .005" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"hfield_data": heights.ravel()},
    )
    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)
    d.contact.frame.fill_(np.inf)
    d.contact.dist.fill_(np.inf)
    mjw.kinematics(m, d)
    mjw.collision(m, d)
    count = int(d.nacon.numpy()[0])
    if flat:
      self.assertEqual(mjd.ncon, 0)
      self.assertEqual(count, 0)
    else:
      self.assertGreater(mjd.ncon, 0)
      self.assertGreater(count, 0)
      # The box is above ground, contacting the ramped step side. A vertical
      # projection onto ground invents a downward 79mm penetration.
      self.assertTrue(np.all(mjd.contact.dist[: mjd.ncon] > -0.03))
      self.assertTrue(np.all(d.contact.dist.numpy()[:count] > -0.03))
      self.assertTrue(np.all(d.contact.frame.numpy()[:count, 0, 1] > 0.5))
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=1e-7, rtol=1e-6)

  @parameterized.product(nworld=(1, 2), height=(0.0, 0.4))
  def test_flat_patch_matches_plane(self, nworld, height):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <asset><hfield name="flat" nrow="381" ncol="321" size="4 4.75 .6 .1"/></asset>
        <worldbody>
          <geom type="hfield" hfield="flat" pos="0 .55 0"/>
          <body pos=".09384317763333773 1.6838817419527428 .011732103623421555"
                quat=".6706291570814201 -.08753127717973837 -.19728099925919612 -.7097006527535682">
            <freejoint/><geom type="box" size=".085 .03 .005" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      overrides={"hfield_data": np.full(381 * 321, height, dtype=np.float32)},
    )
    # Independent analytic plane oracle, not the CPU prism manifold.
    plane = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><geom type="plane" size="0 0 .1"/>
      <body><freejoint/><geom type="box" size=".085 .03 .005" mass="1"/></body>
      </worldbody></mujoco>
    """)
    plane.geom_pos[0, 2] = float(np.float32(height)) * 0.6
    mujoco.mj_setConst(plane, mujoco.MjData(plane))
    poses = np.tile(mjd.qpos, (nworld, 1))
    poses[:, 2] += float(np.float32(height)) * 0.6
    if nworld == 2:
      poses[1, 0] += 0.1
      poses[1, 2] += 0.012
    d.qpos.assign(poses)
    d.contact.frame.fill_(np.inf)
    d.contact.dist.fill_(np.inf)
    d.contact.pos.fill_(np.inf)
    mjw.kinematics(m, d)
    mjw.collision(m, d)
    count = int(d.nacon.numpy()[0])
    worlds = d.contact.worldid.numpy()[:count]
    for world in range(nworld):
      pd = mujoco.MjData(plane)
      pd.qpos[:] = poses[world]
      mujoco.mj_kinematics(plane, pd)
      mujoco.mj_collision(plane, pd)
      ids = np.flatnonzero(worlds == world)
      self.assertEqual(len(ids), pd.ncon)
      actual = np.column_stack((d.contact.dist.numpy()[ids], d.contact.pos.numpy()[ids]))
      expected = np.column_stack((pd.contact.dist[: pd.ncon], pd.contact.pos[: pd.ncon]))
      actual = actual[np.argsort(actual[:, 0])]
      expected = expected[np.argsort(expected[:, 0])]
      np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
      np.testing.assert_allclose(d.contact.frame.numpy()[ids, 0], np.tile([0, 0, 1], (len(ids), 1)), atol=1e-6)
    np.testing.assert_allclose(d.qpos.numpy(), poses, atol=1e-7)
    self.assertFalse(np.any(d.overflow.numpy()))


if __name__ == "__main__":
  absltest.main()
