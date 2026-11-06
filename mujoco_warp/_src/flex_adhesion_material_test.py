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

"""Flex contact material initialization and persistent-slot regressions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data


class FlexAdhesionMaterialTest(parameterized.TestCase):
  @parameterized.product(nworld=(1, 2), batched=(False, True), deterministic=(0, 2))
  def test_plane_material_and_slot_reuse(self, nworld, batched, deterministic):
    """The flex writer owns adhesion even when a reused contact becomes nonadhesive."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1" adhesion="12"/>
          <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".1 .1 .1"
                    dim="2" mass="1" pos="0 0 .01" radius=".02">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=32,
      njmax=128,
      batch_sizes={"geom_adhesion": nworld if batched else 1},
    )
    m.opt.deterministic = deterministic
    qpos = d.qpos.numpy().copy()
    if nworld == 2:
      qpos[1, 2::3] += 0.002
    d.qpos.assign(qpos)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    # put_data seeds contacts from native; poison once, then reuse real writer outputs.
    d.contact.adhesion.fill_(wp.inf)
    retained_positions = []
    for strength in (12.0, 0.0, 3.0):
      material = np.full(m.geom_adhesion.shape, strength, dtype=np.float32)
      if nworld == 2 and batched:
        material[1] *= 0.5
      m.geom_adhesion.assign(material)
      mjw.collision(m, d)
      np.testing.assert_array_equal(d.overflow.numpy(), 0)
      ncon = int(d.nacon.numpy()[0])
      self.assertEqual(ncon, 4 * nworld)
      geoms = d.contact.geom.numpy().copy()[:ncon]
      worldids = d.contact.worldid.numpy().copy()[:ncon]
      adhesion = d.contact.adhesion.numpy().copy()[:ncon]
      positions = d.contact.pos.numpy().copy()[:ncon]
      self.assertTrue(np.all(geoms[:, 0] == 0))
      self.assertTrue(np.all(geoms[:, 1] == -1))
      for world in range(nworld):
        mjm.geom_adhesion[:] = material[world % len(material)]
        native = mujoco.MjData(mjm)
        native.qpos[:] = qpos[world]
        mujoco.mj_forward(mjm, native)
        mask = worldids == world
        self.assertEqual(int(mask.sum()), native.ncon)
        np.testing.assert_allclose(adhesion[mask], native.contact.adhesion, rtol=1e-5, atol=1e-5)
        retained_positions.append(positions[mask].copy())
      if nworld == 2:
        self.assertFalse(np.array_equal(retained_positions[-2], retained_positions[-1]))
        if batched and strength:
          self.assertFalse(np.array_equal(adhesion[worldids == 0], adhesion[worldids == 1]))

  @parameterized.parameters(1, 2)
  def test_flex_flex_material_is_zero(self, nworld):
    """Flex-only contacts must not inherit an unrelated geom's adhesive material."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="sphere" size=".1" pos="5 0 0" adhesion="17"/>
          <flexcomp name="a" type="grid" count="3 3 1" spacing=".1 .1 .1"
                    dim="2" mass="1" pos="0 0 0" radius=".02">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
          <flexcomp name="b" type="grid" count="3 3 1" spacing=".1 .1 .1"
                    dim="2" mass="1" pos="0 0 .01" radius=".02">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=128,
      njmax=1024,
    )
    qpos = d.qpos.numpy().copy()
    if nworld == 2:
      qpos[1, 2::3] += 0.002
    d.qpos.assign(qpos)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    d.contact.adhesion.fill_(wp.inf)
    mjw.collision(m, d)
    np.testing.assert_array_equal(d.overflow.numpy(), 0)
    ncon = int(d.nacon.numpy()[0])
    worldids = d.contact.worldid.numpy().copy()[:ncon]
    adhesion = d.contact.adhesion.numpy().copy()[:ncon]
    np.testing.assert_array_equal(d.contact.geom.numpy()[:ncon], -1)
    positions = d.contact.pos.numpy().copy()[:ncon]
    for world in range(nworld):
      native = mujoco.MjData(mjm)
      native.qpos[:] = qpos[world]
      mujoco.mj_forward(mjm, native)
      self.assertGreater(native.ncon, 0)
      np.testing.assert_array_equal(native.contact.geom, -1)
      np.testing.assert_array_equal(native.contact.adhesion, 0)
      self.assertTrue(np.any(worldids == world))
      np.testing.assert_allclose(adhesion[worldids == world], 0, rtol=1e-5, atol=1e-5)
    if nworld == 2:
      self.assertFalse(np.array_equal(positions[worldids == 0], positions[worldids == 1]))


if __name__ == "__main__":
  absltest.main()
