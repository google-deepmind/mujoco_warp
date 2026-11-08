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
"""Canonical contact ordering, independent of insertion order."""

import dataclasses

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src.collision_driver import _sort_contacts


class ContactSortTest(parameterized.TestCase):
  @parameterized.parameters(1, 2)
  def test_permutation(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body pos="0 0 .09" euler=".03 .02 0">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=16,
    )
    qpos = np.tile(mjd.qpos, (nworld, 1))
    if nworld == 2:
      qpos[1, 0] += 0.5
      qpos[1, 2] -= 0.01
    d.qpos.assign(qpos)
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertGreater(count, 1)
    self.assertLess(count, d.naconmax)
    original = {
      f.name: getattr(d.contact, f.name).numpy().copy()
      for f in dataclasses.fields(d.contact)
      if getattr(d.contact, f.name).size
    }
    if nworld == 2:
      pos = original["pos"][:count]
      world = original["worldid"][:count]
      self.assertNotEqual(float(pos[world == 0, 0].mean()), float(pos[world == 1, 0].mean()))
    # Test the entire record permutation, including fields not used in sort keys.
    original["efc_address"][:count] = np.arange(count)[:, None]
    expected = None
    for seed in range(4):
      permutation = np.random.default_rng(seed).permutation(count)
      for name, array in original.items():
        shuffled = array.copy()
        shuffled[:count] = array[permutation]
        getattr(d.contact, name).assign(shuffled)
      _sort_contacts(m, d)
      result = {name: getattr(d.contact, name).numpy().copy() for name in original}
      if expected is None:
        expected = result
      for name, array in result.items():
        np.testing.assert_array_equal(array, expected[name], err_msg=name)
        np.testing.assert_array_equal(array[count:], original[name][count:], err_msg=name)
    # Warmed kernels and sorting scratch allocations must support graph capture.
    if wp.get_device().is_cuda:
      with wp.ScopedCapture() as capture:
        _sort_contacts(m, d)
      wp.capture_launch(capture.graph)
      for name, array in expected.items():
        np.testing.assert_array_equal(getattr(d.contact, name).numpy(), array, err_msg=name)
    d.nacon.zero_()
    _sort_contacts(m, d)
    for name, array in expected.items():
      np.testing.assert_array_equal(getattr(d.contact, name).numpy(), array, err_msg=name)


if __name__ == "__main__":
  wp.init()
  absltest.main()
