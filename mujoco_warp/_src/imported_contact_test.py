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

"""Public native-contact import and collision-free constraint assembly regressions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import types


def _import_contacts(nworld):
  mjm, mjd, m, _ = test_data.fixture(
    xml="""
    <mujoco>
      <option jacobian="sparse" cone="elliptic" solver="Newton" tolerance="1e-10"/>
      <worldbody>
        <geom type="plane" size="2 2 .1"/>
        <body pos="0 0 .09"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
      </worldbody>
    </mujoco>
    """,
    nconmax=8,
    njmax=32,
  )
  mjd.qvel[:] = np.arange(1, mjm.nv + 1) / 128
  mujoco.mj_forward(mjm, mjd)
  # A fresh public import, not contacts produced by a preceding Warp forward/collision.
  d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=8, njmax=32)
  return mjm, mjd, m, d


class ImportedContactTest(parameterized.TestCase):
  @parameterized.parameters(1, 2)
  def test_put_data_contact_ids(self, nworld):
    """Native contacts have a neutral Warp-only collision ID, including spare capacity."""
    _, mjd, _, d = _import_contacts(nworld)
    self.assertGreater(mjd.ncon, 1)
    self.assertEqual(int(d.nacon.numpy()[0]), nworld * mjd.ncon)
    # This is a constructor contract, not an assumption about wp.empty's initial bits.
    # Initcheck is needed to reliably detect the omission when an allocator returns zeros.
    np.testing.assert_array_equal(d.contact.geomcollisionid.numpy(), 0)
    np.testing.assert_array_equal(d.contact.worldid.numpy()[: nworld * mjd.ncon], np.repeat(np.arange(nworld), mjd.ncon))
    self.assertFalse(np.any(d.contact.type.numpy()[: nworld * mjd.ncon] & int(types.ContactType.SENSOR)))

  @parameterized.product(
    nworld=(1, 2),
    mode=("standalone", "external"),
    deterministic=(mjw.DeterminismType.CONSTRAINT, mjw.DeterminismType.CONSTRAINT | mjw.DeterminismType.CONTACTS),
  )
  def test_imported_contact_constraints(self, nworld, mode, deterministic):
    """The first consumer can sort an untouched public import, without collision replay."""
    mjm, mjd, m, d = _import_contacts(nworld)
    self.assertGreater(mjd.ncon, 1)
    m.opt.deterministic = deterministic
    qvel = d.qvel.numpy().copy()
    qvel *= np.arange(1, nworld + 1)[:, None]
    d.qvel.assign(qvel)
    expected_force, expected_qacc = [], []
    for world in range(nworld):
      mjd.qvel[:] = qvel[world]
      mujoco.mj_forward(mjm, mjd)
      expected_force.append(mjd.qfrc_constraint.copy())
      expected_qacc.append(mjd.qacc.copy())
    expected_force, expected_qacc = np.array(expected_force), np.array(expected_qacc)
    expected_nefc = mjd.nefc

    # Do not write any Contact field, especially geomcollisionid, before its first consumer.
    d.nefc.fill_(-1)
    d.efc.J.fill_(wp.inf)
    d.efc.pos.fill_(wp.inf)
    if mode == "standalone":
      mjw.make_constraint(m, d)
    else:
      m.opt.run_collision_detection = False
      d.qacc.fill_(wp.inf)
      d.qfrc_constraint.fill_(wp.inf)
      mjw.forward(m, d)
      np.testing.assert_allclose(d.qacc.numpy().copy(), expected_qacc, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(d.qfrc_constraint.numpy().copy(), expected_force, rtol=1e-5, atol=1e-5)
      if nworld == 2:
        self.assertFalse(np.array_equal(expected_qacc[0], expected_qacc[1]))

    np.testing.assert_array_equal(d.nefc.numpy(), expected_nefc)
    np.testing.assert_array_equal(d.overflow.numpy(), 0)
    np.testing.assert_array_equal(d.contact.geomcollisionid.numpy()[: nworld * mjd.ncon], 0)
    addresses = d.contact.efc_address.numpy().copy()
    ids = d.efc.id.numpy().copy()
    worlds = d.contact.worldid.numpy().copy()
    for contact in range(nworld * mjd.ncon):
      active = addresses[contact][addresses[contact] >= 0]
      self.assertLen(active, 3)
      np.testing.assert_array_equal(ids[worlds[contact], active], contact)
    for world in range(nworld):
      self.assertTrue(np.all(np.isfinite(d.efc.pos.numpy()[world, :expected_nefc])))
      rownnz = d.efc.J_rownnz.numpy().copy()[world, :expected_nefc]
      self.assertTrue(np.all(np.isfinite(d.efc.J.numpy()[world, 0, : int(rownnz.sum())])))


if __name__ == "__main__":
  absltest.main()
