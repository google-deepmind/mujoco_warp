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
"""Tests for GPU determinism (contact sorting)."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import collision_driver

_NSTEPS = 10
_CONTACT_FIELDS = (
  "dist",
  "pos",
  "frame",
  "includemargin",
  "friction",
  "solref",
  "solreffriction",
  "solimp",
  "dim",
  "geom",
  "flex",
  "vert",
  "efc_address",
  "worldid",
  "type",
  "geomcollisionid",
)


def _run_and_collect_contacts(path, nworld, nsteps, deterministic):
  """Run simulation and return contact geom arrays from last step."""
  _, _, m, d = test_data.fixture(path=path, nworld=nworld)
  m.opt.deterministic = deterministic
  for _ in range(nsteps):
    mjw.step(m, d)
  nacon = d.nacon.numpy()[0]
  return {
    "nacon": nacon,
    "geom": d.contact.geom.numpy()[:nacon].copy(),
    "dist": d.contact.dist.numpy()[:nacon].copy(),
    "pos": d.contact.pos.numpy()[:nacon].copy(),
    "frame": d.contact.frame.numpy()[:nacon].copy(),
    "dim": d.contact.dim.numpy()[:nacon].copy(),
    "worldid": d.contact.worldid.numpy()[:nacon].copy(),
    "geomcollisionid": d.contact.geomcollisionid.numpy()[:nacon].copy(),
  }


def _copy_contact_fields(d):
  """Return copies of every contact array."""
  return {field: getattr(d.contact, field).numpy().copy() for field in _CONTACT_FIELDS}


def _write_contact_fields(d, contact_fields):
  """Write full contact arrays back to device memory."""
  for field, values in contact_fields.items():
    arr = getattr(d.contact, field)
    wp.copy(arr, wp.array(values, dtype=arr.dtype, device=arr.device))


def _permute_active_contacts(contact_fields, nacon, perm):
  """Return a copy with the active contacts permuted by `perm`."""
  permuted = {field: values.copy() for field, values in contact_fields.items()}
  for field, values in permuted.items():
    values[:nacon] = values[perm]
  return permuted


def _sorted_contact_order(contact_fields, nacon):
  """Return stable sorted indices for the active contacts."""
  geom = contact_fields["geom"]
  worldid = contact_fields["worldid"]
  geomcollisionid = contact_fields["geomcollisionid"]
  return sorted(
    range(nacon),
    key=lambda idx: (
      int(worldid[idx]),
      int(geom[idx, 0]),
      int(geom[idx, 1]),
      int(geomcollisionid[idx]),
    ),
  )


class ContactSortDeterminismTest(parameterized.TestCase):
  """Tests that contact sorting produces deterministic contact ordering."""

  @parameterized.parameters(
    ("collision.xml", 1),
    ("collision.xml", 4),
    ("humanoid/humanoid.xml", 1),
    ("humanoid/humanoid.xml", 4),
  )
  def test_contact_ordering_deterministic(self, path, nworld):
    """Contacts are bitwise identical across multiple runs."""
    nruns = 3
    results = [_run_and_collect_contacts(path, nworld, _NSTEPS, True) for _ in range(nruns)]

    # Verify contacts were generated.
    self.assertGreater(results[0]["nacon"], 0, f"No contacts for {path}")

    for run in range(1, nruns):
      self.assertEqual(results[0]["nacon"], results[run]["nacon"])
      np.testing.assert_array_equal(
        results[0]["geom"],
        results[run]["geom"],
        err_msg=f"Contact geom ordering differs: run 0 vs run {run}",
      )

  @parameterized.parameters(
    ("collision.xml", 1),
    ("humanoid/humanoid.xml", 1),
  )
  def test_contact_fields_deterministic(self, path, nworld):
    """All contact fields are bitwise identical across runs."""
    nruns = 3
    results = [_run_and_collect_contacts(path, nworld, _NSTEPS, True) for _ in range(nruns)]

    self.assertGreater(results[0]["nacon"], 0)

    for run in range(1, nruns):
      self.assertEqual(results[0]["nacon"], results[run]["nacon"])
      for field in ("dist", "pos", "frame", "geom", "dim", "worldid", "geomcollisionid"):
        np.testing.assert_array_equal(
          results[0][field],
          results[run][field],
          err_msg=f"{field} differs: run 0 vs run {run}",
        )

  def test_contacts_sorted_by_geom(self):
    """Contacts are sorted by (worldid, geom0, geom1) after deterministic step."""
    result = _run_and_collect_contacts("collision.xml", 1, _NSTEPS, True)

    nacon = result["nacon"]
    self.assertGreater(nacon, 1)

    geom = result["geom"]
    worldid = result["worldid"]

    # Verify sorted: (worldid, geom0, geom1) is non-decreasing.
    for i in range(1, nacon):
      key_prev = (worldid[i - 1], geom[i - 1, 0], geom[i - 1, 1])
      key_curr = (worldid[i], geom[i, 0], geom[i, 1])
      self.assertLessEqual(
        key_prev,
        key_curr,
        f"Contacts not sorted at index {i}: {key_prev} > {key_curr}",
      )

  def test_sort_contacts_reorders_mixed_contacts(self):
    """Sorting restores deterministic contact order after contacts are mixed."""
    _, _, m, d = test_data.fixture(path="collision.xml", nworld=4)
    m.opt.deterministic = False

    mjw.forward(m, d)

    nacon = d.nacon.numpy()[0]
    self.assertGreaterEqual(nacon, 5)

    original = _copy_contact_fields(d)
    perm = np.concatenate((np.arange(1, nacon, 2), np.arange(0, nacon, 2)))
    self.assertFalse(np.array_equal(perm, np.arange(nacon)))

    mixed = _permute_active_contacts(original, nacon, perm)
    _write_contact_fields(d, mixed)

    expected_order = _sorted_contact_order(mixed, nacon)
    expected = _permute_active_contacts(mixed, nacon, expected_order)

    collision_driver._sort_contacts(m, d)

    actual = _copy_contact_fields(d)
    self.assertEqual(d.nacon.numpy()[0], nacon)

    for field in _CONTACT_FIELDS:
      np.testing.assert_array_equal(
        actual[field][:nacon],
        expected[field][:nacon],
        err_msg=f"{field} was not permuted into deterministic order",
      )

  def test_deterministic_flag_default_false(self):
    """The deterministic flag defaults to False."""
    _, _, m, _ = test_data.fixture(path="collision.xml")
    self.assertFalse(m.opt.deterministic)


if __name__ == "__main__":
  absltest.main()
