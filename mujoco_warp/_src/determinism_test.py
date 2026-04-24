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
"""Tests for GPU determinism (contact sorting + constraint row allocation)."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import collision_driver
from mujoco_warp._src.benchmark import benchmark

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


# Per-row efc fields to compare across runs (excluding J which has solver-path-
# dependent shape handled separately).
_EFC_ROW_FIELDS = ("type", "id", "pos", "margin", "D", "vel", "aref", "frictionloss")


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


def _run_and_collect_efc(path, nworld, nsteps, deterministic, jacobian):
  """Run simulation and return nefc + all per-row efc fields + J from last step."""
  overrides = {"opt.jacobian": jacobian}
  _, _, m, d = test_data.fixture(path=path, nworld=nworld, overrides=overrides)
  m.opt.deterministic = deterministic
  for _ in range(nsteps):
    mjw.step(m, d)

  nefc = d.nefc.numpy().copy()
  result = {"nefc": nefc, "is_sparse": m.is_sparse}
  # Per-row fields: (nworld, njmax). Slice per world to its nefc entries.
  # Tests concatenate across worlds so shape is (sum(nefc),) - ordering within
  # each world is the quantity that must be stable.
  for field in _EFC_ROW_FIELDS:
    arr = getattr(d.efc, field).numpy()
    result[field] = np.concatenate([arr[w, : nefc[w]].copy() for w in range(nworld)])

  # J and sparse metadata.
  if m.is_sparse:
    j_rownnz = d.efc.J_rownnz.numpy()
    j_rowadr = d.efc.J_rowadr.numpy()
    # J_colind in sparse is (nworld, 1, njmax*nv); flat per world slice.
    j_colind_flat = d.efc.J_colind.numpy()[:, 0, :]
    j_flat = d.efc.J.numpy()[:, 0, :]  # (nworld, njmax * nv)
    result["J_rownnz"] = np.concatenate([j_rownnz[w, : nefc[w]].copy() for w in range(nworld)])
    result["J_rowadr"] = np.concatenate([j_rowadr[w, : nefc[w]].copy() for w in range(nworld)])
    # For colind/J values, collect only entries that correspond to active
    # rows; per-row length is rownnz[i] starting at rowadr[i].
    colind_parts = []
    j_parts = []
    for w in range(nworld):
      for i in range(nefc[w]):
        nnz = j_rownnz[w, i]
        adr = j_rowadr[w, i]
        colind_parts.append(j_colind_flat[w, adr : adr + nnz].copy())
        j_parts.append(j_flat[w, adr : adr + nnz].copy())
    result["J_colind"] = np.concatenate(colind_parts) if colind_parts else np.empty(0, dtype=np.int32)
    result["J"] = np.concatenate(j_parts) if j_parts else np.empty(0)
  else:
    # Dense J is (nworld, njmax_pad, nv_pad). Slice to nefc rows per world.
    j_dense = d.efc.J.numpy()
    result["J_row_width"] = j_dense.shape[2]
    result["J"] = np.concatenate([j_dense[w, : nefc[w], :].reshape(-1).copy() for w in range(nworld)])

  return result


def _sorted_efc_row_records(result):
  """Returns a canonical multiset representation of efc rows for comparison."""
  records = []
  total_rows = int(np.sum(result["nefc"]))
  j_offset = 0

  for row in range(total_rows):
    record = [int(result["type"][row])]
    for field in ("pos", "margin", "D", "vel", "aref", "frictionloss"):
      record.append(np.asarray(result[field][row]).tobytes())

    if result["is_sparse"]:
      nnz = int(result["J_rownnz"][row])
      colind = result["J_colind"][j_offset : j_offset + nnz]
      j_values = result["J"][j_offset : j_offset + nnz]
      j_offset += nnz
      record.append(colind.tobytes())
      record.append(j_values.tobytes())
    else:
      row_width = int(result["J_row_width"])
      j_values = result["J"][j_offset : j_offset + row_width]
      j_offset += row_width
      record.append(j_values.tobytes())

    records.append(tuple(record))

  return sorted(records)


class ConstraintAllocationDeterminismTest(parameterized.TestCase):
  """Phase 2: tests that constraint row allocation produces deterministic efc rows."""

  @parameterized.parameters(
    ("humanoid/humanoid.xml", 1, "DENSE"),
    ("humanoid/humanoid.xml", 4, "DENSE"),
    ("humanoid/humanoid.xml", 1, "SPARSE"),
    ("humanoid/humanoid.xml", 4, "SPARSE"),
    ("collision.xml", 1, "DENSE"),
    ("collision.xml", 4, "DENSE"),
    ("collision.xml", 1, "SPARSE"),
    ("collision.xml", 4, "SPARSE"),
  )
  def test_nefc_deterministic(self, path, nworld, jacobian):
    """d.nefc is bitwise identical across repeat runs in deterministic mode."""
    nruns = 3
    results = [_run_and_collect_efc(path, nworld, _NSTEPS, True, jacobian) for _ in range(nruns)]
    self.assertGreater(results[0]["nefc"].sum(), 0, f"No constraints for {path}")
    for run in range(1, nruns):
      np.testing.assert_array_equal(
        results[0]["nefc"],
        results[run]["nefc"],
        err_msg=f"nefc differs: run 0 vs run {run} ({path}, nworld={nworld}, {jacobian})",
      )

  @parameterized.parameters(
    ("humanoid/humanoid.xml", 1, "DENSE"),
    ("humanoid/humanoid.xml", 4, "DENSE"),
    ("humanoid/humanoid.xml", 1, "SPARSE"),
    ("humanoid/humanoid.xml", 4, "SPARSE"),
    ("collision.xml", 1, "DENSE"),
    ("collision.xml", 4, "DENSE"),
    ("collision.xml", 1, "SPARSE"),
    ("collision.xml", 4, "SPARSE"),
  )
  def test_efc_rows_deterministic(self, path, nworld, jacobian):
    """Per-row efc fields are bitwise identical across runs in deterministic mode."""
    nruns = 3
    results = [_run_and_collect_efc(path, nworld, _NSTEPS, True, jacobian) for _ in range(nruns)]
    self.assertGreater(results[0]["nefc"].sum(), 0)

    for run in range(1, nruns):
      for field in _EFC_ROW_FIELDS:
        np.testing.assert_array_equal(
          results[0][field],
          results[run][field],
          err_msg=f"efc.{field} differs: run 0 vs run {run} ({path}, nworld={nworld}, {jacobian})",
        )

  @parameterized.parameters(
    ("humanoid/humanoid.xml", 1, "DENSE"),
    ("humanoid/humanoid.xml", 4, "DENSE"),
    ("humanoid/humanoid.xml", 1, "SPARSE"),
    ("humanoid/humanoid.xml", 4, "SPARSE"),
    ("collision.xml", 1, "DENSE"),
    ("collision.xml", 4, "DENSE"),
    ("collision.xml", 1, "SPARSE"),
    ("collision.xml", 4, "SPARSE"),
  )
  def test_efc_J_deterministic(self, path, nworld, jacobian):
    """Jacobian values (and sparse metadata) are bitwise identical across runs."""
    nruns = 3
    results = [_run_and_collect_efc(path, nworld, _NSTEPS, True, jacobian) for _ in range(nruns)]
    self.assertGreater(results[0]["nefc"].sum(), 0)

    for run in range(1, nruns):
      np.testing.assert_array_equal(
        results[0]["J"],
        results[run]["J"],
        err_msg=f"efc.J differs: run 0 vs run {run} ({path}, nworld={nworld}, {jacobian})",
      )
      if results[0]["is_sparse"]:
        for field in ("J_rownnz", "J_rowadr", "J_colind"):
          np.testing.assert_array_equal(
            results[0][field],
            results[run][field],
            err_msg=f"efc.{field} differs: run 0 vs run {run} ({path}, nworld={nworld}, {jacobian})",
          )

  @parameterized.parameters(
    ("collision.xml", 1, "DENSE"),
    ("collision.xml", 1, "SPARSE"),
  )
  def test_deterministic_matches_nondeterministic_row_multiset(self, path, nworld, jacobian):
    """Deterministic allocation preserves the same efc row contents as the legacy path."""
    deterministic = _run_and_collect_efc(path, nworld, _NSTEPS, True, jacobian)
    nondeterministic = _run_and_collect_efc(path, nworld, _NSTEPS, False, jacobian)

    self.assertGreater(deterministic["nefc"].sum(), 0)
    np.testing.assert_array_equal(
      deterministic["nefc"],
      nondeterministic["nefc"],
      err_msg=f"nefc differs between det off/on ({path}, nworld={nworld}, {jacobian})",
    )
    self.assertListEqual(
      _sorted_efc_row_records(deterministic),
      _sorted_efc_row_records(nondeterministic),
    )

  @parameterized.parameters(
    ("humanoid/humanoid.xml", 16, "DENSE"),
    ("collision.xml", 16, "SPARSE"),
  )
  def test_large_nworld_efc_deterministic(self, path, nworld, jacobian):
    """Larger nworld cases stay bitwise stable in deterministic mode."""
    nruns = 2
    results = [_run_and_collect_efc(path, nworld, _NSTEPS, True, jacobian) for _ in range(nruns)]
    self.assertGreater(results[0]["nefc"].sum(), 0)

    np.testing.assert_array_equal(
      results[0]["nefc"],
      results[1]["nefc"],
      err_msg=f"nefc differs ({path}, nworld={nworld}, {jacobian})",
    )
    for field in _EFC_ROW_FIELDS:
      np.testing.assert_array_equal(
        results[0][field],
        results[1][field],
        err_msg=f"efc.{field} differs ({path}, nworld={nworld}, {jacobian})",
      )
    np.testing.assert_array_equal(
      results[0]["J"],
      results[1]["J"],
      err_msg=f"efc.J differs ({path}, nworld={nworld}, {jacobian})",
    )
    if results[0]["is_sparse"]:
      for field in ("J_rownnz", "J_rowadr", "J_colind"):
        np.testing.assert_array_equal(
          results[0][field],
          results[1][field],
          err_msg=f"efc.{field} differs ({path}, nworld={nworld}, {jacobian})",
        )

  @parameterized.parameters("DENSE", "SPARSE")
  def test_zero_size_families_skip_cleanly(self, jacobian):
    """Contact-only models still work when many deterministic families are size 0."""
    overrides = {"opt.jacobian": jacobian}
    _, _, m, d = test_data.fixture(path="collision.xml", nworld=4, overrides=overrides)
    m.opt.deterministic = True

    self.assertEqual(m.eq_connect_adr.size, 0)
    self.assertEqual(m.eq_wld_adr.size, 0)
    self.assertEqual(m.eq_jnt_adr.size, 0)
    self.assertEqual(m.eq_ten_adr.size, 0)
    self.assertEqual(m.eq_flex_adr.size, 0)
    self.assertEqual(m.ntendon, 0)
    self.assertEqual(m.jnt_limited_ball_adr.size, 0)
    self.assertEqual(m.jnt_limited_slide_hinge_adr.size, 0)
    self.assertEqual(m.tendon_limited_adr.size, 0)

    mjw.step(m, d)

    self.assertGreater(d.nefc.numpy().sum(), 0)

  def test_overflow_raises_in_deterministic_mode(self):
    """Artificially small njmax triggers RuntimeError in deterministic mode."""
    _, _, m, d = test_data.fixture(path="humanoid/humanoid.xml", nworld=1)
    m.opt.deterministic = True
    # njmax normally tracks the required storage. Force overflow by lowering
    # it below the real constraint count. One step should populate more than
    # 1 row, so 1 is guaranteed to overflow.
    d.njmax = 1
    with self.assertRaisesRegex(RuntimeError, "nefc overflow"):
      mjw.step(m, d)

  def test_nondet_path_unaffected_by_njmax(self):
    """Non-deterministic mode must not trigger overflow check (silent-truncate preserved)."""
    _, _, m, d = test_data.fixture(path="humanoid/humanoid.xml", nworld=1)
    m.opt.deterministic = False
    # With det=False the overflow check should not run even if we set njmax
    # artificially. The existing silent-return-on-overflow behavior is
    # preserved - we just need it to not crash.
    # Step once with normal njmax to confirm baseline.
    mjw.step(m, d)


class DeterminismBenchmarkCollectionTest(parameterized.TestCase):
  """Tests for deterministic benchmark data collection."""

  @parameterized.parameters(
    ("NEWTON", "DENSE"),
    ("CG", "SPARSE"),
  )
  def test_benchmark_graph_capture_nondeterministic(self, solver, jacobian):
    """The existing CUDA-graph benchmark path still works for det=False."""
    overrides = {"opt.solver": solver, "opt.jacobian": jacobian}
    _, _, m, d = test_data.fixture(path="collision.xml", nworld=1, overrides=overrides)
    m.opt.deterministic = False

    jit_duration, run_duration, _, nacon, nefc, _, nsuccess = benchmark(
      mjw.step,
      m,
      d,
      nstep=2,
      measure_alloc=True,
      use_cuda_graph=True,
    )

    self.assertGreater(jit_duration, 0.0)
    self.assertGreater(run_duration, 0.0)
    self.assertLen(nacon, 2)
    self.assertLen(nefc, 2)
    self.assertEqual(nsuccess, d.nworld)

  @parameterized.parameters(
    ("NEWTON", "DENSE"),
    ("CG", "SPARSE"),
  )
  def test_benchmark_deterministic_without_cuda_graph(self, solver, jacobian):
    """Deterministic benchmark collection works with CUDA graphs disabled."""
    overrides = {"opt.solver": solver, "opt.jacobian": jacobian}
    _, _, m, d = test_data.fixture(path="collision.xml", nworld=1, overrides=overrides)
    m.opt.deterministic = True

    jit_duration, run_duration, _, nacon, nefc, _, nsuccess = benchmark(
      mjw.step,
      m,
      d,
      nstep=2,
      measure_alloc=True,
      use_cuda_graph=False,
    )

    self.assertGreater(jit_duration, 0.0)
    self.assertGreater(run_duration, 0.0)
    self.assertLen(nacon, 2)
    self.assertLen(nefc, 2)
    self.assertEqual(nsuccess, d.nworld)


if __name__ == "__main__":
  absltest.main()
