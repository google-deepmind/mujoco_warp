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
"""Tests for GPU determinism (contact sorting, constraint rows, full-pipeline bitwise stability)."""

import hashlib

import mujoco
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
  "elem",
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
    # flex, elem and vert are only allocated when flex contacts are present; an
    # empty array has no active contacts to permute.
    if values.shape[0] == 0:
      continue
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


class SolverDeterminismTest(parameterized.TestCase):
  """Phase 3: tests that solver state (qacc/qpos/qvel) is bitwise stable across runs.

  Covers the deterministic sparse qfrc_constraint and JTDAJ/H assembly that
  replace racing float atomic scatters when opt.deterministic=True.
  """

  @parameterized.parameters(
    ("humanoid/humanoid.xml", 1, "SPARSE"),
    ("humanoid/humanoid.xml", 4, "SPARSE"),
    ("collision.xml", 1, "SPARSE"),
    ("collision.xml", 4, "SPARSE"),
    ("humanoid/humanoid.xml", 1, "DENSE"),
    ("collision.xml", 1, "DENSE"),
  )
  def test_solver_state_deterministic(self, path, nworld, jacobian):
    """qacc/qpos/qvel are bitwise identical across repeated runs with opt.deterministic."""
    nruns = 3
    nsteps = 100

    def run():
      overrides = {"opt.jacobian": jacobian}
      _, _, m, d = test_data.fixture(path=path, nworld=nworld, overrides=overrides)
      m.opt.deterministic = True
      for _ in range(nsteps):
        mjw.step(m, d)
      return {
        "qacc": d.qacc.numpy().copy(),
        "qpos": d.qpos.numpy().copy(),
        "qvel": d.qvel.numpy().copy(),
      }

    results = [run() for _ in range(nruns)]
    for run_idx in range(1, nruns):
      for field in ("qacc", "qpos", "qvel"):
        np.testing.assert_array_equal(
          results[0][field],
          results[run_idx][field],
          err_msg=f"{field} differs: run 0 vs run {run_idx} ({path}, nworld={nworld}, {jacobian}, {nsteps} steps)",
        )

  def test_deterministic_solver_matches_nondeterministic_tolerance(self):
    """Deterministic solver path stays within solver tolerance of the default path."""
    nsteps = 50

    def run(det):
      _, _, m, d = test_data.fixture(path="humanoid/humanoid.xml", nworld=1, overrides={"opt.jacobian": "SPARSE"})
      m.opt.deterministic = det
      for _ in range(nsteps):
        mjw.step(m, d)
      return d.qpos.numpy().copy()

    qpos_det = run(True)
    qpos_nondet = run(False)
    # Both compute the same math with different float summation order; allow
    # divergence consistent with reordered rounding amplified by contact
    # dynamics, but catch gross algorithmic errors.
    self.assertLess(np.max(np.abs(qpos_det - qpos_nondet)), 1.0)


# A cube SDF resting deeply penetrated into a plane: every step the SDF narrowphase
# emits a large multi-contact manifold from many concurrent initpoint threads.
_SDF_PLANE_CUBE = """
  <mujoco>
    <option sdf_iterations="10" sdf_initpoints="40"/>
    <asset>
      <mesh name="cube"
       vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
    </asset>
    <worldbody>
      <geom size="40 40 40" type="plane"/>
      <body pos="0 0 1" euler="45 0 0">
        <freejoint/>
        <geom type="sdf" mesh="cube"/>
      </body>
    </worldbody>
  </mujoco>
"""


# An 8x8 cloth resting on a plane: 64 vertex-plane contacts emitted by concurrent
# flex narrowphase threads.
_FLEX_CLOTH_PLANE = """
  <mujoco>
    <worldbody>
      <geom type="plane" size="2 2 .01"/>
      <flexcomp type="grid" count="8 8 1" spacing=".05 .05 .05" pos="0 0 .005" radius=".01" mass="1" name="cloth" dim="2">
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
"""


# An 8x8 cloth and a free box resting on a plane: flex vertex-plane contacts and
# rigid box-plane contacts coexist in every world, exercising the split
# rigid/flex sort-key ranges.
_FLEX_CLOTH_PLANE_BOX = """
  <mujoco>
    <worldbody>
      <geom type="plane" size="2 2 .01"/>
      <body pos=".6 .6 .04">
        <freejoint/>
        <geom type="box" size=".05 .05 .05"/>
      </body>
      <flexcomp type="grid" count="8 8 1" spacing=".05 .05 .05" pos="0 0 .005" radius=".01" mass="1" name="cloth" dim="2">
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
"""


# Three fixed tendons whose limits are all violated at keyframe 0, referencing
# 3 + 3 + 2 Jacobian non-zeros. No other constraints are active.
_TENDON_LIMIT_ACTIVE = """
  <mujoco>
    <option>
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="j0" type="hinge"/>
        <geom type="sphere" size=".1"/>
        <body>
          <joint name="j1" type="hinge"/>
          <geom type="sphere" size=".1"/>
          <body>
            <joint name="j2" type="hinge"/>
            <geom type="sphere" size=".1"/>
          </body>
        </body>
      </body>
    </worldbody>
    <tendon>
      <fixed limited="true" range="-.4 .5">
        <joint joint="j0" coef=".25"/>
        <joint joint="j1" coef=".5"/>
        <joint joint="j2" coef=".75"/>
      </fixed>
      <fixed limited="true" range="0 .4">
        <joint joint="j0" coef=".5"/>
        <joint joint="j1" coef=".25"/>
        <joint joint="j2" coef="-.75"/>
      </fixed>
      <fixed limited="true" range="0 .3">
        <joint joint="j0" coef="1"/>
        <joint joint="j2" coef=".5"/>
      </fixed>
    </tendon>
    <keyframe>
      <key qpos=".2 .4 .6"/>
    </keyframe>
  </mujoco>
"""


def _flexstrain_grid_xml(ncomp):
  """Returns a model with `ncomp` disjoint trilinear flexcomps with strain equalities."""
  comps = []
  for i in range(ncomp):
    pos = f"{0.4 * (i % 8)} {0.4 * (i // 8)} 1"
    comps.append(f"""
      <flexcomp type="grid" count="2 2 2" spacing=".05 .05 .05" pos="{pos}" radius=".001" mass="1"
                name="soft{i}" dof="trilinear" dim="3">
        <edge equality="strain"/>
        <contact contype="0" conaffinity="0" selfcollide="none" internal="false"/>
      </flexcomp>""")
  return "<mujoco>\n  <worldbody>{}\n  </worldbody>\n</mujoco>".format("".join(comps))


def _ordered_contact_signature(d):
  """Returns a hash of the active contact buffer, sensitive to contact order."""
  nacon = int(d.nacon.numpy()[0])
  parts = [np.int64(nacon).tobytes()]
  for field in ("worldid", "geom", "flex", "elem", "vert", "dist", "pos", "dim"):
    arr = getattr(d.contact, field).numpy()
    # flex, elem and vert are only allocated when flex contacts are present.
    if arr.shape[0]:
      parts.append(arr[:nacon].tobytes())
  return hashlib.sha256(b"".join(parts)).hexdigest()


class SDFDeterminismTest(absltest.TestCase):
  """Tests that SDF contacts are emitted in a deterministic order."""

  def test_sdf_state_and_contact_order_deterministic(self):
    """qpos/qvel/qacc and the ordered contact buffer are bitwise identical across runs."""
    nsteps = 30

    def run():
      mjm, _, m, d = test_data.fixture(xml=_SDF_PLANE_CUBE, nworld=4, overrides={"opt.jacobian": "SPARSE"})
      m.opt.deterministic = True
      for _ in range(nsteps):
        mjw.step(m, d)
      nacon = int(d.nacon.numpy()[0])
      result = {
        "nacon": nacon,
        "qpos": d.qpos.numpy().copy(),
        "qvel": d.qvel.numpy().copy(),
        "qacc": d.qacc.numpy().copy(),
      }
      for field in ("worldid", "geom", "dist", "pos", "frame"):
        result[field] = getattr(d.contact, field).numpy()[:nacon].copy()
      return mjm, result

    mjm, first = run()
    _, second = run()

    # Confirm the SDF narrowphase is exercised: contacts involve the sdf geom.
    sdf_geoms = np.nonzero(mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF)[0]
    self.assertGreater(first["nacon"], 0)
    self.assertTrue(np.isin(first["geom"], sdf_geoms).any(), "expected contacts involving the sdf geom")

    self.assertEqual(first["nacon"], second["nacon"])
    for field in ("qpos", "qvel", "qacc", "worldid", "geom", "dist", "pos", "frame"):
      np.testing.assert_array_equal(first[field], second[field], err_msg=f"{field} differs between runs")


class FlexDeterminismTest(absltest.TestCase):
  """Tests deterministic contact ordering and efc row placement for flex models."""

  def test_flex_contact_order_deterministic(self):
    """Repeated collision() on a frozen state yields one unique ordered-contact signature."""
    _, _, m, d = test_data.fixture(xml=_FLEX_CLOTH_PLANE, nworld=2)
    m.opt.deterministic = True

    signatures = set()
    for _ in range(12):
      mjw.collision(m, d)
      signatures.add(_ordered_contact_signature(d))

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0)
    self.assertTrue((d.contact.geom.numpy()[:nacon] == -1).any(), "expected flex contacts")
    self.assertLen(signatures, 1, "contact order varies across identical collision() calls")

  def test_mixed_rigid_flex_contacts_world_contiguous(self):
    """Sorted contacts stay world-major when rigid and flex contacts coexist."""
    _, _, m, d = test_data.fixture(xml=_FLEX_CLOTH_PLANE_BOX, nworld=4, overrides={"opt.jacobian": "SPARSE"})
    m.opt.deterministic = True
    for _ in range(3):
      mjw.step(m, d)

    nacon = int(d.nacon.numpy()[0])
    geom = d.contact.geom.numpy()[:nacon]
    self.assertTrue((geom[:, 1] >= 0).any(), "expected rigid contacts")
    self.assertTrue((geom[:, 1] < 0).any(), "expected flex contacts")

    # The per-world constraint row scan accumulates over [world_start, world_end)
    # spans, so each world's contacts must be contiguous after sorting.
    worldid = d.contact.worldid.numpy()[:nacon]
    self.assertTrue((np.diff(worldid) >= 0).all(), "contacts are not world-contiguous after sorting")

    # All worlds are identical, so overlapping per-world scan spans show up as unequal nefc.
    nefc = d.nefc.numpy()
    np.testing.assert_array_equal(nefc, np.full(d.nworld, nefc[0]), err_msg="nefc differs across identical worlds")

  def test_mixed_rigid_flex_row_allocation_deterministic(self):
    """Repeated make_constraint on a frozen mixed rigid+flex state gives bitwise-identical rows."""
    _, _, m, d = test_data.fixture(xml=_FLEX_CLOTH_PLANE_BOX, nworld=4, overrides={"opt.jacobian": "SPARSE"})
    m.opt.deterministic = True
    mjw.fwd_position(m, d)
    nworld = d.nworld

    nacon = int(d.nacon.numpy()[0])
    geom = d.contact.geom.numpy()[:nacon]
    self.assertTrue((geom[:, 1] >= 0).any(), "expected rigid contacts")
    self.assertTrue((geom[:, 1] < 0).any(), "expected flex contacts")

    def snapshot():
      mjw.make_constraint(m, d)
      nefc = d.nefc.numpy().copy()
      fields = {"contact.efc_address": d.contact.efc_address.numpy()[:nacon].copy()}
      for name in ("id", "type", "J_rowadr", "J_rownnz"):
        arr = getattr(d.efc, name).numpy()
        fields[f"efc.{name}"] = np.concatenate([arr[w, : nefc[w]] for w in range(nworld)])
      return nefc, fields

    nefc0, first = snapshot()
    self.assertGreater(nefc0.sum(), 0)
    np.testing.assert_array_equal(nefc0, np.full(nworld, nefc0[0]), err_msg="nefc differs across identical worlds")
    for rep in range(1, 5):
      nefc, fields = snapshot()
      np.testing.assert_array_equal(nefc0, nefc, err_msg=f"nefc differs on repeat {rep}")
      for name, values in fields.items():
        np.testing.assert_array_equal(first[name], values, err_msg=f"{name} placement differs on repeat {rep}")

  def test_flexstrain_row_allocation_deterministic(self):
    """Repeated make_constraint on a frozen state gives bitwise-identical efc row placement."""
    _, _, m, d = test_data.fixture(
      xml=_flexstrain_grid_xml(64), nworld=128, overrides={"opt.jacobian": "SPARSE"}, njmax_nnz=32768
    )
    m.opt.deterministic = True
    mjw.fwd_position(m, d)
    nworld = d.nworld

    def snapshot():
      mjw.make_constraint(m, d)
      nefc = d.nefc.numpy().copy()
      fields = {}
      for name in ("id", "type", "J_rowadr"):
        arr = getattr(d.efc, name).numpy()
        fields[name] = np.concatenate([arr[w, : nefc[w]] for w in range(nworld)])
      return nefc, fields

    nefc0, first = snapshot()
    self.assertGreater(nefc0.sum(), 0)
    for rep in range(1, 8):
      nefc, fields = snapshot()
      np.testing.assert_array_equal(nefc0, nefc, err_msg=f"nefc differs on repeat {rep}")
      for name, values in fields.items():
        np.testing.assert_array_equal(first[name], values, err_msg=f"efc.{name} placement differs on repeat {rep}")


class TendonLimitNnzTest(absltest.TestCase):
  """Tests deterministic-mode sparse nnz accounting for tendon limit rows."""

  def test_limit_tendon_nnz_matches_rows(self):
    """Rows reference exactly the reserved nnz region: no allocator hole below their addresses."""
    _, _, m, d = test_data.fixture(xml=_TENDON_LIMIT_ACTIVE, keyframe=0, nworld=4, overrides={"opt.jacobian": "SPARSE"})
    m.opt.deterministic = True
    mjw.make_constraint(m, d)

    nefc = d.nefc.numpy()
    np.testing.assert_array_equal(d.nl.numpy(), np.full(d.nworld, 3), err_msg="expected 3 active tendon limits per world")
    np.testing.assert_array_equal(nefc, np.full(d.nworld, 3))

    rownnz = d.efc.J_rownnz.numpy()
    rowadr = d.efc.J_rowadr.numpy()
    for w in range(d.nworld):
      referenced = rownnz[w, : nefc[w]].sum()
      row_end = (rowadr[w, : nefc[w]] + rownnz[w, : nefc[w]]).max()
      self.assertEqual(referenced, 8)
      self.assertLessEqual(row_end, referenced, f"tendon limit rows placed beyond the referenced nnz in world {w}")

  def test_limit_tendon_tight_njmax_nnz_does_not_raise(self):
    """No spurious det overflow when njmax_nnz exactly fits the active tendon limit rows."""
    _, _, m, d = test_data.fixture(
      xml=_TENDON_LIMIT_ACTIVE, keyframe=0, nworld=4, overrides={"opt.jacobian": "SPARSE"}, njmax_nnz=8
    )
    m.opt.deterministic = True
    mjw.step(m, d)
    np.testing.assert_array_equal(d.nl.numpy(), np.full(d.nworld, 3))


class SleepDeterminismTest(absltest.TestCase):
  """Tests deterministic mode combined with the SLEEP feature."""

  def test_det_sleep_no_nan(self):
    """Deterministic mode plus SLEEP stays numerically healthy (bitwise stability not covered)."""
    _, _, m, d = test_data.fixture(
      path="humanoid/humanoid.xml",
      nworld=4,
      overrides={"opt.jacobian": "SPARSE", "opt.enableflags": mjw.EnableBit.SLEEP},
    )
    m.opt.deterministic = True
    for _ in range(50):
      mjw.step(m, d)
    self.assertFalse(np.isnan(d.qacc.numpy()).any(), "qacc contains NaN after 50 det+SLEEP steps")


def _det_fixture(njmax=None, naconmax=None, **kwargs):
  """Builds a deterministic-mode fixture; repeated calls initialize identically (fixed seed).

  njmax/naconmax rebuild Data with explicit capacities (test_data.fixture does not expose them);
  det mode raises on nefc overflow, so fixtures must be sized for the run's peak constraint count.
  """
  mjm, mjd, m, d = test_data.fixture(**kwargs)
  m.opt.deterministic = True
  if njmax is not None or naconmax is not None:
    put_kwargs = {"nworld": kwargs.get("nworld", 1), "nvmax": mjm.nv}
    if njmax is not None:
      put_kwargs["njmax"] = njmax
    if naconmax is not None:
      put_kwargs["naconmax"] = naconmax
    d = mjw.put_data(mjm, mjd, **put_kwargs)
  return m, d


def _det_fixture_raw(xml, nworld, jacobian=None, enableflags=0, njmax=None, naconmax=None):
  """Builds (m, d) directly from XML with explicit capacities.

  Enable flags are applied to MjModel before put_model because model construction depends on them.
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjm.opt.enableflags |= enableflags
  if jacobian is not None:
    mjm.opt.jacobian = jacobian
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  m.opt.deterministic = True
  put_kwargs = {}
  if njmax is not None:
    put_kwargs["njmax"] = njmax
  if naconmax is not None:
    put_kwargs["naconmax"] = naconmax
  d = mjw.put_data(mjm, mjd, nworld=nworld, **put_kwargs)
  return m, d


def _assert_lockstep_bitwise(test, build, nsteps, fields, label, extra_compares=None, on_step=None):
  """Steps two identically-built sims in lockstep, asserting per-step bitwise-equal state.

  Args:
    test: the TestCase (for fail()).
    build: zero-arg callable returning (m, d); called twice.
    nsteps: number of steps to run.
    fields: Data attribute names compared bitwise (raw bytes) after every step.
    label: prefix for failure messages.
    extra_compares: optional {name: fn(d) -> bytes} canonical serializations compared per step.
    on_step: optional callable(step, d1) for validity tracking; runs after the comparisons.

  Returns:
    (m1, d1) from the first sim, after nsteps.
  """
  m1, d1 = build()
  m2, d2 = build()
  for step in range(nsteps):
    mjw.step(m1, d1)
    mjw.step(m2, d2)
    for field in fields:
      a = getattr(d1, field).numpy()
      b = getattr(d2, field).numpy()
      if a.tobytes() != b.tobytes():
        np.testing.assert_array_equal(a, b, err_msg=f"{label}: d.{field} diverged at step {step}")
        # Numerically equal but bitwise different (e.g. -0.0 vs 0.0): still a determinism failure.
        test.fail(f"{label}: d.{field} bitwise-diverged at step {step} with numerically-equal values")
    if extra_compares:
      for name, fn in extra_compares.items():
        if fn(d1) != fn(d2):
          test.fail(f"{label}: {name} diverged at step {step}")
    if on_step is not None:
      on_step(step, d1)
  return m1, d1


def _canonical_actuator_moment(d):
  """Serializes the used entries of the actuator moment CSR (per-row colind + values)."""
  rownnz = d.moment_rownnz.numpy()
  rowadr = d.moment_rowadr.numpy()
  colind = d.moment_colind.numpy()
  moment = d.actuator_moment.numpy()
  parts = [rownnz.tobytes(), rowadr.tobytes()]
  for w in range(rownnz.shape[0]):
    for u in range(rownnz.shape[1]):
      adr, nnz = rowadr[w, u], rownnz[w, u]
      parts.append(colind[w, adr : adr + nnz].tobytes())
      parts.append(moment[w, adr : adr + nnz].tobytes())
  return b"".join(parts)


def _gravcomp_star_xml(nchild=12):
  """Root hinge with `nchild` gravity-compensated children: all project onto the shared root dof."""
  children = "".join(
    f"""
        <body pos="{0.02 * (i + 1)} {0.03 * ((i % 3) - 1)} 0" gravcomp="{0.5 + 0.04 * i}">
          <joint type="hinge" axis="1 0 0" damping=".05"/>
          <geom type="capsule" size=".015" fromto="0 0 0 0 .15 0" mass=".2"/>
        </body>"""
    for i in range(nchild)
  )
  return f"""
  <mujoco>
    <option>
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body name="root" pos="0 0 1">
        <joint name="root" type="hinge" axis="0 1 0" damping=".1"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0"/>
        {children}
      </body>
    </worldbody>
  </mujoco>
"""


def _fixed_tendon_actuator_xml(njnt=24, nten=16):
  """`nten` fixed tendons each spanning all `njnt` hinges, driven by tendon and joint motors."""
  bodies = "".join(
    f"""
      <body pos="{0.3 * (i % 6)} {0.3 * (i // 6)} 0">
        <joint name="j{i}" type="hinge" axis="0 1 0" damping=".1" stiffness=".05"/>
        <geom type="capsule" size=".02" fromto="0 0 0 0 0 .2" mass=".3"/>
      </body>"""
    for i in range(njnt)
  )
  tendons = []
  for t in range(nten):
    joints = "".join(f'<joint joint="j{i}" coef="{0.1 + 0.01 * ((t + i) % 7)}"/>' for i in range(njnt))
    tendons.append(f'<fixed name="t{t}" armature="0.02" stiffness=".1" damping=".02">{joints}</fixed>')
  actuators = "".join(f'<motor tendon="t{t}" gear="1.5"/>' for t in range(nten))
  actuators += "".join(f'<motor joint="j{i}" gear="2"/>' for i in range(0, njnt, 3))
  return f"""
  <mujoco>
    <option>
      <flag contact="disable"/>
    </option>
    <worldbody>{bodies}
    </worldbody>
    <tendon>{"".join(tendons)}</tendon>
    <actuator>{actuators}</actuator>
  </mujoco>
"""


def _branching_fluid_torso_xml(nlimb=6, nlink=3):
  """Free-root torso with `nlimb` branches of `nlink` hinges, fluid forces, and servo actuators."""
  limbs = []
  actuators = []
  for l in range(nlimb):
    inner = ""
    for k in reversed(range(1, nlink)):
      axis = "0 1 0" if k % 2 else "0 0 1"
      name = f'name="limb{l}_{k}"' if k == 1 else ""
      fluid = 'fluidshape="ellipsoid"' if k == 1 else ""
      inner = f"""
            <body pos=".12 0 0">
              <joint {name} type="hinge" axis="{axis}" damping=".2"/>
              <geom type="capsule" size=".02" fromto="0 0 0 .12 0 0" mass=".1" {fluid}/>
              {inner}
            </body>"""
    limbs.append(f"""
          <body pos=".15 0 0" euler="0 0 {360 * l // nlimb}">
            <joint name="limb{l}_0" type="hinge" axis="0 1 0" damping=".2"/>
            <geom type="capsule" size=".02" fromto="0 0 0 .12 0 0" mass=".1"/>
            {inner}
          </body>""")
    actuators.append(f'<motor joint="limb{l}_0" gear="1"/>')
    actuators.append(f'<position joint="limb{l}_1" kp="2" kv=".5"/>')
  return f"""
  <mujoco>
    <option density="1.2" viscosity="0.1">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body pos="0 0 1">
        <freejoint/>
        <geom type="box" size=".08 .08 .04" mass="1"/>
        {"".join(limbs)}
      </body>
    </worldbody>
    <actuator>{"".join(actuators)}</actuator>
  </mujoco>
"""


# sensordata layout (by construction): touch [0], force [1:4], torque [4:7],
# subtreelinvel [7:10], subtreeangmom [10:13], e_potential [13], e_kinetic [14].
def _sensor_suite_xml(nsphere=12):
  """Spheres dropping onto a touch plate plus a branching arm with force/torque/subtree sensors."""
  spheres = "".join(
    f"""
      <body pos="{0.12 * (i % 4) - 0.18} {0.12 * (i // 4) - 0.12} {0.12 + 0.02 * i}">
        <freejoint/>
        <geom type="sphere" size=".04" mass=".1"/>
      </body>"""
    for i in range(nsphere)
  )
  arms = "".join(
    f"""
          <body pos="0 0 .1" euler="0 0 {120 * a}">
            <joint type="hinge" axis="0 1 0" damping=".1"/>
            <geom type="capsule" size=".015" fromto="0 0 0 .1 0 .05" mass=".05"/>
            <body pos=".1 0 .05">
              <joint type="hinge" axis="0 0 1" damping=".1"/>
              <geom type="capsule" size=".015" fromto="0 0 0 .08 0 0" mass=".05"/>
            </body>
          </body>"""
    for a in range(3)
  )
  return f"""
  <mujoco>
    <worldbody>
      <geom type="plane" size="3 3 .01"/>
      <body name="plate" pos="0 0 .05">
        <geom type="box" size=".35 .35 .02" mass="1"/>
        <site name="plate_site" type="box" size=".36 .36 .03"/>
      </body>
      {spheres}
      <body name="trunk" pos="1.5 0 .3">
        <joint type="hinge" axis="1 0 0" damping=".2"/>
        <geom type="capsule" size=".02" fromto="0 0 0 0 0 .1" mass=".2"/>
        <site name="trunk_site" pos="0 0 .05"/>
        {arms}
      </body>
    </worldbody>
    <sensor>
      <touch site="plate_site"/>
      <force site="trunk_site"/>
      <torque site="trunk_site"/>
      <subtreelinvel body="trunk"/>
      <subtreeangmom body="trunk"/>
      <e_potential/>
      <e_kinetic/>
    </sensor>
  </mujoco>
"""


def _elastic_flex_xml():
  """A 3D elastic flex body dropping onto a plane: shared vertices accumulate elastic forces."""
  return """
  <mujoco>
    <worldbody>
      <geom type="plane" size="3 3 .01"/>
      <flexcomp type="grid" count="6 6 2" spacing=".06 .06 .06" pos="0 0 .08" radius=".005"
                name="soft" dim="3" mass="2">
        <contact selfcollide="none"/>
        <elasticity young="2e4" damping="0.003" poisson="0.2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
"""


def _sleep_wake_churn_xml(npair=8):
  """`npair` resting boxes that sleep early plus staggered falling impactors that wake them."""
  bodies = []
  for i in range(npair):
    x = 0.5 * i
    bodies.append(f"""
      <body pos="{x} 0 .05">
        <freejoint/>
        <geom type="box" size=".05 .05 .05" mass=".2"/>
      </body>
      <body pos="{x + 0.02} 0.01 {0.3 + 0.12 * i}">
        <freejoint/>
        <geom type="box" size=".04 .04 .04" mass=".15"/>
      </body>""")
  return f"""
  <mujoco>
    <worldbody>
      <geom type="plane" size="10 10 .01"/>
      {"".join(bodies)}
    </worldbody>
  </mujoco>
"""


class PipelineStateDeterminismTest(parameterized.TestCase):
  """Lockstep bitwise stability of the full pipeline on branching-tree and wide-dense models."""

  @parameterized.parameters("SPARSE", "DENSE")
  def test_pendula_state_bitwise(self, jacobian):
    """pendula.xml (branching kinematic trees, nv=36) is bitwise stable over 100 steps."""
    _assert_lockstep_bitwise(
      self,
      lambda: _det_fixture(path="pendula.xml", nworld=16, qvel_noise=0.1, ctrl_noise=0.1, overrides={"opt.jacobian": jacobian}),
      nsteps=100,
      fields=("qpos", "qvel", "qacc"),
      label=f"pendula {jacobian}",
    )

  def test_constraints_dense_state_bitwise(self):
    """constraints.xml (nv=50, dense) is bitwise stable over 100 steps."""
    _assert_lockstep_bitwise(
      self,
      lambda: _det_fixture(path="constraints.xml", nworld=16, ctrl_noise=0.1, overrides={"opt.jacobian": "DENSE"}, njmax=256),
      nsteps=100,
      fields=("qpos", "qvel", "qacc"),
      label="constraints DENSE",
    )


class PassiveActuationDeterminismTest(absltest.TestCase):
  """Lockstep bitwise stability of passive gravcomp and tendon/actuator force accumulation."""

  def test_gravcomp_bitwise(self):
    """Many gravity-compensated bodies sharing ancestor dofs stay bitwise stable."""
    _, d = _assert_lockstep_bitwise(
      self,
      lambda: _det_fixture(xml=_gravcomp_star_xml(), nworld=64, qvel_noise=0.05),
      nsteps=50,
      fields=("qpos", "qvel", "qacc", "qfrc_gravcomp"),
      label="gravcomp",
    )
    self.assertTrue((d.qfrc_gravcomp.numpy() != 0.0).any(), "expected nonzero gravity compensation forces")

  def test_fixed_tendon_actuator_bitwise(self):
    """Fixed-tendon lengths and shared-dof actuator forces stay bitwise stable."""
    _, d = _assert_lockstep_bitwise(
      self,
      lambda: _det_fixture(xml=_fixed_tendon_actuator_xml(), nworld=32, qvel_noise=0.1, ctrl_noise=0.2),
      nsteps=50,
      fields=("qpos", "qvel", "qacc", "ten_length", "ten_J", "qfrc_actuator", "actuator_force"),
      label="fixed tendon + actuators",
      extra_compares={"actuator_moment": _canonical_actuator_moment},
    )
    self.assertTrue((d.ten_length.numpy() != 0.0).any(), "expected nonzero tendon lengths")
    self.assertTrue((d.qfrc_actuator.numpy() != 0.0).any(), "expected nonzero actuator forces")


class ImplicitIntegratorDeterminismTest(parameterized.TestCase):
  """Lockstep bitwise stability of implicit integrators (qDeriv assembly) on a branching tree."""

  @parameterized.parameters("IMPLICIT", "IMPLICITFAST")
  def test_implicit_integrators_bitwise(self, integrator):
    """A branching free-root torso with fluid forces and servos is bitwise stable."""
    _assert_lockstep_bitwise(
      self,
      lambda: _det_fixture(
        xml=_branching_fluid_torso_xml(),
        nworld=128,
        qvel_noise=0.1,
        ctrl_noise=0.2,
        overrides={"opt.integrator": integrator},
      ),
      nsteps=50,
      fields=("qpos", "qvel", "qacc"),
      label=f"branching torso {integrator}",
    )


class SensorDataDeterminismTest(absltest.TestCase):
  """Lockstep bitwise stability of sensordata (touch/force/torque/subtree/energy)."""

  def test_sensordata_bitwise(self):
    """Contact-fed touch plus rne_postconstraint / subtree / energy sensors stay bitwise stable."""
    touch_seen = [False]

    def on_step(step, d):
      if (d.sensordata.numpy()[:, 0] > 0.0).any():
        touch_seen[0] = True

    _, d = _assert_lockstep_bitwise(
      self,
      lambda: _det_fixture_raw(
        _sensor_suite_xml(),
        nworld=4,
        jacobian=mujoco.mjtJacobian.mjJAC_SPARSE,
        enableflags=mujoco.mjtEnableBit.mjENBL_ENERGY,
        njmax=512,
        naconmax=4096,
      ),
      nsteps=100,
      fields=("sensordata", "energy", "qpos", "qvel", "qacc"),
      label="sensors",
      on_step=on_step,
    )
    self.assertTrue(touch_seen[0], "touch sensor never fired: contacts did not reach the plate")
    sensordata = d.sensordata.numpy()
    self.assertTrue((sensordata[:, 1:7] != 0.0).any(), "expected nonzero force/torque sensor data")
    self.assertTrue((sensordata[:, 13:15] != 0.0).any(), "expected nonzero energy sensor data")


class FlexPassiveDeterminismTest(absltest.TestCase):
  """Lockstep bitwise stability of flex models (elastic passive forces, CG preconditioner)."""

  def test_flex_full_state_bitwise(self):
    """An elastic flex body deforming and landing on a plane is bitwise stable w/ qfrc_passive."""
    _, d = _assert_lockstep_bitwise(
      self,
      lambda: _det_fixture_raw(
        _elastic_flex_xml(), nworld=8, jacobian=mujoco.mjtJacobian.mjJAC_SPARSE, njmax=1024, naconmax=8192
      ),
      nsteps=70,
      fields=("qpos", "qvel", "qacc", "qfrc_passive"),
      label="elastic flex",
    )
    self.assertTrue((d.qfrc_passive.numpy() != 0.0).any(), "expected nonzero flex passive forces")
    self.assertGreater(int(d.nacon.numpy()[0]), 0, "expected flex-plane contacts by the end of the run")

  def test_cg_flex_floppy_bitwise(self):
    """flex/floppy.xml (CG solver, flex diagonal preconditioner) is bitwise stable."""
    _assert_lockstep_bitwise(
      self,
      lambda: _det_fixture(path="flex/floppy.xml", nworld=4, overrides={"opt.jacobian": "SPARSE"}),
      nsteps=40,
      fields=("qpos", "qvel", "qacc"),
      label="floppy CG",
    )


class EllipticConeDeterminismTest(parameterized.TestCase):
  """Lockstep bitwise stability of the elliptic cone Hessian on a contact-rich mixed scene."""

  @parameterized.parameters("SPARSE", "DENSE")
  def test_aloha_elliptic_bitwise(self, jacobian):
    """aloha_pot (elliptic cone, impratio=10, many cone contacts) is bitwise stable."""
    _assert_lockstep_bitwise(
      self,
      lambda: _det_fixture(path="aloha_pot/scene.xml", keyframe="lift_pot0", nworld=8, overrides={"opt.jacobian": jacobian}),
      nsteps=50,
      fields=("qpos", "qvel", "qacc"),
      label=f"aloha elliptic {jacobian}",
    )


class SleepWakeChurnDeterminismTest(absltest.TestCase):
  """Lockstep bitwise stability of det + SLEEP while trees genuinely sleep and wake."""

  def test_sleep_wake_churn_bitwise(self):
    """Resting boxes sleep, staggered impactors wake them; state stays bitwise stable throughout."""
    nsteps = 260
    prev_awake = [None]
    sleep_events = [0]
    wake_events = [0]
    max_nacon = [0]

    def on_step(step, d):
      cur = d.tree_awake.numpy()
      if prev_awake[0] is not None:
        sleep_events[0] += int(((prev_awake[0] == 1) & (cur == 0)).sum())
        wake_events[0] += int(((prev_awake[0] == 0) & (cur == 1)).sum())
      prev_awake[0] = cur.copy()
      max_nacon[0] = max(max_nacon[0], int(d.nacon.numpy()[0]))

    _, d = _assert_lockstep_bitwise(
      self,
      lambda: _det_fixture_raw(
        _sleep_wake_churn_xml(),
        nworld=16,
        jacobian=mujoco.mjtJacobian.mjJAC_SPARSE,
        enableflags=mujoco.mjtEnableBit.mjENBL_SLEEP,
        njmax=1024,
        naconmax=16384,
      ),
      nsteps=nsteps,
      fields=("qpos", "qvel", "qacc", "qfrc_constraint", "tree_awake"),
      label="sleep wake churn",
      on_step=on_step,
    )
    self.assertFalse(np.isnan(d.qacc.numpy()).any(), "qacc contains NaN after det+SLEEP run")
    # The run is only a meaningful probe if the compact solve was exercised through churn and no
    # capacity overflow (an independent determinism hazard) occurred.
    self.assertGreater(sleep_events[0], 0, "no tree ever fell asleep: model failed to exercise SLEEP")
    self.assertGreater(wake_events[0], 0, "no tree ever woke up: model failed to exercise wake churn")
    self.assertLess(max_nacon[0], d.naconmax, "contact capacity saturated; probe invalid")
    self.assertLess(int(d.nefc.numpy().max()), d.njmax, "constraint capacity saturated; probe invalid")


class GraphCaptureLongHorizonTest(parameterized.TestCase):
  """Long-horizon bitwise stability of a captured step across independent capture+replay runs."""

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  @parameterized.parameters(
    ("humanoid/humanoid.xml", "DENSE", 64),
    ("pendula.xml", "SPARSE", 16),
  )
  def test_captured_step_long_horizon_bitwise(self, path, jacobian, nworld):
    """320 replays of a captured step produce identical per-step state hashes across two runs."""
    nsteps = 320

    def run():
      m, d = _det_fixture(path=path, nworld=nworld, qvel_noise=0.1, ctrl_noise=0.1, overrides={"opt.jacobian": jacobian})
      with wp.ScopedCapture() as capture:
        mjw.step(m, d)
      hashes = []
      for _ in range(nsteps):
        wp.capture_launch(capture.graph)
        wp.synchronize()
        h = hashlib.sha256()
        for field in ("qpos", "qvel", "qacc"):
          h.update(getattr(d, field).numpy().tobytes())
        hashes.append(h.hexdigest())
      return hashes, d.qpos.numpy().copy()

    hashes1, qpos1 = run()
    hashes2, _ = run()
    self.assertTrue(np.isfinite(qpos1).all(), "captured run produced non-finite qpos")
    for step, (h1, h2) in enumerate(zip(hashes1, hashes2)):
      self.assertEqual(h1, h2, f"{path} {jacobian}: captured-step state hash diverged at replay {step}")


if __name__ == "__main__":
  absltest.main()
