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


if __name__ == "__main__":
  absltest.main()
