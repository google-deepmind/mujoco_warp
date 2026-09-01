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

"""Tests for io functions."""

import dataclasses
import tempfile
import warnings
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp
from mujoco_warp import ConeType
from mujoco_warp import IntegratorType
from mujoco_warp import test_data
from mujoco_warp._src import io
from mujoco_warp._src import types
from mujoco_warp._src import warp_util
from mujoco_warp._src.io import put_model


def _allocate_worlds(
  candidates: list[tuple[int, float]],
  nworld: int,
) -> list[int]:
  """Assign worlds contiguously by prm fraction (largest remainder method).

  Returns list of length nworld with candidate indices (not mesh IDs).
  """
  total_prm = sum(prm for _, prm in candidates)
  if total_prm <= 0:
    # uniform if all prm are zero
    total_prm = len(candidates)
    candidates = [(mid, 1.0) for mid, _ in candidates]
  # largest remainder method for exact allocation
  quotas = [(prm / total_prm) * nworld for _, prm in candidates]
  floors = [int(q) for q in quotas]
  remainders = [(quotas[i] - floors[i], i) for i in range(len(candidates))]
  allocated = sum(floors)
  # distribute remaining slots by largest fractional remainder
  remainders.sort(key=lambda x: -x[0])
  for j in range(nworld - allocated):
    floors[remainders[j][1]] += 1
  assignment = []
  for idx, count in enumerate(floors):
    assignment.extend([idx] * count)
  return assignment


def _populate_dependent_fields(m, spec, padded_model, dataid_table, nworld, geom_variants, body_variants):
  """Compile each unique variant and set per-world dependent fields.

  Updates: geom_size, geom_aabb, geom_rbound, geom_pos, body_mass,
  body_subtreemass, body_inertia, body_invweight0, body_ipos, body_iquat.

  Saves and restores spec state so the spec is not left mutated.
  """
  # Identify unique dataid rows (variant configurations)
  unique_rows = {}
  for w in range(nworld):
    key = tuple(dataid_table[w])
    if key not in unique_rows:
      unique_rows[key] = w  # first world with this config

  if len(unique_rows) <= 1:
    return  # nothing to do if all worlds are the same

  # Save spec state so we can restore after compilation (index-based to
  # handle unnamed geoms)
  spec_geoms = list(spec.geoms)
  saved_geom_state = {}
  for idx, g in enumerate(spec_geoms):
    saved_geom_state[idx] = (
      g.meshname,
      g.contype,
      g.conaffinity,
      g.mass,
    )

  # Build index map: geom_id (in padded_model) -> spec geom index
  geom_id_to_spec_idx = {}
  for idx, g in enumerate(spec_geoms):
    if g.name:
      gid = mujoco.mj_name2id(padded_model, mujoco.mjtObj.mjOBJ_GEOM, g.name)
      if gid >= 0:
        geom_id_to_spec_idx[gid] = idx

  # For unnamed geoms, match by body and order within body
  body_geom_order = {}  # body_name -> list of (geom_id, spec_idx)
  for idx, g in enumerate(spec_geoms):
    if not g.name and g.type == mujoco.mjtGeom.mjGEOM_MESH:
      # find parent body name via spec
      for b in spec.bodies:
        if any(bg is g for bg in b.geoms):
          if b.name not in body_geom_order:
            body_geom_order[b.name] = []
          body_geom_order[b.name].append(idx)
          break

  # Match unnamed geoms by position in body
  for body_name, spec_indices in body_geom_order.items():
    body_id = mujoco.mj_name2id(padded_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    unnamed_model_geoms = [
      gid
      for gid in range(padded_model.ngeom)
      if padded_model.geom_bodyid[gid] == body_id
      and padded_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
      and mujoco.mj_id2name(padded_model, mujoco.mjtObj.mjOBJ_GEOM, gid) == ""
    ]
    for k, spec_idx in enumerate(spec_indices):
      if k < len(unnamed_model_geoms):
        geom_id_to_spec_idx[unnamed_model_geoms[k]] = spec_idx

  # Compile each unique variant to get reference field values
  compiled_variants = {}  # key -> compiled MjModel
  for key, first_world in unique_rows.items():
    # Apply this variant's mesh assignments to the spec
    for geom_id, candidates in geom_variants.items():
      mesh_id = dataid_table[first_world, geom_id]
      if mesh_id >= 0 and geom_id in geom_id_to_spec_idx:
        mesh_name = mujoco.mj_id2name(padded_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        geom = spec_geoms[geom_id_to_spec_idx[geom_id]]
        geom.meshname = mesh_name

    for body_name, variants in body_variants.items():
      body = next(b for b in spec.bodies if b.name == body_name)
      mesh_geoms = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]
      # get model geom ids for ALL mesh geoms in this body (named + unnamed)
      body_id = mujoco.mj_name2id(padded_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
      mesh_geom_ids = [
        gid
        for gid in range(padded_model.ngeom)
        if padded_model.geom_bodyid[gid] == body_id and padded_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
      ]
      # find variant by matching dataid
      for var_meshes, _ in variants:
        if len(mesh_geom_ids) > 0 and len(var_meshes) > 0:
          if dataid_table[first_world, mesh_geom_ids[0]] == var_meshes[0]:
            for k, geom in enumerate(mesh_geoms):
              if k < len(var_meshes):
                mesh_name = mujoco.mj_id2name(padded_model, mujoco.mjtObj.mjOBJ_MESH, var_meshes[k])
                geom.meshname = mesh_name
                geom.contype = 1
                geom.conaffinity = 1
              else:
                geom.contype = 0
                geom.conaffinity = 0
                geom.mass = 0
            break

    compiled_variants[key] = spec.compile()

  # Restore spec state
  for idx, g in enumerate(spec_geoms):
    if idx in saved_geom_state:
      meshname, contype, conaffinity, mass = saved_geom_state[idx]
      g.meshname = meshname
      g.contype = contype
      g.conaffinity = conaffinity
      g.mass = mass

  # Now build per-world arrays from compiled variants
  ngeom = padded_model.ngeom
  nbody = padded_model.nbody

  geom_size = np.zeros((nworld, ngeom, 3), dtype=np.float32)
  geom_rbound = np.zeros((nworld, ngeom), dtype=np.float32)
  geom_aabb = np.zeros((nworld, ngeom, 2, 3), dtype=np.float32)
  geom_pos = np.zeros((nworld, ngeom, 3), dtype=np.float32)
  body_mass = np.zeros((nworld, nbody), dtype=np.float32)
  body_subtreemass = np.zeros((nworld, nbody), dtype=np.float32)
  body_inertia = np.zeros((nworld, nbody, 3), dtype=np.float32)
  body_invweight0 = np.zeros((nworld, nbody, 2), dtype=np.float32)
  body_ipos = np.zeros((nworld, nbody, 3), dtype=np.float32)
  body_iquat = np.zeros((nworld, nbody, 4), dtype=np.float32)

  for w in range(nworld):
    key = tuple(dataid_table[w])
    ref = compiled_variants[key]
    geom_size[w] = ref.geom_size
    geom_rbound[w] = ref.geom_rbound
    geom_aabb[w] = ref.geom_aabb.reshape(ngeom, 2, 3)
    geom_pos[w] = ref.geom_pos
    body_mass[w] = ref.body_mass
    body_subtreemass[w] = ref.body_subtreemass
    body_inertia[w] = ref.body_inertia
    body_invweight0[w] = ref.body_invweight0
    body_ipos[w] = ref.body_ipos
    body_iquat[w] = ref.body_iquat

  m.geom_size = wp.array(geom_size, dtype=wp.vec3)
  m.geom_rbound = wp.array(geom_rbound, dtype=float)
  m.geom_aabb = wp.array(geom_aabb, dtype=wp.vec3)
  m.geom_pos = wp.array(geom_pos, dtype=wp.vec3)
  m.body_mass = wp.array(body_mass, dtype=float)
  m.body_subtreemass = wp.array(body_subtreemass, dtype=float)
  m.body_inertia = wp.array(body_inertia, dtype=wp.vec3)
  m.body_invweight0 = wp.array(body_invweight0, dtype=wp.vec2)
  m.body_ipos = wp.array(body_ipos, dtype=wp.vec3)
  m.body_iquat = wp.array(body_iquat, dtype=wp.quat)


def per_world_mesh(spec: mujoco.MjSpec, nworld: int):
  """Per-world mesh randomization from custom/tuple annotations.

  Returns:
    Tuple of (Model, padded MjModel).
  """
  spec = spec.copy()
  model = spec.compile()

  # no-op if no tuples
  if model.ntuple == 0:
    return put_model(model), model

  body_names = {b.name for b in spec.bodies if b.name}

  # --- Pad bodies to max variant geom count ---
  padded = False
  for tuple_id in range(model.ntuple):
    tuple_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TUPLE, tuple_id)
    if tuple_name not in body_names:
      continue
    start = model.tuple_adr[tuple_id]
    size = model.tuple_size[tuple_id]

    # find max mesh geoms across all variants
    max_geoms = 0
    max_variant_meshes = []
    for i in range(size):
      if model.tuple_objtype[start + i] != mujoco.mjtObj.mjOBJ_TUPLE:
        continue
      var_tuple_id = model.tuple_objid[start + i]
      var_start = model.tuple_adr[var_tuple_id]
      var_size = model.tuple_size[var_tuple_id]
      var_meshes = []
      for j in range(var_size):
        if model.tuple_objtype[var_start + j] == mujoco.mjtObj.mjOBJ_MESH:
          var_meshes.append(model.tuple_objid[var_start + j])
      if len(var_meshes) > max_geoms:
        max_geoms = len(var_meshes)
        max_variant_meshes = var_meshes

    # count current mesh geoms in body
    body = next(b for b in spec.bodies if b.name == tuple_name)
    current_mesh_geoms = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]

    # pad if needed
    if max_geoms > len(current_mesh_geoms):
      for k in range(len(current_mesh_geoms), max_geoms):
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, max_variant_meshes[k])
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh_name
        geom.contype = 0
        geom.conaffinity = 0
      padded = True

  # rebuild model from padded spec
  if padded:
    model = spec.compile()

  m = put_model(model)

  geom_names = {g.name for g in spec.geoms}
  body_names = {b.name for b in spec.bodies if b.name}
  # resolve ambiguity: names matching both geom and body are treated as body-level only
  ambiguous = geom_names & body_names
  geom_names = geom_names - ambiguous
  ngeom = model.ngeom

  # Start from base dataid tiled for all worlds
  base_dataid = model.geom_dataid.copy()
  dataid_table = np.tile(base_dataid, (nworld, 1))  # (nworld, ngeom)

  # Track which geoms have been randomized so we can compile variants
  geom_variants = {}  # geom_id -> list of (mesh_id, prm)
  body_variants = {}  # body_name -> list of (variant_meshes, prm)

  # --- Geom-level tuples ---
  for tuple_id in range(model.ntuple):
    tuple_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TUPLE, tuple_id)
    if tuple_name not in geom_names:
      continue
    start = model.tuple_adr[tuple_id]
    size = model.tuple_size[tuple_id]
    # skip body-level tuples (those containing tuple-type elements)
    if any(model.tuple_objtype[start + i] == mujoco.mjtObj.mjOBJ_TUPLE for i in range(size)):
      continue

    candidates = []
    for i in range(size):
      if model.tuple_objtype[start + i] != mujoco.mjtObj.mjOBJ_MESH:
        continue
      mesh_id = model.tuple_objid[start + i]
      prm = model.tuple_objprm[start + i]
      candidates.append((mesh_id, prm))

    if not candidates:
      continue

    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tuple_name)
    geom_variants[geom_id] = candidates
    assignment = _allocate_worlds(candidates, nworld)
    for w in range(nworld):
      dataid_table[w, geom_id] = candidates[assignment[w]][0]

  # --- Body-level tuples ---
  for tuple_id in range(model.ntuple):
    tuple_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TUPLE, tuple_id)
    if tuple_name not in body_names:
      continue
    start = model.tuple_adr[tuple_id]
    size = model.tuple_size[tuple_id]

    # collect variant info
    variants = []
    for i in range(size):
      if model.tuple_objtype[start + i] != mujoco.mjtObj.mjOBJ_TUPLE:
        continue
      var_tuple_id = model.tuple_objid[start + i]
      prm = model.tuple_objprm[start + i]

      # read variant tuple's mesh list
      var_start = model.tuple_adr[var_tuple_id]
      var_size = model.tuple_size[var_tuple_id]
      var_meshes = []
      for j in range(var_size):
        if model.tuple_objtype[var_start + j] == mujoco.mjtObj.mjOBJ_MESH:
          var_meshes.append(model.tuple_objid[var_start + j])
      variants.append((var_meshes, prm))

    if not variants:
      continue

    # find all mesh geoms in this body (including unnamed padded geoms)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tuple_name)
    mesh_geom_ids = [
      gid for gid in range(ngeom) if model.geom_bodyid[gid] == body_id and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
    ]

    body_variants[tuple_name] = variants

    # allocate worlds
    prm_candidates = [(0, prm) for _, prm in variants]  # dummy mesh_id
    assignment = _allocate_worlds(prm_candidates, nworld)

    for w in range(nworld):
      variant_idx = assignment[w]
      var_meshes = variants[variant_idx][0]
      for k, geom_id in enumerate(mesh_geom_ids):
        if k < len(var_meshes):
          dataid_table[w, geom_id] = var_meshes[k]
        else:
          dataid_table[w, geom_id] = -1  # disable unused geom slot

  # no-op if no randomization found
  if not geom_variants and not body_variants:
    return m, model

  m.geom_dataid = wp.array(dataid_table, dtype=int)

  # Populate dependent per-world fields from variant compilations
  _populate_dependent_fields(m, spec, model, dataid_table, nworld, geom_variants, body_variants)

  return m, model


def _assert_eq(a, b, name):
  tol = 5e-4
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


# NOTE: modify io_jax_test _IO_TEST_MODELS if changed here.
_IO_TEST_MODELS = (
  "pendula.xml",
  "collision_sdf/tactile.xml",
  "flex/floppy.xml",
  "actuation/tendon_force_limit.xml",
  "hfield/hfield.xml",
)


_MESH_RANDOMIZE_XML = """
<mujoco>
  <asset>
    <mesh name="cube_small" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    <mesh name="cube_large" vertex="0 0 0  2 0 0  0 2 0  0 0 2"/>
    <mesh name="object_A_0" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    <mesh name="object_A_1" vertex="1 0 0  2 0 0  1 1 0  1 0 1"/>
    <mesh name="object_A_2" vertex="0 1 0  1 1 0  0 2 0  0 1 1"/>
    <mesh name="object_B_0" vertex="0 0 0  3 0 0  0 3 0  0 0 3"/>
    <mesh name="object_B_1" vertex="3 0 0  6 0 0  3 3 0  3 0 3"/>
  </asset>
  <worldbody>
    <body pos="0 0 2">
      <freejoint/>
      <geom name="cube" type="mesh" mesh="cube_small"/>
    </body>
    <body name="object" pos="0 0 1">
      <freejoint/>
      <geom name="object_col_0" type="mesh" mesh="object_B_0"/>
      <geom name="object_col_1" type="mesh" mesh="object_B_1"/>
    </body>
  </worldbody>
  <custom>
    <tuple name="cube">
      <element objtype="mesh" objname="cube_small" prm="0.5"/>
      <element objtype="mesh" objname="cube_large" prm="0.5"/>
    </tuple>
    <tuple name="object_A">
      <element objtype="mesh" objname="object_A_0" prm="0"/>
      <element objtype="mesh" objname="object_A_1" prm="0"/>
      <element objtype="mesh" objname="object_A_2" prm="0"/>
    </tuple>
    <tuple name="object_B">
      <element objtype="mesh" objname="object_B_0" prm="0"/>
      <element objtype="mesh" objname="object_B_1" prm="0"/>
    </tuple>
    <tuple name="object">
      <element objtype="tuple" objname="object_A" prm="0.6"/>
      <element objtype="tuple" objname="object_B" prm="0.4"/>
    </tuple>
  </custom>
</mujoco>
"""


class IOTest(parameterized.TestCase):
  @parameterized.named_parameters(
    dict(
      testcase_name="control_timestamps",
      times=np.array([0.0, 0.1, 0.4]),
      expected_indices=[0, 1, 1, 1, 2, 2, 2],
    ),
    dict(
      testcase_name="interval_boundaries",
      times=np.array([0.0, 0.1, 0.4, 0.6]),
      expected_indices=[0, 1, 1, 1, 2, 2],
    ),
    dict(
      testcase_name="substep_interval",
      times=np.array([0.0, 0.04, 0.06, 0.2]),
      expected_indices=[0, 2],
    ),
    dict(
      testcase_name="floating_point_boundaries",
      times=np.array([0.0, 0.3, 0.6, 0.9]),
      expected_indices=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    ),
  )
  def test_load_trajectory_npz_roundtrip(self, times, expected_indices):
    model = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <option timestep="0.1"/>
        <worldbody><body><joint name="joint"/><geom size="0.1"/></body></worldbody>
        <actuator><motor joint="joint"/></actuator>
      </mujoco>
      """
    )
    data = mujoco.MjData(model)
    ctrl = np.array([[1.0], [2.0], [3.0]])

    with tempfile.TemporaryDirectory() as tmp_dir:
      trajectory_path = f"{tmp_dir}/trajectory.npz"
      np.savez(
        trajectory_path,
        ctrl=ctrl,
        times=times,
        qpos=np.array([[0.25]]),
        qvel=np.array([[0.5]]),
      )
      loaded_ctrl = io.load_trajectory(trajectory_path, model, data)

    np.testing.assert_array_equal(loaded_ctrl, ctrl[expected_indices])
    np.testing.assert_array_equal(data.qpos, [0.25])
    np.testing.assert_array_equal(data.qvel, [0.5])

  @parameterized.parameters((47, 48), (48, 64), (63, 64), (64, 80))
  def test_augmented_cholesky_padding(self, nv, expected):
    _, nv_pad = io._get_padded_sizes(nv, 0, False, types.TILE_SIZE_JTDAJ_DENSE, augment_cholesky=True)
    self.assertEqual(nv_pad, expected)

  @parameterized.parameters((15, 16), (16, 32), (31, 32), (32, 48))
  def test_augmented_cholesky_nvmax_padding(self, nvmax, expected):
    self.assertEqual(io._nvmax_pad(nvmax), expected)

  def test_default_njmax_nnz_includes_scalar_joint_limits(self):
    """Include slide and hinge limits in the sparse Jacobian capacity."""
    mjm = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <option jacobian="sparse"/>
        <worldbody>
          <body>
            <joint type="slide" limited="true" range="-1 1" frictionloss="1"/>
            <joint type="hinge" limited="true" range="-1 1" frictionloss="1"/>
            <geom size="0.1" contype="0" conaffinity="0"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )

    self.assertEqual(io._default_njmax_nnz(mjm, nconmax=0, njmax=4), 4)

  def test_make_put_data(self):
    """Tests that make_data and put_data are producing the same shapes for all arrays."""
    mjm, _, _, d = test_data.fixture("pendula.xml", nvmax=None)
    md = mjwarp.make_data(mjm)

    # same number of fields
    self.assertEqual(len(d.__dict__), len(md.__dict__))

    # test shapes for all arrays
    for attr, val in md.__dict__.items():
      if isinstance(val, wp.array):
        self.assertEqual(val.shape, getattr(d, attr).shape, f"{attr} shape mismatch")

  def test_put_data_builds_jtdaj_block_list(self):
    """put_data builds the sparse-Newton JTDAJ block list straight from the loaded efc.

    make_constraint builds efc.jtdaj_* while assembling constraints, but put_data does not run it;
    it must populate the list so a put_data state is directly solvable, matching make_constraint.
    """
    mjm, mjd, m, d = test_data.fixture(
      "constraints.xml",
      keyframe=2,
      overrides={
        "opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE,
        "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
      },
    )
    self.assertTrue(m.is_sparse)
    self.assertGreater(mjd.nefc, 0)

    nblock = int(d.efc.jtdaj_nblock.numpy()[0])
    adr = d.efc.jtdaj_adr.numpy()[0, :nblock].copy()
    nrow = d.efc.jtdaj_nrow.numpy()[0, :nblock].copy()

    # the blocks partition the active rows exactly (contiguous, no gaps, covering [0, nefc))
    self.assertGreater(nblock, 0)
    self.assertEqual(adr[0], 0)
    np.testing.assert_array_equal(adr[1:], np.cumsum(nrow)[:-1])
    self.assertEqual(int(nrow.sum()), mjd.nefc)

    # each block is one constraint instance: a maximal run of rows sharing (efc_type, efc_id)
    etype = mjd.efc_type[: mjd.nefc]
    eid = mjd.efc_id[: mjd.nefc]
    for a, n in zip(adr.tolist(), nrow.tolist()):
      self.assertTrue((etype[a : a + n] == etype[a]).all())
      self.assertTrue((eid[a : a + n] == eid[a]).all())
      if a > 0:  # maximal: the preceding row belongs to a different instance
        self.assertTrue(etype[a - 1] != etype[a] or eid[a - 1] != eid[a])

  @parameterized.parameters(types.Model, types.Data, types.RenderContext)
  def test_is_batched_metadata(self, dataclass_type):
    """Verify that all fields specified with 'nworld' or '*' are marked with _is_batched=True."""
    fixture_outputs = test_data.fixture("pendula.xml")
    if dataclass_type is types.RenderContext:
      obj = mjwarp.create_render_context(fixture_outputs[0], nworld=1)
    else:
      obj = next(o for o in fixture_outputs if isinstance(o, dataclass_type))

    for f in dataclasses.fields(dataclass_type):
      if not warp_util.is_array_spec(f.type):
        continue
      spec_shape = getattr(f.type, "shape", ())
      if not (spec_shape and spec_shape[0] in ("*", "nworld")):
        continue
      arr = getattr(obj, f.name)
      if arr is None:
        continue
      self.assertTrue(
        getattr(arr, "_is_batched", False),
        msg=(
          f"{dataclass_type.__name__} field '{f.name}' has shape spec {spec_shape} "
          "but its instantiated array does not have _is_batched=True"
        ),
      )

  @parameterized.parameters(*_IO_TEST_MODELS)
  def test_put_data_sizes(self, xml):
    EXPECTED_SIZES = {
      "pendula.xml": (48, 64),
      "collision_sdf/tactile.xml": (64, 256),
      "flex/floppy.xml": (256, 512),
      "actuation/tendon_force_limit.xml": (48, 64),
      "actuation/tendon_force_limit.xml": (48, 64),
      "hfield/hfield.xml": (96, 384),
    }
    _, _, _, d = test_data.fixture(xml)
    nconmax_expected, njmax_expected = EXPECTED_SIZES[xml]
    self.assertEqual(d.naconmax, nconmax_expected)
    self.assertEqual(d.njmax, njmax_expected)

  def test_get_data_into_m(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body pos="0 0 0" >
            <geom type="box" pos="0 0 0" size=".5 .5 .5" />
            <joint type="hinge" />
          </body>
          <body pos="0 0 0.1">
            <geom type="sphere" size="0.5"/>
            <freejoint/>
          </body>
        </worldbody>
      </mujoco>
    """)

    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    mjd_ref = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd_ref)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mjd.M.fill(-123)

    mjwarp.get_data_into(mjd, mjm, d)
    np.testing.assert_allclose(mjd.qLD, mjd_ref.qLD)
    np.testing.assert_allclose(mjd.M, mjd_ref.M)

  @parameterized.named_parameters(
    dict(testcase_name="nworld=1", nworld=1, world_id=0),
    dict(testcase_name="nworld=2_world_id=1", nworld=2, world_id=1),
  )
  def test_get_data_into(self, nworld, world_id):
    # keyframe=0: ncon=8, nefc=32
    mjm, mjd, _, d = test_data.fixture("humanoid/humanoid.xml", keyframe=0, nworld=nworld)

    # keyframe=2: ncon=0, nefc=0
    mujoco.mj_resetDataKeyframe(mjm, mjd, 2)
    d.time.fill_(0.12345)

    # check that mujoco._functions._realloc_con_efc allocates for contact and efc
    mjwarp.get_data_into(mjd, mjm, d, world_id=world_id)
    self.assertEqual(mjd.ncon, 8)
    self.assertEqual(mjd.nefc, 32)

    # compare fields
    self.assertEqual(d.solver_niter.numpy()[world_id], mjd.solver_niter[0])
    self.assertEqual(d.nacon.numpy()[0], mjd.ncon * nworld)
    self.assertEqual(d.ne.numpy()[world_id], mjd.ne)
    self.assertEqual(d.nf.numpy()[world_id], mjd.nf)
    self.assertEqual(d.nl.numpy()[world_id], mjd.nl)
    self.assertEqual(d.nisland.numpy()[world_id], mjd.nisland)
    _assert_eq(d.time.numpy()[world_id], mjd.time, "time")

    for field in [
      "energy",
      "qpos",
      "qvel",
      "act",
      "qacc_warmstart",
      "ctrl",
      "qfrc_applied",
      "xfrc_applied",
      "eq_active",
      "mocap_pos",
      "mocap_quat",
      "qacc",
      "act_dot",
      "xpos",
      "xquat",
      "xmat",
      "xipos",
      "ximat",
      "xanchor",
      "xaxis",
      "geom_xpos",
      "geom_xmat",
      "site_xpos",
      "site_xmat",
      "cam_xpos",
      "cam_xmat",
      "light_xpos",
      "light_xdir",
      "subtree_com",
      "cdof",
      "cinert",
      "flexvert_xpos",
      "flexedge_length",
      "flexedge_velocity",
      "actuator_length",
      "crb",
      # TODO(team): qLDiagInv sparse factorization
      "ten_velocity",
      "actuator_velocity",
      "cvel",
      "cdof_dot",
      "qfrc_bias",
      "qfrc_spring",
      "qfrc_damper",
      "qfrc_gravcomp",
      "qfrc_fluid",
      "qfrc_passive",
      "subtree_linvel",
      "subtree_angmom",
      "actuator_force",
      "qfrc_actuator",
      "qfrc_smooth",
      "qacc_smooth",
      "qfrc_constraint",
      "qfrc_inverse",
      # TODO(team): qM
      # TODO(team): qLD
      "cacc",
      "cfrc_int",
      "cfrc_ext",
      "ten_length",
      "ten_J",
      "ten_wrapadr",
      "ten_wrapnum",
      "wrap_obj",
      "wrap_xpos",
      "sensordata",
    ]:
      _assert_eq(
        getattr(d, field).numpy()[world_id].reshape(-1),
        getattr(mjd, field).reshape(-1),
        field,
      )

    # actuator_moment
    actuator_moment_dense = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(actuator_moment_dense, mjd.actuator_moment, mjd.moment_rownnz, mjd.moment_rowadr, mjd.moment_colind)
    wp_actuator_moment = np.zeros((mjm.nu, mjm.nv))
    mujoco.mju_sparse2dense(
      wp_actuator_moment,
      d.actuator_moment.numpy()[world_id],
      d.moment_rownnz.numpy()[world_id],
      d.moment_rowadr.numpy()[world_id],
      d.moment_colind.numpy()[world_id],
    )
    _assert_eq(
      wp_actuator_moment.reshape(-1),
      actuator_moment_dense.reshape(-1),
      "actuator_moment",
    )

    # contact
    ncon = int(d.nacon.numpy()[0] / nworld)
    for field in [
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
      # TODO(team): efc_address
    ]:
      _assert_eq(
        getattr(d.contact, field).numpy()[world_id * ncon : world_id * ncon + ncon].reshape(-1),
        getattr(mjd.contact, field).reshape(-1),
        field,
      )

    # efc
    nefc = d.nefc.numpy()[world_id]
    for field in [
      "type",
      "id",
      "pos",
      "margin",
      "D",
      "vel",
      "aref",
      "frictionloss",
      "state",
      "force",
    ]:
      _assert_eq(
        getattr(d.efc, field).numpy()[world_id, :nefc].reshape(-1),
        getattr(mjd, "efc_" + field).reshape(-1),
        field,
      )

  def test_get_data_into_realloc_island(self):
    """Verifies initial copy, arena reallocation after reset, and dynamic expansion of islands."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable"/>
        </option>
        <size memory="100M"/>
        <worldbody>
          <body name="b1">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="b2" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="b3" pos="10 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="b4" pos="11 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld name="w1" body1="b1" body2="b2"/>
          <weld name="w2" body1="b3" body2="b4" active="false"/>
        </equality>
      </mujoco>
    """,
      nworld=1,
    )
    mjwarp.forward(m, d)
    expected_nisland = int(d.nisland.numpy()[0])
    expected_nidof = int(d.nidof.numpy()[0])
    self.assertEqual(expected_nisland, 1)
    self.assertGreater(expected_nidof, 0)

    mjwarp.get_data_into(mjd, mjm, d, world_id=0)
    self.assertEqual(mjd.nisland, expected_nisland)
    self.assertEqual(mjd.nidof, expected_nidof)
    self.assertEqual(mjd.island_idofadr.shape[0], expected_nisland)
    self.assertEqual(mjd.ifrc_smooth.shape[0], expected_nidof)

    np.testing.assert_array_equal(mjd.tree_island, d.tree_island.numpy()[0])
    np.testing.assert_array_equal(mjd.dof_island, d.dof_island.numpy()[0])
    np.testing.assert_array_equal(mjd.island_idofadr[:expected_nisland], d.island_idofadr.numpy()[0, :expected_nisland])
    np.testing.assert_array_equal(mjd.island_dofadr[:expected_nisland], d.island_dofadr.numpy()[0, :expected_nisland])
    np.testing.assert_array_equal(mjd.island_nv[:expected_nisland], d.island_nv.numpy()[0, :expected_nisland])
    np.testing.assert_array_equal(mjd.island_nefc[:expected_nisland], d.island_nefc.numpy()[0, :expected_nisland])
    np.testing.assert_array_equal(mjd.island_ne[:expected_nisland], d.island_ne.numpy()[0, :expected_nisland])
    np.testing.assert_array_equal(mjd.island_nf[:expected_nisland], d.island_nf.numpy()[0, :expected_nisland])
    np.testing.assert_array_equal(mjd.island_iefcadr[:expected_nisland], d.island_iefcadr.numpy()[0, :expected_nisland])
    np.testing.assert_array_equal(mjd.map_dof2idof[: mjm.nv], d.map_dof2idof.numpy()[0, : mjm.nv])
    np.testing.assert_array_equal(mjd.map_idof2dof[: mjm.nv], d.map_idof2dof.numpy()[0, : mjm.nv])
    nefc = int(d.nefc.numpy()[0])
    np.testing.assert_array_equal(mjd.map_efc2iefc[:nefc], d.map_efc2iefc.numpy()[0, :nefc])
    np.testing.assert_array_equal(mjd.map_iefc2efc[:nefc], d.map_iefc2efc.numpy()[0, :nefc])
    np.testing.assert_array_equal(mjd.efc_island[:nefc], d.efc.island.numpy()[0, :nefc])

    mujoco.mj_resetData(mjm, mjd)
    self.assertEqual(mjd.nisland, 0)
    mjwarp.get_data_into(mjd, mjm, d, world_id=0)
    self.assertEqual(mjd.nisland, expected_nisland)
    self.assertEqual(mjd.nidof, expected_nidof)
    self.assertEqual(mjd.island_idofadr.shape[0], expected_nisland)
    d.eq_active.fill_(True)
    mjwarp.forward(m, d)
    expected_nisland2 = int(d.nisland.numpy()[0])
    expected_nidof2 = int(d.nidof.numpy()[0])
    self.assertEqual(expected_nisland2, 2)
    self.assertGreater(expected_nidof2, expected_nidof)

    mjwarp.get_data_into(mjd, mjm, d, world_id=0)
    self.assertEqual(mjd.nisland, expected_nisland2)
    self.assertEqual(mjd.nidof, expected_nidof2)
    self.assertEqual(mjd.island_idofadr.shape[0], expected_nisland2)
    self.assertEqual(mjd.ifrc_smooth.shape[0], expected_nidof2)

  def test_get_data_into_islands_disabled(self):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag island="disable"/>
        </option>
        <worldbody>
          <body name="b">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
    """,
      nworld=1,
    )
    mjwarp.forward(m, d)
    mjwarp.get_data_into(mjd, mjm, d)
    self.assertEqual(mjd.nisland, 0)
    self.assertEqual(mjd.nidof, 0)

  @parameterized.product(
    xml=_IO_TEST_MODELS,
    cone=list(ConeType),
    integrator=list(IntegratorType),
  )
  def test_get_data_into_io_test_models(self, xml, cone, integrator):
    """Tests get_data_into for field coverage across diverse model types."""
    mjm, _, m, d = test_data.fixture(xml, nworld=2, overrides={"opt.cone": cone, "opt.integrator": integrator})
    mjwarp.step(m, d)

    for world_id in range(2):
      # Create reference MjData from warp data (resizes contact/efc fields internally)
      mjd = mujoco.MjData(mjm)
      mjwarp.get_data_into(mjd, mjm, d, world_id=world_id)

      # Compare key fields, including flex/tendon data not covered by humanoid.xml
      for field in [
        "qpos",
        "qvel",
        "qacc",
        "ctrl",
        "act",
        "flexvert_xpos",
        "flexedge_length",
        "flexedge_velocity",
        "ten_length",
        "ten_velocity",
        "actuator_length",
        "actuator_velocity",
        "actuator_force",
        "xpos",
        "xquat",
        "geom_xpos",
        "tree_island",
      ]:
        if field == "tree_island" and d.nisland.numpy()[0] == 0:
          continue
        if getattr(mjd, field).size > 0:
          _assert_eq(
            getattr(mjd, field).reshape(-1),
            getattr(d, field).numpy()[world_id].reshape(-1),
            f"{field} (model: {xml}, world: {world_id})",
          )

      # flexedge_J
      if xml == "flex/floppy.xml":
        _assert_eq(
          mjd.flexedge_J.reshape(-1),
          d.flexedge_J.numpy()[world_id].reshape(-1),
          f"flexedge_J (world: {world_id})",
        )

  def test_ellipsoid_fluid_model(self):
    mjm = mujoco.MjModel.from_xml_string(
      """
    <mujoco>
      <option density="1.1" viscosity="0.05"/>
      <worldbody>
        <body>
          <geom type="sphere" size=".15" fluidshape="ellipsoid"/>
          <freejoint/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    m = mjwarp.put_model(mjm)

    np.testing.assert_allclose(m.geom_fluid.numpy(), mjm.geom_fluid)
    self.assertTrue(m.has_fluid)

    body_has = m.body_fluid_ellipsoid.numpy()
    self.assertTrue(body_has[mjm.geom_bodyid[0]])
    self.assertFalse(body_has[0])

  def test_jacobian_auto(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option jacobian="auto"/>
        <worldbody>
          <replicate count="11">
          <body>
            <geom type="sphere" size=".1"/>
            <freejoint/>
            </body>
          </replicate>
        </worldbody>
      </mujoco>
    """)
    mjwarp.put_model(mjm)

  def test_put_data_qLD(self):
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="hinge"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    d = mjwarp.put_data(mjm, mjd)
    self.assertTrue((d.qLD.numpy() == 0.0).all())

    mujoco.mj_forward(mjm, mjd)
    mjd.M[:] = 0.0
    d = mjwarp.put_data(mjm, mjd)
    self.assertTrue((d.qLD.numpy() == 0.0).all())

    mujoco.mj_forward(mjm, mjd)
    # For block-dense models d.qLD is the packed Cholesky factor recomputed from M, so
    # zero the mass matrix as well as qLD to drive the factor to zero on both paths.
    mjd.qLD[:] = 0.0
    mjd.M[:] = 0.0
    d = mjwarp.put_data(mjm, mjd)
    self.assertTrue((d.qLD.numpy() == 0.0).all())

  def test_static_geom_collision_with_put_data(self):
    """Test that static geoms (ground plane) work correctly with put_data."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option timestep="0.02"/>
        <worldbody>
          <geom name="ground" type="plane" pos="0 0 0" size="0 0 1"/>
          <body name="box" pos="0 0 0.6">
            <freejoint/>
            <geom name="box" type="box" size="0.5 0.5 0.5"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd, nconmax=16, njmax=16)

    # let the box fall and settle on the ground
    for _ in range(30):
      mjwarp.step(m, d)

    # check that box is above ground
    # box center should be at z ≈ 0.5 when resting on ground
    box_z = d.xpos.numpy()[0, 1, 2]  # world 0, body 1 (box), z coordinate
    self.assertGreater(box_z, 0.4, msg=f"Box fell through ground plane (z={box_z}, should be > 0.4)")

  def test_make_data_nccdmax_exceeds_nconmax(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    with self.assertRaises(ValueError, msg="nccdmax.*nconmax"):
      mjwarp.make_data(mjm, nconmax=16, nccdmax=17)

  def test_make_data_naccdmax_exceeds_naconmax(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    with self.assertRaises(ValueError, msg="naccdmax.*naconmax"):
      mjwarp.make_data(mjm, nconmax=16, naconmax=16, naccdmax=17)

  def test_make_data_naccdmax_default(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    data = mjwarp.make_data(mjm, naconmax=5, njmax=3, naccdmax=None)
    self.assertEqual(data.naccdmax, 5, "naccdmax=None should default to naconmax")

  def test_put_data_naccdmax_default(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    mjd = mujoco.MjData(mjm)
    data = mjwarp.put_data(mjm, mjd, naconmax=5, njmax=3, naccdmax=None)
    self.assertEqual(data.naccdmax, 5, "naccdmax=None should default to naconmax")

  def test_make_data_naccdmax_from_nccdmax(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    data = mjwarp.make_data(mjm, nconmax=5, nccdmax=3)
    self.assertEqual(data.naccdmax, 3, "naccdmax from nccdmax")

  def test_put_data_naccdmax_from_nccdmax(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    mjd = mujoco.MjData(mjm)
    data = mjwarp.put_data(mjm, mjd, nconmax=5, nccdmax=3)
    self.assertEqual(data.naccdmax, 3, "naccdmax from nccdmax")

  def test_make_data_naccdmax_from_nccdmax_nworld(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    data = mjwarp.make_data(mjm, nworld=3, nconmax=7, nccdmax=5)
    self.assertEqual(data.naccdmax, 15, "naccdmax from nccdmax and nworld")

  def test_put_data_nccdmax_exceeds_nconmax(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    mjd = mujoco.MjData(mjm)
    with self.assertRaises(ValueError, msg="nccdmax.*nconmax"):
      mjwarp.put_data(mjm, mjd, nconmax=16, nccdmax=17)

  def test_put_data_naccdmax_exceeds_naconmax(self):
    mjm = mujoco.MjModel.from_xml_string("<mujoco/>")
    mjd = mujoco.MjData(mjm)
    with self.assertRaises(ValueError, msg="naccdmax.*naconmax"):
      mjwarp.put_data(mjm, mjd, nconmax=16, naconmax=16, naccdmax=17)

  @parameterized.parameters(1, 2)
  def test_put_data_island(self, nworld):
    """Test that all island fields are correctly initialized from MjData by put_data."""
    mjm, mjd, _, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="a1">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="a2" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="b1" pos="5 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="b2" pos="6 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="c_unconstrained" pos="10 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="a1" body2="a2"/>
          <weld body1="b1" body2="b2"/>
        </equality>
      </mujoco>
      """,
      nworld=nworld,
    )

    self.assertGreater(mjd.nisland, 0)
    self.assertGreater(mjd.nidof, 0)
    self.assertGreater(mjd.nefc, 0)

    nisland = mjd.nisland
    nidof = mjd.nidof
    nefc = mjd.nefc

    fields = [
      ("tree_island", mjm.ntree),
      ("dof_island", mjm.nv),
      ("island_dofadr", nisland),
      ("island_idofadr", nisland),
      ("island_nv", nisland),
      ("island_nefc", nisland),
      ("island_ne", nisland),
      ("island_nf", nisland),
      ("island_iefcadr", nisland),
      ("map_dof2idof", mjm.nv),
      ("map_idof2dof", mjm.nv),
      ("map_efc2iefc", nefc),
      ("map_iefc2efc", nefc),
    ]

    for w in range(nworld):
      # Test warp helper island ID arrays
      dof_islandid = d.dof_islandid.numpy()[w]
      np.testing.assert_array_equal(dof_islandid[:nidof], mjd.dof_island[mjd.map_idof2dof[:nidof]])
      np.testing.assert_array_equal(dof_islandid[nidof:], np.full(mjm.nv - nidof, -1))
      np.testing.assert_array_equal(d.efc_islandid.numpy()[w, :nefc], mjd.efc_island[mjd.map_iefc2efc[:nefc]])

      # Test roundtrip with get_data_into
      result = mujoco.MjData(mjm)
      mujoco.mj_forward(mjm, result)
      mjwarp.get_data_into(result, mjm, d, world_id=w)

      for obj in (d, result):
        get_val = lambda name: getattr(obj, name).numpy()[w] if obj is d else getattr(obj, name)
        self.assertEqual(get_val("nisland"), nisland)
        self.assertEqual(get_val("nidof"), nidof)
        for name, sz in fields:
          np.testing.assert_array_equal(get_val(name)[:sz], getattr(mjd, name)[:sz])
        np.testing.assert_array_equal(
          obj.efc.island.numpy()[w, :nefc] if obj is d else obj.efc_island[:nefc],
          mjd.efc_island[:nefc],
        )

  @parameterized.parameters(1, 2)
  def test_put_data_island_unconstrained(self, nworld):
    """Test put_data island initialization for unconstrained model (nisland=0)."""
    mjm, mjd, _, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="free_body">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    self.assertEqual(mjd.nisland, 0)
    self.assertEqual(mjd.nefc, 0)

    for w in range(nworld):
      self.assertEqual(d.nisland.numpy()[w], 0)
      self.assertEqual(d.nidof.numpy()[w], 0)
      np.testing.assert_array_equal(d.tree_island.numpy()[w], np.full(mjm.ntree, -1))
      np.testing.assert_array_equal(d.dof_island.numpy()[w], np.full(mjm.nv, -1))
      np.testing.assert_array_equal(d.dof_islandid.numpy()[w], np.full(mjm.nv, -1))
      np.testing.assert_array_equal(d.efc_islandid.numpy()[w], np.full(d.njmax, -1))

  @parameterized.parameters(1, 2)
  def test_put_data_island_sleep(self, nworld):
    """Test put_data with sleeping islands and verify they remain asleep after a Warp step."""
    mjm, mjd, _, _ = test_data.fixture(
      xml="""
      <mujoco>
        <option sleep_tolerance="0.01" gravity="0 0 0">
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="a1">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="a2" pos="1 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="b1" pos="5 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="b2" pos="6 0 0">
            <joint type="free"/>
            <geom size=".1"/>
          </body>
          <body name="c1" pos="10 0 0">
            <joint name="j_c1" type="free"/>
            <geom size=".1"/>
          </body>
          <body name="c2" pos="11 0 0">
            <joint name="j_c2" type="free"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="a1" body2="a2"/>
          <weld body1="b1" body2="b2"/>
          <weld body1="c1" body2="c2"/>
        </equality>
      </mujoco>
      """
    )

    # Set velocity for island 2 so it stays awake while islands 0 and 1 go to sleep
    mjd.qvel[24] = 1.0
    mjd.qvel[30] = 1.0

    for _ in range(mujoco.mjMINAWAKE + 1):
      mujoco.mj_step(mjm, mjd)

    self.assertEqual(mjd.nisland, 1)
    self.assertTrue(np.all(mjd.tree_asleep[:4] >= 0))
    self.assertTrue(np.all(mjd.tree_asleep[4:] < 0))
    self.assertTrue(np.all(mjd.tree_awake[:4] == 0))
    self.assertTrue(np.all(mjd.tree_awake[4:] == 1))
    self.assertEqual(mjd.body_awake[0], types.SleepState.STATIC)
    self.assertTrue(np.all(mjd.body_awake[1:5] == types.SleepState.ASLEEP))
    self.assertTrue(np.all(mjd.body_awake[5:7] == types.SleepState.AWAKE))

    m = mjwarp.put_model(mjm)
    d = io.put_data(mjm, mjd, nworld=nworld)

    nisland = mjd.nisland
    nidof = mjd.nidof
    nefc = mjd.nefc

    fields = [
      ("tree_island", mjm.ntree),
      ("dof_island", mjm.nv),
      ("island_dofadr", nisland),
      ("island_idofadr", nisland),
      ("island_nv", nisland),
      ("island_nefc", nisland),
      ("island_ne", nisland),
      ("island_nf", nisland),
      ("island_iefcadr", nisland),
      ("map_dof2idof", mjm.nv),
      ("map_idof2dof", mjm.nv),
      ("map_efc2iefc", nefc),
      ("map_iefc2efc", nefc),
    ]

    for w in range(nworld):
      # Test roundtrip with get_data_into
      result = mujoco.MjData(mjm)
      io.get_data_into(result, mjm, d, world_id=w)

      for obj in (d, result):
        get_val = lambda name: getattr(obj, name).numpy()[w] if obj is d else getattr(obj, name)
        self.assertEqual(get_val("nisland"), nisland)
        self.assertEqual(get_val("nidof"), nidof)
        for name, sz in fields:
          np.testing.assert_array_equal(get_val(name)[:sz], getattr(mjd, name)[:sz])
        np.testing.assert_array_equal(
          obj.efc.island.numpy()[w, :nefc] if obj is d else obj.efc_island[:nefc],
          mjd.efc_island[:nefc],
        )
        np.testing.assert_array_equal(get_val("tree_asleep"), mjd.tree_asleep)
        np.testing.assert_array_equal(get_val("body_awake"), mjd.body_awake)

      np.testing.assert_array_equal(d.dof_islandid.numpy()[w, :nidof], mjd.dof_island[mjd.map_idof2dof[:nidof]])
      np.testing.assert_array_equal(d.efc_islandid.numpy()[w, :nefc], mjd.efc_island[mjd.map_iefc2efc[:nefc]])
      np.testing.assert_array_equal(d.tree_awake.numpy()[w], (mjd.tree_asleep < 0).astype(int))

    mjwarp.step(m, d)

    # Verify that sleeping islands remain asleep and the awake island remains awake after stepping.
    for w in range(nworld):
      np.testing.assert_array_equal(d.tree_asleep.numpy()[w][:4], mjd.tree_asleep[:4])
      self.assertTrue(np.all(d.tree_asleep.numpy()[w][4:] < 0))
      np.testing.assert_array_equal(d.tree_awake.numpy()[w], [0, 0, 0, 0, 1, 1])
      self.assertEqual(d.ntree_awake.numpy()[w], 2)
      self.assertEqual(d.body_awake.numpy()[w, 0], types.SleepState.STATIC)
      self.assertTrue(np.all(d.body_awake.numpy()[w, 1:5] == types.SleepState.ASLEEP))
      self.assertTrue(np.all(d.body_awake.numpy()[w, 5:7] == types.SleepState.AWAKE))
      self.assertEqual(d.nbody_awake.numpy()[w], 3)
      self.assertEqual(d.nv_awake.numpy()[w], 12)
      np.testing.assert_array_equal(d.qvel.numpy()[w, :24], np.zeros(24))
      np.testing.assert_array_equal(d.qacc.numpy()[w, :24], np.zeros(24))
      self.assertTrue(np.any(d.qvel.numpy()[w, 24:] != 0.0))

  def test_noslip_solver(self):
    with self.assertRaises(NotImplementedError):
      test_data.fixture(
        xml="""
      <mujoco>
        <option noslip_iterations="1"/>
      </mujoco>
      """
      )

  @parameterized.parameters(*_IO_TEST_MODELS)
  def test_reset_data(self, xml):
    reset_datafield = [
      "ne",
      "nf",
      "nl",
      "nefc",
      "time",
      "energy",
      "qpos",
      "qvel",
      "act",
      "ctrl",
      "eq_active",
      "qfrc_applied",
      "xfrc_applied",
      "qacc",
      "qacc_warmstart",
      "act_dot",
      "sensordata",
      "mocap_pos",
      "mocap_quat",
      "M",
      "tree_asleep",
      "tree_awake",
      "body_awake",
      "body_awake_ind",
      "dof_awake_ind",
      "ntree_awake",
      "nbody_awake",
      "nv_awake",
    ]

    mjm, mjd, m, d = test_data.fixture(xml)
    naconmax = d.naconmax

    # data fields
    for arr in reset_datafield:
      attr = getattr(d, arr)
      if attr.dtype == float:
        attr.fill_(wp.nan)
      else:
        attr.fill_(-1)

    for arr in d.contact.__dataclass_fields__:
      attr = getattr(d.contact, arr)
      if attr.dtype == float:
        attr.fill_(wp.nan)
      else:
        attr.fill_(-1)

    mujoco.mj_resetData(mjm, mjd)

    # set nacon in order to zero all contact memory
    wp.copy(d.nacon, wp.array([naconmax], dtype=int))
    mjwarp.reset_data(m, d)

    for arr in reset_datafield:
      d_arr = getattr(d, arr).numpy()
      for i in range(d_arr.shape[0]):
        di_arr = d_arr[i]
        if arr == "M":
          di_arr = di_arr.reshape(-1)[: mjd.M.size]
        _assert_eq(di_arr, getattr(mjd, arr), arr)

    _assert_eq(d.nacon.numpy(), 0, "nacon")

    for arr in d.contact.__dataclass_fields__:
      if arr == "efc_address":
        _assert_eq(getattr(d.contact, arr).numpy(), -1, arr)
      else:
        _assert_eq(getattr(d.contact, arr).numpy(), 0.0, arr)

  def test_reset_data_world(self):
    """Tests per-world reset."""
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    m = mjwarp.put_model(mjm)
    d = mjwarp.make_data(mjm, nworld=2)

    # nonzero values
    qvel = wp.array(np.array([[1.0], [2.0]]), dtype=float)

    wp.copy(d.qvel, qvel)

    # reset both worlds
    mjwarp.reset_data(m, d)

    _assert_eq(d.qvel.numpy()[0], 0.0, "qvel[0]")
    _assert_eq(d.qvel.numpy()[1], 0.0, "qvel[1]")

    wp.copy(d.qvel, qvel)

    # don't reset second world
    reset10 = wp.array(np.array([True, False]), dtype=bool)
    mjwarp.reset_data(m, d, reset=reset10)

    _assert_eq(d.qvel.numpy()[0], 0.0, "qvel[0]")
    _assert_eq(d.qvel.numpy()[1], 2.0, "qvel[1]")

    wp.copy(d.qvel, qvel)

    # don't reset both worlds
    reset00 = wp.array(np.array([False, False], dtype=bool))
    mjwarp.reset_data(m, d, reset=reset00)

    _assert_eq(d.qvel.numpy()[0], 1.0, "qvel[0]")
    _assert_eq(d.qvel.numpy()[1], 2.0, "qvel[1]")

    wp.copy(d.qvel, qvel)

    # int arrays are tolerated as a reset mask (nonzero means reset)
    reset10_int = wp.array(np.array([1, 0]), dtype=int)
    mjwarp.reset_data(m, d, reset=reset10_int)

    _assert_eq(d.qvel.numpy()[0], 0.0, "qvel[0]")
    _assert_eq(d.qvel.numpy()[1], 2.0, "qvel[1]")

  def test_reset_data_reset_invalid(self):
    """Tests that reset_data validates the reset argument."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
    </mujoco>
    """,
      nworld=2,
    )

    with self.assertRaisesRegex(ValueError, "reset array must have shape"):
      mjwarp.reset_data(m, d, reset=wp.array(np.array([True, False, True]), dtype=bool))

    with self.assertRaisesRegex(ValueError, "reset array must be of bool or integer type"):
      mjwarp.reset_data(m, d, reset=wp.array(np.array([1.0, 0.0]), dtype=float))

    with self.assertRaisesRegex(ValueError, "reset must be None or a wp.array"):
      mjwarp.reset_data(m, d, reset=[True, False])

  def test_reset_data_keyframe(self):
    """Tests that reset_data_keyframe matches mj_resetDataKeyframe."""
    reset_datafield = ["time", "qpos", "qvel", "act", "mocap_pos", "mocap_quat", "ctrl"]
    key = 0

    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="mocap1" mocap="true">
          <geom type="sphere" size="0.1"/>
        </body>
        <body>
          <joint type="slide" name="slide1"/>
          <geom type="sphere" size="1"/>
        </body>
      </worldbody>
      <actuator>
        <general joint="slide1" dyntype="integrator"/>
      </actuator>
      <keyframe>
        <key name="k0" time="0.5" qpos="0.3" qvel="0.4" act="0.6" ctrl="0.7"
             mpos="0.1 0.2 0.3" mquat="0.7071068 0.7071068 0 0"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=key,
    )

    # corrupt data
    for arr in reset_datafield:
      attr = getattr(d, arr)
      if attr.dtype == float:
        attr.fill_(wp.nan)
      else:
        attr.fill_(-1)

    mjwarp.reset_data_keyframe(m, d, key)

    for arr in reset_datafield:
      _assert_eq(getattr(d, arr).numpy()[0], getattr(mjd, arr), arr)

  def test_reset_data_keyframe_world(self):
    """Tests per-world reset for reset_data_keyframe, skipping worlds via an invalid key."""
    key = 0

    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
      <keyframe>
        <key name="k0" qpos="0.5"/>
      </keyframe>
    </mujoco>
    """,
      nworld=2,
    )

    # nonzero values
    qpos = wp.array(np.array([[1.0], [2.0]]), dtype=float)

    wp.copy(d.qpos, qpos)

    # reset both worlds
    mjwarp.reset_data_keyframe(m, d, key)

    _assert_eq(d.qpos.numpy()[0], 0.5, "qpos[0]")
    _assert_eq(d.qpos.numpy()[1], 0.5, "qpos[1]")

    wp.copy(d.qpos, qpos)

    # don't reset second world: give it an out-of-range key
    key10 = wp.array(np.array([0, -1]), dtype=int)
    mjwarp.reset_data_keyframe(m, d, key10)

    _assert_eq(d.qpos.numpy()[0], 0.5, "qpos[0]")
    _assert_eq(d.qpos.numpy()[1], 2.0, "qpos[1]")

    wp.copy(d.qpos, qpos)

    # don't reset either world
    key00 = wp.array(np.array([-1, -1]), dtype=int)
    mjwarp.reset_data_keyframe(m, d, key00)

    _assert_eq(d.qpos.numpy()[0], 1.0, "qpos[0]")
    _assert_eq(d.qpos.numpy()[1], 2.0, "qpos[1]")

  def test_reset_data_keyframe_per_world(self):
    """Tests reset_data_keyframe with a per-world keyframe array."""
    reset_datafield = ["time", "qpos", "qvel", "act", "mocap_pos", "mocap_quat", "ctrl"]

    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="mocap1" mocap="true">
          <geom type="sphere" size="0.1"/>
        </body>
        <body>
          <joint type="slide" name="slide1"/>
          <geom type="sphere" size="1"/>
        </body>
      </worldbody>
      <actuator>
        <general joint="slide1" dyntype="integrator"/>
      </actuator>
      <keyframe>
        <key name="k0" time="0.1" qpos="0.2" qvel="0.3" act="0.4" ctrl="0.5"
             mpos="0.1 0 0" mquat="1 0 0 0"/>
        <key name="k1" time="0.6" qpos="0.7" qvel="0.8" act="0.9" ctrl="1.0"
             mpos="0 0.2 0" mquat="0.7071068 0.7071068 0 0"/>
      </keyframe>
    </mujoco>
    """,
      nworld=4,
    )

    # get reference values using the plain mujoco API
    mjd0 = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd0, 0)
    mjd1 = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd1, 1)

    # corrupt data
    for arr in reset_datafield:
      getattr(d, arr).fill_(-1.0)

    # world 0 -> key 0
    # world 1 -> key 1
    # world 2 -> key < 0 (reset skipped)
    # world 3 -> key >= nkey (reset skipped)
    key = wp.array(np.array([0, 1, -1, 2]), dtype=int)
    mjwarp.reset_data_keyframe(m, d, key)

    for arr in reset_datafield:
      d_arr = getattr(d, arr).numpy()
      expected = [
        getattr(mjd0, arr),  # reference value for world 0
        getattr(mjd1, arr),  # reference value for world 1
        -1.0,  # corrupted value
        -1.0,  # corrupted value
      ]
      for worldid, exp in enumerate(expected):
        _assert_eq(d_arr[worldid], exp, f"{arr}[{worldid}]")

  def test_reset_data_keyframe_no_keyframes(self):
    """Tests reset_data_keyframe on a model without keyframes (nkey == 0)."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
    </mujoco>
    """,
      nworld=2,
    )
    self.assertEqual(m.nkey, 0)

    with self.assertRaisesRegex(ValueError, r"key \(0\) must be in \[0, 0\)"):
      mjwarp.reset_data_keyframe(m, d, 0)

    qpos = wp.array(np.array([[1.0], [2.0]]), dtype=float)
    wp.copy(d.qpos, qpos)

    # every world has an out-of-range key, so nothing is reset
    mjwarp.reset_data_keyframe(m, d, wp.array(np.array([0, 0]), dtype=int))

    _assert_eq(d.qpos.numpy()[0], 1.0, "qpos[0]")
    _assert_eq(d.qpos.numpy()[1], 2.0, "qpos[1]")

  def test_reset_data_keyframe_key_invalid(self):
    """Tests that reset_data_keyframe validates the key argument."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
      <keyframe>
        <key name="k0" qpos="0.5"/>
      </keyframe>
    </mujoco>
    """,
      nworld=2,
    )

    with self.assertRaisesRegex(ValueError, r"key \(-1\) must be in \[0, 1\)"):
      mjwarp.reset_data_keyframe(m, d, -1)
    with self.assertRaisesRegex(ValueError, r"key \(1\) must be in \[0, 1\)"):
      mjwarp.reset_data_keyframe(m, d, 1)
    with self.assertRaisesRegex(ValueError, "key array must have shape"):
      mjwarp.reset_data_keyframe(m, d, wp.array(np.array([0, 0, 0]), dtype=int))
    with self.assertRaisesRegex(ValueError, "key array must be of integer type"):
      mjwarp.reset_data_keyframe(m, d, wp.array(np.array([0.0, 0.0]), dtype=float))
    with self.assertRaisesRegex(ValueError, "key must be an int or a wp.array"):
      mjwarp.reset_data_keyframe(m, d, 0.5)

  def test_reset_data_keyframe_numpy_int(self):
    """Tests that reset_data_keyframe accepts numpy integer scalars."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
      <keyframe>
        <key name="k0" qpos="0.5"/>
      </keyframe>
    </mujoco>
    """
    )

    for key in (np.int32(0), np.int64(0)):
      d.qpos.fill_(0.0)
      mjwarp.reset_data_keyframe(m, d, key)
      _assert_eq(d.qpos.numpy()[0], 0.5, "qpos")

  def test_sdf(self):
    """Tests that an SDF can be loaded."""
    mjm, mjd, m, d = test_data.fixture("collision_sdf/cow.xml")

    self.assertIsInstance(m.oct_aabb, wp.array)
    self.assertEqual(m.oct_aabb.dtype, wp.vec3)
    self.assertEqual(len(m.oct_aabb.shape), 2)
    if m.oct_aabb.size > 0:
      self.assertEqual(m.oct_aabb.shape[1], 2)

  def test_plugin(self):
    with self.assertRaises(NotImplementedError):
      test_data.fixture(
        xml="""
      <mujoco>
        <extension>
          <plugin plugin="mujoco.pid"/>
          <plugin plugin="mujoco.sensor.touch_grid"/>
          <plugin plugin="mujoco.elasticity.cable"/>
        </extension>
        <worldbody>
          <geom type="plane" size="10 10 .001"/>
          <body>
            <joint name="joint" type="slide"/>
            <geom type="sphere" size=".1"/>
            <site name="site"/>
          </body>
          <composite type="cable" curve="s" count="41 1 1" size="1" offset="-.3 0 .6" initial="none">
            <plugin plugin="mujoco.elasticity.cable">
              <config key="twist" value="1e7"/>
              <config key="bend" value="4e6"/>
              <config key="vmax" value="0.05"/>
            </plugin>
            <joint kind="main" damping=".015"/>
            <geom type="capsule" size=".005" rgba=".8 .2 .1 .1" condim="1"/>
          </composite>
        </worldbody>
        <actuator>
          <plugin plugin="mujoco.pid" joint="joint"/>
        </actuator>
        <sensor>
          <plugin plugin="mujoco.sensor.touch_grid" objtype="site" objname="site">
            <config key="size" value="7 7"/>
            <config key="fov" value="45 45"/>
            <config key="gamma" value="0"/>
            <config key="nchannel" value="3"/>
          </plugin>
        </sensor>
      </mujoco>
      """
      )

  def test_contact_sensor_maxmatch(self):
    _, _, m, _ = test_data.fixture(
      xml="""
    <mujoco>
    </mujoco>
    """
    )

    self.assertEqual(m.opt.contact_sensor_maxmatch, 64)

    _, _, m, _ = test_data.fixture(
      xml="""
    <mujoco>
      <custom>
        <numeric data="5" name="contact_sensor_maxmatch"/>
      </custom>
    </mujoco>
    """
    )

    self.assertEqual(m.opt.contact_sensor_maxmatch, 5)

  @parameterized.parameters(
    '<worldbody><geom type="sphere" size=".1" condim="3" friction="0 0.1 0.1"/></worldbody>',
    '<worldbody><geom type="sphere" size=".1" condim="4" friction="1 0 0.1"/></worldbody>',
    '<worldbody><geom type="sphere" size=".1" condim="6" friction="1 1 0"/></worldbody>',
    """
      <worldbody>
        <geom name="g1" type="sphere" size=".1"/>
        <geom name="g2" type="sphere" size=".1" pos="0.5 0 0"/>
      </worldbody>
      <contact>
        <pair geom1="g1" geom2="g2" condim="3" friction="0 1 1 1 1"/>
      </contact>
    """,
    """
      <worldbody>
        <geom name="g1" type="sphere" size=".1"/>
        <geom name="g2" type="sphere" size=".1" pos="0.5 0 0"/>
      </worldbody>
      <contact>
        <pair geom1="g1" geom2="g2" condim="4" friction="1 0 0 1 1"/>
      </contact>
    """,
    """
      <worldbody>
        <geom name="g1" type="sphere" size=".1"/>
        <geom name="g2" type="sphere" size=".1" pos="0.5 0 0"/>
      </worldbody>
      <contact>
        <pair geom1="g1" geom2="g2" condim="6" friction="1 1 1 0 0"/>
      </contact>
    """,
  )
  def test_small_friction_warning(self, xml):
    """Tests that a warning is raised for small friction values."""
    with self.assertWarns(UserWarning):
      mjwarp.put_model(mujoco.MjModel.from_xml_string(f"<mujoco>{xml}</mujoco>"))

  @parameterized.product(active=["true", "false"], make_data=[True, False])
  def test_eq_active(self, active, make_data):
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
    <mujoco>
      <worldbody>
        <body name="body1">
          <joint/>
          <geom size=".1"/>
        </body>
        <body name="body2">
          <joint/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="body1" body2="body2" active="{active}"/>
      </equality>
    </mujoco>
    """
    )
    if make_data:
      d = mjwarp.make_data(mjm)

    _assert_eq(d.eq_active.numpy()[0], mjd.eq_active, "eq_active")

  def test_tree_structure_fields(self):
    """Tests that tree structure fields match between types.Model and mjModel."""
    mjm, _, m, _ = test_data.fixture("pendula.xml")

    # verify fields match MuJoCo
    for field in ["ntree", "tree_dofadr", "tree_dofnum", "tree_bodynum", "body_treeid", "dof_treeid"]:
      m_val = getattr(m, field)
      mjm_val = getattr(mjm, field)
      if isinstance(m_val, wp.array):
        m_val = m_val.numpy()
      np.testing.assert_array_equal(m_val, mjm_val, err_msg=f"mismatch: {field}")

  def test_model_batched_fields(self):
    """Test Model batched fields."""
    nworld = 2
    mjm, _, m, d = test_data.fixture("humanoid/humanoid.xml", keyframe=0, nworld=nworld)

    for f in dataclasses.fields(m):
      # TODO(team): test arrays that are warp only
      if not hasattr(mjm, f.name):
        continue
      if isinstance(f.type, wp.array) or type(f.type).__name__ == "_ArrayAnnotation":
        # get fields
        arr = getattr(m, f.name)
        mj_arr = getattr(mjm, f.name)

        if not hasattr(mj_arr, "shape"):
          continue

        # check that field is not empty
        if 0 in mj_arr.shape + arr.shape:
          continue

        # check for batched field
        if hasattr(arr, "_is_batched") and arr._is_batched:
          assert arr.shape[0] == 1

          # reshape if necessary
          if f.name in ("cam_mat0"):
            mj_arr = mj_arr.reshape((-1, 3, 3))

          # set batched field
          setattr(m, f.name, wp.array(np.tile(mj_arr, (nworld,) + arr.shape[1:]), dtype=f.type.dtype))

    mjwarp.forward(m, d)
    mjwarp.reset_data(m, d)
    mjwarp.forward(m, d)

  def test_put_model_batch_sizes(self):
    """Test put_model can allocate selected batched Model fields per world."""
    mjm = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <asset>
          <texture name="red" type="2d" builtin="flat" width="4" height="4" rgb1="1 0 0" rgb2="1 0 0"/>
          <texture name="green" type="2d" builtin="flat" width="4" height="4" rgb1="0 1 0" rgb2="0 1 0"/>
          <material name="mat" texture="red" rgba="0.5 0.6 0.7 1"/>
        </asset>
        <worldbody>
          <geom type="sphere" size="0.1" material="mat"/>
        </worldbody>
      </mujoco>
      """
    )

    m_default = mjwarp.put_model(mjm)
    batch_sizes = {"mat_texid": 3, "geom_size": 2, "mat_rgba": 4}
    m = mjwarp.put_model(mjm, batch_sizes=batch_sizes)

    nrole = int(mujoco.mjtTextureRole.mjNTEXROLE)
    self.assertEqual(tuple(m.mat_texid.shape), (3, mjm.nmat, nrole))
    self.assertEqual(tuple(m.geom_size.shape), (2, mjm.ngeom))
    self.assertEqual(tuple(m.mat_rgba.shape), (4, mjm.nmat))

    np.testing.assert_array_equal(m.mat_texid.numpy(), np.repeat(m_default.mat_texid.numpy(), 3, axis=0))
    np.testing.assert_allclose(m.geom_size.numpy(), np.repeat(m_default.geom_size.numpy(), 2, axis=0))
    np.testing.assert_allclose(m.mat_rgba.numpy(), np.repeat(m_default.mat_rgba.numpy(), 4, axis=0))

    # field has batched dimension and defaults to batch size 1
    for f in dataclasses.fields(types.Model):
      if not warp_util.is_array_spec(f.type):
        continue
      spec_shape = getattr(f.type, "shape", ())
      if not spec_shape or spec_shape[0] != "*":
        continue
      val = getattr(m, f.name)
      if val is not None:
        expected = batch_sizes.get(f.name, 1)
        self.assertEqual(val.shape[0], expected, f"Field {f.name} does not have batch dimension {expected}")

    red_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_TEXTURE, "red")
    green_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_TEXTURE, "green")
    mat_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "mat")
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)

    mat_texid = m.mat_texid.numpy()
    mat_texid[:, mat_id, rgb_role] = [red_id, green_id, red_id]
    m.mat_texid.assign(mat_texid)
    np.testing.assert_array_equal(m.mat_texid.numpy()[:, mat_id, rgb_role], [red_id, green_id, red_id])

  def test_put_model_batch_sizes_errors(self):
    """Test invalid put_model batch_sizes requests are rejected."""
    mjm = mujoco.MjModel.from_xml_string(
      """
      <mujoco>
        <worldbody>
          <geom type="sphere" size="0.1"/>
        </worldbody>
      </mujoco>
      """
    )

    with self.assertRaisesRegex(ValueError, "not a batched array field"):
      mjwarp.put_model(mjm, batch_sizes={"missing": 2})
    with self.assertRaisesRegex(ValueError, "not a batched array field"):
      mjwarp.put_model(mjm, batch_sizes={"nmat": 2})
    with self.assertRaisesRegex(ValueError, "not a batched array field"):
      mjwarp.put_model(mjm, batch_sizes={"geom_type": 2})
    with self.assertRaisesRegex(ValueError, "must be positive"):
      mjwarp.put_model(mjm, batch_sizes={"mat_texid": 0})

  def test_domain_randomize_cranklength(self):
    """Test cranklength can be modified per-world after put_model (2D)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="j1" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    cl_np = m.actuator_cranklength.numpy()
    self.assertEqual(cl_np.ndim, 2)
    self.assertEqual(cl_np.shape, (1, mjm.nu))

    # verify we can write a new value per-world
    cl_np[0, 0] = 0.42
    wp.copy(
      m.actuator_cranklength,
      wp.array(cl_np, dtype=m.actuator_cranklength.dtype),
    )
    cl_read = m.actuator_cranklength.numpy()
    np.testing.assert_allclose(cl_read[0, 0], 0.42, atol=1e-6)

  def test_check_toolkit_driver_warns(self):
    """Tests that check_toolkit_driver warns."""
    mock_device = mock.MagicMock()
    mock_device.is_cuda = True
    with mock.patch("warp.get_device", return_value=mock_device):
      with mock.patch("warp.is_conditional_graph_supported", return_value=False):
        with self.assertWarns(UserWarning):
          warp_util.check_toolkit_driver()

  def test_put_data_nefc_zero_dense(self):
    """put_data succeeds for dense models with nefc=0 and non-empty efc_J."""
    # A tendon with frictionloss causes MuJoCo to pre-allocate efc_J with
    # size nv even when nefc=0, causing reshape((0, nv)) to fail.
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body pos="0 0 1">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>
            <site name="s1" pos="0 0 0.1"/>
            <body pos="0.3 0 0">
              <joint type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.05" mass="0.2"/>
              <site name="s2" pos="0 0 -0.05"/>
            </body>
          </body>
        </worldbody>
        <tendon>
          <spatial limited="true" range="0 0.5"
            damping="2.0" stiffness="10.0" frictionloss="0.5">
            <site site="s1"/>
            <site site="s2"/>
          </spatial>
        </tendon>
      </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    self.assertFalse(mujoco.mj_isSparse(mjm))
    self.assertEqual(mjd.nefc, 0)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    if m.is_sparse:
      self.assertEqual(d.efc.J.shape[2], d.njmax * m.nv)
    else:
      self.assertEqual(d.efc.J.shape[2], m.nv_pad)

  def test_mesh_randomize_geom_level(self):
    """Test per-world mesh assignment for geom-level tuples."""
    nworld = 4
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    m, _ = per_world_mesh(spec, nworld)

    cube_geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    cube_s_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MESH, "cube_small")
    cube_l_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MESH, "cube_large")

    dataid = m.geom_dataid.numpy()

    self.assertEqual(dataid.shape, (nworld, m.ngeom))
    self.assertEqual(dataid[0, cube_geom_id], cube_s_id)
    self.assertEqual(dataid[1, cube_geom_id], cube_s_id)
    self.assertEqual(dataid[2, cube_geom_id], cube_l_id)
    self.assertEqual(dataid[3, cube_geom_id], cube_l_id)

  def test_mesh_randomize_dependent_fields(self):
    """Test that dependent per-world fields match compiled variant values."""
    nworld = 4
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    m, _ = per_world_mesh(spec, nworld)

    cube_geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    cube_body_id = mjm.geom_bodyid[cube_geom_id]

    geom = next(g for g in spec.geoms if g.name == "cube")
    geom.meshname = "cube_small"
    ref_s = spec.compile()
    geom.meshname = "cube_large"
    ref_l = spec.compile()

    geom_size = m.geom_size.numpy()
    geom_rbound = m.geom_rbound.numpy()
    body_mass = m.body_mass.numpy()

    np.testing.assert_allclose(geom_size[0, cube_geom_id], ref_s.geom_size[cube_geom_id], atol=1e-6)
    np.testing.assert_allclose(geom_rbound[0, cube_geom_id], ref_s.geom_rbound[cube_geom_id], atol=1e-6)
    np.testing.assert_allclose(body_mass[0, cube_body_id], ref_s.body_mass[cube_body_id], atol=1e-6)

    np.testing.assert_allclose(geom_size[2, cube_geom_id], ref_l.geom_size[cube_geom_id], atol=1e-6)
    np.testing.assert_allclose(geom_rbound[2, cube_geom_id], ref_l.geom_rbound[cube_geom_id], atol=1e-6)
    np.testing.assert_allclose(body_mass[2, cube_body_id], ref_l.body_mass[cube_body_id], atol=1e-6)

    self.assertNotAlmostEqual(
      float(geom_rbound[0, cube_geom_id]),
      float(geom_rbound[2, cube_geom_id]),
    )

  def test_mesh_randomize_body_level(self):
    """Test per-world mesh assignment for body-level tuples with padding."""
    nworld = 10
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    # default object has 2 geoms (variant B), ngeom = 3
    self.assertEqual(mjm.ngeom, 3)

    m, _ = per_world_mesh(spec, nworld)

    # per_world_mesh pads object to 3 geoms (variant A max), ngeom = 4
    self.assertEqual(m.ngeom, 4)

    # use padded model for ID lookups
    padded_mjm = spec.compile()

    dataid = m.geom_dataid.numpy()

    object_A_0_id = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_MESH, "object_A_0")
    object_A_1_id = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_MESH, "object_A_1")
    object_A_2_id = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_MESH, "object_A_2")
    object_B_0_id = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_MESH, "object_B_0")
    object_B_1_id = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_MESH, "object_B_1")

    object_col_0 = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_GEOM, "object_col_0")
    object_col_1 = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_GEOM, "object_col_1")
    # object_col_2 is the padded geom (unnamed, last geom)
    object_col_2 = m.ngeom - 1

    # prm=[0.6, 0.4], nworld=10 → worlds 0-5 get variant A, 6-9 get variant B
    for w in range(6):
      self.assertEqual(dataid[w, object_col_0], object_A_0_id)
      self.assertEqual(dataid[w, object_col_1], object_A_1_id)
      self.assertEqual(dataid[w, object_col_2], object_A_2_id)

    # Variant B: 2 pieces → geom slot 2 disabled (dataid = -1)
    for w in range(6, 10):
      self.assertEqual(dataid[w, object_col_0], object_B_0_id)
      self.assertEqual(dataid[w, object_col_1], object_B_1_id)
      self.assertEqual(dataid[w, object_col_2], -1)

  def test_mesh_randomize_backward_compat(self):
    """Models without tuples: per_world_mesh is a no-op."""
    spec = mujoco.MjSpec.from_string("""
    <mujoco>
      <worldbody>
        <body>
          <freejoint/>
          <geom type="sphere" size="0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    mjm = spec.compile()
    m, _ = per_world_mesh(spec, nworld=4)

    dataid = m.geom_dataid.numpy()
    self.assertEqual(dataid.shape[0], 1)
    self.assertEqual(dataid.shape[1], mjm.ngeom)

  # --- _allocate_worlds unit tests ---

  def test_allocate_worlds_rounding(self):
    """Largest remainder: prm=[0.5, 0.5], nworld=3 sums to exactly 3."""
    candidates = [(0, 0.5), (1, 0.5)]
    assignment = _allocate_worlds(candidates, nworld=3)
    self.assertEqual(len(assignment), 3)
    # each candidate should get at least 1 world
    self.assertGreaterEqual(assignment.count(0), 1)
    self.assertGreaterEqual(assignment.count(1), 1)

  def test_allocate_worlds_uniform(self):
    """Uniform fallback: prm=[0, 0, 0] assigns each at least 1 world."""
    candidates = [(0, 0.0), (1, 0.0), (2, 0.0)]
    assignment = _allocate_worlds(candidates, nworld=5)
    self.assertEqual(len(assignment), 5)
    for idx in range(3):
      self.assertGreaterEqual(assignment.count(idx), 1)

  def test_allocate_worlds_single_candidate(self):
    """Single candidate gets all worlds."""
    candidates = [(0, 1.0)]
    assignment = _allocate_worlds(candidates, nworld=10)
    self.assertEqual(len(assignment), 10)
    self.assertTrue(all(a == 0 for a in assignment))

  # --- per_world_mesh edge case tests ---

  def test_mesh_randomize_nworld_1(self):
    """Body-level randomization with nworld=1 doesn't crash."""
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    m, padded_mjm = per_world_mesh(spec, nworld=1)

    dataid = m.geom_dataid.numpy()
    self.assertEqual(dataid.shape[0], 1)
    # check it doesn't crash on forward
    d = mjwarp.make_data(padded_mjm)
    mjwarp.forward(m, d)

  def test_mesh_randomize_equal_variant_geoms(self):
    """No padding when all body variants have the same geom count."""
    xml = """
    <mujoco>
      <asset>
        <mesh name="a0" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
        <mesh name="a1" vertex="1 0 0  2 0 0  1 1 0  1 0 1"/>
        <mesh name="b0" vertex="0 0 0  3 0 0  0 3 0  0 0 3"/>
        <mesh name="b1" vertex="3 0 0  6 0 0  3 3 0  3 0 3"/>
      </asset>
      <worldbody>
        <body name="obj" pos="0 0 1">
          <freejoint/>
          <geom name="obj_0" type="mesh" mesh="a0"/>
          <geom name="obj_1" type="mesh" mesh="a1"/>
        </body>
      </worldbody>
      <custom>
        <tuple name="var_a">
          <element objtype="mesh" objname="a0" prm="0"/>
          <element objtype="mesh" objname="a1" prm="0"/>
        </tuple>
        <tuple name="var_b">
          <element objtype="mesh" objname="b0" prm="0"/>
          <element objtype="mesh" objname="b1" prm="0"/>
        </tuple>
        <tuple name="obj">
          <element objtype="tuple" objname="var_a" prm="0.5"/>
          <element objtype="tuple" objname="var_b" prm="0.5"/>
        </tuple>
      </custom>
    </mujoco>
    """
    spec = mujoco.MjSpec.from_string(xml)
    mjm = spec.compile()
    original_ngeom = mjm.ngeom

    m, _ = per_world_mesh(spec, nworld=4)

    # no padding should have occurred
    self.assertEqual(m.ngeom, original_ngeom)

  def test_mesh_randomize_mixed_geom_and_body(self):
    """Both geom-level and body-level tuples in the same model."""
    nworld = 10
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    m, _ = per_world_mesh(spec, nworld)

    dataid = m.geom_dataid.numpy()
    self.assertEqual(dataid.shape[0], nworld)

    # use padded model for ID lookups
    padded_mjm = spec.compile()

    # geom-level cube should have both small and large across worlds
    cube_geom_id = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    cube_s_id = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_MESH, "cube_small")
    cube_l_id = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_MESH, "cube_large")
    cube_variants = set(dataid[:, cube_geom_id])
    self.assertIn(cube_s_id, cube_variants)
    self.assertIn(cube_l_id, cube_variants)

    # body-level object should have variant A and B across worlds
    object_col_0 = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_GEOM, "object_col_0")
    object_A_0_id = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_MESH, "object_A_0")
    object_B_0_id = mujoco.mj_name2id(padded_mjm, mujoco.mjtObj.mjOBJ_MESH, "object_B_0")
    object_variants = set(dataid[:, object_col_0])
    self.assertIn(object_A_0_id, object_variants)
    self.assertIn(object_B_0_id, object_variants)

  def test_mesh_randomize_idempotent(self):
    """Calling per_world_mesh twice on the same spec produces same result."""
    nworld = 4
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    m1, _ = per_world_mesh(spec, nworld)
    dataid1 = m1.geom_dataid.numpy().copy()

    # reset spec and do it again
    spec2 = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm2 = spec2.compile()

    m2, _ = per_world_mesh(spec2, nworld)
    dataid2 = m2.geom_dataid.numpy()

    np.testing.assert_array_equal(dataid1, dataid2)

  def test_mesh_randomize_spec_not_mutated(self):
    """Spec is restored to original state after per_world_mesh."""
    nworld = 4
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    # record original mesh assignments
    orig_meshnames = {g.name: g.meshname for g in spec.geoms if g.name}

    m, _ = per_world_mesh(spec, nworld)

    # verify spec geoms were restored
    for g in spec.geoms:
      if g.name and g.name in orig_meshnames:
        self.assertEqual(g.meshname, orig_meshnames[g.name], f"meshname for geom {g.name} was mutated")

  def test_mesh_randomize_body_ipos_iquat(self):
    """Per-world body_ipos, body_iquat, geom_pos are propagated."""
    nworld = 4
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    m, _ = per_world_mesh(spec, nworld)

    # body_ipos should be (nworld, nbody, 3)
    body_ipos = m.body_ipos.numpy()
    self.assertEqual(body_ipos.shape[0], nworld)

    # body_iquat should be (nworld, nbody, 4)
    body_iquat = m.body_iquat.numpy()
    self.assertEqual(body_iquat.shape[0], nworld)

    # geom_pos should be (nworld, ngeom, 3)
    geom_pos = m.geom_pos.numpy()
    self.assertEqual(geom_pos.shape[0], nworld)

  def test_margin_multiccd_box_box(self):
    """MULTICCD + box-box with margin raises NotImplementedError."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="box" size=".1 .1 .1" margin="0.01"/>
          </body>
          <body pos="0 0 .5">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_margin_multiccd_box_mesh(self):
    """MULTICCD + box-mesh with margin raises NotImplementedError."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="box" size=".1 .1 .1" margin="0.01"/>
          </body>
          <body pos="0 0 .5">
            <freejoint/>
            <geom type="mesh" mesh="m"/>
          </body>
        </worldbody>
        <asset>
          <mesh name="m" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
        </asset>
      </mujoco>
    """)
    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_margin_multiccd_mesh_mesh(self):
    """MULTICCD + mesh-mesh with margin raises NotImplementedError."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="mesh" mesh="m" margin="0.01"/>
          </body>
          <body pos="0 0 .5">
            <freejoint/>
            <geom type="mesh" mesh="m"/>
          </body>
        </worldbody>
        <asset>
          <mesh name="m" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
        </asset>
      </mujoco>
    """)
    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_margin_box_box_nativeccd_disabled(self):
    """Box-box with margin and NATIVECCD disabled succeeds without error."""
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="box" size=".1 .1 .1" margin="0.01"/>
          </body>
          <body pos="0 0 .5">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    mjwarp.put_model(mjm)

  def test_margin_pair_box_box(self):
    """Pair with margin on box-box raises NotImplementedError."""
    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(
        mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body>
              <freejoint/>
              <geom name="b1" type="box" size=".1 .1 .1"/>
            </body>
            <body pos="0 0 .5">
              <freejoint/>
              <geom name="b2" type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
          <contact>
            <pair geom1="b1" geom2="b2" margin="0.01"/>
          </contact>
        </mujoco>
      """)
      )

  @parameterized.parameters(*_IO_TEST_MODELS)
  def test_kernel_recompilation(self, xml):
    """Test that subsequent steps do not trigger kernel recompilation."""
    _, _, m, d = test_data.fixture(xml)
    mjwarp.step(m, d)
    wp.synchronize()

    created_kernels = []
    original_init = wp.Kernel.__init__

    def _tracking_init(self_kernel, *args, **kwargs):
      res = original_init(self_kernel, *args, **kwargs)
      created_kernels.append(self_kernel.key)
      return res

    # Second step: with cache enabled, should trigger zero new kernel instantiations
    with mock.patch.object(wp.Kernel, "__init__", _tracking_init):
      mjwarp.step(m, d)
      wp.synchronize()

      self.assertEmpty(
        created_kernels,
        f"Kernels were re-created on a subsequent step call: {created_kernels}",
      )

  def test_flex_equality_sleep_error(self):
    """Verify loading flex equality with sleep raises NotImplementedError."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="b1"/>
        <body name="b2"/>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjm.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    mjm.eq_type[0] = mujoco.mjtEq.mjEQ_FLEX

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  @parameterized.parameters(
    mujoco.mjtSleepPolicy.mjSLEEP_NEVER,
    mujoco.mjtSleepPolicy.mjSLEEP_ALLOWED,
    mujoco.mjtSleepPolicy.mjSLEEP_INIT,
  )
  def test_tree_sleep_policy_error(self, policy):
    """Verify loading a model with an unsupported sleep policy raises NotImplementedError."""
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm, _, _, _ = test_data.fixture(xml=xml)
    mjm.tree_sleep_policy[0] = policy

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_reset_data_sleep(self):
    """Verify resetting sleep-related fields on a multi-world setup."""
    mjm, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
    </mujoco>
    """,
      nworld=2,
    )

    # Set non-default sleep states (asleep) on both worlds
    # a tree is asleep if tree_asleep >= 0
    # There is 1 tree in this model (since 1 body is dynamic)
    tree_asleep = np.array([[5], [10]], dtype=np.int32)
    wp.copy(d.tree_asleep, wp.array(tree_asleep, dtype=int))

    # Set different tree_awake, body_awake
    tree_awake = np.array([[0], [0]], dtype=np.int32)
    wp.copy(d.tree_awake, wp.array(tree_awake, dtype=int))

    body_awake = np.full((2, 2), types.SleepState.ASLEEP, dtype=np.int32)
    wp.copy(d.body_awake, wp.array(body_awake, dtype=int))

    # Reset world 0 only
    reset0 = wp.array([True, False], dtype=bool)
    mjwarp.reset_data(m, d, reset=reset0)

    # Assert world 0 sleep fields were reset to fully awake
    np.testing.assert_array_equal(
      d.tree_asleep.numpy()[0],
      [-(1 + types.MJ_MINAWAKE)],
    )
    np.testing.assert_array_equal(d.tree_awake.numpy()[0], [1])
    # Body 0 is static/world body (gets STATIC). Body 1 has slide joint (gets AWAKE).
    np.testing.assert_array_equal(
      d.body_awake.numpy()[0],
      [types.SleepState.STATIC, types.SleepState.AWAKE],
    )

    # Assert world 1 sleep fields were NOT modified
    np.testing.assert_array_equal(d.tree_asleep.numpy()[1], [10])
    np.testing.assert_array_equal(d.tree_awake.numpy()[1], [0])
    np.testing.assert_array_equal(
      d.body_awake.numpy()[1],
      [types.SleepState.ASLEEP, types.SleepState.ASLEEP],
    )

  def test_ls_parallel_deprecation(self):
    _, _, m, _ = test_data.fixture(xml="<mujoco/>")

    # Constructor with ls_parallel argument raises TypeError (unrecognized parameter)
    with self.assertRaises(TypeError):
      types.Option(timestep=wp.array([0.01], dtype=float), ls_parallel=True)

    # Accessing (reading) the fields raises AttributeError
    with self.assertRaises(AttributeError):
      _ = m.opt.ls_parallel
    with self.assertRaises(AttributeError):
      _ = m.opt.ls_parallel_min_step

    # Directly setting any value raises AttributeError
    with self.assertRaises(AttributeError):
      m.opt.ls_parallel = False
    with self.assertRaises(AttributeError):
      m.opt.ls_parallel_min_step = 0.0

    # Test that override_model raises ValueError with helpful message
    from mujoco_warp._src.io import override_model

    with self.assertRaisesRegex(ValueError, "ls_parallel was removed in MuJoCo Warp 3.9.1"):
      override_model(m, {"opt.ls_parallel": True})
    with self.assertRaisesRegex(ValueError, "ls_parallel_min_step was removed in MuJoCo Warp 3.9.1"):
      override_model(m, {"opt.ls_parallel_min_step": 0.01})

  def test_put_data_contact_batching(self):
    """Tests that contacts are batched correctly (tiled, not repeated) across worlds."""
    nworld = 2
    mjm, mjd, _, d = test_data.fixture("humanoid/humanoid.xml", keyframe=0, nworld=nworld)
    self.assertGreater(mjd.ncon, 1, "The test model must have at least 2 contacts")

    # For each contact field, verify that it is tiled correctly
    ncon = mjd.ncon
    for field in [
      "dist",
      "pos",
      "frame",
      "includemargin",
      "friction",
      "solref",
      "solreffriction",
      "solimp",
      "adhesion",
      "dim",
      "geom",
    ]:
      val_device = getattr(d.contact, field).numpy()
      val_host = getattr(mjd.contact, field)
      for w in range(nworld):
        device_slice = val_device[w * ncon : (w + 1) * ncon].reshape(-1)
        host_slice = val_host[:ncon].reshape(-1)
        _assert_eq(device_slice, host_slice, f"{field} mismatch in world {w}")

  def test_flex_interp_negative_success(self):
    """Test that put_model succeeds for negative flex_interp (shell elements)."""
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                  pos="0 0 0.5" name="cube" dim="3" mass="1" radius="0.005"
                  dof="trilinear">
          <contact selfcollide="none"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    # Modify flex_interp to be negative (simulating quad shells or unsupported order)
    mjm.flex_interp[0] = -1

    # Should succeed without NotImplementedError
    m = mjwarp.put_model(mjm)
    self.assertEqual(m.has_3d_flex, True)

  # TODO(team): remove after implementing multicontact support for CCD pairs.
  @parameterized.parameters(
    ("cylinder", "box"),
    ("cylinder", "cylinder"),
    ("cylinder", "mesh"),
    ("capsule", "cylinder"),
    ("capsule", "mesh"),
  )
  def test_unsupported_multiccd_warning(self, geom1_type, geom2_type):
    """Tests warning for unsupported multicontact CCD pairs when MULTICCD is enabled."""

    def _make_geom_xml(gtype: str) -> str:
      if gtype == "mesh":
        return '<geom type="mesh" mesh="m"/>'
      elif gtype in ("cylinder", "capsule"):
        return f'<geom type="{gtype}" size=".1 .1"/>'
      elif gtype == "sphere":
        return '<geom type="sphere" size=".1"/>'
      else:
        return f'<geom type="{gtype}" size=".1 .1 .1"/>'

    mesh_asset = '<mesh name="m" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>' if "mesh" in (geom1_type, geom2_type) else ""
    xml = f"""
      <mujoco>
        <asset>
          {mesh_asset}
        </asset>
        <worldbody>
          <body>
            <freejoint/>
            {_make_geom_xml(geom1_type)}
          </body>
          <body pos="0 0 .5">
            <freejoint/>
            {_make_geom_xml(geom2_type)}
          </body>
        </worldbody>
      </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)

    with self.assertWarns(UserWarning):
      mjwarp.put_model(mjm)

    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      mjwarp.put_model(mjm)

  def test_flex_internal_collision(self):
    """Test that flex internal collision raises NotImplementedError."""
    xml = """
      <mujoco>
        <worldbody>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" internal="true" margin="0.05"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """
    with self.assertRaises(NotImplementedError):
      test_data.fixture(xml=xml)


if __name__ == "__main__":
  wp.init()
  absltest.main()
