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
from mujoco_warp._src import warp_util
from mujoco_warp._src.io import put_model
from mujoco_warp._src.io import set_length_range
from mujoco_warp._src.types import SPARSE_CONSTRAINT_JACOBIAN


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

  # Save spec state so we can restore after compilation
  saved_geom_state = {}
  for g in spec.geoms:
    if g.name:
      saved_geom_state[g.name] = (
        g.meshname,
        g.contype,
        g.conaffinity,
      )

  # Compile each unique variant to get reference field values
  compiled_variants = {}  # key -> compiled MjModel
  for key, first_world in unique_rows.items():
    # Apply this variant's mesh assignments to the spec
    for geom_id, candidates in geom_variants.items():
      mesh_id = dataid_table[first_world, geom_id]
      if mesh_id >= 0:
        mesh_name = mujoco.mj_id2name(padded_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        geom_name = mujoco.mj_id2name(padded_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        geom = next(g for g in spec.geoms if g.name == geom_name)
        geom.meshname = mesh_name

    for body_name, variants in body_variants.items():
      body = next(b for b in spec.bodies if b.name == body_name)
      mesh_geoms = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]
      # determine which variant this world uses
      mesh_geom_ids = [mujoco.mj_name2id(padded_model, mujoco.mjtObj.mjOBJ_GEOM, g.name) for g in mesh_geoms if g.name]
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
            break

    compiled_variants[key] = spec.compile()

  # Restore spec state
  for g in spec.geoms:
    if g.name and g.name in saved_geom_state:
      meshname, contype, conaffinity = saved_geom_state[g.name]
      g.meshname = meshname
      g.contype = contype
      g.conaffinity = conaffinity

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


def per_world_mesh(m, spec: mujoco.MjSpec, nworld: int):
  """Per-world mesh randomization from custom/tuple annotations."""
  model = spec.compile()

  # no-op if no tuples
  if model.ntuple == 0:
    return m

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
    return m

  m.geom_dataid = wp.array(dataid_table, dtype=int)

  # Populate dependent per-world fields from variant compilations
  _populate_dependent_fields(m, spec, model, dataid_table, nworld, geom_variants, body_variants)

  return m


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

# TODO: Add more cameras for testing projection and intrinsics
_CAMERA_TEST_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam1" pos="0 -3 2" xyaxes="1 0 0 0 0.6 0.8" resolution="64 64" output="rgb"/>
    <camera name="cam2" pos="0 3 2" xyaxes="-1 0 0 0 0.6 0.8" resolution="32 32" output="depth"/>
    <camera name="cam3" pos="3 0 2" xyaxes="0 1 0 -0.6 0 0.8" resolution="16 16" output="rgb depth"/>
    <geom type="plane" size="5 5 0.1"/>
    <geom type="sphere" size="0.5" pos="0 0 1"/>
  </worldbody>
</mujoco>
"""

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
  def test_make_put_data(self):
    """Tests that make_data and put_data are producing the same shapes for all arrays."""
    mjm, _, _, d = test_data.fixture("pendula.xml")
    md = mjwarp.make_data(mjm)

    # same number of fields
    self.assertEqual(len(d.__dict__), len(md.__dict__))

    # test shapes for all arrays
    for attr, val in md.__dict__.items():
      if isinstance(val, wp.array):
        self.assertEqual(val.shape, getattr(d, attr).shape, f"{attr} shape mismatch")

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

    mjd.qLD.fill(-123)
    mjd.qM.fill(-123)

    mjwarp.get_data_into(mjd, mjm, d)
    np.testing.assert_allclose(mjd.qLD, mjd_ref.qLD)
    np.testing.assert_allclose(mjd.qM, mjd_ref.qM)

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
    mjd.qM[:] = 0.0
    d = mjwarp.put_data(mjm, mjd)
    self.assertTrue((d.qLD.numpy() == 0.0).all())

    mujoco.mj_forward(mjm, mjd)
    mjd.qLD[:] = 0.0
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
      "qM",
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
        if arr == "qM":
          di_arr = di_arr.reshape(-1)[: mjd.qM.size]
        _assert_eq(di_arr, getattr(mjd, arr), arr)

    _assert_eq(d.nacon.numpy(), 0, "nacon")

    for arr in d.contact.__dataclass_fields__:
      _assert_eq(getattr(d.contact, arr).numpy(), 0.0, arr)

  def test_reset_data_world(self):
    """Tests per-world reset."""
    _MJCF = """
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="slide"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(_MJCF)
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

  def test_sdf(self):
    """Tests that an SDF can be loaded."""
    mjm, mjd, m, d = test_data.fixture("collision_sdf/cow.xml")

    self.assertIsInstance(m.oct_aabb, wp.array)
    self.assertEqual(m.oct_aabb.dtype, wp.vec3)
    self.assertEqual(len(m.oct_aabb.shape), 2)
    if m.oct_aabb.size > 0:
      self.assertEqual(m.oct_aabb.shape[1], 2)

  def test_implicit_integrator_fluid_model(self):
    """Tests for implicit integrator with fluid model."""
    with self.assertRaises(NotImplementedError):
      test_data.fixture(
        xml="""
        <mujoco>
          <option viscosity="1" density="1" integrator="implicitfast"/>
          <worldbody>
            <body>
              <geom type="sphere" size=".1"/>
              <freejoint/>
            </body>
          </worldbody>
        </mujoco>
        """
      )

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

  def test_ls_parallel(self):
    _, _, m, _ = test_data.fixture(
      xml="""
    <mujoco>
    </mujoco>
    """
    )

    self.assertEqual(m.opt.ls_parallel, False)

    _, _, m, _ = test_data.fixture(
      xml="""
    <mujoco>
      <custom>
        <numeric data="1" name="ls_parallel"/>
      </custom>
    </mujoco>
    """
    )

    self.assertEqual(m.opt.ls_parallel, True)

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

  def test_set_const_qpos0_modification(self):
    """Test set_const recomputes fields after qpos0 modification."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="link1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <site name="s1" pos="0.1 0 0"/>
          <body name="link2" pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom name="g2" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
            <site name="s2" pos="0.4 0 0"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="tendon1">
          <site site="s1"/>
          <site site="s2"/>
        </spatial>
      </tendon>
    </mujoco>
    """
    )

    mjm.qpos0[:] = [0.3, 0.5]
    m.qpos0.numpy()[0, :] = [0.3, 0.5]

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")
    _assert_eq(m.tendon_invweight0.numpy()[0], mjm.tendon_invweight0, "tendon_invweight0")
    _assert_eq(m.tendon_length0.numpy()[0], mjm.tendon_length0, "tendon_length0")

  def test_set_const_body_mass_modification(self):
    """Test set_const recomputes fields after body_mass modification."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="link1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <body name="link2" pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom name="g2" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor1" joint="j1" gear="1"/>
        <motor name="motor2" joint="j2" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    new_mass = 3.0
    mjm.body_mass[1] = new_mass
    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = new_mass
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")
    _assert_eq(m.body_subtreemass.numpy()[0], mjm.body_subtreemass, "body_subtreemass")
    _assert_eq(m.actuator_acc0.numpy()[0], mjm.actuator_acc0, "actuator_acc0")
    _assert_eq(m.body_invweight0.numpy()[0, 1, 0], mjm.body_invweight0[1, 0], "body_invweight0")

  @parameterized.named_parameters(
    dict(testcase_name="dense", jacobian="dense"),
    dict(testcase_name="sparse", jacobian="sparse"),
  )
  def test_set_const_meaninertia(self, jacobian):
    """Test meaninertia computation matches MuJoCo after qpos0/mass changes."""
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
    <mujoco>
      <option jacobian="{jacobian}"/>
      <worldbody>
        <body name="link1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <body name="link2" pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom name="g2" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    # Test initial value matches
    _assert_eq(m.stat.meaninertia.numpy()[0], mjm.stat.meaninertia, "meaninertia initial")

    # Modify qpos0 and verify meaninertia updates
    new_qpos0 = np.array([0.5, 0.3])
    mjm.qpos0[:] = new_qpos0
    qpos0_np = m.qpos0.numpy()
    qpos0_np[0, :] = new_qpos0
    wp.copy(m.qpos0, wp.array(qpos0_np, dtype=m.qpos0.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.stat.meaninertia.numpy()[0], mjm.stat.meaninertia, "meaninertia after qpos0 change")

    # Modify body mass and verify meaninertia updates
    new_mass = 3.0
    mjm.body_mass[1] = new_mass
    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = new_mass
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.stat.meaninertia.numpy()[0], mjm.stat.meaninertia, "meaninertia after mass change")

  def test_set_const_freejoint(self):
    """Test set_const with freejoint (6 DOFs with special averaging)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="floating" pos="0 0 1">
          <freejoint/>
          <geom name="box" type="box" size="0.1 0.2 0.3" mass="2.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    new_mass = 5.0
    mjm.body_mass[1] = new_mass
    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = new_mass
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")
    _assert_eq(m.body_invweight0.numpy()[0, 1], mjm.body_invweight0[1], "body_invweight0")

  def test_set_const_balljoint(self):
    """Test set_const with ball joint (3 DOFs with averaging)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="arm">
          <joint name="ball" type="ball"/>
          <geom name="box" type="box" size="0.1 0.2 0.3" mass="2.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    new_inertia = np.array([0.1, 0.2, 0.3])
    mjm.body_inertia[1] = new_inertia
    body_inertia_np = m.body_inertia.numpy()
    body_inertia_np[0, 1] = new_inertia
    wp.copy(m.body_inertia, wp.array(body_inertia_np, dtype=m.body_inertia.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")

  def test_set_const_static_body(self):
    """Test set_const with static body (welded to world)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="static_body" pos="1 0 0">
          <geom name="static_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
        </body>
        <body name="dynamic_body">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom name="dynamic_geom" type="sphere" size="0.1" mass="2.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.body_invweight0.numpy()[0, 1], [0.0, 0.0], "body_invweight0")
    self.assertGreater(m.body_invweight0.numpy()[0, 2, 0], 0.0)
    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")

  def test_set_const_preserves_qpos(self):
    """Test that qpos is restored after set_const."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="mass">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom name="mass_geom" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    # Set qpos to a specific value
    mjd.qpos[0] = 0.5
    mujoco.mj_forward(mjm, mjd)
    d.qpos.numpy()[0, 0] = 0.5

    qpos_before = d.qpos.numpy().copy()
    mjwarp.set_const(m, d)

    _assert_eq(d.qpos.numpy(), qpos_before, "qpos")

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
      if isinstance(f.type, wp.array):
        # get fields
        arr = getattr(m, f.name)
        mj_arr = getattr(mjm, f.name)

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

  def test_set_fixed_body_subtreemass(self):
    """Test body_subtreemass accumulation for multi-level tree."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="root">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="sphere" size="0.1" mass="1.0"/>
          <body name="child1" pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom name="g2" type="sphere" size="0.1" mass="2.0"/>
            <body name="grandchild1" pos="0.5 0 0">
              <joint name="j3" type="hinge" axis="0 0 1"/>
              <geom name="g3" type="sphere" size="0.1" mass="3.0"/>
            </body>
          </body>
          <body name="child2" pos="0 0.5 0">
            <joint name="j4" type="hinge" axis="0 0 1"/>
            <geom name="g4" type="sphere" size="0.1" mass="4.0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    # Modify body masses and recompute
    mjm.body_mass[1] = 10.0  # root
    mjm.body_mass[2] = 20.0  # child1
    mjm.body_mass[3] = 30.0  # grandchild1
    mjm.body_mass[4] = 40.0  # child2

    body_mass_np = m.body_mass.numpy()
    body_mass_np[0, 1] = 10.0
    body_mass_np[0, 2] = 20.0
    body_mass_np[0, 3] = 30.0
    body_mass_np[0, 4] = 40.0
    wp.copy(m.body_mass, wp.array(body_mass_np, dtype=m.body_mass.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.body_subtreemass.numpy()[0], mjm.body_subtreemass, "body_subtreemass")

    # Verify: root=10+(20+30)+40=100, child1=20+30=50, grandchild1=30, child2=40
    np.testing.assert_allclose(m.body_subtreemass.numpy()[0, 1], 100.0, rtol=1e-6)
    np.testing.assert_allclose(m.body_subtreemass.numpy()[0, 2], 50.0, rtol=1e-6)
    np.testing.assert_allclose(m.body_subtreemass.numpy()[0, 3], 30.0, rtol=1e-6)
    np.testing.assert_allclose(m.body_subtreemass.numpy()[0, 4], 40.0, rtol=1e-6)

  def test_set_fixed_ngravcomp(self):
    """Test ngravcomp counting with gravcomp bodies."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="body1" gravcomp="1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="sphere" size="0.1" mass="1.0"/>
        </body>
        <body name="body2" pos="1 0 0" gravcomp="0">
          <joint name="j2" type="hinge" axis="0 0 1"/>
          <geom name="g2" type="sphere" size="0.1" mass="1.0"/>
        </body>
        <body name="body3" pos="2 0 0" gravcomp="1">
          <joint name="j3" type="hinge" axis="0 0 1"/>
          <geom name="g3" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    self.assertEqual(m.ngravcomp, mjm.ngravcomp)
    self.assertEqual(m.ngravcomp, 2)  # body1 and body3

  def test_set_const_camera_light_positions(self):
    """Test camera and light reference position computations."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="body1" pos="1 2 3">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="sphere" size="0.1" mass="1.0"/>
          <camera name="cam1" pos="0.1 0.2 0.3"/>
          <light name="light1" pos="0.4 0.5 0.6" dir="0 0 -1"/>
        </body>
        <body name="body2" pos="4 5 6">
          <joint name="j2" type="hinge" axis="0 0 1"/>
          <geom name="g2" type="sphere" size="0.1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.cam_pos0.numpy()[0, 0], mjm.cam_pos0[0], "cam_pos0")
    _assert_eq(m.cam_poscom0.numpy()[0, 0], mjm.cam_poscom0[0], "cam_poscom0")
    _assert_eq(m.cam_mat0.numpy()[0, 0].flatten(), mjm.cam_mat0[0], "cam_mat0")
    _assert_eq(m.light_pos0.numpy()[0, 0], mjm.light_pos0[0], "light_pos0")
    _assert_eq(m.light_poscom0.numpy()[0, 0], mjm.light_poscom0[0], "light_poscom0")
    _assert_eq(m.light_dir0.numpy()[0, 0], mjm.light_dir0[0], "light_dir0")

  def test_set_const_idempotent(self):
    """Test calling set_const twice gives same results."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="link1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <body name="link2" pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom name="g2" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor1" joint="j1" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    mjwarp.set_const(m, d)
    dof_invweight0_1 = m.dof_invweight0.numpy().copy()
    body_invweight0_1 = m.body_invweight0.numpy().copy()
    body_subtreemass_1 = m.body_subtreemass.numpy().copy()
    actuator_acc0_1 = m.actuator_acc0.numpy().copy()

    mjwarp.set_const(m, d)
    _assert_eq(m.dof_invweight0.numpy(), dof_invweight0_1, "dof_invweight0")
    _assert_eq(m.body_invweight0.numpy(), body_invweight0_1, "body_invweight0")
    _assert_eq(m.body_subtreemass.numpy(), body_subtreemass_1, "body_subtreemass")
    _assert_eq(m.actuator_acc0.numpy(), actuator_acc0_1, "actuator_acc0")

  def test_set_const_full_pipeline(self):
    """Test complete set_const matches MuJoCo for complex model."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="torso" pos="0 0 1">
          <freejoint/>
          <geom name="torso_geom" type="box" size="0.1 0.2 0.3" mass="10.0"/>
          <body name="arm" pos="0.2 0 0">
            <joint name="shoulder" type="ball"/>
            <geom name="arm_geom" type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" mass="2.0"/>
            <site name="arm_site" pos="0.15 0 0"/>
            <body name="forearm" pos="0.3 0 0">
              <joint name="elbow" type="hinge" axis="0 1 0"/>
              <geom name="forearm_geom" type="capsule" fromto="0 0 0 0.25 0 0" size="0.04" mass="1.0"/>
              <site name="hand_site" pos="0.25 0 0"/>
            </body>
          </body>
          <body name="leg" pos="0 0 -0.3">
            <joint name="hip" type="hinge" axis="0 1 0"/>
            <geom name="leg_geom" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.06" mass="3.0"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="arm_tendon">
          <site site="arm_site"/>
          <site site="hand_site"/>
        </spatial>
      </tendon>
      <actuator>
        <motor name="elbow_motor" joint="elbow" gear="1"/>
        <motor name="hip_motor" joint="hip" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    mjm.qpos0[7:11] = [0.9, 0.1, 0.1, 0.1]
    mjm.qpos0[11] = 0.5
    mjm.qpos0[12] = 0.3

    qpos0_np = m.qpos0.numpy()
    qpos0_np[0, 7:11] = [0.9, 0.1, 0.1, 0.1]
    qpos0_np[0, 11] = 0.5
    qpos0_np[0, 12] = 0.3
    wp.copy(m.qpos0, wp.array(qpos0_np, dtype=m.qpos0.dtype))

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    _assert_eq(m.body_subtreemass.numpy()[0], mjm.body_subtreemass, "body_subtreemass")
    _assert_eq(m.dof_invweight0.numpy()[0], mjm.dof_invweight0, "dof_invweight0")
    _assert_eq(m.tendon_invweight0.numpy()[0], mjm.tendon_invweight0, "tendon_invweight0")
    _assert_eq(m.tendon_length0.numpy()[0], mjm.tendon_length0, "tendon_length0")
    _assert_eq(m.actuator_acc0.numpy()[0], mjm.actuator_acc0, "actuator_acc0")

    for i in range(mjm.nbody):
      _assert_eq(m.body_invweight0.numpy()[0, i], mjm.body_invweight0[i], f"body_invweight0[{i}]")

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  def test_set_const_graph_capture(self):
    """Test that set_const_0 is compatible with CUDA graph capture."""
    _, _, m, d = test_data.fixture("humanoid/humanoid.xml", keyframe=0)

    with wp.ScopedCapture() as capture:
      mjwarp.set_const_0(m, d)
      # TODO(team): set_const_fixed

    wp.capture_launch(capture.graph)

  def test_set_const_actuator_acc0_per_world(self):
    """Test actuator_acc0 has 2D shape [nworld, nu] and values match MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body name="link1">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom name="g1" type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor1" joint="j1" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    acc0_np = m.actuator_acc0.numpy()
    self.assertEqual(acc0_np.ndim, 2)
    self.assertEqual(acc0_np.shape, (1, mjm.nu))
    _assert_eq(acc0_np[0], mjm.actuator_acc0, "actuator_acc0")

  def test_set_const_dampratio(self):
    """Test dampratio resolution for position actuator matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <body pos="0.5 0 0">
            <joint name="j2" type="hinge" axis="0 0 1"/>
            <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position joint="j1" kp="100" dampratio="1.0"/>
        <position joint="j2" kp="50" dampratio="0.5"/>
      </actuator>
    </mujoco>
    """
    )

    # Set new dampratio values (positive biasprm[2]) to exercise resolution
    new_dampratio = [2.0, 0.8]
    for i in range(mjm.nu):
      mjm.actuator_biasprm[i, 2] = new_dampratio[i]
    mujoco.mj_setConst(mjm, mjd)

    bp = m.actuator_biasprm.numpy()
    for i in range(mjm.nu):
      bp[0, i, 2] = new_dampratio[i]
    wp.copy(m.actuator_biasprm, wp.array(bp, dtype=m.actuator_biasprm.dtype))
    mjwarp.set_const(m, d)

    biasprm_np = m.actuator_biasprm.numpy()
    for i in range(mjm.nu):
      _assert_eq(
        biasprm_np[0, i, 2],
        mjm.actuator_biasprm[i, 2],
        f"actuator_biasprm[{i}][2]",
      )

  def test_set_const_dampratio_explicit_kv(self):
    """Test actuator with explicit negative kv is NOT modified by dampratio."""
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
        <general joint="j1" gainprm="100"
                 biastype="affine" biasprm="0 -100 -10"/>
      </actuator>
    </mujoco>
    """
    )

    mujoco.mj_setConst(mjm, mjd)
    mjwarp.set_const(m, d)

    biasprm_np = m.actuator_biasprm.numpy()
    _assert_eq(
      biasprm_np[0, 0, 2],
      mjm.actuator_biasprm[0, 2],
      "actuator_biasprm[0][2]",
    )

  def test_set_length_range_joint_limited(self):
    """Test set_length_range for joint-limited actuator matches joint range."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="j1" type="hinge" axis="0 0 1" limited="true" range="-90 90"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="j1" gear="2"/>
      </actuator>
    </mujoco>
    """
    )

    set_length_range(m, d)

    lr_np = m.actuator_lengthrange.numpy()
    # range stored in radians: [-pi/2, pi/2], gear=2 => [-pi, pi]
    expected_lo = mjm.jnt_range[0, 0] * 2.0
    expected_hi = mjm.jnt_range[0, 1] * 2.0
    np.testing.assert_allclose(lr_np[0, 0, 0], expected_lo, atol=1e-5)
    np.testing.assert_allclose(lr_np[0, 0, 1], expected_hi, atol=1e-5)

  def test_set_length_range_tendon_limited(self):
    """Test set_length_range for tendon-limited actuator matches tendon range."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
          <site name="s1" pos="0.1 0 0"/>
          <body pos="0.5 0 0">
            <joint type="hinge" axis="0 0 1"/>
            <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
            <site name="s2" pos="0.4 0 0"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="t1" limited="true" range="0.1 0.5">
          <site site="s1"/>
          <site site="s2"/>
        </spatial>
      </tendon>
      <actuator>
        <motor tendon="t1" gear="1"/>
      </actuator>
    </mujoco>
    """
    )

    set_length_range(m, d)

    lr_np = m.actuator_lengthrange.numpy()
    # tendon range is not in degrees, so [0.1, 0.5] stays as-is with gear=1
    expected_lo = mjm.tendon_range[0, 0]
    expected_hi = mjm.tendon_range[0, 1]
    np.testing.assert_allclose(lr_np[0, 0, 0], expected_lo, atol=1e-5)
    np.testing.assert_allclose(lr_np[0, 0, 1], expected_hi, atol=1e-5)

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

  @parameterized.parameters(1, 4)
  def test_bvh_creation(self, nworld):
    """Test that the BVH is created correctly for single world and multiple worlds."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=nworld)
    rc = mjwarp.create_render_context(mjm, nworld=nworld, cam_res=(64, 64), use_textures=False)

    self.assertIsNotNone(rc)
    self.assertEqual(rc.nrender, mjm.ncam)

    self.assertEqual(rc.lower.shape, (nworld * rc.bvh_ngeom,), "lower")
    self.assertEqual(rc.upper.shape, (nworld * rc.bvh_ngeom,), "upper")
    self.assertEqual(rc.group.shape, (nworld * rc.bvh_ngeom,), "group")
    self.assertEqual(rc.group_root.shape, (nworld,), "group_root")

    self.assertIsNotNone(rc.bvh_id)
    self.assertNotEqual(rc.bvh_id, 0, "bvh_id")

    group_np = rc.group.numpy()
    _assert_eq(group_np, np.repeat(np.arange(nworld), rc.bvh_ngeom), "render context group values")

  def test_output_buffers(self):
    """Test that the output rgb and depth buffers have correct shapes and addresses."""
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 24
    rc = mjwarp.create_render_context(mjm, cam_res=(width, height), render_rgb=True, render_depth=True)

    expected_total = 3 * width * height

    self.assertEqual(rc.nrender, 3, "nrender")
    self.assertEqual(rc.rgb_data.shape, (1, expected_total), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (1, expected_total), "depth_data")

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()
    _assert_eq(rgb_adr, [0, width * height, 2 * width * height], "rgb_adr")
    _assert_eq(depth_adr, [0, width * height, 2 * width * height], "depth_adr")

  def test_heterogeneous_camera(self):
    """Tests render context with different resolutions and output."""
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    cam_res = [(64, 64), (32, 32), (16, 16)]
    rc = mjwarp.create_render_context(mjm, cam_res=cam_res, render_rgb=True, render_depth=True)

    self.assertEqual(rc.nrender, 3, "nrender")
    _assert_eq(rc.cam_res.numpy(), cam_res, "cam_res")

    expected_total = 64 * 64 + 32 * 32 + 16 * 16
    self.assertEqual(rc.rgb_data.shape, (1, expected_total), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (1, expected_total), "depth_data")

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()
    _assert_eq(rgb_adr, [0, 64 * 64, 64 * 64 + 32 * 32], "rgb_adr")
    _assert_eq(depth_adr, [0, 64 * 64, 64 * 64 + 32 * 32], "depth_adr")

    # Test that results are same when reading from mjmodel fields loaded through xml
    rc_xml = mjwarp.create_render_context(mjm, render_rgb=True, render_depth=True)
    self.assertEqual(rc.rgb_data.shape, rc_xml.rgb_data.shape, "rgb_data")
    self.assertEqual(rc.depth_data.shape, rc_xml.depth_data.shape, "depth_data")
    _assert_eq(rc.rgb_adr.numpy(), rc_xml.rgb_adr.numpy(), "rgb_adr")
    _assert_eq(rc.depth_adr.numpy(), rc_xml.depth_adr.numpy(), "depth_adr")

  def test_cam_active_filtering(self):
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 32

    rc = mjwarp.create_render_context(mjm, cam_res=(width, height), cam_active=[True, False, True])

    self.assertEqual(rc.nrender, 2, "nrender")

    expected_total = 2 * width * height
    self.assertEqual(rc.rgb_data.shape, (1, expected_total), "rgb_data")

  def test_rgb_only_and_depth_only(self):
    """Test that disabling rgb or depth correctly reduces the shape and invalidates the address."""
    mjm, mjd, m, d = test_data.fixture(xml=_CAMERA_TEST_XML)
    width, height = 32, 32
    pixels = width * height

    rc = mjwarp.create_render_context(
      mjm,
      cam_res=(width, height),
      render_rgb=[True, False, True],
      render_depth=[False, True, True],
    )

    self.assertEqual(rc.rgb_data.shape, (1, 2 * pixels), "rgb_data")
    self.assertEqual(rc.depth_data.shape, (1, 2 * pixels), "depth_data")
    _assert_eq(rc.rgb_adr.numpy(), [0, -1, pixels], "rgb_adr")
    _assert_eq(rc.depth_adr.numpy(), [-1, 0, pixels], "depth_adr")
    _assert_eq(rc.render_rgb.numpy(), [True, False, True], "render_rgb")
    _assert_eq(rc.render_depth.numpy(), [False, True, True], "render_depth")

    # Test that results are same when reading from mjmodel fields loaded through xml
    rc_xml = mjwarp.create_render_context(mjm, cam_res=(width, height))
    self.assertEqual(rc.rgb_data.shape, rc_xml.rgb_data.shape, "rgb_data")
    self.assertEqual(rc.depth_data.shape, rc_xml.depth_data.shape, "depth_data")
    _assert_eq(rc.rgb_adr.numpy(), rc_xml.rgb_adr.numpy(), "rgb_adr")
    _assert_eq(rc.depth_adr.numpy(), rc_xml.depth_adr.numpy(), "depth_adr")
    _assert_eq(rc.render_rgb.numpy(), rc_xml.render_rgb.numpy(), "render_rgb")
    _assert_eq(rc.render_depth.numpy(), rc_xml.render_depth.numpy(), "render_depth")

  def test_render_context_with_textures(self):
    # TODO: remove after mjwarp depends on warp >= 1.12 in pyproject.toml
    if not hasattr(wp, "Texture2D"):
      self.skipTest("Skipping test that requires warp >= 1.12")
      return

    mjm, mjd, m, d = test_data.fixture("mug/mug.xml")
    rc = mjwarp.create_render_context(mjm, render_rgb=True, render_depth=True, use_textures=True)
    self.assertTrue(rc.use_textures, "use_textures")
    self.assertEqual(rc.textures.shape, (mjm.ntex,), "textures")

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

    if SPARSE_CONSTRAINT_JACOBIAN:
      self.assertEqual(d.efc.J.shape[2], d.njmax * m.nv)
    else:
      self.assertEqual(d.efc.J.shape[2], m.nv_pad)

  def test_mesh_randomize_geom_level(self):
    """Test per-world mesh assignment for geom-level tuples."""
    nworld = 4
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    m = mjwarp.put_model(mjm)
    m = per_world_mesh(m, spec, nworld)

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

    m = mjwarp.put_model(mjm)
    m = per_world_mesh(m, spec, nworld)

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

    m = mjwarp.put_model(mjm)
    m = per_world_mesh(m, spec, nworld)

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
    m = mjwarp.put_model(mjm)
    m = per_world_mesh(m, spec, nworld=4)

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

    m = mjwarp.put_model(mjm)
    m = per_world_mesh(m, spec, nworld=1)

    dataid = m.geom_dataid.numpy()
    self.assertEqual(dataid.shape[0], 1)
    # check it doesn't crash on forward
    # recompile from spec to get padded model (per_world_mesh may add geoms)
    mjm = spec.compile()
    d = mjwarp.make_data(mjm)
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

    m = mjwarp.put_model(mjm)
    m = per_world_mesh(m, spec, nworld=4)

    # no padding should have occurred
    self.assertEqual(m.ngeom, original_ngeom)

  def test_mesh_randomize_mixed_geom_and_body(self):
    """Both geom-level and body-level tuples in the same model."""
    nworld = 10
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    m = mjwarp.put_model(mjm)
    m = per_world_mesh(m, spec, nworld)

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

    m1 = mjwarp.put_model(mjm)
    m1 = per_world_mesh(m1, spec, nworld)
    dataid1 = m1.geom_dataid.numpy().copy()

    # reset spec and do it again
    spec2 = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm2 = spec2.compile()

    m2 = mjwarp.put_model(mjm2)
    m2 = per_world_mesh(m2, spec2, nworld)
    dataid2 = m2.geom_dataid.numpy()

    np.testing.assert_array_equal(dataid1, dataid2)

  def test_mesh_randomize_spec_not_mutated(self):
    """Spec is restored to original state after per_world_mesh."""
    nworld = 4
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    # record original mesh assignments
    orig_meshnames = {g.name: g.meshname for g in spec.geoms if g.name}

    m = mjwarp.put_model(mjm)
    m = per_world_mesh(m, spec, nworld)

    # verify spec geoms were restored
    for g in spec.geoms:
      if g.name and g.name in orig_meshnames:
        self.assertEqual(g.meshname, orig_meshnames[g.name], f"meshname for geom {g.name} was mutated")

  def test_mesh_randomize_body_ipos_iquat(self):
    """Per-world body_ipos, body_iquat, geom_pos are propagated."""
    nworld = 4
    spec = mujoco.MjSpec.from_string(_MESH_RANDOMIZE_XML)
    mjm = spec.compile()

    m = mjwarp.put_model(mjm)
    m = per_world_mesh(m, spec, nworld)

    # body_ipos should be (nworld, nbody, 3)
    body_ipos = m.body_ipos.numpy()
    self.assertEqual(body_ipos.shape[0], nworld)

    # body_iquat should be (nworld, nbody, 4)
    body_iquat = m.body_iquat.numpy()
    self.assertEqual(body_iquat.shape[0], nworld)

    # geom_pos should be (nworld, ngeom, 3)
    geom_pos = m.geom_pos.numpy()
    self.assertEqual(geom_pos.shape[0], nworld)


if __name__ == "__main__":
  wp.init()
  absltest.main()
