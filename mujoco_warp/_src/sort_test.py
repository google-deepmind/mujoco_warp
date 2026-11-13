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

"""Canonical contact and constraint ordering tests (without replaying stage outputs)."""

import dataclasses
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import collision_driver
from mujoco_warp._src import constraint
from mujoco_warp._src import sort
from mujoco_warp._src import types

_SCALAR_KEYS = ("pos", "margin", "D", "vel", "aref", "frictionloss")
_SORTED = mjw.DeterminismType.CONTACTS | mjw.DeterminismType.CONSTRAINT


class SortTest(parameterized.TestCase):
  @parameterized.product(nworld=(1, 2), field=_SCALAR_KEYS)
  def test_constraint_scalar_keys(self, nworld, field):
    """Each scalar breaks ties in order, including signed zero, infinities, and NaNs."""
    bits = np.array([0xFFC00001, 0xFF800000, 0xBF800000, 0x80000000, 0, 0x3F800000, 0x7F800000, 0x7FC00001], dtype=np.uint32)
    base = np.array([5, 2, 7, 0, 6, 1, 4, 3])
    values = np.stack([bits[np.roll(base, world)] for world in range(nworld)])
    ordered = np.where(values >> 31, ~values, values ^ np.uint32(0x80000000))
    expected = np.argsort(ordered, axis=1)
    # Earlier fields tie; later fields disagree with this field's order.
    index = _SCALAR_KEYS.index(field)
    data = [np.zeros_like(values) if i < index else values if i == index else ~values for i in range(len(_SCALAR_KEYS))]
    keys = sort._ConstraintKeys()
    keys.primary = wp.zeros((nworld, len(bits)), dtype=wp.vec4i)
    keys.J = wp.zeros((nworld, len(bits), 1), dtype=wp.uint32)
    keys.J_rownnz = wp.empty((nworld, 0), dtype=int)
    keys.pos = wp.array(data[0], dtype=wp.uint32)
    keys.margin = wp.array(data[1], dtype=wp.uint32)
    keys.D = wp.array(data[2], dtype=wp.uint32)
    keys.vel = wp.array(data[3], dtype=wp.uint32)
    keys.aref = wp.array(data[4], dtype=wp.uint32)
    keys.frictionloss = wp.array(data[5], dtype=wp.uint32)
    keys.nv = 1
    count = wp.full(nworld, len(bits), dtype=int)
    order = sort.permutation(count, len(bits), keys, sort._constraint_less)
    np.testing.assert_array_equal(order.numpy(), expected)
    if nworld == 2:
      self.assertFalse(np.array_equal(expected[0], expected[1]))

  @parameterized.product(nworld=(1, 2), jacobian=("dense", "sparse"), cone=("pyramidal", "elliptic"))
  def test_unused_contact_addresses(self, nworld, jacobian, cone):
    """Unused contact components cannot overwrite another contact's row key."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1" condim="1"/>
          <body pos="0 0 .09">
            <freejoint/>
            <geom size=".1" condim="1"/>
          </body>
          <body pos="1 0 .08">
            <freejoint/>
            <geom size=".1" condim="6"/>
          </body>
          <body pos="2 0 .07">
            <freejoint/>
            <geom size=".1" condim="6"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=8,
      njmax=64,
      overrides={"opt.jacobian": jacobian, "opt.cone": cone},
    )
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    for array in _arrays(d.contact):
      if array.size:
        values = array.numpy().copy()
        values[:count] = values[:count][::-1]
        array.assign(values)
    constraint.make_constraint(m, d)
    # Solver outputs also have to follow the row permutation, even though assembly doesn't set them.
    solved = (d.efc.force, d.efc.state, d.efc.island)
    for array in solved:
      array.assign(np.tile(np.arange(array.shape[1]) + 10, (nworld, 1)))
    before = [(a, wp.clone(a)) for a in _arrays(d.efc)]
    # CPU .numpy() arrays alias Warp storage: retain independent pre/post-sort snapshots.
    addresses = d.contact.efc_address.numpy().copy()
    sort.constraints(m, d)
    expected = [a.numpy().copy() for a, _ in before]
    dims = d.contact.dim.numpy()
    worlds = d.contact.worldid.numpy()
    sorted_addresses = d.contact.efc_address.numpy()
    for actual in [array.numpy() for array in solved]:
      for conid in range(count):
        nrow = max(1, 2 * (dims[conid] - 1)) if cone == "pyramidal" else dims[conid]
        np.testing.assert_array_equal(actual[worlds[conid], sorted_addresses[conid, :nrow]], addresses[conid, :nrow] + 10)
    for _ in range(3):
      for target, source in before:
        wp.copy(target, source)
      poisoned = addresses.copy()
      for conid in range(count):
        if dims[conid] == 1:
          other = np.flatnonzero((worlds[:count] == worlds[conid]) & (dims[:count] == 6))[0]
          poisoned[conid, 1:] = addresses[other, 0]
      d.contact.efc_address.assign(poisoned)
      sort.constraints(m, d)
      for (actual, _), value in zip(before, expected):
        np.testing.assert_array_equal(actual.numpy(), value)

  @parameterized.parameters("nxn", "sap_tile", "sap_segmented")
  def test_incremental_collision(self, broadphase):
    """Sorting preserves contacts appended for newly awakened bodies after the contact filter."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .09">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body pos="1 0 .08">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      nconmax=8,
      overrides={"opt.broadphase": broadphase, "opt.deterministic": mjw.DeterminismType.CONTACTS},
    )
    # Wake body 1 only on the second pass, so its new contact sorts *before* the old one.
    states = np.tile([types.SleepState.STATIC, types.SleepState.ASLEEP, types.SleepState.AWAKE], (2, 1))
    d.body_awake.assign(states)
    previous = wp.clone(d.body_awake)
    calls = []

    def contactfilter(m, data):
      calls.append(int(data.nacon.numpy()[0]))

    m.callback.contactfilter = contactfilter
    mjw.collision(m, d)
    self.assertEqual(d.nacon.numpy()[0], 2)
    states[:, 1] = types.SleepState.AWAKE
    d.body_awake.assign(states)
    collision_driver.collision(m, d, awake_prev=previous)
    self.assertEqual(calls, [2, 4])
    np.testing.assert_array_equal(d.contact.worldid.numpy()[:4], [0, 0, 1, 1])
    np.testing.assert_array_equal(d.contact.geom.numpy()[:4], [[0, 1], [0, 2], [0, 1], [0, 2]])
    expected = d.contact.pos.numpy()[:4].copy()
    mjw.collision(m, d)
    np.testing.assert_array_equal(d.contact.pos.numpy()[:4], expected)

  def test_sdf_contacts(self):
    """SDF samples with the same geomcollisionid get a geometric canonical order."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sdf_iterations="10" sdf_initpoints="40"/>
        <asset>
          <mesh name="cube"
                vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1
                        -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <geom type="sdf" mesh="cube"/>
          <body pos="0 0 1.8" euler="45 0 0">
            <freejoint/>
            <geom type="sdf" mesh="cube"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      nconmax=128,
      njmax=512,
      overrides={"opt.deterministic": _SORTED},
    )
    _assert_collision_repeat(self, m, d)

  def test_flex_contacts(self):
    """Real flex contacts retain their feature identity through both sorting stages."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" pos="0 0 .01" radius=".02">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      nconmax=128,
      njmax=512,
      overrides={"opt.deterministic": mjw.DeterminismType.ALL, "opt.jacobian": "sparse"},
    )
    _assert_collision_repeat(self, m, d)

  @parameterized.product(
    nworld=(1, 2, 4),
    jacobian=("dense", "sparse"),
    cone=("pyramidal", "elliptic"),
    mode=("constraint", "external", "standalone"),
  )
  def test_interleaved_constraints(self, nworld, jacobian, cone, mode):
    """Canonical rows do not assume CONTACTS implies collision actually sorted the input."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .09">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body pos="1 0 .08">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=8,
      njmax=32,
      qvel_noise=0.1,
      overrides={"opt.jacobian": jacobian, "opt.cone": cone},
    )
    qpos = d.qpos.numpy().copy()
    qvel = d.qvel.numpy().copy()
    for world in range(nworld):
      qpos[world, 2] -= 0.003 * world
      qvel[world] *= world + 1
    d.qpos.assign(qpos)
    d.qvel.assign(qvel)
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertEqual(count, 2 * nworld)
    contacts = [(array, array.numpy().copy()) for array in _arrays(d.contact) if array.size]
    # Do not test uninitialized put_data metadata.
    d.efc.force.zero_()
    d.efc.state.zero_()
    d.efc.island.zero_()
    m.opt.deterministic = mjw.DeterminismType.CONSTRAINT
    if mode != "constraint":
      m.opt.deterministic |= mjw.DeterminismType.CONTACTS
    m.opt.run_collision_detection = mode != "external"

    def stage():
      if mode == "external":
        mjw.fwd_position(m, d)
      else:
        constraint.make_constraint(m, d)

    def snapshot(active):
      result = {}
      nefc = d.nefc.numpy()
      ids = d.efc.id.numpy()
      geoms = d.contact.geom.numpy()
      worldids = d.contact.worldid.numpy()
      nrow = 4 if cone == "pyramidal" else 3
      np.testing.assert_array_equal(nefc, nrow * np.bincount(worldids[:active], minlength=nworld))
      rows = {
        "type": d.efc.type.numpy(),
        "pos": d.efc.pos.numpy(),
        "margin": d.efc.margin.numpy(),
        "D": d.efc.D.numpy(),
        "vel": d.efc.vel.numpy(),
        "aref": d.efc.aref.numpy(),
        "frictionloss": d.efc.frictionloss.numpy(),
        "Jqvel": d.efc.Jqvel.numpy(),
      }
      J = d.efc.J.numpy()
      if m.is_sparse:
        rownnz = d.efc.J_rownnz.numpy()
        rowadr = d.efc.J_rowadr.numpy()
        colind = d.efc.J_colind.numpy()
        blocks = _assert_jtdaj_blocks(self, d)
      for world, n in enumerate(nefc):
        for name, value in rows.items():
          result[world, name] = value[world, :n]
        result[world, "id"] = geoms[ids[world, :n]]
        if m.is_sparse:
          nnz = int(rownnz[world, :n].sum())
          result[world, "J_rownnz"] = rownnz[world, :n]
          result[world, "J_rowadr"] = rowadr[world, :n]
          result[world, "J"] = J[world, 0, :nnz]
          result[world, "J_colind"] = colind[world, 0, :nnz]
          result[world, "jtdaj_adr"], result[world, "jtdaj_nrow"] = blocks[world]
        else:
          result[world, "J"] = J[world, :n, : m.nv]
      addresses = d.contact.efc_address.numpy()
      for i in range(active):
        for address in addresses[i]:
          if address >= 0:
            self.assertEqual(ids[worldids[i], address], i)
      return {name: value.copy() for name, value in result.items()}

    # The captured stage has to read the contact order and the active counts of every replay.
    with wp.ScopedCapture() as capture:
      stage()
    for active in (count - 1, count):
      d.nacon.fill_(active)
      reference = None
      for shift in range(4):
        order = np.roll(np.arange(active)[::-1], shift)
        for array, value in contacts:
          shuffled = value.copy()
          shuffled[:active] = value[order]
          array.assign(shuffled)
        if reference is None:
          stage()
          reference = snapshot(active)
        else:
          wp.capture_launch(capture.graph)
          for name, value in snapshot(active).items():
            np.testing.assert_array_equal(value, reference[name], err_msg=str(name))
    if nworld > 1:
      self.assertFalse(np.array_equal(reference[0, "pos"], reference[1, "pos"]))

    for world in range(nworld):
      mjd.qpos[:] = qpos[world]
      mjd.qvel[:] = qvel[world]
      mujoco.mj_forward(mjm, mjd)
      for name, expected in (
        ("pos", mjd.efc_pos),
        ("margin", mjd.efc_margin),
        ("D", mjd.efc_D),
        ("vel", mjd.efc_vel),
        ("aref", mjd.efc_aref),
        ("frictionloss", mjd.efc_frictionloss),
      ):
        np.testing.assert_allclose(reference[world, name], expected, rtol=1e-5, atol=1e-5)
      actual = _dense_jacobian(m, d, world)
      expected = np.zeros_like(actual, dtype=np.float64)
      if m.is_sparse:
        mujoco.mju_sparse2dense(expected, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
      else:
        expected[:] = mjd.efc_J.reshape(expected.shape)
      np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.product(nworld=(1, 2), jacobian=("dense", "sparse"), cone=("pyramidal", "elliptic"))
  def test_forward_capture(self, nworld, jacobian, cone):
    """Real collision/constraint stages repeat exactly, with and without graph capture."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body name="a" pos="0 0 .09">
            <freejoint/>
            <geom type="box" size=".1 .12 .1"/>
          </body>
          <body name="b" pos="1 0 .08">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
          </body>
          <body pos="2 0 1">
            <joint name="j1" range=".1 1" frictionloss=".1"/>
            <geom size=".1"/>
          </body>
          <body pos="3 0 1">
            <joint name="j2" range=".2 1" frictionloss=".2"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <equality>
          <connect body1="a" body2="b" anchor=".5 0 .09"/>
          <joint joint1="j1" joint2="j2"/>
        </equality>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=32,
      njmax=128,
      qvel_noise=0.1,
      overrides={"opt.jacobian": jacobian, "opt.cone": cone},
    )
    qpos = d.qpos.numpy().copy()
    qvel = d.qvel.numpy().copy()
    if nworld == 2:
      qpos[1, 2] -= 0.002
      qvel[1] *= 1.5
    d.qpos.assign(qpos)
    d.qvel.assign(qvel)
    # Isolate sorting from Warp's deterministic-atomic workspaces in conditional solver subgraphs.
    m.opt.graph_conditional = False
    m.opt.iterations = 4
    initial = [(a, wp.clone(a)) for a in _arrays(d)]

    def restore():
      for target, value in initial:
        wp.copy(target, value)

    def snapshot():
      ncon = int(d.nacon.numpy()[0])
      nefc = d.nefc.numpy()
      result = {"qacc": d.qacc.numpy(), "qfrc_constraint": d.qfrc_constraint.numpy()}
      result["dof_frictionadr"] = _assert_friction_addresses(self, d)
      for name, array in vars(d.contact).items():
        if array.size:
          result["contact", name] = array.numpy()[:ncon]
      rows = {
        "type": d.efc.type.numpy(),
        "id": d.efc.id.numpy(),
        "pos": d.efc.pos.numpy(),
        "margin": d.efc.margin.numpy(),
        "D": d.efc.D.numpy(),
        "vel": d.efc.vel.numpy(),
        "aref": d.efc.aref.numpy(),
        "frictionloss": d.efc.frictionloss.numpy(),
        "force": d.efc.force.numpy(),
        "state": d.efc.state.numpy(),
      }
      if m.is_sparse:
        _assert_jtdaj_blocks(self, d)
      for world, n in enumerate(nefc):
        for name, value in rows.items():
          result[world, name] = value[world, :n]
        result[world, "J"] = _dense_jacobian(m, d, world)
      return {name: value.copy() for name, value in result.items()}

    m.opt.deterministic = mjw.DeterminismType.NONE
    mjw.forward(m, d)
    unsorted_qacc = d.qacc.numpy().copy()
    restore()
    m.opt.deterministic = mjw.DeterminismType.ALL
    mjw.forward(m, d)
    reference = snapshot()
    # Row reordering changes the solver's float32 reduction order.
    np.testing.assert_allclose(reference["qacc"], unsorted_qacc, rtol=5e-5, atol=5e-5)
    with wp.ScopedCapture() as capture:
      mjw.forward(m, d)
    for i in range(6):
      restore()
      if i % 2:
        wp.capture_launch(capture.graph)
      else:
        mjw.forward(m, d)
      for name, value in snapshot().items():
        np.testing.assert_array_equal(value, reference[name], err_msg=str(name))
    if nworld > 1:
      self.assertFalse(np.array_equal(reference["qacc"][0], reference["qacc"][1]))

  @parameterized.product(nworld=(1, 2), key=("geometry", "flex", "material", "surfacevel"))
  def test_contact_permutation(self, nworld, key):
    """All fields follow a canonical permutation, including equal pair/point IDs."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=16,
    )
    count = 8 * nworld
    # A model without flexes does not allocate the flex feature fields.
    d.contact.flex = wp.zeros(d.naconmax, dtype=wp.vec2i)
    d.contact.elem = wp.zeros(d.naconmax, dtype=wp.vec2i)
    d.contact.vert = wp.zeros(d.naconmax, dtype=wp.vec2i)
    contact = vars(d.contact)
    reference = {}
    for i, (name, array) in enumerate(contact.items()):
      # Distinct values in every field, including those that a canonical key must not use.
      value = array.numpy()
      reference[name] = ((np.arange(value.size) * 7 + 3 * i) % 97).reshape(value.shape).astype(value.dtype)
    reference["worldid"][:count] = np.repeat(np.arange(nworld), 8)
    reference["geom"][:count] = [0, 1]
    reference["geomcollisionid"][:count] = 0  # SDF/flex contacts need geometric tie breakers.
    reference["pos"][:count] = np.column_stack((np.tile(np.arange(8), nworld), np.zeros((count, 2))))
    reference["flex"][:count] = -1
    reference["elem"][:count] = -1
    reference["vert"][:count] = -1
    if key == "flex":
      reference["geom"][:count, 0] = -1
      reference["flex"][:count, 0] = 0
      reference["elem"][:count, 0] = np.tile(np.arange(8), nworld)
      reference["pos"][:count] = 0
    elif key != "geometry":
      # Same feature, point and frame, but different materials supplied by a contact filter.
      for name in ("pos", "frame", "dist", "dim", "type", "includemargin"):
        reference[name][:count] = 0
      if key == "material":
        reference["friction"][:count, 0] = np.tile(np.arange(8), nworld)
      else:
        for name in ("friction", "solref", "solreffriction", "solimp", "adhesion"):
          reference[name][:count] = 0
        reference["surfacevel"][:count, 0] = np.tile(np.arange(8), nworld)
    # The captured sort has to read the contact order and the active count of every replay.
    with wp.ScopedCapture() as capture:
      sort.contacts(d)
    for active in (count, count - 3, 2):
      d.nacon.fill_(active)
      for shift in (1, 3, 5):
        permutation = np.roll(np.arange(active)[::-1], shift)
        for name, value in reference.items():
          shuffled = value.copy()
          shuffled[:active] = value[permutation]
          contact[name].assign(shuffled)
        if shift == 1:
          sort.contacts(d)
        else:
          wp.capture_launch(capture.graph)
        for name, value in reference.items():
          np.testing.assert_array_equal(contact[name].numpy(), value, err_msg=name)

  @parameterized.product(nworld=(1, 2), jacobian=("dense", "sparse"), equality=("connect", "weld", "interleaved"))
  def test_equality_forces(self, nworld, jacobian, equality):
    """Sorting preserves equality components and the subtype layout used by force sensors."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity=".25 .5 -.75" tolerance="1e-10">
          <flag contact="disable"/>
        </option>
        <worldbody>
          <body name="a" pos="0 0 1" quat=".5 .5 .5 .5">
            <freejoint/>
            <geom type="box" size=".1 .12 .15" mass="1"/>
            <site name="sa" pos=".02 .03 .04"/>
          </body>
          <body name="b" pos="1 0 1" quat=".5 -.5 .5 .5">
            <freejoint/>
            <geom type="box" size=".12 .15 .1" mass="2"/>
            <site name="sb" pos="-.03 .02 .01"/>
          </body>
          <body name="c" pos="2 0 1" quat=".5 .5 -.5 .5">
            <freejoint/>
            <geom type="box" size=".15 .1 .12" mass="1.5"/>
            <site name="sc"/>
          </body>
          <body name="d" pos="3 0 1" quat=".5 .5 .5 -.5">
            <freejoint/>
            <geom type="box" size=".1 .15 .12" mass=".8"/>
            <site name="sd"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="b" anchor="1 0 1"/>
          <connect body1="a" anchor=".03125 -.0625 .015625"/>
          <weld body1="d" anchor="3 0 1"/>
          <connect body1="c" anchor="-.0625 .015625 .03125"/>
        </equality>
        <sensor>
          <force site="sa"/>
          <torque site="sa"/>
          <force site="sb"/>
          <torque site="sb"/>
          <force site="sc"/>
          <torque site="sc"/>
          <force site="sd"/>
          <torque site="sd"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=0,
      njmax=32,
      overrides={"opt.jacobian": jacobian},
    )
    # Unequal spatial loads and nonzero velocities expose swaps of both force and torque components.
    # Explicit masses of order one and moderate, binary-exact inputs keep float32 force cancellation
    # within the standard tolerance.
    qvel = d.qvel.numpy().copy()
    applied = d.xfrc_applied.numpy().copy()
    active = d.eq_active.numpy().copy()
    eqtype = {"connect": mujoco.mjtEq.mjEQ_CONNECT, "weld": mujoco.mjtEq.mjEQ_WELD}
    for world in range(nworld):
      for body in range(4):
        qvel[world, 6 * body : 6 * body + 3] = np.array([0.001953125, -0.00390625, 0.005859375]) * (1 + 0.25 * world)
      applied[world, 1:] = (
        np.arange(1, mjm.nbody)[:, None] * np.array([0.25, -0.5, 0.75, 0.03125, 0.0625, -0.09375]) * (1 + 0.25 * world)
      )
      if equality != "interleaved":
        active[world] = mjm.eq_type == eqtype[equality]
    if nworld == 2:
      # Different active group counts also change the starting offset of the weld partition.
      active[1, 3] = False
    d.qvel.assign(qvel)
    d.xfrc_applied.assign(applied)
    d.eq_active.assign(active)
    fields = {
      "qacc": d.qacc,
      "qfrc_constraint": d.qfrc_constraint,
      "cfrc_ext": d.cfrc_ext,
      "cfrc_int": d.cfrc_int,
      "sensordata": d.sensordata,
    }
    expected = {name: [] for name in fields}
    for world in range(nworld):
      mjd.qvel[:] = qvel[world]
      mjd.xfrc_applied[:] = applied[world]
      mjd.eq_active[:] = active[world]
      mujoco.mj_forward(mjm, mjd)
      mujoco.mj_rnePostConstraint(mjm, mjd)
      expected["qacc"].append(mjd.qacc.copy())
      expected["qfrc_constraint"].append(mjd.qfrc_constraint.copy())
      expected["cfrc_ext"].append(mjd.cfrc_ext.copy())
      expected["cfrc_int"].append(mjd.cfrc_int.copy())
      expected["sensordata"].append(mjd.sensordata.copy())

    # The XML interleaves welds and connects, so sorted assembly has to reorder whole groups.
    for deterministic in (mjw.DeterminismType.NONE, mjw.DeterminismType.CONSTRAINT):
      m.opt.deterministic = deterministic
      for array in fields.values():
        array.fill_(wp.inf)
      mjw.forward(m, d)
      for name, array in fields.items():
        np.testing.assert_allclose(array.numpy(), expected[name], rtol=1e-5, atol=1e-5, err_msg=name)
    if nworld == 2:
      for name, array in fields.items():
        self.assertFalse(np.array_equal(array.numpy()[0], array.numpy()[1]), msg=name)

    ids = d.efc.id.numpy()
    if m.is_sparse:
      blocks = _assert_jtdaj_blocks(self, d)
      rownnz = d.efc.J_rownnz.numpy()
      rowadr = d.efc.J_rowadr.numpy()
      colind = d.efc.J_colind.numpy()
    for world, n in enumerate(d.nefc.numpy()):
      eqids = sorted(np.flatnonzero(active[world]), key=lambda eqid: (mjm.eq_type[eqid], eqid))
      sizes = [3 if mjm.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT else 6 for i in eqids]
      np.testing.assert_array_equal(ids[world, :n], np.repeat(eqids, sizes))
      if m.is_sparse:
        starts, widths = blocks[world]
        np.testing.assert_array_equal(widths, sizes)
        # Every remapped block must still cover exactly one equality, with common sparse support.
        nnz = rownnz[world]
        adr = rowadr[world]
        cols = colind[world, 0]
        for start, width in zip(starts, widths):
          for row in range(start + 1, start + width):
            np.testing.assert_array_equal(cols[adr[row] : adr[row] + nnz[row]], cols[adr[start] : adr[start] + nnz[start]])

  @parameterized.parameters("dense", "sparse")
  def test_flex_equalities(self, jacobian):
    """Flex edge rows sharing an equality ID are ordered by their Jacobians, not allocation."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" pos="0 0 1">
            <edge equality="true"/>
            <contact contype="0" conaffinity="0"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      njmax=64,
      qpos_noise=0.01,
      qvel_noise=0.1,
      overrides={"opt.jacobian": jacobian},
    )
    mjw.forward(m, d)
    baseline = d.qacc.numpy().copy()
    m.opt.deterministic = mjw.DeterminismType.ALL
    reference = None
    for _ in range(4):
      mjw.forward(m, d)
      np.testing.assert_allclose(d.qacc.numpy(), baseline, rtol=5e-5, atol=5e-5)
      current = [_dense_jacobian(m, d, world) for world in range(d.nworld)]
      if reference is not None:
        np.testing.assert_array_equal(current, reference)
      reference = current

  @parameterized.parameters(1, 2)
  def test_flex_strain_blocks(self, nworld):
    """The strain modes of a cell stay one block, in assembly order, with its head row first."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag contact="disable"/>
        </option>
        <worldbody>
          <flexcomp name="cube" type="grid" count="3 3 3" spacing=".1 .1 .1" pos="0 0 .5" dim="3" radius=".005"
                    dof="trilinear" cellcount="2 2 1">
            <edge equality="strain"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      qvel_noise=0.05,
      overrides={"opt.jacobian": "sparse"},
    )
    if nworld == 2:
      qpos = d.qpos.numpy().copy()
      qpos[1] += 0.01 * np.arange(1, m.nq + 1) / m.nq
      d.qpos.assign(qpos)
    mjw.forward(m, d)
    baseline = d.qacc.numpy().copy()
    unsorted = [_dense_jacobian(m, d, world) for world in range(nworld)]
    ids = d.efc.id.numpy().copy()
    rowtype = d.efc.type.numpy().copy()
    m.opt.deterministic = mjw.DeterminismType.CONSTRAINT
    d.efc.jtdaj_adr.fill_(-1)
    d.efc.jtdaj_nrow.fill_(-1)
    d.qacc.fill_(wp.inf)
    mjw.forward(m, d)
    np.testing.assert_allclose(d.qacc.numpy(), baseline, rtol=5e-5, atol=5e-5)
    blocks = _assert_jtdaj_blocks(self, d)
    sorted_ids = d.efc.id.numpy()
    for world, n in enumerate(d.nefc.numpy()):
      self.assertGreater(n, 1)
      starts, widths = blocks[world]
      self.assertLen(starts, mjm.neq)
      self.assertGreater(widths.min(), 1)
      # Each cell is one equality: its block holds every row of that equality.
      for start, width in zip(starts, widths):
        np.testing.assert_array_equal(sorted_ids[world, start : start + width], sorted_ids[world, start])
      self.assertLen(np.unique(sorted_ids[world, starts]), len(starts))
      # Rows keep the mode order of their cell, and cells are ordered by equality ID.
      order = np.lexsort((ids[world, :n], rowtype[world, :n]))
      np.testing.assert_array_equal(sorted_ids[world, :n], ids[world, :n][order])
      np.testing.assert_array_equal(_dense_jacobian(m, d, world), unsorted[world][order])
    if nworld == 2:
      self.assertFalse(np.array_equal(baseline[0], baseline[1]))

  @parameterized.product(nworld=(1, 2), jacobian=("dense", "sparse"))
  def test_friction_addresses(self, nworld, jacobian):
    """DOF friction addresses follow the rows that sparse Newton adds to the Hessian diagonal."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <joint name="j0" axis="0 1 0" frictionloss=".3"/>
            <geom size=".1" pos="0 0 -.3"/>
            <body pos="0 0 -.6">
              <joint name="j1" axis="0 1 0" range=".1 1"/>
              <geom size=".1" pos="0 0 -.3"/>
              <body pos="0 0 -.6">
                <joint name="j2" axis="1 0 0" frictionloss=".2"/>
                <geom size=".1" pos="0 0 -.3"/>
                <body pos="0 0 -.6">
                  <joint name="j3" axis="0 1 0" frictionloss=".1"/>
                  <geom size=".1" pos="0 0 -.3"/>
                </body>
              </body>
            </body>
          </body>
        </worldbody>
        <equality>
          <joint joint1="j0" joint2="j3"/>
        </equality>
      </mujoco>
      """,
      nworld=nworld,
      njmax=16,
      qvel_noise=0.5,
      overrides={"opt.jacobian": jacobian},
    )
    qvel = d.qvel.numpy().copy()
    if nworld == 2:
      qvel[1] *= -1.5
      d.qvel.assign(qvel)
    expected = []
    for world in range(nworld):
      mjd.qvel[:] = qvel[world]
      mujoco.mj_forward(mjm, mjd)
      expected.append(mjd.qacc.copy())
    mjw.forward(m, d)
    np.testing.assert_allclose(d.qacc.numpy(), expected, rtol=1e-5, atol=1e-5)
    nefc = d.nefc.numpy()
    np.testing.assert_array_equal(d.nf.numpy(), 3)
    # One thread assembles the friction rows in DOF order, so they are already canonical. Reverse
    # the single-row constraints of each world, as racing device threads can, before sorting.
    for name, array in vars(d.efc).items():
      if name in ("jtdaj_adr", "jtdaj_nrow", "jtdaj_nblock", "Ma") or (m.is_sparse and name in ("J", "J_colind")):
        continue
      if array.size:
        values = array.numpy().copy()
        for world, n in enumerate(nefc):
          values[world, :n] = values[world, :n][::-1]
        array.assign(values)
    last = nefc[:, None] - 1
    address = d.dof_frictionadr.numpy()
    d.dof_frictionadr.assign(np.where(address >= 0, last - address, -1))
    if m.is_sparse:
      d.efc.jtdaj_adr.assign(
        np.where(np.arange(d.njmax) < d.efc.jtdaj_nblock.numpy()[:, None], last - d.efc.jtdaj_adr.numpy(), 0)
      )
    sort.constraints(m, d)
    address = _assert_friction_addresses(self, d)
    np.testing.assert_array_equal(np.sum(address >= 0, axis=1), 3)
    if m.is_sparse:
      _assert_jtdaj_blocks(self, d)
    d.qacc.fill_(wp.inf)
    mjw.solve(m, d)
    np.testing.assert_allclose(d.qacc.numpy(), expected, rtol=1e-5, atol=1e-5)
    if nworld == 2:
      self.assertFalse(np.array_equal(expected[0], expected[1]))

  @parameterized.parameters("rows", "nonzeros")
  def test_constraint_overflow(self, overflow):
    """Do not permute incomplete rows or alter the counters used to report overflow."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <joint frictionloss=".1"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      njmax=4,
      overrides={"opt.jacobian": "sparse"},
    )
    mjw.forward(m, d)
    nnz = wp.full(d.nworld, d.njmax_nnz + 1 if overflow == "nonzeros" else 1, dtype=int)
    if overflow == "rows":
      d.nefc.fill_(d.njmax + 1)
    before = [(a, a.numpy().copy()) for a in _arrays(d)]
    sort.constraints(m, d, nnz)
    for a, value in before:
      np.testing.assert_array_equal(a.numpy(), value)

  @parameterized.parameters(0, 1, 7, 8, 13)
  def test_contact_count_bounds(self, count):
    """Zero, partial, full, and overflowing buffers preserve counts and inactive slots."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nconmax=8,
    )
    d.nacon.fill_(count)
    d.contact.worldid.zero_()
    d.contact.geom.zero_()
    d.contact.geomcollisionid.zero_()
    positions = np.column_stack((np.arange(d.naconmax)[::-1], np.zeros((d.naconmax, 2)))).astype(np.float32)
    d.contact.pos.assign(positions)
    sort.contacts(d)
    n = min(count, d.naconmax)
    np.testing.assert_array_equal(d.contact.pos.numpy()[:n], positions[:n][::-1])
    np.testing.assert_array_equal(d.contact.pos.numpy()[n:], positions[n:])
    self.assertEqual(d.nacon.numpy()[0], count)

  @parameterized.product(nworld=(1, 2), cone=("elliptic", "pyramidal"), jacobian=("dense", "sparse"))
  def test_repeated_mixed_dimension_contacts(self, nworld, cone, jacobian):
    """Sorting must not transfer old rigid rows to unused flex address components."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity=".25 .5 -.75" iterations="30" tolerance="1e-9">
          <flag warmstart="disable"/>
        </option>
        <worldbody>
          <geom type="plane" size="5 5 .1" condim="1"/>
          <body pos="-1 0 .095">
            <freejoint/>
            <geom size=".1" condim="1" priority="1" margin=".01" gap=".04"/>
          </body>
          <body pos="1 0 .095">
            <freejoint/>
            <geom size=".1" condim="6" priority="1" friction=".5 .5 .5" margin=".01" gap=".04"/>
          </body>
          <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".1 .1 .1" pos="0 0 .004" dim="2">
            <contact condim="3" selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=32,
      njmax=128,
      overrides={"opt.cone": cone, "opt.jacobian": jacobian},
    )
    m.opt.deterministic = _SORTED
    qpos = d.qpos.numpy().copy()
    qvel = np.full((nworld, m.nv), 0.03125, dtype=np.float32)
    if nworld == 2:
      qpos[1, 2] += 0.002
      qpos[1, 9] -= 0.001
      qvel[1] = -0.0625
    d.qpos.assign(qpos)
    d.qvel.assign(qvel)
    expected = []
    for world in range(nworld):
      mjd = mujoco.MjData(mjm)
      mjd.qpos[:], mjd.qvel[:] = qpos[world], qvel[world]
      mujoco.mj_forward(mjm, mjd)
      expected.append(mjd.qacc.copy())
    if nworld == 2:
      self.assertFalse(np.array_equal(expected[0], expected[1]))

    # Start from unwritten output; subsequent calls reuse actual previous outputs.
    d.contact.efc_address.fill_(-1)
    for _ in range(3):
      mjw.forward(m, d)
      np.testing.assert_array_equal(d.overflow.numpy(), 0)
      ncon = int(d.nacon.numpy()[0])
      addresses = d.contact.efc_address.numpy().copy()[:ncon]
      dimensions = d.contact.dim.numpy().copy()[:ncon]
      for conid, condim in enumerate(dimensions):
        width = int(condim) if cone == "elliptic" else max(1, 2 * (int(condim) - 1))
        np.testing.assert_array_equal(addresses[conid, width:], -1)
      np.testing.assert_allclose(d.qacc.numpy(), expected, rtol=1e-5, atol=1e-5)

  @parameterized.product(nworld=(1, 2), cone=("elliptic", "pyramidal"), jacobian=("dense", "sparse"))
  def test_standalone_active_to_gap(self, nworld, cone, jacobian):
    """Assembly owns output addresses even when externally supplied contacts become inactive."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .095">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=4,
      njmax=16,
      overrides={"opt.cone": cone, "opt.jacobian": jacobian},
    )
    m.opt.deterministic = _SORTED
    if nworld == 2:
      qpos = d.qpos.numpy().copy()
      qpos[1, 2] -= 0.002
      d.qpos.assign(qpos)
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertGreater(count, 0)
    self.assertTrue(np.all(d.contact.efc_address.numpy()[:count, 0] >= 0))
    # The external producer updates contact inputs, not the previous assembly's output addresses.
    distance = d.contact.dist.numpy().copy()
    distance[:count] = d.contact.includemargin.numpy()[:count] + 0.01
    d.contact.dist.assign(distance)
    mjw.make_constraint(m, d)
    np.testing.assert_array_equal(d.nefc.numpy(), 0)
    np.testing.assert_array_equal(d.contact.efc_address.numpy()[:count], -1)

  @parameterized.product(nworld=(1, 2), cone=("elliptic", "pyramidal"), deterministic=(mjw.DeterminismType.NONE, _SORTED))
  def test_collision_clears_flex_addresses(self, nworld, cone, deterministic):
    """Fresh collision contacts have no associated constraint rows, just as in native MuJoCo."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1" condim="1"/>
          <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".1 .1 .1" pos="0 0 .004" dim="2">
            <contact condim="3" selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=16,
      njmax=64,
      overrides={"opt.cone": cone},
    )
    m.opt.deterministic = deterministic
    qvel = np.full((nworld, m.nv), 0.03125, dtype=np.float32)
    if nworld == 2:
      qvel[1] = -0.0625
    d.qvel.assign(qvel)
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertGreater(count, 0)
    self.assertTrue(np.all(d.contact.efc_address.numpy()[:count, 0] >= 0))
    mjw.collision(m, d)
    np.testing.assert_array_equal(d.contact.efc_address.numpy()[:count], -1)
    np.testing.assert_array_equal(d.overflow.numpy(), 0)
    native = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, native)
    mujoco.mj_collision(mjm, native)
    np.testing.assert_array_equal(native.contact.efc_address, -1)

  @parameterized.parameters(1, 2)
  def test_put_data_contact_ids(self, nworld):
    """Native contacts have a neutral Warp-only collision ID, including spare capacity."""
    _, mjd, _, d = _import_mj_contacts(nworld)
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
    deterministic=(mjw.DeterminismType.CONSTRAINT, _SORTED),
  )
  def test_imported_contact_constraints(self, nworld, mode, deterministic):
    """The first consumer can sort an untouched public import, without collision replay."""
    mjm, mjd, m, d = _import_mj_contacts(nworld)
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
    pos = d.efc.pos.numpy()
    rownnz = d.efc.J_rownnz.numpy()
    J = d.efc.J.numpy()
    for world in range(nworld):
      self.assertTrue(np.all(np.isfinite(pos[world, :expected_nefc])))
      self.assertTrue(np.all(np.isfinite(J[world, 0, : int(rownnz[world, :expected_nefc].sum())])))

  @parameterized.product(nworld=(1, 3), capacity=(5, 64, 200), large=(False, True))
  def test_permutation_blocks(self, nworld, capacity, large):
    """Counted blocks and merge passes are one stable sort for every block width and capacity."""
    # Few distinct values, so that many rows compare equal and only their index orders them.
    primary = ((np.arange(nworld * capacity * 4, dtype=np.int32) * 7 + 3) % 3).reshape(nworld, capacity, 4)
    counts = np.array([capacity - 1, capacity // 2, 0])[:nworld]
    keys = sort._ConstraintKeys()
    keys.primary = wp.array(primary, dtype=wp.vec4i)
    keys.J = wp.zeros((nworld, capacity, 1), dtype=wp.uint32)
    keys.J_rownnz = wp.empty((nworld, 0), dtype=int)
    zeros = wp.zeros((nworld, capacity), dtype=wp.uint32)
    keys.pos, keys.margin, keys.D, keys.vel, keys.aref, keys.frictionloss = (zeros,) * 6
    keys.nv = 1
    inverse = wp.empty((nworld, capacity), dtype=int)
    with mock.patch.object(sort, "_RANK_LARGE", 0 if large else sort._RANK_LARGE):
      # Wide blocks and no merge pass for a small capacity, narrow blocks and merges otherwise.
      width = sort._rank_width(nworld, capacity)
      self.assertEqual(width, 16 if large and capacity > 16 else min(capacity, 64) if capacity != 5 else 8)
      order = sort.permutation(wp.array(counts, dtype=int), capacity, keys, sort._constraint_less, inverse)
    expected = np.tile(np.arange(capacity), (nworld, 1))
    for world, n in enumerate(counts):
      rows = primary[world, :n]
      expected[world, :n] = np.lexsort((np.arange(n), rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
    np.testing.assert_array_equal(order.numpy(), expected)
    np.testing.assert_array_equal(np.take_along_axis(inverse.numpy(), expected, axis=1), expected * 0 + np.arange(capacity))

  @parameterized.product(nworld=(1, 3), buckets=(True, False))
  def test_contact_paths(self, nworld, buckets):
    """Per-world buckets and merge passes over the buffer give the same stable canonical order."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=100,
    )
    self.assertEqual(d.naconmax <= sort._BUCKET_NCONMAX * nworld, True)
    count = d.naconmax - 7
    contact = {name: array for name, array in vars(d.contact).items() if array.size}
    reference = {}
    for i, (name, array) in enumerate(contact.items()):
      value = array.numpy()
      # Few distinct values with a different stride per field, so that contacts tie on leading keys.
      reference[name] = ((np.arange(value.size) * (2 * i + 3) % 101) % 3).reshape(value.shape).astype(value.dtype)
    # IDs outside the batch are ordered like any other world ID.
    reference["worldid"] = ((np.arange(d.naconmax, dtype=np.int32) * 3) % (nworld + 4)) - 2
    # Contacts with equal keys keep the order of the buffer: copy the keys of some contacts.
    pairs = np.arange(2 * (count // 3)).reshape(2, -1).T
    for name, value in reference.items():
      value[pairs[:, 1]] = value[pairs[:, 0]]
    # Not a key: it identifies the contact that each sorted slot came from.
    reference["efc_address"] = np.arange(reference["efc_address"].size, dtype=np.int32).reshape(reference["efc_address"].shape)
    for name, value in reference.items():
      contact[name].assign(value)
    d.nacon.fill_(count)
    with mock.patch.object(sort, "_BUCKET_NCONMAX", sort._BUCKET_NCONMAX if buckets else 0):
      sort.contacts(d)

    # Least significant key first: the index, then the keys of _contact_less from last to first.
    columns = [np.arange(count)]
    for name in (
      "geomcollisionid",
      "surfacevel",
      "adhesion",
      "solimp",
      "solreffriction",
      "solref",
      "friction",
      "includemargin",
      "type",
      "dim",
      "dist",
      "frame",
      "pos",
      "geom",
      "worldid",
    ):
      value = reference[name][:count].reshape(count, -1)
      columns.extend(value[:, i] for i in reversed(range(value.shape[1])))
    order = np.lexsort(columns)
    self.assertFalse(np.array_equal(order, np.arange(count)))
    for name, value in reference.items():
      expected = value.copy()
      expected[:count] = value[:count][order]
      np.testing.assert_array_equal(contact[name].numpy(), expected, err_msg=name)
    self.assertEqual(d.nacon.numpy()[0], count)

  def test_contact_bucket_ties(self):
    """Equal contacts keep the order of the buffer, not the order in which a bucket was filled."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      nconmax=4,
    )
    count = 6
    d.nacon.fill_(count)
    for array in _arrays(d.contact):
      array.zero_()
    # Three equal contacts per world. The grouped copy holds them in reverse buffer order.
    source = np.array([2, 1, 0, 5, 4, 3, 0, 0], dtype=np.int32)
    worldid = np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=np.int32)
    address = np.arange(d.contact.efc_address.size, dtype=np.int32).reshape(d.contact.efc_address.shape)
    copy = types.Contact(*(wp.clone(array) for array in _arrays(d.contact)))
    copy.worldid.assign(worldid)
    copy.efc_address.assign(address[source])
    wp.launch(
      sort._contact_place,
      dim=d.naconmax,
      inputs=[
        d.nacon,
        sort._contact_keys(copy),
        wp.array([0, 0, 3, 6], dtype=int),
        wp.array([0, 3, 3, 0], dtype=int),
        wp.array(source, dtype=int),
        sort._contact_rows(copy),
      ],
      outputs=[sort._contact_rows(d.contact)],
    )
    np.testing.assert_array_equal(d.contact.efc_address.numpy()[:count], address[:count])
    np.testing.assert_array_equal(d.contact.worldid.numpy()[:count], worldid[:count])
    np.testing.assert_array_equal(d.contact.efc_address.numpy()[count:], 0)

  @parameterized.parameters(True, False)
  def test_sparse_move(self, scan):
    """Packed nonzeros follow the sorted rows across threads, empty rows and rows that stay."""
    nworld, njmax, capacity = 2, 7, 28
    counts = np.array([7, 4], dtype=np.int32)
    order = np.array([[3, 0, 6, 1, 2, 5, 4], [0, 1, 3, 2, 4, 5, 6]], dtype=np.int32)
    rownnz = np.array([[0, 5, 3, 2, 0, 6, 1], [4, 0, 3, 2, 9, 9, 9]], dtype=np.int32)
    # Rows are stored in an arbitrary order. The first rows of the second world are in place and
    # its last row moves, so that nonzeros past the end of a world would move as well.
    rowadr = np.array([[9, 12, 0, 3, 9, 5, 11], [0, 4, 8, 4, 0, 0, 0]], dtype=np.int32)
    colind = ((np.arange(nworld * capacity, dtype=np.int32) * 17 + 5) % 100).reshape(nworld, capacity)
    J = (np.arange(1, nworld * capacity + 1, dtype=np.float32) * 0.125).reshape(nworld, capacity)
    expected_colind, expected_J = colind.copy(), J.copy()
    local = np.zeros((nworld, njmax), dtype=np.int32)
    sorted_nnz = np.zeros((nworld + 1, njmax), dtype=np.int32)
    for world in range(nworld):
      offset = 0
      for row in range(counts[world]):
        old = order[world, row]
        nnz = rownnz[world, old]
        local[world, row] = offset
        sorted_nnz[world, row] = nnz
        expected_colind[world, offset : offset + nnz] = colind[world, rowadr[world, old] : rowadr[world, old] + nnz]
        expected_J[world, offset : offset + nnz] = J[world, rowadr[world, old] : rowadr[world, old] + nnz]
        offset += nnz
    self.assertTrue(np.any(rowadr[1, order[1, :2]] == local[1, :2]))
    if scan:
      # One exclusive scan over all worlds, which ends in a row of zeros.
      flat = sorted_nnz.reshape(-1)
      offsets = wp.array((np.cumsum(flat) - flat).reshape(sorted_nnz.shape), dtype=int)
      total = wp.empty(0, dtype=int)
    else:
      offsets = wp.empty((nworld, njmax), dtype=int)
      total = wp.empty(nworld, dtype=int)
      wp.launch(
        sort._sparse_offsets,
        dim=nworld,
        inputs=[wp.array(rownnz, dtype=int), wp.array(counts, dtype=int), wp.array(order, dtype=int)],
        outputs=[offsets, total],
      )
      np.testing.assert_array_equal(total.numpy(), sorted_nnz[:nworld].sum(axis=1))
      for world in range(nworld):
        np.testing.assert_array_equal(offsets.numpy()[world, : counts[world]], local[world, : counts[world]])
    colind_wp, J_wp = wp.array(colind, dtype=int), wp.array(J, dtype=float)
    # Elements that do not move are not copied: poison their slots in the copy.
    colind_sorted, J_sorted = wp.full(colind.shape, -1, dtype=int), wp.full(J.shape, wp.nan, dtype=float)
    inputs = [wp.array(rowadr, dtype=int), wp.array(counts, dtype=int), wp.array(order, dtype=int), offsets, total]
    chunks = (nworld, (capacity + sort._CHUNK - 1) // sort._CHUNK)
    wp.launch(sort._sparse_move, dim=chunks, inputs=inputs + [colind_wp, J_wp, False], outputs=[colind_sorted, J_sorted])
    self.assertTrue(np.any(colind_sorted.numpy() == -1))
    wp.launch(sort._sparse_move, dim=chunks, inputs=inputs + [colind_sorted, J_sorted, True], outputs=[colind_wp, J_wp])
    np.testing.assert_array_equal(colind_wp.numpy(), expected_colind)
    np.testing.assert_array_equal(J_wp.numpy(), expected_J)

  def test_dense_rows(self):
    """Wide rows move a few columns per thread, up to a width that is not a multiple of them."""
    counts = np.array([5, 3], dtype=np.int32)
    order = np.array([[1, 0, 2, 4, 3], [2, 1, 0, 3, 4]], dtype=np.int32)
    width = 2 * sort._CHUNK - 1
    value = np.arange(2 * 5 * width, dtype=np.uint32).reshape(2, 5, width)
    expected = value.copy()
    for world, n in enumerate(counts):
      expected[world, :n] = value[world, order[world, :n]]
    value_wp = wp.array(value, dtype=wp.uint32)
    # Rows that stay in place are not copied: poison the copy.
    copy = wp.full(value.shape, 0xFFFFFFFF, dtype=wp.uint32)
    count, order_wp = wp.array(counts, dtype=int), wp.array(order, dtype=int)
    wp.launch(sort._gather, dim=(2, 5, 2), inputs=[count, order_wp, value_wp], outputs=[copy])
    np.testing.assert_array_equal(copy.numpy()[0, 2], 0xFFFFFFFF)
    np.testing.assert_array_equal(copy.numpy()[0, 0], value[0, 1])
    wp.launch(sort._restore, dim=(2, 5, 2), inputs=[count, order_wp, copy], outputs=[value_wp])
    np.testing.assert_array_equal(value_wp.numpy(), expected)

  @parameterized.product(nworld=(1, 2), jacobian=("dense", "sparse"), ordered=(True, False))
  def test_contact_rank_paths(self, nworld, jacobian, ordered):
    """Rows take the same contact order from a walk of the rows of a world and from merge passes."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .09">
            <freejoint/>
            <geom type="box" size=".1 .12 .1"/>
          </body>
          <body pos="1 0 .08">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body pos="2 0 1">
            <joint range=".1 1" frictionloss=".1"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=16,
      njmax=64,
      overrides={"opt.jacobian": jacobian},
    )
    self.assertLessEqual(d.njmax, sort._ROW_RANK_NJMAX)
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertGreater(count, 2 * nworld)
    sort.contacts(d)
    if not ordered:
      # The canonical order in reverse, so that every contact row has to move.
      for array in _arrays(d.contact):
        if array.size:
          values = array.numpy().copy()
          values[:count] = values[:count][::-1]
          array.assign(values)
    constraint.make_constraint(m, d)
    before = d.efc.J.numpy().copy()
    initial = [(a, wp.clone(a)) for a in _arrays(d)]
    results = []
    for row_rank_njmax in (sort._ROW_RANK_NJMAX, 0):
      for target, value in initial:
        wp.copy(target, value)
      with mock.patch.object(sort, "_ROW_RANK_NJMAX", row_rank_njmax):
        sort.constraints(m, d)
      if not ordered:
        self.assertFalse(np.array_equal(d.efc.J.numpy(), before))
      results.append({name: array.numpy().copy() for name, array in vars(d.efc).items() if array.size})
      results[-1]["efc_address"] = d.contact.efc_address.numpy().copy()
      results[-1]["dof_frictionadr"] = d.dof_frictionadr.numpy().copy()
      # Contact rows follow the canonical order of their contacts, whatever their index.
      nefc = d.nefc.numpy()
      for world in range(nworld):
        rows = results[-1]["type"][world, : nefc[world]] >= int(types.ConstraintType.CONTACT_FRICTIONLESS)
        ids = results[-1]["id"][world, : nefc[world]][rows]
        self.assertGreater(ids.size, 4)
        np.testing.assert_array_equal(ids, np.sort(ids)[:: 1 if ordered else -1])
    for name, value in results[0].items():
      np.testing.assert_array_equal(value, results[1][name], err_msg=name)

  @parameterized.product(nworld=(1, 2), cone=("pyramidal", "elliptic"))
  def test_sparse_offsets_paths(self, nworld, cone):
    """A scan over all worlds and a walk of each world pack the same sorted sparse rows."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .09">
            <freejoint/>
            <geom type="box" size=".1 .12 .1"/>
          </body>
          <body pos="1 0 .08">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body pos="2 0 1">
            <joint range=".1 1" frictionloss=".1"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=16,
      njmax=64,
      overrides={"opt.jacobian": "sparse", "opt.cone": cone},
    )
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertGreater(count, 2 * nworld)
    # Unsorted contacts, so that rows and their nonzeros have to move.
    for array in _arrays(d.contact):
      if array.size:
        values = array.numpy().copy()
        values[:count] = values[:count][::-1]
        array.assign(values)
    constraint.make_constraint(m, d)
    before = d.efc.J.numpy().copy()
    initial = [(a, wp.clone(a)) for a in _arrays(d)]
    results = []
    for scan_max in (sort._SCAN_MAX, 0):
      for target, value in initial:
        wp.copy(target, value)
      with mock.patch.object(sort, "_SCAN_MAX", scan_max):
        sort.constraints(m, d)
      self.assertFalse(np.array_equal(d.efc.J.numpy(), before))
      _assert_jtdaj_blocks(self, d)
      results.append({name: array.numpy().copy() for name, array in vars(d.efc).items() if array.size})
      results[-1]["efc_address"] = d.contact.efc_address.numpy().copy()
      results[-1]["dense"] = np.array([_dense_jacobian(m, d, world) for world in range(nworld)], dtype=object)
    for name, value in results[0].items():
      if name == "dense":
        for a, b in zip(value, results[1][name]):
          np.testing.assert_array_equal(a, b)
      else:
        np.testing.assert_array_equal(value, results[1][name], err_msg=name)

  @parameterized.product(nworld=(1, 2), jacobian=("dense", "sparse"), path=("rows", "head", "merge", "placed"))
  def test_noncontact_jqvel(self, nworld, jacobian, path):
    """Rows other than contact rows hold no Jqvel after the sort, and contact rows keep theirs.

    The paths are the launches for every row, those that order the rows that precede sorted contact
    rows, as one block or by merging blocks, and contact rows that were placed before they were
    filled.
    """
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .09">
            <freejoint/>
            <geom type="box" size=".1 .12 .1"/>
          </body>
          <body pos="1 0 .08">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body pos="2 0 1">
            <joint range=".1 1" frictionloss=".1"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=16,
      njmax=64,
      overrides={"opt.jacobian": jacobian},
      qvel_noise=0.1,
    )
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    sort.contacts(d)
    arguments = []
    if path == "placed":
      # The contacts give their rows their canonical addresses. The sort waits for the stale values.
      m.opt.deterministic = _SORTED
      with mock.patch.object(sort, "constraints", lambda *args: arguments.extend(args[2:])):
        constraint.make_constraint(m, d)
      self.assertEqual(arguments[1].numpy()[0], 0)
    else:
      # The canonical order in reverse, so that every contact row has to move.
      for array in _arrays(d.contact):
        if array.size:
          values = array.numpy().copy()
          values[:count] = values[:count][::-1]
          array.assign(values)
      constraint.make_constraint(m, d)
    active = np.arange(d.njmax) < d.nefc.numpy()[:, None]
    contact = active & (d.efc.type.numpy() >= int(types.ConstraintType.CONTACT_FRICTIONLESS))
    # Only contact rows write Jqvel: the slots of the other rows hold the values of earlier steps.
    Jqvel = d.efc.Jqvel.numpy().copy()
    Jqvel[~contact] = 1.0
    d.efc.Jqvel.assign(Jqvel)
    worldid = d.contact.worldid.numpy()[:count, None]
    address = d.contact.efc_address.numpy()[:count].copy()
    if path != "rows":
      m.opt.deterministic = _SORTED
    head_njmax, lanes = (0, 1 << 30) if path == "merge" else (sort._HEAD_NJMAX, sort._LANES)
    launch, kernels = wp.launch, []

    def spy_launch(kernel, *args, **kwargs):
      kernels.append(kernel)
      return launch(kernel, *args, **kwargs)

    spies = (
      mock.patch.object(wp, "launch", spy_launch),
      mock.patch.object(sort, "_HEAD_NJMAX", head_njmax),
      mock.patch.object(sort, "_LANES", lanes),
    )
    with spies[0], spies[1], spies[2]:
      sort.constraints(m, d, *arguments)
    self.assertIn(sort._constraint_primary, kernels)
    self.assertEqual(sort._constraint_unsorted in kernels, path != "rows")
    self.assertEqual(sort._constraint_rank in kernels, path in ("head", "placed"))
    self.assertEqual(sort._contact_unsorted in kernels, path != "placed")
    if not m.is_sparse:
      self.assertEqual(sort._gather in kernels, path in ("rows", "merge"))
    Jqvel_sorted = d.efc.Jqvel.numpy()
    address_sorted = d.contact.efc_address.numpy()[:count]
    contact = active & (d.efc.type.numpy() >= int(types.ConstraintType.CONTACT_FRICTIONLESS))
    self.assertTrue(np.any(active & ~contact, axis=1).all())
    np.testing.assert_array_equal(Jqvel_sorted[active & ~contact], 0.0)
    # Unused slots are not rows: they are left alone.
    np.testing.assert_array_equal(Jqvel_sorted[~active], 1.0)
    used = address >= 0
    # Contact rows that were placed stay where they are. The others move.
    self.assertEqual(np.array_equal(address_sorted, address), path == "placed")
    self.assertTrue(np.any(Jqvel[worldid, address][used] != 0.0))
    np.testing.assert_array_equal(Jqvel_sorted[worldid, address_sorted][used], Jqvel[worldid, address][used])

  @parameterized.product(nworld=(1, 2), jacobian=("dense", "sparse"), cone=("pyramidal", "elliptic"), ordered=(True, False))
  def test_contact_rows(self, nworld, jacobian, cone, ordered):
    """Contact rows are allocated where the sort puts them, which changes no sorted result."""
    results, before = [], []
    real = sort.constraints

    def spy(m, d, nnz=None, unsorted=None):
      before.append((d.contact.efc_address.numpy().copy(), d.efc.J_rowadr.numpy().copy(), unsorted))
      return real(m, d, nnz, unsorted)

    _, _, m, d = test_data.fixture(
      xml=_ROWS_XML, nworld=nworld, nconmax=16, njmax=64, overrides={"opt.jacobian": jacobian, "opt.cone": cone}
    )
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    sort.contacts(d)
    # A frictionless contact follows contacts with friction in the buffer, not among the rows.
    dim = d.contact.dim.numpy()[:count]
    self.assertEqual(np.sum(dim == 1), nworld)
    self.assertTrue(np.all(dim[np.flatnonzero(dim == 1) - 1] > 1))
    if not ordered:
      # Neither in canonical order nor grouped by world, when there are two.
      order = np.concatenate([np.arange(0, count, 2), np.arange(1, count, 2)])[::-1]
      for array in _arrays(d.contact):
        if array.size:
          values = array.numpy().copy()
          values[:count] = values[:count][order]
          array.assign(values)
    m.opt.deterministic = _SORTED
    # Both paths assemble and sort the same state.
    initial = [(a, wp.clone(a)) for a in _arrays(d)]
    for placed in (True, False):
      for target, value in initial:
        wp.copy(target, value)
      with mock.patch.object(sort, "constraints", spy):
        if placed:
          constraint.make_constraint(m, d)
        else:
          with mock.patch.object(sort, "contact_rows", return_value=None):
            constraint.make_constraint(m, d)
      self.assertLen(before, len(results) + 1)
      address, rowadr, unsorted = before[-1]
      self.assertEqual(unsorted is not None, placed)
      rowtype = d.efc.type.numpy()
      nefc = d.nefc.numpy()
      self.assertTrue(np.all(nefc <= d.njmax))
      contact_rows = (rowtype >= int(types.ConstraintType.CONTACT_FRICTIONLESS)) & (np.arange(d.njmax) < nefc[:, None])
      self.assertGreater(contact_rows.sum(), 9 * nworld)
      self.assertGreater((~contact_rows & (np.arange(d.njmax) < nefc[:, None])).sum(), 2 * nworld)
      if placed and ordered:
        # The sort found every contact row, and its nonzeros, where the contacts had put them.
        np.testing.assert_array_equal(address[:count], d.contact.efc_address.numpy()[:count])
        if m.is_sparse:
          np.testing.assert_array_equal(rowadr[contact_rows], d.efc.J_rowadr.numpy()[contact_rows])
      else:
        self.assertFalse(np.array_equal(address[:count], d.contact.efc_address.numpy()[:count]))
      if m.is_sparse:
        _assert_jtdaj_blocks(self, d)
      results.append(_sorted_rows(m, d))
    for name, value in results[0].items():
      np.testing.assert_array_equal(value, results[1][name], err_msg=name)

  @parameterized.product(sparse=(False, True), unsorted=(0, 1))
  def test_contact_rows_place(self, sparse, unsorted):
    """Each contact takes the rows after those that precede it, in worlds without overflow."""
    njmax, njmax_nnz, capacity = 20, 200, 32
    worldid = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3], dtype=np.int32)
    dim = np.array([3, 1, 3, 1, 4, 3, 3, 1, 3, 1, 3], dtype=np.int32)
    active = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    rownnz = np.array([5, 2, 7, 3, 6, 4, 4, 2, 3, 3, 2], dtype=np.int32)
    nrow = np.where(dim == 1, 1, 2 * (dim - 1))
    ncon, nworld = len(worldid), 4
    # Rows and nonzeros of the constraints that precede the contacts of each world.
    row_base = np.array([2, 0, 1, 0], dtype=np.int32)
    nnz_base = np.array([3, 0, 2, 0], dtype=np.int32)
    address = np.full((ncon + 2, 6), -1, dtype=np.int32)
    efc_id = np.full((nworld, capacity), -7, dtype=np.int32)
    efc_rownnz = np.full((nworld, capacity), -7, dtype=np.int32)
    efc_rowadr = np.full((nworld, capacity), -7, dtype=np.int32)
    nefc, nnz = row_base.copy(), nnz_base.copy()
    # Contacts allocate in the reverse order of the buffer.
    for conid in range(ncon)[::-1]:
      if active[conid]:
        world = worldid[conid]
        for i in range(nrow[conid]):
          address[conid, i] = nefc[world] + i
          efc_id[world, nefc[world] + i] = conid
          efc_rownnz[world, nefc[world] + i] = rownnz[conid]
          efc_rowadr[world, nefc[world] + i] = nnz[world] + i * rownnz[conid]
        nefc[world] += nrow[conid]
        nnz[world] += nrow[conid] * rownnz[conid]
    self.assertTrue(np.all(nefc <= njmax))
    # Worlds that lost rows or nonzeros, with room in the arrays for the rows they claim.
    nefc[2] = njmax + 1
    nnz[3] = njmax_nnz + 1
    expected_address, expected_id = address.copy(), efc_id.copy()
    expected_rownnz, expected_rowadr = efc_rownnz.copy(), efc_rowadr.copy()
    placed = [0, 1] if sparse else [0, 1, 3]
    for world in placed if not unsorted else []:
      row, adr = row_base[world], nnz_base[world]
      contacts = np.flatnonzero((worldid == world) & active)
      # Frictionless contacts first, each group in the order of the buffer.
      for conid in sorted(contacts, key=lambda c: (dim[c] != 1, c)):
        for i in range(nrow[conid]):
          expected_address[conid, i] = row + i
          expected_id[world, row + i] = conid
          expected_rownnz[world, row + i] = rownnz[conid]
          expected_rowadr[world, row + i] = adr + i * rownnz[conid]
        row += nrow[conid]
        adr += nrow[conid] * rownnz[conid]
    self.assertFalse(np.array_equal(expected_address, address) and not unsorted)
    # Blocks: one per row before the contacts, then one per contact in the order of allocation.
    block_adr = np.full((nworld, capacity), -1, dtype=np.int32)
    block_nrow = np.full((nworld, capacity), -1, dtype=np.int32)
    nblock = np.zeros(nworld, dtype=np.int32)
    expected_adr, expected_nrow = block_adr.copy(), block_nrow.copy()
    for world in range(nworld):
      contacts = [c for c in range(ncon) if worldid[c] == world and active[c]]
      heads = list(range(row_base[world])) + [address[c, 0] for c in contacts[::-1]]
      widths = [1] * row_base[world] + [nrow[c] for c in contacts[::-1]]
      nblock[world] = len(heads)
      block_adr[world, : len(heads)], block_nrow[world, : len(heads)] = heads, widths
      if world in placed and not unsorted:
        contacts = sorted(contacts, key=lambda c: (dim[c] != 1, c))
        heads = list(range(row_base[world])) + [expected_address[c, 0] for c in contacts]
        widths = [1] * row_base[world] + [nrow[c] for c in contacts]
      expected_adr[world, : len(heads)], expected_nrow[world, : len(heads)] = heads, widths
    self.assertFalse(np.array_equal(expected_adr, block_adr) and not unsorted)
    address_wp, id_wp = wp.array(address, dtype=int), wp.array(efc_id, dtype=int)
    rownnz_wp, rowadr_wp = wp.array(efc_rownnz, dtype=int), wp.array(efc_rowadr, dtype=int)
    adr_wp, nrow_wp = wp.array(block_adr, dtype=int), wp.array(block_nrow, dtype=int)
    worldid_wp = wp.array(np.append(worldid, [0, 0]), dtype=int)
    nacon = wp.array([ncon], dtype=int)
    # The buffer is canonical in the fields that order it: the world, then the geoms.
    geom = np.stack([np.arange(ncon + 2), np.arange(ncon + 2)], axis=1)
    if unsorted:
      # The last contact of the first world precedes its predecessor.
      geom[4] = 0
    contact = types.Contact(
      dist=wp.zeros(ncon + 2, dtype=float),
      pos=wp.zeros(ncon + 2, dtype=wp.vec3),
      frame=wp.zeros(ncon + 2, dtype=wp.mat33),
      includemargin=wp.zeros(ncon + 2, dtype=float),
      friction=wp.zeros(ncon + 2, dtype=types.vec5),
      solref=wp.zeros(ncon + 2, dtype=wp.vec2),
      solreffriction=wp.zeros(ncon + 2, dtype=wp.vec2),
      solimp=wp.zeros(ncon + 2, dtype=types.vec5),
      dim=wp.array(np.append(dim, [3, 3]), dtype=int),
      geom=wp.array(geom, dtype=wp.vec2i),
      flex=wp.zeros(0, dtype=wp.vec2i),
      elem=wp.zeros(0, dtype=wp.vec2i),
      vert=wp.zeros(0, dtype=wp.vec2i),
      efc_address=address_wp,
      worldid=worldid_wp,
      type=wp.zeros(ncon + 2, dtype=int),
      geomcollisionid=wp.zeros(ncon + 2, dtype=int),
      adhesion=wp.zeros(ncon + 2, dtype=float),
      surfacevel=wp.zeros(ncon + 2, dtype=wp.vec3),
    )
    flag = wp.zeros(1, dtype=int)
    rows = wp.full(ncon + 2, -9, dtype=int)
    contact_nnz = wp.full(ncon + 2 if sparse else 0, -9, dtype=int)
    wp.launch(
      sort._contact_row_count,
      dim=ncon + 2,
      inputs=[
        int(types.ConeType.PYRAMIDAL),
        contact.dim,
        address_wp,
        worldid_wp,
        rownnz_wp,
        nacon,
        sort._contact_keys(contact),
      ],
      outputs=[flag, rows, contact_nnz],
    )
    self.assertEqual(flag.numpy()[0], ncon if unsorted else 0)
    first = np.append(True, worldid[1:] != worldid[:-1])
    expected_rows = np.where(active, nrow + np.where(dim == 1, sort._ROWS_FRICTIONLESS, 0), 0) + first * sort._ROWS_FIRST
    np.testing.assert_array_equal(rows.numpy()[:ncon], expected_rows)
    np.testing.assert_array_equal(rows.numpy()[ncon:], -9)
    if sparse:
      np.testing.assert_array_equal(contact_nnz.numpy()[:ncon][active], (nrow * rownnz)[active])
    nnz_wp = wp.array(nnz, dtype=int) if sparse else wp.empty(0, dtype=int)
    wp.launch(
      sort._contact_rows_place,
      dim=ncon + 2,
      inputs=[
        wp.array(nefc, dtype=int),
        worldid_wp,
        njmax,
        njmax_nnz,
        nacon,
        nnz_wp,
        wp.array(nblock, dtype=int),
        flag,
        rows,
        contact_nnz,
      ],
      outputs=[address_wp, id_wp, adr_wp, nrow_wp, rownnz_wp, rowadr_wp],
    )
    np.testing.assert_array_equal(address_wp.numpy(), expected_address)
    np.testing.assert_array_equal(id_wp.numpy(), expected_id)
    np.testing.assert_array_equal(adr_wp.numpy(), expected_adr)
    np.testing.assert_array_equal(nrow_wp.numpy(), expected_nrow)
    if sparse:
      np.testing.assert_array_equal(rownnz_wp.numpy(), expected_rownnz)
      np.testing.assert_array_equal(rowadr_wp.numpy(), expected_rowadr)
    else:
      np.testing.assert_array_equal(rownnz_wp.numpy(), efc_rownnz)
      np.testing.assert_array_equal(rowadr_wp.numpy(), efc_rowadr)

  def test_constraint_head(self):
    """The passes only order the rows that precede a sorted tail of contact rows."""
    limit, free, cone = (
      int(types.ConstraintType[name]) for name in ("LIMIT_JOINT", "CONTACT_FRICTIONLESS", "CONTACT_PYRAMIDAL")
    )
    cases = [
      # Row types, row IDs, row count, rows that the passes have to order.
      ([limit, limit, free, cone, cone, cone], [2, 1, 0, 1, 2, 2], 6, 2),
      ([limit, limit, cone, cone, free, cone], [1, 2, 0, 1, 2, 3], 6, 6),
      ([limit, cone, limit, cone, cone, cone], [1, 0, 2, 1, 2, 3], 6, 6),
      ([limit, limit, limit, limit, limit, limit], [3, 1, 2, 0, 5, 4], 6, 6),
      ([free, cone, cone, cone, cone, cone], [0, 1, 2, 3, 4, 5], 6, 0),
      ([limit, cone, cone, cone, cone, cone], [0, 5, 4, 6, 7, 8], 6, 6),
      ([limit, limit, cone, cone, limit, limit], [2, 1, 3, 3, 0, 0], 4, 2),
      ([cone, cone, cone, limit, limit, limit], [3, 2, 1, 0, 0, 0], 0, 0),
      ([limit, limit, limit, cone, cone, cone], [2, 1, 0, 9, 8, 7], 3, 3),
      ([limit, limit, cone, cone, cone, cone], [1, 0, 4, 4, 4, 5], 5, 2),
      ([limit, limit, free, cone, cone, cone], [1, 2, 0, 1, 2, 3], 6, 2),
      ([limit, limit, limit, limit, limit, limit], [0, 1, 2, 3, 4, 5], 6, 6),
    ]
    nworld, capacity = len(cases), 6
    primary = np.zeros((nworld, capacity, 4), dtype=np.int32)
    primary[:, :, 0] = [case[0] for case in cases]
    primary[:, :, 2] = [case[1] for case in cases]
    counts = np.array([case[2] for case in cases], dtype=np.int32)
    keys = sort._ConstraintKeys()
    keys.primary = wp.array(primary, dtype=wp.vec4i)
    keys.J = wp.zeros((nworld, capacity, 1), dtype=wp.uint32)
    keys.J_rownnz = wp.empty((nworld, 0), dtype=int)
    zeros = wp.zeros((nworld, capacity), dtype=wp.uint32)
    keys.pos, keys.margin, keys.D, keys.vel, keys.aref, keys.frictionloss = (zeros,) * 6
    keys.nv = 1
    count = wp.array(counts, dtype=int)
    expected_inverse = wp.empty((nworld, capacity), dtype=int)
    expected = sort.permutation(count, capacity, keys, sort._constraint_less, expected_inverse)
    self.assertFalse(np.array_equal(expected.numpy()[0], np.arange(capacity)))
    # Worlds in which some row precedes its predecessor.
    disordered = np.array([np.any(expected.numpy()[world, :n] != np.arange(n)) for world, n in enumerate(counts)])
    self.assertGreater(disordered.sum(), 3)
    self.assertGreater((~disordered & (counts > 0)).sum(), 1)
    head, disorder = wp.zeros(nworld, dtype=int), wp.zeros(nworld, dtype=int)
    wp.launch(sort._constraint_unsorted, dim=(nworld, capacity), inputs=[count, keys], outputs=[head, disorder])
    np.testing.assert_array_equal(head.numpy(), [case[3] for case in cases])
    np.testing.assert_array_equal(disorder.numpy(), disordered)
    # Ordering the head alone is the stable order of every row, with merge passes or as one block.
    for block in (True, False):
      inverse = wp.empty((nworld, capacity), dtype=int)
      if block:
        order = wp.empty((nworld, capacity), dtype=int)
        ordered = wp.full(nworld, -1, dtype=int)
        # Two threads per world stride over the rows, and the rows past the head keep their index.
        wp.launch(
          sort._constraint_rank,
          dim=(nworld, 2),
          inputs=[head, disorder, keys, wp.full(nworld, capacity, dtype=int), 2],
          outputs=[order, inverse, ordered],
        )
        # A world whose rows are all in order has no row to order.
        np.testing.assert_array_equal(ordered.numpy(), np.where(disordered, head.numpy(), 0))
      else:
        order = sort.permutation(head, capacity, keys, sort._constraint_less, inverse, 2)
      np.testing.assert_array_equal(order.numpy(), expected.numpy())
      np.testing.assert_array_equal(inverse.numpy(), expected_inverse.numpy())

  @parameterized.parameters(1, 3)
  def test_constraint_rank(self, nworld):
    """One block of rows is the stable sort of the merge passes, also for equal keys."""
    capacity = 40
    # Few distinct primary keys and positions, so that rows tie on the primary key, or on every key:
    # the primary keys repeat every 11 rows and the positions every 7.
    primary = ((np.arange(nworld * capacity * 4, dtype=np.int32) * 7 + 3) % 11 % 2).reshape(nworld, capacity, 4)
    pos = ((np.arange(nworld * capacity, dtype=np.uint32) * 5 + 1) % 7 % 2).reshape(nworld, capacity)
    self.assertTrue(np.array_equal(primary[0, 0], primary[0, 11]) and pos[0, 0] != pos[0, 11])
    self.assertTrue(np.array_equal(primary[0, 1], primary[0, 23]) and pos[0, 1] == pos[0, 23])
    counts = np.array([capacity, capacity - 3, 0])[:nworld]
    keys = sort._ConstraintKeys()
    keys.primary = wp.array(primary, dtype=wp.vec4i)
    keys.J = wp.zeros((nworld, capacity, 1), dtype=wp.uint32)
    keys.J_rownnz = wp.empty((nworld, 0), dtype=int)
    zeros = wp.zeros((nworld, capacity), dtype=wp.uint32)
    keys.pos = wp.array(pos, dtype=wp.uint32)
    keys.margin, keys.D, keys.vel, keys.aref, keys.frictionloss = (zeros,) * 5
    keys.nv = 1
    count = wp.array(counts, dtype=int)
    expected_inverse = wp.empty((nworld, capacity), dtype=int)
    expected = sort.permutation(count, capacity, keys, sort._constraint_less, expected_inverse)
    reference = np.tile(np.arange(capacity), (nworld, 1))
    for world, n in enumerate(counts):
      rows = primary[world, :n]
      reference[world, :n] = np.lexsort((np.arange(n), pos[world, :n], rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))
    np.testing.assert_array_equal(expected.numpy(), reference)
    ordered = wp.empty(nworld, dtype=int)
    disorder = wp.ones(nworld, dtype=int)
    for stride, read in ((capacity, wp.full(nworld, capacity, dtype=int)), (7, wp.empty(0, dtype=int))):
      order, inverse = wp.full((nworld, capacity), -3, dtype=int), wp.full((nworld, capacity), -3, dtype=int)
      wp.launch(
        sort._constraint_rank,
        dim=(nworld, stride),
        inputs=[count, disorder, keys, read, stride],
        outputs=[order, inverse, ordered],
      )
      np.testing.assert_array_equal(ordered.numpy(), counts)
      if read.size:
        np.testing.assert_array_equal(order.numpy(), reference)
        np.testing.assert_array_equal(inverse.numpy(), expected_inverse.numpy())
      else:
        # Only the rows of the count are written when no later launch reads the others.
        for world, n in enumerate(counts):
          np.testing.assert_array_equal(order.numpy()[world, :n], reference[world, :n])
          np.testing.assert_array_equal(inverse.numpy()[world, :n], expected_inverse.numpy()[world, :n])
          np.testing.assert_array_equal(order.numpy()[world, n:], -3)
          np.testing.assert_array_equal(inverse.numpy()[world, n:], -3)

  def test_block_place(self):
    """Blocks follow the new position of their head row, and their index when heads are equal."""
    capacity = 8
    inverse = np.array([[3, 0, 6, 1, 7, 2, 5, 4], [1, 0, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    adr = np.array([[4, 0, 2, 6, 1, 1, 1, 1], [3, 5, 3, 0, 2, 2, 2, 2]], dtype=np.int32)
    nrow = np.array([[1, 2, 3, 4, 9, 9, 9, 9], [5, 6, 7, 8, 9, 9, 9, 9]], dtype=np.int32)
    nblock = np.array([4, 4], dtype=np.int32)
    expected_adr = np.full((2, capacity), -1, dtype=np.int32)
    expected_nrow = np.full((2, capacity), -1, dtype=np.int32)
    for world in range(2):
      heads = inverse[world, adr[world, : nblock[world]]]
      order = np.lexsort((np.arange(nblock[world]), heads))
      expected_adr[world, : nblock[world]] = heads[order]
      expected_nrow[world, : nblock[world]] = nrow[world, order]
    # Two blocks of the second world share a head row.
    np.testing.assert_array_equal(expected_nrow[1, :4], [8, 5, 7, 6])
    adr_out, nrow_out = wp.full((2, capacity), -1, dtype=int), wp.full((2, capacity), -1, dtype=int)
    wp.launch(
      sort._block_place,
      dim=(2, capacity),
      inputs=[wp.array(adr, dtype=int), wp.array(nrow, dtype=int), wp.array(nblock, dtype=int), wp.array(inverse, dtype=int)],
      outputs=[adr_out, nrow_out],
    )
    np.testing.assert_array_equal(adr_out.numpy(), expected_adr)
    np.testing.assert_array_equal(nrow_out.numpy(), expected_nrow)

  @parameterized.product(stride=(1, 2, 8), bounded=(False, True))
  def test_sparse_lanes(self, stride, bounded):
    """Striding threads pack the same nonzeros, up to the last row that is not in place."""
    nworld, njmax, capacity = 2, 5, 24
    counts = np.array([5, 4], dtype=np.int32)
    order = np.array([[3, 0, 4, 1, 2], [1, 0, 2, 3, 4]], dtype=np.int32)
    rownnz = np.array([[0, 5, 3, 2, 6], [2, 3, 1, 4, 9]], dtype=np.int32)
    # The rows of the first world are stored in an arbitrary order. Those of the second world are
    # packed in their old order, so that its last two sorted rows are in place.
    rowadr = np.array([[9, 12, 0, 3, 5], [0, 2, 5, 6, 0]], dtype=np.int32)
    colind = ((np.arange(nworld * capacity, dtype=np.int32) * 17 + 5) % 100).reshape(nworld, capacity)
    J = (np.arange(1, nworld * capacity + 1, dtype=np.float32) * 0.125).reshape(nworld, capacity)
    expected_colind, expected_J = colind.copy(), J.copy()
    offsets = np.zeros((nworld, njmax), dtype=np.int32)
    total = np.zeros(nworld, dtype=np.int32)
    last = np.zeros(nworld, dtype=np.int32)
    for world in range(nworld):
      offset = 0
      for row in range(counts[world]):
        old = order[world, row]
        nnz = rownnz[world, old]
        offsets[world, row] = offset
        expected_colind[world, offset : offset + nnz] = colind[world, rowadr[world, old] : rowadr[world, old] + nnz]
        expected_J[world, offset : offset + nnz] = J[world, rowadr[world, old] : rowadr[world, old] + nnz]
        if rowadr[world, old] != offset:
          last[world] = offset + nnz
        offset += nnz
      total[world] = offset
    np.testing.assert_array_equal(last, [total[0], 5])
    self.assertGreater(total[1], last[1])
    count, order_wp, offsets_wp = wp.array(counts, dtype=int), wp.array(order, dtype=int), wp.array(offsets, dtype=int)
    rowadr_wp = wp.array(rowadr, dtype=int)
    unpacked = wp.zeros(nworld, dtype=int)
    wp.launch(
      sort._sparse_unpacked,
      dim=(nworld, njmax),
      inputs=[wp.array(rownnz, dtype=int), rowadr_wp, count, order_wp, offsets_wp],
      outputs=[unpacked],
    )
    np.testing.assert_array_equal(unpacked.numpy(), last)
    bound = unpacked if bounded else wp.array(total, dtype=int)
    colind_wp, J_wp = wp.array(colind, dtype=int), wp.array(J, dtype=float)
    # Elements that do not move are not copied: poison their slots in the copy.
    colind_sorted, J_sorted = wp.full(colind.shape, -1, dtype=int), wp.full(J.shape, wp.nan, dtype=float)
    inputs = [rowadr_wp, count, order_wp, offsets_wp, bound]
    wp.launch(
      sort._sparse_move_lanes,
      dim=(nworld, stride),
      inputs=inputs + [colind_wp, J_wp, False, stride],
      outputs=[colind_sorted, J_sorted],
    )
    np.testing.assert_array_equal(colind_sorted.numpy()[0, : total[0]], expected_colind[0, : total[0]])
    np.testing.assert_array_equal(colind_sorted.numpy()[1, :5], expected_colind[1, :5])
    np.testing.assert_array_equal(colind_sorted.numpy()[1, 5:], -1)
    np.testing.assert_array_equal(colind_sorted.numpy()[0, total[0] :], -1)
    wp.launch(
      sort._sparse_move_lanes,
      dim=(nworld, stride),
      inputs=inputs + [colind_sorted, J_sorted, True, stride],
      outputs=[colind_wp, J_wp],
    )
    np.testing.assert_array_equal(colind_wp.numpy(), expected_colind)
    np.testing.assert_array_equal(J_wp.numpy(), expected_J)

  @parameterized.product(
    jacobian=("dense", "sparse"),
    ordered=(True, False),
    path=(
      ("_LANES", 3),
      ("_LANES", 1 << 30),
      ("_BLOCK_RANK_NJMAX", 0),
      ("_HEAD_NJMAX", 0),
      ("_BUCKET_NCONMAX", 0),
      ("CONTACTS", 0),
    ),
  )
  def test_constraint_paths(self, jacobian, ordered, path):
    """The launches chosen from the static sizes all give the same sorted rows."""
    results = []
    real = sort.constraints
    placed = []

    def spy(m, d, nnz=None, unsorted=None):
      placed.append(unsorted is not None)
      return real(m, d, nnz, unsorted)

    # The kernels that each assembly launches, with their dimensions, and its merge passes.
    launch, permutation = wp.launch, sort.permutation
    launched, merged, widths = [], [], []

    def spy_launch(kernel, *args, **kwargs):
      launched[-1][kernel] = kwargs.get("dim")
      return launch(kernel, *args, **kwargs)

    def spy_permutation(*args, **kwargs):
      merged[-1] += 1
      widths[-1].append(args[5] if len(args) > 5 else None)
      return permutation(*args, **kwargs)

    _, _, m, d = test_data.fixture(xml=_ROWS_XML, nworld=2, nconmax=16, njmax=96, overrides={"opt.jacobian": jacobian})
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    sort.contacts(d)
    if not ordered:
      order = np.concatenate([np.arange(0, count, 2), np.arange(1, count, 2)])[::-1]
      for array in _arrays(d.contact):
        if array.size:
          values = array.numpy().copy()
          values[:count] = values[:count][order]
          array.assign(values)
    # Both paths assemble and sort the same state.
    initial = [(a, wp.clone(a)) for a in _arrays(d)]
    for patch in (None, path):
      launched.append({})
      merged.append(0)
      widths.append([])
      for target, value in initial:
        wp.copy(target, value)
      # Without the contact sort, the launches that order and move every row are chosen.
      unsorted_contacts = patch is not None and patch[0] == "CONTACTS"
      m.opt.deterministic = mjw.DeterminismType.CONSTRAINT if unsorted_contacts else _SORTED
      before = d.efc.J.numpy().copy()
      spies = (
        mock.patch.object(sort, "constraints", spy),
        mock.patch.object(wp, "launch", spy_launch),
        mock.patch.object(sort, "permutation", spy_permutation),
      )
      with spies[0], spies[1], spies[2]:
        if patch is None:
          # More chunks than lanes, and rows for a merge pass when a head is not one block.
          self.assertGreater(min(d.njmax * m.nv, d.njmax_nnz) // sort._CHUNK, sort._LANES)
          self.assertGreater(d.njmax, sort._RANK_WIDTH)
          constraint.make_constraint(m, d)
        elif unsorted_contacts:
          constraint.make_constraint(m, d)
        else:
          with mock.patch.object(sort, patch[0], patch[1]):
            constraint.make_constraint(m, d)
      self.assertFalse(np.array_equal(d.efc.J.numpy(), before))
      if m.is_sparse:
        _assert_jtdaj_blocks(self, d)
      results.append(_sorted_rows(m, d))
      # The static sizes choose the launches: none of the faster ones is skipped by default.
      name = patch[0] if patch else None
      lanes = not unsorted_contacts and patch != ("_LANES", 1 << 30)
      one_block = not unsorted_contacts and name != "_HEAD_NJMAX"
      kernels = launched[-1]
      self.assertEqual(merged[-1], int(not one_block) + int(m.is_sparse and name == "_BLOCK_RANK_NJMAX"))
      # A head is ordered by wide blocks. The blocks for every row are chosen from the sizes.
      if not one_block:
        self.assertEqual(widths[-1][0], None if unsorted_contacts else sort._RANK_WIDTH)
      self.assertEqual(sort._constraint_rank in kernels, one_block)
      if one_block:
        self.assertEqual(tuple(kernels[sort._constraint_rank]), (d.nworld, sort._ROW_LANES))
      self.assertEqual(sort._constraint_unsorted in kernels, not unsorted_contacts)
      self.assertEqual(sort._contact_row_count in kernels, name not in ("_BUCKET_NCONMAX", "CONTACTS"))
      self.assertEqual(sort._contact_unsorted in kernels, name in ("_BUCKET_NCONMAX", "CONTACTS"))
      if m.is_sparse:
        self.assertEqual(sort._sparse_unpacked in kernels, not unsorted_contacts)
        self.assertEqual(sort._block_place in kernels, name != "_BLOCK_RANK_NJMAX")
        self.assertEqual(sort._block_gather in kernels, name == "_BLOCK_RANK_NJMAX")
        self.assertEqual(sort._sparse_move_lanes in kernels, lanes)
        self.assertEqual(sort._sparse_move in kernels, not lanes)
      else:
        self.assertEqual(sort._gather in kernels, not lanes)
        self.assertEqual(sort._restore in kernels, not lanes)
    # Contacts only place their rows while each walks few contacts of its world.
    self.assertEqual(placed, [True, path[0] not in ("_BUCKET_NCONMAX", "CONTACTS")])
    for name, value in results[0].items():
      np.testing.assert_array_equal(value, results[1][name], err_msg=name)

  @parameterized.parameters(1, 4)
  def test_dense_lanes_batch(self, nworld):
    """The row launches also move the dense rows of a batch of small worlds."""
    results = []
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .09">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body pos="1 0 .08">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=4,
      njmax=12,
      overrides={"opt.deterministic": _SORTED},
    )
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertEqual(count, 2 * nworld)
    # Contacts out of order: their rows are allocated out of order, and move.
    self.assertFalse(np.array_equal(d.contact.geom.numpy()[:count], d.contact.geom.numpy()[:count][::-1]))
    for array in _arrays(d.contact):
      if array.size:
        values = array.numpy().copy()
        values[:count] = values[:count][::-1]
        array.assign(values)
    # A world holds fewer chunks than lanes; a batch of four worlds holds more.
    nchunk = (m.nv + sort._CHUNK - 1) // sort._CHUNK
    self.assertLessEqual(d.njmax * nchunk, sort._LANES)
    self.assertEqual(nworld * d.njmax * nchunk > sort._LANES, nworld == 4)
    launch = wp.launch
    kernels = []

    def spy_launch(kernel, *args, **kwargs):
      kernels.append(kernel)
      return launch(kernel, *args, **kwargs)

    # Both widths assemble and sort the same state.
    initial = [(a, wp.clone(a)) for a in _arrays(d)]
    for width in (sort._LANES, 1 << 30):
      for target, value in initial:
        wp.copy(target, value)
      kernels.clear()
      before = d.efc.J.numpy().copy()
      with mock.patch.object(wp, "launch", spy_launch), mock.patch.object(sort, "_LANES", width):
        constraint.make_constraint(m, d)
      # The rows were allocated in the order of the buffer: they moved to their canonical place.
      np.testing.assert_array_equal(d.efc.J.numpy(), before)
      separate = width != sort._LANES or nworld == 1
      self.assertEqual(sort._gather in kernels, separate)
      self.assertEqual(sort._restore in kernels, separate)
      results.append(_sorted_rows(m, d))
    for name, value in results[0].items():
      np.testing.assert_array_equal(value, results[1][name], err_msg=name)

  @parameterized.product(nworld=(1, 3), surfacevel=("written", "stale"), flex=(True, False))
  def test_contact_buffer(self, nworld, surfacevel, flex):
    """Contacts placed from the arrays that the narrowphase wrote equal contacts sorted in place."""
    _, _, m, d = test_data.fixture(
      xml=_FLEX_XML if flex else _CONTACTS_XML,
      nworld=nworld,
      nconmax=100,
      overrides={"opt.deterministic": mjw.DeterminismType.CONTACTS},
    )
    count = d.naconmax - 7
    contact = {name: array for name, array in vars(d.contact).items() if array.size}
    self.assertEqual("flex" in contact, flex)
    reference = {}
    for field, (name, array) in enumerate(contact.items()):
      value = array.numpy()
      # Values from -1 to 2 that repeat with another period in every field. A negative geom is a
      # flex, whose features precede the position.
      index = np.arange(value.size, dtype=np.int64)
      reference[name] = ((index * (2 * field + 7) + field) % (97 + 2 * field) % 4 - 1).reshape(value.shape).astype(value.dtype)
    # World IDs from two below the batch to one above it.
    reference["worldid"] = ((np.arange(d.naconmax, dtype=np.int32) * 5 + 1) % 11 % (nworld + 4) - 2).astype(np.int32)
    # Contacts with equal keys keep the order of the buffer. Other contacts only share the keys
    # that the sort groups by world, so that a later key decides their order.
    shuffled = (np.arange(count) * 37 + 11) % count
    self.assertLen(np.unique(shuffled), count)
    pairs = shuffled[: 4 * (count // 5)].reshape(-1, 2)
    for name, value in reference.items():
      first = name in ("worldid", "geom", "pos", "flex", "elem", "vert")
      rows = pairs if first else pairs[: len(pairs) // 2]
      value[rows[:, 1]] = value[rows[:, 0]]
    if flex:
      # Not a key of a rigid geom: its flex features tell equal contacts apart.
      equal = pairs[: len(pairs) // 2]
      reference["geom"][equal] = 1
      reference["vert"][equal[:, 1]] = 5
    # Every new contact has the same addresses; the unused slots identify themselves.
    reference["efc_address"][:count] = -1
    reference["efc_address"][count:] = 100 + np.arange(reference["efc_address"][count:].size).reshape(7, -1)
    for name, value in reference.items():
      contact[name].assign(value)
    d.nacon.fill_(count)
    sort.contacts(d)
    expected = {name: array.numpy().copy() for name, array in contact.items()}
    self.assertFalse(np.array_equal(expected["pos"], reference["pos"]))

    source = sort.contact_buffer(m, d)
    self.assertEqual(source is None, flex)
    if flex:
      # A buffer that also holds the flex features of its contacts. Addresses do not move.
      source = types.Contact(*(wp.empty_like(array) for array in _arrays(d.contact)))
      source.efc_address.fill_(7)
    written_to = vars(source)
    for name, value in reference.items():
      if name == "efc_address":
        contact[name].assign(value)
        continue
      # The buffer holds the new contacts. The contact buffer holds old contacts and unused slots.
      written, old = value.copy(), value.copy()
      written[count:] = 7
      old[:count] = 7
      stale = name == "surfacevel" and surfacevel != "written"
      written_to[name].assign(np.full_like(value, 7) if stale else written)
      contact[name].assign(value if stale else old)
    sort.contacts(d, source, surfacevel == "written")
    for name, value in expected.items():
      np.testing.assert_array_equal(contact[name].numpy(), value, err_msg=name)
    self.assertEqual(d.nacon.numpy()[0], count)

  @parameterized.parameters("buffer", "none", "merge", "incremental", "filter", "flex", "limit", "above")
  def test_contact_buffer_choice(self, case):
    """The narrowphase writes to the buffer of the sort only while the sort can place from it."""
    _, _, m, d = test_data.fixture(
      xml=_FLEX_XML if case == "flex" else _CONTACTS_XML,
      nworld=2,
      # The largest buffer whose contacts are ranked within their worlds, and the next one.
      nconmax={"limit": 256, "above": 257}.get(case, 64),
      overrides={"opt.deterministic": mjw.DeterminismType.NONE if case == "none" else mjw.DeterminismType.CONTACTS},
    )
    self.assertEqual(d.naconmax > sort._BUCKET_NCONMAX * d.nworld, case == "above")
    if case == "filter":
      m.callback.contactfilter = lambda m, d: None
    launch, contact_buffer = wp.launch, sort.contact_buffer
    launched, buffers = [], []

    def spy_launch(kernel, *args, **kwargs):
      launched.append((kernel, kwargs.get("outputs", [])))
      return launch(kernel, *args, **kwargs)

    def spy_buffer(*args):
      buffers.append(contact_buffer(*args))
      return buffers[-1]

    spies = (
      mock.patch.object(wp, "launch", spy_launch),
      mock.patch.object(sort, "contact_buffer", spy_buffer),
      mock.patch.object(sort, "_BUCKET_NCONMAX", 0 if case == "merge" else sort._BUCKET_NCONMAX),
    )
    with spies[0], spies[1], spies[2]:
      if case == "incremental":
        collision_driver.collision(m, d, awake_prev=wp.clone(d.body_awake))
      else:
        mjw.collision(m, d)
    self.assertGreater(d.nacon.numpy()[0], d.nworld)
    self.assertLen(buffers, 1)
    self.assertEqual(buffers[0] is not None, case in ("buffer", "limit"))
    kernels = [kernel for kernel, _ in launched]
    for kernel in (sort._contact_group_keys, sort._contact_place_once):
      self.assertEqual(kernel in kernels, case in ("buffer", "limit"), kernel.key)
    for kernel in (sort._contact_group, sort._contact_place):
      self.assertEqual(kernel in kernels, case in ("incremental", "filter", "flex"), kernel.key)
    self.assertEqual(sort._contact_gather in kernels, case in ("merge", "above"))
    self.assertEqual(any(kernel.module is sort._contact_buckets.module for kernel in kernels), case != "none")
    # The narrowphase launches write the contact count, and the contacts to the buffer if any.
    narrowphase = [outputs for kernel, outputs in launched if kernel.module is not sort._contact_buckets.module]
    narrowphase = [outputs for outputs in narrowphase if any(output is d.nacon for output in outputs)]
    self.assertNotEmpty(narrowphase)
    for outputs in narrowphase:
      self.assertEqual(any(output is d.contact.pos for output in outputs), case not in ("buffer", "limit"))
      # Every new contact gets the same addresses, which are written in place.
      self.assertTrue(any(output is d.contact.efc_address for output in outputs))
    if case in ("buffer", "limit"):
      self.assertIs(buffers[0].efc_address, d.contact.efc_address)
      buffer = vars(buffers[0])
      self.assertLen(buffer, 19)
      for name, array in vars(d.contact).items():
        allocated = name not in ("efc_address", "flex", "elem", "vert")
        self.assertEqual(buffer[name] is not array, allocated, name)

  @parameterized.product(nworld=(1, 5), surfacevel=(True, False), sdf=(True, False))
  def test_contact_buffer_collision(self, nworld, surfacevel, sdf):
    """A narrowphase that writes to the buffer of the sort gives the contacts of a sort in place."""
    xml = _SDF_XML if sdf else _CONTACTS_XML
    _, _, m, d = test_data.fixture(
      xml=xml.replace('<geom type="plane"', '<geom type="plane" surfacevel=".5 .25 0 0 0 1"') if surfacevel else xml,
      nworld=nworld,
      nconmax=128 if sdf else 64,
      qpos_noise=0.002,
      overrides={"opt.deterministic": mjw.DeterminismType.CONTACTS},
    )
    self.assertEqual(m.flg_surfacevel, surfacevel)
    contact = {name: array for name, array in vars(d.contact).items() if array.size}
    results = []
    for buffer in (True, False, None):
      # Old contacts, whose addresses and surface velocities differ from those of new contacts.
      for name, array in contact.items():
        array.fill_(3)
      m.opt.deterministic = mjw.DeterminismType.NONE if buffer is None else mjw.DeterminismType.CONTACTS
      with mock.patch.object(sort, "contact_buffer", sort.contact_buffer if buffer else lambda *args: None):
        mjw.collision(m, d)
      results.append({name: array.numpy().copy() for name, array in contact.items()})
    count = int(d.nacon.numpy()[0])
    self.assertGreater(count, 4 * nworld)
    self.assertLess(count, d.naconmax)
    for name, value in results[0].items():
      np.testing.assert_array_equal(value, results[1][name], err_msg=name)
      np.testing.assert_array_equal(value[count:], 3, err_msg=name)
    self.assertTrue(np.all(np.diff(results[0]["worldid"][:count]) >= 0))
    # The sort only moves contacts: unsorted collision detection writes the same ones.
    for name, value in results[0].items():
      rows = [np.sort(result[name][:count].reshape(count, -1), axis=0) for result in (results[0], results[2])]
      np.testing.assert_array_equal(rows[0], rows[1], err_msg=name)
    np.testing.assert_array_equal(results[2]["efc_address"][:count], -1)
    if surfacevel:
      self.assertTrue(np.any(results[0]["surfacevel"][:count] != 0))
    else:
      np.testing.assert_array_equal(results[0]["surfacevel"], 3)


_CONTACTS_XML = """
<mujoco>
  <option>
    <flag sleep="enable"/>
  </option>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .09">
      <freejoint/>
      <geom size=".1"/>
    </body>
    <body pos=".5 0 .095">
      <freejoint/>
      <geom type="box" size=".1 .1 .1"/>
    </body>
    <body pos=".5 0 .29">
      <freejoint/>
      <geom type="box" size=".1 .1 .1"/>
    </body>
    <body pos="1 0 .045">
      <freejoint/>
      <geom type="capsule" size=".05 .1" euler="0 90 0"/>
    </body>
    <body pos="1.5 0 .09">
      <freejoint/>
      <geom type="ellipsoid" size=".1 .1 .1"/>
    </body>
  </worldbody>
</mujoco>
"""

_SDF_XML = """
<mujoco>
  <option sdf_iterations="10" sdf_initpoints="20"/>
  <asset>
    <mesh name="cube" vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1" scale=".1 .1 .1"/>
  </asset>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <geom type="sdf" mesh="cube" pos="0 0 .1"/>
    <body pos="0 0 .28" euler="45 0 0">
      <freejoint/>
      <geom type="sdf" mesh="cube"/>
    </body>
    <body pos="1 0 .09">
      <freejoint/>
      <geom size=".1"/>
    </body>
  </worldbody>
</mujoco>
"""

_FLEX_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" pos="0 0 .01" radius=".02">
      <edge equality="true"/>
      <contact selfcollide="none"/>
    </flexcomp>
  </worldbody>
</mujoco>
"""

_ROWS_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .09">
      <freejoint/>
      <geom type="box" size=".1 .12 .1"/>
    </body>
    <body pos="1 0 .08">
      <freejoint/>
      <geom size=".1" condim="1" priority="1"/>
    </body>
    <body pos="2 0 .08">
      <freejoint/>
      <geom size=".1"/>
    </body>
    <body pos="3 0 1">
      <joint range=".1 1" frictionloss=".1"/>
      <geom size=".1"/>
    </body>
    <body pos="4 0 1">
      <joint axis="0 1 0" range=".2 1"/>
      <geom size=".1"/>
    </body>
    <body pos="5 0 1">
      <joint axis="1 0 0" range=".3 1"/>
      <geom size=".1"/>
    </body>
  </worldbody>
</mujoco>
"""


# Solver outputs of an earlier step, which the assembly does not write: they stay where they were.
_STALE = ("force", "state", "island")


def _sorted_rows(m, d):
  result = {name: array.numpy().copy() for name, array in vars(d.efc).items() if array.size and name not in _STALE}
  result["efc_address"] = d.contact.efc_address.numpy().copy()
  result["dof_frictionadr"] = d.dof_frictionadr.numpy().copy()
  for world in range(d.nworld):
    result[f"dense{world}"] = _dense_jacobian(m, d, world)
  return result


def _arrays(value):
  for item in vars(value).values():
    if isinstance(item, wp.array):
      yield item
    elif dataclasses.is_dataclass(item):
      yield from _arrays(item)


def _import_mj_contacts(nworld):
  mjm, mjd, m, _ = test_data.fixture(
    xml="""
    <mujoco>
      <option jacobian="sparse" cone="elliptic" tolerance="1e-10"/>
      <worldbody>
        <geom type="plane" size="2 2 .1"/>
        <body pos="0 0 .09">
          <freejoint/>
          <geom type="box" size=".1 .1 .1"/>
        </body>
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


def _assert_collision_repeat(test, m, d):
  reference = None
  for _ in range(4):
    mjw.collision(m, d)
    constraint.make_constraint(m, d)
    count = int(d.nacon.numpy()[0])
    test.assertGreater(count, d.nworld)
    test.assertLessEqual(count, d.naconmax)
    test.assertTrue(np.all(d.nefc.numpy() <= d.njmax))
    current = {name: array.numpy()[:count].copy() for name, array in vars(d.contact).items() if array.size}
    for world in range(d.nworld):
      current[world, "J"] = _dense_jacobian(m, d, world)
    if reference is not None:
      for name, value in current.items():
        np.testing.assert_array_equal(value, reference[name], err_msg=str(name))
    reference = current


def _assert_jtdaj_blocks(test, d):
  """Sorted blocks tile every row but the DOF friction rows, which sparse Newton adds separately."""
  nefc = d.nefc.numpy()
  nblock = d.efc.jtdaj_nblock.numpy()
  adr = d.efc.jtdaj_adr.numpy()
  nrow = d.efc.jtdaj_nrow.numpy()
  rowtype = d.efc.type.numpy()
  blocks = []
  for world, n in enumerate(nefc):
    starts = adr[world, : nblock[world]].copy()
    widths = nrow[world, : nblock[world]].copy()
    rows = [row for start, width in zip(starts, widths) for row in range(start, start + width)]
    np.testing.assert_array_equal(rows, np.flatnonzero(rowtype[world, :n] != types.ConstraintType.FRICTION_DOF))
    blocks.append((starts, widths))
  return blocks


def _assert_friction_addresses(test, d):
  """Every DOF friction row is the address of its DOF."""
  address = d.dof_frictionadr.numpy().copy()
  rowtype = d.efc.type.numpy()
  ids = d.efc.id.numpy()
  for world, n in enumerate(d.nefc.numpy()):
    rows = np.flatnonzero(rowtype[world, :n] == types.ConstraintType.FRICTION_DOF)
    dofs = np.flatnonzero(address[world] >= 0)
    np.testing.assert_array_equal(np.sort(address[world, dofs]), rows)
    np.testing.assert_array_equal(ids[world, address[world, dofs]], dofs)
  return address


def _dense_jacobian(m, d, world):
  n = int(d.nefc.numpy()[world])
  if not m.is_sparse:
    return d.efc.J.numpy()[world, :n, : m.nv].copy()
  result = np.zeros((n, m.nv), dtype=np.float64)
  mujoco.mju_sparse2dense(
    result,
    d.efc.J.numpy()[world, 0].astype(np.float64),
    d.efc.J_rownnz.numpy()[world, :n],
    d.efc.J_rowadr.numpy()[world, :n],
    d.efc.J_colind.numpy()[world, 0],
  )
  return result


if __name__ == "__main__":
  absltest.main()
