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

  @parameterized.product(nworld=(1, 2), key=("geometry", "flex", "material"))
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
    elif key == "material":
      # Same feature, point and frame, but different materials supplied by a contact filter.
      for name in ("pos", "frame", "dist", "dim", "type", "includemargin"):
        reference[name][:count] = 0
      reference["friction"][:count, 0] = np.tile(np.arange(8), nworld)
    # The captured sort has to read the contact order and the active count of every replay.
    with wp.ScopedCapture() as capture:
      collision_driver._sort_contacts(m, d)
    for active in (count, count - 3, 2):
      d.nacon.fill_(active)
      for shift in (1, 3, 5):
        permutation = np.roll(np.arange(active)[::-1], shift)
        for name, value in reference.items():
          shuffled = value.copy()
          shuffled[:active] = value[permutation]
          contact[name].assign(shuffled)
        if shift == 1:
          collision_driver._sort_contacts(m, d)
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
    collision_driver._sort_contacts(m, d)
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
