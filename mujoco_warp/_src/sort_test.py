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
import itertools

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import collision_driver
from mujoco_warp._src import constraint
from mujoco_warp._src import types


class SortTest(parameterized.TestCase):
  @parameterized.parameters(*itertools.product((1, 2), ("dense", "sparse"), ("pyramidal", "elliptic")))
  def test_unused_contact_addresses(self, nworld, jacobian, cone):
    """Unused contact components cannot overwrite another contact's row key."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco><worldbody>
        <geom type="plane" size="5 5 .1" condim="1"/>
        <body pos="0 0 .09"><freejoint/><geom size=".1" condim="1"/></body>
        <body pos="1 0 .08"><freejoint/><geom size=".1" condim="6"/></body>
        <body pos="2 0 .07"><freejoint/><geom size=".1" condim="6"/></body>
      </worldbody></mujoco>
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
        values = array.numpy()
        values[:count] = values[:count][::-1]
        array.assign(values)
    constraint.make_constraint(m, d)
    # Solver outputs also have to follow the row permutation, even though assembly doesn't set them.
    for name in ("force", "state", "island"):
      array = getattr(d.efc, name)
      array.assign(np.tile(np.arange(array.shape[1]) + 10, (nworld, 1)))
    before = [(a, wp.clone(a)) for a in _arrays(d.efc)]
    addresses = d.contact.efc_address.numpy()
    constraint._sort_constraints(m, d)
    expected = [a.numpy() for a, _ in before]
    dims = d.contact.dim.numpy()
    worlds = d.contact.worldid.numpy()
    sorted_addresses = d.contact.efc_address.numpy()
    for conid in range(count):
      nrow = max(1, 2 * (dims[conid] - 1)) if cone == "pyramidal" else dims[conid]
      for name in ("force", "state", "island"):
        actual = getattr(d.efc, name).numpy()[worlds[conid], sorted_addresses[conid, :nrow]]
        np.testing.assert_array_equal(actual, addresses[conid, :nrow] + 10)
    for _ in range(3):
      for target, source in before:
        wp.copy(target, source)
      poisoned = addresses.copy()
      for conid in range(count):
        if dims[conid] == 1:
          other = np.flatnonzero((worlds[:count] == worlds[conid]) & (dims[:count] == 6))[0]
          poisoned[conid, 1:] = addresses[other, 0]
      d.contact.efc_address.assign(poisoned)
      constraint._sort_constraints(m, d)
      for (actual, _), value in zip(before, expected):
        np.testing.assert_array_equal(actual.numpy(), value)

  @parameterized.parameters("nxn", "sap_tile", "sap_segmented")
  def test_incremental_collision(self, broadphase):
    """Sorting preserves contacts appended for newly awakened bodies after the contact filter."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco><option><flag sleep="enable"/></option><worldbody>
        <geom type="plane" size="5 5 .1"/>
        <body pos="0 0 .09"><freejoint/><geom size=".1"/></body>
        <body pos="1 0 .08"><freejoint/><geom size=".1"/></body>
      </worldbody></mujoco>
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
          <mesh name="cube" vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <geom type="sdf" mesh="cube"/>
          <body pos="0 0 1.8" euler="45 0 0"><freejoint/><geom type="sdf" mesh="cube"/></body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      nconmax=128,
      njmax=512,
      overrides={"opt.deterministic": mjw.DeterminismType.CONTACTS | mjw.DeterminismType.CONSTRAINT},
    )
    _assert_collision_repeat(self, m, d)

  def test_flex_contacts(self):
    """Real flex contacts retain their feature identity through both sorting stages."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco><worldbody>
        <geom type="plane" size="5 5 .1"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" mass="1" pos="0 0 .01" radius=".02">
          <edge equality="true"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </worldbody></mujoco>
      """,
      nworld=2,
      nconmax=128,
      njmax=512,
      overrides={"opt.deterministic": mjw.DeterminismType.ALL, "opt.jacobian": "sparse"},
    )
    _assert_collision_repeat(self, m, d)

  @parameterized.parameters(*itertools.product((1, 2, 4), ("dense", "sparse"), ("pyramidal", "elliptic")))
  def test_interleaved_constraints(self, nworld, jacobian, cone):
    """Constraint sorting alone handles external contacts with interleaved world IDs."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco><worldbody>
        <geom type="plane" size="5 5 .1"/>
        <body pos="0 0 .09"><freejoint/><geom size=".1"/></body>
        <body pos="1 0 .08"><freejoint/><geom size=".1"/></body>
      </worldbody></mujoco>
      """,
      nworld=nworld,
      nconmax=8,
      njmax=32,
      qvel_noise=0.1,
      overrides={"opt.jacobian": jacobian, "opt.cone": cone},
    )
    qpos = d.qpos.numpy()
    qvel = d.qvel.numpy()
    for world in range(nworld):
      qpos[world, 2] -= 0.003 * world
      qvel[world] *= world + 1
    d.qpos.assign(qpos)
    d.qvel.assign(qvel)
    mjw.forward(m, d)
    count = int(d.nacon.numpy()[0])
    self.assertEqual(count, 2 * nworld)
    fields = {f.name: getattr(d.contact, f.name).numpy() for f in dataclasses.fields(d.contact)}
    # Do not test uninitialized put_data metadata.
    d.efc.force.zero_()
    d.efc.state.zero_()
    d.efc.island.zero_()
    m.opt.deterministic = mjw.DeterminismType.CONSTRAINT

    def snapshot():
      result = {}
      ids = d.efc.id.numpy()
      geoms = d.contact.geom.numpy()
      for world, n in enumerate(d.nefc.numpy()):
        self.assertEqual(n, 8 if cone == "pyramidal" else 6)
        for name in ("type", "pos", "margin", "D", "vel", "aref", "frictionloss", "Jqvel"):
          result[world, name] = getattr(d.efc, name).numpy()[world, :n]
        result[world, "id"] = geoms[ids[world, :n]]
        if m.is_sparse:
          nnz = d.efc.J_rownnz.numpy()[world, :n]
          adr = d.efc.J_rowadr.numpy()[world, :n]
          result[world, "J_rownnz"] = nnz
          result[world, "J_rowadr"] = adr
          result[world, "J"] = d.efc.J.numpy()[world, 0, : sum(nnz)]
          result[world, "J_colind"] = d.efc.J_colind.numpy()[world, 0, : sum(nnz)]
          blocks = int(d.efc.jtdaj_nblock.numpy()[world])
          starts = d.efc.jtdaj_adr.numpy()[world, :blocks]
          sizes = d.efc.jtdaj_nrow.numpy()[world, :blocks]
          np.testing.assert_array_equal(starts, np.cumsum(sizes) - sizes)
          self.assertEqual(sum(sizes), n)
          result[world, "jtdaj_adr"] = starts
          result[world, "jtdaj_nrow"] = sizes
      addresses = d.contact.efc_address.numpy()
      worldids = d.contact.worldid.numpy()
      for i in range(count):
        for address in addresses[i]:
          if address >= 0:
            self.assertEqual(ids[worldids[i], address], i)
      if not m.is_sparse:
        result["J"] = d.efc.J.numpy()
      return result

    reference = None
    rng = np.random.default_rng(42)
    for _ in range(4):
      order = rng.permutation(count)
      for name, value in fields.items():
        if not value.size:
          continue
        shuffled = value.copy()
        shuffled[:count] = value[order]
        getattr(d.contact, name).assign(shuffled)
      constraint.make_constraint(m, d)
      current = snapshot()
      if reference is not None:
        for name, value in current.items():
          np.testing.assert_array_equal(value, reference[name], err_msg=str(name))
      reference = current
    if nworld > 1:
      self.assertFalse(np.array_equal(reference[0, "pos"], reference[1, "pos"]))

    for world in range(nworld):
      mjd.qpos[:] = qpos[world]
      mjd.qvel[:] = qvel[world]
      mujoco.mj_forward(mjm, mjd)
      for name in ("pos", "margin", "D", "vel", "aref", "frictionloss"):
        np.testing.assert_allclose(reference[world, name], getattr(mjd, "efc_" + name), rtol=1e-5, atol=1e-5)
      actual = _dense_jacobian(m, d, world)
      expected = np.zeros_like(actual, dtype=np.float64)
      if m.is_sparse:
        mujoco.mju_sparse2dense(expected, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
      else:
        expected[:] = mjd.efc_J.reshape(expected.shape)
      np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  @parameterized.parameters(*itertools.product((1, 2), ("dense", "sparse"), ("pyramidal", "elliptic")))
  def test_forward_capture(self, nworld, jacobian, cone):
    """Real collision/constraint stages repeat exactly, with and without CUDA capture."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco><compiler angle="radian"/>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body name="a" pos="0 0 .09"><freejoint/><geom type="box" size=".1 .12 .1"/></body>
          <body name="b" pos="1 0 .08"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
          <body pos="2 0 1"><joint name="j1" range=".1 1" frictionloss=".1"/><geom size=".1"/></body>
          <body pos="3 0 1"><joint name="j2" range=".2 1" frictionloss=".2"/><geom size=".1"/></body>
        </worldbody>
        <equality><connect body1="a" body2="b" anchor=".5 0 .09"/><joint joint1="j1" joint2="j2"/></equality>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=32,
      njmax=128,
      qvel_noise=0.1,
      overrides={"opt.jacobian": jacobian, "opt.cone": cone},
    )
    qpos = d.qpos.numpy()
    qvel = d.qvel.numpy()
    if nworld == 2:
      qpos[1, 2] -= 0.002
      qvel[1] *= 1.5
    d.qpos.assign(qpos)
    d.qvel.assign(qvel)
    m.opt.deterministic = mjw.DeterminismType.ALL
    # Isolate sorting from Warp's deterministic-atomic workspaces in conditional solver subgraphs.
    m.opt.graph_conditional = False
    m.opt.iterations = 4
    initial = [(a, wp.clone(a)) for a in _arrays(d)]

    def restore():
      for target, value in initial:
        wp.copy(target, value)

    def snapshot():
      ncon = int(d.nacon.numpy()[0])
      result = {"qacc": d.qacc.numpy(), "qfrc_constraint": d.qfrc_constraint.numpy()}
      for f in dataclasses.fields(d.contact):
        a = getattr(d.contact, f.name)
        if a.size and f.name not in ("flex", "elem", "vert"):
          result["contact", f.name] = a.numpy()[:ncon]
      for world, n in enumerate(d.nefc.numpy()):
        for name in ("type", "id", "pos", "margin", "D", "vel", "aref", "frictionloss", "force", "state"):
          result[world, name] = getattr(d.efc, name).numpy()[world, :n]
        result[world, "J"] = _dense_jacobian(m, d, world)
        if m.is_sparse:
          nb = d.efc.jtdaj_nblock.numpy()[world]
          starts = d.efc.jtdaj_adr.numpy()[world, :nb]
          sizes = d.efc.jtdaj_nrow.numpy()[world, :nb]
          np.testing.assert_array_equal(starts, np.cumsum(sizes) - sizes)
          self.assertEqual(sum(sizes), n)
      return result

    m.opt.deterministic = mjw.DeterminismType.NONE
    mjw.forward(m, d)
    unsorted_qacc = d.qacc.numpy()
    restore()
    m.opt.deterministic = mjw.DeterminismType.ALL
    mjw.forward(m, d)
    reference = snapshot()
    # Row reordering changes the solver's float32 reduction order.
    np.testing.assert_allclose(reference["qacc"], unsorted_qacc, rtol=5e-5, atol=5e-5)
    graph = None
    if wp.get_device().is_cuda:
      with wp.ScopedCapture() as capture:
        mjw.forward(m, d)
      graph = capture.graph
    for i in range(6):
      restore()
      if graph and i % 2:
        wp.capture_launch(graph)
      else:
        mjw.forward(m, d)
      for name, value in snapshot().items():
        np.testing.assert_array_equal(value, reference[name], err_msg=str(name))
    if nworld > 1:
      self.assertFalse(np.array_equal(reference["qacc"][0], reference["qacc"][1]))

  @parameterized.parameters(*itertools.product((1, 2), ("geometry", "flex", "material")))
  def test_contact_permutation(self, nworld, key):
    """All fields follow a canonical permutation, including equal pair/point IDs."""
    _, _, m, d = test_data.fixture(
      xml="""<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>""",
      nworld=nworld,
      nconmax=16,
    )
    count = 8 * nworld
    rng = np.random.default_rng(17)
    reference = {}
    for field in dataclasses.fields(d.contact):
      array = getattr(d.contact, field.name)
      if array.shape[0] == 0:
        array = wp.zeros(d.naconmax, dtype=array.dtype)
        setattr(d.contact, field.name, array)
      value = array.numpy()
      value[:] = rng.integers(0, 100, value.shape)
      reference[field.name] = value
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
    d.nacon.fill_(count)
    for _ in range(3):
      permutation = rng.permutation(count)
      for name, value in reference.items():
        shuffled = value.copy()
        shuffled[:count] = value[permutation]
        getattr(d.contact, name).assign(shuffled)
      collision_driver._sort_contacts(m, d)
      for name, value in reference.items():
        np.testing.assert_array_equal(getattr(d.contact, name).numpy(), value, err_msg=name)

  @parameterized.parameters(
    *itertools.product((1, 2), ("dense", "sparse"), ("connect", "weld", "interleaved"), ("solved", "forward"))
  )
  def test_equality_forces(self, nworld, jacobian, equality, stage):
    """Sorting preserves equality components and the subtype layout used by force sensors."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="1 2 -3" tolerance="1e-10"><flag contact="disable"/></option>
        <worldbody>
          <body name="a" pos="0 0 1" quat=".5 .5 .5 .5">
            <freejoint/><geom type="box" size=".1 .12 .15" mass="1"/><site name="sa" pos=".02 .03 .04"/>
          </body>
          <body name="b" pos="1 0 1" quat=".5 -.5 .5 .5">
            <freejoint/><geom type="box" size=".12 .15 .1" mass="2"/><site name="sb" pos="-.03 .02 .01"/>
          </body>
          <body name="c" pos="2 0 1" quat=".5 .5 -.5 .5">
            <freejoint/><geom type="box" size=".15 .1 .12" mass="1.5"/><site name="sc"/>
          </body>
          <body name="d" pos="3 0 1" quat=".5 .5 .5 -.5">
            <freejoint/><geom type="box" size=".1 .15 .12" mass=".8"/><site name="sd"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="b" anchor="1 0 1"/><connect body1="a" anchor=".03125 -.0625 .015625"/>
          <weld body1="d" anchor="3 0 1"/><connect body1="c" anchor="-.0625 .015625 .03125"/>
        </equality>
        <sensor>
          <force site="sa"/><torque site="sa"/><force site="sb"/><torque site="sb"/>
          <force site="sc"/><torque site="sc"/><force site="sd"/><torque site="sd"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=0,
      njmax=32,
      overrides={"opt.jacobian": jacobian},
    )
    # Unequal spatial loads and nonzero velocities expose swaps of both force and torque components.
    qvel = d.qvel.numpy()
    applied = d.xfrc_applied.numpy()
    active = d.eq_active.numpy()
    for world in range(nworld):
      for body in range(4):
        qvel[world, 6 * body : 6 * body + 3] = np.array([0.01, -0.02, 0.03]) * (1 + 0.25 * world)
      applied[world, 1:] = np.arange(1, mjm.nbody)[:, None] * np.array([0.7, -1.1, 1.9, 0.13, 0.23, -0.31]) * (1 + 0.25 * world)
      if equality != "interleaved":
        active[world] = mjm.eq_type == getattr(mujoco.mjtEq, "mjEQ_" + equality.upper())
    if nworld == 2:
      # Different active group counts also change the starting offset of the weld partition.
      active[1, 3] = False
    d.qvel.assign(qvel)
    d.xfrc_applied.assign(applied)
    d.eq_active.assign(active)
    fields = ("qacc", "qfrc_constraint", "cfrc_ext", "cfrc_int", "sensordata")
    expected = {name: [] for name in fields}
    for world in range(nworld):
      mjd.qvel[:] = qvel[world]
      mjd.xfrc_applied[:] = applied[world]
      mjd.eq_active[:] = active[world]
      mujoco.mj_forward(mjm, mjd)
      mujoco.mj_rnePostConstraint(mjm, mjd)
      for name in fields:
        expected[name].append(getattr(mjd, name).copy())

    def check():
      for name in fields:
        np.testing.assert_allclose(getattr(d, name).numpy(), expected[name], rtol=1e-5, atol=1e-5, err_msg=name)

    m.opt.deterministic = mjw.DeterminismType.NONE
    for name in fields:
      getattr(d, name).fill_(wp.inf)
    mjw.forward(m, d)
    check()
    if stage == "solved":
      # Isolate the permutation from the solver: the solved forces must retain their XYZ meaning.
      # Reverse whole groups to exercise group ordering without scrambling their components.
      for world, n in enumerate(d.nefc.numpy()):
        order = np.argsort(-d.efc.id.numpy()[world, :n], kind="stable")
        for field in dataclasses.fields(d.efc):
          if field.name.startswith("jtdaj_") or field.name == "Ma":
            continue
          if m.is_sparse and field.name in ("J", "J_colind"):
            continue
          array = getattr(d.efc, field.name)
          if array.size:
            values = array.numpy()
            values[world, :n] = values[world, order]
            array.assign(values)
        if m.is_sparse:
          nb = int(d.efc.jtdaj_nblock.numpy()[world])
          starts = d.efc.jtdaj_adr.numpy()
          starts[world, :nb] = np.argsort(order)[starts[world, :nb]]
          d.efc.jtdaj_adr.assign(starts)
      constraint._sort_constraints(m, d)
      for name in ("cfrc_ext", "cfrc_int", "sensordata"):
        getattr(d, name).fill_(wp.inf)
      mjw.rne_postconstraint(m, d)
      mjw.sensor_acc(m, d)
    else:
      m.opt.deterministic = mjw.DeterminismType.CONSTRAINT
      for name in fields:
        getattr(d, name).fill_(wp.inf)
      mjw.forward(m, d)
    check()
    if nworld == 2:
      for name in fields:
        self.assertFalse(np.array_equal(getattr(d, name).numpy()[0], getattr(d, name).numpy()[1]), msg=name)

    for world in range(nworld):
      eqids = sorted(np.flatnonzero(active[world]), key=lambda eqid: (mjm.eq_type[eqid], eqid))
      sizes = [3 if mjm.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT else 6 for i in eqids]
      n = int(d.nefc.numpy()[world])
      np.testing.assert_array_equal(d.efc.id.numpy()[world, :n], np.repeat(eqids, sizes))
      if m.is_sparse:
        nb = int(d.efc.jtdaj_nblock.numpy()[world])
        starts = d.efc.jtdaj_adr.numpy()[world, :nb]
        widths = d.efc.jtdaj_nrow.numpy()[world, :nb]
        np.testing.assert_array_equal(widths, sizes)
        np.testing.assert_array_equal(starts, np.cumsum(sizes) - sizes)
        # Every remapped block must still cover exactly one equality, with common sparse support.
        nnz = d.efc.J_rownnz.numpy()[world]
        adr = d.efc.J_rowadr.numpy()[world]
        cols = d.efc.J_colind.numpy()[world, 0]
        for start, width in zip(starts, widths):
          for row in range(start + 1, start + width):
            np.testing.assert_array_equal(cols[adr[row] : adr[row] + nnz[row]], cols[adr[start] : adr[start] + nnz[start]])

  @parameterized.parameters("dense", "sparse")
  def test_flex_equalities(self, jacobian):
    """Flex edge rows sharing an equality ID are ordered by their Jacobians, not allocation."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco><worldbody>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" mass="1" pos="0 0 1">
          <edge equality="true"/>
          <contact contype="0" conaffinity="0"/>
        </flexcomp>
      </worldbody></mujoco>
      """,
      nworld=2,
      njmax=64,
      qpos_noise=0.01,
      qvel_noise=0.1,
      overrides={"opt.jacobian": jacobian},
    )
    mjw.forward(m, d)
    baseline = d.qacc.numpy()
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
      xml="""<mujoco><worldbody><body><joint frictionloss=".1"/><geom size=".1"/></body></worldbody></mujoco>""",
      nworld=2,
      njmax=4,
      overrides={"opt.jacobian": "sparse"},
    )
    mjw.forward(m, d)
    nnz = wp.full(d.nworld, d.njmax_nnz + 1 if overflow == "nonzeros" else 1, dtype=int)
    if overflow == "rows":
      d.nefc.fill_(d.njmax + 1)
    before = [(a, a.numpy()) for a in _arrays(d)]
    constraint._sort_constraints(m, d, nnz)
    for a, value in before:
      np.testing.assert_array_equal(a.numpy(), value)

  @parameterized.parameters(0, 1, 7, 8, 13)
  def test_contact_count_bounds(self, count):
    """Zero, partial, full, and overflowing buffers preserve counts and inactive slots."""
    _, _, m, d = test_data.fixture(
      xml="""<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>""",
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


def _arrays(value):
  for field in dataclasses.fields(value):
    item = getattr(value, field.name)
    if isinstance(item, wp.array):
      yield item
    elif dataclasses.is_dataclass(item):
      yield from _arrays(item)


def _assert_collision_repeat(test, m, d):
  reference = None
  for _ in range(4):
    mjw.collision(m, d)
    constraint.make_constraint(m, d)
    count = int(d.nacon.numpy()[0])
    test.assertGreater(count, d.nworld)
    test.assertLessEqual(count, d.naconmax)
    test.assertTrue(np.all(d.nefc.numpy() <= d.njmax))
    current = {}
    for f in dataclasses.fields(d.contact):
      a = getattr(d.contact, f.name)
      if a.size:
        current[f.name] = a.numpy()[:count]
    for world in range(d.nworld):
      current[world, "J"] = _dense_jacobian(m, d, world)
    if reference is not None:
      for name, value in current.items():
        np.testing.assert_array_equal(value, reference[name], err_msg=str(name))
    reference = current


def _dense_jacobian(m, d, world):
  n = int(d.nefc.numpy()[world])
  if not m.is_sparse:
    return d.efc.J.numpy()[world, :n, : m.nv]
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
