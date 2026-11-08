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
"""Constraint row permutation preserves axes, sparse storage and contact addresses."""

import dataclasses

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src.constraint import _sort_constraints


class ConstraintSortTest(parameterized.TestCase):
  @parameterized.product(nworld=(1, 2), sparse=(False, True), elliptic=(False, True))
  def test_permutation(self, nworld, sparse, elliptic):
    """Rows, contact addresses and Newton blocks survive shuffled object groups."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <compiler angle="radian"/>
        <worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body name="box0" pos="0 0 .09" euler=".03 .02 0">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
          </body>
          <body name="box1" pos=".4 0 .08" euler=".02 .03 0">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
            <body pos="0 0 .3">
              <joint name="hinge" type="hinge" axis="1 0 0" range="-.1 .1" frictionloss=".05"/>
              <joint name="hinge2" type="hinge" axis="0 1 0" range="-.1 .1" frictionloss=".07"/>
              <geom type="capsule" size=".03 .1"/>
            </body>
          </body>
        </worldbody>
        <equality><connect body1="box0" body2="box1" anchor=".2 0 .1"/></equality>
      </mujoco>
      """,
      overrides={"opt.jacobian": "sparse" if sparse else "dense", "opt.cone": "elliptic" if elliptic else "pyramidal"},
      nworld=nworld,
      nconmax=32,
      njmax=128,
      qvel_noise=0.01,
    )
    qpos = np.tile(mjd.qpos, (nworld, 1))
    if nworld == 2:
      qpos[1, 0] += 0.5
      qpos[1, 2] -= 0.005
    qpos[:, -1] = 0.2  # Exercise joint-limit and DOF-friction rows as well as equality/contact.
    d.qpos.assign(qpos)
    mjw.forward(m, d)
    baseline_qacc = d.qacc.numpy().copy()
    counts = d.nefc.numpy().copy()
    ncon = int(d.nacon.numpy()[0])
    worlds = d.contact.worldid.numpy().copy()
    original = {f.name: getattr(d.efc, f.name).numpy().copy() for f in dataclasses.fields(d.efc)}
    addresses = d.contact.efc_address.numpy().copy()
    friction_addresses = d.dof_frictionadr.numpy().copy()
    row_fields = ("type", "id", "pos", "margin", "D", "vel", "aref", "frictionloss", "Jqvel")
    expected = None
    for seed in range(4):
      arrays = {k: v.copy() for k, v in original.items()}
      address = addresses.copy()
      friction_address = friction_addresses.copy()
      for world in range(nworld):
        count = int(counts[world])
        groups = {}
        for row in range(count):
          key = (int(original["type"][world, row]), int(original["id"][world, row]))
          groups.setdefault(key, []).append(row)
        blocks = list(groups.values())
        rng = np.random.default_rng(seed)
        permutation = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        inverse = np.argsort(permutation)
        valid_friction = friction_address[world] >= 0
        friction_address[world, valid_friction] = inverse[friction_address[world, valid_friction]]
        for name in row_fields:
          arrays[name][world, :count] = original[name][world, permutation]
        if sparse:
          for name in ("J_rowadr", "J_rownnz"):
            arrays[name][world, :count] = original[name][world, permutation]
          nblock = int(original["jtdaj_nblock"][world])
          block_perm = rng.permutation(nblock)
          arrays["jtdaj_adr"][world, :nblock] = inverse[original["jtdaj_adr"][world, block_perm]]
          arrays["jtdaj_nrow"][world, :nblock] = original["jtdaj_nrow"][world, block_perm]
        else:
          arrays["J"][world, :count] = original["J"][world, permutation]
        for contact in range(ncon):
          if worlds[contact] == world:
            valid = address[contact] >= 0
            address[contact, valid] = inverse[address[contact, valid]]
      for name, array in arrays.items():
        if array.size:
          getattr(d.efc, name).assign(array)
      d.contact.efc_address.assign(address)
      d.dof_frictionadr.assign(friction_address)
      _sort_constraints(m, d)
      result = {
        name: getattr(d.efc, name).numpy().copy()
        for name in (*row_fields, "J", "J_rowadr", "J_rownnz", "J_colind", "jtdaj_adr", "jtdaj_nrow")
      }
      result["friction_address"] = d.dof_frictionadr.numpy().copy()
      result["contact_address"] = d.contact.efc_address.numpy().copy()
      if expected is None:
        expected = result
      np.testing.assert_array_equal(result["contact_address"][:ncon], expected["contact_address"][:ncon])
      np.testing.assert_array_equal(result["friction_address"], expected["friction_address"])
      for world in range(nworld):
        count = int(counts[world])
        for name in row_fields:
          np.testing.assert_array_equal(result[name][world, :count], expected[name][world, :count], err_msg=name)
        if sparse:
          nnz = int(d.efc.J_rownnz.numpy()[world, :count].sum())
          for name in ("J", "J_colind"):
            np.testing.assert_array_equal(result[name][world, 0, :nnz], expected[name][world, 0, :nnz], err_msg=name)
          for name in ("J_rowadr", "J_rownnz"):
            np.testing.assert_array_equal(result[name][world, :count], expected[name][world, :count], err_msg=name)
          nblock = int(original["jtdaj_nblock"][world])
          for name in ("jtdaj_adr", "jtdaj_nrow"):
            np.testing.assert_array_equal(result[name][world, :nblock], expected[name][world, :nblock], err_msg=name)
        else:
          np.testing.assert_array_equal(result["J"][world, :count], expected["J"][world, :count])
    m.opt.deterministic = mjw.DeterminismType.CONSTRAINT
    mjw.forward(m, d)
    np.testing.assert_allclose(d.qacc.numpy(), baseline_qacc, atol=1e-4, rtol=1e-4)
    # Capture the canonicalizer after a fresh forward pass.
    expected["contact_address"] = d.contact.efc_address.numpy().copy()
    if wp.get_device().is_cuda:
      with wp.ScopedCapture() as capture:
        _sort_constraints(m, d)
      wp.capture_launch(capture.graph)
      np.testing.assert_array_equal(d.contact.efc_address.numpy()[:ncon], expected["contact_address"][:ncon])


if __name__ == "__main__":
  wp.init()
  absltest.main()
