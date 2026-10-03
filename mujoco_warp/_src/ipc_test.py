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
"""Tests for barrier-free augmented-Lagrangian IPC contact (mj_ipc)."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import collision_continuous as cc
from mujoco_warp._src import ipc
from mujoco_warp._src.types import OverflowType


def _pt_tri_closest(p, a, b, c):
  ab = b - a
  ac = c - a
  ap = p - a
  d1 = float(np.dot(ab, ap))
  d2 = float(np.dot(ac, ap))
  if d1 <= 0.0 and d2 <= 0.0:
    return float(np.linalg.norm(p - a)), a
  bp = p - b
  d3 = float(np.dot(ab, bp))
  d4 = float(np.dot(ac, bp))
  if d3 >= 0.0 and d4 <= d3:
    return float(np.linalg.norm(p - b)), b
  vc = d1 * d4 - d3 * d2
  if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
    t = d1 / (d1 - d3)
    cp = a + ab * t
    return float(np.linalg.norm(p - cp)), cp
  cp_v = p - c
  d5 = float(np.dot(ab, cp_v))
  d6 = float(np.dot(ac, cp_v))
  if d6 >= 0.0 and d5 <= d6:
    return float(np.linalg.norm(p - c)), c
  vb = d5 * d2 - d1 * d6
  if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
    t = d2 / (d2 - d6)
    cp = a + ac * t
    return float(np.linalg.norm(p - cp)), cp
  va = d3 * d6 - d5 * d4
  if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
    t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    cp = b + (c - b) * t
    return float(np.linalg.norm(p - cp)), cp
  den = 1.0 / (va + vb + vc)
  t_b = vb * den
  u_c = vc * den
  cp = a * (1.0 - t_b - u_c) + b * t_b + c * u_c
  return float(np.linalg.norm(p - cp)), cp


class IpcTest(parameterized.TestCase):
  @parameterized.parameters(1, 2)
  def test_free_fall(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1"
                    spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.5"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.01
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    h = float(m.opt.timestep.numpy()[0])
    g = float(m.opt.gravity.numpy()[0][2])
    want_v = g * h

    mjw.step(m, d)

    for w in range(nworld):
      qvel = d.qvel.numpy()[w]
      qpos = d.qpos.numpy()[w]
      for i in range(2, m.nv, 3):
        np.testing.assert_allclose(qvel[i], want_v, atol=1e-5)
      self.assertFalse(np.isnan(qpos[0]))

    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_contact_blocks_tunneling(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
                    radius="0.005" mass="0.05" pos="0 0 0.05"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.01
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    # Drive vertices straight down at 50 m/s
    qvel = d.qvel.numpy()
    for w in range(nworld):
      for i in range(2, m.nv, 3):
        qvel[w, i] = -50.0
    d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)

    mjw.step(m, d)

    for w in range(nworld):
      flexvert_xpos = d.flexvert_xpos.numpy()[w]
      minz = float(np.min(flexvert_xpos[:, 2]))
      self.assertGreater(minz, 0.0)
      self.assertFalse(np.isnan(minz))

    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_mesh_geom_blocks_tunneling(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <asset>
          <mesh name="box" vertex="
            -0.2 -0.2 -0.02  0.2 -0.2 -0.02  0.2 0.2 -0.02  -0.2 0.2 -0.02
            -0.2 -0.2 0.02  0.2 -0.2 0.02  0.2 0.2 0.02  -0.2 0.2 0.02"/>
        </asset>
        <worldbody>
          <geom name="mesh_obs" type="mesh" mesh="box"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
                    radius="0.005" mass="0.05" pos="0 0 0.07"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.01
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    qvel = d.qvel.numpy()
    for w in range(nworld):
      for i in range(2, m.nv, 3):
        qvel[w, i] = -50.0
    d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)

    mjw.step(m, d)

    for w in range(nworld):
      flexvert_xpos = d.flexvert_xpos.numpy()[w]
      minz = float(np.min(flexvert_xpos[:, 2]))
      self.assertGreater(minz, 0.02)
      self.assertFalse(np.isnan(minz))

    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_flexvert_lambda_warmstart(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.05" pos="0 0 0.0008">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.0001
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    mjw.step(m, d)

    for w in range(nworld):
      lam = d.flexvert_lambda.numpy()[w]
      self.assertFalse(np.isnan(lam).any())
      self.assertGreater(float(np.max(lam)), 0.0)

    if nworld == 2:
      self.assertFalse(np.allclose(d.flexvert_lambda.numpy()[0], d.flexvert_lambda.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_constraint_force_reported(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option iterations="400" integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.05" pos="0 0 0.01">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.001
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    d.qfrc_constraint.fill_(wp.inf)
    for _ in range(50):
      mjw.step(m, d)

    expected_weight = 0.05 * 9.81
    for w in range(nworld):
      qfrc = d.qfrc_constraint.numpy()[w]
      self.assertFalse(np.isinf(qfrc).any())
      total_z_force = float(np.sum(qfrc[2::3]))
      np.testing.assert_allclose(total_z_force, expected_weight, rtol=0.05)

    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_contact_recovers_from_penetration(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1"
                    radius="0.005" mass="0.05" pos="0 0 -0.001"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.0005
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    mjw.step(m, d)

    for w in range(nworld):
      qvel = d.qvel.numpy()[w]
      for i in range(2, m.nv, 3):
        self.assertGreater(qvel[i], 0.0)

    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  def _flex_momentum(self, m: mjw.Model, d: mjw.Data, world_id: int = 0):
    mass = 0.0
    P = np.zeros(3)
    body_dofadr = m.body_dofadr.numpy()
    body_dofnum = m.body_dofnum.numpy()
    body_mass = m.body_mass.numpy()[world_id % m.body_mass.shape[0]]
    qvel = d.qvel.numpy()[world_id]
    for b in range(1, m.nbody):
      adr = int(body_dofadr[b])
      if adr < 0 or int(body_dofnum[b]) != 3:
        continue
      mb = float(body_mass[b])
      mass += mb
      P += mb * qvel[adr : adr + 3]
    return mass, P

  @parameterized.parameters(1, 2)
  def test_self_contact_conserves_momentum(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option iterations="400" integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <flexcomp name="lower" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.02" pos="0 0 0.5"/>
          <flexcomp name="upper" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.02" pos="0 0 0.54"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.01
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    upper = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_FLEX, "upper")
    flex_vertnum = mjm.flex_vertnum[upper]
    flex_vertadr = mjm.flex_vertadr[upper]
    qvel = d.qvel.numpy()
    for w in range(nworld):
      for i in range(flex_vertnum):
        bid = mjm.flex_vertbodyid[flex_vertadr + i]
        for j in range(mjm.body_jntnum[bid]):
          jid = mjm.body_jntadr[bid] + j
          if mjm.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE and mjm.jnt_axis[jid, 2] > 0.5:
            qvel[w, mjm.jnt_dofadr[jid]] = -2.0
    d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)

    P0_list = []
    masses = []
    for w in range(nworld):
      mass_w, P0_w = self._flex_momentum(m, d, world_id=w)
      self.assertGreater(mass_w, 0.0)
      masses.append(mass_w)
      P0_list.append(P0_w)
    h = float(m.opt.timestep.numpy()[0])
    g = np.array([float(m.opt.gravity.numpy()[0][k]) for k in range(3)])

    for _ in range(40):
      mjw.step(m, d)
      for w in range(nworld):
        _, P1_w = self._flex_momentum(m, d, world_id=w)
        self.assertFalse(np.isnan(P1_w[2]))
        scale = masses[w] * float(np.linalg.norm(g)) * h
        for k in range(3):
          got = P1_w[k] - P0_list[w][k]
          want = h * masses[w] * g[k]
          np.testing.assert_allclose(got, want, atol=1e-2 * scale)
        P0_list[w] = P1_w

    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_native_collision_skips_the_modes_pairs(self, nworld):
    for ipc_enable in (False, True):
      mjm, mjd, m, d = test_data.fixture(
        xml=f"""
        <mujoco>
          <option integrator="discrete" solver="CG">
            {'<flag ipc="enable"/>' if ipc_enable else ""}
          </option>
          <worldbody>
            <geom name="floor" type="plane" size="0 0 1"/>
            <flexcomp name="cloth" type="grid" dim="2" count="5 5 1" spacing="0.04 0.04 1"
                      radius="0.004" mass="0.1" pos="0 0 0.004">
              <edge equality="true"/>
              <contact selfcollide="none"/>
            </flexcomp>
            <body name="ball" pos="0 0 0.05">
              <freejoint/>
              <geom size="0.02" mass="0.05"/>
            </body>
          </worldbody>
        </mujoco>
        """,
        nworld=nworld,
      )
      if nworld == 2:
        qpos = d.qpos.numpy()
        qpos[1] += 0.005
        d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

      for _ in range(100):
        mjw.step(m, d)

      ncon = int(d.nacon.numpy()[0])
      floor_contacts = 0
      ball_contacts = 0
      geom_bodyid = m.geom_bodyid.numpy()
      contact_geom = d.contact.geom.numpy()
      contact_flex = d.contact.flex.numpy()
      for c in range(ncon):
        f0 = int(contact_flex[c, 0])
        f1 = int(contact_flex[c, 1])
        if (f0 >= 0) == (f1 >= 0):
          continue
        g0 = int(contact_geom[c, 0])
        g1 = int(contact_geom[c, 1])
        g = g0 if g0 >= 0 else g1
        if g < 0:
          continue
        if int(geom_bodyid[g]) == 0:
          floor_contacts += 1
        else:
          ball_contacts += 1

      if ipc_enable:
        self.assertEqual(floor_contacts, 0)
      else:
        self.assertGreater(floor_contacts, 0)
      self.assertGreater(ball_contacts, 0)
      if nworld == 2:
        self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  def test_rejects_incompatible_options(self):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <body pos="0 0 1">
            <freejoint/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )

    mjw.forward(m, d)

    m.opt.integrator = mjw.IntegratorType.EULER
    with self.assertRaises(ValueError):
      mjw.forward(m, d)
    m.opt.integrator = mjw.IntegratorType.DISCRETE

    m.opt.solver = mjw.SolverType.NEWTON
    with self.assertRaises(ValueError):
      mjw.forward(m, d)
    m.opt.solver = mjw.SolverType.CG

    m.opt.enableflags |= mjw.EnableBit.SLEEP
    with self.assertRaises(ValueError):
      mjw.forward(m, d)
    m.opt.enableflags &= ~mjw.EnableBit.SLEEP

    with self.assertRaisesRegex(ValueError, "discrete inverse dynamics is not supported with flag ipc"):
      mjw.inverse(m, d)

    mjw.forward(m, d)

  @parameterized.parameters(1, 2)
  def test_native_rows_excluded_for_ipc_pairs(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="table" type="box" size="0.3 0.3 0.05" pos="0 0 0.05"/>
          <flexcomp name="sheet" type="grid" dim="2" count="5 5 1" spacing="0.04 0.04 1" radius="0.004"
                    mass="0.05" pos="0 0 0.13">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
          <body pos="0 0 0.2"><freejoint/><geom size="0.03" mass="0.2"/></body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.005
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    flex_static = 0
    flex_moving = 0
    body_weldid = m.body_weldid.numpy()
    geom_bodyid = m.geom_bodyid.numpy()
    for _ in range(100):
      mjw.step(m, d)
      ncon = int(d.nacon.numpy()[0])
      contact_geom = d.contact.geom.numpy()
      contact_flex = d.contact.flex.numpy()
      for i in range(ncon):
        f0 = int(contact_flex[i, 0])
        f1 = int(contact_flex[i, 1])
        if (f0 >= 0) == (f1 >= 0):
          continue
        g0 = int(contact_geom[i, 0])
        g1 = int(contact_geom[i, 1])
        g = g0 if g0 >= 0 else g1
        if g >= 0:
          if int(body_weldid[int(geom_bodyid[g])]) == 0:
            flex_static += 1
          else:
            flex_moving += 1

    self.assertEqual(flex_static, 0)
    self.assertGreater(flex_moving, 0)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_native_rows_kept_for_unsupported_geoms(self, nworld):
    geoms = [
      ("cylinder", "0.2 0.1", False),
      ("ellipsoid", "0.2 0.2 0.1", False),
      ("box", "0.2 0.2 0.1", True),
    ]
    for gtype, gsize, supported in geoms:
      mjm, mjd, m, d = test_data.fixture(
        xml=f"""
        <mujoco>
          <option integrator="discrete" solver="CG">
            <flag ipc="enable"/>
          </option>
          <worldbody>
            <geom name="drum" type="{gtype}" size="{gsize}" pos="0 0 0.2"/>
            <flexcomp name="sheet" type="grid" dim="2" count="5 5 1" spacing="0.04 0.04 1" radius="0.004"
                      mass="0.05" pos="0 0 0.35">
              <edge equality="true"/>
              <contact selfcollide="none"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        nworld=nworld,
      )
      if nworld == 2:
        qpos = d.qpos.numpy()
        qpos[1] += 0.005
        d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

      native = 0
      for _ in range(120):
        mjw.step(m, d)
        native += int(d.nacon.numpy()[0])

      self.assertEqual(native > 0, not supported, f"{gtype}: native contacts only for unsupported types")
      for w in range(nworld):
        zmin = float(np.min(d.flexvert_xpos.numpy()[w, :, 2]))
        self.assertGreater(zmin, 0.2, f"{gtype}: sheet rests on top in world {w}")
      if nworld == 2:
        self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @absltest.skipIf(not wp.get_device().is_cuda, "CUDA required for graph capture test")
  @parameterized.parameters(1, 2)
  def test_cuda_graph_capture_without_warmup(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" integrator="discrete">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="1 1 0.1"/>
          <flexcomp name="sheet" type="grid" dim="2" count="4 4 1" spacing="0.04 0.04 1" radius="0.004"
                    mass="0.05" pos="0 0 0.1">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.01
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    for _ in range(5):
      wp.capture_launch(capture.graph)
    wp.synchronize()

    self.assertTrue(np.all(np.isfinite(d.qpos.numpy())))
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_pinned_flex_body_integrates(self, nworld):
    """A body carrying pinned flex vertices is integrated like any articulated body."""
    q = []
    for flag in ['<flag ipc="enable"/>', ""]:
      _, _, m, d = test_data.fixture(
        xml=f"""
        <mujoco>
          <option integrator="discrete" solver="CG" iterations="400">
            {flag}
          </option>
          <worldbody>
            <geom name="floor" type="plane" size="0 0 1"/>
            <body name="arm" pos="0 0 0.2">
              <joint name="carrier" type="slide" stiffness="50" damping="2"/>
              <geom type="capsule" fromto="0 0 0  0.1 0 0" size="0.005" mass="0.02" contype="0" conaffinity="0"/>
              <flexcomp name="paddle" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                        radius="0.004" mass="0.05" pos="0.06 0 0">
                <contact selfcollide="none"/>
                <pin id="0 1 2 3 4 5 6 7 8"/>
              </flexcomp>
            </body>
          </worldbody>
        </mujoco>
        """,
        nworld=nworld,
      )
      qpos = d.qpos.numpy()
      qpos[0, 0] = -0.05
      if nworld == 2:
        qpos[1, 0] = -0.02
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

      for step_i in range(200):
        mjw.step(m, d)
        if step_i == 10 and nworld == 2:
          self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))
      q.append(d.qpos.numpy()[:, 0].copy())

    for w in range(nworld):
      q0_init = -0.05 if w == 0 else -0.02
      self.assertGreater(abs(float(q[1][w]) - q0_init), 0.005)
      np.testing.assert_allclose(q[0][w], q[1][w], atol=1e-4)

  @parameterized.parameters(1, 2)
  def test_slide_carrier_moves_once(self, nworld):
    """A body carrying pinned flex vertices moves by h*v per step, not 9*h*v."""
    qx = []
    cx = []
    for flag in ['<flag ipc="enable"/>', ""]:
      _, _, m, d = test_data.fixture(
        xml=f"""
        <mujoco>
          <option integrator="discrete" solver="CG" gravity="0 0 0">
            {flag}
          </option>
          <worldbody>
            <body name="carrier" pos="0 0 0.5">
              <joint type="slide" axis="1 0 0"/>
              <joint type="slide" axis="0 1 0"/>
              <joint type="slide"/>
              <geom size="0.01" mass="0.1" contype="0" conaffinity="0"/>
              <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.03 0.03 1"
                        radius="0.004" mass="0.05" pos="0.06 0 0">
                <pin id="0 1 2 3 4 5 6 7 8"/>
                <edge equality="true"/>
                <contact selfcollide="none"/>
              </flexcomp>
            </body>
          </worldbody>
        </mujoco>
        """,
        nworld=nworld,
      )
      qvel = np.zeros((nworld, m.nv), dtype=np.float32)
      qvel[:, 0] = 1.0
      if nworld == 2:
        qvel[1, 0] = 2.0
      d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)
      for _ in range(10):
        mjw.step(m, d)
      qx.append(d.qpos.numpy()[:, 0].copy())
      cx.append(np.mean(d.flexvert_xpos.numpy()[:, :, 0], axis=1))
      for w in range(nworld):
        want_v = 1.0 if w == 0 else 2.0
        np.testing.assert_allclose(d.qvel.numpy()[w, 0], want_v, atol=1e-5)

    for w in range(nworld):
      want_x = 0.02 if w == 0 else 0.04
      np.testing.assert_allclose(qx[0][w], want_x, atol=1e-5)
      np.testing.assert_allclose(qx[0][w], qx[1][w], atol=1e-5)
      np.testing.assert_allclose(cx[0][w], cx[1][w], atol=1e-5)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_pinned_paddle_carries_cloth(self, nworld):
    """Free cloth dropped on pinned paddle loads arm: sag increases."""
    q = []
    for with_cloth in (False, True):
      _, _, m, d = test_data.fixture(
        xml=f"""
        <mujoco>
          <option integrator="discrete" solver="CG" iterations="400">
            <flag ipc="enable"/>
          </option>
          <worldbody>
            <geom name="floor" type="plane" size="0 0 1"/>
            <body name="arm" pos="0 0 0.2">
              <joint name="carrier" type="slide" stiffness="50" damping="2"/>
              <geom type="capsule" fromto="0 0 0  0.1 0 0" size="0.005" mass="0.02" contype="0" conaffinity="0"/>
              <flexcomp name="paddle" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                        radius="0.004" mass="0.05" pos="0.06 0 0">
                <contact selfcollide="none"/>
                <pin id="0 1 2 3 4 5 6 7 8"/>
              </flexcomp>
            </body>
            {'<flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.03 0.03 1" radius="0.004" mass="0.05" pos="0.06 0 0.24"><edge equality="true"/><contact selfcollide="none"/></flexcomp>' if with_cloth else ""}
          </worldbody>
        </mujoco>
        """,
        nworld=nworld,
      )
      if nworld == 2:
        qpos = d.qpos.numpy()
        qpos[1, 0] = -0.01
        d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)
      for _ in range(200):
        mjw.step(m, d)
      q.append(d.qpos.numpy()[:, 0].copy())

    for w in range(nworld):
      self.assertGreater(abs(float(q[0][w])), 1e-3)
      self.assertGreater(abs(float(q[1][w])) - abs(float(q[0][w])), 3e-3)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_rigid_no_contact_matches_discrete(self, nworld):
    """Rigid bodies with no contact follow identical trajectory with and without IPC flag."""
    _, _, m1, d1 = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.005" integrator="discrete" solver="CG"/>
        <worldbody>
          <body pos="0 0 1">
            <joint type="free"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    _, _, m2, d2 = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.005" integrator="discrete" solver="CG"/>
        <worldbody>
          <body pos="0 0 1">
            <joint type="free"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    m2.opt.enableflags |= mjw.EnableBit.IPC

    if nworld == 2:
      qpos1 = d1.qpos.numpy()
      qpos1[1, 0] += 0.05
      d1.qpos = wp.array(qpos1, dtype=float, device=d1.qpos.device)
      qpos2 = d2.qpos.numpy()
      qpos2[1, 0] += 0.05
      d2.qpos = wp.array(qpos2, dtype=float, device=d2.qpos.device)

    for _ in range(20):
      mjw.step(m1, d1)
      mjw.step(m2, d2)

    np.testing.assert_allclose(d1.qpos.numpy(), d2.qpos.numpy(), atol=1e-6)
    np.testing.assert_allclose(d1.qvel.numpy(), d2.qvel.numpy(), atol=1e-6)
    if nworld == 2:
      self.assertFalse(np.allclose(d1.qpos.numpy()[0], d1.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_acceleration_sensors_see_contact(self, nworld):
    """Accelerometer on body contacting floor reads reaction force post-commit."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG"/>
        <worldbody>
          <geom type="plane" size="0 0 1"/>
          <body name="ball" pos="0 0 0.1">
            <freejoint/>
            <geom size="0.1" mass="1"/>
            <site name="imu"/>
          </body>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
                    radius="0.005" mass="0.05" pos="1 0 0.5">
            <contact contype="0" conaffinity="0"/>
            <pin id="0 1 2 3 4 5 6 7 8"/>
          </flexcomp>
        </worldbody>
        <sensor>
          <accelerometer site="imu"/>
          <framelinacc objtype="body" objname="ball"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
    )
    m.opt.enableflags |= mjw.EnableBit.IPC

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1, 0] += 0.05
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    d.cacc.fill_(wp.inf)
    d.sensordata.fill_(wp.inf)

    for _ in range(50):
      mjw.step(m, d)

    acc = d.sensordata.numpy()
    self.assertFalse(np.isinf(acc).any())
    for w in range(nworld):
      self.assertAlmostEqual(float(acc[w, 2]), 9.81, delta=0.2)
      self.assertAlmostEqual(float(acc[w, 5]), 9.81, delta=0.2)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_smooth_geoms_cover_triangle_interior(self, nworld):
    """Small obstacle under cloth triangle interior tents the cloth upward."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG" iterations="400">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <geom name="obstacle" size="0.005" pos="0.021 0.019 0.02"/>
          <flexcomp name="cloth" type="grid" dim="2" count="5 5 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.1" pos="0 0 0.05">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.005
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    for _ in range(80):
      mjw.step(m, d)

    f_vertadr = int(mjm.flex_vertadr[0])
    f_vertnum = int(mjm.flex_vertnum[0])
    f_elemadr = int(mjm.flex_elemdataadr[0])
    f_elemnum = int(mjm.flex_elemnum[0])
    flex_elem = mjm.flex_elem[f_elemadr : f_elemadr + 3 * f_elemnum].reshape(-1, 3)

    for w in range(nworld):
      xv = d.flexvert_xpos.numpy()[w, f_vertadr : f_vertadr + f_vertnum]
      p = np.array([0.021, 0.019, 0.02])
      best_dist = 1e30
      best_cp = np.zeros(3)
      for e in range(f_elemnum):
        a = xv[flex_elem[e, 0]]
        b = xv[flex_elem[e, 1]]
        c = xv[flex_elem[e, 2]]
        dd, cp = _pt_tri_closest(p, a, b, c)
        if dd < best_dist:
          best_dist = dd
          best_cp = cp
      self.assertGreater(best_cp[2], p[2])
      self.assertGreater(best_dist, 0.9 * 0.005)
    if nworld == 2:
      self.assertFalse(np.allclose(d.flexvert_xpos.numpy()[0], d.flexvert_xpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_pinned_paddle_stops_on_plane(self, nworld):
    """Pinned paddle on vertical slide joint falls under gravity and stops on floor."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <body name="paddle_carrier" pos="0 0 0.05">
            <joint type="slide" damping="0.05"/>
            <geom size="0.02" mass="0.1"/>
            <flexcomp name="paddle" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                      radius="0.004" mass="0.05">
              <pin id="0 1 2 3 4 5 6 7 8"/>
            </flexcomp>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qvel = d.qvel.numpy()
      qvel[1, 0] = -1.5
      d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)

    for _ in range(100):
      mjw.step(m, d)

    f_vertadr = int(mjm.flex_vertadr[0])
    f_vertnum = int(mjm.flex_vertnum[0])
    verts = d.flexvert_xpos.numpy()[:, f_vertadr : f_vertadr + f_vertnum]
    for w in range(nworld):
      zmin = float(np.min(verts[w, :, 2]))
      carrier_z = float(d.qpos.numpy()[w, 0])
      self.assertGreater(zmin, -0.005)
      self.assertLess(carrier_z, -0.03)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_articulated_capsule_does_not_tunnel(self, nworld):
    """Articulated capsule on hinge swinging onto floor does not tunnel."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="3 3 0.1"/>
          <body pos="0 0 0.3">
            <joint axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.05" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qvel = d.qvel.numpy()
      qvel[1, 0] = 2.0
      d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)

    worst = [1e9] * nworld
    for _ in range(150):
      mjw.step(m, d)
      for w in range(nworld):
        cz = float(d.geom_xpos.numpy()[w, 1, 2])
        axz = float(d.geom_xmat.numpy()[w, 1, 2, 2])
        lowest = cz - abs(0.2 * axz) - 0.05
        if lowest < worst[w]:
          worst[w] = lowest
        self.assertFalse(np.isnan(d.qvel.numpy()[w, 0]))

    for w in range(nworld):
      self.assertGreater(worst[w], -0.05)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_flag_overrides_passive_contact(self, nworld):
    """Verifies that flag ipc=enable overrides flex passive contact."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG" iterations="400">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <flexcomp name="cloth" type="grid" dim="2" count="5 5 1" spacing=".04 .04 1"
                    radius=".004" mass=".1" pos="0 0 .2">
            <contact selfcollide="auto" passive="true"/>
            <pin id="0 4 20 24"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.01
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    for _ in range(50):
      mjw.step(m, d)
      self.assertFalse(np.isnan(d.qpos.numpy()[0, 0]))

    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_articulated_kinetic_in_flex_matches_tree(self, nworld):
    """Articulated bodies in unified flex solver match the kinetics of tree alone."""
    _, _, m1, d1 = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <body pos="0 0 1">
            <joint axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    _, _, m2, d2 = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <body pos="0 0 1">
            <joint axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
          </body>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
                    radius="0.005" mass="0.05" pos="3 0 1">
            <pin id="0 2 6 8"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qvel1 = d1.qvel.numpy()
      qvel1[1, 0] = 1.5
      d1.qvel = wp.array(qvel1, dtype=float, device=d1.qvel.device)
      qvel2 = d2.qvel.numpy()
      qvel2[1, 0] = 1.5
      d2.qvel = wp.array(qvel2, dtype=float, device=d2.qvel.device)

    for _ in range(150):
      mjw.step(m1, d1)
      mjw.step(m2, d2)
      self.assertFalse(np.isnan(d2.qpos.numpy()[0, 0]))

    np.testing.assert_allclose(d1.qpos.numpy()[:, 0], d2.qpos.numpy()[:, 0], atol=1e-5)
    if nworld == 2:
      self.assertFalse(np.allclose(d1.qpos.numpy()[0], d1.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_rigid_contact_matches_euler(self, nworld):
    """Rigid contact under discrete + IPC matches Euler trajectory to tolerance."""
    _, _, m_ipc, d_ipc = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="3 3 0.1"/>
          <body pos="0 0 0.4">
            <freejoint/>
            <geom size="0.1" mass="1" condim="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    _, _, m_euler, d_euler = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="3 3 0.1"/>
          <body pos="0 0 0.4">
            <freejoint/>
            <geom size="0.1" mass="1" condim="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    m_ipc.opt.integrator = mjw.IntegratorType.DISCRETE
    m_ipc.opt.solver = mjw.SolverType.CG
    m_ipc.opt.enableflags |= mjw.EnableBit.IPC

    m_euler.opt.integrator = mjw.IntegratorType.EULER

    if nworld == 2:
      qpos = d_ipc.qpos.numpy()
      qpos[1, 2] = 0.5
      d_ipc.qpos = wp.array(qpos, dtype=float, device=d_ipc.qpos.device)
      d_euler.qpos = wp.array(qpos, dtype=float, device=d_euler.qpos.device)

    for _ in range(200):
      mjw.step(m_ipc, d_ipc)
      mjw.step(m_euler, d_euler)

    np.testing.assert_allclose(d_ipc.qpos.numpy()[:, :3], d_euler.qpos.numpy()[:, :3], atol=1e-2)
    if nworld == 2:
      self.assertFalse(np.allclose(d_ipc.qpos.numpy()[0], d_ipc.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_contact_flag_and_masks_filter_geom_pairs(self, nworld):
    """Collision filtering via contype/conaffinity and disableflags applies to IPC pairs."""
    _, _, m_c, d_c = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG" iterations="400">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.05" pos="0 0 0.05">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    _, _, m_f, d_f = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG" iterations="400">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.05" pos="0 0 0.05">
            <contact selfcollide="none" contype="0" conaffinity="0"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qpos_c = d_c.qpos.numpy()
      qpos_c[1] += 0.01
      d_c.qpos = wp.array(qpos_c, dtype=float, device=d_c.qpos.device)
      qpos_f = d_f.qpos.numpy()
      qpos_f[1] += 0.01
      d_f.qpos = wp.array(qpos_f, dtype=float, device=d_f.qpos.device)

    for _ in range(100):
      mjw.step(m_c, d_c)
      mjw.step(m_f, d_f)

    for w in range(nworld):
      # Colliding cloth rests on floor (z > 0)
      self.assertGreater(float(np.min(d_c.flexvert_xpos.numpy()[w, :, 2])), 0.0)
      # Filtered cloth tunnels through floor under gravity (z < 0)
      self.assertLess(float(np.min(d_f.flexvert_xpos.numpy()[w, :, 2])), -0.05)
    if nworld == 2:
      self.assertFalse(np.allclose(d_c.flexvert_xpos.numpy()[0], d_c.flexvert_xpos.numpy()[1]))
      self.assertFalse(np.allclose(d_f.flexvert_xpos.numpy()[0], d_f.flexvert_xpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_cylinder_flex_no_duplicate_contacts(self, nworld):
    """Verifies cylinder-flex contacts use native discrete contacts without IPC duplication."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="cyl" type="cylinder" size="0.2 0.05" pos="0 0 0.1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.05" pos="0 0 0.2">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.02
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    for _ in range(50):
      mjw.step(m, d)

    # Native contacts are generated for cylinder (unsupported in IPC)
    self.assertGreater(int(d.nacon.numpy()[0]), 0)
    # Cloth rests on top of cylinder (z > 0.1)
    for w in range(nworld):
      self.assertGreater(float(np.min(d.flexvert_xpos.numpy()[w, :, 2])), 0.1)
    if nworld == 2:
      self.assertFalse(np.allclose(d.flexvert_xpos.numpy()[0], d.flexvert_xpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_ipc_omitted_pair_admission_at_proposal(self, nworld):
    """Verifies that omitted candidate pairs outside initial bounds are admitted at proposal."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="sheet" type="grid" dim="2" count="4 4 1" spacing="0.03 0.03 1"
                    radius="0.003" mass="0.04" pos="0 0 0.15">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qvel = d.qvel.numpy()
      qvel[1, 2] = -1.0  # high downward velocity in world 1
      d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)

    for _ in range(40):
      mjw.step(m, d)

    # Sheet must not penetrate through floor
    for w in range(nworld):
      zmin = float(np.min(d.flexvert_xpos.numpy()[w, :, 2]))
      self.assertGreater(zmin, 0.0)
    if nworld == 2:
      self.assertFalse(np.allclose(d.flexvert_xpos.numpy()[0], d.flexvert_xpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_ipc_reach_filtering_simulation(self, nworld):
    """Verifies that IPC reach filtering prunes distant pairs while keeping contact integrity."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="sheet" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.05" pos="0 0 0.05">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1, 2] += 0.02
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    for _ in range(30):
      mjw.step(m, d)

    for w in range(nworld):
      zmin_w = float(np.min(d.flexvert_xpos.numpy()[w, :, 2]))
      self.assertGreater(zmin_w, 0.0)
    if nworld == 2:
      self.assertFalse(np.allclose(d.flexvert_xpos.numpy()[0], d.flexvert_xpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_ipc_flex_sphere_collision(self, nworld):
    """Verifies that a flex cloth falling onto a static sphere contacts without falling through."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.005" integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="ball" size="0.1" pos="0 0 0.1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.05" pos="0 0 0.25">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.01
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    for _ in range(25):
      mjw.step(m, d)

    for w in range(nworld):
      verts = d.flexvert_xpos.numpy()[w]
      zmin = float(np.min(verts[:, 2]))
      zmax = float(np.max(verts[:, 2]))
      self.assertGreaterEqual(zmax, 0.20)
      self.assertGreaterEqual(zmin, 0.15)
      # Also verify off-center vertices (y != 0) remain outside sphere radius (0.1)
      dists = np.linalg.norm(verts - np.array([0.0, 0.0, 0.1]), axis=1)
      self.assertGreaterEqual(float(np.min(dists)), 0.098)
    if nworld == 2:
      self.assertFalse(np.allclose(d.flexvert_xpos.numpy()[0], d.flexvert_xpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_ipc_multi_cloth_stacking(self, nworld):
    """Verifies that multiple flex cloths falling on each other stack without interpenetrating."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.005" integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="cloth1" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1"
                    radius="0.005" mass="0.05" pos="0 0 0.05">
            <contact selfcollide="none"/>
          </flexcomp>
          <flexcomp name="cloth2" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1"
                    radius="0.005" mass="0.05" pos="0 0 0.15">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.01
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    for _ in range(40):
      mjw.step(m, d)

    for w in range(nworld):
      verts = d.flexvert_xpos.numpy()[w]
      v1 = verts[:4]
      v2 = verts[4:]
      self.assertGreaterEqual(float(np.min(v1[:, 2])), 0.0)
      self.assertGreater(float(np.min(v2[:, 2] - v1[:, 2])), 0.0)
    if nworld == 2:
      self.assertFalse(np.allclose(d.flexvert_xpos.numpy()[0], d.flexvert_xpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_ipc_pair_admit_enters_working_set(self, nworld):
    """Verifies that admitted pairs are linearized and residual-active pairs enter ws.pair_in_ws."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.005" integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="ball" size="0.1" pos="0 0 0.1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.05" pos="0 0 0.205">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1, 2] += 0.001
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    for _ in range(5):
      mjw.step(m, d)

    ws = ipc.get_ipc_workspace(m, d)
    mjw.forward(m, d)
    ipc.ipc(m, d, ws=ws)
    for w in range(nworld):
      naset_w = int(ws.naset.numpy()[w])
      in_ws = ws.pair_in_ws.numpy()[w, :naset_w]
      lniv = ws.pair_lniv.numpy()[w, :naset_w]
      ws_slots = np.where(in_ws > 0)[0]
      self.assertGreater(naset_w, 0)
      self.assertGreater(len(ws_slots), 0)
      # Every working-set pair is linearized and in the active set
      self.assertTrue(np.all(lniv[ws_slots] > 0))
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @absltest.skipIf(not wp.get_device().is_cuda, "CUDA graph capture requires a CUDA device")
  @parameterized.parameters(1, 2)
  def test_ipc_graph_capture(self, nworld):
    """Verifies that IPC simulation captures into CUDA graph with graph_conditional."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="sheet" type="grid" dim="2" count="4 4 1" spacing="0.03 0.03 1"
                    radius="0.003" mass="0.04" pos="0 0 0.15">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      overrides={"opt.graph_conditional": True},
    )
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.01
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    # Warmup step before capture
    mjw.step(m, d)

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    for _ in range(10):
      wp.capture_launch(capture.graph)
    wp.synchronize()

    for w in range(nworld):
      self.assertGreater(float(d.time.numpy()[w]), 0.02)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @absltest.skipIf(not wp.get_device().is_cuda, "CUDA graph capture requires a CUDA device")
  @parameterized.parameters(1, 2)
  def test_ipc_graph_conditional_parity(self, nworld):
    """Verifies numerical parity between eager execution and captured graph execution."""
    _, _, m_eager, d_eager = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="sheet" type="grid" dim="2" count="4 4 1" spacing="0.03 0.03 1"
                    radius="0.003" mass="0.04" pos="0 0 0.15">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      overrides={"opt.graph_conditional": False},
    )
    _, _, m_graph, d_graph = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="0 0 1"/>
          <flexcomp name="sheet" type="grid" dim="2" count="4 4 1" spacing="0.03 0.03 1"
                    radius="0.003" mass="0.04" pos="0 0 0.15">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      overrides={"opt.graph_conditional": True},
    )

    if nworld == 2:
      qpos = d_eager.qpos.numpy()
      qpos[1] += 0.01
      d_eager.qpos = wp.array(qpos, dtype=float, device=d_eager.qpos.device)
      d_graph.qpos = wp.array(qpos, dtype=float, device=d_graph.qpos.device)

    # Warmup both models to compile kernels before capture
    mjw.step(m_eager, d_eager)
    mjw.step(m_graph, d_graph)

    with wp.ScopedCapture() as capture:
      mjw.step(m_graph, d_graph)

    for _ in range(10):
      mjw.step(m_eager, d_eager)
      wp.capture_launch(capture.graph)
    wp.synchronize()

    np.testing.assert_allclose(d_graph.qpos.numpy(), d_eager.qpos.numpy(), atol=1e-5)
    np.testing.assert_allclose(d_graph.qvel.numpy(), d_eager.qvel.numpy(), atol=1e-5)
    if nworld == 2:
      self.assertFalse(np.allclose(d_graph.qpos.numpy()[0], d_graph.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_static_geom_primitives_parity(self, nworld):
    """Verifies flex cloth colliding with static capsule and box geoms matches MuJoCo C."""
    for gtype, gsize in [("capsule", "0.04 0.02"), ("box", "0.1 0.1 0.04")]:
      m_c, d_c, m, d = test_data.fixture(
        xml=f"""
        <mujoco>
          <option integrator="discrete" solver="CG">
            <flag ipc="enable"/>
          </option>
          <worldbody>
            <geom name="obs" type="{gtype}" size="{gsize}" pos="0 0 0.05"/>
            <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                      radius="0.004" mass="0.05" pos="0 0 0.13">
              <edge equality="true"/>
              <contact selfcollide="none"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        nworld=nworld,
      )
      if nworld == 2:
        qpos = d.qpos.numpy()
        qpos[1] += 0.005
        d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

      d.qacc.fill_(wp.inf)
      for _ in range(25):
        mujoco.mj_step(m_c, d_c)
        mjw.step(m, d)

      self.assertFalse(np.isinf(d.qacc.numpy()).any())
      np.testing.assert_allclose(d.flexvert_xpos.numpy()[0], d_c.flexvert_xpos, atol=2e-3)
      if nworld == 2:
        self.assertFalse(np.allclose(d.flexvert_xpos.numpy()[0], d.flexvert_xpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_unhulled_mesh_uses_native_collision(self, nworld):
    """Verifies static mesh without convex hull (mesh_graphadr < 0) uses native collision."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <asset>
          <mesh name="box" vertex="
            -0.2 -0.2 -0.02  0.2 -0.2 -0.02  0.2 0.2 -0.02  -0.2 0.2 -0.02
            -0.2 -0.2 0.02  0.2 -0.2 0.02  0.2 0.2 0.02  -0.2 0.2 0.02"/>
        </asset>
        <worldbody>
          <geom name="mesh_obs" type="mesh" mesh="box"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.05" pos="0 0 0.04">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    # Mark mesh as unhulled (mesh_graphadr = -1)
    m.mesh_graphadr = wp.array([-1], dtype=int, device=m.mesh_graphadr.device)
    ws = ipc.get_ipc_workspace(m, d)
    ipc.ipc_init_topology(m, ws)
    cc.ipc_update_geom_features(m, d, ws)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, 1e30, 1e30)
    for w in range(nworld):
      self.assertEqual(int(ws.ncand.numpy()[w]), 0)

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.005
      d.qpos = wp.array(qpos, dtype=float, device=d.qpos.device)

    native_contacts = 0
    for _ in range(30):
      mjw.step(m, d)
      native_contacts += int(d.nacon.numpy()[0])

    self.assertGreater(native_contacts, 0)
    for w in range(nworld):
      self.assertGreater(float(np.min(d.flexvert_xpos.numpy()[w, :, 2])), 0.01)
    if nworld == 2:
      self.assertFalse(np.allclose(d.flexvert_xpos.numpy()[0], d.flexvert_xpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_per_world_geom_variation(self, nworld):
    """Verifies per-world static geom positions in nworld=2 produce distinct resting heights."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="box" type="box" size="0.2 0.2 0.05" pos="0 0 0.05"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.04 0.04 1"
                    radius="0.004" mass="0.05" pos="0 0 0.125">
            <edge equality="true"/>
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      # Shift box in world 1 higher by 0.02m in d.geom_xpos (static worldbody geom)
      gxpos = d.geom_xpos.numpy()
      gxpos[1, 0, 2] = 0.07
      d.geom_xpos = wp.array(gxpos, dtype=wp.vec3, device=d.geom_xpos.device)

    for _ in range(60):
      mjw.step(m, d)

    z0 = float(np.mean(d.flexvert_xpos.numpy()[0, :, 2]))
    self.assertGreater(z0, 0.095)
    if nworld == 2:
      z1 = float(np.mean(d.flexvert_xpos.numpy()[1, :, 2]))
      self.assertGreater(z1 - z0, 0.015)

  @parameterized.parameters(1, 2)
  def test_ipc_workspace_and_overflow(self, nworld):
    """Verifies IPC workspace container sizing and overflow flag reporting."""
    mjm, mjd, m, _ = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG" jacobian="sparse">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="1 1 0.1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1"
                    spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.0005">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    d_sized = mjw.make_data(mjm, nworld=nworld, nconmax=16, nccdmax=8)
    ws_sized = ipc.get_ipc_workspace(m, d_sized)
    self.assertEqual(ws_sized.max_aset, 16)
    self.assertEqual(ws_sized.max_cand, 8)

    # Test BROADPHASE overflow when nconmax=2 (< 4 vertex-plane candidates in world 0)
    d_bp = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=2)
    if nworld == 2:
      qpos = d_bp.qpos.numpy()
      qpos[1, 2::3] += 1.0
      d_bp.qpos = wp.array(qpos, dtype=float, device=d_bp.qpos.device)
    mjw.step(m, d_bp)
    self.assertTrue(bool(int(d_bp.overflow.numpy()[0]) & int(OverflowType.BROADPHASE)))
    if nworld == 2:
      self.assertEqual(int(d_bp.overflow.numpy()[1]), 0)

    # Test NARROWPHASE overflow when active set capacity max_aset is exceeded
    d_np = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=4)
    if nworld == 2:
      fv = d_np.flexvert_xpos.numpy()
      fv[1, :, 2] += 1.0
      d_np.flexvert_xpos = wp.array(fv, dtype=wp.vec3, device=d_np.flexvert_xpos.device)
    d_np.naconmax = 2 * nworld
    ws_np = ipc.get_ipc_workspace(m, d_np)
    self.assertEqual(ws_np.max_aset, 2)
    self.assertEqual(ws_np.max_cand, 4)
    d_np.overflow.zero_()
    cc.ipc_update_geom_features(m, d_np, ws_np)
    cc.ipc_discover_candidates(m, d_np, ws_np, d_np.flexvert_xpos, d_np.flexvert_xpos, d_np.flexvert_xpos, 1e30, 1e30)
    ipc.ipc_seed_active_set(m, d_np, ws_np)
    self.assertTrue(bool(int(d_np.overflow.numpy()[0]) & int(OverflowType.NARROWPHASE)))
    if nworld == 2:
      self.assertEqual(int(d_np.overflow.numpy()[1]), 0)

    # Test NEFC overflow when njmax=1 (< 4 active contact rows in world 0)
    d_nefc = mjw.put_data(mjm, mjd, nworld=nworld, njmax=1)
    if nworld == 2:
      qpos = d_nefc.qpos.numpy()
      qpos[1, 2::3] += 1.0
      d_nefc.qpos = wp.array(qpos, dtype=float, device=d_nefc.qpos.device)
    mjw.step(m, d_nefc)
    self.assertTrue(bool(int(d_nefc.overflow.numpy()[0]) & int(OverflowType.NEFC)))
    if nworld == 2:
      self.assertEqual(int(d_nefc.overflow.numpy()[1]), 0)

    # Test NJMAX_NNZ overflow when njmax_nnz=2 (< 3 nonzeros per vertex-plane row in world 0)
    d_nnz = mjw.put_data(mjm, mjd, nworld=nworld, njmax=16, njmax_nnz=2)
    if nworld == 2:
      qpos = d_nnz.qpos.numpy()
      qpos[1, 2::3] += 1.0
      d_nnz.qpos = wp.array(qpos, dtype=float, device=d_nnz.qpos.device)
    mjw.step(m, d_nnz)
    self.assertTrue(bool(int(d_nnz.overflow.numpy()[0]) & int(OverflowType.NJMAX_NNZ)))
    if nworld == 2:
      self.assertEqual(int(d_nnz.overflow.numpy()[1]), 0)


if __name__ == "__main__":
  absltest.main()
