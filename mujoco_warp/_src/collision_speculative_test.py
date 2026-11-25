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
"""Tests for speculative continuous collision detection."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import BroadphaseType
from mujoco_warp import ConeType
from mujoco_warp import DeterminismType
from mujoco_warp import DisableBit
from mujoco_warp import IntegratorType
from mujoco_warp import SolverType
from mujoco_warp import test_data
from mujoco_warp._src import island
from mujoco_warp._src import sleep
from mujoco_warp._src.types import ConstraintState
from mujoco_warp._src.types import ContactType
from mujoco_warp._src.types import OverflowType


def _fill_contact_outputs(d: mjw.Data):
  """Poison contact output buffers before running collision."""
  d.nacon.zero_()
  d.contact.dist.fill_(wp.inf)
  d.contact.pos.fill_(wp.vec3(wp.inf))
  d.contact.frame.fill_(wp.mat33(wp.inf))
  d.contact.includemargin.fill_(wp.inf)
  d.contact.solref.fill_(wp.vec2(wp.inf))
  d.contact.dim.fill_(-1)
  d.contact.geom.fill_(wp.vec2i(-1, -1))
  d.contact.worldid.fill_(-1)
  d.contact.type.fill_(-1)
  d.contact.geomcollisionid.fill_(-1)
  d.contact.adhesion.fill_(wp.inf)


def _extract_world_contacts(d: mjw.Data, world: int = 0) -> list[dict[str, object]]:
  """Return active constraint contacts for a world sorted canonically."""
  nacon = min(int(d.nacon.numpy()[0]), d.naconmax)
  worldids = d.contact.worldid.numpy()[:nacon]
  ctypes = d.contact.type.numpy()[:nacon]
  geoms = d.contact.geom.numpy()[:nacon]
  dists = d.contact.dist.numpy()[:nacon]
  pos = d.contact.pos.numpy()[:nacon]
  frames = d.contact.frame.numpy()[:nacon]
  dims = d.contact.dim.numpy()[:nacon]
  includemargins = d.contact.includemargin.numpy()[:nacon]
  solrefs = d.contact.solref.numpy()[:nacon]
  adhesion = d.contact.adhesion.numpy()[:nacon]
  surfacevel = d.contact.surfacevel.numpy()[:nacon]

  out = []
  for i in range(nacon):
    if worldids[i] != world:
      continue
    if not (int(ctypes[i]) & int(ContactType.CONSTRAINT)):
      continue
    g0, g1 = int(geoms[i, 0]), int(geoms[i, 1])
    out.append(
      {
        "geom": (g0, g1),
        "dist": float(dists[i]),
        "pos": pos[i].copy(),
        "normal": frames[i, 0].copy(),
        "dim": int(dims[i]),
        "includemargin": float(includemargins[i]),
        "solref": solrefs[i].copy(),
        "type": int(ctypes[i]),
        "adhesion": float(adhesion[i]),
        "surfacevel": surfacevel[i].copy(),
      }
    )
  out.sort(key=lambda c: (c["geom"][0], c["geom"][1], round(c["pos"][0], 3), round(c["pos"][1], 3), round(c["pos"][2], 3)))
  return out


def _extract_mj_contacts(mjd: mujoco.MjData) -> list[dict[str, object]]:
  """Return MuJoCo C contacts sorted canonically."""
  out = []
  for i in range(mjd.ncon):
    con = mjd.contact[i]
    g0, g1 = int(con.geom[0]), int(con.geom[1])
    normal = np.array(con.frame[:3], dtype=np.float64)
    out.append(
      {
        "geom": (g0, g1),
        "dist": float(con.dist),
        "pos": np.array(con.pos, dtype=np.float64),
        "normal": normal,
        "dim": int(con.dim),
        "includemargin": float(con.includemargin),
        "solref": np.array(con.solref, dtype=np.float64),
        "exclude": int(con.exclude),
        "surfacevel": np.array(con.surfacevel, dtype=np.float64),
      }
    )
  out.sort(key=lambda c: (c["geom"][0], c["geom"][1], round(c["pos"][0], 3), round(c["pos"][1], 3), round(c["pos"][2], 3)))
  return out


def _assert_contact_parity(
  testcase: absltest.TestCase,
  mjd: mujoco.MjData,
  d: mjw.Data,
  world: int = 0,
  atol_dist: float = 2e-3,
  atol_pos: float = 2e-3,
  atol_normal: float = 5e-3,
):
  """Compare active constraint contacts for a world against MuJoCo C."""
  ref = _extract_mj_contacts(mjd)
  got = _extract_world_contacts(d, world=world)
  testcase.assertEqual(len(got), len(ref), f"contact count mismatch: got={got}, ref={ref}")

  # Match contacts per geom pair using greedy nearest position to handle orientation ordering
  used = [False] * len(got)
  for r in ref:
    best_idx = -1
    best_err = np.inf
    for j, g in enumerate(got):
      if used[j]:
        continue
      if g["geom"] != r["geom"] and g["geom"] != (r["geom"][1], r["geom"][0]):
        continue
      err = float(np.linalg.norm(g["pos"] - r["pos"]))
      if err < best_err:
        best_err = err
        best_idx = j
    testcase.assertGreaterEqual(best_idx, 0, f"missing contact for geom pair {r['geom']}: got={got}, ref={ref}")
    used[best_idx] = True
    g = got[best_idx]
    sign = 1.0 if g["geom"] == r["geom"] else -1.0
    np.testing.assert_allclose(g["dist"], r["dist"], atol=atol_dist, rtol=1e-2)
    np.testing.assert_allclose(g["pos"], r["pos"], atol=atol_pos, rtol=1e-2)
    np.testing.assert_allclose(g["normal"], sign * r["normal"], atol=atol_normal, rtol=1e-2)
    np.testing.assert_allclose(g["solref"], r["solref"], atol=1e-4, rtol=1e-4)
    testcase.assertEqual(g["dim"], r["dim"])


def _assert_pair_contacts(
  testcase: absltest.TestCase,
  got: list[dict[str, object]],
  ref: list[dict[str, object]],
  pair: tuple[int, int],
  check_xy: bool = True,
  cylinder_center_xy: tuple[float, float] | None = None,
  cylinder_radius: float | None = None,
):
  """Compare contacts for a specific unordered geom pair against MuJoCo C."""
  g_pair = [c for c in got if tuple(sorted(c["geom"])) == pair]
  r_pair = [c for c in ref if tuple(sorted(c["geom"])) == pair]
  testcase.assertEqual(len(g_pair), len(r_pair), f"pair={pair}: got={g_pair}, ref={r_pair}")
  for gc, rc in zip(g_pair, r_pair):
    sign = 1.0 if gc["geom"] == rc["geom"] else -1.0
    np.testing.assert_allclose(gc["dist"], rc["dist"], atol=2e-3, rtol=1e-2)
    np.testing.assert_allclose(gc["normal"], sign * rc["normal"], atol=5e-3, rtol=1e-2)
    if check_xy:
      np.testing.assert_allclose(gc["pos"], rc["pos"], atol=2e-3, rtol=1e-2)
    else:
      testcase.assertAlmostEqual(float(gc["pos"][2]), float(rc["pos"][2]), delta=2e-3)
      if cylinder_center_xy is not None and cylinder_radius is not None:
        cx, cy = cylinder_center_xy
        testcase.assertAlmostEqual(float(np.hypot(gc["pos"][0] - cx, gc["pos"][1] - cy)), cylinder_radius, delta=2e-3)


def _assert_body_moment_parity(testcase: absltest.TestCase, mjm: mujoco.MjModel, mjd: mujoco.MjData, d: mjw.Data):
  """Compare sparse actuator_moment row 0 against MuJoCo C."""
  c_moment = np.zeros((mjm.nu, mjm.nv))
  mujoco.mju_sparse2dense(c_moment, mjd.actuator_moment, mjd.moment_rownnz, mjd.moment_rowadr, mjd.moment_colind)
  w_moment = np.zeros((mjm.nu, mjm.nv))
  nnz = int(d.moment_rownnz.numpy()[0, 0])
  adr = int(d.moment_rowadr.numpy()[0, 0])
  colind = d.moment_colind.numpy()[0, adr : adr + nnz]
  w_moment[0, colind] = d.actuator_moment.numpy()[0, adr : adr + nnz]
  np.testing.assert_allclose(w_moment, c_moment, atol=1e-4, rtol=1e-4)


def _sync_state(mjm: mujoco.MjModel, mjd: mujoco.MjData, m: mjw.Model, d: mjw.Data):
  """Copy MjData qpos/qvel into world 0 of d and run forward kinematics and collision on both."""
  mujoco.mj_fwdPosition(mjm, mjd)
  d.qpos.assign(np.broadcast_to(mjd.qpos, (d.nworld, mjm.nq)).astype(np.float32))
  d.qvel.assign(np.broadcast_to(mjd.qvel, (d.nworld, mjm.nv)).astype(np.float32))
  _fill_contact_outputs(d)
  mjw.kinematics(m, d)
  mjw.com_pos(m, d)
  mjw.collision(m, d)


def _sync_forward(mjm: mujoco.MjModel, mjd: mujoco.MjData, m: mjw.Model, d: mjw.Data):
  """Copy MjData state into d, poison outputs, and run mj_forward / mjw.forward."""
  d.qpos.assign(np.broadcast_to(mjd.qpos, (d.nworld, mjm.nq)).astype(np.float32))
  d.qvel.assign(np.broadcast_to(mjd.qvel, (d.nworld, mjm.nv)).astype(np.float32))
  if mjm.nu > 0:
    d.ctrl.assign(np.broadcast_to(mjd.ctrl, (d.nworld, mjm.nu)).astype(np.float32))
  d.qacc.fill_(wp.inf)
  _fill_contact_outputs(d)
  mujoco.mj_forward(mjm, mjd)
  mjw.forward(m, d)


def _step_captured(mjm: mujoco.MjModel, mjd: mujoco.MjData, m: mjw.Model, d: mjw.Data, nsteps: int):
  """Capture mjw.step in a CUDA graph and advance both MuJoCo C and Warp for nsteps."""
  with wp.ScopedCapture() as capture:
    mjw.step(m, d)
  for _ in range(nsteps):
    mujoco.mj_step(mjm, mjd)
    wp.capture_launch(capture.graph)


def _card_pad_xml(offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
  """Return thin tilted card vs box pad XML shifted by offset."""
  ox, oy, oz = offset
  return f"""
  <mujoco>
    <option timestep="0.008" gravity="0 0 0">
      <flag speculative="enable"/>
    </option>
    <worldbody>
      <geom name="pad" type="box" pos="{ox} {oy} {oz}" size="0.026 0.014 0.0032"/>
      <body name="card" pos="{ox} {oy} {oz + 0.00385}" euler="0.2 0.1 0">
        <freejoint/>
        <geom name="card_geom" type="box" size="0.032 0.047 0.00035" mass="0.002"/>
      </body>
    </worldbody>
  </mujoco>
  """


def _wall_gap_sphere_xml() -> str:
  """Return wall + sphere XML with gap > 0 for in-gap replacement tests."""
  return """
  <mujoco>
    <option timestep="0.01" gravity="0 0 0">
      <flag speculative="enable"/>
    </option>
    <worldbody>
      <geom name="wall" type="box" size="0.5 0.5 0.02"/>
      <body pos="0 0 0.15">
        <freejoint/>
        <geom size="0.05" gap="0.20"/>
      </body>
    </worldbody>
  </mujoco>
  """


class CollisionSpeculativeTest(parameterized.TestCase):
  """Tests for speculative continuous collision detection."""

  def test_toi_pullback_and_grazing(self):
    """Test TOI, moving away, grazing near-miss, plane hit, oblique pullback, and positive dist."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <default>
          <geom solref="0.1 1"/>
        </default>
        <worldbody>
          <geom name="ground" type="plane" size="2 2 0.01" pos="0 0 -1"/>
          <geom name="wall" type="box" size="0.5 0.5 0.02"/>
          <body name="ball" pos="0 0 0.27">
            <freejoint/>
            <geom name="sphere" size="0.05"/>
          </body>
          <body name="box1" pos="2 0 0.5">
            <freejoint/>
            <geom name="b1" type="box" size="0.05 0.05 0.05" mass="100"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )

    # Straight hit onto thin box wall: TOI ~ 0.4, dist < 0, speculative flag set
    mujoco.mj_resetData(mjm, mjd)
    mjd.qvel[2] = -50.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    got = _extract_world_contacts(d)
    self.assertEqual(len(got), 1)
    self.assertTrue(bool(got[0]["type"] & int(ContactType.SPECULATIVE)))
    self.assertAlmostEqual(0.8 + got[0]["dist"] / 50.0, 0.392, delta=0.02)
    self.assertAlmostEqual(got[0]["pos"][2], 0.22, delta=1e-4)
    sign = 1.0 if got[0]["geom"] == (1, 2) else -1.0
    self.assertAlmostEqual(sign * got[0]["normal"][2], 1.0, delta=1e-4)

    # Moving upward away from wall: no contact
    mujoco.mj_resetData(mjm, mjd)
    mjd.qvel[2] = 50.0
    _sync_state(mjm, mjd, m, d)
    self.assertEqual(mjd.ncon, 0)
    self.assertEqual(len(_extract_world_contacts(d)), 0)

    # Grazing near-miss skimming 1 cm above box top: no contact
    mujoco.mj_resetData(mjm, mjd)
    mjd.qpos[0] = -0.8
    mjd.qpos[2] = 0.08
    mjd.qvel[0] = 50.0
    _sync_state(mjm, mjd, m, d)
    self.assertEqual(mjd.ncon, 0)
    self.assertEqual(len(_extract_world_contacts(d)), 0)

    # Plane vs moving sphere
    mujoco.mj_resetData(mjm, mjd)
    mjd.qpos[2] = -0.75
    mjd.qvel[2] = -50.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    self.assertEqual(len(_extract_world_contacts(d)), 1)

    # Oblique 45 deg sphere hitting free box -> pulled-back contact beneath sphere t=0 center
    mujoco.mj_resetData(mjm, mjd)
    mjd.qpos[0] = 1.85
    mjd.qpos[2] = 0.75
    mjd.qvel[0] = 30.0
    mjd.qvel[2] = -30.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    got = _extract_world_contacts(d)
    self.assertEqual(len(got), 1)
    self.assertAlmostEqual(got[0]["pos"][0], 1.85, delta=1e-4)
    self.assertAlmostEqual(got[0]["pos"][2], 0.70, delta=1e-4)

    # Line-of-sight occlusion: wall at z=0 occludes ground at z=-1
    mujoco.mj_resetData(mjm, mjd)
    mjd.qvel[2] = -200.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    got = _extract_world_contacts(d)
    self.assertEqual(len(got), 1)
    self.assertEqual(set(got[0]["geom"]), {1, 2})

    # Oblique ~76 deg impact
    mujoco.mj_resetData(mjm, mjd)
    mjd.qpos[0] = -0.10
    mjd.qpos[2] = 0.09
    mjd.qvel[0] = 20.0
    mjd.qvel[2] = -5.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    self.assertEqual(len(_extract_world_contacts(d)), 1)

    # Positive speculative dist (t > 1 - b0 with default solref) still activates as a constraint
    mjm_pos, mjd_pos, m_pos, d_pos = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall" type="box" size="0.5 0.5 0.02"/>
          <body pos="0 0 0.27">
            <freejoint/>
            <geom name="sphere" size="0.05"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjd_pos.qvel[2] = -50.0
    _sync_forward(mjm_pos, mjd_pos, m_pos, d_pos)
    _assert_contact_parity(self, mjd_pos, d_pos, atol_dist=5e-3)
    got_pos = _extract_world_contacts(d_pos)
    self.assertEqual(len(got_pos), 1)
    self.assertGreater(got_pos[0]["dist"], 0.5)
    self.assertEqual(int(d_pos.nefc.numpy()[0]), mjd_pos.nefc)
    self.assertGreater(int(d_pos.nefc.numpy()[0]), 0)
    np.testing.assert_allclose(d_pos.qacc.numpy()[0], mjd_pos.qacc, atol=1e-2, rtol=1e-2)

  def test_moving_pair_and_rotational_sweep(self):
    """Test two moving boxes, capsule on box/plane, pure rotation swing, and CA iteration cap."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <default>
          <geom solref="0.1 1"/>
        </default>
        <worldbody>
          <geom name="ground" type="plane" size="2 2 0.01" pos="0 0 -1"/>
          <geom name="wall" type="box" size="0.5 0.5 0.02"/>
          <body name="box1" pos="2 0 0.5">
            <freejoint/>
            <geom name="b1" type="box" size="0.05 0.05 0.05" mass="100"/>
          </body>
          <body name="box2" pos="2 0 0.8">
            <freejoint/>
            <geom name="b2" type="box" size="0.05 0.05 0.05"/>
          </body>
          <body name="rod" pos="-2 0 -0.90">
            <freejoint/>
            <geom name="cap" type="capsule" size="0.02 0.2" quat="0.70710678 0 0.70710678 0"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )

    # Two moving boxes traveling same direction at different speeds -> 4 face contacts
    mujoco.mj_resetData(mjm, mjd)
    mjd.qvel[2] = -10.0
    mjd.qvel[8] = -60.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    got = _extract_world_contacts(d)
    self.assertEqual(len(got), 4)
    for c in got:
      self.assertAlmostEqual(c["pos"][2], (0.1 * 0.55 + 0.6 * 0.75) / 0.7, delta=1e-4)

    # Horizontal capsule falling onto box wall -> 2 edge-face contacts
    mujoco.mj_resetData(mjm, mjd)
    mjd.qpos[14] = 0.0
    mjd.qpos[15] = 0.2
    mjd.qpos[16] = 0.24
    mjd.qvel[14] = -50.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    self.assertEqual(len(_extract_world_contacts(d)), 2)

    # Horizontal capsule falling onto ground plane -> 2 contacts
    mujoco.mj_resetData(mjm, mjd)
    mjd.qvel[14] = -50.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    self.assertEqual(len(_extract_world_contacts(d)), 2)

    # Pure rotational capsule swing (dw != 0, dx = 0) into ground plane
    mujoco.mj_resetData(mjm, mjd)
    mjd.qvel[16] = 100.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    self.assertEqual(len(_extract_world_contacts(d)), 1)

    # Rapid spin hitting 16-step cap and linear extrapolation fallback
    mujoco.mj_resetData(mjm, mjd)
    mjd.qpos[0] = -0.10
    mjd.qpos[2] = 0.09
    mjd.qvel[0] = 20.0
    mjd.qvel[2] = -5.0
    mjd.qvel[5] = 500.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d, atol_dist=5e-3, atol_pos=5e-3)
    self.assertGreaterEqual(len(_extract_world_contacts(d)), 1)

  @parameterized.product(
    integrator=(
      IntegratorType.EULER,
      IntegratorType.IMPLICIT,
      IntegratorType.IMPLICITFAST,
      IntegratorType.DISCRETE,
      IntegratorType.RK4,
    ),
    disable_refsafe=(False, True),
  )
  def test_contact_dist_integrators_and_solref(self, integrator: IntegratorType, disable_refsafe: bool):
    """Test speculative contact distance across integrators, refsafe, and all 4 solref regimes."""
    overrides: dict[str, object] = {"opt.integrator": integrator}
    if disable_refsafe:
      overrides["opt.disableflags"] = DisableBit.REFSAFE

    # Four decoupled wall-sphere pairs cover standard, direct, k0==0, and mixed_invalid solref
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall_std" type="box" pos="-1.5 0 0" size="0.2 0.2 0.02" solref="0.005 1"/>
          <geom name="wall_dir" type="box" pos="-0.5 0 0" size="0.2 0.2 0.02" solref="-2000 -50"/>
          <geom name="wall_k0" type="box" pos="0.5 0 0" size="0.2 0.2 0.02" solref="0 -499"/>
          <geom name="wall_mix" type="box" pos="1.5 0 0" size="0.2 0.2 0.02" solref="0.02 -1"/>
          <body pos="-1.5 0 0.27">
            <freejoint/>
            <geom size="0.05" solref="0.005 1" solimp="0.995 0.995 0.001"/>
          </body>
          <body pos="-0.5 0 0.27">
            <freejoint/>
            <geom size="0.05" solref="-2000 -50" solimp="0.995 0.995 0.001"/>
          </body>
          <body pos="0.5 0 0.27">
            <freejoint/>
            <geom size="0.05" solref="0 -499" solimp="0.995 0.995 0.001"/>
          </body>
          <body pos="1.5 0 0.27">
            <freejoint/>
            <geom size="0.05" solref="0.02 -1" solimp="0.995 0.995 0.001"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides=overrides,
    )

    for b in range(4):
      mjd.qvel[6 * b + 2] = -50.0
    _sync_forward(mjm, mjd, m, d)

    _assert_contact_parity(self, mjd, d, atol_dist=5e-3)
    self.assertEqual(len(_extract_world_contacts(d)), 4)
    self.assertEqual(int(d.nefc.numpy()[0]), mjd.nefc)
    self.assertGreater(int(d.nefc.numpy()[0]), 0)
    # First 3 decoupled free bodies (standard, direct, k0==0) match MuJoCo C qacc and 1-step state;
    # standard pair has k0 = 4.0 > imp = 0.995, exercising discrete REFSAFE clamping.
    qacc_atol = max(2e-2, 2e-4 * float(np.max(np.abs(mjd.qacc[:18]))))
    np.testing.assert_allclose(d.qacc.numpy()[0, :18], mjd.qacc[:18], atol=qacc_atol, rtol=2e-2)

    mujoco.mj_resetData(mjm, mjd)
    for b in range(3):
      mjd.qvel[6 * b + 2] = -50.0
    d.qpos.assign(np.array([mjd.qpos], dtype=np.float32))
    d.qvel.assign(np.array([mjd.qvel], dtype=np.float32))
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    np.testing.assert_allclose(d.qpos.numpy()[0, :21], mjd.qpos[:21], atol=2e-3, rtol=2e-2)
    np.testing.assert_allclose(d.qvel.numpy()[0, :18], mjd.qvel[:18], atol=5e-2, rtol=2e-2)

  def test_priority_with_direct_solref(self):
    """Test geom priority precedence over direct (negative) solref on speculative contacts."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall_hi" type="box" pos="-1.0 0 0" size="0.2 0.2 0.02" priority="1" solref="0.05 1"/>
          <geom name="wall_lo" type="box" pos="0.0 0 0" size="0.2 0.2 0.02" solref="-2000 -50"/>
          <geom name="wall_eq" type="box" pos="1.0 0 0" size="0.2 0.2 0.02" solref="0.05 1"/>
          <body pos="-1.0 0 0.27">
            <freejoint/>
            <geom name="ball_lo" size="0.05" solref="-2000 -50"/>
          </body>
          <body pos="0.0 0 0.27">
            <freejoint/>
            <geom name="ball_hi" size="0.05" priority="1" solref="0.05 1"/>
          </body>
          <body pos="1.0 0 0.27">
            <freejoint/>
            <geom name="ball_eq" size="0.05" solref="-2000 -50"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )

    for b in range(3):
      mjd.qvel[6 * b + 2] = -50.0
    _sync_forward(mjm, mjd, m, d)

    _assert_contact_parity(self, mjd, d, atol_dist=5e-3)
    got = _extract_world_contacts(d)
    self.assertEqual(len(got), 3)
    by_pair = {tuple(sorted(c["geom"])): c for c in got}
    # Priority 1 wins in both directions despite the other geom having direct solref <= 0
    np.testing.assert_allclose(by_pair[(0, 3)]["solref"], [0.05, 1.0], atol=1e-5)
    np.testing.assert_allclose(by_pair[(1, 4)]["solref"], [0.05, 1.0], atol=1e-5)
    # Equal priority falls back to elementwise min
    np.testing.assert_allclose(by_pair[(2, 5)]["solref"], [-2000.0, -50.0], atol=1e-5)
    qacc_atol = max(2e-2, 2e-4 * float(np.max(np.abs(mjd.qacc))))
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, atol=qacc_atol, rtol=2e-2)

  @parameterized.product(
    integrator=(IntegratorType.EULER, IntegratorType.DISCRETE),
    speculative=(False, True),
  )
  def test_high_speed_convex_geoms_no_tunnel(self, integrator: IntegratorType, speculative: bool):
    """Test high-speed convex geoms tunnel without speculative and arrest like C when enabled."""
    overrides: dict[str, object] = {"opt.integrator": integrator}
    if not speculative:
      overrides["opt.enableflags"] = "~SPECULATIVE"

    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <asset>
          <mesh name="dummy" vertex="0 0 0  0.01 0 0  0 0.01 0  0 0 0.01"/>
          <mesh name="octa" vertex="0.04 0 0  -0.04 0 0  0 0.04 0  0 -0.04 0  0 0 0.04  0 0 -0.04"/>
        </asset>
        <worldbody>
          <geom name="ground" type="plane" size="4 4 0.01" pos="0 0 -2"/>
          <geom name="thin_wall" type="box" size="3.0 1.0 0.02"/>
          <body name="fast_sphere" pos="-1.5 0 0.25">
            <freejoint/>
            <geom size="0.04" gap="0.05" mass="1"/>
          </body>
          <body name="fast_box" pos="-0.9 0 0.25">
            <freejoint/>
            <geom type="box" size="0.04 0.04 0.04" gap="0.20" mass="1"/>
          </body>
          <body name="fast_capsule" pos="-0.3 0 0.25">
            <freejoint/>
            <geom type="capsule" size="0.03 0.02" mass="1"/>
          </body>
          <body name="fast_cylinder" pos="0.3 0 0.25">
            <freejoint/>
            <geom type="cylinder" size="0.04 0.04" mass="1"/>
          </body>
          <body name="fast_ellipsoid" pos="0.9 0 0.25">
            <freejoint/>
            <geom type="ellipsoid" size="0.04 0.03 0.04" mass="1"/>
          </body>
          <body name="fast_mesh" pos="1.5 0 0.25">
            <freejoint/>
            <geom type="mesh" mesh="octa" mass="1"/>
          </body>
          <body name="plane_ball" pos="0 2.0 -1.75">
            <freejoint/>
            <geom size="0.04" mass="1"/>
          </body>
          <body name="mover_a" pos="2.4 0 0.25">
            <freejoint/>
            <geom size="0.04" mass="1"/>
          </body>
          <body name="mover_b" pos="2.4 0.0 0.55">
            <freejoint/>
            <geom size="0.04" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides=overrides,
    )

    if not speculative:
      # Verify tunneling when speculative contacts are disabled
      for b in range(6):
        mjd.qvel[6 * b + 2] = -50.0
      d.qvel.assign(np.array([mjd.qvel], dtype=np.float32))
      mjw.step(m, d)
      qpos_off = d.qpos.numpy()[0]
      for b in range(6):
        self.assertLess(qpos_off[7 * b + 2], -0.20)
      return

    # Verify arrest and parity with MuJoCo C over 5 steps when speculative is enabled
    self.assertGreater(int(mjm.mesh_vertadr[1]), 0)
    self.assertGreater(int(mjm.mesh_faceadr[1]), 0)
    for b in range(7):
      mjd.qvel[6 * b + 2] = -50.0
    # Moving-vs-moving pair approaching each other
    mjd.qvel[6 * 7 + 2] = 20.0
    mjd.qvel[6 * 8 + 2] = -40.0
    d.qvel.assign(np.array([mjd.qvel], dtype=np.float32))
    _step_captured(mjm, mjd, m, d, 5)

    qpos = d.qpos.numpy()[0]
    qvel = d.qvel.numpy()[0]
    for b in range(6):
      self.assertGreater(qpos[7 * b + 2], 0.02)
    self.assertGreater(qpos[7 * 6 + 2], -2.0)
    self.assertGreater(qpos[7 * 8 + 2], qpos[7 * 7 + 2])
    # All 9 bodies match C on vertical position and velocity; cylinder 16-gon tie-breaking and
    # smooth ellipsoid GJK float32 tolerance can tilt slightly at 50 m/s, shifting z/vz slightly
    for b in range(9):
      z_tol = 2e-3 if b == 3 else 1e-3
      vz_tol = 1e-1 if b in (3, 4) else 2e-2
      self.assertAlmostEqual(float(qpos[7 * b + 2]), float(mjd.qpos[7 * b + 2]), delta=z_tol)
      self.assertAlmostEqual(float(qvel[6 * b + 2]), float(mjd.qvel[6 * b + 2]), delta=vz_tol)
    # Non-curved/non-16-gon bodies match full 6D state tightly
    for b in (0, 1, 2, 5, 6, 7, 8):
      np.testing.assert_allclose(qpos[7 * b : 7 * b + 7], mjd.qpos[7 * b : 7 * b + 7], atol=5e-3, rtol=2e-2)
      np.testing.assert_allclose(qvel[6 * b : 6 * b + 6], mjd.qvel[6 * b : 6 * b + 6], atol=5e-2, rtol=2e-2)

  @parameterized.parameters(False, True)
  def test_card_pad_multicontact(self, disable_multiccd: bool):
    """Test thin tilted card vs box pad multicontact gating via DisableBit.MULTICCD."""
    overrides = {"opt.disableflags": DisableBit.MULTICCD} if disable_multiccd else {}
    mjm, mjd, m, d = test_data.fixture(xml=_card_pad_xml(), overrides=overrides)
    mjd.qvel[2] = -2.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    self.assertEqual(len(_extract_world_contacts(d)), 1 if disable_multiccd else 4)

  @parameterized.parameters(False, True)
  def test_plane_primitive_multicontact(self, disable_multiccd: bool):
    """Test plane vs box/cylinder/capsule multicontact stays active even with MULTICCD disabled."""
    overrides = {"opt.disableflags": DisableBit.MULTICCD} if disable_multiccd else {}
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="ground" type="plane" size="4 4 0.01"/>
          <body pos="-1 0 0.25">
            <freejoint/>
            <geom type="box" size="0.05 0.05 0.05"/>
          </body>
          <body pos="0 0 0.25">
            <freejoint/>
            <geom type="cylinder" size="0.05 0.05"/>
          </body>
          <body pos="1 0 0.25">
            <freejoint/>
            <geom type="capsule" size="0.02 0.1" euler="0 90 0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides=overrides,
    )
    for b in range(3):
      mjd.qvel[6 * b + 2] = -50.0
    _sync_state(mjm, mjd, m, d)
    got = _extract_world_contacts(d)
    ref = _extract_mj_contacts(mjd)
    # 4 box + 4 cylinder + 2 capsule = 10 plane contacts regardless of DisableBit.MULTICCD
    self.assertEqual(len(got), 10)
    self.assertEqual(len(ref), 10)
    _assert_pair_contacts(self, got, ref, (0, 1))
    _assert_pair_contacts(self, got, ref, (0, 3))
    # Cylinder cap 16-gon quad is rotationally symmetric around z; verify count, dist, normal, z, r
    _assert_pair_contacts(self, got, ref, (0, 2), check_xy=False, cylinder_center_xy=(0.0, 0.0), cylinder_radius=0.05)

  @parameterized.parameters(False, True)
  def test_mesh_and_cylinder_multicontact(self, disable_multiccd: bool):
    """Test mesh/cylinder multicontact with non-zero mesh_*adr offsets and last-mesh index."""
    overrides = {"opt.disableflags": DisableBit.MULTICCD} if disable_multiccd else {}
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <asset>
          <mesh name="dummy" vertex="0 0 0  0.01 0 0  0 0.01 0  0 0 0.01"/>
          <mesh name="cube" vertex="-0.05 -0.05 -0.05  0.05 -0.05 -0.05  0.05 0.05 -0.05  -0.05 0.05 -0.05
                                    -0.05 -0.05  0.05  0.05 -0.05  0.05  0.05 0.05  0.05  -0.05 0.05  0.05"/>
        </asset>
        <worldbody>
          <geom name="ground" type="plane" size="4 4 0.01"/>
          <geom name="pad1" type="box" size="0.2 0.2 0.02" pos="1 0 0"/>
          <geom name="pad2" type="box" size="0.2 0.2 0.02" pos="2 0 0"/>
          <body pos="0 0 0.25">
            <freejoint/>
            <geom type="mesh" mesh="cube"/>
          </body>
          <body pos="1 0 0.25">
            <freejoint/>
            <geom type="mesh" mesh="cube"/>
          </body>
          <body pos="2 0 0.25">
            <freejoint/>
            <geom type="cylinder" size="0.05 0.05"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides=overrides,
    )
    # Verify cube sits at last mesh index with non-zero vertex/face/polygon offsets
    self.assertEqual(mjm.nmesh, 2)
    self.assertGreater(int(mjm.mesh_vertadr[1]), 0)
    self.assertGreater(int(mjm.mesh_faceadr[1]), 0)
    self.assertGreater(int(mjm.mesh_polyadr[1]), 0)

    for b in range(3):
      mjd.qvel[6 * b + 2] = -50.0
    _sync_state(mjm, mjd, m, d)
    got = _extract_world_contacts(d)
    ref = _extract_mj_contacts(mjd)
    expected = 4 + 1 + 1 if disable_multiccd else 4 + 4 + 4
    self.assertEqual(len(got), expected)
    self.assertEqual(len(ref), expected)
    _assert_pair_contacts(self, got, ref, (0, 3))
    _assert_pair_contacts(self, got, ref, (1, 4), check_xy=not disable_multiccd)
    if disable_multiccd:
      # Without multicontact, parallel mesh-box faces may return any point on the contact face
      g_mb = [c for c in got if tuple(sorted(c["geom"])) == (1, 4)][0]
      self.assertGreaterEqual(float(g_mb["pos"][0]), 0.95 - 2e-3)
      self.assertLessEqual(float(g_mb["pos"][0]), 1.05 + 2e-3)
      self.assertGreaterEqual(float(g_mb["pos"][1]), -0.05 - 2e-3)
      self.assertLessEqual(float(g_mb["pos"][1]), 0.05 + 2e-3)
    _assert_pair_contacts(
      self,
      got,
      ref,
      (2, 5),
      check_xy=False,
      cylinder_center_xy=(2.0, 0.0),
      cylinder_radius=None if disable_multiccd else 0.05,
    )

  def test_witness_transport_and_rotation(self):
    """Test spinning geoms preserve TOI and multicontact via witness transport."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <asset>
          <mesh name="dummy" vertex="0 0 0  0.01 0 0  0 0.01 0  0 0 0.01"/>
          <mesh name="octa" vertex="0.06 0 0  -0.06 0 0  0 0.06 0  0 -0.06 0  0 0 0.06  0 0 -0.06
                                    0.04 0.04 0.02  -0.04 0.04 0.02  0.04 -0.04 0.02  -0.04 -0.04 0.02"/>
        </asset>
        <worldbody>
          <geom name="floor_box" type="box" size="1.0 1.0 0.02"/>
          <body name="moving_box" pos="0 0 0.12">
            <freejoint/>
            <geom name="box_geom" type="box" size="0.05 0.05 0.02" mass="0.5"/>
          </body>
          <body name="moving_cap" pos="0.5 0 0.12">
            <freejoint/>
            <geom name="cap_geom" type="capsule" size="0.02 0.04" euler="0 90 0" mass="0.5"/>
          </body>
          <geom name="target_ell" type="ellipsoid" size="0.08 0.05 0.03" pos="-3.0 0 0"/>
          <body name="moving_ell" pos="-3.0 0 0.10" euler="15 -10 25">
            <freejoint/>
            <geom name="ell_geom" type="ellipsoid" size="0.03 0.06 0.02" mass="0.5"/>
          </body>
          <geom name="target_mesh" type="mesh" mesh="octa" pos="3.0 0 0"/>
          <body name="moving_mesh" pos="3.0 0 0.18" euler="10 20 -15">
            <freejoint/>
            <geom name="mesh_geom" type="mesh" mesh="octa" mass="0.5"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjd.qvel[2] = -25.0
    mjd.qvel[5] = 8.0
    mjd.qvel[8] = -25.0
    mjd.qvel[11] = 10.0
    mjd.qvel[14] = -25.0
    mjd.qvel[15] = 6.0
    mjd.qvel[20] = -25.0
    mjd.qvel[22] = 8.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d, atol_dist=5e-3, atol_pos=5e-3)

    got = _extract_world_contacts(d)
    by_pair: dict[tuple[int, int], int] = {}
    for c in got:
      key = tuple(sorted(c["geom"]))
      by_pair[key] = by_pair.get(key, 0) + 1
    floor_id = mjm.geom("floor_box").id
    box_id = mjm.geom("box_geom").id
    cap_id = mjm.geom("cap_geom").id
    target_ell_id = mjm.geom("target_ell").id
    ell_id = mjm.geom("ell_geom").id
    target_mesh_id = mjm.geom("target_mesh").id
    mesh_id = mjm.geom("mesh_geom").id
    self.assertEqual(by_pair[tuple(sorted((floor_id, box_id)))], 4)
    self.assertEqual(by_pair[tuple(sorted((floor_id, cap_id)))], 2)
    self.assertEqual(by_pair[tuple(sorted((target_ell_id, ell_id)))], 1)
    self.assertGreaterEqual(by_pair[tuple(sorted((target_mesh_id, mesh_id)))], 1)

  def test_stationary_thin_target_and_card_vs_mesh(self):
    """Test thick moving sphere catching thin stationary card and thin card vs mesh wedge."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <asset>
          <mesh name="wedge" vertex="-0.04 -0.04 0  0.04 -0.04 0  0 0.04 0
                                     -0.04 -0.04 0.02  0.04 -0.04 0.02  0 0.04 0.01"/>
        </asset>
        <worldbody>
          <geom name="stationary_card" type="box" size="0.045 0.032 0.00035"/>
          <body name="moving_finger" pos="0 0 0.012">
            <freejoint/>
            <geom name="finger_pad" size="0.010" mass="0.2"/>
          </body>
          <geom name="wedge_geom" type="mesh" mesh="wedge" pos="0.3 0 0.02"/>
          <body name="fast_card" pos="0.3 0 0.045">
            <freejoint/>
            <geom name="fast_card_geom" type="box" size="0.03 0.02 0.00035" mass="0.005"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjd.qvel[2] = -0.8
    mjd.qvel[8] = -4.0
    d.qvel.assign(np.array([mjd.qvel], dtype=np.float32))
    _fill_contact_outputs(d)

    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)

    _assert_contact_parity(self, mjd, d, atol_dist=5e-3, atol_pos=5e-3)
    self.assertGreaterEqual(len(_extract_world_contacts(d)), 2)
    self.assertGreater(d.qpos.numpy()[0, 2], 0.008)
    self.assertGreater(d.qpos.numpy()[0, 9], 0.010)
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=5e-3, rtol=5e-3)

  def test_ray_occlusion_barriers_and_partial_multicontact(self):
    """Test ray occlusion by box, sphere, plane, offset mesh barriers, and partial multicontact."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <asset>
          <mesh name="dummy" vertex="0 0 0  0.01 0 0  0 0.01 0  0 0 0.01"/>
          <mesh name="slab" vertex="-0.5 -0.5 -0.02  0.5 -0.5 -0.02  0.5 0.5 -0.02  -0.5 0.5 -0.02
                                    -0.5 -0.5  0.02  0.5 -0.5  0.02  0.5 0.5  0.02  -0.5 0.5  0.02"/>
        </asset>
        <worldbody>
          <geom name="far_wall" type="box" size="4 1 0.02" pos="0 0 -0.2"/>
          <geom name="box_barrier" type="box" size="0.3 0.3 0.02" pos="-1.5 0 0"/>
          <geom name="sphere_barrier" size="0.08" pos="-0.5 0 0"/>
          <geom name="mesh_barrier" type="mesh" mesh="slab" pos="0.5 0 0"/>
          <geom name="partial_barrier" type="box" size="0.2 0.2 0.02" pos="1.75 0 0"/>
          <body pos="-1.5 0 0.20">
            <freejoint/>
            <geom size="0.03"/>
          </body>
          <body pos="-0.5 0 0.20">
            <freejoint/>
            <geom size="0.03"/>
          </body>
          <body pos="0.5 0 0.20">
            <freejoint/>
            <geom size="0.03"/>
          </body>
          <body pos="1.5 0 0.20">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.02"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    # Verify mesh_barrier hits last mesh with non-zero face/vertex offsets in ray_mesh
    self.assertGreater(int(mjm.mesh_vertadr[1]), 0)
    self.assertGreater(int(mjm.mesh_faceadr[1]), 0)

    for b in range(4):
      mjd.qvel[6 * b + 2] = -60.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)

    # None of the 3 occluded spheres contact far_wall (geom 0); partial box gets 2 corners on it
    got = _extract_world_contacts(d)
    far_pairs = [c["geom"] for c in got if 0 in c["geom"]]
    self.assertEqual(far_pairs, [(0, 8), (0, 8)])

    # Plane barrier occluding far wall behind plane
    mjm_pl, mjd_pl, m_pl, d_pl = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="plane_barrier" type="plane" size="2 2 0.01"/>
          <geom name="far_wall" type="box" size="1 1 0.02" pos="0 0 -0.2"/>
          <body pos="0 0 0.20">
            <freejoint/>
            <geom size="0.03"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjd_pl.qvel[2] = -60.0
    _sync_state(mjm_pl, mjd_pl, m_pl, d_pl)
    _assert_contact_parity(self, mjd_pl, d_pl)
    got_pl = _extract_world_contacts(d_pl)
    self.assertEqual(len(got_pl), 1)
    self.assertEqual(got_pl[0]["geom"], (0, 2))

  def test_ray_occlusion_exclude_and_explicit_pair(self):
    """Test ray occlusion respects <exclude> and explicit <pair> overrides."""
    # Excluded shield does not occlude far wall
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall" type="box" size="1 1 0.02"/>
          <body name="shield" pos="0 0 0.15">
            <geom name="shield_geom" type="box" size="0.5 0.5 0.02"/>
          </body>
          <body name="ball" pos="0 0 0.30">
            <freejoint/>
            <geom name="ball_geom" size="0.03" mass="1"/>
          </body>
        </worldbody>
        <contact>
          <exclude body1="ball" body2="shield"/>
        </contact>
      </mujoco>
      """
    )
    mjd.qvel[2] = -50.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    self.assertEqual(len(_extract_world_contacts(d)), 1)

    # Explicit pair: wall1 occludes wall2 even with contype=0/conaffinity=0
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <body name="wall1_body" mocap="true">
            <geom name="wall1" type="box" size="1 1 0.02" contype="0" conaffinity="0"/>
          </body>
          <body name="wall2_body" mocap="true" pos="0 0 -0.15">
            <geom name="wall2" type="box" size="1 1 0.02" contype="0" conaffinity="0"/>
          </body>
          <body name="ball" pos="0 0 0.25">
            <freejoint/>
            <geom name="ball_geom" size="0.03" contype="0" conaffinity="0" mass="1"/>
          </body>
        </worldbody>
        <contact>
          <pair geom1="ball_geom" geom2="wall1"/>
          <pair geom1="ball_geom" geom2="wall2"/>
        </contact>
      </mujoco>
      """
    )
    mjd.qvel[2] = -60.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    got = _extract_world_contacts(d)
    self.assertEqual(len(got), 1)
    self.assertEqual(got[0]["geom"], (2, 0))

  def test_ray_occlusion_weld_orthogonal_and_floor(self):
    """Test ray occlusion ignores same moving weld, orthogonal overhang, and supporting floor."""
    # Same moving weld: rotor hub does not self-occlude paddle hitting target fin
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="target_fin" type="box" pos="0.1864 0.0725 0" euler="0 0 -68.75" size="0.005 0.0015 0.02"/>
          <body name="rotor">
            <joint name="spin"/>
            <geom name="paddle" type="capsule" fromto="0 -0.20 0 0 0.20 0" size="0.003" mass="0.05"/>
            <geom name="hub" type="cylinder" size="0.175 0.02" mass="0.1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjd.qvel[0] = -150.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    got = _extract_world_contacts(d)
    self.assertTrue(any(c["geom"] == (1, 0) for c in got))

    # Orthogonal swept catch: overhang blocking static [x1_0, x2_0] does not occlude swept paths
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="overhang" type="box" size="0.05 0.004 0.004" pos="0 -0.01 0.01"/>
          <body name="finger" pos="0 -0.02 0">
            <joint name="slide_y" type="slide" axis="0 1 0"/>
            <geom name="finger_geom" size="0.004" mass="0.2"/>
          </body>
          <body name="card" pos="0 0 0.02">
            <joint name="slide_z" type="slide"/>
            <geom name="card_geom" type="box" size="0.03 0.02 0.00035" mass="0.01"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjd.qvel[0] = 4.0
    mjd.qvel[1] = -4.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    got = _extract_world_contacts(d)
    self.assertTrue(any(c["geom"] == (1, 2) for c in got))

    # Supporting floor: plate settled in floor margin sliding into vertical wall gets 4 contacts
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="2 2 0.01" margin="0.001" condim="1"/>
          <geom name="wall" type="box" size="0.02 0.5 0.2" pos="0.155 0 0"/>
          <body name="plate" pos="-0.06 0 0.0035">
            <freejoint/>
            <geom name="plate_geom" type="box" size="0.05 0.05 0.00035" margin="0.001" condim="1" mass="0.1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjd.qvel[0] = 5.0
    mujoco.mj_step(mjm, mjd)
    mujoco.mj_step(mjm, mjd)
    self.assertLess(mjd.qvel[2], 0.0)
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    got = _extract_world_contacts(d)
    wall_contacts = [c for c in got if c["geom"] == (1, 2)]
    self.assertEqual(len(wall_contacts), 4)

  def test_canonical_newton_scene(self):
    """Test Newton's cradle speculative scene (solref='0 -499', k0 == 0) parity with MuJoCo C."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.004">
          <flag speculative="enable"/>
        </option>
        <default>
          <joint axis="0 1 0" frictionloss="0.08"/>
          <default class="plate">
            <geom type="box" pos="0 0 -1.0" size="0.0015 0.08 0.08" mass="0.5" condim="1" solref="0 -499" solimp="0.995 0.995 0.001"/>
          </default>
        </default>
        <worldbody>
          <body name="pendulum1" pos="-0.04 0 1.40" euler="0 58 0">
            <joint name="j1"/>
            <geom name="p1" class="plate"/>
          </body>
          <body name="pendulum2" pos="-0.02 0 1.40">
            <joint name="j2"/>
            <geom name="p2" class="plate"/>
          </body>
          <body name="pendulum3" pos="0 0 1.40">
            <joint name="j3"/>
            <geom name="p3" class="plate"/>
          </body>
          <body name="pendulum4" pos="0.02 0 1.40">
            <joint name="j4"/>
            <geom name="p4" class="plate"/>
          </body>
          <body name="pendulum5" pos="0.04 0 1.40">
            <joint name="j5"/>
            <geom name="p5" class="plate"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    # Start pendulum 1 right before impact with bottom speed ~3 m/s
    mjd.qpos[0] = -np.deg2rad(56.5)
    mjd.qvel[0] = -3.0
    d.qpos.assign(np.array([mjd.qpos], dtype=np.float32))
    d.qvel.assign(np.array([mjd.qvel], dtype=np.float32))
    _step_captured(mjm, mjd, m, d, 15)

    # Momentum transfers from plate 1 through plates 2-4 to launch plate 5
    self.assertLess(d.qvel.numpy()[0, 4], -2.5)
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=5e-3, rtol=5e-3)
    np.testing.assert_allclose(d.qvel.numpy()[0], mjd.qvel, atol=5e-2, rtol=5e-2)

  def test_canonical_chain_scene(self):
    """Test interlocking capsule chain speculative scene parity with MuJoCo C."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.006">
          <flag speculative="enable"/>
        </option>
        <default>
          <geom type="capsule" size="0.006" friction="0.3"/>
        </default>
        <worldbody>
          <geom name="ground" type="plane" size="0 0 0.01"/>
          <geom name="catch_bar" fromto="0 -0.25 1.88 0 0.25 1.88" size="0.006"/>
          <body name="link1" pos="0 0 1.80">
            <freejoint/>
            <geom fromto="-0.06 0  0.16   0.06 0  0.16"/>
            <geom fromto="-0.06 0 -0.16   0.06 0 -0.16"/>
            <geom fromto="-0.06 0 -0.16  -0.06 0  0.16"/>
            <geom fromto=" 0.06 0 -0.16   0.06 0  0.16"/>
          </body>
          <body name="link2" pos="0 0 1.62">
            <freejoint/>
            <geom fromto="0 -0.06  0.16  0  0.06  0.16"/>
            <geom fromto="0 -0.06 -0.16  0  0.06 -0.16"/>
            <geom fromto="0 -0.06 -0.16  0 -0.06  0.16"/>
            <geom fromto="0  0.06 -0.16  0  0.06  0.16"/>
          </body>
          <body name="link3" pos="0 0 1.44">
            <freejoint/>
            <geom fromto="-0.06 0  0.16   0.06 0  0.16"/>
            <geom fromto="-0.06 0 -0.16   0.06 0 -0.16"/>
            <geom fromto="-0.06 0 -0.16  -0.06 0  0.16"/>
            <geom fromto=" 0.06 0 -0.16   0.06 0  0.16"/>
          </body>
          <body name="link4" pos="0.015 0 1.26">
            <freejoint/>
            <geom fromto="0 -0.06  0.16  0  0.06  0.16"/>
            <geom fromto="0 -0.06 -0.16  0  0.06 -0.16"/>
            <geom fromto="0 -0.06 -0.16  0 -0.06  0.16"/>
            <geom fromto="0  0.06 -0.16  0  0.06  0.16"/>
            <geom type="sphere" pos="0 0 -0.20" size="0.02"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    # Give links initial downward velocity so interlocking catches occur within 20 steps
    for b in range(4):
      mjd.qvel[6 * b + 2] = -2.0
    d.qvel.assign(np.array([mjd.qvel], dtype=np.float32))
    _step_captured(mjm, mjd, m, d, 20)

    # Link 1 stays caught on the catch bar rather than tunneling through
    self.assertGreater(d.qpos.numpy()[0, 2], 1.65)
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=2e-2, rtol=2e-2)

  def test_canonical_flipper_scene(self):
    """Test rotating bat/flipper hitting pitched ball parity with MuJoCo C."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="ground" type="plane" size="0 0 0.01"/>
          <body name="pitcher_plunger" pos="0.83 0.05 0.42">
            <joint name="plunger_slide" type="slide" axis="-1 0 0" springref="0.22" stiffness="2200" damping="30" range="0 0.18"/>
            <geom name="plunger_head" type="box" size="0.01 0.04 0.04" mass="2.0"/>
          </body>
          <body name="ball" pos="0.795 0.05 0.42">
            <freejoint/>
            <geom name="baseball" size="0.02" mass="0.05" solref="0.004 0.5"/>
          </body>
          <body name="bat_body" pos="0 -0.30 0.40" euler="0 0 65">
            <joint name="bat_swing" axis="0 0 -1" springref="80" stiffness="21" damping="0.3"/>
            <geom name="bat" type="box" pos="0 0.30 0" size="0.006 0.30 0.07" mass="0.4" solref="0.004 0.5"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )

    _step_captured(mjm, mjd, m, d, 15)

    # Bat strikes pitched ball and reverses its x-velocity from negative to positive
    self.assertGreater(d.qvel.numpy()[0, 1], 1.0)
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(d.qvel.numpy()[0], mjd.qvel, atol=5e-2, rtol=2e-2)

  def test_discrete_active_and_gap_replacement(self):
    """Test active discrete contact suppresses speculative and gap contact replaces only on hit."""
    # Active discrete contact (dist < margin) suppresses speculative contact
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall" type="box" size="0.5 0.5 0.02" margin="0.05"/>
          <body pos="0 0 0.10">
            <freejoint/>
            <geom size="0.05" margin="0.05"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    # dist = 0.10 - 0.02 - 0.05 = 0.03 < margin = 0.10 -> active discrete contact only
    mjd.qvel[2] = -50.0
    _sync_state(mjm, mjd, m, d)
    _assert_contact_parity(self, mjd, d)
    got = _extract_world_contacts(d)
    self.assertEqual(len(got), 1)
    self.assertFalse(bool(got[0]["type"] & int(ContactType.SPECULATIVE)))

    # Gap > 0: speculative replaces in-gap contact on hit and preserves it on miss
    mjm_g, mjd_g, m_g, d_g = test_data.fixture(xml=_wall_gap_sphere_xml())
    # Hit case: speculative contact replaces in-gap discrete contact
    mjd_g.qvel[2] = -50.0
    _sync_state(mjm_g, mjd_g, m_g, d_g)
    _assert_contact_parity(self, mjd_g, d_g)
    got_hit = _extract_world_contacts(d_g)
    self.assertEqual(len(got_hit), 1)
    self.assertTrue(bool(got_hit[0]["type"] & int(ContactType.SPECULATIVE)))

    # Miss case (moving away): in-gap discrete contact preserved (not speculative, exclude == 1)
    mujoco.mj_resetData(mjm_g, mjd_g)
    mjd_g.qvel[2] = 50.0
    _sync_state(mjm_g, mjd_g, m_g, d_g)
    _assert_contact_parity(self, mjd_g, d_g)
    got_miss = _extract_world_contacts(d_g)
    self.assertEqual(len(got_miss), 1)
    self.assertFalse(bool(got_miss[0]["type"] & int(ContactType.SPECULATIVE)))
    out_miss = mujoco.MjData(mjm_g)
    mjw.get_data_into(out_miss, mjm_g, d_g)
    self.assertEqual(out_miss.ncon, 1)
    self.assertEqual(out_miss.contact[0].exclude, 1)

  def test_naconmax_overflow_preserves_in_gap_contact(self):
    """Test naconmax overflow during speculative emission does not clear existing in-gap contact."""
    mjm, mjd, m, d = test_data.fixture(
      xml=_wall_gap_sphere_xml(),
      nconmax=1,
      overrides={"opt.warn_overflow": 0},
    )
    self.assertEqual(d.naconmax, 1)
    # Discrete narrowphase fills slot 0 with the in-gap contact; speculative overflows naconmax=1
    mjd.qvel[2] = -50.0
    _sync_state(mjm, mjd, m, d)
    got = _extract_world_contacts(d)
    self.assertEqual(len(got), 1)
    self.assertTrue(bool(got[0]["type"] & int(ContactType.CONSTRAINT)))
    self.assertFalse(bool(got[0]["type"] & int(ContactType.SPECULATIVE)))

  def test_adhesion_and_nativeccd(self):
    """Test adhesion condim, replaced-gap and sensor-only qfrc_passive, and NATIVECCD disable."""
    mjm_a, mjd_a, m_a, d_a = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall" type="box" size="1 1 0.02"/>
          <body pos="-0.4 0 0.15">
            <freejoint/>
            <geom name="adh_gap_ball" size="0.05" condim="4" gap="0.20" adhesion="20"/>
          </body>
          <body pos="0.4 0 0.15">
            <freejoint/>
            <geom name="sensor_only_ball" size="0.05" contype="0" conaffinity="0" adhesion="20"/>
          </body>
        </worldbody>
        <sensor>
          <distance geom1="wall" geom2="sensor_only_ball" cutoff="1.0"/>
        </sensor>
      </mujoco>
      """
    )
    mjd_a.qvel[2] = -50.0
    d_a.qfrc_passive.fill_(wp.inf)
    _sync_forward(mjm_a, mjd_a, m_a, d_a)

    _assert_contact_parity(self, mjd_a, d_a, atol_dist=5e-3)
    got_a = _extract_world_contacts(d_a)
    self.assertEqual(len(got_a), 1)
    self.assertEqual(got_a[0]["dim"], 4)
    self.assertEqual(got_a[0]["adhesion"], 20.0)
    # Cleared in-gap contact does not double adhesion; sensor-only contact gets zero adhesion
    np.testing.assert_allclose(d_a.qfrc_passive.numpy()[0], mjd_a.qfrc_passive, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(d_a.qfrc_passive.numpy()[0, 6:], 0.0, atol=1e-6)
    qacc_atol = max(2e-2, 2e-4 * float(np.max(np.abs(mjd_a.qacc))))
    np.testing.assert_allclose(d_a.qacc.numpy()[0], mjd_a.qacc, atol=qacc_atol, rtol=2e-2)

    # DisableBit.NATIVECCD disables speculative contacts completely
    mjm_n, mjd_n, m_n, d_n = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable" nativeccd="disable"/>
        </option>
        <worldbody>
          <geom name="wall" type="box" size="0.5 0.5 0.02"/>
          <body pos="0 0 0.25">
            <freejoint/>
            <geom size="0.05"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjd_n.qvel[2] = -50.0
    _sync_state(mjm_n, mjd_n, m_n, d_n)
    self.assertEqual(mjd_n.ncon, 0)
    self.assertEqual(len(_extract_world_contacts(d_n)), 0)

  @parameterized.product(
    nworld=(1, 2),
    broadphase=(BroadphaseType.NXN, BroadphaseType.SAP_TILE, BroadphaseType.SAP_SEGMENTED),
  )
  def test_multiworld_and_broadphase(self, nworld: int, broadphase: BroadphaseType):
    """Test multi-world distinct states and all broadphase modes including offset-AABB meshes."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <asset>
          <mesh name="dummy" vertex="0 0 0  0.01 0 0  0 0.01 0  0 0 0.01"/>
          <mesh name="offset_cube" vertex="0.10 -0.04 -0.04  0.18 -0.04 -0.04  0.18 0.04 -0.04  0.10 0.04 -0.04
                                           0.10 -0.04  0.04  0.18 -0.04  0.04  0.18 0.04  0.04  0.10 0.04  0.04"/>
        </asset>
        <worldbody>
          <geom name="wall" type="box" size="1 1 0.02"/>
          <body pos="0 0 0.25">
            <freejoint/>
            <geom size="0.04"/>
          </body>
          <body pos="0.3 0 0.25">
            <freejoint/>
            <geom type="mesh" mesh="offset_cube"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      overrides={"opt.broadphase": broadphase},
    )

    qvel = np.zeros((nworld, mjm.nv), dtype=np.float32)
    qvel[0, 2] = -50.0
    qvel[0, 8] = -50.0
    if nworld == 2:
      # World 1 moves upward -> no speculative contacts
      qvel[1, 2] = 50.0
      qvel[1, 8] = 50.0

    d.qvel.assign(qvel)
    d.qacc.fill_(wp.inf)
    _fill_contact_outputs(d)
    mjw.forward(m, d)

    # Compare world 0 against MuJoCo C
    mjd.qvel[:] = qvel[0]
    mujoco.mj_forward(mjm, mjd)
    _assert_contact_parity(self, mjd, d, world=0)
    self.assertGreaterEqual(len(_extract_world_contacts(d, world=0)), 2)

    if nworld == 2:
      self.assertEqual(len(_extract_world_contacts(d, world=1)), 0)
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  def test_batched_model_fields(self):
    """Test per-world batched opt.timestep, geom_margin, and geom_solref in speculative contacts."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall" type="box" size="0.5 0.5 0.02" solref="0.1 1"/>
          <body pos="0 0 0.27">
            <freejoint/>
            <geom name="ball" size="0.05" solref="0.1 1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      batch_sizes={"geom_margin": 2, "geom_solref": 2},
    )

    # World 0: dt=0.01, margin=0.0, solref=[0.1, 1.0]
    # World 1: dt=0.005, margin=0.01, solref=[0.02, 1.0]
    m.opt.timestep = wp.array([0.01, 0.005], dtype=float)
    margin_arr = m.geom_margin.numpy()
    margin_arr[0, :] = 0.0
    margin_arr[1, :] = 0.01
    m.geom_margin.assign(margin_arr)
    solref_arr = m.geom_solref.numpy()
    solref_arr[0, :, :] = [0.1, 1.0]
    solref_arr[1, :, :] = [0.02, 1.0]
    m.geom_solref.assign(solref_arr)

    qvel = np.zeros((2, mjm.nv), dtype=np.float32)
    qvel[0, 2] = -50.0
    qvel[1, 2] = -50.0
    d.qvel.assign(qvel)
    d.qacc.fill_(wp.inf)
    _fill_contact_outputs(d)
    mjw.forward(m, d)

    for w, (dt_w, margin_w, solref_w) in enumerate(((0.01, 0.0, [0.1, 1.0]), (0.005, 0.01, [0.02, 1.0]))):
      mjm.opt.timestep = dt_w
      mjm.geom_margin[:] = margin_w
      mjm.geom_solref[:, :] = solref_w
      mujoco.mj_resetData(mjm, mjd)
      mjd.qvel[2] = -50.0
      mujoco.mj_forward(mjm, mjd)
      _assert_contact_parity(self, mjd, d, world=w, atol_dist=5e-3)
      np.testing.assert_allclose(d.qacc.numpy()[w], mjd.qacc, atol=2e-2, rtol=2e-2)

    # Distinct per-world batched fields produce distinct contact distances and accelerations
    c0 = _extract_world_contacts(d, world=0)[0]
    c1 = _extract_world_contacts(d, world=1)[0]
    self.assertNotAlmostEqual(c0["dist"], c1["dist"], places=2)
    self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.parameters(
    (1, BroadphaseType.NXN, True),
    (1, BroadphaseType.NXN, False),
    (2, BroadphaseType.NXN, True),
    (2, BroadphaseType.SAP_TILE, True),
    (2, BroadphaseType.SAP_SEGMENTED, True),
  )
  def test_sleep_graph_conditional(self, nworld: int, broadphase: BroadphaseType, wake: bool):
    """Test speculative contacts wake or keep asleep sleeping trees inside graph conditional."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable" sleep="enable"/>
        </option>
        <worldbody>
          <body name="sleeper">
            <freejoint/>
            <geom name="sleeper_box" type="box" size="0.1 0.1 0.02" mass="1"/>
          </body>
          <body name="projectile" pos="0 0 0.25">
            <freejoint/>
            <geom name="proj_sphere" size="0.04" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      overrides={"opt.broadphase": broadphase},
    )
    self.assertTrue(m.opt.graph_conditional)

    qvel = np.zeros((nworld, mjm.nv), dtype=np.float32)
    qvel[0, 8] = -50.0 if wake else 50.0
    if nworld == 2:
      qvel[1, 8] = 50.0
    d.qvel.assign(qvel)

    # Put sleeper (tree 0) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 0] = 0
    d.tree_asleep.assign(tree_asleep)
    sleep.update_sleep(m, d)
    for w in range(nworld):
      self.assertEqual(int(d.tree_awake.numpy()[w, 0]), 0)

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    wp.capture_launch(capture.graph)

    mjd.qvel[:] = qvel[0]
    mjd.tree_asleep[0] = 0
    mujoco.mj_step(mjm, mjd)
    self.assertEqual(int(d.tree_awake.numpy()[0, 0]), int(wake))
    if not wake:
      self.assertEqual(len(_extract_world_contacts(d)), 0)
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(d.qvel.numpy()[0], mjd.qvel, atol=5e-2, rtol=2e-2)

    if nworld == 2:
      # World 1 misses sleeper so tree 0 remains asleep
      self.assertEqual(int(d.tree_awake.numpy()[1, 0]), 0)
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_explicit_pair_with_collision_sensor(self, nworld: int):
    """Test explicit <pair> with collision sensors yields speculative contact and t=0 sensors."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall" type="box" size="0.5 0.5 0.02"/>
          <body pos="0 0 0.27">
            <freejoint/>
            <geom name="ball" size="0.05"/>
          </body>
        </worldbody>
        <contact>
          <pair geom1="wall" geom2="ball"/>
        </contact>
        <sensor>
          <distance geom1="wall" geom2="ball" cutoff="1.0"/>
          <normal geom1="wall" geom2="ball" cutoff="1.0"/>
          <fromto geom1="wall" geom2="ball" cutoff="1.0"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
    )
    qvel = np.zeros((nworld, mjm.nv), dtype=np.float32)
    qvel[0, 2] = -50.0
    if nworld == 2:
      qvel[1, 2] = 50.0
    d.qvel.assign(qvel)
    d.sensordata.fill_(wp.inf)
    _fill_contact_outputs(d)

    mjw.forward(m, d)

    # World 0: speculative constraint contact AND exact t=0 distance/normal/fromto sensor readings
    mjd.qvel[:] = qvel[0]
    mujoco.mj_forward(mjm, mjd)
    _assert_contact_parity(self, mjd, d, world=0)
    self.assertEqual(len(_extract_world_contacts(d, world=0)), 1)
    np.testing.assert_allclose(d.sensordata.numpy()[0], mjd.sensordata, atol=1e-4, rtol=1e-4)

    if nworld == 2:
      self.assertEqual(len(_extract_world_contacts(d, world=1)), 0)
      # Sensor still measures the exact t=0 distance (0.20 m) even without speculative hit
      self.assertAlmostEqual(float(d.sensordata.numpy()[1, 0]), 0.20, places=4)

  def test_contact_sensor_speculative_and_replaced_gap(self):
    """Test <contact> sensor reports speculative hits and ignores cleared replaced-gap contacts."""
    # Pure speculative contact and replaced in-gap contact in the same scene
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall" type="box" size="2 1 0.02"/>
          <body pos="-0.5 0 0.27">
            <freejoint/>
            <geom name="ball_spec" size="0.05"/>
          </body>
          <body pos="0.5 0 0.15">
            <freejoint/>
            <geom name="ball_gap" size="0.05" gap="0.20"/>
          </body>
        </worldbody>
        <sensor>
          <contact geom1="wall" geom2="ball_spec" reduce="mindist" num="2" data="found dist pos normal"/>
          <contact geom1="wall" geom2="ball_gap" reduce="mindist" num="2" data="found dist pos normal"/>
        </sensor>
      </mujoco>
      """
    )
    mjd.qvel[2] = -50.0
    mjd.qvel[8] = -50.0
    d.sensordata.fill_(wp.inf)
    _sync_forward(mjm, mjd, m, d)

    _assert_contact_parity(self, mjd, d, atol_dist=5e-3)
    got_sensor = d.sensordata.numpy()[0]
    # Both sensors find exactly 1 contact (no double counting of the cleared in-gap contact)
    self.assertEqual(got_sensor[0], 1.0)
    self.assertEqual(got_sensor[16], 1.0)
    np.testing.assert_allclose(got_sensor, mjd.sensordata, atol=5e-3, rtol=1e-2)

  def test_surfacevel_contactfilter_and_determinism(self):
    """Test contact sorting, surfacevel, contactfilter callback, and DeterminismType.ALL."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="belt" type="box" size="1 1 0.02" surfacevel="0.5 -0.2 0 0 0 0"/>
          <body pos="-0.2 0 0.25">
            <freejoint/>
            <geom size="0.04"/>
          </body>
          <body pos="0.2 0 0.25">
            <freejoint/>
            <geom type="box" size="0.04 0.04 0.04"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.deterministic": DeterminismType.CONTACTS},
    )
    callback_nacon = []

    def _contactfilter(model: mjw.Model, data: mjw.Data):
      del model
      callback_nacon.append(int(data.nacon.numpy()[0]))

    m.callback.contactfilter = _contactfilter
    mjd.qvel[2] = -50.0
    mjd.qvel[8] = -50.0
    d.contact.surfacevel.fill_(wp.vec3(wp.inf))
    _sync_forward(mjm, mjd, m, d)

    _assert_contact_parity(self, mjd, d)
    self.assertEqual(len(callback_nacon), 1)
    self.assertGreaterEqual(callback_nacon[0], 5)
    for c in _extract_world_contacts(d):
      self.assertFalse(np.allclose(c["surfacevel"], 0.0))

    # DeterminismType.ALL: repeated forward runs from identical state produce bit-identical outputs
    mjm_det, _, m_det, d_det = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall" type="box" size="2 1 0.02"/>
          <body pos="-0.3 0 0.25">
            <freejoint/>
            <geom type="box" size="0.04 0.04 0.04"/>
          </body>
          <body pos="0.3 0 0.25">
            <freejoint/>
            <geom size="0.04"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.deterministic": DeterminismType.ALL},
    )
    qvel_det = np.zeros((1, mjm_det.nv), dtype=np.float32)
    qvel_det[0, 2] = -50.0
    qvel_det[0, 8] = -50.0
    d_det.qvel.assign(qvel_det)
    d_det.qacc.fill_(wp.inf)
    _fill_contact_outputs(d_det)
    mjw.forward(m_det, d_det)
    ncon1 = int(d_det.nacon.numpy()[0])
    dist1 = d_det.contact.dist.numpy()[:ncon1].copy()
    pos1 = d_det.contact.pos.numpy()[:ncon1].copy()
    qacc1 = d_det.qacc.numpy().copy()

    d_det.qvel.assign(qvel_det)
    d_det.qacc.fill_(wp.inf)
    _fill_contact_outputs(d_det)
    mjw.forward(m_det, d_det)
    ncon2 = int(d_det.nacon.numpy()[0])
    self.assertEqual(ncon1, ncon2)
    np.testing.assert_array_equal(d_det.contact.dist.numpy()[:ncon2], dist1)
    np.testing.assert_array_equal(d_det.contact.pos.numpy()[:ncon2], pos1)
    np.testing.assert_array_equal(d_det.qacc.numpy(), qacc1)

  def test_hfield_coexistence(self):
    """Test discrete hfield contacts coexist cleanly alongside speculative convex contacts."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <asset>
          <hfield name="hf" nrow="2" ncol="2" size="1 1 0.1 0.1"/>
        </asset>
        <worldbody>
          <geom type="hfield" hfield="hf"/>
          <geom name="wall" type="box" size="0.5 0.5 0.02" pos="3 0 0"/>
          <body pos="0 0 0.05">
            <freejoint/>
            <geom size="0.08"/>
          </body>
          <body pos="3 0 0.25">
            <freejoint/>
            <geom size="0.04"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjd.qvel[8] = -50.0
    _sync_state(mjm, mjd, m, d)
    got = _extract_world_contacts(d)
    self.assertTrue(any(set(c["geom"]) == {0, 2} for c in got))
    self.assertTrue(any(set(c["geom"]) == {1, 3} and bool(c["type"] & int(ContactType.SPECULATIVE)) for c in got))

  def test_ccd_overflow_clamping(self):
    """Test OverflowType.CCD clamping when naccdmax is smaller than multicontact pairs."""
    mjm, mjd, m, _ = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="wall" type="box" size="2 1 0.02"/>
          <body pos="-0.5 0 0.25">
            <freejoint/>
            <geom type="box" size="0.04 0.04 0.04"/>
          </body>
          <body pos="0.5 0 0.25">
            <freejoint/>
            <geom type="box" size="0.04 0.04 0.04"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.warn_overflow": 0},
    )
    mjd.qvel[2] = -50.0
    mjd.qvel[8] = -50.0
    d = mjw.put_data(mjm, mjd, naccdmax=1)
    d.overflow.zero_()
    _fill_contact_outputs(d)
    mjw.collision(m, d)
    self.assertTrue(bool(int(d.overflow.numpy()[0]) & int(OverflowType.CCD)))
    # At least one pair succeeds cleanly without out-of-bounds crash
    self.assertGreaterEqual(len(_extract_world_contacts(d)), 1)

  @parameterized.parameters("dense", "sparse")
  def test_elliptic_newton_positive_speculative_dist(self, jacobian: str):
    """Test elliptic-cone Newton Hessian assembly when a speculative contact has dist > margin."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0" cone="elliptic" tolerance="1e-10">
          <flag speculative="enable" warmstart="disable" island="disable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="2 2 0.01"/>
          <body pos="0 0 0.27">
            <freejoint/>
            <geom name="ball" size="0.05" friction="0.2 0.005 0.0001"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.jacobian": jacobian},
    )
    # t ~= 0.79 > 1 - b0 = 0.0 -> dist > includemargin = 0; tangential slip enters CONE
    mjd.qvel[0] = 30.0
    mjd.qvel[2] = -28.0
    _sync_forward(mjm, mjd, m, d)
    self.assertEqual(mjd.ncon, 1)
    self.assertGreater(float(mjd.contact.dist[0]), float(mjd.contact.includemargin[0]))
    self.assertEqual(int(mjd.efc_state[0]), int(ConstraintState.CONE))

    _assert_contact_parity(self, mjd, d)
    self.assertEqual(m.opt.cone, ConeType.ELLIPTIC)
    self.assertEqual(int(d.efc.state.numpy()[0, 0]), int(ConstraintState.CONE))
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, atol=5e-2, rtol=1e-2)
    nefc = int(d.nefc.numpy()[0])
    np.testing.assert_allclose(d.efc.force.numpy()[0, :nefc], mjd.efc_force, atol=5e-2, rtol=1e-2)

  def test_body_adhesion_actuator_speculative(self):
    """Test body/adhesion transmission with positive-dist speculative and replaced gap contacts."""
    # Positive-dist speculative contact (t > 1 - b0) with active body/adhesion actuator
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable" warmstart="disable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="2 2 0.01"/>
          <body name="ball" pos="0 0 0.27">
            <freejoint/>
            <geom name="ball_geom" size="0.05"/>
          </body>
        </worldbody>
        <actuator>
          <adhesion name="adh" body="ball" gain="10" ctrlrange="0 1"/>
        </actuator>
      </mujoco>
      """
    )
    mjd.qvel[2] = -28.0
    mjd.ctrl[0] = 0.5
    d.actuator_moment.fill_(wp.inf)
    _sync_forward(mjm, mjd, m, d)
    self.assertEqual(mjd.ncon, 1)
    self.assertGreater(float(mjd.contact.dist[0]), float(mjd.contact.includemargin[0]))

    _assert_contact_parity(self, mjd, d)
    _assert_body_moment_parity(self, mjm, mjd, d)
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, atol=5e-2, rtol=1e-2)

    # Replaced in-gap discrete contact must not be double-counted in body transmission
    mjm_g, mjd_g, m_g, d_g = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable" warmstart="disable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="2 2 0.01" margin="0.005" gap="0.25"/>
          <body name="ball" pos="0 0 0.20">
            <freejoint/>
            <geom name="ball_geom" size="0.05"/>
          </body>
        </worldbody>
        <actuator>
          <adhesion name="adh" body="ball" gain="10" ctrlrange="0 1"/>
        </actuator>
      </mujoco>
      """
    )
    mjd_g.qvel[0] = 5.0
    mjd_g.qvel[2] = -50.0
    mjd_g.ctrl[0] = 0.5
    d_g.actuator_moment.fill_(wp.inf)
    _sync_forward(mjm_g, mjd_g, m_g, d_g)
    self.assertEqual(mjd_g.ncon, 1)

    _assert_contact_parity(self, mjd_g, d_g)
    _assert_body_moment_parity(self, mjm_g, mjd_g, d_g)
    np.testing.assert_allclose(d_g.qacc.numpy()[0], mjd_g.qacc, atol=5e-2, rtol=1e-2)

  def test_large_world_offset_invariance(self):
    """Test pair-local translation invariance at large world offsets for box-box and plane-box."""
    # Thin tilted card vs box pad at (200, 200, 200) vs origin and MuJoCo C
    mjm0, mjd0, m0, d0 = test_data.fixture(xml=_card_pad_xml())
    mjm_far, mjd_far, m_far, d_far = test_data.fixture(xml=_card_pad_xml((200.0, 200.0, 200.0)))
    mjd0.qvel[2] = -2.0
    mjd_far.qvel[2] = -2.0
    _sync_forward(mjm0, mjd0, m0, d0)
    _sync_forward(mjm_far, mjd_far, m_far, d_far)

    _assert_contact_parity(self, mjd_far, d_far, atol_dist=2e-4, atol_pos=2e-3)
    got0 = _extract_world_contacts(d0)
    got_far = _extract_world_contacts(d_far)
    self.assertEqual(len(got_far), 4)
    offset = np.array([200.0, 200.0, 200.0])
    for c0, cf in zip(got0, got_far):
      np.testing.assert_allclose(cf["dist"], c0["dist"], atol=1e-4, rtol=1e-3)
      np.testing.assert_allclose(cf["pos"] - offset, c0["pos"], atol=5e-4, rtol=1e-3)
    np.testing.assert_allclose(d_far.qacc.numpy()[0], d0.qacc.numpy()[0], atol=5e-2, rtol=1e-2)

    # Tilted plane vs thin box far along the plane at ~(200, 200, 200)
    mjm_pl, mjd_pl, m_pl, d_pl = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" pos="200 200 200" euler="5 -3 0" size="10 10 0.01"/>
          <body pos="200 200 200.25" euler="5 -3 0">
            <freejoint/>
            <geom type="box" size="0.05 0.05 0.01"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    mjd_pl.qvel[2] = -50.0
    _sync_state(mjm_pl, mjd_pl, m_pl, d_pl)
    _assert_contact_parity(self, mjd_pl, d_pl, atol_dist=5e-3, atol_pos=5e-3)
    self.assertEqual(len(_extract_world_contacts(d_pl)), 4)

  def test_tactile_contact_force_and_postconstraint(self):
    """Test <tactile>, rne_postconstraint cfrc_ext, and contact_force skip replaced gap contacts."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <asset>
          <mesh name="pad_mesh" vertex="-0.2 -0.2 -0.02  0.2 -0.2 -0.02  0.2 0.2 -0.02  -0.2 0.2 -0.02
                                        -0.2 -0.2  0.02  0.2 -0.2  0.02  0.2 0.2  0.02  -0.2 0.2  0.02"/>
        </asset>
        <worldbody>
          <body name="pad_body">
            <site name="pad_site"/>
            <geom name="pad" type="mesh" mesh="pad_mesh"/>
          </body>
          <body name="ball" pos="0 0 0.15">
            <freejoint/>
            <geom name="ball_geom" size="0.05" gap="0.20"/>
          </body>
        </worldbody>
        <sensor>
          <tactile geom="pad" mesh="pad_mesh"/>
          <force site="pad_site"/>
        </sensor>
      </mujoco>
      """
    )
    mjd.qvel[2] = -50.0
    d.sensordata.fill_(wp.inf)
    d.cfrc_ext.fill_(wp.spatial_vector(wp.inf))
    _sync_forward(mjm, mjd, m, d)

    _assert_contact_parity(self, mjd, d, atol_dist=5e-3)
    np.testing.assert_allclose(d.sensordata.numpy()[0], mjd.sensordata, atol=5e-2, rtol=2e-2)
    np.testing.assert_allclose(d.cfrc_ext.numpy()[0], mjd.cfrc_ext, atol=5e-2, rtol=2e-2)

    # Verify contact_force returns 0 on the cleared in-gap slot and matches C on the speculative hit
    nacon = int(d.nacon.numpy()[0])
    ctypes = d.contact.type.numpy()[:nacon]
    active_ids = [i for i in range(nacon) if int(ctypes[i]) & int(ContactType.CONSTRAINT)]
    cleared_ids = [i for i in range(nacon) if int(ctypes[i]) == 0]
    self.assertEqual(len(active_ids), 1)
    self.assertEqual(len(cleared_ids), 1)

    frc_active = wp.zeros(1, dtype=wp.spatial_vector)
    mjw.contact_force(m, d, wp.array([active_ids[0]], dtype=int), False, frc_active)
    c_frc = np.zeros(6)
    mujoco.mj_contactForce(mjm, mjd, 0, c_frc)
    np.testing.assert_allclose(frc_active.numpy()[0], c_frc, atol=5e-2, rtol=2e-2)

    frc_cleared = wp.full(1, wp.spatial_vector(wp.inf), dtype=wp.spatial_vector)
    mjw.contact_force(m, d, wp.array([cleared_ids[0]], dtype=int), False, frc_cleared)
    np.testing.assert_allclose(frc_cleared.numpy()[0], 0.0, atol=1e-6)

  @parameterized.parameters(SolverType.CG, SolverType.NEWTON)
  def test_island_coupling_and_determinism_with_replaced_gap(self, solver: SolverType):
    """Test multi-body speculative island coupling and deterministic contact sorting."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" gravity="0 0 0">
          <flag speculative="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0">
            <freejoint/>
            <geom size="0.05" gap="0.20"/>
          </body>
          <body name="b2" pos="0 0 0.22">
            <freejoint/>
            <geom size="0.05" gap="0.20"/>
          </body>
          <body name="b3" pos="0 0 0.44">
            <freejoint/>
            <geom size="0.05"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      overrides={"opt.solver": solver, "opt.deterministic": DeterminismType.CONTACTS},
    )
    qvel = np.zeros((2, mjm.nv), dtype=np.float32)
    # World 0: b3 -> b2 -> b1 chain couples all 3 bodies into a single constraint island
    qvel[0, 8] = -25.0
    qvel[0, 14] = -55.0
    # World 1: separating bodies -> in-gap non-constraint contacts only, 0 islands
    qvel[1, 8] = 10.0
    qvel[1, 14] = 20.0
    d.qvel.assign(qvel)
    d.qacc.fill_(wp.inf)
    d.nisland.fill_(-1)
    d.tree_island.fill_(-1)
    _fill_contact_outputs(d)

    mjw.forward(m, d)
    island.island(m, d)
    mjd.qvel[:] = qvel[0]
    mujoco.mj_forward(mjm, mjd)

    self.assertEqual(int(d.nisland.numpy()[0]), 1)
    self.assertEqual(int(d.nisland.numpy()[0]), mjd.nisland)
    self.assertEqual(int(d.nisland.numpy()[1]), 0)
    _assert_contact_parity(self, mjd, d, world=0, atol_dist=5e-3)
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, atol=5e-2, rtol=2e-2)

    # DeterminismType.CONTACTS produces identical sorted contact buffers across runs with tombstones
    nacon = int(d.nacon.numpy()[0])
    types1 = d.contact.type.numpy()[:nacon].copy()
    pos1 = d.contact.pos.numpy()[:nacon].copy()
    d.qvel.assign(qvel)
    _fill_contact_outputs(d)
    mjw.forward(m, d)
    np.testing.assert_array_equal(d.contact.type.numpy()[:nacon], types1)
    np.testing.assert_array_equal(d.contact.pos.numpy()[:nacon], pos1)


if __name__ == "__main__":
  wp.init()
  absltest.main()
