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

"""Tests for full-DOF 2D flex collisions with static heightfields."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import collision_flex


class FlexHeightfieldTest(parameterized.TestCase):
  @parameterized.product(nworld=(1, 2), slope=(False, True), height=(0.005, 0.012, 0.03))
  def test_surface_contacts(self, nworld, slope, height):
    """Check analytic planes and native distances in a translated, rotated terrain frame."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option ccd_tolerance="1e-7"/>
        <asset><hfield name="terrain" nrow="3" ncol="3" size="1 1 .1 .1"/></asset>
        <worldbody>
          <geom type="hfield" hfield="terrain" pos=".3 -.2 .4" euler="17 -11 23"
                margin=".002" friction=".8 .02 .001" solmix=".3"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1"
                    pos=".07 .03 .005" dim="2" mass=".1" radius=".01">
            <contact selfcollide="none" margin=".002" friction=".6 .01 .002" solmix=".7"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=128,
      njmax=1024,
    )
    sx, sy = (0.02, 0.03) if slope else (0.0, 0.0)
    x, y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    mjm.hfield_data[:] = ((0.04 + sx * x + sy * y) / 0.1).ravel()
    m.hfield_data.assign(mjm.hfield_data)
    rotation = mjd.geom_xmat[0].reshape(3, 3).copy()
    origin = mjd.geom_xpos[0].copy()
    normal_local = np.array([-sx, -sy, 1.0])
    scale = np.linalg.norm(normal_local)
    normal_local /= scale
    normal = rotation @ normal_local
    local = mjd.flexvert_xpos.copy()
    body_pos = mjm.body_pos[mjm.flex_vertbodyid]
    qpos = d.qpos.numpy()
    for w in range(nworld):
      local[:, 2] = 0.04 + sx * local[:, 0] + sy * local[:, 1] + (height + w * 0.0005) * scale
      qpos[w] = ((local @ rotation.T + origin) - body_pos).ravel()
    d.qpos.assign(qpos)
    d.nacon.fill_(-1)
    for array in (d.flexvert_xpos, d.contact.dist, d.contact.pos, d.contact.frame):
      array.fill_(wp.inf)
    for array in (d.contact.geom, d.contact.flex, d.contact.elem, d.contact.vert, d.contact.worldid):
      array.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    n = d.nacon.numpy()[0]
    np.testing.assert_array_equal(d.overflow.numpy(), 0)
    distances = []
    for w in range(nworld):
      mjd.qpos[:] = qpos[w]
      mujoco.mj_forward(mjm, mjd)
      indices = np.flatnonzero(d.contact.worldid.numpy()[:n] == w)
      if height == 0.03:
        self.assertEmpty(indices)
        self.assertEqual(mjd.ncon, 0)
        continue
      self.assertNotEmpty(indices)
      self.assertGreater(mjd.ncon, 0)
      expected = height + w * 0.0005 - 0.01
      dist = d.contact.dist.numpy()[indices]
      np.testing.assert_allclose(dist, expected, atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(mjd.contact.dist, expected, atol=1e-5, rtol=1e-5)
      np.testing.assert_allclose(d.contact.frame.numpy()[indices, 0], np.tile(normal, (len(indices), 1)), atol=1e-4)
      pos_local = (d.contact.pos.numpy()[indices] - origin) @ rotation
      signed_height = pos_local @ normal_local - 0.04 / scale
      np.testing.assert_allclose(signed_height, 0.5 * expected, atol=1e-5, rtol=1e-5)
      np.testing.assert_array_equal(d.contact.geom.numpy()[indices], np.tile([0, -1], (len(indices), 1)))
      np.testing.assert_array_equal(d.contact.flex.numpy()[indices], np.tile([-1, 0], (len(indices), 1)))
      np.testing.assert_array_equal(d.contact.vert.numpy()[indices], -1)
      elems = d.contact.elem.numpy()[indices]
      np.testing.assert_array_equal(elems[:, 0], -1)
      self.assertTrue(((elems[:, 1] >= 0) & (elems[:, 1] < mjm.nflexelem)).all())
      for field in ("friction", "solref", "solimp", "includemargin", "dim"):
        actual = getattr(d.contact, field).numpy()[indices]
        reference = getattr(mjd.contact, field)[0]
        np.testing.assert_allclose(actual, np.broadcast_to(reference, actual.shape), atol=1e-5, rtol=1e-5)
      distances.append(dist.mean())
    if nworld == 2 and height != 0.03:
      self.assertGreater(distances[1] - distances[0], 0.0004)

  @parameterized.product(nworld=(1, 2), side=(False, True), separated=(False, True))
  def test_finite_terrain(self, nworld, side, separated):
    """Side/bottom contacts use the finite prism, not an infinite top plane."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option ccd_tolerance="1e-7"/>
        <asset><hfield name="terrain" nrow="3" ncol="3" size="1 1 .1 .1"/></asset>
        <worldbody>
          <geom type="hfield" hfield="terrain"/>
          <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".02 .02 .02"
                    pos=".4 .2 .005" dim="2" radius=".01">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=128,
    )
    local = mjd.flexvert_xpos.copy()
    qpos = d.qpos.numpy()
    normal = np.array([1, 0, 0]) if side else np.array([0, 0, -1])
    offset = 0.03 if separated else 0.005
    for w in range(nworld):
      vertices = local.copy()
      if side:
        vertices[:, 0] = 1 + offset + w * 0.0005
        vertices[:, 2] = local[:, 0] - 0.45
      else:
        vertices[:, 2] = -0.1 - offset - w * 0.0005
      qpos[w] = (vertices - mjm.body_pos[mjm.flex_vertbodyid]).ravel()
    d.qpos.assign(qpos)
    d.contact.dist.fill_(wp.inf)
    d.contact.frame.fill_(wp.inf)
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)
    n = d.nacon.numpy()[0]
    np.testing.assert_array_equal(d.overflow.numpy(), 0)
    for w in range(nworld):
      mjd.qpos[:] = qpos[w]
      mujoco.mj_forward(mjm, mjd)
      indices = np.flatnonzero(d.contact.worldid.numpy()[:n] == w)
      if separated:
        self.assertEmpty(indices)
        self.assertEqual(mjd.ncon, 0)
      else:
        self.assertNotEmpty(indices)
        self.assertGreater(mjd.ncon, 0)
        expected = offset + w * 0.0005 - 0.01
        np.testing.assert_allclose(d.contact.dist.numpy()[indices], expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(mjd.contact.dist, expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(d.contact.frame.numpy()[indices, 0], np.tile(normal, (len(indices), 1)), atol=1e-4)

  @parameterized.parameters(1, 2)
  def test_graph_rollout(self, nworld):
    """Captured steps retain valid contacts and independently evolving worlds."""
    if not wp.get_device().is_cuda:
      self.skipTest("CUDA graph requires a CUDA device")
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep=".001"/>
        <asset><hfield name="terrain" nrow="3" ncol="3" size="1 1 .1 .1"/></asset>
        <worldbody>
          <geom type="hfield" hfield="terrain"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1"
                    pos=".07 .03 .005" dim="2" mass=".1" radius=".01">
            <contact selfcollide="none"/><edge equality="true"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=128,
      njmax=1024,
    )
    qpos = d.qpos.numpy()
    if nworld == 2:
      qpos[1, 2::3] += 0.001
    d.qpos.assign(qpos)
    d.qvel.fill_(0.01)
    mjw.step(m, d)
    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    for _ in range(30):
      wp.capture_launch(capture.graph)
    self.assertTrue(np.isfinite(d.qpos.numpy()).all())
    self.assertTrue(np.isfinite(d.qvel.numpy()).all())
    self.assertTrue(np.isfinite(d.qacc.numpy()).all())
    np.testing.assert_array_equal(d.overflow.numpy(), 0)
    self.assertGreater(d.nacon.numpy()[0], 0)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))
      untouched = d.qpos.numpy()[1].copy()
      mjw.reset_data(m, d, wp.array([True, False], dtype=bool))
      np.testing.assert_array_equal(d.qpos.numpy()[1], untouched)
      wp.capture_launch(capture.graph)
      self.assertTrue(np.isfinite(d.qacc.numpy()).all())
      np.testing.assert_array_equal(d.overflow.numpy(), 0)

  @parameterized.parameters(1, 2)
  def test_nonplanar_batched_terrain(self, nworld):
    """Different per-world heightfield IDs select different nonplanar terrain assets."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option ccd_tolerance="1e-7"/>
        <asset>
          <hfield name="terrain" nrow="5" ncol="5" size=".3 .3 .1 .1"/>
          <hfield name="other" nrow="4" ncol="4" size=".4 .4 .08 .15"/>
        </asset>
        <worldbody>
          <geom type="hfield" hfield="terrain"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".07 .07 .07"
                    pos=".013 .021 .025" dim="2" mass=".1" radius=".01">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=256,
      batch_sizes={"geom_dataid": nworld},
    )
    rng = np.random.default_rng(7)
    mjm.hfield_data[:] = rng.uniform(0.15, 0.3, size=mjm.nhfielddata)
    m.hfield_data.assign(mjm.hfield_data)
    m.geom_dataid.assign(np.arange(nworld).reshape(nworld, 1))
    d.contact.dist.fill_(wp.inf)
    d.contact.pos.fill_(wp.inf)
    d.contact.frame.fill_(wp.inf)
    d.contact.elem.fill_(-1)
    d.nacon.fill_(-1)
    mjw.collision(m, d)
    n = d.nacon.numpy()[0]
    ws = collision_flex._allocate_flex_workspace(m, d)
    collision_flex._detect_elem_geom_candidates(m, d, ws)
    ncand = ws.ncand.numpy()[0]
    means = []
    for w in range(nworld):
      mjm.geom_dataid[0] = w
      mujoco.mj_forward(mjm, mjd)
      ids = np.flatnonzero(d.contact.worldid.numpy()[:n] == w)
      self.assertNotEmpty(ids)
      self.assertTrue(np.isfinite(d.contact.pos.numpy()[ids]).all())
      self.assertTrue(np.isfinite(d.contact.frame.numpy()[ids]).all())
      # The shared flex filter can assign a coincident vertex contact to the adjacent
      # element. Compare native per-element depths before that existing deduplication.
      candidates = np.flatnonzero(ws.worldid.numpy()[:ncand] == w)
      elems = ws.elem.numpy()[candidates, 1]
      for elem in np.unique(elems):
        actual = ws.dist.numpy()[candidates[elems == elem]].min()
        reference = mjd.contact.dist[mjd.contact.elem[:, 1] == elem].min()
        np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)
      means.append(d.contact.dist.numpy()[ids].mean())
    np.testing.assert_array_equal(d.overflow.numpy(), 0)
    if nworld == 2:
      self.assertNotAlmostEqual(means[0], means[1], places=5)

  @parameterized.parameters("ccd", "candidate", "hfield")
  def test_overflow(self, capacity):
    """Report exhausted scratch/candidate/prism capacity without writing out of bounds."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <asset><hfield name="terrain" nrow="25" ncol="25" size=".12 .12 .1 .1"/></asset>
        <worldbody>
          <geom type="hfield" hfield="terrain"/>
          <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".2 .2 .2"
                    pos=".001 .001 .005" dim="2" radius=".01">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      nconmax=128,
    )
    m.opt.warn_overflow = 0
    if capacity == "ccd":
      d.naccdmax = 0
      expected = mjw.OverflowType.CCD
    elif capacity == "candidate":
      d.naconmax = 1
      expected = mjw.OverflowType.NARROWPHASE
    else:
      expected = mjw.OverflowType.HFIELD
    # Test the detect stage directly so an intentionally tiny candidate buffer is not
    # confused with contact output capacity or the downstream deduplication stage.
    ws = collision_flex._allocate_flex_workspace(m, d)
    collision_flex._detect_elem_geom_candidates(m, d, ws)
    self.assertTrue(((d.overflow.numpy() & int(expected)) != 0).all())

  def test_unsupported_1d_flex(self):
    with self.assertRaisesRegex(NotImplementedError, "full-DOF 2D flexes"):
      test_data.fixture(
        xml="""
        <mujoco>
          <asset><hfield name="terrain" nrow="3" ncol="3" size="1 1 .1 .1"/></asset>
          <worldbody>
            <geom type="hfield" hfield="terrain"/>
            <flexcomp name="flex" type="grid" count="2 1 1" dim="1">
              <contact selfcollide="none"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
      )

  def test_moving_heightfield(self):
    with self.assertRaisesRegex(NotImplementedError, "static heightfield"):
      test_data.fixture(
        xml="""
        <mujoco>
          <asset><hfield name="terrain" nrow="3" ncol="3" size="1 1 .1 .1"/></asset>
          <worldbody>
            <body mocap="true"><geom type="hfield" hfield="terrain"/></body>
            <flexcomp name="cloth" type="grid" count="2 2 1" dim="2">
              <contact selfcollide="none"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
      )

  @parameterized.parameters(1, 2)
  def test_collision_masks(self, nworld):
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <asset><hfield name="terrain" nrow="3" ncol="3" size="1 1 .1 .1"/></asset>
        <worldbody>
          <geom type="hfield" hfield="terrain" contype="1" conaffinity="1"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1"
                    pos=".07 .03 .005" dim="2" radius=".01">
            <contact selfcollide="none" contype="2" conaffinity="2"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    d.nacon.fill_(-1)
    mjw.collision(m, d)
    self.assertEqual(d.nacon.numpy()[0], 0)


if __name__ == "__main__":
  absltest.main()
