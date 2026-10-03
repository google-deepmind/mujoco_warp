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
"""Tests for continuous collision detection (CCD) and conservative advancement."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp import test_data
from mujoco_warp._src import collision_continuous as cc
from mujoco_warp._src import ipc
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import OverflowType

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _eval_pt_tri_kernel(
  # In:
  p: wp.vec3,
  a: wp.vec3,
  b: wp.vec3,
  c: wp.vec3,
  # Out:
  d_out: wp.array[float],
  cp_out: wp.array[wp.vec3],
  w_out: wp.array[wp.vec3],
):
  d, cp, w = cc.pt_tri_dist(p, a, b, c)
  d_out[0] = d
  cp_out[0] = cp
  w_out[0] = w


@wp.kernel
def _eval_seg_seg_kernel(
  # In:
  p1: wp.vec3,
  p2: wp.vec3,
  q1: wp.vec3,
  q2: wp.vec3,
  # Out:
  d_out: wp.array[float],
  cp1_out: wp.array[wp.vec3],
  cp2_out: wp.array[wp.vec3],
  st_out: wp.array[wp.vec2],
):
  d, cp1, cp2, s, t = cc.seg_seg_dist(p1, p2, q1, q2)
  d_out[0] = d
  cp1_out[0] = cp1
  cp2_out[0] = cp2
  st_out[0] = wp.vec2(s, t)


@wp.kernel
def _eval_geom_dist_kernel(
  # In:
  gt: int,
  gsz: wp.vec3,
  gp: wp.vec3,
  gR: wp.mat33,
  p: wp.vec3,
  empty_v: wp.array[wp.vec3],
  empty_vn: wp.array[wp.vec3],
  empty_i: wp.array[int],
  # Out:
  gap_out: wp.array[float],
):
  gap, _n, _liv, _lcw, _lniv = cc.pair_gap(
    empty_v,
    empty_vn,
    empty_i,
    empty_i,
    empty_i,
    int(cc.FlexPairType.FLEX_VERT_GEOM),
    wp.vec4i(0, 0, 0, 0),
    gt,
    gsz,
    0.0,
    gp,
    gR,
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(0.0, 0.0, 0.0),
    p,
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(0.0, 0.0, 0.0),
    -1,
    -1,
    -1,
  )
  gap_out[0] = gap


class ContinuousCollisionTest(parameterized.TestCase):
  def test_point_triangle_distance_kernel(self):
    a = wp.vec3(0.0, 0.0, 0.0)
    b = wp.vec3(1.0, 0.0, 0.0)
    c = wp.vec3(0.0, 1.0, 0.0)
    p = wp.vec3(0.2, 0.2, 0.5)

    out_d = wp.full(1, wp.inf, dtype=float)
    out_cp = wp.full(1, wp.vec3(wp.inf, wp.inf, wp.inf), dtype=wp.vec3)
    out_w = wp.full(1, wp.vec3(wp.inf, wp.inf, wp.inf), dtype=wp.vec3)

    wp.launch(
      _eval_pt_tri_kernel,
      dim=1,
      inputs=[p, a, b, c],
      outputs=[out_d, out_cp, out_w],
    )

    np.testing.assert_allclose(out_d.numpy()[0], 0.5, atol=1e-5)
    np.testing.assert_allclose(out_cp.numpy()[0], [0.2, 0.2, 0.0], atol=1e-5)
    np.testing.assert_allclose(out_w.numpy()[0], [0.6, 0.2, 0.2], atol=1e-5)

  def test_segment_segment_distance_kernel(self):
    p1 = wp.vec3(-1.0, 0.0, 0.0)
    p2 = wp.vec3(1.0, 0.0, 0.0)
    q1 = wp.vec3(0.0, -1.0, 0.3)
    q2 = wp.vec3(0.0, 1.0, 0.3)

    out_d = wp.full(1, wp.inf, dtype=float)
    out_cp1 = wp.full(1, wp.vec3(wp.inf, wp.inf, wp.inf), dtype=wp.vec3)
    out_cp2 = wp.full(1, wp.vec3(wp.inf, wp.inf, wp.inf), dtype=wp.vec3)
    out_st = wp.full(1, wp.vec2(wp.inf, wp.inf), dtype=wp.vec2)

    wp.launch(
      _eval_seg_seg_kernel,
      dim=1,
      inputs=[p1, p2, q1, q2],
      outputs=[out_d, out_cp1, out_cp2, out_st],
    )

    np.testing.assert_allclose(out_d.numpy()[0], 0.3, atol=1e-5)
    np.testing.assert_allclose(out_cp1.numpy()[0], [0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out_cp2.numpy()[0], [0.0, 0.0, 0.3], atol=1e-5)
    np.testing.assert_allclose(out_st.numpy()[0], [0.5, 0.5], atol=1e-5)

  @parameterized.parameters(1, 2)
  def test_ipc_topology_and_candidates(self, nworld):
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1"
                    spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.004"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      fv = d.flexvert_xpos.numpy()
      fv[1, :, 2] += 1.0
      d.flexvert_xpos = wp.array(fv, dtype=wp.vec3, device=d.flexvert_xpos.device)
    ws = ipc.get_ipc_workspace(m, d)
    ws.fidx.fill_(-1)
    ws.aset_type.fill_(-1)
    ipc.ipc_init_topology(m, ws)
    cc.ipc_update_geom_features(m, d, ws)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, 1e30, 1e30)
    ipc.ipc_seed_active_set(m, d, ws)

    self.assertTrue(m.has_2d_flex)
    self.assertEqual(m.nflexvert, 9)
    self.assertEqual(int((ws.fidx.numpy() >= 0).sum()), 9)
    self.assertGreater(int(ws.naset.numpy()[0]), 0)
    if nworld == 2:
      self.assertEqual(int(ws.naset.numpy()[1]), 0)

  @parameterized.parameters(1, 2)
  def test_advance_wfree_kernel(self, nworld):
    nv = 6
    own = wp.array([True] * nv, dtype=bool)
    is_throttled = wp.array([True] * nv, dtype=bool)
    alpha_vals = [0.5 if w == 0 else 0.25 for w in range(nworld)]
    alpha_min = wp.array(alpha_vals, dtype=float)
    last_ls_alpha = wp.array([1.0] * nworld, dtype=float)
    newton_converged = wp.array([0] * nworld, dtype=int)

    w_np = np.array([[1.0 + w, 2.0 + w, 3.0 + w, 4.0 + w, 5.0 + w, 6.0 + w] for w in range(nworld)], dtype=np.float32)
    w = wp.array(w_np, dtype=float)
    wfree = wp.zeros((nworld, nv), dtype=float)
    beta = wp.zeros(nworld, dtype=float)
    world_done = wp.zeros(nworld, dtype=int)
    stall = wp.full(nworld, -1, dtype=int)

    wp.launch(
      ipc.ipc_advance_wfree_kernel,
      dim=(nworld, nv),
      inputs=[own, is_throttled, alpha_min, w, world_done],
      outputs=[wfree],
    )
    wp.launch(
      ipc.ipc_advance_world_state_kernel,
      dim=nworld,
      inputs=[alpha_min, last_ls_alpha, newton_converged],
      outputs=[beta, world_done, stall],
    )

    for w_idx in range(nworld):
      np.testing.assert_allclose(wfree.numpy()[w_idx], alpha_vals[w_idx] * w_np[w_idx], atol=1e-5)
      np.testing.assert_allclose(beta.numpy()[w_idx], alpha_vals[w_idx], atol=1e-5)
      self.assertEqual(world_done.numpy()[w_idx], 0)
      self.assertEqual(stall.numpy()[w_idx], 0)

  def test_geom_distance_kernel(self):
    empty_v = wp.zeros(0, dtype=wp.vec3)
    empty_vn = wp.zeros(0, dtype=wp.vec3)
    empty_i = wp.zeros(0, dtype=int)
    out_gap = wp.full(1, wp.inf, dtype=float)

    def eval_gap(gt, gsz, gp, gR, p):
      wp.launch(
        _eval_geom_dist_kernel,
        dim=1,
        inputs=[gt, gsz, gp, gR, p, empty_v, empty_vn, empty_i],
        outputs=[out_gap],
      )
      return float(out_gap.numpy()[0])

    gR_id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    # Box half-extents 0.1 in x, y, z
    gap_box = eval_gap(
      int(GeomType.BOX),
      wp.vec3(0.1, 0.1, 0.1),
      wp.vec3(0.0, 0.0, 0.0),
      gR_id,
      wp.vec3(0.5, 0.0, 0.0),
    )
    self.assertAlmostEqual(gap_box, 0.4, places=5)

    # Box interior point
    gap_box_int = eval_gap(
      int(GeomType.BOX),
      wp.vec3(0.1, 0.1, 0.1),
      wp.vec3(0.0, 0.0, 0.0),
      gR_id,
      wp.vec3(0.0, 0.0, 0.0),
    )
    self.assertAlmostEqual(gap_box_int, -0.1, places=5)

    # Sphere radius 0.1 at (1, 0, 0)
    gap_sph = eval_gap(
      int(GeomType.SPHERE),
      wp.vec3(0.1, 0.0, 0.0),
      wp.vec3(1.0, 0.0, 0.0),
      gR_id,
      wp.vec3(1.3, 0.0, 0.0),
    )
    self.assertAlmostEqual(gap_sph, 0.2, places=5)

    # Plane at z = -1.0 with normal +z
    gap_pl = eval_gap(
      int(GeomType.PLANE),
      wp.vec3(0.0, 0.0, 0.0),
      wp.vec3(0.0, 0.0, -1.0),
      gR_id,
      wp.vec3(0.3, -0.2, 0.0),
    )
    self.assertAlmostEqual(gap_pl, 1.0, places=5)

    # Capsule radius 0.05, half-height 0.2 along z at origin
    gap_cap = eval_gap(
      int(GeomType.CAPSULE),
      wp.vec3(0.05, 0.2, 0.0),
      wp.vec3(0.0, 0.0, 0.0),
      gR_id,
      wp.vec3(0.15, 0.0, 0.0),
    )
    self.assertAlmostEqual(gap_cap, 0.10, places=5)

  @parameterized.parameters(1, 2)
  def test_ipc_reach_filtering(self, nworld):
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cloth1" type="grid" dim="2" count="2 2 1"
                    spacing="0.05 0.05 1" radius="0.005" mass="0.05">
            <contact selfcollide="none"/>
          </flexcomp>
          <flexcomp name="cloth2" type="grid" dim="2" count="2 2 1"
                    spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 1.0">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    ws = ipc.get_ipc_workspace(m, d)
    cc.ipc_update_geom_features(m, d, ws)

    # Zero displacement prunes cloths 1.0m apart
    ws.aset_type.fill_(-1)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, 0.08, 0.08)
    ipc.ipc_seed_active_set(m, d, ws)
    for w in range(nworld):
      self.assertEqual(int(ws.naset.numpy()[w]), 0)

    # Large displacement sweeping cloth2 toward cloth1 in world 0 admits inter-flex candidate pairs
    dto_np = d.flexvert_xpos.numpy().copy()
    dto_np[0, 4:, 2] = 0.0
    dto = wp.array(dto_np, dtype=wp.vec3, device=d.flexvert_xpos.device)
    ws.aset_type.fill_(-1)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, dto, 1e30, 1e30)
    ipc.ipc_seed_active_set(m, d, ws)
    self.assertGreater(int(ws.naset.numpy()[0]), 0)
    if nworld == 2:
      self.assertEqual(int(ws.naset.numpy()[1]), 0)

  @parameterized.parameters(1, 2)
  def test_ipc_exclude_filtering(self, nworld):
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="obstacle">
            <geom type="box" size="0.1 0.1 0.1"/>
          </body>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1"
                    spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.02">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
        <contact>
          <exclude body1="obstacle" body2="cloth_0"/>
          <exclude body1="obstacle" body2="cloth_1"/>
          <exclude body1="obstacle" body2="cloth_2"/>
          <exclude body1="obstacle" body2="cloth_3"/>
        </contact>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      fv = d.flexvert_xpos.numpy()
      fv[1, :, 2] += 0.01
      d.flexvert_xpos = wp.array(fv, dtype=wp.vec3, device=d.flexvert_xpos.device)
    ws = ipc.get_ipc_workspace(m, d)
    ws.aset_type.fill_(-1)
    cc.ipc_update_geom_features(m, d, ws)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, 1e30, 1e30)
    ipc.ipc_seed_active_set(m, d, ws)
    # All flex vertices excluded against obstacle geom
    for w in range(nworld):
      self.assertEqual(int(ws.naset.numpy()[w]), 0)

  @parameterized.parameters(1, 2)
  def test_ipc_static_geom_retained(self, nworld):
    """Verifies that static geoms are discovered in candidate pairs when approached."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom name="sphere" size="0.3" pos="0 0 0.3"/>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1"
                    spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.88">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    ws = ipc.get_ipc_workspace(m, d)
    ws.aset_type.fill_(-1)
    cc.ipc_update_geom_features(m, d, ws)
    dto_np = d.flexvert_xpos.numpy().copy()
    dto_np[0, :, 2] -= 0.30
    dto = wp.array(dto_np, dtype=wp.vec3, device=d.flexvert_xpos.device)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, dto, 0.08, 0.08)
    ipc.ipc_seed_active_set(m, d, ws)
    naset = int(ws.naset.numpy()[0])
    pair_type = ws.aset_type.numpy()[0, :naset]
    # 4 vertices vs 1 static sphere = 4 FLEX_VERT_GEOM pairs plus 2 GEOM_CORNER_TRI pairs
    self.assertEqual((pair_type == int(cc.FlexPairType.FLEX_VERT_GEOM)).sum(), 4)
    self.assertEqual(naset, 6)
    if nworld == 2:
      self.assertEqual(int(ws.naset.numpy()[1]), 0)

  @parameterized.parameters(1, 2)
  def test_flex_flex_midsurface_contact_admitted(self, nworld):
    """Verifies flex-flex pairs use midsurface offset for admission and r_min for standoff."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <flexcomp name="c1" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1"
                    radius="0.001" mass="0.05">
            <contact selfcollide="none"/>
          </flexcomp>
          <flexcomp name="c2" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1"
                    radius="0.002" mass="0.05" pos="0 0 0.0031">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      fv = d.flexvert_xpos.numpy()
      fv[1, :, 2] += 0.001
      d.flexvert_xpos = wp.array(fv, dtype=wp.vec3, device=d.flexvert_xpos.device)
    ws = ipc.get_ipc_workspace(m, d)
    ws.aset_type.fill_(-1)
    ws.aset_standoff.fill_(wp.inf)
    cc.ipc_update_geom_features(m, d, ws)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, -1e30, -1e30)
    ipc.ipc_seed_active_set(m, d, ws)

    for w in range(nworld):
      naset_w = int(ws.naset.numpy()[w])
      self.assertGreater(naset_w, 0)
      types = ws.aset_type.numpy()[w, :naset_w]
      standoffs = ws.aset_standoff.numpy()[w, :naset_w]
      flex_flex_idx = np.where(types == int(cc.FlexPairType.FLEX_VERT_TRI))[0]
      self.assertGreater(len(flex_flex_idx), 0)
      for idx in flex_flex_idx:
        self.assertAlmostEqual(float(standoffs[idx]), 0.001, places=6)

  @parameterized.parameters(1, 2)
  def test_dynamic_geom_corners_and_edges(self, nworld):
    """Verifies box corners and edges are dynamically transformed per world."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="box" type="box" size="0.1 0.2 0.3"/>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1"
                    radius="0.005" mass="0.05" pos="0 0 0.35">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      gxpos = d.geom_xpos.numpy()
      gxpos[1, 0] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
      d.geom_xpos = wp.array(gxpos, dtype=wp.vec3, device=d.geom_xpos.device)

    ws = ipc.get_ipc_workspace(m, d)
    ws.ngv.fill_(-1)
    ws.nge.fill_(-1)
    ws.geom_corners.fill_(wp.vec3(wp.inf, wp.inf, wp.inf))
    ws.geom_edges.fill_(wp.vec3(wp.inf, wp.inf, wp.inf))
    ipc.ipc_init_topology(m, ws)
    cc.ipc_update_geom_features(m, d, ws)

    for w in range(nworld):
      self.assertEqual(int(ws.ngv.numpy()[w]), 8)
      self.assertEqual(int(ws.nge.numpy()[w]), 12)

    corners_np = ws.geom_corners.numpy()[:, :8]
    edges_np = ws.geom_edges.numpy()[:, :12]
    np.testing.assert_allclose(np.abs(corners_np[0]), np.broadcast_to([0.1, 0.2, 0.3], (8, 3)), atol=1e-6)
    for e in range(12):
      edge_len = float(np.linalg.norm(edges_np[0, e, 1] - edges_np[0, e, 0]))
      self.assertTrue(any(abs(edge_len - want) < 1e-5 for want in (0.2, 0.4, 0.6)))
    if nworld == 2:
      np.testing.assert_allclose(corners_np[1] - corners_np[0], np.broadcast_to([1.0, 2.0, 3.0], (8, 3)), atol=1e-5)

  @parameterized.parameters(1, 2)
  def test_ipc_candidates_midstep_discovery(self, nworld):
    """Verifies ipc_discover_candidates discovers new candidates mid-step when fsweep expands."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="1 1 0.1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1"
                    radius="0.005" mass="0.05" pos="0 0 0.05">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    ws = ipc.get_ipc_workspace(m, d)
    ipc.ipc_init_topology(m, ws)
    cc.ipc_update_geom_features(m, d, ws)

    x = d.flexvert_xpos
    # At z = 0.05 with zero fsweep, cloth is outside band -> 0 admitted
    cc.ipc_discover_candidates(m, d, ws, x, x, x, -1e30, -1e30)
    ipc.ipc_seed_active_set(m, d, ws)
    self.assertEqual(int(ws.naset.numpy().sum()), 0)

    # With dto moving vertices by -0.046m in world 0, dl/fsweep expands bound in world 0
    dto_np = x.numpy().copy()
    dto_np[0, :, 2] -= 0.046
    if nworld == 2:
      dto_np[1, :, 2] -= 0.01
    dto = wp.array(dto_np, dtype=wp.vec3, device=x.device)
    cc.ipc_discover_candidates(m, d, ws, x, x, dto, 0.0, 0.002)
    self.assertEqual(int(ws.ncand.numpy()[0]), 4)
    if nworld == 2:
      self.assertEqual(int(ws.ncand.numpy()[1]), 0)

  def test_pinned_carrier_must_translate(self):
    """Verifies put_model rejects non-slide carriers and free vertices on jointed bodies in IPC."""
    with self.assertRaisesRegex(ValueError, "not a slide"):
      test_data.fixture(
        xml="""
        <mujoco>
          <option integrator="discrete" solver="CG">
            <flag ipc="enable"/>
          </option>
          <worldbody>
            <body name="carrier" pos="0 0 0.5">
              <joint axis="0 1 0"/>
              <geom size="0.02" mass="0.1"/>
              <flexcomp name="cloth" type="grid" dim="2" count="2 2 1"
                        spacing="0.05 0.05 1" radius="0.005" mass="0.05">
                <pin id="0"/>
              </flexcomp>
            </body>
          </worldbody>
        </mujoco>
        """
      )

    with self.assertRaisesRegex(ValueError, "free vertices under a jointed body"):
      test_data.fixture(
        xml="""
        <mujoco>
          <option integrator="discrete" solver="CG">
            <flag ipc="enable"/>
          </option>
          <worldbody>
            <body name="carrier" pos="0 0 0.5">
              <joint type="slide" axis="1 0 0"/>
              <geom size="0.02" mass="0.1"/>
              <flexcomp name="cloth" type="grid" dim="2" count="2 2 1"
                        spacing="0.05 0.05 1" radius="0.005" mass="0.05">
                <pin id="0"/>
              </flexcomp>
            </body>
          </worldbody>
        </mujoco>
        """
      )

  @parameterized.parameters(1, 2)
  def test_ipc_candidate_overflow(self, nworld):
    """Verifies candidate discovery sets OverflowType.BROADPHASE when nccdmax is exceeded."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1"
                    spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.004">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=2,
    )
    if nworld == 2:
      fv = d.flexvert_xpos.numpy()
      fv[1, :, 2] += 1.0
      d.flexvert_xpos = wp.array(fv, dtype=wp.vec3, device=d.flexvert_xpos.device)
    ws = ipc.get_ipc_workspace(m, d)
    self.assertEqual(ws.max_aset, 2)
    self.assertEqual(ws.max_cand, 2)
    d.overflow.zero_()
    cc.ipc_update_geom_features(m, d, ws)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, 1e30, 1e30)
    self.assertGreater(int(ws.ncand.numpy()[0]), ws.max_cand)
    self.assertTrue(bool(int(d.overflow.numpy()[0]) & int(OverflowType.BROADPHASE)))
    if nworld == 2:
      self.assertEqual(int(ws.ncand.numpy()[1]), 0)
      self.assertEqual(int(d.overflow.numpy()[1]), 0)


if __name__ == "__main__":
  absltest.main()
