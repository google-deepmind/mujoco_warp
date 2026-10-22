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


@wp.kernel
def _eval_geom_point_dist_kernel(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  # In:
  gt: int,
  gp: wp.vec3,
  gR: wp.mat33,
  va: int,
  pa: int,
  pn: int,
  pos: wp.array[wp.vec3],
  # Out:
  dist_out: wp.array[float],
  normal_out: wp.array[wp.vec3],
):
  i = wp.tid()
  dist, n = cc.geom_point_dist_normal(
    mesh_vert,
    mesh_polynormal,
    mesh_polyvertadr,
    mesh_polyvertnum,
    mesh_polyvert,
    gt,
    wp.vec3(0.0, 0.0, 0.0),
    gp,
    gR,
    pos[i],
    va,
    pa,
    pn,
    1e30,
  )
  dist_out[i] = dist
  normal_out[i] = n


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

  def test_geom_dist_mesh(self):
    """Verifies signed mesh distance and normal at face, edge, vertex, and interior points."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <asset>
          <mesh name="box" vertex=" -0.1 -0.1 -0.1  0.1 -0.1 -0.1  0.1 0.1 -0.1  -0.1 0.1 -0.1 -0.1 -0.1 0.1  0.1 -0.1 0.1  0.1 0.1 0.1  -0.1 0.1 0.1"/>
        </asset>
        <worldbody>
          <geom type="mesh" mesh="box" pos="0.5 0 0"/>
        </worldbody>
      </mujoco>
      """
    )
    gp = wp.vec3(*d.geom_xpos.numpy()[0, 0])
    gR = wp.mat33(*d.geom_xmat.numpy()[0, 0].flatten())
    offset = np.array([0.5, 0.0, 0.0])
    query = np.array(
      [
        [0.0, 0.0, 0.3],
        [0.2, 0.0, 0.2],
        [0.2, 0.2, 0.2],
        [0.0, 0.0, 0.05],
      ]
    )
    pos = wp.array(query + offset, dtype=wp.vec3)
    dist = wp.full(4, wp.inf, dtype=float)
    normal = wp.full(4, wp.vec3(wp.inf, wp.inf, wp.inf), dtype=wp.vec3)
    wp.launch(
      _eval_geom_point_dist_kernel,
      dim=4,
      inputs=[
        m.mesh_vert,
        m.mesh_polynormal,
        m.mesh_polyvertadr,
        m.mesh_polyvertnum,
        m.mesh_polyvert,
        int(GeomType.MESH),
        gp,
        gR,
        int(mjm.mesh_vertadr[0]),
        int(mjm.mesh_polyadr[0]),
        int(mjm.mesh_polynum[0]),
        pos,
      ],
      outputs=[dist, normal],
    )
    s2 = np.sqrt(0.5)
    s3 = np.sqrt(1.0 / 3.0)
    np.testing.assert_allclose(dist.numpy(), [0.2, np.sqrt(0.02), np.sqrt(0.03), -0.05], atol=1e-5)
    np.testing.assert_allclose(
      normal.numpy(),
      [[0.0, 0.0, 1.0], [s2, 0.0, s2], [s3, s3, s3], [0.0, 0.0, 1.0]],
      atol=1e-5,
    )

  @parameterized.parameters(1, 2)
  def test_ipc_advance_toi(self, nworld):
    """Verifies conservative advancement stops an approaching vertex at 20% of its initial gap."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.001" mass="0.05" pos="0 0 0.01">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    xfree = d.flexvert_xpos.numpy()
    x = xfree.copy()
    x[:, :, 2] -= 0.05
    if nworld == 2:
      xfree[1, :, 2] += 1.0
      x[1, :, 2] = xfree[1, :, 2] - 0.01
    ws = ipc.get_ipc_workspace(m, d)
    ws.world_done.zero_()
    ws.xfree.assign(xfree)
    ws.x.assign(x)
    cc.ipc_update_geom_features(m, d, ws)
    cc.ipc_discover_candidates(m, d, ws, ws.xfree, ws.xfree, ws.x, 3.0 * ipc.IPC_GHAT, ipc.IPC_GHAT)
    ws.alpha_min.fill_(wp.inf)
    cc.ipc_advance(m, d, ws)

    ncand = int(ws.ncand.numpy()[0])
    self.assertGreater(ncand, 0)
    g0 = float(np.min(ws.cand_ld0.numpy()[0, :ncand]))
    alpha = float(ws.alpha_min.numpy()[0])
    np.testing.assert_allclose(alpha, 0.8 * g0 / 0.05, rtol=1e-2)
    if nworld == 2:
      self.assertEqual(int(ws.ncand.numpy()[1]), 0)
      self.assertEqual(float(ws.alpha_min.numpy()[1]), 1.0)

  @parameterized.parameters(1, 2)
  def test_flex_edge_edge_candidates(self, nworld):
    """Verifies FLEX_EDGE_EDGE candidates between crossing edges of two flexes."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <flexcomp name="a" type="grid" dim="2" count="2 2 1" spacing="0.1 0.1 1" radius="0.001" mass="0.05">
            <contact selfcollide="none"/>
          </flexcomp>
          <flexcomp name="b" type="grid" dim="2" count="2 2 1" spacing="0.1 0.1 1" radius="0.001" mass="0.05" pos="0 0 0.002" euler="0 0 45">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      fv = d.flexvert_xpos.numpy()
      fv[1, 4:, 2] += 1.0
      d.flexvert_xpos = wp.array(fv, dtype=wp.vec3, device=d.flexvert_xpos.device)
    ws = ipc.get_ipc_workspace(m, d)
    ws.world_done.zero_()
    ws.ncand.fill_(-1)
    cc.ipc_update_geom_features(m, d, ws)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, 3.0 * ipc.IPC_GHAT, ipc.IPC_GHAT)

    ncand = int(ws.ncand.numpy()[0])
    types = ws.cand_type.numpy()[0, :ncand]
    idxs = ws.cand_idx.numpy()[0, :ncand]
    ee = idxs[types == int(cc.FlexPairType.FLEX_EDGE_EDGE)]
    self.assertGreater(len(ee), 0)
    vflex = m.flex_vertflexid.numpy()
    for a1, b1, a2, b2 in ee:
      self.assertNotEqual(vflex[a1], vflex[a2])
      self.assertEqual(vflex[a1], vflex[b1])
      self.assertEqual(vflex[a2], vflex[b2])
    self.assertEqual(len({tuple(sorted(((a1, b1), (a2, b2)))) for a1, b1, a2, b2 in ee}), len(ee))
    if nworld == 2:
      self.assertEqual(int(ws.ncand.numpy()[1]), 0)

  @parameterized.parameters(1, 2)
  def test_ipc_topology_and_candidates(self, nworld):
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.004"/>
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
    ipc.ipc_init_topology(m, d, ws)
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

    wfree_np = wfree.numpy()
    beta_np = beta.numpy()
    world_done_np = world_done.numpy()
    stall_np = stall.numpy()
    for w_idx in range(nworld):
      np.testing.assert_allclose(wfree_np[w_idx], alpha_vals[w_idx] * w_np[w_idx], atol=1e-5)
      np.testing.assert_allclose(beta_np[w_idx], alpha_vals[w_idx], atol=1e-5)
      self.assertEqual(world_done_np[w_idx], 0)
      self.assertEqual(stall_np[w_idx], 0)

  def test_geom_distance_kernel(self):
    empty_v = wp.zeros(0, dtype=wp.vec3)
    empty_vn = wp.zeros(0, dtype=wp.vec3)
    empty_i = wp.zeros(0, dtype=int)
    out_gap = wp.full(1, wp.inf, dtype=float)

    def eval_gap(gt, gsz, gp, gR, p):
      out_gap.fill_(wp.inf)
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
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <flexcomp name="cloth1" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05">
            <contact selfcollide="none"/>
          </flexcomp>
          <flexcomp name="cloth2" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 1.0">
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
    naset_np = ws.naset.numpy()
    for w in range(nworld):
      self.assertEqual(int(naset_np[w]), 0)

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
  def test_ipc_static_geom_ignores_body_filters(self, nworld):
    """Verifies flex vs static geom candidates ignore body exclude rules, matching MuJoCo C."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <body name="obstacle">
            <geom type="box" size="0.1 0.1 0.1"/>
          </body>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.102">
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
      fv[1, :, 2] += 1.0
      d.flexvert_xpos = wp.array(fv, dtype=wp.vec3, device=d.flexvert_xpos.device)
    ws = ipc.get_ipc_workspace(m, d)
    ws.naset.fill_(-1)
    cc.ipc_update_geom_features(m, d, ws)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, 3.0 * ipc.IPC_GHAT, ipc.IPC_GHAT)
    ipc.ipc_seed_active_set(m, d, ws)
    self.assertGreater(int(ws.naset.numpy()[0]), 0)
    if nworld == 2:
      self.assertEqual(int(ws.naset.numpy()[1]), 0)

  @parameterized.parameters(1, 2)
  def test_ipc_static_geom_retained(self, nworld):
    """Verifies that static geoms are discovered in candidate pairs when approached."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom name="sphere" size="0.3" pos="0 0 0.3"/>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.88">
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
          <flexcomp name="c1" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.001" mass="0.05">
            <contact selfcollide="none"/>
          </flexcomp>
          <flexcomp name="c2" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.002" mass="0.05" pos="0 0 0.0031">
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

    naset_np = ws.naset.numpy()
    types_np = ws.aset_type.numpy()
    standoffs_np = ws.aset_standoff.numpy()
    for w in range(nworld):
      naset_w = int(naset_np[w])
      self.assertGreater(naset_w, 0)
      types = types_np[w, :naset_w]
      standoffs = standoffs_np[w, :naset_w]
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
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.35">
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
    ipc.ipc_init_topology(m, d, ws)
    cc.ipc_update_geom_features(m, d, ws)

    ngv_np = ws.ngv.numpy()
    nge_np = ws.nge.numpy()
    for w in range(nworld):
      self.assertEqual(int(ngv_np[w]), 8)
      self.assertEqual(int(nge_np[w]), 12)

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
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.05">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    ws = ipc.get_ipc_workspace(m, d)
    ipc.ipc_init_topology(m, d, ws)
    cc.ipc_update_geom_features(m, d, ws)

    x = d.flexvert_xpos
    # At z = 0.05 with zero fsweep, cloth is outside band -> 0 admitted
    ws.ncand.fill_(-1)
    ws.naset.fill_(-1)
    cc.ipc_discover_candidates(m, d, ws, x, x, x, -1e30, -1e30)
    ipc.ipc_seed_active_set(m, d, ws)
    self.assertEqual(int(ws.naset.numpy().sum()), 0)

    # With dto moving vertices by -0.046m in world 0, dl/fsweep expands bound in world 0
    dto_np = x.numpy().copy()
    dto_np[0, :, 2] -= 0.046
    if nworld == 2:
      dto_np[1, :, 2] -= 0.01
    dto = wp.array(dto_np, dtype=wp.vec3, device=x.device)
    ws.ncand.fill_(-1)
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
              <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05">
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
              <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05">
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
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.004">
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
    ws.ncand.fill_(-1)
    cc.ipc_update_geom_features(m, d, ws)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, 1e30, 1e30)
    self.assertGreater(int(ws.ncand.numpy()[0]), ws.max_cand)
    self.assertTrue(bool(int(d.overflow.numpy()[0]) & int(OverflowType.BROADPHASE)))
    if nworld == 2:
      self.assertEqual(int(ws.ncand.numpy()[1]), 0)
      self.assertEqual(int(d.overflow.numpy()[1]), 0)

  @parameterized.parameters(1, 2)
  def test_multi_instance_mesh_features_and_overflow(self, nworld):
    """Verifies per-geom-instance mesh feature capacity and overflow reporting."""
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    verts = []
    for z in (-0.02, 0.02):
      for a in angles:
        verts.append(f"{0.1 * np.cos(a):.6f} {0.1 * np.sin(a):.6f} {z:.6f}")
    vert_str = "  ".join(verts)

    _, _, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <asset>
          <mesh name="prism" vertex="{vert_str}"/>
        </asset>
        <worldbody>
          <geom name="p0" type="mesh" mesh="prism" pos="-0.3 0 0"/>
          <geom name="p1" type="mesh" mesh="prism"/>
          <geom name="p2" type="mesh" mesh="prism" pos="0.3 0 0"/>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.05">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    ws = ipc.get_ipc_workspace(m, d)
    d.overflow.zero_()
    ws.geom_corners.fill_(wp.vec3(wp.inf))
    ws.geom_edges.fill_(wp.vec3(wp.inf))
    cc.ipc_update_geom_features(m, d, ws)

    ngv_np = ws.ngv.numpy()
    nge_np = ws.nge.numpy()
    overflow_np = d.overflow.numpy()
    corner_geom_np = ws.corner_geom.numpy()
    edge_geom_np = ws.edge_geom.numpy()
    for w in range(nworld):
      self.assertEqual(int(ngv_np[w]), 96)
      self.assertEqual(int(nge_np[w]), 144)
      self.assertEqual(int(overflow_np[w]), 0)
      corner_geoms = corner_geom_np[w, :96]
      edge_geoms = edge_geom_np[w, :144]
      self.assertEqual([int((corner_geoms == gi).sum()) for gi in range(3)], [32, 32, 32])
      self.assertEqual([int((edge_geoms == gi).sum()) for gi in range(3)], [48, 48, 48])

    # Undersized feature buffers report OverflowType.BROADPHASE and preserve unclamped counts
    ws.max_ngv = 56
    ws.max_nge = 132
    ws.geom_corner_loc = wp.empty((nworld, 56), dtype=wp.vec3, device=d.qpos.device)
    ws.geom_corners = wp.empty((nworld, 56), dtype=wp.vec3, device=d.qpos.device)
    ws.corner_geom = wp.empty((nworld, 56), dtype=int, device=d.qpos.device)
    ws.geom_edge_loc = wp.empty((nworld, 132, 2), dtype=wp.vec3, device=d.qpos.device)
    ws.geom_edges = wp.empty((nworld, 132, 2), dtype=wp.vec3, device=d.qpos.device)
    ws.edge_geom = wp.empty((nworld, 132), dtype=int, device=d.qpos.device)

    d.overflow.zero_()
    ipc.ipc_init_topology(m, d, ws)
    cc.ipc_update_geom_features(m, d, ws)
    ngv_np = ws.ngv.numpy()
    nge_np = ws.nge.numpy()
    overflow_np = d.overflow.numpy()
    for w in range(nworld):
      self.assertEqual(int(ngv_np[w]), 96)
      self.assertEqual(int(nge_np[w]), 144)
      self.assertTrue(bool(int(overflow_np[w]) & int(OverflowType.BROADPHASE)))

  @parameterized.parameters(1, 2)
  def test_ipc_discover_candidates_same_body(self, nworld):
    """Verifies cross-flex candidates sharing a rigid body (sameBody) are filtered out."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <flexcomp name="c1" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.001" mass="0.05">
            <contact selfcollide="none"/>
            <pin id="0"/>
          </flexcomp>
          <flexcomp name="c2" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.002" mass="0.05" pos="0 0 0.0031">
            <contact selfcollide="none"/>
            <pin id="0"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    if nworld == 2:
      fv = d.flexvert_xpos.numpy()
      fv[1, 4:, 2] += 1.0
      d.flexvert_xpos = wp.array(fv, dtype=wp.vec3, device=d.flexvert_xpos.device)
    ws = ipc.get_ipc_workspace(m, d)
    ws.ncand.fill_(-1)
    cc.ipc_update_geom_features(m, d, ws)
    cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, -1e30, -1e30)

    ncand0 = int(ws.ncand.numpy()[0])
    self.assertGreater(ncand0, 0)
    cand_types = ws.cand_type.numpy()[0, :ncand0]
    cand_idxs = ws.cand_idx.numpy()[0, :ncand0]
    vbody = m.flex_vertbodyid.numpy()
    for t, idx in zip(cand_types, cand_idxs):
      if t == int(cc.FlexPairType.FLEX_VERT_TRI):
        v, v0, v1, v2 = idx
        if vbody[v] >= 0:
          self.assertNotIn(vbody[v], (vbody[v0], vbody[v1], vbody[v2]))
      elif t == int(cc.FlexPairType.FLEX_EDGE_EDGE):
        a1, b1, a2, b2 = idx
        for va in (a1, b1):
          if vbody[va] >= 0:
            self.assertNotIn(vbody[va], (vbody[a2], vbody[b2]))
    if nworld == 2:
      self.assertEqual(int(ws.ncand.numpy()[1]), 0)

  @parameterized.parameters(1, 2)
  def test_flex_grid_matches_all_pairs(self, nworld):
    """Verifies the flex hash grid finds exactly the all-pairs flex-flex candidates."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <flexcomp name="cloth" type="grid" dim="2" count="7 7 1" spacing="0.01 0.01 1" radius="0.001" mass="0.05"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    # fold the cloth's right half over its left half, 2 mm above (4 mm in the second world)
    fv = d.flexvert_xpos.numpy()
    for w in range(nworld):
      right = fv[w, :, 0] > 1e-6
      fv[w, right, 0] *= -1.0
      fv[w, right, 2] += 0.002 * (w + 1)
    d.flexvert_xpos = wp.array(fv, dtype=wp.vec3)
    flexflex = (int(cc.FlexPairType.FLEX_VERT_TRI), int(cc.FlexPairType.FLEX_EDGE_EDGE))

    def candidates(min_pairs, ws):
      cc.BP_GRID_MIN_PAIRS = min_pairs
      ws.ncand.fill_(-1)
      cc.ipc_discover_candidates(m, d, ws, d.flexvert_xpos, d.flexvert_xpos, d.flexvert_xpos, 3.0 * ipc.IPC_GHAT, ipc.IPC_GHAT)
      ncand = ws.ncand.numpy()
      types = ws.cand_type.numpy()
      idxs = ws.cand_idx.numpy()
      ld0 = ws.cand_ld0.numpy()
      return [
        sorted((int(types[w, c]), tuple(idxs[w, c]), float(ld0[w, c])) for c in range(ncand[w]) if types[w, c] in flexflex)
        for w in range(nworld)
      ]

    default_min_pairs = cc.BP_GRID_MIN_PAIRS
    try:
      cc.BP_GRID_MIN_PAIRS = 0
      ws = ipc.get_ipc_workspace(m, d)
      ws.world_done.zero_()
      cc.ipc_update_geom_features(m, d, ws)
      grid = candidates(0, ws)
      all_pairs = candidates(1 << 62, ws)
    finally:
      cc.BP_GRID_MIN_PAIRS = default_min_pairs

    for w in range(nworld):
      self.assertGreater(len(grid[w]), 0)
      self.assertEqual(grid[w], all_pairs[w])
    for t in flexflex:
      self.assertTrue(any(c[0] == t for c in grid[0]))
    if nworld == 2:
      self.assertNotEqual(len(grid[0]), len(grid[1]))

  @parameterized.parameters(1, 2)
  def test_capsule_box_feature_partial_overflow(self, nworld):
    """Verifies capsule/box feature extraction initializes slots up to capacity."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag ipc="enable"/>
        </option>
        <worldbody>
          <geom type="capsule" size="0.05 0.2"/>
          <geom type="box" size="0.1 0.1 0.1" pos="0.3 0 0"/>
          <flexcomp name="cloth" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 1" radius="0.005" mass="0.05" pos="0 0 0.5">
            <contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    ws = ipc.get_ipc_workspace(m, d)
    # 1 capsule (2 corners, 1 edge) + 1 box (8 corners, 12 edges) = 10 corners, 13 edges
    # Bisect capsule at max_ngv=1 first, then bisect box at max_ngv=5, max_nge=6
    for max_c, max_e in [(1, 1), (5, 6)]:
      ws.max_ngv = max_c
      ws.max_nge = max_e
      ws.geom_corner_loc = wp.full((nworld, max_c), wp.vec3(wp.inf), dtype=wp.vec3, device=d.qpos.device)
      ws.geom_corners = wp.full((nworld, max_c), wp.vec3(wp.inf), dtype=wp.vec3, device=d.qpos.device)
      ws.corner_geom = wp.full((nworld, max_c), -1, dtype=int, device=d.qpos.device)
      ws.geom_edge_loc = wp.full((nworld, max_e, 2), wp.vec3(wp.inf), dtype=wp.vec3, device=d.qpos.device)
      ws.geom_edges = wp.full((nworld, max_e, 2), wp.vec3(wp.inf), dtype=wp.vec3, device=d.qpos.device)
      ws.edge_geom = wp.full((nworld, max_e), -1, dtype=int, device=d.qpos.device)

      d.overflow.zero_()
      ipc.ipc_init_topology(m, d, ws)
      cc.ipc_update_geom_features(m, d, ws)
      ngv_np = ws.ngv.numpy()
      nge_np = ws.nge.numpy()
      overflow_np = d.overflow.numpy()
      geom_corners_np = ws.geom_corners.numpy()
      corner_geom_np = ws.corner_geom.numpy()
      geom_edges_np = ws.geom_edges.numpy()
      edge_geom_np = ws.edge_geom.numpy()
      for w in range(nworld):
        self.assertEqual(int(ngv_np[w]), 10)
        self.assertEqual(int(nge_np[w]), 13)
        self.assertTrue(bool(int(overflow_np[w]) & int(OverflowType.BROADPHASE)))
        self.assertTrue(np.all(np.isfinite(geom_corners_np[w])))
        self.assertTrue(np.all(corner_geom_np[w] >= 0))
        self.assertTrue(np.all(np.isfinite(geom_edges_np[w])))
        self.assertTrue(np.all(edge_geom_np[w] >= 0))


if __name__ == "__main__":
  absltest.main()
