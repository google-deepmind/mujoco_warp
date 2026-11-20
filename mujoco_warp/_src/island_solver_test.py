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
"""Tests for the island Newton solver."""

import contextlib
import dataclasses
import itertools
import warnings
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import ConeType
from mujoco_warp import test_data
from mujoco_warp._src import io
from mujoco_warp._src import island
from mujoco_warp._src import island_solver


def _holding(islands: bool):
  """Unless islands, the sleeping model has too many trees: the world solver holds sleeping DOFs."""
  return contextlib.nullcontext() if islands else mock.patch.object(island, "MAX_NTREE", 0)


def _fixture(dense_nv: int = island.DENSE_NV, merge: bool = True, cone: int = ConeType.PYRAMIDAL):
  """Islands of every kind: single boxes, stacks (every factorization), a weld, an arm, a chain."""
  # the chain (18 DOFs) is a group wider than a tile; the arm's first hinge is past its limit
  mjm, mjd, m, _ = test_data.fixture(
    xml="""
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .1"/>
      <body name="c0" pos="8 0 .03"><freejoint/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
        </body></body></body></body></body></body></body></body></body></body></body></body>
      </body>
        <body name="arm" pos="7 0 1">
          <joint type="hinge" axis="0 1 0" range="-.5 -.1" frictionloss=".5"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
          <body pos=".3 0 0">
            <joint type="hinge" axis="0 1 0" frictionloss=".2"/>
            <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
          </body>
        </body>
        <body name="s0" pos="0 0 .049"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
        <body name="s1" pos="1 0 .049"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
        <replicate count="6" offset="0 0 .099"><body name="k" pos="2 0 .049"><freejoint/><geom type="box" size=".05 .05 .05"/></body></replicate>
        <replicate count="9" offset="0 0 .099"><body name="b" pos="5 0 .049"><freejoint/><geom type="box" size=".05 .05 .05"/></body></replicate>
        <replicate count="12" offset="0 0 .099"><body name="t" pos="6 0 .049"><freejoint/><geom type="box" size=".05 .05 .05"/></body></replicate>
        <body name="fly" pos="4 0 2"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
        <body name="w0" pos="3 0 1"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
        <body name="w1" pos="3.2 0 1"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
      </worldbody>
      <equality><weld body1="w0" body2="w1"/></equality>
    </mujoco>
    """,
    overrides={"opt.cone": cone},
  )
  # the first single box slides (its contacts have a velocity); the second welded box spins (the
  # weld loads every DOF of the pair)
  mjd.qvel[20] = 1.0
  mjd.qvel[-3:] = (1.0, 2.0, 3.0)
  mujoco.mj_forward(mjm, mjd)
  d = mjw.put_data(mjm, mjd)
  m.opt.island_solve = True
  # rows in island order, dense rows for groups up to dense_nv DOFs; no stale Jacobian entry may be
  # used
  d.efc.J.fill_(np.nan)
  island.group_buffers(m, d).J.fill_(np.nan)
  with mock.patch.object(island, "DENSE_NV", dense_nv), mock.patch.object(island, "MERGE", merge):
    mjw.make_constraint(m, d)
  # start away from the solution so every island iterates
  d.qacc_warmstart.zero_()
  return mjm, mjd, m, d


def _assert_Ma(mjm, mjd, d):
  """efc.Ma is M qacc at the solution, as the world solver leaves it."""
  M = np.zeros((mjm.nv, mjm.nv))
  mujoco.mj_fullM(mjm, mjd, M)
  np.testing.assert_allclose(d.efc.Ma.numpy()[0], M @ d.qacc.numpy()[0], rtol=1e-4, atol=1e-4)


def _cone_hessian(m, d, J, rows, dofs, jaref):
  """Curvature of the elliptic contacts in the cone among rows (MuJoCo's cone Hessian)."""
  H = np.zeros((len(dofs), len(dofs)))
  efc_type, efc_id, state, D = (a.numpy()[0] for a in (d.efc.type, d.efc.id, d.efc.state, d.efc.D))
  address, friction, dim = d.contact.efc_address.numpy(), d.contact.friction.numpy(), d.contact.dim.numpy()
  for r in rows:
    if efc_type[r] != mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC:
      continue
    con = efc_id[r]
    if address[con, 0] != r or state[r] != mujoco.mjtConstraintState.mjCNSTRSTATE_CONE:
      continue
    mu = friction[con, 0] * m.opt.impratio_invsqrt.numpy()[0]
    dm = D[r] / (mu * mu * (1.0 + mu * mu))
    tangent = [address[con, k] for k in range(1, dim[con])]
    scale = friction[con, : len(tangent)]
    u = jaref[tangent] * scale
    t = max(np.linalg.norm(u), mujoco.mjMINVAL)
    z0 = mu * J[r, dofs]
    z = scale[:, None] * J[np.ix_(tangent, dofs)]
    p = u @ z / t
    mu_n_over_t = mu * mu * jaref[r] / t
    H += dm * (
      np.outer(z0, z0)
      - mu * (np.outer(z0, p) + np.outer(p, z0))
      + mu_n_over_t * np.outer(p, p)
      + (mu * mu - mu_n_over_t) * z.T @ z
    )
  return H


def _dense_J(d):
  """World 0's Jacobian, including the rows of islands in slots (kept only there)."""
  nefc = d.nefc.numpy()[0]
  rownnz, rowadr = d.efc.J_rownnz.numpy()[0], d.efc.J_rowadr.numpy()[0]
  colind, J = d.efc.J_colind.numpy()[0, 0], d.efc.J.numpy()[0, 0]
  nv = d.qacc.shape[1]
  out = np.zeros((nefc, nv))
  for r in range(nefc):
    out[r, colind[rowadr[r] : rowadr[r] + rownnz[r]]] = J[rowadr[r] : rowadr[r] + rownnz[r]]
  slots = d.__dict__.get("_island_groups")
  if slots is not None:
    io._island_dense_rows(out, d, slots, 0)
  return out


_ROWS_XML = """
<mujoco>
  <worldbody>
    <geom type="plane" size="10 10 .1"/>
    <body pos="0 0 1"><joint name="a" type="hinge" axis="0 1 0" range="-.2 .2" frictionloss=".1"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/></body>
    <body pos="1 0 1"><joint name="b" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/></body>
    <body pos="2 0 1"><joint name="c" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/></body>
    <body pos="3 0 1"><joint name="d" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/></body>
    <body name="e" pos="4 0 1"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    <body name="f" pos="4.2 0 1"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    <body pos="5 0 .049"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    <body pos="6 0 .049"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    <body pos="6 0 .148"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
  </worldbody>
  <tendon>
    <fixed name="ab" limited="true" range="-.01 .01" frictionloss=".2"><joint joint="a" coef="1"/><joint joint="b" coef="-1"/></fixed>
  </tendon>
  <equality>
    <joint joint1="c" joint2="d"/>
    <connect body1="e" body2="f" anchor=".1 0 0"/>
  </equality>
  <keyframe><key qpos=".3 .1 0 0 4 0 1 1 0 0 0 4.2 0 1 1 0 0 0 5 0 .049 1 0 0 0 6 0 .049 1 0 0 0 6 0 .148 1 0 0 0"/></keyframe>
</mujoco>
"""


class IslandSolverTest(parameterized.TestCase):
  def test_rows_in_island_order(self):
    """Island-mode constraint rows: the same rows, grouped by island, and the same islands."""
    mjm, mjd, m, d = test_data.fixture(xml=_ROWS_XML, keyframe=0, overrides={"opt.cone": ConeType.PYRAMIDAL})
    self.assertTrue(m.is_sparse)

    def rows(island_solve):
      m.opt.island_solve = island_solve
      # without slots, so islands can be rediscovered from the sparse rows
      with mock.patch.object(island, "DENSE_NV", 0):
        mjw.make_constraint(m, d)
      nefc = d.nefc.numpy()[0]
      etype, eid = d.efc.type.numpy()[0, :nefc], d.efc.id.numpy()[0, :nefc]
      order = np.lexsort((np.arange(nefc), eid, etype))
      J = _dense_J(d)
      fields = [etype, eid, d.efc.D.numpy()[0, :nefc], d.efc.aref.numpy()[0, :nefc], J]
      return [f[order] for f in fields]

    world = rows(False)
    island_rows = rows(True)
    for name, a, b in zip(("type", "id", "D", "aref", "J"), island_rows, world):
      np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, err_msg=name)

    # same islands as discovery from the finished rows
    tree_island, nisland = d.tree_island.numpy().copy(), d.nisland.numpy().copy()
    island.island(m, d)
    np.testing.assert_array_equal(tree_island, d.tree_island.numpy())
    np.testing.assert_array_equal(nisland, d.nisland.numpy())
    # tendon (a, b), joint equality (c, d), connect (e, f), floor box, stack
    self.assertEqual(nisland[0], 5)

    # each island's rows are contiguous and ordered equality, friction, limit, contact
    nefc = d.nefc.numpy()[0]
    row_island = d.efc.island.numpy()[0, :nefc]
    self.assertTrue(np.all(np.diff(row_island) >= 0))
    rank = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3}  # MuJoCo constraint types to classes
    etype = d.efc.type.numpy()[0, :nefc]
    for i in range(nisland[0]):
      cls = [rank[t] for t in etype[row_island == i]]
      self.assertEqual(cls, sorted(cls))
      self.assertEqual(d.island_ne.numpy()[0, i], cls.count(0))
      self.assertEqual(d.island_nf.numpy()[0, i], cls.count(1))
      self.assertEqual(d.island_nefc.numpy()[0, i], len(cls))

  def test_dense_rows(self):
    """Narrow island groups write their rows only as dense rows of whole tiles, local columns."""
    rows_fixture = test_data.fixture(xml=_ROWS_XML, keyframe=0, overrides={"opt.cone": ConeType.PYRAMIDAL})
    widths = set()
    multi_tree = False
    for merge, (_, _, m, d) in itertools.product((True, False), (rows_fixture, _fixture())):
      m.opt.island_solve = True
      self.enterContext(mock.patch.object(island, "MERGE", merge))

      def rows():
        nefc = d.nefc.numpy()[0]
        etype, eid = d.efc.type.numpy()[0, :nefc], d.efc.id.numpy()[0, :nefc]
        order = np.lexsort((np.arange(nefc), eid, etype))
        return _dense_J(d)[order]

      with mock.patch.object(island, "DENSE_NV", 0):
        mjw.make_constraint(m, d)
      expect = rows()
      # every dense entry must be written (stale values from other steps are not zero)
      island.group_buffers(m, d).J.fill_(np.nan)
      mjw.make_constraint(m, d)
      np.testing.assert_allclose(rows(), expect, atol=1e-6)

      g = island.group_buffers(m, d)
      group_dense, J = g.group_dense.numpy()[0], g.J.numpy()[0]
      nv, rownnz = g.group_nv.numpy()[0], d.efc.J_rownnz.numpy()[0]
      ngroup = g.ngroup.numpy()[0]
      # dense rows for groups up to the dense width, of one tree or several
      idof2dof, idofadr = d.map_idof2dof.numpy()[0], g.group_idofadr.numpy()[0]
      wide = np.where(nv[:ngroup] <= island.dense_width(m), island.dense_width(m), 0)
      np.testing.assert_array_equal(group_dense[:ngroup, 0], wide)
      widths.update(group_dense[:ngroup, 0])
      for i in range(ngroup):
        width, _, nrow, rowadr = group_dense[i]
        if width:
          np.testing.assert_array_equal(rownnz[rowadr : rowadr + nrow], 0)
          np.testing.assert_array_equal(J[rowadr : rowadr + nrow, nv[i] : width], 0.0)
          trees = np.unique(m.dof_treeid.numpy()[idof2dof[idofadr[i] : idofadr[i] + nv[i]]])
          multi_tree |= len(trees) > 1
    # dense rows (the chain; the welded pair, two trees) and sparse rows (the stacks)
    self.assertTrue(multi_tree)
    self.assertIn(0, widths)
    self.assertGreater(max(widths), 0)

  def test_dense_width(self):
    """Dense problems are as wide as the model's largest tree of 17 to 32 DOFs (padded to 4).

    The box falls freely, so the chain is the only group: the island solver's own path is skipped.
    """
    self.enterContext(mock.patch.object(island, "MIN_NV", 0))
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="10 10 .1"/>
          <body name="c0" pos="8 0 .03"><freejoint/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .1 0 0" size=".03"/>
            </body></body></body></body></body></body></body></body></body></body></body></body>
          </body>
          <body pos="0 0 2"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.cone": ConeType.PYRAMIDAL, "opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE},
    )
    # trees: the chain (18 DOFs, padded to 20) and the box (6)
    self.assertEqual(m.island_tree_nv, 20)
    m.opt.island_solve = True
    self.assertTrue(island.island_rows(m))
    mjw.make_constraint(m, d)
    d.qacc_warmstart.zero_()
    # every DOF must be written, the box's too
    d.qacc.fill_(np.nan)
    d.qfrc_constraint.fill_(np.nan)
    d.efc.Ma.fill_(np.nan)
    island_solver.solve(m, d)
    g = island.group_buffers(m, d)
    self.assertEqual(list(g.group_dense.numpy()[0, : g.ngroup.numpy()[0], 0]), [island.dense_width(m)])
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=2e-3)
    np.testing.assert_allclose(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, rtol=1e-3, atol=1e-3)
    _assert_Ma(mjm, mjd, d)

  def test_solve(self):
    """The island solve reaches MuJoCo's solution, with dense, mixed and sparse islands.

    Single trees (boxes, the chain, the arm) take the dense world solver; several trees (stacks, the
    welded pair, merged groups) the block path.
    """
    for (dense_nv, merge), cone in itertools.product(
      ((32, True), (32, False), (16, True), (0, True)), (ConeType.PYRAMIDAL, ConeType.ELLIPTIC)
    ):
      with self.subTest(dense_nv=dense_nv, merge=merge, cone=cone.name):
        mjm, mjd, m, d = _fixture(dense_nv, merge, cone)
        d.efc.Ma.fill_(np.nan)
        island_solver.solve(m, d)
        # the chain's accelerations end within about 1e-3 of MuJoCo's (atomic sums are unordered)
        np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=2e-3)
        np.testing.assert_allclose(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, rtol=1e-3, atol=1e-3)
        _assert_Ma(mjm, mjd, d)
        # from a nonzero start point (M qacc at the start matters)
        mjm, mjd, m, d = _fixture(dense_nv, merge, cone)
        d.qacc_warmstart.assign(0.5 * mjd.qacc[None])
        island_solver.solve(m, d)
        np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=2e-3)

  def test_hessian(self):
    """Each island's H block is M + J'DJ over its quadratic rows, after incremental updates."""
    niter = 0
    for dense_nv, merge, cone in (
      (32, True, ConeType.PYRAMIDAL),
      (32, False, ConeType.PYRAMIDAL),
      (0, True, ConeType.PYRAMIDAL),
      (32, True, ConeType.ELLIPTIC),
      (32, False, ConeType.ELLIPTIC),
      (0, True, ConeType.ELLIPTIC),
    ):
      with self.subTest(dense_nv=dense_nv, merge=merge, cone=cone.name):
        niter = max(niter, self._check_hessian(dense_nv, merge, cone))
    # incremental updates (niter is unwritten for dense groups)
    self.assertGreater(niter, 1)

  def _check_hessian(self, dense_nv, merge, cone):
    mjm, mjd, m, d = _fixture(dense_nv, merge, cone)
    ctx = island_solver.solve(m, d)
    elliptic = cone == ConeType.ELLIPTIC
    # elliptic: H is assembled at the final point (quadratic rows and the cone curvature)
    if elliptic:
      quadratic = d.efc.state.numpy()[0] == mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC

    M = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, mjd, M)
    J = _dense_J(d)
    D = d.efc.D.numpy()[0]
    H, row2dof, row_quad = ctx.H.numpy()[0], ctx.row2dof.numpy()[0], ctx.row_quad.numpy()[0]
    iefc2efc = d.map_iefc2efc.numpy()[0]
    g = island.group_buffers(m, d)
    island_nv, island_nefc, island_iefcadr = (a.numpy()[0] for a in (g.group_nv, g.group_nefc, g.group_iefcadr))
    ntiles = []
    width = g.group_dense.numpy()[0, :, 0]
    state = d.efc.state.numpy()[0]
    ndense = 0
    for i in range(g.ngroup.numpy()[0]):
      # groups with dense rows (single trees) and sparse ones alike: a block of the pool
      ndense += width[i] > 0
      c, tilebase, vbase = ctx.ntile.numpy()[0, i], ctx.tilebase.numpy()[0, i], ctx.vbase.numpy()[0, i]
      ntiles.append(c)
      dofs = row2dof[vbase : vbase + island_nv[i]]
      rows = range(island_iefcadr[i], island_iefcadr[i] + island_nefc[i])
      Hi = M[np.ix_(dofs, dofs)].copy()
      # convergence is measured relative to the island's inertia trace, as in MuJoCo
      np.testing.assert_allclose(ctx.scale.numpy()[0, i], 1.0 / np.trace(Hi), rtol=1e-5)
      for r in rows:
        if quadratic[r] if elliptic else row_quad[r]:
          Jr = J[iefc2efc[r], dofs]
          Hi += D[iefc2efc[r]] * np.outer(Jr, Jr)
      if elliptic:
        Hi += _cone_hessian(m, d, J, rows, dofs, ctx.row_jaref.numpy()[0])
      for a in range(len(dofs)):
        for b in range(a, len(dofs)):
          T = island_solver._TILE
          entry = H[T * (tilebase + (a // T) * c + b // T) + a % T, b % T]
          np.testing.assert_allclose(entry, Hi[a, b], rtol=1e-4, atol=1e-4, err_msg=f"group {i} ({a}, {b})")
    # a block over one tile per side
    self.assertGreater(max(ntiles), 1)
    # merging joins the single trees with their neighbours
    if dense_nv == 32 and not merge:
      self.assertGreater(ndense, 0)
    return ctx.niter.numpy()[0, : g.ngroup.numpy()[0]].max()

  def test_blocked_flips(self):
    """A blocked group whose rows flip is factored again: its factor is that of its Hessian."""
    spheres = "".join(
      f'<body pos="{0.09 * i:.2f} {0.09 * j:.2f} {0.049 + 0.09 * k:.3f}"><freejoint/><geom type="sphere" size=".05"/></body>'
      for i in range(3)
      for j in range(2)
      for k in range(2)
    )
    # without warmstart, the first steps cross state boundaries
    _, _, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option jacobian="sparse" cone="pyramidal" tolerance="1e-12"><flag warmstart="disable"/></option>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          {spheres}
          <body pos="3 3 1"><freejoint/><geom type="sphere" size=".05"/></body>
        </worldbody>
      </mujoco>
      """,
      qvel_noise=1.0,
      nworld=16,
    )
    m.opt.island_solve = True
    mjw.forward(m, d)
    ctx = island_solver.solve(m, d)
    g = island.group_buffers(m, d)
    H, U, niter = ctx.H.numpy(), ctx.U.numpy(), ctx.niter.numpy()
    nblocked = 0
    for w in range(d.nworld):
      for i in range(g.ngroup.numpy()[w]):
        n = g.group_nv.numpy()[w, i]
        if n <= island_solver._ONE_TILE_NV:
          continue
        nblocked += 1
        # U packs the group's factor row-major; H keeps its upper tiles in the pool
        tile = island_solver._TILE
        ntile, tilebase = ctx.ntile.numpy()[w, i], ctx.tilebase.numpy()[w, i]
        size = ntile * tile
        factor = np.triu(U[w].reshape(-1)[tilebase * tile * tile :][: size * size].reshape(size, size)[:n, :n])
        r, c = np.triu_indices(n)
        hess = np.zeros((n, n))
        hess[r, c] = H[w, tile * (tilebase + (r // tile) * ntile + c // tile) + r % tile, c % tile]
        hess = hess + np.triu(hess, 1).T
        np.testing.assert_allclose(factor.T @ factor, hess, atol=1e-5 * np.abs(hess).max(), err_msg=f"world {w}")
    self.assertGreater(nblocked, 0)
    self.assertGreater(niter.max(), 1)

  def test_gradient_scale(self):
    """The stored gradient times grad_scale is the gradient at the final point.

    Islands whose rows keep their states keep their search direction and gradient (scaled), as in
    the world solver's stable-state fast path; the welded pair has only equality rows.
    """
    _, _, m, d = _fixture()
    ctx = island_solver.solve(m, d)
    g = island.group_buffers(m, d)
    nisland = g.ngroup.numpy()[0]
    grad_true = ctx.Ma.numpy()[0] - ctx.qfrc_smooth.numpy()[0] - ctx.frc.numpy()[0]
    grad, scale = ctx.grad.numpy()[0, :, 0], ctx.grad_scale.numpy()[0, :nisland]
    vbase, nv = ctx.vbase.numpy()[0, :nisland], g.group_nv.numpy()[0, :nisland]
    for i in range(nisland):
      rows = slice(vbase[i], vbase[i] + nv[i])
      # the gradient is a difference of force terms: its float32 noise scales with them, not with
      # the (converged, near zero) gradient
      forces = max(np.abs(t[rows]).max() for t in (ctx.Ma.numpy()[0], ctx.qfrc_smooth.numpy()[0], ctx.frc.numpy()[0]))
      atol = 1e-4 + 1e-5 * forces
      np.testing.assert_allclose(scale[i] * grad[rows], grad_true[rows], rtol=1e-3, atol=atol, err_msg=f"island {i}")
    self.assertTrue(np.any(np.abs(scale - 1.0) > 1e-3))

  def test_islands_converge_independently(self):
    """Islands stop at their own iteration; the world reports the slowest."""
    _, _, m, d = _fixture(merge=False)
    ctx = island_solver.solve(m, d)
    g = island.group_buffers(m, d)
    niter = ctx.niter.numpy()[0, : g.ngroup.numpy()[0]]
    self.assertLess(niter.min(), niter.max())
    self.assertGreaterEqual(d.solver_niter.numpy()[0], niter.max())

  def test_nefc_overflow(self):
    """On nefc overflow the islands keep the world solver's rows (the last contacts drop)."""
    mjm, mjd, m, d = _fixture()
    nefc = int(d.nefc.numpy()[0])
    njmax = nefc // 2
    kept = []
    for island_solve in (False, True):
      m.opt.island_solve = island_solve
      # put_data checks njmax against mjd's rows: start from fresh data with the same state
      d = mjw.put_data(mjm, mujoco.MjData(mjm), nconmax=d.naconmax, njmax=njmax)
      d.qpos.assign(mjd.qpos[None])
      d.qvel.assign(mjd.qvel[None])
      mjw.forward(m, d)
      kept.append(np.unique(d.efc.type.numpy()[0, :njmax], return_counts=True))
    self.assertGreater(d.nefc.numpy()[0], njmax)
    for k1, k0 in zip(kept[1], kept[0]):
      np.testing.assert_array_equal(k1, k0)
    g = island.group_buffers(m, d)
    n, ng = d.nisland.numpy()[0], g.ngroup.numpy()[0]
    for adr, nrow in ((d.island_iefcadr, d.island_nefc), (g.group_iefcadr, g.group_nefc)):
      adr, nrow = adr.numpy()[0, : max(n, ng)], nrow.numpy()[0, : max(n, ng)]
      self.assertLessEqual((adr + nrow)[nrow > 0].max(), njmax)
    self.assertTrue(np.isfinite(d.qacc.numpy()).all())

  def test_incomplete_cone(self):
    """A cone cut by nefc overflow is dropped (no force, satisfied), as by the world solver."""
    mjm, mjd, m, d = _fixture(cone=ConeType.ELLIPTIC)
    # njmax ends inside a contact's cone (rows drop in class order, so the same in both solvers)
    nefc = int(d.nefc.numpy()[0])
    efc_type, efc_id = d.efc.type.numpy()[0, :nefc], d.efc.id.numpy()[0, :nefc]
    rows = np.nonzero(efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)[0]
    conid = efc_id[rows[len(rows) // 2]]
    njmax = int(np.nonzero((efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC) & (efc_id == conid))[0][0]) + 2
    force = []
    for island_solve in (False, True):
      m.opt.island_solve = island_solve
      d = mjw.put_data(mjm, mujoco.MjData(mjm), nconmax=d.naconmax, njmax=njmax)
      d.qpos.assign(mjd.qpos[None])
      d.qvel.assign(mjd.qvel[None])
      # a stale force and state, which the dropped rows must not keep
      d.efc.force.fill_(np.nan)
      d.efc.state.fill_(int(mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC))
      mjw.step(m, d)
      t, i, f, st = (a.numpy()[0, :njmax] for a in (d.efc.type, d.efc.id, d.efc.force, d.efc.state))
      cut = (t == mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC) & (i == i[-1])
      force.append(f[cut])
      np.testing.assert_array_equal(st[cut], mujoco.mjtConstraintState.mjCNSTRSTATE_SATISFIED)
    self.assertEqual(len(force[0]), 2)
    np.testing.assert_array_equal(force[0], 0.0)
    np.testing.assert_array_equal(force[1], force[0])

  def test_sleep_islands(self):
    """With sleep, the islands sleep reads are those of the rows the island solver solves."""
    self.enterContext(mock.patch.object(island, "MIN_NV", 0))
    xml = """
    <mujoco>
      <option jacobian="sparse"><flag sleep="enable"/></option>
      <worldbody>
        <geom type="plane" size="5 5 .1"/>
        <body pos="0 0 .09"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
        <body pos="2 0 1">
          <joint name="j0" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
          <body pos=".3 0 0">
            <joint name="j1" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02"/>
          </body>
        </body>
      </worldbody>
      <equality><joint joint1="j0" joint2="j1"/></equality>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    # one island per group: the arm's joint equality gets dense rows, without sparse entries
    with mock.patch.object(island, "MERGE", False):
      partitions = []
      for islands in (False, True):
        with _holding(islands):
          m = mjw.put_model(mjm)
          d = mjw.put_data(mjm, mjd, nconmax=16, njmax=64)
          self.assertEqual(island.island_rows(m), islands)
          mjw.forward(m, d)
          labels = d.tree_island.numpy()[0]
          # islands as sets of trees, independent of their numbering
          partitions.append(sorted(tuple(np.flatnonzero(labels == i)) for i in range(d.nisland.numpy()[0])))
    self.assertEqual(partitions[0], [(0,), (1,)])
    self.assertEqual(partitions[1], partitions[0])

  def test_sleep(self):
    """Sleeping islands are skipped as the world solver holds them: same states, no force."""
    self.enterContext(mock.patch.object(island, "MIN_NV", 0))
    xml = """
    <mujoco>
      <option jacobian="sparse"><flag sleep="enable"/></option>
      <worldbody>
        <geom type="plane" size="5 5 .1"/>
        <body pos="0 0 .1"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
        <body pos="0 0 .35"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
        <body pos="1 0 .1"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
        <body pos="1 0 4"><freejoint/><geom type="sphere" size=".08"/></body>
        <body pos="0 2 1">
          <joint type="hinge" axis="0 1 0" range="-10 10" limited="true" damping=".1"/>
          <geom type="capsule" fromto="0 0 0 .5 0 0" size=".05"/>
        </body>
        <body pos="2 2 .5"><joint name="spin" type="hinge" axis="0 0 1"/><geom type="box" size=".3 .05 .05"/></body>
      </worldbody>
      <actuator><motor joint="spin"/></actuator>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mjd.ctrl[0] = 0.5
    mujoco.mj_forward(mjm, mjd)
    hinge = mjm.jnt_dofadr[4]
    runs = []
    for islands in (False, True):
      with _holding(islands):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nconmax=32, njmax=128)
        self.assertEqual(island.island_rows(m), islands)
        states = []
        for i in range(800):
          # trees fall asleep after the solve: check those asleep before and after the step
          awake_before = d.tree_awake.numpy()[0].copy()
          mjw.step(m, d)
          if i % 10 == 9:
            states.append(d.tree_asleep.numpy()[0].copy())
            if islands:
              awake = d.tree_awake.numpy()[0] | awake_before
              for t in range(mjm.ntree):
                if not awake[t]:
                  dofs = slice(mjm.tree_dofadr[t], mjm.tree_dofadr[t] + mjm.tree_dofnum[t])
                  for a in (d.qacc, d.qfrc_constraint, d.efc.Ma):
                    np.testing.assert_array_equal(a.numpy()[0, dofs], 0.0, err_msg=f"step {i} tree {t}")
              # the limit row of the hinge, which falls asleep at its limit
              nefc = d.nefc.numpy()[0]
              limit = (d.efc.type.numpy()[0, :nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT).nonzero()[0]
              if not awake[mjm.dof_treeid[hinge]] and len(limit):
                np.testing.assert_array_equal(d.efc.force.numpy()[0, limit], 0.0, err_msg=f"step {i}")
        runs.append((np.array(states), d.qpos.numpy()[0]))
    (states0, qpos0), (states1, qpos1) = runs
    asleep0, asleep1 = states0 >= 0, states1 >= 0
    # a ball wakes a sleeping box mid-run; both then fall asleep again (rounding shifts when trees
    # cross the sleep threshold, so compare the settled end)
    self.assertTrue((asleep1[:-1] & ~asleep1[1:]).any())
    np.testing.assert_array_equal(asleep1[-1], asleep0[-1])
    self.assertEqual(asleep1[-1].sum(), mjm.ntree - 1)
    np.testing.assert_allclose(qpos1, qpos0, atol=1e-3)

  @parameterized.parameters(ConeType.PYRAMIDAL, ConeType.ELLIPTIC)
  def test_graph_capture(self, cone):
    """The island solve replays from a graph capture (with side-stream branches), any loop."""
    mjm, mjd, m, _ = _fixture(cone=cone)
    self.assertTrue(island.island_rows(m))
    # one step from a cold start, so the islands iterate (atomics make several steps drift apart)
    mjd.qacc_warmstart[:] = 0.0
    qacc, niter = [], []
    for capture, conditional in ((False, True), (True, True), (True, False)):
      m.opt.graph_conditional = conditional
      d = mjw.put_data(mjm, mjd)
      if capture:
        with wp.ScopedCapture() as cap:
          mjw.step(m, d)
        wp.capture_launch(cap.graph)
      else:
        mjw.step(m, d)
      qacc.append(d.qacc.numpy()[0])
      niter.append(d.solver_niter.numpy()[0])
    self.assertGreater(niter[0], 1)
    for i in (1, 2):
      self.assertEqual(niter[i], niter[0])
      np.testing.assert_allclose(qacc[i], qacc[0], rtol=1e-5, atol=1e-5 * np.abs(qacc[0]).max())

  def test_no_iterations(self):
    """Without iterations, islands keep the warmstart; DOFs outside all islands take qacc_smooth."""
    mjm, mjd, m, d = _fixture()
    mjd.qacc_warmstart[:] = np.linspace(-1.0, 1.0, mjm.nv)
    m.opt.iterations = 0
    # without iterations, qacc_smooth is not skipped
    m.opt.run_qacc_smooth = False
    self.assertTrue(island.island_rows(m))
    d = mjw.put_data(mjm, mjd)
    mjw.forward(m, d)
    free = d.dof_island.numpy()[0] < 0
    self.assertTrue(free.any() and not free.all())
    qacc = d.qacc.numpy()[0]
    np.testing.assert_allclose(qacc[~free], mjd.qacc_warmstart[~free], rtol=1e-6)
    np.testing.assert_allclose(qacc[free], d.qacc_smooth.numpy()[0, free], rtol=1e-6)
    self.assertEqual(d.solver_niter.numpy()[0], 0)

  def test_wide_tree(self):
    """A group with a tree over two tiles wide (blocked) matches the world solver."""
    n = 65
    chain = "".join(
      f'<body pos="0 0 .05"><joint type="hinge" axis="0 1 0"/><geom type="capsule" size=".02 .025"/>' for _ in range(n)
    )
    xml = f"""
    <mujoco>
      <option jacobian="sparse"/>
      <worldbody>
        <body pos="0 0 1"><joint name="jd" type="hinge" axis="0 1 0"/><geom type="sphere" size=".05"/></body>
        <body name="c" pos="1 0 2">
          <joint name="jc" type="hinge" axis="0 1 0"/><geom type="capsule" size=".02 .025"/>{chain}{"</body>" * n}
        </body>
      </worldbody>
      <equality><joint joint1="jd" joint2="jc"/></equality>
    </mujoco>
    """
    qacc = []
    for island_solve in (False, True):
      _, _, m, d = test_data.fixture(xml=xml, qvel_noise=0.1)
      m.opt.island_solve = island_solve
      d.qacc.fill_(np.nan)
      mjw.forward(m, d)
      qacc.append(d.qacc.numpy()[0])
    self.assertEqual(m.nv, n + 2)
    self.assertTrue(island.rows_in_island_order(d))
    # a stiff chain: both solves stop at the tolerance
    np.testing.assert_allclose(qacc[1], qacc[0], rtol=1e-3, atol=1e-3 * np.abs(qacc[0]).max())

  def test_put_data_rows(self):
    """put_data loads MuJoCo's rows, in type order: the world solver solves them."""
    _, mjd, m, d = test_data.fixture(xml=_ROWS_XML, keyframe=0)
    m.opt.island_solve = True
    self.assertTrue(island.island_rows(m))
    d.qacc.fill_(np.nan)
    mjw.solve(m, d)
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-4, atol=1e-4)

  def test_reused_context(self):
    """A solve reads nothing stale from the previous step's context (cached per Data)."""
    for cone in (ConeType.PYRAMIDAL, ConeType.ELLIPTIC):
      mjm, mjd, m, d = _fixture(cone=cone)
      island_solver.solve(m, d)
      qacc = d.qacc.numpy()[0].copy()
      ctx = d.__dict__["_island_context"][1]
      for f in dataclasses.fields(ctx):
        arr = getattr(ctx, f.name)
        if isinstance(arr, wp.array) and arr.size and arr.dtype == wp.float32:
          arr.fill_(np.nan)
      d.qacc_warmstart.zero_()
      island_solver.solve(m, d)
      np.testing.assert_allclose(d.qacc.numpy()[0], qacc, rtol=1e-4, atol=1e-4 * np.abs(qacc).max())

  def test_worlds(self):
    """The worlds of a batch solve alike."""
    mjm, mjd, m, _ = _fixture()
    mjd.qacc_warmstart[:] = 0.0
    d = mjw.put_data(mjm, mjd, nworld=3)
    mjw.step(m, d)
    qacc = d.qacc.numpy()
    np.testing.assert_allclose(qacc[1:], qacc[:1].repeat(2, axis=0), rtol=1e-4, atol=1e-4 * np.abs(qacc[0]).max())

  def test_run_qacc_smooth(self):
    """Without qacc_smooth, the trees outside all islands are groups without rows, solved alike."""
    mjm, mjd, m, _ = _fixture()
    qacc = []
    for run_qacc_smooth in (True, False):
      m.opt.run_qacc_smooth = run_qacc_smooth
      self.assertTrue(island.island_rows(m))
      d = mjw.put_data(mjm, mjd)
      d.qacc_smooth.fill_(np.nan)
      mjw.forward(m, d)
      self.assertEqual(np.isnan(d.qacc_smooth.numpy()).all(), not run_qacc_smooth)
      qacc.append(d.qacc.numpy()[0])
    self.assertTrue((d.dof_island.numpy()[0] < 0).any())
    np.testing.assert_allclose(qacc[1], qacc[0], rtol=1e-5, atol=1e-5 * np.abs(qacc[0]).max())

  # small worlds (one group), a dense Jacobian, deterministic rows, too many trees
  @parameterized.parameters(
    (2, "sparse", 0, island.MAX_NTREE),
    (8, "dense", 0, island.MAX_NTREE),
    (8, "sparse", 1, island.MAX_NTREE),
    (8, "sparse", 0, 4),
  )
  def test_fallback(self, nbox, jacobian, deterministic, max_ntree):
    """Models the island solver does not apply to are solved by the world solver."""
    self.enterContext(mock.patch.object(island, "MAX_NTREE", max_ntree))
    results = []
    for island_solve in (True, False):
      _, _, m, d = test_data.fixture(
        xml=f"""
        <mujoco>
          <worldbody>
            <geom type="plane" size="10 10 .1"/>
            <replicate count="{nbox}" offset="1 0 0">
              <body pos="0 0 .099"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
            </replicate>
          </worldbody>
        </mujoco>
        """,
        overrides={"opt.jacobian": jacobian},
      )
      m.opt.island_solve = island_solve
      if deterministic:
        self.assertEqual(island.island_rows(m), island_solve)
        m.opt.deterministic = int(mjw.DeterminismType.CONSTRAINT)
      # only unsupported features warn
      with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        self.assertFalse(island.island_rows(m))
      self.assertEqual(len(caught), int(island_solve and 2 < nbox <= max_ntree), [str(w.message) for w in caught])
      mjw.step(m, d)
      results.append(d.qacc.numpy()[0])
    # the same solve up to the order of atomic sums
    np.testing.assert_allclose(results[0], results[1], rtol=1e-5, atol=1e-6)

  def test_njmax_nnz_overflow(self):
    """Island rows past their sparse storage stay empty: no stale layout, as rows in world order."""
    self.enterContext(mock.patch.object(island, "MIN_NV", 0))
    self.enterContext(mock.patch.object(island, "DENSE_NV", 0))
    xml = """
    <mujoco>
      <option jacobian="sparse"/>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body pos="0 0 1"><joint range="-.1 .1"/><geom size=".05" contype="0" conaffinity="0"/></body>
        <replicate count="4" offset="1 0 0"><body pos="1 0 1"><freejoint/><geom size=".1"/></body></replicate>
      </worldbody>
      <keyframe><key qpos=".5  1 0 .09 1 0 0 0  2 0 .09 1 0 0 0  3 0 .09 1 0 0 0  4 0 .09 1 0 0 0"/></keyframe>
    </mujoco>
    """
    for cone, overflow in itertools.product((ConeType.PYRAMIDAL, ConeType.ELLIPTIC), (True, False)):
      mjm, mjd, m, d = test_data.fixture(
        xml=xml, overrides={"opt.cone": cone, "opt.island_solve": True}, njmax_nnz=30 if overflow else None
      )
      self.assertTrue(island.island_rows(m))
      mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
      d.qpos.assign(mjd.qpos[None])
      # stale layout from a previous step: rows covering the whole storage
      d.efc.J_rowadr.fill_(0)
      d.efc.J_rownnz.fill_(mjm.nv)
      d.overflow.zero_()
      mjw.fwd_position(m, d)

      nrow = min(d.nefc.numpy()[0], d.njmax)
      adr, nnz = d.efc.J_rowadr.numpy()[0, :nrow], d.efc.J_rownnz.numpy()[0, :nrow]
      self.assertEqual(bool(d.overflow.numpy()[0] & int(mjw.OverflowType.NJMAX_NNZ)), overflow)
      np.testing.assert_array_less(-1, adr)
      np.testing.assert_array_less(adr + nnz, d.njmax_nnz + 1)
      kept = nnz > 0
      order = np.argsort(adr[kept])
      ends, starts = (adr + nnz)[kept][order], adr[kept][order]
      np.testing.assert_array_less(ends[:-1], starts[1:] + 1, err_msg="overlapping rows")
      contact = d.efc.type.numpy()[0, :nrow] >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
      # contacts keep all their rows: those without storage are empty
      self.assertEqual(set(nnz[contact]), {0, 6} if overflow else {6})
      colind = d.efc.J_colind.numpy()[0, 0]
      for a, n in zip(adr, nnz):
        self.assertTrue(np.all((colind[a : a + n] >= 0) & (colind[a : a + n] < mjm.nv)))

  def test_noslip_solves_worlds(self):
    """Noslip reads the Jacobian rows dense groups leave empty: it keeps the world solver."""
    results = []
    for island_solve in (True, False):
      mjm, mjd, m, d = test_data.fixture(xml=_ROWS_XML, keyframe=0, overrides={"opt.noslip_iterations": 5})
      m.opt.island_solve = island_solve
      self.assertFalse(island.island_rows(m))
      mjw.step(m, d)
      results.append(d.qacc.numpy()[0])
    np.testing.assert_allclose(results[0], results[1], rtol=1e-4, atol=1e-3)

  def test_dense_island_groups(self):
    """Islands of up to the dense width merge only up to it: their groups keep dense rows."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="10 10 .1"/>
          <replicate count="3" offset="1 0 0">
            <body pos="0 0 .049"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
          </replicate>
          <body pos="0 2 1"><joint type="hinge"/><geom size=".05"/>
            <body pos=".1 0 0"><joint type="hinge"/><geom size=".05"/></body>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.island_solve": True, "opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE},
    )
    self.enterContext(mock.patch.object(island, "MIN_NV", 0))
    mjw.make_constraint(m, d)
    g = island.group_buffers(m, d)
    ngroup = g.ngroup.numpy()[0]
    # three resting boxes (6 DOFs each, dense width 16): a pair and a single, both with dense rows
    self.assertEqual(island.dense_width(m), 16)
    self.assertEqual(list(g.group_nv.numpy()[0, :ngroup]), [12, 6])
    self.assertEqual(list(g.group_dense.numpy()[0, :ngroup, 0]), [16, 16])

  def test_island_groups(self):
    """Consecutive islands share a group up to MERGE_NV DOFs; the solution is unchanged."""
    mjm, mjd, m, d = _fixture()
    g = island.group_buffers(m, d)
    nv, group = d.island_nv.numpy()[0], g.island_group.numpy()[0]
    nisland, ngroup = d.nisland.numpy()[0], g.ngroup.numpy()[0]
    self.assertLess(ngroup, nisland)
    group_nv = g.group_nv.numpy()[0, :ngroup]
    for k in range(ngroup):
      members = nv[:nisland][group[:nisland] == k]
      self.assertEqual(members.sum(), group_nv[k])
      if len(members) > 1:
        # islands that could have dense rows keep them
        wide = island.dense_width(m)
        self.assertLessEqual(group_nv[k], wide if members.max() <= wide else island.MERGE_NV)
    # islands with equality or friction rows join groups too (the arm, the welded pair)
    ne_nf = d.island_ne.numpy()[0, :nisland] + d.island_nf.numpy()[0, :nisland]
    first = np.r_[True, group[1:nisland] != group[: nisland - 1]]
    self.assertTrue(np.any((ne_nf > 0) & ~first))
    island_solver.solve(m, d)
    np.testing.assert_allclose(d.qacc.numpy()[0], mjd.qacc, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  absltest.main()
