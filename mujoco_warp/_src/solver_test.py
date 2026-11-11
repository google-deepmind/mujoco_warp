# Copyright 2025 The Newton Developers
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

"""Tests for solver functions."""

import copy
import dataclasses

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import ConeType
from mujoco_warp import SolverType
from mujoco_warp import test_data
from mujoco_warp._src import island
from mujoco_warp._src import solver
from mujoco_warp._src import types

# tolerance for difference between MuJoCo and MJWarp solver calculations - mostly
# due to float precision
_TOLERANCE = 5e-3

# tolerance for noslip comparisons against MuJoCo
_NOSLIP_TOLERANCE = 1e-4


def _assert_eq(a, b, name, tol=_TOLERANCE * 20):
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _set_inertia_scale(m, d, ctx):
  """Set the inverse trace of M, as init_context does, for tests which skip it."""
  wp.launch(solver._inertia_scale, dim=d.nworld, inputs=[m.nv, m.M_rownnz, m.M_rowadr, d.M], outputs=[ctx.inertia_scale])


def _noslip_model(mjm, noslip_iterations):
  mjm = copy.deepcopy(mjm)
  mjm.opt.noslip_iterations = noslip_iterations
  return mjm


def _perturb_worlds(mjm, mjd, d, nworld, dqvel):
  """Returns MuJoCo data for each world, world 1 has perturbed velocities."""
  mjds = [mjd]
  if nworld == 2:
    mjd1 = copy.copy(mjd)
    mjd1.qvel[:] += dqvel
    mujoco.mj_forward(mjm, mjd1)
    qvel = d.qvel.numpy()
    qvel[1] = mjd1.qvel
    d.qvel.assign(qvel)
    mjds.append(mjd1)
  return mjds


def _noslip_states(efc_type, efc_id, efc_force, efc_state, contact_friction):
  """Returns sorted constraint states, excluding elliptic contacts on the cone boundary.

  The noslip solver projects elliptic contact forces onto the cone, so the state of these
  contacts (SATISFIED or CONE) depends on rounding.
  """
  states = []
  rowid = 0
  while rowid < efc_type.size:
    if efc_type[rowid] != mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC:
      states.append(efc_state[rowid])
      rowid += 1
      continue
    friction = contact_friction[efc_id[rowid]]
    dim = 1
    while rowid + dim < efc_type.size and efc_id[rowid + dim] == efc_id[rowid] and efc_type[rowid + dim] == efc_type[rowid]:
      dim += 1
    force = efc_force[rowid : rowid + dim]
    T = np.linalg.norm(force[1:] / friction[: dim - 1])
    if abs(force[0] - T) > 1e-4 * max(1.0, abs(force[0])):
      states.extend(efc_state[rowid : rowid + dim])
    rowid += dim
  return np.sort(states)


def _mul_m(mjm, mjd, vec):
  res = np.zeros(mjm.nv)
  mujoco.mj_mulM(mjm, mjd, res, vec)
  return res


def _qcqp_reference(A, b, d, r):
  """Port of mju_QCQP: min 0.5 x' A x + x' b s.t. sum (x_i / d_i)^2 <= r^2."""
  n = b.size
  A = A * np.outer(d, d)
  b = b * d
  res = np.zeros(n)
  la = 0.0
  for _ in range(20):
    Ala = A + la * np.eye(n)

    # Cholesky with 1e-10 rank threshold
    L = np.zeros((n, n))
    for j in range(n):
      diag = Ala[j, j] - L[j, :j] @ L[j, :j]
      if diag < 1e-10:
        return np.zeros(n), False
      L[j, j] = np.sqrt(diag)
      for i in range(j + 1, n):
        L[i, j] = (Ala[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]

    res = -np.linalg.solve(Ala, b)
    val = res @ res - r * r
    if val < 1e-10:
      break
    deriv = -2.0 * res @ np.linalg.solve(Ala, res)
    delta = -val / deriv
    if delta < 1e-10:
      break
    la += delta

  return res * d, la != 0.0


@wp.kernel
def _qcqp_kernel(
  # In:
  n: int,
  A_in: wp.array[types.mat55],
  b_in: wp.array[types.vec5],
  d_in: wp.array[types.vec5],
  r_in: wp.array[float],
  # Out:
  res2_out: wp.array[types.vec5],
  active2_out: wp.array[int],
  res3_out: wp.array[types.vec5],
  active3_out: wp.array[int],
  res_out: wp.array[types.vec5],
  active_out: wp.array[int],
):
  tid = wp.tid()
  A = A_in[tid]
  b = b_in[tid]
  d = d_in[tid]
  r = r_in[tid]
  if n == 2:
    res2, active2 = solver._qcqp2(A, b, d, r)
    res2_out[tid] = res2
    active2_out[tid] = int(active2)
  if n == 3:
    res3, active3 = solver._qcqp3(A, b, d, r)
    res3_out[tid] = res3
    active3_out[tid] = int(active3)
  res, active = solver._qcqp(A, b, d, r, n)
  res_out[tid] = res
  active_out[tid] = int(active)


def _elliptic_dense_hessian_reference(m, d, ctx):
  contact_dim = d.contact.dim.numpy()
  contact_dist = d.contact.dist.numpy()
  contact_efc_address = d.contact.efc_address.numpy()
  contact_friction = d.contact.friction.numpy()
  contact_includemargin = d.contact.includemargin.numpy()
  contact_worldid = d.contact.worldid.numpy()
  efc_D = d.efc.D.numpy()
  efc_J = d.efc.J.numpy()
  efc_state = d.efc.state.numpy()
  impratio_invsqrt = m.opt.impratio_invsqrt.numpy()
  jaref = ctx.Jaref.numpy()
  h = np.zeros((d.nworld, m.nv, m.nv), dtype=np.float64)

  for conid in range(d.nacon.numpy()[0]):
    if contact_dist[conid] - contact_includemargin[conid] >= 0.0:
      continue
    worldid = contact_worldid[conid]
    efcid0 = contact_efc_address[conid, 0]
    if efcid0 < 0 or efc_state[worldid, efcid0] != types.ConstraintState.CONE.value:
      continue

    condim = contact_dim[conid]
    friction = contact_friction[conid]
    mu = friction[0] * impratio_invsqrt[worldid % impratio_invsqrt.shape[0]]
    mu2 = mu * mu
    dm = efc_D[worldid, efcid0] / (mu2 * (1.0 + mu2))
    if dm == 0.0:
      continue

    u = np.zeros(condim)
    scale = np.zeros(condim)
    u[0] = jaref[worldid, efcid0] * mu
    scale[0] = mu
    jac = np.zeros((condim, m.nv))
    for dim in range(condim):
      efcid = contact_efc_address[conid, dim]
      if efcid < 0:
        continue
      if dim > 0:
        scale[dim] = friction[dim - 1]
        u[dim] = jaref[worldid, efcid] * scale[dim]
      jac[dim] = efc_J[worldid, efcid, : m.nv]

    t = max(np.linalg.norm(u[1:]), types.MJ_MINVAL)
    inv_t = 1.0 / t
    mu_n_over_t = mu * u[0] * inv_t
    cone = np.zeros((condim, condim))
    for dim1 in range(condim):
      for dim2 in range(dim1 + 1):
        if dim1 == 0 and dim2 == 0:
          value = 1.0
        elif dim2 == 0:
          value = -mu * (u[dim1] * inv_t)
        else:
          value = mu_n_over_t * (u[dim1] * inv_t) * (u[dim2] * inv_t)
          if dim1 == dim2:
            value += mu2 - mu_n_over_t
        value *= dm * scale[dim1] * scale[dim2]
        cone[dim1, dim2] = value
        cone[dim2, dim1] = value

    h[worldid] += np.triu(jac.T @ cone @ jac)

  return h


def _elliptic_sparse_hessian_reference(m, d, ctx):
  contact_dim = d.contact.dim.numpy()
  contact_efc_address = d.contact.efc_address.numpy()
  contact_friction = d.contact.friction.numpy()
  contact_worldid = d.contact.worldid.numpy()
  efc_D = d.efc.D.numpy()
  efc_J = d.efc.J.numpy()
  efc_J_colind = d.efc.J_colind.numpy()
  efc_J_rowadr = d.efc.J_rowadr.numpy()
  efc_J_rownnz = d.efc.J_rownnz.numpy()
  efc_state = d.efc.state.numpy()
  impratio_invsqrt = m.opt.impratio_invsqrt.numpy()
  jaref = ctx.Jaref.numpy()
  h = np.zeros((d.nworld, m.nv, m.nv), dtype=np.float64)

  for conid in range(d.nacon.numpy()[0]):
    worldid = contact_worldid[conid]
    efcid0 = contact_efc_address[conid, 0]
    if efcid0 < 0 or efc_state[worldid, efcid0] != types.ConstraintState.CONE.value:
      continue

    condim = contact_dim[conid]
    friction = contact_friction[conid]
    mu = friction[0] * impratio_invsqrt[worldid % impratio_invsqrt.shape[0]]
    mu2 = mu * mu
    dm = efc_D[worldid, efcid0] / (mu2 * (1.0 + mu2))
    if dm == 0.0:
      continue

    u = np.zeros(condim)
    scale = np.zeros(condim)
    u[0] = jaref[worldid, efcid0] * mu
    scale[0] = mu
    jac = np.zeros((condim, m.nv))
    for dim in range(condim):
      efcid = contact_efc_address[conid, dim]
      if efcid < 0:
        continue
      if dim > 0:
        scale[dim] = friction[dim - 1]
        u[dim] = jaref[worldid, efcid] * scale[dim]
      rowadr = efc_J_rowadr[worldid, efcid]
      rownnz = efc_J_rownnz[worldid, efcid]
      cols = efc_J_colind[worldid, 0, rowadr : rowadr + rownnz]
      jac[dim, cols] = efc_J[worldid, 0, rowadr : rowadr + rownnz]

    t = max(np.linalg.norm(u[1:]), types.MJ_MINVAL)
    inv_t = 1.0 / t
    mu_n_over_t = mu * u[0] * inv_t
    cone = np.zeros((condim, condim))
    for dim1 in range(condim):
      for dim2 in range(dim1 + 1):
        if dim1 == 0 and dim2 == 0:
          value = 1.0
        elif dim2 == 0:
          value = -mu * (u[dim1] * inv_t)
        else:
          value = mu_n_over_t * (u[dim1] * inv_t) * (u[dim2] * inv_t)
          if dim1 == dim2:
            value += mu2 - mu_n_over_t
        value *= dm * scale[dim1] * scale[dim2]
        cone[dim1, dim2] = value
        cone[dim2, dim1] = value

    h[worldid] += np.triu(jac.T @ cone @ jac)

  return h


@wp.kernel
def _eval_elliptic_delta(
  # In:
  alpha: float,
  d_values: wp.array[float],
  frictionloss_values: wp.array[float],
  quad: wp.vec3,
  quad1: wp.vec3,
  quad2: wp.vec3,
  friction: types.vec5,
  # Out:
  result_out: wp.array[wp.vec3],
):
  result_out[0] = solver._compute_efc_eval_pt_elliptic(
    0,
    alpha,
    0,
    0,
    1.0,
    int(types.ConstraintType.CONTACT_ELLIPTIC),
    d_values,
    frictionloss_values,
    0.0,
    0.0,
    quad,
    friction,
    0,
    quad1,
    quad2,
  )


def _sparse_groups_fixture(nworld):
  """Hand-made JTDAJ groups over sparse rows, with the list of groups that each dof should get.

  Head rows and dofs are not in order, one support is wider than a block, some groups have rows
  that do not follow each other in the sparse arrays, dof 39 is in every group of world 0 and dof
  0 in none, and row 0 is a dof friction row outside the groups. The second world has no group
  and the third reports more groups than the capacity, which is more than a word of bits.
  """
  nv, njmax, ngroup_max = 40, 80, 36
  rng = np.random.default_rng(nworld)
  # (rows, width of the support, rows follow each other) per group
  world_groups = (
    [(1, 3, True), (4, 35, True), (2, 5, False), (6, 8, True), (3, 4, False)],
    [],
    [(int(rng.integers(1, 3)), int(rng.integers(1, 7)), bool(rng.integers(2))) for _ in range(ngroup_max)],
  )[:nworld]

  nnz = 900
  res = dict(
    nv=nv,
    jtdaj_adr=np.full((nworld, ngroup_max), -1, dtype=np.int32),
    jtdaj_nrow=np.full((nworld, ngroup_max), -1, dtype=np.int32),
    nblock=np.zeros(nworld, dtype=np.int32),
    rownnz=np.zeros((nworld, njmax), dtype=np.int32),
    rowadr=np.zeros((nworld, njmax), dtype=np.int32),
    colind=np.zeros((nworld, 1, nnz), dtype=np.int32),
    nf=np.ones(nworld, dtype=np.int32),
    nefc=np.zeros(nworld, dtype=np.int32),
    frictionadr=np.full((nworld, nv), -1, dtype=np.int32),
    adr=np.zeros((nworld, nv + 1), dtype=np.int32),
    entry=np.full((nworld, nnz, 3), -5, dtype=np.int32),
    group=np.full((nworld, ngroup_max, 4), -5, dtype=np.int32),
    slot_adr=np.full((nworld, ngroup_max + 1), -5, dtype=np.int32),
  )
  for worldid, groups in enumerate(world_groups):
    res["nblock"][worldid] = len(groups) + (3 if len(groups) == ngroup_max else 0)
    # row 0: friction of dof 5
    res["frictionadr"][worldid, 5] = 0
    res["rownnz"][worldid, 0] = 1
    res["colind"][worldid, 0, 0] = 5
    row, adr = 1, 1
    lists = [[] for _ in range(nv)]
    entries = {}
    # rows are handed out in another order than the groups
    for groupid in rng.permutation(len(groups)):
      nrow, width, contiguous = groups[groupid]
      support = rng.permutation(np.arange(1, nv - 1))[:width]
      if worldid == 0:
        support = rng.permutation(np.append(support, nv - 1))
      res["jtdaj_adr"][worldid, groupid] = row
      res["jtdaj_nrow"][worldid, groupid] = nrow
      for member in range(nrow):
        res["rownnz"][worldid, row + member] = len(support)
        res["rowadr"][worldid, row + member] = adr + member * (len(support) + (0 if contiguous else 2))
        res["colind"][worldid, 0, res["rowadr"][worldid, row + member] :][: len(support)] = support
      # a single row always follows itself
      sign = 1 if contiguous or nrow == 1 else -1
      res["group"][worldid, groupid] = [adr, len(support), row, sign * nrow]
      # the stride in the low 16 bits and the signed number of rows above them
      code = len(support) + 65536 * sign * nrow
      entries[groupid] = [(dofid, [adr + i, row, code]) for i, dofid in enumerate(support)]
      adr += nrow * (len(support) + 2)
      row += nrow
    assert adr <= nnz and row <= njmax
    res["nefc"][worldid] = row
    slot = 0
    for groupid in range(len(groups)):
      for dofid, entry in entries[groupid]:
        lists[dofid].append(entry)
      # a slot per element of the upper triangle of the block of the group
      res["slot_adr"][worldid, groupid] = slot
      slot += len(entries[groupid]) * (len(entries[groupid]) + 1) // 2
    # the end of the last group, and the number of slots after the group capacity
    res["slot_adr"][worldid, len(groups)] = slot
    res["slot_adr"][worldid, ngroup_max] = slot
    res["adr"][worldid, 1:] = np.cumsum([len(dof_list) for dof_list in lists])
    total = res["adr"][worldid, nv]
    if total:
      res["entry"][worldid, :total] = [entry for dof_list in lists for entry in dof_list]
  return res


class SolverTest(parameterized.TestCase):
  def test_newton_decrement_termination(self):
    """Newton stops when only its local model predicts convergence."""
    solver_niter = wp.zeros(1, dtype=int)
    nsolving = wp.ones(1, dtype=int)
    done = wp.zeros(1, dtype=bool)
    overflow = wp.zeros(1, dtype=int)

    wp.launch(
      solver._solve_done(types.OverflowType.NONE),
      dim=1,
      inputs=[
        wp.array([1.0e-6], dtype=float),
        2,
        wp.array([1.0], dtype=float),
        wp.array([4.0e-12], dtype=float),
        wp.array([1.0e-6], dtype=float),
        wp.array([2.0e-6], dtype=float),
        wp.array([1.0], dtype=float),
        done,
      ],
      outputs=[solver_niter, overflow, nsolving, done],
    )

    self.assertTrue(done.numpy()[0])
    self.assertEqual(solver_niter.numpy()[0], 1)
    self.assertEqual(nsolving.numpy()[0], 0)
    self.assertEqual(overflow.numpy()[0], 0)

  def test_solve_done_iterations_overflow(self):
    """Newton records overflow when iteration limit reached without convergence."""
    solver_niter = wp.zeros(1, dtype=int)
    nsolving = wp.ones(1, dtype=int)
    done = wp.zeros(1, dtype=bool)
    overflow = wp.zeros(1, dtype=int)

    wp.launch(
      solver._solve_done(types.OverflowType.NONE),
      dim=1,
      inputs=[
        wp.array([1.0e-6], dtype=float),
        1,
        wp.array([1.0], dtype=float),
        wp.array([1.0], dtype=float),
        wp.array([1.0], dtype=float),
        wp.array([1.0], dtype=float),
        wp.array([1.0], dtype=float),
        done,
      ],
      outputs=[solver_niter, overflow, nsolving, done],
    )

    self.assertTrue(done.numpy()[0])
    self.assertEqual(solver_niter.numpy()[0], 1)
    self.assertEqual(nsolving.numpy()[0], 0)
    self.assertTrue(overflow.numpy()[0] & types.OverflowType.ITERATIONS)

  def test_solve_done_zero_alpha_early_out(self):
    """Newton terminates immediately without overflow when linesearch alpha == 0."""
    solver_niter = wp.zeros(1, dtype=int)
    nsolving = wp.ones(1, dtype=int)
    done = wp.zeros(1, dtype=bool)
    overflow = wp.zeros(1, dtype=int)

    wp.launch(
      solver._solve_done(types.OverflowType.NONE),
      dim=1,
      inputs=[
        wp.array([1.0e-6], dtype=float),
        100,
        wp.array([1.0], dtype=float),
        wp.array([1.0], dtype=float),
        wp.array([1.0], dtype=float),
        wp.array([0.0], dtype=float),
        wp.array([0.0], dtype=float),
        done,
      ],
      outputs=[solver_niter, overflow, nsolving, done],
    )

    self.assertTrue(done.numpy()[0])
    self.assertEqual(solver_niter.numpy()[0], 1)
    self.assertEqual(nsolving.numpy()[0], 0)
    self.assertEqual(overflow.numpy()[0], 0)

  def test_solve_cg_finalize_iterations_overflow(self):
    """CG records overflow when iteration limit reached without convergence."""
    solver_niter = wp.zeros(1, dtype=int)
    nsolving = wp.ones(1, dtype=int)
    done = wp.zeros(1, dtype=bool)
    overflow = wp.zeros(1, dtype=int)
    beta = wp.zeros(1, dtype=float)

    wp.launch(
      solver._solve_beta_finalize_tiled(types.OverflowType.NONE),
      dim=(1, 32),
      inputs=[
        1,
        wp.array([1.0e-6], dtype=float),
        1,
        wp.array([1.0], dtype=float),
        wp.zeros((1, 1), dtype=float),
        wp.zeros((1, 1), dtype=float),
        wp.zeros((1, 1), dtype=float),
        wp.zeros((1, 1), dtype=float),
        wp.array([1.0], dtype=float),
        wp.array([1.0], dtype=float),
        wp.array([1.0], dtype=float),
        done,
      ],
      outputs=[solver_niter, overflow, beta, nsolving, done],
    )

    self.assertTrue(done.numpy()[0])
    self.assertEqual(solver_niter.numpy()[0], 1)
    self.assertEqual(nsolving.numpy()[0], 0)
    self.assertTrue(overflow.numpy()[0] & types.OverflowType.ITERATIONS)

  def test_solve_cg_finalize_zero_alpha_early_out(self):
    """CG terminates immediately without overflow when linesearch alpha == 0."""
    solver_niter = wp.zeros(1, dtype=int)
    nsolving = wp.ones(1, dtype=int)
    done = wp.zeros(1, dtype=bool)
    overflow = wp.zeros(1, dtype=int)
    beta = wp.zeros(1, dtype=float)

    wp.launch(
      solver._solve_beta_finalize_tiled(types.OverflowType.NONE),
      dim=(1, 32),
      inputs=[
        1,
        wp.array([1.0e-6], dtype=float),
        100,
        wp.array([1.0], dtype=float),
        wp.zeros((1, 1), dtype=float),
        wp.zeros((1, 1), dtype=float),
        wp.zeros((1, 1), dtype=float),
        wp.zeros((1, 1), dtype=float),
        wp.array([1.0], dtype=float),
        wp.array([1.0], dtype=float),
        wp.array([0.0], dtype=float),
        done,
      ],
      outputs=[solver_niter, overflow, beta, nsolving, done],
    )

    self.assertTrue(done.numpy()[0])
    self.assertEqual(solver_niter.numpy()[0], 1)
    self.assertEqual(nsolving.numpy()[0], 0)
    self.assertEqual(overflow.numpy()[0], 0)

  # Transition cases use powers of two so the exact delta is below the absolute-cost ulp.
  @parameterized.named_parameters(
    ("middle_zone", 1.0e-4, (0.0, 0.0, 0.0), (0.5, 1.0e-4, 1.0), (0.0, 0.0, 1.0e8), (-0.5, -5000.0, 1.0)),
    (
      "quadratic_zone",
      1.0e-4,
      (1.25e7, -5000.0, 0.5),
      (-2.0, 0.0, 1.0),
      (0.0, 0.0, 1.0),
      (-0.5, -5000.0, 1.0),
    ),
    (
      "cone_to_quadratic",
      2.44140625e-4,
      (33546241.0, -8192.0, 33554432.0),
      (-0.999755859375, -1.0, 0.0),
      (-1.0, 1.0, 16777216.0),
      (0.5, 8192.0, 67108864.0),
    ),
    (
      "quadratic_to_cone",
      2.44140625e-4,
      (33546241.0, -8192.0, 33554432.0),
      (-1.0, 1.0, 0.0),
      (0.999755859375, 1.0, 16777216.0),
      (-0.5, 0.0, 0.0),
    ),
    (
      "satisfied_to_quadratic",
      2.44140625e-4,
      (1.0, -16384.0, 67108864.0),
      (2.44140625e-4, -2.0, 0.0),
      (0.0, 0.0, 16777216.0),
      (1.0, 16384.0, 134217728.0),
    ),
    (
      "tangential_cancellation",
      1.0,
      (0.0, 0.0, 0.0),
      (0.0, 0.0, 2.0**-26),
      (-1.0, 1.0, 1.0),
      (-0.5, 0.0, 1.0),
    ),
  )
  def test_elliptic_shifted_cost_preserves_small_delta(self, alpha, quad, quad1, quad2, expected):
    """Elliptic cost deltas remain visible below the absolute-cost ulp."""
    result = wp.empty(1, dtype=wp.vec3)
    wp.launch(
      _eval_elliptic_delta,
      dim=1,
      inputs=[
        alpha,
        wp.zeros(1, dtype=float),
        wp.zeros(1, dtype=float),
        wp.vec3(*quad),
        wp.vec3(*quad1),
        wp.vec3(*quad2),
        types.vec5(1.0, 0.0, 0.0, 0.0, 0.0),
      ],
      outputs=[result],
    )

    np.testing.assert_allclose(result.numpy()[0], expected, rtol=1.0e-6, atol=1.0e-6)

  @parameterized.product(
    cone=tuple(ConeType),
    solver_=tuple(SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_constraint_update(self, cone, solver_, jacobian):
    """Tests _update_constraint function is correct."""
    for keyframe in range(3):
      mjm, mjd, m, d = test_data.fixture(
        "constraints.xml",
        keyframe=keyframe,
        overrides={"opt.solver": solver_, "opt.cone": cone, "opt.jacobian": jacobian, "opt.iterations": 0},
      )

      def update_constraints(qacc):
        jaref = np.zeros(mjd.nefc, dtype=float)
        cost = np.zeros(1)
        mujoco.mj_mulJacVec(mjm, mjd, jaref, qacc)
        mujoco.mj_constraintUpdate(mjm, mjd, jaref - mjd.efc_aref, cost, 0)

      update_constraints(mjd.qacc)

      # solve with 0 iterations just initializes constraints and then exits
      d.efc.force.fill_(wp.inf)
      d.qfrc_constraint.fill_(wp.inf)
      ctx = solver._create_solver_context(m, d)
      solver._solve(m, d, ctx)

      # Get the ordering indices based on efc_force, efc_state for MJWarp
      nefc = d.nefc.numpy()[0]
      efc_force = d.efc.force.numpy()[0, :nefc]
      efc_state = d.efc.state.numpy()[0, :nefc]
      # Get the ordering indices based on efc_force, efc_state for MuJoCo
      mjd_efc_force = mjd.efc_force[:nefc]
      mjd_efc_state = mjd.efc_state[:nefc]

      # Create sorting keys using lexsort (more efficient for multiple keys)
      d_sort_indices = np.lexsort((efc_force, efc_state))
      mjd_sort_indices = np.lexsort((mjd_efc_force, mjd_efc_state))

      solver.init_context(m, d, ctx, grad=False)
      qfrc_constraint = d.qfrc_constraint.numpy()[0]

      efc_sorted_force = efc_force[d_sort_indices]
      efc_sorted_state = efc_state[d_sort_indices]
      mjd_sorted_force = mjd_efc_force[mjd_sort_indices]
      mjd_sorted_state = mjd_efc_state[mjd_sort_indices]

      _assert_eq(efc_sorted_state, mjd_sorted_state, "efc_state")
      _assert_eq(efc_sorted_force, mjd_sorted_force, "efc_force")
      _assert_eq(qfrc_constraint, mjd.qfrc_constraint, "qfrc_constraint")

  @parameterized.product(
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_init_linesearch(self, cone, jacobian):
    """Test linesearch initialization."""
    for keyframe in range(3):
      mjm, mjd, m, d = test_data.fixture(
        "constraints.xml",
        keyframe=keyframe,
        overrides={
          "opt.iterations": 0,
          "opt.ls_iterations": 1,
          "opt.cone": cone,
          "opt.jacobian": jacobian,
        },
      )

      # One step to obtain more non-zeros results
      mjw.step(m, d)

      # Create a SolverContext to access internal solver arrays
      ctx = solver._create_solver_context(m, d)
      solver._solve(m, d, ctx)

      # Calculate target values
      nefc = d.nefc.numpy()[0]
      ctx_search_np = ctx.search.numpy()[0]
      if m.is_sparse:
        efc_J_np = np.zeros((nefc, m.nv))
        mujoco.mju_sparse2dense(
          efc_J_np,
          d.efc.J.numpy()[0, 0],
          d.efc.J_rownnz.numpy()[0, :nefc],
          d.efc.J_rowadr.numpy()[0, :nefc],
          d.efc.J_colind.numpy()[0, 0],
        )
      else:
        efc_J_np = d.efc.J.numpy()[0, :nefc, : m.nv]
      efc_Ma_np = d.efc.Ma.numpy()[0]
      ctx_Jaref_np = ctx.Jaref.numpy()[0][:nefc]
      efc_D_np = d.efc.D.numpy()[0][:nefc]
      qfrc_smooth_np = d.qfrc_smooth.numpy()[0]

      target_mv = np.zeros(mjm.nv)
      mujoco.mj_mulM(mjm, mjd, target_mv, ctx_search_np)
      target_jv = efc_J_np @ ctx_search_np
      target_quad_gauss = np.array(
        [
          0.0,
          np.dot(ctx_search_np, efc_Ma_np - qfrc_smooth_np),
          0.5 * np.dot(ctx_search_np, target_mv),
        ]
      )
      target_quad = np.transpose(
        np.vstack(
          [
            0.5 * ctx_Jaref_np * ctx_Jaref_np * efc_D_np,
            target_jv * ctx_Jaref_np * efc_D_np,
            0.5 * target_jv * target_jv * efc_D_np,
          ]
        )
      )

      # Reset and launch linesearch
      ctx.jv.fill_(wp.inf)
      ctx.search_unchanged.fill_(False)
      solver._linesearch(m, d, ctx)

      # mv and jv are always written
      ctx_mv = ctx.mv.numpy()[0]
      ctx_jv = ctx.jv.numpy()[0]
      _assert_eq(ctx_mv, target_mv, "mv")
      _assert_eq(ctx_jv[:nefc], target_jv[:nefc], "jv")

  @parameterized.product(
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC), jacobian=(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  )
  def test_update_gradient_CG(self, cone, jacobian):
    """Test _update_gradient function is correct for the CG solver."""
    mjm, mjd, m, d = test_data.fixture(
      "humanoid/humanoid.xml",
      keyframe=0,
      overrides={"opt.cone": cone, "opt.solver": SolverType.CG, "opt.jacobian": jacobian, "opt.iterations": 0},
    )

    # Create SolverContext and initialize
    ctx = solver._create_solver_context(m, d)
    solver.init_context(m, d, ctx, grad=True)

    # Calculate Mgrad with Mujoco C
    mj_Mgrad = np.zeros(shape=(1, mjm.nv), dtype=float)
    mj_grad = np.tile(ctx.grad.numpy()[:, : mjm.nv], (1, 1))
    mujoco.mj_solveM(mjm, mjd, mj_Mgrad, mj_grad)

    ctx_Mgrad = ctx.Mgrad.numpy()[0, : mjm.nv]
    _assert_eq(ctx_Mgrad, mj_Mgrad[0], name="Mgrad")

  def test_linesearch_accepts_sub_float32_improvement(self):
    """Line search should not lose small improvements on large absolute costs."""
    _, _, m, d = test_data.fixture(
      "constraints.xml",
      overrides={
        "opt.cone": ConeType.PYRAMIDAL,
        "opt.jacobian": mujoco.mjtJacobian.mjJAC_DENSE,
        "opt.iterations": 0,
        "opt.ls_iterations": 50,
      },
    )

    ctx = solver._create_solver_context(m, d)
    _set_inertia_scale(m, d, ctx)

    d.ne = wp.array([0], dtype=int)
    d.nf = wp.array([0], dtype=int)
    d.nefc = wp.array([1], dtype=int)
    d.nacon = wp.array([0], dtype=int)
    d.M = wp.zeros(d.M.shape, dtype=float)
    d.qacc = wp.zeros(d.qacc.shape, dtype=float)
    d.efc.Ma = wp.zeros(d.efc.Ma.shape, dtype=float)
    d.qfrc_smooth = wp.zeros(d.qfrc_smooth.shape, dtype=float)

    efc_j = np.zeros(d.efc.J.shape, dtype=np.float32)
    efc_j[0, 0, 0] = 1.0
    d.efc.J = wp.array(efc_j, dtype=float)

    efc_d = np.zeros(d.efc.D.shape, dtype=np.float32)
    efc_d[0, 0] = 0.004
    d.efc.D = wp.array(efc_d, dtype=float)
    d.efc.frictionloss = wp.zeros(d.efc.frictionloss.shape, dtype=float)

    search = np.zeros(ctx.search.shape, dtype=np.float32)
    search[0, 0] = 1.0
    ctx.search = wp.array(search, dtype=float)
    jaref = np.zeros(ctx.Jaref.shape, dtype=np.float32)
    jaref[0, 0] = -1.0
    ctx.Jaref = wp.array(jaref, dtype=float)
    ctx.search_dot = wp.array([1.0], dtype=float)
    ctx.done = wp.array([False], dtype=bool)
    ctx.search_unchanged.fill_(False)

    solver._linesearch(m, d, ctx)

    self.assertGreater(d.qacc.numpy()[0, 0], 0.5)
    self.assertGreater(ctx.improvement.numpy()[0], 0.001)

    qfrc_smooth = np.zeros(d.qfrc_smooth.shape, dtype=np.float32)
    qfrc_smooth[0, 0] = -0.0005
    d.qfrc_smooth = wp.array(qfrc_smooth, dtype=float)
    d.qacc = wp.zeros(d.qacc.shape, dtype=float)
    d.efc.Ma = wp.zeros(d.efc.Ma.shape, dtype=float)

    search = np.zeros(ctx.search.shape, dtype=np.float32)
    search[0, 0] = -2.0
    ctx.search = wp.array(search, dtype=float)
    jaref = np.zeros(ctx.Jaref.shape, dtype=np.float32)
    jaref[0, 0] = 1.0
    ctx.Jaref = wp.array(jaref, dtype=float)
    ctx.search_dot = wp.array([4.0], dtype=float)
    ctx.done = wp.array([False], dtype=bool)
    ctx.search_unchanged.fill_(False)

    solver._linesearch(m, d, ctx)

    self.assertLess(d.qacc.numpy()[0, 0], -1.0)
    self.assertLess(ctx.improvement.numpy()[0], 0.001)
    self.assertGreater(ctx.improvement.numpy()[0], 0.0004)

  def test_linesearch_requires_cost_improvement_to_converge(self):
    """Line search should not converge at an unimproved bound."""
    _, _, m, d = test_data.fixture(
      "constraints.xml",
      overrides={
        "opt.cone": ConeType.PYRAMIDAL,
        "opt.jacobian": mujoco.mjtJacobian.mjJAC_DENSE,
        "opt.iterations": 0,
        "opt.ls_iterations": 50,
      },
    )
    ctx = solver._create_solver_context(m, d)

    # Exercise derivative convergence with a loose scaled tolerance.
    ctx.inertia_scale.fill_(1.0 / (1.0e9 * m.nv))
    d.ne = wp.array([1], dtype=int)
    d.nf = wp.array([0], dtype=int)
    d.nefc = wp.array([4], dtype=int)
    d.nacon = wp.array([0], dtype=int)
    d.M.zero_()
    d.qacc.zero_()
    d.efc.Ma.zero_()
    d.qfrc_smooth.zero_()

    efc_j = np.zeros(d.efc.J.shape, dtype=np.float32)
    efc_j[0, :4, 0] = [1.0, -1.0, -1.0, 1.0]
    d.efc.J.assign(efc_j)

    efc_d = np.zeros(d.efc.D.shape, dtype=np.float32)
    efc_d[0, :4] = [0.0030866014, 1.7318993, 4555.986, 0.14425136]
    d.efc.D.assign(efc_d)
    d.efc.frictionloss.zero_()

    search = np.zeros(ctx.search.shape, dtype=np.float32)
    search[0, 0] = 1.0
    ctx.search.assign(search)
    jaref = np.zeros(ctx.Jaref.shape, dtype=np.float32)
    jaref[0, :4] = [-4.634305, 1.5213286, 0.03916324, 1.388601]
    ctx.Jaref.assign(jaref)
    ctx.search_dot.fill_(1.0)
    ctx.done.fill_(False)
    ctx.search_unchanged.fill_(False)

    solver._linesearch(m, d, ctx)

    self.assertGreater(d.qacc.numpy()[0, 0], 0.03)
    self.assertGreater(ctx.improvement.numpy()[0], 5.0e-4)

  def test_linesearch_converges_at_stationary_start(self):
    """Line search should converge when its start is already the line minimum."""
    # Captured from a box resting on a plane, warm-started at its float32 solution: the slope at
    # alpha = 0 and every cost change are rounding noise, so no step strictly improves the cost.
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="5 5 0.1"/>
          <body pos="0 0 0.049">
            <freejoint/>
            <geom type="box" size="0.05 0.05 0.05"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    ctx = solver._create_solver_context(m, d)
    _set_inertia_scale(m, d, ctx)

    efc_J = [
      [0.0, 1.0, 1.0, -5.3886324e-05, 0.05, -0.05],
      [0.0, -1.0, 1.0, -0.09994611, 0.05, 0.05],
      [-1.0, 0.0, 1.0, -0.05, 0.09994611, -0.05],
      [1.0, 0.0, 1.0, -0.05, 5.3886324e-05, 0.05],
      [0.0, 1.0, 1.0, -5.3886324e-05, -0.05, 0.05],
      [0.0, -1.0, 1.0, -0.09994611, -0.05, -0.05],
      [-1.0, 0.0, 1.0, -0.05, -5.3886324e-05, -0.05],
      [1.0, 0.0, 1.0, -0.05, -0.09994611, 0.05],
      [0.0, 1.0, 1.0, 0.09994611, 0.05, -0.05],
      [0.0, -1.0, 1.0, 5.3886324e-05, 0.05, 0.05],
      [-1.0, 0.0, 1.0, 0.05, 0.09994611, 0.05],
      [1.0, 0.0, 1.0, 0.05, 5.3886324e-05, -0.05],
      [0.0, 1.0, 1.0, 0.09994611, -0.05, 0.05],
      [0.0, -1.0, 1.0, 5.3886324e-05, -0.05, -0.05],
      [-1.0, 0.0, 1.0, 0.05, -5.3886324e-05, 0.05],
      [1.0, 0.0, 1.0, 0.05, -0.09994611, -0.05],
    ]
    efc_D = 2.27938
    jaref = [
      -0.26898766,
      -0.26898766,
      -0.26898766,
      -0.26898766,
      -0.26898766,
      -0.26898763,
      -0.26898766,
      -0.26898766,
      -0.26898766,
      -0.26898766,
      -0.26898766,
      -0.26898766,
      -0.2689877,
      -0.26898763,
      -0.26898763,
      -0.26898766,
    ]
    search = [-2.6907532e-09, 1.0365673e-08, -8.1853896e-17, -2.0852168e-08, -1.2571091e-07, 9.2155496e-08]
    search_dot = 2.4845367e-14
    efc_Ma = [-3.1322562e-09, -1.57873e-09, -2.102178e-09, -1.3895347e-10, 1.2078877e-10, -7.200439e-11]
    qfrc_smooth = [-1.323489e-23, 0.0, -9.81, 2.3210504e-25, -3.9414281e-25, -6.260127e-30]

    nefc = len(efc_J)
    d.nefc.fill_(nefc)
    d.ne.zero_()
    d.nf.zero_()
    d.efc.frictionloss.zero_()
    J = np.zeros(d.efc.J.shape, dtype=np.float32)
    J[0, :nefc, :6] = efc_J
    d.efc.J.assign(J)
    D = np.zeros(d.efc.D.shape, dtype=np.float32)
    D[0, :nefc] = efc_D
    d.efc.D.assign(D)
    Jaref = np.zeros(ctx.Jaref.shape, dtype=np.float32)
    Jaref[0, :nefc] = jaref
    ctx.Jaref.assign(Jaref)
    ctx.search.assign(np.array([search], dtype=np.float32))
    ctx.search_dot.fill_(search_dot)
    d.efc.Ma.assign(np.array([efc_Ma], dtype=np.float32))
    d.qfrc_smooth.assign(np.array([qfrc_smooth], dtype=np.float32))
    ctx.search_unchanged.zero_()
    ctx.done.fill_(False)
    d.overflow.zero_()
    ctx.alpha.fill_(wp.inf)

    solver._linesearch(m, d, ctx)

    self.assertFalse(d.overflow.numpy()[0] & types.OverflowType.LS_ITERATIONS)
    self.assertEqual(ctx.alpha.numpy()[0], 0.0)

  def _slide_linesearch(self, efc_D, frictionloss, jaref):
    """Line search on friction rows of one slide joint whose slopes cancel exactly at alpha = 0."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" jacobian="dense" iterations="0"/>
        <worldbody>
          <body>
            <joint type="slide"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      njmax=5,
    )
    ctx = solver._create_solver_context(m, d)
    _set_inertia_scale(m, d, ctx)
    d.ne.zero_()
    d.nf.fill_(5)
    d.nefc.fill_(5)
    d.nacon.zero_()
    d.M.fill_(1.0)
    d.qacc.zero_()
    d.efc.Ma.zero_()
    d.qfrc_smooth.zero_()
    d.efc.J.fill_(1.0)
    d.efc.D.assign(np.array([efc_D], dtype=np.float32))
    d.efc.frictionloss.assign(np.array([frictionloss], dtype=np.float32))
    ctx.Jaref.assign(np.array([jaref], dtype=np.float32))
    ctx.search.fill_(2.0**-24)
    ctx.search_dot.fill_(2.0**-48)
    ctx.done.fill_(False)
    ctx.search_unchanged.fill_(False)
    d.overflow.zero_()
    ctx.alpha.fill_(wp.inf)

    solver._linesearch(m, d, ctx)
    return d, ctx

  def test_linesearch_converges_when_float32_drops_slope_terms(self):
    """Line search should converge at a minimum whose slope float32 cannot resolve."""
    # from newton-physics/newton#4212: the float32 reduction loses the small slope terms, and no
    # step strictly improves the cost
    stiffness = 2.0**8
    offset = 2.0**28
    large = stiffness * offset
    small = 8.0
    small_offset = 1.5 * 2.0**10
    d, ctx = self._slide_linesearch(
      [stiffness, 1.0, 1.0, 1.0, 1.0],
      [2 * large, small, large, 0.0, small],
      [offset, small_offset, -16 * large, 0.0, -small_offset],
    )

    self.assertFalse(d.overflow.numpy()[0] & types.OverflowType.LS_ITERATIONS)
    self.assertEqual(ctx.alpha.numpy()[0], 0.0)

  def test_linesearch_rejects_rounding_improvement_at_stationary_start(self):
    """Line search should not take a step whose cost improvement is float32 rounding."""
    # the slope at alpha = 0 is exactly zero, but rounding of the ~1e18 costs reports an improvement
    large = 2.0**36
    small = 8.0
    d, ctx = self._slide_linesearch(
      [1.0, 1.0, 1.0, 1.0, 1.0],
      [large, small, large, 0.0, small],
      [1.0e9, 1.0e9, -1.0e9, 1.0e9, -1.0e9],
    )

    self.assertEqual(ctx.alpha.numpy()[0], 0.0)
    self.assertEqual(d.qacc.numpy()[0, 0], 0.0)

  @parameterized.parameters(1, 2)
  def test_linesearch_accepts_converged_newton_point(self, nworld):
    """Linesearch should accept a Newton point at the minimizer whatever its derivative sign."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <joint type="slide"/>
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={
        "opt.cone": ConeType.PYRAMIDAL,
        "opt.jacobian": mujoco.mjtJacobian.mjJAC_DENSE,
        "opt.iterations": 0,
        "opt.ls_iterations": 50,
      },
      nworld=nworld,
      njmax=2,
    )
    ctx = solver._create_solver_context(m, d)
    _set_inertia_scale(m, d, ctx)

    # cost 0.5 * (alpha - target)^2 from an equality row plus 0.5 * (alpha - 1)^2 from an
    # inequality row active for alpha < 1: Newton from 0 undershoots, and the second Newton step
    # lands exactly on the minimizer alpha = target, where the derivative is zero and bracketing
    # alone rejects it
    d.ne.fill_(1)
    d.nf.fill_(0)
    d.nefc.fill_(2)
    d.nacon.fill_(0)
    d.M.zero_()
    d.qacc.zero_()
    d.efc.Ma.zero_()
    d.qfrc_smooth.zero_()

    efc_j = np.zeros(d.efc.J.shape, dtype=np.float32)
    efc_j[:, :2, 0] = 1.0
    d.efc.J.assign(efc_j)

    efc_d = np.zeros(d.efc.D.shape, dtype=np.float32)
    efc_d[:, :2] = 1.0
    d.efc.D.assign(efc_d)
    d.efc.frictionloss.zero_()

    search = np.zeros(ctx.search.shape, dtype=np.float32)
    search[:, 0] = 1.0
    ctx.search.assign(search)

    targets = np.array([4.0, 6.0][:nworld], dtype=np.float32)
    jaref = np.zeros(ctx.Jaref.shape, dtype=np.float32)
    jaref[:, 0] = -targets
    jaref[:, 1] = -1.0
    ctx.Jaref.assign(jaref)
    ctx.search_dot.fill_(1.0)
    ctx.done.fill_(False)
    ctx.search_unchanged.fill_(False)

    solver._linesearch(m, d, ctx)

    np.testing.assert_allclose(d.qacc.numpy()[:, 0], targets, rtol=1e-6)

  @parameterized.product(
    case=(
      # Inactive at alpha=0 (Jaref > 0), active at lo_alpha_in = 2^-20 with |lo_in[1]| < 1e-6.
      dict(
        nf=0, qfrc_smooth=2.0**-20, efc_j=(-1.0, -1.0), jaref=(2.0**-22, 2.0**-21), expected=(2.5 * 2.0**-22, 1.5 * 2.0**-21)
      ),
      # Quadratic frictionloss (|x| < rf): unshifted cost(0) ulp >= 4.0 hides improvement.
      dict(nf=1, efc_j=(-(2.0**-13), -(2.0**-13)), frictionloss=16384.0, jaref=(8192.0, 12288.0), expected=(1.0, 1.5)),
      # Linear frictionloss (|x| >= rf): unshifted x = start + dx rounds to start.
      dict(nf=1, efc_j=(-(2.0**-13), 2.0**-12), frictionloss=8192.0, jaref=(16384.0, -16384.0), expected=(1.0, 2.0)),
      # Linear-to-quadratic crossing (+/-rf): unshifted cost(x) - cost(start) rounds to 0.
      dict(
        nf=1,
        efc_d=11.0,
        m=(15.0 * 2.0**-27, 3.75 * 2.0**-27),
        efc_j=(-3.0 * 2.0**-27, 1.5 * 2.0**-27),
        frictionloss=5.0,
        jaref=(float(np.float32(5.0) / np.float32(11.0)), -float(np.float32(5.0) / np.float32(11.0))),
        expected=(1.0, 2.0),
      ),
    ),
    nworld=(1, 2),
  )
  def test_linesearch_small_gtol_and_shifted_frictionloss(self, case, nworld):
    """Linesearch brackets slopes below 1e-6 and preserves sub-ulp frictionloss deltas."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" jacobian="dense"/>
        <worldbody>
          <body>
            <joint type="slide"/>
            <geom size="0.1" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      njmax=1,
    )
    ctx = solver._create_solver_context(m, d)
    _set_inertia_scale(m, d, ctx)
    d.ne.zero_()
    d.nf.fill_(case["nf"])
    d.nefc.fill_(1)
    d.nacon.zero_()
    mass = np.zeros(d.M.shape, dtype=np.float32)
    mass[:, 0] = case.get("m", (1.0, 1.0))[:nworld]
    d.M.assign(mass)
    d.qacc.zero_()
    d.efc.Ma.zero_()
    d.qfrc_smooth.fill_(case.get("qfrc_smooth", 0.0))
    efc_j = np.zeros(d.efc.J.shape, dtype=np.float32)
    efc_j[:, 0, 0] = case["efc_j"][:nworld]
    d.efc.J.assign(efc_j)
    d.efc.D.fill_(case.get("efc_d", 1.0))
    d.efc.frictionloss.fill_(case.get("frictionloss", 0.0))
    jaref = np.zeros(ctx.Jaref.shape, dtype=np.float32)
    jaref[:, 0] = case["jaref"][:nworld]
    ctx.Jaref.assign(jaref)
    ctx.search.fill_(1.0)
    ctx.search_dot.fill_(1.0)
    ctx.done.fill_(False)
    ctx.search_unchanged.fill_(False)
    d.overflow.zero_()
    ctx.alpha.fill_(wp.inf)

    solver._linesearch(m, d, ctx)

    expected = np.array(case["expected"][:nworld], dtype=np.float32)
    np.testing.assert_allclose(ctx.alpha.numpy(), expected, rtol=1e-6)
    if nworld == 2:
      self.assertFalse(np.allclose(ctx.alpha.numpy()[0], ctx.alpha.numpy()[1]))

  def test_linesearch_iterations_overflow(self):
    """Linesearch records overflow when iteration limit reached without convergence."""
    _, _, m, d = test_data.fixture(
      "constraints.xml",
      overrides={
        "opt.cone": ConeType.PYRAMIDAL,
        "opt.jacobian": mujoco.mjtJacobian.mjJAC_DENSE,
        "opt.iterations": 0,
        "opt.ls_iterations": 0,
      },
    )
    ctx = solver._create_solver_context(m, d)

    ctx.inertia_scale.fill_(1.0 / (1.0e9 * m.nv))
    d.ne = wp.array([1], dtype=int)
    d.nf = wp.array([0], dtype=int)
    d.nefc = wp.array([4], dtype=int)
    d.nacon = wp.array([0], dtype=int)
    d.M.zero_()
    d.qacc.zero_()
    d.efc.Ma.zero_()
    d.qfrc_smooth.zero_()

    efc_j = np.zeros(d.efc.J.shape, dtype=np.float32)
    efc_j[0, :4, 0] = [1.0, -1.0, -1.0, 1.0]
    d.efc.J.assign(efc_j)

    efc_d = np.zeros(d.efc.D.shape, dtype=np.float32)
    efc_d[0, :4] = [0.0030866014, 1.7318993, 4555.986, 0.14425136]
    d.efc.D.assign(efc_d)
    d.efc.frictionloss.zero_()

    search = np.zeros(ctx.search.shape, dtype=np.float32)
    search[0, 0] = 1.0
    ctx.search.assign(search)
    jaref = np.zeros(ctx.Jaref.shape, dtype=np.float32)
    jaref[0, :4] = [-4.634305, 1.5213286, 0.03916324, 1.388601]
    ctx.Jaref.assign(jaref)
    ctx.search_dot.fill_(1.0)
    ctx.done.fill_(False)
    ctx.search_unchanged.fill_(False)

    solver._linesearch(m, d, ctx)

    self.assertTrue(d.overflow.numpy()[0] & types.OverflowType.LS_ITERATIONS)

  @parameterized.parameters(
    (ConeType.PYRAMIDAL, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_DENSE),
    (ConeType.ELLIPTIC, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_DENSE),
    (ConeType.PYRAMIDAL, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_SPARSE),
    (ConeType.ELLIPTIC, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_SPARSE),
    (ConeType.PYRAMIDAL, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_DENSE),
    (ConeType.ELLIPTIC, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_DENSE),
    (ConeType.PYRAMIDAL, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_SPARSE),
    (ConeType.ELLIPTIC, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_solve(self, cone, solver_, iterations, ls_iterations, jacobian):
    """Tests solve."""
    for keyframe in range(3):
      mjm, mjd, m, d = test_data.fixture(
        "constraints.xml",
        keyframe=keyframe,
        overrides={
          "opt.jacobian": jacobian,
          "opt.cone": cone,
          "opt.solver": solver_,
          "opt.iterations": iterations,
          "opt.ls_iterations": ls_iterations,
        },
      )

      mujoco.mj_forward(mjm, mjd)

      d.qacc.fill_(wp.inf)
      d.qfrc_constraint.fill_(wp.inf)
      d.efc.force.fill_(wp.inf)

      if solver_ == mujoco.mjtSolver.mjSOL_CG:
        mjw.factor_m(m, d)
      mjw.solve(m, d)

      def cost(qacc):
        jaref = np.zeros(mjd.nefc, dtype=float)
        cost = np.zeros(1)
        mujoco.mj_mulJacVec(mjm, mjd, jaref, qacc)
        mujoco.mj_constraintUpdate(mjm, mjd, jaref - mjd.efc_aref, cost, 0)
        return cost

      mj_cost = cost(mjd.qacc)
      mjwarp_cost = cost(d.qacc.numpy()[0])
      self.assertLessEqual(mjwarp_cost, mj_cost * 1.025)

      if m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON:
        _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
        _assert_eq(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, "qfrc_constraint")
        _assert_eq(d.efc.force.numpy()[0, : mjd.nefc], mjd.efc_force, "efc_force")

  @parameterized.parameters(
    (ConeType.PYRAMIDAL, SolverType.NEWTON, mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtIntegrator.mjINT_EULER),
    (ConeType.ELLIPTIC, SolverType.NEWTON, mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtIntegrator.mjINT_EULER),
    (ConeType.PYRAMIDAL, SolverType.NEWTON, mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtIntegrator.mjINT_EULER),
    (ConeType.ELLIPTIC, SolverType.NEWTON, mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtIntegrator.mjINT_EULER),
    (ConeType.PYRAMIDAL, SolverType.NEWTON, mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtIntegrator.mjINT_DISCRETE),
    (ConeType.PYRAMIDAL, SolverType.NEWTON, mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtIntegrator.mjINT_DISCRETE),
    (ConeType.PYRAMIDAL, SolverType.CG, mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtIntegrator.mjINT_EULER),
  )
  def test_solve_deterministic(self, cone, solver_, jacobian, integrator):
    """Tests that the solve without atomics matches the default solve and MuJoCo."""
    for keyframe in range(3):
      for nworld in (1, 2):
        res = {}
        for deterministic in (types.DeterminismType.NONE, types.DeterminismType.ATOMICS):
          mjm, mjd, m, d = test_data.fixture(
            "constraints.xml",
            keyframe=keyframe,
            nworld=nworld,
            overrides={
              "opt.jacobian": jacobian,
              "opt.cone": cone,
              "opt.solver": solver_,
              "opt.integrator": integrator,
              "opt.deterministic": deterministic,
            },
          )

          d.qacc.fill_(wp.inf)
          d.qfrc_constraint.fill_(wp.inf)

          mjw.forward(m, d)
          res[deterministic] = (d.qacc.numpy(), d.qfrc_constraint.numpy())

        for i, name in enumerate(("qacc", "qfrc_constraint")):
          default, det = res[types.DeterminismType.NONE][i], res[types.DeterminismType.ATOMICS][i]
          # the sums run in a different order: allow rounding at the scale of the largest element
          np.testing.assert_allclose(det, default, rtol=1e-3, atol=1e-4 * np.abs(default).max(), err_msg=name)
          if solver_ == SolverType.NEWTON:
            for worldid in range(nworld):
              _assert_eq(det[worldid], getattr(mjd, name), name)

  @parameterized.parameters(1, 2)
  def test_update_gradient_grad_deterministic(self, nworld):
    """Tests the gradient norm without atomics when qfrc_smooth is padded."""
    # the discrete integrator passes a qfrc_smooth with nv_pad columns
    nv, nv_pad = 3, 4
    rng = np.random.default_rng(0)
    Ma = rng.uniform(1.0, 2.0, (nworld, nv)).astype(np.float32)
    qfrc_smooth = rng.uniform(1.0, 2.0, (nworld, nv_pad)).astype(np.float32)
    qfrc_constraint = rng.uniform(1.0, 2.0, (nworld, nv)).astype(np.float32)
    expected = Ma - qfrc_smooth[:, :nv] - qfrc_constraint

    for deterministic in (False, True):
      grad = wp.full((nworld, nv), wp.inf, dtype=float)
      # the default kernel accumulates, the deterministic one assigns
      grad_dot = wp.full(nworld, 0.0 if not deterministic else wp.inf, dtype=float)
      wp.launch(
        solver._update_gradient_grad(False, nv if deterministic else 0, deterministic),
        dim=(nworld, nv),
        inputs=[
          wp.array(qfrc_smooth),
          wp.array(qfrc_constraint),
          wp.array(Ma),
          wp.zeros(nworld, dtype=int),
          wp.zeros(nworld, dtype=bool),
        ],
        outputs=[grad, grad_dot],
      )
      np.testing.assert_allclose(grad.numpy(), expected, rtol=1e-6)
      np.testing.assert_allclose(grad_dot.numpy(), np.sum(expected * expected, axis=1), rtol=1e-5)

  @parameterized.named_parameters(
    ("max3", (3,)),
    ("max4_mixed", (3, 4)),
    ("max6_mixed", (3, 4, 6)),
  )
  def test_solve_elliptic_sparse_diagonal_inertia(self, condims):
    """Fused sparse cone blocks match the reference Hessian and solve."""
    n = 12  # 72 dofs -> sparse path
    bodies = "".join(
      f'<body pos="{(i % 4) * 0.25 - 0.4:.3f} {(i // 4) * 0.25 - 0.3:.3f} 0.020">'
      f'<freejoint/><geom type="box" size="0.04 0.05 0.03" mass="0.5" condim="{condims[i % len(condims)]}"/></body>'
      for i in range(n)
    )
    xml = f"""
    <mujoco>
      <option cone="elliptic" solver="Newton" jacobian="sparse" iterations="20" ls_iterations="20" integrator="implicitfast"/>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 .1" friction="1 0.01 0.001" condim="1"/>
        {bodies}
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, _ = test_data.fixture(xml=xml)
    self.assertTrue(m.is_sparse)

    mjd.qvel[0::6] = 1.5  # slide -> cone middle zone
    mujoco.mj_forward(mjm, mjd)
    self.assertGreater(mjd.nefc, 0)
    self.assertEqual(set(mjd.contact.dim[: mjd.ncon]), set(condims))

    d = mjw.put_data(mjm, mjd)
    ctx = solver._create_solver_context(m, d)
    solver.init_context(m, d, ctx, grad=True)
    fused_h = ctx.h.numpy().copy()

    ctx.done.fill_(False)
    wp.launch(
      solver._update_gradient_init_h_sparse(False),
      dim=(d.nworld, m.nv_pad, solver._INIT_H_LANES),
      inputs=[m.nv, m.M_elemid, d.nf, d.M, d.efc.D, d.efc.state, d.dof_frictionadr, d.cdof_dof, ctx.done],
      outputs=[ctx.h],
    )
    jtdaj_kernel = solver._JTDACJ_sparse(False, ConeType.PYRAMIDAL, 3, m.block_dim.update_gradient_JTDAJ_sparse)
    groups_per_world = solver._jtdaj_groups_per_world(d.nworld, d.njmax, jtdaj_kernel)
    wp.launch(
      jtdaj_kernel,
      dim=(d.nworld, groups_per_world, solver._JTDAJ_THREADS_PER_GROUP),
      inputs=[
        m.opt.impratio_invsqrt,
        d.contact.friction,
        d.contact.dim,
        d.efc.id,
        d.efc.jtdaj_adr,
        d.efc.jtdaj_nrow,
        d.efc.jtdaj_nblock,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.efc.D,
        d.efc.state,
        d.dof_cdof,
        ctx.Jaref,
        ctx.done,
        groups_per_world,
      ],
      outputs=[ctx.h],
      block_dim=m.block_dim.update_gradient_JTDAJ_sparse,
    )
    reference_h = ctx.h.numpy()[:, : m.nv, : m.nv] + _elliptic_sparse_hessian_reference(m, d, ctx)
    np.testing.assert_allclose(fused_h[:, : m.nv, : m.nv], reference_h, rtol=1e-5, atol=1e-6)

    d.qacc.fill_(wp.inf)
    d.qfrc_constraint.fill_(wp.inf)
    d.efc.force.fill_(wp.inf)
    mjw.solve(m, d)

    qacc = d.qacc.numpy()[0]
    self.assertTrue(np.all(np.isfinite(qacc)), "Newton solve produced non-finite qacc")
    _assert_eq(qacc, mjd.qacc, "qacc")

  @absltest.skipIf(not wp.get_device().is_cuda, "CUDA occupancy queries are not used on CPU")
  @parameterized.parameters(
    (True, ConeType.ELLIPTIC, 6, 32),
    (False, ConeType.PYRAMIDAL, 3, 128),
  )
  def test_jtdaj_occupancy_reuses_launch_module(self, compact, cone, max_condim, block_dim):
    """The occupancy query and JTDAJ launch must share one compiled module."""
    device = wp.get_device()

    kernel = solver._JTDACJ_sparse(compact, cone, max_condim, block_dim)
    solver._jtdaj_groups_per_world(1, 100, kernel)
    self.assertIn((device.context, block_dim), kernel.module.execs)
    occupancy_module = kernel.module.load(device)
    launch_module = kernel.module.load(device, block_dim=block_dim)
    self.assertIs(occupancy_module, launch_module)
    self.assertEqual(len(kernel.module.execs), 1)

    inc_kernel = solver._update_gradient_h_incremental_sparse(compact)
    solver._jtdaj_groups_per_world(1, 100, inc_kernel)
    occupancy_inc_module = inc_kernel.module.load(device)
    launch_inc_module = inc_kernel.module.load(device, block_dim=inc_kernel.module.options["block_dim"])
    self.assertIs(occupancy_inc_module, launch_inc_module)
    self.assertEqual(len(inc_kernel.module.execs), 1)

  def test_cg_tiled_kernels_are_unique(self):
    """Ensure CG tiled kernels with custom block_dim do not share solver.py's module."""
    shared_module = solver._solve_init_efc.module
    self.assertNotEqual(solver._solve_init_search_cg_tiled.module, shared_module)
    self.assertNotEqual(solver._update_gradient_grad_tiled.module, shared_module)
    self.assertNotEqual(solver._solve_search_update_cg_tiled.module, shared_module)

  def test_elliptic_dense_hessian(self):
    """Structured dense cone contraction matches the reference Hessian."""
    condims = (3, 4, 6)
    bodies = "".join(
      f'<body pos="{i * 0.25 - 0.25:.3f} 0 0.020">'
      f'<freejoint/><geom type="box" size="0.04 0.05 0.03" mass="0.5" condim="{condim}"/></body>'
      for i, condim in enumerate(condims)
    )
    xml = f"""
    <mujoco>
      <option cone="elliptic" solver="Newton" jacobian="dense" iterations="20" ls_iterations="20"/>
      <worldbody>
        <geom type="plane" size="5 5 .1" friction="1 0.01 0.001" condim="1"/>
        {bodies}
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, _ = test_data.fixture(xml=xml)
    self.assertFalse(m.is_sparse)

    mjd.qvel[0::6] = 1.5
    mujoco.mj_forward(mjm, mjd)
    self.assertEqual(set(mjd.contact.dim[: mjd.ncon]), set(condims))

    d = mjw.put_data(mjm, mjd)
    ctx = solver._create_solver_context(m, d)
    solver.init_context(m, d, ctx, grad=True)
    ctx.h.zero_()
    ctx.done.fill_(False)

    if wp.get_device().is_cuda:
      dim_block = (wp.get_device().sm_count * 6 * 256 + m.dof_tri_row.size - 1) // m.dof_tri_row.size
      nblocks_perblock = (d.naconmax + dim_block - 1) // dim_block
    else:
      dim_block = 1
      nblocks_perblock = d.naconmax
    wp.launch(
      solver._update_gradient_JTCJ_dense(0),
      dim=(dim_block, m.dof_tri_row.size),
      inputs=[
        m.opt.impratio_invsqrt,
        m.dof_tri_row,
        m.dof_tri_col,
        d.contact.dist,
        d.contact.includemargin,
        d.contact.friction,
        d.contact.dim,
        d.contact.efc_address,
        d.contact.worldid,
        d.efc.J,
        d.efc.D,
        d.efc.state,
        d.naconmax,
        d.nacon,
        ctx.Jaref,
        ctx.done,
        nblocks_perblock,
        dim_block,
      ],
      outputs=[ctx.h],
    )

    np.testing.assert_allclose(
      ctx.h.numpy()[:, : m.nv, : m.nv],
      _elliptic_dense_hessian_reference(m, d, ctx),
      rtol=1e-5,
      atol=1e-6,
    )

  @parameterized.parameters("dense", "sparse")
  def test_elliptic_hessian_scale_invariance(self, jacobian):
    """Cone curvature is degree-0 homogeneous: scaling Jaref leaves Hessian unchanged."""
    mjm, mjd, m, _ = test_data.fixture(
      xml=f"""
      <mujoco>
        <option cone="elliptic" solver="Newton" jacobian="{jacobian}"/>
        <worldbody>
          <geom type="plane" size="1 1 .1" friction="1 0.01 0.001"/>
          <body pos="0 0 0.02"><freejoint/><geom type="box" size="0.04 0.05 0.03" mass="0.5"/></body>
        </worldbody>
      </mujoco>
      """
    )
    mjd.qvel[0] = 1.5
    mujoco.mj_forward(mjm, mjd)

    d = mjw.put_data(mjm, mjd)
    ctx = solver._create_solver_context(m, d)
    solver.init_context(m, d, ctx, grad=True)
    jaref = ctx.Jaref.numpy().copy()

    def _get_h(scale):
      ctx.Jaref = wp.array(jaref * scale, dtype=float, device=ctx.Jaref.device)
      solver._update_gradient(m, d, ctx)
      return ctx.h.numpy()[:, : m.nv, : m.nv].copy()

    np.testing.assert_allclose(_get_h(1e-8), _get_h(1.0), rtol=1e-5, atol=1e-6)

  @parameterized.parameters(
    (ConeType.PYRAMIDAL, SolverType.CG, 25, 5),
    (ConeType.PYRAMIDAL, SolverType.NEWTON, 2, 4),
  )
  def test_solve_batch(self, cone, solver_, iterations, ls_iterations):
    """Tests solve (batch)."""
    mjm0, mjd0, _, _ = test_data.fixture(
      "humanoid/humanoid.xml",
      keyframe=0,
      overrides={"opt.cone": cone, "opt.solver": solver_, "opt.iterations": iterations, "opt.ls_iterations": ls_iterations},
    )
    qacc_warmstart0 = mjd0.qacc_warmstart.copy()
    mujoco.mj_forward(mjm0, mjd0)
    mjd0.qacc_warmstart = qacc_warmstart0

    mjm1, mjd1, _, _ = test_data.fixture(
      "humanoid/humanoid.xml",
      keyframe=2,
      overrides={"opt.cone": cone, "opt.solver": solver_, "opt.iterations": iterations, "opt.ls_iterations": ls_iterations},
    )
    qacc_warmstart1 = mjd1.qacc_warmstart.copy()
    mujoco.mj_forward(mjm1, mjd1)
    mjd1.qacc_warmstart = qacc_warmstart1

    mjm2, mjd2, _, _ = test_data.fixture(
      "humanoid/humanoid.xml",
      keyframe=1,
      overrides={"opt.cone": cone, "opt.solver": solver_, "opt.iterations": iterations, "opt.ls_iterations": ls_iterations},
    )
    qacc_warmstart2 = mjd2.qacc_warmstart.copy()
    mujoco.mj_forward(mjm2, mjd2)
    mjd2.qacc_warmstart = qacc_warmstart2

    nefc_active = mjd0.nefc + mjd1.nefc + mjd2.nefc
    ne_active = mjd0.ne + mjd1.ne + mjd2.ne

    mjm, mjd, m, _ = test_data.fixture(
      "humanoid/humanoid.xml",
      overrides={"opt.cone": cone, "opt.solver": solver_, "opt.iterations": iterations, "opt.ls_iterations": ls_iterations},
    )
    d = mjw.put_data(mjm, mjd, nworld=3, njmax=2 * nefc_active)

    d.nefc = wp.array([nefc_active, nefc_active, nefc_active], dtype=wp.int32, ndim=1)
    d.ne = wp.array([ne_active, ne_active, ne_active], dtype=wp.int32, ndim=1)

    qacc_warmstart = np.vstack(
      [
        np.expand_dims(qacc_warmstart0, axis=0),
        np.expand_dims(qacc_warmstart1, axis=0),
        np.expand_dims(qacc_warmstart2, axis=0),
      ]
    )

    def _csr_M(mjmw, mjdw):
      return np.asarray(mjdw.M, dtype=np.float32)

    M = np.stack([_csr_M(mjm0, mjd0), _csr_M(mjm1, mjd1), _csr_M(mjm2, mjd2)])
    qacc_smooth = np.vstack(
      [
        np.expand_dims(mjd0.qacc_smooth, axis=0),
        np.expand_dims(mjd1.qacc_smooth, axis=0),
        np.expand_dims(mjd2.qacc_smooth, axis=0),
      ]
    )
    qfrc_smooth = np.vstack(
      [
        np.expand_dims(mjd0.qfrc_smooth, axis=0),
        np.expand_dims(mjd1.qfrc_smooth, axis=0),
        np.expand_dims(mjd2.qfrc_smooth, axis=0),
      ]
    )

    # Reshape the Jacobians
    efc_J0 = mjd0.efc_J.reshape((mjd0.nefc, mjm0.nv))
    efc_J1 = mjd1.efc_J.reshape((mjd1.nefc, mjm1.nv))
    efc_J2 = mjd2.efc_J.reshape((mjd2.nefc, mjm2.nv))

    if m.is_sparse:
      nv = m.nv
      njmax = d.njmax
      J_rownnz = np.zeros((3, njmax), dtype=np.int32)
      J_rowadr = np.zeros((3, njmax), dtype=np.int32)
      J_colind = np.zeros((3, 1, njmax * nv), dtype=np.int32)
      J_vals = np.zeros((3, 1, njmax * nv), dtype=np.float32)

      for w, efc_Jw, nefc_w in [(0, efc_J0, mjd0.nefc), (1, efc_J1, mjd1.nefc), (2, efc_J2, mjd2.nefc)]:
        offset = 0
        for i in range(nefc_w):
          cols = np.nonzero(efc_Jw[i])[0]
          J_rowadr[w, i] = offset
          J_rownnz[w, i] = len(cols)
          J_colind[w, 0, offset : offset + len(cols)] = cols
          J_vals[w, 0, offset : offset + len(cols)] = efc_Jw[i, cols]
          offset += len(cols)

      d.efc.J = wp.from_numpy(J_vals, dtype=wp.float32)
      d.efc.J_rownnz = wp.from_numpy(J_rownnz, dtype=wp.int32)
      d.efc.J_rowadr = wp.from_numpy(J_rowadr, dtype=wp.int32)
      d.efc.J_colind = wp.from_numpy(J_colind, dtype=wp.int32)
    else:
      efc_J_fill = np.zeros((3, d.njmax_pad, m.nv_pad))
      efc_J_fill[0, : mjd0.nefc, : mjm0.nv] = efc_J0
      efc_J_fill[1, : mjd1.nefc, : mjm1.nv] = efc_J1
      efc_J_fill[2, : mjd2.nefc, : mjm2.nv] = efc_J2
      d.efc.J = wp.from_numpy(efc_J_fill, dtype=wp.float32)

    # Similarly for D and aref values
    efc_D0 = mjd0.efc_D[: mjd0.nefc]
    efc_D1 = mjd1.efc_D[: mjd1.nefc]
    efc_D2 = mjd2.efc_D[: mjd2.nefc]

    efc_D_fill = np.zeros((3, d.njmax))
    efc_D_fill[0, : mjd0.nefc] = efc_D0
    efc_D_fill[1, : mjd1.nefc] = efc_D1
    efc_D_fill[2, : mjd2.nefc] = efc_D2

    efc_aref0 = mjd0.efc_aref[: mjd0.nefc]
    efc_aref1 = mjd1.efc_aref[: mjd1.nefc]
    efc_aref2 = mjd2.efc_aref[: mjd2.nefc]

    efc_aref_fill = np.zeros((3, d.njmax))
    efc_aref_fill[0, : mjd0.nefc] = efc_aref0
    efc_aref_fill[1, : mjd1.nefc] = efc_aref1
    efc_aref_fill[2, : mjd2.nefc] = efc_aref2

    d.qacc_warmstart = wp.from_numpy(qacc_warmstart, dtype=wp.float32)
    d.M = wp.from_numpy(M, dtype=wp.float32)
    d.qacc_smooth = wp.from_numpy(qacc_smooth, dtype=wp.float32)
    d.qfrc_smooth = wp.from_numpy(qfrc_smooth, dtype=wp.float32)
    d.efc.D = wp.from_numpy(efc_D_fill, dtype=wp.float32)
    d.efc.aref = wp.from_numpy(efc_aref_fill, dtype=wp.float32)

    if solver_ == SolverType.CG:
      m0 = mjw.put_model(mjm0)
      d0 = mjw.put_data(mjm0, mjd0)
      mjw.factor_m(m0, d0)
      qLD0 = d0.qLD.numpy()

      m1 = mjw.put_model(mjm1)
      d1 = mjw.put_data(mjm1, mjd1)
      mjw.factor_m(m1, d1)
      qLD1 = d1.qLD.numpy()

      m2 = mjw.put_model(mjm2)
      d2 = mjw.put_data(mjm2, mjd2)
      mjw.factor_m(m2, d2)
      qLD2 = d2.qLD.numpy()

      qLD = np.vstack([qLD0, qLD1, qLD2])
      d.qLD = wp.from_numpy(qLD, dtype=wp.float32)

    d.qacc.fill_(wp.inf)
    d.qfrc_constraint.fill_(wp.inf)
    d.efc.force.fill_(wp.inf)
    solver.solve(m, d)

    def cost(m, d, qacc):
      jaref = np.zeros(d.nefc, dtype=float)
      cost = np.zeros(1)
      mujoco.mj_mulJacVec(m, d, jaref, qacc)
      mujoco.mj_constraintUpdate(m, d, jaref - d.efc_aref, cost, 0)
      return cost

    mj_cost0 = cost(mjm0, mjd0, mjd0.qacc)
    mjwarp_cost0 = cost(mjm0, mjd0, d.qacc.numpy()[0])
    self.assertLessEqual(mjwarp_cost0, mj_cost0 * 1.025)

    mj_cost1 = cost(mjm1, mjd1, mjd1.qacc)
    mjwarp_cost1 = cost(mjm1, mjd1, d.qacc.numpy()[1])
    self.assertLessEqual(mjwarp_cost1, mj_cost1 * 1.025)

    mj_cost2 = cost(mjm2, mjd2, mjd2.qacc)
    mjwarp_cost2 = cost(mjm2, mjd2, d.qacc.numpy()[2])
    self.assertLessEqual(mjwarp_cost2, mj_cost2 * 1.025)

    if m.opt.solver == SolverType.NEWTON:
      _assert_eq(d.qacc.numpy()[0], mjd0.qacc, "qacc0")
      _assert_eq(d.qacc.numpy()[1], mjd1.qacc, "qacc1")
      _assert_eq(d.qacc.numpy()[2], mjd2.qacc, "qacc2")

      _assert_eq(d.qfrc_constraint.numpy()[0], mjd0.qfrc_constraint, "qfrc_constraint0")
      _assert_eq(d.qfrc_constraint.numpy()[1], mjd1.qfrc_constraint, "qfrc_constraint1")
      _assert_eq(d.qfrc_constraint.numpy()[2], mjd2.qfrc_constraint, "qfrc_constraint2")

      # Get world 0 forces - equality constraints at start, inequality constraints later
      nieq0 = mjd0.nefc - mjd0.ne
      nieq1 = mjd1.nefc - mjd1.ne
      nieq2 = mjd2.nefc - mjd2.ne
      world0_eq_forces = d.efc.force.numpy()[0, : mjd0.ne]
      world0_ineq_forces = d.efc.force.numpy()[0, ne_active : ne_active + nieq0]
      world0_forces = np.concatenate([world0_eq_forces, world0_ineq_forces])
      _assert_eq(world0_forces, mjd0.efc_force, "efc_force0")

      # Get world 1 forces
      world1_eq_forces = d.efc.force.numpy()[1, : mjd1.ne]
      world1_ineq_forces = d.efc.force.numpy()[1, ne_active : ne_active + nieq1]
      world1_forces = np.concatenate([world1_eq_forces, world1_ineq_forces])
      _assert_eq(world1_forces, mjd1.efc_force, "efc_force1")

      # Get world 2 forces
      world2_eq_forces = d.efc.force.numpy()[2, : mjd2.ne]
      world2_ineq_forces = d.efc.force.numpy()[2, ne_active : ne_active + nieq2]
      world2_forces = np.concatenate([world2_eq_forces, world2_ineq_forces])
      _assert_eq(world2_forces, mjd2.efc_force, "efc_force2")

  @parameterized.product(
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_frictionloss(self, jacobian):
    """Tests solver with frictionloss."""
    for keyframe in range(3):
      overrides = {"opt.jacobian": jacobian}
      _, mjd, m, d = test_data.fixture(
        "constraints.xml",
        keyframe=keyframe,
        overrides=overrides,
      )
      mjw.solve(m, d)

      _assert_eq(d.nf.numpy()[0], mjd.nf, "nf")
      _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
      _assert_eq(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, "qfrc_constraint")
      _assert_eq(d.efc.force.numpy()[0, : mjd.nefc], mjd.efc_force, "efc_force")

  @parameterized.product(
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    condim=(3, 4, 6),
    jacobian=(
      mujoco.mjtJacobian.mjJAC_DENSE,
      mujoco.mjtJacobian.mjJAC_SPARSE,
    ),
    nworld=(1, 2),
  )
  def test_noslip(self, cone, condim, jacobian, nworld):
    """Tests noslip solver against MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option noslip_iterations="3" noslip_tolerance="0" impratio="3"
                gravity="3 0 -9.81"/>
        <worldbody>
          <geom type="plane" size="5 5 .1" condim="{condim}" friction="1 .05 .01"/>
          <body pos="0 0 .099">
            <freejoint/>
            <geom size=".1" condim="{condim}" friction="1 .05 .01"/>
          </body>
          <body pos="1 0 1">
            <joint axis="0 1 0" frictionloss=".5"/>
            <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
          </body>
          <body pos="-1 0 1">
            <joint type="slide" frictionloss="2"/>
            <geom size=".05"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="0 0 0 0 0 0 2 0"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      overrides={"opt.jacobian": jacobian, "opt.cone": cone},
      nworld=nworld,
    )
    mjds = _perturb_worlds(mjm, mjd, d, nworld, [0.3, -0.2, 0.0, 0.0, 0.0, 0.5, -1.0, 0.5])

    # main solver forces without noslip, used to check that noslip has an effect
    mjm_slip = _noslip_model(mjm, noslip_iterations=0)
    mjd_slip = mujoco.MjData(mjm_slip)
    mujoco.mj_resetDataKeyframe(mjm_slip, mjd_slip, 0)
    mujoco.mj_forward(mjm_slip, mjd_slip)

    self.assertGreater(np.abs(mjd.qacc - mjd_slip.qacc).max(), 1e-6)

    d.nefc.fill_(-1)
    d.qacc.fill_(wp.inf)
    d.qfrc_constraint.fill_(wp.inf)
    d.efc.force.fill_(wp.inf)
    d.efc.Ma.fill_(wp.inf)
    d.efc.state.fill_(-1)
    mjw.forward(m, d)

    for worldid, mjd_world in enumerate(mjds):
      nefc = d.nefc.numpy()[worldid]
      self.assertEqual(nefc, mjd_world.nefc)
      _assert_eq(d.qacc.numpy()[worldid], mjd_world.qacc, f"qacc{worldid}", tol=_NOSLIP_TOLERANCE)
      _assert_eq(
        d.qfrc_constraint.numpy()[worldid],
        mjd_world.qfrc_constraint,
        f"qfrc_constraint{worldid}",
        tol=_NOSLIP_TOLERANCE,
      )
      _assert_eq(
        np.sort(d.efc.force.numpy()[worldid, :nefc]),
        np.sort(mjd_world.efc_force),
        f"efc_force{worldid}",
        tol=_NOSLIP_TOLERANCE,
      )
      _assert_eq(d.efc.Ma.numpy()[worldid], _mul_m(mjm, mjd_world, mjd_world.qacc), f"Ma{worldid}", tol=_NOSLIP_TOLERANCE)
      efc_state = _noslip_states(
        d.efc.type.numpy()[worldid, :nefc],
        d.efc.id.numpy()[worldid, :nefc],
        d.efc.force.numpy()[worldid, :nefc],
        d.efc.state.numpy()[worldid, :nefc],
        d.contact.friction.numpy(),
      )
      mjd_efc_state = _noslip_states(
        mjd_world.efc_type,
        mjd_world.efc_id,
        mjd_world.efc_force,
        mjd_world.efc_state,
        mjd_world.contact.friction,
      )
      np.testing.assert_array_equal(efc_state, mjd_efc_state)

    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.product(
    condim=(3, 4, 6),
    jacobian=(
      mujoco.mjtJacobian.mjJAC_DENSE,
      mujoco.mjtJacobian.mjJAC_SPARSE,
    ),
    nworld=(1, 2),
  )
  def test_noslip_elliptic_singular(self, condim, jacobian, nworld):
    """Tests noslip solver with a singular elliptic friction block against MuJoCo."""
    # the sphere only translates in x and z: the frictional rows other than the x-tangent have
    # zero Jacobian, so their A diagonal is clamped to 1e-10 and the (scaled) friction block fails
    # the SPD check in the QCQP (det or Cholesky pivot < 1e-10), which returns a zero force
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option noslip_iterations="3" noslip_tolerance="0" impratio="3" cone="elliptic"
                gravity="3 0 -9.81"/>
        <worldbody>
          <geom type="plane" size="5 5 .1" condim="{condim}" friction="1 .05 .01"/>
          <body pos="0 0 .099">
            <joint type="slide" axis="1 0 0"/>
            <joint type="slide"/>
            <geom size=".1" condim="{condim}" friction="1 .05 .01"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="1 0"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      overrides={"opt.jacobian": jacobian},
      nworld=nworld,
    )
    mjds = _perturb_worlds(mjm, mjd, d, nworld, [-0.5, 0.0])

    self.assertEqual(mjd.ncon, 1)
    self.assertEqual(mjd.nefc, condim)
    self.assertGreater(mjd.efc_force[0], 0.0)

    # check that the friction block of A = J M^-1 J' is singular
    if mujoco.mj_isSparse(mjm):
      efc_J = np.zeros((mjd.nefc, mjm.nv))
      mujoco.mju_sparse2dense(efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
    else:
      efc_J = mjd.efc_J.reshape((mjd.nefc, mjm.nv))
    minv_jt = np.zeros((mjd.nefc, mjm.nv))
    mujoco.mj_solveM(mjm, mjd, minv_jt, efc_J)
    afriction = (efc_J @ minv_jt.T)[1:, 1:]
    self.assertLess(np.abs(np.diag(afriction)).min(), 1e-10)

    d.nefc.fill_(-1)
    d.qacc.fill_(wp.inf)
    d.qfrc_constraint.fill_(wp.inf)
    d.efc.force.fill_(wp.inf)
    d.efc.Ma.fill_(wp.inf)
    mjw.forward(m, d)

    for worldid, mjd_world in enumerate(mjds):
      nefc = d.nefc.numpy()[worldid]
      self.assertEqual(nefc, mjd_world.nefc)
      _assert_eq(d.qacc.numpy()[worldid], mjd_world.qacc, f"qacc{worldid}", tol=_NOSLIP_TOLERANCE)
      _assert_eq(
        d.qfrc_constraint.numpy()[worldid],
        mjd_world.qfrc_constraint,
        f"qfrc_constraint{worldid}",
        tol=_NOSLIP_TOLERANCE,
      )
      _assert_eq(d.efc.force.numpy()[worldid, :nefc], mjd_world.efc_force, f"efc_force{worldid}", tol=_NOSLIP_TOLERANCE)
      _assert_eq(d.efc.Ma.numpy()[worldid], _mul_m(mjm, mjd_world, mjd_world.qacc), f"Ma{worldid}", tol=_NOSLIP_TOLERANCE)

    if nworld == 2:
      self.assertFalse(np.allclose(d.efc.force.numpy()[0, :condim], d.efc.force.numpy()[1, :condim]))

  @parameterized.product(cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC), nworld=(1, 2))
  def test_noslip_converged(self, cone, nworld):
    """Tests noslip solver at convergence with redundant contacts (box on plane)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option noslip_iterations="200" noslip_tolerance="0" impratio="3"/>
        <worldbody>
          <geom type="plane" size="5 5 .1" condim="4"/>
          <body pos="0 0 .099">
            <freejoint/>
            <geom type="box" size=".1 .1 .1" condim="4"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel=".5 .2 0 0 0 1"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      overrides={"opt.cone": cone},
      nworld=nworld,
    )
    mjds = _perturb_worlds(mjm, mjd, d, nworld, [-0.3, 0.2, 0.0, 0.0, 0.0, -0.5])

    d.nefc.fill_(-1)
    d.qacc.fill_(wp.inf)
    d.qfrc_constraint.fill_(wp.inf)
    mjw.forward(m, d)

    # contact forces are not unique (4 contacts, 6 dofs), compare joint-space quantities, with
    # a looser tolerance for the slowly converging float32 Gauss-Seidel iterations
    for worldid, mjd_world in enumerate(mjds):
      self.assertEqual(d.nefc.numpy()[worldid], mjd_world.nefc)
      _assert_eq(d.qacc.numpy()[worldid], mjd_world.qacc, f"qacc{worldid}", tol=1e-2)
      _assert_eq(d.qfrc_constraint.numpy()[worldid], mjd_world.qfrc_constraint, f"qfrc_constraint{worldid}", tol=1e-2)

    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.product(
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    solver_=(SolverType.CG, SolverType.NEWTON),
    jacobian=(
      mujoco.mjtJacobian.mjJAC_DENSE,
      mujoco.mjtJacobian.mjJAC_SPARSE,
    ),
  )
  def test_noslip_step_batch(self, cone, solver_, jacobian):
    """Tests stepping with noslip for multiple worlds against MuJoCo."""
    nworld, nstep = 2, 20
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option noslip_iterations="3" noslip_tolerance="0" impratio="3"
                gravity="3 0 -9.81"/>
        <worldbody>
          <geom type="plane" size="5 5 .1" condim="6" friction="1 .05 .01"/>
          <body pos="0 0 .099">
            <freejoint/>
            <geom size=".1" condim="6" friction="1 .05 .01"/>
          </body>
          <body pos="1 0 1">
            <joint axis="0 1 0" frictionloss=".5"/>
            <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
          </body>
          <body pos="-1 0 1">
            <joint type="slide" frictionloss="2"/>
            <geom size=".05"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="0 0 0 0 0 0 2 0"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      overrides={
        "opt.jacobian": jacobian,
        "opt.cone": cone,
        "opt.solver": solver_,
        "opt.integrator": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
      },
      nworld=nworld,
    )

    # different initial velocity per world
    qvel = np.tile(mjd.qvel, (nworld, 1))
    qvel[1] *= 0.5
    d.qvel.assign(qvel)

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    for _ in range(nstep):
      wp.capture_launch(capture.graph)

    for worldid in range(nworld):
      mjd_world = mujoco.MjData(mjm)
      mujoco.mj_resetDataKeyframe(mjm, mjd_world, 0)
      mjd_world.qvel[:] = qvel[worldid]
      mujoco.mj_step(mjm, mjd_world, nstep)

      _assert_eq(d.qpos.numpy()[worldid], mjd_world.qpos, f"qpos{worldid}", tol=_NOSLIP_TOLERANCE)
      _assert_eq(d.qvel.numpy()[worldid], mjd_world.qvel, f"qvel{worldid}", tol=_NOSLIP_TOLERANCE)

    self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_noslip_sleep(self, nworld):
    """Tests noslip solver with sleeping enabled against MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option noslip_iterations="3" noslip_tolerance="0" impratio="3" cone="elliptic"
                integrator="implicitfast">
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="5 5 .1" condim="6"/>
          <body pos="0 0 .099">
            <freejoint/>
            <geom size=".1" condim="6"/>
          </body>
          <body pos="1 0 .099">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="1 .5 0 0 0 3 0 0 0 0 0 0"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      nworld=nworld,
    )
    mjds = _perturb_worlds(mjm, mjd, d, nworld, [-0.5, 0.0, 0.0, 0.0, 0.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    nstep = 50
    for _ in range(nstep):
      wp.capture_launch(capture.graph)

    for worldid, mjd_world in enumerate(mjds):
      mujoco.mj_step(mjm, mjd_world, nstep)
      np.testing.assert_array_equal(d.tree_awake.numpy()[worldid], mjd_world.tree_awake)
      _assert_eq(d.qpos.numpy()[worldid], mjd_world.qpos, f"qpos{worldid}", tol=_NOSLIP_TOLERANCE)
      _assert_eq(d.qvel.numpy()[worldid], mjd_world.qvel, f"qvel{worldid}", tol=_NOSLIP_TOLERANCE)

    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_noslip_tolerance(self, nworld):
    """Tests noslip solver early termination."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option noslip_iterations="50" noslip_tolerance="1e-4" impratio="3"
                gravity="3 0 -9.81"/>
        <worldbody>
          <geom type="plane" size="5 5 .1" friction="1 .05 .01"/>
          <body pos="0 0 .099">
            <freejoint/>
            <geom size=".1" friction="1 .05 .01"/>
          </body>
          <body pos="1 0 1">
            <joint axis="0 1 0" frictionloss=".5"/>
            <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
          </body>
          <body pos="-1 0 1">
            <joint type="slide" frictionloss="2"/>
            <geom size=".05"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="0 0 0 0 0 0 2 0"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      nworld=nworld,
    )
    mjds = _perturb_worlds(mjm, mjd, d, nworld, [0.3, -0.2, 0.0, 0.0, 0.0, 0.5, -1.0, 0.5])

    d.solver_niter.fill_(-1)
    d.qacc.fill_(wp.inf)
    mjw.forward(m, d)

    mjm_slip = _noslip_model(mjm, noslip_iterations=0)
    mjd_slip = mujoco.MjData(mjm_slip)
    mujoco.mj_resetDataKeyframe(mjm_slip, mjd_slip, 0)
    mujoco.mj_forward(mjm_slip, mjd_slip)
    noslip_niter = mjd.solver_niter[0] - mjd_slip.solver_niter[0]
    self.assertLess(noslip_niter, 50)

    for worldid, mjd_world in enumerate(mjds):
      self.assertEqual(d.solver_niter.numpy()[worldid], mjd_world.solver_niter[0])
      _assert_eq(d.qacc.numpy()[worldid], mjd_world.qacc, f"qacc{worldid}", tol=_NOSLIP_TOLERANCE)

    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.parameters(2, 3, 4, 5)
  def test_qcqp(self, n):
    """Tests the QCQP solvers against a reference implementation of mju_QCQP."""
    rng = np.random.default_rng(n)
    A, b, d, r = [], [], [], []
    expected = []
    for case in range(8):
      An = rng.normal(size=(n, n))
      An = An @ An.T + 0.1 * np.eye(n)
      bn = rng.normal(size=n)
      dn = rng.uniform(0.5, 1.5, size=n)
      xn = -np.linalg.solve(An, bn)
      rn = np.linalg.norm(xn / dn)
      if case == 0:
        # singular: fails the SPD check
        An = np.zeros((n, n))
        An[0, 0] = 1.0
      elif case % 2:
        # unconstrained minimum outside of the constraint set
        rn *= 0.3
      else:
        # unconstrained minimum inside of the constraint set
        rn *= 2.0

      A.append(np.pad(An, (0, 5 - n)))
      b.append(np.pad(bn, (0, 5 - n)))
      d.append(np.pad(dn, (0, 5 - n)))
      r.append(rn)
      expected.append(_qcqp_reference(An, bn, dn, rn))

    ncase = len(expected)
    A_in = wp.array(np.array(A), dtype=types.mat55)
    b_in = wp.array(np.array(b), dtype=types.vec5)
    d_in = wp.array(np.array(d), dtype=types.vec5)
    r_in = wp.array(np.array(r), dtype=float)
    res_out = [wp.zeros(ncase, dtype=types.vec5) for _ in range(3)]
    active_out = [wp.full(ncase, -1, dtype=int) for _ in range(3)]
    wp.launch(
      _qcqp_kernel,
      dim=ncase,
      inputs=[n, A_in, b_in, d_in, r_in],
      outputs=[res_out[0], active_out[0], res_out[1], active_out[1], res_out[2], active_out[2]],
    )

    # mju_QCQP2 for n = 2, mju_QCQP3 for n = 3, mju_QCQP for all n
    outputs = [(res_out[2], active_out[2])]
    if n == 2:
      outputs.append((res_out[0], active_out[0]))
    if n == 3:
      outputs.append((res_out[1], active_out[1]))

    for res, active in outputs:
      res = res.numpy()
      active = active.numpy()
      for case, (res_expected, active_expected) in enumerate(expected):
        self.assertEqual(active[case], int(active_expected), f"active{case}")
        _assert_eq(res[case, :n], res_expected, f"res{case}", tol=1e-3)
        np.testing.assert_array_equal(res[case, n:], 0.0)

        if case == 0:
          np.testing.assert_array_equal(res[case], 0.0)
        elif case % 2:
          # solution on the constraint boundary
          self.assertTrue(active[case])
          _assert_eq(np.linalg.norm(res[case, :n] / d[case][:n]), r[case], f"norm{case}", tol=1e-3)
        else:
          self.assertFalse(active[case])

  def test_incremental_vs_full_hessian(self):
    """Tests that incremental Hessian updates produce same result as full recomputation."""
    total_any_changes = False
    for keyframe in range(3):
      mjm, mjd, m, d = test_data.fixture(
        "humanoid/humanoid.xml",
        keyframe=keyframe,
        overrides={
          "opt.cone": ConeType.PYRAMIDAL,
          "opt.solver": SolverType.NEWTON,
          "opt.iterations": 5,
          "opt.ls_iterations": 10,
        },
      )

      def _run_solver(d, update_fn, track=False):
        """Run solver iterations with a given gradient update function."""
        d.qacc.zero_()
        d.qfrc_constraint.zero_()
        d.efc.force.zero_()
        ctx = solver._create_solver_context(m, d)
        solver.init_context(m, d, ctx, grad=True)
        any_changes = False
        ctx.search_unchanged.fill_(False)
        for _ in range(m.opt.iterations):
          solver._linesearch(m, d, ctx)
          if track:
            ctx.quad_changed_count.zero_()
          solver._update_constraint(m, d, ctx, track_changes=track)
          if track:
            wp.synchronize()
            if np.any(ctx.quad_changed_count.numpy() > 0):
              any_changes = True
          update_fn(m, d, ctx)
        return d.qacc.numpy().copy(), any_changes

      qacc_full, _ = _run_solver(mjw.put_data(mjm, mjd), solver._update_gradient)
      qacc_inc, any_changes = _run_solver(mjw.put_data(mjm, mjd), solver._update_gradient_incremental, track=True)
      total_any_changes = total_any_changes or any_changes

      _assert_eq(qacc_inc, qacc_full, f"qacc keyframe={keyframe}")

    self.assertTrue(total_any_changes, "no state changes detected across any keyframe")

  @parameterized.parameters(1, 3)
  def test_quad_changed_ids(self, nworld):
    """Rows whose bit is set are listed in row order, and the bits are cleared."""
    njmax = 100
    bits = solver._QUAD_CHANGED_BITS
    # first and last bit of a word, several words, a world without changed rows
    rows = ([0, 7, bits - 1, bits, 2 * bits - 1, 2 * bits, 3 * bits, njmax - 1], [], [5, 2 * bits + 2])[:nworld]
    flag = np.zeros((nworld, njmax // bits + 1), dtype=np.int32)
    for worldid, world_rows in enumerate(rows):
      for row in world_rows:
        flag[worldid, row // bits] |= 1 << (row % bits)

    count = wp.array([len(world_rows) for world_rows in rows], dtype=int)
    flag = wp.array(flag, dtype=int)
    ids = wp.full((nworld, njmax), -1, dtype=int)
    wp.launch(solver._quad_changed_ids, dim=nworld, inputs=[count], outputs=[flag, ids])

    ids = ids.numpy()
    for worldid, world_rows in enumerate(rows):
      self.assertEqual(ids[worldid, : len(world_rows)].tolist(), world_rows)
      np.testing.assert_array_equal(ids[worldid, len(world_rows) :], -1)
    np.testing.assert_array_equal(flag.numpy(), 0)

  @parameterized.product(nworld=(1, 2, 3), lanes=(1, 4, 16, 32), slots=(False, True))
  def test_qfrc_index(self, nworld, lanes, slots):
    """Each dof lists the groups that contain it in group order, and nothing else is written."""
    if wp.get_device().is_cpu and lanes > 1:
      self.skipTest("a block is one thread on the CPU")

    fixture = _sparse_groups_fixture(nworld)
    nv, nnz, ngroup_max = fixture["nv"], fixture["colind"].shape[2], fixture["jtdaj_adr"].shape[1]
    cursor = wp.full((nworld, nv), 7, dtype=int)
    chunk_count = wp.full((nworld, lanes), 7, dtype=int)
    group = wp.full((nworld, ngroup_max), wp.vec4i(-5), dtype=wp.vec4i)
    # a single element when the slots are not used
    slot_adr = wp.full((nworld, ngroup_max + 1 if slots else 1), -5, dtype=int)
    adr = wp.full((nworld, nv + 1), -5, dtype=int)
    entry = wp.full((nworld, nnz), wp.vec3i(-5), dtype=wp.vec3i)
    wp.launch(
      solver._qfrc_index,
      dim=(nworld, lanes),
      inputs=[
        wp.array(fixture[name], dtype=int) for name in ("jtdaj_adr", "jtdaj_nrow", "nblock", "rownnz", "rowadr", "colind")
      ],
      outputs=[cursor, chunk_count, group, slot_adr, adr, entry],
      block_dim=lanes,
    )

    np.testing.assert_array_equal(adr.numpy(), fixture["adr"])
    np.testing.assert_array_equal(entry.numpy(), fixture["entry"])
    np.testing.assert_array_equal(group.numpy(), fixture["group"])
    np.testing.assert_array_equal(slot_adr.numpy(), fixture["slot_adr"] if slots else -5)

  @parameterized.product(nworld=(1, 2, 3), compact=(False, True))
  def test_qfrc_constraint_sparse_groups(self, nworld, compact):
    """The sum over the lists of each dof matches efc_J.T @ efc_force and the default kernel."""
    fixture = _sparse_groups_fixture(nworld)
    nv, njmax, nnz = fixture["nv"], fixture["rownnz"].shape[1], fixture["colind"].shape[2]
    rng = np.random.default_rng(3)
    J = rng.uniform(-1.0, 1.0, (nworld, 1, nnz)).astype(np.float32)
    force = rng.uniform(-1.0, 1.0, (nworld, njmax)).astype(np.float32)
    force[rng.uniform(size=force.shape) < 0.2] = 0.0
    # the compacted solve drops dof 7, which is in most groups, and moves the others
    cdof = np.arange(nv, dtype=np.int32)
    if compact:
      cdof = np.concatenate((np.arange(1, 8), [-1], np.arange(8, nv - 1), [0])).astype(np.int32)
    nv_out = nv - int(compact)

    expected = np.zeros((nworld, nv_out))
    nefc, rownnz, rowadr, colind = fixture["nefc"], fixture["rownnz"], fixture["rowadr"], fixture["colind"]
    for worldid in range(nworld):
      for efcid in range(nefc[worldid]):
        for i in range(rownnz[worldid, efcid]):
          dofid = cdof[colind[worldid, 0, rowadr[worldid, efcid] + i]]
          if dofid >= 0:
            expected[worldid, dofid] += J[worldid, 0, rowadr[worldid, efcid] + i] * force[worldid, efcid]
    self.assertGreater(np.abs(expected[0]).max(), 0.5)

    done = wp.zeros(nworld, dtype=bool)
    arrays = {name: wp.array(fixture[name], dtype=int) for name in fixture if isinstance(fixture[name], np.ndarray)}
    J, force = wp.array(J, dtype=float), wp.array(force, dtype=float)
    cdof = wp.array(np.tile(cdof, (nworld, 1)), dtype=int)

    # the default kernel
    qfrc_default = wp.zeros((nworld, nv_out), dtype=float)
    wp.launch(
      solver._update_constraint_init_qfrc_constraint_sparse(compact, 0),
      dim=(nworld, njmax),
      inputs=[arrays[name] for name in ("nefc", "rownnz", "rowadr", "colind")] + [J, force, cdof, arrays["nefc"], done],
      outputs=[qfrc_default],
    )
    np.testing.assert_allclose(qfrc_default.numpy(), expected, rtol=1e-5, atol=1e-5)

    # the lists, then one thread per dof
    lanes = 1 if wp.get_device().is_cpu else solver._QFRC_INDEX_LANES
    group = wp.empty(fixture["jtdaj_adr"].shape, dtype=wp.vec4i)
    adr = wp.empty((nworld, nv + 1), dtype=int)
    entry = wp.empty((nworld, nnz), dtype=wp.vec3i)
    wp.launch(
      solver._qfrc_index,
      dim=(nworld, lanes),
      inputs=[arrays[name] for name in ("jtdaj_adr", "jtdaj_nrow", "nblock", "rownnz", "rowadr", "colind")],
      outputs=[
        wp.empty((nworld, nv), dtype=int),
        wp.empty((nworld, lanes), dtype=int),
        group,
        wp.empty((nworld, 1), dtype=int),
        adr,
        entry,
      ],
      block_dim=lanes,
    )
    qfrc = wp.full((nworld, nv_out), wp.inf, dtype=float)
    wp.launch(
      solver._update_constraint_init_qfrc_constraint_sparse(compact, nv, True, True),
      dim=(nworld, nv),
      inputs=[arrays["nf"], arrays["rowadr"], J, force, cdof, arrays["frictionadr"], adr, entry, arrays["nefc"], done],
      outputs=[qfrc],
    )
    np.testing.assert_allclose(qfrc.numpy(), expected, rtol=1e-5, atol=1e-5)

  @parameterized.parameters(1, 2)
  def test_qfrc_index_solve(self, nworld):
    """The lists of the solve hold every nonzero of the group head rows, in group order."""
    _, _, m, d = test_data.fixture(
      "constraints.xml",
      keyframe=1,
      nworld=nworld,
      overrides={"opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE, "opt.deterministic": types.DeterminismType.ATOMICS},
    )
    mjw.fwd_position(m, d)
    mjw.fwd_velocity(m, d)
    mjw.fwd_actuation(m, d)
    mjw.fwd_acceleration(m, d)
    ctx = solver._create_solver_context(m, d)
    solver.init_context(m, d, ctx, grad=True)

    adr, entry, group = ctx.qfrc_adr.numpy(), ctx.qfrc_entry.numpy(), ctx.jtdaj_group.numpy()
    nblock, jtdaj_adr, jtdaj_nrow = d.efc.jtdaj_nblock.numpy(), d.efc.jtdaj_adr.numpy(), d.efc.jtdaj_nrow.numpy()
    rownnz, rowadr, colind = d.efc.J_rownnz.numpy(), d.efc.J_rowadr.numpy(), d.efc.J_colind.numpy()
    self.assertGreater(nblock.min(), 1)
    self.assertGreater(jtdaj_nrow.max(), 1)
    for worldid in range(nworld):
      lists = [[] for _ in range(m.nv)]
      for groupid in range(nblock[worldid]):
        head_row = jtdaj_adr[worldid, groupid]
        head_adr, support = rowadr[worldid, head_row], rownnz[worldid, head_row]
        # the rows of a group follow each other in the sparse arrays
        self.assertEqual(group[worldid, groupid].tolist(), [head_adr, support, head_row, jtdaj_nrow[worldid, groupid]])
        for i in range(support):
          # the stride in the low 16 bits and the number of rows above them
          code = support + 65536 * jtdaj_nrow[worldid, groupid]
          lists[colind[worldid, 0, head_adr + i]].append([head_adr + i, head_row, code])
      self.assertEqual(adr[worldid, 0], 0)
      for dofid in range(m.nv):
        self.assertEqual(entry[worldid, adr[worldid, dofid] : adr[worldid, dofid + 1]].tolist(), lists[dofid])
      self.assertGreater(max(len(dof_list) for dof_list in lists), 1)

  def test_qfrc_constraint_sparse_packing(self):
    """A list entry holds a stride of 16 bits and a signed number of rows of 16 bits."""
    stride_max = solver._QFRC_ENTRY_STRIDE - 1
    nrow_max = solver._QFRC_ENTRY_STRIDE // 2 - 1
    njmax = nrow_max + 2
    rng = np.random.default_rng(0)
    force = rng.uniform(-1.0, 1.0, (1, njmax)).astype(np.float32)
    J = rng.uniform(-1.0, 1.0, (1, 1, 2 * stride_max + 16)).astype(np.float32)
    # the rows from the third one on do not follow each other in the sparse arrays
    rowadr = np.zeros((1, njmax), dtype=np.int32)
    rowadr[0, 2:] = 40000 + 2 * np.arange(njmax - 2)
    # two rows with the widest stride, the most rows one after the other, and the most rows that
    # do not follow each other: address of the dof in the head row, head row, stride and rows
    entry = [[5, 0, stride_max + 65536 * 2], [7, 2, 1 + 65536 * nrow_max], [40001, 2, 3 - 65536 * nrow_max]]
    rows = force[0, 2 : 2 + nrow_max].astype(np.float64)
    expected = [
      J[0, 0, 5] * force[0, 0] + J[0, 0, 5 + stride_max] * force[0, 1],
      np.dot(J[0, 0, 7 : 7 + nrow_max], rows),
      np.dot(J[0, 0, rowadr[0, 2 : 2 + nrow_max] + 1], rows),
    ]

    qfrc = wp.full((1, 3), wp.inf, dtype=float)
    wp.launch(
      solver._update_constraint_init_qfrc_constraint_sparse(False, 3, True, True),
      dim=(1, 3),
      inputs=[
        wp.zeros(1, dtype=int),
        wp.array(rowadr, dtype=int),
        wp.array(J, dtype=float),
        wp.array(force, dtype=float),
        wp.array(np.arange(3, dtype=np.int32)[None], dtype=int),
        wp.full((1, 3), -1, dtype=int),
        wp.array(np.arange(4, dtype=np.int32)[None], dtype=int),
        wp.array(np.array([entry], dtype=np.int32), dtype=wp.vec3i),
        wp.ones(1, dtype=int),
        wp.zeros(1, dtype=bool),
      ],
      outputs=[qfrc],
    )
    np.testing.assert_allclose(qfrc.numpy()[0], expected, rtol=1e-3, atol=1e-2)

  def test_qfrc_index_packing_limit(self):
    """Without room for the rows of a group in a list entry, the solve adds the rows in order."""
    qfrc = {}
    for njmax in (None, solver._QFRC_ENTRY_STRIDE // 2):
      _, _, m, d = test_data.fixture(
        "constraints.xml",
        keyframe=1,
        njmax=njmax,
        overrides={"opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE, "opt.deterministic": types.DeterminismType.ATOMICS},
      )
      mjw.fwd_position(m, d)
      mjw.fwd_velocity(m, d)
      mjw.fwd_actuation(m, d)
      mjw.fwd_acceleration(m, d)
      ctx = solver._create_solver_context(m, d)
      solver.init_context(m, d, ctx, grad=True)
      # no lists: one block per world adds the rows
      self.assertEqual(ctx.qfrc_adr is None, njmax is not None)
      self.assertEqual(ctx.jtdaj_slot is None, njmax is not None or wp.get_device().is_cpu)
      qfrc[njmax] = d.qfrc_constraint.numpy().copy()

    scale = np.abs(qfrc[None]).max()
    self.assertGreater(scale, 1.0e-3)
    np.testing.assert_allclose(qfrc[solver._QFRC_ENTRY_STRIDE // 2], qfrc[None], rtol=1e-4, atol=1e-5 * scale)

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE)
  def test_incremental_hessian_deterministic(self, jacobian):
    """Tests the incremental Hessian without atomics against a full rebuild."""
    any_changes = False
    for keyframe in range(3):
      _, _, m, d = test_data.fixture(
        "humanoid/humanoid.xml",
        keyframe=keyframe,
        nworld=2,
        overrides={
          "opt.jacobian": jacobian,
          "opt.cone": ConeType.PYRAMIDAL,
          "opt.solver": SolverType.NEWTON,
          "opt.iterations": 5,
          "opt.ls_iterations": 10,
          "opt.deterministic": types.DeterminismType.ATOMICS,
        },
      )

      # start away from the solution so that rows change state
      d.qacc.zero_()
      d.qfrc_constraint.zero_()
      d.efc.force.zero_()
      ctx = solver._create_solver_context(m, d)
      solver.init_context(m, d, ctx, grad=True)
      ctx.search_unchanged.fill_(False)
      for _ in range(m.opt.iterations):
        solver._linesearch(m, d, ctx)
        ctx.quad_changed_ids.fill_(-1)
        ctx.quad_changed_count.zero_()
        solver._update_constraint(m, d, ctx, track_changes=True)

        count = ctx.quad_changed_count.numpy()
        if m.is_sparse:
          # one bit per changed row, which the sparse incremental kernel reads and clears
          flag = ctx.quad_changed_flag.numpy()
          for worldid in range(d.nworld):
            self.assertEqual(sum(bin(word).count("1") for word in flag[worldid]), count[worldid])
        else:
          # the changed rows are listed in row order
          ids = ctx.quad_changed_ids.numpy()
          for worldid in range(d.nworld):
            changed = ids[worldid, : count[worldid]]
            self.assertTrue(np.all(changed >= 0) and np.all(np.diff(changed) > 0), f"changed rows: {changed}")
        any_changes = any_changes or bool(count.any())

        solver._update_gradient_incremental(m, d, ctx)
        np.testing.assert_array_equal(ctx.quad_changed_flag.numpy(), 0)
        h_inc = np.triu(ctx.h.numpy()[:, : m.nv, : m.nv])
        solver._update_gradient(m, d, ctx)
        h_full = np.triu(ctx.h.numpy()[:, : m.nv, : m.nv])
        np.testing.assert_allclose(h_inc, h_full, rtol=1e-4, atol=1e-5 * np.abs(h_full).max())

    self.assertTrue(any_changes, "no state changes detected across any keyframe")

  @parameterized.product(nworld=(1, 3), compact=(False, True))
  def test_jtdaj_slot_dofs(self, nworld, compact):
    """Each slot of the Hessian block of a group gets its element of H."""
    fixture = _sparse_groups_fixture(nworld)
    nv, ngroup_max = fixture["nv"], fixture["jtdaj_adr"].shape[1]
    # the compacted solve drops dof 7, which is in most groups, and moves the others
    cdof = np.arange(nv, dtype=np.int32)
    if compact:
      cdof = np.concatenate((np.arange(1, 8), [-1], np.arange(8, nv - 1), [0])).astype(np.int32)

    nslot = int(fixture["slot_adr"][:, ngroup_max].max()) + 3
    expected = np.full((nworld, nslot), -7, dtype=np.int32)
    for worldid in range(nworld):
      for groupid in range(min(fixture["nblock"][worldid], ngroup_max)):
        head_adr, support = fixture["group"][worldid, groupid, :2]
        slot = fixture["slot_adr"][worldid, groupid]
        for col in range(support):
          for row in range(col + 1):
            dof_row = cdof[fixture["colind"][worldid, 0, head_adr + row]]
            dof_col = cdof[fixture["colind"][worldid, 0, head_adr + col]]
            code = -1 if min(dof_row, dof_col) < 0 else (min(dof_row, dof_col) << 16) + max(dof_row, dof_col)
            expected[worldid, slot + col * (col + 1) // 2 + row] = code
    self.assertEqual(bool((expected == -1).any()), compact)

    slot_dof = wp.full((nworld, nslot), -7, dtype=int)
    wp.launch(
      solver._jtdaj_slot_dofs(compact),
      dim=(nworld, 3, solver._JTDAJ_THREADS_PER_GROUP),
      inputs=[
        wp.array(fixture["nblock"], dtype=int),
        wp.array(fixture["colind"], dtype=int),
        wp.array(np.tile(cdof, (nworld, 1)), dtype=int),
        wp.array(fixture["group"], dtype=wp.vec4i),
        wp.array(fixture["slot_adr"], dtype=int),
        3,
      ],
      outputs=[slot_dof],
    )
    np.testing.assert_array_equal(slot_dof.numpy(), expected)

    # one slot short of the world with most slots: that world is left alone
    total = fixture["slot_adr"][:, ngroup_max]
    short = int(total.max()) - 1
    slot_dof = wp.full((nworld, short), -7, dtype=int)
    wp.launch(
      solver._jtdaj_slot_dofs(compact),
      dim=(nworld, 3, solver._JTDAJ_THREADS_PER_GROUP),
      inputs=[
        wp.array(fixture["nblock"], dtype=int),
        wp.array(fixture["colind"], dtype=int),
        wp.array(np.tile(cdof, (nworld, 1)), dtype=int),
        wp.array(fixture["group"], dtype=wp.vec4i),
        wp.array(fixture["slot_adr"], dtype=int),
        3,
      ],
      outputs=[slot_dof],
    )
    expected = expected[:, :short].copy()
    expected[total > short] = -7
    np.testing.assert_array_equal(slot_dof.numpy(), expected)

  @parameterized.parameters(1, 2)
  def test_sparse_hessian_add_order(self, nworld):
    """The slots of an element of H are added in group order, and done worlds are left alone."""
    lanes = 1 if wp.get_device().is_cpu else 2 * solver._JTDAJ_THREADS_PER_GROUP
    rng = np.random.default_rng(nworld)
    nv = 12
    elements = [(row, col) for col in range(nv) for row in range(col + 1)]
    # slots of each group: some need several passes of the lanes, one has none
    sizes = [3, 70, 2, 0, 40, 1, 78, 33, 5, 64, 0, 65]
    slot_adr = np.concatenate(([0], np.cumsum(sizes))).astype(np.int32)
    nslot = int(slot_adr[-1])
    # slots after the last group hold a value that is not added
    slot_dof = np.zeros((nworld, nslot + 3), dtype=np.int32)
    slot = np.full((nworld, nslot + 3), 7.0, dtype=np.float32)
    expected = np.full((nworld, nv, nv), 0.5, dtype=np.float32)
    for worldid in range(nworld):
      for groupid, size in enumerate(sizes):
        # an element of H has a single slot in each group
        for i, element in enumerate(rng.permutation(len(elements))[:size]):
          row, col = elements[element]
          # magnitudes far apart: float32 addition in another order gives another result
          value = np.float32(rng.uniform(-1.0, 1.0) * 10.0 ** rng.integers(-3, 8))
          kind = rng.uniform()
          if kind < 0.1:
            value = np.float32(0.0)
          # a dof that the compacted solve drops
          code = -1 if kind > 0.9 else (row << 16) + col
          slot_dof[worldid, slot_adr[groupid] + i] = code
          slot[worldid, slot_adr[groupid] + i] = value
          if code >= 0:
            expected[worldid, row, col] += value
    # the last world reports more groups than the capacity
    nblock = np.full(nworld, len(sizes), dtype=np.int32)
    nblock[-1] += 4
    done = np.zeros(nworld, dtype=bool)

    for case in ("fits", "done", "short"):
      h = wp.full((nworld, nv, nv), 0.5, dtype=float)
      skip = wp.full(nworld, case == "short", dtype=bool)
      done[0] = case == "done"
      wp.launch(
        solver._JTDACJ_sparse_add,
        dim=(nworld, lanes),
        inputs=[
          wp.array(nblock, dtype=int),
          wp.array(np.tile(slot_adr, (nworld, 1)), dtype=int),
          wp.array(slot_dof, dtype=int),
          # one slot short of the groups: nothing is added
          wp.array(np.ascontiguousarray(slot[:, : nslot - 1]) if case == "short" else slot, dtype=float),
          wp.array(done, dtype=bool),
        ],
        outputs=[skip, h],
        block_dim=lanes,
      )
      result = expected.copy()
      if case == "short":
        result[:] = 0.5
      if case == "done":
        result[0] = 0.5
      np.testing.assert_array_equal(h.numpy(), result)
      # the worlds added here or done, which the kernel that adds group by group skips
      np.testing.assert_array_equal(skip.numpy(), case != "short")

  @parameterized.product(cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC), nworld=(1, 2))
  def test_sparse_hessian_two_kernels(self, cone, nworld):
    """The sparse Hessian from the slots of each group matches the one added group by group."""
    h = {}
    for deterministic in (types.DeterminismType.NONE, types.DeterminismType.ATOMICS):
      for keyframe in range(3):
        _, _, m, d = test_data.fixture(
          "constraints.xml",
          keyframe=keyframe,
          nworld=nworld,
          overrides={
            "opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE,
            "opt.cone": cone,
            "opt.deterministic": deterministic,
          },
        )
        mjw.fwd_position(m, d)
        mjw.fwd_velocity(m, d)
        mjw.fwd_actuation(m, d)
        mjw.fwd_acceleration(m, d)
        ctx = solver._create_solver_context(m, d)
        solver.init_context(m, d, ctx, grad=False)
        if deterministic:
          # with the slots, whatever the device and the number of worlds
          solver._qfrc_constraint_index(m, d, ctx, slots=True)

        def hessian(slots):
          if deterministic:
            ctx.jtdaj_slot = None if slots is None else wp.full((nworld, slots), wp.inf, dtype=float)
            ctx.jtdaj_skip = wp.zeros(nworld, dtype=bool)
          ctx.h.fill_(wp.inf)
          solver._update_gradient(m, d, ctx)
          return np.triu(ctx.h.numpy()[:, : m.nv, : m.nv]).copy()

        if not deterministic:
          h[keyframe] = hessian(None)
          continue

        nslot = ctx.jtdaj_slot_adr.numpy()[:, -1]
        self.assertGreater(nslot.min(), 10)
        # group by group
        h_groups = hessian(None)
        scale = np.abs(h[keyframe]).max()
        np.testing.assert_allclose(h_groups, h[keyframe], rtol=1e-4, atol=1e-5 * scale)
        # from the slots
        h_slots = hessian(int(nslot.max()))
        np.testing.assert_allclose(h_slots, h[keyframe], rtol=1e-4, atol=1e-5 * scale)
        np.testing.assert_allclose(h_slots, h_groups, rtol=1e-5, atol=1e-6 * scale)
        self.assertTrue(ctx.jtdaj_skip.numpy().all())
        self.assertTrue(np.isfinite(ctx.jtdaj_slot.numpy()[0, : nslot[0]]).all())
        # one slot short of the last world: it is added group by group, the others from the slots
        h_short = hessian(int(nslot[-1]) - 1)
        self.assertFalse(ctx.jtdaj_skip.numpy()[-1])
        self.assertTrue(np.isinf(ctx.jtdaj_slot.numpy()[-1]).all())
        np.testing.assert_array_equal(h_short[-1], h_groups[-1])
        np.testing.assert_array_equal(h_short[:-1], h_slots[:-1])

  @parameterized.product(nworld=(1, 3), compact=(False, True))
  def test_incremental_hessian_sparse_flags(self, nworld, compact):
    """The deterministic sparse kernel adds the rows whose bit is set, in row order."""
    nv, njmax = 9, 100
    bits = solver._QUAD_CHANGED_BITS
    rng = np.random.default_rng(5)
    # first and last bit of a word, several words, a world without changed rows
    rows = ([0, 7, bits - 1, bits, 2 * bits - 1, 2 * bits, 3 * bits, njmax - 1], [], [5, 2 * bits + 2])[:nworld]

    # every row has its own sparse support, so wide that its entries need more than one pass of
    # the lanes on the CPU and that the rows overlap
    rownnz = rng.integers(3, nv + 1, (nworld, njmax)).astype(np.int32)
    rowadr = np.zeros((nworld, njmax), dtype=np.int32)
    rowadr[:, 1:] = np.cumsum(rownnz, axis=1)[:, :-1]
    nnz = int(rownnz.sum(axis=1).max())
    colind = np.zeros((nworld, 1, nnz), dtype=np.int32)
    for worldid in range(nworld):
      for efcid in range(njmax):
        adr = rowadr[worldid, efcid]
        colind[worldid, 0, adr : adr + rownnz[worldid, efcid]] = rng.permutation(nv)[: rownnz[worldid, efcid]]
    J = rng.uniform(-1.0, 1.0, (nworld, 1, nnz)).astype(np.float32)
    D = rng.uniform(1.0, 2.0, (nworld, njmax)).astype(np.float32)
    quadratic = types.ConstraintState.QUADRATIC.value
    state = np.where(rng.uniform(size=(nworld, njmax)) < 0.5, quadratic, types.ConstraintState.SATISFIED.value)
    # the compacted solve drops dof 2 and moves the others
    cdof = np.arange(nv, dtype=np.int32)
    if compact:
      cdof = np.array([3, 0, -1, 1, 2, 4, 5, 7, 6], dtype=np.int32)
    h0 = rng.uniform(-1.0, 1.0, (nworld, nv, nv)).astype(np.float32)

    expected = h0.astype(np.float64)
    flag = np.zeros((nworld, njmax // bits + 1), dtype=np.int32)
    ids = np.full((nworld, njmax), -1, dtype=np.int32)
    for worldid, world_rows in enumerate(rows):
      ids[worldid, : len(world_rows)] = world_rows
      for efcid in world_rows:
        flag[worldid, efcid // bits] |= 1 << (efcid % bits)
        sign = D[worldid, efcid] if state[worldid, efcid] == quadratic else -D[worldid, efcid]
        adr = rowadr[worldid, efcid]
        for i in range(rownnz[worldid, efcid]):
          for j in range(i + 1):
            dofi, dofj = cdof[colind[worldid, 0, adr + i]], cdof[colind[worldid, 0, adr + j]]
            if dofi >= 0 and dofj >= 0:
              expected[worldid, min(dofi, dofj), max(dofi, dofj)] += sign * J[worldid, 0, adr + i] * J[worldid, 0, adr + j]
    self.assertGreater(np.abs(expected - h0).max(), 1.0)

    count = wp.array([len(world_rows) for world_rows in rows], dtype=int)
    inputs = [
      wp.array(rownnz, dtype=int),
      wp.array(rowadr, dtype=int),
      wp.array(colind, dtype=int),
      wp.array(J, dtype=float),
      wp.array(D, dtype=float),
      wp.array(state, dtype=int),
      wp.array(np.tile(cdof, (nworld, 1)), dtype=int),
    ]

    # the default kernel, from the list of rows
    h_default = wp.array(h0, dtype=float)
    wp.launch(
      solver._update_gradient_h_incremental_sparse(compact),
      dim=(nworld, 2, solver._JTDAJ_THREADS_PER_GROUP),
      inputs=inputs + [wp.array(ids, dtype=int), count, 2],
      outputs=[h_default],
    )
    np.testing.assert_allclose(h_default.numpy(), expected, rtol=1e-5, atol=1e-5)

    lanes = 1 if wp.get_device().is_cpu else 2 * solver._JTDAJ_THREADS_PER_GROUP
    flag = wp.array(flag, dtype=int)
    h = wp.array(h0, dtype=float)
    kernel = solver._update_gradient_h_incremental_sparse(compact, 0, lanes, True)
    wp.launch(kernel, dim=(nworld, lanes), inputs=inputs + [count], outputs=[flag, h], block_dim=lanes)
    np.testing.assert_allclose(h.numpy(), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(flag.numpy(), 0)

    # the sum of each element runs in row order: the result is the sequential float32 sum
    sequential = h0.copy()
    for worldid, world_rows in enumerate(rows):
      for efcid in world_rows:
        sign = D[worldid, efcid] if state[worldid, efcid] == quadratic else -D[worldid, efcid]
        adr = rowadr[worldid, efcid]
        for i in range(rownnz[worldid, efcid]):
          for j in range(i + 1):
            dofi, dofj = cdof[colind[worldid, 0, adr + i]], cdof[colind[worldid, 0, adr + j]]
            value = np.float32(np.float32(sign * J[worldid, 0, adr + i]) * J[worldid, 0, adr + j])
            if dofi >= 0 and dofj >= 0 and value != 0.0:
              sequential[worldid, min(dofi, dofj), max(dofi, dofj)] += value
    np.testing.assert_array_equal(h.numpy(), sequential)

  @parameterized.product(nworld=(1, 2), rows_from=("put_data", "make_constraint"))
  def test_sparse_hessian_dof_friction(self, nworld, rows_from):
    """Sparse Newton H, full and incremental, includes quadratic DOF and tendon friction rows."""
    mjm, mjd, m, _ = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="Newton" jacobian="sparse"/>
        <worldbody>
          <body>
            <joint name="j0" axis="0 1 0" frictionloss="0.5"/>
            <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
            <body pos=".3 0 0">
              <joint axis="0 1 0" frictionloss="2" range="-1 1"/>
              <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
              <body pos=".3 0 0">
                <joint axis="1 0 0" frictionloss="0.01"/>
                <joint name="j3" type="slide" axis="1 0 0"/>
                <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
              </body>
            </body>
          </body>
        </worldbody>
        <tendon>
          <fixed frictionloss="20">
            <joint joint="j0" coef="1"/>
            <joint joint="j3" coef="-0.5"/>
          </fixed>
        </tendon>
      </mujoco>
      """,
      nworld=nworld,
    )
    self.assertTrue(m.is_sparse)
    mjd.qpos[1] = 0.99  # activate the second joint's limit
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd, nworld=nworld)

    mjds = [mjd]
    if nworld == 2:
      # perturb world 1 so the two worlds have distinct Hessians, using the slide joint because
      # moving the root hinge only rotates the whole chain and leaves the Hessian unchanged
      mjd1 = copy.copy(mjd)
      mjd1.qpos[3] = 0.1
      mujoco.mj_forward(mjm, mjd1)
      qpos = d.qpos.numpy()
      qpos[1] = mjd1.qpos
      d.qpos.assign(qpos)
      mjw.kinematics(m, d)
      mjw.com_pos(m, d)
      mjw.crb(m, d)
      mjds.append(mjd1)

    if rows_from == "make_constraint":
      mjw.make_constraint(m, d)
    ctx = solver._create_solver_context(m, d)
    solver.init_context(m, d, ctx, grad=True)

    nefc = d.nefc.numpy()
    efc_type = d.efc.type.numpy()
    friction_dof, friction_tendon = types.ConstraintType.FRICTION_DOF, types.ConstraintType.FRICTION_TENDON
    for w in range(nworld):
      self.assertEqual(set(efc_type[w, : nefc[w]]), {friction_dof, friction_tendon, types.ConstraintType.LIMIT_JOINT})

    rownnz = d.efc.J_rownnz.numpy()
    rowadr = d.efc.J_rowadr.numpy()
    colind = d.efc.J_colind.numpy()[:, 0]
    J = d.efc.J.numpy()[:, 0]
    D = d.efc.D.numpy()

    def _check_h(state):
      h_actual = ctx.h.numpy()
      for w in range(nworld):
        h = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(mjm, mjds[w], h)
        for efcid in np.flatnonzero(state[w, : nefc[w]] == types.ConstraintState.QUADRATIC.value):
          row = np.zeros(m.nv)
          adr = rowadr[w, efcid]
          row[colind[w, adr : adr + rownnz[w, efcid]]] = J[w, adr : adr + rownnz[w, efcid]]
          h += D[w, efcid] * np.outer(row, row)
        np.testing.assert_allclose(np.triu(h_actual[w, : m.nv, : m.nv]), np.triu(h), rtol=1e-5, atol=1e-6)
      if nworld == 2:
        self.assertFalse(np.allclose(np.triu(h_actual[0, : m.nv, : m.nv]), np.triu(h_actual[1, : m.nv, : m.nv])))

    # full rebuild with every row quadratic
    state = np.full((nworld, d.njmax), types.ConstraintState.QUADRATIC.value, dtype=int)
    d.efc.state.assign(state)
    ctx.done.fill_(False)
    ctx.h.fill_(wp.inf)
    solver._update_gradient(m, d, ctx)
    _check_h(state)

    # incremental update: release the first DOF friction row and the tendon friction row
    ids = np.zeros(ctx.quad_changed_ids.shape, dtype=int)
    counts = np.zeros(nworld, dtype=int)
    for w in range(nworld):
      types_w = efc_type[w, : nefc[w]]
      changed = [np.flatnonzero(types_w == friction_dof)[0], np.flatnonzero(types_w == friction_tendon)[0]]
      state[w, changed] = types.ConstraintState.LINEARPOS.value
      ids[w, : len(changed)] = changed
      counts[w] = len(changed)
    d.efc.state.assign(state)
    ctx.quad_changed_ids.assign(ids)
    ctx.quad_changed_count.assign(counts)
    solver._update_gradient_incremental(m, d, ctx)
    _check_h(state)

  def test_qfrc_constraint_early_convergence(self):
    """Sparse qfrc_constraint must survive for a world that converges before the batch.

    The sparse solve clears qfrc_constraint each iteration and only rewrites
    unconverged worlds; a world that converges early must keep its value rather
    than be zeroed by a later iteration's clear. Uses CG, which has no post-solve
    recovery. World 0 rests on the floor (converges immediately, contacts active);
    world 1 is kicked so it takes several more iterations.
    """
    xml = """
    <mujoco>
      <option jacobian="sparse" solver="CG" iterations="20" ls_iterations="10"/>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 .1"/>
        <body pos="0 0 .05"><freejoint/><geom type="sphere" size=".05"/></body>
        <body pos=".3 0 .05"><freejoint/><geom type="sphere" size=".05"/></body>
        <body pos=".6 0 .05"><freejoint/><geom type="sphere" size=".05"/></body>
        <body pos=".9 0 .05"><freejoint/><geom type="sphere" size=".05"/></body>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    for _ in range(200):  # settle onto the floor: active contacts, warmstart near solution
      mujoco.mj_step(mjm, mjd)
    mujoco.mj_forward(mjm, mjd)
    mjd.qacc_warmstart = mjd.qacc

    m = mjw.put_model(mjm)
    self.assertTrue(m.is_sparse)
    d = mjw.put_data(mjm, mjd, nworld=2)

    qvel = d.qvel.numpy()
    qvel[1] += 5.0  # kick world 1 only
    d.qvel = wp.array(qvel, dtype=float)

    mjw.step(m, d)

    nefc = d.nefc.numpy()
    efc_force = d.efc.force.numpy()
    niter = d.solver_niter.numpy()
    qfrc = np.abs(d.qfrc_constraint.numpy()).sum(axis=1)

    # Precondition: both worlds have active constraint forces and converge at
    # different iteration counts (world 0 early), the setup that triggers the bug.
    for w in range(2):
      self.assertGreater(np.abs(efc_force[w, : nefc[w]]).sum(), 0.0, f"world {w} has no constraint force")
    self.assertLess(niter[0], niter[1], "worlds did not converge at staggered iterations")

    self.assertGreater(qfrc[0], 0.0, "early-converged world was zeroed")
    self.assertGreater(qfrc[1], 0.0)

  @parameterized.parameters(("dense", 1), ("sparse", 2))
  def test_discrete_cg_coupled_tendon(self, jacobian, nworld):
    """Discrete CG does not false-certify at 0 iterations on stiff coupled modes."""
    mjm, _, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option integrator="discrete" solver="CG" jacobian="{jacobian}"
                timestep="0.015625" gravity="0 0 0" tolerance="1e-5" iterations="50"/>
        <worldbody>
          <body>
            <joint name="j1" type="slide"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
          <body>
            <joint name="j2" type="slide"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
        </worldbody>
        <tendon>
          <fixed stiffness="1048576">
            <joint joint="j1" coef="1"/>
            <joint joint="j2" coef="-1"/>
          </fixed>
        </tendon>
        <equality>
          <joint joint1="j1" solimp="0.01 0.01 0.01" solref="0 -1e-6"/>
          <joint joint1="j2" solimp="0.01 0.01 0.01" solref="0 -1e-6"/>
        </equality>
      </mujoco>
      """,
      nworld=nworld,
    )

    mjm_ref = copy.deepcopy(mjm)
    mjm_ref.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mjm_ref.opt.tolerance = 0.0

    qfrc_list = [[0.0, 10.0 * (w + 1)] for w in range(nworld)]
    mjd_refs = []
    for w in range(nworld):
      ref_d = mujoco.MjData(mjm_ref)
      ref_d.qfrc_applied[:] = qfrc_list[w]
      mujoco.mj_forward(mjm_ref, ref_d)
      mjd_refs.append(ref_d)

    d.qfrc_applied.assign(wp.array(qfrc_list, dtype=float))
    d.qacc_warmstart.zero_()
    d.qacc.fill_(wp.inf)
    d.solver_niter.fill_(-1)
    mjw.forward(m, d)

    niter = d.solver_niter.numpy()
    qacc = d.qacc.numpy()
    for w in range(nworld):
      self.assertGreater(niter[w], 0)
      _assert_eq(qacc[w], mjd_refs[w].qacc, "qacc", tol=1e-4)

    if nworld > 1:
      self.assertFalse(np.allclose(qacc[0], qacc[1]))

    # exact warmstart still certifies in 0 iterations
    d.qacc_warmstart.assign(wp.array([ref.qacc for ref in mjd_refs], dtype=float))
    d.qacc.fill_(wp.inf)
    d.solver_niter.fill_(-1)
    mjw.forward(m, d)
    niter_warm = d.solver_niter.numpy()
    for w in range(nworld):
      self.assertEqual(niter_warm[w], 0)

  def test_solver_context_inplace_cholesky(self):
    """In-place blocked Cholesky aliases hfactor to h only for padded non-incremental Newton."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option cone="elliptic"/>
        <worldbody>
          <replicate count="7" offset="0.2 0 0">
            <body>
              <freejoint/>
              <geom size="0.05"/>
            </body>
          </replicate>
        </worldbody>
      </mujoco>
      """,
    )
    self.assertGreater(m.nv, solver._BLOCK_CHOLESKY_DIM)
    h_shape = (d.nworld, m.nv_pad, m.nv_pad)

    # Non-incremental Newton with nv > _BLOCK_CHOLESKY_DIM aliases hfactor to h only when requested.
    ctx_inplace = solver._create_solver_context(m, d, inplace_cholesky=True)
    ctx_default = solver._create_solver_context(m, d)
    self.assertEqual(ctx_inplace.hfactor.shape, h_shape)
    self.assertIs(ctx_inplace.hfactor, ctx_inplace.h)
    self.assertEqual(ctx_default.hfactor.shape, h_shape)
    self.assertIsNot(ctx_default.hfactor, ctx_default.h)

    # Incremental pyramidal Newton, CG, and small-nv solves never alias hfactor to h.
    m_pyr = dataclasses.replace(m, opt=dataclasses.replace(m.opt, cone=ConeType.PYRAMIDAL))
    ctx_pyr = solver._create_solver_context(m_pyr, d, inplace_cholesky=True)
    self.assertEqual(ctx_pyr.hfactor.shape, h_shape)
    self.assertIsNot(ctx_pyr.hfactor, ctx_pyr.h)

    m_cg = dataclasses.replace(m, opt=dataclasses.replace(m.opt, solver=SolverType.CG))
    ctx_cg = solver._create_solver_context(m_cg, d, inplace_cholesky=True)
    self.assertEqual(ctx_cg.h.shape, (d.nworld, 0, 0))
    self.assertEqual(ctx_cg.hfactor.shape, (d.nworld, 0, 0))

    m_small = dataclasses.replace(m, nv=solver._BLOCK_CHOLESKY_DIM)
    self.assertEqual(solver._create_solver_context(m_small, d, inplace_cholesky=True).hfactor.shape, (d.nworld, 0, 0))

    # Runtime guards reject missing augmented RHS padding and skip_unchanged with aliased buffers.
    m_nopad = dataclasses.replace(m, nv_pad=m.nv)
    for nvmax in (0, m.nv):
      with self.assertRaisesRegex(ValueError, "inplace_cholesky requires an augmented padding column"):
        solver._create_solver_context(m_nopad, dataclasses.replace(d, nvmax=nvmax), inplace_cholesky=True)
    ctx_compact = solver._create_solver_context(m_nopad, dataclasses.replace(d, nvmax=m.nv - 1), inplace_cholesky=True)
    self.assertIs(ctx_compact.hfactor, ctx_compact.h)

    with self.assertRaisesRegex(ValueError, "skip_unchanged requires separate h and hfactor buffers"):
      solver._cholesky_factorize_solve(m, d, ctx_inplace, skip_unchanged=True)

  @parameterized.product(
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
    compact=(False, True),
    nworld=(1, 2),
  )
  def test_solve_newton_elliptic_inplace_cholesky(self, jacobian, compact, nworld):
    """Multi-iteration elliptic Newton solve matches MuJoCo C and separate hfactor."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option cone="elliptic">
          <flag warmstart="disable"/>
        </option>
        <worldbody>
          <geom type="plane" size="5 5 0.1"/>
          <replicate count="7" offset="0.25 0 0">
            <body pos="-0.75 0 0.03" euler="8 -6 12">
              <freejoint/>
              <geom type="box" size="0.05 0.04 0.04" condim="6"/>
            </body>
          </replicate>
        </worldbody>
      </mujoco>
      """,
      qvel_noise=1.0,
      overrides={
        "opt.jacobian": jacobian,
        "opt.enableflags": types.EnableBit.SLEEP if compact else 0,
        "opt.deterministic": types.DeterminismType.ALL,
      },
      nworld=nworld,
      nvmax=42 if compact else -1,
    )
    mjds = _perturb_worlds(mjm, mjd, d, nworld, 0.5)

    d.qacc.fill_(wp.inf)
    d.qfrc_constraint.fill_(wp.inf)
    d.efc.force.fill_(wp.inf)
    d.solver_niter.fill_(-1)

    # Verify graph-captured solve / solve_compact parity against MuJoCo C.
    with wp.ScopedCapture() as capture:
      mjw.forward(m, d)
    wp.capture_launch(capture.graph)

    qacc = d.qacc.numpy()
    qfrc = d.qfrc_constraint.numpy()
    self.assertTrue(np.all(d.nefc.numpy() > 0))
    self.assertTrue(np.all(d.solver_niter.numpy() > 1))
    for w, mjd_w in enumerate(mjds):
      _assert_eq(qacc[w, : mjm.nv], mjd_w.qacc, f"qacc_world_{w}", tol=_TOLERANCE)
      _assert_eq(qfrc[w, : mjm.nv], mjd_w.qfrc_constraint, f"qfrc_constraint_world_{w}", tol=_TOLERANCE)
    if nworld == 2:
      self.assertFalse(np.allclose(qacc[0, : mjm.nv], qacc[1, : mjm.nv]))

    # Verify NaN-poisoned in-place Cholesky matches a separate hfactor buffer bitwise.
    if not compact and nworld == 1:
      ctx_sep = solver._create_solver_context(m, d, inplace_cholesky=False)
      solver._solve(m, d, ctx_sep)
      qacc_sep = d.qacc.numpy().copy()
      qfrc_sep = d.qfrc_constraint.numpy().copy()
      force_sep = d.efc.force.numpy().copy()

      d.qacc.fill_(wp.inf)
      d.qfrc_constraint.fill_(wp.inf)
      d.efc.force.fill_(wp.inf)
      ctx_inplace = solver._create_solver_context(m, d, inplace_cholesky=True)
      ctx_inplace.h.fill_(wp.nan)
      solver._solve(m, d, ctx_inplace)
      nefc0 = d.nefc.numpy()[0]
      np.testing.assert_array_equal(d.qacc.numpy()[:, : m.nv], qacc_sep[:, : m.nv])
      np.testing.assert_array_equal(d.qfrc_constraint.numpy()[:, : m.nv], qfrc_sep[:, : m.nv])
      np.testing.assert_array_equal(d.efc.force.numpy()[0, :nefc0], force_sep[0, :nefc0])


_FRICTION_CHAIN_XML = """
<mujoco>
  <option timestep="0.01" solver="Newton" cone="pyramidal" jacobian="dense">
    <flag warmstart="disable"/>
  </option>
  <worldbody>
    <body>
      <joint type="hinge" axis="0 1 0" frictionloss="1.5"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="1"/>
      <body pos="0.2 0 0">
        <joint type="hinge" axis="0 1 0" frictionloss="0.8"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" mass="0.5"/>
        <body pos="0.2 0 0">
          <joint type="hinge" axis="1 0 0" frictionloss="2.0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0.15 0 0" mass="0.3"/>
          <body pos="0.15 0 0">
            <joint type="hinge" axis="0 1 0" frictionloss="0.5"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.2"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class SolverFastPathTest(parameterized.TestCase):
  @parameterized.parameters(False, True)
  def test_fast_path_friction_transitions(self, compact):
    """qfrc_constraint must stay consistent when friction rows cross linear zones."""
    mjm = mujoco.MjModel.from_xml_string(_FRICTION_CHAIN_XML)
    if compact:
      # SLEEP routes every solve through solve_compact, even with nothing asleep
      mjm.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    nworld = 64
    d = mjw.put_data(mjm, mjd, nworld=nworld, nvmax=mjm.nv if compact else None)
    rng = np.random.default_rng(3)
    qvel = rng.uniform(-8.0, 8.0, size=(nworld, mjm.nv)).astype(np.float32)
    wp.copy(d.qvel, wp.array(qvel, dtype=float))

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    max_rel = 0.0
    cap_steps = 0
    for _ in range(20):
      wp.capture_launch(capture.graph)
      cap_steps += int(d.solver_niter.numpy().max() >= mjm.opt.iterations)
      nefc = d.nefc.numpy()
      J = d.efc.J.numpy()
      force = d.efc.force.numpy()
      got = d.qfrc_constraint.numpy()
      for w in range(nworld):
        n = nefc[w]
        if n == 0:
          continue
        ref = J[w, :n, : mjm.nv].astype(np.float64).T @ force[w, :n].astype(np.float64)
        scale = max(np.abs(ref).max(), 1.0)
        max_rel = max(max_rel, np.abs(ref - got[w]).max() / scale)
    self.assertLess(max_rel, 1e-4, f"qfrc_constraint inconsistent with J^T f: rel {max_rel:.3e}")
    # Friction chatter legitimately caps a couple of steps here even with full
    # per-iteration rebuilds; an exhausted stale ray that never re-anchors spins
    # to the cap on most steps (warmstart is off, so every solve starts far away).
    self.assertLess(cap_steps, 5, f"solver stalled: {cap_steps}/20 steps hit the iteration cap")


# Basic weld constraint model.
_WELD_XML = """
<mujoco>
  <worldbody>
    <body name="b1">
      <joint type="free"/>
      <geom size=".1"/>
    </body>
    <body name="b2" pos="1 0 0">
      <joint type="free"/>
      <geom size=".1"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="b1" body2="b2"/>
  </equality>
</mujoco>"""

# Mixed-constraint model (taken from mujoco's island_efc.xml C test).
_RICH_MODEL_XML = """
<mujoco>
  <default>
    <geom size=".1"/>
  </default>

  <worldbody>
    <body>
      <joint type="slide" axis="0 0 1" range="0 1" limited="true"/>
      <geom/>
    </body>

    <body pos=".25 0 0">
      <joint type="slide" axis="1 0 0"/>
      <geom/>
    </body>

    <body pos="0 0 0.25">
      <joint type="slide" axis="0 0 1"/>
      <geom/>
      <body pos="0 -.15 0">
        <joint name="hinge1" axis="0 1 0"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
        <body pos="-.2 0 0">
          <joint axis="0 1 0"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
        </body>
      </body>
    </body>

    <body pos=".5 0 0">
      <joint type="slide" axis="0 0 1" frictionloss="15"/>
      <geom type="box" size=".08 .08 .02" euler="0 10 0"/>
    </body>

    <body pos="-.5 0 0">
      <joint axis="0 1 0" frictionloss=".01"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 -.2 0 0"/>
    </body>

    <body pos="0 0 .5">
      <joint name="hinge2" axis="0 1 0"/>
      <geom type="box" size=".08 .02 .08"/>
    </body>

    <body pos=".5 0 .1">
      <freejoint/>
      <geom type="box" size=".03 .03 .03" pos="0.01 0.01 0.01"/>
    </body>

    <site name="0" pos="-.45 -.05 .35"/>
    <body pos="-.5 0 .3" name="connect">
      <freejoint/>
      <geom type="box" size=".05 .05 .05"/>
      <site name="1" pos=".05 -.05 .05"/>
    </body>
  </worldbody>

  <equality>
    <joint joint1="hinge1" joint2="hinge2"/>
    <connect body1="connect" body2="world" anchor="-.05 -.05 .05"/>
    <connect site1="0" site2="1"/>
  </equality>
</mujoco>"""

# Leg & hfield model (adapted from mujoco's 2humanoid100.xml C test).
_LEGS_HFIELD_XML = """
<mujoco>
  <option density="1.225" viscosity="1.8e-5" wind="0 0 1">
    <flag energy="enable"/>
  </option>

  <asset>
    <hfield name="hfield" nrow="3" ncol="3" size=".2 .2 .03 .03"
            elevation="1 0 1
                       0 1 0
                       1 0 1"/>
  </asset>

  <default>
    <joint armature="1" damping="10"/>
    <default class="hip0">
      <joint springref="30" stiffness="60"/>
    </default>
    <default class="hip1">
      <joint limited="true" range="-60 60" stiffness="10"/>
    </default>
  </default>

  <worldbody>
    <geom name="floor" type="plane" size="4 4 .1" margin="0.01" gap="0.005"/>
    <geom type="hfield" hfield="hfield" pos="-.4 .6 .05"/>
    <body name="head" pos="0 0 .7" gravcomp="0.5">
      <geom type="ellipsoid" size=".2 .2 .4" density="200"/>
      <freejoint/>
      <body euler="0 0 0" pos=".2 0 -.2">
        <joint name="hipz_0" class="hip1" axis="0 0 1"/>
        <joint name="hipy_0" class="hip0" axis="0 1 0"/>
        <geom type="capsule" size=".04" fromto="0 0 0 .2 0 -.25"/>
        <body pos=".2 0 -.25">
          <joint name="knee_0" axis="0 1 0"
                 limited="true" range="-160 -2" stiffness="40" springref="-30"/>
          <geom type="capsule" size=".03" fromto="0 0 0 -.2 0 -.25"/>
        </body>
      </body>
      <body euler="0 0 180" pos="-.2 0 -.2">
        <joint name="hipz_1" class="hip1" axis="0 0 1"/>
        <joint name="hipy_1" class="hip0" axis="0 1 0"/>
        <geom type="capsule" size=".04" fromto="0 0 0 .2 0 -.25"/>
        <body pos=".2 0 -.25">
          <joint name="knee_1" axis="0 1 0"
                 limited="true" range="-160 -2" stiffness="40" springref="-30"/>
          <geom type="capsule" size=".03" fromto="0 0 0 -.2 0 -.25"/>
        </body>
      </body>
    </body>
    <body name="box1" pos=".5 0 .1">
      <freejoint/>
      <geom type="box" size=".05 .05 .05"/>
    </body>
    <body name="box2" pos="-.5 0 .1">
      <freejoint/>
      <geom type="box" size=".05 .05 .05"/>
    </body>
  </worldbody>
</mujoco>"""


# An actuated 2-hinge arm (tree 0, dofs 0-1) plus two free bodies (trees 1 and 2).
_COMPACT_ARM_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body>
      <joint name="j0" type="hinge"/>
      <geom type="capsule" fromto="0 0 0 0 0 .3" size=".05"/>
      <body pos="0 0 .3">
        <joint name="j1" type="hinge"/>
        <geom type="capsule" fromto="0 0 0 0 0 .3" size=".05"/>
      </body>
    </body>
    <body pos="1 0 1"><freejoint/><geom name="ball0" type="sphere" size=".1"/></body>
    <body pos="2 0 1"><freejoint/><geom name="ball1" type="sphere" size=".1"/></body>
  </worldbody>
  <actuator>
    <motor joint="j0"/>
    <motor joint="j1"/>
  </actuator>
</mujoco>
"""


# Two free spheres resting (penetrating) on a floor, plus an actuated hinge arm.
# Generates contacts so the constrained solver does real work.
_COMPACT_CONTACT_XML = """
<mujoco>
  <option jacobian="sparse" solver="Newton" iterations="20"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body>
      <joint name="j0" type="hinge"/>
      <geom type="capsule" fromto="0 0 .5 .3 0 .5" size=".05"/>
    </body>
    <body pos="1 0 .08"><freejoint/><geom type="sphere" size=".1"/></body>
    <body pos="2 0 .08"><freejoint/><geom type="sphere" size=".1"/></body>
  </worldbody>
  <actuator><motor joint="j0"/></actuator>
</mujoco>
"""


def _put_compact(xml: str, nvmax: int | None = None, sparse: bool = False):
  """Build (mjm, mjd, m, d) with the compact workspace allocated via nvmax.

  The model has no SLEEP flag, so forward() runs the full baseline solve; passing nvmax
  still allocates the compact arrays (nvmax defaults to nv) so the compacted smooth/constrained
  solves can be invoked directly and compared against that baseline.
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  if sparse:
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nvmax=mjm.nv if nvmax is None else nvmax)
  return mjm, mjd, m, d


class CompactSolverTest(parameterized.TestCase):
  """Tests for the compacted smooth and constrained solves (solver.py compact path)."""

  def test_smooth_solve_equivalence_all_active(self):
    """With every tree active and nvmax=nv, compacted qacc_smooth matches baseline."""
    _, _, m, d = _put_compact(_COMPACT_ARM_XML, sparse=True)
    self.assertTrue(m.is_sparse)
    mjw.forward(m, d)
    baseline = d.qacc_smooth.numpy().copy()

    d.tree_awake = wp.array(np.ones((d.nworld, m.ntree), dtype=int), dtype=int)
    island.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], m.nv)

    solver.smooth_solve_compact(m, d)

    np.testing.assert_allclose(d.qacc_smooth.numpy(), baseline, rtol=1e-4, atol=1e-5)

  def test_smooth_solve_partial_active_freezes_rest(self):
    """Active trees match baseline (M is block-diagonal); inactive DOFs are frozen to 0."""
    _, _, m, d = _put_compact(_COMPACT_ARM_XML, sparse=True)
    mjw.forward(m, d)
    baseline = d.qacc_smooth.numpy().copy()

    # only the actuated arm tree (dofs 0-1) is active
    d.tree_awake = wp.array([[1, 0, 0]], dtype=int)
    island.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], 2)

    solver.smooth_solve_compact(m, d)

    out = d.qacc_smooth.numpy()
    np.testing.assert_allclose(out[0, :2], baseline[0, :2], rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(out[0, 2:], np.zeros(m.nv - 2))

  def test_constrained_solve_equivalence_all_active(self):
    """With every tree active and nvmax=nv, the compacted Newton solve matches baseline qacc."""
    for cone in ("pyramidal", "elliptic"):
      with self.subTest(cone=cone):
        xml = _COMPACT_CONTACT_XML.replace('iterations="20"', f'iterations="20" cone="{cone}"')
        _, _, m, d = _put_compact(xml)
        self.assertTrue(m.is_sparse)
        mjw.forward(m, d)  # full baseline solve (also builds efc.J, M)
        self.assertGreater(d.nacon.numpy()[0], 0)  # contacts exist
        baseline_qacc = d.qacc.numpy().copy()

        d.tree_awake = wp.array(np.ones((d.nworld, m.ntree), dtype=int), dtype=int)
        island.update_active_dofs(m, d)
        self.assertEqual(d.ncdof.numpy()[0], m.nv)

        solver.solve_compact(m, d)

        np.testing.assert_allclose(d.qacc.numpy(), baseline_qacc, rtol=1e-3, atol=1e-4)

  @parameterized.parameters(ConeType.PYRAMIDAL, ConeType.ELLIPTIC)
  def test_compact_hessian_dof_friction(self, cone):
    """With a tree asleep, the compact Hessian places DOF friction rows at compact DOFs."""
    # iterations="0" keeps qacc at the warmstart, which the compact solve also starts from
    # reference: the awake sphere alone, whose DOFs are the compact DOFs of the full model
    _, _, m_ref, d_ref = test_data.fixture(
      xml="""
      <mujoco>
        <option jacobian="sparse" solver="Newton" iterations="0"/>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .08">
            <joint type="free" frictionloss=".5"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
    )
    mjw.forward(m_ref, d_ref)
    ctx_ref = solver._create_solver_context(m_ref, d_ref)
    ctx_ref.h.fill_(wp.inf)
    solver.init_context(m_ref, d_ref, ctx_ref, grad=True)
    href = ctx_ref.h.numpy()[0]

    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option jacobian="sparse" solver="Newton" iterations="0"/>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body>
            <joint frictionloss="5"/>
            <geom type="capsule" fromto="0 0 .5 .3 0 .5" size=".05"/>
          </body>
          <body pos="0 0 .08">
            <joint type="free" frictionloss=".5"/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
    )
    mjw.forward(m, d)
    d.tree_awake.assign(np.array([[0, 1]], dtype=int))
    island.update_active_dofs(m, d)
    solver._compact_gather(m, d)

    # the compact (m, d) that solve_compact passes to the solver
    nvp = d.nvmax_pad
    m2 = dataclasses.replace(m, nv=nvp, nv_pad=nvp, is_sparse=False, dof_tri_row=d.cdof_tri_row, dof_tri_col=d.cdof_tri_col)
    efc2 = dataclasses.replace(d.efc, J=d.cJ, Ma=d.cMa)
    d2 = dataclasses.replace(
      d,
      M=d.cM,
      qfrc_smooth=d.cqfrc_smooth,
      qacc_smooth=d.cqacc_smooth,
      qacc_warmstart=d.cqacc_warmstart,
      qacc=d.cqacc,
      qfrc_constraint=d.cqfrc_constraint,
      efc=efc2,
    )
    wp.copy(d2.qacc, d2.qacc_warmstart)  # as _solve does
    ctx = solver._create_solver_context(m2, d2)
    ctx.compact_m_full = m
    ctx.compact_d_full = d
    ctx.h.fill_(wp.inf)
    solver.init_context(m2, d2, ctx, grad=True, compact=True)
    h = ctx.h.numpy()[0]

    nv = m_ref.nv
    self.assertEqual(d.ncdof.numpy()[0], nv)
    nefc = d_ref.nefc.numpy()[0]
    quadratic = d_ref.efc.state.numpy()[0, :nefc] == types.ConstraintState.QUADRATIC
    dof_friction = d_ref.efc.type.numpy()[0, :nefc] == types.ConstraintType.FRICTION_DOF
    self.assertTrue((quadratic & dof_friction).any())
    np.testing.assert_allclose(np.triu(h[:nv, :nv]), np.triu(href[:nv, :nv]), rtol=1e-4, atol=1e-5)

  def test_solve_compact_populates_islands(self):
    """When using the compact solver via mjw.solve, island mapping fields are updated."""
    mjm = mujoco.MjModel.from_xml_string(_COMPACT_CONTACT_XML)
    mjm.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP  # compact newton solver
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    # nv = 13. We set nvmax = 8 to size the compacted block below nv.
    d = mjw.put_data(mjm, mjd, nvmax=8)

    # Artificially set tree_island map on device
    d.tree_island = wp.array([[2, 2, 2]], dtype=int)
    d.nisland = wp.array([3], dtype=int)

    # Run solver which triggers solve_compact
    mjw.solve(m, d)

    # Assert that d.dof_island and d.efc.island are updated based on the new tree_island
    dof_island = d.dof_island.numpy()[0]
    np.testing.assert_array_equal(dof_island, [2] * m.nv)

    nefc = d.nefc.numpy()[0]
    efc_island = d.efc.island.numpy()[0]
    self.assertGreater(nefc, 0)
    np.testing.assert_array_equal(efc_island[:nefc], [2] * nefc)
    np.testing.assert_array_equal(efc_island[nefc:], [-1] * (d.njmax - nefc))

  def test_qfrc_constraint_initialization(self):
    """Verify that qfrc_constraint is initially zeroed when nefc==0."""
    xml = """
    <mujoco>
      <option solver="CG" jacobian="sparse"/>
      <worldbody>
        <geom type="plane" size="10 10 .001"/>
        <body name="sphere" pos="0 0 0.04">
          <freejoint/>
          <geom type="sphere" size="0.05" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml)

    # 1. Step once (starts in contact because Z=0.04 < radius=0.05)
    mjw.step(m, d)

    self.assertGreater(d.nefc.numpy()[0], 0, "Should have active constraints")
    self.assertGreater(np.max(np.abs(d.qfrc_constraint.numpy()[0])), 1.0, "Should have non-zero constraint forces")

    # 2. Teleport sphere up (no contacts)
    qpos_np = d.qpos.numpy()
    qpos_np[0, 2] = 1.0  # Move up to Z=1.0
    wp.copy(d.qpos, wp.array(qpos_np))
    d.qvel.zero_()

    # 3. Step again (will update nefc to 0 and should zero qfrc_constraint)
    mjw.step(m, d)

    self.assertEqual(d.nefc.numpy()[0], 0, "Should have no active constraints after teleport")
    qfrc_constraint_post = d.qfrc_constraint.numpy()[0]
    self.assertLess(
      np.max(np.abs(qfrc_constraint_post)),
      1e-5,
      f"qfrc_constraint should be zeroed, but got max abs: {np.max(np.abs(qfrc_constraint_post))}",
    )

  def test_discrete_cg_init_context_prec_fold(self):
    """Verifies solver.init_context folds constraint/contact metric into ctx.tw_diag for 1D rods."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG" timestep="0.005"/>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <flexcomp name="rod" type="grid" dim="1" count="3 1 1" spacing="0.05 0.05 1"
                    pos="0 0 0.002" radius="0.005" mass="0.05">
            <contact passive="true" selfcollide="none"/>
            <elasticity young="1e4" damping="1e-2"/>
          </flexcomp>
        </worldbody>
        <equality>
          <connect body1="rod_0" anchor="0 0 0"/>
        </equality>
      </mujoco>
      """,
      qvel_noise=0.2,
    )

    mjw.forward(m, d)

    ctx = solver._create_solver_context(m, d)
    self.assertIsNotNone(ctx.tw_diag)
    ctx.tw_diag.zero_()
    solver.init_context(m, d, ctx, grad=True)
    self.assertGreater(np.linalg.norm(ctx.tw_diag.numpy()), 0.0)

    # Also verify covered 3D flex triples fold passive-contact + efc diagonals
    # into 1 / total_diag and match C MuJoCo qacc under solver="CG".
    mjm3, mjd3, m3, d3 = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG" timestep="0.005" tolerance="1e-10" iterations="200"/>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <flexcomp name="cube" type="grid" dim="3" count="2 2 2" spacing="0.05 0.05 0.05"
                    pos="0 0 0.022" radius="0.006" mass="0.2">
            <contact passive="true" selfcollide="none"/>
            <elasticity young="2e4" poisson="0.3" damping="1e-2"/>
          </flexcomp>
        </worldbody>
        <equality>
          <connect body1="cube_0" anchor="0 0 0.022"/>
        </equality>
      </mujoco>
      """,
      qvel_noise=0.2,
    )
    mujoco.mj_forward(mjm3, mjd3)
    mjw.forward(m3, d3)
    ctx3 = solver._create_solver_context(m3, d3)
    self.assertIsNotNone(ctx3.tw_diag)
    solver.init_context(m3, d3, ctx3, grad=True)
    # Contact/equality DOFs have larger total diagonal -> strictly smaller 1/diag than qHDiagInv
    self.assertTrue(np.any(ctx3.tw_diag.numpy()[0] < d3.qHDiagInv.numpy()[0] - 1e-6))
    np.testing.assert_allclose(d3.qacc.numpy()[0], mjd3.qacc, atol=1e-3, rtol=1e-2)

  @parameterized.parameters(
    (1, "pyramidal"),
    (2, "pyramidal"),
    (2, "elliptic"),
  )
  def test_newton_sparse_tiles_blocked_cholesky(self, nworld, cone):
    """Verifies Newton blocked Cholesky with sparse-tile and diagonal-tile awareness (nv > 96)."""
    # 18 free spheres (nv = 108, ntiles = 7): first 3 spheres in ground contact + equality
    # constraint, remaining 15 spheres in free fall (purely diagonal tiles).
    # Parameterized over pyramidal (incremental skip_unchanged path) and elliptic (non-incremental
    # inplace_cholesky path).
    bodies_xml = []
    for i in range(18):
      z = 0.015 if i < 3 else 0.5
      bodies_xml.append(
        f'<body name="b{i}" pos="{i * 0.08} 0 {z}"><freejoint/><geom type="sphere" size="0.02" mass="0.1"/></body>'
      )
    xml = f"""
    <mujoco>
      <option integrator="Euler" solver="Newton" cone="{cone}" timestep="0.005" iterations="20" tolerance="1e-10">
        <flag warmstart="disable"/>
      </option>
      <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        {"".join(bodies_xml)}
      </worldbody>
      <equality>
        <connect body1="b0" body2="b2" anchor="0 0 0"/>
      </equality>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld, qvel_noise=0.5, nconmax=16, njmax=64)
    self.assertGreater(m.nv, solver._SPARSE_BLOCK_CHOLESKY_DIM)

    qpos_np = d.qpos.numpy().copy()
    qvel_np = d.qvel.numpy().copy()
    if nworld > 1:
      # Perturb world 1 so sphere 8 (in tile 3) is also in ground contact, producing distinct
      # per-world tile masks.
      qpos_np[1, 8 * 7 + 2] = 0.012
      qvel_np[1, 8 * 6 : 9 * 6] = np.array([-0.2, 0.15, -0.3, 0.4, -0.25, 0.1], dtype=np.float32)
      d.qpos.assign(qpos_np)
      d.qvel.assign(qvel_np)

    d.qacc.fill_(wp.inf)
    d.qfrc_constraint.fill_(wp.inf)
    mjw.forward(m, d)

    ctx = solver._create_solver_context(m, d, inplace_cholesky=True)
    self.assertIsNotNone(ctx.h_tile_nz)
    self.assertIsNotNone(ctx.h_row_mask)
    self.assertIsNotNone(ctx.h_col_mask)
    self.assertIsNotNone(ctx.h_tile_is_diag)
    self.assertIsNotNone(ctx.h_rhs_work)
    if cone == "elliptic":
      self.assertEqual(ctx.hfactor.ptr, ctx.h.ptr)
    else:
      self.assertNotEqual(ctx.hfactor.ptr, ctx.h.ptr)
    solver.init_context(m, d, ctx, grad=True)
    if nworld > 1:
      self.assertFalse(np.array_equal(ctx.h_tile_is_diag.numpy()[0], ctx.h_tile_is_diag.numpy()[1]))

    for w in range(nworld):
      mjd.qpos[:] = qpos_np[w]
      mjd.qvel[:] = qvel_np[w]
      mujoco.mj_forward(mjm, mjd)
      np.testing.assert_allclose(d.qacc.numpy()[w], mjd.qacc, atol=1e-3, rtol=1e-3)
      np.testing.assert_allclose(d.qfrc_constraint.numpy()[w], mjd.qfrc_constraint, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
  wp.init()
  absltest.main()
