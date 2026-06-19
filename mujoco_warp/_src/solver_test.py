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

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import ConeType
from mujoco_warp import SolverType
from mujoco_warp import test_data
from mujoco_warp._src import io
from mujoco_warp._src import island
from mujoco_warp._src import solver
from mujoco_warp._src import types

# tolerance for difference between MuJoCo and MJWarp solver calculations - mostly
# due to float precision
_TOLERANCE = 5e-3


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 20  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SolverTest(parameterized.TestCase):
  def test_M_fullm_upper_indices_are_row_sorted(self):
    """Sparse M seeding uses upper-triangle row-sorted writes."""
    _, _, m, _ = test_data.fixture("humanoid/humanoid.xml")

    lower_row = np.repeat(np.arange(m.nv), m.M_rownnz.numpy())
    lower_col = m.M_colind.numpy()
    upper_row = m.M_fullm_upper_i.numpy()
    upper_col = m.M_fullm_upper_j.numpy()
    upper_elemid = m.M_fullm_upper_elemid.numpy()

    self.assertEqual(upper_row.size, lower_row.size)
    self.assertTrue(np.all(upper_row <= upper_col))
    self.assertTrue(np.all(upper_row[:-1] <= upper_row[1:]))
    np.testing.assert_array_equal(upper_row, lower_col[upper_elemid])
    np.testing.assert_array_equal(upper_col, lower_row[upper_elemid])

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

    solver._linesearch(m, d, ctx)

    self.assertLess(d.qacc.numpy()[0, 0], -1.0)
    self.assertLess(ctx.improvement.numpy()[0], 0.001)
    self.assertGreater(ctx.improvement.numpy()[0], 0.0004)

  def test_island_linesearch_accepts_sub_float32_improvement(self):
    """Island line search should not lose small improvements on large absolute costs.

    Drives the island line-search kernel directly on a single island with one
    active inequality constraint. The island gauss cost carries a large constant
    (1e8) whose float32 resolution (~8) dwarfs the real improvement (0.5), so a
    line search that compares absolute costs rounds the improvement to zero and
    refuses to step. The shifted formulation evaluates the improvement directly
    and takes the Newton step.
    """
    # Scenario: 1 world, 1 island, 1 dof, 1 active inequality constraint.
    # Constraint: D=1, Jaref=-1, jv=1 -> active at alpha=0 (cost0=0.5), reaches
    # the boundary x=0 at the Newton step alpha=1, an improvement of 0.5.
    one2d_f = lambda v: wp.array([[v]], dtype=float)
    one2d_i = lambda v: wp.array([[v]], dtype=int)

    alpha_out = wp.array([[0.0]], dtype=float)
    wp.launch(
      solver._linesearch_kernel_island,
      dim=(1, 1),
      inputs=[
        # Model
        wp.array([1e-8], dtype=float),  # opt_tolerance
        wp.array([0.01], dtype=float),  # opt_ls_tolerance
        50,  # opt_ls_iterations
        wp.array([1.0], dtype=float),  # opt_impratio_invsqrt (unused, no elliptic)
        wp.array([1.0], dtype=float),  # stat_meaninertia
        # Data in
        wp.array([1], dtype=int),  # nefc
        wp.array([1], dtype=int),  # nisland
        wp.zeros(1, dtype=types.vec5),  # contact_friction (unused)
        wp.zeros(1, dtype=int),  # contact_dim (unused)
        wp.zeros((1, 3), dtype=int),  # contact_efc_address (unused)
        wp.array([1], dtype=int),  # nidof
        one2d_i(1),  # island_nv
        one2d_i(0),  # island_ne
        one2d_i(0),  # island_nf
        one2d_i(0),  # island_efcadr
        one2d_i(1),  # island_nefc
        one2d_i(0),  # map_efc2iefc (unused for inequality)
        1,  # njmax
        wp.array([0], dtype=int),  # nacon
        one2d_i(0),  # island_idofadr
        # In
        one2d_i(int(types.ConstraintType.LIMIT_JOINT)),  # iefc_type -> else (inequality)
        one2d_i(0),  # iefc_id
        one2d_f(1.0),  # iefc_D
        one2d_f(0.0),  # iefc_frictionloss
        one2d_f(-1.0),  # Jaref (active: < 0)
        one2d_f(1.0),  # jv
        one2d_f(0.0),  # mv
        one2d_f(0.0),  # search (gauss curvature 0)
        one2d_f(0.0),  # ifrc_smooth
        one2d_f(0.0),  # iMa
        one2d_f(1.0),  # island_search_dot
        one2d_f(1e8),  # island_gauss (large constant offset)
        wp.array([[False]], dtype=bool),  # island_done
      ],
      outputs=[alpha_out],  # island_alpha
    )

    # The minimizing step is alpha=1 (constraint boundary). Absolute-cost
    # bookkeeping loses the sub-resolution improvement and returns alpha=0.
    self.assertGreater(alpha_out.numpy()[0, 0], 0.5)

  @parameterized.parameters(
    (ConeType.PYRAMIDAL, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_DENSE, False),
    (ConeType.ELLIPTIC, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_DENSE, False),
    (ConeType.PYRAMIDAL, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_SPARSE, False),
    (ConeType.ELLIPTIC, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_SPARSE, False),
    (ConeType.PYRAMIDAL, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_DENSE, False),
    (ConeType.ELLIPTIC, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_DENSE, False),
    (ConeType.PYRAMIDAL, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_SPARSE, False),
    (ConeType.ELLIPTIC, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_SPARSE, False),
    # Island Solver Paths:
    (ConeType.PYRAMIDAL, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_DENSE, True),
    (ConeType.ELLIPTIC, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_DENSE, True),
    (ConeType.PYRAMIDAL, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_SPARSE, True),
    (ConeType.ELLIPTIC, SolverType.CG, 10, 5, mujoco.mjtJacobian.mjJAC_SPARSE, True),
    (ConeType.PYRAMIDAL, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_DENSE, True),
    (ConeType.ELLIPTIC, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_DENSE, True),
    (ConeType.PYRAMIDAL, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_SPARSE, True),
    (ConeType.ELLIPTIC, SolverType.NEWTON, 5, 10, mujoco.mjtJacobian.mjJAC_SPARSE, True),
  )
  def test_solve(self, cone, solver_, iterations, ls_iterations, jacobian, enable_islands):
    """Tests solve."""
    for keyframe in range(3):
      if enable_islands:
        io.ENABLE_ISLANDS = True
      try:
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
      finally:
        if enable_islands:
          io.ENABLE_ISLANDS = False

      if enable_islands:
        m.opt.disableflags &= ~types.DisableBit.ISLAND
        island.island(m, d)

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

    M0 = np.zeros((mjm0.nv, mjm0.nv))
    M1 = np.zeros((mjm1.nv, mjm1.nv))
    M2 = np.zeros((mjm2.nv, mjm2.nv))
    mujoco.mju_sym2dense(M0, mjd0.M, mjm0.M_rownnz, mjm0.M_rowadr, mjm0.M_colind)
    mujoco.mju_sym2dense(M1, mjd1.M, mjm1.M_rownnz, mjm1.M_rowadr, mjm1.M_colind)
    mujoco.mju_sym2dense(M2, mjd2.M, mjm2.M_rownnz, mjm2.M_rowadr, mjm2.M_colind)

    M = np.vstack(
      [
        np.expand_dims(M0, axis=0),
        np.expand_dims(M1, axis=0),
        np.expand_dims(M2, axis=0),
      ]
    )
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
    enable_islands=(True, False),
  )
  def test_frictionloss(self, jacobian, enable_islands):
    """Tests solver with frictionloss."""
    for keyframe in range(3):
      overrides = {"opt.jacobian": jacobian}
      if enable_islands:
        io.ENABLE_ISLANDS = True
      try:
        _, mjd, m, d = test_data.fixture(
          "constraints.xml",
          keyframe=keyframe,
          overrides=overrides,
        )
      finally:
        if enable_islands:
          io.ENABLE_ISLANDS = False
      if enable_islands:
        m.opt.disableflags &= ~types.DisableBit.ISLAND
        island.island(m, d)
      mjw.solve(m, d)

      _assert_eq(d.nf.numpy()[0], mjd.nf, "nf")
      _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
      _assert_eq(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, "qfrc_constraint")
      _assert_eq(d.efc.force.numpy()[0, : mjd.nefc], mjd.efc_force, "efc_force")

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
        wp.launch(solver._solve_init_search, dim=(d.nworld, m.nv), inputs=[ctx.Mgrad], outputs=[ctx.search, ctx.search_dot])
        any_changes = False
        for _ in range(m.opt.iterations):
          solver._linesearch(m, d, ctx)
          if track:
            ctx.changed_efc_count.zero_()
          solver._update_constraint(m, d, ctx, track_changes=track)
          if track:
            wp.synchronize()
            if np.any(ctx.changed_efc_count.numpy() > 0):
              any_changes = True
          update_fn(m, d, ctx)
          wp.launch(solver._solve_zero_search_dot, dim=(d.nworld), inputs=[ctx.done], outputs=[ctx.search_dot])
          wp.launch(
            solver._solve_search_update,
            dim=(d.nworld, m.nv),
            inputs=[m.opt.solver, ctx.Mgrad, ctx.search, ctx.beta, ctx.done],
            outputs=[ctx.search, ctx.search_dot],
          )
        return d.qacc.numpy().copy(), any_changes

      qacc_full, _ = _run_solver(mjw.put_data(mjm, mjd), solver._update_gradient)
      qacc_inc, any_changes = _run_solver(mjw.put_data(mjm, mjd), solver._update_gradient_incremental, track=True)
      total_any_changes = total_any_changes or any_changes

      _assert_eq(qacc_inc, qacc_full, f"qacc keyframe={keyframe}")

    self.assertTrue(total_any_changes, "no state changes detected across any keyframe")


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


class IslandSolverTest(parameterized.TestCase):
  """Tests for the parallel island solver."""

  def setUp(self):
    super().setUp()
    io.ENABLE_ISLANDS = True

  def tearDown(self):
    io.ENABLE_ISLANDS = False
    super().tearDown()

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
    """Tests solve parity with islands enabled."""
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

      # Run island execution
      m.opt.disableflags &= ~types.DisableBit.ISLAND
      island.island(m, d)

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

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE)
  def test_frictionloss(self, jacobian):
    """Tests solver with frictionloss under islands."""
    for keyframe in range(3):
      overrides = {"opt.jacobian": jacobian}
      _, mjd, m, d = test_data.fixture(
        "constraints.xml",
        keyframe=keyframe,
        overrides=overrides,
      )
      m.opt.disableflags &= ~types.DisableBit.ISLAND
      island.island(m, d)
      mjw.solve(m, d)

      _assert_eq(d.nf.numpy()[0], mjd.nf, "nf")
      _assert_eq(d.qacc.numpy()[0], mjd.qacc, "qacc")
      _assert_eq(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, "qfrc_constraint")
      _assert_eq(d.efc.force.numpy()[0, : mjd.nefc], mjd.efc_force, "efc_force")

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_single_island_weld(self, solver, jacobian):
    """Single island: weld constraint between two free bodies."""
    xml = _WELD_XML

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

    nefc = d_monolithic.nefc.numpy()[0]
    np.testing.assert_allclose(
      d_island.efc.force.numpy()[0, :nefc],
      d_monolithic.efc.force.numpy()[0, :nefc],
      atol=1e-4,
    )

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_multi_island_weld(self, solver, jacobian):
    """Two independent weld pairs form two separate islands."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="a1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b1" pos="10 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="11 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="a1" body2="a2"/>
        <weld body1="b1" body2="b2"/>
      </equality>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_three_islands(self, solver, jacobian):
    """Three independent weld pairs form three separate islands."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="a1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b1" pos="5 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="6 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="c1" pos="10 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="c2" pos="11 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="a1" body2="a2"/>
        <weld body1="b1" body2="b2"/>
        <weld body1="c1" body2="c2"/>
      </equality>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_contact_constraint(self, solver, jacobian):
    """Contact constraints from ground plane collision."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body name="box1" pos="0 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="1"/>
        </body>
        <body name="box2" pos="5 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="1"/>
        </body>
      </worldbody>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_contact_with_friction(self, solver, jacobian):
    """Contact constraints with friction (condim=3, pyramidal)."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body name="box" pos="0 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="3" friction="0.5"/>
        </body>
      </worldbody>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_friction_joint(self, solver, jacobian):
    """Hinge joint with frictionloss generates friction constraints."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="arm1">
          <joint type="hinge" axis="0 0 1" frictionloss="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.5 0 0" size=".05"/>
          <body name="arm2" pos="0.5 0 0">
            <joint type="hinge" axis="0 0 1" frictionloss="0.2"/>
            <geom type="capsule" fromto="0 0 0 0.5 0 0" size=".05"/>
          </body>
        </body>
      </worldbody>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

    nefc = d_monolithic.nefc.numpy()[0]
    np.testing.assert_allclose(
      d_island.efc.force.numpy()[0, :nefc],
      d_monolithic.efc.force.numpy()[0, :nefc],
      atol=1e-4,
    )

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_joint_limit(self, solver, jacobian):
    """Joint with active limits generates inequality constraints."""
    xml = """
    <mujoco>
      <compiler autolimits="true"/>

      <worldbody>
        <body name="arm">
          <joint type="hinge" axis="0 0 1" range="-30 30" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0.5 0 0" size=".05"/>
        </body>
      </worldbody>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_connect_constraint(self, solver, jacobian):
    """Connect (point) constraint between two bodies."""
    xml = """
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
        <connect body1="b1" body2="b2" anchor="0.5 0 0"/>
      </equality>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

    nefc = d_monolithic.nefc.numpy()[0]
    np.testing.assert_allclose(
      d_island.efc.force.numpy()[0, :nefc],
      d_monolithic.efc.force.numpy()[0, :nefc],
      atol=1e-4,
    )

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_mixed_constrained_unconstrained(self, solver, jacobian):
    """Mix of constrained bodies (forming islands) and unconstrained bodies."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="free1" pos="-5 0 1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="free2" pos="5 0 1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="a1" body2="a2"/>
      </equality>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_multi_world(self, solver, jacobian):
    """Island solver with multiple parallel worlds (nworld=4)."""
    xml = _WELD_XML

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      nworld=4,
      overrides={
        "opt.solver": solver,
        "opt.jacobian": jacobian,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(xml=xml, nworld=4)
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    for w in range(4):
      np.testing.assert_allclose(d_island.qacc.numpy()[w], d_monolithic.qacc.numpy()[w], atol=1e-4)
      np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[w], d_monolithic.qfrc_constraint.numpy()[w], atol=1e-4)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_warmstart_disabled(self, solver, jacobian):
    """Island solver with warmstart disabled."""
    xml = _WELD_XML

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
      "opt.disableflags": types.DisableBit.WARMSTART,
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_mujoco_c_parity(self, solver, jacobian):
    """Island solver qacc should be close to MuJoCo C reference."""
    xml = _WELD_XML

    mjm, mjd, m, d = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.jacobian": jacobian,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND

    # MuJoCo C reference
    mujoco.mj_forward(mjm, mjd)
    qacc_mjc = mjd.qacc.copy()

    # MuJoCo Warp island solver
    mjw.forward(m, d)
    qacc_warp = d.qacc.numpy()[0]

    np.testing.assert_allclose(
      qacc_warp,
      qacc_mjc,
      atol=1e-3,
      err_msg="qacc mismatch between island solver and MuJoCo C",
    )

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_multi_step(self, solver, jacobian):
    """Island solver produces stable multi-step simulation."""
    xml = _WELD_XML

    mjm, mjd, m, d_island = test_data.fixture(
      xml=xml,
      overrides={
        "opt.solver": solver,
        "opt.jacobian": jacobian,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND

    # Run 10 steps with island solver
    for _ in range(10):
      mjw.forward(m, d_island)
      mjw.step(m, d_island)

    qacc = d_island.qacc.numpy()[0]
    # Check that accelerations are finite and not NaN
    self.assertTrue(np.all(np.isfinite(qacc)), msg=f"Non-finite qacc after 10 steps: {qacc}")

    # Verify reasonable magnitude (shouldn't blow up)
    self.assertLess(np.max(np.abs(qacc)), 1e6, msg=f"qacc magnitude too large: {np.max(np.abs(qacc))}")

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_chain_single_island(self, solver, jacobian):
    """Three bodies in a chain form one island with two welds."""
    xml = """
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
        <body name="b3" pos="2 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="b1" body2="b2"/>
        <weld body1="b2" body2="b3"/>
      </equality>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

    nefc = d_monolithic.nefc.numpy()[0]
    np.testing.assert_allclose(
      d_island.efc.force.numpy()[0, :nefc],
      d_monolithic.efc.force.numpy()[0, :nefc],
      atol=1e-4,
    )

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_asymmetric_islands(self, solver, jacobian):
    """Islands of different sizes: one small (1 weld) and one large (chain)."""
    xml = """
    <mujoco>
      <worldbody>
        <body name="a1">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="a2" pos="1 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b1" pos="5 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="6 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b3" pos="7 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
        <body name="b4" pos="8 0 0">
          <joint type="free"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <equality>
        <weld body1="a1" body2="a2"/>
        <weld body1="b1" body2="b2"/>
        <weld body1="b2" body2="b3"/>
        <weld body1="b3" body2="b4"/>
      </equality>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_hinge_chain_with_limits(self, solver, jacobian):
    """Kinematic chain with hinge joints and active limits."""
    xml = """
    <mujoco>
      <compiler autolimits="true"/>

      <worldbody>
        <body name="link1">
          <joint type="hinge" axis="0 1 0" range="-45 45" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.5 0 0" size=".03"/>
          <body name="link2" pos="0.5 0 0">
            <joint type="hinge" axis="0 1 0" range="-45 45" damping="0.1"/>
            <geom type="capsule" fromto="0 0 0 0.5 0 0" size=".03"/>
          </body>
        </body>
      </worldbody>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_ball_joint_limit(self, solver, jacobian):
    """Ball joint with active limit generates inequality constraints."""
    xml = """
    <mujoco>
      <compiler autolimits="true"/>

      <worldbody>
        <body>
          <joint type="ball" range="0 30"/>
          <geom type="box" size=".1 .2 .3" pos=".1 .2 .3"/>
        </body>
      </worldbody>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_contact_elliptic(self, solver, jacobian):
    """Contact constraints with elliptic friction cone (condim=3)."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body name="box" pos="0 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="3" friction="0.5"/>
        </body>
      </worldbody>
    </mujoco>"""

    overrides = {
      "opt.cone": types.ConeType.ELLIPTIC,
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_contact_elliptic_condim4(self, solver, jacobian):
    """Elliptic friction cones with condim=4 (normal + 2 tangential + spin)."""
    xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="10 10 .01"/>
        <body name="box" pos="0 0 0.15">
          <joint type="free"/>
          <geom name="box" type="box" size=".1 .1 .1" condim="4" friction="0.5 0.3 0.01"/>
        </body>
      </worldbody>
    </mujoco>"""

    overrides = {
      "opt.cone": types.ConeType.ELLIPTIC,
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_contact_elliptic_multi_island(self, solver, jacobian):
    """Two separate bodies with elliptic contacts forming separate islands."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body name="box1" pos="0 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="3" friction="0.5"/>
        </body>
        <body name="box2" pos="5 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="3" friction="0.5"/>
        </body>
      </worldbody>
    </mujoco>"""

    overrides = {
      "opt.cone": types.ConeType.ELLIPTIC,
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.product(
    warmstart=[False, True],
    solver=list(types.SolverType),
    cone=list(types.ConeType),
  )
  def test_islands_equivalent_forward(self, warmstart, solver, cone):
    """Island vs. monolithic forward parity across solver/cone/warmstart combos.

    Mirrors MuJoCo C's IslandsEquivalentForward test: a single forward call
    comparing island and monolithic solvers across all combinations of
    solver type, cone type, and warmstart on a rich model with mixed
    constraint types.
    """
    xml = _RICH_MODEL_XML

    overrides = {
      "opt.solver": solver,
      "opt.cone": cone,
      "opt.iterations": 100,
      "opt.tolerance": "0",
      "opt.ls_iterations": 20,
    }
    if not warmstart:
      overrides["opt.disableflags"] = types.DisableBit.WARMSTART

    # Monolithic (islands disabled by default via DisableBit.ISLAND)
    overrides_monolithic = dict(overrides)
    _, _, m_monolithic, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides_monolithic,
    )
    m_monolithic.opt.disableflags |= types.DisableBit.ISLAND

    # step once to populate warmstart, then forward
    mjw.step(m_monolithic, d_monolithic)
    mjw.forward(m_monolithic, d_monolithic)

    # Island
    _, _, m_island, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m_island.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.step(m_island, d_island)
    mjw.forward(m_island, d_island)

    np.testing.assert_allclose(
      d_island.qacc.numpy()[0],
      d_monolithic.qacc.numpy()[0],
      atol=1e-2 if solver == types.SolverType.CG else 1e-3,
      rtol=1e-2,
    )

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_islands_equivalent(self, solver, jacobian):
    """Multi-step island vs. monolithic parity with synchronized state.

    Mirrors MuJoCo C's IslandsEquivalent test: runs multiple forward calls
    comparing island and monolithic solvers while keeping the state
    synchronized, verifying convergence agreement over time.
    """
    xml = _RICH_MODEL_XML

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 60,
      "opt.tolerance": "0",
      "opt.ls_iterations": 60,
    }

    _, _, m_monolithic, d_monolithic = test_data.fixture(xml=xml, overrides=overrides)
    m_monolithic.opt.disableflags |= types.DisableBit.ISLAND
    _, _, m_island, d_island = test_data.fixture(xml=xml, overrides=overrides)
    m_island.opt.disableflags &= ~types.DisableBit.ISLAND

    nv = m_monolithic.nv

    # Run 5 synchronized steps
    for step in range(5):
      # Synchronize state: copy monolithic qpos/qvel to island
      d_island.qpos.assign(d_monolithic.qpos)
      d_island.qvel.assign(d_monolithic.qvel)

      mjw.forward(m_monolithic, d_monolithic)
      mjw.forward(m_island, d_island)

      np.testing.assert_allclose(
        d_island.qacc.numpy()[0, :nv],
        d_monolithic.qacc.numpy()[0, :nv],
        atol=1e-2 if solver == types.SolverType.CG else 1e-3,
        rtol=1e-2,
        err_msg=f"qacc mismatch at step {step}, solver={solver.name}",
      )

      mjw.step(m_monolithic, d_monolithic)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
    keyframe=(0, 1, 2),
  )
  def test_constraints_xml_island_parity(self, solver, jacobian, keyframe):
    """Complex multi-island model: constraints.xml with 5-9 islands.

    Exercises tendon limits, tendon friction, joint equality, tendon equality,
    ball joint limits, connect/weld, and mixed contact dimensions (1/3/4/6).
    Keyframe 0: 5 islands, no contacts. Keyframe 1: 9 islands, frictionless
    and pyramidal contacts. Keyframe 2: 9 islands, joint limits active.
    """
    _, _, m, d_monolithic = test_data.fixture(
      "constraints.xml",
      keyframe=keyframe,
      overrides={
        "opt.solver": solver,
        "opt.jacobian": jacobian,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      "constraints.xml",
      keyframe=keyframe,
      overrides={
        "opt.solver": solver,
        "opt.jacobian": jacobian,
        "opt.iterations": 100,
        "opt.tolerance": "1e-10",
      },
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    qacc_island = d_island.qacc.numpy()[0]
    qacc_monolithic = d_monolithic.qacc.numpy()[0]
    scale = 0.5 * (np.linalg.norm(qacc_island) + np.linalg.norm(qacc_monolithic))
    rtol = 1e-3 if solver == types.SolverType.CG else 1e-4
    tol = max(scale * rtol, 1e-8)
    np.testing.assert_allclose(
      qacc_island,
      qacc_monolithic,
      atol=tol,
      err_msg=f"qacc mismatch: solver={solver.name}, jacobian={jacobian}, keyframe={keyframe}",
    )

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_tendon_limit_and_friction(self, solver, jacobian):
    """Tendon limits and friction exercise generic Jacobian scan in edge discovery."""
    xml = """
    <mujoco>
      <compiler autolimits="true"/>
      <worldbody>
        <body name="b1">
          <joint name="j1" type="hinge" axis="1 0 0"/>
          <geom size=".1"/>
        </body>
        <body name="b2" pos="1 0 0">
          <joint name="j2" type="hinge" axis="1 0 0"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <tendon>
        <fixed name="t1" range="-0.5 0.5" frictionloss="0.1">
          <joint joint="j1" coef="1"/>
          <joint joint="j2" coef="-1"/>
        </fixed>
      </tendon>
    </mujoco>"""

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

    nefc = d_monolithic.nefc.numpy()[0]
    np.testing.assert_allclose(
      d_island.efc.force.numpy()[0, :nefc],
      d_monolithic.efc.force.numpy()[0, :nefc],
      atol=1e-4,
    )

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_contact_elliptic_condim6(self, solver, jacobian):
    """Elliptic friction cones with condim=6 (normal + 2 tangential + torsional + 2 rolling)."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="10 10 .01"/>
        <body name="box" pos="0 0 0.15">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" condim="6" friction="0.5 0.3 0.01"/>
        </body>
      </worldbody>
    </mujoco>"""

    overrides = {
      "opt.cone": types.ConeType.ELLIPTIC,
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-3)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-3)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_island_solver_no_graph_conditional(self, solver, jacobian):
    """Island solver with graph_conditional=False uses Python for loop."""
    xml = _WELD_XML

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "1e-10",
    }

    _, _, m, d_monolithic = test_data.fixture(
      xml=xml,
      overrides=overrides,
    )
    m.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m, d_monolithic)

    _, _, m, d_island = test_data.fixture(
      xml=xml,
      overrides={**overrides, "opt.graph_conditional": False},
    )
    m.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m, d_island)

    np.testing.assert_allclose(d_island.qacc.numpy()[0], d_monolithic.qacc.numpy()[0], atol=1e-4)
    np.testing.assert_allclose(d_island.qfrc_constraint.numpy()[0], d_monolithic.qfrc_constraint.numpy()[0], atol=1e-4)

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
    cone=list(types.ConeType),
  )
  def test_constraint_update_island_parity(self, solver, jacobian, cone):
    """Per-island constraint force parity across sparsity and cone types.

    Mirrors MuJoCo C's ConstraintUpdateImpl test in
    engine_core_constraint_test.cc: simulates for several steps, then
    verifies that the island solver produces the same qacc and efc_state
    as the monolithic solver across all combinations of Jacobian sparsity
    and cone type.
    """
    # This is the island_efc.xml model used in the C test:
    # 7 equalities (1:joint, 6:connect), 1 limit, 2 friction constraints,
    # 1 unconstrained dof, 4 islands, 2 with non-contiguous dofs.
    xml = _RICH_MODEL_XML

    overrides = {
      "opt.solver": solver,
      "opt.cone": cone,
      "opt.jacobian": jacobian,
      "opt.iterations": 100,
      "opt.tolerance": "0",
      "opt.ls_iterations": 20,
    }
    _, _, m_monolithic, d_monolithic = test_data.fixture(xml=xml, overrides=overrides)
    m_monolithic.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m_monolithic, d_monolithic)

    _, _, m_island, d_island = test_data.fixture(xml=xml, overrides=overrides)
    m_island.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m_island, d_island)

    np.testing.assert_allclose(
      d_island.qacc.numpy()[0],
      d_monolithic.qacc.numpy()[0],
      atol=1e-3,
      err_msg=f"qacc mismatch: solver={solver.name}, jac={jacobian}, cone={cone.name}",
    )

  @parameterized.product(
    solver=list(types.SolverType),
    iterations=[30, 60],
  )
  def test_solver_model_island_parity(self, solver, iterations):
    """Rich model island parity with iteration count sweep.

    Mirrors MuJoCo C's IslandsEquivalent test in engine_solver_test.cc:
    uses a complex model with mesh, heightfield, fluid, gravcomp, and
    multiple legs. Tests at different iteration counts to verify
    convergence behavior. The C test uses model.xml with CG+Sparse.
    """
    xml = _LEGS_HFIELD_XML

    overrides_monolithic = {
      "opt.solver": solver,
      "opt.tolerance": "0",
      "opt.iterations": iterations,
      "opt.ls_iterations": iterations,
    }
    # Single forward to establish contact
    _, _, m_monolithic, d_monolithic = test_data.fixture(xml=xml, overrides=overrides_monolithic)
    m_monolithic.opt.disableflags |= types.DisableBit.ISLAND
    mjw.forward(m_monolithic, d_monolithic)

    _, _, m_island, d_island = test_data.fixture(xml=xml, overrides=overrides_monolithic)
    m_island.opt.disableflags &= ~types.DisableBit.ISLAND
    mjw.forward(m_island, d_island)

    np.testing.assert_allclose(
      d_island.qacc.numpy()[0],
      d_monolithic.qacc.numpy()[0],
      atol=1e-3,
      err_msg=f"qacc mismatch: solver={solver.name}, iters={iterations}",
    )

  @parameterized.product(
    solver=list(types.SolverType),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_solver_model_island_multi_step(self, solver, jacobian):
    """Multi-step island parity on a rich model with synchronized state.

    Mirrors MuJoCo C's IslandsEquivalent test multi-step pattern:
    runs multiple steps, synchronizing state between island and monolithic
    after each step to prevent divergence. Uses the same rich model with
    mesh, heightfield, fluid, and multiple contact islands.
    """
    xml = _LEGS_HFIELD_XML

    nsteps = 10
    nv = 26  # 6 (head freejoint) + 2*4 (legs) + 2*6 (boxes) = 26

    overrides = {
      "opt.solver": solver,
      "opt.jacobian": jacobian,
      "opt.iterations": 60,
      "opt.tolerance": "0",
    }

    _, _, m_monolithic, d_monolithic = test_data.fixture(xml=xml, overrides=overrides)
    m_monolithic.opt.disableflags |= types.DisableBit.ISLAND
    _, _, m_island, d_island = test_data.fixture(xml=xml, overrides=overrides)
    m_island.opt.disableflags &= ~types.DisableBit.ISLAND

    for step in range(nsteps):
      mjw.forward(m_monolithic, d_monolithic)
      mjw.forward(m_island, d_island)

      np.testing.assert_allclose(
        d_island.qacc.numpy()[0, :nv],
        d_monolithic.qacc.numpy()[0, :nv],
        atol=1e-2 if solver == types.SolverType.CG else 1e-3,
        rtol=1e-2,
        err_msg=f"qacc mismatch at step {step}: solver={solver.name}",
      )

      # Integrate monolithic, then sync state to island
      mjw.step(m_monolithic, d_monolithic)
      d_island.qpos.assign(d_monolithic.qpos)
      d_island.qvel.assign(d_monolithic.qvel)
      d_island.act.assign(d_monolithic.act)
      d_island.time.assign(d_monolithic.time)
      d_island.qacc_warmstart.assign(d_monolithic.qacc_warmstart)


if __name__ == "__main__":
  wp.init()
  absltest.main()
