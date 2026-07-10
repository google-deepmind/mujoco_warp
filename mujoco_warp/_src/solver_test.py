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
from mujoco_warp._src import island
from mujoco_warp._src import solver

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
    m.stat.meaninertia.fill_(1.0e9)
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

    solver._linesearch(m, d, ctx)

    self.assertGreater(d.qacc.numpy()[0, 0], 0.03)
    self.assertGreater(ctx.improvement.numpy()[0], 5.0e-4)

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

  def test_solve_elliptic_sparse_diagonal_inertia(self):
    """Sparse elliptic Newton solve over diagonal-inertia bodies must not NaN (jtcj sizing)."""
    n = 12  # 72 dofs -> sparse path
    bodies = "".join(
      f'<body pos="{(i % 4) * 0.25 - 0.4:.3f} {(i // 4) * 0.25 - 0.3:.3f} 0.020">'
      f'<freejoint/><geom type="box" size="0.04 0.05 0.03" mass="0.5"/></body>'
      for i in range(n)
    )
    xml = f"""
    <mujoco>
      <option cone="elliptic" solver="Newton" jacobian="sparse" iterations="20" ls_iterations="20" integrator="implicitfast"/>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 .1" friction="1 0.01 0.001"/>
        {bodies}
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, _ = test_data.fixture(xml=xml)
    self.assertTrue(m.is_sparse)

    mjd.qvel[0::6] = 1.5  # slide -> cone middle zone
    mujoco.mj_forward(mjm, mjd)
    self.assertGreater(mjd.nefc, 0)

    d = mjw.put_data(mjm, mjd)
    d.qacc.fill_(wp.inf)
    d.qfrc_constraint.fill_(wp.inf)
    d.efc.force.fill_(wp.inf)
    mjw.solve(m, d)

    qacc = d.qacc.numpy()[0]
    self.assertTrue(np.all(np.isfinite(qacc)), "Newton solve produced non-finite qacc")
    _assert_eq(qacc, mjd.qacc, "qacc")

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
            ctx.quad_changed_count.zero_()
          solver._update_constraint(m, d, ctx, track_changes=track)
          if track:
            wp.synchronize()
            if np.any(ctx.quad_changed_count.numpy() > 0):
              any_changes = True
          update_fn(m, d, ctx)
          wp.launch(
            solver._solve_zero_search_dot(False),
            dim=(d.nworld),
            inputs=[ctx.state_changed_count, ctx.done],
            outputs=[ctx.search_dot],
          )
          wp.launch(
            solver._solve_search_update(False),
            dim=(d.nworld, m.nv),
            inputs=[m.opt.solver, ctx.state_changed_count, ctx.Mgrad, ctx.search, ctx.beta, ctx.done],
            outputs=[ctx.search, ctx.search_dot],
          )
        return d.qacc.numpy().copy(), any_changes

      qacc_full, _ = _run_solver(mjw.put_data(mjm, mjd), solver._update_gradient)
      qacc_inc, any_changes = _run_solver(mjw.put_data(mjm, mjd), solver._update_gradient_incremental, track=True)
      total_any_changes = total_any_changes or any_changes

      _assert_eq(qacc_inc, qacc_full, f"qacc keyframe={keyframe}")

    self.assertTrue(total_any_changes, "no state changes detected across any keyframe")

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
  def test_fast_path_friction_transitions(self):
    """qfrc_constraint must stay consistent when friction rows cross linear zones."""
    mjm = mujoco.MjModel.from_xml_string(_FRICTION_CHAIN_XML)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    nworld = 64
    d = mjw.put_data(mjm, mjd, nworld=nworld)
    rng = np.random.default_rng(3)
    qvel = rng.uniform(-8.0, 8.0, size=(nworld, mjm.nv)).astype(np.float32)
    wp.copy(d.qvel, wp.array(qvel, dtype=float))

    max_rel = 0.0
    cap_steps = 0
    for _ in range(20):
      mjw.step(m, d)
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


class CompactSolverTest(absltest.TestCase):
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
    _, _, m, d = _put_compact(_COMPACT_CONTACT_XML)
    self.assertTrue(m.is_sparse)
    mjw.forward(m, d)  # full baseline solve (also builds efc.J, M)
    self.assertGreater(d.nacon.numpy()[0], 0)  # contacts exist
    baseline_qacc = d.qacc.numpy().copy()

    d.tree_awake = wp.array(np.ones((d.nworld, m.ntree), dtype=int), dtype=int)
    island.update_active_dofs(m, d)
    self.assertEqual(d.ncdof.numpy()[0], m.nv)

    solver.solve_compact(m, d)

    np.testing.assert_allclose(d.qacc.numpy(), baseline_qacc, rtol=1e-3, atol=1e-4)

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


if __name__ == "__main__":
  wp.init()
  absltest.main()
