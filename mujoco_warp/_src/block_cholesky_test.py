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

"""Tests for block_cholesky, in particular the pivot floor (issue #1415)."""

import numpy as np
import warp as wp
from absl.testing import absltest

from mujoco_warp._src.block_cholesky import create_blocked_cholesky_factorize_solve_func
from mujoco_warp._src.solver import _update_gradient_cholesky

_NV = 48
_BLOCK = 16
_NV_PLAIN = 24
_FLOAT32_EPS = 1.1920929e-07


def _make_issue_hessian(nv: int, nefc: int, seed: int):
  """Builds H = M + J' D J per issue #1415.

  Rank-deficient stiff contact rows (large efc_D, ||J' D J|| ~ 1e8) leave the
  tiny rigid-body modes of M (~3e-4, light bodies in a near-singular
  configuration) in the null space of J. Assembled in float64 the matrix is SPD;
  assembled in float32 the roundoff error (~ ||H|| * eps32) swamps the small
  eigenvalues and the matrix becomes numerically indefinite.
  """
  rng = np.random.default_rng(seed)
  q, _ = np.linalg.qr(rng.normal(size=(nv, nv)))
  m_mat = (q * rng.uniform(3e-4, 1e-3, size=nv)) @ q.T
  jac = rng.normal(size=(nefc, nv)) * 30.0
  efc_d = rng.uniform(100.0, 2700.0, size=nefc)
  h64 = m_mat + (jac.T * efc_d) @ jac
  jf, df, mf = jac.astype(np.float32), efc_d.astype(np.float32), m_mat.astype(np.float32)
  h32 = mf + (jf.T * df) @ jf
  h32 = 0.5 * (h32 + h32.T)
  return h64, h32


def _make_spd(nv: int, seed: int):
  rng = np.random.default_rng(seed)
  chol = np.tril(rng.normal(size=(nv, nv))) + nv * np.eye(nv)
  return (chol @ chol.T).astype(np.float32)


@wp.kernel(module="unique")
def _raw_tile_cholesky(
  # In:
  A: wp.array2d[float],
  # Out:
  out: wp.array2d[float],
):
  t = wp.tile_load(A, shape=(_NV, _NV), offset=(0, 0), storage="shared")
  wp.tile_cholesky_inplace(t, fill_mode="upper")
  wp.tile_store(out, t)


@wp.kernel(module="unique", module_options={"enable_mathdx_gemm": False})
def _blocked_factorize_solve(
  # In:
  A: wp.array2d[float],
  b: wp.array2d[float],
  # Out:
  U: wp.array2d[float],
  x: wp.array2d[float],
):
  wp.static(create_blocked_cholesky_factorize_solve_func(_BLOCK, _NV))(A, b, _NV, U, x)


def _run_blocked(h32: np.ndarray, b: np.ndarray):
  a_wp = wp.array(h32, dtype=float)
  b_wp = wp.array(b.reshape(_NV, 1).astype(np.float32), dtype=float)
  u_wp = wp.zeros((_NV, _NV), dtype=float)
  x_wp = wp.zeros((_NV, 1), dtype=float)
  wp.launch_tiled(_blocked_factorize_solve, dim=(1,), inputs=[a_wp, b_wp], outputs=[u_wp, x_wp], block_dim=32)
  wp.synchronize()
  return u_wp.numpy(), x_wp.numpy()[:, 0]


class BlockCholeskyTest(absltest.TestCase):
  def test_blocked_cholesky_well_conditioned(self):
    """Blocked factorize+solve matches a float64 reference solve."""
    h = _make_spd(_NV, seed=1)
    b = np.linspace(-1.0, 1.0, _NV)
    u, x = _run_blocked(h, b)
    x_ref = np.linalg.solve(h.astype(np.float64), b)
    rel = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    self.assertLess(rel, 1.0e-5, f"well-conditioned solve rel err {rel:.3e}")
    u_ref = np.linalg.cholesky(h.astype(np.float64)).T
    rel_u = np.abs(np.triu(u) - u_ref).max() / np.abs(u_ref).max()
    self.assertLess(rel_u, 1.0e-5, f"well-conditioned factor rel err {rel_u:.3e}")

  def test_blocked_cholesky_float32_indefinite_no_nan(self):
    """Pivot floor regression test for issue #1415.

    Before the fix, factorizing a Hessian that float32 roundoff made
    numerically indefinite produced NaNs in the factor (sqrt of a negative
    pivot) which propagated through qfrc_constraint -> qacc -> qpos. With the
    scale-relative pivot floor the factor and solution stay finite and bounded;
    deficient directions respond like springs of stiffness ~eps32 * ||H||.
    """
    h64, h32 = _make_issue_hessian(_NV, nefc=40, seed=3)

    # precondition: SPD in float64, numerically indefinite in float32
    self.assertGreater(np.linalg.eigvalsh(h64).min(), 0.0)
    eigmin32 = np.linalg.eigvalsh(h32.astype(np.float64)).min()
    self.assertLess(eigmin32, 0.0, "test matrix must be float32-indefinite")

    # demonstrate the pathology: the unguarded tile cholesky (the pre-fix code
    # path) produces NaNs on this matrix
    a_wp = wp.array(h32, dtype=float)
    out_wp = wp.zeros((_NV, _NV), dtype=float)
    wp.launch_tiled(_raw_tile_cholesky, dim=(1,), inputs=[a_wp], outputs=[out_wp], block_dim=32)
    wp.synchronize()
    self.assertFalse(
      np.isfinite(np.triu(out_wp.numpy())).all(),
      "unguarded tile_cholesky no longer NaNs; the guard in block_cholesky may be redundant",
    )

    # the guarded blocked factorize+solve stays finite and bounded
    b = np.linspace(-1.0, 1.0, _NV)
    u, x = _run_blocked(h32, b)
    self.assertTrue(np.isfinite(np.triu(u)).all(), "factor contains non-finite values")
    self.assertTrue(np.isfinite(x).all(), "solution contains non-finite values")
    # loose runaway guard: an unguarded cascade overflows float32 (~1e38) while
    # the floored solve responds at worst like 1/mindiag per deficient direction
    self.assertLess(np.abs(x).max(), 1.0 / _FLOAT32_EPS, "solution magnitude indicates a pivot cascade")

  def test_update_gradient_cholesky_indefinite_no_nan(self):
    """The nv <= 32 single-tile solver path is also protected (issue #1415)."""
    h64, h32 = _make_issue_hessian(_NV_PLAIN, nefc=18, seed=5)
    self.assertLess(np.linalg.eigvalsh(h32.astype(np.float64)).min(), 0.0)

    kernel = _update_gradient_cholesky(_NV_PLAIN)
    grad = np.linspace(-1.0, 1.0, _NV_PLAIN).astype(np.float32)
    grad_wp = wp.array(grad.reshape(1, _NV_PLAIN), dtype=float)
    h_wp = wp.array(h32.reshape(1, _NV_PLAIN, _NV_PLAIN), dtype=float)
    done_wp = wp.zeros(1, dtype=bool)
    mgrad_wp = wp.zeros((1, _NV_PLAIN), dtype=float)
    wp.launch_tiled(kernel, dim=1, inputs=[grad_wp, h_wp, done_wp], outputs=[mgrad_wp], block_dim=32)
    wp.synchronize()
    x = mgrad_wp.numpy()[0]
    self.assertTrue(np.isfinite(x).all(), "Mgrad contains non-finite values")

    # a well-conditioned matrix through the same kernel still matches numpy
    h_spd = _make_spd(_NV_PLAIN, seed=1)
    h_wp2 = wp.array(h_spd.reshape(1, _NV_PLAIN, _NV_PLAIN), dtype=float)
    wp.launch_tiled(kernel, dim=1, inputs=[grad_wp, h_wp2, done_wp], outputs=[mgrad_wp], block_dim=32)
    wp.synchronize()
    x2 = mgrad_wp.numpy()[0]
    x2_ref = np.linalg.solve(h_spd.astype(np.float64), grad.astype(np.float64))
    rel = np.linalg.norm(x2 - x2_ref) / np.linalg.norm(x2_ref)
    self.assertLess(rel, 1.0e-5, f"well-conditioned Mgrad rel err {rel:.3e}")


if __name__ == "__main__":
  wp.init()
  absltest.main()
