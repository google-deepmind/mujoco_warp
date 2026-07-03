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

from functools import lru_cache
from typing import Any

import warp as wp

from mujoco_warp._src.types import MJ_MINVAL

# float32 machine epsilon; the pivot floor is scaled relative to the matrix magnitude,
# mirroring the mindiag argument of MuJoCo's mju_cholFactor (issue #1415)
_FLOAT32_EPS = 1.1920929e-07


@wp.func
def tile_diag_is_finite(t: Any, size: int) -> bool:
  """Returns True if the leading size x size diagonal of tile t is all finite."""
  finite = bool(True)
  for i in range(size):
    if not wp.isfinite(t[i, i]):
      finite = False
  return finite


@wp.func
def tile_diag_is_healthy(t: Any, size: int, floor: float) -> bool:
  """Returns True if the leading diagonal of tile t is finite and >= floor.

  The factor diagonal is sqrt(pivot), so pass sqrt(mindiag). This catches rank
  deficiency that factors to finite-but-garbage pivots without producing a NaN:
  a pure non-finite scan misses those, and backends may return finite garbage
  rather than NaN on indefinite input.
  """
  healthy = bool(True)
  for i in range(size):
    if (not wp.isfinite(t[i, i])) or t[i, i] < floor:
      healthy = False
  return healthy


@wp.func
def cholesky_mindiag(A: wp.array2d[float], matrix_size: int) -> float:
  """Scale-relative pivot floor for A: eps32 * max |diag(A)|, at least MJ_MINVAL."""
  max_diag = float(0.0)
  for i in range(matrix_size):
    max_diag = wp.max(max_diag, wp.abs(A[i, i]))
  return wp.max(_FLOAT32_EPS * max_diag, float(MJ_MINVAL))


@wp.func
def tile_load_elementwise(t: Any, A: wp.array2d[float], size: int):
  """Reloads the leading size x size block of A into tile t element-wise."""
  for a in range(size):
    for b in range(size):
      t[a, b] = A[a, b]


@wp.func
def rebuild_diagonal_block(t: Any, A: wp.array2d[float], U: wp.array2d[float], k: int, block_size: int):
  """Reloads diagonal block A[k:k+bs, k:k+bs] into tile t.

  Reapplies the Schur complement updates from the already-factorized block rows
  of U element-wise.
  """
  for a in range(block_size):
    for b in range(block_size):
      s = A[k + a, k + b]
      for r in range(k):
        s -= U[r, k + a] * U[r, k + b]
      t[a, b] = s


@wp.func
def tile_cholesky_floored_inplace(t: Any, size: int, mindiag: float) -> wp.int64:
  """In-place upper Cholesky of the leading size x size block of tile t.

  Each pivot is floored at mindiag like MuJoCo's mju_cholFactor, so factorizing
  a numerically indefinite matrix yields a finite factor instead of NaNs (#1415).

  A floored pivot is treated as rank deficient: it is set to sqrt(mindiag) and
  the remainder of its row is zeroed, decoupling the direction (a spring of
  stiffness mindiag). Dividing the row by a floored pivot instead (as
  mju_cholFactor does in float64) amplifies consecutive deficient rows
  quadratically and overflows float32. Returns a bitmask of the deficient rows
  so callers can decouple them from trailing blocks as well.

  Element-wise and redundant across the block's threads; this is only correct
  because every thread computes identical values in identical order.
  """
  deficient = wp.int64(0)
  for j in range(size):
    s = t[j, j]
    for k in range(j):
      s -= t[k, j] * t[k, j]
    if s >= mindiag:
      piv = wp.sqrt(s)
      t[j, j] = piv
      inv = 1.0 / piv
      for i in range(j + 1, size):
        s2 = t[j, i]
        for k in range(j):
          s2 -= t[k, j] * t[k, i]
        t[j, i] = s2 * inv
    else:  # small, negative or NaN pivot
      deficient |= wp.int64(1) << wp.int64(j)
      t[j, j] = wp.sqrt(mindiag)
      for i in range(j + 1, size):
        t[j, i] = 0.0
  return deficient


@wp.func
def tile_zero_deficient_rows(t: Any, deficient: wp.int64, size: int):
  """Zeroes the rows of tile t flagged in the deficient bitmask."""
  for j in range(size):
    if ((deficient >> wp.int64(j)) & wp.int64(1)) != wp.int64(0):
      for i in range(size):
        t[j, i] = 0.0


@lru_cache(maxsize=None)
def create_blocked_cholesky_factorize_solve_func(block_size: int, matrix_size_static: int):
  @wp.func
  def blocked_cholesky_factorize_solve_func(
    # In:
    A: wp.array2d[float],
    b: wp.array2d[float],
    matrix_size: int,
    # Out:
    U: wp.array2d[float],
    x: wp.array2d[float],
  ):
    """Block Cholesky factorization and solve while keeping the forward RHS live."""
    rhs_tile = wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), storage="shared", bounds_check=False)

    # hoisted: same value for every diagonal block; O(nv) once per matrix,

    # so the strengthened health check adds no per-block cost.

    mindiag = cholesky_mindiag(A, matrix_size)

    sqrt_mindiag = wp.sqrt(mindiag)

    for k in range(0, matrix_size, block_size):
      end = k + block_size
      rhs_view = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(k, 0))

      A_kk_tile = wp.tile_load(
        A, shape=(block_size, block_size), offset=(k, k), storage="shared", bounds_check=False, aligned=True
      )

      for j in range(0, k, block_size):
        U_block = wp.tile_load(
          U, shape=(block_size, block_size), offset=(j, k), storage="shared", bounds_check=False, aligned=True
        )
        wp.tile_matmul(wp.tile_transpose(U_block), U_block, A_kk_tile, alpha=-1.0)

        y_block = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))
        wp.tile_matmul(wp.tile_transpose(U_block), y_block, rhs_view, alpha=-1.0)

      wp.tile_cholesky_inplace(A_kk_tile, fill_mode="upper")

      # float32 roundoff in the assembly of H can push its smallest eigenvalues
      # negative when ||H|| is large; the unguarded factorization above then
      # produces NaNs. Detect this and refactor the block with a scale-relative
      # per-pivot floor, cf. mju_cholFactor's mindiag (issue #1415). The rebuild
      # is element-wise, so no cooperative tile op runs inside the branch.
      deficient = wp.int64(0)
      if not tile_diag_is_healthy(A_kk_tile, block_size, sqrt_mindiag):
        rebuild_diagonal_block(A_kk_tile, A, U, k, block_size)
        deficient = tile_cholesky_floored_inplace(A_kk_tile, block_size, mindiag)

      wp.tile_store(U, A_kk_tile, offset=(k, k), bounds_check=False, aligned=True)

      wp.tile_lower_solve_inplace(wp.tile_transpose(A_kk_tile), rhs_view)

      for i in range(end, matrix_size, block_size):
        A_ki_tile = wp.tile_load(
          A, shape=(block_size, block_size), offset=(k, i), storage="shared", bounds_check=False, aligned=True
        )

        for j in range(0, k, block_size):
          U_jk_tile = wp.tile_load(
            U, shape=(block_size, block_size), offset=(j, k), storage="shared", bounds_check=False, aligned=True
          )
          U_ji_tile = wp.tile_load(
            U, shape=(block_size, block_size), offset=(j, i), storage="shared", bounds_check=False, aligned=True
          )
          wp.tile_matmul(wp.tile_transpose(U_jk_tile), U_ji_tile, A_ki_tile, alpha=-1.0)

        wp.tile_lower_solve_inplace(wp.tile_transpose(A_kk_tile), A_ki_tile)

        # decouple rank-deficient rows from the trailing blocks (issue #1415)
        if deficient != wp.int64(0):
          tile_zero_deficient_rows(A_ki_tile, deficient, block_size)

        wp.tile_store(U, A_ki_tile, offset=(k, i), bounds_check=False, aligned=True)

    for i in range(matrix_size - block_size, -1, -block_size):
      i_end = i + block_size
      tmp_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))
      for j in range(i_end, matrix_size, block_size):
        U_tile = wp.tile_load(
          U, shape=(block_size, block_size), offset=(i, j), storage="shared", bounds_check=False, aligned=True
        )
        x_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))
        wp.tile_matmul(U_tile, x_tile, tmp_tile, alpha=-1.0)

      U_tile = wp.tile_load(
        U, shape=(block_size, block_size), offset=(i, i), storage="shared", bounds_check=False, aligned=True
      )
      wp.tile_upper_solve_inplace(U_tile, tmp_tile)

    wp.tile_store(x, rhs_tile, offset=(0, 0), bounds_check=False)

  return blocked_cholesky_factorize_solve_func


@lru_cache(maxsize=None)
def create_blocked_cholesky_solve_func(block_size: int, matrix_size_static: int):
  @wp.func
  def blocked_cholesky_solve_func(
    # In:
    U: wp.array2d[float],
    b: wp.array2d[float],
    matrix_size: int,
    # Out:
    x: wp.array2d[float],
  ):
    """Block Cholesky solve.

    Solves A x = b given the Cholesky factor U (A = U^T U) using blocked forward and backward
    substitution.
    """
    rhs_tile = wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), storage="shared", bounds_check=False)

    # Forward substitution: solve U^T y = b
    for i in range(0, matrix_size, block_size):
      rhs_view = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))
      for j in range(0, i, block_size):
        U_block = wp.tile_load(
          U, shape=(block_size, block_size), offset=(j, i), storage="shared", bounds_check=False, aligned=True
        )
        y_block = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))
        wp.tile_matmul(wp.tile_transpose(U_block), y_block, rhs_view, alpha=-1.0)

      U_tile = wp.tile_load(
        U, shape=(block_size, block_size), offset=(i, i), storage="shared", bounds_check=False, aligned=True
      )
      wp.tile_lower_solve_inplace(wp.tile_transpose(U_tile), rhs_view)

    # Backward substitution: solve U x = y
    for i in range(matrix_size - block_size, -1, -block_size):
      i_end = i + block_size
      tmp_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(i, 0))
      for j in range(i_end, matrix_size, block_size):
        U_tile = wp.tile_load(
          U, shape=(block_size, block_size), offset=(i, j), storage="shared", bounds_check=False, aligned=True
        )
        x_tile = wp.tile_view(rhs_tile, shape=(block_size, 1), offset=(j, 0))
        wp.tile_matmul(U_tile, x_tile, tmp_tile, alpha=-1.0)
      U_tile = wp.tile_load(
        U, shape=(block_size, block_size), offset=(i, i), storage="shared", bounds_check=False, aligned=True
      )

      wp.tile_upper_solve_inplace(U_tile, tmp_tile)

    wp.tile_store(x, rhs_tile, offset=(0, 0), bounds_check=False)

  return blocked_cholesky_solve_func
