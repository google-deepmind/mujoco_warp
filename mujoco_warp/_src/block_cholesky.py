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

import warp as wp


@lru_cache(maxsize=None)
def create_blocked_cholesky_func(block_size: int, matrix_size: int):
  @wp.func
  def blocked_cholesky_func(
    # In:
    A: wp.array(dtype=float, ndim=2),
    # Out:
    L: wp.array(dtype=float, ndim=2),
  ):
    """Computes the Cholesky factorization of a symmetric positive definite matrix A in blocks.

    It returns a lower-triangular matrix L such that A = L L^T.
    """
    # workaround for compile error
    n = matrix_size

    #wp.printf("matrix_size: %d, n: %d\n", matrix_size, n)

    # Process the matrix in blocks along its leading dimension.
    for k in range(0, n, block_size):
      end = k + block_size

      # Load current diagonal block A[k:end, k:end]
      # and update with contributions from previously computed blocks.
      A_kk_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(k, k), storage="shared")

      if k > 0:
        for j in range(0, k, block_size):
          L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j))
          L_block_T = wp.tile_transpose(L_block)
          L_L_T_block = wp.tile_matmul(L_block, L_block_T)
          A_kk_tile -= L_L_T_block

      # Compute the Cholesky factorization for the block
      L_kk_tile = wp.tile_cholesky(A_kk_tile)
      wp.tile_store(L, L_kk_tile, offset=(k, k))

      # Process the blocks below the current block
      for i in range(end, matrix_size, block_size):
        A_ik_tile = wp.tile_load(A, shape=(block_size, block_size), offset=(i, k), storage="shared")

        if k > 0:
          for j in range(0, k, block_size):
            L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
            L_2_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j))
            L_T_tile = wp.tile_transpose(L_2_tile)
            L_L_T_tile = wp.tile_matmul(L_tile, L_T_tile)
            A_ik_tile -= L_L_T_tile

        t = wp.tile_transpose(A_ik_tile)
        tmp = wp.tile_lower_solve(L_kk_tile, t)
        sol_tile = wp.tile_transpose(tmp)

        wp.tile_store(L, sol_tile, offset=(i, k))

  return blocked_cholesky_func


@lru_cache(maxsize=None)
def create_blocked_cholesky_solve_func(block_size: int, matrix_size: int):
  @wp.func
  def blocked_cholesky_solve_func(
    # In:
    L: wp.array(dtype=float, ndim=2),
    b: wp.array(dtype=float, ndim=2),
    tmp: wp.array(dtype=float, ndim=2),
    # Out:
    x: wp.array(dtype=float, ndim=2),
  ):
    """Block Cholesky factorization and solve.

    Solves A x = b given the Cholesky factor L (A = L L^T) using blocked forward and backward
    substitution.
    """

    # Forward substitution: solve L y = b
    for i in range(0, matrix_size, block_size):
      rhs_tile = wp.tile_load(b, shape=(block_size, 1), offset=(i, 0))
      for j in range(0, i, block_size):
        L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
        y_block = wp.tile_load(tmp, shape=(block_size, 1), offset=(j, 0))
        Ly_block = wp.tile_matmul(L_block, y_block)
        rhs_tile -= Ly_block
      
      L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i))
      y_tile = wp.tile_lower_solve(L_tile, rhs_tile)
      wp.tile_store(tmp, y_tile, offset=(i, 0))

    # Backward substitution: solve L^T x = y
    for i in range(matrix_size - block_size, -1, -block_size):
      i_end = i + block_size
      rhs_tile = wp.tile_load(tmp, shape=(block_size, 1), offset=(i, 0))
      for j in range(i_end, matrix_size, block_size):
        L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(j, i))
        L_T_tile = wp.tile_transpose(L_tile)
        x_tile = wp.tile_load(x, shape=(block_size, 1), offset=(j, 0))
        L_T_x_tile = wp.tile_matmul(L_T_tile, x_tile)
        rhs_tile -= L_T_x_tile
      L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i))

      x_tile = wp.tile_upper_solve(wp.tile_transpose(L_tile), rhs_tile)
      wp.tile_store(x, x_tile, offset=(i, 0))

  return blocked_cholesky_solve_func
