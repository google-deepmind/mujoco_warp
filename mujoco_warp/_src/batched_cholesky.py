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
"""Lane-cooperative batched dense Cholesky factorize+solve for small nv (AMD/HIP).

Motivation: on HIP/gfx942 wp.tile_cholesky has no MathDx/MFMA path and falls back
to a scalar column-serial algorithm that every one of the 64 block threads executes
REDUNDANTLY on a shared tile, routing every element access through LDS. Profiling
showed ~17.8k dependent LDS ops per 27x27 factorize+solve -> the kernel is a
dependent-LDS-latency-bound serial chain (~69% of the Newton step at ~4% of peak).

This kernel instead assigns one 64-lane wavefront per world (preserving occupancy)
and distributes the work across lanes: lane i owns row i. Each column step does a
serial sqrt on the pivot lane plus a parallel rank-1 style update across the active
lanes, publishing shared state with a single tile_scatter_masked (which emits the
required __syncthreads). This turns the O(n^3) dependent-LDS chain into O(n^2)
barrier-free shared reads with O(n) barriers.

Reads of the ORIGINAL matrix are taken from the UPPER triangle (A[k,i], k<i) to match
the existing fill_mode="upper" contract of update_gradient_cholesky (h may only have
its upper triangle populated). The lower triangle is overwritten with the L factor
(A = L L^T); the solves then read L from the lower triangle.
"""

import os

import warp as wp

from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"enable_backward": False})

# Use the lane-cooperative batched dense Cholesky (HIP/gfx942 fast path) instead of
# the redundant scalar wp.tile_cholesky. Defaults on; set MJW_BATCHED_CHOLESKY=0 to
# fall back to the stock kernels for A/B comparison.
USE_BATCHED_CHOLESKY = os.environ.get("MJW_BATCHED_CHOLESKY", "1") != "0"

# Only route matrices no larger than this through the batched kernel. One matrix maps
# to one 64-lane wavefront with lane == row/column, so the matrix dimension must not
# exceed the wavefront width (64). Larger matrices keep the existing blocked path.
BATCHED_CHOLESKY_MAX_DIM = 64


@cache_kernel
def batched_cholesky_factorize_solve(nv: int, nv_pad: int):
  """Factory: returns a kernel that solves A x = b for a batch of SPD systems.

  Kernel signature matches update_gradient_cholesky so it is a drop-in replacement:
    inputs  = [grad (nworld, nv_pad), h (nworld, nv_pad, nv_pad), done (nworld,)]
    outputs = [Mgrad (nworld, nv_pad)]
  Launch with wp.launch_tiled(dim=nworld, block_dim=>=nv).
  """
  N = wp.static(nv)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    ctx_grad_in: wp.array2d(dtype=float),
    h_in: wp.array3d(dtype=float),
    ctx_done_in: wp.array(dtype=bool),
    # Out:
    ctx_Mgrad_out: wp.array2d(dtype=float),
  ):
    worldid, lane = wp.tid()

    # Uniform across the block (one world per block) -> no divergence.
    if ctx_done_in[worldid]:
      return

    # A: shared working tile. Upper triangle keeps the original SPD matrix;
    # lower triangle + diagonal are overwritten with the L factor.
    A = wp.tile_load(h_in[worldid], shape=(N, N), storage="shared")
    # y: shared RHS/solution vector (starts as b, ends as x = A^-1 b).
    y = wp.tile_load(ctx_grad_in[worldid], shape=(N,), storage="shared")

    # ---- Factorization: A = L L^T (left-looking, lower) ----
    for k in range(N):
      # Pivot: L[k,k] = sqrt(A[k,k] - sum_{j<k} L[k,j]^2)   (lane k only)
      d = float(0.0)
      if lane == k:
        s = wp.tile_extract(A, k, k)
        for j in range(k):
          r = wp.tile_extract(A, k, j)
          s -= r * r
        # Floor the squared pivot at MJ_MINVAL before the sqrt. For an SPD system
        # s > 0 in exact arithmetic, but a near-singular world combined with
        # gfx950's FP reduction order (which differs from gfx942/CUDA) can drive s
        # slightly <= 0; sqrt() would then emit NaN and 1/L[k,k] Inf, corrupting the
        # whole solve. This mirrors MuJoCo's mju_cholFactor mindiag handling and is
        # a no-op when s >> MJ_MINVAL, so well-conditioned pivots (and gfx942/CUDA
        # numerics) are unchanged.
        d = wp.sqrt(wp.max(s, MJ_MINVAL))
      wp.tile_scatter_masked(A, k, k, d, lane == k)  # publish L[k,k] (+ barrier)
      inv = 1.0 / wp.tile_extract(A, k, k)

      # Column k below the diagonal: L[i,k] = (A[k,i] - sum_{j<k} L[i,j] L[k,j]) / L[k,k]
      val = float(0.0)
      has = bool(False)
      if lane > k and lane < N:
        s = wp.tile_extract(A, k, lane)  # UPPER triangle = original off-diagonal
        for j in range(k):
          s -= wp.tile_extract(A, lane, j) * wp.tile_extract(A, k, j)
        val = s * inv
        has = True
      row = wp.min(lane, N - 1)
      wp.tile_scatter_masked(A, row, k, val, has)  # write L[i,k] (+ barrier)

    # ---- Forward solve: L y = b (column-oriented) ----
    for k in range(N):
      yk = float(0.0)
      if lane == k:
        yk = wp.tile_extract(y, k) / wp.tile_extract(A, k, k)
      wp.tile_scatter_masked(y, k, yk, lane == k)
      ykk = wp.tile_extract(y, k)
      upd = float(0.0)
      has = bool(False)
      if lane > k and lane < N:
        upd = wp.tile_extract(y, lane) - wp.tile_extract(A, lane, k) * ykk
        has = True
      row = wp.min(lane, N - 1)
      wp.tile_scatter_masked(y, row, upd, has)

    # ---- Back solve: L^T x = y  (L^T[i,k] = L[k,i]) ----
    for kk in range(N):
      k = N - 1 - kk
      xk = float(0.0)
      if lane == k:
        xk = wp.tile_extract(y, k) / wp.tile_extract(A, k, k)
      wp.tile_scatter_masked(y, k, xk, lane == k)
      xkk = wp.tile_extract(y, k)
      upd = float(0.0)
      has = bool(False)
      if lane < k:  # lane < k <= N-1, always in-range
        upd = wp.tile_extract(y, lane) - wp.tile_extract(A, k, lane) * xkk
        has = True
      wp.tile_scatter_masked(y, lane, upd, has)

    wp.tile_store(ctx_Mgrad_out[worldid], y)

  return kernel


@cache_kernel
def batched_factor_solve_i(tile_size: int):
  """Factory: lane-cooperative UPPER Cholesky (A = U^T U) + solve for a diagonal
  block of the mass matrix, drop-in for smooth._tile_cholesky_factorize_solve.

  Stores the UPPER factor U into L (matching fill_mode="upper") so the separate
  solve-only path (_tile_cholesky_solve) can reuse the cached factor. Signature:
    inputs  = [M (nworld, nv_pad, nv_pad), y (nworld, nv_pad), adr (nnode,)]
    outputs = [x (nworld, nv_pad), L (nworld, nv_pad, nv_pad)]
  Launch with wp.launch_tiled(dim=(nworld, nnode), block_dim=>=tile_size).
  """
  N = wp.static(tile_size)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    M_in: wp.array3d(dtype=float),
    y_in: wp.array2d(dtype=float),
    adr: wp.array(dtype=int),
    # Out:
    x_out: wp.array2d(dtype=float),
    L_out: wp.array3d(dtype=float),
  ):
    worldid, nodeid, lane = wp.tid()
    dofid = adr[nodeid]

    # Shared working tile; upper triangle + diagonal hold U after factorization.
    A = wp.tile_load(M_in[worldid], shape=(N, N), offset=(dofid, dofid), storage="shared")
    y = wp.tile_load(y_in[worldid], shape=(N,), offset=(dofid,), storage="shared")

    # ---- Factorization: A = U^T U (left-looking, upper); lane == column ----
    for k in range(N):
      # U[k,k] = sqrt(A[k,k] - sum_{i<k} U[i,k]^2)   (lane k only)
      d = float(0.0)
      if lane == k:
        s = wp.tile_extract(A, k, k)
        for i in range(k):
          u = wp.tile_extract(A, i, k)
          s -= u * u
        # See batched_cholesky_factorize_solve: floor the squared pivot at
        # MJ_MINVAL so a near-singular world on gfx950 cannot produce a negative
        # value under the sqrt (NaN) or a zero pivot (Inf in 1/U[k,k]). Mirrors
        # MuJoCo's mju_cholFactor mindiag handling; no-op for s >> MJ_MINVAL.
        d = wp.sqrt(wp.max(s, MJ_MINVAL))
      wp.tile_scatter_masked(A, k, k, d, lane == k)
      inv = 1.0 / wp.tile_extract(A, k, k)

      # Row k, cols j>k: U[k,j] = (A[k,j] - sum_{i<k} U[i,k] U[i,j]) / U[k,k]
      val = float(0.0)
      has = bool(False)
      if lane > k and lane < N:
        s = wp.tile_extract(A, k, lane)
        for i in range(k):
          s -= wp.tile_extract(A, i, k) * wp.tile_extract(A, i, lane)
        val = s * inv
        has = True
      col = wp.min(lane, N - 1)
      wp.tile_scatter_masked(A, k, col, val, has)

    # Persist the upper factor U (matches fill_mode="upper" for later reuse).
    wp.tile_store(L_out[worldid], A, offset=(dofid, dofid))

    # ---- Forward solve: U^T z = y  ((U^T)[i,k] = U[k,i]) ----
    for k in range(N):
      zk = float(0.0)
      if lane == k:
        zk = wp.tile_extract(y, k) / wp.tile_extract(A, k, k)
      wp.tile_scatter_masked(y, k, zk, lane == k)
      zkk = wp.tile_extract(y, k)
      upd = float(0.0)
      has = bool(False)
      if lane > k and lane < N:
        upd = wp.tile_extract(y, lane) - wp.tile_extract(A, k, lane) * zkk
        has = True
      row = wp.min(lane, N - 1)
      wp.tile_scatter_masked(y, row, upd, has)

    # ---- Back solve: U x = z  (U[i,k] = A[i,k], i<k) ----
    for kk in range(N):
      k = N - 1 - kk
      xk = float(0.0)
      if lane == k:
        xk = wp.tile_extract(y, k) / wp.tile_extract(A, k, k)
      wp.tile_scatter_masked(y, k, xk, lane == k)
      xkk = wp.tile_extract(y, k)
      upd = float(0.0)
      has = bool(False)
      if lane < k:
        upd = wp.tile_extract(y, lane) - wp.tile_extract(A, lane, k) * xkk
        has = True
      wp.tile_scatter_masked(y, lane, upd, has)

    wp.tile_store(x_out[worldid], y, offset=(dofid,))

  return kernel
