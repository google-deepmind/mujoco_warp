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
"""Newton constraint solver over islands (opt.island_solve).

make_constraint writes an island's rows contiguously. Single trees of up to island.dense_width(m)
DOFs store their Jacobian as dense rows (island-local columns) and are solved by the dense world
solver; other groups keep sparse rows. On the island solver's block path, a group's DOFs are
gathered into tile-aligned rows, and its inertia and Hessian are dense blocks of _TILE x _TILE tiles
in a per-world pool. Every island is solved with its own step size
and convergence test, as MuJoCo's island solver does. Kernels run over the work that exists: rows,
DOFs, islands and the tile pairs of each island's Hessian.
"""

import dataclasses

import warp as wp

from mujoco_warp._src import island
from mujoco_warp._src import types
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_augmented_factorize_solve_newton_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_newton_func
from mujoco_warp._src.block_cholesky import create_newton_tile_func
from mujoco_warp._src.block_cholesky import solve_search_sums
from mujoco_warp._src.solver import _JTDAJ_THREADS_PER_GROUP
from mujoco_warp._src.solver import IslandProblems
from mujoco_warp._src.solver import _active_check
from mujoco_warp._src.solver import _constraint_row
from mujoco_warp._src.solver import _converged
from mujoco_warp._src.solver import _is_cone
from mujoco_warp._src.solver import _is_equality
from mujoco_warp._src.solver import _is_friction
from mujoco_warp._src.solver import _JTDACJ_sparse
from mujoco_warp._src.solver import _jtdaj_groups_per_world
from mujoco_warp._src.solver import _linesearch_iterative_kernel
from mujoco_warp._src.solver import _rewrite_cone_rows
from mujoco_warp._src.solver import _row_weight
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import efc_threads_per_world
from mujoco_warp._src.warp_util import side_stream

wp.set_module_options({"enable_backward": False})

# the island tile pool's tile size (H and aligned DOF rows; U holds blocked groups' factors)
_TILE = 32
# groups up to this many DOFs (two pool tiles per side) are factored as one tile, larger ones
# blocked (_factor), as the world solver factors its H
_ONE_TILE_NV = types.H_ONE_TILE_MAX


def _island_pool_size(nv: int, ntree: int, tile: int) -> tuple[int, int]:
  """Tiles and aligned rows the tile pool needs at most (up to ntree groups of nv DOFs in all)."""
  ntile = -(-(nv * nv + 2 * tile * nv + tile * tile * ntree) // (tile * tile))
  # groups factored blocked take _blocked_rows each
  nrow = -(-(nv + (tile - 1) * ntree) // tile) * tile + nv // (_ONE_TILE_NV + 1) * _blocked_rows(nv)
  return ntile, nrow


# threads per group of the per-group kernels (factorization as _update_gradient_cholesky)
_ISLAND_THREADS = 64
# threads per group factored as one tile: more groups stay resident than with _ISLAND_THREADS
_TILE_THREADS = 32


@dataclasses.dataclass
class IslandContext:
  """Per-step island-ordered solver state (see the module docstring)."""

  # layout: tiles per side of the groups this solver solves (0 for dense and sleeping groups), and
  # the aligned DOF rows a world uses
  tilebase: wp.array2d[int]
  ntile: wp.array2d[int]
  nrow_used: wp.array[int]
  vbase: wp.array2d[int]
  row2dof: wp.array2d[int]
  dof2row: wp.array2d[int]
  dof_poolrow: wp.array2d[int]
  dof_local: wp.array2d[int]
  # DOFs in tile-aligned island rows (padding is zero)
  qacc: wp.array2d[float]
  Ma: wp.array2d[float]
  qfrc_smooth: wp.array2d[float]
  frc: wp.array2d[float]
  grad: wp.array3d[float]
  search: wp.array3d[float]
  mv: wp.array2d[float]
  # per constraint row (make_constraint writes rows in island order)
  row_jaref: wp.array2d[float]
  row_jv: wp.array2d[float]
  # whether a row's curvature is in H (H is updated only where this flips), and the weight of
  # its J'J term in the next H update (D or -D where it flipped, else 0)
  row_quad: wp.array2d[bool]
  row_w: wp.array2d[float]
  # a world's rows that flipped at the last evaluation (_h_flips applies them)
  quad_changed_ids: wp.array2d[int]
  quad_changed_count: wp.array[int]
  # elliptic cones: per-contact quadratics along the search (see _cone_quad), and the virtual
  # rows of cone-state rows (_cone_virtual_row: coefficients, weight, contact's first row)
  row_cone: wp.array2d[wp.vec3]
  cone_coef: wp.array2d[types.vec6]
  cone_weight: wp.array2d[float]
  cone_base: wp.array2d[int]
  # per-island dense blocks: Hessian and its factor
  H: wp.array3d[float]
  U: wp.array3d[float]
  # per-island solver state
  scale: wp.array2d[float]
  done: wp.array2d[bool]
  niter: wp.array2d[int]
  alpha: wp.array2d[float]
  improvement: wp.array2d[float]
  search_dot: wp.array2d[float]
  newton_decrement: wp.array2d[float]
  # whether a row of an island flipped at the last evaluation (its H and factor are updated)
  changed: wp.array2d[bool]
  # a row of the group changed state at the last evaluation (else the search direction is kept, as
  # the world solver's stable-state fast path)
  state_changed: wp.array2d[bool]
  # the search direction is unchanged: the group is done or stable (as the world solver's)
  search_unchanged: wp.array2d[bool]
  grad_scale: wp.array2d[float]
  # compacted lists of (world * ntree + group), see _group_layout
  group_count: wp.array[int]
  group_list: wp.array2d[int]
  nsolving: wp.array[int]
  world_done: wp.array[bool] = None  # all False: the initial J'DJ of every world


@wp.func
def _pool_rc(tilebase: int, ntile: int, la: int, lb: int) -> wp.vec2i:
  """Pool (row, column) of entry (la, lb) of an island block, read from its upper triangle."""
  i = wp.min(la, lb)
  j = wp.max(la, lb)
  return wp.vec2i(_TILE * (tilebase + (i // _TILE) * ntile + j // _TILE) + i % _TILE, j % _TILE)


@wp.func
def _stable(niter: int, state_changed: bool) -> bool:
  """No row changed state since the last step: the search direction is kept."""
  return niter > 0 and not state_changed


@wp.func
def _blocked(ndof: int) -> bool:
  return ndof > _ONE_TILE_NV


@wp.func
def _ntile(ndof: int) -> int:
  """Pool tiles per side; groups factored blocked have a row more (the augmented Cholesky's)."""
  return wp.where(_blocked(ndof), (ndof + _TILE) // _TILE, (ndof + _TILE - 1) // _TILE)


def _blocked_rows(nv: int) -> int:
  """Layout rows of a group factored blocked: the static size of the blocked Cholesky's vectors."""
  return (nv + _TILE) // _TILE * _TILE


@wp.func
def _group_rows(ntile: int, ndof: int, blocked_rows: int) -> int:
  # groups factored blocked: the blocked Cholesky's static vector size
  return wp.where(ntile > 0 and _blocked(ndof), blocked_rows, ntile * _TILE)


@cache_kernel
def _group_layout(ntree: int, blocked_rows: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    tree_dofadr: wp.array[int],
    tree_dofnum: wp.array[int],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    tree_island_in: wp.array2d[int],
    # In:
    ngroup_in: wp.array[int],
    group_nv_in: wp.array2d[int],
    group_dense_in: wp.array2d[wp.vec4i],
    group_asleep_in: wp.array2d[bool],
    tree_group_in: wp.array2d[int],
    island_group_in: wp.array2d[int],
    # Out:
    group_count_out: wp.array[int],
    group_list_out: wp.array2d[int],
    done_out: wp.array2d[bool],
    niter_out: wp.array2d[int],
    alpha_out: wp.array2d[float],
    improvement_out: wp.array2d[float],
    scale_out: wp.array2d[float],
    ntile_out: wp.array2d[int],
    tilebase_out: wp.array2d[int],
    vbase_out: wp.array2d[int],
    nrow_used_out: wp.array[int],
    row2dof_out: wp.array2d[int],
    iqacc_out: wp.array2d[float],
    iqfrc_smooth_out: wp.array2d[float],
    ifrc_out: wp.array2d[float],
    Ma_out: wp.array2d[float],
    grad_out: wp.array3d[float],
    search_out: wp.array3d[float],
    mv_out: wp.array2d[float],
    search_unchanged_out: wp.array2d[bool],
    changed_out: wp.array2d[bool],
    state_changed_out: wp.array2d[bool],
    grad_scale_out: wp.array2d[float],
    quad_changed_count_out: wp.array[int],
    world_done_out: wp.array[bool],
  ):
    """A world's awake groups by path (all, one tile, blocked), with their pool tiles and rows.

    Starts the awake groups' iterations, with their inverse inertia traces.
    """
    worldid, lane = wp.tid()
    lanes = wp.block_dim()
    ngroup = ngroup_in[worldid]
    # a group's inertia trace, tree by tree (a group's DOFs are whole trees)
    trace = wp.tile_zeros(shape=(ntree,), dtype=float, storage="shared")
    for it in range((ntree + lanes - 1) // lanes):
      tree = it * lanes + lane
      tg = int(-1)
      tsum = float(0.0)
      if tree < ntree:
        island = tree_island_in[worldid, tree]
        if island >= 0:
          tg = island_group_in[worldid, island]
        else:
          tg = tree_group_in[worldid, tree]
        if tg >= 0:
          # the diagonal is the last entry of a row of the lower triangle
          for dof in range(tree_dofadr[tree], tree_dofadr[tree] + tree_dofnum[tree]):
            tsum += M_in[worldid, M_rowadr[dof] + M_rownnz[dof] - 1]
      wp.tile_scatter_add(trace, wp.max(tg, 0), tsum, tg >= 0)
    _sync()
    for g in range(lane, ntree, lanes):
      ntile = int(0)
      # _done sets the island solver's groups'
      search_unchanged_out[worldid, g] = True
      changed_out[worldid, g] = False
      state_changed_out[worldid, g] = False
      grad_scale_out[worldid, g] = 1.0
      if g < ngroup:
        packed = worldid * ntree + g
        if group_asleep_in[worldid, g]:
          done_out[worldid, g] = True
        else:
          done_out[worldid, g] = False
          niter_out[worldid, g] = 0
          alpha_out[worldid, g] = 0.0
          improvement_out[worldid, g] = 0.0
          # MuJoCo scales an island's convergence tests by its inertia trace, not meaninertia * nv
          scale_out[worldid, g] = 1.0 / wp.max(types.MJ_MINVAL, trace[g])
          group_list_out[0, wp.atomic_add(group_count_out, 0, 1)] = packed  # kernel_analyzer: ignore[atomic]
          ndof = group_nv_in[worldid, g]
          ntile = _ntile(ndof)
          # lists 1, 2: factored as one tile; 3: blocked
          c = wp.where(_blocked(ndof), 3, ntile)
          group_list_out[c, wp.atomic_add(group_count_out, c, 1)] = packed  # kernel_analyzer: ignore[atomic]
          if ntile > 1:
            wp.atomic_add(group_count_out, 4, 1)  # kernel_analyzer: ignore[atomic]
      ntile_out[worldid, g] = ntile
    _sync()
    tiles = wp.tile_load(ntile_out[worldid], shape=ntree)
    wp.tile_store(tilebase_out[worldid], wp.tile_scan_exclusive(wp.tile_map(wp.mul, tiles, tiles)))
    # blocked_rows as a tile, not captured by a nested func: the kernel cache does not hash captures
    # of functions passed to tile_map
    rows = wp.tile_map(
      _group_rows,
      tiles,
      wp.tile_load(group_nv_in[worldid], shape=ntree),
      wp.tile_full(ntree, wp.static(blocked_rows), dtype=int),
    )
    wp.tile_store(vbase_out[worldid], wp.tile_scan_exclusive(rows))
    nrow = wp.tile_sum(rows)[0]
    if lane == 0:
      quad_changed_count_out[worldid] = 0
      world_done_out[worldid] = False
      nrow_used_out[worldid] = nrow
    for row in range(lane, nrow, lanes):
      row2dof_out[worldid, row] = -1
      iqacc_out[worldid, row] = 0.0
      iqfrc_smooth_out[worldid, row] = 0.0
      ifrc_out[worldid, row] = 0.0
      Ma_out[worldid, row] = 0.0
      grad_out[worldid, row, 0] = 0.0
      search_out[worldid, row, 0] = 0.0
      mv_out[worldid, row] = 0.0

  return kernel


@cache_kernel
def _init_rows(width: int):
  """width: of the dense rows; the row loop unrolls (past max_unroll) so its loads overlap."""

  @wp.kernel(module="unique", enable_backward=False, module_options={"max_unroll": width})
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_aref_in: wp.array2d[float],
    efc_island_in: wp.array2d[int],
    # In:
    island_dense_in: wp.array2d[wp.vec4i],
    vbase_in: wp.array2d[int],
    dof2row_in: wp.array2d[int],
    iJ_in: wp.array3d[float],
    iqacc_in: wp.array2d[float],
    lanes: int,
    # Out:
    row_jaref_out: wp.array2d[float],
  ):
    """J qacc - aref of a row at the start point, as _mul_jac_vec's subtract_aref mode."""
    worldid, lane = wp.tid()
    for row in range(lane, wp.min(nefc_in[worldid], efc_aref_in.shape[1]), lanes):
      islandid = efc_island_in[worldid, row]
      block = island_dense_in[worldid, islandid]
      jaref = -efc_aref_in[worldid, row]
      if block[0] > 0:
        vbase = vbase_in[worldid, islandid]
        for c in range(wp.static(width)):
          jaref += iJ_in[worldid, row, c] * iqacc_in[worldid, vbase + c]
      else:
        rowadr = efc_J_rowadr_in[worldid, row]
        for k in range(efc_J_rownnz_in[worldid, row]):
          adr = rowadr + k
          jaref += efc_J_in[worldid, 0, adr] * iqacc_in[worldid, dof2row_in[worldid, efc_J_colind_in[worldid, 0, adr]]]
      row_jaref_out[worldid, row] = jaref

  return kernel


@cache_kernel
def _jv(width: int):
  """J search per row, as _linesearch_jv_fused_kernel; groups with an unchanged search keep it."""

  @wp.kernel(module="unique", enable_backward=False, module_options={"max_unroll": width})
  def kernel(
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_island_in: wp.array2d[int],
    # In:
    island_dense_in: wp.array2d[wp.vec4i],
    vbase_in: wp.array2d[int],
    dof2row_in: wp.array2d[int],
    iJ_in: wp.array3d[float],
    search_in: wp.array2d[float],
    search_unchanged_in: wp.array2d[bool],
    # Out:
    row_jv_out: wp.array2d[float],
    quad_changed_count_out: wp.array[int],
  ):
    worldid, row = wp.tid()
    # the flips _rows lists next (_h_flips has applied the last, which may run beside _grad)
    if row == 0:
      quad_changed_count_out[worldid] = 0
    if row >= wp.min(nefc_in[worldid], row_jv_out.shape[1]):
      return
    g = efc_island_in[worldid, row]
    if g < 0:
      return
    if search_unchanged_in[worldid, g]:
      return
    jv = float(0.0)
    if island_dense_in[worldid, g][0] > 0:
      vbase = vbase_in[worldid, g]
      for c in range(wp.static(width)):
        jv += iJ_in[worldid, row, c] * search_in[worldid, vbase + c]
    else:
      rowadr = efc_J_rowadr_in[worldid, row]
      for k in range(efc_J_rownnz_in[worldid, row]):
        adr = rowadr + k
        jv += efc_J_in[worldid, 0, adr] * search_in[worldid, dof2row_in[worldid, efc_J_colind_in[worldid, 0, adr]]]
    row_jv_out[worldid, row] = jv

  return kernel


@wp.func_native(snippet="WP_TILE_SYNC();")
def _sync():
  pass


@wp.kernel(module="unique", enable_backward=False)
def _pool_seed(
  # Model:
  ntree: int,
  dof_treeid: wp.array[int],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  M_hinit_i: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  dof_frictionadr_in: wp.array2d[int],
  island_nv_in: wp.array2d[int],
  # In:
  nblock: int,
  group_count_in: wp.array[int],
  group_list_in: wp.array2d[int],
  row2dof_in: wp.array2d[int],
  tilebase_in: wp.array2d[int],
  ntile_in: wp.array2d[int],
  vbase_in: wp.array2d[int],
  skip_done: bool,
  done_in: wp.array2d[bool],
  group_dense_in: wp.array2d[wp.vec4i],
  # Out:
  H_out: wp.array3d[float],
):
  """Each group's block of M in the H pool, with quadratic DOF friction rows' D on the diagonal.

  DOF friction rows have no JTDAJ block (as in _update_gradient_init_h_sparse).
  """
  bid, lane = wp.tid()
  nthread = wp.block_dim()
  for e in range(bid, group_count_in[0], nblock):
    worldid = group_list_in[0, e] // ntree
    g = group_list_in[0, e] % ntree
    if skip_done:
      if done_in[worldid, g]:
        continue
    # groups with dense rows: _dense_rows_h assembles H from M
    if group_dense_in[worldid, g][0] > 0:
      continue
    n = island_nv_in[worldid, g]
    ntile = ntile_in[worldid, g]
    tilebase = tilebase_in[worldid, g]
    vbase = vbase_in[worldid, g]
    if ntile == 1:
      # the part of the tile the factorization reads (_dense_factor: half size for small groups)
      size = wp.where(n <= _TILE // 2, _TILE // 2, _TILE)
      for idx in range(lane, size * size, nthread):
        r = idx // size
        c = idx % size
        H_out[worldid, _TILE * tilebase + r, c] = wp.where(r == c and r >= n, 1.0, 0.0)
    else:
      # the upper tiles the factorization reads, then identity on the padding diagonal
      for t in range(ntile * ntile):
        if t // ntile <= t % ntile:
          wp.tile_store(H_out[worldid], wp.tile_zeros(shape=(_TILE, _TILE), dtype=float), offset=(_TILE * (tilebase + t), 0))
      _sync()
      for la in range(n + lane, ntile * _TILE, nthread):
        rc = _pool_rc(tilebase, ntile, la, la)
        H_out[worldid, rc[0], rc[1]] = 1.0
    _sync()
    col = int(0)
    while col < n:
      tree = dof_treeid[row2dof_in[worldid, vbase + col]]
      dofadr = tree_dofadr[tree]
      last = dofadr + tree_dofnum[tree] - 1
      beg = M_rowadr[dofadr]
      base = col - dofadr
      for ec in range(beg + lane, M_rowadr[last] + M_rownnz[last], nthread):
        rc = _pool_rc(tilebase, ntile, M_colind[ec] + base, M_hinit_i[ec] + base)
        H_out[worldid, rc[0], rc[1]] = M_in[worldid, ec]
      col += tree_dofnum[tree]
    _sync()
    for la in range(lane, n, nthread):
      efcid = dof_frictionadr_in[worldid, row2dof_in[worldid, vbase + la]]
      if efcid >= 0:
        if efc_state_in[worldid, efcid] == types.ConstraintState.QUADRATIC.value:
          rc = _pool_rc(tilebase, ntile, la, la)
          H_out[worldid, rc[0], rc[1]] += efc_D_in[worldid, efcid]
    _sync()


@wp.kernel
def _mul_m_search(
  # Model:
  ntree: int,
  M_mulm_rowadr: wp.array[int],
  M_mulm_col: wp.array[int],
  M_mulm_madr: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  island_nv_in: wp.array2d[int],
  # In:
  group_count_in: wp.array[int],
  group_list_in: wp.array2d[int],
  vbase_in: wp.array2d[int],
  row2dof_in: wp.array2d[int],
  search_in: wp.array2d[float],
  search_unchanged_in: wp.array2d[bool],
  nslot: int,
  # Out:
  mv_out: wp.array2d[float],
):
  """M search of a group's DOF for the linesearch, warp slots striding over the groups.

  An unchanged search direction (done or stable) keeps it.
  """
  slot, lane = wp.tid()
  for e in range(slot, group_count_in[0], nslot):
    worldid = group_list_in[0, e] // ntree
    g = group_list_in[0, e] % ntree
    if search_unchanged_in[worldid, g]:
      continue
    vbase = vbase_in[worldid, g]
    for la in range(lane, island_nv_in[worldid, g], _TILE):
      # sparse M: a group's DOFs are whole trees in order (column col of DOF dof: row + col - dof)
      row = vbase + la
      dof = row2dof_in[worldid, row]
      mv = float(0.0)
      for k in range(M_mulm_rowadr[dof], M_mulm_rowadr[dof + 1]):
        mv += M_in[worldid, M_mulm_madr[k]] * search_in[worldid, row + M_mulm_col[k] - dof]
      mv_out[worldid, row] = mv


@cache_kernel
def _rows(initial: bool, elliptic: bool):
  INITIAL = initial
  ELLIPTIC = elliptic
  # the world solver's row evaluation; elliptic: the virtual rows of the dense groups' H
  ROW = _constraint_row(elliptic)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    nefc_in: wp.array[int],
    contact_friction_in: wp.array[types.vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_D_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    efc_island_in: wp.array2d[int],
    nacon_in: wp.array[int],
    # In:
    done_in: wp.array2d[bool],
    row_jaref_in: wp.array2d[float],
    lanes: int,
    # Data out:
    efc_force_out: wp.array2d[float],
    efc_state_out: wp.array2d[int],
    # Out:
    row_quad_out: wp.array2d[bool],
    row_w_out: wp.array2d[float],
    quad_changed_ids_out: wp.array2d[int],
    quad_changed_count_out: wp.array[int],
    changed_out: wp.array2d[bool],
    state_changed_out: wp.array2d[bool],
    cone_coef_out: wp.array2d[types.vec6],
    cone_weight_out: wp.array2d[float],
    cone_base_out: wp.array2d[int],
  ):
    """Force and state of a row; rows whose quadratic state flips are listed for _h_flips.

    Elliptic: a cone-state row's curvature is a weighted virtual row over its contact's rows
    (_cone_virtual_row), which _dense_rows_h folds into the dense groups' tile product.
    """
    worldid, lane = wp.tid()
    for row in range(lane, wp.min(nefc_in[worldid], efc_island_in.shape[1]), lanes):
      islandid = efc_island_in[worldid, row]
      if done_in[worldid, islandid]:
        continue
      constraint_type = efc_type_in[worldid, row]
      old_state = efc_state_out[worldid, row]
      state = wp.static(ROW)(
        opt_impratio_invsqrt,
        contact_friction_in,
        contact_dim_in,
        contact_efc_address_in,
        efc_type_in,
        efc_id_in,
        efc_D_in,
        efc_frictionloss_in,
        nacon_in,
        worldid,
        row,
        _is_equality(constraint_type),
        _is_friction(constraint_type),
        row_jaref_in,
        efc_force_out,
        efc_state_out,
        cone_coef_out,
        cone_weight_out,
        cone_base_out,
      )
      if state < 0:
        continue
      if wp.static(not INITIAL):
        if old_state != state:
          state_changed_out[worldid, islandid] = True
      D = efc_D_in[worldid, row]
      quad = state == types.ConstraintState.QUADRATIC.value
      w = float(0.0)
      if wp.static(ELLIPTIC):
        # H is assembled again at every point (_JTDACJ_sparse): no incremental update and no
        # stable-state shortcut
        if wp.static(not INITIAL):
          changed_out[worldid, islandid] = True
          state_changed_out[worldid, islandid] = True
      elif wp.static(INITIAL):
        row_quad_out[worldid, row] = quad
        w = wp.where(quad, D, 0.0)
      elif quad != row_quad_out[worldid, row]:
        row_quad_out[worldid, row] = quad
        changed_out[worldid, islandid] = True
        w = wp.where(quad, D, -D)
      row_w_out[worldid, row] = w

      # a flipped row's rank-1 update of H is listed for _h_flips, as in the world solver
      if wp.static(not INITIAL):
        if w != 0.0:
          flip = wp.atomic_add(quad_changed_count_out, worldid, 1)  # kernel_analyzer: ignore[atomic]
          quad_changed_ids_out[worldid, flip] = row

  return kernel


@wp.kernel
def _frc_rows(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_force_in: wp.array2d[float],
  efc_island_in: wp.array2d[int],
  # In:
  island_dense_in: wp.array2d[wp.vec4i],
  done_in: wp.array2d[bool],
  niter_in: wp.array2d[int],
  state_changed_in: wp.array2d[bool],
  dof2row_in: wp.array2d[int],
  lanes: int,
  # Out:
  frc_out: wp.array2d[float],
):
  """J'force of the sparse rows of changed groups (the final _grad recovers the rest)."""
  worldid, lane = wp.tid()
  for row in range(lane, wp.min(nefc_in[worldid], efc_island_in.shape[1]), lanes):
    g = efc_island_in[worldid, row]
    if done_in[worldid, g] or island_dense_in[worldid, g][0] > 0:
      continue
    if _stable(niter_in[worldid, g], state_changed_in[worldid, g]):
      continue
    force = efc_force_in[worldid, row]
    if force != 0.0:
      rowadr = efc_J_rowadr_in[worldid, row]
      dof2row = dof2row_in[worldid]
      colind = efc_J_colind_in[worldid, 0]
      for k in range(efc_J_rownnz_in[worldid, row]):
        wp.atomic_add(
          frc_out[worldid], dof2row[colind[rowadr + k]], efc_J_in[worldid, 0, rowadr + k] * force
        )  # kernel_analyzer: ignore[atomic]


@wp.kernel(module="unique", enable_backward=False)
def _h_flips(
  # Data in:
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_island_in: wp.array2d[int],
  # In:
  island_dense_in: wp.array2d[wp.vec4i],
  tilebase_in: wp.array2d[int],
  ntile_in: wp.array2d[int],
  vbase_in: wp.array2d[int],
  dof2row_in: wp.array2d[int],
  iJ_in: wp.array3d[float],
  row_w_in: wp.array2d[float],
  quad_changed_ids_in: wp.array2d[int],
  quad_changed_count_in: wp.array[int],
  slots: int,
  # Out:
  H_out: wp.array3d[float],
):
  """A flipped row's rank-1 update w J'J of its group's H, as _update_gradient_h_incremental_sparse.

  A warp per listed row: the lanes split the upper triangle of the row's dense or sparse entries.
  """
  worldid, slot, lane = wp.tid()
  for c in range(slot, quad_changed_count_in[worldid], slots):
    row = quad_changed_ids_in[worldid, c]
    g = efc_island_in[worldid, row]
    w = row_w_in[worldid, row]
    width = island_dense_in[worldid, g][0]
    if width > 0:
      # a dense group is one pool tile
      row0 = _TILE * tilebase_in[worldid, g]
      for entry in range(lane, width * (width + 1) // 2, wp.static(_JTDAJ_THREADS_PER_GROUP)):
        i = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)
        j = entry - i * (i + 1) // 2
        h = w * iJ_in[worldid, row, i] * iJ_in[worldid, row, j]
        if h != 0.0:
          wp.atomic_add(H_out[worldid, row0 + j], i, h)  # kernel_analyzer: ignore[atomic]
    else:
      rowadr = efc_J_rowadr_in[worldid, row]
      rownnz = efc_J_rownnz_in[worldid, row]
      tilebase = tilebase_in[worldid, g]
      ntile = ntile_in[worldid, g]
      vbase = vbase_in[worldid, g]
      for entry in range(lane, rownnz * (rownnz + 1) // 2, wp.static(_JTDAJ_THREADS_PER_GROUP)):
        i = int((wp.sqrt(float(8 * entry + 1)) - 1.0) * 0.5)
        j = entry - i * (i + 1) // 2
        h = w * efc_J_in[worldid, 0, rowadr + i] * efc_J_in[worldid, 0, rowadr + j]
        if h != 0.0:
          rowi = dof2row_in[worldid, efc_J_colind_in[worldid, 0, rowadr + i]]
          rowj = dof2row_in[worldid, efc_J_colind_in[worldid, 0, rowadr + j]]
          rc = _pool_rc(tilebase, ntile, rowi - vbase, rowj - vbase)
          wp.atomic_add(H_out[worldid, rc[0]], rc[1], h)  # kernel_analyzer: ignore[atomic]


@wp.func
def _quad_weight(D: float, state: int) -> float:
  return wp.where(state == types.ConstraintState.QUADRATIC.value, D, 0.0)


@cache_kernel
def _dense_rows_h(width: int, elliptic: bool):
  """H = M + J'DJ over a group's dense rows, as _update_gradient_JTDAJ_dense_tiled.

  Elliptic: cone-state rows enter as their virtual rows (_rows), as in the world solver.
  """
  W = width
  K = 16
  ELLIPTIC = elliptic

  @wp.kernel(module="unique", enable_backward=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Model:
    ntree: int,
    dof_treeid: wp.array[int],
    tree_dofadr: wp.array[int],
    tree_dofnum: wp.array[int],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    M_colind: wp.array[int],
    M_hinit_i: wp.array[int],
    # Data in:
    M_in: wp.array2d[float],
    efc_D_in: wp.array2d[float],
    efc_state_in: wp.array2d[int],
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    nblock: int,
    initial: bool,
    group_dense_in: wp.array2d[wp.vec4i],
    group_nv_in: wp.array2d[int],
    tilebase_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    row2dof_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    cone_coef_in: wp.array2d[types.vec6],
    cone_weight_in: wp.array2d[float],
    cone_base_in: wp.array2d[int],
    iJ_in: wp.array3d[float],
    # Out:
    H_out: wp.array3d[float],
  ):
    bid, lane = wp.tid()
    lanes = wp.block_dim()
    for e in range(bid, group_count_in[0], nblock):
      worldid = group_list_in[0, e] // ntree
      g = group_list_in[0, e] % ntree
      block = group_dense_in[worldid, g]
      if block[0] == 0:
        continue
      if not initial:
        if done_in[worldid, g]:
          continue
      rowadr = block[3]
      nrow = block[2]
      row0 = _TILE * tilebase_in[worldid, g]
      # views ending at the group's last row: the loads zero-fill past it (and past column W)
      J = wp.array(ptr=iJ_in[worldid].ptr, shape=(rowadr + nrow, W), dtype=float)
      D = wp.array(ptr=efc_D_in[worldid].ptr, shape=(rowadr + nrow,), dtype=float)
      state = wp.array(ptr=efc_state_in[worldid].ptr, shape=(rowadr + nrow,), dtype=int)
      # M's upper triangle, tree by tree (a group's DOFs are whole trees in order, each tree's CSR
      # entries contiguous); identity on padding. Uniform trip counts: the scatters are collective
      m_tile = wp.tile_zeros(shape=(wp.static(W * W),), dtype=float, storage="shared")
      ndof = group_nv_in[worldid, g]
      vbase = vbase_in[worldid, g]
      c = int(0)
      while c < ndof:
        tree = dof_treeid[row2dof_in[worldid, vbase + c]]
        dofadr = tree_dofadr[tree]
        last = dofadr + tree_dofnum[tree] - 1
        beg = M_rowadr[dofadr]
        nent = M_rowadr[last] + M_rownnz[last] - beg
        base = c - dofadr
        for it in range((nent + lanes - 1) // lanes):
          ent = it * lanes + lane
          enable = ent < nent
          ec = beg + wp.where(enable, ent, 0)
          pos = (M_colind[ec] + base) * W + M_hinit_i[ec] + base
          wp.tile_scatter_add(m_tile, pos, wp.where(enable, M_in[worldid, ec], 0.0), enable)
        c += tree_dofnum[tree]
      for it in range((W + lanes - 1) // lanes):
        i = it * lanes + lane
        enable = i >= ndof and i < W
        wp.tile_scatter_add(m_tile, wp.where(enable, i * W + i, 0), 1.0, enable)
      h = wp.tile_reshape(m_tile, (W, W))
      for k in range(0, nrow, K):
        if wp.static(ELLIPTIC):
          J_k = wp.tile_load(J, shape=(K, W), offset=(rowadr + k, 0), bounds_check=True, storage="shared")
          state_k = wp.tile_load(state, shape=K, offset=rowadr + k, bounds_check=True)
          # not a shared wp.func: Warp does not reserve shared memory for a tile returned by one
          active = wp.tile_map(_active_check, wp.tile_arange(K, dtype=int), wp.tile_ones(shape=K, dtype=int) * (nrow - k))
          D_k = wp.tile_map(
            _row_weight,
            wp.tile_load(D, shape=K, offset=rowadr + k, bounds_check=True),
            state_k,
            wp.tile_load(cone_weight_in[worldid], shape=K, offset=rowadr + k, bounds_check=True),
            active,
          )
          if wp.tile_reduce(wp.add, wp.tile_map(_is_cone, state_k, active))[0] > 0:
            _rewrite_cone_rows(
              iJ_in, efc_state_in, cone_coef_in, cone_base_in, worldid, lane, lanes, rowadr + nrow, rowadr + k, K, W, J_k
            )
        else:
          J_k = wp.tile_load(J, shape=(K, W), offset=(rowadr + k, 0), bounds_check=True)
          D_k = wp.tile_map(
            _quad_weight,
            wp.tile_load(D, shape=K, offset=rowadr + k, bounds_check=True),
            wp.tile_load(state, shape=K, offset=rowadr + k, bounds_check=True),
          )
        JD = wp.tile_map(wp.mul, wp.tile_transpose(J_k), wp.tile_broadcast(D_k, shape=(W, K)))
        h += wp.tile_matmul(JD, J_k)
      wp.tile_store(H_out[worldid], h, offset=(row0, 0))
      # the rest of the tile: identity on the padding diagonal
      if wp.static(W < _TILE):
        for idx in range(lane, wp.static(_TILE * _TILE), lanes):
          r = idx // _TILE
          c = idx % _TILE
          if r >= W or c >= W:
            H_out[worldid, row0 + r, c] = wp.where(r == c, 1.0, 0.0)

  return kernel


@cache_kernel
def _done(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    ntree: int,
    opt_tolerance: wp.array[float],
    opt_iterations: int,
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    group_nv_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    scale_in: wp.array2d[float],
    niter_in: wp.array2d[int],
    alpha_in: wp.array2d[float],
    improvement_in: wp.array2d[float],
    grad_in: wp.array2d[float],
    grad_scale_in: wp.array2d[float],
    newton_decrement_in: wp.array2d[float],
    state_changed_in: wp.array2d[bool],
    # Data out:
    solver_niter_out: wp.array[int],
    overflow_out: wp.array[int],
    # Out:
    done_out: wp.array2d[bool],
    nsolving_out: wp.array[int],
    search_unchanged_out: wp.array2d[bool],
  ):
    """Convergence test per group, as _solve_done (while states hold, grad_scale scales grad)."""
    e = wp.tid()
    if e >= group_count_in[0]:
      return
    worldid = group_list_in[0, e] // ntree
    g = group_list_in[0, e] % ntree
    if done_out[worldid, g]:
      return
    niter = niter_in[worldid, g]
    converged = False
    if niter > 0:
      vbase = vbase_in[worldid, g]
      grad_scale = grad_scale_in[worldid, g]
      grad_dot = float(0.0)
      for la in range(group_nv_in[worldid, g]):
        grad_dot += grad_in[worldid, vbase + la] * grad_in[worldid, vbase + la]
      grad_dot *= grad_scale * grad_scale
      # MuJoCo scales an island's tests by its inertia trace
      converged = _converged(
        opt_tolerance[worldid % opt_tolerance.shape[0]],
        scale_in[worldid, g],
        alpha_in[worldid, g],
        improvement_in[worldid, g],
        grad_dot,
        newton_decrement_in[worldid, g] * grad_scale * grad_scale,
      )
    search_unchanged_out[worldid, g] = True
    if converged or niter >= opt_iterations:
      done_out[worldid, g] = True
      if not converged and opt_iterations > 0:
        if wp.static(bool(warn_overflow & OverflowType.ITERATIONS)):
          wp.printf(
            "solver iterations limit reached - please increase iterations beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
            opt_iterations,
          )
        wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.ITERATIONS))
      wp.atomic_max(solver_niter_out, worldid, niter)  # kernel_analyzer: ignore[atomic]
    else:
      wp.atomic_add(nsolving_out, 0, 1)  # kernel_analyzer: ignore[atomic]
      search_unchanged_out[worldid, g] = _stable(niter, state_changed_in[worldid, g])

  return kernel


@cache_kernel
def _grad(final: bool):
  FINAL = final

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    ntree: int,
    # Data in:
    efc_force_in: wp.array2d[float],
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    nblock: int,
    group_dense_in: wp.array2d[wp.vec4i],
    iJ_in: wp.array3d[float],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    niter_in: wp.array2d[int],
    state_changed_in: wp.array2d[bool],
    Ma_in: wp.array2d[float],
    iqfrc_smooth_in: wp.array2d[float],
    grad_scale_in: wp.array2d[float],
    # Out:
    frc_out: wp.array2d[float],
    grad_out: wp.array2d[float],
    nsolving_out: wp.array[int],
  ):
    """J'force of dense rows and the gradient per group, as _update_gradient_grad.

    The final pass recovers J'force that stable groups skipped from their gradient.
    """
    slot, lane = wp.tid()
    if wp.static(not FINAL):
      # _done counts the groups still iterating (a memset node costs more)
      if slot == 0 and lane == 0:
        nsolving_out[0] = 0
    for e in range(slot, group_count_in[0], nblock):
      worldid = group_list_in[0, e] // ntree
      g = group_list_in[0, e] % ntree
      if wp.static(not FINAL):
        if done_in[worldid, g] or _stable(niter_in[worldid, g], state_changed_in[worldid, g]):
          continue
      block = group_dense_in[worldid, g]
      if wp.static(FINAL):
        # J'force is at the solution unless the last iteration was stable (it skips J'force)
        if not _stable(niter_in[worldid, g], state_changed_in[worldid, g]):
          continue
      vbase = vbase_in[worldid, g]
      if wp.static(FINAL):
        grad_scale = grad_scale_in[worldid, g]
        for col in range(lane, ntile_in[worldid, g] * _TILE, _TILE):
          row = vbase + col
          frc_out[worldid, row] = Ma_in[worldid, row] - iqfrc_smooth_in[worldid, row] - grad_scale * grad_out[worldid, row]
        continue
      for col in range(lane, ntile_in[worldid, g] * _TILE, _TILE):
        row = vbase + col
        frc = float(0.0)
        if block[0] > 0:
          if col < block[0]:
            # independent loads (no test of the force first): the row loop is latency bound
            for r in range(block[3], block[3] + block[2]):
              frc += iJ_in[worldid, r, col] * efc_force_in[worldid, r]
          frc_out[worldid, row] = frc
        else:
          frc = frc_out[worldid, row]
        if wp.static(not FINAL):
          grad_out[worldid, row] = Ma_in[worldid, row] - iqfrc_smooth_in[worldid, row] - frc

  return kernel


@cache_kernel
def _factor(rows: int):
  """Newton directions of blocked groups (H packed into U), as _update_gradient_cholesky_blocked."""
  BS = types.TILE_SIZE_JTDAJ_DENSE
  factorize = create_blocked_cholesky_augmented_factorize_solve_newton_func(BS, rows, rows)
  solve_only = create_blocked_cholesky_solve_newton_func(BS, rows, rows)

  @wp.kernel(module="unique", enable_backward=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Model:
    ntree: int,
    # Data in:
    nisland_in: wp.array[int],
    island_nv_in: wp.array2d[int],
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    nblock: int,
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    niter_in: wp.array2d[int],
    state_changed_in: wp.array2d[bool],
    grad_in: wp.array3d[float],
    H_in: wp.array3d[float],
    # Out:
    changed_out: wp.array2d[bool],
    U_out: wp.array3d[float],
    search_out: wp.array3d[float],
    search_dot_out: wp.array2d[float],
    newton_decrement_out: wp.array2d[float],
    grad_scale_out: wp.array2d[float],
  ):
    bid, lane = wp.tid()
    nthread = wp.block_dim()
    for e in range(bid, group_count_in[3], nblock):
      packed = group_list_in[3, e]
      worldid = packed // ntree
      islandid = packed % ntree
      if islandid >= nisland_in[worldid]:
        continue
      if done_in[worldid, islandid]:
        continue
      if _stable(niter_in[worldid, islandid], state_changed_in[worldid, islandid]):
        continue
      n = island_nv_in[worldid, islandid]
      ntile = ntile_in[worldid, islandid]
      tilebase = tilebase_in[worldid, islandid]
      vbase = vbase_in[worldid, islandid]
      R = ntile * _TILE
      # with the augmented row
      size = (n + BS) // BS * BS
      A = wp.array(ptr=U_out[worldid].ptr + wp.uint64(tilebase * _TILE * _TILE * 4), shape=(R, R), dtype=float)
      b = wp.array(ptr=grad_in[worldid].ptr + wp.uint64(vbase * 4), shape=(rows, 1), dtype=float)
      x = wp.array(ptr=search_out[worldid].ptr + wp.uint64(vbase * 4), shape=(rows, 1), dtype=float)
      if changed_out[worldid, islandid] or niter_in[worldid, islandid] == 0:
        # the factorization reads the upper triangle: copy the pool's upper tiles, whole (_pool_seed
        # set identity on the padding)
        for t in range(ntile * ntile):
          ti = t // ntile
          tj = t % ntile
          if ti <= tj:
            tile = wp.tile_load(H_in[worldid], shape=(_TILE, _TILE), offset=(_TILE * (tilebase + t), 0))
            wp.tile_store(A, tile, offset=(_TILE * ti, _TILE * tj))
        _sync()
        sums = wp.static(factorize)(A, b, size, A, x)
      else:
        sums = wp.static(solve_only)(A, b, size, x)
      _sync()  # every thread has read the flag
      if lane == 0:
        changed_out[worldid, islandid] = False
        search_dot_out[worldid, islandid] = sums[0]
        newton_decrement_out[worldid, islandid] = sums[1]
        grad_scale_out[worldid, islandid] = 1.0

  return kernel


def _tile_newton(n: int, ntile: int):
  """Newton direction of a group factored as one n x n tile, refactored every time."""
  newton_tile = create_newton_tile_func(n)

  if ntile == 1:
    return newton_tile

  R = n - _TILE
  PAD = max(ntile * _TILE - n, 1)

  @wp.func
  def func(
    # In:
    H_in: wp.array2d[float],
    row0: int,
    grad_in: wp.array[float],
    vbase: int,
    # Out:
    search_out: wp.array[float],
  ) -> wp.vec2:
    # the block's upper tiles (0, 0), (0, 1), (1, 1), cut to n; the factorization reads no others
    U = wp.tile_zeros(shape=(n, n), dtype=float, storage="shared")
    wp.tile_assign(U, wp.tile_load(H_in, shape=(_TILE, _TILE), offset=(row0, 0)))
    wp.tile_assign(U, wp.tile_load(H_in, shape=(_TILE, R), offset=(row0 + _TILE, 0)), offset=(0, _TILE))
    wp.tile_assign(U, wp.tile_load(H_in, shape=(R, R), offset=(row0 + 3 * _TILE, 0)), offset=(_TILE, _TILE))
    wp.tile_cholesky_inplace(U, fill_mode="upper")
    b = wp.tile_load(grad_in, shape=n, offset=vbase)
    x = wp.tile_cholesky_solve(U, b, fill_mode="upper")
    wp.tile_store(search_out, wp.tile_map(wp.neg, x), offset=vbase)
    if wp.static(n < ntile * _TILE):
      # the group's DOF rows past n are padding
      wp.tile_store(search_out, wp.tile_zeros(shape=PAD, dtype=float), offset=vbase + n)
    return wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, b, x))[0]

  return func


@cache_kernel
def _dense_factor(ntile: int):
  """Newton directions of one-tile groups; small ones factor at half size (cuSolverDx is slower)."""
  N = _TILE if ntile == 1 else _ONE_TILE_NV
  HALF = _TILE // 2
  full = _tile_newton(N, ntile)
  half = _tile_newton(HALF, 1) if ntile == 1 else full

  @wp.kernel(module="unique", enable_backward=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # Model:
    ntree: int,
    # Data in:
    nisland_in: wp.array[int],
    island_nv_in: wp.array2d[int],
    # In:
    group_count_in: wp.array[int],
    group_list_in: wp.array2d[int],
    cls: int,
    nblock: int,
    tilebase_in: wp.array2d[int],
    ntile_in: wp.array2d[int],
    vbase_in: wp.array2d[int],
    done_in: wp.array2d[bool],
    niter_in: wp.array2d[int],
    state_changed_in: wp.array2d[bool],
    grad_in: wp.array2d[float],
    H_in: wp.array3d[float],
    # Out:
    search_out: wp.array2d[float],
    search_dot_out: wp.array2d[float],
    newton_decrement_out: wp.array2d[float],
    grad_scale_out: wp.array2d[float],
  ):
    bid, lane = wp.tid()
    for e in range(bid, group_count_in[cls], nblock):
      worldid = group_list_in[cls, e] // ntree
      islandid = group_list_in[cls, e] % ntree
      if islandid >= nisland_in[worldid]:
        continue
      if done_in[worldid, islandid]:
        continue
      if ntile_in[worldid, islandid] != wp.static(ntile):
        continue
      nv = island_nv_in[worldid, islandid]
      if nv > N:
        continue
      if _stable(niter_in[worldid, islandid], state_changed_in[worldid, islandid]):
        continue
      vbase = vbase_in[worldid, islandid]
      row0 = _TILE * tilebase_in[worldid, islandid]
      sums = wp.vec2(0.0)
      if wp.static(ntile == 1) and nv <= HALF:
        sums = wp.static(half)(H_in[worldid], row0, grad_in[worldid], vbase, search_out[worldid])
      else:
        sums = wp.static(full)(H_in[worldid], row0, grad_in[worldid], vbase, search_out[worldid])
      if lane == 0:
        search_dot_out[worldid, islandid] = sums[0]
        newton_decrement_out[worldid, islandid] = sums[1]
        grad_scale_out[worldid, islandid] = 1.0

  return kernel


@wp.kernel
def _scatter_dofs(
  # In:
  lanes: int,
  nrow_used_in: wp.array[int],
  row2dof_in: wp.array2d[int],
  iqacc_in: wp.array2d[float],
  frc_in: wp.array2d[float],
  iMa_in: wp.array2d[float],
  # Data out:
  qacc_out: wp.array2d[float],
  qfrc_constraint_out: wp.array2d[float],
  efc_Ma_out: wp.array2d[float],
):
  worldid, lane = wp.tid()
  for row in range(lane, nrow_used_in[worldid], lanes):
    dofid = row2dof_in[worldid, row]
    if dofid >= 0:
      qacc_out[worldid, dofid] = iqacc_in[worldid, row]
      qfrc_constraint_out[worldid, dofid] = frc_in[worldid, row]
      efc_Ma_out[worldid, dofid] = iMa_in[worldid, row]


@wp.kernel
def _dofs(
  # Model:
  dof_treeid: wp.array[int],
  M_mulm_rowadr: wp.array[int],
  M_mulm_col: wp.array[int],
  M_mulm_madr: wp.array[int],
  # Data in:
  qacc_warmstart_in: wp.array2d[float],
  M_in: wp.array2d[float],
  tree_awake_in: wp.array2d[int],
  qfrc_smooth_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  map_dof2idof_in: wp.array2d[int],
  # In:
  warmstart: bool,
  dof_group_in: wp.array2d[int],
  group_idofadr_in: wp.array2d[int],
  tilebase_in: wp.array2d[int],
  ntile_in: wp.array2d[int],
  vbase_in: wp.array2d[int],
  # Data out:
  qacc_out: wp.array2d[float],
  qfrc_constraint_out: wp.array2d[float],
  efc_Ma_out: wp.array2d[float],
  # Out:
  row2dof_out: wp.array2d[int],
  dof2row_out: wp.array2d[int],
  dof_poolrow_out: wp.array2d[int],
  dof_local_out: wp.array2d[int],
  iqacc_out: wp.array2d[float],
  iqfrc_smooth_out: wp.array2d[float],
  iMa_out: wp.array2d[float],
):
  """A DOF's aligned row, pool row (first tile column), local index, start point and smooth force.

  DOFs outside all groups have no constraints (M qacc_smooth = qfrc_smooth); sleeping ones rest.
  """
  worldid, dofid = wp.tid()
  row = int(-1)
  poolrow = int(-1)
  local = int(-1)
  g = dof_group_in[worldid, dofid]
  if g >= 0:
    ntile = ntile_in[worldid, g]
    if ntile > 0:
      local = map_dof2idof_in[worldid, dofid] - group_idofadr_in[worldid, g]
      row = vbase_in[worldid, g] + local
      poolrow = _TILE * (tilebase_in[worldid, g] + (local // _TILE) * ntile) + local % _TILE
      row2dof_out[worldid, row] = dofid
      iqacc_out[worldid, row] = wp.where(warmstart, qacc_warmstart_in[worldid, dofid], qacc_smooth_in[worldid, dofid])
      iqfrc_smooth_out[worldid, row] = qfrc_smooth_in[worldid, dofid]
      # M qacc at the start point (M's columns of a DOF are in its tree)
      ma = float(0.0)
      for k in range(M_mulm_rowadr[dofid], M_mulm_rowadr[dofid + 1]):
        col = M_mulm_col[k]
        if warmstart:
          qacc = qacc_warmstart_in[worldid, col]
        else:
          qacc = qacc_smooth_in[worldid, col]
        ma += M_in[worldid, M_mulm_madr[k]] * qacc
      iMa_out[worldid, row] = ma
  dof2row_out[worldid, dofid] = row
  dof_poolrow_out[worldid, dofid] = poolrow
  dof_local_out[worldid, dofid] = local
  if tree_awake_in:
    if tree_awake_in[worldid, dof_treeid[dofid]] == 0:
      qacc_out[worldid, dofid] = 0.0
      qfrc_constraint_out[worldid, dofid] = 0.0
      efc_Ma_out[worldid, dofid] = 0.0
      return
  if g < 0:
    qacc_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]
    qfrc_constraint_out[worldid, dofid] = 0.0
    efc_Ma_out[worldid, dofid] = qfrc_smooth_in[worldid, dofid]


def _create_context(m: types.Model, d: types.Data) -> IslandContext:
  """The solver's state, allocated once per Data (in a graph capture, allocations are nodes)."""
  key = (d.nworld, m.nv, m.ntree, d.njmax, m.opt.cone)
  ctx = d.__dict__.get("_island_context")
  if ctx is not None and ctx[0] == key:
    return ctx[1]
  ntile_pool, nrow = _island_pool_size(m.nv, m.ntree, _TILE)
  nworld, njmax, ntree = d.nworld, d.njmax, m.ntree
  elliptic = m.opt.cone == types.ConeType.ELLIPTIC
  pool = (nworld, ntile_pool * _TILE, _TILE)
  ctx = IslandContext(
    tilebase=wp.empty((nworld, ntree), dtype=int),
    ntile=wp.empty((nworld, ntree), dtype=int),
    nrow_used=wp.empty(nworld, dtype=int),
    vbase=wp.empty((nworld, ntree), dtype=int),
    row2dof=wp.empty((nworld, nrow), dtype=int),
    dof2row=wp.empty((nworld, m.nv), dtype=int),
    dof_poolrow=wp.empty((nworld, m.nv), dtype=int),
    dof_local=wp.empty((nworld, m.nv), dtype=int),
    qacc=wp.empty((nworld, nrow), dtype=float),
    Ma=wp.empty((nworld, nrow), dtype=float),
    qfrc_smooth=wp.empty((nworld, nrow), dtype=float),
    frc=wp.empty((nworld, nrow), dtype=float),
    grad=wp.empty((nworld, nrow, 1), dtype=float),
    search=wp.empty((nworld, nrow, 1), dtype=float),
    mv=wp.empty((nworld, nrow), dtype=float),
    row_jaref=wp.empty((nworld, njmax), dtype=float),
    row_jv=wp.empty((nworld, njmax), dtype=float),
    row_quad=wp.empty((nworld, njmax), dtype=bool),
    row_w=wp.empty((nworld, njmax), dtype=float),
    quad_changed_ids=wp.empty((nworld, njmax), dtype=int),
    quad_changed_count=wp.empty(nworld, dtype=int),
    row_cone=wp.empty((nworld, njmax) if elliptic else (0, 0), dtype=wp.vec3),
    # weights are tile-loaded in whole row chunks, as the dense rows
    cone_coef=wp.empty((nworld, njmax) if elliptic else (0, 0), dtype=types.vec6),
    cone_weight=wp.empty((nworld, njmax + _TILE) if elliptic else (0, 0), dtype=float),
    cone_base=wp.empty((nworld, njmax) if elliptic else (0, 0), dtype=int),
    H=wp.empty(pool, dtype=float),
    U=wp.empty(pool, dtype=float),
    scale=wp.empty((nworld, ntree), dtype=float),
    done=wp.empty((nworld, ntree), dtype=bool),
    niter=wp.empty((nworld, ntree), dtype=int),
    alpha=wp.empty((nworld, ntree), dtype=float),
    improvement=wp.empty((nworld, ntree), dtype=float),
    search_dot=wp.empty((nworld, ntree), dtype=float),
    newton_decrement=wp.empty((nworld, ntree), dtype=float),
    changed=wp.empty((nworld, ntree), dtype=bool),
    state_changed=wp.empty((nworld, ntree), dtype=bool),
    search_unchanged=wp.empty((nworld, ntree), dtype=bool),
    grad_scale=wp.empty((nworld, ntree), dtype=float),
    group_count=wp.empty(5, dtype=int),
    world_done=wp.empty(nworld, dtype=bool),
    group_list=wp.empty((4, nworld * ntree), dtype=int),
    nsolving=wp.empty(1, dtype=int),
  )
  d.__dict__["_island_context"] = (key, ctx)
  return ctx


@wp.kernel(module="unique", enable_backward=False, grid_stride=False)
def _reset_counts(
  # Out:
  group_count_out: wp.array[int],
  nsolving_out: wp.array[int],
):
  """Zero the cached context's global counters before _group_layout counts the groups."""
  for i in range(group_count_out.shape[0]):
    group_count_out[i] = 0
  nsolving_out[0] = 0


def _lanes(nworld: int, nrow: int) -> int:
  """Threads per world of the row kernels, which loop over the rows that exist."""
  return min(nrow, max(32, -(-(1 << 19) // nworld // 32) * 32))


def _nblock(per_sm: int = 1) -> int:
  """Persistent blocks for the group kernels (they stride over the groups that exist)."""
  # at least one: a CPU device has no SMs
  return max(1, wp.get_device().sm_count * 32 * per_sm)


def _seed_pool(m: types.Model, d: types.Data, ctx: IslandContext, g: island.GroupBuffers, skip_done: bool = False):
  """H = M (identity on padding) for the block path's groups (skip_done: those still iterating)."""
  wp.launch_tiled(
    _pool_seed,
    dim=_nblock(),
    inputs=[
      m.ntree,
      m.dof_treeid,
      m.tree_dofadr,
      m.tree_dofnum,
      m.M_rownnz,
      m.M_rowadr,
      m.M_colind,
      m.M_hinit_i,
      d.M,
      d.efc.D,
      d.efc.state,
      d.dof_frictionadr,
      g.group_nv,
      _nblock(),
      ctx.group_count,
      ctx.group_list,
      ctx.row2dof,
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
      skip_done,
      ctx.done,
      g.group_dense,
    ],
    outputs=[ctx.H],
    block_dim=_ISLAND_THREADS,
  )


def _evaluate(m: types.Model, d: types.Data, ctx: IslandContext, g: island.GroupBuffers, initial: bool) -> wp.Stream | None:
  """Constraint state at the current point, J'force of sparse islands and the H updates."""
  rows_kernel = _rows(initial, m.opt.cone == types.ConeType.ELLIPTIC)
  rows_lanes = efc_threads_per_world(d.nworld, d.njmax, d.qacc.device, rows_kernel)
  wp.launch(
    rows_kernel,
    dim=(d.nworld, rows_lanes),
    inputs=[
      m.opt.impratio_invsqrt,
      d.nefc,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.frictionloss,
      g.efc_group,
      d.nacon,
      ctx.done,
      ctx.row_jaref,
      rows_lanes,
    ],
    outputs=[
      d.efc.force,
      d.efc.state,
      ctx.row_quad,
      ctx.row_w,
      ctx.quad_changed_ids,
      ctx.quad_changed_count,
      ctx.changed,
      ctx.state_changed,
      ctx.cone_coef,
      ctx.cone_weight,
      ctx.cone_base,
    ],
  )

  elliptic = m.opt.cone == types.ConeType.ELLIPTIC

  def sparse_h():
    # dense rows have no constraint groups; elliptic: a block per constraint group (a CPU block is
    # one thread, as in _update_gradient)
    threads_per_group = 1 if elliptic and wp.get_device().is_cpu else _JTDAJ_THREADS_PER_GROUP
    block_dim = threads_per_group if elliptic else m.block_dim.update_gradient_JTDAJ_sparse
    kernel = _JTDACJ_sparse(m.opt.cone, max(3, int(m.nmaxcondim)) if elliptic else 3, block_dim, _TILE)
    groups_per_world = _jtdaj_groups_per_world(d.nworld, d.njmax, kernel)
    wp.launch(
      kernel,
      dim=(d.nworld, groups_per_world, threads_per_group),
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
        ctx.dof_poolrow,
        ctx.dof_local,
        ctx.row_jaref,
        ctx.world_done,
        groups_per_world,
        g.efc_group,
        ctx.done,
      ],
      outputs=[ctx.H],
      block_dim=block_dim,
    )

  def dense_h():
    wp.launch_tiled(
      _dense_rows_h(g.J.shape[2], elliptic),
      dim=_nblock(),
      inputs=[
        m.ntree,
        m.dof_treeid,
        m.tree_dofadr,
        m.tree_dofnum,
        m.M_rownnz,
        m.M_rowadr,
        m.M_colind,
        m.M_hinit_i,
        d.M,
        d.efc.D,
        d.efc.state,
        ctx.group_count,
        ctx.group_list,
        _nblock(),
        initial,
        g.group_dense,
        g.group_nv,
        ctx.tilebase,
        ctx.vbase,
        ctx.row2dof,
        ctx.done,
        ctx.cone_coef,
        ctx.cone_weight,
        ctx.cone_base,
        g.J,
      ],
      outputs=[ctx.H],
      block_dim=_ISLAND_THREADS,
    )

  def flips():
    slots = _jtdaj_groups_per_world(d.nworld, d.njmax, _h_flips)
    wp.launch(
      _h_flips,
      dim=(d.nworld, slots, _JTDAJ_THREADS_PER_GROUP),
      inputs=[
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        g.efc_group,
        g.group_dense,
        ctx.tilebase,
        ctx.ntile,
        ctx.vbase,
        ctx.dof2row,
        g.J,
        ctx.row_w,
        ctx.quad_changed_ids,
        ctx.quad_changed_count,
        slots,
      ],
      outputs=[ctx.H],
    )

  # H is assembled initially and, for elliptic cones (curvature depends on jaref), at every point;
  # else the rows that flipped update it. It needs only the row states: it runs beside J'force and
  # the gradient (a second stream while capturing), joined before the factorization.
  h_stream = side_stream(d.qacc.device)
  with wp.ScopedStream(h_stream):
    if initial or elliptic:
      _seed_pool(m, d, ctx, g, skip_done=not initial)
      sparse_h()
      dense_h()
    else:
      flips()

  frc_lanes = efc_threads_per_world(d.nworld, d.njmax, d.qacc.device, _frc_rows)
  wp.launch(
    _frc_rows,
    dim=(d.nworld, frc_lanes),
    inputs=[
      d.nefc,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.force,
      g.efc_group,
      g.group_dense,
      ctx.done,
      ctx.niter,
      ctx.state_changed,
      ctx.dof2row,
      frc_lanes,
    ],
    outputs=[ctx.frc],
  )
  return h_stream


def _launch_grad(d: types.Data, ctx: IslandContext, g: island.GroupBuffers, final: bool):
  nrow = ctx.qacc.shape[1]
  # warp slots striding over the groups, several per block (full occupancy; no tile operations)
  nslot = _nblock(4)
  wp.launch(
    _grad(final),
    dim=(nslot, _TILE),
    inputs=[
      g.group_dense.shape[1],
      d.efc.force,
      ctx.group_count,
      ctx.group_list,
      nslot,
      g.group_dense,
      g.J,
      ctx.ntile,
      ctx.vbase,
      ctx.done,
      ctx.niter,
      ctx.state_changed,
      ctx.Ma,
      ctx.qfrc_smooth,
      ctx.grad_scale,
    ],
    outputs=[ctx.frc, ctx.grad.reshape((d.nworld, nrow)), ctx.nsolving],
    block_dim=4 * _TILE,
  )


def _direction(m: types.Model, d: types.Data, ctx: IslandContext, g: island.GroupBuffers, h_stream: wp.Stream | None):
  """Gradient and Newton direction at the current point (as the world solver's _update_gradient)."""
  nrow = ctx.qacc.shape[1]
  _launch_grad(d, ctx, g, final=False)
  if h_stream is not None:
    wp.wait_stream(h_stream)

  def dense_factor(ntile: int):
    wp.launch_tiled(
      _dense_factor(ntile),
      dim=_nblock(),
      inputs=[
        m.ntree,
        g.ngroup,
        g.group_nv,
        ctx.group_count,
        ctx.group_list,
        ntile,
        _nblock(),
        ctx.tilebase,
        ctx.ntile,
        ctx.vbase,
        ctx.done,
        ctx.niter,
        ctx.state_changed,
        ctx.grad.reshape((d.nworld, nrow)),
        ctx.H,
      ],
      outputs=[ctx.search.reshape((d.nworld, nrow)), ctx.search_dot, ctx.newton_decrement, ctx.grad_scale],
      block_dim=_TILE_THREADS,
    )

  def blocked_factor():
    wp.launch_tiled(
      _factor(_blocked_rows(m.nv)),
      dim=_nblock(),
      inputs=[
        m.ntree,
        g.ngroup,
        g.group_nv,
        ctx.group_count,
        ctx.group_list,
        _nblock(),
        ctx.tilebase,
        ctx.ntile,
        ctx.vbase,
        ctx.done,
        ctx.niter,
        ctx.state_changed,
        ctx.grad,
        ctx.H,
      ],
      outputs=[ctx.changed, ctx.U, ctx.search, ctx.search_dot, ctx.newton_decrement, ctx.grad_scale],
      block_dim=m.block_dim.update_gradient_cholesky_blocked,
    )

  def multi_tile():
    dense_factor(2)
    blocked_factor()

  # groups of several tiles only when some group has them (one-tile groups nearly always)
  dense_factor(1)
  if m.opt.graph_conditional:
    wp.capture_if(ctx.group_count[4:5], on_true=multi_tile)
  else:
    multi_tile()


def _launch_done(m: types.Model, d: types.Data, ctx: IslandContext, g: island.GroupBuffers):
  # ctx.nsolving is zeroed by _grad
  nrow = ctx.qacc.shape[1]
  wp.launch(
    _done(int(m.opt.warn_overflow)),
    dim=d.nworld * m.ntree,
    inputs=[
      m.ntree,
      m.opt.tolerance,
      m.opt.iterations,
      ctx.group_count,
      ctx.group_list,
      g.group_nv,
      ctx.vbase,
      ctx.scale,
      ctx.niter,
      ctx.alpha,
      ctx.improvement,
      ctx.grad.reshape((d.nworld, nrow)),
      ctx.grad_scale,
      ctx.newton_decrement,
      ctx.state_changed,
    ],
    outputs=[d.solver_niter, d.overflow, ctx.done, ctx.nsolving, ctx.search_unchanged],
  )


def _iteration(m: types.Model, d: types.Data, ctx: IslandContext, g: island.GroupBuffers):
  """A Newton iteration, as the world solver's: linesearch, constraints, direction, done test."""
  nrow = ctx.qacc.shape[1]
  # warp slots striding over the groups (a warp per possible group would mostly exit at once)
  nslot = min(d.nworld * m.ntree, _nblock(4))
  # J search beside M search, as the world solver's (a second stream while capturing)
  jv_stream = side_stream(d.qacc.device)
  with wp.ScopedStream(jv_stream):
    wp.launch(
      _jv(g.J.shape[2]),
      dim=(d.nworld, d.njmax),
      inputs=[
        d.nefc,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        g.efc_group,
        g.group_dense,
        ctx.vbase,
        ctx.dof2row,
        g.J,
        ctx.search.reshape((d.nworld, nrow)),
        ctx.search_unchanged,
      ],
      outputs=[ctx.row_jv, ctx.quad_changed_count],
    )
  wp.launch(
    _mul_m_search,
    dim=(nslot, _TILE),
    inputs=[
      m.ntree,
      m.M_mulm_rowadr,
      m.M_mulm_col,
      m.M_mulm_madr,
      d.M,
      g.group_nv,
      ctx.group_count,
      ctx.group_list,
      ctx.vbase,
      ctx.row2dof,
      ctx.search.reshape((d.nworld, nrow)),
      ctx.search_unchanged,
      nslot,
    ],
    outputs=[ctx.mv],
  )
  if jv_stream is not None:
    wp.wait_stream(jv_stream)
  # the world solver's linesearch over the groups (_linesearch_iterative_kernel, island mode)
  flat = d.nworld * m.ntree
  problems = IslandProblems()
  problems.ntree = m.ntree
  # persistent blocks: a block per possible group would mostly exit at once
  problems.nblock = min(flat, _nblock())
  problems.count = ctx.group_count[0:1]
  problems.list = ctx.group_list[0]
  problems.rowadr = g.group_iefcadr.reshape(flat)
  problems.dofadr = ctx.vbase.reshape(flat)
  problems.ndof = g.group_nv.reshape(flat)
  problems.frc = ctx.frc
  problems.niter = ctx.niter.reshape(flat)
  problems.grad_scale = ctx.grad_scale.reshape(flat)
  problems.changed = ctx.changed.reshape(flat)
  problems.state_changed = ctx.state_changed.reshape(flat)
  wp.launch_tiled(
    # incremental as the world solver's: a noise-sized step rebuilds the group (_ALPHA_NOISE_EPS)
    _linesearch_iterative_kernel(
      m.opt.ls_iterations, m.opt.cone, False, True, m.opt.cone != types.ConeType.ELLIPTIC, int(m.opt.warn_overflow), False, True
    ),
    dim=problems.nblock,
    inputs=[
      0,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.impratio_invsqrt,
      g.group_ne.reshape(flat),
      g.group_nf.reshape(flat),
      g.group_nefc.reshape(flat),
      ctx.qfrc_smooth,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      d.efc.frictionloss,
      d.njmax,
      d.nacon,
      ctx.scale.reshape(flat),
      None,
      ctx.row_jaref,
      ctx.search.reshape((d.nworld, nrow)),
      ctx.search_dot.reshape(flat),
      ctx.mv,
      ctx.row_jv,
      ctx.row_cone,
      ctx.done.reshape(flat),
      problems,
    ],
    outputs=[
      ctx.qacc,
      ctx.Ma,
      d.overflow,
      ctx.row_jaref,
      ctx.row_jv,
      ctx.row_cone,
      ctx.improvement.reshape(flat),
      ctx.alpha.reshape(flat),
      None,
    ],
    block_dim=m.block_dim.linesearch_iterative,
  )
  h_stream = _evaluate(m, d, ctx, g, initial=False)
  _direction(m, d, ctx, g, h_stream)
  _launch_done(m, d, ctx, g)


def _solve_groups(m: types.Model, d: types.Data, ctx: IslandContext, g: island.GroupBuffers):
  """Solve the awake groups: one-tile groups whole, wider ones on the block path."""
  nrow = ctx.qacc.shape[1]
  init_kernel = _init_rows(g.J.shape[2])
  init_lanes = efc_threads_per_world(d.nworld, d.njmax, d.qacc.device, init_kernel)
  wp.launch(
    init_kernel,
    dim=(d.nworld, init_lanes),
    inputs=[
      d.nefc,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.aref,
      g.efc_group,
      g.group_dense,
      ctx.vbase,
      ctx.dof2row,
      g.J,
      ctx.qacc,
      init_lanes,
    ],
    outputs=[ctx.row_jaref],
  )
  # initial constraint evaluation (H = M + J'DJ over the quadratic rows), direction and done test
  h_stream = _evaluate(m, d, ctx, g, initial=True)
  _direction(m, d, ctx, g, h_stream)
  _launch_done(m, d, ctx, g)

  # each iteration ends with ctx.nsolving = the number of islands still iterating
  if m.opt.graph_conditional:
    wp.capture_while(ctx.nsolving, while_body=_iteration, m=m, d=d, ctx=ctx, g=g)
  else:
    for _ in range(m.opt.iterations):
      _iteration(m, d, ctx, g)

  # J'force where the last iteration skipped it, then scatter back
  _launch_grad(d, ctx, g, final=True)
  wp.launch(
    _scatter_dofs,
    dim=(d.nworld, _lanes(d.nworld, nrow)),
    inputs=[_lanes(d.nworld, nrow), ctx.nrow_used, ctx.row2dof, ctx.qacc, ctx.frc, ctx.Ma],
    outputs=[d.qacc, d.qfrc_constraint, d.efc.Ma],
  )


def solve(m: types.Model, d: types.Data) -> IslandContext:
  """Solve every island with its own Newton iteration and scatter the results back."""
  # make_constraint found the islands and wrote the rows in island order
  ctx = _create_context(m, d)
  wp.launch(_reset_counts, dim=1, outputs=[ctx.group_count, ctx.nsolving])
  # islands are solved in groups (see island.MERGE)
  g = island.group_buffers(m, d)
  d.solver_niter.zero_()
  nrow = ctx.qacc.shape[1]
  wp.launch_tiled(
    _group_layout(m.ntree, _blocked_rows(m.nv)),
    dim=d.nworld,
    inputs=[
      m.tree_dofadr,
      m.tree_dofnum,
      m.M_rownnz,
      m.M_rowadr,
      d.M,
      d.tree_island,
      g.ngroup,
      g.group_nv,
      g.group_dense,
      g.group_asleep,
      g.tree_group,
      g.island_group,
    ],
    outputs=[
      ctx.group_count,
      ctx.group_list,
      ctx.done,
      ctx.niter,
      ctx.alpha,
      ctx.improvement,
      ctx.scale,
      ctx.ntile,
      ctx.tilebase,
      ctx.vbase,
      ctx.nrow_used,
      ctx.row2dof,
      ctx.qacc,
      ctx.qfrc_smooth,
      ctx.frc,
      ctx.Ma,
      ctx.grad,
      ctx.search,
      ctx.mv,
      ctx.search_unchanged,
      ctx.changed,
      ctx.state_changed,
      ctx.grad_scale,
      ctx.quad_changed_count,
      ctx.world_done,
    ],
    block_dim=m.block_dim.island_scan_sizes,
  )
  wp.launch(
    _dofs,
    dim=(d.nworld, m.nv),
    inputs=[
      m.dof_treeid,
      m.M_mulm_rowadr,
      m.M_mulm_col,
      m.M_mulm_madr,
      d.qacc_warmstart,
      d.M,
      d.tree_awake if m.opt.enableflags & types.EnableBit.SLEEP else None,
      d.qfrc_smooth,
      d.qacc_smooth,
      d.map_dof2idof,
      not (m.opt.disableflags & types.DisableBit.WARMSTART),
      g.dof_group,
      g.group_idofadr,
      ctx.tilebase,
      ctx.ntile,
      ctx.vbase,
    ],
    outputs=[
      d.qacc,
      d.qfrc_constraint,
      d.efc.Ma,
      ctx.row2dof,
      ctx.dof2row,
      ctx.dof_poolrow,
      ctx.dof_local,
      ctx.qacc,
      ctx.qfrc_smooth,
      ctx.Ma,
    ],
  )
  # unconditional: entering a conditional body costs more than kernels without groups
  _solve_groups(m, d, ctx, g)

  return ctx
