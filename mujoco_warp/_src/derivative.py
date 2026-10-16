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

from typing import Optional

import warp as wp

from mujoco_warp._src import history
from mujoco_warp._src import math
from mujoco_warp._src import passive
from mujoco_warp._src import smooth
from mujoco_warp._src import support
from mujoco_warp._src import util_misc
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_factorize_solve_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_func
from mujoco_warp._src.passive import build_efm_contact
from mujoco_warp._src.passive import ellipsoid_max_moment
from mujoco_warp._src.passive import geom_semiaxes
from mujoco_warp._src.support import next_act
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import TILE_SIZE_JTDAJ_DENSE
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import EffCholContext
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import vec6
from mujoco_warp._src.types import vec10
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def _eff_block_solve_vertex(
  # Model:
  efm_dofid: wp.array[int],
  # Data in:
  efm_L_in: wp.array2d[float],
  # In:
  rhs_in: wp.array2d[float],
  worldid: int,
  k: int,
  # Out:
  res_out: wp.array2d[float],
):
  """3x3 block solve of covered vertex k."""
  i = efm_dofid[k]
  base = 9 * k
  L00 = efm_L_in[worldid, base + 0]
  L10 = efm_L_in[worldid, base + 3]
  L11 = efm_L_in[worldid, base + 4]
  L20 = efm_L_in[worldid, base + 6]
  L21 = efm_L_in[worldid, base + 7]
  L22 = efm_L_in[worldid, base + 8]

  b0 = rhs_in[worldid, i]
  b1 = rhs_in[worldid, i + 1]
  b2 = rhs_in[worldid, i + 2]

  r0 = b0 / L00
  r1 = (b1 - L10 * r0) / L11
  r2 = (b2 - (L20 * r0 + L21 * r1)) / L22
  x2 = r2 / L22
  x1 = (r1 - L21 * x2) / L11
  x0 = (r0 - L10 * x1 - L20 * x2) / L00

  res_out[worldid, i] = x0
  res_out[worldid, i + 1] = x1
  res_out[worldid, i + 2] = x2


@wp.func
def _eff_chol_find(adr: int, num: int, ind: wp.array[int], target: int) -> int:
  """Position of target in the sorted ind[adr : adr + num], or -1."""
  lo = adr
  hi = adr + num - 1
  while lo <= hi:
    mid = (lo + hi) // 2
    c = ind[mid]
    if c == target:
      return mid
    elif c < target:
      lo = mid + 1
    else:
      hi = mid - 1
  return -1


@wp.func
def _eff_chol_block(colptr: wp.array[int], rowind: wp.array[int], row: int, col: int) -> int:
  """Index of offdiagonal block (row, col), row > col in elimination order, or -1."""
  return _eff_chol_find(colptr[col], colptr[col + 1] - colptr[col], rowind, row)


@wp.func
def _eff_chol_unit(r: int, c: int, v: float) -> wp.mat33:
  out = wp.mat33(0.0)
  out[r, c] = v
  return out


@wp.func
def _eff_chol_col(colind: wp.array[int], adr: int, a: int, dense: bool) -> int:
  if dense:
    return a
  return colind[adr + a]


@wp.func
def _eff_chol_hash_insert(table: wp.array[int], key: int) -> bool:
  """Inserts a positive key into an open-addressing hash set; False if it is already present.

  A full table admits the key, so the pair list overflows and reports instead.
  """
  size = table.shape[0]
  h = (key * 1640531527) & (size - 1)
  for _ in range(size):
    old = wp.atomic_cas(table, h, 0, key)
    if old == 0:
      return True
    if old == key:
      return False
    h = (h + 1) & (size - 1)
  return True


@wp.func
def _eff_chol_row_pairs(
  # Model:
  efm_dofblk: wp.array[int],
  efm_chol_adjptr: wp.array[int],
  efm_chol_adj: wp.array[int],
  # In:
  worldid: int,
  colind: wp.array[int],
  val: wp.array[float],
  adr: int,
  nnz: int,
  dense: bool,
  # Out:
  xcount_out: wp.array[int],
  xpair_out: wp.array2d[wp.vec2i],
  xhash_out: wp.array2d[int],
):
  """Records the distinct vertex pairs a folded term couples that no flex edge joins.

  A per-world open-addressing hash set of the pair keys drops pairs another term already added.
  """
  for a in range(nnz):
    if val[adr + a] == 0.0:
      continue
    k = efm_dofblk[_eff_chol_col(colind, adr, a, dense)]
    if k < 0:
      continue
    for b in range(a + 1, nnz):
      if val[adr + b] == 0.0:
        continue
      q = efm_dofblk[_eff_chol_col(colind, adr, b, dense)]
      if q < 0 or q == k:
        continue
      if _eff_chol_find(efm_chol_adjptr[k], efm_chol_adjptr[k + 1] - efm_chol_adjptr[k], efm_chol_adj, q) >= 0:
        continue
      lo = wp.min(k, q)
      hi = wp.max(k, q)
      if _eff_chol_hash_insert(xhash_out[worldid], lo * EFF_CHOL_KEY + hi + 1):
        x = wp.atomic_add(xcount_out, worldid, 1)
        if x < xpair_out.shape[1]:
          xpair_out[worldid, x] = wp.vec2i(lo, hi)


@wp.func
def _eff_chol_row_fold(
  # Model:
  efm_dofid: wp.array[int],
  efm_dofblk: wp.array[int],
  efm_chol_rank: wp.array[int],
  # In:
  worldid: int,
  scale: float,
  colind: wp.array[int],
  val: wp.array[float],
  adr: int,
  nnz: int,
  dense: bool,
  colptr: wp.array[int],
  rowind: wp.array[int],
  # Out:
  Ld_out: wp.array2d[wp.mat33],
  Lo_out: wp.array2d[wp.mat33],
):
  """Adds scale * v * v' of a folded term to the covered block (its pairs are in the pattern)."""
  for a in range(nnz):
    ia = _eff_chol_col(colind, adr, a, dense)
    k = efm_dofblk[ia]
    va = val[adr + a]
    if k < 0 or va == 0.0:
      continue
    ca = ia - efm_dofid[k]
    rk = efm_chol_rank[k]
    sva = scale * va
    for b in range(nnz):
      ib = _eff_chol_col(colind, adr, b, dense)
      q = efm_dofblk[ib]
      vb = val[adr + b]
      if q < 0 or vb == 0.0:
        continue
      cb = ib - efm_dofid[q]
      rq = efm_chol_rank[q]
      if k == q:
        wp.atomic_add(Ld_out[worldid], rk, _eff_chol_unit(ca, cb, sva * vb))
      elif rk > rq:
        blk = _eff_chol_block(colptr, rowind, rk, rq)
        if blk >= 0:
          wp.atomic_add(Lo_out[worldid], blk, _eff_chol_unit(ca, cb, sva * vb))


@wp.func
def _eff_chol_allowed(aptr: wp.array[int], aend: wp.array[int], adj: wp.array[int], rk: int, rq: int) -> bool:
  """Whether the vertices at ranks rk and rq share a flex edge or are coupled by a folded term."""
  hi = wp.max(rk, rq)
  return _eff_chol_find(aptr[hi], aend[hi] - aptr[hi], adj, wp.min(rk, rq)) >= 0


@wp.func
def _eff_chol_block_scan(local: int) -> int:
  """Exclusive prefix of one value per thread over the block."""
  return wp.untile(wp.tile_scan_exclusive(wp.tile(local)))


@wp.func
def _eff_chol_block_sum(local: int) -> int:
  """Sum of one value per thread over the block, on every thread."""
  return wp.tile_sum(wp.tile(local))[0]


@wp.func
def _eff_chol_row_walk(
  # In:
  i: int,
  k: int,
  prev: int,
  fill: bool,
  parent: wp.array[int],
  first: wp.array[int],
  post: wp.array[int],
  # Out:
  cnt_out: wp.array[int],
  rowind_out: wp.array[int],
):
  """Visits the factor entries (i, t) of row i reached from its neighbor k.

  The walk climbs the elimination tree from k toward i and stops at the first ancestor of prev, the
  previous neighbor in postorder, whose path already visited the rest.
  """
  t = k
  while t >= 0 and t != i:
    if prev >= 0 and first[t] <= post[prev] and post[prev] <= post[t]:
      return
    slot = wp.atomic_add(cnt_out, t, 1)
    if fill:
      rowind_out[slot] = i
    t = parent[t]


@wp.func
def _eff_chol_lower_bound(adr: int, num: int, ind: wp.array[int], target: int) -> int:
  """First position in the sorted ind[adr : adr + num] holding a value >= target."""
  lo = adr
  hi = adr + num
  while lo < hi:
    mid = (lo + hi) // 2
    if ind[mid] < target:
      lo = mid + 1
    else:
      hi = mid
  return lo


@wp.func
def _eff_chol_level_col(entptr: wp.array[int], cstart: int, cend: int, u: int) -> int:
  """Position c in [cstart, cend) of the column holding flattened entry u."""
  lo = cstart
  hi = cend - 1
  while lo < hi:
    mid = (lo + hi + 1) // 2
    if entptr[mid] <= u:
      lo = mid
    else:
      hi = mid - 1
  return lo


@wp.func
def _eff_chol_diag(A: wp.mat33, mindiag: float) -> wp.mat33:
  """Cholesky factor of a 3x3 diagonal block, pivots clamped to mindiag."""
  l00 = wp.sqrt(wp.max(A[0, 0], mindiag))
  l10 = A[1, 0] / l00
  l20 = A[2, 0] / l00
  l11 = wp.sqrt(wp.max(A[1, 1] - l10 * l10, mindiag))
  l21 = (A[2, 1] - l20 * l10) / l11
  l22 = wp.sqrt(wp.max(A[2, 2] - l20 * l20 - l21 * l21, mindiag))
  return wp.mat33(l00, 0.0, 0.0, l10, l11, 0.0, l20, l21, l22)


@wp.func
def _eff_chol_diag_ok(A: wp.mat33, mindiag: float) -> bool:
  """Whether every pivot of a 3x3 diagonal block exceeds mindiag."""
  if not (A[0, 0] > mindiag):
    return False
  l00 = wp.sqrt(A[0, 0])
  l10 = A[1, 0] / l00
  l20 = A[2, 0] / l00
  d1 = A[1, 1] - l10 * l10
  if not (d1 > mindiag):
    return False
  l21 = (A[2, 1] - l20 * l10) / wp.sqrt(d1)
  return A[2, 2] - l20 * l20 - l21 * l21 > mindiag


@wp.func
def _eff_chol_inv_lower(L: wp.mat33) -> wp.mat33:
  """Inverse of a lower-triangular 3x3 block."""
  a = 1.0 / L[0, 0]
  b = 1.0 / L[1, 1]
  c = 1.0 / L[2, 2]
  i10 = -L[1, 0] * a * b
  i21 = -L[2, 1] * b * c
  i20 = -(L[2, 0] * a + L[2, 1] * i10) * c
  return wp.mat33(a, 0.0, 0.0, i10, b, 0.0, i20, i21, c)


@wp.func
def _eff_chol_sinv(A: wp.mat33, mindiag: float) -> wp.mat33:
  """Inverse of a diagonal block through its clamped 3x3 Cholesky factor, (L L')^-1."""
  Li = _eff_chol_inv_lower(_eff_chol_diag(A, mindiag))
  return wp.transpose(Li) * Li


@wp.kernel
def _qderiv_actuator_passive_vel(
  # Model:
  opt_timestep: wp.array[float],
  actuator_dyntype: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_ctrlspec: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_actnum: wp.array[int],
  actuator_dynprm: wp.array2d[vec10],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  actuator_actlimited: wp.array[bool],
  actuator_actrange: wp.array2d[wp.vec2],
  actuator_actearly: wp.array[bool],
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_acc0: wp.array2d[float],
  actuator_lengthrange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  actuator_velocity_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  # Out:
  vel_out: wp.array2d[float],
):
  worldid, actid = wp.tid()

  actuator_gainprm_id = worldid % actuator_gainprm.shape[0]
  actuator_biasprm_id = worldid % actuator_biasprm.shape[0]
  actuator_dynprm_id = worldid % actuator_dynprm.shape[0]
  opt_timestep_id = worldid % opt_timestep.shape[0]

  bias = float(0.0)
  R_dcmotor = float(0.0)
  K_dcmotor = float(0.0)
  te_dcmotor = float(0.0)

  if actuator_gaintype[actid] == GainType.AFFINE:
    gain = actuator_gainprm[actuator_gainprm_id, actid][2]
  elif actuator_gaintype[actid] == GainType.MUSCLE:
    gain = util_misc.muscle_gain_vel_deriv(
      actuator_length_in[worldid, actid],
      actuator_velocity_in[worldid, actid],
      actuator_lengthrange[worldid % actuator_lengthrange.shape[0], actid],
      actuator_acc0[worldid % actuator_acc0.shape[0], actid],
      actuator_gainprm[actuator_gainprm_id, actid],
    )
  elif actuator_gaintype[actid] == GainType.DCMOTOR:
    gain = 0.0
    dynprm = actuator_dynprm[actuator_dynprm_id, actid]
    gainprm = actuator_gainprm[actuator_gainprm_id, actid]
    te = dynprm[0]
    K = gainprm[1]

    # controller velocity derivative dV/dω using nameplate resistance
    dVdw = 0.0
    if (actuator_ctrlspec[actid] & 7) != 0:
      R0 = wp.max(MJ_MINVAL, gainprm[0])
      dVdw = -gainprm[6] * R0 / K + K

    # winding resistance at current temperature
    R = util_misc.dcmotor_resistance(act_in, worldid, actuator_actadr[actid], dynprm, gainprm)

    if te > 0.0:
      # stateful current with actearly: d(K*next_act)/dω
      # includes both back-EMF (-K) and controller (dVdw) through act_dot
      s = 1.0 - wp.exp(-opt_timestep[opt_timestep_id] / te)
      bias += K * (dVdw - K) * s / R
    elif dVdw != 0.0:
      # stateless: controller terms only (back-EMF handled in bias block)
      bias += K * dVdw / R

    # LuGre: force includes -sigma1*z_dot, z_dot = a*z + v
    # d(sigma1*z_dot)/dv = sigma1*(da/dv*z + 1), ignoring higher-order da/dv*z
    sigma1 = dynprm[6]
    if sigma1 > 0.0:
      bias -= sigma1

    R_dcmotor = R
    K_dcmotor = K
    te_dcmotor = te
  else:
    gain = 0.0

  if actuator_biastype[actid] == BiasType.AFFINE:
    bias += actuator_biasprm[actuator_biasprm_id, actid][2]
  elif actuator_biastype[actid] == BiasType.DCMOTOR:
    if R_dcmotor > 0.0:
      if te_dcmotor <= 0.0:
        bias += -K_dcmotor * K_dcmotor / R_dcmotor
    else:
      dynprm = actuator_dynprm[actuator_dynprm_id, actid]
      te = dynprm[0]
      if te <= 0.0:
        gainprm = actuator_gainprm[actuator_gainprm_id, actid]
        K = gainprm[1]
        R = util_misc.dcmotor_resistance(act_in, worldid, actuator_actadr[actid], dynprm, gainprm)
        bias += -K * K / R

  if bias == 0.0 and gain == 0.0:
    vel_out[worldid, actid] = 0.0
    return

  # skip if force is clamped by forcerange
  if actuator_forcelimited[actid]:
    force = actuator_force_in[worldid, actid]
    forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]
    if force <= forcerange[0] or force >= forcerange[1]:
      vel_out[worldid, actid] = 0.0
      return

  vel = float(bias)
  if actuator_dyntype[actid] != DynType.NONE:
    if gain != 0.0:
      act_adr = actuator_actadr[actid] + actuator_actnum[actid] - 1

      # use next activation if actearly is set (matching forward pass)
      if actuator_actearly[actid]:
        act = next_act(
          opt_timestep[opt_timestep_id],
          actuator_dyntype[actid],
          actuator_dynprm[actuator_dynprm_id, actid],
          actuator_actrange[worldid % actuator_actrange.shape[0], actid],
          act_in[worldid, act_adr],
          act_dot_in[worldid, act_adr],
          1.0,
          actuator_actlimited[actid],
        )
      else:
        act = act_in[worldid, act_adr]

      vel += gain * act
  else:
    if gain != 0.0:
      vel += gain * ctrl_in[worldid, actid]

  vel_out[worldid, actid] = vel


@wp.func
def _nonzero_mask(x: float) -> float:
  """Returns 1.0 for non-zero input, 0.0 otherwise."""
  if x != 0.0:
    return 1.0
  return 0.0


@wp.kernel
def _qderiv_actuator_passive_actuation_sparse(
  # Model:
  M_elemid: wp.array2d[int],
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  # In:
  vel_in: wp.array2d[float],
  # Out:
  qDeriv_out: wp.array2d[float],
):
  worldid, actid = wp.tid()

  vel = vel_in[worldid, actid]
  if vel == 0.0:
    return

  rownnz = moment_rownnz_in[worldid, actid]
  rowadr = moment_rowadr_in[worldid, actid]

  for i in range(rownnz):
    rowadri = rowadr + i
    moment_i = actuator_moment_in[worldid, rowadri]
    if moment_i == 0.0:
      continue
    dofi = moment_colind_in[worldid, rowadri]

    for j in range(i + 1):
      rowadrj = rowadr + j
      moment_j = actuator_moment_in[worldid, rowadrj]
      if moment_j == 0.0:
        continue
      dofj = moment_colind_in[worldid, rowadrj]

      elemid = M_elemid[dofi, dofj]
      if elemid >= 0:
        contrib = moment_i * moment_j * vel
        wp.atomic_add(qDeriv_out[worldid], elemid, contrib)


@wp.kernel
def _qderiv_actuator_passive(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  M_elemid: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  M_in: wp.array2d[float],
  # In:
  Mi: wp.array[int],
  Mj: wp.array[int],
  qDeriv_in: wp.array2d[float],
  # Out:
  qDeriv_out: wp.array2d[float],
):
  worldid, elemid = wp.tid()

  dofiid = Mi[elemid]
  dofjid = Mj[elemid]

  # Off-pattern (dofiid, dofjid) pairs have no CSR entry (madr < 0).
  madr = M_elemid[dofiid, dofjid]
  if madr < 0:
    return

  qderiv = qDeriv_in[worldid, madr]

  if not (opt_disableflags & DisableBit.DAMPER) and dofiid == dofjid:
    damping = dof_damping[worldid % dof_damping.shape[0], dofiid]
    dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofiid]
    v = qvel_in[worldid, dofiid]
    qderiv -= util_misc._poly_force_deriv(damping, dpoly, v, 1)

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  qDeriv_out[worldid, madr] = M_in[worldid, madr] - qderiv


# TODO(team): improve performance with tile operations?
@wp.kernel
def _qderiv_tendon_damping(
  # Model:
  ntendon: int,
  opt_timestep: wp.array[float],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  M_elemid: wp.array2d[int],
  # Data in:
  ten_J_in: wp.array2d[float],
  ten_velocity_in: wp.array2d[float],
  # In:
  Mi: wp.array[int],
  Mj: wp.array[int],
  # Out:
  qDeriv_out: wp.array2d[float],
):
  worldid, elemid = wp.tid()
  dofiid = Mi[elemid]
  dofjid = Mj[elemid]

  # Off-pattern (dofiid, dofjid) pairs have no CSR entry (madr < 0).
  madr = M_elemid[dofiid, dofjid]
  if madr < 0:
    return

  qderiv = float(0.0)
  tendon_damping_id = worldid % tendon_damping.shape[0]
  for tenid in range(ntendon):
    damping = tendon_damping[tendon_damping_id, tenid]
    dpoly = tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]
    if damping == 0.0 and dpoly[0] == 0.0 and dpoly[1] == 0.0:
      continue

    rownnz = ten_J_rownnz[tenid]
    rowadr = ten_J_rowadr[tenid]
    Ji = float(0.0)
    Jj = float(0.0)
    for k in range(rownnz):
      if Ji != 0.0 and Jj != 0.0:
        break
      sparseid = rowadr + k
      colind = ten_J_colind[sparseid]
      if colind == dofiid:
        Ji = ten_J_in[worldid, sparseid]
      if colind == dofjid:
        Jj = ten_J_in[worldid, sparseid]

    v = ten_velocity_in[worldid, tenid]
    qderiv -= Ji * Jj * util_misc._poly_force_deriv(damping, dpoly, v, 1)

  qderiv *= opt_timestep[worldid % opt_timestep.shape[0]]

  qDeriv_out[worldid, madr] -= qderiv


@wp.kernel
def deriv_rne_cvel_cdof_dot(
  # Model:
  body_parentid: wp.array[int],
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  body_dofadr: wp.array[int],
  jnt_type: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  body_tree_: wp.array[int],
  # Out:
  Dcvel_out: wp.array3d[wp.spatial_vector],
  Dcdof_dot_out: wp.array3d[wp.spatial_vector],
):
  """Forward pass: compute d(cvel)/d(qvel_k) and d(cdof_dot)/d(qvel_k).

  Mirrors the accumulation order of comvel for each joint type.

  Dcdof_dot for rotation DOFs of free joints (dofid+0..2) is zero because the
  forward pass sets cdof_dot[dofid+0..2] = 0.  The Dcdof_dot array is
  zero-initialized so no explicit write is needed.
  """
  worldid, nodeid, dofid = wp.tid()
  bodyid = body_tree_[nodeid]
  dofadr = body_dofadr[bodyid]
  jntid = body_jntadr[bodyid]
  jntnum = body_jntnum[bodyid]
  pid = body_parentid[bodyid]

  cdof = cdof_in[worldid]

  # Initialize from parent
  cvel_k = Dcvel_out[worldid, pid, dofid]

  if jntnum == 0:
    Dcvel_out[worldid, bodyid, dofid] = cvel_k
    return

  dof_i = dofadr

  for j in range(jntid, jntid + jntnum):
    jnttype = jnt_type[j]

    if jnttype == 0:  # FREE
      # rotation DOFs (dof_i+0..2) contribute to cvel
      if dofid >= dof_i and dofid < dof_i + 3:
        cvel_k += cdof[dofid]

      # cdof_dot for rotation DOFs is zero (set in forward kinematics),
      # so Dcdof_dot for rotation DOFs is zero (from wp.zeros init)

      # derivative of cdof_dot for translation DOFs 3,4,5
      Dcdof_dot_out[worldid, dof_i + 3, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 3])
      Dcdof_dot_out[worldid, dof_i + 4, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 4])
      Dcdof_dot_out[worldid, dof_i + 5, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 5])

      # translation DOFs (dof_i+3..5) contribute to cvel
      if dofid >= dof_i + 3 and dofid < dof_i + 6:
        cvel_k += cdof[dofid]

      dof_i += 6

    elif jnttype == 1:  # BALL
      Dcdof_dot_out[worldid, dof_i + 0, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 0])
      Dcdof_dot_out[worldid, dof_i + 1, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 1])
      Dcdof_dot_out[worldid, dof_i + 2, dofid] = math.motion_cross(cvel_k, cdof[dof_i + 2])

      if dofid >= dof_i and dofid < dof_i + 3:
        cvel_k += cdof[dofid]

      dof_i += 3
    else:  # HINGE or SLIDE
      Dcdof_dot_out[worldid, dof_i, dofid] = math.motion_cross(cvel_k, cdof[dof_i])

      if dofid == dof_i:
        cvel_k += cdof[dof_i]

      dof_i += 1

  Dcvel_out[worldid, bodyid, dofid] = cvel_k


@wp.kernel
def deriv_rne_cacc_cfrcbody_forward(
  # Model:
  body_parentid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  cinert_in: wp.array2d[vec10],
  cvel_in: wp.array2d[wp.spatial_vector],
  cdof_dot_in: wp.array2d[wp.spatial_vector],
  # In:
  body_tree_: wp.array[int],
  Dcvel_in: wp.array3d[wp.spatial_vector],
  Dcdof_dot_in: wp.array3d[wp.spatial_vector],
  # Out:
  Dcacc_out: wp.array3d[wp.spatial_vector],
  Dcfrcbody_out: wp.array3d[wp.spatial_vector],
):
  """Forward pass: compute d(cacc)/d(qvel_k) and d(cfrc_body)/d(qvel_k)."""
  worldid, nodeid, dofid = wp.tid()
  bodyid = body_tree_[nodeid]
  dofadr = body_dofadr[bodyid]
  dofnum = body_dofnum[bodyid]
  pid = body_parentid[bodyid]

  qvel = qvel_in[worldid]

  dcacc = Dcacc_out[worldid, pid, dofid]

  for j in range(dofadr, dofadr + dofnum):
    # Term 1: d(cdof_dot * qvel)/d(qvel_k) when j == dofid
    if j == dofid:
      dcacc += cdof_dot_in[worldid, j]

    # Term 2: cdof_dot depends on cvel which depends on qvel_k
    dcdofdot = Dcdof_dot_in[worldid, j, dofid]
    dcacc += dcdofdot * qvel[j]

  Dcacc_out[worldid, bodyid, dofid] = dcacc

  # d(cfrc_body)/d(qvel_k)
  cinert = cinert_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]
  dcvel = Dcvel_in[worldid, bodyid, dofid]

  # term1 = cinert * d(cacc)/d(qvel_k)
  term1 = math.inert_vec(cinert, dcacc)

  # term2 = d(cvel x* (cinert * cvel))/d(qvel_k)
  cinert_cvel = math.inert_vec(cinert, cvel)
  cinert_dcvel = math.inert_vec(cinert, dcvel)
  term2 = math.motion_cross_force(dcvel, cinert_cvel) + math.motion_cross_force(cvel, cinert_dcvel)

  Dcfrcbody_out[worldid, bodyid, dofid] = term1 + term2


@wp.kernel
def deriv_rne_cfrcbody_backward(
  # Model:
  body_parentid: wp.array[int],
  # In:
  body_tree_: wp.array[int],
  # Out:
  Dcfrcbody_out: wp.array3d[wp.spatial_vector],
):
  """Backward pass: accumulate d(cfrc_body) from children to parents."""
  worldid, nodeid, dofid = wp.tid()
  bodyid = body_tree_[nodeid]
  pid = body_parentid[bodyid]

  # body_tree never contains bodyid=0 (worldbody), so pid >= 0 is always valid.
  # Siblings at the same level may share a parent; atomic_add handles this.
  val = Dcfrcbody_out[worldid, bodyid, dofid]
  wp.atomic_add(Dcfrcbody_out[worldid, pid], dofid, val)


@wp.kernel
def deriv_rne_body2jnt_sparse(
  # Model:
  dof_bodyid: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  timestep: wp.array[float],
  Di: wp.array[int],
  Dj: wp.array[int],
  Dcfrcbody_in: wp.array3d[wp.spatial_vector],
  # Out:
  qDeriv_out: wp.array2d[float],
):
  """Project body-space RNE derivatives into joint-space qDeriv (sparse)."""
  worldid, elemid = wp.tid()
  dt = timestep[worldid % timestep.shape[0]]

  i = Di[elemid]
  j = Dj[elemid]

  body_i = dof_bodyid[i]
  dcfrc = Dcfrcbody_in[worldid, body_i, j]
  term = wp.dot(cdof_in[worldid, i], dcfrc)

  wp.atomic_add(qDeriv_out[worldid], elemid, dt * term)


def deriv_rne_vel(m: Model, d: Data, out: wp.array2d[float]):
  """Compute RNE velocity derivatives and add to the output.

  Implements the analytical derivative of inverse-dynamics Coriolis/centrifugal
  forces with respect to joint velocities.

  Args:
    m: The model (device).
    d: The data (device).
    out: D-structure output array (nworld, nD) to accumulate RNE terms into.
  """
  # TODO(team): consider caching these allocations
  Dcvel = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)
  Dcdof_dot = wp.zeros((d.nworld, m.nv, m.nv), dtype=wp.spatial_vector)
  Dcacc = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)
  Dcfrcbody = wp.zeros((d.nworld, m.nbody, m.nv), dtype=wp.spatial_vector)

  # Forward pass 1: compute Dcvel and Dcdof_dot
  for body_tree in m.body_tree:
    wp.launch(
      deriv_rne_cvel_cdof_dot,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[
        m.body_parentid,
        m.body_jntnum,
        m.body_jntadr,
        m.body_dofadr,
        m.jnt_type,
        d.cdof,
        body_tree,
      ],
      outputs=[Dcvel, Dcdof_dot],
    )

  # Forward pass 2: compute Dcacc and Dcfrcbody
  for body_tree in m.body_tree:
    wp.launch(
      deriv_rne_cacc_cfrcbody_forward,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[
        m.body_parentid,
        m.body_dofnum,
        m.body_dofadr,
        d.qvel,
        d.cinert,
        d.cvel,
        d.cdof_dot,
        body_tree,
        Dcvel,
        Dcdof_dot,
      ],
      outputs=[Dcacc, Dcfrcbody],
    )

  # Backward pass: accumulate Dcfrcbody from children to parents
  for body_tree in reversed(m.body_tree):
    wp.launch(
      deriv_rne_cfrcbody_backward,
      dim=(d.nworld, body_tree.size, m.nv),
      inputs=[m.body_parentid, body_tree],
      outputs=[Dcfrcbody],
    )

  # Project body-space derivatives into joint-space qDeriv (always sparse D-structure)
  wp.launch(
    deriv_rne_body2jnt_sparse,
    dim=(d.nworld, m.qD_fullm_i.size),
    inputs=[m.dof_bodyid, d.cdof, m.opt.timestep, m.qD_fullm_i, m.qD_fullm_j, Dcfrcbody],
    outputs=[out],
  )


@wp.func
def _geom_ellipsoid_fluid_B(
  # In:
  semiaxes: wp.vec3,
  blunt_drag_coef: float,
  slender_drag_coef: float,
  ang_drag_coef: float,
  kutta_lift_coef: float,
  magnus_lift_coef: float,
  virtual_mass: wp.vec3,
  virtual_inertia: wp.vec3,
  ang_vel: wp.vec3,
  lin_vel: wp.vec3,
  density: float,
  viscosity: float,
  flg_dragonly: bool,
) -> tuple[wp.mat33, wp.mat33, wp.mat33, wp.mat33]:
  # B = [[B00, B01], [B10, B11]] where rows are [ang; lin], cols are [ang; lin]
  B00 = wp.mat33(0.0)  # torque wrt ang_vel
  B01 = wp.mat33(0.0)  # torque wrt lin_vel
  B10 = wp.mat33(0.0)  # force wrt ang_vel
  B11 = wp.mat33(0.0)  # force wrt lin_vel

  d_max = wp.max(wp.max(semiaxes[0], semiaxes[1]), semiaxes[2])
  d_min = wp.min(wp.min(semiaxes[0], semiaxes[1]), semiaxes[2])
  d_mid = semiaxes[0] + semiaxes[1] + semiaxes[2] - d_max - d_min
  inv_dmax = wp.where(d_max > MJ_MINVAL, 1.0 / d_max, 0.0)
  s0 = semiaxes[0] * inv_dmax
  s1 = semiaxes[1] * inv_dmax
  s2 = semiaxes[2] * inv_dmax

  norm = wp.length(lin_vel)
  inv_norm = wp.where(norm > MJ_MINVAL, 1.0 / norm, 0.0)
  v = lin_vel * inv_norm
  x = v[0]
  y = v[1]
  z = v[2]
  xx = x * x
  yy = y * y
  zz = z * z

  s12_sq = (s1 * s2) * (s1 * s2)
  s20_sq = (s2 * s0) * (s2 * s0)
  s01_sq = (s0 * s1) * (s0 * s1)

  proj_num = s12_sq * xx + s20_sq * yy + s01_sq * zz
  inv_proj_num = wp.where(proj_num > MJ_MINVAL, 1.0 / proj_num, 0.0)
  a = s12_sq * inv_proj_num
  b = s20_sq * inv_proj_num
  c = s01_sq * inv_proj_num
  aa = a * a
  bb = b * b
  cc = c * c

  proj_denom = aa * xx + bb * yy + cc * zz
  area_scale = d_max * d_max * wp.sqrt(proj_num)

  if not flg_dragonly:
    skew_lin_vel = wp.skew(lin_vel)
    skew_ang_vel = wp.skew(ang_vel)
    if density > 0.0:
      # --- added mass forces ---
      density_vm = density * virtual_mass
      density_vi = density * virtual_inertia
      virtual_lin_mom = wp.cw_mul(density_vm, lin_vel)
      virtual_ang_mom = wp.cw_mul(density_vi, ang_vel)
      skew_virtual_lin_mom = wp.skew(virtual_lin_mom)

      # torque += cross(virtual_ang_mom, ang_vel) -> B00
      B00 += wp.skew(virtual_ang_mom) - skew_ang_vel @ wp.diag(density_vi)

      # torque += cross(virtual_lin_mom, lin_vel) -> B01
      B01 += skew_virtual_lin_mom - skew_lin_vel @ wp.diag(density_vm)

      # force += cross(virtual_lin_mom, ang_vel) -> B10
      B10 += skew_virtual_lin_mom
      # Da = d/d(vlm via lin) = _skew_neg(ang), scaled by density*vm -> B11
      B11 += -skew_ang_vel @ wp.diag(density_vm)

    # --- Magnus force: force += magnus_coef * cross(ang_vel, lin_vel) ---
    volume = wp.static(4.0 / 3.0 * wp.pi) * semiaxes[0] * semiaxes[1] * semiaxes[2]
    magnus_coef = magnus_lift_coef * density * volume
    B10 -= skew_lin_vel * magnus_coef
    B11 += skew_ang_vel * magnus_coef

    # --- Kutta lift (3x3 -> B11) ---
    df_denom = wp.where(
      proj_denom > MJ_MINVAL,
      wp.pi * kutta_lift_coef * density * area_scale * norm / wp.sqrt(proj_denom),
      0.0,
    )

    dfx_coef = yy * (a - b) + zz * (a - c)
    dfy_coef = xx * (b - a) + zz * (b - c)
    dfz_coef = xx * (c - a) + yy * (c - b)
    proj_term = wp.where(proj_denom > MJ_MINVAL, 1.0 / proj_denom, 0.0)
    cos_term = 1.0

    D = wp.skew(wp.vec3(b - c, c - a, a - b)) * 2.0

    df_coef = wp.vec3(dfx_coef, dfy_coef, dfz_coef)
    inner_term = wp.vec3(
      aa * proj_term - a + cos_term,
      bb * proj_term - b + cos_term,
      cc * proj_term - c + cos_term,
    )

    D += wp.outer(df_coef, inner_term)

    V = wp.diag(v)
    D = V @ D @ V - wp.diag(df_coef)

    D *= df_denom
    B11 += D

  # --- viscous drag (3x3 -> B11) ---
  eq_sphere_D = wp.static(2.0 / 3.0) * (semiaxes[0] + semiaxes[1] + semiaxes[2])
  A_max = wp.pi * d_max * d_mid

  A_proj = wp.pi * area_scale * wp.sqrt(proj_denom)

  lin_coef = viscosity * wp.static(3.0 * wp.pi) * eq_sphere_D
  quad_coef = density * (A_proj * blunt_drag_coef + slender_drag_coef * (A_max - A_proj))
  Aproj_coef = density * norm * (blunt_drag_coef - slender_drag_coef)
  dA_coef = wp.where(proj_denom > MJ_MINVAL, wp.pi * area_scale / wp.sqrt(proj_denom), 0.0)

  dAproj_dv = wp.vec3(
    Aproj_coef * dA_coef * a * x * (b * yy * (a - b) + c * zz * (a - c)),
    Aproj_coef * dA_coef * b * y * (a * xx * (b - a) + c * zz * (b - c)),
    Aproj_coef * dA_coef * c * z * (a * xx * (c - a) + b * yy * (c - b)),
  )

  inner = xx + yy + zz
  D = (wp.outer(v, v) + wp.diag(wp.vec3(inner))) * (-quad_coef * norm)
  D -= wp.outer(v, dAproj_dv)
  D -= wp.diag(wp.vec3(lin_coef))

  B11 += D

  # --- viscous torque (3x3 -> B00) ---
  lin_visc_torq_coef = wp.pi * eq_sphere_D * eq_sphere_D * eq_sphere_D
  I_max = wp.static(8.0 / 15.0 * wp.pi) * d_mid * d_max * d_max * d_max * d_max
  II = wp.vec3(
    ellipsoid_max_moment(semiaxes, 0),
    ellipsoid_max_moment(semiaxes, 1),
    ellipsoid_max_moment(semiaxes, 2),
  )

  mom_coef = wp.vec3(
    ang_drag_coef * II[0] + slender_drag_coef * (I_max - II[0]),
    ang_drag_coef * II[1] + slender_drag_coef * (I_max - II[1]),
    ang_drag_coef * II[2] + slender_drag_coef * (I_max - II[2]),
  )

  mom_visc = wp.cw_mul(ang_vel, mom_coef)
  norm_mom = wp.length(mom_visc)
  density_scaled = math.safe_div(density, norm_mom)

  mom_sq = -density_scaled * wp.cw_mul(wp.cw_mul(ang_vel, mom_coef), mom_coef)

  torq_lin_coef = viscosity * lin_visc_torq_coef
  diag_val = wp.dot(ang_vel, mom_sq) - torq_lin_coef

  D = wp.outer(ang_vel, mom_sq) + wp.diag(wp.vec3(diag_val))
  B00 += D

  return B00, B01, B10, B11


@wp.func
def _deriv_ellipsoid_fluid(
  # Model:
  opt_integrator: int,
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_fluid: wp.array2d[float],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  bodyid: int,
  rootid: int,
  geomadr: int,
  geomnum: int,
  cdof_i: wp.spatial_vector,
  cdof_j: wp.spatial_vector,
  wind: wp.vec3,
  density: float,
  viscosity: float,
) -> float:
  """Compute one body's ellipsoid fluid derivative contribution for a DOF pair.

  Returns the scalar J_i^T @ B @ J_j contribution accumulated across geoms.
  """
  flg_dragonly = opt_integrator == IntegratorType.DISCRETE
  is_symmetrize = flg_dragonly or opt_integrator == IntegratorType.IMPLICITFAST

  # Body kinematics
  xipos = xipos_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]
  ang_global = wp.spatial_top(cvel)
  lin_global = wp.spatial_bottom(cvel)
  subtree_root = subtree_com_in[worldid, rootid]
  lin_com = lin_global - wp.cross(xipos - subtree_root, ang_global)

  qderiv_contrib = float(0.0)

  cdof_ang_i = wp.vec3(cdof_i[0], cdof_i[1], cdof_i[2])
  cdof_lin_i = wp.vec3(cdof_i[3], cdof_i[4], cdof_i[5])
  cdof_ang_j = wp.vec3(cdof_j[0], cdof_j[1], cdof_j[2])
  cdof_lin_j = wp.vec3(cdof_j[3], cdof_j[4], cdof_j[5])

  for g in range(geomnum):
    geomid = geomadr + g
    coef = geom_fluid[geomid, 0]
    if coef <= 0.0:
      continue

    size = geom_size[worldid % geom_size.shape[0], geomid]
    semiaxes = geom_semiaxes(size, geom_type[geomid])
    geom_rot = geom_xmat_in[worldid, geomid]
    geom_rotT = wp.transpose(geom_rot)
    geom_pos = geom_xpos_in[worldid, geomid]

    # compute local velocity
    lin_point = lin_com + wp.cross(ang_global, geom_pos - xipos)
    l_ang = geom_rotT @ ang_global
    l_lin = geom_rotT @ lin_point

    if wind[0] != 0.0 or wind[1] != 0.0 or wind[2] != 0.0:
      l_lin -= geom_rotT @ wind

    ang_vel = l_ang
    lin_vel = l_lin

    # read fluid coefficients
    blunt_drag_coef = geom_fluid[geomid, 1]
    slender_drag_coef = geom_fluid[geomid, 2]
    ang_drag_coef = geom_fluid[geomid, 3]
    kutta_lift_coef = geom_fluid[geomid, 4]
    magnus_lift_coef = geom_fluid[geomid, 5]
    virtual_mass = wp.vec3(geom_fluid[geomid, 6], geom_fluid[geomid, 7], geom_fluid[geomid, 8])
    virtual_inertia = wp.vec3(geom_fluid[geomid, 9], geom_fluid[geomid, 10], geom_fluid[geomid, 11])

    B00, B01, B10, B11 = _geom_ellipsoid_fluid_B(
      semiaxes,
      blunt_drag_coef,
      slender_drag_coef,
      ang_drag_coef,
      kutta_lift_coef,
      magnus_lift_coef,
      virtual_mass,
      virtual_inertia,
      ang_vel,
      lin_vel,
      density,
      viscosity,
      flg_dragonly=flg_dragonly,
    )

    # symmetrize for implicitfast or discrete
    if is_symmetrize:
      B00 = 0.5 * (B00 + wp.transpose(B00))
      B11 = 0.5 * (B11 + wp.transpose(B11))
      B01_sym = 0.5 * (B01 + wp.transpose(B10))
      B01 = B01_sym
      B10 = wp.transpose(B01_sym)

    # --- Jacobian transformation: J_i^T @ B @ J_j ---
    offset = geom_pos - subtree_root

    jac_p_i = cdof_lin_i + wp.cross(cdof_ang_i, offset)
    la_i = geom_rotT @ cdof_ang_i
    ll_i = geom_rotT @ jac_p_i

    jac_p_j = cdof_lin_j + wp.cross(cdof_ang_j, offset)
    la_j = geom_rotT @ cdof_ang_j
    ll_j = geom_rotT @ jac_p_j

    # B @ J_j = [B00 @ la_j + B01 @ ll_j; B10 @ la_j + B11 @ ll_j]
    Bj_ang = B00 @ la_j + B01 @ ll_j
    Bj_lin = B10 @ la_j + B11 @ ll_j

    # J_i^T @ (B @ J_j) = la_i . Bj_ang + ll_i . Bj_lin
    qderiv_contrib += wp.dot(la_i, Bj_ang) + wp.dot(ll_i, Bj_lin)

  return qderiv_contrib


@wp.kernel
def _qderiv_ellipsoid_fluid(
  # Model:
  opt_timestep: wp.array[float],
  opt_wind: wp.array[wp.vec3],
  opt_density: wp.array[float],
  opt_viscosity: wp.array[float],
  opt_integrator: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_geomnum: wp.array[int],
  body_geomadr: wp.array[int],
  dof_bodyid: wp.array[int],
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_fluid: wp.array2d[float],
  body_fluid_ellipsoid_adr: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  M_elemid: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  Mi: wp.array[int],
  Mj: wp.array[int],
  # Out:
  qDeriv_out: wp.array2d[float],
):
  """Compute ellipsoid fluid force derivative contribution to qDeriv.

  Parallelized over (world, fluid_body, elem). For each fluid body and DOF
  pair, computes the 6x6 derivative matrix B in local geom frame via
  _deriv_ellipsoid_fluid and accumulates J_i^T @ B @ J_j into qDeriv.
  """
  worldid, fluid_idx, elemid = wp.tid()

  bodyid = body_fluid_ellipsoid_adr[fluid_idx]

  dofiid = Mi[elemid]
  dofjid = Mj[elemid]

  madr = M_elemid[dofiid, dofjid]
  if madr < 0:
    return

  # dofiid is the "deeper" DOF (Mi >= Mj in tree ordering).
  # Any body that has dofiid in its chain also has dofjid.
  bodyid_i = dof_bodyid[dofiid]

  if bodyid_i == 0:
    return

  if body_isdofancestor[bodyid, dofiid] == 0:
    return

  wind = opt_wind[worldid % opt_wind.shape[0]]
  density = opt_density[worldid % opt_density.shape[0]]
  viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  if density <= 0.0 and viscosity <= 0.0:
    return

  cdof_i = cdof_in[worldid, dofiid]
  cdof_j = cdof_in[worldid, dofjid]

  contrib = _deriv_ellipsoid_fluid(
    opt_integrator,
    geom_type,
    geom_size,
    geom_fluid,
    xipos_in,
    geom_xpos_in,
    geom_xmat_in,
    subtree_com_in,
    cvel_in,
    worldid,
    bodyid,
    body_rootid[bodyid],
    body_geomadr[bodyid],
    body_geomnum[bodyid],
    cdof_i,
    cdof_j,
    wind,
    density,
    viscosity,
  )

  contrib *= timestep

  if contrib != 0.0:
    wp.atomic_add(qDeriv_out[worldid], madr, -contrib)


@wp.func
def _deriv_box_fluid(
  # Model:
  opt_integrator: int,
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  # In:
  worldid: int,
  bodyid: int,
  lvel: wp.spatial_vector,
  density: float,
  viscosity: float,
) -> wp.spatial_matrix:
  B = wp.spatial_matrix(0.0)

  mass = body_mass[worldid % body_mass.shape[0], bodyid]
  inertia = body_inertia[worldid % body_inertia.shape[0], bodyid]
  scl = 6.0 / mass

  # Equivalent inertia box
  box = wp.vec3(
    wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl),
    wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl),
    wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl),
  )

  # Viscous force and torque
  if viscosity > 0.0:
    diam = (box[0] + box[1] + box[2]) * wp.static(1.0 / 3.0)

    # Rotational viscosity
    visc_rot = -wp.pi * diam * diam * diam * viscosity
    B[0, 0] += visc_rot
    B[1, 1] += visc_rot
    B[2, 2] += visc_rot

    # Translational viscosity
    visc_lin = wp.static(-3.0 * wp.pi) * diam * viscosity
    B[3, 3] += visc_lin
    B[4, 4] += visc_lin
    B[5, 5] += visc_lin

  # Lift and drag force and torque
  if density > 0.0:
    term0 = box[1] * box[1] * box[1] * box[1] + box[2] * box[2] * box[2] * box[2]
    term1 = box[0] * box[0] * box[0] * box[0] + box[2] * box[2] * box[2] * box[2]
    term2 = box[0] * box[0] * box[0] * box[0] + box[1] * box[1] * box[1] * box[1]

    inv_32 = wp.static(1.0 / 32.0)
    B[0, 0] -= density * box[0] * term0 * wp.abs(lvel[0]) * inv_32
    B[1, 1] -= density * box[1] * term1 * wp.abs(lvel[1]) * inv_32
    B[2, 2] -= density * box[2] * term2 * wp.abs(lvel[2]) * inv_32

    B[3, 3] -= density * box[1] * box[2] * wp.abs(lvel[3])
    B[4, 4] -= density * box[0] * box[2] * wp.abs(lvel[4])
    B[5, 5] -= density * box[0] * box[1] * wp.abs(lvel[5])

  if opt_integrator == IntegratorType.IMPLICITFAST or opt_integrator == IntegratorType.DISCRETE:
    B = 0.5 * (B + wp.transpose(B))

  return B


@wp.func
def _get_jac_column_local(
  # Model:
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  # Data in:
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  point_global: wp.vec3,
  bodyid: int,
  dofid: int,
  worldid: int,
  b_imat: wp.mat33,
) -> wp.spatial_vector:
  offset = point_global - subtree_com_in[worldid, body_rootid[bodyid]]
  cdof_val = cdof_in[worldid, dofid]
  cdof_ang = wp.spatial_top(cdof_val)
  cdof_lin = wp.spatial_bottom(cdof_val)

  jacp = cdof_lin + wp.cross(cdof_ang, offset)
  jacr = cdof_ang

  b_imat_T = wp.transpose(b_imat)
  jacp_loc = b_imat_T @ jacp
  jacr_loc = b_imat_T @ jacr
  return wp.spatial_vector(jacr_loc, jacp_loc)


@wp.kernel
def _qderiv_box_fluid(
  # Model:
  opt_timestep: wp.array[float],
  opt_wind: wp.array[wp.vec3],
  opt_density: wp.array[float],
  opt_viscosity: wp.array[float],
  opt_integrator: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  dof_bodyid: wp.array[int],
  body_fluid_box_adr: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  M_elemid: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  Mi: wp.array[int],
  Mj: wp.array[int],
  # Out:
  qDeriv_out: wp.array2d[float],
):
  worldid, fluid_idx, elemid = wp.tid()

  bodyid = body_fluid_box_adr[fluid_idx]

  dofiid = Mi[elemid]
  dofjid = Mj[elemid]

  madr = M_elemid[dofiid, dofjid]
  if madr < 0:
    return

  bodyid_i = dof_bodyid[dofiid]

  if bodyid_i == 0:
    return

  if body_isdofancestor[bodyid, dofiid] == 0:
    return

  wind = opt_wind[worldid % opt_wind.shape[0]]
  density = opt_density[worldid % opt_density.shape[0]]
  viscosity = opt_viscosity[worldid % opt_viscosity.shape[0]]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  if density <= 0.0 and viscosity <= 0.0:
    return

  # Body velocity and kinematics
  b_ipos = xipos_in[worldid, bodyid]
  b_imat = ximat_in[worldid, bodyid]
  subtree_root = subtree_com_in[worldid, body_rootid[bodyid]]

  vel_subtree = cvel_in[worldid, bodyid]
  v_subtree_ang = wp.vec3(vel_subtree[0], vel_subtree[1], vel_subtree[2])
  v_subtree_lin = wp.vec3(vel_subtree[3], vel_subtree[4], vel_subtree[5])

  lin_com = v_subtree_lin - wp.cross(b_ipos - subtree_root, v_subtree_ang)
  b_imat_T = wp.transpose(b_imat)
  v_local_ang = b_imat_T @ v_subtree_ang
  v_local_lin = b_imat_T @ lin_com
  wind_local = b_imat_T @ wind

  lvel = wp.spatial_vector(v_local_ang, v_local_lin - wind_local)

  B = _deriv_box_fluid(
    opt_integrator,
    body_mass,
    body_inertia,
    worldid,
    bodyid,
    lvel,
    density,
    viscosity,
  )

  # Jacobian transformation: J_i^T @ B @ J_j
  J_i = _get_jac_column_local(
    body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, b_ipos, bodyid, dofiid, worldid, b_imat
  )
  J_j = _get_jac_column_local(
    body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, b_ipos, bodyid, dofjid, worldid, b_imat
  )

  contrib = wp.dot(J_i, B @ J_j) * timestep

  if contrib != 0.0:
    wp.atomic_add(qDeriv_out[worldid], madr, -contrib)


@event_scope
def deriv_smooth_vel(m: Model, d: Data, out: wp.array2d[float]):
  """Analytical derivative of smooth forces w.r.t. velocities.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    out: M - dt * qDeriv (derivatives of smooth forces w.r.t velocities).
  """
  Mi = m.M_fullm_i
  Mj = m.M_fullm_j

  if not (m.opt.disableflags & (DisableBit.ACTUATION | DisableBit.DAMPER)):
    # TODO(team): only clear elements not set by _qderiv_actuator_passive
    out.zero_()
    if m.nactuator > 0 and not (m.opt.disableflags & DisableBit.ACTUATION):
      vel = wp.empty((d.nworld, m.nactuator), dtype=float)
      wp.launch(
        _qderiv_actuator_passive_vel,
        dim=(d.nworld, m.nactuator),
        inputs=[
          m.opt.timestep,
          m.actuator_dyntype,
          m.actuator_gaintype,
          m.actuator_biastype,
          m.actuator_ctrlspec,
          m.actuator_actadr,
          m.actuator_actnum,
          m.actuator_dynprm,
          m.actuator_gainprm,
          m.actuator_biasprm,
          m.actuator_actlimited,
          m.actuator_actrange,
          m.actuator_actearly,
          m.actuator_forcelimited,
          m.actuator_forcerange,
          m.actuator_acc0,
          m.actuator_lengthrange,
          d.act,
          d.ctrl,
          d.act_dot,
          d.actuator_length,
          d.actuator_velocity,
          d.actuator_force,
        ],
        outputs=[vel],
      )
      # out (qDeriv) is in M-structure.
      wp.launch(
        _qderiv_actuator_passive_actuation_sparse,
        dim=(d.nworld, m.nactuator),
        inputs=[
          m.M_elemid,
          d.moment_rownnz,
          d.moment_rowadr,
          d.moment_colind,
          d.actuator_moment,
          vel,
        ],
        outputs=[out],
      )
    wp.launch(
      _qderiv_actuator_passive,
      dim=(d.nworld, Mi.size),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.dof_damping,
        m.dof_dampingpoly,
        m.M_elemid,
        d.qvel,
        d.M,
        Mi,
        Mj,
        out,
      ],
      outputs=[out],
    )
  else:
    # TODO(team): directly utilize M for these settings
    wp.copy(out, d.M)

  if not (m.opt.disableflags & DisableBit.DAMPER):
    wp.launch(
      _qderiv_tendon_damping,
      dim=(d.nworld, Mi.size),
      inputs=[
        m.ntendon,
        m.opt.timestep,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_damping,
        m.tendon_dampingpoly,
        m.M_elemid,
        d.ten_J,
        d.ten_velocity,
        Mi,
        Mj,
      ],
      outputs=[out],
    )
  if m.has_fluid:
    wp.launch(
      _qderiv_ellipsoid_fluid,
      dim=(d.nworld, m.body_fluid_ellipsoid_adr.size, Mi.size),
      inputs=[
        m.opt.timestep,
        m.opt.wind,
        m.opt.density,
        m.opt.viscosity,
        m.opt.integrator,
        m.body_parentid,
        m.body_rootid,
        m.body_geomnum,
        m.body_geomadr,
        m.dof_bodyid,
        m.geom_type,
        m.geom_size,
        m.geom_fluid,
        m.body_fluid_ellipsoid_adr,
        m.body_isdofancestor,
        m.M_elemid,
        d.xipos,
        d.geom_xpos,
        d.geom_xmat,
        d.subtree_com,
        d.cdof,
        d.cvel,
        Mi,
        Mj,
      ],
      outputs=[out],
    )
    wp.launch(
      _qderiv_box_fluid,
      dim=(d.nworld, m.body_fluid_box_adr.size, Mi.size),
      inputs=[
        m.opt.timestep,
        m.opt.wind,
        m.opt.density,
        m.opt.viscosity,
        m.opt.integrator,
        m.body_parentid,
        m.body_rootid,
        m.body_mass,
        m.body_inertia,
        m.dof_bodyid,
        m.body_fluid_box_adr,
        m.body_isdofancestor,
        m.M_elemid,
        d.xipos,
        d.ximat,
        d.subtree_com,
        d.cdof,
        d.cvel,
        Mi,
        Mj,
      ],
      outputs=[out],
    )


@wp.kernel
def _eff_diag_stiff(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  qpos_spring: wp.array2d[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_stiffness: wp.array2d[float],
  jnt_stiffnesspoly: wp.array2d[wp.vec2],
  # Data in:
  qpos_in: wp.array2d[float],
  # Data out:
  efm_diag_out: wp.array2d[float],
):
  worldid, jntid = wp.tid()
  dofid = jnt_dofadr[jntid]
  jnttype = jnt_type[jntid]
  h = opt_timestep[worldid % opt_timestep.shape[0]]

  if opt_disableflags & DisableBit.SPRING:
    return

  stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], jntid]
  spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], jntid]
  if stiffness == 0.0 and spoly[0] == 0.0 and spoly[1] == 0.0:
    return

  qposid = jnt_qposadr[jntid]
  qpos_spring_id = worldid % qpos_spring.shape[0]

  if jnttype == JointType.FREE:
    dif = wp.vec3(
      qpos_in[worldid, qposid + 0] - qpos_spring[qpos_spring_id, qposid + 0],
      qpos_in[worldid, qposid + 1] - qpos_spring[qpos_spring_id, qposid + 1],
      qpos_in[worldid, qposid + 2] - qpos_spring[qpos_spring_id, qposid + 2],
    )
    r = wp.length(dif)
    k = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, r, 0))
    for i in range(3):
      efm_diag_out[worldid, dofid + i] = h * k

    rot = wp.normalize(
      wp.quat(
        qpos_in[worldid, qposid + 3],
        qpos_in[worldid, qposid + 4],
        qpos_in[worldid, qposid + 5],
        qpos_in[worldid, qposid + 6],
      )
    )
    ref = wp.quat(
      qpos_spring[qpos_spring_id, qposid + 3],
      qpos_spring[qpos_spring_id, qposid + 4],
      qpos_spring[qpos_spring_id, qposid + 5],
      qpos_spring[qpos_spring_id, qposid + 6],
    )
    dif_rot = math.quat_sub(rot, ref)
    r_rot = wp.length(dif_rot)
    k_rot = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, r_rot, 0))
    for i in range(3):
      efm_diag_out[worldid, dofid + 3 + i] = h * k_rot

  elif jnttype == JointType.BALL:
    rot = wp.normalize(
      wp.quat(
        qpos_in[worldid, qposid + 0],
        qpos_in[worldid, qposid + 1],
        qpos_in[worldid, qposid + 2],
        qpos_in[worldid, qposid + 3],
      )
    )
    ref = wp.quat(
      qpos_spring[qpos_spring_id, qposid + 0],
      qpos_spring[qpos_spring_id, qposid + 1],
      qpos_spring[qpos_spring_id, qposid + 2],
      qpos_spring[qpos_spring_id, qposid + 3],
    )
    dif_rot = math.quat_sub(rot, ref)
    r_rot = wp.length(dif_rot)
    k_rot = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, r_rot, 0))
    for i in range(3):
      efm_diag_out[worldid, dofid + i] = h * k_rot

  else:
    x = qpos_in[worldid, qposid] - qpos_spring[qpos_spring_id, qposid]
    k = wp.max(0.0, util_misc._poly_force_deriv(stiffness, spoly, x, 0))
    efm_diag_out[worldid, dofid] = h * k


@wp.kernel
def _eff_diag_damp_shift(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  # Data in:
  qvel_in: wp.array2d[float],
  # Data out:
  efm_c_out: wp.array2d[float],
  efm_diag_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  h = opt_timestep[worldid % opt_timestep.shape[0]]

  damp_deriv = float(0.0)
  if not (opt_disableflags & DisableBit.DAMPER):
    damping = dof_damping[worldid % dof_damping.shape[0], dofid]
    dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofid]
    v = qvel_in[worldid, dofid]
    damp_deriv = wp.max(0.0, util_misc._poly_force_deriv(damping, dpoly, v, 1))

  ck = efm_diag_out[worldid, dofid]
  efm_diag_out[worldid, dofid] = h * damp_deriv + h * ck
  efm_c_out[worldid, dofid] = -ck * qvel_in[worldid, dofid]


@wp.kernel
def _eff_tendon_shift(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_stiffness: wp.array2d[float],
  tendon_stiffnesspoly: wp.array2d[wp.vec2],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  tendon_lengthspring: wp.array2d[wp.vec2],
  # Data in:
  ten_J_in: wp.array2d[float],
  ten_length_in: wp.array2d[float],
  ten_velocity_in: wp.array2d[float],
  # Data out:
  efm_c_out: wp.array2d[float],
  efm_ts_out: wp.array2d[float],
):
  worldid, tenid = wp.tid()
  h = opt_timestep[worldid % opt_timestep.shape[0]]

  k = float(0.0)
  if not (opt_disableflags & DisableBit.SPRING):
    length = ten_length_in[worldid, tenid]
    range_spring = tendon_lengthspring[worldid % tendon_lengthspring.shape[0], tenid]
    lower = range_spring[0]
    upper = range_spring[1]
    x = float(0.0)
    if length > upper:
      x = length - upper
    elif length < lower:
      x = length - lower
    if x != 0.0:
      stiff = tendon_stiffness[worldid % tendon_stiffness.shape[0], tenid]
      spoly = tendon_stiffnesspoly[worldid % tendon_stiffnesspoly.shape[0], tenid]
      k = wp.max(0.0, util_misc._poly_force_deriv(stiff, spoly, x, 0))

  b = float(0.0)
  if not (opt_disableflags & DisableBit.DAMPER):
    damping = tendon_damping[worldid % tendon_damping.shape[0], tenid]
    dpoly = tendon_dampingpoly[worldid % tendon_dampingpoly.shape[0], tenid]
    v = ten_velocity_in[worldid, tenid]
    b = wp.max(0.0, util_misc._poly_force_deriv(damping, dpoly, v, 1))

  ts = h * h * k + h * b
  tk = h * k
  efm_ts_out[worldid, tenid] = ts

  if k > 0.0:
    ckv = tk * ten_velocity_in[worldid, tenid]
    rowadr = ten_J_rowadr[tenid]
    rownnz = ten_J_rownnz[tenid]
    for j in range(rownnz):
      sparseid = rowadr + j
      col = ten_J_colind[sparseid]
      val = ten_J_in[worldid, sparseid]
      wp.atomic_sub(efm_c_out, worldid, col, ckv * val)


@wp.kernel
def _eff_actuator_actuation(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  actuator_dyntype: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_ctrladr: wp.array[int],
  actuator_ctrlspec: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_actnum: wp.array[int],
  actuator_dynprm: wp.array2d[vec10],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  actuator_actlimited: wp.array[bool],
  actuator_actrange: wp.array2d[wp.vec2],
  actuator_actearly: wp.array[bool],
  actuator_forcelimited: wp.array[bool],
  actuator_forcerange: wp.array2d[wp.vec2],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  actuator_acc0: wp.array2d[float],
  actuator_lengthrange: wp.array2d[wp.vec2],
  # Data in:
  act_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  act_dot_in: wp.array2d[float],
  actuator_length_in: wp.array2d[float],
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  actuator_velocity_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  # Data out:
  efm_ca_out: wp.array2d[float],
  efm_as_out: wp.array2d[float],
):
  worldid, actid = wp.tid()
  h = opt_timestep[worldid % opt_timestep.shape[0]]

  if opt_disableflags & DisableBit.ACTUATION:
    return

  if actuator_forcelimited[actid]:
    force = actuator_force_in[worldid, actid]
    forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]
    if force <= forcerange[0] or force >= forcerange[1]:
      efm_as_out[worldid, actid] = 0.0
      return

  gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], actid]
  biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], actid]
  dynprm = actuator_dynprm[worldid % actuator_dynprm.shape[0], actid]

  bias_len = float(0.0)
  gain_len = float(0.0)
  bias_vel = float(0.0)
  gain_vel = float(0.0)

  biastype = actuator_biastype[actid]
  if biastype == BiasType.AFFINE:
    bias_len += biasprm[1]
    bias_vel += biasprm[2]
  elif biastype == BiasType.MUSCLE:
    bias_len += util_misc.muscle_bias_len_deriv(
      actuator_length_in[worldid, actid],
      actuator_lengthrange[worldid % actuator_lengthrange.shape[0], actid],
      actuator_acc0[worldid % actuator_acc0.shape[0], actid],
      biasprm,
    )
  elif biastype == BiasType.DCMOTOR:
    if dynprm[0] <= 0.0:
      R = gainprm[0]
      K = gainprm[1]
      slots = util_misc.dcmotor_slots(dynprm, gainprm)
      slot_Ta = slots[2]
      if slot_Ta >= 0:
        T = act_in[worldid, actuator_actadr[actid] + slot_Ta]
        R *= 1.0 + gainprm[2] * (T + dynprm[4] - gainprm[3])
      bias_vel -= math.safe_div(K * K, R)

  gaintype = actuator_gaintype[actid]
  if gaintype == GainType.AFFINE:
    gain_len = gainprm[1]
    gain_vel = gainprm[2]
  elif gaintype == GainType.MUSCLE:
    lenrange = actuator_lengthrange[worldid % actuator_lengthrange.shape[0], actid]
    acc0 = actuator_acc0[worldid % actuator_acc0.shape[0], actid]
    gain_len = util_misc.muscle_gain_len_deriv(
      actuator_length_in[worldid, actid],
      actuator_velocity_in[worldid, actid],
      lenrange,
      acc0,
      gainprm,
    )
    gain_vel = util_misc.muscle_gain_vel_deriv(
      actuator_length_in[worldid, actid],
      actuator_velocity_in[worldid, actid],
      lenrange,
      acc0,
      gainprm,
    )
  elif gaintype == GainType.DCMOTOR:
    te = dynprm[0]
    R0 = gainprm[0]
    K = gainprm[1]
    R = gainprm[0]
    slots = util_misc.dcmotor_slots(dynprm, gainprm)
    slot_Ta = slots[2]
    if slot_Ta >= 0:
      T = act_in[worldid, actuator_actadr[actid] + slot_Ta]
      R *= 1.0 + gainprm[2] * (T + dynprm[4] - gainprm[3])

    dVdl = float(0.0)
    dVdw = float(0.0)
    if (actuator_ctrlspec[actid] & 7) != 0:
      dVdl = math.safe_div(-gainprm[4] * R0, K)
      dVdw = math.safe_div(-gainprm[6] * R0, K) + K

    if te > 0.0:
      s = 1.0 - wp.exp(-h / te)
      bias_len += math.safe_div(K * dVdl * s, R0)
      bias_vel += math.safe_div(K * (dVdw - K) * s, R)
    else:
      if dVdl != 0.0:
        bias_len += math.safe_div(K * dVdl, R0)
      if dVdw != 0.0:
        bias_vel += math.safe_div(K * dVdw, R)

    sigma1 = dynprm[6]
    if sigma1 > 0.0:
      bias_vel -= sigma1

  if gain_len != 0.0 or gain_vel != 0.0:
    u_val = float(0.0)
    if actuator_dyntype[actid] != DynType.NONE:
      act_adr = actuator_actadr[actid] + actuator_actnum[actid] - 1
      if actuator_actearly[actid]:
        u_val = next_act(
          h,
          actuator_dyntype[actid],
          dynprm,
          actuator_actrange[worldid % actuator_actrange.shape[0], actid],
          act_in[worldid, act_adr],
          act_dot_in[worldid, act_adr],
          1.0,
          actuator_actlimited[actid],
        )
      else:
        u_val = act_in[worldid, act_adr]
    else:
      cadr = actuator_ctrladr[actid]
      if cadr >= 0:
        u_val = ctrl_in[worldid, cadr]
        if not (opt_disableflags & DisableBit.CLAMPCTRL) and actuator_ctrllimited[cadr]:
          crange = actuator_ctrlrange[worldid % actuator_ctrlrange.shape[0], cadr]
          u_val = wp.clamp(u_val, crange[0], crange[1])
    bias_len += gain_len * u_val
    bias_vel += gain_vel * u_val

  gp = wp.max(0.0, -bias_len)
  gv = wp.max(0.0, -bias_vel)

  as_val = h * h * gp + h * gv
  ak_val = h * gp
  efm_as_out[worldid, actid] = as_val

  if gp > 0.0:
    ckv = ak_val * actuator_velocity_in[worldid, actid]
    rowadr = moment_rowadr_in[worldid, actid]
    rownnz = moment_rownnz_in[worldid, actid]
    for j in range(rownnz):
      sparseid = rowadr + j
      col = moment_colind_in[worldid, sparseid]
      val = actuator_moment_in[worldid, sparseid]
      wp.atomic_sub(efm_ca_out, worldid, col, ckv * val)


@wp.kernel
def _eff_init_sdiag(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # Data in:
  efm_diag_in: wp.array2d[float],
  # Data out:
  qH_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  diag_elem = M_rowadr[dofid] + M_rownnz[dofid] - 1
  qH_out[worldid, diag_elem] += efm_diag_in[worldid, dofid]


@wp.kernel
def _eff_add_tendon_qH_diag(
  # Model:
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # Data in:
  ten_J_in: wp.array2d[float],
  efm_ts_in: wp.array2d[float],
  # Data out:
  qH_out: wp.array2d[float],
):
  worldid, t = wp.tid()
  ts = efm_ts_in[worldid, t]
  if ts == 0.0:
    return
  rowadr = ten_J_rowadr[t]
  rownnz = ten_J_rownnz[t]
  for j in range(rownnz):
    sparseid = rowadr + j
    dof = ten_J_colind[sparseid]
    val = ten_J_in[worldid, sparseid]
    term = ts * val * val
    diag_elem = M_rowadr[dof] + M_rownnz[dof] - 1
    wp.atomic_add(qH_out, worldid, diag_elem, term)


@wp.kernel
def _eff_add_actuator_qH_diag(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  efm_as_in: wp.array2d[float],
  # Data out:
  qH_out: wp.array2d[float],
):
  worldid, a = wp.tid()
  as_val = efm_as_in[worldid, a]
  if as_val == 0.0:
    return
  rowadr = moment_rowadr_in[worldid, a]
  rownnz = moment_rownnz_in[worldid, a]
  for j in range(rownnz):
    sparseid = rowadr + j
    dof = moment_colind_in[worldid, sparseid]
    val = actuator_moment_in[worldid, sparseid]
    term = as_val * val * val
    diag_elem = M_rowadr[dof] + M_rownnz[dof] - 1
    wp.atomic_add(qH_out, worldid, diag_elem, term)


@wp.kernel
def _eff_add_fluid_qH(
  # Data in:
  efm_fluid_in: wp.array2d[float],
  # Data out:
  qH_out: wp.array2d[float],
):
  worldid, madr = wp.tid()
  qH_out[worldid, madr] += efm_fluid_in[worldid, madr]


@cache_kernel
def _eff_mul_fluid(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    M_mulm_rowadr: wp.array[int],
    M_mulm_col: wp.array[int],
    M_mulm_madr: wp.array[int],
    # Data in:
    efm_fluid_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    skip_in: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, dofid = wp.tid()
    if wp.static(check_skip):
      if skip_in[worldid]:
        return

    acc = float(0.0)
    start = M_mulm_rowadr[dofid]
    end = M_mulm_rowadr[dofid + 1]
    for k in range(start, end):
      col = M_mulm_col[k]
      madr = M_mulm_madr[k]
      acc += efm_fluid_in[worldid, madr] * vec_in[worldid, col]

    res_out[worldid, dofid] += acc

  return kernel


@cache_kernel
def _eff_mul_diag(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    efm_diag_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, dofid = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return
    res_out[worldid, dofid] += efm_diag_in[worldid, dofid] * vec_in[worldid, dofid]

  return kernel


# threads per row of the effective-metric stiffness product
EFF_MUL_SPLIT = 8


@cache_kernel
def _eff_mul_diag_and_csr(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    efm_K_rownnz: wp.array[int],
    efm_K_rowadr: wp.array[int],
    efm_K_colind: wp.array[int],
    # Data in:
    efm_diag_in: wp.array2d[float],
    efm_K_val_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    skip_in: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    """Row dofid of diag * vec + K * vec, its entries split over EFF_MUL_SPLIT threads."""
    worldid, dofid, part = wp.tid()
    if wp.static(check_skip):
      if skip_in[worldid]:
        return
    s = float(0.0)
    if part == 0:
      s = efm_diag_in[worldid, dofid] * vec_in[worldid, dofid]
    rowadr = efm_K_rowadr[dofid]
    for adr in range(rowadr + part, rowadr + efm_K_rownnz[dofid], EFF_MUL_SPLIT):
      s += efm_K_val_in[worldid, adr] * vec_in[worldid, efm_K_colind[adr]]
    wp.atomic_add(res_out, worldid, dofid, s)

  return kernel


@cache_kernel
def _eff_mul_add_tendon(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    ten_J_rownnz: wp.array[int],
    ten_J_rowadr: wp.array[int],
    ten_J_colind: wp.array[int],
    # Data in:
    ten_J_in: wp.array2d[float],
    efm_ts_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, t = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return

    s = efm_ts_in[worldid, t]
    if s == 0.0:
      return

    rowadr = ten_J_rowadr[t]
    rownnz = ten_J_rownnz[t]
    dot = float(0.0)
    for j in range(rownnz):
      sparseid = rowadr + j
      dot += ten_J_in[worldid, sparseid] * vec_in[worldid, ten_J_colind[sparseid]]
    dot *= s
    for j in range(rownnz):
      sparseid = rowadr + j
      wp.atomic_add(res_out, worldid, ten_J_colind[sparseid], dot * ten_J_in[worldid, sparseid])

  return kernel


@cache_kernel
def _eff_mul_add_actuator(check_skip: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    moment_rownnz_in: wp.array2d[int],
    moment_rowadr_in: wp.array2d[int],
    moment_colind_in: wp.array2d[int],
    actuator_moment_in: wp.array2d[float],
    efm_as_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, a = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return

    s = efm_as_in[worldid, a]
    if s == 0.0:
      return

    rowadr = moment_rowadr_in[worldid, a]
    rownnz = moment_rownnz_in[worldid, a]
    dot = float(0.0)
    for j in range(rownnz):
      sparseid = rowadr + j
      dot += actuator_moment_in[worldid, sparseid] * vec_in[worldid, moment_colind_in[worldid, sparseid]]
    dot *= s
    for j in range(rownnz):
      sparseid = rowadr + j
      wp.atomic_add(res_out, worldid, moment_colind_in[worldid, sparseid], dot * actuator_moment_in[worldid, sparseid])

  return kernel


@wp.kernel
def _eff_rhs(
  # Model:
  opt_enableflags: int,
  dof_treeid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  qfrc_smooth_in: wp.array2d[float],
  efm_c_in: wp.array2d[float],
  efm_ca_in: wp.array2d[float],
  # Out:
  rhs_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if (opt_enableflags & EnableBit.SLEEP) and tree_awake_in[worldid, dof_treeid[dofid]] == 0:
    rhs_out[worldid, dofid] = 0.0
    return
  rhs_out[worldid, dofid] = qfrc_smooth_in[worldid, dofid] + efm_c_in[worldid, dofid] + efm_ca_in[worldid, dofid]


@wp.func
def _find_csr_col(rowadr: int, rownnz: int, target_col: int, colind: wp.array[int]) -> int:
  lo = rowadr
  hi = rowadr + rownnz - 1
  while lo <= hi:
    mid = (lo + hi) // 2
    c = colind[mid]
    if c == target_col:
      return mid
    elif c < target_col:
      lo = mid + 1
    else:
      hi = mid - 1
  return -1


@wp.kernel
def _eff_flex_stretch_stiff_vert(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_stiffness: wp.array[float],
  flex_damping: wp.array[float],
  flex_rigid: wp.array[bool],
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  flex_vertflexid: wp.array[int],
  flex_simple: wp.array[bool],
  # Data in:
  xmat_in: wp.array2d[wp.mat33],
  flexvert_hessian_in: wp.array2d[vec6],
  # Data out:
  efm_K_val_out: wp.array2d[float],
):
  worldid, vertid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  f = flex_vertflexid[vertid]
  if f < 0 or flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] < 2 or not flex_simple[f]:
    return

  stiffness_adr = flex_stiffnessadr[f]
  if stiffness_adr < 0 or flex_stiffness[stiffness_adr] == 0.0:
    return

  s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep
  s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep
  scale = s1 + s2 * flex_damping[f]
  if scale == 0.0:
    return

  bi = body_weldid[flex_vertbodyid[vertid]]
  if body_dofnum[bi] != 3:
    return
  dof_i = body_dofadr[bi]
  if efm_K_rownnz[dof_i] == 0:
    return

  h = flexvert_hessian_in[worldid, vertid]
  block = wp.mat33(h[0], h[1], h[2], h[1], h[3], h[4], h[2], h[4], h[5])
  R_bi = xmat_in[worldid, bi]
  blkd = scale * (wp.transpose(R_bi) * block * R_bi)

  for r in range(3):
    row = dof_i + r
    idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_i, efm_K_colind)
    if idx >= 0:
      for c in range(3):
        wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])


@wp.kernel
def _eff_flex_stretch_stiff_edge(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_edge: wp.array[wp.vec2i],
  flex_stiffness: wp.array[float],
  flex_damping: wp.array[float],
  flex_rigid: wp.array[bool],
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  flex_edgeflexid: wp.array[int],
  flex_simple: wp.array[bool],
  # Data in:
  xmat_in: wp.array2d[wp.mat33],
  flexedge_hessian_in: wp.array2d[wp.mat33],
  # Data out:
  efm_K_val_out: wp.array2d[float],
):
  worldid, edgeid, side = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  f = flex_edgeflexid[edgeid]
  if f < 0 or flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] < 2 or not flex_simple[f]:
    return

  stiffness_adr = flex_stiffnessadr[f]
  if stiffness_adr < 0 or flex_stiffness[stiffness_adr] == 0.0:
    return

  s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep
  s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep
  scale = s1 + s2 * flex_damping[f]
  if scale == 0.0:
    return

  va = flex_vertadr[f]
  edge = flex_edge[edgeid]
  vi = edge[side]
  vj = edge[1 - side]

  bi = body_weldid[flex_vertbodyid[va + vi]]
  bj = body_weldid[flex_vertbodyid[va + vj]]
  if body_dofnum[bi] != 3 or body_dofnum[bj] != 3:
    return

  dof_i = body_dofadr[bi]
  dof_j = body_dofadr[bj]
  if efm_K_rownnz[dof_i] == 0:
    return

  h = flexedge_hessian_in[worldid, edgeid]
  block = wp.transpose(h) if side == 1 else h
  blkd = scale * (wp.transpose(xmat_in[worldid, bi]) * block * xmat_in[worldid, bj])

  for r in range(3):
    row = dof_i + r
    idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)
    if idx >= 0:
      for c in range(3):
        wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])


@wp.kernel
def _eff_flex_bend_stiff(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_edgeadr: wp.array[int],
  flex_edgenum: wp.array[int],
  flex_bendingadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_edge: wp.array[wp.vec2i],
  flex_edgeflap: wp.array[wp.vec2i],
  flex_bending: wp.array[float],
  flex_damping: wp.array[float],
  flex_rigid: wp.array[bool],
  flex_centered: wp.array[bool],
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  flex_edgeflexid: wp.array[int],
  flex_simple: wp.array[bool],
  # Data in:
  xquat_in: wp.array2d[wp.quat],
  xmat_in: wp.array2d[wp.mat33],
  # Data out:
  efm_K_val_out: wp.array2d[float],
):
  worldid, edgeid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  f = flex_edgeflexid[edgeid]

  if f < 0:
    return

  if flex_interp[f] != 0 or flex_rigid[f] or flex_dim[f] != 2 or not flex_simple[f]:
    return

  bendingadr = flex_bendingadr[f]
  if bendingadr < 0:
    return

  s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep
  s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep
  scale = s1 + s2 * flex_damping[f]
  if scale == 0.0:
    return

  local_edgeid = edgeid - flex_edgeadr[f]
  vbase = flex_vertadr[f]

  edge = flex_edge[edgeid]
  flap = flex_edgeflap[edgeid]

  if flap[1] == -1:
    return

  verts = wp.vec4i(edge[0], edge[1], flap[0], flap[1])
  bend_offset = bendingadr + 17 * local_edgeid

  # keep only convex stencils so the metric stays SPD
  if not passive.flex_bend_convex(flex_bending, bend_offset):
    return

  dofs = wp.vec4i(-1, -1, -1, -1)
  bodies = wp.vec4i(-1, -1, -1, -1)
  for k in range(4):
    bk = body_weldid[flex_vertbodyid[vbase + verts[k]]]
    bodies[k] = bk
    if body_dofnum[bk] == 3:
      dofs[k] = body_dofadr[bk]

  for i in range(4):
    dof_i = dofs[i]
    if dof_i < 0 or efm_K_rownnz[dof_i] == 0:
      continue
    bi = bodies[i]
    R_bi_T = wp.transpose(xmat_in[worldid, bi])
    qi = xquat_in[worldid, bi]
    for j in range(4):
      dof_j = dofs[j]
      if dof_j < 0:
        continue
      q = scale * flex_bending[bend_offset + 4 * i + j]
      if q == 0.0:
        continue
      bj = bodies[j]
      if bi == bj or qi == xquat_in[worldid, bj]:
        for r in range(3):
          row = dof_i + r
          idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)
          if idx >= 0:
            wp.atomic_add(efm_K_val_out, worldid, idx + r, q)
      else:
        blkd = q * (R_bi_T * xmat_in[worldid, bj])
        for r in range(3):
          row = dof_i + r
          idx = _find_csr_col(efm_K_rowadr[row], efm_K_rownnz[row], dof_j, efm_K_colind)
          if idx >= 0:
            for c in range(3):
              wp.atomic_add(efm_K_val_out, worldid, idx + c, blkd[r, c])


@wp.kernel
def _eff_flex_interp_stiff(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_nodeadr: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_stiffness: wp.array[float],
  flex_damping: wp.array[float],
  flex_centered: wp.array[bool],
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  flex_cell_map: wp.array[wp.vec4i],
  # Data in:
  xmat_in: wp.array2d[wp.mat33],
  flexnode_xpos_in: wp.array2d[wp.vec3],
  # Data out:
  efm_K_val_out: wp.array2d[float],
):
  worldid, cellid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  mapping = flex_cell_map[cellid]
  f = mapping[0]
  ci = mapping[1]
  cj = mapping[2]
  ck = mapping[3]

  order = flex_interp[f]
  if order <= 0:
    return

  stiffness_adr_base = flex_stiffnessadr[f]
  if stiffness_adr_base < 0:
    return

  s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else timestep * timestep
  s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else timestep
  iscale = -(s1 + s2 * flex_damping[f])
  if iscale == 0.0:
    return

  npc = (order + 1) * (order + 1) * (order + 1)
  ndof_cell = 3 * npc
  cellnum = flex_cellnum[f]
  cy = cellnum[1]
  cz = cellnum[2]
  nstart = flex_nodeadr[f]
  ny_g = cy * order + 1
  nz_g = cz * order + 1

  cell_idx = ci * cy * cz + cj * cz + ck
  k_base = stiffness_adr_base + cell_idx * ndof_cell * ndof_cell
  if flex_stiffness[k_base] == 0.0:
    return

  cell_quat = support.compute_interp_cell_quat(flexnode_xpos_in, order, ci, cj, ck, cy, cz, ny_g, nz_g, nstart, worldid)
  R = wp.quat_to_matrix(cell_quat)
  RT = wp.transpose(R)

  for li_i in range(order + 1):
    for lj_i in range(order + 1):
      for lk_i in range(order + 1):
        idx_i = (li_i * (order + 1) + lj_i) * (order + 1) + lk_i
        gidx_i = (ci * order + li_i) * ny_g * nz_g + (cj * order + lj_i) * nz_g + (ck * order + lk_i)
        bi = flex_nodebodyid[nstart + gidx_i]
        if body_dofnum[bi] != 3:
          continue
        dof_i = body_dofadr[bi]
        T_i_T = wp.transpose(xmat_in[worldid, bi]) * R

        for li_j in range(order + 1):
          for lj_j in range(order + 1):
            for lk_j in range(order + 1):
              idx_j = (li_j * (order + 1) + lj_j) * (order + 1) + lk_j
              gidx_j = (ci * order + li_j) * ny_g * nz_g + (cj * order + lj_j) * nz_g + (ck * order + lk_j)
              bj = flex_nodebodyid[nstart + gidx_j]
              if body_dofnum[bj] != 3:
                continue
              dof_j = body_dofadr[bj]
              T_j = RT * xmat_in[worldid, bj]

              K_ij = wp.mat33(0.0)
              for r in range(3):
                for c in range(3):
                  K_ij[r, c] = flex_stiffness[k_base + (3 * idx_i + r) * ndof_cell + (3 * idx_j + c)]

              blkd = iscale * (T_i_T * K_ij * T_j)
              for r in range(3):
                row = dof_i + r
                rowadr = efm_K_rowadr[row]
                rownnz = efm_K_rownnz[row]
                for c in range(3):
                  col = dof_j + c
                  idx = _find_csr_col(rowadr, rownnz, col, efm_K_colind)
                  if idx >= 0:
                    wp.atomic_add(efm_K_val_out, worldid, idx, blkd[r, c])


@wp.func
def _extract_3x3_diag_block(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  efm_K_val_in: wp.array2d[float],
  # In:
  i: int,
  worldid: int,
) -> wp.mat33:
  B = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for r in range(3):
    row = i + r
    m_adr = M_rowadr[row]
    m_nnz = M_rownnz[row]
    for a in range(m_nnz):
      c = M_colind[m_adr + a]
      if c >= i and c < i + 3:
        B[r, c - i] += M_in[worldid, m_adr + a]

    k_adr = efm_K_rowadr[row]
    k_nnz = efm_K_rownnz[row]
    for a in range(k_nnz):
      c = efm_K_colind[k_adr + a]
      if c >= i and c < i + 3:
        B[r, c - i] += efm_K_val_in[worldid, k_adr + a]
  return B


@wp.func
def _write_cholesky33(
  # In:
  worldid: int,
  base: int,
  B00: float,
  B10: float,
  B11: float,
  B20: float,
  B21: float,
  B22: float,
  # Out:
  out: wp.array2d[float],
):
  L00 = wp.sqrt(wp.max(MJ_MINVAL, B00))
  L10 = B10 / L00
  L20 = B20 / L00
  L11 = wp.sqrt(wp.max(MJ_MINVAL, B11 - L10 * L10))
  L21 = (B21 - L20 * L10) / L11
  L22 = wp.sqrt(wp.max(MJ_MINVAL, B22 - (L20 * L20 + L21 * L21)))

  out[worldid, base + 0] = L00
  out[worldid, base + 1] = 0.0
  out[worldid, base + 2] = 0.0
  out[worldid, base + 3] = L10
  out[worldid, base + 4] = L11
  out[worldid, base + 5] = 0.0
  out[worldid, base + 6] = L20
  out[worldid, base + 7] = L21
  out[worldid, base + 8] = L22


@wp.kernel
def _eff_factor_blocks(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  efm_dofid: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  efm_K_val_in: wp.array2d[float],
  # Data out:
  efm_L_out: wp.array2d[float],
):
  worldid, k = wp.tid()
  i = efm_dofid[k]
  B = _extract_3x3_diag_block(
    M_rownnz, M_rowadr, M_colind, efm_K_rownnz, efm_K_rowadr, efm_K_colind, M_in, efm_K_val_in, i, worldid
  )
  _write_cholesky33(worldid, 9 * k, B[0, 0], B[1, 0], B[1, 1], B[2, 0], B[2, 1], B[2, 2], efm_L_out)


@wp.kernel
def eff_block_solve(
  # Model:
  efm_dofid: wp.array[int],
  # Data in:
  efm_L_in: wp.array2d[float],
  # In:
  rhs_in: wp.array2d[float],
  # Out:
  res_out: wp.array2d[float],
):
  worldid, k = wp.tid()
  _eff_block_solve_vertex(efm_dofid, efm_L_in, rhs_in, worldid, k, res_out)


@wp.kernel
def _eff_copy_backbone_rhs(
  # Model:
  nv: int,
  efm_dofblk: wp.array[int],
  # In:
  vec_in: wp.array2d[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if dofid < nv:
    vec_out[worldid, dofid] = 0.0 if efm_dofblk[dofid] >= 0 else vec_in[worldid, dofid]


@wp.kernel
def _eff_prec_efm0_solve(
  # Model:
  nefm0dof: int,
  efm0_dofid: wp.array[int],
  efm0_L_rownnz: wp.array[int],
  efm0_L_rowadr: wp.array[int],
  efm0_L_colind: wp.array[int],
  efm0_L: wp.array[float],
  # In:
  vec_in: wp.array2d[float],
  # Out:
  res_out: wp.array2d[float],
):
  worldid = wp.tid()
  for r in range(nefm0dof):
    dof_r = efm0_dofid[r]
    res_out[worldid, dof_r] = vec_in[worldid, dof_r]

  i = nefm0dof - 1
  while i >= 0:
    adr = efm0_L_rowadr[i]
    nnz = efm0_L_rownnz[i]
    diag_idx = adr + nnz - 1
    dof_i = efm0_dofid[i]
    val_i = res_out[worldid, dof_i] / efm0_L[diag_idx]
    res_out[worldid, dof_i] = val_i
    if val_i != 0.0:
      for j in range(nnz - 1):
        c = efm0_L_colind[adr + j]
        res_out[worldid, efm0_dofid[c]] -= efm0_L[adr + j] * val_i
    i -= 1

  for i in range(nefm0dof):
    adr = efm0_L_rowadr[i]
    nnz = efm0_L_rownnz[i]
    diag_idx = adr + nnz - 1
    dof_i = efm0_dofid[i]
    acc = res_out[worldid, dof_i]
    for j in range(nnz - 1):
      c = efm0_L_colind[adr + j]
      acc -= efm0_L[adr + j] * res_out[worldid, efm0_dofid[c]]
    res_out[worldid, dof_i] = acc / efm0_L[diag_idx]


@wp.kernel
def zero_sleeping_dofs(
  # Model:
  dof_treeid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if tree_awake_in[worldid, dof_treeid[dofid]] == 0:
    vec_out[worldid, dofid] = 0.0


@cache_kernel
def _eff_flex_interp_mul(check_skip: bool, is_shift: bool):
  """Applies corotated interpolated flex stiffness/damping operator to a DOF vector."""

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_simple: wp.array[int],
    flex_interp: wp.array[int],
    flex_cellnum: wp.array[wp.vec3i],
    flex_nodeadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_nodebodyid: wp.array[int],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_cell_map: wp.array[wp.vec4i],
    # Data in:
    xmat_in: wp.array2d[wp.mat33],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    flexnode_xpos_in: wp.array2d[wp.vec3],
    # In:
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, cellid = wp.tid()
    if wp.static(check_skip):
      if skip[worldid]:
        return
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    if timestep == 0.0:
      return

    mapping = flex_cell_map[cellid]
    f = mapping[0]
    ci = mapping[1]
    cj = mapping[2]
    ck = mapping[3]

    order = flex_interp[f]
    if order <= 0:
      return

    stiffness_adr_base = flex_stiffnessadr[f]
    if stiffness_adr_base < 0:
      return

    if wp.static(is_shift):
      if opt_disableflags & DisableBit.SPRING:
        return
      scale = timestep
    else:
      s1 = 0.0 if (opt_disableflags & DisableBit.SPRING) else -timestep * timestep
      s2 = 0.0 if (opt_disableflags & DisableBit.DAMPER) else -timestep
      scale = s1 + s2 * flex_damping[f]
      if scale == 0.0:
        return

    npc = (order + 1) * (order + 1) * (order + 1)
    ndof_cell = 3 * npc
    cellnum = flex_cellnum[f]
    cy = cellnum[1]
    cz = cellnum[2]
    nstart = flex_nodeadr[f]
    ny_g = cy * order + 1
    nz_g = cz * order + 1

    cell_idx = ci * cy * cz + cj * cz + ck
    k_base = stiffness_adr_base + cell_idx * ndof_cell * ndof_cell
    if flex_stiffness[k_base] == 0.0:
      return

    cell_quat = support.compute_interp_cell_quat(flexnode_xpos_in, order, ci, cj, ck, cy, cz, ny_g, nz_g, nstart, worldid)
    cell_quat_inv = wp.quat(-cell_quat[0], -cell_quat[1], -cell_quat[2], cell_quat[3])

    for li_i in range(order + 1):
      for lj_i in range(order + 1):
        for lk_i in range(order + 1):
          idx_i = (li_i * (order + 1) + lj_i) * (order + 1) + lk_i
          gidx_i = (ci * order + li_i) * ny_g * nz_g + (cj * order + lj_i) * nz_g + (ck * order + lk_i)
          bi = flex_nodebodyid[nstart + gidx_i]
          if bi == 0:
            continue

          frc_corot = wp.vec3(0.0, 0.0, 0.0)
          for li_j in range(order + 1):
            for lj_j in range(order + 1):
              for lk_j in range(order + 1):
                idx_j = (li_j * (order + 1) + lj_j) * (order + 1) + lk_j
                gidx_j = (ci * order + li_j) * ny_g * nz_g + (cj * order + lj_j) * nz_g + (ck * order + lk_j)
                bj = flex_nodebodyid[nstart + gidx_j]
                if bj == 0:
                  continue

                v_world_j = wp.vec3(0.0, 0.0, 0.0)
                if body_simple[bj] == 2 and body_dofnum[bj] == 3:
                  dof_j = body_dofadr[bj]
                  v_loc_j = wp.vec3(vec_in[worldid, dof_j], vec_in[worldid, dof_j + 1], vec_in[worldid, dof_j + 2])
                  v_world_j = xmat_in[worldid, bj] * v_loc_j
                else:
                  pt_j = flexnode_xpos_in[worldid, nstart + gidx_j]
                  offset_j = pt_j - subtree_com_in[worldid, body_rootid[bj]]
                  cur_b = bj
                  while cur_b > 0:
                    d_adr = body_dofadr[cur_b]
                    d_num = body_dofnum[cur_b]
                    for d_idx in range(d_adr, d_adr + d_num):
                      cdof = cdof_in[worldid, d_idx]
                      jacp = wp.spatial_bottom(cdof) + wp.cross(wp.spatial_top(cdof), offset_j)
                      v_world_j += jacp * vec_in[worldid, d_idx]
                    cur_b = body_parentid[cur_b]

                v_corot_j = wp.quat_rotate(cell_quat_inv, v_world_j)

                for r in range(3):
                  row = 3 * idx_i + r
                  acc = float(0.0)
                  for c in range(3):
                    col = 3 * idx_j + c
                    acc += flex_stiffness[k_base + row * ndof_cell + col] * v_corot_j[c]
                  frc_corot[r] += acc

          frc_world = wp.quat_rotate(cell_quat, frc_corot)
          if body_simple[bi] == 2 and body_dofnum[bi] == 3:
            dof_i = body_dofadr[bi]
            frc_loc = wp.transpose(xmat_in[worldid, bi]) * frc_world
            for r in range(3):
              wp.atomic_add(res_out, worldid, dof_i + r, scale * frc_loc[r])
          else:
            pt_i = flexnode_xpos_in[worldid, nstart + gidx_i]
            offset_i = pt_i - subtree_com_in[worldid, body_rootid[bi]]
            cur_b = bi
            while cur_b > 0:
              d_adr = body_dofadr[cur_b]
              d_num = body_dofnum[cur_b]
              for d_idx in range(d_adr, d_adr + d_num):
                cdof = cdof_in[worldid, d_idx]
                jacp = wp.spatial_bottom(cdof) + wp.cross(wp.spatial_top(cdof), offset_i)
                wp.atomic_add(res_out, worldid, d_idx, scale * wp.dot(jacp, frc_world))
              cur_b = body_parentid[cur_b]

  return kernel


@wp.kernel(module="unique")
def _pcg_init_tiled(
  # Model:
  nv: int,
  nactuator: int,
  ntendon: int,
  opt_enableflags: int,
  dof_treeid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  efm_ts_in: wp.array2d[float],
  efm_as_in: wp.array2d[float],
  # In:
  rhs_in: wp.array2d[float],
  z0_in: wp.array2d[float],
  flex_any: bool,
  tendon_any: bool,
  actuator_any: bool,
  # Data out:
  qacc_out: wp.array2d[float],
  # Out:
  efm_r_out: wp.array2d[float],
  efm_p_out: wp.array2d[float],
  efm_rz_out: wp.array[float],
  efm_bn_out: wp.array[float],
  efm_done_out: wp.array[bool],
  efm_nsolving_out: wp.array[int],
):
  worldid, tid = wp.tid()
  BLOCK_DIM = wp.block_dim()
  sleep_enabled = (opt_enableflags & EnableBit.SLEEP) != 0

  active = flex_any
  if not active and tendon_any:
    for t in range(ntendon):
      if efm_ts_in[worldid, t] != 0.0:
        active = True
        break
  if not active and actuator_any:
    for a in range(nactuator):
      if efm_as_in[worldid, a] != 0.0:
        active = True
        break

  if not active:
    for dofid in range(tid, nv, BLOCK_DIM):
      z = z0_in[worldid, dofid]
      if sleep_enabled and tree_awake_in[worldid, dof_treeid[dofid]] == 0:
        z = 0.0
      qacc_out[worldid, dofid] = z
      efm_r_out[worldid, dofid] = 0.0
      efm_p_out[worldid, dofid] = 0.0
    if tid == 0:
      efm_bn_out[worldid] = 0.0
      efm_rz_out[worldid] = 0.0
      efm_done_out[worldid] = True
      wp.atomic_sub(efm_nsolving_out, 0, 1)
    return

  local_bn = float(0.0)
  local_rz = float(0.0)
  for dofid in range(tid, nv, BLOCK_DIM):
    b = rhs_in[worldid, dofid]
    z = z0_in[worldid, dofid]
    if sleep_enabled and tree_awake_in[worldid, dof_treeid[dofid]] == 0:
      b = 0.0
      z = 0.0
    efm_r_out[worldid, dofid] = b
    efm_p_out[worldid, dofid] = z
    qacc_out[worldid, dofid] = 0.0
    local_bn += b * b
    local_rz += b * z

  bn_tile = wp.tile(local_bn, preserve_type=True)
  bn_sum = wp.tile_reduce(wp.add, bn_tile)
  rz_tile = wp.tile(local_rz, preserve_type=True)
  rz_sum = wp.tile_reduce(wp.add, rz_tile)
  if tid == 0:
    bn = bn_sum[0]
    efm_bn_out[worldid] = bn
    efm_rz_out[worldid] = rz_sum[0]
    done = bn <= MJ_MINVAL
    efm_done_out[worldid] = done
    if done:
      wp.atomic_sub(efm_nsolving_out, 0, 1)


@wp.kernel(module="unique")
def _pcg_pAp_step_and_check_tiled(
  # Model:
  nv: int,
  opt_tolerance: wp.array[float],
  # In:
  efm_p_in: wp.array2d[float],
  efm_Ap_in: wp.array2d[float],
  efm_rz_in: wp.array[float],
  efm_bn_in: wp.array[float],
  efm_done_in: wp.array[bool],
  # Data out:
  qacc_out: wp.array2d[float],
  # Out:
  efm_r_out: wp.array2d[float],
  efm_done_out: wp.array[bool],
  efm_nsolving_out: wp.array[int],
):
  worldid, tid = wp.tid()
  if efm_done_in[worldid]:
    return

  local_pAp = float(0.0)
  BLOCK_DIM = wp.block_dim()
  for dofid in range(tid, nv, BLOCK_DIM):
    local_pAp += efm_p_in[worldid, dofid] * efm_Ap_in[worldid, dofid]

  pAp_tile = wp.tile(local_pAp, preserve_type=True)
  pAp_sum = wp.tile_reduce(wp.add, pAp_tile)
  pAp = pAp_sum[0]
  if pAp <= 0.0:
    if tid == 0:
      efm_done_out[worldid] = True
      wp.atomic_sub(efm_nsolving_out, 0, 1)
    return

  alpha = efm_rz_in[worldid] / pAp

  local_rr = float(0.0)
  for dofid in range(tid, nv, BLOCK_DIM):
    p = efm_p_in[worldid, dofid]
    ap = efm_Ap_in[worldid, dofid]
    qacc_out[worldid, dofid] += alpha * p
    r = efm_r_out[worldid, dofid] - alpha * ap
    efm_r_out[worldid, dofid] = r
    local_rr += r * r

  rr_tile = wp.tile(local_rr, preserve_type=True)
  rr_sum = wp.tile_reduce(wp.add, rr_tile)
  if tid == 0:
    tol_user = opt_tolerance[worldid % opt_tolerance.shape[0]]
    tol_base = wp.max(tol_user, 1e-5)
    tol = tol_base * tol_base * efm_bn_in[worldid]
    if rr_sum[0] < tol:
      efm_done_out[worldid] = True
      wp.atomic_sub(efm_nsolving_out, 0, 1)


@wp.kernel(module="unique")
def _pcg_beta_and_p_update_tiled(
  # Model:
  nv: int,
  # In:
  efm_r_in: wp.array2d[float],
  efm_z_in: wp.array2d[float],
  efm_rz_in: wp.array[float],
  efm_done_in: wp.array[bool],
  max_iter: int,
  # Out:
  efm_p_out: wp.array2d[float],
  efm_rz_out: wp.array[float],
  efm_nsolving_out: wp.array[int],
  efm_iter_out: wp.array[int],
):
  worldid, tid = wp.tid()
  if worldid == 0 and tid == 0:
    it = efm_iter_out[0] + 1
    efm_iter_out[0] = it
    if it >= max_iter:
      efm_nsolving_out[0] = 0

  if efm_done_in[worldid]:
    return

  local_rznew = float(0.0)
  BLOCK_DIM = wp.block_dim()
  for dofid in range(tid, nv, BLOCK_DIM):
    local_rznew += efm_r_in[worldid, dofid] * efm_z_in[worldid, dofid]

  rz_tile = wp.tile(local_rznew, preserve_type=True)
  rz_sum = wp.tile_reduce(wp.add, rz_tile)
  rznew = rz_sum[0]
  rz = efm_rz_in[worldid]
  beta = math.safe_div(rznew, rz)
  if tid == 0:
    efm_rz_out[worldid] = rznew

  for dofid in range(tid, nv, BLOCK_DIM):
    efm_p_out[worldid, dofid] = efm_z_in[worldid, dofid] + beta * efm_p_out[worldid, dofid]


@cache_kernel
def _pcg_check_convergence_warning(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    efm_done_in: wp.array[bool],
    # Data out:
    overflow_out: wp.array[int],
  ):
    worldid = wp.tid()
    if not efm_done_in[worldid]:
      if wp.static(bool(warn_overflow & OverflowType.ITERATIONS)):
        wp.printf(
          "eff_solve PCG did not converge (world %d)\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
          worldid,
        )
      overflow_out[worldid] = overflow_out[worldid] | wp.static(OverflowType.ITERATIONS)

  return kernel


@wp.kernel
def _eff_contact_shift(
  # Model:
  opt_timestep: wp.array[float],
  # Data in:
  qvel_in: wp.array2d[float],
  contact_worldid_in: wp.array[int],
  # In:
  efm_con_dof_in: wp.array2d[int],
  efm_con_val_in: wp.array2d[float],
  efm_con_scale_in: wp.array[float],
  efm_con_nnz_in: wp.array[int],
  # Data out:
  efm_c_out: wp.array2d[float],
):
  """Applies passive contact smooth-force velocity shift c += -h*k*Jn^T*(Jn*v)."""
  cid = wp.tid()
  nnz = efm_con_nnz_in[cid]
  if nnz == 0:
    return
  scale = efm_con_scale_in[cid]
  if scale == float(0.0):
    return
  worldid = contact_worldid_in[cid]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  neg_scale_dt = -scale / timestep
  dot = float(0.0)
  for a in range(nnz):
    dof = efm_con_dof_in[cid, a]
    dot += efm_con_val_in[cid, a] * qvel_in[worldid, dof]
  scaled_dot = dot * neg_scale_dt
  for a in range(nnz):
    dof = efm_con_dof_in[cid, a]
    wp.atomic_add(efm_c_out, worldid, dof, scaled_dot * efm_con_val_in[cid, a])


@cache_kernel
def _eff_mul_add_contact(check_skip: bool):
  """Multiplies vector by passive contact rank-1 metric curvature h^2*k*Jn^T*Jn."""

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    contact_worldid_in: wp.array[int],
    # In:
    efm_con_dof_in: wp.array2d[int],
    efm_con_val_in: wp.array2d[float],
    efm_con_scale_in: wp.array[float],
    efm_con_nnz_in: wp.array[int],
    vec_in: wp.array2d[float],
    skip: wp.array[bool],
    # Out:
    res_out: wp.array2d[float],
  ):
    cid = wp.tid()
    nnz = efm_con_nnz_in[cid]
    if nnz == 0:
      return
    worldid = contact_worldid_in[cid]
    if wp.static(check_skip):
      if skip[worldid]:
        return
    scale = efm_con_scale_in[cid]
    if scale == float(0.0):
      return
    dot = float(0.0)
    for a in range(nnz):
      dof = efm_con_dof_in[cid, a]
      dot += efm_con_val_in[cid, a] * vec_in[worldid, dof]
    dot *= scale
    for a in range(nnz):
      dof = efm_con_dof_in[cid, a]
      wp.atomic_add(res_out, worldid, dof, dot * efm_con_val_in[cid, a])

  return kernel


@event_scope
def eff_build(m: Model, d: Data):
  """Builds the implicit effective metric for the current position state."""
  if m.opt.integrator != IntegratorType.DISCRETE:
    return

  d.efm_c.zero_()
  d.efm_ca.zero_()
  d.efm_diag.zero_()
  d.efm_ts.zero_()
  d.efm_as.zero_()

  if m.nflex > 0 and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
    passive.flex_hessian(m, d)

  if m.nefmK > 0:
    d.efm_K_val.zero_()
    d.efm_L.zero_()

    wp.launch(
      _eff_flex_stretch_stiff_vert,
      dim=(d.nworld, m.nflexvert),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.flex_dim,
        m.flex_interp,
        m.flex_stiffnessadr,
        m.flex_vertbodyid,
        m.flex_stiffness,
        m.flex_damping,
        m.flex_rigid,
        m.efm_K_rownnz,
        m.efm_K_rowadr,
        m.efm_K_colind,
        m.flex_vertflexid,
        m.flex_simple,
        d.xmat,
        d.flexvert_hessian,
      ],
      outputs=[d.efm_K_val],
    )

    wp.launch(
      _eff_flex_stretch_stiff_edge,
      dim=(d.nworld, m.nflexedge, 2),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.flex_dim,
        m.flex_interp,
        m.flex_vertadr,
        m.flex_stiffnessadr,
        m.flex_vertbodyid,
        m.flex_edge,
        m.flex_stiffness,
        m.flex_damping,
        m.flex_rigid,
        m.efm_K_rownnz,
        m.efm_K_rowadr,
        m.efm_K_colind,
        m.flex_edgeflexid,
        m.flex_simple,
        d.xmat,
        d.flexedge_hessian,
      ],
      outputs=[d.efm_K_val],
    )

    wp.launch(
      _eff_flex_bend_stiff,
      dim=(d.nworld, m.nflexedge),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.flex_dim,
        m.flex_interp,
        m.flex_vertadr,
        m.flex_edgeadr,
        m.flex_edgenum,
        m.flex_bendingadr,
        m.flex_vertbodyid,
        m.flex_edge,
        m.flex_edgeflap,
        m.flex_bending,
        m.flex_damping,
        m.flex_rigid,
        m.flex_centered,
        m.efm_K_rownnz,
        m.efm_K_rowadr,
        m.efm_K_colind,
        m.flex_edgeflexid,
        m.flex_simple,
        d.xquat,
        d.xmat,
      ],
      outputs=[d.efm_K_val],
    )

    if m.flex_interp_assemblable:
      wp.launch(
        _eff_flex_interp_stiff,
        dim=(d.nworld, m.nflexintcell),
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          m.body_dofnum,
          m.body_dofadr,
          m.flex_interp,
          m.flex_cellnum,
          m.flex_nodeadr,
          m.flex_stiffnessadr,
          m.flex_nodebodyid,
          m.flex_stiffness,
          m.flex_damping,
          m.flex_centered,
          m.efm_K_rownnz,
          m.efm_K_rowadr,
          m.efm_K_colind,
          m.flex_cell_map,
          d.xmat,
          d.flexnode_xpos,
        ],
        outputs=[d.efm_K_val],
      )

    wp.launch(
      _eff_factor_blocks,
      dim=(d.nworld, m.nefmdof),
      inputs=[
        m.M_rownnz,
        m.M_rowadr,
        m.M_colind,
        m.efm_K_rownnz,
        m.efm_K_rowadr,
        m.efm_K_colind,
        m.efm_dofid,
        d.M,
        d.efm_K_val,
      ],
      outputs=[d.efm_L],
    )


@event_scope
def eff_shift(m: Model, d: Data):
  """Refreshes velocity-stage terms and computes the smooth force shift c = -h*K*v."""
  if m.opt.integrator != IntegratorType.DISCRETE:
    return

  d.efm_diag.zero_()

  wp.launch(
    _eff_diag_stiff,
    dim=(d.nworld, m.njnt),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.qpos_spring,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      d.qpos,
    ],
    outputs=[d.efm_diag],
  )

  wp.launch(
    _eff_diag_damp_shift,
    dim=(d.nworld, m.nv),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.dof_damping,
      m.dof_dampingpoly,
      d.qvel,
    ],
    outputs=[d.efm_c, d.efm_diag],
  )

  wp.launch(
    _eff_tendon_shift,
    dim=(d.nworld, m.ntendon),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.tendon_stiffness,
      m.tendon_stiffnesspoly,
      m.tendon_damping,
      m.tendon_dampingpoly,
      m.tendon_lengthspring,
      d.ten_J,
      d.ten_length,
      d.ten_velocity,
    ],
    outputs=[d.efm_c, d.efm_ts],
  )

  if not (m.opt.disableflags & DisableBit.SPRING):
    passive.flex_hessian(m, d)
    passive.flex_bend_mul(m, d, d.efm_c, d.qvel, s1_in=-1.0, s2_in=0.0, use_timestep=True, non_simple_only=False)
    passive.flex_stretch_mul(m, d, d.efm_c, d.qvel, s1_in=-1.0, s2_in=0.0, use_timestep=True, non_simple_only=False)

    wp.launch(
      _eff_flex_interp_mul(False, True),
      dim=(d.nworld, m.nflexintcell),
      inputs=[
        m.opt.timestep,
        m.opt.disableflags,
        m.body_parentid,
        m.body_rootid,
        m.body_dofnum,
        m.body_dofadr,
        m.body_simple,
        m.flex_interp,
        m.flex_cellnum,
        m.flex_nodeadr,
        m.flex_stiffnessadr,
        m.flex_nodebodyid,
        m.flex_stiffness,
        m.flex_damping,
        m.flex_cell_map,
        d.xmat,
        d.subtree_com,
        d.cdof,
        d.flexnode_xpos,
        d.qvel,
        m.body_is_free,
      ],
      outputs=[d.efm_c],
    )

  if passive.flex_passive_contact(m) and not (
    (m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)
  ):
    efm_con_dof, efm_con_val, efm_con_scale, _, efm_con_nnz = build_efm_contact(m, d)
    wp.launch(
      _eff_contact_shift,
      dim=d.naconmax,
      inputs=[
        m.opt.timestep,
        d.qvel,
        d.contact.worldid,
        efm_con_dof,
        efm_con_val,
        efm_con_scale,
        efm_con_nnz,
      ],
      outputs=[d.efm_c],
    )

  d.efm_fluid.zero_()
  if m.has_fluid and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
    Mi = m.M_fullm_i
    Mj = m.M_fullm_j
    wp.launch(
      _qderiv_ellipsoid_fluid,
      dim=(d.nworld, m.body_fluid_ellipsoid_adr.size, Mi.size),
      inputs=[
        m.opt.timestep,
        m.opt.wind,
        m.opt.density,
        m.opt.viscosity,
        m.opt.integrator,
        m.body_parentid,
        m.body_rootid,
        m.body_geomnum,
        m.body_geomadr,
        m.dof_bodyid,
        m.geom_type,
        m.geom_size,
        m.geom_fluid,
        m.body_fluid_ellipsoid_adr,
        m.body_isdofancestor,
        m.M_elemid,
        d.xipos,
        d.geom_xpos,
        d.geom_xmat,
        d.subtree_com,
        d.cdof,
        d.cvel,
        Mi,
        Mj,
      ],
      outputs=[d.efm_fluid],
    )
    wp.launch(
      _qderiv_box_fluid,
      dim=(d.nworld, m.body_fluid_box_adr.size, Mi.size),
      inputs=[
        m.opt.timestep,
        m.opt.wind,
        m.opt.density,
        m.opt.viscosity,
        m.opt.integrator,
        m.body_parentid,
        m.body_rootid,
        m.body_mass,
        m.body_inertia,
        m.dof_bodyid,
        m.body_fluid_box_adr,
        m.body_isdofancestor,
        m.M_elemid,
        d.xipos,
        d.ximat,
        d.subtree_com,
        d.cdof,
        d.cvel,
        Mi,
        Mj,
      ],
      outputs=[d.efm_fluid],
    )


@event_scope
def eff_actuation(m: Model, d: Data):
  """Evaluates actuation-stage metric terms and factors the backbone qH."""
  if m.opt.integrator != IntegratorType.DISCRETE:
    return

  if m.nhistory > 0:
    ctrl = wp.empty((d.nworld, m.nu), dtype=float)
    history.read_ctrl_delayed(m, d, ctrl)
  else:
    ctrl = d.ctrl

  d.efm_ca.zero_()
  wp.launch(
    _eff_actuator_actuation,
    dim=(d.nworld, m.nactuator),
    inputs=[
      m.opt.timestep,
      m.opt.disableflags,
      m.actuator_dyntype,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_ctrladr,
      m.actuator_ctrlspec,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_dynprm,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_actlimited,
      m.actuator_actrange,
      m.actuator_actearly,
      m.actuator_forcelimited,
      m.actuator_forcerange,
      m.actuator_ctrllimited,
      m.actuator_ctrlrange,
      m.actuator_acc0,
      m.actuator_lengthrange,
      d.act,
      ctrl,
      d.act_dot,
      d.actuator_length,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.actuator_velocity,
      d.actuator_force,
    ],
    outputs=[d.efm_ca, d.efm_as],
  )

  wp.copy(d.qH, d.M)
  wp.launch(
    _eff_init_sdiag,
    dim=(d.nworld, m.nv),
    inputs=[
      m.M_rownnz,
      m.M_rowadr,
      d.efm_diag,
    ],
    outputs=[d.qH],
  )
  wp.launch(
    _eff_add_tendon_qH_diag,
    dim=(d.nworld, m.ntendon),
    inputs=[
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.M_rownnz,
      m.M_rowadr,
      d.ten_J,
      d.efm_ts,
    ],
    outputs=[d.qH],
  )
  wp.launch(
    _eff_add_actuator_qH_diag,
    dim=(d.nworld, m.nactuator),
    inputs=[
      m.M_rownnz,
      m.M_rowadr,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.efm_as,
    ],
    outputs=[d.qH],
  )

  if m.has_fluid and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
    wp.launch(
      _eff_add_fluid_qH,
      dim=(d.nworld, m.nC),
      inputs=[d.efm_fluid],
      outputs=[d.qH],
    )

  # Factor qH into qHLD and qHDiagInv
  if m.M_tiles:
    smooth._factor_blocks(m, d, d.qH, d.qHLD, d.qHDiagInv)
  if d.qHLD.shape[1] > m.qLD_block_total:
    smooth._factor_i_sparse(m, d, d.qH, d.qHLD[:, m.qLD_block_total :], d.qHDiagInv)


def eff_mul_m_scratch(m: Model, d: Data) -> tuple[Optional[tuple], Optional[tuple]]:
  """Builds the eff_mul_m efm_con and flex_scratch arguments.

  eff_mul_m allocates them on each call if they are not passed, which is not allowed inside
  wp.capture_while: build them before the loop instead.
  """
  efm_con = build_efm_contact(m, d) if passive.flex_passive_contact(m) else None
  flex_scratch = passive.flex_stretch_scratch(m, d) if m.nflex and m.nflexvert else None
  return efm_con, flex_scratch


@event_scope
def eff_mul_m(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  skip: Optional[wp.array] = None,
  efm_con: Optional[tuple] = None,
  flex_scratch: Optional[tuple] = None,
):
  """Multiplies vector by effective metric M_hat = M + h*D + h^2*K.

  Inside wp.capture_while, where memory cannot be allocated, pass efm_con and flex_scratch
  from eff_mul_m_scratch, built before the loop.
  """
  support.mul_m(m, d, res, vec, skip=skip)
  if m.opt.integrator == IntegratorType.DISCRETE:
    check_skip = skip is not None
    skip_arr = skip if check_skip else m.body_is_free
    if m.nefmK > 0:
      wp.launch(
        _eff_mul_diag_and_csr(check_skip),
        dim=(d.nworld, m.nv, EFF_MUL_SPLIT),
        inputs=[
          m.efm_K_rownnz,
          m.efm_K_rowadr,
          m.efm_K_colind,
          d.efm_diag,
          d.efm_K_val,
          vec,
          skip_arr,
        ],
        outputs=[res],
      )
    else:
      wp.launch(
        _eff_mul_diag(check_skip),
        dim=(d.nworld, m.nv),
        inputs=[d.efm_diag, vec, skip_arr],
        outputs=[res],
      )
    wp.launch(
      _eff_mul_add_tendon(check_skip),
      dim=(d.nworld, m.ntendon),
      inputs=[
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        d.ten_J,
        d.efm_ts,
        vec,
        skip_arr,
      ],
      outputs=[res],
    )
    wp.launch(
      _eff_mul_add_actuator(check_skip),
      dim=(d.nworld, m.nactuator),
      inputs=[
        d.moment_rownnz,
        d.moment_rowadr,
        d.moment_colind,
        d.actuator_moment,
        d.efm_as,
        vec,
        skip_arr,
      ],
      outputs=[res],
    )
    if passive.flex_passive_contact(m) and not (
      (m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)
    ):
      if efm_con is None:
        efm_con_dof, efm_con_val, efm_con_scale, _, efm_con_nnz = build_efm_contact(m, d)
      else:
        efm_con_dof, efm_con_val, efm_con_scale, _, efm_con_nnz = efm_con
      wp.launch(
        _eff_mul_add_contact(check_skip),
        dim=d.naconmax,
        inputs=[
          d.contact.worldid,
          efm_con_dof,
          efm_con_val,
          efm_con_scale,
          efm_con_nnz,
          vec,
          skip_arr,
        ],
        outputs=[res],
      )
    if m.has_fluid and not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)):
      wp.launch(
        _eff_mul_fluid(check_skip),
        dim=(d.nworld, m.nv),
        inputs=[
          m.M_mulm_rowadr,
          m.M_mulm_col,
          m.M_mulm_madr,
          d.efm_fluid,
          vec,
          skip_arr,
        ],
        outputs=[res],
      )
    if (m.nefmK == 0 or m.has_non_simple_flex) and not (
      (m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)
    ):
      passive.flex_bend_mul(m, d, res, vec, s1_in=1.0, s2_in=1.0, use_timestep=True, non_simple_only=(m.nefmK > 0), skip=skip)
      passive.flex_stretch_mul(
        m,
        d,
        res,
        vec,
        s1_in=1.0,
        s2_in=1.0,
        use_timestep=True,
        non_simple_only=(m.nefmK > 0),
        skip=skip,
        scratch=flex_scratch,
      )
    if not m.flex_interp_assemblable and not (
      (m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER)
    ):
      wp.launch(
        _eff_flex_interp_mul(check_skip, False),
        dim=(d.nworld, m.nflexintcell),
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          m.body_parentid,
          m.body_rootid,
          m.body_dofnum,
          m.body_dofadr,
          m.body_simple,
          m.flex_interp,
          m.flex_cellnum,
          m.flex_nodeadr,
          m.flex_stiffnessadr,
          m.flex_nodebodyid,
          m.flex_stiffness,
          m.flex_damping,
          m.flex_cell_map,
          d.xmat,
          d.subtree_com,
          d.cdof,
          d.flexnode_xpos,
          vec,
          skip_arr,
        ],
        outputs=[res],
      )
    if m.opt.enableflags & EnableBit.SLEEP:
      wp.launch(
        zero_sleeping_dofs,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_treeid, d.tree_awake],
        outputs=[res],
      )


@wp.kernel
def _eff_build_blocks_raw(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  efm_dofid: wp.array[int],
  is_sparse: bool,
  # Data in:
  nefc_in: wp.array[int],
  M_in: wp.array2d[float],
  efm_K_val_in: wp.array2d[float],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  # Out:
  epB_out: wp.array2d[float],
):
  worldid, k = wp.tid()
  i = efm_dofid[k]
  B = _extract_3x3_diag_block(
    M_rownnz, M_rowadr, M_colind, efm_K_rownnz, efm_K_rowadr, efm_K_colind, M_in, efm_K_val_in, i, worldid
  )

  if not is_sparse:
    nefc = nefc_in[worldid]
    for r in range(nefc):
      D = efc_D_in[worldid, r]
      if D != 0.0:
        j0 = efc_J_in[worldid, r, i]
        j1 = efc_J_in[worldid, r, i + 1]
        j2 = efc_J_in[worldid, r, i + 2]
        if j0 != 0.0 or j1 != 0.0 or j2 != 0.0:
          B[0, 0] += D * j0 * j0
          B[0, 1] += D * j0 * j1
          B[0, 2] += D * j0 * j2
          B[1, 0] += D * j1 * j0
          B[1, 1] += D * j1 * j1
          B[1, 2] += D * j1 * j2
          B[2, 0] += D * j2 * j0
          B[2, 1] += D * j2 * j1
          B[2, 2] += D * j2 * j2

  base = 9 * k
  for r in range(3):
    for c in range(3):
      epB_out[worldid, base + 3 * r + c] = B[r, c]


@wp.kernel
def _eff_fold_tendon(
  # Model:
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  efm_dofid: wp.array[int],
  efm_dofblk: wp.array[int],
  # Data in:
  ten_J_in: wp.array2d[float],
  efm_ts_in: wp.array2d[float],
  # Out:
  epB_out: wp.array2d[float],
):
  worldid, t = wp.tid()
  s = efm_ts_in[worldid, t]
  if s == 0.0:
    return
  rowadr = ten_J_rowadr[t]
  rownnz = ten_J_rownnz[t]
  for a in range(rownnz):
    ia = ten_J_colind[rowadr + a]
    k = efm_dofblk[ia]
    if k < 0:
      continue
    base = efm_dofid[k]
    s_va = s * ten_J_in[worldid, rowadr + a]
    for b in range(rownnz):
      ib = ten_J_colind[rowadr + b]
      if efm_dofblk[ib] == k:
        vb = ten_J_in[worldid, rowadr + b]
        wp.atomic_add(epB_out, worldid, 9 * k + 3 * (ia - base) + (ib - base), s_va * vb)


@wp.kernel
def _eff_fold_actuator(
  # Model:
  efm_dofid: wp.array[int],
  efm_dofblk: wp.array[int],
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  efm_as_in: wp.array2d[float],
  # Out:
  epB_out: wp.array2d[float],
):
  worldid, u = wp.tid()
  s = efm_as_in[worldid, u]
  if s == 0.0:
    return
  rowadr = moment_rowadr_in[worldid, u]
  rownnz = moment_rownnz_in[worldid, u]
  for a in range(rownnz):
    ia = moment_colind_in[worldid, rowadr + a]
    k = efm_dofblk[ia]
    if k < 0:
      continue
    base = efm_dofid[k]
    s_va = s * actuator_moment_in[worldid, rowadr + a]
    for b in range(rownnz):
      ib = moment_colind_in[worldid, rowadr + b]
      if efm_dofblk[ib] == k:
        vb = actuator_moment_in[worldid, rowadr + b]
        wp.atomic_add(epB_out, worldid, 9 * k + 3 * (ia - base) + (ib - base), s_va * vb)


@wp.kernel
def _eff_fold_efc_sparse(
  # Model:
  efm_dofid: wp.array[int],
  efm_dofblk: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  # Out:
  epB_out: wp.array2d[float],
):
  worldid, r = wp.tid()
  if r >= nefc_in[worldid]:
    return
  D = efc_D_in[worldid, r]
  if D == 0.0:
    return
  adr = efc_J_rowadr_in[worldid, r]
  nnz = efc_J_rownnz_in[worldid, r]
  for a in range(nnz):
    ia = efc_J_colind_in[worldid, 0, adr + a]
    k = efm_dofblk[ia]
    if k < 0:
      continue
    base = efm_dofid[k]
    D_ja = D * efc_J_in[worldid, 0, adr + a]
    for b in range(nnz):
      ib = efc_J_colind_in[worldid, 0, adr + b]
      if efm_dofblk[ib] == k:
        jb = efc_J_in[worldid, 0, adr + b]
        wp.atomic_add(epB_out, worldid, 9 * k + 3 * (ia - base) + (ib - base), D_ja * jb)


@wp.kernel
def _eff_fold_contact_rank1(
  # Model:
  efm_dofid: wp.array[int],
  efm_dofblk: wp.array[int],
  # Data in:
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  efm_con_dof_in: wp.array2d[int],
  efm_con_val_in: wp.array2d[float],
  efm_con_scale_in: wp.array[float],
  efm_con_nnz_in: wp.array[int],
  # Out:
  epB_out: wp.array2d[float],
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  scale = efm_con_scale_in[cid]
  if scale == 0.0:
    return
  worldid = contact_worldid_in[cid]
  nnz = efm_con_nnz_in[cid]
  for a in range(nnz):
    ia = efm_con_dof_in[cid, a]
    k = efm_dofblk[ia]
    if k < 0:
      continue
    base = efm_dofid[k]
    scale_va = scale * efm_con_val_in[cid, a]
    for b in range(nnz):
      ib = efm_con_dof_in[cid, b]
      if efm_dofblk[ib] == k:
        vb = efm_con_val_in[cid, b]
        wp.atomic_add(epB_out, worldid, 9 * k + 3 * (ia - base) + (ib - base), scale_va * vb)


@wp.kernel
def _eff_factor_folded_blocks(
  # In:
  epB_in: wp.array2d[float],
  # Out:
  epL_out: wp.array2d[float],
):
  worldid, k = wp.tid()
  base = 9 * k
  _write_cholesky33(
    worldid,
    base,
    epB_in[worldid, base + 0],
    epB_in[worldid, base + 3],
    epB_in[worldid, base + 4],
    epB_in[worldid, base + 6],
    epB_in[worldid, base + 7],
    epB_in[worldid, base + 8],
    epL_out,
  )


@wp.kernel
def _eff_build_uncovered_map(
  # Model:
  nv: int,
  efm_dofblk: wp.array[int],
  # Out:
  epU_out: wp.array[int],
  ep_loc_out: wp.array[int],
):
  tid = wp.tid()
  if tid != 0:
    return
  cnt = int(0)
  for i in range(nv):
    if efm_dofblk[i] < 0:
      epU_out[cnt] = i
      ep_loc_out[i] = cnt
      cnt += 1
    else:
      ep_loc_out[i] = -1


@wp.kernel
def _eff_build_dense_uncovered_base(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  has_fluid: bool,
  # Data in:
  M_in: wp.array2d[float],
  efm_diag_in: wp.array2d[float],
  efm_fluid_in: wp.array2d[float],
  # In:
  epU_in: wp.array[int],
  ep_loc_in: wp.array[int],
  # Out:
  epS_out: wp.array3d[float],
):
  worldid, ua = wp.tid()
  i = epU_in[ua]
  wp.atomic_add(epS_out, worldid, ua, ua, efm_diag_in[worldid, i])
  rowadr = M_rowadr[i]
  rownnz = M_rownnz[i]
  for a in range(rowadr, rowadr + rownnz):
    j = M_colind[a]
    ub = ep_loc_in[j]
    if ub < 0:
      continue
    v = M_in[worldid, a]
    if has_fluid:
      v += efm_fluid_in[worldid, a]
    wp.atomic_add(epS_out, worldid, ua, ub, v)
    if ub != ua:
      wp.atomic_add(epS_out, worldid, ub, ua, v)


@wp.kernel
def _eff_fold_dense_uncovered_tendon(
  # Model:
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  # Data in:
  ten_J_in: wp.array2d[float],
  efm_ts_in: wp.array2d[float],
  # In:
  ep_loc_in: wp.array[int],
  # Out:
  epS_out: wp.array3d[float],
):
  worldid, t = wp.tid()
  s = efm_ts_in[worldid, t]
  if s == 0.0:
    return
  rowadr = ten_J_rowadr[t]
  rownnz = ten_J_rownnz[t]
  for a in range(rownnz):
    ia = ten_J_colind[rowadr + a]
    ua = ep_loc_in[ia]
    if ua < 0:
      continue
    s_va = s * ten_J_in[worldid, rowadr + a]
    for b in range(rownnz):
      ib = ten_J_colind[rowadr + b]
      ub = ep_loc_in[ib]
      if ub >= 0:
        vb = ten_J_in[worldid, rowadr + b]
        wp.atomic_add(epS_out, worldid, ua, ub, s_va * vb)


@wp.kernel
def _eff_fold_dense_uncovered_actuator(
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  efm_as_in: wp.array2d[float],
  # In:
  ep_loc_in: wp.array[int],
  # Out:
  epS_out: wp.array3d[float],
):
  worldid, u = wp.tid()
  s = efm_as_in[worldid, u]
  if s == 0.0:
    return
  rowadr = moment_rowadr_in[worldid, u]
  rownnz = moment_rownnz_in[worldid, u]
  for a in range(rownnz):
    ia = moment_colind_in[worldid, rowadr + a]
    ua = ep_loc_in[ia]
    if ua < 0:
      continue
    s_va = s * actuator_moment_in[worldid, rowadr + a]
    for b in range(rownnz):
      ib = moment_colind_in[worldid, rowadr + b]
      ub = ep_loc_in[ib]
      if ub >= 0:
        vb = actuator_moment_in[worldid, rowadr + b]
        wp.atomic_add(epS_out, worldid, ua, ub, s_va * vb)


@wp.kernel
def _eff_fold_dense_uncovered_contact(
  # Data in:
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  efm_con_dof_in: wp.array2d[int],
  efm_con_val_in: wp.array2d[float],
  efm_con_scale_in: wp.array[float],
  efm_con_nnz_in: wp.array[int],
  ep_loc_in: wp.array[int],
  # Out:
  epS_out: wp.array3d[float],
):
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  scale = efm_con_scale_in[cid]
  if scale == 0.0:
    return
  worldid = contact_worldid_in[cid]
  nnz = efm_con_nnz_in[cid]
  for a in range(nnz):
    ia = efm_con_dof_in[cid, a]
    ua = ep_loc_in[ia]
    if ua < 0:
      continue
    scale_va = scale * efm_con_val_in[cid, a]
    for b in range(nnz):
      ib = efm_con_dof_in[cid, b]
      ub = ep_loc_in[ib]
      if ub >= 0:
        vb = efm_con_val_in[cid, b]
        wp.atomic_add(epS_out, worldid, ua, ub, scale_va * vb)


@wp.kernel
def _eff_fold_dense_uncovered_efc(
  # Model:
  is_sparse: bool,
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  # In:
  n_u: int,
  epU_in: wp.array[int],
  ep_loc_in: wp.array[int],
  # Out:
  epS_out: wp.array3d[float],
):
  worldid, r = wp.tid()
  if r >= nefc_in[worldid]:
    return
  D = efc_D_in[worldid, r]
  if D <= 0.0:
    return
  if is_sparse:
    adr = efc_J_rowadr_in[worldid, r]
    nnz = efc_J_rownnz_in[worldid, r]
    for a in range(nnz):
      ia = efc_J_colind_in[worldid, 0, adr + a]
      ua = ep_loc_in[ia]
      if ua < 0:
        continue
      D_ja = D * efc_J_in[worldid, 0, adr + a]
      for b in range(nnz):
        ib = efc_J_colind_in[worldid, 0, adr + b]
        ub = ep_loc_in[ib]
        if ub >= 0:
          jb = efc_J_in[worldid, 0, adr + b]
          wp.atomic_add(epS_out, worldid, ua, ub, D_ja * jb)
  else:
    for ua in range(n_u):
      ia = epU_in[ua]
      ja = efc_J_in[worldid, r, ia]
      if ja == 0.0:
        continue
      D_ja = D * ja
      for ub in range(n_u):
        ib = epU_in[ub]
        jb = efc_J_in[worldid, r, ib]
        if jb != 0.0:
          wp.atomic_add(epS_out, worldid, ua, ub, D_ja * jb)


@wp.kernel
def _eff_pad_and_clamp_dense_uncovered(
  # In:
  n_u: int,
  # Out:
  epS_out: wp.array3d[float],
):
  worldid, ua = wp.tid()
  if ua < n_u:
    epS_out[worldid, ua, ua] = wp.max(epS_out[worldid, ua, ua], MJ_MINVAL)
  else:
    epS_out[worldid, ua, ua] = 1.0


@cache_kernel
def _eff_cholesky_factor(tile_size: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Out:
    epS_out: wp.array3d[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)
    mat_tile = wp.tile_load(epS_out[worldid], shape=(TILE_SIZE, TILE_SIZE))
    wp.tile_cholesky_inplace(mat_tile, fill_mode="upper")
    wp.tile_store(epS_out[worldid], mat_tile)

  return kernel


@cache_kernel
def _eff_cholesky_factor_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # In:
    epS_in: wp.array3d[float],
    epS_rhs_in: wp.array3d[float],
    # Out:
    epS_factor_out: wp.array3d[float],
    epS_sol_out: wp.array3d[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)
    wp.static(create_blocked_cholesky_factorize_solve_func(TILE_SIZE, matrix_size))(
      epS_in[worldid],
      epS_rhs_in[worldid],
      matrix_size,
      epS_factor_out[worldid],
      epS_sol_out[worldid],
    )

  return kernel


# the uncovered dense block's inverse is formed when its padded size fits one shared-memory tile
EFF_UNCOVERED_INV_MAX = 64


@cache_kernel
def _eff_uncovered_inverse(size: int, factored: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # In:
    epS_factor_in: wp.array3d[float],
    # Out:
    epS_out: wp.array3d[float],
  ):
    """Overwrites epS with the inverse of the uncovered block, A^-1 = U^-1 U^-T for A = U'U.

    Without factored, epS holds A and is factored here.
    """
    worldid = wp.tid()
    if wp.static(factored):
      U = wp.tile_load(epS_factor_in[worldid], shape=(size, size), storage="shared")
    else:
      U = wp.tile_load(epS_out[worldid], shape=(size, size), storage="shared")
      wp.tile_cholesky_inplace(U, fill_mode="upper")
    eye = wp.tile_diag_add(wp.tile_zeros(shape=(size, size), dtype=float), wp.tile_ones(shape=size, dtype=float))
    Y = wp.tile_upper_solve(U, eye)
    inv = wp.tile_zeros(shape=(size, size), dtype=float)
    wp.tile_matmul(Y, wp.tile_transpose(Y), inv)
    wp.tile_store(epS_out[worldid], inv)

  return kernel


@wp.kernel
def _eff_uncovered_inv_mul(
  # In:
  n_u: int,
  epU_in: wp.array[int],
  epS_in: wp.array3d[float],
  vec_in: wp.array2d[float],
  # Out:
  res_out: wp.array2d[float],
):
  """Res = A^-1 vec on uncovered dof a, from the uncovered block's inverse."""
  worldid, a = wp.tid()
  s = float(0.0)
  for b in range(n_u):
    s += epS_in[worldid, a, b] * vec_in[worldid, epU_in[b]]
  res_out[worldid, epU_in[a]] = s


@wp.kernel
def _eff_gather_uncovered_rhs(
  # In:
  n_u: int,
  epU_in: wp.array[int],
  vec_in: wp.array2d[float],
  # Out:
  epS_rhs_out: wp.array2d[float],
):
  worldid, ua = wp.tid()
  if ua < n_u:
    epS_rhs_out[worldid, ua] = vec_in[worldid, epU_in[ua]]
  else:
    epS_rhs_out[worldid, ua] = 0.0


@cache_kernel
def _eff_cholesky_solve(tile_size: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # In:
    epS_in: wp.array3d[float],
    epS_rhs_in: wp.array2d[float],
    # Out:
    epS_sol_out: wp.array2d[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)
    mat_tile = wp.tile_load(epS_in[worldid], shape=(TILE_SIZE, TILE_SIZE))
    rhs_tile = wp.tile_load(epS_rhs_in[worldid], shape=TILE_SIZE)
    sol_tile = wp.tile_cholesky_solve(mat_tile, rhs_tile, fill_mode="upper")
    wp.tile_store(epS_sol_out[worldid], sol_tile)

  return kernel


@cache_kernel
def _eff_cholesky_solve_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"enable_mathdx_gemm": False})
  def kernel(
    # In:
    epS_factor_in: wp.array3d[float],
    epS_rhs_in: wp.array3d[float],
    # Out:
    epS_sol_out: wp.array3d[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)
    wp.static(create_blocked_cholesky_solve_func(TILE_SIZE, matrix_size))(
      epS_factor_in[worldid],
      epS_rhs_in[worldid],
      matrix_size,
      epS_sol_out[worldid],
    )

  return kernel


@wp.kernel
def _eff_scatter_uncovered_sol(
  # In:
  epU_in: wp.array[int],
  epS_sol_in: wp.array2d[float],
  # Out:
  res_out: wp.array2d[float],
):
  worldid, ua = wp.tid()
  res_out[worldid, epU_in[ua]] = epS_sol_in[worldid, ua]


@event_scope
def eff_prec_fold(
  m: Model,
  d: Data,
  out: Optional[wp.array] = None,
  epB: Optional[wp.array] = None,
  efm_con: Optional[tuple] = None,
  epS: Optional[wp.array] = None,
  epS_factor: Optional[wp.array] = None,
  epS_rhs: Optional[wp.array] = None,
  epS_sol: Optional[wp.array] = None,
  epU: Optional[wp.array] = None,
  ep_loc: Optional[wp.array] = None,
) -> wp.array:
  """Folds rank-1 metric terms and active/inactive efc rows into 3x3 preconditioner blocks."""
  epL = out if out is not None else wp.empty_like(d.efm_L)
  if m.nefmdof == 0:
    return epL
  if epB is None:
    epB = wp.empty_like(d.efm_L)
  dofblk = m.efm_dofblk

  wp.launch(
    _eff_build_blocks_raw,
    dim=(d.nworld, m.nefmdof),
    inputs=[
      m.M_rownnz,
      m.M_rowadr,
      m.M_colind,
      m.efm_K_rownnz,
      m.efm_K_rowadr,
      m.efm_K_colind,
      m.efm_dofid,
      m.is_sparse,
      d.nefc,
      d.M,
      d.efm_K_val,
      d.efc.J,
      d.efc.D,
    ],
    outputs=[epB],
  )

  wp.launch(
    _eff_fold_tendon,
    dim=(d.nworld, m.ntendon),
    inputs=[
      m.ten_J_rownnz,
      m.ten_J_rowadr,
      m.ten_J_colind,
      m.efm_dofid,
      dofblk,
      d.ten_J,
      d.efm_ts,
    ],
    outputs=[epB],
  )

  wp.launch(
    _eff_fold_actuator,
    dim=(d.nworld, m.nactuator),
    inputs=[
      m.efm_dofid,
      dofblk,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.efm_as,
    ],
    outputs=[epB],
  )

  if m.is_sparse:
    wp.launch(
      _eff_fold_efc_sparse,
      dim=(d.nworld, d.njmax),
      inputs=[
        m.efm_dofid,
        dofblk,
        d.nefc,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.efc.D,
      ],
      outputs=[epB],
    )

  con_tuple = efm_con
  if passive.flex_passive_contact(m):
    if con_tuple is None:
      con_tuple = build_efm_contact(m, d)
    efm_con_dof, efm_con_val, efm_con_scale, _, efm_con_nnz = con_tuple
    wp.launch(
      _eff_fold_contact_rank1,
      dim=d.naconmax,
      inputs=[
        m.efm_dofid,
        dofblk,
        d.contact.worldid,
        d.nacon,
        efm_con_dof,
        efm_con_val,
        efm_con_scale,
        efm_con_nnz,
      ],
      outputs=[epB],
    )

  wp.launch(
    _eff_factor_folded_blocks,
    dim=(d.nworld, m.nefmdof),
    inputs=[epB],
    outputs=[epL],
  )

  if epS is not None and epU is not None and ep_loc is not None:
    n_u = epU.shape[0]
    n_u_pad = epS.shape[1]
    wp.launch(
      _eff_build_uncovered_map,
      dim=1,
      inputs=[m.nv, dofblk],
      outputs=[epU, ep_loc],
    )
    epS.zero_()
    wp.launch(
      _eff_build_dense_uncovered_base,
      dim=(d.nworld, n_u),
      inputs=[
        m.M_rownnz,
        m.M_rowadr,
        m.M_colind,
        m.has_fluid,
        d.M,
        d.efm_diag,
        d.efm_fluid,
        epU,
        ep_loc,
      ],
      outputs=[epS],
    )
    wp.launch(
      _eff_fold_dense_uncovered_tendon,
      dim=(d.nworld, m.ntendon),
      inputs=[
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        d.ten_J,
        d.efm_ts,
        ep_loc,
      ],
      outputs=[epS],
    )
    wp.launch(
      _eff_fold_dense_uncovered_actuator,
      dim=(d.nworld, m.nactuator),
      inputs=[
        d.moment_rownnz,
        d.moment_rowadr,
        d.moment_colind,
        d.actuator_moment,
        d.efm_as,
        ep_loc,
      ],
      outputs=[epS],
    )
    if passive.flex_passive_contact(m) and con_tuple is not None:
      efm_con_dof, efm_con_val, efm_con_scale, _, efm_con_nnz = con_tuple
      wp.launch(
        _eff_fold_dense_uncovered_contact,
        dim=d.naconmax,
        inputs=[
          d.contact.worldid,
          d.nacon,
          efm_con_dof,
          efm_con_val,
          efm_con_scale,
          efm_con_nnz,
          ep_loc,
        ],
        outputs=[epS],
      )
    wp.launch(
      _eff_fold_dense_uncovered_efc,
      dim=(d.nworld, d.njmax),
      inputs=[
        m.is_sparse,
        d.nefc,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.efc.D,
        n_u,
        epU,
        ep_loc,
      ],
      outputs=[epS],
    )
    wp.launch(
      _eff_pad_and_clamp_dense_uncovered,
      dim=(d.nworld, n_u_pad),
      inputs=[n_u],
      outputs=[epS],
    )
    if n_u_pad <= EFF_UNCOVERED_INV_MAX and (n_u <= 32 or epS_factor is not None):
      factored = n_u > 32
      if factored:
        epS_rhs.zero_()
        wp.launch_tiled(
          _eff_cholesky_factor_blocked(TILE_SIZE_JTDAJ_DENSE, n_u_pad),
          dim=d.nworld,
          inputs=[epS, epS_rhs.reshape(shape=(d.nworld, n_u_pad, 1))],
          outputs=[epS_factor, epS_sol.reshape(shape=(d.nworld, n_u_pad, 1))],
          block_dim=m.block_dim.update_gradient_cholesky_blocked,
        )
      wp.launch_tiled(
        _eff_uncovered_inverse(n_u_pad, factored),
        dim=d.nworld,
        inputs=[epS_factor if factored else epS],
        outputs=[epS],
        block_dim=m.block_dim.update_gradient_cholesky,
      )
    elif n_u <= 32:
      wp.launch_tiled(
        _eff_cholesky_factor(n_u),
        dim=d.nworld,
        inputs=[],
        outputs=[epS],
        block_dim=m.block_dim.update_gradient_cholesky,
      )
    elif epS_factor is not None and epS_rhs is not None and epS_sol is not None:
      epS_rhs.zero_()
      wp.launch_tiled(
        _eff_cholesky_factor_blocked(TILE_SIZE_JTDAJ_DENSE, n_u_pad),
        dim=d.nworld,
        inputs=[epS, epS_rhs.reshape(shape=(d.nworld, n_u_pad, 1))],
        outputs=[epS_factor, epS_sol.reshape(shape=(d.nworld, n_u_pad, 1))],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )

  return epL


# MuJoCo C EFF_CHOL_PIVOT: pivots below this fraction of the largest diagonal are rank deficient
EFF_CHOL_PIVOT = 1.0e-12
# vertex pair key stride: covered blocks have at most 8192 vertices (Model.efm_chol)
EFF_CHOL_KEY = 8192
# dense root: the top levels of the elimination tree (at most ROOT_MAX vertices, used from ROOT_MIN)
# are factored as one dense block in ROOT_TILE x ROOT_TILE tiles
EFF_CHOL_ROOT_MAX = 448
EFF_CHOL_ROOT_MIN = 8
# bytes of D and Dinv over all worlds, sized to stay resident in a large L2 (96 MB on RTX 6000 Ada)
EFF_CHOL_ROOT_BYTES = 80 * 1024 * 1024
EFF_CHOL_ROOT_TILE = 32
# dense root solve: rows split into chunks of this many columns
EFF_CHOL_ROOT_CHUNK = 32


@wp.func_native(snippet="WP_TILE_SYNC();")
def _eff_chol_sync(): ...


def eff_chol_active(m: Model) -> bool:
  """Whether the covered flex block uses the Cholesky preconditioner (MuJoCo C effCholApplies)."""
  return bool(m.efm_chol and m.opt.integrator == IntegratorType.DISCRETE and (m.opt.enableflags & EnableBit.IPC))


def create_eff_chol_context(m: Model, nworld: int, njmax: int) -> EffCholContext:
  """Allocates the covered-block Cholesky preconditioner workspace.

  The per-solve pattern grows with the vertex pairs the folded terms couple, so the buffers scale
  with njmax: room for six pairs per constraint row, and a factor of four times the static factor's
  blocks plus two blocks per pair. A world that outgrows them reports OverflowType.PRECONDITIONER
  and leaves the preconditioner on its 3x3 blocks for that solve.
  """
  nb = m.nefmdof
  nxmax = max(6 * njmax, 1)
  namax = m.nefmcholadj // 2 + nxmax
  nlmax = 4 * m.nefmcholL + 2 * nxmax
  tile = EFF_CHOL_ROOT_TILE
  # dense root size: the largest whose D and Dinv over all worlds stay within the L2-sized budget
  rmax = min(nb, EFF_CHOL_ROOT_MAX, int((EFF_CHOL_ROOT_BYTES / (8 * max(nworld, 1))) ** 0.5) // 3)
  nrd = (3 * rmax + tile - 1) // tile * tile
  return EffCholContext(
    Ld=wp.empty((nworld, nb), dtype=wp.mat33),
    Lo=wp.empty((nworld, nlmax), dtype=wp.mat33),
    Sinv=wp.empty((nworld, nb), dtype=wp.mat33),
    colptr=wp.empty((nworld, nb + 1), dtype=int),
    rowind=wp.empty((nworld, nlmax), dtype=int),
    lvlptr=wp.empty((nworld, nb + 1), dtype=int),
    lvlcol=wp.empty((nworld, nb), dtype=int),
    pairptr=wp.empty((nworld, nb + 1), dtype=int),
    entptr=wp.empty((nworld, nb + 1), dtype=int),
    nlevel=wp.empty(nworld, dtype=int),
    lvlent=wp.empty((nworld, nb + 1), dtype=int),
    ent=wp.empty((nworld, nlmax), dtype=wp.vec3i),
    xcount=wp.empty(nworld, dtype=int),
    xpair=wp.empty((nworld, nxmax), dtype=wp.vec2i),
    xhash=wp.empty((nworld, 1 << (2 * nxmax - 1).bit_length()), dtype=int),
    aptr=wp.empty((nworld, nb + 1), dtype=int),
    aend=wp.empty((nworld, nb), dtype=int),
    adj=wp.empty((nworld, namax), dtype=int),
    adjpost=wp.empty((nworld, namax), dtype=int),
    parent=wp.empty((nworld, nb), dtype=int),
    first=wp.empty((nworld, nb), dtype=int),
    post=wp.empty((nworld, nb), dtype=int),
    level=wp.empty((nworld, nb), dtype=int),
    cnt=wp.empty((nworld, nb), dtype=int),
    cur=wp.empty((nworld, nb), dtype=int),
    rootlv=wp.empty(nworld, dtype=int),
    nroot=wp.empty(nworld, dtype=int),
    rpos=wp.empty((nworld, nb), dtype=int),
    ridx=wp.empty((nworld, nrd), dtype=int),
    D=wp.empty((nworld, nrd, nrd), dtype=float),
    Dinv=wp.empty((nworld, nrd, nrd), dtype=float),
    mindiag=wp.empty(nworld, dtype=float),
    rvec=wp.empty((nworld, nrd), dtype=float),
    y=wp.empty((nworld, 1, 3 * nb), dtype=float),
    acc=wp.empty((nworld, 3 * nb), dtype=float),
    status=wp.empty(nworld, dtype=bool),
  )


@wp.kernel
def _eff_chol_rank1_tendon(
  # Model:
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  efm_dofid: wp.array[int],
  efm_dofblk: wp.array[int],
  efm_chol_rank: wp.array[int],
  efm_chol_adjptr: wp.array[int],
  efm_chol_adj: wp.array[int],
  # Data in:
  ten_J_in: wp.array2d[float],
  efm_ts_in: wp.array2d[float],
  # In:
  fold: bool,
  colptr_in: wp.array2d[int],
  rowind_in: wp.array2d[int],
  status_in: wp.array[bool],
  # Out:
  xcount_out: wp.array[int],
  xpair_out: wp.array2d[wp.vec2i],
  xhash_out: wp.array2d[int],
  Ld_out: wp.array2d[wp.mat33],
  Lo_out: wp.array2d[wp.mat33],
):
  worldid, t = wp.tid()
  if fold and not status_in[worldid]:
    return
  s = efm_ts_in[worldid, t]
  if s <= 0.0:
    return
  adr = ten_J_rowadr[t]
  nnz = ten_J_rownnz[t]
  if fold:
    _eff_chol_row_fold(
      efm_dofid,
      efm_dofblk,
      efm_chol_rank,
      worldid,
      s,
      ten_J_colind,
      ten_J_in[worldid],
      adr,
      nnz,
      False,
      colptr_in[worldid],
      rowind_in[worldid],
      Ld_out,
      Lo_out,
    )
  else:
    _eff_chol_row_pairs(
      efm_dofblk,
      efm_chol_adjptr,
      efm_chol_adj,
      worldid,
      ten_J_colind,
      ten_J_in[worldid],
      adr,
      nnz,
      False,
      xcount_out,
      xpair_out,
      xhash_out,
    )


@wp.kernel
def _eff_chol_rank1_actuator(
  # Model:
  efm_dofid: wp.array[int],
  efm_dofblk: wp.array[int],
  efm_chol_rank: wp.array[int],
  efm_chol_adjptr: wp.array[int],
  efm_chol_adj: wp.array[int],
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  efm_as_in: wp.array2d[float],
  # In:
  fold: bool,
  colptr_in: wp.array2d[int],
  rowind_in: wp.array2d[int],
  status_in: wp.array[bool],
  # Out:
  xcount_out: wp.array[int],
  xpair_out: wp.array2d[wp.vec2i],
  xhash_out: wp.array2d[int],
  Ld_out: wp.array2d[wp.mat33],
  Lo_out: wp.array2d[wp.mat33],
):
  worldid, u = wp.tid()
  if fold and not status_in[worldid]:
    return
  s = efm_as_in[worldid, u]
  if s <= 0.0:
    return
  adr = moment_rowadr_in[worldid, u]
  nnz = moment_rownnz_in[worldid, u]
  if fold:
    _eff_chol_row_fold(
      efm_dofid,
      efm_dofblk,
      efm_chol_rank,
      worldid,
      s,
      moment_colind_in[worldid],
      actuator_moment_in[worldid],
      adr,
      nnz,
      False,
      colptr_in[worldid],
      rowind_in[worldid],
      Ld_out,
      Lo_out,
    )
  else:
    _eff_chol_row_pairs(
      efm_dofblk,
      efm_chol_adjptr,
      efm_chol_adj,
      worldid,
      moment_colind_in[worldid],
      actuator_moment_in[worldid],
      adr,
      nnz,
      False,
      xcount_out,
      xpair_out,
      xhash_out,
    )


@wp.kernel
def _eff_chol_efc_sparse(
  # Model:
  efm_dofid: wp.array[int],
  efm_dofblk: wp.array[int],
  efm_chol_rank: wp.array[int],
  efm_chol_adjptr: wp.array[int],
  efm_chol_adj: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  # In:
  fold: bool,
  colptr_in: wp.array2d[int],
  rowind_in: wp.array2d[int],
  status_in: wp.array[bool],
  # Out:
  xcount_out: wp.array[int],
  xpair_out: wp.array2d[wp.vec2i],
  xhash_out: wp.array2d[int],
  Ld_out: wp.array2d[wp.mat33],
  Lo_out: wp.array2d[wp.mat33],
):
  worldid, r = wp.tid()
  if fold and not status_in[worldid]:
    return
  if r >= nefc_in[worldid]:
    return
  D = efc_D_in[worldid, r]
  if D <= 0.0:
    return
  adr = efc_J_rowadr_in[worldid, r]
  nnz = efc_J_rownnz_in[worldid, r]
  if fold:
    _eff_chol_row_fold(
      efm_dofid,
      efm_dofblk,
      efm_chol_rank,
      worldid,
      D,
      efc_J_colind_in[worldid, 0],
      efc_J_in[worldid, 0],
      adr,
      nnz,
      False,
      colptr_in[worldid],
      rowind_in[worldid],
      Ld_out,
      Lo_out,
    )
  else:
    _eff_chol_row_pairs(
      efm_dofblk,
      efm_chol_adjptr,
      efm_chol_adj,
      worldid,
      efc_J_colind_in[worldid, 0],
      efc_J_in[worldid, 0],
      adr,
      nnz,
      False,
      xcount_out,
      xpair_out,
      xhash_out,
    )


@wp.kernel
def _eff_chol_efc_dense(
  # Model:
  nv: int,
  efm_dofid: wp.array[int],
  efm_dofblk: wp.array[int],
  efm_chol_rank: wp.array[int],
  efm_chol_adjptr: wp.array[int],
  efm_chol_adj: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  # In:
  fold: bool,
  colptr_in: wp.array2d[int],
  rowind_in: wp.array2d[int],
  status_in: wp.array[bool],
  # Out:
  xcount_out: wp.array[int],
  xpair_out: wp.array2d[wp.vec2i],
  xhash_out: wp.array2d[int],
  Ld_out: wp.array2d[wp.mat33],
  Lo_out: wp.array2d[wp.mat33],
):
  worldid, r = wp.tid()
  if fold and not status_in[worldid]:
    return
  if r >= nefc_in[worldid]:
    return
  D = efc_D_in[worldid, r]
  if D <= 0.0:
    return
  if fold:
    _eff_chol_row_fold(
      efm_dofid,
      efm_dofblk,
      efm_chol_rank,
      worldid,
      D,
      efm_dofid,
      efc_J_in[worldid, r],
      0,
      nv,
      True,
      colptr_in[worldid],
      rowind_in[worldid],
      Ld_out,
      Lo_out,
    )
  else:
    _eff_chol_row_pairs(
      efm_dofblk,
      efm_chol_adjptr,
      efm_chol_adj,
      worldid,
      efm_dofid,
      efc_J_in[worldid, r],
      0,
      nv,
      True,
      xcount_out,
      xpair_out,
      xhash_out,
    )


@wp.kernel
def _eff_chol_adj_count(
  # Model:
  efm_chol_rank: wp.array[int],
  efm_chol_vert: wp.array[int],
  efm_chol_adjptr: wp.array[int],
  efm_chol_adj: wp.array[int],
  # Out:
  cnt_out: wp.array2d[int],
):
  """Counts the flex edge neighbors earlier in the elimination order of the vertex at rank i."""
  worldid, i = wp.tid()
  k = efm_chol_vert[i]
  c = int(0)
  for p in range(efm_chol_adjptr[k], efm_chol_adjptr[k + 1]):
    if efm_chol_rank[efm_chol_adj[p]] < i:
      c += 1
  cnt_out[worldid, i] = c


@wp.kernel
def _eff_chol_adj_count_extra(
  # Model:
  efm_chol_rank: wp.array[int],
  # In:
  xcount_in: wp.array[int],
  xpair_in: wp.array2d[wp.vec2i],
  # Out:
  cnt_out: wp.array2d[int],
):
  """Counts each extra pair on the row of its later vertex in the elimination order."""
  worldid, x = wp.tid()
  if x >= xcount_in[worldid]:
    return
  pr = xpair_in[worldid, x]
  wp.atomic_add(cnt_out, worldid, wp.max(efm_chol_rank[pr[0]], efm_chol_rank[pr[1]]), 1)


@wp.kernel(module="unique", enable_backward=False)
def _eff_chol_adj_scan(
  # Model:
  nefmdof: int,
  # In:
  cnt_in: wp.array2d[int],
  # Out:
  aptr_out: wp.array2d[int],
):
  """Row starts of the lower adjacency, one block per world."""
  worldid, tid = wp.tid()
  nthread = wp.block_dim()
  chunk = (nefmdof + nthread - 1) // nthread
  s0 = wp.min(tid * chunk, nefmdof)
  s1 = wp.min(s0 + chunk, nefmdof)
  local = int(0)
  for i in range(s0, s1):
    local += cnt_in[worldid, i]
  run = _eff_chol_block_scan(local)
  for i in range(s0, s1):
    aptr_out[worldid, i] = run
    run += cnt_in[worldid, i]
  if tid == nthread - 1:
    aptr_out[worldid, nefmdof] = run


@wp.kernel
def _eff_chol_adj_fill(
  # Model:
  efm_chol_rank: wp.array[int],
  efm_chol_vert: wp.array[int],
  efm_chol_adjptr: wp.array[int],
  efm_chol_adj: wp.array[int],
  # In:
  aptr_in: wp.array2d[int],
  # Out:
  adj_out: wp.array2d[int],
  cur_out: wp.array2d[int],
):
  """Writes the earlier flex edge neighbors of the vertex at rank i, by rank."""
  worldid, i = wp.tid()
  k = efm_chol_vert[i]
  pos = aptr_in[worldid, i]
  for p in range(efm_chol_adjptr[k], efm_chol_adjptr[k + 1]):
    r = efm_chol_rank[efm_chol_adj[p]]
    if r < i:
      adj_out[worldid, pos] = r
      pos += 1
  cur_out[worldid, i] = pos


@wp.kernel
def _eff_chol_adj_fill_extra(
  # Model:
  efm_chol_rank: wp.array[int],
  # In:
  xcount_in: wp.array[int],
  xpair_in: wp.array2d[wp.vec2i],
  # Out:
  adj_out: wp.array2d[int],
  cur_out: wp.array2d[int],
):
  """Appends each extra pair to the row of its later vertex in the elimination order."""
  worldid, x = wp.tid()
  if x >= xcount_in[worldid]:
    return
  pr = xpair_in[worldid, x]
  ra = efm_chol_rank[pr[0]]
  rb = efm_chol_rank[pr[1]]
  slot = wp.atomic_add(cur_out, worldid, wp.max(ra, rb), 1)
  adj_out[worldid, slot] = wp.min(ra, rb)


@wp.kernel
def _eff_chol_adj_sort(
  # In:
  aptr_in: wp.array2d[int],
  # Out:
  aend_out: wp.array2d[int],
  adj_out: wp.array2d[int],
):
  """Sorts and deduplicates the lower adjacency of the vertex at rank i."""
  worldid, i = wp.tid()
  start = aptr_in[worldid, i]
  end = aptr_in[worldid, i + 1]
  adj = adj_out[worldid]
  for p in range(start + 1, end):
    v = adj[p]
    q = p
    while q > start and adj[q - 1] > v:
      adj[q] = adj[q - 1]
      q -= 1
    adj[q] = v
  wpos = start
  for p in range(start, end):
    if p == start or adj[p] != adj[p - 1]:
      adj[wpos] = adj[p]
      wpos += 1
  aend_out[worldid, i] = wpos


@wp.kernel(module="unique", enable_backward=False)
def _eff_chol_etree(
  # Model:
  nefmdof: int,
  # In:
  aptr_in: wp.array2d[int],
  aend_in: wp.array2d[int],
  adj_in: wp.array2d[int],
  # Out:
  parent_out: wp.array2d[int],
  first_out: wp.array2d[int],
  post_out: wp.array2d[int],
  level_out: wp.array2d[int],
  cnt_out: wp.array2d[int],
  cur_out: wp.array2d[int],
  lvlptr_out: wp.array2d[int],
  lvlcol_out: wp.array2d[int],
  nlevel_out: wp.array[int],
  rootlv_out: wp.array[int],
  nroot_out: wp.array[int],
  rpos_out: wp.array2d[int],
  ridx_out: wp.array2d[int],
):
  """Elimination tree, postorder intervals and levels of the lower adjacency, one block per world.

  The tree uses Liu's algorithm with path compression; the threads climb from a row's neighbors
  together. Each vertex's subtree occupies the postorder interval [first, post], and the levels
  group the columns that factor together. The top levels become the dense root (rpos: position in
  the root, or -1).
  """
  worldid, tid = wp.tid()
  nthread = wp.block_dim()
  nb = nefmdof
  parent = parent_out[worldid]
  first = first_out[worldid]
  post = post_out[worldid]
  level = level_out[worldid]
  cnt = cnt_out[worldid]
  cur = cur_out[worldid]
  aptr = aptr_in[worldid]
  aend = aend_in[worldid]
  adj = adj_in[worldid]

  rpos = rpos_out[worldid]
  lvlptr = lvlptr_out[worldid]
  lvlcol = lvlcol_out[worldid]

  # elimination tree: cur holds the compressed ancestors; concurrent climbs only write i into cur
  # and parent, so a thread reading i stops where another thread's climb continues
  for j in range(tid, nb, nthread):
    parent[j] = -1
    cur[j] = -1
    first[j] = 1
    level[j] = 0
    rpos[j] = -1
  _eff_chol_sync()
  for i in range(nb):
    for p in range(aptr[i] + tid, aend[i], nthread):
      r = adj[p]
      while r >= 0:
        a = cur[r]
        cur[r] = i
        if a < 0:
          parent[r] = i
          r = -1
        elif a == i:
          r = -1
        else:
          r = a
    _eff_chol_sync()

  # subtree sizes (in first) and levels: children precede their parents
  if tid == 0:
    for j in range(nb):
      p = parent[j]
      if p >= 0:
        first[p] += first[j]
        level[p] = wp.max(level[p], level[j] + 1)
  _eff_chol_sync()

  # columns grouped by level, cnt is the level cursor
  nlevel_local = int(0)
  for j in range(tid, nb, nthread):
    nlevel_local = wp.max(nlevel_local, level[j] + 1)
  nlevel = wp.tile_reduce(wp.max, wp.tile(nlevel_local))[0]
  for lv in range(tid, nlevel + 1, nthread):
    lvlptr[lv] = 0
  _eff_chol_sync()
  for j in range(tid, nb, nthread):
    wp.atomic_add(lvlptr, level[j] + 1, 1)
  _eff_chol_sync()

  if tid == 0:
    for lv in range(nlevel):
      lvlptr[lv + 1] += lvlptr[lv]
      cnt[lv] = lvlptr[lv]

    # postorder intervals: parents precede their children in reverse order, cur is the child cursor
    root = int(0)
    for jr in range(nb):
      j = nb - 1 - jr
      size = first[j]
      p = parent[j]
      start = root
      if p < 0:
        root += size
      else:
        start = cur[p]
        cur[p] = start + size
      first[j] = start
      post[j] = start + size - 1
      cur[j] = start
  _eff_chol_sync()

  # stable scatter by level, a chunk of columns at a time: rank among the chunk's columns on a level
  for c0 in range(0, nb, nthread):
    j = c0 + tid
    lv = int(-1)
    if j < nb:
      lv = level[j]
    rank = int(0)
    last = bool(True)
    for k in range(c0, wp.min(c0 + nthread, nb)):
      if level[k] == lv:
        if k < j:
          rank += 1
        elif k > j:
          last = False
    if j < nb:
      lvlcol[cnt[lv] + rank] = j
    _eff_chol_sync()
    if j < nb and last:
      cnt[lv] += rank + 1
    _eff_chol_sync()
  for j in range(tid, nb, nthread):
    cnt[j] = 0

  # dense root: the top levels, as many as fit; closed under ancestors, which sit on higher levels
  rmax = wp.min(EFF_CHOL_ROOT_MAX, ridx_out.shape[1] // 3)
  r0 = nlevel
  while r0 > 0 and lvlptr[nlevel] - lvlptr[r0 - 1] <= rmax:
    r0 -= 1
  nroot = lvlptr[nlevel] - lvlptr[r0]
  if nroot < EFF_CHOL_ROOT_MIN:
    nroot = 0
    r0 = nlevel
  for i in range(tid, nroot, nthread):
    c = lvlcol[lvlptr[r0] + i]
    rpos[c] = i
    for a in range(3):
      ridx_out[worldid, 3 * i + a] = 3 * c + a
  if tid == 0:
    nlevel_out[worldid] = nlevel
    rootlv_out[worldid] = r0
    nroot_out[worldid] = nroot


@wp.kernel
def _eff_chol_row_count(
  # In:
  aptr_in: wp.array2d[int],
  aend_in: wp.array2d[int],
  adj_in: wp.array2d[int],
  parent_in: wp.array2d[int],
  first_in: wp.array2d[int],
  post_in: wp.array2d[int],
  # Out:
  adjpost_out: wp.array2d[int],
  cnt_out: wp.array2d[int],
):
  """Sorts the lower adjacency of row i by postorder and counts its factor entries per column."""
  worldid, i = wp.tid()
  start = aptr_in[worldid, i]
  end = aend_in[worldid, i]
  post = post_in[worldid]
  adjpost = adjpost_out[worldid]
  for p in range(start, end):
    v = adj_in[worldid, p]
    key = post[v]
    q = p
    while q > start and post[adjpost[q - 1]] > key:
      adjpost[q] = adjpost[q - 1]
      q -= 1
    adjpost[q] = v
  prev = int(-1)
  for p in range(start, end):
    k = adjpost[p]
    _eff_chol_row_walk(i, k, prev, False, parent_in[worldid], first_in[worldid], post, cnt_out[worldid], cnt_out[worldid])
    prev = k


@cache_kernel
def _eff_chol_col_scan(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nefmdof: int,
    # Data in:
    njmax_in: int,
    # In:
    xcount_in: wp.array[int],
    xpair_in: wp.array2d[wp.vec2i],
    cnt_in: wp.array2d[int],
    lvlcol_in: wp.array2d[int],
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    colptr_out: wp.array2d[int],
    rowind_out: wp.array2d[int],
    pairptr_out: wp.array2d[int],
    entptr_out: wp.array2d[int],
    cur_out: wp.array2d[int],
    status_out: wp.array[bool],
  ):
    """Column starts of the factor and the per-level entry and update prefixes, one block per world.

    A world whose extra pairs or factor entries outgrow the buffers reports overflow and leaves the
    preconditioner on its 3x3 blocks (status False).
    """
    worldid, tid = wp.tid()
    nthread = wp.block_dim()
    nb = nefmdof
    chunk = (nb + nthread - 1) // nthread
    s0 = wp.min(tid * chunk, nb)
    s1 = wp.min(s0 + chunk, nb)
    cnt = cnt_in[worldid]
    lvlcol = lvlcol_in[worldid]

    nnz_local = int(0)
    ent_local = int(0)
    pair_local = int(0)
    for i in range(s0, s1):
      nnz_local += cnt[i]
      ln = cnt[lvlcol[i]]
      ent_local += ln
      pair_local += ln * (ln + 1) // 2
    nnz = _eff_chol_block_scan(nnz_local)
    ent = _eff_chol_block_scan(ent_local)
    pair = _eff_chol_block_scan(pair_local)
    total = _eff_chol_block_sum(nnz_local)
    for i in range(s0, s1):
      colptr_out[worldid, i] = nnz
      cur_out[worldid, i] = nnz
      nnz += cnt[i]
      entptr_out[worldid, i] = ent
      pairptr_out[worldid, i] = pair
      ln = cnt[lvlcol[i]]
      ent += ln
      pair += ln * (ln + 1) // 2
    if tid == nthread - 1:
      colptr_out[worldid, nb] = nnz
      entptr_out[worldid, nb] = ent
      pairptr_out[worldid, nb] = pair

    if tid == 0:
      xcount = xcount_in[worldid]
      ok = xcount <= xpair_in.shape[1] and total <= rowind_out.shape[1]
      status_out[worldid] = ok
      if not ok:
        if wp.static(bool(warn_overflow & OverflowType.PRECONDITIONER)):
          wp.printf(
            "flex Cholesky preconditioner overflow (%d vertex pairs, %d factor blocks) - please increase njmax beyond %d\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.PRECONDITIONER (or = 0 for all)\n",
            xcount,
            total,
            njmax_in,
          )
        overflow_out[worldid] = overflow_out[worldid] | wp.static(OverflowType.PRECONDITIONER)

  return kernel


@wp.kernel
def _eff_chol_row_fill(
  # In:
  aptr_in: wp.array2d[int],
  aend_in: wp.array2d[int],
  adjpost_in: wp.array2d[int],
  parent_in: wp.array2d[int],
  first_in: wp.array2d[int],
  post_in: wp.array2d[int],
  status_in: wp.array[bool],
  # Out:
  cur_out: wp.array2d[int],
  rowind_out: wp.array2d[int],
):
  """Writes row i into the factor columns it reaches."""
  worldid, i = wp.tid()
  if not status_in[worldid]:
    return
  prev = int(-1)
  for p in range(aptr_in[worldid, i], aend_in[worldid, i]):
    k = adjpost_in[worldid, p]
    _eff_chol_row_walk(
      i, k, prev, True, parent_in[worldid], first_in[worldid], post_in[worldid], cur_out[worldid], rowind_out[worldid]
    )
    prev = k


@wp.kernel
def _eff_chol_col_sort(
  # In:
  colptr_in: wp.array2d[int],
  status_in: wp.array[bool],
  # Out:
  rowind_out: wp.array2d[int],
  Lo_out: wp.array2d[wp.mat33],
):
  """Sorts the rows of factor column j and clears its blocks."""
  worldid, j = wp.tid()
  if not status_in[worldid]:
    return
  start = colptr_in[worldid, j]
  end = colptr_in[worldid, j + 1]
  rowind = rowind_out[worldid]
  # Shell sort: columns near the root are long
  gap = int(1)
  while 3 * gap + 1 < end - start:
    gap = 3 * gap + 1
  while gap > 0:
    for p in range(start + gap, end):
      v = rowind[p]
      q = p
      while q >= start + gap and rowind[q - gap] > v:
        rowind[q] = rowind[q - gap]
        q -= gap
      rowind[q] = v
    gap = gap // 3
  for p in range(start, end):
    Lo_out[worldid, p] = wp.mat33(0.0)


@wp.kernel
def _eff_chol_level_entries(
  # In:
  colptr_in: wp.array2d[int],
  rowind_in: wp.array2d[int],
  lvlptr_in: wp.array2d[int],
  lvlcol_in: wp.array2d[int],
  entptr_in: wp.array2d[int],
  level_in: wp.array2d[int],
  nlevel_in: wp.array[int],
  status_in: wp.array[bool],
  # Out:
  lvlent_out: wp.array2d[int],
  ent_out: wp.array2d[wp.vec3i],
):
  """Lists the offdiagonal blocks in level order as (column, block, row), and each level's start."""
  worldid, c = wp.tid()
  if not status_in[worldid]:
    return
  j = lvlcol_in[worldid, c]
  lv = level_in[worldid, j]
  e0 = entptr_in[worldid, c]
  if lvlptr_in[worldid, lv] == c:
    lvlent_out[worldid, lv] = e0
  if c == 0:
    nlevel = nlevel_in[worldid]
    lvlent_out[worldid, nlevel] = entptr_in[worldid, lvlptr_in[worldid, nlevel]]
  base = colptr_in[worldid, j]
  for q in range(colptr_in[worldid, j + 1] - base):
    ent_out[worldid, e0 + q] = wp.vec3i(j, base + q, rowind_in[worldid, base + q])


@wp.kernel
def _eff_chol_fill_mk(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  efm_K_rownnz: wp.array[int],
  efm_K_rowadr: wp.array[int],
  efm_K_colind: wp.array[int],
  efm_dofid: wp.array[int],
  efm_dofblk: wp.array[int],
  efm_chol_rank: wp.array[int],
  # Data in:
  M_in: wp.array2d[float],
  efm_K_val_in: wp.array2d[float],
  # In:
  colptr_in: wp.array2d[int],
  rowind_in: wp.array2d[int],
  aptr_in: wp.array2d[int],
  aend_in: wp.array2d[int],
  adj_in: wp.array2d[int],
  status_in: wp.array[bool],
  # Out:
  Ld_out: wp.array2d[wp.mat33],
  Lo_out: wp.array2d[wp.mat33],
):
  """Adds row a of vertex k's M and K blocks to the factor pattern.

  A K block between vertices that are neither joined by a flex edge nor coupled by a folded term is
  dropped, its Frobenius norm added to both diagonal blocks by the vertex earlier in dof order.
  """
  worldid, k, a = wp.tid()
  if not status_in[worldid]:
    return
  i0 = efm_dofid[k]
  i = i0 + a
  rk = efm_chol_rank[k]
  aptr = aptr_in[worldid]
  aend = aend_in[worldid]
  adj = adj_in[worldid]
  diag = wp.mat33(0.0)
  dropped = float(0.0)
  kadr = efm_K_rowadr[i]
  for j in range(kadr, kadr + efm_K_rownnz[i]):
    col = efm_K_colind[j]
    q = efm_dofblk[col]
    if q < 0:
      continue
    v = efm_K_val_in[worldid, j]
    q0 = efm_dofid[q]
    rq = efm_chol_rank[q]
    if q == k:
      diag[a, col - q0] = diag[a, col - q0] + v
    elif _eff_chol_allowed(aptr, aend, adj, rk, rq):
      if rk > rq:
        blk = _eff_chol_block(colptr_in[worldid], rowind_in[worldid], rk, rq)
        if blk >= 0:
          wp.atomic_add(Lo_out[worldid], blk, _eff_chol_unit(a, col - q0, v))
    elif q > k:
      # the block's first entry in the earliest row holding it computes the norm
      if j > kadr and efm_K_colind[j - 1] >= q0:
        continue
      seen = bool(False)
      for a2 in range(a):
        adr2 = efm_K_rowadr[i0 + a2]
        end2 = adr2 + efm_K_rownnz[i0 + a2]
        lb = _eff_chol_lower_bound(adr2, end2 - adr2, efm_K_colind, q0)
        if lb < end2 and efm_K_colind[lb] < q0 + 3:
          seen = True
      if seen:
        continue
      f2 = float(0.0)
      for a2 in range(3):
        adr2 = efm_K_rowadr[i0 + a2]
        end2 = adr2 + efm_K_rownnz[i0 + a2]
        lb = _eff_chol_lower_bound(adr2, end2 - adr2, efm_K_colind, q0)
        while lb < end2 and efm_K_colind[lb] < q0 + 3:
          kv = efm_K_val_in[worldid, lb]
          f2 += kv * kv
          lb += 1
      f = wp.sqrt(f2)
      dropped += f
      wp.atomic_add(Ld_out[worldid], rq, wp.mat33(f, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, f))
  madr = M_rowadr[i]
  for j in range(madr, madr + M_rownnz[i]):
    col = M_colind[j]
    if col >= i0:
      diag[a, col - i0] = diag[a, col - i0] + M_in[worldid, j]
      if col != i:
        diag[col - i0, a] = diag[col - i0, a] + M_in[worldid, j]
  diag += wp.mat33(dropped, 0.0, 0.0, 0.0, dropped, 0.0, 0.0, 0.0, dropped)
  wp.atomic_add(Ld_out[worldid], rk, diag)


@wp.kernel(module="unique", enable_backward=False)
def _eff_chol_factor_sparse(
  # Model:
  nefmdof: int,
  # In:
  colptr_in: wp.array2d[int],
  rowind_in: wp.array2d[int],
  lvlptr_in: wp.array2d[int],
  lvlcol_in: wp.array2d[int],
  pairptr_in: wp.array2d[int],
  lvlent_in: wp.array2d[int],
  ent_in: wp.array2d[wp.vec3i],
  rootlv_in: wp.array[int],
  rpos_in: wp.array2d[int],
  # Out:
  Ld_out: wp.array2d[wp.mat33],
  Lo_out: wp.array2d[wp.mat33],
  Sinv_out: wp.array2d[wp.mat33],
  D_out: wp.array3d[float],
  mindiag_out: wp.array[float],
  status_out: wp.array[bool],
):
  """Blocked right-looking factorization L S L' by elimination-tree level, one block per world.

  L has identity diagonal blocks and S is block diagonal (the Schur complement pivots), so the
  solve needs no 3x3 triangular solves. Each level stores its pivots' inverses in Sinv and scales
  its offdiagonal blocks L_ij = A_ij S_jj^-1, each thread inverting the pivot it needs, then
  applies its rank-1 updates L_ia S_jj L_ib' to Ld and Lo. Both phases are flattened across the
  level's columns so narrow levels near the root still use the whole block. The levels below the
  dense root run here; their updates of the root go to its dense Schur complement D.
  """
  worldid, tid = wp.tid()
  ok = status_out[worldid]
  # every thread reads status before any pivot can clear it
  _eff_chol_sync()
  if not ok:
    return
  nthread = wp.block_dim()
  colptr = colptr_in[worldid]
  rowind = rowind_in[worldid]
  lvlptr = lvlptr_in[worldid]
  lvlcol = lvlcol_in[worldid]
  pairptr = pairptr_in[worldid]
  lvlent = lvlent_in[worldid]
  ent = ent_in[worldid]
  rpos = rpos_in[worldid]
  D = D_out[worldid]
  local = float(0.0)
  for j in range(tid, nefmdof, nthread):
    Ajj = Ld_out[worldid, j]
    local = wp.max(local, wp.max(Ajj[0, 0], wp.max(Ajj[1, 1], Ajj[2, 2])))
  maxdiag = wp.tile_max(wp.tile(local))[0]
  mindiag = wp.max(MJ_MINVAL, EFF_CHOL_PIVOT * maxdiag)
  if tid == 0:
    mindiag_out[worldid] = mindiag
  for lv in range(rootlv_in[worldid]):
    cstart = lvlptr[lv]
    cend = lvlptr[lv + 1]
    # the level's columns have received every update from their descendants
    for c in range(cstart + tid, cend, nthread):
      j = lvlcol[c]
      A = Ld_out[worldid, j]
      if not _eff_chol_diag_ok(A, mindiag):
        status_out[worldid] = False
      Sinv_out[worldid, j] = _eff_chol_sinv(A, mindiag)
    # offdiagonal blocks L_ij = A_ij S_jj^-1
    for u in range(lvlent[lv] + tid, lvlent[lv + 1], nthread):
      e = ent[u]
      j = e[0]
      p = e[1]
      Lo_out[worldid, p] = Lo_out[worldid, p] * _eff_chol_sinv(Ld_out[worldid, j], mindiag)
    _eff_chol_sync()
    # rank-1 updates of the ancestors L_ia S_jj L_ib': pair u of the level -> (column, row a, row b)
    for u in range(pairptr[cstart] + tid, pairptr[cend], nthread):
      c = _eff_chol_level_col(pairptr, cstart, cend, u)
      j = lvlcol[c]
      p = u - pairptr[c]
      a = int((wp.sqrt(8.0 * float(p) + 1.0) - 1.0) * 0.5)
      while (a + 1) * (a + 2) // 2 <= p:
        a += 1
      while a * (a + 1) // 2 > p:
        a -= 1
      b = p - a * (a + 1) // 2
      base = colptr[j]
      ra = rowind[base + a]
      P = Lo_out[worldid, base + a] * Ld_out[worldid, j]
      Q = Lo_out[worldid, base + b]
      PQ = P * wp.transpose(Q)
      rb = rowind[base + b]
      kb = rpos[rb]
      if kb >= 0:
        # rows above a root vertex are root vertices
        ka = rpos[ra]
        for x in range(3):
          for z in range(3):
            wp.atomic_sub(D, 3 * ka + x, 3 * kb + z, PQ[x, z])
      elif a == b:
        wp.atomic_sub(Ld_out[worldid], ra, PQ)
      else:
        blk = _eff_chol_block(colptr, rowind, ra, rb)
        if blk >= 0:
          wp.atomic_sub(Lo_out[worldid], blk, PQ)
    _eff_chol_sync()


@wp.kernel
def _eff_chol_root_gather(
  # In:
  colptr_in: wp.array2d[int],
  rowind_in: wp.array2d[int],
  lvlptr_in: wp.array2d[int],
  lvlcol_in: wp.array2d[int],
  rootlv_in: wp.array[int],
  nroot_in: wp.array[int],
  rpos_in: wp.array2d[int],
  Ld_in: wp.array2d[wp.mat33],
  Lo_in: wp.array2d[wp.mat33],
  status_in: wp.array[bool],
  # Out:
  Sinv_out: wp.array2d[wp.mat33],
  D_out: wp.array3d[float],
  Dinv_out: wp.array3d[float],
):
  """Writes the dense root into D and the identity into Dinv, both zeroed before.

  D receives the root's lower triangle with identity on its padding; Dinv starts the inverse factor.
  """
  worldid, q = wp.tid()
  if not status_in[worldid]:
    return
  nroot = nroot_in[worldid]
  if q >= 3 * nroot:
    # padding keeps the dense tiles positive definite
    D_out[worldid, q, q] = 1.0
    Dinv_out[worldid, q, q] = 1.0
    return
  if q % 3 != 0:
    return
  for a in range(3):
    Dinv_out[worldid, q + a, q + a] = 1.0
  i = q // 3
  c = lvlcol_in[worldid, lvlptr_in[worldid, rootlv_in[worldid]] + i]
  A = Ld_in[worldid, c]
  for a in range(3):
    for b in range(3):
      D_out[worldid, q + a, q + b] = A[a, b]
  # the root's pivots live in its dense factor
  Sinv_out[worldid, c] = wp.identity(3, dtype=float)
  for p in range(colptr_in[worldid, c], colptr_in[worldid, c + 1]):
    k = rpos_in[worldid, rowind_in[worldid, p]]
    G = Lo_in[worldid, p]
    for a in range(3):
      for b in range(3):
        D_out[worldid, 3 * k + a, q + b] = G[a, b]


@wp.kernel(module="unique", enable_backward=False)
def _eff_chol_root_panel(
  # In:
  k: int,
  nroot_in: wp.array[int],
  status_in: wp.array[bool],
  # Out:
  D_out: wp.array3d[float],
  Dinv_out: wp.array3d[float],
):
  """Step k of the dense root's Cholesky and inverse factor, one tile per block.

  Tile t > k is L_tk = D_tk L_kk^-T, tile t <= k is row k of the inverse factor X_kt = L_kk^-1 B_kt;
  each block factors D_kk itself.
  """
  worldid, t, tid = wp.tid()
  ko = k * EFF_CHOL_ROOT_TILE
  to = t * EFF_CHOL_ROOT_TILE
  n = 3 * nroot_in[worldid]
  if not status_in[worldid] or ko >= n or to >= n:
    return
  L = wp.tile_load(D_out[worldid], shape=(EFF_CHOL_ROOT_TILE, EFF_CHOL_ROOT_TILE), offset=(ko, ko), storage="shared")
  wp.tile_cholesky_inplace(L)
  if t > k:
    B = wp.tile_load(D_out[worldid], shape=(EFF_CHOL_ROOT_TILE, EFF_CHOL_ROOT_TILE), offset=(to, ko), storage="shared")
    Xt = wp.tile_lower_solve(L, wp.tile_transpose(B))
    wp.tile_store(D_out[worldid], wp.tile_transpose(Xt), offset=(to, ko))
  else:
    B = wp.tile_load(Dinv_out[worldid], shape=(EFF_CHOL_ROOT_TILE, EFF_CHOL_ROOT_TILE), offset=(ko, to), storage="shared")
    wp.tile_store(Dinv_out[worldid], wp.tile_lower_solve(L, B), offset=(ko, to))


@wp.kernel(module="unique", enable_backward=False)
def _eff_chol_root_update(
  # In:
  k: int,
  nroot_in: wp.array[int],
  status_in: wp.array[bool],
  # Out:
  D_out: wp.array3d[float],
  Dinv_out: wp.array3d[float],
):
  """Updates after step k of the dense root, one tile (i, j) per block.

  Trailing D_ij -= L_ik L_jk' for k < j <= i, inverse rows B_ij -= L_ik X_kj for j <= k < i, and
  tile (k, k) stores L_kk.
  """
  worldid, i, j, tid = wp.tid()
  n = 3 * nroot_in[worldid]
  ko = k * EFF_CHOL_ROOT_TILE
  io = i * EFF_CHOL_ROOT_TILE
  jo = j * EFF_CHOL_ROOT_TILE
  if not status_in[worldid] or io >= n or j > i or i < k:
    return
  if i == k:
    if j == k:
      L = wp.tile_load(D_out[worldid], shape=(EFF_CHOL_ROOT_TILE, EFF_CHOL_ROOT_TILE), offset=(ko, ko), storage="shared")
      wp.tile_cholesky_inplace(L)
      wp.tile_store(D_out[worldid], L, offset=(ko, ko))
    return
  Lik = wp.tile_load(D_out[worldid], shape=(EFF_CHOL_ROOT_TILE, EFF_CHOL_ROOT_TILE), offset=(io, ko), storage="shared")
  if j > k:
    Ljk = wp.tile_load(D_out[worldid], shape=(EFF_CHOL_ROOT_TILE, EFF_CHOL_ROOT_TILE), offset=(jo, ko), storage="shared")
    Aij = wp.tile_load(D_out[worldid], shape=(EFF_CHOL_ROOT_TILE, EFF_CHOL_ROOT_TILE), offset=(io, jo), storage="shared")
    wp.tile_matmul(Lik, wp.tile_transpose(Ljk), Aij, alpha=-1.0)
    wp.tile_store(D_out[worldid], Aij, offset=(io, jo))
  else:
    Xkj = wp.tile_load(Dinv_out[worldid], shape=(EFF_CHOL_ROOT_TILE, EFF_CHOL_ROOT_TILE), offset=(ko, jo), storage="shared")
    Bij = wp.tile_load(Dinv_out[worldid], shape=(EFF_CHOL_ROOT_TILE, EFF_CHOL_ROOT_TILE), offset=(io, jo), storage="shared")
    wp.tile_matmul(Lik, Xkj, Bij, alpha=-1.0)
    wp.tile_store(Dinv_out[worldid], Bij, offset=(io, jo))


@wp.kernel
def _eff_chol_root_check(
  # In:
  nroot_in: wp.array[int],
  mindiag_in: wp.array[float],
  D_in: wp.array3d[float],
  # Out:
  status_out: wp.array[bool],
):
  """Rank-deficient dense root: a pivot L_qq^2 at or below the sparse factor's threshold."""
  worldid, q = wp.tid()
  if q >= 3 * nroot_in[worldid]:
    return
  L = D_in[worldid, q, q]
  if not (L * L > mindiag_in[worldid]):
    status_out[worldid] = False


@wp.kernel
def _eff_chol_root_transpose(
  # In:
  nroot_in: wp.array[int],
  Dinv_in: wp.array3d[float],
  status_in: wp.array[bool],
  # Out:
  D_out: wp.array3d[float],
):
  """Stores the inverse factor transposed in D, so both solve sweeps read it coalesced."""
  worldid, r, c = wp.tid()
  n = 3 * nroot_in[worldid]
  if not status_in[worldid] or r >= n or c > r:
    return
  D_out[worldid, c, r] = Dinv_in[worldid, r, c]


@wp.kernel(module="unique", enable_backward=False)
def _eff_chol_solve_sparse(
  # Model:
  efm_dofid: wp.array[int],
  efm_chol_vert: wp.array[int],
  # Data in:
  efm_L_in: wp.array2d[float],
  # In:
  vec_in: wp.array2d[float],
  epU_in: wp.array[int],
  epS_in: wp.array3d[float],
  lvlptr_in: wp.array2d[int],
  nlevel_in: wp.array[int],
  lvlent_in: wp.array2d[int],
  ent_in: wp.array2d[wp.vec3i],
  rootlv_in: wp.array[int],
  nroot_in: wp.array[int],
  ridx_in: wp.array2d[int],
  Lo_in: wp.array2d[wp.mat33],
  Sinv_in: wp.array2d[wp.mat33],
  status_in: wp.array[bool],
  # Out:
  y_out: wp.array3d[float],
  acc_out: wp.array2d[float],
  rvec_out: wp.array2d[float],
  res_out: wp.array2d[float],
):
  """Res = (L S L')^-1 vec on the covered dofs by elimination-tree level, one block per world.

  vec is gathered into elimination order in y and the result scattered from acc. A world without a
  valid factor takes its 3x3 block solves from efm_L (when given) instead. With epU, the uncovered
  dofs take the inverse of their dense block epS.

  Below the dense root, each level scatters L_ij y_j to its ancestors' y (unit diagonal), one sync
  per level; then acc = S^-1 y, and the dense root's input is gathered into rvec. The dense root
  (_eff_chol_root_solve) and the backward sweep (_eff_chol_solve_backward) follow.
  """
  worldid, tid = wp.tid()
  nthread = wp.block_dim()
  nb = efm_chol_vert.shape[0]
  # uncovered dofs, from the inverse of their dense block (when given): one product per thread
  n_u = epU_in.shape[0]
  for a in range(tid, n_u, nthread):
    res_out[worldid, epU_in[a]] = 0.0
  _eff_chol_sync()
  for u in range(tid, n_u * n_u, nthread):
    a = u // n_u
    b = u - a * n_u
    wp.atomic_add(res_out, worldid, epU_in[a], epS_in[worldid, a, b] * vec_in[worldid, epU_in[b]])
  if not status_in[worldid]:
    if efm_L_in.shape[1] > 0:
      for k in range(tid, nb, nthread):
        _eff_block_solve_vertex(efm_dofid, efm_L_in, vec_in, worldid, k, res_out)
    return
  lvlptr = lvlptr_in[worldid]
  lvlent = lvlent_in[worldid]
  ent = ent_in[worldid]
  Lo = Lo_in[worldid]
  Sinv = Sinv_in[worldid]
  x = y_out[worldid, 0]
  acc = acc_out[worldid]
  nlevel = nlevel_in[worldid]
  r0 = rootlv_in[worldid]
  nr = 3 * nroot_in[worldid]
  ridx = ridx_in[worldid]
  rvec = rvec_out[worldid]
  # gather: y in elimination order
  for j in range(tid, nb, nthread):
    i = efm_dofid[efm_chol_vert[j]]
    x[3 * j] = vec_in[worldid, i]
    x[3 * j + 1] = vec_in[worldid, i + 1]
    x[3 * j + 2] = vec_in[worldid, i + 2]
  _eff_chol_sync()
  # forward: L w = y in place
  for lv in range(r0):
    for u in range(lvlent[lv] + tid, lvlent[lv + 1], nthread):
      e = ent[u]
      j = e[0]
      i = e[2]
      upd = Lo[e[1]] * wp.vec3(x[3 * j], x[3 * j + 1], x[3 * j + 2])
      wp.atomic_sub(x, 3 * i, upd[0])
      wp.atomic_sub(x, 3 * i + 1, upd[1])
      wp.atomic_sub(x, 3 * i + 2, upd[2])
    _eff_chol_sync()
  # pivots: acc = S^-1 w (S_R = I), then the dense root's input in rvec and its outputs zeroed
  for j in range(tid, nb, nthread):
    v = Sinv[j] * wp.vec3(x[3 * j], x[3 * j + 1], x[3 * j + 2])
    acc[3 * j] = v[0]
    acc[3 * j + 1] = v[1]
    acc[3 * j + 2] = v[2]
  _eff_chol_sync()
  for r in range(tid, nr, nthread):
    rvec[r] = x[ridx[r]]
    x[ridx[r]] = 0.0
    acc[ridx[r]] = 0.0


@wp.kernel
def _eff_chol_root_solve(
  # In:
  transpose: bool,
  nroot_in: wp.array[int],
  ridx_in: wp.array2d[int],
  D_in: wp.array3d[float],
  Dinv_in: wp.array3d[float],
  rvec_in: wp.array2d[float],
  status_in: wp.array[bool],
  # Out:
  y_out: wp.array3d[float],
  acc_out: wp.array2d[float],
):
  """One column chunk of a dense root row, over all blocks.

  Forward, acc_R += X y_R from rvec (X' read coalesced from D); with transpose, y_R += X' acc_R.
  """
  worldid, u = wp.tid()
  if not status_in[worldid]:
    return
  nr = 3 * nroot_in[worldid]
  q = u // D_in.shape[1]
  r = u - q * D_in.shape[1]
  if r >= nr:
    return
  ridx = ridx_in[worldid]
  v = float(0.0)
  if transpose:
    c1 = nr - q * EFF_CHOL_ROOT_CHUNK
    if c1 <= r:
      return
    acc = acc_out[worldid]
    for c in range(wp.max(c1 - EFF_CHOL_ROOT_CHUNK, r), c1):
      v += Dinv_in[worldid, c, r] * acc[ridx[c]]
    wp.atomic_add(y_out[worldid, 0], ridx[r], v)
  else:
    c0 = q * EFF_CHOL_ROOT_CHUNK
    if c0 > r:
      return
    for c in range(c0, wp.min(c0 + EFF_CHOL_ROOT_CHUNK, r + 1)):
      v += D_in[worldid, c, r] * rvec_in[worldid, c]
    wp.atomic_add(acc_out[worldid], ridx[r], v)


@wp.kernel(module="unique", enable_backward=False)
def _eff_chol_solve_backward(
  # Model:
  efm_dofid: wp.array[int],
  efm_chol_vert: wp.array[int],
  # In:
  lvlent_in: wp.array2d[int],
  ent_in: wp.array2d[wp.vec3i],
  rootlv_in: wp.array[int],
  nroot_in: wp.array[int],
  ridx_in: wp.array2d[int],
  Lo_in: wp.array2d[wp.mat33],
  y_in: wp.array3d[float],
  status_in: wp.array[bool],
  # Out:
  acc_out: wp.array2d[float],
  res_out: wp.array2d[float],
):
  """Backward sweep L' y = acc below the dense root, one block per world, and the scatter to res."""
  worldid, tid = wp.tid()
  if not status_in[worldid]:
    return
  nthread = wp.block_dim()
  nb = efm_chol_vert.shape[0]
  lvlent = lvlent_in[worldid]
  ent = ent_in[worldid]
  Lo = Lo_in[worldid]
  x = y_in[worldid, 0]
  acc = acc_out[worldid]
  r0 = rootlv_in[worldid]
  ridx = ridx_in[worldid]
  for r in range(tid, 3 * nroot_in[worldid], nthread):
    acc[ridx[r]] = x[ridx[r]]
  _eff_chol_sync()
  for lvr in range(r0):
    lv = r0 - 1 - lvr
    for u in range(lvlent[lv] + tid, lvlent[lv + 1], nthread):
      e = ent[u]
      j = e[0]
      i = e[2]
      t = wp.transpose(Lo[e[1]]) * wp.vec3(acc[3 * i], acc[3 * i + 1], acc[3 * i + 2])
      wp.atomic_sub(acc, 3 * j, t[0])
      wp.atomic_sub(acc, 3 * j + 1, t[1])
      wp.atomic_sub(acc, 3 * j + 2, t[2])
    _eff_chol_sync()
  for j in range(tid, nb, nthread):
    i = efm_dofid[efm_chol_vert[j]]
    res_out[worldid, i] = acc[3 * j]
    res_out[worldid, i + 1] = acc[3 * j + 1]
    res_out[worldid, i + 2] = acc[3 * j + 2]


def eff_chol_factor(m: Model, d: Data, chol: EffCholContext, fold: bool):
  """Assembles and factors the covered flex block of the metric (MuJoCo C effCholFill).

  The matrix is M + K on the covered vertex triples. K couplings between vertices that are neither
  joined by a flex edge nor coupled by a folded term are dropped, each replaced by its Frobenius
  norm on both diagonal blocks. With fold, the tendon and actuator rank-1 terms and every efc row
  (scaled by efc_D) are folded in, and the vertex pairs they couple join the pattern
  (mj_effCholFoldFactor). The factor runs in the model's elimination order on the static flex-edge
  pattern extended by those pairs, its symbolic structure computed in parallel from the elimination
  tree and row subtrees. A rank-deficient factor, or one that outgrows its buffers (reported as
  OverflowType.PRECONDITIONER), leaves the preconditioner on its 3x3 blocks (status False).
  """
  nb = m.nefmdof
  nxmax = chol.xpair.shape[1]
  chol.Ld.zero_()
  chol.xcount.zero_()
  if fold:
    chol.xhash.zero_()

  def fold_terms(fold_pass: bool):
    rows_in = [fold_pass, chol.colptr, chol.rowind, chol.status]
    rows_out = [chol.xcount, chol.xpair, chol.xhash, chol.Ld, chol.Lo]
    wp.launch(
      _eff_chol_rank1_tendon,
      dim=(d.nworld, m.ntendon),
      inputs=[
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.efm_dofid,
        m.efm_dofblk,
        m.efm_chol_rank,
        m.efm_chol_adjptr,
        m.efm_chol_adj,
        d.ten_J,
        d.efm_ts,
      ]
      + rows_in,
      outputs=rows_out,
    )
    wp.launch(
      _eff_chol_rank1_actuator,
      dim=(d.nworld, m.nactuator),
      inputs=[
        m.efm_dofid,
        m.efm_dofblk,
        m.efm_chol_rank,
        m.efm_chol_adjptr,
        m.efm_chol_adj,
        d.moment_rownnz,
        d.moment_rowadr,
        d.moment_colind,
        d.actuator_moment,
        d.efm_as,
      ]
      + rows_in,
      outputs=rows_out,
    )
    if m.is_sparse:
      wp.launch(
        _eff_chol_efc_sparse,
        dim=(d.nworld, d.njmax),
        inputs=[
          m.efm_dofid,
          m.efm_dofblk,
          m.efm_chol_rank,
          m.efm_chol_adjptr,
          m.efm_chol_adj,
          d.nefc,
          d.efc.J_rownnz,
          d.efc.J_rowadr,
          d.efc.J_colind,
          d.efc.J,
          d.efc.D,
        ]
        + rows_in,
        outputs=rows_out,
      )
    else:
      wp.launch(
        _eff_chol_efc_dense,
        dim=(d.nworld, d.njmax),
        inputs=[
          m.nv,
          m.efm_dofid,
          m.efm_dofblk,
          m.efm_chol_rank,
          m.efm_chol_adjptr,
          m.efm_chol_adj,
          d.nefc,
          d.efc.J,
          d.efc.D,
        ]
        + rows_in,
        outputs=rows_out,
      )

  # vertex pairs the folded terms couple that no flex edge joins
  if fold:
    fold_terms(False)

  # lower adjacency in elimination order: flex edges plus the extra pairs, sorted and deduplicated
  static_adj = [m.efm_chol_rank, m.efm_chol_vert, m.efm_chol_adjptr, m.efm_chol_adj]
  wp.launch(_eff_chol_adj_count, dim=(d.nworld, nb), inputs=static_adj, outputs=[chol.cnt])
  if fold:
    wp.launch(
      _eff_chol_adj_count_extra, dim=(d.nworld, nxmax), inputs=[m.efm_chol_rank, chol.xcount, chol.xpair], outputs=[chol.cnt]
    )
  wp.launch_tiled(
    _eff_chol_adj_scan, dim=d.nworld, inputs=[nb, chol.cnt], outputs=[chol.aptr], block_dim=m.block_dim.eff_chol_factor
  )
  wp.launch(_eff_chol_adj_fill, dim=(d.nworld, nb), inputs=static_adj + [chol.aptr], outputs=[chol.adj, chol.cur])
  if fold:
    wp.launch(
      _eff_chol_adj_fill_extra,
      dim=(d.nworld, nxmax),
      inputs=[m.efm_chol_rank, chol.xcount, chol.xpair],
      outputs=[chol.adj, chol.cur],
    )
  wp.launch(_eff_chol_adj_sort, dim=(d.nworld, nb), inputs=[chol.aptr], outputs=[chol.aend, chol.adj])

  # symbolic factorization: elimination tree, then the row subtrees give the factor's columns
  wp.launch_tiled(
    _eff_chol_etree,
    dim=d.nworld,
    inputs=[nb, chol.aptr, chol.aend, chol.adj],
    outputs=[
      chol.parent,
      chol.first,
      chol.post,
      chol.level,
      chol.cnt,
      chol.cur,
      chol.lvlptr,
      chol.lvlcol,
      chol.nlevel,
      chol.rootlv,
      chol.nroot,
      chol.rpos,
      chol.ridx,
    ],
    block_dim=m.block_dim.eff_chol_etree,
  )
  tree = [chol.parent, chol.first, chol.post]
  wp.launch(
    _eff_chol_row_count,
    dim=(d.nworld, nb),
    inputs=[chol.aptr, chol.aend, chol.adj] + tree,
    outputs=[chol.adjpost, chol.cnt],
  )
  wp.launch_tiled(
    _eff_chol_col_scan(m.opt.warn_overflow),
    dim=d.nworld,
    inputs=[nb, d.njmax, chol.xcount, chol.xpair, chol.cnt, chol.lvlcol],
    outputs=[d.overflow, chol.colptr, chol.rowind, chol.pairptr, chol.entptr, chol.cur, chol.status],
    block_dim=m.block_dim.eff_chol_factor,
  )
  wp.launch(
    _eff_chol_row_fill,
    dim=(d.nworld, nb),
    inputs=[chol.aptr, chol.aend, chol.adjpost] + tree + [chol.status],
    outputs=[chol.cur, chol.rowind],
  )
  wp.launch(_eff_chol_col_sort, dim=(d.nworld, nb), inputs=[chol.colptr, chol.status], outputs=[chol.rowind, chol.Lo])
  wp.launch(
    _eff_chol_level_entries,
    dim=(d.nworld, nb),
    inputs=[chol.colptr, chol.rowind, chol.lvlptr, chol.lvlcol, chol.entptr, chol.level, chol.nlevel, chol.status],
    outputs=[chol.lvlent, chol.ent],
  )

  # numeric assembly and factorization
  if fold:
    fold_terms(True)
  wp.launch(
    _eff_chol_fill_mk,
    dim=(d.nworld, nb, 3),
    inputs=[
      m.M_rownnz,
      m.M_rowadr,
      m.M_colind,
      m.efm_K_rownnz,
      m.efm_K_rowadr,
      m.efm_K_colind,
      m.efm_dofid,
      m.efm_dofblk,
      m.efm_chol_rank,
      d.M,
      d.efm_K_val,
      chol.colptr,
      chol.rowind,
      chol.aptr,
      chol.aend,
      chol.adj,
      chol.status,
    ],
    outputs=[chol.Ld, chol.Lo],
  )
  chol.D.zero_()
  chol.Dinv.zero_()
  wp.launch(
    _eff_chol_root_gather,
    dim=(d.nworld, chol.D.shape[1]),
    inputs=[
      chol.colptr,
      chol.rowind,
      chol.lvlptr,
      chol.lvlcol,
      chol.rootlv,
      chol.nroot,
      chol.rpos,
      chol.Ld,
      chol.Lo,
      chol.status,
    ],
    outputs=[chol.Sinv, chol.D, chol.Dinv],
  )
  wp.launch_tiled(
    _eff_chol_factor_sparse,
    dim=d.nworld,
    inputs=[
      nb,
      chol.colptr,
      chol.rowind,
      chol.lvlptr,
      chol.lvlcol,
      chol.pairptr,
      chol.lvlent,
      chol.ent,
      chol.rootlv,
      chol.rpos,
    ],
    outputs=[chol.Ld, chol.Lo, chol.Sinv, chol.D, chol.mindiag, chol.status],
    block_dim=m.block_dim.eff_chol_factor,
  )

  # dense root: blocked right-looking Cholesky by tile panel, the inverse factor alongside
  ntile = chol.D.shape[1] // EFF_CHOL_ROOT_TILE
  root = [chol.nroot, chol.status]
  for k in range(ntile):
    wp.launch_tiled(
      _eff_chol_root_panel,
      dim=(d.nworld, ntile),
      inputs=[k] + root,
      outputs=[chol.D, chol.Dinv],
      block_dim=m.block_dim.eff_chol_root,
    )
    wp.launch_tiled(
      _eff_chol_root_update,
      dim=(d.nworld, ntile, ntile),
      inputs=[k] + root,
      outputs=[chol.D, chol.Dinv],
      block_dim=m.block_dim.eff_chol_root,
    )
  wp.launch(
    _eff_chol_root_check, dim=(d.nworld, chol.D.shape[1]), inputs=[chol.nroot, chol.mindiag, chol.D], outputs=[chol.status]
  )
  nrd = chol.D.shape[1]
  wp.launch(_eff_chol_root_transpose, dim=(d.nworld, nrd, nrd), inputs=[chol.nroot, chol.Dinv, chol.status], outputs=[chol.D])


def eff_chol_solve(
  m: Model,
  d: Data,
  chol: EffCholContext,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  efm_L: Optional[wp.array] = None,
  epU: Optional[wp.array] = None,
  epS_inv: Optional[wp.array] = None,
):
  """Overwrites res on the covered dofs with H^-1 * vec where the factor is valid.

  With efm_L, a world without a valid factor takes its covered dofs' 3x3 block solves instead.
  With epU and epS_inv, the uncovered dofs take the inverse of their dense block.
  """
  wp.launch_tiled(
    _eff_chol_solve_sparse,
    dim=d.nworld,
    inputs=[
      m.efm_dofid,
      m.efm_chol_vert,
      efm_L if efm_L is not None else wp.empty((d.nworld, 0), dtype=float),
      vec,
      epU if epU is not None else wp.empty(0, dtype=int),
      epS_inv if epS_inv is not None else wp.empty((d.nworld, 0, 0), dtype=float),
      chol.lvlptr,
      chol.nlevel,
      chol.lvlent,
      chol.ent,
      chol.rootlv,
      chol.nroot,
      chol.ridx,
      chol.Lo,
      chol.Sinv,
      chol.status,
    ],
    outputs=[chol.y, chol.acc, chol.rvec, res],
    block_dim=m.block_dim.eff_chol_solve,
  )
  # dense root over all blocks: acc_R = X y_R, then y_R = X' acc_R
  nrd = chol.D.shape[1]
  ntask = nrd * ((nrd + EFF_CHOL_ROOT_CHUNK - 1) // EFF_CHOL_ROOT_CHUNK)
  for transpose in (False, True):
    wp.launch(
      _eff_chol_root_solve,
      dim=(d.nworld, ntask),
      inputs=[transpose, chol.nroot, chol.ridx, chol.D, chol.Dinv, chol.rvec, chol.status],
      outputs=[chol.y, chol.acc],
    )
  wp.launch_tiled(
    _eff_chol_solve_backward,
    dim=d.nworld,
    inputs=[
      m.efm_dofid,
      m.efm_chol_vert,
      chol.lvlent,
      chol.ent,
      chol.rootlv,
      chol.nroot,
      chol.ridx,
      chol.Lo,
      chol.y,
      chol.status,
    ],
    outputs=[chol.acc, res],
    block_dim=m.block_dim.eff_chol_solve,
  )


def _eff_uncovered_solve(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  epS: wp.array,
  epS_factor: Optional[wp.array],
  epS_rhs: wp.array,
  epS_sol: wp.array,
  epU: wp.array,
):
  """Res = A^-1 vec on the uncovered dofs by Cholesky solves of the uncovered block."""
  n_u = epU.shape[0]
  n_u_pad = epS.shape[1]
  wp.launch(
    _eff_gather_uncovered_rhs,
    dim=(d.nworld, n_u_pad),
    inputs=[n_u, epU, vec],
    outputs=[epS_rhs],
  )
  if n_u <= 32:
    wp.launch_tiled(
      _eff_cholesky_solve(n_u),
      dim=d.nworld,
      inputs=[epS, epS_rhs],
      outputs=[epS_sol],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
  elif epS_factor is not None:
    wp.launch_tiled(
      _eff_cholesky_solve_blocked(TILE_SIZE_JTDAJ_DENSE, n_u_pad),
      dim=d.nworld,
      inputs=[epS_factor, epS_rhs.reshape(shape=(d.nworld, n_u_pad, 1))],
      outputs=[epS_sol.reshape(shape=(d.nworld, n_u_pad, 1))],
      block_dim=m.block_dim.update_gradient_cholesky_blocked,
    )
  wp.launch(
    _eff_scatter_uncovered_sol,
    dim=(d.nworld, n_u),
    inputs=[epU, epS_sol],
    outputs=[res],
  )


@event_scope
def eff_prec(
  m: Model,
  d: Data,
  res: wp.array2d[float],
  vec: wp.array2d[float],
  epL: Optional[wp.array] = None,
  epS: Optional[wp.array] = None,
  epS_factor: Optional[wp.array] = None,
  epS_rhs: Optional[wp.array] = None,
  epS_sol: Optional[wp.array] = None,
  epU: Optional[wp.array] = None,
  eff_chol: Optional[EffCholContext] = None,
):
  """Preconditions residual vector using 3x3 flex block Cholesky and backbone qHLD.

  With eff_chol, the covered flex dofs use its Cholesky factor where it is full rank.
  """
  if m.efm0_active:
    if m.nefm0dof < m.nv:
      wp.launch(
        _eff_copy_backbone_rhs,
        dim=(d.nworld, m.nv),
        inputs=[m.nv, m.efm_dofblk, vec],
        outputs=[res],
      )
      smooth.solve_LD(m, d, d.qHLD, d.qHDiagInv, res, res)
    wp.launch(
      _eff_prec_efm0_solve,
      dim=d.nworld,
      inputs=[
        m.nefm0dof,
        m.efm0_dofid,
        m.efm0_L_rownnz,
        m.efm0_L_rowadr,
        m.efm0_L_colind,
        m.efm0_L,
        vec,
      ],
      outputs=[res],
    )
    if m.opt.enableflags & EnableBit.SLEEP:
      wp.launch(
        zero_sleeping_dofs,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_treeid, d.tree_awake],
        outputs=[res],
      )
    return

  L_arr = epL if epL is not None else d.efm_L
  if 3 * m.nefmdof == m.nv:
    if eff_chol is not None:
      eff_chol_solve(m, d, eff_chol, res, vec, efm_L=L_arr)
    else:
      wp.launch(
        eff_block_solve,
        dim=(d.nworld, m.nefmdof),
        inputs=[m.efm_dofid, L_arr, vec],
        outputs=[res],
      )
    if m.opt.enableflags & EnableBit.SLEEP:
      wp.launch(
        zero_sleeping_dofs,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_treeid, d.tree_awake],
        outputs=[res],
      )
    return

  if epS is not None and epU is not None and epS_rhs is not None and epS_sol is not None:
    n_u = epU.shape[0]
    n_u_pad = epS.shape[1]
    inverse = n_u_pad <= EFF_UNCOVERED_INV_MAX and (n_u <= 32 or epS_factor is not None)
    if inverse and eff_chol is not None:
      # the Cholesky solve kernel applies the uncovered block's inverse too
      eff_chol_solve(m, d, eff_chol, res, vec, efm_L=L_arr, epU=epU, epS_inv=epS)
    elif inverse:
      wp.launch(_eff_uncovered_inv_mul, dim=(d.nworld, n_u), inputs=[n_u, epU, epS, vec], outputs=[res])
    else:
      _eff_uncovered_solve(m, d, res, vec, epS, epS_factor, epS_rhs, epS_sol, epU)
    if eff_chol is not None:
      if not inverse:
        eff_chol_solve(m, d, eff_chol, res, vec, efm_L=L_arr)
    else:
      wp.launch(
        eff_block_solve,
        dim=(d.nworld, m.nefmdof),
        inputs=[m.efm_dofid, L_arr, vec],
        outputs=[res],
      )
    if m.opt.enableflags & EnableBit.SLEEP:
      wp.launch(
        zero_sleeping_dofs,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_treeid, d.tree_awake],
        outputs=[res],
      )
    return

  dofblk = m.efm_dofblk
  wp.launch(
    _eff_copy_backbone_rhs,
    dim=(d.nworld, m.nv),
    inputs=[m.nv, dofblk, vec],
    outputs=[res],
  )
  smooth.solve_LD(m, d, d.qHLD, d.qHDiagInv, res, res)
  if eff_chol is not None:
    eff_chol_solve(m, d, eff_chol, res, vec, efm_L=L_arr)
  else:
    wp.launch(
      eff_block_solve,
      dim=(d.nworld, m.nefmdof),
      inputs=[m.efm_dofid, L_arr, vec],
      outputs=[res],
    )
  if m.opt.enableflags & EnableBit.SLEEP:
    wp.launch(
      zero_sleeping_dofs,
      dim=(d.nworld, m.nv),
      inputs=[m.dof_treeid, d.tree_awake],
      outputs=[res],
    )


@event_scope
def eff_solve(m: Model, d: Data, qacc: wp.array2d[float], qfrc: Optional[wp.array] = None):
  """Solves the effective metric system M_hat * qacc = qfrc_eff."""
  efm_r = wp.empty((d.nworld, m.nv), dtype=float)
  efm_z = wp.empty((d.nworld, m.nv), dtype=float)
  efm_p = wp.empty((d.nworld, m.nv), dtype=float)
  efm_Ap = wp.empty((d.nworld, m.nv), dtype=float)
  efm_rz = wp.empty(d.nworld, dtype=float)
  efm_bn = wp.empty(d.nworld, dtype=float)
  efm_done = wp.empty(d.nworld, dtype=bool)
  efm_nsolving = wp.empty(1, dtype=int)
  efm_iter = wp.empty(1, dtype=int)

  if qfrc is None:
    rhs = efm_r
    wp.launch(
      _eff_rhs,
      dim=(d.nworld, m.nv),
      inputs=[
        m.opt.enableflags,
        m.dof_treeid,
        d.tree_awake,
        d.qfrc_smooth,
        d.efm_c,
        d.efm_ca,
      ],
      outputs=[rhs],
    )
  else:
    rhs = qfrc

  has_spring_or_damper = not ((m.opt.disableflags & DisableBit.SPRING) and (m.opt.disableflags & DisableBit.DAMPER))
  has_efm_tendon = (m.has_tendon_stiffness and not (m.opt.disableflags & DisableBit.SPRING)) or (
    m.has_tendon_damping and not (m.opt.disableflags & DisableBit.DAMPER)
  )
  has_efm_actuator = m.has_efm_actuator and not (m.opt.disableflags & DisableBit.ACTUATION)
  has_flex_any = has_spring_or_damper and (
    m.nefmK > 0 or m.has_non_simple_flex or m.efm0_active or passive.flex_passive_contact(m) or not m.flex_interp_assemblable
  )
  if not has_flex_any and not has_efm_tendon and not has_efm_actuator:
    smooth.solve_LD(m, d, d.qHLD, d.qHDiagInv, qacc, rhs)
    if m.opt.enableflags & EnableBit.SLEEP:
      wp.launch(
        zero_sleeping_dofs,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_treeid, d.tree_awake],
        outputs=[qacc],
      )
    return

  efm_nsolving.fill_(d.nworld)
  efm_iter.zero_()
  # allocated here since _pcg_iteration may run inside wp.capture_while
  efm_con, flex_scratch = eff_mul_m_scratch(m, d)

  # under IPC the covered block is preconditioned by the Cholesky factor of M + K (mj_effSolve)
  eff_chol = None
  if eff_chol_active(m):
    eff_chol = create_eff_chol_context(m, d.nworld, d.njmax)
    eff_chol_factor(m, d, eff_chol, fold=False)

  eff_prec(m, d, efm_z, rhs, eff_chol=eff_chol)
  wp.launch_tiled(
    _pcg_init_tiled,
    dim=d.nworld,
    inputs=[
      m.nv,
      m.nactuator,
      m.ntendon,
      m.opt.enableflags,
      m.dof_treeid,
      d.tree_awake,
      d.efm_ts,
      d.efm_as,
      rhs,
      efm_z,
      has_flex_any,
      has_efm_tendon,
      has_efm_actuator,
    ],
    outputs=[qacc, efm_r, efm_p, efm_rz, efm_bn, efm_done, efm_nsolving],
    block_dim=m.block_dim.eff_pcg,
  )

  max_it = m.opt.iterations

  def _pcg_iteration():
    eff_mul_m(m, d, efm_Ap, efm_p, skip=efm_done, efm_con=efm_con, flex_scratch=flex_scratch)
    wp.launch_tiled(
      _pcg_pAp_step_and_check_tiled,
      dim=d.nworld,
      inputs=[
        m.nv,
        m.opt.tolerance,
        efm_p,
        efm_Ap,
        efm_rz,
        efm_bn,
        efm_done,
      ],
      outputs=[
        qacc,
        efm_r,
        efm_done,
        efm_nsolving,
      ],
      block_dim=m.block_dim.eff_pcg,
    )
    eff_prec(m, d, efm_z, efm_r, eff_chol=eff_chol)
    wp.launch_tiled(
      _pcg_beta_and_p_update_tiled,
      dim=d.nworld,
      inputs=[
        m.nv,
        efm_r,
        efm_z,
        efm_rz,
        efm_done,
        max_it,
      ],
      outputs=[
        efm_p,
        efm_rz,
        efm_nsolving,
        efm_iter,
      ],
      block_dim=m.block_dim.eff_pcg,
    )

  if max_it > 0:
    if m.opt.graph_conditional:
      wp.capture_while(efm_nsolving, while_body=_pcg_iteration)
    else:
      for _ in range(max_it):
        _pcg_iteration()

  wp.launch(
    _pcg_check_convergence_warning(int(m.opt.warn_overflow)),
    dim=d.nworld,
    inputs=[efm_done],
    outputs=[d.overflow],
  )
