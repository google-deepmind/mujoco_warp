"""custom adjoint definitions for MuJoCo Warp autodifferentiation.

This module centralizes all ``@wp.func_grad`` registrations, the
implicit differentiation adjoint for the constraint solver, and the
smooth constraint adjoint for friction gradient signal.

Import this module via ``grad.py`` dont import it directly
"""

import os

import warp as wp

from mujoco_warp._src import math
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_factorize_solve_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_func
from mujoco_warp._src.collision_smooth import compute_k_imp
from mujoco_warp._src.warp_util import cache_kernel

# ---------------------------------------------------------------------------
# Phase 3: efc-level gradient kernels for collision chain
# ---------------------------------------------------------------------------


@wp.kernel
def _efc_J_grad_kernel(
  # Model:
  nv: int,
  # Data in:
  nefc_in: wp.array[int],
  efc_force_in: wp.array2d[float],
  njmax_in: int,
  # In:
  v_in: wp.array2d[float],
  # Out:
  efc_J_grad_out: wp.array3d[float],
):
  """Compute adj_efc_J[i, j] = v[j] * efc_force[i].

  From KKT: F(qacc) = M*qacc - qfrc_smooth - J^T*f = 0
  The derivative of J^T*f w.r.t. J[i,j] is f[i] * delta, and the
  adjoint vector v gives the sensitivity: adj_J[i,j] = v[j] * f[i].
  """
  worldid, efcid, dofid = wp.tid()
  if efcid < nefc_in[worldid] and dofid < nv:
    efc_J_grad_out[worldid, efcid, dofid] = v_in[worldid, dofid] * efc_force_in[worldid, efcid]


@wp.kernel
def _efc_pos_grad_kernel(
  # Model:
  opt_timestep: wp.array[float],
  opt_disableflags: int,
  # Data in:
  contact_dist_in: wp.array[float],
  contact_includemargin_in: wp.array[float],
  contact_solref_in: wp.array[wp.vec2],
  contact_solimp_in: wp.array[types.vec5],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  efc_aref_grad_in: wp.array2d[float],
  # Out:
  efc_pos_grad_out: wp.array2d[float],
):
  """Compute adj_efc_pos from adj_efc_aref.

  From efc_aref = -k * imp * pos - b * vel, d(aref)/d(pos) = -k*imp.
  So adj_efc_pos = adj_efc_aref * (-k * imp).
  We iterate over contacts and their first dimension (normal direction).
  """
  conid = wp.tid()
  if conid >= nacon_in[0]:
    return
  if not (contact_type_in[conid] & 1):  # ContactType.CONSTRAINT
    return

  efcid = contact_efc_address_in[conid, 0]
  if efcid < 0:
    return

  worldid = contact_worldid_in[conid]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]

  solref = contact_solref_in[conid]
  solimp = contact_solimp_in[conid]
  includemargin = contact_includemargin_in[conid]
  pos_val = contact_dist_in[conid] - includemargin

  k_imp = compute_k_imp(opt_disableflags, solref, solimp, pos_val, timestep)

  # d(aref)/d(pos) = -k * imp
  daref_dpos = -k_imp[0] * k_imp[1]

  adj_aref = efc_aref_grad_in[worldid, efcid]
  efc_pos_grad_out[worldid, efcid] = adj_aref * daref_dpos


# ---------------------------------------------------------------------------
# Smooth constraint adjoint: friction Hessian correction kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _smooth_hessian_friction_correction(
  # Model:
  nv: int,
  # Data in:
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  nacon_in: wp.array[int],
  # In:
  friction_viscosity: float,
  friction_scale: float,
  # Out:
  H_out: wp.array3d[float],
):
  """Apply friction smoothing correction to the Hessian.

  For each friction constraint row (dimid > 0):
    - QUADRATIC (active): delta_D = D * (friction_scale - 1.0)  [reduces stiffness]
    - Otherwise (SATISFIED etc): delta_D = friction_viscosity    [adds viscous term]

  Applies delta_D * J_row^T * J_row to H via atomic_add.
  """
  conid, dimid = wp.tid()

  if conid >= nacon_in[0]:
    return

  # Only process constraint contacts
  if not (contact_type_in[conid] & 1):  # ContactType.CONSTRAINT = 1
    return

  # Skip normal direction (dimid=0) — only modify friction rows
  if dimid == 0:
    return

  condim = contact_dim_in[conid]
  if condim == 1:
    return  # frictionless contact, no friction rows
  if dimid >= 2 * (condim - 1):
    return  # beyond valid friction dimensions

  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return

  worldid = contact_worldid_in[conid]

  D = efc_D_in[worldid, efcid]
  state = efc_state_in[worldid, efcid]

  # Compute delta_D: difference between smooth D and what's currently in H
  # QUADRATIC state (value=1): constraint was active, D is in H → reduce it
  # SATISFIED state (value=0): constraint was inactive, 0 in H → add viscous
  delta_D = float(0.0)
  if state == 1:  # QUADRATIC
    delta_D = D * (friction_scale - 1.0)
  else:
    delta_D = friction_viscosity

  if delta_D == 0.0:
    return

  # Apply delta_D * J_row^T * J_row to H
  for i in range(nv):
    Ji = efc_J_in[worldid, efcid, i]
    if Ji == 0.0:
      continue
    for j in range(nv):
      Jj = efc_J_in[worldid, efcid, j]
      if Jj == 0.0:
        continue
      wp.atomic_add(H_out, worldid, i, j, delta_D * Ji * Jj)


# ---------------------------------------------------------------------------
# Smooth constraint adjoint: friction gradient bypass kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _friction_bypass_correction(
  # Model:
  nv: int,
  # Data in:
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  nacon_in: wp.array[int],
  # In:
  v_hessian_in: wp.array2d[float],
  v_free_in: wp.array2d[float],
  bypass_kf: float,
  # Out:
  v_out: wp.array2d[float],
):
  """Friction gradient bypass: restore tangential gradients attenuated by H^{-1}.

  For each friction constraint face (dimid > 0), computes:
    delta = J_fric . (v_free - v_hessian)   [gradient lost to friction attenuation]
    v_out += kf * J_fric^T * delta            [inject it back, scaled by kf]

  v_hessian = H^{-1} * adj_qacc  (attenuated in friction directions)
  v_free    = M^{-1} * adj_qacc  (what gradient would be without constraints)

  This makes the backward pass produce dflex-like friction gradients while
  keeping the forward physics unchanged.
  """
  conid, dimid = wp.tid()

  if conid >= nacon_in[0]:
    return

  # Only process constraint contacts
  if not (contact_type_in[conid] & 1):  # ContactType.CONSTRAINT = 1
    return

  # Skip normal direction (dimid=0) — only bypass friction rows
  if dimid == 0:
    return

  condim = contact_dim_in[conid]
  if condim == 1:
    return  # frictionless contact, no friction rows
  if dimid >= 2 * (condim - 1):
    return  # beyond valid friction dimensions

  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return

  worldid = contact_worldid_in[conid]

  # Compute delta = J_fric . (v_free - v_hessian) for this friction face
  delta = float(0.0)
  for dofid in range(nv):
    J_val = efc_J_in[worldid, efcid, dofid]
    if J_val != 0.0:
      delta += J_val * (v_free_in[worldid, dofid] - v_hessian_in[worldid, dofid])

  # Apply correction: v_out += kf * J_fric^T * delta
  if delta != 0.0:
    scaled_delta = bypass_kf * delta
    for dofid in range(nv):
      J_val = efc_J_in[worldid, efcid, dofid]
      if J_val != 0.0:
        wp.atomic_add(v_out, worldid, dofid, scaled_delta * J_val)


@wp.kernel
def _friction_bypass_correction_normalized(
  # Model:
  nv: int,
  # Data in:
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  nacon_in: wp.array[int],
  # In:
  v_hessian_in: wp.array2d[float],
  v_free_in: wp.array2d[float],
  bypass_kf: float,
  max_ratio: float,
  norm_eps: float,
  # Out:
  v_out: wp.array2d[float],
):
  """Normalized and capped friction bypass correction.

  Projects the free-body delta onto each friction row and injects only a
  bounded fraction of that projected component.

  Compared to _friction_bypass_correction this avoids scaling by ||J_row||^2
  and prevents over-injection when contact rows become poorly conditioned.
  """
  conid, dimid = wp.tid()

  if conid >= nacon_in[0]:
    return

  # Only process constraint contacts
  if not (contact_type_in[conid] & 1):  # ContactType.CONSTRAINT = 1
    return

  # Skip normal direction (dimid=0) - only bypass friction rows
  if dimid == 0:
    return

  condim = contact_dim_in[conid]
  if condim == 1:
    return
  if dimid >= 2 * (condim - 1):
    return

  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return

  worldid = contact_worldid_in[conid]

  delta = float(0.0)
  j_norm2 = float(0.0)
  for dofid in range(nv):
    J_val = efc_J_in[worldid, efcid, dofid]
    if J_val != 0.0:
      delta += J_val * (v_free_in[worldid, dofid] - v_hessian_in[worldid, dofid])
      j_norm2 += J_val * J_val

  if j_norm2 <= norm_eps:
    return

  # Row-normalized projection coefficient.
  base_coeff = delta / j_norm2
  coeff = bypass_kf * base_coeff

  # Bound injected magnitude relative to the projected free-body component.
  max_coeff = wp.abs(base_coeff) * max_ratio
  abs_coeff = wp.abs(coeff)
  if abs_coeff > max_coeff and abs_coeff > 0.0:
    coeff = coeff * (max_coeff / abs_coeff)

  if coeff == 0.0:
    return

  for dofid in range(nv):
    J_val = efc_J_in[worldid, efcid, dofid]
    if J_val != 0.0:
      wp.atomic_add(v_out, worldid, dofid, coeff * J_val)


# Penalty-model adjoint: friction damping kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _penalty_friction_damping(
  # Model:
  nv: int,
  # Data in:
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  nacon_in: wp.array[int],
  # In:
  v_free_in: wp.array2d[float],
  damping_alpha: float,
  # Out:
  v_out: wp.array2d[float],
):
  """Apply penalty-model friction damping to the free-body adjoint.

  For each friction face: v_out -= alpha * J_fric^T * (J_fric . v_free)

  This attenuates v in friction directions by factor (1 - alpha), mimicking
  dflex's penalty friction gradient where d(v_next)/d(v_prev) has eigenvalues
  < 1 in friction-constrained directions.  Provides natural BPTT decay that
  prevents gradient explosion while preserving gradient direction.
  """
  conid, dimid = wp.tid()

  if conid >= nacon_in[0]:
    return

  if not (contact_type_in[conid] & 1):
    return

  # Friction rows only (dimid > 0)
  if dimid == 0:
    return

  condim = contact_dim_in[conid]
  if condim == 1:
    return
  if dimid >= 2 * (condim - 1):
    return

  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return

  worldid = contact_worldid_in[conid]

  # Project v_free onto this friction face
  proj = float(0.0)
  for dofid in range(nv):
    J_val = efc_J_in[worldid, efcid, dofid]
    if J_val != 0.0:
      proj += J_val * v_free_in[worldid, dofid]

  # Subtract friction damping: v_out -= alpha * J^T * proj
  if proj != 0.0:
    scaled = damping_alpha * proj
    for dofid in range(nv):
      J_val = efc_J_in[worldid, efcid, dofid]
      if J_val != 0.0:
        wp.atomic_add(v_out, worldid, dofid, -scaled * J_val)


@wp.func_grad(math.quat_integrate)
def _quat_integrate_grad(q: wp.quat, v: wp.vec3, dt: float, adj_ret: wp.quat):
  """Custom adjoint avoiding gradient singularity at |v|=0."""
  EPS = float(1e-10)
  norm_v = wp.length(v)
  norm_v_sq = norm_v * norm_v
  half_angle = dt * norm_v * 0.5

  # sinc-safe rotation quaternion construction
  if norm_v > EPS:
    s_over_nv = wp.sin(half_angle) / norm_v  # sin(dt|v|/2) / |v|
    c = wp.cos(half_angle)
    # d(s_over_nv)/dv_j = ds_coeff * v_j
    ds_coeff = (c * dt * 0.5 - s_over_nv) / norm_v_sq
  else:
    s_over_nv = dt * 0.5
    c = 1.0
    # Taylor limit: (c*dt/2 - s_over_nv) / |v|^2 -> -dt^3/24
    ds_coeff = -dt * dt * dt / 24.0

  q_rot = wp.quat(
    c,
    s_over_nv * v[0],
    s_over_nv * v[1],
    s_over_nv * v[2],
  )

  # recompute forward intermediates
  q_len = wp.length(q)
  q_inv_len = 1.0 / wp.max(q_len, EPS)
  q_n = wp.quat(
    q[0] * q_inv_len,
    q[1] * q_inv_len,
    q[2] * q_inv_len,
    q[3] * q_inv_len,
  )

  q_res = math.mul_quat(q_n, q_rot)
  res_len = wp.length(q_res)
  res_inv = 1.0 / wp.max(res_len, EPS)

  # result = normalize(q_res)
  # adj_q_res_k = adj_ret_k / |q_res| - q_res_k * dot(adj_ret, q_res) / |q_res|^3
  dot_ar = adj_ret[0] * q_res[0] + adj_ret[1] * q_res[1] + adj_ret[2] * q_res[2] + adj_ret[3] * q_res[3]
  res_inv3 = res_inv * res_inv * res_inv
  adj_qr = wp.quat(
    adj_ret[0] * res_inv - q_res[0] * dot_ar * res_inv3,
    adj_ret[1] * res_inv - q_res[1] * dot_ar * res_inv3,
    adj_ret[2] * res_inv - q_res[2] * dot_ar * res_inv3,
    adj_ret[3] * res_inv - q_res[3] * dot_ar * res_inv3,
  )

  # q_res = mul_quat(q_n, q_rot)
  # adj_q_n  = mul_quat(adj_qr, conj(q_rot))
  # adj_q_rot = mul_quat(conj(q_n), adj_qr)
  q_rot_conj = wp.quat(q_rot[0], -q_rot[1], -q_rot[2], -q_rot[3])
  adj_qn = math.mul_quat(adj_qr, q_rot_conj)

  q_n_conj = wp.quat(q_n[0], -q_n[1], -q_n[2], -q_n[3])
  adj_q_rot = math.mul_quat(q_n_conj, adj_qr)

  # q_rot = (c, s_over_nv * v)
  # d(c)/dv_j = -s_over_nv * dt/2 * v_j
  # d(s_over_nv * v_i)/dv_j = ds_coeff * v_j * v_i + s_over_nv * delta_ij
  sv_dot = adj_q_rot[1] * v[0] + adj_q_rot[2] * v[1] + adj_q_rot[3] * v[2]
  common = -s_over_nv * dt * 0.5 * adj_q_rot[0] + ds_coeff * sv_dot
  adj_v_val = wp.vec3(
    common * v[0] + s_over_nv * adj_q_rot[1],
    common * v[1] + s_over_nv * adj_q_rot[2],
    common * v[2] + s_over_nv * adj_q_rot[3],
  )

  # adj_dt from q_rot dependency on dt
  # d(c)/d(dt)            = -sin(half_angle) * norm_v / 2
  # d(s_over_nv * v_i)/dt = (c / 2) * v_i
  adj_dt_val = adj_q_rot[0] * (-wp.sin(half_angle) * norm_v * 0.5)
  adj_dt_val += sv_dot * c * 0.5

  # q_n = normalize(q)
  # adj_q_k = adj_qn_k / |q| - q_k * dot(adj_qn, q) / |q|^3
  dot_aqn = adj_qn[0] * q[0] + adj_qn[1] * q[1] + adj_qn[2] * q[2] + adj_qn[3] * q[3]
  q_inv_len3 = q_inv_len * q_inv_len * q_inv_len
  adj_q_val = wp.quat(
    adj_qn[0] * q_inv_len - q[0] * dot_aqn * q_inv_len3,
    adj_qn[1] * q_inv_len - q[1] * dot_aqn * q_inv_len3,
    adj_qn[2] * q_inv_len - q[2] * dot_aqn * q_inv_len3,
    adj_qn[3] * q_inv_len - q[3] * dot_aqn * q_inv_len3,
  )

  # accumulate adjoints
  wp.adjoint[q] += adj_q_val
  wp.adjoint[v] += adj_v_val
  wp.adjoint[dt] += adj_dt_val


# ---------------------------------------------------------------------------
# Solver implicit differentiation adjoint
# ---------------------------------------------------------------------------

_BLOCK_CHOLESKY_DIM = 32


@wp.kernel
def _copy_grad_kernel(
  # In:
  src: wp.array2d[float],
  # Out:
  dst_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  dst_out[worldid, dofid] = src[worldid, dofid]


@wp.kernel
def _accumulate_grad_kernel(
  # In:
  src: wp.array2d[float],
  # Out:
  dst_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  dst_out[worldid, dofid] = dst_out[worldid, dofid] + src[worldid, dofid]


@cache_kernel
def _adjoint_cholesky_tile(nv: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    H: wp.array3d[float],
    b: wp.array2d[float],
    # Out:
    out: wp.array2d[float],
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(nv)
    H_tile = wp.tile_load(H[worldid], shape=(TILE_SIZE, TILE_SIZE))
    b_tile = wp.tile_load(b[worldid], shape=(TILE_SIZE,))
    L = wp.tile_cholesky(H_tile)
    x = wp.tile_cholesky_solve(L, b_tile)
    wp.tile_store(out[worldid], x)

  return kernel


@cache_kernel
def _adjoint_cholesky_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    hfactor: wp.array3d[float],
    b: wp.array3d[float],
    nv_runtime: int,
    # Out:
    out: wp.array3d[float],
  ):
    worldid = wp.tid()
    wp.static(create_blocked_cholesky_solve_func(tile_size, matrix_size))(
      hfactor[worldid], b[worldid], nv_runtime, out[worldid]
    )

  return kernel


@cache_kernel
def _adjoint_cholesky_full_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    H: wp.array3d[float],
    b: wp.array3d[float],
    nv_runtime: int,
    hfactor_tmp: wp.array3d[float],
    # Out:
    out: wp.array3d[float],
  ):
    worldid = wp.tid()
    # Fused factorize+solve (upstream replaced the separate factorize func);
    # hfactor_tmp receives the factor as a side effect.
    wp.static(create_blocked_cholesky_factorize_solve_func(tile_size, matrix_size))(
      H[worldid], b[worldid], nv_runtime, hfactor_tmp[worldid], out[worldid]
    )

  return kernel


@wp.kernel
def _padding_h_adjoint(
  # Model:
  nv: int,
  # Out:
  H_out: wp.array3d[float],
):
  worldid, elementid = wp.tid()
  dofid = nv + elementid
  H_out[worldid, dofid, dofid] = 1.0


@wp.kernel
def _symmetrize_upper_kernel(
  # Model:
  nv: int,
  # Out:
  H_out: wp.array3d[float],
):
  """Mirror the upper triangle of H into the lower triangle.

  The Newton solver's dense JTDAJ kernels (full tiled build and the
  incremental active-set update) maintain only the UPPER triangle of the
  Hessian; the lower triangle can hold stale values from earlier solves.
  The adjoint Cholesky consumes the full matrix, so without this the
  factorization sees an asymmetric (possibly indefinite) matrix and
  produces NaNs (observed with contact-rich scenes, e.g. multi-leg ant).
  """
  worldid, elementid = wp.tid()
  # Strict lower-triangle element (row > col) via triangular indexing.
  row = int(0)
  rem = elementid
  size = nv - 1
  while rem >= size and size > 0:
    rem -= size
    size -= 1
    row += 1
  col = row + 1 + rem
  H_out[worldid, col, row] = H_out[worldid, row, col]


def _solve_hessian_system(m: types.Model, d: types.Data, b, out, H=None):
  """Solve H * x = b using stored solver Hessian or a provided H.

  Args:
    m: Model.
    d: Data.
    b: Right-hand side vector (nworld, nv_pad).
    out: Solution vector (nworld, nv_pad).
    H: Optional Hessian override. When provided, always factorizes from
       scratch (ignores stored d.solver_hfactor). Used by smooth adjoint.
  """
  use_stored = H is None
  if use_stored:
    H = d.solver_h

  # The solver only maintains the upper triangle (see kernel docstring).
  # Skip when a stored factor will be used instead of refactorizing.
  will_factorize = m.nv <= _BLOCK_CHOLESKY_DIM or not (use_stored and d.solver_hfactor.shape[1] > 0)
  if will_factorize and m.nv > 1:
    wp.launch(
      _symmetrize_upper_kernel,
      dim=(d.nworld, m.nv * (m.nv - 1) // 2),
      inputs=[m.nv],
      outputs=[H],
    )

  if m.nv <= _BLOCK_CHOLESKY_DIM:
    wp.launch_tiled(
      _adjoint_cholesky_tile(m.nv),
      dim=d.nworld,
      inputs=[H, b],
      outputs=[out],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
  else:
    # The blocked Cholesky kernels operate on nv_pad-sized tiles, so the
    # right-hand side must be nv_pad wide. The incoming adjoint b is only nv
    # wide (e.g. nv=81, nv_pad=96 for a sparse model), so a direct reshape to
    # (nworld, nv_pad, 1) fails the same-total-size check. Copy b into a padded
    # zero buffer first; the trailing padding rows stay zero (they correspond to
    # the padding DOFs handled by _padding_h_adjoint).
    if b.shape[1] != m.nv_pad:
      nv_b = b.shape[1]
      b_pad = wp.zeros((d.nworld, m.nv_pad), dtype=float)
      wp.launch(_copy_grad_kernel, dim=(d.nworld, nv_b), inputs=[b], outputs=[b_pad])
      b = b_pad
    b_3d = b.reshape((d.nworld, m.nv_pad, 1))
    out_3d = out.reshape((d.nworld, m.nv_pad, 1))

    if use_stored and d.solver_hfactor.shape[1] > 0:
      # Solve-only using stored Cholesky factor (original H only).
      # Pass nv_pad (not nv) as the runtime matrix size: the blocked kernels
      # load tiles at block_size-aligned offsets, so a runtime size that is not
      # a multiple of the tile size produces unaligned tile loads and an illegal
      # memory access (CUDA 719). The forward factorization (_padding_h plus
      # _update_gradient_cholesky_blocked) already pads H to nv_pad with an
      # identity block and runs the solve at nv_pad, so the stored factor is
      # nv_pad-wide and the padding rows resolve to zero.
      wp.launch_tiled(
        _adjoint_cholesky_blocked(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
        dim=d.nworld,
        inputs=[d.solver_hfactor, b_3d, m.nv_pad],
        outputs=[out_3d],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )
    else:
      # Full factorize + solve
      if m.nv_pad > m.nv:
        wp.launch(
          _padding_h_adjoint,
          dim=(d.nworld, m.nv_pad - m.nv),
          inputs=[m.nv],
          outputs=[H],
        )
      hfactor_tmp = wp.zeros((d.nworld, m.nv_pad, m.nv_pad), dtype=float)
      # Pass nv_pad (not nv): see the note above on tile alignment. H is padded
      # to nv_pad with an identity block by _padding_h_adjoint, so factorizing
      # and solving the full nv_pad system is exact for the leading nv rows.
      wp.launch_tiled(
        _adjoint_cholesky_full_blocked(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
        dim=d.nworld,
        inputs=[H, b_3d, m.nv_pad, hfactor_tmp],
        outputs=[out_3d],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )


def solver_implicit_adjoint(m: types.Model, d: types.Data, qacc_array=None, qacc_smooth_ref=None, cap=None):
  """Implicit differentiation adjoint for constraint solver.

  Called during tape backward. Reads qacc_array.grad (set by downstream
  integrator adjoint), solves H*v = adj_qacc, accumulates into
  qacc_smooth_ref.grad += M*v.

  Args:
    m: Model containing static simulation parameters.
    d: Data containing mutable simulation state.
    qacc_array: The array whose .grad contains the incoming adjoint.
                Defaults to d.qacc when called from diff_forward().
                Integrators pass their local qacc array when it differs
                from d.qacc (e.g. euler with implicit damping).
    qacc_smooth_ref: The qacc_smooth array whose .grad receives the
                     accumulated adjoint. Captured at record time for
                     correct gradient isolation when intermediate arrays
                     are cloned between substeps. Defaults to d.qacc_smooth.
  """
  nv = m.nv
  if nv == 0:
    return

  if qacc_array is None:
    qacc_array = d.qacc

  if qacc_smooth_ref is None:
    qacc_smooth_ref = d.qacc_smooth

  adj_qacc = qacc_array.grad
  if adj_qacc is None:
    return

  debug_level = os.environ.get("MJW_DEBUG_ADJOINT", "0")
  if debug_level in ("1", "2"):
    import numpy as np

    adj_norm = np.linalg.norm(adj_qacc.numpy())
    print(f"[adjoint] |adj_qacc|={adj_norm:.6e}, njmax={d.njmax}")

  if debug_level == "2" and d.njmax > 0:
    import numpy as np

    efc_state_np = d.efc.state.numpy()
    nefc_np = d.nefc.numpy()
    for w in range(min(d.nworld, 1)):
      ne = nefc_np[w]
      n_quad = int(np.sum(efc_state_np[w, :ne] == 1))
      n_sat = int(np.sum(efc_state_np[w, :ne] == 0))
      H_np = d.solver_h.numpy()[w, :nv, :nv]
      H_diag = np.diag(H_np)
      cond_approx = np.max(H_diag) / max(np.min(H_diag[H_diag > 0]), 1e-30)
      print(
        f"[adjoint:diag] world={w} nefc={ne} QUAD={n_quad} SAT={n_sat}"
        f" H_cond~{cond_approx:.1f}"
        f" H_diag=[{np.min(H_diag):.3e}, {np.max(H_diag):.3e}]"
      )

  if d.njmax == 0:
    # Solver was identity (qacc = qacc_smooth), accumulate adjoint through
    wp.launch(
      _accumulate_grad_kernel,
      dim=(d.nworld, nv),
      inputs=[adj_qacc],
      outputs=[qacc_smooth_ref.grad],
    )
    return

  if m.opt.solver != types.SolverType.NEWTON:
    # CG solver: no Hessian stored, fall back to identity
    wp.launch(
      _accumulate_grad_kernel,
      dim=(d.nworld, nv),
      inputs=[adj_qacc],
      outputs=[qacc_smooth_ref.grad],
    )
    return

  # Solve H * v = adj_qacc. Prefer the record-time captured active-set Hessian (correct contact
  # active set at the fixed point); fall back to the stored d.solver_h otherwise.
  v = wp.zeros((d.nworld, m.nv_pad), dtype=float)
  if cap is not None:
    _solve_hessian_system(m, d, adj_qacc, v, H=cap["H"])
  else:
    _solve_hessian_system(m, d, adj_qacc, v)

  # adj_qacc_smooth += M * v  (accumulate, not overwrite)
  tmp = wp.zeros((d.nworld, m.nv_pad), dtype=float)
  support.mul_m(m, d, tmp, v)
  wp.launch(
    _accumulate_grad_kernel,
    dim=(d.nworld, nv),
    inputs=[tmp],
    outputs=[qacc_smooth_ref.grad],
  )

  # Phase 3: efc-level gradients for the collision chain (use the captured snapshot when
  # available so J grad and the aref->vel->qvel path use the fixed-point active set).
  _efc_level_gradients(m, d, v, cap=cap)


def _efc_level_gradients(m: types.Model, d: types.Data, v, cap=None):
  """Compute efc-level gradients for collision chain (shared by both adjoints).

  When cap (a record-time snapshot) is provided, the contact active set / J / D / pos come from
  the fixed point rather than the cleared backward-time buffers."""
  if d.njmax > 0:
    efc_J = d.efc.J
    if hasattr(efc_J, "grad") and efc_J.grad is not None:
      wp.launch(
        _efc_J_grad_kernel,
        dim=(d.nworld, d.njmax_pad, m.nv_pad),
        inputs=[m.nv, d.nefc, d.efc.force, d.njmax, v],
        outputs=[efc_J.grad],
      )

    efc_aref = d.efc.aref
    efc_pos = d.efc.pos
    # Populate efc.aref.grad = D * (J . v) on the active set (the contact-velocity gradient
    # path: aref carries the Baumgarte -b*vel term). Uses the captured fixed-point J/D/pos.
    if cap is not None and hasattr(efc_aref, "grad") and efc_aref.grad is not None:
      wp.launch(
        _efc_aref_grad_kernel,
        dim=(d.nworld, d.njmax),
        inputs=[cap["nefc"], cap["J_rownnz"], cap["J_rowadr"], cap["J_colind"], cap["J"], cap["D"], cap["pos"], v],
        outputs=[efc_aref.grad],
      )

    if hasattr(efc_aref, "grad") and efc_aref.grad is not None and hasattr(efc_pos, "grad") and efc_pos.grad is not None:
      wp.launch(
        _efc_pos_grad_kernel,
        dim=d.naconmax,
        inputs=[
          m.opt.timestep,
          m.opt.disableflags,
          d.contact.dist,
          d.contact.includemargin,
          d.contact.solref,
          d.contact.solimp,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.nacon,
          efc_aref.grad,
        ],
        outputs=[efc_pos.grad],
      )

    # Inject the contact velocity-dissipation adjoint for genuinely multi-DOF rows directly into
    # qvel.grad with a colind-indexed J^T scatter: adj_qvel = -sum_{i in A} b_i D_i (J_i . v) J_i.
    # The native vel->qvel autodiff of the sparse-J contact assembly indexes the velocity gradient
    # by stored row position rather than the efc.J column index, so it mis-routes rows that couple
    # more than one DOF (the pyramidal friction cone), where the two edges then cancel and the
    # tangential dissipation never reaches qvel. Effectively-single-DOF rows (incl. the normal row
    # of a multi-DOF body) are routed correctly per substep by the native chain, so the kernel
    # gates them out; for the multi-DOF rows it handles, it zeroes efc.aref.grad / efc.vel.grad so
    # the native path cannot double count (efc.aref.grad was scratch for the pos kernel above).
    #
    # KNOWN GAP: combined strong-normal + tangential friction over long rollouts (e.g. a body
    # pressed hard into a frictional plane, ~2x at nsteps=20). Root cause is the record_func order,
    # not this scatter: solver_implicit_adjoint runs in reverse AFTER _advance's backward has
    # already consumed d.qvel.grad and propagated it to the per-substep qvel clone, so this
    # injection into the shared d.qvel.grad accumulates across substeps. Routing to the per-substep
    # clone returns zero (its backward has already run). A proper fix records a dedicated dissipation
    # adjoint inside _advance between _next_velocity backward and the qvel clone backward; that
    # restructures the solver adjoint and is deferred.
    if cap is not None and d.qvel.grad is not None and hasattr(efc_aref, "grad") and efc_aref.grad is not None:
      efc_vel = d.efc.vel
      efc_vel_grad = efc_vel.grad if (hasattr(efc_vel, "grad") and efc_vel.grad is not None) else efc_aref.grad
      wp.launch(
        _qvel_contact_dissipation_kernel,
        dim=(d.nworld, d.njmax),
        inputs=[cap["nefc"], cap["J_rownnz"], cap["J_rowadr"], cap["J_colind"], cap["J"], cap["D"], cap["pos"],
                cap["efc_id"], d.contact.solref, d.contact.solimp, m.opt.timestep, m.opt.disableflags, v],
        outputs=[d.qvel.grad, efc_aref.grad, efc_vel_grad],
      )


# ---------------------------------------------------------------------------
# Smooth constraint adjoint: backward-only friction gradient smoothing
# ---------------------------------------------------------------------------


def solver_smooth_adjoint(
  m: types.Model,
  d: types.Data,
  qacc_array=None,
  qacc_smooth_ref=None,
):
  """Smooth constraint adjoint for friction gradient signal.

  Like solver_implicit_adjoint, but builds a modified Hessian H_smooth that
  reduces friction constraint stiffness and adds viscous friction for
  SATISFIED constraints. This provides non-zero gradients through the friction
  cone dead zone while keeping the forward physics unchanged.

  Parameters are read from d.smooth_friction_viscosity and
  d.smooth_friction_scale. Enable via d.smooth_adjoint = 1.

  Args:
    m: Model containing static simulation parameters.
    d: Data containing mutable simulation state.
    qacc_array: The array whose .grad contains the incoming adjoint.
    qacc_smooth_ref: The qacc_smooth array whose .grad receives the
                     accumulated adjoint.
  """
  nv = m.nv
  if nv == 0:
    return

  if qacc_array is None:
    qacc_array = d.qacc

  if qacc_smooth_ref is None:
    qacc_smooth_ref = d.qacc_smooth

  adj_qacc = qacc_array.grad
  if adj_qacc is None:
    return

  debug_level = os.environ.get("MJW_DEBUG_ADJOINT", "0")
  if debug_level in ("1", "2"):
    import numpy as np

    adj_norm = np.linalg.norm(adj_qacc.numpy())
    print(f"[smooth_adjoint] |adj_qacc|={adj_norm:.6e}, njmax={d.njmax}")

  if d.njmax == 0:
    wp.launch(
      _accumulate_grad_kernel,
      dim=(d.nworld, nv),
      inputs=[adj_qacc],
      outputs=[qacc_smooth_ref.grad],
    )
    return

  if m.opt.solver != types.SolverType.NEWTON:
    wp.launch(
      _accumulate_grad_kernel,
      dim=(d.nworld, nv),
      inputs=[adj_qacc],
      outputs=[qacc_smooth_ref.grad],
    )
    return

  # Read smooth adjoint parameters from Data
  free_body = getattr(d, "smooth_free_body_adjoint", False)
  penalty_alpha = getattr(d, "smooth_penalty_damping_alpha", 0.0)
  surrogate = getattr(d, "smooth_friction_surrogate_adjoint", False)
  surrogate_alpha = float(getattr(d, "smooth_friction_surrogate_alpha", 0.0))
  if surrogate_alpha < 0.0:
    surrogate_alpha = 0.0
  elif surrogate_alpha > 1.0:
    surrogate_alpha = 1.0

  if surrogate:
    friction_viscosity = getattr(d, "smooth_friction_viscosity", 10.0)
    friction_scale = getattr(d, "smooth_friction_scale", 0.01)

    H_smooth = wp.clone(d.solver_h)

    if d.naconmax > 0:
      wp.launch(
        _smooth_hessian_friction_correction,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.nv,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          d.nacon,
          friction_viscosity,
          friction_scale,
        ],
        outputs=[H_smooth],
      )

    v_hessian = wp.zeros((d.nworld, m.nv_pad), dtype=float)
    _solve_hessian_system(m, d, adj_qacc, v_hessian, H=H_smooth)

    from mujoco_warp._src.smooth import solve_m

    v_free = wp.zeros((d.nworld, m.nv_pad), dtype=float)
    solve_m(m, d, v_free, adj_qacc)

    v = wp.clone(v_hessian)
    if d.naconmax > 0:
      # Recover only a controlled fraction of the tangential free-body signal.
      # alpha=0 keeps the full bypass, alpha=1 leaves the smooth/Newton result.
      correction_scale = 1.0 - surrogate_alpha
      correction_cap_ratio = 1.0
      correction_norm_eps = 1.0e-8
      wp.launch(
        _friction_bypass_correction_normalized,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.nv,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.efc.J,
          d.nacon,
          v_hessian,
          v_free,
          correction_scale,
          correction_cap_ratio,
          correction_norm_eps,
        ],
        outputs=[v],
      )

  elif free_body or penalty_alpha > 0.0:
    # Free-body base: v = M^{-1} * adj_qacc
    # Eliminates H^{-1} attenuation entirely.
    from mujoco_warp._src.smooth import solve_m

    v = wp.zeros((d.nworld, m.nv_pad), dtype=float)
    solve_m(m, d, v, adj_qacc)

    # Penalty-model friction damping: attenuate v in friction directions
    # by factor (1 - alpha) per face, mimicking dflex's penalty friction
    # d(v_next)/d(v_prev) eigenvalues.  Provides natural BPTT decay.
    if penalty_alpha > 0.0 and d.naconmax > 0:
      v_free = wp.clone(v)  # save unmodified for projection
      wp.launch(
        _penalty_friction_damping,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.nv,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.efc.J,
          d.nacon,
          v_free,
          penalty_alpha,
        ],
        outputs=[v],
      )

  else:
    # Original smooth adjoint: H_smooth with friction correction + optional bypass
    friction_viscosity = getattr(d, "smooth_friction_viscosity", 10.0)
    friction_scale = getattr(d, "smooth_friction_scale", 0.01)
    bypass_kf = getattr(d, "smooth_friction_bypass_kf", 0.0)

    # Build H_smooth = d.solver_h + friction correction
    H_smooth = wp.clone(d.solver_h)

    if d.naconmax > 0:
      wp.launch(
        _smooth_hessian_friction_correction,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.nv,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          d.nacon,
          friction_viscosity,
          friction_scale,
        ],
        outputs=[H_smooth],
      )

    if debug_level == "2":
      import numpy as np

      H_np = H_smooth.numpy()[0, :nv, :nv]
      H_orig = d.solver_h.numpy()[0, :nv, :nv]
      diff = H_np - H_orig
      print(
        f"[smooth_adjoint:diag] H_smooth diag="
        f"[{np.min(np.diag(H_np)):.3e}, {np.max(np.diag(H_np)):.3e}]"
        f" |delta_H|_F={np.linalg.norm(diff):.3e}"
      )

    # Solve H_smooth * v = adj_qacc
    v = wp.zeros((d.nworld, m.nv_pad), dtype=float)
    _solve_hessian_system(m, d, adj_qacc, v, H=H_smooth)

    if debug_level == "2":
      import numpy as np

      v_np = v.numpy()[0, :nv]
      print(f"[smooth_adjoint:diag] |v|={np.linalg.norm(v_np):.6e} v={v_np}")

    # Friction gradient bypass: restore tangential gradients attenuated by H^{-1}
    if bypass_kf > 0.0 and d.naconmax > 0:
      from mujoco_warp._src.smooth import solve_m

      v_free = wp.zeros((d.nworld, m.nv_pad), dtype=float)
      solve_m(m, d, v_free, adj_qacc)

      wp.launch(
        _friction_bypass_correction,
        dim=(d.naconmax, m.nmaxpyramid),
        inputs=[
          m.nv,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.contact.type,
          d.efc.J,
          d.nacon,
          v,
          v_free,
          bypass_kf,
        ],
        outputs=[v],
      )

      if debug_level == "2":
        import numpy as np

        v_bypass = v.numpy()[0, :nv]
        print(f"[smooth_adjoint:diag] bypass kf={bypass_kf} |v_after_bypass|={np.linalg.norm(v_bypass):.6e}")

  # adj_qacc_smooth += M * v
  tmp = wp.zeros((d.nworld, m.nv_pad), dtype=float)
  support.mul_m(m, d, tmp, v)
  wp.launch(
    _accumulate_grad_kernel,
    dim=(d.nworld, nv),
    inputs=[tmp],
    outputs=[qacc_smooth_ref.grad],
  )

  # Phase 3: efc-level gradients for collision chain
  _efc_level_gradients(m, d, v)


@wp.kernel
def _jtdaj_geom_kernel(
  nefc_in: wp.array(dtype=int),
  efc_J_rownnz_in: wp.array2d(dtype=int),
  efc_J_rowadr_in: wp.array2d(dtype=int),
  efc_J_colind_in: wp.array3d(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_pos_in: wp.array2d(dtype=float),
  h_out: wp.array3d(dtype=float),
):
  """h += sum_{active efc} D * J^T J. Active = penetrating (pos<0) and stiff (D>0).

  Captured before the solve, where efc.state is not yet populated; the geometric test
  (penetration and nonzero stiffness) is the active set available at that point."""
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  dd = efc_D_in[worldid, efcid]
  if not (efc_pos_in[worldid, efcid] < 0.0 and dd > 0.0):
    return
  rownnz = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]
  for a in range(rownnz):
    ia = efc_J_colind_in[worldid, 0, rowadr + a]
    va = efc_J_in[worldid, 0, rowadr + a]
    for b in range(rownnz):
      ib = efc_J_colind_in[worldid, 0, rowadr + b]
      vb = efc_J_in[worldid, 0, rowadr + b]
      wp.atomic_add(h_out, worldid, ia, ib, dd * va * vb)


def capture_contact_adjoint_state(m: types.Model, d: types.Data):
  """Rebuild and snapshot the active-set Hessian + efc quantities at record time (forward).

  Returns None when not applicable (no constraints, non-Newton, dense path, or H not retained),
  in which case the backward falls back to d.solver_h."""
  if not (m.is_sparse and d.njmax > 0 and m.opt.solver == types.SolverType.NEWTON):
    return None
  from mujoco_warp._src import solver as _S
  done = wp.zeros(d.nworld, dtype=bool)
  H = wp.zeros((d.nworld, m.nv_pad, m.nv_pad), dtype=float)
  wp.launch(_S._update_gradient_init_h_sparse, dim=(d.nworld, m.nv_pad, m.nv_pad),
            inputs=[m.nv, m.M_elemid, d.M, done], outputs=[H])
  wp.launch(_jtdaj_geom_kernel, dim=(d.nworld, d.njmax),
            inputs=[d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.D, d.efc.pos],
            outputs=[H])
  return {
    "H": H,
    "nefc": wp.clone(d.nefc),
    "J": wp.clone(d.efc.J),
    "D": wp.clone(d.efc.D),
    "pos": wp.clone(d.efc.pos),
    "state": wp.clone(d.efc.state),
    "J_rownnz": wp.clone(d.efc.J_rownnz),
    "J_rowadr": wp.clone(d.efc.J_rowadr),
    "J_colind": wp.clone(d.efc.J_colind),
    "efc_id": wp.clone(d.efc.id),
  }


@wp.kernel
def _efc_aref_grad_kernel(
  nefc_in: wp.array(dtype=int),
  efc_J_rownnz_in: wp.array2d(dtype=int),
  efc_J_rowadr_in: wp.array2d(dtype=int),
  efc_J_colind_in: wp.array3d(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_pos_in: wp.array2d(dtype=float),
  v_in: wp.array2d(dtype=float),
  efc_aref_grad_out: wp.array2d(dtype=float),
):
  """adj_aref[i] = D[i] * (J[i,:] . v) on the penetrating active set (pos<0, D>0)."""
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  dd = efc_D_in[worldid, efcid]
  if not (efc_pos_in[worldid, efcid] < 0.0 and dd > 0.0):
    efc_aref_grad_out[worldid, efcid] = 0.0
    return
  rownnz = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]
  jv = float(0.0)
  for k in range(rownnz):
    col = efc_J_colind_in[worldid, 0, rowadr + k]
    jv += efc_J_in[worldid, 0, rowadr + k] * v_in[worldid, col]
  efc_aref_grad_out[worldid, efcid] = dd * jv


@wp.kernel
def _qvel_contact_dissipation_kernel(
  nefc_in: wp.array(dtype=int),
  efc_J_rownnz_in: wp.array2d(dtype=int),
  efc_J_rowadr_in: wp.array2d(dtype=int),
  efc_J_colind_in: wp.array3d(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_pos_in: wp.array2d(dtype=float),
  efc_id_in: wp.array2d(dtype=int),
  contact_solref_in: wp.array(dtype=wp.vec2),
  contact_solimp_in: wp.array(dtype=types.vec5),
  opt_timestep_in: wp.array(dtype=float),
  opt_disableflags: int,
  v_in: wp.array2d(dtype=float),
  qvel_grad_out: wp.array2d(dtype=float),
  efc_aref_grad_out: wp.array2d(dtype=float),
  efc_vel_grad_out: wp.array2d(dtype=float),
):
  """Inject the contact velocity-dissipation adjoint directly into qvel.grad.

  The constraint solve qacc = H^{-1}(qfrc + J_A^T D_A aref_A) carries a Baumgarte term
  aref_i = -k*imp*pos_i - b_i*vel_i with vel_i = J_i . qvel. Its reverse-mode contribution to
  qvel is adj_qvel = -sum_{i in A} b_i D_i (J_i . v) J_i, with v = H^{-1} adj_qacc, over the
  active set (penetrating pos<0 and stiff D>0).

  We scatter this J^T product ourselves (atomic_add over the sparse row, indexed by colind),
  rather than route it through efc.vel.grad / efc.aref.grad. The native vel->qvel autodiff of the
  sparse-J assembly indexes the velocity gradient by stored row position instead of efc.J column
  index, so it mis-projects rows that couple more than one DOF (the pyramidal friction cone, where
  the two edges then cancel and the tangential dissipation never reaches qvel). We only handle
  those multi-column rows here (effectively-single-DOF rows are routed correctly per substep by
  the native chain) and, for the rows we do handle, zero their efc.aref.grad / efc.vel.grad so the
  native path cannot double count. efc.aref.grad was used as scratch by the pos kernel before this.
  """
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  dd = efc_D_in[worldid, efcid]
  if not (efc_pos_in[worldid, efcid] < 0.0 and dd > 0.0):
    return
  rownnz = efc_J_rownnz_in[worldid, efcid]
  rowadr = efc_J_rowadr_in[worldid, efcid]
  # Count the genuinely nonzero columns. Rows that effectively touch a single DOF (a normal row
  # of a multi-DOF body still stores an explicit zero for the other DOF, so rownnz over-counts)
  # are routed correctly per substep by the native efc.aref/efc.vel autodiff chain, so leave them
  # to it. Only rows that genuinely couple more than one DOF (the pyramidal friction cone) are
  # column-mis-projected by the native scatter and need this direct J^T injection.
  nnz_real = int(0)
  for k in range(rownnz):
    if efc_J_in[worldid, 0, rowadr + k] != 0.0:
      nnz_real += 1
  if nnz_real <= 1:
    return
  conid = efc_id_in[worldid, efcid]
  solref = contact_solref_in[conid]
  solimp = contact_solimp_in[conid]
  timestep = opt_timestep_in[worldid % opt_timestep_in.shape[0]]
  dmax = wp.clamp(solimp[1], types.MJ_MINIMP, types.MJ_MAXIMP)
  timeconst = solref[0]
  if not (opt_disableflags & int(types.DisableBit.REFSAFE.value)):
    timeconst = wp.max(timeconst, 2.0 * timestep)
  b = 2.0 / (dmax * timeconst)
  b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)
  jv = float(0.0)
  for k in range(rownnz):
    col = efc_J_colind_in[worldid, 0, rowadr + k]
    jv += efc_J_in[worldid, 0, rowadr + k] * v_in[worldid, col]
  factor = -b * dd * jv
  for k in range(rownnz):
    col = efc_J_colind_in[worldid, 0, rowadr + k]
    wp.atomic_add(qvel_grad_out, worldid, col, factor * efc_J_in[worldid, 0, rowadr + k])
  # Kill the native velocity propagation for this row so it is not counted twice.
  efc_aref_grad_out[worldid, efcid] = 0.0
  efc_vel_grad_out[worldid, efcid] = 0.0

