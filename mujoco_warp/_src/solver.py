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

from math import ceil
from math import sqrt

import warp as wp

from mujoco_warp._src import math
from mujoco_warp._src import smooth
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_func
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})

_BLOCK_CHOLESKY_DIM = 32


@wp.func
def _rescale(nv: int, stat_meaninertia: float, value: float) -> float:
  return value / (stat_meaninertia * float(nv))


@wp.func
def _in_bracket(x: wp.vec3, y: wp.vec3) -> bool:
  return (x[1] < y[1] and y[1] < 0.0) or (x[1] > y[1] and y[1] > 0.0)


@wp.func
def _eval_pt_direct(jaref: float, jv: float, d: float, alpha: float) -> wp.vec3:
  """Eval quadratic constraint, return (cost, grad, hessian)."""
  x = jaref + alpha * jv
  jvD = jv * d
  return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)


@wp.func
def _eval_pt_direct_alpha_zero(jaref: float, jv: float, d: float) -> wp.vec3:
  """Eval quadratic constraint at alpha=0."""
  jvD = jv * d
  return wp.vec3(0.5 * d * jaref * jaref, jvD * jaref, jv * jvD)


@wp.func
def _eval_pt_direct_3alphas(
  jaref: float, jv: float, d: float, lo_alpha: float, hi_alpha: float, mid_alpha: float
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval quadratic constraint for 3 alphas."""
  x_lo = jaref + lo_alpha * jv
  x_hi = jaref + hi_alpha * jv
  x_mid = jaref + mid_alpha * jv
  jvD = jv * d
  hessian = jv * jvD
  half_d = 0.5 * d
  return (
    wp.vec3(half_d * x_lo * x_lo, jvD * x_lo, hessian),
    wp.vec3(half_d * x_hi * x_hi, jvD * x_hi, hessian),
    wp.vec3(half_d * x_mid * x_mid, jvD * x_mid, hessian),
  )


@wp.func
def _eval_cost(quad: wp.vec3, alpha: float) -> float:
  return alpha * alpha * quad[2] + alpha * quad[1] + quad[0]


@wp.func
def _eval_pt(quad: wp.vec3, alpha: float) -> wp.vec3:
  """Eval quad polynomial at alpha, return (cost, grad, hessian)."""
  aq2 = alpha * quad[2]
  return wp.vec3(
    alpha * aq2 + alpha * quad[1] + quad[0],
    2.0 * aq2 + quad[1],
    2.0 * quad[2],
  )


@wp.func
def _eval_pt_3alphas(quad: wp.vec3, lo_alpha: float, hi_alpha: float, mid_alpha: float) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval quad polynomial for 3 alphas."""
  q0, q1, q2 = quad[0], quad[1], quad[2]
  hessian = 2.0 * q2
  lo_aq2 = lo_alpha * q2
  hi_aq2 = hi_alpha * q2
  mid_aq2 = mid_alpha * q2
  return (
    wp.vec3(lo_alpha * lo_aq2 + lo_alpha * q1 + q0, 2.0 * lo_aq2 + q1, hessian),
    wp.vec3(hi_alpha * hi_aq2 + hi_alpha * q1 + q0, 2.0 * hi_aq2 + q1, hessian),
    wp.vec3(mid_alpha * mid_aq2 + mid_alpha * q1 + q0, 2.0 * mid_aq2 + q1, hessian),
  )


@wp.func
def _eval_frictionloss_pt(x: float, f: float, rf: float, jv: float, d: float) -> wp.vec3:
  """Eval frictionloss and return (cost, grad, hessian). x = Jaref + alpha * jv."""
  if (-rf < x) and (x < rf):
    jvD = jv * d
    return wp.vec3(0.5 * d * x * x, jvD * x, jv * jvD)
  elif x <= -rf:
    return wp.vec3(f * (-0.5 * rf - x), -f * jv, 0.0)
  else:
    return wp.vec3(f * (-0.5 * rf + x), f * jv, 0.0)


@wp.func
def _eval_frictionloss_pt_one(x: float, f: float, rf: float, half_d: float, jvD: float, hessian: float, f_jv: float) -> wp.vec3:
  """Eval frictionloss with precomputed shared values."""
  if (-rf < x) and (x < rf):
    return wp.vec3(half_d * x * x, jvD * x, hessian)
  elif x <= -rf:
    return wp.vec3(f * (-0.5 * rf - x), -f_jv, 0.0)
  else:
    return wp.vec3(f * (-0.5 * rf + x), f_jv, 0.0)


@wp.func
def _eval_frictionloss_pt_3alphas(
  x_lo: float, x_hi: float, x_mid: float, f: float, rf: float, jv: float, d: float
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Eval frictionloss for 3 x values with shared precomputation."""
  jvD = jv * d
  half_d = 0.5 * d
  hessian = jv * jvD
  f_jv = f * jv
  return (
    _eval_frictionloss_pt_one(x_lo, f, rf, half_d, jvD, hessian, f_jv),
    _eval_frictionloss_pt_one(x_hi, f, rf, half_d, jvD, hessian, f_jv),
    _eval_frictionloss_pt_one(x_mid, f, rf, half_d, jvD, hessian, f_jv),
  )


@wp.func
def _eval_elliptic(
  # In:
  impratio_invsqrt: float,
  friction: types.vec5,
  quad: wp.vec3,
  quad1: wp.vec3,
  quad2: wp.vec3,
  alpha: float,
) -> wp.vec3:
  mu = friction[0] * impratio_invsqrt

  u0 = quad1[0]
  v0 = quad1[1]
  uu = quad1[2]
  uv = quad2[0]
  vv = quad2[1]
  dm = quad2[2]

  # compute N, Tsqr
  N = u0 + alpha * v0
  Tsqr = uu + alpha * (2.0 * uv + alpha * vv)

  # no tangential force: top or bottom zone
  if Tsqr <= 0.0:
    # bottom zone: quadratic cost
    if N < 0.0:
      return _eval_pt(quad, alpha)

    # top zone: nothing to do
  # otherwise regular processing
  else:
    # tangential force
    T = wp.sqrt(Tsqr)

    # N >= mu * T : top zone
    if N >= mu * T:
      # nothing to do
      pass
    # mu * N + T <= 0 : bottom zone
    elif mu * N + T <= 0.0:
      return _eval_pt(quad, alpha)

    # otherwise middle zone
    else:
      # derivatives
      N1 = v0
      T1 = (uv + alpha * vv) / T
      T2 = vv / T - (uv + alpha * vv) * T1 / (T * T)

      # add to cost
      cost = wp.vec3(
        0.5 * dm * (N - mu * T) * (N - mu * T),
        dm * (N - mu * T) * (N1 - mu * T1),
        dm * ((N1 - mu * T1) * (N1 - mu * T1) + (N - mu * T) * (-mu * T2)),
      )

      return cost

  return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _log_scale(min_value: float, max_value: float, num_values: int, i: int) -> float:
  step = (wp.log(max_value) - wp.log(min_value)) / wp.max(1.0, float(num_values - 1))
  return wp.exp(wp.log(min_value) + float(i) * step)


@wp.kernel
def linesearch_parallel_fused(
  # Model:
  opt_ls_iterations: int,
  opt_impratio_invsqrt: wp.array(dtype=float),
  opt_ls_parallel_min_step: float,
  # Data in:
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_efc_address_in: wp.array2d(dtype=int),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  efc_D_in: wp.array2d(dtype=float),
  efc_frictionloss_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_jv_in: wp.array2d(dtype=float),
  efc_quad_in: wp.array2d(dtype=wp.vec3),
  efc_quad_gauss_in: wp.array(dtype=wp.vec3),
  efc_done_in: wp.array(dtype=bool),
  njmax_in: int,
  nacon_in: wp.array(dtype=int),
  # Out:
  cost_out: wp.array2d(dtype=float),
):
  worldid, alphaid = wp.tid()

  if efc_done_in[worldid]:
    return

  alpha = _log_scale(opt_ls_parallel_min_step, 1.0, opt_ls_iterations, alphaid)

  out = _eval_cost(efc_quad_gauss_in[worldid], alpha)

  ne = ne_in[worldid]
  nf = nf_in[worldid]

  # TODO(team): _eval with option to only compute cost
  for efcid in range(min(njmax_in, nefc_in[worldid])):
    # equality
    if efcid < ne:
      out += _eval_cost(efc_quad_in[worldid, efcid], alpha)
    # friction
    elif efcid < ne + nf:
      # search point, friction loss, bound (rf)
      start = efc_Jaref_in[worldid, efcid]
      dir = efc_jv_in[worldid, efcid]
      x = start + alpha * dir
      f = efc_frictionloss_in[worldid, efcid]
      rf = math.safe_div(f, efc_D_in[worldid, efcid])

      # -bound < x < bound : quadratic
      if (-rf < x) and (x < rf):
        quad = efc_quad_in[worldid, efcid]
      # x < -bound: linear negative
      elif x <= -rf:
        quad = wp.vec3(f * (-0.5 * rf - start), -f * dir, 0.0)
      # bound < x : linear positive
      else:
        quad = wp.vec3(f * (-0.5 * rf + start), f * dir, 0.0)

      out += _eval_cost(quad, alpha)
    # limit and contact
    elif efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC:
      # extract contact info
      conid = efc_id_in[worldid, efcid]

      if conid >= nacon_in[0]:
        continue

      efcid0 = contact_efc_address_in[conid, 0]
      if efcid != efcid0:
        continue

      friction = contact_friction_in[conid]
      mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

      # unpack quad
      efcid1 = contact_efc_address_in[conid, 1]
      efcid2 = contact_efc_address_in[conid, 2]
      u0 = efc_quad_in[worldid, efcid1][0]
      v0 = efc_quad_in[worldid, efcid1][1]
      uu = efc_quad_in[worldid, efcid1][2]
      uv = efc_quad_in[worldid, efcid2][0]
      vv = efc_quad_in[worldid, efcid2][1]
      dm = efc_quad_in[worldid, efcid2][2]

      # compute N, Tsqr
      N = u0 + alpha * v0
      Tsqr = uu + alpha * (2.0 * uv + alpha * vv)

      # no tangential force: top or bottom zone
      if Tsqr <= 0.0:
        # bottom zone: quadratic cost
        if N < 0.0:
          out += _eval_cost(efc_quad_in[worldid, efcid], alpha)
      # otherwise regular processing
      else:
        # tangential force
        T = wp.sqrt(Tsqr)

        # N >= mu * T : top zone
        if N >= mu * T:
          # nothing to do
          pass
        # mu * N + T <= 0 : bottom zone
        elif mu * N + T <= 0.0:
          out += _eval_cost(efc_quad_in[worldid, efcid], alpha)
        # otherwise middle zone
        else:
          out += 0.5 * dm * (N - mu * T) * (N - mu * T)
    else:
      # search point
      x = efc_Jaref_in[worldid, efcid] + alpha * efc_jv_in[worldid, efcid]

      # active
      if x < 0.0:
        out += _eval_cost(efc_quad_in[worldid, efcid], alpha)

  cost_out[worldid, alphaid] = out


@wp.kernel
def linesearch_parallel_best_alpha(
  # Model:
  opt_ls_iterations: int,
  opt_ls_parallel_min_step: float,
  # Data in:
  efc_done_in: wp.array(dtype=bool),
  # In:
  cost_in: wp.array2d(dtype=float),
  # Data out:
  efc_alpha_out: wp.array(dtype=float),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  bestid = int(0)
  best_cost = float(wp.inf)
  for i in range(opt_ls_iterations):
    cost = cost_in[worldid, i]
    if cost < best_cost:
      best_cost = cost
      bestid = i

  efc_alpha_out[worldid] = _log_scale(opt_ls_parallel_min_step, 1.0, opt_ls_iterations, bestid)


def _linesearch_parallel(m: types.Model, d: types.Data, cost: wp.array2d(dtype=float)):
  """Parallel linesearch with setup and teardown kernels."""
  dofs_per_thread = 20 if m.nv > 50 else 50
  threads_per_efc = ceil(m.nv / dofs_per_thread)

  # quad_gauss = [gauss, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv]
  if threads_per_efc > 1:
    d.efc.quad_gauss.zero_()

  wp.launch(
    linesearch_prepare_gauss(m.nv, dofs_per_thread),
    dim=(d.nworld, threads_per_efc),
    inputs=[d.qfrc_smooth, d.efc.Ma, d.efc.search, d.efc.gauss, d.efc.mv, d.efc.done],
    outputs=[d.efc.quad_gauss],
  )

  # quad = [0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D]
  wp.launch(
    linesearch_prepare_quad,
    dim=(d.nworld, d.njmax),
    inputs=[
      m.opt.impratio_invsqrt,
      d.nefc,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.Jaref,
      d.efc.jv,
      d.efc.done,
      d.nacon,
    ],
    outputs=[d.efc.quad],
  )

  wp.launch(
    linesearch_parallel_fused,
    dim=(d.nworld, m.opt.ls_iterations),
    inputs=[
      m.opt.ls_iterations,
      m.opt.impratio_invsqrt,
      m.opt.ls_parallel_min_step,
      d.ne,
      d.nf,
      d.nefc,
      d.contact.friction,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.frictionloss,
      d.efc.Jaref,
      d.efc.jv,
      d.efc.quad,
      d.efc.quad_gauss,
      d.efc.done,
      d.njmax,
      d.nacon,
    ],
    outputs=[cost],
  )

  wp.launch(
    linesearch_parallel_best_alpha,
    dim=(d.nworld),
    inputs=[m.opt.ls_iterations, m.opt.ls_parallel_min_step, d.efc.done, cost],
    outputs=[d.efc.alpha],
  )

  # Teardown: update qacc, Ma, Jaref
  wp.launch(
    linesearch_qacc_ma,
    dim=(d.nworld, m.nv),
    inputs=[d.efc.search, d.efc.mv, d.efc.alpha, d.efc.done],
    outputs=[d.qacc, d.efc.Ma],
  )

  wp.launch(
    linesearch_jaref,
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, d.efc.jv, d.efc.alpha, d.efc.done],
    outputs=[d.efc.Jaref],
  )


# kernel_analyzer: off
@wp.func
def _compute_efc_eval_pt_pyramidal(
  efcid: int,
  alpha: float,
  ne: int,
  nf: int,
  # Per-row data:
  efc_D: float,
  efc_frictionloss: wp.array(dtype=float),
  efc_Jaref: float,
  efc_jv: float,
) -> wp.vec3:
  """Compute for pyramidal cones (no elliptic contact data needed)."""
  # Limit/other constraint
  if efcid >= ne + nf:
    x = efc_Jaref + alpha * efc_jv
    if x < 0.0:
      return _eval_pt_direct(efc_Jaref, efc_jv, efc_D, alpha)
    return wp.vec3(0.0)

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    f = efc_frictionloss[efcid]
    x = efc_Jaref + alpha * efc_jv
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(x, f, rf, efc_jv, efc_D)

  # Equality constraint
  return _eval_pt_direct(efc_Jaref, efc_jv, efc_D, alpha)


@wp.func
def _compute_efc_eval_pt_elliptic(
  efcid: int,
  alpha: float,
  ne: int,
  nf: int,
  impratio_invsqrt: float,
  # Per-row data (arrays for deferred load):
  efc_type: int,
  efc_D_in: wp.array(dtype=float),
  efc_frictionloss: wp.array(dtype=float),
  efc_Jaref: float,
  efc_jv: float,
  efc_quad: wp.vec3,
  # Contact data (for elliptic):
  contact_friction: types.vec5,
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> wp.vec3:
  """Compute for elliptic cones (includes elliptic contact data)."""
  # Contact/limit/other constraints
  if efcid >= ne + nf:
    # Contact elliptic
    if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # Not primary row
        return wp.vec3(0.0)
      return _eval_elliptic(impratio_invsqrt, contact_friction, efc_quad, quad1, quad2, alpha)

    # Limit/other constraint
    x = efc_Jaref + alpha * efc_jv
    if x < 0.0:
      return _eval_pt(efc_quad, alpha)
    return wp.vec3(0.0)

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    x = efc_Jaref + alpha * efc_jv
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(x, f, rf, efc_jv, efc_D)

  # Equality constraint
  return _eval_pt(efc_quad, alpha)


@wp.func
def _compute_efc_eval_pt_alpha_zero_pyramidal(
  efcid: int,
  ne: int,
  nf: int,
  # Per-row data:
  efc_D: float,
  efc_frictionloss: wp.array(dtype=float),
  efc_Jaref: float,
  efc_jv: float,
) -> wp.vec3:
  """Optimized version for alpha=0.0, pyramidal cones."""
  # Limit/other constraint
  if efcid >= ne + nf:
    if efc_Jaref < 0.0:
      return _eval_pt_direct_alpha_zero(efc_Jaref, efc_jv, efc_D)
    return wp.vec3(0.0)

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(efc_Jaref, f, rf, efc_jv, efc_D)

  # Equality constraint
  return _eval_pt_direct_alpha_zero(efc_Jaref, efc_jv, efc_D)


@wp.func
def _compute_efc_eval_pt_alpha_zero_elliptic(
  efcid: int,
  ne: int,
  nf: int,
  impratio_invsqrt: float,
  # Per-row data (arrays for deferred load):
  efc_type: int,
  efc_D_in: wp.array(dtype=float),
  efc_frictionloss: wp.array(dtype=float),
  efc_Jaref: float,
  efc_jv: float,
  efc_quad: wp.vec3,
  # Contact data (for elliptic):
  contact_friction: types.vec5,
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> wp.vec3:
  """Optimized version for alpha=0.0, elliptic cones."""
  # Contact/limit/other constraints
  if efcid >= ne + nf:
    # Contact elliptic
    if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # Not primary row
        return wp.vec3(0.0)
      return _eval_elliptic(impratio_invsqrt, contact_friction, efc_quad, quad1, quad2, 0.0)

    # Limit/other constraint
    if efc_Jaref < 0.0:
      return wp.vec3(efc_quad[0], efc_quad[1], 2.0 * efc_quad[2])
    return wp.vec3(0.0)

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt(efc_Jaref, f, rf, efc_jv, efc_D)

  # Equality constraint
  return wp.vec3(efc_quad[0], efc_quad[1], 2.0 * efc_quad[2])


@wp.func
def _compute_efc_eval_pt_3alphas_pyramidal(
  efcid: int,
  lo_alpha: float,
  hi_alpha: float,
  mid_alpha: float,
  ne: int,
  nf: int,
  # Per-row data:
  efc_D: float,
  efc_frictionloss: wp.array(dtype=float),
  efc_Jaref: float,
  efc_jv: float,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Compute (cost, gradient, hessian) for 3 alphas, pyramidal cones.

  Returns a tuple of 3 vec3s for (lo_alpha, hi_alpha, mid_alpha).
  Constraint types checked in order: limit/other -> friction -> equality.
  """
  # Limit/other constraints: active only when x < 0
  if efcid >= ne + nf:
    x_lo = efc_Jaref + lo_alpha * efc_jv
    x_hi = efc_Jaref + hi_alpha * efc_jv
    x_mid = efc_Jaref + mid_alpha * efc_jv
    pt_lo, pt_hi, pt_mid = _eval_pt_direct_3alphas(efc_Jaref, efc_jv, efc_D, lo_alpha, hi_alpha, mid_alpha)
    r_lo = wp.where(x_lo < 0.0, pt_lo, wp.vec3(0.0))
    r_hi = wp.where(x_hi < 0.0, pt_hi, wp.vec3(0.0))
    r_mid = wp.where(x_mid < 0.0, pt_mid, wp.vec3(0.0))
    return (r_lo, r_hi, r_mid)

  # Friction constraint - needs quad for frictionloss computation
  if efcid >= ne:
    x_lo = efc_Jaref + lo_alpha * efc_jv
    x_hi = efc_Jaref + hi_alpha * efc_jv
    x_mid = efc_Jaref + mid_alpha * efc_jv
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, efc_jv, efc_D)

  # Equality constraint: always active
  return _eval_pt_direct_3alphas(efc_Jaref, efc_jv, efc_D, lo_alpha, hi_alpha, mid_alpha)


@wp.func
def _compute_efc_eval_pt_3alphas_elliptic(
  efcid: int,
  lo_alpha: float,
  hi_alpha: float,
  mid_alpha: float,
  ne: int,
  nf: int,
  impratio_invsqrt: float,
  # Per-row data (arrays for deferred load):
  efc_type: int,
  efc_D_in: wp.array(dtype=float),
  efc_frictionloss: wp.array(dtype=float),
  efc_Jaref: float,
  efc_jv: float,
  efc_quad: wp.vec3,
  # Contact data (for elliptic):
  contact_friction: types.vec5,
  efc_address0: int,
  quad1: wp.vec3,
  quad2: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Compute (cost, gradient, hessian) for 3 alphas, elliptic cones.

  Returns a tuple of 3 vec3s for (lo_alpha, hi_alpha, mid_alpha).
  Constraint types checked in order: contact elliptic/limit/other -> friction -> equality.
  """
  # x = search point, needed for friction and limit constraints
  x_lo = efc_Jaref + lo_alpha * efc_jv
  x_hi = efc_Jaref + hi_alpha * efc_jv
  x_mid = efc_Jaref + mid_alpha * efc_jv

  # Contact/limit/other constraints
  if efcid >= ne + nf:
    # Contact elliptic: uses special elliptic cone evaluation
    if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
      if efcid != efc_address0:  # secondary rows contribute nothing
        return (wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0))
      return (
        _eval_elliptic(impratio_invsqrt, contact_friction, efc_quad, quad1, quad2, lo_alpha),
        _eval_elliptic(impratio_invsqrt, contact_friction, efc_quad, quad1, quad2, hi_alpha),
        _eval_elliptic(impratio_invsqrt, contact_friction, efc_quad, quad1, quad2, mid_alpha),
      )

    # Limit/other constraints: active only when x < 0
    pt_lo, pt_hi, pt_mid = _eval_pt_3alphas(efc_quad, lo_alpha, hi_alpha, mid_alpha)
    r_lo = wp.where(x_lo < 0.0, pt_lo, wp.vec3(0.0))
    r_hi = wp.where(x_hi < 0.0, pt_hi, wp.vec3(0.0))
    r_mid = wp.where(x_mid < 0.0, pt_mid, wp.vec3(0.0))
    return (r_lo, r_hi, r_mid)

  # Friction constraint - load D and frictionloss only here
  if efcid >= ne:
    efc_D = efc_D_in[efcid]
    f = efc_frictionloss[efcid]
    rf = math.safe_div(f, efc_D)
    return _eval_frictionloss_pt_3alphas(x_lo, x_hi, x_mid, f, rf, efc_jv, efc_D)

  # Equality constraint: always active
  return _eval_pt_3alphas(efc_quad, lo_alpha, hi_alpha, mid_alpha)


# kernel_analyzer: on

# =============================================================================
# Iterative Linesearch
# =============================================================================
#
# Iterative linesearch implementation using Warp's tiled execution model with
# parallel reductions over constraint (EFC) rows.
#
# Key optimizations:
#
# 1. KERNEL FUSION - Reduces kernel launch overhead by combining:
#    - linesearch_jv_fused: jv = J @ search (for small nv <= 50)
#    - linesearch_prepare_quad: quad coefficients (pyramidal: computed directly,
#      elliptic: computed in a prepare phase with __syncthreads barrier)
#    - linesearch_prepare_gauss: quad_gauss via tile reduction over DOFs
#    - linesearch_qacc_ma: qacc and Ma updates at kernel end
#    - linesearch_jaref: Jaref update at kernel end
#
# 2. PARALLEL REDUCTIONS - Uses wp.tile_reduce for summing cost/gradient/hessian
#    contributions across EFC rows within each world. The main iteration loop
#    packs 3 vec3 reductions into a single mat33 reduction for efficiency.
#
# 3. COMPILE-TIME SPECIALIZATION via factory parameters:
#    - cone_type: Eliminates elliptic cone branches for pyramidal-only models
#    - ls_iterations: Enables loop unrolling for the main bracket search
#    - fuse_jv: Conditionally includes jv computation based on nv size
#
# 4. DIRECT EVALUATION (pyramidal only) - For equality and limit constraints,
#    computes cost/gradient/hessian directly from (Jaref, jv, efc_D, alpha)
#    without intermediate quad coefficients, using _eval_pt_direct functions.
#
# 5. BATCHED 3-ALPHA EVALUATION - The main iteration loop evaluates 3 alpha
#    values per iteration (lo_next, hi_next, mid). Instead of calling
#    _compute_efc_eval_pt 3 times per constraint row (which would repeat
#    constraint type checks and data loads), we use _compute_efc_eval_pt_3alphas
#    which:
#    - Performs constraint type branching once per row
#    - Loads efc_D, efc_frictionloss, contact data once
#    - Computes x = Jaref + alpha * jv for all 3 alphas
#    - For pyramidal direct evaluation: shares jvD = jv * efc_D, hessian = jv * jvD
#    - For quad-based evaluation: uses _eval_pt_3alphas which computes the
#      constant hessian (2.0 * quad[2]) once and reuses for all 3 alphas
#
# 6. DEFERRED DATA LOADING - efc_D and efc_frictionloss are only loaded
#    inside the constraint branches where they're needed, reducing register
#    pressure for other constraint types.
#
# Trade-offs:
# - Requires block synchronization (__syncthreads) for elliptic quad preparation
# - Separate kernel compilation for each (block_dim, ls_iterations, cone_type,
#   fuse_jv) combination (cached by Warp)
#
# Optimizations attempted but not beneficial:
# - Caching EFC data (Jaref, jv, quad, etc.) in shared memory tiles for reuse
#   across the p0, lo_in, and main iteration loops.
#
# =============================================================================


@cache_kernel
def linesearch_iterative(ls_iterations: int, cone_type: types.ConeType, fuse_jv: bool):
  """Factory for iterative linesearch kernel.

  Args:
    block_dim: Number of threads per block for tile reductions.
    ls_iterations: Max linesearch iterations (compile-time constant for loop optimization).
    cone_type: Friction cone type (PYRAMIDAL or ELLIPTIC) for compile-time optimization.
    fuse_jv: Whether to compute jv = J @ search in-kernel (efficient for small nv).
  """
  LS_ITERATIONS = ls_iterations
  IS_ELLIPTIC = cone_type == types.ConeType.ELLIPTIC
  FUSE_JV = fuse_jv

  # Native snippet for CUDA __syncthreads()
  @wp.func_native(snippet="WP_TILE_SYNC();")
  def _syncthreads():
    pass

  # Select specialized helper functions based on cone type
  if IS_ELLIPTIC:
    _compute_efc_eval_pt = _compute_efc_eval_pt_elliptic
    _compute_efc_eval_pt_alpha_zero = _compute_efc_eval_pt_alpha_zero_elliptic
    _compute_efc_eval_pt_3alphas = _compute_efc_eval_pt_3alphas_elliptic
  else:
    _compute_efc_eval_pt = _compute_efc_eval_pt_pyramidal
    _compute_efc_eval_pt_alpha_zero = _compute_efc_eval_pt_alpha_zero_pyramidal
    _compute_efc_eval_pt_3alphas = _compute_efc_eval_pt_3alphas_pyramidal

  # kernel_analyzer: off
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    opt_tolerance: wp.array(dtype=float),
    opt_ls_tolerance: wp.array(dtype=float),
    opt_impratio_invsqrt: wp.array(dtype=float),
    stat_meaninertia: float,
    # Data in:
    ne_in: wp.array(dtype=int),
    nf_in: wp.array(dtype=int),
    nefc_in: wp.array(dtype=int),
    contact_friction_in: wp.array(dtype=types.vec5),
    contact_dim_in: wp.array(dtype=int),
    contact_efc_address_in: wp.array2d(dtype=int),
    efc_type_in: wp.array2d(dtype=int),
    efc_id_in: wp.array2d(dtype=int),
    efc_D_in: wp.array2d(dtype=float),
    efc_frictionloss_in: wp.array2d(dtype=float),
    efc_Jaref_in: wp.array2d(dtype=float),
    efc_search_dot_in: wp.array(dtype=float),
    efc_J_in: wp.array3d(dtype=float),
    efc_gauss_in: wp.array(dtype=float),
    qfrc_smooth_in: wp.array2d(dtype=float),
    efc_done_in: wp.array(dtype=bool),
    njmax_in: int,
    nacon_in: wp.array(dtype=int),
    efc_search_in: wp.array2d(dtype=float),
    efc_mv_in: wp.array2d(dtype=float),
    # Data in/out:
    efc_quad_inout: wp.array2d(dtype=wp.vec3),
    efc_jv_inout: wp.array2d(dtype=float),
    # Data out:
    qacc_out: wp.array2d(dtype=float),
    efc_Ma_out: wp.array2d(dtype=float),
    efc_Jaref_out: wp.array2d(dtype=float),
  ):
    worldid, tid = wp.tid()

    if efc_done_in[worldid]:
      return

    ne = ne_in[worldid]
    nf = nf_in[worldid]
    nefc = wp.min(njmax_in, nefc_in[worldid])

    # jv = J @ search (fused for small nv)
    if wp.static(FUSE_JV):
      for efcid in range(tid, nefc, wp.block_dim()):
        jv = float(0.0)
        for i in range(nv):
          jv += efc_J_in[worldid, efcid, i] * efc_search_in[worldid, i]
        efc_jv_inout[worldid, efcid] = jv

      _syncthreads()  # ensure all jv values are written before reading

    # quad coefficients (elliptic only, requires barrier sync)
    if wp.static(IS_ELLIPTIC):
      # elliptic-only config values
      impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
      nacon = nacon_in[0]

      for efcid in range(tid, nefc, wp.block_dim()):
        Jaref = efc_Jaref_in[worldid, efcid]
        jv = efc_jv_inout[worldid, efcid]
        efc_D = efc_D_in[worldid, efcid]

        # scalar quadratic coefficients
        jvD = jv * efc_D
        quad = wp.vec3(0.5 * Jaref * Jaref * efc_D, jvD * Jaref, 0.5 * jv * jvD)

        # non-contact constraints: write quad immediately
        if efc_type_in[worldid, efcid] != types.ConstraintType.CONTACT_ELLIPTIC:
          efc_quad_inout[worldid, efcid] = quad
        else:
          # CONTACT_ELLIPTIC: only primary row of active contacts writes
          conid = efc_id_in[worldid, efcid]
          if conid < nacon:
            efcid0 = contact_efc_address_in[conid, 0]
            if efcid == efcid0:
              # primary row: accumulate secondary rows and write quad, quad1, quad2
              dim = contact_dim_in[conid]
              friction = contact_friction_in[conid]
              mu = friction[0] * impratio_invsqrt

              u0 = Jaref * mu
              v0 = jv * mu

              uu = float(0.0)
              uv = float(0.0)
              vv = float(0.0)
              for j in range(1, dim):
                efcidj = contact_efc_address_in[conid, j]
                if efcidj >= 0:
                  jvj = efc_jv_inout[worldid, efcidj]
                  jarefj = efc_Jaref_in[worldid, efcidj]
                  dj = efc_D_in[worldid, efcidj]
                  DJj = dj * jarefj

                  quad += wp.vec3(0.5 * jarefj * DJj, jvj * DJj, 0.5 * jvj * dj * jvj)

                  # rescale to make primal cone circular
                  frictionj = friction[j - 1]
                  uj = jarefj * frictionj
                  vj = jvj * frictionj

                  uu += uj * uj
                  uv += uj * vj
                  vv += vj * vj

              efc_quad_inout[worldid, efcid] = quad

              efcid1 = contact_efc_address_in[conid, 1]
              efc_quad_inout[worldid, efcid1] = wp.vec3(u0, v0, uu)

              mu2 = mu * mu
              efcid2 = contact_efc_address_in[conid, 2]
              efc_quad_inout[worldid, efcid2] = wp.vec3(uv, vv, efc_D / (mu2 * (1.0 + mu2)))

      _syncthreads()  # ensure all quads are written before reading

    # gtol (tolerance values loaded here, deferred from kernel start)
    tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
    ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]
    snorm = wp.sqrt(efc_search_dot_in[worldid])
    scale = stat_meaninertia * wp.float(nv)
    gtol = tolerance * ls_tolerance * snorm * scale

    # p0 via parallel reduction
    local_p0 = wp.vec3(0.0)
    for efcid in range(tid, nefc, wp.block_dim()):
      if wp.static(IS_ELLIPTIC):
        efc_type = efc_type_in[worldid, efcid]
        efc_id = 0
        contact_friction = types.vec5(0.0)
        efc_addr0 = int(0)
        quad1 = wp.vec3(0.0)
        quad2 = wp.vec3(0.0)

        if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
          efc_id = efc_id_in[worldid, efcid]
          contact_friction = contact_friction_in[efc_id]
          efc_addr0 = contact_efc_address_in[efc_id, 0]
          efc_addr1 = contact_efc_address_in[efc_id, 1]
          efc_addr2 = contact_efc_address_in[efc_id, 2]
          quad1 = efc_quad_inout[worldid, efc_addr1]
          quad2 = efc_quad_inout[worldid, efc_addr2]

        local_p0 += _compute_efc_eval_pt_alpha_zero(
          efcid,
          ne,
          nf,
          impratio_invsqrt,
          efc_type,
          efc_D_in[worldid],
          efc_frictionloss_in[worldid],
          efc_Jaref_in[worldid, efcid],
          efc_jv_inout[worldid, efcid],
          efc_quad_inout[worldid, efcid],
          contact_friction,
          efc_addr0,
          quad1,
          quad2,
        )
      else:
        # direct evaluation for pyramidal cones (no intermediate quad)
        local_p0 += _compute_efc_eval_pt_alpha_zero(
          efcid,
          ne,
          nf,
          efc_D_in[worldid, efcid],
          efc_frictionloss_in[worldid],
          efc_Jaref_in[worldid, efcid],
          efc_jv_inout[worldid, efcid],
        )

    # at this point, every thread has computed some contributions to p0 in local_p0
    # we now create a tile of all local_p0 contributions and reduce them to a single value
    # this is done in parallel using a tile reduction
    p0_tile = wp.tile(local_p0, preserve_type=True)
    p0_sum = wp.tile_reduce(wp.add, p0_tile)

    # quad_gauss = [gauss, search.T @ Ma - search.T @ qfrc_smooth, 0.5 * search.T @ mv]
    local_gauss = wp.vec2(0.0)  # vec2 since component 0 is constant (efc_gauss_in)
    for dofid in range(tid, nv, wp.block_dim()):
      search = efc_search_in[worldid, dofid]
      local_gauss += wp.vec2(
        search * (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]),
        0.5 * search * efc_mv_in[worldid, dofid],
      )

    gauss_tile = wp.tile(local_gauss, preserve_type=True)
    gauss_sum = wp.tile_reduce(wp.add, gauss_tile)
    gauss_reduced = gauss_sum[0]
    efc_quad_gauss = wp.vec3(efc_gauss_in[worldid], gauss_reduced[0], gauss_reduced[1])

    # add quad_gauss contribution to p0
    p0 = wp.vec3(efc_quad_gauss[0], efc_quad_gauss[1], 2.0 * efc_quad_gauss[2]) + p0_sum[0]

    # lo_in at lo_alpha_in = -p0[1] / p0[2]
    lo_alpha_in = -math.safe_div(p0[1], p0[2])

    local_lo_in = wp.vec3(0.0)
    for efcid in range(tid, nefc, wp.block_dim()):
      if wp.static(IS_ELLIPTIC):
        efc_type = efc_type_in[worldid, efcid]
        efc_id = 0
        contact_friction = types.vec5(0.0)
        efc_addr0 = int(0)
        quad1 = wp.vec3(0.0)
        quad2 = wp.vec3(0.0)

        if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
          efc_id = efc_id_in[worldid, efcid]
          contact_friction = contact_friction_in[efc_id]
          efc_addr0 = contact_efc_address_in[efc_id, 0]
          efc_addr1 = contact_efc_address_in[efc_id, 1]
          efc_addr2 = contact_efc_address_in[efc_id, 2]
          quad1 = efc_quad_inout[worldid, efc_addr1]
          quad2 = efc_quad_inout[worldid, efc_addr2]

        local_lo_in += _compute_efc_eval_pt(
          efcid,
          lo_alpha_in,
          ne,
          nf,
          impratio_invsqrt,
          efc_type,
          efc_D_in[worldid],
          efc_frictionloss_in[worldid],
          efc_Jaref_in[worldid, efcid],
          efc_jv_inout[worldid, efcid],
          efc_quad_inout[worldid, efcid],
          contact_friction,
          efc_addr0,
          quad1,
          quad2,
        )
      else:
        # direct evaluation for pyramidal cones (no intermediate quad)
        local_lo_in += _compute_efc_eval_pt(
          efcid,
          lo_alpha_in,
          ne,
          nf,
          efc_D_in[worldid, efcid],
          efc_frictionloss_in[worldid],
          efc_Jaref_in[worldid, efcid],
          efc_jv_inout[worldid, efcid],
        )

    lo_in_tile = wp.tile(local_lo_in, preserve_type=True)
    lo_in_sum = wp.tile_reduce(wp.add, lo_in_tile)
    lo_in = _eval_pt(efc_quad_gauss, lo_alpha_in) + lo_in_sum[0]

    # initialize bounds
    lo_less = lo_in[1] < p0[1]
    lo = wp.where(lo_less, lo_in, p0)
    lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)
    hi = wp.where(lo_less, p0, lo_in)
    hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)

    # main iterative loop
    alpha = float(0.0)

    for _ in range(LS_ITERATIONS):
      lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])
      hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])
      mid_alpha = 0.5 * (lo_alpha + hi_alpha)

      local_lo = wp.vec3(0.0)
      local_hi = wp.vec3(0.0)
      local_mid = wp.vec3(0.0)

      for efcid in range(tid, nefc, wp.block_dim()):
        if wp.static(IS_ELLIPTIC):
          efc_type = efc_type_in[worldid, efcid]
          efc_id = 0
          contact_friction = types.vec5(0.0)
          efc_addr0 = int(0)
          quad1 = wp.vec3(0.0)
          quad2 = wp.vec3(0.0)

          if efc_type == types.ConstraintType.CONTACT_ELLIPTIC:
            efc_id = efc_id_in[worldid, efcid]
            contact_friction = contact_friction_in[efc_id]
            efc_addr0 = contact_efc_address_in[efc_id, 0]
            efc_addr1 = contact_efc_address_in[efc_id, 1]
            efc_addr2 = contact_efc_address_in[efc_id, 2]
            quad1 = efc_quad_inout[worldid, efc_addr1]
            quad2 = efc_quad_inout[worldid, efc_addr2]

          # compute all 3 alphas at once, sharing constraint type checking
          r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(
            efcid,
            lo_next_alpha,
            hi_next_alpha,
            mid_alpha,
            ne,
            nf,
            impratio_invsqrt,
            efc_type,
            efc_D_in[worldid],
            efc_frictionloss_in[worldid],
            efc_Jaref_in[worldid, efcid],
            efc_jv_inout[worldid, efcid],
            efc_quad_inout[worldid, efcid],
            contact_friction,
            efc_addr0,
            quad1,
            quad2,
          )
        else:
          # direct evaluation for pyramidal cones (no intermediate quad)
          r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(
            efcid,
            lo_next_alpha,
            hi_next_alpha,
            mid_alpha,
            ne,
            nf,
            efc_D_in[worldid, efcid],
            efc_frictionloss_in[worldid],
            efc_Jaref_in[worldid, efcid],
            efc_jv_inout[worldid, efcid],
          )
        local_lo += r_lo
        local_hi += r_hi
        local_mid += r_mid

      # reduce with packed mat33 (3 vec3s into columns: col0=lo, col1=hi, col2=mid)
      local_combined = wp.mat33(
        local_lo[0],
        local_hi[0],
        local_mid[0],
        local_lo[1],
        local_hi[1],
        local_mid[1],
        local_lo[2],
        local_hi[2],
        local_mid[2],
      )

      # reduce with packed mat33 (3 vec3s into columns: col0=lo, col1=hi, col2=mid)
      # this is faster than 3 vec3 reductions because it avoids synchronization barriers
      combined_tile = wp.tile(local_combined, preserve_type=True)
      combined_sum = wp.tile_reduce(wp.add, combined_tile)
      result = combined_sum[0]

      # extract columns back to vec3s and add quad_gauss contributions
      gauss_lo, gauss_hi, gauss_mid = _eval_pt_3alphas(efc_quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha)
      lo_next = gauss_lo + wp.vec3(result[0, 0], result[1, 0], result[2, 0])
      hi_next = gauss_hi + wp.vec3(result[0, 1], result[1, 1], result[2, 1])
      mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])

      # bracket swapping
      # swap lo:
      swap_lo_lo_next = _in_bracket(lo, lo_next)
      lo = wp.where(swap_lo_lo_next, lo_next, lo)
      lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)
      swap_lo_mid = _in_bracket(lo, mid)
      lo = wp.where(swap_lo_mid, mid, lo)
      lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)
      swap_lo_hi_next = _in_bracket(lo, hi_next)
      lo = wp.where(swap_lo_hi_next, hi_next, lo)
      lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)
      swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next

      # swap hi:
      swap_hi_hi_next = _in_bracket(hi, hi_next)
      hi = wp.where(swap_hi_hi_next, hi_next, hi)
      hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)
      swap_hi_mid = _in_bracket(hi, mid)
      hi = wp.where(swap_hi_mid, mid, hi)
      hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)
      swap_hi_lo_next = _in_bracket(hi, lo_next)
      hi = wp.where(swap_hi_lo_next, lo_next, hi)
      hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)
      swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next

      # check for convergence
      ls_done = (not swap_lo and not swap_hi) or (lo[1] < 0.0 and lo[1] > -gtol) or (hi[1] > 0.0 and hi[1] < gtol)

      # update alpha if improved
      improved = lo[0] < p0[0] or hi[0] < p0[0]
      lo_better = lo[0] < hi[0]
      alpha = wp.where(improved and lo_better, lo_alpha, alpha)
      alpha = wp.where(improved and not lo_better, hi_alpha, alpha)

      if ls_done:
        break

    # qacc and Ma update
    for dofid in range(tid, nv, wp.block_dim()):
      qacc_out[worldid, dofid] += alpha * efc_search_in[worldid, dofid]
      efc_Ma_out[worldid, dofid] += alpha * efc_mv_in[worldid, dofid]

    # Jaref update
    for efcid in range(tid, nefc, wp.block_dim()):
      efc_Jaref_out[worldid, efcid] += alpha * efc_jv_inout[worldid, efcid]

  # kernel_analyzer: on
  return kernel


def _linesearch_iterative(m: types.Model, d: types.Data, fuse_jv: bool):
  """Iterative linesearch with parallel reductions over efc rows and dofs.

  Args:
    m: Model.
    d: Data.
    fuse_jv: Whether jv is computed in-kernel (True) or pre-computed (False).
  """
  wp.launch_tiled(
    linesearch_iterative(m.opt.ls_iterations, m.opt.cone, fuse_jv),
    dim=d.nworld,
    inputs=[
      m.nv,
      m.opt.tolerance,
      m.opt.ls_tolerance,
      m.opt.impratio_invsqrt,
      m.stat.meaninertia,
      d.ne,
      d.nf,
      d.nefc,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.frictionloss,
      d.efc.Jaref,
      d.efc.search_dot,
      d.efc.J,
      d.efc.gauss,
      d.qfrc_smooth,
      d.efc.done,
      d.njmax,
      d.nacon,
      d.efc.search,
      d.efc.mv,
    ],
    outputs=[d.efc.quad, d.efc.jv, d.qacc, d.efc.Ma, d.efc.Jaref],
    block_dim=m.block_dim.linesearch_iterative,
  )


@wp.kernel
def linesearch_zero_jv(
  # Data in:
  nefc_in: wp.array(dtype=int),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_jv_out: wp.array2d(dtype=float),
):
  worldid, efcid = wp.tid()

  if efcid >= nefc_in[worldid]:
    return

  if efc_done_in[worldid]:
    return

  efc_jv_out[worldid, efcid] = 0.0


@cache_kernel
def linesearch_jv_fused(nv: int, dofs_per_thread: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array(dtype=int),
    efc_J_in: wp.array3d(dtype=float),
    efc_search_in: wp.array2d(dtype=float),
    efc_done_in: wp.array(dtype=bool),
    # Data out:
    efc_jv_out: wp.array2d(dtype=float),
  ):
    worldid, efcid, dofstart = wp.tid()

    if efcid >= nefc_in[worldid]:
      return

    if efc_done_in[worldid]:
      return

    jv_out = float(0.0)

    if wp.static(dofs_per_thread >= nv):
      for i in range(wp.static(min(dofs_per_thread, nv))):
        jv_out += efc_J_in[worldid, efcid, i] * efc_search_in[worldid, i]
      efc_jv_out[worldid, efcid] = jv_out

    else:
      for i in range(wp.static(dofs_per_thread)):
        ii = dofstart * wp.static(dofs_per_thread) + i
        if ii < nv:
          jv_out += efc_J_in[worldid, efcid, ii] * efc_search_in[worldid, ii]
      wp.atomic_add(efc_jv_out, worldid, efcid, jv_out)

  return kernel


@cache_kernel
def linesearch_prepare_gauss(nv: int, dofs_per_thread: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    qfrc_smooth_in: wp.array2d(dtype=float),
    efc_Ma_in: wp.array2d(dtype=float),
    efc_search_in: wp.array2d(dtype=float),
    efc_gauss_in: wp.array(dtype=float),
    efc_mv_in: wp.array2d(dtype=float),
    efc_done_in: wp.array(dtype=bool),
    # Data out:
    efc_quad_gauss_out: wp.array(dtype=wp.vec3),
  ):
    worldid, dofstart = wp.tid()

    if efc_done_in[worldid]:
      return

    quad_gauss_1 = float(0.0)
    quad_gauss_2 = float(0.0)

    if wp.static(dofs_per_thread >= nv):
      for i in range(wp.static(nv)):
        search = efc_search_in[worldid, i]
        quad_gauss_1 += search * (efc_Ma_in[worldid, i] - qfrc_smooth_in[worldid, i])
        quad_gauss_2 += 0.5 * search * efc_mv_in[worldid, i]

      quad_gauss_0 = efc_gauss_in[worldid]
      efc_quad_gauss_out[worldid] = wp.vec3(quad_gauss_0, quad_gauss_1, quad_gauss_2)

    else:
      for i in range(wp.static(dofs_per_thread)):
        ii = dofstart * wp.static(dofs_per_thread) + i
        if ii < nv:
          search = efc_search_in[worldid, ii]
          quad_gauss_1 += search * (efc_Ma_in[worldid, ii] - qfrc_smooth_in[worldid, ii])
          quad_gauss_2 += 0.5 * search * efc_mv_in[worldid, ii]

      if dofstart == 0:
        quad_gauss_0 = efc_gauss_in[worldid]
        wp.atomic_add(efc_quad_gauss_out, worldid, wp.vec3(quad_gauss_0, quad_gauss_1, quad_gauss_2))
      else:
        wp.atomic_add(efc_quad_gauss_out, worldid, wp.vec3(0.0, quad_gauss_1, quad_gauss_2))

  return kernel


@wp.kernel
def linesearch_prepare_quad(
  # Model:
  opt_impratio_invsqrt: wp.array(dtype=float),
  # Data in:
  nefc_in: wp.array(dtype=int),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  efc_D_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_jv_in: wp.array2d(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  nacon_in: wp.array(dtype=int),
  # Data out:
  efc_quad_out: wp.array2d(dtype=wp.vec3),
):
  worldid, efcid = wp.tid()

  if efcid >= nefc_in[worldid]:
    return

  if efc_done_in[worldid]:
    return

  Jaref = efc_Jaref_in[worldid, efcid]
  jv = efc_jv_in[worldid, efcid]
  efc_D = efc_D_in[worldid, efcid]

  # init with scalar quadratic
  quad = wp.vec3(0.5 * Jaref * Jaref * efc_D, jv * Jaref * efc_D, 0.5 * jv * jv * efc_D)

  # elliptic cone: extra processing
  if efc_type_in[worldid, efcid] == types.ConstraintType.CONTACT_ELLIPTIC:
    # extract contact info
    conid = efc_id_in[worldid, efcid]

    if conid >= nacon_in[0]:
      return

    efcid0 = contact_efc_address_in[conid, 0]

    if efcid != efcid0:
      return

    dim = contact_dim_in[conid]
    friction = contact_friction_in[conid]
    mu = friction[0] * opt_impratio_invsqrt[worldid]

    u0 = Jaref * mu
    v0 = jv * mu

    uu = float(0.0)
    uv = float(0.0)
    vv = float(0.0)
    for j in range(1, dim):
      # complete vector quadratic (for bottom zone)
      efcidj = contact_efc_address_in[conid, j]
      if efcidj < 0:
        return
      jvj = efc_jv_in[worldid, efcidj]
      jarefj = efc_Jaref_in[worldid, efcidj]
      dj = efc_D_in[worldid, efcidj]
      DJj = dj * jarefj

      quad += wp.vec3(
        0.5 * jarefj * DJj,
        jvj * DJj,
        0.5 * jvj * dj * jvj,
      )

      # rescale to make primal cone circular
      frictionj = friction[j - 1]
      uj = jarefj * frictionj
      vj = jvj * frictionj

      # accumulate sums of squares
      uu += uj * uj
      uv += uj * vj
      vv += vj * vj

    quad1 = wp.vec3(u0, v0, uu)
    efcid1 = contact_efc_address_in[conid, 1]
    efc_quad_out[worldid, efcid1] = quad1

    mu2 = mu * mu
    quad2 = wp.vec3(uv, vv, efc_D / (mu2 * (1.0 + mu2)))
    efcid2 = contact_efc_address_in[conid, 2]
    efc_quad_out[worldid, efcid2] = quad2

  efc_quad_out[worldid, efcid] = quad


@wp.kernel
def linesearch_qacc_ma(
  # Data in:
  efc_search_in: wp.array2d(dtype=float),
  efc_mv_in: wp.array2d(dtype=float),
  efc_alpha_in: wp.array(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  qacc_out: wp.array2d(dtype=float),
  efc_Ma_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()

  if efc_done_in[worldid]:
    return

  alpha = efc_alpha_in[worldid]
  qacc_out[worldid, dofid] += alpha * efc_search_in[worldid, dofid]
  efc_Ma_out[worldid, dofid] += alpha * efc_mv_in[worldid, dofid]


@wp.kernel
def linesearch_jaref(
  # Data in:
  nefc_in: wp.array(dtype=int),
  efc_jv_in: wp.array2d(dtype=float),
  efc_alpha_in: wp.array(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_Jaref_out: wp.array2d(dtype=float),
):
  worldid, efcid = wp.tid()

  if efcid >= nefc_in[worldid]:
    return

  if efc_done_in[worldid]:
    return

  efc_Jaref_out[worldid, efcid] += efc_alpha_in[worldid] * efc_jv_in[worldid, efcid]


@event_scope
def _linesearch(m: types.Model, d: types.Data, cost: wp.array2d(dtype=float)):
  """Linesearch for constraint solver.

  Args:
    m: Model
    d: Data
    cost: Scratch array for storing costs per (world, alpha) - used for parallel mode
  """
  # mv = qM @ search (common to both parallel and iterative)
  support.mul_m(m, d, d.efc.mv, d.efc.search, skip=d.efc.done)

  # Fuse jv computation in-kernel for small nv (iterative only)
  # Parallel linesearch always requires jv pre-computed
  fuse_jv = m.nv <= 50 and not m.opt.ls_parallel

  # jv = J @ search (when not fused into iterative kernel)
  if not fuse_jv:
    dofs_per_thread = 20 if m.nv > 50 else 50
    threads_per_efc = ceil(m.nv / dofs_per_thread)

    if threads_per_efc > 1:
      wp.launch(
        linesearch_zero_jv,
        dim=(d.nworld, d.njmax),
        inputs=[d.nefc, d.efc.done],
        outputs=[d.efc.jv],
      )

    wp.launch(
      linesearch_jv_fused(m.nv, dofs_per_thread),
      dim=(d.nworld, d.njmax, threads_per_efc),
      inputs=[d.nefc, d.efc.J, d.efc.search, d.efc.done],
      outputs=[d.efc.jv],
    )

  if m.opt.ls_parallel:
    _linesearch_parallel(m, d, cost)
  else:
    _linesearch_iterative(m, d, fuse_jv)


@wp.kernel
def solve_init_efc(
  # Data out:
  solver_niter_out: wp.array(dtype=int),
  efc_search_dot_out: wp.array(dtype=float),
  efc_cost_out: wp.array(dtype=float),
  efc_done_out: wp.array(dtype=bool),
):
  worldid = wp.tid()
  efc_cost_out[worldid] = wp.inf
  solver_niter_out[worldid] = 0
  efc_done_out[worldid] = False
  efc_search_dot_out[worldid] = 0.0


@cache_kernel
def solve_init_jaref(nv: int, dofs_per_thread: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array(dtype=int),
    qacc_in: wp.array2d(dtype=float),
    efc_J_in: wp.array3d(dtype=float),
    efc_aref_in: wp.array2d(dtype=float),
    # Data out:
    efc_Jaref_out: wp.array2d(dtype=float),
  ):
    worldid, efcid, dofstart = wp.tid()

    if efcid >= nefc_in[worldid]:
      return

    jaref = float(0.0)

    if wp.static(dofs_per_thread >= nv):
      for i in range(wp.static(min(dofs_per_thread, nv))):
        jaref += efc_J_in[worldid, efcid, i] * qacc_in[worldid, i]
      efc_Jaref_out[worldid, efcid] = jaref - efc_aref_in[worldid, efcid]

    else:
      for i in range(wp.static(dofs_per_thread)):
        ii = dofstart * wp.static(dofs_per_thread) + i
        if ii < nv:
          jaref += efc_J_in[worldid, efcid, ii] * qacc_in[worldid, ii]

      if dofstart == 0:
        wp.atomic_add(efc_Jaref_out, worldid, efcid, jaref - efc_aref_in[worldid, efcid])
      else:
        wp.atomic_add(efc_Jaref_out, worldid, efcid, jaref)

  return kernel


@wp.kernel
def solve_init_search(
  # Data in:
  efc_Mgrad_in: wp.array2d(dtype=float),
  # Data out:
  efc_search_out: wp.array2d(dtype=float),
  efc_search_dot_out: wp.array(dtype=float),
):
  worldid, dofid = wp.tid()
  search = -1.0 * efc_Mgrad_in[worldid, dofid]
  efc_search_out[worldid, dofid] = search
  wp.atomic_add(efc_search_dot_out, worldid, search * search)


@wp.kernel
def update_constraint_init_cost(
  # Data in:
  efc_cost_in: wp.array(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_gauss_out: wp.array(dtype=float),
  efc_cost_out: wp.array(dtype=float),
  efc_prev_cost_out: wp.array(dtype=float),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  efc_gauss_out[worldid] = 0.0
  efc_prev_cost_out[worldid] = efc_cost_in[worldid]
  efc_cost_out[worldid] = 0.0


@wp.kernel
def update_constraint_efc(
  # Model:
  opt_impratio_invsqrt: wp.array(dtype=float),
  # Data in:
  ne_in: wp.array(dtype=int),
  nf_in: wp.array(dtype=int),
  nefc_in: wp.array(dtype=int),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  efc_D_in: wp.array2d(dtype=float),
  efc_frictionloss_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  nacon_in: wp.array(dtype=int),
  # Data out:
  efc_force_out: wp.array2d(dtype=float),
  efc_cost_out: wp.array(dtype=float),
  efc_state_out: wp.array2d(dtype=int),
):
  worldid, efcid = wp.tid()

  if efcid >= nefc_in[worldid]:
    return

  if efc_done_in[worldid]:
    return

  efc_D = efc_D_in[worldid, efcid]
  Jaref = efc_Jaref_in[worldid, efcid]

  ne = ne_in[worldid]
  nf = nf_in[worldid]

  if efcid < ne:
    # equality
    efc_force_out[worldid, efcid] = -efc_D * Jaref
    efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC
    wp.atomic_add(efc_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)
  elif efcid < ne + nf:
    # friction
    f = efc_frictionloss_in[worldid, efcid]
    rf = math.safe_div(f, efc_D)
    if Jaref <= -rf:
      efc_force_out[worldid, efcid] = f
      efc_state_out[worldid, efcid] = types.ConstraintState.LINEARNEG
      wp.atomic_add(efc_cost_out, worldid, -f * (0.5 * rf + Jaref))
    elif Jaref >= rf:
      efc_force_out[worldid, efcid] = -f
      efc_state_out[worldid, efcid] = types.ConstraintState.LINEARPOS
      wp.atomic_add(efc_cost_out, worldid, -f * (0.5 * rf - Jaref))
    else:
      efc_force_out[worldid, efcid] = -efc_D * Jaref
      efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC
      wp.atomic_add(efc_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)
  elif efc_type_in[worldid, efcid] != types.ConstraintType.CONTACT_ELLIPTIC:
    # limit, frictionless contact, pyramidal friction cone contact
    if Jaref >= 0.0:
      efc_force_out[worldid, efcid] = 0.0
      efc_state_out[worldid, efcid] = types.ConstraintState.SATISFIED
    else:
      efc_force_out[worldid, efcid] = -efc_D * Jaref
      efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC
      wp.atomic_add(efc_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)
  else:  # elliptic friction cone contact
    conid = efc_id_in[worldid, efcid]

    if conid >= nacon_in[0]:
      return

    dim = contact_dim_in[conid]
    friction = contact_friction_in[conid]
    mu = friction[0] * opt_impratio_invsqrt[worldid]

    efcid0 = contact_efc_address_in[conid, 0]
    if efcid0 < 0:
      return

    N = efc_Jaref_in[worldid, efcid0] * mu

    ufrictionj = float(0.0)
    TT = float(0.0)
    for j in range(1, dim):
      efcidj = contact_efc_address_in[conid, j]
      if efcidj < 0:
        return
      frictionj = friction[j - 1]
      uj = efc_Jaref_in[worldid, efcidj] * frictionj
      TT += uj * uj
      if efcid == efcidj:
        ufrictionj = uj * frictionj

    if TT <= 0.0:
      T = 0.0
    else:
      T = wp.sqrt(TT)

    # top zone
    if (N >= mu * T) or ((T <= 0.0) and (N >= 0.0)):
      efc_force_out[worldid, efcid] = 0.0
      efc_state_out[worldid, efcid] = types.ConstraintState.SATISFIED
    # bottom zone
    elif (mu * N + T <= 0.0) or ((T <= 0.0) and (N < 0.0)):
      efc_force_out[worldid, efcid] = -efc_D * Jaref
      efc_state_out[worldid, efcid] = types.ConstraintState.QUADRATIC
      wp.atomic_add(efc_cost_out, worldid, 0.5 * efc_D * Jaref * Jaref)
    # middle zone
    else:
      dm = math.safe_div(efc_D_in[worldid, efcid0], mu * mu * (1.0 + mu * mu))
      nmt = N - mu * T

      force = -dm * nmt * mu

      if efcid == efcid0:
        efc_force_out[worldid, efcid] = force
        wp.atomic_add(efc_cost_out, worldid, 0.5 * dm * nmt * nmt)
      else:
        efc_force_out[worldid, efcid] = -math.safe_div(force, T) * ufrictionj

      efc_state_out[worldid, efcid] = types.ConstraintState.CONE


@wp.kernel
def update_constraint_init_qfrc_constraint(
  # Data in:
  nefc_in: wp.array(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_force_in: wp.array2d(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  njmax_in: int,
  # Data out:
  qfrc_constraint_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()

  if efc_done_in[worldid]:
    return

  sum_qfrc = float(0.0)
  for efcid in range(min(njmax_in, nefc_in[worldid])):
    efc_J = efc_J_in[worldid, efcid, dofid]
    force = efc_force_in[worldid, efcid]
    sum_qfrc += efc_J * force

  qfrc_constraint_out[worldid, dofid] = sum_qfrc


@cache_kernel
def update_constraint_gauss_cost(nv: int, dofs_per_thread: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    qacc_in: wp.array2d(dtype=float),
    qfrc_smooth_in: wp.array2d(dtype=float),
    qacc_smooth_in: wp.array2d(dtype=float),
    efc_Ma_in: wp.array2d(dtype=float),
    efc_done_in: wp.array(dtype=bool),
    # Data out:
    efc_gauss_out: wp.array(dtype=float),
    efc_cost_out: wp.array(dtype=float),
  ):
    worldid, dofstart = wp.tid()

    if efc_done_in[worldid]:
      return

    gauss_cost = float(0.0)

    if wp.static(dofs_per_thread >= nv):
      for i in range(wp.static(min(dofs_per_thread, nv))):
        gauss_cost += (efc_Ma_in[worldid, i] - qfrc_smooth_in[worldid, i]) * (qacc_in[worldid, i] - qacc_smooth_in[worldid, i])
      efc_gauss_out[worldid] += 0.5 * gauss_cost
      efc_cost_out[worldid] += 0.5 * gauss_cost

    else:
      for i in range(wp.static(dofs_per_thread)):
        ii = dofstart * wp.static(dofs_per_thread) + i
        if ii < nv:
          gauss_cost += (efc_Ma_in[worldid, ii] - qfrc_smooth_in[worldid, ii]) * (
            qacc_in[worldid, ii] - qacc_smooth_in[worldid, ii]
          )
      wp.atomic_add(efc_gauss_out, worldid, gauss_cost)
      wp.atomic_add(efc_cost_out, worldid, gauss_cost)

  return kernel


def _update_constraint(m: types.Model, d: types.Data):
  """Update constraint arrays after each solve iteration."""
  wp.launch(
    update_constraint_init_cost,
    dim=(d.nworld),
    inputs=[d.efc.cost, d.efc.done],
    outputs=[d.efc.gauss, d.efc.cost, d.efc.prev_cost],
  )

  wp.launch(
    update_constraint_efc,
    dim=(d.nworld, d.njmax),
    inputs=[
      m.opt.impratio_invsqrt,
      d.ne,
      d.nf,
      d.nefc,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.D,
      d.efc.frictionloss,
      d.efc.Jaref,
      d.efc.done,
      d.nacon,
    ],
    outputs=[d.efc.force, d.efc.cost, d.efc.state],
  )

  # qfrc_constraint = efc_J.T @ efc_force
  wp.launch(
    update_constraint_init_qfrc_constraint,
    dim=(d.nworld, m.nv),
    inputs=[d.nefc, d.efc.J, d.efc.force, d.efc.done, d.njmax],
    outputs=[d.qfrc_constraint],
  )

  # if we are only using 1 thread, it makes sense to do more dofs and skip the atomics.
  # For more than 1 thread, dofs_per_thread is lower for better load balancing.
  if m.nv > 50:
    dofs_per_thread = 20
  else:
    dofs_per_thread = 50

  threads_per_efc = ceil(m.nv / dofs_per_thread)

  # gauss = 0.5 * (Ma - qfrc_smooth).T @ (qacc - qacc_smooth)
  wp.launch(
    update_constraint_gauss_cost(m.nv, dofs_per_thread),
    dim=(d.nworld, threads_per_efc),
    inputs=[d.qacc, d.qfrc_smooth, d.qacc_smooth, d.efc.Ma, d.efc.done],
    outputs=[d.efc.gauss, d.efc.cost],
  )


@wp.kernel
def update_gradient_zero_grad_dot(
  # Data in:
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_grad_dot_out: wp.array(dtype=float),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  efc_grad_dot_out[worldid] = 0.0


@wp.kernel
def update_gradient_grad(
  # Data in:
  qfrc_smooth_in: wp.array2d(dtype=float),
  qfrc_constraint_in: wp.array2d(dtype=float),
  efc_Ma_in: wp.array2d(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_grad_out: wp.array2d(dtype=float),
  efc_grad_dot_out: wp.array(dtype=float),
):
  worldid, dofid = wp.tid()

  if efc_done_in[worldid]:
    return

  grad = efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - qfrc_constraint_in[worldid, dofid]
  efc_grad_out[worldid, dofid] = grad
  wp.atomic_add(efc_grad_dot_out, worldid, grad * grad)


@wp.kernel
def update_gradient_set_h_qM_lower_sparse(
  # Model:
  qM_fullm_i: wp.array(dtype=int),
  qM_fullm_j: wp.array(dtype=int),
  # Data in:
  qM_in: wp.array3d(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Out:
  h_out: wp.array3d(dtype=float),
):
  worldid, elementid = wp.tid()

  if efc_done_in[worldid]:
    return

  i = qM_fullm_i[elementid]
  j = qM_fullm_j[elementid]
  h_out[worldid, i, j] += qM_in[worldid, 0, elementid]


@wp.func
def state_check(D: float, state: int) -> float:
  if state == types.ConstraintState.QUADRATIC.value:
    return D
  else:
    return 0.0


@wp.func
def active_check(tid: int, threshold: int) -> float:
  if tid >= threshold:
    return 0.0
  else:
    return 1.0


@cache_kernel
def update_gradient_JTDAJ_sparse_tiled(tile_size: int, njmax: int):
  TILE_SIZE = tile_size

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array(dtype=int),
    efc_J_in: wp.array3d(dtype=float),
    efc_D_in: wp.array2d(dtype=float),
    efc_state_in: wp.array2d(dtype=int),
    efc_done_in: wp.array(dtype=bool),
    # Out:
    h_out: wp.array3d(dtype=float),
  ):
    worldid, elementid = wp.tid()

    if efc_done_in[worldid]:
      return

    nefc = nefc_in[worldid]

    # get lower diagonal index
    i = (int(sqrt(float(1 + 8 * elementid))) - 1) // 2
    j = elementid - (i * (i + 1)) // 2

    offset_i = i * TILE_SIZE
    offset_j = j * TILE_SIZE

    sum_val = wp.tile_zeros(shape=(TILE_SIZE, TILE_SIZE), dtype=wp.float32)

    # Each tile processes looping over all constraints, producing 1 output tile
    for k in range(0, njmax, TILE_SIZE):
      if k >= nefc:
        break

      # AD: leaving bounds-check disabled here because I'm not entirely sure that
      # everything always hits the fast path. The padding takes care of any
      # potential OOB accesses.
      J_ki = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(k, offset_i), bounds_check=False)

      if offset_i != offset_j:
        J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE, TILE_SIZE), offset=(k, offset_j), bounds_check=False)
      else:
        wp.tile_assign(J_kj, J_ki, (0, 0))

      D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE, offset=k, bounds_check=False)
      state = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE, offset=k, bounds_check=False)

      D_k = wp.tile_map(state_check, D_k, state)

      # force unused elements to be zero
      tid_tile = wp.tile_arange(TILE_SIZE, dtype=int)
      threshold_tile = wp.tile_ones(shape=TILE_SIZE, dtype=int) * (nefc - k)

      active_tile = wp.tile_map(active_check, tid_tile, threshold_tile)
      D_k = wp.tile_map(wp.mul, active_tile, D_k)

      J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_ki), wp.tile_broadcast(D_k, shape=(TILE_SIZE, TILE_SIZE)))

      sum_val += wp.tile_matmul(J_ki, J_kj)

    # AD: setting bounds_check to True explicitly here because for some reason it was
    # slower to disable it.
    wp.tile_store(h_out[worldid], sum_val, offset=(offset_i, offset_j), bounds_check=True)

  return kernel


@cache_kernel
def update_gradient_JTDAJ_dense_tiled(nv_padded: int, tile_size: int, njmax: int):
  if njmax < tile_size:
    tile_size = njmax

  TILE_SIZE_K = tile_size

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    nefc_in: wp.array(dtype=int),
    qM_in: wp.array3d(dtype=float),
    efc_J_in: wp.array3d(dtype=float),
    efc_D_in: wp.array2d(dtype=float),
    efc_state_in: wp.array2d(dtype=int),
    efc_done_in: wp.array(dtype=bool),
    # Out:
    h_out: wp.array3d(dtype=float),
  ):
    worldid = wp.tid()

    if efc_done_in[worldid]:
      return

    nefc = nefc_in[worldid]

    sum_val = wp.tile_load(qM_in[worldid], shape=(nv_padded, nv_padded), bounds_check=True)

    # Each tile processes one output tile by looping over all constraints
    for k in range(0, njmax, TILE_SIZE_K):
      if k >= nefc:
        break

      # AD: leaving bounds-check disabled here because I'm not entirely sure that
      # everything always hits the fast path. The padding takes care of any
      #  potential OOB accesses.
      J_kj = wp.tile_load(efc_J_in[worldid], shape=(TILE_SIZE_K, nv_padded), offset=(k, 0), bounds_check=False)

      # state check
      D_k = wp.tile_load(efc_D_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)
      state = wp.tile_load(efc_state_in[worldid], shape=TILE_SIZE_K, offset=k, bounds_check=False)

      D_k = wp.tile_map(state_check, D_k, state)

      # force unused elements to be zero
      tid_tile = wp.tile_arange(TILE_SIZE_K, dtype=int)
      threshold_tile = wp.tile_ones(shape=TILE_SIZE_K, dtype=int) * (nefc - k)

      active_tile = wp.tile_map(active_check, tid_tile, threshold_tile)
      D_k = wp.tile_map(wp.mul, active_tile, D_k)

      J_ki = wp.tile_map(wp.mul, wp.tile_transpose(J_kj), wp.tile_broadcast(D_k, shape=(nv_padded, TILE_SIZE_K)))

      sum_val += wp.tile_matmul(J_ki, J_kj)

    wp.tile_store(h_out[worldid], sum_val, bounds_check=False)

  return kernel


# TODO(thowell): combine with JTDAJ ?
@wp.kernel
def update_gradient_JTCJ(
  # Model:
  opt_impratio_invsqrt: wp.array(dtype=float),
  dof_tri_row: wp.array(dtype=int),
  dof_tri_col: wp.array(dtype=int),
  # Data in:
  contact_dist_in: wp.array(dtype=float),
  contact_includemargin_in: wp.array(dtype=float),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_efc_address_in: wp.array2d(dtype=int),
  contact_worldid_in: wp.array(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  efc_D_in: wp.array2d(dtype=float),
  efc_Jaref_in: wp.array2d(dtype=float),
  efc_state_in: wp.array2d(dtype=int),
  efc_done_in: wp.array(dtype=bool),
  naconmax_in: int,
  nacon_in: wp.array(dtype=int),
  # In:
  nblocks_perblock: int,
  dim_block: int,
  # Out:
  h_out: wp.array3d(dtype=float),
):
  conid_start, elementid = wp.tid()

  dof1id = dof_tri_row[elementid]
  dof2id = dof_tri_col[elementid]

  for i in range(nblocks_perblock):
    conid = conid_start + i * dim_block

    if conid >= min(nacon_in[0], naconmax_in):
      return

    worldid = contact_worldid_in[conid]
    if efc_done_in[worldid]:
      continue

    condim = contact_dim_in[conid]

    if condim == 1:
      continue

    # check contact status
    if contact_dist_in[conid] - contact_includemargin_in[conid] >= 0.0:
      continue

    efcid0 = contact_efc_address_in[conid, 0]
    if efc_state_in[worldid, efcid0] != types.ConstraintState.CONE:
      continue

    fri = contact_friction_in[conid]
    mu = fri[0] * opt_impratio_invsqrt[worldid]

    mu2 = mu * mu
    dm = math.safe_div(efc_D_in[worldid, efcid0], mu2 * (1.0 + mu2))

    if dm == 0.0:
      continue

    n = efc_Jaref_in[worldid, efcid0] * mu
    u = types.vec6(n, 0.0, 0.0, 0.0, 0.0, 0.0)

    tt = float(0.0)
    for j in range(1, condim):
      efcidj = contact_efc_address_in[conid, j]
      uj = efc_Jaref_in[worldid, efcidj] * fri[j - 1]
      tt += uj * uj
      u[j] = uj

    if tt <= 0.0:
      t = 0.0
    else:
      t = wp.sqrt(tt)
    t = wp.max(t, types.MJ_MINVAL)
    ttt = wp.max(t * t * t, types.MJ_MINVAL)

    h = float(0.0)

    for dim1id in range(condim):
      if dim1id == 0:
        efcid1 = efcid0
      else:
        efcid1 = contact_efc_address_in[conid, dim1id]

      efc_J11 = efc_J_in[worldid, efcid1, dof1id]
      efc_J12 = efc_J_in[worldid, efcid1, dof2id]

      ui = u[dim1id]

      for dim2id in range(0, dim1id + 1):
        if dim2id == 0:
          efcid2 = efcid0
        else:
          efcid2 = contact_efc_address_in[conid, dim2id]

        efc_J21 = efc_J_in[worldid, efcid2, dof1id]
        efc_J22 = efc_J_in[worldid, efcid2, dof2id]

        uj = u[dim2id]

        # set first row/column: (1, -mu/t * u)
        if dim1id == 0 and dim2id == 0:
          hcone = 1.0
        elif dim1id == 0:
          hcone = -math.safe_div(mu, t) * uj
        elif dim2id == 0:
          hcone = -math.safe_div(mu, t) * ui
        else:
          hcone = mu * math.safe_div(n, ttt) * ui * uj

          # add to diagonal: mu^2 - mu * n / t
          if dim1id == dim2id:
            hcone += mu2 - mu * math.safe_div(n, t)

        # pre and post multiply by diag(mu, friction) scale by dm
        if dim1id == 0:
          fri1 = mu
        else:
          fri1 = fri[dim1id - 1]

        if dim2id == 0:
          fri2 = mu
        else:
          fri2 = fri[dim2id - 1]

        hcone *= dm * fri1 * fri2

        if hcone != 0.0:
          h += hcone * efc_J11 * efc_J22

          if dim1id != dim2id:
            h += hcone * efc_J12 * efc_J21

    h_out[worldid, dof1id, dof2id] += h


@cache_kernel
def update_gradient_cholesky(tile_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    efc_grad_in: wp.array2d(dtype=float),
    efc_done_in: wp.array(dtype=bool),
    # In:
    h_in: wp.array3d(dtype=float),
    # Data out:
    efc_Mgrad_out: wp.array2d(dtype=float),
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)

    if efc_done_in[worldid]:
      return

    mat_tile = wp.tile_load(h_in[worldid], shape=(TILE_SIZE, TILE_SIZE))
    fact_tile = wp.tile_cholesky(mat_tile)
    input_tile = wp.tile_load(efc_grad_in[worldid], shape=TILE_SIZE)
    output_tile = wp.tile_cholesky_solve(fact_tile, input_tile)
    wp.tile_store(efc_Mgrad_out[worldid], output_tile)

  return kernel


@cache_kernel
def update_gradient_cholesky_blocked(tile_size: int, matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    efc_grad_in: wp.array3d(dtype=float),  # kernel_analyzer: ignore
    efc_done_in: wp.array(dtype=bool),
    # In:
    h_in: wp.array3d(dtype=float),
    hfactor: wp.array3d(dtype=float),
    # Data out:
    efc_Mgrad_out: wp.array3d(dtype=float),  # kernel_analyzer: ignore
  ):
    worldid = wp.tid()
    TILE_SIZE = wp.static(tile_size)

    if efc_done_in[worldid]:
      return

    # We need matrix size both as a runtime input as well as a static input:
    # static input is needed to specify the tile sizes for the compiler
    # runtime input is needed for the loop bounds, otherwise warp will unroll
    # unconditionally leading to shared memory capacity issues.

    wp.static(create_blocked_cholesky_func(TILE_SIZE))(h_in[worldid], matrix_size, hfactor[worldid])
    wp.static(create_blocked_cholesky_solve_func(TILE_SIZE, matrix_size))(
      hfactor[worldid], efc_grad_in[worldid], matrix_size, efc_Mgrad_out[worldid]
    )

  return kernel


@wp.kernel
def padding_h(nv: int, efc_done_in: wp.array(dtype=bool), h_out: wp.array3d(dtype=float)):
  worldid, elementid = wp.tid()

  if efc_done_in[worldid]:
    return

  dofid = nv + elementid
  h_out[worldid, dofid, dofid] = 1.0


def _update_gradient(m: types.Model, d: types.Data, h: wp.array3d(dtype=float), hfactor: wp.array3d(dtype=float)):
  # grad = Ma - qfrc_smooth - qfrc_constraint
  wp.launch(update_gradient_zero_grad_dot, dim=(d.nworld), inputs=[d.efc.done], outputs=[d.efc.grad_dot])

  wp.launch(
    update_gradient_grad,
    dim=(d.nworld, m.nv),
    inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, d.efc.done],
    outputs=[d.efc.grad, d.efc.grad_dot],
  )

  if m.opt.solver == types.SolverType.CG:
    smooth.solve_m(m, d, d.efc.Mgrad, d.efc.grad)
  elif m.opt.solver == types.SolverType.NEWTON:
    # h = qM + (efc_J.T * efc_D * active) @ efc_J
    if m.opt.is_sparse:
      num_blocks_ceil = ceil(m.nv / types.TILE_SIZE_JTDAJ_SPARSE)
      lower_triangle_dim = int(num_blocks_ceil * (num_blocks_ceil + 1) / 2)
      wp.launch_tiled(
        update_gradient_JTDAJ_sparse_tiled(types.TILE_SIZE_JTDAJ_SPARSE, d.njmax),
        dim=(d.nworld, lower_triangle_dim),
        inputs=[
          d.nefc,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          d.efc.done,
        ],
        outputs=[h],
        block_dim=m.block_dim.update_gradient_JTDAJ_sparse,
      )

      wp.launch(
        update_gradient_set_h_qM_lower_sparse,
        dim=(d.nworld, m.qM_fullm_i.size),
        inputs=[m.qM_fullm_i, m.qM_fullm_j, d.qM, d.efc.done],
        outputs=[h],
      )
    else:
      nv_padded = d.efc.J.shape[2]
      wp.launch_tiled(
        update_gradient_JTDAJ_dense_tiled(nv_padded, types.TILE_SIZE_JTDAJ_DENSE, d.njmax),
        dim=d.nworld,
        inputs=[
          d.nefc,
          d.qM,
          d.efc.J,
          d.efc.D,
          d.efc.state,
          d.efc.done,
        ],
        outputs=[h],
        block_dim=m.block_dim.update_gradient_JTDAJ_dense,
      )

    if m.opt.cone == types.ConeType.ELLIPTIC:
      # Optimization: launching update_gradient_JTCJ with limited number of blocks on a GPU.
      # Profiling suggests that only a fraction of blocks out of the original
      # d.njmax blocks do the actual work. It aims to minimize #CTAs with no
      # effective work. It launches with #blocks that's proportional to the number
      # of SMs on the GPU. We can now query the SM count:
      # https://github.com/NVIDIA/warp/commit/f3814e7e5459e5fd13032cf0fddb3daddd510f30

      # make dim_block and nblocks_perblock static for update_gradient_JTCJ to allow
      # loop unrolling
      if wp.get_device().is_cuda:
        sm_count = wp.get_device().sm_count

        # Here we assume one block has 256 threads. We use a factor of 6, which
        # can be changed in the future to fine-tune the perf. The optimal factor will
        # depend on the kernel's occupancy, which determines how many blocks can
        # simultaneously run on the SM. TODO: This factor can be tuned further.
        dim_block = ceil((sm_count * 6 * 256) / m.dof_tri_row.size)
      else:
        # fall back for CPU
        dim_block = d.naconmax

      nblocks_perblock = int((d.naconmax + dim_block - 1) / dim_block)

      wp.launch(
        update_gradient_JTCJ,
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
          d.efc.Jaref,
          d.efc.state,
          d.efc.done,
          d.naconmax,
          d.nacon,
          nblocks_perblock,
          dim_block,
        ],
        outputs=[h],
      )

    if m.nv <= _BLOCK_CHOLESKY_DIM:
      wp.launch_tiled(
        update_gradient_cholesky(m.nv),
        dim=d.nworld,
        inputs=[d.efc.grad, d.efc.done, h],
        outputs=[d.efc.Mgrad],
        block_dim=m.block_dim.update_gradient_cholesky,
      )
    else:
      wp.launch(
        padding_h,
        dim=(d.nworld, m.nv_pad - m.nv),
        inputs=[m.nv, d.efc.done],
        outputs=[h],
      )

      wp.launch_tiled(
        update_gradient_cholesky_blocked(types.TILE_SIZE_JTDAJ_DENSE, m.nv_pad),
        dim=d.nworld,
        inputs=[d.efc.grad.reshape(shape=(d.nworld, d.efc.grad.shape[1], 1)), d.efc.done, h, hfactor],
        outputs=[d.efc.Mgrad.reshape(shape=(d.nworld, d.efc.Mgrad.shape[1], 1))],
        block_dim=m.block_dim.update_gradient_cholesky_blocked,
      )
  else:
    raise ValueError(f"Unknown solver type: {m.opt.solver}")


@wp.kernel
def solve_prev_grad_Mgrad(
  # Data in:
  efc_grad_in: wp.array2d(dtype=float),
  efc_Mgrad_in: wp.array2d(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_prev_grad_out: wp.array2d(dtype=float),
  efc_prev_Mgrad_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()

  if efc_done_in[worldid]:
    return

  efc_prev_grad_out[worldid, dofid] = efc_grad_in[worldid, dofid]
  efc_prev_Mgrad_out[worldid, dofid] = efc_Mgrad_in[worldid, dofid]


@wp.kernel
def solve_beta(
  # Model:
  nv: int,
  # Data in:
  efc_grad_in: wp.array2d(dtype=float),
  efc_Mgrad_in: wp.array2d(dtype=float),
  efc_prev_grad_in: wp.array2d(dtype=float),
  efc_prev_Mgrad_in: wp.array2d(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_beta_out: wp.array(dtype=float),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  beta_num = float(0.0)
  beta_den = float(0.0)
  for dofid in range(nv):
    prev_Mgrad = efc_prev_Mgrad_in[worldid][dofid]
    beta_num += efc_grad_in[worldid, dofid] * (efc_Mgrad_in[worldid, dofid] - prev_Mgrad)
    beta_den += efc_prev_grad_in[worldid, dofid] * prev_Mgrad

  efc_beta_out[worldid] = wp.max(0.0, beta_num / wp.max(types.MJ_MINVAL, beta_den))


@wp.kernel
def solve_zero_search_dot(
  # Data in:
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_search_dot_out: wp.array(dtype=float),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  efc_search_dot_out[worldid] = 0.0


@wp.kernel
def solve_search_update(
  # Model:
  opt_solver: int,
  # Data in:
  efc_Mgrad_in: wp.array2d(dtype=float),
  efc_search_in: wp.array2d(dtype=float),
  efc_beta_in: wp.array(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  efc_search_out: wp.array2d(dtype=float),
  efc_search_dot_out: wp.array(dtype=float),
):
  worldid, dofid = wp.tid()

  if efc_done_in[worldid]:
    return

  search = -1.0 * efc_Mgrad_in[worldid, dofid]

  if opt_solver == types.SolverType.CG:
    search += efc_beta_in[worldid] * efc_search_in[worldid, dofid]

  efc_search_out[worldid, dofid] = search
  wp.atomic_add(efc_search_dot_out, worldid, search * search)


@wp.kernel
def solve_done(
  # Model:
  nv: int,
  opt_tolerance: wp.array(dtype=float),
  opt_iterations: int,
  stat_meaninertia: float,
  # Data in:
  efc_grad_dot_in: wp.array(dtype=float),
  efc_cost_in: wp.array(dtype=float),
  efc_prev_cost_in: wp.array(dtype=float),
  efc_done_in: wp.array(dtype=bool),
  # Data out:
  solver_niter_out: wp.array(dtype=int),
  efc_done_out: wp.array(dtype=bool),
  # Out:
  nsolving_out: wp.array(dtype=int),
):
  worldid = wp.tid()

  if efc_done_in[worldid]:
    return

  solver_niter_out[worldid] += 1
  tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]

  improvement = _rescale(nv, stat_meaninertia, efc_prev_cost_in[worldid] - efc_cost_in[worldid])
  gradient = _rescale(nv, stat_meaninertia, wp.sqrt(efc_grad_dot_in[worldid]))
  done = (improvement < tolerance) or (gradient < tolerance)
  if done or solver_niter_out[worldid] == opt_iterations:
    # if the solver has converged or the maximum number of iterations has been reached then
    # mark this world as done and remove it from the number of unconverged worlds
    efc_done_out[worldid] = True
    wp.atomic_add(nsolving_out, 0, -1)


@event_scope
def _solver_iteration(
  m: types.Model,
  d: types.Data,
  h: wp.array3d(dtype=float),
  hfactor: wp.array3d(dtype=float),
  step_size_cost: wp.array2d(dtype=float),
  nsolving: wp.array(dtype=int),
):
  _linesearch(m, d, step_size_cost)

  if m.opt.solver == types.SolverType.CG:
    wp.launch(
      solve_prev_grad_Mgrad,
      dim=(d.nworld, m.nv),
      inputs=[d.efc.grad, d.efc.Mgrad, d.efc.done],
      outputs=[d.efc.prev_grad, d.efc.prev_Mgrad],
    )

  _update_constraint(m, d)
  _update_gradient(m, d, h, hfactor)

  # polak-ribiere
  if m.opt.solver == types.SolverType.CG:
    wp.launch(
      solve_beta,
      dim=d.nworld,
      inputs=[m.nv, d.efc.grad, d.efc.Mgrad, d.efc.prev_grad, d.efc.prev_Mgrad, d.efc.done],
      outputs=[d.efc.beta],
    )

  wp.launch(solve_zero_search_dot, dim=(d.nworld), inputs=[d.efc.done], outputs=[d.efc.search_dot])

  wp.launch(
    solve_search_update,
    dim=(d.nworld, m.nv),
    inputs=[m.opt.solver, d.efc.Mgrad, d.efc.search, d.efc.beta, d.efc.done],
    outputs=[d.efc.search, d.efc.search_dot],
  )

  wp.launch(
    solve_done,
    dim=d.nworld,
    inputs=[
      m.nv,
      m.opt.tolerance,
      m.opt.iterations,
      m.stat.meaninertia,
      d.efc.grad_dot,
      d.efc.cost,
      d.efc.prev_cost,
      d.efc.done,
    ],
    outputs=[d.solver_niter, d.efc.done, nsolving],
  )


def create_context(
  m: types.Model, d: types.Data, h: wp.array3d(dtype=float), hfactor: wp.array3d(dtype=float), grad: bool = True
):
  # initialize some efc arrays
  wp.launch(
    solve_init_efc,
    dim=(d.nworld),
    outputs=[d.solver_niter, d.efc.search_dot, d.efc.cost, d.efc.done],
  )

  # jaref = d.efc_J @ d.qacc - d.efc_aref

  # if we are only using 1 thread, it makes sense to do more dofs as we can also skip the
  # init kernel. For more than 1 thread, dofs_per_thread is lower for better load balancing.

  if m.nv > 50:
    dofs_per_thread = 20
  else:
    dofs_per_thread = 50

  threads_per_efc = ceil(m.nv / dofs_per_thread)
  # we need to clear the jaref array if we're doing atomic adds.
  if threads_per_efc > 1:
    d.efc.Jaref.zero_()

  wp.launch(
    solve_init_jaref(m.nv, dofs_per_thread),
    dim=(d.nworld, d.njmax, threads_per_efc),
    inputs=[d.nefc, d.qacc, d.efc.J, d.efc.aref],
    outputs=[d.efc.Jaref],
  )

  # Ma = qM @ qacc
  support.mul_m(m, d, d.efc.Ma, d.qacc, skip=d.efc.done)

  _update_constraint(m, d)

  if grad:
    _update_gradient(m, d, h, hfactor)


@event_scope
def solve(m: types.Model, d: types.Data):
  if d.njmax == 0 or m.nv == 0:
    wp.copy(d.qacc, d.qacc_smooth)
    d.solver_niter.fill_(0)
  else:
    _solve(m, d)


def _solve(m: types.Model, d: types.Data):
  """Finds forces that satisfy constraints."""
  if not (m.opt.disableflags & types.DisableBit.WARMSTART):
    wp.copy(d.qacc, d.qacc_warmstart)
  else:
    wp.copy(d.qacc, d.qacc_smooth)

  # Newton solver Hessian
  if m.opt.solver == types.SolverType.NEWTON:
    h = wp.zeros((d.nworld, m.nv_pad, m.nv_pad), dtype=float)
    if m.nv > _BLOCK_CHOLESKY_DIM:
      hfactor = wp.zeros((d.nworld, m.nv_pad, m.nv_pad), dtype=float)
    else:
      hfactor = wp.empty((d.nworld, 0, 0), dtype=float)
  else:
    h = wp.empty((d.nworld, 0, 0), dtype=float)
    hfactor = wp.empty((d.nworld, 0, 0), dtype=float)

  # create context
  create_context(m, d, h, hfactor, grad=True)

  # search = -Mgrad
  wp.launch(
    solve_init_search,
    dim=(d.nworld, m.nv),
    inputs=[d.efc.Mgrad],
    outputs=[d.efc.search, d.efc.search_dot],
  )

  step_size_cost = wp.empty((d.nworld, m.opt.ls_iterations if m.opt.ls_parallel else 0), dtype=float)

  nsolving = wp.full(shape=(1,), value=d.nworld, dtype=int)
  if m.opt.iterations != 0 and m.opt.graph_conditional:
    # Note: the iteration kernel (indicated by while_body) is repeatedly launched
    # as long as condition_iteration is not zero.
    # condition_iteration is a warp array of size 1 and type int, it counts the number
    # of worlds that are not converged, it becomes 0 when all worlds are converged.
    # When the number of iterations reaches m.opt.iterations, solver_niter
    # becomes zero and all worlds are marked as converged to avoid an infinite loop.
    # note: we only launch the iteration kernel if everything is not done
    wp.capture_while(
      nsolving, while_body=_solver_iteration, m=m, d=d, h=h, hfactor=hfactor, step_size_cost=step_size_cost, nsolving=nsolving
    )
  else:
    # This branch is mostly for when JAX is used as it is currently not compatible
    # with CUDA graph conditional.
    # It should be removed when JAX becomes compatible.
    for _ in range(m.opt.iterations):
      _solver_iteration(m, d, h, hfactor, step_size_cost, nsolving)
