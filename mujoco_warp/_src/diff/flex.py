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
"""Analytic backward for the discrete step of stock 3D flex elasticity.

The discrete step with optional flex edge equalities (EqType.FLEX) solves, per world,

  A(q) a - b(q, v) - J(q)^T f(q, v, a) = 0,  v' = v + h a,  q' = q + h v'

with the effective metric A = M + h^2 K(q) + h d D_f(q) and b = f_spring(q) + f_damper(q, v)
+ qfrc_applied + J_body^T xfrc_applied + const - h K(q) v, where d is flex_damping, (K, D_f) are
the cached stretch Hessian for 3D elements (passive.flex_hessian), and (J, f) are the regularized
edge equality forces of constraint.make_constraint / regularize_constraint. The reverse solves the
symmetric adjoint system H lam = adj_a in _solve_adjoint_system (using the forward's solver:
matrix-free PCG/CG or Newton Cholesky) and accumulates
adj_theta = lam^T (db/dtheta - dA/dtheta a + d(J^T f)/dtheta) in _accumulate_flex_vjp by
differentiating one scalar per 3D element and per active edge equality. For 3D stretch, H is not
the force Jacobian: the default law clamps the tension in H only, and 3D stable Neo-Hookean (SNH)
elements project H to the PSD cone.

Only models whose every moving DOF belongs to a rooted 3-slider body are admitted, so the mass
matrix, gravity and vertex/body translational Jacobians do not depend on qpos.

Limits:
  - The 3D tension clamp, 3D PSD projection, and 3D default-law explicit damper gate
    (passive._flex_elasticity applies the explicit damper force only when
    kD = flex_damping / timestep > 0, unlike MuJoCo C which has no kD > 0 gate) make the step map
    piecewise smooth. The reverse differentiates the exact branch the forward takes at the
    evaluation point: on a boundary (including rest states and flex_damping == 0 for a 3D default
    flex compiled with edge Jacobians, where kD > 0 is False), it returns the one-sided derivative
    of the taken branch.
  - The position derivative of the SNH projection uses the analytic eigensystem of
    passive.snh_project, which is exact for an isotropic element metric only. Anisotropic SNH
    models and non-finite SNH metrics are rejected, as are flexes mixing the two laws: when
    first recorded, and on every eager recorded step while flex_stiffness requires a gradient
    on an SNH model (not during CUDA graph capture or replay). In-place edits of a
    flex_stiffness that is not differentiated, or on a model compiled without SNH elements, are
    not re-checked.
  - The model gate is cached per array identity: joint stiffness, damping, friction loss, limits
    or gravity compensation written in place after the first recorded step are not re-checked and
    give wrong gradients. Replace the array (or the Model) instead.
  - Accuracy follows the forward: float32 and iterative solves (eff_solve / CG) stopped at their
    respective tolerances, so gradients degrade with the stiffness to mass ratio like the step does.
  - With default opt.run_qacc_smooth=True and SolverType.NEWTON, worlds with no active equality
    rows (nefc == 0) solve the forward via iteration-capped PCG (eff_solve) while the backward
    linear solve is direct Newton Cholesky (when d.njmax > 0). Gradients are accurate only once
    opt.iterations converges the forward (forward OverflowType.ITERATIONS clear), or with
    opt.run_qacc_smooth=False so both forward and backward use direct Newton Cholesky.
  - During CUDA graph capture diff skips step_supported, so a model captured cold without an eager
    recorded step is not checked by the model or stiffness gate.
  - Backward scratch buffers and solver workspaces are allocated inline on device.
"""

import dataclasses
import weakref
from typing import TypeVar

import numpy as np
import warp as wp

from mujoco_warp._src import constraint
from mujoco_warp._src import derivative
from mujoco_warp._src import diff
from mujoco_warp._src import passive
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src.types import FLEX_STIFFNESS_3D
from mujoco_warp._src.types import MJ_MAXIMP
from mujoco_warp._src.types import MJ_MINIMP
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DeterminismType
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import EqType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import SolverType
from mujoco_warp._src.types import mat43
from mujoco_warp._src.types import mat63
from mujoco_warp._src.types import mat66
from mujoco_warp._src.types import vec5
from mujoco_warp._src.types import vec6
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

# The element kernels differentiate through @wp.funcs defined here and in passive.
wp.set_module_options({"enable_backward": True})

# eff_solve stops at a relative residual of 1e-5. One refinement pass reaches the float32 floor.
_SOLVE_REFINEMENTS = 1
# Minimum relative residual tolerance for adjoint PCG convergence checks (above float32 noise).
_PCG_CONV_TOL_FLOOR = 1.0e-4
# Eigen-directions of an element Hessian: three stretch modes and three symmetric and skew pairs.
_SNH_MODES = 9
# Relative residual of the isotropic fit above which an SNH element metric is rejected. The qpos
# gradient error measured against MuJoCo C is about 0.3 times the largest residual, on uniform
# grids and on a sliver-heavy tetrahedral mesh alike. A compiled isotropic material is at 1e-8 on
# a regular grid, but float32 storage of flex_stiffness and flex_vert0 alone reaches 4e-5 on
# millimetre elements in a decimetre bounding box, so the tolerance sits above that.
_SNH_ISOTROPY_TOL = 1.0e-4

_PARAMETER_FIELDS = (
  "flex_stiffness",
  "flex_damping",
  "flexedge_length0",
  "eq_solref",
  "eq_solimp",
)
_STEP_IN_FIELDS = ("qpos", "qvel")
_STEP_OUT_FIELDS = ("qpos", "qvel", "qacc")
_USER_FIELDS = ("qfrc_applied", "xfrc_applied")
_KINEMATICS_OUT_FIELDS = ("site_xpos", "xpos", "xquat")
_UNSUPPORTED_USER_FIELDS = ("ctrl", "mocap_pos", "mocap_quat")

_T = TypeVar("_T")


class vec9f(wp.types.vector(length=9, dtype=float)):
  pass


class mat66d(wp.types.matrix(shape=(6, 6), dtype=wp.float64)):
  pass


vec9 = vec9f


@wp.func
def _rows4(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3) -> mat43:
  return mat43(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2])


@wp.func
def _rows6(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, e: wp.vec3, f: wp.vec3) -> mat63:
  return mat63(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2], e[0], e[1], e[2], f[0], f[1], f[2])


@wp.func
def _edges(v: mat43) -> mat63:
  """Edge differences of per-vertex vectors, in passive.stretch_edge_endpoints(3) order."""
  return _rows6(v[0] - v[1], v[1] - v[2], v[2] - v[0], v[2] - v[3], v[0] - v[3], v[1] - v[3])


@wp.func
def _edge_dot(a: mat63, b: mat63) -> vec6:
  return vec6(
    wp.dot(a[0], b[0]), wp.dot(a[1], b[1]), wp.dot(a[2], b[2]), wp.dot(a[3], b[3]), wp.dot(a[4], b[4]), wp.dot(a[5], b[5])
  )


@wp.func
def _col_prod(r: mat63, i: int, j: int) -> vec6:
  return vec6(r[0, i] * r[0, j], r[1, i] * r[1, j], r[2, i] * r[2, j], r[3, i] * r[3, j], r[4, i] * r[4, j], r[5, i] * r[5, j])


@wp.func
def _max0(v: vec6) -> vec6:
  return vec6(wp.max(v[0], 0.0), wp.max(v[1], 0.0), wp.max(v[2], 0.0), wp.max(v[3], 0.0), wp.max(v[4], 0.0), wp.max(v[5], 0.0))


@wp.func
def _max_minval(v: vec6) -> vec6:
  return vec6(
    wp.max(v[0], MJ_MINVAL),
    wp.max(v[1], MJ_MINVAL),
    wp.max(v[2], MJ_MINVAL),
    wp.max(v[3], MJ_MINVAL),
    wp.max(v[4], MJ_MINVAL),
    wp.max(v[5], MJ_MINVAL),
  )


@wp.func
def _max0(v: wp.vec3) -> wp.vec3:
  return wp.vec3(wp.max(v[0], 0.0), wp.max(v[1], 0.0), wp.max(v[2], 0.0))


@wp.func
def _metric6(
  # Model:
  flex_stiffness: wp.array[float],
  # In:
  stiffness_adr: int,
) -> mat66:
  """Edge metric of passive.stretch_metric, unpacked without loops so the reverse is exact."""
  k00 = flex_stiffness[stiffness_adr]
  k01 = flex_stiffness[stiffness_adr + 1]
  k02 = flex_stiffness[stiffness_adr + 2]
  k03 = flex_stiffness[stiffness_adr + 3]
  k04 = flex_stiffness[stiffness_adr + 4]
  k05 = flex_stiffness[stiffness_adr + 5]
  k11 = flex_stiffness[stiffness_adr + 6]
  k12 = flex_stiffness[stiffness_adr + 7]
  k13 = flex_stiffness[stiffness_adr + 8]
  k14 = flex_stiffness[stiffness_adr + 9]
  k15 = flex_stiffness[stiffness_adr + 10]
  k22 = flex_stiffness[stiffness_adr + 11]
  k23 = flex_stiffness[stiffness_adr + 12]
  k24 = flex_stiffness[stiffness_adr + 13]
  k25 = flex_stiffness[stiffness_adr + 14]
  k33 = flex_stiffness[stiffness_adr + 15]
  k34 = flex_stiffness[stiffness_adr + 16]
  k35 = flex_stiffness[stiffness_adr + 17]
  k44 = flex_stiffness[stiffness_adr + 18]
  k45 = flex_stiffness[stiffness_adr + 19]
  k55 = flex_stiffness[stiffness_adr + 20]
  return mat66(
    k00, k01, k02, k03, k04, k05,
    k01, k11, k12, k13, k14, k15,
    k02, k12, k22, k23, k24, k25,
    k03, k13, k23, k33, k34, k35,
    k04, k14, k24, k34, k44, k45,
    k05, k15, k25, k35, k45, k55,
  )  # fmt: skip


@wp.func
def _det3(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> float:
  return wp.dot(a, wp.cross(b, c))


@wp.func
def _snh_cubic_jvp(s: vec6, w: vec6, gamma: float) -> vec6:
  """Derivative of passive.snh_cubic along w, the column action of passive._snh_cubic_metric."""
  a = s[0]
  b = s[2]
  c = s[4]
  d = 0.5 * (s[0] + s[2] - s[1])
  e = 0.5 * (s[0] + s[4] - s[5])
  f = 0.5 * (s[2] + s[4] - s[3])
  h0 = w[0]
  h1 = w[2]
  h2 = w[4]
  h3 = 0.5 * (w[0] + w[2] - w[1])
  h4 = 0.5 * (w[0] + w[4] - w[5])
  h5 = 0.5 * (w[2] + w[4] - w[3])
  dc0 = h1 * c + b * h2 - 2.0 * f * h5
  dc1 = h0 * c + a * h2 - 2.0 * e * h4
  dc2 = h0 * b + a * h1 - 2.0 * d * h3
  dc3 = h4 * f + e * h5 - h2 * d - c * h3
  dc4 = h3 * f + d * h5 - h1 * e - b * h4
  dc5 = h3 * e + d * h4 - h0 * f - a * h5
  scale = 2.0 * gamma
  return vec6(
    scale * (dc0 + dc3 + dc4),
    -scale * dc3,
    scale * (dc1 + dc3 + dc5),
    -scale * dc5,
    scale * (dc2 + dc4 + dc5),
    -scale * dc4,
  )


@wp.func
def _snh_tension(metric: mat66, s: vec6, gamma: float) -> vec6:
  return passive.snh_cubic(metric * s, s, gamma)


@wp.func
def _snh_metric_mul(metric: mat66, s: vec6, gamma: float, w: vec6) -> vec6:
  """Applies d(tension)/d(elongation), the edge metric plus the cubic term."""
  return metric * w + _snh_cubic_jvp(s, w, gamma)


@wp.func
def _vol_d1(g: mat63, da: mat63) -> float:
  """First directional derivative of det(x1 - x0, x2 - x0, x3 - x0)."""
  a = -g[0]
  b = g[2]
  c = -g[4]
  return _det3(-da[0], b, c) + _det3(a, da[2], c) + _det3(a, b, -da[4])


@wp.func
def _vol_d2(g: mat63, da: mat63, db: mat63) -> float:
  """Second directional derivative of det(x1 - x0, x2 - x0, x3 - x0)."""
  a = -g[0]
  b = g[2]
  c = -g[4]
  aa = -da[0]
  ba = da[2]
  ca = -da[4]
  ab = -db[0]
  bb = db[2]
  cb = -db[4]
  return _det3(aa, bb, c) + _det3(aa, b, cb) + _det3(ab, ba, c) + _det3(a, ba, cb) + _det3(ab, b, ca) + _det3(a, bb, ca)


@wp.func
def _snh_force_dot(metric: mat66, gamma: float, k22: float, k23: float, g: mat63, s: vec6, da: mat63) -> float:
  """Contracts the SNH spring force of passive._flex_elasticity with vertex vectors."""
  tension = _snh_tension(metric, s, gamma)
  volume = k23 * _det3(-g[0], g[2], -g[4])
  return -wp.dot(tension, _edge_dot(g, da)) - 2.0 * k22 * (volume - 1.0) * k23 * _vol_d1(g, da)


@wp.func
def _snh_energy_d2(metric: mat66, gamma: float, k22: float, k23: float, g: mat63, s: vec6, da: mat63, db: mat63) -> float:
  """Second directional derivative of the SNH energy, i.e. minus the spring force Jacobian."""
  tension = _snh_tension(metric, s, gamma)
  ga = _edge_dot(g, da)
  gb = _edge_dot(g, db)
  val = 2.0 * wp.dot(ga, _snh_metric_mul(metric, s, gamma, gb)) + wp.dot(tension, _edge_dot(da, db))
  volume = k23 * _det3(-g[0], g[2], -g[4])
  dva = k23 * _vol_d1(g, da)
  dvb = k23 * _vol_d1(g, db)
  return val + 2.0 * k22 * (dva * dvb + (volume - 1.0) * k23 * _vol_d2(g, da, db))


@wp.func
def _snh_spectrum(
  metric: mat66, gamma: float, k22: float, s: vec6, sigma: wp.vec3, r: mat63
) -> tuple[wp.mat33, wp.vec3, wp.vec3]:
  """Unclamped stretch block and twist/flip eigenvalues of passive.snh_project.

  r holds the rest edges in the frame of the right singular vectors.
  """
  tension = _snh_tension(metric, s, gamma)
  sq0 = _col_prod(r, 0, 0)
  sq1 = _col_prod(r, 1, 1)
  sq2 = _col_prod(r, 2, 2)
  prod0 = _snh_metric_mul(metric, s, gamma, sq0)
  prod1 = _snh_metric_mul(metric, s, gamma, sq1)
  prod2 = _snh_metric_mul(metric, s, gamma, sq2)
  geo = wp.vec3(wp.dot(tension, sq0), wp.dot(tension, sq1), wp.dot(tension, sq2))

  volume_stiffness = 2.0 * k22
  pressure = volume_stiffness * (sigma[0] * sigma[1] * sigma[2] - 1.0)
  cof = wp.vec3(sigma[1] * sigma[2], sigma[0] * sigma[2], sigma[0] * sigma[1])

  a00 = 2.0 * sigma[0] * sigma[0] * wp.dot(sq0, prod0) + volume_stiffness * cof[0] * cof[0] + geo[0]
  a11 = 2.0 * sigma[1] * sigma[1] * wp.dot(sq1, prod1) + volume_stiffness * cof[1] * cof[1] + geo[1]
  a22 = 2.0 * sigma[2] * sigma[2] * wp.dot(sq2, prod2) + volume_stiffness * cof[2] * cof[2] + geo[2]
  a01 = 2.0 * sigma[0] * sigma[1] * wp.dot(sq0, prod1) + volume_stiffness * cof[0] * cof[1] + pressure * sigma[2]
  a02 = 2.0 * sigma[0] * sigma[2] * wp.dot(sq0, prod2) + volume_stiffness * cof[0] * cof[2] + pressure * sigma[1]
  a12 = 2.0 * sigma[1] * sigma[2] * wp.dot(sq1, prod2) + volume_stiffness * cof[1] * cof[2] + pressure * sigma[0]
  stretch = wp.mat33(a00, a01, a02, a01, a11, a12, a02, a12, a22)

  dir01 = _col_prod(r, 0, 1)
  dir02 = _col_prod(r, 0, 2)
  dir12 = _col_prod(r, 1, 2)
  mat01 = wp.dot(dir01, _snh_metric_mul(metric, s, gamma, dir01))
  mat02 = wp.dot(dir02, _snh_metric_mul(metric, s, gamma, dir02))
  mat12 = wp.dot(dir12, _snh_metric_mul(metric, s, gamma, dir12))
  geo01 = 0.5 * (geo[0] + geo[1])
  geo02 = 0.5 * (geo[0] + geo[2])
  geo12 = 0.5 * (geo[1] + geo[2])
  sum01 = sigma[0] + sigma[1]
  sum02 = sigma[0] + sigma[2]
  sum12 = sigma[1] + sigma[2]
  dif01 = sigma[0] - sigma[1]
  dif02 = sigma[0] - sigma[2]
  dif12 = sigma[1] - sigma[2]
  symmetric = wp.vec3(
    sum01 * sum01 * mat01 + geo01 - pressure * sigma[2],
    sum02 * sum02 * mat02 + geo02 - pressure * sigma[1],
    sum12 * sum12 * mat12 + geo12 - pressure * sigma[0],
  )
  skew = wp.vec3(
    dif01 * dif01 * mat01 + geo01 + pressure * sigma[2],
    dif02 * dif02 * mat02 + geo02 + pressure * sigma[1],
    dif12 * dif12 * mat12 + geo12 + pressure * sigma[0],
  )
  return stretch, symmetric, skew


@wp.func
def _snh_projected_form(stretch: wp.mat33, symmetric: wp.vec3, skew: wp.vec3, a: wp.mat33, b: wp.mat33) -> float:
  """Bilinear form of passive.snh_projected_block summed over an element.

  a and b are the two vertex fields mapped to deformation gradient increments and rotated to the
  singular frame, U^T (sum_i a_i grad_i^T) V.
  """
  val = wp.dot(wp.vec3(a[0, 0], a[1, 1], a[2, 2]), stretch * wp.vec3(b[0, 0], b[1, 1], b[2, 2]))
  val += 0.5 * symmetric[0] * (a[0, 1] + a[1, 0]) * (b[0, 1] + b[1, 0])
  val += 0.5 * symmetric[1] * (a[0, 2] + a[2, 0]) * (b[0, 2] + b[2, 0])
  val += 0.5 * symmetric[2] * (a[1, 2] + a[2, 1]) * (b[1, 2] + b[2, 1])
  val += 0.5 * skew[0] * (a[0, 1] - a[1, 0]) * (b[0, 1] - b[1, 0])
  val += 0.5 * skew[1] * (a[0, 2] - a[2, 0]) * (b[0, 2] - b[2, 0])
  val += 0.5 * skew[2] * (a[1, 2] - a[2, 1]) * (b[1, 2] - b[2, 1])
  return val


@wp.func
def _grad_increment(e: mat63, grad0: wp.vec3, grad1: wp.vec3, grad2: wp.vec3) -> wp.mat33:
  """Deformation gradient increment of a vertex field given its edge differences."""
  return -wp.outer(e[0], grad0) + wp.outer(e[2], grad1) - wp.outer(e[4], grad2)


@wp.func
def _clamp_gap(a: float, b: float) -> float:
  """One minus the divided difference of max(0, x) between eigenvalues a and b."""
  if a >= 0.0 and b >= 0.0:
    return 0.0
  if a < 0.0 and b < 0.0:
    return 1.0
  hi = wp.max(a, b)
  lo = wp.min(a, b)
  return -lo / (hi - lo)


@wp.func
def _psd_clamp(A: wp.mat33) -> wp.mat33:
  """PSD projection of the stretch block, as in passive.snh_project."""
  if passive.snh_positive3(A):
    return A
  eig, Q = passive.snh_eigen3(A)
  D = wp.mat33(wp.max(eig[0], 0.0), 0.0, 0.0, 0.0, wp.max(eig[1], 0.0), 0.0, 0.0, 0.0, wp.max(eig[2], 0.0))
  return Q * D * wp.transpose(Q)


@wp.func_grad(_psd_clamp)
def _adj_psd_clamp(A: wp.mat33, ret_bar: wp.mat33):
  # Daleckii-Krein: the reverse of a spectral function scales the eigenbasis cotangent by divided
  # differences, which stay bounded at repeated eigenvalues.
  if passive.snh_positive3(A):
    wp.adjoint[A] += ret_bar
    return
  eig, Q = passive.snh_eigen3(A)
  M = wp.transpose(Q) * ret_bar * Q
  for i in range(3):
    for j in range(3):
      M[i, j] = M[i, j] * (1.0 - _clamp_gap(eig[i], eig[j]))
  wp.adjoint[A] += Q * M * wp.transpose(Q)


@wp.func
def _svd_frozen(F: wp.mat33) -> tuple[wp.mat33, wp.vec3, wp.mat33]:
  U, sigma, V = passive.snh_svd(F)
  return U, sigma, V


@wp.func_grad(_svd_frozen)
def _adj_svd_frozen(F: wp.mat33, u_bar: wp.mat33, sigma_bar: wp.vec3, v_bar: wp.mat33):
  # Held fixed: the element kernel treats the singular frame as data.
  wp.adjoint[F] += wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.func
def _eigen_frozen(A: wp.mat33) -> tuple[wp.vec3, wp.mat33]:
  eig, Q = passive.snh_eigen3(A)
  return eig, Q


@wp.func_grad(_eigen_frozen)
def _adj_eigen_frozen(A: wp.mat33, eig_bar: wp.vec3, q_bar: wp.mat33):
  # Held fixed: only used to weight frozen directions.
  wp.adjoint[A] += wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.func
def _mode(Q: wp.mat33, m: int) -> wp.mat33:
  """Unit eigen-direction m of the element Hessian in the singular frame.

  Modes 0-2 are the stretch block eigenvectors, 3-5 the symmetric and 6-8 the skew pair modes.
  """
  h = 0.7071067811865476
  if m < 3:
    return wp.mat33(Q[0, m], 0.0, 0.0, 0.0, Q[1, m], 0.0, 0.0, 0.0, Q[2, m])
  if m == 3:
    return wp.mat33(0.0, h, 0.0, h, 0.0, 0.0, 0.0, 0.0, 0.0)
  if m == 4:
    return wp.mat33(0.0, 0.0, h, 0.0, 0.0, 0.0, h, 0.0, 0.0)
  if m == 5:
    return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, h, 0.0, h, 0.0)
  if m == 6:
    return wp.mat33(0.0, h, 0.0, -h, 0.0, 0.0, 0.0, 0.0, 0.0)
  if m == 7:
    return wp.mat33(0.0, 0.0, h, 0.0, 0.0, 0.0, -h, 0.0, 0.0)
  return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, h, 0.0, -h, 0.0)


@wp.func
def _mode_coefficients(Q: wp.mat33, a: wp.mat33) -> vec9:
  return vec9(
    wp.ddot(_mode(Q, 0), a),
    wp.ddot(_mode(Q, 1), a),
    wp.ddot(_mode(Q, 2), a),
    wp.ddot(_mode(Q, 3), a),
    wp.ddot(_mode(Q, 4), a),
    wp.ddot(_mode(Q, 5), a),
    wp.ddot(_mode(Q, 6), a),
    wp.ddot(_mode(Q, 7), a),
    wp.ddot(_mode(Q, 8), a),
  )


@wp.func
def _spectrum_frozen(
  metric: mat66, gamma: float, k22: float, s: vec6, sigma: wp.vec3, r: mat63
) -> tuple[wp.mat33, wp.vec3, wp.vec3]:
  stretch, symmetric, skew = _snh_spectrum(metric, gamma, k22, s, sigma, r)
  return stretch, symmetric, skew


@wp.func_grad(_spectrum_frozen)
def _adj_spectrum_frozen(
  metric: mat66,
  gamma: float,
  k22: float,
  s: vec6,
  sigma: wp.vec3,
  r: mat63,
  stretch_bar: wp.mat33,
  symmetric_bar: wp.vec3,
  skew_bar: wp.vec3,
):
  # Held fixed: only used to weight frozen directions.
  wp.adjoint[sigma] += wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _clamp_directions(Q: wp.mat33, eig: vec9, alpha: vec9, beta: vec9, m: int, nmode: int) -> tuple[wp.mat33, wp.mat33]:
  """Frozen directions paired with clamped mode m in the Daleckii-Krein sum."""
  left = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  right = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for n in range(nmode):
    gap = _clamp_gap(eig[m], eig[n])
    direction = _mode(Q, n)
    left += (gap * alpha[n]) * direction
    if eig[n] >= 0.0:
      right += (gap * beta[n]) * direction
  return left, right


@wp.func_grad(_clamp_directions)
def _adj_clamp_directions(
  Q: wp.mat33, eig: vec9, alpha: vec9, beta: vec9, m: int, nmode: int, left_bar: wp.mat33, right_bar: wp.mat33
):
  # Held fixed.
  wp.adjoint[Q] += wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.func
def _mode_edges(U: wp.mat33, direction: wp.mat33, r: mat63) -> mat63:
  """Edge differences of the vertex displacement that realises a singular-frame direction."""
  P = U * direction
  return _rows6(P * r[0], P * r[1], P * r[2], P * r[3], P * r[4], P * r[5])


@wp.func
def _snh_clamp_correction(
  # In:
  metric: mat66,
  gamma: float,
  k22: float,
  k23: float,
  g: mat63,
  s: vec6,
  U: wp.mat33,
  r: mat63,
  Q: wp.mat33,
  eig: vec9,
  alpha: vec9,
  beta: vec9,
  nmode: int,
) -> float:
  """Part of the energy Hessian removed by the PSD projection, with frozen eigen-directions.

  Its derivative in the live edge vectors g is the Daleckii-Krein derivative of the negative part
  of the Hessian contracted with the frozen coefficients alpha and beta.
  """
  # nmode is a runtime value so the loop is not unrolled; each pass only accumulates.
  val = float(0.0)
  for m in range(nmode):
    if eig[m] < 0.0:
      left, right = _clamp_directions(Q, eig, alpha, beta, m, nmode)
      own = _mode(Q, m)
      val += _snh_energy_d2(metric, gamma, k22, k23, g, s, _mode_edges(U, left, r), _mode_edges(U, beta[m] * own, r))
      val += _snh_energy_d2(metric, gamma, k22, k23, g, s, _mode_edges(U, alpha[m] * own, r), _mode_edges(U, right, r))
  return val


@wp.func
def _phi_default(
  # In:
  metric: mat66,
  g: mat63,
  s: vec6,
  rate_mask: vec6,
  dv: mat63,
  dlam: mat63,
  dacc: mat63,
  damping: float,
  timestep: float,
) -> float:
  """lam^T (b - A a) of one element for the default (metric) law.

  g, dv, dlam and dacc are the edge differences of vertex positions, velocities, the adjoint
  solution and the acceleration; s is the squared-length elongation.
  """
  tension = metric * s
  glam = _edge_dot(g, dlam)

  # damper of passive._flex_elasticity: elongation rate d * ldot * (2 l - h * ldot) per edge.
  # passive._flex_elasticity gates the explicit damper force strictly on kD = d / h > 0.
  stretch_rate = vec6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  if damping > 0.0 and timestep > 0.0:
    rate = wp.cw_mul(_edge_dot(g, dv), rate_mask)
    lsq = _edge_dot(g, g)
    stretch_rate = damping * (2.0 * rate - timestep * wp.cw_div(wp.cw_mul(rate, rate), _max_minval(lsq)))
  phi = -wp.dot(tension + metric * stretch_rate, glam)

  # effective metric and velocity shift share the Hessian of passive.stretch_stiffness_block
  du = (timestep * timestep + timestep * damping) * dacc + timestep * dv
  phi -= 2.0 * wp.dot(glam, metric * _edge_dot(g, du)) + wp.dot(_max0(tension), _edge_dot(dlam, du))
  return phi


@wp.func
def _phi_snh(
  # In:
  metric: mat66,
  gamma: float,
  k22: float,
  k23: float,
  x: mat43,
  xdot: mat43,
  damping: float,
  metric_c: mat66,
  gamma_c: float,
  k22_c: float,
  k23_c: float,
  x_c: mat43,
  xdot_c: mat43,
  damping_c: float,
  lam: mat43,
  acc: mat43,
  len0sq: vec6,
  len0sq_c: vec6,
  rest0: wp.vec3,
  rest1: wp.vec3,
  rest2: wp.vec3,
  timestep: float,
  nmode: int,
) -> float:
  """lam^T (b - A a) of one stable Neo-Hookean element.

  Arguments suffixed _c are copies held fixed by the reverse.

  The projected Hessian is differentiated in two parts that sum to its total derivative: stiffness,
  rest length, velocity and damping through the forward's projection formula at a frozen position,
  and position through the energy's third derivative weighted by the frozen projection.
  """
  dlam = _edges(lam)

  # spring force, live in position, rest length and stiffness
  g = _edges(x)
  s = _edge_dot(g, g) - len0sq
  phi = _snh_force_dot(metric, gamma, k22, k23, g, s, dlam)

  # frozen singular frame of the deformation gradient
  g_c = _edges(x_c)
  s_c = _edge_dot(g_c, g_c) - len0sq
  grad0_c = wp.cross(rest1, rest2) * k23_c
  grad1_c = wp.cross(rest2, rest0) * k23_c
  grad2_c = wp.cross(rest0, rest1) * k23_c
  U, sigma_c, V = _svd_frozen(_grad_increment(g_c, grad0_c, grad1_c, grad2_c))
  UT = wp.transpose(U)
  VT = wp.transpose(V)
  r = _edges(_rows4(wp.vec3(0.0, 0.0, 0.0), VT * rest0, VT * rest1, VT * rest2))

  # projected Hessian form at the frozen position: effective metric, shift, and Rayleigh damper.
  # The deformation gradient is linear in k23, so the singular values scale and the frame is fixed.
  # passive._flex_stretch_mul_scale drops the Rayleigh damper (s1_in == 0) when timestep <= 0.
  damping_rayleigh = damping if timestep > 0.0 else 0.0
  du = _edges((timestep * timestep + timestep * damping) * acc + (timestep + damping_rayleigh) * xdot)
  grad0 = wp.cross(rest1, rest2) * k23
  grad1 = wp.cross(rest2, rest0) * k23
  grad2 = wp.cross(rest0, rest1) * k23
  sigma_scale = k23 / k23_c if k23_c != 0.0 else 0.0
  stretch, symmetric, skew = _snh_spectrum(metric, gamma, k22, s_c, sigma_c * sigma_scale, r)
  lam_hat = UT * _grad_increment(dlam, grad0, grad1, grad2) * V
  u_hat = UT * _grad_increment(du, grad0, grad1, grad2) * V
  phi -= _snh_projected_form(_psd_clamp(stretch), _max0(symmetric), _max0(skew), lam_hat, u_hat)

  # position derivative of the same form: unprojected energy Hessian minus the clamped part
  damping_rayleigh_c = damping_c if timestep > 0.0 else 0.0
  du_c = _edges((timestep * timestep + timestep * damping_c) * acc + (timestep + damping_rayleigh_c) * xdot_c)
  s_pos = _edge_dot(g, g) - len0sq_c
  s_c_frozen = _edge_dot(g_c, g_c) - len0sq_c
  phi -= _snh_energy_d2(metric_c, gamma_c, k22_c, k23_c, g, s_pos, dlam, du_c)
  stretch_c, symmetric_c, skew_c = _spectrum_frozen(metric_c, gamma_c, k22_c, s_c_frozen, sigma_c, r)
  clamped = (
    not passive.snh_positive3(stretch_c)
    or symmetric_c[0] < 0.0
    or symmetric_c[1] < 0.0
    or symmetric_c[2] < 0.0
    or skew_c[0] < 0.0
    or skew_c[1] < 0.0
    or skew_c[2] < 0.0
  )
  if clamped:
    eig3, Q = _eigen_frozen(stretch_c)
    eig = vec9(eig3[0], eig3[1], eig3[2], symmetric_c[0], symmetric_c[1], symmetric_c[2], skew_c[0], skew_c[1], skew_c[2])
    alpha = _mode_coefficients(Q, UT * _grad_increment(dlam, grad0_c, grad1_c, grad2_c) * V)
    beta = _mode_coefficients(Q, UT * _grad_increment(du_c, grad0_c, grad1_c, grad2_c) * V)
    phi += _snh_clamp_correction(metric_c, gamma_c, k22_c, k23_c, g, s_pos, U, r, Q, eig, alpha, beta, nmode)
  return phi


@wp.func
def _elem_len0sq(
  # Model:
  flex_elemedge: wp.array[int],
  flexedge_length0: wp.array[float],
  # In:
  edge_adr: int,
  elemedge_adr: int,
) -> vec6:
  l0 = flexedge_length0[edge_adr + flex_elemedge[elemedge_adr]]
  l1 = flexedge_length0[edge_adr + flex_elemedge[elemedge_adr + 1]]
  l2 = flexedge_length0[edge_adr + flex_elemedge[elemedge_adr + 2]]
  l3 = flexedge_length0[edge_adr + flex_elemedge[elemedge_adr + 3]]
  l4 = flexedge_length0[edge_adr + flex_elemedge[elemedge_adr + 4]]
  l5 = flexedge_length0[edge_adr + flex_elemedge[elemedge_adr + 5]]
  return vec6(l0 * l0, l1 * l1, l2 * l2, l3 * l3, l4 * l4, l5 * l5)


@wp.func
def _elem_rate_mask(
  # Model:
  flex_elemedge: wp.array[int],
  flexedge_J_rownnz: wp.array[int],
  # In:
  edge_adr: int,
  elemedge_adr: int,
) -> vec6:
  """One for edges whose velocity smooth._flex_edges computes, zero without an edge Jacobian."""
  return vec6(
    wp.where(flexedge_J_rownnz[edge_adr + flex_elemedge[elemedge_adr]] > 0, 1.0, 0.0),
    wp.where(flexedge_J_rownnz[edge_adr + flex_elemedge[elemedge_adr + 1]] > 0, 1.0, 0.0),
    wp.where(flexedge_J_rownnz[edge_adr + flex_elemedge[elemedge_adr + 2]] > 0, 1.0, 0.0),
    wp.where(flexedge_J_rownnz[edge_adr + flex_elemedge[elemedge_adr + 3]] > 0, 1.0, 0.0),
    wp.where(flexedge_J_rownnz[edge_adr + flex_elemedge[elemedge_adr + 4]] > 0, 1.0, 0.0),
    wp.where(flexedge_J_rownnz[edge_adr + flex_elemedge[elemedge_adr + 5]] > 0, 1.0, 0.0),
  )


@wp.func
def _elem_rows(
  # In:
  vec: wp.array2d[wp.vec3],
  worldid: int,
  verts: wp.vec4i,
) -> mat43:
  return _rows4(vec[worldid, verts[0]], vec[worldid, verts[1]], vec[worldid, verts[2]], vec[worldid, verts[3]])


# The element kernels are separate modules: a model compiles only the law it uses.
@cache_kernel
def _flex3d_phi_default(deterministic: bool = False):
  module_options = {"enable_backward": True}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_edgeadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_elemedgeadr: wp.array[int],
    flex_elem: wp.array[int],
    flex_elemedge: wp.array[int],
    flexedge_length0: wp.array[float],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_rigid: wp.array[bool],
    flexedge_J_rownnz: wp.array[int],
    flex_elemflexid: wp.array[int],
    # In:
    vert_pos: wp.array2d[wp.vec3],
    vert_vel: wp.array2d[wp.vec3],
    vert_lam: wp.array2d[wp.vec3],
    vert_acc: wp.array2d[wp.vec3],
    # Out:
    phi_out: wp.array2d[float],
  ):
    worldid, elemid = wp.tid()
    f = flex_elemflexid[elemid]
    if flex_dim[f] != 3 or flex_rigid[f]:
      return
    stiffness_adr_base = flex_stiffnessadr[f]
    if stiffness_adr_base < 0:
      return
    # MuJoCo convention: a zero first entry disables stretch stiffness for the whole flex
    if flex_stiffness[stiffness_adr_base] == 0.0:
      return
    local_elemid = elemid - flex_elemadr[f]
    stiffness_adr = stiffness_adr_base + local_elemid * wp.static(FLEX_STIFFNESS_3D)
    if flex_stiffness[stiffness_adr + 21] != 0.0:
      return

    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    vbase = flex_vertadr[f]
    elem_adr = flex_elemdataadr[f] + local_elemid * 4
    verts = wp.vec4i(
      vbase + flex_elem[elem_adr],
      vbase + flex_elem[elem_adr + 1],
      vbase + flex_elem[elem_adr + 2],
      vbase + flex_elem[elem_adr + 3],
    )
    edge_adr = flex_edgeadr[f]
    elemedge_adr = flex_elemedgeadr[f] + local_elemid * 6

    g = _edges(_elem_rows(vert_pos, worldid, verts))
    phi_out[worldid, elemid] = _phi_default(
      _metric6(flex_stiffness, stiffness_adr),
      g,
      _edge_dot(g, g) - _elem_len0sq(flex_elemedge, flexedge_length0, edge_adr, elemedge_adr),
      _elem_rate_mask(flex_elemedge, flexedge_J_rownnz, edge_adr, elemedge_adr),
      _edges(_elem_rows(vert_vel, worldid, verts)),
      _edges(_elem_rows(vert_lam, worldid, verts)),
      _edges(_elem_rows(vert_acc, worldid, verts)),
      flex_damping[f],
      timestep,
    )

  return kernel


@cache_kernel
def _flex3d_phi_snh(deterministic: bool = False):
  module_options = {"enable_backward": True}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_edgeadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_stiffnessadr: wp.array[int],
    flex_elemedgeadr: wp.array[int],
    flex_elem: wp.array[int],
    flex_elemedge: wp.array[int],
    flex_vert0: wp.array[wp.vec3],
    flexedge_length0: wp.array[float],
    flex_size: wp.array[wp.vec3],
    flex_stiffness: wp.array[float],
    flex_damping: wp.array[float],
    flex_rigid: wp.array[bool],
    flex_elemflexid: wp.array[int],
    # In:
    vert_pos: wp.array2d[wp.vec3],
    vert_vel: wp.array2d[wp.vec3],
    vert_pos_frozen: wp.array2d[wp.vec3],
    vert_vel_frozen: wp.array2d[wp.vec3],
    stiffness_frozen: wp.array[float],
    damping_frozen: wp.array[float],
    length0_frozen: wp.array[float],
    vert_lam: wp.array2d[wp.vec3],
    vert_acc: wp.array2d[wp.vec3],
    nmode: int,
    # Out:
    phi_out: wp.array2d[float],
  ):
    worldid, elemid = wp.tid()
    f = flex_elemflexid[elemid]
    if flex_dim[f] != 3 or flex_rigid[f]:
      return
    stiffness_adr_base = flex_stiffnessadr[f]
    if stiffness_adr_base < 0:
      return
    # MuJoCo convention: a zero first entry disables stretch stiffness for the whole flex
    if stiffness_frozen[stiffness_adr_base] == 0.0:
      return
    local_elemid = elemid - flex_elemadr[f]
    stiffness_adr = stiffness_adr_base + local_elemid * wp.static(FLEX_STIFFNESS_3D)
    if stiffness_frozen[stiffness_adr + 21] == 0.0:
      return

    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    vbase = flex_vertadr[f]
    elem_adr = flex_elemdataadr[f] + local_elemid * 4
    verts = wp.vec4i(
      vbase + flex_elem[elem_adr],
      vbase + flex_elem[elem_adr + 1],
      vbase + flex_elem[elem_adr + 2],
      vbase + flex_elem[elem_adr + 3],
    )
    edge_adr = flex_edgeadr[f]
    elemedge_adr = flex_elemedgeadr[f] + local_elemid * 6

    # rest edges from vertex 0, as in passive.snh_project
    vert0 = flex_vert0[verts[0]]
    scale = 2.0 * flex_size[f]
    phi_out[worldid, elemid] = _phi_snh(
      _metric6(flex_stiffness, stiffness_adr),
      flex_stiffness[stiffness_adr + 21],
      flex_stiffness[stiffness_adr + 22],
      flex_stiffness[stiffness_adr + 23],
      _elem_rows(vert_pos, worldid, verts),
      _elem_rows(vert_vel, worldid, verts),
      flex_damping[f],
      _metric6(stiffness_frozen, stiffness_adr),
      stiffness_frozen[stiffness_adr + 21],
      stiffness_frozen[stiffness_adr + 22],
      stiffness_frozen[stiffness_adr + 23],
      _elem_rows(vert_pos_frozen, worldid, verts),
      _elem_rows(vert_vel_frozen, worldid, verts),
      damping_frozen[f],
      _elem_rows(vert_lam, worldid, verts),
      _elem_rows(vert_acc, worldid, verts),
      _elem_len0sq(flex_elemedge, flexedge_length0, edge_adr, elemedge_adr),
      _elem_len0sq(flex_elemedge, length0_frozen, edge_adr, elemedge_adr),
      wp.cw_mul(scale, flex_vert0[verts[1]] - vert0),
      wp.cw_mul(scale, flex_vert0[verts[2]] - vert0),
      wp.cw_mul(scale, flex_vert0[verts[3]] - vert0),
      timestep,
      nmode,
    )

  return kernel


@wp.func
def _eq_kbimp(
  # In:
  solref: wp.vec2,
  solimp: vec5,
  pos_imp: float,
) -> wp.vec3:
  """Constraint (k, b, imp) matching constraint._contact_kbimp for discrete equality rows.

  Evaluates the smooth interpolation at dmin == dmax (when width > MJ_MINVAL) rather than the
  0.5 * (dmin + dmax) fast path so derivatives with respect to (dmin, dmax) match MuJoCo C, while
  matching constraint._contact_kbimp's clamp to dmax when dmin > dmax and power > 1.
  """
  timeconst = solref[0]
  dampratio = solref[1]
  dmin = wp.clamp(solimp[0], MJ_MINIMP, MJ_MAXIMP)
  dmax = wp.clamp(solimp[1], MJ_MINIMP, MJ_MAXIMP)
  width_raw = solimp[2]
  width = wp.max(MJ_MINVAL, width_raw)
  mid = wp.clamp(solimp[3], MJ_MINIMP, MJ_MAXIMP)
  power = wp.max(1.0, solimp[4])

  dmax_sq = dmax * dmax
  if solref[0] <= 0.0:
    k = -solref[0] / wp.max(MJ_MINVAL, dmax_sq)
  else:
    k = 1.0 / wp.max(MJ_MINVAL, dmax_sq * timeconst * timeconst * dampratio * dampratio)
  if solref[1] <= 0.0:
    b = -solref[1] / wp.max(MJ_MINVAL, dmax)
  else:
    b = 2.0 / wp.max(MJ_MINVAL, dmax * timeconst)

  imp_x = wp.abs(pos_imp) / width
  if width_raw <= MJ_MINVAL:
    imp = 0.5 * (dmin + dmax)
  elif imp_x <= 0.0:
    imp = dmin
  elif imp_x >= 1.0:
    imp = dmax
  elif power == 1.0:
    imp = dmin + imp_x * (dmax - dmin)
  elif imp_x <= mid:
    imp_y = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
    if dmin > dmax:
      imp = dmax
    else:
      imp = dmin + imp_y * (dmax - dmin)
  else:
    imp_y = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)
    if dmin > dmax:
      imp = dmax
    else:
      imp = dmin + imp_y * (dmax - dmin)

  return wp.vec3(k, b, imp)


@cache_kernel
def _flex_eq_phi(deterministic: bool = False):
  module_options = {"enable_backward": True}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    opt_timestep: wp.array[float],
    flex_vertadr: wp.array[int],
    flex_edgeadr: wp.array[int],
    flex_edgenum: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flexedge_length0: wp.array[float],
    flexedge_invweight0: wp.array[float],
    eq_obj1id: wp.array[int],
    eq_solref: wp.array2d[wp.vec2],
    eq_solimp: wp.array2d[vec5],
    eq_flex_adr: wp.array[int],
    flexedge_efcadr: wp.array[int],
    # Data in:
    eq_active_in: wp.array2d[bool],
    # In:
    vert_pos: wp.array2d[wp.vec3],
    vert_vel: wp.array2d[wp.vec3],
    vert_lam: wp.array2d[wp.vec3],
    vert_acc: wp.array2d[wp.vec3],
    # Out:
    phi_out: wp.array3d[float],
  ):
    worldid, eqflexid, edgeid = wp.tid()
    eqid = eq_flex_adr[eqflexid]
    if not eq_active_in[worldid, eqid]:
      return
    f = eq_obj1id[eqid]
    edgeadr = flex_edgeadr[f]
    if edgeid < edgeadr or edgeid >= edgeadr + flex_edgenum[f]:
      return
    if flexedge_efcadr[edgeid] < 0:
      return

    vbase = flex_vertadr[f]
    v = flex_edge[edgeid]
    v0 = vbase + v[0]
    v1 = vbase + v[1]

    vec = vert_pos[worldid, v1] - vert_pos[worldid, v0]
    len_sq = wp.dot(vec, vec)
    # Zero-length edges have J = 0 in smooth._flex_edges; skip before sqrt to avoid Inf gradients.
    if len_sq <= MJ_MINVAL * MJ_MINVAL:
      return
    length = wp.sqrt(len_sq)
    axis = vec / length

    pos = length - flexedge_length0[edgeid]
    vel = wp.dot(axis, vert_vel[worldid, v1] - vert_vel[worldid, v0])
    j_acc = wp.dot(axis, vert_acc[worldid, v1] - vert_acc[worldid, v0])
    j_lam = wp.dot(axis, vert_lam[worldid, v1] - vert_lam[worldid, v0])

    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    solref = eq_solref[worldid % eq_solref.shape[0], eqid]
    solimp = eq_solimp[worldid % eq_solimp.shape[0], eqid]
    kbimp = _eq_kbimp(solref, solimp, pos)
    k = kbimp[0]
    b = kbimp[1]
    imp = kbimp[2]

    invweight = flexedge_invweight0[edgeid]
    f_denom = 1.0 + timestep * b + timestep * timestep * k * imp
    R0 = wp.max(MJ_MINVAL, (1.0 - imp) * invweight / imp)
    Rmin = wp.max(MJ_MINVAL, (1.0 - MJ_MAXIMP) * invweight / MJ_MAXIMP)
    R = wp.max(R0 / f_denom, Rmin)
    D = 1.0 / R

    aref = (-k * imp * pos - (b + timestep * k * imp) * vel) / f_denom
    phi_out[worldid, eqflexid, edgeid] = -j_lam * D * (j_acc - aref)

  return kernel


@wp.kernel(enable_backward=False)
def _integrator_vjp(
  # Model:
  opt_timestep: wp.array[float],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # In:
  adj_qpos_next: wp.array2d[float],
  adj_qvel_next: wp.array2d[float],
  adj_qacc_next: wp.array2d[float],
  # Out:
  adj_qvel_out: wp.array2d[float],
  adj_qacc_out: wp.array2d[float],
  nonfinite_out: wp.array[int],
):
  worldid, dofid = wp.tid()
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  # v' = v + h a, q' = q + h v' for slide joints
  qpos_bar = float(0.0)
  if adj_qpos_next:
    qpos_bar = adj_qpos_next[worldid, dofid]
  qvel_bar = float(0.0)
  if adj_qvel_next:
    qvel_bar = adj_qvel_next[worldid, dofid]
  qacc_bar = float(0.0)
  if adj_qacc_next:
    qacc_bar = adj_qacc_next[worldid, dofid]
  vel_bar = qvel_bar + timestep * qpos_bar
  a_bar = timestep * vel_bar + qacc_bar
  adj_qvel_out[worldid, dofid] = vel_bar
  adj_qacc_out[worldid, dofid] = a_bar
  if not (
    wp.isfinite(qpos_in[worldid, dofid])
    and wp.isfinite(qvel_in[worldid, dofid])
    and wp.isfinite(qacc_in[worldid, dofid])
    and wp.isfinite(a_bar)
  ):
    wp.atomic_max(nonfinite_out, worldid, 1)  # kernel_analyzer: ignore[atomic]


@wp.kernel(enable_backward=False)
def _write_input_adjoints(
  # In:
  adj_qpos_next: wp.array2d[float],
  adj_qvel: wp.array2d[float],
  res_qpos: wp.array2d[float],
  res_qvel: wp.array2d[float],
  nonfinite: wp.array[int],
  # Out:
  adj_qpos_out: wp.array2d[float],
  adj_qvel_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  bad = nonfinite[worldid] != 0
  if adj_qpos_out:
    if bad:
      adj_qpos_out[worldid, dofid] = wp.nan
    else:
      qpos_bar = res_qpos[worldid, dofid]
      if adj_qpos_next:
        qpos_bar += adj_qpos_next[worldid, dofid]
      adj_qpos_out[worldid, dofid] += qpos_bar
  if adj_qvel_out:
    if bad:
      adj_qvel_out[worldid, dofid] = wp.nan
    else:
      adj_qvel_out[worldid, dofid] += adj_qvel[worldid, dofid] + res_qvel[worldid, dofid]


@wp.kernel(enable_backward=False)
def _slide_kinematics_vjp(
  # Model:
  nsite: int,
  jnt_qposadr: wp.array[int],
  jnt_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  body_subtreenum: wp.array[int],
  # Data in:
  xaxis_in: wp.array2d[wp.vec3],
  # In:
  adj_xpos: wp.array2d[wp.vec3],
  adj_site_xpos: wp.array2d[wp.vec3],
  # Out:
  res_qpos_out: wp.array2d[float],
):
  """Maps body and site position cotangents to slide-joint qpos cotangents."""
  worldid, jntid = wp.tid()
  bodyid = jnt_bodyid[jntid]
  subtree_end = bodyid + body_subtreenum[bodyid]
  force = wp.vec3(0.0)
  if adj_xpos:
    for child in range(bodyid, subtree_end):
      force += adj_xpos[worldid, child]
  if adj_site_xpos:
    for siteid in range(nsite):
      sb = site_bodyid[siteid]
      if sb >= bodyid and sb < subtree_end:
        force += adj_site_xpos[worldid, siteid]
  res_qpos_out[worldid, jnt_qposadr[jntid]] += wp.dot(force, xaxis_in[worldid, jntid])


@wp.kernel(enable_backward=False)
def _slide_xfrc_vjp(
  # Model:
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  lam: wp.array2d[float],
  # Out:
  adj_xfrc_applied_out: wp.array2d[wp.spatial_vector],
):
  """Maps solve cotangent lam to body xfrc_applied cotangents for rooted 3-slider bodies."""
  worldid, bodyid = wp.tid()
  weldid = body_weldid[bodyid]
  dofnum = body_dofnum[weldid]
  if dofnum == 0:
    return
  dofadr = body_dofadr[weldid]
  force = wp.vec3(0.0)
  for k in range(dofnum):
    force += lam[worldid, dofadr + k] * wp.spatial_bottom(cdof_in[worldid, dofadr + k])
  # xfrc_applied stores force followed by torque, unlike motion-space spatial vectors.
  adj_xfrc_applied_out[worldid, bodyid] += wp.spatial_vector(force, wp.vec3(0.0))


@wp.kernel(enable_backward=False)
def _accum_spatial(
  # In:
  adj_in: wp.array2d[wp.spatial_vector],
  # Out:
  adj_out: wp.array2d[wp.spatial_vector],
):
  w, i = wp.tid()
  adj_out[w, i] += adj_in[w, i]


@wp.kernel(enable_backward=False)
def _sanitize_bwd_state(
  # Model:
  qpos0: wp.array2d[float],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # In:
  rhs: wp.array2d[float],
  nonfinite: wp.array[int],
  # Data out:
  qpos_out: wp.array2d[float],
  qvel_out: wp.array2d[float],
  qacc_out: wp.array2d[float],
  # Out:
  rhs_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if nonfinite[worldid] != 0:
    qpos_out[worldid, dofid] = qpos0[worldid % qpos0.shape[0], dofid]
    qvel_out[worldid, dofid] = 0.0
    qacc_out[worldid, dofid] = 0.0
    rhs_out[worldid, dofid] = 0.0
  else:
    qpos_out[worldid, dofid] = qpos_in[worldid, dofid]
    qvel_out[worldid, dofid] = qvel_in[worldid, dofid]
    qacc_out[worldid, dofid] = qacc_in[worldid, dofid]
    rhs_out[worldid, dofid] = rhs[worldid, dofid]


@wp.kernel(enable_backward=False)
def _fill_nonfinite_nan(
  # In:
  nonfinite: wp.array[int],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if nonfinite[worldid] != 0:
    vec_out[worldid, dofid] = wp.nan


@wp.kernel(enable_backward=False)
def _or_overflow(
  # In:
  src: wp.array[int],
  # Out:
  dst_out: wp.array[int],
):
  worldid = wp.tid()
  dst_out[worldid] = dst_out[worldid] | src[worldid]


@wp.kernel(enable_backward=False)
def _mask_empty_efc(
  # Data in:
  nefc_in: wp.array[int],
  # In:
  vec: wp.array2d[float],
  # Out:
  masked_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  masked_out[worldid, dofid] = vec[worldid, dofid] if nefc_in[worldid] == 0 else 0.0


@cache_kernel
def _pcg_residual_and_scale_squares(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Data in:
    qHDiagInv_in: wp.array2d[float],
    # In:
    rhs: wp.array2d[float],
    ax: wp.array2d[float],
    x: wp.array2d[float],
    # Out:
    res_squares_out: wp.array[float],
    scale_squares_out: wp.array[float],
  ):
    worldid, dofid = wp.tid()
    r = rhs[worldid, dofid] - ax[worldid, dofid]
    wp.atomic_add(res_squares_out, worldid, r * r)
    qh_inv = qHDiagInv_in[worldid, dofid]
    if qh_inv > MJ_MINVAL:
      sx = x[worldid, dofid] / qh_inv
      wp.atomic_add(scale_squares_out, worldid, sx * sx)

  return kernel


@cache_kernel
def _check_pcg_convergence(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    opt_tolerance: wp.array[float],
    # In:
    rhs_squares: wp.array[float],
    res_squares: wp.array[float],
    scale_squares: wp.array[float],
    # Data out:
    overflow_out: wp.array[int],
  ):
    worldid = wp.tid()
    rhs_sq = rhs_squares[worldid]
    if not wp.isfinite(rhs_sq) or rhs_sq <= 0.0:
      return
    tol = wp.max(opt_tolerance[worldid % opt_tolerance.shape[0]], _PCG_CONV_TOL_FLOOR)
    res_sq = res_squares[worldid]
    scale_sq = scale_squares[worldid]
    if not wp.isfinite(res_sq) or not wp.isfinite(scale_sq) or wp.sqrt(res_sq) > tol * wp.sqrt(wp.max(1.0, scale_sq)):
      if wp.static(bool(warn_overflow & OverflowType.ITERATIONS)):
        wp.printf(
          "adjoint PCG did not converge (rel residual > %g) - consider increasing opt.iterations\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.ITERATIONS (or = 0 for all)\n",
          tol,
        )
      overflow_out[worldid] = overflow_out[worldid] | OverflowType.ITERATIONS

  return kernel


@cache_kernel
def _sum_squares(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # In:
    vec: wp.array2d[float],
    # Out:
    sum_out: wp.array[float],
  ):
    worldid, dofid = wp.tid()
    v = vec[worldid, dofid]
    wp.atomic_add(sum_out, worldid, v * v)

  return kernel


@wp.kernel(enable_backward=False)
def _normalize(
  # In:
  vec: wp.array2d[float],
  sum_squares: wp.array[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  sq = sum_squares[worldid]
  if wp.isfinite(sq) and sq > 0.0:
    vec_out[worldid, dofid] = vec[worldid, dofid] / wp.sqrt(sq)
  else:
    vec_out[worldid, dofid] = 0.0


@wp.kernel(enable_backward=False)
def _residual(
  # In:
  rhs: wp.array2d[float],
  # Out:
  res_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  res_out[worldid, dofid] = rhs[worldid, dofid] - res_out[worldid, dofid]


@wp.kernel(enable_backward=False)
def _add_scaled(
  # In:
  vec: wp.array2d[float],
  sum_squares: wp.array[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  vec_out[worldid, dofid] += wp.sqrt(sum_squares[worldid]) * vec[worldid, dofid]


@wp.kernel(enable_backward=False)
def _scale(
  # In:
  sum_squares: wp.array[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  sq = sum_squares[worldid]
  if not wp.isfinite(sq):
    vec_out[worldid, dofid] = wp.nan
  else:
    vec_out[worldid, dofid] *= wp.sqrt(sq)


def _normalized(vec: wp.array, vec_out: wp.array, deterministic: bool = False) -> wp.array:
  """Writes vec scaled to unit norm per world and returns the squared norms."""
  device = vec.device
  sum_squares = wp.zeros(vec.shape[0], dtype=float, device=device)
  wp.launch(_sum_squares(deterministic), dim=vec.shape, inputs=[vec], outputs=[sum_squares], device=device)
  wp.launch(_normalize, dim=vec.shape, inputs=[vec, sum_squares], outputs=[vec_out], device=device)
  return sum_squares


@event_scope
def metric_solve(m: Model, d: Data, x: wp.array, rhs: wp.array, overflow_out: wp.array | None = None):
  """Solves the effective-metric system A x = rhs with the forward's matrix-free PCG.

  derivative.eff_solve returns zero for a right-hand side below MJ_MINVAL and stops at a relative
  residual of 1e-5. Cotangents can be arbitrarily small and need more digits, so the right-hand
  side is normalised per world and the solution is refined on the true residual.
  """
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  unit = wp.empty_like(rhs)
  work = wp.empty_like(rhs)
  delta = wp.empty_like(rhs)
  efm_con, flex_scratch = derivative.eff_mul_m_scratch(m, d)
  m_inner = dataclasses.replace(
    m,
    opt=dataclasses.replace(m.opt, warn_overflow=m.opt.warn_overflow & ~int(OverflowType.ITERATIONS)),
  )
  rhs_squares = _normalized(rhs, unit, det_atomics)
  derivative.eff_solve(m_inner, d, x, unit)
  for _ in range(_SOLVE_REFINEMENTS):
    derivative.eff_mul_m(m, d, work, x, efm_con=efm_con, flex_scratch=flex_scratch)
    wp.launch(_residual, dim=rhs.shape, inputs=[unit], outputs=[work])
    res_squares = _normalized(work, work, det_atomics)
    derivative.eff_solve(m_inner, d, delta, work)
    wp.launch(_add_scaled, dim=rhs.shape, inputs=[delta, res_squares], outputs=[x])
  if overflow_out is not None:
    derivative.eff_mul_m(m, d, work, x, efm_con=efm_con, flex_scratch=flex_scratch)
    res_squares = wp.zeros(rhs.shape[0], dtype=float, device=rhs.device)
    scale_squares = wp.zeros(rhs.shape[0], dtype=float, device=rhs.device)
    wp.launch(
      _pcg_residual_and_scale_squares(det_atomics),
      dim=rhs.shape,
      inputs=[d.qHDiagInv, unit, work, x],
      outputs=[res_squares, scale_squares],
      device=rhs.device,
    )
    wp.launch(
      _check_pcg_convergence(int(m.opt.warn_overflow)),
      dim=rhs.shape[0],
      inputs=[m.opt.tolerance, rhs_squares, res_squares, scale_squares],
      outputs=[overflow_out],
      device=rhs.device,
    )
  wp.launch(_scale, dim=rhs.shape, inputs=[rhs_squares], outputs=[x])


@cache_kernel
def _add_jtdj_mul(is_sparse: bool, max_records: int = 0, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN
    module_options["deterministic_max_records"] = max_records

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # Model:
    nv: int,
    # Data in:
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    # In:
    vec_in: wp.array2d[float],
    # Out:
    res_out: wp.array2d[float],
  ):
    worldid, efcid = wp.tid()
    if efcid >= nefc_in[worldid]:
      return
    d_val = efc_D_in[worldid, efcid]
    if d_val == 0.0:
      return
    if wp.static(is_sparse):
      rownnz = efc_J_rownnz_in[worldid, efcid]
      rowadr = efc_J_rowadr_in[worldid, efcid]
      jx = float(0.0)
      for i in range(rownnz):
        sid = rowadr + i
        jx += efc_J_in[worldid, 0, sid] * vec_in[worldid, efc_J_colind_in[worldid, 0, sid]]
      djx = d_val * jx
      for i in range(rownnz):
        sid = rowadr + i
        col = efc_J_colind_in[worldid, 0, sid]
        wp.atomic_add(res_out, worldid, col, efc_J_in[worldid, 0, sid] * djx)
    else:
      jx = float(0.0)
      for i in range(nv):
        jx += efc_J_in[worldid, efcid, i] * vec_in[worldid, i]
      djx = d_val * jx
      for i in range(nv):
        jval = efc_J_in[worldid, efcid, i]
        if jval != 0.0:
          wp.atomic_add(res_out, worldid, i, jval * djx)

  return kernel


@cache_kernel
def _dot_worlds(deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @wp.kernel(module="unique", module_options=module_options)
  def kernel(
    # In:
    a: wp.array2d[float],
    b: wp.array2d[float],
    # Out:
    dot_out: wp.array[float],
  ):
    worldid, dofid = wp.tid()
    wp.atomic_add(dot_out, worldid, a[worldid, dofid] * b[worldid, dofid])

  return kernel


@wp.kernel(enable_backward=False)
def _cg_scale_factors(
  # In:
  norm_squares: wp.array[float],
  unit_dot_z: wp.array[float],
  inertia_scale: wp.array[float],
  # Out:
  scale_out: wp.array[float],
):
  worldid = wp.tid()
  sq = norm_squares[worldid]
  udz = unit_dot_z[worldid]
  if not wp.isfinite(sq) or sq <= 0.0 or not wp.isfinite(udz):
    scale_out[worldid] = 0.0
    return
  isc = inertia_scale[worldid]
  s_p = wp.sqrt(2.0 / wp.max(MJ_MINVAL, isc * udz))
  s_g = 1.0 / wp.max(MJ_MINVAL, isc)
  scale_out[worldid] = wp.max(s_p, s_g)


@wp.kernel(enable_backward=False)
def _mul_scale(
  # In:
  vec: wp.array2d[float],
  scale: wp.array[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  vec_out[worldid, dofid] = scale[worldid] * vec[worldid, dofid]


@wp.kernel(enable_backward=False)
def _div_scale(
  # In:
  scale: wp.array[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  s = scale[worldid]
  vec_out[worldid, dofid] = vec_out[worldid, dofid] / s if s > 0.0 else 0.0


@wp.kernel(enable_backward=False)
def _copy_scaled(
  # In:
  vec: wp.array2d[float],
  sum_squares: wp.array[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  sq = sum_squares[worldid]
  if not wp.isfinite(sq):
    vec_out[worldid, dofid] = wp.nan
  else:
    vec_out[worldid, dofid] = wp.sqrt(sq) * vec[worldid, dofid]


@event_scope
def _cg_constrained_solve(
  m: Model,
  d_bwd: Data,
  x: wp.array,
  rhs: wp.array,
  overflow_out: wp.array | None = None,
):
  """Solves (A + J^T D J) x = rhs matrix-free via the forward's CG solver.

  The right-hand side is normalised per world and scaled so both the initial preconditioned energy
  0.5 * isc * b^T P^{-1} b and gradient norm isc * |b| are at least 1, preventing early termination
  on stiff constraints or small cotangents without modifying solver tolerances or iterations. One
  iterative refinement pass on the true residual recovers full float32 accuracy.
  """
  nworld, nv = rhs.shape
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  unit = wp.empty_like(rhs)
  work = wp.empty_like(rhs)
  delta = wp.empty_like(rhs)
  z = wp.empty_like(rhs)
  scaled_rhs = wp.empty_like(rhs)
  uncon_rhs = wp.empty_like(rhs)
  uncon_sol = wp.empty_like(rhs)
  unit_dot_z = wp.empty(nworld, dtype=float)
  scale = wp.empty(nworld, dtype=float)
  inertia_scale = wp.empty(nworld, dtype=float)
  zero_nv = wp.zeros((nworld, nv), dtype=float)
  zero_efc = wp.zeros((nworld, d_bwd.njmax), dtype=float)
  efm_con, flex_scratch = derivative.eff_mul_m_scratch(m, d_bwd)

  solver.launch_inertia_scale(m, d_bwd.M, inertia_scale)
  tw_diag = wp.empty((nworld, nv), dtype=float) if m.nefmdof > 0 else None
  if tw_diag is not None:
    derivative.eff_prec_fold(m, d_bwd, tw_diag=tw_diag)
  solve_ld_scratch = (
    wp.empty((nworld, m.nv_pad), dtype=float) if d_bwd.qLD.shape[1] > m.qLD_block_total and det_atomics else None
  )

  opt_lin = dataclasses.replace(
    m.opt,
    disableflags=m.opt.disableflags & ~int(DisableBit.WARMSTART),
    run_qacc_smooth=True,
  )
  m_lin = dataclasses.replace(m, opt=opt_lin)
  d_bwd.overflow.zero_()

  def _cg_pass(b_unit: wp.array, b_squares: wp.array, x_out: wp.array):
    if m.nefmK > 0 or m.nflexrodhinge > 0 or m.efm0_active:
      derivative.eff_prec(m, d_bwd, z, b_unit, x_scratch=solve_ld_scratch, tw_diag=tw_diag)
    else:
      smooth.solve_LD(m, d_bwd, d_bwd.qHLD, d_bwd.qHDiagInv, z, b_unit, x_scratch=solve_ld_scratch)
    unit_dot_z.zero_()
    wp.launch(_dot_worlds(det_atomics), dim=rhs.shape, inputs=[b_unit, z], outputs=[unit_dot_z])
    wp.launch(_cg_scale_factors, dim=nworld, inputs=[b_squares, unit_dot_z, inertia_scale], outputs=[scale])
    wp.launch(_mul_scale, dim=rhs.shape, inputs=[b_unit, scale], outputs=[scaled_rhs])
    # solver.solve skips CG on worlds with nefc == 0 and returns qacc_smooth; solve those worlds
    # with eff_solve (0 PCG iterations when every world has nefc > 0).
    wp.launch(_mask_empty_efc, dim=rhs.shape, inputs=[d_bwd.nefc, b_unit], outputs=[uncon_rhs])
    derivative.eff_solve(m_lin, d_bwd, uncon_sol, uncon_rhs)
    wp.launch(_mul_scale, dim=rhs.shape, inputs=[uncon_sol, scale], outputs=[uncon_sol])
    d_lin = dataclasses.replace(
      d_bwd,
      qfrc_smooth=scaled_rhs,
      efm_c=zero_nv,
      efm_ca=zero_nv,
      qacc_warmstart=zero_nv,
      qacc_smooth=uncon_sol,
      qacc=x_out,
      efc=dataclasses.replace(d_bwd.efc, aref=zero_efc),
    )
    solver.solve(m_lin, d_lin)
    wp.launch(_div_scale, dim=rhs.shape, inputs=[scale], outputs=[x_out])

  max_records = (min(nv, m.max_efc_J_rownnz) if m.is_sparse else nv) if det_atomics else 0
  rhs_squares = _normalized(rhs, unit, det_atomics)
  _cg_pass(unit, rhs_squares, x)
  for _ in range(_SOLVE_REFINEMENTS):
    derivative.eff_mul_m(m, d_bwd, work, x, efm_con=efm_con, flex_scratch=flex_scratch)
    wp.launch(
      _add_jtdj_mul(m.is_sparse, max_records, det_atomics),
      dim=(nworld, d_bwd.njmax),
      inputs=[
        nv,
        d_bwd.nefc,
        d_bwd.efc.J_rownnz,
        d_bwd.efc.J_rowadr,
        d_bwd.efc.J_colind,
        d_bwd.efc.J,
        d_bwd.efc.D,
        x,
      ],
      outputs=[work],
    )
    wp.launch(_residual, dim=rhs.shape, inputs=[unit], outputs=[work])
    res_squares = _normalized(work, work, det_atomics)
    _cg_pass(work, res_squares, delta)
    wp.launch(_add_scaled, dim=rhs.shape, inputs=[delta, res_squares], outputs=[x])
  if overflow_out is not None:
    wp.launch(_or_overflow, dim=nworld, inputs=[d_bwd.overflow], outputs=[overflow_out])
  wp.launch(_scale, dim=rhs.shape, inputs=[rhs_squares], outputs=[x])


@event_scope
def _newton_constrained_solve(
  m: Model,
  d_bwd: Data,
  x: wp.array,
  rhs: wp.array,
):
  """Solves (A + J^T D J) x = rhs via the forward's Newton Hessian assembly and Cholesky."""
  nworld, nv = rhs.shape
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  unit = wp.empty_like(rhs)
  rhs_squares = _normalized(rhs, unit, det_atomics)

  solver_ctx = solver.create_solver_context(m, d_bwd)
  solver_ctx.efm_con, solver_ctx.flex_scratch = derivative.eff_mul_m_scratch(m, d_bwd)

  zero_nv = wp.zeros((nworld, nv), dtype=float)
  zero_efc = wp.zeros((nworld, d_bwd.njmax), dtype=float)
  d_lin = dataclasses.replace(
    d_bwd,
    qfrc_smooth=unit,
    efm_c=zero_nv,
    efm_ca=zero_nv,
    qacc=zero_nv,
    qfrc_constraint=wp.zeros((nworld, nv), dtype=float),
    efc=dataclasses.replace(d_bwd.efc, aref=zero_efc),
  )
  solver.init_context(m, d_lin, solver_ctx, grad=True)
  wp.launch(_copy_scaled, dim=rhs.shape, inputs=[solver_ctx.search, rhs_squares], outputs=[x])


@event_scope
def _solve_adjoint_system(
  m: Model,
  d_bwd: Data,
  lam: wp.array,
  rhs: wp.array,
  has_eq: bool,
  use_newton: bool,
  overflow_out: wp.array | None = None,
):
  """Solves the symmetric adjoint system H lam = rhs via the forward's solver."""
  if use_newton:
    _newton_constrained_solve(m, d_bwd, lam, rhs)
  elif has_eq:
    _cg_constrained_solve(m, d_bwd, lam, rhs, overflow_out)
  else:
    metric_solve(m, d_bwd, lam, rhs, overflow_out)


# Single-source-of-truth field tuples for Model structure and SNH survey caches. Every Model
# scalar and array read in _model_reasons / _stiffness_reasons / _stiffness_survey must appear
# here so cache keys, per-array weakref validation, and gate reads cannot drift apart.
_MODEL_REASON_SCALARS = (
  "nflex",
  "nv",
  "neq",
  "nu",
  "na",
  "ntendon",
  "nplastic",
  "nmocap",
  "has_non_simple_flex",
  "has_flex_passive",
  "has_flex_edge_stiff_damp",
  "has_fluid",
  "flg_adhesion",
  "flg_surfacevel",
)
_MODEL_REASON_ARRAYS = (
  "flex_dim",
  "flex_interp",
  "flex_simple",
  "flex_passive",
  "flex_edgestiffness",
  "flex_edgedamping",
  "flex_edgeequality",
  "flex_plasticadr",
  "flex_plasticmode",
  "jnt_type",
  "body_dofnum",
  "body_simple",
  "dof_parentid",
  "body_dofadr",
  "eq_type",
  "eq_flex_adr",
  "eq_obj1id",
  "flex_efcnum",
  "flex_efcnnz",
  "body_gravcomp",
  "jnt_limited",
  "jnt_stiffness",
  "jnt_stiffnesspoly",
  "dof_damping",
  "dof_dampingpoly",
  "dof_frictionloss",
)
_STIFFNESS_SURVEY_SCALARS = ("nflex", "nflexelem", "nflexstiffness", "has_flex_snh")
_STIFFNESS_SURVEY_ARRAYS = (
  "flex_stiffness",
  "flex_dim",
  "flex_elem",
  "flex_vert0",
  "flex_size",
  "flex_stiffnessadr",
  "flex_elemadr",
  "flex_elemflexid",
  "flex_vertadr",
  "flex_elemdataadr",
)


class _HostArrays:
  """Lazy `.numpy()` reader restricted to a declared tuple of keyed Model array fields."""

  def __init__(self, m: Model, allowed: tuple[str, ...]):
    self._m = m
    self._allowed = frozenset(allowed)
    self._cache: dict[str, np.ndarray] = {}

  def __getitem__(self, name: str) -> np.ndarray:
    if name not in self._allowed:
      raise KeyError(f"Model array {name!r} is not in the cache key tuple")
    arr = self._cache.get(name)
    if arr is None:
      src = vars(self._m)[name]
      if src.device.is_capturing:
        raise NotImplementedError("record a supported step before CUDA graph capture")
      arr = src.numpy()
      self._cache[name] = arr
    return arr


def _cache_key(m: Model, scalars: tuple[str, ...], arrays: tuple[str, ...]) -> tuple[int, ...]:
  return (
    *(int(vars(m)[name]) for name in scalars),
    *(id(vars(m)[name]) for name in arrays),
  )


def _cache_refs(
  m: Model, arrays: tuple[str, ...], cache: dict[tuple[int, ...], _T], key: tuple[int, ...]
) -> tuple[weakref.ReferenceType[wp.array], ...]:
  def _evict(_ref, _cache=cache, _key=key):
    _cache.pop(_key, None)

  return tuple(weakref.ref(vars(m)[name], _evict) for name in arrays)


def _refs_match(m: Model, arrays: tuple[str, ...], refs: tuple[weakref.ReferenceType[wp.array], ...]) -> bool:
  return all(ref() is vars(m)[name] for ref, name in zip(refs, arrays))


def _shares_memory(a: wp.array, b: wp.array) -> bool:
  """Returns whether two nonempty arrays share any memory."""
  return bool(a.size and b.size) and a.ptr < b.ptr + b.capacity and b.ptr < a.ptr + a.capacity


@wp.kernel(module="unique", enable_backward=False)
def _snh_stiffness_survey(
  # Model:
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_stiffnessadr: wp.array[int],
  flex_elem: wp.array[int],
  flex_vert0: wp.array[wp.vec3],
  flex_size: wp.array[wp.vec3],
  flex_stiffness: wp.array[float],
  flex_elemflexid: wp.array[int],
  # Out:
  survey_out: wp.array2d[float],
):
  """Per-element SNH law mix and isotropic-metric residual survey in float64."""
  elemid = wp.tid()
  f = flex_elemflexid[elemid]
  if flex_dim[f] != 3:
    return
  adr = flex_stiffnessadr[f]
  # a zero first entry disables the whole flex
  if adr < 0 or flex_stiffness[adr] == 0.0:
    return
  local_elemid = elemid - flex_elemadr[f]
  k_adr = adr + local_elemid * wp.static(FLEX_STIFFNESS_3D)
  if flex_stiffness[k_adr + 21] == 0.0:
    if survey_out[0, f] == 0.0:
      wp.atomic_max(survey_out, 0, f, 1.0)  # kernel_analyzer: ignore[atomic]
    return
  if survey_out[1, f] == 0.0:
    wp.atomic_max(survey_out, 1, f, 1.0)  # kernel_analyzer: ignore[atomic]

  # Unpack symmetric 6x6 edge metric in float64 and flag any non-finite SNH entry.
  nonfinite = not (
    wp.isfinite(flex_stiffness[k_adr + 21])
    and wp.isfinite(flex_stiffness[k_adr + 22])
    and wp.isfinite(flex_stiffness[k_adr + 23])
  )
  metric = mat66d()
  idx = int(0)
  for i in range(6):
    for j in range(i, 6):
      val = wp.float64(flex_stiffness[k_adr + idx])
      if not wp.isfinite(val):
        nonfinite = True
      metric[i, j] = val
      metric[j, i] = val
      idx += 1

  # Rest-state vertex coordinates in float64: vert0 * (2 * size).
  vadr = flex_vertadr[f]
  eadr = flex_elemdataadr[f] + local_elemid * 4
  s_f = flex_size[f]
  sx = wp.float64(2.0) * wp.float64(s_f[0])
  sy = wp.float64(2.0) * wp.float64(s_f[1])
  sz = wp.float64(2.0) * wp.float64(s_f[2])

  v0_raw = flex_vert0[vadr + flex_elem[eadr]]
  v1_raw = flex_vert0[vadr + flex_elem[eadr + 1]]
  v2_raw = flex_vert0[vadr + flex_elem[eadr + 2]]
  v3_raw = flex_vert0[vadr + flex_elem[eadr + 3]]

  r0 = wp.vec3d(wp.float64(v0_raw[0]) * sx, wp.float64(v0_raw[1]) * sy, wp.float64(v0_raw[2]) * sz)
  r1 = wp.vec3d(wp.float64(v1_raw[0]) * sx, wp.float64(v1_raw[1]) * sy, wp.float64(v1_raw[2]) * sz)
  r2 = wp.vec3d(wp.float64(v2_raw[0]) * sx, wp.float64(v2_raw[1]) * sy, wp.float64(v2_raw[2]) * sz)
  r3 = wp.vec3d(wp.float64(v3_raw[0]) * sx, wp.float64(v3_raw[1]) * sy, wp.float64(v3_raw[2]) * sz)
  for c in range(3):
    if not (wp.isfinite(r0[c]) and wp.isfinite(r1[c]) and wp.isfinite(r2[c]) and wp.isfinite(r3[c])):
      nonfinite = True
  if nonfinite:
    if survey_out[3, f] == 0.0:
      wp.atomic_max(survey_out, 3, f, 1.0)  # kernel_analyzer: ignore[atomic]
    return

  # Elongation of edge e is 2 r_e^T E r_e, in passive.stretch_edge_endpoints(3) order.
  strain = mat66d()
  for e in range(6):
    if e == 0:
      re = r0 - r1
    elif e == 1:
      re = r1 - r2
    elif e == 2:
      re = r2 - r0
    elif e == 3:
      re = r2 - r3
    elif e == 4:
      re = r0 - r3
    else:
      re = r1 - r3
    strain[e, 0] = wp.float64(2.0) * re[0] * re[0]
    strain[e, 1] = wp.float64(2.0) * re[1] * re[1]
    strain[e, 2] = wp.float64(2.0) * re[2] * re[2]
    strain[e, 3] = wp.float64(4.0) * re[0] * re[1]
    strain[e, 4] = wp.float64(4.0) * re[0] * re[2]
    strain[e, 5] = wp.float64(4.0) * re[1] * re[2]

  # Quadratic part of an isotropic energy in Green strain (E00, E11, E22, E01, E02, E12):
  # orthogonal projection of T = S^T M S onto span(B0, B1), where B0 = diag(1, 1, 1, 2, 2, 2)
  # and B1 = outer([1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]). Gram [[15, 3], [3, 9]]^-1 =
  # (1 / 42) [[3, -1], [-1, 5]].
  tensor = wp.transpose(strain) * (metric * strain)
  c0 = tensor[0, 0] + tensor[1, 1] + tensor[2, 2] + wp.float64(2.0) * (tensor[3, 3] + tensor[4, 4] + tensor[5, 5])
  c1 = (
    tensor[0, 0]
    + tensor[0, 1]
    + tensor[0, 2]
    + tensor[1, 0]
    + tensor[1, 1]
    + tensor[1, 2]
    + tensor[2, 0]
    + tensor[2, 1]
    + tensor[2, 2]
  )
  a0 = (wp.float64(3.0) * c0 - c1) / wp.float64(42.0)
  a1 = (-c0 + wp.float64(5.0) * c1) / wp.float64(42.0)

  norm_t_sq = wp.float64(0.0)
  norm_diff_sq = wp.float64(0.0)
  for a in range(6):
    for b in range(6):
      t_ab = tensor[a, b]
      fit_ab = wp.float64(0.0)
      if a == b:
        fit_ab += a0 if a < 3 else wp.float64(2.0) * a0
      if a < 3 and b < 3:
        fit_ab += a1
      diff_ab = t_ab - fit_ab
      norm_t_sq += t_ab * t_ab
      norm_diff_sq += diff_ab * diff_ab

  denom = wp.max(wp.sqrt(norm_t_sq), wp.float64(MJ_MINVAL))
  dev = wp.sqrt(norm_diff_sq) / denom
  dev_f32 = float(dev)
  if not wp.isfinite(dev_f32):
    if survey_out[3, f] == 0.0:
      wp.atomic_max(survey_out, 3, f, 1.0)  # kernel_analyzer: ignore[atomic]
    return
  if dev_f32 > survey_out[2, f]:
    wp.atomic_max(survey_out, 2, f, dev_f32)  # kernel_analyzer: ignore[atomic]


def _stiffness_survey(m: Model) -> tuple[bool, float, bool, bool]:
  """Inspects flex_stiffness on device with a tiny per-flex readback.

  Returns whether a flex mixes default and SNH elements, the largest finite relative deviation of
  an SNH element metric from an isotropic material, whether any SNH element has a non-finite
  metric or rest geometry, and whether any element uses SNH.
  """
  if m.nflex == 0 or m.nflexelem == 0 or m.nflexstiffness == 0:
    return False, 0.0, False, False
  device = m.flex_stiffness.device
  if device.is_capturing:
    raise NotImplementedError("record a supported step before CUDA graph capture")
  survey = wp.zeros((4, m.nflex), dtype=float, device=device)
  wp.launch(
    _snh_stiffness_survey,
    dim=m.nflexelem,
    inputs=[
      m.flex_dim,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemdataadr,
      m.flex_stiffnessadr,
      m.flex_elem,
      m.flex_vert0,
      m.flex_size,
      m.flex_stiffness,
      m.flex_elemflexid,
    ],
    outputs=[survey],
    device=device,
    record_tape=False,
  )
  out = survey.numpy()
  has_default = out[0] != 0.0
  has_snh = out[1] != 0.0
  # The forward keys the Rayleigh damper on the first element but the forces on each element.
  mixed = bool(np.any(has_default & has_snh))
  pure_snh = has_snh & ~has_default
  worst = float(out[2, pure_snh].max()) if np.any(pure_snh) else 0.0
  nonfinite = bool(np.any(out[3, pure_snh] != 0.0)) if np.any(pure_snh) else False
  return mixed, worst, nonfinite, bool(np.any(has_snh))


# Cache by all inspected scalar values and array identities (weakref-validated per array) so
# dataclasses.replace(m) shallow copies hit without retaining models or their device buffers.
_SUPPORTED_CACHE: dict[
  tuple[int, ...],
  tuple[tuple[weakref.ReferenceType[wp.array], ...], tuple[str, ...], int, int],
] = {}
_STIFFNESS_CACHE: dict[
  tuple[int, ...],
  tuple[tuple[weakref.ReferenceType[wp.array], ...], tuple[str, ...]],
] = {}


def _model_reasons(m: Model) -> tuple[list[str], int, int]:
  """Inspects the fixed model structure and caches the verdict per array identity.

  Array contents are read once per model: joint stiffness, damping, friction loss, limits or
  gravity compensation edited in place after the first recorded step are not re-checked. Option
  flags and flex_stiffness are checked in step_supported.

  Returns (reasons, need_rows, need_nnz) where need_rows/need_nnz are the total flex edge equality
  constraint rows and Jacobian nonzeros across all EqType.FLEX records.
  """
  key = _cache_key(m, _MODEL_REASON_SCALARS, _MODEL_REASON_ARRAYS)
  cached = _SUPPORTED_CACHE.get(key)
  if cached is not None and _refs_match(m, _MODEL_REASON_ARRAYS, cached[0]):
    if not cached[1] or m.flex_dim.device.is_capturing:
      return list(cached[1]), cached[2], cached[3]

  h = _HostArrays(m, _MODEL_REASON_ARRAYS)
  reasons = []
  if m.nflex == 0:
    reasons.append("the discrete integrator without a flex")
  else:
    flex_dim = h["flex_dim"]
    if np.any(flex_dim != 3):
      reasons.append("flexes with dimension other than 3 (1D rods and 2D shells)")
    if np.any(h["flex_interp"] != 0):
      reasons.append("interpolated flexes")
    if m.has_non_simple_flex or not np.all(h["flex_simple"]):
      reasons.append("flex vertices on bodies that are not 3-slider bodies")
    if m.has_flex_passive or np.any(h["flex_passive"] != 0):
      reasons.append("passive flex contact")
    if m.has_flex_edge_stiff_damp or np.any(h["flex_edgestiffness"] != 0.0) or np.any(h["flex_edgedamping"] != 0.0):
      reasons.append("flex edge stiffness or damping")
    flex_eq = h["flex_edgeequality"]
    if np.any((flex_eq != 0) & (flex_eq != 1)):
      reasons.append("flex strain equality constraints")
    if np.any(h["flex_plasticadr"] >= 0) or np.any(h["flex_plasticmode"] != 0):
      reasons.append("flex plasticity")
  if m.nv == 0:
    reasons.append("models with zero degrees of freedom (nv == 0)")
  if np.any(h["jnt_type"] != JointType.SLIDE):
    reasons.append("joints other than the slide joints of flex vertex bodies (rigid bodies)")
  else:
    dofnum = h["body_dofnum"]
    moving = dofnum > 0
    simple = h["body_simple"][moving] == 2
    rooted = h["dof_parentid"][h["body_dofadr"][moving]] == -1
    if not (np.all(dofnum[moving] == 3) and np.all(simple) and np.all(rooted)):
      reasons.append("moving bodies that are not free-standing 3-slider bodies")
  if m.nu or m.na:
    reasons.append("actuators")
  if m.ntendon:
    reasons.append("tendons")
  if m.nmocap:
    reasons.append("mocap bodies")
  need_rows = 0
  need_nnz = 0
  if m.neq:
    eq_type = h["eq_type"]
    if np.any(eq_type != int(EqType.FLEX)):
      reasons.append("equality constraints other than flex edge equality")
    elif m.eq_flex_adr.size:
      flex_ids = h["eq_obj1id"][h["eq_flex_adr"]]
      need_rows = int(np.sum(h["flex_efcnum"][flex_ids]))
      need_nnz = int(np.sum(h["flex_efcnnz"][flex_ids]))
  if m.nplastic:
    reasons.append("plasticity")
  if m.has_fluid:
    reasons.append("fluid forces")
  if m.flg_adhesion:
    reasons.append("adhesion")
  if m.flg_surfacevel:
    reasons.append("surface velocity")
  if np.any(h["body_gravcomp"] != 0.0):
    reasons.append("gravity compensation")
  if np.any(h["jnt_limited"] != 0):
    reasons.append("joint limits")
  if np.any(h["jnt_stiffness"] != 0.0) or np.any(h["jnt_stiffnesspoly"] != 0.0):
    reasons.append("joint stiffness")
  if np.any(h["dof_damping"] != 0.0) or np.any(h["dof_dampingpoly"] != 0.0):
    reasons.append("joint damping")
  if np.any(h["dof_frictionloss"] != 0.0):
    reasons.append("joint friction loss")

  refs = _cache_refs(m, _MODEL_REASON_ARRAYS, _SUPPORTED_CACHE, key)
  _SUPPORTED_CACHE[key] = (refs, tuple(reasons), need_rows, need_nnz)
  return reasons, need_rows, need_nnz


def _stiffness_reasons(m: Model) -> list[str]:
  """Checks element laws and SNH metrics, re-checking while SNH flex_stiffness is optimised.

  Models without SNH elements and models whose stiffness is not differentiated are checked once.
  The survey reads back four scalars per flex, so during CUDA graph capture it returns the cached
  verdict instead of launching a readback.
  """
  key = _cache_key(m, _STIFFNESS_SURVEY_SCALARS, _STIFFNESS_SURVEY_ARRAYS)
  cached = _STIFFNESS_CACHE.get(key)
  if cached is not None and _refs_match(m, _STIFFNESS_SURVEY_ARRAYS, cached[0]):
    cached_reasons = cached[1]
    if m.flex_stiffness.device.is_capturing:
      return list(cached_reasons)
    if not cached_reasons and (not m.has_flex_snh or not m.flex_stiffness.requires_grad):
      return []
  reasons = []
  mixed, anisotropy, nonfinite, has_snh = _stiffness_survey(m)
  if mixed:
    reasons.append("flexes mixing default and stable Neo-Hookean elements")
  if has_snh and not m.has_flex_snh:
    reasons.append("stable Neo-Hookean elements enabled after put_model (m.has_flex_snh is False)")
  if nonfinite:
    reasons.append("stable Neo-Hookean elements with a non-finite stiffness metric or rest geometry")
  if anisotropy > _SNH_ISOTROPY_TOL:
    reasons.append(
      f"stable Neo-Hookean elements with an anisotropic stiffness metric (relative deviation {anisotropy:.1e}); "
      "the analytic eigensystem of the projected Hessian is only differentiable for isotropic materials"
    )
  refs = _cache_refs(m, _STIFFNESS_SURVEY_ARRAYS, _STIFFNESS_CACHE, key)
  _STIFFNESS_CACHE[key] = (refs, tuple(reasons))
  return reasons


def step_supported(m: Model, d: Data, d_out: Data | None = None) -> tuple[str, ...]:
  """Returns the reasons step() has no analytic backward for this model and data."""
  if d_out is None or d_out is d:
    return ("step() differentiation requires distinct input and output Data; in-place is unsupported",)
  for name in ("nworld", "naconmax", "njmax"):
    if vars(d)[name] != vars(d_out)[name]:
      return (f"step() differentiation expected matching {name}",)
  # The backward linearizes at the input state, so the step must not overwrite it.
  for name in _STEP_IN_FIELDS:
    if any(_shares_memory(vars(d)[name], vars(d_out)[out]) for out in _STEP_OUT_FIELDS):
      return (f"step() differentiation requires distinct state buffers; input {name} aliases output state",)
  if m.opt.integrator != IntegratorType.DISCRETE:
    return ("step() differentiation supports only IntegratorType.DISCRETE with stock 3D flexes",)

  reasons, need_rows, need_nnz = _model_reasons(m)
  if not reasons:
    reasons = _stiffness_reasons(m)
  # Option flags can change between steps and are never cached.
  if m.opt.disableflags & (DisableBit.SPRING | DisableBit.DAMPER):
    reasons.append("disabled springs or dampers")
  if m.opt.enableflags & EnableBit.SLEEP:
    reasons.append("sleep")
  if m.opt.solver not in (SolverType.CG, SolverType.NEWTON):
    reasons.append("solvers other than CG and NEWTON")
  if m.opt.iterations <= 0:
    reasons.append("opt.iterations <= 0")
  if m.opt.noslip_iterations > 0:
    reasons.append("noslip solver iterations")
  for name in ("passive", "control", "act_dyn", "act_gain", "act_bias"):
    if vars(m.callback)[name]:
      reasons.append(f"the {name} callback")
  datas = (("d", d), ("d_out", d_out))
  if m.is_sparse and d.njmax_nnz != d_out.njmax_nnz:
    reasons.append(f"mismatched sparse njmax_nnz between d ({d.njmax_nnz}) and d_out ({d_out.njmax_nnz})")
  if not (m.opt.disableflags & (DisableBit.CONSTRAINT | DisableBit.CONTACT)):
    if any(data.njmax and data.naconmax for _, data in datas):
      reasons.append("contacts (disable contacts or constraints; contact rows are not differentiated)")
  eq_enabled = not (m.opt.disableflags & (DisableBit.CONSTRAINT | DisableBit.EQUALITY))
  if eq_enabled and need_rows:
    for label, data in datas:
      if data.njmax < need_rows:
        reasons.append(f"{label}.njmax ({data.njmax}) smaller than flex equality rows ({need_rows})")
      if m.is_sparse and data.njmax_nnz < need_nnz:
        reasons.append(f"{label}.njmax_nnz ({data.njmax_nnz}) smaller than flex equality Jacobian nonzeros ({need_nnz})")
  parameters = [
    prefix + name
    for prefix, obj in (("", m), ("opt.", m.opt))
    for name, value in vars(obj).items()
    if isinstance(value, wp.array) and value.size and value.requires_grad and prefix + name not in _PARAMETER_FIELDS
  ]
  if parameters:
    reasons.append("parameter gradients for " + ", ".join(sorted(parameters)))
  inputs = [
    name
    for name in _UNSUPPORTED_USER_FIELDS
    if any(vars(data)[name].size and vars(data)[name].requires_grad for _, data in datas)
  ]
  if inputs:
    reasons.append("input gradients for " + ", ".join(inputs))
  if reasons:
    return ("step() with the discrete integrator supports only stock 3D flexes and not: " + "; ".join(reasons),)
  return ()


@event_scope
def _accumulate_flex_vjp(
  m: Model,
  d_bwd: Data,
  xvel: wp.array,
  xlam: wp.array,
  xacc: wp.array,
  has_eq: bool,
  adj_xpos: wp.array,
  adj_xvel: wp.array,
):
  """Accumulates local flex-law and edge-equality residual VJPs into state/param grads."""
  nworld = d_bwd.nworld
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)
  xpos = d_bwd.flexvert_xpos
  damping_grad = m.flex_damping.grad if m.flex_damping.requires_grad else None
  length0_grad = m.flexedge_length0.grad if m.flexedge_length0.requires_grad else None

  # Per-law guards avoid compiling unused module="unique" VJP kernels and cloning scratch buffers.
  if m.nflexelem > 0 and m.nflexstiffness > 0:
    stiffness_grad = m.flex_stiffness.grad if m.flex_stiffness.requires_grad else None
    phi_elem = wp.zeros((nworld, m.nflexelem), dtype=float, requires_grad=True)
    phi_elem.grad.fill_(1.0)
    elem = [
      (m.opt.timestep, None),
      (m.flex_dim, None),
      (m.flex_vertadr, None),
      (m.flex_edgeadr, None),
      (m.flex_elemadr, None),
      (m.flex_elemdataadr, None),
      (m.flex_stiffnessadr, None),
      (m.flex_elemedgeadr, None),
      (m.flex_elem, None),
      (m.flex_elemedge, None),
    ]
    stretch_pairs = [
      *elem,
      (m.flexedge_length0, length0_grad),
      (m.flex_stiffness, stiffness_grad),
      (m.flex_damping, damping_grad),
      (m.flex_rigid, None),
      (m.flexedge_J_rownnz, None),
      (m.flex_elemflexid, None),
      (xpos, adj_xpos),
      (xvel, adj_xvel),
      (xlam, None),
      (xacc, None),
    ]
    diff.launch_vjp(_flex3d_phi_default(det_atomics), (nworld, m.nflexelem), stretch_pairs, [phi_elem], [phi_elem.grad])
    if m.has_flex_snh:
      # Frozen copies must not alias an array whose gradient Warp would pick up implicitly.
      stiffness_frozen = wp.clone(m.flex_stiffness, requires_grad=False) if m.flex_stiffness.requires_grad else m.flex_stiffness
      damping_frozen = wp.clone(m.flex_damping, requires_grad=False) if m.flex_damping.requires_grad else m.flex_damping
      length0_frozen = (
        wp.clone(m.flexedge_length0, requires_grad=False) if m.flexedge_length0.requires_grad else m.flexedge_length0
      )
      snh_pairs = [
        *elem,
        (m.flex_vert0, None),
        (m.flexedge_length0, length0_grad),
        (m.flex_size, None),
        (m.flex_stiffness, stiffness_grad),
        (m.flex_damping, damping_grad),
        (m.flex_rigid, None),
        (m.flex_elemflexid, None),
        (xpos, adj_xpos),
        (xvel, adj_xvel),
        (xpos, None),
        (xvel, None),
        (stiffness_frozen, None),
        (damping_frozen, None),
        (length0_frozen, None),
        (xlam, None),
        (xacc, None),
        (_SNH_MODES, _SNH_MODES),
      ]
      diff.launch_vjp(_flex3d_phi_snh(det_atomics), (nworld, m.nflexelem), snh_pairs, [phi_elem], [phi_elem.grad])

  if has_eq:
    solref_grad = m.eq_solref.grad if m.eq_solref.requires_grad else None
    solimp_grad = m.eq_solimp.grad if m.eq_solimp.requires_grad else None
    phi_eq = wp.zeros((nworld, m.eq_flex_adr.size, m.nflexedge), dtype=float, requires_grad=True)
    phi_eq.grad.fill_(1.0)
    eq_pairs = [
      (m.opt.timestep, None),
      (m.flex_vertadr, None),
      (m.flex_edgeadr, None),
      (m.flex_edgenum, None),
      (m.flex_edge, None),
      (m.flexedge_length0, length0_grad),
      (m.flexedge_invweight0, None),
      (m.eq_obj1id, None),
      (m.eq_solref, solref_grad),
      (m.eq_solimp, solimp_grad),
      (m.eq_flex_adr, None),
      (m.flexedge_efcadr, None),
      (d_bwd.eq_active, None),
      (xpos, adj_xpos),
      (xvel, adj_xvel),
      (xlam, None),
      (xacc, None),
    ]
    diff.launch_vjp(_flex_eq_phi(det_atomics), (nworld, m.eq_flex_adr.size, m.nflexedge), eq_pairs, [phi_eq], [phi_eq.grad])


@event_scope
def step_backward(m: Model, d: Data, d_out: Data | None = None):
  """Maps d_out state gradients to d and to supported flex/equality parameters.

  Reads d.qpos, d.qvel, d.eq_active and d_out.qacc. Position-dependent quantities are recomputed
  from d on inline scratch storage, so d_out may have been refreshed by fwd_kinematics since the
  step.

  Body and site poses of d_out are step outputs computed before integration, as for the other
  integrators: their gradients map to d.qpos through the kinematics at d.qpos and are consumed.
  Overwritten output gradients on d_out are zeroed after accumulation into d.

  Iterative adjoint solves (matrix-free PCG and CG) OR ``OverflowType.ITERATIONS`` into
  ``d_out.overflow`` (and may emit the overflow warning) if they do not converge within
  ``m.opt.iterations``; the direct Newton Cholesky backward solve (``SolverType.NEWTON`` with
  ``d.njmax > 0``) does not write ``d_out.overflow`` (though with default
  ``opt.run_qacc_smooth=True`` and ``nefc == 0``, the forward itself runs PCG and may set
  ``d_out.overflow``).

  Worlds with non-finite input state or output cotangents short-circuit the replay and adjoint
  solve so healthy worlds keep finite state gradients (and finite per-world batched parameter
  gradients), while shared Model parameter gradients follow sum-over-worlds semantics and become
  NaN if any world is non-finite.
  """
  if d_out is None or d_out is d:
    raise NotImplementedError("step() differentiation requires distinct input and output Data; in-place is unsupported")
  nworld, nv = d.nworld, m.nv
  has_eq = bool(m.eq_flex_adr.size and d.njmax > 0 and not (m.opt.disableflags & (DisableBit.CONSTRAINT | DisableBit.EQUALITY)))
  use_newton = m.opt.solver == SolverType.NEWTON and d.njmax > 0

  adj_qvel = wp.empty((nworld, nv), dtype=float)
  adj_qacc = wp.empty((nworld, nv), dtype=float)
  nonfinite = wp.zeros(nworld, dtype=int)
  wp.launch(
    _integrator_vjp,
    dim=(nworld, nv),
    inputs=[m.opt.timestep, d.qpos, d.qvel, d_out.qacc, d_out.qpos.grad, d_out.qvel.grad, d_out.qacc.grad],
    outputs=[adj_qvel, adj_qacc, nonfinite],
  )

  # Short-circuit worlds with non-finite state/cotangent before replay and iterative solves so
  # NaN states never burn full opt.iterations in PCG/CG, then restore NaN into lam for those worlds.
  d_bwd = diff.clone_nograd(d)
  adj_qacc_clean = wp.empty((nworld, nv), dtype=float)
  wp.launch(
    _sanitize_bwd_state,
    dim=(nworld, nv),
    inputs=[m.qpos0, d.qpos, d.qvel, d_out.qacc, adj_qacc, nonfinite],
    outputs=[d_bwd.qpos, d_bwd.qvel, d_bwd.qacc, adj_qacc_clean],
  )

  # Replay the position stage of forward.forward; the metric factorisation may have been skipped.
  smooth.kinematics(m, d_bwd)
  smooth.com_pos(m, d_bwd)
  smooth.flex(m, d_bwd)
  smooth.crb(m, d_bwd)
  if has_eq:
    d_bwd.nacon.zero_()
    constraint.make_constraint(m, d_bwd)
  elif use_newton:
    d_bwd.ne.zero_()
    d_bwd.nf.zero_()
    d_bwd.nl.zero_()
    d_bwd.nefc.zero_()
    d_bwd.nacon.zero_()
    d_bwd.efc.jtdaj_nblock.zero_()
  derivative.eff_build(m, d_bwd)
  derivative.eff_shift(m, d_bwd)
  derivative.eff_actuation(dataclasses.replace(m, opt=dataclasses.replace(m.opt, run_qacc_smooth=True)), d_bwd)
  if has_eq:
    constraint.regularize_constraint(m, d_bwd)

  lam = wp.empty((nworld, nv), dtype=float)
  _solve_adjoint_system(m, d_bwd, lam, adj_qacc_clean, has_eq, use_newton, d_out.overflow)
  wp.launch(_fill_nonfinite_nan, dim=(nworld, nv), inputs=[nonfinite], outputs=[lam])

  # The stretch forces, Hessians, dampers and edge equalities act on world-frame vertices.
  xlam = wp.empty((nworld, m.nflexvert), dtype=wp.vec3)
  xacc = wp.empty((nworld, m.nflexvert), dtype=wp.vec3)
  xvel = wp.empty((nworld, m.nflexvert), dtype=wp.vec3)
  passive.flex_gather(m, d_bwd, xlam, lam)
  passive.flex_gather(m, d_bwd, xacc, d_bwd.qacc)
  passive.flex_gather(m, d_bwd, xvel, d_bwd.qvel)

  adj_xpos = wp.zeros((nworld, m.nflexvert), dtype=wp.vec3)
  adj_xvel = wp.zeros((nworld, m.nflexvert), dtype=wp.vec3)
  _accumulate_flex_vjp(m, d_bwd, xvel, xlam, xacc, has_eq, adj_xpos, adj_xvel)

  res_qpos = wp.zeros((nworld, nv), dtype=float)
  res_qvel = wp.zeros((nworld, nv), dtype=float)
  passive.flex_scatter(m, d_bwd, res_qpos, adj_xpos)
  passive.flex_scatter(m, d_bwd, res_qvel, adj_xvel)

  adj_site_xpos = d_out.site_xpos.grad if m.nsite > 0 else None
  if d_out.xpos.grad is not None or adj_site_xpos is not None:
    wp.launch(
      _slide_kinematics_vjp,
      dim=(nworld, m.njnt),
      inputs=[
        m.nsite,
        m.jnt_qposadr,
        m.jnt_bodyid,
        m.site_bodyid,
        m.body_subtreenum,
        d_bwd.xaxis,
        d_out.xpos.grad,
        adj_site_xpos,
      ],
      outputs=[res_qpos],
    )
  for name in _KINEMATICS_OUT_FIELDS:
    arr = vars(d_out)[name]
    if arr.size and arr.grad is not None:
      arr.grad.zero_()

  wp.launch(
    _write_input_adjoints,
    dim=(nworld, nv),
    inputs=[d_out.qpos.grad, adj_qvel, res_qpos, res_qvel, nonfinite],
    outputs=[d.qpos.grad, d.qvel.grad],
  )

  # Applied forces enter qfrc_smooth linearly; rooted 3-slider Jacobians do not depend on qpos.
  if d.qfrc_applied.grad is not None:
    wp.launch(diff.accum_cols, dim=(nworld, nv), inputs=[lam], outputs=[d.qfrc_applied.grad])
  if d.xfrc_applied.grad is not None:
    wp.launch(
      _slide_xfrc_vjp,
      dim=(nworld, m.nbody),
      inputs=[m.body_weldid, m.body_dofnum, m.body_dofadr, d_bwd.cdof, lam],
      outputs=[d.xfrc_applied.grad],
    )
  if (
    d.qfrc_applied.grad is not None
    and d_out.qfrc_applied.grad is not None
    and d.qfrc_applied.grad.ptr != d_out.qfrc_applied.grad.ptr
  ):
    wp.launch(diff.accum_cols, dim=d.qfrc_applied.shape, inputs=[d_out.qfrc_applied.grad], outputs=[d.qfrc_applied.grad])
  if (
    d.xfrc_applied.grad is not None
    and d_out.xfrc_applied.grad is not None
    and d.xfrc_applied.grad.ptr != d_out.xfrc_applied.grad.ptr
  ):
    wp.launch(_accum_spatial, dim=d.xfrc_applied.shape, inputs=[d_out.xfrc_applied.grad], outputs=[d.xfrc_applied.grad])

  # Overwritten outputs and distinct user-input copies must be consumed after accumulation.
  for name in _STEP_OUT_FIELDS + _USER_FIELDS:
    output = vars(d_out)[name]
    source = vars(d)[name]
    if output.grad is not None and (source.grad is None or output.grad.ptr != source.grad.ptr):
      output.grad.zero_()


def register() -> None:
  """Registers the analytic backward rule for `step` on discrete 3D flexes."""
  diff.register_adjoint(
    "step",
    step_backward,
    in_fields=_STEP_IN_FIELDS + _USER_FIELDS,
    out_fields=_STEP_OUT_FIELDS + _USER_FIELDS + _KINEMATICS_OUT_FIELDS,
    model_fields=_PARAMETER_FIELDS,
    supported=step_supported,
  )
