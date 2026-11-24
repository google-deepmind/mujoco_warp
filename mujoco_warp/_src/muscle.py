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

"""Compliant-tendon muscle actuators, as in engine_muscle.c.

Millard2012 is OpenSim's Millard2012EquilibriumMuscle; Geyer2010 is the muscle of Geyer and Herr
(2010), with the curves of Geyer et al. (2003) as Song and Geyer implement them. The actuator state
is act = [fiber, activation] with fiber = l_ce / l_opt - 1. The fiber advances by one backward-Euler
step of the force equilibrium, and the force is gain * act + bias at the solved fiber, which equals
the tendon force.

The Millard2012 curves are OpenSim's quintic Bezier corners, evaluated exactly: the knots of a curve
are computed from its parameters on demand, so nothing is tabulated.
"""

import warp as wp

from mujoco_warp._src import util_misc
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec10
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})

# curves, as mjtMuscleCurve
_ACTIVE = 0
_PASSIVE = 1
_TENDON = 2
_VELOCITY = 3

# sin(acos(0.1)): maximum pennation of OpenSim's MuscleFixedWidthPennationModel
_SINPHIMAX = 0.99498743710662

# OpenSim's MIN_NONZERO_DAMPING_COEFFICIENT: smaller fiber damping is no damping
_MINDAMPING = 0.001

# margin, relative to the tendon slack length, before a rigid tendon is called buckled; OpenSim's
# SimTK::SignificantReal is below single-precision roundoff
_BUCKLE = 1.0e-6

# fiber Newton: residual tolerance in F_max, step tolerance in l_opt, iteration cap
_TOL = 1.0e-6
_STEPTOL = 1.0e-7
_MAXITER = 20


@wp.func
def _classic(dynprm: vec10, biasprm: vec10) -> bool:
  """OpenSim's classic Hill model: compliant tendon, no damping."""
  return dynprm[3] < _MINDAMPING and biasprm[9] == 0.0


@wp.func
def _scale_curviness(c: float) -> float:
  return 0.1 + 0.8 * c


@wp.func
def _nknot(curve: int) -> int:
  if curve == _ACTIVE:
    return 6
  if curve == _VELOCITY:
    return 5
  return 3


@wp.func
def _knot(curve: int, k: int, dynprm: vec10, gainprm: vec10, biasprm: vec10, classic: bool) -> wp.vec3:
  """Knot k of a curve: (x, y, dy/dx), OpenSim SmoothSegmentedFunctionFactory."""
  if curve == _ACTIVE:
    x0 = gainprm[5]
    x1 = gainprm[6]
    x3 = gainprm[7]
    slope = gainprm[8]
    ylow = float(0.0)
    if classic:
      ylow = 0.1
    xs = 0.95
    y1 = 1.0 - slope * (xs - x1)
    if k == 0:
      return wp.vec3(x0, ylow, 0.0)
    if k == 1:
      return wp.vec3(x0 + 0.5 * (x1 - x0), 0.5 * y1, 1.25 * y1 / (x1 - x0))
    if k == 2:
      return wp.vec3(x1 + 0.5 * (xs - x1), y1 + 0.5 * (1.0 - y1), slope)
    if k == 3:
      return wp.vec3(1.0, 1.0, 0.0)
    if k == 4:
      return wp.vec3(1.05 + 0.5 * (x3 - 1.05), 0.5, -1.0 / ((x3 - 0.05) - 1.05))
    return wp.vec3(x3, ylow, 0.0)

  if curve == _PASSIVE:
    xzero = 1.0 + biasprm[0]
    xiso = 1.0 + biasprm[1]
    klow = biasprm[2]
    kiso = biasprm[3]
    if k == 0:
      return wp.vec3(xzero, 0.0, 0.0)
    if k == 1:
      xlow = xzero + wp.min(0.1 / kiso, 0.1 * (xiso - xzero))
      xfoot = xzero + 0.5 * (xlow - xzero)
      return wp.vec3(xlow, klow * (xlow - xfoot), klow)
    return wp.vec3(xiso, 1.0, kiso)

  if curve == _TENDON:
    kiso = biasprm[6]
    ftoe = biasprm[7]
    xiso = 1.0 + biasprm[5]
    xtoe = (ftoe - 1.0) / kiso + xiso
    if k == 0:
      return wp.vec3(1.0, 0.0, 0.0)
    if k == 1:
      xfoot = 1.0 + (xtoe - 1.0) / 10.0
      xmid = (0.5 * ftoe - 1.0) / kiso + xiso
      dydxmid = 0.5 * ftoe / (xmid - xfoot)
      xctrl = xfoot + 0.5 * (xmid - xfoot)
      return wp.vec3(xctrl, dydxmid * (xctrl - xfoot), dydxmid)
    return wp.vec3(xtoe, ftoe, kiso)

  # force-velocity: slopes at vmax from the damping, as OpenSim sets them
  fmax = dynprm[4]
  nearcon = dynprm[5]
  nearecc = dynprm[7]
  slopecon = float(0.0)
  slopeecc = float(0.0)
  if classic:
    slopecon = 0.5 * nearcon
    slopeecc = 0.5 * nearecc
  if k == 0:
    return wp.vec3(-1.0, 0.0, slopecon)
  if k == 1:
    return wp.vec3(-0.9, 0.05 * (nearcon + slopecon), nearcon)
  if k == 2:
    return wp.vec3(0.0, 1.0, dynprm[6])
  if k == 3:
    return wp.vec3(0.9, fmax - 0.05 * (nearecc + slopeecc), nearecc)
  return wp.vec3(1.0, fmax, slopeecc)


@wp.func
def _curviness(curve: int, section: int, dynprm: vec10, biasprm: vec10) -> float:
  """Scaled curviness of a section."""
  if curve == _ACTIVE:
    return _scale_curviness(1.0)
  if curve == _PASSIVE:
    return _scale_curviness(biasprm[4])
  if curve == _TENDON:
    return _scale_curviness(biasprm[8])
  if section < 2:
    return _scale_curviness(dynprm[8])
  return _scale_curviness(dynprm[9])


@wp.func
def _quintic(p0: float, p1: float, p2: float, p3: float, u: float) -> wp.vec2:
  """Value and derivative of the quintic Bezier with control points (p0, p1, p1, p2, p2, p3)."""
  b1 = 5.0 * (p1 - p0)
  b2 = 10.0 * (p0 - p1)
  b3 = 10.0 * (p2 - p0)
  b4 = 5.0 * (p0 + 2.0 * p1 - 3.0 * p2)
  b5 = p3 + 5.0 * p2 - 5.0 * p1 - p0
  value = ((((b5 * u + b4) * u + b3) * u + b2) * u + b1) * u + p0
  deriv = (((5.0 * b5 * u + 4.0 * b4) * u + 3.0 * b3) * u + 2.0 * b2) * u + b1
  return wp.vec2(value, deriv)


@wp.func
def curve(curve: int, x: float, dynprm: vec10, gainprm: vec10, biasprm: vec10) -> wp.vec2:
  """Millard2012 curve at x: (value, slope), OpenSim SmoothSegmentedFunction."""
  classic = _classic(dynprm, biasprm)
  n = _nknot(curve)

  # linear extrapolation
  first = _knot(curve, 0, dynprm, gainprm, biasprm, classic)
  if x <= first[0]:
    return wp.vec2(first[1] + first[2] * (x - first[0]), first[2])
  last = _knot(curve, n - 1, dynprm, gainprm, biasprm, classic)
  if x >= last[0]:
    return wp.vec2(last[1] + last[2] * (x - last[0]), last[2])

  # section
  i = int(0)
  k0 = first
  k1 = _knot(curve, 1, dynprm, gainprm, biasprm, classic)
  while i < n - 2 and x >= k1[0]:
    i += 1
    k0 = k1
    k1 = _knot(curve, i + 1, dynprm, gainprm, biasprm, classic)

  # corner control points: the end tangents meet at (xc, yc)
  x0 = k0[0]
  y0 = k0[1]
  d0 = k0[2]
  x1 = k1[0]
  y1 = k1[1]
  d1 = k1[2]
  xc = 0.5 * (x0 + x1)
  if wp.abs(d0 - d1) > 1.4901161193847656e-08:
    xc = (y1 - y0 - x1 * d1 + x0 * d0) / (d0 - d1)
  yc = (xc - x1) * d1 + y1
  c = _curviness(curve, i, dynprm, biasprm)
  px1 = x0 + c * (xc - x0)
  px2 = x1 + c * (xc - x1)
  py1 = y0 + c * (yc - y0)
  py2 = y1 + c * (yc - y1)

  # invert x(u) = x: x(u) is increasing, Newton safeguarded by bisection
  lo = float(0.0)
  hi = float(1.0)
  u = (x - x0) / (x1 - x0)
  for _ in range(40):
    xu = _quintic(x0, px1, px2, x1, u)
    f = xu[0] - x
    if f == 0.0:
      break
    if f > 0.0:
      hi = u
    else:
      lo = u
    nxt = u - f / xu[1]
    if not (nxt > lo and nxt < hi):
      nxt = 0.5 * (lo + hi)
    step = nxt - u
    u = nxt
    if wp.abs(step) < 1.0e-7:
      break

  xu = _quintic(x0, px1, px2, x1, u)
  yu = _quintic(y0, py1, py2, y1, u)
  return wp.vec2(yu[0], yu[1] / xu[1])


# Geyer2010: the eccentric force enhancement f_V(v_max), fixed because Song's eccentric branch
# joins the shortening branch at rest only for 1.5, and the eccentric curvature factor
_GEYER_N = 1.5
_GEYER_ECC = 7.56

# Geyer2010 has no minimum fiber length: the buffer element holds the fiber
_NOMIN = -1.0e10

_BUFFER = 4

vec8 = wp.types.vector(length=8, dtype=float)


@wp.func
def _geyer_active(x: float, width: float, logc: float) -> wp.vec2:
  """Active force-length, exp(ln(c) |(x - 1) / w|^3)."""
  u = (x - 1.0) / width
  au = wp.abs(u)
  y = wp.exp(logc * au * au * au)
  return wp.vec2(y, y * (3.0 * logc) * (u * au) / width)


@wp.func
def _geyer_stretch(x: float, rest: float, strain: float) -> wp.vec2:
  """Spring stretched beyond its rest length, ((x - rest) / strain)^2."""
  if x <= rest:
    return wp.vec2(0.0, 0.0)
  t = (x - rest) / strain
  return wp.vec2(t * t, 2.0 * t / strain)


@wp.func
def _geyer_compress(x: float, rest: float, strain: float) -> wp.vec2:
  """Spring compressed below its rest length, ((rest - x) / strain)^2."""
  if x >= rest:
    return wp.vec2(0.0, 0.0)
  t = (rest - x) / strain
  return wp.vec2(t * t, -2.0 * t / strain)


@wp.func
def _geyer_velocity(x: float, K: float) -> wp.vec2:
  """Force-velocity: Hill's hyperbola shortening, Song's hyperbola lengthening, beyond v_max too."""
  if x <= 0.0:
    den = 1.0 - K * x
    return wp.vec2((1.0 + x) / den, (1.0 + K) / (den * den))
  c = 1.0 - _GEYER_ECC * K
  b = c * x - 2.0
  return wp.vec2(_GEYER_N - (x - 1.0) / b, (2.0 - c) / (b * b))


@wp.func
def geyer_curve(curve: int, x: float, dynprm: vec10, gainprm: vec10, biasprm: vec10) -> wp.vec2:
  """Geyer2010 curve at x: (value, slope)."""
  width = gainprm[4]
  if curve == _ACTIVE:
    return _geyer_active(x, width, wp.log(gainprm[5]))
  if curve == _PASSIVE:
    return _geyer_stretch(x, biasprm[0], biasprm[1])
  if curve == _TENDON:
    return _geyer_stretch(x, 1.0, biasprm[2])
  if curve == _VELOCITY:
    return _geyer_velocity(x, dynprm[4])
  return _geyer_compress(x, 1.0 - width, 0.5 * width)


@wp.func
def _pennation(width: float, length: float) -> wp.vec3:
  """Fixed-width pennation: (cos(phi), sin(phi), dphi/dl), dphi at the clamped angle."""
  if width == 0.0:
    return wp.vec3(1.0, 0.0, 0.0)
  s = wp.min(width / length, _SINPHIMAX)
  c = wp.sqrt(1.0 - s * s)
  return wp.vec3(c, s, -(s / length) / c)


@wp.func
def _width(gainprm: vec10) -> float:
  return gainprm[1] * wp.sin(gainprm[4])


@wp.func
def _lmin(geyer: bool, gainprm: vec10) -> float:
  if geyer:
    return _NOMIN
  return wp.max(gainprm[5] * gainprm[1], _width(gainprm) / _SINPHIMAX)


@wp.func
def _beta(geyer: bool, dynprm: vec10) -> float:
  if geyer or dynprm[3] >= _MINDAMPING:
    return dynprm[3]
  return 0.0


@wp.func
def _rigid_flag(geyer: bool, biasprm: vec10) -> bool:
  if geyer:
    return biasprm[3] != 0.0
  return biasprm[9] != 0.0


@wp.func
def _millard_eval(
  length: float, lprev: float, lmtu: float, activation: float, dt: float, dynprm: vec10, gainprm: vec10, biasprm: vec10
) -> vec8:
  """Millard2012: f_T = cos(phi) (a f_L f_V + f_P + beta v)."""
  lopt = gainprm[1]
  lslack = gainprm[2]
  vmax = gainprm[3] * lopt
  beta = _beta(False, dynprm)
  pen = _pennation(_width(gainprm), length)
  cosphi = pen[0]
  sinphi = pen[1]
  dphi = pen[2]

  ln = length / lopt
  dln = 1.0 / lopt
  dvn = float(0.0)
  if dt > 0.0:
    dvn = 1.0 / (dt * vmax)
  vn = (length - lprev) * dvn

  lt = (lmtu - length * cosphi) / lslack
  dlt = (length * sinphi * dphi - cosphi) / lslack

  fl = curve(_ACTIVE, ln, dynprm, gainprm, biasprm)
  fp = curve(_PASSIVE, ln, dynprm, gainprm, biasprm)
  fv = curve(_VELOCITY, vn, dynprm, gainprm, biasprm)
  ft = curve(_TENDON, lt, dynprm, gainprm, biasprm)

  fiber = activation * fl[0] * fv[0] + fp[0] + beta * vn
  dfiber = activation * (fl[1] * dln * fv[0] + fl[0] * fv[1] * dvn) + fp[1] * dln + beta * dvn

  res = ft[0] - cosphi * fiber
  dres = ft[1] * dlt + sinphi * dphi * fiber - cosphi * dfiber
  return vec8(cosphi, fl[0] * fv[0], fp[0] + beta * vn, ft[0], ft[1], dlt, res, dres)


@wp.func
def _geyer_eval(
  length: float, lprev: float, lmtu: float, activation: float, dt: float, dynprm: vec10, gainprm: vec10, biasprm: vec10
) -> vec8:
  """Geyer2010: f_SE + f_BE = f_V (f_PE + a f_L) + beta v."""
  lopt = gainprm[1]
  lslack = gainprm[2]
  vmax = gainprm[3] * lopt
  width = gainprm[4]
  beta = dynprm[3]

  ln = length / lopt
  dln = 1.0 / lopt
  dvn = float(0.0)
  if dt > 0.0:
    dvn = 1.0 / (dt * vmax)
  vn = (length - lprev) * dvn
  lt = (lmtu - length) / lslack
  dlt = -1.0 / lslack

  fl = _geyer_active(ln, width, wp.log(gainprm[5]))
  fp = _geyer_stretch(ln, biasprm[0], biasprm[1])
  fb = _geyer_compress(ln, 1.0 - width, 0.5 * width)
  fv = _geyer_velocity(vn, dynprm[4])
  ft = _geyer_stretch(lt, 1.0, biasprm[2])

  active = fv[0] * fl[0]
  passive = fv[0] * fp[0] + beta * vn - fb[0]
  res = ft[0] - (activation * active + passive)
  dres = ft[1] * dlt - (
    activation * (fv[1] * dvn * fl[0] + fv[0] * fl[1] * dln)
    + fv[1] * dvn * fp[0]
    + fv[0] * fp[1] * dln
    + beta * dvn
    - fb[1] * dln
  )
  return vec8(1.0, active, passive, ft[0], ft[1], dlt, res, dres)


@wp.func
def _eval(
  geyer: bool,
  length: float,
  lprev: float,
  lmtu: float,
  activation: float,
  dt: float,
  dynprm: vec10,
  gainprm: vec10,
  biasprm: vec10,
) -> vec8:
  """Fiber force balance, backward Euler with v = (length - lprev) / dt, isometric for dt <= 0.

  Returns (cos(phi), active, passive, f_T, d f_T / d lt, d lt / d length, residual,
  d residual / d length).
  """
  if geyer:
    return _geyer_eval(length, lprev, lmtu, activation, dt, dynprm, gainprm, biasprm)
  return _millard_eval(length, lprev, lmtu, activation, dt, dynprm, gainprm, biasprm)


@wp.func
def _step(
  geyer: bool, length: float, lmtu: float, activation: float, dt: float, dynprm: vec10, gainprm: vec10, biasprm: vec10
) -> float:
  """One backward-Euler step of the fiber: Newton from the current length, steps below l_opt / 2."""
  lopt = gainprm[1]
  lmin = _lmin(geyer, gainprm)
  lprev = length
  maxstep = 0.5 * lopt
  l = wp.max(length, lmin)
  for _ in range(_MAXITER):
    e = _eval(geyer, l, lprev, lmtu, activation, dt, dynprm, gainprm, biasprm)
    if wp.abs(e[6]) < _TOL or e[7] == 0.0:
      break
    step = wp.clamp(-e[6] / e[7], -maxstep, maxstep)
    l = wp.max(lmin, l + step)
    if wp.abs(step) < _STEPTOL * lopt:
      break
  return l


@wp.func
def _isometric(geyer: bool, lmtu: float, activation: float, dynprm: vec10, gainprm: vec10, biasprm: vec10) -> float:
  """Isometric equilibrium: Newton safeguarded by a bracket, independent of the current fiber."""
  lopt = gainprm[1]

  # the residual is positive toward short fibers: at the minimum length, where the fiber is
  # clamped if it is not, or below the shortest length tried
  lo = _lmin(geyer, gainprm)
  e = vec8(0.0)
  if geyer:
    lo = float(0.0)
    for _ in range(64):
      e = _eval(geyer, lo, lo, lmtu, activation, 0.0, dynprm, gainprm, biasprm)
      if e[6] > 0.0:
        break
      lo -= lopt
  else:
    e = _eval(geyer, lo, lo, lmtu, activation, 0.0, dynprm, gainprm, biasprm)
    if e[6] <= 0.0:
      return lo

  hi = lo + lopt
  for _ in range(64):
    e = _eval(geyer, hi, hi, lmtu, activation, 0.0, dynprm, gainprm, biasprm)
    if e[6] < 0.0:
      break
    lo = hi
    hi += lopt
  if e[6] >= 0.0:
    return hi

  l = 0.5 * (lo + hi)
  for _ in range(100):
    e = _eval(geyer, l, l, lmtu, activation, 0.0, dynprm, gainprm, biasprm)
    if wp.abs(e[6]) < _TOL:
      break
    if e[6] > 0.0:
      lo = l
    else:
      hi = l
    nxt = l
    if e[7] != 0.0:
      nxt = l - e[6] / e[7]
    if not (nxt > lo and nxt < hi):
      nxt = 0.5 * (lo + hi)
    if nxt == l:
      break
    l = nxt
  return l


@wp.func
def _rigid_length(geyer: bool, lmtu: float, gainprm: vec10) -> float:
  """Rigid-tendon fiber length, for Millard2012 OpenSim's FixedWidthPennation calcFiberLength."""
  along = lmtu - gainprm[2]
  if geyer:
    return along
  width = _width(gainprm)
  l = along
  if width > 0.0:
    lmax = width / _SINPHIMAX
    if along >= lmax * wp.sqrt(1.0 - _SINPHIMAX * _SINPHIMAX):
      l = wp.sqrt(along * along + width * width)
    else:
      l = lmax
  return wp.max(l, _lmin(False, gainprm))


@wp.func
def _millard_rigid(lmtu: float, vmtu: float, activation: float, dynprm: vec10, gainprm: vec10, biasprm: vec10) -> wp.vec4:
  """Millard2012 rigid tendon: (gain, bias, d force / d v_mtu, d force / d l_mtu).

  OpenSim's ignore_tendon_compliance.
  """
  fmax = gainprm[0]
  lopt = gainprm[1]
  lslack = gainprm[2]
  vmax = gainprm[3] * lopt
  beta = _beta(False, dynprm)
  width = _width(gainprm)
  lmin = _lmin(False, gainprm)

  l = _rigid_length(False, lmtu, gainprm)
  pen = _pennation(width, l)
  cosphi = pen[0]
  dcos = -pen[1] * pen[2]

  # a buckled tendon imposes no velocity, a clamped fiber carries no force
  buckled = lmtu - l * cosphi < lslack * (1.0 - _BUCKLE)
  v = float(0.0)
  dvn = float(0.0)
  if not buckled:
    v = vmtu * cosphi
    dvn = vmtu * dcos / vmax
  if l < lmin or (l <= lmin and v <= 0.0):
    return wp.vec4(0.0)

  vn = v / vmax
  fl = curve(_ACTIVE, l / lopt, dynprm, gainprm, biasprm)
  fp = curve(_PASSIVE, l / lopt, dynprm, gainprm, biasprm)
  fv = wp.vec2(1.0, 0.0)
  if not buckled:
    fv = curve(_VELOCITY, vn, dynprm, gainprm, biasprm)
  fiber = activation * fl[0] * fv[0] + fp[0] + beta * vn
  if fiber <= 0.0:
    return wp.vec4(0.0)

  gain = -fmax * cosphi * fl[0] * fv[0]
  bias = -fmax * cosphi * (fp[0] + beta * vn)
  dvel = float(0.0)
  if not buckled:
    dvel = -fmax * cosphi * cosphi * (activation * fl[0] * fv[1] + beta) / vmax

  # length: the fiber moves with the path, along / l on the pennated branch
  dl = float(1.0)
  if width > 0.0:
    if l > width / _SINPHIMAX:
      dl = (lmtu - lslack) / l
    else:
      dl = 0.0
  dlen = float(0.0)
  if l > lmin:
    dfiber = activation * (fl[1] / lopt * fv[0] + fl[0] * fv[1] * dvn) + fp[1] / lopt + beta * dvn
    dlen = -fmax * (dcos * fiber + cosphi * dfiber) * dl
  return wp.vec4(gain, bias, dvel, dlen)


@wp.func
def act_dot(
  # In:
  geyer: bool,
  control: float,
  activation: float,
  fiber: float,
  lmtu: float,
  force_act: float,
  dt: float,
  dynprm: vec10,
  gainprm: vec10,
  biasprm: vec10,
) -> wp.vec2:
  """Activation and fiber act_dot.

  geyer selects Geyer2010, else Millard2012. force_act is the activation that multiplies the gain,
  which with actearly depends on the activation act_dot; pass a negative value to use the
  activation at the end of the step.
  """
  # activation, time constants scaled by the activation as in Millard et al. (2013); Millard2012
  # as OpenSim's MuscleFirstOrderActivationDynamicModel, excitation floored at the minimum
  # activation; Geyer2010 with a smooth switch between the time constants
  act_dot_activation = float(0.0)
  if geyer:
    prm = vec10(dynprm[0], dynprm[1], dynprm[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    act_dot_activation = util_misc.muscle_dynamics(control, activation, prm)
  else:
    prm = vec10(dynprm[0], dynprm[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    act_dot_activation = util_misc.muscle_dynamics(wp.max(control, dynprm[2]), activation, prm)

  # fiber: one backward-Euler step, or the rigid-tendon length
  lopt = gainprm[1]
  a = force_act
  if a < 0.0:
    a = activation + dt * act_dot_activation
  l = lopt * (1.0 + fiber)
  lnext = float(0.0)
  if _rigid_flag(geyer, biasprm):
    lnext = _rigid_length(geyer, lmtu, gainprm)
  else:
    lnext = _step(geyer, l, lmtu, a, dt, dynprm, gainprm, biasprm)
  return wp.vec2(act_dot_activation, (lnext - l) / (lopt * dt))


@wp.func
def gain_bias(
  # In:
  geyer: bool,
  fiber: float,
  fiber_dot: float,
  lmtu: float,
  vmtu: float,
  activation: float,
  dt: float,
  dynprm: vec10,
  gainprm: vec10,
  biasprm: vec10,
) -> wp.vec2:
  """Gain and bias at the fiber solved by act_dot."""
  rigid = _rigid_flag(geyer, biasprm)
  if rigid and not geyer:
    r = _millard_rigid(lmtu, vmtu, activation, dynprm, gainprm, biasprm)
    return wp.vec2(r[0], r[1])

  fmax = gainprm[0]
  lopt = gainprm[1]
  lmin = _lmin(geyer, gainprm)
  l = wp.max(lmin, lopt * (1.0 + fiber + dt * fiber_dot))
  if rigid:
    l = _rigid_length(geyer, lmtu, gainprm)
  lprev = lopt * (1.0 + fiber)
  e = _eval(geyer, l, lprev, lmtu, activation, dt, dynprm, gainprm, biasprm)

  # clamped fiber: the force balance does not hold, the tendon force is transmitted
  if l <= lmin and e[6] < 0.0:
    return wp.vec2(0.0, -fmax * e[3])

  # Geyer2010 rigid tendon: the fiber force at the path beyond the tendon, which only pulls
  if rigid and activation * e[1] + e[2] <= 0.0:
    return wp.vec2(0.0, 0.0)
  return wp.vec2(-fmax * e[0] * e[1], -fmax * e[0] * e[2])


@wp.func
def force_derivs(
  # In:
  geyer: bool,
  fiber: float,
  fiber_dot: float,
  lmtu: float,
  vmtu: float,
  activation: float,
  dt: float,
  dynprm: vec10,
  gainprm: vec10,
  biasprm: vec10,
) -> wp.vec2:
  """Force derivatives: (d force / d length, d force / d velocity).

  Through the fiber solve with a compliant tendon, whose force depends on lengths alone, and with
  the fiber moving with the path with a rigid tendon: with the path velocity for Millard2012, with
  the change of the fiber over the step for Geyer2010.
  """
  rigid = _rigid_flag(geyer, biasprm)
  if rigid and not geyer:
    r = _millard_rigid(lmtu, vmtu, activation, dynprm, gainprm, biasprm)
    return wp.vec2(r[3], r[2])

  fmax = gainprm[0]
  lopt = gainprm[1]
  lslack = gainprm[2]
  lmin = _lmin(geyer, gainprm)
  l = wp.max(lmin, lopt * (1.0 + fiber + dt * fiber_dot))
  if rigid:
    l = _rigid_length(geyer, lmtu, gainprm)
  lprev = lopt * (1.0 + fiber)
  e = _eval(geyer, l, lprev, lmtu, activation, dt, dynprm, gainprm, biasprm)

  # Geyer2010 rigid tendon: d force / d length of the fiber force
  if rigid:
    if activation * e[1] + e[2] <= 0.0:
      return wp.vec2(0.0, 0.0)
    return wp.vec2(-fmax * (e[4] * e[5] - e[7]), 0.0)

  # tendon stiffness, in series with the fiber unless the fiber is clamped
  ktendon = fmax * e[4] / lslack
  if (l <= lmin and e[6] < 0.0) or e[7] == 0.0:
    return wp.vec2(-ktendon, 0.0)
  return wp.vec2(-ktendon * (1.0 - e[4] * e[5] / e[7]), 0.0)


@wp.func
def equilibrium(geyer: bool, lmtu: float, activation: float, dynprm: vec10, gainprm: vec10, biasprm: vec10) -> float:
  """Isometric equilibrium fiber state, l_ce / l_opt - 1."""
  l = float(0.0)
  if _rigid_flag(geyer, biasprm):
    l = _rigid_length(geyer, lmtu, gainprm)
  else:
    l = _isometric(geyer, lmtu, activation, dynprm, gainprm, biasprm)
  return l / gainprm[1] - 1.0


@wp.kernel
def _equilibrium(
  # Model:
  actuator_dyntype: wp.array[int],
  actuator_actadr: wp.array[int],
  actuator_actnum: wp.array[int],
  actuator_dynprm: wp.array2d[vec10],
  actuator_gainprm: wp.array2d[vec10],
  actuator_biasprm: wp.array2d[vec10],
  # Data in:
  actuator_length_in: wp.array2d[float],
  # Data out:
  act_out: wp.array2d[float],
  act_dot_out: wp.array2d[float],
):
  worldid, uid = wp.tid()
  dyntype = actuator_dyntype[uid]
  if dyntype != DynType.MILLARD2012 and dyntype != DynType.GEYER2010:
    return
  fiber = actuator_actadr[uid]
  if fiber < 0:
    return
  activation = fiber + actuator_actnum[uid] - 1
  act_out[worldid, fiber] = equilibrium(
    dyntype == DynType.GEYER2010,
    actuator_length_in[worldid, uid],
    act_out[worldid, activation],
    actuator_dynprm[worldid % actuator_dynprm.shape[0], uid],
    actuator_gainprm[worldid % actuator_gainprm.shape[0], uid],
    actuator_biasprm[worldid % actuator_biasprm.shape[0], uid],
  )
  act_dot_out[worldid, fiber] = 0.0


@event_scope
def muscle_equilibrium(m: Model, d: Data):
  """Sets compliant-tendon muscle fibers to their isometric equilibrium at the actuator lengths.

  Call after forward, which computes the actuator lengths.
  """
  if not m.nu:
    return
  wp.launch(
    _equilibrium,
    dim=(d.nworld, m.nu),
    inputs=[
      m.actuator_dyntype,
      m.actuator_actadr,
      m.actuator_actnum,
      m.actuator_dynprm,
      m.actuator_gainprm,
      m.actuator_biasprm,
      d.actuator_length,
    ],
    outputs=[d.act, d.act_dot],
  )
