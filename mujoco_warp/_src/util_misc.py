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

import warp as wp

from .types import MJ_MINVAL
from .types import vec10


@wp.func
def muscle_gain_length(length: float, lmin: float, lmax: float) -> float:
  """Normalized muscle length-gain curve."""
  if (lmin <= length) and (length <= lmax):
    # mid-ranges (maximum is at 1.0)
    a = 0.5 * (lmin + 1.0)
    b = 0.5 * (1.0 + lmax)

    if length <= a:
      x = (length - lmin) / wp.max(MJ_MINVAL, a - lmin)
      return 0.5 * x * x
    elif length <= 1.0:
      x = (1.0 - length) / wp.max(MJ_MINVAL, 1.0 - a)
      return 1.0 - 0.5 * x * x
    elif length <= b:
      x = (length - 1.0) / wp.max(MJ_MINVAL, b - 1.0)
      return 1.0 - 0.5 * x * x
    else:
      x = (lmax - length) / wp.max(MJ_MINVAL, lmax - b)
      return 0.5 * x * x
  else:
    return 0.0


@wp.func
def muscle_gain(
  len: float, vel: float, lengthrange: wp.vec2, acc0: float, prm: vec10
) -> float:
  """Muscle active force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax)."""

  # unpack parameters
  range_ = wp.vec2(prm[0], prm[1])
  force = prm[2]
  scale = prm[3]
  lmin = prm[4]
  lmax = prm[5]
  vmax = prm[6]
  fvmax = prm[8]

  # scale force if negative
  if force < 0.0:
    force = scale / wp.max(MJ_MINVAL, acc0)

  # optimum length
  L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])

  # normalized length and velocity
  L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)
  V = vel / wp.max(MJ_MINVAL, L0 * vmax)

  # length curve
  FL = muscle_gain_length(L, lmin, lmax)

  # velocity curve
  y = fvmax - 1.0
  if V <= -1.0:
    FV = 0.0
  elif V <= 0.0:
    FV = (V + 1.0) * (V + 1.0)
  elif V <= y:
    FV = fvmax - (y - V) * (y - V) / wp.max(MJ_MINVAL, y)
  else:
    FV = fvmax

  # compute FVL and scale, make it negative
  return -force * FL * FV


@wp.func
def muscle_bias(len: float, lengthrange: wp.vec2, acc0: float, prm: vec10) -> float:
  """Muscle passive force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax)."""

  # unpack parameters
  range_ = wp.vec2(prm[0], prm[1])
  force = prm[2]
  scale = prm[3]
  lmax = prm[5]
  fpmax = prm[7]

  # scale force if negative
  if force < 0.0:
    force = scale / wp.max(MJ_MINVAL, acc0)

  # optimum length
  L0 = (lengthrange[1] - lengthrange[0]) / wp.max(MJ_MINVAL, range_[1] - range_[0])

  # normalized length
  L = range_[0] + (len - lengthrange[0]) / wp.max(MJ_MINVAL, L0)

  # half-quadratic to (L0 + lmax) / 2, linear beyond
  b = 0.5 * (1.0 + lmax)
  if L <= 1.0:
    return 0.0
  elif L <= b:
    x = (L - 1.0) / wp.max(MJ_MINVAL, b - 1.0)
    return -force * fpmax * 0.5 * x * x
  else:
    x = (L - b) / wp.max(MJ_MINVAL, b - 1.0)
    return -force * fpmax * (0.5 + x)


@wp.func
def _sigmoid(x: float) -> float:
  """Sigmoid function over 0 <= x <= 1 using quintic polynomial."""

  # fast return
  if x <= 0.0:
    return 0.0

  if x >= 1.0:
    return 1.0

  # sigmoid f(x) = 6 * x^5 - 15 * x^4 + 10 * x^3
  # solution of f(0) = f'(0) = f''(0) = 0, f(1) = 1, f'(1) = f''(1) = 0
  return x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)


@wp.func
def muscle_dynamics_timescale(
  dctrl: float, tau_act: float, tau_deact: float, smooth_width: float
) -> float:
  """Muscle time constant with optional smoothing."""

  # hard switching
  if smooth_width < MJ_MINVAL:
    if dctrl > 0.0:
      return tau_act
    else:
      return tau_deact
  else:  # smooth switching
    # scale by width, center around 0.5 midpoint, rescale to bounds
    return tau_deact + (tau_act - tau_deact) * _sigmoid(dctrl / smooth_width + 0.5)


@wp.func
def muscle_dynamics(ctrl: float, act: float, prm: vec10) -> float:
  """Muscle activation dynamics, prm = (tau_act, tau_deact, smooth_width)."""

  # clamp control
  ctrlclamp = wp.clamp(ctrl, 0.0, 1.0)

  # clamp activation
  actclamp = wp.clamp(act, 0.0, 1.0)

  # compute timescales as in Millard et al. (2013) https://doi.org/10.1115/1.4023390
  tau_act = prm[0] * (0.5 + 1.5 * actclamp)  # activation timescale
  tau_deact = prm[1] / (0.5 + 1.5 * actclamp)  # deactivation timescale
  smooth_width = prm[2]  # width of smoothing sigmoid
  dctrl = ctrlclamp - act  # excess excitation

  tau = muscle_dynamics_timescale(dctrl, tau_act, tau_deact, smooth_width)

  # filter output
  return dctrl / wp.max(MJ_MINVAL, tau)
