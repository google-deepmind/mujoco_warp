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
"""Utilities for analytic backward passes."""

import dataclasses

import warp as wp

from mujoco_warp._src import math
from mujoco_warp._src.types import Data

wp.set_module_options({"enable_backward": False})


def _launch_vjp(kernel, dim, pairs, outputs, adj_outputs):
  """Launches a generated reverse with inputs paired to their cotangents."""
  wp.launch(
    kernel,
    dim=dim,
    inputs=[value for value, _ in pairs],
    outputs=outputs,
    adj_inputs=[adjoint for _, adjoint in pairs],
    adj_outputs=adj_outputs,
    adjoint=True,
  )


def _clone_nograd(d: Data) -> Data:
  """Deep-clone a Data's wp.arrays, grads off, so replay never mutates the tape-tracked d_out."""

  def clone(value):
    if isinstance(value, wp.array):
      array = wp.clone(value)
      array.requires_grad = False
      return array
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
      return dataclasses.replace(
        value, **{field.name: clone(getattr(value, field.name)) for field in dataclasses.fields(value)}
      )
    return value

  return clone(d)


# Column grad-seed / arithmetic primitives (per-(world, col), out-of-place unless noted).
# dst_out[w,i] = -src[w,i] (seed adj_r = -lam for the smooth-param residual adjoint)
@wp.kernel
def _neg_cols(src: wp.array2d[float], dst_out: wp.array2d[float]):
  w, i = wp.tid()
  dst_out[w, i] = -src[w, i]


# out = a - b: integrator-direct adjoint minus residual-VJP scatter (dr/dtheta)^T lam
@wp.kernel
def _sub_cols(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float]):
  w, i = wp.tid()
  out[w, i] = a[w, i] - b[w, i]


# out += a
@wp.kernel
def _accum_cols(a: wp.array2d[float], out: wp.array2d[float]):
  w, i = wp.tid()
  out[w, i] = out[w, i] + a[w, i]


# Safe sqrt keeps the standard forward value and suppresses the singular adjoint at zero.
# This avoids 0 * inf when sqrt(0) is evaluated in an unselected wp.where operand.


_wp_sqrt = wp.sqrt


@wp.func
def safe_sqrt(x: float):
  return _wp_sqrt(x)


@wp.func_grad(safe_sqrt)
def _adj_safe_sqrt_f32(x: wp.float32, adj_ret: wp.float32):
  if x > wp.float32(0.0):
    wp.adjoint[x] += adj_ret / (wp.float32(2.0) * _wp_sqrt(x))


@wp.func
def safe_sqrt(x: wp.float64):  # concrete overload (same name) so float64 sqrt still works
  return _wp_sqrt(x)


@wp.func_grad(safe_sqrt)
def _adj_safe_sqrt_f64(x: wp.float64, adj_ret: wp.float64):
  if x > wp.float64(0.0):
    wp.adjoint[x] += adj_ret / (wp.float64(2.0) * _wp_sqrt(x))


# Custom quaternion adjoints preserve derivatives at zero velocity and identity rotations,
# where Warp's guarded length and normalize adjoints return zero.


@wp.func
def _sinc(x: float) -> float:
  """sin(x)/x with a series fallback near zero."""
  if x * x < 1.0e-4:
    return 1.0 - x * x / 6.0
  return wp.sin(x) / x


@wp.func
def _dsinc_x(x: float) -> float:
  """sinc'(x)/x = (x*cos(x) - sin(x))/x^3, smooth and even; series near zero."""
  if x * x < 1.0e-4:
    return -1.0 / 3.0 + x * x / 30.0
  return (x * wp.cos(x) - wp.sin(x)) / (x * x * x)


def _adj_quat_integrate(q: wp.quat, v: wp.vec3, dt: float, adj_ret: wp.quat):
  # smooth-form recompute: out = normalize(qn*r), qn = normalize(q), r = (cos(x), (dt/2)*sinc(x)*v)
  nq2 = wp.dot(q, q)
  if nq2 == 0.0:
    return
  half_dt = 0.5 * dt
  x = half_dt * wp.length(v)
  s = half_dt * _sinc(x)
  r = wp.quat(wp.cos(x), s * v[0], s * v[1], s * v[2])
  qn = q / _wp_sqrt(nq2)
  u = math.mul_quat(qn, r)
  nu = wp.length(u)
  out = u / nu
  # final normalize
  adj_u = (adj_ret - out * wp.dot(out, adj_ret)) / nu
  # u = qn*r: transpose of quaternion left/right multiplication is multiplication by the conjugate
  adj_qn = math.mul_quat(adj_u, math.quat_inv(r))
  adj_r = math.mul_quat(math.quat_inv(qn), adj_u)
  # qn = normalize(q)
  adj_q = (adj_qn - qn * wp.dot(qn, adj_qn)) / _wp_sqrt(nq2)
  # r's x-chain collapses to smooth even functions of x times v (no v/|v| left):
  # d(cos x)/dv = -sinc(x)*(dt/2)^2*v, d((dt/2)sinc(x))/dv = (dt/2)^3*(sinc'(x)/x)*v
  adj_rv = wp.vec3(adj_r[1], adj_r[2], adj_r[3])
  coef = half_dt * half_dt * (-adj_r[0] * _sinc(x) + half_dt * wp.dot(v, adj_rv) * _dsinc_x(x))
  wp.adjoint[q] += adj_q
  wp.adjoint[v] += s * adj_rv + coef * v
  # NOTE: dt adjoint intentionally not propagated (opt_timestep is not a differentiated parameter)


def _adj_quat_to_vel(quat: wp.quat, adj_ret: wp.vec3):
  axis = wp.vec3(quat[1], quat[2], quat[3])
  s2 = wp.dot(axis, axis)
  w = quat[0]
  w2 = w * w
  n2 = s2 + w2
  if n2 == 0.0:
    return
  ga = wp.dot(axis, adj_ret)
  # vel = scale*axis with scale = wrapped_speed/|axis|; dd = (d scale/d|axis|)/|axis|
  if s2 < 1.0e-4 * w2:
    # atan series in t2 = (|axis|/w)^2, valid for either sign of w (angle near 0 or 2*pi)
    t2 = s2 / w2
    scale = (2.0 / w) * (1.0 - t2 / 3.0)
    dd = -(4.0 / (3.0 * w * w2)) * (1.0 - 1.2 * t2)
  else:
    sin_a_2 = _wp_sqrt(s2)
    speed = 2.0 * wp.atan2(sin_a_2, w)
    # same pi-wrap branch as the forward: the adjoint is the taken branch's one-sided derivative
    if speed > wp.pi:
      speed -= 2.0 * wp.pi
    scale = speed / sin_a_2
    dd = (2.0 * w / n2 - scale) / s2
  adj_axis = scale * adj_ret + (ga * dd) * axis
  # d scale/d w = -2/n2 exactly, in both branches (the wrap shift is constant in w)
  wp.adjoint[quat] += wp.quat(-2.0 * ga / n2, adj_axis[0], adj_axis[1], adj_axis[2])


@wp.func
def _adj_rotation(rotation: wp.mat33, adj_rotation: wp.mat33) -> wp.vec3:
  """Maps a rotation-matrix cotangent to its angular tangent cotangent."""
  return (
    wp.cross(
      wp.vec3(rotation[0, 0], rotation[1, 0], rotation[2, 0]),
      wp.vec3(adj_rotation[0, 0], adj_rotation[1, 0], adj_rotation[2, 0]),
    )
    + wp.cross(
      wp.vec3(rotation[0, 1], rotation[1, 1], rotation[2, 1]),
      wp.vec3(adj_rotation[0, 1], adj_rotation[1, 1], adj_rotation[2, 1]),
    )
    + wp.cross(
      wp.vec3(rotation[0, 2], rotation[1, 2], rotation[2, 2]),
      wp.vec3(adj_rotation[0, 2], adj_rotation[1, 2], adj_rotation[2, 2]),
    )
  )
