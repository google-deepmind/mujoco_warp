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
"""Shared adjoint leaves: safe_sqrt, a non-grad Data clone, forward-only accumulate kernels."""

import dataclasses

import warp as wp

from mujoco_warp._src.types import Data
from mujoco_warp._src.types import vec5

wp.set_module_options({"enable_backward": False})


# safe sqrt: forward byte-identical to wp.sqrt, custom grad 0 at x<=0. wp.sqrt's reverse is
# 0.5/sqrt(x) -> inf at 0, and warp differentiates both wp.where arms, so sqrt(0) in an untaken
# branch yields 0*inf = NaN. definitions here; adjoint.py installs the global wp.sqrt swap.
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


def _clone_nograd(d: Data) -> Data:
  """Deep-clone a Data's wp.arrays, grads off, so replay never mutates the tape-tracked d_out."""

  def cl(o):
    if isinstance(o, wp.array):
      c = wp.clone(o)
      c.requires_grad = False
      return c
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
      return dataclasses.replace(o, **{f.name: cl(getattr(o, f.name)) for f in dataclasses.fields(o)})
    return o

  return cl(d)


# Column grad-seed / arithmetic primitives (per-(world, col), out-of-place unless noted).
# dst_out[w,i] = src[w,i] over dst_out's columns (seed r.grad = lam from ctx.Mgrad[:, :nv])
@wp.kernel
def _copy_cols(src: wp.array2d[float], dst_out: wp.array2d[float]):
  w, i = wp.tid()
  dst_out[w, i] = src[w, i]


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


# out += a (accumulate the RNE-bias dqpos into the buffer _sub_cols subtracts)
@wp.kernel
def _accum_cols(a: wp.array2d[float], out: wp.array2d[float]):
  w, i = wp.tid()
  out[w, i] = out[w, i] + a[w, i]


# IFT-minus accumulate into shared param grads: grad_out -= res_in, one overload per param dtype.
# per-(world,dof) float param, e.g. dof_frictionloss
@wp.kernel
def _accum_neg(res_in: wp.array2d[float], grad_out: wp.array2d[float]):
  w, i = wp.tid()
  grad_out[w, i] -= res_in[w, i]


# per-constraint wp.vec2 param, e.g. *_solref
@wp.kernel
def _accum_neg_vec2(res_in: wp.array2d[wp.vec2], grad_out: wp.array2d[wp.vec2]):
  w, i = wp.tid()
  grad_out[w, i] -= res_in[w, i]


# per-body vec3 param, e.g. body_inertia, body_ipos
@wp.kernel
def _accum_neg_vec3(res_in: wp.array2d[wp.vec3], grad_out: wp.array2d[wp.vec3]):
  w, i = wp.tid()
  grad_out[w, i] -= res_in[w, i]


# per-constraint vec5 param, e.g. *_solimp
@wp.kernel
def _accum_neg_vec5(res_in: wp.array2d[vec5], grad_out: wp.array2d[vec5]):
  w, i = wp.tid()
  grad_out[w, i] -= res_in[w, i]


# per-body wp.quat param (body_iquat)
@wp.kernel
def _accum_neg_quat(res_in: wp.array2d[wp.quat], grad_out: wp.array2d[wp.quat]):
  w, i = wp.tid()
  grad_out[w, i] -= res_in[w, i]


# seed adj_phi = +1 per contact (phi = lam^T r_c already folds in the IFT seed lam)
@wp.kernel
def _fill_ones(out: wp.array[float]):
  out[wp.tid()] = 1.0
