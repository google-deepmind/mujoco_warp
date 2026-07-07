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
"""Shared adjoint helpers: safe_sqrt plus forward-only column / IFT-minus accumulate kernels."""

import warp as wp

from mujoco_warp._src.types import vec5

wp.set_module_options({"enable_backward": False})


# Safe sqrt: forward byte-identical to wp.sqrt, custom grad 0 at x<=0. wp.sqrt's reverse is
# 0.5/sqrt(x) -> inf at 0, and Warp differentiates BOTH wp.where arms, so sqrt(0) in an untaken
# branch yields 0*inf = NaN. Definitions here; adjoint.py installs the global wp.sqrt swap.
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


# Column grad-seed / arithmetic primitives (per-(world, col), out-of-place unless noted).
@wp.kernel
def _copy_cols(src: wp.array2d[float], dst_out: wp.array2d[float]):
  """dst_out[w,i] = src[w,i] over dst_out's columns (seed r.grad = lam from ctx.Mgrad[:, :nv])."""
  w, i = wp.tid()
  dst_out[w, i] = src[w, i]


@wp.kernel
def _neg_cols(src: wp.array2d[float], dst_out: wp.array2d[float]):
  """Compute dst_out[w,i] = -src[w,i] (seed adj_r = -lam for the smooth-param residual adjoint)."""
  w, i = wp.tid()
  dst_out[w, i] = -src[w, i]


@wp.kernel
def _sub_cols(a: wp.array2d[float], b: wp.array2d[float], out: wp.array2d[float]):
  """Write out = a - b: integrator-direct adjoint minus residual-VJP scatter (dr/dtheta)^T lam."""
  w, i = wp.tid()
  out[w, i] = a[w, i] - b[w, i]


@wp.kernel
def _accum_cols(a: wp.array2d[float], out: wp.array2d[float]):
  """Compute out += a (accumulate the RNE-bias dqpos into the buffer _sub_cols subtracts)."""
  w, i = wp.tid()
  out[w, i] = out[w, i] + a[w, i]


# IFT-minus accumulate into shared param grads: grad_out -= res_in, one overload per param dtype.
@wp.kernel
def _accum_neg(res_in: wp.array2d[float], grad_out: wp.array2d[float]):
  """Compute grad_out -= res_in per-(world,dof) FLOAT param (e.g. dof_frictionloss); IFT minus."""
  w, i = wp.tid()
  grad_out[w, i] -= res_in[w, i]


@wp.kernel
def _accum_neg_vec2(res_in: wp.array2d[wp.vec2], grad_out: wp.array2d[wp.vec2]):
  """Compute grad_out -= res_in per-constraint wp.vec2 model param (e.g. *_solref); IFT minus."""
  w, i = wp.tid()
  grad_out[w, i] -= res_in[w, i]


@wp.kernel
def _accum_neg_vec3(res_in: wp.array2d[wp.vec3], grad_out: wp.array2d[wp.vec3]):
  """Compute grad_out -= res_in per-body vec3 param (e.g. body_inertia, body_ipos); IFT minus."""
  w, i = wp.tid()
  grad_out[w, i] -= res_in[w, i]


@wp.kernel
def _accum_neg_vec5(res_in: wp.array2d[vec5], grad_out: wp.array2d[vec5]):
  """Compute grad_out -= res_in per-constraint vec5 model param (e.g. *_solimp); IFT minus."""
  w, i = wp.tid()
  grad_out[w, i] -= res_in[w, i]


@wp.kernel
def _accum_neg_quat(res_in: wp.array2d[wp.quat], grad_out: wp.array2d[wp.quat]):
  """Compute grad_out -= res_in per-body wp.quat model param (body_iquat); IFT minus."""
  w, i = wp.tid()
  grad_out[w, i] -= res_in[w, i]


@wp.kernel
def _fill_ones(out: wp.array[float]):
  """Seed adj_phi = +1 per contact (phi = lam^T r_c already folds in the IFT seed lam)."""
  out[wp.tid()] = 1.0
