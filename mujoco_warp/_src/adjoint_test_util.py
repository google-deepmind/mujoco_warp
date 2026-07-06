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
"""Test-only oracles for the analytic step backward (the FD-of-rne smooth-dqpos reference)."""

import warp as wp

from mujoco_warp._src import forward
from mujoco_warp._src import passive
from mujoco_warp._src import smooth
from mujoco_warp._src import smooth_adjoint
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model

# kernel_analyzer: off -- test-only oracle kernels (moved verbatim from adjoint.py); param
# banners are intentionally omitted (not part of the shipped kernel surface).


@wp.kernel
def _perturb_col(base: wp.array2d[float], col: int, delta: float, out: wp.array2d[float]):
  """Write out = base with column `col` shifted by delta (one raw qpos coordinate)."""
  w, j = wp.tid()
  v = base[w, j]
  if j == col:
    v += delta
  out[w, j] = v


@wp.kernel
def _rsmooth_value(
  qfrc_bias: wp.array2d[float],
  qfrc_passive: wp.array2d[float],
  qfrc_actuator: wp.array2d[float],
  r_out: wp.array2d[float],
):
  """r_smooth = M*qacc - qfrc_smooth = rne(flg_acc).qfrc_bias - qfrc_passive - qfrc_actuator."""
  w, i = wp.tid()
  r_out[w, i] = qfrc_bias[w, i] - qfrc_passive[w, i] - qfrc_actuator[w, i]


@wp.kernel
def _fd_qpos_contract(
  r_plus: wp.array2d[float],
  r_minus: wp.array2d[float],
  lam: wp.array2d[float],
  nv: int,
  two_eps: float,
  col: int,
  adj_qpos: wp.array2d[float],
):
  """adj_qpos[:,col] += -sum_i (r_plus_i - r_minus_i)/(2 eps) * lam_i (central-FD dqpos VJP)."""
  w = wp.tid()
  s = float(0.0)
  for i in range(nv):
    s += (r_plus[w, i] - r_minus[w, i]) * lam[w, i]
  adj_qpos[w, col] += -s / two_eps


def _recompute_smooth_forces(m: Model, d: Data):
  """Recompute qfrc_{bias,passive,actuator} from d.qpos/qvel/ctrl (rne at frozen d.qacc)."""
  smooth.kinematics(m, d)
  smooth.com_pos(m, d)
  smooth.tendon(m, d)
  smooth.transmission(m, d)
  smooth.com_vel(m, d)
  passive.passive(m, d)
  smooth.rne(m, d, flg_acc=True)
  smooth.tendon_bias(m, d, d.qfrc_bias)
  forward.fwd_actuation(m, d)


def fd_smooth_qpos_backward(m: Model, d: Data, d_out: Data, lam: wp.array, res_qpos: wp.array, adj_qpos: wp.array):
  """Central-FD dqpos VJP: contract dr_smooth/dqpos with lam into adj_qpos (res_qpos unused).

  Drop-in for adjoint.smooth_qpos_backward (mock-patch over it); host eps loop -- NOT capture-safe.
  """
  nworld = d.qpos.shape[0]
  nq = d.qpos.shape[1]
  nv = m.nv
  # FD-of-rne TEST ORACLE: central-difference r_smooth at qpos +/- eps and contract with lam
  # (adj_qpos += -(dr_smooth/dqpos)^T lam). NOT capture-safe (host loop over nq); runs on a
  # separate non-grad clone -- never mutate the grad-tracked d_out.
  eps = 1.0e-4
  fd = smooth_adjoint._clone_for_fd(d_out)
  wp.copy(fd.qvel, d.qvel)
  if m.nu > 0:
    wp.copy(fd.ctrl, d.ctrl)
  r_plus = wp.empty((nworld, nv), dtype=float)
  r_minus = wp.empty((nworld, nv), dtype=float)
  for col in range(nq):
    wp.launch(_perturb_col, dim=(nworld, nq), inputs=[d.qpos, col, eps], outputs=[fd.qpos])
    _recompute_smooth_forces(m, fd)
    wp.launch(_rsmooth_value, dim=(nworld, nv), inputs=[fd.qfrc_bias, fd.qfrc_passive, fd.qfrc_actuator], outputs=[r_plus])
    wp.launch(_perturb_col, dim=(nworld, nq), inputs=[d.qpos, col, -eps], outputs=[fd.qpos])
    _recompute_smooth_forces(m, fd)
    wp.launch(_rsmooth_value, dim=(nworld, nv), inputs=[fd.qfrc_bias, fd.qfrc_passive, fd.qfrc_actuator], outputs=[r_minus])
    wp.launch(_fd_qpos_contract, dim=nworld, inputs=[r_plus, r_minus, lam, nv, 2.0 * eps, col], outputs=[adj_qpos])


# kernel_analyzer: on
