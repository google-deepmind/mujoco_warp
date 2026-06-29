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
"""End-to-end gradient-based system identification example.

Demonstrates that the differentiable step produces gradients usable for an
actual sys ID problem: recovering an unknown physical parameter from observed
motion.

A pendulum swings under gravity with an unknown viscous friction coefficient
theta (friction torque tau = -theta * qvel). A ground-truth trajectory is
generated with theta_true, then theta is recovered from a wrong initial guess
by gradient descent on the trajectory-matching loss. Gradients dL/dtheta flow
through the differentiable qfrc_applied path.

A one-step prediction-error (teacher-forcing) formulation is used: the state is
reset to the observation at each transition and a single differentiable step is
taken. This is the standard equation-error formulation for identification and
keeps the scalar optimization fast and well conditioned.

Run:
    uv run python contrib/diff_examples/sysid_friction.py
"""

# The gradient path records forward-only kernels (constraint/solver internals)
# on the tape; Warp warns once per such kernel on backward. They are expected
# and not relevant to this example, so quiet them to keep the output readable.
import warnings as _warnings

import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src.grad import enable_grad

_warnings.filterwarnings("ignore", message="Running the tape backwards", category=UserWarning)

XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body pos="0 0 0">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

N_STEPS = 80
N_ITERS = 60
SUBSAMPLE = 4  # use every 4th transition for the gradient (scalar param: plenty)
THETA_TRUE = 0.35  # ground-truth viscous friction coefficient
THETA_INIT = 0.02  # deliberately wrong starting guess
Q0 = 1.4  # initial angle (released from near-horizontal)


@wp.kernel
def _viscous_force(
  # Data in:
  qvel_in: wp.array2d[float],
  # In:
  theta: wp.array[float],
  # Data out:
  qfrc_applied_out: wp.array2d[float],
):
  wid, j = wp.tid()
  qfrc_applied_out[wid, j] = -theta[0] * qvel_in[wid, j]


@wp.kernel
def _onestep_loss(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  obs_next: wp.array[float],
  loss: wp.array[float],
):
  wid, j = wp.tid()
  diff = qpos_in[wid, j] - obs_next[0]
  wp.atomic_add(loss, 0, diff * diff)


def rollout_observed(mjm, m, theta_val):
  """Generate the ground-truth observed trajectory (no grad). Returns qpos, qvel sequences."""
  _, _, _, d = test_data.fixture(xml=XML)
  mjw.reset_data(m, d)
  d.qpos = wp.array(np.array([[Q0]], dtype=np.float32), dtype=float)
  theta = wp.array([theta_val], dtype=float)
  qpos_seq = np.zeros(N_STEPS + 1, dtype=np.float32)
  qvel_seq = np.zeros(N_STEPS + 1, dtype=np.float32)
  qpos_seq[0] = d.qpos.numpy()[0, 0]
  qvel_seq[0] = d.qvel.numpy()[0, 0]
  for k in range(N_STEPS):
    wp.launch(_viscous_force, dim=(1, mjm.nv), inputs=[d.qvel, theta, d.qfrc_applied])
    mjw.step(m, d)
    qpos_seq[k + 1] = d.qpos.numpy()[0, 0]
    qvel_seq[k + 1] = d.qvel.numpy()[0, 0]
  return qpos_seq, qvel_seq


def main():
  wp.init()
  mjm, mjd, m, d = test_data.fixture(xml=XML)

  qpos_obs, qvel_obs = rollout_observed(mjm, m, THETA_TRUE)

  theta_val = THETA_INIT
  mt = vt = 0.0
  lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8

  print(f"theta_true = {THETA_TRUE}, theta_init = {THETA_INIT}")
  print(f"{'iter':>5} {'loss':>14} {'theta':>10} {'dL/dtheta':>14}")
  hist = []
  for it in range(N_ITERS):
    # Accumulate the one-step prediction-error gradient over the transitions.
    g_acc = 0.0
    l_acc = 0.0
    for k in range(0, N_STEPS, SUBSAMPLE):
      d = mjw.make_diff_data(mjm)
      enable_grad(d)
      mjw.reset_data(m, d)
      # Teacher forcing: start from the observed state at step k.
      d.qpos = wp.array(np.array([[qpos_obs[k]]], dtype=np.float32), dtype=float, requires_grad=True)
      d.qvel = wp.array(np.array([[qvel_obs[k]]], dtype=np.float32), dtype=float, requires_grad=True)
      obs_next = wp.array([qpos_obs[k + 1]], dtype=float)

      theta = wp.array([theta_val], dtype=float, requires_grad=True)
      loss = wp.zeros(1, dtype=float, requires_grad=True)
      tape = wp.Tape()
      with tape:
        wp.launch(_viscous_force, dim=(1, mjm.nv), inputs=[d.qvel, theta, d.qfrc_applied])
        mjw.step(m, d)
        wp.launch(_onestep_loss, dim=(1, 1), inputs=[d.qpos, obs_next, loss])
      tape.backward(loss=loss)
      g_acc += float(np.nan_to_num(theta.grad.numpy()[0]))
      l_acc += float(loss.numpy()[0])
      tape.zero()

    g = g_acc
    lval = l_acc
    hist.append((theta_val, lval))
    if it % 10 == 0 or it == N_ITERS - 1:
      print(f"{it:>5} {lval:>14.6e} {theta_val:>10.5f} {g:>14.4e}")

    mt = b1 * mt + (1 - b1) * g
    vt = b2 * vt + (1 - b2) * (g * g)
    mhat = mt / (1 - b1 ** (it + 1))
    vhat = vt / (1 - b2 ** (it + 1))
    lr_it = lr * (0.5 ** (it / 25.0))  # decay to settle the estimate
    theta_val -= lr_it * mhat / (np.sqrt(vhat) + eps)
    theta_val = float(np.clip(theta_val, 0.0, 5.0))

  rel_err = abs(theta_val - THETA_TRUE) / THETA_TRUE
  print("\nresult")
  print(f"  theta_true      : {THETA_TRUE}")
  print(f"  theta_recovered : {theta_val:.5f}")
  print(f"  relative error  : {rel_err * 100:.3f} %")
  print(f"  loss start->end : {hist[0][1]:.6e} -> {hist[-1][1]:.6e}")
  converged = rel_err < 0.02
  print(f"  converged       : {converged}")
  return converged


if __name__ == "__main__":
  raise SystemExit(0 if main() else 1)
