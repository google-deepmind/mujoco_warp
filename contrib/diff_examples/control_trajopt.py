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
"""End-to-end gradient-based trajectory optimization (control) example.

Demonstrates that the differentiable step produces gradients usable for an
actual control problem, not just gradients that match finite differences.

A 2-link actuated pendulum optimizes a control sequence by gradient descent
(Adam) so the final joint configuration reaches a target. The loss is the
squared final-state error; dL/dctrl flows through the whole multi-step rollout
(smooth dynamics + integrator) via the Warp tape.

Run:
    uv run python contrib/diff_examples/control_trajopt.py
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

# 2-link actuated pendulum (acrobot-like), sparse jacobian, no contacts so the
# gradient path is the smooth-dynamics + integrator chain end to end.
XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body pos="0 0 0">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="12"/>
    <motor joint="j1" gear="12"/>
  </actuator>
</mujoco>
"""

N_STEPS = 60  # rollout horizon (0.6 s at dt=0.01)
N_ITERS = 200  # optimization iterations
TARGET = np.array([1.0, -0.6], dtype=np.float32)  # desired final joint angles (dynamically reachable)


@wp.kernel
def _final_qpos_loss(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  target: wp.array[float],
  loss: wp.array[float],
):
  wid, j = wp.tid()
  d = qpos_in[wid, j] - target[j]
  wp.atomic_add(loss, 0, d * d)


def main():
  wp.init()
  mjm, mjd, m, d = test_data.fixture(xml=XML)
  nu = mjm.nu
  nq = mjm.nq

  target = wp.array(TARGET, dtype=float)

  # Control sequence we optimize: (N_STEPS, nu). Start at zero (hangs down).
  u = np.zeros((N_STEPS, nu), dtype=np.float32)

  # Adam optimizer state.
  mt = np.zeros_like(u)
  vt = np.zeros_like(u)
  lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8

  qpos0 = mjd.qpos.copy()
  qvel0 = mjd.qvel.copy()

  print(f"{'iter':>5} {'loss':>14} {'final_q0':>10} {'final_q1':>10} {'|grad|':>12}")
  loss_hist = []
  for it in range(N_ITERS):
    # Fresh diff data each iteration (clean tape / state).
    d = mjw.make_diff_data(mjm)
    enable_grad(d)
    mjw.reset_data(m, d)
    d.qpos = wp.array(qpos0.reshape(1, -1), dtype=float, requires_grad=True)
    d.qvel = wp.array(qvel0.reshape(1, -1), dtype=float, requires_grad=True)

    # Per-step control arrays, all requires_grad so the tape tracks each.
    ctrl_arrays = [wp.array(u[k].reshape(1, -1), dtype=float, requires_grad=True) for k in range(N_STEPS)]

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      for k in range(N_STEPS):
        wp.copy(d.ctrl, ctrl_arrays[k])
        mjw.step(m, d)
      wp.launch(_final_qpos_loss, dim=(1, nq), inputs=[d.qpos, target, loss])
    tape.backward(loss=loss)

    g = np.stack([ctrl_arrays[k].grad.numpy()[0, :nu] for k in range(N_STEPS)], axis=0)
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    lval = float(loss.numpy()[0])
    fq = d.qpos.numpy()[0, :nq].copy()
    loss_hist.append(lval)
    if it % 10 == 0 or it == N_ITERS - 1:
      print(f"{it:>5} {lval:>14.6e} {fq[0]:>10.4f} {fq[1]:>10.4f} {np.linalg.norm(g):>12.4e}")

    # Adam update.
    mt = b1 * mt + (1 - b1) * g
    vt = b2 * vt + (1 - b2) * (g * g)
    mhat = mt / (1 - b1 ** (it + 1))
    vhat = vt / (1 - b2 ** (it + 1))
    u -= lr * mhat / (np.sqrt(vhat) + eps)
    u = np.clip(u, -1.0, 1.0)
    tape.zero()

  final_q = d.qpos.numpy()[0, :nq].copy()
  err = float(np.linalg.norm(final_q - TARGET))
  print("\nresult")
  print(f"  target final angles : {TARGET}")
  print(f"  reached final angles: {final_q}")
  print(f"  loss start -> end   : {loss_hist[0]:.6e} -> {loss_hist[-1]:.6e}")
  print(f"  reduction factor    : {loss_hist[0] / max(loss_hist[-1], 1e-30):.1f}x")
  print(f"  final state error   : {err:.4e} rad")
  converged = loss_hist[-1] < 0.05 * loss_hist[0] and err < 0.15
  print(f"  converged           : {converged}")
  return converged


if __name__ == "__main__":
  raise SystemExit(0 if main() else 1)
