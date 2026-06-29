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
"""Contact-rich Hopper control through a PyTorch-MuJoCo-Warp bridge.

The demo validates selected action derivatives against central finite differences,
then trains a short-horizon neural policy through 32 differentiable physics
substeps. Each substep is an isolated Warp tape wrapped by torch.autograd.Function,
which is the intended SHAC-style integration boundary.

A CUDA-enabled PyTorch installation is required in addition to this project.

Run:
  python contrib/diff_examples/contact_hopper_torch.py \
    --output viz_out/contact_hopper_control_run.npz
"""

import argparse
import json
import warnings

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

XML = r"""
<mujoco model="hopper-contact-control">
  <compiler angle="radian"/>
  <option timestep="0.005" integrator="Euler" jacobian="sparse" solver="Newton" iterations="30"/>
  <default>
    <joint limited="true" armature="1" damping="1"/>
    <geom condim="3" solimp="0.8 0.8 0.01 0.5 2" margin="0.001" friction="0.9 0.005 0.0001"/>
    <general ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <geom name="floor" size="20 20 0.125" type="plane"/>
    <body name="torso" pos="0 0 1.25">
      <joint name="rootx" pos="0 0 -1.25" axis="1 0 0" type="slide" limited="false" armature="0" damping="0"/>
      <joint name="rootz" axis="0 0 1" type="slide" ref="1.25" limited="false" armature="0" damping="0"/>
      <joint name="rooty" axis="0 1 0" type="hinge" limited="false" armature="0" damping="0"/>
      <geom size="0.05 0.2" type="capsule"/>
      <body name="thigh" pos="0 0 -0.2">
        <joint name="thigh_joint" type="hinge" axis="0 -1 0" range="-2.61799 0"/>
        <geom size="0.05 0.225" pos="0 0 -0.225" type="capsule"/>
        <body name="leg" pos="0 0 -0.7">
          <joint name="leg_joint" pos="0 0 0.25" type="hinge" axis="0 -1 0" range="-2.61799 0"/>
          <geom size="0.04 0.25" type="capsule"/>
          <body name="foot" pos="0 0 -0.25">
            <joint name="foot_joint" type="hinge" axis="0 -1 0" range="-0.785398 0.785398"/>
            <geom size="0.06 0.195" pos="0.06 0 0" quat="0.707107 0 -0.707107 0" type="capsule"
                  friction="2 0.005 0.0001"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general joint="thigh_joint" gear="200 0 0 0 0 0"/>
    <general joint="leg_joint" gear="200 0 0 0 0 0"/>
    <general joint="foot_joint" gear="200 0 0 0 0 0"/>
  </actuator>
</mujoco>
"""

DEVICE = "cuda"
SUBSTEPS = 4
HORIZON = 16
GAMMA = 0.99
SNAPSHOT_EPOCHS = (0, 10, 30, 60, 120)

warnings.filterwarnings("ignore", message="Running the tape backwards", category=UserWarning)
wp.config.log_level = wp.LOG_ERROR
wp.init()
MJM = mujoco.MjModel.from_xml_string(XML)
M = mjw.put_model(MJM)
PRE = mjw.make_data(MJM)
for _ in range(24):
  mjw.step(M, PRE)
wp.synchronize_device()
QPOS0 = torch.tensor(PRE.qpos.numpy()[0], dtype=torch.float32, device=DEVICE)[None]
QVEL0 = torch.tensor(PRE.qvel.numpy()[0], dtype=torch.float32, device=DEVICE)[None]
INITIAL_CONTACTS = int(PRE.nacon.numpy()[0])


class Step(torch.autograd.Function):
  """One MuJoCo Warp step as an isolated PyTorch autograd node."""

  @staticmethod
  def forward(ctx, qpos, qvel, ctrl):
    inputs = []
    for tensor in (qpos.contiguous(), qvel.contiguous(), ctrl.contiguous()):
      base = wp.from_torch(tensor, requires_grad=False)
      inputs.append(wp.from_torch(tensor, requires_grad=True, grad=wp.zeros_like(base)))

    d = mjw.make_diff_data(MJM)
    qpos_out = wp.empty_like(d.qpos, requires_grad=True)
    qvel_out = wp.empty_like(d.qvel, requires_grad=True)
    tape = wp.Tape()
    with tape:
      wp.copy(d.qpos, inputs[0])
      wp.copy(d.qvel, inputs[1])
      wp.copy(d.ctrl, inputs[2])
      mjw.step(M, d)
      wp.copy(qpos_out, d.qpos)
      wp.copy(qvel_out, d.qvel)
    wp.synchronize_device()
    ctx.tape = tape
    ctx.inputs = tuple(inputs)
    ctx.outputs = (qpos_out, qvel_out)
    ctx.data = d
    return wp.to_torch(qpos_out, requires_grad=False), wp.to_torch(qvel_out, requires_grad=False)

  @staticmethod
  def backward(ctx, qpos_grad, qvel_grad):
    qpos_out, qvel_out = ctx.outputs
    ctx.tape.backward(
      grads={
        qpos_out: wp.from_torch(qpos_grad.contiguous()),
        qvel_out: wp.from_torch(qvel_grad.contiguous()),
      }
    )
    wp.synchronize_device()
    return tuple(wp.to_torch(value.grad).clone() for value in ctx.inputs)


def observation(qpos, qvel):
  return torch.cat((qpos[:, 1:], qvel), dim=-1)


def step_action(qpos, qvel, action):
  for _ in range(SUBSTEPS):
    qpos, qvel = Step.apply(qpos, qvel, action)
  return qpos, qvel


def reward(qpos, qvel, action):
  forward = qvel[:, 0]
  posture = -0.25 * (qpos[:, 1] - 1.25).square() - 0.25 * qpos[:, 2].square()
  effort = -1.0e-3 * action.square().sum(dim=-1)
  return forward + 1.0 + posture + effort


def rollout_actions(actions):
  qpos = QPOS0.clone()
  qvel = QVEL0.clone()
  total = torch.zeros((), device=DEVICE)
  discount = 1.0
  for action in actions:
    qpos, qvel = step_action(qpos, qvel, action[None])
    total = total + discount * reward(qpos, qvel, action[None]).sum()
    discount *= GAMMA
  return -total, qpos, qvel


def gradient_check():
  actions = torch.zeros((8, MJM.nu), device=DEVICE, requires_grad=True)
  loss, qpos, _ = rollout_actions(actions)
  loss.backward()
  indices = ((0, 0), (0, 1), (2, 0), (4, 1), (6, 2))
  ad = np.asarray([float(actions.grad[i, j]) for i, j in indices])
  fd = []
  epsilon = 2.0e-3
  with torch.no_grad():
    for i, j in indices:
      plus = actions.detach().clone()
      minus = actions.detach().clone()
      plus[i, j] += epsilon
      minus[i, j] -= epsilon
      fd.append((float(rollout_actions(plus)[0]) - float(rollout_actions(minus)[0])) / (2.0 * epsilon))
  fd = np.asarray(fd)
  relative_error = float(np.linalg.norm(ad - fd) / max(np.linalg.norm(fd), 1.0e-12))
  cosine = float(np.dot(ad, fd) / max(np.linalg.norm(ad) * np.linalg.norm(fd), 1.0e-12))
  result = {
    "phase": "gradcheck",
    "loss": float(loss),
    "qpos": qpos.detach().cpu().tolist(),
    "ad": ad.tolist(),
    "fd": fd.tolist(),
    "relative_error": relative_error,
    "cosine": cosine,
  }
  print(json.dumps(result, sort_keys=True))
  if relative_error >= 2.0e-3 or cosine <= 0.999:
    raise RuntimeError(f"gradient check failed: {result}")
  return result


def contact_counts(qpos_trajectory, qvel_trajectory):
  data = mujoco.MjData(MJM)
  counts = []
  for qpos, qvel in zip(qpos_trajectory[1:], qvel_trajectory[1:]):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(MJM, data)
    counts.append(data.ncon)
  return np.asarray(counts, dtype=np.int32)


def train(epochs=121):
  torch.manual_seed(7)
  policy = torch.nn.Sequential(torch.nn.Linear(11, 64), torch.nn.Tanh(), torch.nn.Linear(64, 3)).to(DEVICE)
  torch.nn.init.zeros_(policy[-1].weight)
  torch.nn.init.zeros_(policy[-1].bias)
  optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-3)

  returns = np.zeros(epochs, dtype=np.float32)
  gradient_norms = np.zeros(epochs, dtype=np.float32)
  snapshot_epochs = []
  snapshot_qpos = []
  snapshot_qvel = []
  snapshot_contacts = []

  for epoch in range(epochs):
    optimizer.zero_grad()
    qpos = QPOS0.clone()
    qvel = QVEL0.clone()
    qpos_trajectory = [qpos.detach().cpu().numpy()[0].copy()]
    qvel_trajectory = [qvel.detach().cpu().numpy()[0].copy()]
    total = torch.zeros((), device=DEVICE)
    discount = 1.0
    for _ in range(HORIZON):
      action = torch.tanh(policy(observation(qpos, qvel)))
      qpos, qvel = step_action(qpos, qvel, action)
      qpos_trajectory.append(qpos.detach().cpu().numpy()[0].copy())
      qvel_trajectory.append(qvel.detach().cpu().numpy()[0].copy())
      total = total + discount * reward(qpos, qvel, action).sum()
      discount *= GAMMA
    loss = -total
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
    optimizer.step()

    returns[epoch] = float(total)
    gradient_norms[epoch] = gradient_norm
    if epoch in SNAPSHOT_EPOCHS or epoch == epochs - 1:
      qpos_np = np.asarray(qpos_trajectory)
      qvel_np = np.asarray(qvel_trajectory)
      snapshot_epochs.append(epoch)
      snapshot_qpos.append(qpos_np)
      snapshot_qvel.append(qvel_np)
      snapshot_contacts.append(contact_counts(qpos_np, qvel_np))
      print(
        json.dumps(
          {
            "phase": "train",
            "epoch": epoch,
            "return": float(total),
            "gradient_norm": gradient_norm,
            "qpos": qpos.detach().cpu().tolist(),
          },
          sort_keys=True,
        )
      )

  result = {
    "phase": "summary",
    "start_return": float(returns[0]),
    "final_return": float(returns[-1]),
    "delta": float(returns[-1] - returns[0]),
    "finite": bool(np.isfinite(returns).all() and np.isfinite(gradient_norms).all()),
    "initial_contacts": INITIAL_CONTACTS,
  }
  print(json.dumps(result, sort_keys=True))
  if not result["finite"] or result["delta"] <= 1.0:
    raise RuntimeError(f"optimization did not improve enough: {result}")
  return {
    "returns": returns,
    "gradient_norms": gradient_norms,
    "snapshot_epochs": np.asarray(snapshot_epochs, dtype=np.int32),
    "snapshot_qpos": np.asarray(snapshot_qpos),
    "snapshot_qvel": np.asarray(snapshot_qvel),
    "snapshot_contacts": np.asarray(snapshot_contacts),
    **result,
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type=int, default=121)
  parser.add_argument("--output")
  args = parser.parse_args()

  check = gradient_check()
  result = train(args.epochs)
  if args.output:
    np.savez_compressed(
      args.output,
      **result,
      model_xml=np.asarray(XML),
      gradcheck_ad=np.asarray(check["ad"]),
      gradcheck_fd=np.asarray(check["fd"]),
      gradcheck_relative_error=check["relative_error"],
      gradcheck_cosine=check["cosine"],
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
  main()
