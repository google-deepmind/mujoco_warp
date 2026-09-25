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
"""Identify Unitree G1 armature scale from a recorded motion."""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import _demo
import _viz
import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

_G1 = Path(__file__).resolve().parents[2] / "benchmarks" / "unitree_g1"
_SCENE = _G1 / "scene_flat.xml"
_ROBOT = _G1 / "unitree_g1_mjlab.xml"
_MOTIONS = {"shuffle_dance": _G1 / "shuffle_dance.npz"}
_JOINT_QPOS_START = 7
_INITIAL_ARMATURE_SCALE = 2.0
_ARMATURE_SCALE_MIN = 0.1
_ARMATURE_SCALE_MAX = 10.0


@dataclass
class SysidG1Args(_demo.Args):
  sim_dt: float = 0.005
  cone: str = "pyramidal"
  iterations: int = 400
  lr: float = 0.03
  num_envs: int = 64
  viz_every: int = 50

  horizon: int = 200
  tbptt: int | None = 40
  eval_horizon: int = field(default=200, metadata={"help": "forward-only physics steps after the fitted motion"})
  motion: str = field(default="shuffle_dance", metadata={"help": "recorded motion", "choices": tuple(_MOTIONS)})
  spread: float = field(default=1.5, metadata={"help": "initial armature-scale spread"})


@wp.kernel
def _set_armature(
  # In:
  armature_scale: wp.array[float],
  base: wp.array[float],
  # Out:
  armature_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  armature_out[worldid, dofid] = armature_scale[worldid] * base[dofid]


@wp.kernel
def _set_ctrl(
  # Model:
  nu: int,
  # In:
  control: wp.array2d[float],
  step: wp.array[int],
  repeat: int,
  # Data out:
  ctrl_out: wp.array2d[float],
):
  worldid, actuatorid = wp.tid()
  frame = step[0] // repeat
  ctrl_out[worldid, actuatorid] = control[frame, actuatorid]


@wp.kernel
def _loss(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  reference: wp.array2d[float],
  step: wp.array[int],
  repeat: int,
  joint_start: int,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid, jointid = wp.tid()
  if (step[0] + 1) % repeat:
    return
  frame = (step[0] + 1) // repeat
  error = qpos_in[worldid, joint_start + jointid] - reference[frame, joint_start + jointid]
  wp.atomic_add(loss_out, 0, loss_scale * error * error)


@wp.kernel(enable_backward=False)
def _clamp_armature_scale(
  # In:
  minimum: float,
  maximum: float,
  # Out:
  armature_scale_out: wp.array[float],
):
  worldid = wp.tid()
  armature_scale_out[worldid] = wp.clamp(armature_scale_out[worldid], minimum, maximum)


def _initial_armature_scale(num_envs: int, spread: float, sampling="random", seed=0):
  scale = np.full(num_envs, _INITIAL_ARMATURE_SCALE, dtype=np.float32)
  if num_envs > 1:
    scale += _demo.sample_range(num_envs, -spread, spread, sampling, seed)
  return scale


def progress(args, model, data):
  reference = model.dof_armature_reference
  error = np.linalg.norm(model.dof_armature - reference, axis=1) / np.linalg.norm(reference)
  error_scale = 0.5  # map a 50% relative error to a raw progress of 0.5
  return 1.0 / (1.0 + (error / error_scale) ** 2.0)


class SysidG1(_demo.Demo):
  Args = SysidG1Args
  name = "sysid_g1"
  demo_type = _demo.DemoType.SYS_ID
  loss_type = _demo.LossType.PER_STEP
  model_fields = ("dof_armature", "dof_invweight0", "body_invweight0")
  data_kwargs = {"nconmax": 48, "njmax": 192}
  layout = _viz.Layout(columns=8, spacing=2.7)

  def __init__(self, args: SysidG1Args):
    mjm = mujoco.MjModel.from_xml_path(str(_SCENE))
    mjm.opt.iterations = 10
    mjm.opt.ls_iterations = 20
    with np.load(_MOTIONS[args.motion]) as motion:
      qpos = motion["qpos"].astype(np.float32)
      qvel = motion["qvel"].astype(np.float32)
      control = motion["ctrl"].astype(np.float32)
      times = motion["times"]

    frame_dt = float(np.median(np.diff(times)))
    self.repeat = _demo.control_steps(args.sim_dt, frame_dt)
    available = min(len(control), len(qpos) - 1)
    available_horizon = available * self.repeat
    if not 1 <= args.horizon <= available_horizon:
      raise ValueError(f"horizon must be in [1, {available_horizon}]")
    if args.horizon % self.repeat:
      raise ValueError(f"horizon must be divisible by {self.repeat} to align with motion frames")

    self.fit_frames = args.horizon // self.repeat
    available_eval_horizon = available_horizon - args.horizon
    if not 0 <= args.eval_horizon <= available_eval_horizon:
      raise ValueError(f"eval_horizon must be in [0, {available_eval_horizon}]")
    total_steps = args.horizon + args.eval_horizon
    total_frames = (total_steps + self.repeat - 1) // self.repeat
    super().__init__(
      args,
      mjm,
      args.horizon,
      eval_horizon=args.eval_horizon,
      loss_scale=1.0 / args.num_envs,  # Consider full reference trajectory per env.
    )
    self.reference_qpos = qpos[: total_frames + 1]
    center = self.reference_qpos[:, :2].mean(axis=0)
    self.camera = _viz.Camera(lookat=(center[0], center[1], 0.65), distance=3.5, azimuth=140.0, elevation=-22.0)
    self.init_qpos = np.tile(qpos[0], (args.num_envs, 1))
    self.qvel0 = np.tile(qvel[0], (args.num_envs, 1))
    self.control = wp.array(control[:total_frames], dtype=float)
    self.reference = wp.array(self.reference_qpos, dtype=float)
    self.num_joints = mjm.nq - _JOINT_QPOS_START

    self.armature_scale = wp.array(
      _initial_armature_scale(args.num_envs, args.spread, args.env_sampling, args.seed),
      dtype=float,
      requires_grad=True,
    )
    self.params = [self.armature_scale]
    self.base_armature = wp.array(mjm.dof_armature.astype(np.float32), dtype=float)
    armature = np.tile(mjm.dof_armature, (args.num_envs, 1)).astype(np.float32)
    self.m.dof_armature = wp.array(armature, dtype=float, requires_grad=True)
    self.model_params = [self.m.dof_armature]

    repeated = np.repeat(self.reference_qpos[:-1], self.repeat, axis=0)
    self.ghost_qpos = np.concatenate((repeated, self.reference_qpos[-1:]))[: total_steps + 1]
    self.project()

  def prepare_step(self, d):
    wp.launch(
      _set_armature,
      dim=(self.args.num_envs, self.m.nv),
      inputs=[self.armature_scale, self.base_armature],
      outputs=[self.m.dof_armature],
    )
    wp.launch(
      _set_ctrl,
      dim=(self.args.num_envs, self.m.nu),
      inputs=[self.m.nu, self.control, self.step_index, self.repeat],
      outputs=[d.ctrl],
    )

  def step_loss(self, d, d_out):
    del d
    wp.launch(
      _loss,
      dim=(self.args.num_envs, self.num_joints),
      inputs=[d_out.qpos, self.reference, self.step_index, self.repeat, _JOINT_QPOS_START, self.loss_scale],
      outputs=[self.loss],
    )

  def project(self):
    wp.launch(
      _clamp_armature_scale,
      dim=self.args.num_envs,
      inputs=[_ARMATURE_SCALE_MIN, _ARMATURE_SCALE_MAX],
      outputs=[self.armature_scale],
    )
    wp.launch(
      _set_armature,
      dim=(self.args.num_envs, self.m.nv),
      inputs=[self.armature_scale, self.base_armature],
      outputs=[self.m.dof_armature],
    )
    mjw.set_const_0(self.m, self.datas[0])

  def viz_mjm(self):
    return _viz.ghost_model(_SCENE, _ROBOT)

  def viz_trajectory(self, trajectory):
    return _viz.ghost_trajectory(trajectory, self.ghost_qpos)

if __name__ == "__main__":
  _demo.run(SysidG1)
