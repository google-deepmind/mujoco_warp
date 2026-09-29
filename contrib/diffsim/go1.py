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
"""Optimize open-loop controls for Unitree Go1 motions."""

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path

import _demo
import go1_tasks as tasks
import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

_GO1 = Path(__file__).resolve().parents[2] / "benchmarks" / "unitree_go1"
_SETTLE_TIME = 0.8
_ACTION_LIMIT = 1.2


@dataclass(frozen=True)
class TaskCfg:
  name: str
  scene: str
  weights: dict[str, float]
  speed: float = 0.0
  takeoff_speed: float = 0.0
  height: float = 0.0


class Task:
  """Runtime task state and loss dispatch."""

  def __init__(self, cfg: TaskCfg, mjm: mujoco.MjModel, horizon: int, init_qpos, home):
    self.cfg = cfg
    self.sim_dt = float(mjm.opt.timestep)
    self.horizon = horizon
    self.home = wp.array(home, dtype=float)
    setup = tasks._SETUP.get(cfg.name)
    if setup:
      setup(self, mjm, init_qpos)

  @property
  def uses_sites(self):
    return self.cfg.name in ("gallop", "spin")

  def loss(self, data, data_out, controls, step, horizon, ctrl_steps, loss_scale, loss):
    tasks._LOSS[self.cfg.name](self, data, data_out, step, horizon, loss_scale, loss)
    tasks.regularize(self, controls, data.ctrl, step, ctrl_steps, loss_scale, loss)


TASKS = {name: TaskCfg(name=name, **kwargs["_cfg"]) for name, kwargs in tasks.TASK_KWARGS.items()}


@dataclass
class Go1Args(_demo.Args):
  sim_dt: float = 0.005
  integrator: str = "euler"
  cone: str = "pyramidal"
  impratio: float = 1.0
  ctrl_dt: float = field(default=0.01, metadata={"help": "control interval"})
  horizon: int | None = None
  iterations: int | None = None
  lr: float | None = None
  num_envs: int | None = None
  spread: float | None = None
  seed: int = 0

  task: str = field(default="gallop", metadata={"help": "motion", "choices": tuple(TASKS)})


def _task_args(args: Go1Args) -> Go1Args:
  values = {}
  for name, default in tasks.TASK_KWARGS[args.task].items():
    if name in ("_cfg", "layout", "camera"):
      continue
    value = getattr(args, name)
    values[name] = default if value is None else value
  return replace(args, **values)


@wp.kernel
def _set_ctrl(
  # Model:
  nu: int,
  # In:
  ctrl_steps: int,
  controls: wp.array[float],
  step: wp.array[int],
  # Data out:
  ctrl_out: wp.array2d[float],
):
  worldid, actuatorid = wp.tid()
  control_step = step[0] // ctrl_steps
  index = (control_step * ctrl_out.shape[0] + worldid) * nu + actuatorid
  ctrl_out[worldid, actuatorid] = controls[index]


@wp.kernel(enable_backward=False)
def _clamp_controls(
  # Model:
  nu: int,
  # In:
  home: wp.array[float],
  # Out:
  controls_out: wp.array[float],
):
  index = wp.tid()
  actuatorid = index % nu
  controls_out[index] = wp.clamp(
    controls_out[index],
    home[actuatorid] - _ACTION_LIMIT,
    home[actuatorid] + _ACTION_LIMIT,
  )


def progress(args, model, data):
  cfg = TASKS[args.task]
  return tasks._PROGRESS[cfg.name](cfg, args.sim_dt, model, data)


class Go1(_demo.Demo):
  Args = Go1Args
  demo_type = _demo.DemoType.TRAJ_OPT
  loss_type = _demo.LossType.PER_STEP
  trace_body = "trunk"
  data_kwargs = {"nconmax": 64, "njmax": 128}

  def __init__(self, args: Go1Args):
    preset = tasks.TASK_KWARGS[args.task]
    args = _task_args(args)
    cfg = TASKS[args.task]
    if args.num_envs < 1:
      raise ValueError("num_envs must be positive")
    if args.horizon < 1:
      raise ValueError("horizon must be positive")
    self.name = f"go1_{args.task}"
    self.layout = preset["layout"]
    self.camera = preset["camera"]
    self.trace_body = None if cfg.name == "spin" else "trunk"

    mjm = mujoco.MjModel.from_xml_path(str(_GO1 / cfg.scene))
    super().__init__(args, mjm, args.horizon)
    if mjm.nu != 12:
      raise ValueError(f"Go1 tasks require 12 actuators, got {mjm.nu}")

    self.ctrl_steps = _demo.control_steps(args.sim_dt, args.ctrl_dt)
    self.home, init_qpos = self._settled_state()
    self.init_qpos = np.repeat(init_qpos[None, :], args.num_envs, axis=0).astype(np.float32)
    self.task = Task(cfg, mjm, args.horizon, init_qpos, self.home)
    if self.task.uses_sites:
      for data in self.datas:
        data.site_xpos.requires_grad = True

    num_controls = (args.horizon + self.ctrl_steps - 1) // self.ctrl_steps
    initial = tasks._initial_controls(
      cfg,
      self.home,
      args.spread,
      args.num_envs,
      args.seed,
      num_controls,
      args.ctrl_dt,
      args.env_sampling,
    )
    self.controls = wp.array(initial, dtype=float, requires_grad=True)
    self.params = [self.controls]

  def _settled_state(self):
    key = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    if key < 0:
      raise ValueError("Go1 model is missing the init_state keyframe")
    home = self.mjm.key_ctrl[key].astype(np.float32)

    mjd = mujoco.MjData(self.mjm)
    mujoco.mj_resetDataKeyframe(self.mjm, mjd, key)
    mjd.ctrl[:] = home
    mujoco.mj_forward(self.mjm, mjd)
    current, output = (mjw.put_data(self.mjm, mjd) for _ in range(2))
    for _ in range(round(_SETTLE_TIME / self.mjm.opt.timestep)):
      mjw.step(self.m, current, output)
      current, output = output, current
    return home, current.qpos.numpy()[0]

  def prepare_step(self, data):
    if self.task.uses_sites:
      mjw.fwd_kinematics(self.m, data)
    wp.launch(
      _set_ctrl,
      dim=(self.args.num_envs, self.m.nu),
      inputs=[self.m.nu, self.ctrl_steps, self.controls, self.step_index],
      outputs=[data.ctrl],
    )

  def step_loss(self, data, data_out):
    if self.task.uses_sites:
      mjw.fwd_kinematics(self.m, data_out)
    self.task.loss(
      data,
      data_out,
      self.controls,
      self.step_index,
      self.horizon,
      self.ctrl_steps,
      self.loss_scale,
      self.loss,
    )

  def project(self):
    wp.launch(
      _clamp_controls,
      dim=self.controls.size,
      inputs=[self.m.nu, self.task.home],
      outputs=[self.controls],
    )


if __name__ == "__main__":
  _demo.run(Go1)
