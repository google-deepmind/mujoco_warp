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
"""Optimize open-loop controls for dm_control suite tasks."""

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import _demo
import dm_suite_tasks as tasks
import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw


@dataclass(frozen=True)
class TaskCfg:
  name: str
  assets: dict[str, str]
  indices: dict[str, int]
  weights: dict[str, float]
  loss_type: _demo.LossType = _demo.LossType.PER_STEP
  init_ctrl_noise: float = 0.0
  init_qpos: tuple[float, ...] | None = None
  init_qpos_range: tuple[int, float, float] | None = None
  target: tuple[float, float, float] | None = None
  sites: tuple[str, ...] = ()


class Task:
  """Runtime task state and loss dispatch."""

  def __init__(self, cfg: TaskCfg, mjm: mujoco.MjModel, sim_dt: float, target_body: int, init_qpos: np.ndarray):
    self.cfg = cfg
    self.sim_dt = sim_dt
    self.target_body = target_body
    site_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, name) for name in cfg.sites]
    if any(site_id < 0 for site_id in site_ids):
      raise ValueError(f"{cfg.name} model is missing a configured site")
    self.site_ids = wp.array(site_ids, dtype=int) if site_ids else None
    self.init_qpos = wp.array(init_qpos, dtype=float)

  @property
  def uses_sites(self):
    return self.site_ids is not None

  def loss(self, m, d, d_out, step, horizon, loss_scale, loss):
    tasks._LOSS[self.cfg.name](self, m, d, d_out, step, horizon, loss_scale, loss)


TASKS = {name: TaskCfg(name=name, **kwargs["_cfg"]) for name, kwargs in tasks.TASK_KWARGS.items()}


@dataclass
class DmSuiteArgs(_demo.Args):
  sim_dt: float | None = None
  horizon: int | None = None
  iterations: int | None = None
  lr: float | None = None
  num_envs: int | None = None
  env_sampling: str | None = None
  viz_stride: int | None = None
  spread: float | None = None

  task: str = field(default="cartpole", metadata={"help": "task", "choices": tuple(TASKS)})


def _task_args(args: DmSuiteArgs) -> DmSuiteArgs:
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
  controls: wp.array[float],
  step: wp.array[int],
  ctrl_steps: int,
  # Data out:
  ctrl_out: wp.array2d[float],
):
  worldid, actuator = wp.tid()
  action = step[0] // ctrl_steps
  ctrl_out[worldid, actuator] = controls[(action * ctrl_out.shape[0] + worldid) * nu + actuator]


@wp.kernel(enable_backward=False)
def _clamp_ctrl(
  # Out:
  controls_out: wp.array[float],
):
  index = wp.tid()
  controls_out[index] = wp.clamp(controls_out[index], -1.0, 1.0)


def _targets(cfg: TaskCfg, num_envs: int, sampling: str, seed: int):
  if cfg.target is None:
    return np.zeros((num_envs, 2), dtype=np.float32)
  radius, low, high = cfg.target
  if num_envs > 1:
    angles = _demo.sample_range(num_envs, low, high, sampling, seed)
  else:
    angles = np.array([(low + high) / 2.0])
  return np.stack((radius * np.cos(angles), radius * np.sin(angles)), axis=1).astype(np.float32)


def _initial_controls(cfg: TaskCfg, num_controls: int, num_envs: int, nu: int, seed: int):
  rng = np.random.default_rng(seed)
  controls = rng.normal(0.0, cfg.init_ctrl_noise, (num_controls, num_envs, nu))
  return controls.reshape(-1)


def progress(args, model, data):
  cfg = TASKS[args.task]
  return tasks._PROGRESS[cfg.name](cfg, model, data)


class DmSuite(_demo.Demo):
  Args = DmSuiteArgs
  demo_type = _demo.DemoType.TRAJ_OPT
  loss_type = _demo.LossType.PER_STEP
  model_fields = ("body_pos",)

  def __init__(self, args: DmSuiteArgs):
    preset = tasks.TASK_KWARGS[args.task]
    args = _task_args(args)
    cfg = TASKS[args.task]
    if args.horizon < 1:
      raise ValueError("horizon must be positive")
    self.name = args.task
    self.layout = preset["layout"]
    self.camera = preset["camera"]
    self.ctrl_steps = _demo.control_steps(args.sim_dt, args.ctrl_dt)
    num_controls = (args.horizon + self.ctrl_steps - 1) // self.ctrl_steps
    targets = _targets(cfg, args.num_envs, args.env_sampling, args.seed)
    marker = targets[0] if cfg.target is not None else None
    mjm = mujoco.MjModel.from_xml_string(tasks.model_xml(cfg.assets, marker))
    self.loss_type = cfg.loss_type
    super().__init__(args, mjm, args.horizon)
    target = -1
    if cfg.target is not None:
      target = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "target")
      body_pos = np.tile(mjm.body_pos, (args.num_envs, 1, 1)).astype(np.float32)
      body_pos[:, target, :2] = targets
      self.m.body_pos = wp.array(body_pos, dtype=wp.vec3)

    self.init_qpos = self._initial_qpos(cfg)
    self.task = Task(cfg, mjm, args.sim_dt, target, self.init_qpos)
    if self.task.uses_sites:
      # Site costs differentiate analytically through mjw.fwd_kinematics into qpos.
      for data in self.datas:
        data.site_xpos.requires_grad = True
    controls = _initial_controls(cfg, num_controls, args.num_envs, mjm.nu, args.seed)
    self.controls = wp.array(controls, dtype=float, requires_grad=True)
    self.params = [self.controls]

  def _initial_qpos(self, cfg):
    if cfg.init_qpos is not None:
      qpos = np.asarray(cfg.init_qpos)
    else:
      qpos = self.mjm.qpos0

    qpos = np.tile(qpos, (self.args.num_envs, 1)).astype(np.float32)
    if cfg.init_qpos_range and self.args.num_envs > 1:
      index, low, high = cfg.init_qpos_range
      low = high - self.args.spread * (high - low)
      qpos[:, index] = _demo.sample_range(
        self.args.num_envs,
        low,
        high,
        self.args.env_sampling,
        self.args.seed,
      )
    return qpos

  def prepare_step(self, d):
    wp.launch(
      _set_ctrl,
      dim=(self.args.num_envs, self.mjm.nu),
      inputs=[self.mjm.nu, self.controls, self.step_index, self.ctrl_steps],
      outputs=[d.ctrl],
    )

  def step_loss(self, d, d_out):
    if self.task.uses_sites:
      mjw.fwd_kinematics(self.m, d_out)
    self.task.loss(self.m, d, d_out, self.step_index, self.horizon, self.loss_scale, self.loss)

  def project(self):
    wp.launch(_clamp_ctrl, dim=self.controls.shape, outputs=[self.controls])


if __name__ == "__main__":
  _demo.run(DmSuite)
