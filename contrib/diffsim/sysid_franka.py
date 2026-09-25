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
"""Identify Franka per-link inertial scales from a joint-space excitation."""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import _demo
import _inertia
import _viz
import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

_FRANKA = Path(__file__).resolve().parents[2] / "benchmarks" / "franka_fr3"
_SCENE = _FRANKA / "scene_sysid.xml"
_ROBOT = _FRANKA / "fr3.xml"
_LINKS = tuple(f"fr3_link{i}" for i in range(1, 7))
_NUM_JOINTS = 7
_KP = (40.0, 30.0, 50.0, 50.0, 35.0, 25.0, 10.0)
_KV = (4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0)

# Bend the shoulder to separate the proximal joint axes without large excursions.
_SHOULDER_HOME = -0.75
_ELBOW_HOME = -2.0
_CHIRP_FREQUENCIES = (0.1, 3.0)
_CHIRP_JOINT_AMPLITUDES = (0.5, 0.5, 0.25, 0.5, 0.25, 0.25, 0.5)
# Relative durations: ramp up, peak, ramp down, home.
_CHIRP_PHASES = np.array((0.2, 0.48333333333333334, 0.3, 0.016666666666666666))


@dataclass
class SysidFrankaArgs(_demo.Args):
  sim_dt: float = 0.005
  iterations: int = 400
  lr: float = 0.05
  num_envs: int = 64
  seed: int = 0
  tbptt: int | None = 200
  viz_every: int = 50

  horizon: int = 600
  eval_horizon: int = field(default=600, metadata={"help": "forward-only excitation steps after the fitted rollout"})
  spread: float = field(default=0.9, metadata={"help": "initial log-inertial-scale half-width"})


@wp.kernel
def _set_ctrl(
  # In:
  control: wp.array2d[float],
  step: wp.array[int],
  # Data out:
  ctrl_out: wp.array2d[float],
):
  worldid, actuatorid = wp.tid()
  ctrl_out[worldid, actuatorid] = control[step[0], actuatorid]


@wp.kernel
def _loss(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  reference: wp.array2d[float],
  step: wp.array[int],
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid, jointid = wp.tid()
  error = qpos_in[worldid, jointid] - reference[step[0] + 1, jointid]
  wp.atomic_add(loss_out, 0, loss_scale * error * error)


def _home_data(model):
  data = mujoco.MjData(model)
  # Refresh derived constants after configuring the model.
  mujoco.mj_setConst(model, data)
  if model.nkey:
    mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  return data


def _phase_steps(horizon):
  """Allocates phase lengths on the simulation grid without cumulative rounding drift."""
  ends = np.rint(np.cumsum(_CHIRP_PHASES) * horizon).astype(int)
  ends[-1] = horizon
  return np.diff(np.concatenate(([0], ends)))


def _chirp(step, duration, frequencies, ramp_up, ramp_down):
  """Linear sine sweep in step units, with frequencies in cycles per step."""
  low, high = frequencies
  phase = 2.0 * np.pi * (low * step + 0.5 * (high - low) * step**2 / duration)
  envelope = np.ones_like(step, dtype=float)
  rising = step < ramp_up
  envelope[rising] = 0.5 * (1.0 - np.cos(np.pi * step[rising] / ramp_up))
  falling = (step > duration - ramp_down) & (step <= duration)
  remaining = duration - step[falling]
  envelope[falling] = 0.5 * (1.0 - np.cos(np.pi * remaining / ramp_down))
  envelope[step > duration] = 0.0
  return envelope * np.sin(phase)


def _excitation(model, horizon):
  if horizon < 2:
    raise ValueError("excitation horizon must be at least two")

  step = np.arange(horizon)
  ramp_up, _, ramp_down, home_steps = _phase_steps(horizon)
  frequencies = np.asarray(_CHIRP_FREQUENCIES) * model.opt.timestep
  limits = model.actuator_ctrlrange[:_NUM_JOINTS]
  home = model.key_ctrl[0, :_NUM_JOINTS] if model.nkey else limits.mean(axis=1)

  signal = _chirp(step, horizon - home_steps, frequencies, ramp_up, ramp_down)
  amplitude = np.asarray(_CHIRP_JOINT_AMPLITUDES)
  control = home + amplitude[None] * signal[:, None]
  return np.clip(control, limits[:, 0], limits[:, 1]).astype(np.float32)


def _reference_rollout(model, data, control):
  m = mjw.put_model(model)
  d, d_out = (mjw.put_data(model, data) for _ in range(2))
  qpos = [d.qpos.numpy()[0].copy()]
  for command in control:
    d.ctrl.assign(command[None])
    mjw.step(m, d, d_out)
    qpos.append(d_out.qpos.numpy()[0].copy())
    d, d_out = d_out, d
  return np.asarray(qpos, dtype=np.float32)


def _initial_parameters(target, num_envs, spread, sampling, seed):
  """Samples initial per-link log scales around the reference value."""
  if spread < 0.0:
    raise ValueError("spread must be nonnegative")
  target = np.asarray(target, dtype=np.float32)
  return _demo.sample_range(num_envs, target - spread, target + spread, sampling, seed)


def progress(args, model, data):
  scale = np.exp(model.inertial_scale)
  error = np.sqrt(np.mean((scale - 1.0) ** 2, axis=1))
  error_scale = 0.5  # map a 50% relative error to a raw progress of 0.5
  return 1.0 / (1.0 + (error / error_scale) ** 2.0)


class SysidFranka(_demo.Demo):
  Args = SysidFrankaArgs
  name = "sysid_franka"
  demo_type = _demo.DemoType.SYS_ID
  loss_type = _demo.LossType.PER_STEP
  model_fields = _inertia._MODEL_FIELDS
  layout = _viz.Layout(columns=8, spacing=1.6)
  camera = _viz.Camera(lookat=(0.35, 0.0, 0.4), distance=3.0, azimuth=110.0, elevation=-28.0)

  def __init__(self, args: SysidFrankaArgs):
    if args.horizon < 2:
      raise ValueError("horizon must be at least two")
    if args.eval_horizon < 0:
      raise ValueError("eval_horizon must be nonnegative")
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    _demo.set_sim_params(model, args.sim_dt, args.integrator, args.cone, args.impratio)
    model.actuator_gainprm[:, 0] = _KP
    model.actuator_biasprm[:, 1] = -np.asarray(_KP)
    model.actuator_biasprm[:, 2] = -np.asarray(_KV)
    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    model.body_gravcomp[:] = 1.0
    model.key_qpos[0, 1] = model.key_ctrl[0, 1] = _SHOULDER_HOME
    model.key_qpos[0, 3] = model.key_ctrl[0, 3] = _ELBOW_HOME
    home = _home_data(model)
    control = _excitation(model, args.horizon + args.eval_horizon)
    reference_qpos = _reference_rollout(model, home, control)
    super().__init__(
      args,
      model,
      args.horizon,
      eval_horizon=args.eval_horizon,
      loss_scale=1.0 / args.num_envs,  # Consider full reference trajectory per env.
    )

    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in _LINKS]
    body_link = np.full(model.nbody, -1, dtype=np.int32)
    body_link[body_ids] = np.arange(len(body_ids))
    base_mass = model.body_mass.astype(np.float32)
    base_inertia = model.body_inertia.astype(np.float32)
    target = np.zeros((len(body_ids), 1))

    initial = _initial_parameters(
      target,
      args.num_envs,
      args.spread,
      args.env_sampling,
      args.seed,
    )
    self.inertial_scale = wp.array(initial.reshape(-1), dtype=float, requires_grad=True)
    self.params = [self.inertial_scale]
    self.m.body_mass = wp.array(np.tile(base_mass, (args.num_envs, 1)), dtype=float, requires_grad=True)
    self.m.body_inertia = wp.array(np.tile(base_inertia, (args.num_envs, 1, 1)), dtype=wp.vec3, requires_grad=True)
    self.model_params = [self.m.body_mass, self.m.body_inertia]

    self.num_links = len(body_ids)
    self.body_link = wp.array(body_link, dtype=int)
    self.base_mass = wp.array(base_mass, dtype=float)
    self.base_inertia = wp.array(base_inertia, dtype=wp.vec3)
    self.control = wp.array(control, dtype=float)
    self.reference = wp.array(reference_qpos, dtype=float)
    self.reference_qpos = reference_qpos
    self.init_qpos = np.tile(home.qpos, (args.num_envs, 1)).astype(np.float32)
    self.qvel0 = np.tile(home.qvel, (args.num_envs, 1)).astype(np.float32)
    self.project()

  def prepare_step(self, d):
    _inertia.project(
      self.inertial_scale,
      self.num_links,
      self.body_link,
      self.base_mass,
      self.base_inertia,
      self.m.body_mass,
      self.m.body_inertia,
    )
    wp.launch(
      _set_ctrl,
      dim=(self.args.num_envs, self.m.nu),
      inputs=[self.control, self.step_index],
      outputs=[d.ctrl],
    )

  def project(self):
    _inertia.project(
      self.inertial_scale,
      self.num_links,
      self.body_link,
      self.base_mass,
      self.base_inertia,
      self.m.body_mass,
      self.m.body_inertia,
    )
    mjw.set_const(self.m, self.datas[0])

  def model_state(self):
    state = super().model_state()
    state.inertial_scale = self.inertial_scale.numpy().reshape(self.args.num_envs, self.num_links)
    return state

  def step_loss(self, d, d_out):
    del d
    wp.launch(
      _loss,
      dim=(self.args.num_envs, _NUM_JOINTS),
      inputs=[d_out.qpos, self.reference, self.step_index, self.loss_scale],
      outputs=[self.loss],
    )

  def viz_mjm(self):
    return _viz.ghost_model(_SCENE, _ROBOT)

  def viz_trajectory(self, trajectory):
    return _viz.ghost_trajectory(trajectory, self.reference_qpos)


if __name__ == "__main__":
  _demo.run(SysidFranka)
