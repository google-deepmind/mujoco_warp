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
"""Task presets and losses for the Go1 trajectory demo."""

import _demo
import _viz
import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src import math as mjmath

_FEET = ("FR", "FL", "RR", "RL")
_JUMP_TAKEOFF_WINDOW = (0.40, 0.50)
_JUMP_LAND_TIME = 1.20
_JUMP_LAND_MARGIN = 0.10
_GALLOP_PHASES = np.array((0.05, 0.0, 0.35, 0.4))
_GALLOP_DUTY = 0.3
_GALLOP_CADENCE = 3.5
_GALLOP_HEIGHT = 0.10
_GALLOP_TARGET_DISTANCE = 1.0
_GALLOP_TARGET_SPEED = 1.0
_GALLOP_RAIBERT_REACH = 0.045
_GALLOP_RAIBERT_GAIN = 0.20
_SPIN_RAIBERT_REACH = 0.04
_SPIN_RAIBERT_GAIN = 0.05


TASK_KWARGS = {
  "jump": {
    "horizon": 400,
    "iterations": 160,
    "lr": 0.01,
    "num_envs": 4,
    "tbptt": None,
    "spread": 0.03,
    "layout": _viz.Layout(columns=1, spacing=(2.0, 1.4)),
    "camera": _viz.Camera(lookat=(0.0, 0.0, 0.35), distance=4.0, azimuth=135.0, elevation=-14.0),
    "_cfg": {
      "scene": "scene_flat.xml",
      "takeoff_speed": 1.5,
      "height": 0.35,
      "weights": {
        "vertical_velocity": 100.0,
        "landing": 200.0,
        "velocity": 20.0,
        "control": 2e-3,
      },
    },
  },
  "gallop": {
    "horizon": 300,
    "iterations": 160,
    "lr": 0.03,
    "num_envs": 64,
    "tbptt": None,
    "spread": 0.03,
    "layout": _viz.Layout(columns=8, spacing=2.0),
    "camera": _viz.Camera(lookat=(0.8, 0.0, 0.3), distance=4.0, azimuth=135.0, elevation=-18.0),
    "_cfg": {
      "scene": "scene_flat.xml",
      "speed": 1.2,
      "height": 0.30,
      "weights": {
        "velocity": 1.0,
        "height": 1.0,
        "upright": 1.0,
        "yaw": 0.3,
        "angular": 1.0,
        "gait": 40.0,
        "slip": 3.0,
        "raibert": 8.0,
        "control": 2e-3,
      },
    },
  },
  "handstand": {
    "horizon": 200,
    "iterations": 160,
    "lr": 0.03,
    "num_envs": 64,
    "tbptt": None,
    "spread": 0.12,
    "layout": _viz.Layout(columns=8, spacing=2.0),
    "camera": _viz.Camera(lookat=(0.1, 0.0, 0.35), distance=2.3, azimuth=135.0, elevation=-8.0),
    "_cfg": {
      "scene": "scene_flat.xml",
      "height": 0.25,
      "weights": {"orientation": 2.0, "head": 8.0, "balance": 0.02, "control": 2e-3},
    },
  },
  "spin": {
    "horizon": 600,
    "iterations": 160,
    "lr": 0.003,
    "num_envs": 64,
    "tbptt": 200,
    "spread": 0.03,
    "layout": _viz.Layout(columns=8, spacing=1.5),
    "camera": _viz.Camera(lookat=(0.0, 0.0, 0.3), distance=2.3, azimuth=135.0, elevation=-25.0),
    "_cfg": {
      "scene": "scene_flat.xml",
      "speed": 4.5,
      "weights": {
        "rotation": 1.0,
        "linear": 2.0,
        "vertical": 20.0,
        "angular": 5.0,
        "height": 500.0,
        "upright": 500.0,
        "posture": 0.2,
        "raibert": 100.0,
        "control": 2e-3,
        "smooth": 20.0,
      },
    },
  },
}


def _initial_controls(cfg, home, spread, num_envs, seed, num_controls, ctrl_dt, env_sampling="random"):
  if cfg.name == "spin":
    initial = _demo.sample_range(num_envs, home - spread, home + spread, env_sampling, seed)
    rng = np.random.default_rng(seed)
    phase = np.arange(num_controls) * 8.0 * np.pi * ctrl_dt
    controls = np.tile(home, (num_controls, num_envs, 1))
    if env_sampling == "random":
      scale = rng.uniform(0.9, 1.1, (1, num_envs, 1))
    elif env_sampling == "linspace":
      scale = np.linspace(0.9, 1.1, num_envs)[None, :, None]
    else:
      raise ValueError(f"unknown environment sampling: {env_sampling}")
    scale *= 0.80
    controls[:, :, (0, 2, 4, 6)] -= 0.055 * scale * np.sin(phase)[:, None, None]
    controls[:, :, (1, 3, 5, 7)] += 0.29 * scale * np.sin(phase + 1.5 * np.pi)[:, None, None] * np.array((-1, -1, -1, 1))
    controls[:, :, 8:12] += 0.41 * scale * np.sin(phase + 0.5 * np.pi)[:, None, None] * np.array((-1, -1, 1, 1))
    controls += (initial - home)[None]
    controls = np.clip(controls, home - 1.2, home + 1.2)
    return controls.astype(np.float32).reshape(-1)
  else:
    # Builds a near-home control sequence for each optimization world.
    initial = _demo.sample_range(num_envs, home - spread, home + spread, env_sampling, seed)
    return np.repeat(initial[None], num_controls, axis=0).reshape(-1)


def _foot_step_heights(time, phases, duty, cadence, height):
  phase = time * 2.0 * np.pi * cadence + np.pi
  angle = (phase + np.pi - 2.0 * np.pi * phases) % (2.0 * np.pi) - np.pi
  angle *= 0.5 / (1.0 - duty)
  value = np.abs(np.cos(np.clip(angle, -np.pi / 2.0, np.pi / 2.0)))
  return height * np.where(value >= 1e-6, value, 0.0)


def _smooth_transition(time, start, stop, initial, final):
  progress = np.clip((time - start) / (stop - start), 0.0, 1.0)
  blend = progress * progress * (3.0 - 2.0 * progress)
  return initial + blend * (final - initial)


def _yaw_turns(trajectory):
  qw, qx, qy, qz = np.moveaxis(trajectory[:, :, 3:7], -1, 0)
  yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
  yaw = np.unwrap(yaw, axis=1)
  return (yaw[:, -1] - yaw[:, 0]) / (2.0 * np.pi)


def _jump_weights(horizon, timestep):
  """Builds the takeoff and post-landing loss windows."""
  time = (np.arange(horizon) + 1) * timestep
  takeoff = ((time >= _JUMP_TAKEOFF_WINDOW[0]) & (time <= _JUMP_TAKEOFF_WINDOW[1])).astype(np.float32)
  landing = _smooth_transition(time, _JUMP_LAND_TIME, _JUMP_LAND_TIME + 0.2, 0.0, 1.0)
  return takeoff, landing


def _foot_ids(mjm):
  ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, name) for name in _FEET]
  if any(siteid < 0 for siteid in ids):
    raise ValueError("Go1 foot sites are missing")
  return np.asarray(ids, dtype=np.int32)


def setup_gallop(task, mjm, init_qpos):
  feet_ids = _foot_ids(mjm)
  mjd = mujoco.MjData(mjm)
  mjd.qpos[:] = init_qpos
  mujoco.mj_forward(mjm, mjd)
  feet = mjd.site_xpos[feet_ids]
  heights = np.array(
    [
      _foot_step_heights(
        (step + 1) * task.sim_dt,
        _GALLOP_PHASES,
        _GALLOP_DUTY,
        _GALLOP_CADENCE,
        _GALLOP_HEIGHT,
      )
      for step in range(task.horizon)
    ]
  )
  stance = 1.0 - np.minimum(1.0, heights / _GALLOP_HEIGHT)
  reach = np.zeros_like(heights)
  reach[1:] = (heights[1:] > 0.01) & (heights[1:] <= heights[:-1])

  task.foot_ids = wp.array(feet_ids, dtype=int)
  task.foot_heights = wp.array(heights + feet[:, 2], dtype=float)
  task.stance = wp.array(stance, dtype=float)
  task.reach = wp.array(reach, dtype=float)
  task.foot_offsets = wp.array(feet[:, 0] - init_qpos[0], dtype=float)
  task.raibert_reach = _GALLOP_RAIBERT_REACH
  task.raibert_gain = _GALLOP_RAIBERT_GAIN


def setup_jump(task, _mjm, init_qpos):
  takeoff, landing = _jump_weights(task.horizon, task.sim_dt)
  task.reference_qpos = wp.array(init_qpos, dtype=float)
  task.takeoff = wp.array(takeoff, dtype=float)
  task.landing = wp.array(landing, dtype=float)
  task.base_height = float(init_qpos[2])


def setup_spin(task, mjm, init_qpos):
  foot_ids = _foot_ids(mjm)
  mjd = mujoco.MjData(mjm)
  mjd.qpos[:] = init_qpos
  mujoco.mj_forward(mjm, mjd)
  feet = mjd.site_xpos[foot_ids]

  task.reference_qpos = wp.array(init_qpos[7:], dtype=float)
  task.reference_height = float(init_qpos[2])
  task.foot_ids = wp.array(foot_ids, dtype=int)
  task.foot_offsets = wp.array(feet[:, :2] - init_qpos[:2], dtype=float)


@wp.func
def _foot_slip(current: wp.vec3, previous: wp.vec3, inv_timestep: float) -> float:
  velocity = (current - previous) * inv_timestep
  return velocity[0] * velocity[0] + velocity[1] * velocity[1]


@wp.kernel
def _gallop_loss(
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  site_xpos_in: wp.array2d[wp.vec3],
  # In:
  previous_site_xpos_in: wp.array2d[wp.vec3],
  foot_ids: wp.array[int],
  foot_heights: wp.array2d[float],
  stance_in: wp.array2d[float],
  reach_in: wp.array2d[float],
  foot_offsets: wp.array[float],
  target_speed: float,
  target_height: float,
  raibert_reach: float,
  raibert_gain: float,
  timestep: float,
  weight_velocity: float,
  weight_height: float,
  weight_upright: float,
  weight_yaw: float,
  weight_angular: float,
  weight_gait: float,
  weight_slip: float,
  weight_raibert: float,
  step: wp.array[int],
  horizon: int,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  stepid = step[0]
  speed = target_speed * wp.min(1.0, float(stepid + 1) / (0.3 * float(horizon)))
  qw = qpos_in[worldid, 3]
  qx = qpos_in[worldid, 4]
  qy = qpos_in[worldid, 5]
  qz = qpos_in[worldid, 6]
  orientation = wp.quat(qw, qx, qy, qz)
  velocity = mjmath.rot_vec_quat(
    wp.vec3(qvel_in[worldid, 0], qvel_in[worldid, 1], qvel_in[worldid, 2]),
    mjmath.quat_inv(orientation),
  )
  forward_velocity = velocity[0]
  gait = float(0.0)
  slip = float(0.0)
  raibert = float(0.0)
  for foot in range(4):
    siteid = foot_ids[foot]
    position = site_xpos_in[worldid, siteid]
    height = position[2] - foot_heights[stepid, foot]
    gait += height * height
    slip += stance_in[stepid, foot] * _foot_slip(position, previous_site_xpos_in[worldid, siteid], 1.0 / timestep)
    target = foot_offsets[foot] + raibert_reach * forward_velocity + raibert_gain * (forward_velocity - speed)
    error = position[0] - qpos_in[worldid, 0] - target
    raibert += reach_in[stepid, foot] * error * error
  upright = 4.0 * (qx * qx + qy * qy)
  yaw = wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
  loss = weight_velocity * ((forward_velocity - speed) ** 2.0 + velocity[1] ** 2.0)
  loss += weight_height * (qpos_in[worldid, 2] - target_height) ** 2.0
  loss += weight_upright * upright + weight_yaw * yaw * yaw + weight_angular * qvel_in[worldid, 5] ** 2.0
  loss += weight_gait * gait + weight_slip * slip + weight_raibert * raibert
  wp.atomic_add(loss_out, 0, loss_scale * loss)


@wp.kernel
def _jump_loss(
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  # In:
  reference_qpos: wp.array[float],
  target_speed: float,
  takeoff_in: wp.array[float],
  landing_in: wp.array[float],
  weight_takeoff: float,
  weight_landing: float,
  weight_velocity: float,
  step: wp.array[int],
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  stepid = step[0]
  takeoff = takeoff_in[stepid]
  landing = landing_in[stepid]

  landing_state = float(0.0)
  velocity = float(0.0)
  for coordinate in range(19):
    landing_error = qpos_in[worldid, coordinate] - reference_qpos[coordinate]
    landing_state += landing_error * landing_error
  for dof in range(18):
    velocity += qvel_in[worldid, dof] * qvel_in[worldid, dof]
  takeoff_speed_error = qvel_in[worldid, 2] - target_speed
  loss = takeoff * weight_takeoff * takeoff_speed_error * takeoff_speed_error
  loss += landing * (weight_landing * landing_state + weight_velocity * velocity)
  wp.atomic_add(loss_out, 0, loss_scale * loss)


@wp.kernel
def _handstand_loss(
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  # In:
  target_height: float,
  weight_orientation: float,
  weight_head: float,
  weight_balance: float,
  step: wp.array[int],
  horizon: int,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  qw = qpos_in[worldid, 3]
  qx = qpos_in[worldid, 4]
  qy = qpos_in[worldid, 5]
  qz = qpos_in[worldid, 6]
  alignment = 0.70710678 * (qw + qy)
  forward_z = 2.0 * (qx * qz - qw * qy)
  below = wp.max(target_height - qpos_in[worldid, 2] - 0.3 * forward_z, 0.0)
  omega = qvel_in[worldid, 3] ** 2.0 + qvel_in[worldid, 4] ** 2.0 + qvel_in[worldid, 5] ** 2.0
  progress = float(step[0] + 1) / float(horizon)
  loss = weight_orientation * (1.0 - alignment * alignment) + weight_head * below * below + weight_balance * omega
  wp.atomic_add(loss_out, 0, loss_scale * progress * loss)


@wp.kernel
def _spin_loss(
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  site_xpos_in: wp.array2d[wp.vec3],
  # In:
  reference_qpos: wp.array[float],
  reference_height: float,
  foot_ids: wp.array[int],
  foot_offsets: wp.array2d[float],
  target_speed: float,
  raibert_reach: float,
  raibert_gain: float,
  weight_rotation: float,
  weight_linear: float,
  weight_vertical: float,
  weight_angular: float,
  weight_height: float,
  weight_upright: float,
  weight_posture: float,
  weight_raibert: float,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  qw = qpos_in[worldid, 3]
  qx = qpos_in[worldid, 4]
  qy = qpos_in[worldid, 5]
  qz = qpos_in[worldid, 6]
  wx = qvel_in[worldid, 3]
  wy = qvel_in[worldid, 4]
  wz = qvel_in[worldid, 5]

  row_z = wp.vec3(
    2.0 * (qx * qz - qw * qy),
    2.0 * (qy * qz + qw * qx),
    1.0 - 2.0 * (qx * qx + qy * qy),
  )
  yaw_rate = wp.dot(row_z, wp.vec3(wx, wy, wz))
  angular_xy = wp.max(wx * wx + wy * wy + wz * wz - yaw_rate * yaw_rate, 0.0)

  posture = float(0.0)
  for joint in range(12):
    error = qpos_in[worldid, 7 + joint] - reference_qpos[joint]
    posture += error * error

  r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
  r01 = 2.0 * (qx * qy - qw * qz)
  r10 = 2.0 * (qx * qy + qw * qz)
  r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
  angle = raibert_reach * yaw_rate + raibert_gain * (yaw_rate - target_speed)
  cos_angle = wp.cos(angle)
  sin_angle = wp.sin(angle)
  raibert = float(0.0)
  for foot in range(4):
    offset_x = foot_offsets[foot, 0]
    offset_y = foot_offsets[foot, 1]
    nominal_x = r00 * offset_x + r01 * offset_y
    nominal_y = r10 * offset_x + r11 * offset_y
    target_x = cos_angle * nominal_x - sin_angle * nominal_y
    target_y = sin_angle * nominal_x + cos_angle * nominal_y
    position = site_xpos_in[worldid, foot_ids[foot]]
    error_x = position[0] - qpos_in[worldid, 0] - target_x
    error_y = position[1] - qpos_in[worldid, 1] - target_y
    raibert += error_x * error_x + error_y * error_y

  height = qpos_in[worldid, 2] - reference_height
  upright = row_z[0] * row_z[0] + row_z[1] * row_z[1] + (row_z[2] - 1.0) * (row_z[2] - 1.0)
  loss = weight_rotation * (yaw_rate - target_speed) ** 2.0
  loss += weight_linear * (qvel_in[worldid, 0] ** 2.0 + qvel_in[worldid, 1] ** 2.0)
  loss += weight_vertical * qvel_in[worldid, 2] ** 2.0
  loss += weight_angular * angular_xy + weight_posture * wp.sqrt(posture)
  loss += weight_height * height * height + weight_upright * upright + weight_raibert * raibert
  wp.atomic_add(loss_out, 0, loss_scale * loss)


@wp.kernel
def _control_loss(
  # Data in:
  ctrl_in: wp.array2d[float],
  # In:
  controls: wp.array[float],
  home: wp.array[float],
  step: wp.array[int],
  ctrl_steps: int,
  weight: float,
  weight_smooth: float,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  loss = float(0.0)
  for actuator in range(12):
    error = ctrl_in[worldid, actuator] - home[actuator]
    loss += weight * error * error
    if step[0] >= ctrl_steps and step[0] % ctrl_steps == 0:
      control_step = step[0] // ctrl_steps
      index = (control_step * ctrl_in.shape[0] + worldid) * ctrl_in.shape[1] + actuator
      previous = ((control_step - 1) * ctrl_in.shape[0] + worldid) * ctrl_in.shape[1]
      delta = controls[index] - controls[previous + actuator]
      loss += weight_smooth * delta * delta
  wp.atomic_add(loss_out, 0, loss_scale * loss)


def gallop_loss(task, d, d_out, step, horizon, loss_scale, loss):
  weights = task.cfg.weights
  wp.launch(
    _gallop_loss,
    dim=d_out.qpos.shape[0],
    inputs=[
      d_out.qpos,
      d_out.qvel,
      d_out.site_xpos,
      d.site_xpos,
      task.foot_ids,
      task.foot_heights,
      task.stance,
      task.reach,
      task.foot_offsets,
      task.cfg.speed,
      task.cfg.height,
      task.raibert_reach,
      task.raibert_gain,
      task.sim_dt,
      weights["velocity"],
      weights["height"],
      weights["upright"],
      weights["yaw"],
      weights["angular"],
      weights["gait"],
      weights["slip"],
      weights["raibert"],
      step,
      horizon,
      loss_scale,
    ],
    outputs=[loss],
  )


def jump_loss(task, d, d_out, step, horizon, loss_scale, loss):
  del d, horizon
  weights = task.cfg.weights
  wp.launch(
    _jump_loss,
    dim=d_out.qpos.shape[0],
    inputs=[
      d_out.qpos,
      d_out.qvel,
      task.reference_qpos,
      task.cfg.takeoff_speed,
      task.takeoff,
      task.landing,
      weights["vertical_velocity"],
      weights["landing"],
      weights["velocity"],
      step,
      loss_scale,
    ],
    outputs=[loss],
  )


def handstand_loss(task, d, d_out, step, horizon, loss_scale, loss):
  del d
  weights = task.cfg.weights
  wp.launch(
    _handstand_loss,
    dim=d_out.qpos.shape[0],
    inputs=[
      d_out.qpos,
      d_out.qvel,
      task.cfg.height,
      weights["orientation"],
      weights["head"],
      weights["balance"],
      step,
      horizon,
      loss_scale,
    ],
    outputs=[loss],
  )


def spin_loss(task, d, d_out, step, horizon, loss_scale, loss):
  del d, step, horizon
  weights = task.cfg.weights
  wp.launch(
    _spin_loss,
    dim=d_out.qpos.shape[0],
    inputs=[
      d_out.qpos,
      d_out.qvel,
      d_out.site_xpos,
      task.reference_qpos,
      task.reference_height,
      task.foot_ids,
      task.foot_offsets,
      task.cfg.speed,
      _SPIN_RAIBERT_REACH,
      _SPIN_RAIBERT_GAIN,
      weights["rotation"],
      weights["linear"],
      weights["vertical"],
      weights["angular"],
      weights["height"],
      weights["upright"],
      weights["posture"],
      weights["raibert"],
      loss_scale,
    ],
    outputs=[loss],
  )


def regularize(task, controls, ctrl, step, ctrl_steps, loss_scale, loss):
  wp.launch(
    _control_loss,
    dim=ctrl.shape[0],
    inputs=[
      ctrl,
      controls,
      task.home,
      step,
      ctrl_steps,
      task.cfg.weights.get("control", 0.0),
      task.cfg.weights.get("smooth", 0.0),
      loss_scale,
    ],
    outputs=[loss],
  )


def gallop_progress(cfg, sim_dt, model, data):
  del cfg, sim_dt, model
  trajectory = data.qpos
  translation = trajectory[:, -1, 0] - trajectory[:, 0, 0]
  translation_error = (translation - _GALLOP_TARGET_DISTANCE) / _GALLOP_TARGET_DISTANCE
  speed_error = (data.qvel[:, -1, 0] - _GALLOP_TARGET_SPEED) / _GALLOP_TARGET_SPEED
  normalized_error = translation_error**2.0 + speed_error**2.0
  return 1.0 / (1.0 + normalized_error) ** 2.0


def jump_progress(cfg, sim_dt, model, data):
  trajectory = data.qpos
  initial = trajectory[:, 0]
  final = trajectory[:, -1]
  jump_height = cfg.height - initial[:, 2]
  peak = trajectory[:, :, 2].max(axis=1)
  jump_error = np.maximum(cfg.height - peak, 0.0) / jump_height
  landing_distance = np.linalg.norm(final[:, :3] - initial[:, :3], axis=1)
  landing_position = np.maximum(landing_distance - _JUMP_LAND_MARGIN, 0.0) / jump_height
  landing_velocity = np.linalg.norm(data.qvel[:, -1, :3], axis=1) / cfg.takeoff_speed
  upright_error = 2.0 * (final[:, 4] ** 2.0 + final[:, 5] ** 2.0)
  normalized_error = jump_error**2.0 + landing_position**2.0 + landing_velocity**2.0 + upright_error**2.0
  return 1.0 / (1.0 + normalized_error) ** 2.0


def handstand_progress(cfg, sim_dt, model, data):
  del cfg, sim_dt, model
  quat = data.qpos[:, -1, 3:7]
  target_alignment = (quat[:, 0] + quat[:, 2]) / np.sqrt(2.0)
  return np.clip(target_alignment**2.0, 0.0, 1.0)


def spin_progress(cfg, sim_dt, model, data):
  del cfg, sim_dt, model
  # One 360-degree turn completes the task; additional rotation remains at 1.
  return np.clip(_yaw_turns(data.qpos), 0.0, 1.0)


_SETUP = {
  "jump": setup_jump,
  "gallop": setup_gallop,
  "spin": setup_spin,
}
_LOSS = {
  "jump": jump_loss,
  "gallop": gallop_loss,
  "handstand": handstand_loss,
  "spin": spin_loss,
}
_PROGRESS = {
  "jump": jump_progress,
  "gallop": gallop_progress,
  "handstand": handstand_progress,
  "spin": spin_progress,
}
