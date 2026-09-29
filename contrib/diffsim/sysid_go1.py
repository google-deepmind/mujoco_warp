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
"""Identify Unitree Go1 joint friction and damping from a standing excitation."""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import _demo
import _viz
import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw

_GO1 = Path(__file__).resolve().parents[2] / "benchmarks" / "unitree_go1"
_SCENE = _GO1 / "scene_flat.xml"
_ROBOT = _GO1 / "unitree_go1_mjlab.xml"
_NUM_JOINTS = 12
_ELASTIC_BAND_POINT = (0.0, 0.0, 0.8)
_ELASTIC_BAND_CLEARANCE = 0.02
_ELASTIC_BAND_STIFFNESS = 200.0
_ELASTIC_BAND_DAMPING = 100.0
_GANTRY_SEGMENTS = 8
_GANTRY_RADIUS = 0.012
_GANTRY_RGBA = (0.48, 0.58, 0.70, 1.0)

_REFERENCE_DAMPING = 2.0
_REFERENCE_FRICTIONLOSS = 0.2
_DAMPING_MIN = 0.0
_DAMPING_MAX = 7.0
_FRICTIONLOSS_MIN = 1.0e-4
_FRICTIONLOSS_MAX = 0.5

_CHIRP_FREQUENCIES = (0.5, 2.0)
_CHIRP_RAMP_DURATION = 0.5
_CHIRP_NUM_HARMONICS = 5
_CHIRP_AMPLITUDES = {"hip": 0.1575, "thigh": 0.21, "calf": 0.42}
# A fixed stance adjustment keeps the rear calves clear during the excitation.
_REAR_CALF_OFFSET = 0.15
# Standing base pose: 0.305 m height, about 3.4 degrees pitch.
_HOME_BASE_QPOS = (0.0, 0.0, 0.305, 1.0, 0.0, 0.03, 0.0)


@dataclass
class SysidGo1Args(_demo.Args):
  """Go1 joint-friction and damping system-identification arguments."""

  sim_dt: float = 0.005
  iterations: int = 400
  lr: float = 0.03
  num_envs: int = 64
  tbptt: int | None = 100
  viz_every: int = 50

  eval_horizon: int = field(default=500, metadata={"help": "forward-only excitation steps; 500 is 2.5 s at 200 Hz"})
  horizon: int = 500
  spread: float = field(default=3.0, metadata={"help": "initial log-parameter half-width, capped at physical bounds"})
  gantry: bool = field(default=False, metadata={"help": "enable the unilateral safety gantry"})


@wp.kernel
def _set_joint_parameters(
  # In:
  dof_ids: wp.array[int],
  frictionloss_scale: wp.array[float],
  damping_scale: wp.array[float],
  frictionloss_unit: float,
  damping_unit: float,
  # Model out:
  dof_frictionloss_out: wp.array2d[float],
  dof_damping_out: wp.array2d[float],
):
  worldid, jointid = wp.tid()
  parameterid = worldid * dof_ids.shape[0] + jointid
  dofid = dof_ids[jointid]
  dof_frictionloss_out[worldid, dofid] = frictionloss_unit * frictionloss_scale[parameterid]
  dof_damping_out[worldid, dofid] = damping_unit * damping_scale[parameterid]


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
def _set_elastic_band_force(
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  # In:
  body_id: int,
  anchor: wp.vec3,
  rest_length: float,
  stiffness: float,
  damping: float,
  # Data out:
  xfrc_applied_out: wp.array2d[wp.spatial_vector],
):
  worldid = wp.tid()
  position = wp.vec3(qpos_in[worldid, 0], qpos_in[worldid, 1], qpos_in[worldid, 2])
  velocity = wp.vec3(qvel_in[worldid, 0], qvel_in[worldid, 1], qvel_in[worldid, 2])
  delta = anchor - position
  distance = wp.length(delta)
  direction = wp.vec3(0.0)
  magnitude = 0.0
  if distance > 1.0e-6:
    direction = delta / distance
  if distance > rest_length:
    radial_velocity = wp.dot(velocity, direction)
    magnitude = wp.max(stiffness * (distance - rest_length) - damping * radial_velocity, 0.0)
  xfrc_applied_out[worldid, body_id] = wp.spatial_vector(magnitude * direction, wp.vec3(0.0))


@wp.kernel
def _loss(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  reference: wp.array2d[float],
  qpos_ids: wp.array[int],
  step: wp.array[int],
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid, jointid = wp.tid()
  qposid = qpos_ids[jointid]
  error = qpos_in[worldid, qposid] - reference[step[0] + 1, qposid]
  wp.atomic_add(loss_out, 0, loss_scale * error * error)


@wp.kernel(enable_backward=False)
def _clamp_parameters(
  # In:
  frictionloss_min: float,
  frictionloss_max: float,
  damping_min: float,
  damping_max: float,
  # Out:
  frictionloss_scale_out: wp.array[float],
  damping_scale_out: wp.array[float],
):
  parameterid = wp.tid()
  frictionloss_scale_out[parameterid] = wp.clamp(
    frictionloss_scale_out[parameterid],
    frictionloss_min,
    frictionloss_max,
  )
  damping_scale_out[parameterid] = wp.clamp(damping_scale_out[parameterid], damping_min, damping_max)


def _actuated_joints(model):
  joint_ids = model.actuator_trnid[:, 0].astype(np.int32)
  if model.nu != _NUM_JOINTS or len(np.unique(joint_ids)) != _NUM_JOINTS:
    raise ValueError(f"Go1 system identification requires {_NUM_JOINTS} actuators on distinct joints")
  if np.any(model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
    raise ValueError("Go1 system identification requires hinge-joint actuators")
  return joint_ids, model.jnt_qposadr[joint_ids].astype(np.int32), model.jnt_dofadr[joint_ids].astype(np.int32)


def _home_data(model, qpos_ids, control):
  """Initializes the standing joint targets and the specified free-base pose."""
  key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  if key < 0:
    raise ValueError("Go1 model is missing the init_state keyframe")
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, key)
  data.qpos[:7] = _HOME_BASE_QPOS
  data.qpos[qpos_ids] = control[0]
  data.qvel[:] = 0.0
  data.ctrl[:] = control[0]
  mujoco.mj_normalizeQuat(model, data.qpos)
  mujoco.mj_forward(model, data)
  return data


def _excitation(model, horizon, eval_horizon):
  """Excites joints with five chirped harmonics and uniformly spaced joint phases."""
  if horizon < 2:
    raise ValueError("excitation horizon must be at least two")
  if eval_horizon < 0:
    raise ValueError("excitation evaluation horizon must be nonnegative")

  key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  if key < 0:
    raise ValueError("Go1 model is missing the init_state keyframe")
  home = model.key_ctrl[key].astype(np.float64)

  time = np.arange(horizon + eval_horizon) * model.opt.timestep
  duration = time[-1]
  midpoint = 0.5 * duration
  low, high = _CHIRP_FREQUENCIES
  edge = np.minimum(time, duration - time)
  ramp = np.clip(edge / _CHIRP_RAMP_DURATION, 0.0, 1.0)
  envelope = ramp**3 * (10.0 - 15.0 * ramp + 6.0 * ramp**2)
  edge_cycles = low * edge + 0.5 * (high - low) * edge**2 / midpoint
  midpoint_cycles = 0.5 * (low + high) * midpoint
  cycles = np.where(time <= midpoint, edge_cycles, 2.0 * midpoint_cycles - edge_cycles)

  harmonics = np.arange(1, _CHIRP_NUM_HARMONICS + 1)
  phase = 2.0 * np.pi * cycles[:, None, None] * harmonics[None, :, None]
  schroeder_phase = -np.pi * harmonics * (harmonics - 1) / _CHIRP_NUM_HARMONICS
  joint_phase = 2.0 * np.pi * harmonics[:, None] * np.arange(model.nu)[None] / model.nu
  signal = np.sum(np.sin(phase + schroeder_phase[None, :, None] + joint_phase[None]) / harmonics[None, :, None], axis=1)
  signal *= envelope[:, None]
  peak = np.max(np.abs(signal), axis=0, keepdims=True)
  signal = np.divide(signal, peak, out=np.zeros_like(signal), where=peak > 0.0)

  amplitudes = np.empty(model.nu)
  for actuatorid in range(model.nu):
    name = model.actuator(actuatorid).name
    leg, joint_type, *_ = name.split("_")
    if joint_type not in _CHIRP_AMPLITUDES:
      raise ValueError(f"unknown Go1 actuator: {name}")
    if joint_type == "calf" and leg in ("RR", "RL"):
      home[actuatorid] += _REAR_CALF_OFFSET
    amplitudes[actuatorid] = _CHIRP_AMPLITUDES[joint_type]
  return (home + signal * amplitudes).astype(np.float32)


def _reference_rollout(model, data, control):
  m = mjw.put_model(model)
  d, d_out = (mjw.put_data(model, data, nconmax=32, njmax=128) for _ in range(2))
  qpos = [d.qpos.numpy()[0].copy()]
  for command in control:
    d.ctrl.assign(command[None])
    mjw.step(m, d, d_out)
    qpos.append(d_out.qpos.numpy()[0].copy())
    d, d_out = d_out, d
  return np.asarray(qpos, dtype=np.float32)


def _elastic_band_length(reference_qpos):
  distance = np.linalg.norm(np.asarray(_ELASTIC_BAND_POINT) - reference_qpos[..., :3], axis=-1)
  return float(np.max(distance) + _ELASTIC_BAND_CLEARANCE)


def _initial_parameters(num_envs, num_joints, spread, sampling, seed):
  """Samples independent per-joint parameters uniformly in log space within physical bounds."""
  if not np.isfinite(spread) or spread < 0.0:
    raise ValueError("spread must be finite and nonnegative")

  def sample(reference, lower_bound, upper_bound, sample_seed):
    low = np.log(reference) - spread
    if lower_bound > 0.0:
      low = max(low, np.log(lower_bound))
    high = min(np.log(reference) + spread, np.log(upper_bound))
    log_values = _demo.sample_range(num_envs, np.full(num_joints, low), np.full(num_joints, high), sampling, sample_seed)
    return np.exp(log_values).astype(np.float32)

  frictionloss = sample(_REFERENCE_FRICTIONLOSS, _FRICTIONLOSS_MIN, _FRICTIONLOSS_MAX, seed + 1)
  damping = sample(_REFERENCE_DAMPING, _DAMPING_MIN, _DAMPING_MAX, seed)
  return frictionloss, damping


def progress(args, model, data):
  dof_ids = model.jnt_dofadr[model.actuator_trnid[:, 0]]
  frictionloss = model.dof_frictionloss[:, dof_ids]
  damping = model.dof_damping[:, dof_ids]
  friction_reference = model.dof_frictionloss_reference[dof_ids]
  damping_reference = model.dof_damping_reference[dof_ids]

  relative_friction_error = (frictionloss - friction_reference) / friction_reference
  relative_damping_error = (damping - damping_reference) / damping_reference
  relative_errors = np.concatenate((relative_friction_error, relative_damping_error), axis=1)
  error = np.sqrt(np.mean(relative_errors**2, axis=1))
  error_scale = 0.5  # map a 50% relative error to a raw progress of 0.5
  return 1.0 / (1.0 + (error / error_scale) ** 2.0)


class SysidGo1(_demo.Demo):
  Args = SysidGo1Args
  name = "sysid_go1"
  demo_type = _demo.DemoType.SYS_ID
  loss_type = _demo.LossType.PER_STEP
  model_fields = ("dof_frictionloss", "dof_damping")
  data_kwargs = {"nconmax": 32, "njmax": 128}
  layout = _viz.Layout(columns=8, spacing=2.0)

  def __init__(self, args: SysidGo1Args):
    if args.horizon < 2:
      raise ValueError("horizon must be at least two")
    if args.eval_horizon < 0:
      raise ValueError("eval_horizon must be nonnegative")

    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    _demo.set_sim_params(model, args.sim_dt, args.integrator, args.cone, args.impratio)
    model.opt.iterations = 10
    model.opt.ls_iterations = 50
    joint_ids, qpos_ids, dof_ids = _actuated_joints(model)
    control = _excitation(model, args.horizon, args.eval_horizon)

    model.dof_frictionloss[dof_ids] = _REFERENCE_FRICTIONLOSS
    model.dof_damping[dof_ids] = _REFERENCE_DAMPING
    home = _home_data(model, qpos_ids, control)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if body_id < 0:
      raise ValueError("Go1 model is missing the trunk body")
    reference_qpos = _reference_rollout(model, home, control)
    elastic_band_length = _elastic_band_length(reference_qpos) if args.gantry else None

    super().__init__(
      args,
      model,
      args.horizon,
      eval_horizon=args.eval_horizon,
      loss_scale=1.0 / args.num_envs,  # Consider full reference trajectory per env.
    )
    if args.gantry:
      self.datas[0].xfrc_applied.requires_grad = True
    frictionloss, damping = _initial_parameters(args.num_envs, _NUM_JOINTS, args.spread, args.env_sampling, args.seed)
    self.frictionloss_scale = wp.array(
      (frictionloss / _REFERENCE_FRICTIONLOSS).reshape(-1),
      dtype=float,
      requires_grad=True,
    )
    self.damping_scale = wp.array(
      (damping / _REFERENCE_DAMPING).reshape(-1),
      dtype=float,
      requires_grad=True,
    )
    self.params = [self.frictionloss_scale, self.damping_scale]

    base_frictionloss = np.tile(model.dof_frictionloss, (args.num_envs, 1)).astype(np.float32)
    base_damping = np.tile(model.dof_damping, (args.num_envs, 1)).astype(np.float32)
    self.m.dof_frictionloss = wp.array(base_frictionloss, dtype=float, requires_grad=True)
    self.m.dof_damping = wp.array(base_damping, dtype=float, requires_grad=True)
    self.model_params = [self.m.dof_frictionloss, self.m.dof_damping]

    self.joint_names = tuple(model.joint(jointid).name for jointid in joint_ids)
    self.dof_ids = wp.array(dof_ids, dtype=int)
    self.qpos_ids = wp.array(qpos_ids, dtype=int)
    self.control = wp.array(control, dtype=float)
    self.reference = wp.array(reference_qpos, dtype=float)
    self.reference_qpos = reference_qpos
    center = reference_qpos[:, :2].mean(axis=0)
    self.camera = _viz.Camera(lookat=(center[0], center[1], 0.65), distance=3.2, azimuth=135.0, elevation=-15.0)
    self.init_qpos = np.tile(home.qpos, (args.num_envs, 1)).astype(np.float32)
    self.qvel0 = np.tile(home.qvel, (args.num_envs, 1)).astype(np.float32)
    self.body_id = body_id
    self.elastic_band_point = wp.vec3(*_ELASTIC_BAND_POINT)
    self.elastic_band_length = elastic_band_length

  def reset(self):
    super().reset()
    if self.args.gantry:
      self.datas[0].xfrc_applied.zero_()

  def prepare_step(self, d):
    wp.launch(
      _set_joint_parameters,
      dim=(self.args.num_envs, _NUM_JOINTS),
      inputs=[
        self.dof_ids,
        self.frictionloss_scale,
        self.damping_scale,
        _REFERENCE_FRICTIONLOSS,
        _REFERENCE_DAMPING,
      ],
      outputs=[self.m.dof_frictionloss, self.m.dof_damping],
    )
    wp.launch(
      _set_ctrl,
      dim=(self.args.num_envs, self.m.nu),
      inputs=[self.control, self.step_index],
      outputs=[d.ctrl],
    )
    if self.args.gantry:
      wp.launch(
        _set_elastic_band_force,
        dim=self.args.num_envs,
        inputs=[
          d.qpos,
          d.qvel,
          self.body_id,
          self.elastic_band_point,
          self.elastic_band_length,
          _ELASTIC_BAND_STIFFNESS,
          _ELASTIC_BAND_DAMPING,
        ],
        outputs=[d.xfrc_applied],
      )

  def step_loss(self, d, d_out):
    del d
    wp.launch(
      _loss,
      dim=(self.args.num_envs, _NUM_JOINTS),
      inputs=[d_out.qpos, self.reference, self.qpos_ids, self.step_index, self.loss_scale],
      outputs=[self.loss],
    )

  def project(self):
    wp.launch(
      _clamp_parameters,
      dim=self.frictionloss_scale.size,
      inputs=[
        _FRICTIONLOSS_MIN / _REFERENCE_FRICTIONLOSS,
        _FRICTIONLOSS_MAX / _REFERENCE_FRICTIONLOSS,
        _DAMPING_MIN / _REFERENCE_DAMPING,
        _DAMPING_MAX / _REFERENCE_DAMPING,
      ],
      outputs=[self.frictionloss_scale, self.damping_scale],
    )

  def viz_mjm(self):
    return _visualization_mjm(self.elastic_band_length)

  def viz_trajectory(self, trajectory):
    reference = np.broadcast_to(self.reference_qpos, trajectory.shape)
    if not self.args.gantry:
      return np.concatenate((trajectory, reference), axis=-1)
    gantry = _gantry_visual_qpos(trajectory)
    return np.concatenate((trajectory, reference, gantry), axis=-1)


def _visualization_mjm(elastic_band_length):
  """Builds solid/ghost Go1 copies and, when enabled, the visual gantry."""
  scene = _viz.ghost_model(_SCENE, _ROBOT, compile=False)
  if elastic_band_length is not None:
    _add_visual_gantry(scene, elastic_band_length)
  return scene.compile()


def _add_visual_gantry(scene, elastic_band_length):
  """Adds the segmented elastic-band visualization to a scene spec."""
  segment_length = 1.05 * elastic_band_length / _GANTRY_SEGMENTS
  for segmentid in range(_GANTRY_SEGMENTS):
    body = scene.worldbody.add_body(name=f"gantry_segment_{segmentid}")
    body.add_freejoint(name=f"gantry_segment_{segmentid}_joint")
    geom = body.add_geom(name=f"gantry_segment_{segmentid}_geom")
    geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    geom.fromto[:] = (0.0, 0.0, -0.5 * segment_length, 0.0, 0.0, 0.5 * segment_length)
    geom.size[0] = _GANTRY_RADIUS
    geom.mass = 0.01
    geom.rgba = _GANTRY_RGBA
    geom.contype = 0
    geom.conaffinity = 0

  anchor = scene.worldbody.add_geom(name="gantry_anchor")
  anchor.type = mujoco.mjtGeom.mjGEOM_SPHERE
  anchor.pos[:] = _ELASTIC_BAND_POINT
  anchor.size[0] = 2.5 * _GANTRY_RADIUS
  anchor.rgba = _GANTRY_RGBA
  anchor.contype = 0
  anchor.conaffinity = 0


def _gantry_visual_qpos(trajectory):
  root = trajectory[..., :3]
  anchor = np.asarray(_ELASTIC_BAND_POINT, dtype=trajectory.dtype)
  delta = anchor - root
  direction = delta / np.linalg.norm(delta, axis=-1, keepdims=True)

  # Quaternion rotating the capsule's local +z axis onto the tether direction.
  qw = np.sqrt(0.5 * (1.0 + direction[..., 2]))
  quaternion = np.stack(
    (
      qw,
      -direction[..., 1] / (2.0 * qw),
      direction[..., 0] / (2.0 * qw),
      np.zeros_like(qw),
    ),
    axis=-1,
  )
  fractions = (np.arange(_GANTRY_SEGMENTS, dtype=trajectory.dtype) + 0.5) / _GANTRY_SEGMENTS
  centers = root[..., None, :] + fractions[None, None, :, None] * delta[..., None, :]
  quaternions = np.broadcast_to(quaternion[..., None, :], (*centers.shape[:-1], 4))
  return np.concatenate((centers, quaternions), axis=-1).reshape((*trajectory.shape[:-1], 7 * _GANTRY_SEGMENTS))


if __name__ == "__main__":
  _demo.run(SysidGo1)
