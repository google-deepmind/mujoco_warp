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
"""Identify surface friction from a box sliding down an incline."""

from dataclasses import dataclass
from dataclasses import field

import _demo
import _viz
import mujoco
import numpy as np
import warp as wp

_INCLINE_ANGLE = np.deg2rad(25.0)
_COS_ANGLE = np.cos(_INCLINE_ANGLE)
_SIN_ANGLE = np.sin(_INCLINE_ANGLE)
_QUAT_W = np.cos(0.5 * _INCLINE_ANGLE)
_QUAT_Y = np.sin(0.5 * _INCLINE_ANGLE)
_GRAVITY_X = 9.81 * _SIN_ANGLE
_GRAVITY_Z = -9.81 * _COS_ANGLE
_CRITICAL_FRICTION = np.tan(_INCLINE_ANGLE)
_INITIAL_FRICTION = 0.2
_FRICTION_MIN = 0.02
_FRICTION_MAX = 1.2
_SETTLE_TIME = 0.16


@dataclass
class SysidInclineArgs(_demo.Args):
  sim_dt: float = 0.004
  integrator: str = "euler"
  iterations: int = 60
  lr: float = 0.02
  optimizer: str = "adam"
  num_envs: int = 4
  env_sampling: str = "linspace"

  horizon: int = 100
  eval_horizon: int = field(default=100, metadata={"help": "forward-only steps after the fitted rollout"})
  spread: float = field(default=0.15, metadata={"help": "initial friction spread"})


@wp.kernel
def _set_friction(
  # In:
  friction: wp.array[float],
  base: wp.array[wp.vec3],
  # Out:
  friction_out: wp.array2d[wp.vec3],
):
  worldid, geomid = wp.tid()
  value = base[geomid]
  friction_out[worldid, geomid] = wp.vec3(friction[worldid], value[1], value[2])


@wp.kernel
def _loss(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  rest_x: float,
  step: wp.array[int],
  horizon: int,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  if step[0] != horizon - 1:
    return
  displacement = qpos_in[worldid, 0] - rest_x
  wp.atomic_add(loss_out, 0, loss_scale * displacement * displacement)


@wp.kernel(enable_backward=False)
def _clamp_friction(
  # In:
  minimum: float,
  maximum: float,
  # Out:
  friction_out: wp.array[float],
):
  worldid = wp.tid()
  friction_out[worldid] = wp.clamp(friction_out[worldid], minimum, maximum)


def _rest_pose(model):
  friction = model.geom_friction[:, 0].copy()
  model.geom_friction[:, 0] = _FRICTION_MAX
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  for _ in range(round(_SETTLE_TIME / model.opt.timestep)):
    mujoco.mj_step(model, data)
  model.geom_friction[:, 0] = friction
  return data.qpos.astype(np.float32)


def _initial_friction(num_envs: int, spread: float, sampling="linspace", seed=0):
  friction = np.full(num_envs, _INITIAL_FRICTION, dtype=np.float32)
  if num_envs > 1:
    offsets = _demo.sample_range(num_envs, -spread, spread, sampling, seed)
    # Indices run toward the back of the shot, so put the lowest linspace friction last.
    friction += offsets[::-1] if sampling == "linspace" else offsets
  return friction


def _rotate_trajectory(trajectory):
  rotated = np.array(trajectory, copy=True)
  x, y, z = np.moveaxis(trajectory[..., :3], -1, 0)
  rotated[..., 0] = _COS_ANGLE * x + _SIN_ANGLE * z
  rotated[..., 1] = y
  rotated[..., 2] = -_SIN_ANGLE * x + _COS_ANGLE * z

  w, x, y, z = np.moveaxis(trajectory[..., 3:7], -1, 0)
  rotated[..., 3] = _QUAT_W * w - _QUAT_Y * y
  rotated[..., 4] = _QUAT_W * x + _QUAT_Y * z
  rotated[..., 5] = _QUAT_W * y + _QUAT_Y * w
  rotated[..., 6] = _QUAT_W * z - _QUAT_Y * x
  return rotated


def progress(args, model, data):
  friction = model.geom_friction[:, 0, 0]
  # Friction at or above the critical value lies in the one-sided no-slip band.
  gap = np.maximum(_CRITICAL_FRICTION - friction, 0.0)
  error = gap / _CRITICAL_FRICTION
  error_scale = 0.5  # map a 50% relative error to a raw progress of 0.5
  return 1.0 / (1.0 + (error / error_scale) ** 2.0)


class SysidIncline(_demo.Demo):
  Args = SysidInclineArgs
  name = "sysid_incline"
  demo_type = _demo.DemoType.SYS_ID
  loss_type = _demo.LossType.TERMINAL
  model_fields = ("geom_friction",)
  trace_body = "box"
  layout = _viz.Layout(columns=1, spacing=(1.8, 1.15))
  camera = _viz.Camera(lookat=(0.0, 0.0, 0.05), distance=2.2, azimuth=70.0, elevation=-20.0)

  def __init__(self, args: SysidInclineArgs):
    super().__init__(
      args,
      mujoco.MjModel.from_xml_string(_model_xml()),
      args.horizon,
      eval_horizon=args.eval_horizon,
    )
    self.rest_qpos = _rest_pose(self.mjm)
    self.rest_x = float(self.rest_qpos[0])
    self.friction = wp.array(
      _initial_friction(args.num_envs, args.spread, args.env_sampling, args.seed),
      dtype=float,
      requires_grad=True,
    )
    self.params = [self.friction]

    base = self.mjm.geom_friction.astype(np.float32)
    self.base_friction = wp.array(base, dtype=wp.vec3)
    batched = np.tile(base, (args.num_envs, 1, 1))
    self.m.geom_friction = wp.array(batched, dtype=wp.vec3, requires_grad=True)
    self.model_params = [self.m.geom_friction]
    self.init_qpos = np.tile(self.rest_qpos, (args.num_envs, 1))

  def prepare_step(self, d):
    del d
    wp.launch(
      _set_friction,
      dim=(self.args.num_envs, self.m.ngeom),
      inputs=[self.friction, self.base_friction],
      outputs=[self.m.geom_friction],
    )

  def step_loss(self, d, d_out):
    del d
    wp.launch(
      _loss,
      dim=self.args.num_envs,
      inputs=[d_out.qpos, self.rest_x, self.step_index, self.horizon, self.loss_scale],
      outputs=[self.loss],
    )

  def project(self):
    wp.launch(
      _clamp_friction,
      dim=self.args.num_envs,
      inputs=[_FRICTION_MIN, _FRICTION_MAX],
      outputs=[self.friction],
    )

  def viz_mjm(self):
    return mujoco.MjModel.from_xml_string(_model_xml(visual=True))

  def viz_trajectory(self, trajectory):
    return _rotate_trajectory(trajectory)


def _model_xml(visual=False):
  surface = '<geom name="floor" type="plane" size="5 5 0.01" material="groundplane"/>'
  if visual:
    length, width, thickness = 0.9, 0.35, 0.03
    center_x = -thickness * _SIN_ANGLE
    center_z = -thickness * _COS_ANGLE
    ground_z = center_z - length * _SIN_ANGLE - thickness * _COS_ANGLE
    surface = f"""
    <geom name="ground" type="plane" pos="0 0 {ground_z:.5f}" size="5 5 0.01" material="groundplane"/>
    <body name="incline" pos="{center_x:.5f} 0 {center_z:.5f}">
      <geom name="ramp" type="box" size="{length} {width} {thickness}" euler="0 {_INCLINE_ANGLE} 0"
            rgba="0.48 0.58 0.70 1"/>
    </body>"""
  return f"""
<mujoco>
  <compiler angle="radian"/>
  <option gravity="{_GRAVITY_X:.5f} 0 {_GRAVITY_Z:.5f}" iterations="50"/>
  <visual>
    {_demo.MENAGERIE_VISUAL}
  </visual>
  <asset>
    {_demo.MENAGERIE_ASSETS}
  </asset>
  <default>
    <geom condim="3" friction="0.1 0.005 0.0001" solimp="0 0.95 0.001"/>
  </default>
  <worldbody>
    {_demo.MENAGERIE_LIGHTS}
    {surface}
    <body name="box" pos="-0.25 0 0.1">
      <freejoint/>
      <geom name="box" type="box" size="0.1 0.1 0.1" mass="1" rgba="0.9 0.7 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
  _demo.run(SysidIncline)
