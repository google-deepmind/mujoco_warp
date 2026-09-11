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
"""Launch the red ball by optimizing the blue ball's rightward velocity."""

from dataclasses import dataclass
from dataclasses import field

import _demo
import _viz
import mujoco
import numpy as np
import warp as wp

_NUM_BALLS = 5
_RED_BALL = 0
_BLUE_BALL = _NUM_BALLS - 1
_RADIUS = 0.1
_ROD_LENGTH = 0.5
_SPACING = 2.0 * _RADIUS
_VELOCITY = 0.05
_MAX_VELOCITY = 2.5
_GRAVITY = 9.81
_TARGET_ANGLE = np.deg2rad(30.0)


@dataclass
class CradleArgs(_demo.Args):
  sim_dt: float = 0.0005
  iterations: int = 100
  lr: float = 0.05
  num_envs: int = 64
  env_sampling: str = "random"
  viz_stride: int = 40

  horizon: int = 2400
  spread: float = field(default=0.025, metadata={"help": "blue ball's initial forward velocity spread"})


def _initial_velocities(num_envs, spread, sampling, seed):
  if num_envs == 1:
    return np.array([_VELOCITY], dtype=np.float32)
  minimum = max(_VELOCITY - spread, 0.0)
  maximum = min(_VELOCITY + spread, _MAX_VELOCITY)
  return _demo.sample_range(num_envs, minimum, maximum, sampling, seed)


@wp.kernel
def _set_qvel0(
  # In:
  init_vel: wp.array[float],
  blue_ball: int,
  step: wp.array[int],
  # Data out:
  qvel_out: wp.array2d[float],
):
  worldid = wp.tid()
  if step[0] == 0:
    # Negative hinge velocity moves the rightmost ball away from the chain.
    qvel_out[worldid, blue_ball] = -init_vel[worldid]


@wp.kernel
def _loss(
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  # In:
  red_ball: int,
  rod_length: float,
  gravity: float,
  step: wp.array[int],
  horizon: int,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  if step[0] != horizon - 1:
    return
  speed = rod_length * qvel_in[worldid, red_ball]
  height = rod_length * (1.0 - wp.cos(qpos_in[worldid, red_ball]))
  kinetic_height = speed * speed / (2.0 * gravity)
  loss = -100.0 * (height + kinetic_height)  # to centimeter scale
  wp.atomic_add(loss_out, 0, loss_scale * loss)


@wp.kernel(enable_backward=False)
def _clamp_velocity(
  # In:
  maximum: float,
  # Out:
  velocity_out: wp.array[float],
):
  worldid = wp.tid()
  velocity_out[worldid] = wp.clamp(velocity_out[worldid], 0.0, maximum)


def progress(args, model, data):
  angle = data.qpos[:, -1, _RED_BALL] - _TARGET_ANGLE
  error = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
  normalized_error = error**2.0
  return 1.0 / (1.0 + normalized_error) ** 2.0


class Cradle(_demo.Demo):
  Args = CradleArgs
  name = "cradle"
  demo_type = _demo.DemoType.INIT_VAL
  loss_type = _demo.LossType.TERMINAL
  trace_body = None
  layout = _viz.Layout(columns=8, spacing=2.4)
  camera = _viz.Camera(lookat=(0.4, 0.0, 0.15), distance=2.8, azimuth=60.0, elevation=-22.0)
  data_kwargs = {"nconmax": 8, "njmax": 8}

  def __init__(self, args: CradleArgs):
    super().__init__(
      args,
      mujoco.MjModel.from_xml_string(_physics_xml()),
      args.horizon,
    )
    init_vel = _initial_velocities(args.num_envs, args.spread, args.env_sampling, args.seed)
    self.init_vel = wp.array(init_vel, dtype=float, requires_grad=True)
    self.params = [self.init_vel]
    self.init_qpos = np.tile(self.mjd.qpos, (args.num_envs, 1)).astype(np.float32)

  def prepare_step(self, d):
    wp.launch(
      _set_qvel0,
      dim=self.args.num_envs,
      inputs=[self.init_vel, _BLUE_BALL, self.step_index],
      outputs=[d.qvel],
    )

  def step_loss(self, d, d_out):
    del d
    wp.launch(
      _loss,
      dim=self.args.num_envs,
      inputs=[
        d_out.qpos,
        d_out.qvel,
        _RED_BALL,
        _ROD_LENGTH,
        _GRAVITY,
        self.step_index,
        self.horizon,
        self.loss_scale,
      ],
      outputs=[self.loss],
    )

  def project(self):
    wp.launch(_clamp_velocity, dim=self.args.num_envs, inputs=[_MAX_VELOCITY], outputs=[self.init_vel])

  def viz_mjm(self):
    return mujoco.MjModel.from_xml_string(_visual_xml())

def _physics_body(index: int):
  x = index * _SPACING
  return f"""
    <body name="b{index}" pos="{x} 0 {_ROD_LENGTH}">
      <joint name="h{index}" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -{_ROD_LENGTH}" size="0.006" mass="0.02"
            contype="0" conaffinity="0"/>
      <geom name="ball{index}" type="sphere" pos="0 0 -{_ROD_LENGTH}" size="{_RADIUS}" mass="1"
            margin="-0.0005" gap="0.0005"/>
    </body>
  """


def _physics_xml():
  bodies = "".join(_physics_body(index) for index in range(_NUM_BALLS))
  return f"""
<mujoco>
  <option gravity="0 0 -{_GRAVITY}" iterations="100" ls_iterations="50" tolerance="1e-10"/>
  <default>
    <geom condim="1" solref="-1000000 0" solimp="0 0.95 0.001 0.5 2"/>
  </default>
  <worldbody>
    {bodies}
  </worldbody>
</mujoco>
"""


def _frame_xml():
  x0 = -0.1
  x1 = (_NUM_BALLS - 1) * _SPACING + 0.1
  segments = []
  for y in (-0.22, 0.22):
    segments.extend(
      (
        (x0, y, -0.18, x0, y, _ROD_LENGTH),
        (x1, y, -0.18, x1, y, _ROD_LENGTH),
        (x0, y, _ROD_LENGTH, x1, y, _ROD_LENGTH),
        (x0, y, -0.18, x1, y, -0.18),
      )
    )
  return "".join(
    f'<geom type="capsule" fromto="{xa} {ya} {za} {xb} {yb} {zb}" size="0.012" material="frame" contype="0" conaffinity="0"/>'
    for xa, ya, za, xb, yb, zb in segments
  )


def _visual_body(index: int):
  x = index * _SPACING
  material = "red" if index == 0 else "blue" if index == _NUM_BALLS - 1 else "gray"
  return f"""
    <body name="b{index}" pos="{x} 0 {_ROD_LENGTH}">
      <joint name="h{index}" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 -{_ROD_LENGTH} 0 -0.22 0" size="0.0045" mass="0.001"
            material="string" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 -{_ROD_LENGTH} 0 0.22 0" size="0.0045" mass="0.001"
            material="string" contype="0" conaffinity="0"/>
      <body name="ball{index}" pos="0 0 -{_ROD_LENGTH}">
        <geom name="ball{index}" type="sphere" size="{_RADIUS}" mass="1" material="{material}"
              contype="0" conaffinity="0"/>
      </body>
    </body>
  """


def _visual_xml():
  bodies = "".join(_visual_body(index) for index in range(_NUM_BALLS))
  return f"""
<mujoco>
  <visual>
    {_demo.MENAGERIE_VISUAL}
  </visual>
  <asset>
    {_demo.MENAGERIE_ASSETS}
    <material name="red" rgba="0.85 0.2 0.2 1" specular="0.5" shininess="0.5"/>
    <material name="gray" rgba="0.7 0.7 0.75 1" specular="0.5" shininess="0.5"/>
    <material name="blue" rgba="0.2 0.4 0.85 1" specular="0.5" shininess="0.5"/>
    <material name="frame" rgba="0.5 0.5 0.55 1" specular="0.8" shininess="0.6"/>
    <material name="string" rgba="0.82 0.82 0.88 1" specular="0.4" shininess="0.3" emission="0.3"/>
  </asset>
  <worldbody>
    {_demo.MENAGERIE_LIGHTS}
    <geom name="floor" type="plane" pos="0.4 0 -0.2" size="5 5 0.01" material="groundplane"
          contype="0" conaffinity="0"/>
    {_frame_xml()}
    {bodies}
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
  _demo.run(Cradle)
