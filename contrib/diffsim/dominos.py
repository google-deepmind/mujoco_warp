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
"""Optimize the initial push force that topples a spiral of dominoes."""

import math
from dataclasses import dataclass
from dataclasses import field

import _demo
import _viz
import mujoco
import numpy as np
import warp as wp

_NUM_DOMINOES = 16
_HALF_SIZE = (0.06, 0.016, 0.18)
_MASS = 0.2
_INNER_RADIUS = 0.35
_SPIRAL_PITCH = 0.32
_SPACING = 0.12
_PUSH_GRANULARITY = 0.01
_PUSH_MIN = 0.0
_PUSH_MAX = 5.5
_TIP_WEIGHT = 0.1
_FALLEN_HEIGHT = 0.6 * _HALF_SIZE[2]
# A flat domino rests at its half-thickness; one on its edge rests at its half-width.
_FLAT_HEIGHT = 0.5 * (_HALF_SIZE[0] + _HALF_SIZE[1])


@dataclass
class DominosArgs(_demo.Args):
  sim_dt: float = 0.005
  iterations: int = 100
  lr: float = 0.015
  num_envs: int = 64

  horizon: int = 500
  push_init: float = field(default=2.7, metadata={"help": "initial push-force center"})
  spread: float = field(default=0.05, metadata={"help": "initial push-force half-range"})


def _initial_push(num_envs: int, center: float, spread: float, sampling="random", seed=0):
  if num_envs == 1:
    return np.array([center], dtype=np.float32)
  push = _demo.sample_range(num_envs, center - spread, center + spread, sampling, seed)
  push = np.round(push / _PUSH_GRANULARITY) * _PUSH_GRANULARITY
  return np.clip(push, _PUSH_MIN, _PUSH_MAX).astype(np.float32)


@wp.kernel
def _set_push(
  # In:
  push: wp.array[float],
  direction: wp.vec3,
  push_height: float,
  body_id: int,
  step: wp.array[int],
  # Data out:
  xfrc_applied_out: wp.array2d[wp.spatial_vector],
):
  worldid = wp.tid()
  force = wp.vec3(0.0)
  torque = wp.vec3(0.0)
  if step[0] == 0:
    force = push[worldid] * direction
    # xfrc_applied is a COM wrench; r x F makes this a point force at the top.
    torque = wp.cross(wp.vec3(0.0, 0.0, push_height), force)
  xfrc_applied_out[worldid, body_id] = wp.spatial_vector(force, torque)


@wp.kernel
def _loss(
  # Data in:
  qpos_in: wp.array2d[float],
  push_direction: wp.vec3,
  inv_num_dominoes: float,
  inv_standing_height: float,
  tip_weight: float,
  step: wp.array[int],
  horizon: int,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid, dominoid = wp.tid()
  height = qpos_in[worldid, 7 * dominoid + 2] * inv_standing_height
  progress = float(step[0] + 1) / float(horizon)
  value = progress * height * height * inv_num_dominoes
  if dominoid == 0:
    qw = qpos_in[worldid, 3]
    qx = qpos_in[worldid, 4]
    qy = qpos_in[worldid, 5]
    qz = qpos_in[worldid, 6]
    up_x = 2.0 * (qx * qz + qw * qy)
    up_y = 2.0 * (qy * qz - qw * qx)
    value -= tip_weight * progress * (up_x * push_direction[0] + up_y * push_direction[1])
  wp.atomic_add(loss_out, 0, loss_scale * value)


@wp.kernel(enable_backward=False)
def _clamp_push(
  # In:
  minimum: float,
  maximum: float,
  # Out:
  push_out: wp.array[float],
):
  worldid = wp.tid()
  push_out[worldid] = wp.clamp(push_out[worldid], minimum, maximum)


def progress(args, model, data):
  height = data.qpos[:, -1].reshape(data.qpos.shape[0], _NUM_DOMINOES, 7)[:, :, 2]
  fallen = height < _FALLEN_HEIGHT
  fallen[:, -1] = height[:, -1] < _FLAT_HEIGHT
  return np.mean(fallen, axis=1)


class Dominos(_demo.Demo):
  Args = DominosArgs
  name = "dominos"
  demo_type = _demo.DemoType.INIT_VAL
  loss_type = _demo.LossType.PER_STEP
  layout = _viz.Layout(columns=8, spacing=2.2)
  data_kwargs = {"nconmax": 128, "njmax": 384}

  def __init__(self, args: DominosArgs):
    super().__init__(args, mujoco.MjModel.from_xml_string(_model_xml()), args.horizon)
    self.datas[0].xfrc_applied.requires_grad = True
    init_push = _initial_push(args.num_envs, args.push_init, args.spread, args.env_sampling, args.seed)
    self.push = wp.array(init_push, dtype=float, requires_grad=True)
    self.params = [self.push]
    self.init_qpos = np.tile(self.mjd.qpos, (args.num_envs, 1)).astype(np.float32)
    self.body_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "d0")
    poses = _poses()
    _, _, qw, qz = poses[0]
    yaw = 2.0 * math.atan2(qz, qw)
    direction_np = np.array((-math.sin(yaw), math.cos(yaw), 0.0))
    self.direction = wp.vec3(*direction_np)

    xy = np.asarray([(x, y) for x, y, _, _ in poses])
    center = xy.mean(axis=0)
    extent = np.abs(xy - center).max() + 2.0 * _HALF_SIZE[2]
    self.camera = _viz.Camera(
      lookat=(center[0], center[1], 0.05),
      distance=2.2 + 2.0 * extent,
      azimuth=90.0,
      elevation=-55.0,
    )

  def reset(self):
    d = self.datas[0]
    d.qpos.assign(self.init_qpos)
    d.qvel.zero_()
    d.xfrc_applied.zero_()
    d.qacc_warmstart.zero_()
    d.time.zero_()

  def prepare_step(self, d):
    wp.launch(
      _set_push,
      dim=self.args.num_envs,
      inputs=[self.push, self.direction, _HALF_SIZE[2], self.body_id, self.step_index],
      outputs=[d.xfrc_applied],
    )

  def step_loss(self, d, d_out):
    del d
    wp.launch(
      _loss,
      dim=(self.args.num_envs, _NUM_DOMINOES),
      inputs=[
        d_out.qpos,
        self.direction,
        1.0 / _NUM_DOMINOES,
        1.0 / _HALF_SIZE[2],
        _TIP_WEIGHT,
        self.step_index,
        self.horizon,
        self.loss_scale,
      ],
      outputs=[self.loss],
    )

  def project(self):
    wp.launch(_clamp_push, dim=self.args.num_envs, inputs=[_PUSH_MIN, _PUSH_MAX], outputs=[self.push])

def _poses():
  poses = []
  rate = _SPIRAL_PITCH / (2.0 * math.pi)
  theta = 0.0
  for index in range(_NUM_DOMINOES):
    radius = _INNER_RADIUS + rate * theta
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    tangent_x = rate * math.cos(theta) - radius * math.sin(theta)
    tangent_y = rate * math.sin(theta) + radius * math.cos(theta)
    yaw = math.atan2(-tangent_x, tangent_y)
    poses.append((x, y, math.cos(0.5 * yaw), math.sin(0.5 * yaw)))
    theta += _SPACING / math.sqrt(radius * radius + rate * rate)
  return poses


def _body_xml(index, pose):
  x, y, qw, qz = pose
  blend = index / (_NUM_DOMINOES - 1)
  color = (0.9 - 0.7 * blend, 0.25 + 0.55 * blend, 0.85 - 0.65 * blend)
  return f"""
    <body name="d{index}" pos="{x:.4f} {y:.4f} {_HALF_SIZE[2]}" quat="{qw:.5f} 0 0 {qz:.5f}">
      <freejoint/>
      <geom name="d{index}" type="box" size="{_HALF_SIZE[0]} {_HALF_SIZE[1]} {_HALF_SIZE[2]}"
            mass="{_MASS}" rgba="{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} 1"/>
    </body>
  """


def _model_xml():
  bodies = "".join(_body_xml(index, pose) for index, pose in enumerate(_poses()))
  return f"""
<mujoco>
  <option gravity="0 0 -9.81" iterations="50"/>
  <visual>
    {_demo.MENAGERIE_VISUAL}
  </visual>
  <asset>
    {_demo.MENAGERIE_ASSETS}
  </asset>
  <default>
    <geom condim="3" friction="1 0.005 0.0001" solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </default>
  <worldbody>
    {_demo.MENAGERIE_LIGHTS}
    <geom name="floor" type="plane" size="5 5 0.01" material="groundplane"/>
    {bodies}
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
  _demo.run(Dominos)
