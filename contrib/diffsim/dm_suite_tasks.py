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
"""Task models and losses for the dm_suite differentiable simulation demo."""

import _demo
import _viz
import numpy as np
import warp as wp

_ANT_TERMINATION_HEIGHT = 0.27
_ANT_STANDING_HEIGHT = 0.5702
_ANT_TARGET_DISTANCE = 1.0
_ANT_GAIT_FREQUENCY = 1.0
_ANT_HIP_SWING = 0.4
_ANT_ANKLE_STANCE = 1.05
_ANT_ANKLE_LIFT = 0.2
_CHEETAH_TARGET_DISTANCE = 1.0
_TARGET_BASE_Z = 0.009
_TARGET_RINGS = (
  (0.0250, 0.0010, "0.90 0.44 0.46 1"),
  (0.0190, 0.0012, "0.97 0.97 0.97 1"),
  (0.0130, 0.0014, "0.90 0.44 0.46 1"),
  (0.0070, 0.0016, "0.97 0.97 0.97 1"),
  (0.0025, 0.0018, "0.90 0.44 0.46 1"),
)

_COMMON_ASSETS = f"""
<asset>
  {_demo.MENAGERIE_ASSETS}
  <material name="self" rgba="0.85 0.6 0.35 1"/>
  <material name="decoration" rgba="0.35 0.5 0.7 1"/>
</asset>
"""

ASSETS = {
  # Ant adapted from Gym's MIT-licensed model,
  # with uniform density scaled to dm_control's 12.38 kg total mass.
  "ant": {
    "compiler": '<compiler angle="radian"/>',
    "default": """
    <default>
      <motor ctrlrange="-1 1" ctrllimited="true" gear="150"/>
      <geom friction="1 .5 .5" margin=".001" material="self" density="67.963354395469"/>
      <joint limited="true" armature="1" damping="1"
             solreflimit=".04 1" solimplimit="0 .8 .03"/>
    </default>
  """,
    "floor": '<geom name="ground" type="plane" size="100 100 .5" material="groundplane"/>',
    "body": """
    <body name="torso" pos="0 0 .75">
      <freejoint name="root"/>
      <geom name="torso_geom" type="sphere" size=".25"/>

      <body name="front_left_leg">
        <geom name="front_left_aux_geom" type="capsule" size=".08" fromto="0 0 0 .2 .2 0"/>
        <body name="front_left_aux" pos=".2 .2 0">
          <joint name="front_left_hip" range="-.523599 .523599"/>
          <geom name="front_left_leg_geom" type="capsule" size=".08" fromto="0 0 0 .2 .2 0"/>
          <body name="front_left_foot" pos=".2 .2 0">
            <joint name="front_left_ankle" axis="-1 1 0" range=".523599 1.745329"/>
            <geom name="front_left_ankle_geom" type="capsule" size=".08" fromto="0 0 0 .4 .4 0"/>
          </body>
        </body>
      </body>

      <body name="front_right_leg">
        <geom name="front_right_aux_geom" type="capsule" size=".08" fromto="0 0 0 .2 -.2 0"/>
        <body name="front_right_aux" pos=".2 -.2 0">
          <joint name="front_right_hip" range="-.523599 .523599"/>
          <geom name="front_right_leg_geom" type="capsule" size=".08" fromto="0 0 0 .2 -.2 0"/>
          <body name="front_right_foot" pos=".2 -.2 0">
            <joint name="front_right_ankle" axis="1 1 0" range=".523599 1.745329"/>
            <geom name="front_right_ankle_geom" type="capsule" size=".08" fromto="0 0 0 .4 -.4 0"/>
          </body>
        </body>
      </body>

      <body name="back_right_leg">
        <geom name="back_right_aux_geom" type="capsule" size=".08" fromto="0 0 0 -.2 -.2 0"/>
        <body name="back_right_aux" pos="-.2 -.2 0">
          <joint name="back_right_hip" range="-.523599 .523599"/>
          <geom name="back_right_leg_geom" type="capsule" size=".08" fromto="0 0 0 -.2 -.2 0"/>
          <body name="back_right_foot" pos="-.2 -.2 0">
            <joint name="back_right_ankle" axis="-1 1 0" range="-1.745329 -.523599"/>
            <geom name="back_right_ankle_geom" type="capsule" size=".08" fromto="0 0 0 -.4 -.4 0"/>
          </body>
        </body>
      </body>

      <body name="back_left_leg">
        <geom name="back_left_aux_geom" type="capsule" size=".08" fromto="0 0 0 -.2 .2 0"/>
        <body name="back_left_aux" pos="-.2 .2 0">
          <joint name="back_left_hip" range="-.523599 .523599"/>
          <geom name="back_left_leg_geom" type="capsule" size=".08" fromto="0 0 0 -.2 .2 0"/>
          <body name="back_left_foot" pos="-.2 .2 0">
            <joint name="back_left_ankle" axis="1 1 0" range="-1.745329 -.523599"/>
            <geom name="back_left_ankle_geom" type="capsule" size=".08" fromto="0 0 0 -.4 .4 0"/>
          </body>
        </body>
      </body>
    </body>
  """,
    "actuator": """
    <actuator>
      <motor name="front_left_hip" joint="front_left_hip"/>
      <motor name="front_left_ankle" joint="front_left_ankle"/>
      <motor name="front_right_hip" joint="front_right_hip"/>
      <motor name="front_right_ankle" joint="front_right_ankle"/>
      <motor name="back_right_hip" joint="back_right_hip"/>
      <motor name="back_right_ankle" joint="back_right_ankle"/>
      <motor name="back_left_hip" joint="back_left_hip"/>
      <motor name="back_left_ankle" joint="back_left_ankle"/>
    </actuator>
  """,
    "contact": """
    <contact>
      <exclude body1="torso" body2="front_left_aux"/>
      <exclude body1="torso" body2="front_right_aux"/>
      <exclude body1="torso" body2="back_right_aux"/>
      <exclude body1="torso" body2="back_left_aux"/>
    </contact>
  """,
  },
  "cheetah": {
    "compiler": '<compiler settotalmass="14"/>',
    "default": """
    <default>
      <default class="cheetah">
        <joint limited="true" damping=".01" armature=".1" stiffness="8" type="hinge" axis="0 1 0"/>
        <geom contype="1" conaffinity="1" condim="3" friction=".4 .1 .1" material="self" type="capsule"/>
      </default>
      <default class="free"><joint limited="false" damping="0" armature="0" stiffness="0"/></default>
      <motor ctrllimited="true" ctrlrange="-1 1"/>
    </default>
  """,
    "floor": '<geom name="ground" type="plane" conaffinity="1" size="100 16 .5" material="groundplane"/>',
    "body": """
    <body name="torso" pos="0 0 .624" childclass="cheetah">
      <joint name="rootx" type="slide" axis="1 0 0" class="free"/>
      <joint name="rootz" type="slide" axis="0 0 1" class="free"/>
      <joint name="rooty" type="hinge" axis="0 1 0" class="free"/>
      <geom type="capsule" fromto="-.5 0 0 .5 0 0" size="0.046"/>
      <geom type="capsule" pos=".6 0 .1" euler="0 50 0" size="0.046 .15"/>
      <body name="bthigh" pos="-.5 0 0">
        <joint name="bthigh" range="-30 60" stiffness="240" damping="6"/>
        <geom pos=".1 0 -.13" euler="0 -218 0" size="0.046 .145"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint name="bshin" range="-50 50" stiffness="180" damping="4.5"/>
          <geom pos="-.14 0 -.07" euler="0 -116 0" size="0.046 .15"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint name="bfoot" range="-230 50" stiffness="120" damping="3"/>
            <geom pos=".03 0 -.097" euler="0 -15 0" size="0.046 .094"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos=".5 0 0">
        <joint name="fthigh" range="-57 0.4" stiffness="180" damping="4.5"/>
        <geom pos="-.07 0 -.12" euler="0 30 0" size="0.046 .133"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint name="fshin" range="-70 50" stiffness="120" damping="3"/>
          <geom pos=".065 0 -.09" euler="0 -34 0" size="0.046 .106"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint name="ffoot" range="-28 28" stiffness="60" damping="1.5"/>
            <geom pos=".045 0 -.07" euler="0 -34 0" size="0.046 .07"/>
          </body>
        </body>
      </body>
    </body>
  """,
    "actuator": """
    <actuator>
      <motor name="bthigh" joint="bthigh" gear="120"/>
      <motor name="bshin" joint="bshin" gear="90"/>
      <motor name="bfoot" joint="bfoot" gear="60"/>
      <motor name="fthigh" joint="fthigh" gear="90"/>
      <motor name="fshin" joint="fshin" gear="60"/>
      <motor name="ffoot" joint="ffoot" gear="30"/>
    </actuator>
  """,
  },
  "cartpole": {
    "option": '<option><flag contact="disable"/></option>',
    "default": """
    <default>
      <default class="pole">
        <joint type="hinge" axis="0 1 0" damping="2e-6"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" material="self" mass=".1"/>
      </default>
    </default>
  """,
    "floor": '<geom name="floor" pos="0 0 -.05" size="4 4 .2" type="plane" material="groundplane"/>',
    "body": """
    <geom type="capsule" pos="0 .07 1" zaxis="1 0 0" size="0.02 2" material="decoration"/>
    <geom type="capsule" pos="0 -.07 1" zaxis="1 0 0" size="0.02 2" material="decoration"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" limited="true" axis="1 0 0" range="-1.8 1.8"
             solreflimit=".08 1" damping="5e-4"/>
      <geom type="box" size="0.2 0.15 0.1" material="self" mass="1"/>
      <body name="pole" childclass="pole"><joint name="hinge"/><geom/></body>
    </body>
  """,
    "actuator": '<actuator><motor joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/></actuator>',
  },
  "reacher": {
    "option": '<option><flag contact="disable"/></option>',
    "default": """
    <default>
      <joint type="hinge" axis="0 0 1" damping="0.01"/>
      <motor gear="0.05" ctrlrange="-1 1" ctrllimited="true"/>
    </default>
  """,
    "floor": '<geom name="ground" type="plane" size="2 2 .1" material="groundplane"/>',
    "body": """
    <geom type="cylinder" fromto="0 0 0 0 0 0.02" size=".011" material="decoration"/>
    <body name="arm" pos="0 0 .01">
      <geom type="capsule" fromto="0 0 0 .12 0 0" size=".01" material="self"/>
      <joint name="shoulder"/>
      <body name="hand" pos=".12 0 0">
        <geom type="capsule" fromto="0 0 0 .1 0 0" size=".01" material="self"/>
        <joint name="wrist"/>
        <body name="finger" pos=".12 0 0">
          <geom type="sphere" size=".013" material="self"/>
          <site name="tip" size=".001" rgba="0 0 0 0"/>
        </body>
      </body>
    </body>
  """,
    "actuator": '<actuator><motor joint="shoulder"/><motor joint="wrist"/></actuator>',
  },
}


def _target_xml(target):
  if target is None:
    return ""
  rings = "".join(
    f'<site name="target_{index}" type="cylinder" pos="0 0 {_TARGET_BASE_Z + height}" size="{radius} {height}" rgba="{rgba}"/>'
    for index, (radius, height, rgba) in enumerate(_TARGET_RINGS)
  )
  return f'<body name="target" pos="{target[0]} {target[1]} 0">{rings}</body>'


def model_xml(assets, target=None):
  """Builds a task model, optionally showing its target."""
  target_geom = _target_xml(target)
  return f"""
<mujoco>
  {assets.get("compiler", "")}
  {assets.get("option", "")}
  {assets["default"]}
  <visual>
    {_demo.MENAGERIE_VISUAL}
    <map znear="0.02"/>
  </visual>
  {_COMMON_ASSETS}
  <worldbody>
    {_demo.MENAGERIE_LIGHTS}
    {assets["floor"]}
    {assets["body"]}
    {target_geom}
  </worldbody>
  {assets["actuator"]}
  {assets.get("contact", "")}
</mujoco>
"""


TASK_KWARGS = {
  "ant": {
    "sim_dt": 0.01,
    "ctrl_dt": 0.01,
    "horizon": 360,
    "iterations": 200,
    "lr": 0.005,
    "num_envs": 64,
    "env_sampling": "random",
    "tbptt": 50,
    "viz_stride": 2,
    "spread": 1.0,
    "layout": _viz.Layout(columns=8, spacing=3.6),
    "camera": _viz.Camera(lookat=(0.0, 0.0, 0.45), distance=5.0, azimuth=65.0, elevation=-18.0),
    "_cfg": {
      "assets": ASSETS["ant"],
      "init_ctrl_noise": 0.01,
      "indices": {"forward": 0, "height": 2, "quaternion": 3},
      "weights": {
        "forward_velocity": 2.0,
        "lateral_velocity": 1.0,
        "vertical_velocity": 2.0,
        "height": 10.0,
        "fall": 10.0,
        "upright": 2.0,
        "heading": 1.0,
        "angular_velocity": 0.1,
        "gait": 5.0,
        "control": 0.02,
      },
      "init_qpos": (
        0.0,
        0.0,
        _ANT_STANDING_HEIGHT,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        _ANT_ANKLE_STANCE,
        0.0,
        _ANT_ANKLE_STANCE,
        0.0,
        -_ANT_ANKLE_STANCE,
        0.0,
        -_ANT_ANKLE_STANCE,
      ),
      "init_qpos_range": (7, -0.1, 0.1),
    },
  },
  "cheetah": {
    "sim_dt": 0.01,
    "ctrl_dt": 0.01,
    "horizon": 100,
    "iterations": 200,
    "lr": 0.01,
    "num_envs": 64,
    "env_sampling": "random",
    "tbptt": None,
    "viz_stride": 2,
    "spread": 1.0,
    "layout": _viz.Layout(columns=8, spacing=3.4),
    "camera": _viz.Camera(lookat=(0.0, 0.0, 0.6), distance=6.0, azimuth=75.0, elevation=-15.0),
    "_cfg": {
      "assets": ASSETS["cheetah"],
      "init_ctrl_noise": 0.02,
      "indices": {"forward": 0, "height": 1, "pitch": 2},
      "weights": {"forward": 1.0, "height": 2.0, "upright": 2.0, "angular_velocity": 0.1},
    },
  },
  "cartpole": {
    "sim_dt": 0.01,
    "ctrl_dt": 0.01,
    "horizon": 200,
    "iterations": 200,
    "lr": 0.02,
    "num_envs": 4,
    "env_sampling": "linspace",
    "tbptt": None,
    "viz_stride": 2,
    "spread": 1.0,
    "layout": _viz.Layout(columns=1, spacing=(5.0, 1.3)),
    "camera": _viz.Camera(lookat=(0.0, 0.0, 1.0), distance=4.0, azimuth=90.0, elevation=-8.0),
    "_cfg": {
      "assets": ASSETS["cartpole"],
      "indices": {"cart": 0, "pole": 1},
      "weights": {"angle": 1.0, "pole_velocity": 0.1, "cart_position": 0.2, "cart_velocity": 0.2},
      "init_qpos": (0.0, np.pi),
      "init_qpos_range": (1, np.pi - 0.65, np.pi + 0.65),
    },
  },
  "reacher": {
    "sim_dt": 0.02,
    "ctrl_dt": 0.02,
    "horizon": 120,
    "iterations": 100,
    "lr": 0.002,
    "num_envs": 64,
    "env_sampling": "random",
    "tbptt": None,
    "viz_stride": 1,
    "spread": 1.0,
    "layout": _viz.Layout(columns=8, spacing=0.45),
    "camera": _viz.Camera(lookat=(0.0, 0.0, 0.02), distance=0.6, azimuth=90.0, elevation=-60.0),
    "_cfg": {
      "assets": ASSETS["reacher"],
      "indices": {},
      "weights": {},
      "loss_type": _demo.LossType.TERMINAL,
      "init_qpos": (-0.75, 1.5),
      "target": (0.18, -1.0, 1.0),
      "sites": ("tip",),
    },
  },
}


@wp.func
def _ant_gait_target(time: float, leg: int) -> wp.vec2:
  """Returns hip and ankle position targets for one leg."""
  front_sign = 1.0 - 2.0 * float(leg // 2)
  side_sign = 1.0 - 2.0 * float(leg % 2)
  phase = 2.0 * wp.pi * _ANT_GAIT_FREQUENCY * time + float(leg % 2) * wp.pi
  return wp.vec2(
    front_sign * side_sign * _ANT_HIP_SWING * wp.cos(phase),
    front_sign * (_ANT_ANKLE_STANCE - _ANT_ANKLE_LIFT * wp.max(wp.sin(phase), 0.0)),
  )


@wp.kernel
def _ant_loss(
  # In:
  init_qpos: wp.array2d[float],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  ctrl_in: wp.array2d[float],
  # In:
  forward: int,
  height: int,
  quaternion: int,
  timestep: float,
  w_forward_velocity: float,
  w_lateral_velocity: float,
  w_vertical_velocity: float,
  w_height: float,
  w_fall: float,
  w_upright: float,
  w_heading: float,
  w_angular_velocity: float,
  w_gait: float,
  w_control: float,
  step: wp.array[int],
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  time = float(step[0] + 1) * timestep
  qx = qpos_in[worldid, quaternion + 1]
  qy = qpos_in[worldid, quaternion + 2]
  qz = qpos_in[worldid, quaternion + 3]
  upright = 2.0 * (qx * qx + qy * qy)
  heading = 2.0 * (qy * qy + qz * qz)
  height_error = qpos_in[worldid, height] - init_qpos[worldid, height]
  fall = wp.max(_ANT_TERMINATION_HEIGHT - qpos_in[worldid, height], 0.0)

  loss = -w_forward_velocity * qvel_in[worldid, forward]
  loss += w_lateral_velocity * qvel_in[worldid, 1] ** 2.0 + w_vertical_velocity * qvel_in[worldid, 2] ** 2.0
  loss += w_height * height_error * height_error + w_upright * upright + w_heading * heading
  loss += w_angular_velocity * (qvel_in[worldid, 3] ** 2.0 + qvel_in[worldid, 4] ** 2.0 + qvel_in[worldid, 5] ** 2.0)
  loss += w_fall * fall * fall

  for leg in range(4):
    hip = 7 + 2 * leg
    target = _ant_gait_target(time, leg)
    gait_error = wp.vec2(qpos_in[worldid, hip], qpos_in[worldid, hip + 1]) - target
    loss += w_gait * wp.dot(gait_error, gait_error)
  for actuator in range(ctrl_in.shape[1]):
    loss += w_control * ctrl_in[worldid, actuator] ** 2.0
  wp.atomic_add(loss_out, 0, loss_scale * loss)


@wp.kernel
def _cheetah_loss(
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  # In:
  forward: int,
  height: int,
  pitch: int,
  w_forward: float,
  w_height: float,
  w_upright: float,
  w_angular_velocity: float,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  loss = -w_forward * qvel_in[worldid, forward] - w_height * qpos_in[worldid, height]
  upright = 2.0 * (1.0 - wp.cos(qpos_in[worldid, pitch]))
  loss += w_upright * upright + w_angular_velocity * qvel_in[worldid, pitch] ** 2.0
  wp.atomic_add(loss_out, 0, loss_scale * loss)


@wp.kernel
def _cartpole_loss(
  # In:
  init_qpos: wp.array2d[float],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  # In:
  cart: int,
  pole: int,
  w_angle: float,
  w_pole_velocity: float,
  w_cart_position: float,
  w_cart_velocity: float,
  step: wp.array[int],
  horizon: int,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  state_weight = float(step[0] + 1) / float(horizon)
  target_angle = 2.0 * wp.pi * wp.floor(init_qpos[worldid, pole] / (2.0 * wp.pi) + 0.5)
  angle_error = qpos_in[worldid, pole] - target_angle
  loss = w_angle * angle_error * angle_error
  loss += state_weight * (
    w_pole_velocity * qvel_in[worldid, pole] * qvel_in[worldid, pole]
    + w_cart_position * qpos_in[worldid, cart] * qpos_in[worldid, cart]
    + w_cart_velocity * qvel_in[worldid, cart] * qvel_in[worldid, cart]
  )
  wp.atomic_add(loss_out, 0, loss_scale * loss)


@wp.kernel
def _reacher_loss(
  # Model:
  body_pos: wp.array2d[wp.vec3],
  target_body: int,
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  qvel_in: wp.array2d[float],
  # In:
  site_ids: wp.array[int],
  step: wp.array[int],
  horizon: int,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  if step[0] != horizon - 1:
    return
  tip = site_xpos_in[worldid, site_ids[0]]
  target = body_pos[worldid, target_body]
  dx = tip[0] - target[0]
  dy = tip[1] - target[1]
  shoulder_velocity = qvel_in[worldid, 0]
  wrist_velocity = qvel_in[worldid, 1]
  cost = dx * dx + dy * dy + shoulder_velocity * shoulder_velocity + wrist_velocity * wrist_velocity
  wp.atomic_add(loss_out, 0, loss_scale * cost)


def cheetah_loss(task, m, d, d_out, step, horizon, loss_scale, loss):
  del m, d, step, horizon
  indices, weights = task.cfg.indices, task.cfg.weights
  wp.launch(
    _cheetah_loss,
    dim=d_out.qpos.shape[0],
    inputs=[
      d_out.qpos,
      d_out.qvel,
      indices["forward"],
      indices["height"],
      indices["pitch"],
      weights["forward"],
      weights["height"],
      weights["upright"],
      weights["angular_velocity"],
      loss_scale,
    ],
    outputs=[loss],
  )


def ant_loss(task, m, d, d_out, step, horizon, loss_scale, loss):
  del m, horizon
  indices, weights = task.cfg.indices, task.cfg.weights
  wp.launch(
    _ant_loss,
    dim=d_out.qpos.shape[0],
    inputs=[
      task.init_qpos,
      d_out.qpos,
      d_out.qvel,
      d.ctrl,
      indices["forward"],
      indices["height"],
      indices["quaternion"],
      task.sim_dt,
      weights["forward_velocity"],
      weights["lateral_velocity"],
      weights["vertical_velocity"],
      weights["height"],
      weights["fall"],
      weights["upright"],
      weights["heading"],
      weights["angular_velocity"],
      weights["gait"],
      weights["control"],
      step,
      loss_scale,
    ],
    outputs=[loss],
  )


def cartpole_loss(task, m, d, d_out, step, horizon, loss_scale, loss):
  del m, d
  indices, weights = task.cfg.indices, task.cfg.weights
  wp.launch(
    _cartpole_loss,
    dim=d_out.qpos.shape[0],
    inputs=[
      task.init_qpos,
      d_out.qpos,
      d_out.qvel,
      indices["cart"],
      indices["pole"],
      weights["angle"],
      weights["pole_velocity"],
      weights["cart_position"],
      weights["cart_velocity"],
      step,
      horizon,
      loss_scale,
    ],
    outputs=[loss],
  )


def reacher_loss(task, m, d, d_out, step, horizon, loss_scale, loss):
  del d
  wp.launch(
    _reacher_loss,
    dim=d_out.qpos.shape[0],
    inputs=[
      m.body_pos,
      task.target_body,
      d_out.site_xpos,
      d_out.qvel,
      task.site_ids,
      step,
      horizon,
      loss_scale,
    ],
    outputs=[loss],
  )


def ant_progress(cfg, model, data):
  qpos = data.qpos
  forward = cfg.indices["forward"]
  displacement = qpos[:, -1, forward] - qpos[:, 0, forward]
  target_error = np.minimum(displacement, _ANT_TARGET_DISTANCE) - _ANT_TARGET_DISTANCE
  normalized_error = target_error**2.0
  return 1.0 / (1.0 + normalized_error) ** 2.0


def cheetah_progress(cfg, model, data):
  qpos = data.qpos
  forward = cfg.indices["forward"]
  displacement = qpos[:, -1, forward] - qpos[:, 0, forward]
  target_error = np.minimum(displacement, _CHEETAH_TARGET_DISTANCE) - _CHEETAH_TARGET_DISTANCE
  normalized_error = target_error**2.0
  return 1.0 / (1.0 + normalized_error) ** 2.0


def cartpole_progress(cfg, model, data):
  qpos = data.qpos
  pole = cfg.indices["pole"]
  angle = qpos[:, -1, pole]
  angle_error = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
  normalized_error = angle_error**2.0
  return 1.0 / (1.0 + normalized_error) ** 2.0


def reacher_progress(cfg, model, data):
  del cfg
  shoulder, wrist = data.qpos[:, -1].T
  upper_arm = model.mjm.body("hand").pos[0]
  forearm = model.mjm.body("finger").pos[0]
  forearm_angle = shoulder + wrist
  tip = model.mjm.body("arm").pos[:2] + np.column_stack(
    (
      upper_arm * np.cos(shoulder) + forearm * np.cos(forearm_angle),
      upper_arm * np.sin(shoulder) + forearm * np.sin(forearm_angle),
    )
  )
  # NOTE: alternatively, could call fwd_kinematics to compute the tip site position.
  target_body = model.body_names.index("target")
  targets = np.asarray(model.body_pos)[..., target_body, :2]
  normalized_error = np.sum((tip - targets) ** 2.0, axis=1) + np.sum(data.qvel[:, -1] ** 2.0, axis=1)
  return 1.0 / (1.0 + normalized_error) ** 2.0


_LOSS = {
  "ant": ant_loss,
  "cheetah": cheetah_loss,
  "cartpole": cartpole_loss,
  "reacher": reacher_loss,
}

_PROGRESS = {
  "ant": ant_progress,
  "cheetah": cheetah_progress,
  "cartpole": cartpole_progress,
  "reacher": reacher_progress,
}
