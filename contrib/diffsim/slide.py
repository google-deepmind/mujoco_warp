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
"""Optimize puck launch velocities through frictional contact."""

from dataclasses import dataclass
from dataclasses import field

import _demo
import _viz
import mujoco
import numpy as np
import warp as wp

_HALF_HEIGHT = 0.012
_SPHERE_RADIUS = 0.045
_FRICTION = "0.7 0.005 0.0001"
_SPHERE_FRICTION = "0.7 0.005 0.075"
_TERMINAL_VELOCITY_WEIGHT = 0.1
_INITIAL_POSITION_SPREAD = 0.025
_INITIAL_SPEED_SPREAD = 0.08
_SHAPES = {
  "box": 'type="box" size="0.045 0.045 0.012"',
  "cylinder": 'type="cylinder" size="0.045 0.012"',
  "sphere": f'type="sphere" size="{_SPHERE_RADIUS}"',
}
_HEIGHTS = {
  "box": _HALF_HEIGHT,
  "cylinder": _HALF_HEIGHT,
  "sphere": _SPHERE_RADIUS,
}
_APPEARANCE = {
  "box": 'rgba="0.95 0.75 0.25 1"',
  "cylinder": 'rgba="0.35 0.65 0.95 1"',
  "sphere": 'material="sphere"',
}
_SPHERE_MARKERS = """
      <site name="marker_0" type="cylinder" size="0.0063 0.00054" pos="-0.04311 -0.008955 -0.008055"
            zaxis="-0.963 -0.20 -0.18" rgba="0.015 0.02 0.03 1"/>
      <site name="marker_1" type="cylinder" size="0.0063 0.00054" pos="-0.04311 -0.008955 0.008055"
            zaxis="-0.963 -0.20 0.18" rgba="0.015 0.02 0.03 1"/>
      <site name="marker_2" type="cylinder" size="0.0063 0.00054" pos="-0.042975 0.012555 0"
            zaxis="-0.96 0.28 0" rgba="0.015 0.02 0.03 1"/>
"""
_RINGS = (
  (0.090, 0.0020, "0.85 0.12 0.12 1"),
  (0.068, 0.0026, "0.97 0.97 0.97 1"),
  (0.046, 0.0032, "0.85 0.12 0.12 1"),
  (0.024, 0.0038, "0.97 0.97 0.97 1"),
  (0.009, 0.0044, "0.85 0.12 0.12 1"),
)


@dataclass(frozen=True)
class Puck:
  name: str
  start: tuple[float, float]
  target: tuple[float, float]
  velocity: tuple[float, float]


@dataclass
class SlideArgs(_demo.Args):
  sim_dt: float = 0.004
  integrator: str = "euler"
  iterations: int = 100
  lr: float = 0.1
  num_envs: int = 64

  horizon: int = 150
  geom: str = field(
    default="all",
    metadata={"help": "puck shape or group", "choices": tuple(_SHAPES) + ("both", "all")},
  )
  spread: float = field(default=0.5, metadata={"help": "target fan half-angle in radians"})


def _pucks(geom: str):
  if geom in _SHAPES:
    return [Puck(geom, (-0.20, -0.10), (0.50, 0.30), (1.8, 0.7))]
  if geom == "both":
    return [
      Puck("box", (-0.25, -0.12), (0.45, -0.30), (1.7, -0.5)),
      Puck("cylinder", (-0.25, 0.12), (0.50, 0.28), (1.7, 0.5)),
    ]
  if geom == "all":
    return [
      Puck("box", (-0.25, -0.18), (0.45, -0.34), (1.7, -0.6)),
      Puck("cylinder", (-0.25, 0.0), (0.50, 0.0), (1.8, 0.0)),
      Puck("sphere", (-0.25, 0.18), (0.45, 0.34), (1.7, 0.6)),
    ]
  raise ValueError(f"unknown puck geometry: {geom}")


def _initial_positions(pucks, num_envs: int, sampling: str, seed: int):
  starts = np.asarray([puck.start for puck in pucks])
  if num_envs == 1:
    return starts[None].astype(np.float32)
  return _demo.sample_range(
    num_envs,
    starts - _INITIAL_POSITION_SPREAD,
    starts + _INITIAL_POSITION_SPREAD,
    sampling,
    seed,
  )


def _fan_targets(pucks, num_envs: int, spread: float, sampling: str, seed: int):
  angles = _demo.sample_range(num_envs, -spread, spread, sampling, seed) if num_envs > 1 else np.zeros(1)
  starts = np.asarray([puck.start for puck in pucks])
  offsets = np.asarray([puck.target for puck in pucks]) - starts
  targets = []
  for angle in angles:
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    targets.append(starts + offsets @ rotation.T)
  return np.asarray(targets, dtype=np.float32)


def _initial_velocities(pucks, num_envs: int, sampling: str, seed: int):
  velocity = np.asarray([puck.velocity for puck in pucks])
  if num_envs == 1:
    return velocity[None].astype(np.float32)
  scales = _demo.sample_range(
    num_envs,
    np.full(len(pucks), 1.0 - _INITIAL_SPEED_SPREAD),
    np.full(len(pucks), 1.0 + _INITIAL_SPEED_SPREAD),
    sampling,
    seed,
  )
  return (velocity[None] * scales[..., None]).astype(np.float32)


@wp.kernel
def _set_qvel0(
  # In:
  init_vel: wp.array[float],
  num_pucks: int,
  step: wp.array[int],
  # Data out:
  qvel_out: wp.array2d[float],
):
  worldid, puckid = wp.tid()
  if step[0] != 0:
    return
  index = 2 * (worldid * num_pucks + puckid)
  qvel_out[worldid, 6 * puckid] = init_vel[index]
  qvel_out[worldid, 6 * puckid + 1] = init_vel[index + 1]


@wp.kernel
def _loss(
  # Model:
  body_pos: wp.array2d[wp.vec3],
  target_body: wp.array[int],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  # In:
  num_pucks: int,
  step: wp.array[int],
  horizon: int,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid, puckid = wp.tid()
  if step[0] != horizon - 1:
    return
  position = wp.vec2(qpos_in[worldid, 7 * puckid], qpos_in[worldid, 7 * puckid + 1])
  velocity = wp.vec2(qvel_in[worldid, 6 * puckid], qvel_in[worldid, 6 * puckid + 1])
  target = body_pos[worldid, target_body[puckid]]
  delta = position - wp.vec2(target[0], target[1])
  loss = wp.dot(delta, delta) + _TERMINAL_VELOCITY_WEIGHT * wp.dot(velocity, velocity)
  wp.atomic_add(loss_out, 0, loss_scale * loss)


def progress(args, model, data):
  pucks = _pucks(args.geom)
  targets = np.stack(
    [model.body_pos[:, model.body_names.index(f"target_{puck.name}"), :2] for puck in pucks],
    axis=1,
  )
  final = np.stack([data.qpos[:, -1, 7 * index : 7 * index + 2] for index in range(len(pucks))], axis=1)
  velocity = np.stack([data.qvel[:, -1, 6 * index : 6 * index + 2] for index in range(len(pucks))], axis=1)
  distances = np.linalg.norm(final - targets, axis=2)
  speeds = np.linalg.norm(velocity, axis=2)
  normalized_error = distances**2.0 + speeds**2.0
  return np.mean(1.0 / (1.0 + normalized_error) ** 2.0, axis=1)


class Slide(_demo.Demo):
  Args = SlideArgs
  demo_type = _demo.DemoType.INIT_VAL
  loss_type = _demo.LossType.TERMINAL
  model_fields = ("body_pos",)
  layout = _viz.Layout(columns=8, spacing=1.6)
  camera = _viz.Camera(lookat=(0.15, 0.0, 0.0), distance=2.4, azimuth=70.0, elevation=-40.0)

  def __init__(self, args: SlideArgs):
    self.pucks = _pucks(args.geom)
    self.num_pucks = len(self.pucks)
    self.name = "slide" if args.geom == "all" else f"slide_{args.geom}"
    self.trace_body = tuple(puck.name for puck in self.pucks)
    self.targets = _fan_targets(self.pucks, args.num_envs, args.spread, args.env_sampling, args.seed + 2)
    mjm = mujoco.MjModel.from_xml_string(_model_xml(self.pucks, self.targets[0]))
    super().__init__(args, mjm, args.horizon)

    init_pos = _initial_positions(self.pucks, args.num_envs, args.env_sampling, args.seed)
    init_vel = _initial_velocities(self.pucks, args.num_envs, args.env_sampling, args.seed + 1)
    self.init_vel = wp.array(init_vel.ravel(), dtype=float, requires_grad=True)
    self.params = [self.init_vel]
    self.init_qpos = np.tile(self.mjd.qpos, (args.num_envs, 1)).astype(np.float32)
    for index in range(self.num_pucks):
      self.init_qpos[:, 7 * index : 7 * index + 2] = init_pos[:, index]
    target_body = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"target_{puck.name}") for puck in self.pucks]
    body_pos = np.tile(mjm.body_pos, (args.num_envs, 1, 1)).astype(np.float32)
    body_pos[:, target_body, :2] = self.targets
    self.m.body_pos = wp.array(body_pos, dtype=wp.vec3)
    self.target_body = wp.array(target_body, dtype=int)

  def reset(self):
    d = self.datas[0]
    d.qpos.assign(self.init_qpos)
    d.qvel.zero_()
    d.qacc_warmstart.zero_()
    d.time.zero_()

  def prepare_step(self, d):
    wp.launch(
      _set_qvel0,
      dim=(self.args.num_envs, self.num_pucks),
      inputs=[self.init_vel, self.num_pucks, self.step_index],
      outputs=[d.qvel],
    )

  def step_loss(self, d, d_out):
    wp.launch(
      _loss,
      dim=(self.args.num_envs, self.num_pucks),
      inputs=[
        self.m.body_pos,
        self.target_body,
        d_out.qpos,
        d_out.qvel,
        self.num_pucks,
        self.step_index,
        self.horizon,
        self.loss_scale,
      ],
      outputs=[self.loss],
    )

def _body_xml(puck: Puck, index: int):
  x, y = puck.start
  markers = _SPHERE_MARKERS if puck.name == "sphere" else ""
  contact = 'priority="1" condim="6" solimp="0 0.95 0.001"' if puck.name == "sphere" else ""
  friction = _SPHERE_FRICTION if puck.name == "sphere" else _FRICTION
  return f"""
    <body name="{puck.name}" pos="{x} {y} {_HEIGHTS[puck.name]}">
      <freejoint/>
      <geom name="{puck.name}" {_SHAPES[puck.name]} {contact} mass="0.5" contype="{1 << (index + 1)}"
            conaffinity="1" friction="{friction}" {_APPEARANCE[puck.name]}/>
      {markers}
    </body>
  """


def _target_xml(pucks, targets):
  if targets is None:
    return ""
  return "".join(
    f'<body name="target_{puck.name}" pos="{target[0]} {target[1]} 0">'
    + "".join(
      f'<site name="target_{puck.name}_{index}" type="cylinder" size="{radius} {height}" pos="0 0 {height}" rgba="{rgba}"/>'
      for index, (radius, height, rgba) in enumerate(_RINGS)
    )
    + "</body>"
    for puck, target in zip(pucks, targets)
  )


def _model_xml(pucks, targets=None):
  bodies = "".join(_body_xml(puck, index) for index, puck in enumerate(pucks))
  rings = _target_xml(pucks, targets)
  return f"""
<mujoco>
  <option gravity="0 0 -9.81" iterations="50">
    <flag eulerdamp="disable"/>
  </option>
  <visual>
    {_demo.MENAGERIE_VISUAL}
  </visual>
  <asset>
    {_demo.MENAGERIE_ASSETS}
    <material name="sphere" rgba="0.12 0.34 0.72 1" specular="0.25" shininess="0.2" reflectance="0.05"/>
  </asset>
  <default>
    <geom condim="3" solimp="0 0.9 0.02" solref="0.02 1"/>
  </default>
  <worldbody>
    {_demo.MENAGERIE_LIGHTS}
    <geom name="floor" type="plane" size="5 5 0.01" material="groundplane" contype="1" conaffinity="1"
          friction="{_FRICTION}"/>
    {bodies}
    {rings}
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
  _demo.run(Slide)
