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
"""Optimize a ball's initial velocity through floor and wall contacts."""

from dataclasses import dataclass
from dataclasses import field

import _demo
import _viz
import mujoco
import numpy as np
import warp as wp

_START = (0.0, -0.5, 1.0)
_TARGET = (0.0, -2.0, 1.5)
_INITIAL_VELOCITY = (0.0, 4.85, -4.05)


@dataclass
class BounceArgs(_demo.Args):
  sim_dt: float = 0.005
  integrator: str = "euler"
  iterations: int = 100
  lr: float = 0.001
  num_envs: int = 4
  env_sampling: str = "linspace"

  horizon: int = 200
  spread: float = field(default=0.25, metadata={"help": "initial forward velocity spread"})


def _initial_velocities(num_envs: int, spread: float, sampling="linspace", seed=0) -> np.ndarray:
  velocity = np.tile(np.array(_INITIAL_VELOCITY, dtype=np.float32), (num_envs, 1))
  if num_envs > 1:
    velocity[:, 1] += _demo.sample_range(num_envs, -spread, spread, sampling, seed)
  return velocity


@wp.kernel
def _set_qvel0(
  # In:
  init_vel: wp.array[float],
  step: wp.array[int],
  # Data out:
  qvel_out: wp.array2d[float],
):
  worldid, axis = wp.tid()
  if step[0] == 0:
    qvel_out[worldid, axis] = init_vel[worldid * 3 + axis]


@wp.kernel
def _loss(
  # Model:
  geom_pos: wp.array2d[wp.vec3],
  target_geom: int,
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  step: wp.array[int],
  horizon: int,
  loss_scale: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid = wp.tid()
  if step[0] != horizon - 1:
    return
  position = wp.vec3(qpos_in[worldid, 0], qpos_in[worldid, 1], qpos_in[worldid, 2])
  target = geom_pos[worldid % geom_pos.shape[0], target_geom]
  delta = position - target
  loss = wp.dot(delta, delta)
  wp.atomic_add(loss_out, 0, loss_scale * loss)


def progress(args, model, data):
  target = model.geom_pos[model.geom_names.index("target")]
  distance = np.linalg.norm(data.qpos[:, -1, :3] - target, axis=1)
  normalized_error = distance**2.0
  return 1.0 / (1.0 + normalized_error) ** 2.0


class Bounce(_demo.Demo):
  Args = BounceArgs
  name = "bounce"
  demo_type = _demo.DemoType.INIT_VAL
  loss_type = _demo.LossType.TERMINAL
  trace_body = "ball"
  layout = _viz.Layout(columns=4, spacing=3.0)
  camera = _viz.Camera(lookat=(0.0, 0.0, 0.8), distance=8.5, azimuth=50.0, elevation=-18.0)

  def __init__(self, args: BounceArgs):
    super().__init__(args, mujoco.MjModel.from_xml_string(_XML), args.horizon)
    init_vel = _initial_velocities(args.num_envs, args.spread, args.env_sampling, args.seed)
    # Linear velocity only; angular velocity remains zero at init.
    self.init_vel = wp.array(init_vel.ravel(), dtype=float, requires_grad=True)
    self.params = [self.init_vel]
    self.init_qpos = np.tile(self.mjd.qpos, (args.num_envs, 1)).astype(np.float32)
    self.target_geom = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, "target")

  def prepare_step(self, d):
    wp.launch(_set_qvel0, dim=(self.args.num_envs, 3), inputs=[self.init_vel, self.step_index], outputs=[d.qvel])

  def step_loss(self, d, d_out):
    wp.launch(
      _loss,
      dim=self.args.num_envs,
      inputs=[self.m.geom_pos, self.target_geom, d_out.qpos, self.step_index, self.horizon, self.loss_scale],
      outputs=[self.loss],
    )

_XML = f"""
<mujoco>
  <option tolerance="1e-8" iterations="100" ls_iterations="50" gravity="0 0 -9.81"/>
  <visual>
    {_demo.MENAGERIE_VISUAL}
  </visual>
  <asset>
    {_demo.MENAGERIE_ASSETS}
  </asset>
  <default>
    <geom condim="3" friction="0.2" solref="-3000 -2" solimp="0 0.95 0.01"/>
  </default>
  <worldbody>
    {_demo.MENAGERIE_LIGHTS}
    <geom name="floor" type="plane" size="5 5 0.01" material="groundplane"/>
    <geom name="wall" type="box" pos="0 2 1" size="1 0.25 1" rgba="0.48 0.58 0.70 1"/>
    <geom name="target" type="box" pos="{_TARGET[0]} {_TARGET[1]} {_TARGET[2]}" size="0.1 0.1 0.1"
          rgba="0.90 0.44 0.46 0.50" contype="0" conaffinity="0"/>
    <body name="ball" pos="{_START[0]} {_START[1]} {_START[2]}">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1" mass="1" priority="1" solref="-100000 -2"
            solimp="0 0.95 0.001" rgba="0.9 0.7 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
  _demo.run(Bounce)
