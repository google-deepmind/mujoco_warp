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

"""mjwarp-record: record video of MuJoCo Warp rollouts.

Usage: mjwarp-record <mjcf XML path> --video <output_path> [flags]

Example:
  mjwarp-record benchmarks/humanoid/humanoid.xml --video humanoid.mp4 --nworld 1
"""

import inspect
import sys
from typing import Sequence

import mujoco
import warp as wp
from absl import app
from absl import flags
from etils import epath
from PIL import Image

import mujoco_warp as mjw
from mujoco_warp._src import cli

_FUNCS = {
  n: f
  for n, f in inspect.getmembers(mjw, inspect.isfunction)
  if inspect.signature(f).parameters.keys() == {"m", "d"} or inspect.signature(f).parameters.keys() == {"m", "d", "rc"}
}

_OUTPUT = flags.DEFINE_string("output", None, "output video file path", required=True)
_FPS = flags.DEFINE_integer("fps", 30, "frames per second for the video")
_QUALITY = flags.DEFINE_integer("quality", 70, "quality setting for webp/gif (0-100)")
_CAM_DISTANCE = flags.DEFINE_float("cam_distance", 1.5, "camera distance coefficient (multiplier for model extent)")
_CAM_LOOKAT_Z = flags.DEFINE_float("cam_lookat_z", None, "camera lookat z value (absolute); default: mjm.stat.center[2]")
_CAM_AZIMUTH_SPEED = flags.DEFINE_float("cam_azimuth_speed", 0.05, "camera azimuth orbit speed (degrees per step)")


def _main(argv: Sequence[str]):
  """Run the recorder."""
  if len(argv) < 2:
    raise app.UsageError("Missing required input: mjcf path.")
  elif len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")

  wp.config.quiet = flags.FLAGS["verbosity"].value < 1
  wp.init()

  path = epath.Path(argv[1])
  print(f"Loading model from: {path}...\n")
  mjm = cli.load_model(path)
  m, d, rc, ctrls = cli.init_structs(mjw.step, mjm)

  frames = []
  renderer = mujoco.Renderer(mjm, height=cli.RENDER_HEIGHT.value, width=cli.RENDER_WIDTH.value)
  render_every = max(1, int(1.0 / (_FPS.value * mjm.opt.timestep)))

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = mjm.stat.center
  if _CAM_LOOKAT_Z.value is not None:
    cam.lookat[2] = _CAM_LOOKAT_Z.value
  cam.distance = mjm.stat.extent * _CAM_DISTANCE.value
  cam.elevation = -20
  mjd = mujoco.MjData(mjm)

  def callback(step, trace, latency):
    del trace, latency
    if step % render_every != 0:
      return
    # TODO(team): add support for rendering more than one world (overlaid or tiled)
    mjd.qpos[:] = d.qpos.numpy()[0]
    mjd.qvel[:] = d.qvel.numpy()[0]
    mujoco.mj_forward(mjm, mjd)
    # symmetric orbit
    cam.azimuth = 90 + (step - cli.NSTEP.value / 2) * _CAM_AZIMUTH_SPEED.value
    renderer.update_scene(mjd, camera=cam)
    frames.append(renderer.render())

  print(f"Recording {cli.NSTEP.value} steps...")
  cli.unroll(mjw.step, m, d, rc, callback, ctrls)

  print(f"Saving video to {_OUTPUT.value}...")
  if _OUTPUT.value.endswith((".gif", ".webp")):
    frames = [Image.fromarray(f) for f in frames]
    frames[0].save(
      _OUTPUT.value,
      save_all=True,
      append_images=frames[1:],
      duration=int(render_every * mjm.opt.timestep * 1000),
      loop=0,
      # minimize_size=True,
      quality=_QUALITY.value,
    )
  else:
    raise ValueError(f"Unsupported video format: {_OUTPUT.value}")


def main():
  # absl flags assumes __main__ is the main running module for printing usage documentation
  # pyproject bin scripts break this assumption, so manually set argv and docstring
  sys.argv[0] = "mujoco_warp.record"
  sys.modules["__main__"].__doc__ = __doc__
  # default to single world with no noise
  flags.FLAGS.set_default("nworld", 1)
  flags.FLAGS.set_default("noise_std", 0.0)
  flags.FLAGS.set_default("noise_rate", 0.0)
  flags.FLAGS.set_default("render_width", 320)
  flags.FLAGS.set_default("render_height", 240)
  app.run(_main)


if __name__ == "__main__":
  main()
