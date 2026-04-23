# Copyright 2025 The Newton Developers
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

import mediapy as media
import mujoco
import warp as wp
from absl import app
from absl import flags
from etils import epath
from PIL import Image

import mujoco_warp as mjw
from mujoco_warp._src.io import find_keys
from mujoco_warp._src.io import make_trajectory

_FUNCS = {
  n: f
  for n, f in inspect.getmembers(mjw, inspect.isfunction)
  if inspect.signature(f).parameters.keys() == {"m", "d"} or inspect.signature(f).parameters.keys() == {"m", "d", "rc"}
}

_FUNCTION = flags.DEFINE_enum("function", "step", _FUNCS.keys(), "the function to run")
_NSTEP = flags.DEFINE_integer("nstep", 1000, "number of steps per rollout")
_NWORLD = flags.DEFINE_integer("nworld", 1, "number of parallel rollouts")
from mujoco_warp._src.cli import DEVICE as _DEVICE
from mujoco_warp._src.cli import KEYFRAME as _KEYFRAME
from mujoco_warp._src.cli import NCCDMAX as _NCCDMAX
from mujoco_warp._src.cli import NCONMAX as _NCONMAX
from mujoco_warp._src.cli import NJMAX as _NJMAX
from mujoco_warp._src.cli import NJMAX_NNZ as _NJMAX_NNZ
from mujoco_warp._src.cli import OVERRIDE as _OVERRIDE
from mujoco_warp._src.cli import REPLAY as _REPLAY
from mujoco_warp._src.cli import batch_unroll
from mujoco_warp._src.cli import init_model_data
from mujoco_warp._src.cli import load_model as _load_model

_OUTPUT = flags.DEFINE_string("output", None, "output video file path", required=True)
_FPS = flags.DEFINE_integer("fps", 30, "frames per second for the video")
_WIDTH = flags.DEFINE_integer("width", 640, "render width (pixels)")
_HEIGHT = flags.DEFINE_integer("height", 480, "render height (pixels)")


def _main(argv: Sequence[str]):
  """Run the recorder."""
  if len(argv) < 2:
    raise app.UsageError("Missing required input: mjcf path.")
  elif len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")

  wp.config.quiet = flags.FLAGS["verbosity"].value < 1
  wp.init()

  path = epath.Path(argv[1])
  fn = _FUNCS.get(_FUNCTION.value)
  if fn is None:
    raise app.UsageError(f"Unknown function: {_FUNCTION.value}")

  device = wp.get_device(_DEVICE.value)
  if device == "cpu":
    raise ValueError("recorder available for gpu only")
  wp.set_device(device)

  print(f"Loading model from: {path}...\n")
  mjm = _load_model(path)
  mjd = mujoco.MjData(mjm)
  ctrls = None

  if _REPLAY.value:
    keys = find_keys(mjm, _REPLAY.value)
    if not keys:
      raise app.UsageError(f"Key prefix not found: {_REPLAY.value}")
    ctrls = make_trajectory(mjm, keys)
    mujoco.mj_resetDataKeyframe(mjm, mjd, keys[0])
  elif mjm.nkey > 0 and _KEYFRAME.value > -1:
    mujoco.mj_resetDataKeyframe(mjm, mjd, _KEYFRAME.value)
    if ctrls is None:
      ctrls = [mjd.ctrl.copy() for _ in range(_NSTEP.value)]

  m, d = init_model_data(
    mjm, mjd, _NWORLD.value, _NCONMAX.value, _NJMAX.value, _NJMAX_NNZ.value, _NCCDMAX.value, _OVERRIDE.value, device
  )

  frames = []
  renderer = mujoco.Renderer(mjm, height=_HEIGHT.value, width=_WIDTH.value)
  render_every = max(1, int(1.0 / (_FPS.value * mjm.opt.timestep)))

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = mjm.stat.center
  cam.distance = mjm.stat.extent * 1.5
  cam.elevation = -20

  def callback(step, data):
    if step % render_every != 0:
      return
    # TODO(team): add support for rendering more than one world (overlaid or tiled)
    mjd.qpos[:] = data.qpos.numpy()[0]
    mjd.qvel[:] = data.qvel.numpy()[0]
    mujoco.mj_forward(mjm, mjd)

    # symmetric orbit
    cam.azimuth = 90 + (step - _NSTEP.value / 2) * 0.05

    renderer.update_scene(mjd, camera=cam)
    frames.append(renderer.render())

  print(f"Recording {_NSTEP.value} steps...")
  batch_unroll(fn, m, d, _NSTEP.value, ctrls, False, None, device, False, callback=callback, noise=(0.0, 0.0))

  print(f"Saving video to {_OUTPUT.value}...")
  if _OUTPUT.value.endswith((".gif", ".webp")):
    frames = [Image.fromarray(f) for f in frames]
    frames[0].save(
      _OUTPUT.value,
      save_all=True,
      append_images=frames[1:],
      duration=int(1000 / _FPS.value),
      loop=0,
      minimize_size=True,
      quality=70,
    )
  elif _OUTPUT.value.endswith(".mp4"):
    media.write_video(_OUTPUT.value, frames, fps=_FPS.value)
  else:
    raise ValueError(f"Unsupported video format: {_OUTPUT.value}")


def main():
  sys.argv[0] = "mujoco_warp.record"
  sys.modules["__main__"].__doc__ = __doc__
  app.run(_main)


if __name__ == "__main__":
  main()
