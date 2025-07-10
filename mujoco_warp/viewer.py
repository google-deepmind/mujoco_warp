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

"""An example integration of MJWarp with the MuJoCo viewer."""

import logging
import pickle
import time
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp
from absl import app
from absl import flags

import mujoco_warp as mjwarp

_MODEL_PATH = flags.DEFINE_string("mjcf", None, "Path to a MuJoCo MJCF file.", required=True)
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)")
_ENGINE = flags.DEFINE_enum("engine", "mjwarp", ["mjwarp", "mjc"], "Simulation engine")
_CONE = flags.DEFINE_enum("cone", "pyramidal", ["pyramidal", "elliptic"], "Friction cone type")
_LS_PARALLEL = flags.DEFINE_bool("ls_parallel", False, "Engine solver with parallel linesearch")
_VIEWER_GLOBAL_STATE = {
  "running": True,
  "step_once": False,
}
_NCONMAX = flags.DEFINE_integer("nconmax", None, "Maximum number of contacts.")
_NJMAX = flags.DEFINE_integer("njmax", None, "Maximum number of constraints.")
_BROADPHASE = flags.DEFINE_integer("broadphase", None, "Broadphase collision routine.")
_BROADPHASE_FILTER = flags.DEFINE_integer("broadphase_filter", None, "Broadphase collision filter routine.")
_KEYFRAME = flags.DEFINE_integer("keyframe", None, "Keyframe to initialize simulation.")


def key_callback(key: int) -> None:
  if key == 32:  # Space bar
    _VIEWER_GLOBAL_STATE["running"] = not _VIEWER_GLOBAL_STATE["running"]
    logging.info("RUNNING = %s", _VIEWER_GLOBAL_STATE["running"])
  elif key == 46:  # period
    _VIEWER_GLOBAL_STATE["step_once"] = True


def _load_model():
  spec = mujoco.MjSpec.from_file(_MODEL_PATH.value)
  # check if the file has any mujoco.sdf test plugins
  if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
    from mujoco_warp.test_data.collision_sdf.utils import register_sdf_plugins as register_sdf_plugins

    register_sdf_plugins(mjwarp.collision_sdf)
  return spec.compile()


def _compile_step(m, d):
  mjwarp.step(m, d)
  # double warmup to work around issues with compilation during graph capture:
  mjwarp.step(m, d)
  # capture the whole step function as a CUDA graph
  with wp.ScopedCapture() as capture:
    mjwarp.step(m, d)
  return capture.graph


def _main(argv: Sequence[str]) -> None:
  """Launches MuJoCo passive viewer fed by MJWarp."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(f"Loading model from: {_MODEL_PATH.value}.")
  if _MODEL_PATH.value.endswith(".mjb"):
    mjm = mujoco.MjModel.from_binary_path(_MODEL_PATH.value)
  else:
    mjm = _load_model()
  if _CONE.value == "pyramidal":
    mjm.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  elif _CONE.value == "elliptic":
    mjm.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  mjd = mujoco.MjData(mjm)
  if _KEYFRAME.value is not None:
    mujoco.mj_resetDataKeyframe(mjm, mjd, _KEYFRAME.value)
  mujoco.mj_forward(mjm, mjd)

  if _ENGINE.value == "mjc":
    print("Engine: MuJoCo C")
  else:  # mjwarp
    print("Engine: MuJoCo Warp")
    mjm_hash = pickle.dumps(mjm)
    m = mjwarp.put_model(mjm)
    m.opt.ls_parallel = _LS_PARALLEL.value
    if _BROADPHASE.value is not None:
      m.opt.broadphase = _BROADPHASE.value
    if _BROADPHASE_FILTER.value is not None:
      m.opt.broadphase_filter = _BROADPHASE_FILTER.value

    d = mjwarp.put_data(mjm, mjd, nconmax=_NCONMAX.value, njmax=_NJMAX.value)

    if _CLEAR_KERNEL_CACHE.value:
      wp.clear_kernel_cache()

    print("Compiling the model physics step...")
    start = time.time()
    graph = _compile_step(m, d)
    elapsed = time.time() - start
    print(f"Compilation took {elapsed}s.")

  viewer = mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback)
  with viewer:
    while True:
      start = time.time()

      if _ENGINE.value == "mjc":
        mujoco.mj_step(mjm, mjd)
      else:  # mjwarp
        wp.copy(d.ctrl, wp.array([mjd.ctrl.astype(np.float32)]))
        wp.copy(d.act, wp.array([mjd.act.astype(np.float32)]))
        wp.copy(d.xfrc_applied, wp.array([mjd.xfrc_applied.astype(np.float32)]))
        wp.copy(d.qpos, wp.array([mjd.qpos.astype(np.float32)]))
        wp.copy(d.qvel, wp.array([mjd.qvel.astype(np.float32)]))
        wp.copy(d.time, wp.array([mjd.time], dtype=wp.float32))

        hash = pickle.dumps(mjm)
        if hash != mjm_hash:
          mjm_hash = hash
          m = mjwarp.put_model(mjm)
          graph = _compile_step(m, d)

        if _VIEWER_GLOBAL_STATE["running"]:
          wp.capture_launch(graph)
          wp.synchronize()
        elif _VIEWER_GLOBAL_STATE["step_once"]:
          _VIEWER_GLOBAL_STATE["step_once"] = False
          wp.capture_launch(graph)
          wp.synchronize()

        mjwarp.get_data_into(mjd, mjm, d)

      viewer.sync()

      elapsed = time.time() - start
      if elapsed < mjm.opt.timestep:
        time.sleep(mjm.opt.timestep - elapsed)


def main():
  app.run(_main)


if __name__ == "__main__":
  main()
