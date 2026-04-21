import importlib
import os
import re
import shutil
import subprocess
import tempfile
from typing import Sequence

import mujoco
import warp as wp
from absl import app
from absl import flags

import mujoco_warp as mjw
from mujoco_warp._src.io import find_keys
from mujoco_warp._src.io import make_trajectory

_FILTER = flags.DEFINE_string("filter", "", "filter benchmarks by name (regex)", short_name="f")
_CACHE_DIR = flags.DEFINE_string("cache_dir", "mjwarp_benchmarks_cache", "directory to cache benchmark assets")


def _asset_dir(asset: dict) -> str:
  """Returns a base dir for an asset uri in the cache."""
  uri = asset["source"]
  if uri.endswith(".git"):
    ref = asset["ref"]
    return os.path.join(os.path.basename(uri).replace(".git", ""), ref)
  else:
    raise ValueError(f"Unsupported asset uri: {uri}")


def _asset_fetch(asset: dict, dst_dir: str):
  uri = asset["source"]
  if uri.endswith(".git"):
    ref = asset["ref"]
    subprocess.run(["git", "clone", uri, dst_dir, "--depth", "1", "--revision", ref], check=True)
  else:
    raise ValueError(f"Unsupported asset uri: {uri}")


def _main(argv: Sequence[str]):
  script_dir = os.path.dirname(os.path.abspath(__file__))
  cache_dir = os.path.join(tempfile.gettempdir(), _CACHE_DIR.value)

  wp.config.quiet = True
  wp.init()

  # Find all directories in benchmarks/
  for item in os.listdir(script_dir):
    item_path = os.path.join(script_dir, item)
    if not os.path.isdir(item_path) or not os.path.exists(os.path.join(item_path, "__init__.py")):
      continue
    module = importlib.import_module(item)

    for asset in getattr(module, "ASSETS", []):
      asset_dir = os.path.join(cache_dir, _asset_dir(asset))
      if not os.path.exists(asset_dir):
        _asset_fetch(asset, asset_dir)

    for bm in getattr(module, "BENCHMARKS", []):
      name = bm["name"]

      if _FILTER.value and not re.search(_FILTER.value, name):
        continue

      benchmark_dir = os.path.join(cache_dir, name)
      if os.path.exists(benchmark_dir):
        shutil.rmtree(benchmark_dir)
      os.makedirs(benchmark_dir)

      # copy in any assets
      for asset, subpath in bm.get("assets", []):
        shutil.copytree(os.path.join(cache_dir, _asset_dir(asset), subpath), benchmark_dir, dirs_exist_ok=True)

      # copy in benchmark files
      shutil.copytree(item_path, benchmark_dir, dirs_exist_ok=True)

      xml_path = os.path.join(benchmark_dir, bm["mjcf"])

      mjm = mujoco.MjModel.from_xml_path(xml_path)
      mjd = mujoco.MjData(mjm)

      ctrls = None
      if "replay" in bm:
        keys = find_keys(mjm, bm["replay"])
        if not keys:
          raise app.UsageError(f"Key prefix not found: {bm['replay']}")
        ctrls = make_trajectory(mjm, keys)
        mujoco.mj_resetDataKeyframe(mjm, mjd, keys[0])
      elif mjm.nkey > 0 and _KEYFRAME.value > -1:
        mujoco.mj_resetDataKeyframe(mjm, mjd, _KEYFRAME.value)
        if ctrls is None:
          ctrls = [mjd.ctrl.copy() for _ in range(_NSTEP.value)]

      m, d, metrics = mjw.benchmark.init(mjm, mjd, bm["nworld"], bm["nconmax"], bm["njmax"])
      metrics = mjw.benchmark.run(mjw.step, m, d, metrics, bm.get("nstep", 1000), None, True)

      for metric in metrics.to_short().split("\n"):
        print(f"{name}.{metric}")


if __name__ == "__main__":
  app.run(_main)
