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

"""run.py: runs MuJoCo Warp benchmarks.

Usage: python benchmarks/run.py [flags]

Example:
  python benchmarks/run.py -f humanoid
"""

import importlib
import os
import re
import shutil
import subprocess
from typing import Sequence

from absl import app
from absl import flags

_FILTER = flags.DEFINE_string("filter", "", "filter benchmarks by name (regex)", short_name="f")
_ASSET_BASE = flags.DEFINE_string("assets", "/tmp/benchmark_assets", "directory to assemble benchmark assets")
_CLEAR_WARP_CACHE = flags.DEFINE_bool("clear_warp_cache", True, "clear warp caches (kernel, LTO, CUDA compute)")


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
    subprocess.run(
      ["git", "clone", uri, dst_dir, "--depth", "1", "--revision", ref],
      check=True,
    )
  else:
    raise ValueError(f"Unsupported asset uri: {uri}")


def _main(argv: Sequence[str]):
  script_dir = os.path.dirname(os.path.abspath(__file__))

  # Find all directories in benchmarks/
  for item in os.listdir(script_dir):
    item_path = os.path.join(script_dir, item)
    if not os.path.isdir(item_path) or not os.path.exists(os.path.join(item_path, "__init__.py")):
      continue
    module = importlib.import_module(item)

    for asset in getattr(module, "ASSETS", []):
      asset_dir = os.path.join(_ASSET_BASE.value, _asset_dir(asset))
      if not os.path.exists(asset_dir):
        _asset_fetch(asset, asset_dir)

    for bm in getattr(module, "BENCHMARKS", []):
      name = bm["name"]
      nstep = bm.get("nstep", 1000)

      if _FILTER.value and not re.search(_FILTER.value, name):
        continue

      benchmark_dir = os.path.join(_ASSET_BASE.value, name)
      if os.path.exists(benchmark_dir):
        shutil.rmtree(benchmark_dir)
      os.makedirs(benchmark_dir)

      for asset_spec in bm.get("assets", []):
        asset, src_subpath, dst_subpath = (asset_spec + ("",))[:3]
        src_path = os.path.join(_ASSET_BASE.value, _asset_dir(asset), src_subpath)
        dst_path = os.path.join(benchmark_dir, dst_subpath)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

      # copy in benchmark files
      shutil.copytree(item_path, benchmark_dir, dirs_exist_ok=True)

      xml_path = os.path.join(benchmark_dir, bm["mjcf"])

      # Build command for testspeed
      cmd = [
        "mjwarp-testspeed",
        xml_path,
        f"--nworld={bm['nworld']}",
        f"--nstep={nstep}",
        f"--clear_warp_cache={_CLEAR_WARP_CACHE.value}",
        "--format=short",
        "--event_trace=true",
        "--memory=true",
        "--measure_solver=true",
        "--measure_alloc=true",
      ]
      for field in ("nconmax", "njmax", "replay"):
        if field in bm:
          cmd.append(f"--{field}={bm[field]}")

      # Run testspeed
      result = subprocess.run(cmd, capture_output=True, text=True)
      if result.returncode != 0:
        print(f"Error running benchmark {name}:")
        print(result.stderr)
        continue

      # Parse output
      for line in result.stdout.splitlines():
        if not line.strip():
          continue
        parts = line.split()
        if len(parts) >= 2:
          key = parts[0]
          value = " ".join(parts[1:])
          print(f"{name}.{key} {value}")


if __name__ == "__main__":
  app.run(_main)
