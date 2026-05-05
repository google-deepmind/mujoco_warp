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
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from absl import app
from absl import flags

_FILTER = flags.DEFINE_string("filter", "", "filter benchmarks by name (regex)", short_name="f")
_ASSET_BASE = flags.DEFINE_string("assets", "/tmp/benchmark_assets", "directory to assemble benchmark assets")
_CLEAR_WARP_CACHE = flags.DEFINE_bool("clear_warp_cache", True, "clear warp caches (kernel, LTO, CUDA compute)")


def _asset_dir(asset: dict) -> Path:
  """Returns a base dir for an asset uri in the cache."""
  uri = asset["source"]
  if uri.endswith(".git"):
    return Path(uri.split("/")[-1].replace(".git", "")) / asset["ref"]
  raise ValueError(f"Unsupported asset uri: {uri}")


def _asset_fetch(asset: dict, dst_dir: Path):
  uri = asset["source"]
  if uri.endswith(".git"):
    subprocess.run(
      ["git", "clone", uri, str(dst_dir), "--depth", "1", "--revision", asset["ref"]],
      check=True,
    )
  else:
    raise ValueError(f"Unsupported asset uri: {uri}")


def _main(argv: Sequence[str]):
  script_dir = Path(__file__).resolve().parent
  asset_base = Path(_ASSET_BASE.value)

  # Find all directories in benchmarks/
  for item_path in sorted(script_dir.iterdir()):
    if not item_path.is_dir() or not (item_path / "__init__.py").exists():
      continue
    module = importlib.import_module(item_path.name)

    for asset in getattr(module, "ASSETS", []):
      asset_dir = asset_base / _asset_dir(asset)
      if not asset_dir.exists():
        _asset_fetch(asset, asset_dir)

    for bm in getattr(module, "BENCHMARKS", []):
      name = bm["name"]
      nstep = bm.get("nstep", 1000)

      if _FILTER.value and not re.search(_FILTER.value, name):
        continue

      benchmark_dir = asset_base / name
      if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)
      benchmark_dir.mkdir(parents=True)

      for asset_spec in bm.get("assets", []):
        asset, src_subpath, dst_subpath = (asset_spec + ("",))[:3]
        src_root = asset_base / _asset_dir(asset)
        if "*" in src_subpath:
          # glob: copy each match into dst using the *-matched segment as subdir
          # e.g. "ycb/*/google_64k" → star at index 1 of 3 parts → offset -2
          src_parts = Path(src_subpath).parts
          offset = src_parts.index("*") - len(src_parts)
          for src_path in sorted(src_root.glob(src_subpath)):
            if not src_path.is_dir():
              continue
            segment = src_path.parts[offset]
            dst_path = benchmark_dir / dst_subpath / segment
            dst_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
          dst_path = benchmark_dir / dst_subpath
          dst_path.parent.mkdir(parents=True, exist_ok=True)
          shutil.copytree(src_root / src_subpath, dst_path, dirs_exist_ok=True)

      # copy in benchmark files
      shutil.copytree(item_path, benchmark_dir, dirs_exist_ok=True)

      xml_path = benchmark_dir / bm["mjcf"]

      # Build command for testspeed
      cmd = [
        "mjwarp-testspeed",
        str(xml_path),
        f"--nworld={bm['nworld']}",
        f"--nstep={nstep}",
        f"--clear_warp_cache={_CLEAR_WARP_CACHE.value}",
        "--format=short",
        "--event_trace=true",
        "--memory=true",
        "--measure_solver=true",
        "--measure_alloc=true",
      ]
      for field in ("nconmax", "njmax"):
        if field in bm:
          cmd.append(f"--{field}={bm[field]}")
      if "replay" in bm:
        cmd.append(f"--replay={benchmark_dir / bm['replay']}")

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
