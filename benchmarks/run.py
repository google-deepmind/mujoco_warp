#!/usr/bin/python3

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

"""run.py: runs MuJoCo Warp benchmarks.

Usage: python benchmarks/run.py [flags]

Example:
  python benchmarks/run.py -f humanoid
  python benchmarks/run.py --commit abc123f
  python benchmarks/run.py --local
"""

import argparse
import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "git@github.com:google-deepmind/mujoco_warp.git"

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)


def _git(*args, cwd: Path | None = None):
  """Run a git command, returning CompletedProcess."""
  env = os.environ.copy()
  env["TZ"] = "UTC"
  ssh_key = Path.home() / ".ssh" / "id_ed25519_mujoco_warp_nightly"
  if ssh_key.exists():
    env["GIT_SSH_COMMAND"] = f'ssh -i "{ssh_key}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
  log.info("Command: git %s", " ".join(args))
  return subprocess.run(("git",) + args, cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _uv_run(*args, cwd: Path | None = None):
  """Run a uv command, returning CompletedProcess."""
  env = os.environ.copy()
  # bypass corporate index settings that may interfere with uv
  env["UV_DEFAULT_INDEX"] = "https://pypi.org/simple"
  env["PIP_INDEX_URL"] = "https://pypi.org/simple"
  log.info("Command: uv run %s", " ".join(args))
  return subprocess.run(("uv", "run") + args, cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _asset_dir(asset: dict) -> Path:
  """Returns a base dir for an asset uri in the cache."""
  uri = asset["source"]
  if uri.endswith(".git"):
    return Path(uri.split("/")[-1].replace(".git", "")) / asset["ref"]
  raise ValueError(f"Unsupported asset uri: {uri}")


def _asset_fetch(asset: dict, dst_dir: Path):
  uri = asset["source"]
  if uri.endswith(".git"):
    _git("clone", uri, str(dst_dir), "--depth", "1", "--revision", asset["ref"])
  else:
    raise ValueError(f"Unsupported asset uri: {uri}")


def main():
  parser = argparse.ArgumentParser(description="Run MuJoCo Warp benchmarks.")
  parser.add_argument("-f", "--filter", default="", help="filter benchmarks by name (regex)")
  parser.add_argument("--assets", default="/tmp/benchmark_assets", help="directory to assemble benchmark assets")
  parser.add_argument(
    "--clear_warp_cache",
    default=True,
    type=lambda v: v.lower() not in ("false", "0"),
    help="clear warp caches (kernel, LTO, CUDA compute)",
  )
  parser.add_argument("--commit", default=None, help="checkout a specific commit SHA (implies clone)")
  parser.add_argument("--local", action="store_true", help="run from local checkout instead of cloning")
  args = parser.parse_args()

  asset_base = Path(args.assets)

  if args.local:
    work_dir = Path(__file__).resolve().parent.parent
    log.info("Running from local checkout: %s", work_dir)
  else:
    work_dir = Path(tempfile.mkdtemp(prefix="mujoco_warp-run-"))
    log.info("Cloning mujoco_warp to %s...", work_dir)
    _git("clone", REPO_URL, str(work_dir))
    if args.commit:
      _git("checkout", args.commit, cwd=work_dir)
      log.info("Checked out commit %s", args.commit)

  benchmarks_dir = work_dir / "benchmarks"
  if benchmarks_dir.as_posix() not in sys.path:
    sys.path.insert(0, benchmarks_dir.as_posix())

  # discover and run benchmarks
  for item_path in sorted(benchmarks_dir.iterdir()):
    if not item_path.is_dir() or not (item_path / "__init__.py").exists():
      continue

    name = item_path.name
    if name in sys.modules:
      module = importlib.reload(sys.modules[name])
    else:
      module = importlib.import_module(name)

    for asset in getattr(module, "ASSETS", []):
      asset_dir = asset_base / _asset_dir(asset)
      if not asset_dir.exists():
        _asset_fetch(asset, asset_dir)

    for bm in getattr(module, "BENCHMARKS", []):
      bm_name = bm["name"]

      if args.filter and not re.search(args.filter, bm_name):
        continue

      benchmark_dir = asset_base / bm_name
      if benchmark_dir.exists():
        shutil.rmtree(benchmark_dir)
      benchmark_dir.mkdir(parents=True)

      for asset_spec in bm.get("assets", []):
        asset, src_subpath, dst_subpath = (asset_spec + ("",))[:3]
        src_root = asset_base / _asset_dir(asset)
        if "*" in src_subpath:
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
        f"--clear_warp_cache={args.clear_warp_cache}",
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
      if "nstep" in bm:
        cmd.append(f"--nstep={bm['nstep']}")

      # Run testspeed via uv in the work_dir
      try:
        result = _uv_run(*cmd, cwd=work_dir)
      except subprocess.CalledProcessError as e:
        log.error("Benchmark %s failed:\n%s", bm_name, e.stderr)
        continue

      # Parse output
      for line in result.stdout.splitlines():
        if not line.strip():
          continue
        parts = line.split()
        if len(parts) >= 2:
          key = parts[0]
          value = " ".join(parts[1:])
          print(f"{bm_name}.{key} {value}")


if __name__ == "__main__":
  main()
