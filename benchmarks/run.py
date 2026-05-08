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
  python benchmarks/run.py --input git@github.com:google-deepmind/mujoco_warp.git#abc123f
  python benchmarks/run.py --input .
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

_ARGS = None  # module level variable that gets populated with argparse results

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)


# external commands


def _git(*args, cwd: Path | None = None, check: bool = True):
  """Run a git command, returning CompletedProcess."""
  env = os.environ.copy()
  env["TZ"] = "UTC"
  ssh_key = Path.home() / ".ssh" / "id_ed25519_mujoco_warp_nightly"
  if ssh_key.exists():
    env["GIT_SSH_COMMAND"] = f'ssh -i "{ssh_key}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
  log.info("Command: git %s", " ".join(args))
  return subprocess.run(("git",) + args, cwd=cwd, env=env, check=check, capture_output=True, text=True)


def _uv_run(*args, cwd: Path | None = None):
  """Run a uv command, returning CompletedProcess."""
  log.info("Command: uv run %s", " ".join(args))
  return subprocess.run(("uv", "run") + args, cwd=cwd, check=True, capture_output=True, text=True)


# benchmark discovery, assembly, and execution


def _discover_benchmarks(input_dir: str):
  """Discover benchmarks from __init__.py modules under input_dir/benchmarks."""
  benchmarks_dir = Path(input_dir) / "benchmarks"

  if benchmarks_dir.as_posix() not in sys.path:
    sys.path.insert(0, benchmarks_dir.as_posix())

  importlib.invalidate_caches()

  for benchmark in sorted(benchmarks_dir.iterdir()):
    if not (benchmark / "__init__.py").exists():
      continue
    if benchmark.name in sys.modules:
      module = importlib.reload(sys.modules[benchmark.name])
    else:
      module = importlib.import_module(benchmark.name)
    for bm in getattr(module, "BENCHMARKS", []):
      if re.match(_ARGS.filter, bm["name"]):
        bm["_dir"] = benchmark
        yield bm


def _assemble_benchmark(bm: dict):
  """Assemble benchmark files into assets root."""
  benchmark_dir = Path(_ARGS.assets_root) / bm["name"]
  if benchmark_dir.exists():
    shutil.rmtree(benchmark_dir)
  benchmark_dir.mkdir(parents=True)

  for asset_spec in bm.get("assets", []):
    repo, repo_path, dst_path = (asset_spec + ("",))[:3]

    # repo clones are stored in the format: <assets_root>/_git/<repo_source>/<repo_ref>
    repo_dir = Path(_ARGS.assets_root) / "_git" / Path(repo["source"]).stem / repo["ref"]
    if not repo_dir.exists():
      repo_dir.mkdir(parents=True, exist_ok=True)
      _git("clone", repo["source"], repo_dir.as_posix(), "--depth", "1", "--revision", repo["ref"])

    if "*" in repo_path:
      parts = Path(repo_path).parts
      offset = parts.index("*") - len(parts)
      for path in sorted(repo_dir.glob(repo_path)):
        if not path.is_dir():
          continue
        dest = benchmark_dir / dst_path / path.parts[offset]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(path, dest, dirs_exist_ok=True)
    else:
      shutil.copytree(repo_dir / repo_path, benchmark_dir / dst_path, dirs_exist_ok=True)

  # copy benchmark module files on top
  shutil.copytree(bm["_dir"], benchmark_dir, dirs_exist_ok=True)


def _run_benchmark(bm: dict, input_dir: Path) -> dict:
  """Run a single benchmark via uv, returning parsed JSON."""
  mjcf_path = Path(_ARGS.assets_root) / bm["name"] / bm["mjcf"]
  cmd = [
    "mjwarp-testspeed",
    mjcf_path.as_posix(),
    f"--nworld={bm['nworld']}",
    f"--clear_warp_cache={_ARGS.clear_warp_cache}",
    "--format=short",
    "--event_trace=true",
    "--memory=true",
    "--measure_solver=true",
    "--measure_alloc=true",
  ]
  if "nconmax" in bm:
    cmd.append(f"--nconmax={bm['nconmax']}")
  if "njmax" in bm:
    cmd.append(f"--njmax={bm['njmax']}")
  if "replay" in bm:
    replay_path = Path(_ARGS.assets_root) / bm["name"] / bm["replay"]
    cmd.append(f"--replay={replay_path.as_posix()}")
  if "nstep" in bm:
    cmd.append(f"--nstep={bm['nstep']}")

  result = _uv_run(*cmd, cwd=input_dir)

  # parse short-format output into a dict
  data = {}
  for line in result.stdout.splitlines():
    if not line.strip():
      continue
    parts = line.split()
    if len(parts) >= 2:
      data[parts[0]] = " ".join(parts[1:])
  return data


def main():
  global _ARGS
  parser = argparse.ArgumentParser(description="Run MuJoCo Warp benchmarks.")
  parser.add_argument("--input", default=".", help="git uri or path to mujoco_warp repo to benchmark")
  parser.add_argument("-f", "--filter", default=".*", help="filter benchmarks by name (regex)")
  parser.add_argument("--assets_root", default="/tmp/benchmark_assets", help="root directory to assemble benchmark assets")
  parser.add_argument(
    "--clear_warp_cache",
    default=True,
    type=lambda v: v.lower() not in ("false", "0"),
    help="clear warp caches (kernel, LTO, CUDA compute)",
  )

  _ARGS = parser.parse_args()

  def clone_if_needed(uri):
    if ":" not in uri:
      return uri
    path = tempfile.mkdtemp(prefix="mjwarp-run-")
    spec = uri.rsplit("#", 1)
    if len(spec) < 2:
      _git("clone", spec[0], path)
    else:
      _git("clone", spec[0], path, "--branch", spec[1])
    return path

  input_dir = clone_if_needed(_ARGS.input)

  try:
    benchmarks = {}
    for bm in _discover_benchmarks(input_dir):
      _assemble_benchmark(bm)
      benchmarks[bm["name"]] = bm

    log.info("Discovered %d benchmarks: [%s]", len(benchmarks), ", ".join(benchmarks.keys()))

    for name, bm in benchmarks.items():
      log.info("Running benchmark: %s", name)
      try:
        data = _run_benchmark(bm, input_dir)
      except subprocess.CalledProcessError as e:
        log.error("Benchmark %s failed:\n%s", name, e.stderr)
        continue
      for key, value in data.items():
        print(f"{name}.{key} {value}")
  except Exception:
    log.exception("Run failed — temp dir left for diagnosis: input_dir=%s", input_dir)
    sys.exit(1)

  # clean up cloned temp dir on success
  if ":" in _ARGS.input:
    log.info("Cleaning up temp dir %s...", input_dir)
    shutil.rmtree(input_dir, ignore_errors=True)


if __name__ == "__main__":
  main()
