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
  python benchmarks/run.py -f humanoid --view
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

# Ensure the active virtual environment's bin directory is in PATH so 'uv' can be found
_venv_bin = Path(sys.executable).parent.as_posix()
if _venv_bin not in os.environ.get("PATH", ""):
  os.environ["PATH"] = f"{_venv_bin}{os.path.pathsep}{os.environ.get('PATH', '')}"

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


def _ensure_pinned_clone(source: str, ref: str, dst: Path):
  """Make dst a shallow checkout of ref from source, reusing it if it already is one."""
  if (dst / ".git").exists():
    return
  # a dst without .git is a leftover from an interrupted fetch, rebuild it rather than trust it
  shutil.rmtree(dst, ignore_errors=True)
  # "git clone --revision" does this in one step but needs git >= 2.49, newer than the git in
  # current LTS distros. fetch into a sibling directory and rename it into place so a failed or
  # interrupted fetch cannot leave a partial dst that later runs mistake for a good checkout.
  dst.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=f".{dst.name}.", dir=dst.parent))
  try:
    _git("init", "--quiet", staging.as_posix())
    _git("fetch", "--quiet", "--depth", "1", source, ref, cwd=staging)
    _git("checkout", "--quiet", "FETCH_HEAD", cwd=staging)
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise
  staging.rename(dst)


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
    _ensure_pinned_clone(repo["source"], repo["ref"], repo_dir)

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
  benchmark_root = Path(_ARGS.assets_root) / bm["name"]
  cmd = [
    "mjwarp-testspeed",
    (benchmark_root / bm["mjcf"]).as_posix(),
    f"--clear_warp_cache={_ARGS.clear_warp_cache}",
    "--format=short",
    "--event_trace=true",
    "--memory=true",
    "--measure_solver=true",
    "--measure_alloc=true",
  ]
  for field, value in bm.items():
    if field == "replay":
      cmd.append(f"--replay={(benchmark_root / value)}")
    elif field not in ("name", "assets", "mjcf", "_dir"):
      if isinstance(value, (list, tuple)):
        for item in value:
          cmd.append(f"--{field}={item}")
      else:
        cmd.append(f"--{field}={value}")

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


def _view_benchmark(bm: dict, input_dir: Path):
  """Launch mjwarp-viewer for a single benchmark."""
  benchmark_root = Path(_ARGS.assets_root) / bm["name"]
  cmd = [
    "mjwarp-viewer",
    (benchmark_root / bm["mjcf"]).as_posix(),
    "--nworld=1",
  ]
  for field in ("nconmax", "nccdmax", "njmax"):
    if field in bm:
      cmd.append(f"--{field}={bm[field]}")
  if "replay" in bm:
    cmd.append(f"--replay={(benchmark_root / bm['replay'])}")

  log.info("Command: uv run %s", " ".join(cmd))
  subprocess.run(("uv", "run") + tuple(cmd), cwd=input_dir, check=True)


def main():
  global _ARGS
  parser = argparse.ArgumentParser(description="Run MuJoCo Warp benchmarks.")
  parser.add_argument("--input", default=".", help="git uri or path to mujoco_warp repo to benchmark")
  parser.add_argument("-f", "--filter", default=".*", help="filter benchmarks by name (regex)")
  parser.add_argument("--view", action="store_true", help="open the benchmark in mjwarp-viewer instead of running testspeed")
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
  benchmarks = list(_discover_benchmarks(input_dir))

  if _ARGS.view:
    if len(benchmarks) > 1:
      log.warning("--view: multiple benchmarks matched, truncating to first match: %s", benchmarks[0]["name"])
    elif len(benchmarks) == 0:
      log.error("--view: no benchmarks matched the regex filter '%s'", _ARGS.filter)
      sys.exit(1)
    bm = benchmarks[0]
    _assemble_benchmark(bm)
    _view_benchmark(bm, input_dir)
  else:
    log.info("Discovered %d benchmarks: [%s]", len(benchmarks), ", ".join(bm["name"] for bm in benchmarks))
    for bm in benchmarks:
      _assemble_benchmark(bm)
      for key, value in _run_benchmark(bm, input_dir).items():
        print(f"{bm['name']}.{key} {value}")

  # clean up cloned temp dir on success
  if ":" in _ARGS.input:
    log.info("Cleaning up temp dir %s...", input_dir)
    shutil.rmtree(input_dir, ignore_errors=True)


if __name__ == "__main__":
  main()
