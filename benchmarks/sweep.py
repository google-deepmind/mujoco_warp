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

"""sweep.py: runs benchmarks across a range of commits.

Supports two directions:
  forward  - benchmark commits after the last known SHA
  back     - benchmark commits before the earliest known SHA

Each direction accepts an optional target:
  (omitted)  forward sweeps to HEAD, back sweeps to root
  N          process exactly N commits
  <sha>      process commits up to (or back to) a specific commit

Results are stored in per-benchmark JSONL files on the gh-pages branch,
maintained in chronological order. The benchmarked commit range is tracked
in commit_range.json with "from" and "to" fields.

Usage:
  python benchmarks/sweep.py forward                # Sweep to HEAD
  python benchmarks/sweep.py forward 5              # Sweep 5 commits forward
  python benchmarks/sweep.py forward abc123f        # Sweep to specific commit
  python benchmarks/sweep.py back 20                # Sweep back 20 commits
  python benchmarks/sweep.py back abc123f           # Sweep back to specific commit
  python benchmarks/sweep.py back                   # Sweep back to root
  python benchmarks/sweep.py forward -f humanoid    # Filter by name
  python benchmarks/sweep.py forward --mock         # Quick test
"""

import argparse
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Iterable

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


def _uv_run(*args, cwd: Path | None = None):
  """Run a uv command, returning CompletedProcess."""
  log.info("Command: uv run %s", " ".join(args))
  return subprocess.run(("uv", "run") + args, cwd=cwd, check=True, capture_output=True, text=True)


# benchmark discovery, assembly, and execution


def _discover_benchmarks(input_dir: str) -> Iterable[dict]:
  """Discover benchmarks from __init__.py modules under benchmarks_dir."""
  benchmarks_dir = Path(input_dir) / "benchmarks"

  if benchmarks_dir.as_posix() not in sys.path:
    sys.path.append(benchmarks_dir.as_posix())

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
        dst = benchmark_dir / dst_path / path.parts[offset]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(path, dst, dirs_exist_ok=True)
    else:
      shutil.copytree(repo_dir / repo_path, benchmark_dir / dst_path, dirs_exist_ok=True)

  # copy benchmark module files on top
  shutil.copytree(bm["_dir"], benchmark_dir, dirs_exist_ok=True)


def _run_benchmark(bm: dict, input_dir: Path, *, mock: bool) -> dict:
  """Run a single benchmark via uv, returning parsed JSON."""
  mjcf_path = Path(_ARGS.assets_root) / bm["name"] / bm["mjcf"]
  cmd = [
    "mjwarp-testspeed",
    mjcf_path.as_posix(),
    f"--nworld={1 if mock else bm['nworld']}",
    f"--clear_warp_cache={not mock}",
    "--format=json",
    "--event_trace=true",
    "--memory=true",
    "--measure_solver=true",
    "--measure_alloc=true",
  ]
  for field in ("nconmax", "njmax", "function", "render_width", "render_height"):
    if field in bm:
      cmd.append(f"--{field}={bm[field]}")
  if "replay" in bm:
    replay_path = Path(_ARGS.assets_root) / bm["name"] / bm["replay"]
    cmd.append(f"--replay={replay_path.as_posix()}")
  if mock:
    cmd.append("--nstep=10")
  elif "nstep" in bm:
    cmd.append(f"--nstep={bm['nstep']}")

  return json.loads(_uv_run(*cmd, cwd=input_dir).stdout)


def _sweep(input_dir: str, output_dir: str):
  """Run the benchmark sweep."""
  # read commit range
  range_file = Path(output_dir) / "nightly" / "commit_range.json"
  if not range_file.exists():
    log.error("No commit_range.json found at %s", range_file)
    sys.exit(1)

  commit_range = json.loads(range_file.read_text())
  log.info("Current commit range: %s..%s", commit_range["from"][:12], commit_range["to"][:12])

  # determine commits to process
  if _ARGS.direction == "forward":
    end = "HEAD" if _ARGS.target.isdigit() else _ARGS.target
    result = _git("rev-list", "--reverse", f"{commit_range['to']}..{end}", cwd=input_dir)
  else:
    if _ARGS.target == "root" or _ARGS.target.isdigit():
      result = _git("rev-list", f"{commit_range['from']}^", cwd=input_dir)
    else:
      result = _git("rev-list", f"{_ARGS.target}^..{commit_range['from']}^", cwd=input_dir)

  commits = result.stdout.strip().splitlines()
  if _ARGS.target.isdigit():
    commits = commits[: int(_ARGS.target)]

  log.info("Found %d commit(s) to process (%s).", len(commits), _ARGS.direction)

  if not commits:
    return

  # when running backwards, only discover benchmarks once at the start
  # when running forwards, re-discover on every commit (to catch new benchmarks)
  benchmarks: dict[str, dict] = {}

  for i, commit in enumerate(commits):
    log.info("[%d/%d] Processing commit %s", i + 1, len(commits), commit)

    _git("restore", "--staged", "--worktree", ".", cwd=input_dir)
    _git("checkout", commit, cwd=input_dir)

    # get commit timestamp (UTC ISO 8601)
    timestamp = _git("log", "-1", "--format=%cd", "--date=format-local:%Y-%m-%dT%H:%M:%S+00:00", commit, cwd=input_dir)

    # discover and assemble benchmarks
    if _ARGS.direction == "forward" or i == 0:
      benchmarks = {}
      for bm in _discover_benchmarks(input_dir):
        _assemble_benchmark(bm)
        benchmarks[bm["name"]] = bm

      log.info("Discovered %d benchmarks for commit %s: [%s]", len(benchmarks), commit, ", ".join(benchmarks.keys()))

    for name, bm in benchmarks.items():
      log.info("Running benchmark: %s", name)
      try:
        result = _run_benchmark(bm, input_dir, mock=_ARGS.mock)
      except Exception as e:
        log.error("Benchmark %s failed: %s", name, e)
        continue
      result["commit"] = commit
      result["timestamp"] = timestamp.stdout.strip()
      path = Path(output_dir) / "nightly" / f"{name}.jsonl"
      line = json.dumps(result) + "\n"
      if _ARGS.direction == "forward":
        with path.open("a") as f:
          f.write(line)
      else:
        text = path.read_text() if path.exists() else ""
        path.write_text(line + text)
      log.info("Benchmark %s completed.", name)

    # update commit range after each commit for crash safety
    commit_range["to" if _ARGS.direction == "forward" else "from"] = commit
    range_file.write_text(json.dumps(commit_range, indent=2) + "\n")


def main():
  global _ARGS
  parser = argparse.ArgumentParser(description="Sweep benchmarks forward or backward across commits.")
  parser.add_argument(
    "--input", default="git@github.com:google-deepmind/mujoco_warp.git", help="git uri or path to mujoco_warp repo to benchmark"
  )
  parser.add_argument(
    "--output",
    default="git@github.com:google-deepmind/mujoco_warp.git#gh-pages",
    help="git uri or path to repo to write benchmark results",
  )
  parser.add_argument("-f", "--filter", default=".*", help="filter benchmarks by name (regex)")
  parser.add_argument("--mock", action="store_true", help="run with nworld=1 nstep=10 for fast testing")
  parser.add_argument("--dry_run", action="store_true", help="run benchmarks but don't push results")
  parser.add_argument("--assets_root", default="/tmp/benchmark_assets", help="root directory to assemble benchmark assets")
  sub = parser.add_subparsers(dest="direction", required=True)
  fwd = sub.add_parser("forward", help="sweep from last benchmarked commit toward HEAD")
  fwd.add_argument("target", nargs="?", default="HEAD", help="commit SHA or N (default: HEAD)")
  bwd = sub.add_parser("back", help="sweep backward from earliest benchmarked commit")
  bwd.add_argument("target", nargs="?", default="root", help="commit SHA or N (default: root)")

  _ARGS = parser.parse_args()

  def clone_if_needed(uri):
    if ":" not in uri:
      return uri
    path = tempfile.mkdtemp(prefix="mjwarp-sweep")
    spec = uri.rsplit("#", 1)
    if len(spec) < 2:
      _git("clone", spec[0], path)
    else:
      _git("clone", spec[0], path, "--branch", spec[1])
    return path

  input_dir = clone_if_needed(_ARGS.input)
  output_dir = clone_if_needed(_ARGS.output)

  _sweep(input_dir, output_dir)

  if _ARGS.dry_run:
    log.info("Dry run — temp dirs left for inspection: input_dir=%s output_dir=%s", input_dir, output_dir)
    return

  # push results if output was a repo
  if ":" in _ARGS.output and _git("status", "--porcelain", cwd=output_dir).stdout:
    log.info("Pushing changes from %s back to %s.", output_dir, _ARGS.output)
    _git("add", ".", cwd=output_dir)
    msg = f"Update benchmarks ({_ARGS.direction}) - {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}"
    _git("commit", "-m", msg, cwd=output_dir)
    _git("push", "origin", *_ARGS.output.rsplit("#", 1)[1:], cwd=output_dir)

  if ":" in _ARGS.input:
    shutil.rmtree(input_dir, ignore_errors=True)
  if ":" in _ARGS.output:
    shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
  main()
