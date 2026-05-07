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

REPO_URL = "git@github.com:google-deepmind/mujoco_warp.git"
RESULTS_BRANCH = "gh-pages"


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
  env = os.environ.copy()
  # bypass corporate index settings that may interfere with uv
  env["UV_DEFAULT_INDEX"] = "https://pypi.org/simple"
  env["PIP_INDEX_URL"] = "https://pypi.org/simple"
  log.info("Command: uv run %s", " ".join(args))
  return subprocess.run(("uv", "run") + args, cwd=cwd, env=env, check=True, capture_output=True, text=True)


# asset management


def _asset_dir(asset: dict) -> Path:
  """Returns a cache-relative dir for an asset."""
  uri = asset["source"]
  if uri.endswith(".git"):
    return Path(uri.split("/")[-1].replace(".git", "")) / asset["ref"]
  raise ValueError(f"Unsupported asset uri: {uri}")


def _asset_fetch(asset: dict, dst_dir: Path):
  """Clone an asset repository."""
  uri = asset["source"]
  if uri.endswith(".git"):
    _git("clone", uri, str(dst_dir), "--depth", "1", "--revision", asset["ref"])
  else:
    raise ValueError(f"Unsupported asset uri: {uri}")


# benchmark discovery, assembly, and execution


def _discover_benchmarks(benchmarks_dir: Path) -> Iterable[dict]:
  """Discover benchmarks from __init__.py modules under benchmarks_dir.

  Falls back to config.txt for older commits that predate the modular layout.
  Handles re-discovery across git checkouts by reloading previously imported
  benchmark modules.
  """
  if benchmarks_dir.as_posix() not in sys.path:
    sys.path.append(benchmarks_dir.as_posix())

  importlib.invalidate_caches()

  for item_path in sorted(benchmarks_dir.iterdir()):
    if not item_path.is_dir() or not (item_path / "__init__.py").exists():
      continue
    name = item_path.name
    if name in sys.modules:
      # reload to pick up changes from the new checkout
      module = importlib.reload(sys.modules[name])
    else:
      module = importlib.import_module(name)
    assets = getattr(module, "ASSETS", [])
    for bm in getattr(module, "BENCHMARKS", []):
      bm["_module_dir"] = item_path
      bm["_assets"] = assets
      if "replay" in bm:
        bm["replay"] = str(item_path / bm["replay"])
      yield bm


def _assemble_benchmark(bm: dict, asset_base: Path) -> Path:
  """Assemble benchmark files into asset_base/<name>/, returns MJCF path."""
  name = bm["name"]
  module_dir = bm["_module_dir"]
  benchmark_dir = asset_base / name

  if benchmark_dir.exists():
    shutil.rmtree(benchmark_dir)
  benchmark_dir.mkdir(parents=True)

  # fetch external assets
  for asset in bm["_assets"]:
    asset_dir = asset_base / _asset_dir(asset)
    if not asset_dir.exists():
      _asset_fetch(asset, asset_dir)

  # copy asset files into benchmark dir
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

  # copy benchmark module files on top
  shutil.copytree(module_dir, benchmark_dir, dirs_exist_ok=True)
  return benchmark_dir / bm["mjcf"]


def _run_benchmark(bm: dict, xml_path: Path, benchmark_dir: Path, work_dir: Path, *, mock: bool) -> dict | None:
  """Run a single benchmark via uv, returning parsed JSON or None."""
  name = bm["name"]
  nworld = 1 if mock else bm["nworld"]

  cmd = [
    "mjwarp-testspeed",
    str(xml_path),
    f"--nworld={nworld}",
    f"--clear_warp_cache={not mock}",
    "--format=json",
    "--event_trace=true",
    "--memory=true",
    "--measure_solver=true",
    "--measure_alloc=true",
  ]
  for field in ("nconmax", "njmax"):
    if field in bm:
      cmd.append(f"--{field}={bm[field]}")
  if "replay" in bm:
    replay_val = bm["replay"]
    if replay_val.endswith(".npz"):
      # discovery resolves replay to work_dir; remap to assembled benchmark_dir
      replay_val = str(benchmark_dir / Path(replay_val).name)
    cmd.append(f"--replay={replay_val}")
  if mock:
    cmd.append(f"--nstep=10")
  elif "nstep" in bm:
    cmd.append(f"--nstep={bm['nstep']}")

  try:
    result = _uv_run(*cmd, cwd=work_dir)
  except subprocess.CalledProcessError as e:
    log.error("Benchmark %s failed:\n%s", name, e.stderr)
    return None

  # parse JSON from last line of stdout (uv may emit log lines before it)
  try:
    return json.loads(result.stdout.strip().splitlines()[-1])
  except (json.JSONDecodeError, IndexError) as e:
    log.error("Failed to parse JSON for %s: %s", name, e)
    return None


def main():
  parser = argparse.ArgumentParser(description="Sweep benchmarks forward or backward across commits.")
  parser.add_argument("-f", "--filter", default="", help="filter benchmarks by name (regex)")
  parser.add_argument("--mock", action="store_true", help="run with nworld=1 nstep=10 for fast testing")
  parser.add_argument("--dry_run", action="store_true", help="run benchmarks but don't push results")
  parser.add_argument("--results_dir", default="", help="path to gh-pages nightly/ directory (auto-cloned if empty)")
  parser.add_argument("--assets", default="/tmp/benchmark_assets", help="directory to assemble benchmark assets")

  sub = parser.add_subparsers(dest="direction", required=True)
  fwd = sub.add_parser("forward", help="sweep from last benchmarked commit toward HEAD")
  fwd.add_argument("target", nargs="?", default="HEAD", help="commit SHA or N (default: HEAD)")
  bwd = sub.add_parser("back", help="sweep backward from earliest benchmarked commit")
  bwd.add_argument("target", nargs="?", default="root", help="commit SHA or N (default: root)")

  args = parser.parse_args()
  forward = args.direction == "forward"

  # parse target: integer count or commit SHA
  count, target_sha = None, None
  if args.target is not None:
    try:
      count = int(args.target)
    except ValueError:
      target_sha = args.target

  # benchmarks are discovered from the work_dir checkout, not locally
  asset_base = Path(args.assets)

  # clone a fresh copy of the repo to benchmark against
  work_dir = Path(tempfile.mkdtemp(prefix="mujoco_warp-sweep-"))
  log.info("Cloning mujoco_warp to %s...", work_dir)
  _git("clone", REPO_URL, str(work_dir))

  # set up results directory (either local path or fresh clone of gh-pages)
  results_repo = None
  if args.results_dir:
    results_path = Path(args.results_dir)
  else:
    results_repo = Path(tempfile.mkdtemp(prefix="mujoco_warp-results-"))
    log.info("Cloning results branch to %s...", results_repo)
    _git("clone", "--branch", RESULTS_BRANCH, "--depth", "1", REPO_URL, str(results_repo))
    results_path = results_repo / "nightly"

  # read commit range
  range_file = results_path / "commit_range.json"
  if not range_file.exists():
    log.error("No commit_range.json found at %s", range_file)
    sys.exit(1)

  commit_range = json.loads(range_file.read_text())
  log.info("Current range: %s..%s", commit_range["from"][:12], commit_range["to"][:12])

  # determine commits to process
  if forward:
    end = "HEAD" if args.target.isdigit() else args.target
    result = _git("rev-list", "--reverse", f"{commit_range['to']}..{end}", cwd=work_dir)
  else:
    if args.target == "root" or args.target.isdigit():
      result = _git("rev-list", f"{commit_range['from']}^", cwd=work_dir)
    else:
      result = _git("rev-list", f"{target_sha}^..{commit_range['from']}^", cwd=work_dir)

  commits = result.stdout.strip().splitlines()
  if args.target.isdigit():
    commits = commits[: int(args.target)]

  log.info("Found %d commit(s) to process (%s).", len(commits), args.direction)

  if not commits:
    return

  # run benchmarks and write results after each one
  # backward sweeps: discover + assemble once from the newest commit (first
  # in the list), then reuse for every older commit.
  cached_benchmarks: dict[str, tuple[dict, Path]] | None = None

  for i, commit in enumerate(commits):
    log.info("[%d/%d] Processing commit %s", i + 1, len(commits), commit)

    _git("restore", "--staged", "--worktree", ".", cwd=work_dir)
    _git("checkout", commit, cwd=work_dir)

    # get commit timestamp (UTC ISO 8601)
    ts = _git("log", "-1", "--format=%cd", "--date=format-local:%Y-%m-%dT%H:%M:%S+00:00", commit, cwd=work_dir)
    commit_timestamp = ts.stdout.strip()

    # discover and assemble benchmarks
    if forward or cached_benchmarks is None:
      cached_benchmarks = {}
      for bm in _discover_benchmarks(work_dir / "benchmarks"):
        if args.filter and not re.search(args.filter, bm["name"]):
          continue
        xml_path = _assemble_benchmark(bm, asset_base)
        cached_benchmarks[bm["name"]] = (bm, xml_path)

    for name, (bm, xml_path) in cached_benchmarks.items():
      log.info("Running benchmark: %s", name)

      benchmark_dir = asset_base / name
      data = _run_benchmark(bm, xml_path, benchmark_dir, work_dir, mock=args.mock)
      if data is None:
        continue

      data["commit"] = commit
      data["timestamp"] = commit_timestamp
      path = results_path / f"{name}.jsonl"
      text = path.read_text() if path.exists() else ""
      if forward:
        path.write_text(text + json.dumps(data) + "\n")
      else:
        path.write_text(json.dumps(data) + "\n" + text)

      log.info("Benchmark %s completed.", name)

    # update commit range after each commit for crash safety
    commit_range["to" if forward else "from"] = commit
    range_file.write_text(json.dumps(commit_range, indent=2) + "\n")
    log.info("Updated commit range: %s..%s", commit_range["from"][:12], commit_range["to"][:12])

  # push results
  if args.dry_run:
    log.info("Dry run - results written to %s but not pushed.", results_path)
    return

  log.info("Committing and pushing results...")
  _git("add", "nightly/*.jsonl", "nightly/commit_range.json", cwd=results_repo)

  diff = _git("diff", "--staged", "--quiet", cwd=results_repo, check=False)
  if diff.returncode == 0:
    log.info("No changes to commit.")
    return

  msg = f"Update benchmarks ({args.direction}) - {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}"
  _git("commit", "-m", msg, cwd=results_repo)
  _git("push", "origin", RESULTS_BRANCH, cwd=results_repo)
  log.info("Successfully pushed results to %s", RESULTS_BRANCH)


if __name__ == "__main__":
  main()
