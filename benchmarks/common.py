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

"""common.py: helpers shared by the benchmark scripts in this directory.

run.py and sweep.py both import this module. Python puts the running script's own directory on
sys.path, so the import resolves as long as this file sits next to the script, whether that is a
repo checkout or the install directory the systemd nightly uses (see contrib/systemd/README.md).
Keep this module to the standard library for the same reason: the nightly runs it outside any
repo, and shells out to uv for everything that needs mujoco_warp itself.
"""

import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

# Ensure the active virtual environment's bin directory is in PATH so 'uv' can be found
_venv_bin = Path(sys.executable).parent.as_posix()
if _venv_bin not in os.environ.get("PATH", ""):
  os.environ["PATH"] = f"{_venv_bin}{os.path.pathsep}{os.environ.get('PATH', '')}"

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
log = logging.getLogger("mjwarp-benchmarks")


# external commands


def git(*args, cwd: Path | None = None, check: bool = True):
  """Run a git command, returning CompletedProcess."""
  env = os.environ.copy()
  env["TZ"] = "UTC"
  ssh_key = Path.home() / ".ssh" / "id_ed25519_mujoco_warp_nightly"
  if ssh_key.exists():
    env["GIT_SSH_COMMAND"] = f'ssh -i "{ssh_key}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
  log.info("Command: git %s", " ".join(args))
  return subprocess.run(("git",) + args, cwd=cwd, env=env, check=check, capture_output=True, text=True)


def uv_run(*args, cwd: Path | None = None):
  """Run a uv command, returning CompletedProcess."""
  log.info("Command: uv run %s", " ".join(args))
  return subprocess.run(("uv", "run") + args, cwd=cwd, check=True, capture_output=True, text=True)


def ensure_pinned_clone(source: str, ref: str, dst: Path):
  """Make dst a shallow checkout of ref from source, reusing it if it already is one."""
  if (dst / ".git").exists():
    return
  # a dst without .git is a leftover from an interrupted fetch, rebuild it rather than trust it.
  # rmtree only handles directories, so clear a file or symlink at dst separately or the rename
  # below fails
  if dst.is_dir() and not dst.is_symlink():
    shutil.rmtree(dst, ignore_errors=True)
  elif dst.exists() or dst.is_symlink():
    dst.unlink(missing_ok=True)
  # "git clone --revision" does this in one step but needs git >= 2.49, newer than the git in
  # current LTS distros. fetch into a sibling directory and rename it into place so a failed or
  # interrupted fetch cannot leave a partial dst that later runs mistake for a good checkout.
  dst.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix=f".{dst.name}.", dir=dst.parent) as tmp_dir:
    staging = Path(tmp_dir)
    git("init", "--quiet", staging.as_posix())
    git("fetch", "--quiet", "--depth", "1", source, ref, cwd=staging)
    git("checkout", "--quiet", "FETCH_HEAD", cwd=staging)
    staging.rename(dst)


def clone_if_needed(uri: str, prefix: str) -> str:
  """Clone uri into a temp dir if it is a git uri, returning a local path either way."""
  if ":" not in uri:
    return uri
  path = tempfile.mkdtemp(prefix=prefix)
  spec = uri.rsplit("#", 1)
  if len(spec) < 2:
    git("clone", spec[0], path)
  else:
    git("clone", spec[0], path, "--branch", spec[1])
  return path


# benchmark discovery and assembly


def discover_benchmarks(input_dir: str, name_filter: str) -> Iterable[dict]:
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
      if re.match(name_filter, bm["name"]):
        bm["_dir"] = benchmark
        yield bm


def assemble_benchmark(bm: dict, assets_root: str):
  """Assemble benchmark files into assets root."""
  benchmark_dir = Path(assets_root) / bm["name"]
  if benchmark_dir.exists():
    shutil.rmtree(benchmark_dir)
  benchmark_dir.mkdir(parents=True)

  for asset_spec in bm.get("assets", []):
    repo, repo_path, dst_path = (asset_spec + ("",))[:3]

    # repo clones are stored in the format: <assets_root>/_git/<repo_source>/<repo_ref>
    repo_dir = Path(assets_root) / "_git" / Path(repo["source"]).stem / repo["ref"]
    ensure_pinned_clone(repo["source"], repo["ref"], repo_dir)

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
