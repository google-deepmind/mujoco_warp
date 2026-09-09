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

"""common.py: command and asset helpers shared by run.py and sweep.py.

Python puts the running script's own directory on sys.path, so the import resolves as long as this
file sits next to the script, whether that is a repo checkout or the directory the systemd nightly
installs into (see contrib/systemd/README.md). Keep this module to the standard library for the
same reason: the nightly runs it outside any repo, and shells out to uv for everything else.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("mjwarp-benchmarks")


def git(*args, cwd: Path | str | None = None, check: bool = True):
  """Run a git command, returning CompletedProcess."""
  env = os.environ.copy()
  env["TZ"] = "UTC"
  ssh_key = Path.home() / ".ssh" / "id_ed25519_mujoco_warp_nightly"
  if ssh_key.exists():
    env["GIT_SSH_COMMAND"] = f'ssh -i "{ssh_key}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
  log.info("Command: git %s", " ".join(str(a) for a in args))
  return subprocess.run(("git",) + tuple(args), cwd=cwd, env=env, check=check, capture_output=True, text=True)


def uv_run(*args, cwd: Path | str | None = None):
  """Run a uv command, returning CompletedProcess."""
  log.info("Command: uv run %s", " ".join(str(a) for a in args))
  return subprocess.run(("uv", "run") + tuple(args), cwd=cwd, check=True, capture_output=True, text=True)


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


def ensure_pinned_clone(source: str, ref: str, dst: Path | str):
  """Make dst a shallow checkout of ref from source, reusing it if it already is one."""
  dst = Path(dst)
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
