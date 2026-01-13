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

"""Get MyoSim from GitHub."""

import subprocess
import sys
from typing import Sequence

from absl import app
from etils import epath

_MYOSIM_PATH = epath.Path(__file__).parent.parent / "myo_sim"

_MYOSIM_COMMIT_SHA = "33f3ded946f55adbdcf963c99999587aadaf975f"


# TODO(team): shared utility with benchmark/kitchen/populate_scene._clone?
def _clone(repo_url: str, target_path: str, commit_sha: str) -> None:
  """Clone a git repo with progress bar."""
  process = subprocess.Popen(
    ["git", "clone", "--progress", repo_url, target_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
  )

  while True:
    # Read output line by line.
    if not process.stderr.readline() and process.poll() is not None:
      break

  if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, ["git", "clone"])

  # checkout specific commit
  print(f"Checking out commit {commit_sha}")
  subprocess.run(["git", "-C", target_path, "checkout", commit_sha], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main(argv: Sequence[str]):
  """Ensure MyoSim exists, downloading it if necessary."""
  print(f"myosim path: {_MYOSIM_PATH}")
  if not _MYOSIM_PATH.exists():
    print("MyoSim not found. Downloading...")

    try:
      _clone("https://github.com/MyoHub/myo_sim.git", str(_MYOSIM_PATH), _MYOSIM_COMMIT_SHA)
      print("Successfully downloaded MyoSim")
    except subprocess.CalledProcessError as e:
      print(f"Error downloading MyoSim: {e}", file=sys.stderr)
      raise


if __name__ == "__main__":
  app.run(main)
