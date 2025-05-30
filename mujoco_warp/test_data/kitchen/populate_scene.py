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

"""Populate a scene with a robot from MuJoCo Menagerie."""

import os
import subprocess
import sys
from typing import Sequence

import mujoco
from absl import app
from absl import flags
from etils import epath

# The script path
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# menagerie path is used to load robot assets
# resource paths do not have glob implemented, so we use epath.Path
_MENAGERIE_PATH = epath.Path(__file__).parent.parent / "mujoco_menagerie"

# commit sha of the mujoco menagerie github repository
_MENAGERIE_COMMIT_SHA = "14ceccf557cc47240202f2354d684eca58ff8de4"

_MENAGERIE_ROBOTS = {
  "panda": "franka_emika_panda/mjx_panda.xml",
  "fr3": "franka_fr3/fr3.xml",
  "google_robot": "google_robot/robot.xml",
  "gen3": "kinova_gen3/gen3.xml",
  "iiwa_14": "kuka_iiwa_14/iiwa14.xml",
  "tiago": "pal_tiago/tiago.xml",
  "sawyer": "rethink_robotics_sawyer/sawyer.xml",
  "vx300": "trossen_vx300s/vx300s.xml",
  "arm100": "trs_so_arm100/so_arm100.xml",
  "lite6": "ufactory_lite6/lite6.xml",
  "xarm7": "ufactory_xarm7/xarm7.xml",
  "z1": "unitree_z1/z1.xml",
  "ur10e": "universal_robots_ur10e/ur10e.xml",
  "ur5e": "universal_robots_ur5e/ur5e.xml",
  "berkeley_humanoid": "berkeley_humanoid/berkeley_humanoid.xml",
  "t1": "booster_t1/t1.xml",
  "h1": "unitree_h1/h1.xml",
  "g1": "unitree_g1/g1.xml",
  # TODO(team): Investigate why the robot is crashing
  #"talos": "pal_talos/talos.xml",
  "op3": "robotis_op3/op3.xml",
  "spot": "boston_dynamics_spot/spot.xml",
  "anymal_b": "anybotics_anymal_b/anymal_b.xml",
  "anymal_c": "anybotics_anymal_c/anymal_c.xml",
  "barkour_v0": "google_barkour_v0/barkour_v0.xml",
  "a1": "unitree_a1/a1.xml",
  "go1": "unitree_go1/go1.xml",
  "go2": "unitree_go2/go2.xml",
  # TODO(team): Comment this out after the magnetometer sensor has been implemented
  #"cassie": "agility_cassie/cassie.xml",
}

_INPUT = flags.DEFINE_string("input", _SCRIPT_DIR + "/kitchen.xml", "the input scene to populate")
_OUTPUT = flags.DEFINE_string("output", "kitchen_robot.xml", "filename to save the populated scene")
_ROBOT = flags.DEFINE_enum("robot", "g1", _MENAGERIE_ROBOTS.keys(), "the robot to use")


def main(argv: Sequence[str]):
  """Populates an environment with robot from MuJoCo Menagerie."""

  input_path = epath.Path(_INPUT.value)
  if not input_path.exists():
    raise FileNotFoundError("could not load file: {_INPUT.value}")

  robot_path = _load_from_menagerie(_MENAGERIE_ROBOTS[_ROBOT.value])

  # create directory with kitchen + robot assets
  input_dir = input_path.parents[0]
  combined_assets_path = f"{input_dir}/combined_assets/{_ROBOT.value}"
  subprocess.run(f"mkdir -p {combined_assets_path}", shell=True, text=True)
  subprocess.run(f"cp -r {input_dir}/assets {combined_assets_path}", shell=True, text=True)
  # TODO(team): robot without assets (eg, humanoid)
  subprocess.run(f"cp -r {os.path.dirname(robot_path)}/assets {combined_assets_path}", shell=True, text=True)

  # create xml
  spec = mujoco.MjSpec.from_file(input_path.as_posix())
  spec_xml = spec.to_xml().replace("assets/", f"{combined_assets_path}/assets/")
  spec = mujoco.MjSpec.from_string(spec_xml)
  robot = mujoco.MjSpec.from_file(robot_path.as_posix())

  # add robot to environment
  attach_frame = spec.worldbody.add_frame(pos=[1.5, -1.5, 0.0])
  spec.attach(robot, frame=attach_frame, prefix="robot")
  spec_xml = spec.to_xml()  # write to file

  ## Saving the model to xml
  with open(_OUTPUT.value, "w", encoding="utf-8") as f:
    f.write(spec_xml)


def _load_from_menagerie(asset_path: str) -> str:
  """Load an asset from the mujoco menagerie."""
  # Ensure menagerie exists, and otherwise clone it
  _menagerie_exists()
  return _MENAGERIE_PATH / asset_path


def _menagerie_exists() -> None:
  """Ensure mujoco_menagerie exists, downloading it if necessary."""
  if not _MENAGERIE_PATH.exists():
    print("mujoco_menagerie not found. Downloading...")

    try:
      _clone("https://github.com/deepmind/mujoco_menagerie.git", str(_MENAGERIE_PATH), _MENAGERIE_COMMIT_SHA)
      print("Successfully downloaded mujoco_menagerie")
    except subprocess.CalledProcessError as e:
      print(f"Error downloading mujoco_menagerie: {e}", file=sys.stderr)
      raise


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


if __name__ == "__main__":
  app.run(main)
