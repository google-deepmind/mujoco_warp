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

"""Run kitchen benchmarks with various robots."""

import inspect
import os
import subprocess
from typing import Sequence

import mujoco
import numpy as np
import tqdm
import warp as wp
from absl import app
from absl import flags
from dm_control import mjcf
from etils import epath

import mujoco_warp as mjwarp

# The script path to get correct xml and mesh path
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# The menagerie path is used to load robot assets.
# Resource paths do not have glob implemented, so we use a bare epath.Path.
_MENAGERIE_PATH = epath.Path(__file__).parent.parent / "mujoco_menagerie"
# Commit SHA of the menagerie repo.
_MENAGERIE_COMMIT_SHA = "14ceccf557cc47240202f2354d684eca58ff8de4"

_FUNCTION = flags.DEFINE_enum(
  "function",
  "step",
  [n for n, _ in inspect.getmembers(mjwarp, inspect.isfunction)],
  "the function to run",
)
_ROBOT = flags.DEFINE_enum(
  "robot",
  "mujoco_humanoid",
  [
    # Robotic arms
    "panda",
    "fr3",
    "google_robot",
    "gen3",
    "iiwa_14",
    "tiago",
    "sawyer",
    "vx300",
    "arm100",
    "lite6",
    "xarm7",
    "z1",
    "ur10e",
    "ur5e",
    # Humanoids
    "mujoco_humanoid",
    "berkeley_humanoid",
    "t1",
    "h1",
    "g1",
    "talos",
    "op3",
    # Quadrupedal
    "spot",
    "anymal_b",
    "anymal_c",
    "barkour_v0",
    "a1",
    "go1",
    "go2",
    # Bipedal
    "cassie",
  ],
  "the robot to use",
)
_NSTEP = flags.DEFINE_integer("nstep", 1000, "number of steps per rollout")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 8192, "number of parallel rollouts")
_SOLVER = flags.DEFINE_enum("solver", None, ["cg", "newton"], "Override model constraint solver")
_ITERATIONS = flags.DEFINE_integer("iterations", None, "Override model solver iterations")
_LS_ITERATIONS = flags.DEFINE_integer("ls_iterations", None, "Override model linesearch iterations")
_LS_PARALLEL = flags.DEFINE_bool("ls_parallel", False, "solve with parallel linesearch")
_IS_SPARSE = flags.DEFINE_bool("is_sparse", None, "Override model sparse config")
_CONE = flags.DEFINE_enum("cone", "pyramidal", ["pyramidal", "elliptic"], "Friction cone type")
_NCONMAX = flags.DEFINE_integer(
  "nconmax",
  None,
  "Override default maximum number of contacts in a batch physics step.",
)
_NJMAX = flags.DEFINE_integer(
  "njmax",
  None,
  "Override default maximum number of constraints in a batch physics step.",
)
_OUTPUT = flags.DEFINE_enum("output", "text", ["text", "tsv"], "format to print results")
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)")
_EVENT_TRACE = flags.DEFINE_bool("event_trace", False, "Provide a full event trace")
_MEASURE_ALLOC = flags.DEFINE_bool("measure_alloc", False, "Measure how much of nconmax, njmax is used.")


def _main(argv: Sequence[str]):
  """Runs testpeed function."""
  wp.init()

  ## Processing the selected robot
  match _ROBOT.value:
    case "panda":
      robot_path = _load_from_menagerie("franka_emika_panda/mjx_panda.xml")
    case "fr3":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("franka_fr3/fr3.xml")
    case "google_robot":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("google_robot/robot.xml")
    case "gen3":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("kinova_gen3/gen3.xml")
    case "iiwa_14":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("kuka_iiwa_14/iiwa14.xml")
    case "tiago":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("pal_tiago/tiago.xml")
    case "sawyer":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("rethink_robotics_sawyer/sawyer.xml")
    case "vx300":
      robot_path = _load_from_menagerie("trossen_vx300s/vx300s.xml")
    case "arm100":
      robot_path = _load_from_menagerie("trs_so_arm100/so_arm100.xml")
    case "lite6":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("ufactory_lite6/lite6.xml")
    case "xarm7":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("ufactory_xarm7/xarm7.xml")
    case "z1":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("unitree_z1/z1.xml")
    case "ur10e":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("universal_robots_ur10e/ur10e.xml")
    case "ur5e":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("universal_robots_ur5e/ur5e.xml")
    case "mujoco_humanoid":
      robot_path = epath.Path(_SCRIPT_DIR + "/../humanoid/humanoid.xml")
    case "berkeley_humanoid":
      robot_path = _load_from_menagerie("berkeley_humanoid/berkeley_humanoid.xml")
    case "t1":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("booster_t1/t1.xml")
    case "h1":
      robot_path = _load_from_menagerie("unitree_h1/h1.xml")
    case "g1":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("unitree_g1/g1.xml")
    case "talos":  # Do not run on mujoco_warp, not enough shared memory?
      robot_path = _load_from_menagerie("pal_talos/talos.xml")
    case "op3":
      robot_path = _load_from_menagerie("robotis_op3/op3.xml")
    case "spot":  # Do not run on mujoco_warp, sparse not supported with implicit integrator
      robot_path = _load_from_menagerie("boston_dynamics_spot/spot.xml")
    case "anymal_b":
      robot_path = _load_from_menagerie("anybotics_anymal_b/anymal_b.xml")
    case "anymal_c":
      robot_path = _load_from_menagerie("anybotics_anymal_c/anymal_c.xml")
    case "barkour_v0":
      robot_path = _load_from_menagerie("google_barkour_v0/barkour_v0.xml")
    case "a1":
      robot_path = _load_from_menagerie("unitree_a1/a1.xml")
    case "go1":
      robot_path = _load_from_menagerie("unitree_go1/go1.xml")
    case "go2":
      robot_path = _load_from_menagerie("unitree_go2/go2.xml")
    case "cassie":  # Do not run on mujoco_warp, magnetometer sensor not available
      robot_path = _load_from_menagerie("agility_cassie/cassie.xml")
    case _:
      raise FileNotFoundError(f"Robot provided is unknown.")

  ## Verifying that the two xml models exist
  kitchen_path = epath.Path(_SCRIPT_DIR + "/kitchen_env.xml")
  if not kitchen_path.exists():
    raise FileNotFoundError(f"kitchen_env.xml not found.")
  if not robot_path.exists():
    raise FileNotFoundError(f"robot xml not found.")

  ## Loading the two models and creating a xml string
  # Loading the robot first to not have issue with freeejoint
  mjcf_model = mjcf.from_path(robot_path)
  env_model = mjcf.from_path(kitchen_path)
  # Modify position of the robot
  initial_pose = mjcf_model.worldbody.body[0]._get_attribute("pos")
  if initial_pose is None:
    initial_pose = [0.0, 0.0, 0.0]
  mjcf_model.worldbody.body[0]._set_attribute("pos", [1.5, -1.5, initial_pose[2]])
  # Attach the kitchen environment
  mjcf_model.attach(env_model)
  xml_string = mjcf_model.to_xml_string(filename_with_hash=False)

  ## Post-processing to add proper asset path
  # For each mesh in the original kitchen, search the file attribute and replace it by the
  # path from the original model.
  original_model = mujoco.MjSpec.from_file(kitchen_path.as_posix())
  for i in range(len(original_model.meshes)):
    updated_path = _SCRIPT_DIR + "/assets/" + original_model.meshes[i].file
    start_mesh_string = xml_string.find(original_model.meshes[i].name)
    start_file_string = xml_string.find("file", start_mesh_string) + 6
    end_file_string = xml_string.find('"', start_file_string)
    xml_string = xml_string[:start_file_string] + updated_path + xml_string[end_file_string:]
  # For each mesh in the original robot model, search the file attribute and replace it by the
  # path from the original model.
  original_model = mujoco.MjSpec.from_file(robot_path.as_posix())
  for i in range(len(original_model.meshes)):
    if original_model.meshes[i].file == "":
      continue
    updated_path = str(robot_path.parent) + "/assets/" + original_model.meshes[i].file
    start_mesh_section = xml_string.find("<mesh")
    search_string = original_model.meshes[i].file.split("/")[-1]
    start_file_string = xml_string.find('file="' + search_string, start_mesh_section) + 6
    end_file_string = xml_string.find('"', start_file_string)
    xml_string = xml_string[:start_file_string] + updated_path + xml_string[end_file_string:]

  # Same process for textures
  for i in range(len(original_model.textures)):
    if original_model.textures[i].file == "":
      continue
    updated_path = str(robot_path.parent) + "/assets/" + original_model.textures[i].file
    start_texture_section = xml_string.find("<texture")
    search_string = original_model.textures[i].file.split("/")[-1]
    start_file_string = xml_string.find('file="' + search_string, start_texture_section) + 6
    end_texture_string = xml_string.find("/>", start_file_string)
    end_file_string = xml_string.find('"', start_file_string)
    if end_file_string <= end_texture_string:
      xml_string = xml_string[:start_file_string] + updated_path + xml_string[end_file_string:]

  ## Post-processing to remove keyframes
  start_keyframe_string = xml_string.find("<keyframe>")
  end_keyframe_string = xml_string.find("</keyframe>", start_keyframe_string) + 11
  if start_keyframe_string > 1:
    xml_string = xml_string[:start_keyframe_string] + xml_string[end_keyframe_string:]

  ## Creating the mujoco model and adjusting its properties
  mjm = mujoco.MjModel.from_xml_string(xml_string)
  if _CONE.value == "pyramidal":
    mjm.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  elif _CONE.value == "elliptic":
    mjm.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

  if _IS_SPARSE.value == True:
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
  elif _IS_SPARSE.value == False:
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

  if _SOLVER.value == "cg":
    mjm.opt.solver = mujoco.mjtSolver.mjSOL_CG
  elif _SOLVER.value == "newton":
    mjm.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

  if _ITERATIONS.value is not None:
    mjm.opt.iterations = _ITERATIONS.value

  if _LS_ITERATIONS.value is not None:
    mjm.opt.ls_iterations = _LS_ITERATIONS.value

  ## Creating the mujoco data and populating some constraints
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)

  m = mjwarp.put_model(mjm)
  m.opt.ls_parallel = _LS_PARALLEL.value
  d = mjwarp.put_data(mjm, mjd, nworld=_BATCH_SIZE.value, nconmax=_NCONMAX.value, njmax=_NJMAX.value)

  if _CLEAR_KERNEL_CACHE.value:
    wp.clear_kernel_cache()

  solver_name = {1: "CG", 2: "Newton"}[mjm.opt.solver]
  linesearch_name = {True: "parallel", False: "iterative"}[m.opt.ls_parallel]
  print(
    f"Model nbody: {m.nbody} nv: {m.nv} ngeom: {m.ngeom} "
    f"is_sparse: {_IS_SPARSE.value} solver: {solver_name} "
    f"iterations: {m.opt.iterations} ls_iterations: {m.opt.ls_iterations} "
    f"linesearch: {linesearch_name}"
  )
  print(f"Data nworld: {d.nworld} nconmax: {d.nconmax} njmax: {d.njmax}")
  print(f"Rolling out {_NSTEP.value} steps at dt = {m.opt.timestep:.3f}...")
  jit_time, run_time, trace, ncon, nefc = mjwarp.benchmark(
    mjwarp.__dict__[_FUNCTION.value],
    m,
    d,
    _NSTEP.value,
    _EVENT_TRACE.value,
    _MEASURE_ALLOC.value,
  )
  steps = _BATCH_SIZE.value * _NSTEP.value

  name = argv[0]
  if _OUTPUT.value == "text":
    print(f"""
Summary for {_BATCH_SIZE.value} parallel rollouts

 Total JIT time: {jit_time:.2f} s
 Total simulation time: {run_time:.2f} s
 Total steps per second: {steps / run_time:,.0f}
 Total realtime factor: {steps * m.opt.timestep / run_time:,.2f} x
 Total time per step: {1e9 * run_time / steps:.2f} ns""")
    if trace:
      print("\nEvent trace:\n")

      def _print_trace(trace, indent):
        for k, v in trace.items():
          times, sub_trace = v
          if len(times) == 1:
            print("  " * indent + f"{k}: {1e6 * times[0] / steps:.2f}")
          else:
            print("  " * indent + f"{k}: [ ", end="")
            for i in range(len(times)):
              print(f"{1e6 * times[i] / steps:.2f}", end="")
              print(", " if i < len(times) - 1 else " ", end="")
            print("]")
          _print_trace(sub_trace, indent + 1)

      _print_trace(trace, 0)
    if ncon and nefc:
      num_buckets = 10
      idx = 0
      ncon_matrix, nefc_matrix = [], []
      for i in range(num_buckets):
        size = _NSTEP.value // num_buckets + (i < (_NSTEP.value % num_buckets))
        ncon_arr = np.array(ncon[idx : idx + size])
        nefc_arr = np.array(nefc[idx : idx + size])
        ncon_matrix.append([np.mean(ncon_arr), np.std(ncon_arr), np.min(ncon_arr), np.max(ncon_arr)])
        nefc_matrix.append([np.mean(nefc_arr), np.std(nefc_arr), np.min(nefc_arr), np.max(nefc_arr)])
        idx += size

      def _print_table(matrix, headers):
        num_cols = len(headers)
        col_widths = [max(len(f"{row[i]:g}") for row in matrix) for i in range(num_cols)]
        col_widths = [max(col_widths[i], len(headers[i])) for i in range(num_cols)]

        print("  ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(num_cols)))
        print("-" * sum(col_widths) + "--" * 3)  # Separator line
        for row in matrix:
          print("  ".join(f"{row[i]:{col_widths[i]}g}" for i in range(num_cols)))

      print("\nncon alloc:\n")
      _print_table(ncon_matrix, ("mean", "std", "min", "max"))
      print("\nnefc alloc:\n")
      _print_table(nefc_matrix, ("mean", "std", "min", "max"))

  elif _OUTPUT.value == "tsv":
    name = name.split("/")[-1].replace("testspeed_", "")
    print(f"{name}\tjit: {jit_time:.2f}s\tsteps/second: {steps / run_time:.0f}")


def _load_from_menagerie(asset_path: str) -> str:
  """Load an asset from the mujoco menagerie."""
  # Ensure menagerie exists, and otherwise clone it
  _ensure_menagerie_exists()
  return _MENAGERIE_PATH / asset_path


def _ensure_menagerie_exists() -> None:
  """Ensure mujoco_menagerie exists, downloading it if necessary."""
  if not _MENAGERIE_PATH.exists():
    print("mujoco_menagerie not found. Downloading...")

    try:
      _clone_with_progress(
        "https://github.com/deepmind/mujoco_menagerie.git",
        str(_MENAGERIE_PATH),
        _MENAGERIE_COMMIT_SHA,
      )
      print("Successfully downloaded mujoco_menagerie")
    except subprocess.CalledProcessError as e:
      print(f"Error downloading mujoco_menagerie: {e}", file=sys.stderr)
      raise


def _clone_with_progress(repo_url: str, target_path: str, commit_sha: str) -> None:
  """Clone a git repo with progress bar."""
  process = subprocess.Popen(
    ["git", "clone", "--progress", repo_url, target_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
  )

  with tqdm.tqdm(
    desc="Cloning mujoco_menagerie",
    bar_format="{desc}: {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
  ) as pbar:
    pbar.total = 100  # Set to 100 for percentage-based progress.
    current = 0
    while True:
      # Read output line by line.
      output = process.stderr.readline()  # pytype: disable=attribute-error
      if not output and process.poll() is not None:
        break
      if output:
        if "Receiving objects:" in output:
          try:
            percent = int(output.split("%")[0].split(":")[-1].strip())
            if percent > current:
              pbar.update(percent - current)
              current = percent
          except (ValueError, IndexError):
            pass

    # Ensure the progress bar reaches 100%.
    if current < 100:
      pbar.update(100 - current)

  if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, ["git", "clone"])

  # Checkout specific commit.
  print(f"Checking out commit {commit_sha}")
  subprocess.run(
    ["git", "-C", target_path, "checkout", commit_sha],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )


def main():
  app.run(_main)


if __name__ == "__main__":
  main()
