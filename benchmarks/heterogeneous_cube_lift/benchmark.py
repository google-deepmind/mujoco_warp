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

"""Compare mixed and separate cube lifts using per-world tree sleeping (PR #1680).

uv run python benchmarks/heterogeneous_cube_lift/benchmark.py --assets /path/to/menagerie

For each robot, compare 8192 ordinary standalone worlds against 8192 union-model
worlds with only that robot and its cubes awake. Inactive trees use ALWAYS and
active trees use AUTO_NEVER. Setup and reset are excluded from timing.
"""

import argparse
import dataclasses
import hashlib
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src import sleep
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import SleepPolicy

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("franka", "aloha", "so101")
CUBES = {"franka": ("cube",), "aloha": ("cube_left", "cube_right"), "so101": ("cube",)}
MENAGERIE_REF = "affef0836947b64cc06c4ab1cbf0152835693374"


@wp.kernel
def replay_control(sequence: wp.array2d[float], index: wp.array[int], ctrl_out: wp.array2d[float]):
  world, actuator = wp.tid()
  ctrl_out[world, actuator] = sequence[wp.min(index[0], sequence.shape[0] - 1), actuator]


@wp.kernel
def advance(index_out: wp.array[int]):
  index_out[0] += 1


def model(kind, assets, sleeping=True):
  spec = mujoco.MjSpec.from_string(
    '<mujoco><visual><headlight diffuse=".6 .6 .6" ambient=".3 .3 .3" specular="0 0 0"/>'
    '<rgba haze=".15 .25 .35 1"/></visual><worldbody>'
    '<light pos="1.5 0 3" dir="0 0 -1" directional="true"/></worldbody></mujoco>'
  )
  spec.option.timestep = 0.002
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = 10
  spec.option.iterations = 50
  spec.option.ls_iterations = 50
  spec.option.enableflags = int(mujoco.mjtEnableBit.mjENBL_SLEEP) if sleeping else 0
  initial = {}
  for name in KINDS if kind == "mixed" else (kind,):
    prefix = name + "/"
    child = mujoco.MjSpec.from_file(str(Path(__file__).parent / f"{name}.xml"))
    folder = {"franka": "franka_emika_panda", "aloha": "aloha", "so101": "robotstudio_so101"}[name]
    for mesh in child.meshes:
      mesh.file = str(assets / folder / "assets" / Path(mesh.file).name)
    if initial:
      child.delete(child.geom("floor"))
    for light in list(child.lights):
      child.delete(light)
    cm = child.compile()
    replay = np.load(Path(__file__).parent / f"{name}.npz")
    qpos = replay["qpos"][0].copy()
    offset = 1.5 * KINDS.index(name)
    for cube in CUBES[name]:
      qpos[cm.joint(cube + "_joint").qposadr[0]] += offset
    initial[prefix] = cm, qpos, replay["ctrl"]
    for key in list(child.keys):
      child.delete(key)
    spec.attach(child, prefix=prefix, frame=spec.worldbody.add_frame(pos=[offset, 0, 0]))
  mjm = spec.compile()
  mjd = mujoco.MjData(mjm)
  controls = np.zeros((2500, mjm.nu), dtype=np.float32)
  for prefix, (cm, qpos, ctrl) in initial.items():
    for j in range(cm.njnt):
      dest = mjm.joint(prefix + cm.joint(j).name)
      start = cm.jnt_qposadr[j]
      end = cm.jnt_qposadr[j + 1] if j + 1 < cm.njnt else cm.nq
      mjd.qpos[dest.qposadr[0] : dest.qposadr[0] + end - start] = qpos[start:end]
    for a in range(cm.nu):
      controls[:, mjm.actuator(prefix + cm.actuator(a).name).id] = ctrl[:, a]
  mjd.ctrl[:] = controls[0]
  return mjm, mjd, controls


def arrays_bytes(value, seen=None):
  """Count unique Warp array storage, including nested model/data structures."""
  seen = set() if seen is None else seen
  if isinstance(value, wp.array):
    if value.ptr in seen:
      return 0
    seen.add(value.ptr)
    return value.capacity
  if dataclasses.is_dataclass(value):
    return sum(arrays_bytes(getattr(value, f.name), seen) for f in dataclasses.fields(value))
  return 0


def create(kind, args):
  sleeping = args.case.endswith("_mixed")
  mjm, mjd, controls = model("mixed" if sleeping else kind, args.assets, sleeping)
  nworld = args.worlds
  m = mjw.put_model(mjm, batch_sizes={"tree_sleep_policy": nworld})
  policy = np.full((nworld, mjm.ntree), int(SleepPolicy.AUTO_NEVER), dtype=np.int32)
  if sleeping:
    for tree in range(mjm.ntree):
      body = np.flatnonzero(mjm.body_treeid == tree)[0]
      if not mjm.body(body).name.startswith(kind + "/"):
        policy[:, tree] = int(SleepPolicy.ALWAYS)
  m.tree_sleep_policy.assign(policy)
  # Capacities are identical per world except nvmax, which reflects each layout's active DOFs.
  nvmax = int(np.max((policy != int(SleepPolicy.ALWAYS)) @ mjm.tree_dofnum))
  d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=64, nccdmax=64, njmax=256, nvmax=nvmax if sleeping else None)
  state = (kind, mjm, mjd, m, d, policy)
  reset(state)
  return state, wp.array(controls, dtype=float)


def reset(state):
  _, mjm, mjd, m, d, policy = state
  mjw.reset_data(m, d)
  d.qpos.assign(np.tile(mjd.qpos, (d.nworld, 1)).astype(np.float32))
  d.ctrl.assign(np.tile(mjd.ctrl, (d.nworld, 1)).astype(np.float32))
  eq_active = np.tile(mjd.eq_active, (d.nworld, 1))
  for eq in range(mjm.neq):
    if mjm.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
      raise ValueError("Expected finger joint equality constraints")
    for joint in (mjm.eq_obj1id[eq], mjm.eq_obj2id[eq]):
      if joint >= 0:
        tree = mjm.body_treeid[mjm.jnt_bodyid[joint]]
        eq_active[:, eq] &= policy[:, tree] != int(SleepPolicy.ALWAYS)
  d.eq_active.assign(eq_active)
  if m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
    sleep.update_sleep(m, d)


def diagnostics(state):
  kind, mjm, _, _, d, policy = state
  overflow = d.overflow.numpy()
  capacity_mask = int(OverflowType.ALL)
  if np.any(overflow & capacity_mask):
    raise RuntimeError(f"{kind}: overflow: {np.unique(overflow).tolist()}")
  qvel = d.qvel.numpy()
  if not np.isfinite(d.qpos.numpy()).all() or not np.isfinite(qvel).all():
    raise RuntimeError(f"{kind}: nonfinite state")
  inactive = policy[:, mjm.body_treeid[mjm.dof_bodyid]] == int(SleepPolicy.ALWAYS)
  if np.any(qvel[inactive] != 0):
    raise RuntimeError(f"{kind}: inactive tree moved")
  awake = d.tree_awake.numpy()
  # tree_awake is a per-tree flag, not the compacted tree list.
  expected = policy != int(SleepPolicy.ALWAYS)
  if state[3].opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP) and not np.array_equal(awake > 0, expected):
    raise RuntimeError(f"{kind}: unexpected awake trees: {awake[0]}, expected {expected[0]}")
  return {
    "kind": kind,
    "nworld": d.nworld,
    "nq": mjm.nq,
    "nv": mjm.nv,
    "ntree": mjm.ntree,
    "nvmax": d.nvmax,
    "contacts_total": int(d.nacon.numpy()[0]),
    "constraints_max": int(d.nefc.numpy().max()),
    "solver_iterations_mean": float(d.solver_niter.numpy().mean()),
    "overflow_values": np.unique(overflow).tolist(),
  }


def snapshot(state, storage):
  kind, mjm, _, _, d, _ = state
  qpos, qvel = d.qpos.numpy(), d.qvel.numpy()
  for field, array, addresses, total in (("qpos", qpos, mjm.jnt_qposadr, mjm.nq), ("qvel", qvel, mjm.jnt_dofadr, mjm.nv)):
    indices = []
    for j in range(mjm.njnt):
      if mjm.joint(j).name.startswith(kind + "/"):
        indices.extend(range(addresses[j], addresses[j + 1] if j + 1 < mjm.njnt else total))
    values = array[:, indices]
    storage.setdefault(field + "_spread", []).append(np.max(np.abs(values - values[0]), axis=0))
    storage.setdefault(field, []).append(values[0].copy())


def worker(args):
  wp.init()
  wp.set_device(args.device)
  wp.config.log_level = wp.LOG_WARNING
  if not wp.get_device().is_cuda:
    raise ValueError("This benchmark requires CUDA graph execution")
  free_start = wp.get_device().free_memory
  start = time.perf_counter()
  kind = args.case.rsplit("_", 1)[0]
  state, controls = create(kind, args)
  _, mjm, mjd, m, d, _ = state
  index = wp.zeros(1, dtype=int)
  wp.synchronize()
  setup_s = time.perf_counter() - start
  start = time.perf_counter()
  mjw.step(m, d)
  wp.synchronize()
  with wp.ScopedCapture() as capture:
    wp.launch(replay_control, (d.nworld, mjm.nu), inputs=[controls, index], outputs=[d.ctrl])
    mjw.step(m, d)
    wp.launch(advance, 1, outputs=[index])
  for _ in range(args.warmup):
    wp.capture_launch(capture.graph)
  wp.synchronize()
  compile_warmup_s = time.perf_counter() - start
  gpu_ms, wall_ms = [], []
  begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
  samples_diagnostics = []
  for _ in range(args.repeats):
    index.zero_()
    reset(state)
    wp.synchronize()
    start = time.perf_counter()
    wp.record_event(begin)
    for _ in range(args.steps):
      wp.capture_launch(capture.graph)
    wp.record_event(end)
    wp.synchronize()
    wall_ms.append(1000 * (time.perf_counter() - start) / args.steps)
    gpu_ms.append(wp.get_event_elapsed_time(begin, end) / args.steps)
    samples_diagnostics.append(diagnostics(state))
  memory = {
    "model_array_bytes": arrays_bytes(m),
    "data_array_bytes": arrays_bytes(d),
    "replay_array_bytes": arrays_bytes(controls) + arrays_bytes(index),
    "device_used_delta_bytes": free_start - wp.get_device().free_memory,
  }
  # Replay outside timing and sample state and awake-tree checks every 100 ms.
  trajectory = {}
  index.zero_()
  reset(state)
  for step in range(args.steps):
    wp.capture_launch(capture.graph)
    if step % 50 == 0 or step == args.steps - 1:
      diagnostics(state)
      snapshot(state, trajectory)
  lifts = {}
  qpos = d.qpos.numpy()
  for cube in CUBES[kind]:
    adr = mjm.joint(kind + "/" + cube + "_joint").qposadr[0] + 2
    height = qpos[:, adr] - mjd.qpos[adr]
    lifts[cube] = float(height.min())
    if height.min() < 0.03:
      raise RuntimeError(f"{kind}/{cube}: cube did not lift: {height.min()} m")
  np.savez(args.output / f"{args.case}_trajectory.npz", **trajectory)
  result = {
    "case": args.case,
    "minimum_cube_lift_m": lifts,
    "worlds": d.nworld,
    "setup_s": setup_s,
    "compile_warmup_s": compile_warmup_s,
    "gpu_batch_step_ms": gpu_ms,
    "wall_batch_step_ms": wall_ms,
    "fps": d.nworld * 1000 / float(np.median(wall_ms)),
    "memory": memory,
    "diagnostics": samples_diagnostics,
    "gpu": wp.get_device().name,
    "warp": wp.__version__,
    "mujoco": mujoco.__version__,
  }
  (args.output / f"{args.case}.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--assets", type=Path, required=True, help="Menagerie checkout at the documented pinned revision")
  parser.add_argument("--output", type=Path, default=Path("heterogeneous-results"))
  parser.add_argument("--worlds", type=int, default=8192, help="Worlds of EACH type")
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument("--device", default="cuda:0")
  cases = [f"{kind}_{layout}" for kind in KINDS for layout in ("alone", "mixed")]
  parser.add_argument("--case", choices=cases, help=argparse.SUPPRESS)
  args = parser.parse_args()
  args.steps = 2500
  if min(args.worlds, args.repeats, args.warmup) < 1:
    parser.error("Counts and capacities must be positive")
  args.assets = args.assets.resolve()
  args.output = args.output.resolve()
  args.output.mkdir(parents=True, exist_ok=True)
  if args.case:
    worker(args)
    return
  revision = subprocess.check_output(["git", "-C", str(args.assets), "rev-parse", "HEAD"], text=True).strip()
  if revision != MENAGERIE_REF:
    parser.error(f"Expected Menagerie {MENAGERIE_REF}, got {revision}")
  child_args = []
  for key, value in vars(args).items():
    if key not in ("case", "steps"):
      child_args.extend(["--" + key, str(value)])
  results = []
  for case in cases:
    print(f"Running {case} in a fresh process...", flush=True)
    with (args.output / f"{case}.log").open("w") as log:
      subprocess.run(
        ["uv", "run", "python", str(Path(__file__).resolve()), *child_args, "--case", case],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
        cwd=ROOT,
      )
    results.append(json.loads((args.output / f"{case}.json").read_text()))
  parity = {}
  cube_position_errors = {}
  world_spread = {}
  failures = []
  # Contact order can differ across layouts. Bound raw qpos component differences
  # by 0.002 and qvel by 0.1, and report actual errors and within-layout variation.
  for kind in KINDS:
    with (
      np.load(args.output / f"{kind}_mixed_trajectory.npz") as mixed,
      np.load(args.output / f"{kind}_alone_trajectory.npz") as reference,
    ):
      for key in reference.files:
        delta = float(np.max(np.abs(mixed[key] - reference[key])))
        if key.endswith("_spread"):
          world_spread[f"{kind}_mixed/{key}"] = float(mixed[key].max())
          world_spread[f"{kind}_alone/{key}"] = float(reference[key].max())
          tolerance = 0.002 if "qpos" in key else 0.1
          if max(mixed[key].max(), reference[key].max()) > tolerance:
            failures.append(f"{kind}/{key}: within-layout spread exceeds {tolerance}")
          continue
        if key == "qpos":
          # Source scenes append the cubes' free joints after their robot joints.
          for i, cube in enumerate(CUBES[kind]):
            start = mixed[key].shape[1] - 7 * (len(CUBES[kind]) - i)
            cube_position_errors[f"{kind}/{cube}"] = float(
              np.linalg.norm(mixed[key][:, start : start + 3] - reference[key][:, start : start + 3], axis=1).max()
            )
        if not np.allclose(mixed[key], reference[key], atol=0.002 if key == "qpos" else 0.1, rtol=0):
          failures.append(f"{kind}/{key}: max absolute error {delta:g}")
        parity[f"{kind}/{key}"] = delta
  report = {
    "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
    "menagerie_commit": revision,
    "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "trajectory_parity_failures": failures,
    "trajectory_max_absolute_errors": parity,
    "cube_position_errors_m": cube_position_errors,
    "within_layout_max_spread": world_spread,
    "results": results,
  }
  (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
  if failures:
    raise RuntimeError(
      "Trajectory parity failed; timings are not validated equal work. See results.json: " + "; ".join(failures)
    )
  print("robot     worlds   standalone FPS   mixed FPS   mixed / standalone")
  by_case = {r["case"]: r for r in results}
  for kind in KINDS:
    alone, mixed = by_case[kind + "_alone"], by_case[kind + "_mixed"]
    print(f"{kind:8} {args.worlds:7} {alone['fps']:16,.0f} {mixed['fps']:11,.0f} {mixed['fps'] / alone['fps']:19.3f}")
  print("Trajectory parity passed; full samples and metadata:", args.output / "results.json")


if __name__ == "__main__":
  main()
