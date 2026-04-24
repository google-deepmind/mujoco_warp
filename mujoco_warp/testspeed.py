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

"""mjwarp-testspeed: benchmark MuJoCo Warp on an MJCF.

Usage: mjwarp-testspeed <mjcf XML path> [flags]

Example:
  mjwarp-testspeed benchmarks/humanoid/humanoid.xml --nworld 4096 -o "opt.solver=cg"
"""

import dataclasses
import inspect
import json
import shutil
import sys
from typing import Sequence

import numpy as np
import warp as wp
from absl import app
from absl import flags
from etils import epath

import mujoco_warp as mjw

# mjwarp-testspeed has priviledged access to a few internal methods
from mujoco_warp._src import cli
from mujoco_warp._src.collision_driver import MJ_COLLISION_TABLE
from mujoco_warp._src.types import CollisionType

_FUNCS = {
  n: f
  for n, f in inspect.getmembers(mjw, inspect.isfunction)
  if inspect.signature(f).parameters.keys() == {"m", "d"} or inspect.signature(f).parameters.keys() == {"m", "d", "rc"}
}

_FUNCTION = flags.DEFINE_enum("function", "step", _FUNCS.keys(), "the function to benchmark")
_CLEAR_WARP_CACHE = flags.DEFINE_bool("clear_warp_cache", False, "clear warp caches (kernel, LTO, CUDA compute)")
_MEASURE_ALLOC = flags.DEFINE_bool("measure_alloc", False, "print a report of contacts and constraints per step")
_MEASURE_SOLVER = flags.DEFINE_bool("measure_solver", False, "print a report of solver iterations per step")
_NUM_BUCKETS = flags.DEFINE_integer("num_buckets", 10, "number of buckets to summarize rollout measurements")
_MEMORY = flags.DEFINE_bool("memory", False, "print memory report")
_INFO = flags.DEFINE_bool("info", False, "print extra Model and Data info")
_FORMAT = flags.DEFINE_enum("format", "human", ["human", "short", "json"], "output format")


def _dataclass_memory(dataclass, prefix: str = "") -> list[tuple[str, int]]:
  ret = []
  for field in dataclasses.fields(dataclass):
    value = getattr(dataclass, field.name)
    if dataclasses.is_dataclass(value):
      ret.extend(_dataclass_memory(value, prefix=f"{prefix}{field.name}."))
    elif isinstance(value, wp.array):
      ret.append((f"{prefix}{field.name}", value.capacity))
  return ret


def _flatten_trace(trace: dict[str, float]) -> dict[str, float]:
  """Flatten the event trace into a dictionary of metrics."""
  steps = cli.NSTEP.value * cli.NWORLD.value
  metrics = {}

  def flatten(prefix: str, trace):
    for k, v in trace.items():
      times, sub_trace = v
      for i, t in enumerate(times):
        metrics[f"{prefix}{k}{f'[{i}]' if len(times) > 1 else ''}"] = 1e6 * t / steps
      flatten(f"{prefix}{k}.", sub_trace)

  flatten("", trace)

  return metrics


def _sum_trace(stack1, stack2):
  """Recursively sum event trace stacks."""
  ret = {}

  for k in stack1.keys() | stack2.keys():
    if k not in stack1:
      ret[k] = stack2[k]
    elif k not in stack2:
      ret[k] = stack1[k]
    else:
      times1, sub_stack1 = stack1[k]
      times2, sub_stack2 = stack2[k]
      times = [t1 + t2 for t1, t2 in zip(times1, times2)]
      ret[k] = (times, _sum_trace(sub_stack1, sub_stack2))

  return ret


def _print_trace(trace, indent):
  """Recursively print event trace."""
  steps = cli.NSTEP.value * cli.NWORLD.value

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


def _print_table(matrix, headers, title):
  """Print a matrix in a tabular format."""
  num_cols = len(headers)
  col_widths = [max(len(f"{row[i]:g}") for row in matrix) for i in range(num_cols)]
  col_widths = [max(col_widths[i], len(headers[i])) for i in range(num_cols)]

  print(f"\n{title}:\n")
  print("  ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(num_cols)))
  print("-" * sum(col_widths) + "--" * 3)  # Separator line
  for row in matrix:
    print("  ".join(f"{row[i]:{col_widths[i]}g}" for i in range(num_cols)))


def _main(argv: Sequence[str]):
  """Run the benchmark."""
  if len(argv) < 2:
    raise app.UsageError("Missing required input: mjcf path.")
  elif len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")

  if _FUNCTION.value not in _FUNCS:
    raise ValueError(f"Unknown function: {_FUNCTION.value}")

  if _CLEAR_WARP_CACHE.value:
    wp.clear_kernel_cache()
    wp.clear_lto_cache()
    compute_cache = epath.Path("~/.nv/ComputeCache").expanduser()
    if compute_cache.exists():
      shutil.rmtree(compute_cache)
      compute_cache.mkdir()

  wp.config.quiet = flags.FLAGS["verbosity"].value < 1
  wp.init()
  device = wp.get_device(cli.DEVICE.value)
  if device == "cpu":
    raise ValueError("testspeed available for gpu only")

  path = epath.Path(argv[1])

  if _FORMAT.value == "human":
    print(f"Loading model from: {path}...\n")

  mjm = cli.load_model(path)
  free_mem_at_init = wp.get_device(device).free_memory
  m, d, rc, ctrls = cli.init_structs(_FUNCS[_FUNCTION.value], mjm)
  timestep = m.opt.timestep.numpy()[0]

  if _FORMAT.value == "human":
    # Model
    print("Model")
    if _INFO.value:
      size_fields = [f.name for f in dataclasses.fields(m) if f.type is int and getattr(m, f.name) > 0]
    else:
      size_fields = ["nq", "nv", "nu", "nbody", "ngeom"]
    for i, f in enumerate(size_fields):
      print(f"{'  ' if i % 5 == 0 else ' '}{f}: {getattr(m, f)}", end="\n" if i % 5 == 4 or i == len(size_fields) - 1 else "")

    # RenderContext
    if rc is not None:
      print("RenderContext")
      print(f"  shadows: {rc.use_shadows} textures: {rc.use_textures} nlight: {m.nlight} bvh_ngeom: {rc.bvh_ngeom}")
      print(f"  ncam: {rc.nrender} cam_res: {[(int(x[0]), int(x[1])) for x in rc.cam_res.numpy()]}")

    # Option
    print("Option")
    if _INFO.value:
      print(
        f"  timestep: {m.opt.timestep.numpy()[0]:g}\n"
        f"  tolerance: {m.opt.tolerance.numpy()[0]:g} ls_tolerance: {m.opt.ls_tolerance.numpy()[0]:g}\n"
        f"  ccd_tolerance: {m.opt.ccd_tolerance.numpy()[0]:g}\n"
        f"  density: {m.opt.density.numpy()[0]:g} viscosity: {m.opt.viscosity.numpy()[0]:g}\n"
        f"  gravity: {m.opt.gravity.numpy()[0]}\n"
        f"  wind: {m.opt.wind.numpy()[0]} magnetic: {m.opt.magnetic.numpy()[0]}\n"
        f"  integrator: {mjw.IntegratorType(m.opt.integrator).name}\n"
        f"  cone: {mjw.ConeType(m.opt.cone).name}\n"
        f"  solver: {mjw.SolverType(m.opt.solver).name} iterations: {m.opt.iterations} ls_iterations: {m.opt.ls_iterations}\n"
        f"  ccd_iterations: {m.opt.ccd_iterations}\n"
        f"  sdf_initpoints: {m.opt.sdf_initpoints} sdf_iterations: {m.opt.sdf_iterations}\n"
        f"  disableflags: [{mjw.DisableBit(m.opt.disableflags).name or 'none'}]\n"
        f"  enableflags: [{mjw.EnableBit(m.opt.enableflags).name or 'none'}]\n"
        f"  impratio: {1.0 / np.square(m.opt.impratio_invsqrt.numpy()[0]):g}\n"
        f"  is_sparse: {m.is_sparse}\n"
        f"  ls_parallel: {m.opt.ls_parallel} ls_parallel_min_step: {m.opt.ls_parallel_min_step:g}\n"
        f"  has_fluid: {m.has_fluid}\n"
        f"  broadphase: {m.opt.broadphase.name} broadphase_filter: {m.opt.broadphase_filter.name}\n"
        f"  graph_conditional: {m.opt.graph_conditional}\n"
        f"  run_collision_detection: {m.opt.run_collision_detection}\n"
        f"  contact_sensor_maxmatch: {m.opt.contact_sensor_maxmatch}"
      )
      # Colliders
      print("Colliders")
      colliders = {"Primitive": [], "HfieldCCD": [], "CCD": []}
      for trid, count in enumerate(m.geom_pair_type_count):
        if count == 0:
          continue
        # convert triangle index to i, j
        n = len(mjw.GeomType)
        i = mjw.GeomType(int((2 * n + 1 - np.sqrt((2 * n + 1) ** 2 - 8 * trid)) / 2))
        j = mjw.GeomType(trid - i * (2 * n - i - 1) // 2)
        match MJ_COLLISION_TABLE.get((i, j)):
          case CollisionType.PRIMITIVE:
            colliders["Primitive"].append(f"{i.name}-{j.name}: {count}")
          case CollisionType.CONVEX if mjw.GeomType.HFIELD in (i, j):
            colliders["HfieldCCD"].append(f"{i.name}-{j.name}: {count}")
          case CollisionType.CONVEX:
            colliders["CCD"].append(f"{i.name}-{j.name}: {count}")
      if any(colliders.values()):
        for typ, pairs in colliders.items():
          if pairs:
            print(f"  {typ}\n" + "\n".join(f"    {p}" for p in pairs))
      else:
        print("  none")
      print(f"  max collisions: {sum(m.geom_pair_type_count)}")
    else:
      print(
        f"  integrator: {mjw.IntegratorType(m.opt.integrator).name}\n"
        f"  cone: {mjw.ConeType(m.opt.cone).name}\n"
        f"  solver: {mjw.SolverType(m.opt.solver).name} iterations: {m.opt.iterations} ls_iterations: {m.opt.ls_iterations}\n"
        f"  is_sparse: {m.is_sparse}\n"
        f"  ls_parallel: {m.opt.ls_parallel}\n"
        f"  broadphase: {m.opt.broadphase.name} broadphase_filter: {m.opt.broadphase_filter.name}"
      )

    print(f"Data\n  nworld: {d.nworld} naconmax: {d.naconmax} njmax: {d.njmax}")
    print(
      f"Rolling out {cli.NSTEP.value} {_FUNCTION.value}s at dt = {f'{timestep:g}' if timestep < 0.001 else f'{timestep:.3f}'}..."
    )

  nacon, nefc, solver_niter = [], [], []
  runtime = 0.0
  trace = {}

  def callback(step, step_trace, latency):
    nonlocal runtime, trace
    runtime += latency
    nacon.append(np.max([d.nacon.numpy()[0], d.ncollision.numpy()[0]]))
    nefc.append(np.max(d.nefc.numpy()))
    solver_niter.append(d.solver_niter.numpy())
    trace = _sum_trace(trace, step_trace)

  jit_duration = cli.unroll(_FUNCS[_FUNCTION.value], m, d, rc, callback, ctrls)

  nconverged = np.sum(~np.any(np.isnan(d.qpos.numpy()), axis=1))
  steps = cli.NWORLD.value * cli.NSTEP.value
  model_mem = _dataclass_memory(m)
  data_mem = _dataclass_memory(d)

  if _FORMAT.value == "human":
    print(f"""
Summary for {d.nworld} parallel rollouts

Total JIT time: {jit_duration:.2f} s
Total simulation time: {runtime:.2f} s
Total steps per second: {steps / runtime:,.0f}
Total realtime factor: {steps * timestep / runtime:,.2f} x
Total time per step: {1e9 * runtime / steps:.2f} ns
Total converged worlds: {nconverged} / {d.nworld}""")

    if trace:
      print("\nEvent trace:\n")
      _print_trace(trace, 0)

    if _MEASURE_ALLOC.value:
      idx = 0
      nacon_matrix, nefc_matrix = [], []
      for i in range(_NUM_BUCKETS.value):
        size = cli.NSTEP.value // _NUM_BUCKETS.value + (i < (cli.NSTEP.value % _NUM_BUCKETS.value))
        nacon_arr = np.array(nacon[idx : idx + size])
        nefc_arr = np.array(nefc[idx : idx + size])
        nacon_matrix.append([np.mean(nacon_arr), np.std(nacon_arr), np.min(nacon_arr), np.max(nacon_arr)])
        nefc_matrix.append([np.mean(nefc_arr), np.std(nefc_arr), np.min(nefc_arr), np.max(nefc_arr)])
        idx += size

      _print_table(nacon_matrix, ("mean", "std", "min", "max"), "nacon alloc")
      _print_table(nefc_matrix, ("mean", "std", "min", "max"), "nefc alloc")

    if _MEASURE_SOLVER.value:
      idx = 0
      matrix = []
      for i in range(_NUM_BUCKETS.value):
        size = cli.NSTEP.value // _NUM_BUCKETS.value + (i < (cli.NSTEP.value % _NUM_BUCKETS.value))
        arr = np.array(solver_niter[idx : idx + size])
        matrix.append([np.mean(arr), np.std(arr), np.min(arr), np.max(arr)])
        idx += size

      _print_table(matrix, ("mean", "std", "min", "max"), "solver niter")

    if _MEMORY.value:
      total_mem = wp.get_device(device).total_memory
      used_mem = total_mem - wp.get_device(device).free_memory
      other_mem = used_mem
      for mem, name in [(model_mem, "\nModel"), (data_mem, "Data")]:
        if mem is None:
          continue
        other_mem -= sum(c for _, c in mem)
        other_mem_total = sum(c for _, c in mem)
        print(f"{name} memory {other_mem_total / 1024**2:.2f} MiB ({100 * other_mem_total / used_mem:.2f}% of used memory):")
        fields = [(f, c) for f, c in mem if c / used_mem >= 0.01]
        for field, capacity in fields:
          print(f" {field}: {capacity / 1024**2:.2f} MiB ({100 * capacity / used_mem:.2f}%)")
        if not fields:
          print(" (no field >= 1% of used memory)")
      print(f"Other memory: {other_mem / 1024**2:.2f} MiB ({100 * other_mem / used_mem:.2f}% of used memory)")
      print(f"Total memory: {used_mem / 1024**2:.2f} MiB ({100 * used_mem / total_mem:.2f}% of total device memory)")
  else:
    metrics = {
      "jit_duration": jit_duration,
      "run_time": runtime,
      "steps_per_second": steps / runtime,
      "converged_worlds": int(nconverged),
      "model_memory": sum(c for _, c in model_mem),
      "data_memory": sum(c for _, c in data_mem),
      "total_memory": free_mem_at_init - wp.get_device(device).free_memory,
      "ncon_mean": np.mean(nacon) / cli.NWORLD.value,
      "ncon_p95": np.percentile(nacon, 95) / cli.NWORLD.value,
      "nefc_mean": np.mean(nefc),
      "nefc_p95": np.percentile(nefc, 95),
      "solver_niter_mean": np.mean(solver_niter),
      "solver_niter_p95": np.percentile(solver_niter, 95),
    }
    if _FORMAT.value == "short":
      for k, v in (metrics | _flatten_trace(trace)).items():
        print(f"{k}: {v}")
    elif _FORMAT.value == "json":
      print(json.dumps(metrics | _flatten_trace(trace)))


def main():
  # absl flags assumes __main__ is the main running module for printing usage documentation
  # pyproject bin scripts break this assumption, so manually set argv and docstring
  sys.argv[0] = "mujoco_warp.testspeed"
  sys.modules["__main__"].__doc__ = __doc__
  app.run(_main)


if __name__ == "__main__":
  main()
