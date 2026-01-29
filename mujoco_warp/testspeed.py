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
  mjwarp-testspeed benchmark/humanoid/humanoid.xml --nworld 4096 -o "opt.solver=cg"
"""

import dataclasses
import inspect
import json
import shutil
import sys
from typing import Sequence

import mujoco
import numpy as np
import warp as wp
from absl import app
from absl import flags
from etils import epath

import mujoco_warp as mjw

# mjwarp-testspeed has priviledged access to a few internal methods
from mujoco_warp._src.benchmark import benchmark
from mujoco_warp._src.io import find_keys
from mujoco_warp._src.io import make_trajectory
from mujoco_warp._src.io import override_model
from mujoco_warp._src.warning import check_warnings

_FUNCS = {n: f for n, f in inspect.getmembers(mjw, inspect.isfunction) if inspect.signature(f).parameters.keys() == {"m", "d"}}

_FUNCTION = flags.DEFINE_enum("function", "step", _FUNCS.keys(), "the function to benchmark")
_NSTEP = flags.DEFINE_integer("nstep", 1000, "number of steps per rollout")
_NWORLD = flags.DEFINE_integer("nworld", 8192, "number of parallel rollouts")
_NCONMAX = flags.DEFINE_integer("nconmax", None, "override maximum number of contacts per world")
_NJMAX = flags.DEFINE_integer("njmax", None, "override maximum number of constraints per world")
_OVERRIDE = flags.DEFINE_multi_string("override", [], "Model overrides (notation: foo.bar = baz)", short_name="o")
_KEYFRAME = flags.DEFINE_integer("keyframe", 0, "keyframe to initialize simulation.")
_CLEAR_WARP_CACHE = flags.DEFINE_bool("clear_warp_cache", False, "clear warp caches (kernel, LTO, CUDA compute)")
_EVENT_TRACE = flags.DEFINE_bool("event_trace", False, "print an event trace report")
_MEASURE_ALLOC = flags.DEFINE_bool("measure_alloc", False, "print a report of contacts and constraints per step")
_MEASURE_SOLVER = flags.DEFINE_bool("measure_solver", False, "print a report of solver iterations per step")
_NUM_BUCKETS = flags.DEFINE_integer("num_buckets", 10, "number of buckets to summarize rollout measurements")
_DEVICE = flags.DEFINE_string("device", None, "override the default Warp device")
_REPLAY = flags.DEFINE_string("replay", None, "keyframe sequence to replay, keyframe name must prefix match")
_MEMORY = flags.DEFINE_bool("memory", False, "print memory report")
_FORMAT = flags.DEFINE_enum("format", "human", ["human", "short", "json"], "output format for results")
_INFO = flags.DEFINE_bool("info", False, "print Model and Data info")


def _load_model(path: epath.Path) -> mujoco.MjModel:
  if not path.exists():
    resource_path = epath.resource_path("mujoco_warp") / path
    if not resource_path.exists():
      raise FileNotFoundError(f"file not found: {path}\nalso tried: {resource_path}")
    path = resource_path

  if path.suffix == ".mjb":
    return mujoco.MjModel.from_binary_path(path.as_posix())

  spec = mujoco.MjSpec.from_file(path.as_posix())
  # check if the file has any mujoco.sdf test plugins
  if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
    from mujoco_warp.test_data.collision_sdf.utils import register_sdf_plugins as register_sdf_plugins

    register_sdf_plugins(mjw)

  return spec.compile()


def _dataclass_memory(dataclass, prefix: str = "") -> list[tuple[str, int]]:
  ret = []
  for field in dataclasses.fields(dataclass):
    value = getattr(dataclass, field.name)
    if dataclasses.is_dataclass(value):
      ret.extend(_dataclass_memory(value, prefix=f"{prefix}{field.name}."))
    elif isinstance(value, wp.array):
      ret.append((f"{prefix}{field.name}", value.capacity))
  return ret


def _collect_metrics(
  m, d, path: epath.Path, free_mem_at_init, jit_time, run_time, trace, nacon, nefc, solver_niter, nsuccess
) -> dict[str, float]:
  """Collect all metrics into a dictionary."""
  steps = _NWORLD.value * _NSTEP.value
  metrics = {
    "benchmark": path.parent.name + path.stem.replace("scene", "") if path.name.startswith("scene") else path.stem,
    "jit_duration": jit_time,
    "run_time": run_time,
    "steps_per_second": steps / run_time,
    "converged_worlds": int(nsuccess),
  }

  def flatten_trace(prefix: str, trace, metrics):
    for k, v in trace.items():
      times, sub_trace = v
      for i, t in enumerate(times):
        metrics[f"{prefix}{k}{f'[{i}]' if len(times) > 1 else ''}"] = 1e6 * t / steps
      flatten_trace(f"{prefix}{k}.", sub_trace, metrics)

  flatten_trace("", trace, metrics)

  if _MEMORY.value:
    metrics.update(
      {
        "model_memory": sum(c for _, c in _dataclass_memory(m)),
        "data_memory": sum(c for _, c in _dataclass_memory(d)),
        "total_memory": free_mem_at_init - wp.get_device(_DEVICE.value).free_memory,
      }
    )

  if nacon and nefc:
    metrics.update(
      {
        "ncon_mean": np.mean(nacon) / _NWORLD.value,
        "ncon_p95": np.percentile(nacon, 95) / _NWORLD.value,
        "nefc_mean": np.mean(nefc),
        "nefc_p95": np.percentile(nefc, 95),
      }
    )

  if solver_niter:
    metrics.update(
      {
        "solver_niter_mean": np.mean(solver_niter),
        "solver_niter_p95": np.percentile(solver_niter, 95),
      }
    )

  return metrics


def _output_short(*args):
  """Output metrics in a short format."""
  metrics = _collect_metrics(*args)
  benchmark = metrics.pop("benchmark")
  max_key_len = max(len(key) for key in metrics.keys()) + len(benchmark)
  for key, value in metrics.items():
    print(f"{benchmark}:{key:<{max_key_len}} {value}")


def _output_json(*args):
  """Output metrics in a JSON format."""
  metrics = _collect_metrics(*args)
  del metrics["benchmark"]
  print(json.dumps(metrics))


def _output_human(m, d, path: epath.Path, free_mem_at_init, jit_time, run_time, trace, nacon, nefc, solver_niter, nsuccess):
  """Output metrics in a human-readable format."""
  steps = _NWORLD.value * _NSTEP.value
  print(f"""
Summary for {_NWORLD.value} parallel rollouts

Total JIT time: {jit_time:.2f} s
Total simulation time: {run_time:.2f} s
Total steps per second: {steps / run_time:,.0f}
Total realtime factor: {steps * m.opt.timestep.numpy()[0] / run_time:,.2f} x
Total time per step: {1e9 * run_time / steps:.2f} ns
Total converged worlds: {nsuccess} / {d.nworld}""")

  if trace:
    print("\nEvent trace:\n")

    def print_trace(trace, indent):
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
        print_trace(sub_trace, indent + 1)

    print_trace(trace, 0)

  def print_table(matrix, headers, title):
    num_cols = len(headers)
    col_widths = [max(len(f"{row[i]:g}") for row in matrix) for i in range(num_cols)]
    col_widths = [max(col_widths[i], len(headers[i])) for i in range(num_cols)]

    print(f"\n{title}:\n")
    print("  ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(num_cols)))
    print("-" * sum(col_widths) + "--" * 3)  # Separator line
    for row in matrix:
      print("  ".join(f"{row[i]:{col_widths[i]}g}" for i in range(num_cols)))

  if nacon and nefc:
    idx = 0
    nacon_matrix, nefc_matrix = [], []
    for i in range(_NUM_BUCKETS.value):
      size = _NSTEP.value // _NUM_BUCKETS.value + (i < (_NSTEP.value % _NUM_BUCKETS.value))
      nacon_arr = np.array(nacon[idx : idx + size])
      nefc_arr = np.array(nefc[idx : idx + size])
      nacon_matrix.append([np.mean(nacon_arr), np.std(nacon_arr), np.min(nacon_arr), np.max(nacon_arr)])
      nefc_matrix.append([np.mean(nefc_arr), np.std(nefc_arr), np.min(nefc_arr), np.max(nefc_arr)])
      idx += size

    print_table(nacon_matrix, ("mean", "std", "min", "max"), "nacon alloc")
    print_table(nefc_matrix, ("mean", "std", "min", "max"), "nefc alloc")

  if solver_niter:
    idx = 0
    matrix = []
    for i in range(_NUM_BUCKETS.value):
      size = _NSTEP.value // _NUM_BUCKETS.value + (i < (_NSTEP.value % _NUM_BUCKETS.value))
      arr = np.array(solver_niter[idx : idx + size])
      matrix.append([np.mean(arr), np.std(arr), np.min(arr), np.max(arr)])
      idx += size

    print_table(matrix, ("mean", "std", "min", "max"), "solver niter")

  if _MEMORY.value:
    total_mem = wp.get_device(_DEVICE.value).total_memory
    used_mem = free_mem_at_init - wp.get_device(_DEVICE.value).free_memory
    other_mem = used_mem
    for dataclass, name in [(m, "\nModel"), (d, "Data")]:
      mem = _dataclass_memory(dataclass)
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


def _main(argv: Sequence[str]):
  if len(argv) < 2:
    raise app.UsageError("Missing required input: mjcf path.")
  elif len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")

  path = epath.Path(argv[1])
  if _FORMAT.value == "human":
    print(f"Loading model from: {path}...\n")
  mjm = _load_model(path)
  mjd = mujoco.MjData(mjm)
  ctrls = None
  if _REPLAY.value:
    keys = find_keys(mjm, _REPLAY.value)
    if not keys:
      raise app.UsageError(f"Key prefix not find: {_REPLAY.value}")
    ctrls = make_trajectory(mjm, keys)
    mujoco.mj_resetDataKeyframe(mjm, mjd, keys[0])
  elif mjm.nkey > 0 and _KEYFRAME.value > -1:
    mujoco.mj_resetDataKeyframe(mjm, mjd, _KEYFRAME.value)
    if ctrls is None:
      ctrls = [mjd.ctrl.copy() for _ in range(_NSTEP.value)]

  wp.config.quiet = flags.FLAGS["verbosity"].value < 1
  wp.init()
  free_mem_at_init = wp.get_device(_DEVICE.value).free_memory
  if _CLEAR_WARP_CACHE.value:
    wp.clear_kernel_cache()
    wp.clear_lto_cache()
    # Clear CUDA compute cache for truly cold start JIT
    compute_cache = epath.Path("~/.nv/ComputeCache").expanduser()
    if compute_cache.exists():
      shutil.rmtree(compute_cache)
      compute_cache.mkdir()

  with wp.ScopedDevice(_DEVICE.value):
    m = mjw.put_model(mjm)
    override_model(m, _OVERRIDE.value)
    d = mjw.put_data(mjm, mjd, nworld=_NWORLD.value, nconmax=_NCONMAX.value, njmax=_NJMAX.value)
    if _FORMAT.value == "human":
      # Model sizes
      if _INFO.value:
        size_fields = [
          "nq",
          "nv",
          "nu",
          "na",
          "nbody",
          "noct",
          "njnt",
          "nM",
          "nC",
          "ngeom",
          "nsite",
          "ncam",
          "nlight",
          "nflex",
          "nflexvert",
          "nflexedge",
          "nflexelem",
          "nflexelemdata",
          "nflexelemedge",
          "nmesh",
          "nmeshvert",
          "nmeshnormal",
          "nmeshface",
          "nmeshgraph",
          "nmeshpoly",
          "nmeshpolyvert",
          "nmeshpolymap",
          "nhfield",
          "nhfielddata",
          "nmat",
          "npair",
          "nexclude",
          "neq",
          "ntendon",
          "nwrap",
          "nsensor",
          "nmocap",
          "nplugin",
          "ngravcomp",
          "nsensordata",
        ]
      else:
        size_fields = [
          "nq",
          "nv",
          "nu",
          "nbody",
          "ngeom",
        ]

      size_items = [f"{name}: {getattr(m, name)}" for name in size_fields if getattr(m, name) > 0]
      # Wrap sizes at 10 items per line
      sizes_lines = []
      for i in range(0, len(size_items), 5):
        sizes_lines.append("  " + " ".join(size_items[i : i + 5]))
      sizes_str = "\n".join(sizes_lines) + "\n"

      # Parse Option.disableflags and Option.enableflags to show individual flag names
      disable_names = [f.name for f in mjw.DisableBit if m.opt.disableflags & f]
      enable_names = [f.name for f in mjw.EnableBit if m.opt.enableflags & f]
      disableflags_str = ", ".join(disable_names) if disable_names else "none"
      enableflags_str = ", ".join(enable_names) if enable_names else "none"

      # Option fields
      if _INFO.value:
        opt_str = (
          f"Option\n"
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
          f"  disableflags: [{disableflags_str}]\n"
          f"  enableflags: [{enableflags_str}]\n"
          f"  impratio: {1.0 / np.square(m.opt.impratio_invsqrt.numpy()[0]):g}\n"
          f"  is_sparse: {m.opt.is_sparse}\n"
          f"  ls_parallel: {m.opt.ls_parallel} ls_parallel_min_step: {m.opt.ls_parallel_min_step:g}\n"
          f"  has_fluid: {m.opt.has_fluid}\n"
          f"  broadphase: {m.opt.broadphase.name} broadphase_filter: {m.opt.broadphase_filter.name}\n"
          f"  graph_conditional: {m.opt.graph_conditional}\n"
          f"  run_collision_detection: {m.opt.run_collision_detection}\n"
          f"  contact_sensor_maxmatch: {m.opt.contact_sensor_maxmatch}\n"
        )
      else:
        opt_str = (
          f"Option\n"
          f"  integrator: {mjw.IntegratorType(m.opt.integrator).name}\n"
          f"  cone: {mjw.ConeType(m.opt.cone).name}\n"
          f"  solver: {mjw.SolverType(m.opt.solver).name} iterations: {m.opt.iterations} ls_iterations: {m.opt.ls_iterations}\n"
          f"  is_sparse: {m.opt.is_sparse}\n"
          f"  ls_parallel: {m.opt.ls_parallel}\n"
          f"  broadphase: {m.opt.broadphase.name} broadphase_filter: {m.opt.broadphase_filter.name}\n"
        )

      if _INFO.value:
        # Collider types grouped by category
        from mujoco_warp._src import collision_convex
        from mujoco_warp._src import collision_primitive

        def trid_to_types(trid):
          """Convert triangular index back to geom type pair."""
          n = len(mjw.GeomType)
          i = 0
          while (i + 1) * (2 * n - i) // 2 <= trid:
            i += 1
          j = trid - i * (2 * n - i - 1) // 2
          return mjw.GeomType(i), mjw.GeomType(j)

        # Categorize collision pairs
        primitive_pairs = set(collision_primitive._PRIMITIVE_COLLISIONS.keys())
        hfield_ccd_pairs = set(collision_convex._HFIELD_COLLISION_PAIRS)
        ccd_pairs = set(collision_convex._NON_HFIELD_COLLISION_PAIRS)

        primitive_colliders, hfield_ccd_colliders, ccd_colliders = [], [], []
        for trid, count in enumerate(m.geom_pair_type_count):
          if count > 0:
            t1, t2 = trid_to_types(trid)
            pair = (t1, t2)
            pair_str = f"{t1.name}-{t2.name}: {count}"
            if pair in primitive_pairs:
              primitive_colliders.append(pair_str)
            elif pair in hfield_ccd_pairs:
              hfield_ccd_colliders.append(pair_str)
            elif pair in ccd_pairs:
              ccd_colliders.append(pair_str)

        collider_lines = []
        if primitive_colliders:
          primitives = "  Primitive"
          for collider in primitive_colliders:
            primitives += f"\n  {collider}"
          collider_lines.append(primitives)
        if hfield_ccd_colliders:
          hfield = "  HFieldCCD"
          for collider in hfield_ccd_colliders:
            hfield += f"\n  {collider}"
          collider_lines.append(hfield)
        if ccd_colliders:
          ccd = "  CCD"
          for collider in ccd_colliders:
            ccd += f"\n  {collider}"
          collider_lines.append(ccd)
        max_collisions = sum(m.geom_pair_type_count)
        collider_lines.append(f"  max collisions: {max_collisions}")
        collider_str = "Colliders\n" + "\n".join(collider_lines) + "\n" if collider_lines else ""
      else:
        collider_str = ""

      print(
        f"Model\n{sizes_str}{opt_str}{collider_str}"
        f"Data\n  nworld: {d.nworld} naconmax: {d.naconmax} njmax: {d.njmax}\n\n"
        f"Rolling out {_NSTEP.value} steps at dt = {f'{m.opt.timestep.numpy()[0]:g}' if m.opt.timestep.numpy()[0] < 0.001 else f'{m.opt.timestep.numpy()[0]:.3f}'}..."
      )

    fn = _FUNCS[_FUNCTION.value]
    res = benchmark(fn, m, d, _NSTEP.value, ctrls, _EVENT_TRACE.value, _MEASURE_ALLOC.value, _MEASURE_SOLVER.value)

    # Check and emit any overflow warnings to stderr
    check_warnings(d)

    match _FORMAT.value:
      case "short":
        _output_short(m, d, path, free_mem_at_init, *res)
      case "json":
        _output_json(m, d, path, free_mem_at_init, *res)
      case "human":
        _output_human(m, d, path, free_mem_at_init, *res)


def main():
  # absl flags assumes __main__ is the main running module for printing usage documentation
  # pyproject bin scripts break this assumption, so manually set argv and docstring
  sys.argv[0] = "mujoco_warp.testspeed"
  sys.modules["__main__"].__doc__ = __doc__
  app.run(_main)


if __name__ == "__main__":
  main()
