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

import ast
import inspect
import sys
from typing import Sequence

import mujoco
import numpy as np
import warp as wp
from absl import app
from absl import flags
from etils import epath

import mujoco_warp as mjw

_FUNCS = {n: f for n, f in inspect.getmembers(mjw, inspect.isfunction) if inspect.signature(f).parameters.keys() == {"m", "d"}}

_FUNCTION = flags.DEFINE_enum("function", "step", _FUNCS.keys(), "the function to benchmark")
_NSTEP = flags.DEFINE_integer("nstep", 1000, "number of steps per rollout")
_NWORLD = flags.DEFINE_integer("nworld", 8192, "number of parallel rollouts")
_NCONMAX = flags.DEFINE_integer("nconmax", None, "override maximum number of contacts for all worlds")
_NJMAX = flags.DEFINE_integer("njmax", None, "override maximum number of constraints per world")
_OVERRIDE = flags.DEFINE_multi_string("override", [], "Model overrides (notation: foo.bar = baz)", short_name="o")
_KEYFRAME = flags.DEFINE_integer("keyframe", 0, "keyframe to initialize simulation.")
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "clear kernel cache (to calculate full JIT time)")
_EVENT_TRACE = flags.DEFINE_bool("event_trace", False, "print an event trace report")
_MEASURE_ALLOC = flags.DEFINE_bool("measure_alloc", False, "print a report of contacts and constraints per step")
_MEASURE_SOLVER = flags.DEFINE_bool("measure_solver", False, "print a report of solver iterations per step")
_NUM_BUCKETS = flags.DEFINE_integer("num_buckets", 10, "number of buckets to summarize rollout measurements")
_DEVICE = flags.DEFINE_string("device", None, "override the default Warp device")


def _print_table(matrix, headers, title):
  num_cols = len(headers)
  col_widths = [max(len(f"{row[i]:g}") for row in matrix) for i in range(num_cols)]
  col_widths = [max(col_widths[i], len(headers[i])) for i in range(num_cols)]

  print(f"\n{title}:\n")
  print("  ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(num_cols)))
  print("-" * sum(col_widths) + "--" * 3)  # Separator line
  for row in matrix:
    print("  ".join(f"{row[i]:{col_widths[i]}g}" for i in range(num_cols)))


def _print_trace(trace, indent, steps):
  if indent == 0:
    print("\nEvent trace:\n")
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
    _print_trace(sub_trace, indent + 1, steps)


def _load_model(path: epath.Path) -> mujoco.MjModel:
  if not path.exists():
    resource_path = epath.resource_path("mujoco_warp") / path
    if not resource_path.exists():
      raise FileNotFoundError(f"file not found: {path}\nalso tried: {resource_path}")
    path = resource_path

  print(f"Loading model from: {path}...")
  if path.suffix == ".mjb":
    return mujoco.MjModel.from_binary_path(path.as_posix())

  spec = mujoco.MjSpec.from_file(path.as_posix())
  # check if the file has any mujoco.sdf test plugins
  if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
    from mujoco_warp.test_data.collision_sdf.utils import register_sdf_plugins as register_sdf_plugins

    register_sdf_plugins(mjw)

  return spec.compile()


def _override(model: mjw.Model):
  enum_fields = {
    "opt.integrator": mjw.IntegratorType,
    "opt.cone": mjw.ConeType,
    "opt.solver": mjw.SolverType,
    "opt.broadphase": mjw.BroadphaseType,
    "opt.broadphase_filter": mjw.BroadphaseFilter,
  }
  for override in _OVERRIDE.value:
    if "=" not in override:
      raise app.UsageError(f"Invalid override format: {override}")
    key, val = override.split("=", 1)
    key, val = key.strip(), val.strip()

    if key in enum_fields:
      try:
        val = int(enum_fields[key][val.upper()])
      except KeyError:
        raise app.UsageError(f"Unrecognized enum value: {val}")

    obj, attrs = model, key.split(".")
    for i, attr in enumerate(attrs):
      if not hasattr(obj, attr):
        raise app.UsageError(f"Unrecognized model field: {key}")
      if i < len(attrs) - 1:
        obj = getattr(obj, attr)
      else:
        setattr(obj, attr, val)


def _main(argv: Sequence[str]):
  """Runs testpeed app."""

  if len(argv) < 2:
    raise app.UsageError("Missing required input: mjcf path.")
  elif len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")

  mjm = _load_model(epath.Path(argv[1]))
  mjd = mujoco.MjData(mjm)
  if mjm.nkey > 0 and _KEYFRAME.value > -1:
    mujoco.mj_resetDataKeyframe(mjm, mjd, _KEYFRAME.value)
  # populate some constraints
  mujoco.mj_forward(mjm, mjd)

  wp.config.quiet = flags.FLAGS["verbosity"].value < 1
  wp.init()
  if _CLEAR_KERNEL_CACHE.value:
    wp.clear_kernel_cache()

  with wp.ScopedDevice(_DEVICE.value):
    m = mjw.put_model(mjm)
    _override(m)

    broadphase, filter = mjw.BroadphaseType(m.opt.broadphase).name, mjw.BroadphaseFilter(m.opt.broadphase_filter).name
    solver, cone = mjw.SolverType(m.opt.solver).name, mjw.ConeType(m.opt.cone).name
    integrator = mjw.IntegratorType(m.opt.integrator).name
    iterations, ls_iterations, ls_parallel = m.opt.iterations, m.opt.ls_iterations, m.opt.ls_parallel
    print(
      f"  nbody: {m.nbody} nv: {m.nv} ngeom: {m.ngeom} nu: {m.nu} is_sparse: {m.opt.is_sparse}\n"
      f"  broadphase: {broadphase} broadphase_filter: {filter}\n"
      f"  solver: {solver} cone: {cone} iterations: {iterations} ls_iterations: {ls_iterations} ls_parallel: {ls_parallel}\n"
      f"  integrator: {integrator} graph_conditional: {m.opt.graph_conditional}"
    )
    d = mjw.put_data(mjm, mjd, nworld=_NWORLD.value, nconmax=_NCONMAX.value, njmax=_NJMAX.value)
    print(f"Data\n  nworld: {d.nworld} nconmax: {d.nconmax} njmax: {d.njmax}\n")

    print(f"Rolling out {_NSTEP.value} steps at dt = {m.opt.timestep.numpy()[0]:.3f}...")

    fn = _FUNCS[_FUNCTION.value]
    res = mjw.benchmark(fn, m, d, _NSTEP.value, _EVENT_TRACE.value, _MEASURE_ALLOC.value, _MEASURE_SOLVER.value)
    jit_time, run_time, trace, ncon, nefc, solver_niter = res
    steps = _NWORLD.value * _NSTEP.value

    print(f"""
Summary for {_NWORLD.value} parallel rollouts

Total JIT time: {jit_time:.2f} s
Total simulation time: {run_time:.2f} s
Total steps per second: {steps / run_time:,.0f}
Total realtime factor: {steps * m.opt.timestep.numpy()[0] / run_time:,.2f} x
Total time per step: {1e9 * run_time / steps:.2f} ns""")

    if trace:
      _print_trace(trace, 0, steps)

    if ncon and nefc:
      idx = 0
      ncon_matrix, nefc_matrix = [], []
      for i in range(_NUM_BUCKETS.value):
        size = _NSTEP.value // _NUM_BUCKETS.value + (i < (_NSTEP.value % _NUM_BUCKETS.value))
        ncon_arr = np.array(ncon[idx : idx + size])
        nefc_arr = np.array(nefc[idx : idx + size])
        ncon_matrix.append([np.mean(ncon_arr), np.std(ncon_arr), np.min(ncon_arr), np.max(ncon_arr)])
        nefc_matrix.append([np.mean(nefc_arr), np.std(nefc_arr), np.min(nefc_arr), np.max(nefc_arr)])
        idx += size

      _print_table(ncon_matrix, ("mean", "std", "min", "max"), "ncon alloc")
      _print_table(nefc_matrix, ("mean", "std", "min", "max"), "nefc alloc")

    if solver_niter:
      idx = 0
      matrix = []
      for i in range(_NUM_BUCKETS.value):
        size = _NSTEP.value // _NUM_BUCKETS.value + (i < (_NSTEP.value % _NUM_BUCKETS.value))
        arr = np.array(solver_niter[idx : idx + size])
        matrix.append([np.mean(arr), np.std(arr), np.min(arr), np.max(arr)])
        idx += size

      _print_table(matrix, ("mean", "std", "min", "max"), "solver niter")


def main():
  # absl flags assumes __main__ is the main running module for printing usage documentation
  # pyproject bin scripts break this assumption, so manually set argv and docstring
  sys.argv[0] = "mujoco_warp.testspeed"
  sys.modules["__main__"].__doc__ = __doc__
  app.run(_main)


if __name__ == "__main__":
  main()
