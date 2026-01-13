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

import inspect
import sys
from dataclasses import fields
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

_FUNCS = {n: f for n, f in inspect.getmembers(mjw, inspect.isfunction) if inspect.signature(f).parameters.keys() == {"m", "d"}}

_FUNCTION = flags.DEFINE_enum("function", "step", _FUNCS.keys(), "the function to benchmark")
_NSTEP = flags.DEFINE_integer("nstep", 1000, "number of steps per rollout")
_NWORLD = flags.DEFINE_integer("nworld", 8192, "number of parallel rollouts")
_NCONMAX = flags.DEFINE_integer("nconmax", None, "override maximum number of contacts per world")
_NJMAX = flags.DEFINE_integer("njmax", None, "override maximum number of constraints per world")
_OVERRIDE = flags.DEFINE_multi_string("override", [], "Model overrides (notation: foo.bar = baz)", short_name="o")
_KEYFRAME = flags.DEFINE_integer("keyframe", 0, "keyframe to initialize simulation.")
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "clear kernel cache (to calculate full JIT time)")
_EVENT_TRACE = flags.DEFINE_bool("event_trace", False, "print an event trace report")
_MEASURE_ALLOC = flags.DEFINE_bool("measure_alloc", False, "print a report of contacts and constraints per step")
_MEASURE_SOLVER = flags.DEFINE_bool("measure_solver", False, "print a report of solver iterations per step")
_NUM_BUCKETS = flags.DEFINE_integer("num_buckets", 10, "number of buckets to summarize rollout measurements")
_DEVICE = flags.DEFINE_string("device", None, "override the default Warp device")
_REPLAY = flags.DEFINE_string("replay", None, "keyframe sequence to replay, keyframe name must prefix match")
_MEMORY = flags.DEFINE_bool("memory", False, "print memory report")


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


def _dataclass_memory(dataclass, total_bytes, conversion, unit):
  total_memory = 0
  out = ""
  for field in fields(dataclass):
    value = getattr(dataclass, field.name)
    fieldinfo = []
    if isinstance(value, mjw.Contact) or isinstance(value, mjw.Constraint):
      for vfield in fields(value):
        vvalue = getattr(value, vfield.name)
        if type(vvalue) is wp.array:
          fieldinfo.append((f"{field.name}.{vfield.name}", vvalue.capacity))
    elif type(value) is wp.array:
      fieldinfo.append((field.name, value.capacity))
    for array in fieldinfo:
      total_memory += array[1]
      field_percentage = 100 * (array[1] / total_bytes)
      # only print if field contributes at least 1% of total memory usage
      if field_percentage >= 1:
        out += f" {array[0]}: {(array[1] / conversion):.2f} {unit} ({field_percentage:.2f}%)\n"
  out = out or " (no field >= 1% of utilized memory)\n"
  return out, total_memory


def _main(argv: Sequence[str]):
  """Runs testpeed app."""
  if len(argv) < 2:
    raise app.UsageError("Missing required input: mjcf path.")
  elif len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")

  mjm = _load_model(epath.Path(argv[1]))
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

  # populate some constraints
  mujoco.mj_forward(mjm, mjd)

  wp.config.quiet = flags.FLAGS["verbosity"].value < 1
  wp.init()
  if _CLEAR_KERNEL_CACHE.value:
    wp.clear_kernel_cache()

  if (_DEVICE.value or wp.get_device()) == "cpu":
    raise ValueError("testspeed available for gpu only")

  with wp.ScopedDevice(_DEVICE.value):
    m = mjw.put_model(mjm)
    override_model(m, _OVERRIDE.value)

    broadphase, filter = mjw.BroadphaseType(m.opt.broadphase).name, mjw.BroadphaseFilter(m.opt.broadphase_filter).name
    solver, cone = mjw.SolverType(m.opt.solver).name, mjw.ConeType(m.opt.cone).name
    integrator = mjw.IntegratorType(m.opt.integrator).name
    iterations, ls_iterations = m.opt.iterations, m.opt.ls_iterations
    ls_str = f"{'parallel' if m.opt.ls_parallel else 'iterative'} linesearch iterations: {ls_iterations}"
    print(
      f"  nbody: {m.nbody} nv: {m.nv} ngeom: {m.ngeom} nu: {m.nu} is_sparse: {m.opt.is_sparse}\n"
      f"  broadphase: {broadphase} broadphase_filter: {filter}\n"
      f"  solver: {solver} cone: {cone} iterations: {iterations} {ls_str}\n"
      f"  integrator: {integrator} graph_conditional: {m.opt.graph_conditional}"
    )
    d = mjw.put_data(mjm, mjd, nworld=_NWORLD.value, nconmax=_NCONMAX.value, njmax=_NJMAX.value)
    print(f"Data\n  nworld: {d.nworld} naconmax: {d.naconmax} njmax: {d.njmax}\n")

    print(f"Rolling out {_NSTEP.value} steps at dt = {m.opt.timestep.numpy()[0]:.3f}...")

    fn = _FUNCS[_FUNCTION.value]
    res = benchmark(fn, m, d, _NSTEP.value, ctrls, _EVENT_TRACE.value, _MEASURE_ALLOC.value, _MEASURE_SOLVER.value)
    jit_time, run_time, trace, nacon, nefc, solver_niter, nsuccess = res
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
      _print_trace(trace, 0, steps)

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

      _print_table(nacon_matrix, ("mean", "std", "min", "max"), "nacon alloc")
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

    if _MEMORY.value:
      memory_unit = "MB"
      memory_conversion = 1024 * 1024
      total_bytes = wp.get_device(_DEVICE.value).total_memory
      utilized_bytes = wp.get_mempool_used_mem_current(_DEVICE.value)
      utilized_mb = utilized_bytes / memory_conversion
      total_mb = total_bytes / memory_conversion
      utilized_percentage = 100 * (utilized_bytes / total_bytes)
      memory_str = (
        f"\nTotal memory: {utilized_mb:.2f} {memory_unit} / {total_mb:.2f} {memory_unit} ({utilized_percentage:.2f}%)\n"
      )
      model_field_str, model_bytes = _dataclass_memory(m, utilized_bytes, memory_conversion, memory_unit)
      memory_str += f"Model memory ({100 * model_bytes / utilized_bytes:.2f}%):\n"
      memory_str += model_field_str
      data_field_str, data_bytes = _dataclass_memory(d, utilized_bytes, memory_conversion, memory_unit)
      memory_str += f"Data memory ({100 * data_bytes / utilized_bytes:.2f}%):\n"
      memory_str += data_field_str
      print(memory_str)


def main():
  # absl flags assumes __main__ is the main running module for printing usage documentation
  # pyproject bin scripts break this assumption, so manually set argv and docstring
  sys.argv[0] = "mujoco_warp.testspeed"
  sys.modules["__main__"].__doc__ = __doc__
  app.run(_main)


if __name__ == "__main__":
  main()
