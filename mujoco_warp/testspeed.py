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

"""Run benchmarks on various devices."""

import inspect
from typing import Sequence

import mujoco
import numpy as np
import warp as wp
from absl import app
from absl import flags
from etils import epath

import mujoco_warp as mjwarp

_FUNCTION = flags.DEFINE_enum(
  "function",
  "step",
  [n for n, _ in inspect.getmembers(mjwarp, inspect.isfunction)],
  "the function to run",
)
_MJCF = flags.DEFINE_string("mjcf", None, "path to model `.xml` or `.mjb`", required=True)
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
_KEYFRAME = flags.DEFINE_integer("keyframe", 0, "Keyframe to initialize simulation.")
_OUTPUT = flags.DEFINE_enum("output", "text", ["text", "tsv"], "format to print results")
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)")
_EVENT_TRACE = flags.DEFINE_bool("event_trace", False, "Provide a full event trace")
_MEASURE_ALLOC = flags.DEFINE_bool("measure_alloc", False, "Measure how much of nconmax, njmax is used.")
_MEASURE_SOLVER = flags.DEFINE_bool("measure_solver", False, "Measure the number of solver iterations.")
_NUM_BUCKETS = flags.DEFINE_integer("num_buckets", 10, "Number of buckets to summarize measurements.")
_INTEGRATOR = flags.DEFINE_string("integrator", None, "Integrator (mjtIntegrator).")
_DEVICE = flags.DEFINE_string("device", None, "Override the default Warp device.")


def _print_table(matrix, headers):
  num_cols = len(headers)
  col_widths = [max(len(f"{row[i]:g}") for row in matrix) for i in range(num_cols)]
  col_widths = [max(col_widths[i], len(headers[i])) for i in range(num_cols)]

  print("  ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(num_cols)))
  print("-" * sum(col_widths) + "--" * 3)  # Separator line
  for row in matrix:
    print("  ".join(f"{row[i]:{col_widths[i]}g}" for i in range(num_cols)))


def _print_trace(trace, indent, steps):
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


def _load_model(path):
  spec = mujoco.MjSpec.from_file(path)
  # check if the file has any mujoco.sdf test plugins
  if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
    from mujoco_warp.test_data.collision_sdf.utils import register_sdf_plugins as register_sdf_plugins

    register_sdf_plugins(mjwarp.collision_sdf)
  return spec.compile()


def _main(argv: Sequence[str]):
  """Runs testpeed function."""
  wp.init()
  path = epath.Path(_MJCF.value)
  if not path.exists():
    path = epath.resource_path("mujoco_warp") / _MJCF.value
  if not path.exists():
    raise FileNotFoundError(f"file not found: {_MJCF.value}\nalso tried: {path}")
  if path.suffix == ".mjb":
    mjm = mujoco.MjModel.from_binary_path(path.as_posix())
  else:
    mjm = _load_model(path.as_posix())

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

  mjd = mujoco.MjData(mjm)
  if mjm.nkey > 0 and _KEYFRAME.value > -1:
    mujoco.mj_resetDataKeyframe(mjm, mjd, _KEYFRAME.value)
  # populate some constraints
  mujoco.mj_forward(mjm, mjd)

  with wp.ScopedDevice(_DEVICE.value):
    m = mjwarp.put_model(mjm)

    # integrator
    IntegratorType = mjwarp._src.types.IntegratorType
    integrators = {IntegratorType.EULER: "Euler", IntegratorType.IMPLICITFAST: "implicitfast", IntegratorType.RK4: "RK4"}
    integrator = integrators[m.opt.integrator]

    if _INTEGRATOR.value is not None:
      for k, v in integrators.items():
        if _INTEGRATOR.value == v:
          integrator = v
          m.opt.integrator = k

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
      f"linesearch: {linesearch_name} "
      f"integrator: {integrator}"
    )
    print(f"Data nworld: {d.nworld} nconmax: {d.nconmax} njmax: {d.njmax}")
    print(f"Rolling out {_NSTEP.value} steps at dt = {m.opt.timestep.numpy()[0]:.3f}...")
    jit_time, run_time, trace, ncon, nefc, solver_niter = mjwarp.benchmark(
      mjwarp.__dict__[_FUNCTION.value],
      m,
      d,
      _NSTEP.value,
      _EVENT_TRACE.value,
      _MEASURE_ALLOC.value,
      _MEASURE_SOLVER.value,
    )
    steps = _BATCH_SIZE.value * _NSTEP.value

    name = argv[0]
    if _OUTPUT.value == "text":
      print(f"""
Summary for {_BATCH_SIZE.value} parallel rollouts

 Total JIT time: {jit_time:.2f} s
 Total simulation time: {run_time:.2f} s
 Total steps per second: {steps / run_time:,.0f}
 Total realtime factor: {steps * m.opt.timestep.numpy()[0] / run_time:,.2f} x
 Total time per step: {1e9 * run_time / steps:.2f} ns""")

      if trace:
        print("\nEvent trace:\n")
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

        print("\nncon alloc:\n")
        _print_table(ncon_matrix, ("mean", "std", "min", "max"))
        print("\nnefc alloc:\n")
        _print_table(nefc_matrix, ("mean", "std", "min", "max"))

      if solver_niter:
        idx = 0
        matrix = []
        for i in range(_NUM_BUCKETS.value):
          size = _NSTEP.value // _NUM_BUCKETS.value + (i < (_NSTEP.value % _NUM_BUCKETS.value))
          arr = np.array(solver_niter[idx : idx + size])
          matrix.append([np.mean(arr), np.std(arr), np.min(arr), np.max(arr)])
          idx += size

        print("\nsolver niter:\n")
        _print_table(matrix, ("mean", "std", "min", "max"))

    elif _OUTPUT.value == "tsv":
      name = name.split("/")[-1].replace("testspeed_", "")
      print(f"{name}\tjit: {jit_time:.2f}s\tsteps/second: {steps / run_time:.0f}")


def main():
  app.run(_main)


if __name__ == "__main__":
  main()
