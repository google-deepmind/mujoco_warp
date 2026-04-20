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

"""Utilities for benchmarking MuJoCo Warp."""

import dataclasses
import json
import shutil
import time
from typing import Callable, Tuple

import mujoco
import numpy as np
import warp as wp
from etils import epath

from mujoco_warp._src import warp_util
from mujoco_warp._src.io import override_model
from mujoco_warp._src.io import put_data
from mujoco_warp._src.io import put_model
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import RenderContext
from mujoco_warp._src.util_misc import halton


@dataclasses.dataclass
class Metrics:
  """Metrics collected during benchmarking.

  Attributes:
    jit_duration: Time spent JIT compiling (seconds).
    run_time: Total simulation time (seconds).
    steps_per_second: Throughput (steps/sec).
    converged_worlds: Number of worlds that converged (e.g. stayed numerically stable).
    model_memory: Model memory usage (bytes).
    data_memory: Data memory usage (bytes).
    total_memory: Total memory used (bytes).
    ncon_mean: Mean number of contacts per world.
    ncon_p95: 95th percentile of contacts per world.
    nefc_mean: Mean number of constraints.
    nefc_p95: 95th percentile of constraints.
    solver_niter_mean: Mean solver iterations.
    solver_niter_p95: 95th percentile of solver iterations.
    model_mem_breakdown: Detailed model memory breakdown.
    data_mem_breakdown: Detailed data memory breakdown.
    trace_metrics: Flattened event trace timings (microseconds per step).
    raw_trace: Raw event trace data.
    raw_nacon: Raw number of contacts per step.
    raw_nefc: Raw number of constraints per step.
    raw_solver_niter: Raw solver iterations per step.
  """

  free_mem_at_init: float
  jit_duration: float
  run_time: float
  steps_per_second: float
  converged_worlds: int

  model_memory: float
  data_memory: float
  total_memory: float

  ncon_mean: float
  ncon_p95: float
  nefc_mean: float
  nefc_p95: float

  solver_niter_mean: float
  solver_niter_p95: float

  model_mem_breakdown: list[tuple[str, int]]
  data_mem_breakdown: list[tuple[str, int]]

  trace: dict = dataclasses.field(default_factory=dict)
  nacon: list = dataclasses.field(default_factory=list)
  nefc: list = dataclasses.field(default_factory=list)
  solver_niter: list = dataclasses.field(default_factory=list)

  def _flatten_trace(self) -> dict[str, float]:
    """Flatten trace metrics."""
    trace_metrics = {}
    steps = self.steps_per_second * self.run_time
    if not steps:
      return {}

    def flatten(prefix: str, trace, metrics):
      for k, v in trace.items():
        times, sub_trace = v
        for i, t in enumerate(times):
          metrics[f"{prefix}{k}{f'[{i}]' if len(times) > 1 else ''}"] = 1e6 * t / steps
        flatten(f"{prefix}{k}.", sub_trace, metrics)

    flatten("", self.trace, trace_metrics)
    return trace_metrics

  def to_short(self) -> str:
    """Return metrics in a short format string."""
    metrics_dict = dataclasses.asdict(self)

    # Flatten trace metrics
    trace_metrics = self._flatten_trace()
    metrics_dict.update(trace_metrics)

    exclude = {"trace", "nacon", "nefc", "solver_niter", "free_mem_at_init", "model_mem_breakdown", "data_mem_breakdown"}
    filtered_metrics = {k: v for k, v in metrics_dict.items() if k not in exclude and v is not None}

    max_key_len = max(len(key) for key in filtered_metrics.keys())
    lines = []
    for key, value in filtered_metrics.items():
      lines.append(f"{key:<{max_key_len}} {value}")
    return "\n".join(lines)

  def to_json(self) -> str:
    """Return metrics in a JSON format string."""
    metrics_dict = dataclasses.asdict(self)

    # Flatten trace metrics
    trace_metrics = self._flatten_trace()
    metrics_dict.update(trace_metrics)

    exclude = {"trace", "nacon", "nefc", "solver_niter", "free_mem_at_init", "model_mem_breakdown", "data_mem_breakdown"}
    json_metrics = {k: v for k, v in metrics_dict.items() if k not in exclude and v is not None}
    return json.dumps(json_metrics)


def _sum(stack1, stack2):
  ret = {}
  for k in stack1:
    times1, sub_stack1 = stack1[k]
    times2, sub_stack2 = stack2[k]
    times = [t1 + t2 for t1, t2 in zip(times1, times2)]
    ret[k] = (times, _sum(sub_stack1, sub_stack2))
  return ret


@wp.kernel
def ctrl_noise(
  # Model:
  opt_timestep: wp.array[float],
  actuator_ctrllimited: wp.array[bool],
  actuator_ctrlrange: wp.array2d[wp.vec2],
  # Data in:
  ctrl_in: wp.array2d[float],
  # In:
  ctrl_center: wp.array[float],
  step: int,
  ctrlnoisestd: float,
  ctrlnoiserate: float,
  # Data out:
  ctrl_out: wp.array2d[float],
):
  worldid, actid = wp.tid()

  # convert rate and scale to discrete time (Ornstein-Uhlenbeck)
  rate = wp.exp(-opt_timestep[worldid % opt_timestep.shape[0]] / ctrlnoiserate)
  scale = ctrlnoisestd * wp.sqrt(1.0 - rate * rate)

  midpoint = 0.0
  halfrange = 1.0
  ctrlrange = actuator_ctrlrange[worldid % actuator_ctrlrange.shape[0], actid]
  is_limited = actuator_ctrllimited[actid]
  if is_limited:
    midpoint = 0.5 * (ctrlrange[1] + ctrlrange[0])
    halfrange = 0.5 * (ctrlrange[1] - ctrlrange[0])
  if ctrl_center.shape[0] > 0:
    midpoint = ctrl_center[actid]

  # exponential convergence to midpoint at ctrlnoiserate
  ctrl = rate * ctrl_in[worldid, actid] + (1.0 - rate) * midpoint

  # add noise
  ctrl += scale * halfrange * (2.0 * halton((step + 1) * (worldid + 1), actid + 2) - 1.0)

  # clip to range if limited
  if is_limited:
    ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])

  ctrl_out[worldid, actid] = ctrl


def _dataclass_memory(dataclass, prefix: str = "") -> list[tuple[str, int]]:
  ret = []
  for field in dataclasses.fields(dataclass):
    value = getattr(dataclass, field.name)
    if dataclasses.is_dataclass(value):
      ret.extend(_dataclass_memory(value, prefix=f"{prefix}{field.name}."))
    elif isinstance(value, wp.array):
      ret.append((f"{prefix}{field.name}", value.capacity))
  return ret


def init(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nworld: int,
  nconmax: int | None = None,
  njmax: int | None = None,
  njmax_nnz: int | None = None,
  nccdmax: int | None = None,
  override: list[str] | None = None,
  device: str | None = None,
) -> Tuple[Model, Data, Metrics]:
  """Initialize Model, Data and partial Metrics for benchmarking.

  Args:
    mjm: MuJoCo model.
    mjd: MuJoCo data.
    nworld: Number of parallel worlds.
    nconmax: Maximum number of contacts per world.
    njmax: Maximum number of constraints per world.
    njmax_nnz: Maximum number of non-zeros in constraint Jacobian.
    nccdmax: Maximum number of CCD contacts per world.
    override: Model overrides (notation: foo.bar = baz).
    device: Warp device name.

  Returns:
    Tuple of (Model, Data, partial Metrics).
  """
  free_mem_at_init = wp.get_device(device).free_memory

  with wp.ScopedDevice(device):
    if override is not None:
      override_model(mjm, override)
    m = put_model(mjm)
    if override is not None:
      override_model(m, override)
    d = put_data(
      mjm,
      mjd,
      nworld=nworld,
      nconmax=nconmax,
      njmax=njmax,
      njmax_nnz=njmax_nnz,
      nccdmax=nccdmax,
    )

    metrics = Metrics(
      free_mem_at_init=free_mem_at_init,
      jit_duration=0.0,
      run_time=0.0,
      steps_per_second=0.0,
      converged_worlds=0,
      model_memory=0.0,
      data_memory=0.0,
      total_memory=0.0,
      ncon_mean=0.0,
      ncon_p95=0.0,
      nefc_mean=0.0,
      nefc_p95=0.0,
      solver_niter_mean=0.0,
      solver_niter_p95=0.0,
      model_mem_breakdown=[],
      data_mem_breakdown=[],
    )

    return m, d, metrics


def run(
  fn: Callable,
  m: Model,
  d: Data,
  metrics: Metrics,
  nstep: int,
  ctrls: list[np.ndarray] | None = None,
  event_trace: bool = False,
  render_context: RenderContext | None = None,
  device: str = "cuda:0",
  clear_warp_cache: bool = False,
) -> Metrics:
  """Run benchmark.

  Args:
    fn: Function to benchmark.
    m: Model.
    d: Data.
    metrics: Partially filled Metrics object.
    nstep: Number of steps to run.
    ctrls: Optional control trajectory.
    event_trace: Enable event tracing.
    render_context: Optional render context.
    device: Warp device.
    clear_warp_cache: Clear warp caches before running.

  Returns:
    Completed Metrics object.
  """
  with wp.ScopedDevice(device):
    if clear_warp_cache:
      wp.clear_kernel_cache()
      wp.clear_lto_cache()
      compute_cache = epath.Path("~/.nv/ComputeCache").expanduser()
      if compute_cache.exists():
        shutil.rmtree(compute_cache)
        compute_cache.mkdir()

    trace = {}
    nacon, nefc, solver_niter = [], [], []
    center = wp.array([], dtype=wp.float32)

    with warp_util.EventTracer(enabled=event_trace) as tracer:
      # capture the whole function as a CUDA graph
      jit_beg = time.perf_counter()

      if render_context is not None:
        with wp.ScopedCapture() as capture:
          fn(m, d, render_context)
      else:
        with wp.ScopedCapture() as capture:
          fn(m, d)

      jit_end = time.perf_counter()
      jit_duration = jit_end - jit_beg

      graph = capture.graph

      time_vec = np.zeros(nstep)
      for i in range(nstep):
        with wp.ScopedStream(wp.get_stream()):
          if ctrls is not None:
            center = wp.array(ctrls[i], dtype=wp.float32)
          wp.launch(
            ctrl_noise,
            dim=(d.nworld, m.nu),
            inputs=[m.opt.timestep, m.actuator_ctrllimited, m.actuator_ctrlrange, d.ctrl, center, i, 0.01, 0.1],
            outputs=[d.ctrl],
          )
          wp.synchronize()

          run_beg = time.perf_counter()
          wp.capture_launch(graph)
          wp.synchronize()
          run_end = time.perf_counter()

        time_vec[i] = run_end - run_beg
        if trace:
          trace = _sum(trace, tracer.trace())
        else:
          trace = tracer.trace()

        # Always collect allocations and solver iterations
        nacon.append(np.max([d.nacon.numpy()[0], d.ncollision.numpy()[0]]))
        nefc.append(np.max(d.nefc.numpy()))
        solver_niter.append(d.solver_niter.numpy())

      nconverged = np.sum(~np.any(np.isnan(d.qpos.numpy()), axis=1))
      run_duration = np.sum(time_vec)

  steps = d.nworld * nstep
  model_mem = _dataclass_memory(m)
  data_mem = _dataclass_memory(d)

  return Metrics(
    free_mem_at_init=metrics.free_mem_at_init,
    jit_duration=jit_duration,
    run_time=run_duration,
    steps_per_second=steps / run_duration,
    converged_worlds=int(nconverged),
    model_memory=sum(c for _, c in model_mem),
    data_memory=sum(c for _, c in data_mem),
    total_memory=metrics.free_mem_at_init - wp.get_device(device).free_memory,
    ncon_mean=np.mean(nacon) / d.nworld,
    ncon_p95=np.percentile(nacon, 95) / d.nworld,
    nefc_mean=np.mean(nefc),
    nefc_p95=np.percentile(nefc, 95),
    solver_niter_mean=np.mean(solver_niter),
    solver_niter_p95=np.percentile(solver_niter, 95),
    model_mem_breakdown=model_mem,
    data_mem_breakdown=data_mem,
    trace=trace,
    nacon=nacon,
    nefc=nefc,
    solver_niter=solver_niter,
  )
