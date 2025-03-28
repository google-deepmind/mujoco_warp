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

"""Utilities for testing."""

import time
from typing import Callable, Tuple

import mujoco
import numpy as np
import warp as wp
from etils import epath

from . import io
from . import warp_util
from .types import Data
from .types import Model


def fixture(
  fname: str,
  keyframe: int = -1,
  sparse: bool = True,
  cone: int = mujoco.mjtCone.mjCONE_PYRAMIDAL,
):
  path = epath.resource_path("mujoco_warp") / "test_data" / fname
  mjm = mujoco.MjModel.from_xml_path(path.as_posix())
  mjm.opt.cone = cone
  mjm.opt.jacobian = sparse
  mjd = mujoco.MjData(mjm)
  if keyframe > -1:
    mujoco.mj_resetDataKeyframe(mjm, mjd, keyframe)
  # give the system a little kick to ensure we have non-identity rotations
  mjd.qvel = np.random.uniform(-0.01, 0.01, mjm.nv)
  mujoco.mj_step(mjm, mjd, 3)  # let dynamics get state significantly non-zero
  mujoco.mj_forward(mjm, mjd)
  m = io.put_model(mjm)
  d = io.put_data(mjm, mjd)
  return mjm, mjd, m, d


def _sum(stack1, stack2):
  ret = {}
  for k in stack1:
    times1, sub_stack1 = stack1[k]
    times2, sub_stack2 = stack2[k]
    times = [t1 + t2 for t1, t2 in zip(times1, times2)]
    ret[k] = (times, _sum(sub_stack1, sub_stack2))
  return ret


def benchmark(
  fn: Callable[[Model, Data], None],
  m: Model,
  d: Data,
  nstep: int,
  event_trace: bool = False,
  measure_alloc: bool = False,
) -> Tuple[float, float, dict, list, list]:
  """Benchmark a function of Model and Data."""
  jit_beg = time.perf_counter()

  fn(m, d)

  jit_end = time.perf_counter()
  jit_duration = jit_end - jit_beg
  wp.synchronize()

  trace = {}
  ncon, nefc = [], []

  with warp_util.EventTracer(enabled=event_trace) as tracer:
    # capture the whole function as a CUDA graph
    with wp.ScopedCapture() as capture:
      fn(m, d)
    graph = capture.graph

    run_beg = time.perf_counter()
    for _ in range(nstep):
      wp.capture_launch(graph)
      if trace:
        trace = _sum(trace, tracer.trace())
      else:
        trace = tracer.trace()
      if measure_alloc:
        wp.synchronize()
        ncon.append(d.ncon.numpy()[0])
        nefc.append(d.nefc.numpy()[0])
    wp.synchronize()
    run_end = time.perf_counter()
    run_duration = run_end - run_beg

  return jit_duration, run_duration, trace, ncon, nefc
