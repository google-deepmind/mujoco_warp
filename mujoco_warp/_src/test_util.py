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
from typing import Callable, Optional, Tuple

import mujoco
import numpy as np
import warp as wp
from etils import epath

from . import io
from . import warp_util
from .types import ConeType
from .types import Data
from .types import DisableBit
from .types import IntegratorType
from .types import Model
from .types import SolverType


def fixture(
  fname: Optional[str] = None,
  xml: Optional[str] = None,
  keyframe: int = -1,
  actuation: bool = True,
  contact: bool = True,
  constraint: bool = True,
  equality: bool = True,
  gravity: bool = True,
  eulerdamp: Optional[bool] = None,
  cone: Optional[ConeType] = None,
  integrator: Optional[IntegratorType] = None,
  solver: Optional[SolverType] = None,
  iterations: Optional[int] = None,
  ls_iterations: Optional[int] = None,
  ls_parallel: Optional[bool] = None,
  sparse: Optional[bool] = None,
  disableflags: Optional[int] = None,
  kick: bool = False,
  seed: int = 42,
  nworld: int = None,
  nconmax: int = None,
  njmax: int = None,
):
  np.random.seed(seed)
  if fname is not None:
    path = epath.resource_path("mujoco_warp") / "test_data" / fname
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())
  elif xml is not None:
    mjm = mujoco.MjModel.from_xml_string(xml)
  else:
    raise ValueError("either fname or xml must be provided")
  if not actuation:
    mjm.opt.disableflags |= DisableBit.ACTUATION
  if not contact:
    mjm.opt.disableflags |= DisableBit.CONTACT
  if not constraint:
    mjm.opt.disableflags |= DisableBit.CONSTRAINT
  if not equality:
    mjm.opt.disableflags |= DisableBit.EQUALITY
  if not gravity:
    mjm.opt.disableflags |= DisableBit.GRAVITY
  if not eulerdamp:
    mjm.opt.disableflags |= DisableBit.EULERDAMP
  if cone is not None:
    mjm.opt.cone = cone
  if integrator is not None:
    mjm.opt.integrator = integrator
  if disableflags is not None:
    mjm.opt.disableflags |= disableflags
  if solver is not None:
    mjm.opt.solver = solver
  if iterations is not None:
    mjm.opt.iterations = iterations
  if ls_iterations is not None:
    mjm.opt.ls_iterations = ls_iterations
  if sparse is not None:
    if sparse:
      mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    else:
      mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE

  mjd = mujoco.MjData(mjm)
  if keyframe > -1:
    mujoco.mj_resetDataKeyframe(mjm, mjd, keyframe)

  if kick:
    # give the system a little kick to ensure we have non-identity rotations
    mjd.qvel = np.random.uniform(-0.01, 0.01, mjm.nv)
    mjd.ctrl = np.random.normal(scale=0.01, size=mjm.nu)
    mujoco.mj_step(mjm, mjd, 3)  # let dynamics get state significantly non-zero

  if mjm.nmocap:
    mjd.mocap_pos = np.random.random(mjd.mocap_pos.shape)
    mjd.mocap_quat = np.random.random(mjd.mocap_quat.shape)

  mujoco.mj_forward(mjm, mjd)
  m = io.put_model(mjm)

  if ls_parallel is not None:
    m.opt.ls_parallel = ls_parallel

  d = io.put_data(mjm, mjd, nworld=nworld, nconmax=nconmax, njmax=njmax)
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
