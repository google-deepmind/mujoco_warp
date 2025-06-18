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
from .types import EnableBit
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
  passive: bool = True,
  gravity: bool = True,
  clampctrl: bool = True,
  qpos0: bool = False,
  kick: bool = False,
  energy: bool = False,
  eulerdamp: Optional[bool] = None,
  cone: Optional[ConeType] = None,
  integrator: Optional[IntegratorType] = None,
  solver: Optional[SolverType] = None,
  iterations: Optional[int] = None,
  ls_iterations: Optional[int] = None,
  ls_parallel: Optional[bool] = None,
  sparse: Optional[bool] = None,
  disableflags: Optional[int] = None,
  enableflags: Optional[int] = None,
  applied: bool = False,
  nstep: int = 3,
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
  if not passive:
    mjm.opt.disableflags |= DisableBit.PASSIVE
  if not gravity:
    mjm.opt.disableflags |= DisableBit.GRAVITY
  if not clampctrl:
    mjm.opt.disableflags |= DisableBit.CLAMPCTRL
  if not eulerdamp:
    mjm.opt.disableflags |= DisableBit.EULERDAMP

  if energy:
    mjm.opt.enableflags |= EnableBit.ENERGY

  if cone is not None:
    mjm.opt.cone = cone
  if integrator is not None:
    mjm.opt.integrator = integrator
  if disableflags is not None:
    mjm.opt.disableflags |= disableflags
  if enableflags is not None:
    mjm.opt.enableflags |= enableflags
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
  elif qpos0:
    mjd.qpos[:] = mjm.qpos0
  else:
    # set random qpos, underlying code should gracefully handle un-normalized quats
    mjd.qpos[:] = np.random.random(mjm.nq)

  if kick:
    # give the system a little kick to ensure we have non-identity rotations
    mjd.qvel = np.random.uniform(-0.01, 0.01, mjm.nv)
    mjd.ctrl = np.random.uniform(-0.1, 0.1, size=mjm.nu)
  if applied:
    mjd.qfrc_applied = np.random.uniform(-0.1, 0.1, size=mjm.nv)
    mjd.xfrc_applied = np.random.uniform(-0.1, 0.1, size=mjd.xfrc_applied.shape)
  if kick or applied:
    mujoco.mj_step(mjm, mjd, nstep)  # let dynamics get state significantly non-zero

  if mjm.nmocap:
    mjd.mocap_pos = np.random.random(mjd.mocap_pos.shape)
    mocap_quat = np.random.random(mjd.mocap_quat.shape)
    mjd.mocap_quat = mocap_quat

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
  measure_solver_niter: bool = False,
) -> Tuple[float, float, dict, list, list, list]:
  """Benchmark a function of Model and Data."""
  jit_beg = time.perf_counter()

  fn(m, d)

  jit_end = time.perf_counter()
  jit_duration = jit_end - jit_beg
  wp.synchronize()

  trace = {}
  ncon, nefc, solver_niter = [], [], []

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
      if measure_alloc or measure_solver_niter:
        wp.synchronize()
      if measure_alloc:
        ncon.append(d.ncon.numpy()[0])
        nefc.append(d.nefc.numpy()[0])
      if measure_solver_niter:
        solver_niter.append(d.solver_niter.numpy())

    wp.synchronize()
    run_end = time.perf_counter()
    run_duration = run_end - run_beg

  return jit_duration, run_duration, trace, ncon, nefc, solver_niter
