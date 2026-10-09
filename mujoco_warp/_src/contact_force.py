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

"""Experimental physical coefficients for externally supplied rigid contacts.

This API selects fixed-J backward Euler for normal springs. Friction follows
MuJoCo's existing coupled cone metric, not an independent scalar normal law.
Eager forward/step boundaries check errors automatically. Use
launch_contact_force_graph for checked graph replay. Low-level kernel outputs
and raw graph replays remain provisional until check_contact_force_params.
"""

import numpy as np
import warp as wp

from mujoco_warp._src import types
from mujoco_warp._src.types import vec5


@wp.func
def _physical_contact_row(
  # In:
  params: wp.vec2,
  h: float,
  pos: float,
  vel: float,
  condim: int,
  dimid: int,
  elliptic: bool,
  friction: vec5,
  ir: float,
):
  """Return (D, aref) with a common valid cone metric or a sticky error code."""
  k, c = params[0], params[1]
  if not wp.isfinite(k) or not wp.isfinite(c) or not wp.isfinite(h) or k <= 0.0 or c < 0.0 or h <= 0.0:
    return wp.vec2(0.0), int(1)
  rate = c + h * k
  L = h * rate
  if not wp.isfinite(rate) or not wp.isfinite(L) or L <= 0.0:
    return wp.vec2(0.0), int(2)
  # Check every implied row before accepting any row: elliptic projection assumes
  # these exact metric ratios, so independent row clamps are not admissible.
  D = L
  if not elliptic and condim > 1:
    D = L / float(2 * (condim - 1))
  if D < 1.0e-30 or D > 1.0 / types.MJ_MINVAL:
    return wp.vec2(0.0), int(2)
  if elliptic and condim > 1:
    if not wp.isfinite(ir) or ir <= 0.0 or not wp.isfinite(friction[0]) or friction[0] <= 0.0:
      return wp.vec2(0.0), int(2)
    for i in range(1, condim):
      mu = friction[i - 1]
      if not wp.isfinite(mu) or mu <= 0.0:
        return wp.vec2(0.0), int(2)
      ratio = ir * (friction[0] / mu)
      scale = ratio * ratio
      Di = L / scale
      if not wp.isfinite(Di) or Di < 1.0e-30 or Di > 1.0 / types.MJ_MINVAL:
        return wp.vec2(0.0), int(2)
      if i == dimid:
        D = Di
  aref = -((k / rate) * pos + vel) / h
  if not wp.isfinite(aref):
    return wp.vec2(0.0), int(4)
  return wp.vec2(D, aref), int(0)


def _validate_model(m: types.Model) -> None:
  if m.opt.run_collision_detection:
    raise ValueError("Physical contact coefficients require externally supplied contacts.")
  if m.is_sparse:
    raise ValueError("Physical contact coefficients currently require a dense Jacobian.")
  if m.nflex or m.flg_adhesion or m.opt.enableflags & types.EnableBit.SLEEP:
    raise ValueError("Physical contact coefficients do not support flex, adhesion, or sleeping.")
  if m.opt.integrator not in (types.IntegratorType.EULER, types.IntegratorType.IMPLICIT, types.IntegratorType.IMPLICITFAST):
    raise ValueError("Physical contact coefficients require Euler, implicit, or implicitfast integration; RK4 is unsupported.")


def validate_contact_force_params(m: types.Model, d: types.Data) -> bool:
  """Validate static buffer contracts without a device readback."""
  params, error, active = d.contact.force_params, d.contact.force_error, d.contact.force_active
  enabled = params is not None and params.size != 0
  if not enabled:
    if (error is not None and error.size) or (active is not None and active.size):
      raise ValueError("Physical contact buffers must all be empty or enabled.")
    return False
  _validate_model(m)
  if params.shape != (d.naconmax,) or params.dtype != wp.vec2:
    raise ValueError("force_params must have naconmax vec2 entries.")
  if error is None or error.shape != (d.nworld,) or error.dtype != wp.int32:
    raise ValueError("force_error must have nworld int32 entries.")
  if active is None or active.shape != (d.nworld,) or active.dtype != wp.int32:
    raise ValueError("force_active must have nworld int32 entries.")
  if params.device != d.qvel.device or error.device != d.qvel.device or active.device != d.qvel.device:
    raise ValueError("Physical contact buffers must use the data device.")
  return True


def enable_contact_force_params(m: types.Model, d: types.Data) -> None:
  """Enable experimental physical normal coefficients on external rigid contacts.

  Populate contact.force_params with (k [N/m], c [N s/m]); exactly (0,0) retains
  legacy behavior. All other records require finite k>0,c>=0. The whole contact
  metric must satisfy 1e-30 <= D_i <= 1/MJ_MINVAL with finite reference acceleration.
  Invalid rows are neutralized and OR error bits into force_error:
  1 invalid coefficients/step, 2 unsupported metric, 4 nonfinite reference.
  Any error invalidates the world's result until reset; check after graph replay.
  No physical parameter survives reset for that world. This state is not MJCF.
  force_active is derived internal state: callers must not write it. Assembly
  recomputes it to restrict compensated sums to worlds with physical rows.
  """
  _validate_model(m)
  if d.naconmax <= 0:
    raise ValueError("Physical contact coefficients require positive contact capacity.")
  if validate_contact_force_params(m, d):
    return
  d.contact.force_params = wp.zeros(d.naconmax, dtype=wp.vec2, device=d.qvel.device)
  d.contact.force_error = wp.zeros(d.nworld, dtype=int, device=d.qvel.device)
  d.contact.force_active = wp.zeros(d.nworld, dtype=int, device=d.qvel.device)


def check_contact_force_params(d: types.Data) -> None:
  """Synchronize and reject errors recorded by prior physical-contact assembly.

  This reads sticky diagnostics; it does not validate newly assigned coefficients.
  Assemble constraints before checking, and do not mutate their inputs before
  consuming the checked result. Call on the producing stream, outside capture.
  Success covers recorded input validation, not solver convergence or accuracy.
  Errors survive contact reuse and require an explicit world reset.
  """
  error = d.contact.force_error
  if error is not None and error.size:
    if d.qvel.device.is_capturing:
      raise RuntimeError("Physical contact validation cannot synchronize during graph capture.")
    values = error.numpy()
    failed = np.flatnonzero(values)
    if failed.size:
      raise ValueError(f"Invalid physical contact coefficients: worlds {failed.tolist()}, error bits {values[failed].tolist()}")


def check_contact_force_params_eager(d: types.Data) -> None:
  """Check host execution; defer captured work to its checked replay boundary."""
  if not d.qvel.device.is_capturing:
    check_contact_force_params(d)


@wp.kernel
def _validate_timestep(
  # In:
  timestep: wp.array[float],
  # Out:
  errors_out: wp.array[int],
  active_out: wp.array[int],
):
  world = wp.tid()
  active_out[world] = 0
  h = timestep[world % timestep.shape[0]]
  if not wp.isfinite(h) or h <= 0.0:
    wp.atomic_or(errors_out, world, 1)


def validate_contact_force_timestep(m: types.Model, d: types.Data) -> None:
  """Validate each enabled world and clear its derived physical-row activity."""
  wp.launch(
    _validate_timestep,
    d.nworld,
    inputs=[m.opt.timestep],
    outputs=[d.contact.force_error, d.contact.force_active],
    device=d.qvel.device,
  )


def launch_contact_force_graph(d: types.Data, graph: wp.Graph, *, stream: wp.Stream | None = None) -> None:
  """Replay a physical-contact graph and validate its results before returning.

  This is a synchronizing host boundary, outside graph capture. The graph must
  assemble constraints for this Data instance and must not clear or reset recorded
  errors before this function checks them. All graph outputs are provisional until
  this function returns successfully. On error, discard those outputs and reset the
  affected worlds before resuming; this function does not roll back integration.
  Use the producing stream, or establish its dependency before this call.
  Raw wp.capture_launch bypasses this boundary. No thread may consume graph
  outputs while validation is pending; graph/Data ownership is a caller obligation.
  Validation certifies coefficients only, not numerical accuracy or convergence.
  """
  device = d.qvel.device
  if device.is_capturing:
    raise RuntimeError("Checked physical contact replay cannot run during graph capture.")
  if d.contact.force_error is None or not d.contact.force_error.size:
    raise ValueError("Checked physical contact replay requires enabled physical contact buffers.")
  if graph.device != device or (stream is not None and stream.device != device):
    raise ValueError("Graph, stream, and physical contact data must use the same device.")
  with wp.ScopedDevice(device), wp.ScopedStream(stream if stream is not None else device.stream):
    check_contact_force_params(d)
    wp.capture_launch(graph, stream=stream)
    check_contact_force_params(d)


@wp.kernel
def _reset_force_params(
  # In:
  reset: wp.array[bool],
  worldid: wp.array[int],
  partial: bool,
  # Out:
  params_out: wp.array[wp.vec2],
  errors_out: wp.array[int],
  active_out: wp.array[int],
):
  tid = wp.tid()
  if tid < errors_out.shape[0]:
    if not partial or reset[tid]:
      errors_out[tid] = 0
      active_out[tid] = 0
  if tid < params_out.shape[0]:
    world = worldid[tid]
    if not partial or (world >= 0 and world < errors_out.shape[0] and reset[world]):
      params_out[tid] = wp.vec2(0.0)


def reset_contact_force_params(d: types.Data, reset: wp.array | None = None) -> None:
  """Clear coefficient errors and records, including inactive contact-buffer tails.

  This primitive does not restore simulation state. Use reset_data or the
  application's full reset protocol before resuming a rejected simulation.

  Args:
    d: Data with optional physical-contact buffers.
    reset: Optional bool mask of shape (nworld,) on the data device. None clears
      every world. The separate reset_data API also normalizes integer masks.

  Raises:
    ValueError: The supplied mask has an invalid shape, dtype, or device.
  """
  if d.contact.force_params is not None and d.contact.force_params.size:
    if reset is not None and (reset.shape != (d.nworld,) or reset.dtype != wp.bool or reset.device != d.qvel.device):
      raise ValueError("Physical contact reset mask must have nworld bool entries on the data device.")
    wp.launch(
      _reset_force_params,
      max(d.naconmax, d.nworld),
      inputs=[reset, d.contact.worldid, reset is not None],
      outputs=[d.contact.force_params, d.contact.force_error, d.contact.force_active],
      device=d.qvel.device,
    )
