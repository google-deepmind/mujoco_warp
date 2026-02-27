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

import warp as wp

from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model

wp.set_module_options({"enable_backward": False})


@wp.func
def _history_physical_index(cursor: int, n: int, logical: int) -> int:
  """Convert logical index (0=oldest, n-1=newest) to physical index."""
  return (cursor + 1 + logical) % n


@wp.func
def _history_find_index(
  # In:
  buf: wp.array2d(dtype=float),
  worldid: int,
  buf_offset: int,
  n: int,
  cursor: int,
  t: float,
) -> int:
  """Find logical index i such that times[i-1] < t <= times[i].

  Returns 0 if t <= times[oldest], n if t > times[newest].
  Uses linear search on the circular buffer (buffers are small, typically 2-10).
  """
  times_offset = buf_offset + 2

  oldest_phys = _history_physical_index(cursor, n, 0)
  newest_phys = _history_physical_index(cursor, n, n - 1)
  t_oldest = buf[worldid, times_offset + oldest_phys]
  t_newest = buf[worldid, times_offset + newest_phys]

  # before or at first element
  if t <= t_oldest:
    return 0

  # after last element
  if t > t_newest:
    return n

  # linear search: find smallest logical i such that times[phys(i)] >= t
  result = n
  for k in range(n):
    phys_k = _history_physical_index(cursor, n, k)
    if buf[worldid, times_offset + phys_k] >= t:
      result = k
      break

  return result


@wp.func
def _history_read_scalar(
  # In:
  buf: wp.array2d(dtype=float),
  worldid: int,
  buf_offset: int,
  n: int,
  t: float,
  interp: int,
) -> float:
  """Read a scalar value from history buffer at time t.

  interp: 0=zero-order-hold, 1=linear interpolation
  """
  cursor = int(buf[worldid, buf_offset + 1])
  times_offset = buf_offset + 2
  values_offset = buf_offset + 2 + n

  oldest_phys = _history_physical_index(cursor, n, 0)
  newest_phys = _history_physical_index(cursor, n, n - 1)
  t_oldest = buf[worldid, times_offset + oldest_phys]
  t_newest = buf[worldid, times_offset + newest_phys]

  # extrapolate before oldest
  if t <= t_oldest + MJ_MINVAL:
    return buf[worldid, values_offset + oldest_phys]

  # extrapolate after newest
  if t >= t_newest - MJ_MINVAL:
    return buf[worldid, values_offset + newest_phys]

  # find bracketing index
  i = _history_find_index(buf, worldid, buf_offset, n, cursor, t)
  phys_i = _history_physical_index(cursor, n, i)

  # exact match
  if wp.abs(t - buf[worldid, times_offset + phys_i]) < MJ_MINVAL:
    return buf[worldid, values_offset + phys_i]

  phys_lo = _history_physical_index(cursor, n, i - 1)
  phys_hi = phys_i

  # zero-order hold
  if interp == 0:
    return buf[worldid, values_offset + phys_lo]

  # linear interpolation
  dt = buf[worldid, times_offset + phys_hi] - buf[worldid, times_offset + phys_lo]
  alpha = (t - buf[worldid, times_offset + phys_lo]) / dt
  v_lo = buf[worldid, values_offset + phys_lo]
  v_hi = buf[worldid, values_offset + phys_hi]
  return v_lo + alpha * (v_hi - v_lo)


@wp.func
def _history_read_vector(
  # In:
  adr: int,
  buf: wp.array2d(dtype=float),
  worldid: int,
  buf_offset: int,
  n: int,
  dim: int,
  t: float,
  interp: int,
  # Data out:
  sensordata_out: wp.array2d(dtype=float),
) -> int:
  """Read a vector value from history buffer at time t into sensordata.

  Returns 1 on success (value written to sensordata).
  interp: 0=zero-order-hold, 1=linear interpolation
  """
  cursor = int(buf[worldid, buf_offset + 1])
  times_offset = buf_offset + 2
  values_offset = buf_offset + 2 + n

  oldest_phys = _history_physical_index(cursor, n, 0)
  newest_phys = _history_physical_index(cursor, n, n - 1)
  t_oldest = buf[worldid, times_offset + oldest_phys]
  t_newest = buf[worldid, times_offset + newest_phys]

  # extrapolate before oldest: copy oldest
  if t <= t_oldest + MJ_MINVAL:
    for d in range(dim):
      sensordata_out[worldid, adr + d] = buf[worldid, values_offset + oldest_phys * dim + d]
    return 1

  # extrapolate after newest: copy newest
  if t >= t_newest - MJ_MINVAL:
    for d in range(dim):
      sensordata_out[worldid, adr + d] = buf[worldid, values_offset + newest_phys * dim + d]
    return 1

  # find bracketing index
  i = _history_find_index(buf, worldid, buf_offset, n, cursor, t)
  phys_i = _history_physical_index(cursor, n, i)

  # exact match
  if wp.abs(t - buf[worldid, times_offset + phys_i]) < MJ_MINVAL:
    for d in range(dim):
      sensordata_out[worldid, adr + d] = buf[worldid, values_offset + phys_i * dim + d]
    return 1

  phys_lo = _history_physical_index(cursor, n, i - 1)
  phys_hi = phys_i

  # zero-order hold
  if interp == 0:
    for d in range(dim):
      sensordata_out[worldid, adr + d] = buf[worldid, values_offset + phys_lo * dim + d]
    return 1

  # linear interpolation
  dt = buf[worldid, times_offset + phys_hi] - buf[worldid, times_offset + phys_lo]
  alpha = (t - buf[worldid, times_offset + phys_lo]) / dt
  for d in range(dim):
    v_lo = buf[worldid, values_offset + phys_lo * dim + d]
    v_hi = buf[worldid, values_offset + phys_hi * dim + d]
    sensordata_out[worldid, adr + d] = v_lo + alpha * (v_hi - v_lo)
  return 1


@wp.func
def _history_insert_scalar(
  # In:
  worldid: int,
  buf_offset: int,
  n: int,
  t: float,
  value: float,
  # Out:
  buf_out: wp.array2d(dtype=float),
):
  """Insert a scalar value into history buffer at time t."""
  cursor = int(buf_out[worldid, buf_offset + 1])
  times_offset = buf_offset + 2
  values_offset = buf_offset + 2 + n

  i = _history_find_index(buf_out, worldid, buf_offset, n, cursor, t)

  # exact match
  if i < n:
    phys_i = _history_physical_index(cursor, n, i)
    if wp.abs(t - buf_out[worldid, times_offset + phys_i]) < MJ_MINVAL:
      buf_out[worldid, values_offset + phys_i] = value
      return

  # older than oldest: replace oldest
  if i == 0:
    oldest_phys = _history_physical_index(cursor, n, 0)
    buf_out[worldid, times_offset + oldest_phys] = t
    buf_out[worldid, values_offset + oldest_phys] = value
    return

  # newer than newest: advance cursor
  if i == n:
    cursor = (cursor + 1) % n
    buf_out[worldid, buf_offset + 1] = float(cursor)
    buf_out[worldid, times_offset + cursor] = t
    buf_out[worldid, values_offset + cursor] = value
    return

  # out-of-order: shift [1, i-1] left, insert at i-1
  for j in range(i - 1):
    src_phys = _history_physical_index(cursor, n, j + 1)
    dst_phys = _history_physical_index(cursor, n, j)
    buf_out[worldid, times_offset + dst_phys] = buf_out[worldid, times_offset + src_phys]
    buf_out[worldid, values_offset + dst_phys] = buf_out[worldid, values_offset + src_phys]
  insert_phys = _history_physical_index(cursor, n, i - 1)
  buf_out[worldid, times_offset + insert_phys] = t
  buf_out[worldid, values_offset + insert_phys] = value


@wp.func
def _history_insert_vector(
  # In:
  worldid: int,
  buf_offset: int,
  n: int,
  dim: int,
  t: float,
  src: wp.array2d(dtype=float),
  src_adr: int,
  # Out:
  buf_out: wp.array2d(dtype=float),
):
  """Insert a vector value from src[worldid, src_adr:src_adr+dim] into history buffer at time t."""
  cursor = int(buf_out[worldid, buf_offset + 1])
  times_offset = buf_offset + 2
  values_offset = buf_offset + 2 + n

  i = _history_find_index(buf_out, worldid, buf_offset, n, cursor, t)

  slot_phys = -1

  # exact match
  if i < n:
    phys_i = _history_physical_index(cursor, n, i)
    if wp.abs(t - buf_out[worldid, times_offset + phys_i]) < MJ_MINVAL:
      slot_phys = phys_i

  if slot_phys < 0:
    if i == 0:
      # older than oldest: replace oldest
      slot_phys = _history_physical_index(cursor, n, 0)
      buf_out[worldid, times_offset + slot_phys] = t
    elif i == n:
      # newer than newest: advance cursor
      cursor = (cursor + 1) % n
      buf_out[worldid, buf_offset + 1] = float(cursor)
      slot_phys = cursor
      buf_out[worldid, times_offset + slot_phys] = t
    else:
      # out-of-order: shift [1, i-1] left, insert at i-1
      for j in range(i - 1):
        src_phys = _history_physical_index(cursor, n, j + 1)
        dst_phys = _history_physical_index(cursor, n, j)
        buf_out[worldid, times_offset + dst_phys] = buf_out[worldid, times_offset + src_phys]
        for d in range(dim):
          buf_out[worldid, values_offset + dst_phys * dim + d] = buf_out[worldid, values_offset + src_phys * dim + d]
      slot_phys = _history_physical_index(cursor, n, i - 1)
      buf_out[worldid, times_offset + slot_phys] = t

  # copy values
  for d in range(dim):
    buf_out[worldid, values_offset + slot_phys * dim + d] = src[worldid, src_adr + d]


@wp.kernel
def _read_ctrl_delayed_kernel(
  # Model:
  actuator_history: wp.array(dtype=wp.vec2i),
  actuator_historyadr: wp.array(dtype=int),
  actuator_delay: wp.array(dtype=float),
  # Data in:
  time_in: wp.array(dtype=float),
  history_in: wp.array2d(dtype=float),
  ctrl_in: wp.array2d(dtype=float),
  # Data out:
  ctrl_out: wp.array2d(dtype=float),
):
  """Read delayed ctrl for each actuator."""
  worldid, uid = wp.tid()

  hist = actuator_history[uid]
  nsample = hist[0]

  if nsample == 0:
    # no delay: direct copy
    ctrl_out[worldid, uid] = ctrl_in[worldid, uid]
  else:
    interp = hist[1]
    delay = actuator_delay[uid]
    buf_offset = actuator_historyadr[uid]
    t = time_in[worldid] - delay
    ctrl_out[worldid, uid] = _history_read_scalar(history_in, worldid, buf_offset, nsample, t, interp)


@wp.kernel
def _insert_ctrl_history_kernel(
  # Model:
  actuator_history: wp.array(dtype=wp.vec2i),
  actuator_historyadr: wp.array(dtype=int),
  # Data in:
  time_in: wp.array(dtype=float),
  ctrl_in: wp.array2d(dtype=float),
  # Data out:
  history_out: wp.array2d(dtype=float),
):
  """Insert current ctrl into history buffers."""
  worldid, uid = wp.tid()

  hist = actuator_history[uid]
  nsample = hist[0]
  if nsample == 0:
    return

  buf_offset = actuator_historyadr[uid]
  t = time_in[worldid]
  value = ctrl_in[worldid, uid]
  _history_insert_scalar(worldid, buf_offset, nsample, t, value, history_out)


@wp.kernel
def _insert_sensor_history_stage(
  # Model:
  sensor_dim: wp.array(dtype=int),
  sensor_adr: wp.array(dtype=int),
  sensor_history: wp.array(dtype=wp.vec2i),
  sensor_historyadr: wp.array(dtype=int),
  sensor_delay: wp.array(dtype=float),
  sensor_interval: wp.array(dtype=wp.vec2),
  # Data in:
  time_in: wp.array(dtype=float),
  sensordata_in: wp.array2d(dtype=float),
  # In:
  sensor_ids: wp.array(dtype=int),
  # Data out:
  history_out: wp.array2d(dtype=float),
):
  """Insert current sensor values into history buffers for specific sensor IDs."""
  worldid, idx = wp.tid()
  sid = sensor_ids[idx]

  hist = sensor_history[sid]
  nsample = hist[0]
  if nsample == 0:
    return

  buf_offset = sensor_historyadr[sid]
  dim = sensor_dim[sid]
  interval_val = sensor_interval[sid]
  period = interval_val[0]
  t = time_in[worldid]

  if period > 0.0:
    # interval mode: check if condition is satisfied
    time_prev = history_out[worldid, buf_offset]  # user slot stores time_prev
    if time_prev + period <= t:
      # advance time_prev by exact period
      history_out[worldid, buf_offset] = time_prev + period
      # insert sensor value
      _history_insert_vector(worldid, buf_offset, nsample, dim, t, sensordata_in, sensor_adr[sid], history_out)
  else:
    _history_insert_vector(worldid, buf_offset, nsample, dim, t, sensordata_in, sensor_adr[sid], history_out)


@wp.kernel
def _apply_sensor_delay_kernel(
  # Model:
  sensor_dim: wp.array(dtype=int),
  sensor_adr: wp.array(dtype=int),
  sensor_history: wp.array(dtype=wp.vec2i),
  sensor_historyadr: wp.array(dtype=int),
  sensor_delay: wp.array(dtype=float),
  sensor_interval: wp.array(dtype=wp.vec2),
  # Data in:
  time_in: wp.array(dtype=float),
  history_in: wp.array2d(dtype=float),
  # In:
  sensor_ids: wp.array(dtype=int),
  # Data out:
  sensordata_out: wp.array2d(dtype=float),
):
  """Apply delay/interval logic for sensors after computation."""
  worldid, idx = wp.tid()
  sid = sensor_ids[idx]

  hist = sensor_history[sid]
  nsample = hist[0]
  if nsample <= 0:
    return

  delay = sensor_delay[sid]
  dim = sensor_dim[sid]
  interp = hist[1]
  buf_offset = sensor_historyadr[sid]
  t = time_in[worldid]

  if delay > 0.0:
    # delay > 0: read delayed value from buffer
    _history_read_vector(sensor_adr[sid], history_in, worldid, buf_offset, nsample, dim, t - delay, interp, sensordata_out)
  else:
    # interval-only (delay == 0, interval > 0): check interval condition
    interval_val = sensor_interval[sid]
    period = interval_val[0]
    if period > 0.0:
      time_prev = history_in[worldid, buf_offset]  # user slot
      if time_prev + period > t:
        # interval condition not satisfied: read from buffer
        _history_read_vector(sensor_adr[sid], history_in, worldid, buf_offset, nsample, dim, t, interp, sensordata_out)
      # else: interval condition satisfied, keep computed value


def read_ctrl_delayed(m: Model, d: Data, ctrl: wp.array2d(dtype=float)):
  """Read delayed ctrl values for all actuators."""
  if m.nhistory == 0:
    wp.copy(ctrl, d.ctrl)
    return

  wp.launch(
    _read_ctrl_delayed_kernel,
    dim=(d.nworld, m.nu),
    inputs=[
      m.actuator_history,
      m.actuator_historyadr,
      m.actuator_delay,
      d.time,
      d.history,
      d.ctrl,
    ],
    outputs=[ctrl],
  )


def insert_ctrl_history(m: Model, d: Data):
  """Insert current ctrl values into history buffers."""
  if m.nhistory == 0 or m.nu == 0:
    return

  wp.launch(
    _insert_ctrl_history_kernel,
    dim=(d.nworld, m.nu),
    inputs=[
      m.actuator_history,
      m.actuator_historyadr,
      d.time,
      d.ctrl,
    ],
    outputs=[d.history],
  )


def apply_sensor_delay(m: Model, d: Data, sensorid: wp.array(dtype=int)):
  """Apply delay/interval logic for given sensors after computation.

  Also inserts current (undelayed) sensor values into history buffers first.
  This must be called AFTER sensor computation and BEFORE time advance.
  """
  if m.nhistory == 0 or sensorid.shape[0] == 0:
    return

  # First, insert current (fresh) sensor values into history buffers
  wp.launch(
    _insert_sensor_history_stage,
    dim=(d.nworld, sensorid.shape[0]),
    inputs=[
      m.sensor_dim,
      m.sensor_adr,
      m.sensor_history,
      m.sensor_historyadr,
      m.sensor_delay,
      m.sensor_interval,
      d.time,
      d.sensordata,
      sensorid,
    ],
    outputs=[d.history],
  )

  # Then, overwrite sensordata with delayed values
  wp.launch(
    _apply_sensor_delay_kernel,
    dim=(d.nworld, sensorid.shape[0]),
    inputs=[
      m.sensor_dim,
      m.sensor_adr,
      m.sensor_history,
      m.sensor_historyadr,
      m.sensor_delay,
      m.sensor_interval,
      d.time,
      d.history,
      sensorid,
    ],
    outputs=[d.sensordata],
  )


@wp.kernel
def _read_ctrl_kernel(
  # Model:
  actuator_history: wp.array(dtype=wp.vec2i),
  actuator_historyadr: wp.array(dtype=int),
  actuator_delay: wp.array(dtype=float),
  # Data in:
  time_in: wp.array(dtype=float),
  history_in: wp.array2d(dtype=float),
  ctrl_in: wp.array2d(dtype=float),
  # In:
  uid: int,
  interp: int,
  # Out:
  result_out: wp.array(dtype=float),
):
  """Read delayed ctrl for 1 actuator across all worlds."""
  worldid = wp.tid()

  hist = actuator_history[uid]
  nsample = hist[0]

  if nsample == 0:
    result_out[worldid] = ctrl_in[worldid, uid]
  else:
    interp_val = interp
    if interp_val < 0:
      interp_val = hist[1]
    delay = actuator_delay[uid]
    buf_offset = actuator_historyadr[uid]
    t = time_in[worldid] - delay
    result_out[worldid] = _history_read_scalar(history_in, worldid, buf_offset, nsample, t, interp_val)


def read_ctrl(
  m: Model,
  d: Data,
  ctrlid: int,
  time: wp.array(dtype=float),
  interp: int,
  result: wp.array2d(dtype=float),
):
  """Read delayed ctrl for 1 actuator across all worlds.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    ctrlid: actuator index.
    time: query time per world (nworld,).
    interp: interpolation order (-1=model default, 0=ZOH, 1=linear).
    result: output buffer (nworld,).
  """
  wp.launch(
    _read_ctrl_kernel,
    dim=(d.nworld,),
    inputs=[
      m.actuator_history,
      m.actuator_historyadr,
      m.actuator_delay,
      time,
      d.history,
      d.ctrl,
      ctrlid,
      interp,
    ],
    outputs=[result],
  )


@wp.kernel
def _read_sensor_kernel(
  # Model:
  sensor_dim: wp.array(dtype=int),
  sensor_adr: wp.array(dtype=int),
  sensor_history: wp.array(dtype=wp.vec2i),
  sensor_historyadr: wp.array(dtype=int),
  sensor_delay: wp.array(dtype=float),
  # Data in:
  time_in: wp.array(dtype=float),
  history_in: wp.array2d(dtype=float),
  sensordata_in: wp.array2d(dtype=float),
  # In:
  sid: int,
  interp: int,
  # Out:
  result_out: wp.array2d(dtype=float),
):
  """Read delayed sensor for 1 sensor across all worlds."""
  worldid = wp.tid()

  hist = sensor_history[sid]
  nsample = hist[0]
  dim = sensor_dim[sid]
  adr = sensor_adr[sid]

  if nsample == 0:
    for i in range(dim):
      result_out[worldid, i] = sensordata_in[worldid, adr + i]
  else:
    interp_val = interp
    if interp_val < 0:
      interp_val = hist[1]
    delay = sensor_delay[sid]
    buf_offset = sensor_historyadr[sid]
    t = time_in[worldid] - delay
    _history_read_vector(
      adr,
      history_in,
      worldid,
      buf_offset,
      nsample,
      dim,
      t,
      interp_val,
      result_out,
    )


def read_sensor(
  m: Model,
  d: Data,
  sensorid: int,
  time: wp.array(dtype=float),
  interp: int,
  result: wp.array2d(dtype=float),
):
  """Read delayed sensor for 1 sensor across all worlds.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    sensorid: sensor index.
    time: query time per world (nworld,).
    interp: interpolation order (-1=model default, 0=ZOH, 1=linear).
    result: output buffer (nworld, dim).
  """
  wp.launch(
    _read_sensor_kernel,
    dim=(d.nworld,),
    inputs=[
      m.sensor_dim,
      m.sensor_adr,
      m.sensor_history,
      m.sensor_historyadr,
      m.sensor_delay,
      time,
      d.history,
      d.sensordata,
      sensorid,
      interp,
    ],
    outputs=[result],
  )


@wp.kernel
def _init_ctrl_history_kernel(
  # kernel_analyzer: off
  # Model:
  actuator_history: wp.array(dtype=wp.vec2i),
  actuator_historyadr: wp.array(dtype=int),
  # In:
  ctrlid: int,
  times: wp.array(dtype=float),
  values: wp.array2d(dtype=float),
  has_times: int,
  # Data out:
  history_out: wp.array2d(dtype=float),
  # kernel_analyzer: on
):
  """Initialize history buffer for 1 actuator across all worlds."""
  worldid = wp.tid()

  nsample = actuator_history[ctrlid][0]
  buf_offset = actuator_historyadr[ctrlid]

  # preserve user slot
  user = history_out[worldid, buf_offset]

  # cursor = 0 (samples in order, newest at index nsample-1)
  history_out[worldid, buf_offset + 1] = float(nsample - 1)

  times_offset = buf_offset + 2
  values_offset = buf_offset + 2 + nsample

  for i in range(nsample):
    if has_times != 0:
      history_out[worldid, times_offset + i] = times[i]
    history_out[worldid, values_offset + i] = values[worldid, i]

  # restore user slot
  history_out[worldid, buf_offset] = user


def init_ctrl_history(
  m: Model,
  d: Data,
  ctrlid: int,
  times: wp.array(dtype=float),
  values: wp.array2d(dtype=float),
):
  """Initialize history buffer for 1 actuator across all worlds.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    ctrlid: actuator index.
    times: timestamps or None (nworld,).
    values: ctrl values (nworld, nsample).
  """
  has_times = 0 if times is None else 1
  if times is None:
    times = wp.empty(0, dtype=float)

  wp.launch(
    _init_ctrl_history_kernel,
    dim=(d.nworld,),
    inputs=[
      m.actuator_history,
      m.actuator_historyadr,
      ctrlid,
      times,
      values,
      has_times,
    ],
    outputs=[d.history],
  )


# kernel_analyzer: off
@wp.kernel
def _init_sensor_history_kernel(
  # Model:
  sensor_history: wp.array(dtype=wp.vec2i),
  sensor_historyadr: wp.array(dtype=int),
  sensor_dim_arr: wp.array(dtype=int),
  # In:
  sensorid: int,
  times: wp.array(dtype=float),
  values: wp.array2d(dtype=float),
  phase: wp.array(dtype=float),
  has_times: int,
  # Data out:
  history_out: wp.array2d(dtype=float),
):
  # kernel_analyzer: on
  """Initialize history buffer for 1 sensor across all worlds."""
  worldid = wp.tid()

  nsample = sensor_history[sensorid][0]
  dim = sensor_dim_arr[sensorid]
  buf_offset = sensor_historyadr[sensorid]

  # set user slot (phase = last computation time for interval sensors)
  history_out[worldid, buf_offset] = phase[worldid]

  # cursor = 0 (samples in order, newest at index nsample-1)
  history_out[worldid, buf_offset + 1] = float(nsample - 1)

  times_offset = buf_offset + 2
  values_offset = buf_offset + 2 + nsample

  for i in range(nsample):
    if has_times != 0:
      history_out[worldid, times_offset + i] = times[i]
    for j in range(dim):
      history_out[worldid, values_offset + i * dim + j] = values[worldid, i * dim + j]


def init_sensor_history(
  m: Model,
  d: Data,
  sensorid: int,
  times: wp.array(dtype=float),
  values: wp.array2d(dtype=float),
  phase: wp.array(dtype=float),
):
  """Initialize history buffer for 1 sensor across all worlds.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.
    sensorid: sensor index.
    times: timestamps or None (nworld,).
    values: sensor values (nworld, nsample * dim).
    phase: user slot value per world (nworld,).
  """
  has_times = 0 if times is None else 1
  if times is None:
    times = wp.empty(0, dtype=float)

  wp.launch(
    _init_sensor_history_kernel,
    dim=(d.nworld,),
    inputs=[
      m.sensor_history,
      m.sensor_historyadr,
      m.sensor_dim,
      sensorid,
      times,
      values,
      phase,
      has_times,
    ],
    outputs=[d.history],
  )
