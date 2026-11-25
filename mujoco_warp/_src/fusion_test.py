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
"""Tests for stage fusion."""

import contextlib
import functools
from unittest import mock

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import fusion
from mujoco_warp._src.types import DeterminismType

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <camera pos="1 1 1" mode="targetbodycom" target="b1"/>
    <geom type="plane" size="5 5 .1"/>
    <body name="b1" pos="0 0 .09">
      <freejoint/>
      <geom type="capsule" size=".1 .2" pos=".1 0 0"/>
      <site name="s1" pos="0 .1 0"/>
      <body pos=".3 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="box" size=".1 .1 .1"/>
        <body pos=".3 0 0">
          <joint type="ball"/>
          <geom type="sphere" size=".1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge"/>
  </actuator>
  <sensor>
    <framepos objtype="site" objname="s1"/>
    <jointvel joint="hinge"/>
    <accelerometer site="s1"/>
  </sensor>
</mujoco>
"""


@fusion.stage("nworld", "nbody")
def _double(worldid: int, bodyid: int, a_in: wp.array2d[float], b_out: wp.array2d[float]):
  b_out[worldid, bodyid] = 2.0 * a_in[worldid, bodyid]


@fusion.stage("nworld", "nbody")
def _add_own(worldid: int, bodyid: int, b_in: wp.array2d[float], c_out: wp.array2d[float]):
  c_out[worldid, bodyid] = b_in[worldid, bodyid] + 1.0


@fusion.stage("nworld", "nbody")
def _add_root(worldid: int, bodyid: int, b_in: wp.array2d[float], c_out: wp.array2d[float]):
  c_out[worldid, bodyid] = b_in[worldid, 0] + 1.0


@wp.struct
class _Rows:
  x: wp.array2d[float]


@fusion.stage("nworld", "nbody")
def _write_rows(worldid: int, bodyid: int, a_in: wp.array2d[float], rows_out: _Rows):
  rows_out.x[worldid, bodyid] = a_in[worldid, bodyid]


@fusion.stage("nworld", "nbody")
def _read_rows(worldid: int, bodyid: int, rows_in: _Rows, c_out: wp.array2d[float]):
  c_out[worldid, bodyid] = rows_in.x[worldid, 0]


def _kernels(entries):
  return [[[e.stage for e in seg] for seg in g.segments] if hasattr(g, "segments") else "op" for g in fusion._plan(entries)]


class FusionTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    # fuse without a graph capture too, which "auto" only does in one
    fusion.MODE = "fused"
    self.addCleanup(setattr, fusion, "MODE", "auto")

  def test_plan(self):
    """A stage reading only its own thread's result fuses into that thread, other reads don't."""
    _, _, m, d = test_data.fixture(xml=_XML, nworld=2)
    a = wp.array(np.arange(2 * m.nbody, dtype=np.float32).reshape(2, -1))
    b, c, x = wp.empty_like(a), wp.empty_like(a), wp.empty_like(a)

    with fusion.fuse():
      fusion.launch(m, d, _double, a_in=a, b_out=b)
      fusion.launch(m, d, _add_own, b_in=b, c_out=c)
      self.assertEqual(_kernels(fusion._QUEUES[-1]), [[[_double, _add_own]]])
    np.testing.assert_array_equal(c.numpy(), 2.0 * a.numpy() + 1.0)

    with fusion.fuse():
      fusion.launch(m, d, _double, a_in=a, b_out=b)
      fusion.launch(m, d, _add_root, b_in=b, c_out=c)
      self.assertEqual(_kernels(fusion._QUEUES[-1]), [[[_double]], [[_add_root]]])
    np.testing.assert_array_equal(c.numpy(), np.broadcast_to(2.0 * a.numpy()[:, :1] + 1.0, c.shape))

    # a stage moves ahead of queued work that doesn't touch its arrays, not of work that does
    with fusion.fuse():
      fusion.launch(m, d, _double, a_in=a, b_out=b)
      fusion.call(x.zero_, writes=(x,))
      fusion.launch(m, d, _double, a_in=a, b_out=c)
      self.assertEqual(_kernels(fusion._QUEUES[-1]), [[[_double], [_double]], "op"])
      fusion.call(b.zero_, writes=(b,))
      fusion.launch(m, d, _add_own, b_in=b, c_out=c)
      self.assertEqual(_kernels(fusion._QUEUES[-1]), [[[_double], [_double]], "op", "op", [[_add_own]]])
    np.testing.assert_array_equal(c.numpy(), np.ones_like(c.numpy()))

    # zeros, fills and copies run in kernels too
    with fusion.fuse():
      fusion.launch(m, d, _double, a_in=a, b_out=b)
      fusion.zero(x)
      fusion.fill(c, 3.0)
      fusion.copy(b, a)
      self.assertEqual(
        _kernels(fusion._QUEUES[-1]),
        [[[_double], [fusion._fill_stage(wp.float32)], [fusion._fill_stage(wp.float32)]], [[fusion._copy_stage(wp.float32)]]],
      )
    np.testing.assert_array_equal(x.numpy(), 0.0)
    np.testing.assert_array_equal(c.numpy(), 3.0)
    np.testing.assert_array_equal(b.numpy(), a.numpy())

    # arrays in struct arguments are dependencies too
    rows = _Rows()
    rows.x = x
    with fusion.fuse():
      fusion.launch(m, d, _write_rows, a_in=a, rows_out=rows)
      fusion.launch(m, d, _read_rows, rows_in=rows, c_out=c)
      self.assertEqual(_kernels(fusion._QUEUES[-1]), [[[_write_rows]], [[_read_rows]]])

  def test_devices(self):
    """A stage launched on one device and then another runs on each, not on the first's command."""
    if not wp.is_cuda_available():
      self.skipTest("needs a CUDA device")
    fusion.MODE = "auto"
    _, _, m, d = test_data.fixture(xml=_XML)
    for device in ("cpu", "cuda:0"):
      with wp.ScopedDevice(device):
        a = wp.array(np.arange(m.nbody, dtype=np.float32).reshape(1, -1))
        b = wp.empty_like(a)
        fusion.launch(m, d, _double, a_in=a, b_out=b)
        np.testing.assert_array_equal(b.numpy(), 2.0 * a.numpy())

  @parameterized.parameters(1, 2)
  def test_step_matches_immediate(self, nworld):
    """Fused steps write the same Data as launching every stage when called, in fewer kernels."""
    mjm, mjd, _, _ = test_data.fixture(xml=_XML, qvel_noise=0.5, ctrl_noise=0.5)
    m = mjw.put_model(mjm)
    m.opt.deterministic = DeterminismType.ATOMICS | DeterminismType.CONTACTS | DeterminismType.CONSTRAINT

    results, nkernel = [], []
    for mode in ("immediate", "unfused", "fused"):
      d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=32, njmax=64)
      if nworld == 2:
        qvel = d.qvel.numpy()
        qvel[1] += 0.3
        d.qvel.assign(qvel)
      fusion.MODE = mode
      launch, count = fusion._launch_kernel, [0]
      fusion._launch_kernel = lambda k, values: count.__setitem__(0, count[0] + 1) or launch(k, values)
      try:
        for _ in range(3):
          mjw.step(m, d)
      finally:
        fusion._launch_kernel = launch
      results.append(d)
      nkernel.append(count[0])

    immediate, unfused, fused = results
    for field in ("qpos", "qvel", "act", "qacc", "qfrc_constraint", "sensordata", "cam_xpos", "nacon", "nefc"):
      ref = getattr(immediate, field).numpy()
      np.testing.assert_array_equal(getattr(unfused, field).numpy(), ref, err_msg=field)
      np.testing.assert_array_equal(getattr(fused, field).numpy(), ref, err_msg=field)
    self.assertGreater(fused.nacon.numpy()[0], 0)
    self.assertLess(nkernel[2], nkernel[1])
    if nworld == 2:
      self.assertFalse(np.array_equal(fused.qpos.numpy()[0], fused.qpos.numpy()[1]))

  def test_step_work_goes_through_fusion(self):
    """No device work in a step runs directly while stages are queued, which would pass them."""
    mjm, mjd, _, _ = test_data.fixture(xml=_XML, qvel_noise=0.5, ctrl_noise=0.5)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nconmax=32, njmax=64)

    def guard(fn):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        self.assertFalse(fusion.pending(), f"{fn.__name__} runs while stages are queued")
        return fn(*args, **kwargs)

      return wrapper

    with contextlib.ExitStack() as stack:
      for obj, name in (
        (wp, "launch"),
        (wp, "launch_tiled"),
        (wp, "copy"),
        (wp, "clone"),
        (wp, "capture_while"),
        (wp, "capture_if"),
      ):
        stack.enter_context(mock.patch.object(obj, name, guard(getattr(obj, name))))
      for name in ("assign", "numpy"):
        stack.enter_context(mock.patch.object(wp.array, name, guard(getattr(wp.array, name))))
      for _ in range(3):
        mjw.step(m, d)


if __name__ == "__main__":
  absltest.main()
