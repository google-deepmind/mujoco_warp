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

"""Consumption and lifecycle tests for experimental physical contacts."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp._src.contact_force_test import _fixture
from mujoco_warp._src.contact_force_test import _set_params


class ContactForceSafetyTest(parameterized.TestCase):
  @parameterized.parameters("forward", "step", "step1")
  def test_eager_boundary_rejects_before_integrating(self, operation):
    """Invalid physical coefficients cannot produce a successful host step."""
    _, _, m, d = _fixture()
    mjw.enable_contact_force_params(m, d)
    _set_params(d, -1.0, 0.0)
    before = d.qpos.numpy().copy()
    with self.assertRaisesRegex(ValueError, "worlds \\[0\\]"):
      getattr(mjw, operation)(m, d)
    np.testing.assert_array_equal(d.qpos.numpy(), before)

  def test_step2_rejects_sticky_error(self):
    """Split stepping must not bypass a previously rejected contact."""
    _, _, m, d = _fixture()
    mjw.enable_contact_force_params(m, d)
    d.contact.force_error.fill_(1)
    before = d.qpos.numpy().copy()
    with self.assertRaisesRegex(ValueError, "worlds \\[0\\]"):
      mjw.step2(m, d)
    np.testing.assert_array_equal(d.qpos.numpy(), before)

  @parameterized.parameters(0.0, -0.001, float("nan"), float("inf"))
  def test_empty_world_rejects_invalid_timestep(self, timestep):
    """The timestep contract also applies when the contact set is empty."""
    _, _, m, d = _fixture()
    mjw.enable_contact_force_params(m, d)
    d.nacon.zero_()
    m.opt.timestep.fill_(timestep)
    with self.assertRaisesRegex(ValueError, "worlds \\[0\\]"):
      mjw.forward(m, d)

  def test_partial_reset_preserves_unselected_errors_and_inactive_slots(self):
    """Masked reset clears only selected worlds, including inactive tails."""
    _, _, m, d = _fixture(nworld=2)
    mjw.enable_contact_force_params(m, d)
    worldid = np.arange(d.naconmax, dtype=np.int32) % 2
    d.contact.worldid.assign(worldid)
    d.contact.force_params.fill_(wp.vec2(2.0, 3.0))
    d.contact.force_error.assign(np.array([1, 4], dtype=np.int32))
    d.contact.force_active.fill_(1)
    mask = wp.array([True, False], dtype=wp.bool, device=d.qvel.device)
    mjw.reset_contact_force_params(d, mask)
    np.testing.assert_array_equal(d.contact.force_error.numpy(), [0, 4])
    np.testing.assert_array_equal(d.contact.force_active.numpy(), [0, 1])
    params = d.contact.force_params.numpy()
    np.testing.assert_array_equal(params[worldid == 0], 0)
    np.testing.assert_array_equal(params[worldid == 1], np.tile([2, 3], (sum(worldid == 1), 1)))
    mjw.reset_contact_force_params(d)
    mjw.check_contact_force_params(d)
    np.testing.assert_array_equal(d.contact.force_params.numpy(), 0)
    np.testing.assert_array_equal(d.contact.force_active.numpy(), 0)

  def test_checked_graph_recomputes_activity_after_world_transitions(self):
    """Replay rebuilds physical-row activity from current coefficients and counts."""
    if not wp.get_device().is_cuda:
      self.skipTest("CUDA graph capture requires CUDA")
    _, _, m, d = _fixture(nworld=2)
    mjw.enable_contact_force_params(m, d)
    _set_params(d, 100.0, 0.2)
    mjw.forward(m, d)
    with wp.ScopedCapture() as capture:
      mjw.forward(m, d)
    params = d.contact.force_params.numpy()
    params[d.contact.worldid.numpy() == 1] = 0
    d.contact.force_params.assign(params)
    mjw.launch_contact_force_graph(d, capture.graph)
    np.testing.assert_array_equal(d.contact.force_active.numpy(), [1, 0])
    _set_params(d, 0.0, 0.0)
    mjw.launch_contact_force_graph(d, capture.graph)
    np.testing.assert_array_equal(d.contact.force_active.numpy(), [0, 0])
    _set_params(d, 100.0, 0.2)
    mjw.launch_contact_force_graph(d, capture.graph)
    np.testing.assert_array_equal(d.contact.force_active.numpy(), [1, 1])
    d.nacon.zero_()
    mjw.launch_contact_force_graph(d, capture.graph)
    np.testing.assert_array_equal(d.contact.force_active.numpy(), [0, 0])

  def test_checked_graph_replay_and_error_persistence(self):
    """Checked CUDA replay validates before publishing success on its stream."""
    if not wp.get_device().is_cuda:
      self.skipTest("CUDA graph capture requires CUDA")
    _, _, m, d = _fixture(nworld=2)
    mjw.enable_contact_force_params(m, d)
    _set_params(d, 100.0, 0.2)
    mjw.forward(m, d)  # Compile before capture.
    with wp.ScopedCapture() as capture:
      mjw.forward(m, d)
    stream = wp.Stream(d.qvel.device)
    mjw.launch_contact_force_graph(d, capture.graph, stream=stream)
    _set_params(d, -1.0, 0.0)
    with self.assertRaisesRegex(ValueError, "worlds"):
      mjw.launch_contact_force_graph(d, capture.graph, stream=stream)
    errors = d.contact.force_error.numpy().copy()
    _set_params(d, 100.0, 0.2)
    d.nacon.zero_()
    with self.assertRaisesRegex(ValueError, "worlds"):
      mjw.launch_contact_force_graph(d, capture.graph, stream=stream)
    np.testing.assert_array_equal(d.contact.force_error.numpy(), errors)
    mjw.reset_contact_force_params(d)
    mjw.launch_contact_force_graph(d, capture.graph, stream=stream)

  def test_checked_graph_rejects_capture_context(self):
    """A synchronizing checked boundary cannot itself enter a CUDA graph."""
    if not wp.get_device().is_cuda:
      self.skipTest("CUDA graph capture requires CUDA")
    _, _, m, d = _fixture()
    mjw.enable_contact_force_params(m, d)
    mjw.forward(m, d)
    with wp.ScopedCapture() as first:
      mjw.forward(m, d)
    with wp.ScopedCapture():
      with self.assertRaisesRegex(RuntimeError, "capture"):
        mjw.launch_contact_force_graph(d, first.graph)
      with self.assertRaisesRegex(RuntimeError, "capture"):
        mjw.check_contact_force_params(d)


if __name__ == "__main__":
  absltest.main()
