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

"""Tests for the MJWarp viewer."""

from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest

from mujoco_warp import viewer


class PassiveViewerTest(absltest.TestCase):
  """Tests for the passive MuJoCo viewer loop."""

  @mock.patch.object(viewer.time, "sleep")
  @mock.patch.object(viewer.time, "time", side_effect=[0.0, 0.0])
  @mock.patch.object(viewer.mujoco.viewer, "launch_passive")
  def test_stops_when_viewer_is_not_running(self, launch_passive, unused_time, unused_sleep):
    """The passive loop exits after the viewer window closes."""
    passive_viewer = launch_passive.return_value.__enter__.return_value
    passive_viewer.is_running.side_effect = [True, False]
    model = SimpleNamespace(opt=SimpleNamespace(timestep=0.01))
    step_fn = mock.Mock()

    viewer._run_passive_viewer(model, mock.sentinel.data, step_fn)

    self.assertEqual(passive_viewer.is_running.call_count, 2)
    step_fn.assert_called_once_with(model, mock.sentinel.data)
    passive_viewer.sync.assert_called_once_with()


if __name__ == "__main__":
  absltest.main()
