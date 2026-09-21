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

"""CPU checks for saved-state benchmark diagnostics."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from absl.testing import flagsaver

from mujoco_warp import testspeed
from mujoco_warp._src.types import OverflowType


@pytest.mark.parametrize("behavior", ["error", "continue"])
def test_overflow_never_reads_truncated_jacobian(monkeypatch, capsys, behavior):
  def array(value):
    return SimpleNamespace(numpy=lambda: np.asarray(value))

  jacobian = mock.Mock()
  jacobian.numpy.side_effect = AssertionError("An overflowed Jacobian must not be read")
  data = SimpleNamespace(
    nworld=1,
    nacon=array([1]),
    ncollision=array([1]),
    nefc=array([2]),
    solver_niter=array([1]),
    overflow=array([int(OverflowType.NEFC)]),
    qpos=array([[0]]),
    qvel=array([[0]]),
    qacc=array([[0]]),
    efc=SimpleNamespace(J=jacobian),
  )
  model = SimpleNamespace(is_sparse=True, opt=SimpleNamespace(warn_overflow=int(OverflowType.NEFC), timestep=array([0.001])))
  monkeypatch.setattr(testspeed.wp, "init", lambda: None)
  monkeypatch.setattr(testspeed.wp, "get_device", lambda *args: SimpleNamespace(free_memory=1))
  monkeypatch.setattr(testspeed.cli, "load_model", lambda path: SimpleNamespace(nv=1))
  monkeypatch.setattr(testspeed.cli, "load_state_profile", lambda *args: {})
  monkeypatch.setattr(testspeed.cli, "init_structs", lambda *args: (model, data, None, None))

  def unroll(fn, model, data, rc, callback, *args):
    callback(0, {}, 0.001)
    raise StopIteration("callback completed")

  monkeypatch.setattr(testspeed.cli, "unroll", unroll)
  with flagsaver.as_parsed(
    device="cpu",
    clear_warp_cache="false",
    format="json",
    state_profile="states.npz",
    function="step",
    measure_alloc="true",
    overflow_behavior=behavior,
  ):
    if behavior == "error":
      with pytest.raises(SystemExit) as error:
        testspeed._main(["testspeed", "scene.mjb"])
      assert error.value.code == 1
      assert "NEFC" in capsys.readouterr().out
    else:
      with pytest.raises(StopIteration, match="callback completed"):
        testspeed._main(["testspeed", "scene.mjb"])
  jacobian.numpy.assert_not_called()
