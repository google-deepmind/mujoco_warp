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

"""Tests for shared CLI model loading."""

import tempfile
from contextlib import ExitStack
from contextlib import contextmanager
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from etils import epath

from mujoco_warp._src import cli
from mujoco_warp._src import io


class LoadModelTest(parameterized.TestCase):
  @parameterized.product(extension=("xml", "mjb"), jacobian=("dense", "sparse"))
  def test_overrides_before_sparse_layout_selection(self, extension, jacobian):
    original_jacobian = "sparse" if jacobian == "dense" else "dense"
    xml = f"""
      <mujoco>
        <option jacobian="{original_jacobian}" timestep="0.01"/>
        <worldbody><body><freejoint/><geom type="sphere" size="0.1"/></body></worldbody>
      </mujoco>
    """
    with tempfile.TemporaryDirectory() as directory:
      path = epath.Path(directory) / f"model.{extension}"
      if extension == "mjb":
        mujoco.mj_saveModel(mujoco.MjModel.from_xml_string(xml), path.as_posix())
      else:
        path.write_text(xml)

      with flagsaver.as_parsed(override=[f"opt.jacobian={jacobian}", "opt.timestep=0.002"]):
        model = cli.load_model(path)

    self.assertEqual(model.opt.timestep, 0.002)
    # put_model uses this CPU-side decision when allocating the device model.
    self.assertEqual(io.is_sparse(model), jacobian == "sparse")
    self.assertEqual(bool(mujoco.mj_isSparse(model)), jacobian == "sparse")


class StateProfileTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    self.model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.1"/></body></worldbody>
      <keyframe><key/></keyframe></mujoco>
    """)
    self.states = {
      "qpos": np.zeros((2, 7), np.float32),
      "qvel": np.zeros((2, 6), np.float32),
      "ctrl": np.zeros((2, 0), np.float32),
      "times": np.array([0, 0.01], np.float32),
    }

  def load(self, states):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "states.npz"
      np.savez(path, **states)
      with flagsaver.as_parsed(noise_std="0", noise_rate="0", replay="", keyframe="0", nstep="2"):
        return cli.load_state_profile(str(path), self.model)

  def test_preserves_every_float32_input(self):
    self.states["qpos"][:] = np.arange(14, dtype=np.float32).reshape(2, 7)
    loaded = self.load(self.states)
    for key in self.states:
      self.assertEqual(loaded[key].tobytes(), self.states[key].tobytes())

  @parameterized.parameters("shape", "dtype", "finite", "times", "extra", "empty")
  def test_rejects_invalid_state_inputs(self, defect):
    if defect == "shape":
      self.states["qpos"] = self.states["qpos"][:1]
    elif defect == "dtype":
      self.states["qvel"] = self.states["qvel"].astype(np.float64)
    elif defect == "finite":
      self.states["qvel"][0, 0] = np.nan
    elif defect == "times":
      self.states["times"][1] = 0
    elif defect == "extra":
      self.states["qpos0"] = self.states["qpos"][0]
    else:
      self.states = {key: value[:0] for key, value in self.states.items()}
    with self.assertRaises(ValueError):
      self.load(self.states)

  @parameterized.parameters({"noise_std": 0.01}, {"noise_rate": 0.1}, {"replay": "controls.npz"}, {"keyframe": -1})
  def test_rejects_incompatible_runtime_flags(self, **override):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "states.npz"
      np.savez(path, **self.states)
      settings = dict(noise_std=0.0, noise_rate=0.0, replay="", keyframe=0, nstep=2) | override
      with flagsaver.as_parsed(**{key: str(value) for key, value in settings.items()}), self.assertRaises(ValueError):
        cli.load_state_profile(str(path), self.model)

  def test_rejects_partial_state_prefix(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "states.npz"
      np.savez(path, **self.states)
      with (
        flagsaver.as_parsed(noise_std="0", noise_rate="0", replay="", keyframe="0", nstep="1"),
        self.assertRaises(ValueError),
      ):
        cli.load_state_profile(str(path), self.model)

  @parameterized.parameters("na", "nhistory", "nmocap", "nplugin", "nsensor", "ntendon", "nuserdata")
  def test_rejects_unrecorded_state_fields(self, field):
    values = {name: 0 for name in ("na", "nhistory", "nmocap", "nplugin", "nsensor", "ntendon", "nuserdata")}
    values[field] = 1
    self.model = SimpleNamespace(**values, nkey=1)
    with self.assertRaisesRegex(ValueError, "only qpos/qvel state"):
      self.load(self.states)

  def test_rejects_disabled_warmstart(self):
    self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
    with self.assertRaisesRegex(ValueError, "keeps warmstart enabled"):
      self.load(self.states)

  def test_rejects_sleeping(self):
    self.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    with self.assertRaisesRegex(ValueError, "sleeping disabled"):
      self.load(self.states)

  def test_rejects_deterministic_mode(self):
    with (
      mock.patch.object(wp.config, "deterministic", wp.DeterministicMode.RUN_TO_RUN),
      self.assertRaisesRegex(ValueError, "deterministic mode off"),
    ):
      self.load(self.states)

  def test_cpu_restorer_broadcasts_selected_state_and_time(self):
    states = [
      np.arange(6, dtype=np.float32).reshape(2, 3),
      np.arange(4, dtype=np.float32).reshape(2, 2),
      np.array([[7], [11]], np.float32),
      np.array([0.2, 0.4], np.float32),
    ]
    with wp.ScopedDevice("cpu"):
      inputs = [wp.array(value, dtype=float) for value in states]
      index = wp.array([1], dtype=int)
      outputs = [wp.zeros((3, value.shape[1]), dtype=float) for value in states[:3]] + [wp.zeros(3, dtype=float)]
      wp.launch(cli._restore_state, 3, inputs=inputs + [index] + outputs[3:] + outputs[:3])
      for source, output in zip(states, outputs):
        np.testing.assert_array_equal(output.numpy(), np.broadcast_to(source[1], output.shape))

  def test_keyframe_reset_clears_transient_state_before_restoration(self):
    with wp.ScopedDevice("cpu"):
      model = cli.mjw.put_model(self.model)
      data = cli.mjw.put_data(self.model, mujoco.MjData(self.model), nworld=2)
      data.qacc_warmstart.fill_(3)
      data.qfrc_applied.fill_(5)
      data.xfrc_applied.fill_(7)
      cli.mjw.reset_data_keyframe(model, data, 0)
      arrays = [wp.array(self.states[key], dtype=float) for key in ("qpos", "qvel", "ctrl", "times")]
      wp.launch(
        cli._restore_state,
        data.nworld,
        inputs=arrays + [wp.array([1], dtype=int), data.time, data.qpos, data.qvel, data.ctrl],
      )
      for field in (data.qacc_warmstart, data.qfrc_applied, data.xfrc_applied):
        np.testing.assert_array_equal(field.numpy(), 0)
      np.testing.assert_array_equal(data.time.numpy(), self.states["times"][1])

  def test_reset_graph_excluded_from_events_and_each_measured_launch(self):
    captures, events, active = [], [], []
    traced = False

    def operation(name):
      if active:
        active[-1]["operations"].append(name)

    @contextmanager
    def capture():
      graph = {"traced": traced, "operations": []}
      captures.append(graph)
      active.append(graph)
      yield SimpleNamespace(graph=graph)
      active.pop()

    @contextmanager
    def tracer(**kwargs):
      nonlocal traced
      traced = True
      yield SimpleNamespace(trace=lambda: {})
      traced = False

    def clock():
      events.append("clock")
      return float(len(events))

    index = mock.Mock()
    data = SimpleNamespace(
      nworld=2,
      qpos=None,
      qvel=None,
      ctrl=None,
      time=None,
    )

    def step(*args):
      operation("step")

    patches = {
      "get_device": lambda *args: "cpu",
      "get_stream": lambda: None,
      "ScopedDevice": lambda *args: nullcontext(),
      "ScopedStream": lambda *args: nullcontext(),
      "array": lambda value, **kwargs: value,
      "zeros": lambda *args, **kwargs: index,
      "launch": lambda *args, **kwargs: operation("restore"),
      "ScopedCapture": capture,
      "capture_launch": lambda graph: events.append(tuple(graph["operations"])),
      "synchronize": lambda: events.append("sync"),
    }
    with ExitStack() as stack:
      for name, value in patches.items():
        stack.enter_context(mock.patch.object(cli.wp, name, value))
      stack.enter_context(mock.patch.object(cli.warp_util, "EventTracer", tracer))
      stack.enter_context(mock.patch.object(cli.mjw, "step", step))
      stack.enter_context(mock.patch.object(cli.mjw, "reset_data_keyframe", lambda *args: operation("reset")))
      stack.enter_context(mock.patch.object(cli.time, "perf_counter", clock))
      stack.enter_context(flagsaver.as_parsed(device="cpu", nstep="2", event_trace="true"))
      callback = mock.Mock()
      cli.unroll(step, object(), data, None, callback, states=self.states)
    assert captures == [
      {"traced": False, "operations": ["reset", "restore"]},
      {"traced": True, "operations": ["step"]},
    ]
    assert [call.args[0].tolist() for call in index.assign.call_args_list] == [[0], [1]]
    assert [call.args[0] for call in callback.call_args_list] == [0, 1]
    # The final two intervals each synchronize restoration before starting the measured launch.
    assert events[-12:] == [("reset", "restore"), "sync", "clock", ("step",), "sync", "clock"] * 2


if __name__ == "__main__":
  absltest.main()
