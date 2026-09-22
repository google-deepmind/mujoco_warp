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

import mujoco
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


if __name__ == "__main__":
  absltest.main()
