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

"""Numerical regressions for opt-in deterministic arithmetic."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import DeterminismType
from mujoco_warp import test_data
from mujoco_warp._src import smooth
from mujoco_warp._src import types


class DeterministicArithmeticTest(parameterized.TestCase):
  @parameterized.product(nv=(2, 5), captured=(False, True), deterministic=(False, True))
  def test_sparse_substitution_dependencies(self, nv, captured, deterministic):
    """Dependent substitution stages must see the preceding level's writes."""
    if captured and not wp.get_device().is_cuda:
      self.skipTest("CUDA graph required")
    if nv == 2:
      updates, offsets = [(0, 1, 0)], [0, 1]
      coefficients, diagonal = [0.5], [1.0, 1.0]
    else:
      updates = [(0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3), (1, 3, 4), (2, 4, 5)]
      offsets = [0, 4, 6]
      coefficients, diagonal = [0.2, 0.3, 0.1, -0.2, 0.4, 0.5], [1.0, 0.5, 0.25, 2.0, 1.5]
    lower = np.eye(nv)
    for i, k, adr in updates:
      lower[k, i] = coefficients[adr]
    matrix = lower.T @ np.diag(1 / np.array(diagonal)) @ lower
    rhs = np.arange(2, nv + 2, dtype=np.float32)
    expected = np.linalg.solve(matrix, rhs)
    worlds = 9
    result = wp.zeros((worlds, nv), dtype=float)
    block_dim = 128 if wp.get_device().is_cuda else 1
    inputs = [
      wp.array([types.Q_LD_BLOCK_SPARSE] * nv, dtype=int),
      wp.array(np.tile(coefficients, (worlds, 1)), dtype=float),
      wp.array(np.tile(diagonal, (worlds, 1)), dtype=float),
      wp.array(updates, dtype=wp.vec3i),
      wp.array(offsets, dtype=int),
      wp.array(np.tile(rhs, (worlds, 1)), dtype=float),
    ]
    kernel = smooth._solve_LD_sparse_fused(nv, len(offsets) - 1, deterministic)

    def solve():
      wp.launch(kernel, dim=(worlds, block_dim), inputs=inputs, outputs=[result], block_dim=block_dim)

    solve()
    if captured:
      with wp.ScopedCapture() as capture:
        solve()
      for _ in range(3):
        wp.capture_launch(capture.graph)
    np.testing.assert_allclose(result.numpy(), np.tile(expected, (worlds, 1)), rtol=1e-6, atol=1e-6)

  @parameterized.parameters(False, True)
  def test_ball_actuator_records(self, captured):
    """Each of three moment entries must survive eager and graph execution."""
    if captured and not wp.get_device().is_cuda:
      self.skipTest("CUDA graph required")
    xml = """<mujoco><worldbody><body>
      <joint name="ball" type="ball"/><geom type="sphere" size=".1"/>
      </body></worldbody><actuator><motor joint="ball" gear="1 1 1 0 0 0"/>
      </actuator></mujoco>"""
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=1024)
    mjd.ctrl[:] = 1
    mujoco.mj_forward(mjm, mjd)
    d.ctrl.fill_(1)
    mjw.fwd_actuation(m, d)  # Warm ordinary kernels before optional graph capture.
    m.opt.deterministic = DeterminismType.ATOMICS
    if captured:
      with wp.ScopedCapture() as capture:
        mjw.fwd_actuation(m, d)
      for _ in range(3):
        wp.capture_launch(capture.graph)
    else:
      mjw.fwd_actuation(m, d)
    expected = np.tile(mjd.qfrc_actuator, (d.nworld, 1))
    np.testing.assert_array_equal(d.qfrc_actuator.numpy(), expected)

  @parameterized.parameters(False, True)
  def test_public_sparse_factor_solve(self, captured):
    """Exercise deterministic factor_m/solve_m above the dense block threshold."""
    if captured and not wp.get_device().is_cuda:
      self.skipTest("CUDA graph required")
    n = 66
    chain = "".join('<body pos="0 0 .05"><joint axis="0 1 0" armature="1"/><geom type="sphere" size=".02"/>' for _ in range(n))
    xml = "<mujoco><worldbody>" + chain + "</body>" * n + "</worldbody></mujoco>"
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=9)
    m.opt.deterministic = DeterminismType.ATOMICS
    self.assertGreater(len(m.qLD_updates), 0)
    rhs = np.linspace(0.1, 1.0, n)
    reference = np.zeros((1, n))
    mujoco.mj_solveM(mjm, mjd, reference, rhs[None])
    vector = wp.array(np.tile(rhs, (d.nworld, 1)), dtype=float)
    result = wp.zeros((d.nworld, n), dtype=float)

    def factor_solve():
      mjw.factor_m(m, d)
      mjw.solve_m(m, d, result, vector)

    factor_solve()
    if captured:
      with wp.ScopedCapture() as capture:
        factor_solve()
      for _ in range(3):
        wp.capture_launch(capture.graph)
    np.testing.assert_allclose(result.numpy(), np.tile(reference, (d.nworld, 1)), rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
