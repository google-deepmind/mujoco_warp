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
from mujoco_warp._src import derivative
from mujoco_warp._src import smooth
from mujoco_warp._src import types


class DeterministicArithmeticTest(parameterized.TestCase):
  """Check deterministic arithmetic against reference results in eager and graph execution."""

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
      """Launch the sparse substitution kernel into the reusable result buffer."""
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

  @parameterized.product(captured=(False, True), mixed=(False, True))
  def test_public_sparse_factor_solve(self, captured, mixed):
    """Exercise deterministic factor_m/solve_m above the dense block threshold."""
    if captured and not wp.get_device().is_cuda:
      self.skipTest("CUDA graph required")
    n = 66
    chain = "".join('<body pos="0 0 .05"><joint axis="0 1 0" armature="1"/><geom type="sphere" size=".02"/>' for _ in range(n))
    bodies = chain + "</body>" * n
    if mixed:
      small = "".join('<body pos="0 0 .05"><joint armature="1"/><geom size=".02"/>' for _ in range(7))
      bodies += '<body pos="1 0 0">' + small + "</body>" * 8
      bodies += '<body pos="2 0 0"><freejoint/><geom size=".1"/></body>'
    xml = "<mujoco><worldbody>" + bodies + "</worldbody></mujoco>"
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=9)
    m.opt.deterministic = DeterminismType.ATOMICS
    self.assertGreater(len(m.qLD_updates), 0)
    if mixed:
      self.assertGreater(m.qLD_block_total, 0)
    rhs = np.linspace(0.1, 1.0, m.nv)
    reference = np.zeros((1, m.nv))
    mujoco.mj_solveM(mjm, mjd, reference, rhs[None])
    vector = wp.array(np.tile(rhs, (d.nworld, 1)), dtype=float)
    result = wp.zeros((d.nworld, m.nv), dtype=float)

    def factor_solve():
      """Factor the mass matrix and solve the same right-hand side."""
      mjw.factor_m(m, d)
      mjw.solve_m(m, d, result, vector)

    factor_solve()
    if captured:
      with wp.ScopedCapture() as capture:
        factor_solve()
      for _ in range(3):
        wp.capture_launch(capture.graph)
    np.testing.assert_allclose(result.numpy(), np.tile(reference, (d.nworld, 1)), rtol=1e-4, atol=1e-5)

  @parameterized.product(nworld=(1, 2), captured=(False, True))
  def test_refactored_flex_arithmetic(self, nworld, captured):
    """Main's flex Hessian and scatter paths retain numerical parity with ATOMICS."""
    if captured and not wp.get_device().is_cuda:
      self.skipTest("CUDA graph required")
    mjm, mjd, m, d = test_data.fixture(
      xml="""<mujoco>
        <option integrator="discrete" gravity="0 0 0" timestep=".005"/>
        <worldbody>
          <flexcomp name="box" type="grid" count="2 2 2" spacing=".1 .1 .1" dim="3" mass="1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="1000" poisson=".3" damping=".03"/>
          </flexcomp>
        </worldbody>
      </mujoco>""",
      nworld=nworld,
    )
    qpos = np.tile(mjd.qpos, (nworld, 1)).astype(np.float32)
    qvel = np.tile(np.linspace(-0.03, 0.04, mjm.nv, dtype=np.float32), (nworld, 1))
    qpos[0, 0] += 0.01
    if nworld == 2:
      qpos[1, 0] -= 0.02
      qvel[1] *= -1.5
    d.qpos.assign(qpos)
    d.qvel.assign(qvel)

    def evaluate():
      """Recompute flex forces and the implicit force shift from fixed state."""
      d.qfrc_spring.fill_(wp.inf)
      d.qfrc_damper.fill_(wp.inf)
      d.efm_c.fill_(wp.inf)
      d.flex_hessian_valid.zero_()
      mjw.fwd_position(m, d)
      mjw.fwd_velocity(m, d)
      mjw.passive(m, d)
      derivative.eff_shift(m, d)

    m.opt.deterministic = DeterminismType.NONE
    evaluate()
    fields = (d.qfrc_spring, d.qfrc_damper, d.efm_c, d.flexvert_hessian, d.flexedge_hessian)
    reference = [a.numpy().copy() for a in fields]
    m.opt.deterministic = DeterminismType.ATOMICS
    evaluate()
    if captured:
      with wp.ScopedCapture() as capture:
        evaluate()
      for _ in range(3):
        wp.capture_launch(capture.graph)
    for a, expected in zip(fields, reference):
      np.testing.assert_allclose(a.numpy(), expected, rtol=1e-5, atol=1e-5)
    for world in range(nworld):
      mjd.qpos[:] = qpos[world]
      mjd.qvel[:] = qvel[world]
      mujoco.mj_forward(mjm, mjd)
      np.testing.assert_allclose(d.qfrc_spring.numpy()[world], mjd.qfrc_spring, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(d.qfrc_damper.numpy()[world], mjd.qfrc_damper, rtol=1e-5, atol=1e-5)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qfrc_spring.numpy()[0], d.qfrc_spring.numpy()[1]))


if __name__ == "__main__":
  absltest.main()
