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

"""Tests for explicit force-space parameters on externally supplied rigid contacts."""

import itertools

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp._src import solver


def _fixture(cone="elliptic", condim=1, nworld=1):
  mjm = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep=".001" cone="{cone}" impratio="2"/>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 .1"/>
        <body pos=".1 0 .09">
          <freejoint/>
          <geom name="ball" type="sphere" size=".1" mass="1"/>
        </body>
      </worldbody>
      <contact>
        <pair geom1="floor" geom2="ball" condim="{condim}" friction=".7 .3 .02 .004 .006"/>
      </contact>
    </mujoco>
  """)
  mjd = mujoco.MjData(mjm)
  mjd.qvel[:3] = [0.1, -0.2, -0.03]
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=8, njmax=32)
  m.opt.run_collision_detection = False
  return mjm, mjd, m, d


def _set_params(d, stiffness, damping):
  values = np.zeros((d.naconmax, 2), dtype=np.float32)
  values[: int(d.nacon.numpy()[0])] = [stiffness, damping]
  d.contact.force_params.assign(values)


class ContactForceTest(parameterized.TestCase):
  def test_native_factories_leave_override_disabled(self):
    mjm, _, m, copied = _fixture(nworld=2)
    for d in (copied, mjw.make_data(mjm, nworld=2, nconmax=8, njmax=32)):
      self.assertEqual(d.contact.force_params.size, 0)
      self.assertEqual(d.contact.force_error.size, 0)
      self.assertEqual(d.contact.force_active.size, 0)
      mjw.reset_data(m, d)
      self.assertEqual(d.contact.force_params.size, 0)
      self.assertEqual(d.contact.force_error.size, 0)
      self.assertEqual(d.contact.force_active.size, 0)

  @parameterized.parameters("elliptic", "pyramidal")
  def test_disabled_rows_preserve_legacy_outputs(self, cone):
    _, _, m, d = _fixture(cone=cone, condim=6)
    mjw.make_constraint(m, d)
    count = int(d.nefc.numpy()[0])
    before = [getattr(d.efc, name).numpy()[0, :count].copy() for name in ("D", "aref", "vel")]
    mjw.enable_contact_force_params(m, d)
    self.assertEqual(d.contact.force_params.shape, (d.naconmax,))
    self.assertEqual(d.contact.force_error.shape, (d.nworld,))
    self.assertEqual(d.contact.force_active.shape, (d.nworld,))
    for name in ("D", "aref", "vel"):
      getattr(d.efc, name).fill_(wp.nan)
    mjw.make_constraint(m, d)
    for name, expected in zip(("D", "aref", "vel"), before, strict=True):
      np.testing.assert_array_equal(getattr(d.efc, name).numpy()[0, :count], expected)
    mjw.check_contact_force_params(d)

  @parameterized.parameters(*itertools.product(("elliptic", "pyramidal"), (1, 3, 4, 6), ((1.0, 0.0), (1e12, 20.0))))
  def test_physical_rows_beyond_legacy_impedance_bounds(self, cone, condim, coefficients):
    _, _, m, d = _fixture(cone, condim)
    mjw.enable_contact_force_params(m, d)
    stiffness, damping = coefficients
    _set_params(d, stiffness, damping)
    d.efc.D.fill_(wp.nan)
    d.efc.aref.fill_(wp.nan)
    mjw.make_constraint(m, d)
    mjw.check_contact_force_params(d)

    h = 0.001
    compliance_inverse = h * (damping + h * stiffness)
    gap = float(d.contact.dist.numpy()[0] - d.contact.includemargin.numpy()[0])
    address = d.contact.efc_address.numpy()[0]
    address = address[address >= 0]
    jacobian = d.efc.J.numpy()[0, address, : m.nv]
    velocity = jacobian.astype(np.float64) @ d.qvel.numpy()[0].astype(np.float64)
    expected_aref = -velocity / h
    if cone == "pyramidal" and condim > 1:
      expected_d = np.full(len(address), compliance_inverse / (2 * (condim - 1)))
      expected_aref -= stiffness * gap / compliance_inverse
    else:
      expected_d = np.full(len(address), compliance_inverse)
      expected_aref[0] -= stiffness * gap / compliance_inverse
      friction = np.array([0.7, 0.3, 0.02, 0.004, 0.006])
      for row in range(1, len(address)):
        scale = 0.5 * (friction[0] / friction[row - 1]) ** 2
        expected_d[row] /= scale
    np.testing.assert_allclose(d.efc.D.numpy()[0, address], expected_d, rtol=2e-6, atol=0)
    np.testing.assert_allclose(d.efc.aref.numpy()[0, address], expected_aref, rtol=2e-6, atol=2e-3)

  @parameterized.parameters((-1.0, 0.0), (0.0, 1.0), (1.0, -1.0), (np.nan, 0.0), (1.0, np.inf))
  def test_invalid_coefficients_fail_closed_and_remain_sticky(self, stiffness, damping):
    _, _, m, d = _fixture()
    mjw.enable_contact_force_params(m, d)
    _set_params(d, stiffness, damping)
    mjw.make_constraint(m, d)
    self.assertNotEqual(int(d.contact.force_error.numpy()[0]) & 1, 0)
    np.testing.assert_array_equal(d.contact.force_active.numpy(), 0)
    address = int(d.contact.efc_address.numpy()[0, 0])
    self.assertEqual(float(d.efc.D.numpy()[0, address]), 0.0)
    self.assertEqual(float(d.efc.aref.numpy()[0, address]), 0.0)
    with self.assertRaises((ValueError, RuntimeError)):
      mjw.check_contact_force_params(d)
    _set_params(d, 1.0, 0.0)
    mjw.make_constraint(m, d)
    self.assertNotEqual(int(d.contact.force_error.numpy()[0]), 0)
    with self.assertRaises((ValueError, RuntimeError)):
      mjw.check_contact_force_params(d)

  @parameterized.parameters(1e-26, 1e30)
  def test_unrepresentable_metric_is_reported(self, stiffness):
    _, _, m, d = _fixture()
    mjw.enable_contact_force_params(m, d)
    _set_params(d, stiffness, 0.0)
    mjw.make_constraint(m, d)
    self.assertNotEqual(int(d.contact.force_error.numpy()[0]) & 2, 0)
    with self.assertRaises((ValueError, RuntimeError)):
      mjw.check_contact_force_params(d)

  def test_nonfinite_reference_acceleration_is_reported(self):
    _, _, m, d = _fixture()
    mjw.enable_contact_force_params(m, d)
    _set_params(d, 1e4, 0.0)
    d.contact.dist.fill_(-1e35)
    mjw.make_constraint(m, d)
    self.assertNotEqual(int(d.contact.force_error.numpy()[0]) & 4, 0)
    with self.assertRaises((ValueError, RuntimeError)):
      mjw.check_contact_force_params(d)

  def test_reset_preserves_other_world_and_full_reset_clears_tail(self):
    _, _, m, d = _fixture(nworld=2)
    mjw.enable_contact_force_params(m, d)
    values = np.tile(np.array([7.0, 3.0], dtype=np.float32), (d.naconmax, 1))
    d.contact.force_params.assign(values)
    d.contact.force_error.assign(np.array([1, 2], dtype=np.int32))
    worlds = d.contact.worldid.numpy().copy()
    count = int(d.nacon.numpy()[0])
    self.assertEqual(set(worlds[:count]), {0, 1})
    mjw.reset_data(m, d, reset=wp.array([True, False], dtype=bool))
    actual = d.contact.force_params.numpy()
    np.testing.assert_array_equal(actual[:count][worlds[:count] == 0], 0.0)
    np.testing.assert_array_equal(actual[:count][worlds[:count] == 1], values[:count][worlds[:count] == 1])
    np.testing.assert_array_equal(d.contact.force_error.numpy(), [0, 2])
    mjw.reset_data(m, d)
    np.testing.assert_array_equal(d.contact.force_params.numpy(), 0.0)
    np.testing.assert_array_equal(d.contact.force_error.numpy(), 0)
    mjw.check_contact_force_params(d)

  def test_unsupported_model_domains_are_rejected(self):
    for field, value in (("run_collision_detection", True), ("integrator", mjw.IntegratorType.RK4)):
      with self.subTest(field=field):
        _, _, m, d = _fixture()
        setattr(m.opt, field, value)
        with self.assertRaises((ValueError, NotImplementedError)):
          mjw.enable_contact_force_params(m, d)
    _, _, m, d = _fixture()
    m.flg_adhesion = True
    with self.assertRaises((ValueError, NotImplementedError)):
      mjw.enable_contact_force_params(m, d)

  def test_native_collision_cannot_reuse_external_coefficients(self):
    _, _, m, d = _fixture()
    mjw.enable_contact_force_params(m, d)
    _set_params(d, 1e4, 20.0)
    with self.assertRaises(ValueError):
      mjw.collision(m, d)

  def test_discrete_is_rejected_before_enablement_and_after_model_mutation(self):
    """DISCRETE regularization cannot silently change a physical contact metric."""
    _, _, m, d = _fixture()
    m.opt.integrator = mjw.IntegratorType.DISCRETE
    with self.assertRaisesRegex(ValueError, "require Euler, implicit, or implicitfast"):
      mjw.enable_contact_force_params(m, d)
    self.assertEqual(d.contact.force_params.size, 0)
    self.assertEqual(d.contact.force_error.size, 0)

    m.opt.integrator = mjw.IntegratorType.EULER
    mjw.enable_contact_force_params(m, d)
    _set_params(d, 100.0, 0.2)
    d.efc.D.fill_(wp.inf)
    m.opt.integrator = mjw.IntegratorType.DISCRETE
    with self.assertRaisesRegex(ValueError, "require Euler, implicit, or implicitfast"):
      mjw.make_constraint(m, d)
    self.assertTrue(np.isinf(d.efc.D.numpy()).all())

  def test_empty_capacity_is_rejected_without_enabling_buffers(self):
    mjm, _, m, _ = _fixture()
    d = mjw.make_data(mjm, nconmax=0, njmax=0)
    with self.assertRaises(ValueError):
      mjw.enable_contact_force_params(m, d)
    self.assertEqual(d.contact.force_params.size, 0)
    self.assertEqual(d.contact.force_error.size, 0)

  def test_malformed_buffers_are_rejected_before_launch(self):
    _, _, m, d = _fixture()
    mjw.enable_contact_force_params(m, d)
    d.contact.force_params = wp.zeros(d.naconmax - 1, dtype=wp.vec2)
    with self.assertRaises(ValueError):
      mjw.make_constraint(m, d)
    d.contact.force_params = wp.zeros(d.naconmax, dtype=wp.vec2)
    d.contact.force_error = wp.empty(0, dtype=int)
    with self.assertRaises(ValueError):
      mjw.make_constraint(m, d)

  def test_malformed_activity_buffers_are_rejected_before_launch(self):
    _, _, m, d = _fixture()
    mjw.enable_contact_force_params(m, d)
    for label, active in (
      ("missing", None),
      ("empty", wp.empty(0, dtype=int)),
      ("wrong_dtype", wp.zeros(d.nworld, dtype=float)),
    ):
      with self.subTest(case=label):
        d.contact.force_active = active
        with self.assertRaisesRegex(ValueError, "force_active"):
          mjw.make_constraint(m, d)

  def test_activity_is_world_local_and_rebuilt_after_physical_to_legacy(self):
    """Ordinary worlds retain exact legacy sums across physical-row transitions."""
    _, _, legacy_m, legacy_d = _fixture(nworld=2)
    mjw.forward(legacy_m, legacy_d)
    reference = [legacy_d.qacc.numpy(), legacy_d.qfrc_constraint.numpy()]

    _, _, m, d = _fixture(nworld=2)
    mjw.enable_contact_force_params(m, d)
    params = np.zeros((d.naconmax, 2), dtype=np.float32)
    count = int(d.nacon.numpy()[0])
    worldid = d.contact.worldid.numpy()[:count]
    params[:count][worldid == 0] = [100.0, 0.2]
    d.contact.force_params.assign(params)
    mjw.forward(m, d)
    np.testing.assert_array_equal(d.contact.force_active.numpy(), [1, 0])
    for actual, expected in zip((d.qacc.numpy(), d.qfrc_constraint.numpy()), reference, strict=True):
      np.testing.assert_array_equal(actual[1], expected[1])
      self.assertFalse(np.array_equal(actual[0], expected[0]))

    _set_params(d, 0.0, 0.0)
    mjw.forward(m, d)
    np.testing.assert_array_equal(d.contact.force_active.numpy(), [0, 0])
    for actual, expected in zip((d.qacc.numpy(), d.qfrc_constraint.numpy()), reference, strict=True):
      np.testing.assert_array_equal(actual, expected)

    _set_params(d, 100.0, 0.2)
    mjw.make_constraint(m, d)
    np.testing.assert_array_equal(d.contact.force_active.numpy(), [1, 1])
    d.nacon.zero_()
    mjw.make_constraint(m, d)
    np.testing.assert_array_equal(d.contact.force_active.numpy(), [0, 0])

  def test_sparse_physical_mode_is_explicitly_rejected(self):
    mjm, mjd, _, _ = _fixture()
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)
    m.opt.run_collision_detection = False
    with self.assertRaisesRegex(ValueError, "dense"):
      mjw.enable_contact_force_params(m, d)

  def test_compensated_wrench_handles_unequal_weights_and_signed_lever_arms(self):
    count = 6400
    indices = np.arange(count)
    forces = ((1 + indices % 4) * 1e-8).astype(np.float32)
    forces[0] = 1.0
    jacobian = np.column_stack(
      (np.ones(count), np.where(indices % 3, -0.125, 0.125), np.where(indices % 5, 0.0625, -0.0625))
    ).astype(np.float32)
    for order in (indices, indices[::-1], np.random.default_rng(4561).permutation(count)):
      # Dyadic Jacobian entries make products exact; the oracle isolates summation error.
      j = np.stack((jacobian[order], jacobian[order] * np.array([-1, 2, -4], dtype=np.float32)))
      force = np.stack((forces[order], forces[order] * np.float32(0.5)))
      expected = np.einsum("wij,wi->wj", j.astype(np.float64), force.astype(np.float64))
      inputs = [
        wp.array([count, count], dtype=int),
        wp.array(j, dtype=float),
        wp.array(force, dtype=float),
        count,
        wp.ones(2, dtype=int),
        wp.array([1, 1], dtype=int),
        wp.array([False, False], dtype=bool),
      ]
      actual = wp.zeros((2, 3), dtype=float)
      wp.launch(solver._update_constraint_init_qfrc_constraint_dense(False, True), (2, 3), inputs, [actual])
      np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-7, atol=2e-8)
      if np.array_equal(order, indices):
        legacy = wp.zeros_like(actual)
        default = wp.zeros_like(actual)
        wp.launch(solver._update_constraint_init_qfrc_constraint_dense(False, False), (2, 3), inputs, [legacy])
        wp.launch(solver._update_constraint_init_qfrc_constraint_dense(False), (2, 3), inputs, [default])
        np.testing.assert_array_equal(default.numpy(), legacy.numpy())
        self.assertGreater(np.max(np.abs(legacy.numpy() - expected)), 5e-5)

  def test_compensated_wrench_retains_counts_and_fast_path_guards(self):
    inputs = [
      wp.array([4, 2, 0, 4], dtype=int),
      wp.ones((4, 4, 2), dtype=float),
      wp.array(np.tile([1.0, 0.1, 0.2, 0.3], (4, 1)), dtype=float),
      3,
      wp.ones(4, dtype=int),
      wp.array([1, 0, 1, 1], dtype=int),
      wp.array([False, False, False, True], dtype=bool),
    ]
    for fast in (False, True):
      actual = wp.full((4, 2), 7.0, dtype=float)
      wp.launch(solver._update_constraint_init_qfrc_constraint_dense(fast, True), (4, 2), inputs, [actual])
      expected = np.repeat([[1.3], [7.0 if fast else 1.1], [0.0], [7.0]], 2, axis=1)
      np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-7, atol=0)

  def test_compensated_wrench_activity_selects_legacy_per_world(self):
    """An ordinary world retains exact serial sums next to a physical world."""
    count = 6400
    forces = np.full((2, count), 1.0e-8, dtype=np.float32)
    forces[:, 0] = 1.0
    active = wp.array([0, 1], dtype=int)
    inputs = [
      wp.full(2, count, dtype=int),
      wp.ones((2, count, 1), dtype=float),
      wp.array(forces, dtype=float),
      count,
      active,
      wp.ones(2, dtype=int),
      wp.zeros(2, dtype=bool),
    ]
    legacy = wp.full((2, 1), wp.inf, dtype=float)
    actual = wp.full((2, 1), wp.inf, dtype=float)
    wp.launch(solver._update_constraint_init_qfrc_constraint_dense(False, False), (2, 1), inputs, [legacy])
    wp.launch(solver._update_constraint_init_qfrc_constraint_dense(False, True), (2, 1), inputs, [actual])
    np.testing.assert_array_equal(actual.numpy()[0], legacy.numpy()[0])
    self.assertNotEqual(float(actual.numpy()[1, 0]), float(legacy.numpy()[1, 0]))
    np.testing.assert_allclose(actual.numpy()[1, 0], forces[1].astype(np.float64).sum(), rtol=2e-7, atol=0)
    active.zero_()
    wp.launch(solver._update_constraint_init_qfrc_constraint_dense(False, True), (2, 1), inputs, [actual])
    np.testing.assert_array_equal(actual.numpy(), legacy.numpy())


if __name__ == "__main__":
  wp.init()
  absltest.main()
