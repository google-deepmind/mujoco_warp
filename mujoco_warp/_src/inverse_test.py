# Copyright 2025 The Newton Developers
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

"""Tests for inverse dynamics."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import DisableBit
from mujoco_warp import IntegratorType
from mujoco_warp import test_data
from mujoco_warp._src import inverse


def _assert_eq(a, b, name):
  tol = 5e-3  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


_XML = """
<mujoco>
  <option timestep=".01" gravity="-1 -1 -1"/>
  <worldbody>
    <body>
      <geom type="sphere" size=".1" pos=".5 0 0"/>
      <joint name="joint1" type="hinge" axis="0 1 0" damping=".1"/>
      <body>
        <geom type="sphere" size=".2" pos="1 0 0"/>
        <joint name="joint2" type="hinge" axis="0 1 0" damping=".2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="joint1"/>
  </actuator>
  <equality>
    <joint joint1="joint1" joint2="joint2"/>
  </equality>
</mujoco>
"""


class InverseTest(parameterized.TestCase):
  @parameterized.product(
    integrator=(IntegratorType.EULER, IntegratorType.IMPLICITFAST),
    jacobian=(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE),
    invdiscrete=(True, False),
  )
  def test_inverse(self, integrator, jacobian, invdiscrete):
    """Tests inverse dynamics."""
    mjm, mjd, m, d = test_data.fixture(
      xml=_XML,
      qvel_noise=0.01,
      ctrl_noise=0.1,
      qfrc_noise=0.1,
      xfrc_noise=0.1,
      overrides={"opt.integrator": integrator, "opt.disableflags": DisableBit.CONTACT, "opt.jacobian": jacobian},
    )

    # discrete qacc
    if invdiscrete:
      mjm.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE

      # save state
      qpos = mjd.qpos.copy()
      qvel = mjd.qvel.copy()

      # call step, save new qvel
      mujoco.mj_step(mjm, mjd)
      qvel_next = mjd.qvel.copy()

      # reset the state, compute discrete-time (finite-differenced) qacc
      mjd.qpos = qpos
      mjd.qvel = qvel
      qacc_fd = (qvel_next - qvel) / mjm.opt.timestep

      # call forward, overwrite qacc with qacc_fd
      mujoco.mj_forward(mjm, mjd)
      mjd.qacc = qacc_fd

      m = mjw.put_model(mjm)
      d = mjw.put_data(mjm, mjd)

    qacc = d.qacc.numpy()[0].copy()
    qfrc_constraint = d.qfrc_constraint.numpy()[0].copy()

    # qfrc_inverse = qfrc_applied + J.T @ xfrc_applied + qfrc_actuator
    qfrc_xfrc_applied = wp.zeros((d.nworld, m.nv), dtype=float)
    mjw.xfrc_accumulate(m, d, qfrc_xfrc_applied)
    qfrc_inverse = d.qfrc_applied.numpy()[0] + d.qfrc_actuator.numpy()[0] + qfrc_xfrc_applied.numpy()[0]

    for arr in (d.qfrc_constraint, d.qfrc_inverse):
      arr.fill_(wp.inf)

    mjw.inverse(m, d)

    _assert_eq(d.qfrc_constraint.numpy()[0], qfrc_constraint, "qfrc_constraint")
    _assert_eq(d.qfrc_inverse.numpy()[0], qfrc_inverse, "qfrc_inverse")
    _assert_eq(d.qacc.numpy()[0], qacc, "qacc")

  def test_qfrc_constraint_inverse_initialization(self):
    """Verify that qfrc_constraint is zeroed in inverse dynamics when nefc==0."""
    xml = """
    <mujoco>
      <option solver="CG" jacobian="sparse"/>
      <worldbody>
        <geom type="plane" size="10 10 .001"/>
        <body name="sphere" pos="0 0 0.04">
          <freejoint/>
          <geom type="sphere" size="0.05" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml)

    # 1. Position in contact, set some qacc (resisting gravity requires contact force)
    d.qpos.zero_()
    qpos_np = d.qpos.numpy()
    qpos_np[0, 2] = 0.04  # contact
    wp.copy(d.qpos, wp.array(qpos_np))
    d.qvel.zero_()
    d.qacc.zero_()

    mjw.inverse(m, d)

    self.assertGreater(d.nefc.numpy()[0], 0, "Should have active constraints")
    self.assertGreater(np.max(np.abs(d.qfrc_constraint.numpy()[0])), 1.0, "Should have non-zero constraint forces")

    # 2. Teleport sphere up (no contacts)
    qpos_np[0, 2] = 1.0
    wp.copy(d.qpos, wp.array(qpos_np))
    d.qvel.zero_()
    d.qacc.zero_()

    mjw.inverse(m, d)

    self.assertEqual(d.nefc.numpy()[0], 0, "Should have no active constraints after teleport")
    qfrc_constraint_post = d.qfrc_constraint.numpy()[0]
    self.assertLess(
      np.max(np.abs(qfrc_constraint_post)),
      1e-5,
      f"qfrc_constraint should be zeroed, but got max abs: {np.max(np.abs(qfrc_constraint_post))}",
    )

  def test_discrete_acc_eulerdamp(self):
    _, _, m, d = test_data.fixture(
      xml=_XML,
      qvel_noise=0.01,
      ctrl_noise=0.1,
      qfrc_noise=0.1,
      xfrc_noise=0.1,
      overrides={"opt.integrator": IntegratorType.EULER, "opt.disableflags": DisableBit.EULERDAMP},
    )
    qacc = wp.zeros((1, m.nv), dtype=float)
    inverse.discrete_acc(m, d, qacc)
    _assert_eq(qacc.numpy()[0], d.qacc.numpy()[0], "qacc")

  def test_discrete_acc_rk4(self):
    _, _, m, d = test_data.fixture(xml=_XML, overrides={"opt.integrator": IntegratorType.RK4})
    qacc = wp.zeros((1, m.nv), dtype=float)

    with self.assertRaises(NotImplementedError):
      inverse.discrete_acc(m, d, qacc)

  def test_inverse_tendon_armature(self):
    """Tests inverse dynamics with tendon armature."""
    _, _, m, d = test_data.fixture(
      "tendon/armature.xml",
      keyframe=0,
      qvel_noise=0.01,
      ctrl_noise=0.1,
      qfrc_noise=0.1,
      xfrc_noise=0.1,
      overrides={"opt.disableflags": DisableBit.GRAVITY | DisableBit.CONSTRAINT},
    )

    qacc = d.qacc.numpy()[0].copy()
    qfrc_constraint = d.qfrc_constraint.numpy()[0].copy()

    # qfrc_inverse = qfrc_applied + J.T @ xfrc_applied + qfrc_actuator
    qfrc_xfrc_applied = wp.zeros((d.nworld, m.nv), dtype=float)
    mjw.xfrc_accumulate(m, d, qfrc_xfrc_applied)
    qfrc_inverse = d.qfrc_applied.numpy()[0] + d.qfrc_actuator.numpy()[0] + qfrc_xfrc_applied.numpy()[0]

    for arr in (d.qfrc_constraint, d.qfrc_inverse):
      arr.fill_(wp.inf)

    mjw.inverse(m, d)

    _assert_eq(d.qfrc_constraint.numpy()[0], qfrc_constraint, "qfrc_constraint")
    _assert_eq(d.qfrc_inverse.numpy()[0], qfrc_inverse, "qfrc_inverse")
    _assert_eq(d.qacc.numpy()[0], qacc, "qacc")

  @parameterized.parameters(1, 2)
  def test_discrete_acc_discrete(self, nworld):
    _, _, m, d = test_data.fixture(xml=_XML, nworld=nworld, overrides={"opt.integrator": IntegratorType.DISCRETE})
    qacc = wp.zeros((nworld, m.nv), dtype=float)
    d.qacc.fill_(1.23)
    if nworld == 2:
      qacc_np = d.qacc.numpy()
      qacc_np[1] = 4.56
      d.qacc.assign(qacc_np)
    inverse.discrete_acc(m, d, qacc)
    for w in range(nworld):
      _assert_eq(qacc.numpy()[w], d.qacc.numpy()[w], f"qacc_{w}")
    if nworld == 2:
      self.assertFalse(np.allclose(qacc.numpy()[0], qacc.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_joint_inverse_consistency(self, nworld):
    """Native discrete inverse dynamics recovers zero applied force for contacting damped joint."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14" iterations="200"/>
        <worldbody>
          <geom type="plane" size="1 1 .1"/>
          <body pos="0 0 .1">
            <joint name="hinge" type="hinge" axis="0 1 0" damping="2" stiffness="50" springref="10"/>
            <geom type="capsule" size=".02" fromto="0 0 0 .5 0 0" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    for _ in range(20):
      d.qacc.fill_(wp.inf)
      mjw.step(m, d)

    qvel = np.full((nworld, 1), 0.1, dtype=np.float32)
    if nworld == 2:
      qvel[1, 0] = -0.15
    d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)

    d.qacc.fill_(wp.inf)
    mjw.forward(m, d)
    self.assertGreater(int(d.nacon.numpy()[0]), 0)

    d.qfrc_inverse.fill_(wp.inf)
    mjw.inverse(m, d)

    qfrc_passive = d.qfrc_passive.numpy()
    qfrc_constraint = d.qfrc_constraint.numpy()
    qfrc_bias = d.qfrc_bias.numpy()
    qfrc_inv = d.qfrc_inverse.numpy()

    for w in range(nworld):
      scale = np.linalg.norm(qfrc_passive[w]) + np.linalg.norm(qfrc_constraint[w]) + np.linalg.norm(qfrc_bias[w])
      self.assertGreater(scale, 0.0)
      rel_err = np.linalg.norm(qfrc_inv[w]) / scale
      self.assertLess(rel_err, 1e-4)

    if nworld == 2:
      self.assertFalse(np.allclose(qfrc_passive[0], qfrc_passive[1]))

  @parameterized.parameters(1, 2)
  def test_discrete_free_joint_inverse_consistency(self, nworld):
    """Forward/inverse consistency for tumbling free joint with stiffness and damping."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005"/>
        <worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body pos="0 0 .5">
            <joint type="free" stiffness="30" damping="2"/>
            <geom type="box" size=".2 .15 .1" mass="2" pos=".02 -.01 .03"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    qvel = np.zeros((nworld, 6), dtype=np.float32)
    qvel[0] = [0.3, 0.0, -0.2, 0.5, -0.3, 8.0]
    if nworld == 2:
      qvel[1] = [-0.2, 0.1, 0.4, -0.4, 0.6, -5.0]
    d.qvel = wp.array(qvel, dtype=float, device=d.qvel.device)

    # In flight, tumbling spin, no constraints.
    for _ in range(20):
      d.qacc.fill_(wp.inf)
      mjw.step(m, d)

    d.qacc.fill_(wp.inf)
    mjw.forward(m, d)
    self.assertEqual(int(d.nacon.numpy()[0]), 0)

    d.qfrc_inverse.fill_(wp.inf)
    mjw.inverse(m, d)
    qfrc_inv = d.qfrc_inverse.numpy()
    qfrc_pass = d.qfrc_passive.numpy()

    for w in range(nworld):
      scale = 1.0 + float(np.linalg.norm(qfrc_pass[w]))
      err = float(np.linalg.norm(qfrc_inv[w])) / scale
      self.assertLess(err, 5e-5)

    if nworld == 2:
      self.assertFalse(np.allclose(qfrc_inv[0], qfrc_inv[1]))


if __name__ == "__main__":
  wp.init()
  absltest.main()
