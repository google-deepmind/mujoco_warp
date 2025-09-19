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

"""Tests for forward dynamics functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import BiasType
from mujoco_warp import DisableBit
from mujoco_warp import EnableBit
from mujoco_warp import GainType
from mujoco_warp import IntegratorType
from mujoco_warp import test_data

# tolerance for difference between MuJoCo and mjw smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class ForwardTest(parameterized.TestCase):
  # TODO(team): test sparse when actuator_moment and/or ten_J have sparse representation
  @parameterized.product(xml=["humanoid/humanoid.xml", "pendula.xml"])
  def test_fwd_velocity(self, xml):
    _, mjd, m, d = test_data.fixture(xml, qvel_noise=0.01, ctrl_noise=0.1)

    for arr in (d.actuator_velocity, d.qfrc_bias):
      arr.zero_()

    mjw.fwd_velocity(m, d)

    _assert_eq(d.actuator_velocity.numpy()[0], mjd.actuator_velocity, "actuator_velocity")
    _assert_eq(d.qfrc_bias.numpy()[0], mjd.qfrc_bias, "qfrc_bias")

  def test_fwd_velocity_tendon(self):
    _, mjd, m, d = test_data.fixture("tendon/fixed.xml")

    d.ten_velocity.zero_()
    mjw.fwd_velocity(m, d)

    _assert_eq(d.ten_velocity.numpy()[0], mjd.ten_velocity, "ten_velocity")

  @parameterized.product(
    xml=("actuation/actuation.xml", "actuation/actuators.xml", "actuation/muscle.xml"),
    disableflags=(0, DisableBit.ACTUATION),
  )
  def test_actuation(self, xml, disableflags):
    mjm, mjd, m, d = test_data.fixture(xml, keyframe=0, overrides={"opt.disableflags": disableflags})

    for arr in (d.qfrc_actuator, d.actuator_force, d.act_dot):
      arr.zero_()

    mjw.fwd_actuation(m, d)

    _assert_eq(d.qfrc_actuator.numpy()[0], mjd.qfrc_actuator, "qfrc_actuator")
    _assert_eq(d.actuator_force.numpy()[0], mjd.actuator_force, "actuator_force")

    if mjm.na:
      _assert_eq(d.act_dot.numpy()[0], mjd.act_dot, "act_dot")

      # next activations
      mujoco.mj_step(mjm, mjd)
      mjw.step(m, d)

      _assert_eq(d.act.numpy()[0], mjd.act, "act")

    # TODO(team): test actearly

  @parameterized.parameters(0, DisableBit.CLAMPCTRL)
  def test_clampctrl(self, disableflags):
    _, mjd, _, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint" type="slide"/>
          <geom type="sphere" size=".1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="joint" ctrlrange="-1 1"/>
      </actuator>
      <keyframe>
        <key ctrl="2"/>
      </keyframe>
    </mujoco>
    """,
      keyframe=0,
      overrides={"opt.disableflags": disableflags},
    )

    _assert_eq(d.ctrl.numpy()[0], mjd.ctrl, "ctrl")

  def test_fwd_acceleration(self):
    _, mjd, m, d = test_data.fixture("humanoid/humanoid.xml", qvel_noise=0.01, ctrl_noise=0.1)

    for arr in (d.qfrc_smooth, d.qacc_smooth):
      arr.zero_()

    mjw.fwd_acceleration(m, d)

    _assert_eq(d.qfrc_smooth.numpy()[0], mjd.qfrc_smooth, "qfrc_smooth")
    _assert_eq(d.qacc_smooth.numpy()[0], mjd.qacc_smooth, "qacc_smooth")

  @parameterized.product(
    jacobian=(mujoco.mjtJacobian.mjJAC_AUTO, mujoco.mjtJacobian.mjJAC_DENSE), disableflags=(0, DisableBit.EULERDAMP)
  )
  def test_euler(self, jacobian, disableflags):
    mjm, mjd, _, _ = test_data.fixture(
      "pendula.xml",
      qvel_noise=0.01,
      ctrl_noise=0.1,
      overrides={"opt.jacobian": jacobian, "opt.disableflags": DisableBit.CONTACT | disableflags},
    )
    self.assertTrue((mjm.dof_damping > 0).any())

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)
    mujoco.mj_forward(mjm, mjd)
    mujoco.mj_Euler(mjm, mjd)
    mjw.solve(m, d)  # compute efc.Ma
    mjw.euler(m, d)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.qvel.numpy()[0], mjd.qvel, "qvel")
    _assert_eq(d.act.numpy()[0], mjd.act, "act")

  def test_rungekutta4(self):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
        <mujoco>
          <option integrator="RK4" iterations="1" ls_iterations="1">
            <flag constraint="disable"/>
          </option>
          <worldbody>
            <body>
              <joint type="hinge"/>
              <geom type="sphere" size=".1"/>
              <body pos="0.1 0 0">
                <joint type="hinge"/>
                <geom type="sphere" size=".1"/>
              </body>
            </body>
          </worldbody>
          <keyframe>
            <key qpos=".1 .2" qvel=".025 .05"/>
          </keyframe>
        </mujoco>
        """,
      keyframe=0,
    )

    mjw.rungekutta4(m, d)
    mujoco.mj_RungeKutta(mjm, mjd, 4)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.qvel.numpy()[0], mjd.qvel, "qvel")
    _assert_eq(d.time.numpy()[0], mjd.time, "time")
    _assert_eq(d.xpos.numpy()[0], mjd.xpos, "xpos")

    # test rungekutta determinism
    def rk_step() -> wp.array(dtype=wp.float32, ndim=2):
      d.qpos = wp.ones_like(d.qpos)
      d.qvel = wp.ones_like(d.qvel)
      d.act = wp.ones_like(d.act)
      mjw.rungekutta4(m, d)
      return d.qpos

    _assert_eq(rk_step().numpy()[0], rk_step().numpy()[0], "qpos")

  @parameterized.product(
    jacobian=(mujoco.mjtJacobian.mjJAC_AUTO, mujoco.mjtJacobian.mjJAC_DENSE),
    actuation=(0, DisableBit.ACTUATION),
    passive=(0, DisableBit.SPRING | DisableBit.DAMPER),
  )
  def test_implicit(self, jacobian, actuation, passive):
    mjm, mjd, _, _ = test_data.fixture(
      "pendula.xml",
      overrides={
        "opt.jacobian": jacobian,
        "opt.disableflags": DisableBit.CONTACT | actuation | passive,
        "opt.integrator": IntegratorType.IMPLICITFAST,
      },
    )

    mjm.actuator_gainprm[:, 2] = np.random.uniform(low=0.01, high=10.0, size=mjm.actuator_gainprm[:, 2].shape)

    # change actuators to velocity/damper to cover all codepaths
    mjm.actuator_gaintype[3] = GainType.AFFINE
    mjm.actuator_gaintype[6] = GainType.AFFINE
    mjm.actuator_biastype[0:3] = BiasType.AFFINE
    mjm.actuator_biastype[4:6] = BiasType.AFFINE
    mjm.actuator_biasprm[0:3, 2] = -1.0
    mjm.actuator_biasprm[4:6, 2] = -1.0
    mjm.actuator_ctrlrange[3:7] = 10.0
    mjm.actuator_gear[:] = 1.0

    mjd.qvel = np.random.uniform(low=-0.01, high=0.01, size=mjd.qvel.shape)
    mjd.ctrl = np.random.uniform(low=-0.1, high=0.1, size=mjd.ctrl.shape)
    mjd.act = np.random.uniform(low=-0.1, high=0.1, size=mjd.act.shape)
    mujoco.mj_forward(mjm, mjd)

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)

    mujoco.mj_implicit(mjm, mjd)

    mjw.solve(m, d)  # compute efc.Ma
    mjw.implicit(m, d)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.act.numpy()[0], mjd.act, "act")

  def test_implicit_position(self):
    mjm, mjd, m, d = test_data.fixture(
      "actuation/position.xml",
      keyframe=0,
      qvel_noise=0.01,
      ctrl_noise=0.1,
      overrides={"opt.integrator": IntegratorType.IMPLICITFAST},
    )

    mujoco.mj_implicit(mjm, mjd)

    mjw.solve(m, d)  # compute efc.Ma
    mjw.implicit(m, d)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.qvel.numpy()[0], mjd.qvel, "qvel")

  def test_implicit_tendon_damping(self):
    mjm, mjd, m, d = test_data.fixture(
      "tendon/damping.xml",
      keyframe=0,
      qvel_noise=0.01,
      ctrl_noise=0.1,
      overrides={"opt.integrator": IntegratorType.IMPLICITFAST},
    )

    mujoco.mj_implicit(mjm, mjd)

    mjw.solve(m, d)  # compute efc.Ma
    mjw.implicit(m, d)

    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "qpos")
    _assert_eq(d.qvel.numpy()[0], mjd.qvel, "qvel")

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  @parameterized.product(
    xml=("humanoid/humanoid.xml", "pendula.xml", "constraints.xml", "collision.xml"), graph_conditional=(True, False)
  )
  def test_graph_capture(self, xml, graph_conditional):
    # TODO(team): test more environments
    _, _, m, d = test_data.fixture(xml, overrides={"opt.graph_conditional": graph_conditional})

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    # step a few times to ensure no errors at the step boundary
    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)

    self.assertTrue(d.time.numpy()[0] > 0.0)

  def test_forward_energy(self):
    _, mjd, _, d = test_data.fixture(
      "humanoid/humanoid.xml", qvel_noise=0.01, ctrl_noise=0.1, overrides={"opt.enableflags": EnableBit.ENERGY}
    )

    _assert_eq(d.energy.numpy()[0][0], mjd.energy[0], "potential energy")
    _assert_eq(d.energy.numpy()[0][1], mjd.energy[1], "kinetic energy")

  def test_tendon_actuator_force_limits(self):
    for keyframe in range(7):
      _, mjd, m, d = test_data.fixture("actuation/tendon_force_limit.xml", keyframe=keyframe)

      d.actuator_force.zero_()

      mjw.forward(m, d)

      _assert_eq(d.actuator_force.numpy()[0], mjd.actuator_force, "actuator_force")

  @parameterized.product(xml=("humanoid/humanoid.xml",), energy=(0, EnableBit.ENERGY))
  def test_step1(self, xml, energy):
    # TODO(team): test more mjcfs
    mjm, mjd, m, d = test_data.fixture(
      xml, qpos_noise=0.1, qvel_noise=0.01, ctrl_noise=0.1, overrides={"opt.enableflags": energy}
    )

    # some of the fields updated by step1
    step1_field = [
      "xpos",
      "xquat",
      "xmat",
      "xipos",
      "ximat",
      "xanchor",
      "xaxis",
      "geom_xpos",
      "geom_xmat",
      "site_xmat",
      "subtree_com",
      "cinert",
      "cdof",
      "cam_xpos",
      "cam_xmat",
      "light_xpos",
      "light_xdir",
      "ten_length",
      "ten_J",
      "ten_wrapadr",
      "ten_wrapnum",
      "wrap_obj",
      "wrap_xpos",
      "qM",
      "qLD",
      "nefc",
      "efc_type",
      "efc_id",
      "efc_J",
      "efc_pos",
      "efc_margin",
      "efc_D",
      "efc_vel",
      "efc_aref",
      "efc_frictionloss",
      "actuator_length",
      "actuator_moment",
      "actuator_velocity",
      "ten_velocity",
      "cvel",
      "cdof_dot",
      "qfrc_spring",
      "qfrc_damper",
      "qfrc_gravcomp",
      "qfrc_fluid",
      "qfrc_passive",
      "qfrc_bias",
      "energy",
    ]
    if m.nflexvert:
      step1_field += ["flexvert_xpos"]
    if m.nflexedge:
      step1_field += ["flexedge_length", "flexedge_velocity"]

    def _getattr(arr):
      if (len(arr) >= 4) & (arr[:4] == "efc_"):
        return getattr(d.efc, arr[4:]), True
      return getattr(d, arr), False

    for arr in step1_field:
      attr, _ = _getattr(arr)
      if attr.dtype == float:
        attr.fill_(wp.nan)
      elif attr.dtype == int:
        attr.fill_(-1)
      else:
        attr.zero_()

    mujoco.mj_step1(mjm, mjd)
    mjw.step1(m, d)

    for arr in step1_field:
      d_arr, is_nefc = _getattr(arr)
      d_arr = d_arr.numpy()[0]
      mjd_arr = getattr(mjd, arr)
      if arr in ["xmat", "ximat", "geom_xmat", "site_xmat", "cam_xmat"]:
        mjd_arr = mjd_arr.reshape(-1)
        d_arr = d_arr.reshape(-1)
      elif arr == "qM":
        qM = np.zeros((mjm.nv, mjm.nv))
        mujoco.mj_fullM(mjm, qM, mjd.qM)
        mjd_arr = qM
      elif arr == "actuator_moment":
        actuator_moment = np.zeros((mjm.nu, mjm.nv))
        mujoco.mju_sparse2dense(actuator_moment, mjd.actuator_moment, mjd.moment_rownnz, mjd.moment_rowadr, mjd.moment_colind)
        mjd_arr = actuator_moment
      elif arr == "ten_J" and mjm.ntendon:
        ten_J = np.zeros((mjm.ntendon, mjm.nv))
        mujoco.mju_sparse2dense(ten_J, mjd.ten_J, mjd.ten_J_rownnz, mjd.ten_J_rowadr, mjd.ten_J_colind)
        mjd_arr = ten_J
      elif arr == "efc_J":
        if mjd.efc_J.shape[0] != mjd.nefc * mjm.nv:
          efc_J = np.zeros((mjd.nefc, mjm.nv))
          mujoco.mju_sparse2dense(efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
          mjd_arr = efc_J
        else:
          mjd_arr = mjd_arr.reshape((mjd.nefc, mjm.nv))
      elif arr == "qLD":
        vec = np.ones((1, mjm.nv))
        res = np.zeros((1, mjm.nv))
        mujoco.mj_solveM(mjm, mjd, res, vec)

        vec_wp = wp.array(vec, dtype=float)
        res_wp = wp.zeros((1, mjm.nv), dtype=float)
        mjw.solve_m(m, d, res_wp, vec_wp)

        d_arr = res_wp.numpy()[0]
        mjd_arr = res[0]
      if is_nefc:
        d_arr = d_arr[: d.nefc.numpy()[0]]

      _assert_eq(d_arr, mjd_arr, arr)

    # TODO(team): sensor_pos
    # TODO(team): sensor_vel

  @parameterized.product(
    xml=("humanoid/humanoid.xml",),
    integrator=(IntegratorType.EULER, IntegratorType.IMPLICITFAST, IntegratorType.RK4),
  )
  def test_step2(self, xml, integrator):
    mjm, mjd, m, _ = test_data.fixture(xml, qvel_noise=0.01, ctrl_noise=0.1, overrides={"opt.integrator": integrator})

    # some of the fields updated by step2
    step2_field = [
      "act_dot",
      "actuator_force",
      "qfrc_actuator",
      "qfrc_smooth",
      "qacc",
      "qacc_warmstart",
      "qvel",
      "qpos",
      "efc_force",
      "qfrc_constraint",
    ]

    def _getattr(arr):
      if (len(arr) >= 4) & (arr[:4] == "efc_"):
        return getattr(d.efc, arr[4:]), True
      return getattr(d, arr), False

    mujoco.mj_step1(mjm, mjd)

    # input
    ctrl = 0.1 * np.random.rand(mjm.nu)
    qfrc_applied = 0.1 * np.random.rand(mjm.nv)
    xfrc_applied = 0.1 * np.random.rand(mjm.nbody, 6)

    mjd.ctrl = ctrl
    mjd.qfrc_applied = qfrc_applied
    mjd.xfrc_applied = xfrc_applied

    d = mjw.put_data(mjm, mjd)

    for arr in step2_field:
      if arr in ["qpos", "qvel"]:
        continue
      attr, _ = _getattr(arr)
      if attr.dtype == float:
        attr.fill_(wp.nan)
      elif attr.dtype == int:
        attr.fill_(-1)
      else:
        attr.zero_()

    mujoco.mj_step2(mjm, mjd)
    mjw.step2(m, d)

    for arr in step2_field:
      d_arr, is_efc = _getattr(arr)
      d_arr = d_arr.numpy()[0]
      if is_efc:
        d_arr = d_arr[: d.nefc.numpy()[0]]
      _assert_eq(d_arr, getattr(mjd, arr), arr)

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  @parameterized.parameters(IntegratorType)
  def test_act_limited(self, integrator):
    """Test activation limits."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep=".01"/>
        <worldbody>
          <body>
            <joint name="slide" type="slide" axis="1 0 0"/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <actuator>
          <general joint="slide" gainprm="100" biasprm="0 -100" biastype="affine" dynprm="10" dyntype="integrator" actlimited="true" actrange="-1 1"/>
        </actuator>
      </mujoco>
    """,
      overrides={"opt.integrator": integrator},
    )

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    d.ctrl.fill_(1.0)

    # integrating up from 0, we will hit the clamp after 99 steps
    for i in range(200):
      wp.capture_launch(capture.graph)
      act = d.act.numpy()[0, 0]
      # always greater than the lower bound
      self.assertGreater(act, -1.0)
      # after 99 steps we hit the upper bound
      if i < 99:
        self.assertLess(act, 1.0)
      elif i >= 99:
        self.assertAlmostEqual(act, 1.0, 4)

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  def test_damper_dampens(self):
    """Test actuator dampers."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <geom size="1"/>
            <joint name="jnt" type="slide" axis="1 0 0"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="jnt"/>
          <damper joint="jnt" kv="1000" ctrlrange="0 100"/>
        </actuator>
      </mujoco>
    """
    )

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    # move the joint
    wp.copy(d.ctrl, wp.array(np.array([[100.0, 0.0]]), dtype=float))
    for _ in range(100):
      wp.capture_launch(capture.graph)

    # stop the joint with damping
    wp.copy(d.ctrl, wp.array(np.array([[0.0, 100.0]]), dtype=float))
    for _ in range(1000):
      wp.capture_launch(capture.graph)

    self.assertLess(d.qvel.numpy()[0, 0], 1.0e-4)

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  @parameterized.parameters(0, DisableBit.ACTUATION)
  def test_armature_equivalence(self, actuation):
    """Test that joint armature is equivalent to a coupled rotating mass with a gear ratio enforced by an equality."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="link1">
            <joint name="link1" axis="0 -1 0"/>
            <geom type="capsule" size=".02" fromto="0 0 0 1 0 0" mass="0"/>
            <geom type="sphere" size=".1" mass="1" pos="1 0 0"/>
          </body>
          <body name="motor1">
            <joint name="motor1" axis="0 -1 0"/>
            <geom type="box" size=".1 .1 .1" mass="30" contype="0" conaffinity="0"/>
          </body>
          <body name="link2" pos="1.5 0 0">
            <joint name="link2" armature="1.8" axis="0 -1 0"/>
            <geom type="capsule" size=".02" fromto="0 0 0 1 0 0" mass="0"/>
            <geom type="sphere" size=".1" mass="1" pos="1 0 0"/>
          </body>
        </worldbody>
        <equality>
          <joint joint1="motor1" joint2="link1" polycoef="0 3"/>
        </equality>
        <actuator>
          <position name="link1" joint="motor1" kp=".3" ctrlrange="-5 5"/>
          <position name="link2" joint="link2" kp=".3" ctrlrange="-5 5" gear="3"/>
        </actuator>
      </mujoco>
    """,
      overrides={"opt.disableflags": actuation},
    )

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    qpos_mse = 0.0
    nstep = 0
    time = d.time.numpy()[0]

    while time < 4.0:
      wp.copy(d.ctrl, wp.array(2 * time * np.zeros((1, m.nu)), dtype=float))
      wp.capture_launch(capture.graph)
      nstep += 1
      qpos = d.qpos.numpy()[0]
      err = qpos[0] - qpos[2]
      qpos_mse += err * err
      time = d.time.numpy()[0]

    self.assertLess(np.sqrt(qpos_mse / nstep), 1.0e-3)

  @parameterized.parameters(0, DisableBit.EULERDAMP)
  def test_eulerdamp_disable(self, eulerdamp):
    """Tests implicit joint damping works as expected."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <joint axis="1 0 0" damping="2"/>
          <geom type="capsule" size=".01" fromto="0 0 0 0 .1 0"/>
          <body pos="0 .1 0">
            <joint axis="0 1 0" damping="1"/>
            <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """,
      overrides={"opt.disableflags": eulerdamp},
    )

    # step once, call forward, save qvel and qacc
    mjw.step(m, d)
    mjw.forward(m, d)

    qvel = d.qvel.numpy()[0]
    qacc = d.qacc.numpy()[0]

    # second step
    mjw.step(m, d)

    if eulerdamp:
      # compute finite-difference acceleration
      qacc_fd = (d.qvel.numpy()[0] - qvel) / m.opt.timestep.numpy()[0]
      _assert_eq(qacc_fd, qacc, "qacc")
    else:
      self.assertGreater(np.linalg.norm(qacc), 1.0)

  def test_eulerdamp_limit(self):
    """Reducing the timesteps reduces the difference between implicit/explicit."""
    _MJCF = """
    <mujoco>
      <worldbody>
        <body>
          <joint axis="1 0 0" damping="2"/>
          <geom type="capsule" size=".01" fromto="0 0 0 0 .1 0"/>
          <body pos="0 .1 0">
            <joint axis="0 1 0" damping="1"/>
            <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """

    diff_norm_prev = -1

    for timestep in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
      mjm, mjd, m, _ = test_data.fixture(xml=_MJCF)

      m.opt.timestep.fill_(timestep)
      m.opt.disableflags = 0

      # step twice with implicit damping, save qvel
      mujoco.mj_resetData(mjm, mjd)
      d = mjw.put_data(mjm, mjd)
      mjw.step(m, d)
      mjw.step(m, d)
      qvel_imp = d.qvel.numpy()[0]

      # step once, step again without implicit damping, save qvel
      mujoco.mj_resetData(mjm, mjd)
      d = mjw.put_data(mjm, mjd)
      mjw.step(m, d)
      m.opt.disableflags |= DisableBit.EULERDAMP
      mjw.step(m, d)
      qvel_exp = d.qvel.numpy()[0]

      diff_norm = np.linalg.norm(qvel_imp - qvel_exp)

      if diff_norm_prev != -1:
        self.assertLess(diff_norm, diff_norm_prev)

      diff_norm_prev = diff_norm

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  def test_euler_implicitfast_equivalent(self):
    """Test that Euler and implicitfast integrators produce equivalent results (to numerical tolerance)."""
    _MJCF = """
    <mujoco>
      <worldbody>
        <body>
          <joint axis="1 0 0" damping="2"/>
          <geom type="capsule" size=".01" fromto="0 0 0 0 .1 0"/>
          <body pos="0 .1 0">
            <joint axis="0 1 0" damping="1"/>
            <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """

    nstep = 10

    # step 10 times with Euler, save copy of qpos
    mjm, mjd, m, _ = test_data.fixture(xml=_MJCF, overrides={"opt.integrator": IntegratorType.EULER})
    mujoco.mj_resetData(mjm, mjd)
    d = mjw.put_data(mjm, mjd)

    with wp.ScopedCapture() as capture1:
      mjw.step(m, d)

    for _ in range(nstep):
      wp.capture_launch(capture1.graph)

    qpos_euler = d.qpos.numpy()[0].copy()

    # reset, step 10 times with implicitfast
    mjm, mjd, m, _ = test_data.fixture(xml=_MJCF, overrides={"opt.integrator": IntegratorType.IMPLICITFAST})
    mujoco.mj_resetData(mjm, mjd)
    d = mjw.put_data(mjm, mjd)

    with wp.ScopedCapture() as capture2:
      mjw.step(m, d)

    for _ in range(nstep):
      wp.capture_launch(capture2.graph)

    qpos_implicitfast = d.qpos.numpy()[0].copy()

    # expect qpos to be numerically different
    # TODO(team): qpos_euler and qpos_implicit seem to be identical here?
    #             update to implicit integrator
    # self.assertGreater(np.linalg.norm(qpos_euler - qpos_implicitfast), 0)

    # expect qpos to be similar to high precision
    _assert_eq(qpos_euler, qpos_implicitfast, "qpos")

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  def test_joint_actuator_equivalent(self):
    """Tests that dof damping and actuator damping are equivalent (to numerical tolerance)."""
    _MJCF = """
    <mujoco>
      <worldbody>
        <body>
          <joint type="slide" axis="0 0 1" damping="10"/>
          <geom size=".03"/>
          <body>
            <joint axis="0 1 0" damping=".1"/>
            <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
          </body>
        </body>
        <body pos="0 .1 0">
          <joint name="slide" type="slide" axis="0 0 1"/>
          <geom size=".03"/>
          <body>
            <joint name="hinge" axis="0 1 0"/>
            <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <general joint="slide" biastype="affine" biasprm="0 0 -10"/>
        <general joint="hinge" biastype="affine" biasprm="0 0 -.1"/>
      </actuator>
    </mujoco>
    """

    _, _, m, d = test_data.fixture(xml=_MJCF)

    # take 1000 steps with Euler
    with wp.ScopedCapture() as capture1:
      mjw.step(m, d)

    for _ in range(1000):
      wp.capture_launch(capture1.graph)

    # expect corresponding joint values to be significantly difference
    qpos = d.qpos.numpy()[0]
    self.assertGreater(np.abs(qpos[0] - qpos[2]), 1.0e-4)
    self.assertGreater(np.abs(qpos[1] - qpos[3]), 1.0e-4)

    # reset, take 10 steps with implicitfast
    _, _, m, d = test_data.fixture(xml=_MJCF, overrides={"opt.integrator": IntegratorType.IMPLICITFAST})

    with wp.ScopedCapture() as capture2:
      mjw.step(m, d)

    for _ in range(10):
      wp.capture_launch(capture2.graph)

    # expect corresponding joint values to be insignificantly different
    qpos = d.qpos.numpy()[0]
    self.assertLess(np.abs(qpos[0] - qpos[2]), 1.0e-8)
    self.assertLess(np.abs(qpos[1] - qpos[3]), 1.0e-8)

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  def test_energy_conservation(self):
    """Tests energy conservation."""
    _MJCF = """
    <mujoco>
      <option integrator="implicit">
        <flag constraint="disable" energy="enable"/>
      </option>
      <worldbody>
        <geom type="plane" size="1 1 .01" pos="0 0 -1"/>
        <body pos=".15 0 0">
          <joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" size=".02" fromto="0 0 0 .1 0 0"/>
          <body pos=".1 0 0">
            <joint type="slide" axis="1 0 0" stiffness="200"/>
            <geom type="capsule" size=".015" fromto="-.1 0 0 .1 0 0"/>
            <body pos=".1 0 0">
              <joint type="ball"/>
              <geom type="box" size=".02" fromto="0 0 0 0 .1 0"/>
              <body pos="0 .1 0">
                <joint axis="1 0 0"/>
                <geom type="capsule" size=".02" fromto="0 0 0 0 .1 0"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """

    nstep = 500

    # take nstep steps with Euler, measure energy (potential + kinetic)
    _, _, m, d = test_data.fixture(xml=_MJCF, overrides={"opt.integrator": IntegratorType.EULER})

    with wp.ScopedCapture() as capture1:
      mjw.step(m, d)

    for _ in range(nstep):
      wp.capture_launch(capture1.graph)

    energy_euler = d.energy.numpy()[0][0] + d.energy.numpy()[0][1]

    # take nstep steps with implicitfast, measure energy
    _, _, m, d = test_data.fixture(xml=_MJCF, overrides={"opt.integrator": IntegratorType.IMPLICITFAST})

    with wp.ScopedCapture() as capture2:
      mjw.step(m, d)

    for _ in range(nstep):
      wp.capture_launch(capture2.graph)

    energy_implicitfast = d.energy.numpy()[0][0] + d.energy.numpy()[0][1]

    # take nstep steps with RK4, measure energy
    _, _, m, d = test_data.fixture(xml=_MJCF, overrides={"opt.integrator": IntegratorType.RK4})

    with wp.ScopedCapture() as capture3:
      mjw.step(m, d)

    for _ in range(nstep):
      wp.capture_launch(capture3.graph)

    energy_rk4 = d.energy.numpy()[0][0] + d.energy.numpy()[0][1]

    # energy was measured: expect all energies to be nonzero
    self.assertNotEqual(energy_euler, 0.0)
    self.assertNotEqual(energy_implicitfast, 0.0)
    self.assertNotEqual(energy_rk4, 0.0)

    # test conservation: perfectly conserved energy would remain 0.0
    # expect RK4 to be better than implicitfast
    self.assertLess(np.abs(energy_rk4), np.abs(energy_implicitfast))

    # expect implicitfast to be better than Euler
    # TODO(team): assertLess for implicit integrator
    self.assertLessEqual(np.abs(energy_implicitfast), np.abs(energy_euler))

  def test_control_clamping(self):
    """Tests control clamping"""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <body>
          <geom size="1"/>
          <joint name="slide" type="slide" axis="1 0 0"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="unclamped" joint="slide"/>
        <motor name="clamped" joint="slide" ctrllimited="true" ctrlrange="-1 1"/>
      </actuator>
    </mujoco>
    """
    )

    # for the unclamped actuator, ctrl={1, 2} produce different accelerations
    wp.copy(d.ctrl, wp.array(np.array([[1, 0]]), dtype=float))
    mjw.forward(m, d)
    qacc1 = d.qacc.numpy()[0, 0]
    wp.copy(d.ctrl, wp.array(np.array([[2, 0]]), dtype=float))
    mjw.forward(m, d)
    qacc2 = d.qacc.numpy()[0, 0]
    self.assertGreater(np.abs(qacc1 - qacc2), 1e-4)

    # for the clamped actuator, ctrl={1, 2} produce identical accelerations
    d.ctrl.zero_()

    wp.copy(d.ctrl, wp.array(np.array([[0, 1]]), dtype=float))
    mjw.forward(m, d)
    qacc1 = d.qacc.numpy()[0, 0]
    wp.copy(d.ctrl, wp.array(np.array([[0, 2]]), dtype=float))
    mjw.forward(m, d)
    qacc2 = d.qacc.numpy()[0, 0]
    self.assertLess(np.abs(qacc1 - qacc2), 1e-4)

    # d.ctrl[1] remains pristine
    self.assertAlmostEqual(d.ctrl.numpy()[0, 1], 2.0)

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  def test_gravcomp(self):
    """Test gravity compensation."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option gravity="0 0 -10"/>
      <worldbody>
        <body>
          <joint type="slide" axis="0 0 1"/>
          <geom size="1"/>
        </body>
        <body pos="3 0 0" gravcomp="1">
          <joint type="slide" axis="0 0 1"/>
          <geom size="1"/>
        </body>
        <body pos="6 0 0" gravcomp="2">
          <joint type="slide" axis="0 0 1"/>
          <geom size="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    )

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    while d.time.numpy()[0] < 1.0:
      wp.capture_launch(capture.graph)

    time = d.time.numpy()[0]
    dist = 0.5 * np.linalg.norm(m.opt.gravity.numpy()[0]) * time * time

    qpos = d.qpos.numpy()[0]

    # expect that body 1 moved down, allowing some slack from our estimate
    self.assertLess(np.abs(qpos[0] + dist), 0.011)

    # expect that body 2 does not move
    _assert_eq(qpos[1], 0.0, "qpos[1]==0")

    # expect that body 3 moves up the same distance that body 0 moved down
    _assert_eq(qpos[0], -qpos[2], "qpos[0]==-qpos[2]")

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test that requires GPU.")
  def test_eq_active(self):
    """Tests disabling of equality constraints."""
    _MJCF = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="vertical" type="slide" axis="0 0 1"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <equality>
        <joint joint1="vertical"/>
      </equality>
    </mujoco>
    """

    _, _, m, d = test_data.fixture(xml=_MJCF)

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    # simulate for 1 second
    while d.time.numpy()[0] < 1.0:
      wp.capture_launch(capture.graph)

    # expect that the body has barely moved
    self.assertLess(np.abs(d.qpos.numpy()[0, 0]), 0.001)

    # turn the equality off, simulate for another second
    d.eq_active.fill_(False)

    while d.time.numpy()[0] < 2.0:
      wp.capture_launch(capture.graph)

    # expect that the body has fallen about 5m
    qpos = d.qpos.numpy()[0]
    self.assertLess(qpos[0], -4.5)
    self.assertGreater(qpos[0], -5.5)

    # turn the equality back on, simulate for another second
    d.eq_active.fill_(True)

    while d.time.numpy()[0] < 3.0:
      wp.capture_launch(capture.graph)

    # expect that the body has snapped back
    self.assertLess(np.abs(d.qpos.numpy()[0, 0]), 0.001)

  def test_actuator_force_clamping(self):
    """Tests actuator force clamping at joints."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <option integrator="implicitfast"/>
      <worldbody>
        <geom type="plane" size="1 1 .01"/>
        <body pos="0 0 .3">
          <joint name="hinge" damping=".01" actuatorfrcrange="-.4 .4"/>
          <geom type="capsule" size=".01" fromto="0 0 0 .2 0 0"/>
          <geom size=".03" pos=".2 0 0"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor" joint="hinge" ctrlrange="-1 1"/>
        <damper name="damper" joint="hinge" kv="10" ctrlrange="0 1"/>
      </actuator>
      <sensor>
        <actuatorfrc name="motor" actuator="motor"/>
        <actuatorfrc name="damper" actuator="damper"/>
        <jointactuatorfrc name="hinge" joint="hinge"/>
      </sensor>
    </mujoco>
    """
    )

    wp.copy(d.ctrl, wp.array(np.array([[10, 0]]), dtype=float))
    mjw.forward(m, d)

    # expect clamping as specified in the model
    _assert_eq(d.actuator_force.numpy()[0, 0], 1.0, "actuator_force")
    _assert_eq(d.qfrc_actuator.numpy()[0, 0], 0.4, "qfrc_actuator")

    # simulate for 2 seconds to gain velocity
    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    while d.time.numpy()[0] < 2.0:
      wp.capture_launch(capture.graph)

    # activate damper, expect force to be clamped at lower bound
    wp.copy(d.ctrl, wp.array(np.array([[10, 1]]), dtype=float))
    mjw.forward(m, d)
    _assert_eq(d.qfrc_actuator.numpy()[0, 0], -0.4, "qfrc_actuator")

  @absltest.skipIf(not wp.get_device().is_cuda, "Skipping test requiring GPU.")
  def test_damp_ratio(self):
    """Tests that dampratio works as expected."""
    _MJCF = """
    <mujoco>
      <option integrator="implicitfast"/>
      <worldbody>
        <body>
          <joint name="slide1" axis="1 0 0" type="slide"/>
          <geom size=".05"/>
        </body>
        <body pos="0 0 -.15">
          <joint name="slide2" axis="1 0 0" type="slide"/>
          <geom size=".05"/>
        </body>
      </worldbody>
      <actuator>
        <position joint="slide1" kp="10" dampratio=".99"/>
        <position joint="slide2" kp="10" dampratio="1.01"/>
      </actuator>
    </mujoco>
    """

    _, _, m, d = test_data.fixture(xml=_MJCF)

    d.qpos.fill_(-0.1)

    qpos = d.qpos.numpy()[0]
    under_damped = qpos[0]
    over_damped = qpos[1]

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    while d.time.numpy()[0] < 10.0:
      wp.capture_launch(capture.graph)
      qpos = d.qpos.numpy()[0]
      under_damped = np.maximum(under_damped, qpos[0])
      over_damped = np.maximum(over_damped, qpos[1])

    # expect slightly underdamped to slightly overshoot
    self.assertGreater(under_damped, 0.0)
    self.assertLess(under_damped, 1.0e-6)

    # expect slightly overdamped to slightly undershoot
    self.assertLess(over_damped, 0.0)
    self.assertGreater(over_damped, -1.0e-6)


if __name__ == "__main__":
  wp.init()
  absltest.main()
