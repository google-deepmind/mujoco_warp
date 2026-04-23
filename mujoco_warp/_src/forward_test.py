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
from mujoco_warp import DynType
from mujoco_warp import EnableBit
from mujoco_warp import GainType
from mujoco_warp import IntegratorType
from mujoco_warp import test_data

# tolerance for difference between MuJoCo and mjwarp smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class ForwardTest(parameterized.TestCase):
  @parameterized.product(xml=["humanoid/humanoid.xml", "pendula.xml"])
  def test_fwd_velocity(self, xml):
    _, mjd, m, d = test_data.fixture(xml, qvel_noise=0.01, ctrl_noise=0.1)

    for arr in (d.actuator_velocity, d.qfrc_bias):
      arr.fill_(wp.inf)

    mjw.fwd_velocity(m, d)

    _assert_eq(d.actuator_velocity.numpy()[0], mjd.actuator_velocity, "actuator_velocity")
    _assert_eq(d.qfrc_bias.numpy()[0], mjd.qfrc_bias, "qfrc_bias")

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_fwd_velocity_tendon(self, jacobian):
    _, mjd, m, d = test_data.fixture("tendon/fixed.xml", overrides={"opt.jacobian": jacobian})

    d.ten_velocity.fill_(wp.inf)
    mjw.fwd_velocity(m, d)

    _assert_eq(d.ten_velocity.numpy()[0], mjd.ten_velocity, "ten_velocity")

  @parameterized.product(
    xml=("actuation/actuation.xml", "actuation/actuators.xml", "actuation/muscle.xml"),
    disableflags=(0, DisableBit.ACTUATION),
  )
  def test_actuation(self, xml, disableflags):
    mjm, mjd, m, d = test_data.fixture(xml, keyframe=0, overrides={"opt.disableflags": disableflags})

    for arr in (d.qfrc_actuator, d.actuator_force, d.act_dot):
      arr.fill_(wp.inf)

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
      arr.fill_(wp.inf)

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
    def rk_step() -> wp.array2d[wp.float32]:
      d.qpos = wp.ones_like(d.qpos)
      d.qvel = wp.ones_like(d.qvel)
      d.act = wp.ones_like(d.act)
      mjw.rungekutta4(m, d)
      return d.qpos

    _assert_eq(rk_step().numpy()[0], rk_step().numpy()[0], "qpos")

  @parameterized.product(
    jacobian=(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE),
    actuation=(0, DisableBit.ACTUATION),
    spring=(0, DisableBit.SPRING),
    damper=(0, DisableBit.DAMPER),
  )
  def test_implicit(self, jacobian, actuation, spring, damper):
    mjm, mjd, _, _ = test_data.fixture(
      "pendula.xml",
      overrides={
        "opt.jacobian": jacobian,
        "opt.disableflags": DisableBit.CONTACT | actuation | spring | damper,
        "opt.integrator": IntegratorType.IMPLICITFAST,
        # TODO(team): remove override when mujoco warp feature matches mujoco
        "opt.enableflags": EnableBit.INVDISCRETE,
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
    # compute efc.Ma - used by mjw.implicit
    d.efc.Ma = wp.array(mjd.qfrc_constraint + mjd.qfrc_smooth, dtype=wp.float32, shape=(1, -1))

    mujoco.mj_implicit(mjm, mjd)
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

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_implicit_tendon_damping(self, jacobian):
    mjm, mjd, m, d = test_data.fixture(
      "tendon/damping.xml",
      keyframe=0,
      qvel_noise=0.01,
      ctrl_noise=0.1,
      overrides={"opt.integrator": IntegratorType.IMPLICITFAST, "opt.jacobian": jacobian},
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

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_tendon_actuator_force_limits(self, jacobian):
    for keyframe in range(7):
      _, mjd, m, d = test_data.fixture(
        "actuation/tendon_force_limit.xml", keyframe=keyframe, overrides={"opt.jacobian": jacobian}
      )

      d.actuator_force.fill_(wp.inf)

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
      if arr in ("geom_xpos", "geom_xmat"):
        # leave geom_xpos and geom_xmat untouched because they have static data
        continue
      attr, _ = _getattr(arr)
      if arr in ("xquat", "xmat", "ximat"):
        # xquat, xmat, ximat need to retain identity for world body
        attr = attr[:, 1:]
      if attr.dtype == float:
        attr.fill_(wp.nan)
      elif attr.dtype == int:
        attr.fill_(-1)
      else:
        attr.zero_()

    mujoco.mj_step1(mjm, mjd)
    mjw.step1(m, d)

    # Precompute sorting for efc fields to avoid non determinism
    nefc = d.nefc.numpy()[0]
    if nefc > 0:
      nv = m.nv
      if m.is_sparse:
        # Reconstruct dense J from sparse representation
        d_efc_J = np.zeros((nefc, nv))
        mujoco.mju_sparse2dense(
          d_efc_J,
          d.efc.J.numpy()[0, 0],
          d.efc.J_rownnz.numpy()[0, :nefc],
          d.efc.J_rowadr.numpy()[0, :nefc],
          d.efc.J_colind.numpy()[0, 0],
        )
      else:
        d_efc_J = d.efc.J.numpy()[0, :nefc, :nv]
      if mjd.efc_J.shape[0] != mjd.nefc * mjm.nv:
        mjd_efc_J = np.zeros((mjd.nefc, mjm.nv))
        mujoco.mju_sparse2dense(mjd_efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
      else:
        mjd_efc_J = mjd.efc_J.reshape((mjd.nefc, mjm.nv))

      # Sort by efc_type, then efc_J columns (tiebreaker)
      d_sort = np.lexsort((*d_efc_J.T, d.efc.type.numpy()[0, :nefc]))
      mjd_sort = np.lexsort((*mjd_efc_J.T, mjd.efc_type[:nefc]))
      _assert_eq(d_efc_J[d_sort].reshape(-1), mjd_efc_J[mjd_sort].reshape(-1), "efc_J")

      # Check efc_id here as a contact may have a different id
      _assert_eq(sorted(d.efc.id.numpy()[0, :nefc]), sorted(mjd.efc_id[:nefc]), "efc_id")

    for arr in step1_field:
      d_arr, is_efc = _getattr(arr)

      d_arr = d_arr.numpy()[0]
      mjd_arr = getattr(mjd, arr)
      if arr in ["xmat", "ximat", "geom_xmat", "site_xmat", "cam_xmat"]:
        mjd_arr = mjd_arr.reshape(-1)
        d_arr = d_arr.reshape(-1)
      elif arr == "qM":
        qM = np.zeros((mjm.nv, mjm.nv))
        mujoco.mj_fullM(mjm, qM, mjd.qM)
        mjd_arr = qM
        d_arr = d_arr[: mjm.nv, : mjm.nv]
      elif arr == "actuator_moment":
        actuator_moment = np.zeros((mjm.nu, mjm.nv))
        mujoco.mju_sparse2dense(actuator_moment, mjd.actuator_moment, mjd.moment_rownnz, mjd.moment_rowadr, mjd.moment_colind)
        mjd_arr = actuator_moment
        d_arr = np.zeros((mjm.nu, mjm.nv))
        mujoco.mju_sparse2dense(
          d_arr,
          d.actuator_moment.numpy()[0],
          d.moment_rownnz.numpy()[0],
          d.moment_rowadr.numpy()[0],
          d.moment_colind.numpy()[0],
        )
      elif arr == "ten_J":
        # convert warp sparse ten_J to dense for comparison
        d_ten_J = np.zeros((mjm.ntendon, mjm.nv))
        if mjm.ntendon:
          mujoco.mju_sparse2dense(d_ten_J, d_arr, m.ten_J_rownnz.numpy(), m.ten_J_rowadr.numpy(), m.ten_J_colind.numpy())
        d_arr = d_ten_J
        ten_J = np.zeros((mjm.ntendon, mjm.nv))
        if mjm.ntendon:
          mujoco.mju_sparse2dense(ten_J, mjd.ten_J, mjm.ten_J_rownnz, mjm.ten_J_rowadr, mjm.ten_J_colind)
        mjd_arr = ten_J
      elif arr == "efc_J" or arr == "efc_id":
        # Already checked earlier
        continue
      elif arr == "qLD":
        vec = np.ones((1, mjm.nv))
        res = np.zeros((1, mjm.nv))
        mujoco.mj_solveM(mjm, mjd, res, vec)

        vec_wp = wp.array(vec, dtype=float)
        res_wp = wp.zeros((1, mjm.nv), dtype=float)
        mjw.solve_m(m, d, res_wp, vec_wp)

        d_arr = res_wp.numpy()[0]
        mjd_arr = res[0]

      if is_efc:
        nefc = d.nefc.numpy()[0]
        nv = m.nv
        d_arr = d_arr[:nefc]
        d_arr = d_arr[d_sort].reshape(-1)
        mjd_arr = mjd_arr[mjd_sort].reshape(-1)

      _assert_eq(d_arr, mjd_arr, arr)

    # TODO(team): sensor_pos
    # TODO(team): sensor_vel

  @parameterized.product(
    xml=("humanoid/humanoid.xml",),
    integrator=(IntegratorType.EULER, IntegratorType.IMPLICITFAST, IntegratorType.RK4),
  )
  def test_step2(self, xml, integrator):
    # TODO(team): remove enableflags override when mujoco warp feature matches mujoco
    enableflags = EnableBit.INVDISCRETE if integrator == IntegratorType.IMPLICITFAST else 0
    mjm, mjd, m, _ = test_data.fixture(
      xml, qvel_noise=0.01, ctrl_noise=0.1, overrides={"opt.integrator": integrator, "opt.enableflags": enableflags}
    )

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

  def test_step_no_dofs(self):
    """Tests step with no degrees of freedom."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <body>
        <geom type="sphere" size="1"/>
      </body>
    </mujoco>
    """
    )
    mjw.step(m, d)
    self.assertGreater(d.time.numpy()[0], 0.0)

  def test_control_callback(self):
    """Tests control_callback is called during forward and skipped when actuation disabled."""
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <geom size="1"/>
          <joint name="hinge"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="hinge"/>
      </actuator>
    </mujoco>
    """

    @wp.kernel
    def _set_ctrl(ctrl_out: wp.array2d[float]):
      worldid = wp.tid()
      ctrl_out[worldid, 0] = 2.0

    def my_control(m, d):
      wp.launch(_set_ctrl, dim=(d.nworld,), outputs=[d.ctrl])

    _, _, m, d = test_data.fixture(xml=xml)
    m.callback.control = my_control
    mjw.forward(m, d)
    self.assertEqual(d.ctrl.numpy()[0, 0], 2.0)

    # reset ctrl, disable actuation, verify callback not called
    d.ctrl.fill_(5.0)
    m.opt.disableflags |= DisableBit.ACTUATION
    mjw.forward(m, d)
    self.assertEqual(d.ctrl.numpy()[0, 0], 5.0)

  @parameterized.product(
    frequency=(1.5, 0.5),
    timestamp=(0.2, 0.4),
  )
  def test_act_dyn_callback(self, frequency, timestamp):
    """Tests act_dyn_callback with a harmonic oscillator."""

    @wp.kernel
    def _oscillator_act_dot(
      act_in: wp.array2d[float],
      ctrl_in: wp.array2d[float],
      act_dot_out: wp.array2d[float],
    ):
      worldid = wp.tid()
      frequency = wp.static(2.0 * wp.pi) * ctrl_in[worldid, 0]
      act_dot_out[worldid, 0] = -act_in[worldid, 1] * frequency
      act_dot_out[worldid, 1] = act_in[worldid, 0] * frequency

    def oscillator(m, d):
      wp.launch(
        _oscillator_act_dot,
        dim=(d.nworld,),
        inputs=[d.act, d.ctrl],
        outputs=[d.act_dot],
      )

    xml = f"""
    <mujoco>
      <option timestep="1e-4"/>
      <worldbody>
        <body>
          <geom size="1"/>
          <joint name="hinge"/>
        </body>
      </worldbody>
      <actuator>
        <general joint="hinge" dyntype="user" actdim="2"/>
      </actuator>
      <keyframe>
        <key time="{timestamp}" ctrl="{frequency}" act="{np.cos(2 * np.pi * frequency * timestamp)} {np.sin(2 * np.pi * frequency * timestamp)}"/>
      </keyframe>
    </mujoco>
    """

    mjm, _, m, d = test_data.fixture(xml=xml, keyframe=0)
    m.callback.act_dyn = oscillator

    mjw.step(m, d)

    # verify act after one step matches analytical solution at t + dt
    t_next = timestamp + mjm.opt.timestep
    np.testing.assert_allclose(d.act.numpy()[0, 0], np.cos(2 * np.pi * frequency * t_next), atol=1e-3)
    np.testing.assert_allclose(d.act.numpy()[0, 1], np.sin(2 * np.pi * frequency * t_next), atol=1e-3)

  def test_multiflex(self):
    """Tests multiflex model with different flex dimensions."""
    _, _, m, d = test_data.fixture("flex/multiflex.xml")

    mjw.forward(m, d)


class DCMotorTest(parameterized.TestCase):
  def test_dcmotor_stateless_steady_state(self):
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint" type="slide" axis="1 0 0"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <general name="dcmotor" joint="joint" actdim="0"/>
      </actuator>
      <keyframe>
        <key ctrl="12.0" qvel="3.0"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)

    # Override types in Warp model (mjGAIN_DCMOTOR=4, mjBIAS_DCMOTOR=4, mjDYN_NONE=0)
    wp.copy(m.actuator_gaintype, wp.array([int(GainType.DCMOTOR)], dtype=m.actuator_gaintype.dtype))
    wp.copy(m.actuator_biastype, wp.array([int(BiasType.DCMOTOR)], dtype=m.actuator_biastype.dtype))
    wp.copy(m.actuator_dyntype, wp.array([int(DynType.NONE)], dtype=m.actuator_dyntype.dtype))

    # Set parameters
    gainprm = np.zeros((1, 1, 10), dtype=np.float32)
    gainprm[0, 0, 0] = 2.0  # R
    gainprm[0, 0, 1] = 0.05  # K
    wp.copy(m.actuator_gainprm, wp.array(gainprm, dtype=m.actuator_gainprm.dtype))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    # Analytical expected force: 0.29625
    force = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force, 0.29625, atol=1e-5)

  def test_dcmotor_current_filter(self):
    xml = """
    <mujoco>
      <option timestep="0.0001"/>
      <worldbody>
        <body>
          <joint name="joint" type="slide" axis="1 0 0"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <general name="dcmotor" joint="joint" dyntype="integrator" actdim="1"/>
      </actuator>
      <keyframe>
        <key ctrl="10.0" act="8.646647"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)

    # Override types (mjDYN_DCMOTOR=6, mjGAIN_DCMOTOR=4, mjBIAS_DCMOTOR=4)
    wp.copy(m.actuator_dyntype, wp.array([int(DynType.DCMOTOR)], dtype=m.actuator_dyntype.dtype))
    wp.copy(m.actuator_gaintype, wp.array([int(GainType.DCMOTOR)], dtype=m.actuator_gaintype.dtype))
    wp.copy(m.actuator_biastype, wp.array([int(BiasType.DCMOTOR)], dtype=m.actuator_biastype.dtype))

    gainprm = np.zeros((1, 1, 10), dtype=np.float32)
    gainprm[0, 0, 0] = 1.0  # R
    gainprm[0, 0, 1] = 1.0  # K
    wp.copy(m.actuator_gainprm, wp.array(gainprm, dtype=m.actuator_gainprm.dtype))

    dynprm = np.zeros((1, 1, 10), dtype=np.float32)
    dynprm[0, 0, 0] = 0.1  # te
    wp.copy(m.actuator_dynprm, wp.array(dynprm, dtype=m.actuator_dynprm.dtype))

    # Verify derivative instead of looping
    d.act_dot.fill_(wp.inf)
    mjw.forward(m, d)

    # Analytical derivative: (I_ss - I) / te = (10.0 - 8.646647) / 0.1 = 13.53353
    expected_dot = (10.0 - 8.646647) / 0.1

    act_dot = d.act_dot.numpy()[0, 0]
    np.testing.assert_allclose(act_dot, expected_dot, atol=1e-2)

  def test_dcmotor_cogging_torque(self):
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint" type="hinge" axis="0 0 1"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <general name="dcmotor" joint="joint" actdim="0"/>
      </actuator>
      <keyframe>
        <key ctrl="5.0" qpos="1.0"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)

    # Override types in Warp model
    wp.copy(m.actuator_gaintype, wp.array([int(GainType.DCMOTOR)], dtype=m.actuator_gaintype.dtype))
    wp.copy(m.actuator_biastype, wp.array([int(BiasType.DCMOTOR)], dtype=m.actuator_biastype.dtype))
    wp.copy(m.actuator_dyntype, wp.array([int(DynType.NONE)], dtype=m.actuator_dyntype.dtype))

    # Set parameters
    gainprm = np.zeros((1, 1, 10), dtype=np.float32)
    gainprm[0, 0, 0] = 2.0  # R
    gainprm[0, 0, 1] = 0.05  # K
    wp.copy(m.actuator_gainprm, wp.array(gainprm, dtype=m.actuator_gainprm.dtype))

    biasprm = np.zeros((1, 1, 10), dtype=np.float32)
    biasprm[0, 0, 0] = 0.1  # A
    biasprm[0, 0, 1] = 6.0  # Np
    biasprm[0, 0, 2] = 0.0  # phi
    wp.copy(m.actuator_biasprm, wp.array(biasprm, dtype=m.actuator_biasprm.dtype))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    # Analytical expected force: K/R * V + A * sin(Np * pos + phi)
    electrical_force = 0.05 / 2.0 * 5.0
    cogging = 0.1 * np.sin(6.0 * 1.0 + 0.0)
    expected_force = electrical_force + cogging

    force = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force, expected_force, atol=1e-5)

  def test_dcmotor_lugre_viscous_friction(self):
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint" type="slide" axis="1 0 0"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <general name="dcmotor" joint="joint" dyntype="integrator" actdim="1"/>
      </actuator>
      <keyframe>
        <key ctrl="0.0" qvel="2.0"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)

    # Override types in Warp model
    wp.copy(m.actuator_gaintype, wp.array([int(GainType.DCMOTOR)], dtype=m.actuator_gaintype.dtype))
    wp.copy(m.actuator_biastype, wp.array([int(BiasType.DCMOTOR)], dtype=m.actuator_biastype.dtype))
    wp.copy(m.actuator_dyntype, wp.array([int(DynType.DCMOTOR)], dtype=m.actuator_dyntype.dtype))

    # Set parameters
    gainprm = np.zeros((1, 1, 10), dtype=np.float32)
    gainprm[0, 0, 0] = 2.0  # R
    gainprm[0, 0, 1] = 0.05  # K
    wp.copy(m.actuator_gainprm, wp.array(gainprm, dtype=m.actuator_gainprm.dtype))

    dynprm = np.zeros((1, 1, 10), dtype=np.float32)
    dynprm[0, 0, 5] = 100.0  # sigma0
    dynprm[0, 0, 6] = 1.0  # sigma1
    wp.copy(m.actuator_dynprm, wp.array(dynprm, dtype=m.actuator_dynprm.dtype))

    biasprm = np.zeros((1, 1, 10), dtype=np.float32)
    biasprm[0, 0, 3] = 0.5  # coulomb
    biasprm[0, 0, 4] = 0.7  # static
    biasprm[0, 0, 5] = 10.0  # stribeck
    wp.copy(m.actuator_biasprm, wp.array(biasprm, dtype=m.actuator_biasprm.dtype))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    # Analytical expected force:
    # z = 0 initially
    # z_dot = velocity = 2.0
    # lugre_force = sigma0 * z + sigma1 * z_dot = 100 * 0 + 1.0 * 2.0 = 2.0
    # electrical_force = K / R * (ctrl - K * velocity) = 0.05 / 2.0 * (0 - 0.05 * 2.0) = -0.0025
    # expected_force = electrical_force - lugre_force = -2.0025

    expected_force = -2.0025
    force = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force, expected_force, atol=1e-5)

  def test_dcmotor_thermal_affects_force(self):
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint" type="slide" axis="1 0 0"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <general name="dcmotor" joint="joint" dyntype="integrator" actdim="1"/>
      </actuator>
      <keyframe>
        <key ctrl="10.0"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)

    # Override types in Warp model
    wp.copy(m.actuator_gaintype, wp.array([int(GainType.DCMOTOR)], dtype=m.actuator_gaintype.dtype))
    wp.copy(m.actuator_biastype, wp.array([int(BiasType.DCMOTOR)], dtype=m.actuator_biastype.dtype))
    wp.copy(m.actuator_dyntype, wp.array([int(DynType.DCMOTOR)], dtype=m.actuator_dyntype.dtype))

    # Set parameters
    gainprm = np.zeros((1, 1, 10), dtype=np.float32)
    gainprm[0, 0, 0] = 2.0  # R
    gainprm[0, 0, 1] = 0.05  # K
    gainprm[0, 0, 2] = 0.004  # alpha
    gainprm[0, 0, 3] = 25.0  # T0
    wp.copy(m.actuator_gainprm, wp.array(gainprm, dtype=m.actuator_gainprm.dtype))

    dynprm = np.zeros((1, 1, 10), dtype=np.float32)
    dynprm[0, 0, 2] = 1.0  # tau_th > 0 to enable temperature state!
    dynprm[0, 0, 4] = 25.0  # Ta
    wp.copy(m.actuator_dynprm, wp.array(dynprm, dtype=m.actuator_dynprm.dtype))

    # Cold state (T = 0)
    act_numpy = np.zeros((1, 1), dtype=np.float32)
    act_numpy[0, 0] = 0.0
    wp.copy(d.act, wp.array(act_numpy, device=d.act.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)
    force_cold = d.actuator_force.numpy()[0, 0]

    # Expected cold force: K/R * V = 0.05 / 2.0 * 10.0 = 0.25
    np.testing.assert_allclose(force_cold, 0.25, atol=1e-5)

    # Hot state (T = 50)
    act_numpy[0, 0] = 50.0
    wp.copy(d.act, wp.array(act_numpy, device=d.act.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)
    force_hot = d.actuator_force.numpy()[0, 0]

    # Expected hot force: K / (R * (1 + alpha * dT)) * V
    # dT = T + Ta - T0 = 50 + 25 - 25 = 50
    # R_hot = 2.0 * (1 + 0.004 * 50) = 2.0 * 1.2 = 2.4
    # force_hot = 0.05 / 2.4 * 10.0 = 0.5 / 2.4 = 0.208333

    expected_hot_force = 0.5 / 2.4
    np.testing.assert_allclose(force_hot, expected_hot_force, atol=1e-5)
    self.assertLess(force_hot, force_cold)

  def test_dcmotor_stateful_position_mode(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint" type="slide" axis="1 0 0"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <general name="dcmotor" joint="joint" dyntype="user" actdim="2"/>
      </actuator>
      <keyframe>
        <key ctrl="5.0" qvel="0.5"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)

    # Override types in Warp model
    wp.copy(m.actuator_gaintype, wp.array([int(GainType.DCMOTOR)], dtype=m.actuator_gaintype.dtype))
    wp.copy(m.actuator_biastype, wp.array([int(BiasType.DCMOTOR)], dtype=m.actuator_biastype.dtype))
    wp.copy(m.actuator_dyntype, wp.array([int(DynType.DCMOTOR)], dtype=m.actuator_dyntype.dtype))

    # Set parameters
    gainprm = np.zeros((1, 1, 10), dtype=np.float32)
    gainprm[0, 0, 0] = 2.0  # R
    gainprm[0, 0, 1] = 0.05  # K
    gainprm[0, 0, 4] = 2.0  # kp
    gainprm[0, 0, 5] = 0.5  # ki
    gainprm[0, 0, 6] = 0.1  # kv (kd)
    gainprm[0, 0, 7] = 10.0  # vmax
    gainprm[0, 0, 8] = 1.0  # input_mode = position
    wp.copy(m.actuator_gainprm, wp.array(gainprm, dtype=m.actuator_gainprm.dtype))

    dynprm = np.zeros((1, 1, 10), dtype=np.float32)
    dynprm[0, 0, 7] = 10.0  # slew rate
    dynprm[0, 0, 8] = 5.0  # Imax
    wp.copy(m.actuator_dynprm, wp.array(dynprm, dtype=m.actuator_dynprm.dtype))

    # Set initial state for actuators
    # adr=0: slew (u_prev) = 1.0
    # adr=1: ki (x_I) = 2.0
    act_numpy = np.zeros((1, 2), dtype=np.float32)
    act_numpy[0, 0] = 1.0
    act_numpy[0, 1] = 2.0
    wp.copy(d.act, wp.array(act_numpy, device=d.act.device))

    d.act_dot.fill_(wp.inf)
    mjw.forward(m, d)

    # Verify act_dot
    # slew state dot: (1.01 - 1.0) / 0.001 = 10.0
    # integral state dot: u_eff = 1.01
    act_dot = d.act_dot.numpy()[0]
    np.testing.assert_allclose(act_dot[0], 10.0, atol=1e-5)
    np.testing.assert_allclose(act_dot[1], 1.01, atol=1e-5)

    # Verify actuator_force
    # V = 2.0 * 1.01 + 0.5 * 2.0 - 0.1 * 0.5 = 2.97
    # force = K/R * V - K^2/R * omega = 0.025 * 2.97 - 0.000625 = 0.073625
    force = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force, 0.073625, atol=1e-5)

  def test_dcmotor_lugre_exact_integration(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint" type="slide" axis="1 0 0"/>
          <geom size="1" mass="1e6"/>
        </body>
      </worldbody>
      <actuator>
        <general name="dcmotor" joint="joint" dyntype="user" actdim="1"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # Override types in Warp model
    wp.copy(m.actuator_gaintype, wp.array([int(GainType.DCMOTOR)], dtype=m.actuator_gaintype.dtype))
    wp.copy(m.actuator_biastype, wp.array([int(BiasType.DCMOTOR)], dtype=m.actuator_biastype.dtype))
    wp.copy(m.actuator_dyntype, wp.array([int(DynType.DCMOTOR)], dtype=m.actuator_dyntype.dtype))

    # Set parameters
    gainprm = np.zeros((1, 1, 10), dtype=np.float32)
    gainprm[0, 0, 0] = 2.0  # R
    gainprm[0, 0, 1] = 0.05  # K
    wp.copy(m.actuator_gainprm, wp.array(gainprm, dtype=m.actuator_gainprm.dtype))

    dynprm = np.zeros((1, 1, 10), dtype=np.float32)
    dynprm[0, 0, 5] = 100.0  # sigma0
    dynprm[0, 0, 6] = 1.0  # sigma1
    wp.copy(m.actuator_dynprm, wp.array(dynprm, dtype=m.actuator_dynprm.dtype))

    biasprm = np.zeros((1, 1, 10), dtype=np.float32)
    biasprm[0, 0, 3] = 0.5  # coulomb
    biasprm[0, 0, 4] = 0.7  # static
    biasprm[0, 0, 5] = 10.0  # stribeck
    wp.copy(m.actuator_biasprm, wp.array(biasprm, dtype=m.actuator_biasprm.dtype))

    # Set initial state
    z0 = 0.002
    v = 0.5
    act_numpy = np.zeros((1, 1), dtype=np.float32)
    act_numpy[0, 0] = z0
    wp.copy(d.act, wp.array(act_numpy, device=d.act.device))

    qvel_np = np.zeros((1, m.nv), dtype=np.float32)
    qvel_np[0, 0] = v
    wp.copy(d.qvel, wp.array(qvel_np, device=d.qvel.device))

    # Analytical expected z_new:
    sigma0 = 100.0
    F_C = 0.5
    F_S = 0.7
    v_S = 10.0
    h = 0.001

    ratio = v / v_S
    g_v = F_C + (F_S - F_C) * np.exp(-ratio * ratio)
    a = -sigma0 * np.abs(v) / g_v
    exp_ah = np.exp(a * h)
    int_h = (exp_ah - 1.0) / a
    z_new = exp_ah * z0 + int_h * v

    mjw.step(m, d)

    np.testing.assert_allclose(d.act.numpy()[0, 0], z_new, atol=1e-5)

  def test_dcmotor_current_filter_exact_integration(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint" type="slide" axis="1 0 0"/>
          <geom size="1" mass="10000"/>
        </body>
      </worldbody>
      <actuator>
        <general name="dcmotor" joint="joint" dyntype="user" actdim="1"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # Override types in Warp model
    wp.copy(m.actuator_gaintype, wp.array([int(GainType.DCMOTOR)], dtype=m.actuator_gaintype.dtype))
    wp.copy(m.actuator_biastype, wp.array([int(BiasType.DCMOTOR)], dtype=m.actuator_biastype.dtype))
    wp.copy(m.actuator_dyntype, wp.array([int(DynType.DCMOTOR)], dtype=m.actuator_dyntype.dtype))

    # Set parameters
    gainprm = np.zeros((1, 1, 10), dtype=np.float32)
    gainprm[0, 0, 0] = 2.0  # R
    gainprm[0, 0, 1] = 0.05  # K
    wp.copy(m.actuator_gainprm, wp.array(gainprm, dtype=m.actuator_gainprm.dtype))

    dynprm = np.zeros((1, 1, 10), dtype=np.float32)
    dynprm[0, 0, 0] = 0.005  # te (L/R)
    wp.copy(m.actuator_dynprm, wp.array(dynprm, dtype=m.actuator_dynprm.dtype))

    # Set control
    V = 12.0
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = V
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    mjw.step(m, d)

    R = 2.0
    te = 0.01 / R
    h = 0.001
    exact_current = V / R * (1.0 - np.exp(-h / te))
    euler_current = V / R * h / te

    current_actual = d.act.numpy()[0, 0]
    np.testing.assert_allclose(current_actual, exact_current, atol=1e-5)

    # Verify it is better than Euler
    self.assertLess(np.abs(current_actual - exact_current), np.abs(current_actual - euler_current))

  def test_dcmotor_stateful_position_with_current_mode(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint" type="slide" axis="1 0 0"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <general name="dcmotor" joint="joint" dyntype="user" actdim="3" actearly="true"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # Override types in Warp model
    wp.copy(m.actuator_gaintype, wp.array([int(GainType.DCMOTOR)], dtype=m.actuator_gaintype.dtype))
    wp.copy(m.actuator_biastype, wp.array([int(BiasType.DCMOTOR)], dtype=m.actuator_biastype.dtype))
    wp.copy(m.actuator_dyntype, wp.array([int(DynType.DCMOTOR)], dtype=m.actuator_dyntype.dtype))

    # Set parameters
    gainprm = np.zeros((1, 1, 10), dtype=np.float32)
    gainprm[0, 0, 0] = 2.0  # R
    gainprm[0, 0, 1] = 0.05  # K
    gainprm[0, 0, 4] = 2.0  # kp
    gainprm[0, 0, 5] = 0.5  # ki
    gainprm[0, 0, 6] = 0.1  # kv (kd)
    gainprm[0, 0, 7] = 10.0  # vmax
    gainprm[0, 0, 8] = 1.0  # input_mode = position
    wp.copy(m.actuator_gainprm, wp.array(gainprm, dtype=m.actuator_gainprm.dtype))

    dynprm = np.zeros((1, 1, 10), dtype=np.float32)
    dynprm[0, 0, 0] = 0.5  # te (L/R)
    dynprm[0, 0, 7] = 10.0  # slew rate
    dynprm[0, 0, 8] = 5.0  # Imax
    wp.copy(m.actuator_dynprm, wp.array(dynprm, dtype=m.actuator_dynprm.dtype))

    # Set initial state
    # Controller states: slew (0), ki (1), current (2)
    act_numpy = np.zeros((1, 3), dtype=np.float32)
    act_numpy[0, 0] = 1.0  # u_prev
    act_numpy[0, 1] = 2.0  # x_I
    act_numpy[0, 2] = 0.5  # current
    wp.copy(d.act, wp.array(act_numpy, device=d.act.device))

    # Target 5.0 position, velocity 0.5
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = 5.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    qvel_np = np.zeros((1, m.nv), dtype=np.float32)
    qvel_np[0, 0] = 0.5
    wp.copy(d.qvel, wp.array(qvel_np, device=d.qvel.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    act_dot_actual = d.act_dot.numpy()[0]

    np.testing.assert_allclose(act_dot_actual[0], 10.0, atol=1e-3)
    np.testing.assert_allclose(act_dot_actual[1], 1.01, atol=1e-3)
    np.testing.assert_allclose(act_dot_actual[2], 1.945, atol=1e-3)

    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, 0.025097, atol=1e-5)

  def test_dcmotor_int_velocity_equivalence(self):
    xml = """
    <mujoco>
      <worldbody>
        <body pos="0 0 0">
          <joint name="slide1" type="slide" axis="1 0 0"/>
          <geom size=".1"/>
        </body>
        <body pos="0 1 0">
          <joint name="slide2" type="slide" axis="1 0 0"/>
          <geom size=".1"/>
        </body>
      </worldbody>
      <actuator>
        <intvelocity name="intvel" joint="slide1" kp="10" kv="5" actrange="-0.01 0.01"/>
        <dcmotor name="dcmotor" joint="slide2" motorconst="1" resistance="0.2" input="velocity" controller="0 2 0 0 0.01"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # Apply a time-varying velocity command
    for _ in range(10):
      t = d.time.numpy()[0]
      ctrl_val = np.sin(20.0 * t)

      ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
      ctrl_np[0, 0] = ctrl_val
      ctrl_np[0, 1] = ctrl_val
      wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

      mjw.step(m, d)

    # Both actuators should integrate identical states
    np.testing.assert_allclose(d.act.numpy()[0, 0], d.act.numpy()[0, 1], atol=1e-5)

    # Both bodies should move identically
    np.testing.assert_allclose(d.qpos.numpy()[0, 0], d.qpos.numpy()[0, 1], atol=1e-5)
    np.testing.assert_allclose(d.qvel.numpy()[0, 0], d.qvel.numpy()[0, 1], atol=1e-5)
    np.testing.assert_allclose(d.qacc.numpy()[0, 0], d.qacc.numpy()[0, 1], atol=1e-4)

    # Both actuators should produce identical force
    np.testing.assert_allclose(d.actuator_force.numpy()[0, 0], d.actuator_force.numpy()[0, 1], atol=1e-4)

  def test_dcmotor_cogging_bypasses_saturation(self):
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <general name="dcmotor" joint="joint" dyntype="user" actdim="1"
                 forcerange="0 0.001" forcelimited="true"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # Override types in Warp model
    wp.copy(m.actuator_gaintype, wp.array([int(GainType.DCMOTOR)], dtype=m.actuator_gaintype.dtype))
    wp.copy(m.actuator_biastype, wp.array([int(BiasType.DCMOTOR)], dtype=m.actuator_biastype.dtype))
    wp.copy(m.actuator_dyntype, wp.array([int(DynType.DCMOTOR)], dtype=m.actuator_dyntype.dtype))

    # Set parameters
    dynprm = np.zeros((1, 1, 10), dtype=np.float32)
    wp.copy(m.actuator_dynprm, wp.array(dynprm, dtype=m.actuator_dynprm.dtype))

    gainprm = np.zeros((1, 1, 10), dtype=np.float32)
    gainprm[0, 0, 0] = 2.0  # R
    gainprm[0, 0, 1] = 0.05  # K
    wp.copy(m.actuator_gainprm, wp.array(gainprm, dtype=m.actuator_gainprm.dtype))

    biasprm = np.zeros((1, 1, 10), dtype=np.float32)
    biasprm[0, 0, 0] = 0.1  # A
    biasprm[0, 0, 1] = 6.0  # Np
    biasprm[0, 0, 2] = 0.0  # phi
    wp.copy(m.actuator_biasprm, wp.array(biasprm, dtype=m.actuator_biasprm.dtype))

    # Set position
    pos = 1.0
    qpos_np = np.zeros((1, m.nq), dtype=np.float32)
    qpos_np[0, 0] = pos
    wp.copy(d.qpos, wp.array(qpos_np, device=d.qpos.device))

    # Large control to trigger saturation
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = 100.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    A = 0.1
    Np = 6.0
    phi = 0.0
    cogging = A * np.sin(Np * pos + phi)

    force_actual = d.actuator_force.numpy()[0, 0]

    # Expect 0.001 (saturated) + cogging
    np.testing.assert_allclose(force_actual, 0.001 + cogging, atol=1e-5)

  def test_dcmotor_thermal_rise_and_fall(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint" damping="10000"/>
          <geom size="1" mass="10000"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
                 thermal="10 5 0 0 25 25"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    adr = m.actuator_actadr.numpy()[0]

    # Apply voltage
    V = 10.0
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = V
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    # Step 1
    mjw.step(m, d)

    R = 2.0
    P = V * V / R
    C = 5.0
    h = 0.001
    dT1 = h * P / C

    act_actual = d.act.numpy()[0, adr]
    np.testing.assert_allclose(act_actual, dT1, atol=1e-4)

    # Step 2
    mjw.step(m, d)
    RT = 10.0
    dT2 = dT1 + h * (P - dT1 / RT) / C

    act_actual = d.act.numpy()[0, adr]
    np.testing.assert_allclose(act_actual, dT2, atol=1e-4)

    # Step 3 (fall)
    ctrl_np[0, 0] = 0.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    mjw.step(m, d)
    dT3 = dT2 + h * (0.0 - dT2 / RT) / C

    act_actual = d.act.numpy()[0, adr]
    np.testing.assert_allclose(act_actual, dT3, atol=1e-4)

  def test_dcmotor_thermal_steady_state(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint" damping="10000"/>
          <geom size="1" mass="10000"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
                 thermal="0.1 0.1 0 0 25 25"/>
      </actuator>
      <keyframe>
        <key ctrl="10.0" act="5.0"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)

    # Verify derivative is zero instead of looping
    d.act_dot.fill_(wp.inf)
    mjw.forward(m, d)

    adr = m.actuator_actadr.numpy()[0]
    act_dot = d.act_dot.numpy()[0, adr]

    # Expect act_dot to be close to zero
    np.testing.assert_allclose(act_dot, 0.0, atol=1e-4)

  def test_dcmotor_thermal_affects_force_with_controller(self):
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
                 input="position" controller="1.0 1.0 0 5.0 0"
                 thermal="0.1 0.1 0 0.004 25 25"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    adr = m.actuator_actadr.numpy()[0]

    # Set states
    act_np = np.zeros((1, m.na), dtype=np.float32)
    act_np[0, adr] = 1.0  # slew state = ctrl: no rate-limiting applied
    act_np[0, adr + 1] = 0.0  # integral state x_I = 0
    act_np[0, adr + 2] = 50.0  # temperature rise above ambient
    wp.copy(d.act, wp.array(act_np, device=d.act.device))

    # Set control
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = 1.0  # position setpoint = 1.0, qpos = 0, error = 1.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    K = 0.05
    R = 2.0
    alpha = 0.004
    dT = 50.0

    R_hot = R * (1.0 + alpha * dT)
    force_expected = K / R_hot * 1.0  # V = 1.0

    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, force_expected, atol=1e-5)

  def test_dcmotor_stateless_position_mode(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" input="position" controller="2.0 0 0.5 0 0"
                 motorconst="0.05" resistance="2.0"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # Position target 5.0, current pos 0.0, current vel 0.0
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = 5.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    # V = Kp * (u - theta) = 2.0 * 5.0 = 10.0
    # force = K / R * V + bias = (0.05 / 2.0) * 10.0 + 0 = 0.25
    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, 0.25, atol=1e-5)

    # Velocity penalty
    qvel_np = np.zeros((1, m.nv), dtype=np.float32)
    qvel_np[0, 0] = 2.0
    wp.copy(d.qvel, wp.array(qvel_np, device=d.qvel.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    # V = 10.0 - Kd * omega = 10.0 - (0.5 * 2.0) = 9.0
    # bias = - K^2 / R * omega = -0.0025 / 2.0 * 2.0 = -0.0025
    # force = K / R * V + bias = 0.225 - 0.0025 = 0.2225
    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, 0.2225, atol=1e-5)

  def test_dcmotor_stateless_velocity_mode(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" input="velocity" controller="3.0 0 0 0 0"
                 motorconst="0.05" resistance="2.0"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # Velocity target 4.0, current vel 1.0
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = 4.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    qvel_np = np.zeros((1, m.nv), dtype=np.float32)
    qvel_np[0, 0] = 1.0
    wp.copy(d.qvel, wp.array(qvel_np, device=d.qvel.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    # V = Kp * (u - omega) = 3.0 * (4.0 - 1.0) = 9.0
    # bias = - K^2 / R * omega = -0.0025 / 2.0 * 1.0 = -0.00125
    # force = K / R * V + bias = (0.05 / 2.0) * 9.0 - 0.00125 = 0.22375
    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, 0.22375, atol=1e-5)

  def test_dcmotor_stateful_velocity_mode(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" input="velocity" controller="3.0 1.0 0 0 2.0"
                 motorconst="0.05" resistance="2.0"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    adr = m.actuator_actadr.numpy()[0]

    # Set state: x_I = 2.0
    act_np = np.zeros((1, m.na), dtype=np.float32)
    act_np[0, adr] = 2.0
    wp.copy(d.act, wp.array(act_np, device=d.act.device))

    # target vel 4.0, current vel 1.0
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = 4.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    qvel_np = np.zeros((1, m.nv), dtype=np.float32)
    qvel_np[0, 0] = 1.0
    wp.copy(d.qvel, wp.array(qvel_np, device=d.qvel.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    # V = Kp * (u_eff - omega) + Ki * (x_I - length)
    # V = 3.0 * (4.0 - 1.0) + 1.0 * (2.0 - 0.0) = 9.0 + 2.0 = 11.0
    # bias = - K^2/R * omega = -(0.05)^2 / 2.0 * 1.0 = -0.00125
    # force = K/R * V + bias = 0.025 * 11.0 - 0.00125 = 0.275 - 0.00125 = 0.27375
    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, 0.27375, atol=1e-5)

    # repeat with non-zero joint position
    qpos_np = np.zeros((1, m.nq), dtype=np.float32)
    qpos_np[0, 0] = 1.5
    wp.copy(d.qpos, wp.array(qpos_np, device=d.qpos.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    # V = 3.0 * (4.0 - 1.0) + 1.0 * (2.0 - 1.5) = 9.0 + 0.5 = 9.5
    # force = K/R * V + bias = 0.025 * 9.5 - 0.00125 = 0.2375 - 0.00125 = 0.23625
    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, 0.23625, atol=1e-5)

  def test_dcmotor_current_plus_thermal(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint" damping="10000"/>
          <geom size="1" mass="10000"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
                 inductance="0.01 0" thermal="10 5 0 0.004 25 25"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    adr = m.actuator_actadr.numpy()[0]

    # Set states
    dT = 10.0
    current = 3.0
    act_np = np.zeros((1, m.na), dtype=np.float32)
    act_np[0, adr] = dT
    act_np[0, adr + 1] = current
    wp.copy(d.act, wp.array(act_np, device=d.act.device))

    # Set control
    V = 12.0
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = V
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    d.act_dot.fill_(wp.inf)
    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    K = 0.05
    R = 2.0
    te = 0.01 / R
    RT = 10.0
    C = 5.0
    h = 0.001

    R_hot = R * (1.0 + 0.004 * dT)

    # Verify act_dot for temperature
    T_dot_expected = (R_hot * current * current - dT / RT) / C
    act_dot_actual = d.act_dot.numpy()[0, adr]
    np.testing.assert_allclose(act_dot_actual, T_dot_expected, atol=1e-3)

    # Verify act_dot for current
    omega = d.qvel.numpy()[0, 0]
    i_dot_expected = (V / R_hot - K / R_hot * omega - current) / te
    act_dot_actual = d.act_dot.numpy()[0, adr + 1]
    np.testing.assert_allclose(act_dot_actual, i_dot_expected, atol=1e-2)

    # Verify force
    act_dot_from_data = d.act_dot.numpy()[0, adr + 1]
    next_i = current + act_dot_from_data * te * (1.0 - np.exp(-h / te))
    force_expected = K * next_i

    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, force_expected, atol=1e-4)

  def test_dcmotor_current_rate_limit(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint" damping="10000"/>
          <geom size="1" mass="10000"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
                 inductance="0.01 0" saturation="0 0 100"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    adr = m.actuator_actadr.numpy()[0]

    # Set state: current = 0
    act_np = np.zeros((1, m.na), dtype=np.float32)
    act_np[0, adr] = 0.0
    wp.copy(d.act, wp.array(act_np, device=d.act.device))

    # Set control
    V = 12.0
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = V
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    d.act_dot.fill_(wp.inf)
    mjw.forward(m, d)

    dimax = 100.0
    act_dot_actual = d.act_dot.numpy()[0, adr]
    np.testing.assert_allclose(act_dot_actual, dimax, atol=1e-5)

    # reverse: large negative drive
    ctrl_np[0, 0] = -V
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    d.act_dot.fill_(wp.inf)
    mjw.forward(m, d)

    act_dot_actual = d.act_dot.numpy()[0, adr]
    np.testing.assert_allclose(act_dot_actual, -dimax, atol=1e-5)

  def test_dcmotor_voltage_limit(self):
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
                 input="position" controller="1 0 0 0 0 10.0"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    # Vmax = 10.0, ctrl = 20.0
    # force = K/R * Vmax = 0.05 / 2.0 * 10.0 = 0.25
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = 20.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    mjw.forward(m, d)

    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, 0.25, atol=1e-5)

    # negative drive
    ctrl_np[0, 0] = -20.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, -0.25, atol=1e-5)

  def test_dcmotor_integral_clamp(self):
    xml = """
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" input="position" controller="2.0 0.5 0 0 5.0"
                 motorconst="0.05" resistance="2.0"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    adr = m.actuator_actadr.numpy()[0]

    # set integral state to Imax
    act_np = np.zeros((1, m.na), dtype=np.float32)
    act_np[0, adr] = 5.0
    wp.copy(d.act, wp.array(act_np, device=d.act.device))

    # set target to generate positive error (ctrl - length)
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    ctrl_np[0, 0] = 1.0  # target
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    # ensure length = 0
    qpos_np = np.zeros((1, m.nq), dtype=np.float32)
    wp.copy(d.qpos, wp.array(qpos_np, device=d.qpos.device))

    d.act_dot.fill_(wp.inf)
    mjw.forward(m, d)

    # act_dot should be clamped to 0 because act >= Imax and error > 0
    act_dot_actual = d.act_dot.numpy()[0, adr]
    np.testing.assert_allclose(act_dot_actual, 0.0, atol=1e-5)

    # set target to generate negative error
    ctrl_np[0, 0] = -1.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    d.act_dot.fill_(wp.inf)
    mjw.forward(m, d)

    # act_dot should be negative (not clamped)
    act_dot_actual = d.act_dot.numpy()[0, adr]
    np.testing.assert_allclose(act_dot_actual, -1.0, atol=1e-5)

    # set integral state to -Imax
    act_np[0, adr] = -5.0
    wp.copy(d.act, wp.array(act_np, device=d.act.device))

    # set target to generate negative error
    ctrl_np[0, 0] = -1.0
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    d.act_dot.fill_(wp.inf)
    mjw.forward(m, d)

    # act_dot should be clamped to 0 because act <= -Imax and error < 0
    act_dot_actual = d.act_dot.numpy()[0, adr]
    np.testing.assert_allclose(act_dot_actual, 0.0, atol=1e-5)

  def test_dcmotor_lugre_steady_state(self):
    v_S = 10.0
    F_C = 0.5
    F_S = 0.7
    sigma0 = 100.0
    v = 0.5
    ratio = v / v_S
    g_v = F_C + (F_S - F_C) * np.exp(-ratio * ratio)
    z_ss = g_v / sigma0

    xml = f"""
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <body>
          <joint name="joint"/>
          <geom size="1" mass="1e6"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
                 damping="0.01" lugre="100 1 0.5 0.7 10"/>
      </actuator>
      <keyframe>
        <key qpos="0" qvel="{v}" act="{z_ss}"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)

    adr = m.actuator_actadr.numpy()[0]

    # Verify derivative and force instead of looping
    d.act_dot.fill_(wp.inf)
    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    # Verify steady state bristle deflection (loaded from keyframe)
    act_actual = d.act.numpy()[0, adr]
    np.testing.assert_allclose(act_actual, z_ss, atol=1e-4)

    # Verify derivative is zero
    act_dot_actual = d.act_dot.numpy()[0, adr]
    np.testing.assert_allclose(act_dot_actual, 0.0, atol=1e-4)

    # Verify force
    K = 0.05
    R = 2.0
    back_emf = K * K / R * v
    lugre_ss = g_v
    force_expected = -back_emf - lugre_ss

    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, force_expected, atol=1e-3)

  def test_dcmotor_lugre_bristle_spring(self):
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint name="joint"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>
        <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
                 damping="0.01" lugre="100 1 0.5 0.7 10"/>
      </actuator>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    adr = m.actuator_actadr.numpy()[0]

    # Set state: deflection = 0.01
    X = 0.01
    act_np = np.zeros((1, m.na), dtype=np.float32)
    act_np[0, adr] = X
    wp.copy(d.act, wp.array(act_np, device=d.act.device))

    # Set control = 0
    ctrl_np = np.zeros((1, m.nu), dtype=np.float32)
    wp.copy(d.ctrl, wp.array(ctrl_np, device=d.ctrl.device))

    d.actuator_force.fill_(wp.inf)
    mjw.forward(m, d)

    sigma0 = 100.0
    force_expected = -sigma0 * X

    force_actual = d.actuator_force.numpy()[0, 0]
    np.testing.assert_allclose(force_actual, force_expected, atol=1e-5)


if __name__ == "__main__":
  wp.init()
  absltest.main()
