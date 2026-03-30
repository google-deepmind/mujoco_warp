"""Tests for autodifferentiation gradients."""

# When run as a script, Python adds this file's directory (_src/) to sys.path,
# which causes types.py to shadow the stdlib 'types' module.  Replace it with
# the project root so that 'import mujoco_warp' still works.
import os as _os
import sys as _sys

_src_dir = _os.path.dirname(_os.path.abspath(__file__))
_project_root = _os.path.dirname(_os.path.dirname(_src_dir))
if _src_dir in _sys.path:
  _sys.path[_sys.path.index(_src_dir)] = _project_root

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import math
from mujoco_warp._src.grad import enable_grad

# tolerance for AD vs finite-difference comparison
_FD_TOL = 1e-3

# sparse jacobian to avoid tile kernels (which require cuSolverDx)
_SIMPLE_HINGE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="1"/>
    <motor joint="j1" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0.5 -0.3" qvel="0.1 -0.2" ctrl="0.5 -0.5"/>
  </keyframe>
</mujoco>
"""

_SIMPLE_SLIDE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body>
      <joint name="j0" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0.2" qvel="0.1" ctrl="0.5"/>
  </keyframe>
</mujoco>
"""

# 3-link chain with mixed joint axes for non-trivial Coriolis gradient.
# planar 2-link same-axis models have mathematically zero d(qfrc_bias)/d(qvel).
_3LINK_HINGE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="1 0 0"/>
        <geom type="sphere" size="0.1" mass="2"/>
        <body pos="0.3 0 -0.3">
          <joint name="j2" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j0" gear="1"/>
    <motor joint="j1" gear="1"/>
    <motor joint="j2" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0.5 -0.3 0.2" qvel="2.0 -1.0 3.0" ctrl="0.5 -0.5 0.3"/>
  </keyframe>
</mujoco>
"""

_SIMPLE_FREE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body pos="0 0 1">
      <joint name="j0" type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <keyframe>
    <key qpos="0 0 1 1 0 0 0" qvel="0.1 0 0 0 0.1 0"/>
  </keyframe>
</mujoco>
"""


def _fd_gradient(fn, x_np, eps=1e-3):
  """Central-difference gradient of scalar fn w.r.t. x_np."""
  grad = np.zeros_like(x_np)
  for i in range(x_np.size):
    x_plus = x_np.copy()
    x_minus = x_np.copy()
    x_plus.flat[i] += eps
    x_minus.flat[i] -= eps
    grad.flat[i] = (fn(x_plus) - fn(x_minus)) / (2.0 * eps)
  return grad


@wp.kernel
def _sum_xpos_kernel(
  # Data in:
  xpos_in: wp.array2d(dtype=wp.vec3),
  # In:
  loss: wp.array(dtype=float),
):
  worldid, bodyid = wp.tid()
  v = xpos_in[worldid, bodyid]
  wp.atomic_add(loss, 0, v[0] + v[1] + v[2])


@wp.kernel
def _sum_qacc_kernel(
  # Data in:
  qacc_in: wp.array2d(dtype=float),
  # In:
  loss: wp.array(dtype=float),
):
  worldid, dofid = wp.tid()
  wp.atomic_add(loss, 0, qacc_in[worldid, dofid])


class GradSmoothTest(parameterized.TestCase):
  @parameterized.parameters(
    ("hinge", _SIMPLE_HINGE_XML),
    ("slide", _SIMPLE_SLIDE_XML),
  )
  def test_kinematics_grad(self, name, xml):
    """dL/dqpos through kinematics(): loss = sum(xpos)."""
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)
    enable_grad(d)

    # AD gradient
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.kinematics(m, d)
      mjw.com_pos(m, d)
      wp.launch(
        _sum_xpos_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[d.xpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.qpos.grad.numpy()[0, : mjm.nq].copy()
    tape.zero()

    # Finite-difference gradient
    def eval_loss(qpos_np):
      d_fd = mjw.make_data(mjm)
      d_fd.qpos = wp.array(qpos_np.reshape(1, -1), dtype=float)
      mjw.kinematics(m, d_fd)
      mjw.com_pos(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        _sum_xpos_kernel,
        dim=(d_fd.nworld, m.nbody),
        inputs=[d_fd.xpos, l],
      )
      return l.numpy()[0]

    qpos_np = d.qpos.numpy()[0, : mjm.nq]
    fd_grad = _fd_gradient(eval_loss, qpos_np)

    np.testing.assert_allclose(
      ad_grad,
      fd_grad,
      atol=_FD_TOL,
      rtol=_FD_TOL,
      err_msg=f"kinematics grad mismatch ({name})",
    )

  @parameterized.parameters(
    ("3link_hinge", _3LINK_HINGE_XML),
    ("slide", _SIMPLE_SLIDE_XML),
  )
  def test_fwd_velocity_grad(self, name, xml):
    """dL/dqvel through fwd_velocity()."""
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)
    enable_grad(d)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.kinematics(m, d)
      mjw.com_pos(m, d)
      mjw.crb(m, d)
      mjw.factor_m(m, d)
      mjw.transmission(m, d)
      mjw.fwd_velocity(m, d)
      wp.launch(
        _sum_qacc_kernel,
        dim=(d.nworld, m.nv),
        inputs=[d.qfrc_bias, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.qvel.grad.numpy()[0, : mjm.nv].copy()
    tape.zero()

    def eval_loss(qvel_np):
      d_fd = mjw.make_data(mjm)
      # Copy qpos from original
      wp.copy(d_fd.qpos, d.qpos)
      d_fd.qvel = wp.array(qvel_np.reshape(1, -1), dtype=float)
      mjw.kinematics(m, d_fd)
      mjw.com_pos(m, d_fd)
      mjw.crb(m, d_fd)
      mjw.factor_m(m, d_fd)
      mjw.transmission(m, d_fd)
      mjw.fwd_velocity(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        _sum_qacc_kernel,
        dim=(d_fd.nworld, m.nv),
        inputs=[d_fd.qfrc_bias, l],
      )
      return l.numpy()[0]

    qvel_np = d.qvel.numpy()[0, : mjm.nv]
    fd_grad = _fd_gradient(eval_loss, qvel_np)

    np.testing.assert_allclose(
      ad_grad,
      fd_grad,
      atol=_FD_TOL,
      rtol=_FD_TOL,
      err_msg=f"fwd_velocity grad mismatch ({name})",
    )

  @parameterized.parameters(
    ("hinge", _SIMPLE_HINGE_XML),
  )
  def test_fwd_actuation_grad(self, name, xml):
    """dL/dctrl through fwd_actuation()."""
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)
    enable_grad(d)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.kinematics(m, d)
      mjw.com_pos(m, d)
      mjw.crb(m, d)
      mjw.factor_m(m, d)
      mjw.transmission(m, d)
      mjw.fwd_velocity(m, d)
      mjw.fwd_actuation(m, d)
      wp.launch(
        _sum_qacc_kernel,
        dim=(d.nworld, m.nv),
        inputs=[d.qfrc_actuator, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    def eval_loss(ctrl_np):
      d_fd = mjw.make_data(mjm)
      wp.copy(d_fd.qpos, d.qpos)
      wp.copy(d_fd.qvel, d.qvel)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
      mjw.kinematics(m, d_fd)
      mjw.com_pos(m, d_fd)
      mjw.crb(m, d_fd)
      mjw.factor_m(m, d_fd)
      mjw.transmission(m, d_fd)
      mjw.fwd_velocity(m, d_fd)
      mjw.fwd_actuation(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        _sum_qacc_kernel,
        dim=(d_fd.nworld, m.nv),
        inputs=[d_fd.qfrc_actuator, l],
      )
      return l.numpy()[0]

    ctrl_np = d.ctrl.numpy()[0, : mjm.nu]
    fd_grad = _fd_gradient(eval_loss, ctrl_np)

    np.testing.assert_allclose(
      ad_grad,
      fd_grad,
      atol=_FD_TOL,
      rtol=_FD_TOL,
      err_msg=f"fwd_actuation grad mismatch ({name})",
    )

  @absltest.skipIf(
    wp.get_device().is_cuda and wp.get_device().arch < 70,
    "tile kernels (cuSolverDx) require sm_70+",
  )
  def test_euler_step_grad(self):
    """Full Euler step gradient: dL/dctrl through step()."""
    xml = _SIMPLE_HINGE_XML
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)
    enable_grad(d)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(
        _sum_xpos_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[d.xpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    def eval_loss(ctrl_np):
      _, _, _, d_fd = test_data.fixture(xml=xml, keyframe=0)
      enable_grad(d_fd)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
      mjw.step(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        _sum_xpos_kernel,
        dim=(d_fd.nworld, m.nbody),
        inputs=[d_fd.xpos, l],
      )
      return l.numpy()[0]

    ctrl_np = mjd.ctrl.copy()
    fd_grad = _fd_gradient(eval_loss, ctrl_np)

    np.testing.assert_allclose(
      ad_grad,
      fd_grad,
      atol=_FD_TOL,
      rtol=_FD_TOL,
      err_msg="euler step grad mismatch",
    )


@wp.kernel
def _quat_integrate_kernel(
  # In:
  q_in: wp.array(dtype=wp.quat),
  v_in: wp.array(dtype=wp.vec3),
  dt_in: wp.array(dtype=float),
  # Out:
  q_out: wp.array(dtype=wp.quat),
):
  i = wp.tid()
  q_out[i] = math.quat_integrate(q_in[i], v_in[i], dt_in[i])


@wp.kernel
def _quat_loss_kernel(
  # In:
  q: wp.array(dtype=wp.quat),
  loss: wp.array(dtype=float),
):
  i = wp.tid()
  v = q[i]
  wp.atomic_add(loss, 0, v[0] + v[1] + v[2] + v[3])


class GradQuaternionTest(parameterized.TestCase):
  def test_quat_integrate_nonzero_vel(self):
    """quat_integrate gradient at non-zero angular velocity."""
    q_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    dt_np = np.array([0.01], dtype=np.float32)

    q_arr = wp.array([wp.quat(*q_np)], dtype=wp.quat, requires_grad=True)
    v_arr = wp.array([wp.vec3(*v_np)], dtype=wp.vec3, requires_grad=True)
    dt_arr = wp.array(dt_np, dtype=float, requires_grad=True)
    q_out = wp.zeros(1, dtype=wp.quat, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
      wp.launch(_quat_integrate_kernel, dim=1, inputs=[q_arr, v_arr, dt_arr, q_out])
      wp.launch(_quat_loss_kernel, dim=1, inputs=[q_out, loss])
    tape.backward(loss=loss)

    ad_grad_v = v_arr.grad.numpy()[0].copy()
    tape.zero()

    # Finite-difference
    def eval_loss_v(v_test):
      q_a = wp.array([wp.quat(*q_np)], dtype=wp.quat)
      v_a = wp.array([wp.vec3(*v_test)], dtype=wp.vec3)
      dt_a = wp.array(dt_np, dtype=float)
      qo = wp.zeros(1, dtype=wp.quat)
      l = wp.zeros(1, dtype=float)
      wp.launch(_quat_integrate_kernel, dim=1, inputs=[q_a, v_a, dt_a, qo])
      wp.launch(_quat_loss_kernel, dim=1, inputs=[qo, l])
      return l.numpy()[0]

    fd_grad_v = _fd_gradient(eval_loss_v, v_np)

    np.testing.assert_allclose(
      ad_grad_v,
      fd_grad_v,
      atol=5e-3,
      rtol=5e-2,
      err_msg="quat_integrate grad w.r.t. v (nonzero)",
    )

  def test_quat_integrate_zero_vel(self):
    """quat_integrate gradient at zero angular velocity (singularity test)."""
    q_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_np = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dt_np = np.array([0.01], dtype=np.float32)

    q_arr = wp.array([wp.quat(*q_np)], dtype=wp.quat, requires_grad=True)
    v_arr = wp.array([wp.vec3(*v_np)], dtype=wp.vec3, requires_grad=True)
    dt_arr = wp.array(dt_np, dtype=float, requires_grad=True)
    q_out = wp.zeros(1, dtype=wp.quat, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
      wp.launch(_quat_integrate_kernel, dim=1, inputs=[q_arr, v_arr, dt_arr, q_out])
      wp.launch(_quat_loss_kernel, dim=1, inputs=[q_out, loss])
    tape.backward(loss=loss)

    ad_grad_v = v_arr.grad.numpy()[0].copy()
    tape.zero()

    # Should not be NaN or Inf
    self.assertTrue(np.all(np.isfinite(ad_grad_v)), f"quat_integrate grad contains NaN/Inf at zero velocity: {ad_grad_v}")

    # Finite-difference
    def eval_loss_v(v_test):
      q_a = wp.array([wp.quat(*q_np)], dtype=wp.quat)
      v_a = wp.array([wp.vec3(*v_test)], dtype=wp.vec3)
      dt_a = wp.array(dt_np, dtype=float)
      qo = wp.zeros(1, dtype=wp.quat)
      l = wp.zeros(1, dtype=float)
      wp.launch(_quat_integrate_kernel, dim=1, inputs=[q_a, v_a, dt_a, qo])
      wp.launch(_quat_loss_kernel, dim=1, inputs=[qo, l])
      return l.numpy()[0]

    fd_grad_v = _fd_gradient(eval_loss_v, v_np)

    np.testing.assert_allclose(
      ad_grad_v,
      fd_grad_v,
      atol=5e-3,
      rtol=5e-2,
      err_msg="quat_integrate grad w.r.t. v (zero vel)",
    )

  def test_quat_integrate_grad_q(self):
    """quat_integrate gradient w.r.t. input quaternion q."""
    q_np = np.array([0.9239, 0.3827, 0.0, 0.0], dtype=np.float32)  # ~45 deg rotation
    v_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    dt_np = np.array([0.01], dtype=np.float32)

    q_arr = wp.array([wp.quat(*q_np)], dtype=wp.quat, requires_grad=True)
    v_arr = wp.array([wp.vec3(*v_np)], dtype=wp.vec3, requires_grad=True)
    dt_arr = wp.array(dt_np, dtype=float, requires_grad=True)
    q_out = wp.zeros(1, dtype=wp.quat, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
      wp.launch(_quat_integrate_kernel, dim=1, inputs=[q_arr, v_arr, dt_arr, q_out])
      wp.launch(_quat_loss_kernel, dim=1, inputs=[q_out, loss])
    tape.backward(loss=loss)

    ad_grad_q = q_arr.grad.numpy()[0].copy()
    tape.zero()

    def eval_loss_q(q_test):
      q_a = wp.array([wp.quat(*q_test)], dtype=wp.quat)
      v_a = wp.array([wp.vec3(*v_np)], dtype=wp.vec3)
      dt_a = wp.array(dt_np, dtype=float)
      qo = wp.zeros(1, dtype=wp.quat)
      l = wp.zeros(1, dtype=float)
      wp.launch(_quat_integrate_kernel, dim=1, inputs=[q_a, v_a, dt_a, qo])
      wp.launch(_quat_loss_kernel, dim=1, inputs=[qo, l])
      return l.numpy()[0]

    fd_grad_q = _fd_gradient(eval_loss_q, q_np)

    np.testing.assert_allclose(
      ad_grad_q,
      fd_grad_q,
      atol=5e-2,
      rtol=5e-2,
      err_msg="quat_integrate grad w.r.t. q",
    )


class GradUtilTest(absltest.TestCase):
  def test_enable_disable_grad(self):
    """enable_grad / disable_grad toggle requires_grad on Data fields."""
    mjm = mujoco.MjModel.from_xml_string(_SIMPLE_HINGE_XML)
    d = mjw.make_data(mjm)

    # Initially, requires_grad should be False
    self.assertFalse(d.qpos.requires_grad)

    mjw.enable_grad(d)
    self.assertTrue(d.qpos.requires_grad)
    self.assertTrue(d.qvel.requires_grad)
    self.assertTrue(d.ctrl.requires_grad)

    mjw.disable_grad(d)
    self.assertFalse(d.qpos.requires_grad)

  def test_make_diff_data(self):
    """make_diff_data returns Data with gradient tracking enabled."""
    mjm = mujoco.MjModel.from_xml_string(_SIMPLE_HINGE_XML)
    d = mjw.make_diff_data(mjm)

    self.assertTrue(d.qpos.requires_grad)
    self.assertTrue(d.qvel.requires_grad)
    self.assertTrue(d.ctrl.requires_grad)
    self.assertTrue(d.xpos.requires_grad)
    self.assertTrue(d.qacc.requires_grad)

  def test_make_diff_data_custom_fields(self):
    """make_diff_data with a custom field list."""
    mjm = mujoco.MjModel.from_xml_string(_SIMPLE_HINGE_XML)
    d = mjw.make_diff_data(mjm, grad_fields=["qpos", "xpos"])

    self.assertTrue(d.qpos.requires_grad)
    self.assertTrue(d.xpos.requires_grad)
    self.assertFalse(d.qvel.requires_grad)
    self.assertFalse(d.ctrl.requires_grad)


if __name__ == "__main__":
  absltest.main()
