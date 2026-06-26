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
from mujoco_warp._src.grad import _resolve_field
from mujoco_warp._src.grad import enable_grad

# tolerance for AD vs finite-difference comparison
_FD_TOL = 1e-3

# step-level AD requires GPU (Warp tape backward does not produce gradients on CPU)
_REQUIRES_GPU = not wp.get_device().is_cuda or wp.get_device().arch < 70
_REQUIRES_GPU_REASON = "step-level AD requires CUDA with sm_70+"

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

# Freejoint root + hinge child with actuator, for full step gradient test.
_FREE_HINGE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body pos="0 0 1">
      <joint name="root" type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0 0 1 1 0 0 0 0.3" qvel="0.1 0 0 0 0.1 0 -0.2" ctrl="0.5"/>
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


def _assert_step_ctrl_grad(
  test_case,
  xml,
  loss_on="qpos",
  keyframe=0,
  atol=_FD_TOL,
  rtol=_FD_TOL,
  eps=1e-3,
  err_msg="AD vs FD mismatch",
):
  """Compare AD dL/dctrl through step() against finite differences."""
  fixture_kw = dict(xml=xml) | ({"keyframe": keyframe} if keyframe is not None else {})
  mjm, mjd, m, d = test_data.fixture(**fixture_kw)
  enable_grad(d)

  if loss_on == "qpos":
    loss_kernel, loss_dim = _sum_qpos_kernel, (d.nworld, mjm.nq)
    loss_field = lambda dd: dd.qpos
  else:
    loss_kernel, loss_dim = _sum_xpos_kernel, (d.nworld, m.nbody)
    loss_field = lambda dd: dd.xpos

  # AD gradient
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  tape = wp.Tape()
  with tape:
    mjw.step(m, d)
    if loss_on != "qpos":
      # step() does not recompute kinematics after integration, so xpos still
      # reflects pre-step qpos (zero ctrl dependency). Refresh inside the tape
      # so the loss measures end-of-step body positions.
      mjw.kinematics(m, d)
    wp.launch(loss_kernel, dim=loss_dim, inputs=[loss_field(d), loss])
  tape.backward(loss=loss)
  ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
  tape.zero()

  # Finite-difference gradient
  def eval_loss(ctrl_np):
    _, _, _, d_fd = test_data.fixture(**fixture_kw)
    d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
    mjw.step(m, d_fd)
    if loss_on != "qpos":
      mjw.kinematics(m, d_fd)
    l = wp.zeros(1, dtype=float)
    wp.launch(loss_kernel, dim=loss_dim, inputs=[loss_field(d_fd), l])
    return l.numpy()[0]

  ctrl_np = mjd.ctrl.copy()
  fd_grad = _fd_gradient(eval_loss, ctrl_np, eps=eps)

  # Nonzero guard scaled by FD: a single Euler step has dL/dctrl ~ dt^2
  # (e.g. ~2e-7 for dt=2ms qpos losses), so an absolute threshold would
  # reject correct gradients. Require AD to carry at least 10% of FD's norm.
  fd_norm = np.linalg.norm(fd_grad)
  test_case.assertTrue(
    np.linalg.norm(ad_grad) > max(0.1 * fd_norm, 1e-12),
    f"AD gradient should be nonzero, got |ad|={np.linalg.norm(ad_grad):.3e} vs |fd|={fd_norm:.3e}",
  )
  np.testing.assert_allclose(ad_grad, fd_grad, atol=atol, rtol=rtol, err_msg=err_msg)


@wp.kernel
def _sum_xpos_kernel(
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  # In:
  loss: wp.array[float],
):
  worldid, bodyid = wp.tid()
  v = xpos_in[worldid, bodyid]
  wp.atomic_add(loss, 0, v[0] + v[1] + v[2])


@wp.kernel
def _sum_qacc_kernel(
  # Data in:
  qacc_in: wp.array2d[float],
  # In:
  loss: wp.array[float],
):
  worldid, dofid = wp.tid()
  wp.atomic_add(loss, 0, qacc_in[worldid, dofid])


class GradSmoothTest(parameterized.TestCase):
  @parameterized.parameters(
    ("hinge", _SIMPLE_HINGE_XML),
    ("slide", _SIMPLE_SLIDE_XML),
    ("free", _SIMPLE_FREE_XML),
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

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_euler_step_grad(self):
    """Full Euler step gradient: dL/dctrl through step()."""
    _assert_step_ctrl_grad(self, _SIMPLE_HINGE_XML, loss_on="xpos", err_msg="euler step grad mismatch")

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_euler_step_grad_free(self):
    """Full Euler step gradient for freejoint + hinge model: dL/dctrl."""
    _assert_step_ctrl_grad(self, _FREE_HINGE_XML, loss_on="xpos", err_msg="euler step grad (freejoint+hinge) mismatch")


@wp.kernel
def _quat_integrate_kernel(
  # In:
  q_in: wp.array[wp.quat],
  v_in: wp.array[wp.vec3],
  dt_in: wp.array[float],
  # Out:
  q_out: wp.array[wp.quat],
):
  i = wp.tid()
  q_out[i] = math.quat_integrate(q_in[i], v_in[i], dt_in[i])


@wp.kernel
def _quat_loss_kernel(
  # In:
  q: wp.array[wp.quat],
  loss: wp.array[float],
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


_CONTACT_SLIDE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse" solver="Newton" iterations="30"/>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body pos="0 0 0.05">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" gear="1"/>
  </actuator>
</mujoco>
"""

_CONTACT_SLIDE_DENSE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="dense" solver="Newton" iterations="30"/>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body pos="0 0 0.05">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" gear="1"/>
  </actuator>
</mujoco>
"""

_CONTACT_TANGENTIAL_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="dense" solver="Newton" iterations="30"/>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body pos="0 0 0.11">
      <joint name="jx" type="slide" axis="1 0 0"/>
      <joint name="jz" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1" friction="1.0 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="jx" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0 0" qvel="0 0" ctrl="0.2"/>
  </keyframe>
</mujoco>
"""


def _multi_ball_contact_xml(n_bodies, jacobian="dense"):
  """N independent slide-z balls in contact, each with its own motor.

  With n_bodies>32 this drives nv past the tile-kernel cutoff so the backward
  Hessian solve takes the blocked-Cholesky path in _solve_hessian_system.
  """
  bodies, acts = [], []
  for i in range(n_bodies):
    x = 0.5 * i
    bodies.append(
      f'<body pos="{x} 0 0.05"><joint name="s{i}" type="slide" axis="0 0 1"/><geom type="sphere" size="0.1" mass="1"/></body>'
    )
    acts.append(f'<motor joint="s{i}" gear="1"/>')
  return f"""
  <mujoco>
    <option gravity="0 0 -9.81" jacobian="{jacobian}" solver="Newton" iterations="30"/>
    <worldbody>
      <geom type="plane" size="50 50 0.01"/>
      {"".join(bodies)}
    </worldbody>
    <actuator>{"".join(acts)}</actuator>
  </mujoco>
  """


# Tolerance for contact AD tests (relaxed for contacts)
_CONTACT_FD_TOL = 1e-2


@wp.kernel
def _sum_qpos_kernel(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  loss: wp.array[float],
):
  worldid, qid = wp.tid()
  wp.atomic_add(loss, 0, qpos_in[worldid, qid])


@wp.kernel
def _sum_qpos_sq_kernel(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  loss: wp.array[float],
):
  worldid, qid = wp.tid()
  v = qpos_in[worldid, qid]
  wp.atomic_add(loss, 0, v * v)


@wp.kernel
def _sum_qvel_kernel(
  # Data in:
  qvel_in: wp.array2d[float],
  # In:
  loss: wp.array[float],
):
  worldid, vid = wp.tid()
  wp.atomic_add(loss, 0, qvel_in[worldid, vid])


@wp.kernel
def _sum_qpos_x_sq_kernel(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  loss: wp.array[float],
):
  # squared x-position of world 0 (qpos index 0), for the tangential-friction test
  v = qpos_in[0, 0]
  wp.atomic_add(loss, 0, v * v)


class GradSolverAdjointTest(parameterized.TestCase):
  def _step_ctrl_grad_norm(self, xml, smooth_kwargs, settle_steps=60):
    """Return ||d(sum(qpos_next))/d(ctrl)|| on a settled contact state."""
    mjm, _, m, d = test_data.fixture(xml=xml, keyframe=0)
    enable_grad(d)
    mjw.enable_smooth_adjoint(d, **smooth_kwargs)

    # Settle forward-only to get a representative contact state.
    for _ in range(settle_steps):
      mjw.step(m, d)

    qpos_settled = wp.clone(d.qpos)
    qvel_settled = wp.clone(d.qvel)
    ctrl_settled = wp.clone(d.ctrl)

    d = mjw.make_diff_data(mjm)
    enable_grad(d)
    mjw.reset_data(m, d)
    wp.copy(d.qpos, qpos_settled)
    wp.copy(d.qvel, qvel_settled)
    wp.copy(d.ctrl, ctrl_settled)
    mjw.enable_smooth_adjoint(d, **smooth_kwargs)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(
        _sum_qpos_kernel,
        dim=(d.nworld, mjm.nq),
        inputs=[d.qpos, loss],
      )
    tape.backward(loss=loss)
    grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()
    return float(np.linalg.norm(grad))

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_solver_adjoint_contact_step(self):
    """dL/dctrl through step() with active contacts (Newton solver)."""
    _assert_step_ctrl_grad(
      self,
      _CONTACT_SLIDE_XML,
      loss_on="qpos",
      keyframe=None,
      atol=_CONTACT_FD_TOL,
      rtol=_CONTACT_FD_TOL,
      err_msg="solver adjoint contact step grad mismatch",
    )

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_solver_adjoint_no_active_constraints(self):
    """No active contacts: solver adjoint should match Phase 1 (unconstrained)."""
    # Ball high above ground — no contact
    xml = """
    <mujoco>
      <option gravity="0 0 -9.81" jacobian="sparse" solver="Newton" iterations="30"/>
      <worldbody>
        <geom type="plane" size="5 5 0.01"/>
        <body pos="0 0 2.0">
          <joint name="slide" type="slide" axis="0 0 1"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="slide" gear="1"/>
      </actuator>
    </mujoco>
    """
    _assert_step_ctrl_grad(self, xml, loss_on="qpos", keyframe=None, err_msg="solver adjoint no-contact grad mismatch")

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_solver_adjoint_identity_unconstrained(self):
    """njmax==0 (constraints disabled): identity pass-through."""
    _assert_step_ctrl_grad(
      self,
      _SIMPLE_HINGE_XML,
      loss_on="xpos",
      err_msg="solver adjoint identity (unconstrained) grad mismatch",
    )

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_solver_adjoint_dense_jacobian(self):
    """Dense jacobian contact model: dL/dctrl through step()."""
    _assert_step_ctrl_grad(
      self,
      _CONTACT_SLIDE_DENSE_XML,
      loss_on="qpos",
      keyframe=None,
      atol=_CONTACT_FD_TOL,
      rtol=_CONTACT_FD_TOL,
      err_msg="solver adjoint dense jacobian grad mismatch",
    )

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_surrogate_correction_bounded_relative_to_free_body(self):
    """Surrogate tangential correction should stay bounded vs free-body."""
    grad_free = self._step_ctrl_grad_norm(
      _CONTACT_TANGENTIAL_XML,
      smooth_kwargs=dict(
        free_body_adjoint=True,
      ),
    )
    grad_sur_90 = self._step_ctrl_grad_norm(
      _CONTACT_TANGENTIAL_XML,
      smooth_kwargs=dict(
        friction_surrogate_adjoint=True,
        friction_surrogate_alpha=0.9,
      ),
    )
    grad_sur_99 = self._step_ctrl_grad_norm(
      _CONTACT_TANGENTIAL_XML,
      smooth_kwargs=dict(
        friction_surrogate_adjoint=True,
        friction_surrogate_alpha=0.99,
      ),
    )

    self.assertGreater(grad_free, 1.0e-6)
    self.assertLessEqual(grad_sur_90, grad_free * 1.05)
    self.assertLessEqual(grad_sur_99, grad_free * 1.05)

  @parameterized.named_parameters(
    ("dense", "dense"),
    ("sparse", "sparse"),
  )
  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_solver_adjoint_blocked_cholesky_nv_gt_32(self, jacobian):
    """Backward Hessian solve must run for nv>32 (blocked-Cholesky path).

    With more than 32 DOFs the adjoint Hessian solve in _solve_hessian_system
    leaves the per-world tile kernels and takes the blocked-Cholesky branch,
    where the right-hand side is reshaped to nv_pad width. The incoming adjoint
    is only nv wide, so without padding the reshape raises "Reshaped array must
    have the same total size". This regression test exercises that branch (no
    coverage existed for nv>32 before) and asserts the backward pass completes
    and produces finite gradients with nonzero signal.
    """
    n_bodies = 40  # nv = 40 > 32
    xml = _multi_ball_contact_xml(n_bodies, jacobian=jacobian)
    mjm = mujoco.MjModel.from_xml_string(xml)
    m = mjw.put_model(mjm)
    self.assertGreater(m.nv, 32, "model must exceed the tile-kernel cutoff")

    d = mjw.make_diff_data(mjm, nconmax=256, njmax=256)
    enable_grad(d)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(_sum_qpos_kernel, dim=(d.nworld, mjm.nq), inputs=[d.qpos, loss])
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    self.assertTrue(np.all(np.isfinite(ad_grad)), f"blocked-Cholesky adjoint produced non-finite gradients: {ad_grad}")
    self.assertGreater(np.linalg.norm(ad_grad), 1e-12, "blocked-Cholesky adjoint produced an all-zero gradient")


class BlockedCholeskySolveTest(parameterized.TestCase):
  """Unit tests for the blocked Cholesky factorize+solve kernel used by the adjoint.

  These exercise the kernel directly with a known SPD matrix so correctness is
  checked against numpy without the noise of a full step() finite-difference.
  The key regression is nv that is not a multiple of the tile size: the blocked
  kernels load tiles at tile-aligned offsets, so passing nv (rather than nv_pad)
  as the runtime matrix size gives unaligned loads and an illegal memory access.
  """

  @parameterized.named_parameters(
    ("nv_33", 33),
    ("nv_40", 40),
    ("nv_48", 48),
    ("nv_64", 64),
    ("nv_81", 81),
  )
  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_full_blocked_solve_matches_numpy(self, nv):
    from mujoco_warp._src import types as mjw_types
    from mujoco_warp._src.adjoint import _adjoint_cholesky_full_blocked

    tile = mjw_types.TILE_SIZE_JTDAJ_DENSE
    nv_pad = ((nv + tile - 1) // tile) * tile

    rng = np.random.default_rng(0)
    a = rng.standard_normal((nv, nv)).astype(np.float64)
    h = a @ a.T + nv * np.eye(nv)  # well-conditioned SPD
    rhs = rng.standard_normal(nv).astype(np.float64)
    x_ref = np.linalg.solve(h, rhs)

    # Pad to nv_pad with an identity block on the padding diagonal and zero rhs,
    # mirroring _padding_h_adjoint. The padded system is SPD and its leading nv
    # entries equal the original solution.
    h_pad = np.zeros((nv_pad, nv_pad), dtype=np.float32)
    h_pad[:nv, :nv] = h.astype(np.float32)
    for i in range(nv, nv_pad):
      h_pad[i, i] = 1.0
    b_pad = np.zeros((nv_pad, 1), dtype=np.float32)
    b_pad[:nv, 0] = rhs.astype(np.float32)

    h_w = wp.array(h_pad.reshape(1, nv_pad, nv_pad), dtype=float)
    b_w = wp.array(b_pad.reshape(1, nv_pad, 1), dtype=float)
    out_w = wp.zeros((1, nv_pad, 1), dtype=float)
    hfactor_tmp = wp.zeros((1, nv_pad, nv_pad), dtype=float)

    wp.launch_tiled(
      _adjoint_cholesky_full_blocked(tile, nv_pad),
      dim=1,
      inputs=[h_w, b_w, nv_pad, hfactor_tmp],
      outputs=[out_w],
      block_dim=32,
    )
    wp.synchronize()

    x_kernel = out_w.numpy()[0, :nv, 0].astype(np.float64)
    self.assertTrue(np.all(np.isfinite(x_kernel)), f"blocked solve produced non-finite output at nv={nv}")
    np.testing.assert_allclose(x_kernel, x_ref, atol=1e-4, rtol=1e-4, err_msg=f"blocked solve mismatch at nv={nv}")


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

  def test_enable_backward_module_flags(self):
    """Verify enable_backward is set correctly on all AD-relevant modules."""
    from mujoco_warp._src import collision_smooth
    from mujoco_warp._src import derivative
    from mujoco_warp._src import forward as forward_mod
    from mujoco_warp._src import passive
    from mujoco_warp._src import smooth

    # Modules that SHOULD have enable_backward=True
    for mod in [smooth, forward_mod, passive, derivative, collision_smooth]:
      opts = wp.get_module_options(mod)
      self.assertTrue(
        opts.get("enable_backward", False),
        f"{mod.__name__} should have enable_backward=True",
      )

    # Modules that should NOT have enable_backward
    from mujoco_warp._src import collision_driver
    from mujoco_warp._src import constraint
    from mujoco_warp._src import solver

    for mod in [constraint, solver, collision_driver]:
      opts = wp.get_module_options(mod)
      self.assertFalse(
        opts.get("enable_backward", False),
        f"{mod.__name__} should have enable_backward=False",
      )

  def test_enable_grad_all_smooth_fields(self):
    """All SMOOTH_GRAD_FIELDS are toggled by enable_grad."""
    mjm = mujoco.MjModel.from_xml_string(_SIMPLE_HINGE_XML)
    d = mjw.make_data(mjm)

    mjw.enable_grad(d)
    for name in mjw.SMOOTH_GRAD_FIELDS:
      arr = _resolve_field(d, name)
      if arr is not None and isinstance(arr, wp.array):
        self.assertTrue(
          arr.requires_grad,
          f"SMOOTH_GRAD_FIELDS field '{name}' not enabled by enable_grad",
        )

    mjw.disable_grad(d)
    for name in mjw.SMOOTH_GRAD_FIELDS:
      arr = _resolve_field(d, name)
      if arr is not None and isinstance(arr, wp.array):
        self.assertFalse(
          arr.requires_grad,
          f"SMOOTH_GRAD_FIELDS field '{name}' not disabled by disable_grad",
        )

  def test_forward_without_grad_no_error(self):
    """Forward pipeline without enable_grad works (no errors, no gradients)."""
    mjm, mjd, m, d = test_data.fixture(xml=_SIMPLE_HINGE_XML, keyframe=0)
    # Do NOT call enable_grad
    mjw.kinematics(m, d)
    mjw.com_pos(m, d)
    mjw.crb(m, d)

    # Verify no requires_grad is set
    self.assertFalse(d.qpos.requires_grad)
    self.assertFalse(d.xpos.requires_grad)

  def test_diff_step_produces_nonzero_gradients(self):
    """diff_step with enable_grad produces nonzero gradients."""
    mjm, mjd, m, d = test_data.fixture(xml=_SIMPLE_HINGE_XML, keyframe=0)
    enable_grad(d)

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

    ad_grad = d.qpos.grad.numpy()[0, : mjm.nq]
    # With a non-zero keyframe, kinematics gradients should be nonzero
    self.assertTrue(
      np.any(np.abs(ad_grad) > 1e-6),
      "enable_grad + tape should produce nonzero gradients",
    )


# ---- Test models for integrator gradient path ----

_HINGE_EULERDAMP_DISABLED_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse">
    <flag contact="disable" constraint="disable" eulerdamp="disable"/>
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

_HINGE_EULERDAMP_ENABLED_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse"/>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 1 0" damping="1.0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="0 1 0" damping="1.0"/>
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


class GradIntegratorTest(parameterized.TestCase):
  """Tests that exercise the gradient path through the integrator.

  Unlike test_euler_step_grad (which uses loss on xpos and bypasses the
  integrator), these tests use loss on qpos after step(), verifying that
  gradients flow through: ctrl -> actuation -> acceleration -> solver adjoint
  -> integrator -> qpos.
  """

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_euler_qpos_grad_no_eulerdamp(self):
    """dL/dctrl through step() measured on qpos, eulerdamp disabled."""
    _assert_step_ctrl_grad(
      self,
      _HINGE_EULERDAMP_DISABLED_XML,
      loss_on="qpos",
      err_msg="AD vs FD mismatch for dL(qpos)/dctrl (eulerdamp disabled)",
    )

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_euler_qpos_grad_with_eulerdamp(self):
    """dL/dctrl through step() measured on qpos, eulerdamp enabled."""
    _assert_step_ctrl_grad(
      self,
      _HINGE_EULERDAMP_ENABLED_XML,
      loss_on="qpos",
      err_msg="AD vs FD mismatch for dL(qpos)/dctrl (eulerdamp enabled)",
    )

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_multistep_qpos_grad_nonzero(self):
    """dL/dctrl through 2 steps produces nonzero gradient."""
    xml = _HINGE_EULERDAMP_DISABLED_XML
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)
    enable_grad(d)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      mjw.step(m, d)
      wp.launch(
        _sum_qpos_kernel,
        dim=(d.nworld, mjm.nq),
        inputs=[d.qpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    # Multi-step AD vs FD accuracy is limited by shared-array accumulation
    # across steps (a known Warp tape limitation). Here we just verify the
    # gradient is nonzero — single-step FD accuracy is tested above.
    self.assertTrue(
      np.linalg.norm(ad_grad) > 1e-6,
      f"Multi-step AD gradient should be nonzero, got |grad|={np.linalg.norm(ad_grad):.3e}",
    )


_HINGE_EULERDAMP_HIGH_DAMPING_SPARSE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse"/>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 1 0" damping="100.0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="0 1 0" damping="100.0"/>
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

_HINGE_EULERDAMP_HIGH_DAMPING_DENSE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="dense"/>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 1 0" damping="100.0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="0 1 0" damping="100.0"/>
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

_HINGE_EULERDAMP_ENABLED_DENSE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="dense"/>
  <worldbody>
    <body>
      <joint name="j0" type="hinge" axis="0 1 0" damping="1.0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body pos="0 0 -0.5">
        <joint name="j1" type="hinge" axis="0 1 0" damping="1.0"/>
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


class GradEulerDampStressTest(parameterized.TestCase):
  """Stress tests for the euler damping adjoint with high damping and dense jacobian."""

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  @parameterized.named_parameters(
    ("high_damp_sparse", _HINGE_EULERDAMP_HIGH_DAMPING_SPARSE_XML),
    ("high_damp_dense", _HINGE_EULERDAMP_HIGH_DAMPING_DENSE_XML),
    ("normal_damp_dense", _HINGE_EULERDAMP_ENABLED_DENSE_XML),
  )
  def test_euler_damp_adjoint(self, xml):
    """dL/dctrl through step() with eulerdamp enabled, AD matches FD."""
    _assert_step_ctrl_grad(self, xml, loss_on="qpos", err_msg="AD vs FD mismatch for euler damp adjoint")


class GradFlexTest(parameterized.TestCase):
  """Regression tests for differentiating through flex (cloth / soft body) dynamics."""

  _CLOTH_XML = """
<mujoco>
  <option gravity="0 0 -9.81" solver="Newton" iterations="10"/>
  <worldbody>
    <flexcomp name="cloth" type="grid" count="4 4 1" spacing="0.1 0.1 0.1" pos="0 0 1" dim="2" mass="1">
      <elasticity young="1e3" poisson="0.3" thickness="0.01"/>
    </flexcomp>
  </worldbody>
</mujoco>
"""

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_flex_cloth_backward_runs(self):
    """Backward through a cloth step must not fault and must give finite gradients.

    The flex force kernels (_flex_vertices, _flex_edges, _flex_elasticity,
    _flex_bending) scan the flexes to find the one owning each vertex/edge/element.
    When the owning flex id was left uninitialized the autodiff-generated backward
    kernel indexed adjoint arrays with a stale/negative id, faulting with an illegal
    memory access. This exercises the full backward pass over a cloth grid.
    """
    mjm = mujoco.MjModel.from_xml_string(self._CLOTH_XML)
    m = mjw.put_model(mjm)
    self.assertGreater(mjm.nflex, 0, "model must contain a flex")

    d = mjw.make_diff_data(mjm)
    enable_grad(d)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(_sum_qpos_kernel, dim=(d.nworld, mjm.nq), inputs=[d.qpos, loss])
    tape.backward(loss=loss)
    qvel_grad = d.qvel.grad.numpy()
    tape.zero()

    self.assertTrue(np.all(np.isfinite(qvel_grad)), "flex backward produced non-finite gradients")
    self.assertGreater(np.linalg.norm(qvel_grad), 1e-12, "flex backward produced an all-zero gradient")

  _SPATIAL_TENDON_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <site name="anchor" pos="0 0 1"/>
    <body pos="0.2 0 0.7">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" mass="1"/>
      <body pos="0 0 -0.3">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" mass="1"/>
        <site name="tip" pos="0 0 -0.3"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial stiffness="20" damping="2">
      <site site="anchor"/>
      <site site="tip"/>
    </spatial>
  </tendon>
  <actuator>
    <motor joint="j0" gear="1"/>
    <motor joint="j1" gear="1"/>
  </actuator>
  <keyframe>
    <key qpos="0.4 -0.3" qvel="0.1 -0.05" ctrl="0.2 -0.1"/>
  </keyframe>
</mujoco>
"""

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_spatial_site_tendon_backward_runs(self):
    """Backward through a spatial site tendon must not fault and must give finite gradients.

    The _spatial_site_tendon Jacobian walked the body chain with a pointer that decremented
    across nested loops and read a sparse index defined inside an inner while loop. The
    autodiff-generated backward kernel faulted with an illegal memory access on that pointer
    arithmetic. This exercises the full backward pass over a two-link chain spanned by a
    spatial tendon and checks the gradient matches finite differences.
    """
    mjm, mjd, m, d = test_data.fixture(xml=self._SPATIAL_TENDON_XML, keyframe=0)
    self.assertGreater(mjm.ntendon, 0, "model must contain a tendon")
    enable_grad(d)
    mjw.enable_smooth_adjoint(d)

    # Capture the keyframe state before stepping so the FD check evaluates at the same point.
    qpos0 = wp.clone(d.qpos)
    qvel0 = wp.clone(d.qvel)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(_sum_qpos_kernel, dim=(d.nworld, mjm.nq), inputs=[d.qpos, loss])
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    self.assertTrue(np.all(np.isfinite(ad_grad)), "spatial tendon backward produced non-finite gradients")

    # Finite-difference check on dL/dctrl, evaluated at the same keyframe state.
    def eval_loss(ctrl_np):
      d_fd = mjw.make_diff_data(mjm)
      enable_grad(d_fd)
      mjw.reset_data(m, d_fd)
      wp.copy(d_fd.qpos, qpos0)
      wp.copy(d_fd.qvel, qvel0)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
      mjw.enable_smooth_adjoint(d_fd)
      mjw.step(m, d_fd)
      return float(d_fd.qpos.numpy().sum())

    c0 = mjd.ctrl[: mjm.nu].copy()
    eps = 1e-4
    fd_grad = np.zeros(mjm.nu)
    for i in range(mjm.nu):
      cp = c0.copy()
      cp[i] += eps
      cm = c0.copy()
      cm[i] -= eps
      fd_grad[i] = (eval_loss(cp) - eval_loss(cm)) / (2 * eps)
    np.testing.assert_allclose(ad_grad, fd_grad, atol=1e-3, rtol=1e-3, err_msg="spatial tendon grad mismatch")


# Multi-step rollout through a persistent contact: a body resting on the floor, actuated
# upward, so the floor contact is active and the control affects the loss *through* that
# contact across many steps. This is the regime gradient-based policy optimization (e.g.
# SHAC) actually uses, and it is not covered by the single-step contact tests above.
_CONTACT_MULTISTEP_XML = """
<mujoco>
  <option timestep="0.005" gravity="0 0 -9.81" jacobian="sparse" solver="Newton" iterations="30">
    <flag eulerdamp="disable"/>
  </option>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body pos="0 0 0.09">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" gear="1"/>
  </actuator>
</mujoco>
"""


class GradContactMultiStepTest(parameterized.TestCase):
  """Regression: dL/dctrl through a multi-step rollout with an active contact.

  Reverse-mode gradients through a persistent contact over a multi-step rollout must match
  finite differences in sign and magnitude. This is what first-order policy optimization
  (SHAC-style) on contact-rich tasks like hopper or cheetah relies on. The fix is the contact
  active-set Hessian capture plus the efc.aref adjoint (which carries the Baumgarte velocity
  term, i.e. the contact's dissipation of qvel); without them the backward returned a
  free-body gradient through contact, off by 10-100x over a 20-step rollout.
  """

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  @parameterized.parameters(5, 20)
  def test_multistep_contact_ad_matches_fd(self, nsteps):
    mjm = mujoco.MjModel.from_xml_string(_CONTACT_MULTISTEP_XML)
    nu = mjm.nu

    def eval_loss(ctrl_np):
      _, _, m_fd, d_fd = test_data.fixture(xml=_CONTACT_MULTISTEP_XML)
      mjw.reset_data(m_fd, d_fd)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
      for _ in range(nsteps):
        wp.copy(d_fd.ctrl, wp.array(ctrl_np.reshape(1, -1), dtype=float))
        mjw.step(m_fd, d_fd)
      q = d_fd.qpos.numpy()[0]
      return float(np.sum(q * q))

    # AD gradient through the multi-step rollout.
    _, _, m, d = test_data.fixture(xml=_CONTACT_MULTISTEP_XML)
    enable_grad(d)
    ctrl0 = np.full(nu, 0.2, dtype=np.float32)
    ctrl = wp.array(ctrl0.reshape(1, -1), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      for _ in range(nsteps):
        wp.copy(d.ctrl, ctrl)
        mjw.step(m, d)
      wp.launch(_sum_qpos_sq_kernel, dim=(d.nworld, mjm.nq), inputs=[d.qpos, loss])
    tape.backward(loss=loss)
    ad_grad = np.nan_to_num(ctrl.grad.numpy()[0, :nu].copy())
    tape.zero()

    # Finite-difference reference.
    eps = 1e-3
    fd_grad = np.zeros(nu)
    for i in range(nu):
      cp = ctrl0.copy()
      cp[i] += eps
      cm = ctrl0.copy()
      cm[i] -= eps
      fd_grad[i] = (eval_loss(cp) - eval_loss(cm)) / (2 * eps)

    # Require the AD gradient to point the same way as FD and be similarly scaled.
    # Require the AD gradient to match FD in sign and magnitude. The model has a single DOF,
    # so a cosine test is degenerate; compare component-wise (relative + small absolute floor).
    np.testing.assert_allclose(
      ad_grad, fd_grad, rtol=0.2, atol=1e-9,
      err_msg=f"multi-step contact grad mismatch (nsteps={nsteps}): AD={ad_grad} FD={fd_grad}",
    )


# Analytic integrator gradient: for a 1-DOF unit-mass slider with a unit-gear motor and no
# gravity or contact, one semi-implicit Euler step gives qpos1 = qpos0 + dt^2 * ctrl, so
# d(qpos1)/d(ctrl) = dt^2 exactly. This pins the smooth-dynamics control gradient to its
# closed-form value and guards against double-counting in the integrator/solver adjoint
# chain (a diagonal-DOF inertia solve kernel previously had backward enabled in addition to
# the custom adjoint, which doubled dL/dctrl).
_INTEGRATOR_ANALYTIC_XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 0" jacobian="sparse" solver="Newton">
    <flag eulerdamp="disable" contact="disable" constraint="disable"/>
  </option>
  <worldbody>
    <body>
      <joint name="j" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="j" gear="1"/></actuator>
</mujoco>
"""


class GradIntegratorAnalyticTest(parameterized.TestCase):
  """The integrator control gradient must match its closed-form value (no double counting)."""

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_single_step_ctrl_grad_matches_analytic(self):
    mjm = mujoco.MjModel.from_xml_string(_INTEGRATOR_ANALYTIC_XML)
    dt = float(mjm.opt.timestep)
    _, _, m, d = test_data.fixture(xml=_INTEGRATOR_ANALYTIC_XML)
    enable_grad(d)
    ctrl = wp.array(np.zeros((1, mjm.nu), dtype=np.float32), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      wp.copy(d.ctrl, ctrl)
      mjw.step(m, d)
      wp.launch(_sum_qpos_kernel, dim=(d.nworld, mjm.nq), inputs=[d.qpos, loss])
    tape.backward(loss=loss)
    ad = float(np.nan_to_num(ctrl.grad.numpy()[0, 0]))
    # closed form: d(sum qpos)/d(ctrl) = dt^2
    np.testing.assert_allclose(ad, dt * dt, rtol=1e-3, atol=1e-12,
                               err_msg=f"integrator ctrl gradient {ad:.3e} != analytic dt^2 {dt*dt:.3e}")


# Single-step contact dissipation: a body falling onto the floor. One step contracts an
# initial vertical velocity perturbation by the contact, so d(qvel1)/d(qvel0) < 1. The
# free-body (contact-free) adjoint returns 1.0; the contact-adjoint Hessian capture must
# reproduce the finite-difference value. This is the per-step Jacobian whose error compounds
# over a rollout, and it specifically exercises the small-nv path where solver_h is empty.
_CONTACT_DISSIPATION_XML = """
<mujoco>
  <option timestep="0.004" gravity="0 0 -9.81" jacobian="sparse" solver="Newton" iterations="50">
    <flag eulerdamp="disable"/>
  </option>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body pos="0 0 0.095">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="slide" gear="1"/></actuator>
</mujoco>
"""


class GradContactDissipationTest(absltest.TestCase):
  """Regression: the single-step contact Jacobian d(qvel1)/d(qvel0) must match FD (not 1.0).

  A penetrating contact dissipates a velocity perturbation in one step. Without the
  contact-adjoint Hessian (built unconditionally from M + J^T D J, including the small-nv path
  where the solver keeps no explicit Hessian) the backward returns the free-body value 1.0.
  """

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_single_step_qvel_jacobian_matches_fd(self):
    q0 = np.array([-0.005], np.float32)
    v0 = np.array([-0.1], np.float32)

    def step_vz(vz):
      _, _, m, d = test_data.fixture(xml=_CONTACT_DISSIPATION_XML)
      mjw.reset_data(m, d)
      d.qpos = wp.array(q0.reshape(1, -1), dtype=float)
      d.qvel = wp.array(np.array([[vz]], np.float32), dtype=float)
      mjw.step(m, d)
      return float(d.qvel.numpy()[0, 0])

    eps = 1e-4
    fd = (step_vz(v0[0] + eps) - step_vz(v0[0] - eps)) / (2 * eps)

    _, _, m, d = test_data.fixture(xml=_CONTACT_DISSIPATION_XML)
    enable_grad(d)
    qp = wp.array(q0.reshape(1, -1), dtype=float, requires_grad=True)
    qv = wp.array(v0.reshape(1, -1), dtype=float, requires_grad=True)
    d.qpos = qp
    d.qvel = qv
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(_sum_qvel_kernel, dim=(d.nworld, m.nv), inputs=[d.qvel, loss])
    tape.backward(loss=loss)
    ad = float(np.nan_to_num(qv.grad.numpy()[0, 0]))
    self.assertLess(ad, 0.95, f"AD d(vz1)/d(vz0)={ad:.4f} looks like the free-body value (no contact dissipation)")
    np.testing.assert_allclose(ad, fd, rtol=0.05, atol=1e-4,
                               err_msg=f"contact dissipation Jacobian mismatch: AD={ad:.4f} FD={fd:.4f}")


# Multi-body multi-contact: two independent spheres resting on the floor, each actuated. The
# AD path previously corrupted device memory here (the differentiable contact assembly wrote a
# dense Jacobian index into the sparse efc.J array, which for any second contact row scribbled
# over adjacent buffers, resetting nefc to 0 and dropping all contact rows). The gradient was
# then off by ~600x over a 20-step rollout. Guards the sparse efc.J write.
_MULTI_CONTACT_XML = """
<mujoco>
  <option timestep="0.004" gravity="0 0 -9.81" jacobian="sparse" solver="Newton" iterations="30">
    <flag eulerdamp="disable"/>
  </option>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body pos="0 0 0.09"><joint name="ja" type="slide" axis="0 0 1"/><geom type="sphere" size="0.1" mass="1"/></body>
    <body pos="0.5 0 0.09"><joint name="jb" type="slide" axis="0 0 1"/><geom type="sphere" size="0.1" mass="1"/></body>
  </worldbody>
  <actuator>
    <motor joint="ja" gear="1"/>
    <motor joint="jb" gear="1"/>
  </actuator>
</mujoco>
"""


class GradMultiContactTest(parameterized.TestCase):
  """Regression: dL/dctrl through multiple simultaneous contacts must match FD.

  Exercises the multi-body case (multiple efc contact rows) that the single-contact tests miss.
  The bug was a dense-into-sparse Jacobian write in the differentiable contact assembly that
  corrupted nefc for any scene with more than one contact row.
  """

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  @parameterized.parameters(5, 20)
  def test_multibody_contact_grad_matches_fd(self, nsteps):
    mjm = mujoco.MjModel.from_xml_string(_MULTI_CONTACT_XML)
    nu = mjm.nu
    ctrl0 = np.array([0.3, 0.2], dtype=np.float32)

    def eval_loss(ctrl_np):
      _, _, m_fd, d_fd = test_data.fixture(xml=_MULTI_CONTACT_XML)
      mjw.reset_data(m_fd, d_fd)
      for _ in range(nsteps):
        wp.copy(d_fd.ctrl, wp.array(ctrl_np.reshape(1, -1), dtype=float))
        mjw.step(m_fd, d_fd)
      q = d_fd.qpos.numpy()[0]
      return float(np.sum(q * q))

    _, _, m, d = test_data.fixture(xml=_MULTI_CONTACT_XML)
    enable_grad(d)
    ctrl = wp.array(ctrl0.reshape(1, -1), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      for _ in range(nsteps):
        wp.copy(d.ctrl, ctrl)
        mjw.step(m, d)
      wp.launch(_sum_qpos_sq_kernel, dim=(d.nworld, mjm.nq), inputs=[d.qpos, loss])
    tape.backward(loss=loss)
    ad_grad = np.nan_to_num(ctrl.grad.numpy()[0, :nu].copy())
    tape.zero()

    eps = 1e-3
    fd_grad = np.zeros(nu)
    for i in range(nu):
      cp = ctrl0.copy(); cp[i] += eps
      cm = ctrl0.copy(); cm[i] -= eps
      fd_grad[i] = (eval_loss(cp) - eval_loss(cm)) / (2 * eps)

    np.testing.assert_allclose(
      ad_grad, fd_grad, rtol=0.2, atol=1e-9,
      err_msg=f"multi-contact grad mismatch (nsteps={nsteps}): AD={ad_grad} FD={fd_grad}",
    )


# Tangential friction over a rollout. The pyramidal friction rows carry the contact's
# velocity-dissipation gradient. The native sparse-J autodiff mis-projects these multi-column
# rows (the two pyramid edges cancel antisymmetrically), so the tangential d(qvel1)/d(qvel0)
# used to stay at the free-body 1.0. The backward now scatters the dissipation adjoint
# (-b*D*(J.v)*J) for multi-column rows directly into qvel.grad, so the tangential control
# gradient matches finite differences.
_FRICTION_XML = """
<mujoco>
  <option timestep="0.004" gravity="0 0 -9.81" jacobian="sparse" solver="Newton" iterations="50">
    <flag eulerdamp="disable"/>
  </option>
  <worldbody>
    <geom type="plane" size="5 5 0.01" friction="2 0 0"/>
    <body pos="0 0 0.095">
      <joint name="jx" type="slide" axis="1 0 0"/>
      <joint name="jz" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="1" friction="2 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="jx" gear="1"/>
    <motor joint="jz" gear="1"/>
  </actuator>
</mujoco>
"""


class GradFrictionTangentialTest(absltest.TestCase):
  """Reverse-mode tangential (pyramidal friction) control gradient matches finite differences.

  The pyramidal friction cone produces multi-column efc rows whose native velocity-dissipation
  autodiff cancels antisymmetrically, which previously left the tangential gradient at the
  free-body value over a rollout. The backward injects the dissipation adjoint for those rows
  directly into qvel.grad, recovering the correct tangential gradient.
  """

  @absltest.skipIf(_REQUIRES_GPU, _REQUIRES_GPU_REASON)
  def test_tangential_friction_multistep_grad_matches_fd(self):
    nsteps = 20
    ctrl0 = np.array([0.5, 0.0], dtype=np.float32)

    def eval_loss(ctrl_np):
      _, _, m_fd, d_fd = test_data.fixture(xml=_FRICTION_XML)
      mjw.reset_data(m_fd, d_fd)
      for _ in range(nsteps):
        wp.copy(d_fd.ctrl, wp.array(ctrl_np.reshape(1, -1), dtype=float))
        mjw.step(m_fd, d_fd)
      return float(d_fd.qpos.numpy()[0, 0] ** 2)

    _, _, m, d = test_data.fixture(xml=_FRICTION_XML)
    enable_grad(d)
    ctrl = wp.array(ctrl0.reshape(1, -1), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      for _ in range(nsteps):
        wp.copy(d.ctrl, ctrl)
        mjw.step(m, d)
      wp.launch(_sum_qpos_x_sq_kernel, dim=1, inputs=[d.qpos, loss])
    tape.backward(loss=loss)
    ad = float(np.nan_to_num(ctrl.grad.numpy()[0, 0]))
    tape.zero()

    eps = 1e-3
    cp = ctrl0.copy(); cp[0] += eps
    cm = ctrl0.copy(); cm[0] -= eps
    fd = (eval_loss(cp) - eval_loss(cm)) / (2 * eps)

    np.testing.assert_allclose(ad, fd, rtol=0.2, atol=1e-10,
                               err_msg=f"tangential friction grad: AD={ad:.3e} FD={fd:.3e}")


if __name__ == "__main__":
  absltest.main()
