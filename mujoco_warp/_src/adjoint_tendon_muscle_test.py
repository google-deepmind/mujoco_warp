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
"""Tests for analytic tendon and Hill-type muscle adjoints."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp._src import adjoint


@wp.kernel
def _weighted_sum_2d(x: wp.array2d[float], w: wp.array2d[float], loss_out: wp.array[float]):
  worldid, i = wp.tid()
  wp.atomic_add(loss_out, 0, x[worldid, i] * w[worldid, i])  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _weighted_sum_vec3(x: wp.array2d[wp.vec3], w: wp.array2d[wp.vec3], loss_out: wp.array[float]):
  worldid, i = wp.tid()
  wp.atomic_add(loss_out, 0, wp.dot(x[worldid, i], w[worldid, i]))  # kernel_analyzer: ignore[atomic]


def _normalize_quats_mjm(mjm: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
  qpos = qpos.copy()
  for j in range(mjm.njnt):
    jt = mjm.jnt_type[j]
    adr = mjm.jnt_qposadr[j]
    if jt == mujoco.mjtJoint.mjJNT_FREE:
      q = qpos[..., adr + 3 : adr + 7]
      qpos[..., adr + 3 : adr + 7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    elif jt == mujoco.mjtJoint.mjJNT_BALL:
      q = qpos[..., adr : adr + 4]
      qpos[..., adr : adr + 4] = q / np.linalg.norm(q, axis=-1, keepdims=True)
  return qpos


def _mj_step_loss_world(
  mjm: mujoco.MjModel,
  qpos: np.ndarray,
  qvel: np.ndarray,
  act: np.ndarray,
  ctrl: np.ndarray,
  mocap_pos: np.ndarray,
  mocap_quat: np.ndarray,
  weights: dict[str, np.ndarray],
  post_fk: bool = False,
) -> float:
  mjd = mujoco.MjData(mjm)
  mjd.qpos[:] = qpos
  mjd.qvel[:] = qvel
  if mjm.na:
    mjd.act[:] = act
  if mjm.nu:
    mjd.ctrl[:] = ctrl
  if mjm.nmocap:
    mjd.mocap_pos[:] = mocap_pos
    mjd.mocap_quat[:] = mocap_quat
  mujoco.mj_step(mjm, mjd)
  if post_fk:
    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_comPos(mjm, mjd)
    mujoco.mj_tendon(mjm, mjd)
  total = 0.0
  for name, w in weights.items():
    val = getattr(mjd, name)
    total += float(np.sum(val * w))
  return total


def _fd_step_world(
  mjm: mujoco.MjModel,
  qpos: np.ndarray,
  qvel: np.ndarray,
  act: np.ndarray,
  ctrl: np.ndarray,
  mocap_pos: np.ndarray,
  mocap_quat: np.ndarray,
  weights: dict[str, np.ndarray],
  post_fk: bool = False,
  eps: float = 1e-6,
) -> dict[str, np.ndarray]:
  inputs = {
    "qpos": qpos.astype(np.float64).copy(),
    "qvel": qvel.astype(np.float64).copy(),
    "act": act.astype(np.float64).copy(),
    "ctrl": ctrl.astype(np.float64).copy(),
    "mocap_pos": mocap_pos.astype(np.float64).copy(),
    "mocap_quat": mocap_quat.astype(np.float64).copy(),
  }
  grads: dict[str, np.ndarray] = {}
  for key, arr in inputs.items():
    g = np.zeros_like(arr)
    flat = arr.reshape(-1)
    g_flat = g.reshape(-1)
    for idx in range(flat.size):
      orig = flat[idx]
      flat[idx] = orig + eps
      lp = _mj_step_loss_world(
        mjm,
        inputs["qpos"],
        inputs["qvel"],
        inputs["act"],
        inputs["ctrl"],
        inputs["mocap_pos"],
        inputs["mocap_quat"],
        weights,
        post_fk=post_fk,
      )
      flat[idx] = orig - eps
      lm = _mj_step_loss_world(
        mjm,
        inputs["qpos"],
        inputs["qvel"],
        inputs["act"],
        inputs["ctrl"],
        inputs["mocap_pos"],
        inputs["mocap_quat"],
        weights,
        post_fk=post_fk,
      )
      flat[idx] = orig
      g_flat[idx] = (lp - lm) / (2.0 * eps)
    grads[key] = g
  return grads


def _check_step_grads(
  testcase: parameterized.TestCase,
  xml: str,
  *,
  nworld: int = 2,
  seed: int = 0,
  mocap_grad: bool = False,
  post_fk: bool = False,
  qpos_override: np.ndarray | None = None,
  qvel_override: np.ndarray | None = None,
  ctrl_override: np.ndarray | None = None,
  stiffnesspoly: tuple[float, float] | None = None,
  dampingpoly: tuple[float, float] | None = None,
  atol: float = 2e-3,
  rtol: float = 3e-3,
):
  mjw.enable_grad()
  mjm = mujoco.MjModel.from_xml_string(xml)
  if stiffnesspoly is not None and mjm.ntendon:
    mjm.tendon_stiffnesspoly[:, 0] = stiffnesspoly[0]
    mjm.tendon_stiffnesspoly[:, 1] = stiffnesspoly[1]
  if dampingpoly is not None and mjm.ntendon:
    mjm.tendon_dampingpoly[:, 0] = dampingpoly[0]
    mjm.tendon_dampingpoly[:, 1] = dampingpoly[1]
  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=nworld)
  d_out = mjw.make_data(mjm, nworld=nworld)

  rng = np.random.default_rng(seed)
  if qpos_override is not None:
    qpos0 = _normalize_quats_mjm(mjm, qpos_override).astype(np.float32)
  else:
    qpos0 = np.tile(mjm.qpos0, (nworld, 1)) + 0.18 * rng.standard_normal((nworld, mjm.nq))
    qpos0 = _normalize_quats_mjm(mjm, qpos0).astype(np.float32)
  if qvel_override is not None:
    qvel0 = qvel_override.astype(np.float32)
  else:
    qvel0 = (0.35 * rng.standard_normal((nworld, mjm.nv))).astype(np.float32)
  act0 = rng.uniform(0.15, 0.85, size=(nworld, mjm.na)).astype(np.float32) if mjm.na else np.zeros((nworld, 0), np.float32)
  if ctrl_override is not None:
    ctrl0 = ctrl_override.astype(np.float32)
  elif mjm.nu:
    ctrl0 = rng.uniform(0.15, 0.85, size=(nworld, mjm.nu)).astype(np.float32)
  else:
    ctrl0 = np.zeros((nworld, 0), np.float32)

  mocap_pos0 = np.zeros((nworld, mjm.nmocap, 3), dtype=np.float32)
  mocap_quat0 = np.zeros((nworld, mjm.nmocap, 4), dtype=np.float32)
  if mjm.nmocap:
    base_d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, base_d)
    mocap_pos0 = (np.tile(base_d.mocap_pos, (nworld, 1, 1)) + 0.03 * rng.standard_normal((nworld, mjm.nmocap, 3))).astype(
      np.float32
    )
    mq = np.tile(base_d.mocap_quat, (nworld, 1, 1)) + 0.08 * rng.standard_normal((nworld, mjm.nmocap, 4))
    mq /= np.linalg.norm(mq, axis=-1, keepdims=True)
    mocap_quat0 = mq.astype(np.float32)

  d.qpos.assign(qpos0)
  d.qvel.assign(qvel0)
  if mjm.na:
    d.act.assign(act0)
  if mjm.nu:
    d.ctrl.assign(ctrl0)
  if mjm.nmocap:
    d.mocap_pos.assign(mocap_pos0)
    d.mocap_quat.assign(mocap_quat0)
    if mocap_grad:
      d.mocap_pos.requires_grad = True
      d.mocap_quat.requires_grad = True

  w_qpos = rng.standard_normal((nworld, mjm.nq)).astype(np.float32)
  w_qvel = rng.standard_normal((nworld, mjm.nv)).astype(np.float32)
  w_qacc = 0.05 * rng.standard_normal((nworld, mjm.nv)).astype(np.float32)
  w_act = rng.standard_normal((nworld, mjm.na)).astype(np.float32) if mjm.na else np.zeros((nworld, 0), np.float32)
  w_act_dot = 0.1 * rng.standard_normal((nworld, mjm.na)).astype(np.float32) if mjm.na else np.zeros((nworld, 0), np.float32)
  w_site = (
    rng.standard_normal((nworld, mjm.nsite, 3)).astype(np.float32)
    if (post_fk and mjm.nsite)
    else np.zeros((nworld, 0, 3), np.float32)
  )
  w_ten_len = (
    rng.standard_normal((nworld, mjm.ntendon)).astype(np.float32)
    if (post_fk and mjm.ntendon)
    else np.zeros((nworld, 0), np.float32)
  )
  w_ten_J = (
    rng.standard_normal((nworld, m.nJten)).astype(np.float32) if (post_fk and m.nJten) else np.zeros((nworld, 0), np.float32)
  )

  loss = wp.zeros(1, dtype=float, requires_grad=True)
  w_qpos_wp = wp.array(w_qpos, dtype=float)
  w_qvel_wp = wp.array(w_qvel, dtype=float)
  w_qacc_wp = wp.array(w_qacc, dtype=float)
  w_act_wp = wp.array(w_act, dtype=float) if mjm.na else None
  w_act_dot_wp = wp.array(w_act_dot, dtype=float) if mjm.na else None
  w_site_wp = wp.array(w_site, dtype=wp.vec3) if (post_fk and mjm.nsite) else None
  w_ten_len_wp = wp.array(w_ten_len, dtype=float) if (post_fk and mjm.ntendon) else None
  w_ten_J_wp = wp.array(w_ten_J, dtype=float) if (post_fk and m.nJten) else None

  bc = adjoint.create_backward_context(m, d)
  tape = wp.Tape()
  with adjoint.backward_context(bc):
    with tape:
      mjw.step(m, d, d_out)
      if post_fk:
        mjw.fwd_kinematics(m, d_out)
      wp.launch(_weighted_sum_2d, dim=(nworld, mjm.nq), inputs=[d_out.qpos, w_qpos_wp], outputs=[loss])
      wp.launch(_weighted_sum_2d, dim=(nworld, mjm.nv), inputs=[d_out.qvel, w_qvel_wp], outputs=[loss])
      wp.launch(_weighted_sum_2d, dim=(nworld, mjm.nv), inputs=[d_out.qacc, w_qacc_wp], outputs=[loss])
      if mjm.na:
        wp.launch(_weighted_sum_2d, dim=(nworld, mjm.na), inputs=[d_out.act, w_act_wp], outputs=[loss])
        wp.launch(_weighted_sum_2d, dim=(nworld, mjm.na), inputs=[d_out.act_dot, w_act_dot_wp], outputs=[loss])
      if post_fk and mjm.nsite:
        wp.launch(_weighted_sum_vec3, dim=(nworld, mjm.nsite), inputs=[d_out.site_xpos, w_site_wp], outputs=[loss])
      if post_fk and mjm.ntendon:
        wp.launch(_weighted_sum_2d, dim=(nworld, mjm.ntendon), inputs=[d_out.ten_length, w_ten_len_wp], outputs=[loss])
      if post_fk and m.nJten:
        wp.launch(_weighted_sum_2d, dim=(nworld, m.nJten), inputs=[d_out.ten_J, w_ten_J_wp], outputs=[loss])
    tape.backward(loss=loss)

  qpos_ad = d.qpos.grad.numpy()
  qvel_ad = d.qvel.grad.numpy()
  act_ad = d.act.grad.numpy() if mjm.na else None
  ctrl_ad = d.ctrl.grad.numpy() if mjm.nu else None
  mpos_ad = d.mocap_pos.grad.numpy() if (mjm.nmocap and mocap_grad) else None
  mquat_ad = d.mocap_quat.grad.numpy() if (mjm.nmocap and mocap_grad) else None

  for w in range(nworld):
    weights_w: dict[str, np.ndarray] = {
      "qpos": w_qpos[w].astype(np.float64),
      "qvel": w_qvel[w].astype(np.float64),
      "qacc": w_qacc[w].astype(np.float64),
    }
    if mjm.na:
      weights_w["act"] = w_act[w].astype(np.float64)
      weights_w["act_dot"] = w_act_dot[w].astype(np.float64)
    if post_fk and mjm.nsite:
      weights_w["site_xpos"] = w_site[w].astype(np.float64)
    if post_fk and mjm.ntendon:
      weights_w["ten_length"] = w_ten_len[w].astype(np.float64)
    if post_fk and m.nJten:
      # Map sparse mjw ten_J weights into MuJoCo C's ten_J layout.
      rowadr_np = m.ten_J_rowadr.numpy()
      rownnz_np = m.ten_J_rownnz.numpy()
      colind_np = m.ten_J_colind.numpy()
      if mujoco.mj_isSparse(mjm):
        weights_w["ten_J"] = w_ten_J[w].astype(np.float64)
      else:
        dense_w = np.zeros((mjm.ntendon, mjm.nv), dtype=np.float64)
        for tid in range(mjm.ntendon):
          adr = rowadr_np[tid]
          for k in range(rownnz_np[tid]):
            dense_w[tid, colind_np[adr + k]] = w_ten_J[w, adr + k]
        weights_w["ten_J"] = dense_w.reshape(-1)

    fd = _fd_step_world(
      mjm,
      qpos0[w],
      qvel0[w],
      act0[w] if mjm.na else np.zeros(0),
      ctrl0[w] if mjm.nu else np.zeros(0),
      mocap_pos0[w] if mjm.nmocap else np.zeros((0, 3)),
      mocap_quat0[w] if mjm.nmocap else np.zeros((0, 4)),
      weights_w,
      post_fk=post_fk,
    )
    np.testing.assert_allclose(qpos_ad[w], fd["qpos"], atol=atol, rtol=rtol, err_msg=f"qpos world {w}")
    np.testing.assert_allclose(qvel_ad[w], fd["qvel"], atol=atol, rtol=rtol, err_msg=f"qvel world {w}")
    if mjm.na:
      np.testing.assert_allclose(act_ad[w], fd["act"], atol=atol, rtol=rtol, err_msg=f"act world {w}")
    if mjm.nu:
      np.testing.assert_allclose(ctrl_ad[w], fd["ctrl"], atol=atol, rtol=rtol, err_msg=f"ctrl world {w}")
    if mjm.nmocap and mocap_grad:
      np.testing.assert_allclose(mpos_ad[w], fd["mocap_pos"], atol=atol, rtol=rtol, err_msg=f"mocap_pos world {w}")
      np.testing.assert_allclose(mquat_ad[w], fd["mocap_quat"], atol=atol, rtol=rtol, err_msg=f"mocap_quat world {w}")


class AdjointTendonMuscleTest(parameterized.TestCase):
  @parameterized.product(integrator=("Euler", "implicitfast"), actearly=(False, True))
  def test_joint_muscle_and_stateful_affine(self, integrator: str, actearly: bool):
    actearly_str = "true" if actearly else "false"
    xml = f"""
    <mujoco>
      <option timestep="0.005" integrator="{integrator}" solver="Newton" iterations="20"/>
      <worldbody>
        <body pos="0 0 0.5">
          <joint name="j0" type="hinge" axis="0 1 0" damping="0.15"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1.2" contype="0" conaffinity="0"/>
          <body pos="0.3 0 0">
            <joint name="j1" type="slide" axis="1 0 0" stiffness="2.0" damping="0.1"/>
            <geom type="sphere" size="0.04" mass="0.8" contype="0" conaffinity="0"/>
            <body pos="0.15 0 0">
              <joint name="j2" type="hinge" axis="0 0 1" actuatorfrclimited="true" actuatorfrcrange="-100 100"/>
              <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.025" mass="0.5" contype="0" conaffinity="0"/>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <general joint="j0" dyntype="muscle" gaintype="muscle" biastype="muscle"
                 dynprm="0.01 0.04 0" gainprm="0.75 1.05 -1 120 0.5 1.6 1.5 1.3 1.2 0"
                 biasprm="0.75 1.05 -1 120 0.5 1.6 1.5 1.3 1.2 0"
                 gear="12" lengthrange="0.1 1.5" actearly="{actearly_str}"/>
        <general joint="j1" dyntype="none" gaintype="muscle" biastype="muscle"
                 gainprm="0.75 1.05 -1 80 0.5 1.6 1.5 1.3 1.2 0"
                 biasprm="0.75 1.05 -1 80 0.5 1.6 1.5 1.3 1.2 0"
                 lengthrange="0.05 1.2" gear="5"/>
        <general joint="j2" dyntype="filterexact" dynprm="0.04" gaintype="affine" biastype="affine"
                 gainprm="6.0 -1.5 -0.4" biasprm="0.2 -0.8 -0.2" actearly="{actearly_str}" gear="2.5"/>
      </actuator>
    </mujoco>
    """
    _check_step_grads(self, xml, seed=11)

  @parameterized.product(integrator=("Euler", "implicitfast"))
  def test_fixed_tendon_muscle_and_passive(self, integrator: str):
    xml = f"""
    <mujoco>
      <option timestep="0.004" integrator="{integrator}" solver="Newton" iterations="20"/>
      <worldbody>
        <body pos="0 0 0.5">
          <joint name="j0" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03" mass="1.0" contype="0" conaffinity="0"/>
          <body pos="0.25 0 0">
            <joint name="j1" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.025" mass="0.7" contype="0" conaffinity="0"/>
          </body>
          <body pos="0 0.1 0">
            <joint name="j2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size="0.03" mass="0.6" contype="0" conaffinity="0"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <fixed name="t0" stiffness="12.0" damping="1.4" springlength="-0.02 0.02">
          <joint joint="j0" coef="0.04"/>
          <joint joint="j1" coef="-0.03"/>
          <joint joint="j2" coef="0.5"/>
        </fixed>
        <fixed name="t1" stiffness="4.0" damping="0.6" springlength="0.0">
          <joint joint="j0" coef="-0.02"/>
          <joint joint="j2" coef="0.3"/>
        </fixed>
      </tendon>
      <actuator>
        <muscle tendon="t0" force="90" lengthrange="-0.3 0.5" damping="0.3"/>
        <general tendon="t1" dyntype="muscle" gaintype="muscle" biastype="muscle"
                 dynprm="0.01 0.04 0" gainprm="0.75 1.05 -1 60 0.5 1.6 1.5 1.3 1.2 0"
                 biasprm="0.75 1.05 -1 60 0.5 1.6 1.5 1.3 1.2 0"
                 lengthrange="-0.3 0.5" actearly="true"/>
      </actuator>
    </mujoco>
    """
    _check_step_grads(self, xml, seed=22, stiffnesspoly=(2.5, 1.2), dampingpoly=(0.4, 0.25))

  @parameterized.product(integrator=("Euler", "implicitfast"))
  def test_spatial_site_pulley_offtree_and_clamped_force(self, integrator: str):
    xml = f"""
    <mujoco>
      <option timestep="0.004" integrator="{integrator}" solver="Newton" iterations="25"/>
      <worldbody>
        <body name="mocap_anchor" pos="0 0.15 0.6" mocap="true">
          <site name="s_mocap" pos="0.02 -0.01 0.03"/>
        </body>
        <body name="branch_a" pos="-0.25 0 0.5">
          <joint name="ja_free" type="free"/>
          <geom type="sphere" size="0.05" mass="1.1" contype="0" conaffinity="0"/>
          <site name="sa0" pos="0.05 0.02 0"/>
          <body pos="0.2 0 0">
            <joint name="ja_hinge" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.18 0 0" size="0.025" mass="0.6" contype="0" conaffinity="0"/>
            <site name="sa1" pos="0.18 0.01 -0.02"/>
          </body>
        </body>
        <body name="branch_b" pos="0.25 0 0.5">
          <joint name="jb_ball" type="ball"/>
          <geom type="sphere" size="0.05" mass="0.9" contype="0" conaffinity="0"/>
          <site name="sb0" pos="-0.04 0.03 0.01"/>
          <body pos="0 -0.15 0">
            <joint name="jb_slide" type="slide" axis="1 0.2 0"/>
            <geom type="sphere" size="0.035" mass="0.5" contype="0" conaffinity="0"/>
            <site name="sb1" pos="0.01 -0.05 0.02"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="t_pulley" stiffness="8.0" damping="1.2" springlength="0.3 0.35">
          <site site="sa1"/>
          <site site="s_mocap"/>
          <pulley divisor="2.0"/>
          <site site="sa0"/>
          <site site="sb0"/>
          <site site="sb1"/>
        </spatial>
        <spatial name="t_clamped" stiffness="3.0" damping="0.4">
          <site site="sa1"/>
          <site site="sb1"/>
        </spatial>
      </tendon>
      <actuator>
        <muscle tendon="t_pulley" force="110" lengthrange="0.4 1.6" ctrllimited="true" ctrlrange="0 1"/>
        <general tendon="t_clamped" gainprm="50" forcelimited="true" forcerange="-2.0 2.0" gear="1.5"/>
      </actuator>
    </mujoco>
    """
    ctrl = np.array([[0.65, 0.5], [0.35, -0.4]], dtype=np.float32)
    _check_step_grads(self, xml, seed=33, mocap_grad=True, ctrl_override=ctrl)

  @parameterized.product(integrator=("Euler", "implicitfast"))
  def test_spatial_geom_wrapping_sphere_and_cylinder(self, integrator: str):
    xml = f"""
    <mujoco>
      <option timestep="0.004" integrator="{integrator}" solver="Newton" iterations="25"/>
      <worldbody>
        <body name="mocap_obs" pos="0.15 0.02 0.5" mocap="true">
          <geom name="g_mocap_sphere" type="sphere" size="0.055" pos="0 0 0" contype="0" conaffinity="0"/>
          <site name="s_mocap_side" pos="0 0.09 0"/>
        </body>
        <body name="upper" pos="0 0 0.5">
          <joint name="j_shoulder" type="hinge" axis="0 0 1" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" mass="1.0" contype="0" conaffinity="0"/>
          <site name="s_prox_top" pos="0.05 0.045 0.02"/>
          <site name="s_prox_cyl" pos="0.18 0.05 0.03"/>
          <site name="s_prox_in" pos="0.22 0.065 -0.02"/>
          <body name="forearm" pos="0.3 0 0">
            <joint name="j_elbow" type="hinge" axis="0 0.2 1" damping="0.08"/>
            <geom name="g_cyl" type="cylinder" pos="0 0 0" euler="15 10 0" size="0.045 0.08" mass="0.4"
                  contype="0" conaffinity="0"/>
            <site name="s_inside" pos="0.01 0.01 0.0"/>
            <geom type="capsule" fromto="0 0 0 0.28 0 0" size="0.02" mass="0.8" contype="0" conaffinity="0"/>
            <site name="s_dist_top" pos="0.22 0.04 -0.02"/>
            <site name="s_dist_cyl" pos="0.14 0.045 -0.03"/>
            <site name="s_dist_in" pos="0.08 -0.065 0.02"/>
            <site name="s_dist_free" pos="0.22 0.14 0.01"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="t_sphere" stiffness="6.0" damping="0.8">
          <site site="s_prox_top"/>
          <geom geom="g_mocap_sphere" sidesite="s_mocap_side"/>
          <site site="s_dist_top"/>
        </spatial>
        <spatial name="t_cyl" stiffness="5.0" damping="0.7">
          <site site="s_prox_cyl"/>
          <geom geom="g_cyl"/>
          <site site="s_dist_cyl"/>
        </spatial>
        <spatial name="t_inside" stiffness="4.0" damping="0.5">
          <site site="s_prox_in"/>
          <geom geom="g_cyl" sidesite="s_inside"/>
          <site site="s_dist_in"/>
        </spatial>
        <spatial name="t_straight" stiffness="2.0" damping="0.3">
          <site site="s_prox_top"/>
          <geom geom="g_cyl"/>
          <site site="s_dist_free"/>
        </spatial>
      </tendon>
      <actuator>
        <muscle tendon="t_sphere" force="95" lengthrange="0.15 0.65"/>
        <general tendon="t_cyl" dyntype="muscle" gaintype="muscle" biastype="muscle"
                 dynprm="0.01 0.04 0" gainprm="0.75 1.05 -1 85 0.5 1.6 1.5 1.3 1.2 0"
                 biasprm="0.75 1.05 -1 85 0.5 1.6 1.5 1.3 1.2 0"
                 lengthrange="0.08 0.55" actearly="true"/>
        <muscle tendon="t_inside" force="70" lengthrange="0.08 0.55"/>
        <muscle tendon="t_straight" force="60" lengthrange="0.15 0.65"/>
      </actuator>
    </mujoco>
    """
    _check_step_grads(self, xml, seed=44, mocap_grad=True)

  def test_post_step_fwd_kinematics_tendon_outputs(self):
    xml = """
    <mujoco>
      <option timestep="0.004" integrator="implicitfast" solver="Newton" iterations="20"/>
      <worldbody>
        <body name="mocap_obs" pos="0.14 0.02 0.5" mocap="true">
          <geom name="g_sphere" type="sphere" size="0.05" contype="0" conaffinity="0"/>
        </body>
        <body pos="0 0 0.5">
          <joint name="j0" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.28 0 0" size="0.025" mass="1.0" contype="0" conaffinity="0"/>
          <site name="s0" pos="0.05 0.04 0.01"/>
          <site name="s_mid" pos="0.24 0.04 0.02"/>
          <body pos="0.28 0 0">
            <joint name="j1" type="hinge" axis="0 1 0.3"/>
            <geom name="g_cyl" type="cylinder" size="0.04 0.07" contype="0" conaffinity="0" mass="0.3"/>
            <geom type="capsule" fromto="0 0 0 0.24 0 0" size="0.02" mass="0.7" contype="0" conaffinity="0"/>
            <site name="s1" pos="0.18 0.035 -0.02"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <fixed name="t_fix">
          <joint joint="j0" coef="0.05"/>
          <joint joint="j1" coef="-0.03"/>
        </fixed>
        <spatial name="t_wrap" stiffness="4.0" damping="0.6">
          <site site="s0"/>
          <geom geom="g_sphere"/>
          <site site="s_mid"/>
          <geom geom="g_cyl"/>
          <site site="s1"/>
        </spatial>
      </tendon>
      <actuator>
        <muscle tendon="t_fix" force="50" lengthrange="-0.2 0.4"/>
        <muscle tendon="t_wrap" force="80" lengthrange="0.15 0.7"/>
      </actuator>
    </mujoco>
    """
    _check_step_grads(self, xml, seed=55, mocap_grad=True, post_fk=True)

  def test_standalone_fwd_kinematics_tendon_vjp(self):
    xml = """
    <mujoco>
      <worldbody>
        <body name="mocap_obs" pos="0.15 0.01 0.5" mocap="true">
          <geom name="g_sphere" type="sphere" size="0.05" contype="0" conaffinity="0"/>
        </body>
        <body pos="0 0 0.5">
          <joint name="j0" type="hinge" axis="0 0 1"/>
          <geom size="0.025" mass="1.0" contype="0" conaffinity="0"/>
          <site name="s0" pos="0.05 0.04 0.01"/>
          <body pos="0.28 0 0">
            <joint name="j1" type="hinge" axis="0 1 0.2"/>
            <geom size="0.02" mass="0.7" contype="0" conaffinity="0"/>
            <site name="s1" pos="0.18 0.03 -0.01"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <fixed name="t_fix">
          <joint joint="j0" coef="0.04"/>
          <joint joint="j1" coef="-0.025"/>
        </fixed>
        <spatial name="t_sph">
          <site site="s0"/>
          <geom geom="g_sphere"/>
          <site site="s1"/>
        </spatial>
      </tendon>
    </mujoco>
    """
    mjw.enable_grad()
    mjm = mujoco.MjModel.from_xml_string(xml)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=1)
    qpos0 = np.array([[0.14, -0.22]], dtype=np.float32)
    mpos0 = np.array([[[0.15, 0.01, 0.5]]], dtype=np.float32)
    mquat0 = np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    d.qpos.assign(qpos0)
    d.mocap_pos.assign(mpos0)
    d.mocap_quat.assign(mquat0)
    d.mocap_pos.requires_grad = True
    d.mocap_quat.requires_grad = True

    w_len = np.array([[0.8, -1.3]], dtype=np.float32)
    w_J = np.array([[0.5, -0.7, 1.1, -0.9]], dtype=np.float32)
    w_len_wp = wp.array(w_len, dtype=float)
    w_J_wp = wp.array(w_J, dtype=float)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    bc = adjoint.create_backward_context(m, d)
    tape = wp.Tape()
    with adjoint.backward_context(bc):
      with tape:
        mjw.fwd_kinematics(m, d)
        wp.launch(_weighted_sum_2d, dim=(1, mjm.ntendon), inputs=[d.ten_length, w_len_wp], outputs=[loss])
        wp.launch(_weighted_sum_2d, dim=(1, m.nJten), inputs=[d.ten_J, w_J_wp], outputs=[loss])
      tape.backward(loss=loss)

    def eval_fk(qp, mp, mq):
      mjd = mujoco.MjData(mjm)
      mjd.qpos[:] = qp
      mjd.mocap_pos[:] = mp
      mjd.mocap_quat[:] = mq
      mujoco.mj_kinematics(mjm, mjd)
      mujoco.mj_comPos(mjm, mjd)
      mujoco.mj_tendon(mjm, mjd)
      return float(np.sum(mjd.ten_length * w_len[0]) + np.sum(mjd.ten_J * w_J[0]))

    eps = 1e-6
    qp_fd = np.zeros(2)
    for i in range(2):
      p = qpos0[0].astype(np.float64)
      m_ = p.copy()
      p[i] += eps
      m_[i] -= eps
      qp_fd[i] = (eval_fk(p, mpos0[0], mquat0[0]) - eval_fk(m_, mpos0[0], mquat0[0])) / (2.0 * eps)
    mp_fd = np.zeros((1, 3))
    for i in range(3):
      p = mpos0[0].astype(np.float64)
      m_ = p.copy()
      p[0, i] += eps
      m_[0, i] -= eps
      mp_fd[0, i] = (eval_fk(qpos0[0], p, mquat0[0]) - eval_fk(qpos0[0], m_, mquat0[0])) / (2.0 * eps)

    np.testing.assert_allclose(d.qpos.grad.numpy()[0], qp_fd, atol=1e-4, rtol=1e-3)
    np.testing.assert_allclose(d.mocap_pos.grad.numpy()[0], mp_fd, atol=1e-4, rtol=1e-3)

  def test_multistep_muscle_tendon_rollout_with_contact(self):
    xml = """
    <mujoco>
      <option timestep="0.004" integrator="implicitfast" solver="Newton" iterations="30"/>
      <worldbody>
        <geom type="plane" size="2 2 0.01"/>
        <body pos="0 0 0.22">
          <joint name="j0" type="hinge" axis="0 1 0" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 -0.08" size="0.025" mass="1.0" contype="0" conaffinity="0"/>
          <site name="s0" pos="0.04 0 0.04"/>
          <body pos="0.2 0 -0.08">
            <joint name="j1" type="hinge" axis="0 1 0" damping="0.08"/>
            <geom name="g_elbow" type="cylinder" axisangle="0 1 0 90" size="0.035 0.04" mass="0.3"
                  contype="0" conaffinity="0"/>
            <geom type="sphere" pos="0.18 0 -0.11" size="0.035" mass="0.6"/>
            <site name="s1" pos="0.15 0 0.03"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="biceps" stiffness="5.0" damping="0.6">
          <site site="s0"/>
          <geom geom="g_elbow"/>
          <site site="s1"/>
        </spatial>
      </tendon>
      <actuator>
        <muscle tendon="biceps" force="120" lengthrange="0.1 0.6"/>
      </actuator>
    </mujoco>
    """
    mjw.enable_grad()
    horizon = 4
    mjm = mujoco.MjModel.from_xml_string(xml)
    m = mjw.put_model(mjm)
    datas = [mjw.make_data(mjm, nworld=1) for _ in range(horizon + 1)]

    qpos0 = np.array([[0.12, 0.25]], dtype=np.float32)
    qvel0 = np.array([[0.2, -0.35]], dtype=np.float32)
    act0 = np.array([[0.45]], dtype=np.float32)
    ctrls_np = np.array([[[0.8]], [[0.3]], [[0.9]], [[0.2]]], dtype=np.float32)

    datas[0].qpos.assign(qpos0)
    datas[0].qvel.assign(qvel0)
    datas[0].act.assign(act0)
    ctrl_wps = [wp.array(ctrls_np[t], dtype=float, requires_grad=True) for t in range(horizon)]

    w_qpos = np.array([[1.3, -0.8]], dtype=np.float32)
    w_qvel = np.array([[-0.4, 0.7]], dtype=np.float32)
    w_act = np.array([[0.5]], dtype=np.float32)
    w_qpos_wp = wp.array(w_qpos, dtype=float)
    w_qvel_wp = wp.array(w_qvel, dtype=float)
    w_act_wp = wp.array(w_act, dtype=float)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    bc = adjoint.create_backward_context(m, datas[0])
    tape = wp.Tape()
    with adjoint.backward_context(bc):
      with tape:
        for t in range(horizon):
          datas[t].ctrl = ctrl_wps[t]
          mjw.step(m, datas[t], datas[t + 1])
        wp.launch(_weighted_sum_2d, dim=(1, mjm.nq), inputs=[datas[-1].qpos, w_qpos_wp], outputs=[loss])
        wp.launch(_weighted_sum_2d, dim=(1, mjm.nv), inputs=[datas[-1].qvel, w_qvel_wp], outputs=[loss])
        wp.launch(_weighted_sum_2d, dim=(1, mjm.na), inputs=[datas[-1].act, w_act_wp], outputs=[loss])
      tape.backward(loss=loss)

    def rollout_mj(qp, qv, ac, ct):
      mjd = mujoco.MjData(mjm)
      mjd.qpos[:] = qp
      mjd.qvel[:] = qv
      mjd.act[:] = ac
      for t in range(horizon):
        mjd.ctrl[:] = ct[t]
        mujoco.mj_step(mjm, mjd)
      return float(np.sum(mjd.qpos * w_qpos[0]) + np.sum(mjd.qvel * w_qvel[0]) + np.sum(mjd.act * w_act[0]))

    eps = 1e-6
    qp_fd = np.zeros(mjm.nq)
    for i in range(mjm.nq):
      p = qpos0[0].astype(np.float64)
      m_ = p.copy()
      p[i] += eps
      m_[i] -= eps
      qp_fd[i] = (rollout_mj(p, qvel0[0], act0[0], ctrls_np[:, 0]) - rollout_mj(m_, qvel0[0], act0[0], ctrls_np[:, 0])) / (
        2.0 * eps
      )
    qv_fd = np.zeros(mjm.nv)
    for i in range(mjm.nv):
      p = qvel0[0].astype(np.float64)
      m_ = p.copy()
      p[i] += eps
      m_[i] -= eps
      qv_fd[i] = (rollout_mj(qpos0[0], p, act0[0], ctrls_np[:, 0]) - rollout_mj(qpos0[0], m_, act0[0], ctrls_np[:, 0])) / (
        2.0 * eps
      )
    ac_fd = np.zeros(mjm.na)
    for i in range(mjm.na):
      p = act0[0].astype(np.float64)
      m_ = p.copy()
      p[i] += eps
      m_[i] -= eps
      ac_fd[i] = (rollout_mj(qpos0[0], qvel0[0], p, ctrls_np[:, 0]) - rollout_mj(qpos0[0], qvel0[0], m_, ctrls_np[:, 0])) / (
        2.0 * eps
      )
    ct_fd = np.zeros((horizon, mjm.nu))
    for t in range(horizon):
      for i in range(mjm.nu):
        p = ctrls_np[:, 0].astype(np.float64)
        m_ = p.copy()
        p[t, i] += eps
        m_[t, i] -= eps
        ct_fd[t, i] = (rollout_mj(qpos0[0], qvel0[0], act0[0], p) - rollout_mj(qpos0[0], qvel0[0], act0[0], m_)) / (2.0 * eps)

    np.testing.assert_allclose(datas[0].qpos.grad.numpy()[0], qp_fd, atol=3e-3, rtol=5e-3)
    np.testing.assert_allclose(datas[0].qvel.grad.numpy()[0], qv_fd, atol=3e-3, rtol=5e-3)
    np.testing.assert_allclose(datas[0].act.grad.numpy()[0], ac_fd, atol=3e-3, rtol=5e-3)
    ctrl_ad = np.stack([c.grad.numpy()[0] for c in ctrl_wps], axis=0)
    np.testing.assert_allclose(ctrl_ad, ct_fd, atol=3e-3, rtol=5e-3)

  @parameterized.product(integrator=("Euler", "implicitfast"), jacobian=("dense", "sparse"))
  def test_tendon_limits_lower_and_upper_fixed_and_spatial(self, integrator: str, jacobian: str):
    xml = f"""
    <mujoco>
      <option timestep="0.004" integrator="{integrator}" jacobian="{jacobian}" solver="Newton" iterations="30"/>
      <worldbody>
        <body name="mocap_anchor" pos="0 0.12 0.5" mocap="true">
          <geom name="g_wrap" type="sphere" size="0.04" contype="0" conaffinity="0"/>
          <site name="s_anchor" pos="0.02 -0.04 0.01"/>
        </body>
        <body pos="0 0 0.5">
          <joint name="j0" type="hinge" axis="0 0 1" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.025" mass="1.0" contype="0" conaffinity="0"/>
          <site name="s0" pos="0.12 0.03 0"/>
          <body pos="0.25 0 0">
            <joint name="j1" type="hinge" axis="0 1 0.2" damping="0.08"/>
            <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.02" mass="0.7" contype="0" conaffinity="0"/>
            <site name="s1" pos="0.16 0.02 -0.02"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <fixed name="t_fix_lim" limited="true" range="-0.015 0.015" margin="0.005" solreflimit="0.02 1.0">
          <joint joint="j0" coef="0.08"/>
          <joint joint="j1" coef="-0.05"/>
        </fixed>
        <spatial name="t_spa_lim" limited="true" range="0.24 0.29" margin="0.01" solreflimit="0.02 1.0">
          <site site="s_anchor"/>
          <geom geom="g_wrap"/>
          <site site="s0"/>
          <site site="s1"/>
        </spatial>
      </tendon>
      <actuator>
        <muscle tendon="t_spa_lim" force="75" lengthrange="0.15 0.55"/>
      </actuator>
    </mujoco>
    """
    # World 0 hits lower fixed/spatial limits; world 1 hits upper fixed/spatial limits.
    qpos = np.array([[-0.35, 0.20], [0.42, -0.25]], dtype=np.float32)
    qvel = np.array([[-0.30, 0.25], [0.35, -0.20]], dtype=np.float32)
    _check_step_grads(self, xml, seed=66, mocap_grad=True, qpos_override=qpos, qvel_override=qvel)

  @parameterized.product(integrator=("Euler", "implicitfast"), jacobian=("dense", "sparse"))
  def test_tendon_frictionloss_sticking_and_slipping(self, integrator: str, jacobian: str):
    xml = f"""
    <mujoco>
      <option timestep="0.004" integrator="{integrator}" jacobian="{jacobian}" solver="Newton" iterations="35"/>
      <worldbody>
        <body pos="0 0 0.5">
          <joint name="j0" type="hinge" axis="0 0 1" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 0.26 0 0" size="0.025" mass="1.0" contype="0" conaffinity="0"/>
          <site name="s0" pos="0.06 0.05 0.01"/>
          <body pos="0.26 0 0">
            <joint name="j1" type="hinge" axis="0 0.2 1" damping="0.05"/>
            <geom name="g_cyl" type="cylinder" size="0.035 0.06" mass="0.3" contype="0" conaffinity="0"/>
            <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.02" mass="0.7" contype="0" conaffinity="0"/>
            <site name="s1" pos="0.16 0.04 -0.01"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <fixed name="t_fix_fri" frictionloss="0.45">
          <joint joint="j0" coef="0.1"/>
          <joint joint="j1" coef="-0.06"/>
        </fixed>
        <spatial name="t_spa_fri" frictionloss="0.6" stiffness="2.0" damping="0.3">
          <site site="s0"/>
          <geom geom="g_cyl"/>
          <site site="s1"/>
        </spatial>
      </tendon>
      <actuator>
        <general joint="j0" gainprm="4.0"/>
        <general tendon="t_spa_fri" gainprm="5.0"/>
      </actuator>
    </mujoco>
    """
    # World 0 slips at high speed (LINEARPOS/LINEARNEG); world 1 sticks near zero (QUADRATIC).
    qpos = np.array([[0.22, -0.18], [-0.15, 0.20]], dtype=np.float32)
    qvel = np.array([[2.8, -2.4], [0.002, -0.001]], dtype=np.float32)
    ctrl = np.array([[0.8, -0.6], [0.02, -0.01]], dtype=np.float32)
    _check_step_grads(self, xml, seed=77, qpos_override=qpos, qvel_override=qvel, ctrl_override=ctrl)

    # Also verify m.tendon_frictionloss parameter gradient in the slipping regime.
    mjw.enable_grad()
    mjm = mujoco.MjModel.from_xml_string(xml)
    m = mjw.put_model(mjm)
    m.tendon_frictionloss.requires_grad = True
    d = mjw.make_data(mjm, nworld=1)
    d_out = mjw.make_data(mjm, nworld=1)
    d.qpos.assign(qpos[:1])
    d.qvel.assign(qvel[:1])
    d.ctrl.assign(ctrl[:1])
    w_qv = np.array([[1.2, -0.9]], dtype=np.float32)
    w_qv_wp = wp.array(w_qv, dtype=float)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    bc = adjoint.create_backward_context(m, d)
    tape = wp.Tape()
    with adjoint.backward_context(bc):
      with tape:
        mjw.step(m, d, d_out)
        wp.launch(_weighted_sum_2d, dim=(1, mjm.nv), inputs=[d_out.qvel, w_qv_wp], outputs=[loss])
      tape.backward(loss=loss)

    eps = 1e-5
    fl_fd = np.zeros(mjm.ntendon)
    base_fl = mjm.tendon_frictionloss.copy()
    for tid in range(mjm.ntendon):
      mjm.tendon_frictionloss[:] = base_fl
      mjm.tendon_frictionloss[tid] = base_fl[tid] + eps
      mjd_p = mujoco.MjData(mjm)
      mjd_p.qpos[:] = qpos[0]
      mjd_p.qvel[:] = qvel[0]
      mjd_p.ctrl[:] = ctrl[0]
      mujoco.mj_step(mjm, mjd_p)
      mjm.tendon_frictionloss[:] = base_fl
      mjm.tendon_frictionloss[tid] = base_fl[tid] - eps
      mjd_m = mujoco.MjData(mjm)
      mjd_m.qpos[:] = qpos[0]
      mjd_m.qvel[:] = qvel[0]
      mjd_m.ctrl[:] = ctrl[0]
      mujoco.mj_step(mjm, mjd_m)
      fl_fd[tid] = float(np.sum((mjd_p.qvel - mjd_m.qvel) * w_qv[0])) / (2.0 * eps)
    np.testing.assert_allclose(m.tendon_frictionloss.grad.numpy()[0], fl_fd, atol=1e-4, rtol=2e-3)

  @parameterized.product(integrator=("Euler", "implicitfast"), jacobian=("dense", "sparse"))
  def test_tendon_equality_single_and_coupled_polynomial(self, integrator: str, jacobian: str):
    xml = f"""
    <mujoco>
      <option timestep="0.004" integrator="{integrator}" jacobian="{jacobian}" solver="Newton" iterations="35"/>
      <worldbody>
        <body name="arm_a" pos="-0.2 0 0.5">
          <joint name="ja0" type="hinge" axis="0 0 1" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.025" mass="1.0" contype="0" conaffinity="0"/>
          <site name="sa0" pos="0.05 0.04 0.01"/>
          <body pos="0.25 0 0">
            <joint name="ja1" type="hinge" axis="0 0.3 1" damping="0.08"/>
            <geom name="g_cyl_a" type="cylinder" size="0.035 0.05" mass="0.3" contype="0" conaffinity="0"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.6" contype="0" conaffinity="0"/>
            <site name="sa1" pos="0.15 0.035 -0.01"/>
          </body>
        </body>
        <body name="arm_b" pos="0.2 0 0.5">
          <joint name="jb0" type="hinge" axis="0 1 0" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.24 0 0" size="0.025" mass="0.9" contype="0" conaffinity="0"/>
          <site name="sb0" pos="0.05 0.01 0.04"/>
          <body pos="0.24 0 0">
            <joint name="jb1" type="hinge" axis="0 1 0.2" damping="0.08"/>
            <geom name="g_sph_b" type="sphere" size="0.04" mass="0.3" contype="0" conaffinity="0"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.6" contype="0" conaffinity="0"/>
            <site name="sb1" pos="0.15 -0.01 0.035"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <spatial name="t_a">
          <site site="sa0"/>
          <geom geom="g_cyl_a"/>
          <site site="sa1"/>
        </spatial>
        <spatial name="t_b">
          <site site="sb0"/>
          <geom geom="g_sph_b"/>
          <site site="sb1"/>
        </spatial>
        <fixed name="t_single">
          <joint joint="ja0" coef="0.06"/>
          <joint joint="jb1" coef="-0.04"/>
        </fixed>
      </tendon>
      <equality>
        <tendon tendon1="t_single" polycoef="0.01 0 0 0 0" solref="0.02 1.0"/>
        <tendon tendon1="t_a" tendon2="t_b" polycoef="0.015 -0.65 0.45 -0.3 0.2" solref="0.02 1.0"/>
      </equality>
      <actuator>
        <muscle tendon="t_a" force="85" lengthrange="0.1 0.55"/>
        <muscle tendon="t_b" force="80" lengthrange="0.1 0.55"/>
      </actuator>
    </mujoco>
    """
    _check_step_grads(self, xml, seed=88)

  def test_unsupported_tendon_features_rejected(self):
    cases = [
      (
        "armature",
        """
        <mujoco><option solver="Newton"/><worldbody><body><joint name="j"/><geom size="0.1"/></body></worldbody>
        <tendon><fixed armature="0.05"><joint joint="j" coef="1"/></fixed></tendon></mujoco>
        """,
      ),
      (
        "actuator_armature",
        """
        <mujoco><option solver="Newton"/><worldbody><body><joint name="j"/><geom size="0.1"/></body></worldbody>
        <tendon><fixed name="t"><joint joint="j" coef="1"/></fixed></tendon>
        <actuator><general tendon="t" armature="0.05"/></actuator></mujoco>
        """,
      ),
      (
        "actfrclimited",
        """
        <mujoco><option solver="Newton"/><worldbody><body><joint name="j"/><geom size="0.1"/></body></worldbody>
        <tendon><fixed name="t" actuatorfrclimited="true" actuatorfrcrange="-1 1"><joint joint="j" coef="1"/></fixed></tendon>
        <actuator><general tendon="t"/></actuator></mujoco>
        """,
      ),
    ]
    for label, xml in cases:
      mjm = mujoco.MjModel.from_xml_string(xml)
      m = mjw.put_model(mjm)
      d = mjw.make_data(mjm, nworld=1)
      reasons = adjoint.step_supported(m, d)
      self.assertTrue(len(reasons) > 0, msg=f"expected {label} to be rejected")


if __name__ == "__main__":
  wp.init()
  absltest.main()
