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
"""Tests for smooth and ruled convex collision pose adjoints."""

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


def _normalize_free_quats(mjm: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
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


def _mjw_step_loss(
  m: mjw.Model,
  mjm: mujoco.MjModel,
  qpos: np.ndarray,
  qvel: np.ndarray,
  wq: np.ndarray,
  wv: np.ndarray,
) -> float:
  d = mjw.make_data(mjm, nworld=1)
  d_out = mjw.make_data(mjm, nworld=1)
  d.qpos.assign(_normalize_free_quats(mjm, qpos.astype(np.float32))[None, :])
  d.qvel.assign(qvel.astype(np.float32)[None, :])
  mjw.step(m, d, d_out)
  qp_out = d_out.qpos.numpy()[0].astype(np.float64)
  qv_out = d_out.qvel.numpy()[0].astype(np.float64)
  return float(np.sum(qp_out * wq) + np.sum(qv_out * wv))


def _mj_step_loss(
  mjm: mujoco.MjModel,
  qpos: np.ndarray,
  qvel: np.ndarray,
  wq: np.ndarray,
  wv: np.ndarray,
) -> float:
  mjd = mujoco.MjData(mjm)
  mjd.qpos[:] = _normalize_free_quats(mjm, qpos.astype(np.float64))
  mjd.qvel[:] = qvel.astype(np.float64)
  mujoco.mj_step(mjm, mjd)
  return float(np.sum(mjd.qpos * wq) + np.sum(mjd.qvel * wv))


def _project_free_quat_grad(mjm: mujoco.MjModel, qpos: np.ndarray, grad: np.ndarray) -> np.ndarray:
  out = grad.copy()
  for j in range(mjm.njnt):
    jt = mjm.jnt_type[j]
    adr = mjm.jnt_qposadr[j]
    if jt == mujoco.mjtJoint.mjJNT_FREE:
      q = qpos[adr + 3 : adr + 7]
      q = q / np.linalg.norm(q)
      g = out[adr + 3 : adr + 7]
      out[adr + 3 : adr + 7] = g - np.dot(g, q) * q
    elif jt == mujoco.mjtJoint.mjJNT_BALL:
      q = qpos[adr : adr + 4]
      q = q / np.linalg.norm(q)
      g = out[adr : adr + 4]
      out[adr : adr + 4] = g - np.dot(g, q) * q
  return out


def _make_pair_xml(
  geom0_xml: str,
  geom1_xml: str,
  pos0: str,
  euler0: str,
  pos1: str,
  euler1: str,
  cone: str = "pyramidal",
  integrator: str = "implicitfast",
  extra_asset: str = "",
) -> str:
  return f"""
  <mujoco>
    <option timestep="0.004" solver="Newton" iterations="40" ls_iterations="40"
            cone="{cone}" integrator="{integrator}" ccd_iterations="200" ccd_tolerance="1e-10"/>
    <asset>
      {extra_asset}
    </asset>
    <worldbody>
      <body name="b0" pos="{pos0}" euler="{euler0}">
        <freejoint name="j0"/>
        {geom0_xml}
      </body>
      <body name="b1" pos="{pos1}" euler="{euler1}">
        <freejoint name="j1"/>
        {geom1_xml}
      </body>
    </worldbody>
  </mujoco>
  """


class SmoothConvexCollisionAdjointTest(parameterized.TestCase):
  def _check_pair_adjoints(
    self,
    xml: str,
    qvel_scale: float = 0.25,
    rtol: float = 1.5e-2,
    atol: float = 2e-3,
    min_contacts: int = 2,
  ):
    mjw.enable_grad()
    mjm = mujoco.MjModel.from_xml_string(xml)
    # Keep tight float64 EPA tolerances on CPU for the finite-difference reference.
    m = mjw.put_model(mjm)
    mjm_ref = mujoco.MjModel.from_xml_string(xml)
    mjm_ref.opt.ccd_iterations = 400
    mjm_ref.opt.ccd_tolerance = 1e-12

    nworld = 2
    d = mjw.make_data(mjm, nworld=nworld)
    d_out = mjw.make_data(mjm, nworld=nworld)
    self.assertEqual(adjoint.step_supported(m, d), ())

    rng = np.random.default_rng(42)
    qpos = np.tile(mjm.qpos0.astype(np.float32), (nworld, 1))
    qpos[1] += 0.001 * rng.standard_normal(mjm.nq).astype(np.float32)
    qpos = _normalize_free_quats(mjm, qpos)
    qvel = (qvel_scale * rng.standard_normal((nworld, mjm.nv))).astype(np.float32)
    wq = rng.standard_normal((nworld, mjm.nq)).astype(np.float32)
    wv = (0.25 * rng.standard_normal((nworld, mjm.nv))).astype(np.float32)

    d.qpos.assign(qpos)
    d.qvel.assign(qvel)
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
    d_out.qpos.requires_grad = True
    d_out.qvel.requires_grad = True
    wq_wp = wp.array(wq, dtype=float)
    wv_wp = wp.array(wv, dtype=float)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    bc = adjoint.create_backward_context(m, d)
    tape = wp.Tape()
    with adjoint.backward_context(bc):
      with tape:
        mjw.step(m, d, d_out)
        wp.launch(_weighted_sum_2d, dim=(nworld, mjm.nq), inputs=[d_out.qpos, wq_wp], outputs=[loss])
        wp.launch(_weighted_sum_2d, dim=(nworld, mjm.nv), inputs=[d_out.qvel, wv_wp], outputs=[loss])
      tape.backward(loss=loss)

    self.assertGreaterEqual(int(d_out.nacon.numpy()[0]), min_contacts)

    ad_qpos = d.qpos.grad.numpy()
    ad_qvel = d.qvel.grad.numpy()

    eps = 1e-4
    for w in range(nworld):
      qp64 = qpos[w].astype(np.float64)
      qv64 = qvel[w].astype(np.float64)
      wq64 = wq[w].astype(np.float64)
      wv64 = wv[w].astype(np.float64)

      fd_qp = np.zeros(mjm.nq, dtype=np.float64)
      for i in range(mjm.nq):
        qp_p = qp64.copy()
        qp_m = qp64.copy()
        qp_p[i] += eps
        qp_m[i] -= eps
        lp = _mj_step_loss(mjm_ref, qp_p, qv64, wq64, wv64)
        lm = _mj_step_loss(mjm_ref, qp_m, qv64, wq64, wv64)
        fd_qp[i] = (lp - lm) / (2.0 * eps)

      fd_qv = np.zeros(mjm.nv, dtype=np.float64)
      for i in range(mjm.nv):
        qv_p = qv64.copy()
        qv_m = qv64.copy()
        qv_p[i] += eps
        qv_m[i] -= eps
        lp = _mj_step_loss(mjm_ref, qp64, qv_p, wq64, wv64)
        lm = _mj_step_loss(mjm_ref, qp64, qv_m, wq64, wv64)
        fd_qv[i] = (lp - lm) / (2.0 * eps)

      ad_qp_proj = _project_free_quat_grad(mjm, qp64, ad_qpos[w].astype(np.float64))
      fd_qp_proj = _project_free_quat_grad(mjm, qp64, fd_qp)
      grad_norm = max(float(np.linalg.norm(fd_qp_proj)), 1.0)
      cos = float(np.dot(ad_qp_proj, fd_qp_proj) / (np.linalg.norm(ad_qp_proj) * float(np.linalg.norm(fd_qp_proj))))
      rel_l2 = float(np.linalg.norm(ad_qp_proj - fd_qp_proj) / grad_norm)
      self.assertGreater(cos, 0.999)
      self.assertLess(rel_l2, 4.5e-2)
      np.testing.assert_allclose(ad_qp_proj, fd_qp_proj, rtol=3e-2, atol=max(atol, 2e-2 * grad_norm))
      np.testing.assert_allclose(ad_qvel[w], fd_qv, rtol=rtol, atol=max(atol, 5e-3 * float(np.linalg.norm(fd_qv))))

  @parameterized.named_parameters(
    (
      "capsule_barrel_ellipsoid_penetrating",
      '<geom type="capsule" size="0.06 0.18" margin="0.01" friction="0.7 0.005 0.0001"/>',
      '<geom type="ellipsoid" size="0.09 0.06 0.11" margin="0.01" friction="0.7 0.005 0.0001"/>',
      "0 0 0.5",
      "75 15 10",
      "0.148 0.01 0.51",
      "20 -35 40",
      "pyramidal",
      "implicitfast",
      2,
    ),
    (
      "capsule_cap_ellipsoid_elliptic",
      '<geom type="capsule" size="0.05 0.14" margin="0.008" friction="0.6 0.005 0.0001"/>',
      '<geom type="ellipsoid" size="0.08 0.10 0.07" margin="0.008" friction="0.6 0.005 0.0001"/>',
      "0 0 0.5",
      "10 5 0",
      "0.01 -0.01 0.758",
      "30 25 -15",
      "elliptic",
      "Euler",
      2,
    ),
    (
      "capsule_skew_barrel_cylinder_barrel",
      '<geom type="capsule" size="0.05 0.18" margin="0.008" friction="0.8 0.005 0.0001"/>',
      '<geom type="cylinder" size="0.07 0.16" margin="0.008" friction="0.8 0.005 0.0001"/>',
      "0 0 0.5",
      "85 10 25",
      "0.106 0.01 0.52",
      "15 70 -20",
      "pyramidal",
      "implicitfast",
      2,
    ),
    (
      "capsule_barrel_cylinder_rim",
      '<geom type="capsule" size="0.045 0.20" margin="0.006" friction="0.6 0.005 0.0001"/>',
      '<geom type="cylinder" size="0.08 0.12" margin="0.006" friction="0.6 0.005 0.0001"/>',
      "0 0 0.5",
      "90 0 20",
      "0.022 0.0 0.317",
      "0 25 0",
      "elliptic",
      "implicitfast",
      4,
    ),
    (
      "capsule_steep_barrel_cylinder_rim",
      '<geom type="capsule" size="0.045 0.20" margin="0.006" friction="0.6 0.005 0.0001"/>',
      '<geom type="cylinder" size="0.08 0.12" margin="0.006" friction="0.6 0.005 0.0001"/>',
      "0 0 0.5",
      "60 0 20",
      "0.022 0.0 0.317",
      "0 25 0",
      "elliptic",
      "implicitfast",
      2,
    ),
    (
      "capsule_cap_cylinder_flat_cap",
      '<geom type="capsule" size="0.04 0.12" margin="0.005" friction="0.5 0.005 0.0001"/>',
      '<geom type="cylinder" size="0.12 0.10" margin="0.005" friction="0.5 0.005 0.0001"/>',
      "0.02 -0.01 0.655",
      "12 -8 15",
      "0 0 0.40",
      "5 4 -10",
      "pyramidal",
      "implicitfast",
      2,
    ),
    (
      "capsule_barrel_on_cylinder_flat_cap_multiccd",
      '<geom type="capsule" size="0.04 0.08" margin="0.006" friction="0.5 0.005 0.0001"/>',
      '<geom type="cylinder" size="0.10 0.10" margin="0.006" friction="0.5 0.005 0.0001"/>',
      "0.055 0.01 0.539",
      "95 0 18",
      "0 0 0.40",
      "0 0 5",
      "pyramidal",
      "implicitfast",
      4,
    ),
    (
      "capsule_cap_cylinder_rim",
      '<geom type="capsule" size="0.05 0.12" margin="0.005" friction="0.5 0.005 0.0001"/>',
      '<geom type="cylinder" size="0.08 0.10" margin="0.005" friction="0.5 0.005 0.0001"/>',
      "0.068 -0.025 0.625",
      "20 -30 10",
      "0 0 0.40",
      "0 0 0",
      "pyramidal",
      "Euler",
      2,
    ),
    (
      "ellipsoid_cylinder_barrel",
      '<geom type="ellipsoid" size="0.08 0.06 0.10" margin="0.008" friction="0.7 0.005 0.0001"/>',
      '<geom type="cylinder" size="0.07 0.18" margin="0.008" friction="0.7 0.005 0.0001"/>',
      "0 0 0.5",
      "25 -20 35",
      "0.151 0.01 0.51",
      "10 15 -25",
      "elliptic",
      "implicitfast",
      2,
    ),
    (
      "ellipsoid_cylinder_flat_cap",
      '<geom type="ellipsoid" size="0.06 0.08 0.05" margin="0.006" friction="0.6 0.005 0.0001"/>',
      '<geom type="cylinder" size="0.14 0.10" margin="0.006" friction="0.6 0.005 0.0001"/>',
      "0.02 -0.015 0.548",
      "18 12 -22",
      "0 0 0.40",
      "-6 8 10",
      "pyramidal",
      "implicitfast",
      2,
    ),
    (
      "ellipsoid_cylinder_rim",
      '<geom type="ellipsoid" size="0.07 0.06 0.08" margin="0.006" friction="0.6 0.005 0.0001"/>',
      '<geom type="cylinder" size="0.08 0.10" margin="0.006" friction="0.6 0.005 0.0001"/>',
      "0.122 0.01 0.541",
      "30 -15 20",
      "0 0 0.40",
      "5 -5 12",
      "pyramidal",
      "Euler",
      2,
    ),
    (
      "sphere_ellipsoid",
      '<geom type="sphere" size="0.06" margin="0.005" friction="0.7 0.005 0.0001"/>',
      '<geom type="ellipsoid" size="0.09 0.06 0.11" margin="0.005" friction="0.7 0.005 0.0001"/>',
      "0 0 0.5",
      "0 0 0",
      "0.144 0.01 0.51",
      "28 -32 15",
      "pyramidal",
      "implicitfast",
      2,
    ),
    (
      "ellipsoid_ellipsoid",
      '<geom type="ellipsoid" size="0.08 0.05 0.09" margin="0.005" friction="0.7 0.005 0.0001"/>',
      '<geom type="ellipsoid" size="0.07 0.09 0.06" margin="0.005" friction="0.7 0.005 0.0001"/>',
      "0 0 0.5",
      "15 25 -10",
      "0.144 -0.01 0.51",
      "-20 35 25",
      "elliptic",
      "implicitfast",
      2,
    ),
  )
  def test_smooth_convex_pair_step_gradients(
    self, geom0_xml, geom1_xml, pos0, euler0, pos1, euler1, cone, integrator, min_contacts
  ):
    xml = _make_pair_xml(geom0_xml, geom1_xml, pos0, euler0, pos1, euler1, cone=cone, integrator=integrator)
    self._check_pair_adjoints(xml, min_contacts=min_contacts)

  def test_hfield_ellipsoid_step_gradients(self):
    mjw.enable_grad()
    xml = """
    <mujoco>
      <option timestep="0.004" solver="Newton" iterations="40" ls_iterations="40"
              cone="pyramidal" integrator="implicitfast" ccd_iterations="80" ccd_tolerance="1e-7"/>
      <asset>
        <hfield name="hf" nrow="4" ncol="4" size="0.5 0.5 0.1 0.1"
                elevation="0 0.1 0.2 0.1  0.05 0.15 0.25 0.15  0.1 0.2 0.3 0.2  0.05 0.1 0.15 0.1"/>
      </asset>
      <worldbody>
        <geom type="hfield" hfield="hf" margin="0.005" friction="0.6 0.005 0.0001"/>
        <body name="b" pos="0.03 -0.02 0.072" euler="20 -15 25">
          <freejoint name="j"/>
          <geom type="ellipsoid" size="0.08 0.06 0.05" margin="0.005" friction="0.6 0.005 0.0001"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=1)
    d_out = mjw.make_data(mjm, nworld=1)
    self.assertEqual(adjoint.step_supported(m, d), ())

    rng = np.random.default_rng(7)
    qpos = _normalize_free_quats(mjm, mjm.qpos0.astype(np.float32))
    qvel = (0.15 * rng.standard_normal(mjm.nv)).astype(np.float32)
    wq = rng.standard_normal((1, mjm.nq)).astype(np.float32)
    wv = (0.2 * rng.standard_normal((1, mjm.nv))).astype(np.float32)

    d.qpos.assign(qpos[None, :])
    d.qvel.assign(qvel[None, :])
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
    d_out.qpos.requires_grad = True
    d_out.qvel.requires_grad = True
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    bc = adjoint.create_backward_context(m, d)
    tape = wp.Tape()
    with adjoint.backward_context(bc):
      with tape:
        mjw.step(m, d, d_out)
        wp.launch(_weighted_sum_2d, dim=(1, mjm.nq), inputs=[d_out.qpos, wp.array(wq, dtype=float)], outputs=[loss])
        wp.launch(_weighted_sum_2d, dim=(1, mjm.nv), inputs=[d_out.qvel, wp.array(wv, dtype=float)], outputs=[loss])
      tape.backward(loss=loss)

    self.assertGreater(int(d_out.nacon.numpy()[0]), 0)
    ad_qp = _project_free_quat_grad(mjm, qpos.astype(np.float64), d.qpos.grad.numpy()[0].astype(np.float64))
    eps = 1e-4
    fd_qp = np.zeros(mjm.nq, dtype=np.float64)
    for i in range(mjm.nq):
      qp_p, qp_m = qpos.astype(np.float64), qpos.astype(np.float64)
      qp_p[i] += eps
      qp_m[i] -= eps
      lp = _mjw_step_loss(m, mjm, qp_p, qvel, wq[0], wv[0])
      lm = _mjw_step_loss(m, mjm, qp_m, qvel, wq[0], wv[0])
      fd_qp[i] = (lp - lm) / (2.0 * eps)
    fd_qp = _project_free_quat_grad(mjm, qpos.astype(np.float64), fd_qp)
    np.testing.assert_allclose(ad_qp, fd_qp, rtol=2e-2, atol=2e-3)


if __name__ == "__main__":
  wp.init()
  absltest.main()
