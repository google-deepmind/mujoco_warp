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
"""Gates for the reduced backward-only smooth-force replay (smooth_adjoint.py, MJPLAN_ADRNE §0/§14/§15).

P3 gate: the rne(flg_acc=True) replay's qpos VJP (smooth_force_backward) must
  (A) MATCH the adjoint.py manual oracle rne_qpos_vjp (same math, per-depth vs O(nbody^2) reductions); and
  (B) MATCH float64 MuJoCo-C central differences of L = lambda^T qfrc_bias wrt qpos (tangent space).
Over the §14 scene matrix: HINGE chain, FREE root, internal BALL, branching tree, a six-scalar-joint
body, tree depth > 32, a zero-joint (welded) body, and multiple worlds. CUDA + graph-capture gates run
on a CUDA box (this suite is CPU on the dev laptop); the math is device-independent.
"""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp._src import adjoint as _adjoint
from mujoco_warp._src import adjoint_test as _at  # reuse the RNE scenes + float64 FD helpers
from mujoco_warp._src import collision_adjoint as _collision_adjoint
from mujoco_warp._src import forward as _forward
from mujoco_warp._src import smooth as _smooth
from mujoco_warp._src import smooth_adjoint as _smooth_adjoint


def _chain_xml(n):
  """An n-link hinge chain (tree depth n) -- the depth>32 gate."""
  head = '<mujoco><option timestep="0.004" gravity="0 0 -9.81"/><worldbody>'
  body, tail = "", ""
  for i in range(n):
    ax = ("0 1 0", "1 0 0", "0 0 1")[i % 3]
    body += f'<body pos="0 0 {-0.12 if i else 0.6}"><joint type="hinge" axis="{ax}"/>'
    body += f'<geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.02" mass="{0.3 + 0.01 * i}"/>'
    tail += "</body>"
  return head + body + tail + "</worldbody></mujoco>"


# A single body carrying six scalar joints (3 slide + 3 hinge = body_dofnum 6, the max) + a child, mixing
# same-body joint types -- the six-slot / mixed-same-body gate.
_RNE_SIXJOINT = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0.6">
      <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
      <joint type="hinge" axis="1 0 0"/><joint type="hinge" axis="0 1 0"/><joint type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.08 0.06 0.05" mass="1.2"/>
      <body pos="0.1 0 0"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.03" mass="0.4"/></body></body>
  </worldbody></mujoco>
"""

# A hinge body with a rigidly-attached (NO joint) child body -- the zero-joint-body gate.
_RNE_ZEROJOINT = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0.7"><joint type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.08 0.06 0.05" mass="1.5"/>
      <body pos="0.12 0 0"><geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03" mass="0.5"/>
        <body pos="0.2 0 0"><joint type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.04" mass="0.3"/></body></body></body>
  </worldbody></mujoco>
"""

_SCENES = dict(_at._RNE_SCENES)
_SCENES["sixjoint"] = _RNE_SIXJOINT
_SCENES["zerojoint"] = _RNE_ZEROJOINT
_SCENES["deep40"] = _chain_xml(40)  # tree depth 40 > 32 (no artificial _CV_MAX_DEPTH)


def _unlift_to_tangent(mjm, qpos, rq):
  """Map an nq-indexed qpos cotangent rq to an nv-indexed tangent vector (invert _dof_to_qpos)."""
  nv = mjm.nv
  out = np.zeros(nv)
  for j in range(mjm.njnt):
    jt, qa, da = int(mjm.jnt_type[j]), mjm.jnt_qposadr[j], mjm.jnt_dofadr[j]
    if jt == _at._RNE_FREE:
      out[da:da + 3] = rq[qa:qa + 3]
      out[da + 3:da + 6] = _at._rne_unlift(qpos[qa + 3:qa + 7], rq[qa + 3:qa + 7])
    elif jt == _at._RNE_BALL:
      out[da:da + 3] = _at._rne_unlift(qpos[qa:qa + 4], rq[qa:qa + 4])
    else:
      out[da] = rq[qa]
  return out


# Spring scenes (contact-free, damping=0 -> qfrc_passive == qfrc_spring). springref defaults to qpos0, so
# bending qpos gives a nonzero spring force.
_SPRING_HINGE = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0.6"><joint type="hinge" axis="0 1 0" stiffness="4"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.6"/>
      <body pos="0 0 -0.3"><joint type="slide" axis="0 0 1" stiffness="7"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.025" mass="0.4"/></body></body>
  </worldbody></mujoco>
"""
_SPRING_FREE = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody><body pos="0 0 0.6"><joint type="free" stiffness="5"/>
    <geom type="box" size="0.1 0.15 0.08" mass="1.2"/></body></worldbody></mujoco>
"""
_SPRING_BALL = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0.6"><joint type="hinge" axis="0 1 0" stiffness="2"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03" mass="0.6"/>
      <body pos="0.2 0 0"><joint type="ball" stiffness="6"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.025" mass="0.4"/></body></body>
  </worldbody></mujoco>
"""
_SPRING_SCENES = {"hinge_slide": _SPRING_HINGE, "free": _SPRING_FREE, "ball": _SPRING_BALL}


def _spring_fd_qpos_mj(mjm, qpos, lam, eps=1e-6):
  """float64 MuJoCo-C FD of L = λᵀ(-qfrc_spring) wrt qpos in TANGENT space (mj_integratePos). Models have
  damping=0 and no gravcomp/fluid, so qfrc_passive == qfrc_spring."""
  nv = mjm.nv
  out = np.zeros(nv)
  for k in range(nv):
    vals = []
    for s in (+1.0, -1.0):
      qp = qpos.copy()
      e = np.zeros(nv)
      e[k] = 1.0
      mujoco.mj_integratePos(mjm, qp, e, s * eps)
      d2 = mujoco.MjData(mjm)
      d2.qpos[:] = qp
      mujoco.mj_forward(mjm, d2)
      vals.append(-lam @ d2.qfrc_passive)
    out[k] = (vals[0] - vals[1]) / (2.0 * eps)
  return out


class SmoothSpringTest(parameterized.TestCase):
  """smooth_adjoint.spring_qpos_vjp (complete FREE/BALL/SLIDE/HINGE source-AD spring leaf) vs float64
  MuJoCo-C FD of L = λᵀ(-qfrc_spring). The FREE/BALL quaternion springs are the new coverage the
  HINGE/SLIDE-only adjoint.py oracle lacked (MJPLAN_ADRNE §0.2)."""

  @parameterized.parameters(*_SPRING_SCENES.keys())
  def test_spring_qpos(self, name):
    mjm = mujoco.MjModel.from_xml_string(_SPRING_SCENES[name])
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    nv = mjm.nv
    m = mjw.put_model(mjm)
    rng = np.random.default_rng(0)
    qpos0 = mjd.qpos.copy()
    for t in range(3):
      qpos = qpos0.copy()
      mujoco.mj_integratePos(mjm, qpos, rng.standard_normal(nv) * 0.5, 1.0)
      lam = rng.standard_normal(nv)
      d = mjw.put_data(mjm, mjd)
      wp.copy(d.qpos, wp.array(qpos.reshape(1, -1).astype(np.float32), dtype=wp.float32))
      lam_wp = wp.array(lam.reshape(1, -1).astype(np.float32), dtype=wp.float32)

      shadow = _smooth_adjoint.spring_qpos_vjp(m, d, lam_wp).numpy()[0]
      ana_q = _unlift_to_tangent(mjm, qpos, shadow)
      fd_q = _spring_fd_qpos_mj(mjm, qpos, lam)
      cos, rel, nf = _at._rne_metrics(ana_q, fd_q)
      self.assertGreater(nf, 1e-6, f"{name}[{t}]: spring dL/dq ~0 (not exercised)")
      self.assertGreater(cos, 0.999, f"{name}[{t}]: spring vs FD direction off, cos={cos:.5f}")
      self.assertLess(rel, 1e-2, f"{name}[{t}]: spring vs FD magnitude off, rel={rel:.2e}")


# Actuator scenes (contact-free): AFFINE joint-transmission actuators on slide/hinge joints.
_ACT_POS = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0.6"><joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.6"/>
      <body pos="0 0 -0.3"><joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025" mass="0.4"/></body></body>
  </worldbody>
  <actuator>
    <position joint="j0" kp="9" kv="0.7" ctrlrange="-1 1"/>
    <position joint="j1" kp="5" kv="0.4"/>
  </actuator></mujoco>
"""


def _actuator_fd_qpos_mj(mjm, qpos, ctrl, lam, eps=1e-6):
  """float64 MuJoCo-C FD of L = λᵀ(-qfrc_actuator) wrt qpos in TANGENT space (mj_integratePos)."""
  nv = mjm.nv
  out = np.zeros(nv)
  for k in range(nv):
    vals = []
    for s in (+1.0, -1.0):
      qp = qpos.copy()
      e = np.zeros(nv)
      e[k] = 1.0
      mujoco.mj_integratePos(mjm, qp, e, s * eps)
      d2 = mujoco.MjData(mjm)
      d2.qpos[:] = qp
      d2.ctrl[:] = ctrl
      mujoco.mj_forward(mjm, d2)
      vals.append(-lam @ d2.qfrc_actuator)
    out[k] = (vals[0] - vals[1]) / (2.0 * eps)
  return out


class SmoothActuatorTest(parameterized.TestCase):
  """smooth_adjoint.actuator_qpos_vjp (affine JOINT-transmission staged reverse) vs float64 MuJoCo-C FD of
  L = λᵀ(-qfrc_actuator). Position servos (FIXED gain + AFFINE bias) -- the G1/reacher control mode."""

  def test_actuator_qpos(self):
    mjm = mujoco.MjModel.from_xml_string(_ACT_POS)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    nv = mjm.nv
    m = mjw.put_model(mjm)
    rng = np.random.default_rng(0)
    qpos0 = mjd.qpos.copy()
    for t in range(3):
      qpos = qpos0 + rng.standard_normal(nv) * 0.4
      ctrl = rng.standard_normal(mjm.nu) * 0.5
      lam = rng.standard_normal(nv)
      d = mjw.put_data(mjm, mjd)
      wp.copy(d.qpos, wp.array(qpos.reshape(1, -1).astype(np.float32), dtype=wp.float32))
      wp.copy(d.ctrl, wp.array(ctrl.reshape(1, -1).astype(np.float32), dtype=wp.float32))
      _smooth.kinematics(m, d)
      _smooth.com_pos(m, d)
      _smooth.transmission(m, d)
      _smooth.com_vel(m, d)
      _forward.fwd_actuation(m, d)
      lam_wp = wp.array(lam.reshape(1, -1).astype(np.float32), dtype=wp.float32)
      ana = _unlift_to_tangent(mjm, qpos, _smooth_adjoint.actuator_qpos_vjp(m, d, lam_wp).numpy()[0])
      fd = _actuator_fd_qpos_mj(mjm, qpos, ctrl, lam)
      cos, rel, nf = _at._rne_metrics(ana, fd)
      self.assertGreater(nf, 1e-6, f"[{t}]: actuator dL/dq ~0 (not exercised)")
      self.assertGreater(cos, 0.999, f"[{t}]: actuator vs FD direction off, cos={cos:.5f}")
      self.assertLess(rel, 1e-2, f"[{t}]: actuator vs FD magnitude off, rel={rel:.2e}")

  def test_actuator_qpos_stateful(self):
    """Activation (na>0): an intvelocity servo (dyntype=INTEGRATOR, na=1, FIXED gain, AFFINE bias) -- the
    supported stateful case where dfdl=biasprm[1] and ctrl_act(=act) is unused. assert_smooth_supported
    must accept it; the ∂qpos must match float64 FD with the activation state frozen."""
    xml = """
    <mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
      <worldbody><body pos="0 0 0.6"><joint name="j0" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.6"/></body></worldbody>
      <actuator><intvelocity joint="j0" kp="9" actrange="-3 3"/></actuator></mujoco>"""
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    nv = mjm.nv
    m = mjw.put_model(mjm)
    self.assertEqual(int(m.na), 1)
    _smooth_adjoint.assert_smooth_supported(m)  # must NOT raise (stateful + FIXED gain)
    rng = np.random.default_rng(0)
    for t in range(3):
      qpos = mjd.qpos.copy() + rng.standard_normal(nv) * 0.4
      act = rng.standard_normal(mjm.na) * 0.8
      ctrl = rng.standard_normal(mjm.nu) * 0.5
      lam = rng.standard_normal(nv)
      d = mjw.put_data(mjm, mjd)
      wp.copy(d.qpos, wp.array(qpos.reshape(1, -1).astype(np.float32), dtype=wp.float32))
      wp.copy(d.act, wp.array(act.reshape(1, -1).astype(np.float32), dtype=wp.float32))
      wp.copy(d.ctrl, wp.array(ctrl.reshape(1, -1).astype(np.float32), dtype=wp.float32))
      _smooth.kinematics(m, d)
      _smooth.com_pos(m, d)
      _smooth.transmission(m, d)
      _smooth.com_vel(m, d)
      _forward.fwd_actuation(m, d)
      lam_wp = wp.array(lam.reshape(1, -1).astype(np.float32), dtype=wp.float32)
      ana = _unlift_to_tangent(mjm, qpos, _smooth_adjoint.actuator_qpos_vjp(m, d, lam_wp).numpy()[0])

      def L(qp):
        d2 = mujoco.MjData(mjm)
        d2.qpos[:] = qp
        d2.act[:] = act
        d2.ctrl[:] = ctrl
        mujoco.mj_forward(mjm, d2)
        return -lam @ d2.qfrc_actuator

      fd = np.zeros(nv)
      eps = 1e-6
      for k in range(nv):
        qp = qpos.copy()
        e = np.zeros(nv)
        e[k] = 1.0
        mujoco.mj_integratePos(mjm, qp, e, eps)
        Lp = L(qp)
        qp = qpos.copy()
        mujoco.mj_integratePos(mjm, qp, e, -eps)
        fd[k] = (Lp - L(qp)) / (2.0 * eps)
      cos, rel, nf = _at._rne_metrics(ana, fd)
      self.assertGreater(nf, 1e-6, f"[{t}]: stateful actuator dL/dq ~0")
      self.assertGreater(cos, 0.999, f"[{t}]: stateful actuator vs FD, cos={cos:.5f}")
      self.assertLess(rel, 1e-2, f"[{t}]: stateful actuator vs FD, rel={rel:.2e}")


_GRAVCOMP = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0.6" gravcomp="1"><joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.6"/>
      <body pos="0.3 0 0" gravcomp="0.7"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.025" mass="0.4"/></body></body>
  </worldbody></mujoco>
"""


class SmoothGravcompTest(parameterized.TestCase):
  """smooth_adjoint.gravcomp_qpos_vjp (passive-bucket gravity compensation) vs float64 MuJoCo-C FD of
  L = λᵀ(-qfrc_gravcomp). No springs/damping/fluid -> qfrc_passive == qfrc_gravcomp. Source-AD of
  passive._gravity_force's jac_dof contraction routed through the shared kinematic reverse."""

  def test_gravcomp_qpos(self):
    mjm = mujoco.MjModel.from_xml_string(_GRAVCOMP)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    nv = mjm.nv
    m = mjw.put_model(mjm)
    _smooth_adjoint.assert_smooth_supported(m)  # passive-bucket gravcomp must be accepted
    rng = np.random.default_rng(0)
    for t in range(3):
      qpos = mjd.qpos.copy() + rng.standard_normal(nv) * 0.4
      lam = rng.standard_normal(nv)
      d = mjw.put_data(mjm, mjd)
      wp.copy(d.qpos, wp.array(qpos.reshape(1, -1).astype(np.float32), dtype=wp.float32))
      _smooth.kinematics(m, d)
      _smooth.com_pos(m, d)
      lam_wp = wp.array(lam.reshape(1, -1).astype(np.float32), dtype=wp.float32)
      ana = _unlift_to_tangent(mjm, qpos, _smooth_adjoint.gravcomp_qpos_vjp(m, d, lam_wp).numpy()[0])

      def L(qp):
        d2 = mujoco.MjData(mjm)
        d2.qpos[:] = qp
        mujoco.mj_forward(mjm, d2)
        return -lam @ d2.qfrc_passive

      fd = np.zeros(nv)
      eps = 1e-6
      for k in range(nv):
        qp = qpos.copy()
        e = np.zeros(nv)
        e[k] = 1.0
        mujoco.mj_integratePos(mjm, qp, e, eps)
        Lp = L(qp)
        qp = qpos.copy()
        mujoco.mj_integratePos(mjm, qp, e, -eps)
        fd[k] = (Lp - L(qp)) / (2.0 * eps)
      cos, rel, nf = _at._rne_metrics(ana, fd)
      self.assertGreater(nf, 1e-6, f"[{t}]: gravcomp dL/dq ~0 (not exercised)")
      self.assertGreater(cos, 0.999, f"[{t}]: gravcomp vs FD direction off, cos={cos:.5f}")
      self.assertLess(rel, 1e-2, f"[{t}]: gravcomp vs FD magnitude off, rel={rel:.2e}")


# End-to-end scene: hinge arm with joint springs + position actuators (contact-free). step_backward's
# analytic §5 (smooth_adjoint: bias+spring+actuator) must match FD-of-mjw.step and the FD-of-rne path.
_E2E = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0.6"><joint name="j0" type="hinge" axis="0 1 0" stiffness="3"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.6"/>
      <body pos="0 0 -0.3"><joint name="j1" type="hinge" axis="0 1 0" stiffness="2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025" mass="0.4"/></body></body>
  </worldbody>
  <actuator><position joint="j0" kp="8" kv="0.6"/><position joint="j1" kp="5" kv="0.4"/></actuator></mujoco>
"""
# Gravcomp end-to-end scene: under-compensated (gravcomp=0.5) so the dynamics still move qpos.
_E2E_GRAVCOMP = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0.6" gravcomp="0.5"><joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.6"/>
      <body pos="0.3 0 0" gravcomp="0.5"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.025" mass="0.4"/></body></body>
  </worldbody></mujoco>
"""


class SmoothStepBackwardTest(parameterized.TestCase):
  """P7 end-to-end: step_backward's wired analytic §5 (smooth_adjoint bias+spring+actuator+gravcomp) vs
  (A) the retained FD-of-rne path [identical step_backward modulo §5] and (B) float64 FD-of-mjw.step, over
  a BPTT horizon. A per-step bias (or a sign error) would show here. FD path is the explicit test oracle."""

  @parameterized.named_parameters(
    # rel_tol is the analytic-vs-FD-of-rne tolerance; the FD-of-rne reference uses eps=1e-4 float32, so its
    # own truncation sets the floor (gravcomp's jac_dof force FD is noisier). The AUTHORITATIVE correctness
    # gate is the float64-MuJoCo-C standalone test; here cos_r (direction) is the tight check.
    ("spring_actuator", _E2E, (1, 8, 16, 32), 1e-2),
    ("gravcomp", _E2E_GRAVCOMP, (1, 8, 16), 3.5e-2),
  )
  def test_bptt(self, xml, horizons, rel_tol):
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    nv = mjm.nv
    rng = np.random.default_rng(0)
    qpos0 = mjd.qpos.copy() + rng.standard_normal(nv) * 0.3

    prev = _adjoint._USE_ANALYTIC_RNE_QPOS
    try:
      for T in horizons:
        _adjoint._USE_ANALYTIC_RNE_QPOS = True
        ana = _at._rne_bptt_analytic(mjm, mjd, T, qpos0)  # analytic (smooth_adjoint via step_backward §5)
        _adjoint._USE_ANALYTIC_RNE_QPOS = False
        ref = _at._rne_bptt_analytic(mjm, mjd, T, qpos0)  # FD-of-rne path (same step_backward modulo §5)
        fd = _at._rne_bptt_fd(mjm, mjd, T, qpos0)  # float64 FD-of-mjw.step (gold)
        nar, nrr, nf = np.linalg.norm(ana), np.linalg.norm(ref), np.linalg.norm(fd)
        cos_r = float(ana @ ref / (nar * nrr)) if nar > 1e-12 and nrr > 1e-12 else float("nan")
        rel_r = float(np.linalg.norm(ana - ref) / (nrr + 1e-12))
        cos_f = float(ana @ fd / (nar * nf)) if nar > 1e-12 and nf > 1e-12 else float("nan")
        self.assertGreater(nrr, 1e-6, f"T={T}: gradient ~0 (scene not exercising qpos)")
        self.assertGreater(cos_r, 0.9999, f"T={T}: analytic != FD-of-rne, cos={cos_r:.6f} (per-step bias)")
        self.assertLess(rel_r, rel_tol, f"T={T}: analytic != FD-of-rne, rel={rel_r:.2e} (per-step bias)")
        self.assertGreater(cos_f, 0.99, f"T={T}: analytic vs FD-of-step direction off, cos={cos_f:.4f}")
    finally:
      _adjoint._USE_ANALYTIC_RNE_QPOS = prev


class SmoothReplayRneTest(parameterized.TestCase):
  """smooth_adjoint.smooth_force_backward (reduced rne replay) vs the adjoint.py oracle + float64 mj_rne."""

  @parameterized.parameters(*_SCENES.keys())
  def test_rne_replay(self, name):
    mjm = mujoco.MjModel.from_xml_string(_SCENES[name])
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    nv = mjm.nv
    m = mjw.put_model(mjm)
    rng = np.random.default_rng(0)
    qpos0 = mjd.qpos.copy()

    for t in range(4):
      qpos = qpos0.copy()
      mujoco.mj_integratePos(mjm, qpos, rng.standard_normal(nv) * 0.6, 1.0)  # quaternion-safe bend
      qvel = rng.standard_normal(nv) * 0.8
      qacc = rng.standard_normal(nv) * 0.8
      lam = rng.standard_normal(nv)

      d = mjw.put_data(mjm, mjd)
      _at._rne_fwd_mjw(m, d, qpos, qvel, qacc)
      lam_wp = wp.array(lam.reshape(1, -1).astype(np.float32), dtype=wp.float32)

      shadow = _smooth_adjoint.smooth_force_backward(m, d, lam_wp, flg_acc=True).numpy()[0]
      oracle = _smooth_adjoint.rne_qpos_vjp(m, d, lam_wp, flg_acc=True).numpy()[0]

      # (A) shadow must MATCH the manual oracle (identical math; reductions differ only in schedule).
      cos, rel, nf = _at._rne_metrics(shadow, oracle)
      if np.linalg.norm(oracle) > 1e-8:
        self.assertGreater(cos, 0.999990, f"{name}[{t}]: shadow != oracle, cos={cos:.7f}")
        self.assertLess(rel, 1e-4, f"{name}[{t}]: shadow != oracle, rel={rel:.2e}")
      else:
        self.assertLess(float(np.linalg.norm(shadow)), 1e-5, f"{name}[{t}]: shadow nonzero vs ~0 oracle")

      # (B) vs float64 mj_rne FD (un-lifted to tangent); first bent config only (expensive).
      if t == 0:
        ana_q = _unlift_to_tangent(mjm, qpos, shadow)
        fd_q = _at._rne_fd_qpos_mj(mjm, qpos, qvel, qacc, lam)
        if float(np.linalg.norm(fd_q)) < 1e-6:  # degenerate (single free body: qfrc_bias is q-invariant)
          self.assertLess(float(np.linalg.norm(ana_q)), 1e-4, f"{name}: dL/dq should be ~0")
        else:
          cos, rel, _nf = _at._rne_metrics(ana_q, fd_q)
          self.assertGreater(cos, 0.999, f"{name}: shadow vs mj_rne FD direction off, cos={cos:.5f}")
          self.assertLess(rel, 1e-2, f"{name}: shadow vs mj_rne FD magnitude off, rel={rel:.2e}")

  def test_multiworld(self):
    """Batched parity: the shared replay's per-depth reductions must be correct across nworld worlds with
    DISTINCT states (catches any cross-world aliasing in the atomic per-depth accumulation)."""
    mjm = mujoco.MjModel.from_xml_string(_at._RNE_WORM)  # FREE + BALL + HINGE
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    nv, nq = mjm.nv, mjm.nq
    nworld = 4
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld)
    rng = np.random.default_rng(1)

    qpos = np.zeros((nworld, nq))
    qvel = rng.standard_normal((nworld, nv)) * 0.8
    qacc = rng.standard_normal((nworld, nv)) * 0.8
    lam = rng.standard_normal((nworld, nv))
    for wd in range(nworld):
      qp = mjd.qpos.copy()
      mujoco.mj_integratePos(mjm, qp, rng.standard_normal(nv) * 0.6, 1.0)
      qpos[wd] = qp

    wp.copy(d.qpos, wp.array(qpos.astype(np.float32), dtype=wp.float32))
    wp.copy(d.qvel, wp.array(qvel.astype(np.float32), dtype=wp.float32))
    wp.copy(d.qacc, wp.array(qacc.astype(np.float32), dtype=wp.float32))
    _smooth.kinematics(m, d)
    _smooth.com_pos(m, d)
    _smooth.com_vel(m, d)
    _smooth.rne(m, d, flg_acc=True)
    lam_wp = wp.array(lam.astype(np.float32), dtype=wp.float32)

    shadow = _smooth_adjoint.smooth_force_backward(m, d, lam_wp, flg_acc=True).numpy()
    oracle = _smooth_adjoint.rne_qpos_vjp(m, d, lam_wp, flg_acc=True).numpy()
    for wd in range(nworld):
      cos, rel, _nf = _at._rne_metrics(shadow[wd], oracle[wd])
      self.assertGreater(cos, 0.999990, f"world {wd}: shadow != oracle, cos={cos:.7f}")
      self.assertLess(rel, 1e-4, f"world {wd}: shadow != oracle, rel={rel:.2e}")


class SmoothContactMergeTest(parameterized.TestCase):
  """P4: the contact res_cdof/res_subtree_com seeds merge into the bias's shared kinematic reverse.
  Validated on the worm (FREE+BALL+HINGE, nacon=0 -> contact_qpos_vjp's narrowphase no-ops, leaving its
  cdof/subtree reverse as the oracle). Random synthetic seeds exercise the plumbing (the cdof/COM kernels
  are linear in the seeds; the values are validated elsewhere by FD)."""

  def test_contact_seed_merge(self):
    mjm = mujoco.MjModel.from_xml_string(_at._RNE_WORM)
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    nv, nq, nbody = mjm.nv, mjm.nq, mjm.nbody
    m = mjw.put_model(mjm)
    rng = np.random.default_rng(2)
    qpos = mjd.qpos.copy()
    mujoco.mj_integratePos(mjm, qpos, rng.standard_normal(nv) * 0.6, 1.0)
    d = mjw.put_data(mjm, mjd)
    _at._rne_fwd_mjw(m, d, qpos, rng.standard_normal(nv) * 0.8, rng.standard_normal(nv) * 0.8)

    lam = wp.array(rng.standard_normal((1, nv)).astype(np.float32), dtype=wp.float32)
    zero_lam = wp.zeros((1, nv), dtype=float)
    Rc = wp.array(rng.standard_normal((1, nv, 6)).astype(np.float32), dtype=wp.spatial_vector)
    Rs = wp.array(rng.standard_normal((1, nbody, 3)).astype(np.float32), dtype=wp.vec3)

    merged = _smooth_adjoint.smooth_force_backward(m, d, lam, res_cdof_extra=Rc, res_subtree_extra=Rs).numpy()[0]
    bias_only = _smooth_adjoint.smooth_force_backward(m, d, lam).numpy()[0]
    contact_only = _smooth_adjoint.smooth_force_backward(m, d, zero_lam, res_cdof_extra=Rc, res_subtree_extra=Rs).numpy()[0]

    # (1) linearity: merging the contact seeds == summing the separate bias and contact-seed reverses.
    cos, rel, _ = _at._rne_metrics(merged, bias_only + contact_only)
    self.assertGreater(cos, 0.9999990, f"merge not linear, cos={cos:.7f}")
    self.assertLess(rel, 1e-5, f"merge not linear, rel={rel:.2e}")

    # (2) the contact-seed-only reverse matches collision_adjoint.contact_qpos_vjp's cdof/subtree path
    # (zeroed narrowphase pose seeds; nacon=0 makes the narrowphase a no-op). This is the "one common
    # kinematic reverse matches the existing contact-only analytic VJP" gate (§15.5).
    ncon = d.contact.pos.shape[0]
    res_qpos_oracle = wp.zeros((1, nq), dtype=float)
    _collision_adjoint.contact_qpos_vjp(
      m, d,
      wp.zeros(ncon, dtype=wp.vec3),  # res_contact_pos
      wp.zeros(ncon, dtype=wp.mat33),  # res_contact_frame
      wp.zeros_like(d.efc.pos),  # res_efc_pos
      Rs,  # res_subtree_com
      Rc,  # res_cdof
      res_qpos_oracle,
    )
    cos, rel, _ = _at._rne_metrics(contact_only, res_qpos_oracle.numpy()[0])
    self.assertGreater(cos, 0.9999990, f"contact-seed reverse != contact_qpos_vjp, cos={cos:.7f}")
    self.assertLess(rel, 1e-5, f"contact-seed reverse != contact_qpos_vjp, rel={rel:.2e}")


if __name__ == "__main__":
  wp.init()
  absltest.main()
