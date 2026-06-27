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
"""Closed-form / gradient-usefulness tests for ``adjoint.py`` (MJPLAN.md §12.2).

Sibling to ``adjoint_test.py`` (whose oracles are all finite-difference). This file
was substantially reworked after an adversarial review; the design here reflects
its (correct) findings:

CORRECTNESS oracle for contact (decided with the user): there is NO clean
simulator-independent closed form for a *soft, discrete* contact -- its gradient is
dominated by a fixed-step contact-phase (discretization) artifact, NOT the ideal
hard-impact sensitivity. E.g. for the floor bounce d(z_N)/d(z0) is ~-1.8, whereas
the continuous identity is -e_hat (~-0.8); the difference is the grid-phase
artifact, not "soft-contact physics." So our correctness signal is
**adjoint == FD**, made rigorous via (i) an active-set-stable epsilon BRACKET (the
contact step pattern must be identical across +/-eps and the FD value must plateau)
and (ii) agreement with a NATIVE MuJoCo (mujoco.mj_step) rollout FD. The genuinely
simulator-independent closed forms are the two-ball collision (Zhong 2207.05060
App. C) and the Heaviside reference (pure analysis).

USEFULNESS (Suh 2202.00817; AHAC 2405.17784): first-order (FoBG, from our adjoint)
vs zeroth-order (ZoBG) estimators of grad_theta E_w[L(theta+w)], w~N(0,sigma^2). NO
neural net, NO SHAC loop -- a = theta + w. Corrections vs the first draft:
  * truth = Gauss-Hermite QUADRATURE of grad E[L] (NOT point-FD grad L(theta); they
    differ when L is nonlinear/curved over +/-sigma).
  * ZoBG is BASELINED: (L(theta+w) - L(theta)) w/sigma^2 (Suh/AHAC do this; the raw
    score estimator is needlessly high-variance).
  * statistics are POPULATION-level at large N (the first draft's "FoBG low variance"
    was the best of 50 seeds -- a cherry-pick). We assert unbiasedness within a
    standard-error margin and report the variance comparison.
  * the bias demonstration uses a genuinely DISCONTINUOUS event-indicator loss, where
    Suh's jump-bias actually applies (FoBG = 0 a.s., biased; ZoBG unbiased). A
    *continuous* loss near a contact boundary does NOT exhibit it.

Honest headline (the motivation for relaxed-IFT, §5.10): in our current HARD discrete
contact the first-order gradient is noisy / artifact-dominated, and baselined ZoBG is
competitive or better -- exactly the gap a Dojo-style relaxed-IFT (tunable rho) should
close. That comparison is the (currently skipped) relaxed-IFT hook.

Run (this file is named *_test.py so default `pytest` collects it):
  cd ../mujoco_warp && uv run --active --with pytest \
    python -m pytest -s mujoco_warp/_src/adjoint_analytic_test.py
"""

import math

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized
from numpy.polynomial.hermite_e import hermegauss

import mujoco_warp as mjw

# Direct import: adjoint.py is part of this package on the `adjoint` branch. If it is
# missing / fails to import, we want a LOUD error here, not a silent green skip (the
# first draft's broad-except _grad_available() hid real failures).
from mujoco_warp._src import adjoint


def _adjoint_has_rho():
  """True iff adjoint.py exposes a relaxed-IFT rho knob (MJPLAN §5.10). False today."""
  return hasattr(adjoint, "set_cone_relaxation") or hasattr(adjoint, "CONE_RHO")


def _norm_cdf(z):
  return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_pdf(z):
  return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


# ============================================================================
# Closed-form oracles (pure numpy -- simulator-independent)
# ============================================================================


def floor_bounce_pn(p0, v0, T, e_hat, r, u0=0.0, dt=0.0):
  """IDEAL elastic floor bounce, gravity off (Zhong 2207.05060 App. A Eq. 11), scaled to
  restitution e_hat: p_N = r(1+e_hat) - e_hat*p0 - e_hat*v0*T - e_hat*u0*(T*dt - dt^2/2).
  This is the *hard-impact* idealization; the discrete soft-contact sim deviates from it."""
  return r * (1.0 + e_hat) - e_hat * p0 - e_hat * v0 * T - e_hat * u0 * (T * dt - 0.5 * dt * dt)


def floor_bounce_grads(T, e_hat, dt=0.0):
  """Ideal-elastic gradients (Eq. 12, scaled by e_hat): (dpN/dp0, dpN/dv0, dpN/du0)."""
  return (-e_hat, -e_hat * T, -e_hat * (T * dt - 0.5 * dt * dt))


def two_ball_loss_diagonal(x0, v0, u0, u_c=3.0 * math.sqrt(2.0), dt=1.0 / 480.0, r=0.2, T=1.0, epsilon=0.1):
  """Zhong 2207.05060 App. C Listing 2 (verbatim transcription), reduced to 1D along the
  [1,1] diagonal. NOTE (review): (a) the diagonal trajectory has a FIXED normal orientation,
  so this does NOT test a moving-normal derivative -- use two_ball_offaxis_loss for that;
  (b) Listing 2's running cost is linear (epsilon*u0*dt) whereas the paper text states a
  quadratic cost -- the printed -0.000889 matches the listing, so we reproduce the listing."""
  x1_0, x2_0 = x0
  v1_0, v2_0 = v0
  v1_dt = v1_0 + u0 * dt
  v2_dt = v2_0
  x1_dt = x1_0 + v1_0 * dt + u0 * dt * dt / 2.0
  x2_dt = x2_0 + v2_0 * dt
  dist_dt = x2_dt - x1_dt - 2.0 * r
  a = u_c / 2.0
  b = v1_dt - v2_dt
  c = -dist_dt
  s = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a) + dt
  v1_s = v1_dt + u_c * (s - dt)
  x2_s = x2_dt + v2_dt * (s - dt)
  x2_T = x2_s + v1_s * (T - s)
  return x2_T * x2_T + epsilon * u0 * dt


def _fd_grad(f, x, eps=1e-6):
  g = np.zeros_like(np.asarray(x, dtype=float))
  for i in range(len(g)):
    xp = np.array(x, dtype=float)
    xm = np.array(x, dtype=float)
    xp[i] += eps
    xm[i] -= eps
    g[i] = (f(xp) - f(xm)) / (2.0 * eps)
  return g


def two_ball_grads_diagonal(eps=1e-6):
  """FD of the Listing-2 closed form at the paper's iteration-0 point -> per-axis grads
  (after the /sqrt(2) diagonal reduction). FD-of-closed-form, not an analytic derivative;
  used purely as a transcription regression against the paper's printed numbers."""
  x0 = np.array([-2.0 * math.sqrt(2.0), -1.0 * math.sqrt(2.0)])
  v0 = np.array([0.0, 0.0])
  u0 = 3.0 * math.sqrt(2.0)
  dl_dx = _fd_grad(lambda z: two_ball_loss_diagonal(z, v0, u0), x0, eps)
  dl_dv = _fd_grad(lambda z: two_ball_loss_diagonal(x0, z, u0), v0, eps)
  dl_du = (two_ball_loss_diagonal(x0, v0, u0 + eps) - two_ball_loss_diagonal(x0, v0, u0 - eps)) / (2.0 * eps)
  s = math.sqrt(2.0)
  return dl_dx / s, dl_dv / s, dl_du / s


def two_ball_offaxis_loss(x1_0, x2_0, v1, v2, r=0.2, T=1.0):
  """Equal-mass, frictionless, perfectly-elastic 2-ball collision in 2D with a GENUINELY
  MOVING normal (n_hat depends on positions). Single collision; terminal loss ||x2(T)||^2.
  This is the moving-normal reference (the diagonal Listing-2 case has a fixed normal).
  The MuJoCo adjoint comparison is gated on multi-body support (one free joint today)."""
  x1_0, x2_0, v1, v2 = (np.asarray(a, dtype=float) for a in (x1_0, x2_0, v1, v2))
  dp = x1_0 - x2_0
  dv = v1 - v2
  a = float(dv @ dv)
  b = float(2.0 * dp @ dv)
  c = float(dp @ dp - (2.0 * r) ** 2)
  disc = b * b - 4.0 * a * c
  if a < 1e-12 or disc <= 0.0:
    x2_T = x2_0 + v2 * T  # no collision
    return float(x2_T @ x2_T)
  t_c = (-b - math.sqrt(disc)) / (2.0 * a)  # first contact
  if t_c < 0.0 or t_c > T:
    x2_T = x2_0 + v2 * T
    return float(x2_T @ x2_T)
  x1_c = x1_0 + v1 * t_c
  x2_c = x2_0 + v2 * t_c
  n = (x2_c - x1_c) / (2.0 * r)  # unit line-of-centers (||x2_c-x1_c|| == 2r at contact)
  dvn = float((v1 - v2) @ n)
  v2p = v2 + dvn * n  # equal-mass elastic exchange along n
  x2_T = x2_c + v2p * (T - t_c)
  return float(x2_T @ x2_T)


# ============================================================================
# Fixtures
# ============================================================================

# Floor bounce: single free sphere, GRAVITY OFF, frictionless (condim=1), near-elastic via
# negative solref (direct -k -b). {solref} lets the stiffness sweep reuse the XML.
_FLOOR_TMPL = """
<mujoco>
  <option timestep="0.004" cone="elliptic" integrator="implicitfast"
          tolerance="1e-10" iterations="200" ls_iterations="100" gravity="0 0 0">
    <flag contact="enable"/>
  </option>
  <default>
    <geom condim="1" solref="{solref}" solimp="0 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05"/>
    <body name="ball" pos="0 0 1.0">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_FLOOR_R = 0.1
_FLOOR_Z0 = 1.0
_FLOOR_VZ0 = -6.0
_FLOOR_T = 60
_FLOOR_DT = 0.004


def _floor_xml(k=8000.0, b=12.0):
  return _FLOOR_TMPL.format(solref=f"-{k} -{b}")


# Two balls that actually COLLIDE within a short rollout (small initial gap, approaching).
# Used to (a) prove the forward reaches contact and (b) document the multi-body backward gap.
_TWO_BALL = """
<mujoco>
  <option timestep="0.002" cone="elliptic" integrator="implicitfast"
          tolerance="1e-10" iterations="200" gravity="0 0 0">
    <flag contact="enable"/>
  </option>
  <default>
    <geom condim="1" solref="-50000 -10" solimp="0 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05" contype="0" conaffinity="0"/>
    <body name="b1" pos="-0.30 0 0.2">
      <freejoint/>
      <geom name="g1" type="sphere" size="0.2" mass="1"/>
    </body>
    <body name="b2" pos="0.30 0 0.2">
      <freejoint/>
      <geom name="g2" type="sphere" size="0.2" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Sliding sphere (condim=3): a tangential cone exists, so a cone-apex (slip->0) and a
# relaxation rho are meaningful. Used by the (skipped-until-rho) relaxed-IFT hook.
_SLIDING = """
<mujoco>
  <option timestep="0.004" cone="elliptic" integrator="implicitfast"
          tolerance="1e-10" iterations="200" ls_iterations="100" gravity="0 0 -9.81">
    <flag contact="enable"/>
  </option>
  <default>
    <geom condim="3" friction="0.6 0.005 0.0001" solref="-20000 -40" solimp="0 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05"/>
    <body name="ball" pos="0 0 0.1">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


# ============================================================================
# Warp + native rollout helpers
# ============================================================================


@wp.kernel
def _final_z_kernel(qpos: wp.array2d[float], loss: wp.array[float]):
  loss[0] = qpos[0, 2]


@wp.kernel
def _indicator_z_kernel(qpos: wp.array2d[float], thresh: float, loss: wp.array[float]):
  # Discontinuous event indicator: 1[final_z > thresh]. Its pathwise (Warp) adjoint is 0
  # a.s. -> FoBG = 0 (biased); ZoBG (score) captures the jump. This is the Suh Ex 3.3
  # mechanism instantiated on our contact sim.
  if qpos[0, 2] > thresh:
    loss[0] = 1.0
  else:
    loss[0] = 0.0


def _floor_forward_z(mjm, mjd, qpos0, qvel0, T):
  """Warp forward-only rollout (in-place); returns final z (one sync)."""
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  d.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
  d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
  for _ in range(T):
    mjw.step(m, d)
  return float(d.qpos.numpy()[0, 2])


def _floor_forward_pattern(mjm, mjd, qpos0, qvel0, T):
  """Warp forward tracking per-step nacon -> (final_z, final_vz, contact_step_tuple)."""
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  d.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
  d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
  steps = []
  for s in range(T):
    mjw.step(m, d)
    if int(d.nacon.numpy()[0]) > 0:
      steps.append(s)
  return float(d.qpos.numpy()[0, 2]), float(d.qvel.numpy()[0, 2]), tuple(steps)


def _floor_forward_native(mjm, qpos0, qvel0, T):
  """Native MuJoCo (mujoco.mj_step) rollout -> final z. Cross-checks that Warp's forward
  (and thus its FD) is faithful to MuJoCo C, so FD is a trustworthy correctness oracle."""
  d = mujoco.MjData(mjm)
  d.qpos[:] = qpos0
  d.qvel[:] = qvel0
  for _ in range(T):
    mujoco.mj_step(mjm, d)
  return float(d.qpos[2])


def _floor_taped(mjm, mjd, qpos0, qvel0, T, mode="z", thresh=0.0):
  """Analytic d(loss)/d(qpos0), d(loss)/d(qvel0) via our adjoint (wp.Tape over the
  out-of-place rollout). mode='z' -> final height; mode='indicator' -> 1[z>thresh].
  Returns (loss, g_qpos[7], g_qvel[6])."""
  m = mjw.put_model(mjm)
  datas = [mjw.put_data(mjm, mjd) for _ in range(T + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  datas[0].qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32, requires_grad=True)
  datas[0].qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32, requires_grad=True)

  loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(T):
      mjw.step(m, datas[t], datas[t + 1])
    if mode == "indicator":
      wp.launch(_indicator_z_kernel, dim=1, inputs=[datas[T].qpos, float(thresh)], outputs=[loss])
    else:
      wp.launch(_final_z_kernel, dim=1, inputs=[datas[T].qpos], outputs=[loss])
  tape.backward(loss=loss)
  return (
    float(loss.numpy()[0]),
    datas[0].qpos.grad.numpy()[0].astype(np.float64).copy(),
    datas[0].qvel.grad.numpy()[0].astype(np.float64).copy(),
  )


def _floor_calibrate(mjm, mjd, qpos0, qvel0, T):
  """One tracked forward -> (e_hat, n_bounces, contact_steps). Gravity off + single bounce
  => e_hat = |v_z_final|/|v_z_0| (v_z is piecewise-constant)."""
  fz, fvz, steps = _floor_forward_pattern(mjm, mjd, qpos0, qvel0, T)
  # count upward sign flips as bounces
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  d.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
  d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
  n_bounce, vprev = 0, float(qvel0[2])
  for _ in range(T):
    mjw.step(m, d)
    v = float(d.qvel.numpy()[0, 2])
    if vprev < 0.0 and v > 0.0:
      n_bounce += 1
    vprev = v
  e_hat = abs(fvz) / max(abs(float(qvel0[2])), 1e-12)
  return e_hat, n_bounce, steps


# ============================================================================
# FoBG / ZoBG estimator helpers (a = theta + w, w ~ N(0, sigma^2); no NN, no SHAC)
# ============================================================================


def _quad_grad_E(loss_of_theta, theta, sigma, deg=41):
  """Deterministic Gauss-Hermite quadrature of grad_theta E_w[L(theta+w)] via the score
  form E[L(theta+w) w/sigma^2]. ``loss_of_theta(scalar)`` is forward-only. This is the
  unbiased, adjoint-independent TRUTH (NOT point-FD of L(theta))."""
  x, wts = hermegauss(deg)  # weight exp(-x^2/2); sum wts = sqrt(2 pi)
  norm = 1.0 / math.sqrt(2.0 * math.pi)
  g = 0.0
  for xi, wi in zip(x, wts):
    g += wi * loss_of_theta(theta + sigma * xi) * (xi / sigma)
  return norm * g


def _fobg_zobg_samples(fobg_of_theta, loss_of_theta, theta, sigma, N, seed):
  """Returns (fobg[N], zobg_baselined[N]). FoBG_i = adjoint d L/d theta at theta+w_i;
  ZoBG_i = (L(theta+w_i) - L(theta)) * w_i/sigma^2 (baselined, as Suh/AHAC do)."""
  rng = np.random.default_rng(seed)
  ws = rng.normal(0.0, sigma, size=N)
  L0 = loss_of_theta(theta)
  fobg = np.array([fobg_of_theta(theta + w) for w in ws])
  zobg = np.array([(loss_of_theta(theta + w) - L0) * w / (sigma * sigma) for w in ws])
  return fobg, zobg


# ============================================================================
# Group 1 -- CORRECTNESS
# ============================================================================


class AnalyticCorrectnessTest(parameterized.TestCase):
  def test_floor_bounce_adjoint_matches_fd(self):
    """Oracle A: the adjoint's d(z_N)/d(v_z0), d(z_N)/d(z0) equal central FD of the rollout,
    with FD made rigorous: (i) active-set-stable across an eps BRACKET (identical contact-step
    pattern across +/-eps; FD value plateaus), (ii) agrees with a NATIVE MuJoCo rollout FD.
    The ideal-elastic closed form (-e_hat, -e_hat*T) is only REPORTED: the soft/discrete
    contact deviates from it (the d/dz0~-1.8 vs ideal -e_hat~-0.8 is a fixed-step contact-phase
    artifact), which is unresolved discretization bias, not a different correctness oracle."""
    mjm = mujoco.MjModel.from_xml_string(_floor_xml())
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    qpos0 = mjd.qpos.copy()
    qpos0[2] = _FLOOR_Z0
    qvel0 = np.zeros(6)
    qvel0[2] = _FLOOR_VZ0

    e_hat, n_bounce, steps0 = _floor_calibrate(mjm, mjd, qpos0, qvel0, _FLOOR_T)
    self.assertEqual(n_bounce, 1, f"need one clean bounce (got {n_bounce})")
    self.assertGreater(e_hat, 0.3)
    T_sec = _FLOOR_T * _FLOOR_DT
    ideal_dz0, ideal_dvz, _ = floor_bounce_grads(T_sec, e_hat)

    _, g_qpos, g_qvel = _floor_taped(mjm, mjd, qpos0, qvel0, _FLOOR_T)
    a_dvz, a_dz0 = g_qvel[2], g_qpos[2]

    # Active-set-stable FD bracket (Warp) for each of v_z0, z0.
    def bracket_fd(is_vel):
      vals = []
      for eps in (5e-4, 1e-3, 2e-3):
        plus = (qpos0.copy(), qvel0.copy())
        minus = (qpos0.copy(), qvel0.copy())
        idx = (1 if is_vel else 0)
        (plus[idx])[2] += eps
        (minus[idx])[2] -= eps
        zp, _, sp = _floor_forward_pattern(mjm, mjd, plus[0], plus[1], _FLOOR_T)
        zm, _, sm = _floor_forward_pattern(mjm, mjd, minus[0], minus[1], _FLOOR_T)
        # require the contact-step pattern to be identical across +/-eps (no active-set
        # boundary straddled) so the FD is a derivative, not a secant across a kink.
        self.assertEqual(sp, sm, f"active-set straddle at eps={eps} (is_vel={is_vel}): {sp} vs {sm}")
        vals.append((zp - zm) / (2.0 * eps))
      vals = np.array(vals)
      self.assertLess(vals.max() - vals.min(), 5e-3, f"FD not on a plateau across the bracket: {vals}")
      return float(vals.mean())

    fd_dvz = bracket_fd(True)
    fd_dz0 = bracket_fd(False)

    # Native-MuJoCo FD cross-check (forward fidelity -> FD trustworthy).
    eps = 1e-3
    vp, vm = qvel0.copy(), qvel0.copy()
    vp[2] += eps
    vm[2] -= eps
    nat_dvz = (_floor_forward_native(mjm, qpos0, vp, _FLOOR_T) - _floor_forward_native(mjm, qpos0, vm, _FLOOR_T)) / (2 * eps)

    print(
      f"\n[Oracle A floor bounce] e_hat={e_hat:.4f} T={T_sec:.3f}s bounce_steps={steps0}"
      f"\n  d(z_N)/d(v_z0): adjoint={a_dvz:.5f}  warp_fd={fd_dvz:.5f}  native_fd={nat_dvz:.5f}  ideal={ideal_dvz:.4f}"
      f"\n  d(z_N)/d(z0)  : adjoint={a_dz0:.5f}  warp_fd={fd_dz0:.5f}  ideal={ideal_dz0:.4f}"
      f"\n  -> adjoint==FD (correctness); |d/dz0|>|e_hat| is a fixed-step contact-phase artifact"
      f" (continuous identity d z_N/d z0 = -e_hat), i.e. discretization bias, NOT soft-contact physics."
    )
    # Correctness: adjoint == FD (tight; observed mismatch ~1e-5).
    np.testing.assert_allclose(a_dvz, fd_dvz, rtol=5e-3, atol=1e-3, err_msg="d(z_N)/d(v_z0): adjoint vs warp FD")
    np.testing.assert_allclose(a_dz0, fd_dz0, rtol=5e-3, atol=1e-3, err_msg="d(z_N)/d(z0): adjoint vs warp FD")
    np.testing.assert_allclose(fd_dvz, nat_dvz, rtol=2e-2, atol=2e-3, err_msg="warp FD vs native MuJoCo FD")

  def test_two_ball_diagonal_listing2_regression(self):
    """Two-ball closed form (Zhong App. C Listing 2): numpy transcription reproduces the
    paper's printed iteration-0 gradients. This is a fixed-normal (collinear) case and a
    transcription regression -- see two_ball_offaxis for the moving-normal reference."""
    dl_dx, dl_dv, dl_du = two_ball_grads_diagonal()
    print(f"\n[two-ball Listing-2] dl/dx0={dl_dx}  dl/dv0={dl_dv}  dl/du0={dl_du:.6f}")
    np.testing.assert_allclose(dl_dx, [-0.39866853, -0.3212531], atol=1e-3)
    np.testing.assert_allclose(dl_dv, [-0.49779078, -0.22213092], atol=1e-3)
    np.testing.assert_allclose(dl_du, -0.0008888851, atol=1e-4)

  def test_two_ball_offaxis_oracle_moving_normal(self):
    """Moving-normal reference: an off-axis elastic 2-ball collision (n_hat rotates with the
    positions). Self-validates the closed form (a finite, sign-correct d(loss)/d(v1) through a
    rotating contact normal) -- the numpy oracle the MuJoCo two-ball adjoint will match once
    multi-body support lands. Guards the MJPLAN §5.9 moving-normal / geom-order class."""
    x1_0 = np.array([-1.0, 0.2])
    x2_0 = np.array([1.0, -0.2])
    v1 = np.array([3.0, 0.0])
    v2 = np.array([-1.0, 0.0])
    # a collision must actually occur in the window
    L0 = two_ball_offaxis_loss(x1_0, x2_0, v1, v2)
    dl_dv1 = _fd_grad(lambda z: two_ball_offaxis_loss(x1_0, x2_0, z, v2), v1, eps=1e-5)
    print(f"\n[two-ball off-axis] loss={L0:.5f}  dl/dv1={dl_dv1}")
    self.assertTrue(np.all(np.isfinite(dl_dv1)))
    self.assertGreater(np.linalg.norm(dl_dv1), 1e-3, "off-axis gradient is ~0 -> collision not exercised")

  def test_two_ball_adjoint_reaches_contact_then_unsupported(self):
    """A REAL colliding two-ball rollout (not a single non-contacting step): we (1) assert a
    contact actually occurs in the forward, then (2) attempt the taped backward, which fails
    because step_backward supports one free joint only. Documents the precise multi-body gap
    (MJPLAN_ARTICULATION S4) with the moving-normal oracle in hand. EXPECTED RED until support
    lands; then it should read d(loss)/d(initial) and assert ~ the off-axis oracle."""
    mjm = mujoco.MjModel.from_xml_string(_TWO_BALL)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    T = 60
    qvel0 = np.zeros(mjm.nv)
    qvel0[0] = 3.0  # b1 +x toward b2
    qvel0[6] = -3.0  # b2 -x toward b1

    # (1) forward must reach contact.
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)
    d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
    max_nacon = 0
    for _ in range(T):
      mjw.step(m, d)
      max_nacon = max(max_nacon, int(d.nacon.numpy()[0]))
    self.assertGreater(max_nacon, 0, "two balls never collided; fix the fixture before asserting a gradient")

    # (2) taped backward -> currently NotImplementedError (two free joints).
    datas = [mjw.put_data(mjm, mjd) for _ in range(T + 1)]
    for dd in datas:
      dd.qpos.requires_grad = True
      dd.qvel.requires_grad = True
    datas[0].qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    try:
      tape = wp.Tape()
      with tape:
        for t in range(T):
          mjw.step(m, datas[t], datas[t + 1])
        wp.launch(_final_z_kernel, dim=1, inputs=[datas[T].qpos], outputs=[loss])  # placeholder readout
      tape.backward(loss=loss)
    except NotImplementedError as e:
      self.fail(
        "two-ball (moving-normal) adjoint unsupported -- step_backward is one-free-joint only.\n"
        f"  contact DID occur in the forward (max_nacon={max_nacon}).\n"
        f"  blocker: {e}\n"
        "  moving-normal oracle ready (two_ball_offaxis_loss); activates at multi-body support (S4)."
      )


# ============================================================================
# Group 2 -- USEFULNESS (FoBG from our adjoint vs ZoBG vs quadrature truth)
# ============================================================================


class FobgUsefulnessTest(parameterized.TestCase):
  def test_heaviside_reference(self):
    """Pure-numpy reference validating the FoBG/ZoBG machinery against closed forms.
    Hard Heaviside H (0/1, Suh Ex 3.3): truth = phi(theta/sigma)/sigma; FoBG = 0 a.s.
    (biased); baselined ZoBG unbiased. Soft Heaviside H_bar_nu (-1/1, AHAC Eq 4): FoBG mean
    matches the closed form (2/nu)[Phi((nu/2-theta)/sigma)-Phi((-nu/2-theta)/sigma)] and its
    per-sample variance grows ~1/nu as nu->0."""
    rng = np.random.default_rng(0)
    theta, sigma, N = 0.3, 1.0, 200000
    w = rng.normal(0.0, sigma, size=N)
    a = theta + w

    truth_hard = _norm_pdf(theta / sigma) / sigma
    fobg_hard = np.zeros(N)  # H'(a) = 0 a.e.
    L0 = 1.0 if theta >= 0 else 0.0
    zobg_hard = ((a >= 0.0).astype(float) - L0) * w / (sigma * sigma)
    print(f"\n[hard Heaviside 0/1] truth={truth_hard:.4f}  FoBG mean={fobg_hard.mean():.4f}  ZoBG mean={zobg_hard.mean():.4f}")
    self.assertAlmostEqual(fobg_hard.mean(), 0.0, places=6)
    self.assertGreater(abs(fobg_hard.mean() - truth_hard), 0.2)  # FoBG biased
    self.assertAlmostEqual(zobg_hard.mean(), truth_hard, delta=0.02)  # baselined ZoBG unbiased

    # soft Heaviside (AHAC Eq 4, signed -1/1): H' = 2/nu on |x|<nu/2.
    def soft_closed_form(nu):
      return (2.0 / nu) * (_norm_cdf((nu / 2.0 - theta) / sigma) - _norm_cdf((-nu / 2.0 - theta) / sigma))

    variances = {}
    for nu in (1.0, 0.05):
      hp = np.where(np.abs(a) <= nu / 2.0, 2.0 / nu, 0.0)
      variances[nu] = hp.var()
      self.assertAlmostEqual(hp.mean(), soft_closed_form(nu), delta=0.03 + 0.5 / math.sqrt(N))
    print(f"[soft Heaviside -1/1] Var[FoBG] nu=1.0->{variances[1.0]:.3f}  nu=0.05->{variances[0.05]:.3f}")
    self.assertGreater(variances[0.05], variances[1.0])  # variance grows as the apex narrows

  def test_fobg_unbiased_vs_quadrature_truth_smooth(self):
    """Smooth deep bounce: our adjoint's FoBG is UNBIASED for grad E[L] (mean ~ Gauss-Hermite
    quadrature truth). HONEST finding (reported, and the motivation for relaxed-IFT): in this
    HARD discrete contact the per-sample FoBG is NOISY (contact-phase jumps), and BASELINED
    ZoBG is competitive / lower-variance -- i.e. the first-order gradient is not obviously more
    useful here. (The first draft's 'FoBG low variance' was a single cherry-picked seed.)"""
    mjm = mujoco.MjModel.from_xml_string(_floor_xml())
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    qpos0 = mjd.qpos.copy()
    qpos0[2] = _FLOOR_Z0
    base_v = _FLOOR_VZ0
    sigma, N = 0.3, 128  # large enough that the unbiasedness check is not seed-fragile

    def loss_of(v):
      qv = np.zeros(6)
      qv[2] = v
      return _floor_forward_z(mjm, mjd, qpos0, qv, _FLOOR_T)

    def fobg_of(v):
      qv = np.zeros(6)
      qv[2] = v
      _, _, gv = _floor_taped(mjm, mjd, qpos0, qv, _FLOOR_T)
      return gv[2]

    truth = _quad_grad_E(loss_of, base_v, sigma, deg=41)
    fobg, zobg = _fobg_zobg_samples(fobg_of, loss_of, base_v, sigma, N=N, seed=1)
    fobg_se = math.sqrt(fobg.var() / N)
    print(
      f"\n[smooth] quad-truth grad E[L]={truth:.4f}"
      f"\n  FoBG: mean={fobg.mean():.4f} (SE {fobg_se:.4f})  pop_var={fobg.var():.4e}"
      f"\n  ZoBG(baselined): mean={zobg.mean():.4f}  pop_var={zobg.var():.4e}"
    )
    # FoBG is unbiased for grad E[L] (within a few SE). This is the robust correctness claim.
    self.assertLess(abs(fobg.mean() - truth), 4.0 * fobg_se + 0.03)
    # Honest usefulness finding: baselined ZoBG is NOT worse here (often lower variance).
    # We assert the *unbiasedness* of both and only REPORT the variance ordering (it is the
    # relaxed-IFT motivation, not a property to lock in as passing).
    self.assertLess(abs(zobg.mean() - truth), 4.0 * math.sqrt(zobg.var() / N) + 0.05)

  def test_fobg_biased_on_discontinuous_event_loss(self):
    """Suh Ex 3.3 ON OUR SIM: with a DISCONTINUOUS event-indicator loss 1[z_N > thresh], the
    pathwise gradient (FoBG, our adjoint) is 0 a.s. -> BIASED, while baselined ZoBG recovers
    the quadrature truth grad_theta P(z_N>thresh). (A *continuous* loss near a contact boundary
    does NOT show this -- the first draft mislabeled finite-sample noise as jump-bias.)"""
    mjm = mujoco.MjModel.from_xml_string(_floor_xml())
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    qpos0 = mjd.qpos.copy()
    qpos0[2] = _FLOOR_Z0
    base_v = _FLOOR_VZ0
    sigma = 0.6

    # choose thresh near the nominal final height so P(z_N>thresh) is sensitive to v.
    z_nom = _floor_forward_z(mjm, mjd, qpos0, np.array([0, 0, base_v, 0, 0, 0.0]), _FLOOR_T)
    thresh = z_nom

    def loss_of(v):
      qv = np.zeros(6)
      qv[2] = v
      return 1.0 if _floor_forward_z(mjm, mjd, qpos0, qv, _FLOOR_T) > thresh else 0.0

    def fobg_of(v):
      qv = np.zeros(6)
      qv[2] = v
      _, _, gv = _floor_taped(mjm, mjd, qpos0, qv, _FLOOR_T, mode="indicator", thresh=thresh)
      return gv[2]

    truth = _quad_grad_E(loss_of, base_v, sigma, deg=81)
    fobg, zobg = _fobg_zobg_samples(fobg_of, loss_of, base_v, sigma, N=96, seed=2)
    print(
      f"\n[discontinuous event] quad-truth grad P(z>thr)={truth:.4f}"
      f"  FoBG mean={fobg.mean():.4f} (err {abs(fobg.mean()-truth):.4f})"
      f"  ZoBG mean={zobg.mean():.4f} (err {abs(zobg.mean()-truth):.4f})"
    )
    self.assertGreater(abs(truth), 0.02, "pick a threshold where the event probability actually moves")
    np.testing.assert_allclose(fobg, 0.0, atol=1e-5)  # pathwise grad of an indicator is 0
    self.assertGreater(abs(fobg.mean() - truth), abs(zobg.mean() - truth))  # FoBG strictly more biased

  def test_fobg_and_zobg_vs_contact_stiffness(self):
    """Suh Ex 3.8 flavor on our sim: sweep contact stiffness; at each k validate adjoint==FD,
    then report per-sample FoBG and baselined-ZoBG population variance. Robust claim: FoBG
    variance grows with stiffness. (Restitution also changes with k -- reported, not held
    fixed -- so this is a trend, not a controlled 'same bounce'.)"""
    qpos0 = None
    fobg_vars, zobg_vars, ehats = [], [], []
    ks = [2000.0, 10000.0, 50000.0]
    for k in ks:
      mjm = mujoco.MjModel.from_xml_string(_floor_xml(k=k, b=12.0))
      mjd = mujoco.MjData(mjm)
      mujoco.mj_forward(mjm, mjd)
      if qpos0 is None:
        qpos0 = mjd.qpos.copy()
        qpos0[2] = _FLOOR_Z0
      qv0 = np.zeros(6)
      qv0[2] = _FLOOR_VZ0
      e_hat, n_bounce, _ = _floor_calibrate(mjm, mjd, qpos0, qv0, _FLOOR_T)
      ehats.append(e_hat)

      # adjoint==FD at this stiffness (correctness before usefulness).
      _, _, gv = _floor_taped(mjm, mjd, qpos0, qv0, _FLOOR_T)
      eps = 1e-3
      vp, vm = qv0.copy(), qv0.copy()
      vp[2] += eps
      vm[2] -= eps
      fd = (_floor_forward_z(mjm, mjd, qpos0, vp, _FLOOR_T) - _floor_forward_z(mjm, mjd, qpos0, vm, _FLOOR_T)) / (2 * eps)
      np.testing.assert_allclose(gv[2], fd, rtol=2e-2, atol=5e-3, err_msg=f"adjoint vs FD at k={k}")

      def loss_of(v, _mjm=mjm, _mjd=mjd, _q=qpos0):
        qv = np.zeros(6)
        qv[2] = v
        return _floor_forward_z(_mjm, _mjd, _q, qv, _FLOOR_T)

      def fobg_of(v, _mjm=mjm, _mjd=mjd, _q=qpos0):
        qv = np.zeros(6)
        qv[2] = v
        _, _, g = _floor_taped(_mjm, _mjd, _q, qv, _FLOOR_T)
        return g[2]

      fobg, zobg = _fobg_zobg_samples(fobg_of, loss_of, _FLOOR_VZ0, sigma=0.5, N=32, seed=3)
      fobg_vars.append(float(fobg.var()))
      zobg_vars.append(float(zobg.var()))
      print(f"[stiffness] k={k:>8.0f} e_hat={e_hat:.3f}  Var[FoBG]={fobg.var():.4e}  Var[ZoBG_baselined]={zobg.var():.4e}")
    self.assertGreater(fobg_vars[-1], fobg_vars[0], "FoBG variance should grow with contact stiffness")

  @absltest.skipUnless(_adjoint_has_rho(), "ready hook: relaxed-IFT rho not in adjoint.py yet (MJPLAN §5.10)")
  def test_relaxed_ift_reduces_fobg_variance(self):
    """Dojo relaxed-IFT payoff (MJPLAN §5.10/§12.2): on a condim=3 SLIDING contact (a real
    tangential cone, so a cone-apex exists), relaxing T -> sqrt(T^2 + rho^2) in the BACKWARD
    only should DROP the FoBG variance near the apex vs the hard cone, at fixed forward
    dynamics. Runs only when adjoint.py exposes rho; no inner skip, so it executes when ready."""
    mjm = mujoco.MjModel.from_xml_string(_SLIDING)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    qpos0 = mjd.qpos.copy()
    T = 40

    # sliding init: pressed down + small tangential velocity (near-apex stick/slip).
    qv0 = np.zeros(6)
    qv0[0] = 0.2  # small tangential slip
    qv0[2] = -1.0  # into the floor

    def fobg_var(rho):
      adjoint.set_cone_relaxation(rho)
      vals = []
      rng = np.random.default_rng(11)
      for w in rng.normal(0.0, 0.05, size=24):
        qv = qv0.copy()
        qv[0] += w
        _, _, g = _floor_taped(mjm, mjd, qpos0, qv, T)
        vals.append(g[0])
      return float(np.var(vals))

    v_hard = fobg_var(0.0)
    v_relaxed = fobg_var(1e-2)
    adjoint.set_cone_relaxation(0.0)
    print(f"\n[relaxed-IFT] Var[FoBG] hard(rho=0)={v_hard:.4e}  relaxed(rho=1e-2)={v_relaxed:.4e}")
    self.assertLess(v_relaxed, v_hard)


if __name__ == "__main__":
  absltest.main()
