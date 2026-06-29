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

Sibling to ``adjoint_test.py`` (whose oracles are all finite-difference). Reworked
twice after adversarial Codex reviews; the design reflects their findings.

TIERS:
  * FAST (always collected): correctness + closed-form oracles -- Oracle A
    (adjoint==FD), the two-ball Listing-2 regression + off-axis moving-normal
    oracle, the two-ball capability test, the Heaviside reference. Seconds.
  * SLOW (set ``MJW_ANALYTIC_SLOW=1``): the FoBG-vs-ZoBG usefulness
    characterization (taped Monte-Carlo / quadrature). Minutes.

CORRECTNESS oracle for contact: there is NO clean simulator-independent closed
form for a *soft, discrete* contact -- its gradient is dominated by a fixed-step
contact-phase (discretization) artifact (e.g. d(z_N)/d(z0) ~ -1.8 vs the continuous
identity -e_hat ~ -0.8). So the signal is **adjoint == FD**, made rigorous via (i) a
full per-step active-set signature ``(nacon, efc_state)`` that must be identical
across nominal / +eps / -eps (no kink straddled) with the FD on a plateau, and (ii)
a NATIVE MuJoCo (mujoco.mj_step) rollout FD on BOTH derivatives. The genuinely
simulator-independent closed forms are the two-ball collision (Zhong 2207.05060
App. C, incl. an off-axis moving-normal case) and the Heaviside reference.

USEFULNESS (Suh 2202.00817; AHAC 2405.17784): first-order (FoBG, from our adjoint)
vs zeroth-order (ZoBG) estimators of grad_theta E_w[L(theta+w)], w~N(0,sigma^2). No
NN, no SHAC -- a = theta + w. The smooth test is **deterministic Gauss-Hermite
quadrature** (no random seed): it computes the population E[FoBG], Var[FoBG], and
Var[baselined ZoBG] exactly and asserts FoBG unbiasedness for grad E[L] plus the
honest variance ordering. The bias test uses a genuinely DISCONTINUOUS event
indicator (FoBG=0 a.s., biased; ZoBG recovers a quadrature truth within a stated
error bound). ZoBG is baselined: (L(theta+w)-L(theta)) w/sigma^2.

Honest headline (motivates relaxed-IFT, §5.10): in HARD discrete contact the
first-order gradient is noisy/heavy-tailed and baselined ZoBG is competitive or
better -- the gap a Dojo-style relaxed-IFT (tunable rho) should close.

Run all: cd ../mujoco_warp && MJW_ANALYTIC_SLOW=1 uv run --active --with pytest \
  python -m pytest -s mujoco_warp/_src/adjoint_analytic_test.py
"""

import math
import os

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized
from numpy.polynomial.hermite_e import hermegauss

import mujoco_warp as mjw

# Direct import: adjoint.py is part of this package on the `adjoint` branch. A missing
# / broken import must fail LOUDLY here (not become a silent green skip).
from mujoco_warp._src import adjoint

_SLOW = bool(os.environ.get("MJW_ANALYTIC_SLOW"))
_SLOW_REASON = "slow characterization; set MJW_ANALYTIC_SLOW=1 to run"

# step_backward supports exactly one free joint today (MJPLAN_ARTICULATION S4 = multi-body).
_MULTIBODY = False


def _adjoint_has_rho():
  """True iff adjoint.py exposes a CALLABLE relaxed-IFT setter (MJPLAN §5.10). False today."""
  return callable(getattr(adjoint, "set_cone_relaxation", None))


def _norm_cdf(z):
  return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_pdf(z):
  return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


# ============================================================================
# Closed-form oracles (pure numpy -- simulator-independent)
# ============================================================================


def floor_bounce_pn(p0, v0, T, e_hat, r, u0=0.0, dt=0.0):
  """IDEAL elastic floor bounce, gravity off (Zhong 2207.05060 App. A Eq. 11), scaled to
  restitution e_hat. The discrete soft-contact sim deviates from this (see Oracle A)."""
  return r * (1.0 + e_hat) - e_hat * p0 - e_hat * v0 * T - e_hat * u0 * (T * dt - 0.5 * dt * dt)


def floor_bounce_grads(T, e_hat, dt=0.0):
  """Ideal-elastic gradients (Eq. 12, scaled by e_hat): (dpN/dp0, dpN/dv0, dpN/du0)."""
  return (-e_hat, -e_hat * T, -e_hat * (T * dt - 0.5 * dt * dt))


def _fd_grad(f, x, eps=1e-6):
  g = np.zeros_like(np.asarray(x, dtype=float))
  for i in range(len(g)):
    xp = np.array(x, dtype=float)
    xm = np.array(x, dtype=float)
    xp[i] += eps
    xm[i] -= eps
    g[i] = (f(xp) - f(xm)) / (2.0 * eps)
  return g


def two_ball_loss_diagonal(x0, v0, u0, u_c=3.0 * math.sqrt(2.0), dt=1.0 / 480.0, r=0.2, T=1.0, epsilon=0.1):
  """Zhong 2207.05060 App. C Listing 2 (verbatim), reduced to 1D along the [1,1] diagonal.
  Caveats: (a) the diagonal trajectory has a FIXED normal -> does NOT test a moving normal
  (use two_ball_offaxis_*); (b) Listing 2's running cost is linear (epsilon*u0*dt) while the
  paper text says quadratic -- the printed -0.000889 matches the listing, so we reproduce it."""
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


def two_ball_grads_diagonal(eps=1e-6):
  """FD of the Listing-2 closed form at the paper's iteration-0 point (after /sqrt(2)).
  A transcription regression against the paper's printed numbers (not an analytic deriv)."""
  x0 = np.array([-2.0 * math.sqrt(2.0), -1.0 * math.sqrt(2.0)])
  v0 = np.array([0.0, 0.0])
  u0 = 3.0 * math.sqrt(2.0)
  dl_dx = _fd_grad(lambda z: two_ball_loss_diagonal(z, v0, u0), x0, eps)
  dl_dv = _fd_grad(lambda z: two_ball_loss_diagonal(x0, z, u0), v0, eps)
  dl_du = (two_ball_loss_diagonal(x0, v0, u0 + eps) - two_ball_loss_diagonal(x0, v0, u0 - eps)) / (2.0 * eps)
  s = math.sqrt(2.0)
  return dl_dx / s, dl_dv / s, dl_du / s


def two_ball_offaxis_collision(x1_0, x2_0, v1, v2, r=0.2):
  """Equal-mass, frictionless, perfectly-elastic 2-ball collision in 2D (MOVING normal).
  Returns (collided, t_c, n_hat, v1_post, v2_post). collided=False on a grazing/miss."""
  x1_0, x2_0, v1, v2 = (np.asarray(a, dtype=float) for a in (x1_0, x2_0, v1, v2))
  dp = x1_0 - x2_0
  dv = v1 - v2
  a = float(dv @ dv)
  b = float(2.0 * dp @ dv)
  c = float(dp @ dp - (2.0 * r) ** 2)
  disc = b * b - 4.0 * a * c
  if a < 1e-12 or disc <= 1e-9:
    return False, 0.0, np.zeros(2), v1, v2
  t_c = (-b - math.sqrt(disc)) / (2.0 * a)  # first contact
  x1_c = x1_0 + v1 * t_c
  x2_c = x2_0 + v2 * t_c
  n = (x2_c - x1_c) / (2.0 * r)  # unit line-of-centers (||x2_c-x1_c|| == 2r)
  dvn = float((v1 - v2) @ n)
  v1p = v1 - dvn * n  # equal-mass elastic exchange along n
  v2p = v2 + dvn * n
  return True, t_c, n, v1p, v2p


def two_ball_offaxis_loss(x1_0, x2_0, v1, v2, r=0.2, T=1.0):
  """Terminal loss ||x2(T)||^2 through a single off-axis elastic collision (moving normal)."""
  x2_0 = np.asarray(x2_0, dtype=float)
  v2 = np.asarray(v2, dtype=float)
  collided, t_c, _, _, v2p = two_ball_offaxis_collision(x1_0, x2_0, v1, v2, r)
  if not collided or t_c < 0.0 or t_c > T:
    x2_T = x2_0 + v2 * T
    return float(x2_T @ x2_T)
  x2_c = x2_0 + v2 * t_c
  x2_T = x2_c + v2p * (T - t_c)
  return float(x2_T @ x2_T)


# ============================================================================
# Fixtures
# ============================================================================

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
# Rollout helpers (Warp + native)
# ============================================================================


@wp.kernel
def _final_z_kernel(qpos: wp.array2d[float], loss: wp.array[float]):
  loss[0] = qpos[0, 2]


@wp.kernel
def _indicator_z_kernel(qpos: wp.array2d[float], thresh: float, loss: wp.array[float]):
  if qpos[0, 2] > thresh:
    loss[0] = 1.0
  else:
    loss[0] = 0.0


@wp.kernel
def _x2_sq_kernel(qpos: wp.array2d[float], i2: int, loss: wp.array[float]):
  loss[0] = qpos[0, i2] * qpos[0, i2] + qpos[0, i2 + 1] * qpos[0, i2 + 1]


def _floor_final_z(mjm, mjd, qpos0, qvel0, T):
  """Warp forward-only rollout (in-place); final z (one sync). Fast path for ZoBG/quadrature."""
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  d.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
  d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
  for _ in range(T):
    mjw.step(m, d)
  return float(d.qpos.numpy()[0, 2])


def _floor_calibrate(mjm, mjd, qpos0, qvel0, T):
  """Returns (e_hat, n_bounce, signature, contact_steps). Gravity off + single bounce =>
  v_z is piecewise-constant, so e_hat = |v_z_final|/|v_z_0|."""
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  d.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
  d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
  sig = []
  n_bounce, vprev = 0, float(qvel0[2])
  for _ in range(T):
    mjw.step(m, d)
    nac = int(d.nacon.numpy()[0])
    v = float(d.qvel.numpy()[0, 2])
    if vprev < 0.0 and v > 0.0:
      n_bounce += 1
    vprev = v
    if nac > 0:
      addr = d.contact.efc_address.numpy()
      st = d.efc.state.numpy()[0]
      rows = tuple(int(st[addr[c, 0]]) for c in range(nac) if addr[c, 0] >= 0)
    else:
      rows = ()
    sig.append((nac, rows))
  fvz = float(d.qvel.numpy()[0, 2])
  e_hat = abs(fvz) / max(abs(float(qvel0[2])), 1e-12)
  csteps = [i for i, (nac, _) in enumerate(sig) if nac > 0]
  return e_hat, n_bounce, tuple(sig), csteps, fvz


def _floor_signature(mjm, mjd, qpos0, qvel0, T):
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)
  d.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
  d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
  sig = []
  for _ in range(T):
    mjw.step(m, d)
    nac = int(d.nacon.numpy()[0])
    if nac > 0:
      addr = d.contact.efc_address.numpy()
      st = d.efc.state.numpy()[0]
      rows = tuple(int(st[addr[c, 0]]) for c in range(nac) if addr[c, 0] >= 0)
    else:
      rows = ()
    sig.append((nac, rows))
  return float(d.qpos.numpy()[0, 2]), tuple(sig)


def _floor_final_z_native(mjm, qpos0, qvel0, T):
  d = mujoco.MjData(mjm)
  d.qpos[:] = qpos0
  d.qvel[:] = qvel0
  for _ in range(T):
    mujoco.mj_step(mjm, d)
  return float(d.qpos[2])


def _floor_taped(mjm, mjd, qpos0, qvel0, T, mode="z", thresh=0.0):
  """Analytic d(loss)/d(qpos0), d(loss)/d(qvel0) via our adjoint (wp.Tape over the
  out-of-place rollout). Returns (loss, g_qpos[7], g_qvel[6])."""
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


# ============================================================================
# Estimators / quadrature (a = theta + w, w ~ N(0, sigma^2); no NN, no SHAC)
# ============================================================================


def _gh_nodes(deg, sigma):
  """Gauss-Hermite (probabilists') nodes/weights for E_{N(0,sigma^2)}[g] = sum wn_i g(sigma x_i)."""
  x, w = hermegauss(deg)  # weight exp(-x^2/2); sum(w) = sqrt(2 pi)
  return x, w / math.sqrt(2.0 * math.pi)


def _quad_grad_E(loss_of_theta, theta, sigma, deg=121):
  """Deterministic grad_theta E_w[L(theta+w)] via the score form E[L(theta+w) w/sigma^2].
  Adjoint-independent TRUTH. Convergence-checked: also computes deg-20 and raises if they
  disagree beyond tol (so an unconverged degree cannot masquerade as truth)."""

  def _q(d):
    x, wn = _gh_nodes(d, sigma)
    return float(sum(wn_i * loss_of_theta(theta + sigma * xi) * (xi / sigma) for xi, wn_i in zip(x, wn)))

  g = _q(deg)
  g_lo = _q(deg - 20)
  if abs(g - g_lo) > 0.02 * (abs(g) + 1e-6) + 1e-3:
    raise AssertionError(f"quadrature not converged: deg {deg-20}->{g_lo:.5f} vs deg {deg}->{g:.5f}")
  return g


# ============================================================================
# Group 1 -- FAST CORRECTNESS
# ============================================================================


class AnalyticCorrectnessTest(parameterized.TestCase):
  def test_floor_bounce_pn_closed_form_consistency(self):
    """floor_bounce_pn is internally consistent with floor_bounce_grads (FD of the value ==
    the stated gradients). Validates the ideal-elastic reference math itself."""
    T, e, r, dt = 0.24, 0.8, _FLOOR_R, _FLOOR_DT
    eps = 1e-6
    dp0 = (floor_bounce_pn(1 + eps, -3, T, e, r, 2.0, dt) - floor_bounce_pn(1 - eps, -3, T, e, r, 2.0, dt)) / (2 * eps)
    dv0 = (floor_bounce_pn(1, -3 + eps, T, e, r, 2.0, dt) - floor_bounce_pn(1, -3 - eps, T, e, r, 2.0, dt)) / (2 * eps)
    du0 = (floor_bounce_pn(1, -3, T, e, r, 2.0 + eps, dt) - floor_bounce_pn(1, -3, T, e, r, 2.0 - eps, dt)) / (2 * eps)
    gp0, gv0, gu0 = floor_bounce_grads(T, e, dt)
    np.testing.assert_allclose([dp0, dv0, du0], [gp0, gv0, gu0], rtol=1e-5, atol=1e-7)

  def test_floor_bounce_adjoint_matches_fd(self):
    """Oracle A: adjoint's d(z_N)/d(v_z0), d(z_N)/d(z0) == central FD of the rollout, with FD
    made rigorous by (i) a per-step active-set signature (nacon, efc_state) identical across
    nominal/+eps/-eps + an FD plateau, and (ii) a native MuJoCo mj_step FD on BOTH derivatives.
    The ideal-elastic gradient is report-only: the soft/discrete contact deviates from it (a
    fixed-step contact-phase discretization artifact; the continuous identity is d z_N/d z0 =
    -e_hat), which is unresolved discretization bias, NOT a different correctness oracle."""
    mjm = mujoco.MjModel.from_xml_string(_floor_xml())
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    qpos0 = mjd.qpos.copy()
    qpos0[2] = _FLOOR_Z0
    qvel0 = np.zeros(6)
    qvel0[2] = _FLOOR_VZ0

    e_hat, n_bounce, sig0, csteps, fvz = _floor_calibrate(mjm, mjd, qpos0, qvel0, _FLOOR_T)
    # calibration prerequisites: exactly one bounce, a contiguous contact episode, separated
    # (upward terminal velocity) at the end so |v_final|/|v0| is a meaningful restitution.
    self.assertEqual(n_bounce, 1, f"need one clean bounce (got {n_bounce})")
    self.assertTrue(csteps, "no contact occurred")
    self.assertEqual(csteps, list(range(csteps[0], csteps[-1] + 1)), f"contact not contiguous: {csteps}")
    self.assertGreater(fvz, 0.0, "ball not separated (terminal v_z not upward)")
    self.assertGreater(e_hat, 0.3)
    T_sec = _FLOOR_T * _FLOOR_DT
    ideal_dz0, ideal_dvz, _ = floor_bounce_grads(T_sec, e_hat)

    _, g_qpos, g_qvel = _floor_taped(mjm, mjd, qpos0, qvel0, _FLOOR_T)
    a_dvz, a_dz0 = g_qvel[2], g_qpos[2]

    def bracket_fd(is_vel):
      vals = []
      for eps in (5e-4, 1e-3, 2e-3):
        idx = 1 if is_vel else 0
        qp_p, qv_p = qpos0.copy(), qvel0.copy()
        qp_m, qv_m = qpos0.copy(), qvel0.copy()
        (qv_p if is_vel else qp_p)[2] += eps
        (qv_m if is_vel else qp_m)[2] -= eps
        zp, sp = _floor_signature(mjm, mjd, qp_p, qv_p, _FLOOR_T)
        zm, sm = _floor_signature(mjm, mjd, qp_m, qv_m, _FLOOR_T)
        # the adjoint freezes the active set: a faithful FD must NOT cross it. Require the
        # full per-step (nacon, efc_state) signature to equal the nominal on BOTH sides.
        self.assertEqual(sp, sig0, f"+eps={eps} (is_vel={is_vel}) crossed the active set")
        self.assertEqual(sm, sig0, f"-eps={eps} (is_vel={is_vel}) crossed the active set")
        vals.append((zp - zm) / (2.0 * eps))
      vals = np.array(vals)
      self.assertLess(np.ptp(vals), 5e-3 * abs(vals.mean()) + 1e-4, f"FD not on a plateau: {vals}")
      return float(vals.mean())

    fd_dvz = bracket_fd(True)
    fd_dz0 = bracket_fd(False)

    def native_fd(is_vel, eps=1e-3):
      qp_p, qv_p = qpos0.copy(), qvel0.copy()
      qp_m, qv_m = qpos0.copy(), qvel0.copy()
      (qv_p if is_vel else qp_p)[2] += eps
      (qv_m if is_vel else qp_m)[2] -= eps
      return (_floor_final_z_native(mjm, qp_p, qv_p, _FLOOR_T) - _floor_final_z_native(mjm, qp_m, qv_m, _FLOOR_T)) / (
        2 * eps
      )

    nat_dvz = native_fd(True)
    nat_dz0 = native_fd(False)

    print(
      f"\n[Oracle A] e_hat={e_hat:.4f} T={T_sec:.3f}s contact_steps={csteps}"
      f"\n  d(z_N)/d(v_z0): adjoint={a_dvz:.5f}  warp_fd={fd_dvz:.5f}  native_fd={nat_dvz:.5f}  ideal={ideal_dvz:.4f}"
      f"\n  d(z_N)/d(z0)  : adjoint={a_dz0:.5f}  warp_fd={fd_dz0:.5f}  native_fd={nat_dz0:.5f}  ideal={ideal_dz0:.4f}"
      f"\n  -> adjoint==FD == native is correctness; gap to ideal (-e_hat) is a fixed-step contact-phase"
      f" discretization artifact (continuous identity d z_N/d z0 = -e_hat), NOT soft-contact physics."
    )
    np.testing.assert_allclose(a_dvz, fd_dvz, rtol=8e-3, atol=8e-4, err_msg="d(z_N)/d(v_z0): adjoint vs warp FD")
    np.testing.assert_allclose(a_dz0, fd_dz0, rtol=8e-3, atol=8e-4, err_msg="d(z_N)/d(z0): adjoint vs warp FD")
    np.testing.assert_allclose(fd_dvz, nat_dvz, rtol=5e-3, atol=5e-4, err_msg="warp vs native FD (v_z0)")
    np.testing.assert_allclose(fd_dz0, nat_dz0, rtol=5e-3, atol=5e-4, err_msg="warp vs native FD (z0)")

  def test_two_ball_diagonal_listing2_regression(self):
    """Two-ball Listing-2 transcription regression (fixed-normal): reproduces the paper's
    printed iteration-0 gradients (Zhong App. C). See off-axis test for the moving normal."""
    dl_dx, dl_dv, dl_du = two_ball_grads_diagonal()
    print(f"\n[two-ball Listing-2] dl/dx0={dl_dx}  dl/dv0={dl_dv}  dl/du0={dl_du:.6f}")
    np.testing.assert_allclose(dl_dx, [-0.39866853, -0.3212531], atol=1e-3)
    np.testing.assert_allclose(dl_dv, [-0.49779078, -0.22213092], atol=1e-3)
    np.testing.assert_allclose(dl_du, -0.0008888851, atol=1e-4)

  def test_two_ball_offaxis_oracle_moving_normal(self):
    """Moving-normal reference: an off-axis elastic 2-ball collision (n_hat rotates). Asserts a
    genuine (non-grazing) collision, conservation, and a stable (plateaued) gradient. This is
    the numpy oracle the MuJoCo two-ball adjoint will match once multi-body support lands."""
    x1_0 = np.array([-1.0, 0.1])
    x2_0 = np.array([1.0, -0.1])  # y-offset => off-axis, non-grazing (disc>0)
    v1 = np.array([3.0, 0.0])
    v2 = np.array([-1.0, 0.0])
    r, T = 0.2, 1.0

    collided, t_c, n, v1p, v2p = two_ball_offaxis_collision(x1_0, x2_0, v1, v2, r)
    self.assertTrue(collided, "fixture must produce a real (non-grazing) collision")
    self.assertTrue(0.0 < t_c < T, f"collision time {t_c} not interior")
    np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-9)
    # equal-mass elastic: momentum + KE conserved
    np.testing.assert_allclose(v1p + v2p, v1 + v2, atol=1e-9)
    np.testing.assert_allclose(v1p @ v1p + v2p @ v2p, v1 @ v1 + v2 @ v2, atol=1e-9)

    # gradient on a plateau (NOT a singular secant straddling collision/no-collision)
    dl = np.array([_fd_grad(lambda z: two_ball_offaxis_loss(x1_0, x2_0, z, v2, r, T), v1, e) for e in (1e-3, 1e-4, 1e-5)])
    self.assertLess(np.ptp(dl, axis=0).max(), 1e-2, f"FD not on a plateau: {dl}")
    g = dl.mean(0)
    print(f"\n[two-ball off-axis] t_c={t_c:.4f} n={n}  dl/dv1={g}")
    self.assertGreater(np.linalg.norm(g), 1e-2)  # collision actually exercised

  def test_two_ball_multibody_contact_gated(self):
    """Documents the multi-body CONTACT gap WITHOUT exercising the (in-flight) multi-body backward.
    adjoint.step_backward gates the CONTACT residual to a single free body
    (single_free_body = njnt==1 and nq==7 and nv==6); for two free joints that gate is OFF, so the
    contact gradient is omitted. We verify only host-side facts: (a) the scenario genuinely collides
    in the FORWARD (no tape, so adjoint.py is untouched), and (b) the gate is off for it. We do NOT
    call the multi-body backward here -- the articulation path in adjoint.py is in flight and can
    crash. The real moving-normal comparison vs the off-axis oracle is test_two_ball_adjoint_gradient
    (gated on _MULTIBODY until the per-dof multi-body contact scatter lands, MJPLAN_ARTICULATION
    S2/S3)."""
    mjm = mujoco.MjModel.from_xml_string(_TWO_BALL)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    T = 60
    qvel0 = np.zeros(mjm.nv)
    qvel0[0] = 3.0  # b1 +x toward b2
    qvel0[6] = -3.0  # b2 -x toward b1

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)
    d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
    max_nacon = 0
    for _ in range(T):
      mjw.step(m, d)  # forward only -> does NOT invoke the in-flight step_backward
      max_nacon = max(max_nacon, int(d.nacon.numpy()[0]))
    self.assertGreater(max_nacon, 0, "two balls never collided; fix the fixture")
    self.assertFalse(m.njnt == 1 and mjm.nq == 7 and m.nv == 6, "expected the single-free-body contact gate OFF")

  @absltest.skipUnless(_MULTIBODY, "multi-body step_backward not supported yet (MJPLAN_ARTICULATION S4)")
  def test_two_ball_adjoint_gradient(self):
    """Real moving-normal adjoint test (active when multi-body support lands): taped
    d(||x2(T)||^2)/d(qvel0) vs guarded same-model FD, off-axis so the normal rotates."""
    mjm = mujoco.MjModel.from_xml_string(_TWO_BALL)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    T = 60
    qvel0 = np.zeros(mjm.nv)
    qvel0[0], qvel0[1] = 3.0, 0.3  # off-axis approach
    qvel0[6] = -3.0

    def loss_x2(qv):
      m = mjw.put_model(mjm)
      d = mjw.put_data(mjm, mjd)
      d.qvel = wp.array(qv.reshape(1, -1), dtype=wp.float32)
      for _ in range(T):
        mjw.step(m, d)
      x2 = d.qpos.numpy()[0, 7:9]
      return float(x2 @ x2)

    m = mjw.put_model(mjm)
    datas = [mjw.put_data(mjm, mjd) for _ in range(T + 1)]
    for dd in datas:
      dd.qvel.requires_grad = True
    datas[0].qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
    tape = wp.Tape()
    with tape:
      for t in range(T):
        mjw.step(m, datas[t], datas[t + 1])
      wp.launch(_x2_sq_kernel, dim=1, inputs=[datas[T].qpos, 7], outputs=[loss])
    tape.backward(loss=loss)
    g = datas[0].qvel.grad.numpy()[0]
    fd = _fd_grad(loss_x2, qvel0, eps=1e-3)
    np.testing.assert_allclose(g[:6], fd[:6], rtol=5e-2, atol=5e-3)


# ============================================================================
# Group 2 -- USEFULNESS (SLOW; FoBG from our adjoint vs ZoBG vs quadrature truth)
# ============================================================================


class FobgUsefulnessTest(parameterized.TestCase):
  def test_heaviside_reference(self):
    """Pure-numpy reference (FAST) validating the FoBG/ZoBG machinery against closed forms.
    Hard Heaviside (0/1, Suh Ex 3.3): FoBG=0 (biased), baselined ZoBG unbiased to phi/sigma.
    Soft Heaviside (-1/1, AHAC Eq 4): FoBG mean matches the closed form; variance grows ~1/nu."""
    rng = np.random.default_rng(0)
    theta, sigma, N = 0.3, 1.0, 200000
    w = rng.normal(0.0, sigma, size=N)
    a = theta + w

    truth_hard = _norm_pdf(theta / sigma) / sigma
    fobg_hard = np.zeros(N)
    L0 = 1.0 if theta >= 0 else 0.0
    zobg_hard = ((a >= 0.0).astype(float) - L0) * w / (sigma * sigma)
    print(f"\n[hard Heaviside] truth={truth_hard:.4f}  FoBG={fobg_hard.mean():.4f}  ZoBG={zobg_hard.mean():.4f}")
    self.assertAlmostEqual(fobg_hard.mean(), 0.0, places=6)
    self.assertGreater(abs(fobg_hard.mean() - truth_hard), 0.2)
    self.assertAlmostEqual(zobg_hard.mean(), truth_hard, delta=0.02)

    def soft_cf(nu):
      return (2.0 / nu) * (_norm_cdf((nu / 2.0 - theta) / sigma) - _norm_cdf((-nu / 2.0 - theta) / sigma))

    variances = {}
    for nu in (1.0, 0.05):
      hp = np.where(np.abs(a) <= nu / 2.0, 2.0 / nu, 0.0)
      variances[nu] = hp.var()
      self.assertAlmostEqual(hp.mean(), soft_cf(nu), delta=0.03 + 0.5 / math.sqrt(N))
    print(f"[soft Heaviside] Var[FoBG] nu=1.0->{variances[1.0]:.3f}  nu=0.05->{variances[0.05]:.3f}")
    self.assertGreater(variances[0.05], variances[1.0])

  @absltest.skipUnless(_SLOW, _SLOW_REASON)
  def test_zobg_unbiased_smooth(self):
    """Baselined ZoBG is an UNBIASED estimator of grad E[L] on the real sim: its Monte-Carlo
    mean (N=512, forward-only) matches the DETERMINISTIC Gauss-Hermite quadrature truth within a
    CI bound (an independent oracle -> a real estimator check, not a one-batch self-consistency).
    HONEST note: on a *continuous* loss the pathwise FoBG's bias is within MC noise and its
    variance is heavy-tailed (Gauss-Hermite ALIASES the oscillatory dL/dv, so it is NOT a
    reliable FoBG-moment estimator here). We therefore REPORT FoBG but do not assert on it -- the
    clean, robust FoBG failures are test_fobg_biased_on_discontinuous_event_loss (bias: FoBG=0)
    and test_fobg_variance_grows_with_contact_stiffness (variance). Those carry the relaxed-IFT
    motivation (§5.10)."""
    mjm = mujoco.MjModel.from_xml_string(_floor_xml())
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    qpos0 = mjd.qpos.copy()
    qpos0[2] = _FLOOR_Z0
    base_v, sigma = _FLOOR_VZ0, 0.3

    def loss_of(v):
      return _floor_final_z(mjm, mjd, qpos0, np.array([0, 0, v, 0, 0, 0.0]), _FLOOR_T)

    truth = _quad_grad_E(loss_of, base_v, sigma, deg=121)  # convergence-guarded
    L0 = loss_of(base_v)

    ws = np.random.default_rng(1).normal(0.0, sigma, size=512)
    zobg = np.array([(loss_of(base_v + w) - L0) * w / (sigma * sigma) for w in ws])
    se = math.sqrt(zobg.var() / len(ws))

    ws_f = np.random.default_rng(2).normal(0.0, sigma, size=32)
    fobg = np.array([_floor_taped(mjm, mjd, qpos0, np.array([0, 0, base_v + w, 0, 0, 0.0]), _FLOOR_T)[2][2] for w in ws_f])

    print(
      f"\n[smooth] grad E[L] truth={truth:.4f}"
      f"\n  baselined ZoBG: mean={zobg.mean():.4f} (SE {se:.4f})  -> unbiased (asserted vs independent truth)"
      f"\n  FoBG (report only): mean={fobg.mean():.4f}  sample_var={fobg.var():.4e}  (heavy-tailed; see the"
      f" discontinuous-event + stiffness tests for the robust FoBG-vs-ZoBG failures)"
    )
    self.assertLess(abs(zobg.mean() - truth), 4.0 * se + 0.02)  # ZoBG unbiased: MC mean vs independent GH truth

  @absltest.skipUnless(_SLOW, _SLOW_REASON)
  def test_fobg_biased_on_discontinuous_event_loss(self):
    """Suh Ex 3.3 ON OUR SIM with a DISCONTINUOUS event loss 1[z_N>tau]: the pathwise gradient
    (FoBG, our adjoint) is 0 a.s. -> BIASED, while baselined ZoBG recovers the quadrature truth
    grad P(z_N>tau) within a stated error bound. NOTE: a zero upstream cotangent does not
    exercise the contact adjoint -- this is an ESTIMATOR-bias reference, not adjoint validation
    (Oracle A carries adjoint correctness)."""
    mjm = mujoco.MjModel.from_xml_string(_floor_xml())
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    qpos0 = mjd.qpos.copy()
    qpos0[2] = _FLOOR_Z0
    base_v, sigma = _FLOOR_VZ0, 0.6
    thresh = _floor_final_z(mjm, mjd, qpos0, np.array([0, 0, base_v, 0, 0, 0.0]), _FLOOR_T)

    def ind(v):
      return 1.0 if _floor_final_z(mjm, mjd, qpos0, np.array([0, 0, v, 0, 0, 0.0]), _FLOOR_T) > thresh else 0.0

    truth = _quad_grad_E(ind, base_v, sigma, deg=161)
    self.assertGreater(abs(truth), 0.1, "pick a threshold where the event probability actually moves")

    # FoBG = adjoint of an indicator loss -> 0 a.s. (a handful of taped checks suffice)
    fobg = []
    for w in (-0.4, -0.1, 0.1, 0.4):
      _, _, gv = _floor_taped(mjm, mjd, qpos0, np.array([0, 0, base_v + w, 0, 0, 0.0]), _FLOOR_T, mode="indicator", thresh=thresh)
      fobg.append(gv[2])
    fobg = np.array(fobg)
    np.testing.assert_allclose(fobg, 0.0, atol=1e-5)

    # baselined ZoBG via Monte Carlo (forward-only) -> recovers truth within a CI bound.
    rng = np.random.default_rng(2)
    N = 256
    ws = rng.normal(0.0, sigma, size=N)
    L0 = ind(base_v)
    zobg = np.array([(ind(base_v + w) - L0) * w / (sigma * sigma) for w in ws])
    se = math.sqrt(zobg.var() / N)
    print(f"\n[discontinuous event] truth={truth:.4f}  FoBG=0  ZoBG mean={zobg.mean():.4f} (SE {se:.3f})")
    self.assertLess(abs(zobg.mean() - truth), 4.0 * se + 0.05, "ZoBG should recover the truth")
    self.assertGreater(abs(0.0 - truth), abs(zobg.mean() - truth))  # FoBG strictly more biased

  @absltest.skipUnless(_SLOW, _SLOW_REASON)
  def test_fobg_variance_grows_with_contact_stiffness(self):
    """Suh Ex 3.8 flavor on our sim: sweep contact stiffness; at each k validate adjoint==FD,
    then report per-sample FoBG and baselined-ZoBG variance. FoBG variance grows monotonically
    with stiffness (3-point), while baselined ZoBG stays far lower. (Restitution also changes
    with k -- reported, not held fixed.)"""
    qpos0 = None
    fobg_vars, zobg_vars, ehats = [], [], []
    ks = [2000.0, 10000.0, 50000.0]
    rng = np.random.default_rng(3)
    ws = rng.normal(0.0, 0.5, size=32)
    for k in ks:
      mjm = mujoco.MjModel.from_xml_string(_floor_xml(k=k, b=12.0))
      mjd = mujoco.MjData(mjm)
      mujoco.mj_forward(mjm, mjd)
      if qpos0 is None:
        qpos0 = mjd.qpos.copy()
        qpos0[2] = _FLOOR_Z0
      base = np.array([0, 0, _FLOOR_VZ0, 0, 0, 0.0])
      e_hat, _, _, _, _ = _floor_calibrate(mjm, mjd, qpos0, base, _FLOOR_T)
      ehats.append(e_hat)

      _, _, gv = _floor_taped(mjm, mjd, qpos0, base, _FLOOR_T)
      eps = 1e-3
      bp, bm = base.copy(), base.copy()
      bp[2] += eps
      bm[2] -= eps
      fd = (_floor_final_z(mjm, mjd, qpos0, bp, _FLOOR_T) - _floor_final_z(mjm, mjd, qpos0, bm, _FLOOR_T)) / (2 * eps)
      np.testing.assert_allclose(gv[2], fd, rtol=3e-2, atol=5e-3, err_msg=f"adjoint vs FD at k={k}")

      F, L = [], []
      L0 = _floor_final_z(mjm, mjd, qpos0, base, _FLOOR_T)
      for w in ws:
        qv = base.copy()
        qv[2] += w
        _, _, g = _floor_taped(mjm, mjd, qpos0, qv, _FLOOR_T)
        F.append(g[2])
        L.append(_floor_final_z(mjm, mjd, qpos0, qv, _FLOOR_T))
      F = np.array(F)
      Z = (np.array(L) - L0) * ws / (0.5 * 0.5)
      fobg_vars.append(float(F.var()))
      zobg_vars.append(float(Z.var()))
      print(f"[stiffness] k={k:>8.0f} e_hat={e_hat:.3f}  Var[FoBG]={F.var():.4e}  Var[ZoBG]={Z.var():.4e}")
    self.assertLess(fobg_vars[0], fobg_vars[1])
    self.assertLess(fobg_vars[1], fobg_vars[2])  # 3-point monotone

  @absltest.skipUnless(_SLOW and _adjoint_has_rho(), "ready hook: relaxed-IFT rho not in adjoint.py (MJPLAN §5.10)")
  def test_relaxed_ift_reduces_fobg_variance(self):
    """Dojo relaxed-IFT payoff: on a condim=3 SLIDING contact (a real tangential cone, so a
    cone apex exists), relaxing T->sqrt(T^2+rho^2) in the BACKWARD only should DROP the FoBG
    variance near the apex vs the hard cone, at fixed forward dynamics. Runs only when
    adjoint.py exposes a callable rho setter. Restores rho in finally."""
    mjm = mujoco.MjModel.from_xml_string(_SLIDING)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    qpos0 = mjd.qpos.copy()
    T = 40
    base = np.array([0.2, 0, -1.0, 0, 0, 0.0])  # small tangential slip + into the floor (near apex)

    try:
      # backward-only: the forward loss must be identical for any rho.
      l_a, _, _ = _floor_taped(mjm, mjd, qpos0, base, T)
      adjoint.set_cone_relaxation(1e-2)
      l_b, _, _ = _floor_taped(mjm, mjd, qpos0, base, T)
      self.assertAlmostEqual(l_a, l_b, places=5, msg="relaxation must be backward-only (forward unchanged)")

      def fobg_var(rho):
        adjoint.set_cone_relaxation(rho)
        rng = np.random.default_rng(11)
        vals = []
        for w in rng.normal(0.0, 0.05, size=24):
          qv = base.copy()
          qv[0] += w
          _, _, g = _floor_taped(mjm, mjd, qpos0, qv, T)
          vals.append(g[0])
        return float(np.var(vals))

      v_hard = fobg_var(0.0)
      v_relaxed = fobg_var(1e-2)
      print(f"\n[relaxed-IFT] Var[FoBG] hard={v_hard:.4e}  relaxed(rho=1e-2)={v_relaxed:.4e}")
      self.assertLess(v_relaxed, v_hard)
    finally:
      adjoint.set_cone_relaxation(0.0)


# ============================================================================
# Optional visualizations (opt-in: set MJW_ANALYTIC_PLOTS_DIR=<dir>). PNG + NPZ.
# Each panel maps to a Zhong / Suh / AHAC figure (see ../../_scratch/ANALYTIC.md).
# Uses ONLY numpy + the single-free-body taped backward (the stable path).
# ============================================================================


def _render_analytic_plots(outdir):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  os.makedirs(outdir, exist_ok=True)

  def _save(fig, name, **arrs):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=130)
    plt.close(fig)
    if arrs:
      np.savez(os.path.join(outdir, name + ".npz"), **arrs)

  # ---- Panel 1: Heaviside FoBG/ZoBG (Suh Fig 2; AHAC Fig 2 / Suh Fig 4) ----
  sigma = 1.0
  th = np.linspace(-1.5, 1.5, 161)
  F = np.array([_norm_cdf(t / sigma) for t in th])
  gradF = np.array([_norm_pdf(t / sigma) / sigma for t in th])
  rng = np.random.default_rng(0)
  th_z = np.linspace(-1.2, 1.2, 13)
  zobg_mean = np.array(
    [((((t + rng.normal(0, sigma, 8000)) >= 0).astype(float) - (1.0 if t >= 0 else 0.0)) * rng.normal(0, sigma, 8000) / sigma**2).mean() for t in th_z]
  )
  xg = np.linspace(-1.5, 1.5, 161)
  hbar = lambda x, nu: np.clip(2.0 * x / nu, -1.0, 1.0)
  nus = np.geomspace(0.02, 1.0, 40)
  th0 = 0.3
  p = np.array([_norm_cdf((nu / 2 - th0) / sigma) - _norm_cdf((-nu / 2 - th0) / sigma) for nu in nus])
  var_fobg_soft = (2.0 / nus) ** 2 * p * (1 - p)

  fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
  ax[0, 0].plot(th, F, label="F(θ)=E[H]")
  ax[0, 0].plot(th, gradF, label="∇F (true)")
  ax[0, 0].axhline(0, ls="--", c="r", label="FoBG ≡ 0 (biased)")
  ax[0, 0].set_title("Hard Heaviside: FoBG biased to 0   [Suh Fig 2]")
  ax[0, 0].set_xlabel("θ")
  ax[0, 0].legend(fontsize=8)
  ax[0, 1].plot(xg, hbar(xg, 1.0), label="ν=1.0")
  ax[0, 1].plot(xg, hbar(xg, 0.3), label="ν=0.3")
  ax[0, 1].set_title("Soft Heaviside H̄_ν (AHAC Eq 4)   [AHAC Fig 2 left]")
  ax[0, 1].set_xlabel("x")
  ax[0, 1].legend(fontsize=8)
  ax[1, 0].plot(th, gradF, label="∇F (true)")
  ax[1, 0].axhline(0, ls="--", c="r", label="FoBG ≡ 0")
  ax[1, 0].plot(th_z, zobg_mean, "o", ms=4, label="ZoBG (MC)")
  ax[1, 0].set_title("FoBG vs ZoBG vs ∇F   [Suh Fig 2 mid]")
  ax[1, 0].set_xlabel("θ")
  ax[1, 0].legend(fontsize=8)
  ax[1, 1].loglog(nus, var_fobg_soft, "-o", ms=3)
  ax[1, 1].set_title("Var[FoBG] vs slip tol ν (soft)   [AHAC Fig 2 right / Suh Fig 4]")
  ax[1, 1].set_xlabel("ν")
  ax[1, 1].set_ylabel("Var[FoBG]")
  _save(fig, "p1_heaviside_suh_fig2_ahac_fig2", theta=th, F=F, gradF=gradF, zobg_theta=th_z, zobg_mean=zobg_mean, nu=nus, var_fobg_soft=var_fobg_soft)

  # ---- Panel 2: bounce gradient vs v0 (Suh Fig 7; Oracle A grid-phase artifact) ----
  mjm = mujoco.MjModel.from_xml_string(_floor_xml())
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  qpos0 = mjd.qpos.copy()
  qpos0[2] = _FLOOR_Z0
  v0s = np.linspace(-7.5, -4.5, 31)
  Lv, fo, fd = (np.zeros_like(v0s) for _ in range(3))
  dz = np.array([0, 0, 1e-3, 0, 0, 0.0])
  for i, v in enumerate(v0s):
    qv = np.array([0, 0, v, 0, 0, 0.0])
    Lv[i] = _floor_final_z(mjm, mjd, qpos0, qv, _FLOOR_T)
    fo[i] = _floor_taped(mjm, mjd, qpos0, qv, _FLOOR_T)[2][2]
    fd[i] = (_floor_final_z(mjm, mjd, qpos0, qv + dz, _FLOOR_T) - _floor_final_z(mjm, mjd, qpos0, qv - dz, _FLOOR_T)) / 2e-3
  e_hat = _floor_calibrate(mjm, mjd, qpos0, np.array([0, 0, _FLOOR_VZ0, 0, 0, 0.0]), _FLOOR_T)[0]
  ideal = -e_hat * (_FLOOR_T * _FLOOR_DT) * np.ones_like(v0s)
  fig, ax = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
  ax[0].plot(v0s, Lv)
  ax[0].set_ylabel("final z = L(v₀)")
  ax[0].set_title("Bounce: loss vs initial v_z   [Suh Fig 7 top]")
  ax[1].plot(v0s, fo, "-o", ms=3, label="adjoint FoBG")
  ax[1].plot(v0s, fd, "--", label="central FD")
  ax[1].plot(v0s, ideal, ":", c="k", label=f"ideal −ê·T (ê={e_hat:.2f})")
  ax[1].set_xlabel("initial v_z")
  ax[1].set_ylabel("d(z_N)/d(v_z0)")
  ax[1].set_title("adjoint==FD (grid-phase artifact) vs ideal   [Oracle A; DiffTaichi TOI analog]")
  ax[1].legend(fontsize=8)
  _save(fig, "p2_bounce_grad_vs_v0_suh_fig7", v0=v0s, loss=Lv, fobg=fo, fd=fd, ideal=ideal, e_hat=np.array([e_hat]))

  # ---- Panel 3: Var[FoBG] vs Var[ZoBG] vs stiffness (Suh Fig 5) ----
  ks = np.array([2000.0, 5000.0, 10000.0, 20000.0, 50000.0])
  base = np.array([0, 0, _FLOOR_VZ0, 0, 0, 0.0])
  ws = np.random.default_rng(3).normal(0.0, 0.5, 32)
  vF, vZ = [], []
  for k in ks:
    mk = mujoco.MjModel.from_xml_string(_floor_xml(k=k, b=12.0))
    dk = mujoco.MjData(mk)
    mujoco.mj_forward(mk, dk)
    q0 = dk.qpos.copy()
    q0[2] = _FLOOR_Z0
    L0 = _floor_final_z(mk, dk, q0, base, _FLOOR_T)
    Fs, Zs = [], []
    for w in ws:
      qv = base.copy()
      qv[2] += w
      Fs.append(_floor_taped(mk, dk, q0, qv, _FLOOR_T)[2][2])
      Zs.append((_floor_final_z(mk, dk, q0, qv, _FLOOR_T) - L0) * w / 0.25)
    vF.append(float(np.var(Fs)))
    vZ.append(float(np.var(Zs)))
  fig, ax = plt.subplots(figsize=(7, 5))
  ax.loglog(ks, vF, "-o", label="Var[FoBG] (adjoint)")
  ax.loglog(ks, vZ, "-s", label="Var[baselined ZoBG]")
  ax.set_xlabel("contact stiffness k")
  ax.set_ylabel("variance")
  ax.set_title("FoBG vs ZoBG variance vs stiffness   [Suh Fig 5]")
  ax.legend()
  _save(fig, "p3_variance_vs_stiffness_suh_fig5", k=ks, var_fobg=np.array(vF), var_zobg=np.array(vZ))

  # ---- Panel 4: two-ball off-axis, moving normal (Zhong'23 Fig 4; Zhong'22 task3) ----
  x1_0, x2_0 = np.array([-1.0, 0.1]), np.array([1.0, -0.1])
  v1, v2 = np.array([3.0, 0.0]), np.array([-1.0, 0.0])
  collided, t_c, n, v1p, v2p = two_ball_offaxis_collision(x1_0, x2_0, v1, v2, 0.2)
  ts = np.linspace(0, 1.0, 200)
  p1 = np.array([x1_0 + v1 * t if t <= t_c else x1_0 + v1 * t_c + v1p * (t - t_c) for t in ts])
  p2 = np.array([x2_0 + v2 * t if t <= t_c else x2_0 + v2 * t_c + v2p * (t - t_c) for t in ts])
  xc = x1_0 + v1 * t_c
  fig, ax = plt.subplots(figsize=(7, 6))
  ax.plot(p1[:, 0], p1[:, 1], label="ball 1")
  ax.plot(p2[:, 0], p2[:, 1], label="ball 2")
  ax.plot([xc[0], xc[0] + n[0] * 0.6], [xc[1], xc[1] + n[1] * 0.6], "r-", lw=2, label="contact normal n̄")
  ax.plot(*xc, "k.", ms=8)
  ax.set_aspect("equal")
  ax.legend()
  ax.set_title(f"Two-ball off-axis: rotating normal (t_c={t_c:.3f})   [Zhong'23 Fig 4]")
  _save(fig, "p4_two_ball_offaxis_zhong23_fig4", p1=p1, p2=p2, t_c=np.array([t_c]), normal=n)

  # ---- Panel 5: Var[FoBG] vs rollout horizon H (AHAC Fig 3) ----
  Ts = [20, 30, 40, 50, 60, 80, 100]
  ws2 = np.random.default_rng(5).normal(0.0, 0.3, 24)
  vH = []
  for Th in Ts:
    Fs = []
    for w in ws2:
      qv = np.array([0, 0, _FLOOR_VZ0 + w, 0, 0, 0.0])
      Fs.append(_floor_taped(mjm, mjd, qpos0, qv, Th)[2][2])
    vH.append(float(np.var(Fs)))
  fig, ax = plt.subplots(figsize=(7, 5))
  ax.plot(Ts, vH, "-o")
  ax.axvline(38, ls="--", c="gray", label="≈ contact onset")
  ax.set_xlabel("rollout horizon H (steps)")
  ax.set_ylabel("Var[FoBG]")
  ax.set_title("FoBG variance vs horizon (grows after contact)   [AHAC Fig 3]")
  ax.legend()
  _save(fig, "p5_fobg_vs_horizon_ahac_fig3", H=np.array(Ts), var_fobg=np.array(vH))

  print(f"\n[render] wrote 5 panels (PNG+NPZ) to {outdir}")


class AnalyticPlotsTest(parameterized.TestCase):
  @absltest.skipUnless(os.environ.get("MJW_ANALYTIC_PLOTS_DIR"), "set MJW_ANALYTIC_PLOTS_DIR=<dir> to render figures")
  def test_render_analytic_plots(self):
    """Opt-in: renders the analytic-test visualizations (PNG + NPZ) to MJW_ANALYTIC_PLOTS_DIR.
    Maps to Zhong/Suh/AHAC figures (see ../../_scratch/ANALYTIC.md). numpy + single-free-body
    taped backward only (stable path)."""
    _render_analytic_plots(os.environ["MJW_ANALYTIC_PLOTS_DIR"])


if __name__ == "__main__":
  absltest.main()
