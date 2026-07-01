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
"""Tests for analytic gradients of ``mujoco_warp.step`` (see ../../MJPLAN.md).

This file is written *before* ``adjoint.py`` exists. It contains NO gradient
implementation. Its jobs are:

  * P0  -- *document* that ``step`` is currently NOT differentiable: every
           physics module sets ``wp.set_module_options({"enable_backward":
           False})`` (e.g. ``forward.py:49``), so recording ``step`` on a
           ``wp.Tape`` and calling ``tape.backward()`` produces zero/None
           input gradients (and a flood of native "may produce incorrect
           gradients ... enable_backward=False" warnings on stderr). We
           contrast that broken analytic-zero with finite differences of
           ``step``, which are clearly nonzero -- proving the zero gradient is
           wrong, not merely a degenerate state.  ``test_step_not_yet_differentiable``.

  * P1  -- a reusable ground-truth harness that future spikes (MJPLAN.md §8)
           reuse:
             - ``fd_transition``        : central FD of ``mujoco_warp.step``,
             - ``mjd_transition_fd``    : wrapper around MuJoCo C
                                          ``mjd_transitionFD``,
           both producing the transition Jacobians ``A = d(x')/dx`` and
           ``B = d(x')/du`` in the ``ndx = 2*nv + na`` tangent space
           (quaternion / free-joint aware via ``mj_differentiatePos`` /
           ``mj_integratePos``), plus a comparison util. The sanity test
           ``test_fd_matches_mjd_transition_fd`` asserts the two *ground truths*
           agree on a simple model (validating the harness, no analytic
           gradient involved).

  * P2  -- contact / non-contact fixtures for the (future) gradient spikes:
             - a cartpole (smooth, non-contact) fixture,
             - a batched, *contacting* fixture (box-on-plane; and a best-effort
               Unitree G1 ``home``-keyframe loader, see notes below).
           The contact-gradient test bodies are ``xfail``/``skip`` with
           reason="pending adjoint.py".

CONVENTION NOTE (for the reviewer): mujoco_warp's own test suite is class-based
(``absltest`` / ``parameterized.TestCase``) and is run under pytest
(``uv run --with pytest --with pytest-xdist pytest -n 2``; see
``.github/workflows/ci.yml``). The project owner prefers function-style tests,
but repo runnability comes first, so this file matches the existing
``forward_test.py`` / ``smooth_test.py`` class style. FLAGGED tension: if a
function-style file is preferred, these classes can be flattened into pytest
functions + fixtures without touching the harness logic below.

G1 NOTE (for the reviewer): ``benchmarks/unitree_g1/scene_flat.xml`` does NOT
load in this checkout -- the robot's *visual* geoms are ``type="mesh"`` and the
referenced ``assets/*.STL`` files are absent (``assets/`` holds only
``hfield.png``). MuJoCo loads all meshes at compile time even though the
*collision* geoms are primitives (foot capsules), so compilation raises
"Error opening file ... left_hip_pitch_link.STL". ``g1_contact_fixture`` is
therefore guarded: it skips cleanly when the meshes are missing. The
guaranteed-contacting fixture used by the scaffolding is the self-contained
inline ``box_on_plane`` model.

torch NOTE: ``torch`` is not installed in this environment, so P0 uses the pure
``wp.Tape`` path (mark leaf ``wp.array``s ``requires_grad=True``, seed
``out.grad``, ``tape.backward()``, read ``in.grad``) rather than the
``torch.autograd.Function`` bridge that ``adjoint.py`` will eventually expose.
"""

import os
import tempfile
import warnings

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data

# Path to mujoco_warp's bundled Unitree G1 (MJPLAN.md §11). See "G1 NOTE" above:
# its STL meshes may be absent, in which case the fixture skips.
_G1_SCENE = os.path.join(
  os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
  "benchmarks",
  "unitree_g1",
  "scene_flat.xml",
)


# ----------------------------------------------------------------------------
# Tiny models (kept inline so the file is self-contained and CPU-fast).
# ----------------------------------------------------------------------------

# Simplest possible actuated model for the P0 non-differentiability demo: one
# hinge + one motor, no contact. Loads in milliseconds on CPU.
_SINGLE_HINGE = """
<mujoco>
  <option timestep="0.01">
    <flag contact="disable"/>
  </option>
  <worldbody>
    <body pos="0 0 0">
      <joint name="j" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j" gear="1"/>
  </actuator>
</mujoco>
"""

# Free-joint + hinge, no contact: exercises the quaternion-aware tangent space
# (nq != nv) for the P1 harness sanity check. damping/stiffness make the
# velocity/position Jacobians nontrivial.
_FREE_PLUS_HINGE = """
<mujoco>
  <option timestep="0.005">
    <flag contact="disable"/>
  </option>
  <worldbody>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size=".05 .05 .05" mass="1"/>
      <body pos="0.1 0 0">
        <joint name="j" type="hinge" axis="0 1 0" damping="0.2" stiffness="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j" gear="1"/>
  </actuator>
</mujoco>
"""

# A textbook cartpole (slider + hinge), non-contact. The smooth-path baseline
# of MJPLAN.md Stage 3 / spike 4.
_CARTPOLE = """
<mujoco>
  <option timestep="0.01">
    <flag contact="disable"/>
  </option>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" damping="0.05"/>
      <geom type="box" size=".1 .05 .05" mass="1"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" gear="1"/>
  </actuator>
</mujoco>
"""

# A box resting on a plane: a guaranteed-to-load, guaranteed-contacting fixture.
# Elliptic cone + solimp[0]=0 + implicitfast match the MJPLAN.md differentiable
# config (§4). condim=3 friction=0.6 mirrors the G1 foot capsules.
_BOX_ON_PLANE = """
<mujoco>
  <option timestep="0.005" cone="elliptic" integrator="implicitfast"
          tolerance="1e-8" iterations="100" ls_iterations="50">
    <flag contact="enable"/>
  </option>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05" condim="3" friction="0.6"/>
    <body name="box" pos="0 0 0.05">
      <freejoint/>
      <geom name="box" type="box" size=".05 .05 .05" condim="3" friction="0.6"
            solimp="0 0.95 0.001" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# A Z-up port of Newton's example_diffsim_ball / warp's example_bounce: a free
# sphere thrown toward a wall, bouncing off floor + wall, optimized (via its
# initial velocity) to land on a target. This is the multi-step differentiable
# task of MJPLAN.md Stage 1 / the killer-demo precursor (§6.6): the analog of
# the box_on_plane single-step Jacobian, but exercising a *rollout* + scalar
# loss (the natural wp.Tape -> tape.backward(loss) -> qvel.grad path).
#
# Numbers mirror example_diffsim_ball (which is itself Z-up): start (0,-0.5,1),
# init vel (0,5,-5), wall box at (0,2,1), target (0,-2,1.5). The downward
# vz=-5 forces a floor-contact event early in the rollout, so the contact
# gradient is on the differentiable path (a contact-free arc can't pass).
# Differentiable contact config (§4): elliptic cone, solimp[0]=0.
_BOUNCE = """
<mujoco>
  <option timestep="0.004" cone="elliptic" integrator="implicitfast"
          tolerance="1e-8" iterations="100" ls_iterations="50" gravity="0 0 -9.81">
    <flag contact="enable"/>
  </option>
  <default>
    <geom condim="3" friction="0.6" solref="0.02 1" solimp="0 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05"/>
    <geom name="wall" type="box" pos="0 2 1" size="1 0.25 1"/>
    <body name="ball" pos="0 -0.5 1.0">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Bounce rollout config (mirrors example_diffsim_ball's 0.6s episode).
_BOUNCE_T = 150  # steps at dt=0.004 -> 0.6s
_BOUNCE_QVEL0 = (0.0, 5.0, -5.0, 0.0, 0.0, 0.0)  # initial freejoint velocity (lin, ang)
_BOUNCE_TARGET = (0.0, -2.0, 1.5)


# ----------------------------------------------------------------------------
# Leaf-array helpers.
# ----------------------------------------------------------------------------

# The differentiable leaves / outputs of a step are these Data fields (MJPLAN.md
# §3). Nothing is allocated requires_grad today (io.py make_data: zero hits), so
# we flip it on the leaf arrays explicitly.
_STATE_LEAVES = ("qpos", "qvel", "act")
_CONTROL_LEAVES = ("ctrl", "qfrc_applied", "xfrc_applied")


def _mark_requires_grad(d, names):
  """Marks the named ``Data`` arrays ``requires_grad=True`` (allocates .grad)."""
  for name in names:
    getattr(d, name).requires_grad = True


# ----------------------------------------------------------------------------
# P1: ground-truth harness (reused by future spikes).
# ----------------------------------------------------------------------------


def _state_residual_tangent(mjm, qpos, qvel, act, qpos_ref, qvel_ref, act_ref):
  """Maps a perturbed next-state to the ``ndx`` tangent space about a reference.

  dx = [ mj_differentiatePos(qpos_ref -> qpos)  (nv) ,
         qvel - qvel_ref                         (nv) ,
         act  - act_ref                          (na) ].
  """
  nv = mjm.nv
  dpos = np.zeros(nv)
  # mj_differentiatePos(m, out, dt, qpos1, qpos2) -> out = (qpos2 - qpos1)/dt in
  # tangent space; quaternion/free-joint aware. dt=1 gives the raw difference.
  mujoco.mj_differentiatePos(mjm, dpos, 1.0, qpos_ref, qpos)
  return np.concatenate([dpos, qvel - qvel_ref, act - act_ref])


def fd_transition(mjm, mjd, eps=1e-4):
  """Central finite differences of ``mujoco_warp.step``.

  Returns (A, B) in the ndx = 2*nv + na tangent space:
    A = d(qpos', qvel', act') / d(qpos, qvel, act)   shape (ndx, ndx)
    B = d(qpos', qvel', act') / d(ctrl)              shape (ndx, nu)

  Perturbations to qpos use ``mj_integratePos`` so free-joint / ball quaternions
  stay on the manifold; outputs are mapped back with ``mj_differentiatePos``.
  The base state is read from ``mjd`` (so callers control warmstart / settling).
  A fresh ``mjw.Data`` is built per evaluation to avoid cross-contamination.
  """
  nv, na, nu = mjm.nv, mjm.na, mjm.nu
  ndx = 2 * nv + na

  m = mjw.put_model(mjm)

  q0 = mjd.qpos.copy()
  v0 = mjd.qvel.copy()
  a0 = mjd.act.copy()
  c0 = mjd.ctrl.copy()

  def step_from(qpos, qvel, act, ctrl):
    # Fresh Data inherits mjd's qacc_warmstart etc. (put_data copies from mjd).
    d = mjw.put_data(mjm, mjd)
    d.qpos = wp.array(qpos.reshape(1, -1), dtype=wp.float32)
    d.qvel = wp.array(qvel.reshape(1, -1), dtype=wp.float32)
    if na:
      d.act = wp.array(act.reshape(1, -1), dtype=wp.float32)
    d.ctrl = wp.array(ctrl.reshape(1, -1), dtype=wp.float32)
    mjw.step(m, d)
    qn = d.qpos.numpy()[0].copy()
    vn = d.qvel.numpy()[0].copy()
    an = d.act.numpy()[0].copy() if na else np.zeros(0)
    return qn, vn, an

  qn0, vn0, an0 = step_from(q0, v0, a0, c0)

  def tangent(qn, vn, an):
    return _state_residual_tangent(mjm, qn, vn, an, qn0, vn0, an0)

  A = np.zeros((ndx, ndx))
  B = np.zeros((ndx, nu))

  # position columns (perturb on the manifold)
  for i in range(nv):
    dq = np.zeros(nv)
    dq[i] = eps
    qp = q0.copy()
    mujoco.mj_integratePos(mjm, qp, dq, 1.0)
    qm = q0.copy()
    mujoco.mj_integratePos(mjm, qm, -dq, 1.0)
    plus = tangent(*step_from(qp, v0, a0, c0))
    minus = tangent(*step_from(qm, v0, a0, c0))
    A[:, i] = (plus - minus) / (2.0 * eps)

  # velocity columns
  for i in range(nv):
    vp = v0.copy()
    vp[i] += eps
    vm = v0.copy()
    vm[i] -= eps
    plus = tangent(*step_from(q0, vp, a0, c0))
    minus = tangent(*step_from(q0, vm, a0, c0))
    A[:, nv + i] = (plus - minus) / (2.0 * eps)

  # activation columns
  for i in range(na):
    ap = a0.copy()
    ap[i] += eps
    am = a0.copy()
    am[i] -= eps
    plus = tangent(*step_from(q0, v0, ap, c0))
    minus = tangent(*step_from(q0, v0, am, c0))
    A[:, 2 * nv + i] = (plus - minus) / (2.0 * eps)

  # control columns
  for i in range(nu):
    cp = c0.copy()
    cp[i] += eps
    cm = c0.copy()
    cm[i] -= eps
    plus = tangent(*step_from(q0, v0, a0, cp))
    minus = tangent(*step_from(q0, v0, a0, cm))
    B[:, i] = (plus - minus) / (2.0 * eps)

  return A, B


def mjd_transition_fd(mjm, mjd, eps=1e-6, centered=True):
  """MuJoCo C ``mjd_transitionFD`` -> (A, B) in the same ndx tangent space.

  A: (2*nv+na, 2*nv+na), B: (2*nv+na, nu). Runs an ``mj_forward`` and primes
  ``qacc_warmstart`` first, matching how ``test_data.fixture`` leaves the state.
  """
  nv, na, nu = mjm.nv, mjm.na, mjm.nu
  ndx = 2 * nv + na

  mujoco.mj_forward(mjm, mjd)
  mjd.qacc_warmstart = mjd.qacc.copy()

  A = np.zeros((ndx, ndx))
  B = np.zeros((ndx, nu))
  mujoco.mjd_transitionFD(mjm, mjd, eps, centered, A, B, None, None)
  return A, B


def jac_compare(name, got, ref, atol):
  """Returns a dict of error metrics for two Jacobian blocks (max abs/rel)."""
  got = np.asarray(got)
  ref = np.asarray(ref)
  abs_err = np.abs(got - ref)
  denom = np.maximum(np.abs(ref), 1e-9)
  rel_err = abs_err / denom
  return {
    "name": name,
    "max_abs": float(abs_err.max()) if abs_err.size else 0.0,
    "max_rel": float(rel_err.max()) if rel_err.size else 0.0,
    "ref_max": float(np.abs(ref).max()) if ref.size else 0.0,
    "within": bool(abs_err.max() <= atol) if abs_err.size else True,
  }


# ----------------------------------------------------------------------------
# P2: fixtures.
# ----------------------------------------------------------------------------


def cartpole_fixture(nworld=1, **kwargs):
  """Non-contact cartpole (mjm, mjd, m, d). Smooth-path baseline (spike 4)."""
  return test_data.fixture(xml=_CARTPOLE, nworld=nworld, **kwargs)


def box_on_plane_fixture(nworld=4, settle_steps=2, perturb=0.01, seed=0):
  """Batched, *contacting* box-on-plane fixture (guaranteed to load).

  Settles the box onto the floor (a few ``mj_step``s) so the contact set is
  active, then replicates to ``nworld`` worlds with small per-world position
  perturbations to the planar (x, y) base coordinates -- staying within the
  contacting regime (MJPLAN.md §8: a clean init + 1 step would silently test
  only the smooth path).

  Returns (mjm, mjd, m, d). Asserts contact is active (``d.nacon`` > 0).
  """
  mjm = mujoco.MjModel.from_xml_string(_BOX_ON_PLANE)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  for _ in range(settle_steps):
    mujoco.mj_step(mjm, mjd)
  mujoco.mj_forward(mjm, mjd)

  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)

  # Per-world planar perturbation of the free-joint base (qpos[0:2] = x, y),
  # within the contacting regime. World 0 is left unperturbed.
  if nworld > 1 and perturb:
    rng = np.random.default_rng(seed)
    qpos = d.qpos.numpy().copy()
    qpos[1:, 0:2] += rng.uniform(-perturb, perturb, size=(nworld - 1, 2))
    d.qpos = wp.array(qpos, dtype=wp.float32)

  mjw.forward(m, d)
  nacon = int(d.nacon.numpy()[0])
  assert nacon > 0, f"expected active contact, got nacon={nacon}"
  return mjm, mjd, m, d


def g1_meshes_available():
  """True iff the G1 scene compiles (its STL meshes are present)."""
  if not os.path.exists(_G1_SCENE):
    return False
  try:
    mujoco.MjModel.from_xml_path(_G1_SCENE)
    return True
  except Exception:
    return False


def g1_contact_fixture(nworld=4, settle_steps=2, perturb=0.02, seed=0):
  """Best-effort batched, contacting Unitree G1 fixture from the ``home`` keyframe.

  Inits from the ``home`` standing keyframe (base z ~= 0.78, feet on the floor),
  settles a couple steps so the foot capsules are in contact, then replicates to
  ``nworld`` with small perturbations of the home ``qpos`` (joint DOFs only, to
  stay near the standing stance). PD ``<position>`` actuators -> ``ctrl`` is the
  position target (kept at the keyframe ``ctrl``). See "G1 NOTE": skips if the
  meshes are absent.

  Returns (mjm, mjd, m, d).
  """
  if not g1_meshes_available():
    raise unittest_skip("G1 STL meshes absent (benchmarks/unitree_g1/assets); cannot compile scene_flat.xml.")

  mjm = mujoco.MjModel.from_xml_path(_G1_SCENE)
  mjd = mujoco.MjData(mjm)
  key_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "home")
  mujoco.mj_resetDataKeyframe(mjm, mjd, key_id)
  mujoco.mj_forward(mjm, mjd)
  for _ in range(settle_steps):
    mujoco.mj_step(mjm, mjd)
  mujoco.mj_forward(mjm, mjd)

  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)

  if nworld > 1 and perturb:
    rng = np.random.default_rng(seed)
    qpos = d.qpos.numpy().copy()
    # free-joint base = qpos[0:7]; perturb only the actuated joints (qpos[7:]).
    qpos[1:, 7:] += rng.uniform(-perturb, perturb, size=(nworld - 1, qpos.shape[1] - 7))
    d.qpos = wp.array(qpos, dtype=wp.float32)

  mjw.forward(m, d)
  return mjm, mjd, m, d


def unittest_skip(reason):
  """Returns an exception that ``absltest`` treats as a skip when raised."""
  return absltest.SkipTest(reason)


# ----------------------------------------------------------------------------
# P0: demonstrate step is currently NOT differentiable.
# ----------------------------------------------------------------------------


class StepNotDifferentiableTest(parameterized.TestCase):
  """Documents the pre-``adjoint.py`` state: analytic gradients of ``step`` are 0/None."""

  def test_step_not_yet_differentiable(self):
    """Tape-backward gives ~0/None input grads, while FD of ``step`` is nonzero.

    This is the headline P0 deliverable. Because every physics module is built
    with ``enable_backward=False`` (forward.py:49 etc.), recording ``step`` on a
    ``wp.Tape`` records launches with no adjoints; ``tape.backward()`` therefore
    leaves the leaf ``.grad`` arrays at zero. Warp also prints (to the native
    logger / stderr, NOT via Python ``warnings``) one
    "... may produce incorrect gradients ... enable_backward=False" warning per
    recorded kernel. We assert on the zeros (the deterministic signal) and prove
    they are *wrong* by contrasting with central FD of ``step``, which depends on
    qpos/qvel/ctrl with O(1) sensitivity.

    DELETE / FLIP this test once ``adjoint.py`` makes ``step`` differentiable
    (it should then become a real FD-vs-analytic check, i.e. spike 4).
    """
    mjm, mjd, m, d = test_data.fixture(xml=_SINGLE_HINGE, qvel_noise=0.05, ctrl_noise=0.2)

    # ---- analytic gradient via wp.Tape (the broken path) ----
    # leaves we differentiate wrt, plus the output (qvel) we seed a grad on:
    _mark_requires_grad(d, ("qpos", "qvel", "ctrl"))

    # Capture native warnings to stderr is not visible to Python's warnings
    # module; we still wrap in catch_warnings to keep the test output clean and
    # to confirm (documented) that *no* Python-level warning fires.
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      tape = wp.Tape()
      with tape:
        mjw.step(m, d)
      # seed dL/d(qvel') = 1 on every next-velocity component
      d.qvel.grad.fill_(1.0)
      tape.backward()

    qpos_grad = d.qpos.grad.numpy()
    qvel_grad = d.qvel.grad.numpy()
    ctrl_grad = d.ctrl.grad.numpy()

    # The whole point: analytic input gradients are identically zero (broken).
    self.assertEqual(np.abs(qpos_grad).max(), 0.0, "expected zero (broken) qpos grad")
    self.assertEqual(np.abs(ctrl_grad).max(), 0.0, "expected zero (broken) ctrl grad")
    # (qvel.grad is just the seed we wrote; no backward accumulation happened.)

    # The enable_backward warnings are emitted by the native logger, so the
    # Python warnings module sees none. Documented, not asserted as nonzero.
    py_eb_warnings = [w for w in caught if "enable_backward" in str(w.message)]

    # ---- contrast: finite differences of step ARE nonzero ----
    # Rebuild a clean base state and FD d(qvel')/d(ctrl) directly.
    mjm2, mjd2, m2, d2 = test_data.fixture(xml=_SINGLE_HINGE, qvel_noise=0.05, ctrl_noise=0.2)
    q0 = mjd2.qpos.copy()
    v0 = mjd2.qvel.copy()
    c0 = mjd2.ctrl.copy()

    def next_qvel(ctrl):
      dd = mjw.put_data(mjm2, mjd2)
      dd.qpos = wp.array(q0.reshape(1, -1), dtype=wp.float32)
      dd.qvel = wp.array(v0.reshape(1, -1), dtype=wp.float32)
      dd.ctrl = wp.array(ctrl.reshape(1, -1), dtype=wp.float32)
      mjw.step(m2, dd)
      return dd.qvel.numpy()[0].copy()

    eps = 1e-4
    cp = c0.copy()
    cp[0] += eps
    cm = c0.copy()
    cm[0] -= eps
    fd_dqvel_dctrl = (next_qvel(cp) - next_qvel(cm)) / (2.0 * eps)

    # step DOES depend on ctrl -> the analytic zero above is provably wrong.
    self.assertGreater(
      np.abs(fd_dqvel_dctrl).max(),
      1e-3,
      "FD of step wrt ctrl should be clearly nonzero",
    )

    # Surface the evidence in the test log (visible with pytest -s / -v).
    print(
      "\n[P0] step is NOT differentiable yet:"
      f"\n     analytic |d(qvel')/dctrl| via tape.backward() = {np.abs(ctrl_grad).max():.3e} (==0, broken)"
      f"\n     analytic |d(qvel')/dqpos| via tape.backward() = {np.abs(qpos_grad).max():.3e} (==0, broken)"
      f"\n     FD       |d(qvel')/dctrl| of mjw.step          = {np.abs(fd_dqvel_dctrl).max():.3e} (nonzero -> grad is wrong)"
      f"\n     python-level enable_backward warnings captured = {len(py_eb_warnings)}"
      " (0; warp logs them to stderr instead)"
    )


# ----------------------------------------------------------------------------
# P1: harness sanity -- the two ground truths must agree (no analytic grad).
# ----------------------------------------------------------------------------


class GroundTruthHarnessTest(parameterized.TestCase):
  """Validates the FD harness against MuJoCo C ``mjd_transitionFD``."""

  @parameterized.named_parameters(
    ("single_hinge", _SINGLE_HINGE),
    ("free_plus_hinge", _FREE_PLUS_HINGE),
    ("cartpole", _CARTPOLE),
  )
  def test_fd_matches_mjd_transition_fd(self, xml):
    """``fd_transition`` (FD of mjw.step) ~= ``mjd_transition_fd`` (MuJoCo C).

    Both are *ground truths* -> they must agree (this validates the harness
    itself; no analytic gradient is involved). The residual is dominated by
    mjw's float32 step + FD truncation (MuJoCo runs float64), hence the 1e-2
    tolerance from MJPLAN.md spike 1.
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, qvel_noise=0.05, ctrl_noise=0.2)

    A_ref, B_ref = mjd_transition_fd(mjm, mjd, eps=1e-6, centered=True)
    A_fd, B_fd = fd_transition(mjm, mjd, eps=1e-4)

    atol = 1e-2
    a_cmp = jac_compare("A", A_fd, A_ref, atol)
    b_cmp = jac_compare("B", B_fd, B_ref, atol)
    print(f"\n[P1:{self._testMethodName}] A: {a_cmp}\n             B: {b_cmp}")

    np.testing.assert_allclose(A_fd, A_ref, atol=atol, err_msg="A: fd_transition vs mjd_transitionFD")
    np.testing.assert_allclose(B_fd, B_ref, atol=atol, err_msg="B: fd_transition vs mjd_transitionFD")

  def test_fd_matches_mjd_transition_fd_contact(self):
    """Harness sanity in the CONTACT regime (box-on-plane) -- validates the FD
    reference we'll compare ``adjoint.py`` against in Stage 1, BEFORE any analytic
    gradient exists. So a Stage-1 mismatch can't be blamed on a bad FD reference.

    Findings (measured here, important for Stage 1):
      * Contact transition Jacobians have STIFF, large-magnitude entries -- e.g.
        ``A[8,2] = d(vz')/dz ~= -41`` (next vertical velocity vs penetration
        depth = the contact normal stiffness). Comparisons MUST be RELATIVE; a
        pure ``atol`` is meaningless next to an O(40) entry.
      * FD of a stiff contact is eps-sensitive: eps=1e-3 hits contact
        nonlinearity, eps=1e-5 drowns soft entries in float32 roundoff;
        ``eps=1e-4`` is the float32 sweet spot.
      * ``fd_transition`` (warmstart frozen, as the analytic IFT will treat it,
        MJPLAN.md §5.7) and MuJoCo C ``mjd_transitionFD`` (re-steps internally)
        agree to ~1% relative on the stiff entries -> ``mjd_transitionFD`` IS a
        usable contact oracle here (the warmstart difference is negligible at
        this solver tol). Stage 1 will compare adjoint.py against both with rtol.
    """
    mjm, mjd, _, _ = box_on_plane_fixture(nworld=1, perturb=0.0)

    A_ref, _ = mjd_transition_fd(mjm, mjd, eps=1e-6, centered=True)
    A_fd, _ = fd_transition(mjm, mjd, eps=1e-4)  # float32 sweet spot for stiff contact

    # Relative tol (rtol*|ref|) for the stiff entries + small atol for the ~0
    # entries. The lone ~1% offender is the magnitude-41 normal-stiffness term.
    rtol, atol = 3e-2, 1e-2
    big = np.abs(A_ref) > 1.0
    rel_big = float((np.abs(A_fd - A_ref)[big] / np.abs(A_ref)[big]).max()) if big.any() else 0.0
    print(
      f"\n[P1:contact box_on_plane] nv={mjm.nv}  A_ref_max={np.abs(A_ref).max():.1f}"
      f"  max|fd-ref|={np.abs(A_fd - A_ref).max():.3f}  max_rel(|ref|>1)={rel_big:.2%}"
    )
    np.testing.assert_allclose(A_fd, A_ref, rtol=rtol, atol=atol, err_msg="A(contact): fd_transition vs mjd_transitionFD")


# ----------------------------------------------------------------------------
# P2: fixtures + (pending) contact-gradient placeholders.
# ----------------------------------------------------------------------------


class ContactFixtureTest(parameterized.TestCase):
  """Builds the contact fixtures and asserts contact is active (scaffolding)."""

  def test_box_on_plane_contact_active(self):
    """The batched box-on-plane fixture loads and has active contact."""
    mjm, mjd, m, d = box_on_plane_fixture(nworld=4)
    self.assertGreater(int(d.nacon.numpy()[0]), 0)
    self.assertEqual(d.nworld, 4)

  def test_cartpole_fixture_loads(self):
    """The non-contact cartpole fixture loads and steps."""
    mjm, mjd, m, d = cartpole_fixture(nworld=2, qvel_noise=0.01, ctrl_noise=0.1)
    mjw.step(m, d)
    self.assertGreater(d.time.numpy()[0], 0.0)

  def test_g1_contact_fixture(self):
    """The batched G1 ``home``-keyframe fixture loads with feet in contact.

    Skips if the bundled G1 STL meshes are absent (see "G1 NOTE").
    """
    mjm, mjd, m, d = g1_contact_fixture(nworld=4)
    # G1 feet should be in/near contact from the home stance.
    self.assertGreater(int(d.nacon.numpy()[0]), 0)


class ContactGradientTest(parameterized.TestCase):
  """Contact-gradient spikes -- bodies pending ``adjoint.py`` (MJPLAN.md §8)."""

  @absltest.skip("pending adjoint.py: analytic contact gradient not implemented (MJPLAN.md Stage 1)")
  def test_box_on_plane_ift_core_transition_jacobian(self):
    """Spike 1: box-on-plane transition Jacobian, analytic (adjoint.py) vs FD.

    The IFT core is developed on the batched ``box_on_plane`` fixture (G1 meshes
    are deferred to Stage 4, MJPLAN.md §10.8). Once adjoint.py exists, compare its
    analytic A against ``fd_transition`` / ``mjd_transition_fd`` here using a
    RELATIVE tol (contact Jacobians have stiff O(40) entries; see
    ``test_fd_matches_mjd_transition_fd_contact``).
    """
    raise NotImplementedError

  @absltest.skip("pending adjoint.py: elliptic friction (sticking/sliding) gradient (MJPLAN.md Stage 2)")
  def test_g1_elliptic_friction_sticking_sliding(self):
    """Spike 2: feet sticking AND sliding vs mjd_transitionFD, single/multi-step."""
    raise NotImplementedError

  @absltest.skip("pending adjoint.py: tolerance sweep of IFT-vs-FD error (MJPLAN.md Stage 2 / spike 3)")
  def test_solver_tolerance_sweep(self):
    """Spike 3: IFT<->FD error as a function of opt.tolerance / iterations."""
    raise NotImplementedError

  def test_cartpole_transition_jacobian(self):
    """Spike 4 (LANDED): cartpole (slider+hinge, both damped -> the eulerdamp implicit-damping solve)
    smooth BPTT gradient vs a float64 MuJoCo-C FD of the same loss. The successor to
    ``test_step_not_yet_differentiable`` -- same model, now asserting the analytic grad MATCHES the
    reference instead of being zero. (Stronger-damping eulerdamp + the quaternion path: see
    ``SmoothGradientTest``.)"""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 3)")
    from mujoco_warp._src import adjoint  # noqa: F401  (registers the analytic step backward)

    mjm = mujoco.MjModel.from_xml_string(_CARTPOLE)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    ctrl = np.full(mjm.nu, 0.2, np.float32)
    analytic = _smooth_taped_grad(mjm, mjd, 4, "ctrl", ctrl, _sumsq_qpos_kernel, (1, mjm.nq))
    fd = _smooth_mjc_fd_grad(mjm, mjd, 4, "ctrl", ctrl, lambda q: float(np.dot(q, q)))
    _assert_smooth_grad(self, analytic, fd, cos_min=0.999, rel_max=2e-2, tag="cartpole(eulerdamp)")


# ----------------------------------------------------------------------------
# Multi-step differentiable bounce (port of Newton's example_diffsim_ball /
# warp's example_bounce). MJPLAN.md Stage 1 / killer-demo precursor (§6.6).
#
# DESIGN (decided with the user): step(m, d, d_out) is Newton-style -- it copies d's input
# state into d_out and advances d_out (forward + integrator), leaving d untouched (§6.3).
# The MULTI-STEP driver loop here keeps a distinct mjw.Data per step (the example_diffsim_ball
# states[t] pattern); each step is forward'd exactly once so its intermediates (qacc/efc/M)
# survive in d_out for the backward. adjoint.py registers a per-step analytic backward that step()
# injects as one tape.record_func (d_out.grad -> d.grad), chaining BPTT across the buffers.
#
# The natural Warp-only path the user wants working (no torch):
#     tape = wp.Tape()
#     with tape:
#         for t: step(m, datas[t], datas[t+1]);  loss = ||ball_pos - target||^2
#     tape.backward(loss=loss)   # -> datas[0].qvel.grad
# ----------------------------------------------------------------------------


@wp.kernel
def _bounce_loss_kernel(qpos: wp.array2d[float], target: wp.vec3, loss: wp.array[float]):
  """loss = ||ball_pos - target||^2 on world 0 (mirrors example_diffsim_ball.loss_kernel)."""
  delta = wp.vec3(qpos[0, 0] - target[0], qpos[0, 1] - target[1], qpos[0, 2] - target[2])
  loss[0] = wp.dot(delta, delta)


def _grad_available():
  """True iff adjoint.py exists (i.e. step() has an analytic backward registered)."""
  try:
    from mujoco_warp._src import adjoint  # noqa: F401

    return True
  except Exception:
    return False


def bounce_setup():
  """Builds the bounce model + initial state. Returns (mjm, mjd, qpos0, qvel0, target, T)."""
  mjm = mujoco.MjModel.from_xml_string(_BOUNCE)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  qpos0 = mjd.qpos.copy()  # freejoint: [x,y,z, qw,qx,qy,qz] = [0,-0.5,1, 1,0,0,0]
  qvel0 = np.array(_BOUNCE_QVEL0, dtype=np.float64)
  target = np.array(_BOUNCE_TARGET, dtype=np.float64)
  return mjm, mjd, qpos0, qvel0, target, _BOUNCE_T


def _bounce_forward_loss(m, mjm, mjd, qpos0, qvel0, target, T):
  """Forward-only rollout (no tape); returns (loss, max_nacon_over_rollout).

  No-grad rollout, so it reuses one Data with the in-place step (d_out=None).
  """
  d = mjw.put_data(mjm, mjd)
  d.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
  d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
  max_nacon = 0
  for _ in range(T):
    mjw.step(m, d)
    max_nacon = max(max_nacon, int(d.nacon.numpy()[0]))
  qn = d.qpos.numpy()[0][:3].astype(np.float64)
  loss = float(np.dot(qn - target, qn - target))
  return loss, max_nacon


def _bounce_fd_grad(m, mjm, mjd, qpos0, qvel0, target, T, eps=1e-4):
  """Central FD of the scalar loss w.r.t. qvel0[0:3] (float32 sweet spot)."""
  g = np.zeros(3)
  for i in range(3):
    vp = qvel0.copy()
    vp[i] += eps
    vm = qvel0.copy()
    vm[i] -= eps
    lp, _ = _bounce_forward_loss(m, mjm, mjd, qpos0, vp, target, T)
    lm, _ = _bounce_forward_loss(m, mjm, mjd, qpos0, vm, target, T)
    g[i] = (lp - lm) / (2.0 * eps)
  return g


def _taped_bounce_grad(mjm, mjd, qpos0, qvel0, target, T):
  """Analytic grad via wp.Tape over the multi-step rollout (REQUIRES adjoint.py).

  Returns (loss_value, d(loss)/d(qvel0[0:3])). The multi-step machinery is HERE:
  a distinct Data per step; step(m, datas[t], datas[t+1]) advances out-of-place and
  injects the per-step analytic backward (d_out.grad -> d.grad) that chains the rollout.
  """
  m = mjw.put_model(mjm)
  datas = [mjw.put_data(mjm, mjd) for _ in range(T + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  datas[0].qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32, requires_grad=True)
  datas[0].qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32, requires_grad=True)

  loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
  target_v = wp.vec3(float(target[0]), float(target[1]), float(target[2]))

  tape = wp.Tape()
  with tape:
    for t in range(T):
      # Newton-style out-of-place step: reads datas[t], writes datas[t+1] (step copies the
      # input state in and advances it). adjoint.py's hook records the per-step analytic backward
      # (datas[t+1].grad -> datas[t].grad), chaining BPTT over the distinct per-step buffers.
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_bounce_loss_kernel, dim=1, inputs=[datas[T].qpos, target_v], outputs=[loss])

  tape.backward(loss=loss)

  loss_val = float(loss.numpy()[0])
  qvel_grad = datas[0].qvel.grad.numpy()[0][:3].astype(np.float64).copy()
  return loss_val, qvel_grad


def _bounce_viz_xml(xyz):
  """_BOUNCE augmented for rendering ONLY: lights, lit checker floor + skybox, a red
  target marker, and the trajectory drawn as faint spheres (all contype=conaffinity=0,
  so they don't perturb physics). Geometry mirrors _BOUNCE."""
  traj = ""
  for i in range(0, len(xyz), 4):  # sample every 4 steps
    x, y, z = xyz[i]
    traj += (
      f'<geom type="sphere" size="0.04" pos="{x:.4f} {y:.4f} {z:.4f}" '
      f'rgba="0.2 0.5 1 0.5" contype="0" conaffinity="0"/>\n    '
    )
  tx, ty, tz = _BOUNCE_TARGET
  return f"""
<mujoco>
  <option gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="6 6" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 -1 4" dir="0 0.3 -1" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <geom name="floor" type="plane" size="0 0 .05" material="grid"/>
    <geom name="wall" type="box" pos="0 2 1" size="1 0.25 1" rgba="0.5 0.5 0.55 1"/>
    <geom name="target" type="sphere" size="0.13" pos="{tx} {ty} {tz}" rgba="1 0.1 0.1 0.7"
          contype="0" conaffinity="0"/>
    <body name="ball" pos="0 -0.5 1.0">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.1" rgba="0.9 0.7 0.3 1" mass="1"/>
    </body>
    {traj}
  </worldbody>
</mujoco>
"""


def render_bounce(path=None):
  """Renders the bounce scene + initial-velocity trajectory to a PNG (for inspection).

  Rolls out with MuJoCo C (mj_step), augments the scene with lights / target marker /
  trajectory spheres (_bounce_viz_xml), and saves via mujoco.Renderer + PIL. Returns
  the output path; raises if rendering deps (offscreen GL / PIL) are unavailable.
  """
  from PIL import Image  # optional, render-only dependency

  if path is None:
    path = os.path.join(tempfile.gettempdir(), "bounce_scene.png")

  mjm = mujoco.MjModel.from_xml_string(_BOUNCE)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "ball")
  mjd.qvel[:3] = _BOUNCE_QVEL0[:3]
  xyz = [mjd.xpos[bid].copy()]
  for _ in range(_BOUNCE_T):
    mujoco.mj_step(mjm, mjd)
    xyz.append(mjd.xpos[bid].copy())
  xyz = np.array(xyz)

  vm = mujoco.MjModel.from_xml_string(_bounce_viz_xml(xyz))
  vd = mujoco.MjData(vm)
  mujoco.mj_forward(vm, vd)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(vm, cam)
  cam.lookat = [0.0, 0.0, 0.8]
  cam.distance = 8.5
  cam.azimuth = 50.0
  cam.elevation = -18.0
  with mujoco.Renderer(vm, height=960, width=1280) as r:
    r.update_scene(vd, camera=cam)
    img = r.render()
  Image.fromarray(img).save(path)
  return path


class BounceDiffsimTest(parameterized.TestCase):
  """Multi-step differentiable bounce (port of example_diffsim_ball). MJPLAN.md Stage 1."""

  @parameterized.named_parameters(
    ("floor_geom0", 49, "floor", "ball", 2, 2, "CONE"),
    ("wall_geom1", 145, "ball", "wall", 1, 1, "CONE"),
    ("settled_unsaturated_solimp", 120, "floor", "ball", 2, 2, "QUADRATIC"),
  )
  def test_bounce_contact_residual_vjp_matches_fd(
    self, state_step, expected_geom0, expected_geom1, output_vel, input_pos, expected_state
  ):
    """Contact VJP matches FD for both geom orders and position-varying solimp."""
    from mujoco_warp._src import adjoint
    from mujoco_warp._src import types

    mjm, mjd, qpos0, qvel0, _, _ = bounce_setup()
    m = mjw.put_model(mjm)

    # Reach the requested hard-forward state without recording a tape.
    state = mjw.put_data(mjm, mjd)
    state.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
    state.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
    for _ in range(state_step):
      mjw.step(m, state)
    qpos = state.qpos.numpy()[0].copy()
    qvel = state.qvel.numpy()[0].copy()

    d0 = mjw.put_data(mjm, mjd)
    d1 = mjw.put_data(mjm, mjd)
    for d in (d0, d1):
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    d0.qpos = wp.array(qpos.reshape(1, -1), dtype=wp.float32, requires_grad=True)
    d0.qvel = wp.array(qvel.reshape(1, -1), dtype=wp.float32, requires_grad=True)
    mjw.step(m, d0, d1)

    geom0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, expected_geom0)
    geom1 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, expected_geom1)
    self.assertSequenceEqual(d1.contact.geom.numpy()[0].tolist(), [geom0, geom1])
    e0 = int(d1.contact.efc_address.numpy()[0, 0])
    self.assertEqual(int(d1.efc.state.numpy()[0, e0]), int(getattr(types.ConstraintState, expected_state)))

    seed = np.zeros((1, mjm.nv), dtype=np.float32)
    seed[0, output_vel] = 1.0
    d1.qvel.grad.assign(seed)
    adjoint.step_backward(m, d0, d1)
    analytic = float(d0.qpos.grad.numpy()[0, input_pos])

    def next_velocity(qpos_in):
      d = mjw.put_data(mjm, mjd)
      d.qpos = wp.array(qpos_in.reshape(1, -1), dtype=wp.float32)
      d.qvel = wp.array(qvel.reshape(1, -1), dtype=wp.float32)
      mjw.step(m, d)
      return float(d.qvel.numpy()[0, output_vel])

    eps = 1.0e-4
    qp, qm = qpos.copy(), qpos.copy()
    qp[input_pos] += eps
    qm[input_pos] -= eps
    finite_diff = (next_velocity(qp) - next_velocity(qm)) / (2.0 * eps)
    print(
      f"\n[bounce contact VJP] step={state_step} geom={[geom0, geom1]} "
      f"analytic={analytic:.6f} fd={finite_diff:.6f}"
    )
    np.testing.assert_allclose(analytic, finite_diff, rtol=2.0e-2, atol=5.0e-2)

  def test_bounce_scene_contacts_and_fd(self):
    """Runs NOW (no adjoint.py): validates the scene + the FD ground truth.

    Asserts (a) the rollout actually makes contact -- so the contact gradient is
    on the differentiable path, not a contact-free arc -- and (b) the FD gradient
    of the scalar loss w.r.t. initial velocity is finite and nonzero. Pins the
    ground truth BEFORE adjoint.py exists, so a Stage-1 analytic mismatch can't be
    blamed on a bad scene / FD reference (cf. test_fd_matches_..._contact).
    """
    mjm, mjd, qpos0, qvel0, target, T = bounce_setup()
    m = mjw.put_model(mjm)

    loss0, max_nacon = _bounce_forward_loss(m, mjm, mjd, qpos0, qvel0, target, T)
    g_fd = _bounce_fd_grad(m, mjm, mjd, qpos0, qvel0, target, T)

    print(f"\n[bounce scene] T={T} loss0={loss0:.4f} max_nacon={max_nacon}  fd d(loss)/d(qvel0)={g_fd}")
    self.assertGreater(max_nacon, 0, "rollout never made contact -- contact gradient not on path")
    self.assertTrue(np.isfinite(g_fd).all(), "FD gradient is not finite")
    self.assertGreater(np.abs(g_fd).max(), 1e-3, "FD gradient is ~0 -- loss insensitive to qvel0")

  @absltest.skipUnless(_grad_available(), "pending adjoint.py: differentiable step (MJPLAN.md Stage 1)")
  def test_bounce_diffsim_grad(self):
    """Analytic rollout gradient matches central FD of the scalar loss."""
    mjm, mjd, qpos0, qvel0, target, T = bounce_setup()
    m = mjw.put_model(mjm)

    _, g_analytic = _taped_bounce_grad(mjm, mjd, qpos0, qvel0, target, T)
    g_fd = _bounce_fd_grad(m, mjm, mjd, qpos0, qvel0, target, T)
    print(f"\n[bounce grad] analytic={g_analytic}\n              fd      ={g_fd}")
    np.testing.assert_allclose(g_analytic, g_fd, atol=5e-2, err_msg="bounce: analytic vs FD d(loss)/d(qvel0)")

  @absltest.skipUnless(_grad_available(), "pending adjoint.py: differentiable step (MJPLAN.md Stage 1)")
  def test_bounce_diffsim_optimization_smoke(self):
    """Fixed-step SGD improves the endpoint; contact-event jumps may be nonmonotone.

    This is deliberately separate from gradient correctness.  A hard active
    set makes the rollout loss nonsmooth, so a fixed learning rate has no
    per-iteration descent guarantee without clipping or a line search.
    """
    mjm, mjd, qpos0, qvel0, target, T = bounce_setup()
    rate = 0.02
    qvel = qvel0.copy()
    losses = []
    for _ in range(25):
      loss_val, g = _taped_bounce_grad(mjm, mjd, qpos0, qvel, target, T)
      losses.append(loss_val)
      qvel[:3] -= rate * g
    print(f"[bounce optim] loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    self.assertLess(losses[-1], losses[0], "loss did not improve overall")

  @absltest.skipUnless(os.environ.get("MJW_RENDER"), "set MJW_RENDER=1 to render the bounce scene to a PNG")
  def test_render_bounce_scene(self):
    """On-demand render of the bounce scene (lights + target marker + trajectory) for
    visual inspection -- NOT a correctness gate. Gated on MJW_RENDER=1 so normal test
    runs don't do offscreen GL or write files; output path overridable via
    MJW_RENDER_PATH (default <tmp>/bounce_scene.png)."""
    try:
      path = render_bounce(os.environ.get("MJW_RENDER_PATH"))
    except Exception as e:
      self.skipTest(f"rendering unavailable: {type(e).__name__}: {e}")
    print(f"\n[bounce render] wrote {path}")
    self.assertTrue(os.path.exists(path))


_SPIN_SLIDE = """
<mujoco>
  <option timestep="0.004" cone="{cone}" integrator="implicitfast" solver="Newton" iterations="50"
          gravity="0 0 -9.81">
    <flag eulerdamp="disable"/>
  </option>
  <default>
    <geom condim="{condim}" friction="0.8 0.1 0.05" solref="-3000 -2" solimp="0 0.95 0.001"/>
  </default>
  <worldbody>
    <geom type="plane" size="5 5 0.01"/>
    <body pos="0 0 0.099">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <keyframe><key qpos="0 0 0.099 1 0 0 0" qvel="1.0 0.5 -0.3 2.0 1.5 3.0"/></keyframe>
</mujoco>
"""

_SPIN_SLIDE_T = 6  # short rollout: the body is in contact + spinning + sliding throughout


@wp.kernel
def _sumsq_qvel_kernel(qvel: wp.array2d[float], loss: wp.array[float]):
  """loss = ||qvel||^2 on world 0 -- contact-only (quaternion-free) for an isotropic sphere."""
  j = wp.tid()
  wp.atomic_add(loss, 0, qvel[0, j] * qvel[0, j])


def _spin_slide_setup(cone, condim):
  mjm = mujoco.MjModel.from_xml_string(_SPIN_SLIDE.format(cone=cone, condim=condim))
  mjd = mujoco.MjData(mjm)
  mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
  mujoco.mj_forward(mjm, mjd)
  return mjm, mjd


def _spin_slide_forward_loss(m, mjm, mjd, qvel0, T):
  """Forward-only rollout (no tape); returns (||qvel_T||^2, max_nacon)."""
  d = mjw.put_data(mjm, mjd)
  d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
  max_nacon = 0
  for _ in range(T):
    mjw.step(m, d)
    max_nacon = max(max_nacon, int(d.nacon.numpy()[0]))
  v = d.qvel.numpy()[0].astype(np.float64)
  return float(np.dot(v, v)), max_nacon


def _spin_slide_fd_grad(m, mjm, mjd, qvel0, T, eps=1e-4):
  """Central FD of ||qvel_T||^2 w.r.t. qvel0 (all six dofs; float32 sweet spot)."""
  g = np.zeros(6)
  for i in range(6):
    vp = qvel0.copy(); vp[i] += eps
    vm = qvel0.copy(); vm[i] -= eps
    lp, _ = _spin_slide_forward_loss(m, mjm, mjd, vp, T)
    lm, _ = _spin_slide_forward_loss(m, mjm, mjd, vm, T)
    g[i] = (lp - lm) / (2.0 * eps)
  return g


def _taped_spin_slide_grad(mjm, mjd, qvel0, T):
  """Analytic d(||qvel_T||^2)/d(qvel0) through the taped out-of-place rollout (adjoint.py)."""
  m = mjw.put_model(mjm)
  datas = [mjw.put_data(mjm, mjd) for _ in range(T + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  datas[0].qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32, requires_grad=True)

  loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(T):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_sumsq_qvel_kernel, dim=mjm.nv, inputs=[datas[T].qvel, loss])
  tape.backward(loss=loss)
  return np.nan_to_num(datas[0].qvel.grad.numpy()[0].astype(np.float64).copy())


class ContactCondimTest(parameterized.TestCase):
  """Analytic contact gradient matches FD for every cone x condim, incl. the rotational rows."""

  @parameterized.named_parameters(
    ("elliptic_frictionless_1", "elliptic", 1),
    ("elliptic_3", "elliptic", 3),
    ("elliptic_torsion_4", "elliptic", 4),
    ("elliptic_rolling_6", "elliptic", 6),
    ("pyramidal_frictionless_1", "pyramidal", 1),
    ("pyramidal_3", "pyramidal", 3),
    ("pyramidal_torsion_4", "pyramidal", 4),
    ("pyramidal_rolling_6", "pyramidal", 6),
  )
  def test_condim_spin_slide_grad_matches_fd(self, cone, condim):
    """Spinning + sliding free sphere on an elastic plane: analytic d(||qvel_T||^2)/d(qvel0) matches
    central FD for cone in {elliptic, pyramidal} and condim in {1, 3, 4, 6} (4 = torsion, 6 = +rolling
    rotational rows). Verifies adjoint.py's dimid>=3 contact rows + the cdof-based Jacobian."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1)")
    from mujoco_warp._src import adjoint  # noqa: F401  (registers the analytic step backward)

    mjm, mjd = _spin_slide_setup(cone, condim)
    m = mjw.put_model(mjm)
    qvel0 = mjd.qvel.astype(np.float64).copy()

    analytic = _taped_spin_slide_grad(mjm, mjd, qvel0, _SPIN_SLIDE_T)
    fd = _spin_slide_fd_grad(m, mjm, mjd, qvel0, _SPIN_SLIDE_T)
    _, max_nacon = _spin_slide_forward_loss(m, mjm, mjd, qvel0, _SPIN_SLIDE_T)

    na, nf = np.linalg.norm(analytic), np.linalg.norm(fd)
    cos = float(analytic @ fd / (na * nf))
    rel = float(np.linalg.norm(analytic - fd) / (nf + 1e-9))
    print(
      f"\n[condim {cone} {condim}] nacon={max_nacon} cos={cos:.4f} relL2={rel:.4f}"
      f"\n  analytic={analytic}\n  fd      ={fd}"
    )
    self.assertGreater(max_nacon, 0, "rollout never made contact -- contact gradient not on path")
    self.assertGreater(cos, 0.99, f"gradient direction off: cos={cos:.4f}")
    self.assertLess(rel, 0.05, f"relative-L2 error {rel:.4f} too large")


# ----------------------------------------------------------------------------
# S1 smooth-path step-gradient regressions: the analytic backward matches a FLOAT64 MuJoCo-C central
# FD of the scalar rollout loss. Locks the implicit-damping (eulerdamp) integrator adjoint -- the
# adj(a) = M (M+dt*D)^-1 adj(a_damped) remap -- and the free-body quaternion d(quat)/d(omega) path.
# Same BPTT-vs-FD shape as ContactCondimTest; the float64 mjC FD (vs mjw's float32 step FD) gives
# tight (rel < 2e-2) bounds, so a regressed eulerdamp adjoint (rel ~ 0.085 at damping=1) is caught.
# ----------------------------------------------------------------------------

# Two damped hinges (damping=1.0) -> the implicit eulerdamp velocity solve is non-trivially engaged
# (this is the scene where a missing (M+dt*D)^-1 remap shows rel ~ 0.085 vs FD).
_EULERDAMP_ARM = """
<mujoco>
  <option gravity="0 0 -9.81" jacobian="sparse"/>
  <worldbody>
    <body><joint name="j0" type="hinge" axis="0 1 0" damping="1.0"/><geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
      <body pos="0 0 -0.5"><joint name="j1" type="hinge" axis="0 1 0" damping="1.0"/><geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/></body>
    </body>
  </worldbody>
  <actuator><motor joint="j0" gear="1"/><motor joint="j1" gear="1"/></actuator>
  <keyframe><key qpos="0.5 -0.3" qvel="0.1 -0.2" ctrl="0.5 -0.5"/></keyframe>
</mujoco>
"""

# Dzhanibekov / tennis-racket body (3 distinct principal inertias); spun about the INTERMEDIATE axis
# (free body, gravity off, no contact) -> pure asymmetric-inertia rotation + quaternion integration.
_GYRO = """
<mujoco>
  <option timestep="0.005" gravity="0 0 0"><flag contact="disable" constraint="disable"/></option>
  <worldbody>
    <body pos="0 0 0"><freejoint/>
      <geom type="box" pos="0.15 0 0" size="0.125 0.05 0.05" density="100"/>
      <geom type="box" pos="0 0 0" size="0.025 0.1 0.5" density="100"/>
    </body>
  </worldbody>
  <keyframe><key qvel="0 0 0 15 0.1 0.1"/></keyframe>
</mujoco>
"""


# Dzhanibekov body on a DAMPED BALL joint (quaternion qpos[0:4], 3 angular dofs) under Euler+eulerdamp.
# Unlike _GYRO (free joint, ZERO damping) this activates the implicit-damping solve TOGETHER WITH quaternion
# integration -- the configuration that exposes the eulerdamp late-remap (Stage 3).
_EULERDAMP_BALL = """
<mujoco>
  <option timestep="0.05" integrator="Euler" gravity="0 0 0"><flag contact="disable" constraint="disable"/></option>
  <worldbody>
    <body pos="0 0 0">
      <joint type="ball" damping="5"/>
      <geom type="box" pos="0.15 0 0" size="0.125 0.05 0.05" density="100"/>
      <geom type="box" pos="0 0 0" size="0.025 0.1 0.5" density="100"/>
    </body>
  </worldbody>
  <keyframe><key qvel="25 0.5 0.5"/></keyframe>
</mujoco>
"""


_IMPLICITFAST_DAMPED_HINGE = """
<mujoco>
  <option timestep="0.004" integrator="implicitfast" gravity="0 0 0"/>
  <worldbody>
    <body><joint name="hinge" type="hinge" axis="0 1 0" damping="2"/>
      <geom type="box" size="0.065 0.065 0.02" mass="0.676"/></body>
  </worldbody>
  <actuator><position joint="hinge" kp="40"/></actuator>
  <keyframe><key qpos="0.1" qvel="0.2" ctrl="0.3"/></keyframe>
</mujoco>
"""


# Same hinge but POLYNOMIAL (velocity-dependent) damping: damping="linear p0 p1" -> D(v) = 2 + 6|v| + 4.5v².
# Q = M + dt·D(v) now depends on qvel, so a_int does too -> exercises the Stage-4 ∂_v direct term.
_IMPLICITFAST_DAMPINGPOLY_HINGE = """
<mujoco>
  <option timestep="0.004" integrator="implicitfast" gravity="0 0 0"/>
  <worldbody>
    <body><joint name="hinge" type="hinge" axis="0 1 0" damping="2 3 1.5"/>
      <geom type="box" size="0.065 0.065 0.02" mass="0.676"/></body>
  </worldbody>
  <actuator><position joint="hinge" kp="40"/></actuator>
  <keyframe><key qpos="0.1" qvel="0.8" ctrl="0.3"/></keyframe>
</mujoco>
"""


# 2-link arm whose mass matrix M(qpos) depends strongly on the elbow angle -- unlike the single hinge
# above (constant M), this exercises the implicitfast ∂M/∂qpos term. Bent + fast + damped + large dt so
# the dropped direct term dominates the one-step ∂qvel'/∂qpos block (see the expected-fail test below).
_IMPLICITFAST_ARM = """
<mujoco>
  <option timestep="0.02" integrator="implicitfast" gravity="0 0 -9.81" jacobian="sparse"/>
  <worldbody>
    <body><joint name="j0" type="hinge" axis="0 1 0" damping="6"/><geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/>
      <body pos="0 0 -0.5"><joint name="j1" type="hinge" axis="0 1 0" damping="6"/><geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1"/></body>
    </body>
  </worldbody>
  <actuator><motor joint="j0" gear="1"/><motor joint="j1" gear="1"/></actuator>
  <keyframe><key qpos="0.5 -1.2" qvel="2.5 -3.5" ctrl="0.5 -0.5"/></keyframe>
</mujoco>
"""


@wp.kernel
def _sumsq_qpos_kernel(qpos: wp.array2d[float], loss: wp.array[float]):
  i, j = wp.tid()
  wp.atomic_add(loss, 0, qpos[i, j] * qpos[i, j])


@wp.kernel
def _quatvec_kernel(qpos: wp.array2d[float], loss: wp.array[float]):
  """loss = qx^2 + qy^2 + qz^2 = 1 - qw^2 (the free body's unit quaternion, qpos[4:7]); orientation-
  sensitive, unlike sum(qpos^2) where |quat|=1 contributes a constant 1 -> exercises d(quat)/d(omega)."""
  loss[0] = qpos[0, 4] * qpos[0, 4] + qpos[0, 5] * qpos[0, 5] + qpos[0, 6] * qpos[0, 6]


@wp.kernel
def _ballquat_kernel(qpos: wp.array2d[float], loss: wp.array[float]):
  """loss = qx^2 + qy^2 + qz^2 for a BALL joint (unit quaternion at qpos[0:4], vector part [1:4]) --
  orientation-sensitive; the ball-joint analog of _quatvec_kernel (which reads a freejoint's [4:7])."""
  loss[0] = qpos[0, 1] * qpos[0, 1] + qpos[0, 2] * qpos[0, 2] + qpos[0, 3] * qpos[0, 3]


@wp.kernel
def _assign_shared_ctrl(src: wp.array2d[float], dst: wp.array2d[float]):
  """Apply a shared control to a step via a copy KERNEL (NOT wp.copy): the kernel adjoint ACCUMULATES
  src.grad across the rollout, whereas wp.copy's adjoint overwrites -> a shared-leaf BPTT under-count."""
  i, j = wp.tid()
  dst[i, j] = src[i, j]


def _smooth_taped_grad(mjm, mjd, H, wrt, ctrl, loss_kernel, loss_dim):
  """Analytic d(loss)/d(`wrt`) through the out-of-place taped rollout (adjoint.py). wrt='ctrl' (one
  shared control applied every step) or 'qvel' (initial velocity); base state read from mjd."""
  m = mjw.put_model(mjm)
  datas = [mjw.put_data(mjm, mjd) for _ in range(H + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
    if mjm.nu:
      d.ctrl.requires_grad = True
  ctrl_src = None
  if mjm.nu:
    ctrl_src = wp.array(ctrl.reshape(1, -1).astype(np.float32), dtype=wp.float32, requires_grad=(wrt == "ctrl"))
  if wrt == "qvel":
    datas[0].qvel = wp.array(mjd.qvel.reshape(1, -1).astype(np.float32), dtype=wp.float32, requires_grad=True)

  loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(H):
      if mjm.nu:
        wp.launch(_assign_shared_ctrl, dim=ctrl_src.shape, inputs=[ctrl_src], outputs=[datas[t].ctrl])
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(loss_kernel, dim=loss_dim, inputs=[datas[H].qpos], outputs=[loss])
  tape.backward(loss=loss)

  leaf = ctrl_src if wrt == "ctrl" else datas[0].qvel
  return np.nan_to_num(leaf.grad.numpy()[0].astype(np.float64).copy())


def _smooth_mjc_fd_grad(mjm, mjd, H, wrt, ctrl, loss_fn, eps=1e-6):
  """Float64 MuJoCo-C central FD of the scalar loss w.r.t. qvel0 (wrt='qvel') or the per-step ctrl.
  Differentiating the flat qvel0/ctrl vector avoids any qpos tangent-space map; float64 mj_step (vs
  mjw's float32) removes the FD-truncation floor -> a clean, tight reference for the smooth path."""

  def rollout(qv, c):
    md = mujoco.MjData(mjm)
    md.qpos[:] = mjd.qpos
    md.qvel[:] = qv
    for _ in range(H):
      if mjm.nu:
        md.ctrl[:] = c
      mujoco.mj_step(mjm, md)
    return loss_fn(md.qpos)

  if wrt == "qvel":
    x0 = mjd.qvel.astype(np.float64).copy()
    g = np.zeros(mjm.nv)
    for i in range(mjm.nv):
      xp, xm = x0.copy(), x0.copy()
      xp[i] += eps
      xm[i] -= eps
      g[i] = (rollout(xp, ctrl) - rollout(xm, ctrl)) / (2.0 * eps)
    return g
  x0 = ctrl.astype(np.float64).copy()
  g = np.zeros(mjm.nu)
  for i in range(mjm.nu):
    xp, xm = x0.copy(), x0.copy()
    xp[i] += eps
    xm[i] -= eps
    g[i] = (rollout(mjd.qvel, xp) - rollout(mjd.qvel, xm)) / (2.0 * eps)
  return g


def _assert_smooth_grad(testcase, analytic, fd, cos_min, rel_max, tag):
  analytic = np.asarray(analytic, float)
  fd = np.asarray(fd, float)
  na, nf = np.linalg.norm(analytic), np.linalg.norm(fd)
  cos = float(analytic @ fd / (na * nf)) if na > 0 and nf > 0 else 1.0
  rel = float(np.linalg.norm(analytic - fd) / (nf + 1e-12))
  print(f"\n[{tag}] cos={cos:.5f} relL2={rel:.4f}\n  analytic={analytic}\n  fd      ={fd}")
  testcase.assertGreater(nf, 1e-9, f"{tag}: FD gradient ~0 (loss not excited)")
  testcase.assertGreater(cos, cos_min, f"{tag}: gradient direction off, cos={cos:.5f}")
  testcase.assertLess(rel, rel_max, f"{tag}: relative-L2 error {rel:.4f} too large")


class SmoothGradientTest(parameterized.TestCase):
  """S1 smooth-path (no-contact) step gradient: analytic adjoint.py BPTT grad vs float64 MuJoCo-C FD.
  Locks the eulerdamp implicit-damping integrator adjoint and the free-body quaternion path."""

  @parameterized.named_parameters(
    ("eulerdamp_arm_ctrl", "ctrl"),
    ("eulerdamp_arm_qvel", "qvel"),
  )
  def test_eulerdamp_arm_grad_matches_fd(self, wrt):
    """Damped 2-hinge arm (damping=1.0 -> the implicit (M+dt*D)^-1 eulerdamp solve). Analytic
    d(sum qpos^2)/d(`wrt`) vs float64 FD: a regressed eulerdamp adjoint shows rel ~ 0.085 here."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm = mujoco.MjModel.from_xml_string(_EULERDAMP_ARM)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    ctrl = mjd.ctrl.astype(np.float32).copy()
    analytic = _smooth_taped_grad(mjm, mjd, 4, wrt, ctrl, _sumsq_qpos_kernel, (1, mjm.nq))
    fd = _smooth_mjc_fd_grad(mjm, mjd, 4, wrt, ctrl, lambda q: float(np.dot(q, q)))
    _assert_smooth_grad(self, analytic, fd, cos_min=0.999, rel_max=2e-2, tag=f"eulerdamp:{wrt}")

  def test_gyroscopic_orientation_grad_matches_fd(self):
    """Dzhanibekov body (3 distinct inertias) spun about the intermediate axis: analytic
    d(sum quat_vec^2)/d(qvel0) vs float64 FD -> locks the asymmetric-inertia rotational dynamics AND
    the quaternion-integration adjoint (the historical free_body_quat gap, here truly exercised since
    the quat-vec loss is orientation-sensitive)."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm = mujoco.MjModel.from_xml_string(_GYRO)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    analytic = _smooth_taped_grad(mjm, mjd, 10, "qvel", np.zeros(0, np.float32), _quatvec_kernel, 1)
    fd = _smooth_mjc_fd_grad(mjm, mjd, 10, "qvel", np.zeros(0, np.float32),
                             lambda q: float(q[4] ** 2 + q[5] ** 2 + q[6] ** 2))
    _assert_smooth_grad(self, analytic, fd, cos_min=0.99, rel_max=5e-2, tag="gyroscopic")

  def test_eulerdamp_ball_orientation_grad_matches_fd(self):
    """Stage 3 (eulerdamp reorder): a DAMPED ball joint under Euler+eulerdamp, orientation loss
    d(quat_vec²)/d(qvel0). The eulerdamp backward currently replays _advance_state at the raw solver root
    a_s then remaps AFTERWARD, but the forward integrated a_u = (M+dt·D)⁻¹M a_s. Quaternion integration is
    NONLINEAR in qvel', so the linearization point matters -> the late remap is wrong for free/ball DOFs
    (invisible for scalar joints, which are affine in accel; the gyro test has zero damping so a_u=a_s).
    dt-dependent: negligible (<1%) at dt<=0.02, ~60% at dt=0.05 (this fixture). Fixed by reconstructing a_u
    before the replay (KEEP the transpose remap after). Oracle = float64 MuJoCo-C."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm = mujoco.MjModel.from_xml_string(_EULERDAMP_BALL)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    analytic = _smooth_taped_grad(mjm, mjd, 10, "qvel", np.zeros(0, np.float32), _ballquat_kernel, 1)
    fd = _smooth_mjc_fd_grad(mjm, mjd, 10, "qvel", np.zeros(0, np.float32),
                             lambda q: float(q[1] ** 2 + q[2] ** 2 + q[3] ** 2))
    _assert_smooth_grad(self, analytic, fd, cos_min=0.99, rel_max=5e-2, tag="eulerdamp_ball")

  def test_implicitfast_damped_hinge_step_vjp_matches_fd(self):
    """The implicitfast advance uses a_int=(M-dt*dF/dv)^-1 M*a, not the solver's raw qacc."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint

    mjm = mujoco.MjModel.from_xml_string(_IMPLICITFAST_DAMPED_HINGE)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    q0, v0, u0 = float(mjd.qpos[0]), float(mjd.qvel[0]), float(mjd.ctrl[0])

    d0, d1 = mjw.put_data(mjm, mjd), mjw.put_data(mjm, mjd)
    for d in (d0, d1):
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    d0.ctrl.requires_grad = True
    mjw.step(m, d0, d1)
    d1.qvel.grad.assign(np.ones((1, 1), dtype=np.float32))
    adjoint.step_backward(m, d0, d1)
    analytic = np.array([
      d0.qpos.grad.numpy()[0, 0], d0.qvel.grad.numpy()[0, 0], d0.ctrl.grad.numpy()[0, 0]
    ], dtype=np.float64)

    def next_vel(q, v, u):
      d = mjw.put_data(mjm, mjd)
      d.qpos.assign(np.array([[q]], dtype=np.float32))
      d.qvel.assign(np.array([[v]], dtype=np.float32))
      d.ctrl.assign(np.array([[u]], dtype=np.float32))
      mjw.step(m, d)
      return float(d.qvel.numpy()[0, 0])

    eps = 1.0e-4
    fd = np.array([
      (next_vel(q0 + eps, v0, u0) - next_vel(q0 - eps, v0, u0)) / (2.0 * eps),
      (next_vel(q0, v0 + eps, u0) - next_vel(q0, v0 - eps, u0)) / (2.0 * eps),
      (next_vel(q0, v0, u0 + eps) - next_vel(q0, v0, u0 - eps)) / (2.0 * eps),
    ])
    np.testing.assert_allclose(analytic, fd, rtol=2.0e-3, atol=2.0e-3)

  def test_implicitfast_dampingpoly_qvel_vjp_matches_fd(self):
    """Stage 4: dampingpoly makes Q = M + dt·D(v) depend on qvel, so the one-step ∂qvel'/∂qvel carries a
    DIRECT term −∂_v[ yᵀ dt·D(v) a_u ] (D(v) = 2 + 6|v| + 4.5v² here). Analytic ∂qvel'/∂{q,v,u} matches FD of
    mjw.step; WITHOUT the Stage-4 term the qvel channel is wrong -- the nonlinear-damping analog of the
    constant-damping hinge test above."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint

    mjm = mujoco.MjModel.from_xml_string(_IMPLICITFAST_DAMPINGPOLY_HINGE)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    q0, v0, u0 = float(mjd.qpos[0]), float(mjd.qvel[0]), float(mjd.ctrl[0])

    d0, d1 = mjw.put_data(mjm, mjd), mjw.put_data(mjm, mjd)
    for d in (d0, d1):
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    d0.ctrl.requires_grad = True
    mjw.step(m, d0, d1)
    d1.qvel.grad.assign(np.ones((1, 1), dtype=np.float32))
    adjoint.step_backward(m, d0, d1)
    analytic = np.array([
      d0.qpos.grad.numpy()[0, 0], d0.qvel.grad.numpy()[0, 0], d0.ctrl.grad.numpy()[0, 0]
    ], dtype=np.float64)

    def next_vel(q, v, u):
      d = mjw.put_data(mjm, mjd)
      d.qpos.assign(np.array([[q]], dtype=np.float32))
      d.qvel.assign(np.array([[v]], dtype=np.float32))
      d.ctrl.assign(np.array([[u]], dtype=np.float32))
      mjw.step(m, d)
      return float(d.qvel.numpy()[0, 0])

    eps = 1.0e-4
    fd = np.array([
      (next_vel(q0 + eps, v0, u0) - next_vel(q0 - eps, v0, u0)) / (2.0 * eps),
      (next_vel(q0, v0 + eps, u0) - next_vel(q0, v0 - eps, u0)) / (2.0 * eps),
      (next_vel(q0, v0, u0 + eps) - next_vel(q0, v0, u0 - eps)) / (2.0 * eps),
    ])
    np.testing.assert_allclose(analytic, fd, rtol=3.0e-3, atol=3.0e-3)

  def test_implicitfast_config_dependent_mass_qpos_jacobian_matches_fd(self):
    """implicitfast advances a_int = Q^-1 M a_solver (Q = M + dt*D), so the one-step ∂qvel'/∂qpos block
    carries a DIRECT term yᵀ[(∂M/∂qpos)a_solver − (∂Q/∂qpos)a_int] from M(qpos)/Q(qpos). advance_backward
    remaps the solver root (M Q^-1) but DROPS that term -- fine for ~constant M (the single hinge above, or
    the ping-pong paddle whose hinges barely move), WRONG for a config-dependent mass matrix. On a bent,
    fast, damped 2-link arm at large dt the dropped term dominates: the analytic ∂qvel'/∂qpos is off by
    >100% vs float64 MuJoCo-C (the a_solver-root remap + every other channel are already correct -- the
    single-hinge test passes). EXPECTED-FAIL until the ∂M/∂qpos VJP lands (general implicitfast support);
    remove the decorator then."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint

    mjm = mujoco.MjModel.from_xml_string(_IMPLICITFAST_ARM)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    nv = mjm.nv

    # analytic one-step d(qvel1)/d(qpos0): seed each qvel1 row, read the qpos0 adjoint (the arm has no
    # quaternion, so the qpos tangent is trivial). Row k = d(qvel1_k)/d(qpos0).
    ana = np.zeros((nv, nv))
    for k in range(nv):
      d0, d1 = mjw.put_data(mjm, mjd), mjw.put_data(mjm, mjd)
      for d in (d0, d1):
        d.qpos.requires_grad = True
        d.qvel.requires_grad = True
      d0.ctrl.requires_grad = True
      mjw.step(m, d0, d1)
      seed = np.zeros((1, nv), np.float32)
      seed[0, k] = 1.0
      d1.qvel.grad.assign(seed)
      adjoint.step_backward(m, d0, d1)
      ana[k, :] = d0.qpos.grad.numpy()[0].astype(np.float64)

    # float64 MuJoCo-C FD reference for d(qvel1)/d(qpos0).
    def next_qvel(q):
      md = mujoco.MjData(mjm)
      md.qpos[:] = q
      md.qvel[:] = mjd.qvel
      md.ctrl[:] = mjd.ctrl
      mujoco.mj_step(mjm, md)
      return md.qvel.copy()

    eps = 1.0e-6
    fd = np.zeros((nv, nv))
    q0 = mjd.qpos.astype(np.float64)
    for j in range(nv):
      qp, qm = q0.copy(), q0.copy()
      qp[j] += eps
      qm[j] -= eps
      fd[:, j] = (next_qvel(qp) - next_qvel(qm)) / (2.0 * eps)

    rel = float(np.linalg.norm(ana - fd) / (np.linalg.norm(fd) + 1e-12))
    print(f"\n[implicitfast dM/dqpos] rel={rel:.4f}\n analytic=\n{ana}\n fd=\n{fd}")
    self.assertLess(rel, 2.0e-2, f"implicitfast ∂qvel'/∂qpos off by rel={rel:.4f} (dropped ∂M/∂qpos term)")


# ----------------------------------------------------------------------------
# Long-horizon implicitfast + articulated-contact BPTT regression.  A global rollout FD is a poor
# oracle here: perturbing a gain moves impact times and can switch contacts over 160 steps.  Instead,
# build MuJoCo-C's float64 one-step transition Jacobian at every state of the NOMINAL MJW trajectory,
# close each Jacobian through the same feedback law, then compose those local linearizations backward.
# This keeps the oracle on the nominal active-set sequence while still exposing any per-step adjoint
# error: an error that is tiny for one step compounds over H=160.  The scene mirrors the ping-pong
# reproducer but is embedded here so the test has no contrib/example dependency.
# ----------------------------------------------------------------------------

_IMPLICITFAST_CONTACT_HORIZON = 160
_IMPLICITFAST_CONTACT_TARGET_Z = 0.48


def _implicitfast_contact_xml(ball_x, pz_lo, pz_hi, pxy, rtilt):
  joints = (
    '<joint name="px" type="slide" axis="1 0 0" damping="8"/>'
    '<joint name="py" type="slide" axis="0 1 0" damping="8"/>'
    f'<joint name="pz" type="slide" axis="0 0 1" range="{pz_lo} {pz_hi}" damping="20"/>'
    '<joint name="rx" type="hinge" axis="1 0 0" damping="2"/>'
    '<joint name="ry" type="hinge" axis="0 1 0" damping="2"/>'
    '<joint name="rz" type="hinge" axis="0 0 1" damping="2"/>'
  )
  return f"""
<mujoco>
  <option timestep="0.004" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81"
          iterations="100" ls_iterations="50" tolerance="1e-8"><flag contact="enable"/></option>
  <default><geom condim="3" friction="0.4" solref="-16000 -4" solimp="0 0.95 0.001"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05"/>
    <body name="paddle" pos="0 0 0.25">{joints}
      <geom name="paddle_geom" type="box" size="0.065 0.065 0.02"/>
    </body>
    <body name="ball" pos="{ball_x} 0 0.30"><freejoint/>
      <geom name="ball_geom" type="sphere" size="0.025" mass="0.12"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="px" kp="700" ctrlrange="{-pxy} {pxy}"/>
    <position joint="py" kp="700" ctrlrange="{-pxy} {pxy}"/>
    <position joint="pz" kp="1500" ctrlrange="{pz_lo} {pz_hi}"/>
    <position joint="rx" kp="40" ctrlrange="{-rtilt} {rtilt}"/>
    <position joint="ry" kp="40" ctrlrange="{-rtilt} {rtilt}"/>
    <position joint="rz" kp="40" ctrlrange="{-rtilt} {rtilt}"/>
  </actuator>
</mujoco>
"""


@wp.kernel
def _implicitfast_contact_feedback(
  qpos: wp.array2d[float],
  qvel: wp.array2d[float],
  gains: wp.array[float],
  ball_q: int,
  ball_v: int,
  pxy: float,
  pz_lo: float,
  pz_hi: float,
  rtilt: float,
  ctrl: wp.array2d[float],
):
  w = wp.tid()
  bx, by, bz = qpos[w, ball_q], qpos[w, ball_q + 1], qpos[w, ball_q + 2]
  bvx, bvy, bvz = qvel[w, ball_v], qvel[w, ball_v + 1], qvel[w, ball_v + 2]
  ctrl[w, 0] = wp.clamp(bx, -pxy, pxy)
  ctrl[w, 1] = wp.clamp(by, -pxy, pxy)
  ctrl[w, 2] = wp.clamp(gains[0] - gains[1] * (bz - gains[3]) - gains[2] * bvz, pz_lo, pz_hi)
  ctrl[w, 3] = wp.clamp(gains[6] * by + gains[7] * bvy, -rtilt, rtilt)
  ctrl[w, 4] = wp.clamp(gains[4] * bx + gains[5] * bvx, -rtilt, rtilt)
  ctrl[w, 5] = 0.0


@wp.kernel
def _implicitfast_contact_stage_loss(
  qpos: wp.array2d[float], ball_q: int, inv_h: float, center_weight: float, loss: wp.array[float]
):
  bx, by = qpos[0, ball_q], qpos[0, ball_q + 1]
  dz = qpos[0, ball_q + 2] - _IMPLICITFAST_CONTACT_TARGET_Z
  wp.atomic_add(loss, 0, (dz * dz + center_weight * (bx * bx + by * by)) * inv_h)


def _implicitfast_contact_setup(offcenter):
  if offcenter:
    ball_x, center_weight = 0.05, 2.0
    pz_lo, pz_hi, pxy, rtilt = -0.15, 0.15, 0.40, 1.0
    gains = np.array([0.0, 0.8, 0.15, 0.40, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
  else:
    ball_x, center_weight = 0.0, 0.0
    pz_lo, pz_hi, pxy, rtilt = -0.06, 0.08, 0.30, 0.5
    gains = np.array([0.0, 0.0, 0.0, 0.44, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

  mjm = mujoco.MjModel.from_xml_string(_implicitfast_contact_xml(ball_x, pz_lo, pz_hi, pxy, rtilt))
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  free_jnt = int(np.flatnonzero(mjm.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
  ball_q = int(mjm.jnt_qposadr[free_jnt])
  ball_v = int(mjm.jnt_dofadr[free_jnt])
  bounds = (float(pxy), float(pz_lo), float(pz_hi), float(rtilt))
  return mjm, mjd, gains, ball_q, ball_v, bounds, float(center_weight)


def _implicitfast_contact_taped_grad(offcenter, horizon):
  mjm, mjd, gains_np, ball_q, ball_v, bounds, center_weight = _implicitfast_contact_setup(offcenter)
  pxy, pz_lo, pz_hi, rtilt = bounds
  m = mjw.put_model(mjm)
  gains = wp.array(gains_np.astype(np.float32), dtype=float, requires_grad=True)
  datas = [mjw.put_data(mjm, mjd) for _ in range(horizon + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
    d.ctrl.requires_grad = True
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(horizon):
      wp.launch(
        _implicitfast_contact_feedback,
        dim=1,
        inputs=[datas[t].qpos, datas[t].qvel, gains, ball_q, ball_v, pxy, pz_lo, pz_hi, rtilt],
        outputs=[datas[t].ctrl],
      )
      mjw.step(m, datas[t], datas[t + 1])
      wp.launch(
        _implicitfast_contact_stage_loss,
        dim=1,
        inputs=[datas[t + 1].qpos, ball_q, 1.0 / horizon, center_weight],
        outputs=[loss],
      )
  tape.backward(loss=loss)

  trajectory = {
    "qpos": [d.qpos.numpy()[0].astype(np.float64).copy() for d in datas],
    "qvel": [d.qvel.numpy()[0].astype(np.float64).copy() for d in datas],
    "ctrl": [d.ctrl.numpy()[0].astype(np.float64).copy() for d in datas[:-1]],
    "nacon": np.array([int(d.nacon.numpy()[0]) for d in datas[1:]]),
  }
  return gains.grad.numpy().astype(np.float64).copy(), trajectory, (
    mjm, gains_np, ball_q, ball_v, bounds, center_weight
  )


def _implicitfast_feedback_jacobians(qpos, qvel, gains, nv, nu, ball_q, ball_v, bounds):
  """Jacobian of the feedback control w.r.t. tangent state and its eight shared gains."""
  pxy, pz_lo, pz_hi, rtilt = bounds
  fx = np.zeros((nu, 2 * nv), dtype=np.float64)
  fp = np.zeros((nu, len(gains)), dtype=np.float64)
  bx, by, bz = qpos[ball_q : ball_q + 3]
  bvx, bvy, bvz = qvel[ball_v : ball_v + 3]

  if -pxy < bx < pxy:
    fx[0, ball_v] = 1.0
  if -pxy < by < pxy:
    fx[1, ball_v + 1] = 1.0

  z_cmd = gains[0] - gains[1] * (bz - gains[3]) - gains[2] * bvz
  if pz_lo < z_cmd < pz_hi:
    fx[2, ball_v + 2] = -gains[1]
    fx[2, nv + ball_v + 2] = -gains[2]
    fp[2, [0, 1, 2, 3]] = [1.0, -(bz - gains[3]), -bvz, gains[1]]

  x_tilt = gains[6] * by + gains[7] * bvy
  if -rtilt < x_tilt < rtilt:
    fx[3, ball_v + 1] = gains[6]
    fx[3, nv + ball_v + 1] = gains[7]
    fp[3, 6], fp[3, 7] = by, bvy

  y_tilt = gains[4] * bx + gains[5] * bvx
  if -rtilt < y_tilt < rtilt:
    fx[4, ball_v] = gains[4]
    fx[4, nv + ball_v] = gains[5]
    fp[4, 4], fp[4, 5] = bx, bvx
  return fx, fp


def _implicitfast_contact_local_bptt(trajectory, setup, horizon):
  """Compose float64 local mjd_transitionFD Jacobians backward along the nominal MJW trajectory."""
  mjm, gains, ball_q, ball_v, bounds, center_weight = setup
  nv, ndx = mjm.nv, 2 * mjm.nv + mjm.na
  closed_loop_a, gain_b = [], []
  for t in range(horizon):
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = trajectory["qpos"][t]
    mjd.qvel[:] = trajectory["qvel"][t]
    mjd.ctrl[:] = trajectory["ctrl"][t]
    mujoco.mj_forward(mjm, mjd)
    if int(mjd.ncon) != int(trajectory["nacon"][t]):
      raise AssertionError(
        f"local-FD oracle changed contact presence at t={t}: MJC={mjd.ncon}, MJW={trajectory['nacon'][t]}"
      )
    a, b_ctrl = mjd_transition_fd(mjm, mjd, eps=1.0e-6, centered=True)
    fx, fp = _implicitfast_feedback_jacobians(
      trajectory["qpos"][t], trajectory["qvel"][t], gains, nv, mjm.nu, ball_q, ball_v, bounds
    )
    closed_loop_a.append(a + b_ctrl @ fx)
    gain_b.append(b_ctrl @ fp)

  adj_state = np.zeros(ndx, dtype=np.float64)
  grad = np.zeros(len(gains), dtype=np.float64)
  for t in range(horizon - 1, -1, -1):
    bx, by, bz = trajectory["qpos"][t + 1][ball_q : ball_q + 3]
    stage_grad = np.zeros(ndx, dtype=np.float64)
    stage_grad[ball_v : ball_v + 3] = (
      2.0 * np.array([center_weight * bx, center_weight * by, bz - _IMPLICITFAST_CONTACT_TARGET_Z]) / horizon
    )
    adj_out = adj_state + stage_grad
    grad += gain_b[t].T @ adj_out
    adj_state = closed_loop_a[t].T @ adj_out
  return grad


class ImplicitfastLongHorizonTest(parameterized.TestCase):

  @parameterized.named_parameters(("centered", False), ("offcenter", True))
  def test_contact_feedback_bptt_matches_local_transition_fd(self, offcenter):
    """H=160 catches a small per-step reverse error that short/single-step gradient tests cannot."""
    analytic, trajectory, setup = _implicitfast_contact_taped_grad(offcenter, _IMPLICITFAST_CONTACT_HORIZON)
    reference = _implicitfast_contact_local_bptt(trajectory, setup, _IMPLICITFAST_CONTACT_HORIZON)
    self.assertTrue(np.isfinite(analytic).all(), f"analytic gradient is non-finite: {analytic}")
    self.assertTrue(np.isfinite(reference).all(), f"local-FD BPTT gradient is non-finite: {reference}")
    self.assertGreater(np.count_nonzero(trajectory["nacon"]), 0, "rollout never exercises contact")

    norm_a, norm_r = np.linalg.norm(analytic), np.linalg.norm(reference)
    cosine = float(analytic @ reference / (norm_a * norm_r))
    relative = float(np.linalg.norm(analytic - reference) / norm_r)
    print(
      f"\n[implicitfast contact H={_IMPLICITFAST_CONTACT_HORIZON} offcenter={offcenter}] "
      f"cos={cosine:+.6f} rel={relative:.4e}\n  analytic={analytic}\n  local-FD={reference}"
    )
    self.assertGreater(norm_r, 1.0e-6, "reference gradient is unobservable")
    self.assertGreater(cosine, 0.999, f"long-horizon gradient direction drifted: cos={cosine:.6f}")
    self.assertLess(relative, 5.0e-3, f"long-horizon gradient magnitude drifted: rel={relative:.4e}")

    if offcenter:
      # gains[4:6] command hinge-y from ball x/vx.  Their observability proves the off-center normal
      # moment arm really activated the rotational rows; a whole-vector comparison alone could be
      # dominated by the three vertical-juggle gains.
      tilt_a, tilt_r = analytic[4:6], reference[4:6]
      self.assertGreater(np.linalg.norm(tilt_r), 1.0e-5, "off-center hinge-y gain is not exercised")
      tilt_rel = float(np.linalg.norm(tilt_a - tilt_r) / np.linalg.norm(tilt_r))
      self.assertLess(tilt_rel, 1.0e-2, f"off-center rotational channel drifted: rel={tilt_rel:.4e}")


# Contact-free 2-hinge arm with per-joint armature + viscous damping (the PACE-clean smooth-param subset),
# EULER + eulerdamp off (so damping is EXPLICIT in qfrc_passive, matching _residual_smooth_local), excited
# by gravity (qacc) + an initial joint velocity (qvel) so both params are observable.
_SYSID_ARM = """
<mujoco>
  <option timestep="0.004" integrator="Euler" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <worldbody>
    <body pos="0 0 1"><joint name="j0" type="hinge" axis="0 1 0" armature="0.10" damping="0.50"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/>
      <body pos="0.3 0 0"><joint name="j1" type="hinge" axis="0 1 0" armature="0.08" damping="0.40"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/></body></body>
  </worldbody>
  <keyframe><key qpos="0.5 -0.3" qvel="1.5 -1.0"/></keyframe>
</mujoco>
"""

# 1-DOF slide joint with joint Coulomb friction (dof_frictionloss), gravity off, launched with a velocity
# so it SLIDES -> the friction-loss constraint saturates (force = ±frictionloss, the identifiable regime).
_FRICTIONLOSS_SLIDE = """
<mujoco>
  <option timestep="0.004" integrator="Euler" gravity="0 0 0"><flag eulerdamp="disable"/></option>
  <worldbody>
    <body><joint name="s" type="slide" axis="1 0 0" frictionloss="2.0"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
  <keyframe><key qvel="3.0"/></keyframe>
</mujoco>
"""

# Free box launched SLIDING (+x) on a flat plane (elliptic cone, condim 3) -- the CONTACT-PARAMETER
# sys-id case. The box friction (0.5) WINS the elementwise-max geom_friction combine over the floor's
# (0.2), so contact.friction is unambiguously the box geom's -> the max() VJP routes cleanly to one
# component and central FD is two-sided (a friction TIE would make FD one-sided at the max kink).
_FRICTION_SLIDE = """
<mujoco>
  <option timestep="0.004" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81"
          solver="Newton" iterations="50"><flag eulerdamp="disable"/></option>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.01" friction="0.2 0.005 0.0001" condim="3" solimp="0 0.95 0.001"/>
    <body name="box" pos="0 0 0.1"><joint type="free"/>
      <geom name="box" type="box" size="0.1 0.1 0.1" mass="1" friction="0.5 0.005 0.0001" condim="3" solimp="0 0.95 0.001"/></body>
  </worldbody>
  <keyframe><key qpos="0 0 0.1 1 0 0 0" qvel="2.0 0 0 0 0 0"/></keyframe>
</mujoco>
"""


@wp.kernel
def _sumsq_qvel_kernel(qvel: wp.array2d[float], loss: wp.array[float]):
  i = wp.tid()
  wp.atomic_add(loss, 0, qvel[0, i] * qvel[0, i])


def _sysid_taped_grad(mjm, mjd, H, field):
  """Analytic d(||qvel_H||^2)/d(m.<field>) through the taped rollout (adjoint.py). The param adjoint falls
  out of the residual-VJP and ACCUMULATES into m.<field>.grad over the rollout (a shared leaf)."""
  m = mjw.put_model(mjm)
  arr = getattr(m, field)  # put_model stores it broadcast (stride-0); rebuild contiguous for a clean grad
  setattr(m, field, wp.array(arr.numpy(), dtype=arr.dtype, requires_grad=True))  # arr.dtype: float or wp.vec3
  getattr(m, field).grad.zero_()
  datas = [mjw.put_data(mjm, mjd) for _ in range(H + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(H):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_sumsq_qvel_kernel, dim=mjm.nv, inputs=[datas[H].qvel], outputs=[loss])
  tape.backward(loss=loss)
  return np.nan_to_num(getattr(m, field).grad.numpy()[0].astype(np.float64).reshape(-1).copy())


def _sysid_mjc_fd_grad(mjm, mjd, H, field, xml=_SYSID_ARM, eps=1e-5):
  """Float64 MuJoCo-C central FD of ||qvel_H||^2 w.r.t. each component of model.<field> -- the param
  analog of mjd_transitionFD (which has no model-param mode): a clean, engine-independent oracle. Operates
  on the FLATTENED param (so vec-valued fields like body_inertia (nbody,3) are covered component-wise)."""
  shape = getattr(mjm, field).shape

  def rollout(vals):
    m2 = mujoco.MjModel.from_xml_string(xml)
    getattr(m2, field)[:] = vals.reshape(shape)
    md = mujoco.MjData(m2)
    md.qpos[:] = mjd.qpos
    md.qvel[:] = mjd.qvel
    mujoco.mj_forward(m2, md)
    for _ in range(H):
      mujoco.mj_step(m2, md)
    return float(np.dot(md.qvel, md.qvel))

  x0 = getattr(mjm, field).astype(np.float64).reshape(-1).copy()
  g = np.zeros(len(x0))
  for i in range(len(x0)):
    xp, xm = x0.copy(), x0.copy()
    xp[i] += eps
    xm[i] -= eps
    g[i] = (rollout(xp) - rollout(xm)) / (2.0 * eps)
  return g


class SysidGradientTest(parameterized.TestCase):
  """System-identification gradients: dL/d(Model param) via the SAME IFT λ (residual-VJP, §5.11). The
  param adjoints FALL OUT of the AD'd residual -- no per-param hand VJP. Stage 1: dof_armature, dof_damping
  (the AD-clean, contact-free PACE subset). Validated vs float64 MuJoCo-C param FD."""

  @parameterized.named_parameters(("armature", "dof_armature"), ("damping", "dof_damping"))
  def test_smooth_param_grad_matches_fd(self, field):
    """Excited 2-hinge arm: analytic d(||qvel_H||^2)/d(m.<field>) (taped BPTT through the local smooth
    residual) vs float64 MuJoCo-C param FD. Both per-joint armature and viscous damping are AD-clean
    (local; no RNE tree), so the residual-VJP reproduces FD to machine precision."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm = mujoco.MjModel.from_xml_string(_SYSID_ARM)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    analytic = _sysid_taped_grad(mjm, mjd, 20, field)
    fd = _sysid_mjc_fd_grad(mjm, mjd, 20, field)
    _assert_smooth_grad(self, analytic, fd, cos_min=0.999, rel_max=2e-2, tag=f"sysid:{field}")

  @parameterized.named_parameters(("body_mass", "body_mass", 1e-5), ("body_inertia", "body_inertia", 1e-7))
  def test_inertial_param_grad_matches_fd(self, field, eps):
    """Inertial sys-id (body_mass / body_inertia): analytic d(||qvel_H||^2)/d(m.<field>) vs float64 MuJoCo-C
    param FD. mass/inertia enter the smooth residual ONLY through cinert, so the gradient is produced by
    SOURCE-AD of the cinert leaf (no hand-written inertia VJP -- smooth_adjoint.inertia_param_vjp) seeded by
    adj_cinert from the rne-bias reverse. The 2-hinge arm under gravity excites both the gravitational/
    inertial torque (mass) and the rotational inertia (body_inertia)."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm = mujoco.MjModel.from_xml_string(_SYSID_ARM)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    analytic = _sysid_taped_grad(mjm, mjd, 20, field)
    fd = _sysid_mjc_fd_grad(mjm, mjd, 20, field, eps=eps)
    _assert_smooth_grad(self, analytic, fd, cos_min=0.999, rel_max=2e-2, tag=f"sysid:{field}")

  def test_param_grad_observability(self):
    """The gradient honestly reflects identifiability (§5.11). Starting at REST (qvel0=0), the viscous
    damping force is 0 over a single step, so d/d(dof_damping) ~ 0; armature stays observable (gravity
    drives qacc != 0). A nonzero damping grad here would be a bug (un-excited DOF)."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm = mujoco.MjModel.from_xml_string(_SYSID_ARM)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mjd.qvel[:] = 0.0  # at rest: damping un-excited, armature still sees gravity-driven qacc
    mujoco.mj_forward(mjm, mjd)
    g_arm = _sysid_taped_grad(mjm, mjd, 1, "dof_armature")
    g_dmp = _sysid_taped_grad(mjm, mjd, 1, "dof_damping")
    print(f"\n[sysid:observability] H=1 at rest  g_armature={g_arm}  g_damping={g_dmp}")
    self.assertLess(np.linalg.norm(g_dmp), 1e-9, f"damping grad should be ~0 at rest, got {g_dmp}")
    self.assertGreater(np.linalg.norm(g_arm), 1e-6, f"armature grad should be nonzero (gravity), got {g_arm}")

  def test_frictionloss_grad_matches_fd(self):
    """Joint Coulomb friction (dof_frictionloss) via the AD'd NON-CONTACT constraint residual. A sliding
    slide-joint whose friction-loss constraint is SATURATED (force = ±frictionloss): analytic
    d(||qvel_H||^2)/d(dof_frictionloss) vs float64 MuJoCo-C param FD. Identifiable only while slipping --
    the saturated zone is the only frictionloss-dependent one (the gradient honestly reflects it)."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm = mujoco.MjModel.from_xml_string(_FRICTIONLOSS_SLIDE)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    analytic = _sysid_taped_grad(mjm, mjd, 20, "dof_frictionloss")
    fd = _sysid_mjc_fd_grad(mjm, mjd, 20, "dof_frictionloss", xml=_FRICTIONLOSS_SLIDE, eps=1e-3)
    _assert_smooth_grad(self, analytic, fd, cos_min=0.999, rel_max=2e-2, tag="sysid:frictionloss")

  def test_contact_friction_param_grad_matches_fd(self):
    """CONTACT-PARAMETER sys-id: analytic d(||qvel_H||^2)/d(geom_friction) for a box SLIDING on a plane
    (elliptic cone, condim 3) vs float64 MuJoCo-C param FD. The geom_friction -> contact.friction copy
    happens in the UNRECORDED forward, so this gradient used to be exactly 0; adjoint.contact_residual_
    backward now exposes the cone leaf's d(phi)/d(contact.friction) and constraint_adjoint._contact_
    friction_geom_vjp chains it back through the elementwise-max/priority combine into m.geom_friction.
    grad (gated on requires_grad). FD-exact while the box slides continuously; the only non-smooth point
    is the sliding->stuck STOP (cone edge). H=4 keeps it sliding (the box also needs a step or two under
    gravity for the soft contact to develop a normal force, so a from-rest single step is un-excited)."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm = mujoco.MjModel.from_xml_string(_FRICTION_SLIDE)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    analytic = _sysid_taped_grad(mjm, mjd, 4, "geom_friction")
    fd = _sysid_mjc_fd_grad(mjm, mjd, 4, "geom_friction", xml=_FRICTION_SLIDE, eps=1e-4)
    # The gradient must land on the box geom's SLIDE-friction component (the max-winner), ~0 elsewhere.
    self.assertGreater(np.abs(analytic).max(), 1e-3, "geom_friction gradient is unobservable")
    _assert_smooth_grad(self, analytic, fd, cos_min=0.999, rel_max=2e-2, tag="sysid:geom_friction")


# Slide joint pressed against a SOFT lower limit by gravity (settle so it rests on the limit -> the limit
# constraint is active with a FIXED active set; a stiff transient bounce would instead cross the non-diff
# make/break boundary). Soft jnt_solimp dmin=0 set at load (like contact dmin=0).
_LIMIT_SLIDE = """
<mujoco>
  <option timestep="0.004" integrator="Euler" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <worldbody>
    <body><joint name="z" type="slide" axis="0 0 1" limited="true" range="0 1.0"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
  <keyframe><key qpos="0.05" qvel="-0.5"/></keyframe>
</mujoco>
"""

# HINGE limit (scalar LIMIT_JOINT, 1:1 lift): arm pressed against a soft limit by gravity (range upper kept
# small so the arm rests near-horizontal where the gravity torque firmly holds it on the limit).
_HINGE_LIMIT = """
<mujoco>
  <option timestep="0.004" integrator="Euler" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <worldbody>
    <body pos="0 0 1"><joint name="h" type="hinge" axis="0 1 0" limited="true" range="-2.0 0.3"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/></body>
  </worldbody>
  <keyframe><key qpos="0" qvel="0"/></keyframe>
</mujoco>
"""

# BALL limit (LIMIT_JOINT on a quaternion joint -> the -axis angular Jacobian + the quaternion ∂qpos LIFT):
# the capsule swings down under gravity until the ball's rotation angle hits the range limit.
_BALL_LIMIT = """
<mujoco>
  <option timestep="0.004" integrator="Euler" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <worldbody>
    <body pos="0 0 1"><joint name="b" type="ball" range="0 0.6"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/></body>
  </worldbody>
  <keyframe><key qpos="1 0 0 0" qvel="0 0 0"/></keyframe>
</mujoco>
"""

# Two hinges coupled by a JOINT equality (j1 follows j0) -> bilateral, always-active constraint coupling.
_EQUALITY_COUPLE = """
<mujoco>
  <option timestep="0.004" integrator="Euler" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <worldbody>
    <body pos="0 0 1"><joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/></body>
    <body pos="0 0.3 1"><joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/></body>
  </worldbody>
  <equality><joint joint1="j0" joint2="j1"/></equality>
  <keyframe><key qpos="0.3 0.3" qvel="1.0 1.0"/></keyframe>
</mujoco>
"""


def _hang_spectators(n, x0=0.6):
  """``n`` INERT hinge pendulums, each a ``-z`` capsule so at qpos=0 the COM hangs directly below the hinge
  axis (gravity torque ~0 -> equilibrium). With qvel0=0 and light damping they stay at rest, padding nv past
  _MAX_NV WITHOUT polluting ``||qvel_T||^2`` or the state grad (their grad slice is exactly 0 -- which also
  catches a sparse scatter that leaks a cotangent onto a wrong dof). Each adds ONE dof."""
  s = ""
  for i in range(n):
    s += (
      f'<body pos="{x0 + 0.3 * i} 0 1"><joint name="s{i}" type="hinge" axis="0 1 0" damping="1.0"/>'
      f'<geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.03" mass="0.5"/></body>'
    )
  return s


def _padded_equality(n_extra):
  """``n_extra`` inert spectators FIRST (dofs 0..n_extra-1) then the two equality-coupled hinges LAST (dofs
  n_extra, n_extra+1 -> HIGH-index columns beyond _MAX_NV) -> nv=n_extra+2. Routes through the SPARSE
  ``_residual_constraint_sparse`` (nv>_MAX_NV); the high-index active row proves the gather/scatter iterator
  reaches dofs past _MAX_NV for the CONSTRAINT row (not just the smooth path)."""
  xml = f"""
<mujoco>
  <option timestep="0.004" integrator="Euler" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <worldbody>
    {_hang_spectators(n_extra)}
    <body pos="0 0 1"><joint name="j0" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/></body>
    <body pos="0 0.3 1"><joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/></body>
  </worldbody>
  <equality><joint joint1="j0" joint2="j1"/></equality>
</mujoco>
"""
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  qpos0 = np.zeros(mjm.nq)
  qpos0[n_extra] = 0.3
  qpos0[n_extra + 1] = 0.3
  mjd.qpos[:] = qpos0
  mujoco.mj_forward(mjm, mjd)
  qvel0 = np.zeros(mjm.nv)
  qvel0[n_extra] = 1.0
  qvel0[n_extra + 1] = 1.0
  return mjm, mjd, qvel0


def _padded_ball_limit(n_extra):
  """``n_extra`` inert spectators FIRST then a gravity-settled BALL pressed against its limit LAST (the 3 ball
  dofs are the HIGH-index columns >_MAX_NV) -> nv=n_extra+3. Same SPARSE-path/high-index rationale as
  _padded_equality, but for the LIMIT_JOINT ball row (the -axis angular J + the _dof_to_qpos quaternion lift)."""
  xml = f"""
<mujoco>
  <option timestep="0.004" integrator="Euler" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <worldbody>
    {_hang_spectators(n_extra)}
    <body pos="0 0 1"><joint name="b" type="ball" range="0 0.6"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/></body>
  </worldbody>
</mujoco>
"""
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjm.jnt_solimp[:, 0] = 0.0  # soft (unsaturated) limit -> smooth, like the nv<=16 ball test
  mjd = mujoco.MjData(mjm)
  mjd.qpos[:] = 0.0
  mjd.qpos[n_extra] = 1.0  # ball qw (spectators are 1 qpos each, first)
  mujoco.mj_forward(mjm, mjd)
  for _ in range(90):  # settle: ball swings onto the limit (active over the short horizon); spectators at rest
    mujoco.mj_step(mjm, mjd)
  mujoco.mj_forward(mjm, mjd)
  qvel0 = np.zeros(mjm.nv)
  qvel0[n_extra + 1] = 0.3  # kick the ball into the limit -> stays pressed (no active-set change)
  return mjm, mjd, qvel0


def _constraint_qvel_grad(mjm, mjd, T, qvel0):
  """Analytic d(||qvel_T||^2)/d(qvel0) through the taped rollout: exercises the constraint residual's ∂qpos
  (stiffness) + ∂qvel (velocity-coupling) state-grads for the active equality/limit rows."""
  m = mjw.put_model(mjm)
  datas = [mjw.put_data(mjm, mjd) for _ in range(T + 1)]
  for dd in datas:
    dd.qpos.requires_grad = True
    dd.qvel.requires_grad = True
  datas[0].qvel = wp.array(qvel0.reshape(1, -1).astype(np.float32), dtype=float, requires_grad=True)
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(T):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_sumsq_qvel_kernel, dim=mjm.nv, inputs=[datas[T].qvel], outputs=[loss])
  tape.backward(loss=loss)
  return np.nan_to_num(datas[0].qvel.grad.numpy()[0].astype(np.float64))


def _constraint_fd_qvel(mjm, mjd, T, qvel0, eps=1e-6):
  """Float64 MuJoCo-C central FD of ||qvel_T||^2 wrt qvel0 (clean engine-independent state-grad oracle)."""

  def L(qv):
    d = mujoco.MjData(mjm)
    d.qpos[:] = mjd.qpos
    d.qvel[:] = qv
    mujoco.mj_forward(mjm, d)
    for _ in range(T):
      mujoco.mj_step(mjm, d)
    return float(np.dot(d.qvel, d.qvel))

  g = np.zeros(mjm.nv)
  for i in range(mjm.nv):
    vp, vm = qvel0.copy(), qvel0.copy()
    vp[i] += eps
    vm[i] -= eps
    g[i] = (L(vp) - L(vm)) / (2 * eps)
  return g


class ConstraintGradientTest(parameterized.TestCase):
  """NON-CONTACT constraint residual (_residual_constraint) STATE grad: d(||qvel_T||^2)/d(qvel0) through
  equality + joint-limit rows vs float64 MuJoCo-C FD. Validates the constraint STIFFNESS ∂qpos (k·imp·J)
  + velocity coupling ∂qvel (b·J) -- the Task-2 non-contact-constraint grads -- at a FIXED active set."""

  @parameterized.named_parameters(
    ("slide", _LIMIT_SLIDE, 60, [-0.5]),
    ("hinge", _HINGE_LIMIT, 80, [0.4]),
    ("ball", _BALL_LIMIT, 90, [0.0, 0.3, 0.0]),
  )
  def test_joint_limit_state_grad(self, xml, settle, kick):
    """LIMIT_JOINT state-grad -- slide/hinge (scalar, 1:1 dof->qpos) and BALL (the -axis angular Jacobian +
    the quaternion ∂qpos LIFT, _dof_to_qpos's 2·q⊗[0,g]). Analytic d(||qvel_T||^2)/d(qvel0) vs float64
    MuJoCo-C FD, FIXED active set (settled pressed against a soft limit). Without the constraint residual's
    ∂qpos stiffness term the grad is ~0/wrong -- a limit's restoring force IS the penetration stiffness."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm = mujoco.MjModel.from_xml_string(xml)
    mjm.jnt_solimp[:, 0] = 0.0  # soft (unsaturated) limit -> smooth, like contact dmin=0
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    for _ in range(settle):  # settle: rest on the limit (active throughout the short test horizon)
      mujoco.mj_step(mjm, mjd)
    mujoco.mj_forward(mjm, mjd)
    qvel0 = np.array(kick)  # kick into the limit -> stays pressed (no active-set change)
    analytic = _constraint_qvel_grad(mjm, mjd, 8, qvel0)
    fd = _constraint_fd_qvel(mjm, mjd, 8, qvel0)
    _assert_smooth_grad(self, analytic, fd, cos_min=0.999, rel_max=2e-2, tag="constraint:limit")

  def test_equality_state_grad(self):
    """Two hinges coupled by a joint equality (bilateral, always active): analytic
    d(||qvel_T||^2)/d(qvel0) vs float64 MuJoCo-C FD -- locks the equality constraint's ∂qpos/∂qvel."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm = mujoco.MjModel.from_xml_string(_EQUALITY_COUPLE)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    qvel0 = mjd.qvel.astype(np.float64).copy()
    analytic = _constraint_qvel_grad(mjm, mjd, 20, qvel0)
    fd = _constraint_fd_qvel(mjm, mjd, 20, qvel0)
    _assert_smooth_grad(self, analytic, fd, cos_min=0.999, rel_max=2e-2, tag="constraint:equality")

  @parameterized.named_parameters(
    ("equality_nv18", "equality", 16, 12),  # nv=18 > _MAX_NV: SPARSE orchestration, DENSE efc.J iterator
    ("equality_nv34_csr", "equality", 32, 12),  # nv=34 > 32: SPARSE orchestration, CSR efc.J iterator
    ("ball_nv18", "ball", 15, 8),  # nv=18: ball's -axis J + quaternion lift through the sparse scatter
    ("ball_nv34_csr", "ball", 31, 8),  # nv=34: same, CSR efc.J iterator
  )
  def test_noncontact_constraint_state_grad_sparse_nv(self, kind, n_extra, T):
    """nv>_MAX_NV JOINT-equality / BALL-limit state grad through the SPARSE ``_residual_constraint_sparse``
    (the HAZARD guard for MJPLAN_CSR's relaxed gate): the newly-enabled classes must be FD-exact at nv>16 in
    BOTH efc.J layouts (dense iterator at nv=18, CSR at nv=34), and the active row lives at a HIGH-index column
    (>_MAX_NV) so the gather/scatter iterator is proven to reach past the legacy dense unroll bound. Inert
    spectators pad nv (their grad slice must stay 0 -> a leaked scatter cotangent fails). Analytic
    d(||qvel_T||^2)/d(qvel0) vs float64 MuJoCo-C FD."""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm, mjd, qvel0 = (_padded_equality if kind == "equality" else _padded_ball_limit)(n_extra)
    self.assertGreater(mjm.nv, _adjoint._MAX_NV, "scene must have nv>_MAX_NV to route to the sparse path")
    if "csr" in self.id():  # the nv=34 variants must actually take the CSR (is_sparse) efc.J iterator
      self.assertTrue(mjw.put_model(mjm).is_sparse, "nv=34 scene must store efc.J sparse (CSR)")
    analytic = _constraint_qvel_grad(mjm, mjd, T, qvel0)
    fd = _constraint_fd_qvel(mjm, mjd, T, qvel0)
    _assert_smooth_grad(self, analytic, fd, cos_min=0.999, rel_max=2e-2, tag=f"constraint:{kind}:nv{mjm.nv}")

  def test_jnt_solref_sysid(self):
    """Joint-limit solref sys-id: jnt_solref (the limit's solver-reference timeconst/dampratio) falls out
    of the SAME constraint residual by exposing its input-adjoint -- it is fed through _contact_kbimp, so
    its adjoint is auto-computed (no per-param kernel). Analytic d(||qvel_T||^2)/d(jnt_solref) vs float64
    MuJoCo-C FD on the settled soft-limit scene. (eq_solref on a near-rigid equality is correctly ~0 --
    not identifiable -- so jnt_solref is the validatable case.)"""
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1/3)")
    from mujoco_warp._src import adjoint  # noqa: F401

    def _settled():
      mjm = mujoco.MjModel.from_xml_string(_LIMIT_SLIDE)
      mjm.jnt_solimp[:, 0] = 0.0
      mjd = mujoco.MjData(mjm)
      mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
      for _ in range(60):
        mujoco.mj_step(mjm, mjd)
      mujoco.mj_forward(mjm, mjd)
      return mjm, mjd

    T, qvel0 = 8, np.array([-0.5])
    mjm, mjd = _settled()
    m = mjw.put_model(mjm)
    m.jnt_solref = wp.array(m.jnt_solref.numpy(), dtype=wp.vec2, requires_grad=True)  # vec2; contiguous grad
    m.jnt_solref.grad.zero_()
    datas = [mjw.put_data(mjm, mjd) for _ in range(T + 1)]
    for dd in datas:
      dd.qpos.requires_grad = True
      dd.qvel.requires_grad = True
    datas[0].qvel = wp.array(qvel0.reshape(1, -1).astype(np.float32), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      for t in range(T):
        mjw.step(m, datas[t], datas[t + 1])
      wp.launch(_sumsq_qvel_kernel, dim=mjm.nv, inputs=[datas[T].qvel], outputs=[loss])
    tape.backward(loss=loss)
    analytic = m.jnt_solref.grad.numpy()[0].astype(np.float64)  # the limited joint's (timeconst, dampratio)

    base = mjm.jnt_solref.reshape(-1).astype(np.float64).copy()

    def L(vals):
      m2 = mujoco.MjModel.from_xml_string(_LIMIT_SLIDE)
      m2.jnt_solimp[:, 0] = 0.0
      m2.jnt_solref.reshape(-1)[:] = vals
      d = mujoco.MjData(m2)
      d.qpos[:] = mjd.qpos
      d.qvel[:] = qvel0
      mujoco.mj_forward(m2, d)
      for _ in range(T):
        mujoco.mj_step(m2, d)
      return float(np.dot(d.qvel, d.qvel))

    fd = np.zeros(base.size)
    for i in range(base.size):
      vp, vm = base.copy(), base.copy()
      vp[i] += 1e-6
      vm[i] -= 1e-6
      fd[i] = (L(vp) - L(vm)) / 2e-6
    _assert_smooth_grad(self, analytic, fd, cos_min=0.999, rel_max=2e-2, tag="sysid:jnt_solref")


# ============================================================================================
# AD-RNE (MJPLAN_ADRNE §15.3): analytic VJP of smooth.rne (RNE-proper reverse: adjoint.rne_backward)
# + the com_vel reverse (adjoint.comvel_backward), gated vs float64 mj_rne. Swept over randomly-bent
# configs (mj_integratePos -> quaternion-safe) x seeds so no single degenerate config (e.g. a planar
# hopper at qpos=0 whose dL/dqvel is genuinely 0) can mask a coverage gap. NOT wired into
# step_backward yet (FD-of-rne stays the live ∂qpos path; the kinematic ∂qpos reverse is steps 5-7).
# ============================================================================================
from mujoco_warp._src import adjoint as _adjoint  # noqa: E402
from mujoco_warp._src import smooth_adjoint as _smooth_adjoint  # noqa: E402  (rne_backward / comvel_backward / rne_qpos_vjp moved here)
from mujoco_warp._src import smooth as _smooth  # noqa: E402
from mujoco_warp._src import solver as _solver  # noqa: E402  (init_context -> ctx.Jaref / lam for the constraint VJP)
from mujoco_warp._src import constraint_adjoint as _constraint_adjoint  # noqa: E402  (sparse non-contact residual kernels)
from mujoco_warp._src.types import vec10f as _vec10f  # noqa: E402

_RNE_FREE = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody><body pos="0 0 1"><freejoint/>
    <geom type="box" size="0.10 0.20 0.15" mass="1.5"/></body></worldbody></mujoco>
"""
_RNE_HOPPER = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0.71"><joint type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.08 0.06 0.1" mass="3.0"/>
      <body pos="0 0 -0.1"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.035" mass="0.6"/>
        <body pos="0 0 -0.3"><joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03" mass="0.4"/></body></body></body>
  </worldbody></mujoco>
"""
_RNE_WORM = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 0.5" euler="0 0 5"><freejoint/>
      <geom type="capsule" fromto="-0.22 0 0 -0.02 0 0" size="0.1" mass="1"/>
      <body pos="0 0 0"><joint type="ball"/>
        <geom type="capsule" fromto="0.02 0 0 0.22 0 0" size="0.1" mass="1"/></body></body>
  </worldbody></mujoco>
"""
# BRANCHING: fixed-base torso with TWO 2-hinge legs -> torso is a shared ancestor of two child
# subtrees (the only scene that exposes duplicate shared-ancestor reverse writers in the cfrc/cacc
# tree reductions). Mixed hinge axes.
_RNE_BRANCHING = """
<mujoco><option timestep="0.004" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="torso" pos="0 0 1"><joint type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.12 0.10 0.05" mass="2.0"/>
      <body name="legL" pos="0.1 0 -0.05"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.5"/>
        <body pos="0 0 -0.3"><joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.025" mass="0.3"/></body></body>
      <body name="legR" pos="-0.1 0 -0.05"><joint type="hinge" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.5"/>
        <body pos="0 0 -0.3"><joint type="hinge" axis="1 0 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.025" mass="0.3"/></body></body></body>
  </worldbody></mujoco>
"""
_RNE_SCENES = {"free": _RNE_FREE, "hopper": _RNE_HOPPER, "worm": _RNE_WORM, "branching": _RNE_BRANCHING}


def _rne_metrics(ana, fd):
  ana, fd = ana.ravel().astype(np.float64), fd.ravel().astype(np.float64)
  na, nf = np.linalg.norm(ana), np.linalg.norm(fd)
  cos = float(ana @ fd / (na * nf)) if na > 1e-12 and nf > 1e-12 else float("nan")
  rel = float(np.linalg.norm(ana - fd) / (nf + 1e-12))
  return cos, rel, nf


def _rne_fwd_mjw(m, d, qpos, qvel, qacc):
  """Set state and run the smooth forward up through rne (flg_acc) -> qfrc_bias + intermediates."""
  wp.copy(d.qpos, wp.array(qpos.reshape(1, -1).astype(np.float32), dtype=wp.float32))
  wp.copy(d.qvel, wp.array(qvel.reshape(1, -1).astype(np.float32), dtype=wp.float32))
  wp.copy(d.qacc, wp.array(qacc.reshape(1, -1).astype(np.float32), dtype=wp.float32))
  _smooth.kinematics(m, d)
  _smooth.com_pos(m, d)
  _smooth.com_vel(m, d)
  _smooth.rne(m, d, flg_acc=True)


def _rne_fd_mj(mjm, qpos, qvel, qacc, lam, wrt, eps=1e-6):
  """float64 MuJoCo-C FD of L=λᵀ(M·qacc+C) wrt qvel or qacc (qpos fixed -> plain R^nv FD)."""
  nv = mjm.nv
  out = np.zeros(nv)
  for k in range(nv):
    vals = []
    for s in (+1.0, -1.0):
      d2 = mujoco.MjData(mjm)
      d2.qpos[:] = qpos
      qv, qa = qvel.copy(), qacc.copy()
      (qv if wrt == "qvel" else qa)[k] += s * eps
      d2.qvel[:] = qv; d2.qacc[:] = qa
      mujoco.mj_kinematics(mjm, d2); mujoco.mj_comPos(mjm, d2); mujoco.mj_comVel(mjm, d2)
      r = np.zeros(nv); mujoco.mj_rne(mjm, d2, 1, r); vals.append(lam @ r)
    out[k] = (vals[0] - vals[1]) / (2.0 * eps)
  return out


def _rne_fd_intermediate(m, d, name, dtype, lam, eps=1e-3):
  """FD of L=λᵀqfrc_bias wrt each component of an rne INPUT intermediate (re-run smooth.rne only,
  cvel/cdof_dot FROZEN as inputs) -> isolates the rne-PROPER reverse."""
  arr = getattr(d, name)
  base = arr.numpy().copy()
  flat = base.reshape(base.shape[1], -1)
  n, comp = flat.shape
  out = np.zeros((n, comp))
  for i in range(n):
    for c in range(comp):
      pp = base.copy(); pp.reshape(n, comp)[i, c] += eps
      setattr(d, name, wp.array(pp.astype(np.float32), dtype=dtype)); _smooth.rne(m, d, flg_acc=True)
      Lp = float(lam @ d.qfrc_bias.numpy()[0])
      pm = base.copy(); pm.reshape(n, comp)[i, c] -= eps
      setattr(d, name, wp.array(pm.astype(np.float32), dtype=dtype)); _smooth.rne(m, d, flg_acc=True)
      Lm = float(lam @ d.qfrc_bias.numpy()[0])
      out[i, c] = (Lp - Lm) / (2.0 * eps)
  setattr(d, name, arr)  # restore
  return out


def _rne_fd_comvel(m, d, name, dtype, A, G, eps=1e-3):
  """§10.1A.9 isolated com_vel gate: FD of L_cv = Σ A_b·cvel_b + Σ G_i·cdof_dot_i wrt qvel or cdof
  (re-run smooth.com_vel only) -- directly gates comvel_backward incl. the cdof channel."""
  arr = getattr(d, name)
  base = arr.numpy().copy()
  flat = base.reshape(base.shape[1], -1)
  n, comp = flat.shape
  out = np.zeros((n, comp))

  def Lcv():
    _smooth.com_vel(m, d)
    return float((A * d.cvel.numpy()[0]).sum() + (G * d.cdof_dot.numpy()[0]).sum())

  for i in range(n):
    for c in range(comp):
      pp = base.copy(); pp.reshape(n, comp)[i, c] += eps
      setattr(d, name, wp.array(pp.astype(np.float32), dtype=dtype)); Lp = Lcv()
      pm = base.copy(); pm.reshape(n, comp)[i, c] -= eps
      setattr(d, name, wp.array(pm.astype(np.float32), dtype=dtype)); Lm = Lcv()
      out[i, c] = (Lp - Lm) / (2.0 * eps)
  setattr(d, name, arr)
  return out


_RNE_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_RNE_BALL = int(mujoco.mjtJoint.mjJNT_BALL)


def _rne_qmul(a, b):  # Hamilton product, w-first
  return np.array([
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ])


def _rne_unlift(q, dq):  # invert _dof_to_qpos's dq = 2 q⊗[0,g]: g = 1/2 vec(conj(q)⊗dq)
  return 0.5 * _rne_qmul(np.array([q[0], -q[1], -q[2], -q[3]]), dq)[1:]


def _rne_fd_qpos_mj(mjm, qpos, qvel, qacc, lam, eps=1e-6):
  """float64 mj_rne FD of L=λᵀqfrc_bias wrt qpos in TANGENT space (mj_integratePos, quaternion-safe),
  recomputing the whole chain mj_kinematics -> mj_comPos -> mj_comVel -> mj_rne."""
  nv = mjm.nv
  out = np.zeros(nv)
  for k in range(nv):
    vals = []
    for s in (+1.0, -1.0):
      qp = qpos.copy(); e = np.zeros(nv); e[k] = 1.0
      mujoco.mj_integratePos(mjm, qp, e, s * eps)
      dd = mujoco.MjData(mjm); dd.qpos[:] = qp; dd.qvel[:] = qvel; dd.qacc[:] = qacc
      mujoco.mj_kinematics(mjm, dd); mujoco.mj_comPos(mjm, dd); mujoco.mj_comVel(mjm, dd)
      r = np.zeros(nv); mujoco.mj_rne(mjm, dd, 1, r); vals.append(lam @ r)
    out[k] = (vals[0] - vals[1]) / (2.0 * eps)
  return out


class RneBackwardTest(parameterized.TestCase):
  """Analytic RNE-proper reverse (adjoint.rne_backward) + com_vel reverse (adjoint.comvel_backward)
  vs float64 mj_rne, MJPLAN_ADRNE §15.3. Swept over randomly-bent configs so a degenerate config
  cannot mask a coverage gap (a planar hopper at qpos=0 has dL/dqvel==0; bending it exposes the
  real derivative -- |fd| up to ~0.1 with cos=1.0)."""

  @parameterized.parameters(*_RNE_SCENES.keys())
  def test_rne_backward(self, name):
    mjm = mujoco.MjModel.from_xml_string(_RNE_SCENES[name])
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mjd = mujoco.MjData(mjm); mujoco.mj_forward(mjm, mjd)
    nv = mjm.nv
    m = mjw.put_model(mjm)
    rng = np.random.default_rng(0)
    qpos0 = mjd.qpos.copy()
    ndraw = 4
    cos_qvel, cos_qacc = [], []

    for t in range(ndraw):
      qpos = qpos0.copy()
      mujoco.mj_integratePos(mjm, qpos, rng.standard_normal(nv) * 0.6, 1.0)  # quaternion-safe bend
      qvel = rng.standard_normal(nv) * 0.8
      qacc = rng.standard_normal(nv) * 0.8
      lam = rng.standard_normal(nv)

      d = mjw.put_data(mjm, mjd)
      _rne_fwd_mjw(m, d, qpos, qvel, qacc)

      # forward equivalence: the reconstructed/recomputed qfrc_bias must match mj_rne (linearization pt)
      mjd.qpos[:] = qpos; mjd.qvel[:] = qvel; mjd.qacc[:] = qacc
      mujoco.mj_kinematics(mjm, mjd); mujoco.mj_comPos(mjm, mjd); mujoco.mj_comVel(mjm, mjd)
      r_ref = np.zeros(nv); mujoco.mj_rne(mjm, mjd, 1, r_ref)
      fwd_rel = np.linalg.norm(d.qfrc_bias.numpy()[0] - r_ref) / (np.linalg.norm(r_ref) + 1e-9)
      self.assertLess(fwd_rel, 1e-4, f"{name}[{t}]: qfrc_bias != mj_rne (rel={fwd_rel:.2e})")

      # analytic reverse: rne-proper + com_vel (Coriolis path) -> TOTAL qvel adjoint
      lam_wp = wp.array(lam.reshape(1, -1).astype(np.float32), dtype=wp.float32)
      adj = _smooth_adjoint.rne_backward(m, d, lam_wp, flg_acc=True)
      cv = _smooth_adjoint.comvel_backward(m, d, adj["cvel"], adj["cdof_dot"])
      total_qvel = adj["qvel"].numpy()[0] + cv["qvel"].numpy()[0]

      # Gate 2a: TOTAL dL/dqvel + dL/dqacc vs float64 mj_rne (the §14.3 physics gate)
      for wrt, ana, store in (("qvel", total_qvel, cos_qvel), ("qacc", adj["qacc"].numpy()[0], cos_qacc)):
        cos, rel, nf = _rne_metrics(ana, _rne_fd_mj(mjm, qpos, qvel, qacc, lam, wrt))
        self.assertGreater(nf, 1e-4, f"{name}[{t}]: dL/d{wrt} ~0 (degenerate config -- not exercising it)")
        self.assertGreater(cos, 0.999, f"{name}[{t}]: dL/d{wrt} direction off, cos={cos:.5f}")
        self.assertLess(rel, 1e-2, f"{name}[{t}]: dL/d{wrt} magnitude off, rel={rel:.2e}")
        store.append(cos)

      # Gate 2b: intermediate seeds vs FD-through smooth.rne (first bent config only -- expensive)
      if t == 0:
        for ni, dt in (("cdof", wp.spatial_vector), ("cdof_dot", wp.spatial_vector),
                       ("cinert", _vec10f), ("cvel", wp.spatial_vector)):
          ana_i = adj[ni].numpy()[0].copy()
          cos, rel, _nf = _rne_metrics(ana_i, _rne_fd_intermediate(m, d, ni, dt, lam))
          self.assertGreater(cos, 0.999, f"{name}: adj_{ni} direction off, cos={cos:.5f}")
          self.assertLess(rel, 1e-2, f"{name}: adj_{ni} magnitude off, rel={rel:.2e}")
        # §10.1A.9 isolated com_vel VJP gate (random seeds; gates BOTH the qvel and the cdof channels,
        # the latter consumed by the qpos kinematic reverse). Bound-free: no _CV_MAX_* assumptions.
        A = rng.standard_normal((mjm.nbody, 6)); G = rng.standard_normal((nv, 6))
        cvb = _smooth_adjoint.comvel_backward(
          m, d, wp.array(A[None].astype(np.float32), dtype=wp.spatial_vector),
          wp.array(G[None].astype(np.float32), dtype=wp.spatial_vector))
        for ch, dt in (("qvel", wp.float32), ("cdof", wp.spatial_vector)):
          cos, rel, _nf = _rne_metrics(cvb[ch].numpy()[0], _rne_fd_comvel(m, d, ch, dt, A, G))
          self.assertGreater(cos, 0.999, f"{name}: com_vel dLcv/d{ch} direction off, cos={cos:.5f}")
          self.assertLess(rel, 1e-2, f"{name}: com_vel dLcv/d{ch} magnitude off, rel={rel:.2e}")

        # Full RNE-bias ∂qpos (§14.3 qpos column): rne_qpos_vjp (rne_backward + comvel_backward + cinert
        # + the done cdof/subtree VJPs) vs float64 mj_rne FD, un-lifted to tangent. A single free body
        # has qfrc_bias exactly q-invariant -> true dL/dq=0 (degenerate): gate |ana| absolute instead.
        rq = _smooth_adjoint.rne_qpos_vjp(m, d, lam_wp, flg_acc=True).numpy()[0]
        ana_q = np.zeros(nv)
        for j in range(mjm.njnt):
          jt, qa, da = int(mjm.jnt_type[j]), mjm.jnt_qposadr[j], mjm.jnt_dofadr[j]
          if jt == _RNE_FREE:
            ana_q[da:da + 3] = rq[qa:qa + 3]
            ana_q[da + 3:da + 6] = _rne_unlift(qpos[qa + 3:qa + 7], rq[qa + 3:qa + 7])
          elif jt == _RNE_BALL:
            ana_q[da:da + 3] = _rne_unlift(qpos[qa:qa + 4], rq[qa:qa + 4])
          else:
            ana_q[da] = rq[qa]
        fd_q = _rne_fd_qpos_mj(mjm, qpos, qvel, qacc, lam)
        nf_q = float(np.linalg.norm(fd_q))
        if nf_q < 1e-6:  # degenerate: true dL/dq ~ 0 (single free body) -> roundoff floor
          self.assertLess(float(np.linalg.norm(ana_q)), 1e-4, f"{name}: dL/dq should be ~0, got {np.linalg.norm(ana_q):.2e}")
        else:
          cos, rel, _nf = _rne_metrics(ana_q, fd_q)
          self.assertGreater(cos, 0.999, f"{name}: dL/dq direction off, cos={cos:.5f}")
          self.assertLess(rel, 1e-2, f"{name}: dL/dq magnitude off, rel={rel:.2e}")

    print(f"[{name}] nv={nv} ({ndraw} bent configs)  qvel cos.min={min(cos_qvel):+.6f}"
          f"  qacc cos.min={min(cos_qacc):+.6f}")


@wp.kernel
def _rne_sumsq_qvel(qvel: wp.array2d[float], loss: wp.array[float]):
  i = wp.tid()
  wp.atomic_add(loss, 0, qvel[0, i] * qvel[0, i])


def _rne_bptt_analytic(mjm, mjd, T, qpos0):
  """d(Σ qvel_T²)/dqpos0 via the analytic multi-step BPTT (tape over T mjw.step + step_backward)."""
  m = mjw.put_model(mjm)
  datas = [mjw.put_data(mjm, mjd) for _ in range(T + 1)]
  for dd in datas:
    dd.qpos.requires_grad = True
    dd.qvel.requires_grad = True
  datas[0].qpos = wp.array(qpos0.reshape(1, -1).astype(np.float32), dtype=float, requires_grad=True)
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(T):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_rne_sumsq_qvel, dim=mjm.nv, inputs=[datas[T].qvel, loss])
  tape.backward(loss=loss)
  return np.nan_to_num(datas[0].qpos.grad.numpy()[0].astype(np.float64))


def _rne_bptt_fd(mjm, mjd, T, qpos0, eps=1e-4):
  """float64-tangent FD of the real mjw.step rollout (mj_integratePos perturbation of qpos0)."""
  def L(qp):
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)
    d.qpos = wp.array(qp.reshape(1, -1).astype(np.float32), dtype=float)
    for _ in range(T):
      mjw.step(m, d)
    qv = d.qvel.numpy()[0]
    return float(qv @ qv)

  g = np.zeros(mjm.nv)
  for i in range(mjm.nv):
    e = np.zeros(mjm.nv); e[i] = 1.0
    qp, qm = qpos0.copy(), qpos0.copy()
    mujoco.mj_integratePos(mjm, qp, e, eps); mujoco.mj_integratePos(mjm, qm, e, -eps)
    g[i] = (L(qp) - L(qm)) / (2.0 * eps)
  return g


class RneQposBpttTest(parameterized.TestCase):
  """Multi-step FD-of-mjw.step BPTT gate for the ANALYTIC RNE-bias ∂qpos (the definitive accumulation
  test, MJPLAN_ADRNE §14.6). Pure-RNE pendulum (gravity-driven 3-hinge chain, NO contact / passive /
  actuator) so the smooth ∂qpos is fully analytic (rne_qpos_vjp, via _USE_ANALYTIC_RNE_QPOS). Validates
  d(Σ qvel_T²)/dqpos0 over a horizon vs a float64-tangent FD of the real mjw.step rollout -- a per-step
  bias would show as cos degrading with the horizon T."""

  def test_rne_qpos_bptt(self):
    mjm = mujoco.MjModel.from_xml_string(_RNE_HOPPER)  # 3-hinge chain, NO ground plane -> pure-RNE pendulum
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    mjd = mujoco.MjData(mjm); mujoco.mj_forward(mjm, mjd)
    nv = mjm.nv
    rng = np.random.default_rng(0)
    qpos0 = mjd.qpos.copy() + rng.standard_normal(nv) * 0.3  # bend (hinges: qpos == tangent)

    prev = _adjoint._USE_ANALYTIC_RNE_QPOS
    try:
      horizons = (8, 16, 32, 64)
      fd_step = {T: _rne_bptt_fd(mjm, mjd, T, qpos0) for T in horizons}  # ground truth (flag-independent)
      for T in horizons:
        _adjoint._USE_ANALYTIC_RNE_QPOS = True
        ana = _rne_bptt_analytic(mjm, mjd, T, qpos0)  # analytic RNE-bias ∂qpos
        _adjoint._USE_ANALYTIC_RNE_QPOS = False
        ref = _rne_bptt_analytic(mjm, mjd, T, qpos0)  # FD-of-rne ∂qpos (same machinery -> the reference)
        fd = fd_step[T]

        # PRIMARY gate: AD-RNE vs FD-of-rne -- identical step_backward except the ∂qpos method, so a
        # per-step AD-RNE bias would show as drift here (no FD-of-mjw.step truncation noise to hide it).
        nar, nrr = np.linalg.norm(ana), np.linalg.norm(ref)
        cos_r = float(ana @ ref / (nar * nrr)) if nar > 1e-12 and nrr > 1e-12 else float("nan")
        rel_r = float(np.linalg.norm(ana - ref) / (nrr + 1e-12))
        # sanity vs the real rollout FD (looser: f32 multi-step central-diff truncation over a nonlinear
        # pendulum). Both analytic paths share whatever residual this has.
        nf = np.linalg.norm(fd)
        cos_f = float(ana @ fd / (nar * nf)) if nar > 1e-12 and nf > 1e-12 else float("nan")
        rel_f_ad = float(np.linalg.norm(ana - fd) / (nf + 1e-12))
        rel_f_ref = float(np.linalg.norm(ref - fd) / (nf + 1e-12))
        print(f"[bptt T={T:2d}] AD-RNE vs FD-of-rne: cos={cos_r:+.6f} rel={rel_r:.2e}  | "
              f"vs FD-of-step: cos={cos_f:+.4f} rel(ad)={rel_f_ad:.3f} rel(fdrne)={rel_f_ref:.3f}  |ana|={nar:.2e}")
        self.assertGreater(nrr, 1e-6, f"T={T}: gradient ~0 (scene not exercising qpos)")
        self.assertGreater(cos_r, 0.9999, f"T={T}: AD-RNE != FD-of-rne reference, cos={cos_r:.6f} (per-step bias)")
        self.assertLess(rel_r, 1e-2, f"T={T}: AD-RNE != FD-of-rne reference, rel={rel_r:.2e} (per-step bias)")
        self.assertGreater(cos_f, 0.99, f"T={T}: BPTT direction off vs real rollout, cos={cos_f:.4f}")
    finally:
      _adjoint._USE_ANALYTIC_RNE_QPOS = prev


# ----------------------------------------------------------------------------
# nv>16 articulated-contact gate (MJPLAN_ARTICULATION S4). The SPARSE contract-first contact VJP
# (adjoint._USE_SPARSE_CONTACT, the default) must be correct beyond the dense _MAX_NV=16 unroll bound that
# capped the legacy `_residual_contact` kernel (G1 has nv~35). Scene: a free base (6 dof) + a serial hinge
# chain (nv = 6 + n_hinge) ending in a sphere foot vs a plane. Only the foot collides (chain/base geoms
# contype=conaffinity=0) -> ONE clean foot-floor contact whose Jacobian spans EVERY chain dof, including
# dofs > 16. The hinges carry mild damping+stiffness (passive forces -- no constraint rows) to tame the
# chain. NO joint limits / equality / dof-friction, so the (still _MAX_NV-capped) non-contact constraint
# residual stays a no-op and the contact residual is isolated. FD oracle = mjw float32 central difference
# (same forward linearization point as the analytic grad), matching BounceDiffsimTest / ContactCondimTest.
# ----------------------------------------------------------------------------
def _nvchain_xml(n_hinge, cone="elliptic", condim=3, damping=1.0, stiffness=1.0, limit_first=False, tendon=False):
  L = 1.5 / n_hinge  # fixed ~1.5m total chain length regardless of n_hinge -> comparable FD conditioning
  foot_r = 0.04
  base_z = n_hinge * L + foot_r - 0.006  # foot bottom ~6mm below the plane -> active contact at qpos0
  body = ""
  for i in range(n_hinge):
    axis = "1 0 0" if i % 2 == 0 else "0 1 0"
    pos = "0 0 0" if i == 0 else f"0 0 {-L}"
    lim = ' limited="true" range="-0.5 0.5"' if (i == 0 and limit_first) else ""
    foot = (
      f'<geom name="foot" type="sphere" pos="0 0 {-L}" size="{foot_r}" condim="{condim}"/>'
      if i == n_hinge - 1
      else ""
    )
    body += (
      f'<body name="link{i}" pos="{pos}">'
      f'<joint name="j{i}" type="hinge" axis="{axis}" damping="{damping}" stiffness="{stiffness}"{lim}/>'
      f'<geom type="capsule" fromto="0 0 0 0 0 {-L}" size="0.018" contype="0" conaffinity="0"/>'
      f"{foot}"
    )
  body += "</body>" * n_hinge
  # A fixed tendon coupling the first two hinges -> m.ntendon>0. The tendon constraint-row VJP is UNSUPPORTED
  # (MJPLAN_CSR step 7), so its mere structural presence must trip the capability gate (no silent wrong grad).
  tendon_block = (
    '<tendon><fixed name="t0"><joint joint="j0" coef="1"/><joint joint="j1" coef="-1"/></fixed></tendon>'
    if tendon
    else ""
  )
  return f"""
<mujoco>
  <option timestep="0.004" cone="{cone}" integrator="Euler"
          tolerance="1e-8" iterations="100" ls_iterations="50" gravity="0 0 -9.81">
    <flag contact="enable"/>
  </option>
  <default>
    <geom friction="0.7" solref="0.02 1" solimp="0 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05"/>
    <body name="base" pos="0 0 {base_z}">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/>
      {body}
    </body>
  </worldbody>
  {tendon_block}
</mujoco>
"""


def nvchain_setup(n_hinge, cone="elliptic", condim=3):
  """Build the free-base + hinge-chain + foot scene. qpos0 = chain straight down, foot penetrating the
  plane; qvel0 = a small base lateral velocity (so the foot slides -> the cone/friction path is live)."""
  mjm = mujoco.MjModel.from_xml_string(_nvchain_xml(n_hinge, cone, condim))
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  qpos0 = mjd.qpos.copy()
  qvel0 = np.zeros(mjm.nv, dtype=np.float64)
  qvel0[0] = 0.5
  return mjm, mjd, qpos0, qvel0


class ArticulatedContactNvTest(parameterized.TestCase):
  """nv>16 articulated-contact gradient gate -- the contact VJP must be correct beyond _MAX_NV."""

  def _onestep(self, m, mjm, mjd, qpos, qvel):
    d0 = mjw.put_data(mjm, mjd)
    d1 = mjw.put_data(mjm, mjd)
    for d in (d0, d1):
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    d0.qpos = wp.array(qpos.reshape(1, -1), dtype=wp.float32, requires_grad=True)
    d0.qvel = wp.array(qvel.reshape(1, -1), dtype=wp.float32, requires_grad=True)
    mjw.step(m, d0, d1)
    return d0, d1

  def test_articulated_contact_scene_makes_contact(self):
    """Runs without grad: the chain's foot actually contacts the floor and nv>_MAX_NV (so the gate is on
    the differentiable path and the bound is genuinely exercised). Pins the scene before the grad asserts."""
    mjm, mjd, qpos0, qvel0 = nvchain_setup(12)
    self.assertGreater(mjm.nv, _adjoint._MAX_NV)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)
    d.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
    d.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
    mjw.step(m, d)
    self.assertGreaterEqual(int(d.nacon.numpy()[0]), 1, "foot never contacts the floor")

  @parameterized.named_parameters(
    ("nv18_elliptic", 12, "elliptic"),
    ("nv18_pyramidal", 12, "pyramidal"),
    ("nv34_elliptic", 28, "elliptic"),
    ("nv34_pyramidal", 28, "pyramidal"),
  )
  def test_articulated_contact_residual_vjp_matches_fd(self, n_hinge, cone):
    """Single-step contact VJP vs FD on a scene with nv > _MAX_NV. Seeds a random next-velocity cotangent,
    runs step_backward, and checks d0.{qvel,qpos}.grad vs central FD -- crucially on the dofs BEYOND
    _MAX_NV (which the legacy dense kernel would silently drop)."""
    from mujoco_warp._src import adjoint

    mjm, mjd, qpos0, qvel0 = nvchain_setup(n_hinge, cone)
    nv, nq = mjm.nv, mjm.nq
    self.assertGreater(nv, _adjoint._MAX_NV, "scene must have nv > _MAX_NV to exercise the bound")
    m = mjw.put_model(mjm)

    d0, d1 = self._onestep(m, mjm, mjd, qpos0, qvel0)
    self.assertGreaterEqual(int(d1.nacon.numpy()[0]), 1, "no foot-floor contact at the test state")

    rng = np.random.default_rng(0)
    w = rng.standard_normal(nv).astype(np.float32)  # random output cotangent (avoid sum cancellation)
    seed = np.zeros((1, nv), dtype=np.float32)
    seed[0] = w
    d1.qvel.grad.assign(seed)
    adjoint.step_backward(m, d0, d1)
    ana_qvel = d0.qvel.grad.numpy()[0].astype(np.float64).copy()
    ana_qpos = d0.qpos.grad.numpy()[0].astype(np.float64).copy()

    def loss_after_step(qp, qv):
      d = mjw.put_data(mjm, mjd)
      d.qpos = wp.array(qp.reshape(1, -1), dtype=wp.float32)
      d.qvel = wp.array(qv.reshape(1, -1), dtype=wp.float32)
      mjw.step(m, d)
      return float(w @ d.qvel.numpy()[0])

    eps = 1.0e-4
    fd_qvel = np.zeros(nv)
    for i in range(nv):
      vp, vm = qvel0.copy(), qvel0.copy()
      vp[i] += eps
      vm[i] -= eps
      fd_qvel[i] = (loss_after_step(qpos0, vp) - loss_after_step(qpos0, vm)) / (2.0 * eps)
    scalar_q = [0, 1, 2] + list(range(7, nq))  # skip the free-joint quaternion coords 3..6
    fd_qpos = np.zeros(nq)
    for c in scalar_q:
      qp, qm = qpos0.copy(), qpos0.copy()
      qp[c] += eps
      qm[c] -= eps
      fd_qpos[c] = (loss_after_step(qp, qvel0) - loss_after_step(qm, qvel0)) / (2.0 * eps)

    def cos_rel(a, b):
      na, nb = np.linalg.norm(a), np.linalg.norm(b)
      cos = float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 1.0
      return cos, float(np.linalg.norm(a - b) / (nb + 1e-12))

    cos_v, rel_v = cos_rel(ana_qvel, fd_qvel)
    cos_q, rel_q = cos_rel(ana_qpos[scalar_q], fd_qpos[scalar_q])
    hi = list(range(_adjoint._MAX_NV, nv))  # qvel dofs beyond the legacy unroll bound
    qhi = list(range(7 + (_adjoint._MAX_NV - 6), nq))  # the corresponding qpos hinge coords (dofs >= _MAX_NV)
    cos_hi, rel_hi = cos_rel(ana_qvel[hi], fd_qvel[hi])
    cos_qhi, rel_qhi = cos_rel(ana_qpos[qhi], fd_qpos[qhi])
    print(
      f"\n[nvchain {cone} nv={nv}] dqvel cos={cos_v:.5f} rel={rel_v:.3f} | dqpos cos={cos_q:.5f} rel={rel_q:.3f} "
      f"| dofs>16 dqvel cos={cos_hi:.5f} rel={rel_hi:.3f} dqpos cos={cos_qhi:.5f} rel={rel_qhi:.3f}"
    )
    # The >16 dofs must be EXERCISED (FD nonzero) AND correct in AGGREGATE (the dense _MAX_NV path would
    # zero them entirely). Cosine + relative-L2 -- robust to per-entry f32 FD truncation on the stiff
    # long-chain contact columns (the EXACT correctness of the analytic is separately pinned to 1e-5 by
    # test_sparse_vs_dense_oracle_match; these gates verify FD-consistency at the f32 oracle's noise floor).
    self.assertGreater(np.abs(fd_qvel[hi]).max(), 1e-3, "scene does not exercise dofs > _MAX_NV")
    self.assertGreater(cos_hi, 0.99, f"{cone} nv={nv}: dqvel direction wrong on dofs > _MAX_NV (cos={cos_hi:.4f})")
    self.assertLess(rel_hi, 1e-1, f"{cone} nv={nv}: dqvel magnitude wrong on dofs > _MAX_NV (rel={rel_hi:.3f})")
    # NOTE: no per-slice assertion on dqpos[>16] -- with few high dofs (nv=18 -> 2 deepest hinges) those are
    # tiny-magnitude entries (small foot moment arm) where f32 FD relative error explodes. The >16 qpos
    # geometry path is covered by the overall cos_q below + the exact A/B (test_sparse_vs_dense_oracle_match)
    # + the walk-reaches->16 evidence from dqvel[>16]. (At nv=34 the 18-entry slice is clean: cos~0.999.)
    self.assertGreater(cos_v, 0.998, f"{cone} nv={nv}: dqvel direction off (cos={cos_v:.4f})")
    self.assertGreater(cos_q, 0.995, f"{cone} nv={nv}: dqpos direction off (cos={cos_q:.4f})")
    self.assertLess(rel_v, 7e-2, f"{cone} nv={nv}: dqvel magnitude off (rel={rel_v:.3f})")
    self.assertLess(rel_q, 9e-2, f"{cone} nv={nv}: dqpos magnitude off (rel={rel_q:.3f})")

  def test_articulated_contact_rollout_grad_matches_fd(self):
    """Short multi-step rollout: analytic BPTT through the nv>16 contact path vs central FD of a scalar
    base-position loss. The backward processes all dofs each step, so a >16 scatter bug corrupts even the
    base-dof gradient (contact couples base<->chain)."""
    from mujoco_warp._src import adjoint  # noqa: F401

    mjm, mjd, qpos0, qvel0 = nvchain_setup(12)
    nv = mjm.nv
    self.assertGreater(nv, _adjoint._MAX_NV)
    m = mjw.put_model(mjm)
    T = 12
    target = np.array([0.1, 0.0, float(qpos0[2])])  # base xyz target (a small x drift)
    target_v = wp.vec3(float(target[0]), float(target[1]), float(target[2]))

    def rollout(qv, taped):
      if taped:
        datas = [mjw.put_data(mjm, mjd) for _ in range(T + 1)]
        for d in datas:
          d.qpos.requires_grad = True
          d.qvel.requires_grad = True
        datas[0].qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32, requires_grad=True)
        datas[0].qvel = wp.array(qv.reshape(1, -1), dtype=wp.float32, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        tape = wp.Tape()
        with tape:
          for t in range(T):
            mjw.step(m, datas[t], datas[t + 1])
          wp.launch(_bounce_loss_kernel, dim=1, inputs=[datas[T].qpos, target_v], outputs=[loss])
        tape.backward(loss=loss)
        return float(loss.numpy()[0]), datas[0].qvel.grad.numpy()[0][:3].astype(np.float64).copy()
      d = mjw.put_data(mjm, mjd)
      d.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
      d.qvel = wp.array(qv.reshape(1, -1), dtype=wp.float32)
      for _ in range(T):
        mjw.step(m, d)
      qn = d.qpos.numpy()[0][:3].astype(np.float64)
      return float(np.dot(qn - target, qn - target)), None

    _, g_ana = rollout(qvel0, True)
    eps = 1.0e-4
    g_fd = np.zeros(3)
    for i in range(3):
      vp, vm = qvel0.copy(), qvel0.copy()
      vp[i] += eps
      vm[i] -= eps
      lp, _ = rollout(vp, False)
      lm, _ = rollout(vm, False)
      g_fd[i] = (lp - lm) / (2.0 * eps)
    cos = float(g_ana @ g_fd / (np.linalg.norm(g_ana) * np.linalg.norm(g_fd) + 1e-12))
    print(f"\n[nvchain rollout nv={nv} T={T}] ana={g_ana} fd={g_fd} cos={cos:.4f}")
    self.assertGreater(np.abs(g_fd).max(), 1e-4, "rollout loss insensitive to qvel0")
    np.testing.assert_allclose(g_ana, g_fd, rtol=5e-2, atol=5e-2, err_msg="nvchain rollout: analytic vs FD")

  def test_articulated_contact_constraint_capability_assert(self):
    """A model with nv>_MAX_NV AND a still-UNSUPPORTED non-contact row class must RAISE: no silent wrong
    gradient. JOINT-equality / ball / slide/hinge limits are now handled by the SPARSE constraint VJP (so a
    hinge limit at nv>_MAX_NV no longer raises); TENDON rows are not (MJPLAN_CSR step 7), so a tendon's
    structural presence must still trip the gate. (Guards that relaxing the gate for the landed classes did
    not silently open the door to the unlanded ones.)"""
    from mujoco_warp._src import adjoint

    mjm = mujoco.MjModel.from_xml_string(_nvchain_xml(12, "elliptic", tendon=True))
    self.assertGreater(mjm.nv, _adjoint._MAX_NV)
    self.assertGreater(mjm.ntendon, 0)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d0, d1 = self._onestep(m, mjm, mjd, mjd.qpos.copy(), np.zeros(mjm.nv))
    d1.qvel.grad.assign(np.ones((1, mjm.nv), dtype=np.float32))
    with self.assertRaises(NotImplementedError):
      adjoint.step_backward(m, d0, d1)

  def test_sparse_vs_dense_oracle_match(self):
    """Exact A/B (nv=6 bounce): the SPARSE contract-first path must reproduce the legacy DENSE _MAX_NV
    kernel to ~1e-5 (much tighter than FD), pinning the refactor numerically (not just FD-close)."""
    from mujoco_warp._src import adjoint

    mjm, mjd, qpos0, qvel0, _, _ = bounce_setup()
    m = mjw.put_model(mjm)
    state = mjw.put_data(mjm, mjd)
    state.qpos = wp.array(qpos0.reshape(1, -1), dtype=wp.float32)
    state.qvel = wp.array(qvel0.reshape(1, -1), dtype=wp.float32)
    for _ in range(49):  # reach a step with an active floor contact (cf. test_bounce_contact_residual_vjp)
      mjw.step(m, state)
    qpos = state.qpos.numpy()[0].copy()
    qvel = state.qvel.numpy()[0].copy()

    def grad(sparse):
      prev = _adjoint._USE_SPARSE_CONTACT
      _adjoint._USE_SPARSE_CONTACT = sparse
      try:
        d0, d1 = self._onestep(m, mjm, mjd, qpos, qvel)
        d1.qvel.grad.assign(np.ones((1, mjm.nv), dtype=np.float32))
        adjoint.step_backward(m, d0, d1)
        return d0.qpos.grad.numpy()[0].copy(), d0.qvel.grad.numpy()[0].copy()
      finally:
        _adjoint._USE_SPARSE_CONTACT = prev

    gq_s, gv_s = grad(True)
    gq_d, gv_d = grad(False)
    print(f"\n[sparse vs dense] max|dqvel diff|={np.abs(gv_s - gv_d).max():.2e} max|dqpos diff|={np.abs(gq_s - gq_d).max():.2e}")
    np.testing.assert_allclose(gv_s, gv_d, rtol=1e-4, atol=1e-5, err_msg="sparse vs dense oracle: dqvel")
    np.testing.assert_allclose(gq_s, gq_d, rtol=1e-4, atol=1e-5, err_msg="sparse vs dense oracle: dqpos")


# ============================================================================================
# NON-CONTACT constraint residual SPARSE/CSR rework (MJPLAN_CSR.md step 1): isolated gates on the
# constraint_adjoint gather/leaf/scatter + the adjoint._residual_constraint_sparse orchestration --
# (A) gather Z==Jλ + scatter res==Jᵀx̄ (friction P̄-gate), (B) leaf arbitrary-seed VJP vs FD, (C) the
# residual-layer contraction φ=-Σ(Jλ)·efc.force value + FD-VJP (frozen qacc/λ/active-set), (D) CSR-vs-
# dense equivalence. The new path is validated in ISOLATION (the kernels/wrapper called directly);
# step_backward's production routing to it is MJPLAN_CSR step 3.
# ============================================================================================
_NC_TYPES = (_adjoint._EQUALITY, _adjoint._LIMIT_JOINT, _adjoint._FRICTION_DOF)
_NC_POSBEARING = (_adjoint._EQUALITY, _adjoint._LIMIT_JOINT)


def _reconstruct_efc_J(d, w, nefc, nv, is_sparse):
  """Dense (nefc, nv) numpy J from the dense (J[w,row,i]) or CSR (J[w,0,rowadr+k], colind) layout."""
  J = np.zeros((nefc, nv))
  if is_sparse:
    rownnz = d.efc.J_rownnz.numpy()[w]
    rowadr = d.efc.J_rowadr.numpy()[w]
    colind = d.efc.J_colind.numpy()[w, 0]
    vals = d.efc.J.numpy()[w, 0]
    for row in range(nefc):
      for k in range(int(rownnz[row])):
        J[row, int(colind[int(rowadr[row]) + k])] = vals[int(rowadr[row]) + k]
  else:
    Jd = d.efc.J.numpy()[w]
    for row in range(nefc):
      J[row, :] = Jd[row, :nv]
  return J


def _launch_gather(m, d, lam):
  """Run constraint_adjoint._constraint_gather (is_sparse-specialized) -> (Z, invw)."""
  nworld, njmax, nv = d.qpos.shape[0], d.efc.type.shape[1], m.nv
  Z = wp.zeros((nworld, njmax), dtype=float)
  invw = wp.zeros((nworld, njmax), dtype=float)
  wp.launch(
    _constraint_adjoint._constraint_gather(m.is_sparse),
    dim=(nworld, njmax),
    inputs=[m.jnt_dofadr, m.dof_invweight0, lam, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind,
            d.efc.J, d.efc.state, d.efc.type, d.efc.id, d.nefc, nv],
    outputs=[Z, invw],
  )
  return Z, invw


def _launch_scatter(m, d, adjP, adjV):
  """Run constraint_adjoint._constraint_scatter -> (res_qvel, res_dof)."""
  nworld, nv = d.qpos.shape[0], m.nv
  res_qvel = wp.zeros((nworld, nv), dtype=float)
  res_dof = wp.zeros((nworld, nv), dtype=float)
  wp.launch(
    _constraint_adjoint._constraint_scatter(m.is_sparse),
    dim=(nworld, d.efc.type.shape[1]),
    inputs=[d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J, d.efc.state, d.efc.type, d.nefc, nv,
            adjP, adjV],
    outputs=[res_qvel, res_dof],
  )
  return res_qvel, res_dof


def _csr_chain(jacobian, n_hinge=30, limit_hinge=22, fric_hinges=(8, 25)):
  """Free base + n_hinge serial hinges (nv = 6 + n_hinge). A tight limit on hinge[limit_hinge] (its qpos set
  beyond range -> active) + frictionloss on fric_hinges -> active non-contact rows at HIGH dof indices.
  opt.jacobian forces dense vs CSR efc.J. No ground contact (isolates the non-contact residual)."""
  body = ""
  for i in range(n_hinge):
    axis = "1 0 0" if i % 2 == 0 else "0 1 0"
    lim = ' limited="true" range="-0.05 0.05"' if i == limit_hinge else ""
    fr = ' frictionloss="0.3"' if i in fric_hinges else ""
    pos = "0 0 0" if i == 0 else "0 0 -0.1"
    body += f'<body pos="{pos}"><joint type="hinge" axis="{axis}"{lim}{fr} damping="0.5"/><geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.2"/>'
  body += "</body>" * n_hinge
  xml = f"""
<mujoco>
  <option timestep="0.004" integrator="Euler" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <worldbody><body name="base" pos="0 0 1"><freejoint/>
    <geom type="sphere" size="0.05" mass="1"/>{body}</body></worldbody>
</mujoco>"""
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjm.opt.jacobian = {"dense": mujoco.mjtJacobian.mjJAC_DENSE, "sparse": mujoco.mjtJacobian.mjJAC_SPARSE}[jacobian]
  mjm.jnt_solimp[:, 0] = 0.0  # soft limit
  mjd = mujoco.MjData(mjm)
  qadr = mjm.jnt_qposadr[mjm.body_jntadr[mjm.body("base").id] + 1 + limit_hinge]  # the limited hinge's qpos slot
  mjd.qpos[qadr] = 0.2  # violate the +0.05 limit -> active
  mjd.qvel[:] = 0.0
  mjd.qvel[6 + fric_hinges[0]] = 0.5  # drive a frictional dof -> its friction row is active
  mujoco.mj_forward(mjm, mjd)
  return mjm, mjd


def _step_with_grad(m, mjm, mjd):
  """One mjw.step from (mjd.qpos, mjd.qvel) with grad arrays; returns (d0, d1)."""
  d0 = mjw.put_data(mjm, mjd)
  d1 = mjw.put_data(mjm, mjd)
  for d in (d0, d1):
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  mjw.step(m, d0, d1)
  return d0, d1


def _ift_lambda(m, d1):
  """Build the solver context at the converged qacc and return (ctx.Jaref, an injected random λ)."""
  ctx = _solver._create_solver_context(m, d1)
  _solver.init_context(m, d1, ctx, grad=True)
  return ctx.Jaref


class ConstraintSparseResidualTest(parameterized.TestCase):
  """MJPLAN_CSR step 1: the SPARSE/CSR non-contact constraint residual VJP, validated in isolation."""

  @parameterized.named_parameters(("dense", "dense"), ("sparse", "sparse"))
  def test_gather_scatter_transpose(self, jacobian):
    """Gate 5a/5b: the gather reduces Z_e=Σ_i J_ei·λ_i and the scatter applies res=Jᵀx̄ -- both vs a numpy J
    reconstruction of the dense/CSR layout. Friction rows must contribute 0 to res_dof (must-fix #1)."""
    mjm, mjd = _csr_chain(jacobian)
    m = mjw.put_model(mjm)
    self.assertEqual(m.is_sparse, jacobian == "sparse")
    _, d1 = _step_with_grad(m, mjm, mjd)
    nv, nworld = m.nv, 1
    ne = int(d1.nefc.numpy()[0])
    typ = d1.efc.type.numpy()[0, :ne]
    st = d1.efc.state.numpy()[0, :ne]
    J = _reconstruct_efc_J(d1, 0, ne, nv, m.is_sparse)
    active_nc = [r for r in range(ne) if typ[r] in _NC_TYPES and st[r] != _adjoint._SATISFIED]
    self.assertTrue(any(typ[r] == _adjoint._LIMIT_JOINT for r in active_nc), "no active joint-limit row")
    self.assertGreater(nv, _adjoint._MAX_NV, "scene must be nv>_MAX_NV to exercise CSR at high dof indices")

    rng = np.random.default_rng(0)
    lam = wp.array(rng.standard_normal((nworld, nv)).astype(np.float32), dtype=float)
    Z, _ = _launch_gather(m, d1, lam)
    Zn = Z.numpy()[0]
    lam_n = lam.numpy()[0]
    for r in range(ne):
      expect = float(J[r] @ lam_n) if r in active_nc else 0.0
      self.assertAlmostEqual(Zn[r], expect, places=4, msg=f"gather Z[{r}] (type {typ[r]})")

    adjP = wp.array(rng.standard_normal((nworld, d1.efc.type.shape[1])).astype(np.float32), dtype=float)
    adjV = wp.array(rng.standard_normal((nworld, d1.efc.type.shape[1])).astype(np.float32), dtype=float)
    res_qvel, res_dof = _launch_scatter(m, d1, adjP, adjV)
    Pn, Vn = adjP.numpy()[0], adjV.numpy()[0]
    qv_ref = np.zeros(nv)
    dof_ref = np.zeros(nv)
    for r in active_nc:
      qv_ref += J[r] * Vn[r]  # res_qvel += Jᵀ V̄ for ALL non-contact rows
      if typ[r] in _NC_POSBEARING:
        dof_ref += J[r] * Pn[r]  # res_dof += Jᵀ P̄ for POSITION-BEARING rows only (friction excluded)
    np.testing.assert_allclose(res_qvel.numpy()[0], qv_ref, rtol=1e-4, atol=1e-5, err_msg="scatter res_qvel != Jᵀ V̄")
    np.testing.assert_allclose(res_dof.numpy()[0], dof_ref, rtol=1e-4, atol=1e-5, err_msg="scatter res_dof != Jᵀ P̄")
    # the friction dofs must get a res_qvel contribution but NO res_dof contribution
    fric_dofs = [int(d1.efc.id.numpy()[0, r]) for r in active_nc if typ[r] == _adjoint._FRICTION_DOF]
    if fric_dofs:
      contrib = np.zeros(nv)
      for r in active_nc:
        if typ[r] == _adjoint._FRICTION_DOF:
          contrib[int(d1.efc.id.numpy()[0, r])] += J[r] @ np.ones(nv) * Pn[r]
      # (covered by the exact assert above; this just documents the friction rows exist)
      self.assertTrue(any(typ[r] == _adjoint._FRICTION_DOF for r in active_nc), "scene should have a friction row")

  def test_contraction_value_and_vjp(self):
    """Gate 2a/2b: on a settled stiff hinge limit (nonzero force, frozen active set) -- (2a) the leaf value
    Σφ == -Σ_e (Jλ)_e·efc.force_e (the f-anchor is byte-exact); (2b) central FD of the contraction
    φ=-Σ(Jλ)·f at FROZEN qacc/λ/active-set (f recomputed from the ENGINE's efc.{J,aref,D} at the perturbed
    state) vs the wired res_qvel/res_dof. Isolates the residual VJP from the integrator/IFT/RNE."""
    mjm = mujoco.MjModel.from_xml_string(_HINGE_LIMIT)
    mjm.jnt_solimp[:, 0] = 0.0
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    for _ in range(80):
      mujoco.mj_step(mjm, mjd)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    nv = m.nv
    qpos0 = mjd.qpos.astype(np.float64).copy()
    qvel0 = mjd.qvel.astype(np.float64).copy()
    _, d1 = _step_with_grad(m, mjm, mjd)
    ne = int(d1.nefc.numpy()[0])
    type0 = d1.efc.type.numpy()[0, :ne].copy()
    state0 = d1.efc.state.numpy()[0, :ne].copy()
    self.assertIn(_adjoint._LIMIT_JOINT, list(type0), "expected an active joint-limit row")
    qacc0 = d1.qacc.numpy()[0].astype(np.float64).copy()
    ctx_Jaref = _ift_lambda(m, d1)
    rng = np.random.default_rng(1)
    lam_n = rng.standard_normal(nv).astype(np.float64)
    lam = wp.array(lam_n.reshape(1, -1).astype(np.float32), dtype=float)

    # (2a) value: the leaf phi summed over active rows == -Σ Z·efc.force.
    Z, _ = _launch_gather(m, d1, lam)
    Zn = Z.numpy()[0]
    force0 = d1.efc.force.numpy()[0, :ne].astype(np.float64)
    phi_ref = float(-(Zn[:ne] * force0).sum())
    res_qvel = wp.zeros((1, nv), dtype=float)
    res_dof = wp.zeros((1, nv), dtype=float)
    _adjoint._residual_constraint_sparse(m, d1, ctx_Jaref, lam, res_qvel, res_dof)
    # recompute Σφ directly via the leaf forward for the value check
    phi_warp = _contraction_value(m, d1, ctx_Jaref, lam)
    self.assertAlmostEqual(phi_warp, phi_ref, places=4, msg="leaf value != -Σ(Jλ)·efc.force (f-anchor)")

    # (2b) FD of the contraction at frozen qacc/λ/active-set, using the engine's efc recompute.
    def phi_at(qp, qv):
      d = mjw.put_data(mjm, mjd)
      d.qpos = wp.array(qp.reshape(1, -1).astype(np.float32), dtype=float)
      d.qvel = wp.array(qv.reshape(1, -1).astype(np.float32), dtype=float)
      mjw.forward(m, d)
      nn = int(d.nefc.numpy()[0])
      assert nn == ne and (d.efc.type.numpy()[0, :nn] == type0).all() and (d.efc.state.numpy()[0, :nn] == state0).all(), "active set changed across FD"
      Jp = _reconstruct_efc_J(d, 0, nn, nv, m.is_sparse)
      aref = d.efc.aref.numpy()[0, :nn].astype(np.float64)
      D = d.efc.D.numpy()[0, :nn].astype(np.float64)
      jaref = Jp @ qacc0 - aref  # FROZEN qacc0
      f = -D * jaref  # QUADRATIC limit (the settled hinge limit is never saturated)
      Zp = Jp @ lam_n  # FROZEN λ; J recomputed at the perturbed qpos
      return float(-(Zp * f).sum())

    eps = 1e-6
    fd_qvel = np.zeros(nv)
    fd_qpos = np.zeros(nv)
    for i in range(nv):
      vp, vm = qvel0.copy(), qvel0.copy()
      vp[i] += eps
      vm[i] -= eps
      fd_qvel[i] = (phi_at(qpos0, vp) - phi_at(qpos0, vm)) / (2 * eps)
      qp, qm = qpos0.copy(), qpos0.copy()
      qp[i] += eps
      qm[i] -= eps
      fd_qpos[i] = (phi_at(qp, qvel0) - phi_at(qm, qvel0)) / (2 * eps)
    ana_qvel = res_qvel.numpy()[0].astype(np.float64)
    ana_qpos = res_dof.numpy()[0].astype(np.float64)  # 1:1 dof->qpos for the hinge

    def cos_rel(a, b):
      na, nb = np.linalg.norm(a), np.linalg.norm(b)
      cos = float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 1.0
      return cos, float(np.linalg.norm(a - b) / (nb + 1e-12))

    cv, rv = cos_rel(ana_qvel, fd_qvel)
    cq, rq = cos_rel(ana_qpos, fd_qpos)
    print(f"\n[constraint contraction] value warp={phi_warp:.5f} ref={phi_ref:.5f} | dqvel cos={cv:.5f} rel={rv:.4f} | dqpos cos={cq:.5f} rel={rq:.4f}")
    self.assertGreater(np.abs(fd_qpos).max(), 1e-3, "contraction insensitive to qpos (limit stiffness missing)")
    self.assertGreater(cq, 0.999, f"res_dof direction wrong (cos={cq:.5f})")
    self.assertLess(rq, 2e-2, f"res_dof magnitude wrong (rel={rq:.4f})")
    self.assertGreater(cv, 0.999, f"res_qvel direction wrong (cos={cv:.5f})")
    self.assertLess(rv, 2e-2, f"res_qvel magnitude wrong (rel={rv:.4f})")

  def test_csr_vs_dense_equivalence(self):
    """Gate 3: the SAME nv>32 limit+friction scene with opt.jacobian forced DENSE vs SPARSE must yield
    IDENTICAL res_qvel/res_dof from the wrapper (frozen efc data + the same injected λ)."""
    def grads(jacobian):
      mjm, mjd = _csr_chain(jacobian)
      m = mjw.put_model(mjm)
      _, d1 = _step_with_grad(m, mjm, mjd)
      nv = m.nv
      ctx_Jaref = _ift_lambda(m, d1)
      rng = np.random.default_rng(7)
      lam = wp.array(rng.standard_normal((1, nv)).astype(np.float32), dtype=float)
      res_qvel = wp.zeros((1, nv), dtype=float)
      res_dof = wp.zeros((1, nv), dtype=float)
      _adjoint._residual_constraint_sparse(m, d1, ctx_Jaref, lam, res_qvel, res_dof)
      return res_qvel.numpy()[0].copy(), res_dof.numpy()[0].copy()

    qv_d, dof_d = grads("dense")
    qv_s, dof_s = grads("sparse")
    print(f"\n[csr vs dense] max|dqvel diff|={np.abs(qv_s - qv_d).max():.2e} max|dres_dof diff|={np.abs(dof_s - dof_d).max():.2e}")
    np.testing.assert_allclose(qv_s, qv_d, rtol=1e-4, atol=1e-5, err_msg="CSR vs dense: res_qvel")
    np.testing.assert_allclose(dof_s, dof_d, rtol=1e-4, atol=1e-5, err_msg="CSR vs dense: res_dof")


def _contraction_value(m, d1, ctx_Jaref, lam):
  """Σ over active non-contact rows of the leaf φ_e = -Z·f (gather + leaf forward only)."""
  nworld, njmax, nv = d1.qpos.shape[0], d1.efc.type.shape[1], m.nv
  Z, invw = _launch_gather(m, d1, lam)
  for arr in (d1.efc.pos, d1.efc.vel):
    arr.requires_grad = True
  phi = wp.zeros((nworld, njmax), dtype=float, requires_grad=True)
  wp.launch(
    _constraint_adjoint._constraint_row_phi,
    dim=(nworld, njmax),
    inputs=[d1.efc.pos, d1.efc.vel, Z, invw, d1.efc.margin, d1.efc.aref, d1.efc.D, d1.efc.force, ctx_Jaref,
            d1.efc.state, d1.efc.type, d1.efc.id, m.dof_solref, m.dof_solimp, m.dof_frictionloss, m.eq_solref,
            m.eq_solimp, m.jnt_solref, m.jnt_solimp, d1.nefc, m.opt.timestep, m.opt.disableflags],
    outputs=[phi],
  )
  return float(phi.numpy()[0].sum())


# ---------------------------------------------------------------------------------------------------
# fwd_kinematics site_xpos position VJP (differentiable observations) -- the nq!=nv free-base lift.
# Regresses the bug where adjoint._site_jac_vjp wrote its DOF-indexed gradient straight into qpos.grad
# (correct only for fixed-base hinge/slide where qposadr==dofadr; for a free/ball base nq!=nv, so every
# post-free joint was misindexed and the 3 angular dofs were dumped into quaternion slots with no lift ->
# a wrong-direction site gradient). The fix routes the tangent VJP through _dof_to_qpos (the same
# quaternion lift the contact ∂qpos path uses). Airborne (no floor) so this isolates the FK plumbing.
# ---------------------------------------------------------------------------------------------------

_FK_FREEBASE = """
<mujoco>
  <option timestep="0.004"/>
  <worldbody>
    <body name="base" pos="0 0 1.0">
      <freejoint/>
      <geom type="box" size="0.08 0.08 0.04" mass="1.0"/>
      <body name="link" pos="0.08 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.2"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>
        <site name="tip" pos="0.2 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="hinge" kp="5.0"/>
  </actuator>
</mujoco>
"""

# same chain, base WELDED to world (nq==nv, hinge 1:1) -- proves the fix is unchanged for fixed-base models
# (the reacher case that originally validated the hook), where _dof_to_qpos reduces to res_qpos[qadr]+=res_dof[dadr].
_FK_FIXEDBASE = _FK_FREEBASE.replace("<freejoint/>", "")


@wp.kernel
def _site_tipdist_kernel(site_xpos: wp.array2d[wp.vec3], tip: int, tgt: wp.vec3, w: float, loss: wp.array[float]):
  e = site_xpos[0, tip] - tgt
  wp.atomic_add(loss, 0, w * wp.dot(e, e))


def _fk_taped_ctrl_grad(mjm, mjd, m, tip, tgt, H, ctrl):
  """Analytic d(Σ_t ((t+1)/H)^2 ||site_xpos[tip]-tgt||^2)/d(ctrl) via the taped
  step ∘ fwd_kinematics ∘ site_xpos backward, stacked (H, nu)."""
  datas = [mjw.put_data(mjm, mjd) for _ in range(H + 1)]
  for dd in datas:
    dd.qpos.requires_grad = True
    dd.qvel.requires_grad = True
    dd.site_xpos.requires_grad = True
  for t in range(H):
    datas[t].ctrl = wp.array(ctrl[t].reshape(1, -1).astype(np.float32), dtype=float, requires_grad=True)
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(H):
      mjw.step(m, datas[t], datas[t + 1])
      mjw.fwd_kinematics(m, datas[t + 1])
      wp.launch(_site_tipdist_kernel, dim=1,
                inputs=[datas[t + 1].site_xpos, tip, wp.vec3(*tgt), float(((t + 1) / H) ** 2)], outputs=[loss])
  tape.backward(loss=loss)
  return np.array([np.nan_to_num(datas[t].ctrl.grad.numpy()[0]) for t in range(H)], np.float64)


def _fk_fd_ctrl_grad(mjm, mjd, m, tip, tgt, H, ctrl, eps=1e-4):
  """Central FD of the SAME forward rollout w.r.t. ctrl (Euclidean -- no quaternion in the FD itself)."""
  def rollout(cs):
    d = mjw.put_data(mjm, mjd)
    total = 0.0
    for t in range(H):
      d.ctrl = wp.array(cs[t].reshape(1, -1).astype(np.float32), dtype=float)
      mjw.step(m, d)
      mjw.fwd_kinematics(m, d)
      sx = d.site_xpos.numpy()[0, tip].astype(np.float64)
      total += ((t + 1) / H) ** 2 * float(np.sum((sx - tgt) ** 2))
    return total
  g = np.zeros((H, mjm.nu))
  for t in range(H):
    for j in range(mjm.nu):
      cp = ctrl.copy(); cp[t, j] += eps
      cm = ctrl.copy(); cm[t, j] -= eps
      g[t, j] = (rollout(cp) - rollout(cm)) / (2 * eps)
  return g


class FwdKinematicsSiteGradientTest(parameterized.TestCase):
  """Position VJP of forward.fwd_kinematics (site_xpos -> qpos.grad -> ctrl), the SHAC diff-observation
  primitive. Airborne (no contact) isolates the FK plumbing: `freebase` regresses the nq!=nv quaternion
  lift (was cos ~ -0.25 before the fix), `fixedbase` proves the hinge 1:1 path is unchanged."""

  @parameterized.named_parameters(
    ("freebase", _FK_FREEBASE, "free"),
    ("fixedbase", _FK_FIXEDBASE, "fixed"),
  )
  def test_site_ctrl_grad_matches_fd(self, xml, tag):
    if not _grad_available():
      self.skipTest("pending adjoint.py: differentiable step (MJPLAN.md Stage 1)")
    from mujoco_warp._src import adjoint  # noqa: F401  registers step + position backward hooks

    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    tip = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, "tip")
    tgt = mjd.site_xpos[tip].astype(np.float64) + np.array([0.08, 0.0, 0.08])  # pull the tip up+forward
    H = 6
    ctrl = np.full((H, mjm.nu), 0.5, np.float64)      # drive the hinge so tip (and free base) actually move
    analytic = _fk_taped_ctrl_grad(mjm, mjd, m, tip, tgt, H, ctrl).ravel()
    fd = _fk_fd_ctrl_grad(mjm, mjd, m, tip, tgt, H, ctrl).ravel()
    _assert_smooth_grad(self, analytic, fd, cos_min=0.99, rel_max=0.08, tag=f"fk_site:{tag}")


if __name__ == "__main__":
  wp.init()
  absltest.main()
