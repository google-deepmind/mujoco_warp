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


@wp.kernel
def _sumsq_qvel_kernel(qvel: wp.array2d[float], loss: wp.array[float]):
  i = wp.tid()
  wp.atomic_add(loss, 0, qvel[0, i] * qvel[0, i])


def _sysid_taped_grad(mjm, mjd, H, field):
  """Analytic d(||qvel_H||^2)/d(m.<field>) through the taped rollout (adjoint.py). The param adjoint falls
  out of the residual-VJP and ACCUMULATES into m.<field>.grad over the rollout (a shared leaf)."""
  m = mjw.put_model(mjm)
  arr = getattr(m, field)  # put_model stores it broadcast (stride-0); rebuild contiguous for a clean grad
  setattr(m, field, wp.array(arr.numpy(), dtype=wp.float32, requires_grad=True))
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
  return np.nan_to_num(getattr(m, field).grad.numpy()[0].astype(np.float64).copy())


def _sysid_mjc_fd_grad(mjm, mjd, H, field, xml=_SYSID_ARM, eps=1e-5):
  """Float64 MuJoCo-C central FD of ||qvel_H||^2 w.r.t. each component of model.<field> -- the param
  analog of mjd_transitionFD (which has no model-param mode): a clean, engine-independent oracle."""

  def rollout(vals):
    m2 = mujoco.MjModel.from_xml_string(xml)
    getattr(m2, field)[:] = vals
    md = mujoco.MjData(m2)
    md.qpos[:] = mjd.qpos
    md.qvel[:] = mjd.qvel
    mujoco.mj_forward(m2, md)
    for _ in range(H):
      mujoco.mj_step(m2, md)
    return float(np.dot(md.qvel, md.qvel))

  x0 = getattr(mjm, field).astype(np.float64).copy()
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


if __name__ == "__main__":
  wp.init()
  absltest.main()
