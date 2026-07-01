"""Franka Research 3 (FR3) whole-arm INERTIA system identification -- recover every arm link's principal
moments of inertia via the analytic adjoint of differentiable mujoco_warp.

One GT ("real") FR3 arm and NUM_ENVS sys-id arms track the SAME joint-space position chirp (PACE-style
excitation) through a compliant servo -- gravity on, contacts disabled. Each sys-id arm starts with every
link's inertia physically-consistently wrong; recovering theta drives its open-loop trajectory back onto
the GT's. Inertia is parameterized the way ../mjlab does it (Rucker & Wensing RA-L 2022): principal moments
I are realizable iff the mass-covariance eigenvalues sigma_i = (sum(I) - 2*I_i)/2 are positive, so we
optimize the unconstrained theta = log(sigma) (MODE=full) or a per-link log-scale of the whole tensor
(MODE=scale, the demo default).

The whole chain is ONE autodiff graph: a Warp kernel maps theta -> body_inertia ON THE TAPE, mjw.step's
analytic backward writes body_inertia.grad (body_inertia enters the dynamics only through cinert, so its
adjoint comes from the rne-proper reverse routed through the source-AD cinert leaf:
adjoint.smooth_param_backward -> smooth_adjoint.inertia_param_vjp), and tape.backward continues straight
into theta.grad for warp.optim.Adam. No host-side Jacobian composition.

The fit is OPEN-LOOP over the entire chirp: state integrates continuously under the wrong inertia --
teacher-forced resets destroy the horizon-compounding signal that makes weakly-loaded links observable.
Only the GRADIENT is truncated (TBPTT, WINDOW-step segments each on its own tape). The FD cross-check at
startup is a one-time gate: expect cos ~= 1; ratio < 1 is the DELIBERATE truncation bias, not an error.

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/franka_sysid.py
"""

import os
import sys

import mujoco
import numpy as np
import warp as wp
import warp.optim

import mujoco_warp as mjw
from mujoco_warp._src import adjoint  # noqa: F401  registers the analytic step() backward

sys.path.insert(0, os.path.dirname(__file__))
import inertia_param as ip  # noqa: E402  physically-consistent inertia parameterization (host helpers)
import viz  # noqa: E402  shared renderer


# --- what is identified -------------------------------------------------------------------------------

SCENE = "benchmarks/franka_fr3/scene_sysid.xml"  # arm-only FR3 (nq=7), contacts disabled in the XML
NARM = 7
NLINK = int(os.environ.get("MJW_NLINK", 6))  # fr3_link1..fr3_link{NLINK}. link7 is a near-massless flange
# (inertia ~2e-4, 100x smaller than the others) -> dynamically negligible/unidentifiable, excluded.
LINKS = [f"fr3_link{i}" for i in range(1, NLINK + 1)]
MODE = os.environ.get("MJW_MODE", "scale")
#  "scale": ONE log-scale per link (I -> s*I). Every link's scale is identifiable, so |I-I_gt| -> 0 exactly,
#           and a pure-scale error maximizes trajectory divergence per unit of wrongness. The demo default.
#  "full":  the three principal moments (log mass-covariance eigenvalues). Research-grade, but a serial
#           arm's proximal moments are only PARTIALLY identifiable (base-parameter degeneracy: link1 only
#           contributes n^T I n about its fixed axis) -> |I-I_gt| plateaus at that floor, by physics.
NP = 1 if MODE == "scale" else 3  # params per link

# --- excitation (PACE-style joint-space position chirp) ------------------------------------------------

CHIRP_DURATION = float(os.environ.get("MJW_DUR", 3.0))  # sweep duration [s]
AMP_SCALE = float(os.environ.get("MJW_AMP", 3.0))  # x base PACE amplitudes at the slow start; the taper
# shrinks amplitude as the sweep speeds up (constant peak velocity) -> big slow sways, no violent motion
CHIRP_KWARGS = {
  "f0": float(os.environ.get("MJW_F0", 0.1)),
  "f1": float(os.environ.get("MJW_F1", 3.0)),  # the compliant servo puts sqrt(kp/M) inside this band, so
  # inertia shapes the low-frequency sway -- no fast ringing needed
  "jnt_amp": list(AMP_SCALE * np.array([0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.2])),  # PACE base amps
  "ramp_up": 0.4,  # scaled to the short sweep (the 2-3 s defaults suit a 10-20 s sweep)
  "ramp_down": 0.4,
  "pad_start": 0.1,
  "pad_end": 0.1,
  "taper_pivot": float(os.environ.get("MJW_TAPER_PIVOT", 0.7)),  # Hz: full amplitude below, ~1/f above
  "taper_pow": 1.0,  # 1 = constant peak joint velocity across the sweep
}
NEXC = int(os.environ.get("MJW_NEXC", 1))  # excitations fitted jointly (>1 = phase-decorrelated chirps)

# COMPLIANT servo mode for the ID protocol (experimental-setup choice, never the robot's physics): the
# native FR3 servo (kp=4500) tracks so well that inertia never shapes the trajectory; softening kp makes
# the M*A*w^2 tracking-error term a VISIBLE fraction of the commanded amplitude at f<=3 Hz. kv keeps the
# damping ratio moderate (smooth sway, not ringing).
KP_SCALE = float(os.environ.get("MJW_KP", 0.2))
KV_SCALE = float(os.environ.get("MJW_KV", 0.02))

# --- fit ------------------------------------------------------------------------------------------------

WINDOW = int(os.environ.get("MJW_WINDOW", 40))  # truncated-BPTT segment length [steps]. Longer = deeper
# credit assignment per segment, but closer to the chaotic-gradient regime of full-horizon BPTT.
NUM_ENVS = int(os.environ.get("MJW_NUM_ENVS", 4))
PERTURB = float(os.environ.get("MJW_PERTURB", 2.3 if MODE == "scale" else 2.0))  # per-link theta magnitude:
# MODE=scale starts each tensor at e^{+-2.3} (~0.1x..10x) -- wildly wrong but physically valid; a uniform
# scale moves the servo-tracked dynamics less than an anisotropic error of the same theta-norm.
DIVERGE_MIN = float(os.environ.get("MJW_DIVERGE_MIN", 0.25))  # rad: minimum free-run max|dq| vs GT an env's
# initial inertia must produce. Equal theta-distance does NOT mean equal trajectory effect (a draw can land
# on weakly-observable combos and look identical to GT); below-bar envs get their directions redrawn.
LR = float(os.environ.get("MJW_LR", 0.03))
WD = float(os.environ.get("MJW_WD", 0.0 if MODE == "scale" else 2e-3))  # Tikhonov pull toward each env's
# INITIAL theta. scale: 0 (everything identifiable; an anchor would only bias the answer). full: anchors
# the structurally-flat directions where the data gradient is ~noise and bare Adam would random-walk.
STEPS = int(os.environ.get("MJW_STEPS", 160))
BETAS = (float(os.environ.get("MJW_BETA1", 0.7)), float(os.environ.get("MJW_BETA2", 0.95)))  # SHAC betas
# (the dm_suite.py recipe): fast-adapting per-parameter moments, no LR schedule needed
SEED = int(os.environ.get("MJW_SEED", 0))

# --- render ---------------------------------------------------------------------------------------------

RENDER_STRIDE = int(os.environ.get("MJW_RENDER_STRIDE", 2))
# LAYOUT: one FRONT row of all sys-id arms + the GT reference arm alone in a BACK row, centered. Rows run
# along the camera's screen-right axis (derived from CAM_AZIMUTH), so env0..env3 read left->right in the
# same order as the banner numbers, and the GT (farther, so drawn higher on screen at this elevation) is
# visible between/above the front row.
ROW_CENTER = np.array([0.35, 0.0])  # world xy midpoint between the two rows (arm BASES; the FR3's visual
# mass leans ~+0.4m in x, which is why CAM_LOOKAT x sits at 0.75)
ROW_SPACING = 1.2  # env arm spacing along screen-right [m]
ROW_SEP = 1.6      # depth between the env row (front) and the GT row (back) [m]
CAM_LOOKAT = [0.75, 0.0, 0.3]
# LONG LENS: the camera sits far back with a narrow fovy so the view rays across the 3.6m row are near-
# parallel (angular spread ~±8° vs ±20° at the old distance 5.0 / fovy 45) -- all four sys-id arms and
# their ribbons are seen from (nearly) the SAME angle, so identical trajectories LOOK identical per cell.
# CAM_ORTHO=1 goes fully parallel (exactly identical angles; CAM_FOVY then = vertical half-extent [m]).
CAM_DISTANCE = 12.0
CAM_FOVY = 19.0       # deg (perspective) | meters half-height (orthographic)
CAM_ORTHO = int(os.environ.get("MJW_CAM_ORTHO", 0))
CAM_AZIMUTH = 110.0   # the friction-demo view family
CAM_ELEVATION = -28.0
OUT_MP4 = os.environ.get(
  "MJW_RENDER_PATH", os.path.join(os.path.dirname(__file__), "reports", "assets", "franka_sysid.mp4")
)
HISTORY_NPZ = os.environ.get("MJW_HISTORY_PATH", "/tmp/fr3_history.npz")  # optimize() saves the per-iter
# theta history here; MJW_RENDER_FROM=<npz> re-renders the video from it WITHOUT re-fitting (~2 min vs
# ~40 -- iterate on ribbons/camera against the REAL converged history)
W, H, FPS = 1024, 768, 30
GT_RGBA = np.array([1.0, 0.5, 0.12, 1.0], dtype=np.float32)  # solid orange = GT (true-inertia) reference
# Env ribbons are colored by the dm_suite-style OPTIMIZATION COLORMAP (viz.bourke_color_map), not by env
# identity: each env's remaining scale error (log-normalized to its own iter-0 error) sweeps the map
# blue (wrong) -> cyan -> green -> yellow -> ORANGE (converged). The sweep TERMINATES at the GT orange, so
# at x1.0 every cell's track turns orange and reads as the GT ribbon -- while mid-run a geometrically
# merged track that is still cyan/green says "trajectory matches but the params haven't converged yet".
_BOURKE_ORANGE_V = 0.875  # bourke(0,1,0.875) = (1.0, 0.5, 0.0) ~= GT orange; sweep v = 0.875*(1-u)


def _env_ribbon_rgb(serr_e, serr0_e):
  """Per-env ribbon color for one shown iteration: u = normalized log scale error (1 at the env's initial
  wrongness, 0 at x1.0) -> bourke sweep ending exactly on the GT orange."""
  u = float(np.clip(np.log(max(serr_e, 1.0)) / max(np.log(max(serr0_e, 1.0 + 1e-6)), 1e-9), 0.0, 1.0))
  if u <= 0.02:
    return tuple(GT_RGBA[:3])  # snap: converged track IS the GT orange
  return tuple(viz.bourke_color_map(0.0, 1.0, _BOURKE_ORANGE_V * (1.0 - u)))
# Ribbon radii [m]. Judge ON THE ENCODED MP4 AT 1:1 IN MOTION -- zoomed still crops overstate visibility,
# and radii under ~5mm (1-2 px at the camera's ~200 px/m) are destroyed by h264 chroma subsampling. REF
# stays THINNER than ENV so at convergence the orange tucks fully INSIDE the colored tube and the ribbons
# read as one merged track; while trajectories differ, the orange reads as a bold second strand.
TRAIL_STEPS = int(os.environ.get("MJW_TRAIL_STEPS", 0))  # >0: draw only the last N sim steps of each tip
# trace (comet tail) instead of the full-history path; 0 = full history (the accepted look)
RIBBON_W_ENV = 0.010  # env tip trace (optimization-colormap color)
RIBBON_W_REF = 0.006  # GT reference: THINNER than the env tube, drawn with REF_LIFT so it rides as an
# orange STRIPE on top of the colored ribbon when the tracks merge -- reference and progress-hue both stay
# visible at every iteration; at x1.0 the stripe blends into the (now orange) env ribbon
RIBBON_W_GT = 0.0085  # GT arm's own tip trace in its own cell (orange)
REF_LIFT = 0.012  # [m] the env-cell GT reference is drawn OFFSET ALONG THE CAMERA RAY (toward the eye):
# zero pixels of on-screen shift (NOT a spatial offset -- merge semantics intact), but it wins the depth
# test against the coincident env tube, so the orange reads as a stripe ON TOP of the colored ribbon at
# every iteration (user request: the reference must stay visible even after the trajectories merge).


# ========================================================================================================
# excitation
# ========================================================================================================


def prep_excitation(m):
  """Configure the sys-id controller mode: the arm's own position actuators, kp/kv scaled to the compliant
  tracking mode, and the per-joint actuator-force clamp LIFTED (saturating the +-87/+-12 N*m clamps is a
  non-smooth event that corrupts both the fit landscape and the gradient; lifting it is a controller /
  experimental choice -- the robot's mass/inertia/geometry are untouched)."""
  m.actuator_gainprm[:, 0] *= KP_SCALE  # kp (gain on ctrl)
  m.actuator_biasprm[:, 1] *= KP_SCALE  # -kp (bias on qpos)
  m.actuator_biasprm[:, 2] *= KV_SCALE  # -kv (velocity-feedback gain)
  m.jnt_actfrclimited[:] = 0  # lift the per-joint actuator-force clamp (actuatorfrcrange)


def generate_chirp_trajectory(
  n_joints, duration, dt, f0, f1, amplitudes, ramp_up=2.0, ramp_down=3.0, pad_start=0.0, pad_end=0.0,
  taper_pivot=None, taper_pow=1.0,
):
  """Joint-space linear chirp for system identification.

  Args:
      n_joints:   number of joints
      duration:   total sweep duration [s]
      dt:         timestep [s]
      f0, f1:     start / end sweep frequencies [Hz]
      amplitudes: (n_joints,) per-joint amplitude [rad]
      ramp_up:    envelope ramp-up time [s]
      ramp_down:  envelope ramp-down time [s]
      pad_start:  hold-still time before sweep [s]
      pad_end:    hold-still time after sweep [s]
      taper_pivot: optional [Hz] -- amplitude TAPER: full amplitude while the instantaneous frequency is
                  below the pivot, then scaled by (pivot / f)^taper_pow. taper_pow=1 keeps the peak joint
                  VELOCITY (A*2pi*f) constant across the sweep -> big slow sways early, small fast wiggle
                  late, never violent. None = constant amplitude.
      taper_pow:  taper exponent (1 = constant velocity, 2 = constant acceleration amplitude)

  Returns:
      q_offsets: (T, n_joints) joint angle offsets from home config
      t:         (T,) time vector
  """
  T = int(duration / dt)
  t = np.linspace(0, duration, T)

  phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2)

  envelope = np.ones(T)
  n_up = min(int(ramp_up / dt), T // 2)
  n_dn = min(int(ramp_down / dt), T // 2)
  envelope[:n_up] = 0.5 * (1 - np.cos(np.pi * np.arange(n_up) / n_up))
  envelope[-n_dn:] = 0.5 * (1 - np.cos(np.pi * np.arange(n_dn, 0, -1) / n_dn))
  if taper_pivot is not None:
    f_inst = f0 + (f1 - f0) * t / duration  # linear-chirp instantaneous frequency
    envelope = envelope * (taper_pivot / np.maximum(f_inst, taper_pivot)) ** taper_pow

  phase_offsets = np.linspace(0, 2 * np.pi, n_joints, endpoint=False)
  amplitudes = np.asarray(amplitudes)

  q_offsets = np.zeros((T, n_joints))
  for i in range(n_joints):
    q_offsets[:, i] = amplitudes[i] * envelope * np.sin(phase + phase_offsets[i])

  n_pad_start = int(pad_start / dt)
  n_pad_end = int(pad_end / dt)
  if n_pad_start > 0 or n_pad_end > 0:
    q_offsets = np.concatenate(
      [np.zeros((n_pad_start, n_joints)), q_offsets, np.zeros((n_pad_end, n_joints))]
    )
    t = np.arange(len(q_offsets)) * dt

  return q_offsets, t


def chirp_ctrl(home_ctrl, dt, variant=0, **overrides):
  """Position-target chirp: ctrl (T, NARM) = home + the joint-space chirp offsets. `variant>0` rolls the
  per-joint phase assignment (a decorrelated pattern for NEXC>1). `overrides` patch CHIRP_KWARGS for
  one-off probes (e.g. f1=8.0) without mutating the module config."""
  kw = {**CHIRP_KWARGS, **overrides}
  amps = np.asarray(kw.pop("jnt_amp"), dtype=np.float64)
  offsets, _t = generate_chirp_trajectory(NARM, CHIRP_DURATION, dt, kw.pop("f0"), kw.pop("f1"), amps, **kw)
  if variant:
    offsets = offsets[:, np.roll(np.arange(NARM), 3 * variant)]  # permute joint<->phase pairing
  return (home_ctrl[None, :NARM] + offsets).astype(np.float32)


# ========================================================================================================
# inertia parameterization
# ========================================================================================================


@wp.kernel
def _theta_to_body_inertia(theta: wp.array(dtype=float), np_: int, nlink: int,
                           body_param_idx: wp.array(dtype=int), gt: wp.array(dtype=wp.vec3),
                           out: wp.array2d(dtype=wp.vec3)):
  """DIFFERENTIABLE reduced-param -> body_inertia map (recorded on the tape). np_==1 (MODE=scale):
  I = exp(theta)*I_gt (positivity + triangle inequality inherited from I_gt for any theta). np_==3
  (MODE=full): sigma_i = exp(theta_i), I_i = sigma_j + sigma_k (triangle-safe). Non-identified bodies copy
  their GT inertia (constant wrt theta). theta is flat (E*nlink*np_,) so warp's 1-D Adam optimizes it."""
  e, b = wp.tid()  # env, body
  li = body_param_idx[b]
  if li >= 0:
    base = e * nlink * np_ + li * np_
    if np_ == 1:
      out[e, b] = wp.exp(theta[base]) * gt[b]
    else:
      s0 = wp.exp(theta[base + 0])
      s1 = wp.exp(theta[base + 1])
      s2 = wp.exp(theta[base + 2])
      out[e, b] = wp.vec3(s1 + s2, s0 + s2, s0 + s1)
  else:
    out[e, b] = gt[b]


def link_bids(mjm):
  """Body ids of the NLINK identified links."""
  return [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name) for name in LINKS]


def gt_theta(mjm, bids):
  """The recovery target: theta_gt (NLINK, NP) and the GT principal moments (NLINK, 3)."""
  pm = np.array([mjm.body_inertia[b].copy() for b in bids])
  if MODE == "scale":
    return np.zeros((len(bids), 1)), pm
  theta = np.array([ip.principal_to_logsigma(pm[i]) for i in range(len(bids))])
  return theta, pm


def theta_to_principal(theta_link, pm_gt_link):
  """One link's params -> its principal moments (the host mirror of _theta_to_body_inertia)."""
  if MODE == "scale":
    return np.exp(theta_link[0]) * pm_gt_link
  return ip.logsigma_to_principal(theta_link)


def _gt_inertia_w(mjm):
  """GT per-body principal moments as an (nbody,) warp vec3 array (what non-identified bodies copy)."""
  return wp.array(mjw.put_model(mjm).body_inertia.numpy()[0], dtype=wp.vec3)


def _body_param_idx(mjm, bids):
  """(nbody,) int array mapping each body to its param slot (0..NLINK-1) or -1 (not identified)."""
  idx = np.full(mjm.nbody, -1, dtype=np.int32)
  for li, b in enumerate(bids):
    idx[b] = li
  return wp.array(idx, dtype=int)


def _draw_theta(theta_gt, rng):
  """One env's perturbation: RANDOM direction per link, per-link magnitude uniform in [0.5, 1.0] x PERTURB
  (envs open with visibly DIFFERENT wrongness instead of an identical fixed magnitude)."""
  g = rng.normal(size=(NLINK, NP))
  mag = rng.uniform(0.5, 1.0, size=(NLINK, 1)) * PERTURB * np.sqrt(NP)
  g *= mag / np.linalg.norm(g, axis=1, keepdims=True)
  return theta_gt + g


def init_theta(theta_gt, pm_gt, rng, mjd, bids, ctrl, gt_qpos):
  """Per-env initial params (E, NLINK, NP): random-direction perturbations, REDRAWN until each env's
  free-run trajectory visibly diverges from GT (max|dq| >= DIVERGE_MIN). If no draw clears the bar, keep
  the most divergent one seen -- the strongest observable direction the stream offered."""
  rm = mujoco.MjModel.from_xml_path(SCENE)
  prep_excitation(rm)
  n = len(ctrl)
  theta0 = np.empty((NUM_ENVS, NLINK, NP))
  for e in range(NUM_ENVS):
    best_dq, best_cand = -1.0, None
    for attempt in range(24):
      cand = _draw_theta(theta_gt, rng)
      q, _ = _mjc_replay(rm, mjd, bids, [theta_to_principal(cand[li], pm_gt[li]) for li in range(NLINK)], ctrl, n)
      dq = np.abs(q[:, :NARM] - gt_qpos[: n + 1, :NARM]).max()
      if dq > best_dq:
        best_dq, best_cand = dq, cand
      if dq >= DIVERGE_MIN:
        break
    theta0[e] = best_cand
    print(f"[fr3] env{e}: init free-run max|dq|={best_dq:.3f} rad  ({attempt + 1} draw{'s' if attempt else ''})")
  return theta0


# ========================================================================================================
# differentiable fit
# ========================================================================================================


@wp.kernel
def _pose_loss(qpos: wp.array2d[float], tgt: wp.array2d[float], t: int, narm: int, loss: wp.array(dtype=float)):
  e, j = wp.tid()
  d = qpos[e, j] - tgt[t, j]
  wp.atomic_add(loss, 0, d * d)


def _forward_gt(mjm, mjd, ctrl, n):
  """GT trajectory: mjw forward at the TRUE (base-model) inertias, single world -> qpos (n+1, nq)."""
  m = mjw.put_model(mjm)
  d_in, d_out = mjw.put_data(mjm, mjd, nworld=1), mjw.put_data(mjm, mjd, nworld=1)
  qs = [d_in.qpos.numpy()[0].copy()]
  for t in range(n):
    d_in.ctrl = wp.array(ctrl[t].reshape(1, -1).astype(np.float32), dtype=float)
    mjw.step(m, d_in, d_out)
    qs.append(d_out.qpos.numpy()[0].copy())
    d_in, d_out = d_out, d_in
  return np.array(qs)


def run(mjm, mjd, body_pidx, theta_w, gt_w, ctrls, tgts_qpos, n, want_grad=True):
  """OPEN-LOOP trajectory fit with truncated BPTT, batched (nworld=E) over all excitations. Per excitation
  the state integrates CONTINUOUSLY from home through the entire chirp under the env's wrong inertia (no
  state resets); the loss sums the tracking error vs GT over every step. The GRADIENT is truncated: the
  rollout is recorded as n/WINDOW consecutive segments, each on its own tape; a segment back-propagates its
  own loss to theta (through the param kernel re-recorded on that tape) with the segment's INCOMING state
  detached (the wp.copy handoff happens outside any tape). Per-segment theta grads accumulate host-side
  into theta_w.grad. Returns per_env_loss (E,), summed over excitations."""
  E = theta_w.shape[0] // (NLINK * NP)
  R = len(ctrls)
  m = mjw.put_model(mjm)
  nbody = m.nbody
  assert n % WINDOW == 0, f"NF={n} must be a multiple of WINDOW={WINDOW}"
  nseg = n // WINDOW
  body_inertia = wp.zeros((E, nbody), dtype=wp.vec3, requires_grad=want_grad)
  tgt_ws = [wp.array(tgts_qpos[r].astype(np.float32), dtype=float) for r in range(R)]
  # ONE reusable segment chain (WINDOW+1 states): segment s's end state is copied (DETACHED) into slot 0
  d_home = mjw.put_data(mjm, mjd, nworld=E)  # pristine home state (each excitation restarts from it)
  datas = [mjw.put_data(mjm, mjd, nworld=E) for _ in range(WINDOW + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  ctrl_ws = [[wp.array(np.tile(ctrls[r][t], (E, 1)).astype(np.float32), dtype=float) for t in range(n)] for r in range(R)]
  loss_seg = wp.zeros(1, dtype=float, requires_grad=True)
  g_acc = np.zeros(theta_w.shape[0], dtype=np.float64) if want_grad else None
  per_env_loss = np.zeros(E)
  for r in range(R):
    wp.copy(datas[0].qpos, d_home.qpos)  # open-loop restart at home for this excitation (outside any tape)
    wp.copy(datas[0].qvel, d_home.qvel)
    for s in range(nseg):
      if want_grad:  # fresh adjoint state per segment tape (stale .grad values would pollute the reverse)
        theta_w.grad.zero_()
        body_inertia.grad.zero_()
        loss_seg.grad.zero_()
        for d in datas:
          d.qpos.grad.zero_()
          d.qvel.grad.zero_()
      loss_seg.zero_()
      tape = wp.Tape()
      with tape:
        wp.launch(_theta_to_body_inertia, dim=(E, nbody), inputs=[theta_w, NP, NLINK, body_pidx, gt_w],
                  outputs=[body_inertia])
        m.body_inertia = body_inertia  # on THIS tape -> its adjoint chains body_inertia.grad into theta.grad
        for j in range(WINDOW):
          t = s * WINDOW + j  # global step index within this excitation
          datas[j].ctrl = ctrl_ws[r][t]
          mjw.step(m, datas[j], datas[j + 1])
          wp.launch(_pose_loss, dim=(E, NARM), inputs=[datas[j + 1].qpos, tgt_ws[r], t + 1, NARM],
                    outputs=[loss_seg])
      sim = np.array([datas[j].qpos.numpy() for j in range(1, WINDOW + 1)]).transpose(1, 0, 2)  # (E,W,nq)
      ref = tgts_qpos[r][s * WINDOW + 1 : (s + 1) * WINDOW + 1, :NARM][None]
      per_env_loss += ((sim[:, :, :NARM] - ref) ** 2).sum(axis=(1, 2))
      if want_grad:
        tape.backward(loss=loss_seg)
        g_acc += theta_w.grad.numpy().astype(np.float64)  # accumulate this segment's truncated grad
      # DETACHED handoff: the next segment continues from this end state, outside any tape
      wp.copy(datas[0].qpos, datas[WINDOW].qpos)
      wp.copy(datas[0].qvel, datas[WINDOW].qvel)
  if want_grad:
    theta_w.grad = wp.array(g_acc.astype(np.float32), dtype=float)
  return per_env_loss


def _per_link_err(theta_np, theta_gt, pm_gt):
  """Per-link ||I - I_gt|| -> (NLINK,) principal-moment error vector."""
  return np.array([np.linalg.norm(theta_to_principal(theta_np[i], pm_gt[i]) - theta_to_principal(theta_gt[i], pm_gt[i]))
                   for i in range(NLINK)])


def optimize(mjm, mjd, body_pidx, theta0, theta_gt, pm_gt, gt_w, ctrls, tgts_qpos, n, steps=STEPS, lr=LR):
  """warp.optim.Adam recovery of the per-env params, optimized DIRECTLY (theta.grad comes from the tape).
  Params are flat (E*NLINK*NP,) for warp's 1-D Adam kernel. Returns (history, best-iteration index);
  history[k]["theta"] is the pre-step theta at iteration k."""
  E = len(theta0)
  norm = np.sqrt(len(ctrls) * n * NARM)
  theta0_flat = theta0.reshape(-1).astype(np.float64)
  theta_w = wp.array(theta0.reshape(-1).astype(np.float32), dtype=float, requires_grad=True)
  opt = warp.optim.Adam([theta_w], lr=lr, betas=BETAS)
  history = []
  for it in range(steps):
    theta = theta_w.numpy().reshape(E, NLINK, NP).copy()
    per_env_loss = run(mjm, mjd, body_pidx, theta_w, gt_w, ctrls, tgts_qpos, n)
    terr = np.array([_per_link_err(theta[e], theta_gt, pm_gt).sum() for e in range(E)])  # total over links
    # per-env SCALE error (geometric mean over links of the per-link scale-factor error): x1.0 = perfect
    serr = np.exp(np.abs(theta - theta_gt[None]).mean(axis=(1, 2)))
    history.append({"it": it, "theta": theta.copy(), "rmse": np.sqrt(per_env_loss) / norm, "terr": terr,
                    "serr": serr})
    if it % 16 == 0 or it == steps - 1:
      print(f"  [{it:3d}] RMSE={np.array2string(np.sqrt(per_env_loss)/norm, precision=4)}  "
            f"total|dI|={np.array2string(terr, precision=3)}  scale x{np.array2string(serr, precision=2)}")
    # Tikhonov toward the initial guess: anchors the structurally-unidentifiable directions (see WD)
    g = theta_w.grad.numpy().astype(np.float64) + WD * (theta.reshape(-1) - theta0_flat)
    theta_w.grad = wp.array(g.astype(np.float32), dtype=float)
    opt.step([theta_w.grad])
  best = int(np.argmin([h["rmse"].mean() for h in history]))
  np.savez(HISTORY_NPZ, best=best, **{k: np.array([h[k] for h in history]) for k in ("it", "theta", "rmse", "terr", "serr")})
  print(f"[fr3] saved optimization history -> {HISTORY_NPZ}  (re-render: MJW_RENDER_FROM={HISTORY_NPZ})")
  hb = history[best]
  per_link0 = np.mean([_per_link_err(history[0]["theta"][e], theta_gt, pm_gt) for e in range(E)], axis=0)
  per_linkB = np.mean([_per_link_err(hb["theta"][e], theta_gt, pm_gt) for e in range(E)], axis=0)
  print(f"[fr3 inertia sysid] {E} envs, {NLINK} links ({LINKS[0]}..{LINKS[-1]}): recovered principal moments")
  print(f"  total |I-I_gt| per env  {np.array2string(history[0]['terr'],precision=3)} -> {np.array2string(hb['terr'],precision=4)}")
  print(f"  mean per-link |I-I_gt|  {np.array2string(per_link0,precision=4)}\n"
        f"                       -> {np.array2string(per_linkB,precision=5)}")
  print(f"  track RMSE {history[0]['rmse'].mean():.4f} -> {hb['rmse'].mean():.4f}")
  return history, best


def fd_check(mjm, mjd, body_pidx, theta0, gt_w, ctrls, gts_qpos, nf):
  """One-time analytic-vs-FD gate on dL/dtheta at the init points. Returns (per_env_loss, cos, ratio).
  Expect cos ~= 1 (direction exact); ratio < 1 is the DELIBERATE truncation bias of TBPTT, not an error."""
  P = NLINK * NP
  theta_w = wp.array(theta0.reshape(-1).astype(np.float32), dtype=float, requires_grad=True)
  per_env_loss = run(mjm, mjd, body_pidx, theta_w, gt_w, ctrls, gts_qpos, nf)
  dtheta = theta_w.grad.numpy().reshape(NUM_ENVS, P)
  theta0_flat = theta0.reshape(NUM_ENVS, P)
  eps = 5e-3
  fd = np.zeros((NUM_ENVS, P))
  for k in range(P):
    for sgn in (+1, -1):
      tp = theta0_flat.copy()
      tp[:, k] += sgn * eps
      tw = wp.array(tp.reshape(-1).astype(np.float32), dtype=float)
      lp = run(mjm, mjd, body_pidx, tw, gt_w, ctrls, gts_qpos, nf, want_grad=False)
      fd[:, k] += sgn * lp / (2 * eps)
  cos = np.array([float(dtheta[e] @ fd[e] / (np.linalg.norm(dtheta[e]) * np.linalg.norm(fd[e]) + 1e-30))
                  for e in range(NUM_ENVS)])
  ratio = np.array([np.linalg.norm(dtheta[e]) / (np.linalg.norm(fd[e]) + 1e-30) for e in range(NUM_ENVS)])
  return per_env_loss, cos, ratio


# ========================================================================================================
# render: 2x2 grid of sys-id arms (native FR3 look) + ONE separate GT reference arm (solid orange).
# Per-cell comparison = tip-trace ribbons at their TRUE positions: the env's trace (thick, colored) + the
# GT reference (thinner, orange). Wrong inertia -> two separated strands; at x1.0 the colored strand lands
# exactly on the GT path and the ribbons MERGE into one track (the orange tucks inside the colored tube).
# ========================================================================================================


def _row_axes():
  """Camera-aligned unit vectors in world xy: (screen-right, camera-forward) for CAM_AZIMUTH."""
  a = np.radians(CAM_AZIMUTH)
  f = np.array([np.cos(a), np.sin(a)])  # camera forward: points AWAY from the camera
  return np.array([f[1], -f[0]]), f     # screen-right = forward x up

def _grid_offsets(e, n=NUM_ENVS):
  """Env arm e's world-xy offset: the front row, ordered left->right on screen."""
  r, f = _row_axes()
  return ROW_CENTER + (e - (n - 1) / 2) * ROW_SPACING * r - 0.5 * ROW_SEP * f

def _gt_offset():
  """The GT reference arm: alone in the back row, centered."""
  _, f = _row_axes()
  return ROW_CENTER + 0.5 * ROW_SEP * f


def _add_robot(scene, vm, d, qpos, opt, pert, offset, rgba=None):
  """Append the arm's geoms at `qpos`, TRANSLATED by `offset` into the env's grid cell, hide the floor.
  rgba=None keeps the asset's NATIVE materials (the FR3 look); a color (the GT arm's orange) recolors."""
  d.qpos[:] = qpos
  mujoco.mj_forward(vm, d)
  n0 = scene.ngeom
  cat = int(mujoco.mjtCatBit.mjCAT_STATIC) | int(mujoco.mjtCatBit.mjCAT_DYNAMIC)
  mujoco.mjv_addGeoms(vm, d, opt, pert, cat, scene)
  for i in range(n0, scene.ngeom):
    if scene.geoms[i].type == int(mujoco.mjtGeom.mjGEOM_PLANE):
      scene.geoms[i].rgba[3] = 0.0
      continue
    scene.geoms[i].pos[0] += offset[0]
    scene.geoms[i].pos[1] += offset[1]
    if rgba is not None:
      scene.geoms[i].rgba = rgba


def _mjc_replay(rm, mjd, bids, inertia_links, ctrl, n):
  """MuJoCo-C forward at the given per-link principal moments (set on rm in place; orientations stay GT)
  under the position-chirp targets -> (qpos (n+1, nq), tip (n+1, 3)), tip = attachment_site world path."""
  for li, b in enumerate(bids):
    rm.body_inertia[b] = inertia_links[li]
  sid = mujoco.mj_name2id(rm, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
  d = mujoco.MjData(rm)
  d.qpos[:], d.qvel[:] = mjd.qpos, mjd.qvel
  mujoco.mj_forward(rm, d)
  qs = [d.qpos.copy()]
  tips = [d.site_xpos[sid].copy()]
  for t in range(n):
    d.ctrl[:] = ctrl[t]
    mujoco.mj_step(rm, d)
    mujoco.mj_kinematics(rm, d)  # mj_step leaves site_xpos at the PRE-integration state; refresh so the
    qs.append(d.qpos.copy())     # trace ends AT the flange (the lag is up to ~17mm at tip speed)
    tips.append(d.site_xpos[sid].copy())
  return np.array(qs), np.array(tips)


def _banner(h):
  """Per-iteration banner subtext (the friction-demo-style multiplier readout: x1.0 = perfect sys-id)."""
  if MODE == "scale" and "serr" in h:
    s_s = "[" + ",".join(f"{v:.2f}" for v in h["serr"]) + "]"
    return f"iter {h['it']:3d}   inertia x{s_s}   (true x1.0)"
  terr_s = "[" + ",".join(f"{v:.3f}" for v in h["terr"]) + "]"  # fixed-point keeps the banner in frame
  return f"iter {h['it']:3d}   total |I-I_gt| {terr_s}   (true 0)"


def render(mjm, mjd, bids, ctrl_full, gt_qpos, history, best, out_path=OUT_MP4):
  """Video of shown iterations: every arm is replayed in MuJoCo-C at that iteration's inertias (history
  stores the pre-step theta) and drawn as offset geoms; ribbons per the section note above."""
  vm = mujoco.MjModel.from_xml_path(SCENE)
  prep_excitation(vm)
  vm.vis.global_.offwidth, vm.vis.global_.offheight = W, H
  vm.vis.global_.fovy = CAM_FOVY
  vm.vis.global_.orthographic = CAM_ORTHO
  vm.stat.extent = 3.5
  vm.stat.center = np.array([0.1, 0.0, 0.4])
  arm_root = mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_BODY, "base")  # fr3's root body
  if arm_root >= 0:
    vm.body_pos[arm_root] = [0.0, 0.0, -50.0]  # hide the model's OWN arm; every arm is an added, offset geom
  rm = mujoco.MjModel.from_xml_path(SCENE)
  prep_excitation(rm)
  vd, gd = mujoco.MjData(vm), mujoco.MjData(rm)
  opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
  E = history[0]["theta"].shape[0]
  offs = [_grid_offsets(e, E) for e in range(E)]
  gt_off = _gt_offset()
  gt_off3 = np.array([gt_off[0], gt_off[1], 0.0])
  rn = len(ctrl_full)

  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(vm, cam)
  cam.lookat = list(CAM_LOOKAT)
  cam.distance = CAM_DISTANCE
  cam.azimuth = CAM_AZIMUTH
  cam.elevation = CAM_ELEVATION

  el, az = np.radians(CAM_ELEVATION), np.radians(CAM_AZIMUTH)
  ref_lift = -REF_LIFT * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
  # ^ toward the camera eye (rays are near-parallel at the tele distance, so one direction serves the scene)

  parked = mjd.qpos.copy()  # pose for the hidden native arm (vd); all visible arms are decorative geoms
  show = sorted(set([k for k in [0, max(1, best // 4), max(2, best // 2), 3 * best // 4] if k < best] + [best]))
  # GT replay ONCE, and capture pm_gt BEFORE any replay mutates rm (rm is freshly loaded here, so it still
  # carries the TRUE inertias). gt_tip is the orange reference ribbon, drawn inside every env's cell.
  pm_gt = np.array([rm.body_inertia[b].copy() for b in bids])
  _, gt_tip = _mjc_replay(rm, mjd, bids, list(pm_gt), ctrl_full, rn)

  serr0 = np.asarray(history[0]["serr"], dtype=np.float64)  # per-env initial wrongness (colormap anchor)
  frames = []
  for k in show:
    h = history[k]
    sub = _banner(h)
    env_rgb = [_env_ribbon_rgb(float(h["serr"][e]), serr0[e]) for e in range(E)]
    replays = [_mjc_replay(rm, mjd, bids, [theta_to_principal(h["theta"][e][li], pm_gt[li]) for li in range(NLINK)],
                           ctrl_full, rn) for e in range(E)]
    env_q = [r[0] for r in replays]
    env_tip = [r[1] for r in replays]
    hold = 20 if k == best else 0
    idx = list(range(0, rn + 1, RENDER_STRIDE)) + [rn] * hold
    for t in idx:
      sim_poses = [(env_q[e][t], offs[e]) for e in range(E)]
      ghost_q = gt_qpos[t]

      t0 = max(0, t - TRAIL_STEPS) if TRAIL_STEPS > 0 else 0
      t0 -= t0 % 2  # keep the subsample phase fixed as the window slides (no frame-to-frame shimmer)
      trail = slice(t0, t + 1, 2)

      # NOTE: every loop-varying object the closure uses MUST be bound as a default argument. draw() runs
      # lazily inside viz.emit AFTER this loop has finished, so a free variable (late binding) would make
      # every frame use the LAST shown iteration's data -- this exact bug shipped: env_tip was free, so
      # all blocks drew the CONVERGED ribbons (colored tube on the GT path, orange swallowed everywhere).
      def draw(scene, sim_poses=sim_poses, ghost_q=ghost_q, trail=trail, env_tip=env_tip, gt_tip=gt_tip,
               env_rgb=env_rgb):
        for e, (sim_q, off) in enumerate(sim_poses):
          _add_robot(scene, rm, gd, sim_q, opt, pert, off)  # native FR3 materials
          off3 = np.array([off[0], off[1], 0.0])
          viz.add_polyline(scene, (env_tip[e][trail] + off3), env_rgb[e], width=RIBBON_W_ENV)
          viz.add_polyline(scene, (gt_tip[trail] + off3 + ref_lift), GT_RGBA[:3], width=RIBBON_W_REF)
        _add_robot(scene, rm, gd, ghost_q, opt, pert, gt_off, GT_RGBA)
        viz.add_polyline(scene, (gt_tip[trail] + gt_off3), GT_RGBA[:3], width=RIBBON_W_GT)

      frames.append((parked, draw, f"{sub}   [t={t}/{rn}]"))
  if frames:
    frames += [frames[-1]] * 20
  return viz.emit(vm, vd, cam, frames, out_path=out_path, label="ADJOINT (Franka inertia)", w=W, h=H, fps=FPS)


# ========================================================================================================


def main():
  if not os.path.exists(SCENE):
    raise SystemExit(f"run from the mujoco_warp repo root (missing {SCENE})")
  mjm = mujoco.MjModel.from_xml_path(SCENE)
  prep_excitation(mjm)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
  mujoco.mj_forward(mjm, mjd)
  bids = link_bids(mjm)
  if any(b < 0 for b in bids):
    raise SystemExit(f"missing a body in {LINKS}")

  theta_gt, gt_inertia = gt_theta(mjm, bids)  # (NLINK, NP)
  gt_w = _gt_inertia_w(mjm)
  body_pidx = _body_param_idx(mjm, bids)

  home_ctrl = mjm.key_ctrl[0].copy() if mjm.nkey else np.zeros(mjm.nu)
  ctrls = [chirp_ctrl(home_ctrl, mjm.opt.timestep, variant=r) for r in range(NEXC)]
  nf = (len(ctrls[0]) // WINDOW) * WINDOW  # fit the whole chirp, trimmed to a TBPTT-window multiple
  ctrls = [c[:nf] for c in ctrls]

  if from_npz := os.environ.get("MJW_RENDER_FROM"):  # render-only: replay a saved optimization history
    z = np.load(from_npz)
    history = [{k: z[k][i] for k in ("it", "theta", "rmse", "terr", "serr")} for i in range(len(z["it"]))]
    gt_qpos = _forward_gt(mjm, mjd, ctrls[0], nf)
    render(mjm, mjd, bids, ctrls[0], gt_qpos, history, int(z["best"]))
    return

  gts_qpos = [_forward_gt(mjm, mjd, c, nf) for c in ctrls]  # GT trajectory per excitation

  rng = np.random.default_rng(SEED)
  theta0 = init_theta(theta_gt, gt_inertia, rng, mjd, bids, ctrls[0], gts_qpos[0])  # visibly-divergent init

  per_env_loss, cos, ratio = fd_check(mjm, mjd, body_pidx, theta0, gt_w, ctrls, gts_qpos, nf)
  norm = np.sqrt(NEXC * nf * NARM)
  span = gts_qpos[0][:, :NARM].max(0) - gts_qpos[0][:, :NARM].min(0)
  print(f"[fr3] nf={nf} ({CHIRP_DURATION}s chirp {CHIRP_KWARGS['f0']}-{CHIRP_KWARGS['f1']}Hz, position PD, "
        f"gravity on)  {NUM_ENVS} envs  {NEXC} excitation(s)  {NLINK} links")
  print(f"[fr3] joint oscillation (p2p, rad): {np.array2string(span, precision=3)}")
  print(f"[fr3] init windowed-tracking RMSE per env (TBPTT window={WINDOW}): "
        f"{np.array2string(np.sqrt(per_env_loss)/norm, precision=4)} rad")
  print(f"[fr3] dL/dtheta analytic-vs-FD ({NLINK * NP}-dim): cos {np.array2string(cos, precision=3)}  "
        f"ratio {np.array2string(ratio, precision=3)}")

  history, best = optimize(mjm, mjd, body_pidx, theta0, theta_gt, gt_inertia, gt_w, ctrls, gts_qpos, nf)
  if os.environ.get("MJW_NO_RENDER") != "1":
    os.makedirs(os.path.dirname(out_path := OUT_MP4), exist_ok=True)
    render(mjm, mjd, bids, ctrls[0], gts_qpos[0], history, best, out_path=out_path)  # video: excitation 0


if __name__ == "__main__":
  main()
