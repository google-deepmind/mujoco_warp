"""Franka FR3 link-inertia system-identification -- parallel envs, ANALYTIC adjoint gradient.

OPEN-LOOP trajectory fit: a joint-space position CHIRP (PACE-style) is driven through a COMPLIANT servo
so link inertia visibly shapes the low-frequency sway; we recover each arm link's principal moments of
inertia (MODEL parameter m.body_inertia) by matching the recorded trajectory. num_envs envs each start
from a different (wrong) inertia and recover toward the true values via warp.optim.Adam on the analytic
d(loss)/d(theta). theta is a reduced leaf (log-scale per link for MODE=scale) mapped to body_inertia by a
DIFFERENTIABLE kernel recorded on the tape; the gradient is TRUNCATED-BPTT (WINDOW-step segments) with the
theta grad accumulating across the checkpointed segments (demo.Example harness).

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/franka_sysid.py
"""

import os
import sys
from dataclasses import dataclass
from dataclasses import field

import mujoco
import numpy as np
import warp as wp
from absl import app

import mujoco_warp as mjw

sys.path.insert(0, os.path.dirname(__file__))
import demo  # noqa: E402  shared config + gradients + capture/reuse + optimize loop + main
import inertia_param as ip  # noqa: E402  principal-moment <-> log-sigma helpers (MODE=full)
import viz  # noqa: E402  shared renderer (mp4 or live MuJoCo viewer via MJW_VIEWER)

SCENE = "benchmarks/franka_fr3/scene_sysid.xml"  # arm-only FR3 (nq=7), contacts disabled in the XML
NARM = 7
NLINK = 6  # fr3_link1..fr3_link6; link7 is a near-massless flange (unidentifiable), excluded
LINKS = [f"fr3_link{i}" for i in range(1, NLINK + 1)]
MODE = "scale"  # "scale": one log-scale per link (I -> s*I; the demo default, fully identifiable)
NP = 1 if MODE == "scale" else 3
WINDOW = 40  # truncated-BPTT segment length [steps] (= the checkpoint segment)
PERTURB = 3.0  # per-link initial theta magnitude (e^{+-3.0} ~ 0.05x..20x -- bigger initial inertia error -> more sim-vs-ghost delta)
DIVERGE_MIN = 0.25  # rad: min free-run max|dq| vs GT an env's init must produce (else redraw)
WD = 0.0 if MODE == "scale" else 2e-3  # Tikhonov pull toward the initial theta (structurally-flat dirs)
BETAS = (0.7, 0.95)  # SHAC betas
SEED = 0
# CHIRP EXCITATION. LOW-frequency (f 0.15 -> 1.5 Hz), annealed low->high->low, velocity-constant taper
# (taper_pow=1: amp ~ 1/f -> big slow swings at low freq, small fast at high; peak joint speed ~flat = smooth,
# no whipping). ramp_up 0.7 s eases the big low-freq swing in (no startup servo yank); the wrist base is tamed to
# 0.12 (at 2.7x the other joints it whipped ~2x faster and caused a start jitter). Crucially the delta vs the GT
# ghost is driven NOT by raw frequency (higher f reads as "too fast") but by the SOFT servo (KP_SCALE 0.1) + a
# big initial inertia error (PERTURB 3.0): a softer compliant servo shifts the inertia-sensitive band DOWN so the
# wrong-inertia sim arm diverges from the ghost even at LOW freq (~2x the delta, ~0.13-0.16 rad, gentle <1 rad/s).
CHIRP_DURATION, AMP_SCALE = 5.0, 5.0
CHIRP_KWARGS = {"f0": 0.15, "f1": 1.5, "jnt_amp": list(AMP_SCALE * np.array([0.075] * 6 + [0.12])),
                "ramp_up": 0.7, "ramp_down": 0.6, "pad_start": 0.1, "pad_end": 0.1, "taper_pow": 1.0,
                "anneal": True}
KP_SCALE, KV_SCALE = 0.1, 0.02  # SOFT compliant servo (inertia shapes the sway; softer -> more sim-vs-ghost delta at low freq)
W, H, FPS = 1024, 768, 30
ASSETS = os.path.join(os.path.dirname(__file__), "reports", "assets")
# render layout / camera (a long-lens near-parallel view so identical trajectories look identical per cell)
ROW_CENTER, ROW_SPACING, ROW_SEP = np.array([0.35, 0.0]), 1.2, 1.6
CAM_LOOKAT, CAM_DISTANCE, CAM_FOVY, CAM_AZIMUTH, CAM_ELEVATION = [0.75, 0.0, 0.3], 12.0, 19.0, 110.0, -28.0
GT_RGBA = np.array([1.0, 0.5, 0.12, 1.0], dtype=np.float32)
RIBBON_W_ENV, RIBBON_W_REF, RIBBON_W_GT, REF_LIFT = 0.010, 0.006, 0.0085, 0.012
RENDER_STRIDE = 2


# ---- excitation + inertia parameterization (preserved helpers) ---------------------------------------

def prep_excitation(m):
  """Compliant sys-id controller: scale kp/kv, lift the actuator-force clamp (a non-smooth event), and REMOVE
  GRAVITY (opt.gravity=0). Without this the compliant (low-kp) arm SAGS under gravity at the start -- a shoulder
  (j1) droop transient that read as an 'initial jump' (gravity-off -> zero droop, confirmed). body_gravcomp=1
  does NOT work here (MuJoCo 3.10 leaves qfrc_gravcomp=0 at runtime), so we zero gravity instead -- mjw and mjc
  apply it identically, so the optimize and render agree. Zeroing gravity also ISOLATES the inertia response for
  the sys-id: gravity torque is a mass/COM effect that only confounds INERTIA identification. The robot's
  mass/inertia/geometry are untouched -- this is an experimental-setup / controller choice."""
  m.actuator_gainprm[:, 0] *= KP_SCALE
  m.actuator_biasprm[:, 1] *= KP_SCALE
  m.actuator_biasprm[:, 2] *= KV_SCALE
  m.jnt_actfrclimited[:] = 0
  m.opt.gravity[:] = 0.0


def generate_chirp_trajectory(n_joints, duration, dt, f0, f1, amplitudes, ramp_up=2.0, ramp_down=3.0,
                              pad_start=0.0, pad_end=0.0, taper_pow=0.0, anneal=False):
  """Joint-space chirp (PACE sys-id excitation): per-joint sinusoid whose instantaneous frequency sweeps,
  decoupled by even phase offsets, under a Hann ramp-in/out envelope. `anneal=True` sweeps frequency
  low->high->low (triangular: f0->f1 over the first half, f1->f0 over the second) instead of monotonic f0->f1,
  so the high-freq (high-acceleration -> big sim-vs-ghost delta) burst is in the MIDDLE, framed by big slow
  swings at both ends. `taper_pow`(>0) scales amplitude by (f0/f_inst)**taper_pow: HIGH amp at low freq (big
  slow swings), small at high freq (small fast oscillations); taper_pow=1 = velocity-constant (flat peak speed,
  no whipping). Phase is the numerical integral of f_inst so any frequency profile works."""
  T = int(duration / dt)
  t = np.linspace(0, duration, T)
  if anneal:
    h = T // 2
    f_inst = np.concatenate([np.linspace(f0, f1, h), np.linspace(f1, f0, T - h)])   # low -> high -> low
  else:
    f_inst = f0 + (f1 - f0) * t / duration                                          # monotonic low -> high
  phase = 2 * np.pi * np.cumsum(f_inst) * dt
  envelope = np.ones(T)
  n_up, n_dn = min(int(ramp_up / dt), T // 2), min(int(ramp_down / dt), T // 2)
  envelope[:n_up] = 0.5 * (1 - np.cos(np.pi * np.arange(n_up) / n_up))
  envelope[-n_dn:] = 0.5 * (1 - np.cos(np.pi * np.arange(n_dn, 0, -1) / n_dn))
  if taper_pow > 0.0:
    envelope = envelope * (f0 / f_inst) ** taper_pow   # amp ~ (f0/f)^p: big@low-freq -> small@high-freq
  phase_offsets = np.linspace(0, 2 * np.pi, n_joints, endpoint=False)
  amplitudes = np.asarray(amplitudes)
  q_offsets = np.zeros((T, n_joints))
  for i in range(n_joints):
    q_offsets[:, i] = amplitudes[i] * envelope * np.sin(phase + phase_offsets[i])
  n_ps, n_pe = int(pad_start / dt), int(pad_end / dt)
  if n_ps or n_pe:
    q_offsets = np.concatenate([np.zeros((n_ps, n_joints)), q_offsets, np.zeros((n_pe, n_joints))])
  return q_offsets


def chirp_ctrl(home_ctrl, dt, variant=0):
  kw = dict(CHIRP_KWARGS)
  amps = np.asarray(kw.pop("jnt_amp"), dtype=np.float64)
  offsets = generate_chirp_trajectory(NARM, CHIRP_DURATION, dt, kw.pop("f0"), kw.pop("f1"), amps, **kw)
  if variant:
    offsets = offsets[:, np.roll(np.arange(NARM), 3 * variant)]
  return (home_ctrl[None, :NARM] + offsets).astype(np.float32)


@wp.kernel
def _theta_to_body_inertia(theta: wp.array(dtype=float), np_: int, nlink: int,
                           body_param_idx: wp.array(dtype=int), gt: wp.array(dtype=wp.vec3),
                           out: wp.array2d(dtype=wp.vec3)):
  """DIFFERENTIABLE reduced-param -> body_inertia (recorded on the tape). np_==1: I = exp(theta)*I_gt.
  np_==3: sigma_i = exp(theta_i), I = (s1+s2, s0+s2, s0+s1). Non-identified bodies copy GT."""
  e, b = wp.tid()
  li = body_param_idx[b]
  if li >= 0:
    base = e * nlink * np_ + li * np_
    if np_ == 1:
      out[e, b] = wp.exp(theta[base]) * gt[b]
    else:
      s0, s1, s2 = wp.exp(theta[base + 0]), wp.exp(theta[base + 1]), wp.exp(theta[base + 2])
      out[e, b] = wp.vec3(s1 + s2, s0 + s2, s0 + s1)
  else:
    out[e, b] = gt[b]


def link_bids(mjm):
  return [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name) for name in LINKS]


def gt_theta(mjm, bids):
  pm = np.array([mjm.body_inertia[b].copy() for b in bids])
  if MODE == "scale":
    return np.zeros((len(bids), 1)), pm
  return np.array([ip.principal_to_logsigma(pm[i]) for i in range(len(bids))]), pm


def theta_to_principal(theta_link, pm_gt_link):
  if MODE == "scale":
    return np.exp(theta_link[0]) * pm_gt_link
  return ip.logsigma_to_principal(theta_link)


def _gt_inertia_w(mjm):
  return wp.array(mjw.put_model(mjm).body_inertia.numpy()[0], dtype=wp.vec3)


def _body_param_idx(mjm, bids):
  idx = np.full(mjm.nbody, -1, dtype=np.int32)
  for li, b in enumerate(bids):
    idx[b] = li
  return wp.array(idx, dtype=int)


def _mjc_replay(rm, mjd, bids, inertia_links, ctrl, n):
  """MuJoCo-C forward at the given per-link principal moments -> (qpos (n+1, nq), tip (n+1, 3))."""
  for li, b in enumerate(bids):
    rm.body_inertia[b] = inertia_links[li]
  sid = mujoco.mj_name2id(rm, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
  d = mujoco.MjData(rm)
  d.qpos[:], d.qvel[:] = mjd.qpos, mjd.qvel
  mujoco.mj_forward(rm, d)
  qs, tips = [d.qpos.copy()], [d.site_xpos[sid].copy()]
  for t in range(n):
    d.ctrl[:] = ctrl[t]
    mujoco.mj_step(rm, d)
    mujoco.mj_kinematics(rm, d)
    qs.append(d.qpos.copy())
    tips.append(d.site_xpos[sid].copy())
  return np.array(qs), np.array(tips)


def _draw_theta(theta_gt, rng):
  g = rng.normal(size=(NLINK, NP))
  mag = rng.uniform(0.5, 1.0, size=(NLINK, 1)) * PERTURB * np.sqrt(NP)
  g *= mag / np.linalg.norm(g, axis=1, keepdims=True)
  return theta_gt + g


def init_theta(theta_gt, pm_gt, rng, mjd, bids, ctrl, gt_qpos, num_envs):
  """Per-env init (E, NLINK, NP): random-direction perturbations, redrawn until each env's free-run
  visibly diverges from GT (max|dq| >= DIVERGE_MIN)."""
  rm = mujoco.MjModel.from_xml_path(SCENE)
  prep_excitation(rm)
  n = len(ctrl)
  theta0 = np.empty((num_envs, NLINK, NP))
  for e in range(num_envs):
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
  return theta0


@wp.kernel
def _pose_loss(qpos: wp.array2d[float], tgt: wp.array2d[float], t: int, narm: int, loss: wp.array(dtype=float)):
  e, j = wp.tid()
  d = qpos[e, j] - tgt[t, j]
  wp.atomic_add(loss, 0, d * d)


def _forward_gt(mjm, mjd, ctrl, n):
  """GT trajectory: mjw forward at the TRUE inertias, single world -> qpos (n+1, nq)."""
  m = mjw.put_model(mjm)
  d_in, d_out = mjw.put_data(mjm, mjd, nworld=1), mjw.put_data(mjm, mjd, nworld=1)
  qs = [d_in.qpos.numpy()[0].copy()]
  for t in range(n):
    d_in.ctrl = wp.array(ctrl[t].reshape(1, -1).astype(np.float32), dtype=float)
    mjw.step(m, d_in, d_out)
    qs.append(d_out.qpos.numpy()[0].copy())
    d_in, d_out = d_out, d_in
  return np.array(qs)


def _per_link_err(theta_np, theta_gt, pm_gt):
  return np.array([np.linalg.norm(theta_to_principal(theta_np[i], pm_gt[i]) - theta_to_principal(theta_gt[i], pm_gt[i]))
                   for i in range(NLINK)])


@dataclass
class Args(demo.CommonArgs):
  """fr3 inertia sys-id config: CommonArgs + the fit defaults (num_envs/steps/lr; chunk = WINDOW)."""

  num_envs: int = field(default=32, metadata={"help": "parallel envs, each a DIFFERENT init inertia -> distinct trajectories tiled across the field"})
  steps: int = field(default=160, metadata={"help": "Adam steps"})
  lr: float = field(default=0.03, metadata={"help": "Adam learning rate"})
  usd_stride: int = field(default=4, metadata={"help": "USD field: subsample (dt=0.005 -> stride 4 = 1x@50fps)"})
  usd_iters: int = field(default=5, metadata={"help": "USD field: optimization iterations sampled"})
  # dense hero field for the immersed render (render_blender.hero_cam): 17x17 arms at 1.6m spacing so the
  # pushed-in camera shows a big foreground hero arm + a field cropping off the frame edges.
  usd_envs: int = field(default=289, metadata={"help": "USD field: instanced lanes (17x17 dense field)"})
  usd_cols: int = field(default=17, metadata={"help": "USD field: grid columns"})
  usd_xpitch: float = field(default=1.6, metadata={"help": "USD field: column pitch (x) = arm spacing"})
  usd_ypitch: float = field(default=1.6, metadata={"help": "USD field: row pitch (y)"})


class FrankaSysidDemo(demo.Example):
  """FR3 link-inertia sys-id: optimize the reduced leaf theta (Adam) mapped to m.body_inertia by an
  ON-TAPE kernel (so theta.grad accumulates across the truncated-BPTT segments). Time-dependent per-step
  tracking loss vs the GT chirp trajectory; the excitation ctrl is a fixed per-step schedule."""

  Args = Args
  capturable = False  # fixed per-step chirp ctrl + host step index -> eager checkpointed path
  truncated = True    # truncated BPTT: detach the state adjoint every WINDOW steps

  def optimize(self):
    return self.optimize_adam(betas=BETAS)

  # ---- harness hooks ----

  def build_model(self):
    if not os.path.exists(SCENE):
      raise SystemExit(f"run from the mujoco_warp repo root (missing {SCENE})")
    self.args.chunk = WINDOW  # the checkpoint segment = one TBPTT window
    self.mjm = mujoco.MjModel.from_xml_path(SCENE)
    prep_excitation(self.mjm)
    self.mjm.vis.global_.offwidth, self.mjm.vis.global_.offheight = W, H
    self.mjd = mujoco.MjData(self.mjm)
    mujoco.mj_resetDataKeyframe(self.mjm, self.mjd, 0)
    mujoco.mj_forward(self.mjm, self.mjd)
    self.bids = link_bids(self.mjm)
    self.theta_gt, self.pm_gt = gt_theta(self.mjm, self.bids)  # (NLINK, NP), (NLINK, 3)
    home = self.mjm.key_ctrl[0].copy() if self.mjm.nkey else np.zeros(self.mjm.nu)
    ctrl = chirp_ctrl(home, self.mjm.opt.timestep)
    self.nf = (len(ctrl) // WINDOW) * WINDOW  # trim to a TBPTT-window multiple
    self.ctrl = ctrl[: self.nf]
    self.gt_qpos = _forward_gt(self.mjm, self.mjd, self.ctrl, self.nf)  # GT target trajectory (nf+1, nq)
    self.rng = np.random.default_rng(SEED)

  def init_params(self):
    theta0 = init_theta(self.theta_gt, self.pm_gt, self.rng, self.mjd, self.bids, self.ctrl, self.gt_qpos, self.args.num_envs)
    self.theta0_flat = theta0.reshape(-1).astype(np.float64)  # Tikhonov anchor
    return theta0  # (E, NLINK, NP)

  def build_datas(self):
    ne, nu = self.args.num_envs, self.mjm.nu
    self.nT = self.nf  # checkpointed BPTT length (chunk = WINDOW segment buffers)
    self.nbody = self.mjm.nbody
    self.datas = [mjw.put_data(self.mjm, self.mjd, nworld=ne) for _ in range(self.args.chunk + 1)]
    for d in self.datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
      d.ctrl.requires_grad = True  # bind_ctrl scatters the fixed chirp ctrl into this stable buffer
    self.datas[0].qpos = wp.array(np.tile(self.mjd.qpos, (ne, 1)).astype(np.float32), dtype=float, requires_grad=True)
    self.datas[0].qvel = wp.array(np.tile(self.mjd.qvel, (ne, 1)).astype(np.float32), dtype=float, requires_grad=True)
    self.param = wp.array(self.params.reshape(-1).astype(np.float32), dtype=float, requires_grad=True)  # theta (Adam leaf)
    self.accum_leaf = self.param  # theta grad accumulates across segments (reduction is ON the chunk tape)
    self.body_inertia = wp.zeros((ne, self.nbody), dtype=wp.vec3, requires_grad=True)  # reused intermediate
    self.gt_w = _gt_inertia_w(self.mjm)
    self.body_pidx = _body_param_idx(self.mjm, self.bids)
    self.tgt_wp = wp.array(self.gt_qpos.astype(np.float32), dtype=float)  # GT target (indexed by step)
    self.ctrl_flat = [wp.array(np.tile(self.ctrl[t], (ne, 1)).reshape(-1).astype(np.float32), dtype=float)
                      for t in range(self.nf)]  # flat (E*nu,) fixed chirp ctrl per step
    self.loss = wp.zeros(1, dtype=float, requires_grad=True)
    self._viz = [mjw.put_data(self.mjm, self.mjd, nworld=ne) for _ in range(2)]

  def set_params(self):
    pass  # theta -> body_inertia happens in chunk_prologue (ON the tape); datas[0] is the fixed home state

  def chunk_prologue(self):
    # DIFFERENTIABLE reduction re-recorded every chunk so its adjoint accumulates theta.grad
    wp.launch(_theta_to_body_inertia, dim=(self.args.num_envs, self.nbody),
              inputs=[self.param, NP, NLINK, self.body_pidx, self.gt_w], outputs=[self.body_inertia])
    self.m.body_inertia = self.body_inertia  # on THIS tape -> body_inertia.grad chains into theta.grad

  def chunk_step(self, i, t):
    self.bind_ctrl(i, self.ctrl_flat[t])  # scatter the fixed chirp ctrl (rebind is invisible under bc)
    mjw.step(self.m, self.datas[i], self.datas[i + 1])
    wp.launch(_pose_loss, dim=(self.args.num_envs, NARM), inputs=[self.datas[i + 1].qpos, self.tgt_wp, t + 1, NARM],
              outputs=[self.loss])

  def read_grad(self):
    return self.accum_grad + WD * (self.param.numpy().astype(np.float64) - self.theta0_flat)  # theta grad + Tikhonov

  def sim_qpos(self):
    """Full per-step trajectory (E, nf+1, nq) via a fresh forward-only rollout at the current theta."""
    ne, nu = self.args.num_envs, self.mjm.nu
    wp.launch(_theta_to_body_inertia, dim=(ne, self.nbody), inputs=[self.param, NP, NLINK, self.body_pidx, self.gt_w],
              outputs=[self.body_inertia])
    self.m.body_inertia = self.body_inertia
    a, b = self._viz
    a.qpos = wp.array(np.tile(self.mjd.qpos, (ne, 1)).astype(np.float32), dtype=float)
    a.qvel = wp.array(np.tile(self.mjd.qvel, (ne, 1)).astype(np.float32), dtype=float)
    qs = [np.tile(self.mjd.qpos, (ne, 1)).astype(np.float32)]
    for t in range(self.nf):
      wp.launch(demo._scatter_ctrl, dim=(ne, nu), inputs=[self.ctrl_flat[t], nu], outputs=[a.ctrl])
      mjw.step(self.m, a, b)
      qs.append(b.qpos.numpy().copy())  # b = post-step state
      a, b = b, a
    return np.array(qs).transpose(1, 0, 2)  # (E, nf+1, nq)

  def record(self, it, loss, qpos, pnp, g):
    E = self.args.num_envs
    theta = pnp.reshape(E, NLINK, NP)
    norm = np.sqrt(self.nf * NARM)
    per_env = ((qpos[:, 1:, :NARM] - self.gt_qpos[1:self.nf + 1, :NARM][None]) ** 2).sum(axis=(1, 2))
    terr = np.array([_per_link_err(theta[e], self.theta_gt, self.pm_gt).sum() for e in range(E)])
    serr = np.exp(np.abs(theta - self.theta_gt[None]).mean(axis=(1, 2)))  # per-env geometric scale error
    return {"it": it, "loss": loss, "theta": theta.copy(), "rmse": np.sqrt(per_env) / norm, "terr": terr, "serr": serr}

  def progress(self, rec, g):
    return (f"  [{rec['it']:3d}] RMSE={np.array2string(rec['rmse'], precision=4)}  "
            f"total|dI|={np.array2string(rec['terr'], precision=3)}  scale x{np.array2string(rec['serr'], precision=2)}")

  def summary(self, history, best):
    hb = history[best]
    return (f"[fr3 inertia sysid / {self.args.grad}] {self.args.num_envs} envs, {NLINK} links: "
            f"total |I-I_gt| {np.array2string(history[0]['terr'], precision=3)} -> "
            f"{np.array2string(hb['terr'], precision=4)}; RMSE {history[0]['rmse'].mean():.4f} -> {hb['rmse'].mean():.4f}")

  def default_out(self):
    return os.path.join(ASSETS, "franka_sysid.mp4")

  # ---- render: front row of sys-id arms + a back-row GT reference; tip-trace ribbons ----

  def _row_axes(self):
    a = np.radians(CAM_AZIMUTH)
    f = np.array([np.cos(a), np.sin(a)])
    return np.array([f[1], -f[0]]), f

  def _add_robot(self, scene, rm, d, qpos, opt, pert, offset, rgba=None):
    d.qpos[:] = qpos
    mujoco.mj_forward(rm, d)
    n0 = scene.ngeom
    cat = int(mujoco.mjtCatBit.mjCAT_STATIC) | int(mujoco.mjtCatBit.mjCAT_DYNAMIC)
    mujoco.mjv_addGeoms(rm, d, opt, pert, cat, scene)
    for k in range(n0, scene.ngeom):
      if scene.geoms[k].type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        scene.geoms[k].rgba[3] = 0.0
        continue
      scene.geoms[k].pos[0] += offset[0]
      scene.geoms[k].pos[1] += offset[1]
      if rgba is not None:
        scene.geoms[k].rgba = rgba

  def export_usd(self, history, best):
    """--export_usd hook: replay THIS inertia sys-id's convergence across an instanced Franka FIELD for the
    Blender render (franka_sysid_render_blender.py). Samples a spread of iterations; for each, re-runs the
    fixed chirp excitation in MuJoCo-C at that iteration's per-link inertia (env 0, recovering toward the
    true x1.0), subsamples + holds the settled end, tracking frame->iteration. The arm is fixed-base (no
    floor needed); the proto is fr3.xml stripped to named visual geoms."""
    a = self.args
    out_dir = os.path.join(ASSETS, "franka_sysid_render")
    ks = sorted(set(np.linspace(0, len(history) - 1, a.usd_iters).astype(int).tolist()))
    rm = mujoco.MjModel.from_xml_path(SCENE)
    prep_excitation(rm)
    pm_gt = np.array([rm.body_inertia[b].copy() for b in self.bids])  # GT principal moments per link
    rn = self.nf
    gt_qp, _ = _mjc_replay(rm, self.mjd, self.bids, list(pm_gt), self.ctrl, rn)  # GT trajectory -> the ghost (shared)
    idx = list(range(0, rn + 1, a.usd_stride))
    frame_iters = [int(k) for k in ks for _ in range(len(idx) + a.usd_hold)]  # same schedule across every env
    ne = self.args.num_envs
    env_frames = []  # a DISTINCT trajectory per env (each recovers its OWN wrong inertia) -> diverse tiled field
    for e in range(ne):
      fr = []
      for k in ks:
        inertia = [theta_to_principal(history[k]["theta"][e][li], pm_gt[li]) for li in range(NLINK)]
        sim_qp, _ = _mjc_replay(rm, self.mjd, self.bids, inertia, self.ctrl, rn)  # (rn+1, 7) at env e's iter-k inertia
        for t in idx:
          fr.append(np.concatenate([sim_qp[t], gt_qp[t]]))  # [sim(7) | gt(7)] overlaid (sim morphs to the ghost)
        fr += [np.concatenate([sim_qp[-1], gt_qp[-1]])] * a.usd_hold
      env_frames.append([np.asarray(f, np.float64) for f in fr])
    proto = self.multi_proto(os.path.join(os.path.dirname(SCENE), "fr3.xml"),
                             [("sim_", (0.0, 0.0, 0.0)), ("gt_", (0.0, 0.0, 0.0))])  # sim + GT ghost overlaid (like g1)
    assert proto.nq == env_frames[0][0].shape[0], (proto.nq, env_frames[0][0].shape[0])
    offsets = self._usd_grid_offsets(a.usd_envs, a.usd_cols, a.usd_xpitch, a.usd_ypitch)
    fps = max(1, round(1.0 / (rm.opt.timestep * a.usd_stride)))  # 1x real-time (chirp is 1 step/frame)
    out = self.export_field(proto, None, offsets, out_dir, name="franka_sysid_traj", fps=fps,
                            frame_iters=frame_iters, opt_label="link inertia scale", env_frames=env_frames)
    serrs = [round(float(history[ks[-1]]["serr"][e]), 1) for e in range(min(ne, 8))]
    print(f"[export] franka inertia sysid: {ne} DISTINCT envs, iters {ks}; final serr(first8) {serrs} "
          f"(true 1.0); sim+ghost overlay; NF={len(env_frames[0])} -> {out}")

  def render(self, history, best, out):
    vm = mujoco.MjModel.from_xml_path(SCENE)
    prep_excitation(vm)
    vm.vis.global_.offwidth, vm.vis.global_.offheight = W, H
    vm.vis.global_.fovy = CAM_FOVY
    vm.stat.extent, vm.stat.center = 3.5, np.array([0.1, 0.0, 0.4])
    root = mujoco.mj_name2id(vm, mujoco.mjtObj.mjOBJ_BODY, "base")
    if root >= 0:
      vm.body_pos[root] = [0.0, 0.0, -50.0]  # hide the model's own arm; every arm is an added, offset geom
    rm = mujoco.MjModel.from_xml_path(SCENE)
    prep_excitation(rm)
    vd, gd = mujoco.MjData(vm), mujoco.MjData(rm)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    E = history[0]["theta"].shape[0]
    r_ax, f_ax = self._row_axes()
    offs = [ROW_CENTER + (e - (E - 1) / 2) * ROW_SPACING * r_ax - 0.5 * ROW_SEP * f_ax for e in range(E)]
    gt_off = ROW_CENTER + 0.5 * ROW_SEP * f_ax
    rn = self.nf
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(vm, cam)
    cam.lookat, cam.distance, cam.azimuth, cam.elevation = list(CAM_LOOKAT), CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION
    el, az = np.radians(CAM_ELEVATION), np.radians(CAM_AZIMUTH)
    ref_lift = -REF_LIFT * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    parked = self.mjd.qpos.copy()
    pm_gt = np.array([rm.body_inertia[b].copy() for b in self.bids])
    _, gt_tip = _mjc_replay(rm, self.mjd, self.bids, list(pm_gt), self.ctrl, rn)
    serr0 = np.asarray(history[0]["serr"], dtype=np.float64)
    show = sorted(set([k for k in [0, max(1, best // 4), max(2, best // 2), 3 * best // 4] if k < best] + [best]))
    frames = []
    for k in show:
      h = history[k]
      s_s = "[" + ",".join(f"{v:.2f}" for v in h["serr"]) + "]"
      sub = f"iter {h['it']:3d}   inertia x{s_s}   (true x1.0)"
      env_rgb = [tuple(GT_RGBA[:3]) if float(h["serr"][e]) <= 1.02 else
                 tuple(viz.bourke_color_map(0.0, 1.0, 0.875 * (1.0 - float(np.clip(np.log(max(h["serr"][e], 1.0)) /
                 max(np.log(max(serr0[e], 1.0 + 1e-6)), 1e-9), 0.0, 1.0))))) for e in range(E)]
      replays = [_mjc_replay(rm, self.mjd, self.bids, [theta_to_principal(h["theta"][e][li], pm_gt[li]) for li in range(NLINK)],
                             self.ctrl, rn) for e in range(E)]
      env_q, env_tip = [r[0] for r in replays], [r[1] for r in replays]
      hold = 20 if k == best else 0
      for t in list(range(0, rn + 1, RENDER_STRIDE)) + [rn] * hold:
        sim_poses = [(env_q[e][t], offs[e]) for e in range(E)]
        ghost_q = self.gt_qpos[t]
        trail = slice(0, t + 1, 2)

        def draw(scene, sim_poses=sim_poses, ghost_q=ghost_q, trail=trail, env_tip=env_tip, gt_tip=gt_tip, env_rgb=env_rgb):
          for e, (sim_q, off) in enumerate(sim_poses):
            self._add_robot(scene, rm, gd, sim_q, opt, pert, off)
            off3 = np.array([off[0], off[1], 0.0])
            viz.add_polyline(scene, env_tip[e][trail] + off3, env_rgb[e], width=RIBBON_W_ENV)
            viz.add_polyline(scene, gt_tip[trail] + off3 + ref_lift, GT_RGBA[:3], width=RIBBON_W_REF)
          self._add_robot(scene, rm, gd, ghost_q, opt, pert, gt_off, GT_RGBA)
          viz.add_polyline(scene, gt_tip[trail] + np.array([gt_off[0], gt_off[1], 0.0]), GT_RGBA[:3], width=RIBBON_W_GT)

        frames.append((parked, draw, f"{sub}   [t={t}/{rn}]"))
    if frames:
      frames += [frames[-1]] * 20
    return viz.emit(vm, vd, cam, frames, out_path=out, label="ADJOINT (Franka inertia)", w=W, h=H, fps=FPS)


def main(argv):
  del argv  # config comes from the absl-parsed Args
  demo.run(FrankaSysidDemo, demo.parse_args(Args))


if __name__ == "__main__":
  demo.define_flags(Args)
  app.run(main)
