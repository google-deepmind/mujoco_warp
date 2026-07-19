"""Newton's-cradle trajectory-optimization video -- FD or analytic (mjwarp adjoint) gradient.

5 balls on rigid rods (HINGE joints), frictionless elastic ball<->ball contact. The two end balls
get inward initial velocities; the LEFT end's push is fixed and we OPTIMIZE the RIGHT end ball's
release velocity so the impact CANCELS at the middle ball -- loss = (middle ball's final speed)^2,
i.e. bring the middle ball to rest. Across gradient-descent iterations, viz animates each rollout
and traces the MIDDLE ball's path, loss-colored (red=knocked away -> blue=stays put), converging to
"the impact cancels". Many cradles are optimized IN PARALLEL (one lane per env); every env's release
velocity v4 is its own optimized variable, seeded across a spread so lanes start visibly different.

The `--grad` flag picks the gradient source (both descend the SAME batched rollout, per-env):
  * analytic (default): ONE batched (nworld=B) taped backward through differentiable mujoco_warp
    (adjoint.py). The optimized v4 reaches the middle ball *entirely* through ball<->ball contact, so
    this exercises the body<->body (two-moving-bodies) contact d(qpos) path (S3: _narrowphase_recompute
    sphere_sphere -> support.jac_dof). One tape.backward -> independent per-env gradients (verified ==
    per-world single backward); v4 descends and loss -> 0, MATCHING FD. See
    adjoint_test2.py::test_cradle_body_body_contact_grad_matches_fd.
  * fd: per-env central-difference over the MuJoCo-C rollout (the robust baseline).

The shared demo.Example harness owns capture + BackwardContext reuse, the fd/analytic dispatch, the
parallel-GD loop, and the CLI (a single Args dataclass parsed via absl.flags); CradleDemo supplies the
cradle-specific model + forward + loss + rollout + render.

  # analytic adjoint gradient (default) -> reports/assets/cradle_parallel.mp4
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/cradle.py
  # finite-difference baseline -> reports/assets/cradle_fd.mp4
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/cradle.py --grad=fd
  # GPU graph-capture scaling sweep (no video)
  uv run --active python contrib/diffsim/cradle.py --benchmark --envs=1,4,64,256,1024
"""

import os
import sys
import time
import typing
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import mujoco
import numpy as np
import warp as wp
from absl import app

import mujoco_warp as mjw

sys.path.insert(0, os.path.dirname(__file__))
import demo  # noqa: E402  shared config + gradients + capture/reuse + optimize loop + main
import viz as R  # noqa: E402  (bourke_color_map, add_polyline, default_show, emit)

N, R_BALL, L = 5, 0.1, 0.5  # L = pendulum/wire length (0.3 = faster balls / shorter horizon; 0.5 = the reverted-to iteration)
SPACING = 2.0 * R_BALL
PIVOT_X = [SPACING * i for i in range(N)]
MID = N // 2  # the ball we want to keep at rest (index 2)
DT, T = 0.002, 80
# Contact/constraint capacities: 5 balls contact only adjacent neighbours -> <=4 sphere-sphere
# contacts (condim=1 -> 4 rows); 8/8 keeps 2x headroom. Overrides put_data's 48/64 default so the
# per-world contact/efc arrays (and the contact-VJP work baked into the captured graph) stay small.
NCONMAX, NJMAX = 8, 8
V0_FIXED = -3.0  # left end ball inward angular velocity (the fixed "given" push)
V4_INIT = 1.2  # right end ball inward angular velocity = the OPTIMIZED variable (optimum ~ +3.0)
GRAV, BALL_I, MGL = 9.81, 1.0 * L * L, 1.0 * 9.81 * L  # swap_control: hinge inertia + pendulum PE scale
SWAP = 0  # swap_control: the LEFT ball whose rebound apex we target (it leaves at the RIGHT ball's speed)
RED = 0   # launch: the LEFT (red) ball whose launch height we MAXIMIZE, driven by the RIGHT (blue) ball's velocity
SOLREF = "-50000 -10"  # stiff elastic ball<->ball (matches adjoint_test2 _cradle_xml)
W, H = 1024, 768
_BALL_RGBA = ["0.85 0.2 0.2 1"] + ["0.7 0.7 0.75 1"] * (N - 2) + ["0.2 0.4 0.85 1"]  # red end, gray mid, blue end

# --- viz frame (decorative only; contype=0/conaffinity=0, so it never touches the physics) ---
FRAME_HALF_Y = 0.22    # front/back rail spacing -> width of the V-string suspension (swing stays in x-z)
FRAME_MARGIN_X = 0.10  # top/bottom rails run this far past the end pivots
BASE_Z = -0.18         # frame base: posts stand from here up to the pivot rail at z=L
FLOOR_Z = -0.20        # grid floor just under the base so each cradle stands on it

ASSETS = os.path.join(os.path.dirname(__file__), "reports", "assets")  # default output dir


@dataclass
class Args(demo.CommonArgs):
  """cradle config: CommonArgs (grad/device/capture/out/live/...) + cradle defaults + the
  scaling-benchmark and render-grid flags. A single typed object; parsed from the CLI via absl.flags."""

  num_envs: int = field(default=4, metadata={"help": "parallel cradles optimized at once"})
  steps: int = field(default=60, metadata={"help": "gradient-descent steps"})
  lr: float = field(default=0.5, metadata={"help": "gradient-descent learning rate"})
  spread: float = field(default=0.6, metadata={"help": "per-env init v4 fan width"})
  task: str = field(default="middle_rest", metadata={"help": "cradle objective", "choices": ["middle_rest", "swap_control", "launch"]})
  swap_deg: float = field(default=50.0, metadata={"help": "swap_control: target apex angle of the LEFT ball's rebound (deg)"})
  stiff: bool = field(default=False, metadata={"help": "run the OPTIMIZATION in the crisp elastic regime (dt=1e-4, solref=-8e6, hidden margin) -- e.g. to show middle_rest go degenerate; pair with --grad=fd"})
  benchmark: bool = field(default=False, metadata={"help": "run the graph-capture scaling sweep and exit; no video"})
  envs: typing.Optional[str] = field(default=None, metadata={"help": "comma env counts for --benchmark (default 1,4,16,64,256,1024)"})
  cols: int = field(default=2, metadata={"help": "render grid width (columns of cradles)"})
  env_spacing: float = field(default=2.2, metadata={"help": "render grid pitch (> cradle width)"})
  # --export_usd: replay THIS optimization's convergence across an instanced cradle FIELD (Blender render)
  usd_envs: int = field(default=169, metadata={"help": "USD field: number of cradles (13*13 -> square field)"})
  usd_cols: int = field(default=13, metadata={"help": "USD field: grid columns (odd -> true centre column; 13 -> square 13x13)"})
  usd_xpitch: float = field(default=2.4, metadata={"help": "USD field: column pitch (x)"})
  usd_ypitch: float = field(default=2.4, metadata={"help": "USD field: row pitch (y, = xpitch -> uniform)"})
  usd_iters: int = field(default=6, metadata={"help": "USD export: GD iterations sampled across the optimization (convergence)"})
  usd_stride: int = field(default=3, metadata={"help": "USD export: timestep subsample within each iteration's rollout"})
  usd_hold: int = field(default=6, metadata={"help": "USD export: frames held at each rollout's settled end"})
  usd_replicate: bool = field(default=True, metadata={"help": "USD export: replicate one lane across an instanced field (True) or show all num_envs DISTINCT optimization lanes (False)"})
  classic: bool = field(default=False, metadata={"help": "no-optimization CLASSIC Newton's cradle: lift the end ball, release, multi-bounce sim -> self-contained USD"})
  classic_lift_deg: float = field(default=55.0, metadata={"help": "classic: lift angle of the end ball (deg)"})
  classic_secs: float = field(default=4.0, metadata={"help": "classic: simulation duration (s) -- longer -> more bounces"})


@wp.kernel
def _mid_vel_sq_batched(qvel: wp.array2d[float], idx: int, loss: wp.array(dtype=float)):
  """loss = sum_w (middle-ball hinge speed_w)^2. Each env's v4 only drives its own world, so
  d(sum)/d(v4_w) = d(loss_w)/d(v4_w): one backward -> independent per-env gradients. (@wp.kernel must
  stay module-level -- Warp JITs it by module path.)"""
  w = wp.tid()
  wp.atomic_add(loss, 0, qvel[w, idx] * qvel[w, idx])


@wp.kernel
def _ball_energy_sq_batched(qpos: wp.array2d[float], qvel: wp.array2d[float], idx: int,
                            iner: float, mgl: float, target: float, loss: wp.array(dtype=float)):
  """loss = sum_w (E_idx_w - target)^2, E = 1/2 I w^2 + m g L (1-cos th): ball idx's conserved pendulum
  energy (-> its apex height). Smooth in the input velocity (monotonic elastic momentum transfer)."""
  w = wp.tid()
  th = qpos[w, idx]
  om = qvel[w, idx]
  e = 0.5 * iner * om * om + mgl * (1.0 - wp.cos(th))
  d = e - target
  wp.atomic_add(loss, 0, d * d)


@wp.kernel
def _max_ball_height(qpos: wp.array2d[float], idx: int, rod_len: float, hmax: wp.array(dtype=float)):
  """Running max of ball idx's HEIGHT = L*(1-cos theta_idx), read STRAIGHT FROM THE JOINT ANGLE (a pure
  position, no energy/velocity). Called every taped step -> hmax tracks red's PEAK (apex) height over the
  whole flight, which IS monotonic in the push (unlike red's height at a fixed step). The 'launch' task."""
  w = wp.tid()
  hmax[w] = wp.max(hmax[w], rod_len * (1.0 - wp.cos(qpos[w, idx])))


@wp.kernel
def _neg_sum_batched(vals: wp.array(dtype=float), loss: wp.array(dtype=float)):
  """loss = sum_w -(vals_w): minimizing MAXIMIZES the tracked per-env quantity (here red's peak height)."""
  w = wp.tid()
  wp.atomic_add(loss, 0, -vals[w])


@wp.kernel
def _launch_step_loss(qpos: wp.array2d(dtype=float), idx: int, rod_len: float, loss: wp.array(dtype=float)):
  """launch (PER-STEP): accumulate -(red height) = -L*(1-cos th_red) at this taped step into the running
  total, so loss = -sum_t red-height. Minimizing MAXIMIZES red's time-integrated launch height -- a plain
  per-step-accumulated objective (no running max, no carried accumulator), which makes d(loss)/d(loss)=1 a
  constant seed at every chunk: only the per-step (qpos,qvel) state adjoint crosses chunk boundaries. This
  is the shape the checkpointed harness (demo.Example single-step/chunk capture) needs."""
  w = wp.tid()
  wp.atomic_add(loss, 0, -rod_len * (1.0 - wp.cos(qpos[w, idx])))


class CradleDemo(demo.Example):
  """Newton's-cradle: optimize each env's right-end release velocity v4 (params (E,)) so the impact
  cancels at the middle ball. The taped chunk the harness captures/replays is the batched hinge
  rollout + middle-ball final-speed loss; the fd path central-differences the MuJoCo-C rollout."""

  Args = Args

  # ---- harness hooks ----

  def build_model(self):
    self.mjm = mujoco.MjModel.from_xml_string(self.cradle_xml(**self._phys()))
    self.mjd = mujoco.MjData(self.mjm)
    mujoco.mj_forward(self.mjm, self.mjd)

  def init_params(self):
    """Per-env initial v4: a fan from a low anchor up toward (but kept clearly BELOW) the +3.0 cancel
    optimum, so every lane starts with the middle ball visibly knocked away (|v4 - 3| >= ~1.0) and
    converges upward -- not pre-solved. `spread` widens the fan; the top is capped < 3.0 so a wide
    spread can't overshoot. The low anchor 0.8 stays above the ~0.7 wrong-basin floor. num_envs==1 ->
    the FD baseline's V4_INIT so a single-cradle run matches the classic single-env optimization."""
    ne, spread = self.args.num_envs, self.args.spread
    if self.args.task == "launch":  # per-env fan of blue OUTWARD initial speeds; maximizing red's launch drives them UP
      push = lambda deg: np.sqrt(2.0 * GRAV * (1.0 - np.cos(np.radians(deg))) / L)  # push -> blue up-swing/red-apex ~ deg (L-robust)
      if ne == 1:
        return np.array([push(1.5)])  # a TINY initial push (red launches ~1.5deg); optimization grows it big
      return np.array([push(d) for d in np.linspace(1.5, 38.0, ne)])  # fan from a tiny to a big push -> all climb (env0 = the dramatic 1.5deg seed, replicated for the Blender field)
    if self.args.task == "swap_control":
      bhit = np.sqrt(max(2.0 * self._swap_target() / BALL_I, 1e-6))  # right-ball speed the left rebound needs (b > 0 = inward)
      if ne == 1:
        return np.array([1.2 * bhit])
      return np.linspace(0.4 * bhit, 1.7 * bhit, ne)  # spread of right-ball drops -> converge onto the target
    if ne == 1:
      return np.array([V4_INIT], dtype=np.float64)
    lo, hi = 0.8, min(0.8 + 2.0 * spread, 2.2)  # cap < the +3.0 optimum -> no lane starts solved
    return np.linspace(lo, hi, ne)

  def build_datas(self):
    ne = self.args.num_envs
    self.nT = self._nsteps(taped=True)  # analytic BPTT length (checkpointed: only chunk+1 segment buffers)
    self.datas = [mjw.put_data(self.mjm, self.mjd, nworld=ne, nconmax=NCONMAX, njmax=NJMAX) for _ in range(self.args.chunk + 1)]
    for d in self.datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    self.datas[0].qvel = wp.array(self._qvel0(np.zeros(ne)), dtype=float, requires_grad=True)  # optimized leaf = end-ball release velocity
    self.loss = wp.zeros(1, dtype=float, requires_grad=True)

  def chunk_step(self, i, t):
    """One physics step over the segment buffers. launch accumulates its per-step height loss here (the
    carry-free per-step objective); middle_rest/swap are terminal (see terminal_loss). `t` is unused --
    every objective is time-independent, so the captured single-chunk graph replays unchanged."""
    del t
    mjw.step(self.m, self.datas[i], self.datas[i + 1])
    if self.args.task == "launch":
      wp.launch(_launch_step_loss, dim=self.args.num_envs, inputs=[self.datas[i + 1].qpos, RED, L], outputs=[self.loss])

  def terminal_loss(self):
    """middle_rest/swap terminal objective on the FINAL state (datas[0] == state_T after the forward
    pass). launch has no terminal term (its loss is the per-step sum accumulated in chunk_step)."""
    if self.args.task == "launch":
      return
    d = self.datas[0]
    if self.args.task == "swap_control":
      wp.launch(_ball_energy_sq_batched, dim=self.args.num_envs,
                inputs=[d.qpos, d.qvel, SWAP, BALL_I, MGL, self._swap_target()], outputs=[self.loss])
    else:  # middle_rest: minimize the middle ball's final hinge speed^2 (impact cancels)
      wp.launch(_mid_vel_sq_batched, dim=self.args.num_envs, inputs=[d.qvel, MID], outputs=[self.loss])

  def set_params(self):
    self.datas[0].qvel.assign(self._qvel0(self.params))  # release velocities -> datas[0].qvel in place (launch: outward, via _qvel0)

  def read_grad(self):  # launch: qvel_blue = +push (inward at the clack) -> d(loss)/d(push) = +qvel.grad (same as the other tasks)
    return np.nan_to_num(self.datas[0].qvel.grad.numpy())[:, N - 1].copy()

  def _swap_target(self):
    """swap_control: the LEFT ball's pendulum energy to reach swap_deg at its rebound apex (it leaves at
    the RIGHT ball's incoming speed -- the equal-mass elastic velocity swap)."""
    return MGL * (1.0 - np.cos(np.radians(self.args.swap_deg)))

  def rollout_env(self, p):
    """mj_step rollout for one env: end-ball velocities (V0_FIXED, p). middle_rest loss = middle ball's
    final hinge speed^2 (impact cancels). swap_control loss = (LEFT ball's rebound apex energy - target)^2
    -- the left ball leaves at the right ball's speed (velocity swap). Returns (tracked_xyz, qpos, loss)."""
    m = mujoco.MjModel.from_xml_string(self.cradle_xml(**self._phys()))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    task = self.args.task
    track = MID if task == "middle_rest" else 0  # launch/swap watch the left (red) ball
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"ball{track}")
    d.qvel[:] = 0.0
    if task == "launch":
      d.qvel[N - 1] = -p  # blue's OUTWARD initial speed (p = push magnitude): swings up-right, falls back, clacks
    else:
      d.qvel[N - 1] = p
      d.qvel[0] = V0_FIXED
    track_xyz = [d.geom_xpos[gid].copy()]
    qpos = [d.qpos.copy()]
    for _ in range(self._nsteps()):
      mujoco.mj_step(m, d)
      track_xyz.append(d.geom_xpos[gid].copy())
      qpos.append(d.qpos.copy())
    e0 = 0.5 * BALL_I * d.qvel[0] ** 2 + MGL * (1.0 - np.cos(d.qpos[0]))  # left/red ball pendulum energy
    if task == "launch":
      loss = -L * (1.0 - np.cos(d.qpos[RED]))  # MAXIMIZE red's HEIGHT (position), matching the taped loss
    elif task == "swap_control":
      loss = (e0 - self._swap_target()) ** 2
    else:
      loss = d.qvel[MID] ** 2
    return np.array(track_xyz), np.array(qpos), float(loss)

  def fd_grad(self, p, eps=1e-3):
    """Central-difference d(loss)/d(v4) for one cradle (the robust baseline)."""
    return (self.rollout_env(p + eps)[2] - self.rollout_env(p - eps)[2]) / (2 * eps)

  def record(self, it, losses, trajs, qposs):
    return {"it": it, "v4s": self.params.copy(), "losses": losses.copy(), "mid_xyz": trajs, "qpos": qposs}

  def progress(self, it, losses, g):
    return f"  [{it:3d}] v4 {np.round(self.params, 3)}  mean_loss={losses.mean():.5f} best={losses.min():.5f}"

  def summary(self, history, best):
    ml = [h["losses"].mean() for h in history]
    tag = "analytic (adjoint, body<->body S3)" if self.args.grad == "analytic" else "finite diff"
    return (
      f"[cradle optim / {tag}] {self.args.num_envs} envs: "
      f"mean_loss {ml[0]:.5f} -> best {ml[best]:.5f} (iter {best})"
    )

  def default_out(self):
    name = "cradle_parallel" if self.args.grad == "analytic" else "cradle_fd"
    return os.path.join(ASSETS, name + ".mp4")

  # ---- cradle-specific helpers ----

  def _qvel0(self, vs):
    """Initial hinge velocities (B, N): optimized right-end (blue) speed. launch drives ONLY the blue ball
    with an OUTWARD push (+y hinge -> +theta is inward/-x, so outward = -vs): blue swings up-right, falls
    back, then clacks the chain. middle_rest/swap_control instead send blue inward (+vs) + the fixed left push."""
    q = np.zeros((len(vs), N), dtype=np.float32)
    if self.args.task == "launch":
      q[:, N - 1] = np.asarray(vs, dtype=np.float32)  # blue INWARD at the clack (truncated-BPTT taped rollout); the
      # lossless outward swing (shown in the viz) brings blue back to the bottom at exactly this speed vs
    else:
      q[:, N - 1] = vs
      q[:, 0] = V0_FIXED
    return q

  def batched_analytic_grad(self, m, mjm, mjd, v4s):
    """Fresh-alloc reference: per-env d(loss)/d(v4) via a batched (nworld=B) taped backward with NO
    capture and NO BackwardContext reuse -- reallocates T+1 Datas each call. The oracle the captured/
    reused analytic path must match (bitwise on CPU; atomic-add noise on GPU)."""
    num_envs = len(v4s)
    datas = [mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=NCONMAX, njmax=NJMAX) for _ in range(T + 1)]
    for d in datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    datas[0].qvel = wp.array(self._qvel0(v4s), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      for t in range(T):
        mjw.step(m, datas[t], datas[t + 1])
      wp.launch(_mid_vel_sq_batched, dim=num_envs, inputs=[datas[T].qvel, MID], outputs=[loss])
    tape.backward(loss=loss)
    return np.nan_to_num(datas[0].qvel.grad.numpy())[:, N - 1]

  def _sync(self, device):
    """Block until the device's queued work is done, so perf_counter brackets real GPU time (no-op CPU)."""
    if device.is_cuda:
      wp.synchronize_device(device)

  def benchmark(self):
    """Scaling sweep for the graph-captured analytic gradient (effective fps vs env count). For each
    env count: build the captured rollout+backward ONCE, then time the real descent iter (v4 -> grad
    readback -> update), the pure graph replay, and check loss0->lossN still converges. On CPU (no
    graph) it runs the eager reuse path so the table is populated, minus the raw-replay column."""
    a = self.args
    env_counts = [int(x) for x in a.envs.split(",")] if a.envs else [1, 4, 16, 64, 256, 1024]
    steps, warmup = 30, 5
    device = wp.get_device()
    print(f"[cradle bench] device={device}  rollout T={T} (dt={DT})  {steps} timed iters (+{warmup} warmup)  "
          f"capture={'auto' if a.capture is None else a.capture}")
    print(f"[cradle bench] {'envs':>6} {'setup(s)':>9} {'ms/iter':>9} {'iters/s':>9} "
          f"{'fwd Msteps/s':>13} {'raw ms/it':>10} {'raw Msteps/s':>13}   {'loss0->lossN':>20}")
    rows = []
    for nworld in env_counts:
      t0 = time.perf_counter()
      ex = CradleDemo(replace(a, num_envs=nworld, grad="analytic", benchmark=False))
      self._sync(device)
      setup = time.perf_counter() - t0

      v4 = ex.init_params()
      for _ in range(warmup):  # warm the descent (first real replays) + settle the optimizer
        ex.params = v4
        v4 = v4 - a.lr * ex.analytic_grad()
      loss0 = float(ex.loss.numpy()[0]) / nworld  # reported loss = sum_w mid_vel^2 -> mean per env
      self._sync(device)

      t0 = time.perf_counter()  # (b) realistic optim-iter latency (grad readback + update included)
      for _ in range(steps):
        ex.params = v4
        v4 = v4 - a.lr * ex.analytic_grad()
      self._sync(device)
      ms_iter = (time.perf_counter() - t0) / steps
      lossN = float(ex.loss.numpy()[0]) / nworld

      raw = float("nan")  # (c) retired: the analytic path is now a per-CHUNK graph replayed over the
      # trajectory (host-side chunk loop), not one whole-tape graph -- ms_iter already times the real backward.

      fwd_msteps = nworld * T / ms_iter / 1e6
      raw_msteps = nworld * T / raw / 1e6 if raw == raw else float("nan")
      rows.append({"nworld": nworld, "setup": setup, "ms_iter": ms_iter, "raw": raw,
                   "fwd_msteps": fwd_msteps, "loss0": loss0, "lossN": lossN})
      print(f"[cradle bench] {nworld:>6} {setup:>9.2f} {ms_iter * 1e3:>9.3f} {1 / ms_iter:>9.1f} "
            f"{fwd_msteps:>13.2f} {raw * 1e3:>10.3f} {raw_msteps:>13.2f}   {loss0:>8.4f} -> {lossN:<8.4f}")
    return rows

  # ---- physics + viz scene builders ----

  def cradle_xml(self, gap=0.0, solref=SOLREF, solimp="0 0.95 0.001", dt=DT, ball_margin=0.0, ball_gap=0.0):
    """The minimal PHYSICS model: 5 balls on hinge rods, elastic ball<->ball contact. This is the only
    scene that gets stepped/differentiated (rollout + the batched nworld backward); rendering uses
    _cradle_xml_multi instead. No frame/floor/lights -- keep the diff'd model minimal. The classic sim
    overrides solref/solimp/dt + a NEGATIVE ball margin (Codex recipe): a stiff, time-resolved, zero-damping
    elastic contact with a hidden 1mm clearance, so the balls look TOUCHING yet collide sequentially
    (clean one-in/one-out). margin+gap sum across the two contacting geoms -> total -1mm margin / +1mm gap."""
    pivots = [i * (SPACING + gap) for i in range(N)]
    mg = f' margin="{ball_margin}" gap="{ball_gap}"' if (ball_margin or ball_gap) else ""
    bodies = ""
    for i in range(N):
      rod_rgba = "0.15 0.15 0.17 1"
      bodies += (
        f'<body name="b{i}" pos="{pivots[i]:.4f} 0 {L:.4f}">'
        f'<joint name="h{i}" type="hinge" axis="0 1 0"/>'
        f'<geom type="capsule" fromto="0 0 0 0 0 -{L}" size="0.006" rgba="{rod_rgba}" contype="0" conaffinity="0" mass="0.02"/>'
        f'<geom name="ball{i}" type="sphere" size="{R_BALL}" pos="0 0 -{L}" mass="1" rgba="{_BALL_RGBA[i]}"{mg}/>'
        f"</body>"
      )
    opt = (
      f'<option timestep="{dt}" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
      f'solver="Newton" iterations="100" ls_iterations="50" tolerance="1e-10"><flag eulerdamp="disable"/></option>'
    )
    default = f'<default><geom condim="1" solref="{solref}" solimp="{solimp}"/></default>'
    return f"<mujoco>{opt}{default}<worldbody>{bodies}</worldbody></mujoco>"

  def _phys(self):
    """cradle_xml kwargs for the current contact regime: {} = the soft default (smooth gradients);
    --stiff = Codex's crisp elastic recipe (dt=1e-4, stiff zero-damping contact, hidden 1mm margin)."""
    if getattr(self.args, "stiff", False) or self.args.task == "launch":  # launch needs crisp elastic transfer
      return dict(solref="-8000000 0", solimp="0 0.95 0.001 0.5 2", dt=1e-4, ball_margin=-0.0005, ball_gap=0.0005)
    return {}

  def _nsteps(self, viz=False, taped=False):
    """Rollout step count. Soft: the T=80 default. Stiff (dt=1e-4, 20x finer): more steps for the same
    physical time. launch has THREE lengths (truncated BPTT): taped clack / fd swing / viz swing."""
    if self.args.task == "launch":
      # The HEIGHT loss reads red's POSITION L*(1-cos th_red) (no energy). taped=1400 runs blue's clack (blue
      # arrives inward at the push speed) THROUGH red's launch and partway up its flight -- red is still RISING
      # there for the whole range, so its height is monotonic in the push, and the adjoint flows through red's
      # flight AND the sphere-sphere contact back to blue's push. (Taping all the way to red's apex ~3500 steps
      # is exact but Warp's graph capture chokes on that many unrolled steps; 1400 captures fast and the VIZ
      # shows red flying to its full apex regardless.) VIZ (11800) = the full outward swing; fd path = 7800.
      if taped:
        return 4500  # run PAST red's apex so the running-max height captures red's PEAK (apex ~3830-4408 steps
        # across the 1.5-40deg range). This is longer than the clack -> Warp graph capture + the harness fast
        # backward both choke, so run --capture=off (eager, ~80s/iter).
      return 11800 if viz else 7800
    if getattr(self.args, "stiff", False):
      return 4000 if viz else 400  # 400 taped steps = 40ms: through the collision, cheap enough to BPTT
    return T

  def _ball_mat(self, i):
    """Our colors: red left end, blue right end, gray in between."""
    if i == 0:
      return "ball_red"
    if i == N - 1:
      return "ball_blue"
    return "ball_gray"

  def _viz_assets(self):
    """<asset> for the viz scenes: our dark-blue skybox + checker grid, plus metallic materials that
    keep our palette -- red/gray/blue balls (specular so they read as polished), a brushed-steel frame,
    and light emissive strings that stay visible even as thin capsules."""
    return """
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="8 8" reflectance="0.2"/>
    <material name="ball_red"  rgba="0.85 0.2 0.2 1"  specular="0.5" shininess="0.5" reflectance="0.1"/>
    <material name="ball_gray" rgba="0.7 0.7 0.75 1" specular="0.5" shininess="0.5" reflectance="0.15"/>
    <material name="ball_blue" rgba="0.2 0.4 0.85 1" specular="0.5" shininess="0.5" reflectance="0.1"/>
    <material name="frame_mat" rgba="0.5 0.5 0.55 1" specular="0.8" shininess="0.6" reflectance="0.35"/>
    <material name="string_mat" rgba="0.82 0.82 0.88 1" specular="0.4" shininess="0.3" emission="0.3"/>
  </asset>"""

  def _viz_visual(self):
    """<visual> for the viz scenes (unchanged from our theme): soft headlight, haze, offscreen size."""
    return f"""
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="{W}" offheight="{H}"/>
  </visual>"""

  def _frame_geoms(self, ox, oy, oz):
    """A standing cradle frame: two U-rails (front/back at y=+-FRAME_HALF_Y) plus bottom rails, so the
    posts rise from BASE_Z to the pivot rail at z=L and the whole thing sits on the floor. Decorative
    (contype=0/conaffinity=0) -- purely the thing the balls visibly hang from."""
    x0 = PIVOT_X[0] - FRAME_MARGIN_X + ox
    x1 = PIVOT_X[-1] + FRAME_MARGIN_X + ox
    zt, zb = L + oz, BASE_Z + oz
    segs = []
    for fy in (-FRAME_HALF_Y, FRAME_HALF_Y):
      y = fy + oy
      segs += [
        (x0, y, zb, x0, y, zt),  # left post
        (x1, y, zb, x1, y, zt),  # right post
        (x0, y, zt, x1, y, zt),  # top rail (the balls' V-strings pin here)
        (x0, y, zb, x1, y, zb),  # bottom rail (foot)
      ]
    return "".join(
      f'<geom type="capsule" size="0.012" material="frame_mat" contype="0" conaffinity="0" '
      f'fromto="{a:.4f} {b:.4f} {c:.4f} {d:.4f} {e:.4f} {f:.4f}"/>'
      for (a, b, c, d, e, f) in segs
    )

  def _ball_body_viz(self, prefix, i, px, oy, oz):
    """One suspended ball for the viz: the SAME hinge DOF as the physics scene + our-colored ball +
    a V of two strings to the front/back rails (replacing the physics rod). The string tops sit on the
    hinge axis (body-local (0, +-FRAME_HALF_Y, 0)), so they stay pinned to the rails as the ball swings."""
    return (
      f'<body name="{prefix}b{i}" pos="{px:.4f} {oy:.4f} {L + oz:.4f}">'
      f'<joint name="{prefix}h{i}" type="hinge" axis="0 1 0"/>'
      f'<geom name="{prefix}ball{i}" type="sphere" size="{R_BALL}" pos="0 0 -{L}" material="{self._ball_mat(i)}" mass="1"/>'
      f'<geom type="capsule" size="0.0045" material="string_mat" contype="0" conaffinity="0" fromto="0 0 -{L}  0 {-FRAME_HALF_Y:.4f} 0"/>'
      f'<geom type="capsule" size="0.0045" material="string_mat" contype="0" conaffinity="0" fromto="0 0 -{L}  0 {FRAME_HALF_Y:.4f} 0"/>'
      f"</body>"
    )

  def _env_offsets(self, num_envs, cols, spacing):
    """Per-env world offset (B, 3): cradles laid out in a `cols`-wide GRID (x = columns, y = rows/
    depth), centered on 0. A grid (vs a single row) keeps the cradles from visually overlapping."""
    off = np.zeros((num_envs, 3))
    if num_envs > 1:
      rows = int(np.ceil(num_envs / cols))
      for e in range(num_envs):
        r, c = e // cols, e % cols
        off[e, 0] = (c - (cols - 1) / 2.0) * spacing  # columns along x
        off[e, 1] = (r - (rows - 1) / 2.0) * spacing  # rows along y (depth)
    return off

  def _cradle_xml_multi(self, offsets, floor=True):
    """Cradle viz scene REPLICATED per env: each lane is a standing frame (two U-rails + posts + feet)
    with five V-string-suspended balls in our colors (red/gray/blue ends) and a purple rest-target.
    floor=True adds the checker floor + lights (their grid/skybox textures); floor=False drops them so the
    exported USD is self-contained rgba (opens directly in usdview/Preview/Blender). contype=0 throughout."""
    opt = (
      f'<option timestep="{DT}" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
      f'solver="Newton" iterations="100" ls_iterations="50" tolerance="1e-10"><flag eulerdamp="disable"/></option>'
    )
    default = f'<default><geom condim="1" solref="{SOLREF}" solimp="0 0.95 0.001"/></default>'
    bar_cx = 0.5 * (PIVOT_X[0] + PIVOT_X[-1])
    mx = PIVOT_X[MID]
    lanes = ""
    for e, (ox, oy, oz) in enumerate(offsets):
      lanes += self._frame_geoms(ox, oy, oz)
      if self.args.task == "middle_rest" and not self.args.classic:  # rest-target only meaningful for middle_rest
        lanes += (
          f'<geom type="sphere" size="0.045" pos="{mx + ox:.4f} {oy:.4f} {oz:.4f}" rgba="0.5 0 0.5 0.8" '
          f'contype="0" conaffinity="0"/>'
        )
      for i in range(N):
        lanes += self._ball_body_viz(f"e{e}", i, PIVOT_X[i] + ox, oy, oz)
    fsize = max(8.0, 0.5 * (PIVOT_X[-1] - PIVOT_X[0]) + FRAME_MARGIN_X + float(np.abs(offsets).max()) + 5.0)
    ground = (
      f'<light pos="0.4 -1 2" dir="0 0.4 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>'
      f'<light pos="1.5 1 1.5" dir="-0.6 -0.4 -1" diffuse="0.35 0.35 0.4" specular="0.2 0.2 0.2"/>'
      f'<geom name="floor" type="plane" pos="{bar_cx:.4f} 0 {FLOOR_Z}" size="{fsize:.3f} {fsize:.3f} .05" '
      f'material="grid" contype="0" conaffinity="0"/>'
    ) if floor else ""
    return f"""
<mujoco>
  {opt}
  {default}
  {self._viz_visual()}
  {self._viz_assets()}
  <worldbody>
    {ground}
    {lanes}
  </worldbody>
</mujoco>
"""

  # ---- USD trajectory export (--export_usd) ----

  def _usd_grid_offsets(self, n, cols, xpitch, ypitch):
    """Per-env world offset (n, 3) on a cols-wide grid, separate x/y pitch (cradles are much wider in x
    than deep in y, so independent pitches pack the field better). Each cradle is shifted left by
    PIVOT_X[MID] so its middle ball -- the cradle's symmetric x-centre -- lands on its lane origin; the
    whole field is then centred on x=0, so the render frames on x=0 with no mesh-derived centre needed."""
    rows = int(np.ceil(n / cols))
    off = np.zeros((n, 3))
    for e in range(n):
      r, c = divmod(e, cols)
      off[e, 0] = (c - (cols - 1) / 2.0) * xpitch - PIVOT_X[MID]
      off[e, 1] = (r - (rows - 1) / 2.0) * ypitch
    return off

  def _usd_rollout(self, nf, substeps, lift):
    """Real cradle physics (collisions): lift ball 0, release, capture qpos each frame -> (nf, N). One
    rollout replayed across every lane -> the whole field swings in sync (the momentum-transfer clack)."""
    pm = mujoco.MjModel.from_xml_string(self.cradle_xml())
    pd = mujoco.MjData(pm)
    mujoco.mj_forward(pm, pd)
    pd.qpos[0] = lift  # cock the left end ball; the rest hang in contact -> released, it clacks the chain
    mujoco.mj_forward(pm, pd)
    traj = np.empty((nf, pm.nq))
    for f in range(nf):
      for _ in range(substeps):
        mujoco.mj_step(pm, pd)
      traj[f] = pd.qpos
    return traj

  def _cradle_xml_proto(self):
    """A SINGLE cradle at the origin (frame + five V-string balls + rest-target), NO floor/lights -- the
    animated prototype the instanced field references. Keeps the same opt/default/assets/offwidth as the
    multi scene so the materials + the USDExporter framebuffer check are satisfied. nq == N (5 hinges)."""
    opt = (
      f'<option timestep="{DT}" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
      f'solver="Newton" iterations="100" ls_iterations="50" tolerance="1e-10"><flag eulerdamp="disable"/></option>'
    )
    default = f'<default><geom condim="1" solref="{SOLREF}" solimp="0 0.95 0.001"/></default>'
    lanes = self._frame_geoms(0.0, 0.0, 0.0)
    for i in range(N):
      lanes += self._ball_body_viz("e0", i, PIVOT_X[i], 0.0, 0.0)
    return f"<mujoco>{opt}{default}{self._viz_visual()}{self._viz_assets()}<worldbody>{lanes}</worldbody></mujoco>"

  def _write_instanced_field(self, proto_path, offsets, nframes, fps, out_path):
    """Build the instanced field: /World/lane_i is an Xform with the grid-offset translate, holding an
    instanceable child that references the prototype's /World. Every reference is identical, so USD (and
    Blender's usd_import use_instancing) share ONE prototype -> the exporter tessellates a single cradle
    and Cycles syncs a single cradle. Returns out_path."""
    from pxr import Usd, UsdGeom  # lazy: the bpy render venv has no pxr

    if os.path.exists(out_path):
      os.remove(out_path)
    stage = Usd.Stage.CreateNew(out_path)  # anchor at frames/ so the relative proto reference resolves
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # else Blender shrinks the scene 100x
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(0, nframes - 1))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    rel = os.path.relpath(proto_path, os.path.dirname(out_path))  # "cradle_proto.usdc" (same frames/ dir)
    for i, (ox, oy, oz) in enumerate(offsets):
      lane = UsdGeom.Xform.Define(stage, f"/World/lane_{i:04d}")
      lane.AddTranslateOp().Set((float(ox), float(oy), float(oz)))
      cradle = stage.DefinePrim(f"/World/lane_{i:04d}/Cradle")
      cradle.GetReferences().AddReference(rel, "/World")
      cradle.SetInstanceable(True)  # identical refs -> single shared prototype
    stage.GetRootLayer().Save()
    return out_path

  def _cradle_xml_classic(self, gap):
    """ONE gapped cradle for the classic sim's viz: a standing frame + five V-string rgba balls, pivots
    gapped to MATCH the physics so the balls just touch at each clack (no overlap). Self-contained rgba
    (no floor/lights/textures) -> opens directly in usdview/Preview/Blender."""
    pivots = [i * (SPACING + gap) for i in range(N)]
    opt = (
      f'<option timestep="{DT}" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
      f'solver="Newton" iterations="100" ls_iterations="50" tolerance="1e-10"><flag eulerdamp="disable"/></option>'
    )
    default = f'<default><geom condim="1" solref="{SOLREF}" solimp="0 0.95 0.001"/></default>'
    x0, x1 = pivots[0] - FRAME_MARGIN_X, pivots[-1] + FRAME_MARGIN_X
    zt, zb = L, BASE_Z
    segs = []
    for fy in (-FRAME_HALF_Y, FRAME_HALF_Y):
      segs += [(x0, fy, zb, x0, fy, zt), (x1, fy, zb, x1, fy, zt),
               (x0, fy, zt, x1, fy, zt), (x0, fy, zb, x1, fy, zb)]
    frame = "".join(
      f'<geom type="capsule" size="0.012" material="frame_mat" contype="0" conaffinity="0" '
      f'fromto="{a:.4f} {b:.4f} {c:.4f} {d:.4f} {e:.4f} {f:.4f}"/>' for (a, b, c, d, e, f) in segs
    )
    balls = "".join(self._ball_body_viz("e0", i, pivots[i], 0.0, 0.0) for i in range(N))
    return f"<mujoco>{opt}{default}{self._viz_visual()}{self._viz_assets()}<worldbody>{frame}{balls}</worldbody></mujoco>"

  def export_classic(self):
    """CLASSIC Newton's cradle (no optimization): lift the left end ball classic_lift_deg and release --
    the pulse clacks one-in/one-out down the touching chain, the far ball swings out and back, and it
    oscillates for several bounces over classic_secs. Exports ONE cradle to a self-contained USD."""
    a = self.args
    out_dir = os.path.join(ASSETS, "cradle_render")
    cdt = 1e-4  # fine step so the stiff collision is time-resolved (Codex): coarse dt is what smears it
    pm = mujoco.MjModel.from_xml_string(self.cradle_xml(
      gap=0.0, solref="-8000000 0", solimp="0 0.95 0.001 0.5 2", dt=cdt, ball_margin=-0.0005, ball_gap=0.0005))
    pd = mujoco.MjData(pm)
    mujoco.mj_forward(pm, pd)
    pd.qpos[0] = np.radians(a.classic_lift_deg)  # lift the left end ball outward, then release (qvel stays 0)
    mujoco.mj_forward(pm, pd)
    steps = int(round(a.classic_secs / cdt))
    keep = max(1, int(round((1.0 / a.usd_fps) / cdt)))  # sim steps per output frame
    traj = []
    for s in range(steps + 1):
      if s % keep == 0:
        traj.append(pd.qpos.copy())
      mujoco.mj_step(pm, pd)
    tr = np.array(traj)  # (nf, N)
    vm = mujoco.MjModel.from_xml_string(self._cradle_xml_classic(0.0))  # balls visually TOUCHING (hidden margin does the work)
    assert vm.nq == N, (vm.nq, N)
    out = self.write_usd_trajectory(vm, [q for q in tr], out_dir, name="cradle_traj", fps=a.usd_fps)
    amp = np.degrees(np.abs(tr).max(axis=0))  # per-ball max|angle|: clean cradle -> ends big, middle ~0
    print(f"[classic] lift {a.classic_lift_deg}deg {a.classic_secs}s dt{cdt} {len(traj)}f; per-ball max|deg| "
          f"{np.round(amp, 1)} -> {out}")

  def _roll(self, p, steps):
    """MuJoCo-C rollout of one env (end-ball velocities V0_FIXED, p) for `steps` steps -> qpos (steps+1,
    N). The distinct-lane export rolls out longer than the optimization T so the rebounds reach apex."""
    m = mujoco.MjModel.from_xml_string(self.cradle_xml(**self._phys()))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    d.qvel[:] = 0.0
    if self.args.task == "launch":
      d.qvel[N - 1] = -p  # blue's OUTWARD initial speed: swings up-right, falls back, clacks
    else:
      d.qvel[N - 1] = p
      d.qvel[0] = V0_FIXED
    qs = [d.qpos.copy()]
    for _ in range(steps):
      mujoco.mj_step(m, d)
      qs.append(d.qpos.copy())
    return np.array(qs)

  def _export_distinct(self, history, ks, out_dir):
    """Show all num_envs optimization lanes as DISTINCT animated cradles (a small non-instanced field) --
    the parallel diff-sim spread converging. Each lane replays its env's rollout at the sampled iterations
    (rolled out `viz` steps so the end balls' rebounds reach their apex). Writes cradle_traj.usdc."""
    a = self.args
    ne = a.num_envs
    viz = self._nsteps(viz=True) if (a.stiff or a.task == "launch") else (200 if a.task == "swap_control" else T)
    offsets = self._usd_grid_offsets(ne, a.usd_cols, a.usd_xpitch, a.usd_ypitch)
    frames = []
    for k in ks:
      per = [self._roll(history[k]["v4s"][e], viz) for e in range(ne)]  # each (viz+1, N)
      for t in range(0, viz + 1, a.usd_stride):
        frames.append(np.concatenate([per[e][t] for e in range(ne)]))
      frames.extend([frames[-1]] * a.usd_hold)
    vm = mujoco.MjModel.from_xml_string(self._cradle_xml_multi(offsets, floor=False))  # self-contained rgba USD
    assert vm.nq == ne * N, (vm.nq, ne * N)
    out = self.write_usd_trajectory(vm, [np.asarray(f, dtype=np.float64) for f in frames], out_dir,
                                    name="cradle_traj", fps=a.usd_fps)
    p0 = [round(float(history[ks[0]]["v4s"][e]), 2) for e in range(ne)]
    pN = [round(float(history[ks[-1]]["v4s"][e]), 2) for e in range(ne)]
    print(f"[export] {a.task} DISTINCT {ne} lanes ({a.usd_cols} cols), iters {ks}; params {p0} -> {pN}; "
          f"NF={len(frames)} -> {out}")
    return out

  def export_usd(self, history, best):
    """--export_usd hook: replay THIS optimization's trajectory to a Blender-clean animated USD. Default
    (usd_replicate) samples ONE representative lane's convergence across a big INSTANCED field (one shared
    cradle). usd_replicate=False instead shows all num_envs DISTINCT lanes (the parallel spread converging,
    non-instanced small field)."""
    a = self.args
    out_dir = os.path.join(ASSETS, "cradle_render")
    ks = sorted(set(np.linspace(0, len(history) - 1, a.usd_iters).astype(int).tolist()))  # iteration spread
    if not a.usd_replicate:
      return self._export_distinct(history, ks, out_dir)
    env = 0  # representative lane: env 0 seeds the lowest v4 -> the largest (most legible) convergence
    # LAUNCH optimizes the blue end-ball's INITIAL VELOCITY (from rest at the bottom); re-roll each shown
    # iteration's v4 over the long viz horizon (`_roll`: blue at the BOTTOM, qvel=-v -> swings up, clacks,
    # red launches HIGHER each iteration). The taped history["qpos"] is only T=80 -- too short for red's arc.
    viz = self._nsteps(viz=True) if (a.stiff or a.task == "launch") else T
    if a.task == "launch":  # viz swing is ~11800 stiff steps (dt~1e-4) -> pick stride/fps for 1x @ ~50fps
      phys_dt = self._phys()["dt"]
      stride = max(1, round(0.02 / phys_dt))  # output frame = 0.02s of sim
      fps = max(1, round(1.0 / (phys_dt * stride)))
    else:
      stride, fps = a.usd_stride, a.usd_fps
    frames = []
    frame_iters = []  # frame -> GD iteration it shows (the banner subtext the render side can't derive)
    for k in ks:
      qp = self._roll(history[k]["v4s"][env], viz) if a.task == "launch" else history[k]["qpos"][env]  # (viz+1, N)
      sub = list(qp[::stride]) + [qp[-1]] * a.usd_hold  # subsample + brief settled-end hold
      frames.extend(sub)
      frame_iters.extend([int(k)] * len(sub))
    pm = mujoco.MjModel.from_xml_string(self._cradle_xml_proto())  # one animated cradle at the origin
    assert pm.nq == N, (pm.nq, N)
    offsets = self._usd_grid_offsets(a.usd_envs, a.usd_cols, a.usd_xpitch, a.usd_ypitch)
    out = self.export_field(pm, [np.asarray(f, dtype=np.float64) for f in frames], offsets, out_dir,
                            name="cradle_traj", fps=fps, frame_iters=frame_iters, opt_label="init velocity")
    print(f"[export] optim convergence env{env}: iters {ks}, "
          f"v4 {[round(float(history[k]['v4s'][env]), 2) for k in ks]}, "
          f"loss {float(history[ks[0]]['losses'][env]):.3f}->{float(history[ks[-1]]['losses'][env]):.3f}; "
          f"NF={len(frames)}  {pm.ngeom}-geom proto x {a.usd_envs} instances -> {out}")

  def render(self, history, best, out):
    a = self.args
    label = "ADJOINT (cradle)" if a.grad == "analytic" else "FINITE DIFF (cradle)"
    live = True if a.live else None  # None -> honor MJW_VIEWER
    written = self.render_video(history, best, out, cols=a.cols, spacing=a.env_spacing, label=label, live=live)
    if written:  # mp4 written (video mode); skip montage when showing the live viewer
      self.save_montage(written, os.path.splitext(written)[0] + "_montage.png")

  def render_video(self, history, best, out_path, cols=2, spacing=2.2, sample_every=4,
                   label="ADJOINT (cradle)", live=None):
    """Animate every cradle swinging together for selected iterations; trace each env's tracked-ball
    path colored by its loss via the shared Bourke map, with prior iterations' paths persisting faintly
    so the parallel fan visibly converges. mp4, or the live MuJoCo viewer when `live=True` /
    `MJW_VIEWER=1` (via viz.emit). Launch: the rollout is the full ~7800-step swing (not the T=80 default),
    the tracked ball is RED (its launch arc), and both colour and banner use red's ACTUAL apex height
    (z.max-z.min of the rendered arc), so hotter/bigger-angle = higher launch."""
    num_envs = history[0]["mid_xyz"].shape[0]
    offsets = self._env_offsets(num_envs, cols, spacing)
    launch = self.args.task == "launch"
    n_steps = history[0]["qpos"].shape[1] - 1  # actual rollout length (T for the default tasks; ~7800 for launch)
    stride = max(sample_every, n_steps // 90) if launch else sample_every  # cap frames/iter on the long launch swing
    tstride = max(1, n_steps // 160)  # subsample the drawn arc so the polyline stays light

    def _red_peak(hk):  # per-env red apex RISE (m) = z.max - z.min over the rendered arc -- the honest peak,
      mx = hk["mid_xyz"]  # NOT rollout_env's end-of-swing display loss (which reads ~0 once red falls back)
      return mx[:, :, 2].max(axis=1) - mx[:, :, 2].min(axis=1)

    if launch:  # color each lane by red's ACTUAL apex height (hotter = higher launch), 0 -> best
      hi = max(float(_red_peak(h).max()) for h in history) or 1.0

      def _iter_colors(hk):
        return [R.bourke_color_map(0.0, hi, float(v)) for v in _red_peak(hk)]
    else:
      hi = float(max(h["losses"].max() for h in history)) or 1.0

      def _iter_colors(hk):
        return [R.bourke_color_map(0.0, hi, float(v)) for v in hk["losses"]]

    vm = mujoco.MjModel.from_xml_string(self._cradle_xml_multi(offsets))
    vd = mujoco.MjData(vm)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(vm, cam)
    rows = int(np.ceil(num_envs / cols))
    ext_x = spacing * (cols - 1) + (PIVOT_X[-1] - PIVOT_X[0])
    ext_y = spacing * (rows - 1)
    cam.lookat = [0.5 * (PIVOT_X[0] + PIVOT_X[-1]), 0.0, 0.2]  # grid is centered on 0
    cam.distance = max(2.8, 1.1 * max(ext_x, ext_y) + 2.2)
    cam.azimuth = 60.0  # 3/4 view so the grid's depth (rows) reads, not a flat row
    cam.elevation = -22.0

    best_mean = min(h["losses"].mean() for h in history)
    frames = []
    persisted = []  # (mid_xyz_all[B,T+1,3], colors[B]) of completed shown iterations
    for k in R.default_show(len(history), best):
      hk = history[k]
      cols_ = _iter_colors(hk)
      if launch:  # apex angle (deg) from red's ACTUAL rendered arc: z-rise -> th = arccos(1 - h/L)
        deg = np.degrees(np.arccos(np.clip(1.0 - _red_peak(hk) / L, -1.0, 1.0)))
        bdeg = np.degrees(np.arccos(np.clip(1.0 - _red_peak(history[best]).mean() / L, -1.0, 1.0)))
        sub = (
          f"iter {hk['it']:3d}    mean red-apex {deg.mean():.1f}deg    "
          f"best {bdeg:.1f}deg    envs {num_envs}"
        )
      else:
        sub = (
          f"iter {hk['it']:3d}    mean loss {hk['losses'].mean():.4f}    "
          f"best {best_mean:.4f}    envs {num_envs}"
        )
      steps_idx = list(range(0, n_steps + 1, stride)) + [n_steps]
      hold = 20 if k == best else 0
      for t in steps_idx + [n_steps] * hold:
        snap = list(persisted)  # snapshot of prior shown iterations at this frame
        cur = [hk["mid_xyz"][e, : t + 1 : tstride] + offsets[e] for e in range(num_envs)]
        qpos = hk["qpos"][:, t, :].reshape(-1).copy()  # env-major: [env0 h0..h4, env1 ...]

        def draw(scene, snap=snap, cur=cur, cols=cols_):
          for mid_all, pcols in snap:  # prior shown iterations stay on screen, faint
            for e in range(num_envs):
              R.add_polyline(scene, mid_all[e, ::tstride] + offsets[e], pcols[e], width=0.006)
          for e in range(num_envs):  # this iteration: each lane's tracked-ball trail (in its lane)
            R.add_polyline(scene, cur[e], cols[e], width=0.016)

        frames.append((qpos, draw, sub))
      persisted.append((hk["mid_xyz"], cols_))
    if frames:
      frames += [frames[-1]] * 20

    return R.emit(vm, vd, cam, frames, out_path=out_path, label=label, w=W, h=H, fps=30, live=live)

  def save_montage(self, mp4_path, png_path, ncols=5):
    """Sample frames from the rendered mp4 into a grid PNG (for quick inline inspection)."""
    import imageio.v2 as imageio
    from PIL import Image

    rd = imageio.get_reader(mp4_path)
    frames = [f for f in rd]
    rd.close()
    idx = np.linspace(0, len(frames) - 1, 10).astype(int)
    sel = [frames[i] for i in idx]
    h, w, _ = sel[0].shape
    nrows = int(np.ceil(len(sel) / ncols))
    grid = Image.new("RGB", (ncols * w, nrows * h), (0, 0, 0))
    for k, f in enumerate(sel):
      grid.paste(Image.fromarray(f), ((k % ncols) * w, (k // ncols) * h))
    grid.save(png_path)
    print(f"[cradle optim] montage -> {png_path}")


def main(argv):
  del argv  # unused; config comes from the absl-parsed Args
  args = demo.parse_args(Args)
  if args.classic:  # no optimization: a plain multi-bounce Newton's cradle -> self-contained USD
    CradleDemo(args).export_classic()
    return
  demo.run(CradleDemo, args)


if __name__ == "__main__":
  demo.define_flags(Args)
  app.run(main)
