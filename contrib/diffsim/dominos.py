"""Domino spiral chain reaction -- optimize the FIRST domino's forward tipping velocity so the whole
spiral (and in particular the LAST domino) topples, by analytic-adjoint gradient descent through
differentiable mujoco_warp.

A row of thin box dominoes stands on a frictional floor, placed along an ARCHIMEDEAN SPIRAL (ported from
the Newton `domino_spiral` example) and yaw-rotated so each faces the next. Domino 0 is given a single
scalar initial tipping velocity `push`: we set its free-joint LOCAL-x angular velocity qvel0[3] = -push,
which pitches its top along the spiral tangent toward its neighbor. We OPTIMIZE `push` so the tipping
domino knocks its neighbor, which knocks the next, ... all the way to the last one. The variable is
CONSTRAINED push >= PUSH_MIN > 0 (projected each Adam step) so the first domino always falls FORWARD.

  loss = mean_i (z_i / z_stand)^2  +  w_effort * push^2

The first term is a DENSE uprightness objective: each domino's COM height z_i drops from z_stand (upright)
to ~half-thickness (flat). The small effort term makes the optimum INTERIOR. The last domino only moves
THROUGH the domino<->domino contact chain, so d(loss)/d(push) exercises the body<->body contact d(qpos)
adjoint (like cradle.py), here compounded by free-body toppling on RESTING floor contact.

`push` is a wp.array leaf scattered into datas[0].qvel by a kernel INSIDE the tape, so the grad lands on
push.grad and warp.optim.Adam (SHAC betas 0.7, 0.95) steps it in place (demo.Example Style-D). The
`--grad` flag picks analytic (default; also compared to FD every 10 iters as a contact-adjoint TEST BED)
or fd (batched central difference).

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/dominos.py
  uv run --active python contrib/diffsim/dominos.py --grad=fd --no_render          # quick baseline
"""

import math
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
import viz as R  # noqa: E402

N_DOM = 16  # dominoes along the spiral
HX, HY, HZ = 0.06, 0.016, 0.18  # half extents: wide (local x), thin/topple (local y), tall (local z)
MASS = 0.2
MU = 1.0
SPIRAL_INNER_RADIUS = 0.35
SPIRAL_PITCH = 0.32  # radial growth per full turn
DOMINO_SPACING = 0.12  # arc length between neighbors (< 2*HZ so a falling domino reaches the next)
DT, T = 0.005, 340  # 1.7 s rollout -- long enough for the wavefront to traverse the spiral
SOLREF, SOLIMP = "0.01 1", "0.9 0.95 0.001"  # moderately stiff contact (dominoes must not sink while resting)
Z_STAND = HZ  # upright COM height; a toppled domino rests at ~HY -> uprightness term ~(HY/HZ)^2 when flat
PUSH_MIN, PUSH_MAX = 0.1, 30.0  # forward-only constraint (push>0) + a sane upper clamp
W_EFFORT = 2.0e-4  # effort weight -> interior optimum (minimal sufficient shove)
ENV_COLS = 2  # multi-env render grid width
W, H = 1024, 768


@dataclass
class Args(demo.CommonArgs):
  """dominos config: CommonArgs (grad/num_envs/spread/device/capture/out/live/...) + dominos defaults."""

  num_envs: int = field(default=32, metadata={"help": "parallel spirals, each a DIFFERENT fanned push -> distinct topples tiled across the field"})
  spread: float = field(default=0.45, metadata={"help": "per-env initial-push fan (multi-env only)"})
  steps: int = field(default=45, metadata={"help": "Adam steps"})
  lr: float = field(default=0.6, metadata={"help": "Adam learning rate"})
  # dense hero field for the immersed render (render_blender.hero_cam): 15x15 spirals at 2.2m spacing so the
  # pushed-in camera shows a big foreground hero spiral + a field cropping off the frame edges.
  usd_envs: int = field(default=225, metadata={"help": "USD field: instanced lanes (15x15 dense field)"})
  usd_cols: int = field(default=15, metadata={"help": "USD field: grid columns"})
  usd_xpitch: float = field(default=2.2, metadata={"help": "USD field: column pitch (x) = spiral spacing"})
  usd_ypitch: float = field(default=2.2, metadata={"help": "USD field: row pitch (y)"})


@wp.kernel
def _set_push(push: wp.array(dtype=float), qvel: wp.array2d(dtype=float)):
  """Set domino 0's free-joint LOCAL-x angular velocity qvel[3] = -push (NEGATIVE tips forward along the
  spiral tangent). @wp.kernel stays module-level -- Warp JITs it by module path."""
  w = wp.tid()  # one thread per env
  qvel[w, 3] = -push[w]


@wp.kernel
def _accum_upright(qpos: wp.array2d(dtype=float), inv: float, loss: wp.array(dtype=float)):
  """loss += (1/E) * mean_i (z_i / Z_STAND)^2 -- dense uprightness (0 = whole spiral flat)."""
  w, i = wp.tid()  # (env, domino); launch dim=(E, n_dom)
  z = qpos[w, 7 * i + 2] / float(Z_STAND)
  wp.atomic_add(loss, 0, z * z * inv)


@wp.kernel
def _accum_effort(push: wp.array(dtype=float), w_effort: float, loss: wp.array(dtype=float)):
  """loss += w_effort * push_w^2 -- small penalty so the optimum is the minimal forward shove."""
  w = wp.tid()
  wp.atomic_add(loss, 0, w_effort * push[w] * push[w])


class DominosDemo(demo.Example):
  """Domino spiral: optimize domino 0's forward tipping velocity `push` (wp.array leaf) so the whole
  spiral topples. Style-D: `push` scatters into datas[0].qvel via _set_push inside the tape, grad lands
  on push.grad, warp.optim.Adam(betas=0.7,0.95) steps it in place; projected to [PUSH_MIN, PUSH_MAX]."""

  Args = Args
  scatter_params = True  # Style-D: self.param (push) scatters into datas[0].qvel via a kernel

  def optimize(self):
    return self.optimize_adam(betas=(0.7, 0.95))

  # ---- harness hooks ----

  def build_model(self):
    self.mjm = mujoco.MjModel.from_xml_string(self.domino_xml(N_DOM))
    self.mjd = mujoco.MjData(self.mjm)
    mujoco.mj_forward(self.mjm, self.mjd)

  def init_params(self):
    return self._init_push(self.args.num_envs, self.args.spread)  # (E,) per-env initial push

  def build_datas(self):
    ne = self.args.num_envs
    self.nT = T  # checkpointed BPTT length (chunk+1 segment buffers)
    self.datas = [self._put(ne) for _ in range(self.args.chunk + 1)]
    for d in self.datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    self.datas[0].qvel = wp.zeros((ne, self.mjm.nv), dtype=float, requires_grad=True)
    self.param = wp.array(self.params.astype(np.float32), dtype=float, requires_grad=True)  # the push leaf
    self.loss = wp.zeros(1, dtype=float, requires_grad=True)

  def set_params(self):
    # scatter the push leaf into domino-0's free-joint angular velocity (differentiable; re-taped for the
    # scatter backprop -> self.param.grad). Non-taped in the forward pass; taped in backward().
    wp.launch(_set_push, dim=self.args.num_envs, inputs=[self.param], outputs=[self.datas[0].qvel])

  def chunk_step(self, i, t):
    del t  # terminal loss -> time-independent chunk
    mjw.step(self.m, self.datas[i], self.datas[i + 1])

  def terminal_loss(self):
    # uprightness of the FINAL state (datas[0] == state_T) + a push-effort regularizer read straight from
    # self.param. The terminal mini-tape lands d(upright)/dstate_T (state-adjoint seed) AND d(effort)/dparam
    # (direct) on self.param.grad; the scatter backprop then adds d(upright)/dparam via the state path.
    inv = 1.0 / float(N_DOM)  # per-env mean over dominoes; summed over envs by the atomic_add
    wp.launch(_accum_upright, dim=(self.args.num_envs, N_DOM), inputs=[self.datas[0].qpos, inv], outputs=[self.loss])
    wp.launch(_accum_effort, dim=self.args.num_envs, inputs=[self.param, W_EFFORT], outputs=[self.loss])

  def read_grad(self):
    return self.param.grad.numpy()  # d(upright+effort)/dpush

  def fd_step(self, pnp):
    qpos = self.rollout(pnp)
    return self.fd_grad_batched(pnp), float(self._per_env_loss(qpos[:, -1], pnp).sum()), qpos

  def record(self, it, loss, qpos, pnp, g):
    per_env = self._per_env_loss(qpos[:, -1], pnp)
    fallen = self._fallen(qpos[:, -1])
    rec = {"it": it, "loss": loss, "qpos": qpos, "push": pnp, "per_env": per_env, "fallen": fallen}
    if self.args.grad == "analytic" and (it % 10 == 0 or it == self.args.steps - 1):  # analytic-vs-FD test-bed
      fd = self.fd_grad_batched(pnp)
      rec["cos"] = float(g @ fd / (np.linalg.norm(g) * np.linalg.norm(fd) + 1e-12))
      rec["ratio"] = float(np.linalg.norm(g) / (np.linalg.norm(fd) + 1e-12))
    return rec

  def progress(self, rec, g):
    tag = f" cos={rec['cos']:.3f} ratio={rec['ratio']:.2f}" if "cos" in rec else ""
    return (f"  [{rec['it']:3d}] loss={rec['loss']:.4f} push={np.round(rec['push'], 2).tolist()} "
            f"fallen={rec['fallen'].tolist()}/{N_DOM} |g|={np.linalg.norm(g):.3g}{tag}")

  def summary(self, history, best):
    hb = history[best]
    return (f"[dominos x{self.args.num_envs}env/{self.args.grad}] loss {history[0]['loss']:.4f} -> "
            f"best {hb['loss']:.4f} (iter {best}); best fallen {hb['fallen'].tolist()}/{N_DOM}, "
            f"push {np.round(hb['push'], 2).tolist()}")

  def project(self):
    self.param.assign(np.clip(self.param.numpy(), PUSH_MIN, PUSH_MAX).astype(np.float32))  # forward-only

  def default_out(self):
    return os.path.join(os.path.dirname(__file__), "reports", "assets", "dominos.mp4")

  # ---- dominos-specific helpers (spiral scene + fd path) ----

  def _spiral_pose(self, index):
    """Archimedean-spiral pose of domino `index` (ported from the Newton domino_spiral example):
    (x, y, yaw). Neighbors are DOMINO_SPACING apart in ARC LENGTH; yaw faces the next along the tangent."""
    b = SPIRAL_PITCH / (2.0 * math.pi)
    theta = 0.0
    for _ in range(index):
      r = SPIRAL_INNER_RADIUS + b * theta
      theta += DOMINO_SPACING / math.sqrt(r * r + b * b)
    r = SPIRAL_INNER_RADIUS + b * theta
    x, y = r * math.cos(theta), r * math.sin(theta)
    tx = b * math.cos(theta) - r * math.sin(theta)  # spiral tangent
    ty = b * math.sin(theta) + r * math.cos(theta)
    yaw = math.atan2(-tx, ty)  # local +y (thin/topple axis) points along the tangent, toward the next
    return x, y, yaw

  def _poses(self, num_dom=N_DOM):
    out = []
    for i in range(num_dom):
      x, y, yaw = self._spiral_pose(i)
      q = (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))  # rotation about world z
      out.append((x, y, q))
    return out

  def _scene(self, num_dom=N_DOM):
    """Spiral centroid (cx, cy) and half-extent (for framing the camera / spacing the multi-env grid)."""
    xy = np.array([(x, y) for x, y, _ in self._poses(num_dom)])
    cx, cy = xy.mean(0)
    ext = float(np.abs(xy - [cx, cy]).max()) + 2.0 * HZ
    return float(cx), float(cy), ext

  def domino_xml(self, num_dom=N_DOM):
    """Single-world physics scene: a frictional floor + `num_dom` upright box dominoes on the spiral, all
    in the same collision lane so each hits BOTH the floor and its neighbors. put_data replicates it."""
    opt = ('<option timestep="%g" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
           'solver="Newton" iterations="50"><flag eulerdamp="disable"/></option>' % DT)
    default = f'<default><geom condim="3" friction="{MU:g} 0.005 0.0001" solref="{SOLREF}" solimp="{SOLIMP}"/></default>'
    floor = '<geom name="floor" type="plane" size="5 5 0.01" rgba="0.3 0.3 0.35 1"/>'
    bodies = ""
    for i, (x, y, (qw, qx, qy, qz)) in enumerate(self._poses(num_dom)):
      t = i / max(num_dom - 1, 1)
      rgba = f"{0.9 * (1 - t) + 0.2 * t:.3f} {0.25 + 0.55 * t:.3f} {0.85 * (1 - t) + 0.2 * t:.3f} 1"
      bodies += (
        f'<body name="d{i}" pos="{x:.4f} {y:.4f} {HZ:.4f}" quat="{qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f}">'
        f'<freejoint/><geom name="d{i}" type="box" size="{HX} {HY} {HZ}" mass="{MASS}" rgba="{rgba}"/></body>'
      )
    return f'<mujoco>{opt}{default}<worldbody>{floor}{bodies}</worldbody></mujoco>'

  def _per_env_loss(self, qpos_final, push):
    """Per-env loss from a numpy final qpos (E, 7*n_dom): dense uprightness + effort. Matches the kernels."""
    z = qpos_final[:, 2::7][:, :N_DOM] / Z_STAND  # (E, n_dom) COM heights, normalized
    return (z * z).mean(axis=1) + W_EFFORT * push ** 2

  def _fallen(self, qpos_final):
    """Per-env count of dominoes whose COM dropped below 60% of standing height (toppled)."""
    z = qpos_final[:, 2::7][:, :N_DOM]
    return (z < 0.6 * Z_STAND).sum(axis=1)

  def _scatter_push(self, push):
    qv = np.zeros((self.args.num_envs, self.mjm.nv), np.float32)
    qv[:, 3] = -push  # domino 0 local-x pitch (negative = forward), matches _set_push
    return qv

  def _put(self, num_envs):
    """put_data with contact headroom for the whole spiral: each domino touches the floor + up to 2 neighbors."""
    ncon = num_envs * (16 * N_DOM + 64)
    return mjw.put_data(self.mjm, self.mjd, nworld=num_envs, nconmax=ncon, njmax=6 * ncon)

  def rollout(self, push):
    """Batched MuJoCo-warp forward rollout with domino 0 pitched by `push` (E,). Returns (E, T+1, nq)."""
    d = self._put(self.args.num_envs)
    d.qvel = wp.array(self._scatter_push(push), dtype=float)
    qs = [d.qpos.numpy().copy()]
    for _ in range(T):
      mjw.step(self.m, d)
      qs.append(d.qpos.numpy().copy())
    return np.transpose(np.array(qs), (1, 0, 2))

  def fd_grad_batched(self, push, eps=1e-2):
    """Batched central difference d(loss_w)/d(push_w): envs are independent worlds, so perturbing every
    env's push at once and reading per-env loss gives all per-env gradients in TWO batched rollouts."""
    lp = self._per_env_loss(self.rollout(push + eps)[:, -1], push + eps)
    lm = self._per_env_loss(self.rollout(push - eps)[:, -1], push - eps)
    return (lp - lm) / (2 * eps)

  def _init_push(self, num_envs, spread):
    """Per-env initial push (rad/s): the WHOLE POINT is to DISCOVER the velocity that completes the spiral,
    so iteration 0 must barely start it -- the seed topples just ONE domino and the optimizer must climb to
    the ~7 rad/s that finishes the whole chain. Kept just above the stall boundary so there is always
    signal. --num_envs 1 -> a single seed."""
    if num_envs == 1:
      return np.array([2.3], np.float64)  # push=2.3 -> only 1/16 falls: the chain barely gets going
    lo, hi = 2.4, 2.6  # every lane starts partial (~4/16 .. ~6/16), none completes the spiral
    mid = 0.5 * (lo + hi)
    return np.clip(np.linspace(mid - (mid - lo) * spread / 0.45, mid + (hi - mid) * spread / 0.45, num_envs), lo, hi)

  # ---- rendering: the domino spiral (a grid of spirals if multi-env), last-domino path traced ----

  def _env_offsets(self, num_envs, cols=ENV_COLS):
    off = np.zeros((num_envs, 3))
    if num_envs > 1:
      pitch = 2.0 * self._scene()[2] + 0.3  # > spiral diameter so lanes do not overlap
      rows = int(np.ceil(num_envs / cols))
      for e in range(num_envs):
        r, c = e // cols, e % cols
        off[e, 0] = (c - (cols - 1) / 2.0) * pitch
        off[e, 1] = (r - (rows - 1) / 2.0) * pitch
    return off

  def viz_xml_multi(self, offsets):
    """Viz scene: N decorative free bodies per env (contype=0), placed each frame from the rollout qpos.
    Env-major body order (env outer, domino inner) matches the flattened batched qpos; only mj_forward'd."""
    opt = '<option timestep="%g" gravity="0 0 -9.81"/>' % DT
    bodies = ""
    for w in range(len(offsets)):
      for i in range(N_DOM):
        t = i / max(N_DOM - 1, 1)
        rgba = f"{0.9 * (1 - t) + 0.2 * t:.3f} {0.25 + 0.55 * t:.3f} {0.85 * (1 - t) + 0.2 * t:.3f} 1"
        bodies += (
          f'<body name="e{w}d{i}" pos="0 0 {HZ:.4f}"><freejoint/>'
          f'<geom type="box" size="{HX} {HY} {HZ}" mass="{MASS}" contype="0" conaffinity="0" rgba="{rgba}"/></body>'
        )
    fsize = max(3.0, float(np.abs(offsets).max()) + 2.0 * self._scene()[2] + 1.0)
    return f"""
<mujoco>
  {opt}
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/><global offwidth="{W}" offheight="{H}"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="12 12" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0.4 -1 2" dir="0 0.4 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="{fsize:.3f} {fsize:.3f} 0.01" material="grid" contype="0" conaffinity="0"/>
    {bodies}
  </worldbody>
</mujoco>
"""

  def export_usd(self, history, best):
    """--export_usd hook: replay THIS optimization's convergence across an instanced spiral FIELD for the
    Blender render (see dominos_render_blender.py). Samples a spread of iterations; for each, re-rolls the
    stored `push` through the batched sim (dominos' analytic history["qpos"] is only chunk+1 frames, so we
    re-roll the full topple), subsamples + holds the settled end, and tracks frame->iteration. The proto is
    one named-geom spiral with no floor/lights (strip_to_proto); the field replicates lane 0's convergence."""
    a = self.args
    out_dir = os.path.join(os.path.dirname(__file__), "reports", "assets", "dominos_render")
    ks = sorted(set(np.linspace(0, len(history) - 1, a.usd_iters).astype(int).tolist()))
    ne = self.args.num_envs
    qps_all = {k: self.rollout(history[k]["push"]) for k in ks}  # (ne, T+1, nq) per shown iter (all envs)
    env_frames, frame_iters = [], None  # each env a DIFFERENT fanned push -> distinct topple -> diverse field
    for e in range(ne):
      fr, fi = [], []
      for k in ks:
        qp = qps_all[k][e]  # (T+1, nq=7*N_DOM): env e's topple at this iteration
        sub = list(qp[:: a.usd_stride]) + [qp[-1]] * a.usd_hold
        fr.extend(sub)
        fi.extend([int(k)] * len(sub))
      env_frames.append([np.asarray(f, np.float64) for f in fr])
      frame_iters = fi
    proto = self.strip_to_proto(self.domino_xml(N_DOM), from_string=True)  # named d{i} boxes, no floor (shared geometry)
    assert proto.nq == env_frames[0][0].shape[0], (proto.nq, env_frames[0][0].shape[0])
    offsets = self._usd_grid_offsets(a.usd_envs, a.usd_cols, a.usd_xpitch, a.usd_ypitch)
    fps = max(1, round(1.0 / (self.mjm.opt.timestep * a.usd_stride)))  # 1x real-time (was fps=usd_fps=30 -> 0.6x bug)
    out = self.export_field(proto, None, offsets, out_dir, name="dominos_traj", fps=fps,
                            frame_iters=frame_iters, opt_label="init velocity", env_frames=env_frames)
    fallen = [int(history[ks[-1]]["fallen"][e]) for e in range(min(ne, 8))]
    print(f"[export] dominos: {ne} DISTINCT envs, iters {ks}; final fallen(first8) {fallen}; "
          f"NF={len(env_frames[0])} -> {out}")

  def render(self, history, best, out):
    num_envs = self.args.num_envs
    offsets = self._env_offsets(num_envs)
    cx, cy, ext = self._scene()
    vm = mujoco.MjModel.from_xml_string(self.viz_xml_multi(offsets))
    vd = mujoco.MjData(vm)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(vm, cam)
    cam.lookat = [cx, cy, 0.05]
    gridext = float(np.abs(offsets).max()) if num_envs > 1 else 0.0
    cam.distance = 2.2 + 2.0 * ext + 1.6 * gridext
    cam.azimuth = 90.0
    cam.elevation = -55.0  # near top-down so the spiral reads

    last = N_DOM - 1
    hi = max(float(max(h["per_env"].max() for h in history)), 1e-9)

    def flat_qpos(hk, t):
      q = np.zeros(vm.nq)
      for w in range(num_envs):
        for i in range(N_DOM):
          s = (w * N_DOM + i) * 7
          q[s:s + 7] = hk["qpos"][w, t, 7 * i:7 * i + 7]
          q[s:s + 3] += offsets[w]  # shift this env's spiral into its grid cell
      return q

    frames, persisted = [], []
    for k in R.default_show(len(history), best):
      hk = history[k]
      cols = [R.bourke_color_map(0.0, hi, float(hk["per_env"][w])) for w in range(num_envs)]
      paths = [hk["qpos"][w, :, 7 * last:7 * last + 3] + offsets[w] for w in range(num_envs)]  # last-domino path
      hold = 20 if k == best else 0
      sub = (f"iter {hk['it']:3d}    mean loss/env {hk['loss'] / num_envs:.4f}    "
             f"fallen {hk['fallen'].tolist()}/{N_DOM}    envs {num_envs}")
      for t in list(range(0, T + 1, 4)) + [T] * (1 + hold):
        snap, cur = list(persisted), [p[: t + 1] for p in paths]

        def draw(scene, snap=snap, cur=cur, cols=cols):
          for pr, cr in snap:
            for pth, cc in zip(pr, cr):
              R.add_polyline(scene, pth, cc, width=0.006)
          for pth, cc in zip(cur, cols):
            R.add_polyline(scene, pth, cc, width=0.016)

        frames.append((flat_qpos(hk, t), draw, sub))
      persisted.append((paths, cols))
    if frames:
      frames += [frames[-1]] * 20
    label = f"{'ADJOINT' if self.args.grad == 'analytic' else 'FINITE DIFF'} (dominos)"
    live = True if self.args.live else None
    written = R.emit(vm, vd, cam, frames, out_path=out, label=label, w=W, h=H, live=live)
    if written:
      self.save_montage(written, os.path.splitext(written)[0] + "_montage.png")

  def save_montage(self, mp4_path, png_path):
    import imageio.v2 as imageio
    from PIL import Image
    rd = imageio.get_reader(mp4_path)
    fr = [f for f in rd]
    rd.close()
    sel = [fr[i] for i in np.linspace(0, len(fr) - 1, 10).astype(int)]
    hh, ww, _ = sel[0].shape
    grid = Image.new("RGB", (5 * ww, 2 * hh), (0, 0, 0))
    for kk, f in enumerate(sel):
      grid.paste(Image.fromarray(f), ((kk % 5) * ww, (kk // 5) * hh))
    grid.save(png_path)
    print(f"[dominos] montage -> {png_path}")


def main(argv):
  del argv  # unused; config comes from the absl-parsed Args
  demo.run(DominosDemo, demo.parse_args(Args))


if __name__ == "__main__":
  demo.define_flags(Args)
  app.run(main)
