"""Frictional slide-to-target, driven by the analytic adjoint gradient + Adam (works NOW).

Pucks are launched across a flat floor; Coulomb friction decelerates each to a stop. We optimize each
puck's 2-D LAUNCH VELOCITY qvel0[:2] so it stops on its OWN target -- a STATE gradient d(loss)/d(qvel0)
through sustained frictional contact on a single free body (nq=7, nv=6 per puck).

  --geom box | cylinder | both   1 box, 1 cylinder (flat disk), or a box + cylinder (separate lanes).
  --num_envs N                   N PARALLEL envs (batched nworld=N), each converging to its own fanned
                                 target -- optimized together in ONE tape.backward (like bounce/cradle).

Data-parallel over BOTH count axes: the scene is a LIST of object specs (max 2) and the physics is
batched over nworld; the set-velocity / squared-error kernels launch over dim=(nworld, nobj) -- one
thread per (env, puck), the error kernel wp.atomic_add's into the shared loss. One wp.Tape over the
batched rollout, one tape.backward -> dL/d[all launch velocities], optimized with warp.optim.Adam
(SHAC betas 0.7, 0.95). The launch velocity `v` is a wp.array leaf scattered into datas[0].qvel by a
kernel INSIDE the tape, so the grad lands on v.grad and Adam steps it in place (demo.Example Style-D).

WHY SOFT CONTACT + ADAM: with STIFF contact the discrete stick/slip stop is genuinely NON-SMOOTH -- the
exact gradient develops a REAL spike (FD-confirmed) that flips the descent direction and vanilla GD
explodes on it. SOFT contact (wide solimp) regularizes the stop so the analytic gradient is the smooth
"bowl" gradient (cos=1.0 vs FD); Adam (scale-invariant step) converges.

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/slide.py --geom=both --num_envs=4
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
import viz as R  # noqa: E402

DT, T = 0.004, 150
MU = 0.7
SOLIMP, SOLREF = "0 0.9 0.02", "0.02 1"  # SOFT contact: wide impedance ramp regularizes the stick/slip stop
W, H = 1024, 768
ENV_COLS, ENV_SPACING = 2, 1.5  # multi-env render grid: columns and pitch (> scene extent)
_HALF_H = 0.012  # puck half-height (radius 0.045 >> this -> stays flat/stable: cylinder axis || normal, the
# len_sqr=0 degenerate case that exercises the adjoint.py wp.sqrt fix. Thick enough to render above the rings.
_SHAPE = {"box": f'type="box" size="0.045 0.045 {_HALF_H}"', "cylinder": f'type="cylinder" size="0.045 {_HALF_H}"'}
_RGBA = {"box": "0.95 0.75 0.25 1", "cyl": "0.35 0.65 0.95 1"}
_RINGS = [(0.090, 0.0020, "0.85 0.12 0.12 1"), (0.068, 0.0026, "0.97 0.97 0.97 1"),
          (0.046, 0.0032, "0.85 0.12 0.12 1"), (0.024, 0.0038, "0.97 0.97 0.97 1"), (0.009, 0.0044, "0.85 0.12 0.12 1")]


@dataclass
class Args(demo.CommonArgs):
  """slide config: CommonArgs (grad/num_envs/spread/device/capture/out/live/...) + slide defaults."""

  geom: str = field(default="box", metadata={"help": "puck shape", "choices": ["box", "cylinder", "both"]})
  num_envs: int = field(default=32, metadata={"help": "parallel envs, each a DIFFERENT fanned target -> distinct trajectories tiled across the field"})
  spread: float = field(default=0.5, metadata={"help": "per-env target fan half-angle (rad)"})
  steps: int = field(default=200, metadata={"help": "Adam steps"})
  lr: float = field(default=0.1, metadata={"help": "Adam learning rate"})
  # dense hero field for the immersed render (render_blender.hero_cam): 21x21 puck tasks at 1.6m spacing --
  # wider than the ~0.7m slide so adjacent envs' targets DON'T overlap, and enough lanes to crop the edges.
  usd_envs: int = field(default=441, metadata={"help": "USD field: instanced lanes (21x21 dense field)"})
  usd_cols: int = field(default=21, metadata={"help": "USD field: grid columns"})
  usd_xpitch: float = field(default=1.6, metadata={"help": "USD field: column pitch (x) = task spacing (no target overlap)"})
  usd_ypitch: float = field(default=1.6, metadata={"help": "USD field: row pitch (y)"})
  usd_stride: int = field(default=5, metadata={"help": "USD field: subsample (dt=0.004 -> stride 5 = 1x@50fps)"})


@wp.kernel
def _set_qvel(v: wp.array(dtype=float), nobj: int, qvel: wp.array2d(dtype=float)):
  w, o = wp.tid()  # (env, puck); launch dim=(nworld, nobj). @wp.kernel stays module-level.
  base = (w * nobj + o) * 2
  qvel[w, 6 * o + 0] = v[base + 0]
  qvel[w, 6 * o + 1] = v[base + 1]


@wp.kernel
def _sum_sq_err(qpos: wp.array2d(dtype=float), targets: wp.array(dtype=wp.vec2), nobj: int, loss: wp.array(dtype=float)):
  w, o = wp.tid()  # (env, puck); each atomic-adds its squared miss to the shared scalar loss
  ti = w * nobj + o
  dx = qpos[w, 7 * o + 0] - targets[ti][0]
  dy = qpos[w, 7 * o + 1] - targets[ti][1]
  wp.atomic_add(loss, 0, dx * dx + dy * dy)


class SlideDemo(demo.Example):
  """Frictional slide-to-target: optimize per-(env,puck) 2-D launch velocity `v` (wp.array leaf) so
  each puck stops on its target. Style-D: `v` scatters into datas[0].qvel via _set_qvel inside the
  tape, grad lands on v.grad, warp.optim.Adam(betas=0.7,0.95) steps it in place."""

  Args = Args
  scatter_params = True  # Style-D: self.param (launch velocity) scatters into datas[0].qvel via a kernel

  def optimize(self):
    return self.optimize_adam(betas=(0.7, 0.95))

  # ---- harness hooks ----

  def build_model(self):
    self.objects = self.objects_for(self.args.geom)
    self.nobj = len(self.objects)
    self.mjm = mujoco.MjModel.from_xml_string(self.slide_xml(self.objects))
    self.mjd = mujoco.MjData(self.mjm)
    mujoco.mj_forward(self.mjm, self.mjd)
    self.targets_np = self.fan_targets(self.objects, self.args.num_envs, self.args.spread)
    self.targets_wp = wp.array(self.targets_np.astype(np.float32), dtype=wp.vec2)

  def init_params(self):
    # per-(env,obj) launch velocity, env-major flat (num_envs*nobj*2,)
    return np.tile(np.concatenate([o["v0"] for o in self.objects]), self.args.num_envs).astype(np.float32)

  def build_datas(self):
    ne = self.args.num_envs
    self.nT = T  # checkpointed BPTT length (chunk+1 segment buffers)
    self.datas = [mjw.put_data(self.mjm, self.mjd, nworld=ne) for _ in range(self.args.chunk + 1)]
    for d in self.datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    self.datas[0].qvel = wp.zeros((ne, self.mjm.nv), dtype=float, requires_grad=True)
    self.param = wp.array(self.params, dtype=float, requires_grad=True)  # the launch-velocity leaf
    self.loss = wp.zeros(1, dtype=float, requires_grad=True)

  def set_params(self):
    # scatter the launch-velocity leaf into datas[0].qvel (differentiable: harness re-tapes it for the
    # scatter backprop -> self.param.grad). Non-taped in the forward pass; taped in backward().
    wp.launch(_set_qvel, dim=(self.args.num_envs, self.nobj), inputs=[self.param, self.nobj], outputs=[self.datas[0].qvel])

  def chunk_step(self, i, t):
    del t  # terminal loss -> time-independent chunk
    mjw.step(self.m, self.datas[i], self.datas[i + 1])

  def terminal_loss(self):
    # per-(env,puck) squared miss to target at the FINAL state (datas[0] == state_T after the forward pass)
    wp.launch(_sum_sq_err, dim=(self.args.num_envs, self.nobj),
              inputs=[self.datas[0].qpos, self.targets_wp, self.nobj], outputs=[self.loss])

  def read_grad(self):
    return self.param.grad.numpy()  # scatter backprop landed dL/dparam on the launch-velocity leaf

  def fd_step(self, pnp):
    qpos = self.rollout(pnp)
    return self.fd_grad(pnp), self.loss_of(qpos[:, -1]), qpos

  def record(self, it, loss, qpos, pnp, g):
    return {"it": it, "loss": loss, "qpos": qpos, "v": pnp}

  def progress(self, rec, g):
    ne = self.args.num_envs
    return f"  [{rec['it']:3d}] total_loss={rec['loss']:.5f} mean/env={rec['loss'] / ne:.5f} |g|={np.linalg.norm(g):.3g}"

  def summary(self, history, best):
    names = "+".join(o["name"] for o in self.objects)
    return (f"[slide {names} x{self.args.num_envs}env/{self.args.grad}] "
            f"total_loss {history[0]['loss']:.5f} -> best {history[best]['loss']:.6f} (iter {best})")

  def default_out(self):
    name = "slide.mp4" if self.args.geom == "box" else f"slide_{self.args.geom}.mp4"
    return os.path.join(os.path.dirname(__file__), "reports", "assets", name)

  # ---- slide-specific helpers (fd path + scene) ----

  def objects_for(self, geom):
    """Base object specs (max 2). box/cylinder = one puck; both = a box + cylinder in separate lanes."""
    if geom in ("box", "cylinder"):
      return [dict(name=geom, shape=_SHAPE[geom], start=np.array([-0.20, -0.10]), target=np.array([0.50, 0.30]),
                   v0=np.array([1.8, 0.7]), rgba=_RGBA["box" if geom == "box" else "cyl"])]
    return [
      dict(name="box", shape=_SHAPE["box"], start=np.array([-0.25, -0.12]), target=np.array([0.45, -0.30]),
           v0=np.array([1.7, -0.5]), rgba=_RGBA["box"]),
      dict(name="cyl", shape=_SHAPE["cylinder"], start=np.array([-0.25, 0.12]), target=np.array([0.50, 0.28]),
           v0=np.array([1.7, 0.5]), rgba=_RGBA["cyl"]),
    ]

  def fan_targets(self, objects, num_envs, spread):
    """Per-(env,obj) target, env-major (index w*nobj+o), shape (num_envs*nobj, 2). Each env rotates every
    object's (target-start) about its start by a fanned angle in [-spread, +spread]; a single env ->
    angle 0 = the base target, so --num_envs 1 matches the single-object demo."""
    nobj = len(objects)
    angles = np.zeros(1) if num_envs == 1 else np.linspace(-spread, spread, num_envs)
    out = np.zeros((num_envs * nobj, 2))
    for w in range(num_envs):
      ca, sa = np.cos(angles[w]), np.sin(angles[w])
      rot = np.array([[ca, -sa], [sa, ca]])
      for o, ob in enumerate(objects):
        out[w * nobj + o] = ob["start"] + rot @ (ob["target"] - ob["start"])
    return out

  def _body(self, o, i):
    # collision lanes: floor (contype=1, conaffinity=1); puck i (contype=1<<(i+1), conaffinity=1) -> hits
    # the floor but NOT the other puck (independent slides). Replicated across nworld by put_data.
    return (f'<body name="{o["name"]}" pos="{o["start"][0]:.3f} {o["start"][1]:.3f} {_HALF_H}"><freejoint/>'
            f'<geom name="{o["name"]}" {o["shape"]} mass="0.5" contype="{1 << (i + 1)}" conaffinity="1" '
            f'friction="{MU:g} 0.005 0.0001" rgba="{o["rgba"]}"/></body>')

  def slide_xml(self, objects):
    """Single-world physics scene (nobj bodies); put_data(nworld=N) replicates it into N envs."""
    opt = ('<option timestep="%g" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
           'solver="Newton" iterations="50"><flag eulerdamp="disable"/></option>' % DT)
    default = f'<default><geom condim="3" solimp="{SOLIMP}" solref="{SOLREF}"/></default>'
    bodies = "".join(self._body(o, i) for i, o in enumerate(objects))
    floor = f'<geom name="floor" type="plane" size="5 5 0.01" contype="1" conaffinity="1" friction="{MU} 0.005 0.0001"/>'
    return f'<mujoco>{opt}{default}<worldbody>{floor}{bodies}</worldbody></mujoco>'

  def _scatter_qvel_np(self, v0):
    ne, nobj, nv = self.args.num_envs, self.nobj, self.mjm.nv
    qv = np.zeros((ne, nv), np.float32)
    for w in range(ne):
      for o in range(nobj):
        qv[w, 6 * o], qv[w, 6 * o + 1] = v0[(w * nobj + o) * 2], v0[(w * nobj + o) * 2 + 1]
    return qv

  def rollout(self, v0):
    """Batched MuJoCo-C forward rollout (fd path). Returns qpos traj (num_envs, T+1, 7*nobj)."""
    d = mjw.put_data(self.mjm, self.mjd, nworld=self.args.num_envs)
    d.qvel = wp.array(self._scatter_qvel_np(v0), dtype=float)
    qs = [d.qpos.numpy().copy()]
    for _ in range(T):
      mjw.step(self.m, d)
      qs.append(d.qpos.numpy().copy())
    return np.transpose(np.array(qs), (1, 0, 2))

  def loss_of(self, qpos_final):
    s = 0.0
    for w in range(self.args.num_envs):
      for o in range(self.nobj):
        p = qpos_final[w, 7 * o:7 * o + 2]
        s += float(np.sum((p - self.targets_np[w * self.nobj + o]) ** 2))
    return s

  def fd_grad(self, v0, eps=1e-3):
    g = np.zeros(v0.shape[0])
    for j in range(v0.shape[0]):
      vp = v0.copy(); vp[j] += eps
      vm = v0.copy(); vm[j] -= eps
      g[j] = (self.loss_of(self.rollout(vp)[:, -1]) - self.loss_of(self.rollout(vm)[:, -1])) / (2 * eps)
    return g

  # ---- rendering (a grid of pucks + bullseye rings, each lane's slide path traced) ----

  def _env_offsets(self, num_envs, cols=ENV_COLS, spacing=ENV_SPACING):
    off = np.zeros((num_envs, 3))
    if num_envs > 1:
      rows = int(np.ceil(num_envs / cols))
      for e in range(num_envs):
        r, c = e // cols, e % cols
        off[e, 0] = (c - (cols - 1) / 2.0) * spacing
        off[e, 1] = (r - (rows - 1) / 2.0) * spacing
    return off

  def viz_xml_multi(self, objects, offsets, targets_np):
    """Viz scene = the puck(s) + bullseye rings REPLICATED per env, offset into a grid. Env-major body
    order (env outer, puck inner) matches the flattened batched qpos. Bodies are decorative (contype=0);
    the viz model is only mj_forward'd to place geoms at each frame's qpos, never stepped."""
    nobj = len(objects)
    opt = ('<option timestep="%g" gravity="0 0 -9.81"/>' % DT)
    bodies, rings = "", ""
    for w, (ox, oy, _oz) in enumerate(offsets):
      for o, ob in enumerate(objects):
        bodies += (f'<body name="e{w}o{o}" pos="0 0 {_HALF_H}"><freejoint/>'
                   f'<geom {ob["shape"]} mass="0.5" contype="0" conaffinity="0" rgba="{ob["rgba"]}"/></body>')
        tx, ty = targets_np[w * nobj + o]
        rings += "".join(
          f'<geom type="cylinder" size="{r} {z}" pos="{tx + ox:.4f} {ty + oy:.4f} {z}" rgba="{c}" '
          f'contype="0" conaffinity="0"/>' for (r, z, c) in _RINGS)
    fsize = max(3.0, float(np.abs(offsets).max()) + 1.5)
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
    <material name="grid" texture="grid" texuniform="true" texrepeat="10 10" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0.4 -1 2" dir="0 0.4 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="{fsize:.3f} {fsize:.3f} 0.01" material="grid" contype="0" conaffinity="0"/>
    {rings}
    {bodies}
  </worldbody>
</mujoco>
"""

  def _proto_xml(self, env=0):
    """Field-render proto: the named puck bodies (original 9cm size) + the red/white BULLSEYE target rings
    per object (env `env`'s target, the same _RINGS the demo's own render draws), NO floor. The render side
    pastels the pucks; the ring geoms keep their red/white rgba (imported), marking each puck's goal so the
    launch-velocity convergence reads across iterations."""
    opt = ('<option timestep="%g" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
           'solver="Newton" iterations="50"><flag eulerdamp="disable"/></option>' % DT)
    default = f'<default><geom condim="3" solimp="{SOLIMP}" solref="{SOLREF}"/></default>'
    bodies = "".join(self._body(o, i) for i, o in enumerate(self.objects))  # named box/cyl, original size, nq=7*nobj
    rings = ""
    for i in range(self.nobj):
      tx, ty = self.targets_np[env * self.nobj + i]
      rings += "".join(
        f'<geom name="target{i}_ring{j}" type="cylinder" size="{r} {z}" pos="{tx:.4f} {ty:.4f} {z}" '
        f'rgba="{c}" contype="0" conaffinity="0"/>' for j, (r, z, c) in enumerate(_RINGS))
    return f'<mujoco>{opt}{default}<worldbody>{rings}{bodies}</worldbody></mujoco>'

  def export_usd(self, history, best):
    """--export_usd hook: replay THIS launch-velocity optimization across an instanced puck FIELD for the
    Blender render (slide_render_blender.py). Samples a spread of iterations; for each, re-rolls the stored
    launch velocity `v` through the batched sim (slide's analytic history["qpos"] is only chunk+1 frames),
    subsamples + holds the settled end, tracking frame->iteration. Proto = named pucks + target discs."""
    a = self.args
    out_dir = os.path.join(os.path.dirname(__file__), "reports", "assets", "slide_render")
    # early-weighted sampling: slide converges by ~iter 39, so UNIFORM sampling wasted 5/6 shown iters on
    # identical converged (blue) rollouts. Sample DENSELY in the converging window [0, cap] (quadratic
    # spacing -> dense early) + always append the final converged iter as the settled reference, so the
    # loss-mapped ribbons read a full red->blue transition instead of 1 red + 5 blue.
    cap = min(len(history) - 1, 50)
    ks = sorted(set((np.linspace(0, 1, max(a.usd_iters - 1, 1)) ** 2 * cap).round().astype(int).tolist()
                    + [len(history) - 1]))
    ne = self.args.num_envs
    qps_all = {k: self.rollout(history[k]["v"]) for k in ks}  # (ne, T+1, 7*nobj) per shown iter (all envs)
    offsets = self._usd_grid_offsets(a.usd_envs, a.usd_cols, a.usd_xpitch, a.usd_ypitch)
    fps = max(1, round(1.0 / (self.mjm.opt.timestep * a.usd_stride)))  # 1x real-time
    # Each env slides toward its OWN fanned target -> a DISTINCT trajectory + target-ring proto -> diverse field.
    env_frames, env_ribbons, env_protos, frame_iters = [], [], [], None
    for e in range(ne):
      tgt = [self.targets_np[e * self.nobj + o] for o in range(self.nobj)]
      qps = {k: qps_all[k][e] for k in ks}
      miss = {k: [float(np.sum((qps[k][-1, 7 * o:7 * o + 2] - tgt[o]) ** 2)) for o in range(self.nobj)] for k in ks}
      hi = max(max(miss[ks[0]]), 1e-9)          # env e's iter-0 worst miss -> its own Bourke normalizer
      fr, fi = [], []
      paths = [[] for _ in range(self.nobj)]
      pathcols = [[] for _ in range(self.nobj)]
      for k in ks:
        cols_k = [R.bourke_color_map(0.0, hi, miss[k][o]) for o in range(self.nobj)]
        sub = list(qps[k][:: a.usd_stride]) + [qps[k][-1]] * a.usd_hold
        fr.extend(sub)
        fi.extend([int(k)] * len(sub))
        for f_qp in sub:
          for o in range(self.nobj):
            paths[o].append([float(f_qp[7 * o]), float(f_qp[7 * o + 1]), 0.02])  # COM xy, lifted above the floor
            pathcols[o].append(cols_k[o])
      env_frames.append([np.asarray(f, np.float64) for f in fr])
      env_ribbons.append([{"pts": np.asarray(paths[o]), "iters": fi, "width": 0.008, "colors": pathcols[o]}
                          for o in range(self.nobj)])
      env_protos.append(self.strip_to_proto(self._proto_xml(e), from_string=True))  # env e's target rings
      frame_iters = fi  # identical schedule across every env
    assert env_protos[0].nq == env_frames[0][0].shape[0], (env_protos[0].nq, env_frames[0][0].shape[0])
    out = self.export_field(env_protos[0], None, offsets, out_dir, name="slide_traj", fps=fps,
                            frame_iters=frame_iters, opt_label="init velocity",
                            env_frames=env_frames, env_ribbons=env_ribbons, env_protos=env_protos)
    print(f"[export] slide: {ne} DISTINCT envs (fanned targets), iters {ks}; "
          f"NF={len(env_frames[0])} + {self.nobj} paths/env -> {out}")

  def render(self, history, best, out):
    objects, targets_np, num_envs = self.objects, self.targets_np, self.args.num_envs
    nobj = self.nobj
    offsets = self._env_offsets(num_envs)
    vm = mujoco.MjModel.from_xml_string(self.viz_xml_multi(objects, offsets, targets_np))
    vd = mujoco.MjData(vm)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(vm, cam)
    base = np.array([o["start"] for o in objects] + [o["target"] for o in objects]).mean(0)
    cam.lookat = [float(base[0]), float(base[1]), 0.05]
    ext = ENV_SPACING * max(ENV_COLS - 1, int(np.ceil(num_envs / ENV_COLS)) - 1) if num_envs > 1 else 0.0
    cam.distance = (2.1 if num_envs == 1 else 2.8) + 1.4 * ext  # zoomed out to fit the grid
    cam.azimuth = 70.0
    cam.elevation = -40.0 if num_envs > 1 else -38.0

    def miss(hk, w, o):
      return float(np.sum((hk["qpos"][w, -1, 7 * o:7 * o + 2] - targets_np[w * nobj + o]) ** 2))

    hi = max(max(miss(history[0], w, o) for w in range(num_envs) for o in range(nobj)), 1e-9)

    def flat_qpos(hk, t):
      q = np.zeros(vm.nq)
      for w in range(num_envs):
        for o in range(nobj):
          s = (w * nobj + o) * 7
          q[s:s + 7] = hk["qpos"][w, t, 7 * o:7 * o + 7]
          q[s:s + 3] += offsets[w]  # shift this env's puck into its grid cell
      return q

    frames, persisted = [], []
    for k in R.default_show(len(history), best):
      hk = history[k]
      cols = [[R.bourke_color_map(0.0, hi, miss(hk, w, o)) for o in range(nobj)] for w in range(num_envs)]
      paths = [[hk["qpos"][w, :, 7 * o:7 * o + 3] + offsets[w] for o in range(nobj)] for w in range(num_envs)]
      hold = 20 if k == best else 0
      sub = f"iter {hk['it']:3d}    mean miss/env {hk['loss'] / num_envs:.4f}    envs {num_envs}"
      for t in list(range(0, T + 1, 4)) + [T] * (1 + hold):
        snap, cur = list(persisted), [[p[: t + 1] for p in row] for row in paths]

        def draw(scene, snap=snap, cur=cur, cols=cols):
          for pr, cr in snap:
            for row_p, row_c in zip(pr, cr):
              for pth, cc in zip(row_p, row_c):
                R.add_polyline(scene, pth, cc, width=0.006)
          for row_p, row_c in zip(cur, cols):
            for pth, cc in zip(row_p, row_c):
              R.add_polyline(scene, pth, cc, width=0.016)

        frames.append((flat_qpos(hk, t), draw, sub))
      persisted.append((paths, cols))
    if frames:
      frames += [frames[-1]] * 20
    label = f"{'ADJOINT' if self.args.grad == 'analytic' else 'FINITE DIFF'} (sliding)"
    live = True if self.args.live else None
    written = R.emit(vm, vd, cam, frames, out_path=out, label=label, w=W, h=H, live=live)
    if written:
      self.save_montage(written, os.path.splitext(written)[0] + "_montage.png")

  def save_montage(self, mp4_path, png_path):
    import imageio.v2 as imageio
    from PIL import Image
    rd = imageio.get_reader(mp4_path); fr = [f for f in rd]; rd.close()
    sel = [fr[i] for i in np.linspace(0, len(fr) - 1, 10).astype(int)]
    hh, ww, _ = sel[0].shape
    grid = Image.new("RGB", (5 * ww, 2 * hh), (0, 0, 0))
    for kk, f in enumerate(sel):
      grid.paste(Image.fromarray(f), ((kk % 5) * ww, (kk // 5) * hh))
    grid.save(png_path)
    print(f"[slide] montage -> {png_path}")


def main(argv):
  del argv  # unused; config comes from the absl-parsed Args
  demo.run(SlideDemo, demo.parse_args(Args))


if __name__ == "__main__":
  demo.define_flags(Args)
  app.run(main)
