"""Unitree G1 armature system-identification against the shuffle-dance -- ANALYTIC adjoint gradient.

OPEN-LOOP REPLAY vs the RECORDED dance, PARALLEL ENVS. From the dance's initial state we drive the
recorded PD position targets `ctrl` through differentiable mujoco_warp for the whole horizon and score it
by the full-trajectory joint MPJE against the RECORDED dance. Sys-id recovers the joint ARMATURE
(reflected rotor inertia): num_envs envs each start from a DIFFERENT (wrong) armature scale and all
recover to the true x1.0 by minimizing their own full-trajectory MPJE with warp.optim.Adam on the
analytic gradient. One batched (nworld=E) backward yields INDEPENDENT per-env gradients (each env's
armature only affects its own world); the four envs are shown in a 2x2 grid.

CADENCE: the dance is recorded at 0.02 s/frame while the model integrates at dt=0.005 s, so each ctrl
frame is held for SUBSTEPS=4 sim steps. HORIZON (nf frames) is bounded by CHAOS -- beyond ~1 s a legged
robot decorrelates open-loop even at the true armature, so the gradient is TRUNCATED-BPTT: the state is
detached every tbptt_h frames (a short backward chain per chunk) while the shared armature leaf's grad
accumulates across every chunk (the loss is the full-trajectory MPJE). The armature is a MODEL parameter
(m.dof_armature); its grad accumulates over the checkpointed segments (demo.Example harness).

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/g1_sysid.py
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
import viz  # noqa: E402  shared renderer (mp4 or live MuJoCo viewer via MJW_VIEWER)

SCENE = "benchmarks/unitree_g1/scene_flat.xml"  # run from the mujoco_warp repo root
NPZ = "benchmarks/unitree_g1/shuffle_dance.npz"
JADR = 7  # free base qpos [0:7]; joint angles are qpos[7:nq]
SUBSTEPS = 4  # sim steps per recorded ctrl frame (0.02 s frame / 0.005 s dt)
NCONMAX, NJMAX = 48, 192  # per-world constraint limits for scene_flat.xml
ARM_SCALES0 = [2.0, 2.5, 3.0, 3.5]  # per-env initial (wrong) armature scales -- all recover toward 1.0
LANE = 1.4  # 2x2 grid spacing (m)
W, H, FPS = 1024, 768, 30
_GHOST_RGBA = np.array([0.9, 0.68, 0.5, 0.42], dtype=np.float32)  # translucent GHOST = recorded dance
ASSETS = os.path.join(os.path.dirname(__file__), "reports", "assets")


@wp.kernel
def _mpje_accum(qpos: wp.array2d[float], rec: wp.array2d[float], f: int, loss: wp.array(dtype=float)):
  e, j = wp.tid()  # (env e, joint j); each env's armature only affects its own world -> per-env grads
  d = qpos[e, JADR + j] - rec[f, JADR + j]  # time-dependent: compares against the recorded dance frame f
  wp.atomic_add(loss, 0, d * d)


@dataclass
class Args(demo.CommonArgs):
  """g1 sys-id config: CommonArgs (grad/num_envs/steps/lr/device/...) + the dance-fit horizon fields."""

  num_envs: int = field(default=32, metadata={"help": "parallel envs, each a DIFFERENT init armature scale -> distinct trajectories tiled across the field"})
  steps: int = field(default=200, metadata={"help": "Adam steps"})
  lr: float = field(default=0.12, metadata={"help": "Adam learning rate"})
  nf: int = field(default=64, metadata={"help": "dance frames to FIT (must be a multiple of tbptt_h)"})
  tbptt_h: int = field(default=16, metadata={"help": "TBPTT window in FRAMES (chunk = SUBSTEPS*tbptt_h steps)"})
  render_nf: int = field(default=128, metadata={"help": "render horizon in frames (2x the fit -> validation split)"})
  usd_stride: int = field(default=1, metadata={"help": "USD field: subsample (1 = every recorded dance frame -> 1x@50fps)"})
  usd_iters: int = field(default=8, metadata={"help": "USD field: optimization iterations sampled (more = smoother recovery show)"})
  # dense hero field for the immersed render (render_blender.hero_cam): a 15x15 grid of dancers at 2.7m
  # spacing so the pushed-in camera shows a big foreground hero + a field cropping off the frame edges.
  usd_envs: int = field(default=225, metadata={"help": "USD field: instanced lanes (15x15 dense field)"})
  usd_cols: int = field(default=15, metadata={"help": "USD field: grid columns"})
  usd_xpitch: float = field(default=2.7, metadata={"help": "USD field: column pitch (x) = dancer spacing"})
  usd_ypitch: float = field(default=2.7, metadata={"help": "USD field: row pitch (y)"})


class G1SysidDemo(demo.Example):
  """G1 armature sys-id: optimize each env's armature SCALE (self.param, Adam) via the analytic grad of
  the full-trajectory MPJE vs the recorded dance. Model-param leaf m.dof_armature = arm_default*scale;
  its grad accumulates across the truncated-BPTT segments; read_grad projects onto the uniform scale."""

  Args = Args
  capturable = False  # fixed per-frame ctrl schedule + host frame index -> eager checkpointed path
  truncated = True    # truncated BPTT: detach the state adjoint every tbptt_h frames (chunk boundary)

  def optimize(self):
    return self.optimize_adam(betas=(0.9, 0.999))  # default Adam betas (long momentum across the plateau)

  # ---- harness hooks ----

  def build_model(self):
    if not os.path.exists(SCENE):
      raise SystemExit(f"run from the mujoco_warp repo root (missing {SCENE})")
    self.args.chunk = SUBSTEPS * self.args.tbptt_h  # the checkpoint segment = one TBPTT window
    self.NF = self.args.nf
    self.mjm = mujoco.MjModel.from_xml_path(SCENE)
    self.mjm.vis.global_.offwidth, self.mjm.vis.global_.offheight = W, H
    self.mjd = mujoco.MjData(self.mjm)
    self.dance = np.load(NPZ)
    self.rec = self.dance["qpos"].astype(np.float32)  # (>=nf+1, nq) recorded dance = the sys-id target
    self.arm_default = self.mjm.dof_armature.copy().astype(np.float32)  # (nv,)
    self.qpos0 = self.dance["qpos"][0].copy()
    self.qvel0 = self.dance["qvel"][0].copy()
    self.mjd.qpos[:], self.mjd.qvel[:] = self.qpos0, self.qvel0
    mujoco.mj_forward(self.mjm, self.mjd)
    self.njnt = self.mjm.nq - JADR
    self.norm = np.sqrt(self.NF * self.njnt)  # MPJE normalizer

  def init_params(self):
    ne = self.args.num_envs
    s = ARM_SCALES0[:ne] if ne <= len(ARM_SCALES0) else list(np.linspace(0.5, 3.5, ne))  # two-sided wrong scales -> varied field
    return np.array(s, dtype=np.float32)  # (E,) per-env armature scale

  def build_datas(self):
    ne = self.args.num_envs
    self.nT = SUBSTEPS * self.NF  # checkpointed BPTT length (chunk+1 = tbptt window +1 segment buffers)
    self.datas = [mjw.put_data(self.mjm, self.mjd, nworld=ne, nconmax=NCONMAX, njmax=NJMAX) for _ in range(self.args.chunk + 1)]
    for d in self.datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
      d.ctrl.requires_grad = True  # bind_ctrl scatters the fixed dance ctrl into this stable buffer
    self.datas[0].qpos = wp.array(np.tile(self.qpos0, (ne, 1)).astype(np.float32), dtype=float, requires_grad=True)
    self.datas[0].qvel = wp.array(np.tile(self.qvel0, (ne, 1)).astype(np.float32), dtype=float, requires_grad=True)
    # MODEL-parameter leaf: armature = arm_default * scale (host reduce); grad accumulates over all segments
    self.m.dof_armature = wp.array(self.arm_default[None] * self.params[:, None], dtype=float, requires_grad=True)
    self.accum_leaf = self.m.dof_armature
    self.param = wp.array(self.params, dtype=float, requires_grad=True)  # the Adam leaf (scale)
    self.rec_wp = wp.array(self.rec, dtype=float)  # recorded dance on device (indexed by frame)
    self.ctrl_wp = [wp.array(np.tile(self.dance["ctrl"][f], (ne, 1)).astype(np.float32), dtype=float)
                    for f in range(self.NF)]  # per-frame ctrl (E,nu) for the non-taped sim_qpos rollout
    self.ctrl_flat = [wp.array(np.tile(self.dance["ctrl"][f], (ne, 1)).reshape(-1).astype(np.float32), dtype=float)
                      for f in range(self.NF)]  # flat (E*nu,) for bind_ctrl in the taped chunk_step
    self.loss = wp.zeros(1, dtype=float, requires_grad=True)
    self._viz = [mjw.put_data(self.mjm, self.mjd, nworld=ne, nconmax=NCONMAX, njmax=NJMAX) for _ in range(2)]

  def set_params(self):
    self.m.dof_armature.assign((self.arm_default[None] * self.param.numpy()[:, None]).astype(np.float32))

  def chunk_step(self, i, t):
    frame = t // SUBSTEPS  # eager: t is the true global step, so the frame + the modulo below are Python ints
    self.bind_ctrl(i, self.ctrl_flat[frame])  # scatter the fixed dance ctrl -> datas[i].ctrl (a REBIND is not
    # seen by the step under the bc backward-context: the arm falls -> the replay diverges from the dance)
    mjw.step(self.m, self.datas[i], self.datas[i + 1])
    if (t + 1) % SUBSTEPS == 0:  # per-FRAME MPJE vs the recorded dance frame `frame+1`
      wp.launch(_mpje_accum, dim=(self.args.num_envs, self.njnt),
                inputs=[self.datas[i + 1].qpos, self.rec_wp, frame + 1], outputs=[self.loss])

  def read_grad(self):
    # accum_grad = sum over segments of d(loss)/d(dof_armature) (E, nv); project onto the uniform-scale
    # direction (grad w.r.t. scale = <arm_default, d(loss)/d(armature)>)
    return (self.accum_grad * self.arm_default[None]).sum(axis=1)

  def sim_qpos(self):
    """Full per-frame trajectory (E, NF+1, nq) via a fresh forward-only rollout at the current armature
    (checkpointing keeps only chunk+1 buffers). Used for the MPJE readout + the render qpos."""
    ne = self.args.num_envs
    self.m.dof_armature.assign((self.arm_default[None] * self.param.numpy()[:, None]).astype(np.float32))
    a, b = self._viz
    a.qpos = wp.array(np.tile(self.qpos0, (ne, 1)).astype(np.float32), dtype=float)
    a.qvel = wp.array(np.tile(self.qvel0, (ne, 1)).astype(np.float32), dtype=float)
    frames = [np.tile(self.qpos0, (ne, 1)).astype(np.float32)]
    for f in range(self.NF):
      for _ in range(SUBSTEPS):
        a.ctrl = self.ctrl_wp[f]
        mjw.step(self.m, a, b)
        a, b = b, a
      frames.append(a.qpos.numpy().copy())
    return np.array(frames).transpose(1, 0, 2)  # (E, NF+1, nq)

  def record(self, it, loss, qpos, pnp, g):
    per_env = ((qpos[:, 1:, JADR:] - self.rec[1:self.NF + 1, JADR:][None]) ** 2).sum(axis=(1, 2))  # (E,)
    return {"it": it, "loss": loss, "scales": pnp.copy(), "mpje": np.sqrt(per_env) / self.norm, "qpos": qpos}

  def progress(self, rec, g):
    return (f"  [{rec['it']:3d}] scales={np.array2string(rec['scales'], precision=3)}  "
            f"MPJE={np.array2string(rec['mpje'], precision=4)}  |g|~{np.abs(g).mean():.2e}")

  def summary(self, history, best):
    return (f"[g1 sysid / {self.args.grad}] {self.args.num_envs} envs: scales "
            f"{np.array2string(history[0]['scales'], precision=1)} -> "
            f"{np.array2string(history[best]['scales'], precision=3)} (true 1.0)  "
            f"MPJE {history[0]['mpje'].mean():.4f} -> {history[best]['mpje'].mean():.4f}")

  def default_out(self):
    return os.path.join(ASSETS, "g1_sysid.mp4")

  # ---- render: 2x2 grid; each cell = the simulated replay (SOLID) vs the recorded dance (GHOST) ----

  def _mjc_replay(self, rm, scale, render_nf):
    """MuJoCo-C open-loop replay at armature `scale` for render_nf frames -> per-frame qpos (render_nf+1, nq).
    Replays the recovered params PAST the fit horizon (the validation frames)."""
    rm.dof_armature[:] = self.arm_default * scale
    d = mujoco.MjData(rm)
    d.qpos[:], d.qvel[:] = self.dance["qpos"][0], self.dance["qvel"][0]
    mujoco.mj_forward(rm, d)
    qs = [d.qpos.copy()]
    for f in range(render_nf):
      d.ctrl[:] = self.dance["ctrl"][f]
      for _ in range(SUBSTEPS):
        mujoco.mj_step(rm, d)
      qs.append(d.qpos.copy())
    return np.array(qs)

  def _grid_offset(self, e):
    col, row = e % 2, e // 2
    return np.array([(col - 0.5) * LANE, (0.5 - row) * LANE])

  def _add_robot(self, scene, vm, d, qpos, opt, pert, rgba=None):
    d.qpos[:] = qpos
    mujoco.mj_forward(vm, d)
    n0 = scene.ngeom
    mujoco.mjv_addGeoms(vm, d, opt, pert, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scene)
    if rgba is not None:
      for i in range(n0, scene.ngeom):
        scene.geoms[i].rgba = rgba

  def export_usd(self, history, best):
    """--export_usd hook: replay THIS sys-id's convergence across an instanced G1 FIELD for the Blender
    render (g1_sysid_render_blender.py). Samples a spread of iterations; for each, re-runs the recorded
    dance in MuJoCo-C at that iteration's armature scale (env 0, starting at the wrong scale and recovering
    toward 1.0), subsamples + holds the settled end, tracking frame->iteration. The physics replay uses the
    FULL scene (floor contact) but the viz proto is the floor/light-free named-geom robot; both share nq."""
    a = self.args
    out_dir = os.path.join(ASSETS, "g1_sysid_render")
    ks = sorted(set(np.linspace(0, len(history) - 1, a.usd_iters).astype(int).tolist()))
    rm = mujoco.MjModel.from_xml_path(SCENE)  # full scene (needs the floor so the dance stays grounded)
    rn = min(a.render_nf, self.dance["ctrl"].shape[0], self.dance["qpos"].shape[0] - 1)
    ghost = self.dance["qpos"][: rn + 1]  # recorded TARGET dance (nq=36) -> the translucent GHOST overlay (shared)
    idx = list(range(0, rn + 1, a.usd_stride))
    frame_iters = [int(k) for k in ks for _ in range(len(idx) + a.usd_hold)]
    ne = self.args.num_envs
    env_frames = []  # a DISTINCT trajectory per env (each recovers its OWN wrong armature scale) -> diverse field
    for e in range(ne):
      fr = []
      for k in ks:
        sim = self._mjc_replay(rm, float(history[k]["scales"][e]), rn)  # (rn+1, 36) at env e's iter-k scale
        for t in idx:
          fr.append(np.concatenate([sim[t], ghost[t]]))  # [sim(36) | ghost(36)] = 72, overlaid
        fr += [np.concatenate([sim[-1], ghost[-1]])] * a.usd_hold
      env_frames.append([np.asarray(f, np.float64) for f in fr])
    robot = os.path.join(os.path.dirname(SCENE), "unitree_g1_mjlab.xml")
    proto = self.multi_proto(robot, [("sim_", (0.0, 0.0, 0.0)), ("ghost_", (0.0, 0.0, 0.0))])  # sim + ghost overlaid
    assert proto.nq == env_frames[0][0].shape[0], (proto.nq, env_frames[0][0].shape[0])
    offsets = self._usd_grid_offsets(a.usd_envs, a.usd_cols, a.usd_xpitch, a.usd_ypitch)
    fps = max(1, round(1.0 / (self.mjm.opt.timestep * SUBSTEPS * a.usd_stride)))  # 1x real-time (dt*substeps*stride)
    out = self.export_field(proto, None, offsets, out_dir, name="g1_sysid_traj", fps=fps,
                            frame_iters=frame_iters, opt_label="armature scale", env_frames=env_frames)
    scales = [round(float(history[ks[-1]]["scales"][e]), 2) for e in range(min(ne, 8))]
    print(f"[export] g1 sysid: {ne} DISTINCT envs, iters {ks}; final scale(first8) {scales} (true 1.0); "
          f"sim+ghost overlay; NF={len(env_frames[0])} -> {out}")

  def render(self, history, best, out):
    vm = mujoco.MjModel.from_xml_path(SCENE)
    vm.vis.global_.offwidth, vm.vis.global_.offheight = W, H
    vd, gd = mujoco.MjData(vm), mujoco.MjData(vm)
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    rec = self.dance["qpos"]
    E = history[0]["scales"].shape[0]
    offs = [self._grid_offset(e) for e in range(E)]
    rn = min(self.args.render_nf, self.dance["ctrl"].shape[0])
    rm = mujoco.MjModel.from_xml_path(SCENE)

    base_xy = rec[: rn + 1, 0:2].mean(axis=0)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(vm, cam)
    cam.lookat = [float(base_xy[0]), float(base_xy[1]), 0.55]
    cam.distance = 5.6
    cam.azimuth = 140.0
    cam.elevation = -22.0

    parked = self.dance["qpos"][0].copy()
    parked[2] = 100.0  # park the model body off-camera; every robot is drawn as an added geom
    show = [k for k in [0, 15, 35, 60, 95] if k < best] + [best]
    frames = []
    for k in show:
      h = history[k]
      sc = h["scales"]
      sc_s = ", ".join(f"{v:.2f}" for v in sc)
      qlong = [self._mjc_replay(rm, sc[e], rn) for e in range(E)]  # replay past the fit window
      hold = 18 if k == best else 0
      for t in list(range(rn + 1)) + [rn] * hold:
        phase = f"FIT 0-{self.NF}" if t <= self.NF else f"VALIDATION {self.NF}-{rn}"
        sub = f"iter {h['it']:2d}   armature x[{sc_s}] (true 1.0)   frame {min(t, rn):3d}/{rn}   [{phase}]"
        poses = [(qlong[e][t], rec[t], offs[e]) for e in range(E)]

        def draw(scene, poses=poses):
          for sim_q, ghost_q, off in poses:
            gq = ghost_q.copy(); gq[0:2] += off
            self._add_robot(scene, vm, gd, gq, opt, pert, rgba=_GHOST_RGBA)  # recorded dance = orange ghost
            sq = sim_q.copy(); sq[0:2] += off
            self._add_robot(scene, vm, gd, sq, opt, pert)  # simulated replay = solid

        frames.append((parked, draw, sub))
    if frames:
      frames += [frames[-1]] * 20
    return viz.emit(vm, vd, cam, frames, out_path=out, label="ADJOINT (G1 armature)", w=W, h=H, fps=FPS)


def main(argv):
  del argv  # config comes from the absl-parsed Args
  demo.run(G1SysidDemo, demo.parse_args(Args))


if __name__ == "__main__":
  demo.define_flags(Args)
  app.run(main)
