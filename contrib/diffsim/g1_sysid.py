"""Unitree G1 armature system-identification against the shuffle-dance -- ANALYTIC adjoint gradient.

OPEN-LOOP REPLAY vs the RECORDED dance, PARALLEL ENVS. From the dance's initial state we drive the recorded
PD position targets `ctrl` through differentiable mujoco_warp for the whole horizon and score it by the
full-trajectory joint MPJE against the RECORDED dance. Sys-id recovers the joint ARMATURE (reflected rotor
inertia): NUM_ENVS=4 envs each start from a DIFFERENT (wrong) armature scale and all recover to the true x1.0
by minimizing their own full-trajectory MPJE with warp.optim.Adam on the analytic gradient. Like bounce.py /
dm_suite.py, one batched (nworld=4) wp.Tape yields INDEPENDENT per-env gradients (each env's armature only
affects its own world), and the four envs are shown in a 2x2 grid.

CADENCE: the dance is recorded at 0.02 s/frame while the model integrates at dt=0.005 s, so each ctrl frame is
held for SUBSTEPS=4 sim steps. This is not optional -- replaying 1 step/frame drifts, and coarsening the model
to dt=0.02 makes the stiff-PD + contact integration unstable (the G1 falls even at the true armature). At the
correct cadence the dance was reproduced by this model at the default armature, so the MPJE-vs-armature
landscape is a clean bowl with its minimum at the true x1.0 -- which the gradient descends and recovers.

HORIZON (NF=80 frames = 320 BPTT steps, ~1 s) is bounded by CHAOS, not the gradient: a legged robot is chaotic
open-loop, so beyond ~1 s the replay decorrelates from the dance even at the true armature and the bowl
flattens into a noisy plateau (the full 250-frame gradient is directionless -- analytic AND FD both blow up).
The gradient goes through the contact-rich foot-ground rollout (path E); the analytic magnitude under-estimates
FD through deep contact (the contact-BPTT bias) but is same-sign, and Adam's scale-invariance recovers.

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/g1_sysid.py
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
import viz  # noqa: E402  shared renderer (mp4 or live MuJoCo viewer via MJW_VIEWER)

SCENE = "benchmarks/unitree_g1/scene_flat.xml"  # run from the mujoco_warp repo root
NPZ = "benchmarks/unitree_g1/shuffle_dance.npz"
JADR = 7  # free base qpos [0:7]; 29 joint angles are qpos[7:36]
SUBSTEPS = 4  # sim steps per recorded ctrl frame (0.02 s frame / 0.005 s dt)
NCONMAX, NJMAX = 48, 192  # per-world constraint limits for scene_flat.xml (benchmarks/unitree_g1)
NF = int(os.environ.get("MJW_NF", 64))  # dance frames to FIT (the recorded motion is 250 ctrl frames). The
# full fit trajectory is replayed + scored, but the gradient is TRUNCATED-BPTT: the state is DETACHED every
# TBPTT_H frames so each backward chain is short (H*SUBSTEPS steps), dodging the long-horizon chaos/bias that
# made a single 320+-step backward recover to ~1.2 instead of 1.0. The armature (a shared leaf) still
# accumulates its gradient across every chunk, so the loss is the full-trajectory MPJE.
TBPTT_H = int(os.environ.get("MJW_TBPTT_H", 16))  # detach the state every H frames (H*SUBSTEPS = 64 BPTT steps)
RENDER_NF = int(os.environ.get("MJW_RENDER_NF", 2 * NF))  # the video replays 2x the FIT horizon: frames [0,NF]
# are the fitted window, frames [NF, RENDER_NF] are VALIDATION -- recovered params keep tracking the dance
# there, while wrong-armature envs visibly diverge (a train/test-in-time split).
NUM_ENVS = int(os.environ.get("MJW_NUM_ENVS", 4))
ARM_SCALES0 = [2.0, 2.5, 3.0, 3.5]  # per-env initial (wrong) armature scales -- all recover toward 1.0
# (kept moderate: x4+ sags into the chaotic contact regime whose loss plateau traps recovery near ~1.2)
LR = float(os.environ.get("MJW_LR", 0.12))  # low enough that all four envs SETTLE at 1.0 (higher LR wobbles
# on the noisy near-solution contact gradient); STEPS large enough for the x5 env to get there
STEPS = int(os.environ.get("MJW_STEPS", 120))
BETAS = (float(os.environ.get("MJW_BETA1", 0.9)), float(os.environ.get("MJW_BETA2", 0.999)))  # default Adam
# betas (long momentum) help carry the higher-scale envs across the shallow chaotic-decorrelation plateau
LANE = 1.4  # 2x2 grid spacing (m) between the G1 cells
OUT_MP4 = os.environ.get("MJW_RENDER_PATH", os.path.join(os.path.dirname(__file__), "reports", "assets", "g1_sysid.mp4"))
W, H, FPS = 1024, 768, 30
_GHOST_RGBA = np.array([0.9, 0.68, 0.5, 0.42], dtype=np.float32)  # light gray-orange translucent GHOST = recorded
# dance (visible against the blue textured floor, unlike blue)


@wp.kernel
def _mpje_accum(qpos: wp.array2d[float], rec: wp.array2d[float], f: int, loss: wp.array(dtype=float)):
  e, j = wp.tid()  # (env e, joint j); each env's armature only affects its own world -> per-env grads
  d = qpos[e, JADR + j] - rec[f, JADR + j]
  wp.atomic_add(loss, 0, d * d)


def replay_grad(mjm, mjd, ctrl, qvel0, rec, nf, scales, want_grad=True):
  """Batched (nworld=E) open-loop replay at per-env armature `scales`, SUBSTEPS sim steps per recorded ctrl
  frame; loss = sum over (env, joint, frame) of squared error vs the RECORDED dance `rec` over the WHOLE nf
  trajectory. TRUNCATED BPTT: the state is DETACHED every TBPTT_H frames (a fresh requires_grad leaf, no grad
  link to the previous chunk), and each chunk's loss is back-propagated separately -- so every backward chain
  is only H*SUBSTEPS steps (clean) while the shared-leaf dof_armature.grad accumulates the full-trajectory
  gradient. Returns (per_env_loss[E], per-frame sim_qpos[E, nf+1, nq], arm_grad[E, nv])."""
  E = len(scales)
  m = mjw.put_model(mjm)
  arm = np.stack([mjm.dof_armature.copy() * s for s in scales]).astype(np.float32)  # (E, nv)
  m.dof_armature = wp.array(arm, dtype=float, requires_grad=want_grad)
  if want_grad:
    m.dof_armature.grad.zero_()  # accumulates over ALL chunks below -> full-trajectory gradient
  njnt = mjm.nq - JADR
  rec_w = wp.array(rec.astype(np.float32), dtype=float)
  cur_qpos = np.tile(mjd.qpos, (E, 1)).astype(np.float32)  # every env starts at the dance initial state
  cur_qvel = np.tile(qvel0, (E, 1)).astype(np.float32)
  sim_frames = [cur_qpos.copy()]  # per-frame qpos for the render (nf+1) x (E, nq)
  frame = 0
  while frame < nf:
    cf = min(TBPTT_H, nf - frame)  # frames in this truncation chunk
    datas = [mjw.put_data(mjm, mjd, nworld=E, nconmax=NCONMAX, njmax=NJMAX) for _ in range(cf * SUBSTEPS + 1)]
    for d in datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    datas[0].qpos = wp.array(cur_qpos, dtype=float, requires_grad=True)  # DETACHED leaf (breaks the grad chain)
    datas[0].qvel = wp.array(cur_qvel, dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    fstep = []
    tape = wp.Tape()
    with tape:
      step = 0
      for j in range(cf):
        cw = wp.array(np.tile(ctrl[frame + j], (E, 1)).astype(np.float32), dtype=float)
        for _ in range(SUBSTEPS):  # ctrl held constant across the frame's substeps (as recorded)
          datas[step].ctrl = cw
          mjw.step(m, datas[step], datas[step + 1])
          step += 1
        wp.launch(_mpje_accum, dim=(E, njnt), inputs=[datas[step].qpos, rec_w, frame + j + 1], outputs=[loss])
        fstep.append(step)
    if want_grad:
      tape.backward(loss=loss)  # += this chunk's dL/d(armature) into the shared grad
    for s in fstep:
      sim_frames.append(datas[s].qpos.numpy())
    cur_qpos = datas[fstep[-1]].qpos.numpy()  # DETACH: next chunk starts here with no gradient link back
    cur_qvel = datas[fstep[-1]].qvel.numpy()
    frame += cf
  sim_qpos = np.array(sim_frames).transpose(1, 0, 2)  # (E, nf+1, nq)
  per_env_loss = ((sim_qpos[:, 1:, JADR:] - rec[1 : nf + 1, JADR:][None]) ** 2).sum(axis=(1, 2))  # (E,)
  g = np.nan_to_num(m.dof_armature.grad.numpy()) if want_grad else None  # (E, nv)
  return per_env_loss, sim_qpos, g


def optimize(mjm, mjd, ctrl, qvel0, rec, nf, arm_default, scales0, steps=STEPS, lr=LR):
  """warp.optim.Adam recovery of the per-env armature scales by descending each env's full-trajectory MPJE
  vs the dance. One batched backward per iteration -> independent per-env scale gradients (like dm_suite.py)."""
  E = len(scales0)
  norm = np.sqrt(nf * (mjm.nq - JADR))
  s = wp.array(np.array(scales0, np.float32), dtype=float, requires_grad=True)  # per-env scales (E,)
  opt = warp.optim.Adam([s], lr=lr, betas=BETAS)  # default Adam betas -> long momentum to cross the plateau
  history = []
  for it in range(steps):
    sc = s.numpy().copy()
    per_env_loss, sim_qpos, g = replay_grad(mjm, mjd, ctrl, qvel0, rec, nf, sc)
    gscale = (g * arm_default[None]).sum(axis=1)  # project each env's grad onto the uniform-scale direction
    history.append({"it": it, "scales": sc.copy(), "mpje": np.sqrt(per_env_loss) / norm, "qpos": sim_qpos})
    if it % 4 == 0 or it == steps - 1:
      print(f"  [{it:2d}] scales={np.array2string(sc, precision=3)}  MPJE={np.array2string(np.sqrt(per_env_loss)/norm, precision=4)}")
    s.grad = wp.array(gscale.astype(np.float32), dtype=float)
    opt.step([s.grad])
  best = int(np.argmin([h["mpje"].mean() for h in history]))
  print(f"[g1 sysid] {E} envs: scales {np.array2string(np.array(scales0), precision=1)} -> "
        f"{np.array2string(history[best]['scales'], precision=3)} (true 1.0)  "
        f"MPJE {history[0]['mpje'].mean():.4f} -> {history[best]['mpje'].mean():.4f}")
  return history, best


# --- render: 2x2 grid of parallel envs; each cell = the simulated replay (SOLID) vs the recorded dance
#     (translucent blue GHOST via mjv_addGeoms). As each env's armature is recovered the solid G1 converges. ---


def _grid_offsets(e):
  """xy world offset of env e in a 2-col grid (centered on origin)."""
  col, row = e % 2, e // 2
  return np.array([(col - 0.5) * LANE, (0.5 - row) * LANE])


def _add_robot(scene, vm, d, qpos, opt, pert, rgba=None):
  """Append one robot's dynamic (moving-body) geoms at `qpos` to the scene; recolor if rgba given (ghost)."""
  d.qpos[:] = qpos
  mujoco.mj_forward(vm, d)
  n0 = scene.ngeom
  mujoco.mjv_addGeoms(vm, d, opt, pert, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scene)
  if rgba is not None:
    for i in range(n0, scene.ngeom):
      scene.geoms[i].rgba = rgba


def _mjc_replay(rm, arm_default, dance, scale, render_nf):
  """MuJoCo-C open-loop replay at armature `scale` for render_nf frames -> per-frame qpos (render_nf+1, nq).
  Used to REPLAY the recovered/perturbed params past the fit horizon (the validation frames)."""
  rm.dof_armature[:] = arm_default * scale
  d = mujoco.MjData(rm)
  d.qpos[:], d.qvel[:] = dance["qpos"][0], dance["qvel"][0]
  mujoco.mj_forward(rm, d)
  qs = [d.qpos.copy()]
  for f in range(render_nf):
    d.ctrl[:] = dance["ctrl"][f]
    for _ in range(SUBSTEPS):
      mujoco.mj_step(rm, d)
    qs.append(d.qpos.copy())
  return np.array(qs)


def render(mjm, dance, history, best, out_path=OUT_MP4, nf=NF):
  vm = mujoco.MjModel.from_xml_path(SCENE)
  vm.vis.global_.offwidth, vm.vis.global_.offheight = W, H  # scene_flat.xml sets no offscreen size
  vd = mujoco.MjData(vm)
  gd = mujoco.MjData(vm)
  opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
  rec = dance["qpos"]
  E = history[0]["scales"].shape[0]
  offs = [_grid_offsets(e) for e in range(E)]

  rn = min(RENDER_NF, dance["ctrl"].shape[0])  # render horizon (2x the fit horizon nf, capped at dance length)
  rm = mujoco.MjModel.from_xml_path(SCENE)  # separate model for the MuJoCo-C replay past the fit window
  arm_default = rm.dof_armature.copy()

  # center on the robots' actual centroid: the grid offsets are symmetric (mean 0), so the group centroid
  # tracks the dance base -- look at its trajectory mean (the shuffle drifts, so center on the average).
  base_xy = rec[: rn + 1, 0:2].mean(axis=0)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(vm, cam)
  cam.lookat = [float(base_xy[0]), float(base_xy[1]), 0.55]
  cam.distance = 5.6  # frame the 2x2 grid + the ~2 s of dance drift
  cam.azimuth = 140.0
  cam.elevation = -22.0

  parked = dance["qpos"][0].copy()
  parked[2] = 100.0  # park the model's own body off-camera; every robot is drawn as an added geom

  # early iterations then END on `best` (the tightest recovery) with a hold.
  show = [k for k in [0, 15, 35, 60, 95] if k < best] + [best]
  frames = []
  for k in show:
    h = history[k]
    sc, mp = h["scales"], h["mpje"]
    sc_s = ", ".join(f"{v:.2f}" for v in sc)
    qlong = [_mjc_replay(rm, arm_default, dance, sc[e], rn) for e in range(E)]  # replay past the fit window
    hold = 18 if k == best else 0
    idx = list(range(rn + 1)) + [rn] * hold
    for t in idx:
      phase = f"FIT 0-{nf}" if t <= nf else f"VALIDATION {nf}-{rn}"
      sub = f"iter {h['it']:2d}   armature x[{sc_s}] (true 1.0)   frame {min(t, rn):3d}/{rn}   [{phase}]"
      poses = [(qlong[e][t], rec[t], offs[e]) for e in range(E)]

      def draw(scene, poses=poses):
        for sim_q, ghost_q, off in poses:
          gq = ghost_q.copy(); gq[0:2] += off
          _add_robot(scene, vm, gd, gq, opt, pert, rgba=_GHOST_RGBA)  # recorded dance = orange ghost
          sq = sim_q.copy(); sq[0:2] += off
          _add_robot(scene, vm, gd, sq, opt, pert)  # simulated replay = solid, natural colors

      frames.append((parked, draw, sub))
  if frames:
    frames += [frames[-1]] * 20
  return viz.emit(vm, vd, cam, frames, out_path=out_path, label="ADJOINT (G1 armature)", w=W, h=H, fps=FPS)


def main():
  if not os.path.exists(SCENE):
    raise SystemExit(f"run from the mujoco_warp repo root (missing {SCENE})")
  mjm = mujoco.MjModel.from_xml_path(SCENE)  # from_xml_path resolves the scene's <include>
  mjd = mujoco.MjData(mjm)
  dance = np.load(NPZ)
  nf = min(NF, dance["ctrl"].shape[0])
  ctrl = dance["ctrl"][:nf]
  qvel0 = dance["qvel"][0].copy()
  rec = dance["qpos"]  # THE RECORDED DANCE is the sys-id target
  mjd.qpos[:], mjd.qvel[:] = dance["qpos"][0], dance["qvel"][0]  # start from the dance initial state
  mujoco.mj_forward(mjm, mjd)
  arm_default = mjm.dof_armature.copy()
  scales0 = ARM_SCALES0[:NUM_ENVS]
  norm = np.sqrt(nf * (mjm.nq - JADR))

  # analytic vs FD gradient of each env's full-trajectory MPJE w.r.t. its armature scale, at the init points
  per_env_loss, _, g = replay_grad(mjm, mjd, ctrl, qvel0, rec, nf, scales0)
  gscale = (g * arm_default[None]).sum(axis=1)
  eps = 1e-3
  lp, _, _ = replay_grad(mjm, mjd, ctrl, qvel0, rec, nf, [s + eps for s in scales0], want_grad=False)
  lm, _, _ = replay_grad(mjm, mjd, ctrl, qvel0, rec, nf, [s - eps for s in scales0], want_grad=False)
  fd = (lp - lm) / (2 * eps)
  print(f"[g1 replay] NF={nf} substeps={SUBSTEPS}  {NUM_ENVS} envs, init scales {scales0}")
  print(f"[g1 replay] armature-scale grad per env: analytic {np.array2string(gscale, precision=2)}")
  print(f"                                          fd       {np.array2string(fd, precision=2)}  "
        f"(same sign: {np.all(gscale * fd > 0)})")

  history, best = optimize(mjm, mjd, ctrl, qvel0, rec, nf, arm_default, scales0)
  if os.environ.get("MJW_NO_RENDER") != "1":
    os.makedirs(os.path.dirname(out_path := OUT_MP4), exist_ok=True)
    render(mjm, dance, history, best, out_path=out_path, nf=nf)


if __name__ == "__main__":
  main()
