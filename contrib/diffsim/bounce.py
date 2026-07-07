"""Bounce trajectory-optimization video (Newton example_diffsim_ball style) -- FD or analytic gradient.

A free sphere is dropped with an initial velocity and bounces off a FLOOR and a WALL; we optimize
that initial velocity so it lands on a target (loss = ||final position - target||^2). Many envs are
optimized IN PARALLEL (one lane per env, à la rewarped/envs/warp_examples/bounce.py), seeded with a
small per-env spread of the initial velocity so the parallel trajectories are visually distinct and
you watch the whole fan converge. Across iterations, each env's rollout is animated (bouncing off
floor + wall) with its trajectory drawn as a loss-colored polyline (Bourke: red=bad -> blue=converged)
and prior iterations persisting.

The `--grad` flag picks the gradient source (both descend the SAME batched rollout, per-env):
  * analytic (default): ONE batched (nworld=B) wp.Tape over mjw.step through differentiable
    mujoco_warp (adjoint.py). step()'s analytic backward is batched over worlds, so a single
    tape.backward yields independent per-env gradients (verified == per-world single backward).
    Exercises sphere-plane / sphere-box contact + Coulomb friction.
  * fd: per-env central-difference over the MuJoCo-C rollout (the robust baseline).

Physics matches Newton (soft_contact_restitution=1.0, mu=0.2): elastic contact via NEGATIVE solref
(MuJoCo has no restitution coeff), low friction, and LOCAL gradient descent from the initial velocity
so the optimizer stays in the wall-bounce basin (not the contact-free arc). Newton logs to USD; here
we write mp4.

  # analytic adjoint gradient (default) -> reports/assets/bounce_parallel.mp4
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/bounce.py
  # finite-difference baseline -> reports/assets/bounce_fd.mp4
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/bounce.py --grad fd
"""

import argparse
import os
import sys

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw
mjw.enable_grad()

sys.path.insert(0, os.path.dirname(__file__))
import viz  # noqa: E402  (shared renderer: mp4 or live MuJoCo viewer via MJW_VIEWER)

# Newton example_diffsim_ball numbers (Z-up).
START = (0.0, -0.5, 1.0)
QVEL0 = (0.0, 5.0, -5.0, 0.0, 0.0, 0.0)
TARGET = (0.0, -2.0, 1.5)
T = 150  # steps @ dt=0.004 -> 0.6s
SOLREF = "-3000 -2"  # elastic-ish: direct (stiffness, damping), low damping -> bounce
LR, STEPS = 0.04, 160
# condim/friction env-configurable: condim 6 + nonzero torsion/roll exercises the rotational contact
# rows. geom friction is [slide, torsion, roll]; default = condim 3, slide-only (original behavior).
CONDIM = os.environ.get("MJW_BOUNCE_CONDIM", "3")
FRICTION = os.environ.get("MJW_BOUNCE_FRICTION", "0.2")
W, H, FPS = 1024, 768, 30

# parallel-env config
NUM_ENVS = int(os.environ.get("MJW_BOUNCE_NUM_ENVS", "8"))
# per-env spread of the initial velocity (vy, vz) so the parallel envs splay out visibly.
# 0 -> all envs identical (warp_examples/bounce.py replication). Kept small (0.25) so every
# env starts in the converging basin: a wide spread pushes some envs into the long-horizon
# contact-BPTT explosion regime (gradients spike at the high-velocity wall bounce) and GD
# blows them up -- a real instability, not to be clipped past, so we just init conservatively.
SPREAD = float(os.environ.get("MJW_BOUNCE_SPREAD", "0.25"))
# render layout: "offset" replicates the wall+target per env in its own lane (à la
# warp_examples/bounce.py's env_offset); "overlay" superimposes all envs in one shared
# scene. Lane axis/spacing are configurable (walls are 2 units wide -> spacing must be > 2).
LAYOUT = os.environ.get("MJW_BOUNCE_LAYOUT", "offset")
ENV_AXIS = os.environ.get("MJW_BOUNCE_ENV_AXIS", "x")
ENV_SPACING = float(os.environ.get("MJW_BOUNCE_ENV_SPACING", "3.0"))

ASSETS = os.path.join(os.path.dirname(__file__), "reports", "assets")  # default output dir


def bounce_xml(physics_only=True):
  sx, sy, sz = START
  if physics_only:
    return f"""
<mujoco>
  <option timestep="0.004" cone="elliptic" integrator="implicitfast"
          tolerance="1e-8" iterations="100" ls_iterations="50" gravity="0 0 -9.81">
    <flag contact="enable"/>
  </option>
  <default>
    <geom condim="{CONDIM}" friction="{FRICTION}" solref="{SOLREF}" solimp="0 0.95 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 .05"/>
    <geom name="wall" type="box" pos="0 2 1" size="1 0.25 1"/>
    <body name="ball" pos="{sx} {sy} {sz}"><freejoint/>
      <geom name="ball" type="sphere" size="0.1" mass="1"/></body>
  </worldbody>
</mujoco>
"""
  tx, ty, tz = TARGET
  return f"""
<mujoco>
  <option gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="{W}" offheight="{H}"/>
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
    <geom name="target" type="box" size="0.1 0.1 0.1" pos="{tx} {ty} {tz}" rgba="0.5 0 0.5 1"
          contype="0" conaffinity="0"/>
    <body name="ball" pos="{sx} {sy} {sz}"><freejoint/>
      <geom name="ball" type="sphere" size="0.1" rgba="0.9 0.7 0.3 1" mass="1"/></body>
  </worldbody>
</mujoco>
"""


def rollout(qvel0):
  """mj_step rollout. Returns (xyz[T+1,3], qpos[T+1,7], loss, hit_floor, hit_wall)."""
  m = mujoco.MjModel.from_xml_string(bounce_xml())
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
  d.qvel[:6] = qvel0
  xyz = [d.xpos[bid].copy()]
  qpos = [d.qpos.copy()]
  for _ in range(T):
    mujoco.mj_step(m, d)
    xyz.append(d.xpos[bid].copy())
    qpos.append(d.qpos.copy())
  xyz, qpos = np.array(xyz), np.array(qpos)
  loss = float(np.sum((xyz[-1] - np.array(TARGET)) ** 2))
  return xyz, qpos, loss, xyz[:, 2].min() < 0.13, xyz[:, 1].max() > 1.6


# --- gradients: FD (per-env central difference) and analytic (batched mjwarp adjoint) ---


def fd_grad(qvel0, eps=1e-3):
  """Central-difference d(loss)/d(qvel0[:3]) for one env (the robust baseline)."""
  g = np.zeros(3)
  for i in range(3):
    vp = qvel0.copy(); vp[i] += eps
    vm = qvel0.copy(); vm[i] -= eps
    g[i] = (rollout(vp)[2] - rollout(vm)[2]) / (2 * eps)
  return g


@wp.kernel
def _bounce_loss_batched(qpos: wp.array2d[float], target: wp.vec3, loss: wp.array(dtype=float)):
  """loss = sum_w ||ball_pos_w - target||^2 over all worlds. Each world's qvel0 only affects
  its own qpos, so d(sum)/d(qvel0_w) = d(loss_w)/d(qvel0_w): one backward -> per-env grads."""
  w = wp.tid()
  delta = wp.vec3(qpos[w, 0] - target[0], qpos[w, 1] - target[1], qpos[w, 2] - target[2])
  wp.atomic_add(loss, 0, wp.dot(delta, delta))


def taped_bounce_grad_batched(m, mjm, mjd, qpos0, qvel0, target, n_steps):
  """Per-env d(loss)/d(qvel0[:, :3]) via a batched (nworld=B) wp.Tape over the n_steps rollout.

  qpos0: (B, nq), qvel0: (B, nv). Returns qvel_grad: (B, 3). A distinct batched Data per step;
  step(m, datas[t], datas[t+1]) advances out-of-place and injects the per-step analytic backward
  (datas[t+1].grad -> datas[t].grad) over all worlds.
  """
  num_envs = qvel0.shape[0]
  datas = [mjw.put_data(mjm, mjd, nworld=num_envs) for _ in range(n_steps + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  datas[0].qpos = wp.array(qpos0.astype(np.float32), dtype=wp.float32, requires_grad=True)
  datas[0].qvel = wp.array(qvel0.astype(np.float32), dtype=wp.float32, requires_grad=True)

  loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
  target_v = wp.vec3(float(target[0]), float(target[1]), float(target[2]))

  tape = wp.Tape()
  with tape:
    for t in range(n_steps):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_bounce_loss_batched, dim=num_envs, inputs=[datas[n_steps].qpos, target_v], outputs=[loss])

  tape.backward(loss=loss)
  return datas[0].qvel.grad.numpy()[:, :3].astype(np.float64).copy()


def _make_grad_fn(grad_method, mjm, mjd, qpos0_B, target):
  """Return grad_fn(qvel: (B, nv)) -> g: (B, 3), d(loss)/d(qvel0[:3]) per env. 'fd' = per-env central
  difference (MuJoCo-C); 'analytic' = one batched taped mjwarp backward (adjoint.py)."""
  if grad_method == "fd":
    return lambda qvel: np.stack([fd_grad(qvel[e]) for e in range(len(qvel))])
  if grad_method == "analytic":
    m = mjw.put_model(mjm)  # build the warp model once, reuse across iterations
    return lambda qvel: taped_bounce_grad_batched(m, mjm, mjd, qpos0_B, qvel, target, T)
  raise ValueError(f"unknown grad method: {grad_method!r} (want 'fd' or 'analytic')")


def _init_qvel(num_envs, spread):
  """Per-env initial freejoint velocity: QVEL0 with a deterministic spread on (vy, vz)."""
  qvel = np.tile(np.array(QVEL0, dtype=np.float64), (num_envs, 1))
  if num_envs > 1 and spread != 0.0:
    s = np.linspace(-spread, spread, num_envs)
    qvel[:, 1] += s  # vy
    qvel[:, 2] += s[::-1]  # vz (opposite ramp so envs differ in both)
  return qvel


def optimize(grad_method="analytic", num_envs=NUM_ENVS, steps=STEPS, rate=LR, spread=SPREAD):
  """Parallel LOCAL gradient descent: num_envs independent initial-velocity optimizations, all driven
  per iteration by one gradient call (batched analytic backward, or per-env FD)."""
  mjm = mujoco.MjModel.from_xml_string(bounce_xml())
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  qpos0 = mjd.qpos.copy()
  qpos0_B = np.tile(qpos0, (num_envs, 1))
  target = np.array(TARGET)
  grad_fn = _make_grad_fn(grad_method, mjm, mjd, qpos0_B, target)

  qvel = _init_qvel(num_envs, spread)
  history = []
  for it in range(steps):
    # MuJoCo-C rollout per env for the trajectory/video + loss display
    xyz = np.empty((num_envs, T + 1, 3))
    qpos = np.empty((num_envs, T + 1, qpos0.shape[0]))
    losses = np.empty(num_envs)
    hf = hw = False
    for e in range(num_envs):
      xe, qe, le, hfe, hwe = rollout(qvel[e])
      xyz[e], qpos[e], losses[e] = xe, qe, le
      hf, hw = hf or hfe, hw or hwe
    g = grad_fn(qvel)  # one gradient call for all envs
    history.append({"it": it, "qvel": qvel[:, :3].copy(), "losses": losses.copy(), "xyz": xyz, "qpos": qpos})
    if it % 10 == 0:
      print(
        f"  [{it:4d}] mean_loss={losses.mean():.4f} best={losses.min():.4f} "
        f"floor={hf} wall={hw}  mean|g|={np.linalg.norm(g, axis=1).mean():.3f}"
      )
    qvel[:, :3] -= rate * g  # per-env update

  mean_losses = np.array([h["losses"].mean() for h in history])
  best = int(mean_losses.argmin())
  tag = "analytic (adjoint)" if grad_method == "analytic" else "finite diff"
  print(
    f"[bounce optim / {tag}] {num_envs} envs: mean_loss {mean_losses[0]:.3f} -> best {mean_losses[best]:.3f} "
    f"(iter {best})"
  )
  return history, best


# --- multi-env rendering (parallel-env lanes in the shared floor+wall+target scene) ---


def _env_offsets(num_envs, layout=LAYOUT, axis=ENV_AXIS, spacing=ENV_SPACING):
  """Per-env world offset (N, 3). 'offset' -> a lane per env along `axis`, centered on 0;
  'overlay' -> all zeros (every env in the shared scene)."""
  off = np.zeros((num_envs, 3))
  if layout == "offset" and num_envs > 1:
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    off[:, ax] = (np.arange(num_envs) - (num_envs - 1) / 2.0) * spacing
  return off


def _viz_xml_offset(offsets):
  """bounce scene with the wall + target REPLICATED at each env's offset (one shared
  ground plane; the single ball body is parked off-camera and every ball is drawn as an
  added geom). Mirrors bounce_xml(physics_only=False)'s look."""
  tx, ty, tz = TARGET
  lanes = ""
  for ox, oy, oz in offsets:
    lanes += (
      f'<geom type="box" pos="{ox:.3f} {oy + 2.0:.3f} {oz + 1.0:.3f}" size="1 0.25 1" rgba="0.5 0.5 0.55 1"/>\n    '
      f'<geom type="box" size="0.1 0.1 0.1" pos="{ox + tx:.3f} {oy + ty:.3f} {oz + tz:.3f}" '
      f'rgba="0.5 0 0.5 1" contype="0" conaffinity="0"/>\n    '
    )
  return f"""
<mujoco>
  <option gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="{W}" offheight="{H}"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="8 8" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0.4 -1 2" dir="0 0.4 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="0 0 .05" material="grid"/>
    {lanes}
    <body name="ball" pos="0 -0.5 1.0"><freejoint/>
      <geom name="ball" type="sphere" size="0.1" rgba="0.9 0.7 0.3 1" mass="1"/></body>
  </worldbody>
</mujoco>
"""


def render_video(history, best, out_path, sample_every=4, label="ADJOINT (bounce)", layout=LAYOUT, live=None):
  """Animate all envs bouncing together for selected iterations; persist each shown iteration's
  per-env trajectories faintly so the parallel fan visibly converges. Outputs an mp4, or the live
  MuJoCo viewer when `live=True` / `MJW_VIEWER=1` (via viz.emit)."""
  num_envs = history[0]["xyz"].shape[0]
  # ONE shared colormap keyed on loss (Bourke: red=high loss -> blue=converged), same for every
  # env, so the whole fan shifts color as it converges (the lanes already separate envs spatially).
  lo, hi = 0.0, float(max(h["losses"].max() for h in history))

  def _loss_colors(losses):
    return [viz.bourke_color_map(lo, hi, float(v)) for v in losses]

  offsets = _env_offsets(num_envs, layout=layout)  # (N, 3); zeros for overlay
  if layout == "offset":
    vm = mujoco.MjModel.from_xml_string(_viz_xml_offset(offsets))
  else:
    vm = mujoco.MjModel.from_xml_string(bounce_xml(physics_only=False))
  vd = mujoco.MjData(vm)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(vm, cam)
  spread = float(offsets.max(0).sum() - offsets.min(0).sum())  # extent across the lanes
  if layout == "offset":
    # near-perpendicular to the row so every lane is framed at a similar size (a shallower
    # angle foreshortens the far lanes); distance scales to fit the row width.
    cam.lookat = [0.0, 0.0, 0.7]  # lanes are centered on 0
    cam.distance = 7.0 + 0.5 * spread
    cam.azimuth = 78.0
    cam.elevation = -24.0
  else:
    cam.lookat = [0.0, 0.0, 0.8]
    cam.distance = 8.5
    cam.azimuth = 50.0
    cam.elevation = -18.0

  last = len(history) - 1
  show = [k for k in sorted(set([0, 1, 2, 4, 8, 16, 32, 64, 110, best, last])) if k < len(history)]
  best_mean = min(h["losses"].mean() for h in history)

  frames = []
  persisted = []  # (xyz_all[B,T+1,3], colors[B]) of completed shown iterations
  for k in show:
    h = history[k]
    cols = _loss_colors(h["losses"])
    sub = (
      f"iter {h['it']:3d}    mean loss {h['losses'].mean():.3f}    "
      f"best {best_mean:.3f}    envs {num_envs}"
    )
    steps_idx = list(range(0, T + 1, sample_every)) + [T]
    hold = 20 if k == best else 0
    for t in steps_idx + [T] * hold:
      snap = list(persisted)  # snapshot of prior shown iterations at this frame
      trails = [h["xyz"][e, : t + 1] + offsets[e] for e in range(num_envs)]
      balls = [h["xyz"][e, t] + offsets[e] for e in range(num_envs)]

      def draw(scene, snap=snap, trails=trails, balls=balls, cols=cols):
        for xyz_all, pcols in snap:  # prior shown iterations stay on screen, faint
          for e in range(num_envs):
            viz.add_polyline(scene, xyz_all[e, ::2] + offsets[e], pcols[e], width=0.006, alpha=0.25)
        for e in range(num_envs):  # this iteration: growing trail + ball, per env (in its lane)
          viz.add_polyline(scene, trails[e], cols[e], width=0.012)
          viz.add_sphere(scene, balls[e], cols[e], size=0.1)

      frames.append((None, draw, sub))  # qpos=None: the model ball stays parked off-camera
    persisted.append((h["xyz"], cols))
  if frames:
    frames += [frames[-1]] * 20

  # park the model's single body ball far off-camera; every env ball is drawn as an added geom
  vd.qpos[:3] = [0.0, 0.0, 100.0]
  return viz.emit(vm, vd, cam, frames, out_path=out_path, label=label, w=W, h=H, fps=FPS, live=live)


def _default_out(grad_method):
  """reports/assets/bounce_parallel.mp4 for analytic (the report asset), bounce_fd.mp4 for FD."""
  name = "bounce_parallel" if grad_method == "analytic" else "bounce_fd"
  return os.path.join(ASSETS, name + ".mp4")


def main():
  ap = argparse.ArgumentParser(description="Bounce optimization video (FD or mjwarp adjoint gradient).")
  ap.add_argument("--grad", choices=["analytic", "fd"], default="analytic",
                  help="gradient source: 'analytic' = differentiable mujoco_warp adjoint (default); 'fd' = finite diff")
  ap.add_argument("--num-envs", type=int, default=NUM_ENVS, help="parallel envs optimized at once")
  ap.add_argument("--out", default=None,
                  help="output mp4 path (default: reports/assets/bounce_{parallel,fd}.mp4; MJW_RENDER_PATH also honored)")
  ap.add_argument("--live", action="store_true", help="open the live MuJoCo viewer instead of writing an mp4")
  args = ap.parse_args()

  out = args.out or os.environ.get("MJW_RENDER_PATH") or _default_out(args.grad)
  live = True if args.live else None  # None -> honor MJW_VIEWER
  os.makedirs(os.path.dirname(out), exist_ok=True)

  history, best = optimize(grad_method=args.grad, num_envs=args.num_envs)
  label = "ADJOINT (bounce)" if args.grad == "analytic" else "FINITE DIFF (bounce)"
  render_video(history, best, out_path=out, label=label, live=live)


if __name__ == "__main__":
  main()
