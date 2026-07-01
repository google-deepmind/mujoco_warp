"""Domino spiral chain reaction -- optimize the FIRST domino's forward tipping velocity so the whole
spiral (and in particular the LAST domino) topples, by analytic-adjoint gradient descent through
differentiable mujoco_warp.

A row of thin box dominoes stands on a frictional floor, placed along an ARCHIMEDEAN SPIRAL (ported from
the Newton `domino_spiral` example) and yaw-rotated so each faces the next. Domino 0 is given a single
scalar initial tipping velocity `push`: we set its free-joint LOCAL-x angular velocity qvel0[3] = -push,
which pitches its top along the spiral tangent toward its neighbor (push>0 -> tips FORWARD into the row;
push<0 -> backward, away). We OPTIMIZE `push` so the tipping domino knocks its neighbor, which knocks the
next, ... all the way to the last one. The variable is CONSTRAINED push >= PUSH_MIN > 0 (projected each
Adam step) so the first domino always falls FORWARD, never backward.

  loss = mean_i (z_i / z_stand)^2  +  w_effort * push^2

The first term is a DENSE uprightness objective: each domino's COM height z_i drops from z_stand (upright)
to ~half-thickness (flat), so the term is ~1 standing and ~0 fallen -- summing over the spiral gives
gradient signal from every domino the wavefront has reached, not just the (initially untouched) last one.
The small effort term makes the optimum INTERIOR: GD raises `push` until the chain completes, then trims it
back to the minimal sufficient shove. The last domino only moves THROUGH the domino<->domino contact chain,
so d(loss)/d(push) exercises the body<->body (two-moving-bodies) contact d(qpos) adjoint (like cradle.py),
here compounded by free-body toppling (large-rotation quaternion) on RESTING floor contact -- the frontier.

The `--grad` flag picks the gradient source (both descend the SAME batched rollout, per-env):
  * analytic (default): ONE batched (nworld=E) taped backward through differentiable mujoco_warp
    (adjoint.py) -> per-env d(loss)/d(push). Every few iters the analytic gradient is compared to FD
    (cos / magnitude ratio) so the demo doubles as a contact-adjoint TEST BED (like pingpong.py).
  * fd: batched central difference over the mujoco_warp rollout (the robust baseline; each env's push
    is independent so +eps / -eps is two batched rollouts for all envs at once).

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/dominos.py
  uv run --active python contrib/diffsim/dominos.py --grad fd --no-render          # quick baseline
"""

import argparse
import math
import os
import sys

import mujoco
import numpy as np
import warp as wp
import warp.optim

import mujoco_warp as mjw
from mujoco_warp._src import adjoint  # noqa: F401  registers the analytic step backward

sys.path.insert(0, os.path.dirname(__file__))
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
LR, STEPS = 0.6, 45  # from the 1-domino seed it takes ~35 iters to climb to the completing push (~7 rad/s)
SPREAD = 0.45  # per-env initial-push fan (multi-env only) so lanes start visibly different
ENV_COLS = 2  # multi-env render grid width
W, H = 1024, 768


def _spiral_pose(index):
  """Archimedean-spiral pose of domino `index` (ported from the Newton domino_spiral example): returns
  (x, y, yaw). Neighbors are DOMINO_SPACING apart in ARC LENGTH; yaw faces the domino along the tangent."""
  b = SPIRAL_PITCH / (2.0 * math.pi)
  theta = 0.0
  for _ in range(index):
    r = SPIRAL_INNER_RADIUS + b * theta
    theta += DOMINO_SPACING / math.sqrt(r * r + b * b)
  r = SPIRAL_INNER_RADIUS + b * theta
  x, y = r * math.cos(theta), r * math.sin(theta)
  tx = b * math.cos(theta) - r * math.sin(theta)  # spiral tangent
  ty = b * math.sin(theta) + r * math.cos(theta)
  yaw = math.atan2(-tx, ty)  # local +y (thin/topple axis) points along the tangent, toward the next domino
  return x, y, yaw


def _quat_wxyz(yaw):
  return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))  # rotation about world z


def _poses(num_dom=N_DOM):
  return [(_spiral_pose(i)[0], _spiral_pose(i)[1], _quat_wxyz(_spiral_pose(i)[2])) for i in range(num_dom)]


def _scene(num_dom=N_DOM):
  """Spiral centroid (cx, cy) and half-extent (for framing the camera / spacing the multi-env grid)."""
  xy = np.array([(x, y) for x, y, _ in _poses(num_dom)])
  cx, cy = xy.mean(0)
  ext = float(np.abs(xy - [cx, cy]).max()) + 2.0 * HZ
  return float(cx), float(cy), ext


def domino_xml(num_dom=N_DOM):
  """Single-world physics scene: a frictional floor + `num_dom` upright box dominoes on the spiral, all in
  the same collision lane so each domino hits BOTH the floor and its neighbors. put_data(nworld=E)
  replicates it into E envs."""
  opt = ('<option timestep="%g" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
         'solver="Newton" iterations="50"><flag eulerdamp="disable"/></option>' % DT)
  default = f'<default><geom condim="3" friction="{MU:g} 0.005 0.0001" solref="{SOLREF}" solimp="{SOLIMP}"/></default>'
  floor = '<geom name="floor" type="plane" size="5 5 0.01" rgba="0.3 0.3 0.35 1"/>'
  bodies = ""
  for i, (x, y, (qw, qx, qy, qz)) in enumerate(_poses(num_dom)):
    t = i / max(num_dom - 1, 1)
    rgba = f"{0.9 * (1 - t) + 0.2 * t:.3f} {0.25 + 0.55 * t:.3f} {0.85 * (1 - t) + 0.2 * t:.3f} 1"
    bodies += (
      f'<body name="d{i}" pos="{x:.4f} {y:.4f} {HZ:.4f}" quat="{qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f}">'
      f'<freejoint/><geom name="d{i}" type="box" size="{HX} {HY} {HZ}" mass="{MASS}" rgba="{rgba}"/></body>'
    )
  return f'<mujoco>{opt}{default}<worldbody>{floor}{bodies}</worldbody></mujoco>'


@wp.kernel
def _set_push(push: wp.array(dtype=float), qvel: wp.array2d(dtype=float)):
  """Set domino 0's free-joint LOCAL-x angular velocity qvel[3] = -push. NEGATIVE tips the top forward
  along the spiral tangent (toward the next domino); push>0 is the forward-constrained variable."""
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


def _per_env_loss(qpos_final, push, num_envs, n_dom):
  """Per-env loss from a numpy final qpos (E, 7*n_dom): dense uprightness + effort. Matches the kernels."""
  z = qpos_final[:, 2::7][:, :n_dom] / Z_STAND  # (E, n_dom) COM heights, normalized
  return (z * z).mean(axis=1) + W_EFFORT * push ** 2


def _scatter_push(push, num_envs, nv):
  qv = np.zeros((num_envs, nv), np.float32)
  qv[:, 3] = -push  # domino 0 local-x pitch (negative = forward), matches _set_push
  return qv


def _put(mjm, mjd, num_envs, n_dom):
  """put_data with contact headroom for the whole spiral: each domino touches the floor + up to 2 neighbors."""
  ncon = num_envs * (16 * n_dom + 64)
  return mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=ncon, njmax=6 * ncon)


def rollout(m, mjm, mjd, num_envs, n_dom, push):
  """Batched forward rollout with domino 0 pitched by `push` (E,). Returns qpos traj (E, T+1, 7*n_dom)."""
  d = _put(mjm, mjd, num_envs, n_dom)
  d.qvel = wp.array(_scatter_push(push, num_envs, mjm.nv), dtype=float)
  qs = [d.qpos.numpy().copy()]
  for _ in range(T):
    mjw.step(m, d)
    qs.append(d.qpos.numpy().copy())
  return np.transpose(np.array(qs), (1, 0, 2))  # (E, T+1, nq)


def fd_grad(m, mjm, mjd, num_envs, n_dom, push, eps=1e-2):
  """Batched central difference d(loss_w)/d(push_w): envs are independent worlds, so perturbing every
  env's push at once and reading per-env loss gives all per-env gradients in TWO batched rollouts."""
  lp = _per_env_loss(rollout(m, mjm, mjd, num_envs, n_dom, push + eps)[:, -1], push + eps, num_envs, n_dom)
  lm = _per_env_loss(rollout(m, mjm, mjd, num_envs, n_dom, push - eps)[:, -1], push - eps, num_envs, n_dom)
  return (lp - lm) / (2 * eps)


def analytic_grad(m, mjm, mjd, num_envs, n_dom, push_wp):
  """Batched taped rollout + adjoint.py backward -> populates push_wp.grad. Returns (qpos (E,T+1,nq), loss)."""
  datas = [_put(mjm, mjd, num_envs, n_dom) for _ in range(T + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  datas[0].qvel = wp.zeros((num_envs, mjm.nv), dtype=float, requires_grad=True)
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  inv = 1.0 / float(n_dom)  # per-env mean over dominoes; summed over envs by the atomic_add
  push_wp.grad.zero_()
  tape = wp.Tape()
  with tape:
    wp.launch(_set_push, dim=num_envs, inputs=[push_wp], outputs=[datas[0].qvel])
    for t in range(T):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_accum_upright, dim=(num_envs, n_dom), inputs=[datas[T].qpos, inv], outputs=[loss])
    wp.launch(_accum_effort, dim=num_envs, inputs=[push_wp, W_EFFORT], outputs=[loss])
  tape.backward(loss=loss)
  qpos = np.transpose(np.array([datas[t].qpos.numpy() for t in range(T + 1)]), (1, 0, 2))
  return qpos, float(loss.numpy()[0])


def _init_push(num_envs, spread):
  """Per-env initial push (rad/s): the WHOLE POINT is to DISCOVER the velocity that completes the spiral,
  so iteration 0 must barely start it -- the seed topples just ONE domino and the optimizer must climb to
  the ~7 rad/s that finishes the whole chain. Kept just above the stall boundary (push<=~2.25 -> 0 fall,
  where the chain never propagates so the loss is flat at ~1.0 and the gradient is ZERO) so there is always
  signal. --envs 1 -> a single seed."""
  if num_envs == 1:
    return np.array([2.3], np.float64)  # push=2.3 -> only 1/16 falls: the chain barely gets going
  lo, hi = 2.4, 2.6  # every lane starts partial (~4/16 .. ~6/16), none completes the spiral
  mid = 0.5 * (lo + hi)
  return np.clip(np.linspace(mid - (mid - lo) * spread / SPREAD, mid + (hi - mid) * spread / SPREAD, num_envs), lo, hi)


def _fallen(qpos_final, n_dom):
  """Per-env count of dominoes whose COM dropped below 60% of standing height (toppled)."""
  z = qpos_final[:, 2::7][:, :n_dom]
  return (z < 0.6 * Z_STAND).sum(axis=1)


def optimize(num_envs, spread, grad_mode, steps, lr):
  mjm = mujoco.MjModel.from_xml_string(domino_xml(N_DOM))
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)

  push0 = _init_push(num_envs, spread)
  push = wp.array(push0.astype(np.float32), dtype=float, requires_grad=True)
  opt = warp.optim.Adam([push], lr=lr, betas=(0.7, 0.95))

  history, nan_warned = [], False
  for it in range(steps):
    pnp = push.numpy().astype(np.float64).copy()
    if grad_mode == "analytic":
      qpos, L = analytic_grad(m, mjm, mjd, num_envs, N_DOM, push)
      g = np.nan_to_num(push.grad.numpy().astype(np.float64).copy())
    else:
      qpos = rollout(m, mjm, mjd, num_envs, N_DOM, pnp)
      L = float(_per_env_loss(qpos[:, -1], pnp, num_envs, N_DOM).sum())
      g = fd_grad(m, mjm, mjd, num_envs, N_DOM, pnp)
    if not np.isfinite(g).all():
      if not nan_warned:
        print(f"  WARNING: {grad_mode} gradient is NaN/Inf (iter {it}) -- zeroing; optimizer will stall.")
        nan_warned = True
      g = np.nan_to_num(g)

    per_env = _per_env_loss(qpos[:, -1], pnp, num_envs, N_DOM)
    fallen = _fallen(qpos[:, -1], N_DOM)
    rec = {"it": it, "loss": L, "qpos": qpos, "push": pnp, "per_env": per_env, "fallen": fallen}
    if grad_mode == "analytic" and (it % 10 == 0 or it == steps - 1):  # analytic-vs-FD test-bed check
      fd = fd_grad(m, mjm, mjd, num_envs, N_DOM, pnp)
      cos = float(g @ fd / (np.linalg.norm(g) * np.linalg.norm(fd) + 1e-12))
      ratio = float(np.linalg.norm(g) / (np.linalg.norm(fd) + 1e-12))
      rec["cos"], rec["ratio"] = cos, ratio
    history.append(rec)
    if it % 10 == 0 or it == steps - 1:
      tag = f" cos={rec['cos']:.3f} ratio={rec['ratio']:.2f}" if "cos" in rec else ""
      print(f"  [{it:3d}] loss={L:.4f} push={np.round(pnp, 2).tolist()} fallen={fallen.tolist()}/{N_DOM}"
            f" |g|={np.linalg.norm(g):.3g}{tag}")

    push.grad = wp.array(g.astype(np.float32), dtype=float)
    opt.step([push.grad])
    push.assign(np.clip(push.numpy(), PUSH_MIN, PUSH_MAX).astype(np.float32))  # forward-only projection

  best = min(range(len(history)), key=lambda k: history[k]["loss"])
  hb = history[best]
  print(f"[dominos x{num_envs}env/{grad_mode}] loss {history[0]['loss']:.4f} -> best {hb['loss']:.4f} "
        f"(iter {best}); best fallen {hb['fallen'].tolist()}/{N_DOM}, push {np.round(hb['push'], 2).tolist()}")
  return history, best


# --- rendering: the domino spiral (a grid of spirals if multi-env), each lane's last-domino path traced ----


def _env_offsets(num_envs, cols=ENV_COLS):
  off = np.zeros((num_envs, 3))
  if num_envs > 1:
    pitch = 2.0 * _scene()[2] + 0.3  # > spiral diameter so lanes do not overlap
    rows = int(np.ceil(num_envs / cols))
    for e in range(num_envs):
      r, c = e // cols, e % cols
      off[e, 0] = (c - (cols - 1) / 2.0) * pitch
      off[e, 1] = (r - (rows - 1) / 2.0) * pitch
  return off


def viz_xml_multi(offsets):
  """Viz scene: N decorative free bodies per env (contype=0), placed each frame from the rollout qpos. Env-
  major body order (env outer, domino inner) matches the flattened batched qpos; only mj_forward'd."""
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
  fsize = max(3.0, float(np.abs(offsets).max()) + 2.0 * _scene()[2] + 1.0)
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


def render(history, best, num_envs, grad_mode, out_path, live=False):
  offsets = _env_offsets(num_envs)
  cx, cy, ext = _scene()
  vm = mujoco.MjModel.from_xml_string(viz_xml_multi(offsets))
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
  label = f"{'ADJOINT' if grad_mode == 'analytic' else 'FINITE DIFF'} (dominos)"
  R.emit(vm, vd, cam, frames, out_path=out_path, label=label, w=W, h=H, live=live)
  if live:
    return
  import imageio.v2 as imageio
  from PIL import Image
  rd = imageio.get_reader(out_path)
  fr = [f for f in rd]
  rd.close()
  sel = [fr[i] for i in np.linspace(0, len(fr) - 1, 10).astype(int)]
  hh, ww, _ = sel[0].shape
  grid = Image.new("RGB", (5 * ww, 2 * hh), (0, 0, 0))
  for kk, f in enumerate(sel):
    grid.paste(Image.fromarray(f), ((kk % 5) * ww, (kk // 5) * hh))
  png = os.path.splitext(out_path)[0] + "_montage.png"
  grid.save(png)
  print(f"[dominos] video -> {out_path}\n[dominos] montage -> {png}")


def default_out():
  return os.path.join(os.path.dirname(__file__), "reports", "assets", "dominos.mp4")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--grad", choices=["analytic", "fd"], default="analytic")
  ap.add_argument("--envs", type=int, default=1)
  ap.add_argument("--spread", type=float, default=SPREAD, help="per-env initial-push fan (multi-env only)")
  ap.add_argument("--steps", type=int, default=STEPS)
  ap.add_argument("--lr", type=float, default=LR)
  ap.add_argument("--out", default=None)
  ap.add_argument("--no-render", action="store_true")
  ap.add_argument("--live", action="store_true", help="live MuJoCo viewer instead of mp4")
  args = ap.parse_args()

  history, best = optimize(args.envs, args.spread, args.grad, args.steps, args.lr)
  if args.no_render:
    return
  out = args.out or os.environ.get("MJW_RENDER_PATH") or default_out()
  os.makedirs(os.path.dirname(out), exist_ok=True)
  render(history, best, args.envs, args.grad, out, live=args.live)


if __name__ == "__main__":
  main()
