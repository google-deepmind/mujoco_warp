"""Frictional slide-to-target, driven by the analytic adjoint gradient + Adam (works NOW).

Pucks are launched across a flat floor; Coulomb friction decelerates each to a stop. We optimize each
puck's 2-D LAUNCH VELOCITY qvel0[:2] so it stops on its OWN target -- a STATE gradient d(loss)/d(qvel0)
through sustained frictional contact on a single free body (nq=7, nv=6 per puck).

  --geom box | cylinder | both   1 box, 1 cylinder (flat disk), or a box + cylinder (separate lanes).
  --envs N                       N PARALLEL envs (batched nworld=N), each converging to its own fanned
                                 target -- optimized together in ONE tape.backward (like bounce/cradle).

Data-parallel over BOTH count axes: the scene is a LIST of object specs (max 2) and the physics is
batched over nworld; the set-velocity / squared-error kernels launch over dim=(nworld, nobj) -- one
thread per (env, puck), the error kernel wp.atomic_add's into the shared loss -- so one kernel serves
any (nworld x nobj) with no dynamic loop (backward-safe). One wp.Tape over the batched rollout, one
tape.backward -> dL/d[all launch velocities], optimized with warp.optim.Adam (SHAC betas 0.7, 0.95).

WHY SOFT CONTACT + ADAM: with STIFF contact the discrete stick/slip stop is genuinely NON-SMOOTH -- the
exact gradient develops a REAL spike (FD-confirmed) that flips the descent direction and vanilla GD
explodes on it. SOFT contact (wide solimp) regularizes the stop so the analytic gradient is the smooth
"bowl" gradient (cos=1.0 vs FD); Adam (scale-invariant step) converges. The cylinder<->plane adjoint (a
wp.sqrt(0) NaN for the flat resting disk) is fixed in adjoint.py.

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/slide.py --geom both --envs 4
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import warp as wp
import warp.optim

import mujoco_warp as mjw
from mujoco_warp._src import adjoint  # noqa: F401  registers the analytic step backward (+ safe wp.sqrt adjoint)

sys.path.insert(0, os.path.dirname(__file__))
import viz as R  # noqa: E402

DT, T = 0.004, 150
MU = 0.7
LR, STEPS = 0.1, 200
SOLIMP, SOLREF = "0 0.9 0.02", "0.02 1"  # SOFT contact: wide impedance ramp regularizes the stick/slip stop
W, H = 1024, 768
ENV_COLS, ENV_SPACING = 2, 1.5  # multi-env render grid: columns and pitch (> scene extent)
SPREAD = 0.5  # per-env target fan half-angle (rad): env targets rotate about their start
_HALF_H = 0.012  # puck half-height (radius 0.045 >> this -> stays flat/stable: cylinder axis || normal, the
# len_sqr=0 degenerate case that exercises the adjoint.py wp.sqrt fix. Thick enough to render above the rings.
_SHAPE = {"box": f'type="box" size="0.045 0.045 {_HALF_H}"', "cylinder": f'type="cylinder" size="0.045 {_HALF_H}"'}
_RGBA = {"box": "0.95 0.75 0.25 1", "cyl": "0.35 0.65 0.95 1"}
_RINGS = [(0.090, 0.0020, "0.85 0.12 0.12 1"), (0.068, 0.0026, "0.97 0.97 0.97 1"),
          (0.046, 0.0032, "0.85 0.12 0.12 1"), (0.024, 0.0038, "0.97 0.97 0.97 1"), (0.009, 0.0044, "0.85 0.12 0.12 1")]


def objects_for(geom):
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


def fan_targets(objects, num_envs, spread):
  """Per-(env,obj) target, env-major (index w*nobj+o), shape (num_envs*nobj, 2). Each env rotates every
  object's (target-start) about its start by a fanned angle in [-spread, +spread] (env 0..N-1); a single
  env -> angle 0 = the base target, so --envs 1 matches the single-object demo."""
  nobj = len(objects)
  angles = np.zeros(1) if num_envs == 1 else np.linspace(-spread, spread, num_envs)
  out = np.zeros((num_envs * nobj, 2))
  for w in range(num_envs):
    ca, sa = np.cos(angles[w]), np.sin(angles[w])
    rot = np.array([[ca, -sa], [sa, ca]])
    for o, ob in enumerate(objects):
      out[w * nobj + o] = ob["start"] + rot @ (ob["target"] - ob["start"])
  return out


def _body(o, i):
  # collision lanes: floor (contype=1, conaffinity=1); puck i (contype=1<<(i+1), conaffinity=1) -> hits the
  # floor but NOT the other puck (independent slides). Replicated across nworld by put_data.
  return (f'<body name="{o["name"]}" pos="{o["start"][0]:.3f} {o["start"][1]:.3f} {_HALF_H}"><freejoint/>'
          f'<geom name="{o["name"]}" {o["shape"]} mass="0.5" contype="{1 << (i + 1)}" conaffinity="1" '
          f'friction="{MU:g} 0.005 0.0001" rgba="{o["rgba"]}"/></body>')


def slide_xml(objects):
  """Single-world physics scene (nobj bodies); put_data(nworld=N) replicates it into N envs."""
  opt = ('<option timestep="%g" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
         'solver="Newton" iterations="50"><flag eulerdamp="disable"/></option>' % DT)
  default = f'<default><geom condim="3" solimp="{SOLIMP}" solref="{SOLREF}"/></default>'
  bodies = "".join(_body(o, i) for i, o in enumerate(objects))
  floor = f'<geom name="floor" type="plane" size="5 5 0.01" contype="1" conaffinity="1" friction="{MU} 0.005 0.0001"/>'
  return f'<mujoco>{opt}{default}<worldbody>{floor}{bodies}</worldbody></mujoco>'


@wp.kernel
def _set_qvel(v: wp.array(dtype=float), nobj: int, qvel: wp.array2d(dtype=float)):
  w, o = wp.tid()  # (env, puck); launch dim=(nworld, nobj)
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


def _scatter_qvel_np(v0, num_envs, nobj, nv):
  qv = np.zeros((num_envs, nv), np.float32)
  for w in range(num_envs):
    for o in range(nobj):
      qv[w, 6 * o], qv[w, 6 * o + 1] = v0[(w * nobj + o) * 2], v0[(w * nobj + o) * 2 + 1]
  return qv


def rollout(m, mjm, mjd, num_envs, nobj, v0):
  """Batched forward rollout. Returns qpos traj (num_envs, T+1, 7*nobj)."""
  d = mjw.put_data(mjm, mjd, nworld=num_envs)
  d.qvel = wp.array(_scatter_qvel_np(v0, num_envs, nobj, mjm.nv), dtype=float)
  qs = [d.qpos.numpy().copy()]
  for _ in range(T):
    mjw.step(m, d)
    qs.append(d.qpos.numpy().copy())
  return np.transpose(np.array(qs), (1, 0, 2))  # (num_envs, T+1, nq)


def loss_of(qpos_final, targets_np, num_envs, nobj):
  s = 0.0
  for w in range(num_envs):
    for o in range(nobj):
      p = qpos_final[w, 7 * o:7 * o + 2]
      s += float(np.sum((p - targets_np[w * nobj + o]) ** 2))
  return s


def fd_grad(m, mjm, mjd, num_envs, nobj, targets_np, v0, eps=1e-3):
  g = np.zeros(v0.shape[0])
  for j in range(v0.shape[0]):
    vp = v0.copy(); vp[j] += eps
    vm = v0.copy(); vm[j] -= eps
    lp = loss_of(rollout(m, mjm, mjd, num_envs, nobj, vp)[:, -1], targets_np, num_envs, nobj)
    lm = loss_of(rollout(m, mjm, mjd, num_envs, nobj, vm)[:, -1], targets_np, num_envs, nobj)
    g[j] = (lp - lm) / (2 * eps)
  return g


def analytic_grad(m, mjm, mjd, num_envs, nobj, v, targets_wp):
  """Batched taped rollout + adjoint.py backward -> populates v.grad. Returns (qpos (num_envs,T+1,nq), loss)."""
  datas = [mjw.put_data(mjm, mjd, nworld=num_envs) for _ in range(T + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  datas[0].qvel = wp.zeros((num_envs, mjm.nv), dtype=float, requires_grad=True)
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  v.grad.zero_()
  tape = wp.Tape()
  with tape:
    wp.launch(_set_qvel, dim=(num_envs, nobj), inputs=[v, nobj], outputs=[datas[0].qvel])
    for t in range(T):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_sum_sq_err, dim=(num_envs, nobj), inputs=[datas[T].qpos, targets_wp, nobj], outputs=[loss])
  tape.backward(loss=loss)
  qpos = np.transpose(np.array([datas[t].qpos.numpy() for t in range(T + 1)]), (1, 0, 2))
  return qpos, float(loss.numpy()[0])


def optimize(objects, num_envs, spread, grad_mode, steps, lr):
  mjm = mujoco.MjModel.from_xml_string(slide_xml(objects))
  mjd = mujoco.MjData(mjm); mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  nobj = len(objects)
  targets_np = fan_targets(objects, num_envs, spread)
  targets_wp = wp.array(targets_np.astype(np.float32), dtype=wp.vec2)
  v0 = np.tile(np.concatenate([o["v0"] for o in objects]), num_envs).astype(np.float32)  # (num_envs*nobj*2,)
  v = wp.array(v0, dtype=float, requires_grad=True)
  opt = warp.optim.Adam([v], lr=lr, betas=(0.7, 0.95))

  history, nan_warned = [], False
  for it in range(steps):
    vnp = v.numpy().copy()
    if grad_mode == "analytic":
      qpos, L = analytic_grad(m, mjm, mjd, num_envs, nobj, v, targets_wp)
    else:
      qpos = rollout(m, mjm, mjd, num_envs, nobj, vnp)
      L = loss_of(qpos[:, -1], targets_np, num_envs, nobj)
      v.grad = wp.array(fd_grad(m, mjm, mjd, num_envs, nobj, targets_np, vnp).astype(np.float32), dtype=float)
    if not np.isfinite(v.grad.numpy()).all():
      if not nan_warned:
        print(f"  WARNING: {grad_mode} gradient is NaN/Inf (iter {it}) -- zeroing; optimizer will stall.")
        nan_warned = True
      v.grad.zero_()
    history.append({"it": it, "loss": L, "qpos": qpos, "v": vnp})
    if it % 10 == 0 or it == steps - 1:
      print(f"  [{it:3d}] total_loss={L:.5f} mean/env={L / num_envs:.5f} |g|={np.linalg.norm(v.grad.numpy()):.3g}")
    opt.step([v.grad])

  best = min(range(len(history)), key=lambda k: history[k]["loss"])
  print(f"[slide {'+'.join(o['name'] for o in objects)} x{num_envs}env/{grad_mode}] "
        f"total_loss {history[0]['loss']:.5f} -> best {history[best]['loss']:.6f} (iter {best})")
  return history, best, targets_np


def _env_offsets(num_envs, cols=ENV_COLS, spacing=ENV_SPACING):
  off = np.zeros((num_envs, 3))
  if num_envs > 1:
    rows = int(np.ceil(num_envs / cols))
    for e in range(num_envs):
      r, c = e // cols, e % cols
      off[e, 0] = (c - (cols - 1) / 2.0) * spacing
      off[e, 1] = (r - (rows - 1) / 2.0) * spacing
  return off


def viz_xml_multi(objects, offsets, targets_np):
  """Viz scene = the puck(s) + bullseye rings REPLICATED per env, offset into a grid. Env-major body
  order (env outer, puck inner) matches the flattened batched qpos. Bodies are decorative (contype=0):
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


def render(objects, history, best, targets_np, num_envs, grad_mode, out_path, live=False):
  nobj = len(objects)
  offsets = _env_offsets(num_envs)
  vm = mujoco.MjModel.from_xml_string(viz_xml_multi(objects, offsets, targets_np))
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
  label = f"{'ADJOINT' if grad_mode == 'analytic' else 'FINITE DIFF'} (sliding)"
  R.emit(vm, vd, cam, frames, out_path=out_path, label=label, w=W, h=H, live=live)
  if live:
    return
  import imageio.v2 as imageio
  from PIL import Image
  rd = imageio.get_reader(out_path); fr = [f for f in rd]; rd.close()
  sel = [fr[i] for i in np.linspace(0, len(fr) - 1, 10).astype(int)]
  hh, ww, _ = sel[0].shape
  grid = Image.new("RGB", (5 * ww, 2 * hh), (0, 0, 0))
  for kk, f in enumerate(sel):
    grid.paste(Image.fromarray(f), ((kk % 5) * ww, (kk // 5) * hh))
  png = os.path.splitext(out_path)[0] + "_montage.png"
  grid.save(png)
  print(f"[slide] video -> {out_path}\n[slide] montage -> {png}")


def default_out(geom):
  name = "slide.mp4" if geom == "box" else f"slide_{geom}.mp4"
  return os.path.join(os.path.dirname(__file__), "reports", "assets", name)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--geom", choices=["box", "cylinder", "both"], default="box")
  ap.add_argument("--grad", choices=["analytic", "fd"], default="analytic")
  ap.add_argument("--envs", type=int, default=1)
  ap.add_argument("--spread", type=float, default=SPREAD, help="per-env target fan half-angle (rad)")
  ap.add_argument("--steps", type=int, default=STEPS)
  ap.add_argument("--lr", type=float, default=LR)
  ap.add_argument("--out", default=None)
  ap.add_argument("--no-render", action="store_true")
  ap.add_argument("--live", action="store_true", help="live MuJoCo viewer instead of mp4")
  args = ap.parse_args()

  objects = objects_for(args.geom)
  history, best, targets_np = optimize(objects, args.envs, args.spread, args.grad, args.steps, args.lr)
  if args.no_render:
    return
  out = args.out or os.environ.get("MJW_RENDER_PATH") or default_out(args.geom)
  os.makedirs(os.path.dirname(out), exist_ok=True)
  render(objects, history, best, targets_np, args.envs, args.grad, out, live=args.live)


if __name__ == "__main__":
  main()
