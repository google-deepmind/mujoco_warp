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

  # analytic adjoint gradient (default) -> reports/assets/cradle_parallel.mp4
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/cradle.py
  # finite-difference baseline -> reports/assets/cradle_fd.mp4
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/cradle.py --grad fd
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
import viz as R  # noqa: E402  (bourke_color_map, add_polyline, default_show, emit)

N, R_BALL, L = 5, 0.1, 0.5
SPACING = 2.0 * R_BALL
PIVOT_X = [SPACING * i for i in range(N)]
MID = N // 2  # the ball we want to keep at rest (index 2)
DT, T = 0.002, 80
V0_FIXED = -3.0  # left end ball inward angular velocity (the fixed "given" push)
V4_INIT = 1.2  # right end ball inward angular velocity = the OPTIMIZED variable (optimum ~ +3.0)
LR, STEPS = 0.5, 60
SOLREF = "-50000 -10"  # stiff elastic ball<->ball (matches adjoint_test2 _cradle_xml)
W, H = 1024, 768
_BALL_RGBA = ["0.85 0.2 0.2 1"] + ["0.7 0.7 0.75 1"] * (N - 2) + ["0.2 0.4 0.85 1"]  # red end, gray mid, blue end

# parallel-env layout (a COLS-wide grid of cradles; default 4 envs = 2x2) + per-env v4 fan width
NUM_ENVS = int(os.environ.get("MJW_CRADLE_NUM_ENVS", "4"))
ENV_COLS = int(os.environ.get("MJW_CRADLE_COLS", "2"))
ENV_SPACING = float(os.environ.get("MJW_CRADLE_ENV_SPACING", "2.2"))  # grid pitch (> cradle width)
V4_SPREAD = float(os.environ.get("MJW_CRADLE_V4_SPREAD", "0.6"))

# --- viz frame (decorative only; contype=0/conaffinity=0, so it never touches the physics) ---
FRAME_HALF_Y = 0.22    # front/back rail spacing -> width of the V-string suspension (swing stays in x-z)
FRAME_MARGIN_X = 0.10  # top/bottom rails run this far past the end pivots
BASE_Z = -0.18         # frame base: posts stand from here up to the pivot rail at z=L
FLOOR_Z = -0.20        # grid floor just under the base so each cradle stands on it

ASSETS = os.path.join(os.path.dirname(__file__), "reports", "assets")  # default output dir


def _viz_assets():
  """<asset> for the viz scenes: our dark-blue skybox + checker grid, plus metallic materials that
  keep our palette -- red/gray/blue balls (specular so they read as polished), a brushed-steel frame,
  and light emissive strings that stay visible even as thin capsules."""
  return f"""
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


def _viz_visual():
  """<visual> for the viz scenes (unchanged from our theme): soft headlight, haze, offscreen size."""
  return f"""
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="{W}" offheight="{H}"/>
  </visual>"""


def _ball_mat(i):
  """Our colors: red left end, blue right end, gray in between."""
  if i == 0:
    return "ball_red"
  if i == N - 1:
    return "ball_blue"
  return "ball_gray"


def _frame_geoms(ox, oy, oz):
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


def _ball_body_viz(prefix, i, px, oy, oz):
  """One suspended ball for the viz: the SAME hinge DOF as the physics scene + our-colored ball +
  a V of two strings to the front/back rails (replacing the physics rod). The string tops sit on the
  hinge axis (body-local (0, +-FRAME_HALF_Y, 0)), so they stay pinned to the rails as the ball swings."""
  return (
    f'<body name="{prefix}b{i}" pos="{px:.4f} {oy:.4f} {L + oz:.4f}">'
    f'<joint name="{prefix}h{i}" type="hinge" axis="0 1 0"/>'
    f'<geom name="{prefix}ball{i}" type="sphere" size="{R_BALL}" pos="0 0 -{L}" material="{_ball_mat(i)}" mass="1"/>'
    f'<geom type="capsule" size="0.0045" material="string_mat" contype="0" conaffinity="0" fromto="0 0 -{L}  0 {-FRAME_HALF_Y:.4f} 0"/>'
    f'<geom type="capsule" size="0.0045" material="string_mat" contype="0" conaffinity="0" fromto="0 0 -{L}  0 {FRAME_HALF_Y:.4f} 0"/>'
    f"</body>"
  )


def cradle_xml():
  """The minimal PHYSICS model: 5 balls on hinge rods, elastic ball<->ball contact. This is the only
  scene that gets stepped/differentiated (rollout + the batched nworld backward); rendering uses
  _cradle_xml_multi instead. No frame/floor/lights here on purpose -- keep the diff'd model minimal."""
  bodies = ""
  for i in range(N):
    rod_rgba = "0.15 0.15 0.17 1"
    bodies += (
      f'<body name="b{i}" pos="{PIVOT_X[i]:.4f} 0 {L:.4f}">'
      f'<joint name="h{i}" type="hinge" axis="0 1 0"/>'
      f'<geom type="capsule" fromto="0 0 0 0 0 -{L}" size="0.006" rgba="{rod_rgba}" contype="0" conaffinity="0" mass="0.02"/>'
      f'<geom name="ball{i}" type="sphere" size="{R_BALL}" pos="0 0 -{L}" mass="1" rgba="{_BALL_RGBA[i]}"/>'
      f"</body>"
    )
  opt = (
    f'<option timestep="{DT}" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
    f'solver="Newton" iterations="100" ls_iterations="50" tolerance="1e-10"><flag eulerdamp="disable"/></option>'
  )
  default = f'<default><geom condim="1" solref="{SOLREF}" solimp="0 0.95 0.001"/></default>'
  return f"<mujoco>{opt}{default}<worldbody>{bodies}</worldbody></mujoco>"


def rollout(v4):
  """mj_step rollout with end-ball velocities (V0_FIXED, v4). Returns (mid_xyz[T+1,3], qpos[T+1,N], loss).
  loss = (middle ball's final hinge speed)^2 -> 0 when the impact cancels at the middle."""
  m = mujoco.MjModel.from_xml_string(cradle_xml())
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"ball{MID}")
  d.qvel[:] = 0.0
  d.qvel[0] = V0_FIXED
  d.qvel[N - 1] = v4
  mid_xyz = [d.geom_xpos[gid].copy()]
  qpos = [d.qpos.copy()]
  for _ in range(T):
    mujoco.mj_step(m, d)
    mid_xyz.append(d.geom_xpos[gid].copy())
    qpos.append(d.qpos.copy())
  loss = float(d.qvel[MID] ** 2)
  return np.array(mid_xyz), np.array(qpos), loss


# --- gradients: FD (per-env central difference) and analytic (batched mjwarp adjoint) ---


def fd_grad(v4, eps=1e-3):
  """Central-difference d(loss)/d(v4) for one cradle (the robust baseline)."""
  return (rollout(v4 + eps)[2] - rollout(v4 - eps)[2]) / (2 * eps)


@wp.kernel
def _mid_vel_sq_batched(qvel: wp.array2d[float], idx: int, loss: wp.array(dtype=float)):
  """loss = sum_w (middle-ball hinge speed_w)^2. Each env's v4 only drives its own world, so
  d(sum)/d(v4_w) = d(loss_w)/d(v4_w): one backward -> independent per-env gradients."""
  w = wp.tid()
  wp.atomic_add(loss, 0, qvel[w, idx] * qvel[w, idx])


def batched_analytic_grad(m, mjm, mjd, v4s):
  """Per-env d(loss)/d(v4) via a batched (nworld=B) taped multi-body backward (adjoint.py).
  v4s: (B,). Returns g: (B,) -- the gradient w.r.t. each env's right-end ball velocity qvel0[N-1]."""
  num_envs = len(v4s)
  qvel0 = np.zeros((num_envs, N))
  qvel0[:, 0] = V0_FIXED
  qvel0[:, N - 1] = v4s
  datas = [mjw.put_data(mjm, mjd, nworld=num_envs) for _ in range(T + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  datas[0].qvel = wp.array(qvel0.astype(np.float32), dtype=float, requires_grad=True)
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(T):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_mid_vel_sq_batched, dim=num_envs, inputs=[datas[T].qvel, MID], outputs=[loss])
  tape.backward(loss=loss)
  return np.nan_to_num(datas[0].qvel.grad.numpy())[:, N - 1]


def _make_grad_fn(grad_method, mjm, mjd):
  """Return grad_fn(v4s: (B,)) -> g: (B,), d(loss)/d(v4) per env. 'fd' = per-env central difference
  (MuJoCo-C); 'analytic' = one batched taped mjwarp backward (adjoint.py)."""
  if grad_method == "fd":
    return lambda v4s: np.array([fd_grad(v) for v in v4s])
  if grad_method == "analytic":
    m = mjw.put_model(mjm)  # build the warp model once, reuse across iterations
    return lambda v4s: batched_analytic_grad(m, mjm, mjd, v4s)
  raise ValueError(f"unknown grad method: {grad_method!r} (want 'fd' or 'analytic')")


def _init_v4(num_envs, spread):
  """Per-env initial v4: a fan from a low anchor up toward (but kept clearly BELOW) the +3.0 cancel
  optimum, so every lane starts with the middle ball visibly knocked away (|v4 - 3| >= ~1.0) and
  converges upward -- not pre-solved. `spread` widens the fan; the top is capped < 3.0 so a wide
  spread can't overshoot the optimum. The low anchor 0.8 stays above the ~0.7 wrong-basin floor
  (below which the right ball swings outward and that env diverges). num_envs==1 -> the FD baseline's
  V4_INIT so a single-cradle run matches the classic single-env optimization."""
  if num_envs == 1:
    return np.array([V4_INIT], dtype=np.float64)
  lo, hi = 0.8, min(0.8 + 2.0 * spread, 2.2)  # cap < the +3.0 optimum -> no lane starts solved
  return np.linspace(lo, hi, num_envs)


def optimize(grad_method="analytic", num_envs=NUM_ENVS, steps=STEPS, lr=LR, spread=V4_SPREAD):
  """Parallel gradient descent: num_envs cradles, each optimizing its own v4, all driven per
  iteration by one gradient call (batched analytic backward, or per-env FD)."""
  mjm = mujoco.MjModel.from_xml_string(cradle_xml())
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  grad_fn = _make_grad_fn(grad_method, mjm, mjd)

  v4 = _init_v4(num_envs, spread)
  history = []
  for it in range(steps):
    # per-env MuJoCo-C rollout for the trajectory/video + loss display
    mid_xyz = np.empty((num_envs, T + 1, 3))
    qpos = np.empty((num_envs, T + 1, N))
    losses = np.empty(num_envs)
    for e in range(num_envs):
      mid_xyz[e], qpos[e], losses[e] = rollout(v4[e])
    g = grad_fn(v4)  # one gradient call for all envs
    history.append({"it": it, "v4s": v4.copy(), "losses": losses.copy(), "mid_xyz": mid_xyz, "qpos": qpos})
    if it % 10 == 0:
      print(f"  [{it:3d}] v4 {np.round(v4, 3)}  mean_loss={losses.mean():.5f} best={losses.min():.5f}")
    v4 -= lr * g  # per-env update

  mean_losses = np.array([h["losses"].mean() for h in history])
  best = int(mean_losses.argmin())
  tag = "analytic (adjoint, body<->body S3)" if grad_method == "analytic" else "finite diff"
  print(
    f"[cradle optim / {tag}] {num_envs} envs: mean_loss {mean_losses[0]:.5f} -> best {mean_losses[best]:.5f} "
    f"(iter {best})"
  )
  return history, best


# --- multi-env rendering (a grid of cradles, one lane per env, shared loss colormap) ---


def _env_offsets(num_envs, cols=ENV_COLS, spacing=ENV_SPACING):
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


def _cradle_xml_multi(offsets):
  """Cradle viz scene REPLICATED per env: each lane is a standing frame (two U-rails + posts + feet)
  with five V-string-suspended balls in our colors (red/gray/blue ends) and a purple rest-target,
  over one shared checker floor. contype=0 throughout -- the render only plays back qpos, never steps."""
  opt = (
    f'<option timestep="{DT}" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
    f'solver="Newton" iterations="100" ls_iterations="50" tolerance="1e-10"><flag eulerdamp="disable"/></option>'
  )
  default = f'<default><geom condim="1" solref="{SOLREF}" solimp="0 0.95 0.001"/></default>'
  bar_cx = 0.5 * (PIVOT_X[0] + PIVOT_X[-1])
  mx = PIVOT_X[MID]
  lanes = ""
  for e, (ox, oy, oz) in enumerate(offsets):
    lanes += _frame_geoms(ox, oy, oz)
    lanes += (
      f'<geom type="sphere" size="0.045" pos="{mx + ox:.4f} {oy:.4f} {oz:.4f}" rgba="0.5 0 0.5 0.8" '
      f'contype="0" conaffinity="0"/>'
    )
    for i in range(N):
      lanes += _ball_body_viz(f"e{e}", i, PIVOT_X[i] + ox, oy, oz)
  fsize = max(8.0, 0.5 * (PIVOT_X[-1] - PIVOT_X[0]) + FRAME_MARGIN_X + float(np.abs(offsets).max()) + 5.0)
  return f"""
<mujoco>
  {opt}
  {default}
  {_viz_visual()}
  {_viz_assets()}
  <worldbody>
    <light pos="0.4 -1 2" dir="0 0.4 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <light pos="1.5 1 1.5" dir="-0.6 -0.4 -1" diffuse="0.35 0.35 0.4" specular="0.2 0.2 0.2"/>
    <geom name="floor" type="plane" pos="{bar_cx:.4f} 0 {FLOOR_Z}" size="{fsize:.3f} {fsize:.3f} .05"
          material="grid" contype="0" conaffinity="0"/>
    {lanes}
  </worldbody>
</mujoco>
"""


def render_video(history, best, out_path, sample_every=4, label="ADJOINT (cradle)", live=None):
  """Animate every cradle swinging together for selected iterations; trace each env's middle-ball
  path colored by its loss via the shared Bourke map (red=knocked away -> blue=stays put), with
  prior iterations' paths persisting faintly so the parallel fan visibly converges. Outputs an mp4,
  or the live MuJoCo viewer when `live=True` / `MJW_VIEWER=1` (via viz.emit)."""
  num_envs = history[0]["mid_xyz"].shape[0]
  offsets = _env_offsets(num_envs)
  lo, hi = 0.0, float(max(h["losses"].max() for h in history))

  def _loss_colors(losses):
    return [R.bourke_color_map(lo, hi, float(v)) for v in losses]

  vm = mujoco.MjModel.from_xml_string(_cradle_xml_multi(offsets))
  vd = mujoco.MjData(vm)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(vm, cam)
  rows = int(np.ceil(num_envs / ENV_COLS))
  ext_x = ENV_SPACING * (ENV_COLS - 1) + (PIVOT_X[-1] - PIVOT_X[0])
  ext_y = ENV_SPACING * (rows - 1)
  cam.lookat = [0.5 * (PIVOT_X[0] + PIVOT_X[-1]), 0.0, 0.2]  # grid is centered on 0
  cam.distance = max(2.8, 1.1 * max(ext_x, ext_y) + 2.2)
  cam.azimuth = 60.0  # 3/4 view so the grid's depth (rows) reads, not a flat row
  cam.elevation = -22.0

  best_mean = min(h["losses"].mean() for h in history)
  frames = []
  persisted = []  # (mid_xyz_all[B,T+1,3], colors[B]) of completed shown iterations
  for k in R.default_show(len(history), best):
    hk = history[k]
    cols = _loss_colors(hk["losses"])
    sub = (
      f"iter {hk['it']:3d}    mean loss {hk['losses'].mean():.4f}    "
      f"best {best_mean:.4f}    envs {num_envs}"
    )
    steps_idx = list(range(0, T + 1, sample_every)) + [T]
    hold = 20 if k == best else 0
    for t in steps_idx + [T] * hold:
      snap = list(persisted)  # snapshot of prior shown iterations at this frame
      cur = [hk["mid_xyz"][e, : t + 1] + offsets[e] for e in range(num_envs)]
      qpos = hk["qpos"][:, t, :].reshape(-1).copy()  # env-major: [env0 h0..h4, env1 ...]

      def draw(scene, snap=snap, cur=cur, cols=cols):
        for mid_all, pcols in snap:  # prior shown iterations stay on screen, faint
          for e in range(num_envs):
            R.add_polyline(scene, mid_all[e, ::2] + offsets[e], pcols[e], width=0.006)
        for e in range(num_envs):  # this iteration: each lane's middle-ball trail (in its lane)
          R.add_polyline(scene, cur[e], cols[e], width=0.016)

      frames.append((qpos, draw, sub))
    persisted.append((hk["mid_xyz"], cols))
  if frames:
    frames += [frames[-1]] * 20

  return R.emit(vm, vd, cam, frames, out_path=out_path, label=label, w=W, h=H, fps=30, live=live)


def save_montage(mp4_path, png_path, ncols=5):
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


def _default_out(grad_method):
  """reports/assets/cradle_parallel.mp4 for analytic (the report asset), cradle_fd.mp4 for FD."""
  name = "cradle_parallel" if grad_method == "analytic" else "cradle_fd"
  return os.path.join(ASSETS, name + ".mp4")


def main():
  ap = argparse.ArgumentParser(description="Newton's-cradle optimization video (FD or mjwarp adjoint gradient).")
  ap.add_argument("--grad", choices=["analytic", "fd"], default="analytic",
                  help="gradient source: 'analytic' = differentiable mujoco_warp adjoint (default); 'fd' = finite diff")
  ap.add_argument("--num-envs", type=int, default=NUM_ENVS, help="parallel cradles optimized at once")
  ap.add_argument("--out", default=None,
                  help="output mp4 path (default: reports/assets/cradle_{parallel,fd}.mp4; MJW_RENDER_PATH also honored)")
  ap.add_argument("--live", action="store_true", help="open the live MuJoCo viewer instead of writing an mp4")
  args = ap.parse_args()

  out = args.out or os.environ.get("MJW_RENDER_PATH") or _default_out(args.grad)
  live = True if args.live else None  # None -> honor MJW_VIEWER
  os.makedirs(os.path.dirname(out), exist_ok=True)

  history, best = optimize(grad_method=args.grad, num_envs=args.num_envs)
  label = "ADJOINT (cradle)" if args.grad == "analytic" else "FINITE DIFF (cradle)"
  written = render_video(history, best, out_path=out, label=label, live=live)
  if written:  # mp4 written (video mode); skip montage when showing the live viewer
    save_montage(written, os.path.splitext(written)[0] + "_montage.png")


if __name__ == "__main__":
  main()
