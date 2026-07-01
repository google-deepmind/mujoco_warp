"""Incline friction OPTIMIZATION -- 4 boxes on the ground plane, in parallel.

Each box sits FLAT on the ground and slides "down-slope" under an incline modeled as TILTED GRAVITY (the
box stays axis-aligned on a flat plane while a constant down-slope pull acts on it). The 4 envs start at
4 different (too-low) friction values, so they slide different distances; we optimize each env's surface
friction mu -- via the analytic contact-PARAMETER gradient d(loss)/d(geom_friction) from differentiable
mujoco_warp -- until every box STOPS moving (mu -> ~tan(theta)). One batched (nworld=4) wp.Tape gives all
four per-env gradients from a single backward (each env's mu only affects its own world), exactly like
contrib/diffsim/bounce.py's parallel-env optimization.

WHY THIS WORKS NOW: the `geom_friction -> contact.friction` combine (collision_core.contact_params) used to
be dropped (unrecorded forward copy -> analytic mu-gradient == 0). adjoint.contact_residual_backward now
exposes the cone leaf's d(phi)/d(contact.friction) and chains it back through that combine
(constraint_adjoint._contact_friction_geom_vjp) -> d(loss)/d(geom_friction). Single-step FD-EXACT, and on a
continuously-sliding box the multi-step BPTT grad stays FD-exact at every horizon.

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/incline_sysid.py
"""

import os
import sys

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src import adjoint  # noqa: F401  registers the analytic step() backward

sys.path.insert(0, os.path.dirname(__file__))
import viz  # noqa: E402  shared renderer (mp4 or live MuJoCo viewer via MJW_VIEWER)

THETA = 25.0  # incline angle (deg), applied as tilted GRAVITY; mu_crit = tan(25) ~ 0.466
GX = 9.81 * float(np.sin(np.radians(THETA)))  # +x down-slope pull
GZ = -9.81 * float(np.cos(np.radians(THETA)))  # into-the-floor component
T = 140  # rollout steps @ dt=0.004 -> 0.56s (long episode -> the low-mu box slides ~0.6m down the ramp)
NUM_ENVS = int(os.environ.get("MJW_NUM_ENVS", 4))
MU_INIT = np.array([0.35, 0.25, 0.15, 0.05])  # 4 too-low starting frictions, HIGH->low so the low-mu box (slides most) is in the foremost lane
LR = float(os.environ.get("MJW_LR", 0.1))  # small enough that the long-episode grad (~T^2) approaches mu_crit gradually
STEPS = int(os.environ.get("MJW_STEPS", 16))
LANE = 1.15  # y-spacing between the 4 box lanes (wide enough that the ~0.6m slides don't overlap in view)
OUT_MP4 = os.environ.get("MJW_RENDER_PATH", os.path.join(os.path.dirname(__file__), "reports", "assets", "incline_friction.mp4"))
W, H, FPS = 1024, 768, 30
_TAN = float(np.tan(np.radians(THETA)))

# The physics is a FLAT plane + tilted gravity (so the box stays flat and the mu-gradient is exact). For the
# RENDER we rotate the whole scene by R_y(theta) -- tilt the viz plane and rotate each box's pose -- so it
# visually reads as a box on a ramp sliding down-slope (+x physics -> down the visual ramp). Physics unchanged.
_TH = float(np.radians(THETA))
_RY = np.array([[np.cos(_TH), 0.0, np.sin(_TH)], [0.0, 1.0, 0.0], [-np.sin(_TH), 0.0, np.cos(_TH)]])
_QRY = np.array([np.cos(_TH / 2), 0.0, np.sin(_TH / 2), 0.0])  # quat (w,x,y,z) of R_y(theta)


def _quat_mul(a, b):  # (w,x,y,z)
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                   aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def slide_xml(viz_scene=False, n_lanes=0):
  """FLAT plane + box(es). Physics scene = 1 box (batched over worlds); viz scene = n_lanes free boxes
  laid out along y (one per env). The incline is TILTED GRAVITY (+x down-slope); box-plane contact."""
  head = f"""
  <option timestep="0.004" cone="elliptic" integrator="implicitfast" gravity="{GX:.5f} 0 {GZ:.5f}"
          solver="Newton" iterations="50"><flag eulerdamp="disable"/></option>
  <default><geom condim="3" friction="0.1 0.005 0.0001" solimp="0 0.95 0.001"/></default>"""
  if not viz_scene:
    return f"""<mujoco>{head}
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.01"/>
    <body name="box" pos="-0.25 0 0.1"><joint type="free"/>
      <geom name="box" type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
</mujoco>"""
  boxes = ""
  for e in range(n_lanes):
    boxes += (f'<body name="box{e}" pos="-0.25 {(e-(n_lanes-1)/2)*LANE:.3f} 0.1"><joint type="free"/>'
              f'<geom type="box" size="0.1 0.1 0.1" mass="1" rgba="0.9 0.7 0.3 1"/></body>\n    ')
  return f"""<mujoco>{head}
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.25 0.35 1"/><global offwidth="{W}" offheight="{H}"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="8 8" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="-1 -1 2" dir="0.4 0.4 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="5 5 0.01" material="grid" euler="0 {THETA} 0"/>
    {boxes}
  </worldbody>
</mujoco>"""


def _mu_geom_array(mjm, mu_B):
  """A per-world geom_friction override (num_envs, ngeom) vec3 with the slide column set to each env's mu."""
  base = np.tile(mjm.geom_friction.copy(), (len(mu_B), 1, 1))  # (E, ngeom, 3)
  base[:, :, 0] = mu_B[:, None]
  return base


def settle_qrest(mjm):
  """Rest the box flat on the floor at high friction (so it does NOT slide during settling) -> the clean,
  axis-aligned start pose shared by every env."""
  m2 = mujoco.MjModel.from_xml_string(slide_xml())
  m2.geom_friction[:, 0] = 1.2  # > mu_crit -> stays put
  d = mujoco.MjData(m2)
  mujoco.mj_forward(m2, d)
  for _ in range(40):
    mujoco.mj_step(m2, d)
  return d.qpos.copy(), float(d.qpos[0])


@wp.kernel
def _disp_loss_batched(qpos: wp.array2d[float], x_rest: float, loss: wp.array(dtype=float)):
  w = wp.tid()  # each env's mu only affects its own world -> one backward = per-env grads
  d = qpos[w, 0] - x_rest  # down-slope (+x) displacement of env w's box COM from the flat rest pose
  wp.atomic_add(loss, 0, d * d)


def taped_grad_batched(m, mjm, mjd, qpos0_B, x_rest, mu_B):
  """Batched (nworld=E) taped rollout at per-env friction mu_B; loss = sum_w (down-slope disp_w)^2.
  Overrides m.geom_friction with a per-world (E, ngeom) array (requires_grad), back-props once. Returns
  (per-env loss[E], per-env grad dmu[E], per-env box qpos trajectory[E, T+1, nq])."""
  E = len(mu_B)
  m.geom_friction = wp.array(_mu_geom_array(mjm, mu_B).astype(np.float32), dtype=wp.vec3, requires_grad=True)
  datas = [mjw.put_data(mjm, mjd, nworld=E) for _ in range(T + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  datas[0].qpos = wp.array(qpos0_B.astype(np.float32), dtype=wp.float32, requires_grad=True)
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  per_env = wp.zeros(E, dtype=float)
  tape = wp.Tape()
  with tape:
    for t in range(T):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_disp_loss_batched, dim=E, inputs=[datas[T].qpos, float(x_rest)], outputs=[loss])
  tape.backward(loss=loss)
  qpos = np.array([datas[t].qpos.numpy() for t in range(T + 1)]).transpose(1, 0, 2)  # (E, T+1, nq)
  g = m.geom_friction.grad
  dmu = np.zeros(E) if g is None else np.nan_to_num(g.numpy())[:, :, 0].sum(axis=1)  # per-env sum over geoms
  losses = ((qpos[:, -1, 0] - x_rest) ** 2)
  return losses, dmu, qpos


def optimize(num_envs=NUM_ENVS, steps=STEPS, lr=LR):
  mjm = mujoco.MjModel.from_xml_string(slide_xml())
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  qrest, x_rest = settle_qrest(mjm)
  qpos0_B = np.tile(qrest, (num_envs, 1))
  m = mjw.put_model(mjm)
  print(f"[incline] theta={THETA} deg (tilted gravity)  mu_crit=tan(theta)={_TAN:.3f}  {num_envs} envs  rest x={x_rest:.3f}")
  mu = MU_INIT[:num_envs].copy() if num_envs <= len(MU_INIT) else np.linspace(0.05, 0.35, num_envs)
  history = []
  for it in range(steps):
    losses, g, qpos = taped_grad_batched(m, mjm, mjd, qpos0_B, x_rest, mu)
    history.append({"it": it, "mu": mu.copy(), "losses": losses.copy(), "qpos": qpos})
    if it % 4 == 0 or it == steps - 1:
      print(f"  [{it:3d}] mu={np.array2string(mu, precision=3)}  slid={np.array2string(np.sqrt(losses), precision=3)}m  |g|~{np.abs(g).mean():.2e}")
    mu = np.clip(mu - lr * g, 0.02, 1.2)
  best = int(np.array([h["losses"].mean() for h in history]).argmin())
  print(f"[incline friction] {num_envs} envs: mean slid {np.sqrt(history[0]['losses']).mean():.4f}m "
        f"-> {np.sqrt(history[best]['losses']).mean():.4f}m; mu -> {np.array2string(history[best]['mu'], precision=3)} (crit {_TAN:.3f})")
  return history, best


def render(history, best, out_path=OUT_MP4):
  """Animate all envs sliding together per shown iteration, boxes laid out in y-lanes, each box's down-slope
  trail drawn loss-colored (Bourke: red=slides far -> blue=stopped); prior iterations persist faintly."""
  E = history[0]["qpos"].shape[0]
  vm = mujoco.MjModel.from_xml_string(slide_xml(viz_scene=True, n_lanes=E))
  vd = mujoco.MjData(vm)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(vm, cam)
  cam.lookat = [0.05, 0.0, 0.02]
  cam.distance = 4.6  # frame the wide 4-lane layout + the ~0.6m slide (zoomed in a bit)
  cam.azimuth = 68.0  # original view; mu ordered HIGH->low so the low-friction box is in the foremost lane
  cam.elevation = -20.0
  lanes = (np.arange(E) - (E - 1) / 2.0) * LANE
  hi = float(max(h["losses"].max() for h in history)) or 1.0
  last = len(history) - 1
  show = [k for k in sorted(set([0, 1, 2, 3, 5, 8, best, last])) if k < len(history)]

  def env_qpos(h, e, t):  # box e's pose at step t, shifted into its lane and rotated onto the visual ramp
    q = h["qpos"][e, t].copy()
    p = q[0:3] + np.array([0.0, lanes[e], 0.0])
    return np.concatenate([_RY @ p, _quat_mul(_QRY, q[3:7])])

  frames = []
  persisted = []  # (list-of-per-env-trail-xyz, colors) of completed shown iterations
  for k in show:
    h = history[k]
    cols = [viz.bourke_color_map(0.0, hi, float(v)) for v in h["losses"]]
    sub = f"iter {h['it']:3d}    mu {np.array2string(h['mu'][::-1], precision=2)}    mean slid {np.sqrt(h['losses']).mean():.3f}m"  # header lists mu low->high (scene unchanged)
    trails = [(_RY @ (h["qpos"][e, :, 0:3] + np.array([0.0, lanes[e], 0.0])).T).T for e in range(E)]
    steps_idx = list(range(0, T + 1, 3)) + [T]
    hold = 22 if k == best else 0
    for t in steps_idx + [T] * hold:
      qpos = np.concatenate([env_qpos(h, e, t) for e in range(E)])
      snap = list(persisted)

      def draw(scene, snap=snap, cur=[tr[: t + 1] for tr in trails], cols=cols):
        for ptrails, pcols in snap:
          for e in range(E):
            viz.add_polyline(scene, ptrails[e], pcols[e], width=0.006, alpha=0.25)
        for e in range(E):
          viz.add_polyline(scene, cur[e], cols[e], width=0.012)

      frames.append((qpos, draw, sub))
    persisted.append((trails, cols))
  if frames:
    frames += [frames[-1]] * 20
  return viz.emit(vm, vd, cam, frames, out_path=out_path, label="ADJOINT (incline friction)",
                  w=W, h=H, fps=FPS)


def main():
  history, best = optimize()
  render(history, best)


if __name__ == "__main__":
  main()
