"""Incline friction sys-id -- 4 boxes on a flat plane under TILTED GRAVITY, optimized IN PARALLEL.

Each box slides "down-slope" (an incline modeled as tilted gravity so the box stays axis-aligned on a
flat plane). The 4 envs start at 4 too-low friction values and slide different distances; we optimize
each env's surface friction mu -- via the analytic contact-PARAMETER gradient d(loss)/d(geom_friction)
from differentiable mujoco_warp -- until every box STOPS (mu -> ~tan(theta)). loss = (final down-slope
displacement)^2, a TERMINAL loss; the optimized leaf is the MODEL parameter m.geom_friction, whose grad
accumulates across the checkpointed segments (it acts every step). One batched (nworld=4) backward ->
per-env gradients (each env's mu only affects its own world), exactly like bounce/cradle.

The `geom_friction -> contact.friction` combine (collision_core.contact_params) is now differentiable
(adjoint.contact_residual_backward + constraint_adjoint._contact_friction_geom_vjp), so the mu-gradient
is single-step FD-exact and stays FD-exact over the multi-step BPTT of a continuously-sliding box.

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/incline_sysid.py
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/incline_sysid.py --grad=fd
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

THETA = 25.0  # incline angle (deg), applied as tilted GRAVITY; mu_crit = tan(25) ~ 0.466
GX = 9.81 * float(np.sin(np.radians(THETA)))  # +x down-slope pull
GZ = -9.81 * float(np.cos(np.radians(THETA)))  # into-the-floor component
T = 140  # rollout steps @ dt=0.004 -> 0.56s (long episode -> the low-mu box slides ~0.6m down the ramp)
MU_INIT = np.array([0.35, 0.25, 0.15, 0.05])  # 4 too-low starting frictions, HIGH->low so the low-mu box (slides most) is in the foremost lane
LANE = 1.15  # y-spacing between the 4 box lanes (wide enough that the ~0.6m slides don't overlap in view)
W, H, FPS = 1024, 768, 30
_TAN = float(np.tan(np.radians(THETA)))
ASSETS = os.path.join(os.path.dirname(__file__), "reports", "assets")

# The physics is a FLAT plane + tilted gravity (so the box stays flat and the mu-gradient is exact). For
# the RENDER we rotate the whole scene by R_y(theta) so it reads as a box sliding down a ramp. Physics
# unchanged.
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
  laid out along y. The incline is TILTED GRAVITY (+x down-slope); box-plane contact."""
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
  base[:, :, 0] = np.asarray(mu_B)[:, None]
  return base.astype(np.float32)


def settle_qrest(mjm):
  """Rest the box flat on the floor at high friction (so it does NOT slide while settling) -> the clean,
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


@dataclass
class Args(demo.CommonArgs):
  """incline config: CommonArgs (grad/num_envs/steps/lr/chunk/device/...) + incline defaults."""

  num_envs: int = field(default=4, metadata={"help": "parallel boxes (UNTILED 4-env close-up = the MU_INIT frictions)"})
  steps: int = field(default=16, metadata={"help": "gradient-descent steps"})
  lr: float = field(default=0.1, metadata={"help": "gradient-descent learning rate"})
  chunk: int = field(default=20, metadata={"help": "checkpoint segment length (divides T=140)"})
  # UNTILED 4-env view (user preferred the original few-env look over the dense field): one lane per env,
  # stacked across-slope in y like the native render()'s lanes -> 4 big, clearly readable boxes on ramps.
  usd_envs: int = field(default=4, metadata={"help": "USD field: lanes (untiled -> one per env)"})
  usd_cols: int = field(default=1, metadata={"help": "USD field: grid columns (1 -> across-slope y-lanes)"})
  usd_xpitch: float = field(default=1.8, metadata={"help": "USD field: column pitch (x, down-slope; unused at cols=1)"})
  usd_ypitch: float = field(default=1.15, metadata={"help": "USD field: lane pitch (y, across-slope) -- matches native LANE"})


class InclineDemo(demo.Example):
  """Incline friction sys-id: optimize each env's surface friction mu (MODEL param m.geom_friction) so
  the box stops. Terminal loss (final down-slope displacement^2); the friction leaf's grad accumulates
  across the checkpointed segments. Plain GD on the host mu, clipped to a valid range each step."""

  Args = Args

  # ---- harness hooks ----

  def build_model(self):
    self.mjm = mujoco.MjModel.from_xml_string(slide_xml())
    self.mjd = mujoco.MjData(self.mjm)
    mujoco.mj_forward(self.mjm, self.mjd)
    self.qrest, self.x_rest = settle_qrest(self.mjm)

  def init_params(self):
    ne = self.args.num_envs
    mu = MU_INIT[:ne].copy() if ne <= len(MU_INIT) else np.linspace(0.05, 0.35, ne)
    return mu[:, None]  # (E, 1) -- the optimized DOF is a single friction per env

  def build_datas(self):
    ne = self.args.num_envs
    self.nT = T  # checkpointed BPTT length (chunk+1 segment buffers)
    self.datas = [mjw.put_data(self.mjm, self.mjd, nworld=ne) for _ in range(self.args.chunk + 1)]
    for d in self.datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    self.datas[0].qpos = wp.array(np.tile(self.qrest, (ne, 1)).astype(np.float32), dtype=float, requires_grad=True)
    # the MODEL-parameter leaf: friction acts every step, so its grad accumulates across all segments
    self.m.geom_friction = wp.array(_mu_geom_array(self.mjm, self.params[:, 0]), dtype=wp.vec3, requires_grad=True)
    self.accum_leaf = self.m.geom_friction
    self.loss = wp.zeros(1, dtype=float, requires_grad=True)

  def set_params(self):
    self.m.geom_friction.assign(_mu_geom_array(self.mjm, self.params[:, 0]))  # host mu -> model field (in place)

  def chunk_step(self, i, t):
    del t  # terminal loss -> time-independent chunk
    mjw.step(self.m, self.datas[i], self.datas[i + 1])

  def terminal_loss(self):
    wp.launch(_disp_loss_batched, dim=self.args.num_envs, inputs=[self.datas[0].qpos, float(self.x_rest)], outputs=[self.loss])

  def read_grad(self):
    # accum_grad = sum over segments of d(loss)/d(geom_friction) (E, ngeom, 3); the optimized DOF is the
    # slide column summed over geoms (floor + box both carry it, so the MuJoCo combine yields mu).
    return self.accum_grad[:, :, 0].sum(axis=1)[:, None]  # (E, 1)

  def clip_params(self, p):
    return np.clip(p, 0.02, 1.2)  # keep mu in a physically valid range

  def rollout_env(self, p):
    """MuJoCo-C rollout at friction mu=p for one env; returns (box xyz[T+1,3], qpos[T+1,7], loss)."""
    m = mujoco.MjModel.from_xml_string(slide_xml())
    m.geom_friction[:, 0] = float(p[0])
    d = mujoco.MjData(m)
    d.qpos[:] = self.qrest
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
    xyz = [d.xpos[bid].copy()]
    qpos = [d.qpos.copy()]
    for _ in range(T):
      mujoco.mj_step(m, d)
      xyz.append(d.xpos[bid].copy())
      qpos.append(d.qpos.copy())
    xyz, qpos = np.array(xyz), np.array(qpos)
    return xyz, qpos, float((qpos[-1, 0] - self.x_rest) ** 2)

  def fd_grad(self, p, eps=1e-3):
    return np.array([(self.rollout_env(p + eps)[2] - self.rollout_env(p - eps)[2]) / (2 * eps)])

  def record(self, it, losses, trajs, qposs):
    return {"it": it, "mu": self.params[:, 0].copy(), "losses": losses.copy(), "qpos": qposs}

  def progress(self, it, losses, g):
    return (f"  [{it:3d}] mu={np.array2string(self.params[:, 0], precision=3)}  "
            f"slid={np.array2string(np.sqrt(losses), precision=3)}m  |g|~{np.abs(g).mean():.2e}")

  def summary(self, history, best):
    tag = "analytic (contact-param)" if self.args.grad == "analytic" else "finite diff"
    return (f"[incline friction / {tag}] {self.args.num_envs} envs: mean slid "
            f"{np.sqrt(history[0]['losses']).mean():.4f}m -> {np.sqrt(history[best]['losses']).mean():.4f}m; "
            f"mu -> {np.array2string(history[best]['mu'], precision=3)} (crit {_TAN:.3f})")

  def default_out(self):
    return os.path.join(ASSETS, "incline_friction.mp4")

  # ---- Blender FIELD export (instanced tilted-ramp vignettes; one per friction init) ----

  def _proto_xml(self, e):
    """Field proto for env e: a matte RAMP (thin BOX, tilted euler 0/THETA/0 -- strip_to_proto DROPS planes,
    so the ramp must be a box) + one sliding box named `box{e}` (per-env name so the render cycles a pastel
    palette). The ramp's TOP FACE lies on the plane-through-origin that the export's R_y(theta) rotation maps
    the physics floor (z=0) onto, so the rotated box poses land exactly ON the ramp. The box's free-joint qpos
    is overwritten every frame by export_usd; the body pos here is just a placeholder qpos0."""
    ht = 0.03                                             # ramp half-thickness
    nx, nz = float(np.sin(_TH)), float(np.cos(_TH))       # ramp top-face normal after R_y(theta): [sin,0,cos]
    cx, cz = -ht * nx, -ht * nz                           # center so the TOP face passes through the origin plane
    return f"""<mujoco>
  <visual><global offwidth="{W}" offheight="{H}"/></visual>
  <worldbody>
    <geom name="ramp" type="box" size="0.9 0.35 {ht}" euler="0 {THETA} 0" pos="{cx:.5f} 0 {cz:.5f}"/>
    <body name="box{e:03d}" pos="0 0 0.5"><joint type="free"/>
      <geom name="box{e:03d}" type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
</mujoco>"""

  def export_usd(self, history, best):
    """--export_usd hook: replay THIS friction sys-id across an instanced tilted-ramp FIELD for the Blender
    render (incline_sysid_render_blender.py). Each env started at a DIFFERENT too-low friction and slides a
    DIFFERENT distance before converging to mu_crit -> a diverse field. Uses the stored full MuJoCo-C rollout
    `history[k]["qpos"]` (E,T+1,7) directly (incline's `optimize` records the full trajectory, like render()),
    rotating each free-joint pose onto the visual ramp by R_y(theta) WITHOUT the render()'s per-lane y-shift
    (the field offsets handle the grid). A loss-colored skid ribbon per env tracks the down-slope slide."""
    a = self.args
    out_dir = os.path.join(ASSETS, "incline_sysid_render")
    last = len(history) - 1
    ks = sorted(set(np.linspace(0, last, a.usd_iters).astype(int).tolist()))  # spread across the run, end on converged
    ne = a.num_envs
    offsets = self._usd_grid_offsets(a.usd_envs, a.usd_cols, a.usd_xpitch, a.usd_ypitch)
    fps = max(1, round(1.0 / (self.mjm.opt.timestep * a.usd_stride)))  # 1x real-time (qpos is 1 step/frame)

    def rot(q):  # rotate a (7,) free-joint pose onto the visual ramp (no per-lane shift; offsets grid it)
      return np.concatenate([_RY @ q[0:3], _quat_mul(_QRY, q[3:7])])

    env_frames, env_ribbons, env_protos, frame_iters = [], [], [], None
    for e in range(ne):
      hi = max(float(history[0]["losses"][e]), 1e-9)  # env e's iter-0 slide^2 -> its own Bourke normalizer
      fr, fi, pts, pcols = [], [], [], []
      for k in ks:
        qp = history[k]["qpos"][e]  # (T+1, 7) MuJoCo-C rollout at iter k's mu
        col = viz.bourke_color_map(0.0, hi, float(history[k]["losses"][e]))  # far slide -> red, stopped -> blue
        sub = list(qp[:: a.usd_stride]) + [qp[-1]] * a.usd_hold
        for q in sub:
          fr.append(rot(q))
          fi.append(int(k))
          pts.append(_RY @ np.array([q[0], q[1], 0.006]))  # skid track on the ramp surface under the box
          pcols.append(col)
      env_frames.append([np.asarray(f, np.float64) for f in fr])
      env_ribbons.append([{"pts": np.asarray(pts), "iters": fi, "width": 0.01, "colors": pcols}])
      env_protos.append(self.strip_to_proto(self._proto_xml(e), from_string=True))
      frame_iters = fi  # identical schedule across every env
    assert env_protos[0].nq == env_frames[0][0].shape[0], (env_protos[0].nq, env_frames[0][0].shape[0])
    out = self.export_field(env_protos[0], None, offsets, out_dir, name="incline_sysid_traj", fps=fps,
                            frame_iters=frame_iters, opt_label="surface friction",
                            env_frames=env_frames, env_ribbons=env_ribbons, env_protos=env_protos)
    slid = float(np.sqrt([history[ks[-1]]["losses"][e] for e in range(ne)]).mean())
    print(f"[export] incline: {ne} DISTINCT envs (friction inits), iters {ks}, mean_slid {slid:.3f}m; "
          f"NF={len(env_frames[0])} -> {out}")

  # ---- render (y-lanes on the visual ramp; each box's down-slope trail loss-colored) ----

  def render(self, history, best, out):
    E = history[0]["qpos"].shape[0]
    vm = mujoco.MjModel.from_xml_string(slide_xml(viz_scene=True, n_lanes=E))
    vd = mujoco.MjData(vm)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(vm, cam)
    cam.lookat = [0.05, 0.0, 0.02]
    cam.distance = 4.6
    cam.azimuth = 68.0  # mu ordered HIGH->low so the low-friction box is in the foremost lane
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
    persisted = []  # (per-env-trail-xyz, colors) of completed shown iterations
    for k in show:
      h = history[k]
      cols = [viz.bourke_color_map(0.0, hi, float(v)) for v in h["losses"]]
      sub = (f"iter {h['it']:3d}    mu {np.array2string(h['mu'][::-1], precision=2)}    "
             f"mean slid {np.sqrt(h['losses']).mean():.3f}m")
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
    return viz.emit(vm, vd, cam, frames, out_path=out, label="ADJOINT (incline friction)", w=W, h=H, fps=FPS)


def main(argv):
  del argv  # config comes from the absl-parsed Args
  demo.run(InclineDemo, demo.parse_args(Args))


if __name__ == "__main__":
  demo.define_flags(Args)
  app.run(main)
