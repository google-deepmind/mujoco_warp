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
we write mp4. The shared demo.Example harness owns capture + BackwardContext reuse (bounce's first
capture -- it used to reallocate Datas every backward), the fd/analytic dispatch, the parallel-GD
loop, and the CLI; BounceDemo supplies the bounce-specific model + forward + loss + rollout + render.

  # analytic adjoint gradient (default) -> reports/assets/bounce_parallel.mp4
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/bounce.py
  # finite-difference baseline -> reports/assets/bounce_fd.mp4
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/bounce.py --grad=fd
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
import viz  # noqa: E402  (shared renderer: mp4 or live MuJoCo viewer via MJW_VIEWER)

# Newton example_diffsim_ball numbers (Z-up).
START = (0.0, -0.5, 1.0)
QVEL0 = (0.0, 5.0, -5.0, 0.0, 0.0, 0.0)
TARGET = (0.0, -2.0, 1.5)
T = 150  # steps @ dt=0.004 -> 0.6s
SOLREF = "-3000 -2"  # elastic-ish: direct (stiffness, damping), low damping -> bounce
# condim 6 + nonzero torsion/roll exercises the rotational contact rows; default = condim 3, slide-only.
CONDIM = "3"
FRICTION = "0.2"  # geom friction [slide, torsion, roll]
W, H, FPS = 1024, 768, 30
ASSETS = os.path.join(os.path.dirname(__file__), "reports", "assets")  # default output dir


@dataclass
class Args(demo.CommonArgs):
  """bounce config: CommonArgs (grad/device/capture/out/live/...) + bounce defaults + the render-layout
  flags. A single typed object; parsed from the CLI via absl.flags."""

  num_envs: int = field(default=8, metadata={"help": "parallel envs optimized at once"})
  steps: int = field(default=160, metadata={"help": "gradient-descent steps"})
  lr: float = field(default=0.04, metadata={"help": "gradient-descent learning rate"})
  spread: float = field(default=0.25, metadata={"help": "per-env init-velocity spread (visual fan)"})
  layout: str = field(default="offset", metadata={"help": "render layout", "choices": ["offset", "overlay"]})
  env_axis: str = field(default="x", metadata={"help": "lane axis for the offset layout", "choices": ["x", "y", "z"]})
  env_spacing: float = field(default=3.0, metadata={"help": "lane spacing (walls are 2 wide -> > 2)"})


@wp.kernel
def _bounce_loss_batched(qpos: wp.array2d[float], target: wp.vec3, loss: wp.array(dtype=float)):
  """loss = sum_w ||ball_pos_w - target||^2 over all worlds. Each world's qvel0 only affects its own
  qpos, so d(sum)/d(qvel0_w) = d(loss_w)/d(qvel0_w): one backward -> per-env grads. (@wp.kernel must
  stay module-level -- Warp JITs it by module path.)"""
  w = wp.tid()
  delta = wp.vec3(qpos[w, 0] - target[0], qpos[w, 1] - target[1], qpos[w, 2] - target[2])
  wp.atomic_add(loss, 0, wp.dot(delta, delta))


class BounceDemo(demo.Example):
  """Bounce: optimize each env's initial linear velocity (params (E, 3)) so the ball lands on the
  target after bouncing off the floor + wall. The taped chunk the harness captures/replays is the
  batched freejoint rollout + final-distance-to-target loss; the fd path central-differences it."""

  Args = Args

  # ---- harness hooks ----

  def build_model(self):
    self.mjm = mujoco.MjModel.from_xml_string(self.bounce_xml())
    self.mjd = mujoco.MjData(self.mjm)
    mujoco.mj_forward(self.mjm, self.mjd)
    self.qpos0 = self.mjd.qpos.copy()  # freejoint start pose (fixed across iters)
    self.target_v = wp.vec3(float(TARGET[0]), float(TARGET[1]), float(TARGET[2]))

  def init_params(self):
    ne, spread = self.args.num_envs, self.args.spread
    q = np.tile(np.array(QVEL0[:3], dtype=np.float64), (ne, 1))  # (E, 3) linear velocity
    if ne > 1 and spread != 0.0:
      s = np.linspace(-spread, spread, ne)
      q[:, 1] += s  # vy
      q[:, 2] += s[::-1]  # vz (opposite ramp so envs differ in both)
    return q

  def build_datas(self):
    ne = self.args.num_envs
    self.nT = T  # checkpointed BPTT length; only chunk+1 segment buffers are held (not T+1)
    self.datas = [mjw.put_data(self.mjm, self.mjd, nworld=ne) for _ in range(self.args.chunk + 1)]
    for d in self.datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    self.datas[0].qpos = wp.array(np.tile(self.qpos0, (ne, 1)).astype(np.float32), dtype=wp.float32, requires_grad=True)
    self.datas[0].qvel = wp.array(self._full_qvel(self.params), dtype=wp.float32, requires_grad=True)
    self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

  def chunk_step(self, i, t):
    del t  # terminal loss -> time-independent chunk; the captured chunk replays unchanged over the rollout
    mjw.step(self.m, self.datas[i], self.datas[i + 1])

  def terminal_loss(self):
    # datas[0] holds the final state (state_T) after the checkpointed forward pass
    wp.launch(_bounce_loss_batched, dim=self.args.num_envs, inputs=[self.datas[0].qpos, self.target_v], outputs=[self.loss])

  def set_params(self):
    self.datas[0].qvel.assign(self._full_qvel(self.params))  # linear vel -> datas[0].qvel in place

  def read_grad(self):
    return self.datas[0].qvel.grad.numpy()[:, :3].astype(np.float64).copy()

  def rollout_env(self, p):
    """mj_step rollout at initial linear velocity `p` (3,). Returns (xyz[T+1,3], qpos[T+1,7], loss)."""
    m = mujoco.MjModel.from_xml_string(self.bounce_xml())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
    d.qvel[:6] = np.concatenate([p, np.zeros(3)])  # linear p + zero angular
    xyz = [d.xpos[bid].copy()]
    qpos = [d.qpos.copy()]
    for _ in range(T):
      mujoco.mj_step(m, d)
      xyz.append(d.xpos[bid].copy())
      qpos.append(d.qpos.copy())
    xyz, qpos = np.array(xyz), np.array(qpos)
    return xyz, qpos, float(np.sum((xyz[-1] - np.array(TARGET)) ** 2))

  def fd_grad(self, p, eps=1e-3):
    """Central-difference d(loss)/d(p[:3]) for one env (the robust baseline)."""
    g = np.zeros(3)
    for i in range(3):
      vp = p.copy(); vp[i] += eps
      vm = p.copy(); vm[i] -= eps
      g[i] = (self.rollout_env(vp)[2] - self.rollout_env(vm)[2]) / (2 * eps)
    return g

  def record(self, it, losses, trajs, qposs):
    return {"it": it, "qvel": self.params.copy(), "losses": losses.copy(), "xyz": trajs, "qpos": qposs}

  def progress(self, it, losses, g):
    return (
      f"  [{it:4d}] mean_loss={losses.mean():.4f} best={losses.min():.4f} "
      f"mean|g|={np.linalg.norm(g, axis=1).mean():.3f}"
    )

  def summary(self, history, best):
    ml = [h["losses"].mean() for h in history]
    tag = "analytic (adjoint)" if self.args.grad == "analytic" else "finite diff"
    return (
      f"[bounce optim / {tag}] {self.args.num_envs} envs: "
      f"mean_loss {ml[0]:.3f} -> best {ml[best]:.3f} (iter {best})"
    )

  def default_out(self):
    name = "bounce_parallel" if self.args.grad == "analytic" else "bounce_fd"
    return os.path.join(ASSETS, name + ".mp4")

  # ---- bounce-specific helpers ----

  def _full_qvel(self, params):
    """Expand the optimized (E, 3) linear velocity to the full (E, nv) freejoint qvel (angular = 0)."""
    ne = self.args.num_envs
    return np.concatenate([params, np.zeros((ne, self.mjm.nv - 3))], axis=1).astype(np.float32)

  def taped_bounce_grad_batched(self, m, mjm, mjd, qpos0, qvel0, target, n_steps):
    """Fresh-alloc reference: per-env d(loss)/d(qvel0[:, :3]) via a batched (nworld=B) wp.Tape over the
    n_steps rollout, with NO capture and NO BackwardContext reuse (reallocates a Data per step). The
    oracle the captured/reused analytic path must match (bitwise on CPU; atomic-add noise on GPU).

    qpos0: (B, nq), qvel0: (B, nv). Returns qvel_grad: (B, 3).
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

  # ---- physics + viz scene builders ----

  def bounce_xml(self, physics_only=True):
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

  def _env_offsets(self, num_envs, layout, axis, spacing):
    """Per-env world offset (N, 3). 'offset' -> a lane per env along `axis`, centered on 0;
    'overlay' -> all zeros (every env in the shared scene)."""
    off = np.zeros((num_envs, 3))
    if layout == "offset" and num_envs > 1:
      ax = {"x": 0, "y": 1, "z": 2}[axis]
      off[:, ax] = (np.arange(num_envs) - (num_envs - 1) / 2.0) * spacing
    return off

  def _viz_xml_offset(self, offsets):
    """bounce scene with the wall + target REPLICATED at each env's offset (one shared ground plane;
    the single ball body is parked off-camera and every ball is drawn as an added geom). Mirrors
    bounce_xml(physics_only=False)'s look."""
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

  def render(self, history, best, out):
    a = self.args
    label = "ADJOINT (bounce)" if a.grad == "analytic" else "FINITE DIFF (bounce)"
    live = True if a.live else None  # None -> honor MJW_VIEWER
    self.render_video(history, best, out, layout=a.layout, axis=a.env_axis, spacing=a.env_spacing, label=label, live=live)

  def render_video(self, history, best, out_path, layout="offset", axis="x", spacing=3.0,
                   sample_every=4, label="ADJOINT (bounce)", live=None):
    """Animate all envs bouncing together for selected iterations; persist each shown iteration's
    per-env trajectories faintly so the parallel fan visibly converges. mp4, or the live MuJoCo viewer
    when `live=True` / `MJW_VIEWER=1` (via viz.emit)."""
    num_envs = history[0]["xyz"].shape[0]
    # ONE shared colormap keyed on loss (Bourke: red=high loss -> blue=converged), same for every
    # env, so the whole fan shifts color as it converges (the lanes already separate envs spatially).
    lo, hi = 0.0, float(max(h["losses"].max() for h in history))

    def _loss_colors(losses):
      return [viz.bourke_color_map(lo, hi, float(v)) for v in losses]

    offsets = self._env_offsets(num_envs, layout, axis, spacing)  # (N, 3); zeros for overlay
    if layout == "offset":
      vm = mujoco.MjModel.from_xml_string(self._viz_xml_offset(offsets))
    else:
      vm = mujoco.MjModel.from_xml_string(self.bounce_xml(physics_only=False))
    vd = mujoco.MjData(vm)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(vm, cam)
    span = float(offsets.max(0).sum() - offsets.min(0).sum())  # extent across the lanes
    if layout == "offset":
      # near-perpendicular to the row so every lane is framed at a similar size (a shallower
      # angle foreshortens the far lanes); distance scales to fit the row width.
      cam.lookat = [0.0, 0.0, 0.7]  # lanes are centered on 0
      cam.distance = 7.0 + 0.5 * span
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


def main(argv):
  del argv  # unused; config comes from the absl-parsed Args
  demo.run(BounceDemo, demo.parse_args(Args))


if __name__ == "__main__":
  demo.define_flags(Args)
  app.run(main)
