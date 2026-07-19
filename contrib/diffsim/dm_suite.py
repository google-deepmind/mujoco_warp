"""dm_control suite tasks under the differentiable mujoco_warp adjoint -- one switchable, PARALLEL harness.

A TASK REGISTRY (`TASKS`) maps a name to a `Task`: INLINED, flattened MJCF parts (dm_control's
`<include>`s resolved, integrator -> Euler for the analytic adjoint, cradle-style grid visuals; no runtime
dm_control dependency) + an objective + horizon/optimizer/camera config. One shared engine drives every
task, and it runs MANY ENVS IN PARALLEL (batched nworld>1, à la bounce.py / cradle.py): each env optimizes
its OWN control sequence from a per-env init spread, and ONE batched `tape.backward` yields INDEPENDENT
per-env gradients. Switch with `--task`, scale with `--num_envs`.

The per-step controls `ctrls` are a LIST of wp.array leaves (one per action, bound to each step INSIDE the
tape), so tape.backward lands grads on ctrls[t].grad and warp.optim.Adam (SHAC betas 0.7, 0.95) steps them
in place, projected to [-1,1] (demo.Example Style-D, list-param). Preset per task via `--task`; the
demo.Example harness adds CUDA-graph capture + BackwardContext reuse over the taped rollout.

  uv run --active --with imageio --with imageio-ffmpeg --with pillow \
    python contrib/diffsim/dm_suite.py --task=cartpole --num_envs=6

Objectives:
  cartpole  : swing up & balance the pole (maximize cos(pole), cart centered)
  reacher   : each parallel env drives its 2-link fingertip to its OWN target
  cheetah   : run forward (locomotion: maximize forward velocity - action^2, dflex/SHAC HalfCheetah reward)
"""

import os
import sys
import typing
from dataclasses import dataclass
from dataclasses import field

import mujoco
import numpy as np
import warp as wp
import warp.optim
from absl import app

import mujoco_warp as mjw

sys.path.insert(0, os.path.dirname(__file__))
import demo  # noqa: E402  shared config + gradients + capture/reuse + main
import viz  # noqa: E402

W, H, FPS = 1024, 768, 30

# Shared cradle-style visuals (grid floor + skybox + haze), replacing dm_control's include files.
_ASSETS = """
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
             rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="40 40" reflectance="0.2"/>
    <material name="self" rgba="0.85 0.6 0.35 1"/>
    <material name="decoration" rgba="0.35 0.5 0.7 1"/>
  </asset>
"""

# ---- per-task MJCF parts: (header, floor, robot [replicable], actuator) ----
CHEETAH = dict(
  compiler='<compiler settotalmass="14"/>',
  header='<option timestep="0.005" integrator="Euler"/>'
         '<default><default class="cheetah"><joint limited="true" damping=".01" armature=".1" stiffness="8" type="hinge" axis="0 1 0"/>'
         '<geom contype="1" conaffinity="1" condim="3" friction=".4 .1 .1" material="self" type="capsule"/></default>'
         '<default class="free"><joint limited="false" damping="0" armature="0" stiffness="0"/></default>'
         '<motor ctrllimited="true" ctrlrange="-1 1"/></default>',
  floor='<geom name="ground" type="plane" conaffinity="1" pos="0 0 0" size="100 16 .5" material="grid"/>',
  robot="""
    <body name="torso" pos="0 0 .7" childclass="cheetah">
      <joint name="rootx" type="slide" axis="1 0 0" class="free"/>
      <joint name="rootz" type="slide" axis="0 0 1" class="free"/>
      <joint name="rooty" type="hinge" axis="0 1 0" class="free"/>
      <geom name="torso" type="capsule" fromto="-.5 0 0 .5 0 0" size="0.046"/>
      <geom name="head" type="capsule" pos=".6 0 .1" euler="0 50 0" size="0.046 .15"/>
      <body name="bthigh" pos="-.5 0 0"><joint name="bthigh" range="-30 60" stiffness="240" damping="6"/>
        <geom name="bthigh" pos=".1 0 -.13" euler="0 -218 0" size="0.046 .145"/>
        <body name="bshin" pos=".16 0 -.25"><joint name="bshin" range="-50 50" stiffness="180" damping="4.5"/>
          <geom name="bshin" pos="-.14 0 -.07" euler="0 -116 0" size="0.046 .15"/>
          <body name="bfoot" pos="-.28 0 -.14"><joint name="bfoot" range="-230 50" stiffness="120" damping="3"/>
            <geom name="bfoot" pos=".03 0 -.097" euler="0 -15 0" size="0.046 .094"/></body></body></body>
      <body name="fthigh" pos=".5 0 0"><joint name="fthigh" range="-57 .40" stiffness="180" damping="4.5"/>
        <geom name="fthigh" pos="-.07 0 -.12" euler="0 30 0" size="0.046 .133"/>
        <body name="fshin" pos="-.14 0 -.24"><joint name="fshin" range="-70 50" stiffness="120" damping="3"/>
          <geom name="fshin" pos=".065 0 -.09" euler="0 -34 0" size="0.046 .106"/>
          <body name="ffoot" pos=".13 0 -.18"><joint name="ffoot" range="-28 28" stiffness="60" damping="1.5"/>
            <geom name="ffoot" pos=".045 0 -.07" euler="0 -34 0" size="0.046 .07"/></body></body></body>
    </body>""",
  actuator='<actuator><motor name="bthigh" joint="bthigh" gear="120"/><motor name="bshin" joint="bshin" gear="90"/>'
           '<motor name="bfoot" joint="bfoot" gear="60"/><motor name="fthigh" joint="fthigh" gear="90"/>'
           '<motor name="fshin" joint="fshin" gear="60"/><motor name="ffoot" joint="ffoot" gear="30"/></actuator>',
)

CARTPOLE = dict(
  header='<option timestep="0.01" integrator="Euler"><flag contact="disable"/></option>'
         '<default><default class="pole"><joint type="hinge" axis="0 1 0" damping="2e-6"/>'
         '<geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" material="self" mass=".1"/></default></default>',
  floor='<geom name="floor" pos="0 0 -.05" size="4 4 .2" type="plane" material="grid"/>',
  robot="""
    <geom name="rail1" type="capsule" pos="0  .07 1" zaxis="1 0 0" size="0.02 2" material="decoration"/>
    <geom name="rail2" type="capsule" pos="0 -.07 1" zaxis="1 0 0" size="0.02 2" material="decoration"/>
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" limited="true" axis="1 0 0" range="-1.8 1.8" solreflimit=".08 1" damping="5e-4"/>
      <geom name="cart" type="box" size="0.2 0.15 0.1" material="self" mass="1"/>
      <body name="pole_1" childclass="pole"><joint name="hinge_1"/><geom name="pole_1"/></body>
    </body>""",
  actuator='<actuator><motor name="slide" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1"/></actuator>',
)

# dm_control 2-link planar reacher (the EASIEST gradient path: smooth, contact-free). Both links 0.12 m,
# reach = 0.24; wrist limit dropped so the path stays purely smooth. Each env gets its OWN target.
REACH_L1, REACH_L2 = 0.12, 0.12
REACHER = dict(
  header='<option timestep="0.02" integrator="Euler"><flag contact="disable"/></option>'
         '<default><joint type="hinge" axis="0 0 1" damping="0.05"/>'
         '<motor gear="0.05" ctrlrange="-1 1" ctrllimited="true"/></default>',
  floor='<geom name="ground" type="plane" pos="0 0 0" size="2 2 .1" material="grid"/>',
  robot="""
    <geom name="root" type="cylinder" fromto="0 0 0 0 0 0.02" size=".011" material="decoration"/>
    <body name="arm" pos="0 0 .01">
      <geom name="arm" type="capsule" fromto="0 0 0 0.12 0 0" size=".01" material="self"/>
      <joint name="shoulder"/>
      <body name="hand" pos=".12 0 0">
        <geom name="hand" type="capsule" fromto="0 0 0 0.1 0 0" size=".01" material="self"/>
        <joint name="wrist"/>
        <body name="finger" pos=".12 0 0"><geom name="finger" type="sphere" size=".013" material="self"/></body>
      </body>
    </body>""",
  actuator='<actuator><motor name="shoulder" joint="shoulder"/><motor name="wrist" joint="wrist"/></actuator>',
)


@dataclass
class Task:
  name: str
  parts: dict
  T: int
  steps: int
  lr: float
  objective: str
  idx: dict
  w: dict
  init_qpos: object = None      # explicit init (swingup); None -> settle from default
  settle: int = 150
  spread: dict = None           # per-env init fan: {"idx": qpos_idx, "lo": .., "hi": ..}
  target: dict = None           # reacher per-env GOAL fan: {"r":.., "ang_lo":.., "ang_hi":..}
  action_repeat: int = 1        # hold each action for k physics substeps: k*T physical horizon, still T actions
  cam: dict = field(default_factory=dict)


TASKS = {
  "cheetah": Task("cheetah", CHEETAH, T=240, steps=200, lr=0.04, objective="locomotion",
                  idx={"fwd": 0, "z": 1, "pitch": 2}, w={"fwd": 1.0, "up": 0.0, "pitch": 0.2, "ctrl": 0.1},
                  spread={"idx": 3, "lo": -0.25, "hi": 0.25}, cam={"z": 0.6, "track": True}),
  "cartpole": Task("cartpole", CARTPOLE, T=200, steps=200, lr=0.02, objective="cartpole",
                   idx={"cart": 0, "pole": 1},
                   w={"ang": 1.0, "pvel": 0.1, "cpos": 0.2, "cvel": 0.2, "ctrl": 0.0},
                   init_qpos=np.array([0.0, np.pi - 0.25]), settle=0,
                   spread={"idx": 1, "lo": np.pi - 0.65, "hi": np.pi - 0.25}, cam={"z": 1.0, "dist": 4.0}),
  "reacher": Task("reacher", REACHER, T=120, steps=120, lr=0.05, objective="reacher",
                  idx={"sh": 0, "wr": 1}, w={"dist": 1.0, "vel": 0.01, "ctrl": 1e-4},
                  init_qpos=np.array([-0.75, 1.5]), settle=0,
                  target={"r": 0.18, "ang_lo": -1.0, "ang_hi": 1.0},
                  cam={"grid": True, "dim": 0.8, "z": 0.02, "dist": 0.5, "az": 90.0, "el": -60.0,
                       "tw": 0.004, "pw": 0.0022}),
}


# ---------------- batched loss kernels (dim = nworld); @wp.kernel stays module-level ----------------
@wp.kernel
def _loc_reward(qpos: wp.array2d[float], qvel: wp.array2d[float], fwd: int, z: int, pitch: int,
                z_low: float, w_fwd: float, w_up: float, w_pitch: float, loss: wp.array(dtype=float)):
  w = wp.tid()
  wp.atomic_add(loss, 0, -w_fwd * qvel[w, fwd] - w_up * (qpos[w, z] - z_low) + w_pitch * qpos[w, pitch] * qpos[w, pitch])


@wp.kernel
def _cartpole_reward(qpos: wp.array2d[float], qvel: wp.array2d[float], cart: int, pole: int,
                     w_ang: float, w_pvel: float, w_cpos: float, w_cvel: float, ws: float,
                     loss: wp.array(dtype=float)):
  w = wp.tid()
  th = wp.atan2(wp.sin(qpos[w, pole]), wp.cos(qpos[w, pole]))  # normalize_angle(pole): 0 = upright
  wp.atomic_add(loss, 0, w_ang * th * th + w_pvel * qvel[w, pole] * qvel[w, pole]
                + ws * (w_cpos * qpos[w, cart] * qpos[w, cart] + w_cvel * qvel[w, cart] * qvel[w, cart]))


@wp.kernel
def _reacher_reward(qpos: wp.array2d[float], qvel: wp.array2d[float], sh: int, wr: int, l1: float, l2: float,
                    tx: wp.array(dtype=float), ty: wp.array(dtype=float), w_dist: float, w_vel: float,
                    ws: float, loss: wp.array(dtype=float)):
  w = wp.tid()
  th1 = qpos[w, sh]
  th2 = qpos[w, wr]
  fx = l1 * wp.cos(th1) + l2 * wp.cos(th1 + th2)
  fy = l1 * wp.sin(th1) + l2 * wp.sin(th1 + th2)
  dx = fx - tx[w]
  dy = fy - ty[w]
  vel = qvel[w, sh] * qvel[w, sh] + qvel[w, wr] * qvel[w, wr]
  wp.atomic_add(loss, 0, w_dist * (dx * dx + dy * dy) + ws * w_vel * vel)


@wp.kernel
def _ctrl_cost(ctrl: wp.array2d[float], w_: float, loss: wp.array(dtype=float)):
  i, j = wp.tid()
  wp.atomic_add(loss, 0, w_ * ctrl[i, j] * ctrl[i, j])


@wp.kernel
def _clamp01(a: wp.array(dtype=float)):  # in-place clamp the (1D) control params to [-1, 1] after each Adam step
  i = wp.tid()
  a[i] = wp.clamp(a[i], -1.0, 1.0)


@dataclass
class Args(demo.CommonArgs):
  """dm_suite config: --task selects a preset (TASKS); steps/lr/T/repeat/spread/spacing None -> preset."""

  task: str = field(default="cartpole", metadata={"help": "dm task preset", "choices": list(TASKS)})
  num_envs: int = field(default=4, metadata={"help": "parallel envs optimized at once"})
  steps: typing.Optional[int] = field(default=None, metadata={"help": "Adam steps (default: preset)"})
  lr: typing.Optional[float] = field(default=None, metadata={"help": "Adam lr (default: preset)"})
  actions: typing.Optional[int] = field(default=None, metadata={"help": "number of ACTIONS (default: preset)"})
  repeat: typing.Optional[int] = field(default=None, metadata={"help": "action repeat (default: preset)"})
  spread: typing.Optional[float] = field(default=None, metadata={"help": "scale per-env init fan (default: preset)"})
  spacing: typing.Optional[float] = field(default=None, metadata={"help": "render env spacing (default: per-task)"})


class DmSuiteDemo(demo.Example):
  """dm_control suite under the adjoint. Style-D with a LIST of per-step control leaves `self.ctrls`
  bound to each step inside the tape; warp.optim.Adam(betas=0.7,0.95) steps them in place, clamped to
  [-1,1]. `--task` selects a preset from TASKS; best iter = argmax mean per-env score."""

  Args = Args
  capturable = False  # per-step control leaves + host global-step (ws ramp) can't bake into one chunk graph -> eager

  # ---- harness hooks ----

  def build_model(self):
    a = self.args
    self.task = TASKS[a.task]
    task = self.task
    self.T_actions = a.actions if a.actions is not None else task.T
    self.k = a.repeat if a.repeat is not None else task.action_repeat
    self.steps = a.steps if a.steps is not None else task.steps
    self.lr = a.lr if a.lr is not None else task.lr
    self.spacing = a.spacing if a.spacing is not None else {"cartpole": 1.2, "reacher": 0.42, "cheetah": 2.0}.get(task.name, 2.0)
    self.spread_cfg = task.spread
    if a.spread is not None and task.spread:  # scale the init-fan width, ANCHORED at `hi`
      width = (task.spread["hi"] - task.spread["lo"]) * a.spread
      self.spread_cfg = {**task.spread, "lo": task.spread["hi"] - width}
    self.mjm = mujoco.MjModel.from_xml_string(self.single_xml(task.parts))
    self.mjd = mujoco.MjData(self.mjm)
    mujoco.mj_forward(self.mjm, self.mjd)
    self.nu = self.mjm.nu
    self.q0 = self._make_q0(task)

  def _make_q0(self, task):
    N = self.args.num_envs
    if task.init_qpos is not None:
      base = task.init_qpos.copy()
    else:  # settle from the default pose (locomotion) via mjw.step to a resting base pose
      mujoco.mj_resetData(self.mjm, self.mjd)
      mujoco.mj_forward(self.mjm, self.mjd)
      m = mjw.put_model(self.mjm)
      ds = [mjw.put_data(self.mjm, self.mjd) for _ in range(task.settle + 1)]
      for t in range(task.settle):
        mjw.step(m, ds[t], ds[t + 1])
      base = ds[task.settle].qpos.numpy()[0].copy()
    q0 = np.tile(base, (N, 1)).astype(np.float32)
    if self.spread_cfg and N > 1:
      q0[:, self.spread_cfg["idx"]] = np.linspace(self.spread_cfg["lo"], self.spread_cfg["hi"], N)
    return q0

  def init_params(self):
    return np.zeros((self.args.num_envs, self.T_actions * self.nu), np.float32)  # host mirror (unused; ctrls is the leaf)

  def build_datas(self):
    N, task = self.args.num_envs, self.task
    self.T_phys = self.T_actions * self.k
    self.nT = self.T_phys  # checkpointed BPTT length (chunk+1 segment buffers, not T_phys+1)
    assert self.args.chunk % self.k == 0, f"chunk {self.args.chunk} must be a multiple of action_repeat {self.k}"
    self.z_low = float(np.mean(self.q0[:, task.idx["z"]])) - 0.35 if task.objective == "locomotion" else 0.0
    self.tgt = None
    if task.objective == "reacher":
      tg = self.env_targets(task, N)  # (N,2) constant per-env goals
      self.tgt = (wp.array(tg[:, 0], dtype=float), wp.array(tg[:, 1], dtype=float))
    self.datas = [mjw.put_data(self.mjm, self.mjd, nworld=N) for _ in range(self.args.chunk + 1)]
    for d in self.datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
      d.ctrl.requires_grad = True  # bind_ctrl scatters the schedule leaf into this stable buffer
    self.datas[0].qpos = wp.array(self.q0, dtype=float, requires_grad=True)
    self.datas[0].qvel = wp.array(np.zeros((N, self.mjm.nv), np.float32), dtype=float, requires_grad=True)
    # PERSISTENT per-step control leaves (1D, size N*nu as warp.optim.Adam requires); bound per step in chunk_step
    self.ctrls = [wp.array(np.zeros(N * self.nu, np.float32), dtype=float, requires_grad=True) for _ in range(self.T_actions)]
    self.loss = wp.zeros(1, dtype=float, requires_grad=True)
    self._viz = [mjw.put_data(self.mjm, self.mjd, nworld=N) for _ in range(2)]  # ping-pong for sim_qpos

  def set_params(self):
    pass  # the schedule leaves (self.ctrls) are bound per-step in chunk_step; datas[0] is the fixed initial state

  def chunk_step(self, i, t):
    N, k = self.args.num_envs, self.k
    leaf = self.ctrls[t // k]  # per-step control leaf (action repeat k) at global step t
    self.bind_ctrl(i, leaf)  # scatter into datas[i].ctrl (kernel, not a rebind -> correct multi-step ctrl grad)
    mjw.step(self.m, self.datas[i], self.datas[i + 1])
    self._launch_loss(self.datas[i + 1], leaf.reshape((N, self.nu)), t)  # per-step loss (reads global t for ws)

  def sim_qpos(self):
    """Full video/score trajectory (N, T_phys+1, nq). Checkpointing keeps only chunk+1 segment buffers, so
    reconstruct it with a fresh forward-only rollout at the current ctrls (eager; cheap at these horizons)."""
    N, k = self.args.num_envs, self.k
    a, b = self._viz
    a.qpos = wp.array(self.q0, dtype=float)
    a.qvel = wp.array(np.zeros((N, self.mjm.nv), np.float32), dtype=float)
    qs = [self.q0.copy()]
    for t in range(self.T_phys):
      a.ctrl = self.ctrls[t // k].reshape((N, self.nu))
      mjw.step(self.m, a, b)
      qs.append(b.qpos.numpy().copy())
      a, b = b, a
    return np.transpose(np.array(qs), (1, 0, 2))

  def optimize(self):
    """List-param Style-D Adam: per iter, capture-replay the taped rollout (grads land on ctrls[t].grad),
    read the sim qtraj for the score/video, opt.step + clamp01. Best iter = argmax mean per-env score."""
    task, N = self.task, self.args.num_envs
    opt = warp.optim.Adam(self.ctrls, lr=self.lr, betas=(0.7, 0.95))
    print(f"=== dm_suite: {task.name} (objective={task.objective}, actions={self.T_actions}x repeat={self.k}"
          f" -> {self.T_phys} phys steps, steps={self.steps}, envs={N}) ===")
    history = []
    for it in range(self.steps):
      self.backward()  # replay/eager: forward (per-step ctrls) + backward -> ctrls[t].grad
      qtraj = self.sim_qpos()
      score = self.per_env_score(task, qtraj)
      history.append({"it": it, "score": score.copy(), "qtraj": qtraj})
      if it % 20 == 0 or it == self.steps - 1:
        print(self._status(it, score, qtraj))
      opt.step([c.grad for c in self.ctrls])
      for c in self.ctrls:
        wp.launch(_clamp01, dim=c.shape, inputs=[c])  # project back to the ctrl range [-1, 1]
    mean_score = np.array([h["score"].mean() for h in history])
    best = int(mean_score.argmax())
    print(f"[{task.name}] {N} env(s): mean_score {mean_score[0]:+.3f} -> best {mean_score[best]:+.3f} (iter {best})")
    return history, best

  def _status(self, it, score, qtraj):
    task, N = self.task, self.args.num_envs
    gnorm = float(np.linalg.norm(np.array([c.grad.numpy().reshape(N, self.nu) for c in self.ctrls]), axis=(0, 2)).mean())
    extra = ""
    if task.objective == "cartpole":
      extra = f"  |cart|_end={np.abs(qtraj[:, -1, task.idx['cart']]).mean():.2f}"
    elif task.objective == "locomotion":
      z, p, fi = qtraj[:, :, task.idx["z"]], qtraj[:, :, task.idx["pitch"]], task.idx["fwd"]
      fwd = float(np.mean(qtraj[:, -1, fi] - qtraj[:, 0, fi]))
      extra = f"  fwd={fwd:+.2f} peakZ={z.max():+.2f} minZ={z.min():+.2f} |pitch|mx={np.abs(p).max():.2f}"
    return (f"  [{it:3d}] mean_score={score.mean():+.3f}  best={score.max():+.3f}  "
            f"worst={score.min():+.3f}{extra}  mean|g|={gnorm:.2f}")

  def default_out(self):
    return os.environ.get("MJW_RENDER_PATH") or f"/tmp/dm_{self.task.name}.mp4"

  # ---- objective helpers ----

  def _launch_loss(self, d1, ctrl, t):
    task = self.task
    o, ix, w = task.objective, task.idx, task.w
    N, ws = self.args.num_envs, float((t + 1) / self.T_phys)
    if o == "locomotion":
      wp.launch(_loc_reward, dim=N, inputs=[d1.qpos, d1.qvel, ix["fwd"], ix["z"], ix["pitch"], self.z_low,
                                            w["fwd"], w["up"], w["pitch"]], outputs=[self.loss])
    elif o == "cartpole":
      wp.launch(_cartpole_reward, dim=N, inputs=[d1.qpos, d1.qvel, ix["cart"], ix["pole"],
                                                 w["ang"], w["pvel"], w["cpos"], w["cvel"], ws], outputs=[self.loss])
    elif o == "reacher":
      tx, ty = self.tgt
      wp.launch(_reacher_reward, dim=N, inputs=[d1.qpos, d1.qvel, ix["sh"], ix["wr"], REACH_L1, REACH_L2,
                                                tx, ty, w["dist"], w.get("vel", 0.0), ws], outputs=[self.loss])
    if w.get("ctrl", 0.0) > 0.0:
      wp.launch(_ctrl_cost, dim=(N, ctrl.shape[1]), inputs=[ctrl, w["ctrl"]], outputs=[self.loss])

  def per_env_score(self, task, qtraj):
    """Higher = better, per env. qtraj: (N, T+1, nq)."""
    o, ix = task.objective, task.idx
    if o == "locomotion":
      return qtraj[:, -1, ix["fwd"]] - qtraj[:, 0, ix["fwd"]]                      # forward distance
    if o == "cartpole":
      return np.mean(np.cos(qtraj[:, -20:, ix["pole"]]), axis=1)                   # end upright (cos)
    tg = self.env_targets(task, qtraj.shape[0])                                    # (N,2)
    th1, th2 = qtraj[:, -20:, ix["sh"]], qtraj[:, -20:, ix["wr"]]
    fx = REACH_L1 * np.cos(th1) + REACH_L2 * np.cos(th1 + th2)
    fy = REACH_L1 * np.sin(th1) + REACH_L2 * np.sin(th1 + th2)
    d = np.hypot(fx - tg[:, 0:1], fy - tg[:, 1:2])                                 # (N,20) end fingertip->goal
    return -d.mean(axis=1)

  def env_targets(self, task, N):
    """Per-env reacher goal positions (N, 2) on an arc of radius r within reach; N=1 -> the arc midpoint."""
    t = task.target
    angs = np.array([0.5 * (t["ang_lo"] + t["ang_hi"])]) if N == 1 else np.linspace(t["ang_lo"], t["ang_hi"], N)
    return np.stack([t["r"] * np.cos(angs), t["r"] * np.sin(angs)], axis=1).astype(np.float32)

  # ---- scene builders + render ----

  def scene_visual(self, dim=1.0):
    hd, ha, hs = 0.6 * dim, 0.4 * dim, 0.2 * dim
    return (f'<visual><headlight diffuse="{hd} {hd} {hd}" ambient="{ha} {ha} {ha}" specular="{hs} {hs} {hs}"/>'
            '<rgba haze="0.15 0.25 0.35 1"/><global offwidth="1280" offheight="960"/><map znear="0.02"/>'
            f'</visual>{_ASSETS}')

  def scene_light(self, dim=1.0):
    d, s = 0.8 * dim, 0.3 * dim
    return f'<light pos="0 -1 3" dir="0 0.3 -1" diffuse="{d} {d} {d}" specular="{s} {s} {s}"/>'

  def single_xml(self, parts, dim=1.0):
    comp = parts.get("compiler", "")
    return (f'<mujoco>{comp}{parts["header"]}{self.scene_visual(dim)}<worldbody>{self.scene_light(dim)}'
            f'{parts["floor"]}{parts["robot"]}</worldbody>{parts["actuator"]}</mujoco>')

  def grid_dims(self, n):
    """Near-square (nrow, ncol) with nrow*ncol >= n, for a 2D env lattice (ncol columns along x)."""
    ncol = max(1, int(round(n ** 0.5)))
    return -(-n // ncol), ncol

  def multi_xml(self, parts, n, spacing, grid=False, dim=1.0):
    comp = parts.get("compiler", "")
    if n <= 1:
      rep = parts["robot"]
    elif grid:
      nrow, ncol = self.grid_dims(n)
      inner = f'<replicate count="{ncol}" offset="{spacing} 0 0">{parts["robot"]}</replicate>'
      rep = f'<replicate count="{nrow}" offset="0 {spacing} 0">{inner}</replicate>'
    else:
      rep = f'<replicate count="{n}" offset="0 {spacing} 0">{parts["robot"]}</replicate>'
    return (f'<mujoco>{comp}{parts["header"]}{self.scene_visual(dim)}<worldbody>{self.scene_light(dim)}'
            f'{parts["floor"]}{rep}</worldbody></mujoco>')

  def trace(self, task, qtraj_e):
    """Tracked point path (T+1,3) for one env. qtraj_e: (T+1,nq)."""
    o, ix = task.objective, task.idx
    if o == "locomotion":
      return np.stack([qtraj_e[:, ix["fwd"]], np.zeros(len(qtraj_e)), 0.7 + qtraj_e[:, ix["z"]]], 1)
    if o == "cartpole":
      a = qtraj_e[:, ix["pole"]]
      return np.stack([qtraj_e[:, ix["cart"]] + np.sin(a), np.zeros(len(a)), 1.0 + np.cos(a)], 1)
    th1, th2 = qtraj_e[:, ix["sh"]], qtraj_e[:, ix["wr"]]
    fx = REACH_L1 * np.cos(th1) + REACH_L2 * np.cos(th1 + th2)
    fy = REACH_L1 * np.sin(th1) + REACH_L2 * np.sin(th1 + th2)
    return np.stack([fx, fy, np.full(len(th1), 0.01)], 1)

  def render(self, history, best, out, sample_every=4):
    task, N, spacing = self.task, self.args.num_envs, self.spacing
    c = task.cam
    nrow, ncol = self.grid_dims(N)
    use_grid = bool(c.get("grid")) and N > 1 and nrow * ncol == N
    vm = mujoco.MjModel.from_xml_string(self.multi_xml(task.parts, N, spacing, grid=use_grid, dim=c.get("dim", 1.0)))
    vd = mujoco.MjData(vm)
    off = np.zeros((N, 3))
    if use_grid:
      off[:, 0] = (np.arange(N) % ncol) * spacing
      off[:, 1] = (np.arange(N) // ncol) * spacing
    else:
      off[:, 1] = np.arange(N) * spacing
    targets = self.env_targets(task, N) if task.objective == "reacher" else None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(vm, cam)
    ymid = float(off[:, 1].mean())
    if c.get("track"):
      allx = np.concatenate([history[best]["qtraj"][:, :, task.idx["fwd"]].ravel()])
      tx, span = float(allx.mean()), float(max(allx.max() - allx.min(), 1.5))
      cam.lookat = [tx, ymid, c.get("z", 1.0)]
      cam.distance = max(5.0, 1.2 * max(span, (N - 1) * spacing) + 2.0)
    elif use_grid:
      if targets is not None:
        pts = np.concatenate([off[:, :2], off[:, :2] + targets], axis=0)
        margin = 0.15
      else:
        bq = history[best]["qtraj"]
        pts = np.concatenate([(self.trace(task, bq[e]) + off[e])[:, :2] for e in range(N)], axis=0)
        margin = 0.3
      cx, cy = 0.5 * (pts[:, 0].min() + pts[:, 0].max()), 0.5 * (pts[:, 1].min() + pts[:, 1].max())
      ext = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) + margin
      cam.lookat = [cx, cy, c.get("z", 1.0)]
      cam.distance = c.get("dist", 1.0) + ext
    else:
      cam.lookat = [0.0, ymid, c.get("z", 1.0)]
      cam.distance = max(c.get("dist", 4.0), 0.7 * (N - 1) * spacing + c.get("dist", 4.0))
    cam.azimuth = c.get("az", 75.0 if N > 1 else 90.0)
    cam.elevation = c.get("el", -18.0 if N > 1 else -8.0)

    scores = np.concatenate([h["score"] for h in history])
    lo, hi = float(scores.min()), float(scores.max())
    tw, pw = c.get("tw", 0.014), c.get("pw", 0.006)

    def colors(sc):
      return [viz.bourke_color_map(0.0, hi - lo + 1e-9, hi - float(v)) for v in sc]

    Tp = history[0]["qtraj"].shape[1] - 1
    se = max(sample_every, round(Tp / 90))
    best_mean = max(h["score"].mean() for h in history)
    frames, persisted = [], []
    for k in viz.default_show(len(history), best):
      hk = history[k]
      cols = colors(hk["score"])
      traces = [self.trace(task, hk["qtraj"][e]) + off[e] for e in range(N)]
      sub = f"iter {hk['it']:3d}    mean {hk['score'].mean():+.3f}    best {best_mean:+.3f}    envs {N}"
      hold = 20 if k == best else 0
      for t in list(range(0, Tp + 1, se)) + [Tp] * (hold + 1):
        snap = list(persisted)
        qpos = hk["qtraj"][:, t, :].reshape(-1).copy()

        def draw(scene, snap=snap, traces=traces, cols=cols, t=t, targets=targets):
          if targets is not None:
            for e in range(N):
              viz.add_disk(scene, np.array([targets[e, 0], targets[e, 1], 0.012]) + off[e],
                           (0.5, 0.0, 0.5), radius=0.025, alpha=0.85)
          for tr_all, pcols in snap:
            for e in range(N):
              viz.add_polyline(scene, tr_all[e][::2], pcols[e], width=pw, alpha=0.25)
          for e in range(N):
            viz.add_polyline(scene, traces[e][: t + 1], cols[e], width=tw)

        frames.append((qpos, draw, sub))
      persisted.append((traces, cols))
    if frames:
      frames += [frames[-1]] * 20
    live = True if self.args.live else None
    viz.emit(vm, vd, cam, frames, out_path=out, label=f"ADJOINT ({task.name})", w=W, h=H, fps=FPS, live=live)


def main(argv):
  del argv  # unused; config comes from the absl-parsed Args
  demo.run(DmSuiteDemo, demo.parse_args(Args))


if __name__ == "__main__":
  demo.define_flags(Args)
  app.run(main)
