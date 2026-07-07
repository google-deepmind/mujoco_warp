"""dm_control suite tasks under the differentiable mujoco_warp adjoint -- one switchable, PARALLEL harness.

A TASK REGISTRY (`TASKS`) maps a name to a `Task`: INLINED, flattened MJCF parts (dm_control's
`<include>`s resolved, integrator -> Euler for the analytic adjoint, cradle-style grid visuals; no runtime
dm_control dependency) + an objective + horizon/optimizer/camera config. One shared engine drives every
task, and it runs MANY ENVS IN PARALLEL (batched nworld>1, à la bounce.py / cradle.py): each env
optimizes its OWN control sequence from a per-env init spread, and ONE batched `tape.backward` yields
INDEPENDENT per-env gradients (each env's controls only drive its own world). Switch with `--task`, scale
with `--envs`.

  uv run --active --with imageio --with imageio-ffmpeg --with pillow \
    python contrib/diffsim/dm_suite.py --task cartpole --envs 6

Objectives:
  cartpole  : swing up & balance the pole (maximize cos(pole), cart centered)
  reacher   : each parallel env drives its 2-link fingertip to its OWN target
  cheetah   : run forward (locomotion: maximize forward velocity - action^2, dflex/SHAC HalfCheetah reward)

CONTROL BINDING: mujoco_warp's `step(d, d_out)` copies `ctrl` from d into d_out (ctrl is a step-state
field), so each step's control is bound to its OWN requires_grad leaf IMMEDIATELY BEFORE that step
(preloading lets step() propagate ctrl[0] into every later step -> the forward would ignore per-step ctrl).
"""

import argparse
import dataclasses
import os
import sys

import mujoco
import numpy as np
import warp as wp
import warp.optim

import mujoco_warp as mjw
mjw.enable_grad()

sys.path.insert(0, os.path.dirname(__file__))
import viz  # noqa: E402

W, H, FPS = 1024, 768, 30

# Shared cradle-style visuals (grid floor + skybox + haze), replacing dm_control's include files. The
# headlight/light intensity is scaled by a per-task `dim` (cradle defaults at dim=1): a near-top-down view
# like the reacher's washes the floor out under the full headlight, so that task renders at a lower dim.
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


def scene_visual(dim=1.0):
  """`<visual>` (headlight scaled by dim) + the shared `<asset>` block."""
  hd, ha, hs = 0.6 * dim, 0.4 * dim, 0.2 * dim
  return (f'<visual><headlight diffuse="{hd} {hd} {hd}" ambient="{ha} {ha} {ha}" specular="{hs} {hs} {hs}"/>'
          '<rgba haze="0.15 0.25 0.35 1"/><global offwidth="1280" offheight="960"/><map znear="0.02"/>'
          f'</visual>{_ASSETS}')


def scene_light(dim=1.0):
  """Directional key `<light>` (scaled by dim)."""
  d, s = 0.8 * dim, 0.3 * dim
  return f'<light pos="0 -1 3" dir="0 0.3 -1" diffuse="{d} {d} {d}" specular="{s} {s} {s}"/>'

# ---- per-task MJCF parts: (header, floor, robot [replicable], actuator) ----
CHEETAH = dict(
  compiler='<compiler settotalmass="14"/>',
  header='<option timestep="0.005" integrator="Euler"/>'
         '<default><default class="cheetah"><joint limited="true" damping=".01" armature=".1" stiffness="8" type="hinge" axis="0 1 0"/>'
         '<geom contype="1" conaffinity="1" condim="3" friction=".4 .1 .1" material="self" type="capsule"/></default>'
         '<default class="free"><joint limited="false" damping="0" armature="0" stiffness="0"/></default>'
         '<motor ctrllimited="true" ctrlrange="-1 1"/></default>',
  # floor widened in y (dm_control uses 4) so the multi-env LINE of cheetahs (y-lanes) all sit on the plane
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

# dm_control 2-link planar reacher (the EASIEST gradient path: smooth, contact-free). Joints rotate about
# world z with the arm in the horizontal plane, so gravity exerts NO torque about them -> reaching & holding
# a target is a STABLE problem (no unstable inverted equilibrium to hold). Both links are 0.12 m
# (shoulder->wrist and wrist->finger), reach = 0.24. The wrist limit is dropped so the path stays purely
# smooth (no constraint). Each parallel env is given its OWN target -> "N arms each reaching its own goal".
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


def single_xml(parts, dim=1.0):
  """Full single-env MJCF (physics + one robot) from the parts."""
  comp = parts.get("compiler", "")
  return (f'<mujoco>{comp}{parts["header"]}{scene_visual(dim)}<worldbody>{scene_light(dim)}{parts["floor"]}'
          f'{parts["robot"]}</worldbody>{parts["actuator"]}</mujoco>')


def grid_dims(n):
  """Near-square (nrow, ncol) with nrow*ncol >= n, for a 2D env lattice (ncol columns along x)."""
  ncol = max(1, int(round(n ** 0.5)))
  return -(-n // ncol), ncol  # (nrow=ceil(n/ncol), ncol)


def multi_xml(parts, n, spacing, grid=False, dim=1.0):
  """Viz-only MJCF with the robot REPLICATED n times (env-major qpos), one shared floor. Default = a single
  y-row; grid=True uses a near-square 2D lattice via NESTED replicate (inner=x/ncol, outer=y/nrow) so the
  env order is e = row*ncol + col (matches render's off[]). grid needs nrow*ncol == n (caller guarantees)."""
  comp = parts.get("compiler", "")
  if n <= 1:
    rep = parts["robot"]
  elif grid:
    nrow, ncol = grid_dims(n)
    inner = f'<replicate count="{ncol}" offset="{spacing} 0 0">{parts["robot"]}</replicate>'
    rep = f'<replicate count="{nrow}" offset="0 {spacing} 0">{inner}</replicate>'
  else:
    rep = f'<replicate count="{n}" offset="0 {spacing} 0">{parts["robot"]}</replicate>'
  return (f'<mujoco>{comp}{parts["header"]}{scene_visual(dim)}<worldbody>{scene_light(dim)}{parts["floor"]}'
          f'{rep}</worldbody></mujoco>')


@dataclasses.dataclass
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
  target: dict = None           # reacher per-env GOAL fan: {"r":.., "ang_lo":.., "ang_hi":..} (arc of targets)
  action_repeat: int = 1        # hold each action for k physics substeps: k*T physical horizon, still T actions
  cam: dict = dataclasses.field(default_factory=dict)


TASKS = {
  "cheetah": Task("cheetah", CHEETAH, T=240, steps=200, lr=0.04, objective="locomotion",
                  # dflex/SHAC HalfCheetah reward = forward velocity - 0.1*action^2. pitch=0.2 is a MILD upright
                  # bias added on top of the pure dflex reward: the pure reward sometimes finds a fast TUMBLING
                  # gait (pitch->2+); a small pitch^2 penalty keeps the cheetahs running on their feet.
                  idx={"fwd": 0, "z": 1, "pitch": 2}, w={"fwd": 1.0, "up": 0.0, "pitch": 0.2, "ctrl": 0.1},
                  # per-env varied back-thigh start (qpos idx 3) so the parallel cheetahs learn slightly
                  # different gaits -> a staggered race in the line, not identical clones.
                  spread={"idx": 3, "lo": -0.25, "hi": 0.25},
                  cam={"z": 0.6, "track": True}),
  "cartpole": Task("cartpole", CARTPOLE, T=200, steps=200, lr=0.02, objective="cartpole",
                   idx={"cart": 0, "pole": 1},  # dflex/SHAC penalties: angle,pole-vel,cart-pos,cart-vel
                   w={"ang": 1.0, "pvel": 0.1, "cpos": 0.2, "cvel": 0.2, "ctrl": 0.0},  # cart pos/vel END-weighted
                   init_qpos=np.array([0.0, np.pi - 0.25]), settle=0,
                   # one-sided init fan with a MINIMUM tilt (hi = pi-0.25, ~14 deg off bottom) so no env
                   # starts too near the symmetric bottom -> all lean decisively the same way & swing up alike.
                   spread={"idx": 1, "lo": np.pi - 0.65, "hi": np.pi - 0.25}, cam={"z": 1.0, "dist": 4.0}),
  "reacher": Task("reacher", REACHER, T=120, steps=120, lr=0.05, objective="reacher",
                  idx={"sh": 0, "wr": 1}, w={"dist": 1.0, "vel": 0.01, "ctrl": 1e-4},
                  # all envs share this start; GOAL differs. Well-BENT (elbow 1.5 rad, fingertip radius 0.176
                  # << reach 0.24 -> away from the straight-arm singularity) & pointing ~+x, centered in the
                  # goal fan -> gradient descent reaches every goal on both sides (a near-straight init strands
                  # the opposite-elbow side). Verified: worst env 1.0 cm vs 6 cm for a near-singular init.
                  init_qpos=np.array([-0.75, 1.5]), settle=0,
                  # goals on a FRONTAL arc (r<reach 0.24) centered on the start direction, so every target is
                  # reachable without a ~180-deg wrap-around distance local-min (a wide arc stranded one env).
                  target={"r": 0.18, "ang_lo": -1.0, "ang_hi": 1.0},  # tighter arc: every goal easily reachable
                  cam={"grid": True, "dim": 0.8, "z": 0.02, "dist": 0.5, "az": 90.0, "el": -60.0,
                       "tw": 0.004, "pw": 0.0022}),
}


def env_targets(task, N):
  """Per-env reacher goal positions (N, 2) on an arc of radius r within reach; N=1 -> the arc midpoint.
  Deterministic in (task.target, N) so build / loss / score / render agree without threading state."""
  t = task.target
  angs = np.array([0.5 * (t["ang_lo"] + t["ang_hi"])]) if N == 1 else np.linspace(t["ang_lo"], t["ang_hi"], N)
  return np.stack([t["r"] * np.cos(angs), t["r"] * np.sin(angs)], axis=1).astype(np.float32)


# ---------------- batched loss kernels (dim = nworld) ----------------
@wp.kernel
def _loc_reward(qpos: wp.array2d[float], qvel: wp.array2d[float], fwd: int, z: int, pitch: int,
                z_low: float, w_fwd: float, w_up: float, w_pitch: float, loss: wp.array(dtype=float)):
  w = wp.tid()
  wp.atomic_add(loss, 0, -w_fwd * qvel[w, fwd] - w_up * (qpos[w, z] - z_low) + w_pitch * qpos[w, pitch] * qpos[w, pitch])


@wp.kernel
def _cartpole_reward(qpos: wp.array2d[float], qvel: wp.array2d[float], cart: int, pole: int,
                     w_ang: float, w_pvel: float, w_cpos: float, w_cvel: float, ws: float,
                     loss: wp.array(dtype=float)):
  # dflex/SHAC cartpole reward: drive the pole upright (theta->0, normalized) + keep it slow (uniform),
  # and settle the CART centered & still. The cart position/velocity penalties are END-WEIGHTED (ws=(t+1)/T):
  # ~free early so the cart can still pump energy into the swing-up, strong late so the base ends centered
  # and stops swinging (open-loop trajopt's substitute for SHAC's learned centering policy). loss = -reward.
  w = wp.tid()
  th = wp.atan2(wp.sin(qpos[w, pole]), wp.cos(qpos[w, pole]))  # normalize_angle(pole): 0 = upright
  wp.atomic_add(loss, 0, w_ang * th * th + w_pvel * qvel[w, pole] * qvel[w, pole]
                + ws * (w_cpos * qpos[w, cart] * qpos[w, cart] + w_cvel * qvel[w, cart] * qvel[w, cart]))


@wp.kernel
def _reacher_reward(qpos: wp.array2d[float], qvel: wp.array2d[float], sh: int, wr: int, l1: float, l2: float,
                    tx: wp.array(dtype=float), ty: wp.array(dtype=float), w_dist: float, w_vel: float,
                    ws: float, loss: wp.array(dtype=float)):
  # 2-link planar FK fingertip from qpos (matches MuJoCo to 1e-17), minimize distance^2 to THIS env's target
  # every step (reach fast & hold), + end-weighted joint-vel penalty so it settles on the goal without jitter.
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


def _launch_loss(task, d1, ctrl, z_low, t, T, N, loss, tgt=None):
  o, ix, w = task.objective, task.idx, task.w
  ws = float((t + 1) / T)
  if o == "locomotion":
    wp.launch(_loc_reward, dim=N, inputs=[d1.qpos, d1.qvel, ix["fwd"], ix["z"], ix["pitch"], z_low,
                                          w["fwd"], w["up"], w["pitch"]], outputs=[loss])
  elif o == "cartpole":
    wp.launch(_cartpole_reward, dim=N, inputs=[d1.qpos, d1.qvel, ix["cart"], ix["pole"],
                                              w["ang"], w["pvel"], w["cpos"], w["cvel"], ws], outputs=[loss])
  elif o == "reacher":
    tx, ty = tgt  # per-env target coords (wp arrays), built once in run_backward
    wp.launch(_reacher_reward, dim=N, inputs=[d1.qpos, d1.qvel, ix["sh"], ix["wr"], REACH_L1, REACH_L2,
                                              tx, ty, w["dist"], w.get("vel", 0.0), ws], outputs=[loss])
  if w.get("ctrl", 0.0) > 0.0:
    wp.launch(_ctrl_cost, dim=(N, ctrl.shape[1]), inputs=[ctrl, w["ctrl"]], outputs=[loss])


# ---------------- per-env score (numpy; for coloring / metric / best) ----------------
def per_env_score(task, qtraj):
  """Higher = better, per env. qtraj: (N, T+1, nq)."""
  o, ix = task.objective, task.idx
  if o == "locomotion":
    return qtraj[:, -1, ix["fwd"]] - qtraj[:, 0, ix["fwd"]]                      # forward distance
  if o == "cartpole":
    return np.mean(np.cos(qtraj[:, -20:, ix["pole"]]), axis=1)                   # end upright (cos)
  # reacher: higher (~0) = fingertip on the goal
  tg = env_targets(task, qtraj.shape[0])                                         # (N,2)
  th1, th2 = qtraj[:, -20:, ix["sh"]], qtraj[:, -20:, ix["wr"]]
  fx = REACH_L1 * np.cos(th1) + REACH_L2 * np.cos(th1 + th2)
  fy = REACH_L1 * np.sin(th1) + REACH_L2 * np.sin(th1 + th2)
  d = np.hypot(fx - tg[:, 0:1], fy - tg[:, 1:2])                                 # (N,20) end fingertip->goal
  return -d.mean(axis=1)


# ---------------- engine ----------------
def build(task, N):
  mjm = mujoco.MjModel.from_xml_string(single_xml(task.parts))
  mjd = mujoco.MjData(mjm)
  m = mjw.put_model(mjm)
  if task.init_qpos is not None:
    base = task.init_qpos.copy()
  else:
    mujoco.mj_resetData(mjm, mjd)
    mujoco.mj_forward(mjm, mjd)
    ds = [mjw.put_data(mjm, mjd) for _ in range(task.settle + 1)]
    for t in range(task.settle):
      mjw.step(m, ds[t], ds[t + 1])
    base = ds[task.settle].qpos.numpy()[0].copy()
  q0 = np.tile(base, (N, 1)).astype(np.float32)
  if task.spread and N > 1:
    q0[:, task.spread["idx"]] = np.linspace(task.spread["lo"], task.spread["hi"], N)
  return mjm, mjd, m, q0


def run_backward(task, m, mjm, mjd, ctrls, q0, N):
  """Batched (nworld=N) taped rollout using the PERSISTENT per-step control params `ctrls` (1D, size N*nu);
  each is reshaped to (N,nu) for the mjw binding (a data+grad-sharing view). Leaves the gradients in
  ctrls[t].grad for warp.optim.Adam. Returns qtraj (N,T+1,nq)."""
  nu, k = mjm.nu, max(1, task.action_repeat)
  T = len(ctrls) * k                                 # PHYSICS horizon: len(ctrls) actions, each held k substeps
  # z_low: locomotion stay-tall floor (start height - 0.35); unused (0) by cartpole/reacher
  z_low = float(np.mean(q0[:, task.idx["z"]])) - 0.35 if task.objective == "locomotion" else 0.0
  tgt = None
  if task.objective == "reacher":
    tg = env_targets(task, N)                        # (N,2) constant per-env goals
    tgt = (wp.array(tg[:, 0], dtype=float), wp.array(tg[:, 1], dtype=float))
  for c in ctrls:
    c.grad.zero_()                                   # tape.backward ACCUMULATES -> zero before each iter
  datas = [mjw.put_data(mjm, mjd, nworld=N) for _ in range(T + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  datas[0].qpos = wp.array(q0, dtype=float, requires_grad=True)
  datas[0].qvel = wp.array(np.zeros((N, mjm.nv), np.float32), dtype=float, requires_grad=True)
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(T):
      ct = ctrls[t // k].reshape((N, nu))            # ACTION REPEAT: same action leaf for k steps; JIT-bound.
      datas[t].ctrl = ct                             # (step copies ctrl into d_out, so JIT rebind matters)
      mjw.step(m, datas[t], datas[t + 1])
      _launch_loss(task, datas[t + 1], ct, z_low, t, T, N, loss, tgt)
  tape.backward(loss=loss)
  return np.stack([datas[t].qpos.numpy() for t in range(T + 1)], axis=1)


def optimize(task, N):
  mjm, mjd, m, q0 = build(task, N)
  nu = mjm.nu
  # PERSISTENT per-step control params (leaves) optimized by warp.optim.Adam with SHAC betas (0.7, 0.95)
  # -- fast-adapting moments, so no LR schedule is needed. 1D (size N*nu) as warp.optim.Adam requires.
  ctrls = [wp.array(np.zeros(N * nu, np.float32), dtype=float, requires_grad=True) for _ in range(task.T)]
  opt = warp.optim.Adam(ctrls, lr=task.lr, betas=(0.7, 0.95))
  history = []
  for it in range(task.steps):
    qtraj = run_backward(task, m, mjm, mjd, ctrls, q0, N)
    score = per_env_score(task, qtraj)
    history.append({"it": it, "score": score.copy(), "qtraj": qtraj})
    if it % 20 == 0 or it == task.steps - 1:
      gnorm = float(np.linalg.norm(np.array([c.grad.numpy().reshape(N, nu) for c in ctrls]), axis=(0, 2)).mean())
      extra = ""
      if task.objective == "cartpole":
        extra = f"  |cart|_end={np.abs(qtraj[:, -1, task.idx['cart']]).mean():.2f}"
      elif task.objective == "locomotion":  # gait: fwd dist, peak/min torso height, max pitch
        z, p, fi = qtraj[:, :, task.idx["z"]], qtraj[:, :, task.idx["pitch"]], task.idx["fwd"]
        fwd = float(np.mean(qtraj[:, -1, fi] - qtraj[:, 0, fi]))
        extra = f"  fwd={fwd:+.2f} peakZ={z.max():+.2f} minZ={z.min():+.2f} |pitch|mx={np.abs(p).max():.2f}"
      print(f"  [{it:3d}] mean_score={score.mean():+.3f}  best={score.max():+.3f}  "
            f"worst={score.min():+.3f}{extra}  mean|g|={gnorm:.2f}")
    opt.step([c.grad for c in ctrls])                # in-place Adam update of the control params
    for c in ctrls:
      wp.launch(_clamp01, dim=c.shape, inputs=[c])   # project back to the ctrl range [-1, 1]
  mean_score = np.array([h["score"].mean() for h in history])
  best = int(mean_score.argmax())
  print(f"[{task.name}] {N} env(s): mean_score {mean_score[0]:+.3f} -> best {mean_score[best]:+.3f} (iter {best})")
  return history, best, mjm, q0


# ---------------- render ----------------
def trace(task, qtraj_e):
  """Tracked point path (T+1,3) for one env. qtraj_e: (T+1,nq)."""
  o, ix = task.objective, task.idx
  if o == "locomotion":  # cheetah torso path: rootz is a slide from the torso body's z=0.7 -> world = 0.7 + rootz
    return np.stack([qtraj_e[:, ix["fwd"]], np.zeros(len(qtraj_e)), 0.7 + qtraj_e[:, ix["z"]]], 1)
  if o == "cartpole":
    a = qtraj_e[:, ix["pole"]]
    return np.stack([qtraj_e[:, ix["cart"]] + np.sin(a), np.zeros(len(a)), 1.0 + np.cos(a)], 1)
  if o == "reacher":  # fingertip path in the horizontal plane (z = arm height)
    th1, th2 = qtraj_e[:, ix["sh"]], qtraj_e[:, ix["wr"]]
    fx = REACH_L1 * np.cos(th1) + REACH_L2 * np.cos(th1 + th2)
    fy = REACH_L1 * np.sin(th1) + REACH_L2 * np.sin(th1 + th2)
    return np.stack([fx, fy, np.full(len(th1), 0.01)], 1)
  raise ValueError(f"no trace for objective {o}")


def render(task, history, best, N, out_mp4, spacing, sample_every=4):
  c = task.cam
  nrow, ncol = grid_dims(N)
  use_grid = bool(c.get("grid")) and N > 1 and nrow * ncol == N  # 2D lattice only when it tiles N exactly
  vm = mujoco.MjModel.from_xml_string(multi_xml(task.parts, N, spacing, grid=use_grid, dim=c.get("dim", 1.0)))
  vd = mujoco.MjData(vm)
  off = np.zeros((N, 3))
  if use_grid:
    off[:, 0] = (np.arange(N) % ncol) * spacing  # env e = row*ncol + col (matches nested replicate order)
    off[:, 1] = (np.arange(N) // ncol) * spacing
  else:
    off[:, 1] = np.arange(N) * spacing            # single y-row
  # reacher: per-env GOAL markers (gold disks on the plane) so each lane's "reach its own target" reads clearly
  targets = env_targets(task, N) if task.objective == "reacher" else None
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(vm, cam)
  ymid = float(off[:, 1].mean())
  if c.get("track"):
    allx = np.concatenate([history[best]["qtraj"][:, :, task.idx["fwd"]].ravel()])
    tx, span = float(allx.mean()), float(max(allx.max() - allx.min(), 1.5))
    cam.lookat = [tx, ymid, c.get("z", 1.0)]
    cam.distance = max(5.0, 1.2 * max(span, (N - 1) * spacing) + 2.0)
  elif use_grid:
    # frame the CONTENT bbox center, not just the base lattice: reacher = bases + goals; locomotion =
    # the best iterate's per-lane world torso path (so forward travel stays in view as the envs move off-grid)
    if targets is not None:
      pts = np.concatenate([off[:, :2], off[:, :2] + targets], axis=0)
      margin = 0.15
    else:
      bq = history[best]["qtraj"]
      pts = np.concatenate([(trace(task, bq[e]) + off[e])[:, :2] for e in range(N)], axis=0)
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
  tw, pw = c.get("tw", 0.014), c.get("pw", 0.006)  # trace / persisted line widths (thinner for the small reacher)

  def colors(sc):  # best score -> blue, worst -> red (bourke: low->blue, high->red -> pass hi-sc)
    return [viz.bourke_color_map(0.0, hi - lo + 1e-9, hi - float(v)) for v in sc]

  T = history[0]["qtraj"].shape[1] - 1  # PHYSICS horizon (= task.T * action_repeat), not the action count
  se = max(sample_every, round(T / 90))  # subsample long (action-repeat) rollouts to ~90 frames/iteration
  best_mean = max(h["score"].mean() for h in history)
  frames, persisted = [], []
  for k in viz.default_show(len(history), best):
    hk = history[k]
    cols = colors(hk["score"])
    traces = [trace(task, hk["qtraj"][e]) + off[e] for e in range(N)]
    sub = f"iter {hk['it']:3d}    mean {hk['score'].mean():+.3f}    best {best_mean:+.3f}    envs {N}"
    hold = 20 if k == best else 0
    for t in list(range(0, T + 1, se)) + [T] * (hold + 1):
      snap = list(persisted)
      qpos = hk["qtraj"][:, t, :].reshape(-1).copy()   # env-major -> replicate qpos layout

      def draw(scene, snap=snap, traces=traces, cols=cols, t=t, targets=targets):
        if targets is not None:  # goal markers (drawn first so traces overlay them)
          for e in range(N):
            viz.add_disk(scene, np.array([targets[e, 0], targets[e, 1], 0.012]) + off[e],
                         (0.5, 0.0, 0.5), radius=0.025, alpha=0.85)  # purple, matching bounce.py's target
        for tr_all, pcols in snap:
          for e in range(N):
            viz.add_polyline(scene, tr_all[e][::2], pcols[e], width=pw, alpha=0.25)
        for e in range(N):
          viz.add_polyline(scene, traces[e][: t + 1], cols[e], width=tw)

      frames.append((qpos, draw, sub))
    persisted.append((traces, cols))
  if frames:
    frames += [frames[-1]] * 20
  viz.emit(vm, vd, cam, frames, out_path=out_mp4, label=f"ADJOINT ({task.name})", w=W, h=H, fps=FPS)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default=os.environ.get("MJW_TASK", "cartpole"), choices=list(TASKS))
  ap.add_argument("--envs", type=int, default=int(os.environ.get("MJW_ENVS", "4")))
  ap.add_argument("--steps", type=int, default=None)
  ap.add_argument("--T", type=int, default=None, help="number of ACTIONS; physical horizon = T * action-repeat")
  ap.add_argument("--repeat", type=int, default=None, help="action repeat: hold each action k physics substeps")
  ap.add_argument("--lr", type=float, default=None)
  ap.add_argument("--spread", type=float, default=None, help="scale the per-env init fan width (0=identical envs)")
  ap.add_argument("--spacing", type=float, default=None)
  ap.add_argument("--out", default=None)
  ap.add_argument("--no-render", action="store_true")
  args = ap.parse_args()
  task = TASKS[args.task]
  if args.T is not None:
    task.T = args.T
  if args.repeat is not None:
    task.action_repeat = args.repeat
  if args.steps is not None:
    task.steps = args.steps
  if args.lr is not None:
    task.lr = args.lr
  if args.spread is not None and task.spread:  # scale the init-fan width, ANCHORED at `hi` (0 -> identical envs)
    # anchor the near-bottom end (hi) and extend `lo` away from it, so the whole fan stays on ONE side of
    # the equilibrium (poles all tilted the same way -> all swing up to the same side); avoids crossing over.
    width = (task.spread["hi"] - task.spread["lo"]) * args.spread
    task.spread = {**task.spread, "lo": task.spread["hi"] - width}
  N = max(1, args.envs)
  spacing = args.spacing if args.spacing is not None else {"cartpole": 1.2, "reacher": 0.42, "cheetah": 2.0}.get(task.name, 2.0)
  print(f"=== dm_suite: {task.name} (objective={task.objective}, actions={task.T}x repeat={task.action_repeat}"
        f" -> {task.T * task.action_repeat} phys steps, steps={task.steps}, envs={N}) ===")
  history, best, mjm, q0 = optimize(task, N)
  if not args.no_render:
    out = args.out or os.environ.get("MJW_RENDER_PATH", f"/tmp/dm_{task.name}.mp4")
    render(task, history, best, N, out, spacing)


if __name__ == "__main__":
  main()
