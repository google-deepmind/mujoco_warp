"""Unitree Go1 AGILE skills -- open-loop trajectory optimization via analytic-adjoint BPTT (4 tasks).

One task-switchable, PARALLEL (nworld-batched) harness driving the mjlab Go1 through three agile tasks
adapted from dial-mpc:
  * crate_climb    -- leap/climb onto a tall crate: trunk -> on-crate target, stay upright. (scene_crate.xml)
  * precision_jump -- hop forward across a path of stepping-stone cylinders: COM tracks a forward
                      sequence of waypoints, stay upright. (scene_stones.xml)
  * gallop         -- run forward at a target speed, hold ride height, stay upright. (scene_flat.xml)
  * handstand      -- pitch onto the front legs, keep the head supported, and settle. (scene_flat.xml)

Each task loads a static scene XML from benchmarks/unitree_go1/ (Go1 + obstacles), inits from the Go1
`init_state` crouch keyframe, and optimizes an OPEN-LOOP per-step joint-position-target sequence over N
parallel envs. Parallelism is mujoco_warp `nworld` batching (one model, N data worlds, ONE batched
tape.backward -> independent per-env gradients, exactly like bounce.py). dL/dctrl back-propagates
through the whole contact-rich rollout via differentiable mujoco_warp (adjoint.py) -- the path-E case
(articulated body <-> ground/obstacle contact over a long multi-contact horizon).

This is the BPTT SETUP / scene test-harness: open-loop single-shooting is NOT expected to solve these
agile skills (deceptive long-horizon contact gradients) -- iLQR will. It exercises the scenes + the
differentiable-contact gradient. Analytic adjoint only (finite differences are hopeless at 12*T*N dims).
Render = all environments animating in separate y-lanes + per-env COM trails (colored by loss) -> reports/assets/.

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/go1_agile.py --task crate_climb
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/go1_agile.py --task precision_jump --envs 4
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/go1_agile.py --task gallop
  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/go1_agile.py --task handstand --envs 4
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
import viz  # noqa: E402  shared renderer (mp4 or live MuJoCo viewer via MJW_VIEWER)

BENCH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "unitree_go1"))
ASSETS = os.path.join(os.path.dirname(__file__), "reports", "assets")

DT = 0.004
SETTLE = 200          # steps holding the crouch keyframe -> standing at rest
ACTION = 1.2          # per-joint target excursion bound around the home crouch (rad)
SPREAD = 0.12         # per-env constant control-bias spread -> the parallel fan starts distinct
W, H, FPS = 1024, 768, 30

FEET = ["FR", "FL", "RR", "RL"]   # foot site names (front-right/left, rear-right/left)

# dial-mpc gallop gait (dial_mpc.utils.function_utils.get_foot_step + unitree_go2_env gait tables).
# dial foot order [FL,FR,RL,RR] phases [0,0.05,0.4,0.35] -> reordered to OUR order [FR,FL,RR,RL]:
GALLOP_PHASES = [0.05, 0.0, 0.35, 0.4]
GALLOP_DUTY, GALLOP_CADENCE, GALLOP_AMP = 0.3, 3.5, 0.10   # ratio on ground, strides/s, swing height (m)
# Raibert foothold: x*_rel = hip_offset + REACH*v + K*(v - v_desired). REACH ~ T_stance/2 = (DUTY/CADENCE)/2
# (the neutral point -- larger over-strides & brakes); K places feet BEHIND neutral when slow -> accelerate.
RAIBERT_REACH, RAIBERT_K = 0.045, 0.20


def _foot_step_heights(t_sec):
  """dial-mpc get_foot_step: per-foot swing-height target (above ground) at time t_sec, in OUR foot order.
  Returns (4,) in [0, amplitude] (0 in stance, rising to amp mid-swing). This is a CONSTANT reference --
  the gradient flows through the MEASURED foot z (analytic fwd_kinematics site VJP), never through this."""
  tt = t_sec * 2.0 * np.pi * GALLOP_CADENCE + np.pi
  angle = (tt + np.pi - 2.0 * np.pi * np.asarray(GALLOP_PHASES)) % (2.0 * np.pi) - np.pi
  angle = angle * 0.5 / (1.0 - GALLOP_DUTY)                # duty_ratio < 1
  value = np.abs(np.cos(np.clip(angle, -np.pi / 2.0, np.pi / 2.0)))
  return GALLOP_AMP * np.where(value >= 1e-6, value, 0.0)


# precision_jump AIRBORNE PRONK: all 4 feet leave the tall pillars together and the BODY arcs UP into a
# ballistic flight, landing on the next pillars. The COM z-arc (HOP_HEIGHT) is what makes it a real jump --
# a level COM ref can only produce a grounded shuffle (nothing lets all 4 feet leave at once). The tall
# pillars (scene) forbid the floor-drag cheat, so the optimizer must actually launch to clear the gaps.
STONES_RAISE = 0.13         # base raise at build so feet spawn above the tall pillar tops (top z=0.12), settle down
STONES_LEAD = 0.12          # diagnostic hop-0 starts during the first load; hop boundary is launch 2 at frame 168
STONES_CROUCH = 0.04        # load each jump by lowering the trunk while all four feet remain planted
STONES_SWING = 0.15         # foot arc above the pillar top; > COM arc so the legs tuck during flight
HOP_HEIGHT = 0.102          # g*T_flight^2/8 for an honest 72*0.004 = 0.288 s ballistic parabola
STONES_XY_DONE = 0.62       # get over the target early, then descend nearly vertically onto its center
STONES_PREP_STEPS = 24      # physical phase timing is fixed independently of the gradient truncation window
STONES_FIRST_LAUNCH = 48
STONES_FLIGHT_STEPS = 72
STONES_HOP_PERIOD = 120


def _stones_refs(task, foot_targets, qpos0):
  """Per-step reference trajectories for the AIRBORNE PRONK, from the per-STAGE foot_targets (n_stages,4,3).
  Within each hop ALL 4 feet swing SYNCHRONIZED (xy lerps pillar->pillar, z arcs up) AND the trunk COM arcs
  UP by HOP_HEIGHT then forward to the next stance center -> the whole robot launches off the pillars, flies,
  and lands on the next set. Each 72-frame flight is followed by a real planted dwell. Physical phase timing
  is independent of TBPTT, so touchdown can sit inside a backward window and receive post-contact feedback.
  Feet complete their xy transfer before descending, avoiding a
  diagonal strike on the pillar rim. stance_ref is exactly zero throughout flight and one only while planted;
  land_ref ramps up during vertical descent and remains one through the dwell. All targets are CONSTANT; the
  gradient flows through measured foot site_xpos / trunk qpos, never through these."""
  T = task.T
  z0 = float(qpos0[2])
  first_launch = STONES_FIRST_LAUNCH
  flight_len = STONES_FLIGHT_STEPS
  hop_period = STONES_HOP_PERIOD                         # launch frames 48 and 168; touchdown 120 and 240
  stone_z = float(foot_targets[0, 0, 2])               # on-pillar foot-site height (land-on-top target)
  com_offset = np.asarray(qpos0[:2]) - foot_targets[0, :, :2].mean(axis=0)
  com_tar = foot_targets[:, :, :2].mean(axis=1) + com_offset[None, :]  # preserve settled trunk/feet offset
  foot_ref = np.zeros((T, 4, 3), np.float32)
  com_ref = np.zeros((T, 3), np.float32)
  com_vel_ref = np.zeros((T, 3), np.float32)
  stance_ref = np.zeros((T, 4), np.float32)
  land_ref = np.zeros(T, np.float32)
  clear_ref = np.zeros(T, np.float32)                    # second-flight hind-knee tuck/clearance gate
  finish_ref = np.zeros(T, np.float32)                   # second-hop COM completion gate
  pose_ref = np.zeros(T, np.float32)                     # rear landing-pose gate (off during the next load/push)
  raib_ox = np.zeros((T, 4), np.float32)               # RAIBERT: fore-aft offset (foot_x - trunk_x) the swing foot reaches
  raib_w = np.zeros((T, 4), np.float32)                # RAIBERT: descent gate (feet reaching toward the landing pillar)
  for t in range(T):
    frame = t + 1                                       # loss at control t reads the output state d[t+1]
    stage = 0
    flight = -1
    for h in range(task.n_hops):
      launch = first_launch + h * hop_period
      touchdown = launch + flight_len
      if launch < frame <= touchdown:
        flight = h
        break
      if frame > touchdown:
        stage = h + 1

    if flight >= 0:
      h = flight
      dest = h + 1
      launch = first_launch + h * hop_period
      p = float(np.clip((frame - launch) / flight_len, 0.0, 1.0))
      foot_arc = float(np.sin(np.pi * p) ** 2)           # zero vertical speed at lift-off and touchdown
      if p <= 0.5:
        com_arc = 4.0 * p * (1.0 - p)                  # ballistic-like launch and ascent
      else:
        down = 2.0 * p - 1.0
        com_arc = 1.0 - down * down * (3.0 - 2.0 * down)  # absorb to zero vertical speed at touchdown
      cp = p                                              # gallop-style near-constant forward flight speed
      px = min(p / STONES_XY_DONE, 1.0)
      fp = px * px * (3.0 - 2.0 * px)                  # feet get over the destination before descending
      foot_ref[t, :, :2] = (1.0 - fp) * foot_targets[h, :, :2] + fp * foot_targets[dest, :, :2]
      foot_ref[t, :, 2] = stone_z + STONES_SWING * foot_arc
      com_ref[t, :2] = (1.0 - cp) * com_tar[h] + cp * com_tar[dest]
      com_ref[t, 2] = z0 + HOP_HEIGHT * com_arc
      stance_ref[t, :] = 0.0                            # never oppose necessary swing/descent correction
      lp = float(np.clip((p - STONES_XY_DONE) / (1.0 - STONES_XY_DONE), 0.0, 1.0))
      land_ref[t] = lp * lp * (3.0 - 2.0 * lp)          # exact-center/top pull during vertical descent
      pose_ref[t] = land_ref[t] * land_ref[t]
      if h == task.n_hops - 1:
        # The compact rear legs otherwise unfold over the previous pillar row. Keep their knees tucked through
        # mid-flight, then release early enough to open for touchdown. Completion starts inside the same H=32
        # chunk as launch, so its gradient can still train the second push-off instead of only the late descent.
        cin = float(np.clip((p - 0.18) / 0.14, 0.0, 1.0))
        cout = float(np.clip((0.84 - p) / 0.16, 0.0, 1.0))
        cin = cin * cin * (3.0 - 2.0 * cin)
        cout = cout * cout * (3.0 - 2.0 * cout)
        clear_ref[t] = min(cin, cout)
        fg = float(np.clip(p / 0.20, 0.0, 1.0))
        finish_ref[t] = fg * fg * (3.0 - 2.0 * fg)
      raib_ox[t, :] = foot_targets[dest, :, 0] - com_tar[dest, 0]
      raib_w[t, :] = 1.0 if p > 0.5 else 0.0
    else:
      foot_ref[t] = foot_targets[stage]
      com_ref[t, :2] = com_tar[stage]
      stance_ref[t, :] = 1.0
      land_ref[t] = 1.0 if stage > 0 else 0.0
      finish_ref[t] = 1.0 if stage == task.n_hops else 0.0
      next_launch = first_launch + stage * hop_period
      prep = float(np.clip((frame - (next_launch - STONES_PREP_STEPS)) / STONES_PREP_STEPS, 0.0, 1.0)) if stage < task.n_hops else 0.0
      pose_ref[t] = 1.0 if stage > 0 and (stage == task.n_hops or prep <= 0.0) else 0.0
      if prep <= 0.5:
        load = 2.0 * prep
        load_sp = load * load * (3.0 - 2.0 * load)
        com_ref[t, 2] = z0 - STONES_CROUCH * load_sp     # first 12 frames: smooth compression
      else:
        push = 2.0 * prep - 1.0
        com_ref[t, 2] = z0 - STONES_CROUCH * (1.0 - push * push)  # next 12: accelerating extension
      raib_ox[t, :] = foot_targets[stage, :, 0] - com_tar[stage, 0]
  com_vel_ref[0] = (com_ref[0] - np.array([qpos0[0], qpos0[1], qpos0[2]], np.float32)) / task.dt
  com_vel_ref[1:] = (com_ref[1:] - com_ref[:-1]) / task.dt
  return foot_ref, com_ref, com_vel_ref, stance_ref, land_ref, clear_ref, finish_ref, pose_ref, raib_ox, raib_w


def _stones_ctrl_seed(task, home):
  """Force-limited pronk initial guess: compress, extend/sweep back, tuck, then open for touchdown."""
  seed = np.tile(home[None, :], (task.T, 1)).astype(np.float32)
  first_launch, flight_len, hop_period = STONES_FIRST_LAUNCH, STONES_FLIGHT_STEPS, STONES_HOP_PERIOD
  thighs = np.array([1, 3, 5, 7])
  front_thighs = np.array([1, 3])
  rear_thighs = np.array([5, 7])
  calves = np.array([8, 9, 10, 11])
  for t in range(task.T):
    frame = t + 1
    c = seed[t]
    for h in range(task.n_hops):
      launch, touchdown = first_launch + h * hop_period, first_launch + h * hop_period + flight_len
      if launch - STONES_PREP_STEPS < frame <= launch:
        prep = (frame - (launch - STONES_PREP_STEPS)) / STONES_PREP_STEPS
        if prep <= 0.5:
          u = 2.0 * prep
          s = u * u * (3.0 - 2.0 * u)
          c[calves] = home[calves] - 0.45 * s            # flex all knees into a compact load
        else:
          u = 2.0 * prep - 1.0
          s = u * u * (3.0 - 2.0 * u)
          c[calves] = home[calves] - 0.45 + 1.10 * s     # synchronized force-limited leg extension
          c[thighs] = home[thighs] + 0.55 * s            # sweep planted feet back -> forward body impulse
        break
      if launch < frame <= touchdown:
        p = (frame - launch) / flight_len
        ext = home.copy(); ext[calves] += 0.65; ext[thighs] += 0.55
        tuck = home.copy(); tuck[calves] -= 0.45; tuck[thighs] -= 0.25
        if h == task.n_hops - 1:
          # A shared thigh tuck is wrong for the compact stance: front home is 1.4 rad but rear home is
          # 0.4 rad, so subtracting 0.25 from both drives the rear thighs to 0.15 and drops their knees onto
          # the preceding cylinders. Preserve the sagittal foot offsets and hold the safe tuck through apex.
          tuck[front_thighs] = home[front_thighs]
          tuck[rear_thighs] = home[rear_thighs] + 0.50
          if p <= 0.15:
            c[:] = ext
          elif p <= 0.40:
            u = (p - 0.15) / 0.25; s = u * u * (3.0 - 2.0 * u)
            c[:] = (1.0 - s) * ext + s * tuck
          elif p <= 0.68:
            c[:] = tuck
          else:
            u = (p - 0.68) / 0.32; s = u * u * (3.0 - 2.0 * u)
            c[:] = (1.0 - s) * tuck + s * home
        else:
          if p <= 0.15:
            c[:] = ext
          elif p <= 0.45:
            u = (p - 0.15) / 0.30; s = u * u * (3.0 - 2.0 * u)
            c[:] = (1.0 - s) * ext + s * tuck
          else:
            u = (p - 0.45) / 0.55; s = u * u * (3.0 - 2.0 * u)
            c[:] = (1.0 - s) * tuck + s * home
        break
  return seed


@dataclasses.dataclass
class Task:
  name: str
  scene: str                                  # xml in benchmarks/unitree_go1/
  objective: str                              # "reach" | "sequence" | "gallop" | "handstand" | "stones"
  T: int = 220
  steps: int = 120
  lr: float = 0.03
  spread: float = SPREAD                       # per-env init control-bias std (the parallel fan); smaller = tighter, less likely to topple an env
  n_hops: int = 3                              # stones: number of forward stone-to-stone steps (stages = n_hops+1)
  vel_tar: float = 1.0                         # gallop: forward speed (m/s)
  height_tar: float = 0.30                     # gallop: trunk ride height (m)
  lane: float = 1.0                            # per-env y-lane spacing for the render (all envs in one scene)
  dt: float = DT                               # sim timestep (per-task): bigger dt -> more physical time per step (slower hops) at a FIXED horizon T
  tbptt_h: int = 0                             # >0: TRUNCATED BPTT -- detach state every h frames so each backward chain is short (dodges long-horizon chaos/local-optima); grads still accumulate for the full-trajectory loss. 0 = full BPTT
  grad_clip: float = 0.0                       # >0: cap mean per-step |grad| each iter (trajopt hygiene; stops the aggressive-launch overshoot)
  title: str = ""                              # render overlay name: "ADJOINT (<title>)" (defaults to name)
  w: dict = dataclasses.field(default_factory=dict)
  cam: dict = dataclasses.field(default_factory=dict)


TASKS = {
  "crate_climb": Task(
    "crate_climb", "scene_crate.xml", "reach", T=280, steps=120, lr=0.03, lane=1.6, tbptt_h=32, title="crate",
    # TRUNCATED BPTT (32-frame chunks) so the long contact-rich climb doesn't collapse into the rear/flip
    # local optimum of full-T single-shooting. dial-mpc crate reward: trunk -> far crate-top target (xy fwd +
    # z up) + moderate upright (some pitch to climb, but enough to not flip) + airtime lift + FEET-ON-CRATE-TOP
    # (all four feet mount the 0.30 step top, end-weighted) so the body climbs fully ON, not just paws the face.
    w={"xy": 0.8, "z": 2.0, "up": 0.3, "feet": 2.0, "ctrl": 2e-3, "lift": 0.8},
    cam={"lookat": [0.6, 0.0, 0.3], "distance": 3.1, "azimuth": 60.0, "elevation": -14.0},
  ),
  "precision_jump": Task(
    "precision_jump", "scene_stones.xml", "stones", T=300, steps=120, lr=0.02, n_hops=2, lane=1.2, spread=0.0,
    tbptt_h=32, grad_clip=15.0, title="jumps",
    # PRONK (all 4 feet jump together) with dial-mpc COST TERMS: WITHIN-RADIUS foot placement (0 loss once the
    # foot is over the pillar top, forgiving) + UP-VECTOR upright + COM waypoint tracking (reward_pos) + yaw->0
    # (reward_yaw) + ride-height/anti-fall + z arc (lift over gap) + NO-SLIP + LEFT-RIGHT SYMMETRY (straight
    # pronk, left & right feet land TOGETHER). NO Raibert (its velocity-forward foot reach flattens the jump).
    # dt stays 0.004: bigger dt makes contact BPTT gradients too stiff -> diverges (0.006 -> |g| 650).
    # JUMP-DRIVING weights (NOT dial-mpc's alive-dominant proportions -- those are a per-step SURVIVAL reward
    # for RL/MPC; in fixed-horizon trajopt alive*10 just makes "stay upright & don't fall" the low-loss optimum
    # -> the robot barely moves). Forward COM and landing terms dominate here; the one-sided floor term makes
    # a foot in a gap much more expensive than a small arc-tracking error, while alive stays only anti-fall.
    w={"foot": 50.0, "footz": 60.0, "land": 250.0, "floor": 3000.0, "sync": 1000.0,
       "worst": 1000.0, "over": 3000.0, "brake": 30.0, "miss": 6000.0,
       "clear": 20.0, "land_pose": 30.0, "finish": 2500.0, "state_sym": 40.0,
       "com": 300.0, "vel": 8.0, "up": 400.0, "z": 80.0, "omega": 8.0,
       "slip": 3.0, "yaw": 8.0, "sym": 0.5,
       "alive": 3.0, "ctrl": 1e-3},
    cam={"lookat": [0.5, 0.0, 0.15], "distance": 3.2, "azimuth": 60.0, "elevation": -14.0},
  ),
  "gallop": Task(
    "gallop", "scene_flat.xml", "gallop", T=300, steps=120, lr=0.03, vel_tar=1.2, height_tar=0.30, lane=0.9, spread=0.06,
    # forward speed + ride height (loose) + upright + foot-swing gait + no-slip (plant, don't skate)
    # + Raibert placement (swing foot reaches to a velocity-appropriate foothold -> propulsive stride)
    w={"vel": 2.0, "z": 0.5, "up": 0.4, "gait": 40.0, "slip": 3.0, "raibert": 8.0, "ctrl": 2e-3},
    # front 3/4 view (azimuth 135): heads face the camera AND the y-lane envs spread across the frame (no occlusion)
    cam={"track": True, "z": 0.3, "azimuth": 135.0, "elevation": -18.0, "dist0": 2.0, "distk": 0.75},
  ),
  "handstand": Task(
    "handstand", "scene_flat.xml", "handstand", T=250, steps=120, lr=0.03, height_tar=0.25, lane=0.9,
    w={"ori": 2.0, "head": 8.0, "bal": 0.02, "ctrl": 2e-3},  # pitch to +90deg AND hold the head up (front legs support)
    # Front-three-quarter view keeps all four y-lanes visible (90 degrees looked down-lane and occluded them).
    cam={"lookat": [0.1, 0.0, 0.35], "distance": 2.3, "azimuth": 135.0, "elevation": -8.0},
  ),
}


# ---------------- batched loss kernels (dim = nworld) ----------------
# trunk free joint: qpos[w,0:3]=pos, qpos[w,3:7]=quat(w,x,y,z); qx,qy ~ roll,pitch; qvel[w,0:3]=world lin vel.

@wp.kernel
def _reach_loss(qpos: wp.array2d[float], tx: float, ty: float, tz: float, ws: float, z0: float,
                w_xy: float, w_z: float, w_up: float, w_lift: float, loss: wp.array(dtype=float)):
  w = wp.tid()
  x, y, z, qx, qy = qpos[w, 0], qpos[w, 1], qpos[w, 2], qpos[w, 4], qpos[w, 5]
  wp.atomic_add(loss, 0, ws * (w_xy * ((x - tx) * (x - tx) + (y - ty) * (y - ty))
                               + w_z * (z - tz) * (z - tz) + w_up * (qx * qx + qy * qy))
                - w_lift * (z - z0))  # w_lift gated to the flight window in python


@wp.func
def _on_crate(p: wp.vec3, cx_lo: float, cx_hi: float, cy_lo: float, cy_hi: float, ctop: float) -> float:
  # penalty for a foot NOT being on the crate TOP: (z - ctop)^2 to sit at top height + squared xy distance
  # OUTSIDE the crate footprint (0 inside). -> 0 when the foot rests within the top face, grows off it.
  dz = p[2] - ctop
  ox = wp.max(cx_lo - p[0], 0.0) + wp.max(p[0] - cx_hi, 0.0)
  oy = wp.max(cy_lo - p[1], 0.0) + wp.max(p[1] - cy_hi, 0.0)
  return dz * dz + ox * ox + oy * oy


@wp.kernel
def _crate_feet_loss(site_xpos: wp.array2d[wp.vec3], fr: int, fl: int, rr: int, rl: int,
                     cx_lo: float, cx_hi: float, cy_lo: float, cy_hi: float, ctop: float,
                     ws: float, w_feet: float, loss: wp.array(dtype=float)):
  # dial-mpc FEET-ON-CRATE: reward all 4 feet resting on the crate TOP face (within the footprint, at top
  # height). End-weighted (ws) so the feet climb up onto the crate by the end -> the whole body mounts it,
  # not just the front paws pawing the face. Uses the differentiable fwd_kinematics foot site_xpos gradient.
  w = wp.tid()
  s = _on_crate(site_xpos[w, fr], cx_lo, cx_hi, cy_lo, cy_hi, ctop)
  s += _on_crate(site_xpos[w, fl], cx_lo, cx_hi, cy_lo, cy_hi, ctop)
  s += _on_crate(site_xpos[w, rr], cx_lo, cx_hi, cy_lo, cy_hi, ctop)
  s += _on_crate(site_xpos[w, rl], cx_lo, cx_hi, cy_lo, cy_hi, ctop)
  wp.atomic_add(loss, 0, ws * w_feet * s)


@wp.kernel
def _seq_loss(qpos: wp.array2d[float], wx: float, wy: float, wz: float,
              w_xy: float, w_z: float, w_up: float, loss: wp.array(dtype=float)):
  w = wp.tid()
  x, y, z, qx, qy = qpos[w, 0], qpos[w, 1], qpos[w, 2], qpos[w, 4], qpos[w, 5]
  wp.atomic_add(loss, 0, w_xy * ((x - wx) * (x - wx) + (y - wy) * (y - wy))
                + w_z * (z - wz) * (z - wz) + w_up * (qx * qx + qy * qy))


@wp.func
def _foot_slip(cur: wp.vec3, prev: wp.vec3, inv_dt: float) -> float:
  # squared horizontal foot speed (world xy) between consecutive steps = slip velocity when planted
  vx = (cur[0] - prev[0]) * inv_dt
  vy = (cur[1] - prev[1]) * inv_dt
  return vx * vx + vy * vy


@wp.kernel
def _gallop_loss(qpos: wp.array2d[float], qvel: wp.array2d[float],
                 site_xpos: wp.array2d[wp.vec3], prev_xpos: wp.array2d[wp.vec3],
                 fr: int, fl: int, rr: int, rl: int,
                 tzfr: float, tzfl: float, tzrr: float, tzrl: float,
                 sfr: float, sfl: float, srr: float, srl: float,
                 vtar: float, ztar: float, inv_dt: float,
                 w_vel: float, w_z: float, w_up: float, w_gait: float, w_slip: float,
                 loss: wp.array(dtype=float)):
  # gallop = forward speed + ride height + upright + dial-mpc GAIT (foot z tracks a swing-height reference)
  # + NO-SLIP: penalize each foot's horizontal speed WHILE IT IS IN STANCE (stance weight s* -> 1 when the
  # foot should be planted). Without this, open-loop BPTT skates the feet (esp. the hind) to fake forward
  # speed instead of planting and pushing off. Foot-position gradients are ANALYTIC (fwd_kinematics VJP), no FD.
  w = wp.tid()
  vx, z, qx, qy = qvel[w, 0], qpos[w, 2], qpos[w, 4], qpos[w, 5]
  gait = float(0.0)
  d = site_xpos[w, fr][2] - tzfr; gait += d * d
  d = site_xpos[w, fl][2] - tzfl; gait += d * d
  d = site_xpos[w, rr][2] - tzrr; gait += d * d
  d = site_xpos[w, rl][2] - tzrl; gait += d * d
  slip = sfr * _foot_slip(site_xpos[w, fr], prev_xpos[w, fr], inv_dt)
  slip += sfl * _foot_slip(site_xpos[w, fl], prev_xpos[w, fl], inv_dt)
  slip += srr * _foot_slip(site_xpos[w, rr], prev_xpos[w, rr], inv_dt)
  slip += srl * _foot_slip(site_xpos[w, rl], prev_xpos[w, rl], inv_dt)
  wp.atomic_add(loss, 0, w_vel * (vx - vtar) * (vx - vtar) + w_z * (z - ztar) * (z - ztar)
                + w_up * (qx * qx + qy * qy) + w_gait * gait + w_slip * slip)


@wp.kernel
def _handstand_loss(qpos: wp.array2d[float], qvel: wp.array2d[float], htar: float, ws: float,
                    w_ori: float, w_head: float, w_bal: float, loss: wp.array(dtype=float)):
  # REAL handstand on the FRONT legs: (1) pitch the trunk to +90deg (quat -> [.707,0,.707,0]: head/
  # forward points DOWN, rear lifts UP), AND (2) keep the HEAD held UP off the ground (head_z >= htar).
  # A face-plant/headstand also pitches forward but drops the head to the floor; requiring the head to
  # stay up forces the FRONT legs to EXTEND and support the body -> a true handstand, not a head-on-ground
  # pivot. The head site is 0.3 m forward of the trunk origin, so head_z = trunk_z + 0.3*forward_z with
  # forward_z = 2(xz - wy) (= -1 when the head points straight down). All end-weighted (ws=(t+1)/T).
  w = wp.tid()
  qw, qx, qy, qz = qpos[w, 3], qpos[w, 4], qpos[w, 5], qpos[w, 6]
  al = 0.70710678 * (qw + qy)
  ori = 1.0 - al * al
  fz = 2.0 * (qx * qz - qw * qy)
  head_z = qpos[w, 2] + 0.3 * fz
  below = wp.max(htar - head_z, 0.0)                # penalize the head being BELOW htar (near the floor)
  ang = qvel[w, 3] * qvel[w, 3] + qvel[w, 4] * qvel[w, 4] + qvel[w, 5] * qvel[w, 5]
  wp.atomic_add(loss, 0, ws * (w_ori * ori + w_head * below * below + w_bal * ang))


@wp.kernel
def _raibert_loss(site_xpos: wp.array2d[wp.vec3], qpos: wp.array2d[float], qvel: wp.array2d[float],
                  fr: int, fl: int, rr: int, rl: int,
                  ox_fr: float, ox_fl: float, ox_rr: float, ox_rl: float,   # rest foot x-offset from trunk
                  rw_fr: float, rw_fl: float, rw_rr: float, rw_rl: float,   # per-foot reach weight (late swing)
                  c_reach: float, k: float, v_des: float, w_raibert: float, loss: wp.array(dtype=float)):
  # RAIBERT foothold placement: while a foot is in LATE SWING (approaching touchdown, rw>0), reward its
  # fore-aft position RELATIVE TO THE TRUNK to reach a velocity-appropriate landing point
  #   x*_rel = hip_offset + c_reach*v + k*(v - v_des)
  # so the foot lands AHEAD in proportion to speed -> the body then drives forward over the planted foot
  # (a propulsive stride), instead of the short "plant in place" step. Analytic (site_xpos VJP + qpos/qvel).
  w = wp.tid()
  tx, vx = qpos[w, 0], qvel[w, 0]
  reach = c_reach * vx + k * (vx - v_des)
  r = float(0.0)
  d = (site_xpos[w, fr][0] - tx) - (ox_fr + reach); r += rw_fr * d * d
  d = (site_xpos[w, fl][0] - tx) - (ox_fl + reach); r += rw_fl * d * d
  d = (site_xpos[w, rr][0] - tx) - (ox_rr + reach); r += rw_rr * d * d
  d = (site_xpos[w, rl][0] - tx) - (ox_rl + reach); r += rw_rl * d * d
  wp.atomic_add(loss, 0, w_raibert * r)


@wp.kernel
def _stones_loss(site_xpos: wp.array2d[wp.vec3], prev_xpos: wp.array2d[wp.vec3],
                 qpos: wp.array2d[float], qvel: wp.array2d[float],
                 fr: int, fl: int, rr: int, rl: int,
                 tfr: wp.vec3, tfl: wp.vec3, trr: wp.vec3, trl: wp.vec3,
                 cx: float, cy: float, cz: float, cvx: float, cvy: float, cvz: float,
                 sfr: float, sfl: float, srr: float, srl: float, inv_dt: float, r_pillar: float,
                 land_gate: float, clear_gate: float, finish_gate: float, pose_gate: float, land_z: float,
                 front_thigh0: float, front_calf0: float, rear_thigh0: float, rear_calf0: float,
                 w_foot: float, w_footz: float, w_land: float, w_floor: float, w_sync: float,
                 w_worst: float, w_over: float, w_brake: float, w_miss: float,
                 w_clear: float, w_land_pose: float, w_finish: float, w_state_sym: float,
                 w_com: float, w_vel: float, w_up: float, w_z: float, w_omega: float,
                 w_slip: float, w_yaw: float,
                 z_min: float, w_alive: float, loss: wp.array(dtype=float)):
  # AIRBORNE PRONK loss, dial-mpc-style cost terms. Each foot SITE tracks a per-step reference that ARCS UP
  # over the gap and lands on the next pillar. FOOT XY = dial-mpc WITHIN-RADIUS: 0 loss once the foot is over
  # the pillar top (xy within r_pillar of the target), restoring pull only when OFF -> forgiving (no yanking
  # a foot that's already on the pillar) vs a harsh squared pull. z arc tracked normally. TRUNK: COM waypoint
  # (dial-mpc reward_pos) + ride height + UP-VECTOR upright (dial-mpc reward_upright, proper large-tilt form)
  # + yaw->0 (reward_yaw, keep heading straight). NO-SLIP holds a planted foot. All ANALYTIC (site VJP + qpos).
  w = wp.tid()
  r2 = r_pillar * r_pillar
  fp = float(0.0)   # sum of foot xy WITHIN-RADIUS placement penalties (0 when on the pillar top)
  fxy = float(0.0)  # exact moving-reference tracking during flight (needed in every detached chunk)
  fz = float(0.0)   # sum of foot z (arc / land-on-top) errors
  land = float(0.0) # exact-center xy during descent/touchdown
  floor = float(0.0) # one-sided penalty when a foot drops below pillar-top site height
  pfr, pfl, prr, prl = site_xpos[w, fr], site_xpos[w, fl], site_xpos[w, rr], site_xpos[w, rl]
  e = pfr - tfr; d2fr = e[0] * e[0] + e[1] * e[1]; fxy += d2fr; fp += wp.max(d2fr - r2, 0.0); fz += e[2] * e[2]
  land += d2fr
  floor += wp.max(land_z - pfr[2], 0.0) * wp.max(land_z - pfr[2], 0.0)
  e = pfl - tfl; d2fl = e[0] * e[0] + e[1] * e[1]; fxy += d2fl; fp += wp.max(d2fl - r2, 0.0); fz += e[2] * e[2]
  land += d2fl
  floor += wp.max(land_z - pfl[2], 0.0) * wp.max(land_z - pfl[2], 0.0)
  e = prr - trr; d2rr = e[0] * e[0] + e[1] * e[1]; fxy += d2rr; fp += wp.max(d2rr - r2, 0.0); fz += e[2] * e[2]
  land += d2rr
  floor += wp.max(land_z - prr[2], 0.0) * wp.max(land_z - prr[2], 0.0)
  e = prl - trl; d2rl = e[0] * e[0] + e[1] * e[1]; fxy += d2rl; fp += wp.max(d2rl - r2, 0.0); fz += e[2] * e[2]
  land += d2rl
  floor += wp.max(land_z - prl[2], 0.0) * wp.max(land_z - prl[2], 0.0)
  zmean = 0.25 * (pfr[2] + pfl[2] + prr[2] + prl[2])
  sync = (pfr[2] - zmean) * (pfr[2] - zmean) + (pfl[2] - zmean) * (pfl[2] - zmean)
  sync += (prr[2] - zmean) * (prr[2] - zmean) + (prl[2] - zmean) * (prl[2] - zmean)
  slip = sfr * _foot_slip(site_xpos[w, fr], prev_xpos[w, fr], inv_dt)
  slip += sfl * _foot_slip(site_xpos[w, fl], prev_xpos[w, fl], inv_dt)
  slip += srr * _foot_slip(site_xpos[w, rr], prev_xpos[w, rr], inv_dt)
  slip += srl * _foot_slip(site_xpos[w, rl], prev_xpos[w, rl], inv_dt)
  x, y, z = qpos[w, 0], qpos[w, 1], qpos[w, 2]
  qw, qx, qy, qz = qpos[w, 3], qpos[w, 4], qpos[w, 5], qpos[w, 6]
  ux = 2.0 * (qx * qz + qw * qy)             # trunk up-vector = R(quat)*[0,0,1]
  uy = 2.0 * (qy * qz - qw * qx)
  uzm1 = -2.0 * (qx * qx + qy * qy)          # up_z - 1
  upright = ux * ux + uy * uy + uzm1 * uzm1  # ||up - [0,0,1]||^2  (0 = perfectly upright)
  yaw = 2.0 * (qw * qz + qx * qy)            # small-angle yaw (heading); target 0 = straight ahead
  com = w_com * ((x - cx) * (x - cx) + (y - cy) * (y - cy)) + w_z * (z - cz) * (z - cz)
  vel = (qvel[w, 0] - cvx) * (qvel[w, 0] - cvx) + (qvel[w, 1] - cvy) * (qvel[w, 1] - cvy)
  vel += (qvel[w, 2] - cvz) * (qvel[w, 2] - cvz)
  brake = qvel[w, 0] * qvel[w, 0] + qvel[w, 1] * qvel[w, 1] + qvel[w, 2] * qvel[w, 2]
  omega = qvel[w, 3] * qvel[w, 3] + qvel[w, 4] * qvel[w, 4] + qvel[w, 5] * qvel[w, 5]
  worst = wp.max(wp.max(d2fr, d2fl), wp.max(d2rr, d2rl))
  over = wp.max(x - cx - 0.03, 0.0)
  finish_err = x - cx
  # A foot at z=land_z can be resting on ANY cylinder. If it is outside the assigned pillar's safe radius,
  # keep it above the intervening tops until its xy is correct; only then is descent/landing unpenalized.
  clear_z = land_z + 0.055
  miss = float(0.0)
  off = wp.min(wp.max((d2fr - r2) / r2, 0.0), 1.0); low = wp.max(clear_z - pfr[2], 0.0); miss += off * low * low
  off = wp.min(wp.max((d2fl - r2) / r2, 0.0), 1.0); low = wp.max(clear_z - pfl[2], 0.0); miss += off * low * low
  off = wp.min(wp.max((d2rr - r2) / r2, 0.0), 1.0); low = wp.max(clear_z - prr[2], 0.0); miss += off * low * low
  off = wp.min(wp.max((d2rl - r2) / r2, 0.0), 1.0); low = wp.max(clear_z - prl[2], 0.0); miss += off * low * low
  # Pure-qpos rear-leg clearance: the FK adjoint is verified for foot sites, while geom_xpos is not an
  # adjoint output. These one-sided barriers keep the second-flight thighs/knees high and calves flexed.
  rear_clear = wp.max(0.75 - qpos[w, 14], 0.0) * wp.max(0.75 - qpos[w, 14], 0.0)
  rear_clear += wp.max(0.75 - qpos[w, 17], 0.0) * wp.max(0.75 - qpos[w, 17], 0.0)
  rear_clear += 0.5 * wp.max(qpos[w, 15] + 2.05, 0.0) * wp.max(qpos[w, 15] + 2.05, 0.0)
  rear_clear += 0.5 * wp.max(qpos[w, 18] + 2.05, 0.0) * wp.max(qpos[w, 18] + 2.05, 0.0)
  # Controls are projected to exact L/R symmetry, but stiff contact ordering can still split the measured
  # states. Penalize that split directly so a single foot in a gap cannot be traded against its good twin.
  state_sym = (qpos[w, 7] + qpos[w, 10]) * (qpos[w, 7] + qpos[w, 10])
  state_sym += (qpos[w, 8] - qpos[w, 11]) * (qpos[w, 8] - qpos[w, 11])
  state_sym += (qpos[w, 9] - qpos[w, 12]) * (qpos[w, 9] - qpos[w, 12])
  state_sym += (qpos[w, 13] + qpos[w, 16]) * (qpos[w, 13] + qpos[w, 16])
  state_sym += (qpos[w, 14] - qpos[w, 17]) * (qpos[w, 14] - qpos[w, 17])
  state_sym += (qpos[w, 15] - qpos[w, 18]) * (qpos[w, 15] - qpos[w, 18])
  land_pose = (qpos[w, 8] - front_thigh0) * (qpos[w, 8] - front_thigh0)
  land_pose += (qpos[w, 11] - front_thigh0) * (qpos[w, 11] - front_thigh0)
  land_pose += 0.5 * (qpos[w, 9] - front_calf0) * (qpos[w, 9] - front_calf0)
  land_pose += 0.5 * (qpos[w, 12] - front_calf0) * (qpos[w, 12] - front_calf0)
  land_pose += (qpos[w, 14] - rear_thigh0) * (qpos[w, 14] - rear_thigh0)
  land_pose += (qpos[w, 17] - rear_thigh0) * (qpos[w, 17] - rear_thigh0)
  land_pose += 0.5 * (qpos[w, 15] - rear_calf0) * (qpos[w, 15] - rear_calf0)
  land_pose += 0.5 * (qpos[w, 18] - rear_calf0) * (qpos[w, 18] - rear_calf0)
  alive = wp.max(z_min - z, 0.0)             # dial-mpc ALIVE (anti-fall): 0 while upright, penalize trunk COLLAPSING below z_min
  wp.atomic_add(loss, 0, w_foot * ((1.0 - land_gate) * fxy + land_gate * fp) + w_footz * fz
                + land_gate * (w_land * land + w_floor * floor + w_sync * sync
                               + w_worst * worst + w_over * over * over)
                + land_gate * w_miss * miss
                + clear_gate * w_clear * rear_clear + pose_gate * w_land_pose * land_pose
                + finish_gate * w_finish * finish_err * finish_err
                + pose_gate * (w_brake * brake + w_omega * omega)
                + w_state_sym * state_sym
                + com + w_vel * vel + w_up * upright + w_yaw * yaw * yaw
                + w_slip * slip + w_alive * alive * alive)


@wp.kernel
def _ctrl_cost(ctrl: wp.array2d[float], home: wp.array(dtype=float), w: float, loss: wp.array(dtype=float)):
  i, j = wp.tid()
  d = ctrl[i, j] - home[j]
  wp.atomic_add(loss, 0, w * d * d)


@wp.kernel
def _sym_cost(ctrl: wp.array2d[float], w: float, loss: wp.array(dtype=float)):
  # LEFT-RIGHT symmetry of the Go1 mjlab control (order FR_hip,FR_thigh,FL_hip,FL_thigh, RR_hip,RR_thigh,
  # RL_hip,RL_thigh, FR_calf,FL_calf,RR_calf,RL_calf). Hips MIRROR (opposite sign), thigh/calf MATCH. A
  # symmetric control makes the robot push off and land its LEFT & RIGHT feet TOGETHER -> a straight pronk
  # (not a lopsided scramble). Front/rear keep their own fore-aft roles; only the L/R mirror is enforced.
  i = wp.tid()
  s = float(0.0)
  d = ctrl[i, 0] + ctrl[i, 2]; s += d * d      # FR_hip = -FL_hip
  d = ctrl[i, 1] - ctrl[i, 3]; s += d * d      # FR_thigh = FL_thigh
  d = ctrl[i, 4] + ctrl[i, 6]; s += d * d      # RR_hip = -RL_hip
  d = ctrl[i, 5] - ctrl[i, 7]; s += d * d      # RR_thigh = RL_thigh
  d = ctrl[i, 8] - ctrl[i, 9]; s += d * d      # FR_calf = FL_calf
  d = ctrl[i, 10] - ctrl[i, 11]; s += d * d    # RR_calf = RL_calf
  wp.atomic_add(loss, 0, w * s)


@wp.kernel
def _clamp(a: wp.array(dtype=float), lo: wp.array(dtype=float), hi: wp.array(dtype=float), nu: int):
  # project a 1D (N*nu) control-param array back into [lo, hi] per actuator (j = i % nu) after each Adam step
  i = wp.tid()
  a[i] = wp.clamp(a[i], lo[i % nu], hi[i % nu])


@wp.kernel
def _scale(a: wp.array(dtype=float), s: float):
  a[wp.tid()] = a[wp.tid()] * s      # in-place gradient scaling for grad-norm clipping


@wp.kernel
def _project_pronk(a: wp.array(dtype=float), nu: int):
  """Restrict stones controls to a physical left/right-mirrored pronk (ordinary actuator targets only)."""
  i = wp.tid()
  b = i * nu
  hip = 0.5 * (a[b] - a[b + 2]); a[b] = hip; a[b + 2] = -hip
  thigh = 0.5 * (a[b + 1] + a[b + 3]); a[b + 1] = thigh; a[b + 3] = thigh
  hip = 0.5 * (a[b + 4] - a[b + 6]); a[b + 4] = hip; a[b + 6] = -hip
  thigh = 0.5 * (a[b + 5] + a[b + 7]); a[b + 5] = thigh; a[b + 7] = thigh
  calf = 0.5 * (a[b + 8] + a[b + 9]); a[b + 8] = calf; a[b + 9] = calf
  calf = 0.5 * (a[b + 10] + a[b + 11]); a[b + 10] = calf; a[b + 11] = calf


def _active_wp(task, t, params):
  """precision_jump: which forward waypoint is active at step t (advance evenly over the horizon)."""
  wps = params["waypoints"]
  return wps[min(t * len(wps) // task.T, len(wps) - 1)]


def _launch_loss(task, d1, d0, ctrl, home_wp, t, N, params, loss):
  o, w = task.objective, task.w  # d0 = previous step's data (used only for the gallop no-slip term)
  if o == "reach":
    ws = float((t + 1) / task.T)                                     # end-weight: wind up early, be on the crate late
    lift = w.get("lift", 0.0) if (0.3 <= (t + 1) / task.T <= 0.75) else 0.0  # airtime reward in the flight window
    tx, ty, tz = params["target"]
    wp.launch(_reach_loss, dim=N, inputs=[d1.qpos, float(tx), float(ty), float(tz), ws, params["z0"],
                                          w["xy"], w["z"], w["up"], float(lift)], outputs=[loss])
    if w.get("feet", 0.0) > 0.0:                                     # dial-mpc feet-on-crate-top (end-weighted)
      fr, fl, rr, rl = params["foot_ids"]
      cxl, cxh, cyl, cyh, ct = params["crate"]
      wp.launch(_crate_feet_loss, dim=N, inputs=[d1.site_xpos, int(fr), int(fl), int(rr), int(rl),
                                                 cxl, cxh, cyl, cyh, ct, ws, w["feet"]], outputs=[loss])
  elif o == "sequence":
    wx, wy, wz = _active_wp(task, t, params)
    wp.launch(_seq_loss, dim=N, inputs=[d1.qpos, float(wx), float(wy), float(wz),
                                        w["xy"], w["z"], w["up"]], outputs=[loss])
  elif o == "gallop":
    vtar = task.vel_tar * min(1.0, (t + 1) / (0.3 * task.T))          # ramp the speed target up
    hgt = _foot_step_heights((t + 1) * task.dt)                       # per-foot swing height (0..amp)
    h = params["foot_rest_z"] + hgt                                   # -> world-z targets
    sw = 1.0 - np.minimum(1.0, hgt / GALLOP_AMP)                      # stance weight: 1 planted, 0 mid-swing
    fr, fl, rr, rl = params["foot_ids"]
    wp.launch(_gallop_loss, dim=N, inputs=[d1.qpos, d1.qvel, d1.site_xpos, d0.site_xpos,
                                           int(fr), int(fl), int(rr), int(rl),
                                           float(h[0]), float(h[1]), float(h[2]), float(h[3]),
                                           float(sw[0]), float(sw[1]), float(sw[2]), float(sw[3]),
                                           float(vtar), task.height_tar, float(1.0 / task.dt),
                                           w["vel"], w["z"], w["up"], w["gait"], w["slip"]], outputs=[loss])
    ox, rw = params["raibert_ox"], params["reach_w"][t]                 # Raibert foothold placement (late swing)
    wp.launch(_raibert_loss, dim=N, inputs=[d1.site_xpos, d1.qpos, d1.qvel, int(fr), int(fl), int(rr), int(rl),
                                            float(ox[0]), float(ox[1]), float(ox[2]), float(ox[3]),
                                            float(rw[0]), float(rw[1]), float(rw[2]), float(rw[3]),
                                            RAIBERT_REACH, RAIBERT_K, float(vtar), w["raibert"]], outputs=[loss])
  elif o == "handstand":
    ws = float((t + 1) / task.T)                                      # end-weight: wind up early, handstand late
    wp.launch(_handstand_loss, dim=N, inputs=[d1.qpos, d1.qvel, task.height_tar, ws,
                                              w["ori"], w["head"], w["bal"]], outputs=[loss])
  elif o == "stones":
    fr, fl, rr, rl = params["foot_ids"]
    ft = params["foot_ref"][t]                                        # (4,3) per-step arc/placement foot refs
    cr = params["com_ref"][t]                                         # (3,) per-step trunk COM waypoint
    cv = params["com_vel_ref"][t]                                     # (3,) ballistic trunk velocity
    sr = params["stance_ref"][t]                                      # (4,) per-step no-slip stance weights
    lg = params["land_ref"][t]                                        # descent/touchdown exact placement gate
    cg = params["clear_ref"][t]                                       # second-flight rear-knee clearance gate
    fg = params["finish_ref"][t]                                      # second-hop forward-completion gate
    pg = params["pose_ref"][t]                                        # landing pose, disabled during the next load/push
    wp.launch(_stones_loss, dim=N, inputs=[d1.site_xpos, d0.site_xpos, d1.qpos, d1.qvel,
                                           int(fr), int(fl), int(rr), int(rl),
                                           wp.vec3(*ft[0]), wp.vec3(*ft[1]), wp.vec3(*ft[2]), wp.vec3(*ft[3]),
                                           float(cr[0]), float(cr[1]), float(cr[2]),
                                           float(cv[0]), float(cv[1]), float(cv[2]),
                                           float(sr[0]), float(sr[1]), float(sr[2]), float(sr[3]), float(1.0 / task.dt),
                                           params["r_pillar"], float(lg), float(cg), float(fg), float(pg), params["stone_z"],
                                           params["front_thigh0"], params["front_calf0"],
                                           params["rear_thigh0"], params["rear_calf0"],
                                           w["foot"], w["footz"], w["land"], w["floor"], w["sync"],
                                           w["worst"], w["over"], w["brake"], w["miss"],
                                           w["clear"], w["land_pose"], w["finish"], w["state_sym"],
                                           w["com"], w["vel"], w["up"], w["z"], w["omega"],
                                           w["slip"], w.get("yaw", 0.0),
                                           float(params["z0"] - 0.15), w.get("alive", 0.0)], outputs=[loss])
    if w.get("raibert", 0.0) > 0.0:                                   # RAIBERT foothold reach (reuses the gallop kernel):
      ox, rw = params["raib_ox"][t], params["raib_w"][t]              # swing foot reaches its landing stone velocity-appropriately
      wp.launch(_raibert_loss, dim=N, inputs=[d1.site_xpos, d1.qpos, d1.qvel, int(fr), int(fl), int(rr), int(rl),
                                              float(ox[0]), float(ox[1]), float(ox[2]), float(ox[3]),
                                              float(rw[0]), float(rw[1]), float(rw[2]), float(rw[3]),
                                              RAIBERT_REACH, RAIBERT_K, params["v_des"], w["raibert"]], outputs=[loss])
  if w.get("ctrl", 0.0) > 0.0:
    wp.launch(_ctrl_cost, dim=(N, ctrl.shape[1]), inputs=[ctrl, home_wp, w["ctrl"]], outputs=[loss])
  if w.get("sym", 0.0) > 0.0:                        # LEFT-RIGHT symmetry -> straight pronk, feet land together
    wp.launch(_sym_cost, dim=N, inputs=[ctrl, w["sym"]], outputs=[loss])


# ---------------- per-env loss (numpy; for coloring / best-env / reporting) ----------------
def per_env_loss(task, qtraj, params, foot_traj=None, qveltraj=None):
  """qtraj: (N,T+1,nq) [+ foot_traj (N,T+1,4,3) for stones] -> (N,) loss per env, mirroring the kernels."""
  o, w, T = task.objective, task.w, task.T
  q = qtraj[:, 1:, :]                                     # states after each step (t=1..T)
  x, y, z, qx, qy = q[..., 0], q[..., 1], q[..., 2], q[..., 4], q[..., 5]
  up = w.get("up", 0.0) * (qx ** 2 + qy ** 2)
  if o == "reach":
    ws = (np.arange(1, T + 1) / T)[None, :]
    tx, ty, tz = params["target"]
    lift_w = np.where((np.arange(1, T + 1) / T >= 0.3) & (np.arange(1, T + 1) / T <= 0.75), w.get("lift", 0.0), 0.0)[None, :]
    per = ws * (w["xy"] * ((x - tx) ** 2 + (y - ty) ** 2) + w["z"] * (z - tz) ** 2 + up) - lift_w * (z - params["z0"])
    if w.get("feet", 0.0) > 0.0:                          # dial-mpc feet-on-crate-top (mirrors _crate_feet_loss)
      cxl, cxh, cyl, cyh, ct = params["crate"]
      fp = foot_traj[:, 1:, :, :]                          # (N,T,4,3)
      dz = (fp[..., 2] - ct) ** 2
      ox = np.maximum(cxl - fp[..., 0], 0.0) + np.maximum(fp[..., 0] - cxh, 0.0)
      oy = np.maximum(cyl - fp[..., 1], 0.0) + np.maximum(fp[..., 1] - cyh, 0.0)
      per = per + ws[0] * w["feet"] * (dz + ox ** 2 + oy ** 2).sum(-1)
  elif o == "sequence":
    wps = params["waypoints"]
    idx = np.minimum(np.arange(T) * len(wps) // T, len(wps) - 1)
    wx, wy, wz = wps[idx, 0][None, :], wps[idx, 1][None, :], wps[idx, 2][None, :]
    per = w["xy"] * ((x - wx) ** 2 + (y - wy) ** 2) + w["z"] * (z - wz) ** 2 + up
  elif o == "gallop":  # forward speed (FD of trunk x path -- display only) + ride height + upright + gait + no-slip
    vx = np.gradient(qtraj[:, :, 0], task.dt, axis=1)[:, 1:]
    vtar = task.vel_tar * np.minimum(1.0, np.arange(1, T + 1) / (0.3 * T))[None, :]
    hgt = np.array([_foot_step_heights((t + 1) * task.dt) for t in range(T)])                       # (T,4)
    gait = ((foot_traj[:, 1:, :, 2] - (params["foot_rest_z"] + hgt)[None, :, :]) ** 2).sum(-1)      # (N,T)
    sw = 1.0 - np.minimum(1.0, hgt / GALLOP_AMP)                                                    # (T,4) stance wt
    fvel = (foot_traj[:, 1:, :, :2] - foot_traj[:, :-1, :, :2]) / task.dt                           # (N,T,4,2)
    slip = (sw[None, :, :] * (fvel ** 2).sum(-1)).sum(-1)                                           # (N,T)
    ox, rw = params["raibert_ox"], params["reach_w"]                                                # (4,), (T,4)
    foot_relx = foot_traj[:, 1:, :, 0] - qtraj[:, 1:, 0:1]                                          # (N,T,4)
    reach = RAIBERT_REACH * vx + RAIBERT_K * (vx - vtar)                                            # (N,T)
    raib = (rw[None, :, :] * (foot_relx - (ox[None, None, :] + reach[:, :, None])) ** 2).sum(-1)    # (N,T)
    per = (w["vel"] * (vx - vtar) ** 2 + w["z"] * (z - task.height_tar) ** 2 + up
           + w["gait"] * gait + w["slip"] * slip + w["raibert"] * raib)
  elif o == "stones":  # foot-on-stone STEPPING (mirrors _stones_loss): arc/placement + COM-forward + upright + no-slip
    fref, cref, sref = params["foot_ref"], params["com_ref"], params["stance_ref"]   # (T,4,3),(T,3),(T,4)
    cvref = params["com_vel_ref"]                                                     # (T,3)
    lref = params["land_ref"]                                                        # (T,) descent/touchdown gate
    clear_ref, finish_ref, pose_ref = params["clear_ref"], params["finish_ref"], params["pose_ref"]
    fp = foot_traj[:, 1:, :, :]                           # (N,T,4,3) foot sites after each step
    dxy2 = ((fp[..., :2] - fref[None, :, :, :2]) ** 2).sum(-1)                        # (N,T,4) squared xy dist
    exy = np.maximum(dxy2 - params["r_pillar"] ** 2, 0.0)                             # WITHIN-RADIUS (0 on the pillar top)
    ez = (fp[..., 2] - fref[None, :, :, 2]) ** 2                                      # (N,T,4)
    fvel = (foot_traj[:, 1:, :, :2] - foot_traj[:, :-1, :, :2]) / task.dt             # (N,T,4,2) foot horiz vel
    slip = (sref[None, :, :] * (fvel ** 2).sum(-1)).sum(-1)                           # (N,T)
    qw_, qx_, qy_, qz_ = q[..., 3], q[..., 4], q[..., 5], q[..., 6]                   # up-vector upright + yaw (mirror kernel)
    ux, uy, uzm1 = 2 * (qx_ * qz_ + qw_ * qy_), 2 * (qy_ * qz_ - qw_ * qx_), -2 * (qx_ ** 2 + qy_ ** 2)
    upright = ux ** 2 + uy ** 2 + uzm1 ** 2
    yaw = 2 * (qw_ * qz_ + qx_ * qy_)
    com = w["com"] * ((x - cref[None, :, 0]) ** 2 + (y - cref[None, :, 1]) ** 2) + w["z"] * (z - cref[None, :, 2]) ** 2
    qv = qveltraj[:, 1:, :]
    vel = ((qv[..., :3] - cvref[None, :, :]) ** 2).sum(-1)
    brake = (qv[..., :3] ** 2).sum(-1)
    omega = (qv[..., 3:6] ** 2).sum(-1)
    alive = np.maximum(params["z0"] - 0.15 - z, 0.0)     # dial-mpc anti-fall (mirrors _stones_loss)
    below = np.maximum(params["stone_z"] - fp[..., 2], 0.0) ** 2
    off = np.clip((dxy2 - params["r_pillar"] ** 2) / params["r_pillar"] ** 2, 0.0, 1.0)
    miss = (off * np.maximum(params["stone_z"] + 0.055 - fp[..., 2], 0.0) ** 2).sum(-1)
    land = dxy2.sum(-1)                                   # exact center during descent/hold
    worst = dxy2.max(-1)                                  # one missed foot cannot hide behind three good feet
    over = np.maximum(x - cref[None, :, 0] - 0.03, 0.0) ** 2
    finish_err = (x - cref[None, :, 0]) ** 2
    rear_clear = (np.maximum(0.75 - q[..., 14], 0.0) ** 2 + np.maximum(0.75 - q[..., 17], 0.0) ** 2
                  + 0.5 * np.maximum(q[..., 15] + 2.05, 0.0) ** 2
                  + 0.5 * np.maximum(q[..., 18] + 2.05, 0.0) ** 2)
    land_pose = ((q[..., 8] - params["front_thigh0"]) ** 2 + (q[..., 11] - params["front_thigh0"]) ** 2
                 + 0.5 * (q[..., 9] - params["front_calf0"]) ** 2
                 + 0.5 * (q[..., 12] - params["front_calf0"]) ** 2
                 + (q[..., 14] - params["rear_thigh0"]) ** 2 + (q[..., 17] - params["rear_thigh0"]) ** 2
                 + 0.5 * (q[..., 15] - params["rear_calf0"]) ** 2
                 + 0.5 * (q[..., 18] - params["rear_calf0"]) ** 2)
    state_sym = ((q[..., 7] + q[..., 10]) ** 2 + (q[..., 8] - q[..., 11]) ** 2
                 + (q[..., 9] - q[..., 12]) ** 2 + (q[..., 13] + q[..., 16]) ** 2
                 + (q[..., 14] - q[..., 17]) ** 2 + (q[..., 15] - q[..., 18]) ** 2)
    zmean = fp[..., 2].mean(axis=-1, keepdims=True)
    sync = ((fp[..., 2] - zmean) ** 2).sum(-1)             # all four feet touch down at the same height
    trackxy = (1.0 - lref[None, :]) * dxy2.sum(-1) + lref[None, :] * exy.sum(-1)
    per = (w["foot"] * trackxy + w["footz"] * ez.sum(-1)
           + lref[None, :] * (w["land"] * land + w["floor"] * below.sum(-1) + w["sync"] * sync
                              + w["worst"] * worst + w["over"] * over + w["miss"] * miss)
           + clear_ref[None, :] * w["clear"] * rear_clear
           + pose_ref[None, :] * (w["land_pose"] * land_pose + w["brake"] * brake + w["omega"] * omega)
           + finish_ref[None, :] * w["finish"] * finish_err
           + w["state_sym"] * state_sym
           + com + w["vel"] * vel
           + w["up"] * upright + w.get("yaw", 0.0) * yaw ** 2 + w["slip"] * slip
           + w.get("alive", 0.0) * alive ** 2)
    if w.get("raibert", 0.0) > 0.0:                       # RAIBERT foothold reach (mirrors _raibert_loss; FD trunk vel for display)
      rox, rw = params["raib_ox"], params["raib_w"]                                  # (T,4),(T,4)
      vx = np.gradient(qtraj[:, :, 0], task.dt, axis=1)[:, 1:]                             # (N,T)
      reach = RAIBERT_REACH * vx + RAIBERT_K * (vx - params["v_des"])                 # (N,T)
      foot_relx = foot_traj[:, 1:, :, 0] - qtraj[:, 1:, 0:1]                          # (N,T,4)
      raib = (rw[None, :, :] * (foot_relx - (rox[None, :, :] + reach[:, :, None])) ** 2).sum(-1)  # (N,T)
      per = per + w["raibert"] * raib
  else:  # handstand: pitch (+90deg) + head held up + angular settling (mirrors _handstand_loss)
    al = 0.70710678 * (q[..., 3] + qy)
    fz = 2.0 * (q[..., 4] * q[..., 6] - q[..., 3] * qy)
    head_z = z + 0.3 * fz
    below = np.maximum(task.height_tar - head_z, 0.0)
    ang = (qveltraj[:, 1:, 3:6] ** 2).sum(-1)
    ws = (np.arange(1, T + 1) / T)[None, :]
    per = ws * (w["ori"] * (1.0 - al * al) + w["head"] * below ** 2 + w["bal"] * ang)
  return per.mean(axis=1) if o == "stones" else per.sum(axis=1)


def _stones_component_sums(task, qtraj, qveltraj, foot_traj, params):
  """Weighted loss audit for the selected stones iterate (display only; mirrors the analytic terms)."""
  w = task.w
  q, qv, fp = qtraj[0, 1:], qveltraj[0, 1:], foot_traj[0, 1:]
  fref, cref, cvref = params["foot_ref"], params["com_ref"], params["com_vel_ref"]
  sref, lref = params["stance_ref"], params["land_ref"]
  clear_ref, finish_ref, pose_ref = params["clear_ref"], params["finish_ref"], params["pose_ref"]
  dxy2 = ((fp[..., :2] - fref[..., :2]) ** 2).sum(-1)
  fvel = (foot_traj[0, 1:, :, :2] - foot_traj[0, :-1, :, :2]) / task.dt
  qw, qx, qy, qz = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
  ux, uy, uzm1 = 2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), -2 * (qx ** 2 + qy ** 2)
  below = np.maximum(params["stone_z"] - fp[..., 2], 0.0) ** 2
  off = np.clip((dxy2 - params["r_pillar"] ** 2) / params["r_pillar"] ** 2, 0.0, 1.0)
  miss = (off * np.maximum(params["stone_z"] + 0.055 - fp[..., 2], 0.0) ** 2).sum(-1)
  zmean = fp[..., 2].mean(axis=-1, keepdims=True)
  alive = np.maximum(params["z0"] - 0.15 - q[:, 2], 0.0)
  return {
    "footxy": float(w["foot"] * (((1.0 - lref[:, None]) * dxy2
                                    + lref[:, None] * np.maximum(dxy2 - params["r_pillar"] ** 2, 0.0)).sum())),
    "footz": float(w["footz"] * ((fp[..., 2] - fref[..., 2]) ** 2).sum()),
    "landxy": float(w["land"] * (lref[:, None] * dxy2).sum()),
    "floor": float(w["floor"] * (lref[:, None] * below).sum()),
    "sync": float(w["sync"] * (lref[:, None] * (fp[..., 2] - zmean) ** 2).sum()),
    "worst": float(w["worst"] * (lref * dxy2.max(-1)).sum()),
    "over": float(w["over"] * (lref * np.maximum(q[:, 0] - cref[:, 0] - 0.03, 0.0) ** 2).sum()),
    "brake": float(w["brake"] * (pose_ref * (qv[:, :3] ** 2).sum(-1)).sum()),
    "miss": float(w["miss"] * (lref * miss).sum()),
    "clear": float(w["clear"] * (clear_ref * (np.maximum(0.75 - q[:, 14], 0.0) ** 2
                                                + np.maximum(0.75 - q[:, 17], 0.0) ** 2
                                                + 0.5 * np.maximum(q[:, 15] + 2.05, 0.0) ** 2
                                                + 0.5 * np.maximum(q[:, 18] + 2.05, 0.0) ** 2)).sum()),
    "land_pose": float(w["land_pose"] * (pose_ref * ((q[:, 8] - params["front_thigh0"]) ** 2
                                                        + (q[:, 11] - params["front_thigh0"]) ** 2
                                                        + 0.5 * (q[:, 9] - params["front_calf0"]) ** 2
                                                        + 0.5 * (q[:, 12] - params["front_calf0"]) ** 2
                                                        + (q[:, 14] - params["rear_thigh0"]) ** 2
                                                        + (q[:, 17] - params["rear_thigh0"]) ** 2
                                                        + 0.5 * (q[:, 15] - params["rear_calf0"]) ** 2
                                                        + 0.5 * (q[:, 18] - params["rear_calf0"]) ** 2)).sum()),
    "finish": float(w["finish"] * (finish_ref * (q[:, 0] - cref[:, 0]) ** 2).sum()),
    "state_sym": float(w["state_sym"] * ((q[:, 7] + q[:, 10]) ** 2 + (q[:, 8] - q[:, 11]) ** 2
                                           + (q[:, 9] - q[:, 12]) ** 2 + (q[:, 13] + q[:, 16]) ** 2
                                           + (q[:, 14] - q[:, 17]) ** 2 + (q[:, 15] - q[:, 18]) ** 2).sum()),
    "comxy": float(w["com"] * ((q[:, :2] - cref[:, :2]) ** 2).sum()),
    "comz": float(w["z"] * ((q[:, 2] - cref[:, 2]) ** 2).sum()),
    "vel": float(w["vel"] * ((qv[:, :3] - cvref) ** 2).sum()),
    "up": float(w["up"] * (ux ** 2 + uy ** 2 + uzm1 ** 2).sum()),
    "omega": float(w["omega"] * (pose_ref * (qv[:, 3:6] ** 2).sum(-1)).sum()),
    "yaw": float(w["yaw"] * (4 * (qw * qz + qx * qy) ** 2).sum()),
    "slip": float(w["slip"] * (sref * (fvel ** 2).sum(-1)).sum()),
    "alive": float(w["alive"] * (alive ** 2).sum()),
  }


def _print_stones_landings(task, qtraj, foot_traj, params):
  """Print the same boundary landing metrics as _pj_slip.py so rendered 120-step runs are self-diagnosing."""
  q, ft = qtraj[0], foot_traj[0]
  t0 = int(round(STONES_LEAD * task.T))
  hop_len = (task.T - t0) / task.n_hops
  print(f"  selected travel={q[-1, 0] - q[0, 0]:+.3f}m target={params['com_ref'][-1, 0] - q[0, 0]:+.3f}m "
        f"tilt={abs(q[-1, 4]) + abs(q[-1, 5]):.3f}")
  for hop in range(task.n_hops):
    b = t0 + int(round((hop + 1) * hop_len))
    target = params["foot_targets"][hop + 1]
    vals = []
    for i, name in enumerate(FEET):
      err = float(np.linalg.norm(ft[b, i, :2] - target[i, :2]))
      vals.append(f"{name}={err:.3f}m/z{ft[b, i, 2]:.3f}")
    print(f"  hop {hop} landing: " + "  ".join(vals))


def _print_handstand_metrics(task, qtraj, qveltraj):
  """Per-environment selected-iterate support metrics for the four-lane handstand render."""
  late = min(25, task.T)
  for e in range(qtraj.shape[0]):
    q = qtraj[e]
    qv = qveltraj[e]
    dot = float(np.clip(abs(0.70710678 * (q[-1, 3] + q[-1, 5])), 0.0, 1.0))
    ori_deg = float(2.0 * np.degrees(np.arccos(dot)))
    fz = 2.0 * (q[:, 4] * q[:, 6] - q[:, 3] * q[:, 5])
    head_z = q[:, 2] + 0.3 * fz
    omega = np.linalg.norm(qv[:, 3:6], axis=1)
    print(f"  env {e}: final_ori_err={ori_deg:.1f}deg  final_head_z={head_z[-1]:.3f}m  "
          f"late_head_min={head_z[-late:].min():.3f}m  late_omega_mean={omega[-late:].mean():.3f}rad/s")


# ---------------- engine ----------------
def build(task, N):
  mjm = mujoco.MjModel.from_xml_path(os.path.join(BENCH, task.scene))
  mjm.opt.timestep = task.dt          # per-task sim dt (default = the scene's; stones uses a bigger dt for slower hops)
  mjd = mujoco.MjData(mjm)
  m = mjw.put_model(mjm)
  kid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  home = mjm.key_ctrl[kid].astype(np.float32).copy()
  if task.objective == "stones":
    # COMPACT stance: front thighs pitched fwd (+0.5) tuck the front feet BACK, rear thighs (-0.5) tuck the rear
    # feet FWD -> front-to-rear foot spread ~0.28 (vs natural 0.37), feet stay level. Lets the pronk hops be
    # SHORT (0.28, reachable) instead of a 0.37 stance-length leap the Go1 can't torque. (_pose_probe.py)
    home[1] += 0.5; home[3] += 0.5; home[5] -= 0.5; home[7] -= 0.5
  # all tasks settle from the crouch keyframe holding home ctrl (position servos). stones: additionally RAISE
  # the base so the feet spawn ABOVE the tall pillar tops, then settle straight down -> the robot ends standing
  # with each foot on a start-pad pillar (regenerated per pillar height, no baked keyframe needed).
  mujoco.mj_resetDataKeyframe(mjm, mjd, kid)
  mjd.ctrl[:] = home                    # settle holds the (possibly compacted) home ctrl
  if task.objective == "stones":
    mjd.qpos[2] += STONES_RAISE
  mujoco.mj_forward(mjm, mjd)
  ds = [mjw.put_data(mjm, mjd) for _ in range(SETTLE + 1)]
  for t in range(SETTLE):
    mjw.step(m, ds[t], ds[t + 1])
  qpos0 = ds[SETTLE].qpos.numpy()[0].copy()
  if task.objective == "stones":
    # Contact-row ordering leaves milliradian L/R differences after settling; in a chaotic single-shooting
    # rollout those seed the diagonal two-foot solution. Start from the exactly mirrored version of the same
    # physical settled pose, matching the symmetric pads and the mirrored pronk control parameterization.
    qpos0[1] = 0.0
    qnorm = float(np.hypot(qpos0[3], qpos0[5]))
    qpos0[3], qpos0[4], qpos0[5], qpos0[6] = qpos0[3] / qnorm, 0.0, qpos0[5] / qnorm, 0.0
    for r, l in ((7, 10), (13, 16)):
      hip = 0.5 * (qpos0[r] - qpos0[l]); qpos0[r], qpos0[l] = hip, -hip
      thigh = 0.5 * (qpos0[r + 1] + qpos0[l + 1]); qpos0[r + 1], qpos0[l + 1] = thigh, thigh
      calf = 0.5 * (qpos0[r + 2] + qpos0[l + 2]); qpos0[r + 2], qpos0[l + 2] = calf, calf

  params = {"z0": float(qpos0[2])}
  if task.objective == "reach":
    gid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "target")
    params["target"] = mjm.geom_pos[gid].copy()
    if "feet" in task.w:                                 # crate_climb: foot sites + crate-top footprint (dial-mpc feet-on-crate)
      params["foot_ids"] = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, f) for f in FEET]
      cgid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "crate")
      cpos, csize = mjm.geom_pos[cgid], mjm.geom_size[cgid]
      params["crate"] = (float(cpos[0] - csize[0]), float(cpos[0] + csize[0]),      # x_lo, x_hi
                         float(cpos[1] - csize[1]), float(cpos[1] + csize[1]),      # y_lo, y_hi
                         float(cpos[2] + csize[2]))                                  # top z
  elif task.objective == "sequence":
    wps = []
    for i in range(64):
      gid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, f"wp_{i}")
      if gid < 0:
        break
      wps.append(mjm.geom_pos[gid].copy())
    params["waypoints"] = np.array(wps)
  elif task.objective == "stones":
    # foot sites + their on-stone rest positions (fwd_kinematics at the settled qpos0)
    fids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, f) for f in FEET]
    d = mjw.put_data(mjm, mjd)
    d.qpos = wp.array(qpos0.reshape(1, -1).astype(np.float32), dtype=float)
    mjw.fwd_kinematics(m, d)
    foot0 = d.site_xpos.numpy()[0, fids]              # (4,3) FR,FL,RR,RL start (on the start-pad stones)
    # Use the actual pillar centers, not the slightly contact-displaced settled feet, for every landing target.
    # The desired site z is exactly pillar top + spherical-foot radius (~0.143 for the 0.12 m tops).
    def geom_pos(name):
      gid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, name)
      return mjm.geom_pos[gid].copy()

    sgid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "stone_0L")
    fgid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "FR_foot_collision")
    ztar = float(mjm.geom_pos[sgid][2] + mjm.geom_size[sgid][1] + mjm.geom_size[fgid][0])
    pads = np.array([geom_pos(f"pad_{f}")[:2] for f in FEET], np.float32)
    n_stages = task.n_hops + 1                                # stage 0 = start stance; then n_hops forward hops
    ft = np.zeros((n_stages, 4, 3), np.float32)
    ft[0, :, :2] = pads
    ft[0, :, 2] = ztar
    for s in range(1, n_stages):
      front = np.array([geom_pos(f"stone_{s - 1}R")[:2], geom_pos(f"stone_{s - 1}L")[:2]])
      if s == 1:
        rear = pads[:2]                                       # rear feet follow the old front-foot row
      else:
        rear = np.array([geom_pos(f"stone_{s - 2}R")[:2], geom_pos(f"stone_{s - 2}L")[:2]])
      ft[s, 0:2, :2], ft[s, 2:4, :2], ft[s, :, 2] = front, rear, ztar
    params["foot_ids"] = fids
    params["foot_targets"] = ft
    params["n_stages"] = n_stages
    params["r_pillar"] = float(max(mjm.geom_size[sgid][0] - 0.03, 0.02))  # of the target, the foot sphere sits ON the top
    params["stone_z"] = ztar
    fref, cref, cvref, sref, lref, clref, firef, pref, rox, rw = _stones_refs(task, ft, qpos0)  # arc/COM velocity/landing refs
    params["foot_ref"] = fref
    params["com_ref"] = cref
    params["com_vel_ref"] = cvref
    params["stance_ref"] = sref
    params["land_ref"] = lref
    params["clear_ref"] = clref
    params["finish_ref"] = firef
    params["pose_ref"] = pref
    params["front_thigh0"] = float(0.5 * (qpos0[8] + qpos0[11]))
    params["front_calf0"] = float(0.5 * (qpos0[9] + qpos0[12]))
    params["rear_thigh0"] = float(0.5 * (qpos0[14] + qpos0[17]))
    params["rear_calf0"] = float(0.5 * (qpos0[15] + qpos0[18]))
    params["raib_ox"] = rox
    params["raib_w"] = rw
    t0 = int(round(STONES_LEAD * task.T))
    params["v_des"] = float((cref[-1, 0] - cref[t0, 0]) / ((task.T - t0) * task.dt))  # avg forward COM speed while hopping
  elif task.objective == "gallop":
    # foot sites + their flat-stance z (the gait swing-height reference is added on top of this rest z)
    fids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, f) for f in FEET]
    d = mjw.put_data(mjm, mjd)
    d.qpos = wp.array(qpos0.reshape(1, -1).astype(np.float32), dtype=float)
    mjw.fwd_kinematics(m, d)
    foot0 = d.site_xpos.numpy()[0, fids]                  # (4,3) rest foot positions
    params["foot_ids"] = fids
    params["foot_rest_z"] = float(foot0[:, 2].mean())
    params["raibert_ox"] = (foot0[:, 0] - qpos0[0]).astype(np.float64)   # rest foot x offset from trunk (hip)
    # per-foot Raibert "reach" weight over the horizon: 1 during LATE swing (height rising then DESCENDING
    # toward touchdown), 0 in stance/early swing -> only shape where the foot is heading as it lands
    hs = np.array([_foot_step_heights((t + 1) * task.dt) for t in range(task.T)])   # (T,4)
    rw = np.zeros((task.T, 4))
    rw[1:] = ((hs[1:] > 0.01) & (hs[1:] <= hs[:-1])).astype(float)
    params["reach_w"] = rw
  return mjm, mjd, m, np.tile(qpos0, (N, 1)).astype(np.float32), home, params


def run_backward(task, m, mjm, mjd, ctrls, q0, home_wp, N, params):
  """Batched (nworld=N) taped rollout with PERSISTENT per-step control leaves ctrls[t] (1D, N*nu),
  each reshaped to (N,nu) and bound just-in-time (mujoco_warp step(d,d_out) copies ctrl d->d_out).
  Leaves the accumulated gradients in ctrls[t].grad for warp.optim.Adam. Returns qtraj (N,T+1,nq).

  TRUNCATED BPTT (task.tbptt_h > 0, like g1_sysid.py): the rollout is cut into chunks of tbptt_h frames;
  the state is DETACHED at each chunk boundary (a fresh requires_grad leaf, no grad link to the previous
  chunk) and each chunk's loss is back-propagated on its OWN tape. Every backward chain is thus <= tbptt_h
  steps -- dodging the long-horizon contact chaos/local-optima that a full-T single-shooting backward hits --
  while the per-step control grads still ACCUMULATE across chunks, so the objective stays the full trajectory.
  tbptt_h == 0 -> one tape over all T (classic full BPTT)."""
  T, nu = task.T, mjm.nu
  feet = task.objective in ("stones", "gallop") or "feet" in task.w   # loss reads foot site_xpos (differentiable FK)
  H = task.tbptt_h if task.tbptt_h > 0 else T                          # chunk length (T = full BPTT)
  for c in ctrls:
    c.grad.zero_()                                    # tape.backward ACCUMULATES -> zero once, before all chunks
  datas = [mjw.put_data(mjm, mjd, nworld=N) for _ in range(T + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
    if feet:
      d.site_xpos.requires_grad = True               # site_xpos.grad -> qpos.grad via fwd_kinematics hook
  cur_qpos = q0.copy()                                # detached state carried between chunks (numpy round-trip)
  cur_qvel = np.zeros((N, mjm.nv), np.float32)
  t = 0
  while t < T:
    ct = min(H, T - t)                                # frames in this truncation chunk
    datas[t].qpos = wp.array(cur_qpos, dtype=float, requires_grad=True)  # fresh DETACHED leaf (breaks grad chain)
    datas[t].qvel = wp.array(cur_qvel, dtype=float, requires_grad=True)
    if feet:
      mjw.fwd_kinematics(m, datas[t])                 # site_xpos at the chunk-start state
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      for tt in range(t, t + ct):
        c = ctrls[tt].reshape((N, nu))                # (N,nu) view of the 1D leaf; just-in-time bind
        datas[tt].ctrl = c
        mjw.step(m, datas[tt], datas[tt + 1])
        if feet:
          mjw.fwd_kinematics(m, datas[tt + 1])        # refresh site_xpos under the tape -> foot-site gradient
        _launch_loss(task, datas[tt + 1], datas[tt], c, home_wp, tt, N, params, loss)
    tape.backward(loss=loss)                          # += this chunk's grads into ctrls[tt].grad (tt in chunk)
    cur_qpos = datas[t + ct].qpos.numpy()             # DETACH: next chunk starts here with no gradient link back
    cur_qvel = datas[t + ct].qvel.numpy()
    t += ct
  qtraj = np.stack([datas[t].qpos.numpy() for t in range(T + 1)], axis=1)          # (N,T+1,nq)
  qveltraj = np.stack([datas[t].qvel.numpy() for t in range(T + 1)], axis=1)       # (N,T+1,nv)
  foot_traj = None
  if feet:
    fids = params["foot_ids"]
    foot_traj = np.stack([datas[t].site_xpos.numpy()[:, fids] for t in range(T + 1)], axis=1)  # (N,T+1,4,3)
  return qtraj, qveltraj, foot_traj


def optimize(task, N, steps, lr):
  mjm, mjd, m, q0, home, params = build(task, N)
  nu = mjm.nu
  home_wp = wp.array(home, dtype=float)
  lo, hi = (home - ACTION).astype(np.float32), (home + ACTION).astype(np.float32)   # box-clamp targets around home
  lo_wp, hi_wp = wp.array(lo, dtype=float), wp.array(hi, dtype=float)
  rng = np.random.default_rng(0)
  base = np.clip(home[None, :] + rng.normal(0.0, task.spread, (N, nu)).astype(np.float32), lo, hi)  # (N,nu) per-env bias -> fan
  # PERSISTENT per-step control leaves (1D N*nu) optimized by warp.optim.Adam with SHAC betas (0.7, 0.95)
  if task.objective == "stones":
    seed = _stones_ctrl_seed(task, home)
    ctrls = [wp.array(np.tile(seed[t], (N, 1)).reshape(-1), dtype=float, requires_grad=True) for t in range(task.T)]
  else:
    ctrls = [wp.array(base.reshape(-1).copy(), dtype=float, requires_grad=True) for _ in range(task.T)]
  opt = warp.optim.Adam(ctrls, lr=lr, betas=(0.7, 0.95))
  history = []
  for it in range(steps):
    if task.objective == "stones" and steps > 1:
      # The pronk seed finds the useful contact basin early. Decay aggressively thereafter: the previous
      # linear schedule still used ~0.018 at the good iter-18 solution and Adam walked into a row-skipping
      # contact mode. This reaches ~0.0026 by iter 40 and ~0.0005 at iter 119 for landing refinement.
      opt.lr = lr * (0.025 + 0.975 * np.exp(-it / 18.0))
    qtraj, qveltraj, foot_traj = run_backward(task, m, mjm, mjd, ctrls, q0, home_wp, N, params)
    losses = per_env_loss(task, qtraj, params, foot_traj, qveltraj)
    fwd = float(np.mean(qtraj[:, -1, 0] - qtraj[:, 0, 0]))
    peak = float(np.mean(qtraj[:, :, 2].max(axis=1)))
    history.append({"it": it, "losses": losses.copy(), "qtraj": qtraj,
                    "qveltraj": qveltraj, "foot_traj": foot_traj})
    gnorm = float(np.mean([np.linalg.norm(np.nan_to_num(c.grad.numpy())) for c in ctrls]))  # mean per-step |grad|
    if task.grad_clip > 0.0 and gnorm > task.grad_clip:  # trajopt hygiene: scale the whole gradient down so Adam
      s = task.grad_clip / (gnorm + 1e-8)                # can't overshoot into a topple when an aggressive launch
      for c in ctrls:                                    # slams a foot into a pillar edge (stiff contact -> |g| spike)
        wp.launch(_scale, dim=c.shape, inputs=[c.grad, float(s)])
    if it % 10 == 0 or it == steps - 1:
      lname = "mean_loss"
      print(f"  [{it:3d}] {lname}={losses.mean():.3f} best={losses.min():.3f}  "
            f"mean_fwd={fwd:+.2f}m mean_peak_z={peak:.2f}  |g|={gnorm:.2f}")
    opt.step([c.grad for c in ctrls])                 # in-place Adam update of the control leaves
    for c in ctrls:
      if task.objective == "stones":
        wp.launch(_project_pronk, dim=N, inputs=[c, nu])  # exact L/R synchronization, still real force-limited targets
      wp.launch(_clamp, dim=c.shape, inputs=[c, lo_wp, hi_wp, nu])  # project back to home +/- ACTION
  mean_loss = np.array([h["losses"].mean() for h in history])
  best = int(mean_loss.argmin())
  lname = "mean loss"
  print(f"[go1_agile/{task.name}] {N} env(s): {lname} {mean_loss[0]:.3f} -> best {mean_loss[best]:.3f} (iter {best})")
  if task.objective == "stones":
    hb = history[best]
    parts = _stones_component_sums(task, hb["qtraj"], hb["qveltraj"], hb["foot_traj"], params)
    print("  best summed loss terms: " + "  ".join(f"{k}={v:.1f}" for k, v in parts.items()))
    _print_stones_landings(task, hb["qtraj"], hb["foot_traj"], params)
  elif task.objective == "handstand":
    hb = history[best]
    _print_handstand_metrics(task, hb["qtraj"], hb["qveltraj"])
  return history, best, mjm, params


# ---------------- render (all envs in ONE scene, replicated into y-lanes, cradle-style) ----------------
def _lanes_model(task, N, sp):
  """All N envs in ONE scene: the imported single-robot scene replicated into N y-lanes via MjSpec --
  attach a Go1 copy per lane (the from-file equivalent of dm_suite's inline-<replicate>) + duplicate the
  obstacle geoms into each lane. Returns the compiled viz model; qpos is env-major ([env0(nq), env1..])."""
  robot = mujoco.MjSpec.from_file(os.path.join(BENCH, "unitree_go1_mjlab.xml"))
  scene = mujoco.MjSpec.from_file(os.path.join(BENCH, task.scene))               # lane 0 = robot + obstacles + floor
  obst = [(g.type, np.array(g.size), np.array(g.pos), np.array(g.quat), np.array(g.rgba), g.material)
          for g in scene.worldbody.geoms if g.name != "floor"]                    # obstacles to replicate per lane
  for e in range(1, N):
    fr = scene.worldbody.add_frame()
    fr.pos = [0.0, e * sp, 0.0]
    scene.attach(robot.copy(), prefix=f"e{e}_", frame=fr)
    for typ, size, pos, quat, rgba, mat in obst:
      g2 = scene.worldbody.add_geom()
      g2.type, g2.size, g2.quat, g2.rgba = typ, size, quat, rgba
      g2.pos = [float(pos[0]), float(pos[1] + e * sp), float(pos[2])]
      g2.contype, g2.conaffinity = 0, 0                                           # viz-only (render sets qpos, no stepping)
      if mat:
        g2.material = mat
  return scene.compile()


def render(task, history, best, out_path, sample_every=4, live=None):
  N = history[0]["qtraj"].shape[0]
  T, nq_r = task.T, history[0]["qtraj"].shape[2]
  sp = task.lane
  m = _lanes_model(task, N, sp)
  vd = mujoco.MjData(m)
  off = np.array([[0.0, e * sp, 0.0] for e in range(N)])
  lo, hi = 0.0, float(max(h["losses"].max() for h in history))

  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(m, cam)
  c = task.cam
  ymid = (N - 1) * sp / 2.0
  if c.get("track"):  # locomotion: frame the forward extent + all lanes
    allx = history[best]["qtraj"][:, :, 0]
    cam.lookat = [float(0.5 * (allx.min() + allx.max())), ymid, c.get("z", 0.3)]
    span = max(float(allx.max() - allx.min()), (N - 1) * sp)
    cam.distance = c.get("dist0", 2.0) + c.get("distk", 0.9) * span    # dist0/distk tune the zoom
  else:
    cam.lookat = [c["lookat"][0], ymid, c["lookat"][2]]
    cam.distance = c["distance"] + 0.55 * (N - 1) * sp
  cam.azimuth = c.get("azimuth", 115.0)
  cam.elevation = c.get("elevation", -16.0)

  best_mean = min(h["losses"].mean() for h in history)
  frames = []
  show = viz.default_show(len(history), best)
  if task.objective == "stones":
    # Show later drift honestly, but always finish/replay on the selected incumbent instead of holding the
    # final (possibly worse) raw Adam iterate—the old video ended on a row-skipping iter 119 despite best=18.
    show = [k for k in show if k != best] + [best]
  for k in show:
    hk = history[k]
    ccols = [viz.bourke_color_map(lo, hi, float(v)) for v in hk["losses"]]
    loss_name = "mean loss"
    sub = f"iter {hk['it']:3d}   {loss_name} {hk['losses'].mean():.3f}   best {best_mean:.3f}   envs {N}"
    steps_idx = list(range(0, T + 1, sample_every)) + [T]
    hold = 20 if k == best else 0
    for t in steps_idx + [T] * hold:
      q = np.zeros(m.nq)
      for e in range(N):
        q[e * nq_r:(e + 1) * nq_r] = hk["qtraj"][e, t]
        q[e * nq_r + 1] += e * sp                        # place env e in its y-lane (free-joint qpos is absolute)
      cur = [hk["qtraj"][e, : t + 1, 0:3] + off[e] for e in range(N)]

      def draw(scene, cur=cur, ccols=ccols):             # each env's loss-colored COM trail in its lane
        for e in range(N):
          viz.add_polyline(scene, cur[e][::2], ccols[e], width=0.012)

      frames.append((q.copy(), draw, sub))
  if frames:
    frames += [frames[-1]] * 20
  return viz.emit(m, vd, cam, frames, out_path=out_path, label=f"ADJOINT ({task.title or task.name})", w=W, h=H, fps=FPS, live=live)


def main():
  ap = argparse.ArgumentParser(description="Unitree Go1 agile skills -- open-loop BPTT trajectory optimization.")
  ap.add_argument("--task", choices=list(TASKS), default="crate_climb", help="which agile skill to optimize")
  ap.add_argument("--envs", type=int, default=4, help="parallel envs (mujoco_warp nworld batching)")
  ap.add_argument("--steps", type=int, default=None, help="optimization iterations (default: per-task)")
  ap.add_argument("--lr", type=float, default=None, help="Adam learning rate (default: per-task)")
  ap.add_argument("--out", default=None, help="output mp4 (default: reports/assets/go1_<task>.mp4; MJW_RENDER_PATH honored)")
  ap.add_argument("--live", action="store_true", help="open the live MuJoCo viewer instead of writing an mp4")
  args = ap.parse_args()

  task = TASKS[args.task]
  steps = args.steps if args.steps is not None else task.steps
  lr = args.lr if args.lr is not None else task.lr
  out = args.out or os.environ.get("MJW_RENDER_PATH") or os.path.join(ASSETS, f"go1_{task.name}.mp4")
  live = True if args.live else None
  os.makedirs(os.path.dirname(out), exist_ok=True)

  history, best, mjm, params = optimize(task, args.envs, steps, lr)
  render(task, history, best, out_path=out, live=live)


if __name__ == "__main__":
  main()
