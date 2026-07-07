"""Box-pyramid "knock only the TOP cube" -- launch a wrecking ball at a stacked pyramid of cubes and
optimize the ball's launch HEIGHT so it clips ONLY the top cube off, leaving the base standing. Ported
from the Newton `example_pyramid` narrow-phase stress test (github.com/jumyungc/newton .../example_pyramid.py),
shrunk to a tractable 3-2-1 pyramid (6 cubes) + one ball.

  loss = W_BASE * mean_{base cube}(planar displacement / D)^2  -  W_TOP * (top-cube planar displacement / D)^2

The base term keeps the 5 lower cubes put (0 = undisturbed); the top-cube reward makes "knock the top off"
strictly better than "miss everything" (loss 0). At the zero init the ball is aimed at the MIDDLE row, so it
plows a bunch of cubes down (large base term); gradient descent raises the aim until the ball sails over the
base and clips only the top -> the base stays and the loss goes negative. Many pyramids are optimized IN
PARALLEL (one lane per env), each its own launch height, seeded across a spread so the lanes start distinct.

*** THIS IS A STALLING-GATE DEMO (differentiable-mujoco_warp coverage test). *** The load-bearing contacts of
a CUBE pyramid -- cube resting on cube, cube knocking cube -- are BOX-BOX, which the analytic adjoint
(collision_adjoint.py) does NOT differentiate (dqpos stays 0 for box-box; only sphere-*, plane-box, ... are
AD-safe). So the ANALYTIC gradient is biased/near-zero through the pyramid and DOES NOT optimize this task;
`--grad fd` (finite difference over the MuJoCo-C rollout) is the robust optimizer that produces the result +
video. Every few iters we still run the analytic backward and print its cos / magnitude-ratio vs FD, so the
demo DOCUMENTS exactly where the box-box adjoint stalls (contrast cradle.py / dominos.py, whose sphere-sphere
contacts are AD-safe and DO optimize with `--grad analytic`, cos~1). See collision_adjoint.py:40-48.

  uv run --active --with imageio --with imageio-ffmpeg --with pillow python contrib/diffsim/pyramid.py
  uv run --active python contrib/diffsim/pyramid.py --grad analytic --no-render   # inspect the box-box stall
"""

import argparse
import os
import sys

import mujoco
import numpy as np
import warp as wp
import warp.optim

import mujoco_warp as mjw
mjw.enable_grad()

sys.path.insert(0, os.path.dirname(__file__))
import viz as R  # noqa: E402

# --- pyramid geometry (a TRUE 3D pyramid: level r is an (BASE-r)x(BASE-r) grid of cubes at z=(2r+1)*
#     CUBE_HALF, each level centered and offset half a spacing so it NESTS on the four cubes below it -- a
#     stable 4-corner-supported stack. SPACING slightly > 2*CUBE_HALF leaves visible GAPS between cubes) ---
BASE = 3  # base is BASE x BASE cubes; total = sum_{r} (BASE-r)^2 (BASE=3 -> 9+4+1 = 14 cubes)
CUBE_HALF = 0.07  # small cubes so the big ball spans several of them (plows the base broadly, not a punch-through)
SPACING = 2.1 * CUBE_HALF  # > 2*CUBE_HALF -> a visible gap between neighbors (cubes still rest flat on 4 corners)
CUBE_MASS = 0.2
MU = 0.8  # cube<->cube / cube<->floor friction (base must not slide out from under a top hit)

# --- wrecking ball (a big dense free sphere launched horizontally from the -x side toward the pyramid) ---
# The ball clips ONLY the apex when it ARRIVES at the top with its center in (z_apex-CUBE_HALF+R, z_apex+CUBE_HALF+R):
# its bottom then clears the protruding lower layers (ball bottom > their tops) and it hits only the apex. That
# window is 2*CUBE_HALF wide REGARDLESS of R, so the ball can be BIGGER than a cube -- wide enough to plow the
# middle broadly on a flat throw, yet still clip the top cleanly once it learns to throw UP into the window.
BALL_R = 0.11  # BIGGER than a cube half-width -> spans multiple cubes (broad base topple)
BALL_MASS = 0.8  # HEAVY dense wrecking ball (4x a cube)
BALL_X0 = -0.5  # start close enough that the ball REACHES the pyramid (doesn't fall short) and hits its
# staircase front while still RISING: higher vz -> strikes higher up the front, clearing base cubes one level
# at a time, so base disturbance falls smoothly with vz (a learnable, monotone-ish path base->middle->apex).
VX = 4.0  # forward launch (+x); OPTIMIZED variable is the UPWARD launch velocity vz.

DT, T = 0.005, 150  # 0.75 s: ball flight + knock + brief settle
SOLREF, SOLIMP = "0.01 1", "0.9 0.95 0.001"  # moderately stiff (stacked cubes must not sink while resting)

D = 2.0 * CUBE_HALF  # loss length scale (cube width)
# loss = W_BASE * mean_base(1 - exp(-disp^2/SIG^2))  +  W_MISS * (apex final height / Z_TOP)^2.
# First term: a SATURATING per-cube penalty -> it scores HOW MANY base cubes were disturbed (each ->1 once
# moved ~a cube-width), NOT how far they flew. That is the key: a mid "dive-bomb" throw flings a FEW cubes
# VERY far -- with a distance^2 penalty that made a spurious loss HUMP between the flat-plow and the clean
# clip, trapping local descent. Counting disturbed cubes instead is monotone (flat throw disturbs 7 > mid 4 >
# clip 0), so descent from a flat throw (vz=0) flows to the clean clip. Second term: the apex must be KNOCKED
# OFF -> ends low (~0.04); a MISS leaves it standing (~1), so it can't just fly the ball over everything.
W_BASE, W_MISS = 1.0, 1.0
SIG2 = (0.5 * D) ** 2  # saturation scale: a base cube moved ~half a cube-width already counts as "disturbed"
SIG_A2 = D ** 2  # apex "still standing" scale: the miss penalty fades once the apex is knocked ~a cube-width away
VZ_MIN, VZ_MAX = 0.0, 5.0  # upward-throw bounds (projected each step): 0 = flat throw; capped just past the clip
# basin (~vz 3.6-4.5) so the optimizer can't run off to needlessly fast throws. Launch speed at the optimum is
# ~sqrt(VX^2 + vz^2) ~ 5.6 m/s; drop VX for a gentler throw overall.
LR, STEPS = 0.2, 40
SPREAD = 2.0  # per-env initial vz fan (multi-env only): lanes span flat throw (vz=0) .. past the clip basin
VZ_INIT = 0.0  # single-env seed = a flat throw that plows the base at iter 0; the (now monotone) loss lets the
# optimizer learn a bigger and bigger UPWARD throw, sweeping the strike up the pyramid until it clips only the apex.
ENV_COLS = 2  # multi-env render grid width
W, H = 1024, 768


def _cubes(base=BASE):
  """Cube centers (x, y, z) of a true 3D pyramid: level r is a (base-r)x(base-r) grid at z=(2r+1)*CUBE_HALF,
  centered on the origin so each level NESTS half a spacing in on the four cubes below it. Returns the list
  (bottom level first, single apex LAST) and the apex index."""
  cubes = []
  for r in range(base):
    n = base - r
    z = (2 * r + 1) * CUBE_HALF
    for iy in range(n):
      for ix in range(n):
        x = (ix - (n - 1) / 2.0) * SPACING
        y = (iy - (n - 1) / 2.0) * SPACING
        cubes.append((x, y, z))
  return cubes, len(cubes) - 1


CUBES, TOP_C = _cubes()
NCUBE = len(CUBES)
Z_TOP = CUBES[TOP_C][2]  # apex center height
Z_START = 2.0 * CUBE_HALF  # ball spawns LOW (~base-row top). Thrown flat it falls short, so even hitting the
# base needs a small upward vz; the optimizer then learns a BIGGER upward throw to lob onto the top cube.


def pyramid_xml(physics_only=True):
  """Physics scene: frictional floor + NCUBE freejoint box cubes (a stable pyramid) + one freejoint ball
  spawned at (BALL_X0, 0, Z_START); its launch velocity is set per env at rollout time."""
  opt = ('<option timestep="%g" cone="elliptic" integrator="implicitfast" gravity="0 0 -9.81" '
         'solver="Newton" iterations="50"><flag eulerdamp="disable"/></option>' % DT)
  default = f'<default><geom condim="3" friction="{MU:g} 0.005 0.0001" solref="{SOLREF}" solimp="{SOLIMP}"/></default>'
  floor = '<geom name="floor" type="plane" size="5 5 0.01" rgba="0.3 0.3 0.35 1"/>'
  cubes = ""
  for i, (x, y, z) in enumerate(CUBES):
    rgba = "0.95 0.55 0.2 1" if i == TOP_C else "0.55 0.6 0.7 1"
    cubes += (
      f'<body name="c{i}" pos="{x:.4f} {y:.4f} {z:.4f}"><freejoint/>'
      f'<geom name="c{i}" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" mass="{CUBE_MASS}" rgba="{rgba}"/></body>'
    )
  ball = (
    f'<body name="ball" pos="{BALL_X0:.4f} 0 {Z_START:.4f}"><freejoint/>'
    f'<geom name="ball" type="sphere" size="{BALL_R}" mass="{BALL_MASS}" rgba="0.95 0.85 0.2 1"/></body>'
  )
  return f"<mujoco>{opt}{default}<worldbody>{floor}{cubes}{ball}</worldbody></mujoco>"


_MJM = mujoco.MjModel.from_xml_string(pyramid_xml())
_MJD = mujoco.MjData(_MJM)
mujoco.mj_forward(_MJM, _MJD)


def _adr(name):
  """(qpos adr, dof adr) of body `name`'s freejoint."""
  bid = mujoco.mj_name2id(_MJM, mujoco.mjtObj.mjOBJ_BODY, name)
  jid = _MJM.body_jntadr[bid]
  return int(_MJM.jnt_qposadr[jid]), int(_MJM.jnt_dofadr[jid])


BALL_Q, BALL_V = _adr("ball")
CUBE_Q = [_adr(f"c{i}")[0] for i in range(NCUBE)]
TOP_QADR = CUBE_Q[TOP_C]
BASE_QADR = [CUBE_Q[i] for i in range(NCUBE) if i != TOP_C]
N_BASE = len(BASE_QADR)
QINIT_NP = _MJD.qpos.copy()  # initial layout (cubes at rest, ball parked)
_QINIT_WP = wp.array(QINIT_NP.astype(np.float32), dtype=float)
_BASE_QADR_WP = wp.array(np.array(BASE_QADR, np.int32), dtype=int)
_COEF_BASE = W_BASE / N_BASE  # per-cube saturating penalty averaged over the base cubes
_COEF_MISS = W_MISS / (Z_TOP * Z_TOP)  # apex still-standing penalty (apex ends low when knocked off -> ~0)


# --- kernels -------------------------------------------------------------------------------------------


@wp.kernel
def _set_launch(vz: wp.array(dtype=float), ball_v: int, vx: float, qvel: wp.array2d(dtype=float)):
  """Set the ball's launch velocity: fixed forward speed qvel[ball_x]=vx + OPTIMIZED upward throw
  qvel[ball_z]=vz. The ball spawns at the middle height (Z_START), so vz>0 lobs it up toward the apex.
  One thread per env; cube qpos/qvel are left at their replicated init."""
  w = wp.tid()
  qvel[w, ball_v + 0] = vx
  qvel[w, ball_v + 2] = vz[w]


@wp.kernel
def _accum_base(qpos: wp.array2d(dtype=float), qinit: wp.array(dtype=float),
                base_qadr: wp.array(dtype=int), coef: float, sig2: float, loss: wp.array(dtype=float)):
  """loss += coef * sum_{env,base cube} (1 - exp(-planar_disp^2/sig2)) -- SATURATING: counts disturbed cubes."""
  w, j = wp.tid()  # (env, base cube); launch dim=(E, N_BASE)
  a = base_qadr[j]
  dx = qpos[w, a + 0] - qinit[a + 0]
  dy = qpos[w, a + 1] - qinit[a + 1]
  wp.atomic_add(loss, 0, coef * (1.0 - wp.exp(-(dx * dx + dy * dy) / sig2)))


@wp.kernel
def _accum_apex(qpos: wp.array2d(dtype=float), qinit: wp.array(dtype=float),
                top_qadr: int, coef: float, sig2: float, loss: wp.array(dtype=float)):
  """loss += coef * sum_env (apex height)^2 * exp(-apex_planar_disp^2/sig2) -- penalizes the apex STILL STANDING
  IN PLACE (high AND at its original x,y). Once knocked away (plow or clip) the exp -> 0, so a plow that flings
  the apex high doesn't false-fire; only a true MISS (apex untouched) is penalized."""
  w = wp.tid()
  z = qpos[w, top_qadr + 2]
  dx = qpos[w, top_qadr + 0] - qinit[top_qadr + 0]
  dy = qpos[w, top_qadr + 1] - qinit[top_qadr + 1]
  wp.atomic_add(loss, 0, coef * z * z * wp.exp(-(dx * dx + dy * dy) / sig2))


# --- loss / rollout ------------------------------------------------------------------------------------


def _per_env_loss(qpos_final):
  """Per-env loss from a numpy final qpos (E, nq): base preservation + apex still-standing penalty. Matches kernels."""
  base = np.zeros(qpos_final.shape[0])
  for a in BASE_QADR:
    d2 = (qpos_final[:, a + 0] - QINIT_NP[a + 0]) ** 2 + (qpos_final[:, a + 1] - QINIT_NP[a + 1]) ** 2
    base += 1.0 - np.exp(-d2 / SIG2)  # saturating: counts a cube as disturbed once it moves ~half a width
  apex_z = qpos_final[:, TOP_QADR + 2]
  adx = qpos_final[:, TOP_QADR + 0] - QINIT_NP[TOP_QADR + 0]
  ady = qpos_final[:, TOP_QADR + 1] - QINIT_NP[TOP_QADR + 1]
  standing = np.exp(-(adx ** 2 + ady ** 2) / SIG_A2)  # 1 if apex still in place, ~0 once knocked away
  return W_BASE * base / N_BASE + W_MISS * (apex_z / Z_TOP) ** 2 * standing


def _launch_state(vz, num_envs):
  """(qpos0 (E,nq), qvel0 (E,nv)): ball spawns at (BALL_X0,0,Z_START) from the XML init; launch velocity =
  forward VX + upward vz[e]; cubes at rest."""
  qpos = np.tile(QINIT_NP, (num_envs, 1)).astype(np.float32)
  qvel = np.zeros((num_envs, _MJM.nv), np.float32)
  qvel[:, BALL_V + 0] = VX
  qvel[:, BALL_V + 2] = vz
  return qpos, qvel


def _put(mjm, mjd, num_envs):
  """put_data with contact headroom sized to the scene (14 cubes rest on <=4 neighbors + floor, + the ball).
  ~96 contacts/env is plenty; oversizing nconmax/njmax balloons the per-step constraint solve on CPU."""
  ncon = num_envs * 256  # resting pyramid ~116 contacts + impact headroom; njmax tight to keep the solve fast
  return mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=ncon, njmax=4 * ncon)


def rollout(m, mjm, mjd, num_envs, vz):
  """Batched forward rollout with the ball thrown upward at vz (E,). Returns qpos traj (E, T+1, nq)."""
  d = _put(mjm, mjd, num_envs)
  qpos0, qvel0 = _launch_state(vz, num_envs)
  d.qpos = wp.array(qpos0, dtype=float)
  d.qvel = wp.array(qvel0, dtype=float)
  qs = [d.qpos.numpy().copy()]
  for _ in range(T):
    mjw.step(m, d)
    qs.append(d.qpos.numpy().copy())
  return np.transpose(np.array(qs), (1, 0, 2))  # (E, T+1, nq)


def fd_grad(m, mjm, mjd, num_envs, vz, eps=0.6):
  """Batched central difference d(loss_w)/d(vz_w): envs are independent worlds, so perturbing every env's
  upward-throw velocity at once and reading per-env loss gives all per-env gradients in TWO batched rollouts.
  eps is a fairly wide window -- light smoothing so descent skips micro-bumps of the discrete plow landscape."""
  lp = _per_env_loss(rollout(m, mjm, mjd, num_envs, vz + eps)[:, -1])
  lm = _per_env_loss(rollout(m, mjm, mjd, num_envs, vz - eps)[:, -1])
  return (lp - lm) / (2 * eps)


def analytic_grad(m, mjm, mjd, num_envs, vz_wp):
  """Batched taped rollout + adjoint.py backward -> populates vz_wp.grad. Returns (qpos (E,T+1,nq), loss).
  NOTE: the pyramid's box-box contacts are NOT AD-safe, so this gradient is biased (the stall this demo
  documents); compare it to fd_grad."""
  datas = [_put(mjm, mjd, num_envs) for _ in range(T + 1)]
  for d in datas:
    d.qpos.requires_grad = True
    d.qvel.requires_grad = True
  datas[0].qvel = wp.zeros((num_envs, _MJM.nv), dtype=float, requires_grad=True)
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  vz_wp.grad.zero_()
  tape = wp.Tape()
  with tape:
    wp.launch(_set_launch, dim=num_envs, inputs=[vz_wp, BALL_V, VX], outputs=[datas[0].qvel])
    for t in range(T):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_accum_base, dim=(num_envs, N_BASE), inputs=[datas[T].qpos, _QINIT_WP, _BASE_QADR_WP, _COEF_BASE, SIG2], outputs=[loss])
    wp.launch(_accum_apex, dim=num_envs, inputs=[datas[T].qpos, _QINIT_WP, TOP_QADR, _COEF_MISS, SIG_A2], outputs=[loss])
  tape.backward(loss=loss)
  qpos = np.transpose(np.array([datas[t].qpos.numpy() for t in range(T + 1)]), (1, 0, 2))
  return qpos, float(loss.numpy()[0])


def _init_vz(num_envs, spread):
  """Per-env initial upward-throw velocity. vz=0 is a FLAT throw at mid-height -> iteration 0 plows the middle
  (a bunch of cubes topple). GD raises vz (learns to throw UP) until the ball lobs onto the apex only. Multi-
  env fans vz from 0 (flat/plow) up past the clip basin so some lanes converge to the clean clip."""
  if num_envs == 1:
    return np.array([VZ_INIT], np.float64)
  # fan from a small upward throw that plows the base (vz~0.4) up to just past the clip basin, so the video
  # shows the base-plow lanes AND the lanes that descend to the clean apex-only clip (best lane wins).
  return np.clip(np.linspace(0.4, 2.9, num_envs), VZ_MIN, VZ_MAX)


def _fallen(qpos_final):
  """Per-env count of BASE cubes knocked more than 0.5*D off their initial (x,y) -- how much got plowed."""
  n = np.zeros(qpos_final.shape[0], int)
  for a in BASE_QADR:
    d2 = (qpos_final[:, a + 0] - QINIT_NP[a + 0]) ** 2 + (qpos_final[:, a + 1] - QINIT_NP[a + 1]) ** 2
    n += (d2 > (0.5 * D) ** 2).astype(int)
  return n


def _top_off(qpos_final):
  """Per-env: was the top cube knocked off (planar displacement > 0.5*D)?"""
  d2 = (qpos_final[:, TOP_QADR + 0] - QINIT_NP[TOP_QADR + 0]) ** 2 + (qpos_final[:, TOP_QADR + 1] - QINIT_NP[TOP_QADR + 1]) ** 2
  return d2 > (0.5 * D) ** 2


def optimize(num_envs, spread, grad_mode, steps, lr):
  mjm = mujoco.MjModel.from_xml_string(pyramid_xml())
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)

  vz = _init_vz(num_envs, spread)
  vz_wp = wp.array(vz.astype(np.float32), dtype=float, requires_grad=True)
  opt = warp.optim.Adam([vz_wp], lr=lr, betas=(0.7, 0.95))

  history = []
  for it in range(steps):
    vnp = vz_wp.numpy().astype(np.float64).copy()
    if grad_mode == "analytic":
      qpos, L = analytic_grad(m, mjm, mjd, num_envs, vz_wp)
      g = np.nan_to_num(vz_wp.grad.numpy().astype(np.float64).copy())
    else:
      qpos = rollout(m, mjm, mjd, num_envs, vnp)
      L = float(_per_env_loss(qpos[:, -1]).sum())
      # coarse-to-fine FD step: start wide to leap the flat base=4 plateau (little local gradient), shrink to
      # settle inside the clip basin without overshooting into the miss region.
      eps_it = max(0.4, 1.4 - 0.05 * it)
      g = np.nan_to_num(fd_grad(m, mjm, mjd, num_envs, vnp, eps=eps_it))

    per_env = _per_env_loss(qpos[:, -1])
    fallen = _fallen(qpos[:, -1])
    top_off = _top_off(qpos[:, -1])
    rec = {"it": it, "loss": L, "qpos": qpos, "vz": vnp, "per_env": per_env, "fallen": fallen, "top_off": top_off}

    # analytic-vs-FD stall check: the box-box adjoint has NO gradient, so the analytic grad is wrong. vz is a
    # SCALAR per env, so cosine is degenerate (+-1) -- we report per-env SIGN agreement + |analytic|/|fd|. Only
    # for small env counts (the batched analytic backward allocates T+1 Data and is expensive for many worlds).
    check = (it % 8 == 0 or it == steps - 1) and num_envs <= 2
    if check:
      analytic_grad(m, mjm, mjd, num_envs, vz_wp) if grad_mode == "fd" else None
      an = np.nan_to_num(vz_wp.grad.numpy().astype(np.float64).copy()) if grad_mode == "fd" else g
      fd = g if grad_mode == "fd" else np.nan_to_num(fd_grad(m, mjm, mjd, num_envs, vnp))
      rec["agree"] = int(np.sum(np.sign(an) == np.sign(fd)))
      rec["ratio"] = float(np.linalg.norm(an) / (np.linalg.norm(fd) + 1e-12))
    history.append(rec)
    if it % 8 == 0 or it == steps - 1:
      tag = f"  [box-box stall] analytic sign agrees {rec['agree']}/{num_envs}, |an|/|fd|={rec['ratio']:.1f}" if "agree" in rec else ""
      print(f"  [{it:3d}] loss={L:.3f} vz={np.round(vnp, 2).tolist()} plowed={fallen.tolist()}/{N_BASE}"
            f" top_off={top_off.astype(int).tolist()} |g|={np.linalg.norm(g):.3g}{tag}")

    vz_wp.grad = wp.array(g.astype(np.float32), dtype=float)  # descend the CHOSEN mode's gradient
    opt.step([vz_wp.grad])
    vz_wp.assign(np.clip(vz_wp.numpy(), VZ_MIN, VZ_MAX).astype(np.float32))  # projected upward-throw bounds

  best = min(range(len(history)), key=lambda k: history[k]["loss"])
  hb = history[best]
  print(f"[pyramid x{num_envs}env/{grad_mode}] loss {history[0]['loss']:.4f} -> best {hb['loss']:.4f} "
        f"(iter {best}); best plowed {hb['fallen'].tolist()}/{N_BASE}, top_off {hb['top_off'].astype(int).tolist()}, "
        f"vz {np.round(hb['vz'], 2).tolist()}")
  return history, best


# --- rendering: a grid of pyramids (one lane per env), each env's top-cube path traced, loss-colored -----


def _env_offsets(num_envs, cols=ENV_COLS):
  """Lay the lanes out in a single LINE along y (perpendicular to the ball's +x travel), so each lane's ball
  and pyramid never overlap a neighbor (they only need to clear the pyramid WIDTH, not the ball's x-reach)."""
  off = np.zeros((num_envs, 3))
  if num_envs > 1:
    pitch = BASE * SPACING + 0.7  # pyramid width + gap
    for e in range(num_envs):
      off[e, 1] = (e - (num_envs - 1) / 2.0) * pitch
  return off


def viz_xml_multi(offsets):
  """Viz scene: per env, NCUBE decorative box cubes + 1 ball (contype=0), placed each frame from the rollout
  qpos. Env-major body order (env outer, body inner) matches the flattened batched qpos."""
  opt = '<option timestep="%g" gravity="0 0 -9.81"/>' % DT
  bodies = ""
  for w in range(len(offsets)):
    for i in range(NCUBE):
      rgba = "0.95 0.55 0.2 1" if i == TOP_C else "0.55 0.6 0.7 1"
      bodies += (
        f'<body name="e{w}c{i}" pos="0 0 {CUBE_HALF:.4f}"><freejoint/>'
        f'<geom type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" mass="{CUBE_MASS}" '
        f'contype="0" conaffinity="0" rgba="{rgba}"/></body>'
      )
    bodies += (
      f'<body name="e{w}ball" pos="0 0 {Z_START:.4f}"><freejoint/>'
      f'<geom type="sphere" size="{BALL_R}" mass="{BALL_MASS}" contype="0" conaffinity="0" rgba="0.95 0.85 0.2 1"/></body>'
    )
  fsize = max(3.0, float(np.abs(offsets).max()) + BASE * SPACING + 1.0)
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
  vm = mujoco.MjModel.from_xml_string(viz_xml_multi(offsets))
  vd = mujoco.MjData(vm)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(vm, cam)
  cam.lookat = [0.0, 0.0, Z_TOP * 0.5]
  gridext = float(np.abs(offsets).max()) if num_envs > 1 else 0.0
  cam.distance = 1.6 + 2.5 * gridext + BASE * SPACING
  cam.azimuth = 130.0  # 3/4 iso view so the pyramid reads as 3D; the ball flies in +x
  cam.elevation = -20.0

  hi = max(float(max(h["per_env"].max() for h in history)), 1e-9)

  def flat_qpos(hk, t):
    q = np.zeros(vm.nq)
    for w in range(num_envs):
      for i in range(NCUBE + 1):  # cubes then ball
        s = (w * (NCUBE + 1) + i) * 7
        q[s:s + 7] = hk["qpos"][w, t, 7 * i:7 * i + 7]
        q[s:s + 3] += offsets[w]
    return q

  frames, persisted = [], []
  for k in R.default_show(len(history), best):
    hk = history[k]
    cols = [R.bourke_color_map(0.0, hi, float(hk["per_env"][w])) for w in range(num_envs)]
    paths = [hk["qpos"][w, :, 7 * TOP_C:7 * TOP_C + 3] + offsets[w] for w in range(num_envs)]  # top-cube path
    hold = 20 if k == best else 0
    sub = (f"iter {hk['it']:3d}    mean loss/env {hk['loss'] / num_envs:.4f}    "
           f"base plowed {hk['fallen'].tolist()}/{N_BASE}    envs {num_envs}    ({grad_mode})")
    for t in list(range(0, T + 1, 3)) + [T] * (1 + hold):
      snap, cur = list(persisted), [p[: t + 1] for p in paths]

      def draw(scene, snap=snap, cur=cur, cols=cols):
        for pr, cr in snap:
          for pth, cc in zip(pr, cr):
            R.add_polyline(scene, pth, cc, width=0.005)
        for pth, cc in zip(cur, cols):
          R.add_polyline(scene, pth, cc, width=0.014)

      frames.append((flat_qpos(hk, t), draw, sub))
    persisted.append((paths, cols))
  if frames:
    frames += [frames[-1]] * 20
  label = f"pyramid knock-top x{num_envs} ({grad_mode})"
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
  print(f"[pyramid] video -> {out_path}\n[pyramid] montage -> {png}")


def default_out():
  return os.path.join(os.path.dirname(__file__), "reports", "assets", "pyramid.mp4")


def settle_test(num_envs=1):
  """Sanity: with the ball launched OVER everything (misses), the pyramid must stay put (cube drift small)."""
  mjm = mujoco.MjModel.from_xml_string(pyramid_xml())
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  vz = np.full(num_envs, VZ_MAX)  # big upward throw -> ball sails over everything (misses)
  qf = rollout(m, mjm, mjd, num_envs, vz)[:, -1]
  drift = np.zeros(num_envs)
  for a in BASE_QADR + [TOP_QADR]:
    drift = np.maximum(drift, np.sqrt((qf[:, a + 0] - QINIT_NP[a + 0]) ** 2 + (qf[:, a + 1] - QINIT_NP[a + 1]) ** 2))
  print(f"[pyramid settle] big upward throw (vz={VZ_MAX:.1f}) sails over; max cube (x,y) drift = {drift.max():.4f} m (want << {D:.3f})")


def render_scene(out_path):
  """Render a single still of the initial scene (pyramid + parked ball) -- for eyeballing before optimizing."""
  vm = mujoco.MjModel.from_xml_string(viz_xml_multi(np.zeros((1, 3))))
  vd = mujoco.MjData(vm)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(vm, cam)
  cam.lookat = [0.0, 0.0, Z_TOP * 0.5]
  cam.distance = 1.6 + BASE * SPACING
  cam.azimuth = 130.0
  cam.elevation = -20.0
  # place the 1-env scene at the physics init layout (cubes + parked ball)
  q = np.zeros(vm.nq)
  for i in range(NCUBE):
    q[7 * i:7 * i + 7] = QINIT_NP[7 * i:7 * i + 7]
  q[7 * NCUBE:7 * NCUBE + 7] = QINIT_NP[BALL_Q:BALL_Q + 7]
  vd.qpos[:] = q
  mujoco.mj_forward(vm, vd)
  import imageio.v2 as imageio
  with mujoco.Renderer(vm, height=H, width=W) as r:
    r.update_scene(vd, camera=cam)
    imageio.imwrite(out_path, r.render())
  print(f"[pyramid] scene still -> {out_path}")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--grad", choices=["fd", "analytic"], default="fd",
                  help="optimizer gradient: 'fd' (robust; box-box adjoint stalls) or 'analytic' (documents the stall)")
  ap.add_argument("--envs", type=int, default=1)
  ap.add_argument("--spread", type=float, default=SPREAD, help="per-env initial launch-height fan (multi-env only)")
  ap.add_argument("--steps", type=int, default=STEPS)
  ap.add_argument("--lr", type=float, default=LR)
  ap.add_argument("--out", default=None)
  ap.add_argument("--no-render", action="store_true")
  ap.add_argument("--live", action="store_true", help="live MuJoCo viewer instead of mp4")
  ap.add_argument("--settle-test", action="store_true", help="just check the pyramid is stable at rest")
  ap.add_argument("--scene-still", default=None, help="render a single still of the initial scene to this path and exit")
  args = ap.parse_args()

  if args.scene_still:
    render_scene(args.scene_still)
    return
  if args.settle_test:
    settle_test(args.envs)
    return

  history, best = optimize(args.envs, args.spread, args.grad, args.steps, args.lr)
  if args.no_render:
    return
  out = args.out or os.environ.get("MJW_RENDER_PATH") or default_out()
  os.makedirs(os.path.dirname(out), exist_ok=True)
  render(history, best, args.envs, args.grad, out, live=args.live)


if __name__ == "__main__":
  main()
