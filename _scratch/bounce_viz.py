"""Animate the bounce optimization (Newton example_diffsim_ball style) as a video.

Replicates Newton's render() loop: across optimization iterations, the ball is animated
through its rollout (so you watch it bounce off the FLOOR and WALL), each iteration's
trajectory drawn as a polyline colored by its loss via the Bourke colormap (red=bad ->
blue=converged), with prior iterations' trajectories persisting.

Physics matches Newton (soft_contact_restitution=1.0, mu=0.2): elastic contact via NEGATIVE
solref (MuJoCo has no restitution coeff), low friction, and LOCAL gradient descent from the
initial velocity so the optimizer stays in the wall-bounce basin (not the contact-free arc).

Uses the FD gradient (analytic tape grad still WIP). Newton logs to USD; here we write mp4.
"""

import os

import imageio.v2 as imageio
import mujoco
import numpy as np

# Newton example_diffsim_ball numbers (Z-up).
START = (0.0, -0.5, 1.0)
QVEL0 = (0.0, 5.0, -5.0, 0.0, 0.0, 0.0)
TARGET = (0.0, -2.0, 1.5)
T = 150  # steps @ dt=0.004 -> 0.6s
SOLREF = "-3000 -2"  # elastic-ish: direct (stiffness, damping), low damping -> bounce
# condim/friction env-configurable: condim 6 + nonzero torsion/roll exercises the rotational contact
# rows. geom friction is [slide, torsion, roll]; default = condim 3, slide-only (original behavior).
CONDIM = os.environ.get("MJW_BOUNCE_CONDIM", "3")
FRICTION = os.environ.get("MJW_BOUNCE_FRICTION", "0.2")
OUT_MP4 = os.environ.get("MJW_RENDER_PATH", "/tmp/bounce_optim.mp4")
W, H, FPS = 1024, 768, 30


def bourke_color_map(low, high, v):
  """Newton's newton.utils.bourke_color_map: blue(low) -> cyan -> green -> yellow -> red(high)."""
  c = [1.0, 1.0, 1.0]
  dv = high - low
  v = min(max(v, low), high)
  if v < low + 0.25 * dv:
    c[0] = 0.0
    c[1] = 4.0 * (v - low) / dv
  elif v < low + 0.5 * dv:
    c[0] = 0.0
    c[2] = 1.0 + 4.0 * (low + 0.25 * dv - v) / dv
  elif v < low + 0.75 * dv:
    c[0] = 4.0 * (v - low - 0.5 * dv) / dv
    c[2] = 0.0
  else:
    c[1] = 1.0 + 4.0 * (low + 0.75 * dv - v) / dv
    c[2] = 0.0
  return c


def bounce_xml(physics_only=True):
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


def rollout(qvel0):
  """mj_step rollout. Returns (xyz[T+1,3], qpos[T+1,7], loss, hit_floor, hit_wall)."""
  m = mujoco.MjModel.from_xml_string(bounce_xml())
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
  d.qvel[:6] = qvel0
  xyz = [d.xpos[bid].copy()]
  qpos = [d.qpos.copy()]
  for _ in range(T):
    mujoco.mj_step(m, d)
    xyz.append(d.xpos[bid].copy())
    qpos.append(d.qpos.copy())
  xyz, qpos = np.array(xyz), np.array(qpos)
  loss = float(np.sum((xyz[-1] - np.array(TARGET)) ** 2))
  return xyz, qpos, loss, xyz[:, 2].min() < 0.13, xyz[:, 1].max() > 1.6


def fd_grad(qvel0, eps=1e-3):
  g = np.zeros(3)
  for i in range(3):
    vp = qvel0.copy(); vp[i] += eps
    vm = qvel0.copy(); vm[i] -= eps
    g[i] = (rollout(vp)[2] - rollout(vm)[2]) / (2 * eps)
  return g


def optimize(steps=160, rate=0.05):
  """LOCAL gradient descent from init vel (Newton's approach). Returns per-iteration history."""
  qvel = np.array(QVEL0, dtype=np.float64)
  history = []
  for it in range(steps):
    xyz, qpos, loss, hf, hw = rollout(qvel)
    history.append({"it": it, "qvel": qvel[:3].copy(), "loss": loss, "xyz": xyz, "qpos": qpos})
    if it % 20 == 0:
      print(f"  [{it:4d}] loss={loss:.4f} floor={hf} wall={hw}")
    qvel[:3] -= rate * fd_grad(qvel)[:3]
  best = min(range(len(history)), key=lambda k: history[k]["loss"])
  print(f"[optim] loss {history[0]['loss']:.3f} -> best {history[best]['loss']:.3f} (iter {best})")
  return history, best


def _add_polyline(scene, pts, rgba, width=0.012):
  """Append a colored polyline (capsule segments) to an MjvScene."""
  rgba = np.array([*rgba, 1.0], dtype=np.float32)
  for a, b in zip(pts[:-1], pts[1:]):
    if scene.ngeom >= scene.maxgeom:
      break
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.eye(3).flatten(), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         a.astype(np.float64), b.astype(np.float64))
    scene.ngeom += 1


def _make_overlay(label):
  """Returns fn(frame_rgb, subtext) -> frame_rgb with a top banner: big method label + subtext."""
  from PIL import Image, ImageDraw, ImageFont

  def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
      try:
        return ImageFont.truetype(p, sz)
      except Exception:
        pass
    return ImageFont.load_default()

  big, small = _font(48), _font(26)

  def overlay(arr, subtext):
    img = Image.fromarray(arr)
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, W, 92], fill=(0, 0, 0))
    dr.text((24, 6), label, fill=(255, 235, 59), font=big)
    dr.text((26, 62), subtext, fill=(210, 210, 210), font=small)
    return np.asarray(img)

  return overlay


def render_video(history, best, sample_every=4, label="FINITE DIFF"):
  overlay = _make_overlay(label)
  vm = mujoco.MjModel.from_xml_string(bounce_xml(physics_only=False))
  vd = mujoco.MjData(vm)
  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(vm, cam)
  cam.lookat = [0.0, 0.0, 0.8]
  cam.distance = 8.5
  cam.azimuth = 50.0
  cam.elevation = -18.0

  # iterations to display: dense early (big changes), through the best, to the final step
  last = len(history) - 1
  show = sorted(set([0, 1, 2, 4, 7, 11, 16, 24, 36, 54, 80, 110, 140, best, last]))
  show = [k for k in show if k < len(history)]
  lo, hi = 0.0, 7.0  # Newton's bourke range
  persisted = []  # (xyz, rgba) of completed iterations

  frames = []
  with mujoco.Renderer(vm, height=H, width=W) as r:
    for k in show:
      h = history[k]
      rgba = bourke_color_map(lo, hi, h["loss"])
      steps_idx = list(range(0, T + 1, sample_every)) + [T]
      hold = 20 if k == best else 0  # linger on the converged shot
      for fi, t in enumerate(steps_idx + [T] * hold):
        vd.qpos[:] = h["qpos"][t]
        mujoco.mj_forward(vm, vd)
        r.update_scene(vd, camera=cam)
        for xyz_p, c in persisted:  # prior trajectories stay on screen
          _add_polyline(r.scene, xyz_p, c, width=0.008)
        _add_polyline(r.scene, h["xyz"][: t + 1], rgba, width=0.014)  # growing trail
        sub = f"iter {h['it']:3d}    loss {h['loss']:.3f}    best {history[best]['loss']:.3f}"
        frames.append(overlay(r.render(), sub))
      persisted.append((h["xyz"], rgba))
    # final hold on the full converged scene
    for _ in range(20):
      frames.append(frames[-1])

  imageio.mimsave(OUT_MP4, frames, fps=FPS, quality=8, macro_block_size=8)
  print(f"[bounce video] wrote {OUT_MP4}  ({len(frames)} frames, {len(frames)/FPS:.1f}s)")


def main():
  history, best = optimize()
  render_video(history, best)


if __name__ == "__main__":
  main()
