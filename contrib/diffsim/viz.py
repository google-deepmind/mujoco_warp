"""Shared visualization for optimization-trajectory demos.

An optimization "history" is a list of per-iteration dicts:
    {"it": int, "loss": float, "xyz": (T+1,3) tracked-point trajectory, "qpos": (T+1,nq)}
`render_video` animates each shown iteration's rollout (set qpos -> mj_forward -> draw), traces that
iteration's tracked path as a Bourke-loss-colored polyline (blue=converged -> red=bad), persists prior
iterations' paths, and overlays a banner.

OUTPUT MODE (video vs live viewer) is chosen by `emit`:
  - default / `MJW_VIEWER=0`: write an mp4 (offscreen `mujoco.Renderer`, with the banner overlay).
  - `MJW_VIEWER=1` (or `live=True`): open the **live MuJoCo passive viewer** and loop the animation
    (decorative geoms drawn into `viewer.user_scn`); no banner, runs until the window is closed.
Pass `live=` explicitly to override the env var. Custom multi-env demos build their own frame list
and call `emit` directly (same toggle).

Scene scripts (bounce.py, reach.py, …) supply the physics/optimizer + a lit viz MjModel; this module
is scene-agnostic.
"""

import os

import numpy as np


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


def add_polyline(scene, pts, rgba, width=0.012, alpha=1.0):
  """Append a colored polyline (capsule segments) to an MjvScene (alpha-aware)."""
  import mujoco

  rgba = np.array([rgba[0], rgba[1], rgba[2], alpha], dtype=np.float32)
  for a, b in zip(pts[:-1], pts[1:]):
    if scene.ngeom >= scene.maxgeom:
      break
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a.astype(np.float64), b.astype(np.float64))
    scene.ngeom += 1


def add_sphere(scene, pos, rgba, size=0.1):
  """Append a solid sphere geom at `pos` to an MjvScene."""
  import mujoco

  if scene.ngeom >= scene.maxgeom:
    return
  g = scene.geoms[scene.ngeom]
  rgba = np.array([rgba[0], rgba[1], rgba[2], 1.0], dtype=np.float32)
  mujoco.mjv_initGeom(
    g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, size, size]), pos.astype(np.float64), np.eye(3).flatten(), rgba
  )
  scene.ngeom += 1


def add_disk(scene, pos, rgba, radius=0.15, alpha=0.35, thickness=0.003):
  """Append a thin translucent disk (a decorative cylinder lying flat, axis along world z) centered at
  `pos`. Built with mjv_connector -- the same well-defined path add_polyline uses -- so it doesn't depend
  on the cylinder size[] layout. Decorative geoms cast NO shadow: marks a target plane without shadowing."""
  import mujoco

  if scene.ngeom >= scene.maxgeom:
    return
  g = scene.geoms[scene.ngeom]
  rgba = np.array([rgba[0], rgba[1], rgba[2], alpha], dtype=np.float32)
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
  pos = np.asarray(pos, dtype=np.float64)
  lo = pos + np.array([0.0, 0.0, -0.5 * thickness])
  hi = pos + np.array([0.0, 0.0, 0.5 * thickness])
  mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CYLINDER, radius, lo, hi)  # width = disk radius
  scene.ngeom += 1


def make_overlay(w, h, label):
  """Returns overlay(frame_rgb, subtext) -> frame_rgb with a top banner: big label + subtext."""
  from PIL import Image, ImageDraw, ImageFont

  def _font(sz):
    for p in (
      "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
      "/System/Library/Fonts/Helvetica.ttc",
      "/Library/Fonts/Arial.ttf",
    ):
      try:
        return ImageFont.truetype(p, sz)
      except Exception:
        pass
    return ImageFont.load_default()

  big, small = _font(48), _font(26)

  def overlay(arr, subtext):
    img = Image.fromarray(arr)
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, w, 92], fill=(0, 0, 0))
    dr.text((24, 6), label, fill=(255, 235, 59), font=big)
    dr.text((26, 62), subtext, fill=(210, 210, 210), font=small)
    return np.asarray(img)

  return overlay


def default_show(n, best):
  """Iterations to display: dense early (big changes), through `best`, to the final step."""
  show = sorted(set([0, 1, 2, 4, 7, 11, 16, 24, 36, 54, 80, 110, 140, best, n - 1]))
  return [k for k in show if 0 <= k < n]


def _set_cam(dst, cam):
  dst.lookat[:] = cam.lookat
  dst.distance, dst.azimuth, dst.elevation = cam.distance, cam.azimuth, cam.elevation


def emit(viz_model, vd, cam, frames, *, out_path=None, label="", w=1024, h=768, fps=30, live=None):
  """Render a list of frames either to an mp4 or to the live MuJoCo viewer.

  `frames` is a list of `(qpos | None, draw, subtext)`: per frame we set `vd.qpos` (if given),
  `mj_forward`, and call `draw(scene)` to append decorative geoms. `live` defaults to the
  `MJW_VIEWER` env var (1 -> live viewer, else mp4).
  """
  import mujoco

  if live is None:
    live = os.environ.get("MJW_VIEWER", "0") == "1"

  if live:
    import time

    import mujoco.viewer

    with mujoco.viewer.launch_passive(viz_model, vd, show_left_ui=False, show_right_ui=False) as v:
      _set_cam(v.cam, cam)
      while v.is_running():  # loop the animation until the window is closed
        for qpos, draw, _sub in frames:
          if not v.is_running():
            break
          if qpos is not None:
            vd.qpos[:] = qpos
          mujoco.mj_forward(viz_model, vd)
          v.user_scn.ngeom = 0
          draw(v.user_scn)
          v.sync()
          time.sleep(1.0 / fps)
    return None

  import imageio.v2 as imageio

  overlay = make_overlay(w, h, label)
  out = []
  with mujoco.Renderer(viz_model, height=h, width=w, max_geom=30000) as r:
    if frames:  # warmup render: stabilize the first-frame reflection/shadow buffer
      q0 = frames[0][0]
      if q0 is not None:
        vd.qpos[:] = q0
      mujoco.mj_forward(viz_model, vd)
      r.update_scene(vd, camera=cam)
      r.render()
    for qpos, draw, sub in frames:
      if qpos is not None:
        vd.qpos[:] = qpos
      mujoco.mj_forward(viz_model, vd)
      r.update_scene(vd, camera=cam)
      draw(r.scene)
      out.append(overlay(r.render(), sub))
  imageio.mimsave(out_path, out, fps=fps, quality=8, macro_block_size=8)
  print(f"[viz] wrote {out_path}  ({len(out)} frames, {len(out) / fps:.1f}s)")
  return out_path


def render_video(
  viz_model, history, *, cam, n_steps, out_path, label, w=1024, h=768, fps=30, bourke_hi=None,
  sample_every=4, best=None, show=None, traj_width=0.014, persist_width=0.008, subtext=None, live=None,
):
  """Animate an optimization history (single tracked trajectory per iteration) to mp4 or live viewer.
  `history[k]` needs keys it/loss/xyz/qpos (see module doc). `subtext(h, best)` -> str customizes the
  banner. Set `live=True` (or `MJW_VIEWER=1`) for the live viewer instead of an mp4."""
  import mujoco

  vd = mujoco.MjData(viz_model)
  if best is None:
    best = min(range(len(history)), key=lambda k: history[k]["loss"])
  if show is None:
    show = default_show(len(history), best)
  if bourke_hi is None:
    bourke_hi = float(history[0]["loss"])  # iter-0 loss = worst-ish -> red
  if subtext is None:
    def subtext(hk, b):
      return f"iter {hk['it']:3d}    loss {hk['loss']:.4f}    best {history[b]['loss']:.4f}"

  frames = []
  persisted = []  # (xyz, rgba) of completed iterations
  for k in show:
    hk = history[k]
    rgba = bourke_color_map(0.0, bourke_hi, hk["loss"])
    steps_idx = list(range(0, n_steps + 1, sample_every)) + [n_steps]
    hold = 20 if k == best else 0  # linger on the converged shot
    sub = subtext(hk, best)
    for t in steps_idx + [n_steps] * hold:
      snap = list(persisted)  # snapshot so each frame's draw shows the persisted set at that time
      cur = hk["xyz"][: t + 1]

      def draw(scene, snap=snap, cur=cur, rgba=rgba):
        for xyz_p, c in snap:
          add_polyline(scene, xyz_p, c, width=persist_width)
        add_polyline(scene, cur, rgba, width=traj_width)

      frames.append((hk["qpos"][t], draw, sub))
    persisted.append((hk["xyz"], rgba))
  if frames:
    frames += [frames[-1]] * 20  # final hold

  return emit(viz_model, vd, cam, frames, out_path=out_path, label=label, w=w, h=h, fps=fps, live=live)
