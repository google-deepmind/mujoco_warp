# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""mjwarp-render: render an RGB and/or depth image from an MJCF.

Usage: mjwarp-render <mjcf XML path> [flags]

Example:
  mjwarp-render benchmarks/humanoid/humanoid.xml --nworld=1 --cam=0 --width=512 --height=512
  mjwarp-render benchmarks/humanoid/humanoid.xml --splat=my_scene.ply
"""

import sys
from typing import Sequence

import mujoco
import numpy as np
import warp as wp
from absl import app
from absl import flags
from etils import epath
from PIL import Image

import mujoco_warp as mjw
from mujoco_warp._src.io import override_model

_NWORLD = flags.DEFINE_integer("nworld", 1, "number of parallel worlds")
_WORLD = flags.DEFINE_integer("world", 0, "world index to save from")
_CAM = flags.DEFINE_integer("cam", 0, "camera index to render")
_WIDTH = flags.DEFINE_integer("width", 512, "render width (pixels)")
_HEIGHT = flags.DEFINE_integer("height", 512, "render height (pixels)")
_RENDER_RGB = flags.DEFINE_bool("rgb", True, "render RGB image")
_RENDER_DEPTH = flags.DEFINE_bool("depth", True, "render depth image")
_RENDER_SEG = flags.DEFINE_bool("seg", False, "render segmentation image")
_USE_TEXTURES = flags.DEFINE_bool("textures", True, "use textures")
_USE_SHADOWS = flags.DEFINE_bool("shadows", False, "use shadows")
_RENDER_SKYBOX = flags.DEFINE_bool("skybox", False, "render skybox")
_DEVICE = flags.DEFINE_string("device", None, "override the default Warp device")
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "clear Warp kernel cache before rendering")
_OVERRIDE = flags.DEFINE_multi_string("override", [], "Model overrides (notation: foo.bar = baz)", short_name="o")
_OUTPUT_RGB = flags.DEFINE_string("output_rgb", "debug.png", "output path for RGB image")
_OUTPUT_DEPTH = flags.DEFINE_string("output_depth", "debug_depth.png", "output path for depth image")
_DEPTH_SCALE = flags.DEFINE_float("depth_scale", 5.0, "scale factor to map depth to 0..255 for preview")
_TILED = flags.DEFINE_bool("tiled", False, "render a 4x4 tiled grid across 16 worlds at 512x512")
_ROLLOUT = flags.DEFINE_bool("rollout", False, "render a rollout video instead of a single frame")
_ORBIT = flags.DEFINE_bool("orbit", False, "render one camera orbit instead of a single frame")
_ORBIT_CENTER = flags.DEFINE_list("orbit_center", ["0", "0", "1"], "orbit target as x,y,z")
_ORBIT_RADIUS = flags.DEFINE_float("orbit_radius", 4.0, "orbit radius")
_ORBIT_HEIGHT = flags.DEFINE_float("orbit_height", 1.0, "camera height above the orbit target")
_ORBIT_UP = flags.DEFINE_enum("orbit_up", "z", ["x", "y", "z", "-x", "-y", "-z"], "up axis for the camera orbit")
_NSTEPS = flags.DEFINE_integer("nstep", 128, "simulation steps, or frames in an orbit")
_ROLLOUT_OUTPUT = flags.DEFINE_string("output_video", "rollout.gif", "output path for rollout video")
_SPLAT = flags.DEFINE_string("splat", None, "3DGS PLY path to load as splat attributes")

# Map PLY scalar names to NumPy dtypes while reading the binary 3DGS vertex payload.
_PLY_TYPES = {
  "char": "i1",
  "int8": "i1",
  "uchar": "u1",
  "uint8": "u1",
  "short": "<i2",
  "int16": "<i2",
  "ushort": "<u2",
  "uint16": "<u2",
  "int": "<i4",
  "int32": "<i4",
  "uint": "<u4",
  "uint32": "<u4",
  "float": "<f4",
  "float32": "<f4",
  "double": "<f8",
  "float64": "<f8",
}
# Convert degree-zero spherical-harmonic coefficients to RGB values.
_SH_C0 = 0.28209479177387814


def _load_ply_splats(filename) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Loads splat attributes from a binary little-endian 3DGS PLY file."""
  with open(filename, "rb") as file:
    if file.readline().strip() != b"ply":
      raise ValueError("not a PLY file")

    vertex_count = None
    vertex_properties = []
    in_vertex = False
    binary_little_endian = False
    while True:
      line = file.readline()
      if not line:
        raise ValueError("PLY header is missing end_header")
      fields = line.decode("ascii").strip().split()
      if fields[:2] == ["format", "binary_little_endian"]:
        binary_little_endian = True
      elif fields and fields[0] == "format":
        raise ValueError("only binary little-endian PLY files are supported")
      elif fields[:2] == ["element", "vertex"]:
        vertex_count = int(fields[2])
        in_vertex = True
      elif fields and fields[0] == "element":
        in_vertex = False
      elif fields and fields[0] == "property" and in_vertex:
        if len(fields) != 3 or fields[1] not in _PLY_TYPES:
          raise ValueError("unsupported PLY vertex property")
        vertex_properties.append((fields[2], _PLY_TYPES[fields[1]]))
      elif fields == ["end_header"]:
        break

    if vertex_count is None:
      raise ValueError("PLY file has no vertex element")
    if not binary_little_endian:
      raise ValueError("PLY header is missing its binary little-endian format")
    vertices = np.fromfile(file, dtype=np.dtype(vertex_properties), count=vertex_count)
    if len(vertices) != vertex_count:
      raise ValueError("PLY vertex data is truncated")

  names = vertices.dtype.names or ()

  def fields(prefix, count):
    required = [f"{prefix}_{i}" for i in range(count)]
    if not all(name in names for name in required):
      raise ValueError(f"PLY file is missing {prefix} splat attributes")
    return np.column_stack([vertices[name] for name in required]).astype(np.float32)

  if not all(name in names for name in ("x", "y", "z", "opacity")):
    raise ValueError("PLY file is missing required splat attributes")

  positions = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
  rotations_wxyz = fields("rot", 4)
  rotations = rotations_wxyz[:, [1, 2, 3, 0]]
  rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1.0e-12)
  scales = np.exp(fields("scale", 3))
  opacity = 1.0 / (1.0 + np.exp(-np.clip(vertices["opacity"], -80.0, 80.0)))
  color = np.clip(0.5 + _SH_C0 * fields("f_dc", 3), 0.0, 1.0)
  rgba = np.column_stack((color, opacity)).astype(np.float32)
  return positions, rotations, scales, rgba


def _load_model(path: epath.Path) -> mujoco.MjModel:
  if not path.exists():
    resource_path = epath.resource_path("mujoco_warp") / path
    if not resource_path.exists():
      raise FileNotFoundError(f"file not found: {path}\nalso tried: {resource_path}")
    path = resource_path

  print(f"Loading model from: {path}...")
  if path.suffix == ".mjb":
    return mujoco.MjModel.from_binary_path(path.as_posix())

  spec = mujoco.MjSpec.from_file(path.as_posix())
  # register SDF test plugins if present
  if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
    from mujoco_warp.test_data.collision_sdf.utils import register_sdf_plugins as register_sdf_plugins

    register_sdf_plugins(mjw)

  return spec.compile()


def _save_rgb_from_packed(packed_row: np.ndarray, width: int, height: int, out_path: str):
  packed = packed_row.reshape(height, width).astype(np.uint32)
  b = (packed & 0xFF).astype(np.uint8)
  g = ((packed >> 8) & 0xFF).astype(np.uint8)
  r = ((packed >> 16) & 0xFF).astype(np.uint8)
  img = Image.fromarray(np.dstack([r, g, b]))
  img.save(out_path)


def _save_depth(depth_row: np.ndarray, width: int, height: int, scale: float, out_path: str):
  arr = depth_row.reshape(height, width)
  arr = np.clip(arr / max(scale, 1e-6), 0.0, 1.0)
  img = Image.fromarray((arr * 255.0).astype(np.uint8))
  img.save(out_path)


def _rgb_image_from_packed(packed_row: np.ndarray, width: int, height: int) -> np.ndarray:
  """Convert a packed uint32 row into an (H, W, 3) uint8 RGB array."""
  packed = packed_row.reshape(height, width).astype(np.uint32)
  b = (packed & 0xFF).astype(np.uint8)
  g = ((packed >> 8) & 0xFF).astype(np.uint8)
  r = ((packed >> 16) & 0xFF).astype(np.uint8)
  return np.dstack([r, g, b])


def _depth_image_from_row(depth_row: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
  """Convert a depth row into an (H, W) uint8 array using the given scale."""
  arr = depth_row.reshape(height, width)
  arr = np.clip(arr / max(scale, 1e-6), 0.0, 1.0)
  return (arr * 255.0).astype(np.uint8)


def _rgb_frame(
  rc: mjw.RenderContext,
  cam: int,
  world: int,
  width: int,
  height: int,
  grid_rows: int,
  grid_cols: int,
) -> np.ndarray | None:
  rgb_adr = rc.rgb_adr.numpy()
  if rgb_adr[cam] == -1:
    return None

  rgb = rc.rgb_data.numpy()
  start = rgb_adr[cam]
  rows = rgb[:, start : start + width * height]
  if grid_rows == 1 and grid_cols == 1:
    return _rgb_image_from_packed(rows[world], width, height)

  tiles = [_rgb_image_from_packed(rows[i], width, height) for i in range(grid_rows * grid_cols)]
  tiled_rows = [np.concatenate(tiles[i * grid_cols : (i + 1) * grid_cols], axis=1) for i in range(grid_rows)]
  return np.concatenate(tiled_rows, axis=0)


def _set_orbit_camera(
  d: mjw.Data,
  cam: int,
  angle: float,
  center: np.ndarray,
  radius: float,
  height: float,
  up_axis: str,
):
  axis = up_axis[-1]
  up = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
  }[axis]
  if up_axis[0] == "-":
    up = -up
  radial_0 = {
    "x": np.array([0.0, -1.0, 0.0]),
    "y": np.array([0.0, 0.0, -1.0]),
    "z": np.array([0.0, -1.0, 0.0]),
  }[axis]
  radial_1 = np.cross(radial_0, up)
  position = center + radius * (radial_0 * np.cos(angle) + radial_1 * np.sin(angle)) + height * up
  forward = center - position
  forward /= np.linalg.norm(forward)
  z_axis = -forward
  x_axis = np.cross(forward, up)
  x_axis /= np.linalg.norm(x_axis)
  y_axis = np.cross(z_axis, x_axis)

  cam_xpos = d.cam_xpos.numpy()
  cam_xmat = d.cam_xmat.numpy()
  cam_xpos[:, cam] = position
  cam_xmat[:, cam] = np.column_stack((x_axis, y_axis, z_axis))
  d.cam_xpos.assign(cam_xpos)
  d.cam_xmat.assign(cam_xmat)


def _save_tiled_rgb(
  packed_rows: np.ndarray,
  width: int,
  height: int,
  grid_rows: int,
  grid_cols: int,
  out_path: str,
):
  """Tile multiple RGB worlds into a single image and save it."""
  nworld = packed_rows.shape[0]
  expected = grid_rows * grid_cols
  if nworld < expected:
    raise ValueError(f"tiled rendering requires at least {expected} worlds, got {nworld}")

  tiles = []
  for wi in range(expected):
    tiles.append(_rgb_image_from_packed(packed_rows[wi], width, height))

  rows = []
  for r in range(grid_rows):
    row_tiles = tiles[r * grid_cols : (r + 1) * grid_cols]
    rows.append(np.concatenate(row_tiles, axis=1))
  full = np.concatenate(rows, axis=0)
  Image.fromarray(full).save(out_path)


def _save_tiled_depth(
  depth_rows: np.ndarray,
  width: int,
  height: int,
  scale: float,
  grid_rows: int,
  grid_cols: int,
  out_path: str,
):
  """Tile multiple depth worlds into a single image and save it."""
  nworld = depth_rows.shape[0]
  expected = grid_rows * grid_cols
  if nworld < expected:
    raise ValueError(f"tiled rendering requires at least {expected} worlds, got {nworld}")

  tiles = []
  for wi in range(expected):
    tiles.append(_depth_image_from_row(depth_rows[wi], width, height, scale))

  rows = []
  for r in range(grid_rows):
    row_tiles = tiles[r * grid_cols : (r + 1) * grid_cols]
    rows.append(np.concatenate(row_tiles, axis=1))
  full = np.concatenate(rows, axis=0)
  Image.fromarray(full).save(out_path)


def _main(argv: Sequence[str]):
  if len(argv) < 2:
    raise app.UsageError("Missing required input: mjcf path.")
  elif len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")

  mjm = _load_model(epath.Path(argv[1]))
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)

  wp.config.quiet = flags.FLAGS["verbosity"].value < 1
  wp.init()
  if _CLEAR_KERNEL_CACHE.value:
    wp.clear_kernel_cache()

  with wp.ScopedDevice(_DEVICE.value):
    m = mjw.put_model(mjm)

    if _OVERRIDE.value:
      override_model(m, _OVERRIDE.value)

    # Configure parallel worlds and per-camera resolution.
    if _TILED.value:
      # In tiled mode we always use 16 worlds and output a 4x4 grid at 512x512.
      nworld = 16
      grid_rows = 4
      grid_cols = 4
      final_width = 512
      final_height = 512
      render_width = final_width // grid_cols
      render_height = final_height // grid_rows
    else:
      nworld = int(_NWORLD.value)
      grid_rows = grid_cols = 1
      render_width = int(_WIDTH.value)
      render_height = int(_HEIGHT.value)

    d = mjw.put_data(mjm, mjd, nworld=nworld)
    splat_positions = splat_rotations = splat_scales = splat_rgba = None
    if _SPLAT.value:
      splat_positions, splat_rotations, splat_scales, splat_rgba = _load_ply_splats(_SPLAT.value)

    rc = mjw.create_render_context(
      mjm,
      nworld,
      (render_width, render_height),
      _RENDER_RGB.value,
      _RENDER_DEPTH.value,
      _RENDER_SEG.value,
      _USE_TEXTURES.value,
      _USE_SHADOWS.value,
      enabled_geom_groups=[0, 1, 2],
      render_skybox=_RENDER_SKYBOX.value,
      splat_positions=splat_positions,
      splat_rotations=splat_rotations,
      splat_scales=splat_scales,
      splat_rgba=splat_rgba,
    )

    print(f"Model: ncam={m.ncam} nlight={m.nlight} ngeom={m.ngeom}\n")

    world = int(_WORLD.value)
    cam = int(_CAM.value)
    if cam < 0 or cam >= m.ncam:
      raise ValueError(f"camera index out of range: {cam} not in [0, {m.ncam - 1}]")
    if not _TILED.value:
      if world < 0 or world >= d.nworld:
        raise ValueError(f"world index out of range: {world} not in [0, {d.nworld - 1}]")

    cam_res = rc.cam_res.numpy()
    base_width = int(cam_res[cam][0])
    base_height = int(cam_res[cam][1])

    rgb_adr = rc.rgb_adr.numpy()
    depth_adr = rc.depth_adr.numpy()

    if _ORBIT.value:
      if _ROLLOUT.value:
        raise ValueError("camera orbit and physics rollout cannot be enabled together")
      if not _RENDER_RGB.value:
        raise ValueError("camera orbit requires RGB rendering to be enabled (--rgb).")
      if _TILED.value:
        raise ValueError("camera orbit does not support tiled rendering")

      center = np.asarray(_ORBIT_CENTER.value, dtype=float)
      if center.shape != (3,):
        raise ValueError("orbit_center must contain three values")
      frame_count = int(_NSTEPS.value)
      if frame_count < 2:
        raise ValueError("camera orbit requires at least two frames")
      if _ORBIT_RADIUS.value <= 0.0:
        raise ValueError("orbit_radius must be positive")

      print(f"Rendering {frame_count}-frame camera orbit...")
      frames = []
      for frame in range(frame_count):
        angle = 2.0 * np.pi * frame / frame_count
        _set_orbit_camera(d, cam, angle, center, _ORBIT_RADIUS.value, _ORBIT_HEIGHT.value, _ORBIT_UP.value)
        mjw.render(m, d, rc)
        frame_array = _rgb_frame(rc, cam, world, base_width, base_height, grid_rows, grid_cols)
        frames.append(Image.fromarray(frame_array))

      frames[0].save(
        _ROLLOUT_OUTPUT.value,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // 30,
        loop=0,
      )
      print(f"Saved camera orbit to: {_ROLLOUT_OUTPUT.value}")
      return

    if _ROLLOUT.value:
      if not _RENDER_RGB.value:
        raise ValueError("rollout video requires RGB rendering to be enabled (--rgb).")

      # Use the physics timestep to choose how many simulation steps each
      # video frame should cover so that playback is approximately realtime.
      try:
        dt = float(m.opt.timestep.numpy()[0])
      except Exception:
        dt = 1.0 / 60.0

      target_fps = 30.0
      steps_per_frame = max(1, int(round(1.0 / (dt * target_fps))))
      frame_duration_ms = max(1, int(round(1000.0 / target_fps)))

      total_steps = int(_NSTEPS.value)
      print(f"Rendering rollout for {total_steps} steps (dt={dt:.4f}, steps_per_frame={steps_per_frame})...")
      frames = []

      step = 0
      while step < total_steps:
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)

        frame_array = _rgb_frame(rc, cam, world, base_width, base_height, grid_rows, grid_cols)

        if frame_array is not None:
          frames.append(Image.fromarray(frame_array))

        # Advance simulation by the number of steps represented by this frame.
        for _ in range(steps_per_frame):
          if step >= total_steps:
            break
          mjw.step(m, d)
          step += 1

      if not frames:
        raise RuntimeError("no RGB frames were generated during rollout")

      frames[0].save(
        _ROLLOUT_OUTPUT.value,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
      )
      print(f"Saved rollout video to: {_ROLLOUT_OUTPUT.value}")
      return

    # Single-frame rendering path.
    print("Rendering single frame...")
    mjw.render(m, d, rc)

    if _TILED.value:
      # Use all worlds and tile them into a 4x4 grid.
      if rgb_adr[cam] != -1:
        rgb_all = rc.rgb_data.numpy()
        slice_start = rgb_adr[cam]
        slice_end = slice_start + base_width * base_height
        rows = rgb_all[:, slice_start:slice_end]
        _save_tiled_rgb(rows, base_width, base_height, grid_rows, grid_cols, _OUTPUT_RGB.value)
        print(f"Saved tiled RGB to: {_OUTPUT_RGB.value}")

      if depth_adr[cam] != -1:
        depth_all = rc.depth_data.numpy()
        slice_start = depth_adr[cam]
        slice_end = slice_start + base_width * base_height
        rows = depth_all[:, slice_start:slice_end]
        _save_tiled_depth(
          rows,
          base_width,
          base_height,
          _DEPTH_SCALE.value,
          grid_rows,
          grid_cols,
          _OUTPUT_DEPTH.value,
        )
        print(f"Saved tiled depth to: {_OUTPUT_DEPTH.value}")
    else:
      # Original single-world behavior.
      if rgb_adr[cam] != -1:
        rgb = rc.rgb_data.numpy()
        row = rgb[world, rgb_adr[cam] : rgb_adr[cam] + base_width * base_height]
        _save_rgb_from_packed(row, base_width, base_height, _OUTPUT_RGB.value)
        print(f"Saved RGB to: {_OUTPUT_RGB.value}")

      if depth_adr[cam] != -1:
        depth = rc.depth_data.numpy()
        row = depth[world, depth_adr[cam] : depth_adr[cam] + base_width * base_height]
        _save_depth(row, base_width, base_height, _DEPTH_SCALE.value, _OUTPUT_DEPTH.value)
        print(f"Saved depth to: {_OUTPUT_DEPTH.value}")


def main():
  sys.argv[0] = "mujoco_warp.render"
  sys.modules["__main__"].__doc__ = __doc__
  app.run(_main)


if __name__ == "__main__":
  main()
