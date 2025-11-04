# Copyright 2025 The Newton Developers
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
  mjwarp-render benchmark/humanoid/humanoid.xml --nworld=1 --cam=0 --width=512 --height=512 -o "opt.solver=cg"
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
_FOV_DEG = flags.DEFINE_float("fov_deg", 60.0, "vertical field-of-view in degrees")
_RENDER_RGB = flags.DEFINE_bool("rgb", True, "render RGB image")
_RENDER_DEPTH = flags.DEFINE_bool("depth", True, "render depth image")
_USE_TEXTURES = flags.DEFINE_bool("textures", True, "use textures")
_USE_SHADOWS = flags.DEFINE_bool("shadows", True, "use shadows")
_DEVICE = flags.DEFINE_string("device", None, "override the default Warp device")
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "clear Warp kernel cache before rendering")
_OVERRIDE = flags.DEFINE_multi_string("override", [], "Model overrides (notation: foo.bar = baz)", short_name="o")
_OUTPUT_RGB = flags.DEFINE_string("output_rgb", "debug.png", "output path for RGB image")
_OUTPUT_DEPTH = flags.DEFINE_string("output_depth", "debug_depth.png", "output path for depth image")
_DEPTH_SCALE = flags.DEFINE_float("depth_scale", 5.0, "scale factor to map depth to 0..255 for preview")
_ROLL = flags.DEFINE_bool("roll_kinematics", False, "advance simulation before rendering")
_ROLL_STEPS = flags.DEFINE_integer("roll_steps", 50, "number of steps to advance when rolling")


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
    r = (packed & 0xFF).astype(np.uint8)
    g = ((packed >> 8) & 0xFF).astype(np.uint8)
    b = ((packed >> 16) & 0xFF).astype(np.uint8)
    img = Image.fromarray(np.dstack([r, g, b]))
    img.save(out_path)


def _save_depth(depth_row: np.ndarray, width: int, height: int, scale: float, out_path: str):
    arr = depth_row.reshape(height, width)
    arr = np.clip(arr / max(scale, 1e-6), 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    img.save(out_path)


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

        # apply CLI overrides first
        if _OVERRIDE.value:
            override_model(m, _OVERRIDE.value)

        # prepare Data from current MuJoCo state so camera/light transforms are valid
        d = mjw.put_data(
            mjm,
            mjd,
            nworld=_NWORLD.value,
        )

        rc = mjw.create_render_context(
            mjm,
            m,
            d,
            _NWORLD.value,
            _WIDTH.value,
            _HEIGHT.value,
            _USE_TEXTURES.value,
            _USE_SHADOWS.value,
            wp.radians(_FOV_DEG.value),
            _RENDER_RGB.value,
            _RENDER_DEPTH.value,
        )

        # optionally advance the simulation before rendering
        if _ROLL.value and _ROLL_STEPS.value > 0:
            print(f"Rolling kinematics for {_ROLL_STEPS.value} steps...")
            for _ in range(int(_ROLL_STEPS.value)):
                mjw.step(m, d)

        print(
            f"Model: ncam={m.ncam} nlight={m.nlight} ngeom={m.ngeom}\n"
            f"Render: {rc.width}x{rc.height} rgb={rc.render_rgb} depth={rc.render_depth}"
        )

        # render
        print("Rendering...")
        mjw.render(m, d, rc)

        world = int(_WORLD.value)
        cam = int(_CAM.value)
        if world < 0 or world >= d.nworld:
            raise ValueError(f"world index out of range: {world} not in [0, {d.nworld - 1}]")
        if cam < 0 or cam >= m.ncam:
            raise ValueError(f"camera index out of range: {cam} not in [0, {m.ncam - 1}]")

        width = rc.width
        height = rc.height
        num_pixels = width * height

        if rc.render_rgb:
            pixels = rc.pixels.numpy()
            row = pixels[world, cam]
            if row.shape[0] != num_pixels:
                raise RuntimeError("unexpected pixel buffer size")
            _save_rgb_from_packed(row, width, height, _OUTPUT_RGB.value)
            print(f"Saved RGB to: {_OUTPUT_RGB.value}")

        if rc.render_depth:
            depth = rc.depth.numpy()
            row = depth[world, cam]
            if row.shape[0] != num_pixels:
                raise RuntimeError("unexpected depth buffer size")
            _save_depth(row, width, height, _DEPTH_SCALE.value, _OUTPUT_DEPTH.value)
            print(f"Saved depth to: {_OUTPUT_DEPTH.value}")


def main():
    sys.argv[0] = "mujoco_warp.render"
    sys.modules["__main__"].__doc__ = __doc__
    app.run(_main)


if __name__ == "__main__":
    main()
