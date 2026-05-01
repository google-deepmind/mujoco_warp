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

"""Open the backface-cull test scenes in MuJoCo's interactive viewer.

These are the same XMLs (under ``mujoco_warp/test_data/backface_cull/``) used
by the ``test_backface_cull_*`` tests in ``mujoco_warp/_src/render_test.py``.
The render camera (``name="cam"``) sits at the origin, fully enclosed by a
primitive, with a marker box at +Y. To match what the renderer tests see,
press ``[`` / ``]`` in the viewer to cycle to the ``cam`` camera.

Usage:

  uv run python contrib/view_backface_cull.py
  uv run python contrib/view_backface_cull.py --shape=sphere
  uv run python contrib/view_backface_cull.py --shape=all

The viewer blocks until you close the window; with ``--shape=all`` (default)
the scenes open one after another.
"""

from typing import Sequence

import mujoco
import mujoco.viewer
from absl import app
from absl import flags
from etils import epath

_SHAPES = ("sphere", "ellipsoid", "capsule", "cylinder", "box")

_SHAPE = flags.DEFINE_enum(
  "shape",
  "all",
  ("all", *_SHAPES),
  "Which backface-cull scene to open. 'all' opens each in sequence.",
)


def _xml_path(shape: str) -> str:
  base = epath.resource_path("mujoco_warp") / "test_data" / "backface_cull"
  return (base / f"{shape}.xml").as_posix()


def _launch(shape: str) -> None:
  path = _xml_path(shape)
  print(f"[view_backface_cull] opening {shape}: {path}")
  print("  - Press '[' / ']' to cycle to the 'cam' camera (the one tested).")
  print("  - Close the window to continue.")
  mujoco.viewer.launch_from_path(path)


def _main(argv: Sequence[str]) -> None:
  del argv
  shapes = _SHAPES if _SHAPE.value == "all" else (_SHAPE.value,)
  for shape in shapes:
    _launch(shape)


if __name__ == "__main__":
  app.run(_main)
