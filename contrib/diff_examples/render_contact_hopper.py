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
"""Render the audited Hopper optimization output as an H.264 MP4.

Run:
  MUJOCO_GL=egl uv run python contrib/diff_examples/render_contact_hopper.py \
    viz_out/contact_hopper_control_run.npz \
    contrib/diff_examples/media/contact_hopper_control.mp4
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1280
HEIGHT = 720
FPS = 16
BG = "#0b1020"
FG = "#f5f7ff"
CYAN = "#36d7e8"
ORANGE = "#ff9f43"
GREEN = "#4de3a5"
GRID = "#29324a"


def dashboard(data, snapshot_index, rollout_step):
  returns = data["returns"]
  grad_norms = data["gradient_norms"]
  epochs = data["snapshot_epochs"]
  epoch = int(epochs[snapshot_index])
  contacts = int(data["snapshot_contacts"][snapshot_index, min(rollout_step, 15)])
  ad = data["gradcheck_ad"]
  fd = data["gradcheck_fd"]

  fig = plt.figure(figsize=(6.4, 7.2), dpi=100, facecolor=BG)
  grid = fig.add_gridspec(3, 1, hspace=0.48, top=0.88, bottom=0.08, left=0.14, right=0.95)
  fig.text(0.5, 0.955, "DIFFERENTIABLE HOPPER", ha="center", color=FG, fontsize=18, weight="bold")
  fig.text(
    0.5,
    0.915,
    f"epoch {epoch:3d}  |  rollout {rollout_step + 1:02d}/16  |  contacts {contacts}",
    ha="center",
    color=GREEN if contacts else ORANGE,
    fontsize=11,
  )

  ax = fig.add_subplot(grid[0])
  ax.plot(returns, color=CYAN, linewidth=2.2)
  ax.scatter([epoch], [returns[epoch]], color=ORANGE, s=50, zorder=4)
  ax.set_title("discounted return", color=FG, fontsize=11, loc="left")
  ax.set_xlim(0, len(returns) - 1)
  ax.text(0.98, 0.08, f"{returns[epoch]:.3f}", color=FG, ha="right", transform=ax.transAxes, fontsize=13)

  ax2 = fig.add_subplot(grid[1])
  safe_grad = np.maximum(grad_norms, 1.0e-5)
  ax2.semilogy(safe_grad, color=GREEN, linewidth=2.0)
  ax2.scatter([epoch], [safe_grad[epoch]], color=ORANGE, s=50, zorder=4)
  ax2.set_title("policy gradient norm", color=FG, fontsize=11, loc="left")
  ax2.set_xlim(0, len(grad_norms) - 1)
  ax2.text(0.98, 0.08, f"{grad_norms[epoch]:.4f}", color=FG, ha="right", transform=ax2.transAxes, fontsize=13)

  ax3 = fig.add_subplot(grid[2])
  x = np.arange(len(ad))
  ax3.bar(x - 0.18, ad, 0.36, color=CYAN, label="reverse-mode AD")
  ax3.bar(x + 0.18, fd, 0.36, color=ORANGE, label="central FD")
  ax3.set_xticks(x, ["u₀,₀", "u₀,₁", "u₂,₀", "u₄,₁", "u₆,₂"])
  ax3.set_title("32-substep action-gradient check", color=FG, fontsize=11, loc="left")
  ax3.legend(frameon=False, labelcolor=FG, fontsize=8, ncol=2, loc="lower left")
  relative_error = float(data["gradcheck_relative_error"])
  cosine = float(data["gradcheck_cosine"])
  ax3.text(
    0.98,
    0.92,
    f"rel. error {relative_error:.2e}\ncosine {cosine:.10f}",
    color=GREEN,
    ha="right",
    va="top",
    transform=ax3.transAxes,
    fontsize=9,
  )

  for axis in (ax, ax2, ax3):
    axis.set_facecolor(BG)
    axis.tick_params(colors="#aab4d0", labelsize=8)
    axis.grid(color=GRID, alpha=0.7, linewidth=0.6)
    for spine in axis.spines.values():
      spine.set_color(GRID)

  fig.canvas.draw()
  image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  return image


def render(data_path, output_path):
  data = np.load(data_path)
  model_xml = str(data["model_xml"])
  model = mujoco.MjModel.from_xml_string(model_xml)
  model.vis.global_.offheight = HEIGHT
  model.vis.global_.offwidth = WIDTH // 2
  sim_data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH // 2)
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = (0.15, 0.0, 0.75)
  camera.distance = 2.6
  camera.azimuth = 90.0
  camera.elevation = -5.0

  output = os.path.abspath(output_path)
  os.makedirs(os.path.dirname(output), exist_ok=True)
  writer = imageio.get_writer(output, fps=FPS, codec="libx264", quality=8, macro_block_size=None)
  try:
    for snapshot_index, _epoch in enumerate(data["snapshot_epochs"]):
      for rollout_step, qpos in enumerate(data["snapshot_qpos"][snapshot_index, 1:]):
        sim_data.qpos[:] = qpos
        sim_data.qvel[:] = data["snapshot_qvel"][snapshot_index, rollout_step + 1]
        mujoco.mj_forward(model, sim_data)
        camera.lookat[0] = max(0.15, float(qpos[0]))
        renderer.update_scene(sim_data, camera=camera)
        physics = renderer.render()
        panel = dashboard(data, snapshot_index, rollout_step)
        frame = np.concatenate((physics, panel), axis=1)
        writer.append_data(frame)
  finally:
    writer.close()
    renderer.close()
  print(output)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("data")
  parser.add_argument("output")
  args = parser.parse_args()
  render(args.data, args.output)


if __name__ == "__main__":
  main()
