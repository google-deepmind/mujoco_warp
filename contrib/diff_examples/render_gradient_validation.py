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
"""Render the differentiability validation matrix from auditable JSON."""

import argparse
import json
import os

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1280
HEIGHT = 720
FPS = 16
BG = "#0b1020"
PANEL = "#121a2e"
FG = "#f5f7ff"
MUTED = "#aab4d0"
CYAN = "#36d7e8"
ORANGE = "#ff9f43"
GREEN = "#4de3a5"
RED = "#ff5f6d"
GRID = "#29324a"


def style_axis(axis):
  axis.set_facecolor(PANEL)
  axis.tick_params(colors=MUTED, labelsize=8)
  axis.grid(color=GRID, alpha=0.75, linewidth=0.6, axis="x")
  for spine in axis.spines.values():
    spine.set_color(GRID)


def make_frame(evidence, reveal_checks, reveal_mutations, phase):
  checks = evidence["gradient_checks"]
  mutations = evidence["mutation_tests"]
  fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
  grid = fig.add_gridspec(
    2, 2, height_ratios=(3.1, 1.0), hspace=0.34, wspace=0.28, left=0.07, right=0.96, top=0.86, bottom=0.08
  )
  fig.text(0.5, 0.95, "DIFFERENTIABILITY — EVIDENCE MATRIX", ha="center", color=FG, fontsize=20, weight="bold")
  fig.text(
    0.5,
    0.905,
    "central finite differences · mutation tests · multiworld replication · explicit non-claim",
    ha="center",
    color=MUTED,
    fontsize=10,
  )

  axis = fig.add_subplot(grid[0, 0])
  style_axis(axis)
  shown_checks = checks[:reveal_checks]
  labels = [entry["name"] for entry in checks]
  y = np.arange(len(labels))
  axis.set_xscale("log")
  axis.set_xlim(5.0e-7, 2.0e-2)
  axis.set_ylim(-0.7, len(labels) - 0.3)
  axis.set_yticks(y, labels)
  axis.invert_yaxis()
  axis.axvline(5.0e-3, color=ORANGE, linestyle="--", linewidth=1.4, label="5e-3 reference gate")
  for index, entry in enumerate(shown_checks):
    value = entry["relative_error"]
    axis.plot([5.0e-7, value], [index, index], color=CYAN, linewidth=3.0)
    axis.scatter([value], [index], color=GREEN, edgecolor=FG, linewidth=0.4, s=54, zorder=3)
    axis.text(value * 1.18, index, f"{value:.2e}", color=FG, va="center", fontsize=8)
  axis.set_title("AD ↔ finite difference relative error", color=FG, fontsize=12, loc="left", pad=12)
  axis.legend(frameon=False, labelcolor=MUTED, fontsize=8, loc="lower right")

  axis2 = fig.add_subplot(grid[0, 1])
  style_axis(axis2)
  labels2 = [entry["name"].replace("remove ", "− ") for entry in mutations]
  y2 = np.arange(len(labels2))
  axis2.set_xscale("log")
  axis2.set_xlim(1.0e-3, 2.0)
  axis2.set_ylim(-0.7, len(labels2) - 0.3)
  axis2.set_yticks(y2, labels2)
  axis2.invert_yaxis()
  axis2.axvline(5.0e-3, color=ORANGE, linestyle="--", linewidth=1.4)
  for index, entry in enumerate(mutations[:reveal_mutations]):
    value = entry["mutated_relative_error"]
    axis2.barh(index, value, left=1.0e-3, color=RED, height=0.45, alpha=0.88)
    axis2.text(value * 1.04, index, f"{value:.3g}", color=FG, va="center", fontsize=9)
    axis2.text(0.02, index + 0.28, entry["effect"], color=MUTED, va="center", fontsize=7)
  axis2.set_title("mutation test: error after deleting term", color=FG, fontsize=12, loc="left", pad=12)

  bottom = fig.add_subplot(grid[1, :])
  bottom.set_facecolor(PANEL)
  bottom.set_xticks([])
  bottom.set_yticks([])
  for spine in bottom.spines.values():
    spine.set_color(GRID)
  multiworld = evidence["multiworld"]
  bottom.text(0.03, 0.72, "MULTIWORLD", color=CYAN, weight="bold", fontsize=11, transform=bottom.transAxes)
  bottom.text(
    0.03,
    0.26,
    f"64 / 256 worlds   ·   max gradient deviation {multiworld['max_interworld_gradient_deviation']:.1f}   ·   sparse temporals {multiworld['sparse_temporal_memory_mib'][0]:.1f} / {multiworld['sparse_temporal_memory_mib'][1]:.1f} MiB",
    color=FG,
    fontsize=12,
    transform=bottom.transAxes,
  )
  nonclaim = evidence["known_nonclaim"]
  if phase >= 2:
    bottom.text(0.63, 0.72, "NOT SUPPORTED", color=RED, weight="bold", fontsize=11, transform=bottom.transAxes)
    bottom.text(
      0.63,
      0.22,
      f"elliptic cone: qpos {nonclaim['qpos_relative_error']:.3f}, qvel {nonclaim['qvel_relative_error']:.3f}\nneeds coupled per-contact K block",
      color=FG,
      fontsize=11,
      transform=bottom.transAxes,
    )

  fig.canvas.draw()
  frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  return frame


def render(evidence_path, output_path):
  with open(evidence_path, encoding="utf-8") as source:
    evidence = json.load(source)
  checks = evidence["gradient_checks"]
  mutations = evidence["mutation_tests"]
  timeline = []
  for count in range(1, len(checks) + 1):
    timeline.extend([(count, 0, 0)] * 5)
  timeline.extend([(len(checks), 0, 1)] * 10)
  for count in range(1, len(mutations) + 1):
    timeline.extend([(len(checks), count, 1)] * 7)
  timeline.extend([(len(checks), len(mutations), 2)] * 24)

  output = os.path.abspath(output_path)
  os.makedirs(os.path.dirname(output), exist_ok=True)
  writer = imageio.get_writer(output, fps=FPS, codec="libx264", quality=8, macro_block_size=None)
  try:
    for reveal_checks, reveal_mutations, phase in timeline:
      writer.append_data(make_frame(evidence, reveal_checks, reveal_mutations, phase))
  finally:
    writer.close()
  print(output)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("evidence")
  parser.add_argument("output")
  args = parser.parse_args()
  render(args.evidence, args.output)


if __name__ == "__main__":
  main()
