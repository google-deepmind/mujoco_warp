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
"""Serializable optimization recordings for offline metrics and visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

_VERSION = 5
_SUPPORTED_VERSIONS = (4, _VERSION)


@dataclass
class Iteration:
  """The optimization state before one parameter update."""

  index: int
  loss: float
  grad_norm: float
  model: SimpleNamespace
  data: SimpleNamespace
  visualization: SimpleNamespace | None = None


@dataclass(frozen=True)
class Recording:
  """Task-independent input to offline progress and visualization passes."""

  path: Path
  name: str
  iterations: tuple[Iteration, ...]
  mjm: mujoco.MjModel | None
  visual_mjm: mujoco.MjModel | None
  visual_model_fields: dict[str, np.ndarray]
  arguments: dict
  settings: dict
  scene: dict
  layout_columns: int
  layout_spacing: float | tuple[float, float]
  simulation_fps: float
  width: int
  height: int
  contact_forces: bool
  trace_body: tuple[str, ...]
  trace_width: float
  trace_alpha: float

  @property
  def package(self) -> Path:
    return self.path.parent


def _jsonable(value):
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, tuple):
    return [_jsonable(item) for item in value]
  if isinstance(value, list):
    return [_jsonable(item) for item in value]
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  return value


def _mjm_buffer(mjm: mujoco.MjModel) -> np.ndarray:
  buffer = np.empty(mujoco.mj_sizeModel(mjm), dtype=np.uint8)
  mujoco.mj_saveModel(mjm, buffer=buffer)
  return buffer


def _load_mjm(buffer: np.ndarray) -> mujoco.MjModel:
  name = "model.mjb"
  return mujoco.MjModel.from_binary_path(name, assets={name: buffer.tobytes()})


def save(
  path: str | Path,
  *,
  name: str,
  iterations: list[Iteration],
  mjm: mujoco.MjModel | None,
  visual_mjm: mujoco.MjModel | None,
  visual_model_fields: dict[str, np.ndarray],
  arguments: dict,
  settings: dict,
  scene: dict,
  layout_columns: int,
  layout_spacing: float | tuple[float, float],
  simulation_fps: float,
  width: int,
  height: int,
  contact_forces: bool,
  trace_body: tuple[str, ...],
  trace_width: float,
  trace_alpha: float,
) -> Path:
  """Writes a recording without pickle-backed arrays."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  arrays = {}
  entries = []
  for slot, iteration in enumerate(iterations):
    prefix = f"iteration_{slot:04d}"
    arrays[f"{prefix}_qpos"] = np.asarray(iteration.data.qpos)
    if iteration.data.qvel is not None:
      arrays[f"{prefix}_qvel"] = np.asarray(iteration.data.qvel)
    if iteration.data.ctrl is not None:
      arrays[f"{prefix}_ctrl"] = np.asarray(iteration.data.ctrl)
    for field, value in vars(iteration.model).items():
      arrays[f"{prefix}_model_{field}"] = np.asarray(value)
    visualization = iteration.visualization
    if visualization is not None:
      arrays[f"{prefix}_viz_qpos"] = np.asarray(visualization.qpos)
      if visualization.qvel is not None:
        arrays[f"{prefix}_viz_qvel"] = np.asarray(visualization.qvel)
      if visualization.ctrl is not None:
        arrays[f"{prefix}_viz_ctrl"] = np.asarray(visualization.ctrl)
    entries.append(
      {
        "index": iteration.index,
        "loss": iteration.loss,
        "grad_norm": iteration.grad_norm,
        "qvel": iteration.data.qvel is not None,
        "ctrl": iteration.data.ctrl is not None,
        "model_fields": sorted(vars(iteration.model)),
        "visualization": visualization is not None,
        "viz_qvel": visualization is not None and visualization.qvel is not None,
        "viz_ctrl": visualization is not None and visualization.ctrl is not None,
      }
    )

  if mjm is not None:
    arrays["mjm"] = _mjm_buffer(mjm)
  visual_mjm_shared = visual_mjm is not None and visual_mjm is mjm
  if visual_mjm is not None and not visual_mjm_shared:
    arrays["visual_mjm"] = _mjm_buffer(visual_mjm)
  for field, value in visual_model_fields.items():
    arrays[f"visual_model_field_{field}"] = np.asarray(value)

  metadata = {
    "version": _VERSION,
    "name": name,
    "iterations": entries,
    "mjm": mjm is not None,
    "visual_mjm": visual_mjm is not None,
    "visual_mjm_shared": visual_mjm_shared,
    "visual_model_fields": sorted(visual_model_fields),
    "arguments": _jsonable(arguments),
    "settings": _jsonable(settings),
    "scene": _jsonable(scene),
    "layout_columns": layout_columns,
    "layout_spacing": _jsonable(layout_spacing),
    "simulation_fps": simulation_fps,
    "width": width,
    "height": height,
    "contact_forces": contact_forces,
    "trace_body": list(trace_body),
    "trace_width": trace_width,
    "trace_alpha": trace_alpha,
  }
  arrays["metadata"] = np.asarray(json.dumps(metadata, separators=(",", ":")))

  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("wb") as output:
    np.savez_compressed(output, **arrays)
  temporary.replace(path)
  return path


def load(path: str | Path) -> Recording:
  """Loads a recording written by :func:`save`."""
  path = Path(path)
  with np.load(path, allow_pickle=False) as arrays:
    metadata = json.loads(str(arrays["metadata"]))
    version = metadata.get("version")
    if version not in _SUPPORTED_VERSIONS:
      raise ValueError(f"unsupported recording version: {metadata.get('version')}")

    mjm = _load_mjm(arrays["mjm"]) if metadata["mjm"] else None
    if metadata["visual_mjm_shared"]:
      visual_mjm = mjm
    else:
      visual_mjm = _load_mjm(arrays["visual_mjm"]) if metadata["visual_mjm"] else None
    visual_model_fields = {
      name: np.array(arrays[f"visual_model_field_{name}"], copy=True) for name in metadata["visual_model_fields"]
    }
    iterations = []
    for slot, entry in enumerate(metadata["iterations"]):
      prefix = f"iteration_{slot:04d}"
      data = SimpleNamespace(
        qpos=np.array(arrays[f"{prefix}_qpos"], copy=True),
        qvel=np.array(arrays[f"{prefix}_qvel"], copy=True) if entry["qvel"] else None,
        ctrl=np.array(arrays[f"{prefix}_ctrl"], copy=True) if entry["ctrl"] else None,
      )
      model_fields = {name: np.array(arrays[f"{prefix}_model_{name}"], copy=True) for name in entry["model_fields"]}
      visualization = None
      if entry["visualization"]:
        qvel = np.array(arrays[f"{prefix}_viz_qvel"], copy=True) if entry["viz_qvel"] else None
        ctrl = np.array(arrays[f"{prefix}_viz_ctrl"], copy=True) if entry["viz_ctrl"] else None
        visualization = SimpleNamespace(qpos=np.array(arrays[f"{prefix}_viz_qpos"], copy=True), qvel=qvel, ctrl=ctrl)
      iterations.append(
        Iteration(
          entry["index"],
          entry["loss"],
          entry["grad_norm"],
          SimpleNamespace(**model_fields),
          data,
          visualization,
        )
      )

  if version == 4:
    columns = metadata["field_columns"]
    spacing = metadata["field_spacing"]
  else:
    columns = metadata["layout_columns"]
    spacing = metadata["layout_spacing"]
  if isinstance(spacing, list):
    spacing = tuple(spacing)
  return Recording(
    path=path,
    name=metadata["name"],
    iterations=tuple(iterations),
    mjm=mjm,
    visual_mjm=visual_mjm,
    visual_model_fields=visual_model_fields,
    arguments=metadata["arguments"],
    settings=metadata["settings"],
    scene=metadata["scene"],
    layout_columns=columns,
    layout_spacing=spacing,
    simulation_fps=metadata["simulation_fps"],
    width=metadata["width"],
    height=metadata["height"],
    contact_forces=metadata["contact_forces"],
    trace_body=tuple(metadata["trace_body"]),
    trace_width=metadata["trace_width"],
    trace_alpha=metadata["trace_alpha"],
  )
