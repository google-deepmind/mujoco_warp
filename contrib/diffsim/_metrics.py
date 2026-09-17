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
"""Summarize batched optimization metrics for the paper results page."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

_ASSETS = Path(__file__).resolve().parent / "assets"


@dataclass(frozen=True)
class Metric:
  key: str
  label: str
  unit: str | None = None
  direction: str | None = None
  absolute: bool = False


def target_progress(error, scale: float, speed=None, speed_scale: float | None = None):
  """Maps target error and optional terminal speed to smooth unit progress."""
  if scale <= 0.0:
    raise ValueError("scale must be positive")
  if (speed is None) != (speed_scale is None):
    raise ValueError("speed and speed_scale must be provided together")
  normalized_error = (np.asarray(error) / scale) ** 2.0
  if speed is not None:
    if speed_scale <= 0.0:
      raise ValueError("speed_scale must be positive")
    normalized_error += (np.asarray(speed) / speed_scale) ** 2.0
  return 1.0 / (1.0 + normalized_error) ** 2.0


def normalize_progress(performance, lower_bound, upper_bound, *, worst_env=False):
  """Normalizes performance between lower and upper task bounds."""
  performance = np.asarray(performance, dtype=np.float64)
  lower_bound = np.asarray(lower_bound, dtype=np.float64)
  upper_bound = np.asarray(upper_bound, dtype=np.float64)
  if worst_env:
    lower_bound = np.min(lower_bound)
  scale = upper_bound - lower_bound
  if np.any(scale == 0.0):
    raise ValueError("progress bounds must be distinct")
  return (performance - lower_bound) / scale


def mean_confidence_interval(values: np.ndarray, z_score: float = 1.96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns the mean and normal-approximation 95% CI along the environment axis.

  Args:
    values: Array shaped ``(num_envs, ...)``. Environments are treated as
      independent samples.

  Returns:
    Mean, lower bound, and upper bound arrays with the leading environment
    dimension removed.
  """
  values = np.asarray(values, dtype=np.float64)
  if values.ndim < 1 or values.shape[0] < 1:
    raise ValueError("values must contain at least one environment")
  if not np.all(np.isfinite(values)):
    raise ValueError("values must be finite")

  mean = values.mean(axis=0)
  if values.shape[0] == 1:
    return mean, mean.copy(), mean.copy()

  standard_error = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
  margin = z_score * standard_error
  return mean, mean - margin, mean + margin


def _metric_values(history: list[dict], metric: Metric) -> np.ndarray | None:
  if not history or any(metric.key not in record for record in history):
    return None
  values = np.asarray([record[metric.key] for record in history], dtype=np.float64)
  if metric.absolute:
    values = np.abs(values)
  return values


def _summary(
  history: list[dict],
  metric: Metric,
  num_envs: int,
  bounds: tuple[float | None, float | None] | None = None,
) -> dict | None:
  values = _metric_values(history, metric)
  if values is None:
    return None

  if bounds is not None:
    lower_bound, upper_bound = bounds
    tolerance = 1e-6
    if lower_bound is not None:
      if np.any(values < lower_bound - tolerance):
        raise ValueError(f"metric {metric.key!r} must be at least {lower_bound}")
      values = np.maximum(values, lower_bound)
    if upper_bound is not None:
      if np.any(values > upper_bound + tolerance):
        raise ValueError(f"metric {metric.key!r} must be at most {upper_bound}")
      values = np.minimum(values, upper_bound)

  result = {
    "label": metric.label,
    "unit": metric.unit,
    "direction": metric.direction,
  }
  if values.ndim == 2 and values.shape[1] == num_envs:
    mean, lower, upper = mean_confidence_interval(values.T)
    if bounds is not None:
      if lower_bound is not None:
        lower = np.maximum(lower, lower_bound)
        upper = np.maximum(upper, lower_bound)
      if upper_bound is not None:
        lower = np.minimum(lower, upper_bound)
        upper = np.minimum(upper, upper_bound)
    result.update(
      mean=mean.tolist(),
      lower=lower.tolist(),
      upper=upper.tolist(),
      sample_size=num_envs,
    )
  elif values.ndim == 1:
    if not np.all(np.isfinite(values)):
      raise ValueError(f"metric {metric.key!r} must be finite")
    result.update(mean=values.tolist(), lower=None, upper=None, sample_size=None)
  else:
    raise ValueError(
      f"metric {metric.key!r} must contain one scalar or one ({num_envs},) array per iteration; got {values.shape[1:]}"
    )
  if bounds is not None:
    result["range"] = list(bounds)
  return result


def export(
  name: str,
  num_envs: int,
  history: list[dict],
  progress_metric: Mapping[str, object] | None = None,
  settings: Mapping[str, object] | None = None,
  output_dir: str | Path | None = None,
  scene: Mapping[str, int] | None = None,
) -> Path:
  """Writes one task's optimization curves to the paper asset directory."""
  if not history:
    raise ValueError("history must contain at least one optimization record")
  if num_envs < 1:
    raise ValueError("num_envs must be positive")

  progress = Metric(**progress_metric) if progress_metric is not None else None
  loss_key = "losses" if all("losses" in record for record in history) else "loss"
  payload = {
    "task": name,
    "num_envs": num_envs,
    "settings": dict(settings or {}),
    "scene": dict(scene or {}),
    "confidence_interval": "mean +/- 1.96 standard errors",
    "iterations": [int(record.get("iteration", index)) for index, record in enumerate(history)],
    "loss": _summary(history, Metric(loss_key, "Loss", direction="lower"), num_envs),
    "progress": _summary(history, progress, num_envs, bounds=(None, 1.0)) if progress is not None else None,
    "gradients": _summary(history, Metric("grad_norm", "Gradient norm"), num_envs),
  }

  directory = _ASSETS / name if output_dir is None else Path(output_dir)
  directory.mkdir(parents=True, exist_ok=True)
  path = directory / f"{name}_metrics.json"
  path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
  print(f"[{name}] metrics written to {path}")
  return path
