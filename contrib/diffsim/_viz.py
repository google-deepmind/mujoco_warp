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
"""Live visualization and offline USD export for differentiable simulation demos."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import _record
import _usd
import mujoco
import numpy as np
import warp as wp

_ASSETS = Path(__file__).with_name("assets")
_GHOST_RGBA = (0.95, 0.66, 0.42, 0.35)


def _add_polyline(scene, points, rgba, width=0.01):
  for start, end in zip(points[:-1], points[1:]):
    if scene.ngeom >= scene.maxgeom:
      return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
      geom,
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      np.zeros(3),
      np.zeros(3),
      np.eye(3).ravel(),
      np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end)
    scene.ngeom += 1


@dataclass(frozen=True)
class Camera:
  """Free-camera configuration."""

  lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
  distance: float = 4.0
  azimuth: float = 90.0
  elevation: float = -20.0

  def apply(self, camera: mujoco.MjvCamera):
    camera.lookat[:] = self.lookat
    camera.distance = self.distance
    camera.azimuth = self.azimuth
    camera.elevation = self.elevation


@dataclass(frozen=True)
class Layout:
  """Layout for batched USD environments."""

  columns: int = 5
  spacing: float | tuple[float, float] = 2.2


def _name_geoms(spec):
  """Gives unnamed geoms stable names before attaching model copies."""
  seen = {geom.name for geom in spec.geoms if geom.name}
  for geom in spec.geoms:
    if geom.name:
      continue
    base = (getattr(geom, "meshname", "") or "geom").replace(".", "_")
    name = base
    suffix = 1
    while name in seen:
      name = f"{base}_{suffix}"
      suffix += 1
    geom.name = name
    seen.add(name)


def _clean_robot_spec(path):
  """Removes scene furniture from a robot-only copy."""
  spec = mujoco.MjSpec.from_file(str(path))
  for geom in list(spec.geoms):
    if geom.type == mujoco.mjtGeom.mjGEOM_PLANE or any(token in (geom.name or "").lower() for token in ("floor", "ground")):
      spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)
  _name_geoms(spec)
  return spec


def ghost_model(scene_path, model_path, rgba=_GHOST_RGBA, *, compile=True):
  """Builds named solid and translucent robot copies for trajectory comparison."""
  scene = mujoco.MjSpec.from_file(str(scene_path))
  _name_geoms(scene)
  for geom in scene.geoms:
    geom.name = f"sim_{geom.name}"

  ghost = _clean_robot_spec(model_path)
  for geom in ghost.geoms:
    geom.rgba = rgba
    geom.contype = 0
    geom.conaffinity = 0
  frame = scene.worldbody.add_frame()
  scene.attach(ghost, prefix="ghost_", frame=frame)
  return scene.compile() if compile else scene


def ghost_trajectory(trajectory, reference):
  """Appends a shared or batched reference trajectory to simulation qpos."""
  reference = np.broadcast_to(reference, trajectory.shape)
  return np.concatenate((trajectory, reference), axis=-1)


def export(path: str | Path, device=None):
  """Exports a saved optimization recording to USD."""
  recording = _record.load(path)
  rollouts = [
    (iteration.visualization.qpos, iteration.visualization.qvel, iteration.visualization.ctrl, None)
    for iteration in recording.iterations
    if iteration.visualization is not None
  ]
  if recording.visual_mjm is None or not rollouts:
    raise ValueError(f"recording does not contain USD trajectories: {path}")

  source = SimpleNamespace(
    name=recording.name,
    mjm=recording.visual_mjm,
    model_fields=recording.visual_model_fields,
    layout=Layout(recording.layout_columns, recording.layout_spacing),
    args=SimpleNamespace(output=str(recording.package), width=recording.width, height=recording.height),
    rollouts=rollouts,
    simulation_fps=recording.simulation_fps,
    contact_forces=recording.contact_forces,
    trace_body=recording.trace_body,
    trace_width=recording.trace_width,
    trace_alpha=recording.trace_alpha,
  )
  with wp.ScopedDevice(device):
    _usd.export(source)


class Visualizer:
  """Displays trajectories live or records them for offline processing."""

  def __init__(
    self,
    mjm: mujoco.MjModel,
    camera: Camera,
    layout: Layout,
    args,
    name: str,
    *,
    final_iteration: int | None = None,
    trace_width: float = 0.01,
    trace_alpha: float = 0.5,
    arguments: dict | None = None,
    settings: dict | None = None,
    scene: dict | None = None,
    model_fields: dict[str, np.ndarray] | None = None,
    metric_mjm: mujoco.MjModel | None = None,
  ):
    if not 0 <= args.viz_env < args.num_envs:
      raise ValueError(f"viz_env {args.viz_env} is outside [0, {args.num_envs})")
    if trace_width <= 0.0:
      raise ValueError("trace_width must be positive")
    if not 0.0 <= trace_alpha <= 1.0:
      raise ValueError("trace_alpha must be in [0, 1]")

    self.mjm = mjm
    self.metric_mjm = mjm if metric_mjm is None else metric_mjm
    self.model_fields = {name: np.asarray(value) for name, value in (model_fields or {}).items()}
    for field_name, values in self.model_fields.items():
      host_value = getattr(mjm, field_name, None)
      expected = (args.num_envs, *host_value.shape) if host_value is not None else None
      if expected is None or values.shape != expected:
        raise ValueError(f"model field {field_name!r} must have shape {expected}, got {values.shape}")
      if args.viz in ("mjwarp", "mujoco"):
        host_value[:] = values[args.viz_env]
    self.mjd = mujoco.MjData(mjm) if args.viz in ("mjwarp", "mujoco") else None
    self.args = args
    self.name = name
    self.final_iteration = final_iteration
    self.trace_width = trace_width
    self.trace_alpha = trace_alpha
    self.layout = layout
    self.rollouts = []
    self.traces = []
    self.iterations = []
    self._record_enabled = args.viz == "usd" or getattr(args, "metrics", False)
    self.arguments = dict(arguments or {})
    self.settings = dict(settings or {})
    self.scene = dict(scene or {})
    self.trace_body = ()
    self.simulation_fps = None
    self.contact_forces = getattr(args, "contact_forces", False)
    self.scene_option = mujoco.MjvOption()
    self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.contact_forces
    if args.viz_every < 1:
      raise ValueError("viz_every must be positive")
    if args.viz_stride < 1:
      raise ValueError("viz_stride must be positive")
    self.viz_every = 1 if args.viz == "mjwarp" else args.viz_every
    self.camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(self.mjm, self.camera)
    camera.apply(self.camera)

    self.viewer = None
    if args.viz == "mjwarp":
      from mujoco import viewer as mujoco_viewer

      self.viewer = mujoco_viewer.launch_passive(self.mjm, self.mjd, show_left_ui=False, show_right_ui=False)
      camera.apply(self.viewer.cam)
      self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.contact_forces

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close(hold=exc_type is None)

  def should_update(self, iteration: int):
    """Returns whether this optimization iteration will be visualized."""
    return self.args.viz != "none" and (iteration % self.viz_every == 0 or iteration == self.final_iteration)

  def record_iteration(
    self,
    iteration: int,
    loss: float,
    grad_norm: float,
    model: SimpleNamespace,
    data: SimpleNamespace,
    visualization: SimpleNamespace | None = None,
    trace_body=None,
  ):
    """Displays or records one optimizer iteration."""
    qpos = np.asarray(data.qpos)
    if qpos.shape[0] != self.args.num_envs:
      raise ValueError(f"expected {self.args.num_envs} trajectories, got {qpos.shape[0]}")
    if qpos.shape[1] < 2:
      raise ValueError("recorded trajectories must contain at least two states")

    if self._record_enabled and self.iterations and iteration <= self.iterations[-1].index:
      raise ValueError("recorded iteration indices must be strictly increasing")

    rendered = None
    if visualization is not None:
      rendered = self._prepare_visualization(visualization)
      names = () if trace_body is None else ((trace_body,) if isinstance(trace_body, str) else tuple(trace_body))
      if self.trace_body and names != self.trace_body:
        raise ValueError("all recorded iterations must use the same trace bodies")
      self.trace_body = names

      if self.args.viz != "usd":
        frames = rendered.qpos[self.args.viz_env]
        velocities = None if rendered.qvel is None else rendered.qvel[self.args.viz_env]
        controls = None if rendered.ctrl is None else rendered.ctrl[self.args.viz_env]
        trace = self._trace(frames, names)
        if self.args.viz == "mjwarp":
          self._play(frames, velocities, controls, trace)
          if trace:
            self.traces.append(trace)
        else:
          self.rollouts.append((np.array(frames, copy=True), velocities, controls, trace))

    if self._record_enabled:
      self.iterations.append(
        _record.Iteration(
          index=int(iteration),
          loss=float(loss),
          grad_norm=float(grad_norm),
          model=SimpleNamespace(**{name: np.array(value, copy=True) for name, value in vars(model).items()}),
          data=SimpleNamespace(
            qpos=np.array(qpos, copy=True),
            qvel=None if data.qvel is None else np.array(data.qvel, copy=True),
            ctrl=None if data.ctrl is None else np.array(data.ctrl, copy=True),
          ),
          visualization=rendered if self.args.viz == "usd" else None,
        )
      )

  def update(self, iteration: int, qpos: np.ndarray, trace_body=None, *, qvel=None, ctrl=None):
    """Records one visualization-only trajectory for direct callers."""
    if not self.should_update(iteration):
      return
    data = SimpleNamespace(qpos=np.asarray(qpos), qvel=qvel, ctrl=ctrl)
    self.record_iteration(iteration, np.nan, np.nan, SimpleNamespace(), data, data, trace_body)

  def _prepare_visualization(self, trajectory):
    qpos = np.asarray(trajectory.qpos)
    qvel = None if trajectory.qvel is None else np.asarray(trajectory.qvel)
    ctrl = None if trajectory.ctrl is None else np.asarray(trajectory.ctrl)
    if qpos.shape[0] != self.args.num_envs:
      raise ValueError(f"expected {self.args.num_envs} visualization trajectories, got {qpos.shape[0]}")
    if qpos.shape[2] != self.mjm.nq:
      raise ValueError(f"visualization qpos must have width {self.mjm.nq}")

    indices = np.arange(0, qpos.shape[1], self.args.viz_stride)
    if indices[-1] != qpos.shape[1] - 1:
      indices = np.append(indices, qpos.shape[1] - 1)
    frames = qpos[:, indices]

    def sample(value, width, name):
      if value is None:
        return None
      prefix = (self.args.num_envs, qpos.shape[1])
      if value.shape[:2] != prefix or value.ndim != 3 or value.shape[2] > width:
        raise ValueError(f"{name} must have shape ({prefix[0]}, {prefix[1]}, N) with N <= {width}")
      sampled = value[:, indices]
      if sampled.shape[2] == width:
        return sampled
      padded = np.zeros((*sampled.shape[:2], width), dtype=sampled.dtype)
      padded[..., : sampled.shape[2]] = sampled
      return padded

    velocities = sample(qvel, self.mjm.nv, "qvel")
    controls = sample(ctrl, self.mjm.nu, "ctrl")
    if self.contact_forces and (velocities is None or controls is None):
      raise ValueError("contact-force visualization requires qvel and ctrl trajectories")
    duration = (qpos.shape[1] - 1) * self.mjm.opt.timestep
    simulation_fps = (frames.shape[1] - 1) / duration
    if self.simulation_fps is None:
      self.simulation_fps = simulation_fps
    elif not np.isclose(self.simulation_fps, simulation_fps):
      raise ValueError("visualized rollouts must have the same duration")
    return SimpleNamespace(qpos=np.array(frames, copy=True), qvel=velocities, ctrl=controls)

  def close(self, hold=True):
    if self.args.viz == "mjwarp":
      if hold and self.viewer is not None:
        while self.viewer.is_running():
          self.viewer.sync()
          time.sleep(1.0 / self.args.fps)
      if self.viewer is not None:
        self.viewer.close()
    elif self.args.viz == "mujoco":
      self._render()
    if self.iterations:
      path = self._save_recording()
      if self.args.viz == "usd":
        export(path)

  def _save_recording(self):
    if self.args.viz == "usd":
      package = Path(self.args.output) if self.args.output else _usd._ASSETS / self.name
    else:
      package = _ASSETS / self.name
    path = package / f"{self.name}_recording.npz"
    visual_mjm = self.mjm if self.args.viz == "usd" else None
    visual_model_fields = self.model_fields if visual_mjm is not None else {}
    mjm = self.metric_mjm if getattr(self.args, "metrics", False) else None
    simulation_fps = 0.0 if self.simulation_fps is None else self.simulation_fps
    _record.save(
      path,
      name=self.name,
      iterations=self.iterations,
      mjm=mjm,
      visual_mjm=visual_mjm,
      visual_model_fields=visual_model_fields,
      arguments=self.arguments,
      settings=self.settings,
      scene=self.scene,
      layout_columns=self.layout.columns,
      layout_spacing=self.layout.spacing,
      simulation_fps=simulation_fps,
      width=self.args.width,
      height=self.args.height,
      contact_forces=self.contact_forces,
      trace_body=self.trace_body,
      trace_width=self.trace_width,
      trace_alpha=self.trace_alpha,
    )
    print(f"[viz] recording written to {path}")
    return path

  def _set_state(self, qpos, qvel=None, ctrl=None):
    self.mjd.qpos[:] = qpos
    self.mjd.qvel[:] = 0.0 if qvel is None else qvel
    self.mjd.ctrl[:] = 0.0 if ctrl is None else ctrl
    self.mjd.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(self.mjm, self.mjd)

  def _trace(self, frames, body_names):
    if not body_names:
      return None
    body_ids = [mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
    if unknown := [name for name, body_id in zip(body_names, body_ids) if body_id < 0]:
      raise ValueError(f"unknown trace body: {unknown[0]}")
    traces = [[] for _ in body_ids]
    for qpos in frames:
      self._set_state(qpos)
      for points, body_id in zip(traces, body_ids):
        points.append(self.mjd.xpos[body_id].copy())
    return tuple(np.asarray(points) for points in traces)

  def _draw_traces(self, scene, trace, index, previous):
    for traces in previous:
      for points in traces:
        _add_polyline(scene, points, (0.2, 0.55, 0.9, 0.4 * self.trace_alpha), width=0.5 * self.trace_width)
    if trace:
      for points in trace:
        _add_polyline(
          scene,
          points[: index + 1],
          (0.95, 0.55, 0.1, self.trace_alpha),
          width=self.trace_width,
        )

  def _play(self, frames, velocities, controls, trace):
    if self.viewer is None:
      return
    period = 1.0 / self.args.fps
    for index, qpos in enumerate(frames):
      if not self.viewer.is_running():
        break
      start = time.time()
      qvel = None if velocities is None else velocities[index]
      ctrl = None if controls is None else controls[index]
      self._set_state(qpos, qvel, ctrl)
      self.viewer.user_scn.ngeom = 0
      self._draw_traces(self.viewer.user_scn, trace, index, self.traces)
      self.viewer.sync()
      time.sleep(max(0.0, period - (time.time() - start)))

  def _render(self):
    if not self.rollouts:
      return
    try:
      import imageio.v2 as imageio
    except ImportError as err:
      raise RuntimeError("MuJoCo rendering requires imageio and imageio-ffmpeg") from err

    output = Path(self.args.output) if self.args.output else _ASSETS / f"{self.name}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    previous = []
    with imageio.get_writer(output, fps=self.args.fps) as writer:
      with mujoco.Renderer(self.mjm, height=self.args.height, width=self.args.width, max_geom=10000) as renderer:
        for frames, velocities, controls, trace in self.rollouts:
          for index, qpos in enumerate(frames):
            qvel = None if velocities is None else velocities[index]
            ctrl = None if controls is None else controls[index]
            self._set_state(qpos, qvel, ctrl)
            renderer.update_scene(self.mjd, camera=self.camera, scene_option=self.scene_option)
            self._draw_traces(renderer.scene, trace, index, previous)
            writer.append_data(renderer.render())
            frame_count += 1
          if trace:
            previous.append(trace)
    print(f"[viz] wrote {frame_count} frames to {output}")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("recording", type=Path)
  parser.add_argument("--device", help="Warp device, for example cuda:0 or cpu")
  args = parser.parse_args(argv)
  export(args.recording, args.device)


if __name__ == "__main__":
  main()
