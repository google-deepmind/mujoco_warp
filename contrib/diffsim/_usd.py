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
"""PointInstancer USD export for rigid differentiable-simulation scenes.

Rigid geoms and sites are supported. Tendons, flexes, and other deforming
render objects deliberately remain on the standard exporter.
"""

import os
from pathlib import Path

import mujoco
import numpy as np
import scipy.spatial.transform as st
import warp as wp

import mujoco_warp as mjw

_ASSETS = Path(__file__).with_name("assets")
_ROLLOUT_LENGTHS_ATTRIBUTE = "mjwarp:rolloutLengths"


def export(visualizer):
  """Exports recorded environments with batched PointInstancer transforms."""
  if not visualizer.rollouts:
    return
  try:
    from pxr import Usd
    from pxr import UsdGeom
  except ImportError as err:
    raise RuntimeError("USD visualization requires usd-core") from err

  name = visualizer.name
  package = Path(visualizer.args.output) if visualizer.args.output else _ASSETS / name
  package.mkdir(parents=True, exist_ok=True)
  rollout_lengths = [frames.shape[1] for frames, _, _, _ in visualizer.rollouts]
  frame_count = sum(rollout_lengths)
  world_count = visualizer.rollouts[0][0].shape[0]
  if any(frames.shape[0] != world_count for frames, _, _, _ in visualizer.rollouts):
    raise ValueError("all USD rollouts must contain the same number of environments")
  offsets = _layout_offsets(visualizer.layout, world_count)
  template = _template_exporter(visualizer, package)
  objects = _template_objects(template)

  path = package / f"{name}.usdc"
  temporary = package / f".{name}.tmp.usdc"
  if temporary.exists():
    temporary.unlink()
  stage = Usd.Stage.CreateNew(str(temporary))
  try:
    _set_stage_metadata(stage, frame_count, visualizer.simulation_fps)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    _set_rollout_metadata(world, rollout_lengths)
    # Blender stops PointInstancer discovery at defined, typeless ancestors.
    UsdGeom.Scope.Define(stage, "/World/Instances")
    prototype_paths = _copy_prototypes(template, stage, objects)
    _copy_scene_extras(template.stage, stage, [obj["name"] for obj in objects])
    records = _instance_records(objects, prototype_paths, world_count)
    groups = _create_instancers(stage, records)
    _author_animation(visualizer, stage, groups, records, offsets, frame_count)
    stage.GetRootLayer().Save()
    stage = None
    temporary.replace(path)
    _remove_legacy_layers(package, name)
  except BaseException:
    stage = None
    if temporary.exists():
      temporary.unlink()
    raise

  print(f"[viz] wrote {frame_count} frames for {world_count} environments using {len(prototype_paths)} prototypes to {path}")


def _set_stage_metadata(stage, frame_count, frame_rate):
  from pxr import UsdGeom

  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  stage.SetFramesPerSecond(frame_rate)
  stage.SetTimeCodesPerSecond(frame_rate)
  stage.SetStartTimeCode(0)
  stage.SetEndTimeCode(frame_count - 1)


def _set_rollout_metadata(world, rollout_lengths):
  """Records flattened rollout boundaries for presentation renderers."""
  from pxr import Sdf
  from pxr import Vt

  attribute = world.GetPrim().CreateAttribute(_ROLLOUT_LENGTHS_ATTRIBUTE, Sdf.ValueTypeNames.IntArray, custom=True)
  attribute.Set(Vt.IntArray(rollout_lengths))


def _layout_offsets(layout, environments):
  """Returns centered layout offsets for each simulation environment."""
  columns = min(layout.columns, environments)
  if environments < 1 or columns < 1:
    raise ValueError("USD environment count and layout columns must be positive")
  spacing = (layout.spacing, layout.spacing) if isinstance(layout.spacing, (int, float)) else layout.spacing
  rows = (environments + columns - 1) // columns
  offsets = np.zeros((environments, 3))
  for index in range(environments):
    row, column = divmod(index, columns)
    offsets[index, 0] = (column - 0.5 * (columns - 1)) * spacing[0]
    offsets[index, 1] = (row - 0.5 * (rows - 1)) * spacing[1]
  return offsets


def _template_exporter(visualizer, package):
  """Creates static geometry, materials, and textures exactly once."""
  from mujoco.usd.exporter import USDExporter

  mjm = visualizer.mjm
  mjm.vis.global_.offheight = max(mjm.vis.global_.offheight, visualizer.args.height)
  mjm.vis.global_.offwidth = max(mjm.vis.global_.offwidth, visualizer.args.width)
  frames, velocities, controls, _ = visualizer.rollouts[0]
  mjd = mujoco.MjData(mjm)
  mjd.qpos[:] = frames[0, 0]
  mjd.qvel[:] = 0.0 if velocities is None else velocities[0, 0]
  mjd.ctrl[:] = 0.0 if controls is None else controls[0, 0]
  mujoco.mj_forward(mjm, mjd)

  exporter = USDExporter(
    model=mjm,
    height=visualizer.args.height,
    width=visualizer.args.width,
    output_directory="scene",
    output_directory_root=str(package),
    verbose=False,
  )
  exporter.update_scene(mjd)
  _fix_box_cubemaps(exporter)
  _fix_shader_input_types(exporter.stage)
  _clean_for_blender(exporter.stage)
  _rebase_assets(exporter.stage, package / "scene" / "frames", package)
  return exporter


def _template_objects(template):
  """Returns the rigid render objects represented by a static exporter."""
  objects = []
  supported = (mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE)
  for name, usd_object in template.geom_refs.items():
    geom = usd_object.geom
    if geom.objtype not in supported:
      raise NotImplementedError(
        "the PointInstancer exporter supports rigid geoms and sites, but not "
        f"{mujoco.mjtObj(geom.objtype).name} render objects ({name})"
      )
    objects.append(
      {
        "name": usd_object.usd_xform.GetPrim().GetName(),
        "object_type": int(geom.objtype),
        "object_id": int(geom.objid),
        "alpha": float(geom.rgba[3]),
      }
    )

  keys = [(obj["object_type"], obj["object_id"]) for obj in objects]
  if len(set(keys)) != len(keys):
    raise ValueError("PointInstancer rigid object ids must be unique")
  return objects


def _remove_legacy_layers(package, name):
  """Removes obsolete MuJoCo exporter layers superseded by the root stage."""
  frames_directories = [package / "scene" / "frames"]
  frames_directories.extend((package / "scene" / "variants").glob("variant_*/frames"))
  for frames in frames_directories:
    (frames / f"{name}.usdc").unlink(missing_ok=True)
    for prototype in frames.glob(f"{name}_proto*.usdc"):
      prototype.unlink()


def _copy_prototypes(template, destination, objects):
  """Copies static mesh/material specs and removes per-frame root transforms."""
  from pxr import Sdf
  from pxr import UsdGeom

  source = template.stage
  source_world = source.GetPrimAtPath("/World")
  source_material_path = Sdf.Path("/World/_materials")
  materials = source.GetPrimAtPath(source_material_path)
  if materials:
    UsdGeom.Scope.Define(destination, "/World/_materials")
    Sdf.CopySpec(
      source.GetRootLayer(),
      source_material_path,
      destination.GetRootLayer(),
      source_material_path,
    )

  prototype_root = Sdf.Path("/World/Instances/Prototypes")
  prototype_scope = UsdGeom.Scope.Define(destination, prototype_root)
  prototype_scope.MakeInvisible()
  paths = []
  for obj in objects:
    name = obj["name"]
    source_path = source_world.GetPath().AppendChild(name)
    destination_path = prototype_root.AppendChild(name)
    Sdf.CopySpec(source.GetRootLayer(), source_path, destination.GetRootLayer(), destination_path)
    destination.GetRootLayer().GetPrimAtPath(destination_path).specifier = Sdf.SpecifierDef
    prototype = destination.GetPrimAtPath(destination_path)
    xform = UsdGeom.Xformable(prototype)
    xform.ClearXformOpOrder()
    prototype.RemoveProperty("xformOp:transform")
    prototype.RemoveProperty("xformOp:scale")
    prototype.RemoveProperty("visibility")
    paths.append(destination_path)
  return paths


def _copy_scene_extras(source, destination, geom_names):
  """Copies non-geometry scene prims once, excluding the exporter's harsh default light."""
  from pxr import Sdf

  excluded = {"_materials", *geom_names}
  extras = [
    prim
    for prim in source.GetPrimAtPath("/World").GetChildren()
    if prim.GetName() not in excluded and not prim.GetName().startswith("Light_Xform_")
  ]
  for prim in extras:
    path = destination.GetPrimAtPath("/World").GetPath().AppendChild(prim.GetName())
    Sdf.CopySpec(source.GetRootLayer(), prim.GetPath(), destination.GetRootLayer(), path)


def _instance_records(objects, prototype_paths, world_count):
  """Flattens each environment's render objects into stable instance records."""
  environments = []
  object_types = []
  object_ids = []
  paths = []
  alphas = []
  for environment in range(world_count):
    for obj, path in zip(objects, prototype_paths):
      environments.append(environment)
      object_types.append(obj["object_type"])
      object_ids.append(obj["object_id"])
      paths.append(path)
      alphas.append(obj["alpha"])
  return {
    "environments": np.asarray(environments, dtype=np.int32),
    "object_types": np.asarray(object_types, dtype=np.int32),
    "object_ids": np.asarray(object_ids, dtype=np.int32),
    "prototype_paths": paths,
    "alphas": np.asarray(alphas, dtype=np.float32),
  }


def _create_instancers(stage, records):
  """Creates separate opaque and transparent PointInstancer batches."""
  from pxr import Sdf
  from pxr import UsdGeom
  from pxr import Vt

  groups = []
  transparent = records["alphas"] < 1.0 - 1.0e-6
  for name, selection in (
    ("opaque", np.flatnonzero(~transparent)),
    ("transparent", np.flatnonzero(transparent)),
  ):
    if not len(selection):
      continue
    instancer = UsdGeom.PointInstancer.Define(stage, f"/World/Instances/{name}")
    targets = []
    target_indices = {}
    proto_indices = []
    for index in selection:
      path = records["prototype_paths"][index]
      if path not in target_indices:
        target_indices[path] = len(targets)
        targets.append(path)
      proto_indices.append(target_indices[path])
    instancer.GetPrototypesRel().SetTargets(targets)

    environment_indices = records["environments"][selection]
    object_types = records["object_types"][selection]
    object_ids = records["object_ids"][selection]
    source_geom_ids = np.where(object_types == int(mujoco.mjtObj.mjOBJ_GEOM), object_ids, -1).astype(np.int32)
    ids = selection.astype(np.int64)
    instancer.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(proto_indices, dtype=np.int32)))
    instancer.CreateIdsAttr(Vt.Int64Array.FromNumpy(ids))

    primvars = UsdGeom.PrimvarsAPI(instancer)
    environment_primvar = primvars.CreatePrimvar("environmentIndex", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex)
    environment_primvar.Set(Vt.IntArray.FromNumpy(environment_indices))
    geom_primvar = primvars.CreatePrimvar("sourceGeomId", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex)
    geom_primvar.Set(Vt.IntArray.FromNumpy(source_geom_ids))
    type_primvar = primvars.CreatePrimvar("sourceObjectType", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex)
    type_primvar.Set(Vt.IntArray.FromNumpy(object_types))
    object_primvar = primvars.CreatePrimvar("sourceObjectId", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex)
    object_primvar.Set(Vt.IntArray.FromNumpy(object_ids))
    groups.append(
      {
        "selection": selection,
        "positions": instancer.CreatePositionsAttr(),
        "orientations": instancer.CreateOrientationsAttr(),
      }
    )
  return groups


def _animation_states(visualizer):
  """Concatenates recorded rollouts into time-major state arrays."""
  qpos = np.concatenate([rollout[0] for rollout in visualizer.rollouts], axis=1)

  def optional(index, width):
    values = []
    for rollout in visualizer.rollouts:
      value = rollout[index]
      if value is None:
        value = np.zeros((*rollout[0].shape[:2], width), dtype=np.float32)
      values.append(value)
    return np.concatenate(values, axis=1)

  qvel = optional(1, visualizer.mjm.nv)
  ctrl = optional(2, visualizer.mjm.nu)
  return tuple(np.ascontiguousarray(value.transpose(1, 0, 2), dtype=np.float32) for value in (qpos, qvel, ctrl))


def _forward_animation(visualizer):
  """Runs one batched MJWarp forward pass per output frame."""
  qpos, qvel, ctrl = _animation_states(visualizer)
  frame_count = qpos.shape[0]
  world_count = qpos.shape[1]
  mjm = visualizer.mjm
  model = mjw.put_model(mjm, batch_sizes={name: world_count for name in visualizer.model_fields})
  for name, value in visualizer.model_fields.items():
    getattr(model, name).assign(value)
  data = mjw.make_data(mjm, nworld=world_count)
  frame_qpos = wp.array(qpos, dtype=float)
  frame_qvel = wp.array(qvel, dtype=float)
  frame_ctrl = wp.array(ctrl, dtype=float)
  geom_xpos = wp.empty((frame_count, world_count, mjm.ngeom), dtype=wp.vec3)
  geom_xmat = wp.empty((frame_count, world_count, mjm.ngeom), dtype=wp.mat33)
  site_xpos = wp.empty((frame_count, world_count, mjm.nsite), dtype=wp.vec3)
  site_xmat = wp.empty((frame_count, world_count, mjm.nsite), dtype=wp.mat33)
  body_xpos = wp.empty((frame_count, world_count, mjm.nbody), dtype=wp.vec3) if visualizer.trace_body else None

  contacts = None
  if visualizer.contact_forces:
    capacity = data.contact.pos.shape[0]
    contact_ids = wp.array(np.arange(capacity, dtype=np.int32), dtype=int)
    force = wp.empty(capacity, dtype=wp.spatial_vector)
    contacts = {
      "nacon": wp.empty((frame_count, 1), dtype=int),
      "worldid": wp.empty((frame_count, capacity), dtype=int),
      "position": wp.empty((frame_count, capacity), dtype=wp.vec3),
      "force": wp.empty((frame_count, capacity), dtype=wp.spatial_vector),
    }

  for frame in range(frame_count):
    wp.copy(data.qpos, frame_qpos[frame])
    wp.copy(data.qvel, frame_qvel[frame])
    if data.ctrl.size:
      wp.copy(data.ctrl, frame_ctrl[frame])
    if data.act.size:
      data.act.zero_()
    data.qacc_warmstart.zero_()
    mjw.forward(model, data)
    wp.copy(geom_xpos[frame], data.geom_xpos)
    wp.copy(geom_xmat[frame], data.geom_xmat)
    if mjm.nsite:
      wp.copy(site_xpos[frame], data.site_xpos)
      wp.copy(site_xmat[frame], data.site_xmat)
    if body_xpos is not None:
      wp.copy(body_xpos[frame], data.xpos)
    if contacts is not None:
      mjw.contact_force(model, data, contact_ids, True, force)
      wp.copy(contacts["nacon"][frame], data.nacon)
      wp.copy(contacts["worldid"][frame], data.contact.worldid)
      wp.copy(contacts["position"][frame], data.contact.pos)
      wp.copy(contacts["force"][frame], force)

  result = {
    "geom_xpos": geom_xpos.numpy(),
    "geom_xmat": geom_xmat.numpy(),
    "site_xpos": site_xpos.numpy(),
    "site_xmat": site_xmat.numpy(),
    "body_xpos": None if body_xpos is None else body_xpos.numpy(),
    "contacts": None,
  }
  if contacts is not None:
    result["contacts"] = {name: value.numpy() for name, value in contacts.items()}
  return result


def _trace_rollouts(visualizer, result):
  """Extracts requested body traces from the MJWarp position results."""
  names = tuple(visualizer.trace_body)
  mjm = visualizer.mjm
  body_ids = [mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
  if unknown := [name for name, body_id in zip(names, body_ids) if body_id < 0]:
    raise ValueError(f"unknown trace body: {unknown[0]}")

  rollouts = []
  start = 0
  for frames, velocities, controls, _ in visualizer.rollouts:
    stop = start + frames.shape[1]
    environment_traces = [
      tuple(result["body_xpos"][start:stop, environment, body_id] for body_id in body_ids)
      for environment in range(frames.shape[0])
    ]
    rollouts.append((frames, velocities, controls, environment_traces))
    start = stop
  return rollouts


def _contact_rollouts(visualizer, result):
  """Converts MJWarp contact forces to MuJoCo-style display arrows."""
  if not visualizer.contact_forces:
    return None

  lengths = [frames.shape[1] for frames, _, _, _ in visualizer.rollouts]
  world_count = visualizer.rollouts[0][0].shape[0]
  force_rollouts = [[[] for _ in lengths] for _ in range(world_count)]
  contacts = result["contacts"]
  mjm = visualizer.mjm
  scale = float(mjm.vis.map.force)
  radius = float(mjm.stat.meansize * mjm.vis.scale.forcewidth)
  color = np.array(mjm.vis.rgba.contactforce, copy=True)
  start = 0
  for rollout, length in enumerate(lengths):
    for frame in range(start, start + length):
      count = int(contacts["nacon"][frame, 0])
      for environment in range(world_count):
        selection = np.flatnonzero(contacts["worldid"][frame, :count] == environment)
        arrows = []
        for contact in selection:
          point = contacts["position"][frame, contact].copy()
          endpoint = point + scale * contacts["force"][frame, contact, :3]
          arrows.append((point, endpoint, radius, color))
        force_rollouts[environment][rollout].append(arrows)
    start += length
  return force_rollouts


def _author_animation(visualizer, stage, groups, records, offsets, frame_count):
  """Computes poses with MJWarp, then authors two array attributes per group."""
  from pxr import Vt

  world_count = visualizer.rollouts[0][0].shape[0]
  positions = np.empty((len(records["object_ids"]), 3), dtype=np.float32)
  matrices = np.empty((len(records["object_ids"]), 3, 3), dtype=np.float32)
  previous_orientations = None
  layouts = []
  for environment in range(world_count):
    environment_records = np.flatnonzero(records["environments"] == environment)
    types = records["object_types"][environment_records]
    layouts.append(
      {
        "geom": environment_records[types == int(mujoco.mjtObj.mjOBJ_GEOM)],
        "site": environment_records[types == int(mujoco.mjtObj.mjOBJ_SITE)],
      }
    )

  result = _forward_animation(visualizer)

  interval = max(1, frame_count // 10)
  for frame in range(frame_count):
    for environment in range(world_count):
      for object_name in ("geom", "site"):
        record_indices = layouts[environment][object_name]
        object_ids = records["object_ids"][record_indices]
        positions[record_indices] = result[f"{object_name}_xpos"][frame, environment, object_ids] + offsets[environment]
        matrices[record_indices] = result[f"{object_name}_xmat"][frame, environment, object_ids]

    orientations = st.Rotation.from_matrix(matrices).as_quat(canonical=True).astype(np.float32)
    if previous_orientations is not None:
      flip = np.sum(previous_orientations * orientations, axis=-1) < 0.0
      orientations[flip] *= -1.0
    previous_orientations = orientations.copy()

    for group in groups:
      selection = group["selection"]
      group_positions = np.ascontiguousarray(positions[selection])
      group_orientations = np.ascontiguousarray(orientations[selection], dtype=np.float16)
      group["positions"].Set(Vt.Vec3fArray.FromNumpy(group_positions), frame)
      group["orientations"].Set(Vt.QuathArray.FromNumpy(group_orientations), frame)

    if frame + 1 == frame_count or (frame + 1) % interval == 0:
      print(f"[viz] encoded {frame + 1}/{frame_count} frames for {world_count} environments")

  trace_rollouts = _trace_rollouts(visualizer, result)
  force_rollouts = _contact_rollouts(visualizer, result)
  _add_overlays(visualizer, stage, offsets, force_rollouts, trace_rollouts)


def _add_overlays(visualizer, stage, offsets, force_rollouts, trace_rollouts):
  """Adds traces and optional contact forces outside the instanced geometry."""
  from pxr import Gf
  from pxr import UsdGeom

  if any(any(traces) for _, _, _, traces in trace_rollouts):
    root = UsdGeom.Xform.Define(stage, "/World/Overlays")
    _add_traces(
      stage,
      trace_rollouts,
      offsets,
      root.GetPath(),
      visualizer.trace_width,
      visualizer.trace_alpha,
    )

  for environment, offset in enumerate(offsets):
    has_forces = force_rollouts is not None and any(force_rollouts[environment])
    if not has_forces:
      continue
    root = UsdGeom.Xform.Define(stage, f"/World/Overlays/env_{environment:04d}")
    root.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in offset)))
    _add_contact_forces(stage, force_rollouts[environment], root.GetPath())


def _add_traces(stage, rollouts, offsets, root, width, alpha):
  """Adds one batched trajectory-curve primitive for each recorded rollout."""
  from pxr import UsdGeom
  from pxr import Vt

  start = 0
  trace_id = 0
  for frames, _, _, environment_traces in rollouts:
    traces = [
      points + offsets[environment]
      for environment, body_traces in enumerate(environment_traces)
      if body_traces
      for points in body_traces
    ]
    if not traces:
      start += frames.shape[1]
      continue
    for segment in range(1, frames.shape[1]):
      points = np.ascontiguousarray(np.concatenate([trace[segment - 1 : segment + 1] for trace in traces]), dtype=np.float32)
      curve = UsdGeom.BasisCurves.Define(stage, f"{root}/trace_{trace_id}_{segment:04d}")
      curve.CreateTypeAttr(UsdGeom.Tokens.linear)
      curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
      curve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
      curve.CreateCurveVertexCountsAttr([2] * len(traces))
      curve.CreateWidthsAttr([width] * len(points))
      curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
      curve.CreateDisplayColorAttr([(0.95, 0.55, 0.1)])
      curve.CreateDisplayOpacityAttr([alpha])
      visibility = curve.GetVisibilityAttr()
      visibility.Set(UsdGeom.Tokens.invisible, 0)
      visibility.Set(UsdGeom.Tokens.inherited, start + segment)
    start += frames.shape[1]
    trace_id += 1


def _add_contact_forces(stage, rollouts, root):
  """Adds animated force arrows unsupported by MuJoCo's USD exporter."""
  from pxr import Gf
  from pxr import UsdGeom
  from pxr import Vt

  start = 0
  arrow_id = 0
  end_time = int(stage.GetEndTimeCode())
  for frames in rollouts:
    slots = max((len(arrows) for arrows in frames), default=0)
    for slot in range(slots):
      curve = UsdGeom.BasisCurves.Define(stage, f"{root}/contact_force_{arrow_id}")
      curve.CreateTypeAttr(UsdGeom.Tokens.linear)
      curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
      curve.CreateCurveVertexCountsAttr([2, 3])
      points_attr = curve.CreatePointsAttr()
      widths_attr = curve.CreateWidthsAttr()
      curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
      color_attr = curve.CreateDisplayColorAttr()
      visibility = curve.GetVisibilityAttr()
      if start:
        visibility.Set(UsdGeom.Tokens.invisible, 0)

      color_set = False
      for local_frame, arrows in enumerate(frames):
        frame = start + local_frame
        if slot >= len(arrows):
          visibility.Set(UsdGeom.Tokens.invisible, frame)
          continue
        start_point, end_point, radius, rgba = arrows[slot]
        direction = end_point - start_point
        length = float(np.linalg.norm(direction))
        if length <= 1.0e-8:
          visibility.Set(UsdGeom.Tokens.invisible, frame)
          continue
        direction /= length
        reference = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        if abs(float(np.dot(direction, reference))) > 0.9:
          reference = np.array((1.0, 0.0, 0.0), dtype=np.float32)
        side = np.cross(direction, reference)
        side /= np.linalg.norm(side)
        head = min(0.2 * length, 0.12)
        wing = 0.4 * head * side
        base = end_point - head * direction
        points = (start_point, end_point, base + wing, end_point, base - wing)
        points_attr.Set(
          Vt.Vec3fArray([Gf.Vec3f(*(float(value) for value in point)) for point in points]),
          frame,
        )
        widths_attr.Set(Vt.FloatArray([max(2.0 * radius, 0.012)] * len(points)), frame)
        visibility.Set(UsdGeom.Tokens.inherited, frame)
        if not color_set:
          color_attr.Set([tuple(float(value) for value in rgba[:3])])
          color_set = True
      stop = start + len(frames)
      if stop <= end_time:
        visibility.Set(UsdGeom.Tokens.invisible, stop)
      arrow_id += 1
    start += len(frames)


def _rebase_assets(stage, source_directory, destination_directory):
  """Rebases relative assets when the final stage moves out of its package."""
  from pxr import Sdf

  for prim in stage.Traverse():
    for attribute in prim.GetAttributes():
      if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
        continue
      asset = attribute.Get()
      if not asset or not asset.path or Path(asset.path).is_absolute():
        continue
      source = Path(source_directory, asset.path).resolve()
      relative = os.path.relpath(source, destination_directory)
      attribute.Set(Sdf.AssetPath(relative))


def _clean_for_blender(stage):
  """Removes MuJoCo attributes that Blender imports incorrectly."""
  from pxr import UsdGeom

  for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Xformable):
      xform = UsdGeom.Xformable(prim)
      ops = xform.GetOrderedXformOps()
      transforms = [op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTransform]
      if transforms and len(transforms) != len(ops):
        xform.SetXformOpOrder(transforms)
    if prim.IsA(UsdGeom.Imageable):
      visibility = UsdGeom.Imageable(prim).GetVisibilityAttr()
      if visibility and visibility.GetNumTimeSamples():
        visibility.Clear()
        visibility.Set(UsdGeom.Tokens.inherited)


def _fix_shader_input_types(stage):
  """Uses the USD-declared type for primvar-reader names."""
  from pxr import Sdf
  from pxr import UsdShade

  for prim in stage.Traverse():
    if not prim.IsA(UsdShade.Shader):
      continue
    shader = UsdShade.Shader(prim)
    if shader.GetIdAttr().Get() != "UsdPrimvarReader_float2":
      continue
    varname = shader.GetInput("varname")
    if not varname or varname.GetTypeName() == Sdf.ValueTypeNames.String:
      continue
    value = varname.Get()
    prim.RemoveProperty("inputs:varname")
    varname = shader.CreateInput("varname", Sdf.ValueTypeNames.String)
    if value is not None:
      varname.Set(str(value))


def _box_cubemap_uvs(vertices, triangles):
  """Maps box triangles onto MuJoCo's vertical R, L, U, D, F, B strip."""
  center = np.mean(vertices, axis=0)
  triangle_uvs = []
  for triangle in triangles:
    relative = vertices[triangle] - center
    normal = np.cross(relative[1] - relative[0], relative[2] - relative[0])
    axis = int(np.argmax(np.abs(normal)))
    positive = np.mean(relative[:, axis]) > 0.0
    face = 2 * axis + int(not positive)

    for x, y, z in relative:
      if axis == 0:
        denominator = abs(x)
        u = (-z if positive else z) / denominator
        v = y / denominator
      elif axis == 1:
        denominator = abs(y)
        u = x / denominator
        v = (-z if positive else z) / denominator
      else:
        denominator = abs(z)
        u = (x if positive else -x) / denominator
        v = y / denominator
      triangle_uvs.append(((u + 1.0) / 2.0, (face + (v + 1.0) / 2.0) / 6.0))
  return np.asarray(triangle_uvs)


def _fix_box_cubemaps(exporter):
  """Repairs MuJoCo USD box UVs, which otherwise select one cubemap face."""
  rgb_role = mujoco.mjtTextureRole.mjTEXROLE_RGB.value
  for usd_object in exporter.geom_refs.values():
    textures = usd_object.geom_textures
    rgb_texture = textures[rgb_role] if len(textures) > rgb_role else None
    if (
      usd_object.geom.type != mujoco.mjtGeom.mjGEOM_BOX
      or rgb_texture is None
      or rgb_texture[1] != mujoco.mjtTexture.mjTEXTURE_CUBE
    ):
      continue
    primitive = getattr(usd_object, "prim_mesh", None)
    texcoords = getattr(usd_object, "texcoords", None)
    if primitive is not None and texcoords is not None:
      texcoords.Set(_box_cubemap_uvs(primitive.vertices, primitive.triangles))
