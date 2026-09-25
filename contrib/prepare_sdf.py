# Copyright 2026 The MuJoCo Authors.
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

"""Compile native SDF meshes at a chosen depth and correct their signed corner distances.

This is an offline build step, not part of the benchmark runtime. MuJoCo's XML schema
does not currently persist MjsMesh.octree_maxdepth. Its native mesh field can also have
incorrect signs for concave meshes. This utility uses the compiled triangle geometry
to recompute corner distances with Warp's winding-number query on the CPU.

Example:
  uv run contrib/prepare_sdf.py scene.xml --depth 8 --output /tmp/scene.mjb

The generated MJB is specific to the installed MuJoCo version. Build it again after
upgrading MuJoCo. This tool does not apply any MJWarp collision-kernel changes.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import mujoco
import numpy as np
import warp as wp


@wp.kernel
def _signed_distance(mesh: wp.uint64, points: wp.array[wp.vec3], radius: float, distances_out: wp.array[float]):
  index = wp.tid()
  query = wp.mesh_query_point_sign_winding_number(mesh, points[index], radius, 4.0, 0.5)
  if query.result:
    closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    distances_out[index] = query.sign * wp.length(points[index] - closest)
  else:
    distances_out[index] = wp.nan


def correct_distances(model):
  """Replace only native SDF corner coefficients; return per-mesh build evidence."""
  evidence = {}
  corner_signs = np.array([[1 if j & (1 << k) else -1 for k in range(3)] for j in range(8)])
  mesh_ids = np.unique(model.geom_dataid[model.geom_type == mujoco.mjtGeom.mjGEOM_SDF])
  for mesh_id in mesh_ids:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
    root, count = int(model.mesh_octadr[mesh_id]), int(model.mesh_octnum[mesh_id])
    if root < 0 or count == 0:
      raise ValueError(f"SDF mesh {name!r} has no native octree")
    va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    vertices, faces = model.mesh_vert[va : va + vn], model.mesh_face[fa : fa + fn]
    fingerprint = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()
    side = 1 << int(model.oct_depth[root : root + count].max())
    boxes = model.oct_aabb[root : root + count].reshape(-1, 2, 3)
    root_min, spacing = boxes[0, 0] - boxes[0, 1], 2 * boxes[0, 1] / side
    corners = boxes[:, None, 0] + boxes[:, None, 1] * corner_signs
    coordinates = np.rint((corners - root_min) / spacing).astype(np.int64)
    packed = coordinates[:, :, 0] + (side + 1) * (coordinates[:, :, 1] + (side + 1) * coordinates[:, :, 2])
    unique, inverse = np.unique(packed, return_inverse=True)
    grid = np.stack((unique % (side + 1), unique // (side + 1) % (side + 1), unique // (side + 1) ** 2), axis=1)
    points = np.asarray(root_min + grid * spacing, dtype=np.float32)
    del corners, coordinates, packed, grid, unique
    with wp.ScopedDevice("cpu"):
      query_mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3), indices=wp.array(faces.ravel(), dtype=int), support_winding_number=True
      )
      queries = wp.array(points, dtype=wp.vec3)
      distances = wp.empty(len(points), dtype=float)
      radius = float(2 * np.linalg.norm(2 * boxes[0, 1]) + 1)
      wp.launch(_signed_distance, len(points), inputs=[query_mesh.id, queries, radius], outputs=[distances])
      corrected = distances.numpy()[inverse].reshape(count, 8)
    if not np.isfinite(corrected).all():
      raise ValueError(f"Nonfinite triangle distances for SDF mesh {name!r}")
    original = model.oct_coeff[root : root + count]
    away = np.minimum(np.abs(corrected), np.abs(original)) > 1e-6
    changed_signs = np.count_nonzero((np.sign(corrected) != np.sign(original)) & away)
    evidence[name] = {
      "geometry_sha256": fingerprint,
      "octree_root": root,
      "octree_nodes": count,
      "unique_corners": len(points),
      "vertices": int(vn),
      "triangles": int(fn),
      "sign_changes_away_from_surface": int(changed_signs),
      "maximum_distance_change": float(np.abs(corrected - original).max()),
    }
    original[:] = corrected
  return evidence


def prepare(source, output, depth=8):
  """Build a native SDF MJB and adjacent provenance JSON without overwriting artifacts."""
  source, output = Path(source).resolve(), Path(output).resolve()
  report_path = output.with_suffix(".json")
  if output.suffix != ".mjb" or output.exists() or report_path.exists():
    raise ValueError("Choose a new .mjb output path with no existing .mjb or .json")
  if not 1 <= depth <= 10:
    raise ValueError("Octree depth must be between 1 and 10")
  spec = mujoco.MjSpec.from_file(str(source))
  sdf_names = {geom.meshname for geom in spec.geoms if geom.type == mujoco.mjtGeom.mjGEOM_SDF}
  if not sdf_names:
    raise ValueError("The scene has no native SDF meshes")
  mesh_sources = {}
  for mesh in spec.meshes:
    if mesh.name in sdf_names:
      mesh.octree_maxdepth = depth
    if mesh.file:
      path = (source.parent / spec.meshdir / mesh.file).resolve()
      mesh_sources[mesh.name] = {"file": mesh.file, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
  report = {
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "expanded_xml_sha256": hashlib.sha256(spec.to_xml().encode()).hexdigest(),
    "mesh_sources": mesh_sources,
    "mujoco_version": importlib.metadata.version("mujoco"),
    "warp_version": importlib.metadata.version("warp-lang"),
    "octree_maxdepth": depth,
    "method": "CPU triangle distance; winding-number sign, accuracy=4, threshold=0.5",
    "changed_model_fields": ["oct_coeff"],
    "runtime_collision_patches_included": False,
  }
  model = spec.compile()
  report["meshes"] = correct_distances(model)
  report["model_sizes"] = {name: int(getattr(model, name)) for name in ("nq", "nv", "nu", "ngeom", "nmesh", "noct")}
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("xb"):
    pass
  mujoco.mj_saveModel(model, str(output))
  with output.open("rb") as file:
    digest = hashlib.sha256()
    for chunk in iter(lambda: file.read(1024 * 1024), b""):
      digest.update(chunk)
    report["model_sha256"] = digest.hexdigest()
  with report_path.open("x") as file:
    json.dump(report, file, indent=2)
    file.write("\n")
  return report


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path, help="Local scene XML and its referenced assets")
  parser.add_argument("--output", required=True, type=Path, help="New compiled .mjb path; JSON is written beside it")
  parser.add_argument("--depth", type=int, default=8, help="Maximum native SDF octree depth (default: 8)")
  args = parser.parse_args()
  print(json.dumps(prepare(args.source, args.output, args.depth), indent=2), flush=True)
