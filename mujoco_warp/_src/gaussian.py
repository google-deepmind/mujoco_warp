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

from typing import Tuple

import numpy as np
import warp as wp

from mujoco_warp._src.types import MJ_MAXVAL

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_MAX_HITS = 32
_MIN_TRANSMITTANCE = 0.005
_SH_C0 = 0.28209479177387814

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


@wp.kernel
def _compute_gaussian_bounds(
  # In:
  transforms: wp.array[wp.transform],
  scales: wp.array[wp.vec3],
  rgba: wp.array[wp.vec4],
  min_response: float,
  # Out:
  lower_out: wp.array[wp.vec3],
  upper_out: wp.array[wp.vec3],
):
  tid = wp.tid()
  transform = transforms[tid]
  scale = scales[tid]
  opacity = wp.max(rgba[tid][3], 1.0e-6)
  response = wp.clamp(min_response / opacity, 1.0e-6, 0.97)
  radius = wp.sqrt(-2.0 * wp.log(response))

  lo = wp.vec3(MJ_MAXVAL)
  hi = wp.vec3(-MJ_MAXVAL)
  for x in range(2):
    for y in range(2):
      for z in range(2):
        corner = wp.vec3(
          scale[0] * radius * (2.0 * float(x) - 1.0),
          scale[1] * radius * (2.0 * float(y) - 1.0),
          scale[2] * radius * (2.0 * float(z) - 1.0),
        )
        point = wp.transform_point(transform, corner)
        lo = wp.min(lo, point)
        hi = wp.max(hi, point)

  lower_out[tid] = lo
  upper_out[tid] = hi


@wp.func
def ray_gaussian(
  # In:
  transform: wp.transform,
  scale: wp.vec3,
  opacity: float,
  min_response: float,
  pnt: wp.vec3,
  vec: wp.vec3,
  min_distance: float,
  max_distance: float,
) -> Tuple[float, float]:
  """Returns the distance and alpha at which a ray intersects a Gaussian."""
  inv_transform = wp.transform_inverse(transform)
  lpnt = wp.cw_div(wp.transform_point(inv_transform, pnt), scale)
  lvec = wp.cw_div(wp.transform_vector(inv_transform, vec), scale)

  distance = -wp.dot(lpnt, lvec) / wp.dot(lvec, lvec)
  if distance <= min_distance or distance >= max_distance:
    return -1.0, 0.0

  delta = lpnt + lvec * distance
  response = wp.exp(-0.5 * wp.dot(delta, delta))
  alpha = wp.min(response * opacity, 1.0)
  if alpha < min_response or alpha < wp.static(1.0 / 255.0):
    return -1.0, 0.0
  return distance, alpha


@wp.func
def shade_gaussians(
  # In:
  transforms: wp.array[wp.transform],
  scales: wp.array[wp.vec3],
  rgba: wp.array[wp.vec4],
  bvh_id: wp.uint64,
  min_response: float,
  ray_origin: wp.vec3,
  ray_direction: wp.vec3,
  max_distance: float,
) -> tuple[wp.vec3, float, float]:
  min_distance = float(0.0)
  transmittance = float(1.0)
  color = wp.vec3(0.0)
  depth = float(-1.0)

  hit_distances = wp.vector(MJ_MAXVAL, length=_MAX_HITS, dtype=float)
  hit_indices = wp.vector(-1, length=_MAX_HITS, dtype=int)
  hit_alphas = wp.vector(0.0, length=_MAX_HITS, dtype=float)

  while transmittance > wp.static(_MIN_TRANSMITTANCE):
    num_hits = int(0)
    for i in range(wp.static(_MAX_HITS)):
      hit_distances[i] = max_distance
      hit_indices[i] = -1
      hit_alphas[i] = 0.0

    index = int(0)
    query = wp.bvh_query_ray(bvh_id, ray_origin, ray_direction)
    while wp.bvh_query_next(query, index, hit_distances[_MAX_HITS - 1]):
      distance, alpha = ray_gaussian(
        transforms[index],
        scales[index],
        rgba[index][3],
        min_response,
        ray_origin,
        ray_direction,
        min_distance,
        max_distance,
      )
      if distance > 0.0:
        if num_hits < wp.static(_MAX_HITS):
          num_hits += 1
        for i in range(num_hits):
          if distance < hit_distances[i]:
            for j in range(num_hits - 1, i, -1):
              hit_distances[j] = hit_distances[j - 1]
              hit_indices[j] = hit_indices[j - 1]
              hit_alphas[j] = hit_alphas[j - 1]
            hit_distances[i] = distance
            hit_indices[i] = index
            hit_alphas[i] = alpha
            break

    if num_hits == 0:
      break

    for i in range(num_hits):
      index = hit_indices[i]
      alpha = hit_alphas[i]
      color += wp.vec3(rgba[index][0], rgba[index][1], rgba[index][2]) * alpha * transmittance
      transmittance *= 1.0 - alpha
      if depth < 0.0 and transmittance < wp.static(_MIN_TRANSMITTANCE):
        depth = hit_distances[i]

    if num_hits < wp.static(_MAX_HITS):
      break
    min_distance = hit_distances[_MAX_HITS - 1] + 1.0e-6

  return color, transmittance, depth


def create_gaussian_fields(
  positions,
  rotations=None,
  scales=None,
  rgba=None,
  min_response: float = 0.01,
) -> tuple:
  """Creates Gaussian splat fields on device.

  Rotations use Warp's ``(x, y, z, w)`` quaternion convention. Scales are the
  standard deviations of each Gaussian in metres.
  """
  positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3))
  count = positions.shape[0]
  if count == 0:
    raise ValueError("positions must contain at least one Gaussian")
  if rotations is None:
    rotations = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (count, 1))
  else:
    rotations = np.ascontiguousarray(np.asarray(rotations, dtype=np.float32).reshape(count, 4))
  if scales is None:
    scales = np.full((count, 3), 0.01, dtype=np.float32)
  else:
    scales = np.ascontiguousarray(np.asarray(scales, dtype=np.float32).reshape(count, 3))
  if rgba is None:
    rgba = np.ones((count, 4), dtype=np.float32)
  else:
    rgba = np.ascontiguousarray(np.asarray(rgba, dtype=np.float32).reshape(count, 4))

  if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
    raise ValueError("positions and rotations must be finite")
  if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
    raise ValueError("scales must be finite and positive")
  if not np.all(np.isfinite(rgba)) or np.any(rgba < 0.0) or np.any(rgba > 1.0):
    raise ValueError("rgba must be finite and in [0, 1]")
  if not np.isfinite(min_response) or not 0.0 < min_response < 1.0:
    raise ValueError("min_response must be finite and in (0, 1)")

  transforms = wp.array(np.concatenate((positions, rotations), axis=1), dtype=wp.transform)
  scales = wp.array(scales, dtype=wp.vec3)
  rgba = wp.array(rgba, dtype=wp.vec4)
  lower = wp.empty(count, dtype=wp.vec3)
  upper = wp.empty(count, dtype=wp.vec3)
  wp.launch(_compute_gaussian_bounds, dim=count, inputs=[transforms, scales, rgba, min_response], outputs=[lower, upper])
  bvh = wp.Bvh(lower, upper, constructor="sah")
  return transforms, scales, rgba, bvh, bvh.id, lower, upper, min_response, count


def load_gaussian_ply(filename) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Loads Gaussian splats from a binary little-endian 3DGS PLY file."""
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
      raise ValueError(f"PLY file is missing {prefix} Gaussian attributes")
    return np.column_stack([vertices[name] for name in required]).astype(np.float32)

  if not all(name in names for name in ("x", "y", "z", "opacity")):
    raise ValueError("PLY file is missing required Gaussian attributes")

  positions = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
  rotations_wxyz = fields("rot", 4)
  rotations = rotations_wxyz[:, [1, 2, 3, 0]]
  rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1.0e-12)
  scales = np.exp(fields("scale", 3))
  opacity = 1.0 / (1.0 + np.exp(-np.clip(vertices["opacity"], -80.0, 80.0)))
  color = np.clip(0.5 + _SH_C0 * fields("f_dc", 3), 0.0, 1.0)
  rgba = np.column_stack((color, opacity)).astype(np.float32)
  return positions, rotations, scales, rgba
