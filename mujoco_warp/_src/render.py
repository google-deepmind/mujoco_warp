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

from typing import Tuple

import warp as wp

from mujoco_warp._src import bvh
from mujoco_warp._src import math
from mujoco_warp._src.ray import ray_box
from mujoco_warp._src.ray import ray_capsule
from mujoco_warp._src.ray import ray_cylinder
from mujoco_warp._src.ray import ray_ellipsoid
from mujoco_warp._src.ray import ray_flex_with_bvh
from mujoco_warp._src.ray import ray_mesh_with_bvh
from mujoco_warp._src.ray import ray_mesh_with_bvh_anyhit
from mujoco_warp._src.ray import ray_plane
from mujoco_warp._src.ray import ray_sphere
from mujoco_warp._src.render_context import RenderContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import ProjectionType
from mujoco_warp._src.warp_util import event_scope
from mujoco_warp._src.warp_util import nested_kernel

wp.set_module_options({"enable_backward": False})

MAX_NUM_VIEWS_PER_THREAD = 8

BACKGROUND_COLOR = 255 << 24 | int(0.1 * 255.0) << 16 | int(0.1 * 255.0) << 8 | int(0.2 * 255.0)

SPOT_INNER_COS = float(0.95)
SPOT_OUTER_COS = float(0.85)
INV_255 = float(1.0 / 255.0)
SHADOW_MIN_VISIBILITY = float(0.3)  # reduce shadow darkness (0: full black, 1: no shadow)

AMBIENT_UP = wp.vec3(0.0, 0.0, 1.0)
AMBIENT_SKY = wp.vec3(0.4, 0.4, 0.45)
AMBIENT_GROUND = wp.vec3(0.1, 0.1, 0.12)
AMBIENT_INTENSITY = float(0.5)

TILE_W: int = 16
TILE_H: int = 16
THREADS_PER_TILE: int = TILE_W * TILE_H


@wp.func
def _ceil_div(a: int, b: int):
  return (a + b - 1) // b


# Map linear thread id (per image) -> (px, py) using TILE_W x TILE_H tiles
@wp.func
def _tile_coords(tid: int, W: int, H: int):
  tile_id = tid // THREADS_PER_TILE
  local = tid - tile_id * THREADS_PER_TILE

  u = local % TILE_W
  v = local // TILE_W

  tiles_x = _ceil_div(W, TILE_W)
  tile_x = (tile_id % tiles_x) * TILE_W
  tile_y = (tile_id // tiles_x) * TILE_H

  i = tile_x + u
  j = tile_y + v
  return i, j


@event_scope
def render(m: Model, d: Data, rc: RenderContext):
  """Render the current frame.

  Outputs are stored in buffers within the render context.

  Args:
    m: The model on device.
    d: The data on device.
    rc: The render context on device.
  """
  bvh.refit_scene_bvh(m, d, rc)
  if m.nflex:
    bvh.refit_flex_bvh(m, d, rc)
  render_megakernel(m, d, rc)


@wp.func
def compute_ray(
  # In:
  projection: int,
  fovy: float,
  sensorsize: wp.vec2,
  intrinsic: wp.vec4,
  img_w: int,
  img_h: int,
  px: int,
  py: int,
  znear: float,
) -> wp.vec3:
  """Compute ray direction for a pixel with per-world camera parameters.

  This combines _camera_frustum_bounds and build_primary_rays logic for use
  inside a kernel when camera parameters are batched/randomized across worlds.
  """
  if projection == ProjectionType.ORTHOGRAPHIC:
    return wp.vec3(0.0, 0.0, -1.0)

  aspect = float(img_w) / float(img_h)
  sensor_h = sensorsize[1]

  # Check if we have intrinsics (sensorsize[1] != 0)
  if sensor_h != 0.0:
    fx = intrinsic[0]
    fy = intrinsic[1]
    cx = intrinsic[2]
    cy = intrinsic[3]
    sensor_w = sensorsize[0]

    target_aspect = float(img_w) / float(img_h)
    sensor_aspect = sensor_w / sensor_h
    if target_aspect > sensor_aspect:
      sensor_h = sensor_w / target_aspect
    elif target_aspect < sensor_aspect:
      sensor_w = sensor_h * target_aspect

    left = -znear / fx * (sensor_w * 0.5 - cx)
    right = znear / fx * (sensor_w * 0.5 + cx)
    top = znear / fy * (sensor_h * 0.5 - cy)
    bottom = -znear / fy * (sensor_h * 0.5 + cy)
  else:
    fovy_rad = fovy * wp.pi / 180.0
    half_height = znear * wp.tan(0.5 * fovy_rad)
    half_width = half_height * aspect
    left = -half_width
    right = half_width
    top = half_height
    bottom = -half_height

  u = (float(px) + 0.5) / float(img_w)
  v = (float(py) + 0.5) / float(img_h)
  x = left + (right - left) * u
  y = top + (bottom - top) * v

  return wp.normalize(wp.vec3(x, y, -znear))


@wp.func
def pack_rgba_to_uint32(r: wp.uint8, g: wp.uint8, b: wp.uint8, a: wp.uint8) -> wp.uint32:
  """Pack RGBA values into a single uint32 for efficient memory access."""
  return (wp.uint32(a) << wp.uint32(24)) | (wp.uint32(r) << wp.uint32(16)) | (wp.uint32(g) << wp.uint32(8)) | wp.uint32(b)


@wp.func
def pack_rgba_to_uint32(r: float, g: float, b: float, a: float) -> wp.uint32:
  """Pack RGBA values into a single uint32 for efficient memory access."""
  return (wp.uint32(a) << wp.uint32(24)) | (wp.uint32(r) << wp.uint32(16)) | (wp.uint32(g) << wp.uint32(8)) | wp.uint32(b)


@wp.func
def sample_texture_2d(
  # In:
  uv: wp.vec2,
  width: int,
  height: int,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
) -> wp.vec3:
  ix = wp.min(width - 1, int(uv[0] * float(width)))
  iy = wp.min(height - 1, int(uv[1] * float(height)))
  linear_idx = tex_adr + (iy * width + ix)
  packed_rgba = tex_data[linear_idx]
  r = float((packed_rgba >> wp.uint32(16)) & wp.uint32(0xFF)) * INV_255
  g = float((packed_rgba >> wp.uint32(8)) & wp.uint32(0xFF)) * INV_255
  b = float(packed_rgba & wp.uint32(0xFF)) * INV_255
  return wp.vec3(r, g, b)


@wp.func
def sample_texture_plane(
  # In:
  hit_point: wp.vec3,
  pos: wp.vec3,
  rot: wp.mat33,
  tex_repeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
  tex_height: int,
  tex_width: int,
) -> wp.vec3:
  local = wp.transpose(rot) @ (hit_point - pos)
  u = local[0] * tex_repeat[0]
  v = local[1] * tex_repeat[1]
  u = u - wp.floor(u)
  v = v - wp.floor(v)
  v = 1.0 - v
  return sample_texture_2d(
    wp.vec2(u, v),
    tex_width,
    tex_height,
    tex_adr,
    tex_data,
  )


@wp.func
def sample_texture_mesh(
  # In:
  bary_u: float,
  bary_v: float,
  uv_baseadr: int,
  v_idx: wp.vec3i,
  mesh_texcoord: wp.array(dtype=wp.vec2),
  tex_repeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
  tex_height: int,
  tex_width: int,
) -> wp.vec3:
  bw = 1.0 - bary_u - bary_v
  uv0 = mesh_texcoord[uv_baseadr + v_idx.x]
  uv1 = mesh_texcoord[uv_baseadr + v_idx.y]
  uv2 = mesh_texcoord[uv_baseadr + v_idx.z]
  uv = uv0 * bw + uv1 * bary_u + uv2 * bary_v
  u = uv[0] * tex_repeat[0]
  v = uv[1] * tex_repeat[1]
  u = u - wp.floor(u)
  v = v - wp.floor(v)
  v = 1.0 - v
  return sample_texture_2d(
    wp.vec2(u, v),
    tex_width,
    tex_height,
    tex_adr,
    tex_data,
  )


@wp.func
def sample_texture(
  # Model:
  geom_type: wp.array(dtype=int),
  mesh_faceadr: wp.array(dtype=int),
  mesh_face: wp.array(dtype=wp.vec3i),
  # In:
  geom_id: int,
  tex_repeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
  tex_height: int,
  tex_width: int,
  pos: wp.vec3,
  rot: wp.mat33,
  mesh_texcoord: wp.array(dtype=wp.vec2),
  mesh_texcoord_offsets: wp.array(dtype=int),
  hit_point: wp.vec3,
  u: float,
  v: float,
  f: int,
  mesh_id: int,
) -> wp.vec3:
  tex_color = wp.vec3(1.0, 1.0, 1.0)

  if geom_type[geom_id] == GeomType.PLANE:
    tex_color = sample_texture_plane(
      hit_point,
      pos,
      rot,
      tex_repeat,
      tex_adr,
      tex_data,
      tex_height,
      tex_width,
    )

  if geom_type[geom_id] == GeomType.MESH:
    if f < 0 or mesh_id < 0:
      return tex_color

    base_face = mesh_faceadr[mesh_id]
    uv_base = mesh_texcoord_offsets[mesh_id]
    face_global = base_face + f
    tex_color = sample_texture_mesh(
      u,
      v,
      uv_base,
      mesh_face[face_global],
      mesh_texcoord,
      tex_repeat,
      tex_adr,
      tex_data,
      tex_height,
      tex_width,
    )

  return tex_color


@wp.func
def cast_ray(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  bvh_id: wp.uint64,
  group_root: int,
  world_id: int,
  bvh_ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
) -> Tuple[int, float, wp.vec3, float, float, int, int]:
  dist = float(wp.inf)
  normal = wp.vec3(0.0, 0.0, 0.0)
  geom_id = int(-1)
  bary_u = float(0.0)
  bary_v = float(0.0)
  face_idx = int(-1)
  geom_mesh_id = int(-1)

  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
  bounds_nr = int(0)

  while wp.bvh_query_next(query, bounds_nr, dist):
    gi_global = bounds_nr
    gi_bvh_local = gi_global - (world_id * bvh_ngeom)
    gi = enabled_geom_ids[gi_bvh_local]

    # TODO: Investigate branch elimination with static loop unrolling
    if geom_type[gi] == GeomType.PLANE:
      d, n = ray_plane(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.HFIELD:
      d, n, u, v, f, geom_hfield_id = ray_mesh_with_bvh(
        hfield_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        dist,
      )
    if geom_type[gi] == GeomType.SPHERE:
      d, n = ray_sphere(
        geom_xpos_in[world_id, gi],
        geom_size[world_id, gi][0] * geom_size[world_id, gi][0],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.ELLIPSOID:
      d, n = ray_ellipsoid(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CAPSULE:
      d, n = ray_capsule(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CYLINDER:
      d, n = ray_cylinder(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.BOX:
      d, all, n = ray_box(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.MESH:
      d, n, u, v, f, geom_mesh_id = ray_mesh_with_bvh(
        mesh_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        dist,
      )

    if d >= 0.0 and d < dist:
      dist = d
      normal = n
      geom_id = gi
      bary_u = u
      bary_v = v
      face_idx = f

  return geom_id, dist, normal, bary_u, bary_v, face_idx, geom_mesh_id


@wp.func
def cast_ray_first_hit(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  bvh_id: wp.uint64,
  group_root: int,
  world_id: int,
  bvh_ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
  max_dist: float,
) -> bool:
  """A simpler version of casting rays that only checks for the first hit."""
  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
  bounds_nr = int(0)

  while wp.bvh_query_next(query, bounds_nr, max_dist):
    gi_global = bounds_nr
    gi_bvh_local = gi_global - (world_id * bvh_ngeom)
    gi = enabled_geom_ids[gi_bvh_local]

    # TODO: Investigate branch elimination with static loop unrolling
    if geom_type[gi] == GeomType.PLANE:
      d, n = ray_plane(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.HFIELD:
      d, n, u, v, f, geom_hfield_id = ray_mesh_with_bvh(
        hfield_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        max_dist,
      )
    if geom_type[gi] == GeomType.SPHERE:
      d, n = ray_sphere(
        geom_xpos_in[world_id, gi],
        geom_size[world_id, gi][0] * geom_size[world_id, gi][0],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.ELLIPSOID:
      d, n = ray_ellipsoid(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CAPSULE:
      d, n = ray_capsule(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CYLINDER:
      d, n = ray_cylinder(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.BOX:
      d, all, n = ray_box(
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.MESH:
      hit = ray_mesh_with_bvh_anyhit(
        mesh_bvh_id,
        geom_dataid[gi],
        geom_xpos_in[world_id, gi],
        geom_xmat_in[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        max_dist,
      )
      d = 0.0 if hit else -1.0

    if d >= 0.0 and d < max_dist:
      return True

  return False


@wp.func
def compute_lighting(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  use_shadows: bool,
  bvh_id: wp.uint64,
  group_root: int,
  bvh_ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  world_id: int,
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  lightactive: bool,
  lighttype: int,
  lightcastshadow: bool,
  lightpos: wp.vec3,
  lightdir: wp.vec3,
  normal: wp.vec3,
  hitpoint: wp.vec3,
) -> float:
  light_contribution = float(0.0)

  # TODO: We should probably only be looping over active lights
  # in the first place with a static loop of enabled light idx?
  if not lightactive:
    return light_contribution

  L = wp.vec3(0.0, 0.0, 0.0)
  dist_to_light = float(wp.inf)
  attenuation = float(1.0)

  if lighttype == 1:  # directional light
    L = wp.normalize(-lightdir)
  else:
    L, dist_to_light = math.normalize_with_norm(lightpos - hitpoint)
    attenuation = 1.0 / (1.0 + 0.02 * dist_to_light * dist_to_light)
    if lighttype == 0:  # spot light
      spot_dir = wp.normalize(lightdir)
      cos_theta = wp.dot(-L, spot_dir)
      inner = SPOT_INNER_COS
      outer = SPOT_OUTER_COS
      spot_factor = wp.min(1.0, wp.max(0.0, (cos_theta - outer) / (inner - outer)))
      attenuation = attenuation * spot_factor

  ndotl = wp.max(0.0, wp.dot(normal, L))
  if ndotl == 0.0:
    return light_contribution

  visible = float(1.0)

  if use_shadows and lightcastshadow:
    # Nudge the origin slightly along the surface normal to avoid
    # self-intersection when casting shadow rays
    eps = 1.0e-4
    shadow_origin = hitpoint + normal * eps
    # Distance-limited shadows: cap by dist_to_light (for non-directional)
    max_t = float(dist_to_light - 1.0e-3)
    if lighttype == 1:  # directional light
      max_t = float(1.0e8)

    shadow_hit = cast_ray_first_hit(
      geom_type,
      geom_dataid,
      geom_size,
      geom_xpos_in,
      geom_xmat_in,
      bvh_id,
      group_root,
      world_id,
      bvh_ngeom,
      enabled_geom_ids,
      mesh_bvh_id,
      hfield_bvh_id,
      shadow_origin,
      L,
      max_t,
    )

    if shadow_hit:
      visible = SHADOW_MIN_VISIBILITY

  return ndotl * attenuation * visible


@event_scope
def render_megakernel(m: Model, d: Data, rc: RenderContext):
  rc.rgb_data.fill_(wp.uint32(BACKGROUND_COLOR))
  rc.depth_data.fill_(0.0)

  # TODO: Adding "unique" causes kernel re-compilation issues, need to investigate
  # and fix it.
  @nested_kernel(enable_backward="False")
  def _render_megakernel(
    # Model:
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_matid: wp.array2d(dtype=int),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    mesh_faceadr: wp.array(dtype=int),
    mesh_face: wp.array(dtype=wp.vec3i),
    mat_texid: wp.array3d(dtype=int),
    mat_texrepeat: wp.array2d(dtype=wp.vec2),
    mat_rgba: wp.array2d(dtype=wp.vec4),
    light_active: wp.array2d(dtype=bool),
    light_type: wp.array2d(dtype=int),
    light_castshadow: wp.array2d(dtype=bool),
    cam_projection: wp.array(dtype=int),
    cam_fovy: wp.array2d(dtype=float),
    cam_sensorsize: wp.array(dtype=wp.vec2),
    cam_intrinsic: wp.array2d(dtype=wp.vec4),
    # Data in:
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),
    light_xpos: wp.array2d(dtype=wp.vec3),
    light_xdir: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    # In:
    ncam: int,
    use_shadows: bool,
    bvh_ngeom: int,
    cam_res: wp.array(dtype=wp.vec2i),
    cam_id_map: wp.array(dtype=int),
    ray: wp.array(dtype=wp.vec3),
    rgb_adr: wp.array(dtype=int),
    depth_adr: wp.array(dtype=int),
    render_rgb: wp.array(dtype=bool),
    render_depth: wp.array(dtype=bool),
    bvh_id: wp.uint64,
    group_root: wp.array(dtype=int),
    flex_bvh_id: wp.uint64,
    flex_group_root: wp.array(dtype=int),
    enabled_geom_ids: wp.array(dtype=int),
    mesh_bvh_id: wp.array(dtype=wp.uint64),
    mesh_texcoord: wp.array(dtype=wp.vec2),
    mesh_texcoord_offsets: wp.array(dtype=int),
    hfield_bvh_id: wp.array(dtype=wp.uint64),
    flex_rgba: wp.array(dtype=wp.vec4),
    tex_adr: wp.array(dtype=int),
    tex_data: wp.array(dtype=wp.uint32),
    tex_height: wp.array(dtype=int),
    tex_width: wp.array(dtype=int),
    # Out:
    rgb_out: wp.array2d(dtype=wp.uint32),
    depth_out: wp.array2d(dtype=float),
  ):
    world_idx, ray_idx = wp.tid()

    # Map global ray_idx -> (cam_idx, ray_idx_local) using cumulative sizes
    cam_idx = int(-1)
    ray_idx_local = int(-1)
    accum = int(0)
    for i in range(ncam):
      num_i = cam_res[i][0] * cam_res[i][1]
      if ray_idx < accum + num_i:
        cam_idx = i
        ray_idx_local = ray_idx - accum
        break
      accum += num_i
    if cam_idx == -1 or ray_idx_local < 0:
      return

    if not render_rgb[cam_idx] and not render_depth[cam_idx]:
      return

    # Map active camera index to MuJoCo camera ID
    mujoco_cam_id = cam_id_map[cam_idx]

    if wp.static(rc.ray is None):
      img_w = cam_res[cam_idx][0]
      img_h = cam_res[cam_idx][1]
      px = ray_idx_local % img_w
      py = ray_idx_local // img_w
      ray_dir_local_cam = compute_ray(
        cam_projection[mujoco_cam_id],
        cam_fovy[world_idx, mujoco_cam_id],
        cam_sensorsize[mujoco_cam_id],
        cam_intrinsic[world_idx, mujoco_cam_id],
        img_w,
        img_h,
        px,
        py,
        wp.static(rc.znear),
      )
    else:
      ray_dir_local_cam = ray[ray_idx]

    ray_dir_world = cam_xmat[world_idx, mujoco_cam_id] @ ray_dir_local_cam
    ray_origin_world = cam_xpos[world_idx, mujoco_cam_id]

    geom_id, dist, normal, u, v, f, mesh_id = cast_ray(
      geom_type,
      geom_dataid,
      geom_size,
      geom_xpos,
      geom_xmat,
      bvh_id,
      group_root[world_idx],
      world_idx,
      bvh_ngeom,
      enabled_geom_ids,
      mesh_bvh_id,
      hfield_bvh_id,
      ray_origin_world,
      ray_dir_world,
    )

    if wp.static(m.nflex > 0):
      d, n, u, v, f = ray_flex_with_bvh(
        flex_bvh_id,
        flex_group_root[world_idx],
        ray_origin_world,
        ray_dir_world,
        dist,
      )
      if d >= 0.0 and d < dist:
        dist = d
        normal = n
        geom_id = -2

    # Early Out
    if geom_id == -1:
      return

    if render_depth[cam_idx]:
      depth_out[world_idx, depth_adr[cam_idx] + ray_idx_local] = dist

    if not render_rgb[cam_idx]:
      return

    # Shade the pixel
    hit_point = ray_origin_world + ray_dir_world * dist

    if geom_id == -2:
      # TODO: Currently flex textures are not supported, and only the first rgba value
      # is used until further flex support is added.
      color = flex_rgba[0]
    elif geom_matid[world_idx, geom_id] == -1:
      color = geom_rgba[world_idx, geom_id]
    else:
      color = mat_rgba[world_idx, geom_matid[world_idx, geom_id]]

    base_color = wp.vec3(color[0], color[1], color[2])
    hit_color = base_color

    if wp.static(rc.use_textures):
      if geom_id != -2:
        mat_id = geom_matid[world_idx, geom_id]
        if mat_id >= 0:
          tex_id = mat_texid[world_idx, mat_id, 1]
          if tex_id >= 0:
            tex_color = sample_texture(
              geom_type,
              mesh_faceadr,
              mesh_face,
              geom_id,
              mat_texrepeat[world_idx, mat_id],
              tex_adr[tex_id],
              tex_data,
              tex_height[tex_id],
              tex_width[tex_id],
              geom_xpos[world_idx, geom_id],
              geom_xmat[world_idx, geom_id],
              mesh_texcoord,
              mesh_texcoord_offsets,
              hit_point,
              u,
              v,
              f,
              mesh_id,
            )
            base_color = wp.cw_mul(base_color, tex_color)

    len_n = wp.length(normal)
    n = normal if len_n > 0.0 else AMBIENT_UP
    n = wp.normalize(n)
    hemispheric = 0.5 * (wp.dot(n, AMBIENT_UP) + 1.0)
    ambient_color = AMBIENT_SKY * hemispheric + AMBIENT_GROUND * (1.0 - hemispheric)
    result = AMBIENT_INTENSITY * wp.cw_mul(base_color, ambient_color)

    # Apply lighting and shadows
    for l in range(wp.static(m.nlight)):
      light_contribution = compute_lighting(
        geom_type,
        geom_dataid,
        geom_size,
        geom_xpos,
        geom_xmat,
        use_shadows,
        bvh_id,
        group_root[world_idx],
        bvh_ngeom,
        enabled_geom_ids,
        world_idx,
        mesh_bvh_id,
        hfield_bvh_id,
        light_active[world_idx, l],
        light_type[world_idx, l],
        light_castshadow[world_idx, l],
        light_xpos[world_idx, l],
        light_xdir[world_idx, l],
        normal,
        hit_point,
      )
      result = result + base_color * light_contribution

    hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))
    hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))

    rgb_out[world_idx, rgb_adr[cam_idx] + ray_idx_local] = pack_rgba_to_uint32(
      hit_color[0] * 255.0,
      hit_color[1] * 255.0,
      hit_color[2] * 255.0,
      255.0,
    )

  wp.launch(
    kernel=_render_megakernel,
    dim=(d.nworld, rc.total_rays),
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_matid,
      m.geom_size,
      m.geom_rgba,
      m.mesh_faceadr,
      m.mesh_face,
      m.mat_texid,
      m.mat_texrepeat,
      m.mat_rgba,
      m.light_active,
      m.light_type,
      m.light_castshadow,
      m.cam_projection,
      m.cam_fovy,
      m.cam_sensorsize,
      m.cam_intrinsic,
      d.cam_xpos,
      d.cam_xmat,
      d.light_xpos,
      d.light_xdir,
      d.geom_xpos,
      d.geom_xmat,
      rc.ncam,
      rc.use_shadows,
      rc.bvh_ngeom,
      rc.cam_res,
      rc.cam_id_map,
      rc.ray,
      rc.rgb_adr,
      rc.depth_adr,
      rc.render_rgb,
      rc.render_depth,
      rc.bvh_id,
      rc.group_root,
      rc.flex_bvh_id,
      rc.flex_group_root,
      rc.enabled_geom_ids,
      rc.mesh_bvh_id,
      rc.mesh_texcoord,
      rc.mesh_texcoord_offsets,
      rc.hfield_bvh_id,
      rc.flex_rgba,
      rc.tex_adr,
      rc.tex_data,
      rc.tex_height,
      rc.tex_width,
    ],
    outputs=[
      rc.rgb_data,
      rc.depth_data,
    ],
    block_dim=THREADS_PER_TILE,
  )
