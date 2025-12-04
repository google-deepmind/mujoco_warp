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

from . import bvh
from . import math
from .ray import ray_box
from .ray import ray_box_with_normal
from .ray import ray_capsule
from .ray import ray_capsule_with_normal
from .ray import ray_cylinder
from .ray import ray_cylinder_with_normal
from .ray import ray_ellipsoid
from .ray import ray_ellipsoid_with_normal
from .ray import ray_flex_with_bvh
from .ray import ray_mesh_with_bvh
from .ray import ray_plane
from .ray import ray_plane_with_normal
from .ray import ray_sphere
from .ray import ray_sphere_with_normal
from .render_context import RenderContext
from .types import Data
from .types import GeomType
from .types import Model
from .warp_util import event_scope
from .warp_util import nested_kernel

wp.set_module_options({"enable_backward": False})

MAX_NUM_VIEWS_PER_THREAD = 8

BACKGROUND_COLOR = (
  255 << 24 |
  int(0.1 * 255.0) << 16 |
  int(0.1 * 255.0) << 8 |
  int(0.2 * 255.0)
)

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
  bvh.refit_warp_bvh(m, d, rc)
  if m.nflex:
    bvh.refit_flex_bvh(m, d, rc)
  render_megakernel(m, d, rc)


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
  hit_point: wp.vec3,
  geom_pos: wp.vec3,
  geom_rot: wp.mat33,
  mat_texrepeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
  tex_height: int,
  tex_width: int,
) -> wp.vec3:
  local = wp.transpose(geom_rot) @ (hit_point - geom_pos)
  u = local[0] * mat_texrepeat[0]
  v = local[1] * mat_texrepeat[1]
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
  bary_u: float,
  bary_v: float,
  uv_baseadr: int,
  v_idx: wp.vec3i,
  mesh_texcoord: wp.array(dtype=wp.vec2),
  mat_texrepeat: wp.vec2,
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
  u = uv[0] * mat_texrepeat[0]
  v = uv[1] * mat_texrepeat[1]
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
  world_id: int,
  geom_id: int,
  geom_type: wp.array(dtype=int),
  geom_matid: int,
  mat_texid: int,
  mat_texrepeat: wp.vec2,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
  tex_height: int,
  tex_width: int,
  geom_xpos: wp.vec3,
  geom_xmat: wp.mat33,
  mesh_faceadr: wp.array(dtype=int),
  mesh_face: wp.array(dtype=wp.vec3i),
  mesh_texcoord: wp.array(dtype=wp.vec2),
  mesh_texcoord_offsets: wp.array(dtype=int),
  hit_point: wp.vec3,
  u: float,
  v: float,
  f: int,
  mesh_id: int,
) -> wp.vec3:
  tex_color = wp.vec3(1.0, 1.0, 1.0)

  if geom_matid == -1 or mat_texid == -1:
    return tex_color

  if geom_type[geom_id] == GeomType.PLANE:
    tex_color = sample_texture_plane(
      hit_point,
      geom_xpos,
      geom_xmat,
      mat_texrepeat,
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
      mat_texrepeat,
      tex_adr,
      tex_data,
      tex_height,
      tex_width,
    )

  return tex_color


@wp.func
def cast_ray(
  bvh_id: wp.uint64,
  group_root: int,
  world_id: int,
  bvh_ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
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
      h, d, n = ray_plane_with_normal(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.HFIELD:
      h, d, n, u, v, f, geom_hfield_id = ray_mesh_with_bvh(
        hfield_bvh_id,
        geom_dataid[gi],
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        dist,
      )
    if geom_type[gi] == GeomType.SPHERE:
      h, d, n = ray_sphere_with_normal(
        geom_xpos[world_id, gi],
        geom_size[world_id, gi][0] * geom_size[world_id, gi][0],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.ELLIPSOID:
      h, d, n = ray_ellipsoid_with_normal(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CAPSULE:
      h, d, n = ray_capsule_with_normal(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CYLINDER:
      h, d, n = ray_cylinder_with_normal(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.BOX:
      h, d, n = ray_box_with_normal(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.MESH:
      h, d, n, u, v, f, geom_mesh_id = ray_mesh_with_bvh(
        mesh_bvh_id,
        geom_dataid[gi],
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        dist,
      )

    if h and d < dist:
      dist = d
      normal = n
      geom_id = gi
      bary_u = u
      bary_v = v
      face_idx = f

  return geom_id, dist, normal, bary_u, bary_v, face_idx, geom_mesh_id


@wp.func
def cast_ray_first_hit(
  bvh_id: wp.uint64,
  group_root: int,
  world_id: int,
  bvh_ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
  max_dist: float,
) -> bool:
  """A simpler version of cast_ray_first_hit that only checks for the first hit."""
  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root)
  bounds_nr = int(0)

  while wp.bvh_query_next(query, bounds_nr, max_dist):
    gi_global = bounds_nr
    gi_bvh_local = gi_global - (world_id * bvh_ngeom)
    gi = enabled_geom_ids[gi_bvh_local]

    # TODO: Investigate branch elimination with static loop unrolling
    if geom_type[gi] == GeomType.PLANE:
      d = ray_plane(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.HFIELD:
      h, d, n, u, v, f, geom_hfield_id = ray_mesh_with_bvh(
        hfield_bvh_id,
        geom_dataid[gi],
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        max_dist,
      )
    if geom_type[gi] == GeomType.SPHERE:
      d = ray_sphere(
        geom_xpos[world_id, gi],
        geom_size[world_id, gi][0] * geom_size[world_id, gi][0],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.ELLIPSOID:
      d = ray_ellipsoid(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CAPSULE:
      d = ray_capsule(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CYLINDER:
      d = ray_cylinder(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.BOX:
      d, all = ray_box(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.MESH:
      h, d, n, u, v, f, mesh_id = ray_mesh_with_bvh(
        mesh_bvh_id,
        geom_dataid[gi],
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        ray_origin_world,
        ray_dir_world,
        max_dist,
      )

    if d < max_dist:
      return True

  return False


@wp.func
def compute_lighting(
  use_shadows: bool,
  bvh_id: wp.uint64,
  group_root: int,
  bvh_ngeom: int,
  enabled_geom_ids: wp.array(dtype=int),
  world_id: int,
  light_active: bool,
  light_type: int,
  light_castshadow: bool,
  light_xpos: wp.vec3,
  light_xdir: wp.vec3,
  normal: wp.vec3,
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  mesh_bvh_id: wp.array(dtype=wp.uint64),
  hfield_bvh_id: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  hit_point: wp.vec3,
) -> float:

  light_contribution = float(0.0)

  # TODO: We should probably only be looping over active lights
  # in the first place with a static loop of enabled light idx?
  if not light_active:
    return light_contribution

  L = wp.vec3(0.0, 0.0, 0.0)
  dist_to_light = float(wp.inf)
  attenuation = float(1.0)

  if light_type == 1: # directional light
    L = wp.normalize(-light_xdir)
  else:
    L, dist_to_light = math.normalize_with_norm(light_xpos - hit_point)
    attenuation = 1.0 / (1.0 + 0.02 * dist_to_light * dist_to_light)
    if light_type == 0: # spot light
      spot_dir = wp.normalize(light_xdir)
      cos_theta = wp.dot(-L, spot_dir)
      inner = SPOT_INNER_COS
      outer = SPOT_OUTER_COS
      spot_factor = wp.min(1.0, wp.max(0.0, (cos_theta - outer) / (inner - outer)))
      attenuation = attenuation * spot_factor

  ndotl = wp.max(0.0, wp.dot(normal, L))
  if ndotl == 0.0:
    return light_contribution

  visible = float(1.0)

  if use_shadows and light_castshadow:
    # Nudge the origin slightly along the surface normal to avoid
    # self-intersection when casting shadow rays
    eps = 1.0e-4
    shadow_origin = hit_point + normal * eps
    # Distance-limited shadows: cap by dist_to_light (for non-directional)
    max_t = float(dist_to_light - 1.0e-3)
    if light_type == 1:  # directional light
      max_t = float(1.0e+8)

    shadow_hit = cast_ray_first_hit(
      bvh_id,
      group_root,
      world_id,
      bvh_ngeom,
      enabled_geom_ids,
      geom_type,
      geom_dataid,
      geom_size,
        mesh_bvh_id,
        hfield_bvh_id,
      geom_xpos,
      geom_xmat,
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
    # Model and Options
    ncam: int,
    use_shadows: bool,
    bvh_ngeom: int,

    # Camera
    cam_res: wp.array(dtype=wp.vec2i),
    cam_id_map: wp.array(dtype=int),
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),
    rays: wp.array(dtype=wp.vec3),
    rgb_adr: wp.array(dtype=int),
    depth_adr: wp.array(dtype=int),
    render_rgb: wp.array(dtype=bool),
    render_depth: wp.array(dtype=bool),

    # BVH
    bvh_id: wp.uint64,
    group_root: wp.array(dtype=int),
    flex_bvh_id: wp.uint64,
    flex_group_root: wp.array(dtype=int),

    # Geometry
    enabled_geom_ids: wp.array(dtype=int),
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_matid: wp.array2d(dtype=int),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    mesh_bvh_id: wp.array(dtype=wp.uint64),
    mesh_faceadr: wp.array(dtype=int),
    mesh_face: wp.array(dtype=wp.vec3i),
    mesh_texcoord: wp.array(dtype=wp.vec2),
    mesh_texcoord_offsets: wp.array(dtype=int),
    hfield_bvh_id: wp.array(dtype=wp.uint64),

    # Textures
    mat_texid: wp.array3d(dtype=int),
    mat_texrepeat: wp.array2d(dtype=wp.vec2),
    mat_rgba: wp.array2d(dtype=wp.vec4),
    flex_rgba: wp.array(dtype=wp.vec4),
    flex_matid: wp.array(dtype=int),
    tex_adr: wp.array(dtype=int),
    tex_data: wp.array(dtype=wp.uint32),
    tex_height: wp.array(dtype=int),
    tex_width: wp.array(dtype=int),

    # Lights
    light_active: wp.array2d(dtype=bool),
    light_type: wp.array2d(dtype=int),
    light_castshadow: wp.array2d(dtype=bool),
    light_xpos: wp.array2d(dtype=wp.vec3),
    light_xdir: wp.array2d(dtype=wp.vec3),

    # Data
    geom_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),

    # Output
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

    ray_dir_local_cam = rays[ray_idx]
    ray_dir_world = cam_xmat[world_idx, mujoco_cam_id] @ ray_dir_local_cam
    ray_origin_world = cam_xpos[world_idx, mujoco_cam_id]

    geom_id, dist, normal, u, v, f, mesh_id = cast_ray(
      bvh_id,
      group_root[world_idx],
      world_idx,
      bvh_ngeom,
      enabled_geom_ids,
      geom_type,
      geom_dataid,
      geom_size,
        mesh_bvh_id,
        hfield_bvh_id,
      geom_xpos,
      geom_xmat,
      ray_origin_world,
      ray_dir_world,
    )

    if wp.static(m.nflex > 0):
      h, d, n, u, v, f = ray_flex_with_bvh(
        flex_bvh_id,
        flex_group_root[world_idx],
        ray_origin_world,
        ray_dir_world,
        dist,
      )
      if h and d < dist:
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
              world_idx,
              geom_id,
              geom_type,
              mat_id,
              tex_id,
              mat_texrepeat[world_idx, mat_id],
              tex_adr[tex_id],
              tex_data,
              tex_height[tex_id],
              tex_width[tex_id],
              geom_xpos[world_idx, geom_id],
              geom_xmat[world_idx, geom_id],
              mesh_faceadr,
              mesh_face,
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
        use_shadows,
        bvh_id,
        group_root[world_idx],
        bvh_ngeom,
        enabled_geom_ids,
        world_idx,
        light_active[world_idx, l],
        light_type[world_idx, l],
        light_castshadow[world_idx, l],
        light_xpos[world_idx, l],
        light_xdir[world_idx, l],
        normal,
        geom_type,
        geom_dataid,
        geom_size,
        mesh_bvh_id,
        hfield_bvh_id,
        geom_xpos,
        geom_xmat,
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
    dim=(d.nworld, rc.ray.shape[0]),
    inputs=[
      # Model and Options
      rc.ncam,
      rc.use_shadows,
      rc.bvh_ngeom,

      # Camera
      rc.cam_resolutions,
      rc.cam_id_map,
      d.cam_xpos,
      d.cam_xmat,
      rc.ray,
      rc.rgb_adr,
      rc.depth_adr,
      rc.render_rgb,
      rc.render_depth,

      # BVH
      rc.bvh_id,
      rc.group_root,
      rc.flex_bvh_id,
      rc.flex_group_root,

      # Geometry
      rc.enabled_geom_ids,
      m.geom_type,
      m.geom_dataid,
      m.geom_matid,
      m.geom_size,
      m.geom_rgba,
      rc.mesh_bvh_id,
      m.mesh_faceadr,
      m.mesh_face,
      rc.mesh_texcoord,
      rc.mesh_texcoord_offsets,
      rc.hfield_bvh_id,

      # Textures
      m.mat_texid,
      m.mat_texrepeat,
      m.mat_rgba,
      rc.flex_rgba,
      rc.flex_matid,
      rc.tex_adr,
      rc.tex_data,
      rc.tex_height,
      rc.tex_width,

      # Lights
      m.light_active,
      m.light_type,
      m.light_castshadow,
      d.light_xpos,
      d.light_xdir,

      # Data
      d.geom_xpos,
      d.geom_xmat,
    ],
    outputs=[
      rc.rgb_data,
      rc.depth_data,
    ],
    block_dim=THREADS_PER_TILE,
  )
