from typing import Tuple

import warp as wp

from . import bvh
from .ray import ray_plane_with_normal
from .ray import ray_sphere_with_normal
from .ray import ray_capsule_with_normal
from .ray import ray_box_with_normal
from .ray import ray_mesh_with_bvh
from .ray import _ray_plane
from .ray import _ray_sphere
from .ray import _ray_capsule
from .ray import _ray_box
from .types import Data
from .types import Model
from .types import GeomType
from .warp_util import event_scope


MAX_NUM_VIEWS_PER_THREAD = 8

BACKGROUND_COLOR = (
  255 << 24 |
  int(0.2 * 255.0) << 16 |
  int(0.1 * 255.0) << 8 |
  int(0.1 * 255.0)
)

@event_scope
def render(m: Model, d: Data):
  bvh.refit_warp_bvh(m, d)
  render_raytrace_megakernel(m, d)


@wp.func
def compute_camera_ray(
  width: int,
  height: int,
  fov_rad: float,
  px: int,
  py: int,
  cam_xpos: wp.vec3,
  cam_xmat:wp.mat33,
):
  aspect_ratio = float(width) / float(height)
  u = (float(px) + 0.5) / float(width) - 0.5
  v = (float(py) + 0.5) / float(height) - 0.5
  h = wp.tan(fov_rad / 2.0)
  ray_dir_cam_space_x = u * 2.0 * h
  ray_dir_cam_space_y = -v * 2.0 * h / aspect_ratio
  ray_dir_cam_space_z = -1.0
  ray_dir_local_cam = wp.normalize(
    wp.vec3(
      ray_dir_cam_space_x,
      ray_dir_cam_space_y,
      ray_dir_cam_space_z,
    )
  )
  ray_dir_world = cam_xmat @ ray_dir_local_cam
  ray_origin = cam_xpos
  return ray_dir_world, ray_origin


@wp.func
def pack_rgba_to_uint32(r: wp.uint8, g: wp.uint8, b: wp.uint8, a: wp.uint8) -> wp.uint32:
  """Pack RGBA values into a single uint32 for efficient memory access."""
  return (wp.uint32(a) << wp.uint32(24)) | (wp.uint32(r) << wp.uint32(16)) | (wp.uint32(g) << wp.uint32(8)) | wp.uint32(b)


@wp.func
def pack_rgba_to_uint32(r: wp.float32, g: wp.float32, b: wp.float32, a: wp.float32) -> wp.uint32:
  """Pack RGBA values into a single uint32 for efficient memory access."""
  return (wp.uint32(a) << wp.uint32(24)) | (wp.uint32(b) << wp.uint32(16)) | (wp.uint32(g) << wp.uint32(8)) | wp.uint32(r)


@wp.func
def sample_texture_2d(
  uv: wp.vec2,
  width: int,
  height: int,
  tex_adr: int,
  tex_data: wp.array(dtype=wp.uint32),
) -> wp.vec3:
  ix = wp.min(width - 1, wp.int32(uv[0] * wp.float32(width)))
  iy = wp.min(height - 1, wp.int32(uv[1] * wp.float32(height)))
  linear_idx = tex_adr + (iy * width + ix)
  packed_rgba = tex_data[linear_idx]
  r = wp.float32((packed_rgba >> wp.uint32(16)) & wp.uint32(0xFF)) / 255.0
  g = wp.float32((packed_rgba >> wp.uint32(8)) & wp.uint32(0xFF)) / 255.0
  b = wp.float32(packed_rgba & wp.uint32(0xFF)) / 255.0
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
  bary_u: wp.float32,
  bary_v: wp.float32,
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
  u: wp.float32,
  v: wp.float32,
  f: wp.int32,
  mesh_id: wp.int32,
) -> wp.vec3:
  tex_color = wp.vec3(1.0, 1.0, 1.0)

  if geom_matid == -1 or mat_texid == -1:
    return tex_color

  if geom_type[geom_id] == int(GeomType.PLANE.value):
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

  if geom_type[geom_id] == int(GeomType.MESH.value):
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
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
) -> Tuple[wp.int32, wp.float32, wp.vec3, wp.float32, wp.float32, wp.int32, wp.int32]:
  dist = wp.float32(wp.inf)
  normal = wp.vec3(0.0, 0.0, 0.0)
  geom_id = wp.int32(-1)
  bary_u = wp.float32(0.0)
  bary_v = wp.float32(0.0)
  face_idx = wp.int32(-1)
  geom_mesh_id = wp.int32(-1)

  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root, dist)
  bounds_nr = wp.int32(0)

  while wp.bvh_query_next(query, bounds_nr, dist):
    gi_global = bounds_nr
    gi_bvh_local = gi_global - (world_id * bvh_ngeom)
    gi = enabled_geom_ids[gi_bvh_local]

    if geom_type[gi] == GeomType.PLANE:
      h, d, n = ray_plane_with_normal(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.SPHERE:
      h, d, n = ray_sphere_with_normal(
        geom_xpos[world_id, gi],
        geom_size[world_id, gi][0] * geom_size[world_id, gi][0],
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
        mesh_bvh_ids,
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
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  ray_origin_world: wp.vec3,
  ray_dir_world: wp.vec3,
  max_dist: wp.float32,
) -> bool:
  """ A simpler version of cast_ray_first_hit that only checks for the first hit."""
  query = wp.bvh_query_ray(bvh_id, ray_origin_world, ray_dir_world, group_root, max_dist)
  bounds_nr = wp.int32(0)

  while wp.bvh_query_next(query, bounds_nr, max_dist):
    gi_global = bounds_nr
    gi_bvh_local = gi_global - (world_id * bvh_ngeom)
    gi = enabled_geom_ids[gi_bvh_local]

    if geom_type[gi] == GeomType.PLANE:
      d = _ray_plane(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.SPHERE:
      d = _ray_sphere(
        geom_xpos[world_id, gi],
        geom_size[world_id, gi][0] * geom_size[world_id, gi][0],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.CAPSULE:
      d = _ray_capsule(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.BOX:
      d, all = _ray_box(
        geom_xpos[world_id, gi],
        geom_xmat[world_id, gi],
        geom_size[world_id, gi],
        ray_origin_world,
        ray_dir_world,
      )
    if geom_type[gi] == GeomType.MESH:
      h, d, n, u, v, f, mesh_id = ray_mesh_with_bvh(
        mesh_bvh_ids,
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
  mesh_bvh_ids: wp.array(dtype=wp.uint64),
  geom_xpos: wp.array2d(dtype=wp.vec3),
  geom_xmat: wp.array2d(dtype=wp.mat33),
  hit_point: wp.vec3,
) -> wp.float32:

  light_contribution = wp.float32(0.0)

  if not light_active:
    return light_contribution

  L = wp.vec3(0.0, 0.0, 0.0)
  dist_to_light = wp.float32(wp.inf)
  attenuation = wp.float32(1.0)

  if light_type == 1: # directional light
    L = wp.normalize(-light_xdir)
  else:
    to_light = light_xpos - hit_point
    dist_to_light = wp.length(to_light)
    L = wp.normalize(to_light)
    attenuation = 1.0 / (1.0 + 0.02 * dist_to_light * dist_to_light)
    if light_type == 0: # spot light
      spot_dir = wp.normalize(light_xdir)
      cos_theta = wp.dot(-L, spot_dir)
      inner = 0.95
      outer = 0.85
      spot_factor = wp.min(1.0, wp.max(0.0, (cos_theta - outer) / (inner - outer)))
      attenuation = attenuation * spot_factor

  ndotl = wp.max(0.0, wp.dot(normal, L))
  if ndotl == 0.0:
    return light_contribution

  visible = wp.float32(1.0)
  shadow_min_visibility = wp.float32(0.3) # reduce shadow darkness (0: full black, 1: no shadow)

  if use_shadows and light_castshadow:
    # Nudge the origin slightly along the surface normal to avoid
    # self-intersection when casting shadow rays
    eps = 1.0e-4
    shadow_origin = hit_point + normal * eps
    # Distance-limited shadows: cap by dist_to_light (for non-directional)
    max_t = wp.float32(dist_to_light - 1.0e-3)
    if light_type == 1:  # directional light
      max_t = wp.float32(1.0e+8)

    shadow_hit = cast_ray_first_hit(
      bvh_id,
      group_root,
      world_id,
      bvh_ngeom,
      enabled_geom_ids,
      geom_type,
      geom_dataid,
      geom_size,
      mesh_bvh_ids,
      geom_xpos,
      geom_xmat,
      shadow_origin,
      L,
      max_t,
    )

    if shadow_hit:
      visible = shadow_min_visibility

  return ndotl * attenuation * visible


@event_scope
def render_raytrace_megakernel(m: Model, d: Data):
  total_views = d.nworld * m.ncam
  total_pixels = m.render_opt.width * m.render_opt.height
  num_view_groups = (total_views + MAX_NUM_VIEWS_PER_THREAD - 1) // MAX_NUM_VIEWS_PER_THREAD
  if num_view_groups == 0:
    return

  if m.render_opt.render_rgb:
    d.pixels.fill_(wp.uint32(BACKGROUND_COLOR))

  if m.render_opt.render_depth:
    d.depth.fill_(wp.float32(0.0))

  static_nlight = wp.static(m.nlight)

  @wp.kernel
  def _raytrace_megakernel(
    # Model and Options
    nworld: int,
    ncam: int,
    nlight: int,
    ngeom: int,
    bvh_ngeom: int,
    img_width: int,
    img_height: int,
    use_textures: bool,
    use_shadows: bool,
    render_rgb: bool,
    render_depth: bool,

    # Camera
    fov_rad: float,
    cam_xpos: wp.array2d(dtype=wp.vec3),
    cam_xmat: wp.array2d(dtype=wp.mat33),

    # BVH
    bvh_id: wp.uint64,
    group_roots: wp.array(dtype=wp.int32),

    # Geometry
    enabled_geom_ids: wp.array(dtype=int),
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_matid: wp.array2d(dtype=int),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    mesh_bvh_ids: wp.array(dtype=wp.uint64),
    mesh_faceadr: wp.array(dtype=int),
    mesh_face: wp.array(dtype=wp.vec3i),
    mesh_texcoord: wp.array(dtype=wp.vec2),
    mesh_texcoord_offsets: wp.array(dtype=int),

    # Textures
    mat_texid: wp.array3d(dtype=int),
    mat_texrepeat: wp.array2d(dtype=wp.vec2),
    mat_rgba: wp.array2d(dtype=wp.vec4),
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
    out_pixels: wp.array3d(dtype=wp.uint32),
    out_depth: wp.array3d(dtype=wp.float32),
  ):
    tid = wp.tid()

    if tid >= nworld * ncam * img_width * img_height:
      return

    total_views = nworld * ncam
    pixels_per_image = img_width * img_height
    num_view_groups = (total_views + MAX_NUM_VIEWS_PER_THREAD - 1) // MAX_NUM_VIEWS_PER_THREAD

    group_idx = tid // pixels_per_image
    pixel_idx = tid % pixels_per_image

    if group_idx >= num_view_groups:
      return

    px = pixel_idx % img_width
    py = pixel_idx // img_width
    base_view = group_idx * MAX_NUM_VIEWS_PER_THREAD

    for i in range(MAX_NUM_VIEWS_PER_THREAD):
      view = base_view + i
      if view >= total_views:
        break

      world_idx = view // ncam
      cam_idx = view % ncam

      ray_dir_world, ray_origin_world = compute_camera_ray(
        img_width,
        img_height,
        fov_rad,
        px,
        py,
        cam_xpos[world_idx, cam_idx],
        cam_xmat[world_idx, cam_idx],
      )

      geom_id, dist, normal, u, v, f, mesh_id = cast_ray(
        bvh_id,
        group_roots[world_idx],
        world_idx,
        bvh_ngeom,
        enabled_geom_ids,
        geom_type,
        geom_dataid,
        geom_size,
        mesh_bvh_ids,
        geom_xpos,
        geom_xmat,
        ray_origin_world,
        ray_dir_world,
      )

      # Early Out
      if geom_id == -1:
        continue

      # Shade the pixel
      hit_point = ray_origin_world + ray_dir_world * dist

      if geom_matid[world_idx, geom_id] == -1:
        color = geom_rgba[world_idx, geom_id]
      else:
        color = mat_rgba[world_idx, geom_matid[world_idx, geom_id]]

      base_color = wp.vec3(color[0], color[1], color[2])
      hit_color = base_color

      if use_textures:
        mat_id = geom_matid[world_idx, geom_id]
        tex_id = mat_texid[world_idx, mat_id, 1]

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
        base_color = wp.vec3(
          base_color[0] * tex_color[0],
          base_color[1] * tex_color[1],
          base_color[2] * tex_color[2],
        )

      ambient = 0.15
      result = base_color * ambient

      # Apply lighting and shadows
      for l in range(wp.static(static_nlight)):
        light_contribution = compute_lighting(
          use_shadows,
          bvh_id,
          group_roots[world_idx],
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
          mesh_bvh_ids,
          geom_xpos,
          geom_xmat,
          hit_point,
        )
        result = result + base_color * light_contribution

      hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))
      hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))

      if render_rgb:
        out_pixels[world_idx, cam_idx, pixel_idx] = pack_rgba_to_uint32(
          hit_color[0] * 255.0,
          hit_color[1] * 255.0,
          hit_color[2] * 255.0,
          1.0,
        )
      if render_depth:
        out_depth[world_idx, cam_idx, pixel_idx] = dist

  wp.launch(
    kernel=_raytrace_megakernel,
    dim=(num_view_groups * total_pixels),
    inputs=[
      # Model and Options
      d.nworld,
      m.ncam,
      m.nlight,
      m.ngeom,
      m.bvh_ngeom,
      m.render_opt.width,
      m.render_opt.height,
      m.render_opt.use_textures,
      m.render_opt.use_shadows,
      m.render_opt.render_rgb,
      m.render_opt.render_depth,

      # Camera
      m.render_opt.fov_rad,
      d.cam_xpos,
      d.cam_xmat,

      # BVH
      d.bvh_id,
      d.group_roots,

      # Geometry
      m.enabled_geom_ids,
      m.geom_type,
      m.geom_dataid,
      m.geom_matid,
      m.geom_size,
      m.geom_rgba,
      m.mesh_bvh_ids,
      m.mesh_faceadr,
      m.mesh_face,
      m.mesh_texcoord,
      m.mesh_texcoord_offsets,

      # Textures
      m.mat_texid,
      m.mat_texrepeat,
      m.mat_rgba,
      m.tex_adr,
      m.tex_data,
      m.tex_height,
      m.tex_width,

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
      d.pixels,
      d.depth,
    ],
  )


