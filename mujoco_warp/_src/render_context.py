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

import dataclasses
from typing import Optional, Union

import mujoco
import numpy as np
import warp as wp

from . import bvh
from .types import Data
from .types import GeomType
from .types import Model
from .types import ProjectionType

wp.set_module_options({"enable_backward": False})


def _camera_frustum_bounds(
  cam_projection: int,
  cam_fovy: float,
  cam_sensorsize: wp.vec2,
  cam_intrinsic: wp.vec4,
  img_w: int,
  img_h: int,
  znear: float,
) -> tuple[float, float, float, float, bool]:
  """Replicate MuJoCo's frustum computation to derive near-plane bounds."""
  if cam_projection == ProjectionType.ORTHOGRAPHIC:
    half_height = cam_fovy * 0.5
    aspect = img_w / img_h
    half_width = half_height * aspect
    return (-half_width, half_width, half_height, -half_height, True)

  sensorsize = cam_sensorsize
  has_intrinsics = sensorsize[1] != 0.0
  if has_intrinsics:
    fx, fy, cx, cy = cam_intrinsic
    sensor_w, sensor_h = sensorsize

    # Clip sensor size to match desired aspect ratio
    target_aspect = img_w / img_h
    sensor_aspect = sensor_w / sensor_h
    if target_aspect > sensor_aspect:
      sensor_h = sensor_w / target_aspect
    elif target_aspect < sensor_aspect:
      sensor_w = sensor_h * target_aspect

    left = -znear / fx * (sensor_w * 0.5 - cx)
    right = znear / fx * (sensor_w * 0.5 + cx)
    top = znear / fy * (sensor_h * 0.5 - cy)
    bottom = -znear / fy * (sensor_h * 0.5 + cy)
    return (float(left), float(right), float(top), float(bottom), False)

  fovy_rad = np.deg2rad(cam_fovy)
  half_height = znear * np.tan(0.5 * fovy_rad)
  aspect = img_w / img_h
  half_width = half_height * aspect
  return (-half_width, half_width, half_height, -half_height, False)


@dataclasses.dataclass
class RenderContext:
  ncam: int
  cam_res: wp.array(dtype=wp.vec2i)
  cam_id_map: wp.array(dtype=int)
  use_textures: bool
  use_shadows: bool
  geom_count: int
  bvh_ngeom: int
  enabled_geom_ids: wp.array(dtype=int)
  mesh_bvh_id: wp.array(dtype=wp.uint64)
  mesh_bounds_size: wp.array(dtype=wp.vec3)
  mesh_texcoord: wp.array(dtype=wp.vec2)
  mesh_texcoord_offsets: wp.array(dtype=int)
  mesh_facetexcoord: wp.array(dtype=wp.vec3i)
  textures: wp.array(dtype=wp.Texture2D)
  flex_rgba: wp.array(dtype=wp.vec4)
  flex_matid: wp.array(dtype=int)
  hfield_bvh_id: wp.array(dtype=wp.uint64)
  hfield_bounds_size: wp.array(dtype=wp.vec3)
  flex_bvh_id: wp.uint64
  flex_face_point: wp.array(dtype=wp.vec3)
  flex_elem: wp.array(dtype=int)
  flex_faceadr: wp.array(dtype=int)
  flex_nface: int
  flex_group: wp.array(dtype=int)
  flex_group_root: wp.array(dtype=int)
  bvh_id: wp.uint64
  lower: wp.array(dtype=wp.vec3)
  upper: wp.array(dtype=wp.vec3)
  group: wp.array(dtype=int)
  group_root: wp.array(dtype=int)
  ray: wp.array(dtype=wp.vec3)
  rgb_data: wp.array2d(dtype=wp.uint32)
  depth_data: wp.array2d(dtype=wp.float32)
  rgb_adr: wp.array(dtype=int)
  depth_adr: wp.array(dtype=int)

  def __init__(
    self,
    mjm: mujoco.MjModel,
    m: Model,
    d: Data,
    cam_res: Optional[Union[list[tuple[int, int]] | tuple[int, int]]] = None,
    render_rgb: Optional[Union[list[bool] | bool]] = None,
    render_depth: Optional[Union[list[bool] | bool]] = None,
    use_textures: bool = True,
    use_shadows: bool = False,
    enabled_geom_groups: list[int] = [0, 1, 2],
    cam_active: Optional[list[bool]] = None,
    flex_render_smooth: bool = True,
  ):
    # Mesh BVHs
    nmesh = mjm.nmesh
    geom_enabled_idx = [i for i in range(mjm.ngeom) if mjm.geom_group[i] in enabled_geom_groups]
    used_mesh_id = set(
      int(mjm.geom_dataid[g]) for g in geom_enabled_idx if mjm.geom_type[g] == GeomType.MESH and int(mjm.geom_dataid[g]) >= 0
    )
    self.mesh_registry = {}
    mesh_bvh_id = [wp.uint64(0) for _ in range(nmesh)]
    mesh_bounds_size = [wp.vec3(0.0, 0.0, 0.0) for _ in range(nmesh)]

    for i in range(nmesh):
      if i not in used_mesh_id:
        continue
      mesh, half = bvh.build_mesh_bvh(mjm, i)
      self.mesh_registry[mesh.id] = mesh
      mesh_bvh_id[i] = mesh.id
      mesh_bounds_size[i] = half

    self.mesh_bvh_id = wp.array(mesh_bvh_id, dtype=wp.uint64)
    self.mesh_bounds_size = wp.array(mesh_bounds_size, dtype=wp.vec3)

    # HField BVHs
    nhfield = mjm.nhfield
    used_hfield_id = set(
      int(mjm.geom_dataid[g]) for g in geom_enabled_idx if mjm.geom_type[g] == GeomType.HFIELD and int(mjm.geom_dataid[g]) >= 0
    )
    self.hfield_registry = {}
    hfield_bvh_id = [wp.uint64(0) for _ in range(nhfield)]
    hfield_bounds_size = [wp.vec3(0.0, 0.0, 0.0) for _ in range(nhfield)]

    for hid in range(nhfield):
      if hid not in used_hfield_id:
        continue
      hmesh, hhalf = bvh.build_hfield_bvh(mjm, hid)
      self.hfield_registry[hmesh.id] = hmesh
      hfield_bvh_id[hid] = hmesh.id
      hfield_bounds_size[hid] = hhalf

    self.hfield_bvh_id = wp.array(hfield_bvh_id, dtype=wp.uint64)
    self.hfield_bounds_size = wp.array(hfield_bounds_size, dtype=wp.vec3)

    # Flex BVHs
    self.flex_registry = {}
    self.flex_bvh_id = wp.uint64(0)
    self.flex_group_root = wp.zeros(d.nworld, dtype=int)
    if mjm.nflex > 0:
      (
        fmesh,
        face_point,
        flex_groups,
        flex_group_roots,
        flex_shell,
        flex_faceadr,
        flex_nface,
      ) = bvh.build_flex_bvh(mjm, m, d)

      self.flex_registry[fmesh.id] = fmesh
      self.flex_bvh_id = fmesh.id
      self.flex_face_point = face_point
      self.flex_group = flex_groups
      self.flex_group_root = flex_group_roots
      self.flex_dim = mjm.flex_dim
      self.flex_elemnum = mjm.flex_elemnum
      self.flex_elemadr = mjm.flex_elemadr
      self.flex_elemdataadr = mjm.flex_elemdataadr
      self.flex_shell = flex_shell
      self.flex_shellnum = mjm.flex_shellnum
      self.flex_shelldataadr = mjm.flex_shelldataadr
      self.flex_vertadr = mjm.flex_vertadr
      self.flex_faceadr = flex_faceadr
      self.flex_nface = flex_nface
      self.flex_radius = mjm.flex_radius
      self.flex_render_smooth = flex_render_smooth

    textures = []
    for i in range(mjm.ntex):
      textures.append(_create_warp_texture(mjm, i))
    self._textures_registry = textures
    self.textures = wp.array(textures, dtype=wp.Texture2D)

    # Filter active cameras based on cam_active parameter.
    if cam_active is not None:
      assert len(cam_active) == mjm.ncam, f"cam_active must have length {mjm.ncam} (got {len(cam_active)})"
      active_cam_indices = np.nonzero(cam_active)[0]
    else:
      active_cam_indices = list(range(mjm.ncam))

    ncam = len(active_cam_indices)

    # If a global camera resolution is provided, use it for all cameras
    # otherwise check the xml for camera resolutions
    if cam_res is not None:
      if isinstance(cam_res, tuple):
        cam_res = [cam_res] * ncam
      assert len(cam_res) == ncam, (
        f"Camera resolutions must be provided for all active cameras (got {len(cam_res)}, expected {ncam})"
      )
      active_cam_res = cam_res
    else:
      # Extract resolutions only for active cameras
      active_cam_res = mjm.cam_resolution[active_cam_indices]

    self.cam_res = wp.array(active_cam_res, dtype=wp.vec2i)

    if render_rgb and isinstance(render_rgb, bool):
      render_rgb = [render_rgb] * ncam
    elif render_rgb is None:
      render_rgb = [mjm.cam_output[i] & mujoco.mjtCamOutBit.mjCAMOUT_RGB for i in active_cam_indices]

    if render_depth and isinstance(render_depth, bool):
      render_depth = [render_depth] * ncam
    elif render_depth is None:
      render_depth = [mjm.cam_output[i] & mujoco.mjtCamOutBit.mjCAMOUT_DEPTH for i in active_cam_indices]

    assert len(render_rgb) == ncam and len(render_depth) == ncam, (
      f"Render RGB and depth must be provided for all active cameras (got {len(render_rgb)}, {len(render_depth)}, expected {ncam})"
    )

    rgb_adr = -1 * np.ones(ncam, dtype=int)
    depth_adr = -1 * np.ones(ncam, dtype=int)
    rgb_size = np.zeros(ncam, dtype=int)
    depth_size = np.zeros(ncam, dtype=int)
    cam_res = self.cam_res.numpy()
    ri = 0
    di = 0
    total = 0

    for idx in range(ncam):
      if render_rgb[idx]:
        rgb_adr[idx] = ri
        ri += cam_res[idx][0] * cam_res[idx][1]
        rgb_size[idx] = cam_res[idx][0] * cam_res[idx][1]
      if render_depth[idx]:
        depth_adr[idx] = di
        di += cam_res[idx][0] * cam_res[idx][1]
        depth_size[idx] = cam_res[idx][0] * cam_res[idx][1]

      total += cam_res[idx][0] * cam_res[idx][1]

    self.rgb_adr = wp.array(rgb_adr, dtype=int)
    self.depth_adr = wp.array(depth_adr, dtype=int)
    self.rgb_size = wp.array(rgb_size, dtype=int)
    self.depth_size = wp.array(depth_size, dtype=int)
    self.rgb_data = wp.zeros((d.nworld, ri), dtype=wp.uint32)
    self.depth_data = wp.zeros((d.nworld, di), dtype=wp.float32)
    self.render_rgb = wp.array(render_rgb, dtype=bool)
    self.render_depth = wp.array(render_depth, dtype=bool)
    self.znear = mjm.vis.map.znear * mjm.stat.extent
    self.total_rays = int(total)

    # if cam_fovy or cam_intrinsic is batched, we can skip precalculating the rays
    # since we will need to compute the world specific ray in the render kernel
    if m.cam_fovy.shape[0] > 1 or m.cam_intrinsic.shape[0] > 1:
      self.ray = None
    else:
      self.ray = wp.zeros(int(total), dtype=wp.vec3)

      offset = 0
      for idx, cam_id in enumerate(active_cam_indices):
        img_w = cam_res[idx][0]
        img_h = cam_res[idx][1]
        left, right, top, bottom, is_ortho = _camera_frustum_bounds(
          m.cam_projection.numpy()[cam_id].item(),
          m.cam_fovy.numpy()[0, cam_id].item(),
          wp.vec2(m.cam_sensorsize.numpy()[cam_id]),
          wp.vec4(m.cam_intrinsic.numpy()[0, cam_id]),
          img_w,
          img_h,
          self.znear,
        )
        wp.launch(
          kernel=build_primary_rays,
          dim=int(img_w * img_h),
          inputs=[
            offset,
            img_w,
            img_h,
            left,
            right,
            top,
            bottom,
            self.znear,
            int(is_ortho),
          ],
          outputs=[self.ray],
        )
        offset += img_w * img_h

    self.ncam = ncam
    self.cam_id_map = wp.array(active_cam_indices, dtype=int)
    self.use_textures = use_textures
    self.use_shadows = use_shadows
    self.mesh_texcoord = wp.array(mjm.mesh_texcoord, dtype=wp.vec2)
    self.mesh_texcoord_offsets = wp.array(mjm.mesh_texcoordadr, dtype=int)
    self.mesh_facetexcoord = wp.array(mjm.mesh_facetexcoord, dtype=wp.vec3i)
    self.flex_rgba = wp.array(mjm.flex_rgba, dtype=wp.vec4)
    self.flex_matid = wp.array(mjm.flex_matid, dtype=int)
    self.bvh_ngeom = len(geom_enabled_idx)
    self.enabled_geom_ids = wp.array(geom_enabled_idx, dtype=int)
    self.lower = wp.zeros(d.nworld * self.bvh_ngeom, dtype=wp.vec3)
    self.upper = wp.zeros(d.nworld * self.bvh_ngeom, dtype=wp.vec3)
    self.group = wp.zeros(d.nworld * self.bvh_ngeom, dtype=int)
    self.group_root = wp.zeros(d.nworld, dtype=int)
    self.bvh = None
    self.bvh_id = None
    bvh.build_scene_bvh(m, d, self)


@wp.kernel
def build_primary_rays(
  # In:
  offset: int,
  img_w: int,
  img_h: int,
  left: float,
  right: float,
  top: float,
  bottom: float,
  znear: float,
  orthographic: int,
  # Out:
  ray_out: wp.array(dtype=wp.vec3),
):
  tid = wp.tid()
  total = img_w * img_h
  if tid >= total:
    return

  if orthographic:
    ray_out[offset + tid] = wp.vec3(0.0, 0.0, -1.0)
    return

  px = tid % img_w
  py = tid // img_w
  u = (float(px) + 0.5) / float(img_w)
  v = (float(py) + 0.5) / float(img_h)
  x = left + (right - left) * u
  y = top + (bottom - top) * v
  ray_out[offset + tid] = wp.normalize(wp.vec3(x, y, -znear))


@wp.kernel
def _convert_texture_data(
  # In:
  width: int,
  adr: int,
  nc: int,
  tex_data_in: wp.array(dtype=wp.uint8),
  # Out:
  tex_data_out: wp.array3d(dtype=float),
):
  """Convert uint8 texture data to vec4 format for efficient sampling."""
  x, y = wp.tid()
  offset = adr + (y * width + x) * nc
  r = tex_data_in[offset + 0] if nc > 0 else wp.uint8(0)
  g = tex_data_in[offset + 1] if nc > 1 else wp.uint8(0)
  b = tex_data_in[offset + 2] if nc > 2 else wp.uint8(0)
  a = wp.uint8(255)

  tex_data_out[y, x, 0] = float(r) * wp.static(1.0 / 255.0)
  tex_data_out[y, x, 1] = float(g) * wp.static(1.0 / 255.0)
  tex_data_out[y, x, 2] = float(b) * wp.static(1.0 / 255.0)
  tex_data_out[y, x, 3] = float(a) * wp.static(1.0 / 255.0)


def _create_warp_texture(mjm: mujoco.MjModel, tex_id: int) -> wp.array:
  """Create a Warp texture from a MuJoCo model texture data."""
  tex_adr = mjm.tex_adr[tex_id]
  tex_width = mjm.tex_width[tex_id]
  tex_height = mjm.tex_height[tex_id]
  nchannel = mjm.tex_nchannel[tex_id]
  tex_data = wp.zeros((tex_height, tex_width, 4), dtype=float)

  wp.launch(
    _convert_texture_data,
    dim=(tex_width, tex_height),
    inputs=[tex_width, tex_adr, nchannel, wp.array(mjm.tex_data, dtype=wp.uint8)],
    outputs=[tex_data],
  )
  return wp.Texture2D(tex_data, filter_mode=wp.TextureFilterMode.LINEAR)
