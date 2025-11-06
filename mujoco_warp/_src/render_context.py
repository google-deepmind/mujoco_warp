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
from typing import Optional

import mujoco
import warp as wp

from . import bvh
from .types import Data
from .types import GeomType
from .types import Model

wp.set_module_options({"enable_backward": False})


@dataclasses.dataclass
class RenderContext:
  ncam: int
  height: int
  width: int
  render_rgb: bool
  render_depth: bool
  use_textures: bool
  use_shadows: bool
  fov_rad: float
  bvh_ngeom: int
  enabled_geom_ids: wp.array(dtype=int)
  mesh_bvh_ids: wp.array(dtype=wp.uint64)
  mesh_bounds_size: wp.array(dtype=wp.vec3)
  mesh_texcoord: wp.array(dtype=wp.vec2)
  mesh_texcoord_offsets: wp.array(dtype=int)
  mesh_texcoord_num: wp.array(dtype=int)
  tex_adr: wp.array(dtype=int)
  tex_data: wp.array(dtype=wp.uint32)
  tex_height: wp.array(dtype=int)
  tex_width: wp.array(dtype=int)
  bvh_id: wp.uint64
  mesh_bvh_ids: wp.array(dtype=wp.uint64)
  lowers: wp.array(dtype=wp.vec3)
  uppers: wp.array(dtype=wp.vec3)
  groups: wp.array(dtype=int)
  group_roots: wp.array(dtype=int)
  pixels: wp.array3d(dtype=wp.uint32)
  depth: wp.array3d(dtype=wp.float32)

  def __init__(
    self,
    mjm: mujoco.MjModel,
    m: Model,
    d: Data,
    width,
    height,
    use_textures,
    use_shadows,
    render_rgb,
    render_depth,
    enabled_geom_groups = [0, 1, 2],
  ):

    nmesh = mjm.nmesh
    geom_enabled_idx = [i for i in range(mjm.ngeom) if mjm.geom_group[i] in enabled_geom_groups]

    used_mesh_ids = set(
      int(mjm.geom_dataid[g])
      for g in geom_enabled_idx
      if mjm.geom_type[g] == GeomType.MESH and int(mjm.geom_dataid[g]) >= 0
    )

    self.mesh_registry = {}
    mesh_bvh_ids = [wp.uint64(0) for _ in range(nmesh)]
    mesh_bounds_size = [wp.vec3(0.0, 0.0, 0.0) for _ in range(nmesh)]

    for i in range(nmesh):
      if i not in used_mesh_ids:
        continue

      v_start = mjm.mesh_vertadr[i]
      v_end = v_start + mjm.mesh_vertnum[i]
      points = mjm.mesh_vert[v_start:v_end]

      f_start = mjm.mesh_faceadr[i]
      f_end = mjm.mesh_face.shape[0] if (i + 1) >= nmesh else mjm.mesh_faceadr[i + 1]
      indices = mjm.mesh_face[f_start:f_end]
      indices = indices.flatten()

      mesh = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3),
        indices=wp.array(indices, dtype=wp.int32),
        bvh_constructor="sah",
      )
      self.mesh_registry[mesh.id] = mesh
      mesh_bvh_ids[i] = mesh.id

      pmin = points.min(axis=0)
      pmax = points.max(axis=0)
      half = 0.5 * (pmax - pmin)
      mesh_bounds_size[i] = half

    tex_data_packed, tex_adr_packed = _create_packed_texture_data(mjm)

    bvh_ngeom = len(geom_enabled_idx)
    self.bvh_ngeom=bvh_ngeom
    self.ncam=mjm.ncam
    self.width=width
    self.height=height
    self.use_textures=use_textures
    self.use_shadows=use_shadows
    self.fov_rad=wp.radians(mjm.cam_fovy[0])
    self.render_rgb=render_rgb
    self.render_depth=render_depth
    self.enabled_geom_ids=wp.array(geom_enabled_idx, dtype=int)
    self.mesh_bvh_ids=wp.array(mesh_bvh_ids, dtype=wp.uint64)
    self.mesh_bounds_size=wp.array(mesh_bounds_size, dtype=wp.vec3)
    self.mesh_texcoord=wp.array(mjm.mesh_texcoord, dtype=wp.vec2)
    self.mesh_texcoord_offsets=wp.array(mjm.mesh_texcoordadr, dtype=int)
    self.mesh_texcoord_num=wp.array(mjm.mesh_texcoordnum, dtype=int)
    self.tex_adr=tex_adr_packed
    self.tex_data=tex_data_packed
    self.tex_height = wp.array(mjm.tex_height, dtype=int)
    self.tex_width = wp.array(mjm.tex_width, dtype=int)
    self.lowers = wp.zeros(d.nworld * bvh_ngeom, dtype=wp.vec3)
    self.uppers = wp.zeros(d.nworld * bvh_ngeom, dtype=wp.vec3)
    self.groups = wp.zeros(d.nworld * bvh_ngeom, dtype=int)
    self.group_roots = wp.zeros(d.nworld, dtype=int)
    self.pixels = wp.zeros((d.nworld, mjm.ncam, width * height), dtype=wp.uint32)
    self.depth = wp.zeros((d.nworld, mjm.ncam, width * height), dtype=wp.float32)
    self.rays_cam = wp.zeros((width * height), dtype=wp.vec3)
    wp.launch(
      kernel=build_primary_rays,
      dim=width * height,
      inputs=[width, height, self.fov_rad, self.rays_cam],
    )

    self.bvh = None
    self.bvh_id = None
    bvh.build_warp_bvh(m, d, self)


def create_render_context(
  mjm: mujoco.MjModel,
  m: Model,
  d: Data,
  width: int,
  height: int,
  use_textures: Optional[bool] = True,
  use_shadows: Optional[bool] = False,
  render_rgb: Optional[bool] = True,
  render_depth: Optional[bool] = True,
  enabled_geom_groups: Optional[list[int]] = [0, 1, 2],
) -> RenderContext:
  """Creates a render context on device.

    Args:
      mjm (mujoco.MjModel): The model containing kinematic and dynamic information (host).
      m (Model): The model on device.
      d (Data): The data on device.
      width (int): The width to render every camera image.
      height (int): The height to render every camera image.
      use_textures (bool, optional): Whether to use textures. Defaults to True.
      use_shadows (bool, optional): Whether to use shadows. Defaults to False.
      render_rgb (bool, optional): Whether to render RGB images. Defaults to True.
      render_depth (bool, optional): Whether to render depth images. Defaults to True.
      enabled_geom_groups (list[int], optional): The geom groups to render. Defaults to [0, 1, 2].

    Returns:
      RenderContext: The render context containing rendering fields and output arrays (device).
    """

  return RenderContext(
    mjm,
    m,
    d,
    width,
    height,
    use_textures,
    use_shadows,
    render_rgb,
    render_depth,
    enabled_geom_groups,
  )


@wp.kernel
def build_primary_rays(img_w: int, img_h: int, fov_rad: float, rays_cam: wp.array(dtype=wp.vec3)):
  tid = wp.tid()
  total = img_w * img_h
  if tid >= total:
    return
  px = tid % img_w
  py = tid // img_w
  inv_img_w = 1.0 / float(img_w)
  inv_img_h = 1.0 / float(img_h)
  aspect_ratio = float(img_w) * inv_img_h
  u = (float(px) + 0.5) * inv_img_w - 0.5
  v = (float(py) + 0.5) * inv_img_h - 0.5
  h = wp.tan(fov_rad * 0.5)
  dx = u * 2.0 * h
  dy = -v * 2.0 * h / aspect_ratio
  dz = -1.0
  rays_cam[tid] = wp.normalize(wp.vec3(dx, dy, dz))


def _create_packed_texture_data(mjm: mujoco.MjModel) -> tuple[wp.array, wp.array]:
  """Create packed uint32 texture data from uint8 texture data for optimized sampling."""
  if mjm.ntex == 0:
    return wp.array([], dtype=wp.uint32), wp.array([], dtype=int)

  total_size = 0
  for i in range(mjm.ntex):
    total_size += mjm.tex_width[i] * mjm.tex_height[i]

  tex_data_packed = wp.zeros((total_size,), dtype=wp.uint32)
  tex_adr_packed = []

  for i in range(mjm.ntex):
    tex_adr_packed.append(mjm.tex_adr[i] // mjm.tex_nchannel[i])

  nchannel = wp.static(int(mjm.tex_nchannel[0]))

  @wp.kernel
  def convert_texture_to_packed(
    tex_data_uint8: wp.array(dtype=wp.uint8),
    tex_data_packed: wp.array(dtype=wp.uint32),
  ):
    """
    Convert uint8 texture data to packed uint32 format for efficient sampling.
    """
    tid = wp.tid()

    src_idx = tid * nchannel

    r = tex_data_uint8[src_idx + 0] if nchannel > 0 else wp.uint8(0)
    g = tex_data_uint8[src_idx + 1] if nchannel > 1 else wp.uint8(0)
    b = tex_data_uint8[src_idx + 2] if nchannel > 2 else wp.uint8(0)
    a = wp.uint8(255)

    packed = (wp.uint32(a) << wp.uint32(24)) | (wp.uint32(r) << wp.uint32(16)) | (wp.uint32(g) << wp.uint32(8)) | wp.uint32(b)
    tex_data_packed[tid] = packed

  wp.launch(
    convert_texture_to_packed,
    dim=int(total_size),
    inputs=[wp.array(mjm.tex_data, dtype=wp.uint8), tex_data_packed],
  )

  return tex_data_packed, wp.array(tex_adr_packed, dtype=int)
