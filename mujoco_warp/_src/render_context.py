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

wp.set_module_options({"enable_backward": False})


@dataclasses.dataclass
class RenderContext:
  ncam: int
  cam_resolutions: wp.array(dtype=wp.vec2i)
  cam_id_map: wp.array(dtype=int)
  use_textures: bool
  use_shadows: bool
  geom_count: int
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
  hfield_bvh_ids: wp.array(dtype=wp.uint64)
  hfield_bounds_size: wp.array(dtype=wp.vec3)
  flex_bvh_id: wp.uint64
  flex_face_points: wp.array(dtype=wp.vec3)
  flex_shell: wp.array(dtype=int)
  flex_shell_count: int
  flex_elem_count: int
  flex_group: wp.array(dtype=int)
  flex_group_roots: wp.array(dtype=int)
  lowers: wp.array(dtype=wp.vec3)
  uppers: wp.array(dtype=wp.vec3)
  groups: wp.array(dtype=int)
  group_roots: wp.array(dtype=int)
  rays: wp.array(dtype=wp.vec3)
  rgb: wp.array2d(dtype=wp.uint32)
  depth: wp.array2d(dtype=wp.float32)
  rgb_offsets: wp.array(dtype=int)
  depth_offsets: wp.array(dtype=int)

  def __init__(
    self,
    mjm: mujoco.MjModel,
    m: Model,
    d: Data,
    cam_resolutions: Union[list[tuple[int, int]] | tuple[int, int]] = None,
    render_rgb: Union[list[bool] | bool] = True,
    render_depth: Union[list[bool] | bool] = False,
    use_textures: Optional[bool] = True,
    use_shadows: Optional[bool] = False,
    enabled_geom_groups = [0, 1, 2],
    cam_active: Optional[list[bool]] = None,
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

    # Mesh BVHs
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

    # HField BVHs
    nhfield = int(mjm.nhfield)
    used_hfield_ids = set(
      int(mjm.geom_dataid[g])
      for g in geom_enabled_idx
      if mjm.geom_type[g] == GeomType.HFIELD and int(mjm.geom_dataid[g]) >= 0
    )
    self.hfield_registry = {}
    hfield_bvh_ids = [wp.uint64(0) for _ in range(nhfield)]
    hfield_bounds_size = [wp.vec3(0.0, 0.0, 0.0) for _ in range(nhfield)]

    for hid in range(nhfield):
      if hid not in used_hfield_ids:
        continue
      hmesh, hhalf = _make_hfield_mesh(mjm, hid)
      self.hfield_registry[hmesh.id] = hmesh
      hfield_bvh_ids[hid] = hmesh.id
      hfield_bounds_size[hid] = hhalf

    # Flex BVHs
    self.flex_registry = {}
    self.flex_bvh_id = wp.uint64(0)
    self.flex_group_roots = wp.zeros(d.nworld, dtype=int)
    if mjm.nflex > 0:
      fmesh, flex_points, flex_group, flex_group_roots, flex_face_offset, shell_count, elem_count= _make_flex_mesh(mjm, m, d)
      self.flex_registry[fmesh.id] = fmesh
      flex_bvh_id = fmesh.id
      self.flex_bvh_id=flex_bvh_id
      self.flex_face_points=flex_points
      self.flex_shell=wp.array(mjm.flex_shell, dtype=int)
      self.flex_shell_count=shell_count
      self.flex_elem_count=elem_count
      self.flex_group=flex_group
      self.flex_group_roots=flex_group_roots
      self.flex_radius=mjm.flex_radius
      self.flex_face_offset=flex_face_offset

    tex_data_packed, tex_adr_packed = _create_packed_texture_data(mjm)

    # Filter active cameras based on cam_active parameter.
    if cam_active is not None:
      assert len(cam_active) == mjm.ncam, f"cam_active must have length {mjm.ncam} (got {len(cam_active)})"
      active_cam_indices = [i for i in range(mjm.ncam) if cam_active[i]]
    else:
      active_cam_indices = list(range(mjm.ncam))

    n_active_cams = len(active_cam_indices)

    # If a global camera resolution is provided, use it for all cameras
    # otherwise check the xml for camera resolutions
    if cam_resolutions is not None:
      if isinstance(cam_resolutions, tuple):
        cam_resolutions = [cam_resolutions] * n_active_cams
      assert len(cam_resolutions) == n_active_cams, f"Camera resolutions must be provided for all active cameras (got {len(cam_resolutions)}, expected {n_active_cams})"
      active_cam_resolutions = cam_resolutions
    else:
      # Extract resolutions only for active cameras
      active_cam_resolutions = [mjm.cam_resolution[i] for i in active_cam_indices]

    self.cam_resolutions = wp.array(active_cam_resolutions, dtype=wp.vec2i)

    if isinstance(render_rgb, bool):
      render_rgb = [render_rgb] * n_active_cams
    assert len(render_rgb) == n_active_cams, f"Render RGB must be provided for all active cameras (got {len(render_rgb)}, expected {n_active_cams})"

    if isinstance(render_depth, bool):
      render_depth = [render_depth] * n_active_cams
    assert len(render_depth) == n_active_cams, f"Render depth must be provided for all active cameras (got {len(render_depth)}, expected {n_active_cams})"

    rgb_adr = [-1 for _ in range(n_active_cams)]
    depth_adr = [-1 for _ in range(n_active_cams)]
    rgb_size = [0 for _ in range(n_active_cams)]
    depth_size = [0 for _ in range(n_active_cams)]
    cam_resolutions = self.cam_resolutions.numpy()
    ri = 0
    di = 0
    total = 0

    for idx in range(n_active_cams):
      if render_rgb[idx]:
        rgb_adr[idx] = ri
        ri += cam_resolutions[idx][0] * cam_resolutions[idx][1]
        rgb_size[idx] = cam_resolutions[idx][0] * cam_resolutions[idx][1]
      if render_depth[idx]:
        depth_adr[idx] = di
        di += cam_resolutions[idx][0] * cam_resolutions[idx][1]
        depth_size[idx] = cam_resolutions[idx][0] * cam_resolutions[idx][1]

      total += cam_resolutions[idx][0] * cam_resolutions[idx][1]

    self.rgb_adr = wp.array(rgb_adr, dtype=int)
    self.depth_adr = wp.array(depth_adr, dtype=int)
    self.rgb_size = wp.array(rgb_size, dtype=int)
    self.depth_size = wp.array(depth_size, dtype=int)
    self.rgb_data = wp.zeros((d.nworld, ri), dtype=wp.uint32)
    self.depth_data = wp.zeros((d.nworld, di), dtype=wp.float32)
    self.render_rgb=wp.array(render_rgb, dtype=bool)
    self.render_depth=wp.array(render_depth, dtype=bool)
    self.ray_data = wp.zeros(int(total), dtype=wp.vec3)

    offset = 0
    for idx, cam_id in enumerate(active_cam_indices):
      wp.launch(
        kernel=build_primary_rays,
        dim=int(cam_resolutions[idx][0] * cam_resolutions[idx][1]),
        inputs=[offset, cam_resolutions[idx][0], cam_resolutions[idx][1], wp.radians(float(mjm.cam_fovy[cam_id])), self.ray_data],
      )
      offset += cam_resolutions[idx][0] * cam_resolutions[idx][1]

    self.bvh_ngeom=len(geom_enabled_idx)
    self.ncam=n_active_cams
    self.cam_id_map=wp.array(active_cam_indices, dtype=int)
    self.use_textures=use_textures
    self.use_shadows=use_shadows
    self.enabled_geom_ids=wp.array(geom_enabled_idx, dtype=int)
    self.mesh_bvh_ids=wp.array(mesh_bvh_ids, dtype=wp.uint64)
    self.mesh_bounds_size=wp.array(mesh_bounds_size, dtype=wp.vec3)
    self.hfield_bvh_ids=wp.array(hfield_bvh_ids, dtype=wp.uint64)
    self.hfield_bounds_size=wp.array(hfield_bounds_size, dtype=wp.vec3)
    self.mesh_texcoord=wp.array(mjm.mesh_texcoord, dtype=wp.vec2)
    self.mesh_texcoord_offsets=wp.array(mjm.mesh_texcoordadr, dtype=int)
    self.mesh_texcoord_num=wp.array(mjm.mesh_texcoordnum, dtype=int)
    self.tex_adr=tex_adr_packed
    self.tex_data=tex_data_packed
    self.tex_height = wp.array(mjm.tex_height, dtype=int)
    self.tex_width = wp.array(mjm.tex_width, dtype=int)
    self.lowers = wp.zeros(d.nworld * self.bvh_ngeom, dtype=wp.vec3)
    self.uppers = wp.zeros(d.nworld * self.bvh_ngeom, dtype=wp.vec3)
    self.groups = wp.zeros(d.nworld * self.bvh_ngeom, dtype=int)
    self.group_roots = wp.zeros(d.nworld, dtype=int)
    self.bvh = None
    self.bvh_id = None
    bvh.build_warp_bvh(m, d, self)


def create_render_context(
  mjm: mujoco.MjModel,
  m: Model,
  d: Data,
  cam_resolutions: Union[list[tuple[int, int]] | tuple[int, int]],
  render_rgb: Union[list[bool] | bool] = True,
  render_depth: Union[list[bool] | bool] = False,
  use_textures: Optional[bool] = True,
  use_shadows: Optional[bool] = False,
  enabled_geom_groups: list[int] = [0, 1, 2],
  cam_active: Optional[list[bool]] = None,
) -> RenderContext:
  """Creates a render context on device.

    Args:
      mjm: The model containing kinematic and dynamic information on host.
      m: The model on device.
      d: The data on device.
      cam_resolutions: The width and height to render each camera image.
      render_rgb: Whether to render RGB images.
      render_depth: Whether to render depth images.
      use_textures: Whether to use textures.
      use_shadows: Whether to use shadows.
      enabled_geom_groups: The geom groups to render.
      cam_active: List of booleans indicating which cameras to include in rendering.
                  If None, all cameras are included.

    Returns:
      The render context containing rendering fields and output arrays on device.
    """

  return RenderContext(
    mjm,
    m,
    d,
    cam_resolutions,
    render_rgb,
    render_depth,
    use_textures,
    use_shadows,
    enabled_geom_groups,
    cam_active,
  )


@wp.kernel
def build_primary_rays(
  offset: int,
  img_w: int,
  img_h: int,
  fov_rad: float,
  rays: wp.array(dtype=wp.vec3),
):
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
  rays[offset + tid] = wp.normalize(wp.vec3(dx, dy, dz))


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


def _make_hfield_mesh(mjm: mujoco.MjModel, hfieldid: int) -> tuple[wp.Mesh, wp.vec3]:
  """Create a Warp mesh BVH from mjcf heightfield data."""
  nr = int(mjm.hfield_nrow[hfieldid])
  nc = int(mjm.hfield_ncol[hfieldid])
  sz = np.asarray(mjm.hfield_size[hfieldid], dtype=np.float32)

  adr = int(mjm.hfield_adr[hfieldid])
  data = wp.array(mjm.hfield_data[adr: adr + nr * nc], dtype=float)

  width = 0.5 * max(nc - 1, 1)
  height = 0.5 * max(nr - 1, 1)

  @wp.kernel
  def _build_hfield_points_kernel(
    nr: int,
    nc: int,
    sx: float,
    sy: float,
    sz_scale: float,
    width: float,
    height: float,
    data: wp.array(dtype=float),
    points: wp.array(dtype=wp.vec3),
  ):
    tid = wp.tid()
    total = nr * nc
    if tid >= total:
      return
    r = tid // nc
    c = tid % nc
    x = sx * (float(c) / width - 1.0)
    y = sy * (float(r) / height - 1.0)
    z = data[r * nc + c] * sz_scale
    points[tid] = wp.vec3(x, y, z)

  @wp.kernel
  def _build_hfield_indices_kernel(
    nr: int,
    nc: int,
    indices: wp.array(dtype=int),
  ):
    tid = wp.tid()
    ncell = (nr - 1) * (nc - 1)
    if tid >= ncell:
      return
    r = tid // (nc - 1)
    c = tid % (nc - 1)
    i00 = r * nc + c
    i10 = r * nc + (c + 1)
    i01 = (r + 1) * nc + c
    i11 = (r + 1) * nc + (c + 1)
    # first triangle (CCW): i00, i10, i11
    base0 = (2 * tid) * 3
    indices[base0 + 0] = i00
    indices[base0 + 1] = i10
    indices[base0 + 2] = i11
    # second triangle (CCW): i00, i11, i01
    base1 = (2 * tid + 1) * 3
    indices[base1 + 0] = i00
    indices[base1 + 1] = i11
    indices[base1 + 2] = i01

  n_points = int(nr * nc)
  n_triangles = int((nr - 1) * (nc - 1) * 2)
  points = wp.zeros(n_points, dtype=wp.vec3)
  wp.launch(
    kernel=_build_hfield_points_kernel,
    dim=n_points,
    inputs=[nr, nc, float(sz[0]), float(sz[1]), float(sz[2]), float(width), float(height), data, points],
  )

  indices = wp.zeros(n_triangles * 3, dtype=wp.int32)
  wp.launch(
    kernel=_build_hfield_indices_kernel,
    dim=int((nr - 1) * (nc - 1)),
    inputs=[nr, nc, indices],
  )

  mesh = wp.Mesh(
    points=points,
    indices=indices,
    bvh_constructor="sah",
  )

  min_h = float(np.min(data.numpy()))
  max_h = float(np.max(data.numpy()))
  half_z = 0.5 * (max_h - min_h) * float(sz[2])
  bounds_half = wp.vec3(float(sz[0]), float(sz[1]), half_z)
  return mesh, bounds_half

@wp.kernel
def _make_face_2d_elements(
    vert_xpos: wp.array2d(dtype=wp.vec3),
    flex_elem: wp.array(dtype=int),
    elem_count: int,
    radius: float,
    num_face_vertices: int,
    nfaces: int,
    face_points: wp.array(dtype=wp.vec3),
    face_indices: wp.array(dtype=int),
    group: wp.array(dtype=int),
):
    """Create faces from 2D flex elements (triangles). Two faces (top/bottom) per element."""
    worldid, elemid = wp.tid()

    # Get element vertex indices (3 vertices per triangle)
    i0 = flex_elem[elemid * 3 + 0]
    i1 = flex_elem[elemid * 3 + 1]
    i2 = flex_elem[elemid * 3 + 2]

    # Get vertex positions
    v0 = vert_xpos[worldid, i0]
    v1 = vert_xpos[worldid, i1]
    v2 = vert_xpos[worldid, i2]

    # Compute triangle normal (CCW)
    v01 = v1 - v0
    v02 = v2 - v0
    nrm = wp.cross(v01, v02)
    nrm_len = wp.length(nrm)
    if nrm_len < 1e-8:
        nrm = wp.vec3(0.0, 0.0, 1.0)
    else:
        nrm = nrm / nrm_len

    # Offset vertices by +/- radius along the normal to give the cloth thickness
    offset = nrm * radius

    p0_pos = v0 + offset
    p1_pos = v1 + offset
    p2_pos = v2 + offset

    p0_neg = v0 - offset
    p1_neg = v1 - offset
    p2_neg = v2 - offset

    # Per-world offsets for vertices and triangle groups.
    world_vertex_offset = worldid * num_face_vertices
    world_face_offset = worldid * nfaces

    # First face (top): i0, i1, i2
    face_local_top = 2 * elemid
    base0 = world_vertex_offset + face_local_top * 3
    face_points[base0 + 0] = p0_pos
    face_points[base0 + 1] = p1_pos
    face_points[base0 + 2] = p2_pos

    face_indices[base0 + 0] = base0 + 0
    face_indices[base0 + 1] = base0 + 1
    face_indices[base0 + 2] = base0 + 2

    group[world_face_offset + face_local_top] = worldid

    # Second face (bottom): i0, i2, i1 (opposite winding)
    face_local_bottom = 2 * elemid + 1
    base1 = world_vertex_offset + face_local_bottom * 3
    face_points[base1 + 0] = p0_neg
    face_points[base1 + 1] = p2_neg
    face_points[base1 + 2] = p1_neg

    face_indices[base1 + 0] = base1 + 0
    face_indices[base1 + 1] = base1 + 1
    face_indices[base1 + 2] = base1 + 2

    group[world_face_offset + face_local_bottom] = worldid


@wp.kernel
def _make_sides_2d_elements(
    vert_xpos: wp.array2d(dtype=wp.vec3),
    vert_norm: wp.array2d(dtype=wp.vec3),
    shell_pairs: wp.array(dtype=int),
    shell_count: int,
    radius: float,
    face_offset: int,
    num_face_vertices: int,
    nfaces: int,
    face_points: wp.array(dtype=wp.vec3),
    face_indices: wp.array(dtype=int),
    group: wp.array(dtype=int),
):
    """Create side faces from 2D flex shell fragments.

    For each shell fragment (edge i0 -> i1), we emit two triangles:
      - one using +radius
      - one using -radius (i0/i1 swapped)
    """
    worldid, shellid = wp.tid()
    if shellid >= shell_count or worldid >= vert_xpos.shape[0]:
        return

    # Two local vertex indices per shell fragment (assumed dim == 2).
    i0 = shell_pairs[2 * shellid + 0]
    i1 = shell_pairs[2 * shellid + 1]

    nvert = vert_xpos.shape[1]
    if i0 < 0 or i0 >= nvert or i1 < 0 or i1 >= nvert:
        return

    # Per-world offsets for vertices and triangle groups.
    world_vertex_offset = worldid * num_face_vertices
    world_face_offset = worldid * nfaces

    # Two faces per shell fragment (local to this world).
    face_local0 = face_offset + (2 * shellid)
    face_local1 = face_offset + (2 * shellid + 1)

    # ---- First side: (i0, i1) with +radius ----
    base0 = world_vertex_offset + face_local0 * 3
    # k = 0, ind = i0, sign = +1
    pos = vert_xpos[worldid, i0]
    nrm = vert_norm[worldid, i0]
    p = pos + nrm * (radius * 1.0)
    face_points[base0 + 0] = p
    face_indices[base0 + 0] = base0 + 0
    # k = 1, ind = i1, sign = -1
    pos = vert_xpos[worldid, i1]
    nrm = vert_norm[worldid, i1]
    p = pos + nrm * (radius * -1.0)
    face_points[base0 + 1] = p
    face_indices[base0 + 1] = base0 + 1
    # k = 2, ind = i1, sign = +1
    pos = vert_xpos[worldid, i1]
    nrm = vert_norm[worldid, i1]
    p = pos + nrm * (radius * 1.0)
    face_points[base0 + 2] = p
    face_indices[base0 + 2] = base0 + 2

    # ---- Second side: (i1, i0) with -radius ----
    base1 = world_vertex_offset + face_local1 * 3
    neg_radius = -radius
    # k = 0, ind = i1, sign = +1
    pos = vert_xpos[worldid, i1]
    nrm = vert_norm[worldid, i1]
    p = pos + nrm * (neg_radius * 1.0)
    face_points[base1 + 0] = p
    face_indices[base1 + 0] = base1 + 0
    # k = 1, ind = i0, sign = -1
    pos = vert_xpos[worldid, i0]
    nrm = vert_norm[worldid, i0]
    p = pos + nrm * (neg_radius * -1.0)
    face_points[base1 + 1] = p
    face_indices[base1 + 1] = base1 + 1
    # k = 2, ind = i0, sign = +1
    pos = vert_xpos[worldid, i0]
    nrm = vert_norm[worldid, i0]
    p = pos + nrm * (neg_radius * 1.0)
    face_points[base1 + 2] = p
    face_indices[base1 + 2] = base1 + 2

    group[world_face_offset + face_local0] = worldid
    group[world_face_offset + face_local1] = worldid


def _make_flex_mesh(mjm: mujoco.MjModel, m: Model, d: Data) -> wp.Mesh:
    """Create a Warp BVH mesh for flex meshes.

    We create a single mesh for all flex objects across all worlds.

    This implements the core of MuJoCo's flex rendering path for the 2D flex case by:
      * gathering vertex positions for this flex (world 0),
      * building triangle faces for both sides of the cloth, offset by `radius`
        along the element normal so the cloth has thickness,
      * returning a Warp mesh plus an approximate half-extent for BVH bounds.
    """

    dims = mjm.flex_dim
    assert all(dims == 2), "Only 2D flex is supported"

    vert_norm = wp.zeros(d.flexvert_xpos.shape, dtype=wp.vec3)

    @wp.kernel
    def _compute_vert_norm(vert_xpos: wp.array2d(dtype=wp.vec3), flex_elem: wp.array(dtype=int), vert_norm: wp.array2d(dtype=wp.vec3)):
        worldid, vertid = wp.tid()
        if vertid >= vert_xpos.shape[1] or worldid >= vert_xpos.shape[0]:
            return
        i0 = flex_elem[vertid * 3 + 0]
        i1 = flex_elem[vertid * 3 + 1]
        i2 = flex_elem[vertid * 3 + 2]
        v0 = vert_xpos[worldid, i0]
        v1 = vert_xpos[worldid, i1]
        v2 = vert_xpos[worldid, i2]
        nrm = wp.cross(v1 - v0, v2 - v0)
        nlen = wp.length(nrm)
        if nlen > 1e-8:
            nrm = nrm / nlen
        else:
          nrm = wp.vec3(0.0, 0.0, 1.0)
        vert_norm[worldid, vertid] = nrm

    wp.launch(
        kernel=_compute_vert_norm,
        dim=(d.flexvert_xpos.shape[0], d.flexvert_xpos.shape[1]),
        inputs=[d.flexvert_xpos, m.flex_elem, vert_norm],
    )

    radius = mjm.flex_radius
    shell_count = int(sum(mjm.flex_shellnum))
    elem_count = int(sum(mjm.flex_elemnum))

    shell_pairs = wp.array(mjm.flex_shell, dtype=int)
    n_side_faces = 2 * shell_count
    nfaces = 2 * elem_count + n_side_faces
    num_face_vertices = nfaces * 3

    face_points = wp.zeros(num_face_vertices * d.nworld, dtype=wp.vec3)
    face_indices = wp.zeros(num_face_vertices * d.nworld, dtype=wp.int32)
    # One group id per triangle (primitive), used to build per-world BVH roots.
    group = wp.zeros(nfaces * d.nworld, dtype=int)

    # Build top and bottom faces.
    wp.launch(
        kernel=_make_face_2d_elements,
        dim=(d.nworld, elem_count),
        inputs=[
          d.flexvert_xpos,
          m.flex_elem,
          elem_count,
          radius,
          num_face_vertices,
          nfaces,
          face_points,
          face_indices,
          group,
        ],
    )

    face_offset = 2 * elem_count  # index after top and bottom faces
    wp.launch(
        kernel=_make_sides_2d_elements,
        dim=(d.nworld, shell_count),
        inputs=[
          d.flexvert_xpos,
          vert_norm,
          shell_pairs,
          shell_count,
          radius,
          face_offset,
          num_face_vertices,
          nfaces,
          face_points,
          face_indices,
          group,
        ],
    )

    flex_mesh = wp.Mesh(points=face_points, indices=face_indices, groups=group, bvh_constructor="sah")

    group_roots = wp.zeros(d.nworld, dtype=int)
    wp.launch(
      kernel=bvh.compute_bvh_group_roots,
      dim=d.nworld,
      inputs=[flex_mesh.id, group_roots],
    )

    return flex_mesh, face_points, group, group_roots, face_offset, shell_count, elem_count
