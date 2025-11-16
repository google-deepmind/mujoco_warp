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
  flex_bvh_ids: wp.array(dtype=wp.uint64)
  flex_bounds_size: wp.array(dtype=wp.vec3)
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
    # Current only supports 2D flex (cloth)
    nflex = int(m.nflex)
    self.flex_registry = {}
    flex_bvh_ids = [wp.uint64(0) for _ in range(nflex)]
    flex_bounds_size = [wp.vec3(0.0, 0.0, 0.0) for _ in range(nflex)]
    for fid in range(nflex):
      if int(mjm.flex_dim[fid]) == 2:
        fmesh, fhalf = _make_flex_mesh(mjm, d, fid)
        self.flex_registry[fmesh.id] = fmesh
        flex_bvh_ids[fid] = fmesh.id
        flex_bounds_size[fid] = fhalf

    tex_data_packed, tex_adr_packed = _create_packed_texture_data(mjm)

    # If a global camera resolution is provided, use it for all cameras
    # otherwise check the xml for camera resolutions
    if cam_resolutions is not None:
      if isinstance(cam_resolutions, tuple):
        cam_resolutions = [cam_resolutions] * mjm.ncam
      assert len(cam_resolutions) == mjm.ncam, "Camera resolutions must be provided for all cameras"
      self.cam_resolutions = wp.array(cam_resolutions, dtype=wp.vec2i)
    else:
      self.cam_resolutions = wp.array(mjm.cam_resolution, dtype=wp.vec2i)
    
    if isinstance(render_rgb, bool):
      render_rgb = [render_rgb] * mjm.ncam
    assert len(render_rgb) == mjm.ncam, "Render RGB must be provided for all cameras"
    
    if isinstance(render_depth, bool):
      render_depth = [render_depth] * mjm.ncam
    assert len(render_depth) == mjm.ncam, "Render depth must be provided for all cameras"

    rgb_offsets = [-1 for _ in range(mjm.ncam)]
    depth_offsets = [-1 for _ in range(mjm.ncam)]
    cam_resolutions = self.cam_resolutions.numpy()
    ri = 0
    di = 0
    total = 0

    for i in range(mjm.ncam):
      if render_rgb[i]:
        rgb_offsets[i] = ri
        ri += cam_resolutions[i][0] * cam_resolutions[i][1]
      if render_depth[i]:
        depth_offsets[i] = di
        di += cam_resolutions[i][0] * cam_resolutions[i][1]

      total += cam_resolutions[i][0] * cam_resolutions[i][1]

    self.rgb_offsets = wp.array(rgb_offsets, dtype=int)
    self.depth_offsets = wp.array(depth_offsets, dtype=int)
    self.rgb = wp.zeros((d.nworld, ri), dtype=wp.uint32)
    self.depth = wp.zeros((d.nworld, di), dtype=wp.float32)
    self.rays = wp.zeros(int(total), dtype=wp.vec3)

    offset = 0
    for i in range(mjm.ncam):
      wp.launch(
        kernel=build_primary_rays,
        dim=int(cam_resolutions[i][0] * cam_resolutions[i][1]),
        inputs=[offset, cam_resolutions[i][0], cam_resolutions[i][1], wp.radians(float(mjm.cam_fovy[i])), self.rays],
      )
      offset += cam_resolutions[i][0] * cam_resolutions[i][1]

    self.bvh_ngeom=len(geom_enabled_idx)
    self.ncam=mjm.ncam
    self.use_textures=use_textures
    self.use_shadows=use_shadows
    self.render_rgb=render_rgb
    self.render_depth=render_depth
    self.enabled_geom_ids=wp.array(geom_enabled_idx, dtype=int)
    self.mesh_bvh_ids=wp.array(mesh_bvh_ids, dtype=wp.uint64)
    self.mesh_bounds_size=wp.array(mesh_bounds_size, dtype=wp.vec3)
    self.hfield_bvh_ids=wp.array(hfield_bvh_ids, dtype=wp.uint64)
    self.hfield_bounds_size=wp.array(hfield_bounds_size, dtype=wp.vec3)
    self.flex_bvh_ids=wp.array(flex_bvh_ids, dtype=wp.uint64)
    self.flex_bounds_size=wp.array(flex_bounds_size, dtype=wp.vec3)
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
) -> RenderContext:
  """Creates a render context on device.

    Args:
      mjm: The model containing kinematic and dynamic information on host.
      m: The model on device.
      d: The data on device.
      width: The width to render every camera image.
      height: The height to render every camera image.
      use_textures: Whether to use textures.
      use_shadows: Whether to use shadows.
      render_rgb: Whether to render RGB images.
      render_depth: Whether to render depth images.
      enabled_geom_groups: The geom groups to render.

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
    vert_xpos: wp.array(dtype=wp.vec3),
    flex_elem: wp.array(dtype=int),
    elem_start: int,
    elem_count: int,
    radius: float,
    face_points: wp.array(dtype=wp.vec3),
    face_indices: wp.array(dtype=int),
):
    """Create faces from 2D flex elements (triangles). Two faces (top/bottom) per element."""
    tid = wp.tid()
    if tid >= elem_count:
        return

    # Get element vertex indices (3 vertices per triangle)
    elem_base = elem_start + tid * 3
    i0 = flex_elem[elem_base + 0]
    i1 = flex_elem[elem_base + 1]
    i2 = flex_elem[elem_base + 2]

    # Get vertex positions
    v0 = vert_xpos[i0]
    v1 = vert_xpos[i1]
    v2 = vert_xpos[i2]

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
    offset_pos = nrm * radius
    offset_neg = -offset_pos

    p0_pos = v0 + offset_pos
    p1_pos = v1 + offset_pos
    p2_pos = v2 + offset_pos

    p0_neg = v0 + offset_neg
    p1_neg = v1 + offset_neg
    p2_neg = v2 + offset_neg

    # First face (top): i0, i1, i2
    face_base = tid * 6  # 2 faces * 3 vertices
    face_points[face_base + 0] = p0_pos
    face_points[face_base + 1] = p1_pos
    face_points[face_base + 2] = p2_pos

    # Second face (bottom): i0, i2, i1 (opposite winding)
    face_points[face_base + 3] = p0_neg
    face_points[face_base + 4] = p2_neg
    face_points[face_base + 5] = p1_neg

    # Set indices (using sequential indices for the face points)
    idx_base = tid * 6
    face_indices[idx_base + 0] = idx_base + 0
    face_indices[idx_base + 1] = idx_base + 1
    face_indices[idx_base + 2] = idx_base + 2
    face_indices[idx_base + 3] = idx_base + 3
    face_indices[idx_base + 4] = idx_base + 4
    face_indices[idx_base + 5] = idx_base + 5


@wp.kernel
def _make_sides_2d_elements(
    vert_xpos: wp.array(dtype=wp.vec3),
    vert_norm: wp.array(dtype=wp.vec3),
    shell_pairs: wp.array(dtype=int),
    shell_count: int,
    radius: float,
    face_offset: int,
    face_points: wp.array(dtype=wp.vec3),
    face_indices: wp.array(dtype=int),
):
    """Create side faces from 2D flex shell fragments.

    For each shell fragment (edge i0 -> i1), we emit two triangles:
      - one using +radius
      - one using -radius (i0/i1 swapped)
    """
    tid = wp.tid()
    if tid >= shell_count:
        return

    # Two local vertex indices per shell fragment (assumed dim == 2).
    i0 = shell_pairs[2 * tid + 0]
    i1 = shell_pairs[2 * tid + 1]

    nvert = vert_xpos.shape[0]
    if i0 < 0 or i0 >= nvert or i1 < 0 or i1 >= nvert:
        return

    # Two faces per shell fragment.
    face_idx0 = face_offset + 2 * tid
    face_idx1 = face_offset + 2 * tid + 1

    # ---- First side: (i0, i1) with +radius ----
    base0 = face_idx0 * 3
    # k = 0, ind = i0, sign = +1
    pos = vert_xpos[i0]
    nrm = vert_norm[i0]
    p = pos + nrm * (radius * 1.0)
    face_points[base0 + 0] = p
    face_indices[base0 + 0] = base0 + 0
    # k = 1, ind = i1, sign = -1
    pos = vert_xpos[i1]
    nrm = vert_norm[i1]
    p = pos + nrm * (radius * -1.0)
    face_points[base0 + 1] = p
    face_indices[base0 + 1] = base0 + 1
    # k = 2, ind = i1, sign = +1
    pos = vert_xpos[i1]
    nrm = vert_norm[i1]
    p = pos + nrm * (radius * 1.0)
    face_points[base0 + 2] = p
    face_indices[base0 + 2] = base0 + 2

    # ---- Second side: (i1, i0) with -radius ----
    base1 = face_idx1 * 3
    neg_radius = -radius
    # k = 0, ind = i1, sign = +1
    pos = vert_xpos[i1]
    nrm = vert_norm[i1]
    p = pos + nrm * (neg_radius * 1.0)
    face_points[base1 + 0] = p
    face_indices[base1 + 0] = base1 + 0
    # k = 1, ind = i0, sign = -1
    pos = vert_xpos[i0]
    nrm = vert_norm[i0]
    p = pos + nrm * (neg_radius * -1.0)
    face_points[base1 + 1] = p
    face_indices[base1 + 1] = base1 + 1
    # k = 2, ind = i0, sign = +1
    pos = vert_xpos[i0]
    nrm = vert_norm[i0]
    p = pos + nrm * (neg_radius * 1.0)
    face_points[base1 + 2] = p
    face_indices[base1 + 2] = base1 + 2


def _make_flex_mesh(mjm: mujoco.MjModel, d: Data) -> wp.Mesh:
    """Create a Warp BVH mesh for flex meshes.

    We create a single mesh for all flex objects across all worlds.

    This implements the core of MuJoCo's flex rendering path for the 2D flex case by:
      * gathering vertex positions for this flex (world 0),
      * building triangle faces for both sides of the cloth, offset by `radius`
        along the element normal so the cloth has thickness,
      * returning a Warp mesh plus an approximate half-extent for BVH bounds.
    """

    # We assume 2D flex (cloth) and that all flex-related fields exist.
    dim = int(mjm.flex_dim[flexid])

    base_vert = int(mjm.flex_vertadr[flexid])
    nvert = int(mjm.flex_vertnum[flexid])

    flexvert_xpos_np = d.flexvert_xpos.numpy()
    vert_xpos_np = flexvert_xpos_np[0, base_vert : base_vert + nvert, :].astype(np.float32)
    vert_xpos = wp.array(vert_xpos_np, dtype=wp.vec3)

    elem_count = int(mjm.flex_elemnum[flexid])
    elem_start = int(mjm.flex_elemdataadr[flexid])

    flex_elem_global = np.asarray(
        mjm.flex_elem[elem_start : elem_start + elem_count * (dim + 1)],
        dtype=np.int32,
    )
    flex_elem_local = flex_elem_global - base_vert
    flex_elem_tri = flex_elem_local.reshape(elem_count, dim + 1)
    flex_elem_wp = wp.array(flex_elem_local, dtype=int)

    # Build per-vertex normals (vertnorm) in host memory, similar to MuJoCo.
    vertnorm = np.zeros_like(vert_xpos_np, dtype=np.float32)
    eps = 1.0e-8
    for tri in flex_elem_tri:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v0 = vert_xpos_np[i0]
        v1 = vert_xpos_np[i1]
        v2 = vert_xpos_np[i2]
        nrm = np.cross(v1 - v0, v2 - v0)
        nlen = float(np.linalg.norm(nrm))
        if nlen > eps:
            nrm /= nlen
        else:
            continue
        vertnorm[i0] += nrm
        vertnorm[i1] += nrm
        vertnorm[i2] += nrm

    norms = np.linalg.norm(vertnorm, axis=1)
    valid = norms > eps
    if np.any(valid):
        vertnorm[valid] /= norms[valid][:, None]
    if np.any(~valid):
        vertnorm[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    vert_norm = wp.array(vertnorm, dtype=wp.vec3)

    radius = float(getattr(mjm, "flex_radius", [0.0] * mjm.nflex)[flexid])

    # Number of side faces from shell fragments (2 per shell fragment).
    shell_count = int(mjm.flex_shellnum[flexid])
    shell_start = int(mjm.flex_shelldataadr[flexid])
    shell_flat = np.asarray(
        mjm.flex_shell[shell_start : shell_start + shell_count * dim],
        dtype=np.int32,
    ).reshape(shell_count, dim)
    # Convert shell vertex indices to local indices [0, nvert) for this flex.
    shell_pairs_local = (shell_flat - base_vert).astype(np.int32)
    shell_pairs_flat = shell_pairs_local.reshape(shell_count * dim)
    shell_pairs_wp = wp.array(shell_pairs_flat, dtype=int)

    n_side_faces = 2 * shell_count
    nfaces = 2 * elem_count + n_side_faces
    num_face_vertices = nfaces * 3
    face_points = wp.zeros(num_face_vertices, dtype=wp.vec3)
    face_indices = wp.zeros(num_face_vertices, dtype=wp.int32)

    # Build top and bottom faces.
    wp.launch(
        kernel=_make_face_2d_elements,
        dim=elem_count,
        inputs=[vert_xpos, flex_elem_wp, 0, elem_count, radius, face_points, face_indices],
    )

    # Build side faces from flex shell fragments.
    if shell_count > 0:
        face_offset = 2 * elem_count  # index of first side face
        wp.launch(
            kernel=_make_sides_2d_elements,
            dim=shell_count,
            inputs=[
                vert_xpos,
                vert_norm,
                shell_pairs_wp,
                shell_count,
                radius,
                face_offset,
                face_points,
                face_indices,
            ],
        )

    flex_mesh = wp.Mesh(points=face_points, indices=face_indices, bvh_constructor="sah")

    # Compute an approximate half-extent from the generated vertices.
    face_points_np = face_points.numpy()
    pmin = face_points_np.min(axis=0)
    pmax = face_points_np.max(axis=0)
    half = 0.5 * (pmax - pmin)

    return flex_mesh, wp.vec3(float(half[0]), float(half[1]), float(half[2]))