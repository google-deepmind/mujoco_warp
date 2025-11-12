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
    nflex = int(m.nflex)
    self.flex_registry = {}
    flex_bvh_ids = [wp.uint64(0) for _ in range(nflex)]
    flex_bounds_size = [wp.vec3(0.0, 0.0, 0.0) for _ in range(nflex)]
    for fid in range(nflex):
      dim = int(mjm.flex_dim[fid])
      if dim != 2:
        continue
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

    geom_count = len(geom_enabled_idx)
    nflex = int(m.nflex)
    bvh_ngeom = geom_count + nflex
    self.geom_count=geom_count
    self.bvh_ngeom=bvh_ngeom
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
    self.lowers = wp.zeros(d.nworld * bvh_ngeom, dtype=wp.vec3)
    self.uppers = wp.zeros(d.nworld * bvh_ngeom, dtype=wp.vec3)
    self.groups = wp.zeros(d.nworld * bvh_ngeom, dtype=int)
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

def _make_flex_mesh(mjm, d, flexid: int) -> tuple[wp.Mesh, wp.vec3]:
    """Create a Warp BVH mesh for a MuJoCo flex using edges+flaps (no mjvScene)."""

    base_vert = int(mjm.flex_vertadr[flexid])
    nvert     = int(mjm.flex_vertnum[flexid])

    edge_start = int(mjm.flex_edgeadr[flexid])
    nedges     = int(mjm.flex_edgenum[flexid])
    edge_stop  = edge_start + nedges

    # Edges and flaps for *this* flex (local vertex IDs)
    # mjm.flex_edge: flat int array shaped (..., 2)
    # mjm.flex_edgeflap: flat int array shaped (..., 2) with -1 for "no triangle"
    edges_all  = np.asarray(mjm.flex_edge, dtype=np.int32).reshape(-1, 2)
    flaps_all  = np.asarray(mjm.flex_edgeflap, dtype=np.int32).reshape(-1, 2)

    edges  = edges_all[edge_start:edge_stop]
    flaps  = flaps_all[edge_start:edge_stop]

    # Build unique triangles from edges+flaps.
    # Indices must be *local* (0..nvert-1) because points array is a local slice.
    tri_list = []

    for e in range(nedges):
        v0, v1 = int(edges[e, 0]), int(edges[e, 1])
        # two possible triangles per edge (one per "flap" / side)
        f0, f1 = int(flaps[e, 0]), int(flaps[e, 1])

        for f in (f0, f1):
            if f < 0:
                continue

            i, j, k = v0, v1, f  # local indices
            # canonical ownership test: only emit once
            a = min(i, j, k)
            c = max(i, j, k)
            b = i + j + k - a - c

            # current edge in ascending order
            ei0, ei1 = (i, j) if i < j else (j, i)

            # emit only if this edge == (a, b)
            if ei0 == a and ei1 == b:
                tri_list.append((i, j, k))

    if not tri_list:
        # empty mesh fallback
        empty_points = wp.array(np.zeros((0, 3), dtype=np.float32), dtype=wp.vec3)
        empty_indices = wp.array(np.zeros((0,), dtype=np.int32), dtype=wp.int32)
        mesh = wp.Mesh(points=empty_points, indices=empty_indices, bvh_constructor="sah")
        return mesh, wp.vec3(0.0, 0.0, 0.0)

    # Convert to contiguous index array (local indexing)
    tris = np.asarray(tri_list, dtype=np.int32).reshape(-1, 3)
    indices = wp.array(tris.ravel(), dtype=wp.int32)

    # Vertex positions: local slice (Nx3). Use d.flexvert_xpos so you're in world space.
    # If your Warp mesh/BVH should be in local space, swap this for the model-space array.
    pts = np.asarray(d.flexvert_xpos.numpy()[0, base_vert: base_vert + nvert], dtype=np.float32)
    points = wp.array(pts, dtype=wp.vec3)

    flex_mesh = wp.Mesh(points=points, indices=indices, bvh_constructor="sah")

    # Bounds from these points
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    half = 0.5 * (pmax - pmin)
    return flex_mesh, wp.vec3(float(half[0]), float(half[1]), float(half[2]))