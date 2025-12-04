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
  flex_rgba: wp.array(dtype=wp.vec4)
  flex_matid: wp.array(dtype=int)
  bvh_id: wp.uint64
  mesh_bvh_ids: wp.array(dtype=wp.uint64)
  hfield_bvh_ids: wp.array(dtype=wp.uint64)
  hfield_bounds_size: wp.array(dtype=wp.vec3)
  flex_bvh_id: wp.uint64
  flex_face_points: wp.array(dtype=wp.vec3)
  flex_elem: wp.array(dtype=int)
  flex_faceadr: wp.array(dtype=int)
  flex_nfaces: int
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
    flex_render_smooth: Optional[bool] = False,
  ):

    # Mesh BVHs
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
      mesh, half = _build_mesh_bvh(mjm, i)
      self.mesh_registry[mesh.id] = mesh
      mesh_bvh_ids[i] = mesh.id
      mesh_bounds_size[i] = half

    self.mesh_bvh_ids=wp.array(mesh_bvh_ids, dtype=wp.uint64)
    self.mesh_bounds_size=wp.array(mesh_bounds_size, dtype=wp.vec3)

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
      hmesh, hhalf = _build_hfield_mesh(mjm, hid)
      self.hfield_registry[hmesh.id] = hmesh
      hfield_bvh_ids[hid] = hmesh.id
      hfield_bounds_size[hid] = hhalf

    self.hfield_bvh_ids=wp.array(hfield_bvh_ids, dtype=wp.uint64)
    self.hfield_bounds_size=wp.array(hfield_bounds_size, dtype=wp.vec3)

    # Flex BVHs
    self.flex_registry = {}
    self.flex_bvh_id = wp.uint64(0)
    self.flex_group_roots = wp.zeros(d.nworld, dtype=int)
    if mjm.nflex > 0:
      (
        fmesh,
        face_points,
        flex_groups,
        flex_group_roots,
        flex_elem,
        flex_shell,
        flex_faceadr,
        flex_nfaces,
      ) = _make_flex_mesh(mjm, m, d)

      self.flex_registry[fmesh.id] = fmesh
      self.flex_bvh_id = fmesh.id
      self.flex_face_points = face_points
      self.flex_groups = flex_groups
      self.flex_group_roots = flex_group_roots
      self.flex_dim = mjm.flex_dim
      self.flex_elem = flex_elem
      self.flex_elemnum = mjm.flex_elemnum
      self.flex_elemdataadr = mjm.flex_elemdataadr
      self.flex_shell = flex_shell
      self.flex_shellnum = mjm.flex_shellnum
      self.flex_shelldataadr = mjm.flex_shelldataadr
      self.flex_faceadr = flex_faceadr
      self.flex_nfaces = flex_nfaces
      self.flex_radius = mjm.flex_radius
      self.flex_render_smooth = flex_render_smooth

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

    self.ncam=n_active_cams
    self.cam_id_map=wp.array(active_cam_indices, dtype=int)
    self.use_textures=use_textures
    self.use_shadows=use_shadows
    self.mesh_texcoord=wp.array(mjm.mesh_texcoord, dtype=wp.vec2)
    self.mesh_texcoord_offsets=wp.array(mjm.mesh_texcoordadr, dtype=int)
    self.mesh_texcoord_num=wp.array(mjm.mesh_texcoordnum, dtype=int)
    self.tex_adr=tex_adr_packed
    self.tex_data=tex_data_packed
    self.tex_height = wp.array(mjm.tex_height, dtype=int)
    self.tex_width = wp.array(mjm.tex_width, dtype=int)
    self.flex_rgba = wp.array(mjm.flex_rgba, dtype=wp.vec4)
    self.flex_matid = wp.array(mjm.flex_matid, dtype=int)
    self.bvh_ngeom=len(geom_enabled_idx)
    self.enabled_geom_ids=wp.array(geom_enabled_idx, dtype=int)
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
  flex_render_smooth: Optional[bool] = True,
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
    flex_render_smooth,
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


def _build_mesh_bvh(
  mjm: mujoco.MjModel,
  meshid: int,
  constructor: str = "sah",
  leaf_size: int = 1,
) -> tuple[wp.Mesh, wp.vec3]:
  """Create a Warp mesh BVH from mjcf mesh data."""
  v_start = mjm.mesh_vertadr[meshid]
  v_end = v_start + mjm.mesh_vertnum[meshid]
  points = mjm.mesh_vert[v_start:v_end]

  f_start = mjm.mesh_faceadr[meshid]
  f_end = mjm.mesh_face.shape[0] if (meshid + 1) >= mjm.mesh_faceadr.shape[0] else mjm.mesh_faceadr[meshid + 1]
  indices = mjm.mesh_face[f_start:f_end]
  indices = indices.flatten()
  pmin = np.min(points, axis=0)
  pmax = np.max(points, axis=0)
  half = 0.5 * (pmax - pmin)

  points = wp.array(points, dtype=wp.vec3)
  indices = wp.array(indices, dtype=wp.int32)
  mesh = wp.Mesh(points=points, indices=indices, bvh_constructor=constructor, bvh_leaf_size=leaf_size)

  return mesh, half


def _optimize_hfield_mesh(
  data: np.ndarray,
  nr: int,
  nc: int,
  sx: float,
  sy: float,
  sz_scale: float,
  width: float,
  height: float,
) -> tuple[np.ndarray, np.ndarray]:
  """
  Greedy meshing for heightfield optimization.
  
  Merges coplanar adjacent cells into larger rectangles to
  reduce triangle and vertex count.
  """
  points_map = {}
  points_list = []
  indices_list = []

  def get_point_index(r, c):
    if (r, c) in points_map:
      return points_map[(r, c)]

    # Compute vertex position
    x = sx * (float(c) / width - 1.0)
    y = sy * (float(r) / height - 1.0)
    z = float(data[r, c]) * sz_scale

    idx = len(points_list)
    points_list.append([x, y, z])
    points_map[(r, c)] = idx
    return idx

  visited = np.zeros((nr - 1, nc - 1), dtype=bool)

  for r in range(nr - 1):
    for c in range(nc - 1):
      if visited[r, c]:
        continue

      # Check if current cell is planar
      z00 = data[r, c]
      z01 = data[r, c + 1]
      z10 = data[r + 1, c]
      z11 = data[r + 1, c + 1]

      # Approx check for planarity: z00 + z11 == z01 + z10
      is_planar = abs((z00 + z11) - (z01 + z10)) < 1e-5

      if not is_planar:
        # Must emit single cell (2 triangles)
        idx00 = get_point_index(r, c)
        idx01 = get_point_index(r, c + 1)
        idx10 = get_point_index(r + 1, c)
        idx11 = get_point_index(r + 1, c + 1)

        # Tri 1: TL, TR, BR
        indices_list.extend([idx00, idx01, idx11])
        # Tri 2: TL, BR, BL
        indices_list.extend([idx00, idx11, idx10])
        visited[r, c] = True
        continue

      # If planar, try to expand
      slope_x = z01 - z00
      slope_y = z10 - z00
      w = 1
      h = 1

      def fits_plane(rr, cc):
        if rr >= nr - 1 or cc >= nc - 1:
          return False
        # Check planarity of the cell itself
        cz00 = data[rr, cc]
        cz01 = data[rr, cc + 1]
        cz10 = data[rr + 1, cc]
        cz11 = data[rr + 1, cc + 1]
        if abs((cz00 + cz11) - (cz01 + cz10)) >= 1e-5:
          return False

        # Check if it lies on the SAME plane as start cell
        # Expected z at (rr, cc)
        z_pred = z00 + (rr - r) * slope_y + (cc - c) * slope_x
        if abs(cz00 - z_pred) >= 1e-5:
          return False
        
        # Since cell is planar and one corner matches, slopes must match if connected
        cslope_x = cz01 - cz00
        cslope_y = cz10 - cz00
        if abs(cslope_x - slope_x) >= 1e-5 or abs(cslope_y - slope_y) >= 1e-5:
          return False
          
        return True

      # Expand width
      while c + w < nc - 1 and not visited[r, c + w] and fits_plane(r, c + w):
        w += 1

      # Expand height
      while r + h < nr - 1:
        # Check entire row
        row_ok = True
        for k in range(w):
          if visited[r + h, c + k] or not fits_plane(r + h, c + k):
            row_ok = False
            break
        if row_ok:
          h += 1
        else:
          break

      # Mark visited
      visited[r : r + h, c : c + w] = True

      # Emit large quad
      idx_tl = get_point_index(r, c)
      idx_tr = get_point_index(r, c + w)
      idx_bl = get_point_index(r + h, c)
      idx_br = get_point_index(r + h, c + w)

      # Tri 1: TL, TR, BR
      indices_list.extend([idx_tl, idx_tr, idx_br])
      # Tri 2: TL, BR, BL
      indices_list.extend([idx_tl, idx_br, idx_bl])

  return np.array(points_list, dtype=np.float32), np.array(indices_list, dtype=np.int32)


def _build_hfield_mesh(
  mjm: mujoco.MjModel,
  hfieldid: int,
  constructor: str = "sah",
  leaf_size: int = 1,
) -> tuple[wp.Mesh, wp.vec3]:
  """Create a Warp mesh BVH from mjcf heightfield data."""
  nr = int(mjm.hfield_nrow[hfieldid])
  nc = int(mjm.hfield_ncol[hfieldid])
  sz = np.asarray(mjm.hfield_size[hfieldid], dtype=np.float32)

  adr = int(mjm.hfield_adr[hfieldid])
  # Use host data for optimization
  data = mjm.hfield_data[adr: adr + nr * nc].reshape(nr, nc)

  width = 0.5 * max(nc - 1, 1)
  height = 0.5 * max(nr - 1, 1)

  points, indices = _optimize_hfield_mesh(
      data,
      nr,
      nc,
      float(sz[0]),
      float(sz[1]),
      float(sz[2]),
      float(width),
      float(height),
  )
  pmin = np.min(points, axis=0)
  pmax = np.max(points, axis=0)
  half = 0.5 * (pmax - pmin)

  points = wp.array(points, dtype=wp.vec3)
  indices = wp.array(indices, dtype=wp.int32)

  mesh = wp.Mesh(
    points=points,
    indices=indices,
    bvh_constructor=constructor,
    bvh_leaf_size=leaf_size,
  )

  return mesh, half


@wp.kernel
def _make_face_2d_elements(
    vert_xpos: wp.array2d(dtype=wp.vec3),
    flex_elem: wp.array(dtype=int),
    vert_norm: wp.array2d(dtype=wp.vec3),
    elem_adr: int,
    face_offset: int,
    radius: float,
    nfaces: int,
    face_points: wp.array(dtype=wp.vec3),
    face_indices: wp.array(dtype=int),
    group: wp.array(dtype=int),
):
    """Create faces from 2D flex elements (triangles).
    
    Two faces (top/bottom) per element, seperated by the radius of the flex element.
    """
    worldid, elemid = wp.tid()

    base = elem_adr + (elemid * 3)
    i0 = flex_elem[base + 0]
    i1 = flex_elem[base + 1]
    i2 = flex_elem[base + 2]

    v0 = vert_xpos[worldid, i0]
    v1 = vert_xpos[worldid, i1]
    v2 = vert_xpos[worldid, i2]

    n0 = vert_norm[worldid, i0]
    n1 = vert_norm[worldid, i1]
    n2 = vert_norm[worldid, i2]

    p0_pos = v0 + radius * n0
    p1_pos = v1 + radius * n1
    p2_pos = v2 + radius * n2

    p0_neg = v0 - radius * n0
    p1_neg = v1 - radius * n1
    p2_neg = v2 - radius * n2

    world_face_offset = worldid * nfaces

    # First face (top): i0, i1, i2
    face_id0 = world_face_offset + face_offset + (2 * elemid)
    base0 = face_id0 * 3
    face_points[base0 + 0] = p0_pos
    face_points[base0 + 1] = p1_pos
    face_points[base0 + 2] = p2_pos

    face_indices[base0 + 0] = base0 + 0
    face_indices[base0 + 1] = base0 + 1
    face_indices[base0 + 2] = base0 + 2

    group[face_id0] = worldid

    # Second face (bottom): i0, i2, i1 (opposite winding)
    face_id1 = world_face_offset + face_offset + (2 * elemid + 1)
    base1 = face_id1 * 3
    face_points[base1 + 0] = p0_neg
    face_points[base1 + 1] = p2_neg
    face_points[base1 + 2] = p1_neg

    face_indices[base1 + 0] = base1 + 0
    face_indices[base1 + 1] = base1 + 1
    face_indices[base1 + 2] = base1 + 2

    group[face_id1] = worldid


@wp.kernel
def _make_sides_2d_elements(
    vert_xpos: wp.array2d(dtype=wp.vec3),
    vert_norm: wp.array2d(dtype=wp.vec3),
    flex_shell: wp.array(dtype=int),
    shell_adr: int,
    face_offset: int,
    radius: float,
    nfaces: int,
    face_points: wp.array(dtype=wp.vec3),
    face_indices: wp.array(dtype=int),
    groups: wp.array(dtype=int),
):
    """Create side faces from 2D flex shell fragments.

    For each shell fragment (edge i0 -> i1), we emit two triangles:
      - one using +radius
      - one using -radius (i0/i1 swapped)
    """
    worldid, shellid = wp.tid()
    if shellid >= flex_shell.shape[0] or worldid >= vert_xpos.shape[0]:
        return

    base = shell_adr + (2 * shellid)
    i0 = flex_shell[base + 0]
    i1 = flex_shell[base + 1]

    world_face_offset = worldid * nfaces

    # First side i0, i1 with +radius
    face_id0 = world_face_offset + face_offset + (2 * shellid)
    base0 = face_id0 * 3
    face_points[base0 + 0] = vert_xpos[worldid, i0] + vert_norm[worldid, i0] * (radius * 1.0)
    face_points[base0 + 1] = vert_xpos[worldid, i1] + vert_norm[worldid, i1] * (radius * -1.0)
    face_points[base0 + 2] = vert_xpos[worldid, i1] + vert_norm[worldid, i1] * (radius * 1.0)
    face_indices[base0 + 0] = base0 + 0
    face_indices[base0 + 1] = base0 + 1
    face_indices[base0 + 2] = base0 + 2

    # Second side i1, i0 with -radius
    face_id1 = world_face_offset + face_offset + (2 * shellid + 1)
    base1 = face_id1 * 3
    face_points[base1 + 0] = vert_xpos[worldid, i1] + vert_norm[worldid, i1] * (-radius * 1.0)
    face_points[base1 + 1] = vert_xpos[worldid, i0] + vert_norm[worldid, i0] * (-radius * -1.0)
    face_points[base1 + 2] = vert_xpos[worldid, i0] + vert_norm[worldid, i0] * (-radius * 1.0)
    face_indices[base1 + 0] = base1 + 0
    face_indices[base1 + 1] = base1 + 1
    face_indices[base1 + 2] = base1 + 2

    groups[face_id0] = worldid
    groups[face_id1] = worldid


@wp.kernel
def _make_faces_3d_shells(
    vert_xpos: wp.array2d(dtype=wp.vec3),
    flex_shell: wp.array(dtype=int),
    shell_adr: int,
    face_offset: int,
    nfaces: int,
    face_points: wp.array(dtype=wp.vec3),
    face_indices: wp.array(dtype=int),
    groups: wp.array(dtype=int),
):
    """Create faces from 3D flex shell fragments (triangles).

    Each shell fragment contributes a single triangle whose vertices are taken
    directly from the flex vertex positions (one-sided surface).
    """
    worldid, shellid = wp.tid()

    base = shell_adr + (shellid * 3)
    i0 = flex_shell[base + 0]
    i1 = flex_shell[base + 1]
    i2 = flex_shell[base + 2]

    world_face_offset = worldid * nfaces
    face_id = world_face_offset + face_offset + shellid
    base = face_id * 3

    v0 = vert_xpos[worldid, i0]
    v1 = vert_xpos[worldid, i1]
    v2 = vert_xpos[worldid, i2]

    face_points[base + 0] = v0
    face_points[base + 1] = v1
    face_points[base + 2] = v2

    face_indices[base + 0] = base + 0
    face_indices[base + 1] = base + 1
    face_indices[base + 2] = base + 2

    face_id = world_face_offset + face_offset + shellid
    groups[face_id] = worldid


def _make_flex_mesh(mjm: mujoco.MjModel, m: Model, d: Data):
    """Create a Warp Mesh for flex meshes.

    We create a single Warp Mesh (single BVH) for all flex objects across all worlds.

    This implements the core of MuJoCo's flex rendering path for the 2D flex case by:
      * gathering vertex positions for this flex,
      * building triangle faces for both sides of the cloth, offset by `radius`
        along the element normal so the cloth has thickness,
      * returning a Warp mesh plus an approximate half-extent for BVH bounds.
    """

    if any(mjm.flex_dim[i] == 1 for i in range(mjm.nflex)):
      raise ValueError("1D Flex objects are not currently supported.")

    nflex = int(mjm.nflex)
    
    flex_faceadr = [0]
    for f in range(nflex):
      if int(mjm.flex_dim[f]) == 2:
        flex_faceadr.append(flex_faceadr[-1] + (2 * int(mjm.flex_elemnum[f]) + 2 * int(mjm.flex_shellnum[f])))
      elif int(mjm.flex_dim[f]) == 3:
        flex_faceadr.append(flex_faceadr[-1] + int(mjm.flex_shellnum[f]))

    nfaces = int(flex_faceadr[-1])
    flex_faceadr = flex_faceadr[:-1]

    face_points = wp.zeros(nfaces * 3 * d.nworld, dtype=wp.vec3)
    face_indices = wp.zeros(nfaces * 3 * d.nworld, dtype=wp.int32)
    groups = wp.zeros(nfaces * d.nworld, dtype=int)

    vert_norm = wp.zeros(d.flexvert_xpos.shape, dtype=wp.vec3)
    flex_elem = wp.array(mjm.flex_elem, dtype=int)
    flex_shell = wp.array(mjm.flex_shell, dtype=int)

    wp.launch(
      kernel=bvh.accumulate_flex_vertex_normals,
      dim=(d.nworld, m.nflexelem),
      inputs=[
        d.flexvert_xpos,
        flex_elem,
        m.nflexelem,
        vert_norm,
      ],
    )

    wp.launch(
      kernel=bvh.normalize_vertex_normals,
      dim=(d.nworld, m.nflexvert),
      inputs=[vert_norm],
    )

    for f in range(nflex):
      dim = int(mjm.flex_dim[f])
      elem_adr = int(mjm.flex_elemdataadr[f])
      nelem = int(mjm.flex_elemnum[f])
      shell_adr = int(mjm.flex_shelldataadr[f])
      nshell = int(mjm.flex_shellnum[f])
      
      if dim == 2:
        wp.launch(
          kernel=_make_face_2d_elements,
          dim=(d.nworld, nelem),
          inputs=[
            d.flexvert_xpos,
            flex_elem,
            vert_norm,
            elem_adr,
            flex_faceadr[f],
            mjm.flex_radius[f],
            nfaces,
            face_points,
            face_indices,
            groups,
          ],
        )

        wp.launch(
          kernel=_make_sides_2d_elements,
          dim=(d.nworld, nshell),
          inputs=[
            d.flexvert_xpos,
            vert_norm,
            flex_shell,
            shell_adr,
            flex_faceadr[f] + (2 * nelem),
            mjm.flex_radius[f],
            nfaces,
            face_points,
            face_indices,
            groups,
          ],
        )
      elif dim == 3:
        wp.launch(
          kernel=_make_faces_3d_shells,
          dim=(d.nworld, nshell),
          inputs=[
            d.flexvert_xpos,
            flex_shell,
            shell_adr,
            flex_faceadr[f],
            nfaces,
            face_points,
            face_indices,
            groups,
          ],
        )

    flex_mesh = wp.Mesh(
      points=face_points,
      indices=face_indices,
      groups=groups,
      bvh_constructor="sah",
    )

    group_roots = wp.zeros(d.nworld, dtype=int)
    wp.launch(
      kernel=bvh.compute_bvh_group_roots,
      dim=d.nworld,
      inputs=[flex_mesh.id, group_roots],
    )

    return (
      flex_mesh,
      face_points,
      groups,
      group_roots,
      flex_elem,
      flex_shell,
      flex_faceadr,
      nfaces,
    )
