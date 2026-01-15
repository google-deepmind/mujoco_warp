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


def _camera_frustum_bounds(
  mjm: mujoco.MjModel,
  cam_id: int,
  img_w: int,
  img_h: int,
  znear: float,
) -> tuple[float, float, float, float, bool]:
  """Replicate MuJoCo's frustum computation to derive near-plane bounds."""
  orthographic = mjm.cam_projection[cam_id] == mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
  if orthographic:
    half_height = mjm.cam_fovy[cam_id] * 0.5
    aspect = img_w / img_h
    half_width = half_height * aspect
    return (-half_width, half_width, half_height, -half_height, True)

  sensorsize = mjm.cam_sensorsize[cam_id]
  has_intrinsics = sensorsize[1] != 0.0
  if has_intrinsics:
    fx, fy, cx, cy = mjm.cam_intrinsic[cam_id]
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

  fovy_rad = np.deg2rad(float(mjm.cam_fovy[cam_id]))
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
  mesh_texcoord_num: wp.array(dtype=int)
  tex_adr: wp.array(dtype=int)
  tex_data: wp.array(dtype=wp.uint32)
  tex_height: wp.array(dtype=int)
  tex_width: wp.array(dtype=int)
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
      mesh, half = _build_mesh_bvh(mjm, i)
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
      hmesh, hhalf = _build_hfield_mesh(mjm, hid)
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
      ) = _make_flex_mesh(mjm, m, d)

      self.flex_registry[fmesh.id] = fmesh
      self.flex_bvh_id = fmesh.id
      self.flex_face_point = face_point
      self.flex_group = flex_groups
      self.flex_group_root = flex_group_roots
      self.flex_dim = mjm.flex_dim
      self.flex_elemnum = mjm.flex_elemnum
      self.flex_elemdataadr = mjm.flex_elemdataadr
      self.flex_shell = flex_shell
      self.flex_shellnum = mjm.flex_shellnum
      self.flex_shelldataadr = mjm.flex_shelldataadr
      self.flex_faceadr = flex_faceadr
      self.flex_nface = flex_nface
      self.flex_radius = mjm.flex_radius
      self.flex_render_smooth = flex_render_smooth

    tex_data_packed, tex_adr_packed = _create_packed_texture_data(mjm)

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

    if render_rgb is not None:
      if isinstance(render_rgb, bool):
        render_rgb = [render_rgb] * ncam
      assert len(render_rgb) == ncam, (
        f"Render RGB must be provided for all active cameras (got {len(render_rgb)}, expected {ncam})"
      )
    else:
      render_rgb = [mjm.cam_output[i] & mujoco.mjtCamOutBit.mjCAMOUT_RGB for i in active_cam_indices]

    if render_depth is not None:
      if isinstance(render_depth, bool):
        render_depth = [render_depth] * ncam
      assert len(render_depth) == ncam, (
        f"Render depth must be provided for all active cameras (got {len(render_depth)}, expected {ncam})"
      )
    else:
      render_depth = [mjm.cam_output[i] & mujoco.mjtCamOutBit.mjCAMOUT_DEPTH for i in active_cam_indices]

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
    self.ray = wp.zeros(int(total), dtype=wp.vec3)

    offset = 0
    model_znear = mjm.vis.map.znear * mjm.stat.extent
    for idx, cam_id in enumerate(active_cam_indices):
      img_w = cam_res[idx][0]
      img_h = cam_res[idx][1]
      left, right, top, bottom, is_ortho = _camera_frustum_bounds(
        mjm,
        cam_id,
        img_w,
        img_h,
        model_znear,
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
          model_znear,
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
    self.mesh_texcoord_num = wp.array(mjm.mesh_texcoordnum, dtype=int)
    self.tex_adr = tex_adr_packed
    self.tex_data = tex_data_packed
    self.tex_height = wp.array(mjm.tex_height, dtype=int)
    self.tex_width = wp.array(mjm.tex_width, dtype=int)
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
    bvh.build_warp_bvh(m, d, self)


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
    # In:
    tex_data_uint8: wp.array(dtype=wp.uint8),
    # Out:
    tex_data_packed_out: wp.array(dtype=wp.uint32),
  ):
    """Convert uint8 texture data to packed uint32 format for efficient sampling."""
    tid = wp.tid()

    src_idx = tid * nchannel

    r = tex_data_uint8[src_idx + 0] if nchannel > 0 else wp.uint8(0)
    g = tex_data_uint8[src_idx + 1] if nchannel > 1 else wp.uint8(0)
    b = tex_data_uint8[src_idx + 2] if nchannel > 2 else wp.uint8(0)
    a = wp.uint8(255)

    packed = (wp.uint32(a) << wp.uint32(24)) | (wp.uint32(r) << wp.uint32(16)) | (wp.uint32(g) << wp.uint32(8)) | wp.uint32(b)
    tex_data_packed_out[tid] = packed

  wp.launch(
    convert_texture_to_packed,
    dim=int(total_size),
    inputs=[wp.array(mjm.tex_data, dtype=wp.uint8)],
    outputs=[tex_data_packed],
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
  """Greedy meshing for heightfield optimization.

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
  nr = mjm.hfield_nrow[hfieldid]
  nc = mjm.hfield_ncol[hfieldid]
  sz = np.asarray(mjm.hfield_size[hfieldid], dtype=np.float32)

  adr = mjm.hfield_adr[hfieldid]
  # Use host data for optimization
  data = mjm.hfield_data[adr : adr + nr * nc].reshape((nr, nc))

  width = 0.5 * max(nc - 1, 1)
  height = 0.5 * max(nr - 1, 1)

  points, indices = _optimize_hfield_mesh(
    data,
    nr,
    nc,
    sz[0],
    sz[1],
    sz[2],
    width,
    height,
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
  # Model:
  flex_elem: wp.array(dtype=int),
  # Data in:
  flexvert_xpos_in: wp.array2d(dtype=wp.vec3),
  # In:
  flexvert_norm_in: wp.array2d(dtype=wp.vec3),
  elem_adr: int,
  face_offset: int,
  radius: float,
  nfaces: int,
  # Out:
  face_point_out: wp.array(dtype=wp.vec3),
  face_index_out: wp.array(dtype=int),
  group_out: wp.array(dtype=int),
):
  """Create faces from 2D flex elements (triangles).

  Two faces (top/bottom) per element, seperated by the radius of the flex element.
  """
  worldid, elemid = wp.tid()

  base = elem_adr + elemid * 3
  i0 = flex_elem[base + 0]
  i1 = flex_elem[base + 1]
  i2 = flex_elem[base + 2]

  v0 = flexvert_xpos_in[worldid, i0]
  v1 = flexvert_xpos_in[worldid, i1]
  v2 = flexvert_xpos_in[worldid, i2]

  n0 = flexvert_norm_in[worldid, i0]
  n1 = flexvert_norm_in[worldid, i1]
  n2 = flexvert_norm_in[worldid, i2]

  p0_pos = v0 + radius * n0
  p1_pos = v1 + radius * n1
  p2_pos = v2 + radius * n2

  p0_neg = v0 - radius * n0
  p1_neg = v1 - radius * n1
  p2_neg = v2 - radius * n2

  world_face_offset = worldid * nfaces

  # First face (top): i0, i1, i2
  face_id0 = world_face_offset + face_offset + 2 * elemid
  base0 = face_id0 * 3
  face_point_out[base0 + 0] = p0_pos
  face_point_out[base0 + 1] = p1_pos
  face_point_out[base0 + 2] = p2_pos

  face_index_out[base0 + 0] = base0 + 0
  face_index_out[base0 + 1] = base0 + 1
  face_index_out[base0 + 2] = base0 + 2

  group_out[face_id0] = worldid

  # Second face (bottom): i0, i2, i1 (opposite winding)
  face_id1 = world_face_offset + face_offset + 2 * elemid + 1
  base1 = face_id1 * 3
  face_point_out[base1 + 0] = p0_neg
  face_point_out[base1 + 1] = p1_neg
  face_point_out[base1 + 2] = p2_neg

  face_index_out[base1 + 0] = base1 + 0
  face_index_out[base1 + 1] = base1 + 2
  face_index_out[base1 + 2] = base1 + 1

  group_out[face_id1] = worldid


@wp.kernel
def _make_sides_2d_elements(
  # Data in:
  flexvert_xpos_in: wp.array2d(dtype=wp.vec3),
  # In:
  flexvert_norm_in: wp.array2d(dtype=wp.vec3),
  flex_shell_in: wp.array(dtype=int),
  shell_adr: int,
  face_offset: int,
  radius: float,
  nface: int,
  # Out:
  face_point_out: wp.array(dtype=wp.vec3),
  face_index_out: wp.array(dtype=int),
  group_out: wp.array(dtype=int),
):
  """Create side faces from 2D flex shell fragments.

  For each shell fragment (edge i0 -> i1), we emit two triangles:
    - one using +radius
    - one using -radius (i0/i1 swapped)
  """
  worldid, shellid = wp.tid()

  base = shell_adr + 2 * shellid
  i0 = flex_shell_in[base + 0]
  i1 = flex_shell_in[base + 1]

  v0 = flexvert_xpos_in[worldid, i0]
  v1 = flexvert_xpos_in[worldid, i1]

  n0 = flexvert_norm_in[worldid, i0]
  n1 = flexvert_norm_in[worldid, i1]

  neg_radius = -radius

  # First side i0, i1 with +radius
  face_id0 = worldid * nface + face_offset + 2 * shellid
  base0 = face_id0 * 3
  face_point_out[base0 + 0] = v0 + n0 * radius
  face_point_out[base0 + 1] = v1 + n1 * neg_radius
  face_point_out[base0 + 2] = v1 + n1 * radius
  face_index_out[base0 + 0] = base0 + 0
  face_index_out[base0 + 1] = base0 + 1
  face_index_out[base0 + 2] = base0 + 2

  # Second side i1, i0 with -radius
  face_id1 = worldid * nface + face_offset + 2 * shellid + 1
  base1 = face_id1 * 3
  face_point_out[base1 + 0] = v1 + n1 * neg_radius
  face_point_out[base1 + 1] = v0 + n0 * neg_radius
  face_point_out[base1 + 2] = v0 + n0 * radius
  face_index_out[base1 + 0] = base1 + 0
  face_index_out[base1 + 1] = base1 + 1
  face_index_out[base1 + 2] = base1 + 2

  group_out[face_id0] = worldid
  group_out[face_id1] = worldid


@wp.kernel
def _make_faces_3d_shells(
  # Data in:
  flexvert_xpos_in: wp.array2d(dtype=wp.vec3),
  # In:
  flex_shell_in: wp.array(dtype=int),
  shell_adr: int,
  face_offset: int,
  nface: int,
  # Out:
  face_point_out: wp.array(dtype=wp.vec3),
  face_index_out: wp.array(dtype=int),
  group_out: wp.array(dtype=int),
):
  """Create faces from 3D flex shell fragments (triangles).

  Each shell fragment contributes a single triangle whose vertices are taken
  directly from the flex vertex positions (one-sided surface).
  """
  worldid, shellid = wp.tid()

  base = shell_adr + shellid * 3
  i0 = flex_shell_in[base + 0]
  i1 = flex_shell_in[base + 1]
  i2 = flex_shell_in[base + 2]

  face_id = worldid * nface + face_offset + shellid
  base = face_id * 3

  v0 = flexvert_xpos_in[worldid, i0]
  v1 = flexvert_xpos_in[worldid, i1]
  v2 = flexvert_xpos_in[worldid, i2]

  face_point_out[base + 0] = v0
  face_point_out[base + 1] = v1
  face_point_out[base + 2] = v2

  face_index_out[base + 0] = base + 0
  face_index_out[base + 1] = base + 1
  face_index_out[base + 2] = base + 2

  group_out[face_id] = worldid


def _make_flex_mesh(mjm: mujoco.MjModel, m: Model, d: Data):
  """Create a Warp Mesh for flex meshes.

  We create a single Warp Mesh (single BVH) for all flex objects across all worlds

  This implements the core of MuJoCo's flex rendering path for the 2D flex case by:
    * gathering vertex positions for this flex
    * building triangle faces for both sides of the cloth, offset by `radius`
      along the element normal so the cloth has thickness
    * returning a Warp mesh plus an approximate half-extent for BVH bounds
  """
  if (mjm.flex_dim == 1).any():
    raise ValueError("1D Flex objects are not currently supported.")

  nflex = mjm.nflex

  flex_faceadr = [0]
  for f in range(nflex):
    if mjm.flex_dim[f] == 2:
      flex_faceadr.append(flex_faceadr[-1] + 2 * mjm.flex_elemnum[f] + 2 * mjm.flex_shellnum[f])
    elif mjm.flex_dim[f] == 3:
      flex_faceadr.append(flex_faceadr[-1] + mjm.flex_shellnum[f])

  nface = int(flex_faceadr[-1])
  flex_faceadr = flex_faceadr[:-1]

  face_point = wp.zeros(nface * 3 * d.nworld, dtype=wp.vec3)
  face_index = wp.zeros(nface * 3 * d.nworld, dtype=wp.int32)
  group = wp.zeros(nface * d.nworld, dtype=int)

  flexvert_norm = wp.zeros(d.flexvert_xpos.shape, dtype=wp.vec3)
  flex_shell = wp.array(mjm.flex_shell, dtype=int)

  wp.launch(
    kernel=bvh.accumulate_flex_vertex_normals,
    dim=(d.nworld, m.nflexelem),
    inputs=[m.flex_elem, d.flexvert_xpos],
    outputs=[flexvert_norm],
  )

  wp.launch(
    kernel=bvh.normalize_vertex_normals,
    dim=(d.nworld, m.nflexvert),
    inputs=[flexvert_norm],
  )

  for f in range(nflex):
    dim = mjm.flex_dim[f]
    elem_adr = mjm.flex_elemdataadr[f]
    nelem = mjm.flex_elemnum[f]
    shell_adr = mjm.flex_shelldataadr[f]
    nshell = mjm.flex_shellnum[f]

    if dim == 2:
      wp.launch(
        kernel=_make_face_2d_elements,
        dim=(d.nworld, nelem),
        inputs=[
          m.flex_elem,
          d.flexvert_xpos,
          flexvert_norm,
          elem_adr,
          flex_faceadr[f],
          mjm.flex_radius[f],
          nface,
        ],
        outputs=[face_point, face_index, group],
      )

      wp.launch(
        kernel=_make_sides_2d_elements,
        dim=(d.nworld, nshell),
        inputs=[
          d.flexvert_xpos,
          flexvert_norm,
          flex_shell,
          shell_adr,
          flex_faceadr[f] + 2 * nelem,
          mjm.flex_radius[f],
          nface,
        ],
        outputs=[face_point, face_index, group],
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
          nface,
        ],
        outputs=[face_point, face_index, group],
      )

  flex_mesh = wp.Mesh(
    points=face_point,
    indices=face_index,
    groups=group,
    bvh_constructor="sah",
  )

  group_root = wp.zeros(d.nworld, dtype=int)
  wp.launch(
    kernel=bvh.compute_bvh_group_roots,
    dim=d.nworld,
    inputs=[flex_mesh.id],
    outputs=[group_root],
  )

  return (
    flex_mesh,
    face_point,
    group,
    group_root,
    flex_shell,
    flex_faceadr,
    nface,
  )
