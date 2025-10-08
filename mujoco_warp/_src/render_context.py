import dataclasses

import warp as wp
import mujoco
import threading

from .types import Model
from .types import Data
from .types import GeomType
from . import bvh
from . import render


_RENDER_CONTEXT_BUFFERS = {}


def create_render_context(
  mjm: mujoco.MjModel,
  m: Model,
  d: Data,
  nworld: int,
  width: int,
  height: int,
  use_textures: bool,
  use_shadows: bool,
  fov_rad: float,
  render_rgb: bool,
  render_depth: bool,
  enabled_geom_groups = [0, 1, 2],
):
  rc = RenderContext(
    mjm,
    m,
    d,
    nworld,
    width,
    height,
    use_textures,
    use_shadows,
    fov_rad,
    render_rgb,
    render_depth,
    enabled_geom_groups,
  )
  return rc


def create_render_context_in_registry(
  mjm: mujoco.MjModel,
  m: Model,
  d: Data,
  nworld: int,
  width: int,
  height: int,
  use_textures: bool,
  use_shadows: bool,
  fov_rad: float,
  render_rgb: bool,
  render_depth: bool,
  enabled_geom_groups = [0, 1, 2],
):
  rc = RenderContext(
    mjm,
    m,
    d,
    nworld,
    width,
    height,
    use_textures,
    use_shadows,
    fov_rad,
    render_rgb,
    render_depth,
    enabled_geom_groups,
  )
  with threading.Lock():
    key = len(_RENDER_CONTEXT_BUFFERS) + 1
    _RENDER_CONTEXT_BUFFERS[key] = rc
  return RenderContextRegistry(key)


def render_registry(m: Model, d: Data, rc_id: int):
  rc = _RENDER_CONTEXT_BUFFERS[rc_id]
  render.render(m, d, rc)


@wp.kernel
def convert_texture_to_packed(
  size: int,
  nchannel: int,
  tex_data_uint8: wp.array(dtype=wp.uint8),
  tex_data_packed: wp.array(dtype=wp.uint32),
):
  """
  Convert uint8 texture data to packed uint32 format for efficient sampling.
  """
  tid = wp.tid()
  if tid >= size:
    return

  src_idx = tid * nchannel

  r = tex_data_uint8[src_idx + 0] if nchannel > 0 else wp.uint8(0)
  g = tex_data_uint8[src_idx + 1] if nchannel > 1 else wp.uint8(0)
  b = tex_data_uint8[src_idx + 2] if nchannel > 2 else wp.uint8(0)
  a = wp.uint8(255)  # Always use full alpha

  packed = (wp.uint32(a) << wp.uint32(24)) | (wp.uint32(r) << wp.uint32(16)) | (wp.uint32(g) << wp.uint32(8)) | wp.uint32(b)
  tex_data_packed[tid] = packed


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

  wp.launch(
    convert_texture_to_packed,
    dim=(total_size,),
    inputs=[total_size, mjm.tex_nchannel[0], wp.array(mjm.tex_data, dtype=wp.uint8), tex_data_packed]
  )

  return tex_data_packed, wp.array(tex_adr_packed, dtype=int)


@dataclasses.dataclass
class RenderContextRegistry:
  key: int

  def __init__(self, key: int):
    self.key = key
  
  def __del__(self):
    with threading.Lock():
      del _RENDER_CONTEXT_BUFFERS[self.key]


@dataclasses.dataclass
class RenderContext:
  nworld: int
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
  groups: wp.array(dtype=wp.int32)
  group_roots: wp.array(dtype=wp.int32)
  pixels: wp.array3d(dtype=wp.uint32)
  depth: wp.array3d(dtype=wp.float32)

  def __init__(
    self,
    mjm: mujoco.MjModel,
    m: Model,
    d: Data,
    nworld,
    width,
    height,
    use_textures,
    use_shadows,
    fov_rad,
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
        bvh_constructor="sah"
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
    self.nworld=nworld
    self.ncam=mjm.ncam
    self.width=width
    self.height=height
    self.use_textures=use_textures
    self.use_shadows=use_shadows
    self.fov_rad=fov_rad
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
    self.tex_height=wp.array(mjm.tex_height, dtype=int)
    self.tex_width=wp.array(mjm.tex_width, dtype=int)
    self.lowers = wp.zeros((nworld * bvh_ngeom,), dtype=wp.vec3)
    self.uppers = wp.zeros((nworld * bvh_ngeom,), dtype=wp.vec3)
    self.groups = wp.zeros((nworld * bvh_ngeom,), dtype=wp.int32)
    self.group_roots = wp.zeros((nworld,), dtype=wp.int32)
    self.pixels = wp.zeros((nworld, mjm.ncam, width * height), dtype=wp.uint32)
    self.depth = wp.zeros((nworld, mjm.ncam, width * height), dtype=wp.float32)
    
    self.bvh = None
    self.bvh_id = None
    bvh.build_warp_bvh(m, d, self)