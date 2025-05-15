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


import math
from typing import Any

import warp as wp

from .types import Data
from .types import GeomType
from .types import Model
from .warp_util import event_scope


@wp.func
def get_hfield_overlap_range(
  # Model:
  geom_dataid: wp.array(dtype=int),
  geom_rbound: wp.array2d(dtype=float),
  geom_margin: wp.array2d(dtype=float),
  hfield_nrow: wp.array(dtype=int),
  hfield_ncol: wp.array(dtype=int),
  hfield_size: wp.array(dtype=wp.vec4),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  hfield_geom: int,
  other_geom: int,
  worldid: int,
):
  """Returns min/max grid coordinates of height field cells overlapped by other geom's bounds.

  Args:
      geom_dataid: Array of geometry data IDs
      geom_rbound: Array of geometry bounding radii
      geom_margin: Array of geometry margins
      hfield_nrow: Array of heightfield rows
      hfield_ncol: Array of heightfield columns
      hfield_size: Array of heightfield sizes
      geom_xpos_in: Array of geometry positions
      geom_xmat_in: Array of geometry orientation matrices
      hfield_geom: Index of the height field geometry
      other_geom: Index of the other geometry
      worldid: Current world index

  Returns:
      min_i, min_j, max_i, max_j: Grid coordinate bounds
  """
  # Get height field dimensions
  dataid = geom_dataid[hfield_geom]
  nrow = hfield_nrow[dataid]
  ncol = hfield_ncol[dataid]
  size = hfield_size[dataid]  # (x, y, z_top, z_bottom)

  # Get positions and transforms
  hf_pos = geom_xpos_in[worldid, hfield_geom]
  hf_mat = geom_xmat_in[worldid, hfield_geom]
  other_pos = geom_xpos_in[worldid, other_geom]

  # Transform other_pos to height field local space
  rel_pos = other_pos - hf_pos
  local_x = wp.dot(wp.vec3(hf_mat[0, 0], hf_mat[1, 0], hf_mat[2, 0]), rel_pos)
  local_y = wp.dot(wp.vec3(hf_mat[0, 1], hf_mat[1, 1], hf_mat[2, 1]), rel_pos)
  local_z = wp.dot(wp.vec3(hf_mat[0, 2], hf_mat[1, 2], hf_mat[2, 2]), rel_pos)
  local_pos = wp.vec3(local_x, local_y, local_z)

  # Get bounding radius of other geometry (including margin)
  other_rbound = geom_rbound[worldid, other_geom]
  other_margin = geom_margin[worldid, other_geom]
  bound_radius = other_rbound + other_margin

  # Calculate grid resolution
  x_scale = 2.0 * size[0] / wp.float32(ncol - 1)
  y_scale = 2.0 * size[1] / wp.float32(nrow - 1)

  # Calculate min/max grid coordinates that could contain the object
  min_i = wp.max(0, wp.int32((local_pos[0] - bound_radius + size[0]) / x_scale))
  max_i = wp.min(ncol - 2, wp.int32((local_pos[0] + bound_radius + size[0]) / x_scale) + 1)
  min_j = wp.max(0, wp.int32((local_pos[1] - bound_radius + size[1]) / y_scale))
  max_j = wp.min(nrow - 2, wp.int32((local_pos[1] + bound_radius + size[1]) / y_scale) + 1)

  return min_i, min_j, max_i, max_j


@wp.func
def get_hfield_triangle_prism(
  # Model:
  geom_dataid: wp.array(dtype=int),
  hfield_adr: wp.array(dtype=int),
  hfield_nrow: wp.array(dtype=int),
  hfield_ncol: wp.array(dtype=int),
  hfield_size: wp.array(dtype=wp.vec4),
  hfield_data: wp.array(dtype=float),
  # In:
  hfieldid: int,
  hftri_index: int,
) -> wp.mat33:
  """Returns the vertices of a triangular prism for a heightfield triangle.

  Args:
      geom_dataid: Array of geometry data IDs
      hfield_adr: Array of heightfield addresses
      hfield_nrow: Array of heightfield rows
      hfield_ncol: Array of heightfield columns
      hfield_size: Array of heightfield sizes
      hfield_data: Array of heightfield data
      hfieldid: Index of the height field geometry
      hftri_index: Index of the triangle in the heightfield

  Returns:
      3x3 matrix containing the vertices of the triangular prism
  """
  # See https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-hfield

  # Get heightfield dimensions
  dataid = geom_dataid[hfieldid]
  if dataid < 0 or hftri_index < 0:
    return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  nrow = hfield_nrow[dataid]
  ncol = hfield_ncol[dataid]
  size = hfield_size[dataid]  # (x, y, z_top, z_bottom)

  # Calculate which triangle in the grid
  row = (hftri_index // 2) // (ncol - 1)
  col = (hftri_index // 2) % (ncol - 1)

  # Calculate vertices in 2D grid
  x_scale = 2.0 * size[0] / wp.float32(ncol - 1)
  y_scale = 2.0 * size[1] / wp.float32(nrow - 1)

  # Grid coordinates (i, j) for triangle corners
  i0 = col
  j0 = row
  i1 = i0 + 1
  j1 = j0 + 1

  # Convert grid coordinates to local space x, y coordinates
  x0 = wp.float32(i0) * x_scale - size[0]
  y0 = wp.float32(j0) * y_scale - size[1]
  x1 = wp.float32(i1) * x_scale - size[0]
  y1 = wp.float32(j1) * y_scale - size[1]

  # Get height values at corners from hfield_data
  base_addr = hfield_adr[dataid]
  z00 = hfield_data[base_addr + j0 * ncol + i0]
  z01 = hfield_data[base_addr + j1 * ncol + i0]
  z10 = hfield_data[base_addr + j0 * ncol + i1]
  z11 = hfield_data[base_addr + j1 * ncol + i1]

  # Scale heights from range [0, 1] to [0, z_top]
  z_top = size[2]
  z00 = z00 * z_top
  z01 = z01 * z_top
  z10 = z10 * z_top
  z11 = z11 * z_top

  # Set bottom z-value
  z_bottom = -size[3]

  # Compress 6 prism vertices into 3x3 matrix
  # See get_hfield_prism_vertex() for the details
  return wp.mat33(
    x0,
    y0,
    z00,
    x1,
    y1,
    z11,
    wp.where(hftri_index % 2, 1.0, 0.0),
    wp.where(hftri_index % 2, z10, z01),
    z_bottom,
  )


@wp.func
def get_hfield_prism_vertex(prism: wp.mat33, vert_index: int) -> wp.vec3:
  """Extracts vertices from a compressed triangular prism representation.

  The compression scheme stores a 6-vertex triangular prism using a 3x3 matrix:
  - prism[0] = First vertex (x,y,z) - corner (i,j)
  - prism[1] = Second vertex (x,y,z) - corner (i+1,j+1)
  - prism[2,0] = Triangle type flag: 0 for even triangle (using corner (i,j+1)),
                 non-zero for odd triangle (using corner (i+1,j))
  - prism[2,1] = Z-coordinate of the third vertex
  - prism[2,2] = Z-coordinate used for all bottom vertices (common z)

  In this way, we can reconstruct all 6 vertices of the prism by reusing
  coordinates from the stored vertices.

  Args:
      prism: 3x3 compressed representation of a triangular prism
      vert_index: Index of vertex to extract (0-5)

  Returns:
      The 3D coordinates of the requested vertex
  """
  if vert_index == 0 or vert_index == 1:
    return prism[vert_index]  # First two vertices stored directly

  if vert_index == 2:  # Third vertex
    if prism[2][0] == 0:  # Even triangle (i,j+1)
      return wp.vec3(prism[0][0], prism[1][1], prism[2][1])
    else:  # Odd triangle (i+1,j)
      return wp.vec3(prism[1][0], prism[0][1], prism[2][1])

  if vert_index == 3 or vert_index == 4:  # Bottom vertices below 0 and 1
    return wp.vec3(prism[vert_index - 3][0], prism[vert_index - 3][1], prism[2][2])

  if vert_index == 5:  # Bottom vertex below 2
    if prism[2][0] == 0:  # Even triangle
      return wp.vec3(prism[0][0], prism[1][1], prism[2][2])
    else:  # Odd triangle
      return wp.vec3(prism[1][0], prism[0][1], prism[2][2])


@wp.kernel
def _hfield_midphase(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_rbound: wp.array2d(dtype=float),
  geom_margin: wp.array2d(dtype=float),
  hfield_nrow: wp.array(dtype=int),
  hfield_ncol: wp.array(dtype=int),
  hfield_size: wp.array(dtype=wp.vec4),
  # Data in:
  nconmax_in: int,
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  collision_pair_in: wp.array(dtype=wp.vec2i),
  collision_hftri_index_in: wp.array(dtype=int),
  collision_pairid_in: wp.array(dtype=int),
  collision_worldid_in: wp.array(dtype=int),
  ncollision_in: wp.array(dtype=int),
  # Data out:
  collision_pair_out: wp.array(dtype=wp.vec2i),
  collision_hftri_index_out: wp.array(dtype=int),
  collision_pairid_out: wp.array(dtype=int),
  collision_worldid_out: wp.array(dtype=int),
  ncollision_out: wp.array(dtype=int),
):
  """Midphase collision detection for heightfield triangles with other geoms.

  This kernel processes collision pairs where one geom is a heightfield (identified by
  collision_hftri_index_in[pairid] == -1) and expands them into multiple collision pairs,
  one for each potentially colliding triangle.

  Args:
      geom_type: Array of geometry types
      geom_dataid: Array of geometry data IDs
      geom_rbound: Array of geometry bounding radii
      geom_margin: Array of geometry margins
      hfield_nrow: Array of heightfield rows
      hfield_ncol: Array of heightfield columns
      hfield_size: Array of heightfield sizes
      nconmax_in: Max number of collisions
      geom_xpos_in: Array of geometry positions
      geom_xmat_in: Array of geometry orientation matrices
      collision_pair_in: Array of collision pairs
      collision_hftri_index_in: Array of heightfield triangle indices (-1 for heightfield pairs)
      collision_pairid_in: Array of collision pair IDs
      collision_worldid_in: Array of collision world IDs
      ncollision_in: Number of collisions

      collision_pair_out: Output array of collision pairs
      collision_hftri_index_out: Output array of heightfield triangle indices
      collision_pairid_out: Output array of collision pair IDs
      collision_worldid_out: Output array of collision world IDs
      ncollision_out: Output counter for number of collisions
  """
  pairid = wp.tid()

  # Only process pairs that are marked for heightfield collision (-1)
  # The buffer is cleared at the start of each frame in collision_driver.py
  if collision_hftri_index_in[pairid] != -1:
    return

  # Get the collision pair info
  pair = collision_pair_in[pairid]
  worldid = collision_worldid_in[pairid]
  pair_id = collision_pairid_in[pairid]

  # Identify which geom is the heightfield
  g1 = pair[0]
  g2 = pair[1]

  hfield_geom = g1
  other_geom = g2

  # If the first geom is not a heightfield, swap them
  # In theory, shouldn't happen as _add_geom_pair already
  # sorted the pair
  if geom_type[g1] != int(GeomType.HFIELD.value):
    hfield_geom = g2
    other_geom = g1

  # Get min/max grid coordinates for overlap region
  min_i, min_j, max_i, max_j = get_hfield_overlap_range(
    geom_dataid,
    geom_rbound,
    geom_margin,
    hfield_nrow,
    hfield_ncol,
    hfield_size,
    geom_xpos_in,
    geom_xmat_in,
    hfield_geom,
    other_geom,
    worldid,
  )

  # Get hfield dimensions for triangle index calculation
  dataid = geom_dataid[hfield_geom]
  ncol = hfield_ncol[dataid]

  # Loop through grid cells and add pairs for all triangles
  for j in range(min_j, max_j + 1):
    for i in range(min_i, max_i + 1):
      # Each grid cell contains two triangles
      base_idx = ((j * (ncol - 1)) + i) * 2

      # Add both triangles from this cell
      for t in range(2):
        if i == 0 and j == 0 and t == 0:
          # We reuse the initial pair for the 1st triangle
          new_pairid = pairid
        else:
          # For the rest we create a new pair
          new_pairid = wp.atomic_add(ncollision_out, 0, 1)

        if new_pairid >= nconmax_in:
          return

        collision_pair_out[new_pairid] = pair
        collision_hftri_index_out[new_pairid] = base_idx + t
        collision_pairid_out[new_pairid] = pair_id
        collision_worldid_out[new_pairid] = worldid


def hfield_midphase(m: Model, d: Data):
  """Midphase collision detection for heightfield triangles with other geoms.

  This function processes collision pairs from the broadphase where one geom is a heightfield
  and expands them into multiple collision pairs, one for each potentially colliding triangle.
  The function directly writes to the same collision buffers used by _add_geom_pair.

  Args:
      m: Model containing geometry and heightfield data
         - geom_type: Array of geometry types
         - geom_dataid: Array of geometry data IDs
         - hfield_nrow: Array of heightfield rows
         - hfield_ncol: Array of heightfield columns
         - hfield_size: Array of heightfield sizes
         - geom_rbound: Array of geometry bounding radii
         - geom_margin: Array of geometry margins
      d: Data containing current state and collision information
         - nconmax: Maximum number of contacts
         - geom_xpos: Array of geometry positions
         - geom_xmat: Array of geometry orientation matrices
         - collision_pair: Array of collision pairs
         - collision_hftri_index: Array of heightfield triangle indices
         - collision_pairid: Array of collision pair IDs
         - collision_worldid: Array of collision world IDs
         - ncollision: Number of collisions
  """
  # Launch the midphase kernel to expand height field collision pairs
  # We write directly to the same buffers that _add_geom_pair writes to
  wp.launch(
    kernel=_hfield_midphase,
    dim=d.nconmax,  # Launch enough threads to process all potential pairs
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_rbound,
      m.geom_margin,
      m.hfield_nrow,
      m.hfield_ncol,
      m.hfield_size,
      d.nconmax,
      d.geom_xpos,
      d.geom_xmat,
      d.collision_pair,
      d.collision_hftri_index,
      d.collision_pairid,
      d.collision_worldid,
      d.ncollision,
    ],
    outputs=[
      d.collision_pair,
      d.collision_hftri_index,
      d.collision_pairid,
      d.collision_worldid,
      d.ncollision,
    ],
  )
