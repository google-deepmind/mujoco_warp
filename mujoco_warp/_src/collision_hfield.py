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

from .types import Model, Data, GeomType
from .warp_util import event_scope


class HFPrism(wp.types.matrix(shape=(6, 3), dtype=wp.float32)):
  pass


@wp.func
def get_hfield_overlap_range(m: Model, d: Data, hfield_geom: int, other_geom: int, worldid: int):
    """Returns min/max grid coordinates of height field cells overlapped by other geom's bounds.
    
    Args:
        m: Model containing geometry data
        d: Data containing current state 
        hfield_geom: Index of the height field geometry
        other_geom: Index of the other geometry
        worldid: Current world index
        
    Returns:
        min_i, min_j, max_i, max_j: Grid coordinate bounds
    """
    # Get height field dimensions
    dataid = m.geom_dataid[hfield_geom]
    nrow = m.hfield_nrow[dataid]
    ncol = m.hfield_ncol[dataid]
    size = m.hfield_size[dataid]  # (x, y, z_top, z_bottom)
    
    # Get positions and transforms
    hf_pos = d.geom_xpos[worldid, hfield_geom]
    hf_mat = d.geom_xmat[worldid, hfield_geom]
    other_pos = d.geom_xpos[worldid, other_geom]
    
    # Transform other_pos to height field local space
    rel_pos = other_pos - hf_pos
    local_x = wp.dot(wp.vec3(hf_mat[0, 0], hf_mat[1, 0], hf_mat[2, 0]), rel_pos)
    local_y = wp.dot(wp.vec3(hf_mat[0, 1], hf_mat[1, 1], hf_mat[2, 1]), rel_pos)
    local_z = wp.dot(wp.vec3(hf_mat[0, 2], hf_mat[1, 2], hf_mat[2, 2]), rel_pos)
    local_pos = wp.vec3(local_x, local_y, local_z)
    
    # Get bounding radius of other geometry (including margin)
    other_rbound = m.geom_rbound[other_geom]
    other_margin = m.geom_margin[other_geom]
    bound_radius = other_rbound + other_margin
    
    # Calculate grid coordinates that bound the other geometry
    x_scale = 2.0 * size[0] / wp.float32(ncol - 1)
    y_scale = 2.0 * size[1] / wp.float32(nrow - 1)
    
    # Calculate min/max grid coordinates that could contain the object
    min_i = wp.max(0, wp.int32((local_pos[0] - bound_radius + size[0]) / x_scale))
    max_i = wp.min(ncol - 2, wp.int32((local_pos[0] + bound_radius + size[0]) / x_scale) + 1)
    min_j = wp.max(0, wp.int32((local_pos[1] - bound_radius + size[1]) / y_scale))
    max_j = wp.min(nrow - 2, wp.int32((local_pos[1] + bound_radius + size[1]) / y_scale) + 1)
    
    return min_i, min_j, max_i, max_j


@wp.func
def get_hfield_triangle_prism(m: Model, hfield_geom: int, tri_index: int) -> HFPrism:
    """Returns the vertices of a triangular prism for a heightfield triangle.
    
    Args:
        m: Model containing geometry data
        hfield_geom: Index of the height field geometry
        tri_index: Index of the triangle in the heightfield
        
    Returns:
        6x3 matrix containing the vertices of the triangular prism
    """
    # Get heightfield dimensions
    dataid = m.geom_dataid[hfield_geom]
    if dataid < 0 or tri_index < 0:
        return HFPrism()

    nrow = m.hfield_nrow[dataid]
    ncol = m.hfield_ncol[dataid]
    size = m.hfield_size[dataid]  # (x, y, z_top, z_bottom)
 
    # Calculate which triangle in the grid
    row = (tri_index // 2) // (ncol - 1)
    col = (tri_index // 2) % (ncol - 1)
    
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
    base_addr = m.hfield_adr[dataid]
    z00 = m.hfield_data[base_addr + j0 * ncol + i0]
    z01 = m.hfield_data[base_addr + j1 * ncol + i0]
    z10 = m.hfield_data[base_addr + j0 * ncol + i1]
    z11 = m.hfield_data[base_addr + j1 * ncol + i1]
    
    # Scale heights to range [0, 1] and then to [z_bottom, z_top]
    z_range = size[2]
    z00 = z00 * z_range
    z01 = z01 * z_range
    z10 = z10 * z_range
    z11 = z11 * z_range
    
    # Set bottom z-value
    z_bottom = -size[3]
    
    # Determine triangle vertices based on whether it's the first or second triangle
    if (tri_index % 2) == 0:
        # First triangle: (i0 j0), (i0 j1), (i1 j1)
        return HFPrism(# Top face vertices
                       x0, y0, z00,
                       x0, y1, z01,
                       x1, y1, z11,
                       # Bottom face vertices
                       x0, y0, z_bottom,
                       x0, y1, z_bottom,
                       x1, y1, z_bottom)
    else:
        # Second triangle: (i0 j0), (i1 j1), (i1 j0)
        return HFPrism(# Top face vertices
                       x0, y0, z00,
                       x1, y1, z11,
                       x1, y0, z10,
                       # Bottom face vertices
                       x0, y0, z_bottom,
                       x1, y1, z_bottom,
                       x1, y0, z_bottom)

