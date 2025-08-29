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

from typing import Tuple

import warp as wp


@wp.func
def hfield_subgrid(
  # In:
  nrow: int,
  ncol: int,
  size: wp.vec4,
  xmax: float,
  xmin: float,
  ymax: float,
  ymin: float,
) -> Tuple[int, int, int, int]:
  """Returns height field subgrid that overlaps with geom AABB.

  Args:
    nrow: height field number of rows
    ncol: height field number of columns
    size: height field size
    xmax: geom maximum x position
    xmin: geom minimum x position
    ymax: geom maximum y position
    ymin: geom minimum y position

  Returns:
    grid coordinate bounds
  """

  # grid resolution
  x_scale = 0.5 * float(ncol - 1) / size[0]
  y_scale = 0.5 * float(nrow - 1) / size[1]

  # subgrid
  cmin = wp.max(0, int(wp.floor((xmin + size[0]) * x_scale)))
  cmax = wp.min(ncol - 1, int(wp.ceil((xmax + size[0]) * x_scale)))
  rmin = wp.max(0, int(wp.floor((ymin + size[1]) * y_scale)))
  rmax = wp.min(nrow - 1, int(wp.ceil((ymax + size[1]) * y_scale)))

  return cmin, rmin, cmax, rmax


@wp.func
def hfield_triangle_prism(
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
  """Returns triangular prism vertex information in compressed representation.

  Args:
    geom_dataid: geom data ids
    hfield_adr: address for height field
    hfield_nrow: height field number of rows
    hfield_ncol: height field number of columns
    hfield_size: height field sizes
    hfield_data: height field data
    hfieldid: height field geom id
    hftri_index: height field triangle index

  Returns:
    triangular prism vertex information (compressed)
  """
  # https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-hfield

  # get heightfield dimensions
  dataid = geom_dataid[hfieldid]
  if dataid < 0 or hftri_index < 0:
    return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  nrow = hfield_nrow[dataid]
  ncol = hfield_ncol[dataid]
  size = hfield_size[dataid]  # (x, y, z_top, z_bottom)

  # calculate which triangle in the grid
  row = (hftri_index // 2) // (ncol - 1)
  col = (hftri_index // 2) % (ncol - 1)

  # calculate vertices in 2D grid
  x_scale = 2.0 * size[0] / float(ncol - 1)
  y_scale = 2.0 * size[1] / float(nrow - 1)

  # grid coordinates (i, j) for triangle corners
  i0 = col
  j0 = row
  i1 = i0 + 1
  j1 = j0 + 1

  # convert grid coordinates to local space x, y coordinates
  x0 = float(i0) * x_scale - size[0]
  y0 = float(j0) * y_scale - size[1]
  x1 = float(i1) * x_scale - size[0]
  y1 = float(j1) * y_scale - size[1]

  # get height values at corners from hfield_data
  base_addr = hfield_adr[dataid]
  z00 = hfield_data[base_addr + j0 * ncol + i0]
  z01 = hfield_data[base_addr + j1 * ncol + i0]
  z10 = hfield_data[base_addr + j0 * ncol + i1]
  z11 = hfield_data[base_addr + j1 * ncol + i1]

  # scale heights from range [0, 1] to [0, z_top]
  z_top = size[2]
  z00 = z00 * z_top
  z01 = z01 * z_top
  z10 = z10 * z_top
  z11 = z11 * z_top

  x2 = wp.where(hftri_index % 2, 1.0, 0.0)
  y2 = wp.where(hftri_index % 2, z10, z01)
  z22 = -size[3]

  # compress 6 prism vertices into 3x3 matrix, see hfield_prism_vertex for details
  return wp.mat33(x0, y0, z00,
                  x1, y1, z11,
                  x2, y2, z22)  # fmt: off


@wp.func
def hfield_prism_vertex(prism: wp.mat33, vert_index: int) -> wp.vec3:
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
      vert_index: index of vertex to extract (0-5)

  Returns:
      3D coordinates of the requested vertex
  """
  if vert_index == 0 or vert_index == 1:
    return prism[vert_index]  # first two vertices stored directly

  if vert_index == 2:  # third vertex
    if prism[2][0] == 0:  # even triangle (i, j+1)
      return wp.vec3(prism[0][0], prism[1][1], prism[2][1])
    else:  # odd triangle (i+1, j)
      return wp.vec3(prism[1][0], prism[0][1], prism[2][1])

  if vert_index == 3 or vert_index == 4:  # bottom vertices below 0 and 1
    return wp.vec3(prism[vert_index - 3][0], prism[vert_index - 3][1], prism[2][2])

  if vert_index == 5:  # bottom vertex below 2
    if prism[2][0] == 0:  # even triangle
      return wp.vec3(prism[0][0], prism[1][1], prism[2][2])
    else:  # odd triangle
      return wp.vec3(prism[1][0], prism[0][1], prism[2][2])
