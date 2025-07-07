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

import warp as wp

from .collision_gjk_generic import build_ccd_generic
from .collision_hfield import hfield_prism_vertex
from .collision_primitive import Geom
from .types import MJ_MINVAL
from .types import GeomType

# TODO(team): improve compile time to enable backward pass
wp.config.enable_backward = False

FLOAT_MIN = -1e30
FLOAT_MAX = 1e30


@wp.func
def _support(geom: Geom, geomtype: int, dir: wp.vec3):
  index = -1
  local_dir = wp.transpose(geom.rot) @ dir
  if geomtype == int(GeomType.SPHERE.value):
    support_pt = geom.pos + geom.size[0] * dir
  elif geomtype == int(GeomType.BOX.value):
    res = wp.cw_mul(wp.sign(local_dir), geom.size)
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.CAPSULE.value):
    res = local_dir * geom.size[0]
    # add cylinder contribution
    res[2] += wp.sign(local_dir[2]) * geom.size[1]
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.ELLIPSOID.value):
    res = wp.cw_mul(local_dir, geom.size)
    res = wp.normalize(res)
    # transform to ellipsoid
    res = wp.cw_mul(res, geom.size)
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.CYLINDER.value):
    res = wp.vec3(0.0, 0.0, 0.0)
    # set result in XY plane: support on circle
    d = wp.sqrt(wp.dot(local_dir, local_dir))
    if d > MJ_MINVAL:
      scl = geom.size[0] / d
      res[0] = local_dir[0] * scl
      res[1] = local_dir[1] * scl
    # set result in Z direction
    res[2] = wp.sign(local_dir[2]) * geom.size[1]
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.MESH.value):
    max_dist = float(FLOAT_MIN)
    if geom.graphadr == -1 or geom.vertnum < 10:
      if geom.index > -1:
        index = geom.index
        max_dist = wp.dot(geom.vert[geom.index], local_dir)
        support_pt = geom.vert[geom.index]
      # exhaustive search over all vertices
      for i in range(geom.vertnum):
        vert = geom.vert[geom.vertadr + i]
        dist = wp.dot(vert, local_dir)
        if dist > max_dist:
          max_dist = dist
          support_pt = vert
          index = geom.vertadr + i
    else:
      numvert = geom.graph[geom.graphadr]
      vert_edgeadr = geom.graphadr + 2
      vert_globalid = geom.graphadr + 2 + numvert
      edge_localid = geom.graphadr + 2 + 2 * numvert
      # hillclimb until no change
      prev = int(-1)
      imax = int(0)
      if geom.index > -1:
        imax = geom.index
        index = geom.index

      while True:
        prev = int(imax)
        i = int(geom.graph[vert_edgeadr + imax])
        while geom.graph[edge_localid + i] >= 0:
          subidx = geom.graph[edge_localid + i]
          idx = geom.graph[vert_globalid + subidx]
          dist = wp.dot(local_dir, geom.vert[geom.vertadr + idx])
          if dist > max_dist:
            max_dist = dist
            imax = int(subidx)
          i += int(1)
        if imax == prev:
          break
      index = imax
      imax = geom.graph[vert_globalid + imax]
      support_pt = geom.vert[geom.vertadr + imax]

    support_pt = geom.rot @ support_pt + geom.pos
  elif geomtype == int(GeomType.HFIELD.value):
    max_dist = float(FLOAT_MIN)
    for i in range(6):
      vert = hfield_prism_vertex(geom.hfprism, i)
      dist = wp.dot(vert, local_dir)
      if dist > max_dist:
        max_dist = dist
        support_pt = vert
    support_pt = geom.rot @ support_pt + geom.pos

  return index, support_pt


def build_ccd():
  return build_ccd_generic(_support)
