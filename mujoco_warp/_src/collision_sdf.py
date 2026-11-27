# Copyright 2025 The Physics-Next Project Developers
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

from mujoco_warp._src.collision_core import CollisionContext
from mujoco_warp._src.collision_core import contact_params
from mujoco_warp._src.collision_core import write_contact
from mujoco_warp._src.math import make_frame
from mujoco_warp._src.math import quat_to_mat
from mujoco_warp._src.math import upper_trid_index
from mujoco_warp._src.types import MJ_MAXCONPAIR
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5
from mujoco_warp._src.types import vec8
from mujoco_warp._src.types import vec8i
from mujoco_warp._src.types import vec_pluginattr
from mujoco_warp._src.util_misc import halton
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# Fixed stack size for MuJoCo mesh BVH traversal (matches MuJoCo C traverseBVH).
_BVH_STACK_SIZE = 64
vec_bvh_stack = wp.types.vector(length=_BVH_STACK_SIZE, dtype=int)
# Per-pair contact reservoir for mesh-SDF BVH traversal (matches MuJoCo C mjMAXCONPAIR).
vec_mesh_dist = wp.types.vector(length=MJ_MAXCONPAIR, dtype=float)
mat_mesh_vec3 = wp.types.matrix(shape=(MJ_MAXCONPAIR, 3), dtype=float)


@wp.struct
class OptimizationParams:
  rel_mat: wp.mat33
  rel_pos: wp.vec3
  attr1: vec_pluginattr
  attr2: vec_pluginattr


@wp.struct
class AABB:
  min: wp.vec3
  max: wp.vec3


@wp.struct
class VolumeData:
  center: wp.vec3
  half_size: wp.vec3
  oct_aabb: wp.array2d[wp.vec3]
  oct_child: wp.array[vec8i]
  oct_coeff: wp.array[vec8]
  root: int = 0
  valid: bool = False


@wp.func
def get_sdf_params(
  # Model:
  oct_child: wp.array[vec8i],
  oct_aabb: wp.array2d[wp.vec3],
  oct_coeff: wp.array[vec8],
  mesh_octadr: wp.array[int],
  plugin: wp.array[int],
  plugin_attr: wp.array[vec_pluginattr],
  # In:
  g_type: int,
  g_size: wp.vec3,
  plugin_id: int,
  mesh_id: int,
) -> Tuple[vec_pluginattr, int, VolumeData]:
  # default attributes from geom size, first 3 values copied
  attributes = vec_pluginattr()
  attributes[0] = g_size[0]
  attributes[1] = g_size[1]
  attributes[2] = g_size[2]
  plugin_index = -1
  volume_data = VolumeData()

  if g_type == GeomType.SDF and plugin_id != -1:
    attributes = plugin_attr[plugin_id]
    plugin_index = plugin[plugin_id]

  elif (g_type == GeomType.SDF or g_type == GeomType.MESH) and mesh_id != -1 and mesh_octadr[mesh_id] != -1:
    octadr = mesh_octadr[mesh_id]
    volume_data.center = oct_aabb[octadr, 0]
    volume_data.half_size = oct_aabb[octadr, 1]
    volume_data.root = octadr
    volume_data.oct_aabb = oct_aabb
    volume_data.oct_child = oct_child
    volume_data.oct_coeff = oct_coeff
    volume_data.valid = True

  return attributes, plugin_index, volume_data


@wp.func
def transform_aabb(aabb_pos: wp.vec3, aabb_size: wp.vec3, pos: wp.vec3, ori: wp.mat33) -> AABB:
  aabb = AABB()
  aabb.max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)
  aabb.min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)
  for i in range(8):
    vec = wp.vec3(
      aabb_size.x * (1.0 if (i & 1) else -1.0),
      aabb_size.y * (1.0 if (i & 2) else -1.0),
      aabb_size.z * (1.0 if (i & 4) else -1.0),
    )
    frame_vec = ori * (vec + aabb_pos) + pos
    aabb.min = wp.min(aabb.min, frame_vec)
    aabb.max = wp.max(aabb.max, frame_vec)
  return aabb


@wp.func
def radial_field(a: wp.vec3, x: wp.vec3, size: wp.vec3) -> wp.vec3:
  field = wp.cw_div(-size, a)
  field = wp.normalize(field)
  field[0] *= wp.sign(x[0])
  field[1] *= wp.sign(x[1])
  field[2] *= wp.sign(x[2])
  return field


@wp.func
def sphere(p: wp.vec3, size: wp.vec3) -> float:
  return wp.length(p) - size[0]


@wp.func
def box(p: wp.vec3, size: wp.vec3) -> float:
  a = wp.abs(p) - size
  if a[0] >= 0 or a[1] >= 0 or a[2] >= 0:
    z = wp.vec3(0.0, 0.0, 0.0)
    b = wp.max(a, z)
    return wp.norm_l2(b) + wp.min(wp.max(a), 0.0)
  b = radial_field(a, p, size)
  t = -wp.cw_div(a, wp.abs(b))
  return -wp.min(t) * wp.norm_l2(b)


@wp.func
def ellipsoid(p: wp.vec3, size: wp.vec3) -> float:
  scaled_p = wp.vec3(p[0] / size[0], p[1] / size[1], p[2] / size[2])
  k0 = wp.length(scaled_p)
  k1 = wp.length(wp.vec3(p[0] / (size[0] ** 2.0), p[1] / (size[1] ** 2.0), p[2] / (size[2] ** 2.0)))
  if k1 < MJ_MINVAL:
    return -wp.min(size)
  return k0 * (k0 - 1.0) / k1


@wp.func
def capsule(p: wp.vec3, size: wp.vec3) -> float:
  r = size[0]
  h = size[1]
  pz_clamped = wp.clamp(p[2], -h, h)
  diff = wp.vec3(p[0], p[1], p[2] - pz_clamped)
  return wp.length(diff) - r


@wp.func
def cylinder(p: wp.vec3, size: wp.vec3) -> float:
  r = size[0]
  h = size[1]
  dx = wp.length(wp.vec2(p[0], p[1])) - r
  dy = wp.abs(p[2]) - h
  return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))


@wp.func
def grad_sphere(p: wp.vec3) -> wp.vec3:
  c = wp.length(p)
  if c > 1e-9:
    return p / c
  else:
    return wp.vec3(0.0)


@wp.func
def grad_box(p: wp.vec3, size: wp.vec3) -> wp.vec3:
  a = wp.abs(p) - size
  if wp.max(a) < 0.0:
    return radial_field(a, p, size)
  b = wp.max(a, wp.vec3(0.0))
  c = wp.norm_l2(b)
  g = wp.vec3(0.0)
  if c == 0.0:
    for j in range(3):
      if a[j] == 0.0:
        g[j] = -1.0 if p[j] < 0.0 else 1.0
    return wp.normalize(g)
  for j in range(3):
    if a[j] > 0.0:
      g[j] = (b[j] / c) * (-1.0 if p[j] < 0.0 else 1.0)
  return g


@wp.func
def grad_ellipsoid(p: wp.vec3, size: wp.vec3) -> wp.vec3:
  a = wp.vec3(p[0] / size[0], p[1] / size[1], p[2] / size[2])
  b = wp.vec3(a[0] / size[0], a[1] / size[1], a[2] / size[2])
  k0 = wp.length(a)
  k1 = wp.length(b)
  if k0 < MJ_MINVAL or k1 < MJ_MINVAL:
    return wp.vec3(0.0)
  invK0 = 1.0 / k0
  invK1 = 1.0 / k1
  gk0 = b * invK0
  gk1 = wp.vec3(
    b[0] * invK1 / (size[0] * size[0]),
    b[1] * invK1 / (size[1] * size[1]),
    b[2] * invK1 / (size[2] * size[2]),
  )
  df_dk0 = (2.0 * k0 - 1.0) * invK1
  df_dk1 = k0 * (k0 - 1.0) * invK1 * invK1
  raw_grad = gk0 * df_dk0 - gk1 * df_dk1
  norm = wp.length(raw_grad)
  if norm > MJ_MINVAL:
    return raw_grad / norm
  return wp.vec3(0.0)


@wp.func
def grad_capsule(p: wp.vec3, size: wp.vec3) -> wp.vec3:
  h = size[1]
  pz_clamped = wp.clamp(p[2], -h, h)
  diff = wp.vec3(p[0], p[1], p[2] - pz_clamped)
  c = wp.length(diff)
  if c > MJ_MINVAL:
    return diff / c
  else:
    return wp.vec3(0.0)


@wp.func
def grad_cylinder(p: wp.vec3, size: wp.vec3) -> wp.vec3:
  r = size[0]
  h = size[1]

  radial_dist = wp.length(wp.vec2(p[0], p[1]))
  if radial_dist > MJ_MINVAL:
    u = wp.vec3(p[0] / radial_dist, p[1] / radial_dist, 0.0)
  else:
    u = wp.vec3(0.0)

  w = wp.vec3(0.0, 0.0, wp.sign(p[2]))

  dx = radial_dist - r
  dy = wp.abs(p[2]) - h

  if dx > 0.0 and dy > 0.0:
    v = wp.vec2(dx, dy)
    len_v = wp.length(v)
    if len_v > MJ_MINVAL:
      return u * (dx / len_v) + w * (dy / len_v)
    else:
      return wp.vec3(0.0)
  elif dx > 0.0:
    return u
  elif dy > 0.0:
    return w
  else:
    if dx > dy:
      return u
    else:
      return w


@wp.func
def user_sdf(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> float:
  """User-defined SDF function.

  Access attributes via attr[i] where i is the attribute index (0 to _NPLUGINATTR-1).
  """
  wp.printf("ERROR: user_sdf function must be implemented by user code\n")
  return 0.0


@wp.func
def user_sdf_grad(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> wp.vec3:
  """User-defined SDF gradient function.

  Access attributes via attr[i] where i is the attribute index (0 to _NPLUGINATTR-1).
  """
  wp.printf("ERROR: user_sdf_grad function must be implemented by user code\n")
  return wp.vec3(0.0)


@wp.func
def find_oct(
  oct_child: wp.array[vec8i], oct_aabb: wp.array2d[wp.vec3], p: wp.vec3, grad: bool, root: int
) -> Tuple[int, Tuple[vec8, vec8, vec8]]:
  stack = root
  niter = int(100)
  rx = vec8(0.0)
  ry = vec8(0.0)
  rz = vec8(0.0)
  eps = 1e-6

  while niter > 0:
    niter -= 1
    node = stack

    if node == -1:
      wp.printf("ERROR: Invalid node number\n")
      return -1, (rx, ry, rz)

    vmin = oct_aabb[node, 0] - oct_aabb[node, 1]
    vmax = oct_aabb[node, 0] + oct_aabb[node, 1]

    if (
      p[0] + eps < vmin[0]
      or p[0] - eps > vmax[0]
      or p[1] + eps < vmin[1]
      or p[1] - eps > vmax[1]
      or p[2] + eps < vmin[2]
      or p[2] - eps > vmax[2]
    ):
      continue

    coord = wp.cw_div(p - vmin, vmax - vmin)

    # check if the node is a leaf
    # child indices are relative to root (mesh_octadr offset)
    child0 = oct_child[node][0]
    # Evaluate this hot leaf predicate eagerly to avoid branch-heavy codegen.
    if (
      int(child0 == -1)
      & int(oct_child[node][1] == -1)
      & int(oct_child[node][2] == -1)
      & int(oct_child[node][3] == -1)
      & int(oct_child[node][4] == -1)
      & int(oct_child[node][5] == -1)
      & int(oct_child[node][6] == -1)
      & int(oct_child[node][7] == -1)
    ) != 0:
      if not grad:
        for j in range(8):
          rx[j] = (
            (coord[0] if j & 1 else 1.0 - coord[0])
            * (coord[1] if j & 2 else 1.0 - coord[1])
            * (coord[2] if j & 4 else 1.0 - coord[2])
          )
      else:
        # Convert derivatives of cell coordinates to derivatives of physical coordinates.
        inv_cell = wp.cw_div(wp.vec3(1.0), vmax - vmin)
        for j in range(8):
          # fmt: off
          rx[j] = ((1.0 if j & 1 else -1.0) * (coord[1] if j & 2 else 1.0 - coord[1])
                   * (coord[2] if j & 4 else 1.0 - coord[2]) * inv_cell[0])
          ry[j] = ((coord[0] if j & 1 else 1.0 - coord[0]) * (1.0 if j & 2 else -1.0)
                   * (coord[2] if j & 4 else 1.0 - coord[2]) * inv_cell[1])
          rz[j] = ((coord[0] if j & 1 else 1.0 - coord[0]) * (coord[1] if j & 2 else 1.0 - coord[1])
                   * (1.0 if j & 4 else -1.0) * inv_cell[2])
          # fmt: on
      return node, (rx, ry, rz)

    # compute which of 8 children to visit next
    # child indices are stored relative to mesh_octadr, add root offset
    x = 0 if coord[0] < 0.5 else 1
    y = 0 if coord[1] < 0.5 else 1
    z = 0 if coord[2] < 0.5 else 1
    child = oct_child[node][4 * z + 2 * y + x]
    stack = child + root if child != -1 else -1

  wp.print("ERROR: Node not found\n")
  return -1, (rx, ry, rz)


@wp.func
def box_project(center: wp.vec3, half_size: wp.vec3, xyz: wp.vec3) -> Tuple[float, wp.vec3]:
  r = xyz - center
  q = wp.vec3(wp.abs(r[0]) - half_size[0], wp.abs(r[1]) - half_size[1], wp.abs(r[2]) - half_size[2])

  if q[0] <= 0.0 and q[1] <= 0.0 and q[2] <= 0.0:
    return 0.0, xyz

  else:
    dist_sqr = 0.0
    eps = 1e-4
    point = wp.vec3(xyz[0], xyz[1], xyz[2])

    if q[0] >= 0.0:
      dist_sqr += q[0] * q[0]
      if r[0] > 0.0:
        point = wp.vec3(point[0] - (q[0] + eps), point[1], point[2])
      else:
        point = wp.vec3(point[0] + (q[0] + eps), point[1], point[2])

    if q[1] >= 0.0:
      dist_sqr += q[1] * q[1]
      if r[1] > 0.0:
        point = wp.vec3(point[0], point[1] - (q[1] + eps), point[2])
      else:
        point = wp.vec3(point[0], point[1] + (q[1] + eps), point[2])

    if q[2] >= 0.0:
      dist_sqr += q[2] * q[2]
      if r[2] > 0.0:
        point = wp.vec3(point[0], point[1], point[2] - (q[2] + eps))
      else:
        point = wp.vec3(point[0], point[1], point[2] + (q[2] + eps))

    return wp.sqrt(dist_sqr), point


@wp.func
def sample_volume_sdf(xyz: wp.vec3, volume_data: VolumeData) -> float:
  dist0, point = box_project(volume_data.center, volume_data.half_size, xyz)
  node, weights = find_oct(volume_data.oct_child, volume_data.oct_aabb, point, grad=False, root=volume_data.root)
  return dist0 + wp.dot(weights[0], volume_data.oct_coeff[node])


@wp.func
def sample_volume_grad(xyz: wp.vec3, volume_data: VolumeData) -> wp.vec3:
  dist0, point = box_project(volume_data.center, volume_data.half_size, xyz)
  if dist0 > 0:
    h = 1e-4
    dx = wp.vec3(h, 0.0, 0.0)
    dy = wp.vec3(0.0, h, 0.0)
    dz = wp.vec3(0.0, 0.0, h)
    f = sample_volume_sdf(xyz, volume_data)
    grad_x = (sample_volume_sdf(xyz + dx, volume_data) - f) / h
    grad_y = (sample_volume_sdf(xyz + dy, volume_data) - f) / h
    grad_z = (sample_volume_sdf(xyz + dz, volume_data) - f) / h
    return wp.vec3(grad_x, grad_y, grad_z)
  node, weights = find_oct(volume_data.oct_child, volume_data.oct_aabb, point, grad=True, root=volume_data.root)
  grad_x = wp.dot(weights[0], volume_data.oct_coeff[node])
  grad_y = wp.dot(weights[1], volume_data.oct_coeff[node])
  grad_z = wp.dot(weights[2], volume_data.oct_coeff[node])
  return wp.vec3(grad_x, grad_y, grad_z)


@wp.func
def sdf(type: int, p: wp.vec3, attr: vec_pluginattr, sdf_type: int, volume_data: VolumeData) -> float:
  # extract first 3 elements as vec3 for primitive sdf functions
  attr_vec3 = wp.vec3(attr[0], attr[1], attr[2])
  if type == GeomType.PLANE:
    return p[2]
  elif type == GeomType.SPHERE:
    return sphere(p, attr_vec3)
  elif type == GeomType.CAPSULE:
    return capsule(p, attr_vec3)
  elif type == GeomType.CYLINDER:
    return cylinder(p, attr_vec3)
  elif type == GeomType.BOX:
    return box(p, attr_vec3)
  elif type == GeomType.ELLIPSOID:
    return ellipsoid(p, attr_vec3)
  elif type == GeomType.SDF:
    if sdf_type == -1:
      return sample_volume_sdf(p, volume_data)
    else:
      return user_sdf(p, attr, sdf_type)
  elif type == GeomType.MESH and volume_data.valid:
    return sample_volume_sdf(p, volume_data)
  wp.printf("ERROR: SDF type not implemented\n")
  return 0.0


@wp.func
def sdf_grad(type: int, p: wp.vec3, attr: vec_pluginattr, sdf_type: int, volume_data: VolumeData) -> wp.vec3:
  # extract first 3 elements as vec3 for primitive sdf functions
  attr_vec3 = wp.vec3(attr[0], attr[1], attr[2])
  if type == GeomType.PLANE:
    grad = wp.vec3(0.0, 0.0, 1.0)
    return grad
  elif type == GeomType.SPHERE:
    return grad_sphere(p)
  elif type == GeomType.CAPSULE:
    return grad_capsule(p, attr_vec3)
  elif type == GeomType.CYLINDER:
    return grad_cylinder(p, attr_vec3)
  elif type == GeomType.BOX:
    return grad_box(p, attr_vec3)
  elif type == GeomType.ELLIPSOID:
    return grad_ellipsoid(p, attr_vec3)
  elif type == GeomType.SDF:
    if sdf_type == -1:
      return sample_volume_grad(p, volume_data)
    else:
      return user_sdf_grad(p, attr, sdf_type)
  elif type == GeomType.MESH and volume_data.valid:
    return sample_volume_grad(p, volume_data)
  wp.printf("ERROR: SDF grad type not implemented\n")
  return wp.vec3(0.0)


@wp.func
def clearance(
  # In:
  type1: int,
  p1: wp.vec3,
  p2: wp.vec3,
  s1: vec_pluginattr,
  s2: vec_pluginattr,
  sdf_type1: int,
  sdf_type2: int,
  sfd_intersection: bool,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
) -> float:
  sdf1 = sdf(type1, p1, s1, sdf_type1, volume_data1)
  sdf2 = sdf(GeomType.SDF, p2, s2, sdf_type2, volume_data2)
  if sfd_intersection:
    return wp.max(sdf1, sdf2)
  else:
    return sdf1 + sdf2 + wp.abs(wp.max(sdf1, sdf2))


@wp.func
def compute_grad(
  # In:
  type1: int,
  p1: wp.vec3,
  p2: wp.vec3,
  params: OptimizationParams,
  sdf_type1: int,
  sdf_type2: int,
  sfd_intersection: bool,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
) -> Tuple[wp.vec3, float]:
  A = sdf(type1, p1, params.attr1, sdf_type1, volume_data1)
  B = sdf(GeomType.SDF, p2, params.attr2, sdf_type2, volume_data2)
  grad1 = sdf_grad(type1, p1, params.attr1, sdf_type1, volume_data1)
  grad2 = sdf_grad(GeomType.SDF, p2, params.attr2, sdf_type2, volume_data2)
  grad1_transformed = wp.transpose(params.rel_mat) * grad1
  if sfd_intersection:
    dist = wp.max(A, B)
    if A > B:
      gradient = grad1_transformed
    else:
      gradient = grad2
  else:
    dist = A + B + wp.abs(wp.max(A, B))
    gradient = grad2 + grad1_transformed
    max_val = wp.max(A, B)
    if A > B:
      max_grad = grad1_transformed
    else:
      max_grad = grad2
    sign = wp.sign(max_val)
    gradient += max_grad * sign
  return gradient, dist


@wp.func
def gradient_step(
  # In:
  type1: int,
  x: wp.vec3,
  params: OptimizationParams,
  sdf_type1: int,
  sdf_type2: int,
  niter: int,
  sfd_intersection: bool,
  alpha0: float,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
) -> Tuple[float, wp.vec3]:
  amin = 1e-4
  rho = 0.5
  c = 0.1
  dist = float(1e10)
  for i in range(niter):
    alpha = alpha0
    x2 = wp.vec3(x[0], x[1], x[2])
    x1 = params.rel_mat * x2 + params.rel_pos
    grad, dist0 = compute_grad(type1, x1, x2, params, sdf_type1, sdf_type2, sfd_intersection, volume_data1, volume_data2)
    grad_dot = wp.dot(grad, grad)
    if grad_dot < 1e-12:
      return dist0, x
    wolfe = -c * alpha * grad_dot
    while True:
      alpha *= rho
      wolfe *= rho
      x = x2 - grad * alpha
      x1 = params.rel_mat * x + params.rel_pos
      dist = clearance(
        type1,
        x1,
        x,
        params.attr1,
        params.attr2,
        sdf_type1,
        sdf_type2,
        sfd_intersection,
        volume_data1,
        volume_data2,
      )
      if alpha <= amin or (dist - dist0) <= wolfe:
        break
    if dist > dist0:
      return dist0, x2
  return dist, x


@wp.func
def gradient_descent(
  # In:
  type1: int,
  x0_initial: wp.vec3,
  attr1: vec_pluginattr,
  attr2: vec_pluginattr,
  pos1: wp.vec3,
  rot1: wp.mat33,
  pos2: wp.vec3,
  rot2: wp.mat33,
  sdf_type1: int,
  sdf_type2: int,
  sdf_iterations: int,
  linesearch_alpha0: float,
  max_dist: float,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
) -> Tuple[float, wp.vec3, wp.vec3]:
  params = OptimizationParams()
  params.rel_mat = wp.transpose(rot1) * rot2
  params.rel_pos = wp.transpose(rot1) * (pos2 - pos1)
  params.attr1 = attr1
  params.attr2 = attr2
  dist = float(0.0)
  x = x0_initial
  phase = int(0)
  while phase < 2:
    niter = sdf_iterations
    intersection = False
    if phase == 1:
      niter = 1
      intersection = True
    dist, x = gradient_step(
      type1,
      x,
      params,
      sdf_type1,
      sdf_type2,
      niter,
      intersection,
      linesearch_alpha0,
      volume_data1,
      volume_data2,
    )
    phase += 1
  x_1 = params.rel_mat * x + params.rel_pos
  d_out = 2.0 * dist if dist > 0.0 else dist
  if d_out >= max_dist:
    return d_out, wp.vec3(0.0), wp.vec3(0.0)
  grad1 = wp.normalize(wp.transpose(params.rel_mat) * sdf_grad(type1, x_1, params.attr1, sdf_type1, volume_data1))
  grad2 = wp.normalize(sdf_grad(GeomType.SDF, x, params.attr2, sdf_type2, volume_data2))
  n = wp.normalize(grad1 - grad2)
  pos = rot2 * x + pos2
  n = rot2 * n
  if dist > 0.0:
    # separated: offset along normal by half the midsurface residual (sdf1 - sdf2)
    sdf1 = sdf(type1, x_1, params.attr1, sdf_type1, volume_data1)
    sdf2 = sdf(GeomType.SDF, x, params.attr2, sdf_type2, volume_data2)
    return d_out, pos - n * (0.5 * (sdf1 - sdf2)), n
  pos3 = pos + n * (dist * 0.5)
  return dist, pos3, n


@wp.func
def _frank_wolfe_triangle(
  # In:
  v0: wp.vec3,
  v1: wp.vec3,
  v2: wp.vec3,
  volume_data: VolumeData,
  sdf_type: int,
  attr: vec_pluginattr,
  n_iter: int,
  bary_u: float,
  bary_v: float,
) -> Tuple[float, wp.vec3, wp.vec3]:
  # warm start from barycentric point (e.g., closest point from mesh query)
  x0 = v0 * bary_u + v1 * bary_v + v2 * (1.0 - bary_u - bary_v)
  x = x0

  for k in range(n_iter):
    g = sdf_grad(GeomType.SDF, x, attr, sdf_type, volume_data)
    d0 = wp.dot(v0, g)
    d1 = wp.dot(v1, g)
    d2 = wp.dot(v2, g)
    s = v0
    min_d = d0
    if d1 < min_d:
      s = v1
      min_d = d1
    if d2 < min_d:
      s = v2
    dir = s - x
    if k > 0 and wp.dot(dir, g) >= -1e-7:
      break
    gamma = 2.0 / float(k + 2)
    x = x + dir * gamma

  dist = sdf(GeomType.SDF, x, attr, sdf_type, volume_data)
  dist0 = sdf(GeomType.SDF, x0, attr, sdf_type, volume_data)
  if dist0 < dist:
    x = x0
    dist = dist0
  grad = sdf_grad(GeomType.SDF, x, attr, sdf_type, volume_data)
  return dist, x, wp.normalize(grad)


@wp.kernel
def _sdf_candidates(
  # Model:
  geom_type: wp.array[int],
  # Data in:
  naconmax_in: int,
  ncollision_in: wp.array[int],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  # Out:
  nsdf_out: wp.array[int],
  sdf_collisionid_out: wp.array[int],
):
  """Lists the collision candidates that involve an SDF geom."""
  collisionid = wp.tid()
  if collisionid >= wp.min(ncollision_in[0], naconmax_in):
    return

  geoms = collision_pair_in[collisionid]
  t1 = geom_type[geoms[0]]
  t2 = geom_type[geoms[1]]
  if (t1 == GeomType.SDF or t2 == GeomType.SDF) and t1 != GeomType.HFIELD and t2 != GeomType.HFIELD:
    sdfid = wp.atomic_add(nsdf_out, 0, 1)  # kernel_analyzer: ignore[atomic]
    if sdfid < naconmax_in:
      sdf_collisionid_out[sdfid] = collisionid


@wp.kernel
def _mesh_sdf_narrowphase(
  # Model:
  bvh_child: wp.array[wp.vec2i],
  bvh_nodeid: wp.array[int],
  bvh_aabb: wp.array2d[wp.vec3],
  oct_child: wp.array[vec8i],
  oct_aabb: wp.array2d[wp.vec3],
  oct_coeff: wp.array[vec8],
  geom_type: wp.array[int],
  geom_condim: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_size: wp.array2d[wp.vec3],
  geom_aabb: wp.array3d[wp.vec3],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  geom_adhesion: wp.array2d[float],
  mesh_vertadr: wp.array[int],
  mesh_faceadr: wp.array[int],
  mesh_bvhadr: wp.array[int],
  mesh_octadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_face: wp.array[wp.vec3i],
  pair_dim: wp.array[int],
  pair_solref: wp.array2d[wp.vec2],
  pair_solreffriction: wp.array2d[wp.vec2],
  pair_solimp: wp.array2d[vec5],
  pair_margin: wp.array2d[float],
  pair_gap: wp.array2d[float],
  pair_adhesion: wp.array2d[float],
  pair_friction: wp.array2d[vec5],
  plugin: wp.array[int],
  plugin_attr: wp.array[vec_pluginattr],
  geom_plugin_index: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xquat_in: wp.array2d[wp.quat],
  naconmax_in: int,
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_pairid_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
  nsdf_in: wp.array[int],
  sdf_collisionid_in: wp.array[int],
  sdf_iterations: int,
  # Data out:
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_efc_address_out: wp.array2d[int],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  contact_adhesion_out: wp.array[float],
  nacon_out: wp.array[int],
):
  sdfid = wp.tid()
  if sdfid >= wp.min(nsdf_in[0], naconmax_in):
    return
  contact_tid = sdf_collisionid_in[sdfid]
  geoms = collision_pair_in[contact_tid]
  type1 = geom_type[geoms[0]]
  type2 = geom_type[geoms[1]]
  if type1 == GeomType.SDF and type2 != GeomType.SDF:
    geoms = wp.vec2i(geoms[1], geoms[0])
    type1 = type2
    type2 = GeomType.SDF
  if type1 != GeomType.MESH:
    return

  g1 = geoms[0]
  g2 = geoms[1]
  worldid = collision_worldid_in[contact_tid]
  dataid_id = worldid % geom_dataid.shape[0]
  mesh_id1 = geom_dataid[dataid_id, g1]
  if mesh_id1 < 0:
    return
  root = mesh_bvhadr[mesh_id1]
  if root < 0:
    return

  _, margin, gap, condim, friction, solref, solreffriction, solimp, adhesion = contact_params(
    geom_condim,
    geom_priority,
    geom_solmix,
    geom_solref,
    geom_solimp,
    geom_friction,
    geom_margin,
    geom_gap,
    geom_adhesion,
    pair_dim,
    pair_solref,
    pair_solreffriction,
    pair_solimp,
    pair_margin,
    pair_gap,
    pair_adhesion,
    pair_friction,
    collision_pair_in,
    collision_pairid_in,
    contact_tid,
    worldid,
  )

  pos1 = geom_xpos_in[worldid, g1]
  rot1 = quat_to_mat(geom_xquat_in[worldid, g1])
  pos2 = geom_xpos_in[worldid, g2]
  rot2 = quat_to_mat(geom_xquat_in[worldid, g2])

  rot21 = wp.transpose(rot1) * rot2
  pos21 = wp.transpose(rot1) * (pos2 - pos1)
  margin_gap = margin + wp.max(0.0, gap)
  aabb_id = worldid % geom_aabb.shape[0]
  aabb1_pos = geom_aabb[aabb_id, g1, 0]
  aabb1_size = geom_aabb[aabb_id, g1, 1]
  aabb2 = transform_aabb(geom_aabb[aabb_id, g2, 0], geom_aabb[aabb_id, g2, 1], pos21, rot21)
  pad = wp.vec3(0.5 * wp.max(0.0, margin_gap))
  lo = wp.max(aabb1_pos - aabb1_size, aabb2.min) - pad
  size = wp.min(aabb1_pos + aabb1_size, aabb2.max) + pad - lo
  if size[0] < 0.0 or size[1] < 0.0 or size[2] < 0.0:
    return

  size_id = worldid % geom_size.shape[0]
  attr2, sdf_type2, volume_data2 = get_sdf_params(
    oct_child,
    oct_aabb,
    oct_coeff,
    mesh_octadr,
    plugin,
    plugin_attr,
    type2,
    geom_size[size_id, g2],
    geom_plugin_index[g2],
    geom_dataid[dataid_id, g2],
  )

  # traverse MuJoCo's mesh BVH once per pair (MuJoCo C mjc_MeshSDF / traverseBVH)
  rot12 = wp.transpose(rot21)
  pos12 = -(rot12 * pos21)
  root_size = bvh_aabb[root, 1]
  # circumsphere test against SDF (MuJoCo C boxIntersect)
  if sdf(GeomType.SDF, rot12 * bvh_aabb[root, 0] + pos12, attr2, sdf_type2, volume_data2) >= (
    wp.length(root_size) + margin_gap
  ):
    return
  vert_adr = mesh_vertadr[mesh_id1]
  face_adr = mesh_faceadr[mesh_id1]
  # merge nearby triangle contacts (5% of mesh radius) to avoid dense-patch clustering
  merge_dist_sq = wp.dot(root_size, root_size) * 2.5e-3
  buf_dist = vec_mesh_dist(1e10)
  buf_pos = mat_mesh_vec3()
  buf_normal = mat_mesh_vec3()
  stack = vec_bvh_stack()
  stack[0] = root
  top = int(1)
  ncon = int(0)
  while top > 0:
    top -= 1
    node = stack[top]
    leaf_id = bvh_nodeid[node]
    if leaf_id != -1:
      f = mesh_face[face_adr + leaf_id]
      v0 = rot12 * mesh_vert[vert_adr + f[0]] + pos12
      v1 = rot12 * mesh_vert[vert_adr + f[1]] + pos12
      v2 = rot12 * mesh_vert[vert_adr + f[2]] + pos12
      d, x, n_local = _frank_wolfe_triangle(v0, v1, v2, volume_data2, sdf_type2, attr2, sdf_iterations, 1.0 / 3.0, 1.0 / 3.0)
      if d < margin_gap and wp.length(n_local) >= MJ_MINVAL:
        n_world = -(rot2 * n_local)
        p_world = rot2 * x + pos2 + n_world * (0.5 * d)
        # deduplicate against existing cluster or replace shallowest slot when full
        slot = int(-1)
        worst_slot = int(0)
        worst_dist = buf_dist[0]
        for k in range(ncon):
          if wp.length_sq(p_world - buf_pos[k]) < merge_dist_sq:
            slot = k
            break
          if buf_dist[k] > worst_dist:
            worst_dist = buf_dist[k]
            worst_slot = k
        if slot >= 0:
          if d < buf_dist[slot]:
            buf_dist[slot] = d
            buf_pos[slot] = p_world
            buf_normal[slot] = n_world
        elif ncon < MJ_MAXCONPAIR:
          buf_dist[ncon] = d
          buf_pos[ncon] = p_world
          buf_normal[ncon] = n_world
          ncon += 1
        elif d < worst_dist:
          buf_dist[worst_slot] = d
          buf_pos[worst_slot] = p_world
          buf_normal[worst_slot] = n_world
    else:
      children = bvh_child[node]
      c0 = root + children[0] if children[0] != -1 else -1
      c1 = root + children[1] if children[1] != -1 else -1
      # visit child with smaller (sdf - radius) first so deepest contacts are found first
      d0 = (
        sdf(GeomType.SDF, rot12 * bvh_aabb[c0, 0] + pos12, attr2, sdf_type2, volume_data2) - wp.length(bvh_aabb[c0, 1])
        if c0 != -1
        else margin_gap
      )
      d1 = (
        sdf(GeomType.SDF, rot12 * bvh_aabb[c1, 0] + pos12, attr2, sdf_type2, volume_data2) - wp.length(bvh_aabb[c1, 1])
        if c1 != -1
        else margin_gap
      )
      if d0 < d1:
        if d1 < margin_gap and top < _BVH_STACK_SIZE:
          stack[top] = c1
          top += 1
        if d0 < margin_gap and top < _BVH_STACK_SIZE:
          stack[top] = c0
          top += 1
      else:
        if d0 < margin_gap and top < _BVH_STACK_SIZE:
          stack[top] = c0
          top += 1
        if d1 < margin_gap and top < _BVH_STACK_SIZE:
          stack[top] = c1
          top += 1
  pairid = collision_pairid_in[contact_tid]
  for k in range(ncon):
    write_contact(
      naconmax_in,
      wp.min(k, 7),
      buf_dist[k],
      buf_pos[k],
      make_frame(buf_normal[k]),
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      adhesion,
      geoms,
      pairid,
      worldid,
      contact_dist_out,
      contact_pos_out,
      contact_frame_out,
      contact_includemargin_out,
      contact_friction_out,
      contact_solref_out,
      contact_solreffriction_out,
      contact_solimp_out,
      contact_dim_out,
      contact_geom_out,
      contact_efc_address_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      contact_adhesion_out,
      nacon_out,
    )


@wp.kernel
def _sdf_narrowphase(
  # Model:
  oct_child: wp.array[vec8i],
  oct_aabb: wp.array2d[wp.vec3],
  oct_coeff: wp.array[vec8],
  geom_type: wp.array[int],
  geom_condim: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array2d[float],
  geom_solref: wp.array2d[wp.vec2],
  geom_solimp: wp.array2d[vec5],
  geom_size: wp.array2d[wp.vec3],
  geom_aabb: wp.array3d[wp.vec3],
  geom_friction: wp.array2d[wp.vec3],
  geom_margin: wp.array2d[float],
  geom_gap: wp.array2d[float],
  geom_adhesion: wp.array2d[float],
  mesh_octadr: wp.array[int],
  pair_dim: wp.array[int],
  pair_solref: wp.array2d[wp.vec2],
  pair_solreffriction: wp.array2d[wp.vec2],
  pair_solimp: wp.array2d[vec5],
  pair_margin: wp.array2d[float],
  pair_gap: wp.array2d[float],
  pair_adhesion: wp.array2d[float],
  pair_friction: wp.array2d[vec5],
  plugin: wp.array[int],
  plugin_attr: wp.array[vec_pluginattr],
  geom_plugin_index: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xquat_in: wp.array2d[wp.quat],
  naconmax_in: int,
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  collision_pairid_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
  nsdf_in: wp.array[int],
  sdf_collisionid_in: wp.array[int],
  sdf_iterations: int,
  linesearch_alpha0: float,
  # Data out:
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_efc_address_out: wp.array2d[int],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  contact_adhesion_out: wp.array[float],
  nacon_out: wp.array[int],
):
  # a candidate's initial points are adjacent threads, so a warp works on one geom pair
  sdfid, i = wp.tid()
  if sdfid >= wp.min(nsdf_in[0], naconmax_in):
    return
  contact_tid = sdf_collisionid_in[sdfid]
  geoms = collision_pair_in[contact_tid]
  type1 = geom_type[geoms[0]]
  type2 = geom_type[geoms[1]]
  if type1 == GeomType.SDF and type2 != GeomType.SDF:
    geoms = wp.vec2i(geoms[1], geoms[0])
    type1 = type2
    type2 = GeomType.SDF
  if type1 == GeomType.MESH:
    return

  g1 = geoms[0]
  g2 = geoms[1]
  worldid = collision_worldid_in[contact_tid]
  _, margin, gap, condim, friction, solref, solreffriction, solimp, adhesion = contact_params(
    geom_condim,
    geom_priority,
    geom_solmix,
    geom_solref,
    geom_solimp,
    geom_friction,
    geom_margin,
    geom_gap,
    geom_adhesion,
    pair_dim,
    pair_solref,
    pair_solreffriction,
    pair_solimp,
    pair_margin,
    pair_gap,
    pair_adhesion,
    pair_friction,
    collision_pair_in,
    collision_pairid_in,
    contact_tid,
    worldid,
  )

  pos1 = geom_xpos_in[worldid, g1]
  rot1 = quat_to_mat(geom_xquat_in[worldid, g1])
  pos2 = geom_xpos_in[worldid, g2]
  rot2 = quat_to_mat(geom_xquat_in[worldid, g2])

  # g2 frame to g1 frame
  rot21 = wp.transpose(rot1) * rot2
  pos21 = wp.transpose(rot1) * (pos2 - pos1)

  margin_gap = margin + wp.max(0.0, gap)
  aabb_id = worldid % geom_aabb.shape[0]
  aabb1_pos = geom_aabb[aabb_id, g1, 0]
  aabb1_size = geom_aabb[aabb_id, g1, 1]
  aabb2 = transform_aabb(geom_aabb[aabb_id, g2, 0], geom_aabb[aabb_id, g2, 1], pos21, rot21)
  pad = wp.vec3(0.5 * wp.max(0.0, margin_gap))
  lo = wp.max(aabb1_pos - aabb1_size, aabb2.min) - pad
  size = wp.min(aabb1_pos + aabb1_size, aabb2.max) + pad - lo
  # A broadphase candidate gap can admit pairs with no geometric intersection.
  if size[0] < 0.0 or size[1] < 0.0 or size[2] < 0.0:
    return

  dataid_id = worldid % geom_dataid.shape[0]
  size_id = worldid % geom_size.shape[0]
  attr1, sdf_type1, volume_data1 = get_sdf_params(
    oct_child,
    oct_aabb,
    oct_coeff,
    mesh_octadr,
    plugin,
    plugin_attr,
    type1,
    geom_size[size_id, g1],
    geom_plugin_index[g1],
    geom_dataid[dataid_id, g1],
  )
  attr2, sdf_type2, volume_data2 = get_sdf_params(
    oct_child,
    oct_aabb,
    oct_coeff,
    mesh_octadr,
    plugin,
    plugin_attr,
    type2,
    geom_size[size_id, g2],
    geom_plugin_index[g2],
    geom_dataid[dataid_id, g2],
  )

  x1 = lo + wp.cw_mul(size, wp.vec3(halton(i, 2), halton(i, 3), halton(i, 5)))
  x2 = wp.transpose(rot21) * (x1 - pos21)
  dist, pos, n = gradient_descent(
    type1,
    x2,
    attr1,
    attr2,
    pos1,
    rot1,
    pos2,
    rot2,
    sdf_type1,
    sdf_type2,
    sdf_iterations,
    linesearch_alpha0,
    margin_gap,
    volume_data1,
    volume_data2,
  )

  if dist < margin_gap and wp.length(n) >= MJ_MINVAL:
    write_contact(
      naconmax_in,
      0,
      dist,
      pos,
      make_frame(n),
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      adhesion,
      geoms,
      collision_pairid_in[contact_tid],
      worldid,
      contact_dist_out,
      contact_pos_out,
      contact_frame_out,
      contact_includemargin_out,
      contact_friction_out,
      contact_solref_out,
      contact_solreffriction_out,
      contact_solimp_out,
      contact_dim_out,
      contact_geom_out,
      contact_efc_address_out,
      contact_worldid_out,
      contact_type_out,
      contact_geomcollisionid_out,
      contact_adhesion_out,
      nacon_out,
    )


def prepare_sdf_context(d: Data, ctx: CollisionContext):
  """Allocates the list of SDF collision candidates."""
  ctx.nsdf = wp.empty(1, dtype=int)
  ctx.sdf_collisionid = wp.empty(d.naconmax, dtype=int)


@event_scope
def sdf_narrowphase(m: Model, d: Data, ctx: CollisionContext):
  ngeom_types = len(GeomType)
  has_mesh_sdf = bool(m.geom_pair_type_count[upper_trid_index(ngeom_types, GeomType.MESH.value, GeomType.SDF.value)])
  has_sdf_sdf = any(
    m.geom_pair_type_count[upper_trid_index(ngeom_types, g.value, GeomType.SDF.value)]
    for g in GeomType
    if g <= GeomType.SDF and g != GeomType.MESH and g != GeomType.HFIELD
  )
  if not (has_mesh_sdf or has_sdf_sdf):
    return

  if ctx.sdf_collisionid is None:
    prepare_sdf_context(d, ctx)

  # list the SDF candidates so that the threads of a warp do not wait for one candidate
  ctx.nsdf.zero_()
  wp.launch(
    _sdf_candidates,
    dim=d.naconmax,
    inputs=[m.geom_type, d.naconmax, d.ncollision, ctx.collision_pair],
    outputs=[ctx.nsdf, ctx.sdf_collisionid],
  )

  if has_mesh_sdf:
    wp.launch(
      _mesh_sdf_narrowphase,
      dim=d.naconmax,
      inputs=[
        m.bvh_child,
        m.bvh_nodeid,
        m.bvh_aabb,
        m.oct_child,
        m.oct_aabb,
        m.oct_coeff,
        m.geom_type,
        m.geom_condim,
        m.geom_dataid,
        m.geom_priority,
        m.geom_solmix,
        m.geom_solref,
        m.geom_solimp,
        m.geom_size,
        m.geom_aabb,
        m.geom_friction,
        m.geom_margin,
        m.geom_gap,
        m.geom_adhesion,
        m.mesh_vertadr,
        m.mesh_faceadr,
        m.mesh_bvhadr,
        m.mesh_octadr,
        m.mesh_vert,
        m.mesh_face,
        m.pair_dim,
        m.pair_solref,
        m.pair_solreffriction,
        m.pair_solimp,
        m.pair_margin,
        m.pair_gap,
        m.pair_adhesion,
        m.pair_friction,
        m.plugin,
        m.plugin_attr,
        m.geom_plugin_index,
        d.geom_xpos,
        d.geom_xquat,
        d.naconmax,
        ctx.collision_pair,
        ctx.collision_pairid,
        ctx.collision_worldid,
        ctx.nsdf,
        ctx.sdf_collisionid,
        m.opt.sdf_iterations,
      ],
      outputs=[
        d.contact.dist,
        d.contact.pos,
        d.contact.frame,
        d.contact.includemargin,
        d.contact.friction,
        d.contact.solref,
        d.contact.solreffriction,
        d.contact.solimp,
        d.contact.dim,
        d.contact.geom,
        d.contact.efc_address,
        d.contact.worldid,
        d.contact.type,
        d.contact.geomcollisionid,
        d.contact.adhesion,
        d.nacon,
      ],
    )

  if has_sdf_sdf:
    wp.launch(
      _sdf_narrowphase,
      dim=(d.naconmax, m.opt.sdf_initpoints),
      inputs=[
        m.oct_child,
        m.oct_aabb,
        m.oct_coeff,
        m.geom_type,
        m.geom_condim,
        m.geom_dataid,
        m.geom_priority,
        m.geom_solmix,
        m.geom_solref,
        m.geom_solimp,
        m.geom_size,
        m.geom_aabb,
        m.geom_friction,
        m.geom_margin,
        m.geom_gap,
        m.geom_adhesion,
        m.mesh_octadr,
        m.pair_dim,
        m.pair_solref,
        m.pair_solreffriction,
        m.pair_solimp,
        m.pair_margin,
        m.pair_gap,
        m.pair_adhesion,
        m.pair_friction,
        m.plugin,
        m.plugin_attr,
        m.geom_plugin_index,
        d.geom_xpos,
        d.geom_xquat,
        d.naconmax,
        ctx.collision_pair,
        ctx.collision_pairid,
        ctx.collision_worldid,
        ctx.nsdf,
        ctx.sdf_collisionid,
        m.opt.sdf_iterations,
        # initial line-search step; passed at launch so NVRTC 13.1+ cannot unroll the line search
        2.0,
      ],
      outputs=[
        d.contact.dist,
        d.contact.pos,
        d.contact.frame,
        d.contact.includemargin,
        d.contact.friction,
        d.contact.solref,
        d.contact.solreffriction,
        d.contact.solimp,
        d.contact.dim,
        d.contact.geom,
        d.contact.efc_address,
        d.contact.worldid,
        d.contact.type,
        d.contact.geomcollisionid,
        d.contact.adhesion,
        d.nacon,
      ],
    )
