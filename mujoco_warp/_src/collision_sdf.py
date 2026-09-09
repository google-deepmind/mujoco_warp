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
from mujoco_warp._src.collision_core import contact_margin_gap
from mujoco_warp._src.collision_core import contact_material_params
from mujoco_warp._src.collision_core import write_contact
from mujoco_warp._src.math import make_frame
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


@wp.struct
class OptimizationParams:
  rel_mat: wp.mat33
  rel_pos: wp.vec3
  attr1: vec_pluginattr
  attr2: vec_pluginattr


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

  elif g_type == GeomType.SDF and mesh_id != -1:
    octadr = mesh_octadr[mesh_id]
    volume_data.center = oct_aabb[octadr, 0]
    volume_data.half_size = oct_aabb[octadr, 1]
    volume_data.root = octadr
    volume_data.oct_aabb = oct_aabb
    volume_data.oct_child = oct_child
    volume_data.oct_coeff = oct_coeff
    volume_data.valid = True

  elif g_type == GeomType.MESH and mesh_id != -1 and mesh_octadr[mesh_id] != -1:
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
def transform_aabb(aabb_pos: wp.vec3, aabb_size: wp.vec3, pos: wp.vec3, ori: wp.mat33) -> Tuple[wp.vec3, wp.vec3]:
  c = ori * aabb_pos + pos
  h = wp.vec3(
    wp.abs(ori[0, 0]) * aabb_size[0] + wp.abs(ori[0, 1]) * aabb_size[1] + wp.abs(ori[0, 2]) * aabb_size[2],
    wp.abs(ori[1, 0]) * aabb_size[0] + wp.abs(ori[1, 1]) * aabb_size[1] + wp.abs(ori[1, 2]) * aabb_size[2],
    wp.abs(ori[2, 0]) * aabb_size[0] + wp.abs(ori[2, 1]) * aabb_size[1] + wp.abs(ori[2, 2]) * aabb_size[2],
  )
  return c - h, c + h


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
  if k1 != 0.0:
    denom = k1
  else:
    denom = 1e-12
  return k0 * (k0 - 1.0) / denom


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
  if wp.max(a) < 0:
    return radial_field(a, p, size)
  z = wp.vec3(0.0, 0.0, 0.0)
  b = wp.max(a, z)
  c = wp.norm_l2(b)
  g = wp.cw_mul(wp.div(b, c), wp.cw_div(p, wp.abs(p)))
  if a[0] <= 0:
    g[0] = 0.0
  if a[1] <= 0:
    g[1] = 0.0
  if a[2] <= 0:
    g[2] = 0.0
  return g


@wp.func
def grad_ellipsoid(p: wp.vec3, size: wp.vec3) -> wp.vec3:
  a = wp.vec3(p[0] / size[0], p[1] / size[1], p[2] / size[2])
  b = wp.vec3(a[0] / size[0], a[1] / size[1], a[2] / size[2])
  k0 = wp.length(a)
  k1 = wp.length(b)
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
  return raw_grad / wp.length(raw_grad)


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
      inv_cell_size = wp.cw_div(wp.vec3(1.0), vmax - vmin)
      for j in range(8):
        if not grad:
          rx[j] = (
            (coord[0] if j & 1 else 1.0 - coord[0])
            * (coord[1] if j & 2 else 1.0 - coord[1])
            * (coord[2] if j & 4 else 1.0 - coord[2])
          )
        else:
          rx[j] = (
            (1.0 if j & 1 else -1.0)
            * (coord[1] if j & 2 else 1.0 - coord[1])
            * (coord[2] if j & 4 else 1.0 - coord[2])
            * inv_cell_size[0]
          )
          ry[j] = (
            (coord[0] if j & 1 else 1.0 - coord[0])
            * (1.0 if j & 2 else -1.0)
            * (coord[2] if j & 4 else 1.0 - coord[2])
            * inv_cell_size[1]
          )
          rz[j] = (
            (coord[0] if j & 1 else 1.0 - coord[0])
            * (coord[1] if j & 2 else 1.0 - coord[1])
            * (1.0 if j & 4 else -1.0)
            * inv_cell_size[2]
          )
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
  node, weights = find_oct(volume_data.oct_child, volume_data.oct_aabb, point, grad=True, root=volume_data.root)
  grad_x = float(0.0)
  grad_y = float(0.0)
  grad_z = float(0.0)
  if node != -1:
    coeff = volume_data.oct_coeff[node]
    grad_x = wp.dot(weights[0], coeff)
    grad_y = wp.dot(weights[1], coeff)
    grad_z = wp.dot(weights[2], coeff)
  if dist0 > 0.0:
    dir_to_box = wp.normalize(xyz - point) if dist0 > 1e-6 else wp.vec3(0.0)
    if node == -1:
      return dir_to_box
    r = xyz - volume_data.center
    q0 = wp.abs(r[0]) - volume_data.half_size[0]
    q1 = wp.abs(r[1]) - volume_data.half_size[1]
    q2 = wp.abs(r[2]) - volume_data.half_size[2]
    gx = dir_to_box[0] if q0 > 0.0 else grad_x
    gy = dir_to_box[1] if q1 > 0.0 else grad_y
    gz = dir_to_box[2] if q2 > 0.0 else grad_z
    return wp.normalize(wp.vec3(gx, gy, gz))
  grad = wp.vec3(grad_x, grad_y, grad_z)
  grad_len = wp.length(grad)
  if grad_len > 1e-6:
    return grad / grad_len
  return grad


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
  elif type == GeomType.MESH and volume_data.valid:
    return sample_volume_sdf(p, volume_data)
  elif type == GeomType.SDF:
    if sdf_type == -1:
      return sample_volume_sdf(p, volume_data)
    else:
      return user_sdf(p, attr, sdf_type)
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
  elif type == GeomType.MESH and volume_data.valid:
    return sample_volume_grad(p, volume_data)
  elif type == GeomType.SDF:
    if sdf_type == -1:
      return sample_volume_grad(p, volume_data)
    else:
      return user_sdf_grad(p, attr, sdf_type)
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
) -> wp.vec3:
  A = sdf(type1, p1, params.attr1, sdf_type1, volume_data1)
  B = sdf(GeomType.SDF, p2, params.attr2, sdf_type2, volume_data2)
  grad1 = sdf_grad(type1, p1, params.attr1, sdf_type1, volume_data1)
  grad2 = sdf_grad(GeomType.SDF, p2, params.attr2, sdf_type2, volume_data2)
  grad1_transformed = wp.transpose(params.rel_mat) * grad1
  if sfd_intersection:
    if A > B:
      return grad1_transformed
    else:
      return grad2
  else:
    gradient = grad2 + grad1_transformed
    max_val = wp.max(A, B)
    if A > B:
      max_grad = grad1_transformed
    else:
      max_grad = grad2
    sign = wp.sign(max_val)
    gradient += max_grad * sign
    return gradient


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
  max_step: float,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
) -> Tuple[float, wp.vec3]:
  amin = 1e-4
  rho = 0.5
  c = 0.1
  dist = float(1e10)
  for i in range(niter):
    alpha = wp.clamp(max_step, 0.1, 2.0)
    x2 = wp.vec3(x[0], x[1], x[2])
    x1 = params.rel_mat * x2 + params.rel_pos
    grad = compute_grad(type1, x1, x2, params, sdf_type1, sdf_type2, sfd_intersection, volume_data1, volume_data2)
    dist0 = clearance(
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
  max_step: float,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
) -> Tuple[float, wp.vec3, wp.vec3]:
  params = OptimizationParams()
  params.rel_mat = wp.transpose(rot1) * rot2
  params.rel_pos = wp.transpose(rot1) * (pos2 - pos1)
  params.attr1 = attr1
  params.attr2 = attr2
  dist, x = gradient_step(
    type1,
    x0_initial,
    params,
    sdf_type1,
    sdf_type2,
    sdf_iterations,
    False,
    max_step,
    volume_data1,
    volume_data2,
  )
  dist, x = gradient_step(type1, x, params, sdf_type1, sdf_type2, 1, True, max_step, volume_data1, volume_data2)
  x_1 = params.rel_mat * x + params.rel_pos
  grad1 = sdf_grad(type1, x_1, params.attr1, sdf_type1, volume_data1)
  grad1 = wp.transpose(params.rel_mat) * grad1
  grad1 = wp.normalize(grad1)
  grad2 = sdf_grad(GeomType.SDF, x, params.attr2, sdf_type2, volume_data2)
  grad2 = wp.normalize(grad2)
  n = grad1 - grad2
  n = wp.normalize(n)
  pos = rot2 * x + pos2
  n = rot2 * n
  pos3 = pos - n * dist / 2.0
  return dist, pos3, n


@wp.func
def frank_wolfe_triangle(
  # In:
  v0: wp.vec3,
  v1: wp.vec3,
  v2: wp.vec3,
  volume_data: VolumeData,
  sdf_type: int,
  attr: vec_pluginattr,
  n_iter: int,
  sp: int,
) -> Tuple[float, wp.vec3, wp.vec3]:
  u = halton(sp + 1, 2)
  v = halton(sp + 1, 3)
  if u + v > 1.0:
    u = 1.0 - u
    v = 1.0 - v
  b0 = 1.0 - u - v
  b1 = u
  b2 = v
  x = v0 * b0 + v1 * b1 + v2 * b2

  for k in range(n_iter):
    g = wp.vec3(0.0)
    if sdf_type == -1:
      g = sample_volume_grad(x, volume_data)
    else:
      g = user_sdf_grad(x, attr, sdf_type)
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
    gamma = 2.0 / float(k + 2)
    x = x + (s - x) * gamma

  dist = float(0.0)
  normal = wp.vec3(0.0)
  if sdf_type == -1:
    dist = sample_volume_sdf(x, volume_data)
    normal = sample_volume_grad(x, volume_data)
  else:
    dist = user_sdf(x, attr, sdf_type)
    normal = wp.normalize(user_sdf_grad(x, attr, sdf_type))
  return dist, x, normal


@wp.kernel
def _filter_sdf_pairs(
  # Model:
  geom_type: wp.array[int],
  # Data in:
  naconmax_in: int,
  ncollision_in: wp.array[int],
  # In:
  collision_pair_in: wp.array[wp.vec2i],
  # Out:
  nsdf_collision_out: wp.array[int],
  sdf_collision_tid_out: wp.array[int],
):
  tid = wp.tid()
  limit = wp.min(ncollision_in[0], naconmax_in)
  if tid >= limit:
    return

  geoms = collision_pair_in[tid]
  t1 = geom_type[geoms[0]]
  t2 = geom_type[geoms[1]]

  if t1 == GeomType.SDF or t2 == GeomType.SDF:
    pair_idx = wp.atomic_add(nsdf_collision_out, 0, 1)
    if pair_idx < naconmax_in:
      sdf_collision_tid_out[pair_idx] = tid


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
  bvh_mesh_id: wp.array[wp.uint64],
  geom_plugin_index: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  naconmax_in: int,
  # In:
  nsdf_collision_in: wp.array[int],
  grid_stride_in: int,
  collision_pair_in: wp.array[wp.vec2i],
  collision_pairid_in: wp.array[wp.vec2i],
  collision_worldid_in: wp.array[int],
  sdf_collision_tid_in: wp.array[int],
  sdf_initpoints: int,
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
  i, tid = wp.tid()
  if i >= sdf_initpoints:
    return

  limit = wp.min(nsdf_collision_in[0], naconmax_in)
  for slot in range(tid, limit, grid_stride_in):
    contact_tid = sdf_collision_tid_in[slot]
    worldid = collision_worldid_in[contact_tid]
    geoms = collision_pair_in[contact_tid]
    g1 = geoms[0]
    g2 = geoms[1]
    t1 = geom_type[g1]
    t2 = geom_type[g2]
    if t1 == GeomType.SDF and t2 != GeomType.SDF:
      g1, g2 = g2, g1
      t1, t2 = t2, t1
      geoms = wp.vec2i(g1, g2)

    pairid_in = collision_pairid_in[contact_tid]
    pairid = pairid_in[0]

    margin, gap = contact_margin_gap(geom_margin, geom_gap, pair_margin, pair_gap, geoms, pairid, worldid)
    margin_gap = margin + wp.max(0.0, gap)

    pos1 = geom_xpos_in[worldid, g1]
    rot1 = geom_xmat_in[worldid, g1]
    pos2 = geom_xpos_in[worldid, g2]
    rot2 = geom_xmat_in[worldid, g2]

    g1_to_g2_rot = wp.transpose(rot1) * rot2
    g1_to_g2_pos = wp.transpose(rot1) * (pos2 - pos1)

    aabb_id = worldid % geom_aabb.shape[0]
    aabb1_pos = geom_aabb[aabb_id, g1, 0]
    aabb1_size = geom_aabb[aabb_id, g1, 1]
    aabb2_min, aabb2_max = transform_aabb(geom_aabb[aabb_id, g2, 0], geom_aabb[aabb_id, g2, 1], g1_to_g2_pos, g1_to_g2_rot)
    intersect_min = wp.max(aabb1_pos - aabb1_size, aabb2_min)
    intersect_max = wp.min(aabb1_pos + aabb1_size, aabb2_max)

    if (
      intersect_min[0] - margin_gap > intersect_max[0]
      or intersect_min[1] - margin_gap > intersect_max[1]
      or intersect_min[2] - margin_gap > intersect_max[2]
    ):
      continue

    intersect_size = intersect_max - intersect_min

    size1 = geom_size[worldid % geom_size.shape[0], g1]
    size2 = geom_size[worldid % geom_size.shape[0], g2]
    g1_plugin = geom_plugin_index[g1]
    g2_plugin = geom_plugin_index[g2]

    dataid_setid = worldid % geom_dataid.shape[0]
    mesh_id1 = geom_dataid[dataid_setid, g1]
    mesh_id2 = geom_dataid[dataid_setid, g2]

    attr1, g1_plugin_id, volume_data1 = get_sdf_params(
      oct_child,
      oct_aabb,
      oct_coeff,
      mesh_octadr,
      plugin,
      plugin_attr,
      t1,
      size1,
      g1_plugin,
      mesh_id1,
    )

    attr2, g2_plugin_id, volume_data2 = get_sdf_params(
      oct_child,
      oct_aabb,
      oct_coeff,
      mesh_octadr,
      plugin,
      plugin_attr,
      t2,
      size2,
      g2_plugin,
      mesh_id2,
    )

    is_mesh1 = t1 == GeomType.MESH and not volume_data1.valid and mesh_id1 >= 0

    contact_dist = float(1e10)
    contact_pos = wp.vec3(0.0)
    contact_normal = wp.vec3(0.0)
    found_contact = False

    if is_mesh1:
      bvh_id = bvh_mesh_id[mesh_id1]

      u = halton(i + 1, 2)
      v = halton(i + 1, 3)
      w = halton(i + 1, 5)
      p_g1 = intersect_min + wp.cw_mul(intersect_size, wp.vec3(u, v, w))

      g2_to_g1_rot = wp.transpose(rot2) * rot1
      g2_to_g1_pos = wp.transpose(rot2) * (pos1 - pos2)

      p_sdf = g2_to_g1_rot * p_g1 + g2_to_g1_pos

      dist0 = float(0.0)
      grad0 = wp.vec3(0.0)
      if g2_plugin_id != -1:
        dist0 = user_sdf(p_sdf, attr2, g2_plugin_id)
        grad0 = user_sdf_grad(p_sdf, attr2, g2_plugin_id)
      elif volume_data2.valid:
        dist0 = sample_volume_sdf(p_sdf, volume_data2)
        grad0 = sample_volume_grad(p_sdf, volume_data2)

      g_len = wp.length(grad0)
      g_norm = wp.vec3(0.0, 0.0, 1.0)
      if g_len > 1e-4:
        g_norm = grad0 * (1.0 / g_len)
      p_surf_sdf = p_sdf - g_norm * dist0
      p_surf_mesh = wp.transpose(g2_to_g1_rot) * (p_surf_sdf - g2_to_g1_pos)

      query_p = p_g1
      if wp.abs(dist0) < 0.15:
        query_p = p_surf_mesh

      sign = float(0.0)
      target_face = int(-1)
      fu = float(0.0)
      fv = float(0.0)
      max_query_dist = margin + 0.10
      success = wp.mesh_query_point(bvh_id, query_p, max_query_dist, sign, target_face, fu, fv)

      if success and target_face != -1:
        v0 = wp.mesh_get_point(bvh_id, target_face * 3 + 0)
        v1 = wp.mesh_get_point(bvh_id, target_face * 3 + 1)
        v2 = wp.mesh_get_point(bvh_id, target_face * 3 + 2)

        t0_sdf = g2_to_g1_rot * v0 + g2_to_g1_pos
        t1_sdf = g2_to_g1_rot * v1 + g2_to_g1_pos
        t2_sdf = g2_to_g1_rot * v2 + g2_to_g1_pos

        dist, x_sdf, n_sdf = frank_wolfe_triangle(t0_sdf, t1_sdf, t2_sdf, volume_data2, g2_plugin_id, attr2, sdf_iterations, i)

        pos_world = rot2 * x_sdf + pos2
        normal_world = -(rot2 * n_sdf)
        contact_pos = pos_world + normal_world * (dist * 0.5)
        contact_normal = normal_world
        contact_dist = dist
        found_contact = True
    else:
      x_g2 = intersect_min + wp.cw_mul(intersect_size, wp.vec3(halton(i, 2), halton(i, 3), halton(i, 5)))
      x = rot1 * x_g2 + pos1
      x0_initial = wp.transpose(rot2) * (x - pos2)
      aabb_extent = wp.max(intersect_size[0], wp.max(intersect_size[1], intersect_size[2]))
      char_size = wp.max(wp.length(size1), wp.length(size2))
      max_step = wp.clamp(wp.max(aabb_extent, char_size), 0.1, 2.0)
      dist, pos, n = gradient_descent(
        t1,
        x0_initial,
        attr1,
        attr2,
        pos1,
        rot1,
        pos2,
        rot2,
        g1_plugin_id,
        g2_plugin_id,
        sdf_iterations,
        max_step,
        volume_data1,
        volume_data2,
      )
      contact_dist = dist
      contact_pos = pos
      contact_normal = n
      found_contact = True

    if found_contact:
      condim, friction, solref, solreffriction, solimp, adhesion = contact_material_params(
        geom_condim,
        geom_priority,
        geom_solmix,
        geom_solref,
        geom_solimp,
        geom_friction,
        geom_adhesion,
        pair_dim,
        pair_solref,
        pair_solreffriction,
        pair_solimp,
        pair_adhesion,
        pair_friction,
        geoms,
        pairid,
        worldid,
      )
      write_contact(
        naconmax_in,
        0,
        contact_dist,
        contact_pos,
        make_frame(contact_normal),
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        adhesion,
        geoms,
        pairid_in,
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


def _sdf_grid_size(kernel, naconmax: int, sdf_initpoints: int, device) -> int:
  if device.is_cpu:
    return naconmax
  block_size, min_grid_size = wp.get_suggested_block_size(kernel, device)
  target_total_threads = 4 * block_size * min_grid_size
  return max(1, min(naconmax, target_total_threads // max(1, sdf_initpoints)))


@event_scope
def sdf_narrowphase(m: Model, d: Data, ctx: CollisionContext):
  if m.opt.sdf_initpoints <= 0 or d.naconmax <= 0:
    return

  if ctx.sdf_collision_tid is None:
    ctx.sdf_collision_tid = wp.empty(d.naconmax, dtype=int)
    ctx.nsdf_collision = wp.zeros(1, dtype=int)

  ctx.nsdf_collision.zero_()
  wp.launch(
    _filter_sdf_pairs,
    dim=d.naconmax,
    inputs=[
      m.geom_type,
      d.naconmax,
      d.ncollision,
      ctx.collision_pair,
    ],
    outputs=[
      ctx.nsdf_collision,
      ctx.sdf_collision_tid,
    ],
  )

  grid_width = _sdf_grid_size(_sdf_narrowphase, d.naconmax, m.opt.sdf_initpoints, d.ncollision.device)

  wp.launch(
    _sdf_narrowphase,
    dim=(m.opt.sdf_initpoints, grid_width),
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
      m.bvh_mesh_id,
      m.geom_plugin_index,
      d.geom_xpos,
      d.geom_xmat,
      d.naconmax,
      ctx.nsdf_collision,
      grid_width,
      ctx.collision_pair,
      ctx.collision_pairid,
      ctx.collision_worldid,
      ctx.sdf_collision_tid,
      m.opt.sdf_initpoints,
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
