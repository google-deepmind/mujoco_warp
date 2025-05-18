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

import warp as wp
from .types import GeomType, SDFType
from .collision_primitive import Geom
from .collision_primitive import _geom
from .collision_primitive import contact_params
from .collision_primitive import write_contact
from .types import Data
from .types import Model
from .types import vec5
from .math import make_frame
from . import math

@wp.struct
class GradientState:
  dist: float
  x: wp.vec3

@wp.struct
class OptimizationParams:
  rel_mat: wp.mat33
  rel_pos: wp.vec3
  attr1: wp.vec3
  attr2: wp.vec3

@wp.struct
class AABB:
  min: wp.vec3
  max: wp.vec3

@wp.func
def transform_aabb(
  aabb_pos: wp.vec3, aabb_size: wp.vec3, pos: wp.vec3, ori: wp.mat33
) -> AABB:
  aabb = AABB()
  aabb.max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)
  aabb.min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)

  for i in range(8):
    corner = wp.vec3(aabb_size.x, aabb_size.y, aabb_size.z)
    if i % 2 == 0:
      corner.x = -corner.x
    if (i // 2) % 2 == 0:
      corner.y = -corner.y
    if i < 4:
      corner.z = -corner.z
    corner_world = ori * (corner + aabb_pos) + pos
    aabb.max = wp.max(aabb.max, corner_world)
    aabb.min = wp.min(aabb.min, corner_world)

  return aabb

@wp.func
def sphere(p: wp.vec3, size: wp.vec3) -> float:
  return wp.length(p) - size[0]

@wp.func
def ellipsoid(p: wp.vec3, size: wp.vec3) -> float:
  scaled_p = wp.vec3(p[0] / size[0], p[1] / size[1], p[2] / size[2])
  k0 = wp.length(scaled_p)
  k1 = wp.length(
    wp.vec3(p[0] / (size[0] ** 2.0), p[1] / (size[1] ** 2.0), p[2] / (size[2] ** 2.0))
  )
  if k1 != 0.0:
    denom = k1
  else:
    denom = 1e-12
  return k0 * (k0 - 1.0) / denom

@wp.func
def nut(p: wp.vec3, attr: wp.vec3) -> float:
    screw = 12.0
    radius2 = wp.sqrt(p[0]*p[0] + p[1]*p[1]) - attr[0]
    sqrt12 = wp.sqrt(2.0)/2.0

    azimuth = wp.atan2(p[1], p[0])
    triangle = wp.abs((p[2] * screw - azimuth/(wp.pi*2.0)) - wp.floor(p[2] * screw - azimuth/(wp.pi*2.0)) - 0.5)
    thread2 = (radius2 - triangle / screw) * sqrt12

    cone2 = (p[2] - radius2) * sqrt12
    hole = wp.min(thread2, cone2 + 0.5 * sqrt12)
    hole = wp.min(hole, -cone2 - 0.05 * sqrt12)

    k = 6.0 / wp.pi / 2.0
    angle = -wp.floor((wp.atan2(p[1], p[0])) * k + 0.5) / k
    s = wp.vec2(wp.sin(angle), wp.sin(angle + wp.pi * 0.5))
    rot_point = wp.vec2(s.y * p.x - s.x * p.y, 
                        s.x * p.x + s.y * p.y)

    head = rot_point[0] - 0.5
    head = wp.max(head, wp.abs(rot_point[1] + 0.25) - 0.25)
    head = wp.max(head, (rot_point[1] + radius2 - 0.22) * sqrt12)
    
    return wp.max(head, -hole)


@wp.func
def grad_sphere(p: wp.vec3) -> wp.vec3:
  c = wp.length(p)
  if c > 1e-9:
    return p / c
  else:
    wp.vec3(0.0)


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
def grad_nut(
    x: wp.vec3, attr: wp.vec3
) -> wp.vec3:
    grad = wp.vec3()
    eps = 1e-4
    
    f_original =  nut(x, attr)
    x_plus = wp.vec3(x[0] + eps, x[1], x[2])
    f_plus =  nut(x_plus, attr)
    grad[0] = (f_plus - f_original) / eps

    x_plus = wp.vec3(x[0], x[1] + eps, x[2])
    f_plus =  nut(x_plus, attr)
    grad[1] = (f_plus - f_original) / eps

    x_plus = wp.vec3(x[0], x[1], x[2] + eps)
    f_plus =  nut(x_plus, attr)
    grad[2] = (f_plus - f_original) / eps
    return grad

def sdf(type: int, sdf_type: int = 0):
  @wp.func
  def _sdf(p: wp.vec3, attr: wp.vec3) -> float:
    if wp.static(type == GeomType.SPHERE.value):
        return sphere(p, attr)
    elif wp.static(type == GeomType.ELLIPSOID.value):
        return ellipsoid(p, attr)
    elif wp.static(type == GeomType.SDF.value):
        if wp.static(sdf_type == SDFType.NUT.value):
          return nut(p, attr)
  return _sdf


def sdf_grad(type: int, sdf_type: int = 0):
  @wp.func
  def _sdf_grad(p: wp.vec3, attr: wp.vec3) -> wp.vec3:
    if wp.static(type == GeomType.SPHERE.value):
        return grad_sphere(p)
    elif wp.static(type == GeomType.ELLIPSOID.value):
        return grad_ellipsoid(p, attr)
    elif wp.static(type == GeomType.SDF.value):
        if wp.static(sdf_type == SDFType.NUT.value):
          return grad_nut(p, attr)

  return _sdf_grad

def clearance(type1: int, type2: int, sdf_type1: int, sdf_type2: int, sfd_intersection: bool = False):
    @wp.func
    def _clearance(p1: wp.vec3, p2: wp.vec3, s1: wp.vec3, s2: wp.vec3) -> float:
        s = wp.static(sdf(type1, sdf_type1))(p1, s1)
        e = wp.static(sdf(type2, sdf_type2))(p2, s2)
        if sfd_intersection:
            return wp.max(s, e)
        else:
            return s + e + wp.abs(wp.max(s, e))
    return _clearance

def compute_grad(type1: int, type2: int,  sdf_type1: int, sdf_type2: int, sfd_intersection: bool = False):
    @wp.func
    def _compute_grad(
        p1: wp.vec3, p2: wp.vec3, params: OptimizationParams
    ) -> wp.vec3:
        A = wp.static(sdf(type1, sdf_type1))(p1, params.attr1)
        B = wp.static(sdf(type2, sdf_type2))(p2, params.attr2)
        grad1 = wp.static(sdf_grad(type1, sdf_type1))(p1, params.attr1)
        grad2 = wp.static(sdf_grad(type2, sdf_type2))(p2, params.attr2)
        grad1_transformed = params.rel_mat * grad1
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
    return _compute_grad

def gradient_step(type1: int, type2: int,  sdf_type1: int, sdf_type2: int, sfd_intersection: bool = False):
    @wp.func
    def _gradient_step(
        state: GradientState,
        params: OptimizationParams,
    ) -> GradientState:
        amin = 1e-4
        rho = 0.5
        c = 0.1
        alpha = float(2.0)

        x0 = state.x
        x1 = params.rel_mat * x0 + params.rel_pos
        grad = wp.static(compute_grad(type1, type2, sdf_type1, sdf_type2,  sfd_intersection))(x1, x0, params)
        dist0 = wp.static(clearance(type1, type2, sdf_type1, sdf_type2, sfd_intersection))(
            x1, x0, params.attr1, params.attr2
        )
        grad_dot = wp.dot(grad, grad)

        if grad_dot < 1e-12:
            return GradientState(dist0, x0)

        wolfe = -c * alpha * grad_dot
        best_candidate = x0
        best_value = dist0

        while True:
            alpha *= rho
            wolfe *= rho

            candidate = x0 - grad * alpha
            x1_candidate = params.rel_mat * candidate + params.rel_pos
            value = wp.static(clearance(type1, type2, sdf_type1, sdf_type2, sfd_intersection))(
                x1_candidate, candidate, params.attr1, params.attr2
            )

            if alpha <= amin or (value - dist0) <= wolfe:
                best_candidate = candidate
                best_value = value
                break

        if best_value < dist0:
            new_state = GradientState()
            new_state.dist = best_value
            new_state.x = best_candidate
            return new_state
        else:
            original_state = GradientState()
            original_state.dist = dist0
            original_state.x = x0
            return original_state
    return _gradient_step

def gradient_descent(type1: int, type2: int, sdf_type1: int = 0, sdf_type2: int = 0):
  @wp.func
  def _gradient_descent(
    x: wp.vec3,
    niter: int,
    params: OptimizationParams,
  ):

    state = GradientState(1e10, x)

    for _ in range(niter):
      state = wp.static(gradient_step(type1, type2, sdf_type1, sdf_type2,))(state, params)
      #todo early break

    state = wp.static(gradient_step(type1, type2, sdf_type1, sdf_type2, True))(state, params)
    return state.dist, state.x

  return _gradient_descent

@wp.func
def sphere_ellipsoid(
   # Data in:
  nconmax_in: int,
  # In:
  s: Geom,
  e: Geom,
  aabb1: AABB,
  aabb2: AABB,
  worldid: int,
  margin: float,
  gap: float,
  condim: int,
  friction: vec5,
  solref: wp.vec2f,
  solreffriction: wp.vec2f,
  solimp: vec5,
  geoms: wp.vec2i,
  # Data out:
  ncon_out: wp.array(dtype=int),
  contact_dist_out: wp.array(dtype=float),
  contact_pos_out: wp.array(dtype=wp.vec3),
  contact_frame_out: wp.array(dtype=wp.mat33),
  contact_includemargin_out: wp.array(dtype=float),
  contact_friction_out: wp.array(dtype=vec5),
  contact_solref_out: wp.array(dtype=wp.vec2),
  contact_solreffriction_out: wp.array(dtype=wp.vec2),
  contact_solimp_out: wp.array(dtype=vec5),
  contact_dim_out: wp.array(dtype=int),
  contact_geom_out: wp.array(dtype=wp.vec2i),
  contact_worldid_out: wp.array(dtype=int),): 
    
    
    params = OptimizationParams()
    static_type1 =  wp.static(GeomType.SPHERE.value)
    static_type2 =  wp.static(GeomType.ELLIPSOID.value)
    params.rel_mat = wp.transpose(s.rot) * e.rot
    params.rel_pos = wp.transpose(s.rot) * (e.pos - s.pos)
    params.attr1 = s.size 
    params.attr2 = e.size

    # min of the first AABB
    x1 = aabb1.min
    # min of the second AABB
    x2 = aabb2.min
    # min of the intersection
    x = wp.vec3(wp.max(x1[0], x2[0]),wp.max(x1[1], x2[1]),wp.max(x1[2], x2[2])
                )
    x0_transformed = wp.transpose(e.rot) * (x - e.pos)
    dist, pos = wp.static(gradient_descent(static_type1, static_type2))(x0_transformed, 10, params)

    pos_s = params.rel_mat * pos + params.rel_pos
    grad1 = wp.static(sdf_grad(static_type1))(pos_s, params.attr1)
    grad2 = wp.static(sdf_grad(static_type1))(pos, params.attr2) 
    n = grad1 - grad2
    pos = e.rot * pos + e.pos
    n = e.rot * n
    f = wp.normalize(n)
    pos3 = pos - f* dist/2.0
    write_contact(
    nconmax_in,
    dist,
    pos3,
    make_frame(n),
    margin,
    gap,
    condim,
    friction,
    solref,
    solreffriction,
    solimp,
    geoms,
    worldid,
    ncon_out,
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
    contact_worldid_out,
  )

def sdf_sdf(type1: int, type2: int, sdf_type1: int = 0, sdf_type2: int = 0):  
  @wp.func
  def _sdf_sdf (
    # Data in:
    nconmax_in: int,
    # In:
    s1: Geom,
    s2: Geom,
    aabb1: AABB,
    aabb2: AABB,
    attr1: wp.vec3f,
    attr2: wp.vec3f,
    geom_pos1: wp.vec3,
    geom_mat1: wp.mat33,
    geom_pos2: wp.vec3,
    geom_mat2: wp.mat33,
    worldid: int,
    margin: float,
    gap: float,
    condim: int,
    friction: vec5,
    solref: wp.vec2f,
    solreffriction: wp.vec2f,
    solimp: vec5,
    geoms: wp.vec2i,
    # Data out:
    ncon_out: wp.array(dtype=int),
    contact_dist_out: wp.array(dtype=float),
    contact_pos_out: wp.array(dtype=wp.vec3),
    contact_frame_out: wp.array(dtype=wp.mat33),
    contact_includemargin_out: wp.array(dtype=float),
    contact_friction_out: wp.array(dtype=vec5),
    contact_solref_out: wp.array(dtype=wp.vec2),
    contact_solreffriction_out: wp.array(dtype=wp.vec2),
    contact_solimp_out: wp.array(dtype=vec5),
    contact_dim_out: wp.array(dtype=int),
    contact_geom_out: wp.array(dtype=wp.vec2i),
    contact_worldid_out: wp.array(dtype=int),):

      rot1 = math.mul(s1.rot, math.transpose(geom_mat1))
      pos1 = wp.sub(s1.pos, math.mul(rot1, geom_pos1))
      rot2 = math.mul(s2.rot, math.transpose(geom_mat2))
      pos2 = wp.sub(s2.pos, math.mul(rot2, geom_pos2))
    
      params = OptimizationParams()
      params.rel_mat = wp.transpose(rot1) * rot2
      params.rel_pos = wp.transpose(rot1) * (pos2 - pos1)
      params.attr1 = attr1
      params.attr2 = attr2

      x1 = aabb1.min
      x2 = aabb2.min
      x = wp.vec3(wp.max(x1[0], x2[0]),wp.max(x1[1], x2[1]),wp.max(x1[2], x2[2]))
      x0_transformed = wp.transpose(rot2) * (x - pos2)
      dist, pos = wp.static(gradient_descent(type1, type2, sdf_type1, sdf_type2))(x0_transformed, 10, params)
      pos_s = params.rel_mat * pos + params.rel_pos
      grad1 = wp.static(sdf_grad(type1, sdf_type1))(pos_s, params.attr1)
      grad2 = wp.static(sdf_grad(type1, sdf_type2))(pos, params.attr2) 
      n = grad1 - grad2
      pos = rot2 * pos + pos2
      n = rot2 * n
      f = wp.normalize(n)
      pos3 = pos - f* dist/2.0 #todo
      
      write_contact(
      nconmax_in,
      dist,
      pos3,
      make_frame(n),
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
      worldid,
      ncon_out,
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
      contact_worldid_out,
    )
  return _sdf_sdf


@wp.kernel
def _sdf_narrowphase(
  # Model:
  geom_type: wp.array(dtype=int),
  sdf_type: wp.array(dtype=int),
  geom_sdf_plugin_attr: wp.array(dtype=wp.vec3f),
  geom_condim: wp.array(dtype=int),
  geom_dataid: wp.array(dtype=int),
  geom_priority: wp.array(dtype=int),
  geom_solmix: wp.array(dtype=float),
  geom_solref: wp.array(dtype=wp.vec2),
  geom_solimp: wp.array(dtype=vec5),
  geom_size: wp.array(dtype=wp.vec3),
  geom_friction: wp.array(dtype=wp.vec3),
  geom_margin: wp.array(dtype=float),
  geom_gap: wp.array(dtype=float),
  mesh_vertadr: wp.array(dtype=int),
  mesh_vertnum: wp.array(dtype=int),
  pair_dim: wp.array(dtype=int),
  pair_solref: wp.array(dtype=wp.vec2),
  pair_solreffriction: wp.array(dtype=wp.vec2),
  pair_solimp: wp.array(dtype=vec5),
  pair_margin: wp.array(dtype=float),
  pair_gap: wp.array(dtype=float),
  pair_friction: wp.array(dtype=vec5),
  geom_aabb : wp.array2d(dtype=wp.vec3),
  geom_pos: wp.array(dtype=wp.vec3),
  geom_quat: wp.array(dtype=wp.quat),

  # Data in:
  nconmax_in: int,
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  collision_pair_in: wp.array(dtype=wp.vec2i),
  collision_pairid_in: wp.array(dtype=int),
  collision_worldid_in: wp.array(dtype=int),
  ncollision_in: wp.array(dtype=int),
  # Data out:
  ncon_out: wp.array(dtype=int),
  contact_dist_out: wp.array(dtype=float),
  contact_pos_out: wp.array(dtype=wp.vec3),
  contact_frame_out: wp.array(dtype=wp.mat33),
  contact_includemargin_out: wp.array(dtype=float),
  contact_friction_out: wp.array(dtype=vec5),
  contact_solref_out: wp.array(dtype=wp.vec2),
  contact_solreffriction_out: wp.array(dtype=wp.vec2),
  contact_solimp_out: wp.array(dtype=vec5),
  contact_dim_out: wp.array(dtype=int),
  contact_geom_out: wp.array(dtype=wp.vec2i),
  contact_worldid_out: wp.array(dtype=int),
):
  tid = wp.tid()
  if tid >= ncollision_in[0]:
    return
  geoms, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
    geom_condim,
    geom_priority,
    geom_solmix,
    geom_solref,
    geom_solimp,
    geom_friction,
    geom_margin,
    geom_gap,
    pair_dim,
    pair_solref,
    pair_solreffriction,
    pair_solimp,
    pair_margin,
    pair_gap,
    pair_friction,
    collision_pair_in,
    collision_pairid_in,
    tid,
  )
  g1 = geoms[0]
  g2 = geoms[1]

  worldid = collision_worldid_in[tid]

  geom1 = _geom(
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_vertnum,
    geom_xpos_in,
    geom_xmat_in,
    worldid,
    g1,
  )
  geom2 = _geom(
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_vertnum,
    geom_xpos_in,
    geom_xmat_in,
    worldid,
    g2,
  )

  type1 = geom_type[g1]
  type2 = geom_type[g2]
  sdf_type1 = sdf_type[g1]
  sdf_type2 = sdf_type[g2]
  attr1 = geom_sdf_plugin_attr[g1]
  attr2 = geom_sdf_plugin_attr[g2]

  aabb_pos = geom_aabb[g1, 0]
  aabb_size = geom_aabb[g1, 1]
  aabb1 = transform_aabb(aabb_pos, aabb_size, geom1.pos, geom1.rot)
  aabb_pos = geom_aabb[g2, 0]
  aabb_size = geom_aabb[g2, 1]
  aabb2 = transform_aabb(aabb_pos, aabb_size, geom2.pos, geom2.rot)
  pos1 = geom_pos[g1]
  quat1 = geom_quat[g1]
  rot1 = math.quat_to_mat(quat1)
  pos2 = geom_pos[g2]
  quat2 = geom_quat[g2]
  rot2 = math.quat_to_mat(quat2)

  if type1 == int(GeomType.SPHERE.value) and type2 == int(GeomType.ELLIPSOID.value):
    sphere_ellipsoid(
      nconmax_in,
      geom1,
      geom2,
      aabb1,
      aabb2,
      worldid,
      margin,
      gap,
      condim,
      friction,
      solref,
      solreffriction,
      solimp,
      geoms,
      ncon_out,
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
      contact_worldid_out,
    )
  else:
      type1s = wp.static(GeomType.SDF.value)
      sdf_type1s = wp.static(SDFType.NUT.value)
      type2s = wp.static(GeomType.SDF.value)
      sdf_type2s = wp.static(SDFType.NUT.value)
      wp.static(sdf_sdf(type1s, type2s, sdf_type1s, sdf_type2s))(
        nconmax_in,
        geom1,
        geom2,
        aabb1,
        aabb2,
        attr1,
        attr2,
        pos1,
        rot1,
        pos2, 
        rot2,
        worldid,
        margin,
        gap,
        condim,
        friction,
        solref,
        solreffriction,
        solimp,
        geoms,
        ncon_out,
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
        contact_worldid_out,
      )

def sdf_narrowphase(m: Model, d: Data):
  wp.launch(
    _sdf_narrowphase,
    dim=d.nconmax,
    inputs=[
      m.geom_type,
      m.geom_sdf_plugin_type,
      m.geom_sdf_plugin_attr,
      m.geom_condim,
      m.geom_dataid,
      m.geom_priority,
      m.geom_solmix,
      m.geom_solref,
      m.geom_solimp,
      m.geom_size,
      m.geom_friction,
      m.geom_margin,
      m.geom_gap,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.pair_dim,
      m.pair_solref,
      m.pair_solreffriction,
      m.pair_solimp,
      m.pair_margin,
      m.pair_gap,
      m.pair_friction,
      m.geom_aabb,
      m.geom_pos,
      m.geom_quat,
      d.nconmax,
      d.geom_xpos,
      d.geom_xmat,
      d.collision_pair,
      d.collision_pairid,
      d.collision_worldid,
      d.ncollision,
    ],
    outputs=[
      d.ncon,
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
      d.contact.worldid,
    ],
  )
