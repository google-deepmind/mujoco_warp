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

from typing import Tuple, Any

import mujoco
import warp as wp

from .types import ConeType
from .types import Data
from .types import Model
from .types import array2df
from .types import array3df
from .types import vec5
from .warp_util import event_scope
from .warp_util import kernel

wp.set_module_options({"enable_backward": False})


def is_sparse(m: mujoco.MjModel):
  if m.opt.jacobian == mujoco.mjtJacobian.mjJAC_AUTO:
    return m.nv >= 60
  return m.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE


@event_scope
def mul_m(
  m: Model,
  d: Data,
  res: wp.array(ndim=2, dtype=wp.float32),
  vec: wp.array(ndim=2, dtype=wp.float32),
  skip: wp.array(ndim=1, dtype=bool),
):
  """Multiply vector by inertia matrix."""

  if not m.opt.is_sparse:

    def tile_mul(adr: int, size: int, tilesize: int):
      # TODO(team): speed up kernel compile time (14s on 2023 Macbook Pro)
      @kernel
      def mul(
        m: Model,
        d: Data,
        leveladr: int,
        res: array3df,
        vec: array3df,
        skip: wp.array(ndim=1, dtype=bool),
      ):
        worldid, nodeid = wp.tid()

        if skip[worldid]:
          return

        dofid = m.qLD_tile[leveladr + nodeid]
        qM_tile = wp.tile_load(
          d.qM[worldid], shape=(tilesize, tilesize), offset=(dofid, dofid)
        )
        vec_tile = wp.tile_load(vec[worldid], shape=(tilesize, 1), offset=(dofid, 0))
        res_tile = wp.tile_zeros(shape=(tilesize, 1), dtype=wp.float32)
        wp.tile_matmul(qM_tile, vec_tile, res_tile)
        wp.tile_store(res[worldid], res_tile, offset=(dofid, 0))

      wp.launch_tiled(
        mul,
        dim=(d.nworld, size),
        inputs=[
          m,
          d,
          adr,
          res.reshape(res.shape + (1,)),
          vec.reshape(vec.shape + (1,)),
          skip,
        ],
        # TODO(team): develop heuristic for block dim, or make configurable
        block_dim=32,
      )

    qLD_tileadr, qLD_tilesize = m.qLD_tileadr.numpy(), m.qLD_tilesize.numpy()

    for i in range(len(qLD_tileadr)):
      beg = qLD_tileadr[i]
      end = m.qLD_tile.shape[0] if i == len(qLD_tileadr) - 1 else qLD_tileadr[i + 1]
      tile_mul(beg, end - beg, int(qLD_tilesize[i]))

  else:

    @kernel
    def _mul_m_sparse_diag(
      m: Model,
      d: Data,
      res: wp.array(ndim=2, dtype=wp.float32),
      vec: wp.array(ndim=2, dtype=wp.float32),
      skip: wp.array(ndim=1, dtype=bool),
    ):
      worldid, dofid = wp.tid()

      if skip[worldid]:
        return

      res[worldid, dofid] = d.qM[worldid, 0, m.dof_Madr[dofid]] * vec[worldid, dofid]

    @kernel
    def _mul_m_sparse_ij(
      m: Model,
      d: Data,
      res: wp.array(ndim=2, dtype=wp.float32),
      vec: wp.array(ndim=2, dtype=wp.float32),
      skip: wp.array(ndim=1, dtype=bool),
    ):
      worldid, elementid = wp.tid()

      if skip[worldid]:
        return

      i = m.qM_mulm_i[elementid]
      j = m.qM_mulm_j[elementid]
      madr_ij = m.qM_madr_ij[elementid]

      qM = d.qM[worldid, 0, madr_ij]

      wp.atomic_add(res[worldid], i, qM * vec[worldid, j])
      wp.atomic_add(res[worldid], j, qM * vec[worldid, i])

    wp.launch(_mul_m_sparse_diag, dim=(d.nworld, m.nv), inputs=[m, d, res, vec, skip])

    wp.launch(
      _mul_m_sparse_ij,
      dim=(d.nworld, m.qM_madr_ij.size),
      inputs=[m, d, res, vec, skip],
    )


@event_scope
def xfrc_accumulate(m: Model, d: Data, qfrc: array2df):
  @wp.kernel
  def _accumulate(m: Model, d: Data, qfrc: array2df):
    worldid, dofid = wp.tid()
    cdof = d.cdof[worldid, dofid]
    rotational_cdof = wp.vec3(cdof[0], cdof[1], cdof[2])
    jac = wp.spatial_vector(cdof[3], cdof[4], cdof[5], cdof[0], cdof[1], cdof[2])

    dof_bodyid = m.dof_bodyid[dofid]
    accumul = float(0.0)

    for bodyid in range(dof_bodyid, m.nbody):
      # any body that is in the subtree of dof_bodyid is part of the jacobian
      parentid = bodyid
      while parentid != 0 and parentid != dof_bodyid:
        parentid = m.body_parentid[parentid]
      if parentid == 0:
        continue  # body is not part of the subtree
      offset = d.xipos[worldid, bodyid] - d.subtree_com[worldid, m.body_rootid[bodyid]]
      cross_term = wp.cross(rotational_cdof, offset)
      accumul += wp.dot(jac, d.xfrc_applied[worldid, bodyid]) + wp.dot(
        cross_term,
        wp.vec3(
          d.xfrc_applied[worldid, bodyid][0],
          d.xfrc_applied[worldid, bodyid][1],
          d.xfrc_applied[worldid, bodyid][2],
        ),
      )

    qfrc[worldid, dofid] += accumul

  wp.launch(kernel=_accumulate, dim=(d.nworld, m.nv), inputs=[m, d, qfrc])


@wp.func
def bisection(x: wp.array(dtype=int), v: int, a_: int, b_: int) -> int:
  # Binary search for the largest index i such that x[i] <= v
  # x is a sorted array
  # a and b are the start and end indices within x to search
  a = int(a_)
  b = int(b_)
  c = int(0)
  while b - a > 1:
    c = (a + b) // 2
    if x[c] <= v:
      a = c
    else:
      b = c
  c = a
  if c != b and x[b] <= v:
    c = b
  return c


@wp.func
def all_same(v0: wp.vec3, v1: wp.vec3) -> wp.bool:
  dx = abs(v0[0] - v1[0])
  dy = abs(v0[1] - v1[1])
  dz = abs(v0[2] - v1[2])

  return (
    (dx <= 1.0e-9 or dx <= max(abs(v0[0]), abs(v1[0])) * 1.0e-9)
    and (dy <= 1.0e-9 or dy <= max(abs(v0[1]), abs(v1[1])) * 1.0e-9)
    and (dz <= 1.0e-9 or dz <= max(abs(v0[2]), abs(v1[2])) * 1.0e-9)
  )


@wp.func
def any_different(v0: wp.vec3, v1: wp.vec3) -> wp.bool:
  dx = abs(v0[0] - v1[0])
  dy = abs(v0[1] - v1[1])
  dz = abs(v0[2] - v1[2])

  return (
    (dx > 1.0e-9 and dx > max(abs(v0[0]), abs(v1[0])) * 1.0e-9)
    or (dy > 1.0e-9 and dy > max(abs(v0[1]), abs(v1[1])) * 1.0e-9)
    or (dz > 1.0e-9 and dz > max(abs(v0[2]), abs(v1[2])) * 1.0e-9)
  )


@wp.func
def mat33_from_rows(a: wp.vec3, b: wp.vec3, c: wp.vec3):
  return wp.mat33(a, b, c)


@wp.func
def mat33_from_cols(a: wp.vec3, b: wp.vec3, c: wp.vec3):
  return wp.mat33(a.x, b.x, c.x, a.y, b.y, c.y, a.z, b.z, c.z)


@wp.func
def _decode_pyramid(
  pyramid: wp.array(dtype=wp.float32), efc_address: int, mu: vec5, condim: int
) -> wp.spatial_vector:
  """Converts pyramid representation to contact force."""
  force = wp.spatial_vector()

  if condim == 1:
    force[0] = pyramid[efc_address]
    return force

  force[0] = float(0.0)
  for i in range(condim - 1):
    dir1 = pyramid[2 * i + efc_address]
    dir2 = pyramid[2 * i + efc_address + 1]
    force[0] += dir1 + dir2
    force[i + 1] = (dir1 - dir2) * mu[i]

  return force


@wp.func
def contact_force(
  m: Model, d: Data, contact_id: int, to_world_frame: bool = False
) -> wp.spatial_vector:
  """Extract 6D force:torque for one contact, in contact frame by default."""
  force = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  condim = d.contact.dim[contact_id]
  efc_address = d.contact.efc_address[contact_id, 0]

  if contact_id >= 0 and contact_id <= d.ncon[0] and efc_address >= 0:
    if m.opt.cone == int(ConeType.PYRAMIDAL.value):
      force = _decode_pyramid(
        d.efc.force,
        efc_address,
        d.contact.friction[contact_id],
        condim,
      )
    else:
      for i in range(condim):
        force[i] = d.efc.force[d.contact.efc_address[contact_id, i]]

  if to_world_frame:
    # Transform both top and bottom parts of spatial vector by the full contact frame matrix
    t = wp.spatial_top(force) @ d.contact.frame[contact_id]
    b = wp.spatial_bottom(force) @ d.contact.frame[contact_id]
    force = wp.spatial_vector(t, b)

  return force


@wp.kernel
def contact_force_kernel(
  m: Model,
  d: Data,
  force: wp.array(dtype=wp.spatial_vector),
  contact_ids: wp.array(dtype=int),
  to_world_frame: bool,
):
  tid = wp.tid()

  contact_id = contact_ids[tid]

  if contact_id >= d.ncon[0]:
    return

  force[tid] = contact_force(m, d, contact_id, to_world_frame)


@wp.func
def transform_force(
  force: wp.vec3, torque: wp.vec3, offset: wp.vec3
) -> wp.spatial_vector:
  torque -= wp.cross(offset, force)
  return wp.spatial_vector(torque, force)


@wp.func
def transform_force(frc: wp.spatial_vector, offset: wp.vec3) -> wp.spatial_vector:
  force = wp.spatial_top(frc)
  torque = wp.spatial_bottom(frc)
  return transform_force(force, torque, offset)


@wp.func
def jac(
  m: Model,
  d: Data,
  point: wp.vec3,
  bodyid: wp.int32,
  dofid: wp.int32,
  worldid: wp.int32,
) -> Tuple[wp.vec3, wp.vec3]:
  dof_bodyid = m.dof_bodyid[dofid]
  in_tree = int(dof_bodyid == 0)
  parentid = bodyid
  while parentid != 0:
    if parentid == dof_bodyid:
      in_tree = 1
      break
    parentid = m.body_parentid[parentid]

  if not in_tree:
    return wp.vec3(0.0), wp.vec3(0.0)

  offset = point - wp.vec3(d.subtree_com[worldid, m.body_rootid[bodyid]])

  cdof = d.cdof[worldid, dofid]
  cdof_ang = wp.spatial_top(cdof)
  cdof_lin = wp.spatial_bottom(cdof)

  jacp = cdof_lin + wp.cross(cdof_ang, offset)
  jacr = cdof_ang

  return jacp, jacr

@wp.func
def get_batched_array(array: wp.array2d(dtype=Any), worldid: wp.int32):
  """Returns the array slice for the given worldid."""
  modelid = worldid % array.shape[0]
  return array[modelid]

@wp.func
def get_batched_value(array: wp.array2d(dtype=Any), worldid: wp.int32, i: wp.int32):
  modelid = worldid % array.shape[0]
  return array[modelid, i]

@wp.func
def get_batched_value(array: wp.array3d(dtype=Any), worldid: wp.int32, i: wp.int32, j: wp.int32):
  modelid = worldid % array.shape[0]
  return array[modelid, i, j]

@wp.func
def get_batched_value(array: wp.array4d(dtype=Any), worldid: wp.int32, i: wp.int32, j: wp.int32, k: wp.int32):
  modelid = worldid % array.shape[0]
  return array[modelid, i, j, k]
