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

import mujoco
import warp as wp
from packaging import version

from . import math
from .types import Data
from .types import DisableBit
from .types import JointType
from .types import Model
from .types import TrnType
from .types import array2df
from .types import array3df
from .types import vec10
from .warp_util import event_scope
from .warp_util import kernel
from .warp_util import kernel_copy


@event_scope
def kinematics(m: Model, d: Data):
  """Forward kinematics."""

  @kernel
  def _root(m: Model, d: Data):
    worldid = wp.tid()
    d.xpos[worldid, 0] = wp.vec3(0.0)
    d.xquat[worldid, 0] = wp.quat(1.0, 0.0, 0.0, 0.0)
    d.xipos[worldid, 0] = wp.vec3(0.0)
    d.xmat[worldid, 0] = wp.identity(n=3, dtype=wp.float32)
    d.ximat[worldid, 0] = wp.identity(n=3, dtype=wp.float32)

  @kernel
  def _level(m: Model, d: Data, leveladr: int):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    jntadr = m.body_jntadr[bodyid]
    jntnum = m.body_jntnum[bodyid]
    qpos = d.qpos[worldid]

    if jntnum == 0:
      # no joints - apply fixed translation and rotation relative to parent
      pid = m.body_parentid[bodyid]
      xpos = (d.xmat[worldid, pid] * m.body_pos[worldid, bodyid]) + d.xpos[worldid, pid]
      xquat = math.mul_quat(d.xquat[worldid, pid], m.body_quat[worldid, bodyid])
    elif jntnum == 1 and m.jnt_type[jntadr] == wp.static(JointType.FREE.value):
      # free joint
      qadr = m.jnt_qposadr[jntadr]
      xpos = wp.vec3(qpos[qadr], qpos[qadr + 1], qpos[qadr + 2])
      xquat = wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])
      d.xanchor[worldid, jntadr] = xpos
      d.xaxis[worldid, jntadr] = m.jnt_axis[jntadr]
    else:
      # regular or no joints
      # apply fixed translation and rotation relative to parent
      pid = m.body_parentid[bodyid]
      xpos = (d.xmat[worldid, pid] * m.body_pos[worldid, bodyid]) + d.xpos[worldid, pid]
      xquat = math.mul_quat(d.xquat[worldid, pid], m.body_quat[worldid, bodyid])

      for _ in range(jntnum):
        qadr = m.jnt_qposadr[jntadr]
        jnt_type = m.jnt_type[jntadr]
        jnt_axis = m.jnt_axis[jntadr]
        xanchor = math.rot_vec_quat(m.jnt_pos[jntadr], xquat) + xpos
        xaxis = math.rot_vec_quat(jnt_axis, xquat)

        if jnt_type == wp.static(JointType.BALL.value):
          qloc = wp.quat(
            qpos[qadr + 0],
            qpos[qadr + 1],
            qpos[qadr + 2],
            qpos[qadr + 3],
          )
          xquat = math.mul_quat(xquat, qloc)
          # correct for off-center rotation
          xpos = xanchor - math.rot_vec_quat(m.jnt_pos[jntadr], xquat)
        elif jnt_type == wp.static(JointType.SLIDE.value):
          xpos += xaxis * (qpos[qadr] - m.qpos0[worldid, qadr])
        elif jnt_type == wp.static(JointType.HINGE.value):
          qpos0 = m.qpos0[worldid,qadr]
          qloc = math.axis_angle_to_quat(jnt_axis, qpos[qadr] - qpos0)
          xquat = math.mul_quat(xquat, qloc)
          # correct for off-center rotation
          xpos = xanchor - math.rot_vec_quat(m.jnt_pos[jntadr], xquat)

        d.xanchor[worldid, jntadr] = xanchor
        d.xaxis[worldid, jntadr] = xaxis
        jntadr += 1

    d.xpos[worldid, bodyid] = xpos
    xquat = wp.normalize(xquat)
    d.xquat[worldid, bodyid] = xquat
    d.xmat[worldid, bodyid] = math.quat_to_mat(xquat)
    d.xipos[worldid, bodyid] = xpos + math.rot_vec_quat(m.body_ipos[worldid, bodyid], xquat)
    d.ximat[worldid, bodyid] = math.quat_to_mat(
      math.mul_quat(xquat, m.body_iquat[worldid, bodyid])
    )

  @kernel
  def geom_local_to_global(m: Model, d: Data):
    worldid, geomid = wp.tid()
    bodyid = m.geom_bodyid[geomid]
    xpos = d.xpos[worldid, bodyid]
    xquat = d.xquat[worldid, bodyid]
    d.geom_xpos[worldid, geomid] = xpos + math.rot_vec_quat(m.geom_pos[worldid, geomid], xquat)
    d.geom_xmat[worldid, geomid] = math.quat_to_mat(
      math.mul_quat(xquat, m.geom_quat[worldid, geomid])
    )

  @kernel
  def site_local_to_global(m: Model, d: Data):
    worldid, siteid = wp.tid()
    bodyid = m.site_bodyid[siteid]
    xpos = d.xpos[worldid, bodyid]
    xquat = d.xquat[worldid, bodyid]
    d.site_xpos[worldid, siteid] = xpos + math.rot_vec_quat(m.site_pos[worldid, siteid], xquat)
    d.site_xmat[worldid, siteid] = math.quat_to_mat(
      math.mul_quat(xquat, m.site_quat[worldid, siteid])
    )

  wp.launch(_root, dim=(m.nworld), inputs=[m, d])

  body_treeadr = m.body_treeadr.numpy()
  for i in range(1, len(body_treeadr)):
    beg = body_treeadr[i]
    end = m.nbody if i == len(body_treeadr) - 1 else body_treeadr[i + 1]
    wp.launch(_level, dim=(m.nworld, end - beg), inputs=[m, d, beg])

  if m.ngeom:
    wp.launch(geom_local_to_global, dim=(m.nworld, m.ngeom), inputs=[m, d])

  if m.nsite:
    wp.launch(site_local_to_global, dim=(m.nworld, m.nsite), inputs=[m, d])


@event_scope
def com_pos(m: Model, d: Data):
  """Map inertias and motion dofs to global frame centered at subtree-CoM."""

  @kernel
  def subtree_com_init(m: Model, d: Data):
    worldid, bodyid = wp.tid()
    d.subtree_com[worldid, bodyid] = d.xipos[worldid, bodyid] * m.body_mass[worldid, bodyid]

  @kernel
  def subtree_com_acc(m: Model, d: Data, leveladr: int):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    pid = m.body_parentid[bodyid]
    wp.atomic_add(d.subtree_com, worldid, pid, d.subtree_com[worldid, bodyid])

  @kernel
  def subtree_div(m: Model, d: Data):
    worldid, bodyid = wp.tid()
    d.subtree_com[worldid, bodyid] /= m.subtree_mass[worldid,bodyid]

  @kernel
  def cinert(m: Model, d: Data):
    worldid, bodyid = wp.tid()
    mat = d.ximat[worldid, bodyid]
    inert = m.body_inertia[worldid, bodyid]
    mass = m.body_mass[worldid, bodyid]
    dif = d.xipos[worldid, bodyid] - d.subtree_com[worldid, m.body_rootid[bodyid]]
    # express inertia in com-based frame (mju_inertCom)

    res = vec10()
    # res_rot = mat * diag(inert) * mat'
    tmp = mat @ wp.diag(inert) @ wp.transpose(mat)
    res[0] = tmp[0, 0]
    res[1] = tmp[1, 1]
    res[2] = tmp[2, 2]
    res[3] = tmp[0, 1]
    res[4] = tmp[0, 2]
    res[5] = tmp[1, 2]
    # res_rot -= mass * dif_cross * dif_cross
    res[0] += mass * (dif[1] * dif[1] + dif[2] * dif[2])
    res[1] += mass * (dif[0] * dif[0] + dif[2] * dif[2])
    res[2] += mass * (dif[0] * dif[0] + dif[1] * dif[1])
    res[3] -= mass * dif[0] * dif[1]
    res[4] -= mass * dif[0] * dif[2]
    res[5] -= mass * dif[1] * dif[2]
    # res_tran = mass * dif
    res[6] = mass * dif[0]
    res[7] = mass * dif[1]
    res[8] = mass * dif[2]
    # res_mass = mass
    res[9] = mass

    d.cinert[worldid, bodyid] = res

  @kernel
  def cdof(m: Model, d: Data):
    worldid, jntid = wp.tid()
    bodyid = m.jnt_bodyid[jntid]
    dofid = m.jnt_dofadr[jntid]
    jnt_type = m.jnt_type[jntid]
    xaxis = d.xaxis[worldid, jntid]
    xmat = wp.transpose(d.xmat[worldid, bodyid])

    # compute com-anchor vector
    offset = d.subtree_com[worldid, m.body_rootid[bodyid]] - d.xanchor[worldid, jntid]

    res = d.cdof[worldid]
    if jnt_type == wp.static(JointType.FREE.value):
      res[dofid + 0] = wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
      res[dofid + 1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
      res[dofid + 2] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
      # I_3 rotation in child frame (assume no subsequent rotations)
      res[dofid + 3] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
      res[dofid + 4] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
      res[dofid + 5] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
    elif jnt_type == wp.static(JointType.BALL.value):  # ball
      # I_3 rotation in child frame (assume no subsequent rotations)
      res[dofid + 0] = wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset))
      res[dofid + 1] = wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset))
      res[dofid + 2] = wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset))
    elif jnt_type == wp.static(JointType.SLIDE.value):
      res[dofid] = wp.spatial_vector(wp.vec3(0.0), xaxis)
    elif jnt_type == wp.static(JointType.HINGE.value):  # hinge
      res[dofid] = wp.spatial_vector(xaxis, wp.cross(xaxis, offset))

  wp.launch(subtree_com_init, dim=(m.nworld, m.nbody), inputs=[m, d])

  body_treeadr = m.body_treeadr.numpy()

  for i in reversed(range(len(body_treeadr))):
    beg = body_treeadr[i]
    end = m.nbody if i == len(body_treeadr) - 1 else body_treeadr[i + 1]
    wp.launch(subtree_com_acc, dim=(m.nworld, end - beg), inputs=[m, d, beg])

  wp.launch(subtree_div, dim=(m.nworld, m.nbody), inputs=[m, d])
  wp.launch(cinert, dim=(m.nworld, m.nbody), inputs=[m, d])
  wp.launch(cdof, dim=(m.nworld, m.njnt), inputs=[m, d])


@event_scope
def crb(m: Model, d: Data):
  """Composite rigid body inertia algorithm."""

  kernel_copy(d.crb, d.cinert)

  @kernel
  def crb_accumulate(m: Model, d: Data, leveladr: int):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    pid = m.body_parentid[bodyid]
    if pid == 0:
      return
    wp.atomic_add(d.crb, worldid, pid, d.crb[worldid, bodyid])

  @kernel
  def qM_sparse(m: Model, d: Data):
    worldid, dofid = wp.tid()
    madr_ij = m.dof_Madr[dofid]
    bodyid = m.dof_bodyid[dofid]

    # init M(i,i) with armature inertia
    d.qM[worldid, 0, madr_ij] = m.dof_armature[worldid, dofid]

    # precompute buf = crb_body_i * cdof_i
    buf = math.inert_vec(d.crb[worldid, bodyid], d.cdof[worldid, dofid])

    # sparse backward pass over ancestors
    while dofid >= 0:
      d.qM[worldid, 0, madr_ij] += wp.dot(d.cdof[worldid, dofid], buf)
      madr_ij += 1
      dofid = m.dof_parentid[dofid]

  @kernel
  def qM_dense(m: Model, d: Data):
    worldid, dofid = wp.tid()
    bodyid = m.dof_bodyid[dofid]

    # init M(i,i) with armature inertia
    M = m.dof_armature[worldid, dofid]

    # precompute buf = crb_body_i * cdof_i
    buf = math.inert_vec(d.crb[worldid, bodyid], d.cdof[worldid, dofid])
    M += wp.dot(d.cdof[worldid, dofid], buf)

    d.qM[worldid, dofid, dofid] = M

    # sparse backward pass over ancestors
    dofidi = dofid
    dofid = m.dof_parentid[dofid]
    while dofid >= 0:
      qMij = wp.dot(d.cdof[worldid, dofid], buf)
      d.qM[worldid, dofidi, dofid] += qMij
      d.qM[worldid, dofid, dofidi] += qMij
      dofid = m.dof_parentid[dofid]

  body_treeadr = m.body_treeadr.numpy()
  for i in reversed(range(len(body_treeadr))):
    beg = body_treeadr[i]
    end = m.nbody if i == len(body_treeadr) - 1 else body_treeadr[i + 1]
    wp.launch(crb_accumulate, dim=(m.nworld, end - beg), inputs=[m, d, beg])

  d.qM.zero_()
  if m.opt.is_sparse:
    wp.launch(qM_sparse, dim=(m.nworld, m.nv), inputs=[m, d])
  else:
    wp.launch(qM_dense, dim=(m.nworld, m.nv), inputs=[m, d])


def _factor_i_sparse_legacy(m: Model, d: Data, M: array3df, L: array3df, D: array2df):
  """Sparse L'*D*L factorizaton of inertia-like matrix M, assumed spd."""

  @kernel
  def qLD_acc(m: Model, leveladr: int, L: array3df):
    worldid, nodeid = wp.tid()
    update = m.qLD_update_tree[leveladr + nodeid]
    i, k, Madr_ki = update[0], update[1], update[2]
    Madr_i = m.dof_Madr[i]
    # tmp = M(k,i) / M(k,k)
    tmp = L[worldid, 0, Madr_ki] / L[worldid, 0, m.dof_Madr[k]]
    for j in range(m.dof_Madr[i + 1] - Madr_i):
      # M(i,j) -= M(k,j) * tmp
      wp.atomic_sub(L[worldid, 0], Madr_i + j, L[worldid, 0, Madr_ki + j] * tmp)
    # M(k,i) = tmp
    L[worldid, 0, Madr_ki] = tmp

  @kernel
  def qLDiag_div(m: Model, L: array3df, D: array2df):
    worldid, dofid = wp.tid()
    D[worldid, dofid] = 1.0 / L[worldid, 0, m.dof_Madr[dofid]]

  kernel_copy(L, M)

  qLD_update_treeadr = m.qLD_update_treeadr.numpy()

  for i in reversed(range(len(qLD_update_treeadr))):
    if i == len(qLD_update_treeadr) - 1:
      beg, end = qLD_update_treeadr[i], m.qLD_update_tree.shape[0]
    else:
      beg, end = qLD_update_treeadr[i], qLD_update_treeadr[i + 1]
    wp.launch(qLD_acc, dim=(m.nworld, end - beg), inputs=[m, beg, L])

  wp.launch(qLDiag_div, dim=(m.nworld, m.nv), inputs=[m, L, D])


def _factor_i_sparse(m: Model, d: Data, M: array3df, L: array3df, D: array2df):
  """Sparse L'*D*L factorizaton of inertia-like matrix M, assumed spd."""
  if version.parse(mujoco.__version__) <= version.parse("3.2.7"):
    return _factor_i_sparse_legacy(m, d, M, L, D)

  @kernel
  def qLD_acc(m: Model, leveladr: int, L: array3df):
    worldid, nodeid = wp.tid()
    update = m.qLD_update_tree[leveladr + nodeid]
    i, k, Madr_ki = update[0], update[1], update[2]
    Madr_i = m.M_rowadr[i]  # Address of row being updated
    diag_k = m.M_rowadr[k] + m.M_rownnz[k] - 1  # Address of diagonal element of k
    # tmp = M(k,i) / M(k,k)
    tmp = L[worldid, 0, Madr_ki] / L[worldid, 0, diag_k]
    for j in range(m.M_rownnz[i]):
      # M(i,j) -= M(k,j) * tmp
      wp.atomic_sub(L[worldid, 0], Madr_i + j, L[worldid, 0, m.M_rowadr[k] + j] * tmp)
    # M(k,i) = tmp
    L[worldid, 0, Madr_ki] = tmp

  @kernel
  def qLDiag_div(m: Model, L: array3df, D: array2df):
    worldid, dofid = wp.tid()
    diag_i = (
      m.M_rowadr[dofid] + m.M_rownnz[dofid] - 1
    )  # Address of diagonal element of i
    D[worldid, dofid] = 1.0 / L[worldid, 0, diag_i]

  @kernel
  def copy_CSR(L: array3df, M: array3df, mapM2M: wp.array(dtype=wp.int32, ndim=1)):
    worldid, ind = wp.tid()
    L[worldid, 0, ind] = M[worldid, 0, mapM2M[ind]]

  wp.launch(copy_CSR, dim=(m.nworld, m.nM), inputs=[L, M, m.mapM2M])

  qLD_update_treeadr = m.qLD_update_treeadr.numpy()

  for i in reversed(range(len(qLD_update_treeadr))):
    if i == len(qLD_update_treeadr) - 1:
      beg, end = qLD_update_treeadr[i], m.qLD_update_tree.shape[0]
    else:
      beg, end = qLD_update_treeadr[i], qLD_update_treeadr[i + 1]
    wp.launch(qLD_acc, dim=(m.nworld, end - beg), inputs=[m, beg, L])

  wp.launch(qLDiag_div, dim=(m.nworld, m.nv), inputs=[m, L, D])


def _factor_i_dense(m: Model, d: Data, M: wp.array, L: wp.array):
  """Dense Cholesky factorizaton of inertia-like matrix M, assumed spd."""

  # TODO(team): develop heuristic for block dim, or make configurable
  block_dim = 32

  def tile_cholesky(adr: int, size: int, tilesize: int):
    @kernel
    def cholesky(m: Model, leveladr: int, M: array3df, L: array3df):
      worldid, nodeid = wp.tid()
      dofid = m.qLD_tile[leveladr + nodeid]
      M_tile = wp.tile_load(
        M[worldid], shape=(tilesize, tilesize), offset=(dofid, dofid)
      )
      L_tile = wp.tile_cholesky(M_tile)
      wp.tile_store(L[worldid], L_tile, offset=(dofid, dofid))

    wp.launch_tiled(
      cholesky, dim=(m.nworld, size), inputs=[m, adr, M, L], block_dim=block_dim
    )

  qLD_tileadr, qLD_tilesize = m.qLD_tileadr.numpy(), m.qLD_tilesize.numpy()

  for i in range(len(qLD_tileadr)):
    beg = qLD_tileadr[i]
    end = m.qLD_tile.shape[0] if i == len(qLD_tileadr) - 1 else qLD_tileadr[i + 1]
    tile_cholesky(beg, end - beg, int(qLD_tilesize[i]))


def factor_i(m: Model, d: Data, M, L, D=None):
  """Factorizaton of inertia-like matrix M, assumed spd."""

  if m.opt.is_sparse:
    assert D is not None
    _factor_i_sparse(m, d, M, L, D)
  else:
    _factor_i_dense(m, d, M, L)


@event_scope
def factor_m(m: Model, d: Data):
  """Factorizaton of inertia-like matrix M, assumed spd."""
  factor_i(m, d, d.qM, d.qLD, d.qLDiagInv)


@event_scope
def rne(m: Model, d: Data):
  """Computes inverse dynamics using Newton-Euler algorithm."""

  @kernel
  def cacc_gravity(m: Model, d: Data):
    worldid = wp.tid()
    d.rne_cacc[worldid, 0] = wp.spatial_vector(wp.vec3(0.0), -m.opt.gravity)

  @kernel
  def cacc_level(
    m: Model,
    d: Data,
    leveladr: int,
  ):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    dofnum = m.body_dofnum[bodyid]
    pid = m.body_parentid[bodyid]
    dofadr = m.body_dofadr[bodyid]
    local_cacc = d.rne_cacc[worldid, pid]
    for i in range(dofnum):
      local_cacc += d.cdof_dot[worldid, dofadr + i] * d.qvel[worldid, dofadr + i]
    d.rne_cacc[worldid, bodyid] = local_cacc

  @kernel
  def frc_fn(d: Data):
    worldid, bodyid = wp.tid()
    frc = math.inert_vec(d.cinert[worldid, bodyid], d.rne_cacc[worldid, bodyid])
    frc += math.motion_cross_force(
      d.cvel[worldid, bodyid],
      math.inert_vec(d.cinert[worldid, bodyid], d.cvel[worldid, bodyid]),
    )
    d.rne_cfrc[worldid, bodyid] = frc

  @kernel
  def cfrc_fn(m: Model, d: Data, leveladr: int):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    pid = m.body_parentid[bodyid]
    wp.atomic_add(d.rne_cfrc[worldid], pid, d.rne_cfrc[worldid, bodyid])

  @kernel
  def qfrc_bias(m: Model, d: Data):
    worldid, dofid = wp.tid()
    bodyid = m.dof_bodyid[dofid]
    d.qfrc_bias[worldid, dofid] = wp.dot(
      d.cdof[worldid, dofid], d.rne_cfrc[worldid, bodyid]
    )

  if m.opt.disableflags & DisableBit.GRAVITY:
    d.rne_cacc.zero_()
  else:
    wp.launch(cacc_gravity, dim=[m.nworld], inputs=[m, d])

  body_treeadr = m.body_treeadr.numpy()
  for i in range(len(body_treeadr)):
    beg = body_treeadr[i]
    end = m.nbody if i == len(body_treeadr) - 1 else body_treeadr[i + 1]
    wp.launch(cacc_level, dim=(m.nworld, end - beg), inputs=[m, d, beg])

  wp.launch(frc_fn, dim=[m.nworld, m.nbody], inputs=[d])

  for i in reversed(range(len(body_treeadr))):
    beg = body_treeadr[i]
    end = m.nbody if i == len(body_treeadr) - 1 else body_treeadr[i + 1]
    wp.launch(cfrc_fn, dim=[m.nworld, end - beg], inputs=[m, d, beg])

  wp.launch(qfrc_bias, dim=[m.nworld, m.nv], inputs=[m, d])


@event_scope
def transmission(m: Model, d: Data):
  """Computes actuator/transmission lengths and moments."""
  if not m.nu:
    return d

  @kernel
  def _transmission(
    m: Model,
    d: Data,
    # outputs
    length: array2df,
    moment: array3df,
  ):
    worldid, actid = wp.tid()
    qpos = d.qpos[worldid]
    jntid = m.actuator_trnid[actid, 0]
    jnt_typ = m.jnt_type[jntid]
    qadr = m.jnt_qposadr[jntid]
    vadr = m.jnt_dofadr[jntid]
    trntype = m.actuator_trntype[actid]
    gear = m.actuator_gear[actid]
    if trntype == wp.static(TrnType.JOINT.value) or trntype == wp.static(
      TrnType.JOINTINPARENT.value
    ):
      if jnt_typ == wp.static(JointType.FREE.value):
        length[worldid, actid] = 0.0
        if trntype == wp.static(TrnType.JOINTINPARENT.value):
          quat_neg = math.quat_inv(
            wp.quat(qpos[qadr + 3], qpos[qadr + 4], qpos[qadr + 5], qpos[qadr + 6])
          )
          gearaxis = math.rot_vec_quat(wp.spatial_bottom(gear), quat_neg)
          moment[worldid, actid, vadr + 0] = gear[0]
          moment[worldid, actid, vadr + 1] = gear[1]
          moment[worldid, actid, vadr + 2] = gear[2]
          moment[worldid, actid, vadr + 3] = gearaxis[0]
          moment[worldid, actid, vadr + 4] = gearaxis[1]
          moment[worldid, actid, vadr + 5] = gearaxis[2]
        else:
          for i in range(6):
            moment[worldid, actid, vadr + i] = gear[i]
      elif jnt_typ == wp.static(JointType.BALL.value):
        q = wp.quat(qpos[qadr + 0], qpos[qadr + 1], qpos[qadr + 2], qpos[qadr + 3])
        axis_angle = math.quat_to_vel(q)
        gearaxis = wp.spatial_top(gear)  # [:3]
        if trntype == wp.static(TrnType.JOINTINPARENT.value):
          quat_neg = math.quat_inv(q)
          gearaxis = math.rot_vec_quat(gearaxis, quat_neg)
        length[worldid, actid] = wp.dot(axis_angle, gearaxis)
        for i in range(3):
          moment[worldid, actid, vadr + i] = gearaxis[i]
      elif jnt_typ == wp.static(JointType.SLIDE.value) or jnt_typ == wp.static(
        JointType.HINGE.value
      ):
        length[worldid, actid] = qpos[qadr] * gear[0]
        moment[worldid, actid, vadr] = gear[0]
      else:
        wp.printf("unrecognized joint type")
    else:
      # TODO handle site, tendon transmission types
      wp.printf("unhandled transmission type %d\n", trntype)

  wp.launch(
    _transmission,
    dim=[m.nworld, m.nu],
    inputs=[m, d],
    outputs=[d.actuator_length, d.actuator_moment],
  )


@event_scope
def com_vel(m: Model, d: Data):
  """Computes cvel, cdof_dot."""

  @kernel
  def _root(d: Data):
    worldid, elementid = wp.tid()
    d.cvel[worldid, 0][elementid] = 0.0

  @kernel
  def _level(m: Model, d: Data, leveladr: int):
    worldid, nodeid = wp.tid()
    bodyid = m.body_tree[leveladr + nodeid]
    dofid = m.body_dofadr[bodyid]
    jntid = m.body_jntadr[bodyid]
    jntnum = m.body_jntnum[bodyid]
    pid = m.body_parentid[bodyid]

    if jntnum == 0:
      d.cvel[worldid, bodyid] = d.cvel[worldid, pid]
      return

    cvel = d.cvel[worldid, pid]
    qvel = d.qvel[worldid]
    cdof = d.cdof[worldid]

    for j in range(jntid, jntid + jntnum):
      jnttype = m.jnt_type[j]

      if jnttype == wp.static(JointType.FREE.value):
        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]

        d.cdof_dot[worldid, dofid + 3] = math.motion_cross(cvel, cdof[dofid + 3])
        d.cdof_dot[worldid, dofid + 4] = math.motion_cross(cvel, cdof[dofid + 4])
        d.cdof_dot[worldid, dofid + 5] = math.motion_cross(cvel, cdof[dofid + 5])

        cvel += cdof[dofid + 3] * qvel[dofid + 3]
        cvel += cdof[dofid + 4] * qvel[dofid + 4]
        cvel += cdof[dofid + 5] * qvel[dofid + 5]

        dofid += 6
      elif jnttype == wp.static(JointType.BALL.value):
        d.cdof_dot[worldid, dofid + 0] = math.motion_cross(cvel, cdof[dofid + 0])
        d.cdof_dot[worldid, dofid + 1] = math.motion_cross(cvel, cdof[dofid + 1])
        d.cdof_dot[worldid, dofid + 2] = math.motion_cross(cvel, cdof[dofid + 2])

        cvel += cdof[dofid + 0] * qvel[dofid + 0]
        cvel += cdof[dofid + 1] * qvel[dofid + 1]
        cvel += cdof[dofid + 2] * qvel[dofid + 2]

        dofid += 3
      else:
        d.cdof_dot[worldid, dofid] = math.motion_cross(cvel, cdof[dofid])
        cvel += cdof[dofid] * qvel[dofid]

        dofid += 1

    d.cvel[worldid, bodyid] = cvel

  wp.launch(_root, dim=(m.nworld, 6), inputs=[d])

  body_treeadr = m.body_treeadr.numpy()
  for i in range(1, len(body_treeadr)):
    beg = body_treeadr[i]
    end = m.nbody if i == len(body_treeadr) - 1 else body_treeadr[i + 1]
    wp.launch(_level, dim=(m.nworld, end - beg), inputs=[m, d, beg])


def _solve_LD_sparse(
  m: Model, d: Data, L: array3df, D: array2df, x: array2df, y: array2df
):
  """Computes sparse backsubstitution: x = inv(L'*D*L)*y"""

  @kernel
  def x_acc_up(m: Model, L: array3df, x: array2df, leveladr: int):
    worldid, nodeid = wp.tid()
    update = m.qLD_update_tree[leveladr + nodeid]
    i, k, Madr_ki = update[0], update[1], update[2]
    wp.atomic_sub(x[worldid], i, L[worldid, 0, Madr_ki] * x[worldid, k])

  @kernel
  def qLDiag_mul(D: array2df, x: array2df):
    worldid, dofid = wp.tid()
    x[worldid, dofid] *= D[worldid, dofid]

  @kernel
  def x_acc_down(m: Model, L: array3df, x: array2df, leveladr: int):
    worldid, nodeid = wp.tid()
    update = m.qLD_update_tree[leveladr + nodeid]
    i, k, Madr_ki = update[0], update[1], update[2]
    wp.atomic_sub(x[worldid], k, L[worldid, 0, Madr_ki] * x[worldid, i])

  kernel_copy(x, y)

  qLD_update_treeadr = m.qLD_update_treeadr.numpy()

  for i in reversed(range(len(qLD_update_treeadr))):
    if i == len(qLD_update_treeadr) - 1:
      beg, end = qLD_update_treeadr[i], m.qLD_update_tree.shape[0]
    else:
      beg, end = qLD_update_treeadr[i], qLD_update_treeadr[i + 1]
    wp.launch(x_acc_up, dim=(m.nworld, end - beg), inputs=[m, L, x, beg])

  wp.launch(qLDiag_mul, dim=(m.nworld, m.nv), inputs=[D, x])

  for i in range(len(qLD_update_treeadr)):
    if i == len(qLD_update_treeadr) - 1:
      beg, end = qLD_update_treeadr[i], m.qLD_update_tree.shape[0]
    else:
      beg, end = qLD_update_treeadr[i], qLD_update_treeadr[i + 1]
    wp.launch(x_acc_down, dim=(m.nworld, end - beg), inputs=[m, L, x, beg])


def _solve_LD_dense(m: Model, d: Data, L: array3df, x: array2df, y: array2df):
  """Computes dense backsubstitution: x = inv(L'*L)*y"""

  # TODO(team): develop heuristic for block dim, or make configurable
  block_dim = 32

  def tile_cho_solve(adr: int, size: int, tilesize: int):
    @kernel
    def cho_solve(m: Model, L: array3df, x: array2df, y: array2df, leveladr: int):
      worldid, nodeid = wp.tid()
      dofid = m.qLD_tile[leveladr + nodeid]
      y_slice = wp.tile_load(y[worldid], shape=(tilesize,), offset=(dofid,))
      L_tile = wp.tile_load(
        L[worldid], shape=(tilesize, tilesize), offset=(dofid, dofid)
      )
      x_slice = wp.tile_cholesky_solve(L_tile, y_slice)
      wp.tile_store(x[worldid], x_slice, offset=(dofid,))

    wp.launch_tiled(
      cho_solve, dim=(m.nworld, size), inputs=[m, L, x, y, adr], block_dim=block_dim
    )

  qLD_tileadr, qLD_tilesize = m.qLD_tileadr.numpy(), m.qLD_tilesize.numpy()

  for i in range(len(qLD_tileadr)):
    beg = qLD_tileadr[i]
    end = m.qLD_tile.shape[0] if i == len(qLD_tileadr) - 1 else qLD_tileadr[i + 1]
    tile_cho_solve(beg, end - beg, int(qLD_tilesize[i]))


def solve_LD(m: Model, d: Data, L: array3df, D: array2df, x: array2df, y: array2df):
  """Computes backsubstitution: x = qLD * y."""

  if m.opt.is_sparse:
    _solve_LD_sparse(m, d, L, D, x, y)
  else:
    _solve_LD_dense(m, d, L, x, y)


@event_scope
def solve_m(m: Model, d: Data, x: array2df, y: array2df):
  """Computes backsubstitution: x = qLD * y."""
  solve_LD(m, d, d.qLD, d.qLDiagInv, x, y)


def _factor_solve_i_dense(m: Model, d: Data, M: array3df, x: array2df, y: array2df):
  # TODO(team): develop heuristic for block dim, or make configurable
  block_dim = 32

  def tile_cholesky(adr: int, size: int, tilesize: int):
    @kernel(module="unique")
    def cholesky(m: Model, leveladr: int, M: array3df, x: array2df, y: array2df):
      worldid, nodeid = wp.tid()
      dofid = m.qLD_tile[leveladr + nodeid]
      M_tile = wp.tile_load(
        M[worldid], shape=(tilesize, tilesize), offset=(dofid, dofid)
      )
      y_slice = wp.tile_load(y[worldid], shape=(tilesize,), offset=(dofid,))

      L_tile = wp.tile_cholesky(M_tile)
      x_slice = wp.tile_cholesky_solve(L_tile, y_slice)
      wp.tile_store(x[worldid], x_slice, offset=(dofid,))

    wp.launch_tiled(
      cholesky, dim=(m.nworld, size), inputs=[m, adr, M, x, y], block_dim=block_dim
    )

  qLD_tileadr, qLD_tilesize = m.qLD_tileadr.numpy(), m.qLD_tilesize.numpy()

  for i in range(len(qLD_tileadr)):
    beg = qLD_tileadr[i]
    end = m.qLD_tile.shape[0] if i == len(qLD_tileadr) - 1 else qLD_tileadr[i + 1]
    tile_cholesky(beg, end - beg, int(qLD_tilesize[i]))


def factor_solve_i(m, d, M, L, D, x, y):
  if m.opt.is_sparse:
    _factor_i_sparse(m, d, M, L, D)
    _solve_LD_sparse(m, d, L, D, x, y)
  else:
    _factor_solve_i_dense(m, d, M, x, y)
