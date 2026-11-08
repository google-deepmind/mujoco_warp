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
"""VJPs for kinematics and center-of-mass position dependence."""

import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import math
from mujoco_warp._src import support
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model

wp.set_module_options({"enable_backward": False})


@wp.func
def _mocap_quat_vjp(quat: wp.quat, torque: wp.vec3) -> wp.quat:
  """Lifts a world angular cotangent through normalization of the wxyz mocap quaternion."""
  norm2 = wp.dot(quat, quat)
  if norm2 > 0.0:
    return (2.0 / norm2) * math.mul_quat(wp.quat(0.0, torque[0], torque[1], torque[2]), quat)
  return wp.quat(0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _kinematics_mocap_vjp(
  # Model:
  nv: int,
  ngeom: int,
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  body_subtreenum: wp.array[int],
  mocap_bodyid: wp.array[int],
  # Data in:
  mocap_pos_in: wp.array2d[wp.vec3],
  mocap_quat_in: wp.array2d[wp.quat],
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_xipos: wp.array2d[wp.vec3],
  adj_ximat: wp.array2d[wp.mat33],
  adj_geom_xpos: wp.array2d[wp.vec3],
  adj_geom_xmat: wp.array2d[wp.mat33],
  adj_subtree_com: wp.array2d[wp.vec3],
  adj_cdof: wp.array2d[wp.spatial_vector],
  scale: float,
  # Out:
  adj_mocap_pos_out: wp.array2d[wp.vec3],
  adj_mocap_quat_out: wp.array2d[wp.quat],
):
  w, mocapid = wp.tid()
  root = mocap_bodyid[mocapid]
  origin = mocap_pos_in[w, mocapid]
  force, torque = wp.vec3(), wp.vec3()
  for body in range(root, root + body_subtreenum[root]):
    if adj_xipos:
      seed = adj_xipos[w, body]
      force += seed
      torque += wp.cross(xipos_in[w, body] - origin, seed)
      torque += adjoint_util._adj_rotation(ximat_in[w, body], adj_ximat[w, body])
    if adj_subtree_com:
      seed = adj_subtree_com[w, body]
      force += seed
      torque += wp.cross(subtree_com_in[w, body] - origin, seed)
  if adj_geom_xpos:
    for geom in range(ngeom):
      if body_rootid[geom_bodyid[geom]] == root:
        seed = adj_geom_xpos[w, geom]
        force += seed
        torque += wp.cross(geom_xpos_in[w, geom] - origin, seed)
        torque += adjoint_util._adj_rotation(geom_xmat_in[w, geom], adj_geom_xmat[w, geom])
  if adj_cdof:
    for dof in range(nv):
      if body_rootid[dof_bodyid[dof]] == root:
        value, dof_seed = cdof_in[w, dof], adj_cdof[w, dof]
        torque += wp.cross(wp.spatial_top(value), wp.spatial_top(dof_seed))
        torque += wp.cross(wp.spatial_bottom(value), wp.spatial_bottom(dof_seed))
  adj_mocap_pos_out[w, mocapid] += scale * force
  adj_mocap_quat_out[w, mocapid] += scale * _mocap_quat_vjp(mocap_quat_in[w, mocapid], torque)


def kinematics_mocap_backward(m: Model, d_bwd: Data, bc: BackwardContext, *, contact: bool = False, scale: float = -1.0):
  """Accumulates kinematic cotangents from contact and smooth dynamics into mocap inputs."""
  wp.launch(
    _kinematics_mocap_vjp,
    dim=(d_bwd.nworld, m.nmocap),
    inputs=[
      m.nv,
      m.ngeom,
      m.body_rootid,
      m.dof_bodyid,
      m.geom_bodyid,
      m.body_subtreenum,
      m.mocap_bodyid,
      d_bwd.mocap_pos,
      d_bwd.mocap_quat,
      d_bwd.xipos,
      d_bwd.ximat,
      d_bwd.geom_xpos,
      d_bwd.geom_xmat,
      d_bwd.subtree_com,
      d_bwd.cdof,
      bc.adj_xipos if not contact else None,
      bc.adj_ximat if not contact else None,
      bc.res_geom_xpos if contact else None,
      bc.res_geom_xmat if contact else None,
      bc.adj_subtree_com,
      bc.adj_cdof,
      scale,
    ],
    outputs=[bc.adj_mocap_pos, bc.adj_mocap_quat],
  )


@wp.kernel(module="unique", enable_backward=False)
def _cdof_qpos_vjp(
  # Model:
  nv: int,
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  dof_jntid: wp.array[int],
  dof_parentid: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  lanes: int,
  adj_cdof: wp.array2d[wp.spatial_vector],
  # Out:
  adj_dof_out: wp.array2d[float],
):
  w, k, lane = wp.tid()
  cdof_k = cdof_in[w, k]
  jk = dof_jntid[k]
  acc = float(0.0)
  for i in range(lane, nv, lanes):
    ji = dof_jntid[i]
    jti = jnt_type[ji]
    ofs = jnt_dofadr[ji]
    if jti == JointType.FREE and i - ofs < 3:
      continue

    ancestor = bool(False)
    parent = dof_parentid[i]
    while parent >= 0:
      if parent == k:
        ancestor = True
        break
      parent = dof_parentid[parent]

    if not ancestor and ji == jk:
      ancestor = jti == JointType.BALL or (jti == JointType.FREE and k - ofs >= 3)
    if ancestor:
      acc += wp.dot(adj_cdof[w, i], math.motion_cross(cdof_k, cdof_in[w, i]))
  wp.atomic_add(adj_dof_out, w, k, acc)  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _build_ceff(
  # Model:
  nv: int,
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  # Data in:
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_cdof: wp.array2d[wp.spatial_vector],
  adj_subtree_com: wp.array2d[wp.vec3],
  # Out:
  ceff_out: wp.array2d[wp.vec3],
):
  w, b = wp.tid()
  ceff = adj_subtree_com[w, b]
  if b > 0 and body_rootid[b] == b:
    for i in range(nv):
      if body_rootid[dof_bodyid[i]] == b:
        adj = adj_cdof[w, i]
        ceff += wp.cross(wp.spatial_bottom(adj), wp.spatial_top(cdof_in[w, i]))
  ceff_out[w, b] = ceff


@wp.kernel(module="unique", enable_backward=False)
def _subtree_com_qpos_vjp(
  # Model:
  nbody: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_subtreemass: wp.array2d[float],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  lanes: int,
  adj_subtree_com: wp.array2d[wp.vec3],
  # Out:
  adj_dof_out: wp.array2d[float],
):
  w, k, lane = wp.tid()
  wm = w % body_mass.shape[0]
  acc = float(0.0)
  for body in range(1 + lane, nbody, lanes):
    if body_isdofancestor[body, k] == 0:
      continue
    root = body_rootid[body]
    jacp, _ = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      xipos_in[w, body],
      body,
      k,
      w,
    )
    subtree_mass = body_subtreemass[w % body_subtreemass.shape[0], root]
    if subtree_mass < MJ_MINVAL:
      mass = 1.0 if body == root else 0.0
    else:
      mass = body_mass[wm, body] / subtree_mass
    acc += mass * wp.dot(jacp, adj_subtree_com[w, root])
  wp.atomic_add(adj_dof_out, w, k, acc)  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _dof_to_qpos(
  # Model:
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  adj_dof: wp.array2d[float],
  # Out:
  adj_qpos_out: wp.array2d[float],
):
  w, j = wp.tid()
  jt = jnt_type[j]
  qadr = jnt_qposadr[j]
  dadr = jnt_dofadr[j]
  if jt == JointType.FREE or jt == JointType.BALL:
    if jt == JointType.FREE:
      adj_qpos_out[w, qadr + 0] += adj_dof[w, dadr + 0]
      adj_qpos_out[w, qadr + 1] += adj_dof[w, dadr + 1]
      adj_qpos_out[w, qadr + 2] += adj_dof[w, dadr + 2]
      qadr += 3
      dadr += 3
    q = wp.quat(qpos_in[w, qadr + 0], qpos_in[w, qadr + 1], qpos_in[w, qadr + 2], qpos_in[w, qadr + 3])
    grad = wp.vec3(adj_dof[w, dadr + 0], adj_dof[w, dadr + 1], adj_dof[w, dadr + 2])
    norm2 = wp.dot(q, q)
    adj_q = (2.0 / norm2) * math.quat_mul_axis(q, grad) if norm2 > 0.0 else wp.quat()
    adj_qpos_out[w, qadr + 0] += adj_q[0]
    adj_qpos_out[w, qadr + 1] += adj_q[1]
    adj_qpos_out[w, qadr + 2] += adj_q[2]
    adj_qpos_out[w, qadr + 3] += adj_q[3]
  else:
    adj_qpos_out[w, qadr] += adj_dof[w, dadr]


def kinematics_qpos_backward(
  m: Model,
  d: Data,
  qpos: wp.array2d[float],
  adj_cdof: wp.array2d[wp.spatial_vector],
  adj_subtree_com: wp.array2d[wp.vec3],
  adj_dof: wp.array2d[float],
  ceff: wp.array2d[wp.vec3],
  adj_qpos: wp.array2d[float],
):
  """Accumulates the cdof and subtree-COM adjoints into qpos."""
  # Split each DOF's scan across lanes while keeping the partial sums in registers.
  lanes = 8 if d.qpos.device.is_cuda else 1
  wp.launch(
    _cdof_qpos_vjp,
    dim=(d.nworld, m.nv, lanes),
    inputs=[m.nv, m.jnt_type, m.jnt_dofadr, m.dof_jntid, m.dof_parentid, d.cdof, lanes, adj_cdof],
    outputs=[adj_dof],
    block_dim=128,
  )
  wp.launch(
    _build_ceff,
    dim=(d.nworld, m.nbody),
    inputs=[m.nv, m.body_rootid, m.dof_bodyid, d.cdof, adj_cdof, adj_subtree_com],
    outputs=[ceff],
  )
  wp.launch(
    _subtree_com_qpos_vjp,
    dim=(d.nworld, m.nv, lanes),
    inputs=[
      m.nbody,
      m.body_parentid,
      m.body_rootid,
      m.body_mass,
      m.body_subtreemass,
      m.dof_bodyid,
      m.body_isdofancestor,
      d.xipos,
      d.subtree_com,
      d.cdof,
      lanes,
      ceff,
    ],
    outputs=[adj_dof],
    block_dim=128,
  )
  wp.launch(
    _dof_to_qpos,
    dim=(d.nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, qpos, adj_dof],
    outputs=[adj_qpos],
  )
