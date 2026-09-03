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

from mujoco_warp._src import math
from mujoco_warp._src import support
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model

wp.set_module_options({"enable_backward": False})


@wp.kernel
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
  adj_cdof: wp.array2d[wp.spatial_vector],
  # Out:
  adj_dof_out: wp.array2d[float],
):
  w, k = wp.tid()
  cdof_k = cdof_in[w, k]
  jk = dof_jntid[k]
  acc = float(0.0)
  for i in range(nv):
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
  adj_dof_out[w, k] += acc


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


@wp.kernel
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
  adj_subtree_com: wp.array2d[wp.vec3],
  # Out:
  adj_dof_out: wp.array2d[float],
):
  w, k = wp.tid()
  wm = w % body_mass.shape[0]
  acc = float(0.0)
  for body in range(1, nbody):
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
    mass = body_mass[wm, body] / body_subtreemass[w % body_subtreemass.shape[0], root]
    acc += mass * wp.dot(jacp, adj_subtree_com[w, root])
  adj_dof_out[w, k] += acc


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
    adj_q = 2.0 * math.quat_mul_axis(q, grad)
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
  wp.launch(
    _cdof_qpos_vjp,
    dim=(d.nworld, m.nv),
    inputs=[m.nv, m.jnt_type, m.jnt_dofadr, m.dof_jntid, m.dof_parentid, d.cdof, adj_cdof],
    outputs=[adj_dof],
  )
  wp.launch(
    _build_ceff,
    dim=(d.nworld, m.nbody),
    inputs=[m.nv, m.body_rootid, m.dof_bodyid, d.cdof, adj_cdof, adj_subtree_com],
    outputs=[ceff],
  )
  wp.launch(
    _subtree_com_qpos_vjp,
    dim=(d.nworld, m.nv),
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
      ceff,
    ],
    outputs=[adj_dof],
  )
  wp.launch(
    _dof_to_qpos,
    dim=(d.nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, qpos, adj_dof],
    outputs=[adj_qpos],
  )
