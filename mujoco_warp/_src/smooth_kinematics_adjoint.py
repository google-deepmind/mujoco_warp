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
from mujoco_warp._src import smooth
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"enable_backward": False})


@wp.func
def _joint_quat_vjp(quat: wp.quat, tangent: wp.vec3) -> wp.quat:
  """Lifts a joint-local angular cotangent into quaternion coordinates."""
  norm2 = wp.dot(quat, quat)
  return (2.0 / norm2) * math.quat_mul_axis(quat, tangent) if norm2 > 0.0 else wp.quat()


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
  xiquat_in: wp.array2d[wp.quat],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xquat_in: wp.array2d[wp.quat],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_xipos: wp.array2d[wp.vec3],
  adj_xiquat: wp.array2d[wp.quat],
  adj_geom_xpos: wp.array2d[wp.vec3],
  adj_geom_xquat: wp.array2d[wp.quat],
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
      pos_bar = adj_xipos[w, body]
      force += pos_bar
      torque += wp.cross(xipos_in[w, body] - origin, pos_bar)
      torque += adjoint_util._adj_quat_rotation(xiquat_in[w, body], adj_xiquat[w, body])
    if adj_subtree_com:
      pos_bar = adj_subtree_com[w, body]
      force += pos_bar
      torque += wp.cross(subtree_com_in[w, body] - origin, pos_bar)
  if adj_geom_xpos:
    for geom in range(ngeom):
      if body_rootid[geom_bodyid[geom]] == root:
        pos_bar = adj_geom_xpos[w, geom]
        force += pos_bar
        torque += wp.cross(geom_xpos_in[w, geom] - origin, pos_bar)
        torque += adjoint_util._adj_quat_rotation(geom_xquat_in[w, geom], adj_geom_xquat[w, geom])
  if adj_cdof:
    for dof in range(nv):
      if body_rootid[dof_bodyid[dof]] == root:
        value, cdof_bar = cdof_in[w, dof], adj_cdof[w, dof]
        torque += wp.cross(wp.spatial_top(value), wp.spatial_top(cdof_bar))
        torque += wp.cross(wp.spatial_bottom(value), wp.spatial_bottom(cdof_bar))
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
      d_bwd.xiquat,
      d_bwd.geom_xpos,
      d_bwd.geom_xquat,
      d_bwd.subtree_com,
      d_bwd.cdof,
      bc.adj_xipos if not contact else None,
      bc.adj_xiquat if not contact else None,
      bc.res_geom_xpos if contact else None,
      bc.res_geom_xquat if contact else None,
      bc.adj_subtree_com,
      bc.adj_cdof,
      scale,
    ],
    outputs=[bc.adj_mocap_pos, bc.adj_mocap_quat],
  )


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
  origin = wp.vec3()
  if b > 0 and body_rootid[b] == b:
    for index in range(nv):
      i = nv - 1 - index
      if body_rootid[dof_bodyid[i]] == b:
        adj = adj_cdof[w, i]
        origin += wp.cross(wp.spatial_top(cdof_in[w, i]), wp.spatial_bottom(adj))
  ceff_out[w, b] = adj_subtree_com[w, b] - origin


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
    adj_q = _joint_quat_vjp(q, grad)
    adj_qpos_out[w, qadr + 0] += adj_q[0]
    adj_qpos_out[w, qadr + 1] += adj_q[1]
    adj_qpos_out[w, qadr + 2] += adj_q[2]
    adj_qpos_out[w, qadr + 3] += adj_q[3]
  else:
    adj_qpos_out[w, qadr] += adj_dof[w, dadr]


@cache_kernel
def _kinematics_qpos_vjp(nbody: int, contact: bool):
  """Shares body wrenches across joint projections in bounded chunks."""
  tile_size = adjoint_util._body_tile_size(nbody, smooth.WORLD_SUM_BLOCK)

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    njnt: int,
    body_rootid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_geomnum: wp.array[int],
    body_geomadr: wp.array[int],
    body_mass: wp.array2d[float],
    body_subtreemass: wp.array2d[float],
    jnt_type: wp.array[int],
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_bodyid: wp.array[int],
    dof_jntid: wp.array[int],
    geom_pos: wp.array2d[wp.vec3],
    body_subtreenum: wp.array[int],
    # Data in:
    qpos_in: wp.array2d[float],
    xquat_in: wp.array2d[wp.quat],
    xipos_in: wp.array2d[wp.vec3],
    xanchor_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    # In:
    pose_xpos_in: wp.array2d[wp.vec3],
    pose_xquat_in: wp.array2d[wp.quat],
    adj_pos: wp.array2d[wp.vec3],
    adj_quat: wp.array2d[wp.quat],
    adj_cdof: wp.array2d[wp.spatial_vector],
    adj_com: wp.array2d[wp.vec3],
    ceff: wp.array2d[wp.vec3],
    # Out:
    adj_dof_out: wp.array2d[float],
    adj_qpos_out: wp.array2d[float],
  ):
    w, lane = wp.tid()
    positions = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.vec3, storage="shared")
    poses = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")
    motions = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")
    coms = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.spatial_vector, storage="shared")
    # Preserve per-term free-root rotation contractions instead of cancelling summed wrenches.
    root_poses = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.vec3, storage="shared")
    root_motions = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.vec3, storage="shared")
    root_coms = wp.tile_empty(shape=(wp.static(tile_size),), dtype=wp.vec3, storage="shared")

    for body_start in range(0, wp.static(nbody), wp.static(tile_size)):
      for start in range(0, wp.static(tile_size), wp.block_dim()):
        body = body_start + start + lane
        position = wp.vec3()
        pose, motion, com = wp.spatial_vector(), wp.spatial_vector(), wp.spatial_vector()
        root_pose, root_motion, root_com = wp.vec3(), wp.vec3(), wp.vec3()
        if body > 0 and body < wp.static(nbody):
          position = xipos_in[w, body]
          root = body_rootid[body]
          origin = subtree_com_in[w, root]
          root_dadr = body_dofadr[root]
          root_joint = int(0)
          free_root = False
          if root_dadr >= 0 and body_dofnum[root] > 0:
            root_joint = dof_jntid[root_dadr]
            free_root = jnt_type[root_joint] == JointType.FREE
          first, count = body, int(1)
          if wp.static(contact):
            first = body_geomadr[body]
            count = body_geomnum[body]
          for item in range(first, first + count):
            force = adj_pos[w, item]
            quat_bar = adj_quat[w, item]
            if force == wp.vec3() and quat_bar == wp.quat():
              continue
            point = pose_xpos_in[w, item]
            torque = adjoint_util._adj_quat_rotation(pose_xquat_in[w, item], quat_bar)
            pose += smooth.transform_force(force, torque, position - point)
            if free_root:
              local_point = wp.vec3()
              if wp.static(contact):
                local_point = geom_pos[w % geom_pos.shape[0], item]
              for axis in range(3):
                s = cdof_in[w, root_dadr + 3 + axis]
                jacp = adjoint_util._jac_dof_pos(
                  wp.spatial_bottom(s),
                  wp.spatial_top(s),
                  point,
                  xanchor_in[w, root_joint],
                  xquat_in[w, body],
                  local_point,
                  wp.static(contact) and body == root,
                )
                root_pose[axis] += wp.dot(jacp, force) + wp.dot(wp.spatial_top(s), torque)
          first = body_dofadr[body]
          for dof in range(first, first + body_dofnum[body]):
            joint = dof_jntid[dof]
            if jnt_type[joint] == JointType.FREE and dof - jnt_dofadr[joint] < 3:
              continue
            motion += math.motion_cross_force(cdof_in[w, dof], adj_cdof[w, dof])
            if free_root:
              for axis in range(3):
                s = cdof_in[w, root_dadr + 3 + axis]
                root_motion[axis] += wp.dot(adj_cdof[w, dof], math.motion_cross(s, cdof_in[w, dof]))
          mass = body_subtreemass[w % body_subtreemass.shape[0], root]
          weight = float(0.0)
          if mass < MJ_MINVAL:
            weight = 1.0 if body == root else 0.0
          else:
            weight = body_mass[w % body_mass.shape[0], body] / mass
          force = weight * ceff[w, root]
          com = smooth.transform_force(force, wp.vec3(), origin - position)
          if free_root:
            for axis in range(3):
              s = cdof_in[w, root_dadr + 3 + axis]
              jacp = wp.spatial_bottom(s) + wp.cross(wp.spatial_top(s), position - origin)
              root_com[axis] = weight * wp.dot(jacp, ceff[w, root])
        wp.tile_assign(positions, wp.tile(position, preserve_type=True), offset=(start,))
        wp.tile_assign(poses, wp.tile(pose, preserve_type=True), offset=(start,))
        wp.tile_assign(motions, wp.tile(motion, preserve_type=True), offset=(start,))
        wp.tile_assign(coms, wp.tile(com, preserve_type=True), offset=(start,))
        wp.tile_assign(root_poses, wp.tile(root_pose, preserve_type=True), offset=(start,))
        wp.tile_assign(root_motions, wp.tile(root_motion, preserve_type=True), offset=(start,))
        wp.tile_assign(root_coms, wp.tile(root_com, preserve_type=True), offset=(start,))

      for joint in range(lane, njnt, wp.block_dim()):
        body = jnt_bodyid[joint]
        dadr = jnt_dofadr[joint]
        kind = jnt_type[joint]
        pose_sum, motion_sum, com_sum = wp.spatial_vector(), wp.spatial_vector(), wp.spatial_vector()
        root_pose_sum, root_motion_sum, root_com_sum = wp.vec3(), wp.vec3(), wp.vec3()
        end = wp.min(body + body_subtreenum[body], body_start + wp.static(tile_size))
        for child in range(wp.max(body, body_start), end):
          index = child - body_start
          if kind == JointType.FREE:
            root_pose_sum += root_poses[index]
            root_motion_sum += root_motions[index]
            root_com_sum += root_coms[index]
          else:
            wrench = poses[index]
            force = wp.spatial_bottom(wrench)
            pose_sum += smooth.transform_force(force, wp.spatial_top(wrench), xanchor_in[w, joint] - positions[index])
            com_sum += coms[index]
            if child > body:
              motion_sum += motions[index]
        if kind != JointType.FREE and body >= body_start and body < body_start + wp.static(tile_size):
          first = body_dofadr[body]
          for dof in range(first, first + body_dofnum[body]):
            other_joint = dof_jntid[dof]
            if jnt_type[other_joint] == JointType.FREE and dof - jnt_dofadr[other_joint] < 3:
              continue
            coupled = other_joint == joint and (kind == JointType.FREE or kind == JointType.BALL)
            if dadr < dof or coupled:
              motion_sum += math.motion_cross_force(cdof_in[w, dof], adj_cdof[w, dof])
        if kind == JointType.FREE:
          dadr += 3
        count = 3 if kind == JointType.FREE or kind == JointType.BALL else 1
        for axis in range(count):
          if kind == JointType.FREE:
            value = root_pose_sum[axis] + root_motion_sum[axis] + root_com_sum[axis]
          else:
            s = cdof_in[w, dadr + axis]
            pose_value = wp.dot(wp.spatial_top(s), wp.spatial_top(pose_sum))
            if kind == JointType.SLIDE:
              pose_value = wp.dot(wp.spatial_bottom(s), wp.spatial_bottom(pose_sum))
            value = pose_value + wp.dot(s, motion_sum) + wp.dot(s, com_sum)
          if body_start > 0:
            value += adj_dof_out[w, dadr + axis]
          adj_dof_out[w, dadr + axis] = value
      if wp.static(nbody > tile_size):
        # Synchronize readers before the next chunk overwrites shared storage.
        wp.tile_assign(positions, positions)

    for joint in range(lane, njnt, wp.block_dim()):
      kind = jnt_type[joint]
      qadr = jnt_qposadr[joint]
      dadr = jnt_dofadr[joint]
      if kind == JointType.FREE:
        body = jnt_bodyid[joint]
        translation, translation_error = wp.vec3(), wp.vec3()
        mass_sum, mass_error = float(0.0), float(0.0)
        # Sum whole-tree translations directly, compensating geometry and mass sums.
        for child in range(body, body + body_subtreenum[body]):
          first, count = child, int(1)
          if wp.static(contact):
            first = body_geomadr[child]
            count = body_geomnum[child]
          for item in range(first, first + count):
            if wp.static(contact):
              addend = adj_pos[w, item] - translation_error
              total = translation + addend
              translation_error = (total - translation) - addend
              translation = total
            else:
              # Match the body order used to accumulate the cinert COM cotangent.
              translation += adj_pos[w, item]
          mass_addend = body_mass[w % body_mass.shape[0], child] - mass_error
          mass_total = mass_sum + mass_addend
          mass_error = (mass_total - mass_sum) - mass_addend
          mass_sum = mass_total
        translation += adj_com[w, body]
        mass = body_subtreemass[w % body_subtreemass.shape[0], body]
        if mass >= MJ_MINVAL:
          translation += ((mass_sum - mass) / mass) * ceff[w, body]
        for axis in range(3):
          adj_dof_out[w, dadr + axis] = translation[axis]
          adj_qpos_out[w, qadr + axis] += translation[axis]
        qadr += 3
        dadr += 3
      if kind == JointType.FREE or kind == JointType.BALL:
        q = wp.quat(qpos_in[w, qadr], qpos_in[w, qadr + 1], qpos_in[w, qadr + 2], qpos_in[w, qadr + 3])
        tangent = wp.vec3(adj_dof_out[w, dadr], adj_dof_out[w, dadr + 1], adj_dof_out[w, dadr + 2])
        quat_grad = _joint_quat_vjp(q, tangent)
        for axis in range(4):
          adj_qpos_out[w, qadr + axis] += quat_grad[axis]
      else:
        adj_qpos_out[w, qadr] += adj_dof_out[w, dadr]

  return kernel


def kinematics_qpos_vjp(
  m: Model,
  d: Data,
  qpos: wp.array2d,
  adj_pos: wp.array2d,
  adj_quat: wp.array2d,
  adj_cdof: wp.array2d,
  adj_com: wp.array2d,
  ceff: wp.array2d,
  adj_dof: wp.array2d,
  adj_qpos: wp.array2d,
  *,
  contact: bool = False,
):
  """Maps pose, CDOF and root-COM cotangents to qpos, with model and mocap inputs fixed.

  Contact selects geometry poses; otherwise poses are body-inertial frames.
  Adds to adj_qpos and overwrites adj_dof and ceff.
  """
  wp.launch(
    _build_ceff,
    dim=(d.nworld, m.nbody),
    inputs=[m.nv, m.body_rootid, m.dof_bodyid, d.cdof, adj_cdof, adj_com],
    outputs=[ceff],
  )
  wp.launch_tiled(
    _kinematics_qpos_vjp(m.nbody, contact),
    dim=d.nworld,
    inputs=[
      m.njnt,
      m.body_rootid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_geomnum,
      m.body_geomadr,
      m.body_mass,
      m.body_subtreemass,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_bodyid,
      m.dof_jntid,
      m.geom_pos,
      m.body_subtreenum,
      qpos,
      d.xquat,
      d.xipos,
      d.xanchor,
      d.subtree_com,
      d.cdof,
      d.geom_xpos if contact else d.xipos,
      d.geom_xquat if contact else d.xiquat,
      adj_pos,
      adj_quat,
      adj_cdof,
      adj_com,
      ceff,
    ],
    outputs=[adj_dof, adj_qpos],
    block_dim=adjoint_util._body_tile_size(m.nbody, smooth.WORLD_SUM_BLOCK),
  )
