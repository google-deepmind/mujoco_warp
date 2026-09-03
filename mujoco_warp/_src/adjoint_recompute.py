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
"""Differentiable replays of gradient-disabled forward kernels.

Each kernel mirrors the forward implementation named above it. Keep the pair in sync until
the forward kernel can be differentiated directly.
"""

import warp as wp

from mujoco_warp._src import math
from mujoco_warp._src import support
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import vec10

wp.set_module_options({"enable_backward": True})


# smooth._compute_body_inertial_frames
@wp.kernel
def _inertial_frames_recompute(
  # Model:
  body_ipos: wp.array2d[wp.vec3],
  body_iquat: wp.array2d[wp.quat],
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  # Data out:
  xipos_out: wp.array2d[wp.vec3],
  ximat_out: wp.array2d[wp.mat33],
):
  w, b = wp.tid()
  xpos = xpos_in[w, b]
  xquat = xquat_in[w, b]
  xipos_out[w, b] = xpos + math.rot_vec_quat(body_ipos[w % body_ipos.shape[0], b], xquat)
  ximat_out[w, b] = math.quat_to_mat(math.mul_quat(xquat, body_iquat[w % body_iquat.shape[0], b]))


# passive._spring_damper_dof_passive (spring path)
@wp.kernel(enable_backward=True)
def _spring_qfrc_recompute(
  # Model:
  opt_disableflags: int,
  qpos_spring: wp.array2d[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_stiffness: wp.array2d[float],
  jnt_stiffnesspoly: wp.array2d[wp.vec2],
  # Data in:
  qpos_in: wp.array2d[float],
  # Data out:
  qfrc_spring_out: wp.array2d[float],
):
  w, jntid = wp.tid()
  jnttype = jnt_type[jntid]
  dofid = jnt_dofadr[jntid]
  stiffness = jnt_stiffness[w % jnt_stiffness.shape[0], jntid]
  stiffnesspoly = jnt_stiffnesspoly[w % jnt_stiffnesspoly.shape[0], jntid]
  has_stiffness = (stiffness != 0.0 or stiffnesspoly[0] != 0.0 or stiffnesspoly[1] != 0.0) and (
    opt_disableflags & DisableBit.SPRING
  ) == 0
  if not has_stiffness:
    return

  qposid = jnt_qposadr[jntid]
  spring_worldid = w % qpos_spring.shape[0]
  if jnttype == JointType.FREE:
    difx = qpos_in[w, qposid + 0] - qpos_spring[spring_worldid, qposid + 0]
    dify = qpos_in[w, qposid + 1] - qpos_spring[spring_worldid, qposid + 1]
    difz = qpos_in[w, qposid + 2] - qpos_spring[spring_worldid, qposid + 2]
    stiffness_lin = util_misc._poly_force(stiffness, stiffnesspoly, wp.length(wp.vec3(difx, dify, difz)), 0)
    qfrc_spring_out[w, dofid + 0] = -stiffness_lin * difx
    qfrc_spring_out[w, dofid + 1] = -stiffness_lin * dify
    qfrc_spring_out[w, dofid + 2] = -stiffness_lin * difz
    rot = wp.normalize(wp.quat(qpos_in[w, qposid + 3], qpos_in[w, qposid + 4], qpos_in[w, qposid + 5], qpos_in[w, qposid + 6]))
    ref = wp.quat(
      qpos_spring[spring_worldid, qposid + 3],
      qpos_spring[spring_worldid, qposid + 4],
      qpos_spring[spring_worldid, qposid + 5],
      qpos_spring[spring_worldid, qposid + 6],
    )
    dif = math.quat_sub(rot, ref)
    stiffness_rot = util_misc._poly_force(stiffness, stiffnesspoly, wp.length(dif), 0)
    qfrc_spring_out[w, dofid + 3] = -stiffness_rot * dif[0]
    qfrc_spring_out[w, dofid + 4] = -stiffness_rot * dif[1]
    qfrc_spring_out[w, dofid + 5] = -stiffness_rot * dif[2]
  elif jnttype == JointType.BALL:
    rot = wp.normalize(wp.quat(qpos_in[w, qposid + 0], qpos_in[w, qposid + 1], qpos_in[w, qposid + 2], qpos_in[w, qposid + 3]))
    ref = wp.quat(
      qpos_spring[spring_worldid, qposid + 0],
      qpos_spring[spring_worldid, qposid + 1],
      qpos_spring[spring_worldid, qposid + 2],
      qpos_spring[spring_worldid, qposid + 3],
    )
    dif = math.quat_sub(rot, ref)
    stiffness_rot = util_misc._poly_force(stiffness, stiffnesspoly, wp.length(dif), 0)
    qfrc_spring_out[w, dofid + 0] = -stiffness_rot * dif[0]
    qfrc_spring_out[w, dofid + 1] = -stiffness_rot * dif[1]
    qfrc_spring_out[w, dofid + 2] = -stiffness_rot * dif[2]
  else:
    fdif = qpos_in[w, qposid] - qpos_spring[spring_worldid, qposid]
    qfrc_spring_out[w, dofid] = -fdif * util_misc._poly_force(stiffness, stiffnesspoly, fdif, 0)


# passive._gravity_force
@wp.kernel(enable_backward=True)
def _gravity_force_recompute(
  # Model:
  opt_gravity: wp.array[wp.vec3],
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_gravcomp: wp.array2d[float],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # Data out:
  qfrc_gravcomp_out: wp.array2d[float],
):
  worldid, bodyid, dofid = wp.tid()
  bodyid += 1
  gravcomp = body_gravcomp[worldid % body_gravcomp.shape[0], bodyid]
  gravity = opt_gravity[worldid % opt_gravity.shape[0]]
  if gravcomp:
    force = -gravity * body_mass[worldid % body_mass.shape[0], bodyid] * gravcomp
    jac, _ = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      xipos_in[worldid, bodyid],
      bodyid,
      dofid,
      worldid,
    )
    wp.atomic_add(qfrc_gravcomp_out[worldid], dofid, wp.dot(jac, force))


# smooth._cfrc
@wp.kernel
def _rne_cfrc_recompute(
  # Data in:
  cinert_in: wp.array2d[vec10],
  cvel_in: wp.array2d[wp.spatial_vector],
  cacc_in: wp.array2d[wp.spatial_vector],
  # Out:
  cfrc_local_out: wp.array2d[wp.spatial_vector],
):
  w, b = wp.tid()
  frc = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  if b != 0:
    ci = cinert_in[w, b]
    cv = cvel_in[w, b]
    frc = math.inert_vec(ci, cacc_in[w, b]) + math.motion_cross_force(cv, math.inert_vec(ci, cv))
  cfrc_local_out[w, b] = frc


# smooth._cinert
@wp.kernel
def _cinert_recompute(
  # Model:
  body_rootid: wp.array[int],
  body_mass: wp.array2d[float],
  body_inertia: wp.array2d[wp.vec3],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  ximat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  # Data out:
  cinert_out: wp.array2d[vec10],
):
  w, b = wp.tid()
  mat = ximat_in[w, b]
  inert = body_inertia[w % body_inertia.shape[0], b]
  mass = body_mass[w % body_mass.shape[0], b]
  dif = xipos_in[w, b] - subtree_com_in[w, body_rootid[b]]
  # A single constructor avoids Warp's incorrect reverse through component writes.
  i0 = inert[0]
  i1 = inert[1]
  i2 = inert[2]
  d0 = dif[0]
  d1 = dif[1]
  d2 = dif[2]
  r0 = i0 * mat[0, 0] * mat[0, 0] + i1 * mat[0, 1] * mat[0, 1] + i2 * mat[0, 2] * mat[0, 2] + mass * (d1 * d1 + d2 * d2)
  r1 = i0 * mat[1, 0] * mat[1, 0] + i1 * mat[1, 1] * mat[1, 1] + i2 * mat[1, 2] * mat[1, 2] + mass * (d0 * d0 + d2 * d2)
  r2 = i0 * mat[2, 0] * mat[2, 0] + i1 * mat[2, 1] * mat[2, 1] + i2 * mat[2, 2] * mat[2, 2] + mass * (d0 * d0 + d1 * d1)
  r3 = i0 * mat[0, 0] * mat[1, 0] + i1 * mat[0, 1] * mat[1, 1] + i2 * mat[0, 2] * mat[1, 2] - mass * d0 * d1
  r4 = i0 * mat[0, 0] * mat[2, 0] + i1 * mat[0, 1] * mat[2, 1] + i2 * mat[0, 2] * mat[2, 2] - mass * d0 * d2
  r5 = i0 * mat[1, 0] * mat[2, 0] + i1 * mat[1, 1] * mat[2, 1] + i2 * mat[1, 2] * mat[2, 2] - mass * d1 * d2
  cinert_out[w, b] = vec10(r0, r1, r2, r3, r4, r5, mass * d0, mass * d1, mass * d2, mass)
