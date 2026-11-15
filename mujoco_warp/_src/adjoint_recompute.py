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
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import JointType

# Adjoint reductions use unordered atomics; deterministic backward execution is deferred.
wp.set_module_options({"enable_backward": True})


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
  # In:
  spring_params_grad: bool,
  # Data out:
  qfrc_spring_out: wp.array2d[float],
):
  w, jntid = wp.tid()
  jnttype = jnt_type[jntid]
  dofid = jnt_dofadr[jntid]
  stiffness = jnt_stiffness[w % jnt_stiffness.shape[0], jntid]
  stiffnesspoly = jnt_stiffnesspoly[w % jnt_stiffnesspoly.shape[0], jntid]
  # A zero coefficient still has a parameter derivative when fitting springs.
  has_stiffness = (spring_params_grad or stiffness != 0.0 or stiffnesspoly[0] != 0.0 or stiffnesspoly[1] != 0.0) and (
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
