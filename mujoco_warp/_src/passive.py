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

import warp as wp

from . import math
from . import support
from .types import Data
from .types import DisableBit
from .types import JointType
from .types import Model
from .warp_util import event_scope


@wp.kernel
def _spring_damper_dof_passive(
  # Model:
  qpos_spring: wp.array(dtype=float),
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  jnt_stiffness: wp.array(dtype=float),
  dof_damping: wp.array(dtype=float),
  # Data in:
  qpos_in: wp.array2d(dtype=float),
  qvel_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_spring_out: wp.array2d(dtype=float),
  qfrc_damper_out: wp.array2d(dtype=float),
):
  worldid, jntid = wp.tid()
  stiffness = jnt_stiffness[jntid]
  dofid = jnt_dofadr[jntid]
  damping = dof_damping[dofid]

  if stiffness == 0.0 and damping == 0.0:
    return

  jnttype = jnt_type[jntid]
  qposid = jnt_qposadr[jntid]

  if jnttype == wp.static(JointType.FREE.value):
    # spring
    dif = wp.vec3(
      qpos_in[worldid, qposid + 0] - qpos_spring[qposid + 0],
      qpos_in[worldid, qposid + 1] - qpos_spring[qposid + 1],
      qpos_in[worldid, qposid + 2] - qpos_spring[qposid + 2],
    )
    qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]
    qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]
    qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]
    rot = wp.quat(
      qpos_in[worldid, qposid + 3],
      qpos_in[worldid, qposid + 4],
      qpos_in[worldid, qposid + 5],
      qpos_in[worldid, qposid + 6],
    )
    ref = wp.quat(
      qpos_spring[qposid + 3],
      qpos_spring[qposid + 4],
      qpos_spring[qposid + 5],
      qpos_spring[qposid + 6],
    )
    dif = math.quat_sub(rot, ref)
    qfrc_spring_out[worldid, dofid + 3] = -stiffness * dif[0]
    qfrc_spring_out[worldid, dofid + 4] = -stiffness * dif[1]
    qfrc_spring_out[worldid, dofid + 5] = -stiffness * dif[2]

    # damper
    qfrc_damper_out[worldid, dofid + 0] = -damping * qvel_in[worldid, dofid + 0]
    qfrc_damper_out[worldid, dofid + 1] = -damping * qvel_in[worldid, dofid + 1]
    qfrc_damper_out[worldid, dofid + 2] = -damping * qvel_in[worldid, dofid + 2]
    qfrc_damper_out[worldid, dofid + 3] = -damping * qvel_in[worldid, dofid + 3]
    qfrc_damper_out[worldid, dofid + 4] = -damping * qvel_in[worldid, dofid + 4]
    qfrc_damper_out[worldid, dofid + 5] = -damping * qvel_in[worldid, dofid + 5]
  elif jnttype == wp.static(JointType.BALL.value):
    # spring
    rot = wp.quat(
      qpos_in[worldid, qposid + 0],
      qpos_in[worldid, qposid + 1],
      qpos_in[worldid, qposid + 2],
      qpos_in[worldid, qposid + 3],
    )
    ref = wp.quat(
      qpos_spring[qposid + 0],
      qpos_spring[qposid + 1],
      qpos_spring[qposid + 2],
      qpos_spring[qposid + 3],
    )
    dif = math.quat_sub(rot, ref)
    qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]
    qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]
    qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]

    # damper
    qfrc_damper_out[worldid, dofid + 0] = -damping * qvel_in[worldid, dofid + 0]
    qfrc_damper_out[worldid, dofid + 1] = -damping * qvel_in[worldid, dofid + 1]
    qfrc_damper_out[worldid, dofid + 2] = -damping * qvel_in[worldid, dofid + 2]
  else:  # mjJNT_SLIDE, mjJNT_HINGE
    # spring
    fdif = qpos_in[worldid, qposid] - qpos_spring[qposid]
    qfrc_spring_out[worldid, dofid] = -stiffness * fdif

    # damper
    qfrc_damper_out[worldid, dofid] = -damping * qvel_in[worldid, dofid]


@wp.kernel
def _spring_damper_tendon_passive(
  # Model:
  tendon_stiffness: wp.array(dtype=float),
  tendon_damping: wp.array(dtype=float),
  tendon_lengthspring: wp.array(dtype=wp.vec2),
  # Data in:
  ten_velocity_in: wp.array2d(dtype=float),
  ten_length_in: wp.array2d(dtype=float),
  ten_J_in: wp.array3d(dtype=float),
  # Data out:
  qfrc_spring_out: wp.array2d(dtype=float),
  qfrc_damper_out: wp.array2d(dtype=float),
):
  worldid, tenid, dofid = wp.tid()

  stiffness = tendon_stiffness[tenid]
  damping = tendon_damping[tenid]

  if stiffness == 0.0 and damping == 0.0:
    return

  J = ten_J_in[worldid, tenid, dofid]

  if stiffness:
    # compute spring force along tendon
    length = ten_length_in[worldid, tenid]
    lengthspring = tendon_lengthspring[tenid]
    lower = lengthspring[0]
    upper = lengthspring[1]

    if length > upper:
      frc_spring = stiffness * (upper - length)
    elif length < lower:
      frc_spring = stiffness * (lower - length)
    else:
      frc_spring = 0.0

    # transform to joint torque
    wp.atomic_add(qfrc_spring_out[worldid], dofid, J * frc_spring)

  if damping:
    # compute damper linear force along tendon
    frc_damper = -damping * ten_velocity_in[worldid, tenid]

    # transform to joint torque
    wp.atomic_add(qfrc_damper_out[worldid], dofid, J * frc_damper)


@wp.kernel
def _gravity_force(
  # Model:
  opt_gravity: wp.vec3,
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_mass: wp.array(dtype=float),
  body_gravcomp: wp.array(dtype=float),
  dof_bodyid: wp.array(dtype=int),
  # Data in:
  xipos_in: wp.array2d(dtype=wp.vec3),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  # Data out:
  qfrc_gravcomp_out: wp.array2d(dtype=float),
):
  worldid, bodyid, dofid = wp.tid()
  bodyid += 1  # skip world body
  gravcomp = body_gravcomp[bodyid]

  if gravcomp:
    force = -opt_gravity * body_mass[bodyid] * gravcomp

    pos = xipos_in[worldid, bodyid]
    jac, _ = support.jac(body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, pos, bodyid, dofid, worldid)

    wp.atomic_add(qfrc_gravcomp_out[worldid], dofid, wp.dot(jac, force))


@wp.kernel
def _qfrc_passive(
  # Model:
  jnt_actgravcomp: wp.array(dtype=int),
  dof_jntid: wp.array(dtype=int),
  # Data in:
  qfrc_spring_in: wp.array2d(dtype=float),
  qfrc_damper_in: wp.array2d(dtype=float),
  qfrc_gravcomp_in: wp.array2d(dtype=float),
  # In:
  gravcomp: bool,
  # Data out:
  qfrc_passive_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qfrc_passive = qfrc_spring_in[worldid, dofid]
  qfrc_passive += qfrc_damper_in[worldid, dofid]

  # add gravcomp unless added by actuators
  if gravcomp and not jnt_actgravcomp[dof_jntid[dofid]]:
    qfrc_passive += qfrc_gravcomp_in[worldid, dofid]

  qfrc_passive_out[worldid, dofid] = qfrc_passive


@event_scope
def passive(m: Model, d: Data):
  """Adds all passive forces."""
  if m.opt.disableflags & DisableBit.PASSIVE:
    d.qfrc_spring.zero_()
    d.qfrc_damper.zero_()
    d.qfrc_gravcomp.zero_()
    d.qfrc_passive.zero_()
    return

  # TODO(team): mj_ellipsoidFluidModel
  # TODO(team): mj_inertiaBoxFluidModell

  d.qfrc_spring.zero_()
  d.qfrc_damper.zero_()
  wp.launch(
    _spring_damper_dof_passive,
    dim=(d.nworld, m.njnt),
    inputs=[
      m.qpos_spring,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_stiffness,
      m.dof_damping,
      d.qpos,
      d.qvel,
    ],
    outputs=[d.qfrc_spring, d.qfrc_damper],
  )

  if m.ntendon:
    wp.launch(
      _spring_damper_tendon_passive,
      dim=(d.nworld, m.ntendon, m.nv),
      inputs=[
        m.tendon_stiffness,
        m.tendon_damping,
        m.tendon_lengthspring,
        d.ten_velocity,
        d.ten_length,
        d.ten_J,
      ],
      outputs=[
        d.qfrc_spring,
        d.qfrc_damper,
      ],
    )

  d.qfrc_gravcomp.zero_()
  gravcomp = m.ngravcomp and not (m.opt.disableflags & DisableBit.GRAVITY)
  if gravcomp:
    wp.launch(
      _gravity_force,
      dim=(d.nworld, m.nbody - 1, m.nv),
      inputs=[
        m.opt.gravity,
        m.body_parentid,
        m.body_rootid,
        m.body_mass,
        m.body_gravcomp,
        m.dof_bodyid,
        d.xipos,
        d.subtree_com,
        d.cdof,
      ],
      outputs=[d.qfrc_gravcomp],
    )

  wp.launch(
    _qfrc_passive,
    dim=(d.nworld, m.nv),
    inputs=[m.jnt_actgravcomp, m.dof_jntid, d.qfrc_spring, d.qfrc_damper, d.qfrc_gravcomp, gravcomp],
    outputs=[d.qfrc_passive],
  )
