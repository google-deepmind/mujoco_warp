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
from .warp_util import kernel


@event_scope
def passive(m: Model, d: Data):
  """Adds all passive forces."""
  if m.opt.disableflags & DisableBit.PASSIVE:
    d.qfrc_passive.zero_()
    d.qfrc_gravcomp.zero_()
    return

  @kernel
  def _spring(m: Model, d: Data):
    worldid, jntid = wp.tid()
    stiffness = m.jnt_stiffness[jntid]
    dofid = m.jnt_dofadr[jntid]

    if stiffness == 0.0:
      return

    jnt_type = m.jnt_type[jntid]
    qposid = m.jnt_qposadr[jntid]

    if jnt_type == wp.static(JointType.FREE.value):
      dif = wp.vec3(
        d.qpos[worldid, qposid + 0] - m.qpos_spring[qposid + 0],
        d.qpos[worldid, qposid + 1] - m.qpos_spring[qposid + 1],
        d.qpos[worldid, qposid + 2] - m.qpos_spring[qposid + 2],
      )
      d.qfrc_spring[worldid, dofid + 0] = -stiffness * dif[0]
      d.qfrc_spring[worldid, dofid + 1] = -stiffness * dif[1]
      d.qfrc_spring[worldid, dofid + 2] = -stiffness * dif[2]
      rot = wp.quat(
        d.qpos[worldid, qposid + 3],
        d.qpos[worldid, qposid + 4],
        d.qpos[worldid, qposid + 5],
        d.qpos[worldid, qposid + 6],
      )
      ref = wp.quat(
        m.qpos_spring[qposid + 3],
        m.qpos_spring[qposid + 4],
        m.qpos_spring[qposid + 5],
        m.qpos_spring[qposid + 6],
      )
      dif = math.quat_sub(rot, ref)
      d.qfrc_spring[worldid, dofid + 3] = -stiffness * dif[0]
      d.qfrc_spring[worldid, dofid + 4] = -stiffness * dif[1]
      d.qfrc_spring[worldid, dofid + 5] = -stiffness * dif[2]
    elif jnt_type == wp.static(JointType.BALL.value):
      rot = wp.quat(
        d.qpos[worldid, qposid + 0],
        d.qpos[worldid, qposid + 1],
        d.qpos[worldid, qposid + 2],
        d.qpos[worldid, qposid + 3],
      )
      ref = wp.quat(
        m.qpos_spring[qposid + 0],
        m.qpos_spring[qposid + 1],
        m.qpos_spring[qposid + 2],
        m.qpos_spring[qposid + 3],
      )
      dif = math.quat_sub(rot, ref)
      d.qfrc_spring[worldid, dofid + 0] = -stiffness * dif[0]
      d.qfrc_spring[worldid, dofid + 1] = -stiffness * dif[1]
      d.qfrc_spring[worldid, dofid + 2] = -stiffness * dif[2]
    else:  # mjJNT_SLIDE, mjJNT_HINGE
      fdif = d.qpos[worldid, qposid] - m.qpos_spring[qposid]
      d.qfrc_spring[worldid, dofid] = -stiffness * fdif

  @kernel
  def _damper_passive(m: Model, d: Data):
    worldid, dofid = wp.tid()
    damping = m.dof_damping[dofid]
    qfrc_damper = -damping * d.qvel[worldid, dofid]

    d.qfrc_damper[worldid, dofid] = qfrc_damper
    d.qfrc_passive[worldid, dofid] = qfrc_damper + d.qfrc_spring[worldid, dofid]

  # TODO(team): mj_ellipsoidFluidModel
  # TODO(team): mj_inertiaBoxFluidModell

  d.qfrc_spring.zero_()
  wp.launch(_spring, dim=(d.nworld, m.njnt), inputs=[m, d])
  wp.launch(_damper_passive, dim=(d.nworld, m.nv), inputs=[m, d])

  if m.ngravcomp and not (m.opt.disableflags & DisableBit.GRAVITY):
    _gravcomp(m, d)

    # add gravcomp unless added by actuators
    @kernel
    def _qfrc_passive_gravcomp(m: Model, d: Data):
      worldid, dofid = wp.tid()

      if m.jnt_actgravcomp[m.dof_jntid[dofid]]:
        return

      d.qfrc_passive[worldid, dofid] += d.qfrc_gravcomp[worldid, dofid]

    wp.launch(_qfrc_passive_gravcomp, dim=(d.nworld, m.nv), inputs=[m, d])


def _gravcomp(m: Model, d: Data):
  """Applies body-level gravity compensation."""

  d.qfrc_gravcomp.zero_()

  @kernel
  def _gravity_force(m: Model, d: Data):
    worldid, bodyid, dofid = wp.tid()
    bodyid += 1  # skip world body

    gravcomp = m.body_gravcomp[bodyid]

    if gravcomp:
      force = -m.opt.gravity * m.body_mass[bodyid] * gravcomp

      pos = d.xipos[worldid, bodyid]
      jac, _ = support.jac(m, d, pos, bodyid, dofid, worldid)

      wp.atomic_add(d.qfrc_gravcomp[worldid], dofid, wp.dot(jac, force))

  wp.launch(_gravity_force, dim=(d.nworld, m.nbody - 1, m.nv), inputs=[m, d])
