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
from .types import MJ_MINVAL
from .types import Data
from .types import DisableBit
from .types import JointType
from .types import Model
from .types import Option
from .warp_util import event_scope


@wp.kernel
def _spring_passive(
  # Model:
  qpos_spring: wp.array2d(dtype=float),
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  jnt_stiffness: wp.array2d(dtype=float),
  # Data in:
  qpos_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_spring_out: wp.array2d(dtype=float),
):
  worldid, jntid = wp.tid()
  stiffness = jnt_stiffness[worldid, jntid]
  dofid = jnt_dofadr[jntid]

  if stiffness == 0.0:
    return

  jnttype = jnt_type[jntid]
  qposid = jnt_qposadr[jntid]

  if jnttype == wp.static(JointType.FREE.value):
    dif = wp.vec3(
      qpos_in[worldid, qposid + 0] - qpos_spring[worldid, qposid + 0],
      qpos_in[worldid, qposid + 1] - qpos_spring[worldid, qposid + 1],
      qpos_in[worldid, qposid + 2] - qpos_spring[worldid, qposid + 2],
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
      qpos_spring[worldid, qposid + 3],
      qpos_spring[worldid, qposid + 4],
      qpos_spring[worldid, qposid + 5],
      qpos_spring[worldid, qposid + 6],
    )
    dif = math.quat_sub(rot, ref)
    qfrc_spring_out[worldid, dofid + 3] = -stiffness * dif[0]
    qfrc_spring_out[worldid, dofid + 4] = -stiffness * dif[1]
    qfrc_spring_out[worldid, dofid + 5] = -stiffness * dif[2]
  elif jnttype == wp.static(JointType.BALL.value):
    rot = wp.quat(
      qpos_in[worldid, qposid + 0],
      qpos_in[worldid, qposid + 1],
      qpos_in[worldid, qposid + 2],
      qpos_in[worldid, qposid + 3],
    )
    ref = wp.quat(
      qpos_spring[worldid, qposid + 0],
      qpos_spring[worldid, qposid + 1],
      qpos_spring[worldid, qposid + 2],
      qpos_spring[worldid, qposid + 3],
    )
    dif = math.quat_sub(rot, ref)
    qfrc_spring_out[worldid, dofid + 0] = -stiffness * dif[0]
    qfrc_spring_out[worldid, dofid + 1] = -stiffness * dif[1]
    qfrc_spring_out[worldid, dofid + 2] = -stiffness * dif[2]
  else:  # mjJNT_SLIDE, mjJNT_HINGE
    fdif = qpos_in[worldid, qposid] - qpos_spring[worldid, qposid]
    qfrc_spring_out[worldid, dofid] = -stiffness * fdif


@wp.kernel
def _gravity_force(
  # Model:
  opt_gravity: wp.vec3,
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_mass: wp.array2d(dtype=float),
  body_gravcomp: wp.array2d(dtype=float),
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
  gravcomp = body_gravcomp[worldid, bodyid]

  if gravcomp:
    force = -opt_gravity * body_mass[worldid, bodyid] * gravcomp

    pos = xipos_in[worldid, bodyid]
    jac, _ = support.jac(body_parentid, body_rootid, dof_bodyid, subtree_com_in, cdof_in, pos, bodyid, dofid, worldid)

    wp.atomic_add(qfrc_gravcomp_out[worldid], dofid, wp.dot(jac, force))


@wp.kernel
def _box_fluid(
  # Model:
  opt_wind: wp.vec3,
  opt_density: float,
  opt_viscosity: float,
  body_rootid: wp.array(dtype=int),
  body_mass: wp.array2d(dtype=float),
  body_inertia: wp.array2d(dtype=wp.vec3),
  # Data in:
  xipos_in: wp.array2d(dtype=wp.vec3),
  ximat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  # Data out:
  fluid_applied_out: wp.array2d(dtype=wp.spatial_vector),
):
  """Fluid forces based on inertia-box approximation."""

  worldid, bodyid = wp.tid()

  # map from CoM-centered to local body-centered 6D velocity

  # body-inertial
  pos = xipos_in[worldid, bodyid]
  rot = ximat_in[worldid, bodyid]
  rotT = wp.transpose(rot)

  # transform velocity
  cvel = cvel_in[worldid, bodyid]
  torque = wp.spatial_top(cvel)
  force = wp.spatial_bottom(cvel)
  subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]
  dif = pos - subtree_com
  force -= wp.cross(dif, torque)

  lvel_torque = rotT @ torque
  lvel_force = rotT @ force

  if opt_wind[0] or opt_wind[1] or opt_wind[2]:
    # subtract translational component from body velocity
    lvel_force -= rotT @ opt_wind

  lfrc_torque = wp.vec3(0.0)
  lfrc_force = wp.vec3(0.0)

  viscosity = opt_viscosity > 0.0
  density = opt_density > 0.0

  if viscosity or density:
    inertia = body_inertia[worldid, bodyid]
    mass = body_mass[worldid, bodyid]
    scl = 6.0 / mass
    box0 = wp.sqrt(wp.max(MJ_MINVAL, inertia[1] + inertia[2] - inertia[0]) * scl)
    box1 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[2] - inertia[1]) * scl)
    box2 = wp.sqrt(wp.max(MJ_MINVAL, inertia[0] + inertia[1] - inertia[2]) * scl)

  if viscosity:
    # diameter of sphere approximation
    diam = (box0 + box1 + box2) / 3.0

    # angular viscosity
    lfrc_torque = -lvel_torque * wp.pow(diam, 3.0) * wp.pi * opt_viscosity

    # linear viscosity
    lfrc_force = -3.0 * lvel_force * diam * wp.pi * opt_viscosity

  if density:
    # force
    lfrc_force -= wp.vec3(
      0.5 * opt_density * box1 * box2 * wp.abs(lvel_force[0]) * lvel_force[0],
      0.5 * opt_density * box0 * box2 * wp.abs(lvel_force[1]) * lvel_force[1],
      0.5 * opt_density * box0 * box1 * wp.abs(lvel_force[2]) * lvel_force[2],
    )

    # torque
    scl = opt_density / 64.0
    box0_pow4 = wp.pow(box0, 4.0)
    box1_pow4 = wp.pow(box1, 4.0)
    box2_pow4 = wp.pow(box2, 4.0)
    lfrc_torque -= wp.vec3(
      box0 * (box1_pow4 + box2_pow4) * wp.abs(lvel_torque[0]) * lvel_torque[0] * scl,
      box1 * (box0_pow4 + box2_pow4) * wp.abs(lvel_torque[1]) * lvel_torque[1] * scl,
      box2 * (box0_pow4 + box1_pow4) * wp.abs(lvel_torque[2]) * lvel_torque[2] * scl,
    )

  # rotate to global orientation: lfrc -> bfrc
  torque = rot @ lfrc_torque
  force = rot @ lfrc_force

  fluid_applied_out[worldid, bodyid] = wp.spatial_vector(force, torque)


def _fluid(m: Model, d: Data):
  wp.launch(
    _box_fluid,
    dim=(d.nworld, m.nbody),
    inputs=[
      m.opt.wind,
      m.opt.density,
      m.opt.viscosity,
      m.body_rootid,
      m.body_mass,
      m.body_inertia,
      d.xipos,
      d.ximat,
      d.subtree_com,
      d.cvel,
    ],
    outputs=[
      d.fluid_applied,
    ],
  )

  # TODO(team): ellipsoid fluid model

  support.apply_ft(m, d, d.fluid_applied, d.qfrc_fluid)


@wp.kernel
def _qfrc_passive(
  # Model:
  jnt_actgravcomp: wp.array(dtype=int),
  dof_jntid: wp.array(dtype=int),
  dof_damping: wp.array2d(dtype=float),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qfrc_spring_in: wp.array2d(dtype=float),
  qfrc_gravcomp_in: wp.array2d(dtype=float),
  qfrc_fluid_in: wp.array2d(dtype=float),
  # In:
  gravcomp: bool,
  fluid: bool,
  # Data out:
  qfrc_damper_out: wp.array2d(dtype=float),
  qfrc_passive_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()

  # spring
  qfrc_passive = qfrc_spring_in[worldid, dofid]

  # damper
  qfrc_damper = -dof_damping[worldid, dofid] * qvel_in[worldid, dofid]
  qfrc_damper_out[worldid, dofid] = qfrc_damper

  qfrc_passive += qfrc_damper

  # add gravcomp unless added by actuators
  if gravcomp and not jnt_actgravcomp[dof_jntid[dofid]]:
    qfrc_passive += qfrc_gravcomp_in[worldid, dofid]

  # add fluid force
  if fluid:
    qfrc_passive += qfrc_fluid_in[worldid, dofid]

  qfrc_passive_out[worldid, dofid] = qfrc_passive


@event_scope
def passive(m: Model, d: Data):
  """Adds all passive forces."""
  if m.opt.disableflags & DisableBit.PASSIVE:
    d.qfrc_spring.zero_()
    d.qfrc_damper.zero_()
    d.qfrc_gravcomp.zero_()
    d.qfrc_fluid.zero_()
    d.qfrc_passive.zero_()
    return

  d.qfrc_spring.zero_()
  wp.launch(
    _spring_passive,
    dim=(d.nworld, m.njnt),
    inputs=[m.qpos_spring, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_stiffness, d.qpos],
    outputs=[d.qfrc_spring],
  )

  gravcomp = m.ngravcomp and not (m.opt.disableflags & DisableBit.GRAVITY)
  d.qfrc_gravcomp.zero_()

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

  fluid = m.opt.density or m.opt.viscosity or m.opt.wind[0] or m.opt.wind[1] or m.opt.wind[2]
  if fluid:
    _fluid(m, d)

  wp.launch(
    _qfrc_passive,
    dim=(d.nworld, m.nv),
    inputs=[
      m.jnt_actgravcomp,
      m.dof_jntid,
      m.dof_damping,
      d.qvel,
      d.qfrc_spring,
      d.qfrc_gravcomp,
      d.qfrc_fluid,
      gravcomp,
      fluid,
    ],
    outputs=[d.qfrc_damper, d.qfrc_passive],
  )
