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
def _spring_passive(
  # Model:
  qpos_spring: wp.array(dtype=float),
  jnt_type: wp.array(dtype=int),
  jnt_qposadr: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  jnt_stiffness: wp.array(dtype=float),
  # Data in:
  qpos_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_spring_out: wp.array2d(dtype=float),
):
  worldid, jntid = wp.tid()
  stiffness = jnt_stiffness[jntid]
  dofid = jnt_dofadr[jntid]

  if stiffness == 0.0:
    return

  jnttype = jnt_type[jntid]
  qposid = jnt_qposadr[jntid]

  if jnttype == wp.static(JointType.FREE.value):
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
  elif jnttype == wp.static(JointType.BALL.value):
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
  else:  # mjJNT_SLIDE, mjJNT_HINGE
    fdif = qpos_in[worldid, qposid] - qpos_spring[qposid]
    qfrc_spring_out[worldid, dofid] = -stiffness * fdif


@wp.kernel
def _damper_passive(
  # Model:
  dof_damping: wp.array(dtype=float),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  qfrc_spring_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_damper_out: wp.array2d(dtype=float),
  qfrc_passive_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()

  qfrc_damper = -dof_damping[dofid] * qvel_in[worldid, dofid]

  qfrc_damper_out[worldid, dofid] = qfrc_damper
  qfrc_passive_out[worldid, dofid] = qfrc_damper + qfrc_spring_in[worldid, dofid]


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
def _qfrc_passive_gravcomp(
  # Model:
  jnt_actgravcomp: wp.array(dtype=int),
  dof_jntid: wp.array(dtype=int),
  # Data in:
  qfrc_gravcomp_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_passive_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()

  # add gravcomp unless added by actuators
  if jnt_actgravcomp[dof_jntid[dofid]]:
    return

  qfrc_passive_out[worldid, dofid] += qfrc_gravcomp_in[worldid, dofid]


@wp.kernel
def _flex_elasticity(
  # Model:
  body_dofadr: wp.array(dtype=int),
  flex_dim: wp.array(dtype=int),
  flex_vertadr: wp.array(dtype=int),
  flex_edgeadr: wp.array(dtype=int),
  flex_elemedgeadr: wp.array(dtype=int),
  flex_vertbodyid: wp.array(dtype=int),
  flex_elem: wp.array(dtype=int),
  flex_elemedge: wp.array(dtype=int),
  flexedge_length0: wp.array(dtype=float),
  flex_stiffness: wp.array(dtype=float),
  flex_damping: wp.array(dtype=float),
  # Data in:
  flexvert_xpos_in: wp.array2d(dtype=wp.vec3),
  flexedge_length_in: wp.array2d(dtype=float),
  flexedge_velocity_in: wp.array2d(dtype=float),
  # In:
  timestep: float,
  # Data out:
  qfrc_spring_out: wp.array2d(dtype=float),
):
  worldid, t = wp.tid()
  f = 0  # TODO(quaglino): this should become a function of t

  # TODO(quaglino): support dim != 2
  dim = flex_dim[f]
  nedge = 3
  nvert = 3
  kD = flex_damping[f] / timestep

  edges = wp.mat(1, 2, 2, 0, 0, 1, shape=(3, 2), dtype=int)
  gradient = wp.mat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, shape=(3, 6))
  for e in range(nedge):
    vert0 = flex_elem[(dim + 1) * t + edges[e, 0]]
    vert1 = flex_elem[(dim + 1) * t + edges[e, 1]]
    xpos0 = flexvert_xpos_in[worldid, vert0]
    xpos1 = flexvert_xpos_in[worldid, vert1]
    for i in range(3):
      gradient[e, 0 + i] = xpos0[i] - xpos1[i]
      gradient[e, 3 + i] = xpos1[i] - xpos0[i]

  elongation = wp.vec3(0.0, 0.0, 0.0)
  for e in range(nedge):
    idx = flex_elemedge[flex_elemedgeadr[f] + t * nedge + e]
    vel = flexedge_velocity_in[worldid, flex_edgeadr[f] + idx]
    deformed = flexedge_length_in[worldid, flex_edgeadr[f] + idx]
    reference = flexedge_length0[flex_edgeadr[f] + idx]
    previous = deformed - vel * timestep
    elongation[e] = deformed * deformed - reference * reference + (deformed * deformed - previous * previous) * kD

  metric = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  id = 0
  for ed1 in range(nedge):
    for ed2 in range(ed1, nedge):
      metric[ed1, ed2] = flex_stiffness[21 * t + id]
      metric[ed2, ed1] = flex_stiffness[21 * t + id]
      id += 1

  force = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  for ed1 in range(nedge):
    for ed2 in range(nedge):
      for i in range(2):
        for x in range(3):
          force[edges[ed2, i], x] -= elongation[ed1] * gradient[ed2, 3 * i + x] * metric[ed1, ed2]

  for v in range(nvert):
    vert = flex_elem[(dim + 1) * t + v]
    bodyid = flex_vertbodyid[flex_vertadr[f] + vert]
    for x in range(3):
      qfrc_spring_out[worldid, body_dofadr[bodyid] + x] += force[v, x]


@wp.kernel
def _combine_qfrc(
  # Data in:
  qfrc_spring_in: wp.array2d(dtype=float),
  qfrc_damper_in: wp.array2d(dtype=float),
  # Data out:
  qfrc_passive_out: wp.array2d(dtype=float),
):
  worldid, dofid = wp.tid()
  qfrc_passive_out[worldid, dofid] = qfrc_spring_in[worldid, dofid] + qfrc_damper_in[worldid, dofid]


@event_scope
def passive(m: Model, d: Data):
  """Adds all passive forces."""
  if m.opt.disableflags & DisableBit.PASSIVE:
    d.qfrc_passive.zero_()
    d.qfrc_gravcomp.zero_()
    return

  # TODO(team): mj_ellipsoidFluidModel
  # TODO(team): mj_inertiaBoxFluidModell

  d.qfrc_spring.zero_()
  wp.launch(
    _spring_passive,
    dim=(d.nworld, m.njnt),
    inputs=[m.qpos_spring, m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, m.jnt_stiffness, d.qpos],
    outputs=[d.qfrc_spring],
  )
  wp.launch(
    _damper_passive,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_damping, d.qvel, d.qfrc_spring],
    outputs=[d.qfrc_damper, d.qfrc_passive],
  )
  wp.launch(
    _flex_elasticity,
    dim=(d.nworld, m.nflexelem),
    inputs=[
      m.body_dofadr,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_edgeadr,
      m.flex_elemedgeadr,
      m.flex_vertbodyid,
      m.flex_elem,
      m.flex_elemedge,
      m.flexedge_length0,
      m.flex_stiffness,
      m.flex_damping,
      d.flexvert_xpos,
      d.flexedge_length,
      d.flexedge_velocity,
      m.opt.timestep,
    ],
    outputs=[d.qfrc_spring],
  )
  wp.launch(
    _combine_qfrc,
    dim=(d.nworld, m.nv),
    inputs=[d.qfrc_spring, d.qfrc_damper],
    outputs=[d.qfrc_passive],
  )
  if m.ngravcomp and not (m.opt.disableflags & DisableBit.GRAVITY):
    d.qfrc_gravcomp.zero_()
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
      _qfrc_passive_gravcomp,
      dim=(d.nworld, m.nv),
      inputs=[m.jnt_actgravcomp, m.dof_jntid, d.qfrc_gravcomp],
      outputs=[d.qfrc_passive],
    )
