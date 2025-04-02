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

from . import types
from .warp_util import event_scope


@wp.func
def _update_efc_row(
  m: types.Model,
  d: types.Data,
  efcid: wp.int32,
  pos_aref: wp.float32,
  pos_imp: wp.float32,
  invweight: wp.float32,
  solref: wp.vec2f,
  solimp: types.vec5,
  margin: wp.float32,
  refsafe: bool,
  Jqvel: float,
):
  # Calculate kbi
  timeconst = solref[0]
  dampratio = solref[1]
  dmin = solimp[0]
  dmax = solimp[1]
  width = solimp[2]
  mid = solimp[3]
  power = solimp[4]

  if refsafe:
    timeconst = wp.max(timeconst, 2.0 * m.opt.timestep)

  dmin = wp.clamp(dmin, types.MJ_MINIMP, types.MJ_MAXIMP)
  dmax = wp.clamp(dmax, types.MJ_MINIMP, types.MJ_MAXIMP)
  width = wp.max(types.MJ_MINVAL, width)
  mid = wp.clamp(mid, types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, power)

  # See https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
  k = 1.0 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)
  b = 2.0 / (dmax * timeconst)
  k = wp.where(solref[0] <= 0, -solref[0] / (dmax * dmax), k)
  b = wp.where(solref[1] <= 0, -solref[1] / dmax, b)

  imp_x = wp.abs(pos_imp) / width
  imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
  imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)
  imp_y = wp.where(imp_x < mid, imp_a, imp_b)
  imp = dmin + imp_y * (dmax - dmin)
  imp = wp.clamp(imp, dmin, dmax)
  imp = wp.where(imp_x > 1.0, dmax, imp)

  # Update constraints
  d.efc.D[efcid] = 1.0 / wp.max(invweight * (1.0 - imp) / imp, types.MJ_MINVAL)
  d.efc.aref[efcid] = -k * imp * pos_aref - b * Jqvel
  d.efc.pos[efcid] = pos_aref + margin
  d.efc.margin[efcid] = margin


@wp.func
def _jac(
  m: types.Model,
  d: types.Data,
  point: wp.vec3,
  xyz: wp.int32,
  bodyid: wp.int32,
  dofid: wp.int32,
  worldid: wp.int32,
):
  dof_bodyid = m.dof_bodyid[dofid]
  in_tree = int(dof_bodyid == 0)
  parentid = bodyid
  while parentid != 0:
    if parentid == dof_bodyid:
      in_tree = 1
      break
    parentid = m.body_parentid[parentid]

  offset = point - wp.vec3(d.subtree_com[worldid, m.body_rootid[bodyid]])

  cdof = d.cdof[worldid, dofid]
  jac = wp.spatial_bottom(cdof) + wp.cross(wp.spatial_top(cdof), offset)

  return jac[xyz] * float(in_tree)


@wp.kernel
def _efc_limit_slide_hinge(
  m: types.Model,
  d: types.Data,
  refsafe: bool,
):
  worldid, jntlimitedid = wp.tid()
  jntid = m.jnt_limited_slide_hinge_adr[jntlimitedid]

  qpos = d.qpos[worldid, m.jnt_qposadr[jntid]]
  jnt_range = m.jnt_range[jntid]
  dist_min, dist_max = qpos - jnt_range[0], jnt_range[1] - qpos
  pos = wp.min(dist_min, dist_max) - m.jnt_margin[jntid]
  active = pos < 0

  if active:
    efcid = wp.atomic_add(d.nefc, 0, 1)
    d.efc.worldid[efcid] = worldid

    dofadr = m.jnt_dofadr[jntid]

    J = float(dist_min < dist_max) * 2.0 - 1.0
    d.efc.J[efcid, dofadr] = J
    Jqvel = J * d.qvel[worldid, dofadr]

    _update_efc_row(
      m,
      d,
      efcid,
      pos,
      pos,
      m.dof_invweight0[dofadr],
      m.jnt_solref[jntid],
      m.jnt_solimp[jntid],
      m.jnt_margin[jntid],
      refsafe,
      Jqvel,
    )


@wp.kernel
def _efc_contact_pyramidal(
  m: types.Model,
  d: types.Data,
  refsafe: bool,
):
  conid, dimid = wp.tid()

  if conid >= d.ncon[0]:
    return

  if d.contact.dim[conid] != 3:
    return

  includemargin = d.contact.includemargin[conid]
  pos = d.contact.dist[conid] - includemargin
  active = pos < 0

  if active:
    efcid = wp.atomic_add(d.nefc, 0, 1)
    worldid = d.contact.worldid[conid]
    d.efc.worldid[efcid] = worldid

    geom = d.contact.geom[conid]
    body1 = m.geom_bodyid[geom[0]]
    body2 = m.geom_bodyid[geom[1]]

    con_pos = d.contact.pos[conid]
    frame = d.contact.frame[conid]
    friction = d.contact.friction[conid]
    fri0 = friction[0]

    # pyramidal has common invweight across all edges
    invweight = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]
    invweight = invweight + fri0 * fri0 * invweight
    invweight = invweight * 2.0 * fri0 * fri0 / m.opt.impratio

    dimid2 = dimid / 2 + 1

    Jqvel = float(0.0)
    for i in range(m.nv):
      diff_0 = float(0.0)
      diff_i = float(0.0)
      for xyz in range(3):
        jac1p = _jac(m, d, con_pos, xyz, body1, i, worldid)
        jac2p = _jac(m, d, con_pos, xyz, body2, i, worldid)
        jac_dif = jac2p - jac1p
        diff_0 += frame[0, xyz] * jac_dif
        diff_i += frame[dimid2, xyz] * jac_dif

      if dimid % 2 == 0:
        J = diff_0 + diff_i * friction[dimid2 - 1]
      else:
        J = diff_0 - diff_i * friction[dimid2 - 1]

      d.efc.J[efcid, i] = J
      Jqvel += J * d.qvel[worldid, i]

    _update_efc_row(
      m,
      d,
      efcid,
      pos,
      pos,
      invweight,
      d.contact.solref[conid],
      d.contact.solimp[conid],
      includemargin,
      refsafe,
      Jqvel,
    )


@event_scope
def make_constraint(m: types.Model, d: types.Data):
  """Creates constraint jacobians and other supporting data."""

  d.nefc.zero_()

  if not (m.opt.disableflags & types.DisableBit.CONSTRAINT.value):
    d.efc.J.zero_()

    refsafe = not m.opt.disableflags & types.DisableBit.REFSAFE.value

    if not (m.opt.disableflags & types.DisableBit.LIMIT.value) and (
      m.jnt_limited_slide_hinge_adr.size != 0
    ):
      wp.launch(
        _efc_limit_slide_hinge,
        dim=(m.nworld, m.jnt_limited_slide_hinge_adr.size),
        inputs=[m, d, refsafe],
      )

    if (
      not (m.opt.disableflags & types.DisableBit.CONTACT.value)
      and m.opt.cone == types.ConeType.PYRAMIDAL.value
    ):
      wp.launch(
        _efc_contact_pyramidal,
        dim=(d.nconmax, 4),
        inputs=[m, d, refsafe],
      )
