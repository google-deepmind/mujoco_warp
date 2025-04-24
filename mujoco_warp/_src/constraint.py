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
from . import types
from .warp_util import event_scope

wp.config.enable_backward = False


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
  Jqvel: float,
  frictionloss: float,
  id: int,
):
  # Calculate kbi
  timeconst = solref[0]
  dampratio = solref[1]
  dmin = solimp[0]
  dmax = solimp[1]
  width = solimp[2]
  mid = solimp[3]
  power = solimp[4]

  # TODO(team): wp.static?
  if not (m.opt.disableflags & types.DisableBit.REFSAFE.value):
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
  d.efc.frictionloss[efcid] = frictionloss
  d.efc.id[efcid] = id


@wp.kernel
def _efc_equality_connect(
  m: types.Model,
  d: types.Data,
):
  """Calculates constraint rows for connect equality constraints."""

  worldid, i_eq_connect_adr = wp.tid()
  i_eq = m.eq_connect_adr[i_eq_connect_adr]
  if not d.eq_active[worldid, i_eq]:
    return

  necid = wp.atomic_add(d.ne_connect, 0, 3)
  efcid = d.nefc[0] + necid

  for i in range(wp.static(3)):
    d.efc.worldid[efcid + i] = worldid

  data = m.eq_data[i_eq]
  anchor1 = wp.vec3f(data[0], data[1], data[2])
  anchor2 = wp.vec3f(data[3], data[4], data[5])

  obj1id = m.eq_obj1id[i_eq]
  obj2id = m.eq_obj2id[i_eq]

  if m.nsite and m.eq_objtype[i_eq] == wp.static(types.ObjType.SITE.value):
    # body1id stores the index of site_bodyid.
    body1id = m.site_bodyid[obj1id]
    body2id = m.site_bodyid[obj2id]
    pos1 = d.site_xpos[worldid, obj1id]
    pos2 = d.site_xpos[worldid, obj2id]
  else:
    body1id = obj1id
    body2id = obj2id
    pos1 = d.xpos[worldid, body1id] + d.xmat[worldid, body1id] @ anchor1
    pos2 = d.xpos[worldid, body2id] + d.xmat[worldid, body2id] @ anchor2

  # error is difference in global positions
  pos = pos1 - pos2

  # compute Jacobian difference (opposite of contact: 0 - 1)
  Jqvel = wp.vec3f(0.0, 0.0, 0.0)
  for dofid in range(m.nv):  # TODO: parallelize
    jacp1, _ = support.jac(m, d, pos1, body1id, dofid, worldid)
    jacp2, _ = support.jac(m, d, pos2, body2id, dofid, worldid)
    j1mj2 = jacp1 - jacp2
    d.efc.J[efcid + 0, dofid] = j1mj2[0]
    d.efc.J[efcid + 1, dofid] = j1mj2[1]
    d.efc.J[efcid + 2, dofid] = j1mj2[2]
    Jqvel += j1mj2 * d.qvel[worldid, dofid]

  invweight = m.body_invweight0[body1id, 0] + m.body_invweight0[body2id, 0]
  pos_imp = wp.length(pos)

  solref = m.eq_solref[i_eq]
  solimp = m.eq_solimp[i_eq]

  for i in range(3):
    efcidi = efcid + i
    d.efc.worldid[efcidi] = worldid

    _update_efc_row(
      m,
      d,
      efcidi,
      pos[i],
      pos_imp,
      invweight,
      solref,
      solimp,
      wp.float32(0.0),
      Jqvel[i],
      0.0,
      i_eq,
    )


@wp.kernel
def _efc_equality_joint(
  m: types.Model,
  d: types.Data,
):
  worldid, i_eq_joint_adr = wp.tid()
  i_eq = m.eq_jnt_adr[i_eq_joint_adr]
  if not d.eq_active[worldid, i_eq]:
    return

  nejid = wp.atomic_add(d.ne_jnt, 0, 1)
  efcid = d.nefc[0] + d.ne_connect[0] + d.ne_weld[0] + nejid
  d.efc.worldid[efcid] = worldid

  jntid_1 = m.eq_obj1id[i_eq]
  jntid_2 = m.eq_obj2id[i_eq]
  data = m.eq_data[i_eq]
  dofadr1 = m.jnt_dofadr[jntid_1]
  qposadr1 = m.jnt_qposadr[jntid_1]
  d.efc.J[efcid, dofadr1] = 1.0

  if jntid_2 > -1:
    # Two joint constraint
    qposadr2 = m.jnt_qposadr[jntid_2]
    dofadr2 = m.jnt_dofadr[jntid_2]
    dif = d.qpos[worldid, qposadr2] - m.qpos0[qposadr2]

    # Horner's method for polynomials
    rhs = data[0] + dif * (data[1] + dif * (data[2] + dif * (data[3] + dif * data[4])))
    deriv_2 = data[1] + dif * (
      2.0 * data[2] + dif * (3.0 * data[3] + dif * 4.0 * data[4])
    )

    pos = d.qpos[worldid, qposadr1] - m.qpos0[qposadr1] - rhs
    Jqvel = d.qvel[worldid, dofadr1] - d.qvel[worldid, dofadr2] * deriv_2
    invweight = m.dof_invweight0[dofadr1] + m.dof_invweight0[dofadr2]

    d.efc.J[efcid, dofadr2] = -deriv_2
  else:
    # Single joint constraint
    pos = d.qpos[worldid, qposadr1] - m.qpos0[qposadr1] - data[0]
    Jqvel = d.qvel[worldid, dofadr1]
    invweight = m.dof_invweight0[dofadr1]

  # Update constraint parameters
  _update_efc_row(
    m,
    d,
    efcid,
    pos,
    pos,
    invweight,
    m.eq_solref[i_eq],
    m.eq_solimp[i_eq],
    wp.float32(0.0),
    Jqvel,
    0.0,
    i_eq,
  )


@wp.kernel
def _efc_friction(
  m: types.Model,
  d: types.Data,
):
  # TODO(team): tendon
  worldid, dofid = wp.tid()

  if m.dof_frictionloss[dofid] <= 0.0:
    return

  efcid = wp.atomic_add(d.nefc, 0, 1)
  wp.atomic_add(d.nf, 0, 1)
  d.efc.worldid[efcid] = worldid

  d.efc.J[efcid, dofid] = 1.0
  Jqvel = d.qvel[worldid, dofid]

  _update_efc_row(
    m,
    d,
    efcid,
    0.0,
    0.0,
    m.dof_invweight0[dofid],
    m.dof_solref[dofid],
    m.dof_solimp[dofid],
    0.0,
    Jqvel,
    m.dof_frictionloss[dofid],
    dofid,
  )


@wp.kernel
def _efc_equality_weld(
  m: types.Model,
  d: types.Data,
):
  worldid, i_eq_weld_adr = wp.tid()
  i_eq = m.eq_wld_adr[i_eq_weld_adr]
  if not d.eq_active[worldid, i_eq]:
    return

  newid = wp.atomic_add(d.ne_weld, 0, 6)
  efcid = d.nefc[0] + d.ne_connect[0] + newid
  for i in range(wp.static(6)):
    d.efc.worldid[efcid + i] = worldid

  is_site = m.eq_objtype[i_eq] == wp.static(types.ObjType.SITE.value) and m.nsite > 0

  obj1id = m.eq_obj1id[i_eq]
  obj2id = m.eq_obj2id[i_eq]

  data = m.eq_data[i_eq]
  anchor1 = wp.vec3(data[0], data[1], data[2])
  anchor2 = wp.vec3(data[3], data[4], data[5])
  relpose = wp.quat(data[6], data[7], data[8], data[9])
  torquescale = data[10]

  if is_site:
    # body1id stores the index of site_bodyid.
    body1id = m.site_bodyid[obj1id]
    body2id = m.site_bodyid[obj2id]
    pos1 = d.site_xpos[worldid, obj1id]
    pos2 = d.site_xpos[worldid, obj2id]

    quat = math.mul_quat(d.xquat[worldid, body1id], m.site_quat[obj1id])
    quat1 = math.quat_inv(math.mul_quat(d.xquat[worldid, body2id], m.site_quat[obj2id]))

  else:
    body1id = obj1id
    body2id = obj2id
    pos1 = d.xpos[worldid, body1id] + d.xmat[worldid, body1id] @ anchor2
    pos2 = d.xpos[worldid, body2id] + d.xmat[worldid, body2id] @ anchor1

    quat = math.mul_quat(d.xquat[worldid, body1id], relpose)
    quat1 = math.quat_inv(d.xquat[worldid, body2id])

  # compute Jacobian difference (opposite of contact: 0 - 1)
  Jqvelp = wp.vec3f(0.0, 0.0, 0.0)
  Jqvelr = wp.vec3f(0.0, 0.0, 0.0)

  for dofid in range(m.nv):  # TODO: parallelize
    jacp1, jacr1 = support.jac(m, d, pos1, body1id, dofid, worldid)
    jacp2, jacr2 = support.jac(m, d, pos2, body2id, dofid, worldid)

    jacdifp = jacp1 - jacp2
    for i in range(wp.static(3)):
      d.efc.J[efcid + i, dofid] = jacdifp[i]

    jacdifr = (jacr1 - jacr2) * torquescale
    jacdifrq = math.mul_quat(math.quat_mul_axis(quat1, jacdifr), quat)
    jacdifr = 0.5 * wp.vec3(jacdifrq[1], jacdifrq[2], jacdifrq[3])

    for i in range(wp.static(3)):
      d.efc.J[efcid + 3 + i, dofid] = jacdifr[i]

    Jqvelp += jacdifp * d.qvel[worldid, dofid]
    Jqvelr += jacdifr * d.qvel[worldid, dofid]

  # error is difference in global position and orientation
  cpos = pos1 - pos2

  crotq = math.mul_quat(quat1, quat)  # copy axis components
  crot = wp.vec3(crotq[1], crotq[2], crotq[3]) * torquescale

  invweight_t = m.body_invweight0[body1id, 0] + m.body_invweight0[body2id, 0]

  pos_imp = wp.sqrt(wp.length_sq(cpos) + wp.length_sq(crot))

  solref = m.eq_solref[i_eq]
  solimp = m.eq_solimp[i_eq]

  for i in range(3):
    _update_efc_row(
      m,
      d,
      efcid + i,
      cpos[i],
      pos_imp,
      invweight_t,
      solref,
      solimp,
      0.0,
      Jqvelp[i],
      0.0,
      i_eq,
    )

  invweight_r = m.body_invweight0[body1id, 1] + m.body_invweight0[body2id, 1]

  for i in range(3):
    _update_efc_row(
      m,
      d,
      efcid + 3 + i,
      crot[i],
      pos_imp,
      invweight_r,
      solref,
      solimp,
      0.0,
      Jqvelr[i],
      0.0,
      i_eq,
    )


@wp.kernel
def _efc_limit_slide_hinge(
  m: types.Model,
  d: types.Data,
):
  worldid, jntlimitedid = wp.tid()
  jntid = m.jnt_limited_slide_hinge_adr[jntlimitedid]

  qpos = d.qpos[worldid, m.jnt_qposadr[jntid]]
  jnt_range = m.jnt_range[jntid]
  dist_min, dist_max = qpos - jnt_range[0], jnt_range[1] - qpos
  pos = wp.min(dist_min, dist_max) - m.jnt_margin[jntid]
  active = pos < 0

  if active:
    lid = wp.atomic_add(d.nl, 0, 1)
    efcid = lid + d.nefc[0]
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
      Jqvel,
      0.0,
      dofadr,
    )


@wp.kernel
def _efc_limit_ball(
  m: types.Model,
  d: types.Data,
):
  worldid, jntlimitedid = wp.tid()
  jntid = m.jnt_limited_ball_adr[jntlimitedid]
  qposadr = m.jnt_qposadr[jntid]

  qpos = d.qpos[worldid]
  jnt_quat = wp.quat(
    qpos[qposadr + 0], qpos[qposadr + 1], qpos[qposadr + 2], qpos[qposadr + 3]
  )
  axis_angle = math.quat_to_vel(jnt_quat)
  axis, angle = math.normalize_with_norm(axis_angle)
  jnt_margin = m.jnt_margin[jntid]
  jnt_range = m.jnt_range[jntid]

  pos = wp.max(jnt_range[0], jnt_range[1]) - angle - jnt_margin
  active = pos < 0

  if active:
    lid = wp.atomic_add(d.nl, 0, 1)
    efcid = lid + d.nefc[0]
    d.efc.worldid[efcid] = worldid

    dofadr = m.jnt_dofadr[jntid]

    d.efc.J[efcid, dofadr + 0] = -axis[0]
    d.efc.J[efcid, dofadr + 1] = -axis[1]
    d.efc.J[efcid, dofadr + 2] = -axis[2]

    Jqvel = -axis[0] * d.qvel[worldid, dofadr + 0]
    Jqvel -= axis[1] * d.qvel[worldid, dofadr + 1]
    Jqvel -= axis[2] * d.qvel[worldid, dofadr + 2]

    _update_efc_row(
      m,
      d,
      efcid,
      pos,
      pos,
      m.dof_invweight0[dofadr],
      m.jnt_solref[jntid],
      m.jnt_solimp[jntid],
      jnt_margin,
      Jqvel,
      0.0,
      jntid,
    )


@wp.kernel
def _efc_limit_tendon(
  m: types.Model,
  d: types.Data,
):
  worldid, tenlimitedid = wp.tid()
  tenid = m.tendon_limited_adr[tenlimitedid]

  ten_range = m.tendon_range[tenid]
  length = d.ten_length[worldid, tenid]
  dist_min, dist_max = length - ten_range[0], ten_range[1] - length
  ten_margin = m.tendon_margin[tenid]
  pos = wp.min(dist_min, dist_max) - ten_margin
  active = pos < 0

  if active:
    lid = wp.atomic_add(d.nl, 0, 1)
    efcid = d.nefc[0] + lid
    d.efc.worldid[efcid] = worldid

    Jqvel = float(0.0)
    scl = float(dist_min < dist_max) * 2.0 - 1.0

    adr = m.tendon_adr[tenid]
    if m.wrap_type[adr] == wp.static(types.WrapType.JOINT.value):
      ten_num = m.tendon_num[tenid]
      for i in range(ten_num):
        dofadr = m.jnt_dofadr[m.wrap_objid[adr + i]]
        J = scl * d.ten_J[worldid, tenid, dofadr]
        d.efc.J[efcid, dofadr] = J
        Jqvel += J * d.qvel[worldid, dofadr]
    else:
      for i in range(m.nv):
        J = scl * d.ten_J[worldid, tenid, i]
        d.efc.J[efcid, i] = J
        Jqvel += J * d.qvel[worldid, i]

    _update_efc_row(
      m,
      d,
      efcid,
      pos,
      pos,
      m.tendon_invweight0[tenid],
      m.tendon_solref_lim[tenid],
      m.tendon_solimp_lim[tenid],
      ten_margin,
      Jqvel,
      0.0,
      tenid,
    )


@wp.kernel
def _efc_contact_pyramidal(
  m: types.Model,
  d: types.Data,
):
  conid, dimid = wp.tid()

  if conid >= d.ncon[0]:
    return

  condim = d.contact.dim[conid]

  if condim == 1 and dimid > 0:
    return
  elif condim > 1 and dimid >= 2 * (condim - 1):
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

    # pyramidal has common invweight across all edges
    invweight = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]

    if condim > 1:
      dimid2 = dimid / 2 + 1

      friction = d.contact.friction[conid]
      fri0 = friction[0]
      frii = friction[dimid2 - 1]
      invweight = invweight + fri0 * fri0 * invweight
      invweight = invweight * 2.0 * fri0 * fri0 / m.opt.impratio

    Jqvel = float(0.0)
    for i in range(m.nv):
      J = float(0.0)
      Ji = float(0.0)
      jac1p, jac1r = support.jac(m, d, con_pos, body1, i, worldid)
      jac2p, jac2r = support.jac(m, d, con_pos, body2, i, worldid)
      jacp_dif = jac2p - jac1p
      for xyz in range(3):
        J += frame[0, xyz] * jacp_dif[xyz]

        if condim > 1:
          if dimid2 < 3:
            Ji += frame[dimid2, xyz] * jacp_dif[xyz]
          else:
            Ji += frame[dimid2 - 3, xyz] * (jac2r[xyz] - jac1r[xyz])

      if condim > 1:
        if dimid % 2 == 0:
          J += Ji * frii
        else:
          J -= Ji * frii

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
      Jqvel,
      0.0,
      conid,
    )


@wp.kernel
def _efc_contact_elliptic(
  m: types.Model,
  d: types.Data,
):
  conid, dimid = wp.tid()

  if conid >= d.ncon[0]:
    return

  condim = d.contact.dim[conid]

  if dimid > condim - 1:
    return

  includemargin = d.contact.includemargin[conid]
  pos = d.contact.dist[conid] - includemargin
  active = pos < 0.0

  if active:
    efcid = wp.atomic_add(d.nefc, 0, 1)
    worldid = d.contact.worldid[conid]
    d.efc.worldid[efcid] = worldid
    d.contact.efc_address[conid, dimid] = efcid

    geom = d.contact.geom[conid]
    body1 = m.geom_bodyid[geom[0]]
    body2 = m.geom_bodyid[geom[1]]

    cpos = d.contact.pos[conid]
    frame = d.contact.frame[conid]

    # TODO(team): parallelize J and Jqvel computation?
    Jqvel = float(0.0)
    for i in range(m.nv):
      J = float(0.0)
      jac1p, jac1r = support.jac(m, d, cpos, body1, i, worldid)
      jac2p, jac2r = support.jac(m, d, cpos, body2, i, worldid)
      for xyz in range(3):
        if dimid < 3:
          jac_dif = jac2p[xyz] - jac1p[xyz]
          J += frame[dimid, xyz] * jac_dif
        else:
          jac_dif = jac2r[xyz] - jac1r[xyz]
          J += frame[dimid - 3, xyz] * jac_dif

      d.efc.J[efcid, i] = J
      Jqvel += J * d.qvel[worldid, i]

    invweight = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]

    ref = d.contact.solref[conid]
    pos_aref = pos

    if dimid > 0:
      solreffriction = d.contact.solreffriction[conid]

      # non-normal directions use solreffriction (if non-zero)
      if solreffriction[0] or solreffriction[1]:
        ref = solreffriction

      # TODO(team): precompute 1 / impratio
      invweight = invweight / m.opt.impratio
      friction = d.contact.friction[conid]

      if dimid > 1:
        fri0 = friction[0]
        frii = friction[dimid - 1]
        fri = fri0 * fri0 / (frii * frii)
        invweight *= fri

      pos_aref = 0.0

    _update_efc_row(
      m,
      d,
      efcid,
      pos_aref,
      pos,
      invweight,
      ref,
      d.contact.solimp[conid],
      includemargin,
      Jqvel,
      0.0,
      conid,
    )


@wp.kernel
def _num_equality(d: types.Data):
  ne = d.ne_connect[0] + d.ne_weld[0] + d.ne_jnt[0]
  d.ne[0] = ne
  d.nefc[0] += ne


@wp.kernel
def _update_nefc(d: types.Data):
  d.nefc[0] += d.nl[0]


@event_scope
def make_constraint(m: types.Model, d: types.Data):
  """Creates constraint jacobians and other supporting data."""

  with wp.ScopedDevice(m.qpos0.device):
    d.ne.zero_()
    d.ne_connect.zero_()
    d.ne_weld.zero_()
    d.ne_jnt.zero_()
    d.nefc.zero_()
    d.nf.zero_()
    d.nl.zero_()

    if not (m.opt.disableflags & types.DisableBit.CONSTRAINT.value):
      d.efc.J.zero_()

      if not (m.opt.disableflags & types.DisableBit.EQUALITY.value):
        wp.launch(
          _efc_equality_connect,
          dim=(d.nworld, m.eq_connect_adr.size),
          inputs=[m, d],
        )
        wp.launch(
          _efc_equality_weld,
          dim=(d.nworld, m.eq_wld_adr.size),
          inputs=[m, d],
        )
        wp.launch(
          _efc_equality_joint,
          dim=(d.nworld, m.eq_jnt_adr.size),
          inputs=[m, d],
        )

      wp.launch(_num_equality, dim=(1,), inputs=[d])

      if not (m.opt.disableflags & types.DisableBit.FRICTIONLOSS.value):
        wp.launch(
          _efc_friction,
          dim=(d.nworld, m.nv),
          inputs=[m, d],
        )

      # limit
      if not (m.opt.disableflags & types.DisableBit.LIMIT.value):
        limit_ball = m.jnt_limited_ball_adr.size > 0
        if limit_ball:
          wp.launch(
            _efc_limit_ball,
            dim=(d.nworld, m.jnt_limited_ball_adr.size),
            inputs=[m, d],
          )

        limit_slide_hinge = m.jnt_limited_slide_hinge_adr.size > 0
        if limit_slide_hinge:
          wp.launch(
            _efc_limit_slide_hinge,
            dim=(d.nworld, m.jnt_limited_slide_hinge_adr.size),
            inputs=[m, d],
          )

        limit_tendon = m.tendon_limited_adr.size > 0
        if limit_tendon:
          wp.launch(
            _efc_limit_tendon,
            dim=(d.nworld, m.tendon_limited_adr.size),
            inputs=[m, d],
          )

        if limit_ball or limit_slide_hinge or limit_tendon:
          wp.launch(_update_nefc, dim=(1,), inputs=[d])

      # contact
      if not (m.opt.disableflags & types.DisableBit.CONTACT.value):
        if m.opt.cone == types.ConeType.PYRAMIDAL.value:
          wp.launch(
            _efc_contact_pyramidal,
            dim=(d.nconmax, 2 * (m.condim_max - 1) if m.condim_max > 1 else 1),
            inputs=[m, d],
          )
        elif m.opt.cone == types.ConeType.ELLIPTIC.value:
          wp.launch(_efc_contact_elliptic, dim=(d.nconmax, m.condim_max), inputs=[m, d])
