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

from typing import Any, Tuple

import numpy as np
import warp as wp

from . import math
from . import smooth
from .types import MJ_MINVAL
from .types import Data
from .types import DataType
from .types import DisableBit
from .types import Model
from .types import ObjType
from .types import SensorType
from .warp_util import event_scope
from .warp_util import kernel


@wp.func
def _apply_cutoff(m: Model, sensorid: int, sensor: wp.float32) -> wp.float32:
  cutoff = m.sensor_cutoff[sensorid]

  if cutoff > 0.0:
    datatype = m.sensor_datatype[sensorid]
    if datatype == int(DataType.REAL.value):
      return wp.clamp(sensor, -cutoff, cutoff)
    elif datatype == int(DataType.POSITIVE.value):
      return wp.min(sensor, cutoff)

  return sensor


@wp.func
def _apply_cutoff(m: Model, sensorid: int, sensordim: int, sensor: Any) -> Any:
  cutoff = m.sensor_cutoff[sensorid]

  if cutoff > 0.0:
    datatype = m.sensor_datatype[sensorid]
    if datatype == int(DataType.REAL.value):
      for i in range(sensordim):
        sensor[i] = wp.clamp(sensor[i], -cutoff, cutoff)
    elif datatype == int(DataType.POSITIVE.value):
      for i in range(sensordim):
        sensor[i] = wp.min(sensor[i], cutoff)

  return sensor


@wp.func
def _cam_projection(
  m: Model, d: Data, worldid: int, objid: int, refid: int
) -> wp.vec2f:
  sensorsize = m.cam_sensorsize[refid]
  intrinsic = m.cam_intrinsic[refid]
  fovy = m.cam_fovy[refid]
  res = m.cam_resolution[refid]

  target_xpos = d.site_xpos[worldid, objid]
  xpos = d.cam_xpos[worldid, refid]
  xmat = d.cam_xmat[worldid, refid]

  translation = wp.mat44f(
    1.0,
    0.0,
    0.0,
    -xpos[0],
    0.0,
    1.0,
    0.0,
    -xpos[1],
    0.0,
    0.0,
    1.0,
    -xpos[2],
    0.0,
    0.0,
    0.0,
    1.0,
  )
  rotation = wp.mat44f(
    xmat[0, 0],
    xmat[1, 0],
    xmat[2, 0],
    0.0,
    xmat[0, 1],
    xmat[1, 1],
    xmat[2, 1],
    0.0,
    xmat[0, 2],
    xmat[1, 2],
    xmat[2, 2],
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
  )

  # focal transformation matrix (3 x 4)
  if sensorsize[0] != 0.0 and sensorsize[1] != 0.0:
    fx = intrinsic[0] / (sensorsize[0] + MJ_MINVAL) * float(res[0])
    fy = intrinsic[1] / (sensorsize[1] + MJ_MINVAL) * float(res[1])
    focal = wp.mat44f(
      -fx, 0.0, 0.0, 0.0, 0.0, fy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
  else:
    f = 0.5 / wp.tan(fovy * wp.static(wp.pi / 360.0)) * float(res[1])
    focal = wp.mat44f(
      -f, 0.0, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )

  # image matrix (3 x 3)
  image = wp.mat44f(
    1.0,
    0.0,
    0.5 * float(res[0]),
    0.0,
    0.0,
    1.0,
    0.5 * float(res[1]),
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
  )

  # projection matrix (3 x 4): product of all 4 matrices
  # TODO(team): compute proj directly
  proj = image @ focal @ rotation @ translation

  # projection matrix multiples homogenous [x, y, z, 1] vectors
  pos_hom = wp.vec4(target_xpos[0], target_xpos[1], target_xpos[2], 1.0)

  # project world coordinates into pixel space, see:
  # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
  pixel_coord_hom = proj @ pos_hom

  # avoid dividing by tiny numbers
  denom = pixel_coord_hom[2]
  if wp.abs(denom) < MJ_MINVAL:
    denom = wp.clamp(denom, -MJ_MINVAL, MJ_MINVAL)

  # compute projection
  return wp.vec2f(pixel_coord_hom[0], pixel_coord_hom[1]) / denom


@wp.func
def _joint_pos(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.qpos[worldid, m.jnt_qposadr[objid]]


@wp.func
def _tendon_pos(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.ten_length[worldid, objid]


@wp.func
def _actuator_length(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.actuator_length[worldid, objid]


@wp.func
def _ball_quat(m: Model, d: Data, worldid: int, objid: int) -> wp.quat:
  jnt_qposadr = m.jnt_qposadr[objid]
  quat = wp.quat(
    d.qpos[worldid, jnt_qposadr + 0],
    d.qpos[worldid, jnt_qposadr + 1],
    d.qpos[worldid, jnt_qposadr + 2],
    d.qpos[worldid, jnt_qposadr + 3],
  )
  return wp.normalize(quat)


@wp.func
def _frame_pos(
  m: Model, d: Data, worldid: int, objid: int, objtype: int, refid: int
) -> wp.vec3:
  if objtype == int(ObjType.BODY.value):
    xpos = d.xipos[worldid, objid]
    if refid == -1:
      return xpos
    xpos_ref = d.xipos[worldid, refid]
    xmat_ref = d.ximat[worldid, refid]
  elif objtype == int(ObjType.XBODY.value):
    xpos = d.xpos[worldid, objid]
    if refid == -1:
      return xpos
    xpos_ref = d.xpos[worldid, refid]
    xmat_ref = d.xmat[worldid, refid]
  elif objtype == int(ObjType.GEOM.value):
    xpos = d.geom_xpos[worldid, objid]
    if refid == -1:
      return xpos
    xpos_ref = d.geom_xpos[worldid, refid]
    xmat_ref = d.geom_xmat[worldid, refid]
  elif objtype == int(ObjType.SITE.value):
    xpos = d.site_xpos[worldid, objid]
    if refid == -1:
      return xpos
    xpos_ref = d.site_xpos[worldid, refid]
    xmat_ref = d.site_xmat[worldid, refid]

  # TODO(team): camera

  else:  # UNKNOWN
    return wp.vec3(0.0)

  return wp.transpose(xmat_ref) @ (xpos - xpos_ref)


@wp.func
def _frame_axis(
  m: Model, d: Data, worldid: int, objid: int, objtype: int, refid: int, frame_axis: int
) -> wp.vec3:
  if objtype == int(ObjType.BODY.value):
    xmat = d.ximat[worldid, objid]
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
    if refid == -1:
      return axis
    xmat_ref = d.ximat[worldid, refid]
  elif objtype == int(ObjType.XBODY.value):
    xmat = d.xmat[worldid, objid]
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
    if refid == -1:
      return axis
    xmat_ref = d.xmat[worldid, refid]
  elif objtype == int(ObjType.GEOM.value):
    xmat = d.geom_xmat[worldid, objid]
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
    if refid == -1:
      return axis
    xmat_ref = d.geom_xmat[worldid, refid]
  elif objtype == int(ObjType.SITE.value):
    xmat = d.site_xmat[worldid, objid]
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
    if refid == -1:
      return axis
    xmat_ref = d.site_xmat[worldid, refid]

  # TODO(team): camera

  else:  # UNKNOWN
    xmat = wp.identity(3, dtype=wp.float32)
    return wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])

  return wp.transpose(xmat_ref) @ axis


@wp.func
def _frame_quat(
  m: Model, d: Data, worldid: int, objid: int, objtype: int, refid: int
) -> wp.quat:
  if objtype == int(ObjType.BODY.value):
    quat = math.mul_quat(d.xquat[worldid, objid], m.body_iquat[objid])
    if refid == -1:
      return quat
    refquat = math.mul_quat(d.xquat[worldid, refid], m.body_iquat[refid])
  elif objtype == int(ObjType.XBODY.value):
    quat = d.xquat[worldid, objid]
    if refid == -1:
      return quat
    refquat = d.xquat[worldid, refid]
  elif objtype == int(ObjType.GEOM.value):
    quat = math.mul_quat(d.xquat[worldid, m.geom_bodyid[objid]], m.geom_quat[objid])
    if refid == -1:
      return quat
    refquat = math.mul_quat(d.xquat[worldid, m.geom_bodyid[refid]], m.geom_quat[refid])
  elif objtype == int(ObjType.SITE.value):
    quat = math.mul_quat(d.xquat[worldid, m.site_bodyid[objid]], m.site_quat[objid])
    if refid == -1:
      return quat
    refquat = math.mul_quat(d.xquat[worldid, m.site_bodyid[refid]], m.site_quat[refid])

  # TODO(team): camera

  else:  # UNKNOWN
    return wp.quat(1.0, 0.0, 0.0, 0.0)

  return math.mul_quat(math.quat_inv(refquat), quat)


@wp.func
def _subtree_com(m: Model, d: Data, worldid: int, objid: int) -> wp.vec3:
  return d.subtree_com[worldid, objid]


@wp.func
def _clock(m: Model, d: Data, worldid: int) -> wp.float32:
  return d.time[worldid]


@event_scope
def sensor_pos(m: Model, d: Data):
  """Compute position-dependent sensor values."""

  @kernel
  def _sensor_pos(m: Model, d: Data):
    worldid, posid = wp.tid()
    posadr = m.sensor_pos_adr[posid]
    sensortype = m.sensor_type[posadr]
    objid = m.sensor_objid[posadr]
    adr = m.sensor_adr[posadr]

    if sensortype == int(SensorType.CAMPROJECTION.value):
      refid = m.sensor_refid[posadr]
      cam_projection = _cam_projection(m, d, worldid, objid, refid)
      cam_projection = _apply_cutoff(m, posadr, 2, cam_projection)
      d.sensordata[worldid, adr + 0] = cam_projection[0]
      d.sensordata[worldid, adr + 1] = cam_projection[1]
    elif sensortype == int(SensorType.JOINTPOS.value):
      joint_pos = _joint_pos(m, d, worldid, objid)
      joint_pos = _apply_cutoff(m, posadr, joint_pos)
      d.sensordata[worldid, adr] = joint_pos
    elif sensortype == int(SensorType.TENDONPOS.value):
      tendon_pos = _tendon_pos(m, d, worldid, objid)
      tendon_pos = _apply_cutoff(m, posadr, tendon_pos)
      d.sensordata[worldid, adr] = tendon_pos
    elif sensortype == int(SensorType.ACTUATORPOS.value):
      actuator_length = _actuator_length(m, d, worldid, objid)
      actuator_length = _apply_cutoff(m, posadr, actuator_length)
      d.sensordata[worldid, adr] = actuator_length
    elif sensortype == int(SensorType.BALLQUAT.value):
      quat = _ball_quat(m, d, worldid, objid)
      d.sensordata[worldid, adr + 0] = quat[0]
      d.sensordata[worldid, adr + 1] = quat[1]
      d.sensordata[worldid, adr + 2] = quat[2]
      d.sensordata[worldid, adr + 3] = quat[3]
    elif sensortype == int(SensorType.FRAMEPOS.value):
      objtype = m.sensor_objtype[posadr]
      refid = m.sensor_refid[posadr]
      framepos = _frame_pos(m, d, worldid, objid, objtype, refid)
      framepos = _apply_cutoff(m, posadr, 3, framepos)
      d.sensordata[worldid, adr + 0] = framepos[0]
      d.sensordata[worldid, adr + 1] = framepos[1]
      d.sensordata[worldid, adr + 2] = framepos[2]
    elif (
      sensortype == int(SensorType.FRAMEXAXIS.value)
      or sensortype == int(SensorType.FRAMEYAXIS.value)
      or sensortype == int(SensorType.FRAMEZAXIS.value)
    ):
      objtype = m.sensor_objtype[posadr]
      refid = m.sensor_refid[posadr]
      if sensortype == int(SensorType.FRAMEXAXIS.value):
        axis = 0
      elif sensortype == int(SensorType.FRAMEYAXIS.value):
        axis = 1
      elif sensortype == int(SensorType.FRAMEZAXIS.value):
        axis = 2
      frameaxis = _frame_axis(m, d, worldid, objid, objtype, refid, axis)
      d.sensordata[worldid, adr + 0] = frameaxis[0]
      d.sensordata[worldid, adr + 1] = frameaxis[1]
      d.sensordata[worldid, adr + 2] = frameaxis[2]
    elif sensortype == int(SensorType.FRAMEQUAT.value):
      objtype = m.sensor_objtype[posadr]
      refid = m.sensor_refid[posadr]
      frame_quat = _frame_quat(m, d, worldid, objid, objtype, refid)
      d.sensordata[worldid, adr + 0] = frame_quat[0]
      d.sensordata[worldid, adr + 1] = frame_quat[1]
      d.sensordata[worldid, adr + 2] = frame_quat[2]
      d.sensordata[worldid, adr + 3] = frame_quat[3]
    elif sensortype == int(SensorType.SUBTREECOM.value):
      subtree_com = _subtree_com(m, d, worldid, objid)
      subtree_com = _apply_cutoff(m, posadr, 3, subtree_com)
      d.sensordata[worldid, adr + 0] = subtree_com[0]
      d.sensordata[worldid, adr + 1] = subtree_com[1]
      d.sensordata[worldid, adr + 2] = subtree_com[2]
    elif sensortype == int(SensorType.CLOCK.value):
      clock = _clock(m, d, worldid)
      clock = _apply_cutoff(m, posadr, clock)
      d.sensordata[worldid, adr] = clock

  if (m.sensor_pos_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  wp.launch(_sensor_pos, dim=(d.nworld, m.sensor_pos_adr.size), inputs=[m, d])


@wp.func
def _velocimeter(m: Model, d: Data, worldid: int, objid: int) -> wp.vec3:
  bodyid = m.site_bodyid[objid]
  pos = d.site_xpos[worldid, objid]
  rot = d.site_xmat[worldid, objid]
  cvel = d.cvel[worldid, bodyid]
  ang = wp.spatial_top(cvel)
  lin = wp.spatial_bottom(cvel)
  subtree_com = d.subtree_com[worldid, m.body_rootid[bodyid]]
  dif = pos - subtree_com
  return wp.transpose(rot) @ (lin - wp.cross(dif, ang))


@wp.func
def _gyro(m: Model, d: Data, worldid: int, objid: int) -> wp.vec3:
  bodyid = m.site_bodyid[objid]
  rot = d.site_xmat[worldid, objid]
  cvel = d.cvel[worldid, bodyid]
  ang = wp.spatial_top(cvel)
  return wp.transpose(rot) @ ang


@wp.func
def _joint_vel(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.qvel[worldid, m.jnt_dofadr[objid]]


@wp.func
def _tendon_vel(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.ten_velocity[worldid, objid]


@wp.func
def _actuator_vel(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.actuator_velocity[worldid, objid]


@wp.func
def _ball_ang_vel(m: Model, d: Data, worldid: int, objid: int) -> wp.vec3:
  jnt_dofadr = m.jnt_dofadr[objid]
  return wp.vec3(
    d.qvel[worldid, jnt_dofadr + 0],
    d.qvel[worldid, jnt_dofadr + 1],
    d.qvel[worldid, jnt_dofadr + 2],
  )


@wp.func
def _cvel_offset(
  m: Model, d: Data, worldid: int, objtype: int, objid: int
) -> Tuple[wp.spatial_vector, wp.vec3]:
  if objtype == int(ObjType.BODY.value):
    pos = d.xipos[worldid, objid]
    bodyid = objid
  elif objtype == int(ObjType.XBODY.value):
    pos = d.xpos[worldid, objid]
    bodyid = objid
  elif objtype == int(ObjType.GEOM.value):
    pos = d.geom_xpos[worldid, objid]
    bodyid = m.geom_bodyid[objid]
  elif objtype == int(ObjType.SITE.value):
    pos = d.site_xpos[worldid, objid]
    bodyid = m.site_bodyid[objid]
  elif objtype == int(ObjType.CAMERA.value):
    pos = d.cam_xpos[worldid, objid]
    bodyid = m.cam_bodyid[objid]
  else:  # UNKNOWN
    pos = wp.vec3(0.0)
    bodyid = 0

  return d.cvel[worldid, bodyid], pos - d.subtree_com[worldid, m.body_rootid[bodyid]]


@wp.func
def _frame_linvel(
  m: Model, d: Data, worldid: int, objid: int, objtype: int, refid: int, reftype: int
) -> wp.vec3:
  if objtype == int(ObjType.BODY.value):
    xpos = d.xipos[worldid, objid]
  elif objtype == int(ObjType.XBODY.value):
    xpos = d.xpos[worldid, objid]
  elif objtype == int(ObjType.GEOM.value):
    xpos = d.geom_xpos[worldid, objid]
  elif objtype == int(ObjType.SITE.value):
    xpos = d.site_xpos[worldid, objid]
  elif objtype == int(ObjType.CAMERA.value):
    xpos = d.cam_xpos[worldid, objid]
  else:  # UNKNOWN
    xpos = wp.vec3(0.0)

  if reftype == int(ObjType.BODY.value):
    xposref = d.xipos[worldid, refid]
    xmatref = d.ximat[worldid, refid]
  elif reftype == int(ObjType.XBODY.value):
    xposref = d.xpos[worldid, refid]
    xmatref = d.xmat[worldid, refid]
  elif reftype == int(ObjType.GEOM.value):
    xposref = d.geom_xpos[worldid, refid]
    xmatref = d.geom_xmat[worldid, refid]
  elif reftype == int(ObjType.SITE.value):
    xposref = d.site_xpos[worldid, refid]
    xmatref = d.site_xmat[worldid, refid]
  elif reftype == int(ObjType.CAMERA.value):
    xposref = d.cam_xpos[worldid, refid]
    xmatref = d.cam_xmat[worldid, refid]
  else:  # UNKNOWN
    xposref = wp.vec3(0.0)
    xmatref = wp.identity(3, dtype=wp.float32)

  cvel, offset = _cvel_offset(m, d, worldid, objtype, objid)
  cvelref, offsetref = _cvel_offset(m, d, worldid, reftype, refid)
  clinvel = wp.spatial_bottom(cvel)
  cangvel = wp.spatial_top(cvel)
  cangvelref = wp.spatial_top(cvelref)
  xlinvel = clinvel - wp.cross(offset, cangvel)

  if refid > -1:
    clinvelref = wp.spatial_bottom(cvelref)
    xlinvelref = clinvelref - wp.cross(offsetref, cangvelref)
    rvec = xpos - xposref
    rel_vel = xlinvel - xlinvelref + wp.cross(rvec, cangvelref)
    return wp.transpose(xmatref) @ rel_vel
  else:
    return xlinvel


@wp.func
def _frame_angvel(
  m: Model, d: Data, worldid: int, objid: int, objtype: int, refid: int, reftype: int
) -> wp.vec3:
  cvel, _ = _cvel_offset(m, d, worldid, objtype, objid)
  cangvel = wp.spatial_top(cvel)

  if refid > -1:
    if reftype == int(ObjType.BODY.value):
      xmatref = d.ximat[worldid, refid]
    elif reftype == int(ObjType.XBODY.value):
      xmatref = d.xmat[worldid, refid]
    elif reftype == int(ObjType.GEOM.value):
      xmatref = d.geom_xmat[worldid, refid]
    elif reftype == int(ObjType.SITE.value):
      xmatref = d.site_xmat[worldid, refid]
    elif reftype == int(ObjType.CAMERA.value):
      xmatref = d.cam_xmat[worldid, refid]
    else:  # UNKNOWN
      xmatref = wp.identity(3, dtype=wp.float32)

    cvelref, _ = _cvel_offset(m, d, worldid, reftype, refid)
    cangvelref = wp.spatial_top(cvelref)

    return wp.transpose(xmatref) @ (cangvel - cangvelref)
  else:
    return cangvel


@wp.func
def _subtree_linvel(m: Model, d: Data, worldid: int, objid: int) -> wp.vec3:
  return d.subtree_linvel[worldid, objid]


@wp.func
def _subtree_angmom(m: Model, d: Data, worldid: int, objid: int) -> wp.vec3:
  return d.subtree_angmom[worldid, objid]


@event_scope
def sensor_vel(m: Model, d: Data):
  """Compute velocity-dependent sensor values."""

  @kernel
  def _sensor_vel(m: Model, d: Data):
    worldid, velid = wp.tid()
    veladr = m.sensor_vel_adr[velid]
    sensortype = m.sensor_type[veladr]
    objid = m.sensor_objid[veladr]
    adr = m.sensor_adr[veladr]

    if sensortype == int(SensorType.VELOCIMETER.value):
      vel = _velocimeter(m, d, worldid, objid)
      vel = _apply_cutoff(m, veladr, 3, vel)
      d.sensordata[worldid, adr + 0] = vel[0]
      d.sensordata[worldid, adr + 1] = vel[1]
      d.sensordata[worldid, adr + 2] = vel[2]
    elif sensortype == int(SensorType.GYRO.value):
      gyro = _gyro(m, d, worldid, objid)
      gyro = _apply_cutoff(m, veladr, 3, gyro)
      d.sensordata[worldid, adr + 0] = gyro[0]
      d.sensordata[worldid, adr + 1] = gyro[1]
      d.sensordata[worldid, adr + 2] = gyro[2]
    elif sensortype == int(SensorType.JOINTVEL.value):
      joint_vel = _joint_vel(m, d, worldid, objid)
      joint_vel = _apply_cutoff(m, veladr, joint_vel)
      d.sensordata[worldid, adr] = joint_vel
    elif sensortype == int(SensorType.TENDONVEL.value):
      tendon_vel = _tendon_vel(m, d, worldid, objid)
      tendon_vel = _apply_cutoff(m, veladr, tendon_vel)
      d.sensordata[worldid, adr] = tendon_vel
    elif sensortype == int(SensorType.ACTUATORVEL.value):
      actuator_vel = _actuator_vel(m, d, worldid, objid)
      actuator_vel = _apply_cutoff(m, veladr, actuator_vel)
      d.sensordata[worldid, adr] = actuator_vel
    elif sensortype == int(SensorType.BALLANGVEL.value):
      angvel = _ball_ang_vel(m, d, worldid, objid)
      angvel = _apply_cutoff(m, veladr, 3, angvel)
      d.sensordata[worldid, adr + 0] = angvel[0]
      d.sensordata[worldid, adr + 1] = angvel[1]
      d.sensordata[worldid, adr + 2] = angvel[2]
    elif sensortype == int(SensorType.FRAMELINVEL.value):
      objtype = m.sensor_objtype[veladr]
      refid = m.sensor_refid[veladr]
      reftype = m.sensor_reftype[veladr]
      frame_linvel = _frame_linvel(m, d, worldid, objid, objtype, refid, reftype)
      frame_linvel = _apply_cutoff(m, veladr, 3, frame_linvel)
      d.sensordata[worldid, adr + 0] = frame_linvel[0]
      d.sensordata[worldid, adr + 1] = frame_linvel[1]
      d.sensordata[worldid, adr + 2] = frame_linvel[2]
    elif sensortype == int(SensorType.FRAMEANGVEL.value):
      objtype = m.sensor_objtype[veladr]
      refid = m.sensor_refid[veladr]
      reftype = m.sensor_reftype[veladr]
      frame_angvel = _frame_angvel(m, d, worldid, objid, objtype, refid, reftype)
      frame_angvel = _apply_cutoff(m, veladr, 3, frame_angvel)
      d.sensordata[worldid, adr + 0] = frame_angvel[0]
      d.sensordata[worldid, adr + 1] = frame_angvel[1]
      d.sensordata[worldid, adr + 2] = frame_angvel[2]
    elif sensortype == int(SensorType.SUBTREELINVEL.value):
      subtree_linvel = _subtree_linvel(m, d, worldid, objid)
      subtree_linvel = _apply_cutoff(m, veladr, 3, subtree_linvel)
      d.sensordata[worldid, adr + 0] = subtree_linvel[0]
      d.sensordata[worldid, adr + 1] = subtree_linvel[1]
      d.sensordata[worldid, adr + 2] = subtree_linvel[2]
    elif sensortype == int(SensorType.SUBTREEANGMOM.value):
      subtree_angmom = _subtree_angmom(m, d, worldid, objid)
      subtree_angmom = _apply_cutoff(m, veladr, 3, subtree_angmom)
      d.sensordata[worldid, adr + 0] = subtree_angmom[0]
      d.sensordata[worldid, adr + 1] = subtree_angmom[1]
      d.sensordata[worldid, adr + 2] = subtree_angmom[2]

  if (m.sensor_vel_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  if m.sensor_subtree_vel:
    smooth.subtree_vel(m, d)

  wp.launch(_sensor_vel, dim=(d.nworld, m.sensor_vel_adr.size), inputs=[m, d])


@wp.func
def _accelerometer(m: Model, d: Data, worldid: int, objid: int) -> wp.vec3:
  bodyid = m.site_bodyid[objid]
  rot = d.site_xmat[worldid, objid]
  rotT = wp.transpose(rot)
  cvel = d.cvel[worldid, bodyid]
  cvel_top = wp.spatial_top(cvel)
  cvel_bottom = wp.spatial_bottom(cvel)
  cacc = d.cacc[worldid, bodyid]
  cacc_top = wp.spatial_top(cacc)
  cacc_bottom = wp.spatial_bottom(cacc)
  dif = d.site_xpos[worldid, objid] - d.subtree_com[worldid, m.body_rootid[bodyid]]
  ang = rotT @ cvel_top
  lin = rotT @ (cvel_bottom - wp.cross(dif, cvel_top))
  acc = rotT @ (cacc_bottom - wp.cross(dif, cacc_top))
  correction = wp.cross(ang, lin)
  return acc + correction


@wp.func
def _force(m: Model, d: Data, worldid: int, objid: int) -> wp.vec3:
  bodyid = m.site_bodyid[objid]
  cfrc_int = d.cfrc_int[worldid, bodyid]
  site_xmat = d.site_xmat[worldid, objid]
  return wp.transpose(site_xmat) @ wp.spatial_bottom(cfrc_int)


@wp.func
def _torque(m: Model, d: Data, worldid: int, objid: int) -> wp.vec3:
  bodyid = m.site_bodyid[objid]
  cfrc_int = d.cfrc_int[worldid, bodyid]
  site_xmat = d.site_xmat[worldid, objid]
  dif = d.site_xpos[worldid, objid] - d.subtree_com[worldid, m.body_rootid[bodyid]]
  return wp.transpose(site_xmat) @ (
    wp.spatial_top(cfrc_int) - wp.cross(dif, wp.spatial_bottom(cfrc_int))
  )


@wp.func
def _actuator_force(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.actuator_force[worldid, objid]


@wp.func
def _joint_actuator_force(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.qfrc_actuator[worldid, m.jnt_dofadr[objid]]


@wp.func
def _framelinacc(m: Model, d: Data, worldid: int, objid: int, objtype: int) -> wp.vec3:
  if objtype == int(ObjType.BODY.value):
    bodyid = objid
    pos = d.xipos[worldid, objid]
  elif objtype == int(ObjType.XBODY.value):
    bodyid = objid
    pos = d.xpos[worldid, objid]
  elif objtype == int(ObjType.GEOM.value):
    bodyid = m.geom_bodyid[objid]
    pos = d.geom_xpos[worldid, objid]
  elif objtype == int(ObjType.SITE.value):
    bodyid = m.site_bodyid[objid]
    pos = d.site_xpos[worldid, objid]
  # TODO(team): camera
  else:  # UNKNOWN
    bodyid = 0
    pos = wp.vec3(0.0)

  cacc = d.cacc[worldid, bodyid]
  cvel = d.cvel[worldid, bodyid]
  offset = pos - d.subtree_com[worldid, m.body_rootid[bodyid]]
  ang = wp.spatial_top(cvel)
  lin = wp.spatial_bottom(cvel) - wp.cross(offset, ang)
  acc = wp.spatial_bottom(cacc) - wp.cross(offset, wp.spatial_top(cacc))
  correction = wp.cross(ang, lin)

  return acc + correction


@wp.func
def _frameangacc(m: Model, d: Data, worldid: int, objid: int, objtype: int) -> wp.vec3:
  if objtype == int(ObjType.BODY.value) or objtype == int(ObjType.XBODY.value):
    bodyid = objid
  elif objtype == int(ObjType.GEOM.value):
    bodyid = m.geom_bodyid[objid]
  elif objtype == int(ObjType.SITE.value):
    bodyid = m.site_bodyid[objid]
  # TODO(team): camera
  else:  # UNKNOWN
    bodyid = 0

  return wp.spatial_top(d.cacc[worldid, bodyid])


@event_scope
def sensor_acc(m: Model, d: Data):
  """Compute acceleration-dependent sensor values."""

  @kernel
  def _sensor_acc(m: Model, d: Data):
    worldid, accid = wp.tid()
    accadr = m.sensor_acc_adr[accid]
    sensortype = m.sensor_type[accadr]
    objid = m.sensor_objid[accadr]
    adr = m.sensor_adr[accadr]

    if sensortype == int(SensorType.ACCELEROMETER.value):
      accelerometer = _accelerometer(m, d, worldid, objid)
      accelerometer = _apply_cutoff(m, accadr, 3, accelerometer)
      d.sensordata[worldid, adr + 0] = accelerometer[0]
      d.sensordata[worldid, adr + 1] = accelerometer[1]
      d.sensordata[worldid, adr + 2] = accelerometer[2]
    elif sensortype == int(SensorType.FORCE.value):
      force = _force(m, d, worldid, objid)
      force = _apply_cutoff(m, accadr, 3, force)
      d.sensordata[worldid, adr + 0] = force[0]
      d.sensordata[worldid, adr + 1] = force[1]
      d.sensordata[worldid, adr + 2] = force[2]
    elif sensortype == int(SensorType.TORQUE.value):
      torque = _torque(m, d, worldid, objid)
      torque = _apply_cutoff(m, accadr, 3, torque)
      d.sensordata[worldid, adr + 0] = torque[0]
      d.sensordata[worldid, adr + 1] = torque[1]
      d.sensordata[worldid, adr + 2] = torque[2]
    elif sensortype == int(SensorType.ACTUATORFRC.value):
      actuator_force = _actuator_force(m, d, worldid, objid)
      actuator_force = _apply_cutoff(m, accadr, actuator_force)
      d.sensordata[worldid, adr] = actuator_force
    elif sensortype == int(SensorType.JOINTACTFRC.value):
      joint_actuator_force = _joint_actuator_force(m, d, worldid, objid)
      joint_actuator_force = _apply_cutoff(m, accadr, joint_actuator_force)
      d.sensordata[worldid, adr] = joint_actuator_force
    elif sensortype == int(SensorType.FRAMELINACC.value):
      objtype = m.sensor_objtype[accadr]
      frame_linacc = _framelinacc(m, d, worldid, objid, objtype)
      frame_linacc = _apply_cutoff(m, accadr, 3, frame_linacc)
      d.sensordata[worldid, adr + 0] = frame_linacc[0]
      d.sensordata[worldid, adr + 1] = frame_linacc[1]
      d.sensordata[worldid, adr + 2] = frame_linacc[2]
    elif sensortype == int(SensorType.FRAMEANGACC.value):
      objtype = m.sensor_objtype[accadr]
      frame_angacc = _frameangacc(m, d, worldid, objid, objtype)
      frame_angacc = _apply_cutoff(m, accadr, 3, frame_angacc)
      d.sensordata[worldid, adr + 0] = frame_angacc[0]
      d.sensordata[worldid, adr + 1] = frame_angacc[1]
      d.sensordata[worldid, adr + 2] = frame_angacc[2]

  if (m.sensor_acc_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  if m.sensor_rne_postconstraint:
    smooth.rne_postconstraint(m, d)

  wp.launch(_sensor_acc, dim=(d.nworld, m.sensor_acc_adr.size), inputs=[m, d])
