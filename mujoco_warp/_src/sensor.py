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


@wp.func
def _write_scalar(
  # Model:
  sensor_datatype: wp.array(dtype=int),
  sensor_adr: wp.array(dtype=int),
  sensor_cutoff: wp.array(dtype=float),
  # In:
  sensorid: int,
  sensor: Any,
  # Out:
  out: wp.array(dtype=float),
):
  adr = sensor_adr[sensorid]
  cutoff = sensor_cutoff[sensorid]

  if cutoff > 0.0:
    datatype = sensor_datatype[sensorid]
    if datatype == int(DataType.REAL.value):
      out[adr] = wp.clamp(sensor, -cutoff, cutoff)
    elif datatype == int(DataType.POSITIVE.value):
      out[adr] = wp.min(sensor, cutoff)
  else:
    out[adr] = sensor


@wp.func
def _write_vector(
  # Model:
  sensor_datatype: wp.array(dtype=int),
  sensor_adr: wp.array(dtype=int),
  sensor_cutoff: wp.array(dtype=float),
  # In:
  sensorid: int,
  sensordim: int,
  sensor: Any,
  # Out:
  out: wp.array(dtype=float),
):
  adr = sensor_adr[sensorid]
  cutoff = sensor_cutoff[sensorid]

  if cutoff > 0.0:
    datatype = sensor_datatype[sensorid]
    if datatype == int(DataType.REAL.value):
      for i in range(sensordim):
        out[adr + i] = wp.clamp(sensor[i], -cutoff, cutoff)
    elif datatype == int(DataType.POSITIVE.value):
      for i in range(sensordim):
        out[adr + i] = wp.min(sensor[i], cutoff)
  else:
    for i in range(sensordim):
      out[adr + i] = sensor[i]


@wp.func
def _cam_projection(
  # Model:
  cam_fovy: wp.array(dtype=float),
  cam_resolution: wp.array(dtype=wp.vec2i),
  cam_sensorsize: wp.array(dtype=wp.vec2),
  cam_intrinsic: wp.array(dtype=wp.vec4),
  # Data in:
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  cam_xpos_in: wp.array2d(dtype=wp.vec3),
  cam_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  worldid: int,
  objid: int,
  refid: int,
) -> wp.vec2:
  sensorsize = cam_sensorsize[refid]
  intrinsic = cam_intrinsic[refid]
  fovy = cam_fovy[refid]
  res = cam_resolution[refid]

  target_xpos = site_xpos_in[worldid, objid]
  xpos = cam_xpos_in[worldid, refid]
  xmat = cam_xmat_in[worldid, refid]

  translation = wp.mat44f(1.0, 0.0, 0.0, -xpos[0], 0.0, 1.0, 0.0, -xpos[1], 0.0, 0.0, 1.0, -xpos[2], 0.0, 0.0, 0.0, 1.0)
  rotation = wp.mat44f(
    xmat[0, 0], xmat[1, 0], xmat[2, 0], 0.0,
    xmat[0, 1], xmat[1, 1], xmat[2, 1], 0.0,
    xmat[0, 2], xmat[1, 2], xmat[2, 2], 0.0,
    0.0, 0.0, 0.0, 1.0,
  )  # fmt: skip

  # focal transformation matrix (3 x 4)
  if sensorsize[0] != 0.0 and sensorsize[1] != 0.0:
    fx = intrinsic[0] / (sensorsize[0] + MJ_MINVAL) * float(res[0])
    fy = intrinsic[1] / (sensorsize[1] + MJ_MINVAL) * float(res[1])
    focal = wp.mat44f(-fx, 0.0, 0.0, 0.0, 0.0, fy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  else:
    f = 0.5 / wp.tan(fovy * wp.static(wp.pi / 360.0)) * float(res[1])
    focal = wp.mat44f(-f, 0.0, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  # image matrix (3 x 3)
  image = wp.mat44f(
    1.0, 0.0, 0.5 * float(res[0]), 0.0, 0.0, 1.0, 0.5 * float(res[1]), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
def _joint_pos(jnt_qposadr: wp.array(dtype=int), qpos_in: wp.array2d(dtype=float), worldid: int, objid: int) -> float:
  return qpos_in[worldid, jnt_qposadr[objid]]


@wp.func
def _tendon_pos(ten_length_in: wp.array2d(dtype=float), worldid: int, objid: int) -> float:
  return ten_length_in[worldid, objid]


@wp.func
def _actuator_pos(actuator_length_in: wp.array2d(dtype=float), worldid: int, objid: int) -> float:
  return actuator_length_in[worldid, objid]


@wp.func
def _ball_quat(jnt_qposadr: wp.array(dtype=int), qpos_in: wp.array2d(dtype=float), worldid: int, objid: int) -> wp.quat:
  adr = jnt_qposadr[objid]
  quat = wp.quat(
    qpos_in[worldid, adr + 0],
    qpos_in[worldid, adr + 1],
    qpos_in[worldid, adr + 2],
    qpos_in[worldid, adr + 3],
  )
  return wp.normalize(quat)


@wp.func
def _frame_pos(
  # Data in:
  xpos_in: wp.array2d(dtype=wp.vec3),
  xmat_in: wp.array2d(dtype=wp.mat33),
  xipos_in: wp.array2d(dtype=wp.vec3),
  ximat_in: wp.array2d(dtype=wp.mat33),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  cam_xpos_in: wp.array2d(dtype=wp.vec3),
  cam_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  worldid: int,
  objid: int,
  objtype: int,
  refid: int,
  reftype: int,
) -> wp.vec3:
  if objtype == int(ObjType.BODY.value):
    xpos = xipos_in[worldid, objid]
  elif objtype == int(ObjType.XBODY.value):
    xpos = xpos_in[worldid, objid]
  elif objtype == int(ObjType.GEOM.value):
    xpos = geom_xpos_in[worldid, objid]
  elif objtype == int(ObjType.SITE.value):
    xpos = site_xpos_in[worldid, objid]
  elif objtype == int(ObjType.CAMERA.value):
    xpos = cam_xpos_in[worldid, objid]
  else:  # UNKNOWN
    xpos = wp.vec3(0.0)

  if refid == -1:
    return xpos

  if reftype == int(ObjType.BODY.value):
    xpos_ref = xipos_in[worldid, refid]
    xmat_ref = ximat_in[worldid, refid]
  elif objtype == int(ObjType.XBODY.value):
    xpos_ref = xpos_in[worldid, refid]
    xmat_ref = xmat_in[worldid, refid]
  elif reftype == int(ObjType.GEOM.value):
    xpos_ref = geom_xpos_in[worldid, refid]
    xmat_ref = geom_xmat_in[worldid, refid]
  elif reftype == int(ObjType.SITE.value):
    xpos_ref = site_xpos_in[worldid, refid]
    xmat_ref = site_xmat_in[worldid, refid]
  elif reftype == int(ObjType.CAMERA.value):
    xpos_ref = cam_xpos_in[worldid, refid]
    xmat_ref = cam_xmat_in[worldid, refid]

  else:  # UNKNOWN
    xpos_ref = wp.vec3(0.0)
    xmat_ref = wp.identity(3, wp.float32)

  return wp.transpose(xmat_ref) @ (xpos - xpos_ref)


@wp.func
def _frame_axis(
  # Data in:
  xmat_in: wp.array2d(dtype=wp.mat33),
  ximat_in: wp.array2d(dtype=wp.mat33),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  cam_xmat_in: wp.array2d(dtype=wp.mat33),
  # In:
  worldid: int,
  objid: int,
  objtype: int,
  refid: int,
  reftype: int,
  frame_axis: int,
) -> wp.vec3:
  if objtype == int(ObjType.BODY.value):
    xmat = ximat_in[worldid, objid]
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
  elif objtype == int(ObjType.XBODY.value):
    xmat = xmat_in[worldid, objid]
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
  elif objtype == int(ObjType.GEOM.value):
    xmat = geom_xmat_in[worldid, objid]
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
  elif objtype == int(ObjType.SITE.value):
    xmat = site_xmat_in[worldid, objid]
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
  elif objtype == int(ObjType.CAMERA.value):
    xmat = cam_xmat_in[worldid, objid]
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])
  else:  # UNKNOWN
    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])

  if refid == -1:
    return axis

  if reftype == int(ObjType.BODY.value):
    xmat_ref = ximat_in[worldid, refid]
  elif reftype == int(ObjType.XBODY.value):
    xmat_ref = xmat_in[worldid, refid]
  elif reftype == int(ObjType.GEOM.value):
    xmat_ref = geom_xmat_in[worldid, refid]
  elif reftype == int(ObjType.SITE.value):
    xmat_ref = site_xmat_in[worldid, refid]
  elif reftype == int(ObjType.CAMERA.value):
    xmat_ref = cam_xmat_in[worldid, refid]
  else:  # UNKNOWN
    xmat_ref = wp.identity(3, dtype=wp.float32)

  return wp.transpose(xmat_ref) @ axis


@wp.func
def _frame_quat(
  # Model:
  body_iquat: wp.array2d(dtype=wp.quat),
  geom_bodyid: wp.array(dtype=int),
  geom_quat: wp.array2d(dtype=wp.quat),
  site_bodyid: wp.array(dtype=int),
  site_quat: wp.array2d(dtype=wp.quat),
  cam_bodyid: wp.array(dtype=int),
  cam_quat: wp.array2d(dtype=wp.quat),
  # Data in:
  xquat_in: wp.array2d(dtype=wp.quat),
  # In:
  worldid: int,
  objid: int,
  objtype: int,
  refid: int,
  reftype: int,
) -> wp.quat:
  if objtype == int(ObjType.BODY.value):
    quat = math.mul_quat(xquat_in[worldid, objid], body_iquat[worldid, objid])
  elif objtype == int(ObjType.XBODY.value):
    quat = xquat_in[worldid, objid]
  elif objtype == int(ObjType.GEOM.value):
    quat = math.mul_quat(xquat_in[worldid, geom_bodyid[objid]], geom_quat[worldid, objid])
  elif objtype == int(ObjType.SITE.value):
    quat = math.mul_quat(xquat_in[worldid, site_bodyid[objid]], site_quat[worldid, objid])
  elif objtype == int(ObjType.CAMERA.value):
    quat = math.mul_quat(xquat_in[worldid, cam_bodyid[objid]], cam_quat[worldid, objid])
  else:  # UNKNOWN
    quat = wp.quat(1.0, 0.0, 0.0, 0.0)

  if refid == -1:
    return quat

  if reftype == int(ObjType.BODY.value):
    refquat = math.mul_quat(xquat_in[worldid, refid], body_iquat[worldid, refid])
  elif reftype == int(ObjType.XBODY.value):
    refquat = xquat_in[worldid, refid]
  elif reftype == int(ObjType.GEOM.value):
    refquat = math.mul_quat(xquat_in[worldid, geom_bodyid[refid]], geom_quat[worldid, refid])
  elif reftype == int(ObjType.SITE.value):
    refquat = math.mul_quat(xquat_in[worldid, site_bodyid[refid]], site_quat[worldid, refid])
  elif reftype == int(ObjType.CAMERA.value):
    refquat = math.mul_quat(xquat_in[worldid, cam_bodyid[refid]], cam_quat[worldid, refid])
  else:  # UNKNOWN
    refquat = wp.quat(1.0, 0.0, 0.0, 0.0)

  return math.mul_quat(math.quat_inv(refquat), quat)


@wp.func
def _subtree_com(subtree_com_in: wp.array2d(dtype=wp.vec3), worldid: int, objid: int) -> wp.vec3:
  return subtree_com_in[worldid, objid]


@wp.func
def _clock(time_in: wp.array(dtype=float), worldid: int) -> float:
  return time_in[worldid]


@wp.kernel
def _sensor_pos(
  # Model:
  body_iquat: wp.array2d(dtype=wp.quat),
  jnt_qposadr: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  geom_quat: wp.array2d(dtype=wp.quat),
  site_bodyid: wp.array(dtype=int),
  site_quat: wp.array2d(dtype=wp.quat),
  cam_bodyid: wp.array(dtype=int),
  cam_quat: wp.array2d(dtype=wp.quat),
  cam_fovy: wp.array(dtype=float),
  cam_resolution: wp.array(dtype=wp.vec2i),
  cam_sensorsize: wp.array(dtype=wp.vec2),
  cam_intrinsic: wp.array(dtype=wp.vec4),
  sensor_type: wp.array(dtype=int),
  sensor_datatype: wp.array(dtype=int),
  sensor_objtype: wp.array(dtype=int),
  sensor_objid: wp.array(dtype=int),
  sensor_reftype: wp.array(dtype=int),
  sensor_refid: wp.array(dtype=int),
  sensor_adr: wp.array(dtype=int),
  sensor_cutoff: wp.array(dtype=float),
  sensor_pos_adr: wp.array(dtype=int),
  # Data in:
  time_in: wp.array(dtype=float),
  qpos_in: wp.array2d(dtype=float),
  xpos_in: wp.array2d(dtype=wp.vec3),
  xquat_in: wp.array2d(dtype=wp.quat),
  xmat_in: wp.array2d(dtype=wp.mat33),
  xipos_in: wp.array2d(dtype=wp.vec3),
  ximat_in: wp.array2d(dtype=wp.mat33),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  cam_xpos_in: wp.array2d(dtype=wp.vec3),
  cam_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  actuator_length_in: wp.array2d(dtype=float),
  ten_length_in: wp.array2d(dtype=float),
  # Data out:
  sensordata_out: wp.array2d(dtype=float),
):
  worldid, posid = wp.tid()
  sensorid = sensor_pos_adr[posid]
  sensortype = sensor_type[sensorid]
  objid = sensor_objid[sensorid]
  out = sensordata_out[worldid]

  if sensortype == int(SensorType.CAMPROJECTION.value):
    refid = sensor_refid[sensorid]
    vec2 = _cam_projection(
      cam_fovy, cam_resolution, cam_sensorsize, cam_intrinsic, site_xpos_in, cam_xpos_in, cam_xmat_in, worldid, objid, refid
    )
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 2, vec2, out)
  elif sensortype == int(SensorType.JOINTPOS.value):
    val = _joint_pos(jnt_qposadr, qpos_in, worldid, objid)
    _write_scalar(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == int(SensorType.TENDONPOS.value):
    val = _tendon_pos(ten_length_in, worldid, objid)
    _write_scalar(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == int(SensorType.ACTUATORPOS.value):
    val = _actuator_pos(actuator_length_in, worldid, objid)
    _write_scalar(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == int(SensorType.BALLQUAT.value):
    quat = _ball_quat(jnt_qposadr, qpos_in, worldid, objid)
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 4, quat, out)
  elif sensortype == int(SensorType.FRAMEPOS.value):
    objtype = sensor_objtype[sensorid]
    refid = sensor_refid[sensorid]
    reftype = sensor_reftype[sensorid]
    vec3 = _frame_pos(
      xpos_in,
      xmat_in,
      xipos_in,
      ximat_in,
      geom_xpos_in,
      geom_xmat_in,
      site_xpos_in,
      site_xmat_in,
      cam_xpos_in,
      cam_xmat_in,
      worldid,
      objid,
      objtype,
      refid,
      reftype,
    )
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif (
    sensortype == int(SensorType.FRAMEXAXIS.value)
    or sensortype == int(SensorType.FRAMEYAXIS.value)
    or sensortype == int(SensorType.FRAMEZAXIS.value)
  ):
    objtype = sensor_objtype[sensorid]
    refid = sensor_refid[sensorid]
    reftype = sensor_reftype[sensorid]
    if sensortype == int(SensorType.FRAMEXAXIS.value):
      axis = 0
    elif sensortype == int(SensorType.FRAMEYAXIS.value):
      axis = 1
    elif sensortype == int(SensorType.FRAMEZAXIS.value):
      axis = 2
    vec3 = _frame_axis(
      ximat_in, xmat_in, geom_xmat_in, site_xmat_in, cam_xmat_in, worldid, objid, objtype, refid, reftype, axis
    )
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == int(SensorType.FRAMEQUAT.value):
    objtype = sensor_objtype[sensorid]
    refid = sensor_refid[sensorid]
    reftype = sensor_reftype[sensorid]
    quat = _frame_quat(
      body_iquat,
      geom_bodyid,
      geom_quat,
      site_bodyid,
      site_quat,
      cam_bodyid,
      cam_quat,
      xquat_in,
      worldid,
      objid,
      objtype,
      refid,
      reftype,
    )
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 4, quat, out)
  elif sensortype == int(SensorType.SUBTREECOM.value):
    vec3 = _subtree_com(subtree_com_in, worldid, objid)
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == int(SensorType.CLOCK.value):
    val = _clock(time_in, worldid)
    _write_scalar(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)


@event_scope
def sensor_pos(m: Model, d: Data):
  """Compute position-dependent sensor values."""

  if (m.sensor_pos_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  wp.launch(
    _sensor_pos,
    dim=(d.nworld, m.sensor_pos_adr.size),
    inputs=[
      m.body_iquat,
      m.jnt_qposadr,
      m.geom_bodyid,
      m.geom_quat,
      m.site_bodyid,
      m.site_quat,
      m.cam_bodyid,
      m.cam_quat,
      m.cam_fovy,
      m.cam_resolution,
      m.cam_sensorsize,
      m.cam_intrinsic,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_reftype,
      m.sensor_refid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_pos_adr,
      d.time,
      d.qpos,
      d.xpos,
      d.xquat,
      d.xmat,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d.subtree_com,
      d.actuator_length,
      d.ten_length,
    ],
    outputs=[d.sensordata],
  )


@wp.func
def _velocimeter(
  # Model:
  body_rootid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  # Data in:
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  bodyid = site_bodyid[objid]
  pos = site_xpos_in[worldid, objid]
  rot = site_xmat_in[worldid, objid]
  cvel = cvel_in[worldid, bodyid]
  ang = wp.spatial_top(cvel)
  lin = wp.spatial_bottom(cvel)
  subtree_com = subtree_com_in[worldid, body_rootid[bodyid]]
  dif = pos - subtree_com
  return wp.transpose(rot) @ (lin - wp.cross(dif, ang))


@wp.func
def _gyro(
  # Model:
  site_bodyid: wp.array(dtype=int),
  # Data in:
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  bodyid = site_bodyid[objid]
  rot = site_xmat_in[worldid, objid]
  cvel = cvel_in[worldid, bodyid]
  ang = wp.spatial_top(cvel)
  return wp.transpose(rot) @ ang


@wp.func
def _joint_vel(jnt_dofadr: wp.array(dtype=int), qvel_in: wp.array2d(dtype=float), worldid: int, objid: int) -> float:
  return qvel_in[worldid, jnt_dofadr[objid]]


@wp.func
def _tendon_vel(ten_velocity_in: wp.array2d(dtype=float), worldid: int, objid: int) -> float:
  return ten_velocity_in[worldid, objid]


@wp.func
def _actuator_vel(actuator_velocity_in: wp.array2d(dtype=float), worldid: int, objid: int) -> float:
  return actuator_velocity_in[worldid, objid]


@wp.func
def _ball_ang_vel(jnt_dofadr: wp.array(dtype=int), qvel_in: wp.array2d(dtype=float), worldid: int, objid: int) -> wp.vec3:
  adr = jnt_dofadr[objid]
  return wp.vec3(qvel_in[worldid, adr + 0], qvel_in[worldid, adr + 1], qvel_in[worldid, adr + 2])


@wp.func
def _cvel_offset(
  # Model:
  body_rootid: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  cam_bodyid: wp.array(dtype=int),
  # Data in:
  xpos_in: wp.array2d(dtype=wp.vec3),
  xipos_in: wp.array2d(dtype=wp.vec3),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  cam_xpos_in: wp.array2d(dtype=wp.vec3),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  worldid: int,
  objtype: int,
  objid: int,
) -> Tuple[wp.spatial_vector, wp.vec3]:
  if objtype == int(ObjType.BODY.value):
    pos = xipos_in[worldid, objid]
    bodyid = objid
  elif objtype == int(ObjType.XBODY.value):
    pos = xpos_in[worldid, objid]
    bodyid = objid
  elif objtype == int(ObjType.GEOM.value):
    pos = geom_xpos_in[worldid, objid]
    bodyid = geom_bodyid[objid]
  elif objtype == int(ObjType.SITE.value):
    pos = site_xpos_in[worldid, objid]
    bodyid = site_bodyid[objid]
  elif objtype == int(ObjType.CAMERA.value):
    pos = cam_xpos_in[worldid, objid]
    bodyid = cam_bodyid[objid]
  else:  # UNKNOWN
    pos = wp.vec3(0.0)
    bodyid = 0

  return cvel_in[worldid, bodyid], pos - subtree_com_in[worldid, body_rootid[bodyid]]


@wp.func
def _frame_linvel(
  # Model:
  body_rootid: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  cam_bodyid: wp.array(dtype=int),
  # Data in:
  xpos_in: wp.array2d(dtype=wp.vec3),
  xmat_in: wp.array2d(dtype=wp.mat33),
  xipos_in: wp.array2d(dtype=wp.vec3),
  ximat_in: wp.array2d(dtype=wp.mat33),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  cam_xpos_in: wp.array2d(dtype=wp.vec3),
  cam_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  worldid: int,
  objid: int,
  objtype: int,
  refid: int,
  reftype: int,
) -> wp.vec3:
  if objtype == int(ObjType.BODY.value):
    xpos = xipos_in[worldid, objid]
  elif objtype == int(ObjType.XBODY.value):
    xpos = xpos_in[worldid, objid]
  elif objtype == int(ObjType.GEOM.value):
    xpos = geom_xpos_in[worldid, objid]
  elif objtype == int(ObjType.SITE.value):
    xpos = site_xpos_in[worldid, objid]
  elif objtype == int(ObjType.CAMERA.value):
    xpos = cam_xpos_in[worldid, objid]
  else:  # UNKNOWN
    xpos = wp.vec3(0.0)

  if reftype == int(ObjType.BODY.value):
    xposref = xipos_in[worldid, refid]
    xmatref = ximat_in[worldid, refid]
  elif reftype == int(ObjType.XBODY.value):
    xposref = xpos_in[worldid, refid]
    xmatref = xmat_in[worldid, refid]
  elif reftype == int(ObjType.GEOM.value):
    xposref = geom_xpos_in[worldid, refid]
    xmatref = geom_xmat_in[worldid, refid]
  elif reftype == int(ObjType.SITE.value):
    xposref = site_xpos_in[worldid, refid]
    xmatref = site_xmat_in[worldid, refid]
  elif reftype == int(ObjType.CAMERA.value):
    xposref = cam_xpos_in[worldid, refid]
    xmatref = cam_xmat_in[worldid, refid]
  else:  # UNKNOWN
    xposref = wp.vec3(0.0)
    xmatref = wp.identity(3, dtype=float)

  cvel, offset = _cvel_offset(
    body_rootid,
    geom_bodyid,
    site_bodyid,
    cam_bodyid,
    xpos_in,
    xipos_in,
    geom_xpos_in,
    site_xpos_in,
    cam_xpos_in,
    subtree_com_in,
    cvel_in,
    worldid,
    objtype,
    objid,
  )
  cvelref, offsetref = _cvel_offset(
    body_rootid,
    geom_bodyid,
    site_bodyid,
    cam_bodyid,
    xpos_in,
    xipos_in,
    geom_xpos_in,
    site_xpos_in,
    cam_xpos_in,
    subtree_com_in,
    cvel_in,
    worldid,
    reftype,
    refid,
  )
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
  # Model:
  body_rootid: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  cam_bodyid: wp.array(dtype=int),
  # Data in:
  xpos_in: wp.array2d(dtype=wp.vec3),
  xmat_in: wp.array2d(dtype=wp.mat33),
  xipos_in: wp.array2d(dtype=wp.vec3),
  ximat_in: wp.array2d(dtype=wp.mat33),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  cam_xpos_in: wp.array2d(dtype=wp.vec3),
  cam_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  worldid: int,
  objid: int,
  objtype: int,
  refid: int,
  reftype: int,
) -> wp.vec3:
  cvel, _ = _cvel_offset(
    body_rootid,
    geom_bodyid,
    site_bodyid,
    cam_bodyid,
    xpos_in,
    xipos_in,
    geom_xpos_in,
    site_xpos_in,
    cam_xpos_in,
    subtree_com_in,
    cvel_in,
    worldid,
    objtype,
    objid,
  )
  cangvel = wp.spatial_top(cvel)

  if refid > -1:
    if reftype == int(ObjType.BODY.value):
      xmatref = ximat_in[worldid, refid]
    elif reftype == int(ObjType.XBODY.value):
      xmatref = xmat_in[worldid, refid]
    elif reftype == int(ObjType.GEOM.value):
      xmatref = geom_xmat_in[worldid, refid]
    elif reftype == int(ObjType.SITE.value):
      xmatref = site_xmat_in[worldid, refid]
    elif reftype == int(ObjType.CAMERA.value):
      xmatref = cam_xmat_in[worldid, refid]
    else:  # UNKNOWN
      xmatref = wp.identity(3, dtype=float)

    cvelref, _ = _cvel_offset(
      body_rootid,
      geom_bodyid,
      site_bodyid,
      cam_bodyid,
      xpos_in,
      xipos_in,
      geom_xpos_in,
      site_xpos_in,
      cam_xpos_in,
      subtree_com_in,
      cvel_in,
      worldid,
      reftype,
      refid,
    )
    cangvelref = wp.spatial_top(cvelref)

    return wp.transpose(xmatref) @ (cangvel - cangvelref)
  else:
    return cangvel


@wp.func
def _subtree_linvel(subtree_linvel_in: wp.array2d(dtype=wp.vec3), worldid: int, objid: int) -> wp.vec3:
  return subtree_linvel_in[worldid, objid]


@wp.func
def _subtree_angmom(subtree_angmom_in: wp.array2d(dtype=wp.vec3), worldid: int, objid: int) -> wp.vec3:
  return subtree_angmom_in[worldid, objid]


@wp.kernel
def _sensor_vel(
  # Model:
  body_rootid: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  cam_bodyid: wp.array(dtype=int),
  sensor_type: wp.array(dtype=int),
  sensor_datatype: wp.array(dtype=int),
  sensor_objtype: wp.array(dtype=int),
  sensor_objid: wp.array(dtype=int),
  sensor_reftype: wp.array(dtype=int),
  sensor_refid: wp.array(dtype=int),
  sensor_adr: wp.array(dtype=int),
  sensor_cutoff: wp.array(dtype=float),
  sensor_vel_adr: wp.array(dtype=int),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  xpos_in: wp.array2d(dtype=wp.vec3),
  xmat_in: wp.array2d(dtype=wp.mat33),
  xipos_in: wp.array2d(dtype=wp.vec3),
  ximat_in: wp.array2d(dtype=wp.mat33),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  cam_xpos_in: wp.array2d(dtype=wp.vec3),
  cam_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  ten_velocity_in: wp.array2d(dtype=float),
  actuator_velocity_in: wp.array2d(dtype=float),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  subtree_linvel_in: wp.array2d(dtype=wp.vec3),
  subtree_angmom_in: wp.array2d(dtype=wp.vec3),
  # Data out:
  sensordata_out: wp.array2d(dtype=float),
):
  worldid, velid = wp.tid()
  sensorid = sensor_vel_adr[velid]
  sensortype = sensor_type[sensorid]
  objid = sensor_objid[sensorid]
  out = sensordata_out[worldid]

  if sensortype == int(SensorType.VELOCIMETER.value):
    vec3 = _velocimeter(body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cvel_in, worldid, objid)
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == int(SensorType.GYRO.value):
    vec3 = _gyro(site_bodyid, site_xmat_in, cvel_in, worldid, objid)
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == int(SensorType.JOINTVEL.value):
    val = _joint_vel(jnt_dofadr, qvel_in, worldid, objid)
    _write_scalar(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == int(SensorType.TENDONVEL.value):
    val = _tendon_vel(ten_velocity_in, worldid, objid)
    _write_scalar(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == int(SensorType.ACTUATORVEL.value):
    val = _actuator_vel(actuator_velocity_in, worldid, objid)
    _write_scalar(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == int(SensorType.BALLANGVEL.value):
    vec3 = _ball_ang_vel(jnt_dofadr, qvel_in, worldid, objid)
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == int(SensorType.FRAMELINVEL.value):
    objtype = sensor_objtype[sensorid]
    refid = sensor_refid[sensorid]
    reftype = sensor_reftype[sensorid]
    frame_linvel = _frame_linvel(
      body_rootid,
      geom_bodyid,
      site_bodyid,
      cam_bodyid,
      xpos_in,
      xmat_in,
      xipos_in,
      ximat_in,
      geom_xpos_in,
      geom_xmat_in,
      site_xpos_in,
      site_xmat_in,
      cam_xpos_in,
      cam_xmat_in,
      subtree_com_in,
      cvel_in,
      worldid,
      objid,
      objtype,
      refid,
      reftype,
    )
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, frame_linvel, out)
  elif sensortype == int(SensorType.FRAMEANGVEL.value):
    objtype = sensor_objtype[sensorid]
    refid = sensor_refid[sensorid]
    reftype = sensor_reftype[sensorid]
    frame_angvel = _frame_angvel(
      body_rootid,
      geom_bodyid,
      site_bodyid,
      cam_bodyid,
      xpos_in,
      xmat_in,
      xipos_in,
      ximat_in,
      geom_xpos_in,
      geom_xmat_in,
      site_xpos_in,
      site_xmat_in,
      cam_xpos_in,
      cam_xmat_in,
      subtree_com_in,
      cvel_in,
      worldid,
      objid,
      objtype,
      refid,
      reftype,
    )
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, frame_angvel, out)
  elif sensortype == int(SensorType.SUBTREELINVEL.value):
    vec3 = _subtree_linvel(subtree_linvel_in, worldid, objid)
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == int(SensorType.SUBTREEANGMOM.value):
    vec3 = _subtree_angmom(subtree_angmom_in, worldid, objid)
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)


@event_scope
def sensor_vel(m: Model, d: Data):
  """Compute velocity-dependent sensor values."""

  if (m.sensor_vel_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  if m.sensor_subtree_vel:
    smooth.subtree_vel(m, d)

  wp.launch(
    _sensor_vel,
    dim=(d.nworld, m.sensor_vel_adr.size),
    inputs=[
      m.body_rootid,
      m.jnt_dofadr,
      m.geom_bodyid,
      m.site_bodyid,
      m.cam_bodyid,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_reftype,
      m.sensor_refid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_vel_adr,
      d.qvel,
      d.xpos,
      d.xmat,
      d.xipos,
      d.ximat,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d.subtree_com,
      d.ten_velocity,
      d.actuator_velocity,
      d.cvel,
      d.subtree_linvel,
      d.subtree_angmom,
    ],
    outputs=[d.sensordata],
  )


@wp.func
def _accelerometer(
  # Model:
  body_rootid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  # Data in:
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  cacc_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  bodyid = site_bodyid[objid]
  rot = site_xmat_in[worldid, objid]
  rotT = wp.transpose(rot)
  cvel = cvel_in[worldid, bodyid]
  cvel_top = wp.spatial_top(cvel)
  cvel_bottom = wp.spatial_bottom(cvel)
  cacc = cacc_in[worldid, bodyid]
  cacc_top = wp.spatial_top(cacc)
  cacc_bottom = wp.spatial_bottom(cacc)
  dif = site_xpos_in[worldid, objid] - subtree_com_in[worldid, body_rootid[bodyid]]
  ang = rotT @ cvel_top
  lin = rotT @ (cvel_bottom - wp.cross(dif, cvel_top))
  acc = rotT @ (cacc_bottom - wp.cross(dif, cacc_top))
  correction = wp.cross(ang, lin)
  return acc + correction


@wp.func
def _force(
  # Model:
  site_bodyid: wp.array(dtype=int),
  # Data in:
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  cfrc_int_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  bodyid = site_bodyid[objid]
  cfrc_int = cfrc_int_in[worldid, bodyid]
  site_xmat = site_xmat_in[worldid, objid]
  return wp.transpose(site_xmat) @ wp.spatial_bottom(cfrc_int)


@wp.func
def _torque(
  # Model:
  body_rootid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  # Data in:
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cfrc_int_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  worldid: int,
  objid: int,
) -> wp.vec3:
  bodyid = site_bodyid[objid]
  cfrc_int = cfrc_int_in[worldid, bodyid]
  site_xmat = site_xmat_in[worldid, objid]
  dif = site_xpos_in[worldid, objid] - subtree_com_in[worldid, body_rootid[bodyid]]
  return wp.transpose(site_xmat) @ (wp.spatial_top(cfrc_int) - wp.cross(dif, wp.spatial_bottom(cfrc_int)))


@wp.func
def _actuator_force(actuator_force_in: wp.array2d(dtype=float), worldid: int, objid: int) -> float:
  return actuator_force_in[worldid, objid]


@wp.func
def _joint_actuator_force(
  # Model:
  jnt_dofadr: wp.array(dtype=int),
  # Data in:
  qfrc_actuator_in: wp.array2d(dtype=float),
  # In:
  worldid: int,
  objid: int,
) -> float:
  return qfrc_actuator_in[worldid, jnt_dofadr[objid]]


@wp.func
def _framelinacc(
  # Model:
  body_rootid: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  cam_bodyid: wp.array(dtype=int),
  # Data in:
  xpos_in: wp.array2d(dtype=wp.vec3),
  xipos_in: wp.array2d(dtype=wp.vec3),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  cam_xpos_in: wp.array2d(dtype=wp.vec3),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  cacc_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  worldid: int,
  objid: int,
  objtype: int,
) -> wp.vec3:
  if objtype == int(ObjType.BODY.value):
    bodyid = objid
    pos = xipos_in[worldid, objid]
  elif objtype == int(ObjType.XBODY.value):
    bodyid = objid
    pos = xpos_in[worldid, objid]
  elif objtype == int(ObjType.GEOM.value):
    bodyid = geom_bodyid[objid]
    pos = geom_xpos_in[worldid, objid]
  elif objtype == int(ObjType.SITE.value):
    bodyid = site_bodyid[objid]
    pos = site_xpos_in[worldid, objid]
  elif objtype == int(ObjType.CAMERA.value):
    bodyid = cam_bodyid[objid]
    pos = cam_xpos_in[worldid, objid]
  else:  # UNKNOWN
    bodyid = 0
    pos = wp.vec3(0.0)

  cacc = cacc_in[worldid, bodyid]
  cvel = cvel_in[worldid, bodyid]
  offset = pos - subtree_com_in[worldid, body_rootid[bodyid]]
  ang = wp.spatial_top(cvel)
  lin = wp.spatial_bottom(cvel) - wp.cross(offset, ang)
  acc = wp.spatial_bottom(cacc) - wp.cross(offset, wp.spatial_top(cacc))
  correction = wp.cross(ang, lin)

  return acc + correction


@wp.func
def _frameangacc(
  # Model:
  geom_bodyid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  cam_bodyid: wp.array(dtype=int),
  # Data in:
  cacc_in: wp.array2d(dtype=wp.spatial_vector),
  # In:
  worldid: int,
  objid: int,
  objtype: int,
) -> wp.vec3:
  if objtype == int(ObjType.BODY.value) or objtype == int(ObjType.XBODY.value):
    bodyid = objid
  elif objtype == int(ObjType.GEOM.value):
    bodyid = geom_bodyid[objid]
  elif objtype == int(ObjType.SITE.value):
    bodyid = site_bodyid[objid]
  elif objtype == int(ObjType.CAMERA.value):
    bodyid = cam_bodyid[objid]
  else:  # UNKNOWN
    bodyid = 0

  return wp.spatial_top(cacc_in[worldid, bodyid])


@wp.kernel
def _sensor_acc(
  # Model:
  body_rootid: wp.array(dtype=int),
  jnt_dofadr: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  site_bodyid: wp.array(dtype=int),
  cam_bodyid: wp.array(dtype=int),
  sensor_type: wp.array(dtype=int),
  sensor_datatype: wp.array(dtype=int),
  sensor_objtype: wp.array(dtype=int),
  sensor_objid: wp.array(dtype=int),
  sensor_adr: wp.array(dtype=int),
  sensor_cutoff: wp.array(dtype=float),
  sensor_acc_adr: wp.array(dtype=int),
  # Data in:
  xpos_in: wp.array2d(dtype=wp.vec3),
  xipos_in: wp.array2d(dtype=wp.vec3),
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xpos_in: wp.array2d(dtype=wp.vec3),
  site_xmat_in: wp.array2d(dtype=wp.mat33),
  cam_xpos_in: wp.array2d(dtype=wp.vec3),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cvel_in: wp.array2d(dtype=wp.spatial_vector),
  actuator_force_in: wp.array2d(dtype=float),
  qfrc_actuator_in: wp.array2d(dtype=float),
  cacc_in: wp.array2d(dtype=wp.spatial_vector),
  cfrc_int_in: wp.array2d(dtype=wp.spatial_vector),
  # Data out:
  sensordata_out: wp.array2d(dtype=float),
):
  worldid, accid = wp.tid()
  sensorid = sensor_acc_adr[accid]
  sensortype = sensor_type[sensorid]
  objid = sensor_objid[sensorid]
  out = sensordata_out[worldid]

  if sensortype == int(SensorType.ACCELEROMETER.value):
    vec3 = _accelerometer(
      body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cvel_in, cacc_in, worldid, objid
    )
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == int(SensorType.FORCE.value):
    vec3 = _force(site_bodyid, site_xmat_in, cfrc_int_in, worldid, objid)
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == int(SensorType.TORQUE.value):
    vec3 = _torque(body_rootid, site_bodyid, site_xpos_in, site_xmat_in, subtree_com_in, cfrc_int_in, worldid, objid)
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == int(SensorType.ACTUATORFRC.value):
    val = _actuator_force(actuator_force_in, worldid, objid)
    _write_scalar(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == int(SensorType.JOINTACTFRC.value):
    val = _joint_actuator_force(jnt_dofadr, qfrc_actuator_in, worldid, objid)
    _write_scalar(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, val, out)
  elif sensortype == int(SensorType.FRAMELINACC.value):
    objtype = sensor_objtype[sensorid]
    vec3 = _framelinacc(
      body_rootid,
      geom_bodyid,
      site_bodyid,
      cam_bodyid,
      xpos_in,
      xipos_in,
      geom_xpos_in,
      site_xpos_in,
      cam_xpos_in,
      subtree_com_in,
      cvel_in,
      cacc_in,
      worldid,
      objid,
      objtype,
    )
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)
  elif sensortype == int(SensorType.FRAMEANGACC.value):
    objtype = sensor_objtype[sensorid]
    vec3 = _frameangacc(
      geom_bodyid,
      site_bodyid,
      cam_bodyid,
      cacc_in,
      worldid,
      objid,
      objtype,
    )
    _write_vector(sensor_datatype, sensor_adr, sensor_cutoff, sensorid, 3, vec3, out)


@event_scope
def sensor_acc(m: Model, d: Data):
  """Compute acceleration-dependent sensor values."""

  if (m.sensor_acc_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  if m.sensor_rne_postconstraint:
    smooth.rne_postconstraint(m, d)

  wp.launch(
    _sensor_acc,
    dim=(d.nworld, m.sensor_acc_adr.size),
    inputs=[
      m.body_rootid,
      m.jnt_dofadr,
      m.geom_bodyid,
      m.site_bodyid,
      m.cam_bodyid,
      m.sensor_type,
      m.sensor_datatype,
      m.sensor_objtype,
      m.sensor_objid,
      m.sensor_adr,
      m.sensor_cutoff,
      m.sensor_acc_adr,
      d.xpos,
      d.xipos,
      d.geom_xpos,
      d.site_xpos,
      d.site_xmat,
      d.cam_xpos,
      d.subtree_com,
      d.cvel,
      d.actuator_force,
      d.qfrc_actuator,
      d.cacc,
      d.cfrc_int,
    ],
    outputs=[d.sensordata],
  )
