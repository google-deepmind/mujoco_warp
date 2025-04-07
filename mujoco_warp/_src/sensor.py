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

import mujoco

from .types import Data
from .types import DisableBit
from .types import Model
from .types import SensorType
from .warp_util import event_scope
from .warp_util import kernel


@wp.func
def _joint_pos(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.qpos[worldid, m.jnt_qposadr[objid]]


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

    if sensortype == int(SensorType.JOINTPOS.value):
      d.sensordata[worldid, adr] = _joint_pos(m, d, worldid, objid)

  if (m.sensor_pos_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  wp.launch(_sensor_pos, dim=(d.nworld, m.sensor_pos_adr.size), inputs=[m, d])


@wp.func
def _joint_vel(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.qvel[worldid, m.jnt_dofadr[objid]]


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

    if sensortype == int(SensorType.JOINTVEL.value):
      d.sensordata[worldid, adr] = _joint_vel(m, d, worldid, objid)

  if (m.sensor_vel_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  wp.launch(_sensor_vel, dim=(d.nworld, m.sensor_vel_adr.size), inputs=[m, d])


@wp.func
def _actuator_force(m: Model, d: Data, worldid: int, objid: int) -> wp.float32:
  return d.actuator_force[worldid, objid]


@wp.func
def _framelinacc(m: Model, d: Data, worldid: int, objid: int, objtype: int) -> wp.vec3:
  # TODO(team): replace with types.ObjType

  if objtype == int(mujoco.mjtObj.mjOBJ_BODY.value):
    bodyid = objid
    pos = d.xipos[worldid, objid]
  elif objtype == int(mujoco.mjtObj.mjOBJ_XBODY.value):
    bodyid = objid
    pos = d.xpos[worldid, objid]
  elif objtype == int(mujoco.mjtObj.mjOBJ_GEOM.value):
    bodyid = m.geom_bodyid[objid]
    pos = d.geom_xpos[worldid, objid]
  elif objtype == int(mujoco.mjtObj.mjOBJ_SITE.value):
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
  # TODO(team): replace with types.ObjType

  if objtype == int(mujoco.mjtObj.mjOBJ_BODY.value) or objtype == int(
    mujoco.mjtObj.mjOBJ_XBODY.value
  ):
    bodyid = objid
  elif objtype == int(mujoco.mjtObj.mjOBJ_GEOM.value):
    bodyid = m.geom_bodyid[objid]
  elif objtype == int(mujoco.mjtObj.mjOBJ_SITE.value):
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

    if sensortype == int(SensorType.ACTUATORFRC.value):
      d.sensordata[worldid, adr] = _actuator_force(m, d, worldid, objid)
    elif sensortype == int(SensorType.FRAMELINACC.value):
      objtype = m.sensor_objtype[accadr]
      framelinacc = _framelinacc(m, d, worldid, objid, objtype)
      d.sensordata[worldid, adr + 0] = framelinacc[0]
      d.sensordata[worldid, adr + 1] = framelinacc[1]
      d.sensordata[worldid, adr + 2] = framelinacc[2]
    elif sensortype == int(SensorType.FRAMEANGACC.value):
      objtype = m.sensor_objtype[accadr]
      frameangacc = _frameangacc(m, d, worldid, objid, objtype)
      d.sensordata[worldid, adr + 0] = frameangacc[0]
      d.sensordata[worldid, adr + 1] = frameangacc[1]
      d.sensordata[worldid, adr + 2] = frameangacc[2]

  if (m.sensor_acc_adr.size == 0) or (m.opt.disableflags & DisableBit.SENSOR):
    return

  wp.launch(_sensor_acc, dim=(d.nworld, m.sensor_acc_adr.size), inputs=[m, d])
