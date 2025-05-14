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
import dataclasses
import enum

import mujoco
import warp as wp

MJ_MINVAL = mujoco.mjMINVAL
MJ_MAXVAL = mujoco.mjMAXVAL
MJ_MINIMP = mujoco.mjMINIMP  # minimum constraint impedance
MJ_MAXIMP = mujoco.mjMAXIMP  # maximum constraint impedance
MJ_NREF = mujoco.mjNREF
MJ_NIMP = mujoco.mjNIMP


class CamLightType(enum.IntEnum):
  """Type of camera light.

  Members:
    FIXED: pos and rot fixed in body
    TRACK: pos tracks body, rot fixed in global
    TRACKCOM: pos tracks subtree com, rot fixed in body
    TARGETBODY: pos fixed in body, rot tracks target body
    TARGETBODYCOM: pos fixed in body, rot tracks target subtree com
  """

  FIXED = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
  TRACK = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
  TRACKCOM = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
  TARGETBODY = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
  TARGETBODYCOM = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM


class DataType(enum.IntFlag):
  """Sensor data types.

  Members:
    REAL: real values, no constraints
    POSITIVE: positive values, 0 or negative: inactive
  """

  REAL = mujoco.mjtDataType.mjDATATYPE_REAL
  POSITIVE = mujoco.mjtDataType.mjDATATYPE_POSITIVE
  # unsupported: AXIS, QUATERNION


class DisableBit(enum.IntFlag):
  """Disable default feature bitflags.

  Members:
    CONSTRAINT:   entire constraint solver
    EQUALITY:     equality constraints
    FRICTIONLOSS: joint and tendon frictionloss constraints
    LIMIT:        joint and tendon limit constraints
    CONTACT:      contact constraints
    PASSIVE:      passive forces
    GRAVITY:      gravitational forces
    CLAMPCTRL:    clamp control to specified range
    ACTUATION:    apply actuation forces
    REFSAFE:      integrator safety: make ref[0]>=2*timestep
    EULERDAMP:    implicit damping for Euler integration
    FILTERPARENT: disable collisions between parent and child bodies
    SENSOR: sensors
  """

  CONSTRAINT = mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
  EQUALITY = mujoco.mjtDisableBit.mjDSBL_EQUALITY
  FRICTIONLOSS = mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
  LIMIT = mujoco.mjtDisableBit.mjDSBL_LIMIT
  CONTACT = mujoco.mjtDisableBit.mjDSBL_CONTACT
  PASSIVE = mujoco.mjtDisableBit.mjDSBL_PASSIVE
  GRAVITY = mujoco.mjtDisableBit.mjDSBL_GRAVITY
  CLAMPCTRL = mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
  ACTUATION = mujoco.mjtDisableBit.mjDSBL_ACTUATION
  REFSAFE = mujoco.mjtDisableBit.mjDSBL_REFSAFE
  EULERDAMP = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
  FILTERPARENT = mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
  SENSOR = mujoco.mjtDisableBit.mjDSBL_SENSOR
  # unsupported: MIDPHASE, WARMSTART


class TrnType(enum.IntEnum):
  """Type of actuator transmission.

  Members:
    JOINT: force on joint
    JOINTINPARENT: force on joint, expressed in parent frame
    TENDON: force on tendon
  """

  JOINT = mujoco.mjtTrn.mjTRN_JOINT
  JOINTINPARENT = mujoco.mjtTrn.mjTRN_JOINTINPARENT
  TENDON = mujoco.mjtTrn.mjTRN_TENDON
  # unsupported: SITE, SLIDERCRANK, BODY


class DynType(enum.IntEnum):
  """Type of actuator dynamics.

  Members:
    NONE: no internal dynamics; ctrl specifies force
    INTEGRATOR: integrator: da/dt = u
    FILTER: linear filter: da/dt = (u-a) / tau
    FILTEREXACT: linear filter: da/dt = (u-a) / tau, with exact integration
  """

  NONE = mujoco.mjtDyn.mjDYN_NONE
  INTEGRATOR = mujoco.mjtDyn.mjDYN_INTEGRATOR
  FILTER = mujoco.mjtDyn.mjDYN_FILTER
  FILTEREXACT = mujoco.mjtDyn.mjDYN_FILTEREXACT
  # unsupported: MUSCLE, USER


class GainType(enum.IntEnum):
  """Type of actuator gain.

  Members:
    FIXED: fixed gain
    AFFINE: const + kp*length + kv*velocity
  """

  FIXED = mujoco.mjtGain.mjGAIN_FIXED
  AFFINE = mujoco.mjtGain.mjGAIN_AFFINE
  # unsupported: MUSCLE, USER


class BiasType(enum.IntEnum):
  """Type of actuator bias.

  Members:
    NONE: no bias
    AFFINE: const + kp*length + kv*velocity
  """

  NONE = mujoco.mjtBias.mjBIAS_NONE
  AFFINE = mujoco.mjtBias.mjBIAS_AFFINE
  # unsupported: MUSCLE, USER


class JointType(enum.IntEnum):
  """Type of degree of freedom.

  Members:
    FREE:  global position and orientation (quat)       (7,)
    BALL:  orientation (quat) relative to parent        (4,)
    SLIDE: sliding distance along body-fixed axis       (1,)
    HINGE: rotation angle (rad) around body-fixed axis  (1,)
  """

  FREE = mujoco.mjtJoint.mjJNT_FREE
  BALL = mujoco.mjtJoint.mjJNT_BALL
  SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
  HINGE = mujoco.mjtJoint.mjJNT_HINGE

  def dof_width(self) -> int:
    return {0: 6, 1: 3, 2: 1, 3: 1}[self.value]

  def qpos_width(self) -> int:
    return {0: 7, 1: 4, 2: 1, 3: 1}[self.value]


class ConeType(enum.IntEnum):
  """Type of friction cone.

  Members:
    PYRAMIDAL: pyramidal
    ELLIPTIC: elliptic
  """

  PYRAMIDAL = mujoco.mjtCone.mjCONE_PYRAMIDAL
  ELLIPTIC = mujoco.mjtCone.mjCONE_ELLIPTIC


class IntegratorType(enum.IntEnum):
  """Integrator mode.

  Members:
    EULER: semi-implicit Euler
    RK4: 4th-order Runge Kutta
    IMPLICITFAST: implicit in velocity, no rne derivative
  """

  EULER = mujoco.mjtIntegrator.mjINT_EULER
  RK4 = mujoco.mjtIntegrator.mjINT_RK4
  IMPLICITFAST = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  # unsupported: IMPLICIT


class GeomType(enum.IntEnum):
  """Type of geometry.

  Members:
    PLANE: plane
    SPHERE: sphere
    CAPSULE: capsule
    ELLIPSOID: ellipsoid
    CYLINDER: cylinder
    BOX: box
    MESH: mesh
  """

  PLANE = mujoco.mjtGeom.mjGEOM_PLANE
  SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
  CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
  ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
  CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
  BOX = mujoco.mjtGeom.mjGEOM_BOX
  MESH = mujoco.mjtGeom.mjGEOM_MESH
  # unsupported: HFIELD,
  # NGEOMTYPES, ARROW*, LINE, SKIN, LABEL, NONE


class SolverType(enum.IntEnum):
  """Constraint solver algorithm.

  Members:
    CG: Conjugate gradient (primal)
    NEWTON: Newton (primal)
  """

  CG = mujoco.mjtSolver.mjSOL_CG
  NEWTON = mujoco.mjtSolver.mjSOL_NEWTON
  # unsupported: PGS


class SensorType(enum.IntEnum):
  """Type of sensor.

  Members:
    CAMPROJECTION: camera projection
    JOINTPOS: joint position
    TENDONPOS: scalar tendon position
    ACTUATORPOS: actuator position
    BALLQUAT: ball joint orientation
    FRAMEPOS: frame position
    FRAMEXAXIS: frame x-axis
    FRAMEYAXIS: frame y-axis
    FRAMEZAXIS: frame z-axis
    FRAMEQUAT: frame orientation, represented as quaternion
    SUBTREECOM: subtree center of mass
    CLOCK: simulation time
    VELOCIMETER: 3D linear velocity, in local frame
    GYRO: 3D angular velocity, in local frame
    JOINTVEL: joint velocity
    TENDONVEL: scalar tendon velocity
    ACTUATORVEL: actuator velocity
    BALLANGVEL: ball joint angular velocity
    FRAMELINVEL: 3D linear velocity
    FRAMEANGVEL: 3D angular velocity
    SUBTREELINVEL: subtree linear velocity
    SUBTREEANGMOM: subtree angular momentum
    ACCELEROMETER: accelerometer
    FORCE: force
    TORQUE: torque
    ACTUATORFRC: scalar actuator force
    JOINTACTFRC: scalar actuator force, measured at the joint
    FRAMELINACC: 3D linear acceleration
    FRAMEANGACC: 3D angular acceleration
  """

  CAMPROJECTION = mujoco.mjtSensor.mjSENS_CAMPROJECTION
  JOINTPOS = mujoco.mjtSensor.mjSENS_JOINTPOS
  TENDONPOS = mujoco.mjtSensor.mjSENS_TENDONPOS
  ACTUATORPOS = mujoco.mjtSensor.mjSENS_ACTUATORPOS
  BALLQUAT = mujoco.mjtSensor.mjSENS_BALLQUAT
  FRAMEPOS = mujoco.mjtSensor.mjSENS_FRAMEPOS
  FRAMEXAXIS = mujoco.mjtSensor.mjSENS_FRAMEXAXIS
  FRAMEYAXIS = mujoco.mjtSensor.mjSENS_FRAMEYAXIS
  FRAMEZAXIS = mujoco.mjtSensor.mjSENS_FRAMEZAXIS
  FRAMEQUAT = mujoco.mjtSensor.mjSENS_FRAMEQUAT
  SUBTREECOM = mujoco.mjtSensor.mjSENS_SUBTREECOM
  CLOCK = mujoco.mjtSensor.mjSENS_CLOCK
  VELOCIMETER = mujoco.mjtSensor.mjSENS_VELOCIMETER
  GYRO = mujoco.mjtSensor.mjSENS_GYRO
  JOINTVEL = mujoco.mjtSensor.mjSENS_JOINTVEL
  TENDONVEL = mujoco.mjtSensor.mjSENS_TENDONVEL
  ACTUATORVEL = mujoco.mjtSensor.mjSENS_ACTUATORVEL
  BALLANGVEL = mujoco.mjtSensor.mjSENS_BALLANGVEL
  FRAMELINVEL = mujoco.mjtSensor.mjSENS_FRAMELINVEL
  FRAMEANGVEL = mujoco.mjtSensor.mjSENS_FRAMEANGVEL
  SUBTREELINVEL = mujoco.mjtSensor.mjSENS_SUBTREELINVEL
  SUBTREEANGMOM = mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
  ACCELEROMETER = mujoco.mjtSensor.mjSENS_ACCELEROMETER
  FORCE = mujoco.mjtSensor.mjSENS_FORCE
  TORQUE = mujoco.mjtSensor.mjSENS_TORQUE
  ACTUATORFRC = mujoco.mjtSensor.mjSENS_ACTUATORFRC
  JOINTACTFRC = mujoco.mjtSensor.mjSENS_JOINTACTFRC
  FRAMELINACC = mujoco.mjtSensor.mjSENS_FRAMELINACC
  FRAMEANGACC = mujoco.mjtSensor.mjSENS_FRAMEANGACC


class ObjType(enum.IntEnum):
  """Type of object.

  Members:
    UNKNOWN: unknown object type
    BODY: body
    XBODY: body, used to access regular frame instead of i-frame
    GEOM: geom
    SITE: site
    CAMERA: camera
  """

  UNKNOWN = mujoco.mjtObj.mjOBJ_UNKNOWN
  BODY = mujoco.mjtObj.mjOBJ_BODY
  XBODY = mujoco.mjtObj.mjOBJ_XBODY
  GEOM = mujoco.mjtObj.mjOBJ_GEOM
  SITE = mujoco.mjtObj.mjOBJ_SITE
  CAMERA = mujoco.mjtObj.mjOBJ_CAMERA


class EqType(enum.IntEnum):
  """Type of equality constraint.

  Members:
    CONNECT: connect two bodies at a point (ball joint)
    JOINT: couple the values of two scalar joints with cubic
    WELD: fix relative position and orientation of two bodies
  """

  CONNECT = mujoco.mjtEq.mjEQ_CONNECT
  WELD = mujoco.mjtEq.mjEQ_WELD
  JOINT = mujoco.mjtEq.mjEQ_JOINT
  TENDON = mujoco.mjtEq.mjEQ_TENDON
  # unsupported: FLEX, DISTANCE


class WrapType(enum.IntEnum):
  """Type of tendon wrapping object.

  Members:
    JOINT: constant moment arm
    SITE: pass through site
  """

  JOINT = mujoco.mjtWrap.mjWRAP_JOINT
  SITE = mujoco.mjtWrap.mjWRAP_SITE
  # unsupported: PULLEY, SPHERE, CYLINDER


class vec5f(wp.types.vector(length=5, dtype=float)):
  pass


class vec6f(wp.types.vector(length=6, dtype=float)):
  pass


class vec10f(wp.types.vector(length=10, dtype=float)):
  pass


class vec11f(wp.types.vector(length=11, dtype=float)):
  pass


vec5 = vec5f
vec6 = vec6f
vec10 = vec10f
vec11 = vec11f
array2df = wp.array2d(dtype=float)
array3df = wp.array3d(dtype=float)


@dataclasses.dataclass
class Option:
  """Physics options.

  Attributes:
    timestep: simulation timestep
    impratio: ratio of friction-to-normal contact impedance
    tolerance: main solver tolerance
    ls_tolerance: CG/Newton linesearch tolerance
    gravity: gravitational acceleration
    integrator: integration mode (mjtIntegrator)
    cone: type of friction cone (mjtCone)
    solver: solver algorithm (mjtSolver)
    iterations: number of main solver iterations
    ls_iterations: maximum number of CG/Newton linesearch iterations
    disableflags: bit flags for disabling standard features
    is_sparse: whether to use sparse representations
    gjk_iterations: number of Gjk iterations in the convex narrowphase
    epa_iterations: number of Epa iterations in the convex narrowphase
    epa_exact_neg_distance: flag for enabling the distance calculation for non-intersecting case in the convex narrowphase
    depth_extension: distance for which the closest point is not calculated for non-intersecting case in the convex narrowphase
    ls_parallel: evaluate engine solver step sizes in parallel
    wind: wind (for lift, drag, and viscosity)
    density: density of medium
    viscosity: viscosity of medium
  """

  timestep: float
  impratio: float
  tolerance: float
  ls_tolerance: float
  gravity: wp.vec3
  integrator: int
  cone: int
  solver: int
  iterations: int
  ls_iterations: int
  disableflags: int
  is_sparse: bool
  gjk_iterations: int  # warp only
  epa_iterations: int  # warp only
  epa_exact_neg_distance: bool  # warp only
  depth_extension: float  # warp only
  ls_parallel: bool
  wind: wp.vec3
  density: float
  viscosity: float


@dataclasses.dataclass
class Statistic:
  """Model statistics (in qpos0).

  Attributes:
    meaninertia: mean diagonal inertia
  """

  meaninertia: float


@dataclasses.dataclass
class Constraint:
  """Constraint data.

  Attributes:
    worldid: world id                                 (njmax,)
    id: id of object of specific type                 (njmax,)
    J: constraint Jacobian                            (njmax, nv)
    pos: constraint position (equality, contact)      (njmax,)
    margin: inclusion margin (contact)                (njmax,)
    D: constraint mass                                (njmax,)
    aref: reference pseudo-acceleration               (njmax,)
    frictionloss: frictionloss (friction)             (njmax,)
    force: constraint force in constraint space       (njmax,)
    Jaref: Jac*qacc - aref                            (njmax,)
    Ma: M*qacc                                        (nworld, nv)
    grad: gradient of master cost                     (nworld, nv)
    grad_dot: dot(grad, grad)                         (nworld,)
    Mgrad: M / grad                                   (nworld, nv)
    search: linesearch vector                         (nworld, nv)
    search_dot: dot(search, search)                   (nworld,)
    gauss: gauss Cost                                 (nworld,)
    cost: constraint + Gauss cost                     (nworld,)
    prev_cost: cost from previous iter                (nworld,)
    solver_niter: number of solver iterations         (nworld,)
    active: active (quadratic) constraints            (njmax,)
    gtol: linesearch termination tolerance            (nworld,)
    mv: qM @ search                                   (nworld, nv)
    jv: efc_J @ search                                (njmax,)
    quad: quadratic cost coefficients                 (njmax, 3)
    quad_gauss: quadratic cost gauss coefficients     (nworld, 3)
    h: cone hessian                                   (nworld, nv, nv)
    alpha: line search step size                      (nworld,)
    prev_grad: previous grad                          (nworld, nv)
    prev_Mgrad: previous Mgrad                        (nworld, nv)
    beta: polak-ribiere beta                          (nworld,)
    beta_num: numerator of beta                       (nworld,)
    beta_den: denominator of beta                     (nworld,)
    done: solver done                                 (nworld,)
    ls_done: linesearch done                          (nworld,)
    p0: initial point                                 (nworld, 3)
    lo: low point bounding the line search interval   (nworld, 3)
    lo_alpha: alpha for low point                     (nworld,)
    hi: high point bounding the line search interval  (nworld, 3)
    hi_alpha: alpha for high point                    (nworld,)
    lo_next: next low point                           (nworld, 3)
    lo_next_alpha: alpha for next low point           (nworld,)
    hi_next: next high point                          (nworld, 3)
    hi_next_alpha: alpha for next high point          (nworld,)
    mid: loss at mid_alpha                            (nworld, 3)
    mid_alpha: midpoint between lo_alpha and hi_alpha (nworld,)
    cost_candidate: costs associated with step sizes  (nworld, nlsp)
    quad_total_candidate: quad_total for step sizes   (nworld, nlsp, 3)
    u: friction cone (normal and tangents)            (nconmax, 6)
    uu: elliptic cone variables                       (nconmax,)
    uv: elliptic cone variables                       (nconmax,)
    vv: elliptic cone variables                       (nconmax,)
    condim: if contact: condim, else: -1              (njmax,)
  """

  worldid: wp.array(dtype=int)
  id: wp.array(dtype=int)
  J: wp.array2d(dtype=float)
  pos: wp.array(dtype=float)
  margin: wp.array(dtype=float)
  D: wp.array(dtype=float)
  aref: wp.array(dtype=float)
  frictionloss: wp.array(dtype=float)
  force: wp.array(dtype=float)
  Jaref: wp.array(dtype=float)
  Ma: wp.array2d(dtype=float)
  grad: wp.array2d(dtype=float)
  grad_dot: wp.array(dtype=float)
  Mgrad: wp.array2d(dtype=float)
  search: wp.array2d(dtype=float)
  search_dot: wp.array(dtype=float)
  gauss: wp.array(dtype=float)
  cost: wp.array(dtype=float)
  prev_cost: wp.array(dtype=float)
  solver_niter: wp.array(dtype=int)
  active: wp.array(dtype=bool)
  gtol: wp.array(dtype=float)
  mv: wp.array2d(dtype=float)
  jv: wp.array(dtype=float)
  quad: wp.array(dtype=wp.vec3)
  quad_gauss: wp.array(dtype=wp.vec3)
  h: wp.array3d(dtype=float)
  alpha: wp.array(dtype=float)
  prev_grad: wp.array2d(dtype=float)
  prev_Mgrad: wp.array2d(dtype=float)
  beta: wp.array(dtype=float)
  beta_num: wp.array(dtype=float)
  beta_den: wp.array(dtype=float)
  done: wp.array(dtype=bool)
  # linesearch
  ls_done: wp.array(dtype=bool)
  p0: wp.array(dtype=wp.vec3)
  lo: wp.array(dtype=wp.vec3)
  lo_alpha: wp.array(dtype=float)
  hi: wp.array(dtype=wp.vec3)
  hi_alpha: wp.array(dtype=float)
  lo_next: wp.array(dtype=wp.vec3)
  lo_next_alpha: wp.array(dtype=float)
  hi_next: wp.array(dtype=wp.vec3)
  hi_next_alpha: wp.array(dtype=float)
  mid: wp.array(dtype=wp.vec3)
  mid_alpha: wp.array(dtype=float)
  cost_candidate: wp.array2d(dtype=float)
  quad_total_candidate: wp.array2d(dtype=wp.vec3)
  # elliptic cone
  u: wp.array2d(dtype=float)
  uu: wp.array(dtype=float)
  uv: wp.array(dtype=float)
  vv: wp.array(dtype=float)
  condim: wp.array(dtype=int)


@dataclasses.dataclass
class TileSet:
  """Tiling configuration for decomposible block diagonal matrix.

  For non-square, non-block-diagonal tiles, use two tilesets.

  Attributes:
    adr: address of each tile in the set
    size: size of all the tiles in this set
  """

  adr: wp.array(dtype=int)
  size: int


# TODO(team): make Model/Data fields sort order match mujoco


@dataclasses.dataclass
class Model:
  """Model definition and parameters.

  Attributes:
    nq: number of generalized coordinates = dim              ()
    nv: number of degrees of freedom = dim                   ()
    nu: number of actuators/controls = dim                   ()
    na: number of activation states = dim                    ()
    nbody: number of bodies                                  ()
    njnt: number of joints                                   ()
    ngeom: number of geoms                                   ()
    nsite: number of sites                                   ()
    ncam: number of cameras                                  ()
    nlight: number of lights                                 ()
    nexclude: number of excluded geom pairs                  ()
    neq: number of equality constraints                      ()
    nmocap: number of mocap bodies                           ()
    ngravcomp: number of bodies with nonzero gravcomp        ()
    nM: number of non-zeros in sparse inertia matrix         ()
    ntendon: number of tendons                               ()
    nwrap: number of wrap objects in all tendon paths        ()
    nsensor: number of sensors                               ()
    nsensordata: number of elements in sensor data vector    ()
    nmeshvert: number of vertices for all meshes             ()
    nmeshface: number of faces for all meshes                ()
    nlsp: number of step sizes for parallel linsearch        ()
    npair: number of predefined geom pairs                   ()
    opt: physics options
    stat: model statistics
    qpos0: qpos values at default pose                       (nworld, nq)
    qpos_spring: reference pose for springs                  (nworld, nq)
    qM_fullm_i: sparse mass matrix addressing
    qM_fullm_j: sparse mass matrix addressing
    qM_mulm_i: sparse mass matrix addressing
    qM_mulm_j: sparse mass matrix addressing
    qM_madr_ij: sparse mass matrix addressing
    qLD_update_tree: dof tree ordering for qLD updates
    qLD_update_treeadr: index of each dof tree level
    M_rownnz: number of non-zeros in each row of qM             (nv,)
    M_rowadr: index of each row in qM                           (nv,)
    M_colind: column indices of non-zeros in qM                 (nM,)
    mapM2M: index mapping from M (legacy) to M (CSR)            (nM)
    qM_tiles: tiling configuration
    body_tree: list of body ids by tree level
    body_parentid: id of body's parent                       (nbody,)
    body_rootid: id of root above body                       (nbody,)
    body_weldid: id of body that this body is welded to      (nbody,)
    body_mocapid: id of mocap data; -1: none                 (nbody,)
    body_jntnum: number of joints for this body              (nbody,)
    body_jntadr: start addr of joints; -1: no joints         (nbody,)
    body_dofnum: number of motion degrees of freedom         (nbody,)
    body_dofadr: start addr of dofs; -1: no dofs             (nbody,)
    body_geomnum: number of geoms                            (nbody,)
    body_geomadr: start addr of geoms; -1: no geoms          (nbody,)
    body_pos: position offset rel. to parent body            (nworld, nbody, 3)
    body_quat: orientation offset rel. to parent body        (nworld, nbody, 4)
    body_ipos: local position of center of mass              (nworld, nbody, 3)
    body_iquat: local orientation of inertia ellipsoid       (nworld, nbody, 4)
    body_mass: mass                                          (nworld, nbody,)
    body_subtreemass: mass of subtree starting at this body  (nworld, nbody,)
    subtree_mass: mass of subtree                            (nworld, nbody,)
    body_inertia: diagonal inertia in ipos/iquat frame       (nworld, nbody, 3)
    body_invweight0: mean inv inert in qpos0 (trn, rot)      (nworld, nbody, 2)
    body_contype: OR over all geom contypes                  (nbody,)
    body_conaffinity: OR over all geom conaffinities         (nbody,)
    body_gravcomp: antigravity force, units of body weight   (nworld, nbody)
    jnt_type: type of joint (mjtJoint)                       (njnt,)
    jnt_qposadr: start addr in 'qpos' for joint's data       (njnt,)
    jnt_dofadr: start addr in 'qvel' for joint's data        (njnt,)
    jnt_bodyid: id of joint's body                           (njnt,)
    jnt_limited: does joint have limits                      (njnt,)
    jnt_actfrclimited: does joint have actuator force limits (njnt,)
    jnt_solref: constraint solver reference: limit           (nworld, njnt, mjNREF)
    jnt_solimp: constraint solver impedance: limit           (nworld, njnt, mjNIMP)
    jnt_pos: local anchor position                           (nworld, njnt, 3)
    jnt_axis: local joint axis                               (nworld, njnt, 3)
    jnt_stiffness: stiffness coefficient                     (nworld, njnt)
    jnt_range: joint limits                                  (nworld, njnt, 2)
    jnt_actfrcrange: range of total actuator force           (nworld, njnt, 2)
    jnt_margin: min distance for limit detection             (nworld, njnt)
    jnt_limited_slide_hinge_adr: limited/slide/hinge jntadr
    jnt_limited_ball_adr: limited/ball jntadr
    jnt_actgravcomp: is gravcomp force applied via actuators (njnt,)
    dof_bodyid: id of dof's body                             (nv,)
    dof_jntid: id of dof's joint                             (nv,)
    dof_parentid: id of dof's parent; -1: none               (nv,)
    dof_Madr: dof address in M-diagonal                      (nv,)
    dof_armature: dof armature inertia/mass                  (nworld, nv)
    dof_damping: damping coefficient                         (nworld, nv)
    dof_invweight0: diag. inverse inertia in qpos0           (nworld, nv)
    dof_frictionloss: dof friction loss                      (nworld, nv)
    dof_solimp: constraint solver impedance: frictionloss    (nworld, nv, NIMP)
    dof_solref: constraint solver reference: frictionloss    (nworld, nv, NREF)
    dof_tri_row: np.tril_indices                             (mjm.nv)[0]
    dof_tri_col: np.tril_indices                             (mjm.nv)[1]
    geom_type: geometric type (mjtGeom)                      (ngeom,)
    geom_contype: geom contact type                          (ngeom,)
    geom_conaffinity: geom contact affinity                  (ngeom,)
    geom_condim: contact dimensionality (1, 3, 4, 6)         (ngeom,)
    geom_bodyid: id of geom's body                           (ngeom,)
    geom_dataid: id of geom's mesh/hfield; -1: none          (ngeom,)
    geom_group: geom group inclusion/exclusion mask          (ngeom,)
    geom_matid: material id for rendering                    (nworld, ngeom,)
    geom_priority: geom contact priority                     (ngeom,)
    geom_solmix: mixing coef for solref/imp in geom pair     (nworld, ngeom,)
    geom_solref: constraint solver reference: contact        (nworld, ngeom, mjNREF)
    geom_solimp: constraint solver impedance: contact        (nworld, ngeom, mjNIMP)
    geom_size: geom-specific size parameters                 (ngeom, 3)
    geom_aabb: bounding box, (center, size)                  (ngeom, 6)
    geom_rbound: radius of bounding sphere                   (nworld, ngeom,)
    geom_pos: local position offset rel. to body             (nworld, ngeom, 3)
    geom_quat: local orientation offset rel. to body         (nworld, ngeom, 4)
    geom_friction: friction for (slide, spin, roll)          (nworld, ngeom, 3)
    geom_margin: detect contact if dist<margin               (nworld, ngeom,)
    geom_gap: include in solver if dist<margin-gap           (nworld, ngeom,)
    geom_rgba: rgba when material is omitted                 (nworld, ngeom, 4)
    site_bodyid: id of site's body                           (nsite,)
    site_pos: local position offset rel. to body             (nworld, nsite, 3)
    site_quat: local orientation offset rel. to body         (nworld, nsite, 4)
    cam_mode: camera tracking mode (mjtCamLight)             (ncam,)
    cam_bodyid: id of camera's body                          (ncam,)
    cam_targetbodyid: id of targeted body; -1: none          (ncam,)
    cam_pos: position rel. to body frame                     (nworld, ncam, 3)
    cam_quat: orientation rel. to body frame                 (nworld, ncam, 4)
    cam_poscom0: global position rel. to sub-com in qpos0    (nworld, ncam, 3)
    cam_pos0: Cartesian camera position                      (nworld, ncam, 3)
    cam_fovy: y field-of-view (ortho ? len : deg)            (ncam,)
    cam_resolution: resolution: pixels [width, height]       (ncam, 2)
    cam_sensorsize: sensor size: length [width, height]      (ncam, 2)
    cam_intrinsic: [focal length; principal point]           (ncam, 4)
    light_mode: light tracking mode (mjtCamLight)            (nlight,)
    light_bodyid: id of light's body                         (nlight,)
    light_targetbodyid: id of targeted body; -1: none        (nlight,)
    light_pos: position rel. to body frame                   (nworld, nlight, 3)
    light_dir: direction rel. to body frame                  (nworld, nlight, 3)
    light_poscom0: global position rel. to sub-com in qpos0  (nworld, nlight, 3)
    light_pos0: global position rel. to body in qpos0        (nworld, nlight, 3)
    mesh_vertadr: first vertex address                       (nmesh,)
    mesh_vertnum: number of vertices                         (nmesh,)
    mesh_vert: vertex positions for all meshes               (nmeshvert, 3)
    mesh_faceadr: first face address                         (nmesh,)
    mesh_face: face indices for all meshes                   (nface, 3)
    eq_type: constraint type (mjtEq)                         (neq,)
    eq_obj1id: id of object 1                                (neq,)
    eq_obj2id: id of object 2                                (neq,)
    eq_objtype: type of both objects (mjtObj)                (neq,)
    eq_active0: initial enable/disable constraint state      (neq,)
    eq_solref: constraint solver reference                   (nworld, neq, mjNREF)
    eq_solimp: constraint solver impedance                   (nworld, neq, mjNIMP)
    eq_data: numeric data for constraint                     (nworld, neq, mjNEQDATA)
    eq_connect_adr: eq_* addresses of type `CONNECT`
    eq_wld_adr: eq_* addresses of type `WELD`
    eq_jnt_adr: eq_* addresses of type `JOINT`
    eq_ten_adr: eq_* addresses of type `TENDON`              (<=neq,)
    actuator_moment_tiles_nv: tiling configuration
    actuator_moment_tiles_nu: tiling configuration
    actuator_affine_bias_gain: affine bias/gain present
    actuator_trntype: transmission type (mjtTrn)             (nu,)
    actuator_dyntype: dynamics type (mjtDyn)                 (nu,)
    actuator_gaintype: gain type (mjtGain)                   (nu,)
    actuator_biastype: bias type (mjtBias)                   (nu,)
    actuator_trnid: transmission id: joint, tendon, site     (nu, 2)
    actuator_actadr: first activation address; -1: stateless (nu,)
    actuator_actnum: number of activation variables          (nu,)
    actuator_ctrllimited: is control limited                 (nu,)
    actuator_forcelimited: is force limited                  (nu,)
    actuator_actlimited: is activation limited               (nu,)
    actuator_dynprm: dynamics parameters                     (nworld, nu, mjNDYN)
    actuator_gainprm: gain parameters                        (nworld, nu, mjNGAIN)
    actuator_biasprm: bias parameters                        (nworld, nu, mjNBIAS)
    actuator_ctrlrange: range of controls                    (nworld, nu, 2)
    actuator_forcerange: range of forces                     (nworld, nu, 2)
    actuator_actrange: range of activations                  (nworld, nu, 2)
    actuator_gear: scale length and transmitted force        (nworld, nu, 6)
    nxn_geom_pair: valid collision pair geom ids             (<= ngeom * (ngeom - 1) // 2,)
    nxn_pairid: predefined pair id, -1 if not predefined     (<= ngeom * (ngeom - 1) // 2,)
    pair_dim: contact dimensionality                         (npair,)
    pair_geom1: id of geom1                                  (npair,)
    pair_geom2: id of geom2                                  (npair,)
    pair_solref: solver reference: contact normal            (nworld, npair, mjNREF)
    pair_solreffriction: solver reference: contact friction  (nworld, npair, mjNREF)
    pair_solimp: solver impedance: contact                   (nworld, npair, mjNIMP)
    pair_margin: detect contact if dist<margin               (nworld, npair,)
    pair_gap: include in solver if dist<margin-gap           (nworld, npair,)
    pair_friction: tangent1, 2, spin, roll1, 2               (nworld, npair, 5)
    exclude_signature: body1 << 16 + body2                   (nexclude,)
    condim_max: maximum condim for geoms
    tendon_adr: address of first object in tendon's path     (ntendon,)
    tendon_num: number of objects in tendon's path           (ntendon,)
    tendon_limited: does tendon have length limits           (ntendon,)
    tendon_limited_adr: addresses for limited tendons        (<=ntendon,)
    tendon_solref_lim: constraint solver reference: limit    (nworld, ntendon, mjNREF)
    tendon_solimp_lim: constraint solver impedance: limit    (nworld, ntendon, mjNIMP)
    tendon_range: tendon length limits                       (nworld, ntendon, 2)
    tendon_margin: min distance for limit detection          (nworld, ntendon,)
    tendon_length0: tendon length in qpos0                   (nworld, ntendon,)
    tendon_invweight0: inv. weight in qpos0                  (nworld, ntendon,)
    wrap_objid: object id: geom, site, joint                 (nwrap,)
    wrap_prm: divisor, joint coef, or site id                (nwrap,)
    wrap_type: wrap object type (mjtWrap)                    (nwrap,)
    tendon_jnt_adr: joint tendon address                     (<=nwrap,)
    tendon_site_adr: site tendon address                     (<=nwrap,)
    tendon_site_pair_adr: site pair tendon address           (<=nwrap,)
    ten_wrapadr_site: wrap object starting address for sites (ntendon,)
    ten_wrapnum_site: number of site wrap objects per tendon (ntendon,)
    wrap_jnt_adr: addresses for joint tendon wrap object     (<=nwrap,)
    wrap_site_adr: addresses for site tendon wrap object     (<=nwrap,)
    wrap_site_pair_adr: first address for site wrap pair     (<=nwrap,)
    sensor_type: sensor type (mjtSensor)                     (nsensor,)
    sensor_datatype: numeric data type (mjtDataType)         (nsensor,)
    sensor_objtype: type of sensorized object (mjtObj)       (nsensor,)
    sensor_objid: id of sensorized object                    (nsensor,)
    sensor_reftype: type of reference frame (mjtObj)         (nsensor,)
    sensor_refid: id of reference frame; -1: global frame    (nsensor,)
    sensor_dim: number of scalar outputs                     (nsensor,)
    sensor_adr: address in sensor array                      (nsensor,)
    sensor_cutoff: cutoff for real and positive; 0: ignore   (nsensor,)
    sensor_pos_adr: addresses for position sensors           (<=nsensor,)
    sensor_vel_adr: addresses for velocity sensors           (<=nsensor,)
    sensor_acc_adr: addresses for acceleration sensors       (<=nsensor,)
    sensor_subtree_vel: evaluate subtree_vel
    sensor_rne_postconstraint: evaluate rne_postconstraint
    mocap_bodyid: id of body for mocap                       (nmocap,)
    mat_rgba: rgba                                           (nworld, nmat, 4)
  """

  nq: int
  nv: int
  nu: int
  na: int
  nbody: int
  njnt: int
  ngeom: int
  nsite: int
  ncam: int
  nlight: int
  nexclude: int
  neq: int
  nmocap: int
  ngravcomp: int
  nM: int
  ntendon: int
  nwrap: int
  nsensor: int
  nsensordata: int
  nmeshvert: int
  nmeshface: int
  nlsp: int  # warp only
  npair: int
  opt: Option
  stat: Statistic
  qpos0: wp.array2d(dtype=float)
  qpos_spring: wp.array2d(dtype=float)
  qM_fullm_i: wp.array(dtype=int)  # warp only
  qM_fullm_j: wp.array(dtype=int)  # warp only
  qM_mulm_i: wp.array(dtype=int)  # warp only
  qM_mulm_j: wp.array(dtype=int)  # warp only
  qM_madr_ij: wp.array(dtype=int)  # warp only
  qLD_updates: tuple[wp.array(dtype=wp.vec3i), ...]  # warp only
  M_rownnz: wp.array(dtype=int)
  M_rowadr: wp.array(dtype=int)
  M_colind: wp.array(dtype=int)
  mapM2M: wp.array(dtype=int)
  qM_tiles: tuple[TileSet, ...]
  body_tree: tuple[wp.array(dtype=int), ...]
  body_parentid: wp.array(dtype=int)
  body_rootid: wp.array(dtype=int)
  body_weldid: wp.array(dtype=int)
  body_mocapid: wp.array(dtype=int)
  body_jntnum: wp.array(dtype=int)
  body_jntadr: wp.array(dtype=int)
  body_dofnum: wp.array(dtype=int)
  body_dofadr: wp.array(dtype=int)
  body_geomnum: wp.array(dtype=int)
  body_geomadr: wp.array(dtype=int)
  body_pos: wp.array2d(dtype=wp.vec3)
  body_quat: wp.array2d(dtype=wp.quat)
  body_ipos: wp.array2d(dtype=wp.vec3)
  body_iquat: wp.array2d(dtype=wp.quat)
  body_mass: wp.array2d(dtype=float)
  body_subtreemass: wp.array2d(dtype=float)
  subtree_mass: wp.array2d(dtype=float)
  body_inertia: wp.array2d(dtype=wp.vec3)
  body_invweight0: wp.array3d(dtype=float)
  body_contype: wp.array(dtype=int)
  body_conaffinity: wp.array(dtype=int)
  body_gravcomp: wp.array2d(dtype=float)
  jnt_type: wp.array(dtype=int)
  jnt_qposadr: wp.array(dtype=int)
  jnt_dofadr: wp.array(dtype=int)
  jnt_bodyid: wp.array(dtype=int)
  jnt_limited: wp.array(dtype=int)
  jnt_actfrclimited: wp.array(dtype=bool)
  jnt_solref: wp.array2d(dtype=wp.vec2)
  jnt_solimp: wp.array2d(dtype=vec5)
  jnt_pos: wp.array2d(dtype=wp.vec3)
  jnt_axis: wp.array2d(dtype=wp.vec3)
  jnt_stiffness: wp.array2d(dtype=float)
  jnt_range: wp.array3d(dtype=float)
  jnt_actfrcrange: wp.array2d(dtype=wp.vec2)
  jnt_margin: wp.array2d(dtype=float)
  jnt_limited_slide_hinge_adr: wp.array(dtype=int)  # warp only
  jnt_limited_ball_adr: wp.array(dtype=int)  # warp only
  jnt_actgravcomp: wp.array(dtype=int)
  dof_bodyid: wp.array(dtype=int)
  dof_jntid: wp.array(dtype=int)
  dof_parentid: wp.array(dtype=int)
  dof_Madr: wp.array(dtype=int)
  dof_armature: wp.array2d(dtype=float)
  dof_damping: wp.array2d(dtype=float)
  dof_invweight0: wp.array2d(dtype=float)
  dof_frictionloss: wp.array2d(dtype=float)
  dof_solimp: wp.array2d(dtype=vec5)
  dof_solref: wp.array2d(dtype=wp.vec2)
  dof_tri_row: wp.array(dtype=int)  # warp only
  dof_tri_col: wp.array(dtype=int)  # warp only
  geom_type: wp.array(dtype=int)
  geom_contype: wp.array(dtype=int)
  geom_conaffinity: wp.array(dtype=int)
  geom_condim: wp.array(dtype=int)
  geom_bodyid: wp.array(dtype=int)
  geom_dataid: wp.array(dtype=int)
  geom_group: wp.array(dtype=int)
  geom_matid: wp.array2d(dtype=int)
  geom_priority: wp.array(dtype=int)
  geom_solmix: wp.array2d(dtype=float)
  geom_solref: wp.array2d(dtype=wp.vec2)
  geom_solimp: wp.array2d(dtype=vec5)
  geom_size: wp.array2d(dtype=wp.vec3)
  geom_aabb: wp.array(dtype=wp.vec3)
  geom_rbound: wp.array2d(dtype=float)
  geom_pos: wp.array2d(dtype=wp.vec3)
  geom_quat: wp.array2d(dtype=wp.quat)
  geom_friction: wp.array2d(dtype=wp.vec3)
  geom_margin: wp.array2d(dtype=float)
  geom_gap: wp.array2d(dtype=float)
  geom_rgba: wp.array2d(dtype=wp.vec4)
  site_bodyid: wp.array(dtype=int)
  site_pos: wp.array2d(dtype=wp.vec3)
  site_quat: wp.array2d(dtype=wp.quat)
  cam_mode: wp.array(dtype=int)
  cam_bodyid: wp.array(dtype=int)
  cam_targetbodyid: wp.array(dtype=int)
  cam_pos: wp.array2d(dtype=wp.vec3)
  cam_quat: wp.array2d(dtype=wp.quat)
  cam_poscom0: wp.array2d(dtype=wp.vec3)
  cam_pos0: wp.array2d(dtype=wp.vec3)
  cam_fovy: wp.array(dtype=float)
  cam_resolution: wp.array(dtype=wp.vec2i)
  cam_sensorsize: wp.array(dtype=wp.vec2)
  cam_intrinsic: wp.array(dtype=wp.vec4)
  light_mode: wp.array(dtype=int)
  light_bodyid: wp.array(dtype=int)
  light_targetbodyid: wp.array(dtype=int)
  light_pos: wp.array2d(dtype=wp.vec3)
  light_dir: wp.array2d(dtype=wp.vec3)
  light_poscom0: wp.array2d(dtype=wp.vec3)
  light_pos0: wp.array2d(dtype=wp.vec3)
  mesh_vertadr: wp.array(dtype=int)
  mesh_vertnum: wp.array(dtype=int)
  mesh_vert: wp.array(dtype=wp.vec3)
  mesh_faceadr: wp.array(dtype=int)
  mesh_face: wp.array(dtype=wp.vec3i)
  eq_type: wp.array(dtype=int)
  eq_obj1id: wp.array(dtype=int)
  eq_obj2id: wp.array(dtype=int)
  eq_objtype: wp.array(dtype=int)
  eq_active0: wp.array(dtype=bool)
  eq_solref: wp.array2d(dtype=wp.vec2)
  eq_solimp: wp.array2d(dtype=vec5)
  eq_data: wp.array2d(dtype=vec11)
  eq_connect_adr: wp.array(dtype=int)
  eq_wld_adr: wp.array(dtype=int)
  eq_jnt_adr: wp.array(dtype=int)
  eq_ten_adr: wp.array(dtype=int)
  actuator_moment_tiles_nv: tuple[TileSet, ...]
  actuator_moment_tiles_nu: tuple[TileSet, ...]
  actuator_affine_bias_gain: bool  # warp only
  actuator_trntype: wp.array(dtype=int)
  actuator_dyntype: wp.array(dtype=int)
  actuator_gaintype: wp.array(dtype=int)
  actuator_biastype: wp.array(dtype=int)
  actuator_trnid: wp.array(dtype=wp.vec2i)
  actuator_actadr: wp.array(dtype=int)
  actuator_actnum: wp.array(dtype=int)
  actuator_ctrllimited: wp.array(dtype=bool)
  actuator_forcelimited: wp.array(dtype=bool)
  actuator_actlimited: wp.array(dtype=bool)
  actuator_dynprm: wp.array2d(dtype=vec10f)
  actuator_gainprm: wp.array2d(dtype=vec10f)
  actuator_biasprm: wp.array2d(dtype=vec10f)
  actuator_ctrlrange: wp.array2d(dtype=wp.vec2)
  actuator_forcerange: wp.array2d(dtype=wp.vec2)
  actuator_actrange: wp.array2d(dtype=wp.vec2)
  actuator_gear: wp.array2d(dtype=wp.spatial_vector)
  nxn_geom_pair: wp.array(dtype=wp.vec2i)  # warp only
  nxn_pairid: wp.array(dtype=int)  # warp only
  pair_dim: wp.array(dtype=int)
  pair_geom1: wp.array(dtype=int)
  pair_geom2: wp.array(dtype=int)
  pair_solref: wp.array2d(dtype=wp.vec2)
  pair_solreffriction: wp.array2d(dtype=wp.vec2)
  pair_solimp: wp.array2d(dtype=vec5)
  pair_margin: wp.array2d(dtype=float)
  pair_gap: wp.array2d(dtype=float)
  pair_friction: wp.array2d(dtype=vec5)
  exclude_signature: wp.array(dtype=int)
  condim_max: int  # warp only
  tendon_adr: wp.array(dtype=int)
  tendon_num: wp.array(dtype=int)
  tendon_limited: wp.array(dtype=int)
  tendon_limited_adr: wp.array(dtype=int)
  tendon_solref_lim: wp.array2d(dtype=wp.vec2)
  tendon_solimp_lim: wp.array2d(dtype=vec5)
  tendon_range: wp.array2d(dtype=wp.vec2)
  tendon_margin: wp.array2d(dtype=float)
  tendon_length0: wp.array2d(dtype=float)
  tendon_invweight0: wp.array2d(dtype=float)
  wrap_objid: wp.array(dtype=int)
  wrap_prm: wp.array(dtype=float)
  wrap_type: wp.array(dtype=int)
  tendon_jnt_adr: wp.array(dtype=int)  # warp only
  tendon_site_adr: wp.array(dtype=int)  # warp only
  tendon_site_pair_adr: wp.array(dtype=int)  # warp only
  ten_wrapadr_site: wp.array(dtype=int)  # warp only
  ten_wrapnum_site: wp.array(dtype=int)  # warp only
  wrap_jnt_adr: wp.array(dtype=int)  # warp only
  wrap_site_adr: wp.array(dtype=int)  # warp only
  wrap_site_pair_adr: wp.array(dtype=int)  # warp only
  sensor_type: wp.array(dtype=int)
  sensor_datatype: wp.array(dtype=int)
  sensor_objtype: wp.array(dtype=int)
  sensor_objid: wp.array(dtype=int)
  sensor_reftype: wp.array(dtype=int)
  sensor_refid: wp.array(dtype=int)
  sensor_dim: wp.array(dtype=int)
  sensor_adr: wp.array(dtype=int)
  sensor_cutoff: wp.array(dtype=float)
  sensor_pos_adr: wp.array(dtype=int)  # warp only
  sensor_vel_adr: wp.array(dtype=int)  # warp only
  sensor_acc_adr: wp.array(dtype=int)  # warp only
  sensor_subtree_vel: bool  # warp only
  sensor_rne_postconstraint: bool  # warp only
  mocap_bodyid: wp.array(dtype=int)  # warp only
  mat_rgba: wp.array2d(dtype=wp.vec4)


@dataclasses.dataclass
class Contact:
  """Contact data.

  Attributes:
    dist: distance between nearest points; neg: penetration
    pos: position of contact point: midpoint between geoms
    frame: normal is in [0-2], points from geom[0] to geom[1]
    includemargin: include if dist<includemargin=margin-gap
    friction: tangent1, 2, spin, roll1, 2
    solref: constraint solver reference, normal direction
    solreffriction: constraint solver reference, friction directions
    solimp: constraint solver impedance
    dim: contact space dimensionality: 1, 3, 4 or 6
    geom: geom ids; -1 for flex
    efc_address: address in efc; -1: not included
    worldid: world id
  """

  dist: wp.array(dtype=float)
  pos: wp.array(dtype=wp.vec3)
  frame: wp.array(dtype=wp.mat33)
  includemargin: wp.array(dtype=float)
  friction: wp.array(dtype=vec5)
  solref: wp.array(dtype=wp.vec2)
  solreffriction: wp.array(dtype=wp.vec2)
  solimp: wp.array(dtype=vec5)
  dim: wp.array(dtype=int)
  geom: wp.array(dtype=wp.vec2i)
  efc_address: wp.array2d(dtype=int)
  worldid: wp.array(dtype=int)


@dataclasses.dataclass
class Data:
  """Dynamic state that updates each step.

  Attributes:
    nworld: number of worlds                                    ()
    nconmax: maximum number of contacts                         ()
    njmax: maximum number of constraints                        ()
    ncon: number of detected contacts                           ()
    ne: number of equality constraints                          ()
    ne_connect: number of equality connect constraints          ()
    ne_weld: number of equality weld constraints                ()
    ne_jnt: number of equality joint constraints                ()
    ne_ten: number of equality tendon constraints               ()
    nf: number of friction constraints                          ()
    nl: number of limit constraints                             ()
    nefc: number of constraints                                 (1,)
    time: simulation time                                       (nworld,)
    qpos: position                                              (nworld, nq)
    qvel: velocity                                              (nworld, nv)
    act: actuator activation                                    (nworld, na)
    qacc_warmstart: acceleration used for warmstart             (nworld, nv)
    ctrl: control                                               (nworld, nu)
    qfrc_applied: applied generalized force                     (nworld, nv)
    xfrc_applied: applied Cartesian force/torque                (nworld, nbody, 6)
    fluid_applied: applied fluid force/torque                   (nworld, nbody, 6)
    eq_active: enable/disable constraints                       (nworld, neq)
    mocap_pos: position of mocap bodies                         (nworld, nmocap, 3)
    mocap_quat: orientation of mocap bodies                     (nworld, nmocap, 4)
    qacc: acceleration                                          (nworld, nv)
    act_dot: time-derivative of actuator activation             (nworld, na)
    xpos: Cartesian position of body frame                      (nworld, nbody, 3)
    xquat: Cartesian orientation of body frame                  (nworld, nbody, 4)
    xmat: Cartesian orientation of body frame                   (nworld, nbody, 3, 3)
    xipos: Cartesian position of body com                       (nworld, nbody, 3)
    ximat: Cartesian orientation of body inertia                (nworld, nbody, 3, 3)
    xanchor: Cartesian position of joint anchor                 (nworld, njnt, 3)
    xaxis: Cartesian joint axis                                 (nworld, njnt, 3)
    geom_xpos: Cartesian geom position                          (nworld, ngeom, 3)
    geom_xmat: Cartesian geom orientation                       (nworld, ngeom, 3, 3)
    site_xpos: Cartesian site position                          (nworld, nsite, 3)
    site_xmat: Cartesian site orientation                       (nworld, nsite, 3, 3)
    cam_xpos: Cartesian camera position                         (nworld, ncam, 3)
    cam_xmat: Cartesian camera orientation                      (nworld, ncam, 3, 3)
    light_xpos: Cartesian light position                        (nworld, nlight, 3)
    light_xdir: Cartesian light direction                       (nworld, nlight, 3)
    subtree_com: center of mass of each subtree                 (nworld, nbody, 3)
    cdof: com-based motion axis of each dof (rot:lin)           (nworld, nv, 6)
    cinert: com-based body inertia and mass                     (nworld, nbody, 10)
    actuator_length: actuator lengths                           (nworld, nu)
    actuator_moment: actuator moments                           (nworld, nu, nv)
    crb: com-based composite inertia and mass                   (nworld, nbody, 10)
    qM: total inertia (sparse) (nworld, 1, nM) or               (nworld, nv, nv) if dense
    qLD: L'*D*L factorization of M (sparse) (nworld, 1, nM) or  (nworld, nv, nv) if dense
    qLDiagInv: 1/diag(D)                                        (nworld, nv)
    ten_velocity: tendon velocities                             (nworld, ntendon)
    actuator_velocity: actuator velocities                      (nworld, nu)
    cvel: com-based velocity (rot:lin)                          (nworld, nbody, 6)
    cdof_dot: time-derivative of cdof (rot:lin)                 (nworld, nv, 6)
    qfrc_bias: C(qpos,qvel)                                     (nworld, nv)
    qfrc_spring: passive spring force                           (nworld, nv)
    qfrc_damper: passive damper force                           (nworld, nv)
    qfrc_gravcomp: passive gravity compensation force           (nworld, nv)
    qfrc_fluid: passive fluid force                             (nworld, nv)
    qfrc_passive: total passive force                           (nworld, nv)
    subtree_linvel: linear velocity of subtree com              (nworld, nbody, 3)
    subtree_angmom: angular momentum about subtree com          (nworld, nbody, 3)
    subtree_bodyvel: subtree body velocity (ang, vel)           (nworld, nbody, 6)
    actuator_force: actuator force in actuation space           (nworld, nu)
    qfrc_actuator: actuator force                               (nworld, nv)
    qfrc_smooth: net unconstrained force                        (nworld, nv)
    qacc_smooth: unconstrained acceleration                     (nworld, nv)
    qfrc_constraint: constraint force                           (nworld, nv)
    contact: contact data
    efc: constraint data
    rne_cacc: arrays used for smooth.rne                        (nworld, nbody, 6)
    rne_cfrc: arrays used for smooth.rne                        (nworld, nbody, 6)
    qpos_t0: temporary array for rk4                            (nworld, nq)
    qvel_t0: temporary array for rk4                            (nworld, nv)
    act_t0: temporary array for rk4                             (nworld, na)
    qvel_rk: temporary array for rk4                            (nworld, nv)
    qacc_rk: temporary array for rk4                            (nworld, nv)
    act_dot_rk: temporary array for rk4                         (nworld, na)
    qfrc_integration: temporary array for integration           (nworld, nv)
    qacc_integration: temporary array for integration           (nworld, nv)
    act_vel_integration: temporary array for integration        (nworld, nu)
    qM_integration: temporary array for integration             (nworld, nv, nv) if dense
    qLD_integration: temporary array for integration            (nworld, nv, nv) if dense
    qLDiagInv_integration: temporary array for integration      (nworld, nv)
    boxes_sorted: min, max of sorted bounding boxes             (nworld, ngeom, 2)
    sap_projections_lower: broadphase context                   (2*nworld, ngeom)
    sap_projections_upper: broadphase context                   (nworld, ngeom)
    sap_sort_index: broadphase context                          (2*nworld, ngeom)
    sap_range: broadphase context                               (nworld, ngeom)
    sap_cumulative_sum: broadphase context                      (nworld*ngeom,)
    sap_segment_index: broadphase context                       (nworld+1,)
    dyn_geom_aabb: dynamic geometry axis-aligned bounding boxes (nworld, ngeom, 2)
    collision_pair: collision pairs from broadphase             (nconmax,)
    collision_worldid: collision world ids from broadphase      (nconmax,)
    ncollision: collision count from broadphase                 ()
    cacc: com-based acceleration                                (nworld, nbody, 6)
    cfrc_int: com-based interaction force with parent           (nworld, nbody, 6)
    cfrc_ext: com-based external force on body                  (nworld, nbody, 6)
    ten_length: tendon lengths                                  (nworld, ntendon)
    ten_J: tendon Jacobian                                      (nworld, ntendon, nv)
    ten_wrapadr: start address of tendon's path                 (nworld, ntendon)
    ten_wrapnum: number of wrap points in path                  (nworld, ntendon)
    wrap_obj: geomid; -1: site; -2: pulley                      (nworld, nwrap, 2)
    wrap_xpos: Cartesian 3D points in all paths                 (nworld, nwrap, 6)
    sensordata: sensor data array                               (nsensordata,)
  """

  nworld: int  # warp only
  nconmax: int  # warp only
  njmax: int  # warp only

  ncon: wp.array(dtype=int)
  ne: wp.array(dtype=int)
  ne_connect: wp.array(dtype=int)  # warp only
  ne_weld: wp.array(dtype=int)  # warp only
  ne_jnt: wp.array(dtype=int)  # warp only
  ne_ten: wp.array(dtype=int)  # warp only
  nf: wp.array(dtype=int)
  nl: wp.array(dtype=int)
  nefc: wp.array(dtype=int)
  time: wp.array(dtype=float)
  qpos: wp.array2d(dtype=float)
  qvel: wp.array2d(dtype=float)
  act: wp.array2d(dtype=float)
  qacc_warmstart: wp.array2d(dtype=float)
  ctrl: wp.array2d(dtype=float)
  qfrc_applied: wp.array2d(dtype=float)
  xfrc_applied: wp.array2d(dtype=wp.spatial_vector)
  fluid_applied: wp.array2d(dtype=wp.spatial_vector)  # warp only
  eq_active: wp.array2d(dtype=bool)
  mocap_pos: wp.array2d(dtype=wp.vec3)
  mocap_quat: wp.array2d(dtype=wp.quat)
  qacc: wp.array2d(dtype=float)
  act_dot: wp.array2d(dtype=float)
  xpos: wp.array2d(dtype=wp.vec3)
  xquat: wp.array2d(dtype=wp.quat)
  xmat: wp.array2d(dtype=wp.mat33)
  xipos: wp.array2d(dtype=wp.vec3)
  ximat: wp.array2d(dtype=wp.mat33)
  xanchor: wp.array2d(dtype=wp.vec3)
  xaxis: wp.array2d(dtype=wp.vec3)
  geom_xpos: wp.array2d(dtype=wp.vec3)
  geom_xmat: wp.array2d(dtype=wp.mat33)
  site_xpos: wp.array2d(dtype=wp.vec3)
  site_xmat: wp.array2d(dtype=wp.mat33)
  cam_xpos: wp.array2d(dtype=wp.vec3)
  cam_xmat: wp.array2d(dtype=wp.mat33)
  light_xpos: wp.array2d(dtype=wp.vec3)
  light_xdir: wp.array2d(dtype=wp.vec3)
  subtree_com: wp.array2d(dtype=wp.vec3)
  cdof: wp.array2d(dtype=wp.spatial_vector)
  cinert: wp.array2d(dtype=vec10)
  actuator_length: wp.array2d(dtype=float)
  actuator_moment: wp.array3d(dtype=float)
  crb: wp.array2d(dtype=vec10)
  qM: wp.array3d(dtype=float)
  qLD: wp.array3d(dtype=float)
  qLDiagInv: wp.array2d(dtype=float)
  ten_velocity: wp.array2d(dtype=float)
  actuator_velocity: wp.array2d(dtype=float)
  cvel: wp.array2d(dtype=wp.spatial_vector)
  cdof_dot: wp.array2d(dtype=wp.spatial_vector)
  qfrc_bias: wp.array2d(dtype=float)
  qfrc_spring: wp.array2d(dtype=float)
  qfrc_damper: wp.array2d(dtype=float)
  qfrc_gravcomp: wp.array2d(dtype=float)
  qfrc_fluid: wp.array2d(dtype=float)
  qfrc_passive: wp.array2d(dtype=float)
  subtree_linvel: wp.array2d(dtype=wp.vec3)
  subtree_angmom: wp.array2d(dtype=wp.vec3)
  subtree_bodyvel: wp.array2d(dtype=wp.spatial_vector)  # warp only
  actuator_force: wp.array2d(dtype=float)
  qfrc_actuator: wp.array2d(dtype=float)
  qfrc_smooth: wp.array2d(dtype=float)
  qacc_smooth: wp.array2d(dtype=float)
  qfrc_constraint: wp.array2d(dtype=float)
  contact: Contact
  efc: Constraint

  # RK4
  qpos_t0: wp.array2d(dtype=float)
  qvel_t0: wp.array2d(dtype=float)
  act_t0: wp.array2d(dtype=float)
  qvel_rk: wp.array2d(dtype=float)
  qacc_rk: wp.array2d(dtype=float)
  act_dot_rk: wp.array2d(dtype=float)

  # euler + implicit integration
  qfrc_integration: wp.array2d(dtype=float)
  qacc_integration: wp.array2d(dtype=float)
  act_vel_integration: wp.array2d(dtype=float)
  qM_integration: wp.array3d(dtype=float)
  qLD_integration: wp.array3d(dtype=float)
  qLDiagInv_integration: wp.array2d(dtype=float)

  # sweep-and-prune broadphase
  sap_projection_lower: wp.array2d(dtype=float)
  sap_projection_upper: wp.array2d(dtype=float)
  sap_sort_index: wp.array2d(dtype=int)
  sap_range: wp.array2d(dtype=int)
  sap_cumulative_sum: wp.array(dtype=int)
  sap_segment_index: wp.array(dtype=int)

  # collision driver
  collision_pair: wp.array(dtype=wp.vec2i)
  collision_pairid: wp.array(dtype=int)
  collision_worldid: wp.array(dtype=int)
  ncollision: wp.array(dtype=int)

  # rne_postconstraint
  cacc: wp.array2d(dtype=wp.spatial_vector)
  cfrc_int: wp.array2d(dtype=wp.spatial_vector)
  cfrc_ext: wp.array2d(dtype=wp.spatial_vector)

  # tendon
  ten_length: wp.array2d(dtype=float)
  ten_J: wp.array3d(dtype=float)
  ten_wrapadr: wp.array2d(dtype=int)
  ten_wrapnum: wp.array2d(dtype=int)
  wrap_obj: wp.array2d(dtype=wp.vec2i)
  wrap_xpos: wp.array2d(dtype=wp.spatial_vector)

  # sensors
  sensordata: wp.array2d(dtype=float)
