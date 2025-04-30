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


class vec5f(wp.types.vector(length=5, dtype=wp.float32)):
  pass


class vec10f(wp.types.vector(length=10, dtype=wp.float32)):
  pass


class vec11f(wp.types.vector(length=11, dtype=wp.float32)):
  pass


vec5 = vec5f
vec10 = vec10f
vec11 = vec11f
array2df = wp.array2d(dtype=wp.float32)
array3df = wp.array3d(dtype=wp.float32)


@wp.struct
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
  """

  timestep: float
  impratio: wp.float32
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


@wp.struct
class Statistic:
  """Model statistics (in qpos0).

  Attributes:
    meaninertia: mean diagonal inertia
  """

  meaninertia: float


@wp.struct
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

  worldid: wp.array(dtype=wp.int32, ndim=1)
  id: wp.array(dtype=wp.int32, ndim=1)
  J: wp.array(dtype=wp.float32, ndim=2)
  pos: wp.array(dtype=wp.float32, ndim=1)
  margin: wp.array(dtype=wp.float32, ndim=1)
  D: wp.array(dtype=wp.float32, ndim=1)
  aref: wp.array(dtype=wp.float32, ndim=1)
  frictionloss: wp.array(dtype=wp.float32, ndim=1)
  force: wp.array(dtype=wp.float32, ndim=1)
  Jaref: wp.array(dtype=wp.float32, ndim=1)
  Ma: wp.array(dtype=wp.float32, ndim=2)
  grad: wp.array(dtype=wp.float32, ndim=2)
  grad_dot: wp.array(dtype=wp.float32, ndim=1)
  Mgrad: wp.array(dtype=wp.float32, ndim=2)
  search: wp.array(dtype=wp.float32, ndim=2)
  search_dot: wp.array(dtype=wp.float32, ndim=1)
  gauss: wp.array(dtype=wp.float32, ndim=1)
  cost: wp.array(dtype=wp.float32, ndim=1)
  prev_cost: wp.array(dtype=wp.float32, ndim=1)
  solver_niter: wp.array(dtype=wp.int32, ndim=1)
  active: wp.array(dtype=bool, ndim=1)
  gtol: wp.array(dtype=wp.float32, ndim=1)
  mv: wp.array(dtype=wp.float32, ndim=2)
  jv: wp.array(dtype=wp.float32, ndim=1)
  quad: wp.array(dtype=wp.vec3f, ndim=1)
  quad_gauss: wp.array(dtype=wp.vec3f, ndim=1)
  h: wp.array(dtype=wp.float32, ndim=3)
  alpha: wp.array(dtype=wp.float32, ndim=1)
  prev_grad: wp.array(dtype=wp.float32, ndim=2)
  prev_Mgrad: wp.array(dtype=wp.float32, ndim=2)
  beta: wp.array(dtype=wp.float32, ndim=1)
  beta_num: wp.array(dtype=wp.float32, ndim=1)
  beta_den: wp.array(dtype=wp.float32, ndim=1)
  done: wp.array(dtype=bool, ndim=1)
  # linesearch
  ls_done: wp.array(dtype=bool, ndim=1)
  p0: wp.array(dtype=wp.vec3, ndim=1)
  lo: wp.array(dtype=wp.vec3, ndim=1)
  lo_alpha: wp.array(dtype=wp.float32, ndim=1)
  hi: wp.array(dtype=wp.vec3, ndim=1)
  hi_alpha: wp.array(dtype=wp.float32, ndim=1)
  lo_next: wp.array(dtype=wp.vec3, ndim=1)
  lo_next_alpha: wp.array(dtype=wp.float32, ndim=1)
  hi_next: wp.array(dtype=wp.vec3, ndim=1)
  hi_next_alpha: wp.array(dtype=wp.float32, ndim=1)
  mid: wp.array(dtype=wp.vec3, ndim=1)
  mid_alpha: wp.array(dtype=wp.float32, ndim=1)
  cost_candidate: wp.array(dtype=wp.float32, ndim=2)
  quad_total_candidate: wp.array(dtype=wp.vec3f, ndim=2)
  # elliptic cone
  u: wp.array(dtype=wp.float32, ndim=2)
  uu: wp.array(dtype=wp.float32, ndim=1)
  uv: wp.array(dtype=wp.float32, ndim=1)
  vv: wp.array(dtype=wp.float32, ndim=1)
  condim: wp.array(dtype=wp.int32, ndim=1)


@wp.struct
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
    nlsp: number of step sizes for parallel linsearch        ()
    npair: number of predefined geom pairs                   ()
    opt: physics options
    stat: model statistics
    qpos0: qpos values at default pose                       (nq,)
    qpos_spring: reference pose for springs                  (nq,)
    body_tree: BFS ordering of body ids
    body_treeadr: starting index of each body tree level
    actuator_moment_offset_nv: tiling configuration
    actuator_moment_offset_nu: tiling configuration
    actuator_moment_tileadr: tiling configuration
    actuator_moment_tilesize_nv: tiling configuration
    actuator_moment_tilesize_nu: tiling configuration
    alpha_candidate: step size candidates for engine solver  (nlsp,)
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
    qLD_tile: tiling configuration
    qLD_tileadr: tiling configuration
    qLD_tilesize: tiling configuration
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
    body_pos: position offset rel. to parent body            (nbody, 3)
    body_quat: orientation offset rel. to parent body        (nbody, 4)
    body_ipos: local position of center of mass              (nbody, 3)
    body_iquat: local orientation of inertia ellipsoid       (nbody, 4)
    body_mass: mass                                          (nbody,)
    body_subtreemass: mass of subtree starting at this body  (nbody,)
    subtree_mass: mass of subtree                            (nbody,)
    body_inertia: diagonal inertia in ipos/iquat frame       (nbody, 3)
    body_invweight0: mean inv inert in qpos0 (trn, rot)      (nbody, 2)
    body_contype: OR over all geom contypes                  (nbody,)
    body_conaffinity: OR over all geom conaffinities         (nbody,)
    body_gravcomp: antigravity force, units of body weight   (nbody,)
    jnt_type: type of joint (mjtJoint)                       (njnt,)
    jnt_qposadr: start addr in 'qpos' for joint's data       (njnt,)
    jnt_dofadr: start addr in 'qvel' for joint's data        (njnt,)
    jnt_bodyid: id of joint's body                           (njnt,)
    jnt_limited: does joint have limits                      (njnt,)
    jnt_actfrclimited: does joint have actuator force limits (njnt,)
    jnt_solref: constraint solver reference: limit           (njnt, mjNREF)
    jnt_solimp: constraint solver impedance: limit           (njnt, mjNIMP)
    jnt_pos: local anchor position                           (njnt, 3)
    jnt_axis: local joint axis                               (njnt, 3)
    jnt_stiffness: stiffness coefficient                     (njnt,)
    jnt_range: joint limits                                  (njnt, 2)
    jnt_actfrcrange: range of total actuator force           (njnt, 2)
    jnt_margin: min distance for limit detection             (njnt,)
    jnt_limited_slide_hinge_adr: limited/slide/hinge jntadr
    jnt_limited_ball_adr: limited/ball jntadr
    jnt_actgravcomp: is gravcomp force applied via actuators (njnt,)
    dof_bodyid: id of dof's body                             (nv,)
    dof_jntid: id of dof's joint                             (nv,)
    dof_parentid: id of dof's parent; -1: none               (nv,)
    dof_Madr: dof address in M-diagonal                      (nv,)
    dof_armature: dof armature inertia/mass                  (nv,)
    dof_damping: damping coefficient                         (nv,)
    dof_invweight0: diag. inverse inertia in qpos0           (nv,)
    dof_frictionloss: dof friction loss                      (nv,)
    dof_solimp: constraint solver impedance: frictionloss    (nv, NIMP)
    dof_solref: constraint solver reference: frictionloss    (nv, NREF)
    dof_tri_row: np.tril_indices                             (mjm.nv)[0]
    dof_tri_col: np.tril_indices                             (mjm.nv)[1]
    geom_type: geometric type (mjtGeom)                      (ngeom,)
    geom_contype: geom contact type                          (ngeom,)
    geom_conaffinity: geom contact affinity                  (ngeom,)
    geom_condim: contact dimensionality (1, 3, 4, 6)         (ngeom,)
    geom_bodyid: id of geom's body                           (ngeom,)
    geom_dataid: id of geom's mesh/hfield; -1: none          (ngeom,)
    geom_priority: geom contact priority                     (ngeom,)
    geom_solmix: mixing coef for solref/imp in geom pair     (ngeom,)
    geom_solref: constraint solver reference: contact        (ngeom, mjNREF)
    geom_solimp: constraint solver impedance: contact        (ngeom, mjNIMP)
    geom_size: geom-specific size parameters                 (ngeom, 3)
    geom_aabb: bounding box, (center, size)                  (ngeom, 6)
    geom_rbound: radius of bounding sphere                   (ngeom,)
    geom_pos: local position offset rel. to body             (ngeom, 3)
    geom_quat: local orientation offset rel. to body         (ngeom, 4)
    geom_friction: friction for (slide, spin, roll)          (ngeom, 3)
    geom_margin: detect contact if dist<margin               (ngeom,)
    geom_gap: include in solver if dist<margin-gap           (ngeom,)
    site_bodyid: id of site's body                           (nsite,)
    site_pos: local position offset rel. to body             (nsite, 3)
    site_quat: local orientation offset rel. to body         (nsite, 4)
    cam_mode: camera tracking mode (mjtCamLight)             (ncam,)
    cam_bodyid: id of camera's body                          (ncam,)
    cam_targetbodyid: id of targeted body; -1: none          (ncam,)
    cam_pos: position rel. to body frame                     (ncam, 3)
    cam_quat: orientation rel. to body frame                 (ncam, 4)
    cam_poscom0: global position rel. to sub-com in qpos0    (ncam, 3)
    cam_pos0: Cartesian camera position                      (ncam, 3)
    cam_fovy: y field-of-view (ortho ? len : deg)            (ncam,)
    cam_resolution: resolution: pixels [width, height]       (ncam, 2)
    cam_sensorsize: sensor size: length [width, height]      (ncam, 2)
    cam_intrinsic: [focal length; principal point]           (ncam, 4)
    light_mode: light tracking mode (mjtCamLight)            (nlight,)
    light_bodyid: id of light's body                         (nlight,)
    light_targetbodyid: id of targeted body; -1: none        (nlight,)
    light_pos: position rel. to body frame                   (nlight, 3)
    light_dir: direction rel. to body frame                  (nlight, 3)
    light_poscom0: global position rel. to sub-com in qpos0  (nlight, 3)
    light_pos0: global position rel. to body in qpos0        (nworld, nlight, 3)
    mesh_vertadr: first vertex address                       (nmesh,)
    mesh_vertnum: number of vertices                         (nmesh,)
    mesh_vert: vertex positions for all meshes               (nmeshvert, 3)
    eq_type: constraint type (mjtEq)                         (neq,)
    eq_obj1id: id of object 1                                (neq,)
    eq_obj2id: id of object 2                                (neq,)
    eq_objtype: type of both objects (mjtObj)                (neq,)
    eq_active0: initial enable/disable constraint state      (neq,)
    eq_solref: constraint solver reference                   (neq, mjNREF)
    eq_solimp: constraint solver impedance                   (neq, mjNIMP)
    eq_data: numeric data for constraint                     (neq, mjNEQDATA)
    eq_connect_adr: eq_* addresses of type `CONNECT`
    eq_wld_adr: eq_* addresses of type `WELD`
    eq_jnt_adr: eq_* addresses of type `JOINT`
    eq_ten_adr: eq_* addresses of type `TENDON`              (<=neq,)
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
    actuator_dynprm: dynamics parameters                     (nu, mjNDYN)
    actuator_gainprm: gain parameters                        (nu, mjNGAIN)
    actuator_biasprm: bias parameters                        (nu, mjNBIAS)
    actuator_ctrlrange: range of controls                    (nu, 2)
    actuator_forcerange: range of forces                     (nu, 2)
    actuator_actrange: range of activations                  (nu, 2)
    actuator_gear: scale length and transmitted force        (nu, 6)
    exclude_signature: body1 << 16 + body2                   (nexclude,)
    actuator_affine_bias_gain: affine bias/gain present
    nxn_geom_pair: valid collision pair geom ids             (<= ngeom * (ngeom - 1) // 2,)
    nxn_pairid: predefined pair id, -1 if not predefined     (<= ngeom * (ngeom - 1) // 2,)
    pair_dim: contact dimensionality                         (npair,)
    pair_geom1: id of geom1                                  (npair,)
    pair_geom2: id of geom2                                  (npair,)
    pair_solref: solver reference: contact normal            (npair, mjNREF)
    pair_solreffriction: solver reference: contact friction  (npair, mjNREF)
    pair_solimp: solver impedance: contact                   (npair, mjNIMP)
    pair_margin: detect contact if dist<margin               (npair,)
    pair_gap: include in solver if dist<margin-gap           (npair,)
    pair_friction: tangent1, 2, spin, roll1, 2               (npair, 5)
    condim_max: maximum condim for geoms
    tendon_adr: address of first object in tendon's path     (ntendon,)
    tendon_num: number of objects in tendon's path           (ntendon,)
    tendon_limited: does tendon have length limits           (ntendon,)
    tendon_limited_adr: addresses for limited tendons        (<=ntendon,)
    tendon_solref_lim: constraint solver reference: limit    (ntendon, mjNREF)
    tendon_solimp_lim: constraint solver impedance: limit    (ntendon, mjNIMP)
    tendon_range: tendon length limits                       (ntendon, 2)
    tendon_margin: min distance for limit detection          (ntendon,)
    tendon_length0: tendon length in qpos0                   (ntendon,)
    tendon_invweight0: inv. weight in qpos0                  (ntendon,)
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
  nlsp: int  # warp only
  npair: int
  opt: Option
  stat: Statistic
  qpos0: wp.array(dtype=wp.float32, ndim=1)
  qpos_spring: wp.array(dtype=wp.float32, ndim=1)
  body_tree: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_treeadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_moment_offset_nv: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_moment_offset_nu: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_moment_tileadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_moment_tilesize_nv: wp.array(dtype=wp.int32, ndim=1)  # warp only
  actuator_moment_tilesize_nu: wp.array(dtype=wp.int32, ndim=1)  # warp only
  alpha_candidate: wp.array(dtype=wp.float32, ndim=1)  # warp only
  qM_fullm_i: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qM_fullm_j: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qM_mulm_i: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qM_mulm_j: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qM_madr_ij: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_update_tree: wp.array(dtype=wp.vec3i, ndim=1)  # warp only
  qLD_update_treeadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  M_rownnz: wp.array(dtype=wp.int32, ndim=1)
  M_rowadr: wp.array(dtype=wp.int32, ndim=1)
  M_colind: wp.array(dtype=wp.int32, ndim=1)
  mapM2M: wp.array(dtype=wp.int32, ndim=1)
  qLD_tile: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_tileadr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  qLD_tilesize: wp.array(dtype=wp.int32, ndim=1)  # warp only
  body_parentid: wp.array(dtype=wp.int32, ndim=1)
  body_rootid: wp.array(dtype=wp.int32, ndim=1)
  body_weldid: wp.array(dtype=wp.int32, ndim=1)
  body_mocapid: wp.array(dtype=wp.int32, ndim=1)
  body_jntnum: wp.array(dtype=wp.int32, ndim=1)
  body_jntadr: wp.array(dtype=wp.int32, ndim=1)
  body_dofnum: wp.array(dtype=wp.int32, ndim=1)
  body_dofadr: wp.array(dtype=wp.int32, ndim=1)
  body_geomnum: wp.array(dtype=wp.int32, ndim=1)
  body_geomadr: wp.array(dtype=wp.int32, ndim=1)
  body_pos: wp.array(dtype=wp.vec3, ndim=1)
  body_quat: wp.array(dtype=wp.quat, ndim=1)
  body_ipos: wp.array(dtype=wp.vec3, ndim=1)
  body_iquat: wp.array(dtype=wp.quat, ndim=1)
  body_mass: wp.array(dtype=wp.float32, ndim=1)
  body_subtreemass: wp.array(dtype=wp.float32, ndim=1)
  subtree_mass: wp.array(dtype=wp.float32, ndim=1)
  body_inertia: wp.array(dtype=wp.vec3, ndim=1)
  body_invweight0: wp.array(dtype=wp.float32, ndim=2)
  body_contype: wp.array(dtype=wp.int32, ndim=1)
  body_conaffinity: wp.array(dtype=wp.int32, ndim=1)
  body_gravcomp: wp.array(dtype=wp.float32, ndim=1)
  jnt_type: wp.array(dtype=wp.int32, ndim=1)
  jnt_qposadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_dofadr: wp.array(dtype=wp.int32, ndim=1)
  jnt_bodyid: wp.array(dtype=wp.int32, ndim=1)
  jnt_limited: wp.array(dtype=wp.int32, ndim=1)
  jnt_actfrclimited: wp.array(dtype=wp.bool, ndim=1)
  jnt_solref: wp.array(dtype=wp.vec2, ndim=1)
  jnt_solimp: wp.array(dtype=vec5, ndim=1)
  jnt_pos: wp.array(dtype=wp.vec3, ndim=1)
  jnt_axis: wp.array(dtype=wp.vec3, ndim=1)
  jnt_stiffness: wp.array(dtype=wp.float32, ndim=1)
  jnt_range: wp.array(dtype=wp.float32, ndim=2)
  jnt_actfrcrange: wp.array(dtype=wp.vec2, ndim=1)
  jnt_margin: wp.array(dtype=wp.float32, ndim=1)
  jnt_limited_slide_hinge_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  jnt_limited_ball_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  jnt_actgravcomp: wp.array(dtype=wp.int32, ndim=1)
  dof_bodyid: wp.array(dtype=wp.int32, ndim=1)
  dof_jntid: wp.array(dtype=wp.int32, ndim=1)
  dof_parentid: wp.array(dtype=wp.int32, ndim=1)
  dof_Madr: wp.array(dtype=wp.int32, ndim=1)
  dof_armature: wp.array(dtype=wp.float32, ndim=1)
  dof_damping: wp.array(dtype=wp.float32, ndim=1)
  dof_invweight0: wp.array(dtype=wp.float32, ndim=1)
  dof_frictionloss: wp.array(dtype=wp.float32, ndim=1)
  dof_solimp: wp.array(dtype=vec5, ndim=1)
  dof_solref: wp.array(dtype=wp.vec2, ndim=1)
  dof_tri_row: wp.array(dtype=wp.int32, ndim=1)  # warp only
  dof_tri_col: wp.array(dtype=wp.int32, ndim=1)  # warp only
  geom_type: wp.array(dtype=wp.int32, ndim=1)
  geom_contype: wp.array(dtype=wp.int32, ndim=1)
  geom_conaffinity: wp.array(dtype=wp.int32, ndim=1)
  geom_condim: wp.array(dtype=wp.int32, ndim=1)
  geom_bodyid: wp.array(dtype=wp.int32, ndim=1)
  geom_dataid: wp.array(dtype=wp.int32, ndim=1)
  geom_priority: wp.array(dtype=wp.int32, ndim=1)
  geom_solmix: wp.array(dtype=wp.float32, ndim=1)
  geom_solref: wp.array(dtype=wp.vec2, ndim=1)
  geom_solimp: wp.array(dtype=vec5, ndim=1)
  geom_size: wp.array(dtype=wp.vec3, ndim=1)
  geom_aabb: wp.array(dtype=wp.vec3, ndim=2)
  geom_rbound: wp.array(dtype=wp.float32, ndim=1)
  geom_pos: wp.array(dtype=wp.vec3, ndim=1)
  geom_quat: wp.array(dtype=wp.quat, ndim=1)
  geom_friction: wp.array(dtype=wp.vec3, ndim=1)
  geom_margin: wp.array(dtype=wp.float32, ndim=1)
  geom_gap: wp.array(dtype=wp.float32, ndim=1)
  site_bodyid: wp.array(dtype=wp.int32, ndim=1)
  site_pos: wp.array(dtype=wp.vec3, ndim=1)
  site_quat: wp.array(dtype=wp.quat, ndim=1)
  cam_mode: wp.array(dtype=wp.int32, ndim=1)
  cam_bodyid: wp.array(dtype=wp.int32, ndim=1)
  cam_targetbodyid: wp.array(dtype=wp.int32, ndim=1)
  cam_pos: wp.array(dtype=wp.vec3, ndim=1)
  cam_quat: wp.array(dtype=wp.quat, ndim=1)
  cam_poscom0: wp.array(dtype=wp.vec3, ndim=1)
  cam_pos0: wp.array(dtype=wp.vec3, ndim=1)
  cam_fovy: wp.array(dtype=wp.float32, ndim=1)
  cam_resolution: wp.array(dtype=wp.vec2i, ndim=1)
  cam_sensorsize: wp.array(dtype=wp.vec2f, ndim=1)
  cam_intrinsic: wp.array(dtype=wp.vec4f, ndim=1)
  light_mode: wp.array(dtype=wp.int32, ndim=1)
  light_bodyid: wp.array(dtype=wp.int32, ndim=1)
  light_targetbodyid: wp.array(dtype=wp.int32, ndim=1)
  light_pos: wp.array(dtype=wp.vec3, ndim=1)
  light_dir: wp.array(dtype=wp.vec3, ndim=1)
  light_poscom0: wp.array(dtype=wp.vec3, ndim=1)
  light_pos0: wp.array(dtype=wp.vec3, ndim=1)
  mesh_vertadr: wp.array(dtype=wp.int32, ndim=1)
  mesh_vertnum: wp.array(dtype=wp.int32, ndim=1)
  mesh_vert: wp.array(dtype=wp.vec3, ndim=1)
  eq_type: wp.array(dtype=wp.int32, ndim=1)
  eq_obj1id: wp.array(dtype=wp.int32, ndim=1)
  eq_obj2id: wp.array(dtype=wp.int32, ndim=1)
  eq_objtype: wp.array(dtype=wp.int32, ndim=1)
  eq_active0: wp.array(dtype=wp.bool, ndim=1)
  eq_solref: wp.array(dtype=wp.vec2, ndim=1)
  eq_solimp: wp.array(dtype=vec5, ndim=1)
  eq_data: wp.array(dtype=vec11, ndim=1)
  eq_connect_adr: wp.array(dtype=wp.int32, ndim=1)
  eq_wld_adr: wp.array(dtype=wp.int32, ndim=1)
  eq_jnt_adr: wp.array(dtype=wp.int32, ndim=1)
  eq_ten_adr: wp.array(dtype=wp.int32, ndim=1)
  actuator_trntype: wp.array(dtype=wp.int32, ndim=1)
  actuator_dyntype: wp.array(dtype=wp.int32, ndim=1)
  actuator_gaintype: wp.array(dtype=wp.int32, ndim=1)
  actuator_biastype: wp.array(dtype=wp.int32, ndim=1)
  actuator_trnid: wp.array(dtype=wp.int32, ndim=2)
  actuator_actadr: wp.array(dtype=wp.int32, ndim=1)
  actuator_actnum: wp.array(dtype=wp.int32, ndim=1)
  actuator_ctrllimited: wp.array(dtype=wp.bool, ndim=1)
  actuator_forcelimited: wp.array(dtype=wp.bool, ndim=1)
  actuator_actlimited: wp.array(dtype=wp.bool, ndim=1)
  actuator_dynprm: wp.array(dtype=vec10f, ndim=1)
  actuator_gainprm: wp.array(dtype=vec10f, ndim=1)
  actuator_biasprm: wp.array(dtype=vec10f, ndim=1)
  actuator_ctrlrange: wp.array(dtype=wp.vec2, ndim=1)
  actuator_forcerange: wp.array(dtype=wp.vec2, ndim=1)
  actuator_actrange: wp.array(dtype=wp.vec2, ndim=1)
  actuator_gear: wp.array(dtype=wp.spatial_vector, ndim=1)
  exclude_signature: wp.array(dtype=wp.int32, ndim=1)
  actuator_affine_bias_gain: bool  # warp only
  nxn_geom_pair: wp.array(dtype=wp.vec2i, ndim=1)  # warp only
  nxn_pairid: wp.array(dtype=wp.int32, ndim=1)  # warp only

  # predefined geom pairs for collision detection
  pair_dim: wp.array(dtype=wp.int32, ndim=1)
  pair_geom1: wp.array(dtype=wp.int32, ndim=1)
  pair_geom2: wp.array(dtype=wp.int32, ndim=1)
  pair_solref: wp.array(dtype=wp.vec2, ndim=1)
  pair_solreffriction: wp.array(dtype=wp.vec2, ndim=1)
  pair_solimp: wp.array(dtype=vec5, ndim=1)
  pair_margin: wp.array(dtype=wp.float32, ndim=1)
  pair_gap: wp.array(dtype=wp.float32, ndim=1)
  pair_friction: wp.array(dtype=vec5, ndim=1)
  condim_max: int  # warp only
  tendon_adr: wp.array(dtype=wp.int32, ndim=1)
  tendon_num: wp.array(dtype=wp.int32, ndim=1)
  tendon_limited: wp.array(dtype=wp.int32, ndim=1)
  tendon_limited_adr: wp.array(dtype=wp.int32, ndim=1)
  tendon_solref_lim: wp.array(dtype=wp.vec2f, ndim=1)
  tendon_solimp_lim: wp.array(dtype=vec5, ndim=1)
  tendon_range: wp.array(dtype=wp.vec2f, ndim=1)
  tendon_margin: wp.array(dtype=wp.float32, ndim=1)
  tendon_length0: wp.array(dtype=wp.float32, ndim=1)
  tendon_invweight0: wp.array(dtype=wp.float32, ndim=1)
  wrap_objid: wp.array(dtype=wp.int32, ndim=1)
  wrap_prm: wp.array(dtype=wp.float32, ndim=1)
  wrap_type: wp.array(dtype=wp.int32, ndim=1)
  tendon_jnt_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  tendon_site_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  tendon_site_pair_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  ten_wrapadr_site: wp.array(dtype=wp.int32, ndim=1)  # warp only
  ten_wrapnum_site: wp.array(dtype=wp.int32, ndim=1)  # warp only
  wrap_jnt_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  wrap_site_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  wrap_site_pair_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  sensor_type: wp.array(dtype=wp.int32, ndim=1)
  sensor_datatype: wp.array(dtype=wp.int32, ndim=1)
  sensor_objtype: wp.array(dtype=wp.int32, ndim=1)
  sensor_objid: wp.array(dtype=wp.int32, ndim=1)
  sensor_reftype: wp.array(dtype=wp.int32, ndim=1)
  sensor_refid: wp.array(dtype=wp.int32, ndim=1)
  sensor_dim: wp.array(dtype=wp.int32, ndim=1)
  sensor_adr: wp.array(dtype=wp.int32, ndim=1)
  sensor_cutoff: wp.array(dtype=wp.float32, ndim=1)
  sensor_pos_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  sensor_vel_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  sensor_acc_adr: wp.array(dtype=wp.int32, ndim=1)  # warp only
  sensor_subtree_vel: bool  # warp only
  sensor_rne_postconstraint: bool  # warp only
  mocap_bodyid: wp.array(dtype=wp.int32, ndim=1)  # warp only


@wp.struct
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

  dist: wp.array(dtype=wp.float32, ndim=1)
  pos: wp.array(dtype=wp.vec3f, ndim=1)
  frame: wp.array(dtype=wp.mat33f, ndim=1)
  includemargin: wp.array(dtype=wp.float32, ndim=1)
  friction: wp.array(dtype=vec5, ndim=1)
  solref: wp.array(dtype=wp.vec2f, ndim=1)
  solreffriction: wp.array(dtype=wp.vec2f, ndim=1)
  solimp: wp.array(dtype=vec5, ndim=1)
  dim: wp.array(dtype=wp.int32, ndim=1)
  geom: wp.array(dtype=wp.vec2i, ndim=1)
  efc_address: wp.array(dtype=wp.int32, ndim=2)
  worldid: wp.array(dtype=wp.int32, ndim=1)


@wp.struct
class Data:
  """Dynamic state that updates each step.

  Attributes:
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
    nworld: number of worlds                                    ()
    nconmax: maximum number of contacts                         ()
    njmax: maximum number of constraints                        ()
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
    qM_integration: temporary array for integration             (same shape as qM)
    qLD_integration: temporary array for integration            (same shape as qLD)
    qLDiagInv_integration: temporary array for integration      (nworld, nv)
    boxes_sorted: min, max of sorted bounding boxes             (nworld, ngeom, 2)
    box_projections_lower: broadphase context                   (2*nworld, ngeom)
    box_projections_upper: broadphase context                   (nworld, ngeom)
    box_sorting_indexer: broadphase context                     (2*nworld, ngeom)
    ranges: broadphase context                                  (nworld, ngeom)
    cumulative_sum: broadphase context                          (nworld*ngeom,)
    segment_indices: broadphase context                         (nworld+1,)
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

  ncon: wp.array(dtype=wp.int32, ndim=1)
  ne: wp.array(dtype=wp.int32, ndim=1)
  ne_connect: wp.array(dtype=wp.int32, ndim=1)  # warp only
  ne_weld: wp.array(dtype=wp.int32, ndim=1)  # warp only
  ne_jnt: wp.array(dtype=wp.int32, ndim=1)  # warp only
  ne_ten: wp.array(dtype=wp.int32, ndim=1)  # warp only
  nf: wp.array(dtype=wp.int32, ndim=1)
  nl: wp.array(dtype=wp.int32, ndim=1)
  nefc: wp.array(dtype=wp.int32, ndim=1)
  time: wp.array(dtype=wp.float32, ndim=1)
  qpos: wp.array(dtype=wp.float32, ndim=2)
  qvel: wp.array(dtype=wp.float32, ndim=2)
  act: wp.array(dtype=wp.float32, ndim=2)
  qacc_warmstart: wp.array(dtype=wp.float32, ndim=2)
  ctrl: wp.array(dtype=wp.float32, ndim=2)
  qfrc_applied: wp.array(dtype=wp.float32, ndim=2)
  xfrc_applied: wp.array(dtype=wp.spatial_vector, ndim=2)
  eq_active: wp.array(dtype=wp.bool, ndim=2)
  mocap_pos: wp.array(dtype=wp.vec3, ndim=2)
  mocap_quat: wp.array(dtype=wp.quat, ndim=2)
  qacc: wp.array(dtype=wp.float32, ndim=2)
  act_dot: wp.array(dtype=wp.float32, ndim=2)
  xpos: wp.array(dtype=wp.vec3, ndim=2)
  xquat: wp.array(dtype=wp.quat, ndim=2)
  xmat: wp.array(dtype=wp.mat33, ndim=2)
  xipos: wp.array(dtype=wp.vec3, ndim=2)
  ximat: wp.array(dtype=wp.mat33, ndim=2)
  xanchor: wp.array(dtype=wp.vec3, ndim=2)
  xaxis: wp.array(dtype=wp.vec3, ndim=2)
  geom_xpos: wp.array(dtype=wp.vec3, ndim=2)
  geom_xmat: wp.array(dtype=wp.mat33, ndim=2)
  site_xpos: wp.array(dtype=wp.vec3, ndim=2)
  site_xmat: wp.array(dtype=wp.mat33, ndim=2)
  cam_xpos: wp.array(dtype=wp.vec3, ndim=2)
  cam_xmat: wp.array(dtype=wp.mat33, ndim=2)
  light_xpos: wp.array(dtype=wp.vec3, ndim=2)
  light_xdir: wp.array(dtype=wp.vec3, ndim=2)
  subtree_com: wp.array(dtype=wp.vec3, ndim=2)
  cdof: wp.array(dtype=wp.spatial_vector, ndim=2)
  cinert: wp.array(dtype=vec10, ndim=2)
  actuator_length: wp.array(dtype=wp.float32, ndim=2)
  actuator_moment: wp.array(dtype=wp.float32, ndim=3)
  crb: wp.array(dtype=vec10, ndim=2)
  qM: wp.array(dtype=wp.float32, ndim=3)
  qLD: wp.array(dtype=wp.float32, ndim=3)
  qLDiagInv: wp.array(dtype=wp.float32, ndim=2)
  ten_velocity: wp.array(dtype=wp.float32, ndim=2)
  actuator_velocity: wp.array(dtype=wp.float32, ndim=2)
  cvel: wp.array(dtype=wp.spatial_vector, ndim=2)
  cdof_dot: wp.array(dtype=wp.spatial_vector, ndim=2)
  qfrc_bias: wp.array(dtype=wp.float32, ndim=2)
  qfrc_spring: wp.array(dtype=wp.float32, ndim=2)
  qfrc_damper: wp.array(dtype=wp.float32, ndim=2)
  qfrc_gravcomp: wp.array(dtype=wp.float32, ndim=2)
  qfrc_passive: wp.array(dtype=wp.float32, ndim=2)
  subtree_linvel: wp.array(dtype=wp.vec3, ndim=2)
  subtree_angmom: wp.array(dtype=wp.vec3, ndim=2)
  subtree_bodyvel: wp.array(dtype=wp.spatial_vector, ndim=2)  # warp only
  actuator_force: wp.array(dtype=wp.float32, ndim=2)
  qfrc_actuator: wp.array(dtype=wp.float32, ndim=2)
  qfrc_smooth: wp.array(dtype=wp.float32, ndim=2)
  qacc_smooth: wp.array(dtype=wp.float32, ndim=2)
  qfrc_constraint: wp.array(dtype=wp.float32, ndim=2)
  contact: Contact
  efc: Constraint
  nworld: int
  nconmax: int
  njmax: int
  rne_cacc: wp.array(dtype=wp.spatial_vector, ndim=2)
  rne_cfrc: wp.array(dtype=wp.spatial_vector, ndim=2)
  qpos_t0: wp.array(dtype=wp.float32, ndim=2)
  qvel_t0: wp.array(dtype=wp.float32, ndim=2)
  act_t0: wp.array(dtype=wp.float32, ndim=2)
  qvel_rk: wp.array(dtype=wp.float32, ndim=2)
  qacc_rk: wp.array(dtype=wp.float32, ndim=2)
  act_dot_rk: wp.array(dtype=wp.float32, ndim=2)
  qfrc_integration: wp.array(dtype=wp.float32, ndim=2)
  qacc_integration: wp.array(dtype=wp.float32, ndim=2)
  act_vel_integration: wp.array(dtype=wp.float32, ndim=2)
  qM_integration: wp.array(dtype=wp.float32, ndim=3)
  qLD_integration: wp.array(dtype=wp.float32, ndim=3)
  qLDiagInv_integration: wp.array(dtype=wp.float32, ndim=2)

  # sweep-and-prune broadphase
  sap_projection_lower: wp.array(dtype=wp.float32, ndim=2)
  sap_projection_upper: wp.array(dtype=wp.float32, ndim=2)
  sap_sort_index: wp.array(dtype=wp.int32, ndim=2)
  sap_range: wp.array(dtype=wp.int32, ndim=2)
  sap_cumulative_sum: wp.array(dtype=wp.int32, ndim=1)
  sap_segment_index: wp.array(dtype=wp.int32, ndim=1)

  # collision driver
  collision_pair: wp.array(dtype=wp.vec2i, ndim=1)
  collision_pairid: wp.array(dtype=wp.int32, ndim=1)
  collision_worldid: wp.array(dtype=wp.int32, ndim=1)
  ncollision: wp.array(dtype=wp.int32, ndim=1)

  # rne_postconstraint
  cacc: wp.array(dtype=wp.spatial_vector, ndim=2)
  cfrc_int: wp.array(dtype=wp.spatial_vector, ndim=2)
  cfrc_ext: wp.array(dtype=wp.spatial_vector, ndim=2)

  # tendon
  ten_length: wp.array(dtype=wp.float32, ndim=2)
  ten_J: wp.array(dtype=wp.float32, ndim=3)
  ten_wrapadr: wp.array(dtype=wp.int32, ndim=2)
  ten_wrapnum: wp.array(dtype=wp.int32, ndim=2)
  wrap_obj: wp.array(dtype=wp.vec2i, ndim=2)
  wrap_xpos: wp.array(dtype=wp.spatial_vector, ndim=2)

  # sensors
  sensordata: wp.array(dtype=wp.float32, ndim=2)
