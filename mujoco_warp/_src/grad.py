"""Autodifferentiation coordination for MuJoCo Warp.

This module provides utilities for enabling Warp's tape-based reverse-mode
automatic differentiation through the MuJoCo Warp physics pipeline.

Usage::

    import mujoco_warp as mjw

    d = mjw.make_diff_data(mjm)  # Data with gradient tracking
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(loss_kernel, dim=1, inputs=[d.xpos, target, loss])
    tape.backward(loss=loss)
    grad_ctrl = d.ctrl.grad
"""

import warnings
from typing import Callable, Optional, Sequence

import warp as wp

from mujoco_warp._src import ad_flags
from mujoco_warp._src import adjoint as _adjoint  # noqa: F401 (register custom adjoints)
from mujoco_warp._src import io
from mujoco_warp._src import sensor as _sensor
from mujoco_warp._src.forward import forward
from mujoco_warp._src.forward import step
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import SensorType
from mujoco_warp._src.types import SolverType

SMOOTH_GRAD_FIELDS: tuple = (
  # primary state, user-controlled inputs
  "qpos",
  "qvel",
  "ctrl",
  "act",
  "mocap_pos",
  "mocap_quat",
  "xfrc_applied",
  "qfrc_applied",
  # position-dependent outputs
  "xpos",
  "xquat",
  "xmat",
  "xipos",
  "ximat",
  "xanchor",
  "xaxis",
  "geom_xpos",
  "geom_xmat",
  "site_xpos",
  "site_xmat",
  "subtree_com",
  "cinert",
  "crb",
  "cdof",
  # Velocity-dependent outputs
  "cdof_dot",
  "cvel",
  "subtree_linvel",
  "subtree_angmom",
  "actuator_velocity",
  "ten_velocity",
  # body-level intermediate quantities
  "cacc",
  "cfrc_int",
  "cfrc_ext",
  # force/acceleration outputs
  "qfrc_bias",
  "qfrc_spring",
  "qfrc_damper",
  "qfrc_gravcomp",
  "qfrc_fluid",
  "qfrc_passive",
  "qfrc_actuator",
  "qfrc_smooth",
  "qacc",
  "qacc_smooth",
  "actuator_force",
  "act_dot",
  # inertia matrix
  "M",
  "qLD",
  "qLDiagInv",
  # Tendon
  "ten_J",
  "ten_length",
  # actuator
  "actuator_length",
  "actuator_moment",
  # delayed-ctrl ring buffer
  "history",
  # sensor
  "sensordata",
)

SOLVER_GRAD_FIELDS: tuple = ("qfrc_constraint",)

# Sensor types whose sensordata slots carry gradients: smooth position- and
# velocity-stage sensors computed from differentiable kinematic quantities.
# All other types (rangefinder/geomdist ray+collision queries, joint/tendon
# limit sensors, energy, clock, insidesite, and every acceleration-stage
# sensor: touch/tactile/contact, accelerometer, force/torque, actuator force,
# frame accelerations) read non-differentiable inputs and return exactly zero
# gradient on their sensordata slots.
GRAD_SENSOR_TYPES: frozenset = frozenset(
  int(t)
  for t in (
    SensorType.MAGNETOMETER,
    SensorType.CAMPROJECTION,
    SensorType.JOINTPOS,
    SensorType.TENDONPOS,
    SensorType.ACTUATORPOS,
    SensorType.BALLQUAT,
    SensorType.FRAMEPOS,
    SensorType.FRAMEXAXIS,
    SensorType.FRAMEYAXIS,
    SensorType.FRAMEZAXIS,
    SensorType.FRAMEQUAT,
    SensorType.SUBTREECOM,
    SensorType.VELOCIMETER,
    SensorType.GYRO,
    SensorType.JOINTVEL,
    SensorType.TENDONVEL,
    SensorType.ACTUATORVEL,
    SensorType.BALLANGVEL,
    SensorType.FRAMELINVEL,
    SensorType.FRAMEANGVEL,
    SensorType.SUBTREELINVEL,
    SensorType.SUBTREEANGMOM,
  )
)

# Model parameter arrays whose gradients are known to be silently dropped by
# the manual packed-M/qLD adjoints (their only path to the dynamics runs
# through composite inertia and the mass-matrix factorization, which have
# hand-written adjoints that do not differentiate w.r.t. Model parameters).
DROPPED_MODEL_GRAD_FIELDS: tuple = ("body_mass", "body_inertia", "dof_armature")

COLLISION_GRAD_FIELDS: tuple = (
  # Contact geometry (written by smooth_recompute_contacts)
  "contact.dist",
  "contact.pos",
  "contact.frame",
  # Constraint arrays (written by smooth_contact_to_efc)
  "efc.J",
  "efc.pos",
  "efc.D",
  "efc.aref",
  "efc.vel",
)


def _resolve_field(d: Data, name: str):
  """Resolve a field name, supporting dotted paths like 'contact.dist'."""
  if "." in name:
    obj_name, field_name = name.split(".", 1)
    obj = getattr(d, obj_name, None)
    return getattr(obj, field_name, None) if obj else None
  return getattr(d, name, None)


# sensor.py is not in ad_flags._AD_MODULES (only its smooth position/velocity
# kernels differentiate; the rest opt out per kernel), so its module option is
# flipped here alongside the main AD modules.
_sensor_ad_enabled = ad_flags.ad_enabled()


def _enable_sensor_ad() -> None:
  global _sensor_ad_enabled
  if _sensor_ad_enabled:
    return
  _sensor_ad_enabled = True
  wp.set_module_options({"enable_backward": True}, module=_sensor)


def _warn_if_no_differentiable_sensors(mjm) -> None:
  """Warn when sensordata is grad-tracked but no sensor type carries gradients."""
  if mjm.nsensor == 0:
    return
  if any(int(t) in GRAD_SENSOR_TYPES for t in mjm.sensor_type):
    return
  warnings.warn(
    "sensordata is gradient-tracked but this model only contains sensor types with no gradient support "
    "(see mujoco_warp GRAD_SENSOR_TYPES); a loss on sensordata will produce zero gradients.",
    stacklevel=3,
  )


def _warn_if_nondifferentiable_flex(mjm) -> None:
  """Warn for interpolated flexes, whose passive-force gradients are unsupported."""
  if getattr(mjm, "nflex", 0) and any(mjm.flex_interp != 0):
    warnings.warn(
      "Model contains trilinear/quadratic-interpolated flexes; their passive-force gradients are not supported "
      "and will be missing or zero.",
      stacklevel=3,
    )


def _warn_if_flex_contacts_possible(mjm) -> None:
  """Warn that flex contact rows keep their discrete geometry under a tape."""
  import mujoco

  if getattr(mjm, "nflex", 0) and not (mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT):
    warnings.warn(
      "Model contains flexes with contact enabled: flex contact rows are excluded from the differentiable "
      "contact replay (they keep their discrete geometry), so gradients through flex contacts are incomplete "
      "and can be biased.",
      stacklevel=3,
    )


def enable_grad(d: Data, fields: Optional[Sequence[str]] = None, mjm=None) -> None:
  """Enables gradient tracking on Data arrays.

  When mjm is provided, also eagerly allocates the solver retained state
  (Newton Hessian, Cholesky factor, Jaref) used by the implicit-diff
  backward pass; otherwise it is lazily allocated on the first solve.
  Also enables backward-kernel compilation for gradient-path modules
  (see ad_flags.enable_ad); call this before the first step to avoid a
  module recompile.

  Sensor gradients: sensordata is gradient-tracked, but only smooth
  position/velocity-stage sensor types carry gradients (the exact set is
  GRAD_SENSOR_TYPES: frame pos/quat/axis, joint/tendon/actuator pos and
  vel, ball quat/angvel, velocimeter, gyro, magnetometer, camprojection,
  subtree com/linvel/angmom). Contact-dependent sensors (touch, tactile,
  contact), acceleration/force-stage sensors (accelerometer, force,
  torque, actuator force, frame accelerations), rangefinder, geom
  distance, limit and energy sensors return exactly zero gradient on
  their sensordata slots. When mjm is provided and the model contains
  only unsupported sensor types, a warning is emitted.

  Model parameter gradients: setting requires_grad manually on Model
  arrays consumed by auto-differentiated kernels yields gradients (
  verified for actuator_gainprm, jnt_stiffness, dof_damping, geom_size),
  but parameters whose only path to the dynamics runs through the
  manually-adjointed mass-matrix chain are silently dropped (
  DROPPED_MODEL_GRAD_FIELDS: body_mass, body_inertia, dof_armature).
  diff_step/diff_forward warn when these are gradient-tracked.
  """
  ad_flags.enable_ad()
  _enable_sensor_ad()
  if mjm is not None:
    _warn_if_no_differentiable_sensors(mjm)
    _warn_if_nondifferentiable_flex(mjm)
    _warn_if_flex_contacts_possible(mjm)
  if fields is None:
    # Include the collision/solver fields so contact gradients work out of the box: the
    # implicit-diff backward needs efc.{J,D,aref,vel,pos} and qfrc_constraint gradient-tracked
    # to propagate dL/dctrl through an active contact (the aref term carries the Baumgarte
    # velocity, i.e. the contact's dissipation of qvel). Smooth-only tracking silently drops
    # that path and returns a free-body gradient through contact.
    fields = SMOOTH_GRAD_FIELDS + SOLVER_GRAD_FIELDS + COLLISION_GRAD_FIELDS
  for name in fields:
    arr = _resolve_field(d, name)
    if arr is not None and isinstance(arr, wp.array):
      arr.requires_grad = True
  if mjm is not None:
    io.allocate_solver_retained_for_model(mjm, d, grad=True)


def disable_grad(d: Data, fields: Optional[Sequence[str]] = None, mjm=None) -> None:
  """Disables gradient tracking on Data arrays.

  When mjm is provided, also frees the solver retained AD state so the
  forward-only path carries no extra memory.
  """
  if fields is None:
    fields = SMOOTH_GRAD_FIELDS + SOLVER_GRAD_FIELDS + COLLISION_GRAD_FIELDS
  for name in fields:
    arr = _resolve_field(d, name)
    if arr is not None and isinstance(arr, wp.array):
      arr.requires_grad = False
  if mjm is not None:
    io.allocate_solver_retained_for_model(mjm, d, grad=False)


def make_diff_data(
  mjm,
  nworld: int = 1,
  grad_fields: Optional[Sequence[str]] = None,
  **kwargs,
) -> Data:
  """Creates a Data object with gradient tracking enabled.

  See enable_grad for the supported gradient scope: in particular only
  the smooth sensor types in GRAD_SENSOR_TYPES carry sensordata
  gradients, and Model parameters listed in DROPPED_MODEL_GRAD_FIELDS
  do not receive gradients even when manually gradient-tracked.
  """
  d = io.make_data(mjm, nworld=nworld, **kwargs)
  enable_grad(d, fields=grad_fields, mjm=mjm)
  return d


def enable_smooth_adjoint(
  d: Data,
  friction_viscosity: float = 10.0,
  friction_scale: float = 0.01,
  friction_bypass_kf: float = 0.0,
  free_body_adjoint: bool = False,
  penalty_damping_alpha: float = 0.0,
  friction_surrogate_adjoint: bool = False,
  friction_surrogate_alpha: float = 0.0,
) -> None:
  """Enable smooth constraint adjoint for friction gradient signal.

  Modifies the backward pass to build a smooth Hessian where friction
  constraint stiffness is reduced (for active/QUADRATIC constraints) and
  a viscous friction term is added (for satisfied/static constraints).
  The forward physics is unchanged.

  Args:
    d: Data object (must have gradient tracking enabled).
    friction_viscosity: D value added for SATISFIED friction constraints.
        Higher values give stronger gradient signal at zero velocity.
    friction_scale: Scale factor for QUADRATIC friction constraint D in
        the adjoint Hessian. Lower values reduce friction stiffness more,
        giving larger tangential gradients.
    friction_bypass_kf: Scale for friction gradient bypass. After the
        Hessian solve, restores tangential gradient components that were
        attenuated by H^{-1}. 0=off, 1=full bypass, >1=amplified.
    free_body_adjoint: When True, replaces the solver adjoint entirely
        with v = M^{-1} * adj_qacc (free-body assumption). Eliminates
        all constraint attenuation. Overrides friction_scale/bypass_kf.
    penalty_damping_alpha: Friction damping factor for penalty-model
        adjoint. Attenuates v in friction directions by (1-alpha) per
        face, mimicking dflex's bounded BPTT eigenvalues. Implies
        free-body base (M^{-1}). 0=off, 0.1-0.3=typical.
    friction_surrogate_adjoint: When True, keeps the smooth/Newton solve
        as the baseline but replaces friction-face backward projections
        with a damped tangential recovery toward the free-body solution.
        This preserves solver-informed normal-contact handling while using
        a training-oriented surrogate
        in tangential directions.
    friction_surrogate_alpha: Tangential damping factor for the friction
        surrogate branch. 0=full tangential recovery, 0.9=10% recovery,
        1=disabled. Values in 0.8-0.95 are the intended range for
        soft-contact ant experiments.
  """
  d.smooth_adjoint = 1
  d.smooth_friction_viscosity = friction_viscosity
  d.smooth_friction_scale = friction_scale
  d.smooth_friction_bypass_kf = friction_bypass_kf
  d.smooth_free_body_adjoint = free_body_adjoint
  d.smooth_penalty_damping_alpha = penalty_damping_alpha
  d.smooth_friction_surrogate_adjoint = friction_surrogate_adjoint
  d.smooth_friction_surrogate_alpha = friction_surrogate_alpha


def disable_smooth_adjoint(d: Data) -> None:
  """Disable smooth constraint adjoint, reverting to standard implicit diff."""
  d.smooth_adjoint = 0


def _warn_if_cg_solver(m: Model, d: Data):
  """Warn if CG solver is used with constraints (gradients will be zero)."""
  if d.njmax > 0 and m.opt.solver != SolverType.NEWTON:
    warnings.warn(
      "Differentiable solver requires Newton. CG solver gradients through constraints will be zero.",
      stacklevel=3,
    )


def _warn_if_dropped_model_grads(m: Model):
  """Warn when requires_grad is set on Model arrays whose gradients are dropped."""
  tracked = [name for name in DROPPED_MODEL_GRAD_FIELDS if getattr(getattr(m, name, None), "requires_grad", False)]
  if tracked:
    warnings.warn(
      f"requires_grad is set on Model arrays {tracked}, but their gradients are dropped by the manual "
      "mass-matrix adjoints and will be zero.",
      stacklevel=3,
    )


def diff_step(
  m: Model,
  d: Data,
  loss_fn: Callable[[Model, Data], wp.array],
) -> wp.Tape:
  """Runs a differentiable physics step."""
  _warn_if_cg_solver(m, d)
  _warn_if_dropped_model_grads(m)
  tape = wp.Tape()
  with tape:
    step(m, d)
    loss = loss_fn(m, d)
  tape.backward(loss=loss)
  return tape


def diff_forward(
  m: Model,
  d: Data,
  loss_fn: Callable[[Model, Data], wp.array],
) -> wp.Tape:
  """Runs differentiable forward dynamics (no integration)."""
  _warn_if_cg_solver(m, d)
  _warn_if_dropped_model_grads(m)
  tape = wp.Tape()
  with tape:
    forward(m, d)
    loss = loss_fn(m, d)
  tape.backward(loss=loss)
  return tape
