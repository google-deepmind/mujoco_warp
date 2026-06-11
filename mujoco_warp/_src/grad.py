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

from mujoco_warp._src import adjoint as _adjoint  # noqa: F401 (register custom adjoints)
from mujoco_warp._src import io
from mujoco_warp._src.forward import forward
from mujoco_warp._src.forward import step
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
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
  "qM",
  "qLD",
  "qLDiagInv",
  # Tendon
  "ten_J",
  "ten_length",
  # actuator
  "actuator_length",
  "actuator_moment",
  # sensor
  "sensordata",
)

SOLVER_GRAD_FIELDS: tuple = ("qfrc_constraint",)

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


def enable_grad(d: Data, fields: Optional[Sequence[str]] = None) -> None:
  """Enables gradient tracking on Data arrays."""
  if fields is None:
    fields = SMOOTH_GRAD_FIELDS
  for name in fields:
    arr = _resolve_field(d, name)
    if arr is not None and isinstance(arr, wp.array):
      arr.requires_grad = True


def disable_grad(d: Data, fields: Optional[Sequence[str]] = None) -> None:
  """Disables gradient tracking on Data arrays."""
  if fields is None:
    fields = SMOOTH_GRAD_FIELDS + SOLVER_GRAD_FIELDS + COLLISION_GRAD_FIELDS
  for name in fields:
    arr = _resolve_field(d, name)
    if arr is not None and isinstance(arr, wp.array):
      arr.requires_grad = False


def make_diff_data(
  mjm,
  nworld: int = 1,
  grad_fields: Optional[Sequence[str]] = None,
  **kwargs,
) -> Data:
  """Creates a Data object with gradient tracking enabled."""
  d = io.make_data(mjm, nworld=nworld, **kwargs)
  enable_grad(d, fields=grad_fields)
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


def diff_step(
  m: Model,
  d: Data,
  loss_fn: Callable[[Model, Data], wp.array],
) -> wp.Tape:
  """Runs a differentiable physics step."""
  _warn_if_cg_solver(m, d)
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
  tape = wp.Tape()
  with tape:
    forward(m, d)
    loss = loss_fn(m, d)
  tape.backward(loss=loss)
  return tape
