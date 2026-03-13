"""
Autodifferentiation coordination for MuJoCo Warp.

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

from typing import Callable, Optional, Sequence

import warp as wp

from mujoco_warp._src import io
from mujoco_warp._src.forward import forward
from mujoco_warp._src.forward import step
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model

# ensure custom adjoints are registered before any tape recording
from mujoco_warp._src import adjoint as _adjoint  # noqa: F401

# data fields that require gradient tracking for smooth dynamics
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


def enable_grad(d: Data, fields: Optional[Sequence[str]] = None) -> None:
  # enable gradient tracking on Data arrays
  if fields is None:
    fields = SMOOTH_GRAD_FIELDS
  for name in fields:
    arr = getattr(d, name, None)
    if arr is not None and isinstance(arr, wp.array):
      arr.requires_grad = True


#disable gradient tracking on all Data arrays
def disable_grad(d: Data) -> None:
  for name in SMOOTH_GRAD_FIELDS:
    arr = getattr(d, name, None)
    if arr is not None and isinstance(arr, wp.array):
      arr.requires_grad = False


def make_diff_data(
  mjm,
  nworld: int = 1,
  grad_fields: Optional[Sequence[str]] = None,
  **kwargs,
) -> Data:
  # create a Data object with gradient tracking enabled
  d = io.make_data(mjm, nworld=nworld, **kwargs)
  enable_grad(d, fields=grad_fields)
  return d


def diff_step(
  m: Model,
  d: Data,
  loss_fn: Callable[[Model, Data], wp.array],
) -> wp.Tape:
  # run a differentiable physics step
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
  # run differentiable forward dynamics (no integration)
  tape = wp.Tape()
  with tape:
    forward(m, d)
    loss = loss_fn(m, d)
  tape.backward(loss=loss)
  return tape
