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
"""VJPs for constraint, contact, and inertial model parameters."""

import warp as wp

from mujoco_warp._src import adjoint_recompute
from mujoco_warp._src import smooth_adjoint
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": True})


@cache_kernel
def _accum_constraint_params(
  jnt_solref_grad: bool,
  jnt_solimp_grad: bool,
  dof_solref_grad: bool,
  dof_solimp_grad: bool,
  dof_frictionloss_grad: bool,
  eq_solref_grad: bool,
  eq_solimp_grad: bool,
):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    res_jnt_solref: wp.array2d[wp.vec2],
    res_jnt_solimp: wp.array2d[vec5],
    res_dof_solref: wp.array2d[wp.vec2],
    res_dof_solimp: wp.array2d[vec5],
    res_dof_frictionloss: wp.array2d[float],
    res_eq_solref: wp.array2d[wp.vec2],
    res_eq_solimp: wp.array2d[vec5],
    # Out:
    jnt_solref_grad_out: wp.array2d[wp.vec2],
    jnt_solimp_grad_out: wp.array2d[vec5],
    dof_solref_grad_out: wp.array2d[wp.vec2],
    dof_solimp_grad_out: wp.array2d[vec5],
    dof_frictionloss_grad_out: wp.array2d[float],
    eq_solref_grad_out: wp.array2d[wp.vec2],
    eq_solimp_grad_out: wp.array2d[vec5],
  ):
    worldid, paramid = wp.tid()
    if wp.static(jnt_solref_grad):
      if worldid < res_jnt_solref.shape[0] and paramid < res_jnt_solref.shape[1]:
        jnt_solref_grad_out[worldid, paramid] -= res_jnt_solref[worldid, paramid]
    if wp.static(jnt_solimp_grad):
      if worldid < res_jnt_solimp.shape[0] and paramid < res_jnt_solimp.shape[1]:
        jnt_solimp_grad_out[worldid, paramid] -= res_jnt_solimp[worldid, paramid]
    if wp.static(dof_solref_grad):
      if worldid < res_dof_solref.shape[0] and paramid < res_dof_solref.shape[1]:
        dof_solref_grad_out[worldid, paramid] -= res_dof_solref[worldid, paramid]
    if wp.static(dof_solimp_grad):
      if worldid < res_dof_solimp.shape[0] and paramid < res_dof_solimp.shape[1]:
        dof_solimp_grad_out[worldid, paramid] -= res_dof_solimp[worldid, paramid]
    if wp.static(dof_frictionloss_grad):
      if worldid < res_dof_frictionloss.shape[0] and paramid < res_dof_frictionloss.shape[1]:
        dof_frictionloss_grad_out[worldid, paramid] -= res_dof_frictionloss[worldid, paramid]
    if wp.static(eq_solref_grad):
      if worldid < res_eq_solref.shape[0] and paramid < res_eq_solref.shape[1]:
        eq_solref_grad_out[worldid, paramid] -= res_eq_solref[worldid, paramid]
    if wp.static(eq_solimp_grad):
      if worldid < res_eq_solimp.shape[0] and paramid < res_eq_solimp.shape[1]:
        eq_solimp_grad_out[worldid, paramid] -= res_eq_solimp[worldid, paramid]

  return kernel


@cache_kernel
def _accum_inertial_params(body_mass_grad: bool, body_inertia_grad: bool, body_ipos_grad: bool, body_iquat_grad: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    res_body_mass: wp.array2d[float],
    res_body_inertia: wp.array2d[wp.vec3],
    res_body_ipos: wp.array2d[wp.vec3],
    res_body_iquat: wp.array2d[wp.quat],
    # Out:
    body_mass_grad_out: wp.array2d[float],
    body_inertia_grad_out: wp.array2d[wp.vec3],
    body_ipos_grad_out: wp.array2d[wp.vec3],
    body_iquat_grad_out: wp.array2d[wp.quat],
  ):
    worldid, bodyid = wp.tid()
    if wp.static(body_mass_grad):
      if worldid < res_body_mass.shape[0]:
        body_mass_grad_out[worldid, bodyid] -= res_body_mass[worldid, bodyid]
    if wp.static(body_inertia_grad):
      if worldid < res_body_inertia.shape[0]:
        body_inertia_grad_out[worldid, bodyid] -= res_body_inertia[worldid, bodyid]
    if wp.static(body_ipos_grad):
      if worldid < res_body_ipos.shape[0]:
        body_ipos_grad_out[worldid, bodyid] -= res_body_ipos[worldid, bodyid]
    if wp.static(body_iquat_grad):
      if worldid < res_body_iquat.shape[0]:
        body_iquat_grad_out[worldid, bodyid] -= res_body_iquat[worldid, bodyid]

  return kernel


# Constraint parameters.


def _constraint_param_pairs(m: Model, bc: BackwardContext):
  """Returns constraint parameters paired with their residual cotangents."""
  return (
    (m.jnt_solref, bc.res_jnt_solref),
    (m.jnt_solimp, bc.res_jnt_solimp),
    (m.dof_solref, bc.res_dof_solref),
    (m.dof_solimp, bc.res_dof_solimp),
    (m.dof_frictionloss, bc.res_dof_frictionloss),
    (m.eq_solref, bc.res_eq_solref),
    (m.eq_solimp, bc.res_eq_solimp),
  )


def constraint_params_backward(m: Model, bc: BackwardContext):
  """Applies the IFT sign to constraint parameter residuals."""
  params = _constraint_param_pairs(m, bc)
  param_grads = [param.requires_grad for param, _ in params]
  enabled = [param for param, _ in params if param.requires_grad]
  if not enabled:
    return
  wp.launch(
    _accum_constraint_params(*param_grads),
    dim=(max(param.shape[0] for param in enabled), max(param.shape[1] for param in enabled)),
    inputs=[residual for _, residual in params],
    outputs=[param.grad for param, _ in params],
  )


# Contact friction.


@wp.func
def accumulate_geom_friction(
  # Model:
  geom_priority: wp.array[int],
  geom_friction: wp.array2d[wp.vec3],
  # In:
  geom: wp.vec2i,
  worldid: int,
  res_contact_friction: vec5,
  # Out:
  geom_friction_grad: wp.array2d[wp.vec3],
):
  geom1 = geom[0]
  geom2 = geom[1]
  res_geom_friction = wp.vec3(
    res_contact_friction[0] + res_contact_friction[1],
    res_contact_friction[2],
    res_contact_friction[3] + res_contact_friction[4],
  )
  model_worldid = worldid % geom_friction.shape[0]
  priority1 = geom_priority[geom1]
  priority2 = geom_priority[geom2]
  friction1 = geom_friction[model_worldid, geom1]
  friction2 = geom_friction[model_worldid, geom2]
  grad1 = wp.vec3(0.0)
  grad2 = wp.vec3(0.0)
  for i in range(3):
    if priority1 > priority2 or (priority1 == priority2 and friction1[i] >= friction2[i]):
      grad1[i] = res_geom_friction[i]
    else:
      grad2[i] = res_geom_friction[i]
  wp.atomic_add(geom_friction_grad, model_worldid, geom1, -grad1)
  wp.atomic_add(geom_friction_grad, model_worldid, geom2, -grad2)


# Local smooth parameters.


@wp.kernel(enable_backward=True)
def _residual_smooth_local(
  # Model:
  dof_armature: wp.array2d[float],
  dof_damping: wp.array2d[float],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # Out:
  residual_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  residual_out[worldid, dofid] = (
    dof_armature[worldid % dof_armature.shape[0], dofid] * qacc_in[worldid, dofid]
    + dof_damping[worldid % dof_damping.shape[0], dofid] * qvel_in[worldid, dofid]
  )


@event_scope
def inertia_param_vjp(m: Model, d: Data, lam: wp.array2d, bc: BackwardContext | None = None):
  """Accumulates inertial parameter VJPs with the IFT sign."""
  body_mass_grad = m.body_mass.requires_grad
  body_inertia_grad = m.body_inertia.requires_grad
  body_ipos_grad = m.body_ipos.requires_grad
  body_iquat_grad = m.body_iquat.requires_grad
  if not (body_mass_grad or body_inertia_grad or body_ipos_grad or body_iquat_grad):
    return
  if bc is None:
    from mujoco_warp._src import adjoint

    bc = adjoint.create_backward_context(m, d)

  smooth_adjoint.rne_backward(m, d, lam, bc=bc)
  d.xipos.grad.zero_()
  d.ximat.grad.zero_()
  d.subtree_com.grad.zero_()
  bc.res_body_mass.zero_()
  bc.res_body_inertia.zero_()
  bc.res_body_ipos.zero_()
  bc.res_body_iquat.zero_()
  wp.launch(
    adjoint_recompute._cinert_recompute,
    dim=(d.nworld, m.nbody),
    inputs=[m.body_rootid, m.body_mass, m.body_inertia, d.xipos, d.ximat, d.subtree_com],
    outputs=[d.cinert],
    adj_inputs=[None, bc.res_body_mass, bc.res_body_inertia, d.xipos.grad, d.ximat.grad, d.subtree_com.grad],
    adj_outputs=[d.cinert.grad],
    adjoint=True,
  )
  if body_mass_grad:
    smooth_adjoint.gravcomp_mass_vjp(m, d, lam, bc.res_body_mass)

  if body_ipos_grad or body_iquat_grad:
    wp.launch(
      adjoint_recompute._inertial_frames_recompute,
      dim=(d.nworld, m.nbody),
      inputs=[m.body_ipos, m.body_iquat, d.xpos, d.xquat],
      outputs=[d.xipos, d.ximat],
      adj_inputs=[bc.res_body_ipos, bc.res_body_iquat, None, None],
      adj_outputs=[d.xipos.grad, d.ximat.grad],
      adjoint=True,
    )
  model_worlds = max(
    param.shape[0] for param in (m.body_mass, m.body_inertia, m.body_ipos, m.body_iquat) if param.requires_grad
  )
  wp.launch(
    _accum_inertial_params(body_mass_grad, body_inertia_grad, body_ipos_grad, body_iquat_grad),
    dim=(model_worlds, m.nbody),
    inputs=[bc.res_body_mass, bc.res_body_inertia, bc.res_body_ipos, bc.res_body_iquat],
    outputs=[m.body_mass.grad, m.body_inertia.grad, m.body_ipos.grad, m.body_iquat.grad],
  )
