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
"""VJPs for time integration, smooth dynamics, and forward kinematics."""

import dataclasses

import warp as wp

from mujoco_warp._src import adjoint_util
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import smooth
from mujoco_warp._src import smooth_adjoint
from mujoco_warp._src import smooth_kinematics_adjoint
from mujoco_warp._src import support
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import BackwardContext
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec10
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": True})


@wp.kernel
def _advance_state(
  # Model:
  opt_timestep: wp.array[float],
  jnt_type: wp.array[int],
  jnt_qposadr: wp.array[int],
  jnt_dofadr: wp.array[int],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # Data out:
  qpos_out: wp.array2d[float],
  qvel_out: wp.array2d[float],
):
  worldid, jntid = wp.tid()
  dt = opt_timestep[worldid % opt_timestep.shape[0]]
  jnttype = jnt_type[jntid]
  qadr = jnt_qposadr[jntid]
  dadr = jnt_dofadr[jntid]
  qvel_lin = wp.vec3(0.0, 0.0, 0.0)
  qvel_ang = wp.vec3(0.0, 0.0, 0.0)

  if jnttype == JointType.FREE:
    vlx = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 0, 1.0)
    vly = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 1, 1.0)
    vlz = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 2, 1.0)
    vax = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 3, 1.0)
    vay = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 4, 1.0)
    vaz = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 5, 1.0)
    qvel_out[worldid, dadr + 0] = vlx
    qvel_out[worldid, dadr + 1] = vly
    qvel_out[worldid, dadr + 2] = vlz
    qvel_out[worldid, dadr + 3] = vax
    qvel_out[worldid, dadr + 4] = vay
    qvel_out[worldid, dadr + 5] = vaz
    qvel_lin = wp.vec3(vlx, vly, vlz)
    qvel_ang = wp.vec3(vax, vay, vaz)
  elif jnttype == JointType.BALL:
    vx = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 0, 1.0)
    vy = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 1, 1.0)
    vz = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr + 2, 1.0)
    qvel_out[worldid, dadr + 0] = vx
    qvel_out[worldid, dadr + 1] = vy
    qvel_out[worldid, dadr + 2] = vz
    qvel_ang = wp.vec3(vx, vy, vz)
  else:
    v = forward.next_velocity(opt_timestep, qvel_in, qacc_in, worldid, dadr, 1.0)
    qvel_out[worldid, dadr] = v
    qvel_lin = wp.vec3(v, 0.0, 0.0)

  forward.next_position(qpos_in, jnttype, qadr, dt, worldid, qvel_lin, qvel_ang, qpos_out)


@wp.kernel(enable_backward=False)
def _smooth_qvel_vjp(
  # Model:
  opt_timestep: wp.array[float],
  qD_fullm_i: wp.array[int],
  qD_fullm_j: wp.array[int],
  # Data in:
  qLU_in: wp.array2d[float],
  # In:
  M_D: wp.array2d[float],
  lam: wp.array2d[float],
  # Out:
  adj_qvel_out: wp.array2d[float],
):
  worldid, elemid = wp.tid()
  dt = opt_timestep[worldid % opt_timestep.shape[0]]
  i = qD_fullm_i[elemid]
  j = qD_fullm_j[elemid]
  wp.atomic_add(adj_qvel_out[worldid], j, (M_D[worldid, elemid] - qLU_in[worldid, elemid]) / dt * lam[worldid, i])


@wp.kernel(enable_backward=False)
def _smooth_ctrl_vjp(
  # Model:
  actuator_gainprm: wp.array2d[vec10],
  # Data in:
  moment_rownnz_in: wp.array2d[int],
  moment_rowadr_in: wp.array2d[int],
  moment_colind_in: wp.array2d[int],
  actuator_moment_in: wp.array2d[float],
  # In:
  lam: wp.array2d[float],
  # Out:
  adj_ctrl_out: wp.array2d[float],
):
  worldid, actuatorid = wp.tid()
  rowadr = moment_rowadr_in[worldid, actuatorid]
  rownnz = moment_rownnz_in[worldid, actuatorid]
  value = smooth_adjoint._actuator_moment_dot(moment_colind_in, actuator_moment_in, lam, worldid, rowadr, rownnz)
  adj_ctrl_out[worldid, actuatorid] = actuator_gainprm[worldid % actuator_gainprm.shape[0], actuatorid][0] * value


@wp.kernel(enable_backward=False)
def _xfrc_applied_vjp(
  # Model:
  nv: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  lam: wp.array2d[float],
  # Out:
  adj_xfrc_applied_out: wp.array2d[wp.spatial_vector],
):
  worldid, bodyid = wp.tid()
  force = wp.vec3(0.0)
  torque = wp.vec3(0.0)
  for dofid in range(nv):
    jacp, jacr = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      xipos_in[worldid, bodyid],
      bodyid,
      dofid,
      worldid,
    )
    force += lam[worldid, dofid] * jacp
    torque += lam[worldid, dofid] * jacr
  # xfrc_applied stores force followed by torque, unlike motion-space spatial vectors.
  adj_xfrc_applied_out[worldid, bodyid] = wp.spatial_vector(force, torque)


@wp.kernel(enable_backward=True)
def _dampingpoly_Qv_leaf(
  # Model:
  opt_timestep: wp.array[float],
  dof_damping: wp.array2d[float],
  dof_dampingpoly: wp.array2d[wp.vec2],
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # Out:
  out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  dt = opt_timestep[worldid % opt_timestep.shape[0]]
  damping = dof_damping[worldid % dof_damping.shape[0], dofid]
  poly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dofid]
  out[worldid, dofid] = dt * util_misc._poly_force_deriv(damping, poly, qvel_in[worldid, dofid], 1) * qacc_in[worldid, dofid]


@wp.kernel(enable_backward=False)
def _site_jac_vjp(
  # Model:
  nsite: int,
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  body_isdofancestor: wp.array2d[int],
  # Data in:
  site_xpos_in: wp.array2d[wp.vec3],
  subtree_com_in: wp.array2d[wp.vec3],
  cdof_in: wp.array2d[wp.spatial_vector],
  # In:
  adj_site: wp.array2d[wp.vec3],
  # Out:
  adj_dof_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  value = float(0.0)
  for siteid in range(nsite):
    jacp, _ = support.jac_dof(
      body_parentid,
      body_rootid,
      dof_bodyid,
      body_isdofancestor,
      subtree_com_in,
      cdof_in,
      site_xpos_in[worldid, siteid],
      site_bodyid[siteid],
      dofid,
      worldid,
    )
    value += wp.dot(jacp, adj_site[worldid, siteid])
  adj_dof_out[worldid, dofid] += value


@event_scope
def advance_backward(m: Model, d: Data, d_out: Data, bc: BackwardContext):
  """Maps output-state cotangents through the integrator."""
  nworld, nq, nv = d.nworld, d.qpos.shape[1], m.nv
  adj_qpos, adj_qvel, adj_qacc = bc.adj_qpos, bc.adj_qvel, bc.adj_qacc

  deriv_flags = DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER
  implicit = m.opt.integrator == IntegratorType.IMPLICITFAST and bool(deriv_flags & ~m.opt.disableflags)
  eulerdamp = m.opt.integrator == IntegratorType.EULER and not m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER)
  qacc_advance = d_out.qacc

  if implicit:
    derivative.deriv_smooth_vel(m, dataclasses.replace(d_out, qvel=d.qvel), bc.qDeriv)
    qLD, qLDiagInv, qacc_advance = bc.qLD, bc.qLDiagInv, bc.qacc_advance
    smooth.factor_solve_i(m, d_out, bc.qDeriv, qLD, qLDiagInv, qacc_advance, d_out.efc.Ma)
  elif eulerdamp:
    wp.launch(
      forward._compute_damping_deriv,
      dim=(nworld, nv),
      inputs=[m.dof_damping, m.dof_dampingpoly, d.qvel],
      outputs=[bc.damp_deriv],
    )
    wp.copy(bc.qDeriv, d_out.M)
    wp.launch(
      forward._euler_damp_qfrc,
      dim=(nworld, nv),
      inputs=[m.opt.timestep, m.M_rownnz, m.M_rowadr, bc.damp_deriv],
      outputs=[bc.qDeriv],
    )
    support.mul_m(m, d_out, bc.Ma, d_out.qacc)
    qLD, qLDiagInv, qacc_advance = bc.qLD, bc.qLDiagInv, bc.qacc_advance
    smooth.factor_solve_i(m, d_out, bc.qDeriv, qLD, qLDiagInv, qacc_advance, bc.Ma)

  pairs = [
    (m.opt.timestep, None),
    (m.jnt_type, None),
    (m.jnt_qposadr, None),
    (m.jnt_dofadr, None),
    (d.qpos, adj_qpos),
    (d.qvel, adj_qvel),
    (qacc_advance, adj_qacc),
  ]
  adjoint_util._launch_vjp(
    _advance_state,
    (nworld, m.njnt),
    pairs,
    [d_out.qpos, d_out.qvel],
    [d_out.qpos.grad, d_out.qvel.grad],
  )

  if implicit or eulerdamp:
    smooth.solve_LD(m, d_out, qLD, qLDiagInv, bc.y_remap, adj_qacc)
    support.mul_m(m, d_out, bc.adj_qacc_root, bc.y_remap)

    if implicit:
      wp.launch(adjoint_util._sub_cols, dim=(nworld, nv), inputs=[d_out.qacc, qacc_advance], outputs=[bc.w_dir])
      qpos_direct = smooth_adjoint.mass_matrix_qpos_vjp(m, d, bc.y_remap, bc.w_dir, bc=bc)
      wp.launch(adjoint_util._accum_cols, dim=(nworld, nq), inputs=[qpos_direct], outputs=[adj_qpos])

    if not m.opt.disableflags & DisableBit.DAMPER:
      # Q = M + dt*D, where D = -d(force_damper)/dv. Its VJP affects velocity and,
      # when optimized, damping itself. Use a negative seed because qacc = Q^-1 Ma.
      wp.launch(adjoint_util._neg_cols, dim=(nworld, nv), inputs=[bc.y_remap], outputs=[bc.smooth_r.grad])
      pairs = [
        (m.opt.timestep, None),
        (m.dof_damping, m.dof_damping.grad if m.dof_damping.requires_grad else None),
        (m.dof_dampingpoly, None),
        (d.qvel, bc.adj_qvel_damp),
        (qacc_advance, None),
      ]
      adjoint_util._launch_vjp(_dampingpoly_Qv_leaf, (nworld, nv), pairs, [bc.dampingpoly_Qv], [bc.smooth_r.grad])
      wp.launch(adjoint_util._accum_cols, dim=(nworld, nv), inputs=[bc.adj_qvel_damp], outputs=[adj_qvel])

    adj_qacc = bc.adj_qacc_root

  wp.launch(adjoint_util._accum_cols, dim=(nworld, nv), inputs=[d_out.qacc.grad], outputs=[adj_qacc])
  return adj_qpos, adj_qvel, adj_qacc


@event_scope
def smooth_qvel_backward(m: Model, d: Data, d_out: Data, lam: wp.array, adj_qvel: wp.array, bc: BackwardContext):
  """Accumulates the smooth residual VJP in velocity and control space."""
  nworld = d.nworld
  linearization = dataclasses.replace(d_out, qvel=d.qvel)
  derivative.deriv_smooth_vel(m, linearization, bc.qDeriv)
  wp.launch(forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, bc.qDeriv], outputs=[bc.qLU])
  derivative.deriv_rne_vel(m, linearization, bc.qLU, flg_subtract=False)
  wp.launch(forward._map_m2d, dim=(nworld, m.nD), inputs=[m.mapM2D, d_out.M], outputs=[bc.M_D])
  wp.launch(
    _smooth_qvel_vjp,
    dim=(nworld, m.nD),
    inputs=[m.opt.timestep, m.qD_fullm_i, m.qD_fullm_j, bc.qLU, bc.M_D, lam],
    outputs=[adj_qvel],
  )
  if m.nu and d.ctrl.requires_grad:
    wp.launch(
      _smooth_ctrl_vjp,
      dim=(nworld, m.nu),
      inputs=[
        m.actuator_gainprm,
        d_out.moment_rownnz,
        d_out.moment_rowadr,
        d_out.moment_colind,
        d_out.actuator_moment,
        lam,
      ],
      outputs=[d.ctrl.grad],
    )
  if d.xfrc_applied.requires_grad:
    wp.launch(
      _xfrc_applied_vjp,
      dim=(nworld, m.nbody),
      inputs=[
        m.nv,
        m.body_parentid,
        m.body_rootid,
        m.dof_bodyid,
        m.body_isdofancestor,
        d_out.xipos,
        d_out.subtree_com,
        d_out.cdof,
        lam,
      ],
      outputs=[d.xfrc_applied.grad],
    )


@event_scope
def smooth_qpos_backward(m: Model, d: Data, d_out: Data, lam: wp.array, res_qpos: wp.array, bc: BackwardContext):
  """Accumulates the smooth residual VJP in position space."""
  smooth_adjoint.assert_smooth_supported(m)
  linearization = smooth_adjoint.linearize(m, d, d_out, bc)
  for vjp in (
    smooth_adjoint.smooth_force_backward,
    smooth_adjoint.spring_qpos_vjp,
    smooth_adjoint.actuator_qpos_vjp,
    smooth_adjoint.gravcomp_qpos_vjp,
  ):
    residual = vjp(m, linearization, lam, bc=bc)
    wp.launch(adjoint_util._accum_cols, dim=residual.shape, inputs=[residual], outputs=[res_qpos])


@event_scope
def fwd_kinematics_backward(m: Model, d: Data, bc: BackwardContext):
  """Maps site-position cotangents to generalized position coordinates."""
  wp.copy(bc.scratch.qpos, d.qpos)
  if m.nmocap:
    wp.copy(bc.scratch.mocap_pos, d.mocap_pos)
    wp.copy(bc.scratch.mocap_quat, d.mocap_quat)
  smooth.kinematics(m, bc.scratch)
  smooth.com_pos(m, bc.scratch)
  bc.adj_dof.zero_()
  wp.launch(
    _site_jac_vjp,
    dim=(d.nworld, m.nv),
    inputs=[
      m.nsite,
      m.body_parentid,
      m.body_rootid,
      m.dof_bodyid,
      m.site_bodyid,
      m.body_isdofancestor,
      bc.scratch.site_xpos,
      bc.scratch.subtree_com,
      bc.scratch.cdof,
      d.site_xpos.grad,
    ],
    outputs=[bc.adj_dof],
  )
  wp.launch(
    smooth_kinematics_adjoint._dof_to_qpos,
    dim=(d.nworld, m.njnt),
    inputs=[m.jnt_type, m.jnt_qposadr, m.jnt_dofadr, d.qpos, bc.adj_dof],
    outputs=[d.qpos.grad],
  )
