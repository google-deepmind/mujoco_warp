# Copyright 2026 The Newton Developers
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
"""Experimental: compact active DOFs into nv_max-sized arrays for factor/solve.

The active set is tracked per kinematic tree (the unit of coupling in the mass
matrix). A tree becomes active through an actuator (seeded at make_data), an
applied force, or a contact, and stays active (monotonic - no deactivation yet).
Each step the active trees' DOFs are packed into a contiguous [0, ncdof) range
so the dense factor/solver can run at size nv_max instead of nv.
"""

import dataclasses

import warp as wp

from mujoco_warp._src import types
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_factorize_solve_func
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

_TILE_SIZE = types.TILE_SIZE_JTDAJ_DENSE

# A contact only wakes a sleeping tree if one of its bodies moves faster than this
# (squared spatial velocity, |cvel|^2). Keeps settled/resting contacts asleep.
_WAKE_SPEED_SQ = 1.0e-4


def _round_up(x: int, multiple: int) -> int:
  return ((x + multiple - 1) // multiple) * multiple


@wp.kernel
def _seed_qfrc_applied(
  # Model:
  dof_treeid: wp.array[int],
  # Data in:
  qfrc_applied_in: wp.array2d[float],
  # Data out:
  tree_active_out: wp.array2d[bool],
):
  worldid, dofid = wp.tid()
  if qfrc_applied_in[worldid, dofid] != 0.0:
    tree = dof_treeid[dofid]
    if tree >= 0:
      tree_active_out[worldid, tree] = True


@wp.kernel
def _seed_xfrc_applied(
  # Model:
  body_treeid: wp.array[int],
  # Data in:
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  # Data out:
  tree_active_out: wp.array2d[bool],
):
  worldid, bodyid = wp.tid()
  tree = body_treeid[bodyid]
  if tree < 0:
    return
  xfrc = xfrc_applied_in[worldid, bodyid]
  for i in range(6):
    if xfrc[i] != 0.0:
      tree_active_out[worldid, tree] = True
      return


@wp.func
def _body_moving(cvel_in: wp.array2d[wp.spatial_vector], worldid: int, bodyid: int, speed_sq: float) -> bool:
  if bodyid < 0:
    return False
  v = cvel_in[worldid, bodyid]
  return wp.dot(v, v) > speed_sq


@wp.kernel
def _seed_contacts(
  # Model:
  body_treeid: wp.array[int],
  geom_bodyid: wp.array[int],
  # Data in:
  cvel_in: wp.array2d[wp.spatial_vector],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  wake_speed_sq: float,
  # Data out:
  tree_active_out: wp.array2d[bool],
):
  conid = wp.tid()
  if conid >= nacon_in[0]:
    return
  worldid = contact_worldid_in[conid]
  geom = contact_geom_in[conid]

  b0 = wp.where(geom[0] >= 0, geom_bodyid[geom[0]], -1)
  b1 = wp.where(geom[1] >= 0, geom_bodyid[geom[1]], -1)

  # Velocity gate: only wake on contacts where at least one body is moving.
  # Resting contacts (cup on a static table, settled clutter) stay asleep; motion
  # propagates outward from the actuated/forced drivers over successive steps.
  if not (_body_moving(cvel_in, worldid, b0, wake_speed_sq) or _body_moving(cvel_in, worldid, b1, wake_speed_sq)):
    return

  if b0 >= 0 and body_treeid[b0] >= 0:
    tree_active_out[worldid, body_treeid[b0]] = True
  if b1 >= 0 and body_treeid[b1] >= 0:
    tree_active_out[worldid, body_treeid[b1]] = True


@wp.kernel
def _compact_dofs(
  # Model:
  ntree: int,
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  # Data in:
  tree_active_in: wp.array2d[bool],
  nv_max_in: int,
  # Data out:
  ncdof_out: wp.array[int],
  dof_cdof_out: wp.array2d[int],
  cdof_dof_out: wp.array2d[int],
):
  worldid = wp.tid()
  count = int(0)
  for t in range(ntree):
    if tree_active_in[worldid, t]:
      adr = tree_dofadr[t]
      num = tree_dofnum[t]
      for j in range(num):
        dof = adr + j
        if count < nv_max_in:
          dof_cdof_out[worldid, dof] = count
          cdof_dof_out[worldid, count] = dof
        count += 1

  if count > nv_max_in:
    wp.printf(
      "nv_compact overflow: world %d needs %d active DOFs but nv_max = %d (behavior undefined)\n",
      worldid,
      count,
      nv_max_in,
    )
    ncdof_out[worldid] = nv_max_in
  else:
    ncdof_out[worldid] = count


@event_scope
def update_active_dofs(m: types.Model, d: types.Data):
  """Update the persistent active-tree mask and rebuild the compaction maps."""
  # seed newly-active trees from applied forces and contacts (monotonic OR)
  wp.launch(_seed_qfrc_applied, dim=(d.nworld, m.nv), inputs=[m.dof_treeid, d.qfrc_applied], outputs=[d.tree_active])
  wp.launch(_seed_xfrc_applied, dim=(d.nworld, m.nbody), inputs=[m.body_treeid, d.xfrc_applied], outputs=[d.tree_active])

  wp.launch(
    _seed_contacts,
    dim=(d.naconmax,),
    inputs=[m.body_treeid, m.geom_bodyid, d.cvel, d.contact.geom, d.contact.worldid, d.nacon, _WAKE_SPEED_SQ],
    outputs=[d.tree_active],
  )

  # rebuild compaction maps from the (monotonic) active-tree mask
  d.dof_cdof.fill_(-1)
  d.cdof_dof.fill_(-1)
  wp.launch(
    _compact_dofs,
    dim=(d.nworld,),
    inputs=[m.ntree, m.tree_dofadr, m.tree_dofnum, d.tree_active, d.nv_max],
    outputs=[d.ncdof, d.dof_cdof, d.cdof_dof],
  )


@dataclasses.dataclass
class NvCompactContext:
  """Workspace for the compacted dense factor/solve.

  DOF-space arrays are sized by nv_max_pad (nv_max rounded up to the dense tile size)
  so the blocked Cholesky never reads out of bounds on its final partial block.
  """

  nv_max_pad: int
  # convergence tolerances rescaled by nv_full/nv_max_pad so the solver's nv-normalized
  # done-test (solve.py: value/(meaninertia*nv)) matches the full-model baseline.
  tol_c: wp.array
  ls_tol_c: wp.array
  # compacted dof-pair indices for the elliptic JTCJ Hessian (over nv_max_pad, not global nv)
  dof_tri_row_c: wp.array
  dof_tri_col_c: wp.array
  # smooth-solve workspace (Stage 2a)
  M_c: wp.array3d[float]  # (nworld, nv_max_pad, nv_max_pad) compacted dense inertia
  qLD_c: wp.array3d[float]  # (nworld, nv_max_pad, nv_max_pad) upper Cholesky factor
  rhs_c: wp.array3d[float]  # (nworld, nv_max_pad, 1) compacted right-hand side
  x_c: wp.array3d[float]  # (nworld, nv_max_pad, 1) compacted solution
  # constrained-solve workspace (Stage 2b): compacted views fed to the dense Newton solver
  J_c: wp.array3d[float]  # (nworld, njmax_pad, nv_max_pad) compacted dense constraint Jacobian
  Ma_c: wp.array2d[float]  # (nworld, nv_max_pad) M @ qacc workspace
  qfrc_smooth_c: wp.array2d[float]
  qacc_smooth_c: wp.array2d[float]
  qacc_warmstart_c: wp.array2d[float]
  qacc_c: wp.array2d[float]
  qfrc_constraint_c: wp.array2d[float]


@wp.kernel
def _scale_array(src: wp.array[float], scale: float, dst_out: wp.array[float]):
  i = wp.tid()
  dst_out[i] = src[i] * scale


@wp.kernel
def _fill_dof_pairs(n: int, row_out: wp.array[int], col_out: wp.array[int]):
  # Enumerate all (i, j) DOF pairs of the nv_max_pad-wide compacted Hessian. The elliptic
  # JTCJ correction indexes the Hessian via these; the global dof_tri (triu over full nv)
  # would write out of bounds. Pairs with a zero J_c column contribute nothing.
  i, j = wp.tid()
  row_out[i * n + j] = i
  col_out[i * n + j] = j


def get_context(m: types.Model, d: types.Data) -> NvCompactContext:
  """Return a cached compaction context for d, creating it on first use.

  Cached as a dynamic attribute so it is allocated once (outside any graph capture)
  and reused across steps.
  """
  ctx = getattr(d, "_nvcompact_ctx", None)
  if ctx is None or ctx.nv_max_pad != _round_up(max(d.nv_max, 1), _TILE_SIZE):
    ctx = create_nvcompact_context(m, d)
    d._nvcompact_ctx = ctx
  return ctx


def create_nvcompact_context(m: types.Model, d: types.Data) -> NvCompactContext:
  nv_max_pad = _round_up(max(d.nv_max, 1), _TILE_SIZE)
  nworld = d.nworld

  def zeros2d():
    return wp.zeros((nworld, nv_max_pad), dtype=float)

  # match baseline's nv-normalized convergence test: the compact solve passes nv=nv_max_pad
  # (< full nv) to the rescale, making it stricter; scale tolerance up to compensate.
  # Done on-device (no host sync) so context creation is safe inside CUDA graph capture.
  tol_scale = float(m.nv) / float(nv_max_pad)
  tol_c = wp.empty_like(m.opt.tolerance)
  ls_tol_c = wp.empty_like(m.opt.ls_tolerance)
  wp.launch(_scale_array, dim=tol_c.shape[0], inputs=[m.opt.tolerance, tol_scale], outputs=[tol_c])
  wp.launch(_scale_array, dim=ls_tol_c.shape[0], inputs=[m.opt.ls_tolerance, tol_scale], outputs=[ls_tol_c])

  # compacted dof-pair indices for the elliptic JTCJ Hessian (built on-device, capture-safe)
  npair = nv_max_pad * nv_max_pad
  dof_tri_row_c = wp.empty(npair, dtype=int)
  dof_tri_col_c = wp.empty(npair, dtype=int)
  wp.launch(_fill_dof_pairs, dim=(nv_max_pad, nv_max_pad), inputs=[nv_max_pad], outputs=[dof_tri_row_c, dof_tri_col_c])

  return NvCompactContext(
    nv_max_pad=nv_max_pad,
    tol_c=tol_c,
    ls_tol_c=ls_tol_c,
    dof_tri_row_c=dof_tri_row_c,
    dof_tri_col_c=dof_tri_col_c,
    M_c=wp.zeros((nworld, nv_max_pad, nv_max_pad), dtype=float),
    qLD_c=wp.zeros((nworld, nv_max_pad, nv_max_pad), dtype=float),
    rhs_c=wp.zeros((nworld, nv_max_pad, 1), dtype=float),
    x_c=wp.zeros((nworld, nv_max_pad, 1), dtype=float),
    J_c=wp.zeros((nworld, d.njmax_pad, nv_max_pad), dtype=float),
    Ma_c=zeros2d(),
    qfrc_smooth_c=zeros2d(),
    qacc_smooth_c=zeros2d(),
    qacc_warmstart_c=zeros2d(),
    qacc_c=zeros2d(),
    qfrc_constraint_c=zeros2d(),
  )


@wp.kernel
def _pad_identity(
  # Data in:
  ncdof_in: wp.array[int],
  # Out:
  M_c_out: wp.array3d[float],
):
  worldid, ci = wp.tid()
  if ci >= ncdof_in[worldid]:
    M_c_out[worldid, ci, ci] = 1.0


@wp.kernel
def _gather_M_sparse(
  # Model:
  M_rownnz: wp.array[int],
  M_rowadr: wp.array[int],
  M_colind: wp.array[int],
  # Data in:
  M_in: wp.array3d[float],
  dof_cdof_in: wp.array2d[int],
  # Out:
  M_c_out: wp.array3d[float],
):
  worldid, i = wp.tid()
  ci = dof_cdof_in[worldid, i]
  if ci < 0:
    return
  rowadr = M_rowadr[i]
  for k in range(M_rownnz[i]):
    adr = rowadr + k
    cj = dof_cdof_in[worldid, M_colind[adr]]
    if cj >= 0:
      val = M_in[worldid, 0, adr]
      M_c_out[worldid, ci, cj] = val
      M_c_out[worldid, cj, ci] = val


@wp.kernel
def _gather_rhs(
  # Data in:
  dof_cdof_in: wp.array2d[int],
  # In:
  vec_in: wp.array2d[float],
  # Out:
  rhs_out: wp.array3d[float],
):
  worldid, i = wp.tid()
  ci = dof_cdof_in[worldid, i]
  if ci >= 0:
    rhs_out[worldid, ci, 0] = vec_in[worldid, i]


@wp.kernel
def _scatter_solution(
  # Data in:
  dof_cdof_in: wp.array2d[int],
  # In:
  x_in: wp.array3d[float],
  # Out:
  vec_out: wp.array2d[float],
):
  worldid, i = wp.tid()
  ci = dof_cdof_in[worldid, i]
  if ci >= 0:
    vec_out[worldid, i] = x_in[worldid, ci, 0]
  else:
    vec_out[worldid, i] = 0.0  # frozen inactive DOF


@cache_kernel
def _blocked_factor_solve(matrix_size: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    M_in: wp.array3d[float],
    # In:
    rhs_in: wp.array3d[float],
    msize: int,
    # Out:
    U_out: wp.array3d[float],
    x_out: wp.array3d[float],
  ):
    worldid = wp.tid()
    wp.static(create_blocked_cholesky_factorize_solve_func(_TILE_SIZE, matrix_size))(
      M_in[worldid], rhs_in[worldid], msize, U_out[worldid], x_out[worldid]
    )

  return kernel


@event_scope
def smooth_solve_compact(m: types.Model, d: types.Data, ctx: NvCompactContext):
  """Compacted equivalent of solve_m: qacc_smooth[active] = M_c^-1 qfrc_smooth[active].

  Inactive DOFs are frozen (qacc_smooth set to 0). Reads the sparse Model inertia.
  """
  ctx.M_c.zero_()
  ctx.rhs_c.zero_()
  wp.launch(_pad_identity, dim=(d.nworld, ctx.nv_max_pad), inputs=[d.ncdof], outputs=[ctx.M_c])
  wp.launch(
    _gather_M_sparse,
    dim=(d.nworld, m.nv),
    inputs=[m.M_rownnz, m.M_rowadr, m.M_colind, d.M, d.dof_cdof],
    outputs=[ctx.M_c],
  )
  wp.launch(_gather_rhs, dim=(d.nworld, m.nv), inputs=[d.dof_cdof, d.qfrc_smooth], outputs=[ctx.rhs_c])
  wp.launch_tiled(
    _blocked_factor_solve(ctx.nv_max_pad),
    dim=d.nworld,
    inputs=[ctx.M_c, ctx.rhs_c, ctx.nv_max_pad],
    outputs=[ctx.qLD_c, ctx.x_c],
    block_dim=m.block_dim.update_gradient_cholesky_blocked,
  )
  wp.launch(_scatter_solution, dim=(d.nworld, m.nv), inputs=[d.dof_cdof, ctx.x_c], outputs=[d.qacc_smooth])


@wp.kernel
def _gather_dof_vec(
  # Data in:
  dof_cdof_in: wp.array2d[int],
  # In:
  vec_in: wp.array2d[float],
  # Out:
  out: wp.array2d[float],
):
  worldid, i = wp.tid()
  ci = dof_cdof_in[worldid, i]
  if ci >= 0:
    out[worldid, ci] = vec_in[worldid, i]


@wp.kernel
def _scatter_dof_vec(
  # Data in:
  dof_cdof_in: wp.array2d[int],
  # In:
  in_c: wp.array2d[float],
  # Out:
  out: wp.array2d[float],
):
  worldid, i = wp.tid()
  ci = dof_cdof_in[worldid, i]
  if ci >= 0:
    out[worldid, i] = in_c[worldid, ci]
  else:
    out[worldid, i] = 0.0  # frozen inactive DOF


@wp.kernel
def _gather_J_sparse(
  # Data in:
  nefc_in: wp.array[int],
  dof_cdof_in: wp.array2d[int],
  # In:
  J_rownnz_in: wp.array2d[int],
  J_rowadr_in: wp.array2d[int],
  J_colind_in: wp.array3d[int],
  J_in: wp.array3d[float],
  # Out:
  J_c_out: wp.array3d[float],
):
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  rowadr = J_rowadr_in[worldid, efcid]
  for k in range(J_rownnz_in[worldid, efcid]):
    adr = rowadr + k
    cj = dof_cdof_in[worldid, J_colind_in[worldid, 0, adr]]
    if cj >= 0:
      J_c_out[worldid, efcid, cj] = J_in[worldid, 0, adr]


def _gather2d(d, src, dst):
  dst.zero_()
  wp.launch(_gather_dof_vec, dim=(d.nworld, src.shape[1]), inputs=[d.dof_cdof, src], outputs=[dst])


@event_scope
def solve_compact(m: types.Model, d: types.Data, ctx: NvCompactContext):
  """Run the dense Newton constraint solver in compacted DOF space.

  Gathers the active-DOF inertia, constraint Jacobian, and smooth/warmstart vectors
  into nv_max_pad-sized dense workspaces, runs the stock dense Newton solver on a
  shallow-replaced (m, d) at nv_max_pad, then scatters qacc/qfrc_constraint back.
  Inactive DOFs are frozen to 0. Reads the sparse Model inertia and constraint J.
  """
  import dataclasses as _dc

  from mujoco_warp._src import solver

  nvp = ctx.nv_max_pad

  _compact_gather(m, d, ctx)

  # shallow-replace (m, d) so the stock dense Newton solver runs at nv_max_pad.
  # Keep graph-conditional early-exit on CUDA (matches baseline: stops at convergence
  # instead of running all iterations); fall back to the plain loop on CPU.
  gc = m.opt.graph_conditional and wp.get_device().is_cuda
  opt2 = _dc.replace(m.opt, graph_conditional=gc, tolerance=ctx.tol_c, ls_tolerance=ctx.ls_tol_c)
  m2 = _dc.replace(
    m, opt=opt2, nv=nvp, nv_pad=nvp, is_sparse=False, dof_tri_row=ctx.dof_tri_row_c, dof_tri_col=ctx.dof_tri_col_c
  )
  efc2 = _dc.replace(d.efc, J=ctx.J_c, Ma=ctx.Ma_c)
  d2 = _dc.replace(
    d,
    M=ctx.M_c,
    qfrc_smooth=ctx.qfrc_smooth_c,
    qacc_smooth=ctx.qacc_smooth_c,
    qacc_warmstart=ctx.qacc_warmstart_c,
    qacc=ctx.qacc_c,
    qfrc_constraint=ctx.qfrc_constraint_c,
    efc=efc2,
  )

  sctx = solver._create_solver_context(m2, d2)
  solver._solve(m2, d2, sctx)

  _compact_scatter(m, d, ctx)


@event_scope
def _compact_gather(m: types.Model, d: types.Data, ctx: NvCompactContext):
  nvp = ctx.nv_max_pad
  # gather compacted dense inertia (identity-padded tail)
  ctx.M_c.zero_()
  wp.launch(_pad_identity, dim=(d.nworld, nvp), inputs=[d.ncdof], outputs=[ctx.M_c])
  wp.launch(
    _gather_M_sparse,
    dim=(d.nworld, m.nv),
    inputs=[m.M_rownnz, m.M_rowadr, m.M_colind, d.M, d.dof_cdof],
    outputs=[ctx.M_c],
  )
  # gather compacted dense constraint Jacobian (active columns only)
  ctx.J_c.zero_()
  wp.launch(
    _gather_J_sparse,
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, d.dof_cdof, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J],
    outputs=[ctx.J_c],
  )
  # gather compacted DOF-space vectors
  _gather2d(d, d.qfrc_smooth, ctx.qfrc_smooth_c)
  _gather2d(d, d.qacc_smooth, ctx.qacc_smooth_c)
  _gather2d(d, d.qacc_warmstart, ctx.qacc_warmstart_c)


@event_scope
def _compact_scatter(m: types.Model, d: types.Data, ctx: NvCompactContext):
  from mujoco_warp._src import support

  # scatter results back to full DOF space (inactive frozen to 0)
  wp.launch(_scatter_dof_vec, dim=(d.nworld, m.nv), inputs=[d.dof_cdof, ctx.qacc_c], outputs=[d.qacc])
  wp.launch(_scatter_dof_vec, dim=(d.nworld, m.nv), inputs=[d.dof_cdof, ctx.qfrc_constraint_c], outputs=[d.qfrc_constraint])

  # Refresh full d.efc.Ma = M @ qacc. The integrators (Euler/implicit damping) use Ma as
  # the RHS; the compact solve only populated the compacted Ma, so recompute it in full
  # space. Inactive DOFs have qacc=0 so their Ma is 0 and they stay frozen.
  support.mul_m(m, d, d.efc.Ma, d.qacc)
