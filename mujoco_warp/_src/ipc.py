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
"""Barrier-free augmented Lagrangian contact for 2D flexes (mj_ipc).

Matches MuJoCo C engine_ipc.c / engine_ipc.h.
"""

import dataclasses
from typing import Optional

import warp as wp

from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import math
from mujoco_warp._src import sensor
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src.collision_continuous import FlexPairType
from mujoco_warp._src.collision_continuous import eval_pair_gap
from mujoco_warp._src.collision_continuous import ipc_advance
from mujoco_warp._src.collision_continuous import ipc_discover_candidates
from mujoco_warp._src.collision_continuous import ipc_init_geom_features_kernel
from mujoco_warp._src.collision_continuous import ipc_update_geom_features
from mujoco_warp._src.collision_continuous import pair_vert_range
from mujoco_warp._src.collision_continuous import standoff
from mujoco_warp._src.types import ConstraintState
from mujoco_warp._src.types import ConstraintType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import SolverContext
from mujoco_warp._src.types import vec5
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})

IPC_ASET_AGE = 25
IPC_EVICT = 44
IPC_DECAY = 0.9
IPC_ALPHA_LB = 1e-6
IPC_STALL_MAX = 64
IPC_WS_TOL = 1e-8
IPC_DELTACAP = 0.001
IPC_GHAT = 3.0 * IPC_DELTACAP
IPC_VEL_TOL = 0.05
FLEXCONTACT_OMEGA2 = 5e7


@dataclasses.dataclass
class IpcWorkspace:
  """Per-Data scratch arrays for GPU IPC execution."""

  w: wp.array2d  # (nworld, nv) float
  wfree: wp.array2d  # (nworld, nv) float
  wtil: wp.array2d  # (nworld, nv) float
  ddw: wp.array2d  # (nworld, nv) float
  da: wp.array2d  # (nworld, nv) float
  M_da: wp.array2d  # (nworld, nv) float
  qvel_old: wp.array2d  # (nworld, nv) float
  wn: wp.array2d  # (nworld, nv) float

  xold: wp.array2d  # (nworld, nfv) vec3
  x: wp.array2d  # (nworld, nfv) vec3
  xtil: wp.array2d  # (nworld, nfv) vec3
  xfree: wp.array2d  # (nworld, nfv) vec3
  xn: wp.array2d  # (nworld, nfv) vec3
  pvel: wp.array2d  # (nworld, nfv) vec3

  beta: wp.array  # (nworld,) float
  world_done: wp.array  # (nworld,) int
  stall: wp.array  # (nworld,) int
  saved_nefc: wp.array  # (nworld,) int
  active_count: wp.array  # (nworld,) int
  active_nnz: wp.array  # (nworld,) int
  total_solver_niter: wp.array  # (nworld,) int

  newton_converged: wp.array  # (nworld,) int
  last_ls_alpha: wp.array  # (nworld,) float
  ls_done: wp.array  # (nworld,) int
  ls_accept: wp.array  # (nworld,) int
  merit_energy: wp.array  # (nworld,) float
  merit_energy_trial: wp.array  # (nworld,) float

  nvertbody: wp.array  # (nbody,) int
  isflexvert_body: wp.array  # (nbody,) int
  npin: wp.array  # (nbody,) int
  apinned: wp.array  # (ntree,) int
  dofadr: wp.array  # (nfv,) int
  fidx: wp.array  # (nfv,) int
  mass: wp.array2d  # (nworld, nfv) float
  rad: wp.array  # (nfv,) float
  pbody: wp.array  # (nfv,) int
  own: wp.array  # (nv,) bool
  isflexdof: wp.array  # (nv,) bool
  isartictree: wp.array  # (ntree,) bool
  is_throttled: wp.array  # (nv,) bool
  max_chain: int
  pin_has_chain: wp.array  # (nfv,) bool
  pin_num_chain: wp.array  # (nfv,) int
  pin_chain_dof: wp.array2d  # (nfv, max_chain) int
  pin_chain_axis: wp.array3d  # (nworld, nfv, max_chain) vec3

  max_ngv: int
  max_nge: int
  ngv: wp.array  # (nworld,) int
  nge: wp.array  # (nworld,) int
  geom_corner_loc: wp.array2d  # (nworld, max_ngv) vec3
  geom_corners: wp.array2d  # (nworld, max_ngv) vec3
  corner_geom: wp.array2d  # (nworld, max_ngv) int
  geom_edge_loc: wp.array3d  # (nworld, max_nge, 2) vec3
  geom_edges: wp.array3d  # (nworld, max_nge, 2) vec3
  edge_geom: wp.array2d  # (nworld, max_nge) int
  geom_aabb_min: wp.array2d  # (nworld, ngeom) vec3
  geom_aabb_max: wp.array2d  # (nworld, ngeom) vec3

  max_aset: int
  max_cand: int
  naset: wp.array  # (nworld,) int
  aset_type: wp.array2d  # (nworld, max_aset) int
  aset_idx: wp.array2d  # (nworld, max_aset) vec4i
  aset_geom: wp.array2d  # (nworld, max_aset) int
  aset_standoff: wp.array2d  # (nworld, max_aset) float
  aset_mu: wp.array2d  # (nworld, max_aset) float
  aset_ld0: wp.array2d  # (nworld, max_aset) float
  aset_lam: wp.array2d  # (nworld, max_aset) float
  aset_cnt: wp.array2d  # (nworld, max_aset) int

  nret: wp.array  # (nworld,) int
  ret_type: wp.array2d  # (nworld, max_aset) int
  ret_idx: wp.array2d  # (nworld, max_aset) vec4i
  ret_geom: wp.array2d  # (nworld, max_aset) int
  ret_standoff: wp.array2d  # (nworld, max_aset) float
  ret_mu: wp.array2d  # (nworld, max_aset) float
  ret_lam: wp.array2d  # (nworld, max_aset) float
  ret_cnt: wp.array2d  # (nworld, max_aset) int

  ncand: wp.array  # (nworld,) int
  cand_type: wp.array2d  # (nworld, max_cand) int
  cand_idx: wp.array2d  # (nworld, max_cand) vec4i
  cand_geom: wp.array2d  # (nworld, max_cand) int
  cand_ld0: wp.array2d  # (nworld, max_cand) float
  cand_toi: wp.array2d  # (nworld, max_cand) float

  pair_ln: wp.array2d  # (nworld, max_aset) vec3
  pair_lcw: wp.array2d  # (nworld, max_aset) vec4
  pair_liv: wp.array2d  # (nworld, max_aset) vec4i
  pair_lniv: wp.array2d  # (nworld, max_aset) int
  pair_rownnz: wp.array2d  # (nworld, max_aset) int
  pair_in_ws: wp.array2d  # (nworld, max_aset) int

  alpha_min: wp.array  # (nworld,) float
  vmin: wp.array2d  # (nworld, nfv) float
  ws_nadd: wp.array  # (nworld,) int
  ws_done: wp.array  # (nworld,) int
  qfrc_rows: wp.array2d  # (nworld, nv) float
  outer_cond: wp.array  # (1,) int
  outer_iter: wp.array  # (1,) int
  ws_cond: wp.array  # (1,) int
  ws_iter: wp.array  # (1,) int
  ls_cond: wp.array  # (1,) int
  ls_iter: wp.array  # (1,) int
  ws_entry: wp.array2d  # (nworld, nv) float
  init_ncand: wp.array  # (nworld,) int
  dl: wp.array2d  # (nworld, nfv) float
  fsweep: wp.array2d  # (nworld, nflex) float
  solver_ctx: Optional[SolverContext] = None


def ipc_skip_constraint(m: Model) -> bool:
  """Returns True if the constraint stage should be skipped during forward dynamics."""
  return bool((m.opt.enableflags & EnableBit.IPC) and m.opt.integrator == IntegratorType.DISCRETE and m.has_2d_flex)


@wp.kernel
def ipc_count_body_verts_kernel(
  # Model:
  flex_vertbodyid: wp.array[int],
  # Out:
  nvertbody_out: wp.array[int],
):
  """Counts how many flex vertices belong to each body."""
  vg = wp.tid()
  bid = flex_vertbodyid[vg]
  if bid >= 0:
    wp.atomic_add(nvertbody_out, bid, 1)


@wp.kernel
def ipc_init_verts_and_bodies_kernel(
  # Model:
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  body_dofadr: wp.array[int],
  body_mass: wp.array2d[float],
  body_subtreemass: wp.array2d[float],
  jnt_type: wp.array[int],
  dof_armature: wp.array2d[float],
  flex_dim: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_vert: wp.array[wp.vec3],
  flex_radius: wp.array[float],
  flex_centered: wp.array[bool],
  flex_vertflexid: wp.array[int],
  # In:
  nvertbody: wp.array[int],
  # Out:
  dofadr_out: wp.array[int],
  fidx_out: wp.array[int],
  mass_out: wp.array2d[float],
  rad_out: wp.array[float],
  pbody_out: wp.array[int],
  isflexvert_body_out: wp.array[int],
  own_out: wp.array[bool],
  isflexdof_out: wp.array[bool],
):
  """Initializes per-vertex IPC attributes and identifies free 3-slide flex vertices."""
  vg = wp.tid()
  f = flex_vertflexid[vg]
  if flex_dim[f] != 2:
    dofadr_out[vg] = -1
    fidx_out[vg] = -1
    for w in range(mass_out.shape[0]):
      mass_out[w, vg] = 0.0
    rad_out[vg] = 0.0
    pbody_out[vg] = -1
    return

  bid = flex_vertbodyid[vg]
  rad_out[vg] = flex_radius[f]
  pbody_out[vg] = bid
  ja = body_jntadr[bid]
  lv = flex_vert[vg]
  at_origin = flex_centered[f] or (lv[0] == 0.0 and lv[1] == 0.0 and lv[2] == 0.0)
  slides = (
    body_jntnum[bid] == 3
    and jnt_type[ja] == int(JointType.SLIDE)
    and jnt_type[ja + 1] == int(JointType.SLIDE)
    and jnt_type[ja + 2] == int(JointType.SLIDE)
    and nvertbody[bid] == 1
    and at_origin
  )
  if slides:
    da = body_dofadr[bid]
    dofadr_out[vg] = da
    fidx_out[vg] = vg
    for w in range(mass_out.shape[0]):
      bm = body_mass[w % body_mass.shape[0], bid]
      m_base = bm if bm > 0.0 else body_subtreemass[w % body_subtreemass.shape[0], bid]
      mass_out[w, vg] = m_base + dof_armature[w % dof_armature.shape[0], da]
    isflexvert_body_out[bid] = 1
    own_out[da] = True
    own_out[da + 1] = True
    own_out[da + 2] = True
    isflexdof_out[da] = True
    isflexdof_out[da + 1] = True
    isflexdof_out[da + 2] = True
  else:
    dofadr_out[vg] = -1
    fidx_out[vg] = -1
    for w in range(mass_out.shape[0]):
      mass_out[w, vg] = 0.0


@wp.kernel
def ipc_init_trees_kernel(
  # Model:
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  dof_treeid: wp.array[int],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  # In:
  isflexvert_body: wp.array[int],
  # Out:
  isartictree_out: wp.array[bool],
  own_out: wp.array[bool],
):
  """Identifies non-flex articulated trees and marks their DOFs as owned by IPC."""
  b = wp.tid()
  if b == 0 or body_dofnum[b] == 0 or isflexvert_body[b] != 0:
    return
  t = dof_treeid[body_dofadr[b]]
  isartictree_out[t] = True
  t_da = tree_dofadr[t]
  t_dn = tree_dofnum[t]
  for i in range(t_dn):
    own_out[t_da + i] = True


@wp.kernel
def ipc_init_pins_kernel(
  # Model:
  body_parentid: wp.array[int],
  body_jntnum: wp.array[int],
  body_jntadr: wp.array[int],
  body_treeid: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  flex_dim: wp.array[int],
  flex_vertflexid: wp.array[int],
  # In:
  max_chain: int,
  fidx: wp.array[int],
  pbody: wp.array[int],
  isartictree: wp.array[bool],
  # Out:
  pin_has_chain_out: wp.array[bool],
  pin_num_chain_out: wp.array[int],
  pin_chain_dof_out: wp.array2d[int],
  own_out: wp.array[bool],
  apinned_out: wp.array[int],
  npin_out: wp.array[int],
):
  """Extracts slide joint chains for pinned 2D flex vertices on articulated bodies."""
  vg = wp.tid()
  f = flex_vertflexid[vg]
  if flex_dim[f] != 2 or fidx[vg] >= 0:
    return
  b = pbody[vg]
  if b <= 0:
    return
  t = body_treeid[b]
  if t < 0 or not isartictree[t]:
    return

  curr = b
  cnt = int(0)
  while curr > 0:
    jn = body_jntnum[curr]
    ja = body_jntadr[curr]
    for j in range(jn):
      jid = ja + j
      if jnt_type[jid] == int(JointType.SLIDE) and cnt < max_chain:
        da = jnt_dofadr[jid]
        pin_chain_dof_out[vg, cnt] = da
        own_out[da] = True
        cnt += 1
    curr = body_parentid[curr]

  if cnt > 0:
    for k in range(cnt // 2):
      tmp = pin_chain_dof_out[vg, k]
      pin_chain_dof_out[vg, k] = pin_chain_dof_out[vg, cnt - 1 - k]
      pin_chain_dof_out[vg, cnt - 1 - k] = tmp
    pin_has_chain_out[vg] = True
    pin_num_chain_out[vg] = cnt
    apinned_out[t] = 1
    wp.atomic_add(npin_out, b, 1)


@wp.kernel
def ipc_finalize_meta_kernel(
  # Model:
  nv: int,
  nflexvert: int,
  body_subtreemass: wp.array2d[float],
  dof_treeid: wp.array[int],
  # In:
  isflexdof: wp.array[bool],
  apinned: wp.array[int],
  pin_has_chain: wp.array[bool],
  pbody: wp.array[int],
  npin: wp.array[int],
  # Out:
  is_throttled_out: wp.array[bool],
  mass_out: wp.array2d[float],
):
  """Sets per-DOF CCD throttling flags and distributes carrier subtree mass to pinned vertices."""
  tid = wp.tid()
  if tid < nv:
    treeid = dof_treeid[tid]
    is_throttled_out[tid] = isflexdof[tid] or (treeid >= 0 and apinned[treeid] != 0)
  if tid < nflexvert:
    if pin_has_chain[tid]:
      b = pbody[tid]
      np_b = npin[b]
      if np_b > 0:
        inv_np = 1.0 / float(np_b)
        for w in range(mass_out.shape[0]):
          mass_out[w, tid] = body_subtreemass[w % body_subtreemass.shape[0], b] * inv_np


def ipc_init_topology(m: Model, ws: IpcWorkspace):
  """Populates IPC vertex, DOF, pinned-chain, and static geom topology arrays on GPU."""
  ws.nvertbody.zero_()
  ws.isflexvert_body.zero_()
  ws.npin.zero_()
  ws.apinned.zero_()
  ws.own.zero_()
  ws.isflexdof.zero_()
  ws.isartictree.zero_()
  ws.is_throttled.zero_()
  ws.pin_has_chain.zero_()
  ws.pin_num_chain.zero_()

  wp.launch(
    ipc_count_body_verts_kernel,
    dim=m.nflexvert,
    inputs=[m.flex_vertbodyid],
    outputs=[ws.nvertbody],
  )
  wp.launch(
    ipc_init_verts_and_bodies_kernel,
    dim=m.nflexvert,
    inputs=[
      m.body_jntnum,
      m.body_jntadr,
      m.body_dofadr,
      m.body_mass,
      m.body_subtreemass,
      m.jnt_type,
      m.dof_armature,
      m.flex_dim,
      m.flex_vertbodyid,
      m.flex_vert,
      m.flex_radius,
      m.flex_centered,
      m.flex_vertflexid,
      ws.nvertbody,
    ],
    outputs=[
      ws.dofadr,
      ws.fidx,
      ws.mass,
      ws.rad,
      ws.pbody,
      ws.isflexvert_body,
      ws.own,
      ws.isflexdof,
    ],
  )
  wp.launch(
    ipc_init_trees_kernel,
    dim=m.nbody,
    inputs=[
      m.body_dofnum,
      m.body_dofadr,
      m.dof_treeid,
      m.tree_dofadr,
      m.tree_dofnum,
      ws.isflexvert_body,
    ],
    outputs=[ws.isartictree, ws.own],
  )
  wp.launch(
    ipc_init_pins_kernel,
    dim=m.nflexvert,
    inputs=[
      m.body_parentid,
      m.body_jntnum,
      m.body_jntadr,
      m.body_treeid,
      m.jnt_type,
      m.jnt_dofadr,
      m.flex_dim,
      m.flex_vertflexid,
      ws.max_chain,
      ws.fidx,
      ws.pbody,
      ws.isartictree,
    ],
    outputs=[
      ws.pin_has_chain,
      ws.pin_num_chain,
      ws.pin_chain_dof,
      ws.own,
      ws.apinned,
      ws.npin,
    ],
  )
  wp.launch(
    ipc_finalize_meta_kernel,
    dim=max(m.nv, m.nflexvert, 1),
    inputs=[
      m.nv,
      m.nflexvert,
      m.body_subtreemass,
      m.dof_treeid,
      ws.isflexdof,
      ws.apinned,
      ws.pin_has_chain,
      ws.pbody,
      ws.npin,
    ],
    outputs=[ws.is_throttled, ws.mass],
  )
  wp.launch(
    ipc_init_geom_features_kernel,
    dim=ws.ngv.shape[0],
    inputs=[
      m.ngeom,
      m.nmesh,
      m.body_weldid,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_bodyid,
      m.geom_dataid,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_graphadr,
      m.mesh_vert,
      m.mesh_polynum,
      m.mesh_polyadr,
      m.mesh_polyvertadr,
      m.mesh_polyvertnum,
      m.mesh_polyvert,
    ],
    outputs=[
      ws.ngv,
      ws.nge,
      ws.geom_corner_loc,
      ws.corner_geom,
      ws.geom_edge_loc,
      ws.edge_geom,
    ],
  )


def get_ipc_workspace(
  m: Model,
  d: Data,
  device: Optional[wp.Device] = None,
) -> IpcWorkspace:
  """Allocates GPU workspace arrays for IPC and initializes static topology."""
  device = device or d.qacc.device
  nworld = d.nworld
  nv = max(m.nv, 1)
  nbody = max(m.nbody, 1)
  ntree = max(m.ntree, 1)
  ngeom = max(m.ngeom, 1)
  nfv = max(m.nflexvert, 1)
  nflex = max(m.nflex, 1)
  nconmax = max(d.naconmax // nworld, 1)
  nccdmax = max(d.naccdmax // nworld, 1)
  max_chain = max(min(m.njnt, m.nv), 1)
  max_ngv = max(8 * m.ngeom + m.nmeshvert, 1)
  max_nge = max(12 * m.ngeom + m.nmeshpolyvert, 1)
  max_aset = nconmax if m.nflexvert > 0 else 1
  max_cand = nccdmax if m.nflexvert > 0 else 1

  ws = IpcWorkspace(
    w=wp.empty((nworld, nv), dtype=float, device=device),
    wfree=wp.empty((nworld, nv), dtype=float, device=device),
    wtil=wp.empty((nworld, nv), dtype=float, device=device),
    ddw=wp.empty((nworld, nv), dtype=float, device=device),
    da=wp.empty((nworld, nv), dtype=float, device=device),
    M_da=wp.empty((nworld, nv), dtype=float, device=device),
    qvel_old=wp.empty((nworld, nv), dtype=float, device=device),
    wn=wp.empty((nworld, nv), dtype=float, device=device),
    xold=wp.empty((nworld, nfv), dtype=wp.vec3, device=device),
    x=wp.empty((nworld, nfv), dtype=wp.vec3, device=device),
    xtil=wp.empty((nworld, nfv), dtype=wp.vec3, device=device),
    xfree=wp.empty((nworld, nfv), dtype=wp.vec3, device=device),
    xn=wp.empty((nworld, nfv), dtype=wp.vec3, device=device),
    pvel=wp.empty((nworld, nfv), dtype=wp.vec3, device=device),
    beta=wp.zeros(nworld, dtype=float, device=device),
    world_done=wp.zeros(nworld, dtype=int, device=device),
    stall=wp.zeros(nworld, dtype=int, device=device),
    saved_nefc=wp.zeros(nworld, dtype=int, device=device),
    active_count=wp.zeros(nworld, dtype=int, device=device),
    active_nnz=wp.zeros(nworld, dtype=int, device=device),
    total_solver_niter=wp.zeros(nworld, dtype=int, device=device),
    newton_converged=wp.zeros(nworld, dtype=int, device=device),
    last_ls_alpha=wp.zeros(nworld, dtype=float, device=device),
    ls_done=wp.zeros(nworld, dtype=int, device=device),
    ls_accept=wp.zeros(nworld, dtype=int, device=device),
    merit_energy=wp.zeros(nworld, dtype=float, device=device),
    merit_energy_trial=wp.zeros(nworld, dtype=float, device=device),
    nvertbody=wp.zeros(nbody, dtype=int, device=device),
    isflexvert_body=wp.zeros(nbody, dtype=int, device=device),
    npin=wp.zeros(nbody, dtype=int, device=device),
    apinned=wp.zeros(ntree, dtype=int, device=device),
    dofadr=wp.empty(nfv, dtype=int, device=device),
    fidx=wp.empty(nfv, dtype=int, device=device),
    mass=wp.empty((nworld, nfv), dtype=float, device=device),
    rad=wp.empty(nfv, dtype=float, device=device),
    pbody=wp.empty(nfv, dtype=int, device=device),
    own=wp.zeros(nv, dtype=bool, device=device),
    isflexdof=wp.zeros(nv, dtype=bool, device=device),
    isartictree=wp.zeros(ntree, dtype=bool, device=device),
    is_throttled=wp.zeros(nv, dtype=bool, device=device),
    max_chain=max_chain,
    pin_has_chain=wp.zeros(nfv, dtype=bool, device=device),
    pin_num_chain=wp.zeros(nfv, dtype=int, device=device),
    pin_chain_dof=wp.zeros((nfv, max_chain), dtype=int, device=device),
    pin_chain_axis=wp.empty((nworld, nfv, max_chain), dtype=wp.vec3, device=device),
    max_ngv=max_ngv,
    max_nge=max_nge,
    ngv=wp.zeros(nworld, dtype=int, device=device),
    nge=wp.zeros(nworld, dtype=int, device=device),
    geom_corner_loc=wp.empty((nworld, max_ngv), dtype=wp.vec3, device=device),
    geom_corners=wp.empty((nworld, max_ngv), dtype=wp.vec3, device=device),
    corner_geom=wp.empty((nworld, max_ngv), dtype=int, device=device),
    geom_edge_loc=wp.empty((nworld, max_nge, 2), dtype=wp.vec3, device=device),
    geom_edges=wp.empty((nworld, max_nge, 2), dtype=wp.vec3, device=device),
    edge_geom=wp.empty((nworld, max_nge), dtype=int, device=device),
    geom_aabb_min=wp.empty((nworld, ngeom), dtype=wp.vec3, device=device),
    geom_aabb_max=wp.empty((nworld, ngeom), dtype=wp.vec3, device=device),
    max_aset=max_aset,
    max_cand=max_cand,
    naset=wp.zeros(nworld, dtype=int, device=device),
    aset_type=wp.empty((nworld, max_aset), dtype=int, device=device),
    aset_idx=wp.empty((nworld, max_aset), dtype=wp.vec4i, device=device),
    aset_geom=wp.empty((nworld, max_aset), dtype=int, device=device),
    aset_standoff=wp.empty((nworld, max_aset), dtype=float, device=device),
    aset_mu=wp.empty((nworld, max_aset), dtype=float, device=device),
    aset_ld0=wp.empty((nworld, max_aset), dtype=float, device=device),
    aset_lam=wp.empty((nworld, max_aset), dtype=float, device=device),
    aset_cnt=wp.empty((nworld, max_aset), dtype=int, device=device),
    nret=wp.zeros(nworld, dtype=int, device=device),
    ret_type=wp.empty((nworld, max_aset), dtype=int, device=device),
    ret_idx=wp.empty((nworld, max_aset), dtype=wp.vec4i, device=device),
    ret_geom=wp.empty((nworld, max_aset), dtype=int, device=device),
    ret_standoff=wp.empty((nworld, max_aset), dtype=float, device=device),
    ret_mu=wp.empty((nworld, max_aset), dtype=float, device=device),
    ret_lam=wp.empty((nworld, max_aset), dtype=float, device=device),
    ret_cnt=wp.empty((nworld, max_aset), dtype=int, device=device),
    ncand=wp.zeros(nworld, dtype=int, device=device),
    cand_type=wp.empty((nworld, max_cand), dtype=int, device=device),
    cand_idx=wp.empty((nworld, max_cand), dtype=wp.vec4i, device=device),
    cand_geom=wp.empty((nworld, max_cand), dtype=int, device=device),
    cand_ld0=wp.empty((nworld, max_cand), dtype=float, device=device),
    cand_toi=wp.empty((nworld, max_cand), dtype=float, device=device),
    pair_ln=wp.empty((nworld, max_aset), dtype=wp.vec3, device=device),
    pair_lcw=wp.empty((nworld, max_aset), dtype=wp.vec4, device=device),
    pair_liv=wp.empty((nworld, max_aset), dtype=wp.vec4i, device=device),
    pair_lniv=wp.empty((nworld, max_aset), dtype=int, device=device),
    pair_rownnz=wp.empty((nworld, max_aset), dtype=int, device=device),
    pair_in_ws=wp.empty((nworld, max_aset), dtype=int, device=device),
    alpha_min=wp.zeros(nworld, dtype=float, device=device),
    vmin=wp.empty((nworld, nfv), dtype=float, device=device),
    ws_nadd=wp.zeros(nworld, dtype=int, device=device),
    ws_done=wp.zeros(nworld, dtype=int, device=device),
    qfrc_rows=wp.zeros((nworld, nv), dtype=float, device=device),
    outer_cond=wp.zeros(1, dtype=int, device=device),
    outer_iter=wp.zeros(1, dtype=int, device=device),
    ws_cond=wp.zeros(1, dtype=int, device=device),
    ws_iter=wp.zeros(1, dtype=int, device=device),
    ls_cond=wp.zeros(1, dtype=int, device=device),
    ls_iter=wp.zeros(1, dtype=int, device=device),
    ws_entry=wp.empty((nworld, nv), dtype=float, device=device),
    init_ncand=wp.zeros(nworld, dtype=int, device=device),
    dl=wp.empty((nworld, nfv), dtype=float, device=device),
    fsweep=wp.empty((nworld, nflex), dtype=float, device=device),
  )
  ipc_init_topology(m, ws)
  return ws


@wp.func
def _pos_mass(m_val: float) -> float:
  return m_val if m_val > 0.0 else float(1e30)


@wp.func
def _pair_standoff_and_mu(
  # In:
  worldid: int,
  pt: int,
  idx: wp.vec4i,
  rad: wp.array[float],
  mass: wp.array2d[float],
) -> wp.vec2:
  """Computes (standoff, mu) for a pair from vertex radii and masses (mjc_standoff + ipc_muPair)."""
  off, nv = pair_vert_range(pt)
  v0 = idx[off]
  r_min = rad[v0]
  m_raw = _pos_mass(mass[worldid, v0])
  for q in range(1, nv):
    vq = idx[off + q]
    r_min = wp.min(r_min, rad[vq])
    m_raw = wp.min(m_raw, _pos_mass(mass[worldid, vq]))
  m_min = 1e-9 if m_raw >= 1e29 else m_raw
  return wp.vec2(standoff(r_min, IPC_DELTACAP), FLEXCONTACT_OMEGA2 * m_min)


@wp.func
def _cnt_exp(cnt: int) -> int:
  """Maps signed contact age counter to decay exponent (ipc_cntExp)."""
  return cnt if cnt >= 0 else wp.max(0, -cnt - 6)


@wp.func
def _ipc_penalty_scale(mu: float, cnt: int) -> float:
  """Computes decayed augmented-Lagrangian penalty stiffness mu * IPC_DECAY^cexp."""
  return mu * wp.pow(IPC_DECAY, float(_cnt_exp(cnt)))


@wp.func
def _ipc_age_step(a: int, active: bool) -> int:
  """Updates signed contact age counter matching MuJoCo C ipc_ageStep."""
  if active:
    return 0 if (a == 0 or a > 5) else -1
  return a + (1 if a >= 0 else -1)


@wp.func
def _aset_retained(lam: float, cnt: int) -> bool:
  """Returns True if an active-set pair satisfies age and multiplier retention criteria."""
  return (lam > 0.0 or cnt >= -IPC_EVICT) and cnt <= IPC_ASET_AGE


@wp.func
def _con_lam_and_age(
  # Data in:
  flexvert_lambda_in: wp.array2d[float],
  flexvert_conage_in: wp.array2d[int],
  # In:
  worldid: int,
  pt: int,
  idx: wp.vec4i,
) -> tuple[float, int]:
  """Computes warm-started multiplier and contact age across participant vertices (ipc_conAge)."""
  off, nv = pair_vert_range(pt)
  max_lam = float(0.0)
  min_cnt = int(2147483647)
  for q in range(nv):
    vg = idx[off + q]
    lam_v = flexvert_lambda_in[worldid, vg]
    cnt_v = flexvert_conage_in[worldid, vg]
    if lam_v > max_lam:
      max_lam = lam_v
    if cnt_v < min_cnt:
      min_cnt = cnt_v
  return max_lam, (0 if min_cnt == 2147483647 else min_cnt)


@wp.func
def pair_linear_gap(
  # In:
  dd0: float,
  delta: float,
  ln: wp.vec3,
  liv: wp.vec4i,
  lcw: wp.vec4,
  lniv: int,
  x: wp.array2d[wp.vec3],
  xfree: wp.array2d[wp.vec3],
  worldid: int,
) -> float:
  """Evaluates linearized pair gap c_raw(x) = dd0 - delta + sum(lcw * dot(ln, dx))."""
  craw = dd0 - delta
  for k in range(lniv):
    vk = liv[k]
    craw += lcw[k] * wp.dot(ln, x[worldid, vk] - xfree[worldid, vk])
  return craw


@wp.func
def _idx_cmp(idx_a: wp.vec4i, idx_b: wp.vec4i) -> int:
  """Lexicographic comparison of two vec4i index tuples returning -1, 0, or 1."""
  if idx_a[0] != idx_b[0]:
    return -1 if idx_a[0] < idx_b[0] else 1
  if idx_a[1] != idx_b[1]:
    return -1 if idx_a[1] < idx_b[1] else 1
  if idx_a[2] != idx_b[2]:
    return -1 if idx_a[2] < idx_b[2] else 1
  if idx_a[3] != idx_b[3]:
    return -1 if idx_a[3] < idx_b[3] else 1
  return 0


@wp.kernel
def ipc_rank_sort_aset_kernel(
  # In:
  nret: wp.array[int],
  ret_type: wp.array2d[int],
  ret_idx: wp.array2d[wp.vec4i],
  ret_geom: wp.array2d[int],
  ret_standoff: wp.array2d[float],
  ret_mu: wp.array2d[float],
  ret_lam: wp.array2d[float],
  ret_cnt: wp.array2d[int],
  world_done: wp.array[int],
  # Out:
  naset_out: wp.array[int],
  aset_type_out: wp.array2d[int],
  aset_idx_out: wp.array2d[wp.vec4i],
  aset_geom_out: wp.array2d[int],
  aset_standoff_out: wp.array2d[float],
  aset_mu_out: wp.array2d[float],
  aset_lam_out: wp.array2d[float],
  aset_cnt_out: wp.array2d[int],
):
  """Sorts active-set pairs into canonical order by (type, geom, idx)."""
  worldid, i = wp.tid()
  if world_done[worldid] != 0:
    return
  n = wp.min(nret[worldid], ret_type.shape[1])
  if i == 0:
    naset_out[worldid] = n
  if i >= n:
    return

  ti = ret_type[worldid, i]
  gi = ret_geom[worldid, i]
  idxi = ret_idx[worldid, i]
  rank = int(0)
  for j in range(n):
    tj = ret_type[worldid, j]
    if tj != ti:
      if tj < ti:
        rank += 1
      continue
    gj = ret_geom[worldid, j]
    if gj != gi:
      if gj < gi:
        rank += 1
      continue
    c = _idx_cmp(ret_idx[worldid, j], idxi)
    if c < 0 or (c == 0 and j < i):
      rank += 1

  aset_type_out[worldid, rank] = ti
  aset_idx_out[worldid, rank] = idxi
  aset_geom_out[worldid, rank] = gi
  aset_standoff_out[worldid, rank] = ret_standoff[worldid, i]
  aset_mu_out[worldid, rank] = ret_mu[worldid, i]
  aset_lam_out[worldid, rank] = ret_lam[worldid, i]
  aset_cnt_out[worldid, rank] = ret_cnt[worldid, i]


@cache_kernel
def ipc_seed_active_set_kernel(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    flexvert_lambda_in: wp.array2d[float],
    flexvert_conage_in: wp.array2d[int],
    # In:
    rad: wp.array[float],
    mass: wp.array2d[float],
    ncand: wp.array[int],
    cand_type: wp.array2d[int],
    cand_idx: wp.array2d[wp.vec4i],
    cand_geom: wp.array2d[int],
    world_done: wp.array[int],
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    init_ncand_out: wp.array[int],
    nret_out: wp.array[int],
    ret_type_out: wp.array2d[int],
    ret_idx_out: wp.array2d[wp.vec4i],
    ret_geom_out: wp.array2d[int],
    ret_standoff_out: wp.array2d[float],
    ret_mu_out: wp.array2d[float],
    ret_lam_out: wp.array2d[float],
    ret_cnt_out: wp.array2d[int],
  ):
    """Seeds initial active-set scratch from step-start candidates (engine_ipc.c:902-933)."""
    worldid, c = wp.tid()
    if world_done[worldid] != 0:
      return
    n_raw = ncand[worldid]
    n = wp.min(n_raw, cand_type.shape[1])
    max_aset = ret_type_out.shape[1]
    n_seed = wp.min(n, max_aset)
    if c == 0:
      init_ncand_out[worldid] = 1 if n > 0 else 0
      nret_out[worldid] = n_seed
      if n_raw > max_aset:
        if wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)):
          wp.printf(
            "IPC active set overflow - please increase nconmax beyond %u or naconmax beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NARROWPHASE (or = 0 for all)\n",
            max_aset,
            max_aset * nret_out.shape[0],
          )
        wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.NARROWPHASE))
    if c >= n_seed:
      return

    pt = cand_type[worldid, c]
    idx = cand_idx[worldid, c]
    gi = cand_geom[worldid, c]
    sm = _pair_standoff_and_mu(worldid, pt, idx, rad, mass)
    lam, cnt = _con_lam_and_age(flexvert_lambda_in, flexvert_conage_in, worldid, pt, idx)

    ret_type_out[worldid, c] = pt
    ret_idx_out[worldid, c] = idx
    ret_geom_out[worldid, c] = gi
    ret_standoff_out[worldid, c] = sm[0]
    ret_mu_out[worldid, c] = sm[1]
    ret_lam_out[worldid, c] = lam
    ret_cnt_out[worldid, c] = cnt

  return kernel


def ipc_seed_active_set(m: Model, d: Data, ws: IpcWorkspace):
  """Seeds ws.ret from step-start candidates in ws.cand and rank-sorts into ws.aset."""
  wp.launch(
    ipc_seed_active_set_kernel(int(m.opt.warn_overflow)),
    dim=(d.nworld, ws.max_aset),
    inputs=[
      d.flexvert_lambda,
      d.flexvert_conage,
      ws.rad,
      ws.mass,
      ws.ncand,
      ws.cand_type,
      ws.cand_idx,
      ws.cand_geom,
      ws.world_done,
    ],
    outputs=[
      d.overflow,
      ws.init_ncand,
      ws.nret,
      ws.ret_type,
      ws.ret_idx,
      ws.ret_geom,
      ws.ret_standoff,
      ws.ret_mu,
      ws.ret_lam,
      ws.ret_cnt,
    ],
  )
  wp.launch(
    ipc_rank_sort_aset_kernel,
    dim=(d.nworld, ws.max_aset),
    inputs=[
      ws.nret,
      ws.ret_type,
      ws.ret_idx,
      ws.ret_geom,
      ws.ret_standoff,
      ws.ret_mu,
      ws.ret_lam,
      ws.ret_cnt,
      ws.world_done,
    ],
    outputs=[
      ws.naset,
      ws.aset_type,
      ws.aset_idx,
      ws.aset_geom,
      ws.aset_standoff,
      ws.aset_mu,
      ws.aset_lam,
      ws.aset_cnt,
    ],
  )


@wp.kernel
def ipc_compute_vmin_kernel(
  # In:
  ncand: wp.array[int],
  cand_type: wp.array2d[int],
  cand_idx: wp.array2d[wp.vec4i],
  cand_toi: wp.array2d[float],
  world_done: wp.array[int],
  # Out:
  vmin_out: wp.array2d[float],
):
  """Computes earliest time-of-impact per flex vertex across approaching candidates."""
  worldid, c = wp.tid()
  if world_done[worldid] != 0 or c >= wp.min(ncand[worldid], cand_type.shape[1]):
    return
  t = cand_toi[worldid, c]
  if t < 0.0:
    return
  idx = cand_idx[worldid, c]
  off, nv = pair_vert_range(cand_type[worldid, c])
  for q in range(nv):
    wp.atomic_min(vmin_out, worldid, idx[off + q], t)


@cache_kernel
def ipc_retain_active_set_kernel(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # In:
    naset: wp.array[int],
    aset_type: wp.array2d[int],
    aset_idx: wp.array2d[wp.vec4i],
    aset_geom: wp.array2d[int],
    aset_standoff: wp.array2d[float],
    aset_mu: wp.array2d[float],
    aset_lam: wp.array2d[float],
    aset_cnt: wp.array2d[int],
    world_done: wp.array[int],
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    nret_out: wp.array[int],
    ret_type_out: wp.array2d[int],
    ret_idx_out: wp.array2d[wp.vec4i],
    ret_geom_out: wp.array2d[int],
    ret_standoff_out: wp.array2d[float],
    ret_mu_out: wp.array2d[float],
    ret_lam_out: wp.array2d[float],
    ret_cnt_out: wp.array2d[int],
  ):
    """Compacts active-set pairs that pass retention criteria into ret buffer."""
    worldid, s = wp.tid()
    if world_done[worldid] != 0 or s >= wp.min(naset[worldid], aset_type.shape[1]):
      return
    lam = aset_lam[worldid, s]
    cnt = aset_cnt[worldid, s]
    if _aset_retained(lam, cnt):
      r = wp.atomic_add(nret_out, worldid, 1)
      max_aset = ret_type_out.shape[1]
      if r >= max_aset:
        if wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)) and r == max_aset:
          wp.printf(
            "IPC active set overflow - please increase nconmax beyond %u or naconmax beyond %u\n"
            "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NARROWPHASE (or = 0 for all)\n",
            max_aset,
            max_aset * nret_out.shape[0],
          )
        wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.NARROWPHASE))
        return
      ret_type_out[worldid, r] = aset_type[worldid, s]
      ret_idx_out[worldid, r] = aset_idx[worldid, s]
      ret_geom_out[worldid, r] = aset_geom[worldid, s]
      ret_standoff_out[worldid, r] = aset_standoff[worldid, s]
      ret_mu_out[worldid, r] = aset_mu[worldid, s]
      ret_lam_out[worldid, r] = lam
      ret_cnt_out[worldid, r] = cnt

  return kernel


@cache_kernel
def ipc_admit_candidates_kernel(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    flexvert_lambda_in: wp.array2d[float],
    flexvert_conage_in: wp.array2d[int],
    # In:
    rad: wp.array[float],
    mass: wp.array2d[float],
    ncand: wp.array[int],
    cand_type: wp.array2d[int],
    cand_idx: wp.array2d[wp.vec4i],
    cand_geom: wp.array2d[int],
    cand_ld0: wp.array2d[float],
    cand_toi: wp.array2d[float],
    vmin: wp.array2d[float],
    naset: wp.array[int],
    aset_type: wp.array2d[int],
    aset_idx: wp.array2d[wp.vec4i],
    aset_geom: wp.array2d[int],
    aset_lam: wp.array2d[float],
    aset_cnt: wp.array2d[int],
    world_done: wp.array[int],
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    nret_out: wp.array[int],
    ret_type_out: wp.array2d[int],
    ret_idx_out: wp.array2d[wp.vec4i],
    ret_geom_out: wp.array2d[int],
    ret_standoff_out: wp.array2d[float],
    ret_mu_out: wp.array2d[float],
    ret_lam_out: wp.array2d[float],
    ret_cnt_out: wp.array2d[int],
  ):
    """Merges newly admitted CCD candidates into ret, deduplicating retained pairs."""
    worldid, c = wp.tid()
    if world_done[worldid] != 0 or c >= wp.min(ncand[worldid], cand_type.shape[1]):
      return

    dd = cand_ld0[worldid, c]
    if dd >= 1e20:
      return

    pt = cand_type[worldid, c]
    idx = cand_idx[worldid, c]
    gi = cand_geom[worldid, c]
    t = cand_toi[worldid, c]

    admit = int(0)
    if dd <= 0.0:
      admit = 1
    elif t >= 0.0:
      off, nv = pair_vert_range(pt)
      for q in range(nv):
        if t <= vmin[worldid, idx[off + q]] + 1e-12:
          admit = 1
          break

    if admit == 0:
      return

    naset_w = wp.min(naset[worldid], aset_type.shape[1])
    lo = int(0)
    hi = naset_w - 1
    while lo <= hi:
      mid = (lo + hi) >> 1
      tm = aset_type[worldid, mid]
      if tm != pt:
        cmp = -1 if tm < pt else 1
      else:
        gm = aset_geom[worldid, mid]
        if gm != gi:
          cmp = -1 if gm < gi else 1
        else:
          cmp = _idx_cmp(aset_idx[worldid, mid], idx)
      if cmp < 0:
        lo = mid + 1
      elif cmp > 0:
        hi = mid - 1
      else:
        if _aset_retained(aset_lam[worldid, mid], aset_cnt[worldid, mid]):
          return
        break

    sm = _pair_standoff_and_mu(worldid, pt, idx, rad, mass)
    lam, cnt = _con_lam_and_age(flexvert_lambda_in, flexvert_conage_in, worldid, pt, idx)

    s = wp.atomic_add(nret_out, worldid, 1)
    max_aset = ret_type_out.shape[1]
    if s >= max_aset:
      if wp.static(bool(warn_overflow & OverflowType.NARROWPHASE)) and s == max_aset:
        wp.printf(
          "IPC active set overflow - please increase nconmax beyond %u or naconmax beyond %u\n"
          "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.NARROWPHASE (or = 0 for all)\n",
          max_aset,
          max_aset * nret_out.shape[0],
        )
      wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.NARROWPHASE))
      return
    ret_type_out[worldid, s] = pt
    ret_idx_out[worldid, s] = idx
    ret_geom_out[worldid, s] = gi
    ret_standoff_out[worldid, s] = sm[0]
    ret_mu_out[worldid, s] = sm[1]
    ret_lam_out[worldid, s] = lam
    ret_cnt_out[worldid, s] = cnt

  return kernel


def ipc_merge_active_set(m: Model, d: Data, ws: IpcWorkspace):
  """Filters earliest-ToI candidates, retains contacts, and merges pairs (ipc_mergeActiveSet)."""
  nworld = d.nworld
  max_aset = ws.max_aset
  warn_overflow = int(m.opt.warn_overflow)
  ws.vmin.fill_(2.0)
  wp.launch(
    ipc_compute_vmin_kernel,
    dim=(nworld, ws.max_cand),
    inputs=[
      ws.ncand,
      ws.cand_type,
      ws.cand_idx,
      ws.cand_toi,
      ws.world_done,
    ],
    outputs=[ws.vmin],
  )
  ws.nret.zero_()
  wp.launch(
    ipc_retain_active_set_kernel(warn_overflow),
    dim=(nworld, max_aset),
    inputs=[
      ws.naset,
      ws.aset_type,
      ws.aset_idx,
      ws.aset_geom,
      ws.aset_standoff,
      ws.aset_mu,
      ws.aset_lam,
      ws.aset_cnt,
      ws.world_done,
    ],
    outputs=[
      d.overflow,
      ws.nret,
      ws.ret_type,
      ws.ret_idx,
      ws.ret_geom,
      ws.ret_standoff,
      ws.ret_mu,
      ws.ret_lam,
      ws.ret_cnt,
    ],
  )
  wp.launch(
    ipc_admit_candidates_kernel(warn_overflow),
    dim=(nworld, ws.max_cand),
    inputs=[
      d.flexvert_lambda,
      d.flexvert_conage,
      ws.rad,
      ws.mass,
      ws.ncand,
      ws.cand_type,
      ws.cand_idx,
      ws.cand_geom,
      ws.cand_ld0,
      ws.cand_toi,
      ws.vmin,
      ws.naset,
      ws.aset_type,
      ws.aset_idx,
      ws.aset_geom,
      ws.aset_lam,
      ws.aset_cnt,
      ws.world_done,
    ],
    outputs=[
      d.overflow,
      ws.nret,
      ws.ret_type,
      ws.ret_idx,
      ws.ret_geom,
      ws.ret_standoff,
      ws.ret_mu,
      ws.ret_lam,
      ws.ret_cnt,
    ],
  )
  wp.launch(
    ipc_rank_sort_aset_kernel,
    dim=(nworld, max_aset),
    inputs=[
      ws.nret,
      ws.ret_type,
      ws.ret_idx,
      ws.ret_geom,
      ws.ret_standoff,
      ws.ret_mu,
      ws.ret_lam,
      ws.ret_cnt,
      ws.world_done,
    ],
    outputs=[
      ws.naset,
      ws.aset_type,
      ws.aset_idx,
      ws.aset_geom,
      ws.aset_standoff,
      ws.aset_mu,
      ws.aset_lam,
      ws.aset_cnt,
    ],
  )


@wp.func
def _efc_row_cost(t: int, D: float, fl: float, jar: float) -> float:
  """Evaluates primal constraint cost for a native EFC row."""
  if t == int(ConstraintType.EQUALITY):
    return 0.5 * D * jar * jar
  if t == int(ConstraintType.FRICTION_DOF) or t == int(ConstraintType.FRICTION_TENDON):
    R = 1.0 / wp.max(D, 1e-12)
    if jar <= -R * fl:
      return -0.5 * R * fl * fl - fl * jar
    if jar >= R * fl:
      return -0.5 * R * fl * fl + fl * jar
    return 0.5 * D * jar * jar
  if jar < 0.0:
    return 0.5 * D * jar * jar
  return 0.0


@wp.func
def _efc_jar_dense(
  # Model:
  nv: int,
  # Data in:
  efc_J_in: wp.array3d[float],
  efc_aref_in: wp.array2d[float],
  # In:
  worldid: int,
  r: int,
  qacc_eval: wp.array2d[float],
) -> float:
  jar = -efc_aref_in[worldid, r]
  for j in range(nv):
    jar += efc_J_in[worldid, r, j] * qacc_eval[worldid, j]
  return jar


@wp.func
def _efc_jar_sparse(
  # Data in:
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_aref_in: wp.array2d[float],
  # In:
  worldid: int,
  r: int,
  qacc_eval: wp.array2d[float],
) -> float:
  jar = -efc_aref_in[worldid, r]
  nnz = efc_J_rownnz_in[worldid, r]
  adr = efc_J_rowadr_in[worldid, r]
  for k in range(nnz):
    col = efc_J_colind_in[worldid, 0, adr + k]
    val = efc_J_in[worldid, 0, adr + k]
    jar += val * qacc_eval[worldid, col]
  return jar


@wp.func
def _efc_elliptic_cost(
  # In:
  jar0: float,
  D0: float,
  mu: float,
  TT: float,
  quad_tan: float,
) -> float:
  N = jar0 * mu
  T = wp.sqrt(TT) if TT > 0.0 else 0.0
  if (N >= mu * T) or ((T <= 0.0) and (N >= 0.0)):
    return 0.0
  if (mu * N + T <= 0.0) or ((T <= 0.0) and (N < 0.0)):
    return 0.5 * D0 * jar0 * jar0 + quad_tan
  dm = math.safe_div(D0, mu * mu * (1.0 + mu * mu))
  nmt = N - mu * T
  return 0.5 * dm * nmt * nmt


@wp.kernel
def _ipc_init_step_kernel(
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  # In:
  own: wp.array[bool],
  timestep: wp.array[float],
  # Out:
  w_out: wp.array2d[float],
  wtil_out: wp.array2d[float],
  wfree_out: wp.array2d[float],
  ddw_out: wp.array2d[float],
  beta_out: wp.array[float],
  world_done_out: wp.array[int],
  stall_out: wp.array[int],
  total_solver_niter_out: wp.array[int],
  newton_converged_out: wp.array[int],
  last_ls_alpha_out: wp.array[float],
  outer_cond_out: wp.array[int],
  outer_iter_out: wp.array[int],
  init_ncand_out: wp.array[int],
  naset_out: wp.array[int],
):
  worldid, dofid = wp.tid()
  h = timestep[worldid % timestep.shape[0]]
  v = qvel_in[worldid, dofid]
  w_out[worldid, dofid] = 0.0
  wfree_out[worldid, dofid] = 0.0
  ddw_out[worldid, dofid] = 0.0
  if own[dofid]:
    wtil_out[worldid, dofid] = v + h * qacc_smooth_in[worldid, dofid]
  else:
    wtil_out[worldid, dofid] = 0.0
  if dofid == 0:
    beta_out[worldid] = 0.0
    world_done_out[worldid] = 0
    stall_out[worldid] = 0
    total_solver_niter_out[worldid] = 0
    newton_converged_out[worldid] = 0
    last_ls_alpha_out[worldid] = 1.0
    init_ncand_out[worldid] = 0
    naset_out[worldid] = 0
    if worldid == 0:
      outer_cond_out[0] = 1
      outer_iter_out[0] = 0


@wp.kernel
def _ipc_free_flight_init_wx_kernel(
  # Model:
  nv: int,
  # In:
  nfv: int,
  isflexdof: wp.array[bool],
  fidx: wp.array[int],
  init_ncand: wp.array[int],
  wtil: wp.array2d[float],
  xtil: wp.array2d[wp.vec3],
  # Out:
  w_out: wp.array2d[float],
  x_out: wp.array2d[wp.vec3],
):
  worldid, tid = wp.tid()
  if init_ncand[worldid] != 0:
    return
  if tid < nv and isflexdof[tid]:
    w_out[worldid, tid] = wtil[worldid, tid]
  if tid < nfv and fidx[tid] >= 0:
    x_out[worldid, tid] = xtil[worldid, tid]


@wp.kernel
def _ipc_init_verts_step_kernel(
  # Model:
  jnt_bodyid: wp.array[int],
  jnt_axis: wp.array2d[wp.vec3],
  dof_jntid: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  xmat_in: wp.array2d[wp.mat33],
  cdof_in: wp.array2d[wp.spatial_vector],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  # In:
  fidx: wp.array[int],
  dofadr: wp.array[int],
  pbody: wp.array[int],
  pin_has_chain: wp.array[bool],
  pin_num_chain: wp.array[int],
  pin_chain_dof: wp.array2d[int],
  wtil: wp.array2d[float],
  timestep: wp.array[float],
  # Out:
  pin_chain_axis_out: wp.array3d[wp.vec3],
  pvel_out: wp.array2d[wp.vec3],
  xold_out: wp.array2d[wp.vec3],
  xfree_out: wp.array2d[wp.vec3],
  x_out: wp.array2d[wp.vec3],
  xtil_out: wp.array2d[wp.vec3],
):
  """Fuses step-start xold/xfree/x copy, pinned chain axis/pvel update, and xtil = points(wtil)."""
  worldid, vg = wp.tid()
  h = timestep[worldid % timestep.shape[0]]
  x0 = flexvert_xpos_in[worldid, vg]
  xold_out[worldid, vg] = x0
  xfree_out[worldid, vg] = x0
  x_out[worldid, vg] = x0

  pv = wp.vec3(0.0, 0.0, 0.0)
  if fidx[vg] >= 0:
    da = dofadr[vg]
    R = xmat_in[worldid, pbody[vg]]
    wv = wp.vec3(wtil[worldid, da], wtil[worldid, da + 1], wtil[worldid, da + 2])
    xtil_out[worldid, vg] = x0 + (R * wv) * h
  elif pin_has_chain[vg]:
    nchain = pin_num_chain[vg]
    disp = wp.vec3(0.0, 0.0, 0.0)
    for j in range(nchain):
      da = pin_chain_dof[vg, j]
      ax = wp.spatial_bottom(cdof_in[worldid, da])
      if wp.dot(ax, ax) < 1e-12:
        jid = dof_jntid[da]
        bid = jnt_bodyid[jid]
        loc_ax = jnt_axis[worldid % jnt_axis.shape[0], jid]
        ax = xmat_in[worldid, bid] * loc_ax
        ax_len = wp.length(ax)
        if ax_len > 1e-12:
          ax = ax / ax_len
      pin_chain_axis_out[worldid, vg, j] = ax
      pv += ax * qvel_in[worldid, da]
      disp += ax * wtil[worldid, da]
    xtil_out[worldid, vg] = x0 + disp * h
  else:
    xtil_out[worldid, vg] = x0
  pvel_out[worldid, vg] = pv


@wp.kernel
def _ipc_points_kernel(
  # Data in:
  xmat_in: wp.array2d[wp.mat33],
  # In:
  fidx: wp.array[int],
  dofadr: wp.array[int],
  pbody: wp.array[int],
  pin_has_chain: wp.array[bool],
  pin_num_chain: wp.array[int],
  pin_chain_dof: wp.array2d[int],
  pin_chain_axis: wp.array3d[wp.vec3],
  w_arr: wp.array2d[float],
  xold: wp.array2d[wp.vec3],
  timestep: wp.array[float],
  skip_world: wp.array[int],
  # Out:
  x_out: wp.array2d[wp.vec3],
):
  worldid, pt = wp.tid()
  if skip_world[worldid] != 0:
    return
  h = timestep[worldid % timestep.shape[0]]
  x0 = xold[worldid, pt]
  if fidx[pt] >= 0:
    da = dofadr[pt]
    R = xmat_in[worldid, pbody[pt]]
    wv = wp.vec3(w_arr[worldid, da], w_arr[worldid, da + 1], w_arr[worldid, da + 2])
    x_out[worldid, pt] = x0 + (R * wv) * h
  elif pin_has_chain[pt]:
    nchain = pin_num_chain[pt]
    disp = wp.vec3(0.0, 0.0, 0.0)
    for j in range(nchain):
      disp += pin_chain_axis[worldid, pt, j] * w_arr[worldid, pin_chain_dof[pt, j]]
    x_out[worldid, pt] = x0 + disp * h
  else:
    x_out[worldid, pt] = x0


def _points(
  m: Model,
  d: Data,
  ws: IpcWorkspace,
  w_arr: wp.array2d[float],
  x_out: wp.array2d[wp.vec3],
  skip_world: wp.array[int],
):
  """Evaluates world positions of 2D flex vertices under velocity proposal w_arr (ipc_points)."""
  wp.launch(
    _ipc_points_kernel,
    dim=(d.nworld, m.nflexvert),
    inputs=[
      d.xmat,
      ws.fidx,
      ws.dofadr,
      ws.pbody,
      ws.pin_has_chain,
      ws.pin_num_chain,
      ws.pin_chain_dof,
      ws.pin_chain_axis,
      w_arr,
      ws.xold,
      m.opt.timestep,
      skip_world,
    ],
    outputs=[x_out],
  )


@wp.kernel
def _ipc_check_convergence_kernel(
  # Model:
  nv: int,
  # In:
  own: wp.array[bool],
  ddw: wp.array2d[float],
  # Out:
  newton_converged_out: wp.array[int],
):
  worldid, tid = wp.tid()
  local_max = float(0.0)
  BLOCK_DIM = wp.block_dim()

  for i in range(tid, nv, BLOCK_DIM):
    if own[i]:
      a = wp.abs(ddw[worldid, i])
      if a > local_max:
        local_max = a

  max_red = wp.tile_reduce(wp.max, wp.tile(local_max, preserve_type=True))
  if tid == 0:
    newton_converged_out[worldid] = 1 if max_red[0] <= IPC_VEL_TOL else 0


@wp.kernel
def _ipc_clear_flexvert_lambda_kernel(
  # In:
  world_done: wp.array[int],
  # Data out:
  flexvert_lambda_out: wp.array2d[float],
):
  worldid, v = wp.tid()
  if world_done[worldid] != 0:
    return
  flexvert_lambda_out[worldid, v] = 0.0


@wp.kernel
def _ipc_trial_w_kernel(
  # In:
  own: wp.array[bool],
  world_done: wp.array[int],
  ls_done: wp.array[int],
  ls_iter: wp.array[int],
  w: wp.array2d[float],
  ddw: wp.array2d[float],
  # Out:
  wn_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if world_done[worldid] != 0 or ls_done[worldid] != 0:
    return
  alpha = 1.0 / float(1 << ls_iter[0])
  if own[dofid]:
    wn_out[worldid, dofid] = w[worldid, dofid] + alpha * ddw[worldid, dofid]
  else:
    wn_out[worldid, dofid] = 0.0


@wp.kernel
def _ipc_linesearch_eval_trial_kernel(
  # In:
  world_done: wp.array[int],
  ls_done: wp.array[int],
  ls_iter: wp.array[int],
  beta: wp.array[float],
  merit_energy: wp.array[float],
  merit_energy_trial: wp.array[float],
  newton_converged: wp.array[int],
  # Out:
  ls_done_out: wp.array[int],
  last_ls_alpha_out: wp.array[float],
  ls_accept_out: wp.array[int],
):
  worldid = wp.tid()
  if world_done[worldid] != 0 or ls_done[worldid] != 0:
    ls_accept_out[worldid] = 0
    return
  it = ls_iter[0]
  alpha = 1.0 / float(1 << it)
  if merit_energy_trial[worldid] <= merit_energy[worldid] + 1e-12 or newton_converged[worldid] != 0:
    ls_done_out[worldid] = 1
    last_ls_alpha_out[worldid] = alpha
    ls_accept_out[worldid] = 1
  elif it >= 7:
    ls_done_out[worldid] = 1
    if beta[worldid] >= 1.0 - 1e-6:
      last_ls_alpha_out[worldid] = 0.0
      ls_accept_out[worldid] = 0
    else:
      last_ls_alpha_out[worldid] = alpha
      ls_accept_out[worldid] = 1
  else:
    ls_accept_out[worldid] = 0


@wp.kernel
def _ipc_linesearch_commit_wx_kernel(
  # Model:
  nv: int,
  # Data in:
  nworld_in: int,
  # In:
  nfv: int,
  world_done_in: wp.array[int],
  ls_done_in: wp.array[int],
  accept: wp.array[int],
  wn: wp.array2d[float],
  xn: wp.array2d[wp.vec3],
  # Out:
  w_out: wp.array2d[float],
  x_out: wp.array2d[wp.vec3],
  ls_iter_out: wp.array[int],
  ls_cond_out: wp.array[int],
):
  worldid, tid = wp.tid()
  if accept[worldid] != 0:
    if tid < nv:
      w_out[worldid, tid] = wn[worldid, tid]
    if tid < nfv:
      x_out[worldid, tid] = xn[worldid, tid]
  if worldid == 0 and tid == 0:
    it = ls_iter_out[0] + 1
    ls_iter_out[0] = it
    all_done = int(1)
    for w in range(nworld_in):
      if world_done_in[w] == 0 and ls_done_in[w] == 0:
        all_done = int(0)
        break
    ls_cond_out[0] = 0 if (all_done == 1 or it >= 8) else 1


@wp.kernel
def _ipc_eval_merit_contact_kernel(
  # In:
  naset: wp.array[int],
  aset_standoff: wp.array2d[float],
  aset_mu: wp.array2d[float],
  aset_ld0: wp.array2d[float],
  aset_lam: wp.array2d[float],
  aset_cnt: wp.array2d[int],
  pair_ln: wp.array2d[wp.vec3],
  pair_liv: wp.array2d[wp.vec4i],
  pair_lcw: wp.array2d[wp.vec4],
  pair_lniv: wp.array2d[int],
  x_tr: wp.array2d[wp.vec3],
  xfree: wp.array2d[wp.vec3],
  skip_world: wp.array[int],
  # Out:
  merit_out: wp.array[float],
):
  """Evaluates AL contact penalty cost across the active set (ipc_contactCost)."""
  worldid, tid = wp.tid()
  if skip_world[worldid] != 0:
    return

  local_val = float(0.0)
  BLOCK_DIM = wp.block_dim()
  naset_w = wp.min(naset[worldid], aset_ld0.shape[1])

  for s in range(tid, naset_w, BLOCK_DIM):
    lniv = pair_lniv[worldid, s]
    if lniv <= 0:
      continue
    craw = pair_linear_gap(
      aset_ld0[worldid, s],
      aset_standoff[worldid, s],
      pair_ln[worldid, s],
      pair_liv[worldid, s],
      pair_lcw[worldid, s],
      lniv,
      x_tr,
      xfree,
      worldid,
    )
    mu = aset_mu[worldid, s]
    dd = craw - aset_lam[worldid, s] / wp.max(mu, 1e-12)
    if dd < 0.0:
      scale = _ipc_penalty_scale(mu, aset_cnt[worldid, s])
      local_val += 0.5 * scale * dd * dd

  val_tile = wp.tile(local_val, preserve_type=True)
  val_sum = wp.tile_reduce(wp.add, val_tile)
  if tid == 0:
    merit_out[worldid] += val_sum[0]


@wp.kernel
def _ipc_tangent_da_kernel(
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  # In:
  own: wp.array[bool],
  timestep: wp.array[float],
  w: wp.array2d[float],
  skip_world: wp.array[int],
  # Out:
  da_out: wp.array2d[float],
  qacc_eval_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if skip_world[worldid] != 0:
    return
  h = timestep[worldid % timestep.shape[0]]
  if own[dofid]:
    qa = (w[worldid, dofid] - qvel_in[worldid, dofid]) / h
    qacc_eval_out[worldid, dofid] = qa
    da_out[worldid, dofid] = qa - qacc_smooth_in[worldid, dofid]
  else:
    qacc_eval_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]
    da_out[worldid, dofid] = 0.0


@wp.kernel
def _ipc_eval_gauss_energy_kernel(
  # Model:
  nv: int,
  # In:
  timestep: wp.array[float],
  da: wp.array2d[float],
  M_da: wp.array2d[float],
  skip_world: wp.array[int],
  # Out:
  merit_energy_out: wp.array[float],
):
  """Evaluates scaled Gauss energy 0.5 * h^2 * da^T M_hat da (ipc_gaussCost)."""
  worldid, tid = wp.tid()
  if skip_world[worldid] != 0:
    return
  h = timestep[worldid % timestep.shape[0]]
  h2_half = 0.5 * h * h
  local_val = float(0.0)
  BLOCK_DIM = wp.block_dim()
  for dofid in range(tid, nv, BLOCK_DIM):
    local_val += h2_half * da[worldid, dofid] * M_da[worldid, dofid]

  val_tile = wp.tile(local_val, preserve_type=True)
  val_sum = wp.tile_reduce(wp.add, val_tile)
  if tid == 0:
    merit_energy_out[worldid] = val_sum[0]


@cache_kernel
def _ipc_eval_merit_native_efc_kernel(is_sparse: bool):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    opt_impratio_invsqrt: wp.array[float],
    # Data in:
    contact_friction_in: wp.array[vec5],
    contact_dim_in: wp.array[int],
    contact_efc_address_in: wp.array2d[int],
    efc_type_in: wp.array2d[int],
    efc_id_in: wp.array2d[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    efc_J_colind_in: wp.array3d[int],
    efc_J_in: wp.array3d[float],
    efc_D_in: wp.array2d[float],
    efc_aref_in: wp.array2d[float],
    efc_frictionloss_in: wp.array2d[float],
    nacon_in: wp.array[int],
    # In:
    saved_nefc: wp.array[int],
    timestep: wp.array[float],
    qacc_eval: wp.array2d[float],
    skip_world: wp.array[int],
    # Out:
    merit_energy_out: wp.array[float],
  ):
    worldid, tid = wp.tid()
    if skip_world[worldid] != 0:
      return
    h = timestep[worldid % timestep.shape[0]]
    h2 = h * h
    nefc0 = wp.min(saved_nefc[worldid], efc_type_in.shape[1])
    local_val = float(0.0)
    BLOCK_DIM = wp.block_dim()

    for r in range(tid, nefc0, BLOCK_DIM):
      t = efc_type_in[worldid, r]
      c = float(0.0)
      if t == int(ConstraintType.CONTACT_ELLIPTIC):
        conid = efc_id_in[worldid, r]
        if conid >= 0 and conid < wp.min(nacon_in[0], contact_dim_in.shape[0]):
          efcid0 = contact_efc_address_in[conid, 0]
          if r == efcid0:
            dim = contact_dim_in[conid]
            friction = contact_friction_in[conid]
            mu = friction[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]
            jar0 = (
              _efc_jar_sparse(
                efc_J_rownnz_in, efc_J_rowadr_in, efc_J_colind_in, efc_J_in, efc_aref_in, worldid, efcid0, qacc_eval
              )
              if wp.static(is_sparse)
              else _efc_jar_dense(nv, efc_J_in, efc_aref_in, worldid, efcid0, qacc_eval)
            )
            D0 = efc_D_in[worldid, efcid0]
            TT = float(0.0)
            quad_tan = float(0.0)
            for j in range(1, dim):
              efcidj = contact_efc_address_in[conid, j]
              if efcidj >= 0:
                jarj = (
                  _efc_jar_sparse(
                    efc_J_rownnz_in, efc_J_rowadr_in, efc_J_colind_in, efc_J_in, efc_aref_in, worldid, efcidj, qacc_eval
                  )
                  if wp.static(is_sparse)
                  else _efc_jar_dense(nv, efc_J_in, efc_aref_in, worldid, efcidj, qacc_eval)
                )
                uj = jarj * friction[j - 1]
                TT += uj * uj
                quad_tan += 0.5 * efc_D_in[worldid, efcidj] * jarj * jarj
            c = _efc_elliptic_cost(jar0, D0, mu, TT, quad_tan)
      else:
        jar = (
          _efc_jar_sparse(efc_J_rownnz_in, efc_J_rowadr_in, efc_J_colind_in, efc_J_in, efc_aref_in, worldid, r, qacc_eval)
          if wp.static(is_sparse)
          else _efc_jar_dense(nv, efc_J_in, efc_aref_in, worldid, r, qacc_eval)
        )
        c = _efc_row_cost(t, efc_D_in[worldid, r], efc_frictionloss_in[worldid, r], jar)
      local_val += h2 * c

    val_tile = wp.tile(local_val, preserve_type=True)
    val_sum = wp.tile_reduce(wp.add, val_tile)
    if tid == 0:
      merit_energy_out[worldid] += val_sum[0]

  return kernel


@wp.kernel
def _ipc_clear_qfrc_rows_kernel(
  # In:
  ws_done: wp.array[int],
  # Out:
  qfrc_rows_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if ws_done[worldid] != 0:
    return
  qfrc_rows_out[worldid, dofid] = 0.0


@wp.kernel
def _ipc_eval_row_force_dense_kernel(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  efc_force_in: wp.array2d[float],
  # In:
  saved_nefc: wp.array[int],
  ws_done: wp.array[int],
  # Out:
  qfrc_rows_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if ws_done[worldid] != 0:
    return
  njmax = efc_force_in.shape[1]
  nefc0 = wp.min(saved_nefc[worldid], njmax)
  nefc1 = wp.min(nefc_in[worldid], njmax)
  res = float(0.0)
  for r in range(nefc0, nefc1):
    f = efc_force_in[worldid, r]
    if f != 0.0:
      res += efc_J_in[worldid, r, dofid] * f
  qfrc_rows_out[worldid, dofid] = res


@wp.kernel
def _ipc_eval_row_force_sparse_kernel(
  # Data in:
  nefc_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  efc_force_in: wp.array2d[float],
  # In:
  saved_nefc: wp.array[int],
  ws_done: wp.array[int],
  # Out:
  qfrc_rows_out: wp.array2d[float],
):
  worldid, r_idx = wp.tid()
  if ws_done[worldid] != 0:
    return
  njmax = efc_force_in.shape[1]
  nefc0 = wp.min(saved_nefc[worldid], njmax)
  nefc1 = wp.min(nefc_in[worldid], njmax)
  r = nefc0 + r_idx
  if r >= nefc1:
    return
  f = efc_force_in[worldid, r]
  if f == 0.0:
    return
  nnz = efc_J_rownnz_in[worldid, r]
  adr = efc_J_rowadr_in[worldid, r]
  for a in range(nnz):
    col = efc_J_colind_in[worldid, 0, adr + a]
    val = efc_J_in[worldid, 0, adr + a]
    wp.atomic_add(qfrc_rows_out, worldid, col, val * f)


def _efc_row_force(m: Model, d: Data, ws: IpcWorkspace):
  """Accumulates generalized constraint forces from published IPC rows (ipc_efcRowForce)."""
  if m.is_sparse:
    wp.launch(_ipc_clear_qfrc_rows_kernel, dim=(d.nworld, m.nv), inputs=[ws.ws_done], outputs=[ws.qfrc_rows])
    wp.launch(
      _ipc_eval_row_force_sparse_kernel,
      dim=(d.nworld, d.njmax),
      inputs=[
        d.nefc,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.efc.force,
        ws.saved_nefc,
        ws.ws_done,
      ],
      outputs=[ws.qfrc_rows],
    )
  else:
    wp.launch(
      _ipc_eval_row_force_dense_kernel,
      dim=(d.nworld, m.nv),
      inputs=[
        d.nefc,
        d.efc.J,
        d.efc.force,
        ws.saved_nefc,
        ws.ws_done,
      ],
      outputs=[ws.qfrc_rows],
    )


@wp.kernel
def _ipc_add_qfrc_rows_kernel(
  # In:
  qfrc_rows: wp.array2d[float],
  # Data out:
  qfrc_constraint_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  qfrc_constraint_out[worldid, dofid] += qfrc_rows[worldid, dofid]


@wp.kernel
def _ipc_flexvert_age_step_kernel(
  # Model:
  flex_dim: wp.array[int],
  flex_vertflexid: wp.array[int],
  # Data in:
  flexvert_lambda_in: wp.array2d[float],
  # In:
  world_done: wp.array[int],
  # Data out:
  flexvert_conage_out: wp.array2d[int],
):
  worldid, v = wp.tid()
  if world_done[worldid] != 0:
    return
  if flex_dim[flex_vertflexid[v]] != 2:
    return
  flexvert_conage_out[worldid, v] = _ipc_age_step(
    flexvert_conage_out[worldid, v],
    flexvert_lambda_in[worldid, v] > 0.0,
  )


@wp.kernel
def _ipc_eval_pairs_kernel(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  naset: wp.array[int],
  aset_type: wp.array2d[int],
  aset_idx: wp.array2d[wp.vec4i],
  aset_geom: wp.array2d[int],
  aset_standoff: wp.array2d[float],
  aset_mu: wp.array2d[float],
  aset_lam: wp.array2d[float],
  geom_corners: wp.array2d[wp.vec3],
  geom_edges: wp.array3d[wp.vec3],
  world_done: wp.array[int],
  x: wp.array2d[wp.vec3],
  xtil: wp.array2d[wp.vec3],
  xfree: wp.array2d[wp.vec3],
  dofadr: wp.array[int],
  pin_has_chain: wp.array[bool],
  pin_num_chain: wp.array[int],
  # Out:
  aset_ld0_out: wp.array2d[float],
  pair_ln_out: wp.array2d[wp.vec3],
  pair_lcw_out: wp.array2d[wp.vec4],
  pair_liv_out: wp.array2d[wp.vec4i],
  pair_lniv_out: wp.array2d[int],
  pair_rownnz_out: wp.array2d[int],
  pair_in_ws_out: wp.array2d[int],
):
  """Linearizes active-set pairs at xfree using pair_gap and selects initial working set."""
  worldid, s = wp.tid()
  if world_done[worldid] != 0 or s >= wp.min(naset[worldid], aset_type.shape[1]):
    return

  pt = aset_type[worldid, s]
  idx = aset_idx[worldid, s]
  gi = aset_geom[worldid, s]
  delta = aset_standoff[worldid, s]
  mu = aset_mu[worldid, s]

  off, nv = pair_vert_range(pt)
  xf0 = xfree[worldid, idx[off]]
  xf1 = xfree[worldid, idx[off + 1]] if nv > 1 else wp.vec3(0.0, 0.0, 0.0)
  xf2 = xfree[worldid, idx[off + 2]] if nv > 2 else wp.vec3(0.0, 0.0, 0.0)
  xf3 = xfree[worldid, idx[off + 3]] if nv > 3 else wp.vec3(0.0, 0.0, 0.0)

  dd, n, liv, lcw, lniv = eval_pair_gap(
    geom_type,
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_vert,
    mesh_polynum,
    mesh_polyadr,
    mesh_polynormal,
    mesh_polyvertadr,
    mesh_polyvertnum,
    mesh_polyvert,
    geom_xpos_in,
    geom_xmat_in,
    geom_corners,
    geom_edges,
    worldid,
    pt,
    idx,
    gi,
    xf0,
    xf1,
    xf2,
    xf3,
  )

  if pt <= int(FlexPairType.FLEX_EDGE_EDGE) and dd <= 0.0:
    pair_in_ws_out[worldid, s] = 0
    pair_lniv_out[worldid, s] = 0
    pair_rownnz_out[worldid, s] = 0
    aset_ld0_out[worldid, s] = float(1e30)
    return

  row_entries = int(0)
  for k in range(lniv):
    vk = liv[k]
    if dofadr[vk] >= 0:
      row_entries += 3
    elif pin_has_chain[vk]:
      row_entries += pin_num_chain[vk]
  pair_rownnz_out[worldid, s] = row_entries
  if row_entries == 0:
    pair_in_ws_out[worldid, s] = 0
    pair_lniv_out[worldid, s] = 0
    aset_ld0_out[worldid, s] = dd
    return

  iv0 = liv[0]
  xt0 = xtil[worldid, iv0]
  dx0 = x[worldid, iv0] - xf0

  xt1 = wp.vec3(0.0, 0.0, 0.0)
  dx1 = wp.vec3(0.0, 0.0, 0.0)
  if lniv >= 2:
    iv1 = liv[1]
    xt1 = xtil[worldid, iv1]
    dx1 = x[worldid, iv1] - xf1

  xt2 = wp.vec3(0.0, 0.0, 0.0)
  dx2 = wp.vec3(0.0, 0.0, 0.0)
  if lniv >= 3:
    iv2 = liv[2]
    xt2 = xtil[worldid, iv2]
    dx2 = x[worldid, iv2] - xf2

  xt3 = wp.vec3(0.0, 0.0, 0.0)
  dx3 = wp.vec3(0.0, 0.0, 0.0)
  if lniv >= 4:
    iv3 = liv[3]
    xt3 = xtil[worldid, iv3]
    dx3 = x[worldid, iv3] - xf3

  aset_ld0_out[worldid, s] = dd
  pair_ln_out[worldid, s] = n
  pair_lcw_out[worldid, s] = lcw
  pair_liv_out[worldid, s] = liv
  pair_lniv_out[worldid, s] = lniv

  lam_over_mu = aset_lam[worldid, s] / wp.max(mu, 1e-12)
  acc_til = lcw[0] * (xt0 - xf0)
  acc_x = lcw[0] * dx0
  if lniv >= 2:
    acc_til += lcw[1] * (xt1 - xf1)
    acc_x += lcw[1] * dx1
  if lniv >= 3:
    acc_til += lcw[2] * (xt2 - xf2)
    acc_x += lcw[2] * dx2
  if lniv >= 4:
    acc_til += lcw[3] * (xt3 - xf3)
    acc_x += lcw[3] * dx3
  r_til = (dd - delta) + wp.dot(n, acc_til) - lam_over_mu
  r_x = (dd - delta) + wp.dot(n, acc_x) - lam_over_mu
  pair_in_ws_out[worldid, s] = 1 if (r_til < 0.0 or r_x < 0.0) else 0


@wp.kernel
def ipc_check_omitted_pairs_kernel(
  # In:
  naset: wp.array[int],
  aset_standoff: wp.array2d[float],
  aset_mu: wp.array2d[float],
  aset_ld0: wp.array2d[float],
  aset_lam: wp.array2d[float],
  pair_ln: wp.array2d[wp.vec3],
  pair_liv: wp.array2d[wp.vec4i],
  pair_lcw: wp.array2d[wp.vec4],
  pair_lniv: wp.array2d[int],
  xprop: wp.array2d[wp.vec3],
  xfree: wp.array2d[wp.vec3],
  ws_done: wp.array[int],
  # Out:
  pair_in_ws_out: wp.array2d[int],
  ws_nadd_out: wp.array[int],
):
  """Checks omitted active-set pairs for working set expansion (engine_ipc.c:1079-1090)."""
  worldid, s = wp.tid()
  if ws_done[worldid] != 0 or s >= wp.min(naset[worldid], aset_ld0.shape[1]):
    return
  if pair_in_ws_out[worldid, s] != 0:
    return
  lniv = pair_lniv[worldid, s]
  if lniv == 0:
    return
  craw = pair_linear_gap(
    aset_ld0[worldid, s],
    aset_standoff[worldid, s],
    pair_ln[worldid, s],
    pair_liv[worldid, s],
    pair_lcw[worldid, s],
    lniv,
    xprop,
    xfree,
    worldid,
  )
  r = craw - aset_lam[worldid, s] / wp.max(aset_mu[worldid, s], 1e-12)
  if r < -IPC_WS_TOL:
    pair_in_ws_out[worldid, s] = 1
    wp.atomic_add(ws_nadd_out, worldid, 1)


@wp.kernel
def _ipc_init_ws_kernel(
  # Model:
  is_sparse: bool,
  # Data in:
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  # In:
  saved_nefc: wp.array[int],
  world_done_in: wp.array[int],
  # Data out:
  qacc_warmstart_out: wp.array2d[float],
  # Out:
  ws_entry_out: wp.array2d[float],
  ws_done_out: wp.array[int],
  active_count_out: wp.array[int],
  active_nnz_out: wp.array[int],
  ws_cond_out: wp.array[int],
  ws_iter_out: wp.array[int],
):
  worldid, dofid = wp.tid()
  ws_entry_out[worldid, dofid] = qacc_warmstart_out[worldid, dofid]
  if dofid == 0:
    ws_done_out[worldid] = world_done_in[worldid]
    active_count_out[worldid] = 0
    if is_sparse:
      snefc = wp.min(saved_nefc[worldid], efc_J_rowadr_in.shape[1])
      if snefc > 0:
        active_nnz_out[worldid] = efc_J_rowadr_in[worldid, snefc - 1] + efc_J_rownnz_in[worldid, snefc - 1]
      else:
        active_nnz_out[worldid] = 0
    if worldid == 0:
      ws_cond_out[0] = 1
      ws_iter_out[0] = 0


@wp.kernel
def _ipc_warmstart_da_kernel(
  # Data in:
  qacc_warmstart_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  # In:
  ws_done: wp.array[int],
  # Out:
  da_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if ws_done[worldid] != 0:
    return
  da_out[worldid, dofid] = qacc_warmstart_in[worldid, dofid] - qacc_smooth_in[worldid, dofid]


@wp.kernel
def _ipc_warmstart_select_kernel(
  # Data in:
  qacc_smooth_in: wp.array2d[float],
  # In:
  ws_done: wp.array[int],
  cost_warmstart: wp.array[float],
  cost_smooth: wp.array[float],
  # Data out:
  qacc_warmstart_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if ws_done[worldid] != 0:
    return
  if cost_warmstart[worldid] > cost_smooth[worldid]:
    qacc_warmstart_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]


@cache_kernel
def _ipc_publish_assign_rows_kernel(warn_overflow: int):
  @wp.kernel(module="unique")
  def kernel(
    # Model:
    is_sparse: bool,
    # Data in:
    njmax_in: int,
    njmax_nnz_in: int,
    # In:
    naset: wp.array[int],
    ws_done: wp.array[int],
    pair_rownnz: wp.array2d[int],
    saved_nefc: wp.array[int],
    # Data out:
    nefc_out: wp.array[int],
    efc_J_rowadr_out: wp.array2d[int],
    overflow_out: wp.array[int],
    # Out:
    active_count_out: wp.array[int],
    active_nnz_out: wp.array[int],
    pair_in_ws_out: wp.array2d[int],
  ):
    worldid = wp.tid()
    if ws_done[worldid] != 0:
      return
    n = wp.min(naset[worldid], pair_in_ws_out.shape[1])
    snefc = wp.min(saved_nefc[worldid], njmax_in)
    act_cnt = active_count_out[worldid]
    act_nnz = active_nnz_out[worldid]
    overflow_nefc = bool(False)
    overflow_nnz = bool(False)

    for s in range(n):
      if pair_in_ws_out[worldid, s] != 1:
        continue
      row_entries = pair_rownnz[worldid, s]
      if row_entries == 0:
        pair_in_ws_out[worldid, s] = 2
        continue

      if is_sparse:
        if snefc + act_cnt >= njmax_in:
          overflow_nefc = True
          pair_in_ws_out[worldid, s] = 2
          continue
        if act_nnz + row_entries > njmax_nnz_in:
          overflow_nnz = True
          pair_in_ws_out[worldid, s] = 2
          continue
        r = snefc + act_cnt
        efc_J_rowadr_out[worldid, r] = act_nnz
        act_nnz += row_entries
        act_cnt += 1
        pair_in_ws_out[worldid, s] = -(r + 1)
      else:
        if snefc + act_cnt >= njmax_in:
          overflow_nefc = True
          pair_in_ws_out[worldid, s] = 2
          continue
        r = snefc + act_cnt
        act_cnt += 1
        pair_in_ws_out[worldid, s] = -(r + 1)

    if overflow_nefc:
      if wp.static(warn_overflow):
        wp.printf("Error: scalar constraints larger than maximum size (%d)\n", njmax_in)
      wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.NEFC))
    if overflow_nnz:
      if wp.static(warn_overflow):
        wp.printf("Error: constraint Jacobian NNZ larger than maximum size (%d)\n", njmax_nnz_in)
      wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.NJMAX_NNZ))

    active_count_out[worldid] = act_cnt
    active_nnz_out[worldid] = act_nnz
    nefc_out[worldid] = wp.min(snefc + act_cnt, njmax_in)

  return kernel


@wp.kernel
def _ipc_zero_dense_efc_rows_kernel(
  # In:
  naset: wp.array[int],
  ws_done: wp.array[int],
  pair_in_ws: wp.array2d[int],
  # Data out:
  efc_J_out: wp.array3d[float],
):
  worldid, s, c = wp.tid()
  if ws_done[worldid] != 0 or s >= wp.min(naset[worldid], pair_in_ws.shape[1]):
    return
  pws = pair_in_ws[worldid, s]
  if pws >= 0:
    return
  r = -pws - 1
  efc_J_out[worldid, r, c] = 0.0


@wp.func
def _init_published_efc_row(
  # In:
  worldid: int,
  s: int,
  r: int,
  timestep: wp.array[float],
  aset_mu: wp.array2d[float],
  aset_ld0: wp.array2d[float],
  aset_standoff: wp.array2d[float],
  aset_lam: wp.array2d[float],
  aset_cnt: wp.array2d[int],
  # Data out:
  efc_type_out: wp.array2d[int],
  efc_id_out: wp.array2d[int],
  efc_D_out: wp.array2d[float],
  efc_frictionloss_out: wp.array2d[float],
  efc_force_out: wp.array2d[float],
  efc_state_out: wp.array2d[int],
):
  """Initializes published EFC row metadata and returns (h, h2, refc)."""
  h = timestep[worldid % timestep.shape[0]]
  h2 = h * h
  mu = aset_mu[worldid, s]
  dd = aset_ld0[worldid, s]
  delta = aset_standoff[worldid, s]
  lam_over_mu = aset_lam[worldid, s] / wp.max(mu, 1e-12)

  efc_type_out[worldid, r] = int(ConstraintType.CONTACT_FRICTIONLESS)
  efc_id_out[worldid, r] = -1
  efc_D_out[worldid, r] = _ipc_penalty_scale(mu, aset_cnt[worldid, s]) / h2
  efc_frictionloss_out[worldid, r] = 0.0
  efc_force_out[worldid, r] = 0.0
  efc_state_out[worldid, r] = int(ConstraintState.SATISFIED)

  refc = -(dd - delta) + lam_over_mu
  return h, h2, refc


@wp.kernel
def _ipc_publish_dense_pairs_kernel(
  # Data in:
  qvel_in: wp.array2d[float],
  xmat_in: wp.array2d[wp.mat33],
  # In:
  naset: wp.array[int],
  ws_done: wp.array[int],
  aset_standoff: wp.array2d[float],
  aset_mu: wp.array2d[float],
  aset_ld0: wp.array2d[float],
  aset_lam: wp.array2d[float],
  aset_cnt: wp.array2d[int],
  pair_ln: wp.array2d[wp.vec3],
  pair_lcw: wp.array2d[wp.vec4],
  pair_liv: wp.array2d[wp.vec4i],
  pair_lniv: wp.array2d[int],
  xold: wp.array2d[wp.vec3],
  xfree: wp.array2d[wp.vec3],
  pvel: wp.array2d[wp.vec3],
  dofadr: wp.array[int],
  pbody: wp.array[int],
  timestep: wp.array[float],
  pin_has_chain: wp.array[bool],
  pin_num_chain: wp.array[int],
  pin_chain_dof: wp.array2d[int],
  pin_chain_axis: wp.array3d[wp.vec3],
  # Data out:
  efc_type_out: wp.array2d[int],
  efc_id_out: wp.array2d[int],
  efc_J_out: wp.array3d[float],
  efc_D_out: wp.array2d[float],
  efc_aref_out: wp.array2d[float],
  efc_frictionloss_out: wp.array2d[float],
  efc_force_out: wp.array2d[float],
  efc_state_out: wp.array2d[int],
  # Out:
  pair_in_ws_out: wp.array2d[int],
):
  worldid, s = wp.tid()
  if ws_done[worldid] != 0 or s >= wp.min(naset[worldid], pair_in_ws_out.shape[1]):
    return
  pws = pair_in_ws_out[worldid, s]
  if pws >= 0:
    return
  r = -pws - 1
  pair_in_ws_out[worldid, s] = 2

  h, h2, refc = _init_published_efc_row(
    worldid,
    s,
    r,
    timestep,
    aset_mu,
    aset_ld0,
    aset_standoff,
    aset_lam,
    aset_cnt,
    efc_type_out,
    efc_id_out,
    efc_D_out,
    efc_frictionloss_out,
    efc_force_out,
    efc_state_out,
  )

  lniv = pair_lniv[worldid, s]
  ln = pair_ln[worldid, s]
  liv = pair_liv[worldid, s]
  lcw = pair_lcw[worldid, s]

  for k in range(lniv):
    vk = liv[k]
    cwk = lcw[k]
    cwk_h2 = cwk * h2
    xo_minus_xf = xold[worldid, vk] - xfree[worldid, vk]
    da = dofadr[vk]
    if da >= 0:
      R = xmat_in[worldid, pbody[vk]]
      qv = wp.vec3(qvel_in[worldid, da], qvel_in[worldid, da + 1], qvel_in[worldid, da + 2])
      refc -= cwk * wp.dot(ln, xo_minus_xf + (R * qv) * h)
      rtn = wp.transpose(R) * ln
      efc_J_out[worldid, r, da] += cwk_h2 * rtn[0]
      efc_J_out[worldid, r, da + 1] += cwk_h2 * rtn[1]
      efc_J_out[worldid, r, da + 2] += cwk_h2 * rtn[2]
    elif pin_has_chain[vk]:
      refc -= cwk * wp.dot(ln, xo_minus_xf + pvel[worldid, vk] * h)
      nchain = pin_num_chain[vk]
      for j in range(nchain):
        jdof = pin_chain_dof[vk, j]
        ax = pin_chain_axis[worldid, vk, j]
        efc_J_out[worldid, r, jdof] += cwk_h2 * wp.dot(ln, ax)

  efc_aref_out[worldid, r] = refc


@wp.kernel
def _ipc_publish_sparse_pairs_kernel(
  # Data in:
  qvel_in: wp.array2d[float],
  xmat_in: wp.array2d[wp.mat33],
  efc_J_rowadr_in: wp.array2d[int],
  # In:
  naset: wp.array[int],
  ws_done: wp.array[int],
  aset_standoff: wp.array2d[float],
  aset_mu: wp.array2d[float],
  aset_ld0: wp.array2d[float],
  aset_lam: wp.array2d[float],
  aset_cnt: wp.array2d[int],
  pair_ln: wp.array2d[wp.vec3],
  pair_lcw: wp.array2d[wp.vec4],
  pair_liv: wp.array2d[wp.vec4i],
  pair_lniv: wp.array2d[int],
  xold: wp.array2d[wp.vec3],
  xfree: wp.array2d[wp.vec3],
  pvel: wp.array2d[wp.vec3],
  dofadr: wp.array[int],
  pbody: wp.array[int],
  timestep: wp.array[float],
  pin_has_chain: wp.array[bool],
  pin_num_chain: wp.array[int],
  pin_chain_dof: wp.array2d[int],
  pin_chain_axis: wp.array3d[wp.vec3],
  # Data out:
  efc_type_out: wp.array2d[int],
  efc_id_out: wp.array2d[int],
  efc_J_rownnz_out: wp.array2d[int],
  efc_J_colind_out: wp.array3d[int],
  efc_J_out: wp.array3d[float],
  efc_D_out: wp.array2d[float],
  efc_aref_out: wp.array2d[float],
  efc_frictionloss_out: wp.array2d[float],
  efc_force_out: wp.array2d[float],
  efc_state_out: wp.array2d[int],
  # Out:
  pair_in_ws_out: wp.array2d[int],
):
  worldid, s = wp.tid()
  if ws_done[worldid] != 0 or s >= wp.min(naset[worldid], pair_in_ws_out.shape[1]):
    return
  pws = pair_in_ws_out[worldid, s]
  if pws >= 0:
    return
  r = -pws - 1
  pair_in_ws_out[worldid, s] = 2

  liv = pair_liv[worldid, s]
  lniv = pair_lniv[worldid, s]
  rowadr = efc_J_rowadr_in[worldid, r]

  h, h2, refc = _init_published_efc_row(
    worldid,
    s,
    r,
    timestep,
    aset_mu,
    aset_ld0,
    aset_standoff,
    aset_lam,
    aset_cnt,
    efc_type_out,
    efc_id_out,
    efc_D_out,
    efc_frictionloss_out,
    efc_force_out,
    efc_state_out,
  )

  ln = pair_ln[worldid, s]
  lcw = pair_lcw[worldid, s]

  cur = int(0)
  for k in range(lniv):
    vk = liv[k]
    cwk = lcw[k]
    cwk_h2 = cwk * h2
    xo_minus_xf = xold[worldid, vk] - xfree[worldid, vk]
    da = dofadr[vk]
    if da >= 0:
      R = xmat_in[worldid, pbody[vk]]
      qv = wp.vec3(qvel_in[worldid, da], qvel_in[worldid, da + 1], qvel_in[worldid, da + 2])
      refc -= cwk * wp.dot(ln, xo_minus_xf + (R * qv) * h)
      rtn = wp.transpose(R) * ln
      efc_J_colind_out[worldid, 0, rowadr + cur] = da
      efc_J_out[worldid, 0, rowadr + cur] = cwk_h2 * rtn[0]
      cur += 1

      efc_J_colind_out[worldid, 0, rowadr + cur] = da + 1
      efc_J_out[worldid, 0, rowadr + cur] = cwk_h2 * rtn[1]
      cur += 1

      efc_J_colind_out[worldid, 0, rowadr + cur] = da + 2
      efc_J_out[worldid, 0, rowadr + cur] = cwk_h2 * rtn[2]
      cur += 1
    elif pin_has_chain[vk]:
      refc -= cwk * wp.dot(ln, xo_minus_xf + pvel[worldid, vk] * h)
      nchain = pin_num_chain[vk]
      for j in range(nchain):
        jdof = pin_chain_dof[vk, j]
        ax = pin_chain_axis[worldid, vk, j]
        efc_J_colind_out[worldid, 0, rowadr + cur] = jdof
        efc_J_out[worldid, 0, rowadr + cur] = cwk_h2 * wp.dot(ln, ax)
        cur += 1

  efc_aref_out[worldid, r] = refc

  # Insertion sort by column index
  for i in range(1, cur):
    ci = efc_J_colind_out[worldid, 0, rowadr + i]
    ji = efc_J_out[worldid, 0, rowadr + i]
    j = i - 1
    while j >= 0 and efc_J_colind_out[worldid, 0, rowadr + j] > ci:
      efc_J_colind_out[worldid, 0, rowadr + j + 1] = efc_J_colind_out[worldid, 0, rowadr + j]
      efc_J_out[worldid, 0, rowadr + j + 1] = efc_J_out[worldid, 0, rowadr + j]
      j -= 1
    efc_J_colind_out[worldid, 0, rowadr + j + 1] = ci
    efc_J_out[worldid, 0, rowadr + j + 1] = ji

  # Merge duplicate column entries
  w = int(0)
  for i in range(cur):
    if w > 0 and efc_J_colind_out[worldid, 0, rowadr + w - 1] == efc_J_colind_out[worldid, 0, rowadr + i]:
      efc_J_out[worldid, 0, rowadr + w - 1] += efc_J_out[worldid, 0, rowadr + i]
    else:
      if w != i:
        efc_J_colind_out[worldid, 0, rowadr + w] = efc_J_colind_out[worldid, 0, rowadr + i]
        efc_J_out[worldid, 0, rowadr + w] = efc_J_out[worldid, 0, rowadr + i]
      w += 1

  efc_J_rownnz_out[worldid, r] = w


def _efc_publish(m: Model, d: Data, ws: IpcWorkspace, max_aset: int):
  """Assigns and populates EFC rows for newly added working-set pairs (ipc_efcPublish)."""
  nworld = d.nworld
  wp.launch(
    _ipc_publish_assign_rows_kernel(int(m.opt.warn_overflow)),
    dim=nworld,
    inputs=[
      m.is_sparse,
      d.njmax,
      d.njmax_nnz,
      ws.naset,
      ws.ws_done,
      ws.pair_rownnz,
      ws.saved_nefc,
    ],
    outputs=[
      d.nefc,
      d.efc.J_rowadr,
      d.overflow,
      ws.active_count,
      ws.active_nnz,
      ws.pair_in_ws,
    ],
  )
  if m.is_sparse:
    wp.launch(
      _ipc_publish_sparse_pairs_kernel,
      dim=(nworld, max_aset),
      inputs=[
        d.qvel,
        d.xmat,
        d.efc.J_rowadr,
        ws.naset,
        ws.ws_done,
        ws.aset_standoff,
        ws.aset_mu,
        ws.aset_ld0,
        ws.aset_lam,
        ws.aset_cnt,
        ws.pair_ln,
        ws.pair_lcw,
        ws.pair_liv,
        ws.pair_lniv,
        ws.xold,
        ws.xfree,
        ws.pvel,
        ws.dofadr,
        ws.pbody,
        m.opt.timestep,
        ws.pin_has_chain,
        ws.pin_num_chain,
        ws.pin_chain_dof,
        ws.pin_chain_axis,
      ],
      outputs=[
        d.efc.type,
        d.efc.id,
        d.efc.J_rownnz,
        d.efc.J_colind,
        d.efc.J,
        d.efc.D,
        d.efc.aref,
        d.efc.frictionloss,
        d.efc.force,
        d.efc.state,
        ws.pair_in_ws,
      ],
    )
  else:
    wp.launch(
      _ipc_zero_dense_efc_rows_kernel,
      dim=(nworld, max_aset, d.efc.J.shape[2]),
      inputs=[
        ws.naset,
        ws.ws_done,
        ws.pair_in_ws,
      ],
      outputs=[d.efc.J],
    )
    wp.launch(
      _ipc_publish_dense_pairs_kernel,
      dim=(nworld, max_aset),
      inputs=[
        d.qvel,
        d.xmat,
        ws.naset,
        ws.ws_done,
        ws.aset_standoff,
        ws.aset_mu,
        ws.aset_ld0,
        ws.aset_lam,
        ws.aset_cnt,
        ws.pair_ln,
        ws.pair_lcw,
        ws.pair_liv,
        ws.pair_lniv,
        ws.xold,
        ws.xfree,
        ws.pvel,
        ws.dofadr,
        ws.pbody,
        m.opt.timestep,
        ws.pin_has_chain,
        ws.pin_num_chain,
        ws.pin_chain_dof,
        ws.pin_chain_axis,
      ],
      outputs=[
        d.efc.type,
        d.efc.id,
        d.efc.J,
        d.efc.D,
        d.efc.aref,
        d.efc.frictionloss,
        d.efc.force,
        d.efc.state,
        ws.pair_in_ws,
      ],
    )


@wp.kernel
def _ipc_finish_ws_and_init_ls_kernel(
  # In:
  saved_nefc: wp.array[int],
  world_done_in: wp.array[int],
  ws_entry_in: wp.array2d[float],
  # Data out:
  nefc_out: wp.array[int],
  qacc_warmstart_out: wp.array2d[float],
  # Out:
  ls_done_out: wp.array[int],
  ls_cond_out: wp.array[int],
  ls_iter_out: wp.array[int],
):
  worldid, dofid = wp.tid()
  qacc_warmstart_out[worldid, dofid] = ws_entry_in[worldid, dofid]
  if dofid == 0:
    nefc_out[worldid] = saved_nefc[worldid]
    ls_done_out[worldid] = world_done_in[worldid]
    if worldid == 0:
      ls_cond_out[0] = 1
      ls_iter_out[0] = 0


def _efc_restore(m: Model, d: Data, ws: IpcWorkspace):
  """Restores d.nefc and d.qacc_warmstart after working-set solve (ipc_efcRestore)."""
  wp.launch(
    _ipc_finish_ws_and_init_ls_kernel,
    dim=(d.nworld, m.nv),
    inputs=[
      ws.saved_nefc,
      ws.world_done,
      ws.ws_entry,
    ],
    outputs=[
      d.nefc,
      d.qacc_warmstart,
      ws.ls_done,
      ws.ls_cond,
      ws.ls_iter,
    ],
  )


@wp.kernel
def _ipc_post_solve_kernel(
  # Data in:
  solver_niter_in: wp.array[int],
  qvel_in: wp.array2d[float],
  qacc_in: wp.array2d[float],
  # In:
  own: wp.array[bool],
  timestep: wp.array[float],
  w: wp.array2d[float],
  ws_done: wp.array[int],
  # Data out:
  qacc_warmstart_out: wp.array2d[float],
  # Out:
  wprop_out: wp.array2d[float],
  ddw_out: wp.array2d[float],
  total_solver_niter_out: wp.array[int],
):
  worldid, dofid = wp.tid()
  if ws_done[worldid] != 0:
    return
  if dofid == 0:
    total_solver_niter_out[worldid] += solver_niter_in[worldid]
  qacc_warmstart_out[worldid, dofid] = qacc_in[worldid, dofid]
  h = timestep[worldid % timestep.shape[0]]
  if own[dofid]:
    wp_val = qvel_in[worldid, dofid] + h * qacc_in[worldid, dofid]
    wprop_out[worldid, dofid] = wp_val
    ddw_out[worldid, dofid] = wp_val - w[worldid, dofid]
  else:
    wprop_out[worldid, dofid] = 0.0
    ddw_out[worldid, dofid] = 0.0


@wp.kernel
def _ipc_update_and_check_ws_done_kernel(
  # Data in:
  nworld_in: int,
  # In:
  max_rounds: int,
  ws_nadd_in: wp.array[int],
  # Out:
  ws_done_out: wp.array[int],
  ws_iter_out: wp.array[int],
  ws_cond_out: wp.array[int],
):
  worldid = wp.tid()
  if ws_nadd_in[worldid] == 0:
    ws_done_out[worldid] = 1
  if worldid == 0:
    it = ws_iter_out[0] + 1
    ws_iter_out[0] = it
    if it >= max_rounds:
      ws_cond_out[0] = 0
    else:
      has_added = int(0)
      for w in range(nworld_in):
        if ws_nadd_in[w] > 0:
          has_added = int(1)
          break
      ws_cond_out[0] = 1 if has_added != 0 else 0


@wp.kernel
def _ipc_update_lambda_kernel(
  # In:
  naset: wp.array[int],
  aset_type: wp.array2d[int],
  aset_idx: wp.array2d[wp.vec4i],
  aset_standoff: wp.array2d[float],
  aset_mu: wp.array2d[float],
  aset_ld0: wp.array2d[float],
  pair_ln: wp.array2d[wp.vec3],
  pair_liv: wp.array2d[wp.vec4i],
  pair_lcw: wp.array2d[wp.vec4],
  pair_lniv: wp.array2d[int],
  x: wp.array2d[wp.vec3],
  xfree: wp.array2d[wp.vec3],
  world_done: wp.array[int],
  # Data out:
  flexvert_lambda_out: wp.array2d[float],
  # Out:
  aset_lam_out: wp.array2d[float],
  aset_cnt_out: wp.array2d[int],
):
  """Updates AL multipliers and contact ages across active-set pairs (ipc_flexLamUpdate)."""
  worldid, s = wp.tid()
  if world_done[worldid] != 0 or s >= wp.min(naset[worldid], aset_type.shape[1]):
    return

  lam = aset_lam_out[worldid, s]
  lniv = pair_lniv[worldid, s]
  if lniv > 0:
    dd0 = aset_ld0[worldid, s]
    if dd0 < 1e20 or lam > 0.0:
      mu = aset_mu[worldid, s]
      craw = pair_linear_gap(
        dd0,
        aset_standoff[worldid, s],
        pair_ln[worldid, s],
        pair_liv[worldid, s],
        pair_lcw[worldid, s],
        lniv,
        x,
        xfree,
        worldid,
      )
      active = craw - lam / mu <= 0.0
      lam = lam - craw * mu if active else 0.0
      aset_lam_out[worldid, s] = lam
      aset_cnt_out[worldid, s] = _ipc_age_step(aset_cnt_out[worldid, s], active)

  if lam > 0.0:
    idx = aset_idx[worldid, s]
    off, nv = pair_vert_range(aset_type[worldid, s])
    for q in range(nv):
      vg = idx[off + q]
      wp.atomic_max(flexvert_lambda_out, worldid, vg, lam)


def _flex_lam_update(m: Model, d: Data, ws: IpcWorkspace, max_aset: int):
  """Updates AL multipliers and contact ages across pairs and vertices (ipc_flexLamUpdate)."""
  nworld = d.nworld
  nfv = m.nflexvert
  wp.launch(
    _ipc_clear_flexvert_lambda_kernel,
    dim=(nworld, nfv),
    inputs=[ws.world_done],
    outputs=[d.flexvert_lambda],
  )
  wp.launch(
    _ipc_update_lambda_kernel,
    dim=(nworld, max_aset),
    inputs=[
      ws.naset,
      ws.aset_type,
      ws.aset_idx,
      ws.aset_standoff,
      ws.aset_mu,
      ws.aset_ld0,
      ws.pair_ln,
      ws.pair_liv,
      ws.pair_lcw,
      ws.pair_lniv,
      ws.x,
      ws.xfree,
      ws.world_done,
    ],
    outputs=[d.flexvert_lambda, ws.aset_lam, ws.aset_cnt],
  )
  wp.launch(
    _ipc_flexvert_age_step_kernel,
    dim=(nworld, nfv),
    inputs=[
      m.flex_dim,
      m.flex_vertflexid,
      d.flexvert_lambda,
      ws.world_done,
    ],
    outputs=[d.flexvert_conage],
  )


@wp.kernel
def ipc_advance_wfree_kernel(
  # In:
  own: wp.array[bool],
  is_throttled: wp.array[bool],
  alpha_min: wp.array[float],
  w: wp.array2d[float],
  world_done: wp.array[int],
  # Out:
  wfree_out: wp.array2d[float],
):
  """Advances wfree along the accepted search direction w by the CCD step size alpha_min."""
  worldid, dofid = wp.tid()
  if world_done[worldid] != 0:
    return

  ac = alpha_min[worldid]
  if own[dofid]:
    if not is_throttled[dofid]:
      wfree_out[worldid, dofid] = w[worldid, dofid]
    elif ac > IPC_ALPHA_LB:
      wfree_out[worldid, dofid] += ac * (w[worldid, dofid] - wfree_out[worldid, dofid])


@wp.kernel
def ipc_advance_world_state_kernel(
  # In:
  alpha_min: wp.array[float],
  last_ls_alpha: wp.array[float],
  newton_converged: wp.array[int],
  # Out:
  beta_out: wp.array[float],
  world_done_out: wp.array[int],
  stall_out: wp.array[int],
):
  """Updates cumulative step fraction beta, stall counter, and world completion flag."""
  worldid = wp.tid()
  if world_done_out[worldid] != 0:
    return

  ac = alpha_min[worldid]
  beta_val = beta_out[worldid] + (1.0 - beta_out[worldid]) * ac
  beta_out[worldid] = beta_val
  if beta_val >= 1.0 - 1e-6 and (last_ls_alpha[worldid] >= 1.0 - 1e-9 or newton_converged[worldid] != 0):
    world_done_out[worldid] = 1
  elif ac > IPC_ALPHA_LB:
    stall_out[worldid] = 0
  else:
    st = stall_out[worldid] + 1
    stall_out[worldid] = st
    if st >= IPC_STALL_MAX:
      world_done_out[worldid] = 1


@wp.kernel
def _ipc_prepare_commit_kernel(
  # Data in:
  qvel_in: wp.array2d[float],
  qacc_smooth_in: wp.array2d[float],
  # In:
  own: wp.array[bool],
  timestep: wp.array[float],
  total_solver_niter: wp.array[int],
  # Data out:
  solver_niter_out: wp.array[int],
  qacc_out: wp.array2d[float],
  # Out:
  wfree_out: wp.array2d[float],
  qvel_old_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  h = timestep[worldid % timestep.shape[0]]
  if dofid == 0:
    solver_niter_out[worldid] = total_solver_niter[worldid]
  if own[dofid]:
    wf = wfree_out[worldid, dofid]
    qvel_old_out[worldid, dofid] = wf
    qacc_out[worldid, dofid] = (wf - qvel_in[worldid, dofid]) / h
  else:
    wfree_out[worldid, dofid] = 0.0
    qvel_old_out[worldid, dofid] = qvel_in[worldid, dofid]
    qacc_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]


@wp.kernel
def _ipc_check_outer_done_kernel(
  # Data in:
  nworld_in: int,
  # In:
  outer_cap: int,
  world_done_in: wp.array[int],
  # Out:
  outer_iter_out: wp.array[int],
  outer_cond_out: wp.array[int],
):
  tid = wp.tid()
  if tid == 0:
    it = outer_iter_out[0] + 1
    outer_iter_out[0] = it
    all_done = int(1)
    for w in range(nworld_in):
      if world_done_in[w] == 0:
        all_done = int(0)
        break
    outer_cond_out[0] = 0 if (all_done == 1 or it >= outer_cap) else 1


def _eval_efc_energy(
  m: Model,
  d: Data,
  nefc_in: wp.array[int],
  qacc_in: wp.array2d[float],
  skip_world: wp.array[int],
  out_energy: wp.array[float],
):
  """Accumulates scaled EFC primal cost over rows [0, nefc_in) into out_energy."""
  wp.launch_tiled(
    _ipc_eval_merit_native_efc_kernel(m.is_sparse),
    dim=d.nworld,
    inputs=[
      m.nv,
      m.opt.impratio_invsqrt,
      d.contact.friction,
      d.contact.dim,
      d.contact.efc_address,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.D,
      d.efc.aref,
      d.efc.frictionloss,
      d.nacon,
      nefc_in,
      m.opt.timestep,
      qacc_in,
      skip_world,
    ],
    outputs=[out_energy],
    block_dim=m.block_dim.ipc_merit,
  )


def _eval_gauss_and_efc_energy(
  m: Model,
  d: Data,
  ws: IpcWorkspace,
  nefc_in: wp.array[int],
  qacc_in: wp.array2d[float],
  skip_world: wp.array[int],
  out_energy: wp.array[float],
):
  """Evaluates M*da, Gauss energy, and scaled EFC primal energy over [0, nefc_in)."""
  derivative.eff_mul_m(m, d, ws.M_da, ws.da)
  wp.launch_tiled(
    _ipc_eval_gauss_energy_kernel,
    dim=d.nworld,
    inputs=[m.nv, m.opt.timestep, ws.da, ws.M_da, skip_world],
    outputs=[out_energy],
    block_dim=m.block_dim.ipc_merit,
  )
  _eval_efc_energy(m, d, nefc_in, qacc_in, skip_world, out_energy)


def _warmstart(m: Model, d: Data, ws: IpcWorkspace):
  """Selects between qacc_warmstart and qacc_smooth across published EFC rows."""
  nworld = d.nworld
  nv = m.nv
  wp.launch(
    _ipc_warmstart_da_kernel,
    dim=(nworld, nv),
    inputs=[d.qacc_warmstart, d.qacc_smooth, ws.ws_done],
    outputs=[ws.da],
  )
  _eval_gauss_and_efc_energy(m, d, ws, d.nefc, d.qacc_warmstart, ws.ws_done, ws.merit_energy)
  ws.merit_energy_trial.zero_()
  _eval_efc_energy(m, d, d.nefc, d.qacc_smooth, ws.ws_done, ws.merit_energy_trial)
  wp.launch(
    _ipc_warmstart_select_kernel,
    dim=(nworld, nv),
    inputs=[d.qacc_smooth, ws.ws_done, ws.merit_energy, ws.merit_energy_trial],
    outputs=[d.qacc_warmstart],
  )


@event_scope
def _eval_merit_energy(
  m: Model,
  d: Data,
  ws: IpcWorkspace,
  w_in: wp.array2d[float],
  x_in: wp.array2d[wp.vec3],
  out_energy: wp.array[float],
  skip_world: wp.array[int],
):
  """Evaluates total augmented-Lagrangian merit energy (Gauss + native EFC + contact)."""
  out_energy.zero_()
  qacc_eval = ws.qvel_old
  wp.launch(
    _ipc_tangent_da_kernel,
    dim=(d.nworld, m.nv),
    inputs=[
      d.qvel,
      d.qacc_smooth,
      ws.own,
      m.opt.timestep,
      w_in,
      skip_world,
    ],
    outputs=[ws.da, qacc_eval],
  )
  _eval_gauss_and_efc_energy(m, d, ws, ws.saved_nefc, qacc_eval, skip_world, out_energy)
  wp.launch_tiled(
    _ipc_eval_merit_contact_kernel,
    dim=d.nworld,
    inputs=[
      ws.naset,
      ws.aset_standoff,
      ws.aset_mu,
      ws.aset_ld0,
      ws.aset_lam,
      ws.aset_cnt,
      ws.pair_ln,
      ws.pair_liv,
      ws.pair_lcw,
      ws.pair_lniv,
      x_in,
      ws.xfree,
      skip_world,
    ],
    outputs=[out_energy],
    block_dim=m.block_dim.ipc_merit,
  )


@event_scope
def ipc(m: Model, d: Data, ws: Optional[IpcWorkspace] = None):
  """Executes the barrier-free augmented-Lagrangian IPC contact step across all worlds (mj_ipc)."""
  if not m.has_2d_flex or m.nflexvert == 0:
    forward.advance(m, d, d.qacc)
    return

  if ws is None:
    ws = get_ipc_workspace(m, d)
  nworld = d.nworld
  nv = m.nv
  nfv = m.nflexvert
  max_aset = 0 if bool(m.opt.disableflags & (DisableBit.CONTACT | DisableBit.CONSTRAINT)) else ws.max_aset

  ipc_update_geom_features(m, d, ws)

  if ws.solver_ctx is None:
    ws.solver_ctx = solver.create_solver_context(m, d)
  ctx = ws.solver_ctx
  solver.prepare_solver_context(m, d, ctx)

  # Initialize step
  wp.launch(
    _ipc_init_step_kernel,
    dim=(nworld, nv),
    inputs=[
      d.qvel,
      d.qacc_smooth,
      ws.own,
      m.opt.timestep,
    ],
    outputs=[
      ws.w,
      ws.wtil,
      ws.wfree,
      ws.ddw,
      ws.beta,
      ws.world_done,
      ws.stall,
      ws.total_solver_niter,
      ws.newton_converged,
      ws.last_ls_alpha,
      ws.outer_cond,
      ws.outer_iter,
      ws.init_ncand,
      ws.naset,
    ],
  )

  # Update pinned chain axes, carrier velocity, initial vertex positions, and xtil = points(wtil)
  wp.launch(
    _ipc_init_verts_step_kernel,
    dim=(nworld, nfv),
    inputs=[
      m.jnt_bodyid,
      m.jnt_axis,
      m.dof_jntid,
      d.qvel,
      d.xmat,
      d.cdof,
      d.flexvert_xpos,
      ws.fidx,
      ws.dofadr,
      ws.pbody,
      ws.pin_has_chain,
      ws.pin_num_chain,
      ws.pin_chain_dof,
      ws.wtil,
      m.opt.timestep,
    ],
    outputs=[
      ws.pin_chain_axis,
      ws.pvel,
      ws.xold,
      ws.xfree,
      ws.x,
      ws.xtil,
    ],
  )

  # Save existing nefc
  wp.copy(ws.saved_nefc, d.nefc)
  ws.qfrc_rows.zero_()

  # Step-start candidate query at xold over xold -> xtil and seed active set (engine_ipc.c:889-935)
  ipc_discover_candidates(m, d, ws, ws.xold, ws.xold, ws.xtil, 3.0 * IPC_GHAT, IPC_GHAT, IPC_GHAT)
  ipc_seed_active_set(m, d, ws)

  # When a world has zero initial candidates, initialize flex DOFs to wtil (free flight)
  wp.launch(
    _ipc_free_flight_init_wx_kernel,
    dim=(nworld, max(nv, nfv)),
    inputs=[nv, nfv, ws.isflexdof, ws.fidx, ws.init_ncand, ws.wtil, ws.xtil],
    outputs=[ws.w, ws.x],
  )

  # Outer loop
  use_graph_cond = bool(m.opt.graph_conditional and wp.get_device().is_cuda)
  outer_cap = 1024 if use_graph_cond else 32

  def _ipc_outer_iteration():
    # Linearize active set (aset) at xfree and build initial working set
    wp.launch(
      _ipc_eval_pairs_kernel,
      dim=(nworld, max_aset),
      inputs=[
        m.geom_type,
        m.geom_dataid,
        m.geom_size,
        m.mesh_vertadr,
        m.mesh_vert,
        m.mesh_polynum,
        m.mesh_polyadr,
        m.mesh_polynormal,
        m.mesh_polyvertadr,
        m.mesh_polyvertnum,
        m.mesh_polyvert,
        d.geom_xpos,
        d.geom_xmat,
        ws.naset,
        ws.aset_type,
        ws.aset_idx,
        ws.aset_geom,
        ws.aset_standoff,
        ws.aset_mu,
        ws.aset_lam,
        ws.geom_corners,
        ws.geom_edges,
        ws.world_done,
        ws.x,
        ws.xtil,
        ws.xfree,
        ws.dofadr,
        ws.pin_has_chain,
        ws.pin_num_chain,
      ],
      outputs=[
        ws.aset_ld0,
        ws.pair_ln,
        ws.pair_lcw,
        ws.pair_liv,
        ws.pair_lniv,
        ws.pair_rownnz,
        ws.pair_in_ws,
      ],
    )

    # Initialize working-set rounds, warmstart, and incremental EFC publishing state
    wp.launch(
      _ipc_init_ws_kernel,
      dim=(nworld, nv),
      inputs=[
        m.is_sparse,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        ws.saved_nefc,
        ws.world_done,
      ],
      outputs=[
        d.qacc_warmstart,
        ws.ws_entry,
        ws.ws_done,
        ws.active_count,
        ws.active_nnz,
        ws.ws_cond,
        ws.ws_iter,
      ],
    )

    def _ipc_ws_round():
      _efc_publish(m, d, ws, max_aset)

      if not (m.opt.disableflags & DisableBit.WARMSTART):
        _warmstart(m, d, ws)

      # Solve coupled subproblem on GPU across all worlds
      solver.solve(m, d, ctx=ctx, skip=ws.ws_done)
      _efc_row_force(m, d, ws)

      # Post solve: compute proposal velocity in ws.wn, ddw, and update qacc_warmstart
      wp.launch(
        _ipc_post_solve_kernel,
        dim=(nworld, nv),
        inputs=[
          d.solver_niter,
          d.qvel,
          d.qacc,
          ws.own,
          m.opt.timestep,
          ws.w,
          ws.ws_done,
        ],
        outputs=[d.qacc_warmstart, ws.wn, ws.ddw, ws.total_solver_niter],
      )

      _points(m, d, ws, ws.wn, ws.xn, ws.ws_done)
      ws.ws_nadd.zero_()
      wp.launch(
        ipc_check_omitted_pairs_kernel,
        dim=(nworld, max_aset),
        inputs=[
          ws.naset,
          ws.aset_standoff,
          ws.aset_mu,
          ws.aset_ld0,
          ws.aset_lam,
          ws.pair_ln,
          ws.pair_liv,
          ws.pair_lcw,
          ws.pair_lniv,
          ws.xn,
          ws.xfree,
          ws.ws_done,
        ],
        outputs=[ws.pair_in_ws, ws.ws_nadd],
      )
      wp.launch(
        _ipc_update_and_check_ws_done_kernel,
        dim=nworld,
        inputs=[
          nworld,
          8,
          ws.ws_nadd,
        ],
        outputs=[
          ws.ws_done,
          ws.ws_iter,
          ws.ws_cond,
        ],
      )

    if use_graph_cond:
      wp.capture_while(ws.ws_cond, while_body=_ipc_ws_round)
    else:
      for _ in range(8):
        _ipc_ws_round()

    # Restore qacc_warmstart and nefc, and initialize line-search state
    _efc_restore(m, d, ws)

    # Check convergence of the solved direction
    wp.launch_tiled(
      _ipc_check_convergence_kernel,
      dim=nworld,
      inputs=[
        nv,
        ws.own,
        ws.ddw,
      ],
      outputs=[ws.newton_converged],
      block_dim=m.block_dim.ipc_merit,
    )

    # Base merit energy at current w
    _eval_merit_energy(m, d, ws, ws.w, ws.x, ws.merit_energy, ws.world_done)

    def _ipc_ls_round():
      wp.launch(
        _ipc_trial_w_kernel,
        dim=(nworld, nv),
        inputs=[ws.own, ws.world_done, ws.ls_done, ws.ls_iter, ws.w, ws.ddw],
        outputs=[ws.wn],
      )
      _points(m, d, ws, ws.wn, ws.xn, ws.ls_done)
      _eval_merit_energy(m, d, ws, ws.wn, ws.xn, ws.merit_energy_trial, ws.ls_done)

      wp.launch(
        _ipc_linesearch_eval_trial_kernel,
        dim=nworld,
        inputs=[
          ws.world_done,
          ws.ls_done,
          ws.ls_iter,
          ws.beta,
          ws.merit_energy,
          ws.merit_energy_trial,
          ws.newton_converged,
        ],
        outputs=[
          ws.ls_done,
          ws.last_ls_alpha,
          ws.ls_accept,
        ],
      )
      wp.launch(
        _ipc_linesearch_commit_wx_kernel,
        dim=(nworld, max(nv, nfv)),
        inputs=[
          nv,
          nworld,
          nfv,
          ws.world_done,
          ws.ls_done,
          ws.ls_accept,
          ws.wn,
          ws.xn,
        ],
        outputs=[
          ws.w,
          ws.x,
          ws.ls_iter,
          ws.ls_cond,
        ],
      )

    if use_graph_cond:
      wp.capture_while(ws.ls_cond, while_body=_ipc_ls_round)
    else:
      for _ls in range(8):
        _ipc_ls_round()

    # Update persistent pair multipliers across active set and sink into flexvert_lambda
    _flex_lam_update(m, d, ws, max_aset)

    # Post-solve candidate re-query over xfree -> x (engine_ipc.c:1150-1155)
    ipc_discover_candidates(m, d, ws, ws.xfree, ws.xfree, ws.x, 3.0 * IPC_GHAT, 3.0 * IPC_GHAT, IPC_GHAT)

    # Evaluate CCD time of impact and reduce alpha_min per world (mjc_advance)
    ipc_advance(m, d, ws)

    # Earliest-ToI vertex filter and active-set (aset) merge (ipc_mergeActiveSet)
    ipc_merge_active_set(m, d, ws)

    # Advance wfree, beta, and check completion
    wp.launch(
      ipc_advance_wfree_kernel,
      dim=(nworld, nv),
      inputs=[
        ws.own,
        ws.is_throttled,
        ws.alpha_min,
        ws.w,
        ws.world_done,
      ],
      outputs=[ws.wfree],
    )
    wp.launch(
      ipc_advance_world_state_kernel,
      dim=nworld,
      inputs=[
        ws.alpha_min,
        ws.last_ls_alpha,
        ws.newton_converged,
      ],
      outputs=[
        ws.beta,
        ws.world_done,
        ws.stall,
      ],
    )

    # Update xfree = points(wfree)
    _points(m, d, ws, ws.wfree, ws.xfree, ws.world_done)

    wp.launch(
      _ipc_check_outer_done_kernel,
      dim=1,
      inputs=[
        nworld,
        outer_cap,
        ws.world_done,
      ],
      outputs=[
        ws.outer_iter,
        ws.outer_cond,
      ],
    )

  if use_graph_cond:
    wp.capture_while(ws.outer_cond, while_body=_ipc_outer_iteration)
  else:
    for _outer in range(outer_cap):
      _ipc_outer_iteration()

  # Prepare committed accelerations and velocities
  wp.launch(
    _ipc_prepare_commit_kernel,
    dim=(nworld, nv),
    inputs=[
      d.qvel,
      d.qacc_smooth,
      ws.own,
      m.opt.timestep,
      ws.total_solver_niter,
    ],
    outputs=[
      d.solver_niter,
      d.qacc,
      ws.wfree,
      ws.qvel_old,
    ],
  )

  # Evaluate native constraint forces at committed qacc and add pairs' force
  solver.update_constraint(m, d, ctx)
  wp.launch(
    _ipc_add_qfrc_rows_kernel,
    dim=(nworld, nv),
    inputs=[ws.qfrc_rows],
    outputs=[d.qfrc_constraint],
  )

  # Acceleration sensors
  if m.opt.run_rne_postconstraint or (not (m.opt.disableflags & DisableBit.SENSOR) and m.sensor_rne_postconstraint):
    smooth.rne_postconstraint(m, d)
  sensor.sensor_acc(m, d, skip_rne_postconstraint=True)

  # Advance state and restore IPC velocities and warmstart
  forward.advance(m, d, d.qacc, ws.wfree)
  wp.copy(d.qvel, ws.qvel_old)
  wp.copy(d.qacc_warmstart, ws.ws_entry)
