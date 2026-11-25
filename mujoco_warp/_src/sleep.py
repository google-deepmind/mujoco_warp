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

from typing import Optional

import warp as wp

from mujoco_warp._src import fusion
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.types import ContactType
from mujoco_warp._src.types import DeterminismType
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import EqType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import ObjType
from mujoco_warp._src.types import SleepPolicy
from mujoco_warp._src.types import SleepState
from mujoco_warp._src.types import WrapType
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# tree_asleep value for fully awake tree
K_AWAKE_VAL = -(1 + types.MJ_MINAWAKE)


@wp.func
def _sleep_cycle(tree_asleep: wp.array2d[int], ntree: int, worldid: int, treeid: int) -> int:  # kernel_analyzer: ignore
  if treeid < 0 or treeid >= ntree:
    return -1

  smallest = int(treeid)
  current = int(treeid)
  count = int(0)

  for step in range(ntree + 1):  # safe upper bound
    next_tree = tree_asleep[worldid, current]
    if next_tree < 0 or next_tree >= ntree:
      return -1

    if next_tree < smallest:
      smallest = next_tree

    current = next_tree
    count += 1
    if current == treeid:
      break

  return smallest


@wp.func
def _tendon_limit_active(
  tendon_limited: wp.array[int],  # kernel_analyzer: ignore
  tendon_range: wp.array2d[wp.vec2],  # kernel_analyzer: ignore
  tendon_margin: wp.array2d[float],  # kernel_analyzer: ignore
  ten_length: wp.array2d[float],  # kernel_analyzer: ignore
  worldid: int,  # kernel_analyzer: ignore
  tenid: int,
) -> bool:
  if tendon_limited[tenid] == 0:
    return False

  length = ten_length[worldid, tenid]
  margin = tendon_margin[worldid % tendon_margin.shape[0], tenid]
  r = tendon_range[worldid % tendon_range.shape[0], tenid]

  # low limit
  dist_low = length - r[0]
  if dist_low < margin:
    return True

  # high limit
  dist_high = r[1] - length
  if dist_high < margin:
    return True

  return False


@cache_kernel
def _zero_sleep_counters(has_woke: bool = False):
  @fusion.stage("nworld")
  def kernel(
    worldid: int,
    # In:
    woke_in: wp.array[int],
    # Data out:
    ntree_awake_out: wp.array[int],
    nbody_awake_out: wp.array[int],
    nv_awake_out: wp.array[int],
  ):
    if wp.static(has_woke):
      if woke_in[0] == 0:
        return
    ntree_awake_out[worldid] = 0
    nbody_awake_out[worldid] = 0
    nv_awake_out[worldid] = 0

  return kernel


@cache_kernel
def _update_sleep_trees(has_woke: bool = False):
  @fusion.stage("nworld", "ntree")
  def kernel(
    worldid: int,
    treeid: int,
    # Data in:
    tree_asleep_in: wp.array2d[int],
    # In:
    woke_in: wp.array[int],
    # Data out:
    ntree_awake_out: wp.array[int],
    tree_awake_out: wp.array2d[int],
  ):
    if wp.static(has_woke):
      if woke_in[0] == 0:
        return
    is_awake = int(tree_asleep_in[worldid, treeid] < 0)
    tree_awake_out[worldid, treeid] = is_awake
    if is_awake == 1:
      wp.atomic_add(ntree_awake_out, worldid, 1)  # kernel_analyzer: ignore[atomic]

  return kernel


@cache_kernel
def _update_sleep_bodies(has_woke: bool = False, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @fusion.stage("nworld", "nbody", options={**module_options})
  def kernel(
    worldid: int,
    bodyid: int,
    # Model:
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_mocapid: wp.array[int],
    body_treeid: wp.array[int],
    # Data in:
    tree_awake_in: wp.array2d[int],
    # In:
    flg_staticawake: int,
    woke_in: wp.array[int],
    # Data out:
    nbody_awake_out: wp.array[int],
    body_awake_out: wp.array2d[int],
    body_awake_ind_out: wp.array2d[int],
  ):
    if wp.static(has_woke):
      if woke_in[0] == 0:
        return

    tree = body_treeid[bodyid]

    # check static
    if tree < 0:
      root = body_rootid[bodyid]
      mocap = body_mocapid[root]
      if mocap >= 0:
        state = SleepState.AWAKE
      else:
        state = SleepState.AWAKE if flg_staticawake != 0 else SleepState.STATIC
    else:
      state = SleepState.AWAKE if tree_awake_in[worldid, tree] == 1 else SleepState.ASLEEP

    body_awake_out[worldid, bodyid] = state

    if state != SleepState.ASLEEP:
      idx = wp.atomic_add(nbody_awake_out, worldid, 1)
      body_awake_ind_out[worldid, idx] = bodyid

  return kernel


@cache_kernel
def _update_sleep_dofs(has_woke: bool = False, deterministic: bool = False):
  module_options = {"enable_backward": False}
  if deterministic:
    module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN

  @fusion.stage("nworld", "nv", options={**module_options})
  def kernel(
    worldid: int,
    dofid: int,
    # Model:
    body_treeid: wp.array[int],
    dof_bodyid: wp.array[int],
    # Data in:
    body_awake_in: wp.array2d[int],
    # In:
    woke_in: wp.array[int],
    # Data out:
    nv_awake_out: wp.array[int],
    dof_awake_ind_out: wp.array2d[int],
  ):
    if wp.static(has_woke):
      if woke_in[0] == 0:
        return
    bodyid = dof_bodyid[dofid]
    if body_treeid[bodyid] >= 0 and body_awake_in[worldid, bodyid] == SleepState.AWAKE:
      idx = wp.atomic_add(nv_awake_out, worldid, 1)
      dof_awake_ind_out[worldid, idx] = dofid

  return kernel


@event_scope
def update_sleep(
  m: types.Model,
  d: types.Data,
  flg_staticawake: int = 0,
  woke: Optional[wp.array] = None,
):
  """Computes sleeping arrays from tree_asleep."""
  has_woke = woke is not None
  woke_arr = woke if woke is not None else d.nacon
  det_atomics = bool(m.opt.deterministic & DeterminismType.ATOMICS)

  fusion.launch(m, d, _zero_sleep_counters(has_woke), woke_in=woke_arr)

  fusion.launch(m, d, _update_sleep_trees(has_woke), woke_in=woke_arr)

  fusion.launch(m, d, _update_sleep_bodies(has_woke, det_atomics), flg_staticawake=flg_staticawake, woke_in=woke_arr)

  fusion.launch(m, d, _update_sleep_dofs(has_woke, det_atomics), woke_in=woke_arr)


@wp.func
def _wake_tree(
  # Model:
  ntree: int,
  # In:
  worldid: int,
  treeid: int,
  wakeval: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],
) -> int:
  """Wakes tree treeid and its associated cycle, returning number of woke trees."""
  if treeid < 0 or treeid >= ntree:
    return 0

  asleep_val = tree_asleep_out[worldid, treeid]
  if asleep_val < 0:
    if wakeval < asleep_val:
      tree_asleep_out[worldid, treeid] = wakeval
    return 0

  nwoke = int(0)
  current = int(treeid)
  for step in range(ntree + 1):  # safe upper bound
    next_tree = tree_asleep_out[worldid, current]
    if next_tree < 0 or next_tree >= ntree:
      break

    tree_asleep_out[worldid, current] = wakeval
    nwoke += 1
    current = next_tree
    if current == treeid:
      break

  return nwoke


@wp.func
def _tree_can_sleep(
  # Model:
  nbody: int,
  body_treeid: wp.array[int],
  dof_length: wp.array[float],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  tree_sleep_policy: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qfrc_applied_in: wp.array2d[float],
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  treeid: int,
  sleep_tolerance: float,
) -> bool:
  policy = tree_sleep_policy[treeid]
  if policy == SleepPolicy.AUTO_NEVER:
    return False

  # check xfrc_applied
  for b in range(nbody):
    if body_treeid[b] == treeid:
      xfrc = xfrc_applied_in[worldid, b]
      for i in range(6):
        if xfrc[i] != 0.0:
          return False

  # check qfrc_applied
  dofadr = tree_dofadr[treeid]
  dofnum = tree_dofnum[treeid]
  for d in range(dofnum):
    if qfrc_applied_in[worldid, dofadr + d] != 0.0:
      return False

  # check qvel
  for d in range(dofnum):
    dof_idx = dofadr + d
    v = qvel_in[worldid, dof_idx]
    weight = dof_length[dof_idx]
    if sleep_tolerance > 0.0:
      if wp.abs(weight * v) >= sleep_tolerance:
        return False
    else:
      if v != 0.0:
        return False

  return True


@fusion.stage("nworld", "ntree")
def _wake_kernel(
  worldid: int,
  treeid: int,
  # Model:
  nbody: int,
  ntree: int,
  body_treeid: wp.array[int],
  dof_length: wp.array[float],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  tree_sleep_policy: wp.array[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qfrc_applied_in: wp.array2d[float],
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  tree_awake_in: wp.array2d[int],
  # Data out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
  # Out:
  woke_out: wp.array[int],
):
  asleep = int(tree_asleep_out[worldid, treeid] >= 0)
  if asleep == 0:
    if tree_awake_in[worldid, treeid] == 0:
      woke_out[0] = 1
    return

  # if tree_awake mismatch or cannot sleep: wake up
  if tree_awake_in[worldid, treeid] == 1 or not _tree_can_sleep(
    nbody,
    body_treeid,
    dof_length,
    tree_dofadr,
    tree_dofnum,
    tree_sleep_policy,
    qvel_in,
    qfrc_applied_in,
    xfrc_applied_in,
    worldid,
    treeid,
    0.0,  # zero tolerance
  ):
    if _wake_tree(ntree, worldid, treeid, K_AWAKE_VAL, tree_asleep_out) > 0:
      woke_out[0] = 1


@fusion.stage("nworld")
def _wake_non_metric_flex(
  worldid: int,
  # Model:
  ntree: int,
  nflex: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  flex_interp: wp.array[int],
  flex_nodeadr: wp.array[int],
  flex_nodenum: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_edgeequality: wp.array[int],
  flex_rod_has_twist: wp.array[bool],
  flex_eff_possible: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  # Data in:
  ntree_awake_in: wp.array[int],
  # In:
  flg_wake: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
  # Out:
  woke_out: wp.array[int],
):
  if flg_wake != 0 and ntree_awake_in[worldid] >= ntree:
    return
  for f in range(nflex):
    if flex_edgeequality[f] != 0 or not flex_eff_possible[f]:
      continue
    if flex_interp[f] != 0:
      num = flex_nodenum[f]
      adr = flex_nodeadr[f]
      for j in range(num):
        t = body_treeid[flex_nodebodyid[adr + j]]
        if t >= 0:
          if tree_asleep_out[worldid, t] >= 0:
            if flg_wake != 0:
              if _wake_tree(ntree, worldid, t, K_AWAKE_VAL, tree_asleep_out) > 0:
                woke_out[0] = 1
          else:
            tree_asleep_out[worldid, t] = K_AWAKE_VAL
    else:
      num = flex_vertnum[f]
      adr = flex_vertadr[f]
      for j in range(num):
        t = body_treeid[flex_vertbodyid[adr + j]]
        if t >= 0:
          if tree_asleep_out[worldid, t] >= 0:
            if flg_wake != 0:
              if _wake_tree(ntree, worldid, t, K_AWAKE_VAL, tree_asleep_out) > 0:
                woke_out[0] = 1
          else:
            tree_asleep_out[worldid, t] = K_AWAKE_VAL
    if flex_rod_has_twist[f]:
      eadr = flex_rod_edgeadr[f]
      nedge = flex_rod_edgenum[f]
      for e in range(nedge):
        jnt = flex_rodedge_jnt[eadr + e]
        if jnt < 0:
          continue
        t = body_treeid[jnt_bodyid[jnt]]
        if t < 0:
          continue
        if tree_asleep_out[worldid, t] >= 0:
          if flg_wake != 0:
            if _wake_tree(ntree, worldid, t, K_AWAKE_VAL, tree_asleep_out) > 0:
              woke_out[0] = 1
        else:
          tree_asleep_out[worldid, t] = K_AWAKE_VAL


@wp.func
def _flex_contact_body(
  # Model:
  body_treeid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_nodeadr: wp.array[int],
  flex_nodenum: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  # In:
  f: int,
  elem: int,
  vert: int,
) -> int:
  if vert >= 0 and flex_interp[f] == 0:
    return flex_vertbodyid[flex_vertadr[f] + vert]
  if elem >= 0 and flex_interp[f] == 0:
    dim = flex_dim[f]
    edata = flex_elemdataadr[f] + elem * (dim + 1)
    vadr = flex_vertadr[f]
    for j in range(dim + 1):
      b = flex_vertbodyid[vadr + flex_elem[edata + j]]
      if body_treeid[b] >= 0:
        return b
    return flex_vertbodyid[vadr + flex_elem[edata]]
  nodenum = flex_nodenum[f]
  nadr = flex_nodeadr[f]
  for j in range(nodenum):
    b = flex_nodebodyid[nadr + j]
    if body_treeid[b] >= 0:
      return b
  return flex_nodebodyid[nadr]


@fusion.stage("naconmax")
def _wake_collision_kernel(
  conid: int,
  # Model:
  ntree: int,
  body_treeid: wp.array[int],
  geom_bodyid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_nodeadr: wp.array[int],
  flex_nodenum: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  body_awake_in: wp.array2d[int],
  contact_geom_in: wp.array[wp.vec2i],
  contact_flex_in: wp.array[wp.vec2i],
  contact_elem_in: wp.array[wp.vec2i],
  contact_vert_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  nacon_in: wp.array[int],
  # Data out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
  # Out:
  woke_out: wp.array[int],
):
  if conid >= nacon_in[0]:
    return
  ctype = contact_type_in[conid]
  if ctype != 0 and not (ctype & (ContactType.CONSTRAINT | ContactType.PASSIVE)):
    return

  geom_pair = contact_geom_in[conid]
  g1 = geom_pair[0]
  g2 = geom_pair[1]

  if g1 >= 0 and g2 >= 0:
    b1 = geom_bodyid[g1]
    b2 = geom_bodyid[g2]
  else:
    if contact_flex_in.shape[0] == 0:
      return
    flex_pair = contact_flex_in[conid]
    elem_pair = contact_elem_in[conid]
    vert_pair = contact_vert_in[conid]
    if (g1 < 0 and flex_pair[0] < 0) or (g2 < 0 and flex_pair[1] < 0):
      return
    b1 = (
      geom_bodyid[g1]
      if g1 >= 0
      else _flex_contact_body(
        body_treeid,
        flex_dim,
        flex_interp,
        flex_nodeadr,
        flex_nodenum,
        flex_vertadr,
        flex_elemdataadr,
        flex_nodebodyid,
        flex_vertbodyid,
        flex_elem,
        flex_pair[0],
        elem_pair[0],
        vert_pair[0],
      )
    )
    b2 = (
      geom_bodyid[g2]
      if g2 >= 0
      else _flex_contact_body(
        body_treeid,
        flex_dim,
        flex_interp,
        flex_nodeadr,
        flex_nodenum,
        flex_vertadr,
        flex_elemdataadr,
        flex_nodebodyid,
        flex_vertbodyid,
        flex_elem,
        flex_pair[1],
        elem_pair[1],
        vert_pair[1],
      )
    )
  tree1 = body_treeid[b1]
  tree2 = body_treeid[b2]
  worldid = contact_worldid_in[conid]

  if tree1 < 0 or tree2 < 0:
    tree = tree2 if tree1 < 0 else tree1
    b = b1 if tree1 < 0 else b2
    if tree >= 0 and tree_awake_in[worldid, tree] == 0 and body_awake_in[worldid, b] == SleepState.AWAKE:
      if _wake_tree(ntree, worldid, tree, K_AWAKE_VAL, tree_asleep_out) > 0:
        woke_out[0] = 1
    return

  awake1 = tree_awake_in[worldid, tree1]
  awake2 = tree_awake_in[worldid, tree2]

  if awake1 == 1 and awake2 == 1:
    return

  if awake1 == 0 and awake2 == 0:
    return

  # wake sleeping tree
  sleeping_tree = tree2 if awake1 == 1 else tree1
  wakeval = tree_asleep_out[worldid, tree1] if awake1 == 1 else tree_asleep_out[worldid, tree2]

  if _wake_tree(ntree, worldid, sleeping_tree, wakeval, tree_asleep_out) > 0:
    woke_out[0] = 1


@fusion.stage("nworld", "ntendon")
def _wake_tendon_kernel(
  worldid: int,
  tenid: int,
  # Model:
  ntree: int,
  ntendon: int,
  opt_integrator: int,
  opt_disableflags: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  tendon_actuatorid: wp.array[int],
  tendon_limited: wp.array[int],
  tendon_range: wp.array2d[wp.vec2],
  tendon_margin: wp.array2d[float],
  tendon_stiffness: wp.array2d[float],
  tendon_stiffnesspoly: wp.array2d[wp.vec2],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  tendon_frictionloss: wp.array2d[float],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  ten_length_in: wp.array2d[float],
  tree_awake_in: wp.array2d[int],
  # Data out:
  tree_asleep_out: wp.array2d[int],
  # Out:
  woke_out: wp.array[int],
):
  adr = tendon_adr[tenid]
  num = tendon_num[tenid]

  # Pass 1: Check if any tree involved in the tendon is awake
  any_awake = int(0)
  wakeval = int(K_AWAKE_VAL)

  for i in range(num):
    idx = adr + i
    t_type = wrap_type[idx]
    t_objid = wrap_objid[idx]
    t = int(-1)
    if t_type == WrapType.JOINT:
      t = body_treeid[jnt_bodyid[t_objid]]
    elif t_type == WrapType.SITE:
      t = body_treeid[site_bodyid[t_objid]]
    elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
      t = body_treeid[geom_bodyid[t_objid]]

    if t >= 0:
      if tree_awake_in[worldid, t] == 1:
        any_awake = 1
        val = tree_asleep_out[worldid, t]
        if val < wakeval:
          wakeval = val

  # Pass 2: Wake sleeping trees if limit, frictionloss, or discrete metric is active
  if any_awake == 1:
    has_limit = (
      not (opt_disableflags & DisableBit.CONSTRAINT)
      and not (opt_disableflags & DisableBit.LIMIT)
      and _tendon_limit_active(tendon_limited, tendon_range, tendon_margin, ten_length_in, worldid, tenid)
    )
    has_friction = (
      not (opt_disableflags & DisableBit.CONSTRAINT)
      and not (opt_disableflags & DisableBit.FRICTIONLOSS)
      and tendon_frictionloss[worldid % tendon_frictionloss.shape[0], tenid] > 0.0
    )
    is_metric = opt_integrator == IntegratorType.DISCRETE and support.eff_tendon_possible(
      opt_disableflags,
      tendon_actuatorid,
      tendon_stiffness,
      tendon_stiffnesspoly,
      tendon_damping,
      tendon_dampingpoly,
      worldid,
      tenid,
    )
    if has_limit or has_friction or is_metric:
      for i in range(num):
        idx = adr + i
        t_type = wrap_type[idx]
        t_objid = wrap_objid[idx]
        t = int(-1)
        if t_type == WrapType.JOINT:
          t = body_treeid[jnt_bodyid[t_objid]]
        elif t_type == WrapType.SITE:
          t = body_treeid[site_bodyid[t_objid]]
        elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
          t = body_treeid[geom_bodyid[t_objid]]

        if t >= 0:
          if tree_awake_in[worldid, t] == 0:
            if _wake_tree(ntree, worldid, t, wakeval, tree_asleep_out) > 0:
              woke_out[0] = 1


@wp.func
def _tendon_wake_val(
  # Model:
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  # In:
  worldid: int,
  tenid: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],
) -> int:
  """Returns the minimum (most-awake) tree_asleep value if any tree in tendon is awake, else 0."""
  if tenid < 0:
    return 0

  adr = tendon_adr[tenid]
  num = tendon_num[tenid]
  wakeval = int(0)

  for i in range(num):
    idx = adr + i
    t_type = wrap_type[idx]
    t_objid = wrap_objid[idx]
    t = int(-1)
    if t_type == WrapType.JOINT:
      t = body_treeid[jnt_bodyid[t_objid]]
    elif t_type == WrapType.SITE:
      t = body_treeid[site_bodyid[t_objid]]
    elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
      t = body_treeid[geom_bodyid[t_objid]]

    if t >= 0:
      if tree_awake_in[worldid, t] == 1:
        val = tree_asleep_out[worldid, t]
        if wakeval == 0 or val < wakeval:
          wakeval = val

  return wakeval


@wp.func
def _wake_tendon_trees(
  # Model:
  ntree: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  # In:
  worldid: int,
  tenid: int,
  wakeval: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],
) -> int:
  """Wakes up all sleeping trees associated with a tendon."""
  if tenid < 0:
    return 0

  nwoke = int(0)
  adr = tendon_adr[tenid]
  num = tendon_num[tenid]
  for i in range(num):
    idx = adr + i
    t_type = wrap_type[idx]
    t_objid = wrap_objid[idx]
    t = int(-1)
    if t_type == WrapType.JOINT:
      t = body_treeid[jnt_bodyid[t_objid]]
    elif t_type == WrapType.SITE:
      t = body_treeid[site_bodyid[t_objid]]
    elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
      t = body_treeid[geom_bodyid[t_objid]]

    if t >= 0:
      if tree_awake_in[worldid, t] == 0:
        nwoke += _wake_tree(ntree, worldid, t, wakeval, tree_asleep_out)

  return nwoke


@fusion.stage("nworld", "neq")
def _wake_equality_kernel(
  worldid: int,
  eqid: int,
  # Model:
  ntree: int,
  neq: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  flex_interp: wp.array[int],
  flex_nodeadr: wp.array[int],
  flex_nodenum: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_objtype: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  # Data in:
  eq_active_in: wp.array2d[bool],  # kernel_analyzer: ignore
  tree_awake_in: wp.array2d[int],
  # Data out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
  # Out:
  woke_out: wp.array[int],
):
  if not eq_active_in[worldid, eqid]:
    return

  eqtype = eq_type[eqid]
  id1 = eq_obj1id[eqid]
  id2 = eq_obj2id[eqid]

  if eqtype == EqType.CONNECT or eqtype == EqType.WELD or eqtype == EqType.JOINT:
    tree1 = int(-1)
    tree2 = int(-1)

    if eqtype == EqType.CONNECT or eqtype == EqType.WELD:
      if eq_objtype[eqid] == ObjType.BODY:
        tree1 = body_treeid[id1]
        tree2 = body_treeid[id2]
      else:
        tree1 = body_treeid[site_bodyid[id1]]
        tree2 = body_treeid[site_bodyid[id2]]
    elif eqtype == EqType.JOINT:
      tree1 = body_treeid[jnt_bodyid[id1]] if id1 >= 0 else -1
      tree2 = body_treeid[jnt_bodyid[id2]] if id2 >= 0 else -1

    s1 = tree_awake_in[worldid, tree1] if tree1 >= 0 else SleepState.STATIC
    s2 = tree_awake_in[worldid, tree2] if tree2 >= 0 else SleepState.STATIC

    if s1 != SleepState.ASLEEP and s2 != SleepState.ASLEEP:
      return
    if s1 == SleepState.STATIC or s2 == SleepState.STATIC:
      return
    if tree1 == tree2:
      return

    if s1 == SleepState.ASLEEP and s2 == SleepState.ASLEEP:
      cycle1 = _sleep_cycle(tree_asleep_out, ntree, worldid, tree1)
      cycle2 = _sleep_cycle(tree_asleep_out, ntree, worldid, tree2)
      if cycle1 != cycle2:
        w1 = _wake_tree(ntree, worldid, tree1, K_AWAKE_VAL, tree_asleep_out)
        w2 = _wake_tree(ntree, worldid, tree2, K_AWAKE_VAL, tree_asleep_out)
        if w1 + w2 > 0:
          woke_out[0] = 1
    else:
      sleeping_tree = tree1 if s1 == SleepState.ASLEEP else tree2
      if _wake_tree(ntree, worldid, sleeping_tree, K_AWAKE_VAL, tree_asleep_out) > 0:
        woke_out[0] = 1

  elif eqtype == EqType.TENDON:
    ten1 = id1
    ten2 = id2
    w1 = _tendon_wake_val(
      body_treeid,
      jnt_bodyid,
      geom_bodyid,
      site_bodyid,
      tendon_adr,
      tendon_num,
      wrap_type,
      wrap_objid,
      tree_awake_in,
      worldid,
      ten1,
      tree_asleep_out,
    )
    w2 = _tendon_wake_val(
      body_treeid,
      jnt_bodyid,
      geom_bodyid,
      site_bodyid,
      tendon_adr,
      tendon_num,
      wrap_type,
      wrap_objid,
      tree_awake_in,
      worldid,
      ten2,
      tree_asleep_out,
    )

    if w1 < 0 or w2 < 0:
      wakeval = int(K_AWAKE_VAL)
      if w1 < 0 and w1 < wakeval:
        wakeval = w1
      if w2 < 0 and w2 < wakeval:
        wakeval = w2

      nw1 = _wake_tendon_trees(
        ntree,
        body_treeid,
        jnt_bodyid,
        geom_bodyid,
        site_bodyid,
        tendon_adr,
        tendon_num,
        wrap_type,
        wrap_objid,
        tree_awake_in,
        worldid,
        ten1,
        wakeval,
        tree_asleep_out,
      )
      nw2 = _wake_tendon_trees(
        ntree,
        body_treeid,
        jnt_bodyid,
        geom_bodyid,
        site_bodyid,
        tendon_adr,
        tendon_num,
        wrap_type,
        wrap_objid,
        tree_awake_in,
        worldid,
        ten2,
        wakeval,
        tree_asleep_out,
      )
      if nw1 + nw2 > 0:
        woke_out[0] = 1

  elif eqtype == EqType.FLEX or eqtype == EqType.FLEXSTRAIN:
    f = id1
    any_awake = int(0)
    wakeval = int(0)
    if flex_interp[f] != 0:
      num = flex_nodenum[f]
      adr = flex_nodeadr[f]
      for j in range(num):
        treeid = body_treeid[flex_nodebodyid[adr + j]]
        if treeid >= 0 and tree_awake_in[worldid, treeid] != 0:
          any_awake = 1
          wakeval = wp.min(wakeval, tree_asleep_out[worldid, treeid])
    else:
      num = flex_vertnum[f]
      adr = flex_vertadr[f]
      for j in range(num):
        treeid = body_treeid[flex_vertbodyid[adr + j]]
        if treeid >= 0 and tree_awake_in[worldid, treeid] != 0:
          any_awake = 1
          wakeval = wp.min(wakeval, tree_asleep_out[worldid, treeid])

    nedge = flex_rod_edgenum[f] if flex_rod_has_twist[f] else 0
    eadr = flex_rod_edgeadr[f]
    for e in range(nedge):
      jnt = flex_rodedge_jnt[eadr + e]
      treeid = body_treeid[jnt_bodyid[jnt]] if jnt >= 0 else -1
      if treeid >= 0 and tree_awake_in[worldid, treeid] != 0:
        any_awake = 1
        wakeval = wp.min(wakeval, tree_asleep_out[worldid, treeid])

    if any_awake != 0:
      if flex_interp[f] != 0:
        num = flex_nodenum[f]
        adr = flex_nodeadr[f]
        for j in range(num):
          treeid = body_treeid[flex_nodebodyid[adr + j]]
          if treeid >= 0 and tree_asleep_out[worldid, treeid] >= 0:
            if _wake_tree(ntree, worldid, treeid, wakeval, tree_asleep_out) > 0:
              woke_out[0] = 1
      else:
        num = flex_vertnum[f]
        adr = flex_vertadr[f]
        for j in range(num):
          treeid = body_treeid[flex_vertbodyid[adr + j]]
          if treeid >= 0 and tree_asleep_out[worldid, treeid] >= 0:
            if _wake_tree(ntree, worldid, treeid, wakeval, tree_asleep_out) > 0:
              woke_out[0] = 1
      for e in range(nedge):
        jnt = flex_rodedge_jnt[eadr + e]
        treeid = body_treeid[jnt_bodyid[jnt]] if jnt >= 0 else -1
        if treeid >= 0 and tree_asleep_out[worldid, treeid] >= 0:
          if _wake_tree(ntree, worldid, treeid, wakeval, tree_asleep_out) > 0:
            woke_out[0] = 1


@event_scope
def wake(m: types.Model, d: types.Data, woke: Optional[wp.array] = None):
  """Wakes sleeping trees due to user changes/perturbations."""
  if woke is None:
    woke = wp.empty(1, dtype=int)
  fusion.launch(m, d, _wake_kernel, woke_out=woke)
  if m.nflex > 0 and m.opt.integrator != IntegratorType.DISCRETE:
    fusion.launch(m, d, _wake_non_metric_flex, flg_wake=1, woke_out=woke)


@event_scope
def wake_collision(m: types.Model, d: types.Data, woke: Optional[wp.array] = None):
  """Wakes sleeping trees that touch awake trees."""
  if woke is None:
    woke = wp.empty(1, dtype=int)
  fusion.launch(m, d, _wake_collision_kernel, woke_out=woke)


@event_scope
def wake_tendon(m: types.Model, d: types.Data, woke: Optional[wp.array] = None):
  """Wakes sleeping trees with constrained or metric tendons."""
  is_metric = m.opt.integrator == IntegratorType.DISCRETE
  if m.ntendon == 0 or ((m.opt.disableflags & DisableBit.CONSTRAINT) and not is_metric):
    return

  if woke is None:
    woke = wp.empty(1, dtype=int)
  fusion.launch(m, d, _wake_tendon_kernel, woke_out=woke)


@event_scope
def wake_equality(m: types.Model, d: types.Data, woke: Optional[wp.array] = None):
  """Wakes sleeping trees connected by equality constraints."""
  if m.neq == 0:
    return

  if woke is None:
    woke = wp.empty(1, dtype=int)
  fusion.launch(m, d, _wake_equality_kernel, woke_out=woke)


@fusion.stage("nworld", "ntree")
def _sweep_awake_trees(  # kernel_analyzer: ignore
  worldid: int,
  treeid: int,
  # Model:
  nbody: int,
  body_treeid: wp.array[int],
  dof_length: wp.array[float],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  tree_sleep_policy: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  nisland_in: wp.array[int],
  qvel_in: wp.array2d[float],
  qfrc_applied_in: wp.array2d[float],
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  # In:
  opt_sleep_tolerance: wp.array[float],
  # Data out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
  # Out:
  has_sleep_candidate_out: wp.array[int],
):
  # Constrained worlds without island discovery cannot sleep
  if nefc_in[worldid] > 0 and nisland_in[worldid] == 0:
    return
  sleep_tolerance = opt_sleep_tolerance[worldid % opt_sleep_tolerance.shape[0]]
  as_val = tree_asleep_out[worldid, treeid]
  if as_val >= 0:
    return

  if _tree_can_sleep(
    nbody,
    body_treeid,
    dof_length,
    tree_dofadr,
    tree_dofnum,
    tree_sleep_policy,
    qvel_in,
    qfrc_applied_in,
    xfrc_applied_in,
    worldid,
    treeid,
    sleep_tolerance,
  ):
    next_asleep = as_val + 1 if as_val < -1 else as_val
    if as_val < -1:
      tree_asleep_out[worldid, treeid] = next_asleep
    if next_asleep == -1:
      has_sleep_candidate_out[0] = 1
  else:
    tree_asleep_out[worldid, treeid] = K_AWAKE_VAL


@fusion.stage("nworld", "ntree")
def _check_island_can_sleep(
  worldid: int,
  treeid: int,
  # Model:
  ntree: int,
  # Data in:
  nisland_in: wp.array[int],
  tree_asleep_in: wp.array2d[int],
  tree_island_in: wp.array2d[int],
  # Out:
  island_can_sleep_out: wp.array2d[int],
):
  nisland = nisland_in[worldid]
  island_id = tree_island_in[worldid, treeid]
  if island_id >= 0 and island_id < nisland:
    as_val = tree_asleep_in[worldid, treeid]
    if as_val < -1:
      # Not ready to sleep yet
      wp.atomic_min(island_can_sleep_out, worldid, island_id, 0)  # kernel_analyzer: ignore[atomic]


@fusion.stage("nworld")
def _build_cycles(  # kernel_analyzer: ignore
  worldid: int,
  # Model:
  ntree: int,
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  nisland_in: wp.array[int],
  tree_island_in: wp.array2d[int],
  # In:
  island_can_sleep_in: wp.array2d[int],
  has_sleep_candidate_in: wp.array[int],
  # Data out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
  qvel_out: wp.array2d[float],  # kernel_analyzer: ignore
  qacc_out: wp.array2d[float],  # kernel_analyzer: ignore
  # Out:
  nslept_out: wp.array[int],
):
  if has_sleep_candidate_in[0] == 0:
    return
  if nefc_in[worldid] > 0 and nisland_in[worldid] == 0:
    return

  asleep_any = int(0)
  num_islands = nisland_in[worldid]
  for island_id in range(num_islands):
    if island_can_sleep_in[worldid, island_id] == 1:
      first_tree = int(-1)
      prev_tree = int(-1)
      for t in range(ntree):
        if tree_island_in[worldid, t] == island_id:
          if first_tree == -1:
            first_tree = t
          if prev_tree != -1:
            tree_asleep_out[worldid, prev_tree] = t
          prev_tree = t

          # Zero velocities and accelerations
          dofadr = tree_dofadr[t]
          dofnum = tree_dofnum[t]
          for d in range(dofnum):
            qvel_out[worldid, dofadr + d] = 0.0
            qacc_out[worldid, dofadr + d] = 0.0

      if first_tree != -1:
        tree_asleep_out[worldid, prev_tree] = first_tree
        asleep_any = 1

  # Sleep unconstrained trees
  for t in range(ntree):
    island_id = tree_island_in[worldid, t]
    if island_id < 0 or island_id >= num_islands:
      if tree_asleep_out[worldid, t] == -1:
        tree_asleep_out[worldid, t] = t  # self-cycle
        asleep_any = 1
        dofadr = tree_dofadr[t]
        dofnum = tree_dofnum[t]
        for d in range(dofnum):
          qvel_out[worldid, dofadr + d] = 0.0
          qacc_out[worldid, dofadr + d] = 0.0

  if asleep_any != 0:
    nslept_out[0] = 1


@event_scope
def sleep(m: types.Model, d: types.Data) -> wp.array:
  """Puts trees to sleep according to velocity tolerance."""
  if m.ntree == 0:
    return wp.zeros(1, dtype=int)

  has_sleep_candidate = wp.zeros(1, dtype=int)
  nslept = wp.zeros(1, dtype=int)

  # Sweep over awake trees and increment counter if they can sleep
  fusion.launch(m, d, _sweep_awake_trees, has_sleep_candidate_out=has_sleep_candidate)

  if m.nflex > 0 and m.opt.integrator != IntegratorType.DISCRETE:
    fusion.launch(m, d, _wake_non_metric_flex, flg_wake=0, woke_out=has_sleep_candidate)

  # Check which constraint islands can sleep (all trees in island must be asleep)
  island_can_sleep = wp.ones((d.nworld, m.ntree), dtype=int)
  fusion.launch(m, d, _check_island_can_sleep, island_can_sleep_out=island_can_sleep)

  # Build sleep cycles for sleeping islands and sleep unconstrained trees
  fusion.launch(
    m, d, _build_cycles, island_can_sleep_in=island_can_sleep, has_sleep_candidate_in=has_sleep_candidate, nslept_out=nslept
  )
  return nslept
