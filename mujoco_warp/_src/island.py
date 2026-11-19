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

import warnings
from typing import Any, Tuple

import warp as wp

from mujoco_warp._src import passive
from mujoco_warp._src import smooth
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.types import ConstraintType
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import EqType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import ObjType
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def _dsu_find(parent: wp.array2d[int], worldid: int, tree: int):
  """Find a root while shortening its strictly decreasing parent path."""
  current = tree
  while True:
    parent_current = parent[worldid, current]
    if parent_current == current:
      return current
    grandparent = parent[worldid, parent_current]
    if grandparent < parent_current:
      wp.atomic_min(parent, worldid, current, grandparent)  # kernel_analyzer: ignore[atomic]
    current = parent_current
  return current


@wp.func
def _dsu_activate(worldid: int, tree: int, tree_island_out: wp.array2d[int]):
  if tree >= 0:
    wp.atomic_max(tree_island_out, worldid, tree, 0)  # kernel_analyzer: ignore[atomic]


@wp.func
def _dsu_union(parent: wp.array2d[int], worldid: int, tree0: int, tree1: int, tree_island_out: wp.array2d[int]):
  """Activate endpoints and atomically hook the higher root to the lower root."""
  if tree0 < 0:
    tree0 = tree1
    tree1 = -1
  if tree0 < 0:
    return
  _dsu_activate(worldid, tree0, tree_island_out)
  if tree1 < 0:
    return
  _dsu_activate(worldid, tree1, tree_island_out)

  while True:
    root0 = _dsu_find(parent, worldid, tree0)
    root1 = _dsu_find(parent, worldid, tree1)
    if root0 == root1:
      return
    low_root = wp.min(root0, root1)
    high_root = wp.max(root0, root1)
    previous = wp.atomic_cas(parent, worldid, high_root, high_root, low_root)  # kernel_analyzer: ignore[atomic]
    if previous == high_root:
      return


@wp.func
def _union_tendon_trees(
  # Model:
  opt_enableflags: int,
  dof_treeid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  # In:
  worldid: int,
  tenid: int,
  # Data out:
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  start = ten_J_rowadr[tenid]
  end = start + ten_J_rownnz[tenid]
  sleep_enabled = bool(opt_enableflags & EnableBit.SLEEP)
  tree1 = int(-1)
  for j in range(start, end):
    tree2 = dof_treeid[ten_J_colind[j]]
    if tree2 < 0:
      continue
    if sleep_enabled and tree_awake_in[worldid, tree2] == 0:
      continue
    if tree1 < 0:
      tree1 = tree2
    elif tree1 != tree2:
      _dsu_union(island_parent_out, worldid, tree1, tree2, tree_island_out)


@wp.kernel
def _reset_dsu(
  # Data out:
  nisland_out: wp.array[int],
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  """Reset identity parents and active/output markers in parallel."""
  worldid, treeid = wp.tid()
  island_parent_out[worldid, treeid] = treeid
  tree_island_out[worldid, treeid] = -1
  if treeid == 0:
    nisland_out[worldid] = 0


@wp.func
def _is_repeated_fixed_support_efc(
  # Data in:
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  # In:
  worldid: int,
  efcid: int,
) -> bool:
  """Return whether the preceding scalar row has the same fixed tree support."""
  if efcid == 0:
    return False
  return (
    efc_type_in[worldid, efcid - 1] == efc_type_in[worldid, efcid]
    and efc_id_in[worldid, efcid - 1] == efc_id_in[worldid, efcid]
  )


@wp.func
def _flex_contact_bodies(
  # Model:
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_celladr: wp.array[int],
  flex_nodeadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_cellempty: wp.array[bool],
  flex_elem: wp.array[int],
  flex_vert0: wp.array[wp.vec3],
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  # Data in:
  flexvert_xpos_in: wp.array2d[wp.vec3],
  contact_pos_in: wp.array[wp.vec3],
  contact_geom_in: wp.array[wp.vec2i],
  contact_flex_in: wp.array[wp.vec2i],
  contact_elem_in: wp.array[wp.vec2i],
  contact_vert_in: wp.array[wp.vec2i],
  # In:
  worldid: int,
  conid: int,
) -> Tuple[wp.vec4i, wp.vec4i]:
  """Return the bodies of both sides of a flex contact, -1 padded.

  The rows of a flex contact can vanish (e.g. contact point on a hinge axis), so its trees
  are taken from its bodies, as for its Jacobian, rather than from the values of its rows.
  """
  geom = contact_geom_in[conid]
  flex = contact_flex_in[conid]
  elem = contact_elem_in[conid]
  vert = contact_vert_in[conid]
  pos = contact_pos_in[conid]
  bodies0, _weights0 = support.contact_bodies_and_weights(
    jnt_bodyid,
    geom_bodyid,
    flex_dim,
    flex_interp,
    flex_cellnum,
    flex_celladr,
    flex_nodeadr,
    flex_vertadr,
    flex_elemadr,
    flex_elemnum,
    flex_elemdataadr,
    flex_nodebodyid,
    flex_vertbodyid,
    flex_cellempty,
    flex_elem,
    flex_vert0,
    flex_rod_has_twist,
    flex_rod_edgeadr,
    flex_rod_edgenum,
    flex_rodedge_jnt,
    flex_elem_rodedge,
    flex_vert_rodelem,
    flexvert_xpos_in,
    conid,
    0,
    geom,
    flex,
    elem,
    vert,
    pos,
    worldid,
  )
  bodies1, _weights1 = support.contact_bodies_and_weights(
    jnt_bodyid,
    geom_bodyid,
    flex_dim,
    flex_interp,
    flex_cellnum,
    flex_celladr,
    flex_nodeadr,
    flex_vertadr,
    flex_elemadr,
    flex_elemnum,
    flex_elemdataadr,
    flex_nodebodyid,
    flex_vertbodyid,
    flex_cellempty,
    flex_elem,
    flex_vert0,
    flex_rod_has_twist,
    flex_rod_edgeadr,
    flex_rod_edgenum,
    flex_rodedge_jnt,
    flex_elem_rodedge,
    flex_vert_rodelem,
    flexvert_xpos_in,
    conid,
    1,
    geom,
    flex,
    elem,
    vert,
    pos,
    worldid,
  )
  return bodies0, bodies1


@wp.kernel
def _flex_island_dsu(
  # Model:
  opt_enableflags: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  flex_interp: wp.array[int],
  flex_nodeadr: wp.array[int],
  flex_nodenum: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_rod_has_twist: wp.array[bool],
  flex_eff_possible: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  tree_awake_in: wp.array2d[int],
  # In:
  sleep_metric_flex: bool,
  # Data out:
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  """Unions awake dynamic trees of each metric-coupled flex (vertices, nodes, twist joints)."""
  worldid, f = wp.tid()
  if not flex_eff_possible[f]:
    return
  if nefc_in[worldid] == 0 and not sleep_metric_flex:
    return
  sleep_enabled = (opt_enableflags & types.EnableBit.SLEEP) != 0
  use_interp = flex_interp[f] != 0
  num = flex_nodenum[f] if use_interp else flex_vertnum[f]
  adr = flex_nodeadr[f] if use_interp else flex_vertadr[f]
  tree1 = int(-1)
  for j in range(num):
    b = flex_nodebodyid[adr + j] if use_interp else flex_vertbodyid[adr + j]
    tree2 = body_treeid[b]
    if tree2 < 0 or tree2 == tree1 or (sleep_enabled and tree_awake_in[worldid, tree2] == 0):
      continue
    if tree1 < 0:
      tree1 = tree2
    else:
      _dsu_union(island_parent_out, worldid, tree1, tree2, tree_island_out)
  if flex_rod_has_twist[f]:
    eadr = flex_rod_edgeadr[f]
    for e in range(flex_rod_edgenum[f]):
      jnt = flex_rodedge_jnt[eadr + e]
      if jnt < 0:
        continue
      tree2 = body_treeid[jnt_bodyid[jnt]]
      if tree2 < 0 or tree2 == tree1 or (sleep_enabled and tree_awake_in[worldid, tree2] == 0):
        continue
      if tree1 < 0:
        tree1 = tree2
      else:
        _dsu_union(island_parent_out, worldid, tree1, tree2, tree_island_out)


@wp.kernel
def _passive_contact_island_dsu(
  # Model:
  opt_enableflags: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_celladr: wp.array[int],
  flex_nodeadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_cellempty: wp.array[bool],
  flex_elem: wp.array[int],
  flex_vert0: wp.array[wp.vec3],
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  tree_awake_in: wp.array2d[int],
  contact_pos_in: wp.array[wp.vec3],
  contact_geom_in: wp.array[wp.vec2i],
  contact_flex_in: wp.array[wp.vec2i],
  contact_elem_in: wp.array[wp.vec2i],
  contact_vert_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  sleep_metric_flex: bool,
  # Data out:
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  """Unions awake dynamic trees coupled by passive flex contacts under the discrete integrator."""
  cid = wp.tid()
  if cid >= nacon_in[0]:
    return
  if (contact_type_in[cid] & int(types.ContactType.PASSIVE)) == 0:
    return
  worldid = contact_worldid_in[cid]
  if nefc_in[worldid] == 0 and not sleep_metric_flex:
    return
  sleep_enabled = (opt_enableflags & types.EnableBit.SLEEP) != 0
  bodies0, bodies1 = _flex_contact_bodies(
    jnt_bodyid,
    geom_bodyid,
    flex_dim,
    flex_interp,
    flex_cellnum,
    flex_celladr,
    flex_nodeadr,
    flex_vertadr,
    flex_elemadr,
    flex_elemnum,
    flex_elemdataadr,
    flex_nodebodyid,
    flex_vertbodyid,
    flex_cellempty,
    flex_elem,
    flex_vert0,
    flex_rod_has_twist,
    flex_rod_edgeadr,
    flex_rod_edgenum,
    flex_rodedge_jnt,
    flex_elem_rodedge,
    flex_vert_rodelem,
    flexvert_xpos_in,
    contact_pos_in,
    contact_geom_in,
    contact_flex_in,
    contact_elem_in,
    contact_vert_in,
    worldid,
    cid,
  )
  tree1 = int(-1)
  for k in range(8):
    body = bodies0[k % 4] if k < 4 else bodies1[k % 4]
    if body < 0:
      continue
    tree2 = body_treeid[body]
    if tree2 < 0 or tree2 == tree1 or (sleep_enabled and tree_awake_in[worldid, tree2] == 0):
      continue
    if tree1 < 0:
      tree1 = tree2
    else:
      _dsu_union(island_parent_out, worldid, tree1, tree2, tree_island_out)


@wp.kernel
def _island_dsu_metric_couplings(
  # Model:
  nactuator: int,
  ntendon: int,
  opt_disableflags: int,
  opt_enableflags: int,
  body_treeid: wp.array[int],
  dof_treeid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_actuatorid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  tendon_stiffness: wp.array2d[float],
  tendon_stiffnesspoly: wp.array2d[wp.vec2],
  tendon_damping: wp.array2d[float],
  tendon_dampingpoly: wp.array2d[wp.vec2],
  actuator_trntype: wp.array[int],
  actuator_gaintype: wp.array[int],
  actuator_biastype: wp.array[int],
  actuator_trnid: wp.array[wp.vec2i],
  # Data in:
  nefc_in: wp.array[int],
  tree_awake_in: wp.array2d[int],
  # Data out:
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  worldid, idx = wp.tid()
  if nefc_in[worldid] == 0:
    return

  if idx < ntendon:
    if support.eff_tendon_possible(
      opt_disableflags,
      tendon_actuatorid,
      tendon_stiffness,
      tendon_stiffnesspoly,
      tendon_damping,
      tendon_dampingpoly,
      worldid,
      idx,
    ):
      _union_tendon_trees(
        opt_enableflags,
        dof_treeid,
        ten_J_rownnz,
        ten_J_rowadr,
        ten_J_colind,
        tree_awake_in,
        worldid,
        idx,
        tree_island_out,
        island_parent_out,
      )

  if idx < nactuator:
    if support.eff_actuator_possible(opt_disableflags, actuator_gaintype, actuator_biastype, idx):
      trntype = actuator_trntype[idx]
      trnid = actuator_trnid[idx]
      if trntype == TrnType.TENDON:
        _union_tendon_trees(
          opt_enableflags,
          dof_treeid,
          ten_J_rownnz,
          ten_J_rowadr,
          ten_J_colind,
          tree_awake_in,
          worldid,
          trnid[0],
          tree_island_out,
          island_parent_out,
        )
      elif trntype == TrnType.SITE or trntype == TrnType.SLIDERCRANK:
        if trnid[1] >= 0:
          tree1 = body_treeid[site_bodyid[trnid[0]]]
          tree2 = body_treeid[site_bodyid[trnid[1]]]
          if tree1 >= 0 and tree2 >= 0 and tree1 != tree2:
            sleep_enabled = bool(opt_enableflags & EnableBit.SLEEP)
            if not sleep_enabled or (tree_awake_in[worldid, tree1] != 0 and tree_awake_in[worldid, tree2] != 0):
              _dsu_union(island_parent_out, worldid, tree1, tree2, tree_island_out)


@wp.kernel(module="unique", enable_backward=False)
def _island_dsu(
  # Model:
  nv: int,
  opt_enableflags: int,
  body_treeid: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_bodyid: wp.array[int],
  dof_treeid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_celladr: wp.array[int],
  flex_nodeadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_cellempty: wp.array[bool],
  flex_elem: wp.array[int],
  flex_vert0: wp.array[wp.vec3],
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_objtype: wp.array[int],
  is_sparse: bool,
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  tree_awake_in: wp.array2d[int],
  contact_pos_in: wp.array[wp.vec3],
  contact_geom_in: wp.array[wp.vec2i],
  contact_flex_in: wp.array[wp.vec2i],
  contact_elem_in: wp.array[wp.vec2i],
  contact_vert_in: wp.array[wp.vec2i],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  njmax_in: int,
  # In:
  chunk_size: int,
  # Data out:
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  """Process one chunk of one world's active EFC prefix with one strided CUDA warp.

  Blocks are laid out over (world, chunk) rather than world alone, so resident
  parallelism tracks total constraint capacity instead of batch size. A single world
  with a long prefix fills the device the same way a large batch of short ones does.
  """
  worldid, chunkid, lane = wp.tid()
  sleep_enabled = (opt_enableflags & types.EnableBit.SLEEP) != 0
  chunk_beg = chunkid * chunk_size
  chunk_end = wp.min(chunk_beg + chunk_size, wp.min(njmax_in, nefc_in[worldid]))
  for efcid in range(chunk_beg + lane, chunk_end, wp.block_dim()):
    efc_type = efc_type_in[worldid, efcid]
    efc_id = efc_id_in[worldid, efcid]
    repeated = _is_repeated_fixed_support_efc(efc_type_in, efc_id_in, worldid, efcid)
    tree0 = int(-1)
    tree1 = int(-1)
    use_generic = int(0)

    if efc_type == ConstraintType.EQUALITY:
      eq_t = eq_type[efc_id]
      if eq_t == EqType.CONNECT or eq_t == EqType.WELD:
        if repeated:
          continue
        body0 = eq_obj1id[efc_id]
        body1 = eq_obj2id[efc_id]
        if eq_objtype[efc_id] == ObjType.SITE:
          body0 = site_bodyid[body0]
          body1 = site_bodyid[body1]
        tree0 = body_treeid[body0]
        tree1 = body_treeid[body1]
      else:
        use_generic = 1
    elif efc_type == ConstraintType.FRICTION_DOF:
      if repeated:
        continue
      tree0 = dof_treeid[efc_id]
    elif efc_type == ConstraintType.LIMIT_JOINT:
      if repeated:
        continue
      tree0 = dof_treeid[jnt_dofadr[efc_id]]
    elif (
      efc_type == ConstraintType.CONTACT_FRICTIONLESS
      or efc_type == ConstraintType.CONTACT_PYRAMIDAL
      or efc_type == ConstraintType.CONTACT_ELLIPTIC
    ):
      if repeated:
        continue
      geom_pair = contact_geom_in[efc_id]
      if geom_pair[0] >= 0 and geom_pair[1] >= 0:
        tree0 = body_treeid[geom_bodyid[geom_pair[0]]]
        tree1 = body_treeid[geom_bodyid[geom_pair[1]]]
      else:
        # flex: union the trees of the bodies of both sides
        bodies0, bodies1 = _flex_contact_bodies(
          jnt_bodyid,
          geom_bodyid,
          flex_dim,
          flex_interp,
          flex_cellnum,
          flex_celladr,
          flex_nodeadr,
          flex_vertadr,
          flex_elemadr,
          flex_elemnum,
          flex_elemdataadr,
          flex_nodebodyid,
          flex_vertbodyid,
          flex_cellempty,
          flex_elem,
          flex_vert0,
          flex_rod_has_twist,
          flex_rod_edgeadr,
          flex_rod_edgenum,
          flex_rodedge_jnt,
          flex_elem_rodedge,
          flex_vert_rodelem,
          flexvert_xpos_in,
          contact_pos_in,
          contact_geom_in,
          contact_flex_in,
          contact_elem_in,
          contact_vert_in,
          worldid,
          efc_id,
        )
        first_tree = int(-1)
        for k in range(8):
          body = bodies0[k % 4]
          if k >= 4:
            body = bodies1[k % 4]
          if body < 0:
            continue
          tree = body_treeid[body]
          if tree < 0 or (sleep_enabled and tree_awake_in[worldid, tree] == 0):
            continue
          if first_tree < 0:
            first_tree = tree
            _dsu_union(island_parent_out, worldid, tree, -1, tree_island_out)
          elif tree != first_tree:
            _dsu_union(island_parent_out, worldid, first_tree, tree, tree_island_out)
        continue
    else:
      use_generic = 1

    if use_generic == 0:
      if sleep_enabled:
        if tree0 >= 0 and tree_awake_in[worldid, tree0] == 0:
          tree0 = -1
        if tree1 >= 0 and tree_awake_in[worldid, tree1] == 0:
          tree1 = -1
      _dsu_union(island_parent_out, worldid, tree0, tree1, tree_island_out)
      continue

    first_tree = int(-1)
    count = nv
    rowadr = int(0)
    if is_sparse:
      count = efc_J_rownnz_in[worldid, efcid]
      rowadr = efc_J_rowadr_in[worldid, efcid]

    for index in range(count):
      dof = index
      if is_sparse:
        dof = efc_J_colind_in[worldid, 0, rowadr + index]
      elif efc_J_in[worldid, efcid, dof] == 0.0:
        continue
      tree = dof_treeid[dof]
      if tree < 0 or (sleep_enabled and tree_awake_in[worldid, tree] == 0):
        continue
      if first_tree < 0:
        first_tree = tree
        _dsu_union(island_parent_out, worldid, tree, -1, tree_island_out)
      else:
        _dsu_union(island_parent_out, worldid, first_tree, tree, tree_island_out)


@wp.kernel
def _finalize_dsu(
  # Model:
  ntree: int,
  # In:
  island_parent_in: wp.array2d[int],
  # Data out:
  nisland_out: wp.array[int],
  tree_island_out: wp.array2d[int],
):
  """Number islands in ascending tree order and propagate labels in one pass.

  Because _dsu_union hooks higher roots to lower roots, parent(treeid) <= treeid. Therefore
  by the time the loop visits treeid, its parent has already been visited and assigned its
  final island label in tree_island_out[worldid, parent].
  """
  worldid = wp.tid()
  nisland = int(0)
  for treeid in range(ntree):
    if tree_island_out[worldid, treeid] >= 0:
      parent = island_parent_in[worldid, treeid]
      if parent == treeid:
        tree_island_out[worldid, treeid] = nisland
        nisland += 1
      else:
        tree_island_out[worldid, treeid] = tree_island_out[worldid, parent]
  nisland_out[worldid] = nisland


# Smallest EFC chunk worth its own block, and the resident-block count worth reaching for.
_DSU_MIN_CHUNK = 256
_DSU_TARGET_BLOCKS = 2048


@event_scope
def direct_dsu(m: types.Model, d: types.Data, island_parent: wp.array2d[int]):
  """Discover islands with EFC-parallel atomic minimum-root hooks.

  `island_parent` is the (nworld, ntree) disjoint-set workspace, owned by the caller.
  """
  # Discovery blocks are laid out over (world, chunk). A batch large enough to occupy the
  # device keeps one block per world, since extra chunks would mostly launch past the
  # active prefix; a small batch with a long prefix is split until there is resident work.
  max_chunks = max(1, -(-d.njmax // _DSU_MIN_CHUNK))
  nchunk = min(max(1, -(-_DSU_TARGET_BLOCKS // d.nworld)), max_chunks)
  chunk_size = -(-d.njmax // nchunk)
  wp.launch(
    _reset_dsu,
    dim=(d.nworld, m.ntree),
    inputs=[d.nisland, d.tree_island, island_parent],
  )
  if m.nflex > 0:
    has_passive = not ((m.opt.disableflags & types.DisableBit.SPRING) and (m.opt.disableflags & types.DisableBit.DAMPER))
    sleep_metric_flex = (
      bool(m.opt.enableflags & types.EnableBit.SLEEP) and m.opt.integrator == types.IntegratorType.DISCRETE and has_passive
    )
    wp.launch(
      _flex_island_dsu,
      dim=(d.nworld, m.nflex),
      inputs=[
        m.opt.enableflags,
        m.body_treeid,
        m.jnt_bodyid,
        m.flex_interp,
        m.flex_nodeadr,
        m.flex_nodenum,
        m.flex_vertadr,
        m.flex_vertnum,
        m.flex_nodebodyid,
        m.flex_vertbodyid,
        m.flex_rod_has_twist,
        m.flex_eff_possible,
        m.flex_rod_edgeadr,
        m.flex_rod_edgenum,
        m.flex_rodedge_jnt,
        d.nefc,
        d.tree_awake,
        sleep_metric_flex,
      ],
      outputs=[d.tree_island, island_parent],
    )
    if m.opt.integrator == types.IntegratorType.DISCRETE and passive.flex_passive_contact(m):
      wp.launch(
        _passive_contact_island_dsu,
        dim=d.naconmax,
        inputs=[
          m.opt.enableflags,
          m.body_treeid,
          m.jnt_bodyid,
          m.geom_bodyid,
          m.flex_dim,
          m.flex_interp,
          m.flex_cellnum,
          m.flex_celladr,
          m.flex_nodeadr,
          m.flex_vertadr,
          m.flex_elemadr,
          m.flex_elemnum,
          m.flex_elemdataadr,
          m.flex_nodebodyid,
          m.flex_vertbodyid,
          m.flex_cellempty,
          m.flex_elem,
          m.flex_vert0,
          m.flex_rod_has_twist,
          m.flex_rod_edgeadr,
          m.flex_rod_edgenum,
          m.flex_rodedge_jnt,
          m.flex_elem_rodedge,
          m.flex_vert_rodelem,
          d.nefc,
          d.flexvert_xpos,
          d.tree_awake,
          d.contact.pos,
          d.contact.geom,
          d.contact.flex,
          d.contact.elem,
          d.contact.vert,
          d.contact.worldid,
          d.contact.type,
          d.nacon,
          sleep_metric_flex,
        ],
        outputs=[d.tree_island, island_parent],
      )

  wp.launch_tiled(
    _island_dsu,
    dim=(d.nworld, nchunk),
    inputs=[
      m.nv,
      m.opt.enableflags,
      m.body_treeid,
      m.jnt_dofadr,
      m.jnt_bodyid,
      m.dof_treeid,
      m.geom_bodyid,
      m.site_bodyid,
      m.flex_dim,
      m.flex_interp,
      m.flex_cellnum,
      m.flex_celladr,
      m.flex_nodeadr,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemnum,
      m.flex_elemdataadr,
      m.flex_nodebodyid,
      m.flex_vertbodyid,
      m.flex_cellempty,
      m.flex_elem,
      m.flex_vert0,
      m.eq_type,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.is_sparse,
      m.flex_rod_has_twist,
      m.flex_rod_edgeadr,
      m.flex_rod_edgenum,
      m.flex_rodedge_jnt,
      m.flex_elem_rodedge,
      m.flex_vert_rodelem,
      d.nefc,
      d.flexvert_xpos,
      d.tree_awake,
      d.contact.pos,
      d.contact.geom,
      d.contact.flex,
      d.contact.elem,
      d.contact.vert,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.njmax,
      chunk_size,
      d.tree_island,
      island_parent,
    ],
    block_dim=types.BlockDim.island_dsu,
  )
  if m.opt.integrator == IntegratorType.DISCRETE and (m.has_tendon_stiffness or m.has_tendon_damping or m.has_efm_actuator):
    wp.launch(
      _island_dsu_metric_couplings,
      dim=(d.nworld, max(m.ntendon, m.nactuator)),
      inputs=[
        m.nactuator,
        m.ntendon,
        m.opt.disableflags,
        m.opt.enableflags,
        m.body_treeid,
        m.dof_treeid,
        m.site_bodyid,
        m.tendon_actuatorid,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.tendon_stiffness,
        m.tendon_stiffnesspoly,
        m.tendon_damping,
        m.tendon_dampingpoly,
        m.actuator_trntype,
        m.actuator_gaintype,
        m.actuator_biastype,
        m.actuator_trnid,
        d.nefc,
        d.tree_awake,
      ],
      outputs=[d.tree_island, island_parent],
    )
  wp.launch(
    _finalize_dsu,
    dim=d.nworld,
    inputs=[m.ntree, island_parent],
    outputs=[d.nisland, d.tree_island],
  )


@wp.kernel
def _zero_island_counts(
  # Data out:
  nisland_out: wp.array[int],
  nidof_out: wp.array[int],
):
  worldid = wp.tid()
  nisland_out[worldid] = 0
  nidof_out[worldid] = 0


@event_scope
def island(m: types.Model, d: types.Data):
  """Discover constraint islands."""
  if rows_in_island_order(d):
    # make_constraint found them from the constraints' trees (dense rows have no sparse entries)
    return

  if m.ntree == 0:
    wp.launch(
      _zero_island_counts,
      dim=d.nworld,
      outputs=[d.nisland, d.nidof],
    )
    return

  direct_dsu(m, d, wp.empty((d.nworld, m.ntree), dtype=int))


@wp.kernel
def _island_count_dofs(
  # Model:
  dof_treeid: wp.array[int],
  # Data in:
  tree_island_in: wp.array2d[int],
  # Data out:
  dof_island_out: wp.array2d[int],
  island_nv_out: wp.array2d[int],
):
  worldid, dofid = wp.tid()

  island_id = tree_island_in[worldid, dof_treeid[dofid]]
  dof_island_out[worldid, dofid] = island_id
  if island_id >= 0:
    wp.atomic_add(island_nv_out, worldid, island_id, 1)  # kernel_analyzer: ignore[atomic]


# Islands per tile of the island size scans: any number of trees is scanned in chunks.
_SCAN_CHUNK = 1024


@cache_kernel
def _island_scan_sizes(chunk: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Data in:
    island_nv_in: wp.array2d[int],
    island_nefc_in: wp.array2d[int],
    # Data out:
    nidof_out: wp.array[int],
    island_idofadr_out: wp.array2d[int],
    island_iefcadr_out: wp.array2d[int],
  ):
    """Block-wide exclusive scans of island sizes (entries past nisland are zero)."""
    worldid = wp.tid()
    nv_adr = int(0)
    nefc_adr = int(0)
    for base in range(0, island_nv_in.shape[1], chunk):
      nv_tile = wp.tile_load(island_nv_in[worldid], shape=chunk, offset=base)
      nefc_tile = wp.tile_load(island_nefc_in[worldid], shape=chunk, offset=base)
      nv_scan = wp.tile_map(wp.add, wp.tile_scan_exclusive(nv_tile), wp.tile_full(chunk, nv_adr, dtype=int))
      nefc_scan = wp.tile_map(wp.add, wp.tile_scan_exclusive(nefc_tile), wp.tile_full(chunk, nefc_adr, dtype=int))
      wp.tile_store(island_idofadr_out[worldid], nv_scan, offset=base)
      wp.tile_store(island_iefcadr_out[worldid], nefc_scan, offset=base)
      nv_adr += wp.tile_sum(nv_tile)[0]
      nefc_adr += wp.tile_sum(nefc_tile)[0]
    nidof_out[worldid] = nv_adr

  return kernel


@wp.kernel
def _island_map_dofs(
  # Model:
  dof_treeid: wp.array[int],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  # Data in:
  nidof_in: wp.array[int],
  tree_island_in: wp.array2d[int],
  island_idofadr_in: wp.array2d[int],
  # Data out:
  island_dofadr_out: wp.array2d[int],
  map_dof2idof_out: wp.array2d[int],
  map_idof2dof_out: wp.array2d[int],
  # Out:
  idof_islandid_out: wp.array2d[int],
):
  """Map DOFs in ascending global order for deterministic solver inputs.

  A DOF's rank inside its island is fixed by the tree layout: DOFs are contiguous per
  tree and trees are ordered, so the rank is the DOF count of every preceding tree
  carrying the same label. That closed form makes the ascending order reproducible
  without a serial per-world scan.
  """
  worldid, dofid = wp.tid()

  tree = dof_treeid[dofid]
  island_id = tree_island_in[worldid, tree]

  local_idx = dofid - tree_dofadr[tree]
  for t in range(tree):
    if tree_island_in[worldid, t] == island_id:
      local_idx += tree_dofnum[t]

  if island_id >= 0:
    idof = island_idofadr_in[worldid, island_id] + local_idx
    idof_islandid_out[worldid, idof] = island_id
    if local_idx == 0:
      island_dofadr_out[worldid, island_id] = dofid
  else:
    idof = nidof_in[worldid] + local_idx

  map_dof2idof_out[worldid, dofid] = idof
  map_idof2dof_out[worldid, idof] = dofid


@wp.kernel
def _island_chunk_offsets(
  # Data in:
  nefc_in: wp.array[int],
  nisland_in: wp.array[int],
  njmax_in: int,
  # In:
  chunk_size: int,
  # Out:
  chunk_adr_out: wp.array3d[int],
):
  """Turn per-chunk row counts of an island into its rows' offsets from island_iefcadr."""
  worldid, islandid = wp.tid()
  if islandid >= nisland_in[worldid]:
    return

  nchunk = (wp.min(njmax_in, nefc_in[worldid]) + chunk_size - 1) // chunk_size
  adr = int(0)
  for chunk in range(nchunk):
    count = chunk_adr_out[worldid, chunk, islandid]
    chunk_adr_out[worldid, chunk, islandid] = adr
    adr += count


@wp.kernel
def _island_map_constraints(
  # Data in:
  nefc_in: wp.array[int],
  efc_island_in: wp.array2d[int],
  island_iefcadr_in: wp.array2d[int],
  njmax_in: int,
  # In:
  chunk_size: int,
  chunk_adr_in: wp.array3d[int],
  # Data out:
  map_efc2iefc_out: wp.array2d[int],
  map_iefc2efc_out: wp.array2d[int],
  # Out:
  iefc_islandid_out: wp.array2d[int],
):
  """Stable counting sort of rows by island: each island keeps MuJoCo's row layout."""
  worldid, efcid = wp.tid()
  if efcid >= wp.min(njmax_in, nefc_in[worldid]):
    return

  islandid = efc_island_in[worldid, efcid]
  if islandid < 0:
    return

  chunk = efcid // chunk_size
  rank = int(0)
  for prev in range(chunk * chunk_size, efcid):
    if efc_island_in[worldid, prev] == islandid:
      rank += 1

  ic = island_iefcadr_in[worldid, islandid] + chunk_adr_in[worldid, chunk, islandid] + rank
  map_efc2iefc_out[worldid, efcid] = ic
  map_iefc2efc_out[worldid, ic] = efcid
  iefc_islandid_out[worldid, ic] = islandid


@wp.kernel
def _island_count_constraints(
  # Model:
  nv: int,
  body_treeid: wp.array[int],
  jnt_dofadr: wp.array[int],
  jnt_bodyid: wp.array[int],
  dof_treeid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_celladr: wp.array[int],
  flex_nodeadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_cellempty: wp.array[bool],
  flex_elem: wp.array[int],
  flex_vert0: wp.array[wp.vec3],
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_objtype: wp.array[int],
  is_sparse: bool,
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  flexvert_xpos_in: wp.array2d[wp.vec3],
  contact_pos_in: wp.array[wp.vec3],
  contact_geom_in: wp.array[wp.vec2i],
  contact_flex_in: wp.array[wp.vec2i],
  contact_elem_in: wp.array[wp.vec2i],
  contact_vert_in: wp.array[wp.vec2i],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  tree_island_in: wp.array2d[int],
  njmax_in: int,
  # In:
  chunk_size: int,
  # Data out:
  efc_island_out: wp.array2d[int],
  island_nefc_out: wp.array2d[int],
  island_ne_out: wp.array2d[int],
  island_nf_out: wp.array2d[int],
  # Out:
  chunk_adr_out: wp.array3d[int],
):
  """Compute the first non-negative tree for each constraint; count rows per island and chunk."""
  worldid, efcid = wp.tid()

  if efcid >= wp.min(njmax_in, nefc_in[worldid]):
    return

  efc_type = efc_type_in[worldid, efcid]
  efc_id = efc_id_in[worldid, efcid]

  tree = int(-1)
  use_generic = int(0)

  # equality (connect/weld)
  if efc_type == ConstraintType.EQUALITY:
    eq_t = eq_type[efc_id]

    if eq_t == EqType.CONNECT or eq_t == EqType.WELD:
      b1 = eq_obj1id[efc_id]
      b2 = eq_obj2id[efc_id]

      if eq_objtype[efc_id] == ObjType.SITE:
        b1 = site_bodyid[b1]
        b2 = site_bodyid[b2]

      t1 = body_treeid[b1]
      t2 = body_treeid[b2]
      if t1 >= 0:
        tree = t1
      else:
        tree = t2
    else:
      # JOINT, TENDON, FLEX: generic scan
      use_generic = 1

  # joint friction
  elif efc_type == ConstraintType.FRICTION_DOF:
    tree = dof_treeid[efc_id]

  # joint limit
  elif efc_type == ConstraintType.LIMIT_JOINT:
    tree = dof_treeid[jnt_dofadr[efc_id]]

  # contact
  elif (
    efc_type == ConstraintType.CONTACT_FRICTIONLESS
    or efc_type == ConstraintType.CONTACT_PYRAMIDAL
    or efc_type == ConstraintType.CONTACT_ELLIPTIC
  ):
    geom_pair = contact_geom_in[efc_id]
    g1 = geom_pair[0]
    g2 = geom_pair[1]

    if g1 >= 0 and g2 >= 0:
      t1 = body_treeid[geom_bodyid[g1]]
      t2 = body_treeid[geom_bodyid[g2]]
      if t1 >= 0:
        tree = t1
      else:
        tree = t2
    else:
      # flex: first tree of the bodies of both sides
      bodies0, bodies1 = _flex_contact_bodies(
        jnt_bodyid,
        geom_bodyid,
        flex_dim,
        flex_interp,
        flex_cellnum,
        flex_celladr,
        flex_nodeadr,
        flex_vertadr,
        flex_elemadr,
        flex_elemnum,
        flex_elemdataadr,
        flex_nodebodyid,
        flex_vertbodyid,
        flex_cellempty,
        flex_elem,
        flex_vert0,
        flex_rod_has_twist,
        flex_rod_edgeadr,
        flex_rod_edgenum,
        flex_rodedge_jnt,
        flex_elem_rodedge,
        flex_vert_rodelem,
        flexvert_xpos_in,
        contact_pos_in,
        contact_geom_in,
        contact_flex_in,
        contact_elem_in,
        contact_vert_in,
        worldid,
        efc_id,
      )
      for k in range(8):
        body = bodies0[k % 4]
        if k >= 4:
          body = bodies1[k % 4]
        if body >= 0:
          t = body_treeid[body]
          if t >= 0:
            tree = t
            break

  else:
    # generic: scan Jacobian row
    use_generic = 1

  if use_generic:
    count = nv
    rowadr = 0
    if is_sparse:
      count = efc_J_rownnz_in[worldid, efcid]
      rowadr = efc_J_rowadr_in[worldid, efcid]

    for i in range(count):
      dof = i
      if is_sparse:
        sparseid = rowadr + i
        dof = efc_J_colind_in[worldid, 0, sparseid]
      else:
        J_val = efc_J_in[worldid, efcid, dof]
        if J_val == 0.0:
          continue

      t = dof_treeid[dof]
      if t >= 0:
        tree = t
        break

  if tree < 0:
    efc_island_out[worldid, efcid] = -1
    return
  island_id = tree_island_in[worldid, tree]
  efc_island_out[worldid, efcid] = island_id

  if island_id >= 0:
    wp.atomic_add(island_nefc_out, worldid, island_id, 1)  # kernel_analyzer: ignore[atomic]
    wp.atomic_add(chunk_adr_out, worldid, efcid // chunk_size, island_id, 1)  # kernel_analyzer: ignore[atomic]

    if efc_type == ConstraintType.EQUALITY:
      wp.atomic_add(island_ne_out, worldid, island_id, 1)  # kernel_analyzer: ignore[atomic]
    elif efc_type == ConstraintType.FRICTION_DOF or efc_type == ConstraintType.FRICTION_TENDON:
      wp.atomic_add(island_nf_out, worldid, island_id, 1)  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _init_island_arrays(
  # Model:
  nv: int,
  # Data out:
  nidof_out: wp.array[int],
  island_dofadr_out: wp.array2d[int],
  island_idofadr_out: wp.array2d[int],
  island_nv_out: wp.array2d[int],
  island_nefc_out: wp.array2d[int],
  island_ne_out: wp.array2d[int],
  island_nf_out: wp.array2d[int],
  island_iefcadr_out: wp.array2d[int],
):
  worldid, islandid = wp.tid()
  island_dofadr_out[worldid, islandid] = nv
  island_nv_out[worldid, islandid] = 0
  island_nefc_out[worldid, islandid] = 0
  island_ne_out[worldid, islandid] = 0
  island_nf_out[worldid, islandid] = 0
  island_idofadr_out[worldid, islandid] = 0
  island_iefcadr_out[worldid, islandid] = 0
  if islandid == 0:
    nidof_out[worldid] = 0


@wp.kernel
def _init_dof_arrays(
  # Data out:
  dof_island_out: wp.array2d[int],
  map_dof2idof_out: wp.array2d[int],
  map_idof2dof_out: wp.array2d[int],
  # Out:
  idof_islandid_out: wp.array2d[int],
):
  worldid, dofid = wp.tid()
  dof_island_out[worldid, dofid] = -1
  map_dof2idof_out[worldid, dofid] = 0
  map_idof2dof_out[worldid, dofid] = 0
  idof_islandid_out[worldid, dofid] = -1


@wp.kernel
def _init_efc_arrays(
  # Data out:
  efc_island_out: wp.array2d[int],
  map_efc2iefc_out: wp.array2d[int],
  map_iefc2efc_out: wp.array2d[int],
  # Out:
  iefc_islandid_out: wp.array2d[int],
):
  worldid, efcid = wp.tid()
  efc_island_out[worldid, efcid] = -1
  map_efc2iefc_out[worldid, efcid] = 0
  map_iefc2efc_out[worldid, efcid] = 0
  iefc_islandid_out[worldid, efcid] = -1


# Constraint rows per chunk of the island counting sort: a row scans at most this many predecessors.
_MAP_CHUNK = 64


@event_scope
def compute_island_mapping(m: types.Model, d: types.Data):
  """Compute DOF/constraint island mappings after island discovery.

  Populates d.dof_island, d.efc.island, d.island_idofadr, d.island_dofadr,
  d.island_nv, d.island_nefc, d.island_ne, d.island_nf, d.island_iefcadr,
  d.nidof, d.map_dof2idof, d.map_idof2dof, d.dof_islandid, d.map_efc2iefc,
  d.map_iefc2efc, d.efc_islandid.

  Args:
    m: Model.
    d: Data.
  """
  # Ensure dof_islandid / efc_islandid are allocated at the right shape
  if d.dof_islandid.shape[1] != m.nv:
    d.dof_islandid = wp.empty((d.nworld, m.nv), dtype=int)
  if d.efc_islandid.shape[1] != d.njmax:
    d.efc_islandid = wp.empty((d.nworld, d.njmax), dtype=int)
  if d.island_idofadr.shape[1] != m.ntree:
    d.island_idofadr = wp.empty((d.nworld, m.ntree), dtype=int)

  wp.launch(
    _init_island_arrays,
    dim=(d.nworld, m.ntree),
    inputs=[m.nv],
    outputs=[
      d.nidof,
      d.island_dofadr,
      d.island_idofadr,
      d.island_nv,
      d.island_nefc,
      d.island_ne,
      d.island_nf,
      d.island_iefcadr,
    ],
  )
  wp.launch(
    _init_dof_arrays,
    dim=(d.nworld, m.nv),
    inputs=[],
    outputs=[d.dof_island, d.map_dof2idof, d.map_idof2dof, d.dof_islandid],
  )
  wp.launch(
    _init_efc_arrays,
    dim=(d.nworld, d.njmax),
    inputs=[],
    outputs=[d.efc.island, d.map_efc2iefc, d.map_iefc2efc, d.efc_islandid],
  )

  # Count DOFs per island
  wp.launch(
    _island_count_dofs,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_treeid, d.tree_island],
    outputs=[d.dof_island, d.island_nv],
  )

  # Count Constraints per island, and per (chunk, island) for the counting sort
  nchunk = max(1, -(-d.njmax // _MAP_CHUNK))
  chunk_adr = wp.zeros((d.nworld, nchunk, m.ntree), dtype=int)
  wp.launch(
    _island_count_constraints,
    dim=(d.nworld, d.njmax),
    inputs=[
      m.nv,
      m.body_treeid,
      m.jnt_dofadr,
      m.jnt_bodyid,
      m.dof_treeid,
      m.geom_bodyid,
      m.site_bodyid,
      m.flex_dim,
      m.flex_interp,
      m.flex_cellnum,
      m.flex_celladr,
      m.flex_nodeadr,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemnum,
      m.flex_elemdataadr,
      m.flex_nodebodyid,
      m.flex_vertbodyid,
      m.flex_cellempty,
      m.flex_elem,
      m.flex_vert0,
      m.eq_type,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.is_sparse,
      m.flex_rod_has_twist,
      m.flex_rod_edgeadr,
      m.flex_rod_edgenum,
      m.flex_rodedge_jnt,
      m.flex_elem_rodedge,
      m.flex_vert_rodelem,
      d.nefc,
      d.flexvert_xpos,
      d.contact.pos,
      d.contact.geom,
      d.contact.flex,
      d.contact.elem,
      d.contact.vert,
      d.efc.type,
      d.efc.id,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.tree_island,
      d.njmax,
      _MAP_CHUNK,
    ],
    outputs=[d.efc.island, d.island_nefc, d.island_ne, d.island_nf, chunk_adr],
  )

  # 3. Scan sizes
  wp.launch_tiled(
    _island_scan_sizes(min(m.ntree, _SCAN_CHUNK)),
    dim=d.nworld,
    inputs=[d.island_nv, d.island_nefc],
    outputs=[d.nidof, d.island_idofadr, d.island_iefcadr],
    block_dim=types.BlockDim.island_scan_sizes,
  )

  # 4. Map DOFs
  wp.launch(
    _island_map_dofs,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_treeid, m.tree_dofadr, m.tree_dofnum, d.nidof, d.tree_island, d.island_idofadr],
    outputs=[d.island_dofadr, d.map_dof2idof, d.map_idof2dof, d.dof_islandid],
  )

  # 5. Map Constraints
  wp.launch(
    _island_chunk_offsets,
    dim=(d.nworld, m.ntree),
    inputs=[d.nefc, d.nisland, d.njmax, _MAP_CHUNK],
    outputs=[chunk_adr],
  )
  wp.launch(
    _island_map_constraints,
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, d.efc.island, d.island_iefcadr, d.njmax, _MAP_CHUNK, chunk_adr],
    outputs=[d.map_efc2iefc, d.map_iefc2efc, d.efc_islandid],
  )


# make_constraint's row kernels write rows (ROWS_WRITE): in class order per world, or with the
# island layout (opt.island_solve) at their island's address, an island's rows contiguous and in
# class order as in MuJoCo's island arrays. The layout comes from a first pass (ROWS_DISCOVER) that
# merges each active constraint's trees and counts its rows per tree and class.
ROWS_WRITE = 0
ROWS_DISCOVER = 1
ROW_CLASSES = 4
ROW_EQUALITY = 0
ROW_FRICTION = 1
ROW_LIMIT = 2
ROW_CONTACT = 3

# Groups of up to dense_width(m) DOFs (a humanoid, a few resting boxes) write their Jacobian rows
# only to GroupBuffers.J, dense and in group-local columns; their sparse rows are empty.
# DENSE_NV < 32 turns dense rows off.
DENSE_NV = 32
# Groups are runs of consecutive islands of up to MERGE_NV DOFs (e.g. resting boxes); their rows
# are not in class order, so the island solver classifies rows by type.
MERGE = True
MERGE_NV = 32
# models of up to MIN_NV DOFs are one group, the world solver's problem
MIN_NV = MERGE_NV
# models of more trees are the world solver's: _island_world's shared memory grows with ntree
MAX_NTREE = 1024


@wp.func
def rows_dense(
  # Data in:
  tree_island_in: wp.array2d[int],
  # In:
  worldid: int,
  tree: int,
  efcid: int,
  island_dense_in: wp.array2d[wp.vec4i],
) -> int:
  """A constraint's first dense row: efcid in a group with dense rows, else -1 (sparse rows)."""
  if tree < 0:
    return -1
  if island_dense_in[worldid, tree_island_in[worldid, tree]][0] == 0:
    return -1
  return efcid


@wp.func
def dense_zero(
  # In:
  worldid: int,
  srow: int,
  nrow: int,
  # Out:
  dense_J_out: wp.array3d[float],
):
  if srow >= 0:
    # whole rows, past narrower groups' columns too: rows with gaps are expensive to write
    for r in range(nrow):
      for c in range(dense_J_out.shape[2]):
        dense_J_out[worldid, srow + r, c] = 0.0


@wp.func
def dense_set(
  # In:
  worldid: int,
  srow: int,
  k: int,
  dofid: int,
  value: float,
  dof_densecol_in: wp.array2d[int],
  # Out:
  dense_J_out: wp.array3d[float],
):
  """Value of dense row srow + k at a DOF's column (srow < 0: sparse)."""
  if srow >= 0:
    dense_J_out[worldid, srow + k, dof_densecol_in[worldid, dofid]] = value


@wp.func
def _dense_width(ndof: int, wide: int) -> int:
  return wp.where(ndof <= wide, wide, 0)


class GroupBuffers:
  """Island groups and their dense rows, shared by make_constraint and the island solve."""

  def __init__(self, m: types.Model, d: types.Data):
    self.key = (d.nworld, m.ntree, m.nv, d.njmax, dense_width(m))
    shape = (d.nworld, m.ntree)
    self.ngroup = wp.empty(d.nworld, dtype=int)
    self.island_group = wp.empty(shape, dtype=int)
    self.group_idofadr = wp.empty(shape, dtype=int)
    self.group_nv = wp.empty(shape, dtype=int)
    self.group_nefc = wp.empty(shape, dtype=int)
    self.group_iefcadr = wp.empty(shape, dtype=int)
    self.group_ne = wp.empty(shape, dtype=int)
    self.group_nf = wp.empty(shape, dtype=int)
    self.group_island0 = wp.empty(shape, dtype=int)
    self.group_nisland = wp.empty(shape, dtype=int)
    self.group_dense = wp.empty(shape, dtype=wp.vec4i)
    self.group_asleep = wp.empty(shape, dtype=bool)
    self.island_dense = wp.empty(shape, dtype=wp.vec4i)
    self.dof_group = wp.empty((d.nworld, m.nv), dtype=int)
    self.dof_densecol = wp.empty((d.nworld, m.nv), dtype=int)
    self.efc_group = wp.empty((d.nworld, d.njmax), dtype=int)
    # per world scratch of _island_world
    self.tree_iofs = wp.empty(shape, dtype=int)
    # make_constraint's discovery and island layout (allocated once: in a graph capture,
    # allocations and frees are nodes)
    self.island_parent = wp.empty(shape, dtype=int)
    self.tree_nrow = wp.empty((d.nworld, m.ntree * ROW_CLASSES), dtype=int)
    self.island_rowadr = wp.empty((d.nworld, m.ntree * ROW_CLASSES), dtype=int)
    self.island_cursor = wp.empty((d.nworld, m.ntree * ROW_CLASSES), dtype=int)
    self.tree_group = wp.empty(shape, dtype=int)
    self.island_root = wp.empty(shape, dtype=int)
    self.island_acc = wp.empty(shape, dtype=int)
    # dense rows by world row, plus a tile of rows tile kernels read past the last; not zeroed (in a
    # graph capture a fill would run every step)
    self.J = wp.empty((d.nworld, d.njmax + types.TILE_SIZE_JTDAJ_DENSE, dense_width(m)), dtype=float)


def dense_width(m: types.Model) -> int:
  """Width of the dense rows: the model's largest tree of 17 to 32 DOFs, or 16 (see DENSE_NV)."""
  return m.island_tree_nv


def rows_in_island_order(d: types.Data) -> bool:
  """Whether make_constraint last wrote d's rows in island order (put_data loads MuJoCo's)."""
  return d.__dict__.get("_island_rows", False)


def group_buffers(m: types.Model, d: types.Data) -> GroupBuffers:
  buf = d.__dict__.get("_island_groups")
  if buf is None or buf.key != (d.nworld, m.ntree, m.nv, d.njmax, dense_width(m)):
    buf = GroupBuffers(m, d)
    d.__dict__["_island_groups"] = buf
  return buf


@wp.func_native(snippet="WP_TILE_SYNC();")
def _block_sync(): ...


# a shared tile's element, without the barrier that tile element assignment ends with
@wp.func_native(snippet="t.data(wp::tile_coord(i)) = v;")
def _tile_set(t: Any, i: int, v: int): ...


@wp.kernel
def _island_reset(
  # Data out:
  nisland_out: wp.array[int],
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
  tree_nrow_out: wp.array2d[int],
):
  """Union-find and row counts before the constraints discover the islands."""
  worldid, treeid = wp.tid()
  island_parent_out[worldid, treeid] = treeid
  tree_island_out[worldid, treeid] = -1
  for c in range(ROW_CLASSES):
    tree_nrow_out[worldid, treeid * ROW_CLASSES + c] = 0
  if treeid == 0:
    nisland_out[worldid] = 0


@wp.func
def _is_root(active: int, parent: int, tree: int) -> int:
  return wp.where(active >= 0 and parent == tree, 1, 0)


@wp.func
def _island_rows_fit(neq: int, nfrc: int, nrow: int, rowadr: int, rowmax: int) -> wp.vec3i:
  """An island's (equality, friction, all) rows below rowmax: nefc overflow drops the rest."""
  nrow = wp.min(nrow, wp.max(rowmax - rowadr, 0))
  neq = wp.min(neq, nrow)
  return wp.vec3i(neq, wp.min(nfrc, nrow - neq), nrow)


@cache_kernel
def _island_world(ntree: int):
  NT = ntree * ROW_CLASSES

  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nv: int,
    tree_dofnum: wp.array[int],
    # Data in:
    tree_awake_in: wp.array2d[int],
    njmax_in: int,
    # In:
    tree_nrow_in: wp.array2d[int],
    merge: bool,
    merge_nv: int,
    wide: int,
    free_groups: bool,
    # Data out:
    nisland_out: wp.array[int],
    nidof_out: wp.array[int],
    tree_island_out: wp.array2d[int],
    island_dofadr_out: wp.array2d[int],
    island_idofadr_out: wp.array2d[int],
    island_nv_out: wp.array2d[int],
    island_nefc_out: wp.array2d[int],
    island_ne_out: wp.array2d[int],
    island_nf_out: wp.array2d[int],
    island_iefcadr_out: wp.array2d[int],
    # Out:
    island_parent_out: wp.array2d[int],
    island_rowadr_out: wp.array2d[int],
    island_cursor_out: wp.array2d[int],
    tree_iofs_out: wp.array2d[int],
    tree_group_out: wp.array2d[int],
    island_root_out: wp.array2d[int],
    island_acc_out: wp.array2d[int],
    ngroup_out: wp.array[int],
    island_group_out: wp.array2d[int],
    group_idofadr_out: wp.array2d[int],
    group_nv_out: wp.array2d[int],
    group_nefc_out: wp.array2d[int],
    group_iefcadr_out: wp.array2d[int],
    group_ne_out: wp.array2d[int],
    group_nf_out: wp.array2d[int],
    group_island0_out: wp.array2d[int],
    group_nisland_out: wp.array2d[int],
    group_dense_out: wp.array2d[wp.vec4i],
    island_dense_out: wp.array2d[wp.vec4i],
    group_asleep_out: wp.array2d[bool],
  ):
    """A world's islands, rows, DOFs and groups (free_groups: also of the trees outside islands)."""
    worldid, lane = wp.tid()
    lanes = wp.block_dim()

    # roots: point every active tree at its component's minimum tree
    for t in range(lane, ntree, lanes):
      if tree_island_out[worldid, t] >= 0:
        island_parent_out[worldid, t] = _dsu_find(island_parent_out, worldid, t)
    _block_sync()

    # labels: roots numbered in tree order
    roots = wp.tile_map(
      _is_root,
      wp.tile_load(tree_island_out[worldid], shape=ntree),
      wp.tile_load(island_parent_out[worldid], shape=ntree),
      wp.tile_arange(ntree, dtype=int),
    )
    wp.tile_store(island_acc_out[worldid], wp.tile_scan_exclusive(roots))
    nisland = wp.tile_sum(roots)[0]
    _block_sync()
    for t in range(lane, ntree, lanes):
      if tree_island_out[worldid, t] >= 0:
        root = island_parent_out[worldid, t]
        label = island_acc_out[worldid, root]
        tree_island_out[worldid, t] = label
        if root == t:
          island_root_out[worldid, label] = t
    # nefc overflow: one island, whose rows in class order lose the last contacts past njmax as the
    # world solver's do (cutting at an island boundary drops whole islands, limits included)
    if wp.tile_sum(wp.tile_load(tree_nrow_in[worldid], shape=NT))[0] > njmax_in and nisland > 1:
      nisland = 1
      for t in range(lane, ntree, lanes):
        if tree_island_out[worldid, t] >= 0:
          tree_island_out[worldid, t] = 0
    if lane == 0:
      nisland_out[worldid] = nisland
    _block_sync()

    # rows per island and class, and DOFs per island (every lane takes part in each scatter)
    counts = wp.tile_zeros(shape=NT, dtype=int, storage="shared")
    for it in range((NT + lanes - 1) // lanes):
      e = it * lanes + lane
      valid = e < NT
      ec = wp.where(valid, e, 0)
      islandid = tree_island_out[worldid, ec // ROW_CLASSES]
      valid = valid and islandid >= 0
      pos = wp.where(valid, islandid * ROW_CLASSES + ec % ROW_CLASSES, 0)
      wp.tile_scatter_add(counts, pos, wp.where(valid, tree_nrow_in[worldid, ec], 0), valid)
    ndof = wp.tile_zeros(shape=ntree, dtype=int, storage="shared")
    for it in range((ntree + lanes - 1) // lanes):
      t = it * lanes + lane
      valid = t < ntree
      tc = wp.where(valid, t, 0)
      islandid = tree_island_out[worldid, tc]
      valid = valid and islandid >= 0
      wp.tile_scatter_add(ndof, wp.where(valid, islandid, 0), wp.where(valid, tree_dofnum[tc], 0), valid)
    wp.tile_store(island_rowadr_out[worldid], wp.tile_scan_exclusive(counts))
    wp.tile_store(island_cursor_out[worldid], wp.tile_zeros(shape=NT, dtype=int))
    wp.tile_store(island_nv_out[worldid], ndof)
    wp.tile_store(island_idofadr_out[worldid], wp.tile_scan_exclusive(ndof))
    nidof = wp.tile_sum(ndof)[0]
    _block_sync()
    for i in range(lane, ntree, lanes):
      ne = counts[i * ROW_CLASSES + ROW_EQUALITY]
      nf = counts[i * ROW_CLASSES + ROW_FRICTION]
      nefc = ne + nf + counts[i * ROW_CLASSES + ROW_LIMIT] + counts[i * ROW_CLASSES + ROW_CONTACT]
      iefcadr = island_rowadr_out[worldid, i * ROW_CLASSES]
      fit = _island_rows_fit(ne, nf, nefc, iefcadr, njmax_in)
      island_ne_out[worldid, i] = fit[0]
      island_nf_out[worldid, i] = fit[1]
      island_nefc_out[worldid, i] = fit[2]
      island_iefcadr_out[worldid, i] = iefcadr
      # the first DOF of each island is set per DOF (_island_dof_maps)
      island_dofadr_out[worldid, i] = nv
    if lane == 0:
      nidof_out[worldid] = nidof
    _block_sync()

    # the serial passes read and write shared memory and registers: global writes followed by reads
    # would each wait for a round trip
    tisland = wp.tile_load(tree_island_out[worldid], shape=ntree, storage="shared")
    tdofnum = wp.tile_load(tree_dofnum, shape=ntree, storage="shared")
    acc = wp.tile_zeros(shape=ntree, dtype=int, storage="shared")
    if lane == 0:
      # each tree's DOF offset in its island (DOFs in tree order), or among the free trees
      free = int(0)
      for t in range(ntree):
        islandid = tisland[t]
        n = tdofnum[t]
        if islandid >= 0:
          a = acc[islandid]
          tree_iofs_out[worldid, t] = a
          _tile_set(acc, islandid, a + n)
        else:
          tree_iofs_out[worldid, t] = free
          tree_group_out[worldid, t] = -1
          free += n

      # groups: runs of consecutive islands, then free trees, up to merge_nv DOFs that sleep
      # together (an island's trees sleep together: constraints to sleeping trees wake them);
      # islands of up to wide DOFs merge only up to wide: their group keeps dense rows
      g = int(-1)
      gasleep = bool(False)
      gnv = int(0)
      gmax = int(0)
      gnefc = int(0)
      gnisland = int(0)
      idofadr = int(0)
      iefcadr = int(0)
      for u in range(nisland + wp.where(free_groups, ntree, 0)):
        island = u < nisland
        i = wp.where(island, u, 0)
        t = u - nisland
        n = int(0)
        nrow = int(0)
        ne = int(0)
        nf = int(0)
        nefc = int(0)
        asleep = bool(False)
        if island:
          n = ndof[i]
          nrow = (
            counts[i * ROW_CLASSES + ROW_EQUALITY]
            + counts[i * ROW_CLASSES + ROW_FRICTION]
            + counts[i * ROW_CLASSES + ROW_LIMIT]
            + counts[i * ROW_CLASSES + ROW_CONTACT]
          )
          fit = _island_rows_fit(
            counts[i * ROW_CLASSES + ROW_EQUALITY], counts[i * ROW_CLASSES + ROW_FRICTION], nrow, iefcadr, njmax_in
          )
          ne = fit[0]
          nf = fit[1]
          nefc = fit[2]
          if tree_awake_in:
            asleep = tree_awake_in[worldid, island_root_out[worldid, i]] == 0
        else:
          if tisland[t] >= 0:
            continue
          n = tdofnum[t]
          if tree_awake_in:
            asleep = tree_awake_in[worldid, t] == 0
        cap = wp.where(wide > 0 and wp.max(gmax, n) <= wide, wide, merge_nv)
        if g >= 0 and merge and gnv + n <= cap and asleep == gasleep:
          gnv += n
          gmax = wp.max(gmax, n)
          gnefc += nefc
          gnisland += int(island)
        else:
          if g >= 0:
            group_nv_out[worldid, g] = gnv
            group_nefc_out[worldid, g] = gnefc
            group_nisland_out[worldid, g] = gnisland
          g += 1
          gnv = n
          gmax = n
          gnefc = nefc
          gnisland = int(island)
          gasleep = asleep
          group_asleep_out[worldid, g] = asleep
          group_idofadr_out[worldid, g] = idofadr
          group_iefcadr_out[worldid, g] = iefcadr
          group_ne_out[worldid, g] = ne
          group_nf_out[worldid, g] = nf
          group_island0_out[worldid, g] = wp.where(island, i, nisland)
        if island:
          island_group_out[worldid, i] = g
        else:
          tree_group_out[worldid, t] = g
        idofadr += n
        iefcadr += nrow
      if g >= 0:
        group_nv_out[worldid, g] = gnv
        group_nefc_out[worldid, g] = gnefc
        group_nisland_out[worldid, g] = gnisland
      ngroup_out[worldid] = g + 1
      # the tiled kernels read all ntree entries
      for k in range(g + 1, ntree):
        group_nv_out[worldid, k] = 0
        group_nefc_out[worldid, k] = 0
        group_idofadr_out[worldid, k] = 0
        group_iefcadr_out[worldid, k] = 0
        group_asleep_out[worldid, k] = False
      for i in range(nisland, ntree):
        island_group_out[worldid, i] = -1
    _block_sync()

    # dense rows: groups of islands up to wide DOFs (a tile per group gathers their H)
    ngroup = ngroup_out[worldid]
    for i in range(lane, ntree, lanes):
      block = wp.vec4i(0)
      if i < ngroup:
        width = _dense_width(group_nv_out[worldid, i], wide)
        if group_nisland_out[worldid, i] == 0:
          width = 0
        block = wp.vec4i(width, 0, group_nefc_out[worldid, i], group_iefcadr_out[worldid, i])
      group_dense_out[worldid, i] = block
    _block_sync()
    for i in range(lane, ntree, lanes):
      g = island_group_out[worldid, i]
      if g >= 0:
        island_dense_out[worldid, i] = group_dense_out[worldid, g]

  return kernel


@wp.kernel
def _island_dof_maps(
  # Model:
  dof_treeid: wp.array[int],
  tree_dofadr: wp.array[int],
  # Data in:
  nidof_in: wp.array[int],
  tree_island_in: wp.array2d[int],
  island_idofadr_in: wp.array2d[int],
  # In:
  tree_iofs_in: wp.array2d[int],
  tree_group_in: wp.array2d[int],
  island_group_in: wp.array2d[int],
  group_idofadr_in: wp.array2d[int],
  # Data out:
  dof_island_out: wp.array2d[int],
  island_dofadr_out: wp.array2d[int],
  map_dof2idof_out: wp.array2d[int],
  map_idof2dof_out: wp.array2d[int],
  # Out:
  idof_islandid_out: wp.array2d[int],
  dof_group_out: wp.array2d[int],
  dof_densecol_out: wp.array2d[int],
):
  """A DOF's island, island-ordered index (islands in order, DOFs ascending), group and column."""
  worldid, dofid = wp.tid()
  tree = dof_treeid[dofid]
  islandid = tree_island_in[worldid, tree]
  local = tree_iofs_in[worldid, tree] + dofid - tree_dofadr[tree]
  g = int(-1)
  col = int(0)
  if islandid >= 0:
    idof = island_idofadr_in[worldid, islandid] + local
    if local == 0:
      island_dofadr_out[worldid, islandid] = dofid
    g = island_group_in[worldid, islandid]
    col = idof - group_idofadr_in[worldid, g]
  else:
    idof = nidof_in[worldid] + local
    g = tree_group_in[worldid, tree]
  dof_island_out[worldid, dofid] = islandid
  map_dof2idof_out[worldid, dofid] = idof
  map_idof2dof_out[worldid, idof] = dofid
  idof_islandid_out[worldid, idof] = islandid
  dof_group_out[worldid, dofid] = g
  dof_densecol_out[worldid, dofid] = col


@wp.kernel
def _island_rows(
  # Data in:
  nefc_in: wp.array[int],
  nisland_in: wp.array[int],
  island_iefcadr_in: wp.array2d[int],
  # In:
  island_group_in: wp.array2d[int],
  group_asleep_in: wp.array2d[bool],
  # Data out:
  efc_force_out: wp.array2d[float],
  efc_state_out: wp.array2d[int],
  efc_island_out: wp.array2d[int],
  map_efc2iefc_out: wp.array2d[int],
  map_iefc2efc_out: wp.array2d[int],
  # Out:
  efc_group_out: wp.array2d[int],
):
  """Rows are in island order: identity maps, and each row's island and group.

  Rows of sleeping groups are not solved: no force. Slots past nefc are left alone: no reader
  goes past nefc, and writing all njmax slots leaves dirty lines the next kernels pay to evict.
  """
  worldid, efcid = wp.tid()
  if efcid >= nefc_in[worldid]:
    return
  lo = int(0)
  hi = nisland_in[worldid]
  while hi - lo > 1:
    mid = (lo + hi) // 2
    if island_iefcadr_in[worldid, mid] <= efcid:
      lo = mid
    else:
      hi = mid
  efc_island_out[worldid, efcid] = lo
  map_efc2iefc_out[worldid, efcid] = efcid
  map_iefc2efc_out[worldid, efcid] = efcid
  g = island_group_in[worldid, lo]
  efc_group_out[worldid, efcid] = g
  if group_asleep_in[worldid, g]:
    efc_force_out[worldid, efcid] = 0.0
    efc_state_out[worldid, efcid] = types.ConstraintState.SATISFIED


def island_layout(
  m: types.Model,
  d: types.Data,
  island_parent: wp.array2d[int],
  tree_nrow: wp.array2d[int],
  island_rowadr: wp.array2d[int],
  island_cursor: wp.array2d[int],
) -> GroupBuffers:
  """Islands, their row layout, DOF maps and groups, after the constraints discovered them."""
  buf = group_buffers(m, d)
  wp.launch_tiled(
    _island_world(m.ntree),
    dim=d.nworld,
    inputs=[
      m.nv,
      m.tree_dofnum,
      d.tree_awake if m.opt.enableflags & types.EnableBit.SLEEP else None,
      d.njmax,
      tree_nrow,
      MERGE,
      MERGE_NV,
      dense_width(m) if DENSE_NV >= 32 else 0,
      # without qacc_smooth, the island solver solves the free trees
      smooth.skip_qacc_smooth(m, d),
    ],
    outputs=[
      d.nisland,
      d.nidof,
      d.tree_island,
      d.island_dofadr,
      d.island_idofadr,
      d.island_nv,
      d.island_nefc,
      d.island_ne,
      d.island_nf,
      d.island_iefcadr,
      island_parent,
      island_rowadr,
      island_cursor,
      buf.tree_iofs,
      buf.tree_group,
      buf.island_root,
      buf.island_acc,
      buf.ngroup,
      buf.island_group,
      buf.group_idofadr,
      buf.group_nv,
      buf.group_nefc,
      buf.group_iefcadr,
      buf.group_ne,
      buf.group_nf,
      buf.group_island0,
      buf.group_nisland,
      buf.group_dense,
      buf.island_dense,
      buf.group_asleep,
    ],
    # whole warps of trees: its passes are bound by the block's barriers
    block_dim=min(types.BlockDim.island_scan_sizes, -(-m.ntree // 32) * 32),
  )
  wp.launch(
    _island_dof_maps,
    dim=(d.nworld, m.nv),
    inputs=[
      m.dof_treeid,
      m.tree_dofadr,
      d.nidof,
      d.tree_island,
      d.island_idofadr,
      buf.tree_iofs,
      buf.tree_group,
      buf.island_group,
      buf.group_idofadr,
    ],
    outputs=[d.dof_island, d.island_dofadr, d.map_dof2idof, d.map_idof2dof, d.dof_islandid, buf.dof_group, buf.dof_densecol],
  )
  return buf


def island_rows_done(m: types.Model, d: types.Data):
  """Row maps and groups, after the rows are written in island order."""
  buf = group_buffers(m, d)
  wp.launch(
    _island_rows,
    dim=(d.nworld, d.njmax),
    inputs=[d.nefc, d.nisland, d.island_iefcadr, buf.island_group, buf.group_asleep],
    outputs=[d.efc.force, d.efc.state, d.efc.island, d.map_efc2iefc, d.map_iefc2efc, buf.efc_group],
  )


def _island_solve_unsupported(m: types.Model) -> str:
  """The first feature of m the island solver does not support, or ''."""
  unsupported = (
    (m.opt.solver != types.SolverType.NEWTON, "the CG solver"),
    (not m.is_sparse, "a dense Jacobian"),
    (m.nflex > 0, "flex"),
    (m.opt.integrator == types.IntegratorType.DISCRETE, "the discrete integrator"),
    # these read the sparse Jacobian, which groups with dense rows leave empty
    (m.actuator_trntype_body_adr.shape[0] > 0, "body transmissions"),
    (m.opt.noslip_iterations > 0, "noslip"),
    # the island solver's atomic sums are not ordered, and sorted rows lose island order
    (m.opt.deterministic, "opt.deterministic"),
  )
  return next((name for cond, name in unsupported if cond), "")


def island_rows(m: types.Model) -> bool:
  """Whether rows are built in island order for the island solver (opt.island_solve, or sleep).

  Sleep solves the awake islands of models of any size up to MAX_NTREE trees; otherwise the world
  solver holds the sleeping DOFs. Unsupported features warn if opt.island_solve asks for islands.
  """
  if m.opt.disableflags & types.DisableBit.ISLAND or m.ntree > MAX_NTREE:
    return False
  sleep = bool(m.opt.enableflags & types.EnableBit.SLEEP)
  if not (m.opt.island_solve or sleep):
    return False
  if not sleep and (m.nv <= MIN_NV or m.ntree <= 1):
    return False
  unsupported = _island_solve_unsupported(m)
  if unsupported and m.opt.island_solve:
    warnings.warn(f"opt.island_solve does not support {unsupported}: the world solver solves this model.")
  return not unsupported


@wp.func
def rows_discover(
  # In:
  worldid: int,
  tree0: int,
  tree1: int,
  nrow: int,
  rowclass: int,
  # Data out:
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
  tree_nrow_out: wp.array2d[int],
):
  """Merge a constraint's trees and count its rows on one of them."""
  _dsu_union(island_parent_out, worldid, tree0, tree1, tree_island_out)
  tree = wp.where(tree0 >= 0, tree0, tree1)
  if tree >= 0:
    wp.atomic_add(tree_nrow_out, worldid, tree * ROW_CLASSES + rowclass, nrow)  # kernel_analyzer: ignore[atomic]


@wp.func
def rows_tendon_tree(
  # Model:
  dof_treeid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  # In:
  tenid: int,
) -> int:
  """A tree the tendon spans (its first DOF's)."""
  if tenid < 0 or ten_J_rownnz[tenid] == 0:
    return -1
  return dof_treeid[ten_J_colind[ten_J_rowadr[tenid]]]


@wp.func
def rows_discover_tendon(
  # Model:
  dof_treeid: wp.array[int],
  ten_J_rownnz: wp.array[int],
  ten_J_rowadr: wp.array[int],
  ten_J_colind: wp.array[int],
  # In:
  worldid: int,
  tree: int,
  tenid: int,
  # Data out:
  tree_island_out: wp.array2d[int],
  # Out:
  island_parent_out: wp.array2d[int],
):
  """Merge every tree a tendon spans with tree."""
  if tenid < 0:
    return
  rowadr = ten_J_rowadr[tenid]
  for k in range(ten_J_rownnz[tenid]):
    _dsu_union(island_parent_out, worldid, tree, dof_treeid[ten_J_colind[rowadr + k]], tree_island_out)


@wp.func
def rows_write(
  # Data in:
  tree_island_in: wp.array2d[int],
  # In:
  worldid: int,
  tree: int,
  nrow: int,
  rowclass: int,
  nrow_max: int,
  island_rowadr_in: wp.array2d[int],
  # Out:
  island_cursor_out: wp.array2d[int],
) -> int:
  """Address of a constraint's first row in its island (nrow_max if it has no tree)."""
  if tree < 0:
    return nrow_max
  slot = tree_island_in[worldid, tree] * ROW_CLASSES + rowclass
  return island_rowadr_in[worldid, slot] + wp.atomic_add(
    island_cursor_out, worldid, slot, nrow
  )  # kernel_analyzer: ignore[atomic]


@wp.func
def rows_island(
  # Data in:
  tree_island_in: wp.array2d[int],
  # In:
  worldid: int,
  tree0: int,
  tree1: int,
  nrow: int,
  rowclass: int,
  nrow_max: int,
  island_rowadr_in: wp.array2d[int],
  island_dense_in: wp.array2d[wp.vec4i],
  # Data out:
  nefc_out: wp.array[int],
  # Out:
  island_cursor_out: wp.array2d[int],
) -> wp.vec2i:
  """A constraint's rows in island order (counted in nefc): its first row and first dense row."""
  wp.atomic_add(nefc_out, worldid, nrow)  # kernel_analyzer: ignore[atomic]
  tree = wp.where(tree0 >= 0, tree0, tree1)
  efcid = rows_write(tree_island_in, worldid, tree, nrow, rowclass, nrow_max, island_rowadr_in, island_cursor_out)
  return wp.vec2i(efcid, rows_dense(tree_island_in, worldid, tree, efcid, island_dense_in))


@wp.func
def rows_reserve(
  # In:
  worldid: int,
  efcid: int,
  nrow: int,
  rownnz: int,
  nrow_max: int,
  nnz_max: int,
  # Out:
  efc_nnz_out: wp.array[int],
) -> int:
  """Sparse storage of a block of rows, as constraint._reserve_efc_nnz (-1: does not fit)."""
  nrow = wp.min(nrow, nrow_max - efcid)
  if nrow <= 0 or rownnz == 0:
    return 0
  nnz = nrow * rownnz
  rowadr = wp.atomic_add(efc_nnz_out, worldid, nnz)  # kernel_analyzer: ignore[atomic]
  if rowadr + nnz > nnz_max:
    return -1
  return rowadr
