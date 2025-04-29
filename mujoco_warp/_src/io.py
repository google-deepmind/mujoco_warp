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

from typing import Optional, Tuple

import mujoco
import numpy as np
import warp as wp
from packaging import version

from . import support
from . import types


def geom_pair(m: mujoco.MjModel) -> Tuple[np.array, np.array]:
  filterparent = not (m.opt.disableflags & types.DisableBit.FILTERPARENT.value)
  exclude_signature = set(m.exclude_signature)
  predefined_pairs = {(m.pair_geom1[i], m.pair_geom2[i]): i for i in range(m.npair)}

  tri = np.triu_indices(m.ngeom, k=1)  # k=1 to skip self collision pairs

  geompairs = []
  pairids = []
  for geom1, geom2 in zip(*tri):
    bodyid1 = m.geom_bodyid[geom1]
    bodyid2 = m.geom_bodyid[geom2]
    contype1 = m.geom_contype[geom1]
    contype2 = m.geom_contype[geom2]
    conaffinity1 = m.geom_conaffinity[geom1]
    conaffinity2 = m.geom_conaffinity[geom2]
    weldid1 = m.body_weldid[bodyid1]
    weldid2 = m.body_weldid[bodyid2]
    weld_parentid1 = m.body_weldid[m.body_parentid[weldid1]]
    weld_parentid2 = m.body_weldid[m.body_parentid[weldid2]]

    self_collision = weldid1 == weldid2
    parent_child_collision = (
      filterparent
      and (weldid1 != 0)
      and (weldid2 != 0)
      and ((weldid1 == weld_parentid2) or (weldid2 == weld_parentid1))
    )
    mask = (contype1 & conaffinity2) or (contype2 & conaffinity1)
    exclude = (bodyid1 << 16) + (bodyid2) in exclude_signature

    if mask and (not self_collision) and (not parent_child_collision) and (not exclude):
      pairid = -1
    else:
      pairid = -2

    # check for predefined geom pair
    pairid = predefined_pairs.get((geom1, geom2), pairid)
    pairid = predefined_pairs.get((geom2, geom1), pairid)

    pairids.append(pairid)
    geompairs.append([geom1, geom2])

  return np.array(geompairs), np.array(pairids)


def put_model(mjm: mujoco.MjModel) -> types.Model:
  # check supported features
  for field, field_types, field_str in (
    (mjm.actuator_trntype, types.TrnType, "Actuator transmission type"),
    (mjm.actuator_dyntype, types.DynType, "Actuator dynamics type"),
    (mjm.actuator_gaintype, types.GainType, "Gain type"),
    (mjm.actuator_biastype, types.BiasType, "Bias type"),
    (mjm.eq_type, types.EqType, "Equality constraint types"),
    (mjm.geom_type, types.GeomType, "Geom type"),
    (mjm.sensor_type, types.SensorType, "Sensor types"),
    (mjm.wrap_type, types.WrapType, "Wrap types"),
  ):
    unsupported = ~np.isin(field, list(field_types))
    if unsupported.any():
      raise NotImplementedError(f"{field_str} {field[unsupported]} not supported.")

  for n, msg in (
    (mjm.nplugin, "Plugins"),
    (mjm.nflex, "Flexes"),
  ):
    if n > 0:
      raise NotImplementedError(f"{msg} are unsupported.")

  if mjm.tendon_frictionloss.any():
    raise NotImplementedError("Tendon frictionloss is unsupported.")

  # check options
  for opt, opt_types, msg in (
    (mjm.opt.integrator, types.IntegratorType, "Integrator"),
    (mjm.opt.cone, types.ConeType, "Cone"),
    (mjm.opt.solver, types.SolverType, "Solver"),
  ):
    if opt not in set(opt_types):
      raise NotImplementedError(f"{msg} {opt} is unsupported.")

  if mjm.opt.wind.any():
    raise NotImplementedError("Wind is unsupported.")

  if mjm.opt.density > 0 or mjm.opt.viscosity > 0:
    raise NotImplementedError("Fluid forces are unsupported.")

  # TODO(team): remove after solver._update_gradient for Newton solver utilizes tile operations for islands
  nv_max = 60
  if mjm.nv > nv_max and (not mjm.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE):
    raise ValueError(f"Dense is unsupported for nv > {nv_max} (nv = {mjm.nv}).")

  m = types.Model()

  m.nq = mjm.nq
  m.nv = mjm.nv
  m.na = mjm.na
  m.nu = mjm.nu
  m.nbody = mjm.nbody
  m.njnt = mjm.njnt
  m.ngeom = mjm.ngeom
  m.nsite = mjm.nsite
  m.ncam = mjm.ncam
  m.nlight = mjm.nlight
  m.nmocap = mjm.nmocap
  m.nM = mjm.nM
  m.ntendon = mjm.ntendon
  m.nwrap = mjm.nwrap
  m.nsensor = mjm.nsensor
  m.nsensordata = mjm.nsensordata
  m.nlsp = mjm.opt.ls_iterations  # TODO(team): how to set nlsp?
  m.npair = mjm.npair
  m.nexclude = mjm.nexclude
  m.neq = mjm.neq
  m.opt.timestep = mjm.opt.timestep
  m.opt.tolerance = mjm.opt.tolerance
  m.opt.ls_tolerance = mjm.opt.ls_tolerance
  m.opt.gravity = wp.vec3(mjm.opt.gravity)
  m.opt.cone = mjm.opt.cone
  m.opt.solver = mjm.opt.solver
  m.opt.iterations = mjm.opt.iterations
  m.opt.ls_iterations = mjm.opt.ls_iterations
  m.opt.integrator = mjm.opt.integrator
  m.opt.disableflags = mjm.opt.disableflags
  m.opt.impratio = wp.float32(mjm.opt.impratio)
  m.opt.is_sparse = support.is_sparse(mjm)
  m.opt.ls_parallel = False
  # TODO(team) Figure out good default parameters
  m.opt.gjk_iterations = wp.int32(1)  # warp only
  m.opt.epa_iterations = wp.int32(12)  # warp only
  m.opt.epa_exact_neg_distance = wp.bool(False)  # warp only
  m.opt.depth_extension = wp.float32(0.1)  # warp only
  m.stat.meaninertia = mjm.stat.meaninertia

  m.qpos0 = wp.array(mjm.qpos0, dtype=wp.float32, ndim=1)
  m.qpos_spring = wp.array(mjm.qpos_spring, dtype=wp.float32, ndim=1)

  # dof lower triangle row and column indices
  dof_tri_row, dof_tri_col = np.tril_indices(mjm.nv)

  # indices for sparse qM full_m
  is_, js = [], []
  for i in range(mjm.nv):
    j = i
    while j > -1:
      is_.append(i)
      js.append(j)
      j = mjm.dof_parentid[j]
  qM_fullm_i = is_
  qM_fullm_j = js

  # indices for sparse qM mul_m
  is_, js, madr_ijs = [], [], []
  for i in range(mjm.nv):
    madr_ij, j = mjm.dof_Madr[i], i

    while True:
      madr_ij, j = madr_ij + 1, mjm.dof_parentid[j]
      if j == -1:
        break
      is_, js, madr_ijs = is_ + [i], js + [j], madr_ijs + [madr_ij]

  qM_mulm_i, qM_mulm_j, qM_madr_ij = (
    np.array(x, dtype=np.int32) for x in (is_, js, madr_ijs)
  )

  jnt_limited_slide_hinge_adr = np.nonzero(
    mjm.jnt_limited
    & (
      (mjm.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE)
      | (mjm.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    )
  )[0]

  jnt_limited_ball_adr = np.nonzero(
    mjm.jnt_limited & (mjm.jnt_type == mujoco.mjtJoint.mjJNT_BALL)
  )[0]

  # body_tree is BFS ordering of body ids
  # body_treeadr contains starting index of each body tree level
  bodies, body_depth = {}, np.zeros(mjm.nbody, dtype=int) - 1
  for i in range(mjm.nbody):
    body_depth[i] = body_depth[mjm.body_parentid[i]] + 1
    bodies.setdefault(body_depth[i], []).append(i)
  body_tree = np.concatenate([bodies[i] for i in range(len(bodies))])
  tree_off = [0] + [len(bodies[i]) for i in range(len(bodies))]
  body_treeadr = np.cumsum(tree_off)[:-1]

  m.body_tree = wp.array(body_tree, dtype=wp.int32, ndim=1)
  m.body_treeadr = wp.array(body_treeadr, dtype=wp.int32, ndim=1, device="cpu")

  qLD_update_tree = np.empty(shape=(0, 3), dtype=int)
  qLD_update_treeadr = np.empty(shape=(0,), dtype=int)
  qLD_tile = np.empty(shape=(0,), dtype=int)
  qLD_tileadr = np.empty(shape=(0,), dtype=int)
  qLD_tilesize = np.empty(shape=(0,), dtype=int)

  if support.is_sparse(mjm):
    # qLD_update_tree has dof tree ordering of qLD updates for sparse factor m
    # qLD_update_treeadr contains starting index of each dof tree level
    mjd = mujoco.MjData(mjm)
    if version.parse(mujoco.__version__) > version.parse("3.2.7"):
      m.M_rownnz = wp.array(mjd.M_rownnz, dtype=wp.int32, ndim=1)
      m.M_rowadr = wp.array(mjd.M_rowadr, dtype=wp.int32, ndim=1)
      m.M_colind = wp.array(mjd.M_colind, dtype=wp.int32, ndim=1)
      m.mapM2M = wp.array(mjd.mapM2M, dtype=wp.int32, ndim=1)
      qLD_updates, dof_depth = {}, np.zeros(mjm.nv, dtype=int) - 1

      rownnz = mjd.M_rownnz
      rowadr = mjd.M_rowadr

      for k in range(mjm.nv):
        dof_depth[k] = dof_depth[mjm.dof_parentid[k]] + 1
        i = mjm.dof_parentid[k]
        diag_k = rowadr[k] + rownnz[k] - 1
        Madr_ki = diag_k - 1
        while i > -1:
          qLD_updates.setdefault(dof_depth[i], []).append((i, k, Madr_ki))
          i = mjm.dof_parentid[i]
          Madr_ki -= 1

      qLD_update_tree = np.concatenate(
        [qLD_updates[i] for i in range(len(qLD_updates))]
      )
      tree_off = [0] + [len(qLD_updates[i]) for i in range(len(qLD_updates))]
      qLD_update_treeadr = np.cumsum(tree_off)[:-1]
    else:
      qLD_updates, dof_depth = {}, np.zeros(mjm.nv, dtype=int) - 1
      for k in range(mjm.nv):
        dof_depth[k] = dof_depth[mjm.dof_parentid[k]] + 1
        i = mjm.dof_parentid[k]
        Madr_ki = mjm.dof_Madr[k] + 1
        while i > -1:
          qLD_updates.setdefault(dof_depth[i], []).append((i, k, Madr_ki))
          i = mjm.dof_parentid[i]
          Madr_ki += 1

      # qLD_treeadr contains starting indicies of each level of sparse updates
      qLD_update_tree = np.concatenate(
        [qLD_updates[i] for i in range(len(qLD_updates))]
      )
      tree_off = [0] + [len(qLD_updates[i]) for i in range(len(qLD_updates))]
      qLD_update_treeadr = np.cumsum(tree_off)[:-1]

  else:
    # qLD_tile has the dof id of each tile in qLD for dense factor m
    # qLD_tileadr contains starting index in qLD_tile of each tile group
    # qLD_tilesize has the square tile size of each tile group
    tile_corners = [i for i in range(mjm.nv) if mjm.dof_parentid[i] == -1]
    tiles = {}
    for i in range(len(tile_corners)):
      tile_beg = tile_corners[i]
      tile_end = mjm.nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
      tiles.setdefault(tile_end - tile_beg, []).append(tile_beg)
    qLD_tile = np.concatenate([tiles[sz] for sz in sorted(tiles.keys())])
    tile_off = [0] + [len(tiles[sz]) for sz in sorted(tiles.keys())]
    qLD_tileadr = np.cumsum(tile_off)[:-1]
    qLD_tilesize = np.array(sorted(tiles.keys()))

  # tiles for actuator_moment - needs nu + nv tile size and offset
  actuator_moment_offset_nv = np.empty(shape=(0,), dtype=int)
  actuator_moment_offset_nu = np.empty(shape=(0,), dtype=int)
  actuator_moment_tileadr = np.empty(shape=(0,), dtype=int)
  actuator_moment_tilesize_nv = np.empty(shape=(0,), dtype=int)
  actuator_moment_tilesize_nu = np.empty(shape=(0,), dtype=int)

  if not support.is_sparse(mjm):
    # how many actuators for each tree
    tile_corners = [i for i in range(mjm.nv) if mjm.dof_parentid[i] == -1]
    tree_id = mjm.dof_treeid[tile_corners]
    num_trees = int(np.max(tree_id))
    tree = mjm.body_treeid[mjm.jnt_bodyid[mjm.actuator_trnid[:, 0]]]
    counts, ids = np.histogram(tree, bins=np.arange(0, num_trees + 2))
    acts_per_tree = dict(zip([int(i) for i in ids], [int(i) for i in counts]))

    tiles = {}
    act_beg = 0
    for i in range(len(tile_corners)):
      tile_beg = tile_corners[i]
      tile_end = mjm.nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
      tree = int(tree_id[i])
      act_num = acts_per_tree[tree]
      tiles.setdefault((tile_end - tile_beg, act_num), []).append((tile_beg, act_beg))
      act_beg += act_num

    sorted_keys = sorted(tiles.keys())
    actuator_moment_offset_nv = [
      t[0] for key in sorted_keys for t in tiles.get(key, [])
    ]
    actuator_moment_offset_nu = [
      t[1] for key in sorted_keys for t in tiles.get(key, [])
    ]
    tile_off = [0] + [len(tiles[sz]) for sz in sorted(tiles.keys())]
    actuator_moment_tileadr = np.cumsum(tile_off)[:-1]  # offset
    actuator_moment_tilesize_nv = np.array(
      [a[0] for a in sorted_keys]
    )  # for this level
    actuator_moment_tilesize_nu = np.array(
      [int(a[1]) for a in sorted_keys]
    )  # for this level

  m.qM_fullm_i = wp.array(qM_fullm_i, dtype=wp.int32, ndim=1)
  m.qM_fullm_j = wp.array(qM_fullm_j, dtype=wp.int32, ndim=1)
  m.qM_mulm_i = wp.array(qM_mulm_i, dtype=wp.int32, ndim=1)
  m.qM_mulm_j = wp.array(qM_mulm_j, dtype=wp.int32, ndim=1)
  m.qM_madr_ij = wp.array(qM_madr_ij, dtype=wp.int32, ndim=1)
  m.qLD_update_tree = wp.array(qLD_update_tree, dtype=wp.vec3i, ndim=1)
  m.qLD_update_treeadr = wp.array(
    qLD_update_treeadr, dtype=wp.int32, ndim=1, device="cpu"
  )
  m.qLD_tile = wp.array(qLD_tile, dtype=wp.int32, ndim=1)
  m.qLD_tileadr = wp.array(qLD_tileadr, dtype=wp.int32, ndim=1, device="cpu")
  m.qLD_tilesize = wp.array(qLD_tilesize, dtype=wp.int32, ndim=1, device="cpu")
  m.actuator_moment_offset_nv = wp.array(
    actuator_moment_offset_nv, dtype=wp.int32, ndim=1
  )
  m.actuator_moment_offset_nu = wp.array(
    actuator_moment_offset_nu, dtype=wp.int32, ndim=1
  )
  m.actuator_moment_tileadr = wp.array(
    actuator_moment_tileadr, dtype=wp.int32, ndim=1, device="cpu"
  )
  m.actuator_moment_tilesize_nv = wp.array(
    actuator_moment_tilesize_nv, dtype=wp.int32, ndim=1, device="cpu"
  )
  m.actuator_moment_tilesize_nu = wp.array(
    actuator_moment_tilesize_nu, dtype=wp.int32, ndim=1, device="cpu"
  )
  m.alpha_candidate = wp.array(np.linspace(0.0, 1.0, m.nlsp), dtype=wp.float32)
  m.body_dofadr = wp.array(mjm.body_dofadr, dtype=wp.int32, ndim=1)
  m.body_dofnum = wp.array(mjm.body_dofnum, dtype=wp.int32, ndim=1)
  m.body_jntadr = wp.array(mjm.body_jntadr, dtype=wp.int32, ndim=1)
  m.body_jntnum = wp.array(mjm.body_jntnum, dtype=wp.int32, ndim=1)
  m.body_parentid = wp.array(mjm.body_parentid, dtype=wp.int32, ndim=1)
  m.body_mocapid = wp.array(mjm.body_mocapid, dtype=wp.int32, ndim=1)
  mocap_bodyid = np.arange(mjm.nbody)[mjm.body_mocapid >= 0]
  mocap_bodyid = mocap_bodyid[mjm.body_mocapid[mjm.body_mocapid >= 0].argsort()]
  m.mocap_bodyid = wp.array(mocap_bodyid, dtype=wp.int32, ndim=1)
  m.body_weldid = wp.array(mjm.body_weldid, dtype=wp.int32, ndim=1)
  m.body_pos = wp.array(mjm.body_pos, dtype=wp.vec3, ndim=1)
  m.body_quat = wp.array(mjm.body_quat, dtype=wp.quat, ndim=1)
  m.body_ipos = wp.array(mjm.body_ipos, dtype=wp.vec3, ndim=1)
  m.body_iquat = wp.array(mjm.body_iquat, dtype=wp.quat, ndim=1)
  m.body_rootid = wp.array(mjm.body_rootid, dtype=wp.int32, ndim=1)
  m.body_inertia = wp.array(mjm.body_inertia, dtype=wp.vec3, ndim=1)
  m.body_mass = wp.array(mjm.body_mass, dtype=wp.float32, ndim=1)
  m.body_subtreemass = wp.array(mjm.body_subtreemass, dtype=wp.float32, ndim=1)

  subtree_mass = np.copy(mjm.body_mass)
  # TODO(team): should this be [mjm.nbody - 1, 0) ?
  for i in range(mjm.nbody - 1, -1, -1):
    subtree_mass[mjm.body_parentid[i]] += subtree_mass[i]

  m.subtree_mass = wp.array(subtree_mass, dtype=wp.float32, ndim=1)
  m.body_invweight0 = wp.array(mjm.body_invweight0, dtype=wp.float32, ndim=2)
  m.body_geomnum = wp.array(mjm.body_geomnum, dtype=wp.int32, ndim=1)
  m.body_geomadr = wp.array(mjm.body_geomadr, dtype=wp.int32, ndim=1)
  m.body_contype = wp.array(mjm.body_contype, dtype=wp.int32, ndim=1)
  m.body_conaffinity = wp.array(mjm.body_conaffinity, dtype=wp.int32, ndim=1)
  m.jnt_bodyid = wp.array(mjm.jnt_bodyid, dtype=wp.int32, ndim=1)
  m.jnt_limited = wp.array(mjm.jnt_limited, dtype=wp.int32, ndim=1)
  m.jnt_limited_slide_hinge_adr = wp.array(
    jnt_limited_slide_hinge_adr, dtype=wp.int32, ndim=1
  )
  m.jnt_limited_ball_adr = wp.array(jnt_limited_ball_adr, dtype=wp.int32, ndim=1)
  m.jnt_type = wp.array(mjm.jnt_type, dtype=wp.int32, ndim=1)
  m.jnt_solref = wp.array(mjm.jnt_solref, dtype=wp.vec2f, ndim=1)
  m.jnt_solimp = wp.array(mjm.jnt_solimp, dtype=types.vec5, ndim=1)
  m.jnt_qposadr = wp.array(mjm.jnt_qposadr, dtype=wp.int32, ndim=1)
  m.jnt_dofadr = wp.array(mjm.jnt_dofadr, dtype=wp.int32, ndim=1)
  m.jnt_axis = wp.array(mjm.jnt_axis, dtype=wp.vec3, ndim=1)
  m.jnt_pos = wp.array(mjm.jnt_pos, dtype=wp.vec3, ndim=1)
  m.jnt_range = wp.array(mjm.jnt_range, dtype=wp.float32, ndim=2)
  m.jnt_margin = wp.array(mjm.jnt_margin, dtype=wp.float32, ndim=1)
  m.jnt_stiffness = wp.array(mjm.jnt_stiffness, dtype=wp.float32, ndim=1)
  m.jnt_actfrclimited = wp.array(mjm.jnt_actfrclimited, dtype=wp.bool, ndim=1)
  m.jnt_actfrcrange = wp.array(mjm.jnt_actfrcrange, dtype=wp.vec2, ndim=1)
  m.geom_type = wp.array(mjm.geom_type, dtype=wp.int32, ndim=1)
  m.geom_bodyid = wp.array(mjm.geom_bodyid, dtype=wp.int32, ndim=1)
  m.geom_conaffinity = wp.array(mjm.geom_conaffinity, dtype=wp.int32, ndim=1)
  m.geom_contype = wp.array(mjm.geom_contype, dtype=wp.int32, ndim=1)
  m.geom_condim = wp.array(mjm.geom_condim, dtype=wp.int32, ndim=1)
  m.geom_pos = wp.array(mjm.geom_pos, dtype=wp.vec3, ndim=1)
  m.geom_quat = wp.array(mjm.geom_quat, dtype=wp.quat, ndim=1)
  m.geom_size = wp.array(mjm.geom_size, dtype=wp.vec3, ndim=1)
  m.geom_priority = wp.array(mjm.geom_priority, dtype=wp.int32, ndim=1)
  m.geom_solmix = wp.array(mjm.geom_solmix, dtype=wp.float32, ndim=1)
  m.geom_solref = wp.array(mjm.geom_solref, dtype=wp.vec2, ndim=1)
  m.geom_solimp = wp.array(mjm.geom_solimp, dtype=types.vec5, ndim=1)
  m.geom_friction = wp.array(mjm.geom_friction, dtype=wp.vec3, ndim=1)
  m.geom_margin = wp.array(mjm.geom_margin, dtype=wp.float32, ndim=1)
  m.geom_gap = wp.array(mjm.geom_gap, dtype=wp.float32, ndim=1)
  m.geom_aabb = wp.array(mjm.geom_aabb, dtype=wp.vec3, ndim=3)
  m.geom_rbound = wp.array(mjm.geom_rbound, dtype=wp.float32, ndim=1)
  m.geom_dataid = wp.array(mjm.geom_dataid, dtype=wp.int32, ndim=1)
  m.mesh_vertadr = wp.array(mjm.mesh_vertadr, dtype=wp.int32, ndim=1)
  m.mesh_vertnum = wp.array(mjm.mesh_vertnum, dtype=wp.int32, ndim=1)
  m.mesh_vert = wp.array(mjm.mesh_vert, dtype=wp.vec3, ndim=1)
  m.eq_type = wp.array(mjm.eq_type, dtype=wp.int32, ndim=1)
  m.eq_obj1id = wp.array(mjm.eq_obj1id, dtype=wp.int32, ndim=1)
  m.eq_obj2id = wp.array(mjm.eq_obj2id, dtype=wp.int32, ndim=1)
  m.eq_objtype = wp.array(mjm.eq_objtype, dtype=wp.int32, ndim=1)
  m.eq_active0 = wp.array(mjm.eq_active0, dtype=wp.bool, ndim=1)
  m.eq_solref = wp.array(mjm.eq_solref, dtype=wp.vec2, ndim=1)
  m.eq_solimp = wp.array(mjm.eq_solimp, dtype=types.vec5, ndim=1)
  m.eq_data = wp.array(mjm.eq_data, dtype=types.vec11, ndim=1)
  m.site_pos = wp.array(mjm.site_pos, dtype=wp.vec3, ndim=1)
  m.site_quat = wp.array(mjm.site_quat, dtype=wp.quat, ndim=1)
  m.site_bodyid = wp.array(mjm.site_bodyid, dtype=wp.int32, ndim=1)
  m.cam_mode = wp.array(mjm.cam_mode, dtype=wp.int32, ndim=1)
  m.cam_bodyid = wp.array(mjm.cam_bodyid, dtype=wp.int32, ndim=1)
  m.cam_targetbodyid = wp.array(mjm.cam_targetbodyid, dtype=wp.int32, ndim=1)
  m.cam_pos = wp.array(mjm.cam_pos, dtype=wp.vec3, ndim=1)
  m.cam_quat = wp.array(mjm.cam_quat, dtype=wp.quat, ndim=1)
  m.cam_poscom0 = wp.array(mjm.cam_poscom0, dtype=wp.vec3, ndim=1)
  m.cam_pos0 = wp.array(mjm.cam_pos0, dtype=wp.vec3, ndim=1)
  m.light_mode = wp.array(mjm.light_mode, dtype=wp.int32, ndim=1)
  m.light_bodyid = wp.array(mjm.light_bodyid, dtype=wp.int32, ndim=1)
  m.light_targetbodyid = wp.array(mjm.light_targetbodyid, dtype=wp.int32, ndim=1)
  m.light_pos = wp.array(mjm.light_pos, dtype=wp.vec3, ndim=1)
  m.light_dir = wp.array(mjm.light_dir, dtype=wp.vec3, ndim=1)
  m.light_poscom0 = wp.array(mjm.light_poscom0, dtype=wp.vec3, ndim=1)
  m.light_pos0 = wp.array(mjm.light_pos0, dtype=wp.vec3, ndim=1)
  m.dof_bodyid = wp.array(mjm.dof_bodyid, dtype=wp.int32, ndim=1)
  m.dof_jntid = wp.array(mjm.dof_jntid, dtype=wp.int32, ndim=1)
  m.dof_parentid = wp.array(mjm.dof_parentid, dtype=wp.int32, ndim=1)
  m.dof_Madr = wp.array(mjm.dof_Madr, dtype=wp.int32, ndim=1)
  m.dof_armature = wp.array(mjm.dof_armature, dtype=wp.float32, ndim=1)
  m.dof_damping = wp.array(mjm.dof_damping, dtype=wp.float32, ndim=1)
  m.dof_frictionloss = wp.array(mjm.dof_frictionloss, dtype=wp.float32, ndim=1)
  m.dof_solimp = wp.array(mjm.dof_solimp, dtype=types.vec5, ndim=1)
  m.dof_solref = wp.array(mjm.dof_solref, dtype=wp.vec2, ndim=1)
  m.dof_tri_row = wp.from_numpy(dof_tri_row, dtype=wp.int32)
  m.dof_tri_col = wp.from_numpy(dof_tri_col, dtype=wp.int32)
  m.dof_invweight0 = wp.array(mjm.dof_invweight0, dtype=wp.float32, ndim=1)
  m.actuator_trntype = wp.array(mjm.actuator_trntype, dtype=wp.int32, ndim=1)
  m.actuator_trnid = wp.array(mjm.actuator_trnid, dtype=wp.int32, ndim=2)
  m.actuator_ctrllimited = wp.array(mjm.actuator_ctrllimited, dtype=wp.bool, ndim=1)
  m.actuator_ctrlrange = wp.array(mjm.actuator_ctrlrange, dtype=wp.vec2, ndim=1)
  m.actuator_forcelimited = wp.array(mjm.actuator_forcelimited, dtype=wp.bool, ndim=1)
  m.actuator_forcerange = wp.array(mjm.actuator_forcerange, dtype=wp.vec2, ndim=1)
  m.actuator_gaintype = wp.array(mjm.actuator_gaintype, dtype=wp.int32, ndim=1)
  m.actuator_gainprm = wp.array(mjm.actuator_gainprm, dtype=types.vec10f, ndim=1)
  m.actuator_biastype = wp.array(mjm.actuator_biastype, dtype=wp.int32, ndim=1)
  m.actuator_biasprm = wp.array(mjm.actuator_biasprm, dtype=types.vec10f, ndim=1)
  m.actuator_gear = wp.array(mjm.actuator_gear, dtype=wp.spatial_vector, ndim=1)
  m.actuator_actlimited = wp.array(mjm.actuator_actlimited, dtype=wp.bool, ndim=1)
  m.actuator_actrange = wp.array(mjm.actuator_actrange, dtype=wp.vec2, ndim=1)
  m.actuator_actadr = wp.array(mjm.actuator_actadr, dtype=wp.int32, ndim=1)
  m.actuator_actnum = wp.array(mjm.actuator_actnum, dtype=wp.int32, ndim=1)
  m.actuator_dyntype = wp.array(mjm.actuator_dyntype, dtype=wp.int32, ndim=1)
  m.actuator_dynprm = wp.array(mjm.actuator_dynprm, dtype=types.vec10f, ndim=1)
  m.exclude_signature = wp.array(mjm.exclude_signature, dtype=wp.int32, ndim=1)

  # pre-compute indices of equality constraints
  m.eq_connect_adr = wp.array(
    np.nonzero(mjm.eq_type == types.EqType.CONNECT.value)[0], dtype=wp.int32, ndim=1
  )
  m.eq_wld_adr = wp.array(
    np.nonzero(mjm.eq_type == types.EqType.WELD.value)[0], dtype=wp.int32, ndim=1
  )
  m.eq_jnt_adr = wp.array(
    np.nonzero(mjm.eq_type == types.EqType.JOINT.value)[0], dtype=wp.int32, ndim=1
  )

  # short-circuiting here allows us to skip a lot of code in implicit integration
  m.actuator_affine_bias_gain = bool(
    np.any(mjm.actuator_biastype == types.BiasType.AFFINE.value)
    or np.any(mjm.actuator_gaintype == types.GainType.AFFINE.value)
  )

  geompair, pairid = geom_pair(mjm)
  m.nxn_geom_pair = wp.array(geompair, dtype=wp.vec2i, ndim=1)
  m.nxn_pairid = wp.array(pairid, dtype=wp.int32, ndim=1)

  # predefined collision pairs
  m.pair_dim = wp.array(mjm.pair_dim, dtype=wp.int32, ndim=1)
  m.pair_geom1 = wp.array(mjm.pair_geom1, dtype=wp.int32, ndim=1)
  m.pair_geom2 = wp.array(mjm.pair_geom2, dtype=wp.int32, ndim=1)
  m.pair_solref = wp.array(mjm.pair_solref, dtype=wp.vec2, ndim=1)
  m.pair_solreffriction = wp.array(mjm.pair_solreffriction, dtype=wp.vec2, ndim=1)
  m.pair_solimp = wp.array(mjm.pair_solimp, dtype=types.vec5, ndim=1)
  m.pair_margin = wp.array(mjm.pair_margin, dtype=wp.float32, ndim=1)
  m.pair_gap = wp.array(mjm.pair_gap, dtype=wp.float32, ndim=1)
  m.pair_friction = wp.array(mjm.pair_friction, dtype=types.vec5, ndim=1)
  m.condim_max = np.max(mjm.geom_condim)  # TODO(team): get max after filtering

  # tendon
  m.tendon_adr = wp.array(mjm.tendon_adr, dtype=wp.int32, ndim=1)
  m.tendon_num = wp.array(mjm.tendon_num, dtype=wp.int32, ndim=1)
  m.tendon_limited = wp.array(mjm.tendon_limited, dtype=wp.int32, ndim=1)
  m.tendon_limited_adr = wp.array(
    np.nonzero(mjm.tendon_limited)[0], dtype=wp.int32, ndim=1
  )
  m.tendon_solref_lim = wp.array(mjm.tendon_solref_lim, dtype=wp.vec2f, ndim=1)
  m.tendon_solimp_lim = wp.array(mjm.tendon_solimp_lim, dtype=types.vec5, ndim=1)
  m.tendon_range = wp.array(mjm.tendon_range, dtype=wp.vec2f, ndim=1)
  m.tendon_margin = wp.array(mjm.tendon_margin, dtype=wp.float32, ndim=1)
  m.tendon_length0 = wp.array(mjm.tendon_length0, dtype=wp.float32, ndim=1)
  m.tendon_invweight0 = wp.array(mjm.tendon_invweight0, dtype=wp.float32, ndim=1)
  m.wrap_objid = wp.array(mjm.wrap_objid, dtype=wp.int32, ndim=1)
  m.wrap_prm = wp.array(mjm.wrap_prm, dtype=wp.float32, ndim=1)
  m.wrap_type = wp.array(mjm.wrap_type, dtype=wp.int32, ndim=1)

  # fixed tendon
  tendon_jnt_adr = []
  wrap_jnt_adr = []
  for i in range(mjm.ntendon):
    adr = mjm.tendon_adr[i]
    if mjm.wrap_type[adr] == mujoco.mjtWrap.mjWRAP_JOINT:
      tendon_num = mjm.tendon_num[i]
      for j in range(tendon_num):
        tendon_jnt_adr.append(i)
        wrap_jnt_adr.append(adr + j)

  m.tendon_jnt_adr = wp.array(tendon_jnt_adr, dtype=wp.int32, ndim=1)
  m.wrap_jnt_adr = wp.array(wrap_jnt_adr, dtype=wp.int32, ndim=1)

  # spatial tendon
  tendon_site_adr = []
  tendon_site_pair_adr = []
  ten_wrapadr_site = [0]
  ten_wrapnum_site = []
  for i, tendon_num in enumerate(mjm.tendon_num):
    adr = mjm.tendon_adr[i]
    if (mjm.wrap_type[adr : adr + tendon_num] == mujoco.mjtWrap.mjWRAP_SITE).all():
      if i < mjm.ntendon:
        ten_wrapadr_site.append(ten_wrapadr_site[-1] + tendon_num)
      ten_wrapnum_site.append(tendon_num)
      for j in range(tendon_num):
        if j < tendon_num - 1:
          tendon_site_pair_adr.append(i)
        tendon_site_adr.append(i)
    else:
      if i < mjm.ntendon:
        ten_wrapadr_site.append(ten_wrapadr_site[-1])
      ten_wrapnum_site.append(0)

  tendon_site_adr = np.array(tendon_site_adr)
  tendon_site_pair_adr = np.array(tendon_site_pair_adr)
  wrap_site_adr = np.nonzero(mjm.wrap_type == mujoco.mjtWrap.mjWRAP_SITE)[0]
  wrap_site_pair_adr = np.setdiff1d(
    wrap_site_adr[np.nonzero(np.diff(wrap_site_adr) == 1)[0]], mjm.tendon_adr[1:] - 1
  )

  m.tendon_site_adr = wp.array(tendon_site_adr, dtype=wp.int32, ndim=1)
  m.tendon_site_pair_adr = wp.array(tendon_site_pair_adr, dtype=wp.int32, ndim=1)
  m.ten_wrapadr_site = wp.array(ten_wrapadr_site, dtype=wp.int32, ndim=1)
  m.ten_wrapnum_site = wp.array(ten_wrapnum_site, dtype=wp.int32, ndim=1)
  m.wrap_site_adr = wp.array(wrap_site_adr, dtype=wp.int32, ndim=1)
  m.wrap_site_pair_adr = wp.array(wrap_site_pair_adr, dtype=wp.int32, ndim=1)

  # sensors
  m.sensor_type = wp.array(mjm.sensor_type, dtype=wp.int32, ndim=1)
  m.sensor_datatype = wp.array(mjm.sensor_datatype, dtype=wp.int32, ndim=1)
  m.sensor_objtype = wp.array(mjm.sensor_objtype, dtype=wp.int32, ndim=1)
  m.sensor_objid = wp.array(mjm.sensor_objid, dtype=wp.int32, ndim=1)
  m.sensor_reftype = wp.array(mjm.sensor_reftype, dtype=wp.int32, ndim=1)
  m.sensor_refid = wp.array(mjm.sensor_refid, dtype=wp.int32, ndim=1)
  m.sensor_dim = wp.array(mjm.sensor_dim, dtype=wp.int32, ndim=1)
  m.sensor_adr = wp.array(mjm.sensor_adr, dtype=wp.int32, ndim=1)
  m.sensor_cutoff = wp.array(mjm.sensor_cutoff, dtype=wp.float32, ndim=1)
  m.sensor_pos_adr = wp.array(
    np.nonzero(mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_POS)[0],
    dtype=wp.int32,
    ndim=1,
  )
  m.sensor_vel_adr = wp.array(
    np.nonzero(mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_VEL)[0],
    dtype=wp.int32,
    ndim=1,
  )
  m.sensor_acc_adr = wp.array(
    np.nonzero(mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_ACC)[0],
    dtype=wp.int32,
    ndim=1,
  )

  m.sensor_subtree_vel = np.isin(
    mjm.sensor_type,
    [mujoco.mjtSensor.mjSENS_SUBTREELINVEL, mujoco.mjtSensor.mjSENS_SUBTREEANGMOM],
  ).any()
  m.sensor_rne_postconstraint = np.isin(
    mjm.sensor_type,
    [
      mujoco.mjtSensor.mjSENS_ACCELEROMETER,
      mujoco.mjtSensor.mjSENS_FORCE,
      mujoco.mjtSensor.mjSENS_TORQUE,
      mujoco.mjtSensor.mjSENS_FRAMELINACC,
      mujoco.mjtSensor.mjSENS_FRAMEANGACC,
    ],
  ).any()
  return m


def _constraint(
  mjm: mujoco.MjModel, nworld: int, nconmax: int, njmax: int
) -> types.Constraint:
  efc = types.Constraint()

  efc.J = wp.zeros((njmax, mjm.nv), dtype=wp.float32)
  efc.D = wp.zeros((njmax,), dtype=wp.float32)
  efc.pos = wp.zeros((njmax,), dtype=wp.float32)
  efc.aref = wp.zeros((njmax,), dtype=wp.float32)
  efc.frictionloss = wp.zeros((njmax,), dtype=wp.float32)
  efc.force = wp.zeros((njmax,), dtype=wp.float32)
  efc.margin = wp.zeros((njmax,), dtype=wp.float32)
  efc.worldid = wp.zeros((njmax,), dtype=wp.int32)
  efc.id = wp.zeros((njmax,), dtype=wp.int32)

  efc.Jaref = wp.empty(shape=(njmax,), dtype=wp.float32)
  efc.Ma = wp.empty(shape=(nworld, mjm.nv), dtype=wp.float32)
  efc.grad = wp.empty(shape=(nworld, mjm.nv), dtype=wp.float32)
  efc.grad_dot = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.Mgrad = wp.empty(shape=(nworld, mjm.nv), dtype=wp.float32)
  efc.search = wp.empty(shape=(nworld, mjm.nv), dtype=wp.float32)
  efc.search_dot = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.gauss = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.cost = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.prev_cost = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.solver_niter = wp.empty(shape=(nworld,), dtype=wp.int32)
  efc.active = wp.empty(shape=(njmax,), dtype=bool)
  efc.gtol = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.mv = wp.empty(shape=(nworld, mjm.nv), dtype=wp.float32)
  efc.jv = wp.empty(shape=(njmax,), dtype=wp.float32)
  efc.quad = wp.empty(shape=(njmax,), dtype=wp.vec3f)
  efc.quad_gauss = wp.empty(shape=(nworld,), dtype=wp.vec3f)
  efc.h = wp.empty(shape=(nworld, mjm.nv, mjm.nv), dtype=wp.float32)
  efc.alpha = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.prev_grad = wp.empty(shape=(nworld, mjm.nv), dtype=wp.float32)
  efc.prev_Mgrad = wp.empty(shape=(nworld, mjm.nv), dtype=wp.float32)
  efc.beta = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.beta_num = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.beta_den = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.done = wp.empty(shape=(nworld,), dtype=bool)

  efc.ls_done = wp.zeros(shape=(nworld,), dtype=bool)
  efc.p0 = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.lo = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.lo_alpha = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.hi = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.hi_alpha = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.lo_next = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.lo_next_alpha = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.hi_next = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.hi_next_alpha = wp.empty(shape=(nworld,), dtype=wp.float32)
  efc.mid = wp.empty(shape=(nworld,), dtype=wp.vec3)
  efc.mid_alpha = wp.empty(shape=(nworld,), dtype=wp.float32)

  efc.cost_candidate = wp.empty(shape=(nworld, mjm.opt.ls_iterations), dtype=wp.float32)
  efc.quad_total_candidate = wp.empty(
    shape=(nworld, mjm.opt.ls_iterations), dtype=wp.vec3f
  )

  # TODO(team): skip allocation if not elliptic?
  efc.u = wp.empty((nconmax, 6), dtype=wp.float32)
  efc.uu = wp.empty((nconmax,), dtype=wp.float32)
  efc.uv = wp.empty((nconmax,), dtype=wp.float32)
  efc.vv = wp.empty((nconmax,), dtype=wp.float32)
  efc.condim = wp.empty((njmax,), dtype=wp.int32)

  return efc


def make_data(
  mjm: mujoco.MjModel, nworld: int = 1, nconmax: int = -1, njmax: int = -1
) -> types.Data:
  d = types.Data()
  d.nworld = nworld

  # TODO(team): move to Model?
  if nconmax == -1:
    # TODO(team): heuristic for nconmax
    nconmax = nworld * 20
  d.nconmax = nconmax
  if njmax == -1:
    # TODO(team): heuristic for njmax
    njmax = nworld * 20 * 6
  d.njmax = njmax

  d.ncon = wp.zeros(1, dtype=wp.int32)
  d.ne = wp.zeros(1, dtype=wp.int32, ndim=1)
  d.ne_connect = wp.zeros(1, dtype=wp.int32, ndim=1)
  d.ne_weld = wp.zeros(1, dtype=wp.int32, ndim=1)
  d.ne_jnt = wp.zeros(1, dtype=wp.int32, ndim=1)
  d.nefc = wp.zeros(1, dtype=wp.int32, ndim=1)
  d.ne = wp.zeros(1, dtype=wp.int32)
  d.nf = wp.zeros(1, dtype=wp.int32)
  d.nl = wp.zeros(1, dtype=wp.int32)

  d.time = wp.zeros(nworld, dtype=wp.float32, ndim=1)

  qpos0 = np.tile(mjm.qpos0, (nworld, 1))
  d.qpos = wp.array(qpos0, dtype=wp.float32, ndim=2)
  d.qvel = wp.zeros((nworld, mjm.nv), dtype=wp.float32, ndim=2)
  d.qacc_warmstart = wp.zeros((nworld, mjm.nv), dtype=wp.float32, ndim=2)
  d.qfrc_applied = wp.zeros((nworld, mjm.nv), dtype=wp.float32, ndim=2)
  d.mocap_pos = wp.zeros((nworld, mjm.nmocap), dtype=wp.vec3)
  d.mocap_quat = wp.zeros((nworld, mjm.nmocap), dtype=wp.quat)
  d.qacc = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.xanchor = wp.zeros((nworld, mjm.njnt), dtype=wp.vec3)
  d.xaxis = wp.zeros((nworld, mjm.njnt), dtype=wp.vec3)
  d.xmat = wp.zeros((nworld, mjm.nbody), dtype=wp.mat33)
  d.xpos = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.xquat = wp.zeros((nworld, mjm.nbody), dtype=wp.quat)
  d.xipos = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.ximat = wp.zeros((nworld, mjm.nbody), dtype=wp.mat33)
  d.subtree_com = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.geom_xpos = wp.zeros((nworld, mjm.ngeom), dtype=wp.vec3)
  d.geom_xmat = wp.zeros((nworld, mjm.ngeom), dtype=wp.mat33)
  d.site_xpos = wp.zeros((nworld, mjm.nsite), dtype=wp.vec3)
  d.site_xmat = wp.zeros((nworld, mjm.nsite), dtype=wp.mat33)
  d.cam_xpos = wp.zeros((nworld, mjm.ncam), dtype=wp.vec3)
  d.cam_xmat = wp.zeros((nworld, mjm.ncam), dtype=wp.mat33)
  d.light_xpos = wp.zeros((nworld, mjm.nlight), dtype=wp.vec3)
  d.light_xdir = wp.zeros((nworld, mjm.nlight), dtype=wp.vec3)
  d.cinert = wp.zeros((nworld, mjm.nbody), dtype=types.vec10)
  d.cdof = wp.zeros((nworld, mjm.nv), dtype=wp.spatial_vector)
  d.ctrl = wp.zeros((nworld, mjm.nu), dtype=wp.float32)
  d.ten_velocity = wp.zeros((nworld, mjm.ntendon), dtype=wp.float32)
  d.actuator_velocity = wp.zeros((nworld, mjm.nu), dtype=wp.float32)
  d.actuator_force = wp.zeros((nworld, mjm.nu), dtype=wp.float32)
  d.actuator_length = wp.zeros((nworld, mjm.nu), dtype=wp.float32)
  d.actuator_moment = wp.zeros((nworld, mjm.nu, mjm.nv), dtype=wp.float32)
  d.crb = wp.zeros((nworld, mjm.nbody), dtype=types.vec10)
  if support.is_sparse(mjm):
    d.qM = wp.zeros((nworld, 1, mjm.nM), dtype=wp.float32)
    d.qLD = wp.zeros((nworld, 1, mjm.nM), dtype=wp.float32)
  else:
    d.qM = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=wp.float32)
    d.qLD = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=wp.float32)
  d.act_dot = wp.zeros((nworld, mjm.na), dtype=wp.float32)
  d.act = wp.zeros((nworld, mjm.na), dtype=wp.float32)
  d.qLDiagInv = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.cvel = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector)
  d.cdof_dot = wp.zeros((nworld, mjm.nv), dtype=wp.spatial_vector)
  d.qfrc_bias = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.contact = types.Contact()
  d.contact.dist = wp.zeros((nconmax,), dtype=wp.float32)
  d.contact.pos = wp.zeros((nconmax,), dtype=wp.vec3f)
  d.contact.frame = wp.zeros((nconmax,), dtype=wp.mat33f)
  d.contact.includemargin = wp.zeros((nconmax,), dtype=wp.float32)
  d.contact.friction = wp.zeros((nconmax,), dtype=types.vec5)
  d.contact.solref = wp.zeros((nconmax,), dtype=wp.vec2f)
  d.contact.solreffriction = wp.zeros((nconmax,), dtype=wp.vec2f)
  d.contact.solimp = wp.zeros((nconmax,), dtype=types.vec5)
  d.contact.dim = wp.zeros((nconmax,), dtype=wp.int32)
  d.contact.geom = wp.zeros((nconmax,), dtype=wp.vec2i)
  d.contact.efc_address = wp.zeros((nconmax, np.max(mjm.geom_condim)), dtype=wp.int32)
  d.contact.worldid = wp.zeros((nconmax,), dtype=wp.int32)
  d.efc = _constraint(mjm, d.nworld, d.nconmax, d.njmax)
  d.qfrc_passive = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.subtree_linvel = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.subtree_angmom = wp.zeros((nworld, mjm.nbody), dtype=wp.vec3)
  d.subtree_bodyvel = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector)
  d.qfrc_spring = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_damper = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_actuator = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_smooth = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qfrc_constraint = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_smooth = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.xfrc_applied = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector)
  d.eq_active = wp.array(np.tile(mjm.eq_active0, (nworld, 1)), dtype=wp.bool, ndim=2)

  # internal tmp arrays
  d.qfrc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qM_integration = wp.zeros_like(d.qM)
  d.qLD_integration = wp.zeros_like(d.qLD)
  d.qLDiagInv_integration = wp.zeros_like(d.qLDiagInv)
  d.act_vel_integration = wp.zeros_like(d.ctrl)
  d.qpos_t0 = wp.zeros((nworld, mjm.nq), dtype=wp.float32)
  d.qvel_t0 = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.act_t0 = wp.zeros((nworld, mjm.na), dtype=wp.float32)
  d.qvel_rk = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_rk = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.act_dot_rk = wp.zeros((nworld, mjm.na), dtype=wp.float32)

  # sweep-and-prune broadphase
  d.sap_projection_lower = wp.zeros((2 * nworld, mjm.ngeom), dtype=wp.float32)
  d.sap_projection_upper = wp.zeros((nworld, mjm.ngeom), dtype=wp.float32)
  d.sap_sort_index = wp.zeros((2 * nworld, mjm.ngeom), dtype=wp.int32)
  d.sap_range = wp.zeros((nworld, mjm.ngeom), dtype=wp.int32)
  d.sap_cumulative_sum = wp.zeros(nworld * mjm.ngeom, dtype=wp.int32)
  segment_indices_list = [i * mjm.ngeom for i in range(nworld + 1)]
  d.sap_segment_index = wp.array(segment_indices_list, dtype=int)

  # collision driver
  d.collision_pair = wp.empty(nconmax, dtype=wp.vec2i, ndim=1)
  d.collision_pairid = wp.empty(nconmax, dtype=wp.int32, ndim=1)
  d.collision_worldid = wp.empty(nconmax, dtype=wp.int32, ndim=1)
  d.ncollision = wp.zeros(1, dtype=wp.int32, ndim=1)

  # rne_postconstraint
  d.cacc = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector, ndim=2)
  d.cfrc_int = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector, ndim=2)
  d.cfrc_ext = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector, ndim=2)

  # tendon
  d.ten_length = wp.zeros((nworld, mjm.ntendon), dtype=wp.float32, ndim=2)
  d.ten_J = wp.zeros((nworld, mjm.ntendon, mjm.nv), dtype=wp.float32, ndim=3)
  d.ten_wrapadr = wp.zeros((nworld, mjm.ntendon), dtype=wp.int32, ndim=2)
  d.ten_wrapnum = wp.zeros((nworld, mjm.ntendon), dtype=wp.int32, ndim=2)
  d.wrap_obj = wp.zeros((nworld, mjm.nwrap), dtype=wp.vec2i, ndim=2)
  d.wrap_xpos = wp.zeros(
    (nworld, mjm.nwrap), dtype=wp.spatial_vector, ndim=2
  )  # TODO(team): vec6?

  # sensors
  d.sensordata = wp.zeros((nworld, mjm.nsensordata), dtype=wp.float32)

  return d


def put_data(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nworld: Optional[int] = None,
  nconmax: Optional[int] = None,
  njmax: Optional[int] = None,
) -> types.Data:
  d = types.Data()

  # TODO(team): confirm that Data is set correctly for solver with elliptic friction cones

  nworld = nworld or 1
  # TODO(team): better heuristic for nconmax
  nconmax = nconmax or max(512, mjd.ncon * nworld)
  # TODO(team): better heuristic for njmax
  njmax = njmax or max(512, mjd.nefc * nworld)

  if nworld < 1:
    raise ValueError("nworld must be >= 1")

  if nconmax < 1:
    raise ValueError("nconmax must be >= 1")

  if njmax < 1:
    raise ValueError("njmax must be >= 1")

  if nworld * mjd.ncon > nconmax:
    raise ValueError(f"nconmax overflow (nconmax must be >= {nworld * mjd.ncon})")

  if nworld * mjd.nefc > njmax:
    raise ValueError(f"njmax overflow (njmax must be >= {nworld * mjd.nefc})")

  d.nworld = nworld
  # TODO(team): move nconmax and njmax to Model?
  d.nconmax = nconmax
  d.njmax = njmax

  d.ncon = wp.array([mjd.ncon * nworld], dtype=wp.int32, ndim=1)
  d.ne = wp.array([mjd.ne * nworld], dtype=wp.int32, ndim=1)
  d.ne_connect = wp.array(
    [3 * np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_CONNECT) & mjd.eq_active) * nworld],
    dtype=wp.int32,
    ndim=1,
  )
  d.ne_weld = wp.array(
    [6 * np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_WELD) & mjd.eq_active) * nworld],
    dtype=wp.int32,
    ndim=1,
  )
  d.ne_jnt = wp.array(
    [np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_JOINT) & mjd.eq_active) * nworld],
    dtype=wp.int32,
    ndim=1,
  )
  d.nf = wp.array([mjd.nf * nworld], dtype=wp.int32, ndim=1)
  d.nl = wp.array([mjd.nl * nworld], dtype=wp.int32, ndim=1)
  d.nefc = wp.array([mjd.nefc * nworld], dtype=wp.int32, ndim=1)

  d.time = wp.array(mjd.time * np.ones(nworld), dtype=wp.float32, ndim=1)

  # TODO(erikfrey): would it be better to tile on the gpu?
  def tile(x):
    return np.tile(x, (nworld,) + (1,) * len(x.shape))

  if support.is_sparse(mjm):
    qM = np.expand_dims(mjd.qM, axis=0)
    qLD = np.expand_dims(mjd.qLD, axis=0)
    efc_J = np.zeros((mjd.nefc, mjm.nv))
    mujoco.mju_sparse2dense(
      efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind
    )
  else:
    qM = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, qM, mjd.qM)
    qLD = np.linalg.cholesky(qM)
    efc_J = mjd.efc_J.reshape((mjd.nefc, mjm.nv))

  # TODO(taylorhowell): sparse actuator_moment
  actuator_moment = np.zeros((mjm.nu, mjm.nv))
  mujoco.mju_sparse2dense(
    actuator_moment,
    mjd.actuator_moment,
    mjd.moment_rownnz,
    mjd.moment_rowadr,
    mjd.moment_colind,
  )

  d.qpos = wp.array(tile(mjd.qpos), dtype=wp.float32, ndim=2)
  d.qvel = wp.array(tile(mjd.qvel), dtype=wp.float32, ndim=2)
  d.qacc_warmstart = wp.array(tile(mjd.qacc_warmstart), dtype=wp.float32, ndim=2)
  d.qfrc_applied = wp.array(tile(mjd.qfrc_applied), dtype=wp.float32, ndim=2)
  d.mocap_pos = wp.array(tile(mjd.mocap_pos), dtype=wp.vec3, ndim=2)
  d.mocap_quat = wp.array(tile(mjd.mocap_quat), dtype=wp.quat, ndim=2)
  d.qacc = wp.array(tile(mjd.qacc), dtype=wp.float32, ndim=2)
  d.xanchor = wp.array(tile(mjd.xanchor), dtype=wp.vec3, ndim=2)
  d.xaxis = wp.array(tile(mjd.xaxis), dtype=wp.vec3, ndim=2)
  d.xmat = wp.array(tile(mjd.xmat), dtype=wp.mat33, ndim=2)
  d.xpos = wp.array(tile(mjd.xpos), dtype=wp.vec3, ndim=2)
  d.xquat = wp.array(tile(mjd.xquat), dtype=wp.quat, ndim=2)
  d.xipos = wp.array(tile(mjd.xipos), dtype=wp.vec3, ndim=2)
  d.ximat = wp.array(tile(mjd.ximat), dtype=wp.mat33, ndim=2)
  d.subtree_com = wp.array(tile(mjd.subtree_com), dtype=wp.vec3, ndim=2)
  d.geom_xpos = wp.array(tile(mjd.geom_xpos), dtype=wp.vec3, ndim=2)
  d.geom_xmat = wp.array(tile(mjd.geom_xmat), dtype=wp.mat33, ndim=2)
  d.site_xpos = wp.array(tile(mjd.site_xpos), dtype=wp.vec3, ndim=2)
  d.site_xmat = wp.array(tile(mjd.site_xmat), dtype=wp.mat33, ndim=2)
  d.cam_xpos = wp.array(tile(mjd.cam_xpos), dtype=wp.vec3, ndim=2)
  d.cam_xmat = wp.array(tile(mjd.cam_xmat.reshape(-1, 3, 3)), dtype=wp.mat33, ndim=2)
  d.light_xpos = wp.array(tile(mjd.light_xpos), dtype=wp.vec3, ndim=2)
  d.light_xdir = wp.array(tile(mjd.light_xdir), dtype=wp.vec3, ndim=2)
  d.cinert = wp.array(tile(mjd.cinert), dtype=types.vec10, ndim=2)
  d.cdof = wp.array(tile(mjd.cdof), dtype=wp.spatial_vector, ndim=2)
  d.crb = wp.array(tile(mjd.crb), dtype=types.vec10, ndim=2)
  d.qM = wp.array(tile(qM), dtype=wp.float32, ndim=3)
  d.qLD = wp.array(tile(qLD), dtype=wp.float32, ndim=3)
  d.qLDiagInv = wp.array(tile(mjd.qLDiagInv), dtype=wp.float32, ndim=2)
  d.ctrl = wp.array(tile(mjd.ctrl), dtype=wp.float32, ndim=2)
  d.ten_velocity = wp.array(tile(mjd.ten_velocity), dtype=wp.float32, ndim=2)
  d.actuator_velocity = wp.array(tile(mjd.actuator_velocity), dtype=wp.float32, ndim=2)
  d.actuator_force = wp.array(tile(mjd.actuator_force), dtype=wp.float32, ndim=2)
  d.actuator_length = wp.array(tile(mjd.actuator_length), dtype=wp.float32, ndim=2)
  d.actuator_moment = wp.array(tile(actuator_moment), dtype=wp.float32, ndim=3)
  d.cvel = wp.array(tile(mjd.cvel), dtype=wp.spatial_vector, ndim=2)
  d.cdof_dot = wp.array(tile(mjd.cdof_dot), dtype=wp.spatial_vector, ndim=2)
  d.qfrc_bias = wp.array(tile(mjd.qfrc_bias), dtype=wp.float32, ndim=2)
  d.qfrc_passive = wp.array(tile(mjd.qfrc_passive), dtype=wp.float32, ndim=2)
  d.subtree_linvel = wp.array(tile(mjd.subtree_linvel), dtype=wp.vec3, ndim=2)
  d.subtree_angmom = wp.array(tile(mjd.subtree_angmom), dtype=wp.vec3, ndim=2)
  d.subtree_bodyvel = wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector)
  d.qfrc_spring = wp.array(tile(mjd.qfrc_spring), dtype=wp.float32, ndim=2)
  d.qfrc_damper = wp.array(tile(mjd.qfrc_damper), dtype=wp.float32, ndim=2)
  d.qfrc_actuator = wp.array(tile(mjd.qfrc_actuator), dtype=wp.float32, ndim=2)
  d.qfrc_smooth = wp.array(tile(mjd.qfrc_smooth), dtype=wp.float32, ndim=2)
  d.qfrc_constraint = wp.array(tile(mjd.qfrc_constraint), dtype=wp.float32, ndim=2)
  d.qacc_smooth = wp.array(tile(mjd.qacc_smooth), dtype=wp.float32, ndim=2)
  d.act = wp.array(tile(mjd.act), dtype=wp.float32, ndim=2)
  d.act_dot = wp.array(tile(mjd.act_dot), dtype=wp.float32, ndim=2)

  nefc = mjd.nefc
  efc_worldid = np.zeros(njmax, dtype=int)

  for i in range(nworld):
    efc_worldid[i * nefc : (i + 1) * nefc] = i

  nefc_fill = njmax - nworld * nefc

  efc_J_fill = np.vstack(
    [np.repeat(efc_J, nworld, axis=0), np.zeros((nefc_fill, mjm.nv))]
  )
  efc_D_fill = np.concatenate(
    [np.repeat(mjd.efc_D, nworld, axis=0), np.zeros(nefc_fill)]
  )
  efc_pos_fill = np.concatenate(
    [np.repeat(mjd.efc_pos, nworld, axis=0), np.zeros(nefc_fill)]
  )
  efc_aref_fill = np.concatenate(
    [np.repeat(mjd.efc_aref, nworld, axis=0), np.zeros(nefc_fill)]
  )
  efc_frictionloss_fill = np.concatenate(
    [np.repeat(mjd.efc_frictionloss, nworld, axis=0), np.zeros(nefc_fill)]
  )
  efc_force_fill = np.concatenate(
    [np.repeat(mjd.efc_force, nworld, axis=0), np.zeros(nefc_fill)]
  )
  efc_margin_fill = np.concatenate(
    [np.repeat(mjd.efc_margin, nworld, axis=0), np.zeros(nefc_fill)]
  )
  efc_id_fill = np.concatenate(
    [np.repeat(mjd.efc_id, nworld, axis=0), np.zeros(nefc_fill)]
  )

  ncon = mjd.ncon
  condim_max = np.max(mjm.geom_condim)
  con_efc_address = np.zeros((nconmax, condim_max), dtype=int)
  for i in range(nworld):
    for j in range(ncon):
      condim = mjd.contact.dim[j]
      for k in range(condim):
        con_efc_address[i * ncon + j, k] = mjd.nefc * i + mjd.contact.efc_address[j] + k

  con_worldid = np.zeros(nconmax, dtype=int)
  for i in range(nworld):
    con_worldid[i * ncon : (i + 1) * ncon] = i

  ncon_fill = nconmax - nworld * ncon

  con_dist_fill = np.concatenate(
    [np.repeat(mjd.contact.dist, nworld, axis=0), np.zeros(ncon_fill)]
  )
  con_pos_fill = np.vstack(
    [np.repeat(mjd.contact.pos, nworld, axis=0), np.zeros((ncon_fill, 3))]
  )
  con_frame_fill = np.vstack(
    [np.repeat(mjd.contact.frame, nworld, axis=0), np.zeros((ncon_fill, 9))]
  )
  con_includemargin_fill = np.concatenate(
    [np.repeat(mjd.contact.includemargin, nworld, axis=0), np.zeros(ncon_fill)]
  )
  con_friction_fill = np.vstack(
    [np.repeat(mjd.contact.friction, nworld, axis=0), np.zeros((ncon_fill, 5))]
  )
  con_solref_fill = np.vstack(
    [np.repeat(mjd.contact.solref, nworld, axis=0), np.zeros((ncon_fill, 2))]
  )
  con_solreffriction_fill = np.vstack(
    [np.repeat(mjd.contact.solreffriction, nworld, axis=0), np.zeros((ncon_fill, 2))]
  )
  con_solimp_fill = np.vstack(
    [np.repeat(mjd.contact.solimp, nworld, axis=0), np.zeros((ncon_fill, 5))]
  )
  con_dim_fill = np.concatenate(
    [np.repeat(mjd.contact.dim, nworld, axis=0), np.zeros(ncon_fill)]
  )
  con_geom_fill = np.vstack(
    [np.repeat(mjd.contact.geom, nworld, axis=0), np.zeros((ncon_fill, 2))]
  )
  con_efc_address_fill = np.vstack([con_efc_address, np.zeros((ncon_fill, condim_max))])

  d.contact.dist = wp.array(con_dist_fill, dtype=wp.float32, ndim=1)
  d.contact.pos = wp.array(con_pos_fill, dtype=wp.vec3f, ndim=1)
  d.contact.frame = wp.array(con_frame_fill, dtype=wp.mat33f, ndim=1)
  d.contact.includemargin = wp.array(con_includemargin_fill, dtype=wp.float32, ndim=1)
  d.contact.friction = wp.array(con_friction_fill, dtype=types.vec5, ndim=1)
  d.contact.solref = wp.array(con_solref_fill, dtype=wp.vec2f, ndim=1)
  d.contact.solreffriction = wp.array(con_solreffriction_fill, dtype=wp.vec2f, ndim=1)
  d.contact.solimp = wp.array(con_solimp_fill, dtype=types.vec5, ndim=1)
  d.contact.dim = wp.array(con_dim_fill, dtype=wp.int32, ndim=1)
  d.contact.geom = wp.array(con_geom_fill, dtype=wp.vec2i, ndim=1)
  d.contact.efc_address = wp.array(con_efc_address_fill, dtype=wp.int32, ndim=2)
  d.contact.worldid = wp.array(con_worldid, dtype=wp.int32, ndim=1)

  d.efc = _constraint(mjm, d.nworld, d.nconmax, d.njmax)
  d.efc.J = wp.array(efc_J_fill, dtype=wp.float32, ndim=2)
  d.efc.D = wp.array(efc_D_fill, dtype=wp.float32, ndim=1)
  d.efc.pos = wp.array(efc_pos_fill, dtype=wp.float32, ndim=1)
  d.efc.aref = wp.array(efc_aref_fill, dtype=wp.float32, ndim=1)
  d.efc.frictionloss = wp.array(efc_frictionloss_fill, dtype=wp.float32, ndim=1)
  d.efc.force = wp.array(efc_force_fill, dtype=wp.float32, ndim=1)
  d.efc.margin = wp.array(efc_margin_fill, dtype=wp.float32, ndim=1)
  d.efc.worldid = wp.from_numpy(efc_worldid, dtype=wp.int32)
  d.efc.id = wp.from_numpy(efc_id_fill, dtype=wp.int32)

  d.xfrc_applied = wp.array(tile(mjd.xfrc_applied), dtype=wp.spatial_vector, ndim=2)
  d.eq_active = wp.array(tile(mjm.eq_active0), dtype=wp.bool, ndim=2)

  # internal tmp arrays
  d.qfrc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_integration = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qM_integration = wp.zeros_like(d.qM)
  d.qLD_integration = wp.zeros_like(d.qLD)
  d.qLDiagInv_integration = wp.zeros_like(d.qLDiagInv)
  d.act_vel_integration = wp.zeros_like(d.ctrl)
  d.qpos_t0 = wp.zeros((nworld, mjm.nq), dtype=wp.float32)
  d.qvel_t0 = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.act_t0 = wp.zeros((nworld, mjm.na), dtype=wp.float32)
  d.qvel_rk = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.qacc_rk = wp.zeros((nworld, mjm.nv), dtype=wp.float32)
  d.act_dot_rk = wp.zeros((nworld, mjm.na), dtype=wp.float32)

  # broadphase sweep and prune
  d.sap_projection_lower = wp.zeros((2 * nworld, mjm.ngeom), dtype=wp.float32)
  d.sap_projection_upper = wp.zeros((nworld, mjm.ngeom), dtype=wp.float32)
  d.sap_sort_index = wp.zeros((2 * nworld, mjm.ngeom), dtype=wp.int32)
  d.sap_range = wp.zeros((nworld, mjm.ngeom), dtype=wp.int32)
  d.sap_cumulative_sum = wp.zeros(nworld * mjm.ngeom, dtype=wp.int32)
  d.sap_segment_index = wp.array([i * mjm.ngeom for i in range(nworld + 1)], dtype=int)

  # collision driver
  d.collision_pair = wp.empty(nconmax, dtype=wp.vec2i, ndim=1)
  d.collision_pairid = wp.empty(nconmax, dtype=wp.int32, ndim=1)
  d.collision_worldid = wp.empty(nconmax, dtype=wp.int32, ndim=1)
  d.ncollision = wp.zeros(1, dtype=wp.int32, ndim=1)

  # rne_postconstraint
  d.cacc = wp.array(tile(mjd.cacc), dtype=wp.spatial_vector, ndim=2)
  d.cfrc_int = wp.array(tile(mjd.cfrc_int), dtype=wp.spatial_vector, ndim=2)
  d.cfrc_ext = wp.array(tile(mjd.cfrc_ext), dtype=wp.spatial_vector, ndim=2)

  # tendon
  d.ten_length = wp.array(tile(mjd.ten_length), dtype=wp.float32, ndim=2)

  if support.is_sparse(mjm) and mjm.ntendon:
    ten_J = np.zeros((mjm.ntendon, mjm.nv))
    mujoco.mju_sparse2dense(
      ten_J,
      mjd.ten_J.reshape(-1),
      mjd.ten_J_rownnz,
      mjd.ten_J_rowadr,
      mjd.ten_J_colind.reshape(-1),
    )
  else:
    ten_J = mjd.ten_J.reshape((mjm.ntendon, mjm.nv))

  d.ten_J = wp.array(tile(ten_J), dtype=wp.float32, ndim=3)

  d.ten_wrapadr = wp.array(tile(mjd.ten_wrapadr), dtype=wp.int32, ndim=2)
  d.ten_wrapnum = wp.array(tile(mjd.ten_wrapnum), dtype=wp.int32, ndim=2)
  d.wrap_obj = wp.array(tile(mjd.wrap_obj), dtype=wp.vec2i, ndim=2)
  d.wrap_xpos = wp.array(
    tile(mjd.wrap_xpos), dtype=wp.spatial_vector, ndim=2
  )  # TODO(team): vec6?

  # sensors
  d.sensordata = wp.array(tile(mjd.sensordata), dtype=wp.float32, ndim=2)

  return d


def get_data_into(
  result: mujoco.MjData,
  mjm: mujoco.MjModel,
  d: types.Data,
):
  """Gets Data from a device into an existing mujoco.MjData."""
  if d.nworld > 1:
    raise NotImplementedError("only nworld == 1 supported for now")

  ncon = d.ncon.numpy()[0]
  nefc = d.nefc.numpy()[0]

  if ncon != result.ncon or nefc != result.nefc:
    mujoco._functions._realloc_con_efc(result, ncon=ncon, nefc=nefc)

  result.time = d.time.numpy()[0]
  result.ne = d.ne.numpy()[0]
  result.qpos[:] = d.qpos.numpy()[0]
  result.qvel[:] = d.qvel.numpy()[0]
  result.qacc_warmstart = d.qacc_warmstart.numpy()[0]
  result.qfrc_applied = d.qfrc_applied.numpy()[0]
  result.mocap_pos = d.mocap_pos.numpy()[0]
  result.mocap_quat = d.mocap_quat.numpy()[0]
  result.qacc = d.qacc.numpy()[0]
  result.xanchor = d.xanchor.numpy()[0]
  result.xaxis = d.xaxis.numpy()[0]
  result.xmat = d.xmat.numpy().reshape((-1, 9))
  result.xpos = d.xpos.numpy()[0]
  result.xquat = d.xquat.numpy()[0]
  result.xipos = d.xipos.numpy()[0]
  result.ximat = d.ximat.numpy().reshape((-1, 9))
  result.subtree_com = d.subtree_com.numpy()[0]
  result.geom_xpos = d.geom_xpos.numpy()[0]
  result.geom_xmat = d.geom_xmat.numpy().reshape((-1, 9))
  result.site_xpos = d.site_xpos.numpy()[0]
  result.site_xmat = d.site_xmat.numpy().reshape((-1, 9))
  result.cam_xpos = d.cam_xpos.numpy()[0]
  result.cam_xmat = d.cam_xmat.numpy().reshape((-1, 9))
  result.light_xpos = d.light_xpos.numpy()[0]
  result.light_xdir = d.light_xdir.numpy()[0]
  result.cinert = d.cinert.numpy()[0]
  result.cdof = d.cdof.numpy()[0]
  result.crb = d.crb.numpy()[0]
  result.qLDiagInv = d.qLDiagInv.numpy()[0]
  result.ctrl = d.ctrl.numpy()[0]
  result.ten_velocity = d.ten_velocity.numpy()[0]
  result.actuator_velocity = d.actuator_velocity.numpy()[0]
  result.actuator_force = d.actuator_force.numpy()[0]
  result.actuator_length = d.actuator_length.numpy()[0]
  mujoco.mju_dense2sparse(
    result.actuator_moment,
    d.actuator_moment.numpy()[0],
    result.moment_rownnz,
    result.moment_rowadr,
    result.moment_colind,
  )
  result.cvel = d.cvel.numpy()[0]
  result.cdof_dot = d.cdof_dot.numpy()[0]
  result.qfrc_bias = d.qfrc_bias.numpy()[0]
  result.qfrc_passive = d.qfrc_passive.numpy()[0]
  result.subtree_linvel = d.subtree_linvel.numpy()[0]
  result.subtree_angmom = d.subtree_angmom.numpy()[0]
  result.qfrc_spring = d.qfrc_spring.numpy()[0]
  result.qfrc_damper = d.qfrc_damper.numpy()[0]
  result.qfrc_actuator = d.qfrc_actuator.numpy()[0]
  result.qfrc_smooth = d.qfrc_smooth.numpy()[0]
  result.qfrc_constraint = d.qfrc_constraint.numpy()[0]
  result.qacc_smooth = d.qacc_smooth.numpy()[0]
  result.act = d.act.numpy()[0]
  result.act_dot = d.act_dot.numpy()[0]

  result.contact.dist[:] = d.contact.dist.numpy()[:ncon]
  result.contact.pos[:] = d.contact.pos.numpy()[:ncon]
  result.contact.frame[:] = d.contact.frame.numpy()[:ncon].reshape((-1, 9))
  result.contact.includemargin[:] = d.contact.includemargin.numpy()[:ncon]
  result.contact.friction[:] = d.contact.friction.numpy()[:ncon]
  result.contact.solref[:] = d.contact.solref.numpy()[:ncon]
  result.contact.solreffriction[:] = d.contact.solreffriction.numpy()[:ncon]
  result.contact.solimp[:] = d.contact.solimp.numpy()[:ncon]
  result.contact.dim[:] = d.contact.dim.numpy()[:ncon]
  result.contact.efc_address[:] = d.contact.efc_address.numpy()[:ncon, 0]

  if support.is_sparse(mjm):
    result.qM[:] = d.qM.numpy()[0, 0]
    result.qLD[:] = d.qLD.numpy()[0, 0]
    # TODO(team): set efc_J after fix to _realloc_con_efc lands
    # efc_J = d.efc_J.numpy()[0, :nefc]
    # mujoco.mju_dense2sparse(
    #   result.efc_J, efc_J, result.efc_J_rownnz, result.efc_J_rowadr, result.efc_J_colind
    # )
  else:
    qM = d.qM.numpy()
    adr = 0
    for i in range(mjm.nv):
      j = i
      while j >= 0:
        result.qM[adr] = qM[0, i, j]
        j = mjm.dof_parentid[j]
        adr += 1
    mujoco.mj_factorM(mjm, result)
    # TODO(team): set efc_J after fix to _realloc_con_efc lands
    # if nefc > 0:
    #   result.efc_J[:nefc * mjm.nv] = d.efc_J.numpy()[:nefc].flatten()
  result.xfrc_applied[:] = d.xfrc_applied.numpy()[0]
  result.eq_active[:] = d.eq_active.numpy()[0]

  result.efc_D[:] = d.efc.D.numpy()[:nefc]
  result.efc_pos[:] = d.efc.pos.numpy()[:nefc]
  result.efc_aref[:] = d.efc.aref.numpy()[:nefc]
  result.efc_force[:] = d.efc.force.numpy()[:nefc]
  result.efc_margin[:] = d.efc.margin.numpy()[:nefc]

  result.cacc[:] = d.cacc.numpy()[0]
  result.cfrc_int[:] = d.cfrc_int.numpy()[0]
  result.cfrc_ext[:] = d.cfrc_ext.numpy()[0]

  # TODO: other efc_ fields, anything else missing

  # tendon
  result.ten_length[:] = d.ten_length.numpy()[0]
  result.ten_J[:] = d.ten_J.numpy()[0]
  result.ten_wrapadr[:] = d.ten_wrapadr.numpy()[0]
  result.ten_wrapnum[:] = d.ten_wrapnum.numpy()[0]
  result.wrap_obj[:] = d.wrap_obj.numpy()[0]
  result.wrap_xpos[:] = d.wrap_xpos.numpy()[0]

  # sensors
  result.sensordata[:] = d.sensordata.numpy()
