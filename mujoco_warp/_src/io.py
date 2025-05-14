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

from . import types


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

  if mjm.nplugin > 0:
    raise NotImplementedError("Plugins are unsupported.")

  if mjm.nflex > 0:
    raise NotImplementedError("Flex is unsupported.")

  if mjm.tendon_frictionloss.any():
    raise NotImplementedError("Tendon frictionloss is unsupported.")

  if mjm.geom_fluid.any():
    raise NotImplementedError("Ellipsoid fluid model not implemented.")

  # check options
  for opt, opt_types, msg in (
    (mjm.opt.integrator, types.IntegratorType, "Integrator"),
    (mjm.opt.cone, types.ConeType, "Cone"),
    (mjm.opt.solver, types.SolverType, "Solver"),
  ):
    if opt not in set(opt_types):
      raise NotImplementedError(f"{msg} {opt} is unsupported.")

  # TODO(team): remove after solver._update_gradient for Newton solver utilizes tile operations for islands
  nv_max = 60
  if mjm.nv > nv_max and mjm.opt.jacobian == mujoco.mjtJacobian.mjJAC_DENSE:
    raise ValueError(f"Dense is unsupported for nv > {nv_max} (nv = {mjm.nv}).")

  # calculate some fields that cannot be easily computed inline
  nlsp = mjm.opt.ls_iterations  # TODO(team): how to set nlsp?

  # unfortunately we must create Data in order to get some model fields like M_rownnz
  mjd = mujoco.MjData(mjm)

  # dof lower triangle row and column indices (used in solver)
  dof_tri_row, dof_tri_col = np.tril_indices(mjm.nv)

  # indices for sparse qM_fullm (used in solver)
  qM_fullm_i, qM_fullm_j = [], []
  for i in range(mjm.nv):
    j = i
    while j > -1:
      qM_fullm_i.append(i)
      qM_fullm_j.append(j)
      j = mjm.dof_parentid[j]

  # indices for sparse qM mul_m (used in support)
  qM_mulm_i, qM_mulm_j, qM_madr_ij = [], [], []
  for i in range(mjm.nv):
    madr_ij, j = mjm.dof_Madr[i], i

    while True:
      madr_ij, j = madr_ij + 1, mjm.dof_parentid[j]
      if j == -1:
        break
      qM_mulm_i.append(i)
      qM_mulm_j.append(j)
      qM_madr_ij.append(madr_ij)

  # body_tree is a list of body ids grouped by tree level
  bodies, body_depth = {}, np.zeros(mjm.nbody, dtype=int) - 1
  for i in range(mjm.nbody):
    body_depth[i] = body_depth[mjm.body_parentid[i]] + 1
    bodies.setdefault(body_depth[i], []).append(i)
  body_tree = tuple(wp.array(bodies[i], dtype=int) for i in sorted(bodies))

  # qLD_updates has dof tree ordering of qLD updates for sparse factor m
  qLD_updates, dof_depth = {}, np.zeros(mjm.nv, dtype=int) - 1

  for k in range(mjm.nv):
    dof_depth[k] = dof_depth[mjm.dof_parentid[k]] + 1
    i = mjm.dof_parentid[k]
    diag_k = mjd.M_rowadr[k] + mjd.M_rownnz[k] - 1
    Madr_ki = diag_k - 1
    while i > -1:
      qLD_updates.setdefault(dof_depth[i], []).append((i, k, Madr_ki))
      i = mjm.dof_parentid[i]
      Madr_ki -= 1

  qLD_updates = tuple(wp.array(qLD_updates[i], dtype=wp.vec3i) for i in sorted(qLD_updates))

  # qM_tiles records the block diagonal structure of qM
  tile_corners = [i for i in range(mjm.nv) if mjm.dof_parentid[i] == -1]
  tiles = {}
  for i in range(len(tile_corners)):
    tile_beg = tile_corners[i]
    tile_end = mjm.nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
    tiles.setdefault(tile_end - tile_beg, []).append(tile_beg)

  qM_tiles = tuple(types.TileSet(adr=wp.array(tiles[sz], dtype=int), size=sz) for sz in sorted(tiles.keys()))

  # subtree_mass is a precalculated array used in smooth
  subtree_mass = np.copy(mjm.body_mass)
  # TODO(team): should this be [mjm.nbody - 1, 0) ?
  for i in range(mjm.nbody - 1, -1, -1):
    subtree_mass[mjm.body_parentid[i]] += subtree_mass[i]

  # actuator_moment tiles are grouped by dof size and number of actuators
  tree_id = np.arange(len(tile_corners), dtype=np.int32)
  num_trees = int(np.max(tree_id)) if len(tree_id) > 0 else 0
  bodyid = []
  for i in range(mjm.nu):
    trntype = mjm.actuator_trntype[i]
    if trntype == mujoco.mjtTrn.mjTRN_JOINT or trntype == mujoco.mjtTrn.mjTRN_JOINTINPARENT:
      jntid = mjm.actuator_trnid[i, 0]
      bodyid.append(mjm.jnt_bodyid[jntid])
    elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
      tenid = mjm.actuator_trnid[i, 0]
      adr = mjm.tendon_adr[tenid]
      if mjm.wrap_type[adr] == mujoco.mjtWrap.mjWRAP_JOINT:
        ten_num = mjm.tendon_num[tenid]
        for i in range(ten_num):
          bodyid.append(mjm.jnt_bodyid[mjm.wrap_objid[adr + i]])
      else:
        for i in range(mjm.nv):
          bodyid.append(mjm.dof_bodyid[i])
    else:
      raise NotImplementedError(f"Transmission type {trntype} not implemented.")
  tree = mjm.body_treeid[np.array(bodyid, dtype=int)]
  counts, ids = np.histogram(tree, bins=np.arange(0, num_trees + 2))
  acts_per_tree = dict(zip(ids, counts))

  tiles = {}
  act_beg = 0
  for i in range(len(tile_corners)):
    tile_beg = tile_corners[i]
    tile_end = mjm.nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
    tree = int(tree_id[i])
    act_num = acts_per_tree[tree]
    tiles.setdefault((tile_end - tile_beg, act_num), []).append((tile_beg, act_beg))
    act_beg += act_num

  actuator_moment_tiles_nv, actuator_moment_tiles_nu = tuple(), tuple()

  for (nv, nu), adr in sorted(tiles.items()):
    adr_nv = wp.array([nv for nv, _ in adr], dtype=int)
    adr_nu = wp.array([nu for _, nu in adr], dtype=int)
    actuator_moment_tiles_nv += (types.TileSet(adr=adr_nv, size=nv),)
    actuator_moment_tiles_nu += (types.TileSet(adr=adr_nu, size=nu),)

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

  wrap_site_adr = np.nonzero(mjm.wrap_type == mujoco.mjtWrap.mjWRAP_SITE)[0]
  wrap_site_pair_adr = np.setdiff1d(wrap_site_adr[np.nonzero(np.diff(wrap_site_adr) == 1)[0]], mjm.tendon_adr[1:] - 1)

  # mocap
  mocap_bodyid = np.arange(mjm.nbody)[mjm.body_mocapid >= 0]
  mocap_bodyid = mocap_bodyid[mjm.body_mocapid[mjm.body_mocapid >= 0].argsort()]

  # precalculated geom pairs
  filterparent = not (mjm.opt.disableflags & types.DisableBit.FILTERPARENT.value)
  exclude_signature = set(mjm.exclude_signature)
  predefined_pairs = {(mjm.pair_geom1[i], mjm.pair_geom2[i]): i for i in range(mjm.npair)}

  nxn_geom_pair, nxn_pairid = [], []
  for geom1, geom2 in zip(*np.triu_indices(mjm.ngeom, k=1)):  # k=1 skip diagonal
    bodyid1, bodyid2 = mjm.geom_bodyid[geom1], mjm.geom_bodyid[geom2]
    contype1, contype2 = mjm.geom_contype[geom1], mjm.geom_contype[geom2]
    conaffinity1 = mjm.geom_conaffinity[geom1]
    conaffinity2 = mjm.geom_conaffinity[geom2]
    weldid1, weldid2 = mjm.body_weldid[bodyid1], mjm.body_weldid[bodyid2]
    weld_parentid1 = mjm.body_weldid[mjm.body_parentid[weldid1]]
    weld_parentid2 = mjm.body_weldid[mjm.body_parentid[weldid2]]

    self_collision = weldid1 == weldid2
    parent_child_collision = (
      filterparent and (weldid1 != 0) and (weldid2 != 0) and ((weldid1 == weld_parentid2) or (weldid2 == weld_parentid1))
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

    nxn_geom_pair.append((geom1, geom2))
    nxn_pairid.append(pairid)

  def create_nmodel_batched_array(mjm_array, dtype):
    array = wp.array(mjm_array, dtype=dtype)
    array.ndim += 1
    array.shape = (1,) + array.shape
    array.strides = (0,) + array.strides
    return array

  m = types.Model(
    nq=mjm.nq,
    nv=mjm.nv,
    nu=mjm.nu,
    na=mjm.na,
    nbody=mjm.nbody,
    njnt=mjm.njnt,
    ngeom=mjm.ngeom,
    nsite=mjm.nsite,
    ncam=mjm.ncam,
    nlight=mjm.nlight,
    nexclude=mjm.nexclude,
    neq=mjm.nM,
    nmocap=mjm.nmocap,
    ngravcomp=mjm.ngravcomp,
    nM=mjm.nM,
    ntendon=mjm.ntendon,
    nwrap=mjm.nwrap,
    nsensor=mjm.nsensor,
    nsensordata=mjm.nsensordata,
    nmeshvert=mjm.nmeshvert,
    nmeshface=mjm.nmeshface,
    nlsp=nlsp,
    npair=mjm.npair,
    opt=types.Option(
      timestep=mjm.opt.timestep,
      tolerance=mjm.opt.tolerance,
      ls_tolerance=mjm.opt.ls_tolerance,
      gravity=wp.vec3(mjm.opt.gravity),
      wind=wp.vec3(mjm.opt.wind[0], mjm.opt.wind[1], mjm.opt.wind[2]),
      density=mjm.opt.density,
      viscosity=mjm.opt.viscosity,
      cone=mjm.opt.cone,
      solver=mjm.opt.solver,
      iterations=mjm.opt.iterations,
      ls_iterations=mjm.opt.ls_iterations,
      integrator=mjm.opt.integrator,
      disableflags=mjm.opt.disableflags,
      impratio=mjm.opt.impratio,
      is_sparse=mujoco.mj_isSparse(mjm),
      ls_parallel=False,
      gjk_iterations=1,
      epa_iterations=12,
      epa_exact_neg_distance=wp.bool(False),
      depth_extension=0.1,
    ),
    stat=types.Statistic(
      meaninertia=mjm.stat.meaninertia,
    ),
    qpos0=create_nmodel_batched_array(mjm.qpos0, dtype=float),
    qpos_spring=create_nmodel_batched_array(mjm.qpos_spring, dtype=float),
    qM_fullm_i=wp.array(qM_fullm_i, dtype=int),
    qM_fullm_j=wp.array(qM_fullm_j, dtype=int),
    qM_mulm_i=wp.array(qM_mulm_i, dtype=int),
    qM_mulm_j=wp.array(qM_mulm_j, dtype=int),
    qM_madr_ij=wp.array(qM_madr_ij, dtype=int),
    qLD_updates=qLD_updates,
    M_rownnz=wp.array(mjd.M_rownnz, dtype=int),
    M_rowadr=wp.array(mjd.M_rowadr, dtype=int),
    M_colind=wp.array(mjd.M_colind, dtype=int),
    mapM2M=wp.array(mjd.mapM2M, dtype=int),
    qM_tiles=qM_tiles,
    body_tree=body_tree,
    body_parentid=wp.array(mjm.body_parentid, dtype=int),
    body_rootid=wp.array(mjm.body_rootid, dtype=int),
    body_weldid=wp.array(mjm.body_weldid, dtype=int),
    body_mocapid=wp.array(mjm.body_mocapid, dtype=int),
    mocap_bodyid=wp.array(mocap_bodyid, dtype=int),
    body_jntnum=wp.array(mjm.body_jntnum, dtype=int),
    body_jntadr=wp.array(mjm.body_jntadr, dtype=int),
    body_dofnum=wp.array(mjm.body_dofnum, dtype=int),
    body_dofadr=wp.array(mjm.body_dofadr, dtype=int),
    body_geomnum=wp.array(mjm.body_geomnum, dtype=int),
    body_geomadr=wp.array(mjm.body_geomadr, dtype=int),
    body_pos=create_nmodel_batched_array(mjm.body_pos, dtype=wp.vec3),
    body_quat=create_nmodel_batched_array(mjm.body_quat, dtype=wp.quat),
    body_ipos=create_nmodel_batched_array(mjm.body_ipos, dtype=wp.vec3),
    body_iquat=create_nmodel_batched_array(mjm.body_iquat, dtype=wp.quat),
    body_mass=create_nmodel_batched_array(mjm.body_mass, dtype=float),
    body_subtreemass=create_nmodel_batched_array(mjm.body_subtreemass, dtype=float),
    subtree_mass=create_nmodel_batched_array(subtree_mass, dtype=float),
    body_inertia=create_nmodel_batched_array(mjm.body_inertia, dtype=wp.vec3),
    body_invweight0=create_nmodel_batched_array(mjm.body_invweight0, dtype=float),
    body_contype=wp.array(mjm.body_contype, dtype=int),
    body_conaffinity=wp.array(mjm.body_conaffinity, dtype=int),
    body_gravcomp=create_nmodel_batched_array(mjm.body_gravcomp, dtype=float),
    jnt_type=wp.array(mjm.jnt_type, dtype=int),
    jnt_qposadr=wp.array(mjm.jnt_qposadr, dtype=int),
    jnt_dofadr=wp.array(mjm.jnt_dofadr, dtype=int),
    jnt_bodyid=wp.array(mjm.jnt_bodyid, dtype=int),
    jnt_limited=wp.array(mjm.jnt_limited, dtype=int),
    jnt_actfrclimited=wp.array(mjm.jnt_actfrclimited, dtype=bool),
    jnt_solref=create_nmodel_batched_array(mjm.jnt_solref, dtype=wp.vec2),
    jnt_solimp=create_nmodel_batched_array(mjm.jnt_solimp, dtype=types.vec5),
    jnt_pos=create_nmodel_batched_array(mjm.jnt_pos, dtype=wp.vec3),
    jnt_axis=create_nmodel_batched_array(mjm.jnt_axis, dtype=wp.vec3),
    jnt_stiffness=create_nmodel_batched_array(mjm.jnt_stiffness, dtype=float),
    jnt_range=create_nmodel_batched_array(mjm.jnt_range, dtype=float),
    jnt_actfrcrange=create_nmodel_batched_array(mjm.jnt_actfrcrange, dtype=wp.vec2),
    jnt_margin=create_nmodel_batched_array(mjm.jnt_margin, dtype=float),
    # these jnt_limited adrs are used in constraint.py
    jnt_limited_slide_hinge_adr=wp.array(
      np.nonzero(
        mjm.jnt_limited & ((mjm.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE) | (mjm.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
      )[0],
      dtype=int,
    ),
    jnt_limited_ball_adr=wp.array(
      np.nonzero(mjm.jnt_limited & (mjm.jnt_type == mujoco.mjtJoint.mjJNT_BALL))[0],
      dtype=int,
    ),
    jnt_actgravcomp=wp.array(mjm.jnt_actgravcomp, dtype=int),
    dof_bodyid=wp.array(mjm.dof_bodyid, dtype=int),
    dof_jntid=wp.array(mjm.dof_jntid, dtype=int),
    dof_parentid=wp.array(mjm.dof_parentid, dtype=int),
    dof_Madr=wp.array(mjm.dof_Madr, dtype=int),
    dof_armature=create_nmodel_batched_array(mjm.dof_armature, dtype=float),
    dof_damping=create_nmodel_batched_array(mjm.dof_damping, dtype=float),
    dof_invweight0=create_nmodel_batched_array(mjm.dof_invweight0, dtype=float),
    dof_frictionloss=create_nmodel_batched_array(mjm.dof_frictionloss, dtype=float),
    dof_solimp=create_nmodel_batched_array(mjm.dof_solimp, dtype=types.vec5),
    dof_solref=create_nmodel_batched_array(mjm.dof_solref, dtype=wp.vec2),
    dof_tri_row=wp.array(dof_tri_row, dtype=int),
    dof_tri_col=wp.array(dof_tri_col, dtype=int),
    geom_type=wp.array(mjm.geom_type, dtype=int),
    geom_contype=wp.array(mjm.geom_contype, dtype=int),
    geom_conaffinity=wp.array(mjm.geom_conaffinity, dtype=int),
    geom_condim=wp.array(mjm.geom_condim, dtype=int),
    geom_bodyid=wp.array(mjm.geom_bodyid, dtype=int),
    geom_dataid=wp.array(mjm.geom_dataid, dtype=int),
    geom_group=wp.array(mjm.geom_group, dtype=int),
    geom_matid=create_nmodel_batched_array(mjm.geom_matid, dtype=int),
    geom_priority=wp.array(mjm.geom_priority, dtype=int),
    geom_solmix=create_nmodel_batched_array(mjm.geom_solmix, dtype=float),
    geom_solref=create_nmodel_batched_array(mjm.geom_solref, dtype=wp.vec2),
    geom_solimp=create_nmodel_batched_array(mjm.geom_solimp, dtype=types.vec5),
    geom_size=create_nmodel_batched_array(mjm.geom_size, dtype=wp.vec3),
    geom_aabb=wp.array(mjm.geom_aabb, dtype=wp.vec3),
    geom_rbound=create_nmodel_batched_array(mjm.geom_rbound, dtype=float),
    geom_pos=create_nmodel_batched_array(mjm.geom_pos, dtype=wp.vec3),
    geom_quat=create_nmodel_batched_array(mjm.geom_quat, dtype=wp.quat),
    geom_friction=create_nmodel_batched_array(mjm.geom_friction, dtype=wp.vec3),
    geom_margin=create_nmodel_batched_array(mjm.geom_margin, dtype=float),
    geom_gap=create_nmodel_batched_array(mjm.geom_gap, dtype=float),
    geom_rgba=create_nmodel_batched_array(mjm.geom_rgba, dtype=wp.vec4),
    site_bodyid=wp.array(mjm.site_bodyid, dtype=int),
    site_pos=create_nmodel_batched_array(mjm.site_pos, dtype=wp.vec3),
    site_quat=create_nmodel_batched_array(mjm.site_quat, dtype=wp.quat),
    cam_mode=wp.array(mjm.cam_mode, dtype=int),
    cam_bodyid=wp.array(mjm.cam_bodyid, dtype=int),
    cam_targetbodyid=wp.array(mjm.cam_targetbodyid, dtype=int),
    cam_pos=create_nmodel_batched_array(mjm.cam_pos, dtype=wp.vec3),
    cam_quat=create_nmodel_batched_array(mjm.cam_quat, dtype=wp.quat),
    cam_poscom0=create_nmodel_batched_array(mjm.cam_poscom0, dtype=wp.vec3),
    cam_pos0=create_nmodel_batched_array(mjm.cam_pos0, dtype=wp.vec3),
    cam_fovy=wp.array(mjm.cam_fovy, dtype=float),
    cam_resolution=wp.array(mjm.cam_resolution, dtype=wp.vec2i),
    cam_sensorsize=wp.array(mjm.cam_sensorsize, dtype=wp.vec2),
    cam_intrinsic=wp.array(mjm.cam_intrinsic, dtype=wp.vec4),
    light_mode=wp.array(mjm.light_mode, dtype=int),
    light_bodyid=wp.array(mjm.light_bodyid, dtype=int),
    light_targetbodyid=wp.array(mjm.light_targetbodyid, dtype=int),
    light_pos=create_nmodel_batched_array(mjm.light_pos, dtype=wp.vec3),
    light_dir=create_nmodel_batched_array(mjm.light_dir, dtype=wp.vec3),
    light_poscom0=create_nmodel_batched_array(mjm.light_poscom0, dtype=wp.vec3),
    light_pos0=create_nmodel_batched_array(mjm.light_pos0, dtype=wp.vec3),
    mesh_vertadr=wp.array(mjm.mesh_vertadr, dtype=int),
    mesh_vertnum=wp.array(mjm.mesh_vertnum, dtype=int),
    mesh_vert=wp.array(mjm.mesh_vert, dtype=wp.vec3),
    mesh_faceadr=wp.array(mjm.mesh_faceadr, dtype=int),
    mesh_face=wp.array(mjm.mesh_face, dtype=wp.vec3i),
    eq_type=wp.array(mjm.eq_type, dtype=int),
    eq_obj1id=wp.array(mjm.eq_obj1id, dtype=int),
    eq_obj2id=wp.array(mjm.eq_obj2id, dtype=int),
    eq_objtype=wp.array(mjm.eq_objtype, dtype=int),
    eq_active0=wp.array(mjm.eq_active0, dtype=bool),
    eq_solref=create_nmodel_batched_array(mjm.eq_solref, dtype=wp.vec2),
    eq_solimp=create_nmodel_batched_array(mjm.eq_solimp, dtype=types.vec5),
    eq_data=create_nmodel_batched_array(mjm.eq_data, dtype=types.vec11),
    # pre-compute indices of equality constraints
    eq_connect_adr=wp.array(np.nonzero(mjm.eq_type == types.EqType.CONNECT.value)[0], dtype=int),
    eq_wld_adr=wp.array(np.nonzero(mjm.eq_type == types.EqType.WELD.value)[0], dtype=int),
    eq_jnt_adr=wp.array(np.nonzero(mjm.eq_type == types.EqType.JOINT.value)[0], dtype=int),
    eq_ten_adr=wp.array(np.nonzero(mjm.eq_type == types.EqType.TENDON.value)[0], dtype=int),
    actuator_moment_tiles_nv=actuator_moment_tiles_nv,
    actuator_moment_tiles_nu=actuator_moment_tiles_nu,
    actuator_trntype=wp.array(mjm.actuator_trntype, dtype=int),
    actuator_dyntype=wp.array(mjm.actuator_dyntype, dtype=int),
    actuator_gaintype=wp.array(mjm.actuator_gaintype, dtype=int),
    actuator_biastype=wp.array(mjm.actuator_biastype, dtype=int),
    actuator_trnid=wp.array(mjm.actuator_trnid, dtype=wp.vec2i),
    actuator_actadr=wp.array(mjm.actuator_actadr, dtype=int),
    actuator_actnum=wp.array(mjm.actuator_actnum, dtype=int),
    actuator_ctrllimited=wp.array(mjm.actuator_ctrllimited, dtype=bool),
    actuator_forcelimited=wp.array(mjm.actuator_forcelimited, dtype=bool),
    actuator_actlimited=wp.array(mjm.actuator_actlimited, dtype=bool),
    actuator_dynprm=create_nmodel_batched_array(mjm.actuator_dynprm, dtype=types.vec10f),
    actuator_gainprm=create_nmodel_batched_array(mjm.actuator_gainprm, dtype=types.vec10f),
    actuator_biasprm=create_nmodel_batched_array(mjm.actuator_biasprm, dtype=types.vec10f),
    actuator_ctrlrange=create_nmodel_batched_array(mjm.actuator_ctrlrange, dtype=wp.vec2),
    actuator_forcerange=create_nmodel_batched_array(mjm.actuator_forcerange, dtype=wp.vec2),
    actuator_actrange=create_nmodel_batched_array(mjm.actuator_actrange, dtype=wp.vec2),
    actuator_gear=create_nmodel_batched_array(mjm.actuator_gear, dtype=wp.spatial_vector),
    exclude_signature=wp.array(mjm.exclude_signature, dtype=int),
    # short-circuiting here allows us to skip a lot of code in implicit integration
    actuator_affine_bias_gain=bool(
      np.any(mjm.actuator_biastype == types.BiasType.AFFINE.value)
      or np.any(mjm.actuator_gaintype == types.GainType.AFFINE.value)
    ),
    nxn_geom_pair=wp.array(nxn_geom_pair, dtype=wp.vec2i),
    nxn_pairid=wp.array(nxn_pairid, dtype=int),
    pair_dim=wp.array(mjm.pair_dim, dtype=int),
    pair_geom1=wp.array(mjm.pair_geom1, dtype=int),
    pair_geom2=wp.array(mjm.pair_geom2, dtype=int),
    pair_solref=create_nmodel_batched_array(mjm.pair_solref, dtype=wp.vec2),
    pair_solreffriction=create_nmodel_batched_array(mjm.pair_solreffriction, dtype=wp.vec2),
    pair_solimp=create_nmodel_batched_array(mjm.pair_solimp, dtype=types.vec5),
    pair_margin=create_nmodel_batched_array(mjm.pair_margin, dtype=float),
    pair_gap=create_nmodel_batched_array(mjm.pair_gap, dtype=float),
    pair_friction=create_nmodel_batched_array(mjm.pair_friction, dtype=types.vec5),
    condim_max=np.max(mjm.pair_dim) if mjm.npair else np.max(mjm.geom_condim),  # TODO(team): get max after filtering,
    tendon_adr=wp.array(mjm.tendon_adr, dtype=int),
    tendon_num=wp.array(mjm.tendon_num, dtype=int),
    tendon_limited=wp.array(mjm.tendon_limited, dtype=int),
    tendon_limited_adr=wp.array(np.nonzero(mjm.tendon_limited)[0], dtype=wp.int32, ndim=1),
    tendon_solref_lim=create_nmodel_batched_array(mjm.tendon_solref_lim, dtype=wp.vec2f),
    tendon_solimp_lim=create_nmodel_batched_array(mjm.tendon_solimp_lim, dtype=types.vec5),
    tendon_range=create_nmodel_batched_array(mjm.tendon_range, dtype=wp.vec2f),
    tendon_margin=create_nmodel_batched_array(mjm.tendon_margin, dtype=float),
    tendon_length0=create_nmodel_batched_array(mjm.tendon_length0, dtype=float),
    tendon_invweight0=create_nmodel_batched_array(mjm.tendon_invweight0, dtype=float),
    wrap_objid=wp.array(mjm.wrap_objid, dtype=int),
    wrap_prm=wp.array(mjm.wrap_prm, dtype=float),
    wrap_type=wp.array(mjm.wrap_type, dtype=int),
    tendon_jnt_adr=wp.array(tendon_jnt_adr, dtype=int),
    tendon_site_adr=wp.array(tendon_site_adr, dtype=int),
    tendon_site_pair_adr=wp.array(tendon_site_pair_adr, dtype=int),
    ten_wrapadr_site=wp.array(ten_wrapadr_site, dtype=int),
    ten_wrapnum_site=wp.array(ten_wrapnum_site, dtype=int),
    wrap_jnt_adr=wp.array(wrap_jnt_adr, dtype=int),
    wrap_site_adr=wp.array(wrap_site_adr, dtype=int),
    wrap_site_pair_adr=wp.array(wrap_site_pair_adr, dtype=int),
    sensor_type=wp.array(mjm.sensor_type, dtype=int),
    sensor_datatype=wp.array(mjm.sensor_datatype, dtype=int),
    sensor_objtype=wp.array(mjm.sensor_objtype, dtype=int),
    sensor_objid=wp.array(mjm.sensor_objid, dtype=int),
    sensor_reftype=wp.array(mjm.sensor_reftype, dtype=int),
    sensor_refid=wp.array(mjm.sensor_refid, dtype=int),
    sensor_dim=wp.array(mjm.sensor_dim, dtype=int),
    sensor_adr=wp.array(mjm.sensor_adr, dtype=int),
    sensor_cutoff=wp.array(mjm.sensor_cutoff, dtype=float),
    sensor_pos_adr=wp.array(
      np.nonzero(mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_POS)[0],
      dtype=int,
    ),
    sensor_vel_adr=wp.array(
      np.nonzero(mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_VEL)[0],
      dtype=int,
    ),
    sensor_acc_adr=wp.array(
      np.nonzero(mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_ACC)[0],
      dtype=int,
    ),
    sensor_subtree_vel=np.isin(
      mjm.sensor_type,
      [mujoco.mjtSensor.mjSENS_SUBTREELINVEL, mujoco.mjtSensor.mjSENS_SUBTREEANGMOM],
    ).any(),
    sensor_rne_postconstraint=np.isin(
      mjm.sensor_type,
      [
        mujoco.mjtSensor.mjSENS_ACCELEROMETER,
        mujoco.mjtSensor.mjSENS_FORCE,
        mujoco.mjtSensor.mjSENS_TORQUE,
        mujoco.mjtSensor.mjSENS_FRAMELINACC,
        mujoco.mjtSensor.mjSENS_FRAMEANGACC,
      ],
    ).any(),
    mat_rgba=create_nmodel_batched_array(mjm.mat_rgba, dtype=wp.vec4),
  )

  return m


def make_data(mjm: mujoco.MjModel, nworld: int = 1, nconmax: int = -1, njmax: int = -1) -> types.Data:
  # TODO(team): move to Model?
  if nconmax == -1:
    # TODO(team): heuristic for nconmax
    nconmax = nworld * 20
  if njmax == -1:
    # TODO(team): heuristic for njmax
    njmax = nworld * 20 * 6

  if mujoco.mj_isSparse(mjm):
    qM = wp.zeros((nworld, 1, mjm.nM), dtype=float)
    qLD = wp.zeros((nworld, 1, mjm.nM), dtype=float)
  else:
    qM = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float)
    qLD = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float)

  return types.Data(
    nworld=nworld,
    nconmax=nconmax,
    njmax=njmax,
    ncon=wp.zeros(1, dtype=int),
    ne=wp.zeros(1, dtype=int),
    ne_connect=wp.zeros(1, dtype=int),  # warp only
    ne_weld=wp.zeros(1, dtype=int),  # warp only
    ne_jnt=wp.zeros(1, dtype=int),  # warp only
    ne_ten=wp.zeros(1, dtype=int),  # warp only
    nf=wp.zeros(1, dtype=int),
    nl=wp.zeros(1, dtype=int),
    nefc=wp.zeros(1, dtype=int),
    time=wp.zeros(nworld, dtype=float),
    qpos=wp.zeros((nworld, mjm.nq), dtype=float),
    qvel=wp.zeros((nworld, mjm.nv), dtype=float),
    act=wp.zeros((nworld, mjm.na), dtype=float),
    qacc_warmstart=wp.zeros((nworld, mjm.nv), dtype=float),
    ctrl=wp.zeros((nworld, mjm.nu), dtype=float),
    qfrc_applied=wp.zeros((nworld, mjm.nv), dtype=float),
    xfrc_applied=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    fluid_applied=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    eq_active=wp.array(np.tile(mjm.eq_active0, (nworld, 1)), dtype=bool),
    mocap_pos=wp.zeros((nworld, mjm.nmocap), dtype=wp.vec3),
    mocap_quat=wp.zeros((nworld, mjm.nmocap), dtype=wp.quat),
    qacc=wp.zeros((nworld, mjm.nv), dtype=float),
    act_dot=wp.zeros((nworld, mjm.na), dtype=float),
    xpos=wp.zeros((nworld, mjm.nbody), dtype=wp.vec3),
    xquat=wp.zeros((nworld, mjm.nbody), dtype=wp.quat),
    xmat=wp.zeros((nworld, mjm.nbody), dtype=wp.mat33),
    xipos=wp.zeros((nworld, mjm.nbody), dtype=wp.vec3),
    ximat=wp.zeros((nworld, mjm.nbody), dtype=wp.mat33),
    xanchor=wp.zeros((nworld, mjm.njnt), dtype=wp.vec3),
    xaxis=wp.zeros((nworld, mjm.njnt), dtype=wp.vec3),
    geom_xpos=wp.zeros((nworld, mjm.ngeom), dtype=wp.vec3),
    geom_xmat=wp.zeros((nworld, mjm.ngeom), dtype=wp.mat33),
    site_xpos=wp.zeros((nworld, mjm.nsite), dtype=wp.vec3),
    site_xmat=wp.zeros((nworld, mjm.nsite), dtype=wp.mat33),
    cam_xpos=wp.zeros((nworld, mjm.ncam), dtype=wp.vec3),
    cam_xmat=wp.zeros((nworld, mjm.ncam), dtype=wp.mat33),
    light_xpos=wp.zeros((nworld, mjm.nlight), dtype=wp.vec3),
    light_xdir=wp.zeros((nworld, mjm.nlight), dtype=wp.vec3),
    subtree_com=wp.zeros((nworld, mjm.nbody), dtype=wp.vec3),
    cdof=wp.zeros((nworld, mjm.nv), dtype=wp.spatial_vector),
    cinert=wp.zeros((nworld, mjm.nbody), dtype=types.vec10),
    actuator_length=wp.zeros((nworld, mjm.nu), dtype=float),
    actuator_moment=wp.zeros((nworld, mjm.nu, mjm.nv), dtype=float),
    crb=wp.zeros((nworld, mjm.nbody), dtype=types.vec10),
    qM=qM,
    qLD=qLD,
    qLDiagInv=wp.zeros((nworld, mjm.nv), dtype=float),
    ten_velocity=wp.zeros((nworld, mjm.ntendon), dtype=float),
    actuator_velocity=wp.zeros((nworld, mjm.nu), dtype=float),
    cvel=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    cdof_dot=wp.zeros((nworld, mjm.nv), dtype=wp.spatial_vector),
    qfrc_bias=wp.zeros((nworld, mjm.nv), dtype=float),
    qfrc_spring=wp.zeros((nworld, mjm.nv), dtype=float),
    qfrc_damper=wp.zeros((nworld, mjm.nv), dtype=float),
    qfrc_gravcomp=wp.zeros((nworld, mjm.nv), dtype=float),
    qfrc_fluid=wp.zeros((nworld, mjm.nv), dtype=float),
    qfrc_passive=wp.zeros((nworld, mjm.nv), dtype=float),
    subtree_linvel=wp.zeros((nworld, mjm.nbody), dtype=wp.vec3),
    subtree_angmom=wp.zeros((nworld, mjm.nbody), dtype=wp.vec3),
    subtree_bodyvel=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),  # warp only
    actuator_force=wp.zeros((nworld, mjm.nu), dtype=float),
    qfrc_actuator=wp.zeros((nworld, mjm.nv), dtype=float),
    qfrc_smooth=wp.zeros((nworld, mjm.nv), dtype=float),
    qacc_smooth=wp.zeros((nworld, mjm.nv), dtype=float),
    qfrc_constraint=wp.zeros((nworld, mjm.nv), dtype=float),
    contact=types.Contact(
      dist=wp.zeros((nconmax,), dtype=float),
      pos=wp.zeros((nconmax,), dtype=wp.vec3f),
      frame=wp.zeros((nconmax,), dtype=wp.mat33f),
      includemargin=wp.zeros((nconmax,), dtype=float),
      friction=wp.zeros((nconmax,), dtype=types.vec5),
      solref=wp.zeros((nconmax,), dtype=wp.vec2f),
      solreffriction=wp.zeros((nconmax,), dtype=wp.vec2f),
      solimp=wp.zeros((nconmax,), dtype=types.vec5),
      dim=wp.zeros((nconmax,), dtype=int),
      geom=wp.zeros((nconmax,), dtype=wp.vec2i),
      efc_address=wp.zeros(
        (nconmax, np.max(mjm.pair_dim) if mjm.npair else np.max(mjm.geom_condim)),
        dtype=int,
      ),
      worldid=wp.zeros((nconmax,), dtype=int),
    ),
    efc=types.Constraint(
      worldid=wp.zeros((njmax,), dtype=int),
      id=wp.zeros((njmax,), dtype=int),
      J=wp.zeros((njmax, mjm.nv), dtype=float),
      pos=wp.zeros((njmax,), dtype=float),
      margin=wp.zeros((njmax,), dtype=float),
      D=wp.zeros((njmax,), dtype=float),
      aref=wp.zeros((njmax,), dtype=float),
      frictionloss=wp.zeros((njmax,), dtype=float),
      force=wp.zeros((njmax,), dtype=float),
      Jaref=wp.zeros((njmax,), dtype=float),
      Ma=wp.zeros((nworld, mjm.nv), dtype=float),
      grad=wp.zeros((nworld, mjm.nv), dtype=float),
      grad_dot=wp.zeros((nworld,), dtype=float),
      Mgrad=wp.zeros((nworld, mjm.nv), dtype=float),
      search=wp.zeros((nworld, mjm.nv), dtype=float),
      search_dot=wp.zeros((nworld,), dtype=float),
      gauss=wp.zeros((nworld,), dtype=float),
      cost=wp.zeros((nworld,), dtype=float),
      prev_cost=wp.zeros((nworld,), dtype=float),
      solver_niter=wp.zeros((nworld,), dtype=int),
      active=wp.zeros((njmax,), dtype=bool),
      gtol=wp.zeros((nworld,), dtype=float),
      mv=wp.zeros((nworld, mjm.nv), dtype=float),
      jv=wp.zeros((njmax,), dtype=float),
      quad=wp.zeros((njmax,), dtype=wp.vec3f),
      quad_gauss=wp.zeros((nworld,), dtype=wp.vec3f),
      h=wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float),
      alpha=wp.zeros((njmax,), dtype=float),
      prev_grad=wp.zeros((nworld, mjm.nv), dtype=float),
      prev_Mgrad=wp.zeros((nworld, mjm.nv), dtype=float),
      beta=wp.zeros((nworld,), dtype=float),
      beta_num=wp.zeros((nworld,), dtype=float),
      beta_den=wp.zeros((nworld,), dtype=float),
      done=wp.zeros((nworld,), dtype=bool),
      # linesearch
      ls_done=wp.zeros((nworld,), dtype=bool),
      p0=wp.zeros((nworld,), dtype=wp.vec3),
      lo=wp.zeros((nworld,), dtype=wp.vec3),
      lo_alpha=wp.zeros((nworld,), dtype=float),
      hi=wp.zeros((nworld,), dtype=wp.vec3),
      hi_alpha=wp.zeros((nworld,), dtype=float),
      lo_next=wp.zeros((nworld,), dtype=wp.vec3),
      lo_next_alpha=wp.zeros((nworld,), dtype=float),
      hi_next=wp.zeros((nworld,), dtype=wp.vec3),
      hi_next_alpha=wp.zeros((nworld,), dtype=float),
      mid=wp.zeros((nworld,), dtype=wp.vec3),
      mid_alpha=wp.zeros((nworld,), dtype=float),
      cost_candidate=wp.zeros((nworld,), dtype=float),
      quad_total_candidate=wp.zeros((nworld,), dtype=wp.vec3f),
      # elliptic cone
      u=wp.zeros((nworld,), dtype=float),
      uu=wp.zeros((nworld,), dtype=float),
      uv=wp.zeros((nworld,), dtype=float),
      vv=wp.zeros((nworld,), dtype=float),
      condim=wp.zeros((njmax,), dtype=int),
    ),
    # RK4
    qpos_t0=wp.zeros((nworld, mjm.nq), dtype=float),
    qvel_t0=wp.zeros((nworld, mjm.nv), dtype=float),
    act_t0=wp.zeros((nworld, mjm.na), dtype=float),
    qvel_rk=wp.zeros((nworld, mjm.nv), dtype=float),
    qacc_rk=wp.zeros((nworld, mjm.nv), dtype=float),
    act_dot_rk=wp.zeros((nworld, mjm.na), dtype=float),
    # euler + implicit integration
    qfrc_integration=wp.zeros((nworld, mjm.nv), dtype=float),
    qacc_integration=wp.zeros((nworld, mjm.nv), dtype=float),
    act_vel_integration=wp.zeros((nworld, mjm.nu), dtype=float),
    qM_integration=wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float),
    qLD_integration=wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float),
    qLDiagInv_integration=wp.zeros((nworld, mjm.nv), dtype=float),
    # sweep-and-prune broadphase
    sap_projection_lower=wp.zeros((2 * nworld, mjm.ngeom), dtype=float),
    sap_projection_upper=wp.zeros((nworld, mjm.ngeom), dtype=float),
    sap_sort_index=wp.zeros((2 * nworld, mjm.ngeom), dtype=int),
    sap_range=wp.zeros((nworld, mjm.ngeom), dtype=int),
    sap_cumulative_sum=wp.zeros(nworld * mjm.ngeom, dtype=int),
    sap_segment_index=wp.array([i * mjm.ngeom for i in range(nworld + 1)], dtype=int),
    # collision driver
    collision_pair=wp.zeros((nconmax,), dtype=wp.vec2i),
    collision_pairid=wp.zeros((nconmax,), dtype=int),
    collision_worldid=wp.zeros((nconmax,), dtype=int),
    ncollision=wp.zeros((1,), dtype=int),
    # rne_postconstraint
    cacc=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    cfrc_int=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    cfrc_ext=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    # tendon
    ten_length=wp.zeros((nworld, mjm.ntendon), dtype=float),
    ten_J=wp.zeros((nworld, mjm.ntendon, mjm.nv), dtype=float),
    ten_wrapadr=wp.zeros((nworld, mjm.ntendon), dtype=int),
    ten_wrapnum=wp.zeros((nworld, mjm.ntendon), dtype=int),
    wrap_obj=wp.zeros((nworld, mjm.nwrap), dtype=wp.vec2i),
    wrap_xpos=wp.zeros((nworld, mjm.nwrap), dtype=wp.spatial_vector),
    # sensors
    sensordata=wp.zeros((nworld, mjm.nsensordata), dtype=float),
  )


def put_data(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nworld: Optional[int] = None,
  nconmax: Optional[int] = None,
  njmax: Optional[int] = None,
) -> types.Data:
  # TODO(team): confirm that Data is set correctly for solver with elliptic friction cones
  # TODO(team): move nconmax and njmax to Model?
  # TODO(team): decide what to do about unintialized warp-only fields created by put_data
  #             we need to ensure these are only workspace fields and don't carry state

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

  # calculate some fields that cannot be easily computed inline:
  if mujoco.mj_isSparse(mjm):
    qM = np.expand_dims(mjd.qM, axis=0)
    qLD = np.expand_dims(mjd.qLD, axis=0)
    qM_integration = np.zeros((1, mjm.nM), dtype=float)
    qLD_integration = np.zeros((1, mjm.nM), dtype=float)
    efc_J = np.zeros((mjd.nefc, mjm.nv))
    mujoco.mju_sparse2dense(efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
    ten_J = np.zeros((mjm.ntendon, mjm.nv))
    mujoco.mju_sparse2dense(
      ten_J,
      mjd.ten_J.reshape(-1),
      mjd.ten_J_rownnz,
      mjd.ten_J_rowadr,
      mjd.ten_J_colind.reshape(-1),
    )
  else:
    qM = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, qM, mjd.qM)
    if (mjd.qM == 0.0).all() or (mjd.qLD == 0.0).all():
      qLD = np.zeros((mjm.nv, mjm.nv))
    else:
      qLD = np.linalg.cholesky(qM)
    qM_integration = np.zeros((mjm.nv, mjm.nv), dtype=float)
    qLD_integration = np.zeros((mjm.nv, mjm.nv), dtype=float)
    efc_J = mjd.efc_J.reshape((mjd.nefc, mjm.nv))
    ten_J = mjd.ten_J.reshape((mjm.ntendon, mjm.nv))

  # TODO(taylorhowell): sparse actuator_moment
  actuator_moment = np.zeros((mjm.nu, mjm.nv))
  mujoco.mju_sparse2dense(
    actuator_moment,
    mjd.actuator_moment,
    mjd.moment_rownnz,
    mjd.moment_rowadr,
    mjd.moment_colind,
  )

  contact_efc_address = np.zeros((nconmax, np.max(mjm.pair_dim) if mjm.npair else np.max(mjm.geom_condim)), dtype=int)
  for i in range(nworld):
    for j in range(mjd.ncon):
      condim = mjd.contact.dim[j]
      for k in range(condim):
        contact_efc_address[i * mjd.ncon + j, k] = mjd.nefc * i + mjd.contact.efc_address[j] + k

  contact_worldid = np.pad(np.repeat(np.arange(nworld), mjd.ncon), (0, nconmax - nworld * mjd.ncon))
  efc_worldid = np.pad(np.repeat(np.arange(nworld), mjd.nefc), (0, njmax - nworld * mjd.nefc))

  # some helper functions to simplify the data field definitions below

  def arr(x, dtype=None):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    if dtype is None:
      if np.issubdtype(x.dtype, np.integer):
        dtype = wp.int32
      elif np.issubdtype(x.dtype, np.floating):
        dtype = wp.float32
      elif np.issubdtype(x.dtype, np.bool):
        dtype = wp.bool
      else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")
    wp_array = {1: wp.array, 2: wp.array2d, 3: wp.array3d}[x.ndim]
    return wp_array(x, dtype=dtype)

  def tile(x, dtype=None):
    return arr(np.tile(x, (nworld,) + (1,) * len(x.shape)), dtype)

  def padtile(x, length, dtype=None):
    x = np.repeat(x, nworld, axis=0)
    width = ((0, length - x.shape[0]),) + ((0, 0),) * (x.ndim - 1)
    return arr(np.pad(x, width), dtype)

  return types.Data(
    nworld=nworld,
    nconmax=nconmax,
    njmax=njmax,
    ncon=arr([mjd.ncon * nworld]),
    ne=arr([mjd.ne * nworld]),
    ne_connect=arr([3 * nworld * np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_CONNECT) & mjd.eq_active)]),
    ne_weld=arr([6 * nworld * np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_WELD) & mjd.eq_active)]),
    ne_jnt=arr([nworld * np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_JOINT) & mjd.eq_active)]),
    ne_ten=arr([nworld * np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_TENDON) & mjd.eq_active)]),
    nf=arr([mjd.nf * nworld]),
    nl=arr([mjd.nl * nworld]),
    nefc=arr([mjd.nefc * nworld]),
    time=arr(mjd.time * np.ones(nworld)),
    qpos=tile(mjd.qpos),
    qvel=tile(mjd.qvel),
    act=tile(mjd.act),
    qacc_warmstart=tile(mjd.qacc_warmstart),
    ctrl=tile(mjd.ctrl),
    qfrc_applied=tile(mjd.qfrc_applied),
    xfrc_applied=tile(mjd.xfrc_applied, dtype=wp.spatial_vector),
    fluid_applied=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    eq_active=tile(mjd.eq_active.astype(bool)),
    mocap_pos=tile(mjd.mocap_pos, dtype=wp.vec3),
    mocap_quat=tile(mjd.mocap_quat, dtype=wp.quat),
    qacc=tile(mjd.qacc),
    act_dot=tile(mjd.act_dot),
    xpos=tile(mjd.xpos, dtype=wp.vec3),
    xquat=tile(mjd.xquat, dtype=wp.quat),
    xmat=tile(mjd.xmat, dtype=wp.mat33),
    xipos=tile(mjd.xipos, dtype=wp.vec3),
    ximat=tile(mjd.ximat, dtype=wp.mat33),
    xanchor=tile(mjd.xanchor, dtype=wp.vec3),
    xaxis=tile(mjd.xaxis, dtype=wp.vec3),
    geom_xpos=tile(mjd.geom_xpos, dtype=wp.vec3),
    geom_xmat=tile(mjd.geom_xmat, dtype=wp.mat33),
    site_xpos=tile(mjd.site_xpos, dtype=wp.vec3),
    site_xmat=tile(mjd.site_xmat, dtype=wp.mat33),
    cam_xpos=tile(mjd.cam_xpos, dtype=wp.vec3),
    cam_xmat=tile(mjd.cam_xmat, dtype=wp.mat33),
    light_xpos=tile(mjd.light_xpos, dtype=wp.vec3),
    light_xdir=tile(mjd.light_xdir, dtype=wp.vec3),
    subtree_com=tile(mjd.subtree_com, dtype=wp.vec3),
    cdof=tile(mjd.cdof, dtype=wp.spatial_vector),
    cinert=tile(mjd.cinert, dtype=types.vec10),
    actuator_length=tile(mjd.actuator_length),
    actuator_moment=tile(actuator_moment),
    crb=tile(mjd.crb, dtype=types.vec10),
    qM=tile(qM),
    qLD=tile(qLD),
    qLDiagInv=tile(mjd.qLDiagInv),
    ten_velocity=tile(mjd.ten_velocity),
    actuator_velocity=tile(mjd.actuator_velocity),
    cvel=tile(mjd.cvel, dtype=wp.spatial_vector),
    cdof_dot=tile(mjd.cdof_dot, dtype=wp.spatial_vector),
    qfrc_bias=tile(mjd.qfrc_bias),
    qfrc_spring=tile(mjd.qfrc_spring),
    qfrc_damper=tile(mjd.qfrc_damper),
    qfrc_gravcomp=tile(mjd.qfrc_gravcomp),
    qfrc_fluid=tile(mjd.qfrc_fluid),
    qfrc_passive=tile(mjd.qfrc_passive),
    subtree_linvel=tile(mjd.subtree_linvel, dtype=wp.vec3),
    subtree_angmom=tile(mjd.subtree_angmom, dtype=wp.vec3),
    subtree_bodyvel=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    actuator_force=tile(mjd.actuator_force),
    qfrc_actuator=tile(mjd.qfrc_actuator),
    qfrc_smooth=tile(mjd.qfrc_smooth),
    qacc_smooth=tile(mjd.qacc_smooth),
    qfrc_constraint=tile(mjd.qfrc_constraint),
    contact=types.Contact(
      dist=padtile(mjd.contact.dist, nconmax),
      pos=padtile(mjd.contact.pos, nconmax, dtype=wp.vec3),
      frame=padtile(mjd.contact.frame, nconmax, dtype=wp.mat33),
      includemargin=padtile(mjd.contact.includemargin, nconmax),
      friction=padtile(mjd.contact.friction, nconmax, dtype=types.vec5),
      solref=padtile(mjd.contact.solref, nconmax, dtype=wp.vec2f),
      solreffriction=padtile(mjd.contact.solreffriction, nconmax, dtype=wp.vec2f),
      solimp=padtile(mjd.contact.solimp, nconmax, dtype=types.vec5),
      dim=padtile(mjd.contact.dim, nconmax),
      geom=padtile(mjd.contact.geom, nconmax, dtype=wp.vec2i),
      efc_address=arr(contact_efc_address),
      worldid=arr(contact_worldid),
    ),
    efc=types.Constraint(
      worldid=arr(efc_worldid),
      id=padtile(mjd.efc_id, njmax),
      J=padtile(efc_J, njmax),
      pos=padtile(mjd.efc_pos, njmax),
      margin=padtile(mjd.efc_margin, njmax),
      D=padtile(mjd.efc_D, njmax),
      aref=padtile(mjd.efc_aref, njmax),
      frictionloss=padtile(mjd.efc_frictionloss, njmax),
      force=padtile(mjd.efc_force, njmax),
      Jaref=wp.empty(shape=(njmax,), dtype=float),
      Ma=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      grad=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      grad_dot=wp.empty(shape=(nworld,), dtype=float),
      Mgrad=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      search=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      search_dot=wp.empty(shape=(nworld,), dtype=float),
      gauss=wp.empty(shape=(nworld,), dtype=float),
      cost=wp.empty(shape=(nworld,), dtype=float),
      prev_cost=wp.empty(shape=(nworld,), dtype=float),
      solver_niter=wp.empty(shape=(nworld,), dtype=int),
      active=wp.empty(shape=(njmax,), dtype=bool),
      gtol=wp.empty(shape=(nworld,), dtype=float),
      mv=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      jv=wp.empty(shape=(njmax,), dtype=float),
      quad=wp.empty(shape=(njmax,), dtype=wp.vec3f),
      quad_gauss=wp.empty(shape=(nworld,), dtype=wp.vec3f),
      h=wp.empty(shape=(nworld, mjm.nv, mjm.nv), dtype=float),
      alpha=wp.empty(shape=(nworld,), dtype=float),
      prev_grad=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      prev_Mgrad=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      beta=wp.empty(shape=(nworld,), dtype=float),
      beta_num=wp.empty(shape=(nworld,), dtype=float),
      beta_den=wp.empty(shape=(nworld,), dtype=float),
      done=wp.empty(shape=(nworld,), dtype=bool),
      ls_done=wp.zeros(shape=(nworld,), dtype=bool),
      p0=wp.empty(shape=(nworld,), dtype=wp.vec3),
      lo=wp.empty(shape=(nworld,), dtype=wp.vec3),
      lo_alpha=wp.empty(shape=(nworld,), dtype=float),
      hi=wp.empty(shape=(nworld,), dtype=wp.vec3),
      hi_alpha=wp.empty(shape=(nworld,), dtype=float),
      lo_next=wp.empty(shape=(nworld,), dtype=wp.vec3),
      lo_next_alpha=wp.empty(shape=(nworld,), dtype=float),
      hi_next=wp.empty(shape=(nworld,), dtype=wp.vec3),
      hi_next_alpha=wp.empty(shape=(nworld,), dtype=float),
      mid=wp.empty(shape=(nworld,), dtype=wp.vec3),
      mid_alpha=wp.empty(shape=(nworld,), dtype=float),
      cost_candidate=wp.empty(shape=(nworld, mjm.opt.ls_iterations), dtype=float),
      quad_total_candidate=wp.empty(shape=(nworld, mjm.opt.ls_iterations), dtype=wp.vec3f),
      # TODO(team): skip allocation if not ellpitic
      u=wp.empty((nconmax, 6), dtype=float),
      uu=wp.empty((nconmax,), dtype=float),
      uv=wp.empty((nconmax,), dtype=float),
      vv=wp.empty((nconmax,), dtype=float),
      condim=wp.empty((njmax,), dtype=int),
    ),
    # TODO(team): skip allocation if integrator != RK4
    qpos_t0=wp.empty((nworld, mjm.nq), dtype=float),
    qvel_t0=wp.empty((nworld, mjm.nv), dtype=float),
    act_t0=wp.empty((nworld, mjm.na), dtype=float),
    qvel_rk=wp.empty((nworld, mjm.nv), dtype=float),
    qacc_rk=wp.empty((nworld, mjm.nv), dtype=float),
    act_dot_rk=wp.empty((nworld, mjm.na), dtype=float),
    # TODO(team): skip allocation if integrator != euler | implicit
    qfrc_integration=wp.zeros((nworld, mjm.nv), dtype=float),
    qacc_integration=wp.zeros((nworld, mjm.nv), dtype=float),
    act_vel_integration=wp.zeros((nworld, mjm.nu), dtype=float),
    qM_integration=tile(qM_integration),
    qLD_integration=tile(qLD_integration),
    qLDiagInv_integration=wp.zeros((nworld, mjm.nv), dtype=float),
    # TODO(team): skip allocation if broadphase != sap
    sap_projection_lower=wp.zeros((2 * nworld, mjm.ngeom), dtype=float),
    sap_projection_upper=wp.zeros((nworld, mjm.ngeom), dtype=float),
    sap_sort_index=wp.zeros((2 * nworld, mjm.ngeom), dtype=int),
    sap_range=wp.zeros((nworld, mjm.ngeom), dtype=int),
    sap_cumulative_sum=wp.zeros(nworld * mjm.ngeom, dtype=int),
    sap_segment_index=arr([i * mjm.ngeom for i in range(nworld + 1)]),
    # collision driver
    collision_pair=wp.empty(nconmax, dtype=wp.vec2i),
    collision_pairid=wp.empty(nconmax, dtype=int),
    collision_worldid=wp.empty(nconmax, dtype=int),
    ncollision=wp.zeros(1, dtype=int),
    # rne_postconstraint but also smooth
    cacc=tile(mjd.cacc, dtype=wp.spatial_vector),
    cfrc_int=tile(mjd.cfrc_int, dtype=wp.spatial_vector),
    cfrc_ext=tile(mjd.cfrc_ext, dtype=wp.spatial_vector),
    # tendon
    ten_length=tile(mjd.ten_length),
    ten_J=tile(ten_J),
    ten_wrapadr=tile(mjd.ten_wrapadr),
    ten_wrapnum=tile(mjd.ten_wrapnum),
    wrap_obj=tile(mjd.wrap_obj, dtype=wp.vec2i),
    wrap_xpos=tile(mjd.wrap_xpos, dtype=wp.spatial_vector),
    # sensors
    sensordata=tile(mjd.sensordata),
  )


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
  result.qfrc_fluid = d.qfrc_fluid.numpy()[0]
  result.qfrc_passive = d.qfrc_passive.numpy()[0]
  result.subtree_linvel = d.subtree_linvel.numpy()[0]
  result.subtree_angmom = d.subtree_angmom.numpy()[0]
  result.qfrc_spring = d.qfrc_spring.numpy()[0]
  result.qfrc_damper = d.qfrc_damper.numpy()[0]
  result.qfrc_gravcomp = d.qfrc_gravcomp.numpy()[0]
  result.qfrc_fluid = d.qfrc_fluid.numpy()[0]
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

  if mujoco.mj_isSparse(mjm):
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

  # TODO(team): set these efc_* fields after fix to _realloc_con_efc
  # Safely copy only up to the minimum of the destination and source sizes
  # n = min(result.efc_D.shape[0], d.efc.D.numpy()[:nefc].shape[0])
  # result.efc_D[:n] = d.efc.D.numpy()[:nefc][:n]
  # n_pos = min(result.efc_pos.shape[0], d.efc.pos.numpy()[:nefc].shape[0])
  # result.efc_pos[:n_pos] = d.efc.pos.numpy()[:nefc][:n_pos]

  # n_aref = min(result.efc_aref.shape[0], d.efc.aref.numpy()[:nefc].shape[0])
  # result.efc_aref[:n_aref] = d.efc.aref.numpy()[:nefc][:n_aref]

  # n_force = min(result.efc_force.shape[0], d.efc.force.numpy()[:nefc].shape[0])
  # result.efc_force[:n_force] = d.efc.force.numpy()[:nefc][:n_force]

  # n_margin = min(result.efc_margin.shape[0], d.efc.margin.numpy()[:nefc].shape[0])
  # result.efc_margin[:n_margin] = d.efc.margin.numpy()[:nefc][:n_margin]

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
