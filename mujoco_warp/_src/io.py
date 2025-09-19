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

from typing import Any, Optional, Sequence, Union

import mujoco
import numpy as np
import warp as wp

from . import math
from . import warp_util
from .types import MJ_MAX_EPAFACES
from .types import MJ_MAX_EPAHORIZON
from .types import MJ_MAXCONPAIR
from .types import BiasType
from .types import BlockDim
from .types import BroadphaseFilter
from .types import BroadphaseType
from .types import ConeType
from .types import Constraint
from .types import Contact
from .types import Data
from .types import DisableBit
from .types import DynType
from .types import EnableBit
from .types import EqType
from .types import GainType
from .types import GeomType
from .types import IntegratorType
from .types import Model
from .types import Option
from .types import SensorType
from .types import SolverType
from .types import Statistic
from .types import TileSet
from .types import TrnType
from .types import WrapType
from .types import vec5
from .types import vec10

# max number of worlds supported
MAX_WORLDS = 2**24

# tolerance override for float32
_TOLERANCE_F32 = 1.0e-6


def _max_meshdegree(mjm: mujoco.MjModel) -> int:
  if mjm.mesh_polyvertnum.size == 0:
    return 4
  return max(3, mjm.mesh_polymapnum.max())


def _max_npolygon(mjm: mujoco.MjModel) -> int:
  if mjm.mesh_polyvertnum.size == 0:
    return 4
  return max(4, mjm.mesh_polyvertnum.max())


def _nmodel_batched_array(mjm_array, dtype):
  mjm_array = np.array(mjm_array)
  array = wp.array(mjm_array, dtype=dtype)
  # add private attribute for JAX to determine which fields are batched
  array._is_batched = True
  if not mjm_array.shape:  # wp,array always has at least 1 dim, which isn't what we expect for scalars
    array.strides = (0,)
    array.shape = (MAX_WORLDS,)
  else:
    array.strides = (0,) + array.strides
    array.ndim += 1
    array.shape = (MAX_WORLDS,) + array.shape
  return array


def put_model(mjm: mujoco.MjModel) -> Model:
  """
  Creates a model on device.

  Args:
    mjm (mujoco.MjModel): The model containing kinematic and dynamic information (host).

  Returns:
    Model: The model containing kinematic and dynamic information (device).
  """
  # check supported types
  for field, field_types, field_str in (
    (mjm.actuator_trntype, TrnType, "Actuator transmission type"),
    (mjm.actuator_dyntype, DynType, "Actuator dynamics type"),
    (mjm.actuator_gaintype, GainType, "Gain type"),
    (mjm.actuator_biastype, BiasType, "Bias type"),
    (mjm.eq_type, EqType, "Equality constraint types"),
    (mjm.geom_type, GeomType, "Geom type"),
    (mjm.sensor_type, SensorType, "Sensor types"),
    (mjm.wrap_type, WrapType, "Wrap types"),
  ):
    unsupported = ~np.isin(field, list(field_types))
    if unsupported.any():
      raise NotImplementedError(f"{field_str} {field[unsupported]} not supported.")

  # check options
  for opt, opt_types, msg in (
    (mjm.opt.integrator, IntegratorType, "Integrator"),
    (mjm.opt.cone, ConeType, "Cone"),
    (mjm.opt.solver, SolverType, "Solver"),
  ):
    if opt not in set(opt_types):
      raise NotImplementedError(f"{msg} {opt} is unsupported.")

  if mjm.opt.noslip_iterations > 0:
    raise NotImplementedError(f"noslip solver not implemented.")

  if (mjm.opt.viscosity > 0 or mjm.opt.density > 0) and mjm.opt.integrator in (
    mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    mujoco.mjtIntegrator.mjINT_IMPLICIT,
  ):
    raise NotImplementedError(f"Implicit integrators and fluid model not implemented.")

  if mjm.nflex > 1:
    raise NotImplementedError("Only one flex is unsupported.")

  if ((mjm.flex_contype != 0) | (mjm.flex_conaffinity != 0)).any():
    raise NotImplementedError("Flex collisions are not implemented.")

  if mjm.geom_fluid.any():
    raise NotImplementedError("Ellipsoid fluid model not implemented.")

  # TODO(team): remove after _update_gradient for Newton uses tile operations for islands
  nv_max = 60
  if mjm.nv > nv_max and mjm.opt.jacobian == mujoco.mjtJacobian.mjJAC_DENSE:
    raise ValueError(f"Dense is unsupported for nv > {nv_max} (nv = {mjm.nv}).")

  # options and statistics fields
  stat = Statistic(meaninertia=mjm.stat.meaninertia)

  opt = {}

  for name, field in Option.__annotations__.items():
    if not hasattr(mjm.opt, name):
      continue
    val = getattr(mjm.opt, name)
    if field is int:
      opt[name] = val
    elif isinstance(field, wp.types.array):
      opt[name] = _nmodel_batched_array(val, dtype=field.dtype)

  # float32 precision adds noise to solver error, so raise the tolerance to compensate
  opt["tolerance"] = _nmodel_batched_array(max(mjm.opt.tolerance, _TOLERANCE_F32), dtype=float)
  opt["has_fluid"] = mjm.opt.wind.any() or mjm.opt.density > 0 or mjm.opt.viscosity > 0
  # TODO(team): fix the semantics here, MJC is_sparse is for jacobian, but MJW is mass matrix
  opt["is_sparse"] = mujoco.mj_isSparse(mjm)
  # TODO(team): figure out if we can default ls_parallel to True
  opt["ls_parallel"] = False
  # TODO(team): determine good default setting
  opt["ls_parallel_min_step"] = 1.0e-6
  # NOTE: this is updated later when we calculate all possible geom pairs
  opt["broadphase"] = BroadphaseType.NXN.value
  opt["broadphase_filter"] = int(BroadphaseFilter.PLANE | BroadphaseFilter.SPHERE | BroadphaseFilter.OBB)
  opt["graph_conditional"] = warp_util.conditional_graph_supported()
  opt["run_collision_detection"] = True
  # TODO(team): remove legacy gjk
  opt["legacy_gjk"] = False
  opt["contact_sensor_maxmatch"] = 64

  opt = Option(**opt)

  # Model
  model = {}

  for name, field in Model.__annotations__.items():
    if not hasattr(mjm, name):
      continue
    if name == "plugin_attr":
      continue  # we handle plugin_attr in a different way than C MuJoCo
    val = getattr(mjm, name)
    if field is int:
      model[name] = val
    elif name == "oct_aabb":
      # TODO(team): upgrade oct_abb to batched field
      model[name] = wp.array(val, dtype=field.dtype)
    elif isinstance(field, wp.types.array) and field.dtype in (wp.bool, wp.int32, wp.int64):
      # discrete arrays (e.g. types, ids, etc) are not batched
      model[name] = wp.array(val, dtype=field.dtype)
    elif isinstance(field, wp.types.array):
      try:
        model[name] = _nmodel_batched_array(val, dtype=field.dtype)
      except Exception as e:
        raise type(e)(f"Error creating model field '{name}': {e}") from e

  model["opt"] = opt
  model["stat"] = stat
  model["nlsp"] = mjm.opt.ls_iterations  # TODO(team): determine how to set nlsp
  model["nsensortaxel"] = sum(mjm.mesh_vertnum[mjm.sensor_objid[mjm.sensor_type == SensorType.TACTILE]])
  model["block_dim"] = BlockDim()

  # indices for sparse qM_fullm (used in solver)
  qM_fullm_i, qM_fullm_j = [], []
  for i in range(mjm.nv):
    j = i
    while j > -1:
      qM_fullm_i.append(i)
      qM_fullm_j.append(j)
      j = mjm.dof_parentid[j]
  model["qM_fullm_i"] = wp.array(qM_fullm_i, dtype=int)
  model["qM_fullm_j"] = wp.array(qM_fullm_j, dtype=int)

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
  model["qM_mulm_i"] = wp.array(qM_mulm_i, dtype=int)
  model["qM_mulm_j"] = wp.array(qM_mulm_j, dtype=int)
  model["qM_madr_ij"] = wp.array(qM_madr_ij, dtype=int)

  # qLD_updates has dof tree ordering of qLD updates for sparse factor m
  qLD_updates, dof_depth = {}, np.zeros(mjm.nv, dtype=int) - 1

  for k in range(mjm.nv):
    # skip diagonal rows
    if mjm.M_rownnz[k] == 1:
      continue
    dof_depth[k] = dof_depth[mjm.dof_parentid[k]] + 1
    i = mjm.dof_parentid[k]
    diag_k = mjm.M_rowadr[k] + mjm.M_rownnz[k] - 1
    Madr_ki = diag_k - 1
    while i > -1:
      qLD_updates.setdefault(dof_depth[i], []).append((i, k, Madr_ki))
      i = mjm.dof_parentid[i]
      Madr_ki -= 1
  model["qLD_updates"] = tuple(wp.array(qLD_updates[i], dtype=wp.vec3i) for i in sorted(qLD_updates))

  # qM_tiles records the block diagonal structure of qM
  tile_corners = [i for i in range(mjm.nv) if mjm.dof_parentid[i] == -1]
  tiles = {}
  for i in range(len(tile_corners)):
    tile_beg = tile_corners[i]
    tile_end = mjm.nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
    tiles.setdefault(tile_end - tile_beg, []).append(tile_beg)
  model["qM_tiles"] = tuple(TileSet(adr=wp.array(tiles[sz], dtype=int), size=sz) for sz in sorted(tiles.keys()))

  # body_tree is a list of body ids grouped by tree level
  bodies, body_depth = {}, np.zeros(mjm.nbody, dtype=int) - 1
  for i in range(mjm.nbody):
    body_depth[i] = body_depth[mjm.body_parentid[i]] + 1
    bodies.setdefault(body_depth[i], []).append(i)
  model["body_tree"] = tuple(wp.array(bodies[i], dtype=int) for i in sorted(bodies))

  # subtree_mass is a precalculated array used in smooth
  subtree_mass = np.copy(mjm.body_mass)
  for i in range(mjm.nbody - 1, -1, -1):
    subtree_mass[mjm.body_parentid[i]] += subtree_mass[i]
  model["subtree_mass"] = _nmodel_batched_array(subtree_mass, dtype=float)

  # jnt_limited_slide_hinge_adr and jnt_limited_ball_adr are used in constraint.py
  jnt_slide_or_hinge = np.isin(mjm.jnt_type, (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE))
  jnt_ball = mjm.jnt_type == mujoco.mjtJoint.mjJNT_BALL
  model["jnt_limited_slide_hinge_adr"] = wp.array(np.nonzero(mjm.jnt_limited & jnt_slide_or_hinge)[0], dtype=int)
  model["jnt_limited_ball_adr"] = wp.array(np.nonzero(mjm.jnt_limited & jnt_ball)[0], dtype=int)

  # dof lower triangle row and column indices (used in solver)
  dof_tri_row, dof_tri_col = np.tril_indices(mjm.nv)
  model["dof_tri_row"] = wp.array(dof_tri_row, dtype=int)
  model["dof_tri_col"] = wp.array(dof_tri_col, dtype=int)

  # custom plugins
  plugin, plugin_attr = [], []
  geom_plugin_index = np.full_like(mjm.geom_type, -1)

  if mjm.nplugin > 0:
    for i in range(len(mjm.geom_plugin)):
      if mjm.geom_plugin[i] != -1:
        p = mjm.geom_plugin[i]
        geom_plugin_index[i] = len(plugin)
        plugin.append(mjm.plugin[p])
        start = mjm.plugin_attradr[p]
        end = mjm.plugin_attradr[p + 1] if p + 1 < mjm.nplugin else len(mjm.plugin_attr)
        values = mjm.plugin_attr[start:end]
        attr_values = []
        current = []
        for v in values:
          if v == 0:
            if current:
              s = "".join(chr(int(x)) for x in current)
              attr_values.append(float(s))
              current = []
          else:
            current.append(v)
        # Pad with zeros if less than 3
        attr_values += [0.0] * (3 - len(attr_values))
        plugin_attr.append(attr_values[:3])

  model["plugin"] = wp.array(plugin, dtype=int)
  model["plugin_attr"] = wp.array(plugin_attr, dtype=wp.vec3f)
  model["geom_plugin_index"] = wp.array(geom_plugin_index, dtype=int)

  # pre-compute indices of equality constraints
  model["eq_connect_adr"] = (wp.array(np.nonzero(mjm.eq_type == EqType.CONNECT)[0], dtype=int),)
  model["eq_wld_adr"] = wp.array(np.nonzero(mjm.eq_type == EqType.WELD)[0], dtype=int)
  model["eq_jnt_adr"] = wp.array(np.nonzero(mjm.eq_type == EqType.JOINT)[0], dtype=int)
  model["eq_ten_adr"] = wp.array(np.nonzero(mjm.eq_type == EqType.TENDON)[0], dtype=int)

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
    elif trntype == mujoco.mjtTrn.mjTRN_BODY:
      pass
    elif trntype == mujoco.mjtTrn.mjTRN_SITE:
      siteid = mjm.actuator_trnid[i, 0]
      bid = mjm.site_bodyid[siteid]
      while bid > 0:
        bodyid.append(bid)
        bid = mjm.body_parentid[bid]
    elif trntype == mujoco.mjtTrn.mjTRN_SLIDERCRANK:
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
    actuator_moment_tiles_nv += (TileSet(adr=adr_nv, size=nv),)
    actuator_moment_tiles_nu += (TileSet(adr=adr_nu, size=nu),)

  model["actuator_moment_tiles_nv"] = actuator_moment_tiles_nv
  model["actuator_moment_tiles_nu"] = actuator_moment_tiles_nu
  # short-circuiting here allows us to skip a lot of code in implicit integration
  model["actuator_affine_bias_gain"] = np.any(mjm.actuator_biastype == BiasType.AFFINE) or np.any(
    mjm.actuator_gaintype == GainType.AFFINE
  )
  model["actuator_trntype_body_adr"] = wp.array(np.nonzero(mjm.actuator_trntype == TrnType.BODY)[0], dtype=int)

  # precalculate geom pairs for broad phase
  filterparent = not (mjm.opt.disableflags & DisableBit.FILTERPARENT.value)

  geom1, geom2 = np.triu_indices(mjm.ngeom, k=1)
  bodyid1 = mjm.geom_bodyid[geom1]
  bodyid2 = mjm.geom_bodyid[geom2]
  contype1 = mjm.geom_contype[geom1]
  contype2 = mjm.geom_contype[geom2]
  conaffinity1 = mjm.geom_conaffinity[geom1]
  conaffinity2 = mjm.geom_conaffinity[geom2]
  weldid1 = mjm.body_weldid[bodyid1]
  weldid2 = mjm.body_weldid[bodyid2]
  weld_parentid1 = mjm.body_weldid[mjm.body_parentid[weldid1]]
  weld_parentid2 = mjm.body_weldid[mjm.body_parentid[weldid2]]

  self_collision = weldid1 == weldid2
  parent_child_collision = (
    filterparent & (weldid1 != 0) & (weldid2 != 0) & ((weldid1 == weld_parentid2) | (weldid2 == weld_parentid1))
  )
  mask = np.array((contype1 & conaffinity2) | (contype2 & conaffinity1), dtype=bool)
  exclude = np.isin((bodyid1 << 16) + bodyid2, mjm.exclude_signature)

  nxn_pairid = -1 * np.ones(len(geom1), dtype=int)
  nxn_pairid[~(mask & ~self_collision & ~parent_child_collision & ~exclude)] = -2

  # contact pairs
  for i, (g1, g2) in enumerate(zip(mjm.pair_geom1, mjm.pair_geom2)):
    g1, g2 = g2, g1 if g1 > g2 else g1, g2
    nxn_pairid[math.upper_tri_index(mjm.ngeom, g1, g2)] = i

  nxn_geom_pair = np.stack((geom1, geom2), axis=1)
  nxn_geom_pair_filtered = nxn_geom_pair[nxn_pairid > -2]
  model["nxn_geom_pair"] = wp.array(nxn_geom_pair, dtype=wp.vec2i)
  model["nxn_geom_pair_filtered"] = wp.array(nxn_geom_pair_filtered, dtype=wp.vec2i)
  model["nxn_pairid"] = wp.array(nxn_pairid, dtype=int)
  model["nxn_pairid_filtered"] = wp.array(nxn_pairid[nxn_pairid > -2], dtype=int)

  if nxn_geom_pair_filtered.shape[0] >= 250_000:
    opt.broadphase = BroadphaseType.SAP_TILE if mjm.ngeom < 1000 else BroadphaseType.SAP_SEGMENTED

  # count contact pair types
  geom_pair_type_count = np.bincount(
    [
      math.upper_trid_index(len(GeomType), int(mjm.geom_type[geom1[i]]), int(mjm.geom_type[geom2[i]]))
      for i in np.arange(len(geom1))
      if nxn_pairid[i] > -2
    ],
    minlength=len(GeomType) * (len(GeomType) + 1) // 2,
  )
  model["geom_pair_type_count"] = geom_pair_type_count

  # Disable collisions if there are no potentially colliding pairs
  if np.sum(geom_pair_type_count) == 0:
    opt.disableflags |= DisableBit.CONTACT.value

  # used in contact efc row creation
  model["condim_max"] = np.max(np.concatenate((mjm.geom_condim, mjm.pair_dim, [0])))

  # used in tendon efc row creation
  model["tendon_limited_adr"] = wp.array(np.nonzero(mjm.tendon_limited)[0], dtype=int)

  # fixed tendon addresses
  tendon_jnt_adr = []
  wrap_jnt_adr = []
  for i in range(mjm.ntendon):
    adr = mjm.tendon_adr[i]
    if mjm.wrap_type[adr] == mujoco.mjtWrap.mjWRAP_JOINT:
      tendon_num = mjm.tendon_num[i]
      for j in range(tendon_num):
        tendon_jnt_adr.append(i)
        wrap_jnt_adr.append(adr + j)
  model["tendon_jnt_adr"] = wp.array(tendon_jnt_adr, dtype=int)
  model["wrap_jnt_adr"] = wp.array(wrap_jnt_adr, dtype=int)

  # spatial tendon addresses
  tendon_site_pair_adr = []
  tendon_geom_adr = []

  ten_wrapadr_site = [0]
  ten_wrapnum_site = []
  for i, tendon_num in enumerate(mjm.tendon_num):
    adr = mjm.tendon_adr[i]
    # sites
    if (mjm.wrap_type[adr : adr + tendon_num] == mujoco.mjtWrap.mjWRAP_SITE).all():
      if i < mjm.ntendon:
        ten_wrapadr_site.append(ten_wrapadr_site[-1] + tendon_num)
      ten_wrapnum_site.append(tendon_num)
    else:
      if i < mjm.ntendon:
        ten_wrapadr_site.append(ten_wrapadr_site[-1])
      ten_wrapnum_site.append(0)

    # geoms
    for j in range(tendon_num):
      wrap_type = mjm.wrap_type[adr + j]
      if j < tendon_num - 1:
        next_wrap_type = mjm.wrap_type[adr + j + 1]
        if wrap_type == mujoco.mjtWrap.mjWRAP_SITE and next_wrap_type == mujoco.mjtWrap.mjWRAP_SITE:
          tendon_site_pair_adr.append(i)
      if wrap_type == mujoco.mjtWrap.mjWRAP_SPHERE or wrap_type == mujoco.mjtWrap.mjWRAP_CYLINDER:
        tendon_geom_adr.append(i)

  model["tendon_site_pair_adr"] = wp.array(tendon_site_pair_adr, dtype=int)
  model["tendon_geom_adr"] = wp.array(tendon_geom_adr, dtype=int)
  model["ten_wrapadr_site"] = wp.array(ten_wrapadr_site, dtype=int)
  model["ten_wrapnum_site"] = wp.array(ten_wrapnum_site, dtype=int)

  # wrap addresses
  wrap_site_adr = np.nonzero(mjm.wrap_type == mujoco.mjtWrap.mjWRAP_SITE)[0]
  wrap_site_pair_adr = np.setdiff1d(wrap_site_adr[np.nonzero(np.diff(wrap_site_adr) == 1)[0]], mjm.tendon_adr[1:] - 1)
  wrap_geom_adr = np.nonzero(np.isin(mjm.wrap_type, [mujoco.mjtWrap.mjWRAP_SPHERE, mujoco.mjtWrap.mjWRAP_CYLINDER]))[0]
  model["wrap_site_adr"] = wp.array(wrap_site_adr, dtype=int)
  model["wrap_site_pair_adr"] = wp.array(wrap_site_pair_adr, dtype=int)
  model["wrap_geom_adr"] = wp.array(wrap_geom_adr, dtype=int)

  # pulley scaling
  wrap_pulley_scale = np.ones(mjm.nwrap, dtype=float)
  pulley_adr = np.nonzero(mjm.wrap_type == mujoco.mjtWrap.mjWRAP_PULLEY)[0]
  for tadr, tnum in zip(mjm.tendon_adr, mjm.tendon_num):
    for padr in pulley_adr:
      if tadr <= padr < tadr + tnum:
        wrap_pulley_scale[padr : tadr + tnum] = 1.0 / mjm.wrap_prm[padr]
  model["wrap_pulley_scale"] = wp.array(wrap_pulley_scale, dtype=float)

  # sensor custom fields
  sensor_pos = mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_POS
  sensor_pos &= (mjm.sensor_type != SensorType.JOINTLIMITPOS) & (mjm.sensor_type != SensorType.TENDONLIMITPOS)
  model["sensor_pos_adr"] = wp.array(np.nonzero(sensor_pos)[0], dtype=int)
  model["sensor_limitpos_adr"] = wp.array(
    np.nonzero((mjm.sensor_type == SensorType.JOINTLIMITPOS) | (mjm.sensor_type == SensorType.TENDONLIMITPOS))[0], dtype=int
  )
  sensor_vel = mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_VEL
  sensor_vel &= (mjm.sensor_type != SensorType.JOINTLIMITVEL) & (mjm.sensor_type != SensorType.TENDONLIMITVEL)
  model["sensor_vel_adr"] = wp.array(np.nonzero(sensor_vel)[0], dtype=int)
  model["sensor_limitvel_adr"] = wp.array(
    np.nonzero((mjm.sensor_type == SensorType.JOINTLIMITVEL) | (mjm.sensor_type == SensorType.JOINTLIMITVEL))[0], dtype=int
  )
  sensor_acc = mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_ACC
  sensor_acc &= (mjm.sensor_type != SensorType.JOINTLIMITFRC) & (mjm.sensor_type != SensorType.TENDONLIMITFRC)
  sensor_acc &= mjm.sensor_type != SensorType.TENDONACTFRC
  model["sensor_acc_adr"] = wp.array(np.nonzero(sensor_acc)[0], dtype=int)
  sensor_rangefinder = mjm.sensor_type == SensorType.RANGEFINDER
  sensor_rangefinder_adr = np.nonzero(sensor_rangefinder)[0]
  model["sensor_rangefinder_adr"] = wp.array(sensor_rangefinder_adr, dtype=int)
  rangefinder_sensor_adr = np.full(mjm.nsensor, -1)
  rangefinder_sensor_adr[sensor_rangefinder_adr] = np.arange(len(sensor_rangefinder_adr))
  model["rangefinder_sensor_adr"] = wp.array(rangefinder_sensor_adr, dtype=int)
  model["sensor_rangefinder_bodyid"] = wp.array(mjm.site_bodyid[mjm.sensor_objid[sensor_rangefinder]], dtype=int)
  model["sensor_touch_adr"] = wp.array(np.nonzero(mjm.sensor_type == SensorType.TOUCH)[0], dtype=int)
  model["sensor_limitfrc_adr"] = wp.array(
    np.nonzero((mjm.sensor_type == SensorType.JOINTLIMITFRC) | (mjm.sensor_type == SensorType.TENDONLIMITFRC))[0], dtype=int
  )
  model["sensor_e_potential"] = (mjm.sensor_type == SensorType.E_POTENTIAL).any()
  model["sensor_e_kinetic"] = (mjm.sensor_type == SensorType.E_KINETIC).any()
  model["sensor_tendonactfrc_adr"] = wp.array(np.nonzero(mjm.sensor_type == SensorType.TENDONACTFRC)[0], dtype=int)
  model["sensor_subtree_vel"] = np.isin(mjm.sensor_type, (SensorType.SUBTREELINVEL, SensorType.SUBTREEANGMOM)).any()
  model["sensor_contact_adr"] = wp.array(np.nonzero(mjm.sensor_type == SensorType.CONTACT)[0], dtype=int)
  sensor_adr_to_contact_adr = np.clip(np.cumsum(mjm.sensor_type == SensorType.CONTACT) - 1, a_min=0, a_max=None)
  model["sensor_adr_to_contact_adr"] = wp.array(sensor_adr_to_contact_adr, dtype=int)
  rne_sensors = (SensorType.ACCELEROMETER, SensorType.FORCE, SensorType.TORQUE, SensorType.FRAMELINACC, SensorType.FRAMEANGACC)
  model["sensor_rne_postconstraint"] = np.isin(mjm.sensor_type, rne_sensors).any()

  # mocap
  mocap_bodyid = np.arange(mjm.nbody)[mjm.body_mocapid >= 0]
  model["mocap_bodyid"] = mocap_bodyid[mjm.body_mocapid[mjm.body_mocapid >= 0].argsort()]

  # sdf / taxels
  model["has_sdf_geom"] = np.any(mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF)
  model["taxel_vertadr"] = wp.array(
    [
      j + mjm.mesh_vertadr[mjm.sensor_objid[i]]
      for i in range(mjm.nsensor)
      if mjm.sensor_type[i] == SensorType.TACTILE
      for j in range(mjm.mesh_vertnum[mjm.sensor_objid[i]])
    ],
    dtype=int,
  )
  model["taxel_sensorid"] = wp.array(
    [
      i
      for i in range(mjm.nsensor)
      if mjm.sensor_type[i] == SensorType.TACTILE
      for _ in range(mjm.mesh_vertnum[mjm.sensor_objid[i]])
    ],
    dtype=int,
  )

  return Model(**model)


def make_data(mjm: mujoco.MjModel, nworld: int = 1, nconmax: int = -1, njmax: int = -1) -> Data:
  """
  Creates a data object on device.

  Args:
    mjm (mujoco.MjModel): The model containing kinematic and dynamic information (host).
    nworld (int, optional): Number of worlds. Defaults to 1.
    nconmax (int, optional): Maximum number of contacts for all worlds. Defaults to -1.
    njmax (int, optional): Maximum number of constraints per world. Defaults to -1.

  Returns:
    Data: The data object containing the current state and output arrays (device).
  """

  # TODO(team): move to Model?
  if nconmax == -1:
    # TODO(team): heuristic for nconmax
    nconmax = nworld * 20
  if njmax == -1:
    # TODO(team): heuristic for njmax
    njmax = 20 * 6

  if nworld < 1 or nworld > MAX_WORLDS:
    raise ValueError(f"nworld must be >= 1 and <= {MAX_WORLDS}")

  if nconmax < 0:
    raise ValueError("nconmax must be >= 0")

  if njmax < 0:
    raise ValueError("njmax must be >= 0")

  condim = np.concatenate((mjm.geom_condim, mjm.pair_dim))
  condim_max = np.max(condim) if len(condim) > 0 else 0

  max_npolygon = _max_npolygon(mjm)
  max_meshdegree = _max_meshdegree(mjm)

  if mujoco.mj_isSparse(mjm):
    qM = wp.zeros((nworld, 1, mjm.nM), dtype=float)
    qLD = wp.zeros((nworld, 1, mjm.nC), dtype=float)
    qM_integration = wp.zeros((nworld, 1, mjm.nM), dtype=float)
    qLD_integration = wp.zeros((nworld, 1, mjm.nM), dtype=float)
  else:
    qM = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float)
    qLD = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float)
    qM_integration = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float)
    qLD_integration = wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float)

  nsensorcontact = np.sum(mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT)
  nrangefinder = sum(mjm.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER)

  return Data(
    nworld=nworld,
    nconmax=nconmax,
    njmax=njmax,
    solver_niter=wp.zeros(nworld, dtype=int),
    ncon=wp.zeros(1, dtype=int),
    ne=wp.zeros(nworld, dtype=int),
    ne_connect=wp.zeros(nworld, dtype=int),  # warp only
    ne_weld=wp.zeros(nworld, dtype=int),  # warp only
    ne_jnt=wp.zeros(nworld, dtype=int),  # warp only
    ne_ten=wp.zeros(nworld, dtype=int),  # warp only
    nf=wp.zeros(nworld, dtype=int),
    nl=wp.zeros(nworld, dtype=int),
    nefc=wp.zeros(nworld, dtype=int),
    nsolving=wp.zeros(1, dtype=int),  # warp only
    time=wp.zeros(nworld, dtype=float),
    energy=wp.zeros(nworld, dtype=wp.vec2),
    qpos=wp.zeros((nworld, mjm.nq), dtype=float),
    qvel=wp.zeros((nworld, mjm.nv), dtype=float),
    act=wp.zeros((nworld, mjm.na), dtype=float),
    qacc_warmstart=wp.zeros((nworld, mjm.nv), dtype=float),
    qacc_discrete=wp.zeros((nworld, mjm.nv), dtype=float),
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
    geom_skip=wp.zeros(mjm.ngeom, dtype=bool),  # warp only
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
    cinert=wp.zeros((nworld, mjm.nbody), dtype=vec10),
    flexvert_xpos=wp.zeros((nworld, mjm.nflexvert), dtype=wp.vec3),
    flexedge_length=wp.zeros((nworld, mjm.nflexedge), dtype=wp.float32),
    flexedge_velocity=wp.zeros((nworld, mjm.nflexedge), dtype=wp.float32),
    actuator_length=wp.zeros((nworld, mjm.nu), dtype=float),
    actuator_moment=wp.zeros((nworld, mjm.nu, mjm.nv), dtype=float),
    crb=wp.zeros((nworld, mjm.nbody), dtype=vec10),
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
    qfrc_inverse=wp.zeros((nworld, mjm.nv), dtype=float),
    contact=Contact(
      dist=wp.zeros((nconmax,), dtype=float),
      pos=wp.zeros((nconmax,), dtype=wp.vec3f),
      frame=wp.zeros((nconmax,), dtype=wp.mat33f),
      includemargin=wp.zeros((nconmax,), dtype=float),
      friction=wp.zeros((nconmax,), dtype=vec5),
      solref=wp.zeros((nconmax,), dtype=wp.vec2f),
      solreffriction=wp.zeros((nconmax,), dtype=wp.vec2f),
      solimp=wp.zeros((nconmax,), dtype=vec5),
      dim=wp.zeros((nconmax,), dtype=int),
      geom=wp.zeros((nconmax,), dtype=wp.vec2i),
      efc_address=wp.zeros(
        (nconmax, np.maximum(1, 2 * (condim_max - 1))),
        dtype=int,
      ),
      worldid=wp.zeros((nconmax,), dtype=int),
    ),
    efc=Constraint(
      type=wp.zeros((nworld, njmax), dtype=int),
      id=wp.zeros((nworld, njmax), dtype=int),
      J=wp.zeros((nworld, njmax, mjm.nv), dtype=float),
      pos=wp.zeros((nworld, njmax), dtype=float),
      margin=wp.zeros((nworld, njmax), dtype=float),
      D=wp.zeros((nworld, njmax), dtype=float),
      vel=wp.zeros((nworld, njmax), dtype=float),
      aref=wp.zeros((nworld, njmax), dtype=float),
      frictionloss=wp.zeros((nworld, njmax), dtype=float),
      force=wp.zeros((nworld, njmax), dtype=float),
      Jaref=wp.zeros((nworld, njmax), dtype=float),
      Ma=wp.zeros((nworld, mjm.nv), dtype=float),
      grad=wp.zeros((nworld, mjm.nv), dtype=float),
      cholesky_L_tmp=wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float),
      cholesky_y_tmp=wp.zeros((nworld, mjm.nv), dtype=float),
      grad_dot=wp.zeros((nworld,), dtype=float),
      Mgrad=wp.zeros((nworld, mjm.nv), dtype=float),
      search=wp.zeros((nworld, mjm.nv), dtype=float),
      search_dot=wp.zeros((nworld,), dtype=float),
      gauss=wp.zeros((nworld,), dtype=float),
      cost=wp.zeros((nworld,), dtype=float),
      prev_cost=wp.zeros((nworld,), dtype=float),
      state=wp.zeros((nworld, njmax), dtype=int),
      mv=wp.zeros((nworld, mjm.nv), dtype=float),
      jv=wp.zeros((nworld, njmax), dtype=float),
      quad=wp.zeros((nworld, njmax), dtype=wp.vec3f),
      quad_gauss=wp.zeros((nworld,), dtype=wp.vec3f),
      h=wp.zeros((nworld, mjm.nv, mjm.nv), dtype=float),
      alpha=wp.zeros((nworld,), dtype=float),
      prev_grad=wp.zeros((nworld, mjm.nv), dtype=float),
      prev_Mgrad=wp.zeros((nworld, mjm.nv), dtype=float),
      beta=wp.zeros((nworld,), dtype=float),
      done=wp.zeros((nworld,), dtype=bool),
      # linesearch
      cost_candidate=wp.zeros((nworld, mjm.opt.ls_iterations), dtype=float),
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
    qM_integration=qM_integration,
    qLD_integration=qLD_integration,
    qLDiagInv_integration=wp.zeros((nworld, mjm.nv), dtype=float),
    # sweep-and-prune broadphase
    sap_projection_lower=wp.zeros((nworld, mjm.ngeom, 2), dtype=float),
    sap_projection_upper=wp.zeros((nworld, mjm.ngeom), dtype=float),
    sap_sort_index=wp.zeros((nworld, mjm.ngeom, 2), dtype=int),
    sap_range=wp.zeros((nworld, mjm.ngeom), dtype=int),
    sap_cumulative_sum=wp.zeros((nworld, mjm.ngeom), dtype=int),
    sap_segment_index=wp.array(
      np.array([i * mjm.ngeom if i < nworld + 1 else 0 for i in range(2 * nworld)]).reshape((nworld, 2)), dtype=int
    ),
    # collision driver
    collision_pair=wp.zeros((nconmax,), dtype=wp.vec2i),
    collision_pairid=wp.zeros((nconmax,), dtype=int),
    collision_worldid=wp.zeros((nconmax,), dtype=int),
    ncollision=wp.zeros((1,), dtype=int),
    # narrowphase (EPA polytope)
    epa_vert=wp.zeros(shape=(nconmax, 5 + mjm.opt.ccd_iterations), dtype=wp.vec3),
    epa_vert1=wp.zeros(shape=(nconmax, 5 + mjm.opt.ccd_iterations), dtype=wp.vec3),
    epa_vert2=wp.zeros(shape=(nconmax, 5 + mjm.opt.ccd_iterations), dtype=wp.vec3),
    epa_vert_index1=wp.zeros(shape=(nconmax, 5 + mjm.opt.ccd_iterations), dtype=int),
    epa_vert_index2=wp.zeros(shape=(nconmax, 5 + mjm.opt.ccd_iterations), dtype=int),
    epa_face=wp.zeros(shape=(nconmax, 6 + MJ_MAX_EPAFACES * mjm.opt.ccd_iterations), dtype=wp.vec3i),
    epa_pr=wp.zeros(shape=(nconmax, 6 + MJ_MAX_EPAFACES * mjm.opt.ccd_iterations), dtype=wp.vec3),
    epa_norm2=wp.zeros(shape=(nconmax, 6 + MJ_MAX_EPAFACES * mjm.opt.ccd_iterations), dtype=float),
    epa_index=wp.zeros(shape=(nconmax, 6 + MJ_MAX_EPAFACES * mjm.opt.ccd_iterations), dtype=int),
    epa_map=wp.zeros(shape=(nconmax, 6 + MJ_MAX_EPAFACES * mjm.opt.ccd_iterations), dtype=int),
    epa_horizon=wp.zeros(shape=(nconmax, 2 * MJ_MAX_EPAHORIZON), dtype=int),
    multiccd_polygon=wp.zeros(shape=(nconmax, 2 * max_npolygon), dtype=wp.vec3),
    multiccd_clipped=wp.zeros(shape=(nconmax, 2 * max_npolygon), dtype=wp.vec3),
    multiccd_pnormal=wp.zeros(shape=(nconmax, max_npolygon), dtype=wp.vec3),
    multiccd_pdist=wp.zeros(shape=(nconmax, max_npolygon), dtype=float),
    multiccd_idx1=wp.zeros(shape=(nconmax, max_meshdegree), dtype=int),
    multiccd_idx2=wp.zeros(shape=(nconmax, max_meshdegree), dtype=int),
    multiccd_n1=wp.zeros(shape=(nconmax, max_meshdegree), dtype=wp.vec3),
    multiccd_n2=wp.zeros(shape=(nconmax, max_meshdegree), dtype=wp.vec3),
    multiccd_endvert=wp.zeros(shape=(nconmax, max_meshdegree), dtype=wp.vec3),
    multiccd_face1=wp.zeros(shape=(nconmax, max_npolygon), dtype=wp.vec3),
    multiccd_face2=wp.zeros(shape=(nconmax, max_npolygon), dtype=wp.vec3),
    # rne_postconstraint
    cacc=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    cfrc_int=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    cfrc_ext=wp.zeros((nworld, mjm.nbody), dtype=wp.spatial_vector),
    # tendon
    ten_length=wp.zeros((nworld, mjm.ntendon), dtype=float),
    ten_J=wp.zeros((nworld, mjm.ntendon, mjm.nv), dtype=float),
    ten_Jdot=wp.zeros((nworld, mjm.ntendon, mjm.nv), dtype=float),
    ten_bias_coef=wp.zeros((nworld, mjm.ntendon), dtype=float),
    ten_wrapadr=wp.zeros((nworld, mjm.ntendon), dtype=int),
    ten_wrapnum=wp.zeros((nworld, mjm.ntendon), dtype=int),
    ten_actfrc=wp.zeros((nworld, mjm.ntendon), dtype=float),
    wrap_obj=wp.zeros((nworld, mjm.nwrap), dtype=wp.vec2i),
    wrap_xpos=wp.zeros((nworld, mjm.nwrap), dtype=wp.spatial_vector),
    wrap_geom_xpos=wp.zeros((nworld, mjm.nwrap), dtype=wp.spatial_vector),
    # sensors
    sensordata=wp.zeros((nworld, mjm.nsensordata), dtype=float),
    sensor_rangefinder_pnt=wp.zeros((nworld, nrangefinder), dtype=wp.vec3),
    sensor_rangefinder_vec=wp.zeros((nworld, nrangefinder), dtype=wp.vec3),
    sensor_rangefinder_dist=wp.zeros((nworld, nrangefinder), dtype=float),
    sensor_rangefinder_geomid=wp.zeros((nworld, nrangefinder), dtype=int),
    sensor_contact_nmatch=wp.zeros((nworld, nsensorcontact), dtype=int),
    sensor_contact_matchid=wp.zeros((nworld, nsensorcontact, MJ_MAXCONPAIR), dtype=int),
    sensor_contact_criteria=wp.zeros((nworld, nsensorcontact, MJ_MAXCONPAIR), dtype=float),
    sensor_contact_direction=wp.zeros((nworld, nsensorcontact, MJ_MAXCONPAIR), dtype=float),
    # ray
    ray_bodyexclude=wp.zeros(1, dtype=int),
    ray_dist=wp.zeros((nworld, 1), dtype=float),
    ray_geomid=wp.zeros((nworld, 1), dtype=int),
    # mul_m
    energy_vel_mul_m_skip=wp.zeros((nworld,), dtype=bool),
    inverse_mul_m_skip=wp.zeros((nworld,), dtype=bool),
    # actuator
    actuator_trntype_body_ncon=wp.zeros((nworld, np.sum(mjm.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY)), dtype=int),
  )


def put_data(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nworld: Optional[int] = None,
  nconmax: Optional[int] = None,
  njmax: Optional[int] = None,
) -> Data:
  """
  Moves data from host to a device.

  Args:
    mjm (mujoco.MjModel): The model containing kinematic and dynamic information (host).
    mjd (mujoco.MjData): The data object containing current state and output arrays (host).
    nworld (int, optional): The number of worlds. Defaults to 1.
    nconmax (int, optional): The maximum number of contacts for all worlds. Defaults to -1.
    njmax (int, optional): The maximum number of constraints per world. Defaults to -1.

  Returns:
    Data: The data object containing the current state and output arrays (device).
  """
  # TODO(team): move nconmax and njmax to Model?
  # TODO(team): decide what to do about uninitialized warp-only fields created by put_data
  #             we need to ensure these are only workspace fields and don't carry state

  nworld = nworld or 1
  # TODO(team): better heuristic for nconmax
  nconmax = nconmax or max(512, 4 * mjd.ncon * nworld)
  # TODO(team): better heuristic for njmax
  njmax = njmax or max(5, 4 * mjd.nefc)

  if nworld < 1 or nworld > MAX_WORLDS:
    raise ValueError(f"nworld must be >= 1 and <= {MAX_WORLDS}")

  if nconmax < 0:
    raise ValueError("nconmax must be >= 0")

  if njmax < 0:
    raise ValueError("njmax must be >= 0")

  if nworld * mjd.ncon > nconmax:
    raise ValueError(f"nconmax overflow (nconmax must be >= {nworld * mjd.ncon})")

  if mjd.nefc > njmax:
    raise ValueError(f"njmax overflow (njmax must be >= {mjd.nefc})")

  max_npolygon = _max_npolygon(mjm)
  max_meshdegree = _max_meshdegree(mjm)

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

  condim = np.concatenate((mjm.geom_condim, mjm.pair_dim))
  condim_max = np.max(condim) if len(condim) > 0 else 0
  contact_efc_address = np.zeros((nconmax, np.maximum(1, 2 * (condim_max - 1))), dtype=int)
  for i in range(nworld):
    for j in range(mjd.ncon):
      condim = mjd.contact.dim[j]
      efc_address = mjd.contact.efc_address[j]
      if efc_address == -1:
        continue
      if condim == 1:
        nconvar = 1
      else:
        nconvar = condim if mjm.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else 2 * (condim - 1)
      for k in range(nconvar):
        contact_efc_address[i * mjd.ncon + j, k] = mjd.nefc * i + efc_address + k

  contact_worldid = np.pad(np.repeat(np.arange(nworld), mjd.ncon), (0, nconmax - nworld * mjd.ncon))

  ne_connect = int(3 * np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_CONNECT) & mjd.eq_active))
  ne_weld = int(6 * np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_WELD) & mjd.eq_active))
  ne_jnt = int(np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_JOINT) & mjd.eq_active))
  ne_ten = int(np.sum((mjm.eq_type == mujoco.mjtEq.mjEQ_TENDON) & mjd.eq_active))

  efc_type_fill = np.zeros((nworld, njmax))
  efc_id_fill = np.zeros((nworld, njmax))
  efc_J_fill = np.zeros((nworld, njmax, mjm.nv))
  efc_D_fill = np.zeros((nworld, njmax))
  efc_vel_fill = np.zeros((nworld, njmax))
  efc_pos_fill = np.zeros((nworld, njmax))
  efc_aref_fill = np.zeros((nworld, njmax))
  efc_frictionloss_fill = np.zeros((nworld, njmax))
  efc_force_fill = np.zeros((nworld, njmax))
  efc_margin_fill = np.zeros((nworld, njmax))

  nefc = mjd.nefc
  efc_type_fill[:, :nefc] = np.tile(mjd.efc_type, (nworld, 1))
  efc_id_fill[:, :nefc] = np.tile(mjd.efc_id, (nworld, 1))
  efc_J_fill[:, :nefc, :] = np.tile(efc_J, (nworld, 1, 1))
  efc_D_fill[:, :nefc] = np.tile(mjd.efc_D, (nworld, 1))
  efc_vel_fill[:, :nefc] = np.tile(mjd.efc_vel, (nworld, 1))
  efc_pos_fill[:, :nefc] = np.tile(mjd.efc_pos, (nworld, 1))
  efc_aref_fill[:, :nefc] = np.tile(mjd.efc_aref, (nworld, 1))
  efc_frictionloss_fill[:, :nefc] = np.tile(mjd.efc_frictionloss, (nworld, 1))
  efc_force_fill[:, :nefc] = np.tile(mjd.efc_force, (nworld, 1))
  efc_margin_fill[:, :nefc] = np.tile(mjd.efc_margin, (nworld, 1))

  nsensorcontact = np.sum(mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT)
  nrangefinder = sum(mjm.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER)

  # some helper functions to simplify the data field definitions below

  def arr(x, dtype=None):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    if dtype is None:
      if np.issubdtype(x.dtype, np.integer):
        dtype = wp.int32
      elif np.issubdtype(x.dtype, np.floating):
        dtype = wp.float32
      elif np.issubdtype(x.dtype, bool):
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

  return Data(
    nworld=nworld,
    nconmax=nconmax,
    njmax=njmax,
    solver_niter=tile(mjd.solver_niter[0]),
    ncon=arr([mjd.ncon * nworld]),
    ne=wp.full(shape=(nworld), value=mjd.ne),
    ne_connect=wp.full(shape=(nworld), value=ne_connect),
    ne_weld=wp.full(shape=(nworld), value=ne_weld),
    ne_jnt=wp.full(shape=(nworld), value=ne_jnt),
    ne_ten=wp.full(shape=(nworld), value=ne_ten),
    nf=wp.full(shape=(nworld), value=mjd.nf),
    nl=wp.full(shape=(nworld), value=mjd.nl),
    nefc=wp.full(shape=(nworld), value=mjd.nefc),
    nsolving=arr([nworld]),
    time=arr(mjd.time * np.ones(nworld)),
    energy=tile(mjd.energy, dtype=wp.vec2),
    qpos=tile(mjd.qpos),
    qvel=tile(mjd.qvel),
    act=tile(mjd.act),
    qacc_warmstart=tile(mjd.qacc_warmstart),
    qacc_discrete=wp.zeros((nworld, mjm.nv), dtype=float),
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
    geom_skip=wp.zeros(mjm.ngeom, dtype=bool),  # warp only
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
    cinert=tile(mjd.cinert, dtype=vec10),
    flexvert_xpos=tile(mjd.flexvert_xpos, dtype=wp.vec3),
    flexedge_length=tile(mjd.flexedge_length),
    flexedge_velocity=tile(mjd.flexedge_velocity),
    actuator_length=tile(mjd.actuator_length),
    actuator_moment=tile(actuator_moment),
    crb=tile(mjd.crb, dtype=vec10),
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
    qfrc_inverse=tile(mjd.qfrc_inverse),
    contact=Contact(
      dist=padtile(mjd.contact.dist, nconmax),
      pos=padtile(mjd.contact.pos, nconmax, dtype=wp.vec3),
      frame=padtile(mjd.contact.frame, nconmax, dtype=wp.mat33),
      includemargin=padtile(mjd.contact.includemargin, nconmax),
      friction=padtile(mjd.contact.friction, nconmax, dtype=vec5),
      solref=padtile(mjd.contact.solref, nconmax, dtype=wp.vec2f),
      solreffriction=padtile(mjd.contact.solreffriction, nconmax, dtype=wp.vec2f),
      solimp=padtile(mjd.contact.solimp, nconmax, dtype=vec5),
      dim=padtile(mjd.contact.dim, nconmax),
      geom=padtile(mjd.contact.geom, nconmax, dtype=wp.vec2i),
      efc_address=arr(contact_efc_address),
      worldid=arr(contact_worldid),
    ),
    efc=Constraint(
      type=wp.array2d(efc_type_fill, dtype=int),
      id=wp.array2d(efc_id_fill, dtype=int),
      J=wp.array3d(efc_J_fill, dtype=float),
      pos=wp.array2d(efc_pos_fill, dtype=float),
      margin=wp.array2d(efc_margin_fill, dtype=float),
      D=wp.array2d(efc_D_fill, dtype=float),
      vel=wp.array2d(efc_vel_fill, dtype=float),
      aref=wp.array2d(efc_aref_fill, dtype=float),
      frictionloss=wp.array2d(efc_frictionloss_fill, dtype=float),
      force=wp.array2d(efc_force_fill, dtype=float),
      Jaref=wp.empty(shape=(nworld, njmax), dtype=float),
      Ma=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      grad=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      cholesky_L_tmp=wp.empty(shape=(nworld, mjm.nv, mjm.nv), dtype=float),
      cholesky_y_tmp=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      grad_dot=wp.empty(shape=(nworld,), dtype=float),
      Mgrad=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      search=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      search_dot=wp.empty(shape=(nworld,), dtype=float),
      gauss=wp.empty(shape=(nworld,), dtype=float),
      cost=wp.empty(shape=(nworld,), dtype=float),
      prev_cost=wp.empty(shape=(nworld,), dtype=float),
      state=wp.empty(shape=(nworld, njmax), dtype=int),
      mv=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      jv=wp.empty(shape=(nworld, njmax), dtype=float),
      quad=wp.empty(shape=(nworld, njmax), dtype=wp.vec3f),
      quad_gauss=wp.empty(shape=(nworld,), dtype=wp.vec3f),
      h=wp.empty(shape=(nworld, mjm.nv, mjm.nv), dtype=float),
      alpha=wp.empty(shape=(nworld,), dtype=float),
      prev_grad=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      prev_Mgrad=wp.empty(shape=(nworld, mjm.nv), dtype=float),
      beta=wp.empty(shape=(nworld,), dtype=float),
      done=wp.empty(shape=(nworld,), dtype=bool),
      cost_candidate=wp.empty(shape=(nworld, mjm.opt.ls_iterations), dtype=float),
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
    sap_projection_lower=wp.zeros((nworld, mjm.ngeom, 2), dtype=float),
    sap_projection_upper=wp.zeros((nworld, mjm.ngeom), dtype=float),
    sap_sort_index=wp.zeros((nworld, mjm.ngeom, 2), dtype=int),
    sap_range=wp.zeros((nworld, mjm.ngeom), dtype=int),
    sap_cumulative_sum=wp.zeros((nworld, mjm.ngeom), dtype=int),
    sap_segment_index=arr(np.array([i * mjm.ngeom if i < nworld + 1 else 0 for i in range(2 * nworld)]).reshape((nworld, 2))),
    # collision driver
    collision_pair=wp.empty(nconmax, dtype=wp.vec2i),
    collision_pairid=wp.empty(nconmax, dtype=int),
    collision_worldid=wp.empty(nconmax, dtype=int),
    ncollision=wp.zeros(1, dtype=int),
    # narrowphase (EPA polytope)
    epa_vert=wp.zeros(shape=(nconmax, 5 + mjm.opt.ccd_iterations), dtype=wp.vec3),
    epa_vert1=wp.zeros(shape=(nconmax, 5 + mjm.opt.ccd_iterations), dtype=wp.vec3),
    epa_vert2=wp.zeros(shape=(nconmax, 5 + mjm.opt.ccd_iterations), dtype=wp.vec3),
    epa_vert_index1=wp.zeros(shape=(nconmax, 5 + mjm.opt.ccd_iterations), dtype=int),
    epa_vert_index2=wp.zeros(shape=(nconmax, 5 + mjm.opt.ccd_iterations), dtype=int),
    epa_face=wp.zeros(shape=(nconmax, 6 + MJ_MAX_EPAFACES * mjm.opt.ccd_iterations), dtype=wp.vec3i),
    epa_pr=wp.zeros(shape=(nconmax, 6 + MJ_MAX_EPAFACES * mjm.opt.ccd_iterations), dtype=wp.vec3),
    epa_norm2=wp.zeros(shape=(nconmax, 6 + MJ_MAX_EPAFACES * mjm.opt.ccd_iterations), dtype=float),
    epa_index=wp.zeros(shape=(nconmax, 6 + MJ_MAX_EPAFACES * mjm.opt.ccd_iterations), dtype=int),
    epa_map=wp.zeros(shape=(nconmax, 6 + MJ_MAX_EPAFACES * mjm.opt.ccd_iterations), dtype=int),
    epa_horizon=wp.zeros(shape=(nconmax, 2 * MJ_MAX_EPAHORIZON), dtype=int),
    multiccd_polygon=wp.zeros(shape=(nconmax, 2 * max_npolygon), dtype=wp.vec3),
    multiccd_clipped=wp.zeros(shape=(nconmax, 2 * max_npolygon), dtype=wp.vec3),
    multiccd_pnormal=wp.zeros(shape=(nconmax, max_npolygon), dtype=wp.vec3),
    multiccd_pdist=wp.zeros(shape=(nconmax, max_npolygon), dtype=float),
    multiccd_idx1=wp.zeros(shape=(nconmax, max_meshdegree), dtype=int),
    multiccd_idx2=wp.zeros(shape=(nconmax, max_meshdegree), dtype=int),
    multiccd_n1=wp.zeros(shape=(nconmax, max_meshdegree), dtype=wp.vec3),
    multiccd_n2=wp.zeros(shape=(nconmax, max_meshdegree), dtype=wp.vec3),
    multiccd_endvert=wp.zeros(shape=(nconmax, max_meshdegree), dtype=wp.vec3),
    multiccd_face1=wp.zeros(shape=(nconmax, max_npolygon), dtype=wp.vec3),
    multiccd_face2=wp.zeros(shape=(nconmax, max_npolygon), dtype=wp.vec3),
    # rne_postconstraint but also smooth
    cacc=tile(mjd.cacc, dtype=wp.spatial_vector),
    cfrc_int=tile(mjd.cfrc_int, dtype=wp.spatial_vector),
    cfrc_ext=tile(mjd.cfrc_ext, dtype=wp.spatial_vector),
    # tendon
    ten_length=tile(mjd.ten_length),
    ten_J=tile(ten_J),
    ten_Jdot=wp.zeros((nworld, mjm.ntendon, mjm.nv), dtype=float),
    ten_bias_coef=wp.zeros((nworld, mjm.ntendon), dtype=float),
    ten_wrapadr=tile(mjd.ten_wrapadr),
    ten_wrapnum=tile(mjd.ten_wrapnum),
    ten_actfrc=wp.zeros((nworld, mjm.ntendon), dtype=float),
    wrap_obj=tile(mjd.wrap_obj, dtype=wp.vec2i),
    wrap_xpos=tile(mjd.wrap_xpos, dtype=wp.spatial_vector),
    wrap_geom_xpos=wp.zeros((nworld, mjm.nwrap), dtype=wp.spatial_vector),
    # sensors
    sensordata=tile(mjd.sensordata),
    sensor_rangefinder_pnt=wp.zeros((nworld, nrangefinder), dtype=wp.vec3),
    sensor_rangefinder_vec=wp.zeros((nworld, nrangefinder), dtype=wp.vec3),
    sensor_rangefinder_dist=wp.zeros((nworld, nrangefinder), dtype=float),
    sensor_rangefinder_geomid=wp.zeros((nworld, nrangefinder), dtype=int),
    sensor_contact_nmatch=wp.zeros((nworld, nsensorcontact), dtype=int),
    sensor_contact_matchid=wp.zeros((nworld, nsensorcontact, MJ_MAXCONPAIR), dtype=int),
    sensor_contact_criteria=wp.zeros((nworld, nsensorcontact, MJ_MAXCONPAIR), dtype=float),
    sensor_contact_direction=wp.zeros((nworld, nsensorcontact, MJ_MAXCONPAIR), dtype=float),
    # ray
    ray_bodyexclude=wp.zeros(1, dtype=int),
    ray_dist=wp.zeros((nworld, 1), dtype=float),
    ray_geomid=wp.zeros((nworld, 1), dtype=int),
    # mul_m
    energy_vel_mul_m_skip=wp.zeros((nworld,), dtype=bool),
    inverse_mul_m_skip=wp.zeros((nworld,), dtype=bool),
    # actuator
    actuator_trntype_body_ncon=wp.zeros((nworld, np.sum(mjm.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY)), dtype=int),
  )


def get_data_into(
  result: mujoco.MjData,
  mjm: mujoco.MjModel,
  d: Data,
):
  """Gets data from a device into an existing mujoco.MjData.

  Args:
    result (mujoco.MjData): The data object containing the current state and output arrays
                            (host).
    mjm (mujoco.MjModel): The model containing kinematic and dynamic information (host).
    d (Data): The data object containing the current state and output arrays (device).
  """
  if d.nworld > 1:
    raise NotImplementedError("only nworld == 1 supported for now")

  result.solver_niter[0] = d.solver_niter.numpy()[0]

  ncon = d.ncon.numpy()[0]
  nefc = d.nefc.numpy()[0]

  if ncon != result.ncon or nefc != result.nefc:
    mujoco._functions._realloc_con_efc(result, ncon=ncon, nefc=nefc)

  result.time = d.time.numpy()[0]
  result.energy = d.energy.numpy()[0]
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
  result.flexvert_xpos = d.flexvert_xpos.numpy()[0]
  result.flexedge_length = d.flexedge_length.numpy()[0]
  result.flexedge_velocity = d.flexedge_velocity.numpy()[0]
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
  result.qfrc_inverse = d.qfrc_inverse.numpy()[0]
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


@wp.kernel
def _reset_nworld(
  # Model:
  nq: int,
  nv: int,
  nu: int,
  na: int,
  neq: int,
  nsensordata: int,
  qpos0: wp.array2d(dtype=float),
  eq_active0: wp.array(dtype=bool),
  # Data in:
  nworld_in: int,
  # Data out:
  solver_niter_out: wp.array(dtype=int),
  ncon_out: wp.array(dtype=int),
  ne_out: wp.array(dtype=int),
  ne_connect_out: wp.array(dtype=int),
  ne_weld_out: wp.array(dtype=int),
  ne_jnt_out: wp.array(dtype=int),
  ne_ten_out: wp.array(dtype=int),
  nf_out: wp.array(dtype=int),
  nl_out: wp.array(dtype=int),
  nefc_out: wp.array(dtype=int),
  nsolving_out: wp.array(dtype=int),
  time_out: wp.array(dtype=float),
  energy_out: wp.array(dtype=wp.vec2),
  qpos_out: wp.array2d(dtype=float),
  qvel_out: wp.array2d(dtype=float),
  act_out: wp.array2d(dtype=float),
  qacc_warmstart_out: wp.array2d(dtype=float),
  ctrl_out: wp.array2d(dtype=float),
  qfrc_applied_out: wp.array2d(dtype=float),
  eq_active_out: wp.array2d(dtype=bool),
  qacc_out: wp.array2d(dtype=float),
  act_dot_out: wp.array2d(dtype=float),
  sensordata_out: wp.array2d(dtype=float),
):
  worldid = wp.tid()

  solver_niter_out[worldid] = 0
  if worldid == 0:
    ncon_out[0] = 0
  ne_out[worldid] = 0
  ne_connect_out[worldid] = 0
  ne_weld_out[worldid] = 0
  ne_jnt_out[worldid] = 0
  ne_ten_out[worldid] = 0
  nf_out[worldid] = 0
  nl_out[worldid] = 0
  nefc_out[worldid] = 0
  if worldid == 0:
    nsolving_out[0] = nworld_in
  time_out[worldid] = 0.0
  energy_out[worldid] = wp.vec2(0.0, 0.0)
  for i in range(nq):
    qpos_out[worldid, i] = qpos0[worldid, i]
    if i < nv:
      qvel_out[worldid, i] = 0.0
      qacc_warmstart_out[worldid, i] = 0.0
      qfrc_applied_out[worldid, i] = 0.0
      qacc_out[worldid, i] = 0.0
  for i in range(nu):
    ctrl_out[worldid, i] = 0.0
    if i < na:
      act_out[worldid, i] = 0.0
      act_dot_out[worldid, i] = 0.0
  for i in range(neq):
    eq_active_out[worldid, i] = eq_active0[i]
  for i in range(nsensordata):
    sensordata_out[worldid, i] = 0.0


@wp.kernel
def _reset_mocap(
  # Model:
  body_mocapid: wp.array(dtype=int),
  body_pos: wp.array2d(dtype=wp.vec3),
  body_quat: wp.array2d(dtype=wp.quat),
  # Data out:
  mocap_pos_out: wp.array2d(dtype=wp.vec3),
  mocap_quat_out: wp.array2d(dtype=wp.quat),
):
  worldid, bodyid = wp.tid()

  mocapid = body_mocapid[bodyid]

  if mocapid >= 0:
    mocap_pos_out[worldid, mocapid] = body_pos[worldid, bodyid]
    mocap_quat_out[worldid, mocapid] = body_quat[worldid, bodyid]


@wp.kernel
def _reset_contact(
  # Data in:
  ncon_in: wp.array(dtype=int),
  # In:
  nefcaddress: int,
  # Data out:
  contact_dist_out: wp.array(dtype=float),
  contact_pos_out: wp.array(dtype=wp.vec3),
  contact_frame_out: wp.array(dtype=wp.mat33),
  contact_includemargin_out: wp.array(dtype=float),
  contact_friction_out: wp.array(dtype=vec5),
  contact_solref_out: wp.array(dtype=wp.vec2),
  contact_solreffriction_out: wp.array(dtype=wp.vec2),
  contact_solimp_out: wp.array(dtype=vec5),
  contact_dim_out: wp.array(dtype=int),
  contact_geom_out: wp.array(dtype=wp.vec2i),
  contact_efc_address_out: wp.array2d(dtype=int),
  contact_worldid_out: wp.array(dtype=int),
):
  conid = wp.tid()

  if conid >= ncon_in[0]:
    return

  contact_dist_out[conid] = 0.0
  contact_pos_out[conid] = wp.vec3(0.0)
  contact_frame_out[conid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  contact_includemargin_out[conid] = 0.0
  contact_friction_out[conid] = vec5(0.0, 0.0, 0.0, 0.0, 0.0)
  contact_solref_out[conid] = wp.vec2(0.0, 0.0)
  contact_solreffriction_out[conid] = wp.vec2(0.0, 0.0)
  contact_solimp_out[conid] = vec5(0.0, 0.0, 0.0, 0.0, 0.0)
  contact_dim_out[conid] = 0
  contact_geom_out[conid] = wp.vec2i(0, 0)
  for i in range(nefcaddress):
    contact_efc_address_out[conid, i] = 0
  contact_worldid_out[conid] = 0


def reset_data(m: Model, d: Data):
  """Clear data, set defaults."""
  d.xfrc_applied.zero_()
  d.qM.zero_()

  # set mocap_pos/quat = body_pos/quat for mocap bodies
  wp.launch(
    _reset_mocap, dim=(d.nworld, m.nbody), inputs=[m.body_mocapid, m.body_pos, m.body_quat], outputs=[d.mocap_pos, d.mocap_quat]
  )

  # clear contacts
  wp.launch(
    _reset_contact,
    dim=d.nconmax,
    inputs=[d.ncon, d.contact.efc_address.shape[1]],
    outputs=[
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.includemargin,
      d.contact.friction,
      d.contact.solref,
      d.contact.solreffriction,
      d.contact.solimp,
      d.contact.dim,
      d.contact.geom,
      d.contact.efc_address,
      d.contact.worldid,
    ],
  )

  wp.launch(
    _reset_nworld,
    dim=d.nworld,
    inputs=[m.nq, m.nv, m.nu, m.na, m.neq, m.nsensordata, m.qpos0, m.eq_active0, d.nworld],
    outputs=[
      d.solver_niter,
      d.ncon,
      d.ne,
      d.ne_connect,
      d.ne_weld,
      d.ne_jnt,
      d.ne_ten,
      d.nf,
      d.nl,
      d.nefc,
      d.nsolving,
      d.time,
      d.energy,
      d.qpos,
      d.qvel,
      d.act,
      d.qacc_warmstart,
      d.ctrl,
      d.qfrc_applied,
      d.eq_active,
      d.qacc,
      d.act_dot,
      d.sensordata,
    ],
  )


def override_model(model: Union[Model, mujoco.MjModel], overrides: Union[dict[str, Any], Sequence[str]]):
  """Overrides model parameters.

  Overrides are of the format:
    opt.iterations = 1
    opt.ls_parallel = True
    opt.cone = pyramidal
    opt.disableflags = contact | spring
  """

  enum_fields = {
    "opt.broadphase": BroadphaseType,
    "opt.broadphase_filter": BroadphaseFilter,
    "opt.cone": ConeType,
    "opt.disableflags": DisableBit,
    "opt.enableflags": EnableBit,
    "opt.integrator": IntegratorType,
    "opt.solver": SolverType,
  }
  mjw_only_fields = {"opt.broadphase", "opt.broadphase_filter", "opt.ls_parallel", "opt.graph_conditional"}
  mj_only_fields = {"opt.jacobian"}

  if not isinstance(overrides, dict):
    overrides_dict = {}
    for override in overrides:
      if "=" not in override:
        raise ValueError(f"Invalid override format: {override}")
      k, v = override.split("=", 1)
      overrides_dict[k.strip()] = v.strip()
    overrides = overrides_dict

  for key, val in overrides.items():
    # skip overrides on MjModel for properties that are only on mjw.Model
    if key in mjw_only_fields and isinstance(model, mujoco.MjModel):
      continue
    if key in mj_only_fields and isinstance(model, Model):
      continue

    obj, attrs = model, key.split(".")
    for i, attr in enumerate(attrs):
      if not hasattr(obj, attr):
        raise ValueError(f"Unrecognized model field: {key}")
      if i < len(attrs) - 1:
        obj = getattr(obj, attr)
        continue

      typ = type(getattr(obj, attr))

      if key in enum_fields and isinstance(val, str):
        # special case: enum value
        enum_members = val.split("|")
        val = 0
        for enum_member in enum_members:
          enum_member = enum_member.strip().upper()
          if enum_member not in enum_fields[key].__members__:
            raise ValueError(f"Unrecognized enum value for {enum_fields[key].__name__}: {enum_member}")
          val |= int(enum_fields[key][enum_member])
      elif typ is bool and isinstance(val, str):
        # special case: "true", "TRUE", "false", "FALSE" etc.
        if val.upper() not in ("TRUE", "FALSE"):
          raise ValueError(f"Unrecognized value for field: {key}")
        val = val.upper() == "TRUE"
      else:
        val = typ(val)

      setattr(obj, attr, val)


def find_keys(model: mujoco.MjModel, keyname_prefix: str) -> list[int]:
  """Finds keyframes that start with keyname_prefix."""
  keys = []

  for keyid in range(model.nkey):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, keyid)
    if name.startswith(keyname_prefix):
      keys.append(keyid)

  return keys


def make_trajectory(model: mujoco.MjModel, keys: list[int]) -> np.ndarray:
  """Make a ctrl trajectory with linear interpolation."""
  ctrls = []
  prev_ctrl_key = np.zeros(model.nu, dtype=np.float64)
  prev_time, time = 0.0, 0.0

  for key in keys:
    ctrl_key, ctrl_time = model.key_ctrl[key], model.key_time[key]
    if not ctrls and ctrl_time != 0.0:
      raise ValueError("first keyframe must have time 0.0")
    elif ctrls and ctrl_time <= prev_time:
      raise ValueError("keyframes must be in time order")

    while time < ctrl_time:
      frac = (time - prev_time) / (ctrl_time - prev_time)
      ctrls.append(prev_ctrl_key * (1 - frac) + ctrl_key * frac)
      time += model.opt.timestep

    ctrls.append(ctrl_key)
    time += model.opt.timestep
    prev_ctrl_key = ctrl_key
    prev_time = time

  return np.array(ctrls)
