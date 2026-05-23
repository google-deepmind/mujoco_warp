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

import dataclasses
import warnings
from typing import Any, Optional, Sequence

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src import bvh
from mujoco_warp._src import math as mjmath
from mujoco_warp._src import render_util
from mujoco_warp._src import smooth
from mujoco_warp._src import types
from mujoco_warp._src import warp_util
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import vec10
from mujoco_warp._src.util_pkg import check_version

# TODO(team): remove after improving island solver performance
ENABLE_ISLANDS = False


def _is_array_spec(typ) -> bool:
  """Check if a type annotation is an array spec (wp.array instance or bracket annotation)."""
  return isinstance(typ, wp.array) or type(typ).__name__ == "_ArrayAnnotation"


def _create_array(data: Any, spec, sizes: dict[str, int]) -> wp.array | None:
  """Creates a warp array and populates it with data.

  The array shape is determined by a field spec referencing MjModel / MjData array sizes.
  """
  spec_shape = getattr(spec, "shape", (0,))
  shape = None
  if spec_shape != (0,):
    shape = tuple(sizes[dim] if isinstance(dim, str) else dim for dim in spec_shape)

  if data is None and shape is None:
    return None  # nothing to do
  elif data is None:
    array = wp.zeros(shape, dtype=spec.dtype)
  else:
    array = wp.array(np.array(data), dtype=spec.dtype, shape=shape)

  if spec_shape and spec_shape[0] == "*":
    # add private attribute for JAX to determine which fields are batched
    array._is_batched = True
    # also set stride 0 to 0 which is expected legacy behavior (but is deprecated)
    array.strides = (0,) + array.strides[1:]
  return array


def _create_constraint(
  mjm,
  nworld: int,
  njmax: int,
  njmax_nnz: int,
  sizes: dict,
  island_enabled: bool,
  mjd=None,
) -> types.Constraint:
  """Construct a types.Constraint with standard and island local fields allocated properly."""
  efc_kwargs = {"J_rownnz": None, "J_rowadr": None, "J_colind": None, "J": None}
  sparse = is_sparse(mjm)

  for f in dataclasses.fields(types.Constraint):
    if f.name == "itype":
      efc_kwargs[f.name] = wp.empty((nworld, njmax if island_enabled else 0), dtype=int)
    elif f.name == "iid":
      efc_kwargs[f.name] = wp.empty((nworld, njmax if island_enabled else 0), dtype=int)
    elif f.name == "iJ_rownnz":
      efc_kwargs[f.name] = wp.empty((nworld, njmax if island_enabled else 0) if sparse else (nworld, 0), dtype=int)
    elif f.name == "iJ_rowadr":
      efc_kwargs[f.name] = wp.empty((nworld, njmax if island_enabled else 0) if sparse else (nworld, 0), dtype=int)
    elif f.name == "iJ_colind":
      efc_kwargs[f.name] = wp.empty((nworld, 1, njmax_nnz if island_enabled else 0) if sparse else (nworld, 0, 0), dtype=int)
    elif f.name == "iJ":
      efc_kwargs[f.name] = wp.empty(
        (nworld, 1, njmax_nnz if island_enabled else 0) if sparse else (nworld, njmax if island_enabled else 0, mjm.nv),
        dtype=float,
      )
    elif f.name == "iD":
      efc_kwargs[f.name] = wp.empty((nworld, njmax if island_enabled else 0), dtype=float)
    elif f.name == "iaref":
      efc_kwargs[f.name] = wp.empty((nworld, njmax if island_enabled else 0), dtype=float)
    elif f.name == "ifrictionloss":
      efc_kwargs[f.name] = wp.empty((nworld, njmax if island_enabled else 0), dtype=float)
    elif f.name == "iforce":
      efc_kwargs[f.name] = wp.empty((nworld, njmax if island_enabled else 0), dtype=float)
    elif f.name == "istate":
      efc_kwargs[f.name] = wp.empty((nworld, njmax if island_enabled else 0), dtype=int)
    else:
      if f.name in efc_kwargs:
        continue

      if mjd is not None:
        shape = tuple(sizes[dim] if isinstance(dim, str) else dim for dim in f.type.shape)
        val = np.zeros(shape, dtype=f.type.dtype)
        if f.name in ("type", "id", "pos", "margin", "D", "vel", "aref", "frictionloss", "force"):
          val[:, : mjd.nefc] = np.tile(getattr(mjd, "efc_" + f.name), (nworld, 1))
        efc_kwargs[f.name] = wp.array(val, dtype=f.type.dtype)
      else:
        efc_kwargs[f.name] = _create_array(None, f.type, sizes)

  return types.Constraint(**efc_kwargs)


def is_sparse(mjm: mujoco.MjModel) -> bool:
  if mjm.opt.jacobian == mujoco.mjtJacobian.mjJAC_AUTO:
    if mjm.nv > 32:
      return True
    else:
      return False
  else:
    return bool(mujoco.mj_isSparse(mjm))


def put_model(mjm: mujoco.MjModel) -> types.Model:
  """Creates a model on device.

  Args:
    mjm: The model containing kinematic and dynamic information (host).

  Returns:
    The model containing kinematic and dynamic information (device).
  """
  # check for compatible cuda toolkit and driver versions
  warp_util.check_toolkit_driver()

  # model: check supported features in array types
  for field, field_type, mj_type in (
    (mjm.actuator_trntype, types.TrnType, mujoco.mjtTrn),
    (mjm.actuator_dyntype, types.DynType, mujoco.mjtDyn),
    (mjm.actuator_gaintype, types.GainType, mujoco.mjtGain),
    (mjm.actuator_biastype, types.BiasType, mujoco.mjtBias),
    (mjm.eq_type, types.EqType, mujoco.mjtEq),
    (mjm.geom_type, types.GeomType, mujoco.mjtGeom),
    (mjm.sensor_type, types.SensorType, mujoco.mjtSensor),
    (mjm.wrap_type, types.WrapType, mujoco.mjtWrap),
  ):
    missing = ~np.isin(field, field_type)
    if missing.any():
      names = [mj_type(v).name for v in field[missing]]
      raise NotImplementedError(f"{names} not supported.")

  # opt: check supported features in scalar types
  for field, field_type, mj_type in (
    (mjm.opt.integrator, types.IntegratorType, mujoco.mjtIntegrator),
    (mjm.opt.cone, types.ConeType, mujoco.mjtCone),
    (mjm.opt.solver, types.SolverType, mujoco.mjtSolver),
  ):
    if field not in set(field_type):
      raise NotImplementedError(f"{mj_type(field).name} is unsupported.")

  # opt: check supported features in scalar flag types
  for field, field_type, mj_type in (
    (mjm.opt.disableflags, types.DisableBit, mujoco.mjtDisableBit),
    (mjm.opt.enableflags, types.EnableBit, mujoco.mjtEnableBit),
  ):
    unsupported = field & ~np.bitwise_or.reduce(field_type)
    if unsupported:
      raise NotImplementedError(f"{mj_type(unsupported).name} is unsupported.")

  if mjm.opt.noslip_iterations > 0:
    raise NotImplementedError(f"noslip solver not implemented.")

  if (mjm.opt.viscosity > 0 or mjm.opt.density > 0) and mjm.opt.integrator in (
    mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    mujoco.mjtIntegrator.mjINT_IMPLICIT,
  ):
    raise NotImplementedError(f"Implicit integrators and fluid model not implemented.")

  if (mjm.body_plugin != -1).any():
    raise NotImplementedError("Body plugins not supported.")

  if (mjm.actuator_plugin != -1).any():
    raise NotImplementedError("Actuator plugins not supported.")

  if (mjm.sensor_plugin != -1).any():
    raise NotImplementedError("Sensor plugins not supported.")

  # array sizes may change in the future
  if mujoco.mjNPOLY != 2:
    warnings.warn(f"mujoco.mjNPOLY is {mujoco.mjNPOLY}, expected 2. Higher order polynomials may not be supported correctly.")

  # TODO(team): remove after _update_gradient for Newton uses tile operations for islands
  nv_max = 60
  if mjm.nv > nv_max and mjm.opt.jacobian == mujoco.mjtJacobian.mjJAC_DENSE:
    raise ValueError(f"Dense is unsupported for nv > {nv_max} (nv = {mjm.nv}).")

  collision_sensors = (mujoco.mjtSensor.mjSENS_GEOMDIST, mujoco.mjtSensor.mjSENS_GEOMNORMAL, mujoco.mjtSensor.mjSENS_GEOMFROMTO)
  is_collision_sensor = np.isin(mjm.sensor_type, collision_sensors)

  def not_implemented(objtype, objid, geomtype):
    if objtype == mujoco.mjtObj.mjOBJ_BODY:
      geomnum = mjm.body_geomnum[objid]
      geomadr = mjm.body_geomadr[objid]
      for geomid in range(geomadr, geomadr + geomnum):
        if mjm.geom_type[geomid] == geomtype:
          return True
    elif objtype == mujoco.mjtObj.mjOBJ_GEOM:
      if mjm.geom_type[objid] == geomtype:
        return True
    return False

  def _check_friction(name: str, id_: int, condim: int, friction, checks):
    for min_condim, indices in checks:
      if condim >= min_condim:
        for idx in indices:
          if friction[idx] < types.MJ_MINMU:
            warnings.warn(
              f"{name} {id_}: friction[{idx}] ({friction[idx]}) < MJ_MINMU ({types.MJ_MINMU}) with condim={condim} may cause NaN"
            )

  for geomid in range(mjm.ngeom):
    _check_friction("geom", geomid, mjm.geom_condim[geomid], mjm.geom_friction[geomid], [(3, [0]), (4, [1]), (6, [2])])

  for pairid in range(mjm.npair):
    _check_friction("pair", pairid, mjm.pair_dim[pairid], mjm.pair_friction[pairid], [(3, [0]), (4, [1, 2]), (6, [3, 4])])

  # create opt
  opt_kwargs = {f.name: getattr(mjm.opt, f.name, None) for f in dataclasses.fields(types.Option)}
  if hasattr(mjm.opt, "impratio"):
    opt_kwargs["impratio_invsqrt"] = 1.0 / np.sqrt(np.maximum(mjm.opt.impratio, mujoco.mjMINVAL))
  opt = types.Option(**opt_kwargs)

  # islands are disabled by default while performance is being improved
  # override by setting io.ENABLE_ISLANDS = True
  # TODO(team): remove after improving island solver performance
  if not ENABLE_ISLANDS:
    opt.disableflags |= types.DisableBit.ISLAND

  # C MuJoCo tolerance was chosen for float64 architecture, but we default to float32 on GPU
  # adjust the tolerance for lower precision, to avoid the solver spending iterations needlessly
  # bouncing around the optimal solution
  opt.tolerance = max(opt.tolerance, 1e-6)

  # warp only fields
  ls_parallel_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_NUMERIC, "ls_parallel")
  opt.ls_parallel = (ls_parallel_id > -1) and (mjm.numeric_data[mjm.numeric_adr[ls_parallel_id]] == 1)
  opt.ls_parallel_min_step = 1.0e-6  # TODO(team): determine good default setting
  opt.broadphase = types.BroadphaseType.NXN
  opt.broadphase_filter = types.BroadphaseFilter.PLANE | types.BroadphaseFilter.SPHERE | types.BroadphaseFilter.OBB
  opt.graph_conditional = True
  opt.run_collision_detection = True
  contact_sensor_maxmatch_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_NUMERIC, "contact_sensor_maxmatch")
  if contact_sensor_maxmatch_id > -1:
    opt.contact_sensor_maxmatch = mjm.numeric_data[mjm.numeric_adr[contact_sensor_maxmatch_id]]
  else:
    opt.contact_sensor_maxmatch = 64

  # place opt on device
  for f in dataclasses.fields(types.Option):
    if _is_array_spec(f.type):
      setattr(opt, f.name, _create_array(getattr(opt, f.name), f.type, {"*": 1}))
    else:
      setattr(opt, f.name, f.type(getattr(opt, f.name)))

  # create stat
  stat = types.Statistic(meaninertia=_create_array([mjm.stat.meaninertia], types.array("*", float), {"*": 1}))

  # create model
  m = types.Model(**{f.name: getattr(mjm, f.name, None) for f in dataclasses.fields(types.Model)})

  m.opt = opt
  m.stat = stat
  m.callback = types.Callback()

  m.nv_pad = _get_padded_sizes(
    mjm.nv, 0, is_sparse(mjm), types.TILE_SIZE_JTDAJ_SPARSE if is_sparse(mjm) else types.TILE_SIZE_JTDAJ_DENSE
  )[1]
  m.nacttrnbody = (mjm.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY).sum()
  m.nsensortaxel = mjm.mesh_vertnum[mjm.sensor_objid[mjm.sensor_type == mujoco.mjtSensor.mjSENS_TACTILE]].sum()
  m.nsensorcontact = (mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT).sum()
  m.nrangefinder = (mjm.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER).sum()
  condim_arrays = [np.array([0]), mjm.geom_condim, mjm.pair_dim]
  if mjm.nflex > 0:
    condim_arrays.append(mjm.flex_condim)
  m.nmaxcondim = np.concatenate(condim_arrays).max()
  m.nmaxpyramid = np.maximum(1, 2 * (m.nmaxcondim - 1))
  m.has_sdf_geom = (mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF).any()
  m.block_dim = types.BlockDim()
  m.is_sparse = is_sparse(mjm)
  m.has_fluid = mjm.opt.wind.any() or mjm.opt.density > 0 or mjm.opt.viscosity > 0

  # Use max(initial=0) to safely handle the empty-array case when ntendon=0.
  # np.ndarray.max() raises ValueError on zero-size arrays without an initial value.
  m.max_ten_J_rownnz = int(mjm.ten_J_rownnz.max(initial=0))

  # body ids grouped by tree level (depth-based traversal)
  bodies, body_depth = {}, np.zeros(mjm.nbody, dtype=int) - 1
  for i in range(mjm.nbody):
    body_depth[i] = body_depth[mjm.body_parentid[i]] + 1
    bodies.setdefault(body_depth[i], []).append(i)
  m.body_tree = tuple(wp.array(bodies[i], dtype=int) for i in sorted(bodies))

  # branch-based traversal data
  children_count = np.bincount(mjm.body_parentid[1:], minlength=mjm.nbody)
  ancestor_chain = lambda b: ancestor_chain(mjm.body_parentid[b]) + [b] if b else []
  branches = [ancestor_chain(l) for l in np.where(children_count[1:] == 0)[0] + 1]
  m.nbranch = len(branches)

  body_branches = []
  body_branch_start = []
  offset = 0

  for branch in branches:
    body_branches.extend(branch)
    body_branch_start.append(offset)
    offset += len(branch)
  body_branch_start.append(offset)

  m.body_branches = np.array(body_branches, dtype=int)
  m.body_branch_start = np.array(body_branch_start, dtype=int)

  m.mocap_bodyid = np.arange(mjm.nbody)[mjm.body_mocapid >= 0]
  m.mocap_bodyid = m.mocap_bodyid[mjm.body_mocapid[mjm.body_mocapid >= 0].argsort()]
  m.body_fluid_ellipsoid = np.zeros(mjm.nbody, dtype=bool)
  m.body_fluid_ellipsoid[mjm.geom_bodyid[mjm.geom_fluid.reshape(mjm.ngeom, mujoco.mjNFLUID)[:, 0] > 0]] = True
  jnt_limited_slide_hinge = mjm.jnt_limited & np.isin(mjm.jnt_type, (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE))
  m.jnt_limited_slide_hinge_adr = np.nonzero(jnt_limited_slide_hinge)[0]
  m.jnt_limited_ball_adr = np.nonzero(mjm.jnt_limited & (mjm.jnt_type == mujoco.mjtJoint.mjJNT_BALL))[0]
  m.dof_tri_row, m.dof_tri_col = np.triu_indices(mjm.nv)

  # precompute body_isdofancestor: which DOFs affect each body
  # TODO: Investigate alternative approach such as bitmap
  body_isdofancestor = np.zeros((mjm.nbody, m.nv_pad), dtype=np.int32)
  for bodyid in range(mjm.nbody):
    b = bodyid
    while b > 0 and mjm.body_dofnum[b] == 0:
      b = mjm.body_parentid[b]
    if mjm.body_dofnum[b] == 0:
      continue
    dofid = mjm.body_dofadr[b] + mjm.body_dofnum[b] - 1
    while dofid >= 0:
      body_isdofancestor[bodyid, dofid] = 1
      dofid = mjm.dof_parentid[dofid]
  m.body_isdofancestor = body_isdofancestor

  # precalculated geom pairs
  filterparent = not (mjm.opt.disableflags & types.DisableBit.FILTERPARENT)

  geom1, geom2 = np.triu_indices(mjm.ngeom, k=1)
  m.nxn_geom_pair = np.stack((geom1, geom2), axis=1)

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

  nxn_pairid_contact = -1 * np.ones(len(geom1), dtype=int)
  nxn_pairid_contact[~(mask & ~self_collision & ~parent_child_collision & ~exclude)] = -2

  # contact pairs
  def upper_tri_index(n, i, j):
    i, j = (j, i) if j < i else (i, j)
    return (i * (2 * n - i - 3)) // 2 + j - 1

  for i in range(mjm.npair):
    nxn_pairid_contact[upper_tri_index(mjm.ngeom, mjm.pair_geom1[i], mjm.pair_geom2[i])] = i

  sensor_collision_adr = np.nonzero(is_collision_sensor)[0]
  collision_sensor_adr = np.full(mjm.nsensor, -1)
  collision_sensor_adr[sensor_collision_adr] = np.arange(len(sensor_collision_adr))

  nxn_pairid_collision = -1 * np.ones(len(geom1), dtype=int)
  pairids = []
  sensor_collision_start_adr = []
  for i in range(sensor_collision_adr.size):
    sensorid = sensor_collision_adr[i]
    objtype = mjm.sensor_objtype[sensorid]
    objid = mjm.sensor_objid[sensorid]
    reftype = mjm.sensor_reftype[sensorid]
    refid = mjm.sensor_refid[sensorid]

    # get lists of geoms to collide
    if objtype == types.ObjType.BODY:
      n1 = mjm.body_geomnum[objid]
      id1 = mjm.body_geomadr[objid]
    else:
      n1 = 1
      id1 = objid
    if reftype == types.ObjType.BODY:
      n2 = mjm.body_geomnum[refid]
      id2 = mjm.body_geomadr[refid]
    else:
      n2 = 1
      id2 = refid

    # collide all pairs
    for geom1id in range(id1, id1 + n1):
      for geom2id in range(id2, id2 + n2):
        pairid = upper_tri_index(mjm.ngeom, geom1id, geom2id)

        if pairid in pairids:
          sensor_collision_start_adr.append(nxn_pairid_collision[pairid])
        else:
          npairids = len(pairids)
          nxn_pairid_collision[pairid] = npairids
          sensor_collision_start_adr.append(npairids)
          pairids.append(pairid)

  m.nsensorcollision = (nxn_pairid_collision >= 0).sum()
  m.sensor_collision_start_adr = np.array(sensor_collision_start_adr)
  nxn_include = (nxn_pairid_contact > -2) | (nxn_pairid_collision >= 0)

  if nxn_include.sum() < 250_000:
    opt.broadphase = types.BroadphaseType.NXN
  elif mjm.ngeom < 1000:
    opt.broadphase = types.BroadphaseType.SAP_TILE
  else:
    opt.broadphase = types.BroadphaseType.SAP_SEGMENTED

  m.nxn_geom_pair_filtered = m.nxn_geom_pair[nxn_include]
  m.nxn_pairid = np.hstack([nxn_pairid_contact.reshape((-1, 1)), nxn_pairid_collision.reshape((-1, 1))])
  m.nxn_pairid_filtered = m.nxn_pairid[nxn_include]

  # count contact pair types
  def geom_trid_index(i, j):
    i, j = (j, i) if j < i else (i, j)
    return (i * (2 * len(types.GeomType) - i - 1)) // 2 + j

  m.geom_pair_type_count = tuple(
    np.bincount(
      [geom_trid_index(mjm.geom_type[geom1[i]], mjm.geom_type[geom2[i]]) for i in np.arange(len(geom1)) if nxn_include[i]],
      minlength=len(types.GeomType) * (len(types.GeomType) + 1) // 2,
    )
  )

  # check for unsupported margin + multicontact / box-box CCD combinations
  use_multiccd = (mjm.opt.disableflags & types.DisableBit.MULTICCD) == 0
  nativeccd_disabled = mjm.opt.disableflags & types.DisableBit.NATIVECCD
  BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
  MESH = int(mujoco.mjtGeom.mjGEOM_MESH)

  has_boxbox = m.geom_pair_type_count[geom_trid_index(BOX, BOX)] > 0
  has_multiccd_pairs = has_boxbox or (
    use_multiccd
    and (m.geom_pair_type_count[geom_trid_index(BOX, MESH)] > 0 or m.geom_pair_type_count[geom_trid_index(MESH, MESH)] > 0)
  )

  if has_multiccd_pairs:

    def _check_margin(name, t1, t2, margin):
      if use_multiccd:
        raise NotImplementedError(
          f"{name} has non-zero margin ({margin}) with MULTICCD enabled. Set margin to 0 or disable MULTICCD."
        )
      if t1 == BOX and t2 == BOX and not nativeccd_disabled:
        raise NotImplementedError(
          f"{name} has non-zero margin ({margin}) with NATIVECCD enabled. Set margin to 0 or disable NATIVECCD."
        )

    geom_name = lambda g: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)

    for idx in np.nonzero(nxn_include & (nxn_pairid_contact == -1))[0]:
      g1, g2 = int(geom1[idx]), int(geom2[idx])
      t1, t2 = int(mjm.geom_type[g1]), int(mjm.geom_type[g2])
      m1, m2 = float(mjm.geom_margin[g1]), float(mjm.geom_margin[g2])
      if (m1 or m2) and t1 in (BOX, MESH) and t2 in (BOX, MESH):
        _check_margin(f"geom pair ({geom_name(g1)}, {geom_name(g2)})", t1, t2, (m1, m2))

    for pid in range(mjm.npair):
      g1, g2 = int(mjm.pair_geom1[pid]), int(mjm.pair_geom2[pid])
      t1, t2 = int(mjm.geom_type[g1]), int(mjm.geom_type[g2])
      pm = float(mjm.pair_margin[pid])
      if pm and t1 in (BOX, MESH) and t2 in (BOX, MESH):
        _check_margin(f"pair {pid} ({geom_name(g1)}, {geom_name(g2)})", t1, t2, pm)

  m.nmaxpolygon = np.append(mjm.mesh_polyvertnum, 0).max()
  m.nmaxmeshdeg = np.append(mjm.mesh_polymapnum, 0).max()

  # filter plugins for only geom plugins, drop the rest
  m.plugin, m.plugin_attr = [], []
  m.geom_plugin_index = np.full_like(mjm.geom_type, -1)

  for i in range(len(mjm.geom_plugin)):
    if mjm.geom_plugin[i] == -1:
      continue
    p = mjm.geom_plugin[i]
    m.geom_plugin_index[i] = len(m.plugin)
    m.plugin.append(mjm.plugin[p])
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
    if len(attr_values) > types._NPLUGINATTR:
      raise ValueError(f"Plugin has {len(attr_values)} attributes, which exceeds the maximum of {types._NPLUGINATTR}. ")
    # pad with zeros to _NPLUGINATTR
    attr_values += [0.0] * (types._NPLUGINATTR - len(attr_values))
    m.plugin_attr.append(attr_values[: types._NPLUGINATTR])

  # equality constraint addresses
  m.eq_connect_adr = np.nonzero(mjm.eq_type == types.EqType.CONNECT)[0]
  m.eq_wld_adr = np.nonzero(mjm.eq_type == types.EqType.WELD)[0]
  m.eq_jnt_adr = np.nonzero(mjm.eq_type == types.EqType.JOINT)[0]
  m.eq_ten_adr = np.nonzero(mjm.eq_type == types.EqType.TENDON)[0]
  m.eq_flex_adr = np.nonzero(mjm.eq_type == types.EqType.FLEX)[0]

  # fixed tendon
  m.tendon_jnt_adr, m.wrap_jnt_adr = [], []
  for i in range(mjm.ntendon):
    adr = mjm.tendon_adr[i]
    if mjm.wrap_type[adr] == mujoco.mjtWrap.mjWRAP_JOINT:
      tendon_num = mjm.tendon_num[i]
      for j in range(tendon_num):
        m.tendon_jnt_adr.append(i)
        m.wrap_jnt_adr.append(adr + j)

  # spatial tendon
  m.tendon_site_pair_adr, m.tendon_geom_adr = [], []
  m.ten_wrapadr_site, m.ten_wrapnum_site = [0], []
  for i, tendon_num in enumerate(mjm.tendon_num):
    adr = mjm.tendon_adr[i]
    # sites
    if (mjm.wrap_type[adr : adr + tendon_num] == mujoco.mjtWrap.mjWRAP_SITE).all():
      if i < mjm.ntendon:
        m.ten_wrapadr_site.append(m.ten_wrapadr_site[-1] + tendon_num)
      m.ten_wrapnum_site.append(tendon_num)
    else:
      if i < mjm.ntendon:
        m.ten_wrapadr_site.append(m.ten_wrapadr_site[-1])
      m.ten_wrapnum_site.append(0)

    # geoms
    for j in range(tendon_num):
      wrap_type = mjm.wrap_type[adr + j]
      if j < tendon_num - 1:
        next_wrap_type = mjm.wrap_type[adr + j + 1]
        if wrap_type == mujoco.mjtWrap.mjWRAP_SITE and next_wrap_type == mujoco.mjtWrap.mjWRAP_SITE:
          m.tendon_site_pair_adr.append(i)
      if wrap_type == mujoco.mjtWrap.mjWRAP_SPHERE or wrap_type == mujoco.mjtWrap.mjWRAP_CYLINDER:
        m.tendon_geom_adr.append(i)

  m.tendon_limited_adr = np.nonzero(mjm.tendon_limited)[0]
  m.wrap_site_adr = np.nonzero(mjm.wrap_type == mujoco.mjtWrap.mjWRAP_SITE)[0]
  m.wrap_site_pair_adr = np.setdiff1d(m.wrap_site_adr[np.nonzero(np.diff(m.wrap_site_adr) == 1)[0]], mjm.tendon_adr[1:] - 1)
  m.wrap_geom_adr = np.nonzero(np.isin(mjm.wrap_type, [mujoco.mjtWrap.mjWRAP_SPHERE, mujoco.mjtWrap.mjWRAP_CYLINDER]))[0]

  # pulley scaling
  m.wrap_pulley_scale = np.ones(mjm.nwrap, dtype=float)
  pulley_adr = np.nonzero(mjm.wrap_type == mujoco.mjtWrap.mjWRAP_PULLEY)[0]
  for tadr, tnum in zip(mjm.tendon_adr, mjm.tendon_num):
    for padr in pulley_adr:
      if tadr <= padr < tadr + tnum:
        m.wrap_pulley_scale[padr : tadr + tnum] = 1.0 / mjm.wrap_prm[padr]

  m.actuator_trntype_body_adr = np.nonzero(mjm.actuator_trntype == mujoco.mjtTrn.mjTRN_BODY)[0]

  # sensor addresses
  m.sensor_pos_adr = np.nonzero(
    (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_POS)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_JOINTLIMITPOS)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TENDONLIMITPOS)
  )[0]
  m.sensor_limitpos_adr = np.nonzero(
    (mjm.sensor_type == mujoco.mjtSensor.mjSENS_JOINTLIMITPOS) | (mjm.sensor_type == mujoco.mjtSensor.mjSENS_TENDONLIMITPOS)
  )[0]
  m.sensor_vel_adr = np.nonzero(
    (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_VEL)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_JOINTLIMITVEL)
    & (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TENDONLIMITVEL)
  )[0]
  m.sensor_limitvel_adr = np.nonzero(
    (mjm.sensor_type == mujoco.mjtSensor.mjSENS_JOINTLIMITVEL) | (mjm.sensor_type == mujoco.mjtSensor.mjSENS_TENDONLIMITVEL)
  )[0]
  m.sensor_acc_adr = np.nonzero(
    (mjm.sensor_needstage == mujoco.mjtStage.mjSTAGE_ACC)
    & (
      (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TOUCH)
      | (mjm.sensor_type != mujoco.mjtSensor.mjSENS_JOINTLIMITFRC)
      | (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TENDONLIMITFRC)
      | (mjm.sensor_type != mujoco.mjtSensor.mjSENS_TENDONACTFRC)
    )
  )[0]
  m.sensor_rangefinder_adr = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER)[0]
  m.rangefinder_sensor_adr = np.full(mjm.nsensor, -1)
  m.rangefinder_sensor_adr[m.sensor_rangefinder_adr] = np.arange(len(m.sensor_rangefinder_adr))
  m.collision_sensor_adr = np.full(mjm.nsensor, -1)
  m.collision_sensor_adr[sensor_collision_adr] = np.arange(len(sensor_collision_adr))
  m.sensor_touch_adr = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH)[0]
  limitfrc_sensors = (mujoco.mjtSensor.mjSENS_JOINTLIMITFRC, mujoco.mjtSensor.mjSENS_TENDONLIMITFRC)
  m.sensor_limitfrc_adr = np.nonzero(np.isin(mjm.sensor_type, limitfrc_sensors))[0]
  m.sensor_e_potential = (mjm.sensor_type == mujoco.mjtSensor.mjSENS_E_POTENTIAL).any()
  m.sensor_e_kinetic = (mjm.sensor_type == mujoco.mjtSensor.mjSENS_E_KINETIC).any()
  m.sensor_tendonactfrc_adr = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_TENDONACTFRC)[0]
  subtreevel_sensors = (mujoco.mjtSensor.mjSENS_SUBTREELINVEL, mujoco.mjtSensor.mjSENS_SUBTREEANGMOM)
  m.sensor_subtree_vel = np.isin(mjm.sensor_type, subtreevel_sensors).any()
  m.sensor_contact_adr = np.nonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT)[0]
  m.sensor_adr_to_contact_adr = np.clip(np.cumsum(mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT) - 1, a_min=0, a_max=None)
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
  m.sensor_rangefinder_bodyid = mjm.site_bodyid[mjm.sensor_objid[mjm.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER]]
  m.taxel_vertadr = [
    j + mjm.mesh_vertadr[mjm.sensor_objid[i]]
    for i in range(mjm.nsensor)
    if mjm.sensor_type[i] == mujoco.mjtSensor.mjSENS_TACTILE
    for j in range(mjm.mesh_vertnum[mjm.sensor_objid[i]])
  ]
  m.taxel_sensorid = [
    i
    for i in range(mjm.nsensor)
    if mjm.sensor_type[i] == mujoco.mjtSensor.mjSENS_TACTILE
    for j in range(mjm.mesh_vertnum[mjm.sensor_objid[i]])
  ]

  # M_tiles records the block diagonal structure of M
  tile_corners = [i for i in range(mjm.nv) if mjm.dof_parentid[i] == -1]
  tiles = {}
  for i in range(len(tile_corners)):
    tile_beg = tile_corners[i]
    tile_end = mjm.nv if i == len(tile_corners) - 1 else tile_corners[i + 1]
    tiles.setdefault(tile_end - tile_beg, []).append(tile_beg)
  m.M_tiles = tuple(types.TileSet(adr=wp.array(tiles[sz], dtype=int), size=sz) for sz in sorted(tiles.keys()))

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
  m.qLD_updates = tuple(wp.array(qLD_updates[i], dtype=wp.vec3i) for i in sorted(qLD_updates))

  # Build concatenated updates for fused kernel
  all_updates_flat = []
  level_offsets = [0]
  for level in sorted(qLD_updates):
    all_updates_flat.extend(qLD_updates[level])
    level_offsets.append(len(all_updates_flat))
  m.qLD_all_updates = all_updates_flat if all_updates_flat else [(0, 0, 0)]
  m.qLD_level_offsets = level_offsets

  # Indices for sparse M_fullm (used in solver). M_fullm_i/j are built by
  # walking dof_parentid for each dof, so for joint types whose internal block
  # MuJoCo stores diagonal-only in the compact (M_rownnz, M_rowadr) layout
  # (e.g. free joints), the chain-aware layout here has more entries per row
  # than the compact layout.
  m.M_fullm_i, m.M_fullm_j = [], []
  for i in range(mjm.nv):
    j = i
    while j > -1:
      m.M_fullm_i.append(i)
      m.M_fullm_j.append(j)
      j = mjm.dof_parentid[j]
  # M_elemid maps (row, col) -> madr index in the native CSR M layout
  M_elemid = np.full((mjm.nv, mjm.nv), -1, dtype=np.int32)
  for i in range(mjm.nv):
    rowadr = mjm.M_rowadr[i]
    rownnz = mjm.M_rownnz[i]
    for k in range(rownnz):
      madr = rowadr + k
      col = int(mjm.M_colind[madr])
      M_elemid[i, col] = madr
  m.M_elemid = M_elemid

  upper_j, upper_i = np.triu_indices(mjm.nv)
  upper_elemid = M_elemid[upper_i, upper_j]
  valid_mask = upper_elemid != -1
  m.M_fullm_upper_i = upper_j[valid_mask].tolist()
  m.M_fullm_upper_j = upper_i[valid_mask].tolist()
  m.M_fullm_upper_elemid = upper_elemid[valid_mask].tolist()

  # indices for sparse qD_fullm (used in RNE derivatives)
  # D-structure is the full square sparsity pattern (both upper and lower triangle)
  m.qD_fullm_i, m.qD_fullm_j = [], []
  for i in range(mjm.nv):
    rowadr = mjm.D_rowadr[i]
    rownnz = mjm.D_rownnz[i]
    for k in range(rownnz):
      m.qD_fullm_i.append(i)
      m.qD_fullm_j.append(int(mjm.D_colind[rowadr + k]))
  m.nD = mjm.nD

  # Gather-based sparse mul_m: for each row, all (col, madr) including diagonal
  row_elements = [[] for _ in range(mjm.nv)]

  for i in range(mjm.nv):
    rowadr = mjm.M_rowadr[i]
    rownnz = mjm.M_rownnz[i]
    for k in range(rownnz):
      madr = rowadr + k
      col = int(mjm.M_colind[madr])
      row_elements[i].append((col, madr))  # row i gathers M[i,col] * vec[col]
      if i != col:
        row_elements[col].append((i, madr))  # row col gathers M[i,col] * vec[i]

  # Flatten into CSR-like arrays
  m.M_mulm_rowadr = [0]
  m.M_mulm_col = []
  m.M_mulm_madr = []
  for i in range(mjm.nv):
    for col, madr in row_elements[i]:
      m.M_mulm_col.append(col)
      m.M_mulm_madr.append(madr)
    m.M_mulm_rowadr.append(len(m.M_mulm_col))

  m.flexedge_J_rownnz = mjm.flexedge_J_rownnz
  m.flexedge_J_rowadr = mjm.flexedge_J_rowadr
  m.flexedge_J_colind = mjm.flexedge_J_colind.reshape(-1)

  # flex_bendingadr backward compat: flatten old (nflexedge, 17) to 1D
  if not check_version("mujoco>=3.8.1.dev909088123"):
    m.flex_bendingadr = (
      np.array([mjm.flex_edgeadr[i] * 17 for i in range(mjm.nflex)], dtype=int) if mjm.nflex else np.zeros(0, dtype=int)
    )
    m.flex_bending = mjm.flex_bending.ravel()
    m.nflexbending = len(m.flex_bending)

  # place m on device
  sizes = dict({"*": 1}, **{f.name: getattr(m, f.name) for f in dataclasses.fields(types.Model) if f.type is int})
  for f in dataclasses.fields(types.Model):
    if _is_array_spec(f.type):
      setattr(m, f.name, _create_array(getattr(m, f.name), f.type, sizes))

  return m
