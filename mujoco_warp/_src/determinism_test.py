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

"""Tests for deterministic execution."""

import ast
import collections
import contextlib
import dataclasses
import importlib
import inspect
import pkgutil
import textwrap
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import bvh
from mujoco_warp._src import collision_driver
from mujoco_warp._src import constraint
from mujoco_warp._src import render_util
from mujoco_warp._src import set_const
from mujoco_warp._src import util_pkg
from mujoco_warp._src import warp_util
from mujoco_warp._src.io import override_model
from mujoco_warp._src.types import DeterminismType

# steps per run, and runs compared bitwise with the first deterministic run
_NSTEP = 3
_NREPEAT = 2
_RUN_TO_RUN = wp.DeterministicMode.RUN_TO_RUN

# fields that are invariant to contact and constraint row order
_PHYSICS_FIELDS = (
  "qpos qvel act qacc qacc_smooth qfrc_bias qfrc_spring qfrc_damper qfrc_gravcomp qfrc_fluid qfrc_adhesion qfrc_passive "
  "qfrc_actuator qfrc_smooth qfrc_constraint qfrc_inverse actuator_length actuator_velocity actuator_force ten_length "
  "ten_velocity subtree_com subtree_linvel subtree_angmom cinert crb cacc cfrc_int cfrc_ext energy flexvert_xpos"
).split()


def _det_factories():
  """Returns the kernel factories with a `deterministic` argument, keyed by `module.name`."""
  factories = {}
  for info in pkgutil.iter_modules(importlib.import_module("mujoco_warp._src").__path__):
    if info.name.endswith("_test"):
      continue
    module = importlib.import_module(f"mujoco_warp._src.{info.name}")
    for name, obj in vars(module).items():
      if inspect.isfunction(obj) and obj.__module__ == module.__name__ and "__wrapped__" in vars(obj):
        if "deterministic" in inspect.signature(obj).parameters:
          factories[f"{info.name}.{name}"] = obj
  return factories


def _records_only_args(factory):
  """Returns the factory arguments that are only used to size `deterministic_max_records`."""
  func = ast.parse(textwrap.dedent(inspect.getsource(factory.__wrapped__))).body[0]
  records = collections.Counter()
  for node in ast.walk(func):
    if isinstance(node, ast.Assign):
      for target in node.targets:
        if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
          if target.slice.value == "deterministic_max_records":
            records.update(n.id for n in ast.walk(node.value) if isinstance(n, ast.Name))
  uses = collections.Counter(n.id for n in ast.walk(func) if isinstance(n, ast.Name))
  return [arg.arg for arg in func.args.args if records[arg.arg] and records[arg.arg] == uses[arg.arg]]


_DET_FACTORIES = _det_factories()
_RECORDS_ONLY_ARGS = {name: _records_only_args(factory) for name, factory in _DET_FACTORIES.items()}


def _flatten(obj, prefix):
  """Returns the warp arrays reachable from obj, keyed by path."""
  if isinstance(obj, wp.array):
    return {} if obj.dtype is wp.Texture2D else {prefix: obj}
  if dataclasses.is_dataclass(obj) or isinstance(obj, wp.Mesh):
    items = vars(obj).items()
  elif isinstance(obj, dict):
    items = obj.items()
  elif isinstance(obj, (list, tuple)):
    items = enumerate(obj)
  else:
    return {}
  arrays = {}
  for name, value in items:
    arrays.update(_flatten(value, f"{prefix}.{name}"))
  return arrays


def _snapshot(arrays):
  return {name: arr.numpy().copy() for name, arr in arrays.items()}


def _restore(arrays, snapshot):
  for name, arr in arrays.items():
    if arr.size:
      arr.assign(snapshot[name])


def _bitwise_diff(ref, out):
  """Returns the arrays whose bits differ, with the number of differing elements."""
  diff = []
  for name, a in ref.items():
    b = out[name]
    if a.tobytes() != b.tobytes():
      ne = np.frombuffer(a.tobytes(), np.uint8) != np.frombuffer(b.tobytes(), np.uint8)
      diff.append(f"{name}: {np.count_nonzero(ne.reshape(-1, a.itemsize).any(axis=1))} of {a.size}")
  return diff


def _bits(a):
  """Returns the sorted bit patterns of a, which do not depend on element order."""
  return np.sort(a.reshape(-1).view(f"u{a.itemsize}"))


def _perturb(d):
  """Makes world 1 different from world 0."""
  qvel = d.qvel.numpy()
  qvel[1] += 0.1 * np.linspace(-1.0, 1.0, qvel.shape[1])
  d.qvel.assign(qvel)
  ctrl = d.ctrl.numpy()
  ctrl[1] = 0.5 + np.abs(ctrl[1])
  d.ctrl.assign(ctrl)


@contextlib.contextmanager
def _record_launches():
  """Records (factory, deterministic mode, nonzero dim) for each kernel launch."""
  launches = []

  def wrap(launch):
    def wrapper(*args, **kwargs):
      kernel = kwargs["kernel"] if "kernel" in kwargs else args[0]
      dim = kwargs["dim"] if "dim" in kwargs else args[1]
      module = kernel.func.__module__.rsplit(".", 1)[-1]
      factory = kernel.func.__qualname__.split(".<locals>")[0]
      mode = kernel.module.options.get("deterministic", wp.DeterministicMode.NOT_GUARANTEED)
      launches.append((f"{module}.{factory}", mode, bool(np.prod(dim))))
      return launch(*args, **kwargs)

    return wrapper

  with mock.patch.object(wp, "launch", wrap(wp.launch)), mock.patch.object(wp, "launch_tiled", wrap(wp.launch_tiled)):
    yield launches


def _spy(module, name, tag, events, d):
  """Patches module.name to append (tag, called with d, Data snapshot) to events after each call."""
  fn = vars(module)[name]

  def wrapper(*args, **kwargs):
    out = fn(*args, **kwargs)
    events.append((tag, args[1] is d, _snapshot(_flatten(d, "d"))))
    return out

  return mock.patch.object(module, name, wrapper)


class _Replay:
  """Replays the contact and constraint rows of the first run in later runs.

  CONTACTS and CONSTRAINT are not implemented yet, so collision and make_constraint emit rows in a
  racy order. Later runs check that the inputs of these stages are bitwise equal to the first run
  and that their float outputs are bitwise equal up to order, then restore the first run's outputs.
  """

  def __init__(self, test, d):
    self._test = test
    self._d = d
    self._calls = []
    self._recording = True
    self._index = 0

  def _wrap(self, stage, fn):
    def wrapper(*args, **kwargs):
      pre = _snapshot(_flatten(self._d, "d"))
      fn(*args, **kwargs)
      post = _snapshot(_flatten(self._d, "d"))
      if self._recording:
        self._calls.append((stage, pre, post))
        return
      call = f"{stage} call {self._index}"
      self._test.assertLess(self._index, len(self._calls), call)
      ref_stage, ref_pre, ref_post = self._calls[self._index]
      self._index += 1
      self._test.assertEqual(stage, ref_stage, call)
      self._test.assertEqual(_bitwise_diff(ref_pre, pre), [], f"inputs of {call}")
      floats = [name for name, value in post.items() if value.dtype.kind == "f"]
      changed = [name for name in floats if _bits(post[name]).tobytes() != _bits(ref_post[name]).tobytes()]
      self._test.assertEqual(changed, [], f"float outputs of {call}")
      _restore(_flatten(self._d, "d"), ref_post)

    return wrapper

  @contextlib.contextmanager
  def patch(self):
    self._recording, self._index = not self._calls, 0
    with (
      mock.patch.object(collision_driver, "collision", self._wrap("collision", collision_driver.collision)),
      mock.patch.object(constraint, "make_constraint", self._wrap("make_constraint", constraint.make_constraint)),
    ):
      yield
    if not self._recording:
      self._test.assertEqual(self._index, len(self._calls), "missing stage calls")


_TREE_CASES = (
  dict(
    testcase_name="euler",
    overrides={},
    covers=(
      "forward._qfrc_actuator",
      "forward._tendon_actuator_force",
      "passive._gravity_force",
      "passive._spring_damper_tendon_passive",
      "sensor._energy_pos_gravity",
      "sensor._energy_pos_passive_joint",
      "sensor._energy_pos_passive_tendon",
      "sensor._tendon_actuator_force",
      "set_const._accumulate_subtreemass",
      "smooth._angular_momentum",
      "smooth._cfrc_backward",
      "smooth._cfrc_ext_equality",
      "smooth._cfrc_ext_spatial_tendon",
      "smooth._cfrc_ext_tendon_actuator",
      "smooth._cfrc_ext_tendon_constraint",
      "smooth._crb_accumulate",
      "smooth._joint_tendon",
      "smooth._linear_momentum",
      "smooth._spatial_geom_tendon",
      "smooth._spatial_site_tendon",
      "smooth._subtree_com_acc",
      "smooth._tendon_armature",
      "smooth._tendon_bias_coef",
      "smooth._tendon_bias_qfrc",
      "smooth._transmission",
      "solver._JTDACJ_sparse",
      "solver._linesearch_jv_fused_kernel",
      "solver._solve_init_jaref_kernel",
      "solver._update_constraint_efc",
      "solver._update_constraint_init_qfrc_constraint_sparse",
      "solver._update_gradient_grad",
      "solver._update_gradient_h_incremental_sparse",
    ),
  ),
  dict(
    testcase_name="implicit",
    overrides={"opt.integrator": "IMPLICIT"},
    covers=(
      "derivative._qderiv_actuator_passive_actuation_sparse",
      "derivative._qderiv_box_fluid",
      "derivative._qderiv_ellipsoid_fluid",
      "derivative.deriv_rne_body2jnt_sparse",
      "derivative.deriv_rne_cfrcbody_backward",
    ),
  ),
  dict(testcase_name="implicitfast", overrides={"opt.integrator": "IMPLICITFAST"}, covers=()),
  dict(testcase_name="rk4", overrides={"opt.integrator": "RK4"}, covers=()),
  dict(
    testcase_name="discrete",
    overrides={"opt.integrator": "DISCRETE"},
    covers=(
      "derivative._eff_actuator_actuation",
      "derivative._eff_add_actuator_qH_diag",
      "derivative._eff_add_tendon_qH_diag",
      "derivative._eff_mul_add_actuator",
      "derivative._eff_mul_add_tendon",
      "derivative._eff_tendon_shift",
      "solver._add_actuator_metric_dense",
      "solver._add_tendon_metric_dense",
    ),
  ),
  dict(testcase_name="discrete_cg", overrides={"opt.integrator": "DISCRETE", "opt.solver": "CG"}, covers=()),
  dict(
    testcase_name="dense_blocks",
    overrides={"opt.jacobian": "DENSE"},
    covers=("smooth._qLD_acc", "smooth._solve_LD_sparse_level"),
    sparse_blocks=True,
  ),
)

_FLEX_CASES = (
  dict(
    testcase_name="euler",
    overrides={},
    covers=(
      "bvh.accumulate_flex_vertex_normals",
      "passive._flex_bending",
      "passive._flex_elasticity",
      "passive._flex_passive_interp",
      "passive._flex_rod_bending",
      "passive._spring_damper_flexedge_passive",
      "sensor._energy_pos_passive_flex",
    ),
  ),
  dict(
    testcase_name="discrete",
    overrides={"opt.integrator": "DISCRETE"},
    covers=(
      "derivative._eff_flex_bend_stiff",
      "derivative._eff_flex_interp_mul",
      "derivative._eff_flex_interp_stiff",
      "derivative._eff_flex_rod_stiff",
      "derivative._eff_flex_stretch_stiff_edge",
      "derivative._eff_flex_stretch_stiff_vert",
      "passive._flex_bend_mul",
      "passive._flex_hessian_diag",
      "passive._flex_hessian_elem",
      "passive._flex_hessian_mul_edge",
      "passive._flex_rod_mul",
      "passive._flex_scatter_kernel",
      "solver._add_efmK_metric_dense",
    ),
  ),
)

_FLEX_SHELL_CASES = (dict(testcase_name="euler", overrides={}, covers=("passive._flex_passive_bend_interp",)),)

_FLEX_CONTACT_CASES = (
  dict(testcase_name="sparse", overrides={"opt.jacobian": "SPARSE"}, covers=()),
  dict(testcase_name="dense", overrides={"opt.jacobian": "DENSE"}, covers=("constraint._efc_contact_jac_dense_flex",)),
)

_FLEX_PASSIVE_CASES = (
  dict(
    testcase_name="cg",
    overrides={"opt.solver": "CG"},
    covers=(
      "derivative._eff_contact_shift",
      "derivative._eff_fold_actuator",
      "derivative._eff_fold_contact_rank1",
      "derivative._eff_fold_efc_sparse",
      "derivative._eff_fold_tendon",
      "derivative._eff_mul_add_contact",
      "passive._eff_contact_force",
    ),
  ),
  dict(testcase_name="newton", overrides={}, covers=("solver._add_flexcon_metric_dense",)),
)

_CONTACT_CASES = (
  dict(
    testcase_name="sparse",
    overrides={"opt.jacobian": "SPARSE"},
    covers=(
      "constraint._efc_contact_jac_sparse",
      "passive._qfrc_adhesion",
      "sensor._contact_match_builder",
      "sensor._sensor_touch",
      "smooth._cfrc_ext_contact",
      "smooth._transmission_body_moment",
    ),
  ),
  dict(testcase_name="dense", overrides={"opt.jacobian": "DENSE"}, covers=("constraint._efc_contact_jac_dense",)),
  dict(
    testcase_name="elliptic_dense",
    overrides={"opt.jacobian": "DENSE", "opt.cone": "ELLIPTIC"},
    covers=("solver._update_gradient_JTCJ_dense",),
  ),
  dict(testcase_name="elliptic_sparse", overrides={"opt.jacobian": "SPARSE", "opt.cone": "ELLIPTIC"}, covers=()),
  dict(testcase_name="discrete", overrides={"opt.integrator": "DISCRETE"}, covers=()),
)

_SLEEP_CASES = (dict(testcase_name="euler", overrides={}, covers=("sleep._update_sleep_bodies", "sleep._update_sleep_dofs")),)


class DeterminismTest(parameterized.TestCase):
  def _assert_default_path_keys(self):
    """Records-only factory arguments are 0 when deterministic is False, avoiding recompiles."""
    for key in warp_util._KERNEL_CACHE:
      name = f"{key[-2].rsplit('.', 1)[-1]}.{key[-1]}"
      if name not in _DET_FACTORIES:
        continue
      args = dict(zip(inspect.signature(_DET_FACTORIES[name]).parameters, key[:-2]))
      if not args.get("deterministic"):
        for arg in _RECORDS_ONLY_ARGS[name]:
          self.assertEqual(args.get(arg, 0), 0, f"{name} is called with {arg} != 0 on the default path")

  def _check_scene(self, mjm, m, d, covers, replay=False, extra=None, rc=None, perturb=True, tol=1e-3):
    """Compares ATOMICS with NONE and checks that ATOMICS is bitwise reproducible.

    Returns the arrays after the first ATOMICS run.
    """
    if perturb:
      _perturb(d)

    def arrays():
      return {**_flatten(m, "m"), **_flatten(d, "d"), **_flatten(rc, "rc")}

    init = _snapshot(arrays())
    replayer = _Replay(self, d) if replay else None

    def run(mode):
      _restore(arrays(), init)
      m.opt.deterministic = mode
      with replayer.patch() if replayer and mode else contextlib.nullcontext():
        for _ in range(_NSTEP):
          mjw.step(m, d)
        if extra:
          extra(m, d)
      return _snapshot(arrays())

    with _record_launches() as launches:
      none = run(DeterminismType.NONE)
    self.assertEqual([name for name, mode, _ in launches if mode == _RUN_TO_RUN], [])
    if "smooth._solve_LD_sparse_level" in covers:
      self.assertIn("smooth._solve_LD_sparse_fused", {name for name, _, nonzero in launches if nonzero})
    self._assert_default_path_keys()

    with _record_launches() as launches:
      ref = run(DeterminismType.ATOMICS)
    for name, mode, _ in launches:
      self.assertEqual(mode == _RUN_TO_RUN, name in _DET_FACTORIES, name)
    self.assertNotIn("smooth._solve_LD_sparse_fused", {name for name, _, _ in launches})
    launched = {name for name, mode, nonzero in launches if mode == _RUN_TO_RUN and nonzero}
    self.assertEqual(set(covers) - launched, set(), "factories not exercised")

    # NONE and ATOMICS only differ in floating-point summation order
    names = [f"d.{field}" for field in _PHYSICS_FIELDS]
    names += [name for name, value in none.items() if name.startswith(("m.", "rc.")) and value.dtype.kind == "f"]
    for name in names:
      np.testing.assert_allclose(ref[name], none[name], rtol=tol, atol=tol, err_msg=name)
    sensor = np.ones(mjm.nsensordata, dtype=bool)
    for i in np.flatnonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_CONTACT):
      sensor[mjm.sensor_adr[i] : mjm.sensor_adr[i] + mjm.sensor_dim[i]] = False
    np.testing.assert_allclose(ref["d.sensordata"][:, sensor], none["d.sensordata"][:, sensor], rtol=tol, atol=tol)
    self.assertFalse(np.array_equal(ref["d.qacc"][0], ref["d.qacc"][1]), "worlds are not distinct")

    for _ in range(_NREPEAT):
      self.assertEqual(_bitwise_diff(ref, run(DeterminismType.ATOMICS)), [])

    return ref

  @parameterized.named_parameters(*_TREE_CASES)
  def test_tree(self, overrides, covers, sparse_blocks=False):
    """Tree with tendons, actuators, sensors, fluid, gravity compensation, equality and limits."""
    blocks = mock.patch.multiple("mujoco_warp._src.types", M_BLOCK_SCALAR_MAX=0, M_BLOCK_DENSE_MAX=0)
    with blocks if sparse_blocks else contextlib.nullcontext():
      mjm, _, m, d = test_data.fixture(
        xml="""
        <mujoco>
          <option density="1.2" viscosity=".002" wind=".3 0 0" jacobian="sparse">
            <flag contact="disable" energy="enable"/>
          </option>
          <worldbody>
            <body name="torso" pos="0 0 1" gravcomp=".5">
              <freejoint name="root"/>
              <geom type="box" size=".1 .08 .06" fluidshape="ellipsoid"/>
              <site name="torso" pos="0 0 .06"/>
              <body name="arm1" pos=".1 0 0">
                <joint name="hinge1" axis="0 1 0" stiffness="2" damping=".2" frictionloss=".01"/>
                <geom type="capsule" size=".02" fromto="0 0 0 .2 0 0"/>
                <site name="arm1" pos=".1 0 .03"/>
                <body name="arm2" pos=".2 0 0">
                  <joint name="ball2" type="ball" stiffness="1" damping=".1"/>
                  <geom type="capsule" size=".02" fromto="0 0 0 .2 0 0"/>
                  <geom name="cylinder" type="cylinder" size=".04 .02" pos=".1 0 0" zaxis="0 1 0"/>
                  <site name="arm2" pos=".2 0 .03"/>
                  <body name="arm3" pos=".2 0 0">
                    <joint name="hinge3" axis="0 0 1" damping=".1" range=".1 1"/>
                    <joint name="slide3" type="slide" axis="1 0 0" stiffness="5" damping=".5"/>
                    <geom size=".03"/>
                    <site name="arm3" pos=".05 0 0"/>
                  </body>
                </body>
              </body>
              <body name="leg1" pos="0 0 -.06">
                <joint name="hinge4" axis="1 0 0" stiffness="1" springref=".2"/>
                <geom type="capsule" size=".02" fromto="0 0 0 0 0 -.2"/>
                <geom name="sphere" size=".04" pos="0 0 -.1"/>
                <site name="slider" pos="0 0 -.15" zaxis="0 1 0"/>
                <body name="leg2" pos="0 0 -.2">
                  <joint name="hinge5" axis="1 0 0" damping=".1"/>
                  <geom type="capsule" size=".02" fromto="0 0 0 0 0 -.2" fluidshape="ellipsoid"/>
                  <site name="leg2" pos="0 0 -.2"/>
                  <site name="crank" pos="0 .02 -.1"/>
                </body>
              </body>
            </body>
          </worldbody>
          <tendon>
            <fixed name="fixed" armature=".05" stiffness="2" damping=".2" actuatorfrcrange="-.5 .5" range=".05 .5">
              <joint joint="hinge1" coef="1"/>
              <joint joint="hinge3" coef="-.5"/>
              <joint joint="hinge4" coef=".7"/>
            </fixed>
            <spatial name="sites" stiffness="1" damping=".1" armature=".02" frictionloss=".01">
              <site site="torso"/>
              <site site="arm1"/>
              <site site="arm2"/>
              <site site="arm3"/>
            </spatial>
            <spatial name="wrap">
              <site site="torso"/>
              <geom geom="sphere"/>
              <site site="leg2"/>
              <pulley divisor="2"/>
              <site site="arm1"/>
              <geom geom="cylinder"/>
              <site site="arm3"/>
            </spatial>
          </tendon>
          <equality>
            <connect body1="leg2" anchor="0 0 -.2" solref=".05 1"/>
          </equality>
          <actuator>
            <motor joint="root" gear="1 0 0 0 0 1"/>
            <general jointinparent="root" gear="0 1 0 1 0 0"/>
            <motor joint="hinge1"/>
            <position joint="hinge1" kp="10" kv="1"/>
            <motor joint="ball2" gear="1 1 1"/>
            <velocity joint="slide3" kv="2"/>
            <motor tendon="fixed"/>
            <position tendon="sites" kp="5"/>
            <general tendon="wrap" gainprm="2"/>
            <general site="arm3" refsite="torso" gear="1 0 0 0 0 1"/>
            <general site="leg2" gear="0 0 1 0 0 0"/>
            <general cranksite="crank" slidersite="slider" cranklength=".1"/>
          </actuator>
          <sensor>
            <subtreelinvel body="torso"/>
            <subtreeangmom body="torso"/>
            <subtreeangmom body="arm1"/>
            <tendonactuatorfrc tendon="fixed"/>
            <accelerometer site="arm3"/>
            <force site="arm2"/>
            <torque site="leg2"/>
          </sensor>
        </mujoco>
        """,
        overrides=overrides,
        nworld=2,
        qvel_noise=0.1,
        ctrl_noise=0.5,
      )

    def extra(m, d):
      set_const.set_const(m, d)
      mjw.inverse(m, d)

    ref = self._check_scene(mjm, m, d, covers, extra=extra)

    # actuator moment rows are in actuator order with ascending columns
    rownnz, rowadr, colind = ref["d.moment_rownnz"], ref["d.moment_rowadr"], ref["d.moment_colind"]
    self.assertGreater(len(np.unique(rownnz)), 1)
    np.testing.assert_array_equal(rowadr, np.cumsum(rownnz, axis=1) - rownnz)
    for worldid, actid in np.ndindex(rownnz.shape):
      adr = rowadr[worldid, actid]
      self.assertTrue(np.all(np.diff(colind[worldid, adr : adr + rownnz[worldid, actid]]) > 0))

  @parameterized.named_parameters(*_FLEX_CASES)
  def test_flex(self, overrides, covers):
    """1D, 2D, 3D and interpolated flexes with elasticity, energy and render normals."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag contact="disable" energy="enable"/>
        </option>
        <worldbody>
          <flexcomp name="rope" type="grid" count="5 1 1" spacing=".1 .1 .1" dim="1" pos="0 1 1">
            <edge stiffness="10" damping=".1"/>
            <elasticity young="1e4" damping="1e-2"/>
          </flexcomp>
          <flexcomp name="cloth" type="grid" count="4 4 1" spacing=".1 .1 .1" pos="0 0 1">
            <edge damping=".1"/>
            <elasticity young="3e3" poisson=".3" thickness="1e-2" damping="1e-3" elastic2d="both"/>
          </flexcomp>
          <flexcomp name="solid" type="grid" count="3 3 3" spacing=".1 .1 .1" dim="3" pos="1 0 1">
            <elasticity young="1e4" poisson=".3" damping="1e-3"/>
          </flexcomp>
          <flexcomp name="interp" type="grid" count="3 3 3" spacing=".1 .1 .1" dim="3" dof="trilinear" pos="2 0 1">
            <contact selfcollide="none"/>
            <elasticity young="1e4" poisson=".3" damping="1e-3"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      overrides=overrides,
      nworld=2,
      qvel_noise=0.1,
    )
    # TODO(team): remove version check once mujoco >= 3.16.1 is required.
    if not util_pkg.check_version("mujoco>=3.16.1"):
      covers = tuple(
        c for c in covers if c not in ("derivative._eff_flex_rod_stiff", "passive._flex_rod_bending", "passive._flex_rod_mul")
      )
    with _record_launches() as launches:
      rc = render_util.create_render_context(mjm, nworld=2, cam_res=(8, 8), deterministic=True)
    if not overrides:
      rc1 = render_util.create_render_context(mjm, nworld=2, cam_res=(8, 8), deterministic=True)
      self.assertIn(("bvh.accumulate_flex_vertex_normals", _RUN_TO_RUN, True), launches)
      self.assertEqual(
        _bitwise_diff(_snapshot(_flatten(rc.flex_group_root, "root")), _snapshot(_flatten(rc1.flex_group_root, "root"))), []
      )
    self._check_scene(mjm, m, d, covers, extra=lambda m, d: bvh.refit_bvh(m, d, rc), rc=rc)

  @parameterized.named_parameters(*_FLEX_SHELL_CASES)
  def test_flex_shell(self, overrides, covers):
    """Interpolated flex shell with bending."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag contact="disable"/>
        </option>
        <worldbody>
          <flexcomp name="shell" type="grid" count="3 3 3" spacing=".1 .1 .1" dim="3" dof="trilinear" pos="0 0 1">
            <contact selfcollide="none"/>
            <elasticity young="1e4" poisson=".3" thickness=".03" elastic2d="bend"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      overrides=overrides,
      nworld=2,
      qvel_noise=0.1,
    )
    self._check_scene(mjm, m, d, covers)

  def test_flex_attached(self):
    """Interpolated flex attached to a hinged body with the discrete CG solver."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" solver="CG">
          <flag contact="disable"/>
        </option>
        <worldbody>
          <body name="carrier" pos="0 0 1">
            <joint axis="0 1 0"/>
            <geom size=".1"/>
            <flexcomp name="interp" type="grid" count="3 3 3" spacing=".1 .1 .1" dim="3" dof="trilinear">
              <contact selfcollide="none"/>
              <elasticity young="1e4" poisson=".3" damping="1e-3"/>
            </flexcomp>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
      qvel_noise=0.1,
    )
    self._check_scene(mjm, m, d, ("derivative._eff_flex_interp_mul",))

  @parameterized.named_parameters(*_FLEX_CONTACT_CASES)
  def test_flex_contact(self, overrides, covers):
    """Flex contacts with a plane and a free body."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="2 2 .1"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1" pos="0 0 .01">
            <edge damping=".1"/>
            <elasticity young="3e3" poisson=".3" thickness="1e-2" damping="1e-3" elastic2d="both"/>
          </flexcomp>
          <flexcomp name="interp" type="grid" count="3 3 3" spacing=".1 .1 .1" dim="3" dof="trilinear" pos="1 0 .11">
            <contact selfcollide="none"/>
            <elasticity young="1e4" poisson=".3" damping="1e-3"/>
          </flexcomp>
          <body pos="0 0 .3">
            <freejoint/>
            <geom type="box" size=".05 .05 .05"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides=overrides,
      nworld=2,
      qvel_noise=0.1,
    )
    self._check_scene(mjm, m, d, covers, replay=True)

  @parameterized.named_parameters(*_FLEX_PASSIVE_CASES)
  def test_flex_passive(self, overrides, covers):
    """Passive flex contacts with the discrete integrator, a limit, a tendon and actuators."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" jacobian="sparse"/>
        <worldbody>
          <geom size=".2"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1" pos="0 0 .19" mass=".09">
            <contact passive="true"/>
            <elasticity young="1e3" poisson=".3" thickness="1e-3" elastic2d="both"/>
          </flexcomp>
          <body pos="1 0 1">
            <joint name="hinge0" axis="0 1 0" damping=".1"/>
            <geom type="capsule" size=".02" fromto="0 0 0 .2 0 0"/>
            <body pos=".2 0 0">
              <joint name="hinge1" axis="0 1 0" damping=".1" range=".1 1"/>
              <geom type="capsule" size=".02" fromto="0 0 0 .2 0 0"/>
            </body>
          </body>
        </worldbody>
        <tendon>
          <fixed name="fixed" stiffness="1" damping=".1">
            <joint joint="hinge0" coef="1"/>
            <joint joint="hinge1" coef="-1"/>
          </fixed>
        </tendon>
        <actuator>
          <position joint="hinge0" kp="10" kv="1"/>
          <velocity tendon="fixed" kv="1"/>
        </actuator>
      </mujoco>
      """,
      overrides=overrides,
      nworld=2,
      qvel_noise=0.1,
      ctrl_noise=0.5,
    )
    # stiff passive contacts amplify differences in summation order
    self._check_scene(mjm, m, d, covers, replay=True, tol=1e-2)

  @parameterized.named_parameters(*_CONTACT_CASES)
  def test_contact(self, overrides, covers):
    """Contacts with condim 3, 4 and 6, adhesion, and touch, contact and force sensors."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="2 2 .1" adhesion="1"/>
          <body name="b0" pos="0 0 .05">
            <freejoint/>
            <geom type="box" size=".05 .05 .05"/>
            <site name="touch" type="box" size=".06 .06 .06"/>
          </body>
          <body name="b1" pos=".2 0 .05">
            <freejoint/>
            <geom size=".05" condim="6"/>
          </body>
          <body name="b2" pos="0 .2 .05">
            <freejoint/>
            <geom type="capsule" size=".03 .05" euler="0 90 0" condim="4"/>
            <site name="force" pos="0 0 .03"/>
          </body>
        </worldbody>
        <actuator>
          <adhesion body="b0" ctrlrange="0 1" gain="5"/>
        </actuator>
        <sensor>
          <touch site="touch"/>
          <contact body1="b0" reduce="maxforce" num="2"/>
          <contact body1="b1"/>
          <force site="force"/>
        </sensor>
      </mujoco>
      """,
      overrides=overrides,
      nworld=2,
      qvel_noise=0.1,
      ctrl_noise=0.5,
    )
    self._check_scene(mjm, m, d, covers, replay=True)

  @parameterized.named_parameters(*_SLEEP_CASES)
  def test_sleep(self, overrides, covers):
    """Sleeping trees, with awake bodies and dofs in ascending order."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sleep_tolerance=".1">
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body pos="0 0 .05">
            <freejoint/>
            <geom type="box" size=".05 .05 .05"/>
          </body>
          <body pos=".5 0 .05">
            <freejoint/>
            <geom type="box" size=".05 .05 .05"/>
          </body>
          <body pos="1 0 .05">
            <freejoint/>
            <geom type="box" size=".05 .05 .05"/>
          </body>
          <body pos="1.5 0 .3">
            <freejoint/>
            <geom size=".05"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides=overrides,
      nworld=2,
      qvel_noise=0.1,
    )
    # all trees are awake and moving, except tree 1 in world 1 which is asleep in its own island
    tree_asleep = np.full((2, mjm.ntree), -(1 + mujoco.mjMINAWAKE))
    tree_asleep[1, 1] = 1
    d.tree_asleep.assign(tree_asleep)
    qvel = np.tile(0.2 * np.linspace(-1.0, 1.0, mjm.nv), (2, 1))
    qvel[1, 6:12] = 0.0
    d.qvel.assign(qvel)
    for arr in (d.nbody_awake, d.nv_awake, d.body_awake_ind, d.dof_awake_ind):
      arr.fill_(-1)

    ref = self._check_scene(mjm, m, d, covers, replay=True, perturb=False)

    np.testing.assert_array_equal(ref["d.nbody_awake"], [5, 4])
    np.testing.assert_array_equal(ref["d.nv_awake"], [24, 18])
    np.testing.assert_array_equal(ref["d.body_awake_ind"][0, :5], range(5))
    np.testing.assert_array_equal(ref["d.body_awake_ind"][1, :4], [0, 1, 3, 4])
    np.testing.assert_array_equal(ref["d.dof_awake_ind"][0, :24], range(24))
    np.testing.assert_array_equal(ref["d.dof_awake_ind"][1, :18], [*range(6), *range(12, 24)])

  def test_flags(self):
    """DeterminismType has one bit per feature and ALL combines them."""
    self.assertIs(mjw.DeterminismType, DeterminismType)
    self.assertEqual(int(DeterminismType.NONE), 0)
    single = {name: int(flag) for name, flag in DeterminismType.__members__.items() if flag and not (flag & (flag - 1))}
    self.assertEqual(single, {"CONTACTS": 1, "CONSTRAINT": 2, "ATOMICS": 4})
    self.assertEqual(int(DeterminismType.ALL), sum(single.values()))

  def test_option(self):
    """Option.deterministic stores an int bitmask, with bool and None mapped to ALL and NONE."""
    mjm, _, m, _ = test_data.fixture(xml="<mujoco/>")
    self.assertEqual(m.opt.deterministic, DeterminismType.NONE)
    self.assertEqual(mjw.put_model(mjm).opt.deterministic, DeterminismType.NONE)

    for value, expected in (
      (True, DeterminismType.ALL),
      (False, DeterminismType.NONE),
      (None, DeterminismType.NONE),
      (np.bool_(True), DeterminismType.ALL),
      (np.bool_(False), DeterminismType.NONE),
      (DeterminismType.ATOMICS, DeterminismType.ATOMICS),
      (DeterminismType.CONTACTS | DeterminismType.ATOMICS, 5),
      (~DeterminismType.CONTACTS, 6),
      (np.int32(4), DeterminismType.ATOMICS),
      (7, DeterminismType.ALL),
      (-1, DeterminismType.ALL),
    ):
      m.opt.deterministic = value
      self.assertEqual(m.opt.deterministic, expected, value)
      self.assertIs(type(m.opt.deterministic), int)

    m.opt.deterministic &= ~DeterminismType.ATOMICS
    self.assertEqual(m.opt.deterministic, 3)
    m.opt.deterministic |= DeterminismType.ATOMICS
    self.assertEqual(m.opt.deterministic, DeterminismType.ALL)

  @parameterized.parameters(8, 1 << 40, 1.5, 4.0, "ATOMICS")
  def test_option_invalid(self, value):
    """Option.deterministic rejects values that are not DeterminismType bitmasks."""
    _, _, m, _ = test_data.fixture(xml="<mujoco/>")
    with self.assertRaises((TypeError, ValueError)):
      m.opt.deterministic = value
    self.assertEqual(m.opt.deterministic, DeterminismType.NONE)

  @parameterized.named_parameters(
    ("int", {"opt.deterministic": 0}, DeterminismType.NONE),
    ("bool_true", {"opt.deterministic": True}, DeterminismType.ALL),
    ("bool_false", {"opt.deterministic": False}, DeterminismType.NONE),
    ("numpy_bool", {"opt.deterministic": np.bool_(True)}, DeterminismType.ALL),
    ("flag", {"opt.deterministic": DeterminismType.ATOMICS}, DeterminismType.ATOMICS),
    ("complement", {"opt.deterministic": ~DeterminismType.CONTACTS}, 6),
    ("digits", {"opt.deterministic": "4"}, DeterminismType.ATOMICS),
    ("str_true", {"opt.deterministic": "true"}, DeterminismType.ALL),
    ("str_false", {"opt.deterministic": "False"}, DeterminismType.NONE),
    ("name", {"opt.deterministic": "atomics"}, DeterminismType.ATOMICS),
    ("names", {"opt.deterministic": "contacts | atomics"}, 5),
    ("none", ["opt.deterministic = NONE"], DeterminismType.NONE),
    ("all", ["opt.deterministic=ALL"], DeterminismType.ALL),
    ("negate", ["opt.deterministic=~CONTACTS"], DeterminismType.CONSTRAINT),
    ("negate_and_set", ["opt.deterministic=~CONTACTS|ATOMICS"], 6),
  )
  def test_override_model(self, overrides, expected):
    """override_model parses bools, ints and names; negation starts from the current value."""
    _, _, m, _ = test_data.fixture(xml="<mujoco/>")
    m.opt.deterministic = DeterminismType.CONTACTS | DeterminismType.CONSTRAINT
    override_model(m, overrides)
    self.assertEqual(m.opt.deterministic, expected)
    self.assertIs(type(m.opt.deterministic), int)

  @parameterized.parameters("FOO", "~FOO", "ATOMICS|FOO", "8", "-1", "1.5", 8, 1 << 40)
  def test_override_model_invalid(self, value):
    """override_model rejects unknown names and out of range values."""
    _, _, m, _ = test_data.fixture(xml="<mujoco/>")
    with self.assertRaises(ValueError):
      override_model(m, {"opt.deterministic": value})

  def test_override_fixture(self):
    """opt.deterministic overrides apply to mjw.Model and are skipped for mujoco.MjModel."""
    mjm, _, m, _ = test_data.fixture(xml="<mujoco/>", overrides={"opt.deterministic": "ATOMICS"})
    self.assertEqual(m.opt.deterministic, DeterminismType.ATOMICS)
    override_model(mjm, ["opt.deterministic=ALL"])

  @parameterized.parameters(range(8))
  def test_hooks(self, mask):
    """Sort hooks run iff their bit is set, as the last writer of their stage."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sleep_tolerance=".1">
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body pos="0 0 .05">
            <freejoint/>
            <geom type="box" size=".05 .05 .05"/>
          </body>
          <body pos=".5 0 .3">
            <freejoint/>
            <geom size=".05"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      qvel_noise=0.1,
    )
    m.opt.deterministic = mask
    events = []

    def contactfilter(m, d_filter):
      events.append(("contactfilter", d_filter is d, None))

    m.callback.contactfilter = contactfilter
    spies = (
      (collision_driver, "_sort_contacts", "contacts"),
      (collision_driver, "collision", "collision"),
      (constraint, "_sort_constraints", "constraint"),
      (constraint, "make_constraint", "make_constraint"),
    )
    with contextlib.ExitStack() as stack:
      for module, name, tag in spies:
        stack.enter_context(_spy(module, name, tag, events, d))
      launches = stack.enter_context(_record_launches())
      mjw.forward(m, d)

    tags = [tag for tag, _, _ in events]
    self.assertTrue(all(same for _, same, _ in events))
    self.assertEqual([tags.count(stage) for stage in ("collision", "contactfilter", "make_constraint")], [2, 2, 1])
    for hook, stage, bit in (
      ("contacts", "collision", DeterminismType.CONTACTS),
      ("constraint", "make_constraint", DeterminismType.CONSTRAINT),
    ):
      self.assertEqual(tags.count(hook), tags.count(stage) if mask & bit else 0, hook)
      for i, tag in enumerate(tags):
        if tag == hook:
          # contacts are sorted after the user contact filter
          if hook == "contacts":
            self.assertEqual(tags[i - 1], "contactfilter")
          # nothing writes Data between the hook and the end of its stage
          self.assertEqual(tags[i + 1], stage)
          self.assertEqual(_bitwise_diff(events[i][2], events[i + 1][2]), [], hook)

    atomics = bool(mask & DeterminismType.ATOMICS)
    for name, mode, _ in launches:
      self.assertEqual(mode == _RUN_TO_RUN, atomics and name in _DET_FACTORIES, name)

  @parameterized.named_parameters(
    ("default", {}, (2, 1)),
    ("contact_disabled", {"opt.disableflags": "CONTACT"}, (0, 1)),
    ("constraint_disabled", {"opt.disableflags": "CONSTRAINT"}, (0, 1)),
    # contacts provided by the user are not sorted
    ("user_contacts", {"opt.run_collision_detection": False}, (0, 1)),
  )
  def test_hook_preconditions(self, overrides, ncalls):
    """Sort hooks are skipped with the stages that do not run."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sleep_tolerance=".1">
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body pos="0 0 .05">
            <freejoint/>
            <geom type="box" size=".05 .05 .05"/>
          </body>
          <body pos=".5 0 .3">
            <freejoint/>
            <geom size=".05"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides=overrides,
      qvel_noise=0.1,
    )
    m.opt.deterministic = DeterminismType.ALL
    with (
      mock.patch.object(collision_driver, "_sort_contacts") as contacts,
      mock.patch.object(constraint, "_sort_constraints") as constraints,
    ):
      mjw.forward(m, d)
    self.assertEqual((contacts.call_count, constraints.call_count), ncalls)

  @absltest.skipIf(not wp.get_device().is_cuda, "CUDA graph capture requires CUDA")
  @parameterized.parameters(True, False)
  def test_graph_capture(self, graph_conditional):
    """Captured deterministic steps are bitwise equal to uncaptured steps and to each other."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag contact="disable"/>
        </option>
        <worldbody>
          <body pos="0 0 1">
            <freejoint name="root"/>
            <geom type="box" size=".1 .1 .1"/>
            <site name="site0" pos="0 0 .1"/>
            <body pos=".1 0 0">
              <joint name="hinge1" axis="0 1 0" range=".1 1"/>
              <geom type="capsule" size=".02" fromto="0 0 0 .2 0 0"/>
              <body pos=".2 0 0">
                <joint name="hinge2" axis="0 1 0" frictionloss=".01"/>
                <geom type="capsule" size=".02" fromto="0 0 0 .2 0 0"/>
                <site name="site1" pos=".2 0 0"/>
              </body>
            </body>
          </body>
        </worldbody>
        <tendon>
          <spatial name="tendon">
            <site site="site0"/>
            <site site="site1"/>
          </spatial>
        </tendon>
        <actuator>
          <motor joint="root" gear="1 0 0 0 0 1"/>
          <motor joint="hinge1"/>
          <position tendon="tendon" kp="10"/>
          <general site="site1"/>
        </actuator>
      </mujoco>
      """,
      nworld=2,
      qvel_noise=0.1,
      ctrl_noise=0.5,
    )
    _perturb(d)
    m.opt.deterministic = DeterminismType.ATOMICS
    m.opt.graph_conditional = graph_conditional
    init = _snapshot(_flatten(d, "d"))
    mjw.step(m, d)
    ref = _snapshot(_flatten(d, "d"))
    self.assertFalse(np.array_equal(ref["d.qacc"][0], ref["d.qacc"][1]))

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    for _ in range(_NREPEAT):
      _restore(_flatten(d, "d"), init)
      wp.capture_launch(capture.graph)
      self.assertEqual(_bitwise_diff(ref, _snapshot(_flatten(d, "d"))), [])

  def test_mode_toggle(self):
    """Changing opt.deterministic between steps switches kernels without stale state."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag contact="disable"/>
        </option>
        <worldbody>
          <body pos="0 0 1">
            <freejoint name="root"/>
            <geom type="box" size=".1 .1 .1"/>
            <site name="site0" pos="0 0 .1"/>
            <body pos=".1 0 0">
              <joint name="hinge1" axis="0 1 0" range=".1 1"/>
              <geom type="capsule" size=".02" fromto="0 0 0 .2 0 0"/>
              <body pos=".2 0 0">
                <joint name="hinge2" axis="0 1 0" frictionloss=".01"/>
                <geom type="capsule" size=".02" fromto="0 0 0 .2 0 0"/>
                <site name="site1" pos=".2 0 0"/>
              </body>
            </body>
          </body>
        </worldbody>
        <tendon>
          <spatial name="tendon">
            <site site="site0"/>
            <site site="site1"/>
          </spatial>
        </tendon>
        <actuator>
          <motor joint="root" gear="1 0 0 0 0 1"/>
          <motor joint="hinge1"/>
          <position tendon="tendon" kp="10"/>
          <general site="site1"/>
        </actuator>
      </mujoco>
      """,
      nworld=2,
      qvel_noise=0.1,
      ctrl_noise=0.5,
    )
    _perturb(d)
    init = _snapshot(_flatten(d, "d"))
    modes = (DeterminismType.NONE, DeterminismType.ATOMICS, DeterminismType.NONE, DeterminismType.ALL, DeterminismType.NONE)
    for mode in modes:
      m.opt.deterministic = mode
      with _record_launches() as launches:
        mjw.step(m, d)
      for name, launch_mode, _ in launches:
        self.assertEqual(launch_mode == _RUN_TO_RUN, bool(mode & DeterminismType.ATOMICS) and name in _DET_FACTORIES, name)
    toggled = _snapshot(_flatten(d, "d"))

    _restore(_flatten(d, "d"), init)
    m.opt.deterministic = DeterminismType.NONE
    for _ in modes:
      mjw.step(m, d)
    none = _snapshot(_flatten(d, "d"))

    for field in ("qpos", "qvel", "qacc", "actuator_force", "qfrc_constraint"):
      np.testing.assert_allclose(toggled[f"d.{field}"], none[f"d.{field}"], rtol=1e-4, atol=1e-4, err_msg=field)
    self.assertFalse(np.array_equal(none["d.qacc"][0], none["d.qacc"][1]))

  def test_no_dofs(self):
    """All determinism features run on models without degrees of freedom or trees."""
    _, _, m, d = test_data.fixture(xml="<mujoco/>")
    m.opt.deterministic = DeterminismType.ALL
    mjw.step(m, d)

    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="1 1 .1"/>
          <body mocap="true" pos="0 0 .05">
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    m.opt.deterministic = DeterminismType.ALL
    d.body_awake_ind.fill_(-1)
    mjw.step(m, d)
    np.testing.assert_array_equal(d.body_awake_ind.numpy(), [[0, 1]])

  def test_factory_coverage(self):
    """Each deterministic kernel factory is exercised by a scene case."""
    covers = set()
    for case in (
      *_TREE_CASES,
      *_FLEX_CASES,
      *_FLEX_SHELL_CASES,
      *_FLEX_CONTACT_CASES,
      *_FLEX_PASSIVE_CASES,
      *_CONTACT_CASES,
      *_SLEEP_CASES,
    ):
      covers.update(case["covers"])
    self.assertSetEqual(covers, set(_DET_FACTORIES))

  def test_factory_signatures(self):
    """Deterministic factories end with `deterministic=False`, and records-only args are found."""
    self.assertNotEmpty(_DET_FACTORIES)
    for name, factory in _DET_FACTORIES.items():
      param = list(inspect.signature(factory).parameters.values())[-1]
      self.assertEqual((param.name, param.default), ("deterministic", False), name)

    self.assertEqual(_RECORDS_ONLY_ARGS["passive._qfrc_adhesion"], ["nv"])
    self.assertEqual(_RECORDS_ONLY_ARGS["derivative._eff_flex_interp_mul"], ["nv"])
    self.assertEqual(_RECORDS_ONLY_ARGS["solver._JTDACJ_sparse"], ["max_records"])
    self.assertEqual(_RECORDS_ONLY_ARGS["solver._linesearch_jv_fused_kernel"], [])
    self.assertEqual(_RECORDS_ONLY_ARGS["smooth._transmission"], [])

    # kernel cache keys are (*args, module, name)
    _DET_FACTORIES["smooth._transmission"](False)
    self.assertIn((hash(False), "mujoco_warp._src.smooth", "_transmission"), warp_util._KERNEL_CACHE)


if __name__ == "__main__":
  wp.init()
  absltest.main()
