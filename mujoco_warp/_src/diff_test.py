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
"""Tests for the MuJoCo Warp differentiability API."""

import contextlib
import dataclasses
import gc
import inspect

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized
from warp._src import context as wp_context

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import diff

_API_FUNCTIONS = tuple(diff.RULES)


@wp.kernel(module="unique", enable_backward=True)
def _linear_step_leaf(
  # Model:
  body_mass: wp.array2d[float],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  # Data out:
  qpos_out: wp.array2d[float],
):
  w, i = wp.tid()
  qpos_out[w, i] = 2.0 * qpos_in[w, i] + 0.5 * qvel_in[w, i] + 3.0 * body_mass[w % body_mass.shape[0], 1]


@wp.kernel(module="unique", enable_backward=True)
def _weighted_loss_kernel(
  # Data in:
  qpos_in: wp.array2d[float],
  # In:
  weights: wp.array[float],
  # Out:
  loss_out: wp.array[float],
):
  w, i = wp.tid()
  wp.atomic_add(loss_out, 0, weights[w] * qpos_in[w, i])


@wp.kernel(module="unique")
def _fk_bwd_kernel(
  # In:
  site_xpos_grad: wp.array2d[wp.vec3],
  # Out:
  qpos_grad_out: wp.array2d[float],
):
  w = wp.tid()
  g = site_xpos_grad[w, 0]
  qpos_grad_out[w, 0] += (g[0] + g[1] + g[2]) / 3.0


@wp.kernel(module="unique", enable_backward=True)
def _scale_kernel(
  # In:
  x: wp.array2d[float],
  # Out:
  y_out: wp.array2d[float],
):
  w, i = wp.tid()
  y_out[w, i] = 2.0 * x[w, i]


@wp.kernel(module="unique", enable_backward=False)
def _axpy_kernel(
  # In:
  a: float,
  x: wp.array2d[float],
  # Out:
  y_out: wp.array2d[float],
):
  w, i = wp.tid()
  y_out[w, i] += a * x[w, i]


def _leaf_step(m, d, d_out=None):
  """Linear stand-in for `step`: qpos' = 2 qpos + 0.5 qvel + 3 body_mass[1]."""
  target = d if d_out is None else d_out
  wp.launch(_linear_step_leaf, dim=d.qpos.shape, inputs=[m.body_mass, d.qpos, d.qvel], outputs=[target.qpos])


def _leaf_step_bwd(m, d, d_out=None):
  """Backward of `_leaf_step` via the kernel adjoint."""
  target = d if d_out is None else d_out
  adj_out = wp.clone(target.qpos.grad)
  target.qpos.grad.zero_()  # qpos' overwrites qpos when in place
  wp.launch(
    _linear_step_leaf,
    dim=d.qpos.shape,
    inputs=[m.body_mass, d.qpos, d.qvel],
    outputs=[target.qpos],
    adj_inputs=[m.body_mass.grad, d.qpos.grad, d.qvel.grad],
    adj_outputs=[adj_out],
    adjoint=True,
  )


@contextlib.contextmanager
def _requires_grad(*arrays: wp.array):
  """Enables gradients on `arrays` within the context, as a user would."""
  for a in arrays:
    a.requires_grad = True
  yield
  for a in arrays:
    a.requires_grad = False


def _promote(*arrays: wp.array):
  """Enables gradients on `arrays` as an earlier differentiable call would."""
  for a in arrays:
    a.requires_grad = True
    diff._PROMOTED.add(a)


def _register(fn, backward=None, **rule):
  """Wraps `fn` like an API function and registers its rule."""
  wrapped = diff._wrap(fn)
  diff.register_adjoint(fn, backward, **rule)
  return wrapped


def _float_arrays(obj, prefix=""):
  """Returns the paths of the float Warp arrays reachable from a dataclass container."""
  paths = []
  for name, val in vars(obj).items():
    items = enumerate(val) if isinstance(val, (tuple, list)) else ((None, val),)
    for i, item in items:
      path = f"{prefix}{name}" if i is None else f"{prefix}{name}[{i}]"
      if dataclasses.is_dataclass(item):
        paths.extend(_float_arrays(item, f"{path}."))
      elif isinstance(item, wp.array) and diff._is_float(item):
        paths.append(path)
  return paths


class DiffApiTest(parameterized.TestCase):
  """Tests for the differentiability API."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.mjm, _, cls.m, cls.d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <joint/>
            <geom size="0.1"/>
            <site pos="0.1 0 0"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    cls.rc = mjw.create_render_context(cls.mjm)

  def tearDown(self):
    # Restore the API rules, remove test functions, and disable gradients on the shared fixture:
    for k in list(diff.RULES):
      if k in _API_FUNCTIONS:
        diff.RULES[k] = None
      else:
        del diff.RULES[k], diff._PARAMS[k]
    diff._PROMOTED.clear()
    for obj in (self.m, self.d, self.rc):
      for path in diff._FIELDS[type(obj)]:
        arr = diff.get(obj, path)
        if isinstance(arr, wp.array) and arr.requires_grad:
          arr.requires_grad = False
    super().tearDown()

  def _assert_not_supported(self, msg: str, fn, *args, **kwargs):
    """Asserts that calling `fn` under a tape raises `NotImplementedError` without recording."""
    with wp.Tape() as tape, self.assertRaisesRegex(NotImplementedError, msg):
      fn(*args, **kwargs)
    self.assertEmpty(tape.launches)

  def test_differentiable(self):
    # Every public function is wrapped, keeps its metadata, and has no rule:
    public = {name: fn for name, fn in vars(mjw).items() if inspect.isfunction(fn) and not name.startswith("_")}
    self.assertSetEqual(set(public), set(diff.RULES))
    for name, fn in public.items():
      self.assertIsNone(diff.RULES[name])
      self.assertEqual((fn.__name__, fn.__doc__), (name, fn.__wrapped__.__doc__))
      self.assertEqual(diff._PARAMS[name], tuple(inspect.signature(fn).parameters))
    for name, obj in vars(mjw).items():
      if callable(obj) and not name.startswith("_") and name not in public:
        self.assertTrue(inspect.isclass(obj), name)

    # Only public Python functions are wrapped; wrappers pass through outside a tape:
    def custom(m, d):
      """Custom function."""
      return "result"

    namespace = {"custom": custom, "_private": lambda: None, "Model": mjw.Model, "kernel": _scale_kernel, "value": 1}
    unwrapped = dict(namespace)
    diff.differentiable(namespace)
    wrapped = namespace["custom"]
    self.assertEqual(namespace, {**unwrapped, "custom": wrapped})
    self.assertIs(wrapped.__wrapped__, custom)
    self.assertEqual((wrapped.__name__, wrapped.__doc__), ("custom", "Custom function."))
    self.assertNotIn("_private", diff.RULES)
    self.assertEqual(wrapped(self.m, self.d), "result")
    self._assert_not_supported("'custom' is not supported", wrapped, self.m, self.d)

  def test_fields(self):
    # _FIELDS covers every float array of Model, Data, and RenderContext, including nested ones:
    for obj in (self.m, self.d, self.rc):
      self.assertContainsSubset(_float_arrays(obj), diff._FIELDS[type(obj)])
    self.assertIs(diff.get(self.d, "efc.pos"), self.d.efc.pos)
    self.assertIs(diff.get(self.m, "opt.gravity"), self.m.opt.gravity)

    # Bracket annotations (used by the documentation build) are array fields:
    @dataclasses.dataclass
    class Fields:
      a: wp.array[float]
      b: wp.array2d[float]
      c: int

    self.assertEqual(diff._fields(Fields), ("a", "b"))

  def test_warp_internals(self):
    # The Warp internals diff.py relies on:
    self.assertIsNone(wp_context.runtime.tape)
    for arr, expected in (
      (self.d.qpos, True),
      (self.d.site_xpos, True),
      (self.d.xquat, True),
      (self.d.xmat, True),
      (self.d.cdof, True),
      (self.d.nefc, False),
      (wp.zeros(1, dtype=wp.vec2i), False),
    ):
      self.assertEqual(diff._is_float(arr), expected)

    with _requires_grad(self.d.qpos, self.d.qacc), wp.Tape() as tape:
      self.assertIs(wp_context.runtime.tape, tape)
      wp.launch(_scale_kernel, dim=self.d.qpos.shape, inputs=[self.d.qpos], outputs=[self.d.qacc])
      wp.copy(self.d.qacc, self.d.qpos)
      tape.record_func(backward=lambda: None, arrays=[self.d.qpos])
      wp.launch(_axpy_kernel, dim=self.d.qacc.shape, inputs=[1.0, self.d.qacc], outputs=[self.d.qvel])
      self.assertIsNone(diff._call(None, lambda: wp_context.runtime.tape))
      self.assertIs(wp_context.runtime.tape, tape)
    self.assertIsNone(wp_context.runtime.tape)

    # Launches are lists, copies and custom functions are callables, and launches and copies access
    # their arrays:
    launch, copy, func, axpy = tape.launches
    self.assertEqual(
      (launch[0], launch[1], launch[3], launch[4]), (_scale_kernel, self.d.qpos.shape, [self.d.qpos], [self.d.qacc])
    )
    self.assertTrue(callable(copy) and callable(func))
    self.assertEqual([diff._has_adjoint(l) for l in tape.launches], [True, True, True, False])
    self.assertEqual(diff._accessed([copy, func]), {id(self.d.qpos), id(self.d.qacc)})
    self.assertEqual(diff._accessed([axpy]), {id(self.d.qacc), id(self.d.qvel)})

  @parameterized.parameters(*sorted(_API_FUNCTIONS))
  def test_api_function_raises_not_implemented(self, fn_name: str):
    # Functions without a rule raise under a tape before reading their arguments:
    self._assert_not_supported(f"'{fn_name}' is not supported", vars(mjw)[fn_name])

  def test_register_adjoint(self):
    def bwd(*args, **kwargs):
      pass

    def supported(*args, **kwargs):
      return ()

    # Rules register by name or function (wrapped or not), and out_fields defaults to in_fields:
    rule = {"in_fields": ["qpos", "efc.pos"], "model_fields": ["opt.gravity"], "supported": supported}
    self.assertIs(diff.register_adjoint("step", bwd, **rule), bwd)
    expected = diff.AdjointRule(bwd, ("qpos", "efc.pos"), ("qpos", "efc.pos"), ("opt.gravity",), supported=supported)
    self.assertEqual(diff.RULES["step"], expected)

    diff.register_adjoint("step", bwd, in_fields=("qpos",), out_fields=())
    self.assertEqual(diff.RULES["step"], diff.AdjointRule(bwd, ("qpos",)))

    for fn in (mjw.step, mjw.step.__wrapped__):
      diff.register_adjoint(fn, bwd, in_fields=("qvel",))
      self.assertEqual(diff.RULES["step"], diff.AdjointRule(bwd, ("qvel",), ("qvel",)))

    # Without a backward, a native rule is registered and a decorator is returned:
    decorator = diff.register_adjoint("kinematics", in_fields=("qpos",))
    self.assertEqual(diff.RULES["kinematics"], diff.AdjointRule(None, ("qpos",), ("qpos",)))
    self.assertIs(decorator(bwd), bwd)
    self.assertIs(diff.RULES["kinematics"].backward, bwd)

    # Invalid registrations raise without changing the registered rule:
    prev = diff.RULES["step"]
    for target, invalid, msg in (
      ("unknown_fn", {}, "is not registered"),
      (lambda m, d: None, {}, "is not registered"),
      ("step", {"in_fields": "qpos"}, "Invalid Data fields"),
      ("step", {"in_fields": (f for f in ("qpos",))}, "Invalid Data fields"),
      ("step", {"in_fields": ("unknown",)}, "Invalid Data fields"),
      ("step", {"in_fields": ("efc",)}, "Invalid Data fields"),
      ("step", {"out_fields": ("body_mass",)}, "Invalid Data fields"),
      ("step", {"model_fields": "body_mass"}, "Invalid Model fields"),
      ("step", {"model_fields": ("qpos",)}, "Invalid Model fields"),
      ("step", {"model_fields": ("opt",)}, "Invalid Model fields"),
      ("step", {"render_fields": "lower"}, "Invalid RenderContext fields"),
      ("step", {"render_fields": ("qpos",)}, "Invalid RenderContext fields"),
      ("step", {"array_args": "m"}, "Invalid array_args"),
      ("step", {"array_args": ("unknown",)}, "Invalid array_args"),
    ):
      with self.assertRaisesRegex(ValueError, msg):
        diff.register_adjoint(target, bwd, **invalid)
    self.assertIs(diff.RULES["step"], prev)

  @parameterized.parameters(*((cls, p) for cls, paths in diff._FIELDS.items() for p in paths))
  def test_field_requires_grad_raises_not_implemented(self, cls: type, path: str):
    # Gradients on any field the rule does not differentiate raise before execution:
    obj = {mjw.Model: self.m, mjw.Data: self.d, mjw.RenderContext: self.rc}[cls]
    arr = diff.get(obj, path)
    if not isinstance(arr, wp.array):
      self.skipTest(f"{cls.__name__} field '{path}' is not a Warp array.")
    diff.register_adjoint("render", lambda *args: None)
    with _requires_grad(arr):
      self._assert_not_supported(f"{cls.__name__} field '{path}'", mjw.render, self.m, self.d, self.rc)

  @parameterized.parameters(
    ("m", "body_mass", None),
    ("d", "qpos", None),
    ("d_out", "qacc", None),
    ("rc", "lower", None),
    ("m", "dof_damping", "Model"),
    ("m_extra", "body_mass", "Model"),
    ("d", "qacc", "Data"),
    ("d_mid", "qpos", "Data"),
    ("d_out", "qpos", "Data"),
    ("rc", "upper", "RenderContext"),
    ("rc_extra", "lower", "RenderContext"),
  )
  def test_container_roles(self, arg: str, path: str, error: str | None):
    def roles(m, d, d_mid, d_out, m_extra, rc, rc_extra):
      pass

    # The first Model, the first Data, the last Data, and the first RenderContext have the model,
    # input, output, and render roles; other containers have no role:
    wrapped = _register(
      roles,
      lambda *args, **kwargs: None,
      in_fields=("qpos",),
      out_fields=("qacc",),
      model_fields=("body_mass",),
      render_fields=("lower",),
    )
    containers = {
      "m": self.m,
      "d": self.d,
      "d_mid": mjw.make_data(self.mjm),
      "d_out": mjw.make_data(self.mjm),
      "m_extra": mjw.put_model(self.mjm),
      "rc": self.rc,
      "rc_extra": mjw.create_render_context(self.mjm),
    }
    arr = diff.get(containers[arg], path)
    with _requires_grad(arr), wp.Tape() as tape:
      if error is None:
        wrapped(**containers)
      else:
        with self.assertRaisesRegex(NotImplementedError, f"{error} field '{path}'"):
          wrapped(**containers)
    self.assertEqual(arr in tape.gradients, error is None)

  def test_array_args(self):
    def scale(m, d, x, y, z=None):
      wp.launch(_scale_kernel, dim=x.shape, inputs=[x], outputs=[y])

    def scale_bwd(m, d, x, y, z=None):
      wp.launch(_axpy_kernel, dim=x.shape, inputs=[2.0, y.grad], outputs=[x.grad])

    wrapped = _register(scale, scale_bwd, array_args=("x", "y"))
    x = wp.full((1, 3), 1.5, dtype=float)
    y = wp.full((1, 3), wp.inf, dtype=float)
    z = wp.zeros((1, 3))

    # An array that would get gradients as an array_arg raises when also passed as another argument:
    self._assert_not_supported("'scale' with array argument 'z'", wrapped, self.m, self.d, x, y, z=x)
    self.assertFalse(x.requires_grad or y.requires_grad)

    # array_args bind by name (positionally or by keyword), get gradients, and are tracked once:
    with wp.Tape() as tape:
      wrapped(self.m, self.d, y=y, x=x, z=z)
    self.assertCountEqual(map(id, tape.gradients), map(id, (x, y)))
    self.assertFalse(z.requires_grad)
    np.testing.assert_allclose(y.numpy(), 3.0)
    y.grad.fill_(1.0)
    tape.backward()
    np.testing.assert_allclose(x.grad.numpy(), 2.0)

    with wp.Tape() as tape:
      wrapped(self.m, self.d, x, x)
    self.assertCountEqual(map(id, tape.gradients), [id(x)])

    # Empty arrays do not get gradients, and the backward is still recorded:
    x0, y0 = wp.zeros((1, 0)), wp.zeros((1, 0))
    with wp.Tape() as tape:
      wrapped(self.m, self.d, x0, y0)
    self.assertFalse(x0.requires_grad or y0.requires_grad)
    self.assertEmpty(tape.gradients)
    self.assertLen(tape.launches, 1)

    # Other arrays with gradients raise before execution, including arrays promoted as array_args:
    y.fill_(wp.inf)
    with _requires_grad(z):
      for args in ((x, y, z), (x, z, y)):
        self._assert_not_supported("'scale' with array argument 'z'", wrapped, self.m, self.d, *args)
    self.assertTrue(np.all(np.isinf(y.numpy())) and not np.any(z.numpy()))

    # Data fields can be array_args across calls, including fields with user-enabled gradients:
    y_qvel = wp.full(self.d.qvel.shape, wp.inf, dtype=float)
    for _ in range(2):
      with wp.Tape() as tape:
        wrapped(self.m, self.d, self.d.qvel, y_qvel)
    y_qvel.grad.fill_(1.0)
    tape.backward()
    np.testing.assert_allclose(self.d.qvel.grad.numpy(), 2.0)

    with _requires_grad(self.d.qpos), wp.Tape() as tape:
      wrapped(self.m, self.d, self.d.qpos, y_qvel)
    self.assertIn(self.d.qpos, tape.gradients)

    # Re-registering keeps promoted gradients, and arrays no longer in array_args then raise:
    diff.register_adjoint(scale, scale_bwd, array_args=("x",))
    self.assertTrue(y.requires_grad)
    self._assert_not_supported("'scale' with array argument 'y'", wrapped, self.m, self.d, x, y)

    # Disabling gradients after recording raises in the backward instead of reading freed memory:
    with wp.Tape() as tape:
      wrapped(self.m, self.d, x, z)
    x.requires_grad = False
    with self.assertRaisesRegex(RuntimeError, "Gradients for 'scale' were disabled"):
      tape.backward()

  def test_supported(self):
    tapes = []

    def checked(m, d, reasons=None):
      wp.launch(_scale_kernel, dim=d.qpos.shape, inputs=[d.qpos], outputs=[d.qacc])
      return "ok"

    def supported(m, d, reasons=None):
      tapes.append(wp_context.runtime.tape)
      wp.launch(_scale_kernel, dim=d.qpos.shape, inputs=[d.qpos], outputs=[d.qacc])
      mjw.kinematics(m, d)
      return reasons

    wrapped = _register(checked, lambda *args, **kwargs: None, in_fields=("qpos",), supported=supported)

    # Untaped calls skip supported; under a tape, field validation runs before supported:
    with _requires_grad(self.d.qvel):
      self.assertEqual(wrapped(self.m, self.d, reasons="unsupported"), "ok")
      self.assertFalse(self.d.qpos.requires_grad)
      self._assert_not_supported("Data field 'qvel'", wrapped, self.m, self.d, reasons="unsupported")
    self.assertEmpty(tapes)

    # supported receives the call arguments, and its reasons raise before promotion:
    for reasons, expected in ((("first", "second"), "first, second"), ("single", "single")):
      self._assert_not_supported(f"'checked' is not supported: {expected}", wrapped, self.m, self.d, reasons=reasons)
      self.assertFalse(self.d.qpos.requires_grad)

    # supported runs untaped (its launches and API calls are not recorded); (), [], None pass:
    for reasons in ((), [], None):
      with wp.Tape() as tape:
        self.assertEqual(wrapped(self.m, self.d, reasons=reasons), "ok")
      self.assertLen(tape.launches, 1)
      self.assertTrue(callable(tape.launches[0]))
    self.assertEqual(tapes, [None] * 5)

  def test_auto_promotion(self):
    # Gradients on integer arrays raise before execution:
    x_int = wp.zeros((1, 1), dtype=int)

    def int_fields(m, d, x):
      pass

    fn_int = _register(int_fields, lambda *args: None, in_fields=("nefc",), model_fields=("body_parentid",), array_args=("x",))
    for arr, error in (
      (self.d.nefc, "Data field 'nefc'"),
      (self.m.body_parentid, "Model field 'body_parentid'"),
      (x_int, "with array argument 'x'"),
    ):
      with _requires_grad(arr):
        self._assert_not_supported(error, fn_int, self.m, self.d, x_int)

    # Non-empty float arrays (scalar, nested, vec3, quat, mat33, spatial, vec10) with the input,
    # output, or render role, and array_args, get gradients before execution; integer, empty, and
    # (opt-in) Model arrays do not:
    float_paths = ("qpos", "efc.pos", "site_xpos", "xquat", "xmat", "cdof", "cinert")
    seen, calls = [], []

    def promote_roles(m, d, rc, x, y=None, d_out=None):
      seen.append(all(diff.get(d, p).requires_grad for p in float_paths))

    wrapped = _register(
      promote_roles,
      lambda *args, **kwargs: calls.append(1),
      in_fields=(*float_paths, "qvel", "nefc", "ten_length"),
      out_fields=("qacc",),
      model_fields=("body_mass",),
      render_fields=("lower",),
      array_args=("x", "y"),
    )
    d_out = mjw.make_data(self.mjm)
    x, y = wp.zeros((1, 1)), wp.zeros((1, 1), requires_grad=True)
    promoted = (*(diff.get(self.d, p) for p in float_paths), d_out.qacc, self.rc.lower, x)
    unpromoted = (self.d.nefc, self.d.ten_length, self.d.qacc, d_out.qpos, self.m.body_mass)
    with _requires_grad(self.d.qvel), wp.Tape() as tape:
      wrapped(self.m, self.d, self.rc, x, y, d_out=d_out)
      # Arrays with user-enabled gradients are tracked but not promoted:
      self.assertCountEqual(map(id, tape.gradients), map(id, (*promoted, self.d.qvel, y)))
    self.assertEqual(seen, [True])
    self.assertTrue(all(a.requires_grad for a in promoted))
    self.assertFalse(any(a.requires_grad for a in unpromoted))
    self.assertCountEqual(map(id, diff._PROMOTED), map(id, promoted))

    # Promoted gradients persist across calls, and Model fields are tracked when enabled:
    grads = [a.grad for a in promoted]
    with _requires_grad(self.m.body_mass), wp.Tape() as tape:
      wrapped(self.m, self.d, self.rc, x, d_out=d_out)
      self.assertIn(self.m.body_mass, tape.gradients)
    self.assertTrue(all(a.grad is g for a, g in zip(promoted, grads)))

    # In-place calls (d_out omitted, None, or d) give Data both roles, and arrays shared between
    # Data instances are allowed by identity:
    d_shared = dataclasses.replace(self.d, qacc=wp.zeros(self.d.qacc.shape, dtype=float))
    for kwargs in ({}, {"d_out": None}, {"d_out": self.d}, {"d_out": d_shared}):
      with wp.Tape():
        wrapped(self.m, self.d, self.rc, x, **kwargs)
    self.assertTrue(self.d.qacc.requires_grad and d_shared.qacc.requires_grad)

    # Promoted arrays are weakly referenced:
    num_promoted = len(diff._PROMOTED)
    d_temp = mjw.make_data(self.mjm)
    with wp.Tape():
      wrapped(self.m, d_temp, self.rc, x)
    self.assertGreater(len(diff._PROMOTED), num_promoted)
    del d_temp
    gc.collect()
    self.assertLen(diff._PROMOTED, num_promoted)

    # A rule with only integer and empty fields tracks no arrays and still runs its backward:
    diff.register_adjoint(promote_roles, lambda *args, **kwargs: calls.append(1), in_fields=("nefc", "ten_length"))
    with wp.Tape() as tape:
      wrapped(self.m, self.d, self.rc, wp.zeros((1, 1)))
    self.assertEmpty(tape.gradients)
    tape.backward()
    self.assertEqual(calls, [1])

  def test_promoted_field_dependencies(self):
    def deps(m, d, rc, mode):
      if mode == "read":
        wp.launch(_scale_kernel, dim=d.qvel.shape, inputs=[d.qvel], outputs=[d.qacc])
      elif mode == "write":
        wp.launch(_scale_kernel, dim=d.qpos.shape, inputs=[d.qpos], outputs=[d.qvel])
      elif mode == "zero_dim":
        wp.launch(_scale_kernel, dim=(d.nworld, 0), inputs=[d.qvel], outputs=[d.qvel])
      elif mode == "copy_read":
        wp.copy(d.qacc, d.qvel)
      elif mode == "copy_write":
        wp.copy(d.qvel, d.qpos)
      elif mode == "rc_read":
        wp.copy(rc.upper, rc.lower)

    # Fields promoted by an earlier call but not differentiated by this rule are allowed unless
    # read or written, which raises and leaves the tape unchanged (custom and native rules):
    _promote(self.d.qvel, self.rc.lower)
    for backward in (lambda *args: None, None):
      wrapped = _register(deps, backward, in_fields=("qpos",), out_fields=("qacc",), render_fields=("upper",))
      for mode in ("unaccessed", "zero_dim"):
        with wp.Tape():
          wrapped(self.m, self.d, self.rc, mode)
      for mode, expected in (
        ("read", "Data field 'qvel'"),
        ("write", "Data field 'qvel'"),
        ("copy_read", "Data field 'qvel'"),
        ("copy_write", "Data field 'qvel'"),
        ("rc_read", "RenderContext field 'lower'"),
      ):
        self._assert_not_supported(expected, wrapped, self.m, self.d, self.rc, mode)

    # API kernels are compiled with enable_backward=False, so native rules for them raise:
    diff.register_adjoint("kinematics", in_fields=("qpos",), out_fields=("xpos",))
    self._assert_not_supported("'kinematics' requires a backward rule", mjw.kinematics, self.m, self.d)

  @parameterized.product(nworld=(1, 2), mode=("in_place", "out_of_place", "native"))
  def test_rollout(self, nworld: int, mode: str):
    # Gradients of a 3-step rollout of _leaf_step are exact, over two epochs with tape.zero():
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <joint/>
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    step = _register(
      _leaf_step,
      None if mode == "native" else _leaf_step_bwd,
      in_fields=("qpos", "qvel"),
      out_fields=("qpos",),
      model_fields=("body_mass",),
    )
    in_place = mode == "in_place"
    datas = [d] if in_place else [d, *(mjw.make_data(mjm, nworld=nworld) for _ in range(3))]
    qpos0 = np.array([[0.3], [-0.4]][:nworld], dtype=np.float32)
    qvel0 = np.array([[-0.2], [0.5]][:nworld], dtype=np.float32)
    d.qvel.assign(qvel0)
    m.body_mass.requires_grad = True
    body_mass = m.body_mass.numpy()[0, 1]
    seed = np.arange(1, nworld + 1, dtype=np.float32).reshape(nworld, 1)
    # d(final qpos)/d(qvel) of each Data (out-of-place Data after the first have zero qvel):
    qvel_coef = (3.5,) if in_place else (2.0, 1.0, 0.5)

    for _ in range(2):
      d.qpos.assign(qpos0)
      for data in datas[1:]:
        data.qpos.fill_(wp.inf)
      with wp.Tape() as tape:
        for t in range(3):
          if in_place:
            step(m, d)
          else:
            step(d_out=datas[t + 1], d=datas[t], m=m)
      self.assertLen(tape.launches, 3)
      final = datas[-1]
      expected = 8.0 * qpos0 + qvel_coef[0] * qvel0 + 21.0 * body_mass
      np.testing.assert_allclose(final.qpos.numpy(), expected, rtol=1e-6)

      final.qpos.grad.assign(seed)
      tape.backward()
      np.testing.assert_allclose(d.qpos.grad.numpy(), 8.0 * seed, rtol=1e-6)
      for data, coef in zip(datas, qvel_coef):
        np.testing.assert_allclose(data.qvel.grad.numpy(), coef * seed, rtol=1e-6)
      np.testing.assert_allclose(m.body_mass.grad.numpy()[0, 1], 21.0 * seed.sum(), rtol=1e-6)
      if nworld == 2:
        self.assertFalse(np.allclose(final.qpos.numpy()[0], final.qpos.numpy()[1]))
        self.assertFalse(np.allclose(d.qpos.grad.numpy()[0], d.qpos.grad.numpy()[1]))
      tape.zero()

  @parameterized.parameters(1, 2)
  def test_multi_function_composition(self, nworld: int):
    # API functions compose on one tape: step, forward, fwd_kinematics, sensors, and render.
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <camera pos="0 -2 0.5" xyaxes="1 0 0 0 0 1"/>
          <body>
            <joint name="j" type="slide"/>
            <geom size="0.3"/>
            <site name="s" pos="0.1 0 0"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="j"/>
        </actuator>
        <sensor>
          <jointpos joint="j"/>
          <jointvel joint="j"/>
          <accelerometer site="s"/>
        </sensor>
      </mujoco>
      """,
      nworld=nworld,
    )
    d_ref = mjw.put_data(mjm, mjd, nworld=nworld)
    for data in (d, d_ref):
      data.qpos.assign(np.array([[0.3], [-0.4]][:nworld], dtype=np.float32))
      data.qvel.assign(np.array([[-0.2], [0.5]][:nworld], dtype=np.float32))
      data.ctrl.assign(np.array([[0.4], [-0.3]][:nworld], dtype=np.float32))
      for path in ("qacc", "site_xpos", "sensordata"):
        diff.get(data, path).fill_(wp.inf)
    rc = mjw.create_render_context(mjm, nworld=nworld, cam_res=(8, 8), render_rgb=False, render_depth=True)
    rc.depth_data.fill_(wp.inf)

    # The step backward is the _leaf_step adjoint: qpos.grad = 2 g, qvel.grad = 0.5 g, and
    # body_mass.grad = 3 g for output gradient g; forward and sensors contribute nothing. Rules list
    # the promoted fields they overwrite (forward recomputes efc.pos from step):
    diff.register_adjoint(
      "step",
      _leaf_step_bwd,
      in_fields=("qpos", "qvel", "ctrl"),
      out_fields=("qpos", "qvel", "qacc", "efc.pos"),
      model_fields=("body_mass", "opt.gravity"),
    )
    for name, in_f, out_f in (
      ("forward", ("qpos", "qvel", "ctrl"), ("qacc", "efc.pos")),
      ("sensor_pos", ("qpos", "site_xpos"), ("sensordata",)),
      ("sensor_vel", ("qvel", "site_xpos"), ("sensordata",)),
      ("sensor_acc", ("qacc", "qvel", "site_xpos"), ("sensordata",)),
    ):
      diff.register_adjoint(name, lambda *args: None, in_fields=in_f, out_fields=out_f, model_fields=("body_mass",))

    # The fwd_kinematics backward adds the mean site_xpos gradient to qpos.grad, and can call
    # API functions because the tape is paused:
    @diff.register_adjoint("fwd_kinematics", in_fields=("qpos",), out_fields=("site_xpos",), model_fields=("body_mass",))
    def _fk_bwd(m_arg, d_arg):
      mjw.kinematics(m_arg, d_arg)
      wp.launch(_fk_bwd_kernel, dim=nworld, inputs=[d_arg.site_xpos.grad], outputs=[d_arg.qpos.grad])

    depth_seeds = []
    diff.register_adjoint(
      "render",
      lambda m_arg, d_arg, rc_arg: depth_seeds.append(rc_arg.depth_data.grad.numpy().sum()),
      render_fields=("depth_data",),
      model_fields=("body_mass",),
    )
    m.body_mass.requires_grad = True

    # Unsupported fields, including fields supported only by another rule, raise before execution:
    for obj, field, fn, args in (
      (m, "geom_rbound", mjw.step, (m, d)),
      (m, "opt.gravity", mjw.fwd_kinematics, (m, d)),
      (d, "qvel", mjw.fwd_kinematics, (m, d)),
      (rc, "ray", mjw.render, (m, d, rc)),
    ):
      with _requires_grad(diff.get(obj, field)):
        self._assert_not_supported(f"field '{field}'", fn, *args)
    self.assertTrue(np.all(np.isinf(d.site_xpos.numpy())) and np.all(np.isinf(rc.depth_data.numpy())))

    # Untaped API calls can run between recording segments of one tape:
    with wp.Tape() as tape:
      mjw.step(m, d)
      mjw.forward(m, d)
    mjw.kinematics(m, d)
    mjw.camlight(m, d)
    mjw.refit_bvh(m, d, rc)
    with tape:
      mjw.fwd_kinematics(m, d)
      mjw.sensor_pos(m, d)
      mjw.sensor_vel(m, d)
      mjw.sensor_acc(m, d)
      mjw.render(m, d, rc)
    self.assertLen(tape.launches, 7)

    # Recorded functions compute the same results as untaped calls:
    for fn in (mjw.step, mjw.forward, mjw.fwd_kinematics, mjw.sensor_pos, mjw.sensor_vel, mjw.sensor_acc):
      fn(m, d_ref)
    for path in ("qpos", "qvel", "qacc", "site_xpos", "sensordata"):
      np.testing.assert_allclose(diff.get(d, path).numpy(), diff.get(d_ref, path).numpy(), atol=1e-6)
    self.assertTrue(np.all(np.isfinite(rc.depth_data.numpy())))
    if nworld == 2:
      self.assertFalse(np.allclose(d.site_xpos.numpy()[0], d.site_xpos.numpy()[1]))
      self.assertFalse(np.allclose(rc.depth_data.numpy()[0], rc.depth_data.numpy()[1]))
    self.assertTrue(d.efc.pos.requires_grad and rc.depth_data.requires_grad)
    self.assertFalse(m.opt.gravity.requires_grad)

    # Unregistered functions, and rules that read promoted fields they do not differentiate, raise:
    self._assert_not_supported("'energy_pos' is not supported", mjw.energy_pos, m, d)
    for name, in_f, missing in (("com_vel", (), "qvel"), ("fwd_velocity", ("qpos", "qvel"), "qacc")):
      diff.register_adjoint(name, lambda *args: None, in_fields=in_f, model_fields=("body_mass",))
      self._assert_not_supported(f"Data field '{missing}'", vars(mjw)[name], m, d)

    scales = np.arange(1, nworld + 1, dtype=np.float32).reshape(nworld, 1)
    d.qpos.grad.assign(scales)
    d.site_xpos.grad.assign(np.repeat(scales[:, None], 3, axis=-1))
    rc.depth_data.grad.fill_(1.0)
    tape.backward()
    self.assertEqual(depth_seeds, [rc.depth_data.size])
    # qpos.grad = 2 (scales + scales), qvel.grad = 0.5 (2 scales), body_mass.grad = 3 (2 scales):
    np.testing.assert_allclose(d.qpos.grad.numpy(), 4.0 * scales, atol=1e-6)
    np.testing.assert_allclose(d.qvel.grad.numpy(), 1.0 * scales, atol=1e-6)
    np.testing.assert_allclose(m.body_mass.grad.numpy()[0, 1], 6.0 * np.sum(scales), atol=1e-6)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.grad.numpy()[0], d.qpos.grad.numpy()[1]))

    # Untaped API calls after tape.backward() leave gradients intact:
    for fn in (mjw.kinematics, mjw.fwd_position, mjw.forward, mjw.step):
      fn(m, d)
    np.testing.assert_allclose(d.qpos.grad.numpy(), 4.0 * scales, atol=1e-6)
    np.testing.assert_allclose(d.qvel.grad.numpy(), 1.0 * scales, atol=1e-6)

    tape.zero()
    for arr in (d.qpos, d.qvel, m.body_mass):
      np.testing.assert_allclose(arr.grad.numpy(), 0.0, atol=1e-7)

  @parameterized.parameters(1, 2)
  def test_finite_difference_jacobian(self, nworld: int):
    # Gradients of loss = weights' * mul_m(step(qpos, qvel, body_mass)) match central differences:
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body>
            <joint axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.05"/>
            <body pos="0.3 0 0">
              <joint axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.05"/>
            </body>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    step = _register(_leaf_step, in_fields=("qpos", "qvel"), out_fields=("qpos",), model_fields=("body_mass",))

    # The inertia matrix is symmetric, so vec.grad += M res.grad:
    @diff.register_adjoint(mjw.mul_m, array_args=("res", "vec"), model_fields=("body_mass",))
    def mul_m_bwd(m_arg, d_arg, res_arg, vec_arg, skip=None, M=None):
      m_res_grad = wp.zeros(vec_arg.shape, dtype=float)
      mjw.mul_m(m_arg, d_arg, m_res_grad, res_arg.grad)
      wp.launch(_axpy_kernel, dim=vec_arg.shape, inputs=[1.0, m_res_grad], outputs=[vec_arg.grad])

    qpos0 = np.array([[0.3, -0.2], [-0.4, 0.25]][:nworld], dtype=np.float32)
    qvel0 = np.array([[-0.2, 0.1], [0.5, -0.3]][:nworld], dtype=np.float32)
    mass0 = m.body_mass.numpy()
    weights0 = np.arange(1, nworld + 1, dtype=np.float32)
    res = wp.zeros((nworld, m.nv), dtype=float)
    weights = wp.array(weights0, dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    m.body_mass.requires_grad = True

    def eval_loss():
      d.qpos.assign(qpos0)
      d.qvel.assign(qvel0)
      m.body_mass.assign(mass0)
      weights.assign(weights0)
      loss.zero_()
      step(m, d)
      mjw.mul_m(m, d, res, d.qpos)
      wp.launch(_weighted_loss_kernel, dim=res.shape, inputs=[res, weights], outputs=[loss])
      return loss.numpy()[0]

    with wp.Tape() as tape:
      eval_loss()
    tape.backward(loss=loss)
    eps = 1e-2
    for x0, grad, indices in (
      (qpos0, d.qpos.grad.numpy(), np.ndindex(qpos0.shape)),
      (qvel0, d.qvel.grad.numpy(), np.ndindex(qvel0.shape)),
      (mass0, m.body_mass.grad.numpy(), [(0, 1)]),
      (weights0, weights.grad.numpy(), np.ndindex(weights0.shape)),
    ):
      for i in indices:
        xi = x0[i]
        x0[i] = xi + eps
        loss_plus = eval_loss()
        x0[i] = xi - eps
        loss_minus = eval_loss()
        x0[i] = xi
        np.testing.assert_allclose(grad[i], (loss_plus - loss_minus) / (2.0 * eps), rtol=1e-2, atol=1e-4)

  @absltest.skipIf(not wp.get_device().is_cuda, "CUDA graph capture requires a CUDA device.")
  def test_graph_capture(self):
    # Native and custom rules can be captured in a CUDA graph with their backward, skipping
    # supported while capturing and allocating gradients on first promotion inside the capture:
    calls = []

    def native_fn(m, d):
      wp.launch(_scale_kernel, dim=d.qpos.shape, inputs=[d.qpos], outputs=[d.qacc])

    def custom_fn(m, d):
      wp.launch(_scale_kernel, dim=d.qacc.shape, inputs=[d.qacc], outputs=[d.qvel])

    native = _register(native_fn, in_fields=("qpos",), out_fields=("qacc",), supported=lambda *args: calls.append(1))
    custom = _register(
      custom_fn,
      lambda m, d: wp.launch(_axpy_kernel, dim=d.qacc.shape, inputs=[2.0, d.qvel.grad], outputs=[d.qacc.grad]),
      in_fields=("qacc",),
      out_fields=("qvel",),
      supported=lambda *args: calls.append(1),
    )
    with wp.Tape() as tape:
      native(self.m, self.d)
      custom(self.m, self.d)
    tape.backward()
    self.assertLen(calls, 2)

    d_new = mjw.make_data(self.mjm)
    d_new.qpos.fill_(0.25)
    d_new.qacc.fill_(wp.inf)
    d_new.qvel.fill_(wp.inf)
    with wp.ScopedCapture() as capture:
      with wp.Tape() as tape:
        native(self.m, d_new)
        custom(self.m, d_new)
      tape.zero()
      d_new.qvel.grad.fill_(1.5)
      tape.backward()

    self.assertLen(calls, 2)
    for _ in range(2):
      wp.capture_launch(capture.graph)
    np.testing.assert_allclose(d_new.qvel.numpy(), 1.0)
    np.testing.assert_allclose(d_new.qpos.grad.numpy(), 6.0)


if __name__ == "__main__":
  wp.init()
  absltest.main()
