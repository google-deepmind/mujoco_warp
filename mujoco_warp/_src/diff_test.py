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

import dataclasses
import inspect

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized
from warp._src import context as wp_context

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import diff

_ORIG_PARAMS = dict(diff._PARAMS)


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
  site_xpos_in: wp.array2d[wp.vec3],
  # In:
  weights: wp.array[float],
  # Out:
  loss_out: wp.array[float],
):
  w = wp.tid()
  s = site_xpos_in[w, 0]
  wp.atomic_add(loss_out, 0, weights[w] * (qpos_in[w, 0] + s[0] + s[1] + s[2]))


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


@wp.kernel(module="unique")
def _mul_m_bwd_kernel(
  # In:
  scale: float,
  res_grad: wp.array2d[float],
  # Out:
  vec_grad_out: wp.array2d[float],
):
  w, i = wp.tid()
  vec_grad_out[w, i] += scale * res_grad[w, i]


class DiffApiTest(parameterized.TestCase):
  """Tests for physics API coverage, field validation, CUDA graph capture, and custom adjoints."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    _, _, cls.m, cls.d = test_data.fixture(
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

  def tearDown(self):
    for k in diff.RULES:
      diff.RULES[k] = None
    diff._PARAMS.update(_ORIG_PARAMS)
    for s in diff._PROMOTED.values():
      s.clear()
    for obj in (self.m, self.d):
      for path in diff._FIELDS[type(obj)]:
        val = diff.get(obj, path)
        if isinstance(val, wp.array) and val.requires_grad:
          val.requires_grad = False
    super().tearDown()

  def _assert_field_raises(self, obj, path: str):
    val = diff.get(obj, path)
    self.assertIsInstance(val, wp.array)
    val.requires_grad = True
    diff.register_adjoint("step", lambda *args, **kwargs: None)
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, f"field '{path}'"):
      mjw.step(self.m, self.d)

  def test_physics_functions_wrapped(self):
    public = {name for name, obj in vars(mjw).items() if inspect.isfunction(obj) and not name.startswith("_")}
    self.assertSetEqual(public, set(diff.RULES))
    for name in public:
      self.assertIsNone(diff.RULES[name])
    with self.assertRaises(ValueError):
      diff.register_adjoint("unknown_fn", lambda: None)
    with self.assertRaises(ValueError):
      diff.register_adjoint("step", lambda: None, in_fields="qpos")
    with self.assertRaises(ValueError):
      diff.register_adjoint("step", lambda: None, in_fields=("invalid_field",))
    with self.assertRaises(ValueError):
      diff.register_adjoint("step", lambda: None, model_fields=("invalid_model_field",))
    with self.assertRaises(ValueError):
      diff.register_adjoint("mul_m", lambda: None, array_args="res")
    with self.assertRaises(ValueError):
      diff.register_adjoint("mul_m", lambda: None, array_args=("invalid_arg",))

    # _fields supports runtime wp.array and doc/mjwarp/update_types.py wp.array[...] annotations:
    @dataclasses.dataclass
    class _DocStruct:
      a: wp.array[float]
      b: wp.array2d[float]

    self.assertEqual(diff._fields(_DocStruct), ("a", "b"))

    # _Untaped restores the active tape when the call raises:
    with wp.Tape() as tape:
      with self.assertRaises(RuntimeError):
        diff._call_untaped(lambda: (_ for _ in ()).throw(RuntimeError("untaped error")))
      self.assertIs(wp_context.runtime.tape, tape)

  @parameterized.parameters(*sorted(_ORIG_PARAMS))
  def test_api_function_raises_not_implemented(self, fn_name: str):
    # Unregistered functions raise under a tape before reading their arguments:
    with wp.Tape() as tape, self.assertRaisesRegex(NotImplementedError, f"'{fn_name}'"):
      vars(mjw)[fn_name]()
    self.assertEmpty(tape.launches)

  @parameterized.parameters(*diff._FIELDS[mjw.Model])
  def test_model_array_requires_grad_raises_not_implemented(self, path: str):
    self._assert_field_raises(self.m, path)

  @parameterized.parameters(*diff._FIELDS[mjw.Data])
  def test_data_array_requires_grad_raises_not_implemented(self, path: str):
    self._assert_field_raises(self.d, path)

  @parameterized.parameters(1, 2)
  def test_custom_adjoint_and_multi_function_composition(self, nworld: int):
    _, _, m, d = test_data.fixture(
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
      """,
      nworld=nworld,
    )
    d.qpos.assign(wp.array(np.array([[0.3], [-0.4]][:nworld], dtype=np.float32)))
    d.qvel.assign(wp.array(np.array([[-0.2], [0.5]][:nworld], dtype=np.float32)))
    d.site_xpos.fill_(wp.inf)

    @diff.register_adjoint(
      "step",
      in_fields=("qpos", "qvel"),
      out_fields=("qpos", "qvel", "efc.pos"),
      model_fields=("body_mass", "opt.gravity"),
    )
    def _step_bwd(m_arg, d_arg):
      adj_out = wp.clone(d_arg.qpos.grad)
      d_arg.qpos.grad.zero_()
      wp.launch(
        _linear_step_leaf,
        dim=(nworld, 1),
        inputs=[m_arg.body_mass, d_arg.qpos, d_arg.qvel],
        outputs=[d_arg.qpos],
        adj_inputs=[m_arg.body_mass.grad, d_arg.qpos.grad, d_arg.qvel.grad],
        adj_outputs=[adj_out],
        adjoint=True,
      )

    @diff.register_adjoint(
      "fwd_kinematics",
      in_fields=("qpos",),
      out_fields=("site_xpos",),
      model_fields=("body_mass",),
    )
    def _fk_bwd(m_arg, d_arg):
      wp.launch(_fk_bwd_kernel, dim=nworld, inputs=[d_arg.site_xpos.grad], outputs=[d_arg.qpos.grad])

    m.body_mass.requires_grad = True

    # Unsupported fields (including fields supported only by another rule) raise:
    m.geom_rbound.requires_grad = True
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "field 'geom_rbound'"):
      mjw.step(m, d)
    m.geom_rbound.requires_grad = False

    m.opt.gravity.requires_grad = True
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "field 'opt.gravity'"):
      mjw.fwd_kinematics(m, d)
    m.opt.gravity.requires_grad = False

    d.qvel.requires_grad = True
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "field 'qvel'"):
      mjw.fwd_kinematics(m, d)
    d.qvel.requires_grad = False

    with wp.Tape() as tape:
      mjw.step(m, d)
      mjw.fwd_kinematics(m, d)

    self.assertTrue(d.efc.pos.requires_grad)
    self.assertFalse(m.opt.gravity.requires_grad)
    self.assertTrue(np.all(np.isfinite(d.site_xpos.numpy())))
    if nworld == 2:
      self.assertFalse(np.allclose(d.site_xpos.numpy()[0], d.site_xpos.numpy()[1]))

    # Calling an unregistered function on the tape reports the function name:
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "'sensor_pos'"):
      mjw.sensor_pos(m, d)

    scales = np.arange(1, nworld + 1, dtype=np.float32).reshape(nworld, 1)
    qpos_seed = wp.array(scales)
    site_seed = wp.array(np.repeat(scales[:, None], 3, axis=-1), dtype=wp.vec3)
    d.qpos.grad.assign(qpos_seed)
    d.site_xpos.grad.assign(site_seed)
    tape.backward()

    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.grad.numpy()[0], d.qpos.grad.numpy()[1]))
    np.testing.assert_allclose(d.qpos.grad.numpy(), 4.0 * scales, atol=1e-6)
    np.testing.assert_allclose(d.qvel.grad.numpy(), 1.0 * scales, atol=1e-6)
    np.testing.assert_allclose(m.body_mass.grad.numpy()[0, 1], 6.0 * np.sum(scales), atol=1e-6)

    tape.zero()
    for arr in (d.qpos, d.qvel, m.body_mass):
      np.testing.assert_allclose(arr.grad.numpy(), 0.0, atol=1e-7)

    # Verify CUDA graph capture support for taped forward + backward execution:
    if wp.get_device().is_cuda:
      with wp.ScopedCapture() as capture:
        tape.zero()
        with wp.Tape() as captured_tape:
          mjw.step(m, d)
          mjw.fwd_kinematics(m, d)
        d.qpos.grad.assign(qpos_seed)
        d.site_xpos.grad.assign(site_seed)
        captured_tape.backward()

      for _ in range(3):
        wp.capture_launch(capture.graph)
      np.testing.assert_allclose(d.qpos.grad.numpy(), 4.0 * scales, atol=1e-6)
      np.testing.assert_allclose(d.qvel.grad.numpy(), 1.0 * scales, atol=1e-6)
      np.testing.assert_allclose(m.body_mass.grad.numpy()[0, 1], 6.0 * np.sum(scales), atol=1e-6)

  @parameterized.parameters(1, 2)
  def test_native_enable_backward_and_out_of_place_d_out(self, nworld: int):
    mjm, _, m, d0 = test_data.fixture(
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
      """,
      nworld=nworld,
    )
    d1 = mjw.make_data(mjm, nworld=nworld)
    d0.qpos.assign(wp.array(np.array([[0.3], [-0.4]][:nworld], dtype=np.float32)))
    d0.qvel.assign(wp.array(np.array([[-0.2], [0.5]][:nworld], dtype=np.float32)))
    d1.qpos.fill_(wp.inf)
    d1.site_xpos.fill_(wp.inf)

    def step(m, d, d_out=None):
      target = d if d_out is None else d_out
      wp.launch(
        _linear_step_leaf,
        dim=(nworld, 1),
        inputs=[m.body_mass, d.qpos, d.qvel],
        outputs=[target.qpos],
      )

    wrapped_step = diff._wrap(step)
    diff.register_adjoint("step", in_fields=("qpos", "qvel"), out_fields=("qpos",), model_fields=("body_mass",))

    @diff.register_adjoint(
      "fwd_kinematics",
      in_fields=("qpos",),
      out_fields=("site_xpos",),
      model_fields=("body_mass",),
    )
    def _fk_bwd(m_arg, d_arg):
      wp.launch(_fk_bwd_kernel, dim=nworld, inputs=[d_arg.site_xpos.grad], outputs=[d_arg.qpos.grad])

    # Unsupported field on d_out is rejected by _validate:
    d1.qacc.requires_grad = True
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "field 'qacc'"):
      wrapped_step(m, d0, d_out=d1)
    d1.qacc.requires_grad = False

    m.body_mass.requires_grad = True
    scales = np.arange(1, nworld + 1, dtype=np.float32)
    weights = wp.array(scales, dtype=wp.float32)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    # Containers are resolved by type in signature order, independent of keyword order:
    with wp.Tape() as tape:
      wrapped_step(d_out=d1, d=d0, m=m)
      mjw.fwd_kinematics(m, d1)
      wp.launch(_weighted_loss_kernel, dim=nworld, inputs=[d1.qpos, d1.site_xpos, weights], outputs=[loss])

    self.assertTrue(d0.qpos.requires_grad)
    self.assertTrue(d0.qvel.requires_grad)
    self.assertTrue(d1.qpos.requires_grad)
    self.assertFalse(d1.qvel.requires_grad)
    self.assertTrue(np.all(np.isfinite(d1.qpos.numpy())))
    self.assertTrue(np.all(np.isfinite(d1.site_xpos.numpy())))
    if nworld == 2:
      self.assertFalse(np.allclose(d1.qpos.numpy()[0], d1.qpos.numpy()[1]))
      self.assertFalse(np.allclose(d1.site_xpos.numpy()[0], d1.site_xpos.numpy()[1]))

    tape.backward(loss=loss)
    scales_2d = scales.reshape(nworld, 1)
    if nworld == 2:
      self.assertFalse(np.allclose(d0.qpos.grad.numpy()[0], d0.qpos.grad.numpy()[1]))
    np.testing.assert_allclose(d0.qpos.grad.numpy(), 4.0 * scales_2d, atol=1e-6)
    np.testing.assert_allclose(d0.qvel.grad.numpy(), 1.0 * scales_2d, atol=1e-6)
    np.testing.assert_allclose(m.body_mass.grad.numpy()[0, 1], 6.0 * np.sum(scales), atol=1e-6)

  @parameterized.parameters(1, 2)
  def test_custom_adjoint_supported_and_array_args(self, nworld: int):
    _, _, m, d = test_data.fixture(
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
    diff.register_adjoint(
      mjw.step,
      backward=lambda *args, **kwargs: None,
      in_fields=("qpos", "qvel"),
      supported=lambda m_arg, d_arg, d_out=None: ("d_out is required",) if d_out is None else (),
    )
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "d_out is required"):
      mjw.step(m, d)

    res = wp.full((nworld, m.nv), wp.inf, dtype=wp.float32, requires_grad=True)
    vec = wp.array(np.arange(1, nworld + 1, dtype=np.float32).reshape(nworld, 1))
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "'mul_m' is not supported"):
      mjw.mul_m(m, d, res, vec)
    res.requires_grad = False

    diff.register_adjoint(
      mjw.mul_m,
      backward=lambda m_arg, d_arg, res, vec, skip=None, M=None: wp.launch(
        _mul_m_bwd_kernel, dim=(nworld, m_arg.nv), inputs=[2.5, res.grad], outputs=[vec.grad]
      ),
      array_args=("res", "vec"),
    )

    # Passing a requires_grad=True array for an argument not in array_args raises:
    extra_m = wp.zeros((nworld, 1, 1), dtype=wp.float32, requires_grad=True)
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "with array argument is not supported"):
      mjw.mul_m(m, d, res, vec, M=extra_m)

    with wp.Tape() as tape:
      mjw.mul_m(m, d, res, vec)
    self.assertTrue(np.all(np.isfinite(res.numpy())))
    if nworld == 2:
      self.assertFalse(np.allclose(res.numpy()[0], res.numpy()[1]))
    self.assertTrue(res.requires_grad)
    self.assertTrue(vec.requires_grad)

    scales = np.arange(1, nworld + 1, dtype=np.float32).reshape(nworld, 1)
    res.grad.assign(wp.array(2.0 * scales))
    tape.backward()
    if nworld == 2:
      self.assertFalse(np.allclose(vec.grad.numpy()[0], vec.grad.numpy()[1]))
    np.testing.assert_allclose(vec.grad.numpy(), 5.0 * scales, atol=1e-6)
    tape.zero()
    np.testing.assert_allclose(vec.grad.numpy(), 0.0, atol=1e-7)

    # Re-registering a rule resets its previously auto-promoted arrays and enforces the new rule:
    diff.register_adjoint(mjw.mul_m, backward=lambda *args, **kwargs: None, array_args=("res",))
    self.assertFalse(vec.requires_grad)
    with wp.Tape():
      mjw.mul_m(m, d, res, vec)
    self.assertTrue(res.requires_grad)
    self.assertFalse(vec.requires_grad)
    vec.requires_grad = True
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "with array argument is not supported"):
      mjw.mul_m(m, d, res, vec)

  @parameterized.parameters(1, 2)
  def test_render_context_fields(self, nworld: int):
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <camera pos="0 -2 0.5" xyaxes="1 0 0 0 0 1"/>
          <geom type="plane" size="3 3 0.1"/>
          <body pos="0 0 0.5">
            <joint type="slide"/>
            <geom size="0.3"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    d.qpos.assign(wp.array(np.array([[0.0], [0.5]][:nworld], dtype=np.float32)))
    mjw.kinematics(m, d)
    mjw.camlight(m, d)
    rc = mjw.create_render_context(mjm, nworld=nworld, cam_res=(8, 8), render_rgb=False, render_depth=True)
    mjw.refit_bvh(m, d, rc)
    with self.assertRaises(ValueError):
      diff.register_adjoint("render", lambda *args: None, render_fields=("invalid_field",))

    seeds = []
    diff.register_adjoint(
      "render",
      lambda m_arg, d_arg, rc_arg: seeds.append(rc_arg.depth_data.grad.numpy().sum()),
      render_fields=("depth_data",),
    )

    # Unsupported RenderContext fields raise for registered functions:
    rc.ray.requires_grad = True
    with wp.Tape(), self.assertRaisesRegex(NotImplementedError, "RenderContext field 'ray'"):
      mjw.render(m, d, rc)
    rc.ray.requires_grad = False
    rc.depth_data.fill_(wp.inf)
    with wp.Tape() as tape:
      mjw.render(m, d, rc)
    self.assertTrue(rc.depth_data.requires_grad)
    self.assertTrue(np.all(np.isfinite(rc.depth_data.numpy())))
    if nworld == 2:
      self.assertFalse(np.allclose(rc.depth_data.numpy()[0], rc.depth_data.numpy()[1]))

    rc.depth_data.grad.fill_(1.0)
    tape.backward()
    self.assertEqual(seeds, [rc.depth_data.size])


if __name__ == "__main__":
  wp.init()
  absltest.main()
