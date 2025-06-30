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

"""Tests for io functions."""

import dataclasses
import typing
from typing import Any, Dict, Optional, Union

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp

from . import test_util


class IOTest(absltest.TestCase):
  def test_make_put_data(self):
    """Tests that make_data and put_data are producing the same shapes for all arrays."""
    mjm, _, _, d = test_util.fixture("pendula.xml")
    md = mjwarp.make_data(mjm, nconmax=512, njmax=512)

    # same number of fields
    self.assertEqual(len(d.__dict__), len(md.__dict__))

    # test shapes for all arrays
    for attr, val in md.__dict__.items():
      if isinstance(val, wp.array):
        self.assertEqual(val.shape, getattr(d, attr).shape, f"{attr} shape mismatch")

  # TODO(team): sensors

  def test_get_data_into_m(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body pos="0 0 0" >
            <geom type="box" pos="0 0 0" size=".5 .5 .5" />
            <joint type="hinge" />
          </body>
          <body pos="0 0 0.1">
            <geom type="sphere" size="0.5"/>
            <freejoint/>
          </body>
        </worldbody>
      </mujoco>
    """)

    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    mjd_ref = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd_ref)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    mjd.qLD.fill(-123)
    mjd.qM.fill(-123)

    mjwarp.get_data_into(mjd, mjm, d)
    np.testing.assert_allclose(mjd.qLD, mjd_ref.qLD)
    np.testing.assert_allclose(mjd.qM, mjd_ref.qM)

  def test_ellipsoid_fluid_model(self):
    with self.assertRaises(NotImplementedError):
      mjm = mujoco.MjModel.from_xml_string(
        """
      <mujoco>
        <option density="1"/>
        <worldbody>
          <body>
            <geom type="sphere" size=".1" fluidshape="ellipsoid"/>
            <freejoint/>
          </body>
        </worldbody>
      </mujoco>
      """
      )
      mjwarp.put_model(mjm)

  def test_jacobian_auto(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option jacobian="auto"/>
        <worldbody>
          <replicate count="11">
          <body>
            <geom type="sphere" size=".1"/>
            <freejoint/>
            </body>
          </replicate>
        </worldbody>
      </mujoco>
    """)
    mjwarp.put_model(mjm)

  def test_put_data_qLD(self):
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body>
          <geom type="sphere" size="1"/>
          <joint type="hinge"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    mjd = mujoco.MjData(mjm)
    d = mjwarp.put_data(mjm, mjd)
    self.assertTrue((d.qLD.numpy() == 0.0).all())

    mujoco.mj_forward(mjm, mjd)
    mjd.qM[:] = 0.0
    d = mjwarp.put_data(mjm, mjd)
    self.assertTrue((d.qLD.numpy() == 0.0).all())

    mujoco.mj_forward(mjm, mjd)
    mjd.qLD[:] = 0.0
    d = mjwarp.put_data(mjm, mjd)
    self.assertTrue((d.qLD.numpy() == 0.0).all())

  def test_implicitfast_sparse(self):
    with self.assertRaises(NotImplementedError):
      test_util.fixture(
        xml="""
      <mujoco>
        <option integrator="implicitfast" jacobian="sparse"/>
      </mujoco>
      """
      )

  def test_noslip_solver(self):
    with self.assertRaises(NotImplementedError):
      test_util.fixture(
        xml="""
      <mujoco>
        <option noslip_iterations="1"/>
      </mujoco>
      """
      )

  def test_public_api_jax_compat(self):
    """Tests that annotations meet a set of criteria for JAX compat."""
    _check_annotation_compat(mjwarp.Model.__annotations__, "Model.")
    _check_annotation_compat(mjwarp.Data.__annotations__, "Data.")

  def test_types_match_annotations(self):
    """Tests that the types of dataclass fields match the annotations."""
    mjm, _, m, d = test_util.fixture("pendula.xml")

    _check_type_matches_annotation(self, m, "Model.")
    _check_type_matches_annotation(self, d, "Data.")

    d = mjwarp.make_data(mjm, nworld=2)
    _check_type_matches_annotation(self, d, "Data.")


def _get_np_scalar_type(val: Any) -> Optional[Union[bool, int, float]]:
  is_np_scalar = list(isinstance(val, t) for t in (np.integer, np.floating, np.bool_))
  if any(is_np_scalar):
    return [int, float, bool][is_np_scalar.index(True)]


def _check_type_matches_annotation(test_obj, obj: Any, prefix: str = ""):
  assert dataclasses.is_dataclass(obj), prefix + " must be dataclass."
  msg = "Type of {val_type} does not match annotation {type_} for field {prefix}{field_name}"

  for field in dataclasses.fields(obj):
    field_name = field.name
    val = getattr(obj, field_name)
    val_type = type(val)
    type_ = field.type

    if dataclasses.is_dataclass(val):
      test_obj.assertTrue(dataclasses.is_dataclass(type_), msg.format(**locals()))
      _check_type_matches_annotation(test_obj, val, prefix + field_name + ".")
      continue

    np_scalar_type = _get_np_scalar_type(val)
    if np_scalar_type:
      test_obj.assertIsInstance(np_scalar_type(val), type_, msg.format(**locals()))
      continue

    if isinstance(type_, wp.types.array):
      test_obj.assertIsInstance(val, wp.types.array, msg.format(**locals()))
      continue

    origin_type = typing.get_origin(type_)
    if tuple in (val_type, origin_type):
      test_obj.assertEqual(val_type, origin_type, msg.format(**locals()))
      field_name += ".tuple[]"
      type_ = typing.get_args(type_)[0]

      items = val
      for val in items:
        val_type = type(val)
        if dataclasses.is_dataclass(val):
          _check_type_matches_annotation(test_obj, val, prefix + field_name)
          continue

        np_scalar_type = _get_np_scalar_type(val)
        if np_scalar_type:
          test_obj.assertIsInstance(np_scalar_type(val), type_, msg.format(**locals()))
          continue

        if isinstance(type_, wp.types.array):
          test_obj.assertIsInstance(val, wp.types.array, msg.format(**locals()))
          continue

        test_obj.assertEqual(type(val), type_, msg.format(**locals()))
      continue

    test_obj.assertEqual(type(val), field.type, msg.format(**locals()))


def _check_annotation_compat(
  annotations: Dict[str, Any], prefix: str = "", in_cls: bool = False, in_tuple: bool = False
) -> Dict[str, Any]:
  for k, v in annotations.items():
    full_key = f"{prefix}{k}"
    info = f"Found {v} for annotation {full_key}."

    if v in (int, bool, float):
      continue

    if isinstance(v, wp.types.array):
      continue

    if v in wp.types.vector_types:
      raise AssertionError(f"Vector types are not allowed. {info}")

    if typing.get_origin(v) == tuple and (in_cls or in_tuple):
      raise AssertionError(f"Nested args in Model/Data must not be tuple. {info}")

    if typing.get_origin(v) == tuple:
      tuple_args = typing.get_args(v)
      if len(tuple_args) != 2 and tuple_args[1] != ...:
        raise AssertionError(f"Tuple args must be variadic. {info}")

      _check_annotation_compat(
        {"[]": tuple_args[0]},
        prefix=f"{full_key}.tuple",
        in_cls=in_cls,
        in_tuple=True,
      )
      continue

    if hasattr(v, "__class__") and in_cls:
      raise AssertionError(f"Nested object args in Model/Data are not allowed. {info}")

    if hasattr(v, "__class__") and not dataclasses.is_dataclass(v):
      raise AssertionError(f"Args that are objects must be dataclass. {info}")

    if hasattr(v, "__class__") and not v.__module__.startswith("mujoco_warp"):
      raise AssertionError(f"dataclass args must be within the mujoco_warp module. {info}")

    if hasattr(v, "__class__"):
      _check_annotation_compat(v.__annotations__, prefix=f"{full_key}{v.__name__}.", in_cls=True, in_tuple=in_tuple)
      continue

    raise AssertionError(f"Model/Data annotation is not allowed. {info}")


if __name__ == "__main__":
  wp.init()
  absltest.main()
