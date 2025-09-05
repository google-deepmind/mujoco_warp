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
from absl.testing import parameterized

import mujoco_warp as mjwarp

from . import test_util
from .io import MAX_WORLDS

# NOTE: modify io_jax_test _IO_TEST_MODELS if changed here.
_IO_TEST_MODELS = (
  "pendula.xml",
  "collision_sdf/tactile.xml",
  "flex/floppy.xml",
  "actuation/tendon_force_limit.xml",
  "hfield/hfield.xml",
)


class IOTest(parameterized.TestCase):
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

  def test_noslip_solver(self):
    with self.assertRaises(NotImplementedError):
      test_util.fixture(
        xml="""
      <mujoco>
        <option noslip_iterations="1"/>
      </mujoco>
      """
      )

  def test_sdf(self):
    """Tests that an SDF can be loaded."""
    mjm, mjd, m, d = test_util.fixture(fname="collision_sdf/cow.xml", qpos0=True)

    self.assertIsInstance(m.oct_aabb, wp.array)
    self.assertEqual(m.oct_aabb.dtype, wp.vec3)
    self.assertEqual(len(m.oct_aabb.shape), 2)
    if m.oct_aabb.size > 0:
      self.assertEqual(m.oct_aabb.shape[1], 2)


if __name__ == "__main__":
  wp.init()
  absltest.main()
