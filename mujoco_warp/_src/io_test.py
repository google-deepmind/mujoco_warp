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

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp

from . import test_util


class IOTest(absltest.TestCase):
  def test_make_put_data(self):
    """Tests that make_data and put_data are producing the same shapes for all warp arrays."""
    mjm, _, _, d = test_util.fixture("pendula.xml")
    md = mjwarp.make_data(mjm, nconmax=512, njmax=512)

    # same number of fields
    self.assertEqual(len(d.__dict__), len(md.__dict__))

    # test shapes for all arrays
    for attr, val in md.__dict__.items():
      if isinstance(val, wp.array):
        self.assertEqual(val.shape, getattr(d, attr).shape, f"{attr} shape mismatch")

  # TODO(team): sensors

  def test_tendon(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>          
            <geom type="sphere" size=".1"/>
            <site name="site0"/>
            <joint name="slide" type="slide"/>
            <body pos="0 0 .1">
              <geom name="sphere1" type="sphere" size=".1"/>
              <site name="site1"/>
            </body>
          </body>
        </worldbody>  
        <tendon>
          <spatial>
            <site site="site0"/>
            <geom geom="sphere1"/>
            <site site="site1"/>
          </spatial>                      
        </tendon>              
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_geom_type(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <asset>
          <hfield name="hfield" nrow="1" ncol="1" size="1 1 1 1"/>
          <mesh name="mesh" vertex="1 0 0  0 1 0  0 0 1  1 1 0  1 0 1  0 1 1  1 1 1  0 0 0"/>
        </asset>
        <worldbody>
          <geom type="hfield" hfield="hfield"/>             
          <geom type="ellipsoid" size="1 1 1"/>
          <geom type="cylinder" size="1 1"/>
          <geom type="mesh" mesh="mesh"/>
        </worldbody>          
      </mujoco>
    """)

    # TODO(team): sdf

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_actuator_trntype(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body name="body">          
            <geom type="sphere" size=".1"/>
            <site name="site0"/>
            <joint type="slide"/>
          </body>
          <site name="site1"/>
        </worldbody>  
        <tendon>
          <spatial name="tendon">
            <site site="site0"/>
            <site site="site1"/>
          </spatial>                      
        </tendon>
        <actuator>
          <general cranksite="site0" slidersite="site1" cranklength=".1"/>
          <general tendon="tendon"/>
          <general site="site0" refsite="site1"/>
          <general body="body" ctrlrange="0 1"/>
        </actuator>           
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_actuator_dyntype(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>          
            <geom type="sphere" size=".1"/>
            <joint name="slide" type="slide"/>
          </body>
        </worldbody>  
        <actuator>
          <general joint="slide" dyntype="integrator"/>
          <general joint="slide" dyntype="filter"/>
          <general joint="slide" dyntype="muscle"/>
        </actuator>
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_actuator_gaintype(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <site name="siteworld"/>
          <body>          
            <geom type="sphere" size=".1"/>
            <site name="site0"/>
            <joint name="slide" type="slide"/>
          </body>
        </worldbody>  
        <tendon>
          <spatial name="tendon">
            <site site="siteworld"/>
            <site site="site0"/>
          </spatial>                      
        </tendon>
        <actuator>
          <muscle tendon="tendon" lengthrange="0 1"/>
        </actuator>
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_actuator_biastype(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <site name="siteworld"/>
          <body>          
            <geom type="sphere" size=".1"/>
            <site name="site0"/>
            <joint name="slide" type="slide"/>
          </body>
        </worldbody>  
        <tendon>
          <spatial name="tendon">
            <site site="siteworld"/>
            <site site="site0"/>
          </spatial>                      
        </tendon>
        <actuator>
          <muscle tendon="tendon" lengthrange="0 1"/>
        </actuator>
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

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


if __name__ == "__main__":
  wp.init()
  absltest.main()
