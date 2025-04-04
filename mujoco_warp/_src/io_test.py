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
from . import test_util
import warp as wp
from absl.testing import absltest
from etils import epath

import mujoco_warp as mjwarp

# tolerance for difference between MuJoCo and MJWarp smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5

def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)

class IOTest(absltest.TestCase):
  def test_equality(self):
    mjm = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="body1">
          <geom type="sphere" size=".1"/>
          <freejoint/>
        </body>
        <body name="body2">
          <geom type="sphere" size=".1"/>
          <freejoint/>
        </body>
        <body>
          <site name="site1"/>
          <geom type="sphere" size=".1"/>
          <joint name="slide1" type="slide"/>
          <body>
            <site name="site2"/>
            <geom type="sphere" size=".1"/>
            <joint name="slide2" type="slide"/>
          </body>
        </body>
      </worldbody>  
      <tendon>
        <spatial name="tendon1">
          <site site="site1"/>
          <site site="site2"/>
        </spatial>
        <spatial name="tendon2">
          <site site="site1"/>
          <site site="site2"/>
        </spatial>
      </tendon>
      <equality>
        <connect body1="body1" body2="body2" anchor="0 0 0"/>
        <weld body1="body1" body2="body2"/> 
        <joint joint1="slide1" joint2="slide2"/>
        <tendon tendon1="tendon1" tendon2="tendon2"/>
      </equality>              
    </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

    # TODO(team): flex

  def test_sensor(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <geom type="sphere" size=".1"/>
            <joint name="slide" type="slide"/>
          </body>
        </worldbody>   
        <sensor>
          <jointpos joint="slide"/>                      
        </sensor> 
      </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_tendon(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>          
            <geom type="sphere" size=".1"/>
            <site name="site0"/>
            <joint name="slide" type="slide"/>
            <body pos="0 0 .1">
              <geom type="sphere" size=".1"/>
              <site name="site1"/>
            </body>
          </body>
        </worldbody>  
        <tendon>
          <spatial>
            <site site="site0"/>
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

  def test_dense(self):
    path = epath.resource_path("mujoco_warp") / "test_data/humanoid/n_humanoids.xml"
    mjm = mujoco.MjModel.from_xml_path(path.as_posix())

    with self.assertRaises(ValueError):
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

  def test_option_physical_constants(self):
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option wind="1 1 1" density="1" viscosity="1"/>
        <worldbody>
          <body>          
            <geom type="sphere" size=".1"/>
            <freejoint/>
          </body>
        </worldbody> 
    </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

  def test_model_batching(self):
    mjm, mjd, _, _ = test_util.fixture("humanoid/humanoid.xml")

    m = mjwarp.put_model(mjm, nworld=2, expand_fields={"dof_damping"})
    d = mjwarp.put_data(mjm, mjd, nworld=2)

    self.assertEqual(m.nworld, 2)

    # randomize dof_damping
    dof_damping = m.dof_damping.numpy()
    dof_damping[1, :] *= 0.5
    m.dof_damping = wp.from_numpy(dof_damping, dtype=wp.float32)

    mjwarp.passive(m, d)

    # mujoco reference, just have 2 separate model/data strctures
    mujoco.mj_passive(mjm, mjd)

    m2, d2, _, _ = test_util.fixture("humanoid/humanoid.xml")
    d2.qvel = mjd.qvel # need to copy qvel because of randomization
    m2.dof_damping *= 0.5

    mujoco.mj_passive(m2, d2)

    _assert_eq(d.qfrc_damper.numpy()[0, :], mjd.qfrc_damper, "qfrc_damper")
    _assert_eq(d.qfrc_damper.numpy()[1, :], d2.qfrc_damper, "qfrc_damper")


if __name__ == "__main__":
  wp.init()
  absltest.main()
