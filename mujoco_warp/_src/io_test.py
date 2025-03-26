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
      </worldbody>  
      <equality>
        <connect body1="body1" body2="body2" anchor="0 0 0"/>                      
      </equality>              
    </mujoco>
    """)

    with self.assertRaises(NotImplementedError):
      mjwarp.put_model(mjm)

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


if __name__ == "__main__":
  wp.init()
  absltest.main()
