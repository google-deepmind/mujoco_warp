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

"""Tests for sensor functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp

from . import test_util

# tolerance for difference between MuJoCo and MJWarp calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SensorTest(parameterized.TestCase):
  def test_sensor(self):
    """Test sensors."""
    mjm, mjd, m, d = test_util.fixture(
      xml="""
      <mujoco>
        <option gravity="-1 -1 -1"/>
        <worldbody>
          <body name="body0" pos="0.1 0.2 0.3" quat=".05 .1 .15 .2">
            <joint name="slide" type="slide"/>
            <geom name="geom0" type="sphere" size="0.1"/>
            <site name="site0"/>
          </body>
          <body name="body1" pos=".5 .6 .7">
            <joint name="ballquat" type="ball"/>
            <geom type="sphere" size="0.1"/>
            <site name="site1" pos=".1 .2 .3"/>                         
          </body>
          <body name="body2" pos="1 1 1">
            <freejoint/>
            <geom type="sphere" size="0.1"/>
            <site name="site2"/>                         
          </body>
          <body name="body3" pos="2 2 2">
            <joint name="hinge0" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.1" pos=".1 0 0"/>
            <body pos="2 2 2">
              <joint name="hinge1" type="hinge" axis="1 0 0"/>
              <geom type="sphere" size="0.1" pos=".1 0 0"/>
            </body>
          </body>
          <body name="body4" pos="1 0 0">
            <joint type="ball"/>
            <geom type="sphere" size="0.1" pos=".1 0 0"/>
            <body>
              <joint type="ball"/>
              <geom type="sphere" size="0.1" pos=".1 0 0"/>
            </body>
          </body>
          <body pos="10 0 0">
            <joint type="hinge" axis="1 2 3"/>
            <geom type="sphere" size="0.1"/>
            <site name="force_site" pos="1 2 3"/>
          </body>
          <body pos="20 0 0">
            <joint type="slide" axis="1 2 3"/>
            <geom type="sphere" size="0.1"/>
            <site name="torque_site" pos="1 2 3"/>
          </body>
          <body name="body8">
            <joint type="hinge"/>
            <geom type="sphere" size="0.1" pos="1 2 3"/>
            <body name="body9">
              <joint type="hinge"/>
              <geom name="geom9" type="sphere" size="0.1" pos="1 2 3"/>
              <site name="site9" pos=".2 .4 .6"/>        
            </body>
          </body>
          <camera name="camera"/>
          <site name="camera_site" pos="0 0 -1"/>
        </worldbody>
        <actuator>
          <motor name="slide" joint="slide"/>
        </actuator>
        <sensor>
          <camprojection camera="camera" site="camera_site"/>
          <camprojection camera="camera" site="camera_site" cutoff=".001"/>
          <jointpos joint="slide"/>
          <jointpos joint="slide" cutoff=".001"/>
          <actuatorpos actuator="slide"/>
          <actuatorpos actuator="slide" cutoff=".001"/>
          <ballquat joint="ballquat"/>
          <framepos objtype="body" objname="body1"/>
          <framepos objtype="body" objname="body1" cutoff=".001"/>      
          <framepos objtype="body" objname="body1" reftype="body" refname="body0"/>    
          <framepos objtype="xbody" objname="body1"/> 
          <framepos objtype="geom" objname="geom0"/>    
          <framepos objtype="site" objname="site0"/>
          <framexaxis objtype="body" objname="body1"/>
          <framexaxis objtype="body" objname="body1" reftype="body" refname="body0"/>    
          <framexaxis objtype="xbody" objname="body1"/> 
          <framexaxis objtype="geom" objname="geom0"/>    
          <framexaxis objtype="site" objname="site0"/>
          <frameyaxis objtype="body" objname="body1"/> 
          <frameyaxis objtype="body" objname="body1" reftype="body" refname="body0"/>    
          <frameyaxis objtype="xbody" objname="body1"/> 
          <frameyaxis objtype="geom" objname="geom0"/>    
          <frameyaxis objtype="site" objname="site0"/> 
          <framezaxis objtype="body" objname="body1"/>  
          <framezaxis objtype="body" objname="body1" reftype="body" refname="body0"/>    
          <framezaxis objtype="xbody" objname="body1"/> 
          <framezaxis objtype="geom" objname="geom0"/>    
          <framezaxis objtype="site" objname="site0"/>  
          <framequat objtype="body" objname="body1"/>   
          <framequat objtype="body" objname="body1" reftype="body" refname="body0"/>    
          <framequat objtype="xbody" objname="body1"/> 
          <framequat objtype="geom" objname="geom0"/>    
          <framequat objtype="site" objname="site0"/>
          <subtreecom body="body3"/>
          <subtreecom body="body3" cutoff=".001"/>
          <clock/>
          <clock cutoff=".001"/>
          <velocimeter site="site2"/> 
          <velocimeter site="site2" cutoff=".001"/>                           
          <gyro site="site2"/> 
          <gyro site="site2" cutoff=".001"/>       
          <jointvel joint="slide"/>
          <jointvel joint="slide" cutoff=".001"/>
          <actuatorvel actuator="slide"/>
          <actuatorvel actuator="slide" cutoff=".001"/>
          <ballangvel joint="ballquat"/>
          <ballangvel joint="ballquat" cutoff=".001"/>
          <framelinvel objtype="body" objname="body9"/>
          <framelinvel objtype="body" objname="body9" cutoff=".001"/>
          <frameangvel objtype="body" objname="body9"/>
          <frameangvel objtype="body" objname="body9" cutoff=".001"/>
          <framelinvel objtype="xbody" objname="body9"/>
          <frameangvel objtype="xbody" objname="body9"/>
          <framelinvel objtype="geom" objname="geom9"/>
          <frameangvel objtype="geom" objname="geom9"/>
          <framelinvel objtype="site" objname="site9"/>
          <frameangvel objtype="site" objname="site9"/>
          <subtreelinvel body="body4"/>
          <subtreelinvel body="body4" cutoff=".001"/>
          <subtreeangmom body="body4"/>
          <subtreeangmom body="body4" cutoff=".001"/>
          <accelerometer site="force_site"/>
          <accelerometer site="force_site" cutoff=".001"/>
          <force site="force_site"/>
          <force site="force_site" cutoff=".001"/>
          <torque site="torque_site"/>
          <torque site="torque_site" cutoff=".001"/>
          <actuatorfrc actuator="slide"/>
          <actuatorfrc actuator="slide" cutoff=".001"/>
          <jointactuatorfrc joint="slide"/>
          <jointactuatorfrc joint="slide" cutoff=".001"/>            
          <framelinacc objtype="body" objname="body9"/>
          <framelinacc objtype="body" objname="body9" cutoff=".001"/>
          <frameangacc objtype="body" objname="body9"/>
          <frameangacc objtype="body" objname="body9" cutoff=".001"/>
          <framelinacc objtype="xbody" objname="body9"/>
          <frameangacc objtype="xbody" objname="body9"/>
          <framelinacc objtype="geom" objname="geom9"/>
          <frameangacc objtype="geom" objname="geom9"/>
          <framelinacc objtype="site" objname="site9"/>
          <frameangacc objtype="site" objname="site9"/>
        </sensor>
        <keyframe>
          <key qpos="1 .1 .2 .3 .4 1 1 1 1 0 0 0 .25 .35 1 0 0 0 1 0 0 0 0 0 1 1" qvel="2 .2 -.1 .4 .25 .35 .45 -0.1 -0.2 -0.3 .1 -.2 -.5 -0.75 -1 .1 .2 .3 0 0 2 2" ctrl="3"/>
        </keyframe>
      </mujoco>
    """,
      keyframe=0,
      kick=True,
    )

    d.sensordata.zero_()

    mjwarp.sensor_pos(m, d)
    mjwarp.sensor_vel(m, d)
    mjwarp.sensor_acc(m, d)

    _assert_eq(d.sensordata.numpy()[0], mjd.sensordata, "sensordata")

  def test_tendon_sensor(self):
    """Test tendon sensors."""
    _, mjd, m, d = test_util.fixture("tendon/fixed.xml", keyframe=0, sparse=False)

    d.sensordata.zero_()

    mjwarp.sensor_pos(m, d)
    mjwarp.sensor_vel(m, d)

    _assert_eq(d.sensordata.numpy()[0], mjd.sensordata, "sensordata")


if __name__ == "__main__":
  wp.init()
  absltest.main()
