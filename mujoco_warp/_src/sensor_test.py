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

wp.config.verify_cuda = True

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
    mjm = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option gravity="1 2 3"/>
        <worldbody>
          <body name="body0">
            <joint name="slide" type="slide"/>
            <geom type="sphere" size="0.1"/>
          </body>
          <body name="body1">
            <joint type="hinge"/>
            <geom type="sphere" size="0.1" pos="1 2 3"/>
            <body name="body2">
              <joint type="hinge"/>
              <geom name="geom2" type="sphere" size="0.1" pos="1 2 3"/>
              <site name="site2" pos=".2 .4 .6"/>        
            </body>
          </body>
        </worldbody>
        <actuator>
          <motor name="slide" joint="slide"/>
        </actuator>
        <sensor>
          <jointpos joint="slide"/>
          <jointvel joint="slide"/>
          <actuatorfrc actuator="slide"/>
          <framelinacc objtype="body" objname="body2"/>
          <frameangacc objtype="body" objname="body2"/>
          <framelinacc objtype="xbody" objname="body2"/>
          <frameangacc objtype="xbody" objname="body2"/>
          <framelinacc objtype="geom" objname="geom2"/>
          <frameangacc objtype="geom" objname="geom2"/>
          <framelinacc objtype="site" objname="site2"/>
          <frameangacc objtype="site" objname="site2"/>
        </sensor>
        <keyframe>
          <key qpos="1 1 1" qvel="2 2 2" ctrl="3"/>
        </keyframe>
      </mujoco>
    """)

    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    d.sensordata.zero_()

    mjwarp.sensor_pos(m, d)
    mjwarp.sensor_vel(m, d)
    mjwarp.sensor_acc(m, d)

    _assert_eq(d.sensordata.numpy()[0], mjd.sensordata, "sensordata")


if __name__ == "__main__":
  wp.init()
  absltest.main()
