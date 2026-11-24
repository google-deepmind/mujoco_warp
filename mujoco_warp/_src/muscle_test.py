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

"""Tests for compliant-tendon muscle actuators: Millard2012 and Geyer2010."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data

# a point mass hanging from a muscle; the anchor is at z = 0.3 and the insertion at z = qpos
_HANGING = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <site name="anchor" pos="0 0 0.3"/>
    <body>
      <joint name="slide" type="slide" axis="0 0 1" damping="1"/>
      <site name="insertion"/>
      <geom type="sphere" size="0.01" mass="20"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="path">
      <site site="anchor"/>
      <site site="insertion"/>
    </spatial>
  </tendon>
  <actuator>
    {muscle}
  </actuator>
  <keyframe>
    <key qpos="-0.02" act="0.05 0.3" ctrl="0.6"/>
  </keyframe>
</mujoco>
"""

_MUSCLES = {
  "damped": '<millard2012 tendon="path" pennation="0.2"/>',
  "undamped": '<millard2012 tendon="path" pennation="0.2" fiberdamping="0"/>',
  "rigid": '<millard2012 tendon="path" pennation="0.2" rigidtendon="true"/>',
  "retuned": """<millard2012 tendon="path" force="947.75" optlength="0.167" slacklength="0.103"
                  pennation="0.3824" activecurve="0.25 0.77 1.9 0.75"
                  passivecurve="0.0398 0.7005 0.2 3.027"/>""",
  "geyer": '<geyer2010 tendon="path"/>',
  "geyer_tuned": """<geyer2010 tendon="path" force="950" optlength="0.18" slacklength="0.13" vmax="10.36"
                  activecurve="0.96 0.05" passivecurve="1.17 0.47" tendoncurve="0.0325" velocitycurve="3.33"
                  fiberdamping="0.1" tausmooth="0.1"/>""",
  "geyer_rigid": '<geyer2010 tendon="path" fiberdamping="0.1" rigidtendon="true"/>',
}


class MuscleTest(parameterized.TestCase):
  @parameterized.parameters(*_MUSCLES.keys())
  def test_actuation(self, muscle):
    """act_dot and actuator_force match MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(xml=_HANGING.format(muscle=_MUSCLES[muscle]), keyframe=0)
    for arr in (d.act_dot, d.actuator_force):
      arr.fill_(wp.inf)
    mjw.forward(m, d)

    np.testing.assert_allclose(d.act_dot.numpy()[0], mjd.act_dot, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(d.actuator_force.numpy()[0], mjd.actuator_force, rtol=1e-4, atol=1e-3)

  @parameterized.parameters(*_MUSCLES.keys())
  def test_trajectory(self, muscle):
    """A short rollout under a varying excitation matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(xml=_HANGING.format(muscle=_MUSCLES[muscle]), keyframe=0)
    for i in range(200):
      u = 0.5 + 0.4 * np.sin(0.05 * i)
      mjd.ctrl[0] = u
      d.ctrl.fill_(float(u))
      mujoco.mj_step(mjm, mjd)
      mjw.step(m, d)
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=1e-5)
    np.testing.assert_allclose(d.act.numpy()[0], mjd.act, atol=1e-4)
    np.testing.assert_allclose(d.actuator_force.numpy()[0], mjd.actuator_force, rtol=1e-3)

  @parameterized.parameters("damped", "retuned", "geyer", "geyer_tuned")
  def test_equilibrium(self, muscle):
    """muscle_equilibrium matches mj_muscleEquilibrium."""
    mjm, mjd, m, d = test_data.fixture(xml=_HANGING.format(muscle=_MUSCLES[muscle]), keyframe=0)
    mjw.forward(m, d)
    mjw.muscle_equilibrium(m, d)
    mujoco.mj_muscleEquilibrium(mjm, mjd)
    np.testing.assert_allclose(d.act.numpy()[0], mjd.act, atol=1e-5)
    np.testing.assert_allclose(d.act_dot.numpy()[0, 0], 0.0)

  @parameterized.parameters("rigid", "geyer_rigid")
  def test_rigid_short_path(self, muscle):
    """A rigid tendon on a path shorter than the tendon carries no force."""
    mjm, mjd, m, d = test_data.fixture(xml=_HANGING.format(muscle=_MUSCLES[muscle]), keyframe=0)
    d.qpos.fill_(0.3 - 0.15)
    mjw.forward(m, d)
    np.testing.assert_equal(d.actuator_force.numpy()[0, 0], 0.0)

  @parameterized.parameters("damped", "rigid", "geyer", "geyer_rigid")
  def test_implicit(self, muscle):
    """The implicit integrator, which uses the actuator velocity derivative, matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(
      xml=_HANGING.format(muscle=_MUSCLES[muscle]), keyframe=0, overrides={"opt.integrator": "IMPLICITFAST"}
    )
    for _ in range(50):
      mujoco.mj_step(mjm, mjd)
      mjw.step(m, d)
    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=1e-5)
    np.testing.assert_allclose(d.qvel.numpy()[0], mjd.qvel, atol=1e-4)


if __name__ == "__main__":
  wp.init()
  absltest.main()
