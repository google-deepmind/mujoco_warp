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

"""Tests for passive force functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import DisableBit
from mujoco_warp import test_data

# tolerance for difference between MuJoCo and MJWarp passive force calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class PassiveTest(parameterized.TestCase):
  @parameterized.product(spring=(0, DisableBit.SPRING), damper=(0, DisableBit.DAMPER), gravity=(0, DisableBit.GRAVITY))
  def test_passive(self, spring, damper, gravity):
    """Tests passive."""
    _, mjd, m, d = test_data.fixture(
      "pendula.xml",
      qvel_noise=0.01,
      ctrl_noise=0.1,
      qfrc_noise=0.1,
      xfrc_noise=0.1,
      overrides={"opt.disableflags": DisableBit.CONTACT | spring | damper | gravity},
    )

    for arr in (d.qfrc_spring, d.qfrc_damper, d.qfrc_gravcomp, d.qfrc_passive):
      arr.zero_()

    mjw.passive(m, d)

    _assert_eq(d.qfrc_spring.numpy()[0], mjd.qfrc_spring, "qfrc_spring")
    _assert_eq(d.qfrc_damper.numpy()[0], mjd.qfrc_damper, "qfrc_damper")
    _assert_eq(d.qfrc_gravcomp.numpy()[0], mjd.qfrc_gravcomp, "qfrc_gravcomp")
    _assert_eq(d.qfrc_passive.numpy()[0], mjd.qfrc_passive, "qfrc_passive")

  @parameterized.parameters(
    (1, 0, 0, 0, 0),
    (0, 1, 0, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 0, 0, 1),
    (1, 1, 1, 1, 1),
  )
  def test_fluid(self, density, viscosity, wind0, wind1, wind2):
    """Tests fluid model."""
    _, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option density="{density}" viscosity="{viscosity}" wind="{wind0} {wind1} {wind2}"/>
        <worldbody>
          <body>
            <geom type="box" size=".1 .1 .1"/>
            <freejoint/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="1 1 1 1 1 1"/>
        </keyframe>
      </mujoco>
    """,
      keyframe=0,
    )

    for arr in (d.qfrc_passive, d.qfrc_fluid):
      arr.zero_()

    mjw.passive(m, d)

    _assert_eq(d.qfrc_passive.numpy()[0], mjd.qfrc_passive, "qfrc_passive")
    _assert_eq(d.qfrc_fluid.numpy()[0], mjd.qfrc_fluid, "qfrc_fluid")

  def test_ellipsoid_fluid(self):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option density="1.3" viscosity="0.07" wind="0.1 0.2 -0.05"/>
        <worldbody>
          <body>
            <geom type="sphere" size="0.1 0.3 0.005" fluidshape="ellipsoid"/>
            <freejoint/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="0.7 -0.3 0.4 -0.6 0.8 -0.2"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
    )

    max_err = 0.0
    for _ in range(25):
      mjw.step(m, d)
      mujoco.mj_step(mjm, mjd)

      warp_force = d.qfrc_fluid.numpy()[0]
      mj_force = mjd.qfrc_fluid
      diff = np.abs(warp_force - mj_force)
      max_err = max(max_err, float(np.max(diff)))

    self.assertLess(max_err, 5e-4)

  def test_fluid_zero_mass_body(self):
    """Tests fluid forces for zero mass body."""
    _, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option density="1.2" viscosity="0.1"/>
        <worldbody>
          <geom name="floor" type="plane" size="10 10 0.1"/>
          <!-- empty body with freejoint but child has the mass -->
          <body name="empty_root" pos="0 0 0.5">
            <freejoint/>
            <body name="child" pos="0.1 0 0">
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </body>
        </worldbody>
      </mujoco>
    """,
    )

    for arr in (d.qfrc_fluid, d.qfrc_passive):
      arr.fill_(wp.inf)

    mjw.passive(m, d)

    _assert_eq(d.qfrc_fluid.numpy()[0], mjd.qfrc_fluid, "qfrc_fluid")
    _assert_eq(d.qfrc_passive.numpy()[0], mjd.qfrc_passive, "qfrc_passive")

  @parameterized.product(
    jacobian=(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE), gravity=(0, DisableBit.GRAVITY)
  )
  def test_gravcomp(self, jacobian, gravity):
    """Tests gravity compensation."""
    _, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="1 2 3">
          <flag contact="disable"/>
        </option>
        <worldbody>
          <body gravcomp="1">
            <geom type="sphere" size=".1" pos="1 0 0"/>
            <joint name="joint0" type="hinge" axis="0 1 0" actuatorgravcomp="true"/>
          </body>
          <body gravcomp="1">
            <geom type="sphere" size=".1"/>
            <joint name="joint1" type="hinge" axis="1 0 0"/>
            <joint type="hinge" axis="0 1 0"/>
            <joint type="hinge" axis="0 0 1"/>
          </body>
          <body gravcomp="1">
            <geom type="sphere" size=".1"/>
            <joint type="hinge" axis="0 1 0"/>
          </body>
          <body gravcomp="0">
            <geom type="sphere" size=".1"/>
            <joint type="hinge" axis="0 1 0"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="joint0"/>
          <motor joint="joint1"/>
        </actuator>
      </mujoco>
    """,
      overrides={"opt.disableflags": DisableBit.CONTACT | gravity, "opt.jacobian": jacobian},
    )

    for arr in (d.qfrc_passive, d.qfrc_gravcomp, d.qfrc_actuator):
      arr.zero_()

    mjw.passive(m, d)
    mjw.fwd_actuation(m, d)

    _assert_eq(d.qfrc_passive.numpy()[0], mjd.qfrc_passive, "qfrc_passive")
    _assert_eq(d.qfrc_gravcomp.numpy()[0], mjd.qfrc_gravcomp, "qfrc_gravcomp")
    _assert_eq(d.qfrc_actuator.numpy()[0], mjd.qfrc_actuator, "qfrc_actuator")

  def test_polynomial_stiffness(self):
    """Tests polynomial stiffness."""
    xml = """
    <mujoco model="Polynomial Stiffness">
      <default>
        <geom condim="3"/>
        <site rgba="0 .8 0 .5" size=".03"/>
        <tendon width=".01" rgba=".9 .9 .9 1" springlength="0 .3"/>
        <default class="box">
          <geom type="box" size=".2 .2 .2" mass="5"/>
        </default>
      </default>
      <worldbody>
        <geom type="plane" size="3 3 .01"/>
        <light directional="true" diffuse=".4 .4 .4" specular="0 0 0" pos="0 0 5" dir="0 0 -1" castshadow="false"/>
        <site name="anchor_linear"    pos="-1.5 0 2"/>
        <site name="anchor_quadratic" pos="-.5  0 2"/>
        <site name="anchor_cubic"     pos=".5   0 2"/>
        <site name="anchor_combi"     pos="1.5  0 2"/>
        <body name="linear" pos="-1.5 0 1.5">
          <freejoint/>
          <geom class="box" rgba=".2 .6 1 1"/>
          <site name="hook_linear" pos="0 0 .2"/>
        </body>
        <body name="quadratic" pos="-.5 0 1.5">
          <freejoint/>
          <geom class="box" rgba="1 .4 .1 1"/>
          <site name="hook_quadratic" pos="0 0 .2"/>
        </body>
        <body name="cubic" pos=".5 0 1.5">
          <freejoint/>
          <geom class="box" rgba=".2 .8 .2 1"/>
          <site name="hook_cubic" pos="0 0 .2"/>
        </body>
        <body name="combi" pos="1.5 0 1.5">
          <freejoint/>
          <geom class="box" rgba=".8 .2 .8 1"/>
          <site name="hook_combi" pos="0 0 .2"/>
        </body>
      </worldbody>
      <tendon>
        <spatial name="linear" stiffness="100">
          <site site="anchor_linear"/>
          <site site="hook_linear"/>
        </spatial>
        <spatial name="quadratic" stiffness="0 200">
          <site site="anchor_quadratic"/>
          <site site="hook_quadratic"/>
        </spatial>
        <spatial name="cubic" stiffness="0 0 300">
          <site site="anchor_cubic"/>
          <site site="hook_cubic"/>
        </spatial>
        <spatial name="combination" stiffness="100 -400 400">
          <site site="anchor_combi"/>
          <site site="hook_combi"/>
        </spatial>
      </tendon>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml)

    for arr in (d.qfrc_spring, d.qfrc_damper, d.qfrc_passive):
      arr.fill_(wp.inf)

    mjw.passive(m, d)

    _assert_eq(d.qfrc_spring.numpy()[0], mjd.qfrc_spring, "qfrc_spring")
    _assert_eq(d.qfrc_damper.numpy()[0], mjd.qfrc_damper, "qfrc_damper")
    _assert_eq(d.qfrc_passive.numpy()[0], mjd.qfrc_passive, "qfrc_passive")

  def test_polynomial_stiffness_joint(self):
    """Tests polynomial stiffness for joints."""
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint type="slide" stiffness="2 3 4"/>
          <geom size="1" mass="1"/>
        </body>
      </worldbody>
      <keyframe>
        <key qpos="0.5"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)
    d.qfrc_spring.fill_(wp.inf)
    mjw.passive(m, d)
    _assert_eq(d.qfrc_spring.numpy()[0], mjd.qfrc_spring, "qfrc_spring")

  def test_polynomial_damping_joint(self):
    """Tests polynomial damping for joints."""
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint type="slide" damping="2 3 4"/>
          <geom size="1" mass="1"/>
        </body>
      </worldbody>
      <keyframe>
        <key qvel="0.5"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)
    d.qfrc_damper.zero_()
    mjw.passive(m, d)
    _assert_eq(d.qfrc_damper.numpy()[0], mjd.qfrc_damper, "qfrc_damper")

  def test_polynomial_stiffness_fixed_tendon(self):
    """Tests polynomial stiffness for fixed tendons."""
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint type="slide" name="j"/>
          <geom size="1" mass="1"/>
        </body>
      </worldbody>
      <tendon>
        <fixed stiffness="10 5 1">
          <joint joint="j" coef="1"/>
        </fixed>
      </tendon>
      <keyframe>
        <key qpos="2"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)
    d.qfrc_spring.fill_(wp.inf)
    mjw.passive(m, d)
    _assert_eq(d.qfrc_spring.numpy()[0], mjd.qfrc_spring, "qfrc_spring")

  def test_polynomial_damping_tendon(self):
    """Tests polynomial damping for tendons."""
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint type="slide" name="j"/>
          <geom size="1" mass="1"/>
        </body>
      </worldbody>
      <tendon>
        <fixed damping="10 5 1">
          <joint joint="j" coef="1"/>
        </fixed>
      </tendon>
      <keyframe>
        <key qvel="2"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)
    d.qfrc_damper.fill_(wp.inf)
    mjw.passive(m, d)
    _assert_eq(d.qfrc_damper.numpy()[0], mjd.qfrc_damper, "qfrc_damper")

  def test_polynomial_stiffness_energy(self):
    """Tests energy conservation with polynomial stiffness."""
    xml = """
    <mujoco>
      <option timestep="0.0001">
        <flag energy="enable"/>
      </option>
      <worldbody>
        <body>
          <joint type="slide" stiffness="10 5 1"/>
          <geom size="1" mass="1"/>
        </body>
      </worldbody>
      <keyframe>
        <key qpos="2"/>
      </keyframe>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)

    d.energy.fill_(wp.inf)
    mjw.forward(m, d)
    e0 = d.energy.numpy()[0]
    total_energy = e0[0] + e0[1]

    for _ in range(10):
      mjw.step(m, d)
      e = d.energy.numpy()[0]
      self.assertAlmostEqual(e[0] + e[1], total_energy, delta=0.005)


if __name__ == "__main__":
  wp.init()
  absltest.main()
