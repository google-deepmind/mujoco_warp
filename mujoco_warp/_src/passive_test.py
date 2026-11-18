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

import copy
import gc

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import ConeType
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
  @parameterized.product(nworld=(1, 2), damper=(0, DisableBit.DAMPER))
  def test_per_dof_damping(self, nworld, damper):
    """Tests independent damping coefficients for every DOF of free and ball joints."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom size="0.1" contype="0" conaffinity="0"/>
          </body>
          <body pos="1 0 0">
            <joint type="ball"/>
            <geom size="0.1" contype="0" conaffinity="0"/>
          </body>
          <body pos="2 0 0">
            <joint type="hinge"/>
            <joint type="slide"/>
            <geom size="0.1" contype="0" conaffinity="0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      overrides={"opt.disableflags": damper},
      batch_sizes={"dof_damping": nworld, "dof_dampingpoly": nworld},
    )
    damping = np.tile(np.arange(mjm.nv, dtype=np.float32), (nworld, 1))
    poly = np.zeros((nworld, mjm.nv, 2), dtype=np.float32)
    poly[:, 2::2, 0] = 0.3
    poly[:, 3::2, 1] = 0.2
    # Include zero damping on the first free DOF and polynomial-only damping on another.
    damping[:, 2] = 0.0
    velocity = np.tile(np.linspace(-0.5, 0.7, mjm.nv, dtype=np.float32), (nworld, 1))
    if nworld == 2:
      damping[1] *= 0.5
      poly[1] *= 2.0
      velocity[1] *= -1.5
    m.dof_damping.assign(damping)
    m.dof_dampingpoly.assign(poly)
    d.qvel.assign(velocity)
    d.qfrc_damper.fill_(wp.inf)
    d.qfrc_passive.fill_(wp.inf)

    mjw.passive(m, d)

    for world in range(nworld):
      mjm.dof_damping[:] = damping[world]
      mjm.dof_dampingpoly[:] = poly[world]
      mjd.qvel[:] = velocity[world]
      mujoco.mj_passive(mjm, mjd)
      _assert_eq(d.qfrc_damper.numpy()[world], mjd.qfrc_damper, f"qfrc_damper_world_{world}")
      _assert_eq(d.qfrc_passive.numpy()[world], mjd.qfrc_passive, f"qfrc_passive_world_{world}")
    if nworld == 2 and not damper:
      self.assertFalse(np.allclose(d.qfrc_damper.numpy()[0], d.qfrc_damper.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_actuator_damping(self, nworld):
    """Tests actuator damping contribution to joint and tendon damping."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <body>
            <joint name="j0" axis="0 1 0"/>
            <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
            <body pos=".3 0 0">
              <joint name="j1" axis="0 1 0"/>
              <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
              <body pos=".3 0 0">
                <joint name="j2" type="slide" axis="1 0 0"/>
                <geom size=".05"/>
              </body>
            </body>
          </body>
        </worldbody>
        <tendon>
          <fixed name="t0">
            <joint joint="j1" coef="1"/>
            <joint joint="j2" coef="-.5"/>
          </fixed>
        </tendon>
        <actuator>
          <general joint="j0" gear="2" damping=".3 .1 .05"/>
          <general joint="j0" gear="-3" damping=".1 0 0"/>
          <general jointinparent="j1" gear="1.5" damping=".2 0 .1"/>
          <general tendon="t0" gear="2" damping=".4 .2 0"/>
        </actuator>
        <keyframe>
          <key qpos=".1 .2 .05" qvel=".3 -.4 .5"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      nworld=nworld,
      batch_sizes={
        "actuator_damping": nworld,
        "actuator_dampingpoly": nworld,
        "actuator_gear": nworld,
      },
    )

    # actuator damping is not folded into joint and tendon damping
    _assert_eq(m.dof_damping.numpy()[0], mjm.dof_damping, "dof_damping")
    _assert_eq(m.dof_dampingpoly.numpy()[0], mjm.dof_dampingpoly, "dof_dampingpoly")
    _assert_eq(m.tendon_damping.numpy()[0], mjm.tendon_damping, "tendon_damping")
    _assert_eq(
      m.tendon_dampingpoly.numpy()[0],
      mjm.tendon_dampingpoly,
      "tendon_dampingpoly",
    )

    damping = np.tile(mjm.actuator_damping, (nworld, 1))
    poly = np.tile(mjm.actuator_dampingpoly, (nworld, 1, 1))
    gear = np.tile(mjm.actuator_gear, (nworld, 1, 1))
    if nworld == 2:
      damping[1] *= 2.0
      poly[1] *= 0.5
      gear[1] *= 1.5
    m.actuator_damping.assign(damping)
    m.actuator_dampingpoly.assign(poly)
    m.actuator_gear.assign(gear)
    d.qfrc_damper.fill_(wp.inf)
    d.qfrc_passive.fill_(wp.inf)

    mjw.passive(m, d)

    for world in range(nworld):
      mjm.actuator_damping[:] = damping[world]
      mjm.actuator_dampingpoly[:] = poly[world]
      mjm.actuator_gear[:] = gear[world]
      mujoco.mj_passive(mjm, mjd)
      _assert_eq(
        d.qfrc_damper.numpy()[world],
        mjd.qfrc_damper,
        f"qfrc_damper_world_{world}",
      )
      _assert_eq(
        d.qfrc_passive.numpy()[world],
        mjd.qfrc_passive,
        f"qfrc_passive_world_{world}",
      )
    if nworld == 2:
      self.assertFalse(np.allclose(d.qfrc_damper.numpy()[0], d.qfrc_damper.numpy()[1]))

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

  @parameterized.parameters(
    "Euler",
    "implicitfast",
  )
  def test_ellipsoid_fluid(self, integrator):
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option density="1.3" viscosity="0.07" wind="0.1 0.2 -0.05" integrator="{integrator}"/>
        <worldbody>
          <body>
            <geom size="0.1 0.3 0.005" fluidshape="ellipsoid"/>
            <geom size="0.05" pos="0.15 -0.1 0.05" fluidshape="ellipsoid"/>
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

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    max_err = 0.0
    for _ in range(20):
      d.qfrc_fluid.fill_(wp.inf)
      wp.capture_launch(capture.graph)
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

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)

    for _ in range(10):
      wp.capture_launch(capture.graph)
      e = d.energy.numpy()[0]
      self.assertAlmostEqual(e[0] + e[1], total_energy, delta=0.005)

  def test_thin_ellipsoid_fluid(self):
    """Verify that thin ellipsoid geoms do not diverge due to underflow in proj_denom."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" density="1000" viscosity="0.0013"/>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="ellipsoid" size="0.0035 0.0005 0.03" mass="1" fluidshape="ellipsoid" fluidcoef="0.4 7.79 2.81 3.84 0.27"/>
          </body>
        </worldbody>
        <keyframe>
          <key qvel="2.0 1.0 0.5 0.0 0.0 0.0"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
    )

    d.qfrc_fluid.fill_(wp.inf)
    mjw.forward(m, d)
    self.assertTrue(np.isfinite(d.qfrc_fluid.numpy()[0]).all())
    np.testing.assert_allclose(d.qfrc_fluid.numpy()[0], mjd.qfrc_fluid, atol=5e-4, rtol=5e-4)

  def test_kutta_lift_low_speed_scaling(self):
    """Verify exact v^2 scaling of Kutta lift at low speeds."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option density="1.225"/>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="ellipsoid" size=".025 .01 .001" euler="0 -30 0"
                  fluidshape="ellipsoid"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )

    # reference lift/v^2 at v = 10 m/s
    qvel = np.zeros(mjm.nv, dtype=float)
    qvel[0] = 10.0
    mjd.qvel[0] = 10.0
    wp.copy(d.qvel, wp.array([qvel], dtype=float))

    with wp.ScopedCapture() as capture:
      mjw.forward(m, d)

    mujoco.mj_forward(mjm, mjd)
    d.qfrc_fluid.fill_(wp.inf)
    wp.capture_launch(capture.graph)

    warp_force = d.qfrc_fluid.numpy()[0]
    np.testing.assert_allclose(warp_force, mjd.qfrc_fluid, atol=1e-6, rtol=1e-5)

    ref_lift_coeff = warp_force[2] / (10.0 * 10.0)
    self.assertGreater(ref_lift_coeff, 0.0)

    # verify exact v^2 scaling at low speeds (1 m/s, 0.1 m/s, 1 cm/s, 1 mm/s)
    for v in (1.0, 0.1, 0.01, 0.001):
      qvel[0] = v
      wp.copy(d.qvel, wp.array([qvel], dtype=float))
      d.qfrc_fluid.fill_(wp.inf)
      wp.capture_launch(capture.graph)
      lift_coeff = d.qfrc_fluid.numpy()[0, 2] / (v * v)
      np.testing.assert_allclose(lift_coeff, ref_lift_coeff, atol=1e-12 * ref_lift_coeff, rtol=1e-5)

  def test_adhesion_gap_force(self):
    """Test contact creation and passive attraction in the gap band without collision."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <geom type="sphere" size="0.1" pos="0 0 0"/>
          <body pos="0 0 0.22">
            <geom type="sphere" size="0.1" gap="0.05" adhesion="8.0"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
      """,
    )
    d.nacon.fill_(-1)
    d.contact.dim.fill_(-1)
    d.contact.adhesion.fill_(wp.inf)
    d.qfrc_adhesion.fill_(wp.inf)

    mjw.collision(m, d)
    mjw.passive(m, d)

    self.assertGreater(d.nacon.numpy()[0], 0)
    self.assertEqual(d.contact.adhesion.numpy()[0], 8.0)
    self.assertEqual(d.contact.dim.numpy()[0], 1)
    self.assertGreater(np.linalg.norm(d.qfrc_adhesion.numpy()[0]), 0)

  def test_adhesion_capture_dynamics(self):
    """Test dynamic capture of a body released inside the adhesion gap band."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <geom type="sphere" size="0.1" pos="0 0 0"/>
          <body pos="0 0 0.22">
            <geom type="sphere" size="0.1" gap="0.05" adhesion="20.0"/>
            <joint type="slide" axis="0 0 1"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    d.nacon.fill_(-1)
    d.contact.dim.fill_(-1)
    d.contact.adhesion.fill_(wp.inf)
    d.qfrc_adhesion.fill_(wp.inf)

    mjw.collision(m, d)
    mjw.passive(m, d)

    self.assertEqual(d.nacon.numpy()[0], 1)
    self.assertEqual(d.contact.dim.numpy()[0], 1)
    self.assertEqual(d.contact.adhesion.numpy()[0], 20.0)
    self.assertGreater(np.linalg.norm(d.qfrc_adhesion.numpy()[0]), 0)

  def test_adhesion_resting_penetration_equivalence(self):
    """Verify resting penetration equilibrium and passive adhesion forces match MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 -9.81"/>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <body pos="0 0 0.1">
            <geom type="sphere" size="0.1" adhesion="15.0"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    d.qfrc_adhesion.fill_(wp.inf)
    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    for _ in range(20):
      mujoco.mj_step(mjm, mjd)
      wp.capture_launch(capture.graph)

    np.testing.assert_allclose(d.qpos.numpy()[0], mjd.qpos, atol=1e-3)
    np.testing.assert_allclose(
      d.qfrc_adhesion.numpy()[0],
      mjd.qfrc_passive,
      atol=1e-3,
    )

  @parameterized.parameters(
    (0, True),
    (DisableBit.SPRING, True),
    (DisableBit.DAMPER, True),
    (DisableBit.CONSTRAINT, False),
    (DisableBit.CONTACT, False),
  )
  def test_adhesion_disableflags(self, disableflags, adhesion_expected):
    """Verify adhesion passive force logic respects DisableBit.CONTACT."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <geom type="sphere" size="0.1" adhesion="5.0"/>
          <body pos="0 0 0.15">
            <freejoint/>
            <geom type="sphere" size="0.1" adhesion="3.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
    )
    d.qfrc_adhesion.fill_(wp.inf)
    m.opt.disableflags |= disableflags
    mjw.collision(m, d)
    mjw.passive(m, d)

    if adhesion_expected:
      self.assertGreater(np.linalg.norm(d.qfrc_adhesion.numpy()[0]), 0)
    else:
      self.assertEqual(np.linalg.norm(d.qfrc_adhesion.numpy()[0]), 0)

  def test_adhesion_mutual_attraction_two_dynamic_bodies(self):
    """Verify mutual attraction and qfrc_passive parity between two free-floating bodies."""
    _, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <body name="b1" pos="0 0 0">
            <geom name="g1" type="sphere" size="0.1" adhesion="5.0"/>
            <joint name="j1" type="free"/>
          </body>
          <body name="b2" pos="0 0 0.15">
            <geom name="g2" type="sphere" size="0.1" adhesion="3.0"/>
            <joint name="j2" type="free"/>
          </body>
        </worldbody>
      </mujoco>
      """
    )
    d.qfrc_adhesion.fill_(wp.inf)
    d.qfrc_passive.fill_(wp.inf)
    mjw.forward(m, d)

    # Lower sphere pulled +z towards upper sphere; upper sphere pulled -z towards lower sphere.
    _assert_eq(d.qfrc_adhesion.numpy()[0], mjd.qfrc_passive, "qfrc_adhesion")
    _assert_eq(d.qfrc_passive.numpy()[0], mjd.qfrc_passive, "qfrc_passive")

  @parameterized.product(
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    condim=(1, 3, 4, 6),
  )
  def test_adhesion_pyramidal_aref(self, cone, condim):
    """Verify efc_aref for adhesive contacts across cone types and condim."""
    _, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <worldbody>
          <geom type="plane" size="1 1 0.1"/>
          <body pos="0 0 0.09">
            <geom type="sphere" size="0.1" condim="{condim}" adhesion="12.0"/>
            <joint type="free"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
    )
    d.nefc.fill_(-1)
    d.efc.aref.fill_(wp.inf)
    mjw.forward(m, d)

    nefc = mjd.nefc
    self.assertGreater(nefc, 0)
    _assert_eq(d.efc.aref.numpy()[0, :nefc], mjd.efc_aref[:nefc], "efc_aref")

  @parameterized.product(nworld=(1, 2), integrator=("Euler", "RK4", "implicit", "implicitfast", "discrete"))
  def test_spring_plastic(self, nworld, integrator):
    """Verify plastic springs for joints and tendons match MuJoCo across integrators and worlds."""
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option gravity="0 0 0" timestep="0.005" integrator="{integrator}">
          <flag energy="enable"/>
        </option>
        <worldbody>
          <site name="s0"/>
          <body name="b_elastic" pos="0 -1 0">
            <joint name="j_elastic" type="slide" axis="1 0 0" stiffness="10" damping="0.2"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
          <body name="b_slide">
            <joint name="j_slide" type="slide" axis="1 0 0" stiffness="20" springref="0.1" springyield="0.05" damping="0.5"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
            <site name="s1"/>
          </body>
          <body name="b_hinge" pos="0 1 0">
            <joint name="j_hinge" stiffness="15" springref="-0.1" springyield="0.08" damping="0.3"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
          <body name="b_ball" pos="0 2 0">
            <joint name="j_ball" type="ball" stiffness="12" springyield="0.06" damping="0.2"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="t_elastic" stiffness="10" damping="0.1">
            <joint joint="j_elastic" coef="1.0"/>
          </fixed>
          <fixed name="t_fixed" stiffness="25" springlength="0 0.05" springyield="0.04" damping="0.2">
            <joint joint="j_slide" coef="1.0"/>
            <joint joint="j_hinge" coef="-0.5"/>
          </fixed>
          <spatial name="t_spatial" stiffness="30" springlength="0.05 0.1" springyield="0.03" damping="0.2">
            <site site="s0"/>
            <site site="s1"/>
          </spatial>
        </tendon>
        <keyframe>
          <key qpos="0.15 0.25 -0.3 0.98 0.15 0.12 -0.05"
               qvel="0.2 0.4 -0.5 0.3 -0.2 0.1"
               plastic="0.12 -0.12 1 0 0 0 0 0.06 0.05 0.11"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      nworld=nworld,
    )

    # Interleaved elastic joint and tendon produce non-zero plastic offsets.
    self.assertEqual(mjm.jnt_plasticadr[0], -1)
    self.assertEqual(mjm.tendon_plasticadr[0], -1)
    self.assertEqual(m.nplastic, 10)

    mjds = [copy.deepcopy(mjd) for _ in range(nworld)]
    if nworld == 2:
      qpos = d.qpos.numpy().copy()
      qvel = d.qvel.numpy().copy()
      mjds[1].qpos[:] = [-0.1, -0.35, 0.35, 0.96, -0.2, 0.1, 0.17]
      mjds[1].qvel[:] = [-0.3, -0.5, 0.6, -0.25, 0.35, -0.15]
      mujoco.mj_forward(mjm, mjds[1])
      qpos[1] = mjds[1].qpos
      qvel[1] = mjds[1].qvel
      d.qpos.assign(qpos)
      d.qvel.assign(qvel)
      mjw.forward(m, d)

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    for _ in range(10):
      wp.capture_launch(capture.graph)
    for w in range(nworld):
      mujoco.mj_step(mjm, mjds[w], 10)

    # Pre-fill pure output arrays before evaluating passive forces and energy at yielded state.
    d.qfrc_spring.fill_(wp.inf)
    d.qfrc_passive.fill_(wp.inf)
    d.cfrc_ext.fill_(wp.inf)
    d.energy.fill_(wp.inf)
    mjw.forward(m, d)
    mjw.rne_postconstraint(m, d)
    mjw.energy_pos(m, d)
    plastic = d.plastic.numpy()
    qfrc_spring = d.qfrc_spring.numpy()
    qfrc_passive = d.qfrc_passive.numpy()
    qpos = d.qpos.numpy()
    qvel = d.qvel.numpy()
    cfrc_ext = d.cfrc_ext.numpy()
    energy = d.energy.numpy()
    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])
      mujoco.mj_rnePostConstraint(mjm, mjds[w])
      mujoco.mj_energyPos(mjm, mjds[w])
      _assert_eq(plastic[w], mjds[w].plastic, f"plastic_w{w}")
      _assert_eq(qfrc_spring[w], mjds[w].qfrc_spring, f"qfrc_spring_w{w}")
      _assert_eq(qfrc_passive[w], mjds[w].qfrc_passive, f"qfrc_passive_w{w}")
      _assert_eq(qpos[w], mjds[w].qpos, f"qpos_w{w}")
      _assert_eq(qvel[w], mjds[w].qvel, f"qvel_w{w}")
      _assert_eq(cfrc_ext[w], mjds[w].cfrc_ext, f"cfrc_ext_w{w}")
      _assert_eq(energy[w, 0], mjds[w].energy[0], f"energy_pos_w{w}")

    if nworld == 2:
      self.assertFalse(np.allclose(plastic[0], plastic[1]))
      self.assertFalse(np.allclose(qpos[0], qpos[1]))

  def test_spring_plastic_disable_flag(self):
    """Verify DisableBit.SPRING suppresses plastic flow while 1D rod twist still updates."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" timestep="0.005">
          <flag spring="disable"/>
        </option>
        <worldbody>
          <site name="s0"/>
          <body name="b_slide">
            <joint name="j_slide" type="slide" axis="1 0 0" stiffness="20" springyield="0.02"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
            <site name="s1"/>
          </body>
          <flexcomp name="shell" type="direct" dim="2" mass="1"
                    point="2 0 0  3 0 0  2.5 0.5 0  2.5 -0.5 0"
                    element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="both" yield="0.002"/>
            <pin id="0 1 2"/>
          </flexcomp>
          <flexcomp name="box" type="grid" dim="3" pos="5 0 0" dof="trilinear"
                    count="2 2 2" cellcount="1 1 1" spacing="0.5 0.5 0.5" mass="1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.25" yield="0.02"/>
            <pin id="0 1 2 3"/>
          </flexcomp>
          <flexcomp name="rod" type="grid" count="4 1 1" spacing="0.1 0.1 0.1" pos="0 2 0"
                    radius="0.008" dim="1" mass="0.08">
            <pin id="0 1"/>
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="5e5" damping="0.05"/>
          </flexcomp>
        </worldbody>
        <tendon>
          <spatial name="t_spatial" stiffness="30" springlength="0 0.05" springyield="0.02">
            <site site="s0"/>
            <site site="s1"/>
          </spatial>
        </tendon>
      </mujoco>
      """,
    )

    # Displace every plastic element well past its yield threshold and give the 1D rod velocity.
    mjd.qpos[0] = 0.5
    shell_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_FLEX, "shell")
    box_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_FLEX, "box")
    rod_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_FLEX, "rod")
    shell_b = mjm.flex_vertbodyid[mjm.flex_vertadr[shell_id] + 3]
    mjd.qpos[mjm.body_dofadr[shell_b] : mjm.body_dofadr[shell_b] + 3] = [0.1, -0.2, 0.3]
    for i in range(4, 8):
      nb = mjm.flex_nodebodyid[mjm.flex_nodeadr[box_id] + i]
      mjd.qpos[mjm.body_dofadr[nb] : mjm.body_dofadr[nb] + 3] = [0.2, 0.1, 0.15]
    rod_b = mjm.flex_vertbodyid[mjm.flex_vertadr[rod_id] + 3]
    mjd.qvel[mjm.body_dofadr[rod_b] : mjm.body_dofadr[rod_b] + 3] = [0.0, 2.0, 3.0]
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)
    plastic_before = d.plastic.numpy()[0].copy()

    mujoco.mj_step(mjm, mjd, 5)
    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    for _ in range(5):
      wp.capture_launch(capture.graph)

    plastic_after = d.plastic.numpy()[0]
    _assert_eq(plastic_after, mjd.plastic, "plastic_disable_spring")
    # Joint, tendon, shell, and interpolated box plastic states remain frozen.
    rod_adr = mjm.flex_plasticadr[rod_id]
    _assert_eq(plastic_after[:rod_adr], plastic_before[:rod_adr], "non_rod_plastic_frozen")
    # 1D rod reference frames still transport with DisableBit.SPRING enabled.
    self.assertFalse(np.allclose(plastic_after[rod_adr:], plastic_before[rod_adr:]))

  def test_tendon_plastic_unilateral_and_bilateral(self):
    """Verify unilateral slack pinning and bilateral symmetric shifting of plastic tendons."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" timestep="0"/>
        <worldbody>
          <body name="b">
            <joint name="j" type="slide" axis="1 0 0"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="t_unilat" stiffness="20" springlength="0 0.1" springyield="0.02">
            <joint joint="j" coef="1"/>
          </fixed>
          <fixed name="t_bilat" stiffness="20" springlength="0.05 0.1" springyield="0.02">
            <joint joint="j" coef="1"/>
          </fixed>
        </tendon>
      </mujoco>
      """,
    )

    # Tension beyond yield: unilateral keeps lower=0, bilateral shifts both bounds equally.
    mjd.qpos[0] = 0.2
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    p = d.plastic.numpy()[0]
    _assert_eq(p, mjd.plastic, "tension_plastic")
    _assert_eq(p[:2], np.array([0.0, 0.18]), "unilateral_tension")
    _assert_eq(p[2:4], np.array([0.13, 0.18]), "bilateral_tension")

    # Compression beyond yield: unilateral stays slack, bilateral shifts both bounds down.
    mjd.qpos[0] = 0.02
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    p = d.plastic.numpy()[0]
    _assert_eq(p, mjd.plastic, "compression_plastic")
    _assert_eq(p[:2], np.array([0.0, 0.18]), "unilateral_compression_unchanged")
    _assert_eq(p[2:4], np.array([0.04, 0.09]), "bilateral_compression")

  def test_spring_plastic_zero_yield_and_batched_yield(self):
    """Verify runtime zero yield full slip and batched per-world springyield thresholds."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" timestep="0"/>
        <worldbody>
          <body name="b">
            <joint name="j" type="slide" axis="1 0 0" stiffness="20" springyield="0.05"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="t" stiffness="20" springlength="0.05 0.1" springyield="0.05">
            <joint joint="j" coef="1"/>
          </fixed>
        </tendon>
        <keyframe>
          <key qpos="0.25"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      nworld=2,
      batch_sizes={"jnt_springyield": 2, "tendon_springyield": 2},
    )

    # World 0: runtime yield=0 (full plastic slip); world 1: yield=10 (elastic, no slip).
    m.jnt_springyield.assign(np.array([[0.0], [10.0]], dtype=np.float32))
    m.tendon_springyield.assign(np.array([[0.0], [10.0]], dtype=np.float32))
    mjw.step(m, d)

    mjm_w0 = copy.deepcopy(mjm)
    mjm_w0.jnt_springyield[:] = 0.0
    mjm_w0.tendon_springyield[:] = 0.0
    mjd_w0 = copy.deepcopy(mjd)
    mujoco.mj_step(mjm_w0, mjd_w0)

    mjm_w1 = copy.deepcopy(mjm)
    mjm_w1.jnt_springyield[:] = 10.0
    mjm_w1.tendon_springyield[:] = 10.0
    mjd_w1 = copy.deepcopy(mjd)
    mujoco.mj_step(mjm_w1, mjd_w1)

    plastic = d.plastic.numpy()
    _assert_eq(plastic[0], mjd_w0.plastic, "batched_zero_yield_w0")
    _assert_eq(plastic[1], mjd_w1.plastic, "batched_elastic_w1")
    _assert_eq(plastic[0, 0], 0.25, "full_slip_joint")
    _assert_eq(plastic[0, 1:3], np.array([0.2, 0.25]), "full_slip_tendon")
    self.assertFalse(np.allclose(plastic[0], plastic[1]))

  def test_spring_plastic_zero_force_at_yielded_rest(self):
    """Verify yielded joints and tendons produce zero spring force and energy at rest state."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" timestep="0">
          <flag energy="enable"/>
        </option>
        <worldbody>
          <site name="s0"/>
          <body name="b_slide">
            <joint name="j_slide" type="slide" axis="1 0 0" stiffness="20" springyield="0.05"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
          <body name="b_hinge" pos="0 1 0">
            <joint name="j_hinge" stiffness="15" springyield="0.05"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
          <body name="b_ball" pos="0 2 0">
            <joint name="j_ball" type="ball" stiffness="12" springyield="0.05"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
          <body name="b_ten" pos="0 3 0">
            <joint name="j_ten" type="slide" axis="1 0 0"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
          </body>
          <body name="b_spatial" pos="0 4 0">
            <joint name="j_spatial" type="slide" axis="0 1 0"/>
            <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
            <site name="s1"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="t_fixed" stiffness="25" springlength="0.05 0.1" springyield="0.03">
            <joint joint="j_ten" coef="1"/>
          </fixed>
          <spatial name="t_spatial" stiffness="25" springlength="4.0 4.05" springyield="0.02">
            <site site="s0"/>
            <site site="s1"/>
          </spatial>
        </tendon>
        <keyframe>
          <key qpos="0.3 -0.4 0.92 0.3 0.2 -0.15 0.4 0.3"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
    )

    # Yield all springs with a zero-timestep step and confirm every spring yielded.
    plastic_before = d.plastic.numpy()[0].copy()
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    p = d.plastic.numpy()[0].copy()
    _assert_eq(p, mjd.plastic, "yielded_plastic")
    self.assertFalse(np.allclose(p[0], plastic_before[0]))
    self.assertFalse(np.allclose(p[1], plastic_before[1]))
    self.assertFalse(np.allclose(p[2:6], plastic_before[2:6]))
    self.assertFalse(np.allclose(p[6:8], plastic_before[6:8]))
    self.assertFalse(np.allclose(p[8:10], plastic_before[8:10]))

    # Move every joint and tendon body to its yielded rest state and verify zero force/energy.
    mjd.qpos[:6] = p[:6]
    mjd.qpos[6] = p[6]
    mjd.qpos[7] = p[8] - 4.0
    mujoco.mj_forward(mjm, mjd)
    d.qpos.assign(mjd.qpos.reshape(1, -1))
    d.qfrc_spring.fill_(wp.inf)
    d.energy.fill_(wp.inf)
    mjw.forward(m, d)
    mjw.energy_pos(m, d)
    qfrc_spring = d.qfrc_spring.numpy()[0]
    energy_pos = d.energy.numpy()[0, 0]
    _assert_eq(qfrc_spring, mjd.qfrc_spring, "qfrc_spring_at_rest")
    _assert_eq(energy_pos, mjd.energy[0], "energy_at_rest")
    _assert_eq(qfrc_spring, np.zeros(mjm.nv), "zero_spring_force")
    self.assertLess(abs(energy_pos), 1e-4)

  @parameterized.product(nworld=(1, 2), integrator=("Euler", "RK4", "discrete"))
  def test_flex_plastic(self, nworld, integrator):
    """Verify plastic 2D and 3D non-interpolated flexes match MuJoCo across integrators."""
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option gravity="0 0 0" timestep="0.005" integrator="{integrator}"/>
        <worldbody>
          <flexcomp name="tet" type="direct" dim="3" mass="1"
                    point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.25" damping="0.05" yield="0.04"/>
            <pin id="0 1 2"/>
          </flexcomp>
          <flexcomp name="shell" type="direct" dim="2" mass="1"
                    point="2 0 0  3 0 0  2.5 0.5 0  2.5 -0.5 0"
                    element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="both" damping="0.05" yield="0.002"/>
            <pin id="0 1 2"/>
          </flexcomp>
          <flexcomp name="shell_bend" type="direct" dim="2" mass="1"
                    point="4 0 0  5 0 0  4.5 0.5 0  4.5 -0.5 0"
                    element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="bend" damping="0.05" yield="0.002"/>
            <pin id="0 1 2"/>
          </flexcomp>
          <flexcomp name="shell_curved" type="direct" dim="2" mass="1"
                    point="6 0 0  7 0 0  6.3 0.45 0.15  6.65 -0.5 0.2"
                    element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="bend" damping="0.05" yield="0.002"/>
            <pin id="0 1 2"/>
          </flexcomp>
        </worldbody>
        <keyframe>
          <key qpos="0.1 -0.15 0.25  0.05 -0.2 0.35  0.0 0.0 0.25  0.0 -0.08 0.18"
               qvel="0.2 -0.3 0.4  -0.1 0.25 0.5  0.0 0.0 0.3  0.0 -0.15 0.35"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      nworld=nworld,
    )

    self.assertEqual(m.nplastic, 27)  # 6 (tet) + 11 (shell) + 5 (shell_bend) + 5 (shell_curved)
    mjds = [copy.deepcopy(mjd) for _ in range(nworld)]
    if nworld == 2:
      qpos = d.qpos.numpy().copy()
      qvel = d.qvel.numpy().copy()
      mjds[1].qpos[:] = [-0.12, 0.2, -0.15, -0.05, 0.18, -0.3, 0.0, 0.0, -0.28, 0.0, 0.12, -0.22]
      mjds[1].qvel[:] = [-0.25, 0.35, -0.3, 0.15, -0.2, -0.45, 0.0, 0.0, -0.35, 0.0, 0.2, -0.3]
      mujoco.mj_forward(mjm, mjds[1])
      qpos[1] = mjds[1].qpos
      qvel[1] = mjds[1].qvel
      d.qpos.assign(qpos)
      d.qvel.assign(qvel)
      mjw.forward(m, d)

    # Verify stretch plastic yield invalidates cached flex Hessians under discrete integration.
    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    wp.capture_launch(capture.graph)
    if integrator == "discrete":
      valid_after_yield = d.flex_hessian_valid.numpy()
      for w in range(nworld):
        self.assertFalse(valid_after_yield[w, 0])
        self.assertFalse(valid_after_yield[w, 1])
        self.assertTrue(valid_after_yield[w, 2])
        self.assertTrue(valid_after_yield[w, 3])

    for _ in range(9):
      wp.capture_launch(capture.graph)
    for w in range(nworld):
      mujoco.mj_step(mjm, mjds[w], 10)

    d.qfrc_spring.fill_(wp.inf)
    d.qfrc_passive.fill_(wp.inf)
    mjw.forward(m, d)
    plastic = d.plastic.numpy()
    qfrc_spring = d.qfrc_spring.numpy()
    qfrc_passive = d.qfrc_passive.numpy()
    qpos = d.qpos.numpy()
    qvel = d.qvel.numpy()
    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])
      _assert_eq(plastic[w], mjds[w].plastic, f"plastic_w{w}")
      _assert_eq(qfrc_spring[w], mjds[w].qfrc_spring, f"qfrc_spring_w{w}")
      _assert_eq(qfrc_passive[w], mjds[w].qfrc_passive, f"qfrc_passive_w{w}")
      _assert_eq(qpos[w], mjds[w].qpos, f"qpos_w{w}")
      _assert_eq(qvel[w], mjds[w].qvel, f"qvel_w{w}")

    if nworld == 2:
      self.assertFalse(np.allclose(plastic[0], plastic[1]))
      self.assertFalse(np.allclose(qpos[0], qpos[1]))

  @parameterized.parameters(1, 2)
  def test_flex_plastic_snh(self, nworld):
    """Verify plastic 3D Stable Neo-Hookean (elastic3d=1) flex matches MuJoCo with discrete."""
    spec = mujoco.MjSpec.from_string("""
    <mujoco>
      <option gravity="0 0 0" timestep="0.005" integrator="discrete"/>
      <worldbody>
        <flexcomp name="tet" type="direct" dim="3" mass="1"
                  point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
          <contact contype="0" conaffinity="0" selfcollide="none"/>
          <elasticity young="500" poisson="0.25" damping="0.05" yield="0.04"/>
          <pin id="0 1 2"/>
        </flexcomp>
      </worldbody>
      <keyframe>
        <key qpos="0.1 -0.15 0.25" qvel="0.2 -0.3 0.4"/>
      </keyframe>
    </mujoco>
    """)
    spec.flexes[0].elastic3d = 1
    mjm = spec.compile()
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld)

    mjds = [copy.deepcopy(mjd) for _ in range(nworld)]
    if nworld == 2:
      qpos = d.qpos.numpy().copy()
      qvel = d.qvel.numpy().copy()
      mjds[1].qpos[:] = [-0.15, 0.2, -0.1]
      mjds[1].qvel[:] = [-0.3, 0.25, -0.35]
      mujoco.mj_forward(mjm, mjds[1])
      qpos[1] = mjds[1].qpos
      qvel[1] = mjds[1].qvel
      d.qpos.assign(qpos)
      d.qvel.assign(qvel)
      mjw.forward(m, d)

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    for _ in range(10):
      wp.capture_launch(capture.graph)
    for w in range(nworld):
      mujoco.mj_step(mjm, mjds[w], 10)

    d.qfrc_spring.fill_(wp.inf)
    d.qfrc_passive.fill_(wp.inf)
    mjw.forward(m, d)
    plastic = d.plastic.numpy()
    qfrc_spring = d.qfrc_spring.numpy()
    qfrc_passive = d.qfrc_passive.numpy()
    qpos = d.qpos.numpy()
    qvel = d.qvel.numpy()
    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])
      _assert_eq(plastic[w], mjds[w].plastic, f"snh_plastic_w{w}")
      _assert_eq(qfrc_spring[w], mjds[w].qfrc_spring, f"snh_qfrc_spring_w{w}")
      _assert_eq(qfrc_passive[w], mjds[w].qfrc_passive, f"snh_qfrc_passive_w{w}")
      _assert_eq(qpos[w], mjds[w].qpos, f"snh_qpos_w{w}")
      _assert_eq(qvel[w], mjds[w].qvel, f"snh_qvel_w{w}")

    if nworld == 2:
      self.assertFalse(np.allclose(plastic[0], plastic[1]))

  @parameterized.product(nworld=(1, 2), integrator=("Euler", "discrete"))
  def test_flex_plastic_stretch_bend_yield(self, nworld, integrator):
    """Verify separate stretching and bending yield strains of 2D flexes match MuJoCo."""
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option gravity="0 0 0" timestep="0.005" integrator="{integrator}"/>
        <worldbody>
          <flexcomp name="shell0" type="direct" dim="2" mass="1"
                    point="0 0 0  1 0 0  0.5 0.5 0  0.5 -0.5 0"
                    element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="both" damping="0.05" yield="1 0.002"/>
            <pin id="0 1 2"/>
          </flexcomp>
          <flexcomp name="shell1" type="direct" dim="2" mass="1"
                    point="2 0 0  3 0 0  2.5 0.5 0  2.5 -0.5 0"
                    element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="both" damping="0.05" yield="0.002 1"/>
            <pin id="0 1 2"/>
          </flexcomp>
          <flexcomp name="shell2" type="direct" dim="2" mass="1"
                    point="4 0 0  5 0 0  4.5 0.5 0  4.5 -0.5 0"
                    element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="both" damping="0.05" yield="0 0.002"/>
            <pin id="0 1 2"/>
          </flexcomp>
          <flexcomp name="shell3" type="direct" dim="2" mass="1"
                    point="6 0 0  7 0 0  6.5 0.5 0  6.5 -0.5 0"
                    element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="both" damping="0.05" yield="0.002 0"/>
            <pin id="0 1 2"/>
          </flexcomp>
          <flexcomp name="shell4" type="direct" dim="2" mass="1"
                    point="8 0 0  9 0 0  8.5 0.5 0  8.5 -0.5 0"
                    element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="both" damping="0.05" yield="1 1"/>
            <pin id="0 1 2"/>
          </flexcomp>
        </worldbody>
        <keyframe>
          <key qpos="0 -0.15 0.25  0 -0.15 0.25  0 -0.15 0.25  0 -0.15 0.25  0 -0.15 0.25"
               qvel="0 -0.3 0.4  0 -0.3 0.4  0 -0.3 0.4  0 -0.3 0.4  0 -0.3 0.4"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      nworld=nworld,
    )

    # Zero yield strains at runtime on shell4 produce full plastic flow in both modes.
    np.testing.assert_array_equal(mjm.flex_plasticmode, [3, 3, 1, 2, 3])
    self.assertEqual(mjm.nplastic, 44)  # 11 + 11 + 5 + 6 + 11
    mjm.flex_yield[4] = 0.0
    m.flex_yield.assign(mjm.flex_yield)

    mjds = [copy.deepcopy(mjd) for _ in range(nworld)]
    if nworld == 2:
      qvel = d.qvel.numpy().copy()
      mjds[1].qvel[:] = -1.2 * mjds[1].qvel
      mujoco.mj_forward(mjm, mjds[1])
      qvel[1] = mjds[1].qvel
      d.qvel.assign(qvel)
      mjw.forward(m, d)

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    for _ in range(10):
      wp.capture_launch(capture.graph)
    for w in range(nworld):
      mujoco.mj_step(mjm, mjds[w], 10)

    d.qfrc_spring.fill_(wp.inf)
    mjw.forward(m, d)
    plastic_all = d.plastic.numpy()
    qfrc_spring = d.qfrc_spring.numpy()
    qpos = d.qpos.numpy()
    qvel = d.qvel.numpy()
    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])
      _assert_eq(plastic_all[w], mjds[w].plastic, f"plastic_w{w}")
      _assert_eq(qfrc_spring[w], mjds[w].qfrc_spring, f"qfrc_spring_w{w}")
      _assert_eq(qpos[w], mjds[w].qpos, f"qpos_w{w}")
      _assert_eq(qvel[w], mjds[w].qvel, f"qvel_w{w}")

    # Each shell yields only in modes with a small threshold.
    plastic = plastic_all[0]
    stretch_yields = []
    bend_yields = []
    for f in range(mjm.nflex):
      adr = mjm.flex_plasticadr[f]
      mode = mjm.flex_plasticmode[f]
      nstretch = 6 if mode & 2 else 0
      nbend = 5 if mode & 1 else 0
      stretch_yields.append(bool(np.any(plastic[adr : adr + nstretch] != 0)))
      bend_yields.append(bool(np.any(plastic[adr + nstretch : adr + nstretch + nbend] != 0)))
    self.assertEqual(stretch_yields, [False, True, False, True, True])
    self.assertEqual(bend_yields, [True, False, True, False, True])
    if nworld == 2:
      self.assertFalse(np.allclose(plastic_all[0], plastic_all[1]))

  def test_flex_2d_stretch_compression_vs_tension(self):
    """Verify 2D stretch plasticity stays elastic under compression and yields under tension."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" timestep="0"/>
        <worldbody>
          <flexcomp name="tri" type="direct" dim="2" mass="1"
                    point="0 0 0  1 0 0  0.5 0.8660254 0" element="0 1 2">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="1000" poisson="0.3" thickness="0.02" elastic2d="stretch" yield="0.05"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
    )

    rest = mjd.flexvert_xpos.copy()
    # Equibiaxial compression stays elastic (no buckling flow).
    for v in range(3):
      adr = mjm.body_dofadr[mjm.flex_vertbodyid[v]]
      mjd.qpos[adr : adr + 3] = -0.15 * rest[v]
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    p_comp = d.plastic.numpy()[0]
    _assert_eq(p_comp, np.zeros(3), "compression_elastic")
    _assert_eq(p_comp, mjd.plastic, "compression_parity")

    # Equibiaxial tension beyond yield produces positive plastic elongation.
    mujoco.mj_resetData(mjm, mjd)
    for v in range(3):
      adr = mjm.body_dofadr[mjm.flex_vertbodyid[v]]
      mjd.qpos[adr : adr + 3] = 0.15 * rest[v]
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    p_tens = d.plastic.numpy()[0]
    _assert_eq(p_tens, mjd.plastic, "tension_parity")
    self.assertTrue(np.all(p_tens > 0.05))

  def test_flex_2d_bend_edge_equality_and_scale_invariance(self):
    """Verify paper-like edge-equality bending and scale invariance of curved shell bending."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" timestep="0.005"/>
        <worldbody>
          <flexcomp name="paper" type="direct" dim="2" mass="1"
                    point="0 0 0  1 0 0  0.5 0.5 0  0.5 -0.5 0"
                    element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="bend" damping="0.05" yield="0.002"/>
            <edge equality="true"/>
            <pin id="0 1 2"/>
          </flexcomp>
        </worldbody>
        <keyframe>
          <key qpos="0 0 0.25" qvel="0 0 0.4"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
    )

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    for _ in range(10):
      wp.capture_launch(capture.graph)
    mujoco.mj_step(mjm, mjd, 10)
    plastic = d.plastic.numpy()[0]
    _assert_eq(plastic, mjd.plastic, "edge_eq_bend_plastic")
    _assert_eq(d.qpos.numpy()[0], mjd.qpos, "edge_eq_bend_qpos")
    _assert_eq(d.qvel.numpy()[0], mjd.qvel, "edge_eq_bend_qvel")
    self.assertGreater(np.linalg.norm(plastic), 1e-3)
    capture = None
    gc.collect()

    # Folding a curved diamond by the same angle yields the same plastic state across length scales.
    plastics = []
    for s in (0.1, 1.0, 10.0):
      mjm_s, mjd_s, m_s, d_s = test_data.fixture(
        xml=f"""
        <mujoco>
          <option gravity="0 0 0" timestep="0"/>
          <worldbody>
            <flexcomp name="shell" type="direct" dim="2" mass="1"
                      point="0 0 0  {s} 0 0  {0.3 * s} {0.45 * s} {0.15 * s}  {0.65 * s} {-0.5 * s} {0.2 * s}"
                      element="0 1 2  0 3 1">
              <contact contype="0" conaffinity="0" selfcollide="none"/>
              <elasticity young="1000" poisson="0.3" thickness="{0.02 * s}" elastic2d="bend" yield="0.002"/>
              <pin id="0 1 2"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
      )
      y, z, a = -0.5 * s, 0.2 * s, 0.3
      mjd_s.qpos[1] = y * np.cos(a) - z * np.sin(a) - y
      mjd_s.qpos[2] = y * np.sin(a) + z * np.cos(a) - z
      mujoco.mj_forward(mjm_s, mjd_s)
      d_s = mjw.put_data(mjm_s, mjd_s)
      mujoco.mj_step(mjm_s, mjd_s)
      mjw.step(m_s, d_s)
      p_s = d_s.plastic.numpy()[0].copy()
      _assert_eq(p_s, mjd_s.plastic, f"curved_scale_{s}")
      plastics.append(p_s)

    p1_norm = np.linalg.norm(plastics[1])
    self.assertGreater(p1_norm, 0.0)
    np.testing.assert_allclose(plastics[0], plastics[1], atol=1e-4 * p1_norm, rtol=1e-4)
    np.testing.assert_allclose(plastics[2], plastics[1], atol=1e-4 * p1_norm, rtol=1e-4)

  def test_flex_yield_absent_mode_no_state(self):
    """Verify positive yield on an absent elastic mode allocates no plastic state."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <flexcomp name="shell_stretch" type="direct" dim="2" mass="1"
                    point="0 0 0  1 0 0  0.5 0.5 0" element="0 1 2">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="1000" poisson="0.3" thickness="0.02" elastic2d="stretch" yield="0 0.1"/>
          </flexcomp>
          <flexcomp name="shell_bend" type="direct" dim="2" mass="1"
                    point="2 0 0  3 0 0  2.5 0.5 0  2.5 -0.5 0" element="0 1 2  0 3 1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="1000" poisson="0.3" thickness="0.02" elastic2d="bend" yield="0.1 0"/>
          </flexcomp>
          <flexcomp name="tet" type="direct" dim="3" mass="1"
                    point="4 0 0  5 0 0  4 1 0  4 0 1" element="0 1 2 3">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="1000" poisson="0.25" yield="0.05"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
    )

    self.assertEqual(m.nplastic, 6)
    np.testing.assert_array_equal(m.flex_plasticadr.numpy(), [-1, -1, 0])
    np.testing.assert_array_equal(m.flex_plasticmode.numpy(), [0, 0, 2])
    _assert_eq(m.flex_yield.numpy()[2], np.array([0.05, 0.0]), "tet_yield")
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    _assert_eq(d.plastic.numpy()[0], mjd.plastic, "absent_mode_step")

  @parameterized.parameters(0, 1)
  def test_flex_3d_stretch_invariants(self, elastic3d):
    """Verify hydrostatic zero-flow and exact nonlinear volume preservation for 3D stretch."""
    integrator = "discrete" if elastic3d == 1 else "Euler"
    spec = mujoco.MjSpec.from_string(f"""
    <mujoco>
      <option gravity="0 0 0" timestep="0" integrator="{integrator}"/>
      <worldbody>
        <flexcomp name="tet" type="direct" dim="3" mass="1"
                  point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
          <contact contype="0" conaffinity="0" selfcollide="none"/>
          <elasticity young="1000" poisson="0.25" yield="0.05"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """)
    spec.flexes[0].elastic3d = elastic3d
    mjm = spec.compile()
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    rest = mjd.flexvert_xpos.copy()

    # Pure hydrostatic expansion produces zero plastic flow.
    for v in range(4):
      adr = mjm.body_dofadr[mjm.flex_vertbodyid[v]]
      mjd.qpos[adr : adr + 3] = 0.2 * rest[v]
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    p_hydro = d.plastic.numpy()[0]
    _assert_eq(p_hydro, np.zeros(6), f"hydrostatic_e{elastic3d}")
    _assert_eq(p_hydro, mjd.plastic, f"hydrostatic_parity_e{elastic3d}")

    # Large shear yields with exact Gram-determinant volume preservation det(G(g+sp)) == det(G(g)).
    mujoco.mj_resetData(mjm, mjd)
    adr3 = mjm.body_dofadr[mjm.flex_vertbodyid[3]]
    mjd.qpos[adr3 : adr3 + 3] = [0.5, 0.4, -0.1]
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    sp = d.plastic.numpy()[0]
    _assert_eq(sp, mjd.plastic, f"shear_parity_e{elastic3d}")
    self.assertGreater(np.linalg.norm(sp), 0.1)

    g = np.array([mjm.flexedge_length0[mjm.flex_elemedge[e]] ** 2 for e in range(6)])
    q = g + sp

    def _gram_det(v):
      a, b, c = v[0], v[2], v[4]
      d01 = 0.5 * (v[0] + v[2] - v[1])
      d02 = 0.5 * (v[0] + v[4] - v[5])
      d12 = 0.5 * (v[2] + v[4] - v[3])
      return np.linalg.det(np.array([[a, d01, d02], [d01, b, d12], [d02, d12, c]]))

    _assert_eq(_gram_det(q), _gram_det(g), f"volume_preservation_e{elastic3d}")

  @parameterized.product(nworld=(1, 2), integrator=("Euler", "RK4", "discrete"))
  def test_flex_interp_plastic(self, nworld, integrator):
    """Verify plastic 2D and 3D non-cubic interpolated flexes match MuJoCo across integrators."""
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option gravity="0 0 0" timestep="0.005" integrator="{integrator}"/>
        <worldbody>
          <flexcomp name="box" type="grid" dim="3" dof="trilinear"
                    count="2 2 2" cellcount="1 1 1" spacing="2 1 0.5" mass="1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.25" damping="0.05" yield="0.04"/>
            <pin id="0 1 2 3"/>
          </flexcomp>
          <flexcomp name="shell" type="grid" dim="3" pos="3 0 0" dof="trilinear"
                    count="2 2 2" cellcount="1 1 1" spacing="2 1 0.5" mass="1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="500" poisson="0.3" thickness="0.02" elastic2d="both" yield="0.04"/>
            <pin id="0 1 2 3"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    self.assertEqual(m.nplastic, 24)  # 6 (3D cell) + 18 (6 boundary faces * 3)
    idx_q = np.arange(mjm.nq, dtype=np.float64)
    idx_v = np.arange(mjm.nv, dtype=np.float64)
    mjds = [copy.deepcopy(mjd) for _ in range(nworld)]
    qpos_all = np.zeros((nworld, mjm.nq), dtype=np.float32)
    qvel_all = np.zeros((nworld, mjm.nv), dtype=np.float32)
    for w in range(nworld):
      phase = 1.0 + 0.7 * w
      mjds[w].qpos[:] = 0.2 * np.sin(0.9 * idx_q + phase)
      mjds[w].qvel[:] = 0.25 * np.cos(1.1 * idx_v + phase)
      mujoco.mj_forward(mjm, mjds[w])
      qpos_all[w] = mjds[w].qpos
      qvel_all[w] = mjds[w].qvel
    d.qpos.assign(qpos_all)
    d.qvel.assign(qvel_all)
    mjw.forward(m, d)

    with wp.ScopedCapture() as capture:
      mjw.step(m, d)
    for _ in range(10):
      wp.capture_launch(capture.graph)
    for w in range(nworld):
      mujoco.mj_step(mjm, mjds[w], 10)

    d.qfrc_spring.fill_(wp.inf)
    d.qfrc_passive.fill_(wp.inf)
    mjw.forward(m, d)
    plastic = d.plastic.numpy()
    qfrc_spring = d.qfrc_spring.numpy()
    qfrc_passive = d.qfrc_passive.numpy()
    qpos = d.qpos.numpy()
    qvel = d.qvel.numpy()
    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])
      _assert_eq(plastic[w], mjds[w].plastic, f"interp_plastic_w{w}")
      _assert_eq(qfrc_spring[w], mjds[w].qfrc_spring, f"interp_qfrc_spring_w{w}")
      _assert_eq(qfrc_passive[w], mjds[w].qfrc_passive, f"interp_qfrc_passive_w{w}")
      _assert_eq(qpos[w], mjds[w].qpos, f"interp_qpos_w{w}")
      _assert_eq(qvel[w], mjds[w].qvel, f"interp_qvel_w{w}")

    if nworld == 2:
      self.assertFalse(np.allclose(plastic[0], plastic[1]))

  def test_flex_interp_plastic_invariants(self):
    """Verify hydrostatic zero-flow, J2 trace, rigid rotation invariance, and 0.35 saturation."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" timestep="0"/>
        <worldbody>
          <flexcomp name="box" type="grid" dim="3" dof="trilinear"
                    count="2 2 2" cellcount="1 1 1" spacing="2 1 0.5" mass="1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="1000" poisson="0.25" damping="0.001" yield="0.05"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
    )

    rest = mjm.flex_node0.reshape(-1, 3).copy()

    # Pure hydrostatic expansion produces zero plastic strain.
    for i in range(mjm.nflexnode):
      adr = mjm.body_dofadr[mjm.flex_nodebodyid[i]]
      mjd.qpos[adr : adr + 3] = 0.2 * rest[i]
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    _assert_eq(d.plastic.numpy()[0], np.zeros(6), "interp_hydrostatic")

    # Combined stretch and shear yields with zero-trace J2 plastic strain and reduced force.
    mujoco.mj_resetData(mjm, mjd)
    for i in range(mjm.nflexnode):
      adr = mjm.body_dofadr[mjm.flex_nodebodyid[i]]
      dx0, dy0, dz0 = rest[i] - rest[0]
      mjd.qpos[adr + 0] = 0.1 * dy0 + 0.08 * dz0
      mjd.qpos[adr + 1] = 0.1 * dx0 + 0.06 * dz0
      mjd.qpos[adr + 2] = 0.2 * dz0 + 0.08 * dx0 + 0.06 * dy0
    mujoco.mj_forward(mjm, mjd)
    f_before = np.linalg.norm(mjd.qfrc_spring)
    d = mjw.put_data(mjm, mjd)
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    p = d.plastic.numpy()[0].copy()
    _assert_eq(p, mjd.plastic, "interp_yield_parity")
    self.assertGreater(np.linalg.norm(p), 0.05)
    self.assertLess(abs(p[0] + p[1] + p[2]), 1e-5)

    d.qfrc_spring.fill_(wp.inf)
    mjw.forward(m, d)
    self.assertLess(np.linalg.norm(d.qfrc_spring.numpy()[0]), f_before)

    # Exact plastic rest configuration under an arbitrary 3D rigid rotation produces zero force.
    axis = np.array([1.0, 2.0, -1.5])
    axis /= np.linalg.norm(axis)
    quat_rot = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat_rot, axis, 0.7)
    for i in range(mjm.nflexnode):
      dx0 = rest[i] - rest[0]
      x_p = np.array(
        [
          rest[i, 0] + p[0] * dx0[0] + 0.5 * p[3] * dx0[1] + 0.5 * p[4] * dx0[2],
          rest[i, 1] + 0.5 * p[3] * dx0[0] + p[1] * dx0[1] + 0.5 * p[5] * dx0[2],
          rest[i, 2] + 0.5 * p[4] * dx0[0] + 0.5 * p[5] * dx0[1] + p[2] * dx0[2],
        ]
      )
      x_rot = np.zeros(3)
      mujoco.mju_rotVecQuat(x_rot, x_p, quat_rot)
      adr = mjm.body_dofadr[mjm.flex_nodebodyid[i]]
      mjd.qpos[adr : adr + 3] = x_rot - rest[i]
    mujoco.mj_forward(mjm, mjd)
    d.qpos.assign(mjd.qpos.reshape(1, -1))
    d.qfrc_spring.fill_(wp.inf)
    mjw.forward(m, d)
    qfrc_rot = d.qfrc_spring.numpy()[0]
    _assert_eq(qfrc_rot, mjd.qfrc_spring, "interp_rotated_rest_parity")
    self.assertLess(np.linalg.norm(qfrc_rot), 1e-3)

    # Extreme deformation clamps Frobenius norm of plastic strain to 0.35.
    mujoco.mj_resetData(mjm, mjd)
    for i in range(mjm.nflexnode):
      adr = mjm.body_dofadr[mjm.flex_nodebodyid[i]]
      dx0, dy0, dz0 = rest[i] - rest[0]
      mjd.qpos[adr + 0] = 1.5 * dy0
      mjd.qpos[adr + 1] = 1.5 * dx0
      mjd.qpos[adr + 2] = 0.8 * dz0
    mujoco.mj_forward(mjm, mjd)
    d = mjw.put_data(mjm, mjd)
    mujoco.mj_step(mjm, mjd)
    mjw.step(m, d)
    p_sat = d.plastic.numpy()[0]
    _assert_eq(p_sat, mjd.plastic, "interp_saturation_parity")
    fnorm = np.sqrt(p_sat[0] ** 2 + p_sat[1] ** 2 + p_sat[2] ** 2 + 0.5 * (p_sat[3] ** 2 + p_sat[4] ** 2 + p_sat[5] ** 2))
    _assert_eq(fnorm, 0.35, "interp_saturation_norm")


if __name__ == "__main__":
  wp.init()
  absltest.main()
