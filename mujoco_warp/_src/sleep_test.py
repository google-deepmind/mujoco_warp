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

"""Tests for sleep and wake features."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp._src import sleep
from mujoco_warp._src import types
from mujoco_warp._src.types import SleepState

wp.set_module_options({"enable_backward": False})


class SleepTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    io.ENABLE_ISLANDS = True

  def tearDown(self):
    io.ENABLE_ISLANDS = False
    super().tearDown()

  @parameterized.parameters(1, 2)
  def test_sleep_initiation(self, nworld):
    """Verify that a stationary body on a flat plane goes to sleep."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sleep_tolerance="0.01">
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="10 10 .1"/>
          <body name="box" pos="0 0 0.1">
            <joint type="free"/>
            <geom type="box" size=".1 .1 .1" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Set initial state: close to the ground, with zero velocity
    qpos = d.qpos.numpy()
    for w in range(nworld):
      qpos[w, 2] = 0.1  # sit on floor
    d.qpos = wp.array(qpos, dtype=float)

    # Run a few steps to let it come to rest and count down
    for _ in range(15):
      mjwarp.step(m, d)

    # After 15 steps under sleep tolerance, it should have gone to sleep
    tree_asleep = d.tree_asleep.numpy()
    tree_awake = d.tree_awake.numpy()
    body_awake = d.body_awake.numpy()

    for w in range(nworld):
      # The box tree ID is 0. Since it's asleep, it should be in a cycle (value >= 0)
      # Since there's only 1 tree, it should cycle to itself: tree_asleep[w, 0] == 0
      self.assertEqual(tree_asleep[w, 0], 0, f"Expected self-cycle (0) for tree 0 in world {w}, got {tree_asleep[w, 0]}")
      self.assertEqual(tree_awake[w, 0], 0, f"Tree in world {w} should be asleep")
      self.assertEqual(
        body_awake[w, 1], SleepState.ASLEEP, f"Body in world {w} should be asleep"
      )  # body 1 is the box, 0 is world

      # Dof velocity and acceleration should be zeroed
      qvel = d.qvel.numpy()[w]
      qacc = d.qacc.numpy()[w]
      self.assertTrue((qvel == 0.0).all(), f"World {w} qvel not zeroed")
      self.assertTrue((qacc == 0.0).all(), f"World {w} qacc not zeroed")

  @parameterized.parameters(1, 2)
  def test_collision_waking(self, nworld):
    """Verify that a moving body colliding with a sleeping body wakes it up."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="10 10 .1"/>
          <body name="target" pos="0 0 0.1">
            <joint type="free"/>
            <geom type="sphere" size=".1" mass="1.0"/>
          </body>
          <body name="bullet" pos="-0.3 0 0.1">
            <joint type="free"/>
            <geom type="sphere" size=".1" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Set initial state: target is sitting, bullet is moving towards target
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 6] = 20.0  # high bullet velocity along x to ensure collision within few steps
    d.qvel = wp.array(qvel, dtype=float)

    # Put target (tree 0) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 0] = 0  # self-cycle for target
    d.tree_asleep = wp.array(tree_asleep, dtype=int)

    # Update sleep arrays to reflect the target is asleep
    sleep.update_sleep(m, d)
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 0, f"Target tree in world {w} should start asleep")

    # Run physics until contact occurs
    for _ in range(5):
      mjwarp.step(m, d)

    # Target should be wake up because of contact!
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 1, f"Target tree in world {w} should have woken up upon contact")
      self.assertLess(d.tree_asleep.numpy()[w, 0], 0, f"Target tree_asleep in world {w} should be negative (awake)")

  @parameterized.parameters(1, 2)
  def test_tendon_waking(self, nworld):
    """Verify that pulling an awake body wakes up a connected sleeping body through a tendon."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint name="j1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint name="j2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="ten" limited="true" range="0 0.9">
            <joint joint="j1" coef="1.0"/>
            <joint joint="j2" coef="-1.0"/>
          </fixed>
        </tendon>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Put b2 (tree 1) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 1] = 1  # self-cycle for b2
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    # Force b1 (tree 0) to move down, which will pull on b2 via tendon limit
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 0] = -5.0  # b1 moving down
    d.qvel = wp.array(qvel, dtype=float)

    # Run steps to trigger waking
    mjwarp.step(m, d)
    mjwarp.step(m, d)

    # b2 (tree 1) should be woken up by tendon pull!
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 1], 1, f"b2 in world {w} should have woken up due to tendon constraint")

  @parameterized.parameters(1, 2)
  def test_equality_waking(self, nworld):
    """Verify that moving an awake body wakes up a connected sleeping body (weld equality)."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint type="free"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint type="free"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="b1" body2="b2"/>
        </equality>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Put b2 (tree 1) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 1] = 1  # self-cycle for b2
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    # Force b1 (tree 0) to move
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 0] = 2.0  # b1 moving
    d.qvel = wp.array(qvel, dtype=float)

    # Run step
    mjwarp.step(m, d)

    # b2 (tree 1) should be woken up by equality weld!
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 1], 1, f"b2 in world {w} should have woken up due to weld constraint")

  @parameterized.parameters(1, 2)
  def test_waking_unaffected_by_sleeping(self, nworld):
    """Verify that sleeping trees do not affect the physical rollout trajectories of awake trees."""
    xml = """
    <mujoco>
      <option sleep_tolerance="0.01">
        <flag sleep="enable" island="enable"/>
      </option>
      <worldbody>
        <geom type="plane" size="10 10 .1"/>
        <body name="b1" pos="0 0 0.1">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" mass="1.0"/>
        </body>
        <body name="b2" pos="1 0 0.5">
          <joint type="free"/>
          <geom type="box" size=".1 .1 .1" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    # 1. Run with sleep enabled
    _, _, m_sleep, d_sleep = test_data.fixture(
      xml=xml,
      nworld=nworld,
    )

    # Place b1 exactly on the ground so it sleeps
    qpos_sleep = d_sleep.qpos.numpy()
    for w in range(nworld):
      qpos_sleep[w, 2] = 0.1  # b1 sitting
    d_sleep.qpos = wp.array(qpos_sleep, dtype=float)

    # 2. Run with sleep disabled
    _, _, m_nosleep, d_nosleep = test_data.fixture(
      xml=xml,
      nworld=nworld,
    )

    qpos_nosleep = d_nosleep.qpos.numpy()
    for w in range(nworld):
      qpos_nosleep[w, 2] = 0.1  # b1 sitting
    d_nosleep.qpos = wp.array(qpos_nosleep, dtype=float)

    # Step both simulators in parallel
    for step_idx in range(25):
      mjwarp.step(m_sleep, d_sleep)
      mjwarp.step(m_nosleep, d_nosleep)

      # b2 (tree 1) must remain awake in both simulations and have identical states
      qpos_s = d_sleep.qpos.numpy()
      qpos_ns = d_nosleep.qpos.numpy()
      qvel_s = d_sleep.qvel.numpy()
      qvel_ns = d_nosleep.qvel.numpy()

      for w in range(nworld):
        # Verify that b2's position and velocity matches exactly (to 6 decimal places)
        np.testing.assert_allclose(qpos_s[w, 7:14], qpos_ns[w, 7:14], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(qvel_s[w, 6:12], qvel_ns[w, 6:12], rtol=1e-6, atol=1e-6)

    # Verify that b1 indeed went to sleep in d_sleep
    for w in range(nworld):
      self.assertEqual(d_sleep.tree_awake.numpy()[w, 0], 0, f"b1 in world {w} should have gone to sleep")

  @parameterized.parameters(1, 2)
  def test_settle_zero_velocity(self, nworld):
    """Verify that a moving body in zero-gravity with damping goes to sleep.

    It must settle and have exactly zero velocity.
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" sleep_tolerance="0.01">
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="box" pos="0 0 0">
            <joint type="slide" axis="1 0 0" damping="400.0"/>
            <geom type="box" size=".1 .1 .1" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Give initial velocity
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 0] = 5.0
    d.qvel = wp.array(qvel, dtype=float)

    # Run steps to let it settle
    for _ in range(25):
      mjwarp.step(m, d)

    # It must be asleep
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 0, f"Tree in world {w} should have gone to sleep")
      self.assertEqual(d.body_awake.numpy()[w, 1], SleepState.ASLEEP, f"Body in world {w} should have gone to sleep")

      # Velocity and acceleration must be exactly zero (bitwise)
      qvel_final = d.qvel.numpy()[w]
      qacc_final = d.qacc.numpy()[w]
      self.assertEqual(qvel_final[0], 0.0, f"World {w} velocity should be exactly zeroed")
      self.assertEqual(qacc_final[0], 0.0, f"World {w} acceleration should be exactly zeroed")

  @parameterized.parameters(1, 2)
  def test_slack_tendon_sleeping(self, nworld):
    """Verify that a slack tendon does not wake connected sleeping bodies."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint name="j1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint name="j2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="ten" limited="true" range="0 1.5">
            <joint joint="j1" coef="1.0"/>
            <joint joint="j2" coef="-1.0"/>
          </fixed>
        </tendon>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Put b2 (tree 1) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 1] = 1  # self-cycle for b2
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    # Force b1 (tree 0) to move slowly, tendon is well within range 0-1.5, so it is slack
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 0] = 0.1  # moving b1 slightly
    d.qvel = wp.array(qvel, dtype=float)

    # Run step
    mjwarp.step(m, d)

    # b2 (tree 1) must remain asleep because the tendon is slack!
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 1], 0, f"b2 in world {w} should remain asleep since tendon is slack")

  @parameterized.parameters(1, 2)
  def test_equality_sleep_stability(self, nworld):
    """Verify that weld-connected bodies stay asleep together without self-waking loop."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint type="free"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint type="free"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="b1" body2="b2"/>
        </equality>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Put both b1 and b2 to sleep manually in a cycle: tree_asleep[w, 0] = 1, tree_asleep[w, 1] = 0
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 0] = 1
      tree_asleep[w, 1] = 0
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 0, f"b1 in world {w} should start asleep")
      self.assertEqual(d.tree_awake.numpy()[w, 1], 0, f"b2 in world {w} should start asleep")

    # Run step
    mjwarp.step(m, d)

    # Both must remain asleep!
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 0, f"b1 in world {w} should remain asleep after a step")
      self.assertEqual(d.tree_awake.numpy()[w, 1], 0, f"b2 in world {w} should remain asleep after a step")

  def test_mj_wake_tree(self):
    """Direct unit test of cycle-traversal wake_tree function."""

    @wp.kernel(module="unique", enable_backward=False)
    def _test_wake_tree_kernel(
      # Model:
      ntree: int,
      # In:
      worldid: int,
      target_tree: int,
      wakeval: int,
      # Data out:
      tree_asleep_out: wp.array2d[int],
      # Out:
      woke_count_out: wp.array[int],
    ):
      woke_count_out[0] = sleep._wake_tree(ntree, worldid, target_tree, wakeval, tree_asleep_out)

    k_awake = sleep.K_AWAKE_VAL

    # Setup array: shape (1, 4)
    asleep_init = np.array([[k_awake, 2, 1, 3]], dtype=np.int32)
    tree_asleep = wp.array(asleep_init, dtype=int)
    woke_count = wp.zeros((1,), dtype=int)

    # Launch kernel to execute the wp.func on device
    wp.launch(
      _test_wake_tree_kernel,
      dim=1,
      inputs=[4, 0, 0, k_awake],
      outputs=[tree_asleep, woke_count],
    )
    self.assertEqual(woke_count.numpy()[0], 0)
    np.testing.assert_array_equal(tree_asleep.numpy(), asleep_init)

    # Wake tree 1 (part of cycle 1-2) -> woke count should be 2
    wp.launch(
      _test_wake_tree_kernel,
      dim=1,
      inputs=[4, 0, 1, k_awake],
      outputs=[tree_asleep, woke_count],
    )
    self.assertEqual(woke_count.numpy()[0], 2)
    np.testing.assert_array_equal(tree_asleep.numpy()[0], [k_awake, k_awake, k_awake, 3])

    # Wake tree 3 (self-cycle 3) -> woke count should be 1
    wp.launch(
      _test_wake_tree_kernel,
      dim=1,
      inputs=[4, 0, 3, k_awake],
      outputs=[tree_asleep, woke_count],
    )
    self.assertEqual(woke_count.numpy()[0], 1)
    np.testing.assert_array_equal(tree_asleep.numpy()[0], [k_awake, k_awake, k_awake, k_awake])

  @parameterized.parameters(1, 2)
  def test_multitree_tendon_waking(self, nworld):
    """Verify pulling wakes up multiple connected sleeping bodies through a single tendon."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint name="j1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint name="j2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b3" pos="2 0 0.5">
            <joint name="j3" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="ten" limited="true" range="0 0.9">
            <joint joint="j1" coef="1.0"/>
            <joint joint="j2" coef="-1.0"/>
            <joint joint="j3" coef="0.5"/>
          </fixed>
        </tendon>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Put both b2 (tree 1) and b3 (tree 2) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 1] = 1  # self-cycle for b2
      tree_asleep[w, 2] = 2  # self-cycle for b3
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 1, "b1 should start awake")
      self.assertEqual(d.tree_awake.numpy()[w, 1], 0, "b2 should start asleep")
      self.assertEqual(d.tree_awake.numpy()[w, 2], 0, "b3 should start asleep")

    # Force b1 (tree 0) to move down, which will pull on b2 and b3 via tendon limit
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 0] = -5.0  # b1 moving down
    d.qvel = wp.array(qvel, dtype=float)

    # Run two steps to trigger position-phase tendon limit waking
    mjwarp.step(m, d)
    mjwarp.step(m, d)

    # BOTH b2 and b3 should be woken up!
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 1], 1, "b2 should have woken up due to tendon constraint")
      self.assertEqual(d.tree_awake.numpy()[w, 2], 1, "b3 should have woken up due to tendon constraint")

  @parameterized.parameters(1, 2)
  def test_tendon_equality_waking(self, nworld):
    """Verify that tendon equality constraint wakes up connected sleeping bodies."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint name="j1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="0.5 0 0.5">
            <joint name="j2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b3" pos="1.0 0 0.5">
            <joint name="j3" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b4" pos="1.5 0 0.5">
            <joint name="j4" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="ten1">
            <joint joint="j1" coef="1.0"/>
            <joint joint="j2" coef="-1.0"/>
          </fixed>
          <fixed name="ten2">
            <joint joint="j3" coef="1.0"/>
            <joint joint="j4" coef="-1.0"/>
          </fixed>
        </tendon>
        <equality>
          <tendon tendon1="ten1" tendon2="ten2" polycoef="0 1 0 0 0"/>
        </equality>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Enable sleep programmatically
    mjm.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    m.opt.enableflags |= types.EnableBit.SLEEP

    # Put b3 (tree 2) and b4 (tree 3) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 2] = 2  # self-cycle for b3
      tree_asleep[w, 3] = 3  # self-cycle for b4
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 1, "b1 should start awake")
      self.assertEqual(d.tree_awake.numpy()[w, 1], 1, "b2 should start awake")
      self.assertEqual(d.tree_awake.numpy()[w, 2], 0, "b3 should start asleep")
      self.assertEqual(d.tree_awake.numpy()[w, 3], 0, "b4 should start asleep")

    # Force b1 (tree 0) to move down, which will affect ten1 and couple to ten2
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 0] = -5.0  # b1 moving down
    d.qvel = wp.array(qvel, dtype=float)

    # Run step
    mjwarp.step(m, d)

    # Both b3 and b4 should have woken up!
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 2], 1, "b3 should have woken up due to tendon equality constraint")
      self.assertEqual(d.tree_awake.numpy()[w, 3], 1, "b4 should have woken up due to tendon equality constraint")

  @parameterized.parameters(1, 2)
  def test_manual_velocity_wake(self, nworld):
    """Verify that manually setting a velocity on a sleeping body wakes it up."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sleep_tolerance="0.01">
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="10 10 .1"/>
          <body pos="0 0 0.1">
            <joint type="free"/>
            <geom type="box" size=".1 .1 .1" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Put tree 0 to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 0] = 0  # self-cycle for tree 0
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 0, f"Tree 0 in world {w} should start asleep")

    # Manually set the velocity of the box
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 0] = 0.5  # set velocity along x
    d.qvel = wp.array(qvel, dtype=float)

    # Run a step
    mjwarp.step(m, d)

    # The tree should be awake
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 1, f"Tree 0 in world {w} should have woken up because of non-zero velocity")

  @parameterized.parameters(1, 2)
  def test_sleep_rerun_collision(self, nworld):
    """Verify that newly awakened bodies detect collisions in the same step."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="10 10 .1"/>
          <body name="box" pos="0 0 0.09">
            <joint type="free"/>
            <geom name="box_geom" type="box" size=".1 .1 .1" mass="1.0"/>
          </body>
          <body name="sphere" pos="0 0 0.23">
            <joint type="free"/>
            <geom name="sphere_geom" type="sphere" size=".05" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Put box (tree 0) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 0] = 0  # self-cycle for box
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 0, f"Box tree in world {w} should start asleep")
      self.assertEqual(d.tree_awake.numpy()[w, 1], 1, f"Sphere tree in world {w} should start awake")

    # Set sphere velocity downwards so it hits the box
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 8] = -5.0  # Z velocity of the sphere
    d.qvel = wp.array(qvel, dtype=float)

    # Run a single simulation step
    mjwarp.step(m, d)

    # Box should be awake
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 1, f"Box in world {w} should be awake after collision")

    # We expect exactly 5 active contacts per world (4 box-plane corner contacts
    # + 1 sphere-box contact)
    nacon = d.nacon.numpy()[0]
    self.assertEqual(nacon, 5 * nworld, f"Expected {5 * nworld} active contacts across all worlds, but got {nacon}")

  @parameterized.parameters(1, 2)
  def test_sleep_policy_auto_never(self, nworld):
    """Verify that a tree with SleepPolicy.AUTO_NEVER never sleeps."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sleep_tolerance="0.01">
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="10 10 .1"/>
          <body pos="0 0 0.1">
            <joint name="hinge" type="hinge" axis="0 0 1"/>
            <geom type="box" size=".1 .1 .1" mass="1.0"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="hinge"/>
        </actuator>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Policy should be AUTO_NEVER (1)
    self.assertEqual(m.tree_sleep_policy.numpy()[0], int(types.SleepPolicy.AUTO_NEVER))

    # tree should never sleep
    for _ in range(15):
      mjwarp.step(m, d)

    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 1, f"Tree in world {w} with AUTO_NEVER policy should remain awake")

  def test_sleep_multi_world_decoupled(self):
    """Verify that sleeping states and cycles are decoupled and independent.

    They must be independent across different worlds.
    """
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sleep_tolerance="0.01">
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="10 10 .1"/>
          <body name="box" pos="0 0 0.1">
            <joint type="free"/>
            <geom type="box" size=".1 .1 .1" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
    )

    # World 0: awake, World 1: asleep
    # 1. Put tree 0 to sleep in World 1
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[1, 0] = 0  # self-cycle for tree 0 in world 1
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    # 2. Give World 0 a non-zero velocity
    qvel = d.qvel.numpy()
    qvel[0, 0] = 0.5  # velocity along x for world 0
    d.qvel = wp.array(qvel, dtype=float)

    # Verify initial states
    self.assertEqual(d.tree_awake.numpy()[0, 0], 1, "World 0 should start awake")
    self.assertEqual(d.tree_awake.numpy()[1, 0], 0, "World 1 should start asleep")

    # Step simulation
    mjwarp.step(m, d)

    # Verify that World 0 remained awake and World 1 remained asleep
    self.assertEqual(d.tree_awake.numpy()[0, 0], 1, "World 0 should remain awake")
    self.assertEqual(d.tree_awake.numpy()[1, 0], 0, "World 1 should remain asleep")

  def test_wake_collision_cross_world(self):
    """Verify that collisions in one world do not wake up islands in another world."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <geom name="floor" type="plane" size="10 10 .1"/>
          <body name="box1" pos="0 0 0.1">
            <joint type="free"/>
            <geom name="box1_geom" type="box" size=".1 .1 .1" mass="1.0"/>
          </body>
          <body name="box2" pos="1.5 0 0.1">
            <joint type="free"/>
            <geom name="box2_geom" type="box" size=".1 .1 .1" mass="1.0"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=2,
    )

    # World 0: box1 is awake, box2 is asleep. They are far apart (no collision).
    # World 1: box1 is awake, box2 is asleep. But they are close to each other (colliding!).
    qpos = d.qpos.numpy()

    # World 0 positions: box1 at (0, 0, 0.1), box2 at (1.5, 0, 0.1) -> no collision
    qpos[0, 0:3] = [0.0, 0.0, 0.1]
    qpos[0, 7:10] = [1.5, 0.0, 0.1]

    # World 1 positions: box1 at (0, 0, 0.1), box2 at (0.15, 0, 0.1) -> colliding!
    qpos[1, 0:3] = [0.0, 0.0, 0.1]
    qpos[1, 7:10] = [0.15, 0.0, 0.1]

    d.qpos = wp.array(qpos, dtype=float)

    # Manually put box2 (tree 1) to sleep in both worlds.
    # box1 (tree 0) is awake.
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[0, 0] = -1  # awake
    tree_asleep[0, 1] = 1  # asleep (self-cycle 1)

    tree_asleep[1, 0] = -1  # awake
    tree_asleep[1, 1] = 1  # asleep (self-cycle 1)

    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    # Verify starting states:
    self.assertEqual(d.tree_awake.numpy()[0, 0], 1, "World 0 box1 should start awake")
    self.assertEqual(d.tree_awake.numpy()[0, 1], 0, "World 0 box2 should start asleep")
    self.assertEqual(d.tree_awake.numpy()[1, 0], 1, "World 1 box1 should start awake")
    self.assertEqual(d.tree_awake.numpy()[1, 1], 0, "World 1 box2 should start asleep")

    # Step position-dependent phase (kinematics, collision, wake_collision, etc.)
    forward.fwd_position(m, d)

    # Verification:
    # World 1: box1 (awake) and box2 (asleep) collide. So box2 MUST wake up!
    self.assertEqual(d.tree_awake.numpy()[1, 1], 1, "World 1 box2 should wake up due to collision with box1")

    # World 0: box1 (awake) and box2 (asleep) do NOT collide. So box2 MUST remain asleep!
    self.assertEqual(d.tree_awake.numpy()[0, 1], 0, "World 0 box2 should remain asleep (no collision with box1)")


if __name__ == "__main__":
  absltest.main()
