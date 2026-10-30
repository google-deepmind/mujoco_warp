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

import copy

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp
from mujoco_warp import test_data
from mujoco_warp._src import forward
from mujoco_warp._src import island
from mujoco_warp._src import sleep
from mujoco_warp._src import smooth
from mujoco_warp._src import types
from mujoco_warp._src.types import SleepState

wp.set_module_options({"enable_backward": False})


class SleepTest(parameterized.TestCase):
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
    d.qpos.assign(qpos)

    with wp.ScopedCapture() as capture:
      mjwarp.step(m, d)

    # Run a few steps to let it come to rest and count down
    for _ in range(15):
      wp.capture_launch(capture.graph)

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
    d.qvel.assign(qvel)

    # Put target (tree 0) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 0] = 0  # self-cycle for target
    d.tree_asleep.assign(tree_asleep)

    # Update sleep arrays to reflect the target is asleep
    sleep.update_sleep(m, d)
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 0, f"Target tree in world {w} should start asleep")

    with wp.ScopedCapture() as capture:
      mjwarp.step(m, d)

    # Run physics until contact occurs
    for _ in range(5):
      wp.capture_launch(capture.graph)

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
          <geom name="floor" type="plane" size="5 5 .1"/>
          <body name="b1" pos="0 0 0.5">
            <joint name="j1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1"/>
          </body>
          <body name="b2" pos="1 0 0.08">
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
    d.tree_asleep.assign(tree_asleep)
    sleep.update_sleep(m, d)

    # Force b1 (tree 0) to move down, which will pull on b2 via tendon limit
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 0] = -5.0  # b1 moving down
    d.qvel.assign(qvel)

    # Capture and launch steps so graph-conditional tendon waking is exercised
    with wp.ScopedCapture() as capture:
      mjwarp.step(m, d)

    wp.capture_launch(capture.graph)
    wp.capture_launch(capture.graph)

    # b2 (tree 1, body 2) should be woken up by tendon pull, refreshing body state,
    # awake counts, and collision participation with the floor on the waking step
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 1], 1, f"b2 in world {w} should have woken up due to tendon constraint")
      self.assertEqual(d.body_awake.numpy()[w, 2], SleepState.AWAKE, f"b2 body_awake in world {w} should be AWAKE")
      self.assertEqual(d.nbody_awake.numpy()[w], 3, f"nbody_awake in world {w} should include world, b1, and b2")
      self.assertEqual(d.nv_awake.numpy()[w], 2, f"nv_awake in world {w} should include both DOFs")
    self.assertEqual(d.nacon.numpy()[0], nworld, "Tendon-woken b2 should participate in floor collision")

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
    # Run with sleep enabled
    _, _, m_sleep, d_sleep = test_data.fixture(
      xml=xml,
      nworld=nworld,
    )

    # Place b1 exactly on the ground so it sleeps
    qpos_sleep = d_sleep.qpos.numpy()
    for w in range(nworld):
      qpos_sleep[w, 2] = 0.1  # b1 sitting
    d_sleep.qpos.assign(qpos_sleep)

    # Run with sleep disabled
    _, _, m_nosleep, d_nosleep = test_data.fixture(
      xml=xml,
      nworld=nworld,
    )

    qpos_nosleep = d_nosleep.qpos.numpy()
    for w in range(nworld):
      qpos_nosleep[w, 2] = 0.1  # b1 sitting
    d_nosleep.qpos.assign(qpos_nosleep)

    with wp.ScopedCapture() as capture_sleep:
      mjwarp.step(m_sleep, d_sleep)
    with wp.ScopedCapture() as capture_nosleep:
      mjwarp.step(m_nosleep, d_nosleep)

    # Step both simulators in parallel
    for step_idx in range(25):
      wp.capture_launch(capture_sleep.graph)
      wp.capture_launch(capture_nosleep.graph)

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
    d.qvel.assign(qvel)

    with wp.ScopedCapture() as capture:
      mjwarp.step(m, d)

    # Run steps to let it settle
    for _ in range(25):
      wp.capture_launch(capture.graph)

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
    """Verify slack tendon with armature preserves sleeping M and wakes cleanly."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="b1" pos="0 0 0.5">
            <joint name="j1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1" mass="1"/>
          </body>
          <body name="b2" pos="1 0 0.5">
            <joint name="j2" type="slide" axis="0 0 1"/>
            <geom type="sphere" size=".1" mass="1"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="ten" limited="true" range="0 1.5" armature="2.0">
            <joint joint="j1" coef="1.0"/>
            <joint joint="j2" coef="-1.0"/>
          </fixed>
        </tendon>
      </mujoco>
      """,
      nworld=nworld,
    )

    m_expected = d.M.numpy().copy()

    # Put b2 (tree 1) to sleep manually
    tree_asleep = d.tree_asleep.numpy()
    for w in range(nworld):
      tree_asleep[w, 1] = 1  # self-cycle for b2
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)

    # Force b1 (tree 0) to move slowly; tendon is within range 0-1.5, so it is slack
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 0] = 0.1  # moving b1 slightly
    d.qvel = wp.array(qvel, dtype=float)

    # Repeated steps while b2 is asleep must not accumulate tendon_armature into M
    for _ in range(4):
      mjwarp.step(m, d)
      for w in range(nworld):
        self.assertEqual(d.tree_awake.numpy()[w, 1], 0, f"b2 in world {w} should remain asleep since tendon is slack")
      np.testing.assert_allclose(d.M.numpy(), m_expected, rtol=1e-6, atol=1e-6)

    # Pull b1 fast enough to hit the tendon limit and wake b2; M must still match expected
    qvel = d.qvel.numpy()
    for w in range(nworld):
      qvel[w, 0] = -10.0
    d.qvel = wp.array(qvel, dtype=float)
    mjwarp.step(m, d)
    mjwarp.step(m, d)
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 1], 1, f"b2 in world {w} should wake when tendon limit is reached")
    np.testing.assert_allclose(d.M.numpy(), m_expected, rtol=1e-6, atol=1e-6)

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

  def test_wake_tree_invalid_tree_id(self):
    """Verify that calling _wake_tree with an invalid tree ID returns 0."""

    @wp.kernel(module="unique", enable_backward=False)
    def _test_wake_tree_kernel(
      ntree: int,
      worldid: int,
      target_tree: int,
      wakeval: int,
      tree_asleep_out: wp.array2d[int],
      woke_count_out: wp.array[int],
    ):
      woke_count_out[0] = sleep._wake_tree(ntree, worldid, target_tree, wakeval, tree_asleep_out)

    k_awake = sleep.K_AWAKE_VAL
    asleep_init = np.array([[0, 1]], dtype=np.int32)
    tree_asleep = wp.array(asleep_init, dtype=int)
    woke_count = wp.zeros((1,), dtype=int)

    wp.launch(
      _test_wake_tree_kernel,
      dim=1,
      inputs=[2, 0, -1, k_awake],
      outputs=[tree_asleep, woke_count],
    )
    self.assertEqual(woke_count.numpy()[0], 0)
    np.testing.assert_array_equal(tree_asleep.numpy(), asleep_init)

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
      nvmax=4,
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

    with wp.ScopedCapture() as capture:
      mjwarp.step(m, d)

    # tree should never sleep
    for _ in range(15):
      wp.capture_launch(capture.graph)

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

  @parameterized.parameters(1, 2)
  def test_split_step_sparse_block_wake(self, nworld):
    """Verify split-step (step1/step2) factors a sparse-block tree woken during step1."""
    n = 65
    chain = (
      "".join(
        f'<body name="c{i}" pos="0 0 .05"><joint name="jc{i}" type="hinge" axis="0 1 0"/><geom type="capsule" size=".02 .025"/>'
        for i in range(n)
      )
      + "</body>" * n
    )
    _, _, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option>
          <flag sleep="enable" island="enable"/>
        </option>
        <worldbody>
          <body name="driver" pos="0 0 1">
            <joint name="jd" type="hinge" axis="0 1 0"/>
            <geom type="sphere" size=".05"/>
          </body>
          <body pos="1 0 2">
            {chain}
          </body>
        </worldbody>
        <equality>
          <joint joint1="jd" joint2="jc0"/>
        </equality>
      </mujoco>
      """,
      nworld=nworld,
    )
    self.assertGreater(d.qLD.shape[1], m.qLD_block_total)

    qvel = d.qvel.numpy()
    qvel[:, 0] = 2.0
    if nworld == 2:
      qvel[1, 0] = 3.5
    d.qvel = wp.array(qvel, dtype=float)

    # Reference without sleeping
    _, _, m_ref, d_ref = test_data.fixture(
      xml=f"""
      <mujoco>
        <worldbody>
          <body name="driver" pos="0 0 1">
            <joint name="jd" type="hinge" axis="0 1 0"/>
            <geom type="sphere" size=".05"/>
          </body>
          <body pos="1 0 2">
            {chain}
          </body>
        </worldbody>
        <equality>
          <joint joint1="jd" joint2="jc0"/>
        </equality>
      </mujoco>
      """,
      nworld=nworld,
    )
    d_ref.qvel = wp.array(qvel, dtype=float)

    # Put the 65-DOF chain (tree 1) to sleep; step1 wakes the sparse-block tree via wake_equality.
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[:, 1] = 1
    d.tree_asleep = wp.array(tree_asleep, dtype=int)
    sleep.update_sleep(m, d)
    d.qLD.fill_(wp.inf)
    d.qacc_smooth.fill_(wp.inf)
    d.qacc.fill_(wp.inf)

    mjwarp.step1(m, d)
    mjwarp.step2(m, d)

    mjwarp.step1(m_ref, d_ref)
    mjwarp.step2(m_ref, d_ref)

    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 1], 1)
    np.testing.assert_allclose(d.qacc_smooth.numpy(), d_ref.qacc_smooth.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(d.qacc.numpy(), d_ref.qacc.numpy(), rtol=1e-3, atol=1e-4)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.product(
    nworld=(1, 2),
    integrator=(types.IntegratorType.EULER, types.IntegratorType.IMPLICITFAST, types.IntegratorType.IMPLICIT),
  )
  def test_sleep_deterministic_atomics_sparse(self, nworld, integrator):
    """Verify sparse sleeping trees under DeterminismType.ATOMICS zero qacc and wake cleanly."""
    n = 65
    chain = (
      "\n".join(
        f'          <body name="c{i}" pos="0 0 .05">\n'
        f'            <joint name="jc{i}" type="hinge" axis="0 1 0" damping=".2"/>\n'
        f'            <geom type="capsule" size=".02 .025"/>'
        for i in range(n)
      )
      + "\n          "
      + "</body>" * n
    )
    mjm, _, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option gravity="0 0 0">
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <body name="driver" pos="0 0 1">
            <joint name="jd" type="hinge" axis="0 1 0" damping=".5"/>
            <geom size=".05"/>
          </body>
          <body pos="1 0 2">
{chain}
          </body>
        </worldbody>
        <equality>
          <joint joint1="jd" joint2="jc0" active="false"/>
        </equality>
      </mujoco>
      """,
      overrides={
        "opt.integrator": integrator,
        "opt.deterministic": types.DeterminismType.ATOMICS,
      },
      nworld=nworld,
    )
    self.assertGreater(d.qLD.shape[1], m.qLD_block_total)

    # Set distinct driver velocities per world while keeping the 65-DOF chain asleep
    qvel = np.zeros((nworld, mjm.nv), dtype=np.float32)
    qvel[0, 0] = 2.0
    if nworld == 2:
      qvel[1, 0] = 3.5
    d.qvel.assign(qvel)

    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[:, 1] = 1
    d.tree_asleep.assign(tree_asleep)
    sleep.update_sleep(m, d)

    # Corrupt factor/output buffers to ensure ATOMICS sparse solves skip and zero sleeping DOFs
    d.qLD.fill_(wp.inf)
    d.qLDiagInv.fill_(wp.inf)
    d.qacc_smooth.fill_(wp.inf)
    d.qacc.fill_(wp.inf)

    mjwarp.step(m, d)

    qacc_smooth = d.qacc_smooth.numpy()
    qacc = d.qacc.numpy()
    self.assertTrue(np.isfinite(qacc_smooth).all())
    self.assertTrue(np.isfinite(qacc).all())
    np.testing.assert_array_equal(qacc_smooth[:, 1:], 0.0)
    np.testing.assert_array_equal(qacc[:, 1:], 0.0)
    self.assertTrue(np.isfinite(d.qvel.numpy()).all())
    self.assertTrue(np.isfinite(d.qpos.numpy()).all())

    # Verify direct smooth.solve_m also zeroes sleeping sparse-block DOFs under ATOMICS
    x = wp.full((nworld, mjm.nv), wp.inf, dtype=float)
    y = wp.ones((nworld, mjm.nv), dtype=float)
    smooth.solve_m(m, d, x, y)
    x_np = x.numpy()
    self.assertTrue(np.isfinite(x_np).all())
    np.testing.assert_array_equal(x_np[:, 1:], 0.0)

    # Enable joint equality to wake the 65-DOF chain and verify parity against MuJoCo C
    _, _, m_ref, d_ref = test_data.fixture(
      xml=f"""
      <mujoco>
        <option gravity="0 0 0">
          <flag sleep="disable"/>
        </option>
        <worldbody>
          <body name="driver" pos="0 0 1">
            <joint name="jd" type="hinge" axis="0 1 0" damping=".5"/>
            <geom size=".05"/>
          </body>
          <body pos="1 0 2">
{chain}
          </body>
        </worldbody>
        <equality>
          <joint joint1="jd" joint2="jc0" active="true"/>
        </equality>
      </mujoco>
      """,
      overrides={
        "opt.integrator": integrator,
        "opt.deterministic": types.DeterminismType.ATOMICS,
      },
      nworld=nworld,
    )
    wp.copy(d_ref.qpos, d.qpos)
    wp.copy(d_ref.qvel, d.qvel)
    wp.copy(d_ref.qacc_warmstart, d.qacc_warmstart)
    mjwarp.step(m_ref, d_ref)

    mjds = []
    for w in range(nworld):
      mjd_w = mujoco.MjData(mjm)
      mjd_w.qpos[:] = d.qpos.numpy()[w]
      mjd_w.qvel[:] = d.qvel.numpy()[w]
      mjd_w.qacc_warmstart[:] = d.qacc_warmstart.numpy()[w]
      mjd_w.eq_active[:] = 1
      mujoco.mj_step(mjm, mjd_w)
      mjds.append(mjd_w)

    d.eq_active.fill_(True)
    d.qacc_smooth.fill_(wp.inf)
    d.qacc.fill_(wp.inf)
    mjwarp.step(m, d)

    np.testing.assert_array_equal(d.qacc_smooth.numpy(), d_ref.qacc_smooth.numpy())
    np.testing.assert_array_equal(d.qacc.numpy(), d_ref.qacc.numpy())
    np.testing.assert_array_equal(d.qvel.numpy(), d_ref.qvel.numpy())
    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 1], 1)
      np.testing.assert_allclose(d.qacc_smooth.numpy()[w], mjds[w].qacc_smooth, rtol=1e-4, atol=1e-5)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.product(
    nworld=(1, 2),
    integrator=(types.IntegratorType.EULER, types.IntegratorType.IMPLICITFAST, types.IntegratorType.IMPLICIT),
  )
  def test_sleep_unconstrained_and_implicit_zero_m(self, nworld, integrator):
    """Verify njmax == 0 models and zero-initialized d.M on sleeping trees step cleanly."""
    mjm, _, m, _ = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0">
          <flag sleep="enable" contact="disable"/>
        </option>
        <worldbody>
          <body name="asleep" pos="0 0 1">
            <joint type="hinge" damping=".5"/>
            <geom size=".1" mass="100"/>
          </body>
          <body name="awake" pos="1 0 1">
            <joint type="hinge" damping=".5"/>
            <geom size=".1" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.integrator": integrator},
      njmax=0,
      nworld=nworld,
    )

    # Fresh make_data + reset_data leaves d.M == 0 before any awake crb pass runs
    d = mjwarp.make_data(mjm, nworld=nworld, njmax=0)
    mjwarp.reset_data(m, d)
    self.assertEqual(d.njmax, 0)
    np.testing.assert_array_equal(d.M.numpy(), 0.0)

    # Put tree 0 to sleep before d.M is ever populated on tree 0
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[:, 0] = 0
    d.tree_asleep.assign(tree_asleep)
    sleep.update_sleep(m, d)

    qvel = np.zeros((nworld, mjm.nv), dtype=np.float32)
    qvel[0, 1] = 1.5
    if nworld == 2:
      qvel[1, 1] = 2.5
    d.qvel.assign(qvel)

    d.qacc.fill_(wp.inf)
    d.qfrc_constraint.fill_(wp.inf)
    d.efc.Ma.fill_(wp.inf)
    d.solver_niter.fill_(-1)

    mjwarp.step(m, d)

    for w in range(nworld):
      mjd_w = mujoco.MjData(mjm)
      mjd_w.qvel[1] = qvel[w, 1]
      mujoco.mj_step(mjm, mjd_w)
      self.assertEqual(d.tree_awake.numpy()[w, 0], 0)
      self.assertEqual(d.solver_niter.numpy()[w], 0)
      np.testing.assert_array_equal(d.qfrc_constraint.numpy()[w], 0.0)
      np.testing.assert_allclose(d.efc.Ma.numpy()[w], d.qfrc_smooth.numpy()[w], rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(d.qacc.numpy()[w], mjd_w.qacc, rtol=1e-5, atol=1e-6)
      np.testing.assert_allclose(d.qvel.numpy()[w], mjd_w.qvel, rtol=1e-5, atol=1e-6)
      np.testing.assert_allclose(d.qpos.numpy()[w], mjd_w.qpos, rtol=1e-5, atol=1e-6)

    if nworld == 2:
      self.assertFalse(np.allclose(d.qvel.numpy()[0], d.qvel.numpy()[1]))

    # Wake tree 0 via qfrc_applied and verify crb populates d.M and matches MuJoCo C
    qfrc = np.zeros((nworld, mjm.nv), dtype=np.float32)
    qfrc[0, 0] = 3.0
    if nworld == 2:
      qfrc[1, 0] = 5.0
    d.qfrc_applied.assign(qfrc)

    qpos_before = d.qpos.numpy().copy()
    qvel_before = d.qvel.numpy().copy()
    mjwarp.step(m, d)

    for w in range(nworld):
      mjd_w = mujoco.MjData(mjm)
      mjd_w.qpos[:] = qpos_before[w]
      mjd_w.qvel[:] = qvel_before[w]
      mjd_w.qfrc_applied[:] = qfrc[w]
      mujoco.mj_step(mjm, mjd_w)
      self.assertEqual(d.tree_awake.numpy()[w, 0], 1)
      np.testing.assert_allclose(d.qacc.numpy()[w], mjd_w.qacc, rtol=1e-5, atol=1e-6)
      np.testing.assert_allclose(d.qvel.numpy()[w], mjd_w.qvel, rtol=1e-5, atol=1e-6)

  @parameterized.product(
    nworld=(1, 2),
    cone=(types.ConeType.PYRAMIDAL, types.ConeType.ELLIPTIC),
  )
  def test_sleep_active_inertia_scale(self, nworld, cone):
    """Verify a heavy sleeping body does not distort solver inertia_scale on awake bodies."""
    _, _, m_heavy, d_heavy = test_data.fixture(
      xml="""
      <mujoco>
        <option noslip_iterations="2">
          <flag sleep="enable" warmstart="disable"/>
        </option>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body name="heavy" pos="2 0 .1">
            <freejoint/>
            <geom size=".1" mass="1000"/>
          </body>
          <body name="light" pos="0 0 .095">
            <freejoint/>
            <geom size=".1" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
      nworld=nworld,
    )
    _, _, m_light, d_light = test_data.fixture(
      xml="""
      <mujoco>
        <option noslip_iterations="2">
          <flag sleep="enable" warmstart="disable"/>
        </option>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body name="light" pos="0 0 .095">
            <freejoint/>
            <geom size=".1" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
      nworld=nworld,
    )

    # Put the 1000-kg body (tree 0) to sleep while giving the 1-kg body tangential sliding velocity
    tree_asleep = d_heavy.tree_asleep.numpy()
    tree_asleep[:, 0] = 0
    d_heavy.tree_asleep.assign(tree_asleep)
    sleep.update_sleep(m_heavy, d_heavy)

    qvel_heavy = d_heavy.qvel.numpy()
    qvel_light = d_light.qvel.numpy()
    qvel_heavy[0, 6] = 1.0
    qvel_light[0, 0] = 1.0
    if nworld == 2:
      qvel_heavy[1, 6] = 2.0
      qvel_light[1, 0] = 2.0
    d_heavy.qvel.assign(qvel_heavy)
    d_light.qvel.assign(qvel_light)

    d_heavy.solver_niter.fill_(-1)
    d_light.solver_niter.fill_(-1)
    mjwarp.step(m_heavy, d_heavy)
    mjwarp.step(m_light, d_light)

    np.testing.assert_array_equal(d_heavy.solver_niter.numpy(), d_light.solver_niter.numpy())
    np.testing.assert_allclose(d_heavy.qacc.numpy()[:, 6:], d_light.qacc.numpy(), rtol=1e-5, atol=1e-6)
    if nworld == 2:
      self.assertFalse(np.allclose(d_heavy.qacc.numpy()[0, 6:], d_heavy.qacc.numpy()[1, 6:]))

  @parameterized.product(
    nworld=(1, 2),
    cone=(types.ConeType.PYRAMIDAL, types.ConeType.ELLIPTIC),
  )
  def test_sleep_with_disable_island(self, nworld, cone):
    """Verify EnableBit.SLEEP + DisableBit.ISLAND for constrained and unconstrained scenes."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option sleep_tolerance="0.05">
          <flag sleep="enable" island="disable"/>
        </option>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body pos="0 0 .1">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body pos="0 0 .25">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
      nworld=nworld,
    )

    # Put lower sphere (tree 0) to sleep and launch upper sphere (tree 1) downward
    tree_asleep = d.tree_asleep.numpy()
    tree_asleep[:, 0] = 0
    d.tree_asleep.assign(tree_asleep)
    sleep.update_sleep(m, d)

    qvel = d.qvel.numpy()
    qvel[0, 8] = -3.0
    if nworld == 2:
      qvel[1, 8] = -4.0
    d.qvel.assign(qvel)

    with wp.ScopedCapture() as capture:
      mjwarp.step(m, d)

    # Step past MJ_MINAWAKE: constrained trees must stay awake when island discovery is disabled
    for _ in range(15):
      wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(d.tree_awake.numpy(), 1)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

    # Unconstrained scene (nefc == 0) with DisableBit.ISLAND must still transition to sleep
    _, _, m_unc, d_unc = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0" sleep_tolerance="0.01">
          <flag sleep="enable" island="disable" contact="disable"/>
        </option>
        <worldbody>
          <body>
            <joint type="slide" damping="400"/>
            <geom size=".1" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
      nworld=nworld,
    )
    qvel_unc = d_unc.qvel.numpy()
    qvel_unc[0, 0] = 1.0
    if nworld == 2:
      qvel_unc[1, 0] = 2.0
    d_unc.qvel.assign(qvel_unc)

    with wp.ScopedCapture() as capture_unc:
      mjwarp.step(m_unc, d_unc)

    for _ in range(20):
      wp.capture_launch(capture_unc.graph)

    np.testing.assert_array_equal(d_unc.tree_awake.numpy(), 0)
    np.testing.assert_array_equal(d_unc.qvel.numpy(), 0.0)

  @parameterized.product(
    nworld=(1, 2),
    mode=("qfrc_applied", "xfrc_applied", "qvel", "ctrl"),
  )
  def test_split_step_user_perturbation_wake(self, nworld, mode):
    """Verify step1/step2 wakes sleeping trees on user force, velocity, and ctrl inputs."""
    mjm, mjd_base, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0">
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <body name="target" pos="0 0 .2">
            <freejoint/>
            <geom size=".1" mass="1"/>
          </body>
          <body name="pusher" pos="-.201 0 .2">
            <joint name="jp" type="slide" axis="1 0 0"/>
            <geom size=".1" mass="1"/>
          </body>
        </worldbody>
        <actuator>
          <motor joint="jp"/>
        </actuator>
      </mujoco>
      """,
      nworld=nworld,
    )

    # Settle target (tree 0) to sleep naturally in MuJoCo C so mjd and d share identical sleep state
    for _ in range(mujoco.mjMINAWAKE + 1):
      mujoco.mj_step(mjm, mjd_base)
    self.assertEqual(mjd_base.tree_awake[0], 0)
    self.assertEqual(mjd_base.tree_awake[1], 1)

    d = mjwarp.put_data(mjm, mjd_base, nworld=nworld)
    mjds = [copy.copy(mjd_base) for _ in range(nworld)]

    # Run step1 first, then inject user perturbation before step2 (and subsequent step1/step2 pairs)
    mjwarp.step1(m, d)
    for w in range(nworld):
      mujoco.mj_step1(mjm, mjds[w])

    scales = [1.0, 1.5] if nworld == 2 else [1.0]
    if mode == "qfrc_applied":
      qfrc = d.qfrc_applied.numpy()
      for w, s in enumerate(scales):
        qfrc[w, 0] = 5.0 * s
        mjds[w].qfrc_applied[0] = 5.0 * s
      d.qfrc_applied.assign(qfrc)
    elif mode == "xfrc_applied":
      xfrc = d.xfrc_applied.numpy()
      for w, s in enumerate(scales):
        xfrc[w, 1, 0] = 5.0 * s
        mjds[w].xfrc_applied[1, 0] = 5.0 * s
      d.xfrc_applied.assign(xfrc)
    elif mode == "qvel":
      qvel = d.qvel.numpy()
      for w, s in enumerate(scales):
        qvel[w, 0] = 1.0 * s
        mjds[w].qvel[0] = 1.0 * s
      d.qvel.assign(qvel)
    else:
      ctrl = d.ctrl.numpy()
      for w, s in enumerate(scales):
        ctrl[w, 0] = 20.0 * s
        mjds[w].ctrl[0] = 20.0 * s
      d.ctrl.assign(ctrl)

    mjwarp.step2(m, d)
    for w in range(nworld):
      mujoco.mj_step2(mjm, mjds[w])

    for _ in range(5):
      mjwarp.step1(m, d)
      mjwarp.step2(m, d)
      for w in range(nworld):
        mujoco.mj_step1(mjm, mjds[w])
        mujoco.mj_step2(mjm, mjds[w])

    for w in range(nworld):
      self.assertEqual(d.tree_awake.numpy()[w, 0], 1)
      np.testing.assert_array_equal(d.tree_awake.numpy()[w], mjds[w].tree_awake)
      np.testing.assert_allclose(d.qpos.numpy()[w], mjds[w].qpos, rtol=1e-4, atol=1e-5)
      np.testing.assert_allclose(d.qvel.numpy()[w], mjds[w].qvel, rtol=1e-4, atol=1e-5)
    if nworld == 2:
      self.assertFalse(np.allclose(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.product(
    broadphase=(types.BroadphaseType.NXN, types.BroadphaseType.SAP_TILE, types.BroadphaseType.SAP_SEGMENTED),
    graph_conditional=(True, False),
  )
  def test_sleep_pass2_preserves_broadphase_overflow_and_ncollision(self, broadphase, graph_conditional):
    """Verify Pass-2 collision accumulates d.ncollision and preserves broadphase overflow."""
    mjm, _, m, d_full = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body name="asleep_box" pos="0 0 .09">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
          </body>
          <body name="striker" pos="0 0 .23">
            <freejoint/>
            <geom size=".05"/>
          </body>
          <body name="extra_a" pos="2 0 .08">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body name="extra_b" pos="2 0 .22">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={
        "opt.broadphase": broadphase,
        "opt.graph_conditional": graph_conditional,
      },
      nworld=1,
    )

    # Put asleep_box (tree 0) to sleep and move striker downward so Pass 1 wakes tree 0
    tree_asleep = d_full.tree_asleep.numpy()
    tree_asleep[0, 0] = 0
    d_full.tree_asleep.assign(tree_asleep)
    sleep.update_sleep(m, d_full)

    qvel = d_full.qvel.numpy()
    qvel[0, 8] = -5.0
    d_full.qvel.assign(qvel)

    # Accumulated d.ncollision must include both Pass-1 (>= 4) and Pass-2 (1) candidates
    d_full.ncollision.fill_(-1)
    forward.fwd_position(m, d_full)
    total_candidates = int(d_full.ncollision.numpy()[0])
    self.assertGreaterEqual(total_candidates, 4)

    # Steady-state step when no tree wakes (woke == 0) preserves sleeping trees and contacts
    _, _, m_steady, d_steady = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="5 5 .1"/>
          <body name="sleeping" pos="-2 0 .1">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body name="awake" pos="2 0 .08">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={
        "opt.broadphase": broadphase,
        "opt.graph_conditional": graph_conditional,
      },
      nworld=1,
    )
    steady_asleep = d_steady.tree_asleep.numpy()
    steady_asleep[0, 0] = 0
    d_steady.tree_asleep.assign(steady_asleep)
    sleep.update_sleep(m_steady, d_steady)
    with wp.ScopedCapture() as capture_steady:
      mjwarp.step(m_steady, d_steady)
    wp.capture_launch(capture_steady.graph)
    self.assertEqual(d_steady.tree_awake.numpy()[0, 0], 0)
    self.assertEqual(d_steady.nacon.numpy()[0], 1)

    # With naconmax=1, Pass 1 overflows broadphase and Pass 2 must preserve the overflow bit
    d_tight = mjwarp.make_data(mjm, nworld=1, naconmax=1)
    wp.copy(d_tight.qpos, d_full.qpos)
    d_tight.qvel.assign(qvel)
    d_tight.tree_asleep.assign(tree_asleep)
    sleep.update_sleep(m, d_tight)
    m.opt.warn_overflow = 0
    d_tight.overflow.zero_()
    mjwarp.step(m, d_tight)
    self.assertNotEqual(int(d_tight.overflow.numpy()[0]) & int(types.OverflowType.BROADPHASE), 0)


# An actuated 2-hinge arm (tree 0, dofs 0-1) plus two free bodies
# (tree 1, dofs 2-7 and tree 2, dofs 8-13).
_ARM_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body>
      <joint name="j0" type="hinge"/>
      <geom type="capsule" fromto="0 0 0 0 0 .3" size=".05"/>
      <body pos="0 0 .3">
        <joint name="j1" type="hinge"/>
        <geom type="capsule" fromto="0 0 0 0 0 .3" size=".05"/>
      </body>
    </body>
    <body pos="1 0 1"><freejoint/><geom name="ball0" type="sphere" size=".1"/></body>
    <body pos="2 0 1"><freejoint/><geom name="ball1" type="sphere" size=".1"/></body>
  </worldbody>
  <actuator>
    <motor joint="j0"/>
    <motor joint="j1"/>
  </actuator>
</mujoco>
"""


class ActiveDofTest(absltest.TestCase):
  """Tests that the sleep/wake state drives island.update_active_dofs (the active-DOF maps)."""

  def test_actuated_tree_seeded(self):
    """Only the actuated arm tree is active at construction; maps are compact."""
    _, _, m, d = test_data.fixture(xml=_ARM_XML)
    d.tree_awake = wp.array([[1, 0, 0]], dtype=int)
    island.update_active_dofs(m, d)

    self.assertEqual(d.ncdof.numpy()[0], 2)
    # arm dofs 0,1 map to compacted 0,1; the rest are inactive (-1).
    dof_cdof = d.dof_cdof.numpy()[0]
    np.testing.assert_array_equal(dof_cdof[:2], [0, 1])
    self.assertTrue((dof_cdof[2:] == -1).all())
    np.testing.assert_array_equal(d.cdof_dof.numpy()[0][:2], [0, 1])

  def test_applied_force_wakes_tree_per_world(self):
    """qfrc_applied on a free-body DOF wakes its whole tree, independently per world."""
    _, _, m, d = test_data.fixture(xml=_ARM_XML, nworld=2)

    # 1. Set all trees asleep initially
    d.tree_awake.fill_(0)
    d.tree_asleep.fill_(0)

    # 2. Apply force to dof 2 (belongs to tree 1) in world 0
    qfrc = d.qfrc_applied.numpy()
    qfrc[0, 2] = 1.0
    d.qfrc_applied = wp.array(qfrc, dtype=float)

    # 3. Run sleep.wake to wake up trees with forces
    sleep.wake(m, d)
    sleep.update_sleep(m, d)

    island.update_active_dofs(m, d)

    # world 0: arm (2) + free body 1 (6) = 8; world 1: arm only = 2
    np.testing.assert_array_equal(d.ncdof.numpy(), [8, 2])
    np.testing.assert_array_equal(d.dof_cdof.numpy()[0][:8], np.arange(8))
    self.assertTrue((d.dof_cdof.numpy()[1][2:] == -1).all())

  def _fake_contact(self, d, g0, g1):
    d.nacon = wp.array([1], dtype=int)
    geom = d.contact.geom.numpy()
    geom[0] = (g0, g1)
    d.contact.geom = wp.array(geom, dtype=wp.vec2i)
    d.contact.worldid = wp.array(np.zeros(d.naconmax, dtype=int), dtype=int)

  def _set_body_moving(self, m, d, geomid, speed=1.0):
    bodyid = m.geom_bodyid.numpy()[geomid]
    cvel = d.cvel.numpy()
    cvel[0, bodyid, 3] = speed  # linear-x component of spatial velocity
    d.cvel = wp.array(cvel, dtype=wp.spatial_vector)

  def test_moving_contact_wakes_both_trees(self):
    """A contact with a moving body activates the trees of both involved geoms."""
    mjm, _, m, d = test_data.fixture(xml=_ARM_XML)
    ball0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball0")
    ball1 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball1")

    # 1. Set tree 1 (ball0) awake initially, tree 0 and 2 asleep (pointing to themselves)
    asleep = np.array([[0, sleep.K_AWAKE_VAL, 2]], dtype=np.int32)
    d.tree_asleep = wp.array(asleep, dtype=int)
    sleep.update_sleep(m, d)

    self._fake_contact(d, ball0, ball1)

    sleep.wake_collision(m, d)
    sleep.update_sleep(m, d)

    island.update_active_dofs(m, d)

    # both free bodies awake = 6 + 6 = 12 DOFs
    self.assertEqual(d.ncdof.numpy()[0], 12)

  def test_resting_contact_does_not_wake(self):
    """A contact where both bodies are at rest does not wake them."""
    mjm, _, m, d = test_data.fixture(xml=_ARM_XML)
    ball0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball0")
    ball1 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball1")

    # 1. Set all trees asleep initially
    asleep = np.array([[0, 1, 2]], dtype=np.int32)
    d.tree_asleep = wp.array(asleep, dtype=int)
    sleep.update_sleep(m, d)

    self._fake_contact(d, ball0, ball1)

    sleep.wake_collision(m, d)
    sleep.update_sleep(m, d)

    island.update_active_dofs(m, d)

    # both stay asleep
    self.assertEqual(d.ncdof.numpy()[0], 0)

  def test_static_geom_does_not_wake(self):
    """A contact with a static (world) geom only wakes the dynamic side."""
    mjm, _, m, d = test_data.fixture(xml=_ARM_XML)
    floor = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ball0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball0")

    # 1. Set ball0 (tree 1) awake, tree 0 and 2 asleep
    asleep = np.array([[0, sleep.K_AWAKE_VAL, 2]], dtype=np.int32)
    d.tree_asleep = wp.array(asleep, dtype=int)
    sleep.update_sleep(m, d)

    self._fake_contact(d, floor, ball0)

    sleep.wake_collision(m, d)
    sleep.update_sleep(m, d)

    island.update_active_dofs(m, d)

    # only tree 1 (ball0) wakes -> 6 DOFs
    self.assertEqual(d.ncdof.numpy()[0], 6)

  def test_overflow_clamps_and_warns(self):
    """When active DOFs exceed nvmax, ncdof is clamped to nvmax."""
    mjm = mujoco.MjModel.from_xml_string(_ARM_XML)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd, nvmax=4)
    d.tree_awake = wp.array([[1, 1, 1]], dtype=int)

    island.update_active_dofs(m, d)

    self.assertEqual(d.ncdof.numpy()[0], 4)


if __name__ == "__main__":
  absltest.main()
