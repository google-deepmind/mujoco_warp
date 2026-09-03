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

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp._src import types
from mujoco_warp._src.math import closest_segment_to_segment_points
from mujoco_warp._src.math import free_bias_vel_blocks
from mujoco_warp._src.math import lu_factor_6x6
from mujoco_warp._src.math import lu_solve_6x6
from mujoco_warp._src.math import upper_tri_index
from mujoco_warp._src.math import upper_trid_index


@wp.kernel
def _lu_solve_6x6_kernel(
  # In:
  A_in: wp.array2d[float],
  b_in: wp.array[float],
  # Out:
  x_out: wp.array[float],
  success_out: wp.array[bool],
):
  A_work = types.mat66(0.0)
  b_vec = types.vec6(0.0)
  for i in range(6):
    b_vec[i] = b_in[i]
    for j in range(6):
      A_work[i, j] = A_in[i, j]

  A_fact, pivot_work, ok = lu_factor_6x6(A_work)
  if ok:
    success_out[0] = True
    x_vec = lu_solve_6x6(A_fact, pivot_work, b_vec)
    for i in range(6):
      x_out[i] = x_vec[i]


@wp.kernel
def _free_bias_vel_blocks_kernel(
  # In:
  mass: float,
  R: wp.mat33,
  Xi: wp.mat33,
  inertia: wp.vec3,
  s: wp.vec3,
  qvel_rot: wp.vec3,
  # Out:
  lin_out: wp.array2d[float],
  rot_out: wp.array2d[float],
):
  lin, rot = free_bias_vel_blocks(mass, R, Xi, inertia, s, qvel_rot)
  for r in range(3):
    for c in range(3):
      lin_out[r, c] = lin[r, c]
      rot_out[r, c] = rot[r, c]


class ClosestSegmentSegmentPointsTest(absltest.TestCase):
  """Tests for closest segment-to-segment points."""

  def test_closest_segments_points(self):
    """Test closest points between two segments."""
    a0 = wp.vec3([0.73432405, 0.12372768, 0.20272314])
    a1 = wp.vec3([1.10600128, 0.88555209, 0.65209485])
    b0 = wp.vec3([0.85599262, 0.61736299, 0.9843583])
    b1 = wp.vec3([1.84270939, 0.92891793, 1.36343326])

    best_a, best_b = closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [1.09063, 0.85404, 0.63351], 5)
    self.assertSequenceAlmostEqual(best_b, [0.99596, 0.66156, 1.03813], 5)

  def test_intersecting_segments(self):
    """Tests segments that intersect."""
    a0, a1 = wp.vec3([0.0, 0.0, -1.0]), wp.vec3([0.0, 0.0, 1.0])
    b0, b1 = wp.vec3([-1.0, 0.0, 0.0]), wp.vec3([1.0, 0.0, 0.0])

    best_a, best_b = closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 0.0], 5)
    self.assertSequenceAlmostEqual(best_b, [0.0, 0.0, 0.0], 5)

  def test_intersecting_lines(self):
    """Tests that intersecting lines get clipped."""
    a0, a1 = wp.vec3([0.2, 0.2, 0.0]), wp.vec3([1.0, 1.0, 0.0])
    b0, b1 = wp.vec3([0.2, 0.4, 0.0]), wp.vec3([1.0, 2.0, 0.0])

    best_a, best_b = closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.3, 0.3, 0.0], 2)
    self.assertSequenceAlmostEqual(best_b, [0.2, 0.4, 0.0], 2)

  def test_parallel_segments(self):
    """Tests that parallel segments have closest points at the midpoint."""
    a0, a1 = wp.vec3([0.0, 0.0, -1.0]), wp.vec3([0.0, 0.0, 1.0])
    b0, b1 = wp.vec3([1.0, 0.0, -1.0]), wp.vec3([1.0, 0.0, 1.0])

    best_a, best_b = closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 0.0], 5)
    self.assertSequenceAlmostEqual(best_b, [1.0, 0.0, 0.0], 5)

  def test_parallel_offset_segments(self):
    """Tests that offset parallel segments are close at segment endpoints."""
    a0, a1 = wp.vec3([0.0, 0.0, -1.0]), wp.vec3([0.0, 0.0, 1.0])
    b0, b1 = wp.vec3([1.0, 0.0, 1.0]), wp.vec3([1.0, 0.0, 3.0])

    best_a, best_b = closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 1.0], 5)
    self.assertSequenceAlmostEqual(best_b, [1.0, 0.0, 1.0], 5)

  def test_zero_length_segments(self):
    """Test that zero length segments don't return NaNs."""
    a0, a1 = wp.vec3([0.0, 0.0, -1.0]), wp.vec3([0.0, 0.0, -1.0])
    b0, b1 = wp.vec3([1.0, 0.0, 0.1]), wp.vec3([1.0, 0.0, 0.1])

    best_a, best_b = closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, -1.0], 5)
    self.assertSequenceAlmostEqual(best_b, [1.0, 0.0, 0.1], 5)

  def test_overlapping_segments(self):
    """Tests that perfectly overlapping segments intersect at the midpoints."""
    a0, a1 = wp.vec3([0.0, 0.0, -1.0]), wp.vec3([0.0, 0.0, 1.0])
    b0, b1 = wp.vec3([0.0, 0.0, -1.0]), wp.vec3([0.0, 0.0, 1.0])

    best_a, best_b = closest_segment_to_segment_points(a0, a1, b0, b1)
    self.assertSequenceAlmostEqual(best_a, [0.0, 0.0, 0.0], 5)
    self.assertSequenceAlmostEqual(best_b, [0.0, 0.0, 0.0], 5)

  def test_upper_tri_index2(self):
    """Tests upper_tri_index with size 2."""
    arr = []
    for i in range(2):
      for j in range(i + 1, 2):
        arr.append(upper_tri_index(2, i, j))
    self.assertEqual(arr, list(range(0, 1)))

  def test_upper_tri_index10(self):
    """Tests upper_tri_index with size 10."""
    arr = []
    for i in range(10):
      for j in range(i + 1, 10):
        arr.append(upper_tri_index(10, i, j))
    self.assertEqual(arr, list(range(0, 45)))

  def test_upper_trid_index1(self):
    """Tests upper_trid_index with size 1."""
    arr = []
    for i in range(1):
      for j in range(i, 1):
        arr.append(upper_trid_index(1, i, j))
    self.assertEqual(arr, list(range(0, 1)))

  def test_upper_trid_index10(self):
    """Tests upper_trid_index with size 10."""
    arr = []
    for i in range(10):
      for j in range(i, 10):
        arr.append(upper_trid_index(10, i, j))
    self.assertEqual(arr, list(range(0, 55)))

  def test_upper_trid_index10(self):
    """Tests upper_trid_index works with symmetric matrix."""
    self.assertEqual(upper_trid_index(10, 1, 5), upper_trid_index(10, 5, 1))


class DenseLUTest(parameterized.TestCase):
  """Tests for 6x6 dense LU factorization and solve."""

  @parameterized.parameters(
    # General non-singular matrix
    (
      np.array(
        [
          [10.0, 2.0, 1.0, 0.0, 0.0, 0.0],
          [3.0, 12.0, 4.0, 1.0, 0.0, 0.0],
          [1.0, 2.0, 15.0, 2.0, 1.0, 0.0],
          [0.0, 1.0, 3.0, 11.0, 2.0, 1.0],
          [0.0, 0.0, 1.0, 2.0, 9.0, 3.0],
          [0.0, 0.0, 0.0, 1.0, 2.0, 8.0],
        ],
        dtype=np.float32,
      ),
      np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0], dtype=np.float32),
      True,
    ),
    # Zero diagonal matrix requiring pivoting
    (
      np.array(
        [
          [0.0, 1.0, 2.0, 0.0, 0.0, 0.0],
          [1.0, 0.0, 3.0, 1.0, 0.0, 0.0],
          [2.0, 3.0, 0.0, 4.0, 1.0, 0.0],
          [0.0, 1.0, 4.0, 0.0, 5.0, 2.0],
          [0.0, 0.0, 1.0, 5.0, 0.0, 3.0],
          [0.0, 0.0, 0.0, 2.0, 3.0, 0.0],
        ],
        dtype=np.float32,
      ),
      np.array([2.0, -1.0, 4.0, 3.0, -2.0, 5.0], dtype=np.float32),
      True,
    ),
    # Singular matrix
    (
      np.zeros((6, 6), dtype=np.float32),
      np.zeros(6, dtype=np.float32),
      False,
    ),
  )
  def test_lu_solve(self, A_np, b_np, expected_success):
    """Tests 6x6 LU factorization and solve against numpy solution."""
    A_wp = wp.array(A_np, dtype=float)
    b_wp = wp.array(b_np, dtype=float)
    x_wp = wp.zeros(6, dtype=float)
    success_wp = wp.zeros(1, dtype=bool)

    wp.launch(_lu_solve_6x6_kernel, dim=1, inputs=[A_wp, b_wp], outputs=[x_wp, success_wp])

    self.assertEqual(success_wp.numpy()[0], expected_success)
    if expected_success:
      x_ref = np.linalg.solve(A_np, b_np)
      np.testing.assert_allclose(x_wp.numpy(), x_ref, atol=1e-5, rtol=1e-5)


class FreeBiasVelTest(absltest.TestCase):
  """Tests for standalone free body bias velocity derivative."""

  def test_free_bias_vel_blocks(self):
    mass = 2.5
    theta = 0.4
    R_np = np.array(
      [
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
      ],
      dtype=np.float32,
    )
    Xi_np = np.eye(3, dtype=np.float32)
    inertia_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    s_np = np.array([0.05, -0.02, 0.03], dtype=np.float32)
    qvel_rot_np = np.array([1.2, -0.8, 2.1], dtype=np.float32)

    w = R_np @ qvel_rot_np
    Iw = Xi_np @ np.diag(inertia_np) @ Xi_np.T
    ws = np.cross(w, s_np)
    Iww = Iw @ w
    w_dot_s = float(np.dot(w, s_np))
    K = (
      np.outer(s_np, w)
      - np.eye(3) * w_dot_s
      + np.array(
        [
          [0.0, -ws[2], ws[1]],
          [ws[2], 0.0, -ws[0]],
          [-ws[1], ws[0], 0.0],
        ]
      )
    )
    lin_ref = K @ R_np

    def skew(v):
      return np.array(
        [
          [0.0, -v[2], v[1]],
          [v[2], 0.0, -v[0]],
          [-v[1], v[0], 0.0],
        ]
      )

    C = -mass * skew(s_np) @ K + skew(w) @ Iw - skew(Iww)
    rot_ref = R_np.T @ C @ R_np

    lin_wp = wp.zeros((3, 3), dtype=float)
    rot_wp = wp.zeros((3, 3), dtype=float)
    wp.launch(
      _free_bias_vel_blocks_kernel,
      dim=1,
      inputs=[
        mass,
        wp.mat33(R_np),
        wp.mat33(Xi_np),
        wp.vec3(inertia_np),
        wp.vec3(s_np),
        wp.vec3(qvel_rot_np),
      ],
      outputs=[lin_wp, rot_wp],
    )

    np.testing.assert_allclose(lin_wp.numpy(), lin_ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(rot_wp.numpy(), rot_ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  wp.init()
  absltest.main()
