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
"""Tests for sphere_triangle collision primitive."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp._src.collision_primitive_core import sphere_triangle


@wp.kernel
def sphere_triangle_kernel(
  # In:
  sphere_pos: wp.vec3,
  sphere_radius: float,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  # Out:
  dist_out: wp.array(dtype=float),
  pos_out: wp.array(dtype=wp.vec3),
  normal_out: wp.array(dtype=wp.vec3),
):
  dist, pos, normal = sphere_triangle(sphere_pos, sphere_radius, t1, t2, t3, tri_radius)
  dist_out[0] = dist
  pos_out[0] = pos
  normal_out[0] = normal


class SphereTriangleTest(parameterized.TestCase):
  """Tests for sphere_triangle collision."""

  def _run_sphere_triangle(
    self,
    sphere_pos: np.ndarray,
    sphere_radius: float,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
    tri_radius: float,
  ):
    """Helper to run the sphere_triangle kernel and return results."""
    dist = wp.zeros(1, dtype=float)
    pos = wp.zeros(1, dtype=wp.vec3)
    normal = wp.zeros(1, dtype=wp.vec3)

    wp.launch(
      sphere_triangle_kernel,
      dim=1,
      inputs=[
        wp.vec3(sphere_pos),
        sphere_radius,
        wp.vec3(t1),
        wp.vec3(t2),
        wp.vec3(t3),
        tri_radius,
      ],
      outputs=[dist, pos, normal],
    )

    return dist.numpy()[0], pos.numpy()[0], normal.numpy()[0]

  def test_sphere_above_triangle_center(self):
    """Sphere directly above triangle center."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    sphere_pos = np.array([0.5, 0.33, 0.5])
    sphere_radius = 0.2
    tri_radius = 0.0

    dist, pos, normal = self._run_sphere_triangle(sphere_pos, sphere_radius, t1, t2, t3, tri_radius)

    expected_dist = 0.5 - sphere_radius
    np.testing.assert_allclose(dist, expected_dist, atol=1e-5)
    np.testing.assert_allclose(normal, [0, 0, -1], atol=1e-5)

  def test_sphere_penetrating_triangle(self):
    """Sphere penetrating the triangle plane."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    sphere_pos = np.array([0.5, 0.33, 0.1])
    sphere_radius = 0.2
    tri_radius = 0.0

    dist, pos, normal = self._run_sphere_triangle(sphere_pos, sphere_radius, t1, t2, t3, tri_radius)

    expected_dist = 0.1 - sphere_radius
    self.assertLess(dist, 0)
    np.testing.assert_allclose(dist, expected_dist, atol=1e-5)
    np.testing.assert_allclose(normal, [0, 0, -1], atol=1e-5)

  def test_sphere_near_edge(self):
    """Sphere center projects outside triangle, nearest point is on edge."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    sphere_pos = np.array([0.5, -0.3, 0.3])
    sphere_radius = 0.2
    tri_radius = 0.0

    dist, pos, normal = self._run_sphere_triangle(sphere_pos, sphere_radius, t1, t2, t3, tri_radius)

    self.assertGreater(dist, 0)

  def test_sphere_near_vertex(self):
    """Sphere center nearest to a vertex of the triangle."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    sphere_pos = np.array([-0.3, -0.3, 0.0])
    sphere_radius = 0.2
    tri_radius = 0.0

    dist, pos, normal = self._run_sphere_triangle(sphere_pos, sphere_radius, t1, t2, t3, tri_radius)

    expected_vec = sphere_pos - t1
    expected_length = np.linalg.norm(expected_vec)
    expected_dist = expected_length - sphere_radius
    np.testing.assert_allclose(dist, expected_dist, atol=1e-5)

  def test_with_triangle_radius(self):
    """Triangle with non-zero radius (flex element)."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    sphere_pos = np.array([0.5, 0.33, 0.5])
    sphere_radius = 0.2
    tri_radius = 0.1

    dist, pos, normal = self._run_sphere_triangle(sphere_pos, sphere_radius, t1, t2, t3, tri_radius)

    expected_dist = 0.5 - sphere_radius - tri_radius
    np.testing.assert_allclose(dist, expected_dist, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
