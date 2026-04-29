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
"""Tests for sphere_triangle collision primitive."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp._src import collision_primitive_core
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
  dist_out: wp.array[float],
  pos_out: wp.array[wp.vec3],
  normal_out: wp.array[wp.vec3],
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


@wp.kernel
def box_triangle_kernel(
  # In:
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  # Out:
  dist_out: wp.array[wp.vec2],
  pos_out: wp.array[collision_primitive_core.mat23f],
  normal_out: wp.array[collision_primitive_core.mat23f],
):
  dist, pos, normal = collision_primitive_core.box_triangle(box_pos, box_rot, box_size, t1, t2, t3, tri_radius)
  dist_out[0] = dist
  pos_out[0] = pos
  normal_out[0] = normal


class BoxTriangleTest(parameterized.TestCase):
  """Tests for box_triangle collision."""

  def _run_box_triangle(
    self,
    box_pos: np.ndarray,
    box_rot: np.ndarray,
    box_size: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
    tri_radius: float,
  ):
    """Helper to run the box_triangle kernel and return results."""
    dist = wp.zeros(1, dtype=wp.vec2)
    pos = wp.zeros(1, dtype=collision_primitive_core.mat23f)
    normal = wp.zeros(1, dtype=collision_primitive_core.mat23f)

    wp.launch(
      box_triangle_kernel,
      dim=1,
      inputs=[
        wp.vec3(box_pos),
        wp.mat33(
          box_rot[0, 0],
          box_rot[0, 1],
          box_rot[0, 2],
          box_rot[1, 0],
          box_rot[1, 1],
          box_rot[1, 2],
          box_rot[2, 0],
          box_rot[2, 1],
          box_rot[2, 2],
        ),
        wp.vec3(box_size),
        wp.vec3(t1),
        wp.vec3(t2),
        wp.vec3(t3),
        tri_radius,
      ],
      outputs=[dist, pos, normal],
    )

    return dist.numpy()[0], pos.numpy()[0], normal.numpy()[0]

  def test_box_above_triangle(self):
    """Box positioned above a triangle."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    box_pos = np.array([0.5, 0.33, 0.3])
    box_rot = np.eye(3)
    box_size = np.array([0.1, 0.1, 0.1])
    tri_radius = 0.0

    dist, pos, normal = self._run_box_triangle(box_pos, box_rot, box_size, t1, t2, t3, tri_radius)

    self.assertLess(dist[0], collision_primitive_core.MJ_MAXVAL)

  def test_box_penetrating_triangle(self):
    """Box with corner penetrating the triangle."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    # Position box so triangle vertex t1 is inside the box
    box_pos = np.array([0.0, 0.0, 0.05])
    box_rot = np.eye(3)
    box_size = np.array([0.2, 0.2, 0.2])
    tri_radius = 0.0

    dist, pos, normal = self._run_box_triangle(box_pos, box_rot, box_size, t1, t2, t3, tri_radius)

    # Vertex t1 is inside the box, so we should get a contact
    self.assertLess(dist[0], collision_primitive_core.MJ_MAXVAL)

  def test_with_triangle_radius(self):
    """Triangle with non-zero radius (flex element)."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    box_pos = np.array([0.5, 0.33, 0.3])
    box_rot = np.eye(3)
    box_size = np.array([0.1, 0.1, 0.1])
    tri_radius = 0.05

    dist, pos, normal = self._run_box_triangle(box_pos, box_rot, box_size, t1, t2, t3, tri_radius)

    self.assertLess(dist[0], collision_primitive_core.MJ_MAXVAL)


@wp.kernel
def capsule_triangle_kernel(
  # In:
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  # Out:
  dist_out: wp.array[wp.vec2],
  pos_out: wp.array[collision_primitive_core.mat23f],
  normal_out: wp.array[collision_primitive_core.mat23f],
):
  dist, pos, normal = collision_primitive_core.capsule_triangle(
    capsule_pos, capsule_axis, capsule_radius, capsule_half_length, t1, t2, t3, tri_radius
  )
  dist_out[0] = dist
  pos_out[0] = pos
  normal_out[0] = normal


class CapsuleTriangleTest(parameterized.TestCase):
  """Tests for capsule_triangle collision."""

  def _run_capsule_triangle(
    self,
    capsule_pos: np.ndarray,
    capsule_axis: np.ndarray,
    capsule_radius: float,
    capsule_half_length: float,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
    tri_radius: float,
  ):
    """Helper to run the capsule_triangle kernel and return results."""
    dist = wp.zeros(1, dtype=wp.vec2)
    pos = wp.zeros(1, dtype=collision_primitive_core.mat23f)
    normal = wp.zeros(1, dtype=collision_primitive_core.mat23f)

    wp.launch(
      capsule_triangle_kernel,
      dim=1,
      inputs=[
        wp.vec3(capsule_pos),
        wp.vec3(capsule_axis),
        capsule_radius,
        capsule_half_length,
        wp.vec3(t1),
        wp.vec3(t2),
        wp.vec3(t3),
        tri_radius,
      ],
      outputs=[dist, pos, normal],
    )

    return dist.numpy()[0], pos.numpy()[0], normal.numpy()[0]

  def test_capsule_above_triangle_center(self):
    """Capsule directly above triangle center."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    capsule_pos = np.array([0.5, 0.33, 0.5])
    capsule_axis = np.array([0.0, 0.0, 1.0])
    capsule_radius = 0.1
    capsule_half_length = 0.2
    tri_radius = 0.0

    dist, pos, normal = self._run_capsule_triangle(
      capsule_pos, capsule_axis, capsule_radius, capsule_half_length, t1, t2, t3, tri_radius
    )

    expected_dist = 0.5 - capsule_half_length - capsule_radius
    np.testing.assert_allclose(dist[0], expected_dist, atol=1e-5)

  def test_capsule_penetrating_triangle(self):
    """Capsule penetrating the triangle plane."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    capsule_pos = np.array([0.5, 0.33, 0.2])
    capsule_axis = np.array([0.0, 0.0, 1.0])
    capsule_radius = 0.1
    capsule_half_length = 0.15
    tri_radius = 0.0

    dist, pos, normal = self._run_capsule_triangle(
      capsule_pos, capsule_axis, capsule_radius, capsule_half_length, t1, t2, t3, tri_radius
    )

    self.assertLess(dist[0], 0)

  def test_horizontal_capsule(self):
    """Capsule lying horizontally above the triangle."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    capsule_pos = np.array([0.5, 0.33, 0.2])
    capsule_axis = np.array([1.0, 0.0, 0.0])
    capsule_radius = 0.1
    capsule_half_length = 0.3
    tri_radius = 0.0

    dist, pos, normal = self._run_capsule_triangle(
      capsule_pos, capsule_axis, capsule_radius, capsule_half_length, t1, t2, t3, tri_radius
    )

    expected_dist = 0.2 - capsule_radius
    np.testing.assert_allclose(dist[0], expected_dist, atol=1e-5)

  def test_with_triangle_radius(self):
    """Triangle with non-zero radius (flex element)."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    capsule_pos = np.array([0.5, 0.33, 0.5])
    capsule_axis = np.array([0.0, 0.0, 1.0])
    capsule_radius = 0.1
    capsule_half_length = 0.2
    tri_radius = 0.05

    dist, pos, normal = self._run_capsule_triangle(
      capsule_pos, capsule_axis, capsule_radius, capsule_half_length, t1, t2, t3, tri_radius
    )

    expected_dist = 0.5 - capsule_half_length - capsule_radius - tri_radius
    np.testing.assert_allclose(dist[0], expected_dist, atol=1e-5)


@wp.kernel
def cylinder_triangle_kernel(
  # In:
  cylinder_pos: wp.vec3,
  cylinder_axis: wp.vec3,
  cylinder_radius: float,
  cylinder_half_height: float,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  # Out:
  dist_out: wp.array[wp.vec2],
  pos_out: wp.array[collision_primitive_core.mat23f],
  normal_out: wp.array[collision_primitive_core.mat23f],
):
  dist, pos, normal = collision_primitive_core.cylinder_triangle(
    cylinder_pos, cylinder_axis, cylinder_radius, cylinder_half_height, t1, t2, t3, tri_radius
  )
  dist_out[0] = dist
  pos_out[0] = pos
  normal_out[0] = normal


class CylinderTriangleTest(parameterized.TestCase):
  """Tests for cylinder_triangle collision."""

  def _run_cylinder_triangle(
    self,
    cylinder_pos: np.ndarray,
    cylinder_axis: np.ndarray,
    cylinder_radius: float,
    cylinder_half_height: float,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
    tri_radius: float,
  ):
    """Helper to run the cylinder_triangle kernel and return results."""
    dist = wp.zeros(1, dtype=wp.vec2)
    pos = wp.zeros(1, dtype=collision_primitive_core.mat23f)
    normal = wp.zeros(1, dtype=collision_primitive_core.mat23f)

    wp.launch(
      cylinder_triangle_kernel,
      dim=1,
      inputs=[
        wp.vec3(cylinder_pos),
        wp.vec3(cylinder_axis),
        cylinder_radius,
        cylinder_half_height,
        wp.vec3(t1),
        wp.vec3(t2),
        wp.vec3(t3),
        tri_radius,
      ],
      outputs=[dist, pos, normal],
    )

    return dist.numpy()[0], pos.numpy()[0], normal.numpy()[0]

  def test_cylinder_above_triangle(self):
    """Cylinder positioned above the triangle with vertex inside cylinder."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    cylinder_pos = np.array([0.0, 0.0, 0.3])
    cylinder_axis = np.array([0.0, 0.0, 1.0])
    cylinder_radius = 0.2
    cylinder_half_height = 0.2
    tri_radius = 0.0

    dist, pos, normal = self._run_cylinder_triangle(
      cylinder_pos, cylinder_axis, cylinder_radius, cylinder_half_height, t1, t2, t3, tri_radius
    )

    self.assertLess(dist[0], collision_primitive_core.MJ_MAXVAL)

  def test_cylinder_penetrating_triangle(self):
    """Cylinder with cap overlapping the triangle plane."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    # Position cylinder so its top cap penetrates the triangle plane at z=0
    # Cylinder center at z=-0.05 with half_height=0.1 means top cap at z=0.05
    # and vertex t1 at (0,0,0) is within cylinder_radius=0.3 of axis
    cylinder_pos = np.array([0.0, 0.0, -0.05])
    cylinder_axis = np.array([0.0, 0.0, 1.0])
    cylinder_radius = 0.5  # increased radius to ensure triangle is inside
    cylinder_half_height = 0.1
    tri_radius = 0.0

    dist, _, _ = self._run_cylinder_triangle(
      cylinder_pos, cylinder_axis, cylinder_radius, cylinder_half_height, t1, t2, t3, tri_radius
    )

    # Triangle overlaps with cylinder cap, should get contact
    self.assertLess(dist[0], collision_primitive_core.MJ_MAXVAL)

  def test_horizontal_cylinder(self):
    """Cylinder lying horizontally with triangle vertex near its side."""
    # Triangle with a vertex at z=0.2 close to cylinder axis
    t1 = np.array([0.5, 0.0, 0.2])
    t2 = np.array([1.0, 0.0, 0.2])
    t3 = np.array([0.75, 0.5, 0.2])
    # Horizontal cylinder at z=0.2, along x-axis
    cylinder_pos = np.array([0.5, 0.0, 0.2])
    cylinder_axis = np.array([1.0, 0.0, 0.0])
    cylinder_radius = 0.15
    cylinder_half_height = 0.5
    tri_radius = 0.05

    dist, pos, normal = self._run_cylinder_triangle(
      cylinder_pos, cylinder_axis, cylinder_radius, cylinder_half_height, t1, t2, t3, tri_radius
    )

    # Vertex is on the cylinder axis, should get contact with tri_radius
    self.assertLess(dist[0], collision_primitive_core.MJ_MAXVAL)

  def test_with_triangle_radius(self):
    """Triangle with non-zero radius (flex element)."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    cylinder_pos = np.array([0.0, 0.0, 0.3])
    cylinder_axis = np.array([0.0, 0.0, 1.0])
    cylinder_radius = 0.2
    cylinder_half_height = 0.2
    tri_radius = 0.05

    dist, pos, normal = self._run_cylinder_triangle(
      cylinder_pos, cylinder_axis, cylinder_radius, cylinder_half_height, t1, t2, t3, tri_radius
    )

    self.assertLess(dist[0], collision_primitive_core.MJ_MAXVAL)


if __name__ == "__main__":
  absltest.main()
