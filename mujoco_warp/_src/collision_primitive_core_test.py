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
"""Tests for collision primitive core functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp._src import collision_primitive_core
from mujoco_warp._src.collision_primitive_core import box_box
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


@wp.kernel
def box_box_kernel(
  # In:
  box1_pos: wp.array[wp.vec3],
  box1_rot: wp.array[wp.mat33],
  box1_size: wp.array[wp.vec3],
  box2_pos: wp.array[wp.vec3],
  box2_rot: wp.array[wp.mat33],
  box2_size: wp.array[wp.vec3],
  margin: float,
  # Out:
  dist_out: wp.array2d[float],
  pos_out: wp.array2d[wp.vec3],
  normal_out: wp.array2d[wp.vec3],
):
  worldid = wp.tid()
  dist, pos, normal = box_box(
    box1_pos[worldid],
    box1_rot[worldid],
    box1_size[worldid],
    box2_pos[worldid],
    box2_rot[worldid],
    box2_size[worldid],
    margin,
  )
  for i in range(8):
    dist_out[worldid, i] = dist[i]
    pos_out[worldid, i] = pos[i]
    normal_out[worldid, i] = normal[i]


class BoxBoxTest(parameterized.TestCase):
  """Tests for box_box collision primitive."""

  def _run_box_box(
    self,
    box1_pos: np.ndarray,
    box1_rot: np.ndarray,
    box1_size: np.ndarray,
    box2_pos: np.ndarray,
    box2_rot: np.ndarray,
    box2_size: np.ndarray,
    margin: float = 0.0,
    nworld: int = 1,
  ):
    """Helper to run the box_box kernel and return valid contacts."""
    box1_pos_arr = wp.array([wp.vec3(box1_pos)] * nworld, dtype=wp.vec3)
    box1_rot_arr = wp.array([wp.mat33(box1_rot)] * nworld, dtype=wp.mat33)
    box1_size_arr = wp.array([wp.vec3(box1_size)] * nworld, dtype=wp.vec3)
    box2_pos_arr = wp.array([wp.vec3(box2_pos)] * nworld, dtype=wp.vec3)
    box2_rot_arr = wp.array([wp.mat33(box2_rot)] * nworld, dtype=wp.mat33)
    box2_size_arr = wp.array([wp.vec3(box2_size)] * nworld, dtype=wp.vec3)

    dist = wp.zeros((nworld, 8), dtype=float)
    pos = wp.zeros((nworld, 8), dtype=wp.vec3)
    normal = wp.zeros((nworld, 8), dtype=wp.vec3)

    wp.launch(
      box_box_kernel,
      dim=nworld,
      inputs=[
        box1_pos_arr,
        box1_rot_arr,
        box1_size_arr,
        box2_pos_arr,
        box2_rot_arr,
        box2_size_arr,
        margin,
      ],
      outputs=[dist, pos, normal],
    )

    d_all = dist.numpy()
    p_all = pos.numpy()
    n_all = normal.numpy()

    for w in range(1, nworld):
      np.testing.assert_allclose(d_all[w], d_all[0])
      np.testing.assert_allclose(p_all[w], p_all[0])
      np.testing.assert_allclose(n_all[w], n_all[0])

    valid = d_all[0] < collision_primitive_core.MJ_MAXVAL
    return d_all[0, valid], p_all[0, valid], n_all[0, valid]

  @parameterized.parameters(1, 2)
  def test_concentric_boxes(self, nworld):
    """Two identical concentric boxes."""
    size = np.array([1.0, 1.0, 1.0])
    pos = np.array([0.0, 0.0, 0.0])
    rot = np.eye(3)

    dist, p, normal = self._run_box_box(pos, rot, size, pos, rot, size, margin=0.0, nworld=nworld)

    self.assertGreater(len(dist), 0)
    self.assertLessEqual(len(dist), 8)
    np.testing.assert_allclose(dist, -2.0, atol=1e-5)

  @parameterized.parameters(1, 2)
  def test_contained_box_sweep(self, nworld):
    """Smaller box contained inside larger box, translated along each axis."""
    large_size = np.array([1.0, 1.0, 1.0])
    small_size = np.array([0.2, 0.2, 0.2])
    rot = np.eye(3)
    margin = 0.05

    for axis in range(3):
      for direction in [-1.0, 1.0]:
        for offset in [0.5, 0.79, 0.8, 0.81, 1.2, 1.3]:
          box2_pos = np.zeros(3)
          box2_pos[axis] = direction * offset

          dist, p, normal = self._run_box_box(
            np.zeros(3), rot, large_size, box2_pos, rot, small_size, margin=margin, nworld=nworld
          )

          expected_dist = offset - 1.0 - 0.2
          if expected_dist > margin:
            self.assertEqual(len(dist), 0)
          else:
            self.assertGreater(len(dist), 0)
            np.testing.assert_allclose(dist, expected_dist, atol=1e-4)
            for n_vec in normal:
              self.assertAlmostEqual(n_vec[axis], direction, places=4)

  @parameterized.parameters(1, 2)
  def test_canonical_face_edge_45deg(self, nworld):
    """Exact 45-degree edge resting on horizontal face."""
    size = np.array([0.1, 0.1, 0.1])
    rot1 = np.eye(3)
    pos1 = np.zeros(3)

    angle = np.pi / 4.0
    c, s = np.cos(angle), np.sin(angle)
    rot2 = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    diag = 0.1 * np.sqrt(2.0)
    pos2 = np.array([0.0, 0.0, 0.1 + diag - 0.005])

    dist, p, normal = self._run_box_box(pos1, rot1, size, pos2, rot2, size, margin=0.0, nworld=nworld)

    self.assertGreater(len(dist), 0)
    np.testing.assert_allclose(dist, -0.005, atol=1e-4)
    for n_vec in normal:
      np.testing.assert_allclose(n_vec, [0.0, 0.0, 1.0], atol=1e-4)

  @parameterized.parameters(1, 2)
  def test_face_contact_90deg_yaw(self, nworld):
    """Face-to-face contact with 90-degree yaw."""
    size = np.array([0.1, 0.1, 0.1])
    rot1 = np.eye(3)
    pos1 = np.zeros(3)

    rot2 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pos2 = np.array([0.0, 0.0, 0.2 - 0.002])

    dist, p, normal = self._run_box_box(pos1, rot1, size, pos2, rot2, size, margin=0.0, nworld=nworld)

    self.assertGreaterEqual(len(dist), 4)
    self.assertLessEqual(len(dist), 8)
    np.testing.assert_allclose(dist, -0.002, atol=1e-4)
    for n_vec in normal:
      np.testing.assert_allclose(n_vec, [0.0, 0.0, 1.0], atol=1e-4)

  @parameterized.parameters(1, 2)
  def test_edge_edge(self, nworld):
    """Edge-edge contact with strictly winning cross-product axis."""
    size1 = np.array([0.034760485, 0.000712135, 0.005931487])
    size2 = np.array([0.013576184, 0.008216226, 0.001168062])
    pos1 = np.zeros(3)
    pos2 = np.array([0.004439242, -0.004176748, 0.000781405])

    q1 = np.array([0.586643668, 0.399916712, 0.240856973, -0.661742963])
    q2 = np.array([-0.351684968, 0.483281031, 0.120631441, -0.792593959])

    rot1 = np.zeros(9)
    mujoco.mju_quat2Mat(rot1, q1)
    rot1 = rot1.reshape(3, 3)

    rot2 = np.zeros(9)
    mujoco.mju_quat2Mat(rot2, q2)
    rot2 = rot2.reshape(3, 3)

    dist, p, normal = self._run_box_box(pos1, rot1, size1, pos2, rot2, size2, margin=1e-4, nworld=nworld)

    self.assertEqual(len(dist), 1)
    self.assertLess(dist[0], 0.0)

  @parameterized.parameters(1, 2)
  def test_near_aligned_manifold_exact(self, nworld):
    """Resting box tilted by small angles in the near-aligned regime."""
    size = np.array([0.05, 0.05, 0.05])
    pos1 = np.zeros(3)
    rot1 = np.eye(3)

    axis = np.array([1.0, 0.5, 3.0])
    axis /= np.linalg.norm(axis)

    for decade in [-6, -5, -4]:
      angle = 10.0**decade
      k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
      rot2 = np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)
      pos2 = np.array([0.0, 0.0, 0.1 - 1e-5])

      dist, p, normal = self._run_box_box(pos1, rot1, size, pos2, rot2, size, margin=0.0, nworld=nworld)

      self.assertEqual(len(dist), 8, msg=f"Failed for decade {decade}")
      for n_vec in normal:
        self.assertAlmostEqual(abs(n_vec[2]), 1.0, places=4)

  @parameterized.parameters(1, 2)
  def test_shallow_overlap_survives_rounding(self, nworld):
    """Boxes overlapping by 7.1e-8 of their scale survive single-precision rounding."""
    size1 = np.array([0.076610468, 0.219896257, 0.000517949])
    size2 = np.array([0.029346986, 0.000313501, 0.156584486])
    pos1 = np.zeros(3)
    pos2 = np.array([0.109957509, 0.067433357, -0.272632331])

    q1 = np.array([-0.599370003, 0.082370907, -0.386436731, 0.696159005])
    q2 = np.array([0.051869582, -0.755362034, 0.604381323, 0.247913092])

    rot1 = np.zeros(9)
    mujoco.mju_quat2Mat(rot1, q1)
    rot1 = rot1.reshape(3, 3)

    rot2 = np.zeros(9)
    mujoco.mju_quat2Mat(rot2, q2)
    rot2 = rot2.reshape(3, 3)

    dist, p, normal = self._run_box_box(pos1, rot1, size1, pos2, rot2, size2, margin=0.0, nworld=nworld)

    self.assertGreater(len(dist), 0)

  @parameterized.parameters(1, 2)
  def test_deep_penetration(self, nworld):
    """Boxes penetrating deeper than smallest half-size retain contacts."""
    size = np.array([0.5, 0.5, 0.5])
    pos1 = np.zeros(3)
    pos2 = np.array([0.0, 0.0, 0.3])
    rot = np.eye(3)

    dist, p, normal = self._run_box_box(pos1, rot, size, pos2, rot, size, margin=0.0, nworld=nworld)

    self.assertGreater(len(dist), 0)
    np.testing.assert_allclose(dist, -0.7, atol=1e-4)


if __name__ == "__main__":
  absltest.main()
