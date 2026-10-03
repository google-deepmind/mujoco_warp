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
  margin: float,
  # Out:
  dist_out: wp.array[float],
  pos_out: wp.array[wp.vec3],
  normal_out: wp.array[wp.vec3],
):
  dist, pos, normal = sphere_triangle(sphere_pos, sphere_radius, t1, t2, t3, tri_radius, margin)
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
    margin: float = 1.0,
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
        margin,
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

  def test_sphere_center_on_triangle(self):
    """Zero center-to-triangle distance uses the mju_normalize3 fallback normal (1, 0, 0)."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])
    sphere_pos = np.array([0.5, 0.33, 0.0])

    dist, pos, normal = self._run_sphere_triangle(sphere_pos, 0.1, t1, t2, t3, 0.0)

    np.testing.assert_allclose(dist, -0.1, atol=1e-6)
    np.testing.assert_allclose(normal, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pos, [0.55, 0.33, 0.0], atol=1e-6)

  @parameterized.parameters(
    (np.array([0.5, 0.33, 0.5]),),  # rejected by the plane-distance test
    (np.array([0.5, -0.5, 0.0]),),  # in-plane, rejected by the nearest-point test
  )
  def test_sphere_beyond_margin(self, sphere_pos):
    """Beyond margin + radii there is no contact, as in MuJoCo."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])

    dist, _, _ = self._run_sphere_triangle(sphere_pos, 0.2, t1, t2, t3, 0.0, margin=0.01)

    self.assertEqual(dist, collision_primitive_core.MJ_MAXVAL)


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
  margin: float,
  # Out:
  dist_out: wp.array[collision_primitive_core.vec11],
  pos_out: wp.array[collision_primitive_core.mat113],
  normal_out: wp.array[collision_primitive_core.mat113],
):
  dist, pos, normal = collision_primitive_core.box_triangle(box_pos, box_rot, box_size, t1, t2, t3, tri_radius, margin)
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
    margin: float = 1.0,
  ):
    """Helper to run the box_triangle kernel and return results."""
    dist = wp.zeros(1, dtype=collision_primitive_core.vec11)
    pos = wp.zeros(1, dtype=collision_primitive_core.mat113)
    normal = wp.zeros(1, dtype=collision_primitive_core.mat113)

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
        margin,
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

  @parameterized.parameters(
    (np.eye(3),),
    (np.diag([1.0, -1.0, -1.0]),),  # 180 deg about x: bottom face becomes local +z
    (np.diag([-1.0, -1.0, 1.0]),),  # 180 deg about z
  )
  def test_box_contacts_invariant_to_box_symmetry(self, box_rot):
    """Rotations that map the box onto itself must yield the same contacts."""
    t1 = np.array([-1.0, -1.0, 0.0])
    t2 = np.array([1.0, -1.0, 0.0])
    t3 = np.array([0.0, 1.5, 0.0])
    box_pos = np.array([0.0, 0.0, 0.1005])  # bottom face 0.5 mm above the triangle
    box_size = np.array([0.1, 0.1, 0.1])

    dist, pos, normal = self._run_box_triangle(box_pos, box_rot, box_size, t1, t2, t3, 0.0, margin=0.002)

    # the four bottom corners, as in MuJoCo
    hit = dist < collision_primitive_core.MJ_MAXVAL
    self.assertEqual(hit.sum(), 4)
    self.assertFalse(hit[:3].any())  # no triangle vertex is near the box
    np.testing.assert_allclose(dist[hit], [0.0005] * 4, atol=1e-6)
    np.testing.assert_allclose(normal[hit, 2], [-1.0] * 4, atol=1e-6)
    np.testing.assert_allclose(pos[hit, 2], [0.00025] * 4, atol=1e-6)

  def test_box_beyond_margin(self):
    """Candidates farther than margin are discarded."""
    t1 = np.array([-1.0, -1.0, 0.0])
    t2 = np.array([1.0, -1.0, 0.0])
    t3 = np.array([0.0, 1.5, 0.0])
    box_pos = np.array([0.0, 0.0, 0.3])  # bottom face 0.2 above the triangle
    box_size = np.array([0.1, 0.1, 0.1])

    dist, _, _ = self._run_box_triangle(box_pos, np.eye(3), box_size, t1, t2, t3, 0.0, margin=0.01)

    np.testing.assert_equal(dist, [collision_primitive_core.MJ_MAXVAL] * 11)

  def test_box_returns_all_corners_within_margin(self):
    """Every corner within margin is returned, in its MuJoCo corner slot."""
    t1 = np.array([-1.0, -1.0, 0.0])
    t2 = np.array([1.0, -1.0, 0.0])
    t3 = np.array([0.0, 1.5, 0.0])
    cx, sx = np.cos(0.1), np.sin(0.1)
    cy, sy = np.cos(0.05), np.sin(0.05)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    box_rot = rot_x @ rot_y @ np.diag([1.0, -1.0, -1.0])
    box_size = np.array([0.1, 0.2, 0.05])
    signs = np.array([[kx, ky, kz] for kz in (-1, 1) for ky in (-1, 1) for kx in (-1, 1)])
    corner_offset_z = ((signs * box_size) @ box_rot.T)[:, 2]
    box_pos = np.array([0.0, 0.0, 0.001 - corner_offset_z.min()])  # lowest corner 1 mm above

    dist, _, normal = self._run_box_triangle(box_pos, box_rot, box_size, t1, t2, t3, 0.0, margin=0.1)

    corner_z = box_pos[2] + corner_offset_z  # corner i has signs[i], i.e. MuJoCo's bit order
    near = corner_z <= 0.1
    self.assertBetween(near.sum(), 3, 7, "test setup: need more than two, but not all, corners")
    np.testing.assert_equal(dist[:3], [collision_primitive_core.MJ_MAXVAL] * 3)  # triangle vertices are far away
    np.testing.assert_allclose(dist[3:], np.where(near, corner_z, collision_primitive_core.MJ_MAXVAL), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(normal[3:][near, 2], [-1.0] * near.sum(), atol=1e-6)

  def test_box_triangle_vertex_within_margin(self):
    """A triangle vertex just outside a box face but within margin yields a face contact."""
    t1 = np.array([0.0, 0.0, 0.101])  # 1 mm above the top face
    t2 = np.array([0.05, 0.0, 0.2])
    t3 = np.array([0.0, 0.05, 0.2])

    dist, pos, normal = self._run_box_triangle(np.zeros(3), np.eye(3), np.array([0.1, 0.1, 0.1]), t1, t2, t3, 0.0, margin=0.002)

    np.testing.assert_allclose(dist[0], 0.001, atol=1e-6)
    np.testing.assert_allclose(normal[0], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(pos[0], [0.0, 0.0, 0.1005], atol=1e-6)
    np.testing.assert_equal(dist[1:], [collision_primitive_core.MJ_MAXVAL] * 10)

  def test_box_triangle_vertex_at_box_center(self):
    """A triangle vertex at the box center uses MuJoCo's face normal sign (-1 for zero)."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([0.0, 1.0, 2.0])  # triangle in the x = 0 plane, box corners at |x| = 0.1
    t3 = np.array([0.0, -1.0, 2.0])

    dist, pos, normal = self._run_box_triangle(np.zeros(3), np.eye(3), np.array([0.1, 0.1, 0.1]), t1, t2, t3, 0.0, margin=0.01)

    np.testing.assert_allclose(dist[0], -0.1, atol=1e-6)
    np.testing.assert_allclose(normal[0], [-1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pos[0], [-0.05, 0.0, 0.0], atol=1e-6)
    np.testing.assert_equal(dist[1:], [collision_primitive_core.MJ_MAXVAL] * 10)


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
  margin: float,
  # Out:
  dist_out: wp.array[collision_primitive_core.vec5],
  pos_out: wp.array[collision_primitive_core.mat53],
  normal_out: wp.array[collision_primitive_core.mat53],
):
  dist, pos, normal = collision_primitive_core.capsule_triangle(
    capsule_pos, capsule_axis, capsule_radius, capsule_half_length, t1, t2, t3, tri_radius, margin
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
    margin: float = 1.0,
  ):
    """Helper to run the capsule_triangle kernel and return results."""
    dist = wp.zeros(1, dtype=collision_primitive_core.vec5)
    pos = wp.zeros(1, dtype=collision_primitive_core.mat53)
    normal = wp.zeros(1, dtype=collision_primitive_core.mat53)

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
        margin,
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

  def test_triangle_vertex_against_capsule_side(self):
    """A vertex touching the capsule side is found even when both end caps are far."""
    t1 = np.array([0.0, 0.0, 0.06])  # 1 cm into the capsule side
    t2 = np.array([-0.05, 0.0, -0.2])
    t3 = np.array([0.05, 0.0, -0.2])
    capsule_pos = np.array([0.0, 0.0, 0.1])
    capsule_axis = np.array([1.0, 0.0, 0.0])

    dist, pos, normal = self._run_capsule_triangle(capsule_pos, capsule_axis, 0.05, 0.5, t1, t2, t3, 0.0, margin=0.01)

    np.testing.assert_equal(dist[[0, 1, 3, 4]], [collision_primitive_core.MJ_MAXVAL] * 4)
    np.testing.assert_allclose(dist[2], -0.01, atol=1e-6)  # slot 2: triangle vertex t1
    np.testing.assert_allclose(normal[2], [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(pos[2], [0.0, 0.0, 0.055], atol=1e-6)

  def test_triangle_vertex_on_capsule_axis(self):
    """A vertex on the capsule axis gets the mju_normalize3 fallback normal (1, 0, 0)."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.0, 1.0, 0.0])

    dist, pos, normal = self._run_capsule_triangle(
      np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.05, 0.5, t1, t2, t3, 0.0, margin=0.01
    )

    np.testing.assert_equal(dist[[0, 1, 3, 4]], [collision_primitive_core.MJ_MAXVAL] * 4)
    np.testing.assert_allclose(dist[2], -0.05, atol=1e-6)  # slot 2: triangle vertex t1
    np.testing.assert_allclose(normal[2], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pos[2], [0.025, 0.0, 0.0], atol=1e-6)

  def test_capsule_beyond_margin(self):
    """Candidates farther than margin are discarded."""
    t1 = np.array([0.0, 0.0, 0.0])
    t2 = np.array([1.0, 0.0, 0.0])
    t3 = np.array([0.5, 1.0, 0.0])

    dist, _, _ = self._run_capsule_triangle(
      np.array([0.5, 0.33, 0.5]), np.array([0.0, 0.0, 1.0]), 0.1, 0.2, t1, t2, t3, 0.0, margin=0.01
    )

    np.testing.assert_equal(dist, [collision_primitive_core.MJ_MAXVAL] * 5)


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

  def test_flat_cylinder_cap_off_axis_penetration(self):
    """Off-axis triangle vertices penetrating a flat cylinder cap report axial cap contact."""
    # Wide flat disk cylinder with top cap at z = 0.016
    cylinder_pos = np.array([0.0, 0.0, 0.008])
    cylinder_axis = np.array([0.0, 0.0, 1.0])
    cylinder_radius = 0.102
    cylinder_half_height = 0.008
    tri_radius = 0.001
    # Off-axis triangle vertices at r ~ 0.05..0.06 m penetrating top cap by 0.5 mm (z = 0.0155)
    t1 = np.array([0.05, 0.0, 0.0155])
    t2 = np.array([0.06, 0.0, 0.0155])
    t3 = np.array([0.05, 0.01, 0.0155])

    dist, pos, normal = self._run_cylinder_triangle(
      cylinder_pos, cylinder_axis, cylinder_radius, cylinder_half_height, t1, t2, t3, tri_radius
    )

    expected_dist = -(0.016 - 0.0155) - tri_radius
    np.testing.assert_allclose(dist[0], expected_dist, atol=1e-5)
    np.testing.assert_allclose(normal[0], [0.0, 0.0, 1.0], atol=1e-5)


if __name__ == "__main__":
  absltest.main()
