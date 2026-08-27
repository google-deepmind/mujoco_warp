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
"""Tests for ray functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src.types import vec6

# tolerance for difference between MuJoCo and MJX ray calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class RayTest(parameterized.TestCase):
  def test_ray_nothing(self):
    """Tests that ray returns -1 when nothing is hit."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    pnt = wp.array([wp.vec3(12.146, 1.865, 3.895)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]  # Extract from [[-1]]
    dist_np = dist.numpy()[0, 0]  # Extract from [[-1.]]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")
    _assert_eq(normal_np, 0, "normal")

    # test that bvh accelerated produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

  def test_ray_plane(self):
    """Tests ray<>plane matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # looking down at a slight angle
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, 0, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_normal = np.zeros(3, dtype=np.float64)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused, mj_normal)
    _assert_eq(dist_np, mj_dist, "dist")
    _assert_eq(normal_np, mj_normal, "normal")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

    # looking on wrong side of plane
    pnt = wp.array([wp.vec3(0.0, 0.0, -0.5)], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")

    # test that bvh raycast produces the same results
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")

  def test_ray_sphere(self):
    """Tests ray<>sphere matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # looking down at sphere at a slight angle
    pnt = wp.array([wp.vec3(0.0, 0.0, 1.6)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, 1, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_normal = np.zeros(3, dtype=np.float64)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused, mj_normal)
    _assert_eq(dist_np, mj_dist, "dist")
    _assert_eq(normal_np, mj_normal, "normal")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

  def test_ray_capsule(self):
    """Tests ray<>capsule matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # looking down at capsule at a slight angle
    pnt = wp.array([wp.vec3(0.5, 1.0, 1.6)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, 2, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_normal = np.zeros(3, dtype=np.float64)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused, mj_normal)
    _assert_eq(dist_np, mj_dist, "dist")
    _assert_eq(normal_np, mj_normal, "normal")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

    # looking up at capsule from below
    pnt = wp.array([wp.vec3(-0.5, 1.0, 0.05)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, 1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, 2, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_normal = np.zeros(3, dtype=np.float64)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused, mj_normal)
    _assert_eq(dist_np, mj_dist, "dist")
    _assert_eq(normal_np, mj_normal, "normal")

    # test that bvh raycast produces the same results
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

    # looking at cylinder of capsule from the side
    pnt = wp.array([wp.vec3(0.0, 1.0, 0.75)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(1.0, 0.0, 0.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, 2, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_normal = np.zeros(3, dtype=np.float64)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused, mj_normal)
    _assert_eq(dist_np, mj_dist, "dist")
    _assert_eq(normal_np, mj_normal, "normal")

    # test that bvh raycast produces the same results
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

  def test_ray_cylinder(self):
    """Tests ray<>cylinder matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    pnt = wp.array([wp.vec3(2.0, 0.0, 0.05)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, 1.0))], dtype=wp.vec3).reshape((1, 1))

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_normal = np.zeros(3, dtype=np.float64)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt.numpy()[0, 0], vec.numpy()[0, 0], None, 1, -1, mj_geomid, mj_normal)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]

    _assert_eq(geomid_np, mj_geomid[0], "geomid")
    _assert_eq(dist_np, mj_dist, "dist")
    _assert_eq(normal_np, mj_normal, "normal")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

  def test_ray_box(self):
    """Tests ray<>box matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # looking down at box at a slight angle
    pnt = wp.array([wp.vec3(1.0, 0.0, 1.6)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, 3, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_normal = np.zeros(3, dtype=np.float64)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused, mj_normal)
    _assert_eq(dist_np, mj_dist, "dist")
    _assert_eq(normal_np, mj_normal, "normal")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

    # looking up at box from below
    pnt = wp.array([wp.vec3(1.0, 0.0, 0.05)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, 1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, 3, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_normal = np.zeros(3, dtype=np.float64)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused, mj_normal)
    _assert_eq(dist_np, mj_dist, "dist")
    _assert_eq(normal_np, mj_normal, "normal")

    # test that bvh raycast produces the same results
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "bvh geom_id")
    _assert_eq(bvh_dist_np, dist_np, "bvh dist")
    _assert_eq(bvh_normal_np, normal_np, "bvh normal")

  def test_ray_mesh(self):
    """Tests ray<>mesh matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # look at the tetrahedron
    pnt = wp.array([wp.vec3(2.0, 2.0, 2.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(-1.0, -1.0, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, 4, "geom_id")

    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_normal = np.zeros(3, dtype=np.float64)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused, mj_normal)
    _assert_eq(dist_np, mj_dist, "dist-tetrahedron")
    _assert_eq(normal_np, mj_normal, "normal-tetrahedron")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

    # look away from the dodecahedron
    pnt = wp.array([wp.vec3(4.0, 2.0, 2.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(2.0, 1.0, 1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")

    # test that bvh raycast produces the same results
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")

    # look at the dodecahedron
    pnt = wp.array([wp.vec3(4.0, 2.0, 2.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(-2.0, -1.0, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, 5, "geom_id")

    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_normal = np.zeros(3, dtype=np.float64)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused, mj_normal)
    _assert_eq(dist_np, mj_dist, "dist-dodecahedron")
    _assert_eq(normal_np, mj_normal, "normal-dodecahedron")

    # test that bvh raycast produces the same results
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "bvh geom_id")
    _assert_eq(bvh_dist_np, dist_np, "bvh dist")
    _assert_eq(bvh_normal_np, normal_np, "bvh normal")

  def test_ray_hfield(self):
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    pnt = wp.array([wp.vec3(0.0, 2.0, 2.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt.numpy()[0, 0], vec.numpy()[0, 0], None, 1, -1, mj_geomid)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]

    _assert_eq(dist_np, mj_dist, "dist")
    _assert_eq(geomid_np, mj_geomid[0], "geomid")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "bvh dist")

  def test_ray_geomgroup(self):
    """Tests ray geomgroup filter."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # hits plane with geom_group[0] = 1
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    geomgroup = vec6(1, 0, 0, 0, 0, 0)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, geomgroup=geomgroup)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 0, "geom_id")

    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, geomgroup=geomgroup, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")

    # nothing hit with geom_group[0] = 0
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    geomgroup = vec6(0, 0, 0, 0, 0, 0)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, geomgroup=geomgroup)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")

    # test that bvh raycast produces the same results
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, geomgroup=geomgroup, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")

  def test_ray_flg_static(self):
    """Tests ray flg_static filter."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # nothing hit with flg_static = False
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, flg_static=False)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, flg_static=False, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")

  def test_ray_bodyexclude(self):
    """Tests ray bodyexclude filter."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # nothing hit with bodyexclude = 0 (world body)
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, bodyexclude=0)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")
    _assert_eq(normal_np, 0, "normal")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, bodyexclude=0, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

  def test_ray_invisible(self):
    """Tests ray doesn't hit transparent geoms."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # nothing hit with transparent geoms
    m.geom_rgba = wp.array2d([[wp.vec4(0.0, 0.0, 0.0, 0.0)] * 8], dtype=wp.vec4)
    mujoco.mj_forward(mjm, mjd)

    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")
    _assert_eq(normal_np, 0, "normal")

    # test that bvh raycast produces the same results
    rc = mjw.create_render_context(mjm)
    dist, geomid, normal = mjw.ray(m, d, pnt, vec, rc=rc)
    bvh_geomid_np = geomid.numpy()[0, 0]
    bvh_dist_np = dist.numpy()[0, 0]
    bvh_normal_np = normal.numpy()[0, 0]
    _assert_eq(bvh_geomid_np, geomid_np, "geom_id")
    _assert_eq(bvh_dist_np, dist_np, "dist")
    _assert_eq(bvh_normal_np, normal_np, "normal")

  @parameterized.product(
    nworld=[1, 2],
    mode=["shared", "batched"],
  )
  def test_ray_multi_world(self, nworld, mode):
    mjm, mjd, m, d = test_data.fixture("ray.xml", nworld=nworld)

    if mode == "shared":
      pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
      vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
      expected_geomids = [0] * nworld
      expected_hits = [True] * nworld
    else:  # batched
      # World 0 ray: looking down and hits geom 0
      # World 1 ray: looking up (opposite direction) and misses (returns -1)
      all_pnts = [wp.vec3(2.0, 1.0, 3.0), wp.vec3(2.0, 1.0, 3.0)]
      all_vecs = [
        wp.normalize(wp.vec3(0.1, 0.2, -1.0)),
        wp.normalize(wp.vec3(0.1, 0.2, 1.0)),
      ]
      all_geomids = [0, -1]
      all_hits = [True, False]

      pnt = wp.array(all_pnts[:nworld], dtype=wp.vec3).reshape((nworld, 1))
      vec = wp.array(all_vecs[:nworld], dtype=wp.vec3).reshape((nworld, 1))
      expected_geomids = all_geomids[:nworld]
      expected_hits = all_hits[:nworld]

    # Non-BVH path
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)

    self.assertEqual(dist.shape, (nworld, 1))
    self.assertEqual(geomid.shape, (nworld, 1))
    self.assertEqual(normal.shape, (nworld, 1))

    expected_geomids_arr = np.array(expected_geomids, dtype=np.int32).reshape((nworld, 1))
    _assert_eq(geomid.numpy(), expected_geomids_arr, "geom_id")
    hits = np.array(expected_hits).reshape((nworld, 1))
    self.assertTrue(np.all(dist.numpy()[hits] > 0.0))
    _assert_eq(dist.numpy()[~hits], -1.0, "dist miss")

    # BVH-accelerated path
    rc = mjw.create_render_context(mjm, nworld=nworld)
    dist_bvh, geomid_bvh, normal_bvh = mjw.ray(m, d, pnt, vec, rc=rc)

    self.assertEqual(dist_bvh.shape, (nworld, 1))
    self.assertEqual(geomid_bvh.shape, (nworld, 1))
    self.assertEqual(normal_bvh.shape, (nworld, 1))

    _assert_eq(geomid_bvh.numpy(), geomid.numpy(), "geom_id bvh")
    _assert_eq(dist_bvh.numpy(), dist.numpy(), "dist bvh")
    _assert_eq(normal_bvh.numpy()[hits], normal.numpy()[hits], "normal bvh")

  def test_ray_box_miss(self):
    """Tests that rays missing box along different axes return -1."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    # In ray.xml, box is at pos=(1, 0, 1), size=(0.5, 0.25, 0.3)
    # Looking up from below (z=0.05, vec=(0,0,1)): above the box is empty space.
    vec_up = wp.array([wp.vec3(0.0, 0.0, 1.0)], dtype=wp.vec3).reshape((1, 1))

    # 1. Ray looking up at box X position but offset in +Y outside bounds (y=0.4 > 0.25)
    pnt_miss_y_pos = wp.array([wp.vec3(1.0, 0.4, 0.05)], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt_miss_y_pos, vec_up)
    _assert_eq(geomid.numpy()[0, 0], -1, "geom_id_miss_y_pos")
    _assert_eq(dist.numpy()[0, 0], -1.0, "dist_miss_y_pos")

    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_miss_y_pos.numpy()[0, 0], vec_up.numpy()[0, 0], None, 1, -1, unused)
    _assert_eq(geomid.numpy()[0, 0], unused[0], "geom_id_miss_y_pos_vs_mj")
    _assert_eq(dist.numpy()[0, 0], mj_dist, "dist_miss_y_pos_vs_mj")

    rc = mjw.create_render_context(mjm)
    dist_bvh, geomid_bvh, _ = mjw.ray(m, d, pnt_miss_y_pos, vec_up, rc=rc)
    _assert_eq(geomid_bvh.numpy()[0, 0], -1, "geom_id_bvh_miss_y_pos")
    _assert_eq(dist_bvh.numpy()[0, 0], -1.0, "dist_bvh_miss_y_pos")

    # 2. Ray looking up at box X position but offset in -Y outside bounds (y=-0.4 < -0.25)
    pnt_miss_y_neg = wp.array([wp.vec3(1.0, -0.4, 0.05)], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt_miss_y_neg, vec_up)
    _assert_eq(geomid.numpy()[0, 0], -1, "geom_id_miss_y_neg")
    _assert_eq(dist.numpy()[0, 0], -1.0, "dist_miss_y_neg")

    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_miss_y_neg.numpy()[0, 0], vec_up.numpy()[0, 0], None, 1, -1, unused)
    _assert_eq(geomid.numpy()[0, 0], unused[0], "geom_id_miss_y_neg_vs_mj")
    _assert_eq(dist.numpy()[0, 0], mj_dist, "dist_miss_y_neg_vs_mj")

    dist_bvh, geomid_bvh, _ = mjw.ray(m, d, pnt_miss_y_neg, vec_up, rc=rc)
    _assert_eq(geomid_bvh.numpy()[0, 0], -1, "geom_id_bvh_miss_y_neg")
    _assert_eq(dist_bvh.numpy()[0, 0], -1.0, "dist_bvh_miss_y_neg")

    # 3. Ray looking down from above with y=0.4: misses box (geom 3) and hits plane below (geom 0)
    pnt_miss_down = wp.array([wp.vec3(1.0, 0.4, 1.6)], dtype=wp.vec3).reshape((1, 1))
    vec_down = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt_miss_down, vec_down)
    _assert_eq(geomid.numpy()[0, 0], 0, "geom_id_miss_box_hit_plane")
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_miss_down.numpy()[0, 0], vec_down.numpy()[0, 0], None, 1, -1, unused)
    _assert_eq(geomid.numpy()[0, 0], unused[0], "geom_id_miss_box_vs_mj")
    _assert_eq(dist.numpy()[0, 0], mj_dist, "dist_miss_box_vs_mj")

    dist_bvh, geomid_bvh, _ = mjw.ray(m, d, pnt_miss_down, vec_down, rc=rc)
    _assert_eq(geomid_bvh.numpy()[0, 0], geomid.numpy()[0, 0], "geom_id_bvh_miss_box")
    _assert_eq(dist_bvh.numpy()[0, 0], dist.numpy()[0, 0], "dist_bvh_miss_box")

  def test_ray_shape_validation(self):
    """Tests shape validation assertions in ray and rays."""
    mjm, mjd, m, d = test_data.fixture("ray.xml", nworld=2)

    # Invalid world count (3 != 1 and 3 != d.nworld=2)
    pnt_bad_world = wp.zeros((3, 1), dtype=wp.vec3)
    vec_bad_world = wp.zeros((3, 1), dtype=wp.vec3)
    with self.assertRaises(AssertionError):
      mjw.ray(m, d, pnt_bad_world, vec_bad_world)

    # Mismatched pnt and vec shapes
    pnt_mismatch = wp.zeros((1, 1), dtype=wp.vec3)
    vec_mismatch = wp.zeros((2, 1), dtype=wp.vec3)
    with self.assertRaises(AssertionError):
      mjw.ray(m, d, pnt_mismatch, vec_mismatch)

    # ray() called with nray > 1 (should use rays() instead)
    pnt_multi_ray = wp.zeros((1, 2), dtype=wp.vec3)
    vec_multi_ray = wp.zeros((1, 2), dtype=wp.vec3)
    with self.assertRaises(AssertionError):
      mjw.ray(m, d, pnt_multi_ray, vec_multi_ray)

    # rays() with mismatched bodyexclude length
    bodyexclude_bad = wp.empty(1, dtype=int)
    dist_out = wp.empty((2, 2), dtype=float)
    geomid_out = wp.empty((2, 2), dtype=int)
    normal_out = wp.empty((2, 2), dtype=wp.vec3)
    with self.assertRaises(AssertionError):
      mjw.rays(
        m,
        d,
        pnt_multi_ray,
        vec_multi_ray,
        vec6(-1, -1, -1, -1, -1, -1),
        True,
        bodyexclude_bad,
        dist_out,
        geomid_out,
        normal_out,
      )

  @parameterized.product(
    nworld=[1, 2],
    mode=["shared", "batched"],
  )
  def test_rays_multi_world_multi_ray(self, nworld, mode):
    """Tests rays() with multiple rays per world across single/multiple worlds."""
    mjm, mjd, m, d = test_data.fixture("ray.xml", nworld=nworld)
    nray = 3

    # Ray 0: hits plane (geom 0)
    # Ray 1: hits sphere (geom 1)
    # Ray 2: misses everything (looking up)
    pnts_base = [
      wp.vec3(2.0, 1.0, 3.0),
      wp.vec3(0.0, 0.0, 1.6),
      wp.vec3(0.0, 0.0, 3.0),
    ]
    vecs_base = [
      wp.normalize(wp.vec3(0.1, 0.2, -1.0)),
      wp.normalize(wp.vec3(0.1, 0.2, -1.0)),
      wp.vec3(0.0, 0.0, 1.0),
    ]

    if mode == "shared":
      pnt = wp.array([pnts_base], dtype=wp.vec3).reshape((1, nray))
      vec = wp.array([vecs_base], dtype=wp.vec3).reshape((1, nray))
    else:
      pnt_list = [pnts_base for _ in range(nworld)]
      vec_list = [vecs_base for _ in range(nworld)]
      pnt = wp.array(pnt_list, dtype=wp.vec3).reshape((nworld, nray))
      vec = wp.array(vec_list, dtype=wp.vec3).reshape((nworld, nray))

    bodyexclude = wp.full(nray, -1, dtype=int)
    geomgroup = vec6(-1, -1, -1, -1, -1, -1)

    dist = wp.empty((nworld, nray), dtype=float)
    geomid = wp.empty((nworld, nray), dtype=int)
    normal = wp.empty((nworld, nray), dtype=wp.vec3)

    mjw.rays(m, d, pnt, vec, geomgroup, True, bodyexclude, dist, geomid, normal)

    self.assertEqual(dist.shape, (nworld, nray))
    self.assertEqual(geomid.shape, (nworld, nray))
    self.assertEqual(normal.shape, (nworld, nray))

    expected_geomid_arr = np.tile([0, 1, -1], (nworld, 1)).astype(np.int32)
    _assert_eq(geomid.numpy(), expected_geomid_arr, "geomid")
    self.assertTrue(np.all(dist.numpy()[:, :2] > 0.0))
    _assert_eq(dist.numpy()[:, 2], -1.0, "dist miss")

    # BVH path
    rc = mjw.create_render_context(mjm, nworld=nworld)
    dist_bvh = wp.empty((nworld, nray), dtype=float)
    geomid_bvh = wp.empty((nworld, nray), dtype=int)
    normal_bvh = wp.empty((nworld, nray), dtype=wp.vec3)

    mjw.rays(m, d, pnt, vec, geomgroup, True, bodyexclude, dist_bvh, geomid_bvh, normal_bvh, rc=rc)

    _assert_eq(geomid_bvh.numpy(), geomid.numpy(), "bvh geomid")
    _assert_eq(dist_bvh.numpy(), dist.numpy(), "bvh dist")
    _assert_eq(normal_bvh.numpy()[:, :2], normal.numpy()[:, :2], "bvh normal")


if __name__ == "__main__":
  absltest.main()
