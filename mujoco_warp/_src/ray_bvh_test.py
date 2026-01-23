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
"""Tests for BVH-accelerated ray casting functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src.types import vec6

# Tolerance for difference between MuJoCo and mujoco_warp ray calculations
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _ray_bvh_single(m, d, rc, pnt_np, vec_np, geomgroup=None, flg_static=True, bodyexclude=-1):
  """Helper to cast a single ray using BVH and return numpy results."""
  pnt = wp.array([wp.vec3(*pnt_np)], dtype=wp.vec3).reshape((1, 1))
  vec = wp.array([wp.vec3(*vec_np)], dtype=wp.vec3).reshape((1, 1))

  dist = wp.zeros((1, 1), dtype=float)
  geomid = wp.zeros((1, 1), dtype=int)
  normal = wp.zeros((1, 1), dtype=wp.vec3)
  bodyexclude_arr = wp.array([bodyexclude], dtype=int)

  if geomgroup is None:
    geomgroup = vec6(-1, -1, -1, -1, -1, -1)

  mjw.rays_bvh(m, d, rc, pnt, vec, geomgroup, flg_static, bodyexclude_arr, dist, geomid, normal)
  wp.synchronize()

  return dist.numpy()[0, 0], geomid.numpy()[0, 0], normal.numpy()[0, 0]


class RayBvhTest(absltest.TestCase):
  """Tests for BVH-accelerated ray casting."""

  def test_bvh_matches_brute_force(self):
    """BVH ray casting should match brute-force results."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    test_rays = [
      ([0.0, 0.0, 1.6], [0.1, 0.2, -1.0]),  # Looking at sphere
      ([1.0, 0.0, 1.6], [0.0, 0.05, -1.0]),  # Looking at box
      ([0.5, 1.0, 1.6], [0.0, 0.05, -1.0]),  # Looking at capsule
      ([2.0, 1.0, 3.0], [0.1, 0.2, -1.0]),  # Looking at plane
      ([12.0, 1.0, 3.0], [0.0, 0.0, -1.0]),  # Looking at nothing
    ]

    for pnt_np, vec_np in test_rays:
      vec_np = np.array(vec_np)
      vec_np = vec_np / np.linalg.norm(vec_np)

      bvh_dist, bvh_geomid, _ = _ray_bvh_single(m, d, rc, pnt_np, vec_np)

      pnt = wp.array([wp.vec3(*pnt_np)], dtype=wp.vec3).reshape((1, 1))
      vec = wp.array([wp.vec3(*vec_np)], dtype=wp.vec3).reshape((1, 1))
      bf_dist, bf_geomid, _ = mjw.ray(m, d, pnt, vec)
      wp.synchronize()
      bf_dist = bf_dist.numpy()[0, 0]
      bf_geomid = bf_geomid.numpy()[0, 0]

      _assert_eq(bvh_geomid, bf_geomid, f"geomid for ray from {pnt_np}")
      _assert_eq(bvh_dist, bf_dist, f"dist for ray from {pnt_np}")

  def test_ray_nothing(self):
    """Tests that ray returns -1 when nothing is hit."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    pnt = wp.array([wp.vec3(12.146, 1.865, 3.895)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]  # Extract from [[-1]]
    dist_np = dist.numpy()[0, 0]  # Extract from [[-1.]]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")
    _assert_eq(normal_np, 0, "normal")

  def test_ray_plane(self):
    """Tests ray<>plane matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    # looking down at a slight angle
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 0, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # looking on wrong side of plane
    pnt = wp.array([wp.vec3(0.0, 0.0, -0.5)], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")

  def test_ray_sphere(self):
    """Tests ray<>sphere matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    # looking down at sphere at a slight angle
    pnt = wp.array([wp.vec3(0.0, 0.0, 1.6)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 1, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

  def test_ray_capsule(self):
    """Tests ray<>capsule matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    # looking down at capsule at a slight angle
    pnt = wp.array([wp.vec3(0.5, 1.0, 1.6)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 2, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # looking up at capsule from below
    pnt = wp.array([wp.vec3(-0.5, 1.0, 0.05)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, 1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray(m, d, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 2, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # looking at cylinder of capsule from the side
    pnt = wp.array([wp.vec3(0.0, 1.0, 0.75)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(1.0, 0.0, 0.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 2, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

  def test_ray_cylinder(self):
    """Tests ray<>cylinder matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)
    pnt = wp.array([wp.vec3(2.0, 0.0, 0.05)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, 1.0))], dtype=wp.vec3).reshape((1, 1))

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt.numpy()[0, 0], vec.numpy()[0, 0], None, 1, -1, mj_geomid)
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)

    _assert_eq(geomid.numpy()[0, 0], mj_geomid[0], "geomid")
    _assert_eq(dist.numpy()[0, 0], mj_dist, "dist")

  def test_ray_box(self):
    """Tests ray<>box matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    # looking down at box at a slight angle
    pnt = wp.array([wp.vec3(1.0, 0.0, 1.6)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 3, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # looking up at box from below
    pnt = wp.array([wp.vec3(1.0, 0.0, 0.05)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, 1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 3, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

  def test_ray_mesh(self):
    """Tests ray<>mesh matches MuJoCo."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    # look at the tetrahedron
    pnt = wp.array([wp.vec3(2.0, 2.0, 2.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(-1.0, -1.0, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 4, "geom_id")

    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist-tetrahedron")

    # look away from the dodecahedron
    pnt = wp.array([wp.vec3(4.0, 2.0, 2.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(2.0, 1.0, 1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")

    # look at the dodecahedron
    pnt = wp.array([wp.vec3(4.0, 2.0, 2.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(-2.0, -1.0, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 5, "geom_id")

    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist-dodecahedron")

  def test_ray_hfield(self):
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)
    pnt = wp.array([wp.vec3(0.0, 2.0, 2.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)

    mj_geomid = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt.numpy()[0, 0], vec.numpy()[0, 0], None, 1, -1, mj_geomid)

    _assert_eq(dist.numpy()[0, 0], mj_dist, "dist")
    _assert_eq(geomid.numpy()[0, 0], mj_geomid[0], "geomid")

  def test_ray_geomgroup(self):
    """Tests ray geomgroup filter."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    # hits plane with geom_group[0] = 1
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    geomgroup = vec6(1, 0, 0, 0, 0, 0)
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec, geomgroup=geomgroup)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, 0, "geom_id")

    pnt_np, vec_np = pnt.numpy()[0, 0], vec.numpy()[0, 0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(mjm, mjd, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # nothing hit with geom_group[0] = 0
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    geomgroup = vec6(0, 0, 0, 0, 0, 0)
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec, geomgroup=geomgroup)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")

  def test_ray_flg_static(self):
    """Tests ray flg_static filter."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    # nothing hit with flg_static = False
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec, flg_static=False)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")

  def test_ray_bodyexclude(self):
    """Tests ray bodyexclude filter."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    # nothing hit with bodyexclude = 0 (world body)
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec, bodyexclude=0)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")
    _assert_eq(normal_np, 0, "normal")

  def test_ray_invisible(self):
    """Tests ray doesn't hit transparent geoms."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")
    rc = mjw.create_render_context(mjm, m, d)

    # nothing hit with transparent geoms
    m.geom_rgba = wp.array2d([[wp.vec4(0.0, 0.0, 0.0, 0.0)] * 8], dtype=wp.vec4)
    mujoco.mj_forward(mjm, mjd)

    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3).reshape((1, 1))
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3).reshape((1, 1))
    dist, geomid, normal = mjw.ray_bvh(m, d, rc, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0, 0]
    dist_np = dist.numpy()[0, 0]
    normal_np = normal.numpy()[0, 0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")
    _assert_eq(normal_np, 0, "normal")


if __name__ == "__main__":
  absltest.main()
