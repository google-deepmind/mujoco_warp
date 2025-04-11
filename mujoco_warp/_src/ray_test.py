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

import mujoco_warp as mjwarp

from . import test_util
from .types import vec6

# tolerance for difference between MuJoCo and MJX ray calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class RayTest(absltest.TestCase):
  def test_ray_nothing(self):
    """Tests that ray returns -1 when nothing is hit."""
    m, d, _, _ = test_util.fixture("ray.xml")
    mujoco.mj_forward(m, d)
    mx, dx = mjwarp.put_model(m), mjwarp.put_data(m, d)

    pnt = wp.array([wp.vec3(12.146, 1.865, 3.895)], dtype=wp.vec3)
    vec = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]  # Extract from [[-1]]
    dist_np = dist.numpy()[0][0]  # Extract from [[-1.]]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")

  def test_ray_plane(self):
    """Tests ray<>plane matches MuJoCo."""
    m, d, _, _ = test_util.fixture("ray.xml")
    mujoco.mj_forward(m, d)
    mx, dx = mjwarp.put_model(m), mjwarp.put_data(m, d)

    # looking down at a slight angle
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, 0, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0], vec.numpy()[0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(m, d, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # looking on wrong side of plane
    pnt = wp.array([wp.vec3(0.0, 0.0, -0.5)], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")

  def test_ray_sphere(self):
    """Tests ray<>sphere matches MuJoCo."""
    m, d, _, _ = test_util.fixture("ray.xml")
    mujoco.mj_forward(m, d)
    mx, dx = mjwarp.put_model(m), mjwarp.put_data(m, d)

    # looking down at sphere at a slight angle
    pnt = wp.array([wp.vec3(0.0, 0.0, 1.6)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, 1, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0], vec.numpy()[0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(m, d, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

  def test_ray_capsule(self):
    """Tests ray<>capsule matches MuJoCo."""
    m, d, _, _ = test_util.fixture("ray.xml")
    mujoco.mj_forward(m, d)
    mx, dx = mjwarp.put_model(m), mjwarp.put_data(m, d)

    # looking down at capsule at a slight angle
    pnt = wp.array([wp.vec3(0.5, 1.0, 1.6)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, -1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, 2, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0], vec.numpy()[0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(m, d, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # looking up at capsule from below
    pnt = wp.array([wp.vec3(-0.5, 1.0, 0.05)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, 1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, 2, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0], vec.numpy()[0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(m, d, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # looking at cylinder of capsule from the side
    pnt = wp.array([wp.vec3(0.0, 1.0, 0.75)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(1.0, 0.0, 0.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, 2, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0], vec.numpy()[0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(m, d, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

  def test_ray_box(self):
    """Tests ray<>box matches MuJoCo."""
    m, d, _, _ = test_util.fixture("ray.xml")
    mujoco.mj_forward(m, d)
    mx, dx = mjwarp.put_model(m), mjwarp.put_data(m, d)

    # looking down at box at a slight angle
    pnt = wp.array([wp.vec3(1.0, 0.0, 1.6)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, -1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, 3, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0], vec.numpy()[0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(m, d, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # looking up at box from below
    pnt = wp.array([wp.vec3(1.0, 0.0, 0.05)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(0.0, 0.05, 1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, 3, "geom_id")
    pnt_np, vec_np = pnt.numpy()[0], vec.numpy()[0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(m, d, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

  def test_ray_mesh(self):
    """Tests ray<>mesh matches MuJoCo."""
    m, d, _, _ = test_util.fixture("ray.xml")
    mujoco.mj_forward(m, d)
    mx, dx = mjwarp.put_model(m), mjwarp.put_data(m, d)

    # look at the tetrahedron
    pnt = wp.array([wp.vec3(2.0, 2.0, 2.0)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(-1.0, -1.0, -1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, 4, "geom_id")

    pnt_np, vec_np = pnt.numpy()[0], vec.numpy()[0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(m, d, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist-tetrahedron")

    # look away from the dodecahedron
    pnt = wp.array([wp.vec3(4.0, 2.0, 2.0)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(2.0, 1.0, 1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    _assert_eq(geomid_np, -1, "geom_id")

    # look at the dodecahedron
    pnt = wp.array([wp.vec3(4.0, 2.0, 2.0)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(-2.0, -1.0, -1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, 5, "geom_id")

    pnt_np, vec_np = pnt.numpy()[0], vec.numpy()[0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(m, d, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist-dodecahedron")

  def test_ray_geomgroup(self):
    """Tests ray geomgroup filter."""
    m, d, _, _ = test_util.fixture("ray.xml")
    mujoco.mj_forward(m, d)
    mx, dx = mjwarp.put_model(m), mjwarp.put_data(m, d)

    # hits plane with geom_group[0] = 1
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3)
    geomgroup = vec6(1, 0, 0, 0, 0, 0)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec, geomgroup=geomgroup)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, 0, "geom_id")

    pnt_np, vec_np = pnt.numpy()[0], vec.numpy()[0]
    unused = np.zeros(1, dtype=np.int32)
    mj_dist = mujoco.mj_ray(m, d, pnt_np, vec_np, None, 1, -1, unused)
    _assert_eq(dist_np, mj_dist, "dist")

    # nothing hit with geom_group[0] = 0
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3)
    geomgroup = vec6(0, 0, 0, 0, 0, 0)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec, geomgroup=geomgroup)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")

  def test_ray_flg_static(self):
    """Tests ray flg_static filter."""
    m, d, _, _ = test_util.fixture("ray.xml")
    mujoco.mj_forward(m, d)
    mx, dx = mjwarp.put_model(m), mjwarp.put_data(m, d)

    # nothing hit with flg_static = False
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec, flg_static=False)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")

  def test_ray_bodyexclude(self):
    """Tests ray bodyexclude filter."""
    m, d, _, _ = test_util.fixture("ray.xml")
    mujoco.mj_forward(m, d)
    mx, dx = mjwarp.put_model(m), mjwarp.put_data(m, d)

    # nothing hit with bodyexclude = 0 (world body)
    pnt = wp.array([wp.vec3(2.0, 1.0, 3.0)], dtype=wp.vec3)
    vec = wp.array([wp.normalize(wp.vec3(0.1, 0.2, -1.0))], dtype=wp.vec3)
    dist, geomid = mjwarp.ray_geom(mx, dx, pnt, vec, bodyexclude=0)
    wp.synchronize()
    geomid_np = geomid.numpy()[0][0]
    dist_np = dist.numpy()[0][0]
    _assert_eq(geomid_np, -1, "geom_id")
    _assert_eq(dist_np, -1, "dist")


if __name__ == "__main__":
  absltest.main()
