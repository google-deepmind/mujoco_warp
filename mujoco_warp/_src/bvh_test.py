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
"""Tests for BVH functions."""

import dataclasses

import numpy as np
import warp as wp
from absl.testing import absltest

from mujoco_warp import test_data

from . import bvh


@dataclasses.dataclass
class MinimalRenderContext:
  bvh_ngeom: int
  enabled_geom_ids: wp.array
  mesh_bounds_size: wp.array
  hfield_bounds_size: wp.array
  lower: wp.array
  upper: wp.array
  group: wp.array
  group_root: wp.array
  bvh: wp.Bvh = None
  bvh_id: wp.uint64 = None


def _create_minimal_context(mjm, d, enabled_geom_groups=None):
  if enabled_geom_groups is None:
    enabled_geom_groups = [0, 1, 2]

  geom_enabled_idx = [i for i in range(mjm.ngeom) if mjm.geom_group[i] in enabled_geom_groups]
  bvh_ngeom = len(geom_enabled_idx)

  return MinimalRenderContext(
    bvh_ngeom=bvh_ngeom,
    enabled_geom_ids=wp.array(geom_enabled_idx, dtype=int),
    mesh_bounds_size=wp.zeros(max(mjm.nmesh, 1), dtype=wp.vec3),
    hfield_bounds_size=wp.zeros(max(mjm.nhfield, 1), dtype=wp.vec3),
    lower=wp.zeros(d.nworld * bvh_ngeom, dtype=wp.vec3),
    upper=wp.zeros(d.nworld * bvh_ngeom, dtype=wp.vec3),
    group=wp.zeros(d.nworld * bvh_ngeom, dtype=int),
    group_root=wp.zeros(d.nworld, dtype=int),
  )


class BvhTest(absltest.TestCase):
  def test_compute_bvh_bounds(self):
    """Tests that _compute_bvh_bounds kernel computes valid bounds."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml")
    rc = _create_minimal_context(mjm, d)

    wp.launch(
      kernel=bvh._compute_bvh_bounds,
      dim=(d.nworld, rc.bvh_ngeom),
      inputs=[
        m.geom_type,
        m.geom_dataid,
        m.geom_size,
        d.geom_xpos,
        d.geom_xmat,
        d.nworld,
        rc.bvh_ngeom,
        rc.enabled_geom_ids,
        rc.mesh_bounds_size,
        rc.hfield_bounds_size,
        rc.lower,
        rc.upper,
        rc.group,
      ],
    )

    lower = rc.lower.numpy()
    upper = rc.upper.numpy()

    np.testing.assert_allclose(lower < upper, True, err_msg="lower < upper")

  def test_compute_bvh_bounds_multiworld(self):
    """Tests bounds computation with multiple worlds."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=4)
    rc = _create_minimal_context(mjm, d)

    wp.launch(
      kernel=bvh._compute_bvh_bounds,
      dim=(d.nworld, rc.bvh_ngeom),
      inputs=[
        m.geom_type,
        m.geom_dataid,
        m.geom_size,
        d.geom_xpos,
        d.geom_xmat,
        d.nworld,
        rc.bvh_ngeom,
        rc.enabled_geom_ids,
        rc.mesh_bounds_size,
        rc.hfield_bounds_size,
        rc.lower,
        rc.upper,
        rc.group,
      ],
    )

    self.assertEqual(rc.lower.shape[0], 4 * rc.bvh_ngeom, "lower")
    np.testing.assert_array_equal(rc.group.numpy(), np.repeat(np.arange(4), rc.bvh_ngeom), err_msg="group")

  def test_build_scene_bvh(self):
    """Tests that build_scene_bvh creates a valid BVH."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml")
    rc = _create_minimal_context(mjm, d)

    bvh.build_scene_bvh(m, d, rc)

    self.assertIsNotNone(rc.bvh, "bvh")
    self.assertIsNotNone(rc.bvh_id, "bvh_id")

  def test_build_scene_bvh_multiworld(self):
    """Tests BVH construction with multiple worlds."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=8)
    rc = _create_minimal_context(mjm, d)

    bvh.build_scene_bvh(m, d, rc)

    self.assertEqual(rc.lower.shape, (8 * rc.bvh_ngeom,), "lower")
    self.assertEqual(rc.group_root.shape, (8,), "group_root")

  def test_refit_scene_bvh(self):
    """Tests that refit_scene_bvh updates bounds correctly."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml")
    rc = _create_minimal_context(mjm, d)

    bvh.build_scene_bvh(m, d, rc)

    lower_before = rc.lower.numpy()

    geom_xpos = d.geom_xpos.numpy()
    geom_xpos[:, :, 2] += 1.0
    d.geom_xpos = wp.array(geom_xpos, dtype=wp.vec3)

    bvh.refit_scene_bvh(m, d, rc)

    lower_after = rc.lower.numpy()
    self.assertFalse(np.array_equal(lower_before, lower_after), "lower_before != lower_after")

  def test_compute_bvh_group_roots(self):
    """Tests that group roots are computed for each world."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml")
    rc = _create_minimal_context(mjm, d)

    bvh.build_scene_bvh(m, d, rc)

    group_root = rc.group_root.numpy()
    self.assertEqual(len(group_root), d.nworld, "group_root")

  def test_compute_bvh_group_roots_multiworld(self):
    """Tests that each world has a unique group root."""
    mjm, mjd, m, d = test_data.fixture("primitives.xml", nworld=16)
    rc = _create_minimal_context(mjm, d)

    bvh.build_scene_bvh(m, d, rc)

    group_root = rc.group_root.numpy()
    self.assertEqual(rc.group_root.shape[0], 16, "group_root")
    self.assertEqual(len(set(group_root)), 16, "group_root")

  def test_build_mesh_bvh(self):
    """Tests that build_mesh_bvh creates a valid BVH."""
    mjm, mjd, m, d = test_data.fixture("ray.xml")

    mesh, half = bvh.build_mesh_bvh(mjm, 0)

    self.assertNotEqual(mesh.id, wp.uint64(0), "mesh id")
    self.assertFalse(np.array_equal(np.array(half), np.array([0.0, 0.0, 0.0])), "mesh half size")

  def test_accumulate_flex_vertex_normals(self):
    """Tests flex vertex normal accumulation kernel."""
    nworld = 2
    nvert = 4
    nelem = 2

    flexvert_xpos = wp.array(
      [
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
      ],
      dtype=wp.vec3,
    )
    flex_elem = wp.array([0, 1, 2, 1, 3, 2], dtype=int)
    flexvert_norm = wp.zeros((nworld, nvert), dtype=wp.vec3)

    wp.launch(
      kernel=bvh.accumulate_flex_vertex_normals,
      dim=(nworld, nelem),
      inputs=[flex_elem, flexvert_xpos],
      outputs=[flexvert_norm],
    )

    normals = flexvert_norm.numpy()
    self.assertTrue(np.any(normals != 0), "flexvert_norm")

  def test_normalize_vertex_normals(self):
    """Tests flex vertex normal normalization kernel."""
    nworld = 1
    nvert = 3

    flexvert_norm = wp.array(
      [[[0, 0, 2], [0, 3, 0], [4, 0, 0]]],
      dtype=wp.vec3,
    )

    wp.launch(
      kernel=bvh.normalize_vertex_normals,
      dim=(nworld, nvert),
      inputs=[flexvert_norm],
    )

    normals = flexvert_norm.numpy()
    for i in range(nvert):
      norm = np.linalg.norm(normals[0, i])
      np.testing.assert_allclose(norm, 1.0, rtol=1e-5, err_msg="flexvert_norm")


if __name__ == "__main__":
  wp.init()
  absltest.main()
