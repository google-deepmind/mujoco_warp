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

"""Tests for SDF collisions, gradient evaluation, and dynamic stability."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import collision_sdf
from mujoco_warp._src.types import vec8
from mujoco_warp._src.types import vec8i

_CUBE_VERTS = "1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"
_UNIT_CUBE_VERTS = (
  "-0.5 -0.5 -0.5  0.5 -0.5 -0.5  -0.5 0.5 -0.5  0.5 0.5 -0.5  -0.5 -0.5 0.5  0.5 -0.5 0.5  -0.5 0.5 0.5  0.5 0.5 0.5"
)


def _cube_pair_xml(
  geom1: str = 'type="mesh" mesh="cube_mesh"',
  pos: str = "0 0 1.95",
  euler: str = "",
  scale: str = "",
  sdf_mesh: str = "cube_sdf",
  geom_attrs: str = "",
  verts: str = _CUBE_VERTS,
  sensor_cutoff: float | None = None,
) -> str:
  scale_attr = f' scale="{scale}"' if scale else ""
  euler_attr = f' euler="{euler}"' if euler else ""
  extra_attr = f" {geom_attrs}" if geom_attrs else ""
  sensor = f'<sensor><distance geom1="g1" geom2="g2" cutoff="{sensor_cutoff}"/></sensor>' if sensor_cutoff is not None else ""
  return f"""
  <mujoco>
    <asset>
      <mesh name="cube_mesh"{scale_attr} vertex="{verts}"/>
      <mesh name="cube_sdf"{scale_attr} vertex="{verts}"/>
    </asset>
    <worldbody>
      <body>
        <geom name="g1" {geom1}{extra_attr}/>
      </body>
      <body pos="{pos}"{euler_attr}>
        <freejoint/>
        <geom name="g2" type="sdf" mesh="{sdf_mesh}"{extra_attr}/>
      </body>
    </worldbody>
    {sensor}
  </mujoco>
  """


@wp.func
def _make_volume(
  # Model:
  oct_child: wp.array[vec8i],
  oct_aabb: wp.array2d[wp.vec3],
  oct_coeff: wp.array[vec8],
) -> collision_sdf.VolumeData:
  vd = collision_sdf.VolumeData()
  vd.oct_child = oct_child
  vd.oct_aabb = oct_aabb
  vd.oct_coeff = oct_coeff
  vd.root = 0
  vd.center = oct_aabb[0, 0]
  vd.half_size = oct_aabb[0, 1]
  vd.valid = True
  return vd


def _run_collision(m, d, w1_dz: float | None = None, w1_z: float | None = None):
  """Optionally perturbs world 1 z-position, poisons contact outputs, and runs collision."""
  if d.nworld == 2 and (w1_dz is not None or w1_z is not None):
    qpos = d.qpos.numpy()
    qpos[1, 2] = qpos[1, 2] + w1_dz if w1_dz is not None else w1_z
    d.qpos.assign(qpos)
    mjw.kinematics(m, d)

  d.contact.dist.fill_(wp.inf)
  d.contact.pos.fill_(wp.inf)
  d.contact.frame.fill_(wp.inf)
  d.contact.includemargin.fill_(wp.inf)
  d.contact.worldid.fill_(-1)
  d.nacon.fill_(-1)

  mjw.collision(m, d)
  nacon = int(d.nacon.numpy()[0])
  return (
    nacon,
    d.contact.dist.numpy()[:nacon],
    d.contact.pos.numpy()[:nacon],
    d.contact.frame.numpy()[:nacon, 0],
    d.contact.worldid.numpy()[:nacon],
  )


class CollisionSdfTest(parameterized.TestCase):
  def test_sample_volume_sdf_values(self):
    """Tests sample_volume_sdf on a cube octree SDF."""
    _, _, m, _ = test_data.fixture(
      xml=f'<mujoco><asset><mesh name="cube" vertex="{_CUBE_VERTS}"/></asset>'
      '<worldbody><geom type="sdf" mesh="cube"/></worldbody></mujoco>'
    )

    @wp.kernel
    def eval_sdf(
      oct_child: wp.array[collision_sdf.vec8i],
      oct_aabb: wp.array2d[wp.vec3],
      oct_coeff: wp.array[collision_sdf.vec8],
      points: wp.array[wp.vec3],
      out_sdf: wp.array[float],
    ):
      tid = wp.tid()
      out_sdf[tid] = collision_sdf.sample_volume_sdf(points[tid], _make_volume(oct_child, oct_aabb, oct_coeff))

    test_pts = [
      wp.vec3(0.0, 0.0, 0.0),  # center
      wp.vec3(1.0, 0.0, 0.0),  # surface
      wp.vec3(0.0, 1.0, 0.0),  # surface
      wp.vec3(0.0, 0.0, 1.0),  # surface
      wp.vec3(1.5, 0.0, 0.0),  # exterior
    ]
    pts_wp = wp.array(test_pts, dtype=wp.vec3)
    out_wp = wp.zeros(len(test_pts), dtype=float)
    wp.launch(eval_sdf, dim=len(test_pts), inputs=[m.oct_child, m.oct_aabb, m.oct_coeff, pts_wp, out_wp])

    vals = out_wp.numpy()
    self.assertLess(vals[0], -0.5)
    for v in vals[1:4]:
      self.assertAlmostEqual(v, 0.0, delta=0.08)
    self.assertGreater(vals[4], 0.3)

  @parameterized.product(
    nworld=[1, 2],
    case=[
      ("mesh", 'margin="0.1"', _CUBE_VERTS, "0 0 2.05", 2.07, (0.05, 0.07), 0.2, 1.0),
      ("sdf", 'margin="0.1"', _CUBE_VERTS, "0 0 2.05", 2.07, (0.05, 0.07), 0.2, 1.0),
      ("box", 'margin="0.1"', _CUBE_VERTS, "0 0 2.05", 2.07, (0.05, 0.07), 0.2, 1.0),
      ("mesh", 'gap="0.05"', _UNIT_CUBE_VERTS, "0 0 1.04", 1.06, (0.04, 0.06), 0.0, 0.5),
      ("sdf", 'gap="0.05"', _UNIT_CUBE_VERTS, "0 0 1.04", 1.06, (0.04, 0.06), 0.0, 0.5),
    ],
  )
  def test_sdf_margin_gap_clearance(self, nworld, case):
    """Tests positive clearance detection for separated pairs within margin or gap."""
    geomtype, geom_attrs, verts, pos, w1_z, seps, includemargin, z_surface = case
    geom = {"mesh": 'type="mesh" mesh="cube_mesh"', "sdf": 'type="sdf" mesh="cube_mesh"', "box": 'type="box" size="1 1 1"'}[
      geomtype
    ]
    _, _, m, d = test_data.fixture(
      xml=_cube_pair_xml(geom1=geom, pos=pos, geom_attrs=geom_attrs, verts=verts),
      nworld=nworld,
    )

    nacon, dists, positions, _, worldids = _run_collision(m, d, w1_z=w1_z)
    self.assertGreater(nacon, 0, "Expected margin/gap contacts to be generated")
    np.testing.assert_allclose(d.contact.includemargin.numpy()[:nacon], includemargin, atol=1e-6)

    for w, sep in enumerate(seps[:nworld]):
      wdists = dists[worldids == w]
      wpos_z = positions[worldids == w, 2]
      self.assertGreater(len(wdists), 0)
      closest = np.argmin(wdists)
      np.testing.assert_allclose(wdists[closest], sep, atol=0.01, err_msg=f"world {w}")
      np.testing.assert_allclose(wpos_z[closest], z_surface + 0.5 * sep, atol=0.02, err_msg=f"world {w}")

  def test_frank_wolfe_triangle_convergence(self):
    """Tests Frank-Wolfe optimization convergence on a triangle simplex using the kernel func."""
    _, _, m, _ = test_data.fixture(
      xml=f'<mujoco><asset><mesh name="cube" vertex="{_CUBE_VERTS}"/></asset>'
      '<worldbody><geom type="sdf" mesh="cube"/></worldbody></mujoco>'
    )

    @wp.kernel
    def run_fw(
      oct_child: wp.array[collision_sdf.vec8i],
      oct_aabb: wp.array2d[wp.vec3],
      oct_coeff: wp.array[collision_sdf.vec8],
      out_pt: wp.array[wp.vec3],
      out_dist: wp.array[float],
      out_normal: wp.array[wp.vec3],
    ):
      vd = _make_volume(oct_child, oct_aabb, oct_coeff)
      v0, v1, v2 = wp.vec3(1.5, -0.5, -0.5), wp.vec3(1.5, 0.5, -0.5), wp.vec3(1.5, 0.0, 0.5)
      dist, x, normal = collision_sdf._frank_wolfe_triangle(
        v0, v1, v2, vd, -1, collision_sdf.vec_pluginattr(), 30, 1.0 / 3.0, 1.0 / 3.0
      )
      out_pt[0] = x
      out_dist[0] = dist
      out_normal[0] = normal

    out_pt = wp.zeros(1, dtype=wp.vec3)
    out_dist = wp.zeros(1, dtype=float)
    out_normal = wp.zeros(1, dtype=wp.vec3)
    wp.launch(run_fw, dim=1, inputs=[m.oct_child, m.oct_aabb, m.oct_coeff, out_pt, out_dist, out_normal])

    pt, dist, normal = out_pt.numpy()[0], out_dist.numpy()[0], out_normal.numpy()[0]
    self.assertAlmostEqual(pt[0], 1.5, delta=1e-3)
    self.assertAlmostEqual(dist, 0.5, delta=0.05)
    self.assertAlmostEqual(np.linalg.norm(normal), 1.0, delta=1e-3)
    self.assertGreater(normal[0], 0.95)

  @parameterized.parameters(1, 2)
  def test_mesh_sdf_dynamic_stability(self, nworld):
    """Tests dynamic stability over 10 timesteps of a freejoint SDF mesh cube."""
    _, _, m, d = test_data.fixture(xml=_cube_pair_xml(pos="0 0 2.2", euler="30 0 0"), nworld=nworld)

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1, 0] += 0.1  # offset world 1 horizontally
      d.qpos.assign(qpos)

    if wp.get_device().is_cuda:
      with wp.ScopedCapture() as capture:
        mjw.step(m, d)
      for _ in range(10):
        wp.capture_launch(capture.graph)
        self.assertGreater(d.nacon.numpy()[0], 0, "Expected collisions during dynamic step")
    else:
      for _ in range(10):
        mjw.step(m, d)
        self.assertGreater(d.nacon.numpy()[0], 0, "Expected collisions during dynamic step")

    pos, vel = d.qpos.numpy(), d.qvel.numpy()
    self.assertFalse(np.isnan(pos).any(), "NaN found in qpos")
    self.assertFalse(np.isnan(vel).any(), "NaN found in qvel")

    for w in range(nworld):
      self.assertGreater(pos[w, 2], 2.0, f"Cube fell through bottom in world {w}: z={pos[w, 2]}")
      self.assertLess(pos[w, 2], 2.5, f"Cube launched upward in world {w}: z={pos[w, 2]}")
      self.assertLess(np.linalg.norm(vel[w, :3]), 5.0, f"Velocity exploded in world {w}: {vel[w, :3]}")

    if nworld == 2:
      self.assertFalse(np.allclose(pos[0], pos[1]))

  @parameterized.parameters("cube_mesh", "cube_sdf")
  def test_mesh_sdf_octree_mesh(self, mesh):
    """Tests that mesh geoms collide via their triangles even if the mesh has an octree."""
    _, _, m, d = test_data.fixture(xml=_cube_pair_xml(sdf_mesh=mesh))

    # the mesh has an octree when shared with the SDF geom, contacts are the same in both cases
    nacon, dist, _, normal, _ = _run_collision(m, d)
    self.assertGreater(nacon, 0)
    self.assertTrue(np.all((dist > -0.051) & (dist < 1e-3)), dist)
    deepest = np.argmin(dist)
    np.testing.assert_allclose(dist[deepest], -0.05, atol=1e-3)
    np.testing.assert_allclose(normal[deepest], [0.0, 0.0, 1.0], atol=1e-3)

  @parameterized.parameters(1, 2)
  def test_mesh_sdf_small_scale(self, nworld):
    """Tests SDF collision on sub-centimeter scale geometry (10mm cube)."""
    _, _, m, d = test_data.fixture(xml=_cube_pair_xml(pos="0 0 0.0095", scale="0.005 0.005 0.005"), nworld=nworld)

    nacon, dists, _, _, worldids = _run_collision(m, d, w1_dz=0.0002)
    self.assertGreater(nacon, 0, "Expected contacts for small-scale geometry")
    self.assertLess(np.min(dists), -0.0001, f"Expected penetration distance: {dists}")
    self.assertTrue(np.all(dists <= 1e-6), f"Expected non-positive distances: {dists}")

    if nworld == 2:
      self.assertFalse(np.allclose(np.mean(dists[worldids == 0]), np.mean(dists[worldids == 1])))

  @parameterized.parameters(1, 2)
  def test_sdf_sdf_contact_position_midpoint(self, nworld):
    """Tests that SDF-SDF contact position offset has the correct sign along normal."""
    _, _, m, d = test_data.fixture(xml=_cube_pair_xml(geom1='type="sdf" mesh="cube_mesh"', pos="0 0 1.9"), nworld=nworld)

    nacon, dists, positions, normals, worldids = _run_collision(m, d, w1_dz=-0.05)
    self.assertGreater(nacon, 0, "Expected contacts between colliding SDF cubes")
    self.assertTrue(np.all(dists < 0.0), f"Expected penetration dist < 0, got {dists}")
    self.assertTrue(np.all(normals[:, 2] > 0.9), f"Expected normals along +Z, got {normals}")

    # contact position offset must follow: pos + normal * (0.5 * dist)
    mean_z = float(np.mean(positions[:, 2]))
    self.assertLessEqual(mean_z, 0.95, f"Contact pos Z shifted in wrong direction: {mean_z} > 0.95")

    if nworld == 2:
      self.assertFalse(np.allclose(np.mean(positions[worldids == 0, 2]), np.mean(positions[worldids == 1, 2])))

  @parameterized.parameters(
    (1, "0 0 1.95", 0.0, 5.0, (-0.05,)),
    (2, "0 0 1.95", 0.02, 5.0, (-0.05, -0.03)),
    (1, "0 0 3", 0.0, 10.0, (10.0,)),
  )
  def test_sdf_collision_sensor(self, nworld, pos, w1_dz, cutoff, expected):
    """Tests collision distance sensor for penetrating and separated mesh-SDF pairs."""
    _, _, m, d = test_data.fixture(xml=_cube_pair_xml(pos=pos, sensor_cutoff=cutoff), nworld=nworld)

    if nworld == 2 and w1_dz != 0.0:
      qpos = d.qpos.numpy()
      qpos[1, 2] += w1_dz
      d.qpos.assign(qpos)

    d.sensordata.fill_(wp.inf)
    d.nacon.fill_(-1)
    mjw.forward(m, d)

    np.testing.assert_allclose(d.sensordata.numpy()[:, 0], expected, atol=1e-3)

  @parameterized.parameters(1, 2)
  def test_mesh_sdf_large_triangle_interior(self, nworld):
    """Tests Mesh-SDF contact when a large mesh triangle's vertices lie outside the SDF AABB."""
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <asset>
          <mesh name="cube" vertex="{_UNIT_CUBE_VERTS}"/>
          <mesh name="plate" vertex="-5 -5 -0.1  5 -5 -0.1  5 5 -0.1  -5 5 -0.1  -5 -5 0.1  5 -5 0.1  5 5 0.1  -5 5 0.1"/>
        </asset>
        <worldbody>
          <body pos="0 0 -0.05"><geom type="mesh" mesh="plate"/></body>
          <body pos="0 0 0.5"><freejoint/><geom type="sdf" mesh="cube"/></body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    nacon, dists, _, _, worldids = _run_collision(m, d, w1_z=0.48)
    self.assertGreater(nacon, 0)

    for w, z in enumerate([0.5, 0.48][:nworld]):
      mjd.qpos[2] = z
      mujoco.mj_forward(mjm, mjd)
      w_mask = worldids == w
      self.assertTrue(np.any(w_mask))
      np.testing.assert_allclose(np.min(dists[w_mask]), mjd.contact.dist[: mjd.ncon].min(), atol=1e-2)

    if nworld == 2:
      self.assertFalse(np.allclose(np.min(dists[worldids == 0]), np.min(dists[worldids == 1])))

  def test_mesh_sdf_reversed_geom_order(self):
    """Tests Mesh-SDF collision when the SDF geom precedes the mesh geom."""
    _, _, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <asset>
          <mesh name="cube_sdf" vertex="{_CUBE_VERTS}"/>
          <mesh name="cube_mesh" vertex="{_CUBE_VERTS}"/>
        </asset>
        <worldbody>
          <body>
            <geom name="g_sdf" type="sdf" mesh="cube_sdf"/>
          </body>
          <body pos="0 0 1.95">
            <freejoint/>
            <geom name="g_mesh" type="mesh" mesh="cube_mesh"/>
          </body>
        </worldbody>
        <contact>
          <pair geom1="g_sdf" geom2="g_mesh"/>
        </contact>
      </mujoco>
      """
    )

    nacon, dist, _, normal, _ = _run_collision(m, d)
    self.assertGreater(nacon, 0)
    deepest = np.argmin(dist)
    np.testing.assert_array_equal(d.contact.geom.numpy()[deepest], [1, 0])
    np.testing.assert_allclose(dist[deepest], -0.05, atol=1e-3)
    np.testing.assert_allclose(normal[deepest], [0.0, 0.0, -1.0], atol=1e-3)


if __name__ == "__main__":
  absltest.main()
