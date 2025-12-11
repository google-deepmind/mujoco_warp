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

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

from mujoco_warp import Data
from mujoco_warp import GeomType
from mujoco_warp import Model
from mujoco_warp import test_data

from .collision_gjk import ccd
from .collision_gjk import multicontact
from .collision_primitive import Geom
from .types import MJ_MAX_EPAFACES
from .types import MJ_MAX_EPAHORIZON
from .warp_util import nested_kernel


def _geom_dist(
  m: Model,
  d: Data,
  gid1: int,
  gid2: int,
  multiccd=False,
  margin=0.0,
  pos1: wp.vec3 | None = None,
  pos2: wp.vec3 | None = None,
  quat1: wp.quat | None = None,
  quat2: wp.quat | None = None,
):
  # we run multiccd on static scenes so these need to be initialized
  nmaxpolygon = 10 if multiccd else 0
  nmaxmeshdeg = 10 if multiccd else 0
  epa_vert = wp.empty(5 + m.opt.ccd_iterations, dtype=wp.vec3)
  epa_vert1 = wp.empty(5 + m.opt.ccd_iterations, dtype=wp.vec3)
  epa_vert2 = wp.empty(5 + m.opt.ccd_iterations, dtype=wp.vec3)
  epa_vert_index1 = wp.empty(5 + m.opt.ccd_iterations, dtype=int)
  epa_vert_index2 = wp.empty(5 + m.opt.ccd_iterations, dtype=int)
  epa_face = wp.empty(6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations, dtype=wp.vec3i)
  epa_pr = wp.empty(6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations, dtype=wp.vec3)
  epa_norm2 = wp.empty(6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations, dtype=float)
  epa_index = wp.empty(6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations, dtype=int)
  epa_map = wp.empty(6 + MJ_MAX_EPAFACES * m.opt.ccd_iterations, dtype=int)
  epa_horizon = wp.empty(2 * MJ_MAX_EPAHORIZON, dtype=int)
  multiccd_polygon = wp.empty(2 * nmaxpolygon, dtype=wp.vec3)
  multiccd_clipped = wp.empty(2 * nmaxpolygon, dtype=wp.vec3)
  multiccd_pnormal = wp.empty(nmaxpolygon, dtype=wp.vec3)
  multiccd_pdist = wp.empty(nmaxpolygon, dtype=float)
  multiccd_idx1 = wp.empty(nmaxmeshdeg, dtype=int)
  multiccd_idx2 = wp.empty(nmaxmeshdeg, dtype=int)
  multiccd_n1 = wp.empty(nmaxmeshdeg, dtype=wp.vec3)
  multiccd_n2 = wp.empty(nmaxmeshdeg, dtype=wp.vec3)
  multiccd_endvert = wp.empty(nmaxmeshdeg, dtype=wp.vec3)
  multiccd_face1 = wp.empty(nmaxpolygon, dtype=wp.vec3)
  multiccd_face2 = wp.empty(nmaxpolygon, dtype=wp.vec3)

  @nested_kernel(module="unique", enable_backward=False)
  def _ccd_kernel(
    # Model:
    geom_type: wp.array(dtype=int),
    geom_dataid: wp.array(dtype=int),
    geom_size: wp.array2d(dtype=wp.vec3),
    mesh_vertadr: wp.array(dtype=int),
    mesh_vertnum: wp.array(dtype=int),
    mesh_vert: wp.array(dtype=wp.vec3),
    mesh_polynum: wp.array(dtype=int),
    mesh_polyadr: wp.array(dtype=int),
    mesh_polynormal: wp.array(dtype=wp.vec3),
    mesh_polyvertadr: wp.array(dtype=int),
    mesh_polyvertnum: wp.array(dtype=int),
    mesh_polyvert: wp.array(dtype=int),
    mesh_polymapadr: wp.array(dtype=int),
    mesh_polymapnum: wp.array(dtype=int),
    mesh_polymap: wp.array(dtype=int),
    # Data in:
    geom_xpos_in: wp.array2d(dtype=wp.vec3),
    geom_xquat_in: wp.array2d(dtype=wp.quat),
    # In:
    gid1: int,
    gid2: int,
    iterations: int,
    tolerance: wp.array(dtype=float),
    vert: wp.array(dtype=wp.vec3),
    vert1: wp.array(dtype=wp.vec3),
    vert2: wp.array(dtype=wp.vec3),
    vert_index1: wp.array(dtype=int),
    vert_index2: wp.array(dtype=int),
    face: wp.array(dtype=wp.vec3i),
    face_pr: wp.array(dtype=wp.vec3),
    face_norm2: wp.array(dtype=float),
    face_index: wp.array(dtype=int),
    face_map: wp.array(dtype=int),
    horizon: wp.array(dtype=int),
    polygon: wp.array(dtype=wp.vec3),
    clipped: wp.array(dtype=wp.vec3),
    pnormal: wp.array(dtype=wp.vec3),
    pdist: wp.array(dtype=float),
    idx1: wp.array(dtype=int),
    idx2: wp.array(dtype=int),
    n1: wp.array(dtype=wp.vec3),
    n2: wp.array(dtype=wp.vec3),
    endvert: wp.array(dtype=wp.vec3),
    face1: wp.array(dtype=wp.vec3),
    face2: wp.array(dtype=wp.vec3),
    # Out:
    dist_out: wp.array(dtype=float),
    ncon_out: wp.array(dtype=int),
    pos_out: wp.array(dtype=wp.vec3),
  ):
    geom1 = Geom()
    geom1.index = -1
    geomtype1 = geom_type[gid1]
    if wp.static(pos1 == None):
      geom1.pos = geom_xpos_in[0, gid1]
    else:
      geom1.pos = pos1
    if wp.static(quat1 == None):
      geom1.rot = geom_xquat_in[0, gid1]
    else:
      geom1.rot = quat1
    geom1.size = geom_size[0, gid1]
    geom1.margin = margin
    geom1.graphadr = -1
    geom1.mesh_polyadr = -1

    if geom_dataid[gid1] >= 0 and geom_type[gid1] == GeomType.MESH:
      dataid = geom_dataid[gid1]
      geom1.vertadr = mesh_vertadr[dataid]
      geom1.vertnum = mesh_vertnum[dataid]
      geom1.mesh_polynum = mesh_polynum[dataid]
      geom1.mesh_polyadr = mesh_polyadr[dataid]
      geom1.vert = mesh_vert
      geom1.mesh_polynormal = mesh_polynormal
      geom1.mesh_polyvertadr = mesh_polyvertadr
      geom1.mesh_polyvertnum = mesh_polyvertnum
      geom1.mesh_polyvert = mesh_polyvert
      geom1.mesh_polymapadr = mesh_polymapadr
      geom1.mesh_polymapnum = mesh_polymapnum
      geom1.mesh_polymap = mesh_polymap

    geom2 = Geom()
    geom2.index = -1
    geomtype2 = geom_type[gid2]
    if wp.static(pos2 == None):
      geom2.pos = geom_xpos_in[0, gid2]
    else:
      geom2.pos = pos2
    if wp.static(quat2 == None):
      geom2.rot = geom_xquat_in[0, gid2]
    else:
      geom2.rot = quat2
    geom2.size = geom_size[0, gid2]
    geom2.margin = margin
    geom2.graphadr = -1
    geom2.mesh_polyadr = -1

    if geom_dataid[gid2] >= 0 and geom_type[gid2] == GeomType.MESH:
      dataid = geom_dataid[gid2]
      geom2.vertadr = mesh_vertadr[dataid]
      geom2.vertnum = mesh_vertnum[dataid]
      geom2.mesh_polynum = mesh_polynum[dataid]
      geom2.mesh_polyadr = mesh_polyadr[dataid]
      geom2.vert = mesh_vert
      geom2.mesh_polynormal = mesh_polynormal
      geom2.mesh_polyvertadr = mesh_polyvertadr
      geom2.mesh_polyvertnum = mesh_polyvertnum
      geom2.mesh_polyvert = mesh_polyvert
      geom2.mesh_polymapadr = mesh_polymapadr
      geom2.mesh_polymapnum = mesh_polymapnum
      geom2.mesh_polymap = mesh_polymap

    (
      dist,
      ncon,
      x1,
      x2,
      idx,
    ) = ccd(
      tolerance[0],
      1.0e30,
      iterations,
      iterations,
      geom1,
      geom2,
      geomtype1,
      geomtype2,
      geom1.pos,
      geom2.pos,
      vert,
      vert1,
      vert2,
      vert_index1,
      vert_index2,
      face,
      face_pr,
      face_norm2,
      face_index,
      face_map,
      horizon,
    )

    if wp.static(multiccd):
      ncon, _, _ = multicontact(
        polygon,
        clipped,
        pnormal,
        pdist,
        idx1,
        idx2,
        n1,
        n2,
        endvert,
        face1,
        face2,
        vert1,
        vert2,
        vert_index1,
        vert_index2,
        face[idx],
        x1,
        x2,
        geom1,
        geom2,
        geomtype1,
        geomtype2,
      )

    dist_out[0] = dist
    ncon_out[0] = ncon
    pos_out[0] = x1
    pos_out[1] = x2

  dist_out = wp.array(shape=(1,), dtype=float)
  ncon_out = wp.array(shape=(1,), dtype=int)
  pos_out = wp.array(shape=(2,), dtype=wp.vec3)
  wp.launch(
    _ccd_kernel,
    dim=1,
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      m.mesh_vertadr,
      m.mesh_vertnum,
      m.mesh_vert,
      m.mesh_polynum,
      m.mesh_polyadr,
      m.mesh_polynormal,
      m.mesh_polyvertadr,
      m.mesh_polyvertnum,
      m.mesh_polyvert,
      m.mesh_polymapadr,
      m.mesh_polymapnum,
      m.mesh_polymap,
      d.geom_xpos,
      d.geom_xquat,
      gid1,
      gid2,
      m.opt.ccd_iterations,
      m.opt.ccd_tolerance,
      epa_vert,
      epa_vert1,
      epa_vert2,
      epa_vert_index1,
      epa_vert_index2,
      epa_face,
      epa_pr,
      epa_norm2,
      epa_index,
      epa_map,
      epa_horizon,
      multiccd_polygon,
      multiccd_clipped,
      multiccd_pnormal,
      multiccd_pdist,
      multiccd_idx1,
      multiccd_idx2,
      multiccd_n1,
      multiccd_n2,
      multiccd_endvert,
      multiccd_face1,
      multiccd_face2,
    ],
    outputs=[
      dist_out,
      ncon_out,
      pos_out,
    ],
  )
  return dist_out.numpy()[0], ncon_out.numpy()[0], pos_out.numpy()[0], pos_out.numpy()[1]


class GJKTest(absltest.TestCase):
  """Tests for GJK/EPA."""

  def test_spheres_distance(self):
    """Test distance between two spheres."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom name="geom1" type="sphere" pos="-1.5 0 0" size="1"/>
          <geom name="geom2" type="sphere" pos="1.5 0 0" size="1"/>
        </worldbody>
       </mujoco>
       """
    )

    dist, _, x1, x2 = _geom_dist(m, d, 0, 1)
    self.assertEqual(1.0, dist)
    self.assertEqual(-0.5, x1[0])
    self.assertEqual(0.5, x2[0])

  def test_spheres_touching(self):
    """Test two touching spheres have zero distance."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="sphere" pos="-1 0 0" size="1"/>
          <geom type="sphere" pos="1 0 0" size="1"/>
        </worldbody>
       </mujoco>
       """
    )

    dist, _, _, _ = _geom_dist(m, d, 0, 1)
    self.assertEqual(0.0, dist)

  def test_box_mesh_distance(self):
    """Test distance between a mesh and box."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco model="MuJoCo Model">
        <asset>
          <mesh name="smallbox" scale="0.1 0.1 0.1"
                vertex="-1 -1 -1
                         1 -1 -1
                         1  1 -1
                         1  1  1
                         1 -1  1
                        -1  1 -1
                        -1  1  1
                        -1 -1  1"/>
         </asset>
         <worldbody>
           <geom pos="0 0 .90" type="box" size="0.5 0.5 0.1"/>
           <geom pos="0 0 1.2" type="mesh" mesh="smallbox"/>
          </worldbody>
       </mujoco>
       """
    )

    dist, _, _, _ = _geom_dist(m, d, 0, 1)
    self.assertAlmostEqual(0.1, dist)

  def test_sphere_sphere_contact(self):
    """Test penetration depth between two spheres."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="sphere" pos="-1 0 0" size="3"/>
          <geom type="sphere" pos=" 3 0 0" size="3"/>
        </worldbody>
      </mujoco>
      """
    )

    dist, _, _, _ = _geom_dist(m, d, 0, 1, 0)
    self.assertAlmostEqual(-2, dist)

  def test_box_box_contact(self):
    """Test penetration between two boxes."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="box" pos="-1 0 0" size="2.5 2.5 2.5"/>
          <geom type="box" pos="1.5 0 0" size="1 1 1"/>
        </worldbody>
      </mujoco>
      """
    )
    dist, _, x1, x2 = _geom_dist(m, d, 0, 1)
    diff = x1 - x2
    normal = diff / np.linalg.norm(diff)

    self.assertAlmostEqual(-1, dist)
    self.assertAlmostEqual(normal[0], 1)
    self.assertAlmostEqual(normal[1], 0)
    self.assertAlmostEqual(normal[2], 0)

  def test_mesh_mesh_contact(self):
    """Test penetration between two meshes."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <asset>
        <mesh name="box" scale=".5 .5 .1"
              vertex="-1 -1 -1
                       1 -1 -1
                       1  1 -1
                       1  1  1
                       1 -1  1
                      -1  1 -1
                      -1  1  1
                      -1 -1  1"/>
        <mesh name="smallbox" scale=".1 .1 .1"
              vertex="-1 -1 -1
                       1 -1 -1
                       1  1 -1
                       1  1  1
                       1 -1  1
                      -1  1 -1
                      -1  1  1
                      -1 -1  1"/>
      </asset>

      <worldbody>
        <geom pos="0 0 .09" type="mesh" mesh="smallbox"/>
        <geom pos="0 0 -.1" type="mesh" mesh="box"/>
      </worldbody>
    </mujoco>
    """
    )
    dist, _, _, _ = _geom_dist(m, d, 0, 1)
    self.assertAlmostEqual(-0.01, dist)

  def test_cylinder_cylinder_contact(self):
    """Test penetration between two cylinder."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom pos="0 0 0" type="cylinder" size="1 .5"/>
          <geom pos="1.999 0 0" type="cylinder" size="1 .5"/>
        </worldbody>
      </mujoco>
    """
    )

    dist, _, _, _ = _geom_dist(m, d, 0, 1)
    self.assertAlmostEqual(-0.001, dist)

  def test_box_edge(self):
    """Test box edge."""
    _, _, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <geom pos="0 0 2" type="box" name="box2" size="1 1 1"/>
        <geom pos="0 0 4.4" euler="0 90 40" type="box" name="box3" size="1 1 1"/>
      </worldbody>
    </mujoco>"""
    )
    _, ncon, _, _ = _geom_dist(m, d, 0, 1, multiccd=True)
    self.assertEqual(ncon, 2)

  def test_box_box_ccd(self):
    """Test box box."""
    _, _, m, d = test_data.fixture(
      xml="""
       <mujoco>
         <worldbody>
           <geom name="geom1" type="box" pos="0 0 1.9" size="1 1 1"/>
           <geom name="geom2" type="box" pos="0 0 0" size="10 10 1"/>
         </worldbody>
       </mujoco>
       """
    )
    _, ncon, _, _ = _geom_dist(m, d, 0, 1, multiccd=True)
    self.assertEqual(ncon, 4)

  def test_mesh_mesh_ccd(self):
    """Test mesh-mesh multiccd."""
    _, _, m, d = test_data.fixture(
      xml="""
       <mujoco>
         <asset>
           <mesh name="smallbox"
                 vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 -1 1 -1 -1 1 1 -1 -1 1"/>
         </asset>
         <worldbody>
           <geom pos="0 0 2" type="mesh" name="box1" mesh="smallbox"/>
          <geom pos="0 1 3.99" euler="0 0 40" type="mesh" name="box2" mesh="smallbox"/>
         </worldbody>
       </mujoco>
       """
    )

    _, ncon, _, _ = _geom_dist(m, d, 0, 1, multiccd=True)
    self.assertEqual(ncon, 4)

  def test_box_box_ccd2(self):
    """Test box-box multiccd 2."""
    _, _, m, d = test_data.fixture(
      xml="""
       <mujoco>
         <worldbody>
           <geom size="1 1 1" pos="0 0 2" type="box"/>
          <geom size="1 1 1" pos="0 1 3.99" euler="0 0 40" type="box"/>
         </worldbody>
       </mujoco>
       """
    )

    _, ncon, _, _ = _geom_dist(m, d, 0, 1, multiccd=True)
    self.assertEqual(ncon, 4)

  def test_sphere_mesh_margin(self):
    """Test sphere-mesh margin."""
    _, _, m, d = test_data.fixture(
      xml="""
       <mujoco>
         <asset>
           <mesh name="box" scale=".2 .2 .2"
                 vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 -1 1 -1 -1 1 1 -1 -1 1"/>
         </asset>
         <worldbody>
           <geom type="sphere" pos="0 0 .349" size=".1"/>
           <geom type="mesh" mesh="box"/>
         </worldbody>
       </mujoco>
       """
    )

    dist, _, _, _ = _geom_dist(m, d, 0, 1, multiccd=False, margin=0.05)
    self.assertAlmostEqual(dist, -0.001)

  def test_cylinder_box(self):
    """Test cylinder box collision."""
    _, _, m, d = test_data.fixture(
      xml="""
       <mujoco>
         <worldbody>
           <geom type="box" size="1 1 0.1"/>
           <geom type="cylinder" size=".1 .2 .3"/>
         </worldbody>
       </mujoco>
       """,
      overrides=["opt.ccd_iterations=50"],
    )

    pos = wp.vec3(0.00015228791744448245, -0.00074981129728257656, 0.29839199781417846680)
    rot_mat = np.array(
      [
        0.99996972084045410156,
        0.00776371126994490623,
        -0.00043433305108919740,
        -0.00776385562494397163,
        0.99996984004974365234,
        -0.00033095158869400620,
        0.00043175052269361913,
        0.00033431366318836808,
        0.99999988079071044922,
      ]
    )
    rot_quat = np.zeros(4)
    mujoco.mju_mat2Quat(rot_quat, rot_mat)
    rot = wp.quat(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])

    dist, _, _, _ = _geom_dist(m, d, 0, 1, pos2=pos, quat2=rot)
    self.assertAlmostEqual(dist, -0.0016624178339902445)

  def test_box_box_float(self):
    """Test box-box under float32."""
    _, _, m, d = test_data.fixture(
      xml="""
       <mujoco>
         <worldbody>
          <geom size=".025 .025 .025" type="box"/>
          <geom size=".025 .025 .025" type="box"/>
         </worldbody>
       </mujoco>
       """
    )

    pos1 = wp.vec3(-0.17624500393867492676, -0.12375499308109283447, 0.12499777972698211670)
    rot1_mat = np.array(
      [
        1.00000000000000000000,
        -0.00000000184385418045,
        -0.00000025833372774287,
        0.00000000184391857339,
        1.00000000000000000000,
        0.00000024928382913458,
        0.00000025833372774287,
        -0.00000024928382913458,
        1.0000000000000000000,
      ]
    )
    rot1_quat = np.zeros(4)
    mujoco.mju_mat2Quat(rot1_quat, rot1_mat)
    rot1 = wp.quat(rot1_quat[0], rot1_quat[1], rot1_quat[2], rot1_quat[3])

    pos2 = wp.vec3(-0.17624500393867492676, -0.12375499308109283447, 0.17499557137489318848)
    rot2_mat = np.array(
      [
        1.00000000000000000000,
        -0.00000000184292525685,
        0.00000012980596864054,
        0.00000000184294413064,
        1.00000000000000000000,
        -0.00000014602545661546,
        -0.00000012980596864054,
        0.00000014602545661546,
        1.00000000000000000000,
      ]
    )
    rot2_quat = np.zeros(4)
    mujoco.mju_mat2Quat(rot2_quat, rot2_mat)
    rot2 = wp.quat(rot2_quat[0], rot2_quat[1], rot2_quat[2], rot2_quat[3])

    dist, ncon, _, _ = _geom_dist(m, d, 0, 1, multiccd=False, pos1=pos1, quat1=rot1, pos2=pos2, quat2=rot2)
    self.assertEqual(ncon, 1)
    self.assertLess(dist, 0.0001)  # real depth is ~ 2E-6


if __name__ == "__main__":
  wp.init()
  absltest.main()
