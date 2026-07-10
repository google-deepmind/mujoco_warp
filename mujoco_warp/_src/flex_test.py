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

"""Tests for flex features."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import ConeType
from mujoco_warp import test_data
from mujoco_warp._src import bvh

_TOLERANCE = 5e-4


def _sparse2dense(rownnz, rowadr, colind, sparse_val, shape):
  dense = np.zeros(shape, dtype=float)
  if len(sparse_val) > 0:
    mujoco.mju_sparse2dense(dense, sparse_val.ravel(), rownnz, rowadr, colind)
  return dense


def _assert_efc_eq(mjm, m, d, mjd, nefc, name, nv, nworld, tol=_TOLERANCE):
  """Assert equality of efc fields after sorting both sides."""
  # Get the ordering indices based on efc fields for MuJoCo (single world)
  mjd_efc_type = mjd.efc_type[:nefc]
  mjd_efc_id = mjd.efc_id[:nefc]
  mjd_efc_pos = mjd.efc_pos[:nefc]
  mjd_efc_vel = mjd.efc_vel[:nefc]
  mjd_efc_aref = mjd.efc_aref[:nefc]
  mjd_efc_d = mjd.efc_D[:nefc]

  mjd_sort_indices = np.lexsort((mjd_efc_pos, mjd_efc_vel, mjd_efc_aref, mjd_efc_d, mjd_efc_id, mjd_efc_type))

  if mujoco.mj_isSparse(mjm):
    mj_efc_J = np.zeros((mjd.nefc, mjm.nv))
    mujoco.mju_sparse2dense(mj_efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
  else:
    mj_efc_J = mjd.efc_J.reshape((mjd.nefc, mjm.nv))

  if nv > 0:
    mjd_sorted_J = mj_efc_J[mjd_sort_indices].reshape(-1)
  else:
    mjd_sorted_J = mj_efc_J

  mjd_sorted_D = mjd.efc_D[mjd_sort_indices]
  mjd_sorted_vel = mjd.efc_vel[mjd_sort_indices]
  mjd_sorted_aref = mjd.efc_aref[mjd_sort_indices]
  mjd_sorted_pos = mjd.efc_pos[mjd_sort_indices]
  mjd_sorted_margin = mjd.efc_margin[mjd_sort_indices]
  mjd_sorted_type = mjd.efc_type[mjd_sort_indices]

  for w in range(nworld):
    # Get the ordering indices based on efc fields for MJWarp for this world
    efc_type = d.efc.type.numpy()[w, :nefc]
    efc_id = d.efc.id.numpy()[w, :nefc]
    efc_pos = d.efc.pos.numpy()[w, :nefc]
    efc_vel = d.efc.vel.numpy()[w, :nefc]
    efc_aref = d.efc.aref.numpy()[w, :nefc]
    efc_d = d.efc.D.numpy()[w, :nefc]

    d_sort_indices = np.lexsort((efc_pos, efc_vel, efc_aref, efc_d, efc_id, efc_type))

    # convert sparse to dense if necessary
    if m.is_sparse:
      efc_J = np.zeros((nefc, nv))
      mujoco.mju_sparse2dense(
        efc_J,
        d.efc.J.numpy()[w, 0],
        d.efc.J_rownnz.numpy()[w, :nefc],
        d.efc.J_rowadr.numpy()[w, :nefc],
        d.efc.J_colind.numpy()[w, 0],
      )
    else:
      efc_J = d.efc.J.numpy()[w, :nefc, :nv]

    # Sort MJWarp efc fields
    d_sorted_J = efc_J[d_sort_indices, :nv].reshape(-1)

    # Compare sorted data
    np.testing.assert_allclose(d_sorted_J, mjd_sorted_J, atol=tol, rtol=tol, err_msg=f"mismatch: {name}_J (world {w})")

    d_sorted = d.efc.D.numpy()[w, d_sort_indices]
    np.testing.assert_allclose(d_sorted, mjd_sorted_D, atol=tol, rtol=tol, err_msg=f"mismatch: {name}_D (world {w})")

    d_sorted = d.efc.vel.numpy()[w, d_sort_indices]
    np.testing.assert_allclose(d_sorted, mjd_sorted_vel, atol=tol, rtol=tol, err_msg=f"mismatch: {name}_vel (world {w})")

    d_sorted = d.efc.aref.numpy()[w, d_sort_indices]
    np.testing.assert_allclose(
      d_sorted, mjd_sorted_aref, atol=tol * 2, rtol=tol * 2, err_msg=f"mismatch: {name}_aref (world {w})"
    )

    d_sorted = d.efc.pos.numpy()[w, d_sort_indices]
    np.testing.assert_allclose(d_sorted, mjd_sorted_pos, atol=tol, rtol=tol, err_msg=f"mismatch: {name}_pos (world {w})")

    d_sorted = d.efc.margin.numpy()[w, d_sort_indices]
    np.testing.assert_allclose(d_sorted, mjd_sorted_margin, atol=tol, rtol=tol, err_msg=f"mismatch: {name}_margin (world {w})")

    d_sorted = d.efc.type.numpy()[w, d_sort_indices]
    np.testing.assert_allclose(d_sorted, mjd_sorted_type, err_msg=f"mismatch: {name}_type (world {w})")


class FlexKinematicsTest(parameterized.TestCase):
  """Tests for flex kinematics parity with MuJoCo."""

  @parameterized.product(
    xml_and_atol=[
      # dim=1 rope
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="rope" type="grid" count="5 1 1" spacing="0.1 0.1 0.1" dim="1" mass="1">
              <edge damping="0.1"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        1e-5,
      ),
      # dim=2 cloth
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" dim="2" mass="1">
              <edge damping="0.1"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        1e-5,
      ),
      # dim=2 cloth (dof=2d)
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" dim="2" mass="1" dof="2d">
              <edge damping="0.1"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        1e-5,
      ),
      # dim=3 softbody (full)
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="softbody" type="grid" count="3 3 3" spacing="0.1 0.1 0.1" dim="3" mass="1" dof="full">
              <elasticity young="1e4" damping="0.1"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        1e-5,
      ),
      # dim=3 softbody (box type, dof=full)
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="softbody" type="box" count="3 3 3" spacing="0.1 0.1 0.1" dim="3" mass="1" dof="full">
              <elasticity young="1e4" damping="0.1"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        1e-5,
      ),
      # dim=3 softbody (trilinear)
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="softbody" type="grid" count="3 3 3" spacing="0.1 0.1 0.1" dim="3" mass="1" dof="trilinear">
              <contact selfcollide="none"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        1e-5,
      ),
    ],
    nworld=[1, 2],
  )
  def test_kinematics_parity(self, xml_and_atol, nworld):
    xml, atol = xml_and_atol
    mjm, mjd, m, d = test_data.fixture(xml=xml, qpos_noise=0.1, qvel_noise=0.1, nworld=nworld)
    d.flexvert_xpos.fill_(wp.inf)
    if mjm.nflexedge > 0:
      d.flexedge_length.fill_(wp.inf)
      d.flexedge_velocity.fill_(wp.inf)
      d.flexedge_J.fill_(wp.inf)
    if mjm.nflexnode > 0:
      d.flexnode_xpos.fill_(wp.inf)
    mjw.kinematics(m, d)
    mjw.com_pos(m, d)
    mjw.flex(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_comPos(mjm, mjd)
    mujoco.mj_fwdVelocity(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)

    nflexvert = mjm.nflexvert
    if nflexvert > 0:
      for w in range(nworld):
        np.testing.assert_allclose(
          d.flexvert_xpos.numpy()[w, :nflexvert],
          mjd.flexvert_xpos[:nflexvert],
          atol=atol,
          err_msg=f"flexvert_xpos mismatch for world {w}",
        )

    nflexedge = mjm.nflexedge
    if nflexedge > 0 and mjm.nflexnode == 0:
      for w in range(nworld):
        np.testing.assert_allclose(
          d.flexedge_length.numpy()[w, :nflexedge],
          mjd.flexedge_length[:nflexedge],
          atol=atol,
          err_msg=f"flexedge_length mismatch for world {w}",
        )
        np.testing.assert_allclose(
          d.flexedge_velocity.numpy()[w, :nflexedge],
          mjd.flexedge_velocity[:nflexedge],
          atol=atol,
          err_msg=f"flexedge_velocity mismatch for world {w}",
        )

        # Compare Jacobians
        mj_flexedge_J = _sparse2dense(
          mjm.flexedge_J_rownnz,
          mjm.flexedge_J_rowadr,
          mjm.flexedge_J_colind,
          mjd.flexedge_J,
          (mjm.nflexedge, mjm.nv),
        )
        flexedge_J = _sparse2dense(
          m.flexedge_J_rownnz.numpy(),
          m.flexedge_J_rowadr.numpy(),
          m.flexedge_J_colind.numpy(),
          d.flexedge_J.numpy()[w],
          (mjm.nflexedge, mjm.nv),
        )
        np.testing.assert_allclose(
          flexedge_J,
          mj_flexedge_J,
          atol=atol,
          err_msg=f"flexedge_J mismatch for world {w}",
        )

    # Test flexnode_xpos for interpolated flex
    nflexnode = mjm.nflexnode
    if nflexnode > 0:
      for w in range(nworld):
        warp_xpos = d.flexnode_xpos.numpy()[w, :nflexnode]
        nodeadr = mjm.flex_nodeadr[0]
        nodenum = mjm.flex_nodenum[0]
        for n in range(nodenum):
          bodyid = mjm.flex_nodebodyid[nodeadr + n]
          body_xpos = mjd.xpos[bodyid]
          body_xmat = mjd.xmat[bodyid].reshape(3, 3)
          node_local = mjm.flex_node[nodeadr + n]
          expected = body_xpos + body_xmat @ node_local
          np.testing.assert_allclose(
            warp_xpos[n],
            expected,
            atol=atol,
            err_msg=f"flexnode_xpos mismatch for node {n} world {w}",
          )

  @parameterized.parameters(1, 2)
  def test_flex_floppy_kinematics(self, nworld):
    """Test flex kinematics for floppy.xml."""
    mjm, mjd, m, d = test_data.fixture("flex/floppy.xml", nworld=nworld)
    self.assertTrue(m.is_sparse)

    d.flexvert_xpos.fill_(wp.inf)
    d.flexedge_length.fill_(wp.inf)
    d.flexedge_velocity.fill_(wp.inf)
    d.flexedge_J.fill_(wp.inf)

    mjw.kinematics(m, d)
    mjw.com_pos(m, d)
    mjw.flex(m, d)
    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_comPos(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)

    rownnz = mjm.flexedge_J_rownnz
    rowadr = mjm.flexedge_J_rowadr
    colind = mjm.flexedge_J_colind.reshape(-1)

    mj_flexedge_J = np.zeros((mjm.nflexedge, mjm.nv), dtype=float)
    mujoco.mju_sparse2dense(mj_flexedge_J, mjd.flexedge_J.ravel(), rownnz, rowadr, colind)

    for w in range(nworld):
      np.testing.assert_allclose(
        d.flexvert_xpos.numpy()[w], mjd.flexvert_xpos, atol=_TOLERANCE, err_msg=f"flexvert_xpos mismatch (world {w})"
      )
      np.testing.assert_allclose(
        d.flexedge_length.numpy()[w], mjd.flexedge_length, atol=_TOLERANCE, err_msg=f"flexedge_length mismatch (world {w})"
      )
      np.testing.assert_allclose(
        d.flexedge_velocity.numpy()[w],
        mjd.flexedge_velocity,
        atol=_TOLERANCE,
        err_msg=f"flexedge_velocity mismatch (world {w})",
      )

      flexedge_J = np.zeros((mjm.nflexedge, mjm.nv))
      mujoco.mju_sparse2dense(
        flexedge_J,
        d.flexedge_J.numpy()[w].reshape(-1),
        m.flexedge_J_rownnz.numpy(),
        m.flexedge_J_rowadr.numpy(),
        m.flexedge_J_colind.numpy(),
      )
      np.testing.assert_allclose(flexedge_J, mj_flexedge_J, atol=_TOLERANCE, err_msg=f"flexedge_J mismatch (world {w})")

  @parameterized.parameters(1, 2)
  def test_flex_1d_pinned(self, nworld):
    """Tests that 1D flex with pinned vertex computes correct Jacobian and velocity."""
    xml = """
    <mujoco>
      <option gravity="0 0 -10"/>
      <worldbody>
        <body name="rope" pos="0.5 0.5 1.0">
          <geom type="sphere" size="0.02" mass="0.01"/>
          <flexcomp name="line" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                    radius="0.01" dim="1" mass="1">
            <contact contype="0" conaffinity="0"/>
            <edge equality="true" damping="0.01"/>
            <pin id="0"/>
          </flexcomp>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    self.assertEqual(m.nflex, 1)
    self.assertEqual(m.nflexvert, 5)

    d.flexvert_xpos.fill_(wp.inf)
    d.flexedge_length.fill_(wp.inf)
    d.flexedge_velocity.fill_(wp.inf)
    d.flexedge_J.fill_(wp.inf)

    mjw.kinematics(m, d)
    mjw.com_pos(m, d)
    mjw.flex(m, d)
    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_comPos(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)

    # Compare dense Jacobians
    rownnz = mjm.flexedge_J_rownnz
    rowadr = mjm.flexedge_J_rowadr
    colind = mjm.flexedge_J_colind.reshape(-1)

    mj_flexedge_J = np.zeros((mjm.nflexedge, mjm.nv), dtype=float)
    mujoco.mju_sparse2dense(mj_flexedge_J, mjd.flexedge_J.ravel(), rownnz, rowadr, colind)

    for w in range(nworld):
      np.testing.assert_allclose(
        d.flexvert_xpos.numpy()[w], mjd.flexvert_xpos, atol=_TOLERANCE, err_msg=f"flexvert_xpos mismatch (world {w})"
      )
      np.testing.assert_allclose(
        d.flexedge_length.numpy()[w], mjd.flexedge_length, atol=_TOLERANCE, err_msg=f"flexedge_length mismatch (world {w})"
      )
      np.testing.assert_allclose(
        d.flexedge_velocity.numpy()[w],
        mjd.flexedge_velocity,
        atol=_TOLERANCE,
        err_msg=f"flexedge_velocity mismatch (world {w})",
      )

      flexedge_J = np.zeros((mjm.nflexedge, mjm.nv))
      mujoco.mju_sparse2dense(
        flexedge_J,
        d.flexedge_J.numpy()[w].reshape(-1),
        m.flexedge_J_rownnz.numpy(),
        m.flexedge_J_rowadr.numpy(),
        m.flexedge_J_colind.numpy(),
      )
      np.testing.assert_allclose(flexedge_J, mj_flexedge_J, atol=_TOLERANCE, err_msg=f"flexedge_J mismatch (world {w})")


class FlexConstraintTest(parameterized.TestCase):
  """Tests for flex constraints parity with MuJoCo."""

  @parameterized.product(
    xml_and_atol=[
      # 1D Rope with edge equality
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="rope" type="grid" count="5 1 1" spacing="0.1 0.1 0.1" dim="1" mass="1">
              <edge equality="true"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        _TOLERANCE,
      ),
      # 2D Cloth with edge equality
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" dim="2" mass="1">
              <edge equality="true"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        _TOLERANCE,
      ),
      # 3D Trilinear with strain equality
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="softbody" type="grid" count="3 3 3" spacing="0.1 0.1 0.1" dim="3" mass="1" dof="trilinear">
              <contact selfcollide="none"/>
              <edge equality="strain"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        _TOLERANCE,
      ),
    ],
    nworld=[1, 2],
  )
  def test_constraint_parity(self, xml_and_atol, nworld):
    xml, atol = xml_and_atol
    mjm, mjd, m, d = test_data.fixture(xml=xml, qpos_noise=0.05, nworld=nworld)
    d.nefc.fill_(-1)
    d.efc.pos.fill_(wp.inf)
    d.efc.J.fill_(wp.inf)
    mjw.fwd_position(m, d)
    mjw.make_constraint(m, d)

    mujoco.mj_forward(mjm, mjd)

    nefc = mjd.nefc
    for w in range(nworld):
      self.assertEqual(d.nefc.numpy()[w], nefc, f"nefc mismatch for world {w}")

    if nefc == 0:
      return

    # Compare residuals
    for w in range(nworld):
      np.testing.assert_allclose(
        d.efc.pos.numpy()[w, :nefc],
        mjd.efc_pos[:nefc],
        atol=atol,
        err_msg=f"efc_pos mismatch for world {w}",
      )

    # Compare Jacobians
    nv = mjm.nv
    if mujoco.mj_isSparse(mjm):
      mj_efc_J = np.zeros((nefc, nv))
      mujoco.mju_sparse2dense(
        mj_efc_J,
        mjd.efc_J,
        mjd.efc_J_rownnz,
        mjd.efc_J_rowadr,
        mjd.efc_J_colind,
      )
    else:
      mj_efc_J = mjd.efc_J.reshape((nefc, nv))

    for w in range(nworld):
      if m.is_sparse:
        warp_efc_J = np.zeros((nefc, nv))
        mujoco.mju_sparse2dense(
          warp_efc_J,
          d.efc.J.numpy()[w, 0],
          d.efc.J_rownnz.numpy()[w, :nefc],
          d.efc.J_rowadr.numpy()[w, :nefc],
          d.efc.J_colind.numpy()[w, 0],
        )
      else:
        warp_efc_J = d.efc.J.numpy()[w, :nefc, :nv]

      np.testing.assert_allclose(
        warp_efc_J,
        mj_efc_J,
        atol=0.01,
        err_msg=f"efc_J mismatch for world {w}",
      )

  @parameterized.parameters(1, 2)
  def test_flexstrain_rotational_invariance(self, nworld):
    """Test that FLEXSTRAIN residuals are invariant under rigid translation."""
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                  pos="0 0 0.5" name="cube" dim="3" mass="1" radius="0.005"
                  dof="trilinear">
          <contact selfcollide="none"/>
          <edge equality="strain"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    # Get reference residuals
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld)
    mjw.fwd_position(m, d)
    mjw.make_constraint(m, d)

    # Apply uniform translation to all nodes (rigid motion)
    mjd2 = mujoco.MjData(mjm)
    for i in range(0, mjm.nq, 3):
      mjd2.qpos[i] += 0.1  # shift x
    mujoco.mj_forward(mjm, mjd2)

    d2 = mjw.put_data(mjm, mjd2, nworld=nworld)
    mjw.fwd_position(m, d2)
    mjw.make_constraint(m, d2)

    for w in range(nworld):
      nefc = d.nefc.numpy()[w]
      efc_pos_rest = d.efc.pos.numpy()[w, :nefc].copy()
      nefc2 = d2.nefc.numpy()[w]
      efc_pos_shifted = d2.efc.pos.numpy()[w, :nefc2]

      np.testing.assert_allclose(
        efc_pos_shifted,
        efc_pos_rest,
        atol=1e-4,
        err_msg=f"FLEXSTRAIN residuals should be invariant under rigid translation (world {w})",
      )

  @parameterized.parameters(1, 2)
  def test_flexstrain_constraint_rotated(self, nworld):
    """Test FLEXSTRAIN residuals and Jacobians match MuJoCo under large rotation perturbation."""
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                  pos="0 0 0.5" name="cube" dim="3" mass="1" radius="0.005"
                  dof="trilinear">
          <edge equality="strain"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)

    # Apply a rotation perturbation: rotate all node positions around Y axis by 30 degrees
    # (0.5235 radians)
    theta = 0.5235
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    for i in range(0, mjm.nq, 3):
      x = mjd.qpos[i]
      z = mjd.qpos[i + 2]
      mjd.qpos[i] = x * cos_t - z * sin_t
      mjd.qpos[i + 2] = x * sin_t + z * cos_t

    mujoco.mj_forward(mjm, mjd)

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld)

    mjw.fwd_position(m, d)
    mjw.make_constraint(m, d)

    nefc = mjd.nefc
    nv = mjm.nv
    for w in range(nworld):
      self.assertEqual(d.nefc.numpy()[w], nefc)

    # residuals
    for w in range(nworld):
      efc_pos_warp = d.efc.pos.numpy()[w, :nefc]
      efc_pos_mj = mjd.efc_pos[:nefc]
      np.testing.assert_allclose(
        efc_pos_warp,
        efc_pos_mj,
        atol=_TOLERANCE,
        err_msg=f"FLEXSTRAIN residuals don't match MuJoCo under rotation (world {w})",
      )

    # Jacobians
    if mujoco.mj_isSparse(mjm):
      mj_efc_J = np.zeros((nefc, nv))
      mujoco.mju_sparse2dense(mj_efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
    else:
      mj_efc_J = mjd.efc_J.reshape((nefc, nv))

    for w in range(nworld):
      if m.is_sparse:
        warp_efc_J = np.zeros((nefc, nv))
        mujoco.mju_sparse2dense(
          warp_efc_J,
          d.efc.J.numpy()[w, 0],
          d.efc.J_rownnz.numpy()[w, :nefc],
          d.efc.J_rowadr.numpy()[w, :nefc],
          d.efc.J_colind.numpy()[w, 0],
        )
      else:
        warp_efc_J = d.efc.J.numpy()[w, :nefc, :nv]

      np.testing.assert_allclose(
        warp_efc_J,
        mj_efc_J,
        atol=0.01,
        err_msg=f"FLEXSTRAIN Jacobians don't match MuJoCo under rotation (world {w})",
      )

  @parameterized.parameters(1, 2)
  def test_trilinear_contact_qfrc_constraint(self, nworld):
    """Test qfrc_constraint parity for trilinear flex with ground contacts."""
    xml = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <flexcomp type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                  pos="0 0 0.05" name="cube" dim="3" mass="1" radius="0.02"
                  dof="trilinear">
          <edge equality="strain"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)
    mjw.forward(m, d)

    # Verify contacts are generated
    nacon = d.nacon.numpy()[0]
    self.assertEqual(nacon, mjd.ncon * nworld, "Total contacts mismatch")

    contacts_worldid = d.contact.worldid.numpy()[:nacon]
    for w in range(nworld):
      n_world_contacts = np.sum(contacts_worldid == w)
      self.assertEqual(n_world_contacts, mjd.ncon, f"ncon mismatch for world {w}")

    # Verify qfrc_constraint parity
    qfrc_mj = mjd.qfrc_constraint
    for w in range(nworld):
      qfrc_warp = d.qfrc_constraint.numpy()[w]
      np.testing.assert_allclose(
        qfrc_warp,
        qfrc_mj,
        atol=1e-4,
        err_msg=f"qfrc_constraint mismatch for trilinear flex contacts (world {w})",
      )

  @parameterized.product(
    xml=("flex/floppy.xml", "flex/moving_base_strain.xml"),
    cone=(mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtCone.mjCONE_ELLIPTIC),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
    keyframe=[0, 1, 2],
    nworld=[1, 2],
  )
  def test_flex_constraints_parity(self, xml, cone, jacobian, keyframe, nworld):
    """Test constraints parity for flex models from XML files."""
    if xml == "flex/floppy.xml" and jacobian == mujoco.mjtJacobian.mjJAC_DENSE:
      self.skipTest("flex/floppy.xml with dense jacobian not supported")

    mjm, mjd, m, d = test_data.fixture(
      xml,
      keyframe=keyframe,
      overrides={"opt.cone": cone, "opt.jacobian": jacobian},
      nworld=nworld,
    )

    for arr in (d.efc.J, d.efc.D, d.efc.aref, d.efc.pos, d.efc.margin):
      arr.fill_(wp.nan)

    mjw.fwd_position(m, d)

    self.assertEqual(d.nacon.numpy()[0], mjd.ncon * nworld, "nacon mismatch")

    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc", m.nv, nworld)


class FlexPassiveForcesTest(parameterized.TestCase):
  """Tests for flex passive forces parity with MuJoCo."""

  @parameterized.product(
    xml_and_atol=[
      # 3D SVK Elasticity (dof=full)
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="softbody" type="grid" count="3 3 3" spacing="0.1 0.1 0.1" dim="3" mass="1" dof="full">
              <elasticity young="1e4" poisson="0.3"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        _TOLERANCE,
      ),
      # 2D Cloth with bending/stretching (elastic2d=both)
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" dim="2" mass="1">
              <elasticity young="3e3" poisson="0.3" thickness="1e-2" damping="1e-3" elastic2d="both"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        _TOLERANCE,
      ),
      # 2D Cloth with dof=2d (elastic2d=stretch only)
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" dim="2" mass="1" dof="2d">
              <elasticity young="3e3" poisson="0.3" thickness="1e-2" damping="1e-3" elastic2d="stretch"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        _TOLERANCE,
      ),
      # 2D Cloth with stretch only (dof=full)
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" dim="2" mass="1">
              <elasticity young="3e3" poisson="0.3" thickness="1e-2" damping="1e-3" elastic2d="stretch"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        _TOLERANCE,
      ),
      # 2D Cloth with bend only (dof=full)
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" dim="2" mass="1">
              <elasticity young="3e3" poisson="0.3" thickness="1e-2" damping="1e-3" elastic2d="bend"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        _TOLERANCE,
      ),
      # 3D Trilinear Elasticity (dof=trilinear)
      (
        """
        <mujoco>
          <worldbody>
            <flexcomp name="softbody" type="grid" count="3 3 3" spacing="0.1 0.1 0.1" dim="3" mass="1" dof="trilinear">
              <elasticity young="1e4" poisson="0.3" damping="1e-3"/>
              <contact selfcollide="none"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        _TOLERANCE,
      ),
    ],
    nworld=[1, 2],
  )
  def test_passive_forces_parity(self, xml_and_atol, nworld):
    xml, atol = xml_and_atol
    mjm, mjd, m, d = test_data.fixture(xml=xml, qpos_noise=0.05, qvel_noise=0.05, njmax_nnz=20000, nworld=nworld)
    for arr in (d.qfrc_spring, d.qfrc_damper, d.qfrc_passive):
      arr.fill_(wp.inf)

    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.passive(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_passive(mjm, mjd)

    for w in range(nworld):
      np.testing.assert_allclose(
        d.qfrc_passive.numpy()[w],
        mjd.qfrc_passive,
        atol=atol,
        err_msg=f"qfrc_passive mismatch for world {w}",
      )

  @parameterized.product(
    elastic2d_and_atol=[
      ("bend", 5e-3),
      ("stretch", 1e-4),
      ("both", 5e-3),
    ],
    nworld=[1, 2],
  )
  def test_flex_elastic2d_interp(self, elastic2d_and_atol, nworld):
    """Tests 2D interpolated flex passive forces (bending and stretching)."""
    elastic2d, atol = elastic2d_and_atol
    dof = "trilinear"
    xml = f"""
    <mujoco>
      <option gravity="0 0 -9.81">
        <flag contact="disable"/>
      </option>
      <worldbody>
        <flexcomp type="grid" count="4 2 4" spacing=".025 .05 .025" pos="0 0 1"
                  dim="3" cellcount="2 1 2" radius=".001"
                  mass="5" name="softbody" dof="{dof}">
          <elasticity young="1e4" poisson="0.3" damping="1e-3"
                      elastic2d="{elastic2d}" thickness="0.03"/>
          <contact selfcollide="none" internal="false"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(
      xml=xml,
      qpos_noise=0.01,
      nworld=nworld,
    )
    for arr in (d.qfrc_spring, d.qfrc_passive):
      arr.zero_()

    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.passive(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_passive(mjm, mjd)

    for w in range(nworld):
      np.testing.assert_allclose(
        d.qfrc_spring.numpy()[w],
        mjd.qfrc_spring,
        atol=atol,
        err_msg=f"qfrc_spring mismatch for interpolated elastic2d={elastic2d} dof={dof} (world {w})",
      )


class FlexCollisionTest(parameterized.TestCase):
  """Tests for flex collisions."""

  @parameterized.parameters(1, 2)
  def test_plane_cloth_collision(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.004" dim="2" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertGreater(d.nacon.numpy()[0], 0, "Expected contacts in Warp")
    self.assertGreater(mjd.ncon, 0, "Expected contacts in MuJoCo")

    contacts_worldid = d.contact.worldid.numpy()[: d.nacon.numpy()[0]]
    for w in range(nworld):
      self.assertTrue(
        np.any(contacts_worldid == w),
        f"Expected contacts in world {w}",
      )

  @parameterized.parameters(1, 2)
  def test_sphere_cloth_collision(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <geom type="sphere" size="0.1" pos="0 0 0.15"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.05" dim="2" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.collision(m, d)
    self.assertGreater(d.nacon.numpy()[0], 0)

    contacts_worldid = d.contact.worldid.numpy()[: d.nacon.numpy()[0]]
    for w in range(nworld):
      self.assertTrue(
        np.any(contacts_worldid == w),
        f"Expected contacts in world {w}",
      )

  @parameterized.parameters(1, 2)
  def test_self_collision_brute_force(self, nworld):
    # Small cloth, should trigger brute force
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" dim="2" mass="1">
          <contact selfcollide="auto"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)
    xpos = d.flexvert_xpos.numpy()
    for w in range(nworld):
      xpos[w, 2] = xpos[w, 6] + np.array([0.0, 0.0, 0.005])
    d.flexvert_xpos.assign(xpos)

    d.nacon.fill_(-1)
    mjw.collision(m, d)
    self.assertGreater(d.nacon.numpy()[0], 0)

    contacts_worldid = d.contact.worldid.numpy()[: d.nacon.numpy()[0]]
    for w in range(nworld):
      self.assertTrue(
        np.any(contacts_worldid == w),
        f"Expected contacts in world {w}",
      )

  @parameterized.parameters(1, 2)
  def test_self_collision_sap(self, nworld):
    # Larger cloth to trigger SAP (>32 elements)
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp name="cloth" type="grid" count="6 6 1" spacing="0.05 0.05 0.05" dim="2" mass="1">
          <contact selfcollide="auto"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)
    xpos = d.flexvert_xpos.numpy()
    for w in range(nworld):
      xpos[w, 35] = xpos[w, 0] + np.array([0.0, 0.0, 0.01])
    d.flexvert_xpos.assign(xpos)

    d.nacon.fill_(-1)
    mjw.collision(m, d)
    self.assertGreater(d.nacon.numpy()[0], 0)

    contacts_worldid = d.contact.worldid.numpy()[: d.nacon.numpy()[0]]
    for w in range(nworld):
      self.assertTrue(
        np.any(contacts_worldid == w),
        f"Expected contacts in world {w}",
      )

  @parameterized.parameters(1, 2)
  def test_cylinder_cloth_collision(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <geom type="cylinder" size="0.01 0.03" pos="-0.066 -0.033 0.058" euler="90 0 0"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.05" dim="2" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertGreater(d.nacon.numpy()[0], 0, "Expected contacts in Warp")
    self.assertGreater(mjd.ncon, 0, "Expected contacts in MuJoCo")

    contacts_worldid = d.contact.worldid.numpy()[: d.nacon.numpy()[0]]
    for w in range(nworld):
      self.assertTrue(
        np.any(contacts_worldid == w),
        f"Expected contacts in world {w}",
      )

  @parameterized.parameters(1, 2)
  def test_mesh_cloth_collision(self, nworld):
    xml = """
    <mujoco>
      <asset>
        <mesh name="tet" vertex="0 0 0  0.05 0 0  0 0.05 0  0 0 0.05"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="tet" pos="-0.08 -0.05 0.04"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.05" dim="2" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertGreater(d.nacon.numpy()[0], 0, "Expected contacts in Warp")
    self.assertGreater(mjd.ncon, 0, "Expected contacts in MuJoCo")

    contacts_worldid = d.contact.worldid.numpy()[: d.nacon.numpy()[0]]
    for w in range(nworld):
      self.assertTrue(
        np.any(contacts_worldid == w),
        f"Expected contacts in world {w}",
      )

  @parameterized.parameters(1, 2)
  def test_sphere_rope_collision(self, nworld):
    """Test contacts for 1D rope colliding with a sphere (vertex-geom collision in Warp)."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="sphere" size="0.05" pos="0.1 0 0.05"/>
        <flexcomp name="rope" type="grid" count="5 1 1" spacing="0.1 0.1 0.1" pos="0 0 0.08" dim="1" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    # Warp only detects vertex collision (1 contact per world)
    self.assertEqual(d.nacon.numpy()[0], nworld * 1)
    # MuJoCo detects edge collisions (2 contacts)
    self.assertEqual(mjd.ncon, 2)

    for w in range(nworld):
      contacts_worldid = d.contact.worldid.numpy()[: d.nacon.numpy()[0]]
      w_indices = np.where(contacts_worldid == w)[0]
      self.assertEqual(len(w_indices), 1)
      idx = w_indices[0]
      # Verify it is vertex 3 contact
      self.assertEqual(int(d.contact.geom.numpy()[idx, 0]), 0)
      self.assertEqual(int(d.contact.geom.numpy()[idx, 1]), -1)
      self.assertEqual(int(d.contact.flex.numpy()[idx, 0]), -1)
      self.assertEqual(int(d.contact.flex.numpy()[idx, 1]), 0)
      self.assertEqual(int(d.contact.elem.numpy()[idx, 1]), -1)
      self.assertEqual(int(d.contact.vert.numpy()[idx, 1]), 3)

  @parameterized.parameters(1, 2)
  def test_sphere_cloth_contact_generated(self, nworld):
    """Test that contacts are generated between sphere and cloth."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".001"/>
      <size memory="10M"/>

      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>

        <!-- Ground plane -->
        <geom type="plane" size="5 5 .1" pos="0 0 0"/>

        <!-- Sphere positioned just above the cloth -->
        <body pos="0 0 0.12">
          <freejoint/>
          <geom type="sphere" size=".1" mass="1"/>
        </body>

        <!-- Cloth (dim=2 flex) -->
        <flexcomp type="grid" count="4 4 1" spacing=".2 .2 .1" pos="-.3 -.3 0"
                  radius=".02" name="cloth" dim="2" mass=".5">
          <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001"
                   selfcollide="none" conaffinity="1" contype="1"/>
          <edge damping="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    self.assertEqual(mjm.nflex, 1)
    self.assertEqual(mjm.flex_dim[0], 2)

    self.assertEqual(m.nflex, 1)
    self.assertGreater(m.flex_elemnum.numpy()[0], 0)

    mjw.kinematics(m, d)
    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])

    self.assertGreater(nacon, 0, "Expected contacts between sphere and cloth")
    contacts_worldid = d.contact.worldid.numpy()[:nacon]
    for w in range(nworld):
      self.assertTrue(np.any(contacts_worldid == w), f"Expected contacts in world {w}")

  @parameterized.parameters(1, 2)
  def test_sphere_cloth_no_duplicates(self, nworld):
    """Test that duplicate/redundant contacts are filtered out."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" tolerance="1e-6" timestep=".001"/>
        <worldbody>
          <!-- Sphere positioned exactly above a vertex shared by multiple elements -->
          <body pos="0 0 0.1">
            <freejoint/>
            <geom type="sphere" size=".1" mass="1"/>
          </body>
          <!-- Cloth (dim=2 flex) -->
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="-.2 -.2 0"
                    radius=".02" dim="2" mass=".5">
            <contact condim="3" selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    d.nacon.zero_()
    mjw.kinematics(m, d)
    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0)

    pos = d.contact.pos.numpy()[:nacon]
    worldids = d.contact.worldid.numpy()[:nacon]
    for w in range(nworld):
      w_indices = np.where(worldids == w)[0]
      self.assertGreater(len(w_indices), 0, f"Expected contacts in world {w}")
      for idx, i in enumerate(w_indices):
        for j in w_indices[idx + 1 :]:
          dist = np.linalg.norm(pos[i] - pos[j])
          self.assertGreater(dist, 1e-3, f"Duplicate contacts found at positions: {pos[i]} and {pos[j]} in world {w}")

  @parameterized.parameters(1, 2)
  def test_flex_internal_collision(self, nworld):
    """Test that predefined element-vertex internal collisions generate contacts."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" internal="true" margin="0.05"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    self.assertGreater(m.nflexevpair, 0)

    evpair = m.flex_evpair.numpy()[0]
    e = int(evpair[0])
    v = int(evpair[1])

    dim = int(m.flex_dim.numpy()[0])
    elem_data_idx = int(m.flex_elemdataadr.numpy()[0]) + e * (dim + 1)
    v_indices = m.flex_elem.numpy()[elem_data_idx : elem_data_idx + dim + 1]

    v0_global_idx = int(m.flex_vertadr.numpy()[0]) + int(v_indices[0])
    v_global_idx = int(m.flex_vertadr.numpy()[0]) + v

    xpos = d.flexvert_xpos.numpy()
    for w in range(nworld):
      p0 = xpos[w, v0_global_idx]
      xpos[w, v_global_idx] = p0 + np.array([0.0, 0.0, 0.01])
    d.flexvert_xpos.assign(xpos)

    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected at least one contact from internal self-collision")

    contacts_worldid = d.contact.worldid.numpy()[:nacon]
    for w in range(nworld):
      w_indices = np.where(contacts_worldid == w)[0]
      self.assertGreater(len(w_indices), 0, f"Expected contacts in world {w}")
      idx = w_indices[0]
      self.assertEqual(int(d.contact.geom.numpy()[idx, 0]), -1)
      self.assertEqual(int(d.contact.geom.numpy()[idx, 1]), -1)
      self.assertEqual(int(d.contact.flex.numpy()[idx, 0]), 0)
      self.assertEqual(int(d.contact.flex.numpy()[idx, 1]), 0)
      self.assertEqual(int(d.contact.dim.numpy()[idx]), 3)

  @parameterized.parameters(1, 2)
  def test_flex_self_collision_1d(self, nworld):
    """Test active element self-collisions for 1D ropes (Capsule-Capsule)."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="rope" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="1" mass=".5">
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    m.flex_selfcollide.assign(np.array([4], dtype=np.int32))
    m.nflexevpair = 0

    v0_global_idx = int(m.flex_vertadr.numpy()[0])
    v_global_idx = int(m.flex_vertadr.numpy()[0]) + 3
    xpos = d.flexvert_xpos.numpy()
    for w in range(nworld):
      xpos[w, v_global_idx] = xpos[w, v0_global_idx] + np.array([0.0, 0.0, 0.01])
    d.flexvert_xpos.assign(xpos)

    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected at least one contact from 1D self-collision")

    contacts_worldid = d.contact.worldid.numpy()[:nacon]
    for w in range(nworld):
      found = False
      for idx in range(nacon):
        if contacts_worldid[idx] != w:
          continue
        g0 = int(d.contact.geom.numpy()[idx, 0])
        g1 = int(d.contact.geom.numpy()[idx, 1])
        f0 = int(d.contact.flex.numpy()[idx, 0])
        f1 = int(d.contact.flex.numpy()[idx, 1])
        e0 = int(d.contact.elem.numpy()[idx, 0])
        e1 = int(d.contact.elem.numpy()[idx, 1])

        if g0 == -1 and g1 == -1 and f0 == 0 and f1 == 0:
          if (e0 == 0 and e1 == 2) or (e0 == 2 and e1 == 0):
            found = True
            self.assertGreaterEqual(int(d.contact.dim.numpy()[idx]), 3)
            break

      self.assertTrue(found, f"Expected active element self-collision contact between element 0 and 2 not found in world {w}")

  @parameterized.parameters(1, 2)
  def test_flex_self_collision_2d(self, nworld):
    """Test active element self-collisions for 2D meshes (Triangle-Triangle via GJK/EPA)."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="2" mass=".5">
            <contact selfcollide="auto"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    elem_num = m.flex_elemnum.numpy()[0]
    dim = int(m.flex_dim.numpy()[0])
    elem_data_idx = int(m.flex_elemdataadr.numpy()[0])
    elem_verts = m.flex_elem.numpy()[elem_data_idx : elem_data_idx + elem_num * (dim + 1)].reshape(elem_num, dim + 1)

    e1 = -1
    e2 = -1
    for i in range(elem_num):
      for j in range(i + 1, elem_num):
        if len(set(elem_verts[i]) & set(elem_verts[j])) == 0:
          e1 = i
          e2 = j
          break
      if e1 >= 0:
        break

    self.assertGreaterEqual(e1, 0)
    self.assertGreaterEqual(e2, 0)

    vert_adr = int(m.flex_vertadr.numpy()[0])
    xpos = d.flexvert_xpos.numpy()

    for w in range(nworld):
      p_center1 = np.zeros(3)
      for v_idx in elem_verts[e1]:
        p_center1 += xpos[w, vert_adr + v_idx]
      p_center1 /= dim + 1

      p_center2 = np.zeros(3)
      for v_idx in elem_verts[e2]:
        p_center2 += xpos[w, vert_adr + v_idx]
      p_center2 /= dim + 1

      shift = p_center1 - p_center2 + np.array([0.0, 0.0, 0.005])
      for v_idx in elem_verts[e2]:
        xpos[w, vert_adr + v_idx] += shift

    d.flexvert_xpos.assign(xpos)

    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected at least one contact from 2D self-collision")

    contacts_worldid = d.contact.worldid.numpy()[:nacon]
    for w in range(nworld):
      found = False
      for idx in range(nacon):
        if contacts_worldid[idx] != w:
          continue
        g0 = int(d.contact.geom.numpy()[idx, 0])
        g1 = int(d.contact.geom.numpy()[idx, 1])
        f0 = int(d.contact.flex.numpy()[idx, 0])
        f1 = int(d.contact.flex.numpy()[idx, 1])
        elem0 = int(d.contact.elem.numpy()[idx, 0])
        elem1 = int(d.contact.elem.numpy()[idx, 1])

        if g0 == -1 and g1 == -1 and f0 == 0 and f1 == 0:
          if (elem0 == e1 and elem1 == e2) or (elem0 == e2 and elem1 == e1):
            found = True
            self.assertGreaterEqual(int(d.contact.dim.numpy()[idx]), 3)
            break

      self.assertTrue(
        found, f"Expected active element self-collision contact between element {e1} and {e2} not found in world {w}"
      )

  @parameterized.parameters(1, 2)
  def test_flex_self_collision_weld_exclusion(self, nworld):
    """Test self-collision exclusions when vertices are welded to the same body."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="rope" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="1" mass=".5">
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    m.flex_selfcollide.assign(np.array([4], dtype=np.int32))
    m.nflexevpair = 0

    v0_global_idx = int(m.flex_vertadr.numpy()[0])
    v_global_idx = int(m.flex_vertadr.numpy()[0]) + 3
    xpos = d.flexvert_xpos.numpy()
    for w in range(nworld):
      xpos[w, v_global_idx] = xpos[w, v0_global_idx] + np.array([0.0, 0.0, 0.01])
    d.flexvert_xpos.assign(xpos)

    vertbody = m.flex_vertbodyid.numpy()
    vertbody[v0_global_idx] = 1
    vertbody[v_global_idx] = 1
    m.flex_vertbodyid.assign(vertbody)

    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertEqual(nacon, 0, "Expected 0 contacts due to weld same-body exclusion")

  @parameterized.parameters(1, 2)
  def test_flex_self_collision_no_adjacent_contacts(self, nworld):
    """Test that a flat cloth does not generate any self-collision contacts."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco model="Poncho">
        <option solver="CG" tolerance="1e-6" jacobian="sparse"/>
        <worldbody>
          <flexcomp name="cloth" type="grid" count="10 10 1" spacing="0.05 0.05 0.05"
                    radius="0.01" dim="2" rgba="1 0.5 0.5 1" pos="0 0 2" mass=".1">
            <contact selfcollide="auto"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    mjw.kinematics(m, d)
    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertEqual(nacon, 0, f"Expected 0 self-collision contacts on a flat cloth, but got {nacon}")

  @parameterized.parameters(1, 2)
  def test_flex_mesh(self, nworld):
    """Test that contacts are generated between mesh and cloth."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".001"/>
      <size memory="10M"/>

      <asset>
        <mesh name="box" scale="0.1 0.1 0.1"
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
        <light pos="0 0 3" dir="0 0 -1"/>

        <!-- Ground plane -->
        <geom type="plane" size="5 5 .1" pos="0 0 0"/>

        <!-- Mesh positioned just above the cloth -->
        <body pos="0 0 0.12">
          <freejoint/>
          <geom type="mesh" mesh="box" mass="1"/>
        </body>

        <!-- Cloth (dim=2 flex) -->
        <flexcomp type="grid" count="4 4 1" spacing=".2 .2 .1" pos="-.3 -.3 0"
                  radius=".02" name="cloth" dim="2" mass=".5">
          <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001"
                   selfcollide="none" conaffinity="1" contype="1"/>
          <edge damping="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    self.assertEqual(mjm.nflex, 1)
    self.assertEqual(mjm.flex_dim[0], 2)

    self.assertEqual(m.nflex, 1)
    self.assertGreater(m.flex_elemnum.numpy()[0], 0)

    mjw.kinematics(m, d)
    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])

    self.assertGreater(nacon, 0, "Expected contacts between mesh and cloth")
    contacts_worldid = d.contact.worldid.numpy()[:nacon]
    for w in range(nworld):
      self.assertTrue(np.any(contacts_worldid == w), f"Expected contacts in world {w}")

  @parameterized.parameters(1, 2)
  def test_flex_lookup_maps(self, nworld):
    """Test that precomputed flex lookup maps are correctly populated."""
    xml = """
    <mujoco>
      <worldbody>
        <!-- Two distinct grid flex comps to test multi-flex models -->
        <flexcomp name="cloth1" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                  radius=".02" dim="2" mass=".5">
          <contact selfcollide="none" internal="true"/>
        </flexcomp>
        <flexcomp name="cloth2" type="grid" count="4 4 1" spacing=".2 .2 .1" pos="1 1 0"
                  radius=".02" dim="2" mass=".5">
          <contact selfcollide="none" internal="true"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, _ = test_data.fixture(xml=xml, nworld=nworld)

    self.assertEqual(m.nflex, 2)

    flex_elemflexid = m.flex_elemflexid.numpy()
    flex_evpairflexid = m.flex_evpairflexid.numpy()
    flex_shellflexid = m.flex_shellflexid.numpy()
    flex_vertflexid = m.flex_vertflexid.numpy()

    self.assertEqual(len(flex_elemflexid), m.nflexelem)
    self.assertEqual(len(flex_evpairflexid), m.nflexevpair)
    self.assertEqual(len(flex_shellflexid), m.nflexshelldata)
    self.assertEqual(len(flex_vertflexid), m.nflexvert)

    shell_offset = 0
    for i in range(m.nflex):
      elem_start = m.flex_elemadr.numpy()[i]
      elem_num = m.flex_elemnum.numpy()[i]
      np.testing.assert_array_equal(
        flex_elemflexid[elem_start : elem_start + elem_num],
        i,
        err_msg=f"Element mapping mismatch for flex {i}",
      )

      evpair_start = m.flex_evpairadr.numpy()[i]
      evpair_num = m.flex_evpairnum.numpy()[i]
      np.testing.assert_array_equal(
        flex_evpairflexid[evpair_start : evpair_start + evpair_num],
        i,
        err_msg=f"Element-vertex pair mapping mismatch for flex {i}",
      )

      self.assertEqual(m.flex_shelladr.numpy()[i], shell_offset)

      shell_num = m.flex_shellnum.numpy()[i]
      np.testing.assert_array_equal(
        flex_shellflexid[shell_offset : shell_offset + shell_num],
        i,
        err_msg=f"Shell mapping mismatch for flex {i}",
      )
      shell_offset += shell_num

      vert_start = m.flex_vertadr.numpy()[i]
      vert_num = m.flex_vertnum.numpy()[i]
      np.testing.assert_array_equal(
        flex_vertflexid[vert_start : vert_start + vert_num],
        i,
        err_msg=f"Vertex mapping mismatch for flex {i}",
      )

  @parameterized.parameters(1, 2)
  def test_sphere_cloth_pruned_by_broadphase(self, nworld):
    """Test that far-away geoms are successfully pruned by broadphase (yielding 0 contacts)."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".001"/>
      <size memory="10M"/>

      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>

        <!-- Sphere positioned very far away from the cloth -->
        <body pos="10.0 10.0 10.0">
          <freejoint/>
          <geom type="sphere" size=".1" mass="1"/>
        </body>

        <!-- Cloth (dim=2 flex) -->
        <flexcomp type="grid" count="4 4 1" spacing=".2 .2 .1" pos="-.3 -.3 0"
                  radius=".02" name="cloth" dim="2" mass=".5">
          <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001"
                   selfcollide="none" conaffinity="1" contype="1"/>
          <edge damping="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    d.nacon.zero_()
    mjw.kinematics(m, d)
    mjw.collision(m, d)

    self.assertEqual(d.nacon.numpy()[0], 0, "Expected 0 contacts because the sphere is very far away")

  @parameterized.parameters(1, 2)
  def test_sphere_cloth_exact_bounds(self, nworld):
    """Test that the dynamic flex AABB calculation computes the exact expected bounding box."""
    xml = """
    <mujoco>
      <worldbody>
        <!-- Cloth (dim=2 flex) -->
        <flexcomp type="grid" count="4 4 1" spacing=".2 .2 .1" pos="-.3 -.3 0"
                  radius=".02" name="cloth" dim="2" mass=".5">
          <contact selfcollide="none"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    mjw.kinematics(m, d)
    mjw.collision(m, d)

    radius = m.flex_radius.numpy()[0]
    margin = m.flex_margin.numpy()[0] + m.flex_gap.numpy()[0]
    inflate = radius + margin

    vert_adr = m.flex_vertadr.numpy()[0]
    vert_num = m.flex_vertnum.numpy()[0]

    for w in range(nworld):
      aabb_min = d.flex_aabb_min.numpy()[w, 0]
      aabb_max = d.flex_aabb_max.numpy()[w, 0]
      verts = d.flexvert_xpos.numpy()[w, vert_adr : vert_adr + vert_num]
      v_min = np.min(verts, axis=0)
      v_max = np.max(verts, axis=0)
      expected_min = v_min - inflate
      expected_max = v_max + inflate
      np.testing.assert_allclose(aabb_min, expected_min, atol=1e-5)
      np.testing.assert_allclose(aabb_max, expected_max, atol=1e-5)

  @parameterized.parameters(1, 2)
  def test_plane_cloth_contact_generated(self, nworld):
    """Test that contacts are generated between plane and cloth vertices."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".001"/>
      <size memory="10M"/>

      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>

        <!-- Ground plane -->
        <geom type="plane" size="5 5 .1" pos="0 0 0"/>

        <!-- Cloth (dim=2 flex) placed just above the plane -->
        <flexcomp type="grid" count="4 4 1" spacing=".2 .2 .1" pos="-.3 -.3 0.01"
                  radius=".02" name="cloth" dim="2" mass=".5">
          <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001"
                   selfcollide="none" conaffinity="1" contype="1"/>
          <edge damping="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    mjw.kinematics(m, d)
    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected contacts between plane and cloth vertices")

    contact_geom = d.contact.geom.numpy()[:nacon]
    contacts_worldid = d.contact.worldid.numpy()[:nacon]
    for w in range(nworld):
      w_indices = np.where(contacts_worldid == w)[0]
      self.assertGreater(len(w_indices), 0, f"Expected contacts in world {w}")
      plane_contacts = np.sum(contact_geom[w_indices, 0] == 0)
      self.assertGreater(plane_contacts, 0, f"Expected at least one contact with the plane in world {w}")

  @parameterized.parameters(1, 2)
  def test_mixed_flex_broadphase_and_narrowphase(self, nworld):
    """Test that broadphase and narrowphase run correctly with mixed 2D and 3D flexes."""
    xml = """
    <mujoco>
      <worldbody>
        <!-- 2D Cloth -->
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                  radius=".02" dim="2" mass=".5">
          <contact selfcollide="none" internal="true"/>
        </flexcomp>
        <!-- 3D Softbody -->
        <flexcomp name="softbody" type="grid" count="3 3 3" spacing=".2 .2 .2" pos="1 1 0"
                  radius=".02" dim="3" mass="1.0">
          <contact selfcollide="none" internal="true"/>
        </flexcomp>
        <!-- A sphere positioned near the cloth to generate contact -->
        <body pos="0 0 0.05">
          <joint type="free"/>
          <geom type="sphere" size="0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    self.assertEqual(m.nflex, 2)
    self.assertEqual(m.flex_dim.numpy()[0], 2)
    self.assertEqual(m.flex_dim.numpy()[1], 3)

    mjw.kinematics(m, d)
    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected contacts to be generated")
    contacts_worldid = d.contact.worldid.numpy()[:nacon]
    for w in range(nworld):
      self.assertTrue(np.any(contacts_worldid == w), f"Expected contacts in world {w}")

  @parameterized.parameters(1, 2)
  def test_ellipsoid_cloth_contact_generated(self, nworld):
    """Test that contacts are generated between ellipsoid and cloth."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option solver="CG" tolerance="1e-6" timestep=".001"/>
        <size memory="10M"/>

        <worldbody>
          <light pos="0 0 3" dir="0 0 -1"/>

          <!-- Ground plane -->
          <geom type="plane" size="5 5 .1" pos="0 0 0"/>

          <!-- Ellipsoid positioned just above the cloth -->
          <body pos="0 0 0.12">
            <freejoint/>
            <geom type="ellipsoid" size=".1 .15 .08" mass="1"/>
          </body>

          <!-- Cloth (dim=2 flex) -->
          <flexcomp type="grid" count="4 4 1" spacing=".2 .2 .1" pos="-.3 -.3 0"
                    radius=".02" name="cloth" dim="2" mass=".5">
            <contact condim="3" solref="0.01 1" solimp=".95 .99 .0001"
                     selfcollide="none" conaffinity="1" contype="1"/>
            <edge damping="0.01"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )

    self.assertEqual(mjm.nflex, 1)
    self.assertEqual(mjm.flex_dim[0], 2)

    self.assertEqual(m.nflex, 1)
    self.assertGreater(m.flex_elemnum.numpy()[0], 0)

    mjw.kinematics(m, d)
    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])

    self.assertGreater(nacon, 0, "Expected contacts between ellipsoid and cloth")
    contacts_worldid = d.contact.worldid.numpy()[:nacon]
    for w in range(nworld):
      self.assertTrue(np.any(contacts_worldid == w), f"Expected contacts in world {w}")

  @parameterized.named_parameters(
    (
      "hfield",
      """
      <mujoco>
        <asset>
          <hfield name="terrain" nrow="2" ncol="2" size="1 1 0.1 0.1" elevation="0 0 0 0"/>
        </asset>
        <worldbody>
          <geom type="hfield" hfield="terrain"/>
          <flexcomp type="grid" count="2 2 1" spacing=".2 .2 .1" pos="0 0 0.1" name="cloth" dim="2"/>
        </worldbody>
      </mujoco>
      """,
    ),
    (
      "sdf",
      """
      <mujoco>
        <asset>
          <mesh name="cube"
           vertex="1 1 1  1 1 -1  1 -1 1  1 -1 -1  -1 1 1  -1 1 -1  -1 -1 1  -1 -1 -1"/>
        </asset>
        <worldbody>
          <body pos="0 0 1">
            <freejoint/>
            <geom type="sdf" mesh="cube"/>
          </body>
          <flexcomp type="grid" count="2 2 1" spacing=".2 .2 .1" pos="0 0 0.1" name="cloth" dim="2"/>
        </worldbody>
      </mujoco>
      """,
    ),
  )
  def test_unsupported_flex_collision_error(self, xml):
    """Test that loading a model with unsupported geoms and Flex raises NotImplementedError."""
    with self.assertRaises(NotImplementedError):
      test_data.fixture(xml=xml)


class FlexInternalCollisionTest(parameterized.TestCase):
  """Tests for internal (self-collision element-vertex) contacts."""

  def _get_sorted_internal_contacts(self, d_or_mjd, ncon, world_idx=0, is_warp=True):
    """Helper to extract and sort internal contacts (vert[0] >= 0)."""
    contacts = []
    if is_warp:
      contacts_worldid = d_or_mjd.contact.worldid.numpy()[:ncon]
      for i in range(ncon):
        if contacts_worldid[i] != world_idx:
          continue
        vert = d_or_mjd.contact.vert.numpy()[i]
        if vert[0] >= 0:  # Internal element-vertex contact
          contacts.append(
            {
              "dist": d_or_mjd.contact.dist.numpy()[i],
              "pos": d_or_mjd.contact.pos.numpy()[i],
              "frame": d_or_mjd.contact.frame.numpy()[i],
              "geom": d_or_mjd.contact.geom.numpy()[i],
              "flex": d_or_mjd.contact.flex.numpy()[i],
              "elem": d_or_mjd.contact.elem.numpy()[i],
              "vert": vert,
            }
          )
    else:
      for i in range(ncon):
        c = d_or_mjd.contact[i]
        if c.vert[0] >= 0:  # Internal element-vertex contact
          # Extract 3x3 contact frame (normal is first row)
          frame = c.frame.reshape(3, 3)
          contacts.append(
            {
              "dist": c.dist,
              "pos": c.pos,
              "frame": frame,
              "geom": c.geom,
              "flex": c.flex,
              "elem": c.elem,
              "vert": c.vert,
            }
          )
    # Sort by vert[0] (the colliding vertex ID)
    contacts.sort(key=lambda x: x["vert"][0])
    return contacts

  def _assert_contact_parity(self, w_contacts, m_contacts):
    self.assertEqual(
      len(w_contacts),
      len(m_contacts),
      f"Contact count mismatch: Warp {len(w_contacts)} vs MuJoCo {len(m_contacts)}",
    )
    for i in range(len(w_contacts)):
      w_con = w_contacts[i]
      m_con = m_contacts[i]

      # Compare distance and positions.
      np.testing.assert_allclose(
        w_con["dist"],
        m_con["dist"],
        atol=_TOLERANCE,
        err_msg=f"dist mismatch for contact {i}",
      )
      np.testing.assert_allclose(
        w_con["pos"],
        m_con["pos"],
        atol=_TOLERANCE,
        err_msg=f"pos mismatch for contact {i}",
      )

      # Check normal alignment
      w_normal = w_con["frame"][0]
      m_normal = m_con["frame"][0]
      cos_sim = np.dot(w_normal, m_normal)
      self.assertGreater(
        cos_sim,
        0.99,
        f"Normal mismatch for contact {i}: Warp {w_normal}, MuJoCo {m_normal}",
      )

      # Identifiers
      self.assertEqual(w_con["geom"][0], m_con["geom"][0])
      self.assertEqual(w_con["geom"][1], m_con["geom"][1])
      self.assertEqual(w_con["flex"][0], m_con["flex"][0])
      self.assertEqual(w_con["elem"][1], m_con["elem"][1])
      self.assertEqual(w_con["vert"][0], m_con["vert"][0])

  @parameterized.parameters(1, 2)
  def test_internal_collision_cloth(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.05" dim="2" mass="1">
          <contact internal="true" selfcollide="auto"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    # Displace Vertex 2 to Vertex 6 + (0, 0, 0.005)
    # qpos [6, 7, 8] for Vertex 2
    mjd.qpos[6] = 0.2
    mjd.qpos[7] = -0.2
    mjd.qpos[8] = 0.005

    # Replicate to all worlds in Warp
    qpos_np = np.tile(mjd.qpos, (nworld, 1))
    d.qpos.assign(qpos_np.astype(np.float32))

    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    w_contacts_0 = self._get_sorted_internal_contacts(d, d.nacon.numpy()[0], world_idx=0, is_warp=True)
    m_contacts = self._get_sorted_internal_contacts(mjd, mjd.ncon, is_warp=False)
    self.assertGreater(len(w_contacts_0), 0)
    self.assertGreater(len(m_contacts), 0)

    for w in range(nworld):
      w_contacts = self._get_sorted_internal_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      self._assert_contact_parity(w_contacts, m_contacts)

  @parameterized.parameters(1, 2)
  def test_internal_collision_rope(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp name="rope" type="grid" count="5 1 1" spacing="0.1 0.1 0.1" pos="0 0 0.05" dim="1" mass="1">
          <contact internal="true" selfcollide="auto"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    # Displace Vertex 4 to Edge 0 center + (0, 0, 0.005)
    # v4 is joints 12, 13, 14
    mjd.qpos[12] = -0.35
    mjd.qpos[13] = 0.0
    mjd.qpos[14] = 0.005

    qpos_np = np.tile(mjd.qpos, (nworld, 1))
    d.qpos.assign(qpos_np.astype(np.float32))

    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    w_contacts_0 = self._get_sorted_internal_contacts(d, d.nacon.numpy()[0], world_idx=0, is_warp=True)
    m_contacts = self._get_sorted_internal_contacts(mjd, mjd.ncon, is_warp=False)
    self.assertGreater(len(w_contacts_0), 0)
    self.assertGreater(len(m_contacts), 0)

    for w in range(nworld):
      w_contacts = self._get_sorted_internal_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      self._assert_contact_parity(w_contacts, m_contacts)


class FlexDynamicsTest(parameterized.TestCase):
  """Short integration tests (<= 10 steps) comparing trajectories with MuJoCo."""

  @parameterized.product(
    xml_and_atol=[
      # Swinging Rope (gravity + edge equality)
      (
        """
        <mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody>
            <flexcomp name="rope" type="grid" count="5 1 1" spacing="0.1 0.1 0.1" dim="1" mass="1">
              <edge equality="true"/>
              <pin id="0"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        1e-3,
      ),
      # Flapping Cloth (gravity + elasticity + bending)
      (
        """
        <mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody>
            <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" dim="2" mass="1">
              <elasticity young="1e3" poisson="0.3" damping="1e-3"/>
              <pin id="0 2"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        1e-3,
      ),
      # Trilinear Softbody falling under gravity (strain equality)
      (
        """
        <mujoco>
          <option gravity="0 0 -9.81">
            <flag contact="disable"/>
          </option>
          <worldbody>
            <flexcomp type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                      pos="0 0 0.5" name="cube" dim="3" mass="1" radius="0.005"
                      dof="trilinear">
              <contact selfcollide="none"/>
              <edge equality="strain"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        0.01,
      ),
    ],
    nworld=[1, 2],
  )
  def test_dynamics_parity(self, xml_and_atol, nworld):
    xml, atol = xml_and_atol
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    # Run 10 steps
    for _ in range(10):
      mujoco.mj_step(mjm, mjd)
      mjw.step(m, d)

    for w in range(nworld):
      np.testing.assert_allclose(
        d.qpos.numpy()[w],
        mjd.qpos,
        atol=atol,
        err_msg=f"qpos mismatch after 10 steps (world {w})",
      )
      np.testing.assert_allclose(
        d.qvel.numpy()[w],
        mjd.qvel,
        atol=atol,
        err_msg=f"qvel mismatch after 10 steps (world {w})",
      )

  @parameterized.parameters(1, 2)
  def test_trilinear_contact_dynamics(self, nworld):
    """Test trilinear flex contact dynamics over 10 steps."""
    xml = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <flexcomp type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                  pos="0 0 0.05" name="cube" dim="3" mass="1" radius="0.02"
                  dof="trilinear">
          <edge equality="strain"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    # Run 10 steps
    for _ in range(10):
      mujoco.mj_step(mjm, mjd)
      mjw.step(m, d)

    for w in range(nworld):
      np.testing.assert_allclose(
        d.qpos.numpy()[w],
        mjd.qpos,
        atol=1e-2,  # Larger tolerance for contact dynamics
        err_msg=f"qpos mismatch after 10 steps with contact (world {w})",
      )

  @parameterized.parameters(1, 2)
  def test_drape_dynamics(self, nworld):
    """Test cloth draping over a sphere (bending + collision + dynamics)."""
    xml = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <geom type="sphere" size="0.05" pos="0 0 0.02"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.08" dim="2" mass="1">
          <elasticity young="1e3" poisson="0.3" thickness="1e-3" damping="1e-3" elastic2d="both"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    # Run 50 steps
    for _ in range(50):
      mujoco.mj_step(mjm, mjd)
      mjw.step(m, d)

    for w in range(nworld):
      np.testing.assert_allclose(
        d.qpos.numpy()[w],
        mjd.qpos,
        atol=1e-3,  # Tight tolerance for dynamics
        err_msg=f"qpos mismatch after 50 steps drape (world {w})",
      )
      np.testing.assert_allclose(
        d.qvel.numpy()[w],
        mjd.qvel,
        atol=5e-3,  # Slightly larger tolerance for velocities
        err_msg=f"qvel mismatch after 50 steps drape (world {w})",
      )

  @parameterized.parameters(1, 2)
  def test_multiflex(self, nworld):
    """Tests multiflex model with different flex dimensions."""
    mjm, mjd, m, d = test_data.fixture("flex/multiflex.xml", qpos_noise=0.02, nworld=nworld)

    mjw.forward(m, d)
    mujoco.mj_forward(mjm, mjd)

    for w in range(nworld):
      np.testing.assert_allclose(d.qacc.numpy()[w], mjd.qacc, atol=5e-2, rtol=0, err_msg=f"qacc mismatch (world {w})")


class FlexContactParityTest(parameterized.TestCase):
  """Tests for flex contact details parity with MuJoCo."""

  def _get_sorted_contacts(self, d, ncon, world_idx=0, is_warp=True):
    contacts = []
    if is_warp:
      total_nacon = d.nacon.numpy()[0]
      worldids = d.contact.worldid.numpy()[:total_nacon]
      for i in range(total_nacon):
        if worldids[i] == world_idx:
          contacts.append(
            {
              "dist": d.contact.dist.numpy()[i],
              "pos": d.contact.pos.numpy()[i],
              "frame": d.contact.frame.numpy()[i],
              "geom": d.contact.geom.numpy()[i],
              "flex": d.contact.flex.numpy()[i],
              "elem": d.contact.elem.numpy()[i],
              "vert": d.contact.vert.numpy()[i],
            }
          )
    else:
      for i in range(ncon):
        c = d.contact[i]
        contacts.append(
          {
            "dist": c.dist,
            "pos": c.pos,
            "frame": c.frame.reshape(3, 3),
            "geom": c.geom,
            "flex": c.flex,
            "elem": c.elem,
            "vert": c.vert,
          }
        )
    # Sort by (flex, elem, vert, pos) to ensure matching order
    contacts.sort(
      key=lambda c: (
        tuple(c["flex"]),
        tuple(c["elem"]),
        tuple(c["vert"]),
        tuple(c["pos"]),
      )
    )
    return contacts

  def _assert_contact_parity(self, w_contacts, m_contacts, atol=1e-5):
    self.assertEqual(len(w_contacts), len(m_contacts))
    for i, (wc, mc) in enumerate(zip(w_contacts, m_contacts)):
      np.testing.assert_allclose(
        wc["dist"],
        mc["dist"],
        atol=atol,
        err_msg=f"Contact {i} dist mismatch",
      )
      np.testing.assert_allclose(wc["pos"], mc["pos"], atol=atol, err_msg=f"Contact {i} pos mismatch")
      np.testing.assert_allclose(
        wc["frame"],
        mc["frame"],
        atol=atol,
        err_msg=f"Contact {i} frame mismatch",
      )
      np.testing.assert_equal(wc["geom"], mc["geom"], err_msg=f"Contact {i} geom mismatch")
      np.testing.assert_equal(wc["flex"], mc["flex"], err_msg=f"Contact {i} flex mismatch")
      np.testing.assert_equal(wc["elem"], mc["elem"], err_msg=f"Contact {i} elem mismatch")
      np.testing.assert_equal(wc["vert"], mc["vert"], err_msg=f"Contact {i} vert mismatch")

  @parameterized.parameters(1, 2)
  def test_contact_plane_single_vertex(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.02" dim="2" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    # Displace vertex 0 Z downward to penetrate plane
    mjd.qpos[2] = -0.025
    d.qpos.assign(np.tile(mjd.qpos, (nworld, 1)).astype(np.float32))

    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertEqual(d.nacon.numpy()[0], nworld * 1)
    self.assertEqual(mjd.ncon, 1)

    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts)

  @parameterized.parameters(1, 2)
  def test_contact_plane_multi_vertex(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.02" dim="2" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    # Displace 4 corner vertices (0, 2, 6, 8) Z downward
    mjd.qpos[2] = -0.025
    mjd.qpos[8] = -0.025
    mjd.qpos[20] = -0.025
    mjd.qpos[26] = -0.025
    d.qpos.assign(np.tile(mjd.qpos, (nworld, 1)).astype(np.float32))

    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertEqual(d.nacon.numpy()[0], nworld * 4)
    self.assertEqual(mjd.ncon, 4)

    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts)

  @parameterized.parameters(1, 2)
  def test_contact_sphere_localized(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <geom type="sphere" size="0.02" pos="-0.066 -0.033 0.06"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.05" dim="2" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertEqual(d.nacon.numpy()[0], nworld * 1)
    self.assertEqual(mjd.ncon, 1)

    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts)

  @parameterized.parameters(1, 2)
  def test_contact_box_localized(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <geom type="box" size="0.01 0.01 0.01" pos="-0.033 -0.066 0.063" euler="30 45 15"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.05" dim="2" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertEqual(d.nacon.numpy()[0], nworld * 1)
    self.assertEqual(mjd.ncon, 1)

    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts)

  @parameterized.parameters(1, 2)
  def test_contact_capsule_localized(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <geom type="capsule" size="0.01 0.03" pos="-0.066 -0.033 0.058" euler="90 0 0"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing="0.1 0.1 0.1" pos="0 0 0.05" dim="2" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertEqual(d.nacon.numpy()[0], nworld * 4)
    self.assertEqual(mjd.ncon, 4)

    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts)


class FlexContactConstraintTest(parameterized.TestCase):
  """Tests for flex contact constraint generation (efc matrices) parity."""

  @parameterized.product(
    xml=(
      """
          <mujoco>
            <option solver="CG" tolerance="1e-6" timestep=".001"/>
            <worldbody>
              <!-- Sphere positioned to press into the cloth -->
              <body pos="0 0 0.05">
                <freejoint/>
                <geom type="sphere" size=".1" mass="1"/>
              </body>
              <!-- Cloth (dim=2 flex) -->
              <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".3 .3 .1" pos="-.3 -.3 0"
                        radius=".02" dim="2" mass=".5">
                <contact condim="3" selfcollide="none"/>
              </flexcomp>
            </worldbody>
          </mujoco>
          """,
      """
          <mujoco>
            <option solver="CG" tolerance="1e-6" timestep=".001"/>
            <worldbody>
              <!-- Box positioned to press into the soft body -->
              <body pos="0 0 0.1">
                <freejoint/>
                <geom type="box" size=".05 .05 .05" mass="1"/>
              </body>
              <!-- Soft body (dim=3 flex) -->
              <flexcomp name="softbody" type="grid" count="2 2 2" spacing=".15 .15 .15" pos="-.075 -.075 0"
                        radius=".01" dim="3" mass=".5">
                <contact condim="3" selfcollide="none"/>
              </flexcomp>
            </worldbody>
          </mujoco>
          """,
    ),
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
    nworld=[1, 2],
  )
  def test_flex_barycentric_jacobian(self, xml, cone, jacobian, nworld):
    """Test barycentric contact Jacobian calculation for flex."""
    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.cone": cone, "opt.jacobian": jacobian}, nworld=nworld)

    mjw.kinematics(m, d)
    mjw.make_constraint(m, d)

    self.assertGreater(mjd.nefc, 0, "Expected active contacts")
    for w in range(nworld):
      self.assertEqual(d.nefc.numpy()[w], mjd.nefc, "nefc mismatch")

    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, f"efc_flex_dim{m.flex_dim.numpy()[0]}", m.nv, nworld)

  @parameterized.parameters(1, 2)
  def test_flex_3d_simplex_collision(self, nworld):
    """Test 3D simplex flex collision and constraint generation."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".001"/>
      <worldbody>
        <!-- Sphere positioned to press into the soft body shifted -->
        <body pos="-0.055 -0.075 0.1">
          <freejoint/>
          <geom type="sphere" size=".05" mass="1"/>
        </body>
        <!-- Soft body (dim=3 flex, simplex by default) -->
        <flexcomp name="softbody" type="grid" count="2 2 2" spacing=".15 .15 .15" pos="-.075 -.075 0"
                  radius=".01" dim="3" mass=".5">
          <contact condim="3" selfcollide="none" solimp="0.9 0.95 0.1"/>
          <elasticity young="1e4" poisson="0.2" damping="0.002"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=nworld)

    mjw.kinematics(m, d)
    mjw.collision(m, d)
    mjw.make_constraint(m, d)

    self.assertGreater(mjd.nefc, 0, "Expected active contacts")
    for w in range(nworld):
      self.assertEqual(d.nefc.numpy()[w], mjd.nefc, "nefc mismatch")

    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc_flex_3d_simplex", m.nv, nworld, tol=1e-3)

  @parameterized.parameters(1, 2)
  def test_flex_interpolated(self, nworld):
    """Test collision and constraint generation for interpolated flex shells."""
    xml = """
      <mujoco>
        <option solver="CG" tolerance="1e-6" timestep=".001"/>
        <worldbody>
          <body pos="0 0 0.07">
            <freejoint/>
            <geom type="box" size=".05 .05 .05" mass="1"/>
          </body>
          <flexcomp name="softbody" type="grid" count="4 2 4" spacing=".025 .05 .025" pos="0 0 0"
                    dim="3" cellcount="2 1 2" radius=".001"
                    mass="5" dof="trilinear">
            <elasticity young="1e4" poisson="0.3" damping="1e-3"/>
            <contact condim="3" selfcollide="none" internal="false"/>
          </flexcomp>
        </worldbody>
      </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(
      xml=xml,
      overrides={
        "opt.cone": mujoco.mjtCone.mjCONE_ELLIPTIC,
        "opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE,
      },
      nworld=nworld,
    )

    mjw.kinematics(m, d)
    mjw.make_constraint(m, d)

    self.assertGreater(mjd.nefc, 0, "Expected active contacts in MuJoCo")
    for w in range(nworld):
      self.assertEqual(d.nefc.numpy()[w], mjd.nefc, "nefc mismatch")
    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc_flex_interpolated_trilinear", m.nv, nworld)


class FlexBVHTest(parameterized.TestCase):
  """Tests for flex BVH functions."""

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
    flex_elemdataadr = wp.array([0], dtype=int)
    flex_elemadr = wp.array([0], dtype=int)
    flex_elemnum = wp.array([len(flex_elem)], dtype=int)
    flex_vertadr = wp.array([0], dtype=int)
    flex_dim = wp.array([2], dtype=int)
    flexvert_norm = wp.zeros((nworld, nvert), dtype=wp.vec3)

    wp.launch(
      kernel=bvh.accumulate_flex_vertex_normals,
      dim=(nworld, nelem),
      inputs=[1, flex_dim, flex_vertadr, flex_elemadr, flex_elemnum, flex_elemdataadr, flex_elem, flexvert_xpos],
      outputs=[flexvert_norm],
    )

    normals = flexvert_norm.numpy()
    self.assertTrue(np.any(normals != 0), "flexvert_norm")

  @parameterized.parameters(1, 2)
  def test_normalize_vertex_normals(self, nworld):
    """Tests flex vertex normal normalization kernel."""
    nvert = 3

    flexvert_norm = wp.array(
      [[[0, 0, 2], [0, 3, 0], [4, 0, 0]]] * nworld,
      dtype=wp.vec3,
    )

    wp.launch(
      kernel=bvh.normalize_vertex_normals,
      dim=(nworld, nvert),
      inputs=[flexvert_norm],
    )

    normals = flexvert_norm.numpy()
    for w in range(nworld):
      for i in range(nvert):
        norm = np.linalg.norm(normals[w, i])
        np.testing.assert_allclose(norm, 1.0, rtol=1e-5, err_msg="flexvert_norm")

  @parameterized.parameters(1, 2)
  def test_build_flex_bvh(self, nworld):
    """Tests that build_flex_bvh creates a valid BVH."""
    mjm, mjd, m, d = test_data.fixture("flex/floppy.xml", nworld=nworld)

    flex_mesh, group_root = bvh.build_flex_bvh(mjm, mjd, nworld, 0)

    self.assertNotEqual(flex_mesh.id, wp.uint64(0), "flex_mesh id")


class FlexSensorTest(parameterized.TestCase):
  """Tests for flex sensors."""

  @parameterized.parameters(1, 2)
  def test_insidesite_flex_body(self, nworld):
    """Test insidesite uses subtree_com for massless flex parent bodies."""
    _, mjd, m, d = test_data.fixture(
      xml="""
    <mujoco>
      <worldbody>
        <site name="sensor_site" type="sphere" size="2" pos="0 0 .5"/>
        <body name="flex_parent" pos="0 0 .5">
          <flexcomp type="grid" count="3 3 1" spacing=".1 .1 .1"
                    radius=".0" name="softbody" dim="2" mass="1">
            <contact condim="3" selfcollide="none"/>
          </flexcomp>
        </body>
      </worldbody>
      <sensor>
        <insidesite site="sensor_site" objtype="body" objname="flex_parent"/>
      </sensor>
    </mujoco>
    """,
      nworld=nworld,
    )

    d.sensordata.fill_(wp.inf)
    mjw.sensor_pos(m, d)

    for w in range(nworld):
      np.testing.assert_allclose(
        d.sensordata.numpy()[w],
        mjd.sensordata,
        atol=_TOLERANCE,
        err_msg=f"sensordata mismatch (world {w})",
      )


if __name__ == "__main__":
  wp.init()
  absltest.main()
