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
from mujoco_warp._src import collision_core
from mujoco_warp._src import collision_flex
from mujoco_warp._src import io
from mujoco_warp._src import types

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
  def test_mesh_rope_collision(self, nworld):
    """Test contacts for 1D rope colliding with a mesh geom via CCD."""
    xml = """
    <mujoco>
      <asset>
        <mesh name="box_mesh" vertex="-0.05 -0.05 -0.05  0.05 -0.05 -0.05  0.05 0.05 -0.05  -0.05 0.05 -0.05
                                      -0.05 -0.05 0.05   0.05 -0.05 0.05   0.05 0.05 0.05   -0.05 0.05 0.05"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="box_mesh" pos="0.1 0 0.05"/>
        <flexcomp name="rope" type="grid" count="5 1 1" spacing="0.1 0.1 0.1" pos="0 0 0.08" dim="1" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    self.assertGreater(d.nacon.numpy()[0], 0, "Expected contacts in Warp for 1D rope vs mesh")

  @parameterized.parameters(1, 2)
  def test_ellipsoid_rope_collision(self, nworld):
    """Test contacts for 1D rope colliding with an ellipsoid geom via CCD."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="ellipsoid" size="0.05 0.05 0.05" pos="0.1 0 0.05"/>
        <flexcomp name="rope" type="grid" count="5 1 1" spacing="0.1 0.1 0.1" pos="0 0 0.08" dim="1" mass="1">
          <contact condim="3"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    self.assertGreater(d.nacon.numpy()[0], 0, "Expected contacts in Warp for 1D rope vs ellipsoid")

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
  def test_flex_self_collision_bitmask_filtering(self, nworld):
    """Test that flex self-collision requires (flex_contype & flex_conaffinity) != 0."""
    # Disjoint bitmasks: contype="1" conaffinity="2" (1 & 2 == 0)
    _, _, m_disjoint, d_disjoint = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="rope" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="1" mass=".5">
            <contact selfcollide="auto" contype="1" conaffinity="2"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    self.assertFalse(m_disjoint.has_flex_selfcollide)

    # Overlapping bitmasks: contype="3" conaffinity="1" (3 & 1 == 1 != 0)
    _, _, m_overlap, d_overlap = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="rope" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0"
                    radius=".02" dim="1" mass=".5">
            <contact selfcollide="auto" contype="3" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    self.assertTrue(m_overlap.has_flex_selfcollide)

    v0_idx = int(m_overlap.flex_vertadr.numpy()[0])
    v3_idx = v0_idx + 3
    xpos = d_overlap.flexvert_xpos.numpy()
    for w in range(nworld):
      xpos[w, v3_idx] = xpos[w, v0_idx] + np.array([0.0, 0.0, 0.01])
    d_overlap.flexvert_xpos.assign(xpos)

    mjw.collision(m_overlap, d_overlap)
    nacon = int(d_overlap.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected self-collision contacts when contype & conaffinity != 0")

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
          <contact selfcollide="none" internal="false"/>
        </flexcomp>
        <flexcomp name="cloth2" type="grid" count="4 4 1" spacing=".2 .2 .1" pos="1 1 0"
                  radius=".02" dim="2" mass=".5">
          <contact selfcollide="none" internal="false"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, _ = test_data.fixture(xml=xml, nworld=nworld)

    self.assertEqual(m.nflex, 2)

    flex_elemflexid = m.flex_elemflexid.numpy()
    flex_shellflexid = m.flex_shellflexid.numpy()
    flex_vertflexid = m.flex_vertflexid.numpy()

    self.assertEqual(len(flex_elemflexid), m.nflexelem)
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
  def test_plane_cloth_no_fps_limit(self, nworld):
    """Test that flex-geom contacts are not limited to MJ_MAXCONPAIR (50)."""
    xml = """
    <mujoco>
      <option solver="CG" tolerance="1e-6" timestep=".001"/>
      <size memory="10M"/>

      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="5 5 .1" pos="0 0 0"/>
        <flexcomp type="grid" count="10 10 1" spacing=".1 .1 .1" pos="-.5 -.5 0.01"
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
    expected_contacts = 100 * nworld
    self.assertEqual(nacon, expected_contacts, f"Expected {expected_contacts} contacts, got {nacon}")

  @parameterized.parameters(1, 2)
  def test_flex_fps_capping(self, nworld):
    """Test that flex-flex contacts are limited to MJ_MAXCONPAIR (50) via parallel FPS."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cloth1" type="grid" count="8 8 1" spacing=".05 .05 .05" pos="0 0 0"
                    radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
          <flexcomp name="cloth2" type="grid" count="8 8 1" spacing=".05 .05 .05" pos="0 0 0.01"
                    radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=1500,
    )
    d.nacon.fill_(-1)
    d.contact.dist.fill_(wp.inf)
    mjw.kinematics(m, d)
    mjw.collision(m, d)
    nacon = int(d.nacon.numpy()[0])
    self.assertEqual(nacon, types.MJ_MAXCONPAIR * nworld)

  def test_flex_sat_prefilter_conservative(self):
    """Test that broadphase SAT prefilter is conservative and yields identical contacts."""
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp name="cloth" type="grid" count="4 4 1" spacing=".05 .05 .05" pos="0 0 0"
                  radius=".02" dim="2" mass=".5">
          <contact selfcollide="auto" contype="1" conaffinity="1"/>
          <edge damping="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d_sat = test_data.fixture(xml=xml, nworld=1, nconmax=500)
    _, _, _, d_nosat = test_data.fixture(xml=xml, nworld=1, nconmax=500)

    # Deform vertices slightly to create self-collisions
    qpos = d_sat.qpos.numpy()
    qpos[0, 0] += 0.05
    qpos[0, 1] += 0.05
    d_sat.qpos.assign(qpos)
    d_nosat.qpos.assign(qpos)

    d_sat.nacon.fill_(-1)
    d_sat.contact.dist.fill_(wp.inf)
    d_nosat.nacon.fill_(-1)
    d_nosat.contact.dist.fill_(wp.inf)

    orig_sat = collision_flex.ENABLE_SAT_PREFILTER
    try:
      collision_flex.ENABLE_SAT_PREFILTER = True
      mjw.kinematics(m, d_sat)
      mjw.collision(m, d_sat)

      collision_flex.ENABLE_SAT_PREFILTER = False
      mjw.kinematics(m, d_nosat)
      mjw.collision(m, d_nosat)
    finally:
      collision_flex.ENABLE_SAT_PREFILTER = orig_sat

    nacon_sat = int(d_sat.nacon.numpy()[0])
    nacon_nosat = int(d_nosat.nacon.numpy()[0])
    ncoll_sat = int(d_sat.ncollision.numpy()[0])
    ncoll_nosat = int(d_nosat.ncollision.numpy()[0])

    # Verify SAT actively prunes candidate pairs while preserving all contacts
    self.assertGreater(nacon_sat, 0)
    self.assertLess(ncoll_sat, ncoll_nosat)
    self.assertEqual(nacon_sat, nacon_nosat)

    pos_sat = d_sat.contact.pos.numpy()[:nacon_sat]
    pos_nosat = d_nosat.contact.pos.numpy()[:nacon_nosat]
    dist_sat = d_sat.contact.dist.numpy()[:nacon_sat]
    dist_nosat = d_nosat.contact.dist.numpy()[:nacon_nosat]

    def _sort_key(pos, dist):
      return np.lexsort((dist, pos[:, 2], pos[:, 1], pos[:, 0]))

    idx_sat = _sort_key(np.round(pos_sat, 5), np.round(dist_sat, 5))
    idx_nosat = _sort_key(np.round(pos_nosat, 5), np.round(dist_nosat, 5))

    np.testing.assert_allclose(pos_sat[idx_sat], pos_nosat[idx_nosat], atol=1e-5)
    np.testing.assert_allclose(dist_sat[idx_sat], dist_nosat[idx_nosat], atol=1e-5)

  def test_parallel_fps_numpy_parity(self):
    """Test that parallel FPS selects candidates matching a serial NumPy reference."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cloth1" type="grid" count="8 8 1" spacing=".05 .05 .05" pos="0 0 0"
                    radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
          <flexcomp name="cloth2" type="grid" count="8 8 1" spacing=".05 .05 .05" pos="0 0 0.01"
                    radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      qpos_noise=0.005,
      nworld=1,
      nconmax=1500,
    )
    d.nacon.fill_(-1)
    d.contact.dist.fill_(wp.inf)

    mjw.kinematics(m, d)

    ws = collision_flex._allocate_flex_workspace(m, d)
    sap_data = collision_flex._run_flex_sap_sort(m, d)
    ctx = collision_core.create_collision_context(d.naconmax)
    collision_flex._flex_sap_collision(m, d, ctx, ws, is_self=False, sap_data=sap_data)

    ncand = int(ws.ncand.numpy()[0])
    cand_active = ws.cand_active.numpy()[:ncand]
    sort_val = ws.filter_val.numpy()[:ncand]
    cand_active_sorted = ws.cand_active_sorted.numpy()[:ncand]
    cand_pos = ws.pos.numpy()[:ncand]
    cand_dist = ws.dist.numpy()[:ncand]
    cand_elem = ws.elem.numpy()[:ncand]
    num_groups = int(ws.flex_num_groups.numpy()[0])
    group_starts = ws.flex_group_start_indices.numpy()[:num_groups]

    def _tie_break(curr, sel):
      if sel < 0:
        return True
      e1_c, e2_c = cand_elem[curr]
      e1_s, e2_s = cand_elem[sel]
      if e1_c != e1_s:
        return e1_c < e1_s
      if e2_c != e2_s:
        return e2_c < e2_s
      return curr < sel

    self.assertGreater(num_groups, 0)
    for g in range(num_groups):
      g_start = group_starts[g]
      g_end = group_starts[g + 1] if g + 1 < num_groups else ncand
      group_cands = [sort_val[si] for si in range(g_start, g_end) if cand_active_sorted[si] == 1]
      if len(group_cands) <= types.MJ_MAXCONPAIR:
        continue

      best_seed = -1
      min_d = 1e10
      for c_idx in group_cands:
        d_val = cand_dist[c_idx]
        if d_val < min_d:
          min_d = d_val
          best_seed = c_idx
        elif d_val == min_d and _tie_break(c_idx, best_seed):
          min_d = d_val
          best_seed = c_idx

      selected = [best_seed]
      seed_pos = cand_pos[best_seed]
      min_dist = {c_idx: np.float32(np.linalg.norm(cand_pos[c_idx] - seed_pos)) for c_idx in group_cands}

      for _ in range(1, types.MJ_MAXCONPAIR):
        max_d = np.float32(-1e10)
        best_cand = -1
        for c_idx in group_cands:
          if c_idx in selected:
            continue
          md = min_dist[c_idx]
          if md > max_d:
            max_d = md
            best_cand = c_idx
          elif md == max_d and _tie_break(c_idx, best_cand):
            max_d = md
            best_cand = c_idx

        if best_cand < 0 or max_d <= 0.0:
          break

        selected.append(best_cand)
        new_pos = cand_pos[best_cand]
        for c_idx in group_cands:
          d_new = np.float32(np.linalg.norm(cand_pos[c_idx] - new_pos))
          if d_new < min_dist[c_idx]:
            min_dist[c_idx] = d_new

      warp_selected = sorted([c_idx for c_idx in group_cands if cand_active[c_idx] == 1])
      np_selected = sorted(selected)
      self.assertEqual(warp_selected, np_selected)

  @parameterized.parameters(1, 2)
  def test_mixed_flex_broadphase_and_narrowphase(self, nworld):
    """Test that broadphase and narrowphase run correctly with mixed 2D and 3D flexes."""
    xml = """
    <mujoco>
      <worldbody>
        <!-- 2D Cloth -->
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                  radius=".02" dim="2" mass=".5">
          <contact selfcollide="none" internal="false"/>
        </flexcomp>
        <!-- 3D Softbody -->
        <flexcomp name="softbody" type="grid" count="3 3 3" spacing=".2 .2 .2" pos="1 1 0"
                  radius=".02" dim="3" mass="1.0">
          <contact selfcollide="none" internal="false"/>
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

  def test_triangle_sat_separated(self):
    """Tests 2D separating axis test on coplanar, parallel, and intersecting triangles."""

    @wp.kernel
    def eval_sat(
      p0: wp.array[wp.vec3],
      p1: wp.array[wp.vec3],
      p2: wp.array[wp.vec3],
      q0: wp.array[wp.vec3],
      q1: wp.array[wp.vec3],
      q2: wp.array[wp.vec3],
      cutoff_sq: wp.array[float],
      result: wp.array[bool],
    ):
      tid = wp.tid()
      result[tid] = collision_flex._triangle_sat_separated(
        p0[tid],
        p1[tid],
        p2[tid],
        q0[tid],
        q1[tid],
        q2[tid],
        cutoff_sq[tid],
      )

    p0_list = [
      wp.vec3(0.0, 0.0, 0.0),
      wp.vec3(0.0, 0.0, 0.0),
      wp.vec3(0.0, 0.0, 0.0),
      wp.vec3(0.0, -1.0, 0.0),
    ]
    p1_list = [
      wp.vec3(1.0, 0.0, 0.0),
      wp.vec3(1.0, 0.0, 0.0),
      wp.vec3(1.0, 0.0, 0.0),
      wp.vec3(0.0, 1.0, 0.0),
    ]
    p2_list = [
      wp.vec3(0.0, 1.0, 0.0),
      wp.vec3(0.0, 1.0, 0.0),
      wp.vec3(0.0, 1.0, 0.0),
      wp.vec3(0.0, 0.0, 1.0),
    ]

    q0_list = [
      wp.vec3(3.0, 0.0, 0.0),
      wp.vec3(0.0, 0.0, 0.5),
      wp.vec3(0.0, 0.0, 0.1),
      wp.vec3(-0.5, 0.0, 0.5),
    ]
    q1_list = [
      wp.vec3(4.0, 0.0, 0.0),
      wp.vec3(1.0, 0.0, 0.5),
      wp.vec3(1.0, 0.0, 0.1),
      wp.vec3(0.5, 0.0, 0.5),
    ]
    q2_list = [
      wp.vec3(3.0, 1.0, 0.0),
      wp.vec3(0.0, 1.0, 0.5),
      wp.vec3(0.0, 1.0, 0.1),
      wp.vec3(0.0, 0.5, 0.5),
    ]
    cutoff_sq_list = [
      0.01,
      0.04,
      0.04,
      0.0,
    ]
    expected = [True, True, False, False]

    n = len(expected)
    res = wp.zeros(n, dtype=bool)
    wp.launch(
      eval_sat,
      dim=n,
      inputs=[
        wp.array(p0_list, dtype=wp.vec3),
        wp.array(p1_list, dtype=wp.vec3),
        wp.array(p2_list, dtype=wp.vec3),
        wp.array(q0_list, dtype=wp.vec3),
        wp.array(q1_list, dtype=wp.vec3),
        wp.array(q2_list, dtype=wp.vec3),
        wp.array(cutoff_sq_list, dtype=float),
      ],
      outputs=[res],
    )
    np.testing.assert_array_equal(res.numpy(), expected)


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

  @parameterized.parameters(1, 2)
  def test_contact_flex_flex_rope_margin(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp name="rope1" type="grid" count="2 1 1" spacing="0.2 0.2 0.1" pos="0 0 0" radius="0.02" dim="1" mass="0.5">
          <contact selfcollide="none" contype="1" conaffinity="1" margin="0.01"/>
        </flexcomp>
        <flexcomp name="rope2" type="grid" count="2 1 1" spacing="0.2 0.2 0.1" pos="0 0 0.045" euler="0 0 90" radius="0.02" dim="1" mass="0.5">
          <contact selfcollide="none" contype="1" conaffinity="1" margin="0.01"/>
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

    self.assertEqual(d.nacon.numpy()[0], nworld * 1)
    self.assertEqual(mjd.ncon, 1)

    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts)

  @parameterized.parameters(1, 2)
  def test_contact_flex_flex_cloth(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp name="tri1" type="direct" dim="2" radius="0.01" mass="0.5"
                  point="0 0 0  0.1 0 0  0 0.1 0"
                  element="0 1 2">
          <contact selfcollide="none" contype="1" conaffinity="1"/>
        </flexcomp>
        <flexcomp name="tri2" type="direct" dim="2" radius="0.01" mass="0.5"
                  point="0 0 0.015  0.1 0 0.015  0 0.1 0.015"
                  element="0 1 2">
          <contact selfcollide="none" contype="1" conaffinity="1"/>
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

    self.assertEqual(d.nacon.numpy()[0], nworld * 1)
    self.assertEqual(mjd.ncon, 1)

    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts)

  @parameterized.parameters(1, 2)
  def test_contact_flex_flex_tet(self, nworld):
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp name="tet1" type="direct" dim="3" radius="0.01" mass="0.5"
                  point="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"
                  element="0 1 2 3">
          <contact selfcollide="none" contype="1" conaffinity="1"/>
        </flexcomp>
        <flexcomp name="tet2" type="direct" dim="3" radius="0.01" mass="0.5"
                  point="0 0 0.015  0.1 0 0.015  0 0.1 0.015  0 0 0.115"
                  element="0 1 2 3">
          <contact selfcollide="none" contype="1" conaffinity="1"/>
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
      self._assert_contact_parity(w_contacts, m_contacts, atol=1e-4)

  @parameterized.parameters(1, 2)
  def test_flex_flex_trilinear_collision_trajectory(self, nworld):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco model="flex_flex">
        <option solver="CG" tolerance="1e-6" timestep="0.001" integrator="Euler"/>
        <size nconmax="16000" njmax="16000"/>

        <worldbody>
          <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" condim="1"/>

          <!-- Cube 1: Resting on the plane -->
          <flexcomp name="cube1" type="grid" count="8 8 8" spacing="0.07 0.07 0.07" pos="-0.2 0 0.27"
                    radius="0.001" dim="3" mass="5.0" dof="trilinear">
            <contact selfcollide="none" internal="false"/>
            <elasticity young="1e4" damping="0.01" poisson="0.1"/>
          </flexcomp>

          <!-- Cube 2: Falling from above onto the corner of Cube 1 -->
          <flexcomp name="cube2" type="grid" count="8 8 8" spacing="0.07 0.07 0.07" pos="0.0 0 1.0"
                    radius="0.001" dim="3" mass="5.0" dof="trilinear">
            <contact selfcollide="none" internal="false"/>
            <elasticity young="1e4" damping="0.01" poisson="0.1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      nconmax=16000,
      njmax=16000,
    )

    checkpoints = [0, 250, 500, 1000]
    curr_step = 0

    for target_step in checkpoints:
      while curr_step < target_step:
        mujoco.mj_step(mjm, mjd)
        curr_step += 1

      d.qpos.assign(np.tile(mjd.qpos, (nworld, 1)).astype(np.float32))
      d.qvel.assign(np.tile(mjd.qvel, (nworld, 1)).astype(np.float32))

      d.nacon.fill_(-1)
      mjw.forward(m, d)
      mujoco.mj_forward(mjm, mjd)

      for w in range(nworld):
        np.testing.assert_allclose(
          d.flexvert_xpos.numpy()[w],
          mjd.flexvert_xpos,
          atol=1e-4,
          err_msg=f"flexvert_xpos mismatch at step {curr_step} (world {w})",
        )
        np.testing.assert_allclose(
          d.qfrc_passive.numpy()[w],
          mjd.qfrc_passive,
          atol=1e-3,
          rtol=1e-3,
          err_msg=f"qfrc_passive mismatch at step {curr_step} (world {w})",
        )

      # Geom-flex ground plane contacts parity
      for w in range(nworld):
        w_geom_contacts = [
          c for c in self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True) if c["geom"][0] >= 0
        ]
        m_geom_contacts = [c for c in self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False) if c["geom"][0] >= 0]
        if len(w_geom_contacts) == len(m_geom_contacts) and len(m_geom_contacts) > 0:
          self._assert_contact_parity(w_geom_contacts, m_geom_contacts, atol=1e-4)

      # Flex-flex contact count parity and constraint parity
      if curr_step in (0, 250, 500, 1000):
        if curr_step == 250:
          # MuJoCo caps contact pairs at mjMAXCONPAIR=50 (50 plane + 50 flex-flex = 100),
          # whereas Warp detects all 64 vertices on the ground plane (64 + 50 = 114).
          self.assertEqual(d.nacon.numpy()[0], nworld * 114, f"nacon mismatch at step {curr_step}")
          for w in range(nworld):
            self.assertEqual(d.nefc.numpy()[w], 456, f"nefc mismatch at step {curr_step} (world {w})")
        else:
          self.assertEqual(d.nacon.numpy()[0], nworld * mjd.ncon, f"nacon mismatch at step {curr_step}")
          for w in range(nworld):
            self.assertEqual(d.nefc.numpy()[w], mjd.nefc, f"nefc mismatch at step {curr_step} (world {w})")

  @parameterized.parameters(1, 2)
  def test_contact_mesh_flex_3d_parity(self, nworld):
    """Test contact parity for 3D flex soft body colliding with a mesh geom."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <asset>
          <mesh name="box_mesh" vertex="-0.05 -0.05 -0.05  0.05 -0.05 -0.05  0.05 0.05 -0.05  -0.05 0.05 -0.05
                                        -0.05 -0.05 0.05   0.05 -0.05 0.05   0.05 0.05 0.05   -0.05 0.05 0.05"/>
        </asset>
        <worldbody>
          <geom type="mesh" mesh="box_mesh" pos="0 0 0.05"/>
          <flexcomp name="tet" type="direct" dim="3" radius="0.01" mass="0.5"
                    point="0 0 -0.01  0.05 0 -0.06  0 0.05 -0.06  -0.05 0 -0.06"
                    element="0 1 2 3">
            <contact condim="3" selfcollide="none" margin="0.02"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertEqual(d.nacon.numpy()[0], nworld * mjd.ncon)
    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts, atol=1e-4)

  @parameterized.parameters(1, 2)
  def test_contact_ellipsoid_cloth_parity(self, nworld):
    """Test contact parity for 2D cloth triangle colliding with an ellipsoid."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="ellipsoid" size="0.05 0.05 0.05" pos="0 0 0.04"/>
          <flexcomp name="cloth" type="direct" dim="2" radius="0.02" mass="0.5"
                    point="-0.05 -0.05 0  0.05 -0.05 0  0 0.05 0"
                    element="0 1 2">
            <contact condim="3" selfcollide="none" margin="0.02"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertEqual(d.nacon.numpy()[0], nworld * mjd.ncon)
    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts, atol=1e-3)

  @parameterized.parameters(1, 2)
  def test_contact_flex_flex_none_selfcollide(self, nworld):
    """Test contacts between two distinct flex bodies when selfcollide is none."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cloth1" type="direct" dim="2" radius="0.01" mass="0.5"
                    point="-0.05 -0.05 0  0.05 -0.05 0  0 0.05 0"
                    element="0 1 2">
          <contact selfcollide="none" contype="1" conaffinity="1" margin="0.01"/>
        </flexcomp>
        <flexcomp name="cloth2" type="direct" dim="2" radius="0.01" mass="0.5"
                    point="-0.05 -0.05 0.015  0.05 -0.05 0.015  0 0.05 0.015"
                    element="0 1 2">
          <contact selfcollide="none" contype="1" conaffinity="1" margin="0.01"/>
        </flexcomp>
      </worldbody>
    </mujoco>
      """,
      nworld=nworld,
    )
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertEqual(d.nacon.numpy()[0], nworld * mjd.ncon)
    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts, atol=1e-4)

  @parameterized.parameters(1, 2)
  def test_contact_cloth_pinned_to_geom_body_parity(self, nworld):
    """Test that flex elements on the same body as a geom do not collide with that geom."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="carrier" pos="0 0 0">
            <geom name="carrier_box" type="box" size="0.05 0.05 0.05"/>
            <flexcomp name="cloth" type="direct" dim="2" radius="0.01" mass="0.5"
                      point="0 0 0  0.05 0 0  0 0.05 0"
                      element="0 1 2">
              <pin id="0"/>
              <contact condim="3" selfcollide="none" margin="0.02"/>
            </flexcomp>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_flex(mjm, mjd)
    mujoco.mj_collision(mjm, mjd)

    self.assertEqual(d.nacon.numpy()[0], nworld * mjd.ncon)
    for w in range(nworld):
      w_contacts = self._get_sorted_contacts(d, d.nacon.numpy()[0], world_idx=w, is_warp=True)
      m_contacts = self._get_sorted_contacts(mjd, mjd.ncon, is_warp=False)
      self._assert_contact_parity(w_contacts, m_contacts, atol=1e-4)

  def test_3d_flex_interior_layer_culling_in_sap(self):
    """Test that inactive interior elements are projected to MJ_MAXVAL in SAP."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cube" type="grid" dim="3" count="4 4 4" spacing="0.05 0.05 0.05"
                    radius="0.01" mass="1.0">
            <contact condim="3" selfcollide="none" margin="0.02" activelayers="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=1,
    )
    mjw.kinematics(m, d)
    mjw.flex(m, d)

    nelem = m.nflexelem
    sap_lower = wp.empty((d.nworld, nelem), dtype=float)
    sap_upper = wp.empty((d.nworld, nelem), dtype=float)
    sap_sort_index = wp.empty((d.nworld, nelem, 2), dtype=int)
    elem_aabb_lower = wp.empty((d.nworld, nelem), dtype=wp.vec3)
    elem_aabb_upper = wp.empty((d.nworld, nelem), dtype=wp.vec3)
    sap_seg_index = wp.empty(d.nworld + 1, dtype=int)

    wp.launch(
      collision_flex._flex_sap_project,
      dim=(d.nworld, nelem),
      inputs=[
        m.flex_margin,
        m.flex_gap,
        m.flex_activelayers,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_elemadr,
        m.flex_elemdataadr,
        m.flex_elem,
        m.flex_elemlayer,
        m.flex_radius,
        m.flex_elemflexid,
        d.flexvert_xpos,
        d.nworld,
        nelem,
        wp.vec3(1.0, 0.0, 0.0),
      ],
      outputs=[
        sap_lower.reshape((-1, nelem)),
        sap_upper,
        sap_sort_index.reshape((-1, nelem)),
        elem_aabb_lower,
        elem_aabb_upper,
        sap_seg_index,
      ],
    )

    layers = m.flex_elemlayer.numpy()
    proj_lower = sap_lower.numpy()[0]
    for e in range(nelem):
      if layers[e] >= 1:
        self.assertEqual(proj_lower[e], types.MJ_MAXVAL, f"Element {e} (layer {layers[e]}) should be culled in SAP")
      else:
        self.assertLess(proj_lower[e], types.MJ_MAXVAL, f"Element {e} (layer {layers[e]}) should be active in SAP")

  def test_flex_num_groups_zero_on_empty_pass(self):
    """Test that flex_num_groups is reset to 0 when no candidate contacts exist."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <flexcomp name="cloth1" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 0.05"
                    radius="0.01" mass="0.5">
            <contact condim="3" selfcollide="none" margin="0.02"/>
          </flexcomp>
          <flexcomp name="cloth2" type="grid" dim="2" count="2 2 1" spacing="0.05 0.05 0.05"
                    radius="0.01" mass="0.5" pos="10.0 0 0">
            <contact condim="3" selfcollide="none" margin="0.02"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      nworld=1,
    )
    mjw.kinematics(m, d)
    mjw.flex(m, d)

    ws = collision_flex._allocate_flex_workspace(m, d)
    ws.flex_num_groups.fill_(42)
    self.assertEqual(ws.flex_num_groups.numpy()[0], 42)

    collision_flex._flex_geom_collision(m, d, ws)

    self.assertEqual(ws.flex_num_groups.numpy()[0], 0)


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

    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc_flex_3d_simplex", m.nv, nworld, tol=1e-2)

  @parameterized.parameters(1, 2)
  def test_mesh_flex_3d_collision(self, nworld):
    """Test contacts for 3D flex soft body colliding with a mesh geom via CCD."""
    xml = """
    <mujoco>
      <asset>
        <mesh name="box_mesh" vertex="-0.05 -0.05 -0.05  0.05 -0.05 -0.05  0.05 0.05 -0.05  -0.05 0.05 -0.05
                                      -0.05 -0.05 0.05   0.05 -0.05 0.05   0.05 0.05 0.05   -0.05 0.05 0.05"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="box_mesh" pos="-0.055 -0.075 0.1"/>
        <flexcomp name="softbody" type="grid" count="2 2 2" spacing=".15 .15 .15" pos="-.075 -.075 0"
                  radius=".01" dim="3" mass=".5">
          <contact condim="3" selfcollide="none"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    self.assertGreater(d.nacon.numpy()[0], 0, "Expected contacts in Warp for 3D flex vs mesh")

  @parameterized.parameters(1, 2)
  def test_ellipsoid_flex_3d_collision(self, nworld):
    """Test contacts for 3D flex soft body colliding with an ellipsoid geom via CCD."""
    xml = """
    <mujoco>
      <worldbody>
        <geom type="ellipsoid" size="0.05 0.05 0.05" pos="-0.055 -0.075 0.1"/>
        <flexcomp name="softbody" type="grid" count="2 2 2" spacing=".15 .15 .15" pos="-.075 -.075 0"
                  radius=".01" dim="3" mass=".5">
          <contact condim="3" selfcollide="none"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)
    d.nacon.fill_(-1)
    mjw.kinematics(m, d)
    mjw.flex(m, d)
    mjw.collision(m, d)

    self.assertGreater(d.nacon.numpy()[0], 0, "Expected contacts in Warp for 3D flex vs ellipsoid")

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


class FlexFlexCollisionTest(parameterized.TestCase):
  """Tests for flex-flex collisions."""

  @parameterized.product(
    test_case=[
      (
        "1d_1d",
        """
        <mujoco>
          <worldbody>
            <!-- Two line flexes overlapping at origin -->
            <flexcomp name="rope1" type="grid" count="2 1 1" spacing=".2 .2 .1" pos="0 -0.01 0"
                      radius=".02" dim="1" mass=".5">
              <contact selfcollide="none" contype="1" conaffinity="1"/>
            </flexcomp>
            <flexcomp name="rope2" type="grid" count="2 1 1" spacing=".2 .2 .1" pos="0 0.01 0"
                      radius=".02" dim="1" mass=".5">
              <contact selfcollide="none" contype="1" conaffinity="1"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        True,
      ),
      (
        "2d_2d",
        """
        <mujoco>
          <worldbody>
            <!-- Two cloth grids placed one slightly above the other, overlapping in X/Y -->
            <flexcomp name="cloth1" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                      radius=".02" dim="2" mass=".5">
              <contact selfcollide="none" contype="1" conaffinity="1"/>
            </flexcomp>
            <flexcomp name="cloth2" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0.1 0.1 0.02"
                      radius=".02" dim="2" mass=".5">
              <contact selfcollide="none" contype="1" conaffinity="1"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        True,
      ),
      (
        "1d_2d",
        """
        <mujoco>
          <worldbody>
            <!-- Cloth at origin -->
            <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                      radius=".02" dim="2" mass=".5">
              <contact selfcollide="none" contype="1" conaffinity="1"/>
            </flexcomp>
            <!-- Rope passing through the cloth -->
            <flexcomp name="rope" type="grid" count="2 1 1" spacing=".2 .2 .1" pos="0 0 0.01"
                      radius=".02" dim="1" mass=".5">
              <contact selfcollide="none" contype="1" conaffinity="1"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        True,
      ),
      (
        "3d_3d",
        """
        <mujoco>
          <worldbody>
            <flexcomp name="cube1" type="grid" count="3 3 3" spacing="0.05 0.05 0.05" pos="0 0 0.1" dim="3" mass="1" radius="0.01">
              <contact selfcollide="none" contype="1" conaffinity="1" condim="3"/>
            </flexcomp>
            <flexcomp name="cube2" type="grid" count="3 3 3" spacing="0.05 0.05 0.05" pos="0.08 0.08 0.18" dim="3" mass="1" radius="0.01">
              <contact selfcollide="none" contype="1" conaffinity="1" condim="3"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        True,
      ),
      (
        "bitmask_filtering",
        """
        <mujoco>
          <worldbody>
            <!-- Two cloths that would overlap, but contype/conaffinity do not match -->
            <flexcomp name="cloth1" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                      radius=".02" dim="2" mass=".5">
              <contact selfcollide="none" contype="1" conaffinity="2"/>
            </flexcomp>
            <flexcomp name="cloth2" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0.02"
                      radius=".02" dim="2" mass=".5">
              <contact selfcollide="none" contype="4" conaffinity="8"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        False,
      ),
    ],
    nworld=[1, 2],
  )
  def test_flex_flex_collisions(self, test_case, nworld):
    name, xml, expect_contacts = test_case
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    self.assertEqual(m.nflex, 2)

    mjw.kinematics(m, d)
    if m.max_flex_dim == 3:
      mjw.flex(m, d)
    mjw.collision(m, d)

    nacon = int(d.nacon.numpy()[0])
    if expect_contacts:
      self.assertGreater(nacon, 0, f"[{name}] Expected flex-flex contacts")
    else:
      self.assertEqual(nacon, 0, f"[{name}] Expected 0 contacts due to bitmask filtering")

  @parameterized.parameters(1, 2)
  def test_flex_flex_collision_shared_body_filtering(self, nworld):
    """Test that flex-flex collisions exclude elements if vertices share a body."""
    xml = """
    <mujoco>
      <worldbody>
        <flexcomp name="cloth1" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0"
                  radius=".02" dim="2" mass=".5">
          <contact selfcollide="none" contype="1" conaffinity="1"/>
        </flexcomp>
        <flexcomp name="cloth2" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0.02"
                  radius=".02" dim="2" mass=".5">
          <contact selfcollide="none" contype="1" conaffinity="1"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=xml, nworld=nworld)

    # First verify we get contacts normally
    mjw.kinematics(m, d)
    mjw.collision(m, d)
    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected baseline contacts before weld")

    # Now weld all vertices of cloth1 and cloth2 to the same body ID (e.g. 1)
    # So they are treated as sharing bodies.
    vertbody = m.flex_vertbodyid.numpy()
    vertbody[:] = 1
    m.flex_vertbodyid.assign(vertbody)

    # Run collision again
    mjw.collision(m, d)
    nacon = int(d.nacon.numpy()[0])
    self.assertEqual(nacon, 0, "Expected 0 contacts due to shared body exclusion")


class FlexContactNnzTest(parameterized.TestCase):
  """Tests for contact Jacobian non-zero (NNZ) bounds across flex types."""

  @parameterized.named_parameters(
    (
      "rope_plane",
      """
      <mujoco>
        <worldbody>
          <geom type="plane" size="1 1 .01"/>
          <flexcomp name="rope" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 .1" radius=".02" dim="1" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      6,
    ),
    (
      "rope_rope",
      """
      <mujoco>
        <worldbody>
          <flexcomp name="rope1" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0" radius=".02" dim="1" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
          <flexcomp name="rope2" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 .1" radius=".02" dim="1" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      12,
    ),
    (
      "rope_self",
      """
      <mujoco>
        <worldbody>
          <flexcomp name="rope" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0" radius=".02" dim="1" mass=".5">
            <contact selfcollide="auto" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      12,
    ),
    (
      "cloth_plane",
      """
      <mujoco>
        <worldbody>
          <geom type="plane" size="1 1 .01"/>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 .1" radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      9,
    ),
    (
      "cloth_cloth",
      """
      <mujoco>
        <worldbody>
          <flexcomp name="cloth1" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0" radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
          <flexcomp name="cloth2" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 .1" radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      18,
    ),
    (
      "cloth_self",
      """
      <mujoco>
        <worldbody>
          <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0" radius=".02" dim="2" mass=".5">
            <contact selfcollide="auto" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      18,
    ),
    (
      "tet_plane",
      """
      <mujoco>
        <worldbody>
          <geom type="plane" size="1 1 .01"/>
          <flexcomp name="solid" type="grid" count="2 2 2" spacing=".1 .1 .1" pos="0 0 .1" radius=".02" dim="3" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      12,
    ),
    (
      "tet_tet",
      """
      <mujoco>
        <worldbody>
          <flexcomp name="solid1" type="grid" count="2 2 2" spacing=".1 .1 .1" pos="0 0 0" radius=".02" dim="3" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
          <flexcomp name="solid2" type="grid" count="2 2 2" spacing=".1 .1 .1" pos="0 0 .1" radius=".02" dim="3" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      24,
    ),
    (
      "trilinear_3d_solid_plane",
      """
      <mujoco>
        <worldbody>
          <geom type="plane" size="1 1 .01"/>
          <flexcomp name="cube" type="grid" count="3 3 3" spacing=".1 .1 .1" pos="0 0 .1" radius=".02" dim="3" mass=".5" dof="trilinear">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      24,
    ),
    (
      "trilinear_3d_solid_solid",
      """
      <mujoco>
        <worldbody>
          <flexcomp name="cube1" type="grid" count="3 3 3" spacing=".1 .1 .1" pos="0 0 0" radius=".02" dim="3" mass=".5" dof="trilinear">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
          <flexcomp name="cube2" type="grid" count="3 3 3" spacing=".1 .1 .1" pos="0 0 .1" radius=".02" dim="3" mass=".5" dof="trilinear">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      48,
    ),
    (
      "mixed_rope_trilinear",
      """
      <mujoco>
        <worldbody>
          <flexcomp name="rope" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0" radius=".02" dim="1" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
          <flexcomp name="cube" type="grid" count="3 3 3" spacing=".1 .1 .1" pos="0 0 .1" radius=".02" dim="3" mass=".5" dof="trilinear">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      30,
    ),
    (
      "kinematic_tree_rope",
      """
      <mujoco>
        <worldbody>
          <body name="arm" pos="0 0 0">
            <joint name="hinge" type="hinge" axis="0 0 1"/>
            <geom type="sphere" size=".05"/>
            <flexcomp name="rope" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0" radius=".02" dim="1" mass=".5">
              <contact selfcollide="none" contype="1" conaffinity="1"/>
            </flexcomp>
          </body>
          <geom type="plane" size="1 1 .01"/>
        </worldbody>
      </mujoco>
      """,
      8,
    ),
  )
  def test_flex_contact_nnz_static_estimates(self, xml: str, expected_nnz: int):
    """Verifies that _calculate_max_contact_nnz produces expected upper bounds."""
    mjm = mujoco.MjModel.from_xml_string(xml)
    max_nnz = io._calculate_max_contact_nnz(mjm)
    self.assertEqual(max_nnz, expected_nnz)

  @parameterized.named_parameters(
    (
      "rope_collision",
      """
      <mujoco>
        <option jacobian="sparse"/>
        <worldbody>
          <flexcomp name="rope1" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0" radius=".02" dim="1" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
          <flexcomp name="rope2" type="grid" count="4 1 1" spacing=".2 .2 .1" pos="0 0 0.02" radius=".02" dim="1" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
    ),
    (
      "cloth_collision",
      """
      <mujoco>
        <option jacobian="sparse"/>
        <worldbody>
          <flexcomp name="cloth1" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0" radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
          <flexcomp name="cloth2" type="grid" count="3 3 1" spacing=".2 .2 .1" pos="0 0 0.02" radius=".02" dim="2" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
    ),
    (
      "tet_collision",
      """
      <mujoco>
        <option jacobian="sparse"/>
        <worldbody>
          <flexcomp name="solid1" type="grid" count="2 2 2" spacing=".1 .1 .1" pos="0 0 0" radius=".02" dim="3" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
          <flexcomp name="solid2" type="grid" count="2 2 2" spacing=".1 .1 .1" pos="0 0 0.02" radius=".02" dim="3" mass=".5">
            <contact selfcollide="none" contype="1" conaffinity="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
    ),
  )
  def test_flex_contact_nnz_runtime_bound(self, xml: str):
    """Verifies that actual contact Jacobian row NNZ never exceeds the estimated upper bound."""
    mjm, _, m, d = test_data.fixture(xml=xml)
    estimated_max_nnz = io._calculate_max_contact_nnz(mjm)

    mjw.kinematics(m, d)
    if m.max_flex_dim == 3:
      mjw.flex(m, d)
    mjw.collision(m, d)
    mjw.make_constraint(m, d)

    nacon = int(d.nacon.numpy()[0])
    self.assertGreater(nacon, 0, "Expected active contacts")

    nefc = int(d.nefc.numpy()[0])
    efc_types = d.efc.type.numpy()[0, :nefc]
    efc_rownnz = d.efc.J_rownnz.numpy()[0, :nefc]

    contact_types = (
      types.ConstraintType.CONTACT_FRICTIONLESS,
      types.ConstraintType.CONTACT_PYRAMIDAL,
      types.ConstraintType.CONTACT_ELLIPTIC,
    )
    for idx in range(nefc):
      if efc_types[idx] in contact_types:
        actual_nnz = int(efc_rownnz[idx])
        self.assertGreater(actual_nnz, 0)
        self.assertLessEqual(
          actual_nnz,
          estimated_max_nnz,
          f"Contact row {idx} actual NNZ ({actual_nnz}) exceeded static estimate ({estimated_max_nnz})",
        )


if __name__ == "__main__":
  wp.init()
  absltest.main()
