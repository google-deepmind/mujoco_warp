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

"""Tests for constraint functions."""

import itertools
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import ConeType
from mujoco_warp import test_data
from mujoco_warp._src import constraint
from mujoco_warp._src import util_pkg

# tolerance for difference between MuJoCo and MJWarp constraint calculations,
# mostly due to float precision
_TOLERANCE = 5e-4


def _assert_eq(a, b, name, tol=_TOLERANCE):
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def _assert_efc_eq(mjm, m, d, mjd, nefc, name, nv, tol=_TOLERANCE):
  """Assert equality of efc fields after sorting both sides."""
  # Get the ordering indices based on efc fields for MJWarp
  efc_type = d.efc.type.numpy()[0, :nefc]
  efc_id = d.efc.id.numpy()[0, :nefc]
  efc_pos = d.efc.pos.numpy()[0, :nefc]
  efc_vel = d.efc.vel.numpy()[0, :nefc]
  efc_aref = d.efc.aref.numpy()[0, :nefc]
  efc_d = d.efc.D.numpy()[0, :nefc]
  # Get the ordering indices based on efc fields for MuJoCo
  mjd_efc_type = mjd.efc_type[:nefc]
  mjd_efc_id = mjd.efc_id[:nefc]
  mjd_efc_pos = mjd.efc_pos[:nefc]
  mjd_efc_vel = mjd.efc_vel[:nefc]
  mjd_efc_aref = mjd.efc_aref[:nefc]
  mjd_efc_d = mjd.efc_D[:nefc]

  # Create sorting keys using lexsort
  d_sort_indices = np.lexsort((efc_pos, efc_vel, efc_aref, efc_d, efc_id, efc_type))
  mjd_sort_indices = np.lexsort((mjd_efc_pos, mjd_efc_vel, mjd_efc_aref, mjd_efc_d, mjd_efc_id, mjd_efc_type))

  # convert sparse to dense if necessary
  if m.is_sparse:
    efc_J = np.zeros((nefc, nv))
    mujoco.mju_sparse2dense(
      efc_J,
      d.efc.J.numpy()[0, 0],
      d.efc.J_rownnz.numpy()[0, :nefc],
      d.efc.J_rowadr.numpy()[0, :nefc],
      d.efc.J_colind.numpy()[0, 0],
    )
  else:
    efc_J = d.efc.J.numpy()[0]

  # Sort MJWarp efc fields
  d_sorted = efc_J[d_sort_indices, :nv].reshape(-1)

  # Sort MuJoCo efc fields
  # For J matrix, need to reshape to 2D, sort rows, then flatten
  nefc = len(mjd_sort_indices)

  if mujoco.mj_isSparse(mjm):
    mj_efc_J = np.zeros((mjd.nefc, mjm.nv))
    mujoco.mju_sparse2dense(mj_efc_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
  else:
    # efc_J can be preallocated for rows that are later dropped (empty Jacobian)
    mj_efc_J = mjd.efc_J[: mjd.nefc * mjm.nv].reshape((mjd.nefc, mjm.nv))

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

  # Compare sorted data
  _assert_eq(d_sorted, mjd_sorted_J, f"{name}_J", tol=tol)

  d_sorted = d.efc.D.numpy()[0, d_sort_indices]
  _assert_eq(d_sorted, mjd_sorted_D, f"{name}_D", tol=tol)

  d_sorted = d.efc.vel.numpy()[0, d_sort_indices]
  _assert_eq(d_sorted, mjd_sorted_vel, f"{name}_vel", tol=tol)

  d_sorted = d.efc.aref.numpy()[0, d_sort_indices]
  _assert_eq(d_sorted, mjd_sorted_aref, f"{name}_aref", tol=tol * 2)

  d_sorted = d.efc.pos.numpy()[0, d_sort_indices]
  _assert_eq(d_sorted, mjd_sorted_pos, f"{name}_pos", tol=tol)

  d_sorted = d.efc.margin.numpy()[0, d_sort_indices]
  _assert_eq(d_sorted, mjd_sorted_margin, f"{name}_margin", tol=tol)

  d_sorted = d.efc.type.numpy()[0, d_sort_indices]
  _assert_eq(d_sorted, mjd_sorted_type, f"{name}_type")


class ConstraintTest(parameterized.TestCase):
  @parameterized.parameters(
    *itertools.product(
      (ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
      itertools.combinations_with_replacement((1, 3, 4, 6), 2),
      (mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
    )
  )
  def test_condim(self, cone, condims, jacobian):
    """Test condim."""
    condim1, condim2 = condims
    xml = f"""
      <mujoco>
        <worldbody>
          <body>
            <geom type="sphere" size=".1" condim="{condim1}"/>
            <freejoint/>
          </body>
          <body>
            <geom type="sphere" size=".1" condim="{condim2}"/>
          </body>
          <body>
            <geom type="ellipsoid" size=".1 .1 .1" condim="{condim2}"/>
          </body>
        </worldbody>
        <keyframe>
          <key qpos=".10 .11 .12 .7071 .7071 0 0" />
        </keyframe>
      </mujoco>
    """

    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0, overrides={"opt.cone": cone, "opt.jacobian": jacobian})

    # fill with nan to check whether we are not reading uninitialized values
    for arr in (d.efc.J, d.efc.D, d.efc.aref, d.efc.pos, d.efc.margin):
      arr.fill_(wp.nan)

    mjw.make_constraint(m, d)

    _assert_eq(d.nacon.numpy()[0], mjd.ncon, "nacon")
    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc", m.nv)

  @parameterized.product(
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    nworld=(1, 2),
    block_dim=(64, 128),
  )
  def test_sparse_contact_jac_shared_ancestor(self, cone, nworld, block_dim):
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option jacobian="sparse"/>
        <worldbody>
          <body pos="0 0 0.5">
            <freejoint/>
            <geom size="0.02" contype="0" conaffinity="0"/>
            <body pos="0 0.04 0">
              <joint axis="1 0 0"/>
              <joint axis="0 1 0"/>
              <joint/>
              <geom size="0.02" contype="0" conaffinity="0"/>
              <body pos="0 0 -0.15">
                <joint axis="1 0 0"/>
                <joint axis="0 1 0"/>
                <geom size="0.02" contype="0" conaffinity="0"/>
                <body pos="0 -0.02 -0.15">
                  <geom size="0.06" condim="6" friction="0.7 0.02 0.005"/>
                </body>
              </body>
            </body>
            <body pos="0 -0.04 0">
              <joint axis="1 0 0"/>
              <joint axis="0 1 0"/>
              <joint/>
              <geom size="0.02" contype="0" conaffinity="0"/>
              <body pos="0 0 -0.15">
                <joint axis="1 0 0"/>
                <joint axis="0 1 0"/>
                <geom pos="0 0.02 -0.15" size="0.06" condim="6" friction="0.7 0.02 0.005"/>
              </body>
            </body>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
      nworld=nworld,
      qvel_noise=0.5,
    )
    m.block_dim.contact_jac_sparse = block_dim
    self.assertTrue(m.is_sparse)
    self.assertEqual(mjd.ncon, 1)

    mjds = [mjd]
    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      qpos = d.qpos.numpy()
      qvel = d.qvel.numpy()
      qpos[1, 7:] += 0.05
      qvel[1] *= -0.7
      d.qpos.assign(qpos)
      d.qvel.assign(qvel)
      mjd1.qpos[:] = qpos[1]
      mjd1.qvel[:] = qvel[1]
      mujoco.mj_forward(mjm, mjd1)
      self.assertEqual(mjd1.ncon, 1)
      mjds.append(mjd1)
      mjw.fwd_kinematics(m, d)
      mjw.collision(m, d)

    for arr in (d.efc.J, d.efc.Jqvel, d.efc.vel):
      arr.fill_(wp.inf)
    for arr in (d.efc.J_colind, d.efc.J_rownnz, d.efc.J_rowadr):
      arr.fill_(-1)
    mjw.make_constraint(m, d)

    nefc = mjd.nefc
    np.testing.assert_array_equal(d.nefc.numpy(), nefc)
    efc_J = d.efc.J.numpy()
    efc_rownnz = d.efc.J_rownnz.numpy()
    efc_rowadr = d.efc.J_rowadr.numpy()
    efc_colind = d.efc.J_colind.numpy()
    efc_vel = d.efc.vel.numpy()

    J_dense = np.zeros((nworld, nefc, mjm.nv))
    for w in range(nworld):
      np.testing.assert_array_equal(efc_rownnz[w, :nefc], 10)
      mujoco.mju_sparse2dense(
        J_dense[w],
        efc_J[w, 0].astype(np.float64),
        efc_rownnz[w, :nefc],
        efc_rowadr[w, :nefc],
        efc_colind[w, 0],
      )
      self.assertEqual(mjds[w].nefc, nefc)
      mj_J = np.zeros((nefc, mjm.nv))
      mujoco.mju_sparse2dense(mj_J, mjds[w].efc_J, mjds[w].efc_J_rownnz, mjds[w].efc_J_rowadr, mjds[w].efc_J_colind)
      np.testing.assert_allclose(J_dense[w], mj_J, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(efc_vel[w, :nefc], mjds[w].efc_vel[:nefc], rtol=1e-5, atol=1e-5)

    if nworld == 2:
      self.assertFalse(np.allclose(J_dense[0], J_dense[1]))
      self.assertFalse(np.allclose(efc_vel[0, :nefc], efc_vel[1, :nefc]))

  @parameterized.product(
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    nworld=(1, 2),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_contact_row_stride(self, cone, nworld, jacobian):
    """Active contact rows match MuJoCo across launch widths and heterogeneous worlds."""
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option gravity="0 0 0"/>
        <default>
          <geom size="0.1" condim="6" priority="1" adhesion="12"/>
        </default>
        <worldbody>
          <geom type="plane" size="3 3 0.1" priority="0" adhesion="0"/>
          <body pos="0 0 0.095">
            <joint type="free" frictionloss="0.01"/>
            <geom condim="1"/>
          </body>
          <body pos="0.3 0 0.095">
            <freejoint/>
            <geom condim="3"/>
          </body>
          <body pos="0.6 0 0.095">
            <freejoint/>
            <geom condim="4"/>
          </body>
          <body pos="0.9 0 0.095">
            <freejoint/>
            <geom/>
          </body>
          <body pos="1.2 0 0.095">
            <freejoint/>
            <geom adhesion="0"/>
          </body>
          <body pos="1.5 0 0.095">
            <freejoint/>
            <geom/>
          </body>
          <body pos="1.8 0 0.095">
            <freejoint/>
            <geom/>
          </body>
          <body pos="2.1 0 0.095">
            <freejoint/>
            <geom/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      njmax=128,
      qvel_noise=0.1,
      overrides={"opt.cone": cone, "opt.jacobian": jacobian},
    )
    qpos = np.tile(mjm.qpos0, (nworld, 1))
    qvel = d.qvel.numpy()
    if nworld == 2:
      qpos[1, 0::7] += 0.02
      qpos[1, 2::7] -= 0.001
      qpos[1].reshape(-1, 7)[-2:, 2] += 0.2  # the last two spheres have no contact in world 1
      qpos[1, 3:7] = [0.9238795, 0, 0.3826834, 0]
      qvel[1] *= -0.5
    d.qpos.assign(qpos)
    d.qvel.assign(qvel)
    mjds = []
    for world in range(nworld):
      reference = mujoco.MjData(mjm)
      reference.qpos[:] = qpos[world]
      reference.qvel[:] = qvel[world]
      mujoco.mj_forward(mjm, reference)
      mjds.append(reference)
    self.assertGreater(mjds[0].nefc - mjds[0].nf, 32)
    self.assertGreater(mjds[0].nf, 0)
    if nworld == 2:
      self.assertNotEqual(mjds[0].nefc, mjds[1].nefc)

    mjw.fwd_kinematics(m, d)
    mjw.collision(m, d)
    for width in (32, d.njmax):
      for arr in (d.ne, d.nf, d.nl, d.nefc, d.efc.type, d.efc.id, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind):
        arr.fill_(-1)
      for arr in (d.efc.J, d.efc.Jqvel, d.efc.D, d.efc.vel, d.efc.aref, d.efc.pos, d.efc.margin, d.efc.frictionloss):
        arr.fill_(wp.inf)
      d.contact.efc_address.fill_(-1)

      with mock.patch.object(constraint, "efc_threads_per_world", return_value=width):
        mjw.make_constraint(m, d)

      if nworld == 2:
        common_rows = min(reference.nefc for reference in mjds)
        aref = d.efc.aref.numpy()
        self.assertFalse(np.allclose(aref[0, :common_rows], aref[1, :common_rows]))
      contact_geom = d.contact.geom.numpy()
      contact_efc_address = d.contact.efc_address.numpy()
      efc_J = d.efc.J.numpy()
      for world, reference in enumerate(mjds):
        nefc = reference.nefc
        self.assertEqual(d.ne.numpy()[world], reference.ne)
        self.assertEqual(d.nf.numpy()[world], reference.nf)
        self.assertEqual(d.nl.numpy()[world], reference.nl)
        self.assertEqual(d.nefc.numpy()[world], reference.nefc)
        order = np.arange(nefc)
        ids = d.efc.id.numpy()[world, :nefc].copy()
        contact_ids = {tuple(con.geom): i for i, con in enumerate(reference.contact)}
        start = reference.ne + reference.nf + reference.nl
        for row in range(start, nefc):
          conid = ids[row]
          self.assertGreaterEqual(conid, 0)
          ids[row] = contact_ids[tuple(contact_geom[conid])]
          dimid = row - contact_efc_address[conid, 0]
          order[row] = reference.contact[ids[row]].efc_address + dimid

        if m.is_sparse:
          rownnz = d.efc.J_rownnz.numpy()[world, :nefc]
          rowadr = d.efc.J_rowadr.numpy()[world, :nefc]
          colind = d.efc.J_colind.numpy()[world, 0]
          self.assertTrue(np.all(rownnz > 0))
          self.assertTrue(np.all((rowadr >= 0) & (rowadr + rownnz <= d.njmax_nnz)))
          for adr, nnz in zip(rowadr, rownnz):
            self.assertTrue(np.all((colind[adr : adr + nnz] >= 0) & (colind[adr : adr + nnz] < mjm.nv)))
          actual_j = np.zeros((nefc, mjm.nv))
          reference_j = np.zeros_like(actual_j)
          mujoco.mju_sparse2dense(actual_j, efc_J[world, 0], rownnz, rowadr, colind)
          mujoco.mju_sparse2dense(
            reference_j, reference.efc_J, reference.efc_J_rownnz, reference.efc_J_rowadr, reference.efc_J_colind
          )
        else:
          actual_j = efc_J[world, :nefc, : mjm.nv]
          reference_j = reference.efc_J.reshape(nefc, mjm.nv)

        _assert_eq(actual_j, reference_j[order], "J")
        np.testing.assert_array_equal(ids, reference.efc_id[order])
        np.testing.assert_array_equal(d.efc.type.numpy()[world, :nefc], reference.efc_type[order])
        for actual, expected in (
          (d.efc.D, reference.efc_D),
          (d.efc.vel, reference.efc_vel),
          (d.efc.aref, reference.efc_aref),
          (d.efc.pos, reference.efc_pos),
          (d.efc.margin, reference.efc_margin),
          (d.efc.frictionloss, reference.efc_frictionloss),
        ):
          _assert_eq(actual.numpy()[world, :nefc], expected[order], "efc")
        _assert_eq(d.efc.Jqvel.numpy()[world, start:nefc], reference.efc_vel[order[start:]], "Jqvel")

  @parameterized.parameters(
    *itertools.product(
      (1, 2),  # nworld
      (3, 4, 6),  # condim
      (0.5, 20.0),  # impratio
      (False, True),  # hinged
      (False, True),  # with_flex
    )
  )
  def test_elliptic_regularization_floor(self, nworld, condim, impratio, hinged, with_flex):
    joint = '<joint type="hinge" axis="0 1 0"/>' if hinged else "<freejoint/>"
    flex = '<flexcomp name="unused" type="grid" count="2 2 2" spacing=".1 .1 .1" pos="0 0 5" dim="3"/>' if with_flex else ""

    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
        <mujoco>
          <option cone="elliptic" impratio="{impratio}"/>
          <worldbody>
            <geom name="ground" type="plane" size="1 1 .1"/>
            <body pos="0 0 .09">
              {joint}
              <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              <geom name="ball" type="sphere" pos=".1 0 0" size=".1"/>
            </body>
            {flex}
          </worldbody>
          <contact>
            <pair geom1="ground" geom2="ball" condim="{condim}" friction=".7 .3 .02 .004 .006"/>
          </contact>
        </mujoco>
      """,
      nworld=nworld,
    )
    self.assertEqual(mjm.body_invweight0[1, 0] == 0, hinged)
    self.assertEqual(mjd.ncon, 1)

    mjds = [mjd]
    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      qpos = d.qpos.numpy()
      qpos[1, 0] += 0.05
      d.qpos.assign(qpos)
      mjd1.qpos[:] = qpos[1]
      mujoco.mj_forward(mjm, mjd1)
      self.assertEqual(mjd1.ncon, 1)
      mjds.append(mjd1)

      # Refresh per-world contact data after changing the second world's state.
      mjw.fwd_kinematics(m, d)
      mjw.collision(m, d)

    d.efc.D.fill_(wp.inf)
    mjw.make_constraint(m, d)

    nacon = d.nacon.numpy()[0]
    worldid = d.contact.worldid.numpy()[:nacon]
    address = d.contact.efc_address.numpy()
    for w in range(nworld):
      conids = np.flatnonzero(worldid == w)
      self.assertLen(conids, 1)
      actual_address = address[conids[0], :condim]
      self.assertTrue(np.all(actual_address >= 0))
      native_address = mjds[w].contact[0].efc_address
      expected = mjds[w].efc_D[native_address : native_address + condim]
      actual = d.efc.D.numpy()[w, actual_address]
      np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=0)

  @parameterized.parameters(
    *itertools.product(
      ("constraints.xml", "flex/floppy.xml", "flex/moving_base_strain.xml"),
      (mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtCone.mjCONE_ELLIPTIC),
      (mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
    )
  )
  def test_constraints(self, xml, cone, jacobian):
    """Test constraints."""
    if xml == "flex/floppy.xml" and jacobian == mujoco.mjtJacobian.mjJAC_DENSE:
      self.skipTest("flex/floppy.xml with dense jacobian not supported")
    for key in range(3):
      mjm, mjd, m, d = test_data.fixture(xml, keyframe=key, overrides={"opt.cone": cone, "opt.jacobian": jacobian})
      for arr in (d.ne, d.nefc, d.nf, d.nl, d.efc.type):
        arr.fill_(-1)
      for arr in (d.efc.J, d.efc.D, d.efc.vel, d.efc.aref, d.efc.pos, d.efc.margin):
        arr.fill_(wp.inf)

      # compute flexnode_xpos from body kinematics before generating constraints
      if mjm.nflex > 0:
        mjw.fwd_position(m, d)
      mjw.make_constraint(m, d)

      _assert_eq(d.ne.numpy()[0], mjd.ne, "ne")
      _assert_eq(d.nefc.numpy()[0], mjd.nefc, "nefc")
      _assert_eq(d.nf.numpy()[0], mjd.nf, "nf")
      _assert_eq(d.nl.numpy()[0], mjd.nl, "nl")
      _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc", m.nv)

  @parameterized.parameters(
    mujoco.mjtJacobian.mjJAC_DENSE,
    mujoco.mjtJacobian.mjJAC_SPARSE,
  )
  def test_limit_tendon(self, jacobian):
    """Test limit tendon constraints."""
    for keyframe in range(-1, 1):
      mjm, mjd, m, d = test_data.fixture("tendon/tendon_limit.xml", keyframe=keyframe, overrides={"opt.jacobian": jacobian})

      for arr in (d.nefc, d.nl, d.efc.type):
        arr.fill_(-1)
      for arr in (d.efc.J, d.efc.D, d.efc.vel, d.efc.aref, d.efc.pos, d.efc.margin):
        arr.fill_(wp.nan)

      mjw.make_constraint(m, d)

      _assert_eq(d.nefc.numpy()[0], mjd.nefc, "nefc")
      _assert_eq(d.nl.numpy()[0], mjd.nl, "nl")
      _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc", m.nv)

  @parameterized.product(
    xml=(
      # no tendon touches a dof, so nJten is 0
      """
      <mujoco>
        <worldbody>
          <site name="w1"/>
          <site name="w2" pos="1 0 0"/>
          <body pos="0 0 1">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <tendon>
          <spatial name="empty1" frictionloss="1">
            <site site="w1"/>
            <site site="w2"/>
          </spatial>
          <spatial name="empty2" limited="true" range="0 .5">
            <site site="w1"/>
            <site site="w2"/>
          </spatial>
        </tendon>
        <equality>
          <tendon tendon1="empty1" polycoef=".5 0 0 0 0"/>
          <tendon tendon1="empty1" tendon2="empty2"/>
        </equality>
      </mujoco>
      """,
      # empty tendons next to a tendon with dofs
      """
      <mujoco>
        <worldbody>
          <site name="w1"/>
          <site name="w2" pos="1 0 0"/>
          <body pos="0 0 1">
            <freejoint/>
            <geom size=".1"/>
            <site name="b"/>
          </body>
        </worldbody>
        <tendon>
          <spatial name="empty1" frictionloss="1">
            <site site="w1"/>
            <site site="w2"/>
          </spatial>
          <spatial name="empty2" limited="true" range="0 .5">
            <site site="w1"/>
            <site site="w2"/>
          </spatial>
          <spatial name="active" frictionloss="1">
            <site site="w1"/>
            <site site="b"/>
          </spatial>
        </tendon>
        <equality>
          <tendon tendon1="empty1" polycoef=".5 0 0 0 0"/>
          <tendon tendon1="empty1" tendon2="empty2"/>
          <tendon tendon1="empty1" tendon2="active"/>
          <tendon tendon1="active" tendon2="empty1"/>
        </equality>
      </mujoco>
      """,
      # no dofs
      """
      <mujoco>
        <worldbody>
          <site name="w1"/>
          <site name="w2" pos="1 0 0"/>
        </worldbody>
        <tendon>
          <spatial name="empty1" frictionloss="1">
            <site site="w1"/>
            <site site="w2"/>
          </spatial>
          <spatial name="empty2" limited="true" range="0 .5">
            <site site="w1"/>
            <site site="w2"/>
          </spatial>
        </tendon>
        <equality>
          <tendon tendon1="empty1" polycoef=".5 0 0 0 0"/>
          <tendon tendon1="empty1" tendon2="empty2"/>
        </equality>
      </mujoco>
      """,
    ),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_tendon_empty_jacobian(self, xml, jacobian):
    """Tendons with an empty Jacobian do not add equality, friction or limit rows."""
    mjm, mjd, m, d = test_data.fixture(
      xml=xml,
      overrides={"opt.jacobian": jacobian},
      qvel_noise=0.1,
    )

    for arr in (d.nefc, d.ne, d.nf, d.nl, d.efc.type):
      arr.fill_(-1)
    for arr in (d.efc.J, d.efc.D, d.efc.vel, d.efc.aref, d.efc.pos, d.efc.margin):
      arr.fill_(wp.inf)

    mjw.make_constraint(m, d)

    _assert_eq(d.nefc.numpy()[0], mjd.nefc, "nefc")
    _assert_eq(d.ne.numpy()[0], mjd.ne, "ne")
    _assert_eq(d.nf.numpy()[0], mjd.nf, "nf")
    _assert_eq(d.nl.numpy()[0], mjd.nl, "nl")
    if mjd.nefc > 0:
      _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc", m.nv)

  @parameterized.product(
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
    deterministic=(mjw.DeterminismType.NONE, mjw.DeterminismType.CONSTRAINT),
  )
  def test_limit_bilateral_margin(self, jacobian, deterministic):
    """Test limits with both sides of the range within the margin."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag contact="disable"/>
        </option>
        <worldbody>
          <site name="anchor" pos="0 0 1"/>
          <body>
            <joint axis="0 1 0"/>
            <joint name="hinge" axis="1 0 0" range="0 0.1" margin="0.2"/>
            <geom size=".1"/>
          </body>
          <body pos="1 0 0">
            <joint name="slide" type="slide" axis="0 0 1" range="0 0.1" margin="0.2"/>
            <geom size=".1"/>
            <site name="end"/>
          </body>
        </worldbody>
        <tendon>
          <spatial name="tendon" range="1.4 1.42" margin="0.1">
            <site site="anchor"/>
            <site site="end"/>
          </spatial>
        </tendon>
        <sensor>
          <jointlimitpos joint="hinge"/>
          <jointlimitvel joint="hinge"/>
          <jointlimitfrc joint="hinge"/>
          <tendonlimitpos tendon="tendon"/>
          <tendonlimitvel tendon="tendon"/>
          <tendonlimitfrc tendon="tendon"/>
        </sensor>
        <keyframe>
          <key qvel="0.1 0.3 -0.2"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      overrides={"opt.jacobian": jacobian},
    )
    m.opt.deterministic = deterministic

    for arr in (d.nefc, d.nl, d.efc.type):
      arr.fill_(-1)
    for arr in (d.efc.J, d.efc.D, d.efc.vel, d.efc.aref, d.efc.pos, d.efc.margin):
      arr.fill_(wp.inf)

    mjw.make_constraint(m, d)

    # two rows for each limit
    _assert_eq(d.nefc.numpy()[0], 6, "nefc")
    _assert_eq(d.nl.numpy()[0], mjd.nl, "nl")
    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc", m.nv)

    if m.is_sparse:
      # each two-row limit block covers its own rows, so the blocks tile [0, nefc)
      nblock = d.efc.jtdaj_nblock.numpy()[0]
      adr = d.efc.jtdaj_adr.numpy()[0, :nblock]
      nrow = d.efc.jtdaj_nrow.numpy()[0, :nblock]
      rows = sorted(r for a, n in zip(adr, nrow) for r in range(a, a + n))
      self.assertEqual(rows, list(range(d.nefc.numpy()[0])))

    d = mjw.put_data(mjm, mjd)
    for arr in (d.qfrc_constraint, d.sensordata, d.cfrc_ext):
      arr.fill_(wp.inf)
    mjw.forward(m, d)
    mjw.rne_postconstraint(m, d)
    mujoco.mj_rnePostConstraint(mjm, mjd)

    _assert_eq(d.qfrc_constraint.numpy()[0], mjd.qfrc_constraint, "qfrc_constraint")
    _assert_eq(d.sensordata.numpy()[0], mjd.sensordata, "sensordata")
    _assert_eq(d.cfrc_ext.numpy()[0], mjd.cfrc_ext, "cfrc_ext")

  @parameterized.parameters(
    mujoco.mjtJacobian.mjJAC_DENSE,
    mujoco.mjtJacobian.mjJAC_SPARSE,
  )
  def test_equality_tendon(self, jacobian):
    """Test equality tendon constraints."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag contact="disable"/>
        </option>
        <worldbody>
          <body>
            <geom type="sphere" size=".1"/>
            <joint name="joint0" type="hinge"/>
          </body>
          <body>
            <geom type="sphere" size=".1"/>
            <joint name="joint1" type="hinge"/>
          </body>
          <body>
            <geom type="sphere" size=".1"/>
            <joint name="joint2" type="hinge"/>
          </body>
        </worldbody>
        <tendon>
          <fixed name="tendon0">
            <joint joint="joint0" coef=".1"/>
          </fixed>
          <fixed name="tendon1">
            <joint joint="joint1" coef=".2"/>
          </fixed>
          <fixed name="tendon2">
            <joint joint="joint2" coef=".3"/>
          </fixed>
        </tendon>
        <equality>
          <tendon tendon1="tendon0" tendon2="tendon1" polycoef=".1 .2 .3 .4 .5"/>
          <tendon tendon1="tendon2" polycoef="-.1 0 0 0 0"/>
        </equality>
        <keyframe>
          <key qpos=".1 .2 .3"/>
        </keyframe>
      </mujoco>
    """,
      keyframe=0,
      overrides={"opt.jacobian": jacobian},
    )

    for arr in (d.nefc, d.ne, d.efc.type):
      arr.fill_(-1)
    for arr in (d.efc.J, d.efc.D, d.efc.vel, d.efc.aref, d.efc.pos, d.efc.margin):
      arr.fill_(wp.nan)

    mjw.make_constraint(m, d)

    _assert_eq(d.nefc.numpy()[0], mjd.nefc, "nefc")
    _assert_eq(d.ne.numpy()[0], mjd.ne, "ne")
    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc", m.nv)

  def test_equality_tendon_nan(self):
    """A NaN tendon equality row keeps MuJoCo's columns, within its sparse Jacobian storage."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option jacobian="sparse"/>
        <default>
          <geom type="sphere" size=".1" contype="0" conaffinity="0"/>
        </default>
        <worldbody>
          <body>
            <geom/>
            <joint name="joint0" type="hinge"/>
            <body pos=".2 0 0">
              <geom/>
              <joint name="joint1" type="hinge"/>
              <body pos=".2 0 0">
                <geom/>
                <joint name="joint2" type="hinge"/>
              </body>
            </body>
          </body>
        </worldbody>
        <tendon>
          <fixed name="tendon0">
            <joint joint="joint1" coef=".1"/>
          </fixed>
          <fixed name="tendon1">
            <joint joint="joint2" coef=".2"/>
          </fixed>
        </tendon>
        <equality>
          <joint joint1="joint0"/>
          <tendon tendon1="tendon1" tendon2="tendon0"/>
        </equality>
        <keyframe>
          <key qpos=".3 -.2 .1"/>
        </keyframe>
      </mujoco>
    """,
      keyframe=0,
      nworld=2,
    )
    self.assertTrue(m.is_sparse)

    # world 0 has a NaN state, so the coupling derivative of its tendon equality row is NaN
    mjd_nan = mujoco.MjData(mjm)
    mjd_nan.qpos[:] = np.nan
    mujoco.mj_forward(mjm, mjd_nan)
    d.qpos.assign(np.stack((mjd_nan.qpos, mjd.qpos)))

    for arr in (d.nefc, d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind):
      arr.fill_(-1)
    for arr in (d.efc.J, d.qacc):
      arr.fill_(wp.inf)

    mjw.forward(m, d)

    nefc = d.nefc.numpy()
    rownnz = d.efc.J_rownnz.numpy()
    rowadr = d.efc.J_rowadr.numpy()
    colind = d.efc.J_colind.numpy()
    for worldid, mjd_world in enumerate((mjd_nan, mjd)):
      _assert_eq(nefc[worldid], mjd_world.nefc, f"nefc[{worldid}]")
      _assert_eq(rownnz[worldid, : nefc[worldid]], mjd_world.efc_J_rownnz, f"J_rownnz[{worldid}]")
      for r in range(mjd_world.nefc):
        adr = rowadr[worldid, r]
        nnz = rownnz[worldid, r]
        mj_adr = mjd_world.efc_J_rowadr[r]
        mj_nnz = mjd_world.efc_J_rownnz[r]
        _assert_eq(
          colind[worldid, 0, adr : adr + nnz],
          mjd_world.efc_J_colind[mj_adr : mj_adr + mj_nnz],
          f"J_colind[{worldid}, {r}]",
        )

    # world 1 is unaffected by world 0
    J = np.zeros((mjd.nefc, mjm.nv))
    mujoco.mju_sparse2dense(
      J,
      d.efc.J.numpy()[1, 0].astype(np.float64),
      rownnz[1, : mjd.nefc],
      rowadr[1, : mjd.nefc],
      colind[1, 0],
    )
    mj_J = np.zeros((mjd.nefc, mjm.nv))
    mujoco.mju_sparse2dense(mj_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
    _assert_eq(J, mj_J, "J[1]")
    _assert_eq(d.qacc.numpy()[1], mjd.qacc, "qacc[1]")

  def test_efc_address_inactive_contacts(self):
    """Test that efc_address is -1 for inactive contacts in the gap zone."""
    # Sphere at z=0.35 with radius 0.1: dist ~ 0.15 to ground plane.
    # margin=0.1, gap=0.4 => detection at margin+gap=0.5, forces at margin=0.1.
    # dist(0.15) < margin+gap(0.5) => contact is detected.
    # dist(0.15) >= margin(0.1) => contact is NOT active (in gap zone).
    xml = f"""
      <mujoco>
        <worldbody>
          <geom type="plane" size="10 10 .001" margin="0.1" gap="0.4"/>
          <body pos="0 0 0.35">
            <geom type="sphere" size=".1" margin="0.1" gap="0.4"/>
            <freejoint/>
          </body>
        </worldbody>
        <keyframe>
          <key qpos="0 0 0.35 1 0 0 0" />
        </keyframe>
      </mujoco>
    """

    for cone in (ConeType.PYRAMIDAL, ConeType.ELLIPTIC):
      mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0, overrides={"opt.cone": cone})

      # Verify MuJoCo detected a contact but it's not active (nefc == 0)
      self.assertGreater(mjd.ncon, 0, "Expected at least one contact")
      self.assertEqual(mjd.nefc, 0, "Expected no active constraints")

      # Pre-fill efc_address with stale positive values to simulate the bug
      d.contact.efc_address.fill_(999)

      mjw.collision(m, d)
      mjw.make_constraint(m, d)

      # efc_address for written contacts should be -1 (inactive, in gap zone)
      nacon = d.nacon.numpy()[0]
      self.assertGreater(nacon, 0, "Expected at least one contact")
      efc_address = d.contact.efc_address.numpy()[:nacon]
      _assert_eq(efc_address, -1, f"efc_address (cone={cone})")

  @parameterized.product(
    eq_type=("connect", "weld"),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_equality_rows_past_njmax(self, eq_type, jacobian):
    """Equality blocks that straddle njmax keep the rows that fit."""
    nrow = 3 if eq_type == "connect" else 6
    for njmax in (nrow - 1, nrow):
      mjm, mjd, m, d = test_data.fixture(
        xml=f"""
        <mujoco>
          <option>
            <flag contact="disable"/>
          </option>
          <worldbody>
            <body name="body" pos="0 0 1">
              <freejoint/>
              <geom type="box" size=".1 .2 .3"/>
            </body>
          </worldbody>
          <equality>
            <{eq_type} body1="body" anchor=".1 .2 .3" active="false"/>
          </equality>
          <keyframe>
            <key qpos=".1 .2 1.3 .9 .1 .2 .3"/>
          </keyframe>
        </mujoco>
        """,
        keyframe=0,
        njmax=njmax,
        overrides={"opt.jacobian": jacobian},
      )

      mjd.eq_active[0] = 1
      mujoco.mj_forward(mjm, mjd)
      self.assertEqual(mjd.nefc, nrow)
      mj_J = np.zeros((mjd.nefc, mjm.nv))
      if mujoco.mj_isSparse(mjm):
        mujoco.mju_sparse2dense(mj_J, mjd.efc_J, mjd.efc_J_rownnz, mjd.efc_J_rowadr, mjd.efc_J_colind)
      else:
        mj_J = mjd.efc_J.reshape(mjd.nefc, mjm.nv)

      d.eq_active.fill_(True)
      for arr in (d.nefc, d.ne, d.efc.type):
        arr.fill_(-1)
      for arr in (d.efc.J, d.efc.D, d.efc.vel, d.efc.aref, d.efc.pos):
        arr.fill_(wp.inf)

      mjw.make_constraint(m, d)

      self.assertEqual(d.nefc.numpy()[0], nrow)
      n = min(nrow, njmax)
      if m.is_sparse:
        J = np.zeros((n, mjm.nv))
        mujoco.mju_sparse2dense(
          J,
          d.efc.J.numpy()[0, 0].astype(np.float64),
          d.efc.J_rownnz.numpy()[0, :n],
          d.efc.J_rowadr.numpy()[0, :n],
          d.efc.J_colind.numpy()[0, 0],
        )
      else:
        J = d.efc.J.numpy()[0, :n, : mjm.nv]
      _assert_eq(J, mj_J[:n], "J")
      _assert_eq(d.efc.type.numpy()[0, :n], mjd.efc_type[:n], "type")
      _assert_eq(d.efc.pos.numpy()[0, :n], mjd.efc_pos[:n], "pos")
      _assert_eq(d.efc.D.numpy()[0, :n], mjd.efc_D[:n], "D")
      _assert_eq(d.efc.vel.numpy()[0, :n], mjd.efc_vel[:n], "vel")
      _assert_eq(d.efc.aref.numpy()[0, :n], mjd.efc_aref[:n], "aref")

  @parameterized.product(
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    nworld=(1, 2),
    overflow=(True, False),
  )
  def test_njmax_nnz_overflow_rows(self, cone, nworld, overflow):
    """Rows that do not fit in njmax_nnz are dropped instead of keeping stale sparse layouts."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option jacobian="sparse"/>
        <worldbody>
          <geom type="plane" size="10 10 .01"/>
          <body pos="0 0 1">
            <joint range="-.1 .1"/>
            <geom size=".05" contype="0" conaffinity="0"/>
          </body>
          <body pos="1 0 1">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body pos="2 0 1">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body pos="3 0 1">
            <freejoint/>
            <geom size=".1"/>
          </body>
          <body pos="4 0 1">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </worldbody>
        <keyframe>
          <key qpos=".5  1 0 .09 1 0 0 0  2 0 .09 1 0 0 0  3 0 .09 1 0 0 0  4 0 .09 1 0 0 0"/>
        </keyframe>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
      nworld=nworld,
      njmax_nnz=30 if overflow else None,
    )
    self.assertTrue(m.is_sparse)
    self.assertLessEqual(mjm.nv, d.njmax_nnz)

    # limit and contacts active: needs more Jacobian storage than njmax_nnz
    mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    d.qpos.assign(np.tile(mjd.qpos, (nworld, 1)))

    # stale layout from a previous step: rows covering the whole storage
    d.efc.J_rowadr.fill_(0)
    d.efc.J_rownnz.fill_(mjm.nv)
    d.overflow.zero_()

    mjw.fwd_position(m, d)

    nefc = d.nefc.numpy()
    efc_type = d.efc.type.numpy()
    rowadr = d.efc.J_rowadr.numpy()
    rownnz = d.efc.J_rownnz.numpy()
    colind = d.efc.J_colind.numpy()
    nnz_overflow = d.overflow.numpy() & int(mjw.OverflowType.NJMAX_NNZ)
    for worldid in range(nworld):
      self.assertEqual(bool(nnz_overflow[worldid]), overflow)
      nrow = min(nefc[worldid], d.njmax)
      self.assertGreater(nrow, 0)
      adr, nnz = rowadr[worldid, :nrow], rownnz[worldid, :nrow]

      np.testing.assert_array_less(-1, adr)
      np.testing.assert_array_less(adr + nnz, d.njmax_nnz + 1)
      order = np.argsort(adr)
      np.testing.assert_array_less((adr + nnz)[order][:-1], adr[order][1:] + 1, err_msg="overlapping rows")

      is_contact = efc_type[worldid, :nrow] >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
      _assert_eq(nnz[is_contact], 6, "contact rownnz")
      for a, n in zip(adr, nnz):
        cols = colind[worldid, 0, a : a + n]
        self.assertTrue(np.all((cols >= 0) & (cols < mjm.nv)), f"colind {cols}")

  @parameterized.parameters(
    *itertools.product(
      ("jdotv/connect_2d.xml", "jdotv/connect_3d.xml", "jdotv/weld_3d.xml"),
      (mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
    )
  )
  def test_jdotv(self, xml, jacobian):
    """Test Jdotv correction."""
    mjm, mjd, m, d = test_data.fixture(xml, keyframe=0, overrides={"opt.jacobian": jacobian})

    mujoco.mj_forward(mjm, mjd)
    wp.copy(d.qvel, wp.array(mjd.qvel, dtype=float))

    for arr in (d.ne, d.nefc, d.nf, d.nl, d.efc.type):
      arr.fill_(-1)
    for arr in (d.efc.J, d.efc.D, d.efc.vel, d.efc.aref, d.efc.pos, d.efc.margin):
      arr.fill_(wp.inf)

    mjw.make_constraint(m, d)

    _assert_eq(d.ne.numpy()[0], mjd.ne, "ne")
    _assert_eq(d.nefc.numpy()[0], mjd.nefc, "nefc")
    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc", m.nv)

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
  )
  def test_flex_barycentric_jacobian(self, xml, cone, jacobian):
    """Test barycentric contact Jacobian calculation for flex."""
    mjm, mjd, m, d = test_data.fixture(xml=xml, overrides={"opt.cone": cone, "opt.jacobian": jacobian})

    mjw.kinematics(m, d)
    mjw.make_constraint(m, d)

    self.assertGreater(mjd.nefc, 0, "Expected active contacts")
    self.assertEqual(d.nefc.numpy()[0], mjd.nefc, "nefc mismatch")

    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, f"efc_flex_dim{m.flex_dim.numpy()[0]}", m.nv)

  def test_flex_3d_simplex_collision(self):
    """Test 3D simplex flex collision and constraint generation."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
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
    )

    mjw.kinematics(m, d)
    mjw.collision(m, d)
    mjw.make_constraint(m, d)

    self.assertGreater(mjd.nefc, 0, "Expected active contacts")
    self.assertEqual(d.nefc.numpy()[0], mjd.nefc, "nefc mismatch")

    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc_flex_3d_simplex", m.nv, tol=1e-3)

  # TODO(thowell): Add support for quadratic flex interpolation.
  @parameterized.parameters(
    ("2 1 2", "none"),
    ("2 2 2", "bend"),
  )
  def test_flex_interpolated(self, cellcount, elastic2d):
    """Test collision and constraint generation for interpolated volume and shell flexes."""
    xml = f"""
      <mujoco>
        <option solver="CG" tolerance="1e-6" timestep=".001"/>
        <worldbody>
          <body pos="0 0 0.07">
            <freejoint/>
            <geom type="box" size=".05 .05 .05" mass="1"/>
          </body>
          <flexcomp name="softbody" type="grid" count="4 2 4" spacing=".025 .05 .025" pos="0 0 0"
                    dim="3" cellcount="{cellcount}" radius=".001"
                    mass="5" dof="trilinear">
            <elasticity young="1e4" poisson="0.3" damping="1e-3" elastic2d="{elastic2d}" thickness="0.01"/>
            <contact condim="3" selfcollide="none"/>
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
    )

    mjw.kinematics(m, d)
    mjw.make_constraint(m, d)

    self.assertGreater(mjd.nefc, 0, "Expected active contacts in MuJoCo")
    self.assertEqual(d.nefc.numpy()[0], mjd.nefc, "nefc mismatch")
    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, f"efc_flex_interpolated_trilinear_{elastic2d}", m.nv)

  # TODO(team): remove version check once mujoco >= 4.0.2 is required.
  @absltest.skipIf(not util_pkg.check_version("mujoco>=4.0.2"), "Requires MuJoCo >= 4.0.2")
  def test_flex_interpolated_empty_cell(self):
    """Test contact Jacobians of interpolated flex vertices redirected from an empty cell."""
    # cell 1 (x in [0.5, 1]) only contains a 1% sliver and is pruned as empty, so the vertices at
    # x = 0.5 and x = 0.505 are redirected to cell 0 and extrapolated, also for contact weights
    mjm, mjd, m, d = test_data.fixture(
      xml="""
        <mujoco>
          <option solver="CG" tolerance="1e-6" timestep=".001"/>
          <worldbody>
            <geom type="plane" size="5 5 .1"/>
            <flexcomp name="sliver" type="direct" dim="3" dof="trilinear" mass="1"
                      cellcount="2 1 1" radius="0.001"
                      point="0.0 0 0  0.5 0 0  0.0 1 0  0.5 1 0
                             0.0 0 1  0.5 0 1  0.0 1 1  0.5 1 1
                             0.505 0 0  0.505 1 0  0.505 0 1  0.505 1 1
                             1.0 0.5 0.5"
                      element="0 1 2 4  3 2 1 7  5 1 4 7  6 2 4 7  1 2 4 7
                               1 8 3 5  9 3 8 11  10 8 5 11  7 3 5 11  8 3 5 11">
              <contact condim="3" selfcollide="none"/>
              <elasticity young="1e4" poisson="0.3" elastic2d="none"/>
            </flexcomp>
          </worldbody>
        </mujoco>
      """,
      overrides={
        "opt.cone": mujoco.mjtCone.mjCONE_ELLIPTIC,
        "opt.jacobian": mujoco.mjtJacobian.mjJAC_SPARSE,
      },
    )

    # some contact vertex is redirected away from its natural cell
    self.assertGreater(mjm.flex_interp[0], 0)
    cellnum = mjm.flex_cellnum[0]
    natcell = np.clip(np.floor(mjm.flex_vert0 * cellnum).astype(int), 0, cellnum - 1)
    natcell = natcell[:, 0] * cellnum[1] * cellnum[2] + natcell[:, 1] * cellnum[2] + natcell[:, 2]
    redirected = (mjm.flex_vertcell >= 0) & (mjm.flex_vertcell != natcell)
    contact_verts = [v for v, f in zip(mjd.contact.vert.flat, mjd.contact.flex.flat) if f == 0 and v >= 0]
    self.assertTrue(any(redirected[v] for v in contact_verts), "Expected contacts at redirected vertices")

    mjw.kinematics(m, d)
    mjw.make_constraint(m, d)

    self.assertGreater(mjd.nefc, 0, "Expected active contacts in MuJoCo")
    self.assertEqual(d.nefc.numpy()[0], mjd.nefc, "nefc mismatch")
    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "efc_flex_interpolated_empty_cell", m.nv)

  def test_weld_coriolis(self):
    """Test weld Coriolis term Jdotv_r with default torquescale and high velocity."""
    xml = """
      <mujoco>
        <worldbody>
          <body name="body1" pos="0 0 0">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
          </body>
          <body name="body2" pos="0 0 1">
            <freejoint/>
            <geom type="box" size=".1 .1 .1"/>
          </body>
        </worldbody>
        <equality>
          <weld body1="body1" body2="body2"/>
        </equality>
        <keyframe>
          <key qpos="0.1 0.2 0.3 0.5940885257860046 0.3960590171906697 0.49507377148833714 0.49507377148833714
                     -0.1 0.5 0.8 0.502518907629606 -0.30151134457776363 0.8040302522073697 0.10050378152592121"
               qvel="1.0 -2.0 3.0 5.0 10.0 -8.0 -3.0 2.0 1.0 -7.0 4.0 6.0"/>
        </keyframe>
      </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=0)

    mjw.make_constraint(m, d)

    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "weld_coriolis", m.nv, tol=1e-3)

  @parameterized.product(
    xml=[
      """
          <mujoco>
            <worldbody>
              <geom type="plane" size="5 5 .1" surfacevel=".5 0 0  0 0 0"/>
              <body pos="0 0 .099">
                <freejoint/>
                <geom type="box" size=".1 .1 .1" mass="1"/>
              </body>
            </worldbody>
          </mujoco>
          """,
      """
          <mujoco>
            <worldbody>
              <geom type="plane" size="5 5 .1" condim="6" surfacevel="0 0 0  0 0 1"/>
              <body pos="0 0 .099">
                <freejoint/>
                <geom type="box" size=".1 .1 .1" mass="1" condim="6"/>
              </body>
            </worldbody>
          </mujoco>
          """,
      """
          <mujoco>
            <worldbody>
              <geom type="plane" size="5 5 .1" surfacevel="0 0 1  0 0 0"/>
              <body pos="0 0 .099">
                <freejoint/>
                <geom type="box" size=".1 .1 .1" mass="1"/>
              </body>
            </worldbody>
          </mujoco>
          """,
      """
          <mujoco>
            <worldbody>
              <geom type="box" size=".05 .3 .5" pos="-.152 0 .5" surfacevel="0 0 1  0 0 0"/>
              <geom type="box" size=".05 .3 .5" pos=".152 0 .5" surfacevel="0 0 1  0 0 0"/>
              <body pos="0 0 .5">
                <freejoint/>
                <geom type="box" size=".103 .1 .1" mass=".2"/>
              </body>
            </worldbody>
          </mujoco>
          """,
      """
          <mujoco>
            <worldbody>
              <geom type="plane" size="10 10 .1" friction="0 0 0" condim="1" priority="1"/>
              <body name="vehicle" pos="0 0 .15">
                <freejoint/>
                <geom type="box" size=".5 .3 .04" mass="10"/>
                <geom type="box" size=".4 .25 .01" pos="0 0 .05" mass=".01" surfacevel=".3 0 0  0 0 0"/>
              </body>
              <body name="cargo" pos="0 0 .28">
                <freejoint/>
                <geom type="box" size=".08 .08 .08" mass=".1"/>
              </body>
            </worldbody>
            <keyframe>
              <key qvel=".2 0 0 0 0 0  0 0 0 0 0 0"/>
            </keyframe>
          </mujoco>
          """,
    ],
    cone=[ConeType.PYRAMIDAL, ConeType.ELLIPTIC],
    jacobian=[mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE],
  )
  def test_surfacevel(self, xml, cone, jacobian):
    """Test surfacevel calculation."""
    keyframe = 0 if "<keyframe>" in xml else None
    mjm, mjd, m, d = test_data.fixture(xml=xml, keyframe=keyframe, overrides={"opt.cone": cone, "opt.jacobian": jacobian})

    d.efc.vel.fill_(wp.inf)
    d.efc.aref.fill_(wp.inf)
    mjw.make_constraint(m, d)

    self.assertGreater(mjd.nefc, 0)
    _assert_efc_eq(mjm, m, d, mjd, mjd.nefc, "surfacevel", m.nv)

  @parameterized.product(
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    nworld=(1, 2),
  )
  def test_contact_surfacevel_external(self, cone, nworld):
    """Applies external contact.surfacevel when geom_surfacevel is zero."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <geom type="plane" size="2 2 0.01" condim="4" friction="0.8 0.02 0.001"/>
          <body pos="0 0 0.095">
            <freejoint/>
            <geom size="0.1" condim="4" friction="0.8 0.02 0.001"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
      nworld=nworld,
    )
    self.assertFalse(mjm.flg_surfacevel)
    self.assertFalse(m.flg_surfacevel)
    self.assertEqual(mjd.ncon, 1)

    mjm.flg_surfacevel = 1
    m.flg_surfacevel = True

    sv_worlds = [np.array([0.4, -0.25, 0.8], dtype=np.float32)]
    if nworld == 2:
      sv_worlds.append(np.array([-0.3, 0.5, -0.6], dtype=np.float32))

    sv_np = d.contact.surfacevel.numpy()
    worldid = d.contact.worldid.numpy()[:nworld]
    mjds = [mjd]
    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      mujoco.mj_forward(mjm, mjd1)
      mjds.append(mjd1)

    for w in range(nworld):
      conid = int(np.flatnonzero(worldid == w)[0])
      sv_np[conid] = sv_worlds[w]
      mjds[w].contact.surfacevel[0] = sv_worlds[w]
      mujoco.mj_makeConstraint(mjm, mjds[w])
      mujoco.mj_fwdVelocity(mjm, mjds[w])

    d.contact.surfacevel.assign(sv_np)
    d.efc.Jqvel.fill_(wp.inf)
    d.efc.vel.fill_(wp.inf)
    d.efc.aref.fill_(wp.inf)
    mjw.make_constraint(m, d)

    nefc = mjd.nefc
    self.assertGreater(nefc, 1)
    efc_vel = d.efc.vel.numpy()
    efc_aref = d.efc.aref.numpy()
    for w in range(nworld):
      np.testing.assert_allclose(efc_vel[w, :nefc], mjds[w].efc_vel[:nefc], rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(efc_aref[w, :nefc], mjds[w].efc_aref[:nefc], rtol=1e-5, atol=1e-5)

    if nworld == 2:
      self.assertFalse(np.allclose(efc_vel[0, :nefc], efc_vel[1, :nefc]))

  @parameterized.parameters(1, 2)
  def test_discrete_reference_constraint(self, nworld):
    """Verifies that discrete efc_aref and efc_D match MuJoCo C after forward actuation."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco model="test_discrete_ref">
        <option integrator="discrete"/>
        <worldbody>
          <body name="body1" pos="0 0 1">
            <joint name="jnt1" type="slide" axis="0 0 1" damping="2.0"/>
            <geom type="sphere" size="0.1" mass="1.0"/>
          </body>
          <body name="body2" pos="0 0 0.5">
            <joint name="jnt2" type="slide" axis="0 0 1" damping="1.0"/>
            <geom type="sphere" size="0.1" mass="1.0"/>
          </body>
        </worldbody>
        <equality>
          <joint joint1="jnt1" joint2="jnt2" polycoef="0 1 0 0 0" solref="0.01 1" solimp="0.9 0.95 0.001"/>
        </equality>
        <actuator>
          <position joint="jnt1" kp="100" dampratio="1"/>
        </actuator>
      </mujoco>
      """,
      nworld=nworld,
    )

    mjds = [mjd]
    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      ctrl = d.ctrl.numpy()
      ctrl[1] = 0.5
      d.ctrl.assign(ctrl)
      mjd1.ctrl[:] = ctrl[1]
      qvel = d.qvel.numpy()
      qvel[1, 0] = 0.8
      qvel[1, 1] = -0.4
      d.qvel.assign(qvel)
      mjd1.qvel[:] = qvel[1]
      mjds.append(mjd1)

    d.efc.D.fill_(wp.inf)
    d.efc.aref.fill_(wp.inf)

    mjw.forward(m, d)

    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])
      nefc = mjds[w].nefc
      np.testing.assert_allclose(d.efc.D.numpy()[w, :nefc], mjds[w].efc_D[:nefc], rtol=1e-4, atol=1e-5)
      np.testing.assert_allclose(d.efc.aref.numpy()[w, :nefc], mjds[w].efc_aref[:nefc], rtol=1e-4, atol=1e-5)

    if nworld == 2:
      self.assertFalse(np.allclose(d.efc.aref.numpy()[0], d.efc.aref.numpy()[1]))

  # TODO(team): remove version check once mujoco >= 4.0.2 is required.
  @absltest.skipIf(not util_pkg.check_version("mujoco>=4.0.2"), "Requires MuJoCo >= 4.0.2")
  @parameterized.product(
    nworld=(1, 2),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_discrete_refsafe_elliptic_friction(self, nworld, jacobian):
    """Elliptic friction rows use the unclamped normal regularization under discrete refsafe."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.02" integrator="discrete" cone="elliptic" impratio="10">
          <flag warmstart="disable"/>
        </option>
        <worldbody>
          <geom type="plane" size="2 2 0.1" euler="0 20 0" friction="0.5 0.5 0.01" solref="0.002 1"/>
          <body pos="0 0 0.0995">
            <freejoint/>
            <geom size="0.1" mass="1" friction="0.5 0.5 0.01" solref="0.002 1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.jacobian": jacobian},
      nworld=nworld,
    )

    mjds = [mjd]
    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      qvel = d.qvel.numpy()
      qvel[1, :3] = [0.3, -0.2, -0.5]
      d.qvel.assign(qvel)
      mjd1.qvel[:] = qvel[1]
      mjds.append(mjd1)

    d.efc.D.fill_(wp.inf)
    d.qacc.fill_(wp.inf)
    mjw.forward(m, d)

    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])
      nefc = mjds[w].nefc
      self.assertEqual(nefc, 3)
      # refsafe clamps the normal row, so the friction-to-normal ratio is no longer 1/impratio
      self.assertGreater(mjds[w].efc_R[0] / mjds[w].efc_R[1], 2.0 * mjm.opt.impratio)
      _assert_eq(d.efc.D.numpy()[w, :nefc], mjds[w].efc_D[:nefc], f"efc_D_world_{w}")
      _assert_eq(d.qacc.numpy()[w], mjds[w].qacc, f"qacc_world_{w}")

    if nworld == 2:
      self.assertFalse(np.allclose(d.qacc.numpy()[0], d.qacc.numpy()[1]))

  @parameterized.parameters(1, 2)
  def test_regularize_constraint_ceiling(self, nworld):
    """Verifies that regularize_constraint clamps D within the impedance ceiling."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco model="test_reg_ceiling">
        <option integrator="discrete"/>
        <worldbody>
          <body name="body1" pos="0 0 1">
            <joint name="jnt1" type="slide" axis="0 0 1"/>
            <geom type="sphere" size="0.1" mass="1.0"/>
          </body>
        </worldbody>
        <equality>
          <joint joint1="jnt1" polycoef="0 1 0 0 0" solref="0.001 1" solimp="0.999 0.9999 0.001"/>
        </equality>
        <actuator>
          <velocity joint="jnt1" kv="1000"/>
        </actuator>
      </mujoco>
      """,
      nworld=nworld,
    )

    mjds = [mjd]
    if nworld == 2:
      mjd1 = mujoco.MjData(mjm)
      qpos = d.qpos.numpy()
      qpos[1] = 0.005
      d.qpos.assign(qpos)
      mjd1.qpos[:] = qpos[1]
      ctrl = d.ctrl.numpy()
      ctrl[1] = 2.0
      d.ctrl.assign(ctrl)
      mjd1.ctrl[:] = ctrl[1]
      mjds.append(mjd1)

    d.efc.D.fill_(wp.inf)

    mjw.forward(m, d)

    for w in range(nworld):
      mujoco.mj_forward(mjm, mjds[w])
      nefc = mjds[w].nefc
      np.testing.assert_allclose(d.efc.D.numpy()[w, :nefc], mjds[w].efc_D[:nefc], rtol=1e-3, atol=1e-4)

    if nworld == 2:
      self.assertFalse(np.allclose(d.efc.D.numpy()[0], d.efc.D.numpy()[1]))

  @parameterized.product(
    cone=(mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtCone.mjCONE_ELLIPTIC),
    nworld=(1, 2),
  )
  def test_regularize_constraint_contact_and_adhesion(self, cone, nworld):
    """Verifies regularize_constraint scales contact rows by r0 and updates adhesion aref."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.01"/>
        <worldbody>
          <geom type="plane" size="2 2 0.01" friction="0.8 0.01 0.001"/>
          <body name="b0" pos="0 0 0.095">
            <joint name="jx" type="slide" axis="1 0 0" stiffness="10" damping="0.5"/>
            <joint name="jz" type="slide" axis="0 0 1" stiffness="500" damping="20"/>
            <geom type="sphere" size="0.1" mass="1.0" margin="0.02"/>
          </body>
        </worldbody>
        <actuator>
          <adhesion name="adh" body="b0" ctrlrange="0 10" gain="5"/>
        </actuator>
      </mujoco>
      """,
      overrides={"opt.cone": cone},
      nworld=nworld,
    )
    mjd.ctrl[0] = 4.0
    ctrl_np = np.full((nworld, 1), 4.0, dtype=np.float32)
    if nworld == 2:
      ctrl_np[1, 0] = 8.0
      qvel_np = d.qvel.numpy()
      qvel_np[1] = [0.2, -0.3]
      d.qvel = wp.array(qvel_np, dtype=float)
    d.ctrl = wp.array(ctrl_np, dtype=float)
    d.efc.D.fill_(wp.inf)
    d.efc.aref.fill_(wp.inf)

    mujoco.mj_forward(mjm, mjd)
    mjw.forward(m, d)

    nefc = mjd.nefc
    self.assertGreater(nefc, 1)
    _assert_eq(d.efc.D.numpy()[0, :nefc], mjd.efc_D[:nefc], "efc_D")
    _assert_eq(d.efc.aref.numpy()[0, :nefc], mjd.efc_aref[:nefc], "efc_aref")
    if nworld == 2:
      self.assertFalse(np.allclose(d.efc.aref.numpy()[0, :nefc], d.efc.aref.numpy()[1, :nefc]))

  @parameterized.product(
    nworld=(1, 2),
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
    has_flex=(False, True),
  )
  def test_contact_partial_njmax_overflow(self, nworld, cone, jacobian, has_flex):
    """Rejects contacts straddling njmax as a whole and clears unused efc rows."""
    flex_xml = (
      """
          <flexcomp name="f1" type="grid" count="2 2 1" spacing="0.1 0.1 0.1" pos="0 0 5" radius="0.01" dim="2" mass="0.1">
            <contact selfcollide="none" contype="0" conaffinity="0"/>
          </flexcomp>
      """
      if has_flex
      else ""
    )
    mjm, _, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <option>
          <flag sleep="enable"/>
        </option>
        <worldbody>
          <geom type="plane" size="5 5 0.1"/>
          <body name="b1" pos="0 0 1">
            <freejoint/>
            <geom size="0.1" pos="-0.2 0 0"/>
            <geom size="0.1" pos="0.2 0 0"/>
          </body>
          {flex_xml}
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      njmax=5,
      # Reserve enough sparse nnz so parallel contacts overflow njmax rather than njmax_nnz.
      njmax_nnz=200,
      overrides={"opt.cone": cone, "opt.jacobian": jacobian},
    )

    qpos = d.qpos.numpy()
    qpos[:, 2] = 0.05 if nworld == 1 else [0.05, 0.04]
    d.qpos.assign(qpos)

    mjw.kinematics(m, d)
    mjw.com_pos(m, d)
    mjw.crb(m, d)
    mjw.collision(m, d)

    d.efc.D.fill_(wp.inf)
    d.efc.aref.fill_(wp.inf)
    d.efc.type.fill_(99)
    d.efc.id.fill_(99)
    if m.is_sparse:
      d.efc.J_rownnz.fill_(99)
    else:
      d.efc.J.fill_(wp.inf)

    mjw.make_constraint(m, d)

    ndim = 3 if cone == ConeType.ELLIPTIC else 4
    self.assertTrue(np.all(d.nefc.numpy() == 2 * ndim))

    # Exactly one contact per world fits; the straddling contact is rejected across all rows.
    efc_addr = d.contact.efc_address.numpy()[: 2 * nworld, :ndim]
    valid_mask = efc_addr[:, 0] >= 0
    rejected_mask = efc_addr[:, 0] == -1
    self.assertEqual(np.sum(valid_mask), nworld)
    self.assertEqual(np.sum(rejected_mask), nworld)
    np.testing.assert_array_equal(efc_addr[valid_mask], np.tile(np.arange(ndim, dtype=np.int32), (nworld, 1)))
    self.assertTrue(np.all(efc_addr[rejected_mask] == -1))

    # Straddling rows [ndim, njmax) must be cleared rather than retaining stale values.
    tail = slice(ndim, d.njmax)
    efc_aref = d.efc.aref.numpy()
    self.assertTrue(np.all(d.efc.D.numpy()[:, tail] == 0.0))
    self.assertTrue(np.all(efc_aref[:, tail] == 0.0))
    self.assertTrue(np.all(d.efc.type.numpy()[:, tail] == -1))
    self.assertTrue(np.all(d.efc.id.numpy()[:, tail] == -1))
    if m.is_sparse:
      self.assertTrue(np.all((d.overflow.numpy() & int(mjw.OverflowType.NJMAX_NNZ)) == 0))
      self.assertTrue(np.all(d.efc.J_rownnz.numpy()[:, tail] == 0))
      self.assertTrue(np.all(d.efc.jtdaj_nblock.numpy() == 1))
      self.assertTrue(np.all(d.efc.jtdaj_nrow.numpy()[:, 0] == ndim))
    else:
      self.assertTrue(np.all(d.efc.J.numpy()[:, tail, : m.nv] == 0.0))

    # Verify host transfer compacts rejected contacts and remaps island maps cleanly.
    mjw.forward(m, d)
    mjd_out = mujoco.MjData(mjm)
    for w in range(nworld):
      mjw.get_data_into(mjd_out, mjm, d, world_id=w)
      self.assertEqual(mjd_out.nefc, ndim)
      self.assertEqual(np.sum(mjd_out.contact.efc_address == -1), 1)
      if has_flex:
        self.assertEqual(mjd_out.nisland, 1)
        self.assertEqual(int(mjd_out.island_nefc[0]), ndim)
        np.testing.assert_array_equal(mjd_out.map_efc2iefc[mjd_out.map_iefc2efc[:ndim]], np.arange(ndim, dtype=np.int32))
        np.testing.assert_array_equal(mjd_out.map_iefc2efc[mjd_out.map_efc2iefc[:ndim]], np.arange(ndim, dtype=np.int32))

    if nworld == 2:
      self.assertFalse(np.allclose(efc_aref[0, :ndim], efc_aref[1, :ndim]))

  @parameterized.product(
    nworld=(1, 2),
    jacobian=(mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE),
  )
  def test_regularize_constraint_njmax_overflow(self, nworld, jacobian):
    """regularize_constraint stays within njmax and skips cleared rows when nefc > njmax."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.01"/>
        <worldbody>
          <geom type="plane" size="5 5 0.1"/>
          <body pos="0 0 1">
            <joint name="jx" type="slide" axis="1 0 0" stiffness="10" damping="0.5"/>
            <joint name="jz" type="slide" axis="0 0 1" stiffness="500" damping="20"/>
            <geom size="0.1" pos="-0.2 0 0" mass="1"/>
            <geom size="0.1" pos="0.2 0 0" mass="1"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
      njmax=5,
      njmax_nnz=32,
      overrides={"opt.jacobian": jacobian},
    )

    qpos = d.qpos.numpy()
    qpos[:, 1] = -0.95
    if nworld == 2:
      qpos[1, 1] = -0.96
      qvel = d.qvel.numpy()
      qvel[1] = [0.2, -0.3]
      d.qvel.assign(qvel)
    d.qpos.assign(qpos)

    mjw.fwd_position(m, d)
    mjw.fwd_velocity(m, d)

    self.assertTrue(np.all(d.nefc.numpy() == 8))
    self.assertGreater(d.efc.D.shape[1], d.njmax)

    # Sentinel in padded efc.D slots past njmax must remain untouched.
    efc_D_before = d.efc.D.numpy()
    efc_D_before[:, d.njmax :] = 1234.0
    d.efc.D.assign(efc_D_before)

    mjw.fwd_actuation(m, d)

    efc_D = d.efc.D.numpy()
    efc_aref = d.efc.aref.numpy()
    self.assertTrue(np.all(np.isfinite(efc_D[:, :4]) & (efc_D[:, :4] > 0.0)))
    self.assertTrue(np.all(np.isfinite(efc_aref[:, :4])))
    self.assertTrue(np.all(efc_D[:, 4] == 0.0))
    self.assertTrue(np.all(efc_aref[:, 4] == 0.0))
    self.assertTrue(np.all(efc_D[:, d.njmax :] == 1234.0))
    if nworld == 2:
      self.assertFalse(np.allclose(efc_aref[0, :4], efc_aref[1, :4]))


if __name__ == "__main__":
  absltest.main()
