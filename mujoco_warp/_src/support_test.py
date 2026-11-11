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

"""Tests for support functions."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjwarp
from mujoco_warp import ConeType
from mujoco_warp import State
from mujoco_warp import test_data
from mujoco_warp._src import io
from mujoco_warp._src import support
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_augmented_factorize_solve_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_augmented_factorize_solve_newton_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_factorize_solve_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_newton_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_sparse_tiles_factorize_solve_newton_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_sparse_tiles_solve_newton_func

# tolerance for difference between MuJoCo and MJWarp support calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SupportTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    io.ENABLE_ISLANDS = True

  def tearDown(self):
    io.ENABLE_ISLANDS = False
    super().tearDown()

  @parameterized.parameters(mujoco.mjtJacobian.mjJAC_SPARSE, mujoco.mjtJacobian.mjJAC_DENSE)
  def test_mul_m(self, jacobian):
    """Tests mul_m."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml", overrides={"opt.jacobian": jacobian})

    mj_res = np.zeros(mjm.nv)
    mj_vec = np.random.uniform(low=-1.0, high=1.0, size=mjm.nv)
    mujoco.mj_mulM(mjm, mjd, mj_res, mj_vec)

    res = wp.zeros((1, mjm.nv), dtype=wp.float32)
    vec = wp.from_numpy(np.expand_dims(mj_vec, axis=0), dtype=wp.float32)
    mjwarp.mul_m(m, d, res, vec)

    _assert_eq(res.numpy()[0], mj_res, f"mul_m ({jacobian})")

  def test_xfrc_accumulated(self):
    """Tests that xfrc_accumulate output matches mj_xfrcAccumulate."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml")
    xfrc = np.random.randn(*d.xfrc_applied.numpy().shape)
    d.xfrc_applied = wp.from_numpy(xfrc, dtype=wp.spatial_vector)
    qfrc = wp.zeros((1, mjm.nv), dtype=wp.float32)
    mjwarp.xfrc_accumulate(m, d, qfrc)

    qfrc_expected = np.zeros(m.nv)
    xfrc = xfrc[0]
    for i in range(1, m.nbody):
      mujoco.mj_applyFT(mjm, mjd, xfrc[i, :3], xfrc[i, 3:], mjd.xipos[i], i, qfrc_expected)
    np.testing.assert_almost_equal(qfrc.numpy()[0], qfrc_expected, 6)

  @parameterized.product(nworld=(1, 2), flg_add=(False, True), zero_force=(False, True), sparse=(False, True))
  def test_apply_ft_subtrees(self, nworld, flg_add, zero_force, sparse):
    """Tests force projection across branches, welded bodies, and independent trees."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <default>
          <geom size="0.1" contype="0" conaffinity="0"/>
        </default>
        <worldbody>
          <body name="root">
            <freejoint/>
            <geom/>
            <body pos="0.3 0 0">
              <joint type="ball"/>
              <geom/>
              <body pos="0.2 0 0">
                <geom/>
              </body>
            </body>
            <body name="sibling" pos="0 0.3 0">
              <joint axis="1 0 0"/>
              <geom/>
              <body pos="0 0.2 0">
                <joint type="slide"/>
                <geom/>
              </body>
            </body>
          </body>
          <body name="other" pos="1 0 0">
            <freejoint/>
            <geom/>
            <body pos="0 0 0.3">
              <joint/>
              <geom/>
            </body>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    qpos = np.tile(mjd.qpos, (nworld, 1))
    if nworld == 2:
      mujoco.mj_integratePos(mjm, qpos[1], np.linspace(-0.3, 0.4, mjm.nv), 0.2)
    d.qpos.assign(qpos)
    mjwarp.fwd_position(m, d)
    force = np.linspace(-0.7, 0.9, nworld * mjm.nbody * 6).reshape(nworld, mjm.nbody, 6).astype(np.float32)
    # A zero-force body must neither end its own subtree nor hide a subtree boundary.
    for name in ("root", "sibling", "other"):
      force[:, mjm.body(name).id] = 0.0
    if zero_force:
      force[:] = 0.0
    d.xfrc_applied.assign(force)
    initial = np.linspace(-0.2, 0.3, nworld * mjm.nv).reshape(nworld, mjm.nv).astype(np.float32)
    qfrc = wp.array(initial, dtype=float) if flg_add else wp.full((nworld, mjm.nv), wp.inf, dtype=float)
    support.apply_ft(m, d, d.xfrc_applied, qfrc, flg_add, sparse=sparse)

    result = qfrc.numpy()
    for world in range(nworld):
      mjd.qpos[:] = qpos[world]
      mujoco.mj_forward(mjm, mjd)
      expected = initial[world].astype(float) if flg_add else np.zeros(mjm.nv)
      for body in range(1, mjm.nbody):
        mujoco.mj_applyFT(mjm, mjd, force[world, body, :3], force[world, body, 3:], mjd.xipos[body], body, expected)
      _assert_eq(result[world], expected, f"apply_ft_world_{world}")
    if nworld == 2 and (flg_add or not zero_force):
      self.assertFalse(np.allclose(result[0], result[1]))

  @parameterized.parameters(
    (ConeType.PYRAMIDAL, 1, False),
    (ConeType.PYRAMIDAL, 3, False),
    (ConeType.PYRAMIDAL, 4, False),
    (ConeType.PYRAMIDAL, 6, False),
    (ConeType.PYRAMIDAL, 1, True),
    (ConeType.PYRAMIDAL, 3, True),
    (ConeType.PYRAMIDAL, 4, True),
    (ConeType.PYRAMIDAL, 6, True),
    (ConeType.ELLIPTIC, 1, False),
    (ConeType.ELLIPTIC, 3, False),
    (ConeType.ELLIPTIC, 4, False),
    (ConeType.ELLIPTIC, 6, False),
    (ConeType.ELLIPTIC, 1, True),
    (ConeType.ELLIPTIC, 3, True),
    (ConeType.ELLIPTIC, 4, True),
    (ConeType.ELLIPTIC, 6, True),
  )
  def test_contact_force(self, cone, condim, to_world_frame):
    _CONTACT = f"""
      <mujoco>
        <worldbody>
          <geom type="plane" size="10 10 .001"/>
          <body pos="0 0 1">
            <freejoint/>
            <geom fromto="-.4 0 0 .4 0 0" size=".05 .1" type="capsule" condim="{condim}" friction="1 1 1"/>
          </body>
        </worldbody>
        <keyframe>
          <key qpos="0 0 0.04 1 0 0 0" qvel="-1 -1 -1 .1 .1 .1"/>
        </keyframe>
      </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=_CONTACT, keyframe=0, overrides={"opt.cone": cone})

    mj_force = np.zeros(6, dtype=float)
    mujoco.mj_contactForce(mjm, mjd, 0, mj_force)

    contact_ids = wp.zeros(1, dtype=int)
    force = wp.zeros(1, dtype=wp.spatial_vector)

    mjwarp.contact_force(m, d, contact_ids, to_world_frame, force)

    if to_world_frame:
      frame = mjd.contact.frame[0].reshape((3, 3))
      mj_force = np.concatenate([frame.T @ mj_force[:3], frame.T @ mj_force[3:]])

    _assert_eq(force.numpy()[0], mj_force, "contact force")

  @parameterized.product(
    cone=(ConeType.PYRAMIDAL, ConeType.ELLIPTIC),
    condim=(1, 3, 4, 6),
    to_world_frame=(False, True),
  )
  def test_contact_force_adhesion(self, cone, condim, to_world_frame):
    """Tests contact_force reporting for adhesive contacts."""
    mjm, mjd, m, d = test_data.fixture(
      xml=f"""
      <mujoco>
        <worldbody>
          <geom type="plane" size="10 10 .001"/>
          <body pos="0 0 1">
            <freejoint/>
            <geom fromto="-.4 0 0 .4 0 0" size=".05 .1" type="capsule" condim="{condim}" friction="1 1 1" adhesion="12.0"/>
          </body>
        </worldbody>
        <keyframe>
          <key qpos="0 0 0.04 1 0 0 0" qvel="-1 -1 -1 .1 .1 .1"/>
        </keyframe>
      </mujoco>
      """,
      keyframe=0,
      overrides={"opt.cone": cone},
    )

    mj_force = np.zeros(6, dtype=float)
    mujoco.mj_contactForce(mjm, mjd, 0, mj_force)

    contact_ids = wp.zeros(1, dtype=int)
    force = wp.zeros(1, dtype=wp.spatial_vector)
    force.fill_(wp.spatial_vector(wp.inf, wp.inf, wp.inf, wp.inf, wp.inf, wp.inf))

    mjwarp.contact_force(m, d, contact_ids, to_world_frame, force)

    if to_world_frame:
      frame = mjd.contact.frame[0].reshape((3, 3))
      mj_force = np.concatenate([frame.T @ mj_force[:3], frame.T @ mj_force[3:]])

    _assert_eq(force.numpy()[0], mj_force, "contact force with adhesion")

  @parameterized.parameters("constraints.xml", "pendula.xml")
  def test_get_state(self, xml):
    mjm, mjd, m, d = test_data.fixture(xml, keyframe=0, ctrl_noise=1.0, qfrc_noise=1.0, xfrc_noise=1.0, mocap_noise=1.0)

    size = mujoco.mj_stateSize(mjm, mujoco.mjtState.mjSTATE_INTEGRATION)

    mj_state = np.zeros(size)
    mujoco.mj_getState(mjm, mjd, mj_state, mujoco.mjtState.mjSTATE_INTEGRATION)

    mjw_state = wp.zeros((d.nworld, size), dtype=float)
    mjwarp.get_state(m, d, mjw_state, State.INTEGRATION)

    _assert_eq(mjw_state.numpy()[0], mj_state, "state")

    d2 = mjwarp.put_data(mjm, mjd, nworld=2)
    mjw_state2 = wp.zeros((d2.nworld, size), dtype=float)
    active = wp.array([False, True], dtype=bool)
    mjwarp.get_state(m, d2, mjw_state2, State.INTEGRATION, active=active)

    _assert_eq(mjw_state2.numpy()[0], 0, "state0")
    _assert_eq(mjw_state2.numpy()[1], mj_state, "state1")

  @parameterized.parameters("constraints.xml", "pendula.xml")
  def test_set_state(self, xml):
    mjm, mjd, m, d = test_data.fixture(xml, keyframe=0, ctrl_noise=1.0, qfrc_noise=1.0, xfrc_noise=1.0, mocap_noise=1.0)

    size = mujoco.mj_stateSize(mjm, mujoco.mjtState.mjSTATE_INTEGRATION)

    mj_state = np.zeros(size)
    mujoco.mj_getState(mjm, mjd, mj_state, mujoco.mjtState.mjSTATE_INTEGRATION)

    mjw_state = wp.array(np.expand_dims(mj_state, axis=0), dtype=float)
    mjwarp.set_state(m, d, mjw_state, State.INTEGRATION)

    state_integration = [
      "time",
      "qpos",
      "qvel",
      "act",
      "qacc_warmstart",
      "ctrl",
      "qfrc_applied",
      "xfrc_applied",
      "eq_active",
      "mocap_pos",
      "mocap_quat",
    ]

    for field in state_integration:
      _assert_eq(getattr(d, field).numpy()[0], getattr(mjd, field), field)

    d2 = mjwarp.make_data(mjm, nworld=2)
    mjw_state2 = wp.array(np.tile(mj_state, (2, 1)), dtype=float)
    active = wp.array([False, True], dtype=bool)
    mjwarp.set_state(m, d2, mjw_state2, State.INTEGRATION, active=active)

    for field in state_integration:
      _assert_eq(getattr(d2, field).numpy()[1], getattr(mjd, field), field)

  def test_block_cholesky(self):
    """Tests block Cholesky decomposition and solve against numpy using n_humanoid model."""
    mjm, mjd, m, d = test_data.fixture("humanoid/n_humanoid.xml")

    # Add noise and initialize
    np.random.seed(42)
    mjd.qpos[:] += np.random.uniform(-0.1, 0.1, mjm.nq)
    mjd.qvel[:] += np.random.uniform(-0.1, 0.1, mjm.nv)
    mujoco.mj_step(mjm, mjd, 10)
    mujoco.mj_forward(mjm, mjd)

    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd, nworld=1)

    # Run forward to populate everything including efc.h
    mjwarp.forward(m, d)

    # Get the constraint Hessian matrix size
    nv = m.nv
    nv_pad = m.nv_pad
    nworld = d.nworld

    @wp.kernel(module="unique", enable_backward=False)
    def cholesky_kernel(
      grad_in: wp.array3d[float],
      h_in: wp.array3d[float],
      done_in: wp.array[bool],
      hfactor_in: wp.array3d[float],
      Mgrad_out: wp.array3d[float],
    ):
      worldid = wp.tid()
      TILE_SIZE = wp.static(16)

      if done_in[worldid]:
        return

      wp.static(create_blocked_cholesky_factorize_solve_func(TILE_SIZE, nv_pad))(
        h_in[worldid], grad_in[worldid], nv_pad, hfactor_in[worldid], Mgrad_out[worldid]
      )

    @wp.kernel(module="unique", enable_backward=False)
    def fused_cholesky_kernel(
      grad_in: wp.array3d[float],
      h_in: wp.array3d[float],
      done_in: wp.array[bool],
      hfactor_in: wp.array3d[float],
      search_out: wp.array3d[float],
      sums_out: wp.array[wp.vec2],
    ):
      worldid = wp.tid()
      TILE_SIZE = wp.static(16)

      if done_in[worldid]:
        return

      sums = wp.static(create_blocked_cholesky_augmented_factorize_solve_newton_func(TILE_SIZE, nv_pad, nv))(
        h_in[worldid],
        grad_in[worldid],
        nv_pad,
        hfactor_in[worldid],
        search_out[worldid],
      )
      sums_out[worldid] = sums

    @wp.kernel(module="unique", enable_backward=False)
    def cholesky_solve_kernel(
      grad_in: wp.array3d[float],
      hfactor_in: wp.array3d[float],
      search_out: wp.array3d[float],
      sums_out: wp.array[wp.vec2],
    ):
      worldid = wp.tid()
      TILE_SIZE = wp.static(16)

      sums = wp.static(create_blocked_cholesky_solve_newton_func(TILE_SIZE, nv_pad, nv))(
        hfactor_in[worldid],
        grad_in[worldid],
        nv_pad,
        search_out[worldid],
      )
      sums_out[worldid] = sums

    @wp.kernel(module="unique", enable_backward=False)
    def augmented_cholesky_kernel(
      grad_in: wp.array3d[float],
      h_in: wp.array3d[float],
      hfactor_in: wp.array3d[float],
      Mgrad_out: wp.array3d[float],
    ):
      worldid = wp.tid()
      TILE_SIZE = wp.static(16)

      wp.static(create_blocked_cholesky_augmented_factorize_solve_func(TILE_SIZE, nv_pad))(
        h_in[worldid], grad_in[worldid], nv_pad, hfactor_in[worldid], Mgrad_out[worldid]
      )

    @wp.kernel(module="unique", enable_backward=False)
    def augmented_solve_kernel(
      grad_in: wp.array3d[float],
      hfactor_in: wp.array3d[float],
      Mgrad_out: wp.array3d[float],
    ):
      worldid = wp.tid()
      TILE_SIZE = wp.static(16)

      wp.static(create_blocked_cholesky_solve_func(TILE_SIZE, nv_pad))(
        hfactor_in[worldid], grad_in[worldid], nv_pad, Mgrad_out[worldid]
      )

    # Create test vector and fill the built-in arrays
    b = np.random.randn(nv).astype(np.float32)

    # 1. Generate a random SPD matrix for active region
    A = np.random.randn(nv, nv).astype(np.float32)
    SPD_active_hessian = A @ A.T + nv * np.eye(nv, dtype=A.dtype)  # Make symmetric & strongly PD

    # 2. Copy the active SPD region into h_np[0, :nv, :nv]
    h_np = np.zeros((nworld, nv_pad, nv_pad), dtype=float)
    h_np[0, :nv, :nv] = SPD_active_hessian

    # Add identity to the padding region
    padding_size = nv_pad - nv
    if padding_size > 0:
      h_np[0, nv:, nv:] = np.eye(padding_size, dtype=np.float32)
    h = wp.array(h_np, dtype=float)

    # 3. Create inline arrays for grad and done (no longer in d.efc)
    grad_np = np.zeros((nworld, nv_pad))
    grad_np[0, :nv] = b
    grad = wp.array(grad_np, dtype=float)

    # Zero out the temporary arrays to ensure clean state
    L_init = np.zeros((nworld, nv_pad, nv_pad), dtype=np.float32)
    # Initialize padding region to identity
    L_init[0, nv:, nv:] = np.eye(nv_pad - nv, dtype=np.float32)

    hfactor = wp.array(L_init, dtype=float)
    Mgrad = wp.zeros((nworld, nv_pad), dtype=float)
    hfactor_fused = wp.array(L_init, dtype=float)

    search_fused = wp.zeros((nworld, nv), dtype=float)
    sums_fused = wp.zeros(nworld, dtype=wp.vec2)
    search_solve = wp.zeros((nworld, nv), dtype=float)
    sums_solve = wp.zeros(nworld, dtype=wp.vec2)

    # Ensure done is False so kernel executes
    done = wp.zeros((nworld,), dtype=bool)

    wp.launch_tiled(
      cholesky_kernel,
      dim=nworld,
      inputs=[grad.reshape(shape=(nworld, nv_pad, 1)), h, done, hfactor],
      outputs=[Mgrad.reshape(shape=(nworld, nv_pad, 1))],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
    wp.launch_tiled(
      fused_cholesky_kernel,
      dim=nworld,
      inputs=[grad.reshape(shape=(nworld, nv_pad, 1)), h, done, hfactor_fused],
      outputs=[
        search_fused.reshape(shape=(nworld, nv, 1)),
        sums_fused,
      ],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
    wp.launch_tiled(
      cholesky_solve_kernel,
      dim=nworld,
      inputs=[grad.reshape(shape=(nworld, nv_pad, 1)), hfactor_fused],
      outputs=[
        search_solve.reshape(shape=(nworld, nv, 1)),
        sums_solve,
      ],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
    U_result = hfactor.numpy()[0]
    x_result = Mgrad.numpy()[0]
    U_fused_result = hfactor_fused.numpy()[0]
    search_result = search_fused.numpy()[0]
    sums_result = sums_fused.numpy()[0]
    search_solve_result = search_solve.numpy()[0]
    sums_solve_result = sums_solve.numpy()[0]

    # Verify padding outside active region doesn't affect active computation
    # Off-diagonal padding should be zero (active region shouldn't touch padding)
    np.testing.assert_array_equal(
      U_result[nv:, :nv],
      0.0,
      err_msg="padding rows in active region were overwritten",
    )
    np.testing.assert_array_equal(
      U_result[:nv, nv:],
      0.0,
      err_msg="padding columns in active region were overwritten",
    )

    # Check that padding region remains identity after factorization
    padding_size = nv_pad - nv
    if padding_size > 0:
      padding_square = U_result[nv:, nv:]
      expected_identity = np.eye(padding_size, dtype=np.float32)
      np.testing.assert_allclose(
        padding_square,
        expected_identity,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Padding region should remain identity after factorization",
      )

    # Compare with numpy cholesky on symmetrized Hessian
    U_numpy = np.linalg.cholesky(SPD_active_hessian).T
    np.testing.assert_allclose(
      U_result[:nv, :nv],
      U_numpy,
      rtol=1e-4,
      atol=1e-4,
      err_msg="Cholesky decomposition mismatch with numpy",
    )

    # Verify U.T @ U = A (symmetrized Hessian)
    A_reconstructed = U_result[:nv, :nv].T @ U_result[:nv, :nv]
    np.testing.assert_allclose(
      A_reconstructed,
      SPD_active_hessian,
      rtol=1e-5,
      atol=1e-5,
      err_msg="U.T @ U does not equal symmetrized Hessian",
    )

    # Verify the search direction against the symmetrized Hessian solution.
    x_numpy = np.linalg.solve(SPD_active_hessian, b)
    np.testing.assert_allclose(x_result[:nv], x_numpy, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(U_fused_result[:nv, :nv], U_numpy, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(search_result, -x_numpy, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(sums_result, [x_numpy @ x_numpy, b @ x_numpy], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(search_solve_result, -x_numpy, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(sums_solve_result, [x_numpy @ x_numpy, b @ x_numpy], rtol=1e-3, atol=1e-3)

    hfactor_augmented = wp.array(L_init, dtype=float)
    Mgrad_augmented = wp.zeros((nworld, nv_pad), dtype=float)
    wp.launch_tiled(
      augmented_cholesky_kernel,
      dim=nworld,
      inputs=[grad.reshape(shape=(nworld, nv_pad, 1)), h, hfactor_augmented],
      outputs=[Mgrad_augmented.reshape(shape=(nworld, nv_pad, 1))],
      block_dim=m.block_dim.update_gradient_cholesky,
    )

    solve_b = np.random.randn(nv).astype(np.float32)
    solve_grad_np = np.zeros((nworld, nv_pad))
    solve_grad_np[0, :nv] = solve_b
    solve_grad = wp.array(solve_grad_np, dtype=float)
    Mgrad_solve = wp.zeros((nworld, nv_pad), dtype=float)
    wp.launch_tiled(
      augmented_solve_kernel,
      dim=nworld,
      inputs=[solve_grad.reshape(shape=(nworld, nv_pad, 1)), hfactor_augmented],
      outputs=[Mgrad_solve.reshape(shape=(nworld, nv_pad, 1))],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
    np.testing.assert_allclose(hfactor_augmented.numpy()[0, :nv, :nv], U_numpy, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(Mgrad_augmented.numpy()[0, :nv], x_numpy, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(Mgrad_solve.numpy()[0, :nv], np.linalg.solve(SPD_active_hessian, solve_b), rtol=1e-3, atol=1e-3)

  @parameterized.parameters(7, 66)
  def test_tiled_cholesky_sparse_tiles(self, ntiles):
    """Tests reverse sparse-tile blocked Cholesky factorization and solve."""
    rng = np.random.default_rng(123 + ntiles)
    nworld = 1
    tile_size = 16
    nv = ntiles * tile_size - 6
    nv_pad = ntiles * tile_size
    nmask_words = (ntiles + 63) // 64
    npairs = ntiles * (ntiles + 1) // 2

    if ntiles == 7:
      # Single 64-bit word (nwords = 1):
      # - Tiles 0 and 4 are purely diagonal
      # - Tiles (1, 3) and (3, 5) have off-diagonal coupling
      dense_diag_tiles = [1, 2, 3, 5, 6]
      off_diag_pairs = [(1, 3), (3, 5)]
    else:
      # Multi-word 64-bit masks (ntiles = 66, nwords = 2):
      # - Couplings across the bit-64 boundary: (1, 64), (63, 64), (2, 65), (35, 65), (64, 65)
      # - Eliminating tk = 65 induces cross-word fill-in (w_r = 0 < w_i = 1) at (2, 64) and
      #   (35, 64), and intra-word fill-in at (2, 35)
      # - Eliminating tk = 64 induces intra-word fill-in at (1, 63), (1, 2), (1, 35), (2, 63),
      #   and (35, 63)
      dense_diag_tiles = [1, 2, 10, 35, 63, 64, 65]
      off_diag_pairs = [(1, 64), (63, 64), (2, 65), (35, 65), (64, 65)]

    H_active = np.diag(rng.uniform(8.0, 12.0, size=nv).astype(np.float32))
    for t in dense_diag_tiles:
      r0 = t * tile_size
      r1 = min((t + 1) * tile_size, nv)
      sz = r1 - r0
      B = rng.standard_normal((sz, sz)).astype(np.float32) * 0.2
      H_active[r0:r1, r0:r1] += B @ B.T + 4.0 * np.eye(sz, dtype=np.float32)

    for ti, tj in off_diag_pairs:
      r0, r1 = ti * tile_size, (ti + 1) * tile_size
      c0, c1 = tj * tile_size, min((tj + 1) * tile_size, nv)
      C = rng.standard_normal((r1 - r0, c1 - c0)).astype(np.float32) * 0.1
      H_active[r0:r1, c0:c1] = C
      H_active[c0:c1, r0:r1] = C.T

    H_np = np.zeros((nworld, nv_pad, nv_pad), dtype=np.float32)
    H_np[0, :nv, :nv] = H_active
    H_np[0, nv:, nv:] = np.eye(nv_pad - nv, dtype=np.float32)

    b = rng.standard_normal(nv).astype(np.float32)
    grad_np = np.zeros((nworld, nv_pad), dtype=np.float32)
    grad_np[0, :nv] = b

    h = wp.array(H_np, dtype=float)
    grad = wp.array(grad_np, dtype=float)
    done = wp.zeros(nworld, dtype=bool)
    nefc = wp.ones(nworld, dtype=int)
    tile_nz = wp.zeros((nworld, ntiles, ntiles), dtype=int)
    row_mask = wp.zeros((nworld, ntiles, nmask_words), dtype=wp.uint64)
    col_mask = wp.zeros((nworld, ntiles, nmask_words), dtype=wp.uint64)
    tile_is_diag = wp.zeros((nworld, ntiles), dtype=int)
    rhs_work = wp.zeros((nworld, nv_pad, 1), dtype=float)
    hfactor = wp.zeros((nworld, nv_pad, nv_pad), dtype=float)

    # Run sparsity detection, reverse symbolic factorization, and diagonal RHS prep kernels
    wp.launch_tiled(
      mjwarp._src.solver._detect_h_tile_sparsity(tile_size, False, False),
      dim=(nworld, npairs),
      inputs=[done, h, nefc, nefc],
      outputs=[tile_nz],
      block_dim=32,
    )
    wp.launch(
      mjwarp._src.solver._symbolic_h_tile_cholesky(ntiles, False, False),
      dim=nworld,
      inputs=[done, tile_nz, nefc, nefc],
      outputs=[row_mask, col_mask, tile_is_diag],
    )
    wp.launch(
      mjwarp._src.solver._prepare_h_rhs_work(tile_size, nv, False, False),
      dim=(nworld, nv_pad),
      inputs=[done, grad, h, tile_is_diag, nefc, nefc],
      outputs=[hfactor, rhs_work],
    )

    if ntiles == 66:
      # Verify cross-word symbolic fill-in (from eliminating tk = 65 into (2, 64) and (35, 64))
      # and intra-word fill-in (from eliminating tk = 65 and tk = 64 into (2, 35) and (1, 63))
      row_mask_np = row_mask.numpy()[0]
      self.assertNotEqual(row_mask_np[2, 1] & np.uint64(1 << (64 - 64)), 0)
      self.assertNotEqual(row_mask_np[35, 1] & np.uint64(1 << (64 - 64)), 0)
      self.assertNotEqual(row_mask_np[2, 0] & np.uint64(1 << 35), 0)
      self.assertNotEqual(row_mask_np[1, 0] & np.uint64(1 << 63), 0)
      self.assertEqual(tile_is_diag.numpy()[0, 0], 1)

    search_factor = wp.zeros((nworld, nv), dtype=float)
    sums_factor = wp.zeros(nworld, dtype=wp.vec2)

    @wp.kernel(module="unique", enable_backward=False)
    def sparse_factorize_solve_kernel(
      h_in: wp.array3d[float],
      grad_in: wp.array3d[float],
      rhs_work_in: wp.array3d[float],
      row_mask_in: wp.array3d[wp.uint64],
      col_mask_in: wp.array3d[wp.uint64],
      tile_is_diag_in: wp.array2d[int],
      hfactor_out: wp.array3d[float],
      search_out: wp.array3d[float],
      sums_out: wp.array[wp.vec2],
    ):
      worldid = wp.tid()
      sums_out[worldid] = wp.static(create_blocked_cholesky_sparse_tiles_factorize_solve_newton_func(tile_size, nv_pad, nv))(
        h_in[worldid],
        grad_in[worldid],
        rhs_work_in[worldid],
        row_mask_in[worldid],
        col_mask_in[worldid],
        tile_is_diag_in[worldid],
        nv_pad,
        hfactor_out[worldid],
        search_out[worldid],
      )

    @wp.kernel(module="unique", enable_backward=False)
    def sparse_solve_kernel(
      hfactor_in: wp.array3d[float],
      grad_in: wp.array3d[float],
      rhs_work_in: wp.array3d[float],
      row_mask_in: wp.array3d[wp.uint64],
      col_mask_in: wp.array3d[wp.uint64],
      tile_is_diag_in: wp.array2d[int],
      search_out: wp.array3d[float],
      sums_out: wp.array[wp.vec2],
    ):
      worldid = wp.tid()
      sums_out[worldid] = wp.static(create_blocked_cholesky_sparse_tiles_solve_newton_func(tile_size, nv_pad, nv))(
        hfactor_in[worldid],
        grad_in[worldid],
        rhs_work_in[worldid],
        row_mask_in[worldid],
        col_mask_in[worldid],
        tile_is_diag_in[worldid],
        nv_pad,
        search_out[worldid],
      )

    wp.launch_tiled(
      sparse_factorize_solve_kernel,
      dim=nworld,
      inputs=[
        h,
        grad.reshape(shape=(nworld, nv_pad, 1)),
        rhs_work,
        row_mask,
        col_mask,
        tile_is_diag,
      ],
      outputs=[
        hfactor,
        search_factor.reshape(shape=(nworld, nv, 1)),
        sums_factor,
      ],
      block_dim=64,
    )

    x_expected = np.linalg.solve(H_active, b)
    np.testing.assert_allclose(search_factor.numpy()[0], -x_expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
      sums_factor.numpy()[0],
      [x_expected @ x_expected, b @ x_expected],
      rtol=1e-3,
      atol=1e-3,
    )

    # Also test cached-factor solve with a second RHS vector (skip_unchanged=True)
    b2 = rng.standard_normal(nv).astype(np.float32)
    grad2_np = np.zeros((nworld, nv_pad), dtype=np.float32)
    grad2_np[0, :nv] = b2
    grad2 = wp.array(grad2_np, dtype=float)
    zero_changed = wp.zeros(nworld, dtype=int)
    wp.launch(
      mjwarp._src.solver._prepare_h_rhs_work(tile_size, nv, True, False),
      dim=(nworld, nv_pad),
      inputs=[done, grad2, h, tile_is_diag, zero_changed, nefc],
      outputs=[hfactor, rhs_work],
    )
    search_solve = wp.zeros((nworld, nv), dtype=float)
    sums_solve = wp.zeros(nworld, dtype=wp.vec2)
    wp.launch_tiled(
      sparse_solve_kernel,
      dim=nworld,
      inputs=[
        hfactor,
        grad2.reshape(shape=(nworld, nv_pad, 1)),
        rhs_work,
        row_mask,
        col_mask,
        tile_is_diag,
      ],
      outputs=[
        search_solve.reshape(shape=(nworld, nv, 1)),
        sums_solve,
      ],
      block_dim=64,
    )
    x2_expected = np.linalg.solve(H_active, b2)
    np.testing.assert_allclose(search_solve.numpy()[0], -x2_expected, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
      sums_solve.numpy()[0],
      [x2_expected @ x2_expected, b2 @ x2_expected],
      rtol=1e-3,
      atol=1e-3,
    )

  @parameterized.parameters(
    ("pendula.xml", 1),
    ("pendula.xml", 2),
    ("humanoid/n_humanoid.xml", 1),
    ("humanoid/n_humanoid.xml", 5),
    ("constraints.xml", 1),
  )
  def test_jac(self, xml, bodyid):
    """Tests jac against mj_jac."""
    mjm, mjd, m, d = test_data.fixture(xml)

    point = mjd.xipos[bodyid]

    jacp_mj = np.zeros((3, mjm.nv))
    jacr_mj = np.zeros((3, mjm.nv))
    mujoco.mj_jac(mjm, mjd, jacp_mj, jacr_mj, point, bodyid)

    point_wp = wp.array([point], dtype=wp.vec3)
    bodyid_wp = wp.array([bodyid], dtype=int)
    jacp_wp = wp.zeros((1, 3, mjm.nv), dtype=float)
    jacr_wp = wp.zeros((1, 3, mjm.nv), dtype=float)

    mjwarp.jac(m, d, jacp_wp, jacr_wp, point_wp, bodyid_wp)

    _assert_eq(jacp_wp.numpy()[0], jacp_mj, f"jacp ({xml}, body {bodyid})")
    _assert_eq(jacr_wp.numpy()[0], jacr_mj, f"jacr ({xml}, body {bodyid})")

  def test_jac_optional_outputs(self):
    """Tests jac with None outputs."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml")

    point = mjd.xipos[1]
    point_wp = wp.array([point], dtype=wp.vec3)
    bodyid_wp = wp.array([1], dtype=int)

    jacp_wp = wp.zeros((1, 3, mjm.nv), dtype=float)
    jacr_wp = wp.zeros((1, 3, mjm.nv), dtype=float)

    jacp_mj = np.zeros((3, mjm.nv))
    jacr_mj = np.zeros((3, mjm.nv))
    mujoco.mj_jac(mjm, mjd, jacp_mj, jacr_mj, point, 1)

    mjwarp.jac(m, d, jacp_wp, None, point_wp, bodyid_wp)
    _assert_eq(jacp_wp.numpy()[0], jacp_mj, "jacp (optional jacr)")

    mjwarp.jac(m, d, None, jacr_wp, point_wp, bodyid_wp)
    _assert_eq(jacr_wp.numpy()[0], jacr_mj, "jacr (optional jacp)")

    mjwarp.jac(m, d, None, None, point_wp, bodyid_wp)

  def test_jac_nworld(self):
    """Tests jac with multiple worlds."""
    mjm, mjd, m, d = test_data.fixture("pendula.xml", nworld=2)

    bodyid = 1
    point = mjd.xipos[bodyid]

    jacp_mj = np.zeros((3, mjm.nv))
    jacr_mj = np.zeros((3, mjm.nv))
    mujoco.mj_jac(mjm, mjd, jacp_mj, jacr_mj, point, bodyid)

    point_wp = wp.array([point, point], dtype=wp.vec3)
    bodyid_wp = wp.array([bodyid, bodyid], dtype=int)
    jacp_wp = wp.zeros((2, 3, mjm.nv), dtype=float)
    jacr_wp = wp.zeros((2, 3, mjm.nv), dtype=float)

    mjwarp.jac(m, d, jacp_wp, jacr_wp, point_wp, bodyid_wp)

    for w in range(2):
      _assert_eq(jacp_wp.numpy()[w], jacp_mj, f"jacp world {w}")
      _assert_eq(jacr_wp.numpy()[w], jacr_mj, f"jacr world {w}")

  def test_userdata_get_set_reset(self):
    # XML defining a model with nuserdata=4
    xml = """
    <mujoco>
      <size nuserdata="4"/>
      <worldbody>
        <body name="body">
          <joint type="free"/>
          <geom size="0.1" type="sphere"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm, mjd, m, d = test_data.fixture(xml=xml, nworld=2)
    self.assertEqual(m.nuserdata, 4)
    self.assertEqual(d.userdata.shape, (2, 4))

    # Initialize userdata on host MjData
    userdata_val = np.array([1.2, 3.4, 5.6, 7.8])
    mjd.userdata[:] = userdata_val

    # Copy to device and verify
    d = mjwarp.put_data(mjm, mjd, nworld=2)
    _assert_eq(d.userdata.numpy()[0], userdata_val, "put_data userdata")

    # Serialize state with State.INTEGRATION (which includes State.USERDATA)
    size = mujoco.mj_stateSize(mjm, mujoco.mjtState.mjSTATE_INTEGRATION)
    mjw_state = wp.zeros((d.nworld, size), dtype=float)
    mjwarp.get_state(m, d, mjw_state, State.INTEGRATION)

    # Deserialize state using set_state on a fresh Data object
    d2 = mjwarp.make_data(mjm, nworld=2)
    mjwarp.set_state(m, d2, mjw_state, State.INTEGRATION)
    _assert_eq(d2.userdata.numpy()[0], userdata_val, "set_state userdata")

    # Copy back to host and verify via get_data_into
    mjd2 = mujoco.MjData(mjm)
    mjwarp.get_data_into(mjd2, mjm, d2, world_id=0)
    _assert_eq(mjd2.userdata, userdata_val, "get_data_into userdata")

    # Reset data and verify userdata is zeroed
    mjwarp.reset_data(m, d2)
    _assert_eq(d2.userdata.numpy()[0], np.zeros(4), "reset_data userdata")

  @parameterized.parameters(
    # axis, angle, stretch
    ([1.0, 0.0, 0.0], 0.0, [1.0, 1.0, 1.0]),  # Identity
    ([1.0, 0.0, 0.0], np.pi / 2.0, [1.0, 1.0, 1.0]),  # 90 deg around X
    ([0.0, 1.0, 0.0], -np.pi / 4.0, [1.0, 1.0, 1.0]),  # -45 deg around Y
    ([0.0, 0.0, 1.0], np.pi * 0.99, [1.0, 1.0, 1.0]),  # 178.2 deg around Z
    ([1.0, 1.0, 1.0], np.pi / 3.0, [1.0, 1.0, 1.0]),  # arbitrary axis
    ([1.0, 0.0, 0.0], np.pi / 2.0, [2.0, 0.5, 1.0]),  # stretch along X and Y
    ([0.0, 1.0, 0.0], -np.pi / 4.0, [1.5, 1.5, 1.5]),  # uniform scaling
    ([1.0, 2.0, 3.0], np.pi / 6.0, [0.1, 10.0, 2.0]),  # extreme stretch
    ([1.0, 0.0, 0.0], np.pi, [1.0, 1.0, 1.0]),  # unstretched half-turn along X
    ([1.0, 0.0, 0.0], np.pi, [2.5, 1.0, 1.0]),  # stretched half-turn along X
    ([1.0, 1.0, 1.0], np.pi, [1.0, 1.0, 1.0]),  # unstretched half-turn off-axis
    ([1.0, 1.0, 1.0], np.pi, [2.5, 1.0, 1.0]),  # stretched half-turn off-axis
  )
  def test_mat33_to_quat_polar(self, axis, angle, stretch):
    # Rodrigues' rotation formula
    axis = np.array(axis, dtype=float) / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * np.dot(K, K)
    if np.isclose(angle, np.pi):
      # Align stretch along the rotation axis so that identity is a stationary point.
      U = np.eye(3) + (stretch[0] - 1.0) * np.outer(axis, axis)
    else:
      U = np.diag(stretch)
    F = np.dot(R, U)

    q_exp = np.array(
      [
        axis[0] * np.sin(angle / 2.0),
        axis[1] * np.sin(angle / 2.0),
        axis[2] * np.sin(angle / 2.0),
        np.cos(angle / 2.0),
      ]
    )

    F_wp = wp.array([wp.mat33(F.flatten())], dtype=wp.mat33)
    q_out_wp = wp.zeros(1, dtype=wp.quat)

    @wp.kernel
    def test_polar_kernel(
      F_in: wp.array[wp.mat33],
      q_out: wp.array[wp.quat],
    ):
      q_out[0] = support.mat33_to_quat_polar(F_in[0])

    wp.launch(
      test_polar_kernel,
      dim=1,
      inputs=[F_wp],
      outputs=[q_out_wp],
    )

    q_est = q_out_wp.numpy()[0]

    # Align signs (antipodal symmetry)
    if np.dot(q_est, q_exp) < 0.0:
      q_est = -q_est

    _assert_eq(q_est, q_exp, "quat")

  @parameterized.product(
    case=(
      # flex, contact position, closest point on the element
      ("seg", (0.3, 0.2, 0.0), (0.3, 0.0, 0.0)),
      ("seg", (1.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
      ("tri", (0.25, 0.25, 0.1), (0.25, 0.25, 0.0)),
      ("tri", (-0.5, -0.5, 0.0), (0.0, 0.0, 0.0)),
      ("tri", (0.5, -0.2, 0.1), (0.5, 0.0, 0.0)),
      ("tet", (0.1, 0.2, 0.3), (0.1, 0.2, 0.3)),
      ("tet", (0.2, 0.3, -0.05), (0.2, 0.3, 0.0)),
      ("tet", (0.5, 0.5, 0.5), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
    ),
    nworld=(1, 2),
  )
  def test_contact_bodies_and_weights_flex_elem(self, case, nworld):
    """Flex element contact weights are the barycentric coordinates of the closest point."""
    flex, pos, closest = case
    mjm, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option>
          <flag contact="disable" gravity="disable"/>
        </option>
        <worldbody>
          <flexcomp name="seg" type="direct" dim="1" radius=".01"
                    point="0 0 0  1 0 0" element="0 1"/>
          <flexcomp name="tri" type="direct" dim="2" radius=".01"
                    point="0 0 0  1 0 0  0 1 0" element="0 1 2"/>
          <flexcomp name="tet" type="direct" dim="3" radius=".01"
                    point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3"/>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    flexid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_FLEX, flex)

    # translate the flexes and the contact in world 1
    offset = np.array([0.5, -0.25, 1.0])
    flexvert_xpos = d.flexvert_xpos.numpy()
    con_pos = np.tile(np.array(pos), (nworld, 1))
    if nworld == 2:
      flexvert_xpos[1] += offset
      con_pos[1] += offset
      d.flexvert_xpos.assign(flexvert_xpos)

    body_out = wp.full(nworld, wp.vec4i(-1), dtype=wp.vec4i)
    weight_out = wp.full(nworld, wp.vec4(wp.inf), dtype=wp.vec4)

    @wp.kernel
    def contact_weights_kernel(
      # Model:
      jnt_bodyid: wp.array[int],
      geom_bodyid: wp.array[int],
      flex_dim: wp.array[int],
      flex_interp: wp.array[int],
      flex_cellnum: wp.array[wp.vec3i],
      flex_nodeadr: wp.array[int],
      flex_vertadr: wp.array[int],
      flex_elemadr: wp.array[int],
      flex_elemnum: wp.array[int],
      flex_elemdataadr: wp.array[int],
      flex_nodebodyid: wp.array[int],
      flex_vertbodyid: wp.array[int],
      flex_elem: wp.array[int],
      flex_vert0: wp.array[wp.vec3],
      flex_rod_has_twist: wp.array[bool],
      flex_rod_edgeadr: wp.array[int],
      flex_rod_edgenum: wp.array[int],
      flex_rodedge_jnt: wp.array[int],
      flex_elem_rodedge: wp.array[wp.vec2i],
      flex_vert_rodelem: wp.array[int],
      # Data in:
      flexvert_xpos_in: wp.array2d[wp.vec3],
      # In:
      flexid: int,
      con_pos: wp.array[wp.vec3],
      # Out:
      body_out: wp.array[wp.vec4i],
      weight_out: wp.array[wp.vec4],
    ):
      worldid = wp.tid()

      # contact between nothing (side 0) and element 0 of the flex (side 1)
      body, weight = support.contact_bodies_and_weights(
        jnt_bodyid,
        geom_bodyid,
        flex_dim,
        flex_interp,
        flex_cellnum,
        flex_nodeadr,
        flex_vertadr,
        flex_elemadr,
        flex_elemnum,
        flex_elemdataadr,
        flex_nodebodyid,
        flex_vertbodyid,
        flex_elem,
        flex_vert0,
        flex_rod_has_twist,
        flex_rod_edgeadr,
        flex_rod_edgenum,
        flex_rodedge_jnt,
        flex_elem_rodedge,
        flex_vert_rodelem,
        flexvert_xpos_in,
        0,
        1,
        wp.vec2i(-1, -1),
        wp.vec2i(-1, flexid),
        wp.vec2i(-1, 0),
        wp.vec2i(-1, -1),
        con_pos[worldid],
        worldid,
      )
      body_out[worldid] = body
      weight_out[worldid] = weight

    wp.launch(
      contact_weights_kernel,
      dim=nworld,
      inputs=[
        m.jnt_bodyid,
        m.geom_bodyid,
        m.flex_dim,
        m.flex_interp,
        m.flex_cellnum,
        m.flex_nodeadr,
        m.flex_vertadr,
        m.flex_elemadr,
        m.flex_elemnum,
        m.flex_elemdataadr,
        m.flex_nodebodyid,
        m.flex_vertbodyid,
        m.flex_elem,
        m.flex_vert0,
        m.flex_rod_has_twist,
        m.flex_rod_edgeadr,
        m.flex_rod_edgenum,
        m.flex_rodedge_jnt,
        m.flex_elem_rodedge,
        m.flex_vert_rodelem,
        d.flexvert_xpos,
        flexid,
        wp.array(con_pos, dtype=wp.vec3),
      ],
      outputs=[body_out, weight_out],
    )

    # weights are a partition of unity which interpolates the closest point
    dim = mjm.flex_dim[flexid]
    vertadr = mjm.flex_vertadr[flexid]
    body = body_out.numpy()
    weight = weight_out.numpy()
    for w in range(nworld):
      point = np.zeros(3)
      for k in range(dim + 1):
        vertid = np.flatnonzero(mjm.flex_vertbodyid[vertadr : vertadr + mjm.flex_vertnum[flexid]] == body[w, k])
        self.assertLen(vertid, 1)
        point += weight[w, k] * flexvert_xpos[w, vertadr + vertid[0]]
      self.assertTrue(np.all(weight[w, : dim + 1] >= -1e-6), f"negative weight in world {w}")
      _assert_eq(np.sum(weight[w]), 1.0, f"weight sum world {w}")
      _assert_eq(point, np.array(closest) + (offset if w == 1 else 0.0), f"closest point world {w}")

  def test_rod_support_lookups(self):
    """Tests precomputed 1D DER rod lookups: twist joints and incident elements."""
    if io.types.ROD_HINGE_SIZE != 10:
      self.skipTest("Requires 1D flex rod torsion support in MuJoCo")
    mjm, _, m, _ = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <!-- Shuffled element order and reversed middle element (2 -> 1) -->
          <flexcomp name="direct_rod" type="direct" dim="1" radius="0.01" mass="0.1"
                    point="0 0 0  0.1 0 0  0.2 0 0  0.3 0 0  0.4 0 0"
                    element="2 3  2 1  3 4  0 1">
            <contact selfcollide="none" contype="0" conaffinity="0"/>
            <elasticity young="1e6" damping="0.01"/>
          </flexcomp>
          <!-- Both root and tail clamped box rod -->
          <flexcomp name="clamped_box" type="grid" count="5 1 1" spacing="0.1 0.1 0.1" pos="0 0.3 0"
                    radius="0.005" dim="1" mass="0.1">
            <pin id="0 1 3 4"/>
            <contact selfcollide="none" contype="0" conaffinity="0"/>
            <elasticity young="1e6" damping="0.01" section="box" sectionsize="0.02 0.008"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
    )

    @wp.kernel
    def query_rod_kernel(
      flex_vertadr: wp.array[int],
      flex_vertnum: wp.array[int],
      flex_elemadr: wp.array[int],
      flex_elemnum: wp.array[int],
      flex_rod_has_twist: wp.array[bool],
      flex_rod_is_box: wp.array[bool],
      flex_rod_root_clamped: wp.array[bool],
      flex_rod_tail_clamped: wp.array[bool],
      flex_rod_box_size: wp.array[wp.vec2],
      flex_rod_edgeadr: wp.array[int],
      flex_rod_edgenum: wp.array[int],
      flex_rodedge_jnt: wp.array[int],
      flex_elem_rodedge: wp.array[wp.vec2i],
      flex_vert_rodelem: wp.array[int],
      flags_out: wp.array[int],
      box_out: wp.array[wp.vec2],
      elem_jnt_out: wp.array[int],
      elem_sgn_out: wp.array[float],
      vert_elem_out: wp.array[int],
      chain_jnt_out: wp.array[int],
    ):
      f = wp.tid()
      has_tw = support.rod_has_twist(flex_rod_has_twist, f)
      is_box, bsize = support.rod_box_size(flex_rod_is_box, flex_rod_box_size, f)
      rc, tc = support.rod_clamped_ends(flex_rod_root_clamped, flex_rod_tail_clamped, f)
      flags_out[3 * f + 0] = 1 if has_tw else 0
      flags_out[3 * f + 1] = 1 if is_box else 0
      flags_out[3 * f + 2] = (1 if rc else 0) + (2 if tc else 0)
      box_out[f] = bsize
      if f == 0:
        for e in range(4):
          jnt_e, sgn_e = support.rod_edge_twist_jnt(
            flex_elemadr,
            flex_elemnum,
            flex_rod_edgeadr,
            flex_rod_edgenum,
            flex_rodedge_jnt,
            flex_elem_rodedge,
            0,
            e,
          )
          elem_jnt_out[e] = jnt_e
          elem_sgn_out[e] = sgn_e
        for v in range(5):
          vert_elem_out[v] = support.rod_vert_elem(flex_vertadr, flex_vertnum, flex_vert_rodelem, 0, v)
      else:
        for c in range(4):
          chain_jnt_out[c] = support.rod_chain_twist_jnt(flex_rod_edgeadr, flex_rod_edgenum, flex_rodedge_jnt, 1, c)

    flags_out = wp.zeros(6, dtype=int)
    box_out = wp.zeros(2, dtype=wp.vec2)
    elem_jnt_out = wp.zeros(4, dtype=int)
    elem_sgn_out = wp.zeros(4, dtype=float)
    vert_elem_out = wp.zeros(5, dtype=int)
    chain_jnt_out = wp.zeros(4, dtype=int)

    wp.launch(
      query_rod_kernel,
      dim=2,
      inputs=[
        m.flex_vertadr,
        m.flex_vertnum,
        m.flex_elemadr,
        m.flex_elemnum,
        m.flex_rod_has_twist,
        m.flex_rod_is_box,
        m.flex_rod_root_clamped,
        m.flex_rod_tail_clamped,
        m.flex_rod_box_size,
        m.flex_rod_edgeadr,
        m.flex_rod_edgenum,
        m.flex_rodedge_jnt,
        m.flex_elem_rodedge,
        m.flex_vert_rodelem,
      ],
      outputs=[flags_out, box_out, elem_jnt_out, elem_sgn_out, vert_elem_out, chain_jnt_out],
    )

    flags = flags_out.numpy()
    # f=0: has_twist=1, is_box=0, rc/tc=0
    np.testing.assert_array_equal(flags[:3], [1, 0, 0])
    # f=1: has_twist=1, is_box=1, rc=1 + tc=2 -> 3
    np.testing.assert_array_equal(flags[3:], [1, 1, 3])
    _assert_eq(box_out.numpy()[1], np.array([0.02, 0.008]), "box_size")

    # Shuffled direct_rod elements: elem 0=(2,3)->chain 2, elem 1=(2,1)->chain 1 (sgn -1),
    # elem 2=(3,4)->chain 3, elem 3=(0,1)->chain 0
    expected_jnts = [
      mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "direct_rod_twist_2"),
      mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "direct_rod_twist_1"),
      mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "direct_rod_twist"),
      mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "direct_rod_twist_0"),
    ]
    np.testing.assert_array_equal(elem_jnt_out.numpy(), expected_jnts)
    _assert_eq(elem_sgn_out.numpy(), np.array([1.0, -1.0, 1.0, 1.0]), "elem_sgn")

    # Each vertex's chosen element must actually contain that vertex:
    # chain edges: c0->elem 3, c1->elem 1, c2->elem 0, c3->elem 2, v4->elem 2
    vert_elems = vert_elem_out.numpy()
    np.testing.assert_array_equal(vert_elems, [3, 1, 0, 2, 2])
    edata = mjm.flex_elem[:8].reshape(4, 2)
    for v in range(5):
      self.assertIn(v, edata[vert_elems[v]])

    # Clamped ends on f=1: chain edges 0 and 3 have jnt == -1; middle edges have twist_1, twist_2
    c_jnts = chain_jnt_out.numpy()
    self.assertEqual(c_jnts[0], -1)
    self.assertEqual(c_jnts[3], -1)
    self.assertEqual(c_jnts[1], mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "clamped_box_twist_1"))
    self.assertEqual(c_jnts[2], mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "clamped_box_twist_2"))

  def test_rod_box_elem_pose_and_no_state_roll(self):
    """Tests rod_box_elem orthonormal frame, twist rotation, and rotated-root no-state start."""
    if io.types.ROD_HINGE_SIZE != 10:
      self.skipTest("Requires 1D flex rod torsion support in MuJoCo")
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body name="root">
            <joint name="roll" type="hinge" axis="1 0 0"/>
            <geom size="0.01"/>
            <flexcomp name="strip" type="direct" dim="1" radius="0.005" mass="0.1"
                      point="0 0 0  0.1 0 0  0.2 0 0  0.3 0 0"
                      element="0 1  2 1  2 3">
              <pin id="0 1"/>
              <contact selfcollide="none" contype="0" conaffinity="0"/>
              <elasticity young="1e6" damping="0.01" section="box" sectionsize="0.02 0.006"/>
            </flexcomp>
          </body>
        </worldbody>
      </mujoco>
      """,
    )
    # Roll root body by 0.9 rad without initializing plastic state, and twist edge 2 by 0.4 rad
    qpos = d.qpos.numpy().copy()
    qpos[0, mjm.jnt_qposadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "roll")]] = 0.9
    jnt_tw2 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "strip_twist")
    qpos[0, mjm.jnt_qposadr[jnt_tw2]] = 0.4
    d.qpos.assign(qpos)
    d.plastic.zero_()
    mjwarp.kinematics(m, d)
    mjwarp.flex(m, d)

    @wp.kernel
    def box_elem_kernel(
      qpos0: wp.array2d[float],
      jnt_qposadr: wp.array[int],
      flex_vertadr: wp.array[int],
      flex_elemadr: wp.array[int],
      flex_elemnum: wp.array[int],
      flex_elemdataadr: wp.array[int],
      flex_plasticadr: wp.array[int],
      flex_vertbodyid: wp.array[int],
      flex_elem: wp.array[int],
      flex_rod_is_box: wp.array[bool],
      flex_rod_root_clamped: wp.array[bool],
      flex_rod_tail_clamped: wp.array[bool],
      flex_rod_box_size: wp.array[wp.vec2],
      flex_rod_edgeadr: wp.array[int],
      flex_rod_edgenum: wp.array[int],
      flex_rodedge_vert: wp.array[wp.vec2i],
      flex_rodedge_jnt: wp.array[int],
      flex_elem_rodedge: wp.array[wp.vec2i],
      qpos_in: wp.array2d[float],
      plastic_in: wp.array2d[float],
      xmat_in: wp.array2d[wp.mat33],
      flexvert_xpos_in: wp.array2d[wp.vec3],
      flex_rod_frame0_loc_in: wp.array3d[wp.vec3],
      flex_rod_frame0_ref_in: wp.array3d[wp.vec3],
      flex_rod_tail_loc_in: wp.array3d[wp.vec3],
      ok_out: wp.array[int],
      pos_out: wp.array[wp.vec3],
      mat_out: wp.array[wp.mat33],
      size_out: wp.array[wp.vec3],
    ):
      e = wp.tid()
      ok, pos, mat, size = support.rod_box_elem(
        qpos0,
        jnt_qposadr,
        flex_vertadr,
        flex_elemadr,
        flex_elemnum,
        flex_elemdataadr,
        flex_plasticadr,
        flex_vertbodyid,
        flex_elem,
        flex_rod_is_box,
        flex_rod_root_clamped,
        flex_rod_tail_clamped,
        flex_rod_box_size,
        flex_rod_edgeadr,
        flex_rod_edgenum,
        flex_rodedge_vert,
        flex_rodedge_jnt,
        flex_elem_rodedge,
        qpos_in,
        plastic_in,
        xmat_in,
        flexvert_xpos_in,
        flex_rod_frame0_loc_in,
        flex_rod_frame0_ref_in,
        flex_rod_tail_loc_in,
        0,
        0,
        e,
      )
      ok_out[e] = 1 if ok else 0
      pos_out[e] = pos
      mat_out[e] = mat
      size_out[e] = size

    ok_out = wp.zeros(3, dtype=int)
    pos_out = wp.zeros(3, dtype=wp.vec3)
    mat_out = wp.zeros(3, dtype=wp.mat33)
    size_out = wp.zeros(3, dtype=wp.vec3)
    wp.launch(
      box_elem_kernel,
      dim=3,
      inputs=[
        m.qpos0,
        m.jnt_qposadr,
        m.flex_vertadr,
        m.flex_elemadr,
        m.flex_elemnum,
        m.flex_elemdataadr,
        m.flex_plasticadr,
        m.flex_vertbodyid,
        m.flex_elem,
        m.flex_rod_is_box,
        m.flex_rod_root_clamped,
        m.flex_rod_tail_clamped,
        m.flex_rod_box_size,
        m.flex_rod_edgeadr,
        m.flex_rod_edgenum,
        m.flex_rodedge_vert,
        m.flex_rodedge_jnt,
        m.flex_elem_rodedge,
        d.qpos,
        d.plastic,
        d.xmat,
        d.flexvert_xpos,
        d.flex_rod_frame0_loc,
        d.flex_rod_frame0_ref,
        d.flex_rod_tail_loc,
      ],
      outputs=[ok_out, pos_out, mat_out, size_out],
    )

    np.testing.assert_array_equal(ok_out.numpy(), [1, 1, 1])
    mats = mat_out.numpy()
    sizes = size_out.numpy()
    for e in range(3):
      _assert_eq(mats[e].T @ mats[e], np.eye(3), f"orthogonality_{e}")
      _assert_eq(np.linalg.det(mats[e]), 1.0, f"right_handed_{e}")
      _assert_eq(sizes[e], np.array([0.02, 0.006, 0.05]), f"size_{e}")
    # Element 0 and reversed element 1 have theta=0 and follow the rolled root frame via spatial PT
    _assert_eq(mats[0, :, 0], mats[1, :, 0], "m1_elem0_elem1")
    _assert_eq(mats[0, :, 2], np.array([1.0, 0.0, 0.0]), "tangent_elem0")
    _assert_eq(mats[1, :, 2], np.array([-1.0, 0.0, 0.0]), "reversed_tangent_elem1")
    # Element 2 is rotated by theta=0.4 relative to element 0's reference director (m1, m2)
    m1_0, m2_0 = mats[0, :, 0], mats[0, :, 1]
    expected_m1_2 = np.cos(0.4) * m1_0 + np.sin(0.4) * m2_0
    _assert_eq(mats[2, :, 0], expected_m1_2, "twisted_m1_elem2")


if __name__ == "__main__":
  wp.init()
  absltest.main()
