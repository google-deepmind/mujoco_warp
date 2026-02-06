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
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_func
from mujoco_warp._src.block_cholesky import create_blocked_cholesky_solve_func

# tolerance for difference between MuJoCo and MJWarp support calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SupportTest(parameterized.TestCase):
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

  def test_get_state(self):
    mjm, mjd, m, d = test_data.fixture(
      "constraints.xml", keyframe=0, ctrl_noise=1.0, qfrc_noise=1.0, xfrc_noise=1.0, mocap_noise=1.0
    )

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

  def test_set_state(self):
    mjm, mjd, m, d = test_data.fixture(
      "constraints.xml", keyframe=0, ctrl_noise=1.0, qfrc_noise=1.0, xfrc_noise=1.0, mocap_noise=1.0
    )

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

    # Create combined factor and solve kernel as in solver.py
    @wp.kernel(module="unique", enable_backward=False)
    def combined_cholesky_kernel(
      grad_in: wp.array3d(dtype=float),
      h_in: wp.array3d(dtype=float),
      done_in: wp.array(dtype=bool),
      hfactor_in: wp.array3d(dtype=float),
      Mgrad_out: wp.array3d(dtype=float),
    ):
      worldid = wp.tid()
      TILE_SIZE = wp.static(16)

      if done_in[worldid]:
        return

      wp.static(create_blocked_cholesky_func(TILE_SIZE))(h_in[worldid], nv_pad, hfactor_in[worldid])
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

    # 3. Create inline arrays for grad, Mgrad, and done (no longer in d.efc)
    grad_np = np.zeros((nworld, nv_pad))
    grad_np[0, :nv] = b
    grad = wp.array(grad_np, dtype=float)

    # Zero out the temporary arrays to ensure clean state
    L_init = np.zeros((nworld, nv_pad, nv_pad), dtype=np.float32)
    # Initialize padding region to identity
    L_init[0, nv:, nv:] = np.eye(nv_pad - nv, dtype=np.float32)

    hfactor = wp.array(L_init, dtype=float)

    Mgrad = wp.zeros((nworld, nv_pad), dtype=float)

    # Ensure done is False so kernel executes
    done = wp.zeros((nworld,), dtype=bool)

    # Launch with same dimensions as solver.py, using inline arrays
    wp.launch_tiled(
      combined_cholesky_kernel,
      dim=nworld,
      inputs=[grad.reshape(shape=(nworld, nv_pad, 1)), h, done, hfactor],
      outputs=[Mgrad.reshape(shape=(nworld, nv_pad, 1))],
      block_dim=m.block_dim.update_gradient_cholesky,
    )
    wp.synchronize()

    L_result = hfactor.numpy()[0]
    x_result = Mgrad.numpy()[0]

    # Verify padding outside active region doesn't affect active computation
    # Off-diagonal padding should be zero (active region shouldn't touch padding)
    np.testing.assert_array_equal(
      L_result[nv:, :nv],
      0.0,
      err_msg="padding rows in active region were overwritten",
    )
    np.testing.assert_array_equal(
      L_result[:nv, nv:],
      0.0,
      err_msg="padding columns in active region were overwritten",
    )

    # Check that padding region remains identity after factorization
    padding_size = nv_pad - nv
    if padding_size > 0:
      padding_square = L_result[nv:, nv:]
      expected_identity = np.eye(padding_size, dtype=np.float32)
      np.testing.assert_allclose(
        padding_square,
        expected_identity,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Padding region should remain identity after factorization",
      )

    # Compare with numpy cholesky on symmetrized Hessian
    L_numpy = np.linalg.cholesky(SPD_active_hessian)
    np.testing.assert_allclose(
      L_result[:nv, :nv],
      L_numpy,
      rtol=1e-4,
      atol=1e-4,
      err_msg="Cholesky decomposition mismatch with numpy",
    )

    # Verify L @ L.T = A (symmetrized Hessian)
    A_reconstructed = L_result[:nv, :nv] @ L_result[:nv, :nv].T
    np.testing.assert_allclose(
      A_reconstructed,
      SPD_active_hessian,
      rtol=1e-5,
      atol=1e-5,
      err_msg="L @ L.T does not equal symmetrized Hessian",
    )

    # Verify solution: A @ x = b using the symmetrized Hessian
    x_numpy = np.linalg.solve(SPD_active_hessian, b)
    np.testing.assert_allclose(
      x_result[:nv],
      x_numpy,
      rtol=1e-3,
      atol=1e-3,
      err_msg="Solution mismatch with numpy",
    )


if __name__ == "__main__":
  wp.init()
  absltest.main()
