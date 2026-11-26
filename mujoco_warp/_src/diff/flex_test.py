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
"""Tests for the analytic backward of the discrete step with 3D flexes.

Gradients are compared with central finite differences of MuJoCo C (float64) on the same model.
"""

import copy
import dataclasses

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src.types import FLEX_STIFFNESS_3D
from mujoco_warp._src.types import DeterminismType
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import EqType
from mujoco_warp._src.types import OverflowType

# Norm of the gradient error over the norm of the finite-difference reference. Achieved errors
# are 1e-7 to 3e-5: the forward is float32 and its linear solve stops at a residual of 1e-5.
_REL_TOL = 1e-4

# Finite-difference steps of the Data inputs. The default law drops negative edge tension from
# its metric and stable Neo-Hookean clamps eigenvalues, both kinks of the step map, so the qpos
# step is small; the qvel step moves positions by the same amount over one timestep.
_INPUT_EPS = {"qpos": 1e-7, "qvel": 1e-7, "qfrc_applied": 1e-5, "xfrc_applied": 1e-5}


def _central(fn, x0, steps, mask=None):
  """Central differences of fn in the entries of x0 selected by mask."""
  grad = np.zeros_like(x0)
  for i in np.flatnonzero(np.ones(x0.shape, dtype=bool) if mask is None else mask):
    delta = np.zeros_like(x0)
    delta.flat[i] = steps.flat[i]
    grad.flat[i] = (fn(x0 + delta) - fn(x0 - delta)) / (2.0 * steps.flat[i])
  return grad


class _Rollout:
  """A taped mujoco_warp rollout with a linear loss, and the same loss in MuJoCo C."""

  def __init__(self, mjm, mjd, m, d, nstep, params=(), perturb=True):
    rng = np.random.default_rng(0)
    self.m, self.nstep, self.nworld, self.params = m, nstep, d.nworld, params
    # The C solve converges far below the finite-difference step; with equality rows the exact
    # Newton solver also removes CG warmstart jitter between the two sides of a difference:
    self.mjc = copy.deepcopy(mjm)
    self.mjc.opt.tolerance = 1e-13
    self.mjc.opt.iterations = 500
    if mjm.neq:
      self.mjc.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    self.mjd = mujoco.MjData(self.mjc)

    # World 1 gets its own state and applied forces. float32 values are exact in float64, so C
    # starts from the same numbers:
    self.inputs = {}
    for name in _INPUT_EPS:
      value = getattr(d, name).numpy()
      if perturb and d.nworld == 2:
        value[1] *= rng.uniform(0.5, 1.5, value[1].shape)
        getattr(d, name).assign(value)
      self.inputs[name] = value.astype(np.float64)
    self.eq_active = d.eq_active.numpy().copy() if mjm.neq else None
    self.d = [d] + [mjw.put_data(mjm, mjd, nworld=d.nworld, nvmax=mjm.nv) for _ in range(nstep)]
    for data in self.d[1:]:
      data.qpos.fill_(wp.inf)
      data.qvel.fill_(wp.inf)
      data.qacc.fill_(wp.inf)
      if self.eq_active is not None:
        data.eq_active.assign(self.eq_active)
    for name in params:
      getattr(m, name).requires_grad = True
    self.wqpos = rng.standard_normal((d.nworld, mjm.nq))
    self.wqvel = rng.standard_normal((d.nworld, mjm.nv))
    self.seed_qpos = wp.array(self.wqpos, dtype=float)
    self.seed_qvel = wp.array(self.wqvel, dtype=float)

  def backward(self):
    """Tapes the rollout and backpropagates loss = sum_w wqpos[w] . qpos[w] + wqvel[w] . qvel[w]."""
    with wp.Tape() as tape:
      for d, d_out in zip(self.d, self.d[1:]):
        mjw.step(self.m, d, d_out)
    tape.zero()
    wp.copy(self.d[-1].qpos.grad, self.seed_qpos)
    wp.copy(self.d[-1].qvel.grad, self.seed_qvel)
    tape.backward()

  def loss(self, world, **inputs):
    """Loss of one world in MuJoCo C, optionally with some of its Data inputs replaced."""
    mujoco.mj_resetData(self.mjc, self.mjd)
    if self.eq_active is not None:
      self.mjd.eq_active[:] = self.eq_active[world]
    for name, value in self.inputs.items():
      getattr(self.mjd, name)[:] = inputs.get(name, value[world])
    mujoco.mj_step(self.mjc, self.mjd, self.nstep)
    return self.wqpos[world] @ self.mjd.qpos + self.wqvel[world] @ self.mjd.qvel

  def fd_input(self, name, world):
    """Finite differences of one world's loss in a Data input."""
    x0 = self.inputs[name][world]
    eps = _INPUT_EPS[name] / (self.mjc.opt.timestep if name == "qvel" else 1.0)
    return _central(lambda x: self.loss(world, **{name: x}), x0, np.full(x0.shape, eps))

  def fd_param(self, name, eps=1e-6, mask=None):
    """Finite differences of the loss summed over worlds in a Model parameter."""
    array = getattr(self.mjc, name)
    base = array.copy()

    def fn(value):
      array[:] = value
      return sum(self.loss(world) for world in range(self.nworld))

    # One step per entry kind (its position in an element's or an equality's block), so that
    # zero-valued entries get the step of their nonzero peers:
    period = FLEX_STIFFNESS_3D if name == "flex_stiffness" else base[0].size
    kind = np.arange(base.size) % period
    scale = np.zeros(period)
    np.maximum.at(scale, kind, np.abs(base.ravel()))
    grad = _central(fn, base, eps * np.where(scale > 0.0, scale, scale.max())[kind].reshape(base.shape), mask)
    array[:] = base
    return grad

  def grad_param(self, name):
    """Gradient of a Model parameter, in the shape of the MuJoCo C field."""
    return getattr(self.m, name).grad.numpy().astype(np.float64).reshape(getattr(self.mjc, name).shape)


class FlexDiffTest(parameterized.TestCase):
  def _assert_grad(self, name, value, reference):
    error = np.linalg.norm(value - reference) / np.linalg.norm(reference)
    self.assertLess(error, _REL_TOL, f"{name}: relative error {error}")

  def _check_inputs(self, rollout, names=("qpos", "qvel")):
    """Checks the forward against C and the gradients of Data inputs against finite differences."""
    qpos, qvel = rollout.d[-1].qpos.numpy(), rollout.d[-1].qvel.numpy()
    grads = {name: getattr(rollout.d[0], name).grad.numpy().astype(np.float64) for name in names}
    for world in range(rollout.nworld):
      rollout.loss(world)
      # The float32 forward stops its solve at a residual of 1e-5 and C is converged to 1e-13:
      np.testing.assert_allclose(qpos[world], rollout.mjd.qpos, rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(qvel[world], rollout.mjd.qvel, rtol=1e-4, atol=1e-4)
      for name in names:
        self._assert_grad(f"{name}[{world}]", grads[name][world], rollout.fd_input(name, world))
    if rollout.nworld == 2:
      for name in names:
        self.assertFalse(np.allclose(grads[name][0], grads[name][1]))
    # Gradients of the overwritten outputs are consumed:
    np.testing.assert_array_equal(rollout.d[-1].qpos.grad.numpy(), 0.0)
    np.testing.assert_array_equal(rollout.d[-1].qvel.grad.numpy(), 0.0)

  def _check_params(self, rollout, eps=1e-6):
    """Checks the gradients of Model parameters other than flex_stiffness."""
    for name in rollout.params:
      if name != "flex_stiffness":
        # flex_damping enters the metric as h * damping; its step stays far below the timestep:
        fd = rollout.fd_param(name, 1e-7 if name == "flex_damping" else eps)
        self._assert_grad(name, rollout.grad_param(name), fd)

  def _check_stiffness(self, rollout, snh):
    """Checks flex_stiffness metric, SNH selector, and volume entries."""
    stiffness = rollout.mjc.flex_stiffness.reshape(-1, FLEX_STIFFNESS_3D)
    # Entry 21 selects the law. Where it is zero (default law) it is a switch, not a parameter:
    mask = ~((np.arange(FLEX_STIFFNESS_3D) == 21) & (stiffness == 0.0))
    grad = rollout.grad_param("flex_stiffness").reshape(stiffness.shape)
    fd = rollout.fd_param("flex_stiffness", mask=mask.ravel()).reshape(stiffness.shape)
    self._assert_grad("flex_stiffness[:21]", grad[:, :21], fd[:, :21])
    for entry in (21, 22, 23):
      if snh:
        self._assert_grad(f"flex_stiffness[{entry}]", grad[:, entry], fd[:, entry])
      else:
        np.testing.assert_array_equal(grad[:, entry], 0.0)
        np.testing.assert_array_equal(fd[:, entry], 0.0)

  @parameterized.product(solver=("CG", "NEWTON"), nworld=(1, 2))
  def test_default_law(self, solver, nworld):
    """State, flex_stiffness and flex_damping gradients of a default-law solid."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005"><flag contact="disable"/></option>
        <worldbody>
          <flexcomp name="soft" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1" radius="0.001">
            <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      qpos_noise=0.03,
      qvel_noise=0.2,
      overrides={"opt.solver": solver},
      nworld=nworld,
    )
    rollout = _Rollout(mjm, mjd, m, d, nstep=2, params=("flex_stiffness", "flex_damping", "flexedge_length0"))
    rollout.backward()
    self._check_inputs(rollout)
    self._check_stiffness(rollout, snh=False)
    self._check_params(rollout)

  @parameterized.product(solver=("CG", "NEWTON"), nworld=(1, 2))
  def test_stable_neo_hookean(self, solver, nworld):
    """State, flex_stiffness, flex_damping, and length0 gradients of an SNH solid."""
    spec = mujoco.MjSpec.from_string(
      """
      <mujoco>
        <option integrator="discrete" timestep="0.005"><flag contact="disable"/></option>
        <worldbody>
          <flexcomp name="soft" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1" radius="0.001">
            <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/><pin id="0"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """
    )
    spec.flexes[0].elastic3d = 1
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON if solver == "NEWTON" else mujoco.mjtSolver.mjSOL_CG
    mjm = spec.compile()
    mjd = mujoco.MjData(mjm)
    rng = np.random.default_rng(1)
    mjd.qpos[:] = 0.01 * rng.standard_normal(mjm.nq)
    mjd.qvel[:] = 0.2 * rng.standard_normal(mjm.nv)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld)
    rollout = _Rollout(mjm, mjd, m, d, nstep=2, params=("flex_stiffness", "flex_damping", "flexedge_length0"))
    rollout.backward()
    self._check_inputs(rollout)
    self._check_stiffness(rollout, snh=True)
    self._check_params(rollout)

  @parameterized.parameters(1, 2)
  def test_stable_neo_hookean_clamped(self, nworld):
    """An inverted and a flattened tetrahedron whose projected SNH Hessian clamps eigenvalues."""
    spec = mujoco.MjSpec.from_string(
      """
      <mujoco>
        <option integrator="discrete" timestep="0.01" gravity="0 0 0" solver="CG"><flag contact="disable"/></option>
        <worldbody>
          <body name="v0">
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
            <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
          </body>
          <body name="v1" pos="1 0 0">
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
            <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
          </body>
          <body name="v2" pos="0 1 0">
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
            <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
          </body>
          <body name="v3" pos="0 0 1">
            <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
            <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
          </body>
        </worldbody>
        <deformable>
          <flex name="tet" dim="3" body="v0 v1 v2 v3" vertex="0 0 0  0 0 0  0 0 0  0 0 0" element="0 1 2 3">
            <elasticity young="1200" poisson="0.3" damping="0.1"/><contact selfcollide="none"/>
          </flex>
        </deformable>
      </mujoco>
      """
    )
    spec.flexes[0].elastic3d = 1
    mjm = spec.compile()
    mjd = mujoco.MjData(mjm)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld)

    # The last vertex is reflected through the base in world 0 and pressed into it in world 1, as
    # in flex_test.test_flex_3d_snh_passive:
    qpos = np.zeros((nworld, mjm.nq), dtype=np.float32)
    qvel = np.zeros((nworld, mjm.nv), dtype=np.float32)
    qpos[0, 11] = -1.7
    qvel[0] = [0.3, -0.4, 0.2, -0.5, 0.2, -0.1, 0.4, -0.3, 0.5, -0.2, 0.6, -0.8]
    if nworld == 2:
      qpos[1] = [0.05, -0.02, 0.03, 0.12, 0.04, -0.05, -0.03, 0.08, 0.02, 0.04, -0.06, -1.0]
      qvel[1] = [-0.2, 0.5, -0.3, 0.4, -0.1, 0.3, -0.5, 0.2, -0.4, 0.3, -0.7, 0.9]
    d.qpos.assign(qpos)
    d.qvel.assign(qvel)
    rollout = _Rollout(mjm, mjd, m, d, nstep=2, params=("flex_stiffness", "flex_damping", "flexedge_length0"), perturb=False)

    # The clamp is active: the cached Hessian differs from the Jacobian of the spring force:
    mjw.kinematics(m, d)
    mjw.com_pos(m, d)
    mjw.flex(m, d)
    mjw.passive(m, d)
    hessian = d.flexedge_hessian.numpy()[0, 0]

    def spring(x):
      rollout.mjd.qpos[:] = x
      mujoco.mj_forward(rollout.mjc, rollout.mjd)
      return rollout.mjd.qfrc_spring.copy()

    i, j = 3 * mjm.flex_edge[0]
    jacobian = np.zeros((3, 3))
    for c in range(3):
      delta = np.zeros(mjm.nq)
      delta[j + c] = 1e-6
      jacobian[:, c] = -(spring(qpos[0] + delta) - spring(qpos[0] - delta))[i : i + 3] / 2e-6
    self.assertGreater(np.abs(jacobian - hessian).max(), 1.0)

    rollout.backward()
    self._check_inputs(rollout)
    self._check_stiffness(rollout, snh=True)
    self._check_params(rollout)

  @parameterized.product(solver=("CG", "NEWTON"), nworld=(1, 2))
  def test_edge_equality(self, solver, nworld):
    """EqType.FLEX rows (coupled sparse and standalone dense direct-solref) grads."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005"><flag contact="disable"/></option>
        <worldbody>
          <flexcomp name="solid" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1" radius="0.001">
            <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/>
          </flexcomp>
          <flexcomp name="edges" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0.4 0 1" dim="3" mass="1" radius="0.001">
            <edge equality="true" solref="0.02 1" solimp="0.8 0.95 0.05 0.5 2"/><contact selfcollide="none"/><pin id="0 1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      qpos_noise=0.01,
      qvel_noise=0.2,
      overrides={"opt.solver": solver},
      nworld=nworld,
    )
    params = ("flex_damping", "flexedge_length0", "eq_solref", "eq_solimp")
    rollout = _Rollout(mjm, mjd, m, d, nstep=2, params=params)
    rollout.backward()
    self._check_inputs(rollout)
    self._check_params(rollout)

    if solver == "CG" and nworld == 1:
      # Standalone EqType.FLEX dense CG path (nefmK=0, is_sparse=False) with direct solref < 0:
      mjm_d, mjd_d, m_d, d_d = test_data.fixture(
        xml="""
        <mujoco>
          <option integrator="discrete" timestep="0.005" solver="CG"><flag contact="disable"/></option>
          <worldbody>
            <flexcomp name="edges" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1" radius="0.001">
              <edge equality="true" solref="-1000 -50" solimp="0.8 0.95 0.05 0.5 2"/><contact selfcollide="none"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """,
        qpos_noise=0.01,
        qvel_noise=0.2,
      )
      r_dense = _Rollout(mjm_d, mjd_d, m_d, d_d, nstep=1, params=("flexedge_length0", "eq_solref", "eq_solimp"))
      r_dense.backward()
      self._check_inputs(r_dense)
      self._check_params(r_dense)

  def test_mixed_eq_active_cg(self):
    """CG with one unconstrained world (eq_active=False, nefc=0) and one active world."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005" solver="CG"><flag contact="disable"/></option>
        <worldbody>
          <flexcomp name="solid" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1" radius="0.001">
            <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/>
          </flexcomp>
          <flexcomp name="edges" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0.4 0 1" dim="3" mass="1" radius="0.001">
            <edge equality="true" solref="0.02 1" solimp="0.8 0.95 0.05 0.5 2"/><contact selfcollide="none"/><pin id="0"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      qpos_noise=0.01,
      qvel_noise=0.2,
      nworld=2,
    )
    # Override put_model's 1e-6 float32 clamp so CG eq_solref converges to adjoint accuracy:
    m.opt.tolerance.fill_(1e-10)
    d.eq_active.assign(np.array([[False], [True]], dtype=bool))
    params = ("flex_stiffness", "flex_damping", "flexedge_length0", "eq_solref", "eq_solimp")
    rollout = _Rollout(mjm, mjd, m, d, nstep=2, params=params)
    rollout.backward()
    nefc = rollout.d[-1].nefc.numpy()
    self.assertEqual(int(nefc[0]), 0)
    self.assertGreater(int(nefc[1]), 0)
    self._check_inputs(rollout)
    self._check_stiffness(rollout, snh=False)
    self._check_params(rollout)

  @parameterized.parameters(1, 2)
  def test_applied_forces(self, nworld):
    """qfrc_applied and xfrc_applied gradients, with pinned vertices under a rotated parent."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005" solver="CG"><flag contact="disable"/></option>
        <worldbody>
          <body name="parent" pos="0.2 -0.1 0.8" euler="25 -40 15">
            <flexcomp name="solid" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" dim="3" mass="1" radius="0.001">
              <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/><pin id="0"/>
            </flexcomp>
            <flexcomp name="edges" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0.35 0 0" dim="3" mass="1"
                      radius="0.001">
              <edge equality="true" solref="0.02 1" solimp="0.8 0.95 0.05 0.5 2"/><contact selfcollide="none"/><pin id="0"/>
            </flexcomp>
          </body>
        </worldbody>
      </mujoco>
      """,
      qvel_noise=0.2,
      qfrc_noise=2.0,
      xfrc_noise=2.0,
      nworld=nworld,
    )
    rollout = _Rollout(mjm, mjd, m, d, nstep=2)
    rollout.backward()
    self._check_inputs(rollout, names=("qpos", "qvel", "qfrc_applied", "xfrc_applied"))

  def test_output_grads(self):
    """Output qacc/xpos/site_xpos grads, untaped fwd_kinematics refresh, and shared grads."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005" solver="CG"><flag contact="disable"/></option>
        <worldbody>
          <body name="v0" pos="0 0 1">
            <inertial pos="0 0 0" mass="0.25" diaginertia="0.01 0.01 0.01"/>
            <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
          </body>
          <body name="v1" pos="0.1 0 1">
            <inertial pos="0 0 0" mass="0.25" diaginertia="0.01 0.01 0.01"/>
            <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
            <site name="tip" pos="0.02 -0.01 0.03"/>
          </body>
          <body name="v2" pos="0 0.1 1">
            <inertial pos="0 0 0" mass="0.25" diaginertia="0.01 0.01 0.01"/>
            <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
          </body>
          <body name="v3" pos="0 0 1.1">
            <inertial pos="0 0 0" mass="0.25" diaginertia="0.01 0.01 0.01"/>
            <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
          </body>
        </worldbody>
        <deformable>
          <flex name="tet" dim="3" body="v0 v1 v2 v3" vertex="0 0 0  0 0 0  0 0 0  0 0 0" element="0 1 2 3">
            <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/>
          </flex>
        </deformable>
      </mujoco>
      """,
      qpos_noise=0.02,
      qvel_noise=0.2,
    )
    ref = _Rollout(mjm, mjd, m, d, nstep=2, params=("flex_stiffness",))
    ref.backward()
    want_q = ref.d[0].qpos.grad.numpy().copy()
    want_k = ref.m.flex_stiffness.grad.numpy().copy()

    m.flex_stiffness.grad.zero_()
    rollout = _Rollout(mjm, mjd, m, mjw.put_data(mjm, mjd, nvmax=mjm.nv), nstep=2, params=("flex_stiffness",))
    with wp.Tape() as tape:
      for a, b in zip(rollout.d, rollout.d[1:]):
        mjw.step(m, a, b)
    for b in rollout.d[1:]:
      mjw.fwd_kinematics(m, b)
    tape.zero()
    wp.copy(rollout.d[-1].qpos.grad, rollout.seed_qpos)
    wp.copy(rollout.d[-1].qvel.grad, rollout.seed_qvel)
    tape.backward()
    np.testing.assert_allclose(rollout.d[0].qpos.grad.numpy(), want_q, rtol=1e-4, atol=1e-5 * np.abs(want_q).max())
    np.testing.assert_allclose(m.flex_stiffness.grad.numpy(), want_k, rtol=1e-4, atol=1e-5 * np.abs(want_k).max())

    # qacc, xpos, and site_xpos output gradients are mapped to d.qpos and d.qvel and consumed:
    rng = np.random.default_rng(7)
    wqacc = rng.standard_normal((1, mjm.nv))
    wxpos = rng.standard_normal((1, mjm.nbody, 3))
    wsite = rng.standard_normal((1, mjm.nsite, 3))
    rollout.d[0].qpos.grad.zero_()
    rollout.d[0].qvel.grad.zero_()
    with wp.Tape() as tape_out:
      for a, b in zip(rollout.d, rollout.d[1:]):
        mjw.step(m, a, b)
    tape_out.zero()
    wp.copy(rollout.d[-1].qpos.grad, rollout.seed_qpos)
    wp.copy(rollout.d[-1].qvel.grad, rollout.seed_qvel)
    rollout.d[-1].qacc.grad.assign(wqacc.astype(np.float32))
    rollout.d[-1].xpos.grad.assign(wxpos.astype(np.float32))
    rollout.d[-1].site_xpos.grad.assign(wsite.astype(np.float32))
    tape_out.backward()
    np.testing.assert_array_equal(rollout.d[-1].qacc.grad.numpy(), 0.0)
    np.testing.assert_array_equal(rollout.d[-1].xpos.grad.numpy(), 0.0)
    np.testing.assert_array_equal(rollout.d[-1].site_xpos.grad.numpy(), 0.0)

    def _loss(qp, qv):
      return float(
        rollout.loss(0, qpos=qp, qvel=qv)
        + wqacc[0] @ rollout.mjd.qacc
        + np.sum(wxpos[0] * rollout.mjd.xpos)
        + np.sum(wsite[0] * rollout.mjd.site_xpos)
      )

    dq_fd = _central(lambda x: _loss(x, rollout.inputs["qvel"][0]), rollout.inputs["qpos"][0], np.full(mjm.nq, 1e-7))
    dv_fd = _central(
      lambda x: _loss(rollout.inputs["qpos"][0], x),
      rollout.inputs["qvel"][0],
      np.full(mjm.nv, 1e-7 / mjm.opt.timestep),
    )
    self._assert_grad("qpos_with_qacc_xpos_site", rollout.d[0].qpos.grad.numpy()[0], dq_fd)
    self._assert_grad("qvel_with_qacc_xpos_site", rollout.d[0].qvel.grad.numpy()[0], dv_fd)

    # Shared qfrc_applied buffer between input and output accumulates and is not zeroed:
    d_in = mjw.make_data(mjm)
    d_out = dataclasses.replace(mjw.make_data(mjm), qfrc_applied=d_in.qfrc_applied)
    d_in.qpos.assign(rollout.inputs["qpos"].astype(np.float32))
    d_in.qvel.assign(rollout.inputs["qvel"].astype(np.float32))
    with wp.Tape() as tape_sh:
      mjw.step(m, d_in, d_out)
    d_out.qpos.grad.fill_(1.0)
    d_out.qfrc_applied.grad.fill_(0.3)
    tape_sh.backward()
    self.assertGreater(np.linalg.norm(d_in.qfrc_applied.grad.numpy()), 0.3)
    np.testing.assert_array_equal(d_out.qpos.grad.numpy(), 0.0)

  @parameterized.product(snh=(False, True), solver=("CG", "NEWTON"))
  @absltest.skipIf(not wp.get_device().is_cuda, "CUDA graph capture requires a CUDA device.")
  def test_graph_capture(self, snh, solver):
    """A rollout and its backward replay from a CUDA graph match eager with or without warm-up."""
    params = ("flex_stiffness", "flex_damping", "flexedge_length0", "eq_solref", "eq_solimp")

    def _make_rollout():
      if not snh:
        mjm, mjd, m, d = test_data.fixture(
          xml="""
          <mujoco>
            <option integrator="discrete" timestep="0.005"><flag contact="disable"/></option>
            <worldbody>
              <flexcomp name="solid" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1"
                        radius="0.001">
                <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/>
              </flexcomp>
              <flexcomp name="edges" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0.4 0 1" dim="3" mass="1"
                        radius="0.001">
                <edge equality="true" solref="0.02 1" solimp="0.8 0.95 0.05 0.5 2"/><contact selfcollide="none"/>
              </flexcomp>
            </worldbody>
          </mujoco>
          """,
          qpos_noise=0.01,
          qvel_noise=0.2,
          overrides={"opt.solver": solver, "opt.deterministic": "ALL"},
          nworld=2,
        )
        return _Rollout(mjm, mjd, m, d, nstep=2, params=params)
      spec = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <option integrator="discrete" timestep="0.005"><flag contact="disable"/></option>
          <worldbody>
            <flexcomp name="solid" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1" radius="0.001">
              <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/><pin id="0"/>
            </flexcomp>
            <flexcomp name="edges" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0.4 0 1" dim="3" mass="1"
                      radius="0.001">
              <edge equality="true" solref="0.02 1" solimp="0.8 0.95 0.05 0.5 2"/><contact selfcollide="none"/>
            </flexcomp>
          </worldbody>
        </mujoco>
        """
      )
      spec.flexes[0].elastic3d = 1
      spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON if solver == "NEWTON" else mujoco.mjtSolver.mjSOL_CG
      mjm = spec.compile()
      mjd = mujoco.MjData(mjm)
      rng = np.random.default_rng(2)
      mjd.qpos[:] = 0.01 * rng.standard_normal(mjm.nq)
      mjd.qvel[:] = 0.2 * rng.standard_normal(mjm.nv)
      m = mjw.put_model(mjm)
      m.opt.deterministic = DeterminismType.ALL
      d = mjw.put_data(mjm, mjd, nworld=2)
      return _Rollout(mjm, mjd, m, d, nstep=2, params=params)

    ref = _make_rollout()
    ref.backward()
    want_inputs = (ref.d[0].qpos.grad.numpy(), ref.d[0].qvel.grad.numpy())
    want_params = [getattr(ref.m, p).grad.numpy() for p in params]
    for arr in (*want_inputs, *want_params):
      self.assertGreater(np.linalg.norm(arr), 0.0)

    for warm_up in (True, False):
      with self.subTest(warm_up=warm_up):
        rollout = _make_rollout()
        if warm_up:
          rollout.backward()
        with wp.ScopedCapture() as capture:
          rollout.backward()
        # The captured tape.zero() keeps replays from accumulating:
        for _ in range(2):
          wp.capture_launch(capture.graph)
        wp.synchronize()
        got_inputs = (rollout.d[0].qpos.grad.numpy(), rollout.d[0].qvel.grad.numpy())
        got_params = [getattr(rollout.m, p).grad.numpy() for p in params]
        for got, want in zip((*got_inputs, *got_params), (*want_inputs, *want_params)):
          np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

  @parameterized.product(snh=(False, True), solver=("CG", "NEWTON"))
  def test_deterministic_atomics(self, snh, solver):
    """DeterminismType.ATOMICS gives bitwise-equal gradients across repeated backward runs."""
    params = ("flex_stiffness", "flex_damping", "flexedge_length0", "eq_solref", "eq_solimp")

    def _run():
      if not snh:
        mjm, mjd, m, d = test_data.fixture(
          xml="""
          <mujoco>
            <option integrator="discrete" timestep="0.005"><flag contact="disable"/></option>
            <worldbody>
              <flexcomp name="solid" type="grid" count="3 3 3" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1"
                        radius="0.001">
                <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/>
              </flexcomp>
              <flexcomp name="edges" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0.5 0 1" dim="3" mass="1"
                        radius="0.001">
                <edge equality="true" solref="0.02 1" solimp="0.8 0.95 0.05 0.5 2"/><contact selfcollide="none"/>
              </flexcomp>
            </worldbody>
          </mujoco>
          """,
          qpos_noise=0.01,
          qvel_noise=0.2,
          overrides={"opt.solver": solver, "opt.deterministic": "ATOMICS"},
          nworld=2,
        )
      else:
        spec = mujoco.MjSpec.from_string(
          """
          <mujoco>
            <option integrator="discrete" timestep="0.005"><flag contact="disable"/></option>
            <worldbody>
              <flexcomp name="solid" type="grid" count="3 3 3" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1"
                        radius="0.001">
                <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/><pin id="0"/>
              </flexcomp>
              <flexcomp name="edges" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0.5 0 1" dim="3" mass="1"
                        radius="0.001">
                <edge equality="true" solref="0.02 1" solimp="0.8 0.95 0.05 0.5 2"/><contact selfcollide="none"/>
              </flexcomp>
            </worldbody>
          </mujoco>
          """
        )
        spec.flexes[0].elastic3d = 1
        spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON if solver == "NEWTON" else mujoco.mjtSolver.mjSOL_CG
        mjm = spec.compile()
        mjd = mujoco.MjData(mjm)
        rng = np.random.default_rng(3)
        mjd.qpos[:] = 0.01 * rng.standard_normal(mjm.nq)
        mjd.qvel[:] = 0.2 * rng.standard_normal(mjm.nv)
        m = mjw.put_model(mjm)
        m.opt.deterministic = DeterminismType.ATOMICS
        d = mjw.put_data(mjm, mjd, nworld=2)
      rollout = _Rollout(mjm, mjd, m, d, nstep=1, params=params)
      rollout.backward()
      return (
        rollout.d[0].qpos.grad.numpy().copy(),
        rollout.d[0].qvel.grad.numpy().copy(),
        *[getattr(rollout.m, p).grad.numpy().copy() for p in params],
      )

    first = _run()
    second = _run()
    for arr in first:
      self.assertGreater(np.linalg.norm(arr), 0.0)
    for got, want in zip(second, first):
      np.testing.assert_array_equal(got, want)

  @parameterized.named_parameters(
    dict(testcase_name="rod", reason="dimension other than 3", flex='dim="1" count="4 1 1"'),
    dict(testcase_name="shell", reason="dimension other than 3", flex='dim="2" count="3 3 1"'),
    dict(testcase_name="interpolated", reason="interpolated flexes", flex='dim="3" count="3 3 3" dof="trilinear"'),
    dict(
      testcase_name="jointed_parent",
      reason="3-slider bodies",
      option='integrator="discrete" solver="CG"',
      parent='<joint/><geom size="0.02" mass="0.1"/>',
      pin='<pin id="0"/>',
    ),
    dict(
      testcase_name="nv_zero",
      reason="nv == 0",
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005" solver="Newton"><flag contact="disable"/></option>
        <worldbody><body name="b0" pos="0 0 0.2"/><body name="b1" pos="0.1 0 0.2"/></worldbody>
        <deformable>
          <flex name="f" dim="3" body="b0 b0 b1 b1" vertex="0 0 0  0 0.1 0  0 0 0  0 0 0.1" element="0 1 2 3"/>
        </deformable>
        <equality><flex flex="f"/></equality>
      </mujoco>
      """,
    ),
    dict(
      testcase_name="no_flex",
      reason="without a flex",
      xml='<mujoco><option integrator="discrete"/><worldbody><body><joint/><geom size="0.1"/></body></worldbody></mujoco>',
    ),
    dict(testcase_name="rigid_body", reason="rigid bodies", world='<body><freejoint/><geom size="0.05"/></body>'),
    dict(testcase_name="mocap", reason="mocap bodies", world='<body mocap="true"><geom size="0.05"/></body>'),
    dict(
      testcase_name="actuator",
      reason="actuators",
      option='integrator="discrete" solver="CG"',
      parent='<joint name="j"/><geom size="0.02" mass="0.1"/>',
      pin='<pin id="0"/>',
      root='<actuator><motor joint="j"/></actuator>',
    ),
    dict(
      testcase_name="tendon",
      reason="tendons",
      option='integrator="discrete" solver="CG"',
      parent='<joint name="j"/><geom size="0.02" mass="0.1"/>',
      pin='<pin id="0"/>',
      root='<tendon><fixed><joint joint="j" coef="1"/></fixed></tendon>',
    ),
    dict(testcase_name="contact", reason="contacts", flag="", world='<geom size="0.05" pos="0 0 1"/>'),
    dict(
      testcase_name="weld",
      reason="other than flex edge equality",
      root='<equality><weld body1="soft_0" body2="soft_1"/></equality>',
    ),
    dict(
      testcase_name="euler",
      reason="supports only IntegratorType.DISCRETE with stock 3D flexes",
      option='integrator="Euler"',
    ),
    dict(testcase_name="pgs", reason="solvers other than CG and NEWTON", edits={"opt.solver": int(mujoco.mjtSolver.mjSOL_PGS)}),
    dict(testcase_name="noslip", reason="noslip solver", edits={"opt.noslip_iterations": 2}),
    dict(testcase_name="no_iterations", reason="opt.iterations <= 0", option='integrator="discrete" iterations="0"'),
    dict(testcase_name="damper_disabled", reason="disabled springs or dampers", flag='contact="disable" damper="disable"'),
    dict(testcase_name="sleep", reason="sleep", edits={"opt.enableflags": int(EnableBit.SLEEP)}),
    dict(testcase_name="joint_damping", reason="joint damping", edits={"dof_damping": 0.5}),
    dict(testcase_name="joint_stiffness", reason="joint stiffness", edits={"jnt_stiffness": 0.5}),
    dict(testcase_name="joint_limits", reason="joint limits", edits={"jnt_limited": 1}),
    dict(testcase_name="joint_friction", reason="joint friction loss", edits={"dof_frictionloss": 0.2}),
    dict(testcase_name="gravity_comp", reason="gravity compensation", edits={"body_gravcomp": 0.5}),
    dict(testcase_name="edge_damping", reason="flex edge stiffness or damping", edits={"flex_edgedamping": 1.0}),
    dict(testcase_name="passive_contact", reason="passive flex contact", edits={"flex_passive": 1}),
    dict(testcase_name="plasticity", reason="flex plasticity", edits={"flex_plasticmode": 2}),
    dict(testcase_name="adhesion", reason="adhesion", edits={"flg_adhesion": True}),
    dict(testcase_name="surfacevel", reason="surface velocity", edits={"flg_surfacevel": True}),
    dict(testcase_name="callback", reason="passive callback", edits={"callback.passive": lambda m, d: None}),
    dict(
      testcase_name="strain_equality",
      reason="flex strain equality",
      inner='<edge equality="true"/>',
      edits={"flex_edgeequality": 2, "eq_type": int(EqType.FLEXSTRAIN)},
    ),
    dict(testcase_name="body_mass_gradient", reason="gradients for body_mass", edits={"body_mass.requires_grad": True}),
    dict(
      testcase_name="njmax",
      reason=r"d\.njmax \(1\) smaller",
      inner='<edge equality="true"/>',
      data_kwargs={"njmax": 1, "naconmax": 0},
    ),
    dict(
      testcase_name="njmax_nnz",
      reason=r"d\.njmax_nnz \(4\) smaller",
      option='integrator="discrete" jacobian="sparse"',
      inner='<edge equality="true"/>',
      data_kwargs={"njmax": 64, "njmax_nnz": 4, "naconmax": 0},
    ),
    dict(testcase_name="in_place", reason="distinct input and output", step_mode="in_place"),
    dict(testcase_name="aliased_state", reason="input qpos aliases output state", step_mode="aliased"),
  )
  def test_unsupported(
    self,
    reason,
    option='integrator="discrete"',
    flag='contact="disable"',
    parent="",
    flex='dim="3" count="2 2 2"',
    inner='<elasticity young="5e3" poisson="0.3" damping="0.02"/>',
    pin="",
    world="",
    root="",
    edits=None,
    data_kwargs=None,
    step_mode="out_of_place",
    xml=None,
  ):
    """A taped step of an unsupported model names the reason, and neither runs nor records."""
    mjm, mjd, m, d = test_data.fixture(
      xml=xml
      or f"""
      <mujoco>
        <option {option}><flag {flag}/></option>
        <worldbody>
          <body name="parent" pos="0 0 1">
            {parent}
            <flexcomp name="soft" type="grid" spacing="0.1 0.1 0.1" mass="1" radius="0.001" {flex}>
              {inner}
              <contact selfcollide="none"/>
              {pin}
            </flexcomp>
          </body>
          {world}
        </worldbody>
        {root}
      </mujoco>
      """,
    )
    for path, value in (edits or {}).items():
      owner, _, name = path.rpartition(".")
      owner = getattr(m, owner) if owner else m
      if isinstance(getattr(owner, name), wp.array):
        getattr(owner, name).fill_(value)
      else:
        setattr(owner, name, value)
    if data_kwargs is not None:
      d = mjw.make_data(mjm, **data_kwargs)
      d_out = mjw.make_data(mjm, **data_kwargs)
    else:
      d_out = mjw.put_data(mjm, mjd, nvmax=mjm.nv)
    qpos_before = d.qpos.numpy()
    if step_mode == "aliased":
      d_out = dataclasses.replace(d_out, qpos=d.qpos)
    else:
      d_out.qpos.fill_(wp.inf)
    with wp.Tape() as tape, self.assertRaisesRegex(NotImplementedError, reason):
      if step_mode == "in_place":
        mjw.step(m, d)
      else:
        mjw.step(m, d, d_out)
    self.assertEmpty(tape.launches)
    self.assertEmpty(tape.gradients)
    np.testing.assert_array_equal(d.qpos.numpy(), qpos_before)
    if step_mode == "out_of_place":
      np.testing.assert_array_equal(d_out.qpos.numpy(), np.inf)

  def test_stable_neo_hookean_isotropy(self):
    """SNH flex_stiffness is checked as it is optimised; isotropic passes, others fail."""
    spec = mujoco.MjSpec.from_string(
      """
      <mujoco>
        <option integrator="discrete" timestep="0.005" solver="CG"><flag contact="disable"/></option>
        <worldbody>
          <flexcomp name="soft" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1" radius="0.001">
            <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """
    )
    spec.flexes[0].elastic3d = 1
    mjm = spec.compile()
    m = mjw.put_model(mjm)
    m.flex_stiffness.requires_grad = True
    d, d_out = mjw.make_data(mjm), mjw.make_data(mjm)
    stiffness = m.flex_stiffness.numpy().reshape(-1, FLEX_STIFFNESS_3D)

    with wp.Tape() as tape:
      # Scaling the metric and the Lame terms of every element (Young's modulus) stays isotropic:
      scaled = stiffness.copy()
      scaled[:, :23] *= 1.3
      m.flex_stiffness.assign(scaled.reshape(m.flex_stiffness.shape))
      mjw.step(m, d, d_out)
      self.assertLen(tape.launches, 1)

      for label, patch, reason in (
        ("anisotropic", lambda s: s.__setitem__((0, 1), s[0, 1] * 1.001), "anisotropic"),
        ("nonfinite", lambda s: s.__setitem__((1, 1), np.nan), "non-finite"),
        ("mixed_law", lambda s: s.__setitem__((0, 21), 0.0), "mixing default and stable Neo-Hookean"),
      ):
        with self.subTest(case=label):
          bad = stiffness.copy()
          patch(bad)
          m.flex_stiffness.assign(bad.reshape(m.flex_stiffness.shape))
          with self.assertRaisesRegex(NotImplementedError, reason):
            mjw.step(m, d, d_out)
          self.assertLen(tape.launches, 1)

  @parameterized.parameters(1, 2)
  def test_rigid_unaffected(self, nworld):
    """Rigid Euler models match MuJoCo C untaped and reject taped differentiation cleanly."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option timestep="0.005"/>
        <worldbody>
          <body>
            <joint axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.05"/>
            <body pos="0.3 0 0">
              <joint axis="0 1 0"/>
              <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.05"/>
            </body>
          </body>
        </worldbody>
      </mujoco>
      """,
      qpos_noise=0.5,
      qvel_noise=0.5,
      nworld=nworld,
    )
    self.assertEqual(mjm.nplastic, 0)
    rng = np.random.default_rng(4)
    q0 = d.qpos.numpy()
    v0 = d.qvel.numpy()
    if nworld == 2:
      q0[1] *= rng.uniform(0.5, 1.5, q0[1].shape)
      v0[1] *= rng.uniform(0.5, 1.5, v0[1].shape)
      d.qpos.assign(q0)
      d.qvel.assign(v0)
    d_out = mjw.put_data(mjm, mjd, nworld=nworld, nvmax=mjm.nv)
    d_in = mjw.put_data(mjm, mjd, nworld=nworld, nvmax=mjm.nv)
    d_in.qpos.assign(q0)
    d_in.qvel.assign(v0)
    mjw.step(m, d, d_out)
    mjw.step(m, d_in)
    np.testing.assert_allclose(d_out.qpos.numpy(), d_in.qpos.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(d_out.qvel.numpy(), d_in.qvel.numpy(), rtol=1e-6, atol=1e-6)
    q_got, v_got = d_out.qpos.numpy(), d_out.qvel.numpy()
    mjd_c = mujoco.MjData(mjm)
    for w in range(nworld):
      mujoco.mj_resetData(mjm, mjd_c)
      mjd_c.qpos[:] = q0[w]
      mjd_c.qvel[:] = v0[w]
      mujoco.mj_step(mjm, mjd_c)
      np.testing.assert_allclose(q_got[w], mjd_c.qpos, rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(v_got[w], mjd_c.qvel, rtol=1e-4, atol=1e-4)
    with wp.Tape() as tape, self.assertRaisesRegex(NotImplementedError, "IntegratorType.DISCRETE"):
      mjw.step(m, d, d_out)
    self.assertEmpty(tape.launches)
    self.assertFalse(d.plastic.requires_grad)
    self.assertIsNone(d.plastic.grad)

  def test_nonfinite_and_iteration_cap(self):
    """Non-finite isolation with param grads, CG overflow flag, and Newton no-eq solve."""
    for solver, mode in (("CG", "nan_qpos"), ("CG", "nan_rhs"), ("NEWTON", "nan_rhs")):
      with self.subTest(solver=solver, mode=mode):
        mjm, mjd, m, d = test_data.fixture(
          xml="""
          <mujoco>
            <option integrator="discrete" timestep="0.005" iterations="200" tolerance="1e-10">
              <flag contact="disable"/>
            </option>
            <worldbody>
              <flexcomp name="solid" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1"
                        radius="0.001">
                <elasticity young="5e3" poisson="0.3" damping="0.02"/><contact selfcollide="none"/>
              </flexcomp>
              <flexcomp name="edges" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0.4 0 1" dim="3" mass="1"
                        radius="0.001">
                <edge equality="true" solref="0.02 1" solimp="0.8 0.95 0.05 0.5 2"/><contact selfcollide="none"/>
              </flexcomp>
            </worldbody>
          </mujoco>
          """,
          qpos_noise=0.01,
          qvel_noise=0.2,
          overrides={"opt.solver": solver},
          nworld=2,
          batch_sizes={"eq_solref": 2},
        )
        m.opt.warn_overflow = 0
        params = ("flex_stiffness", "flex_damping", "flexedge_length0", "eq_solref", "eq_solimp")
        rollout = _Rollout(mjm, mjd, m, d, nstep=1, params=params)
        if mode == "nan_qpos":
          qpos = rollout.d[0].qpos.numpy()
          qpos[0, 0] = np.nan
          rollout.d[0].qpos.assign(qpos)
        else:
          wq = rollout.seed_qpos.numpy()
          wq[0, 0] = np.nan
          rollout.seed_qpos.assign(wq)
        with wp.Tape() as tape_nf:
          mjw.step(m, rollout.d[0], rollout.d[1])
        tape_nf.zero()
        if mode == "nan_qpos":
          self.assertNotEqual(int(rollout.d[-1].overflow.numpy()[0]) & int(OverflowType.ITERATIONS), 0)
        # Clear forward overflow so we check that the backward itself short-circuits world 0:
        rollout.d[-1].overflow.zero_()
        wp.copy(rollout.d[-1].qpos.grad, rollout.seed_qpos)
        wp.copy(rollout.d[-1].qvel.grad, rollout.seed_qvel)
        tape_nf.backward()

        dqpos = rollout.d[0].qpos.grad.numpy().astype(np.float64)
        dqvel = rollout.d[0].qvel.grad.numpy().astype(np.float64)
        self.assertTrue(np.isnan(dqpos[0]).all())
        self.assertTrue(np.isnan(dqvel[0]).all())
        self.assertEqual(int(rollout.d[-1].overflow.numpy()[0]) & int(OverflowType.ITERATIONS), 0)
        self._assert_grad("qpos[1]", dqpos[1], rollout.fd_input("qpos", 1))
        self._assert_grad("qvel[1]", dqvel[1], rollout.fd_input("qvel", 1))

        # Shared Model parameters use sum-over-worlds semantics and become NaN; batched eq_solref
        # isolates world 1:
        stiff_g = rollout.grad_param("flex_stiffness").reshape(-1, FLEX_STIFFNESS_3D)
        self.assertTrue(np.isnan(stiff_g[: mjm.flex_elemnum[0], :21]).all())
        self.assertTrue(np.isnan(m.flex_damping.grad.numpy()[0]))
        self.assertTrue(np.isnan(m.flexedge_length0.grad.numpy()).all())
        self.assertTrue(np.isnan(m.eq_solimp.grad.numpy()).all())
        solref_g = m.eq_solref.grad.numpy().astype(np.float64)
        self.assertTrue(np.isnan(solref_g[0]).all())
        base_sr = rollout.mjc.eq_solref.copy()

        def _loss_w1(sr):
          rollout.mjc.eq_solref[:] = sr
          val = rollout.loss(1)
          rollout.mjc.eq_solref[:] = base_sr
          return val

        fd_sr1 = _central(_loss_w1, base_sr, 1e-6 * np.abs(base_sr))
        self._assert_grad("eq_solref[1]", solref_g[1], fd_sr1)

    # CG adjoint under an iteration cap flags only unconverged nonzero-RHS worlds and keeps finite
    # parameter gradients:
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <option integrator="discrete" timestep="0.005" solver="CG" iterations="1" tolerance="1e-10">
          <flag contact="disable"/>
        </option>
        <worldbody>
          <flexcomp name="solid" type="grid" count="3 3 3" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1" radius="0.001">
            <elasticity young="5e5" poisson="0.3" damping="0.02"/><contact selfcollide="none"/>
          </flexcomp>
          <flexcomp name="edges" type="grid" count="2 2 2" spacing="0.1 0.1 0.1" pos="0.5 0 1" dim="3" mass="1" radius="0.001">
            <edge equality="true" solref="0.005 1" solimp="0.9 0.99 0.01 0.5 2"/><contact selfcollide="none"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      """,
      qpos_noise=0.01,
      qvel_noise=0.2,
      nworld=2,
    )
    m.opt.warn_overflow = 0
    params = ("flex_stiffness", "flex_damping", "flexedge_length0", "eq_solref", "eq_solimp")
    rollout_cap = _Rollout(mjm, mjd, m, d, nstep=1, params=params)
    wq = rollout_cap.seed_qpos.numpy()
    wv = rollout_cap.seed_qvel.numpy()
    wq[0] = 0.0
    wv[0] = 0.0
    rollout_cap.seed_qpos.assign(wq)
    rollout_cap.seed_qvel.assign(wv)
    with wp.Tape() as tape_cap:
      mjw.step(m, rollout_cap.d[0], rollout_cap.d[1])
    tape_cap.zero()
    rollout_cap.d[-1].overflow.zero_()
    wp.copy(rollout_cap.d[-1].qpos.grad, rollout_cap.seed_qpos)
    wp.copy(rollout_cap.d[-1].qvel.grad, rollout_cap.seed_qvel)
    tape_cap.backward()
    ovf = rollout_cap.d[-1].overflow.numpy()
    self.assertEqual(int(ovf[0]) & int(OverflowType.ITERATIONS), 0)
    self.assertNotEqual(int(ovf[1]) & int(OverflowType.ITERATIONS), 0)
    for name in params:
      grad = getattr(m, name).grad.numpy()
      self.assertTrue(np.isfinite(grad).all(), msg=name)
      self.assertGreater(np.linalg.norm(grad), 0.0, msg=name)

    # Newton without equality rows: the backward linear solve is direct Cholesky. With
    # run_qacc_smooth=False the forward is also direct (exact at iterations=1); with default
    # run_qacc_smooth=True main's forward uses PCG and matches C FD once iterations converges it:
    for case, iters, overrides in (
      ("direct_forward_and_backward_iter1", 1, {"opt.run_qacc_smooth": False}),
      ("default_options_converged_iter100", 100, {}),
    ):
      with self.subTest(newton_no_eq=case):
        mjm_n, mjd_n, m_n, d_n = test_data.fixture(
          xml="""
          <mujoco>
            <option integrator="discrete" timestep="0.005" solver="Newton" iterations="100" tolerance="1e-10">
              <flag contact="disable"/>
            </option>
            <worldbody>
              <flexcomp name="solid" type="grid" count="3 3 3" spacing="0.1 0.1 0.1" pos="0 0 1" dim="3" mass="1"
                        radius="0.001">
                <elasticity young="1.5e4" poisson="0.3" damping="0.02"/><contact selfcollide="none"/>
              </flexcomp>
            </worldbody>
          </mujoco>
          """,
          qpos_noise=0.01,
          qvel_noise=0.2,
          overrides=overrides,
        )
        m_n.opt.iterations = iters
        m_n.opt.warn_overflow = 0
        rollout_n = _Rollout(mjm_n, mjd_n, m_n, d_n, nstep=1, params=("flex_stiffness", "flex_damping"))
        rollout_n.backward()
        self.assertEqual(int(rollout_n.d[-1].overflow.numpy()[0]) & int(OverflowType.ITERATIONS), 0)
        self._check_inputs(rollout_n)
        self._assert_grad(
          "newton_no_eq_damping", rollout_n.grad_param("flex_damping"), rollout_n.fd_param("flex_damping", 1e-7)
        )
        stiff = rollout_n.mjc.flex_stiffness.copy()
        direction = np.where(np.arange(stiff.size) % FLEX_STIFFNESS_3D < 21, stiff, 0.0)
        eps = 1e-6
        rollout_n.mjc.flex_stiffness[:] = stiff + eps * direction
        l_plus = rollout_n.loss(0)
        rollout_n.mjc.flex_stiffness[:] = stiff - eps * direction
        l_minus = rollout_n.loss(0)
        rollout_n.mjc.flex_stiffness[:] = stiff
        terms = (rollout_n.grad_param("flex_stiffness") * direction).ravel()
        self.assertLess(abs(terms.sum() - (l_plus - l_minus) / (2.0 * eps)) / np.linalg.norm(terms), _REL_TOL)


if __name__ == "__main__":
  wp.init()
  absltest.main()
