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

from . import test_util
from .support import contact_force_kernel

wp.config.verify_cuda = True

# tolerance for difference between MuJoCo and MJWarp support calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _load_from_string(xml: str, keyframe: int = -1):
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  if keyframe > -1:
    mujoco.mj_resetDataKeyframe(mjm, mjd, keyframe)
  mujoco.mj_forward(mjm, mjd)

  m = mjwarp.put_model(mjm)
  d = mjwarp.put_data(mjm, mjd)

  return mjm, mjd, m, d


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


class SupportTest(parameterized.TestCase):
  @parameterized.parameters(True, False)
  def test_mul_m(self, sparse):
    """Tests mul_m."""
    mjm, mjd, m, d = test_util.fixture("pendula.xml", sparse=sparse)

    mj_res = np.zeros(mjm.nv)
    mj_vec = np.random.uniform(low=-1.0, high=1.0, size=mjm.nv)
    mujoco.mj_mulM(mjm, mjd, mj_res, mj_vec)

    res = wp.zeros((1, mjm.nv), dtype=wp.float32)
    vec = wp.from_numpy(np.expand_dims(mj_vec, axis=0), dtype=wp.float32)
    skip = wp.zeros((d.nworld), dtype=bool)
    mjwarp.mul_m(m, d, res, vec, skip)

    _assert_eq(res.numpy()[0], mj_res, f"mul_m ({'sparse' if sparse else 'dense'})")

  def test_xfrc_accumulated(self):
    """Tests that xfrc_accumulate ouput matches mj_xfrcAccumulate."""
    np.random.seed(0)
    mjm, mjd, m, d = test_util.fixture("pendula.xml")
    xfrc = np.random.randn(*d.xfrc_applied.numpy().shape)
    d.xfrc_applied = wp.from_numpy(xfrc, dtype=wp.spatial_vector)
    qfrc = wp.zeros((1, mjm.nv), dtype=wp.float32)
    mjwarp.xfrc_accumulate(m, d, qfrc)

    qfrc_expected = np.zeros(m.nv)
    xfrc = xfrc[0]
    for i in range(1, m.nbody):
      mujoco.mj_applyFT(
        mjm, mjd, xfrc[i, :3], xfrc[i, 3:], mjd.xipos[i], i, qfrc_expected
      )
    np.testing.assert_almost_equal(qfrc.numpy()[0], qfrc_expected, 6)

  def test_make_put_data(self):
    """Tests that make_put_data and put_data are producing the same shapes for all warp arrays."""
    mjm, _, _, d = test_util.fixture("pendula.xml")
    md = mjwarp.make_data(mjm)

    # same number of fields
    self.assertEqual(len(d.__dict__), len(md.__dict__))

    # test shapes for all arrays
    for attr, val in md.__dict__.items():
      if isinstance(val, wp.array):
        self.assertEqual(val.shape, getattr(d, attr).shape)

  _CONTACTS = """
    <mujoco>
      <worldbody>
        <body pos="0 0 0.55" euler="1 0 0">
          <joint axis="1 0 0" type="free"/>
          <geom fromto="-0.4 0 0 0.4 0 0" size="0.05" type="capsule" condim="6"/>
        </body>
        <body pos="0 0 0.5" euler="0 1 0">
          <joint axis="1 0 0" type="free"/>
          <geom fromto="-0.4 0 0 0.4 0 0" size="0.05" type="capsule" condim="3"/>
        </body>
        <body pos="0 0 0.445" euler="0 90 0">
          <joint axis="1 0 0" type="free"/>
          <geom fromto="-0.4 0 0 0.4 0 0" size="0.05" type="capsule" condim="1"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_contact_force(self):
    mjm, mjd, m, d = _load_from_string(self._CONTACTS)

    # map MJX contacts to MJ ones
    def _find(g):
      val = (g == mjd.contact.geom).sum(axis=1)
      return np.where(val == 2)[0][0]

    for i in range(mjd.ncon):
      result = np.zeros(6, dtype=float)
      mujoco.mj_contactForce(mjm, mjd, i, result)

      j = i
      force = wp.zeros(1, dtype=wp.spatial_vector)
      wp.launch(
        kernel=contact_force_kernel,
        dim=1,
        inputs=[
          m,
          d,
          force,
          wp.array(
            [
              j,
            ],
            dtype=int,
          ),
          False,
        ],
      )
      force = force.numpy()[0]
      np.testing.assert_allclose(result, force, rtol=1e-5, atol=2)

      # check for zeros after first condim elements
      condim = mjd.contact.dim[j]
      if condim < 6:
        np.testing.assert_allclose(force[condim:], 0, rtol=1e-5, atol=1e-5)

      # test world conversion
      force = wp.zeros(1, dtype=wp.spatial_vector)
      wp.launch(
        kernel=contact_force_kernel,
        dim=1,
        inputs=[
          m,
          d,
          force,
          wp.array(
            [
              j,
            ],
            dtype=int,
          ),
          True,
        ],
      )
      force = force.numpy()[0]

      # back to contact frame
      t = mjd.contact.frame[j].reshape(3, 3) @ force[:3]
      b = mjd.contact.frame[j].reshape(3, 3) @ force[3:]
      force = np.concatenate([t, b])
      np.testing.assert_allclose(result, force, rtol=1e-5, atol=2)


if __name__ == "__main__":
  wp.init()
  absltest.main()
