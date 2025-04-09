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

wp.config.verify_cuda = True

# tolerance for difference between MuJoCo and MJWarp support calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


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


if __name__ == "__main__":
  wp.init()
  absltest.main()
