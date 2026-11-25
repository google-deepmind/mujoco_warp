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
"""Tests for stage fusion."""

import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp import test_data
from mujoco_warp._src import forward
from mujoco_warp._src import fusion

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <camera pos="1 1 1" mode="targetbodycom" target="b1"/>
    <body name="b1" pos="0 0 1">
      <freejoint/>
      <geom type="capsule" size=".1 .2" pos=".1 0 0"/>
      <site pos="0 .1 0"/>
      <body pos=".3 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="box" size=".1 .1 .1"/>
        <body pos=".3 0 0">
          <joint type="ball"/>
          <geom type="sphere" size=".1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@fusion.stage("nworld", "nbody")
def _double(worldid: int, bodyid: int, a_in: wp.array2d[float], b_out: wp.array2d[float]):
  b_out[worldid, bodyid] = 2.0 * a_in[worldid, bodyid]


@fusion.stage("nworld", "nbody")
def _add_own(worldid: int, bodyid: int, b_in: wp.array2d[float], c_out: wp.array2d[float]):
  c_out[worldid, bodyid] = b_in[worldid, bodyid] + 1.0


@fusion.stage("nworld", "nbody")
def _add_root(worldid: int, bodyid: int, b_in: wp.array2d[float], c_out: wp.array2d[float]):
  c_out[worldid, bodyid] = b_in[worldid, 0] + 1.0


class FusionTest(parameterized.TestCase):
  def test_plan(self):
    """A stage reading only its own thread's result fuses into that thread, other reads don't."""
    _, _, m, d = test_data.fixture(xml=_XML, nworld=2)
    a = wp.array(np.arange(2 * m.nbody, dtype=np.float32).reshape(2, -1))
    b = wp.empty_like(a)
    c = wp.empty_like(a)
    sizes = (d.nworld, m.nbody)

    own = [fusion._Launch(_double, {"a_in": a, "b_out": b}, sizes), fusion._Launch(_add_own, {"b_in": b, "c_out": c}, sizes)]
    self.assertEqual([[[ln.stage for ln in seg] for seg in k] for k in fusion._plan(own)], [[[_double, _add_own]]])
    root = [own[0], fusion._Launch(_add_root, {"b_in": b, "c_out": c}, sizes)]
    self.assertEqual([[[ln.stage for ln in seg] for seg in k] for k in fusion._plan(root)], [[[_double]], [[_add_root]]])

    fusion.launch(m, d, _double, _add_own, a_in=a, b_out=b, b_in=b, c_out=c)
    np.testing.assert_array_equal(c.numpy(), 2.0 * a.numpy() + 1.0)
    fusion.launch(m, d, _double, _add_root, a_in=a, b_out=b, b_in=b, c_out=c)
    np.testing.assert_array_equal(c.numpy(), np.broadcast_to(2.0 * a.numpy()[:, :1] + 1.0, c.shape))

  @parameterized.parameters(1, 2)
  def test_fused_matches_unfused(self, nworld):
    """Fused kinematics write the same values as one kernel per stage, in fewer launches."""
    _, _, m, d = test_data.fixture(xml=_XML, nworld=nworld, qpos_noise=0.3)
    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1] += 0.2
      d.qpos.assign(qpos)

    fields = ("xpos", "xquat", "xipos", "geom_xpos", "site_xpos", "subtree_com", "cinert", "cdof", "cam_xpos", "light_xpos")
    results, nlaunch = [], []
    for fuse in (False, True):
      fusion.FUSE = fuse
      fusion._COMMANDS.clear()
      try:
        for f in fields:
          getattr(d, f).fill_(wp.inf)
        forward.fwd_kinematics(m, d)
      finally:
        fusion.FUSE = True
      results.append({f: getattr(d, f).numpy() for f in fields})
      nlaunch.append(sum(len(c) for c in fusion._COMMANDS.values()))

    for f in fields:
      np.testing.assert_array_equal(results[1][f], results[0][f], err_msg=f)
    self.assertEqual(nlaunch, [10, 5])
    if nworld == 2:
      self.assertFalse(np.array_equal(results[1]["xpos"][0], results[1]["xpos"][1]))


if __name__ == "__main__":
  absltest.main()
