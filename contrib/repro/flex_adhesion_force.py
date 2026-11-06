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

"""Manual native-parity reproducer; adhesive flex dynamics are still incomplete."""

import argparse
import json

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--adhesion", type=float, default=12.0)
  parser.add_argument("--gap", action="store_true", help="Use a true gap band instead of a matching contact margin.")
  args = parser.parse_args()
  wp.set_device(args.device)
  mjm = mujoco.MjModel.from_xml_string("""
  <mujoco>
    <option gravity="0 0 -1" jacobian="dense" iterations="100" tolerance="1e-10">
      <flag warmstart="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="5 5 .1" adhesion="12" margin=".04" condim="1"/>
      <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".1 .1 .1"
                dim="2" mass="1" pos="0 0 .03" radius=".02">
        <edge equality="true"/>
        <contact selfcollide="none" condim="1"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  """)
  mjm.geom_adhesion[:] = args.adhesion
  if args.gap:
    mjm.geom_margin[:] = 0
    mjm.geom_gap[:] = 0.04
  native = mujoco.MjData(mjm)
  native.qvel[2::3] = 2
  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=1, nconmax=32, njmax=128)
  # Copy inputs only: no native contacts, force buffers, or constraint rows.
  d.qpos.assign(native.qpos[None].astype(np.float32))
  d.qvel.assign(native.qvel[None].astype(np.float32))
  for name in ("qfrc_adhesion", "qfrc_passive", "qacc"):
    getattr(d, name).fill_(wp.inf)
  mujoco.mj_forward(mjm, native)
  mjw.forward(m, d)
  ncon = int(d.nacon.numpy()[0])
  native_force = np.zeros((native.ncon, 6))
  for cid in range(native.ncon):
    mujoco.mj_contactForce(mjm, native, cid, native_force[cid])
  force = wp.zeros(ncon, dtype=wp.spatial_vector)
  ids = wp.array(np.arange(ncon, dtype=np.int32), dtype=int)
  mjw.contact_force(m, d, ids, False, force)
  actual = {name: getattr(d, name).numpy().copy()[0] for name in ("qfrc_adhesion", "qfrc_passive", "qacc")}
  print(
    json.dumps(
      {
        "device": args.device,
        "adhesion": args.adhesion,
        "gap": args.gap,
        "contacts": {"native": native.ncon, "warp": ncon},
        "normal_force": {"native": native_force[:, 0].tolist(), "warp": force.numpy()[:, 0].tolist()},
        "vertical": {
          name: {"native": getattr(native, name)[2::3].tolist(), "warp": value[2::3].tolist()} for name, value in actual.items()
        },
        "overflow": d.overflow.numpy().tolist(),
      },
      indent=2,
    ),
    flush=True,
  )
  np.testing.assert_array_equal(d.overflow.numpy(), 0)
  assert ncon == native.ncon == 4, "Separated flex contacts are missing from the adhesion gap band"
  for name, value in actual.items():
    np.testing.assert_allclose(value, getattr(native, name), rtol=1e-5, atol=1e-5, err_msg=name)
  np.testing.assert_allclose(force.numpy(), native_force, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  main()
