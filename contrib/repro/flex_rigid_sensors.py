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

"""Reproduce flex contacts contaminating unrelated rigid force/touch sensors.

This standalone diagnostic is not a fix and is not collected by pytest.
"""

import argparse
import json

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cpu", help="Warp device, e.g. cpu or cuda:0")
  parser.add_argument("--sensor", choices=("force", "touch", "both"), default="both")
  parser.add_argument("--flex-collisions", choices=("on", "off"), default="on")
  parser.add_argument("--nworld", type=int, choices=(1, 2), default=1)
  args = parser.parse_args()
  wp.init()
  wp.set_device(args.device)
  collide = int(args.flex_collisions == "on")

  # Binary-exact sphere radius/height keep the rigid-only control within 1e-5.
  # The large site contains both contact regions, but belongs only to the sphere.
  mjm = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option gravity=".25 .5 -.75" cone="elliptic" jacobian="sparse" solver="Newton" iterations="30">
        <flag warmstart="disable"/>
      </option>
      <worldbody>
        <geom type="plane" size="3 3 .1"/>
        <body name="rigid" pos="1 0 .123046875">
          <freejoint/><geom type="sphere" size=".125" mass="1"/>
          <site name="probe" type="sphere" size="2"/>
        </body>
        <flexcomp name="cloth" type="grid" count="2 2 1" spacing=".1 .1 .1" pos="-.5 0 .004" dim="2" mass="1">
          <contact condim="3" selfcollide="none" contype="{collide}" conaffinity="{collide}"/>
        </flexcomp>
      </worldbody>
      <sensor><force site="probe"/><touch site="probe"/></sensor>
    </mujoco>
  """)
  rigid_body = mjm.body("rigid").id
  native = []
  for world in range(args.nworld):
    mjd = mujoco.MjData(mjm)
    mjd.qpos[2] += world * 0.00048828125
    mjd.qvel[:] = (world + 1) * 0.03125
    mujoco.mj_forward(mjm, mjd)
    assert np.any(mjd.contact.geom < 0) == bool(collide), "native flex contact control is inactive"
    native.append(mjd)

  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=args.nworld, nconmax=32, njmax=128)
  # Copy only independent inputs, never native contacts, forces, or sensor outputs.
  d.qpos.assign(np.stack([mjd.qpos.copy() for mjd in native]).astype(np.float32))
  d.qvel.assign(np.stack([mjd.qvel.copy() for mjd in native]).astype(np.float32))
  d.sensordata.fill_(wp.inf)
  d.cfrc_ext.fill_(wp.inf)
  d.cfrc_int.fill_(wp.inf)
  mjw.forward(m, d)
  wp.synchronize()

  actual = d.sensordata.numpy().copy()
  expected = np.stack([mjd.sensordata.copy() for mjd in native])
  cfrc_ext = d.cfrc_ext.numpy().copy()
  cfrc_int = d.cfrc_int.numpy().copy()
  overflow = d.overflow.numpy().copy()
  ncontact = int(d.nacon.numpy().copy()[0])
  geoms = d.contact.geom.numpy().copy()[:ncontact]
  flex = d.contact.flex.numpy().copy()[:ncontact]
  worldids = d.contact.worldid.numpy().copy()[:ncontact]
  finite = all(np.isfinite(value).all() for value in (actual, expected, cfrc_ext, cfrc_int, d.qacc.numpy().copy()))
  report = {
    "device": args.device,
    "sensor": args.sensor,
    "flex_collisions": bool(collide),
    "nworld": args.nworld,
    "finite": bool(finite),
    "overflow": overflow.tolist(),
    "native_contacts": [mjd.ncon for mjd in native],
    "warp_contacts": ncontact,
    "contact_geom": geoms.tolist(),
    "contact_flex": flex.tolist(),
    "contact_worldid": worldids.tolist(),
    "native_force": expected[:, :3].tolist(),
    "warp_force": actual[:, :3].tolist(),
    "native_touch": expected[:, 3:].tolist(),
    "warp_touch": actual[:, 3:].tolist(),
    "native_rigid_cfrc_ext": [mjd.cfrc_ext[rigid_body].copy().tolist() for mjd in native],
    "warp_rigid_cfrc_ext": cfrc_ext[:, rigid_body].tolist(),
    "native_rigid_cfrc_int": [mjd.cfrc_int[rigid_body].copy().tolist() for mjd in native],
    "warp_rigid_cfrc_int": cfrc_int[:, rigid_body].tolist(),
  }
  print(json.dumps(report, indent=2), flush=True)

  assert finite, "non-finite forward output"
  np.testing.assert_array_equal(overflow, 0, err_msg="buffer overflow invalidates the comparison")
  for world in range(args.nworld):
    contacts = geoms[worldids == world]
    assert np.any(np.all(contacts >= 0, axis=1)), "missing rigid contact"
    assert np.any(contacts < 0) == bool(collide), "Warp flex contact control is inactive"
  if args.nworld == 2:
    assert not np.allclose(expected[0, 3:], expected[1, 3:]), "worlds must have different native touch loads"

  selected = {"force": slice(0, 3), "touch": slice(3, 4), "both": slice(None)}[args.sensor]
  np.testing.assert_allclose(
    actual[:, selected], expected[:, selected], rtol=1e-5, atol=1e-5, err_msg=f"{args.sensor} sensor parity"
  )


if __name__ == "__main__":
  main()
