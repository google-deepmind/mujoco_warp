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

import dataclasses
import os

import unittest
import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw


class SetConstTest(unittest.TestCase):

  def test_meaninertia(self):
    # Load a model with some bodies and inertia
    # We create a simple XML on the fly
    xml = """
    <mujoco>
      <worldbody>
        <body pos="0 0 1">
          <joint type="free"/>
          <geom type="sphere" size="0.1"/>
        </body>
        <body pos="1 0 1">
           <joint type="hinge"/>
           <geom type="box" size="0.1 0.2 0.3"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    
    # Run kinematics on host to ensure qM is populated for the reference
    mujoco.mj_kinematics(mjm, mjd)
    mujoco.mj_makeM(mjm, mjd)
    mujoco.mj_setConst(mjm, mjd)
    expected_meaninertia = mjm.stat.meaninertia
    
    # Create Warp model
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm)
    
    # We need to populate d.qM for _set_stat to work, as per our implementation assumption
    # since _set_0 is not implemented yet.
    # We'll copy qM from host MjData
    if hasattr(m, 'opt') and m.opt.is_sparse:
       pass # skip sparse test for now or handle it
    else:
       # Copy qM to device d.qM
       # d.qM shape is (nworld, nv_pad, nv_pad) or similar.
       # We need to be careful with padding.
       # Using mju_fullM to get dense M on host first?
       # mj_makeM produces sparse or dense depending.
       pass # TODO: setup qM correctly
       
    # Actually, let's rely on set_const calling _set_0 which is empty, so we must manually setup qM
    # if we want to test _set_stat in isolation (or as part of set_const).
    # Since _set_0 is a stub, d.qM will be zeros!
    # So we MUST populate d.qM manually here to test _set_stat.
    
    # Get dense M from MuJoCo
    M_dense = np.zeros((mjm.nv, mjm.nv))
    mujoco.mj_fullM(mjm, M_dense, mjd.qM)
    
    # Copy to d.qM
    # We need to handle batch dimension (nworld=1) and padding if any.
    # Check d.qM shape
    qM_device = d.qM.numpy()
    # It might be (1, nv_pad, nv_pad).
    # We just write into top-left corner.
    qM_device[0, :mjm.nv, :mjm.nv] = M_dense
    d.qM = wp.array(qM_device, dtype=float, device=d.qM.device)
    
    # Run set_const (or just _set_stat)
    m_new = mjw.set_const(m, d, mjm)
    
    result_meaninertia = m_new.stat.meaninertia
    
    np.testing.assert_allclose(result_meaninertia, expected_meaninertia, rtol=1e-5)

  def test_set_spring(self):
    # Model with tendons and springs
    xml = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body pos="0 0 2">
          <joint name="j1" type="slide" axis="0 0 1" springdamper="10 1"/>
          <geom type="sphere" size="0.1"/>
        </body>
      </worldbody>
      <tendon>
        <fixed name="t1">
          <joint joint="j1" coef="1"/>
        </fixed>
      </tendon>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    
    # Set qpos_spring to something non-zero
    mjm.qpos_spring[0] = 0.5
    
    # Ensure tendon_lengthspring is -1 (default)
    mjm.tendon_lengthspring[:] = -1
    
    # Run C version
    mujoco.mj_setConst(mjm, mjd)
    expected_lengthspring = mjm.tendon_lengthspring.copy()
    
    # Reset model to ensure we test Warp version properly
    mjm.tendon_lengthspring[:] = -1
    
    # Warp version
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm)
    
    # Run set_const
    m_new = mjw.set_const(m, d, mjm)
    
    result_lengthspring = m_new.tendon_lengthspring.numpy().squeeze()
    
    np.testing.assert_allclose(result_lengthspring, expected_lengthspring.squeeze(), rtol=1e-5)
    
    # Verify qpos was restored (should be 0 as initialized by make_data)
    # make_data initializes qpos to 0 usually (unless qpos0 is used, which is 0 here)
    current_qpos = d.qpos.numpy()[0]
    np.testing.assert_allclose(current_qpos, np.zeros(mjm.nq), atol=1e-6)


if __name__ == '__main__':
  unittest.main()
