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

import warp as wp

import mujoco_warp as mjw
from mujoco_warp import GeomType
from mujoco_warp import test_data

collect_ignore = ["benchmark/mujoco_menagerie"]


def pytest_addoption(parser):
  parser.addoption("--cpu", action="store_true", default=False, help="run tests with cpu")
  parser.addoption(
    "--verify_cuda",
    action="store_true",
    default=False,
    help="run tests with cuda error checking",
  )
  parser.addoption("--lineinfo", action="store_true", default=False, help="add lineinfo to warp kernel")


def pytest_configure(config):
  if config.getoption("--cpu"):
    wp.set_device("cpu")
  if config.getoption("--verify_cuda"):
    wp.config.verify_cuda = True
  if config.getoption("--lineinfo"):
    wp.config.lineinfo = True

  ## initialize primitive colliders
  # clear cache
  mjw._src.collision_primitive._PRIMITIVE_COLLISION_TYPES.clear()
  mjw._src.collision_primitive._PRIMITIVE_COLLISION_FUNC.clear()

  # map enum to string
  geom_str = {
    GeomType.PLANE: "plane",
    GeomType.SPHERE: "sphere",
    GeomType.CAPSULE: "capsule",
    GeomType.ELLIPSOID: "ellipsoid",
    GeomType.CYLINDER: "cylinder",
    GeomType.BOX: "box",
    GeomType.MESH: "mesh",
  }

  # generate kernel
  for geomtype in mjw._src.collision_primitive._PRIMITIVE_COLLISIONS.keys():
    if geomtype[0] == GeomType.PLANE:
      obj0 = '<geom type="plane" size="10 10 .01"/>'
    else:
      obj0 = f"""
      <body>
        <geom type="{geom_str[geomtype[0]]}" size=".1 .1 .1"/>
        <joint/>
      </body>
    """

    if geomtype[1] == GeomType.MESH:
      obj1 = f"""
        <body>
          <geom type="{geom_str[geomtype[1]]}" mesh="mesh" size=".1 .1 .1"/>
          <joint/>
        </body>
      """
      mesh = """
      <asset>
        <mesh name="mesh" builtin="sphere" params="0"/>
      </asset>
      """
    else:
      obj1 = f"""
        <body>
          <geom type="{geom_str[geomtype[1]]}" size=".1 .1 .1"/>
          <joint/>
        </body>
      """
      mesh = ""

    _, _, m, d = test_data.fixture(
      xml=f"""
    <mujoco>
      {mesh}
      <worldbody>
       {obj0}
       {obj1}
      </worldbody>
    </mujoco>
    """
    )

    mjw.primitive_narrowphase(m, d)
