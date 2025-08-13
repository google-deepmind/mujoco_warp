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

import mujoco_warp


class AlohaPot(mujoco_warp.BenchmarkSuite):
  """Aloha robot with a pasta pot on the workbench."""

  path = "aloha_pot/scene.xml"
  batch_size = 8192
  nconmax = 200_000
  njmax = 128


class ApptronikApolloFlat(mujoco_warp.BenchmarkSuite):
  """Apptronik Apollo locomoting on an infinite plane."""

  path = "apptronik_apollo/scene_flat.xml"
  params = mujoco_warp.BenchmarkSuite.params + ("step.euler",)
  batch_size = 8192
  nconmax = 100_000
  njmax = 64


# TODO(team): uncomment once the scene is stable
# class ApptronikApolloHfield(mujoco_warp.BenchmarkSuite):
#   """Apptronik Apollo locomoting on a pyramidal hfield."""

#   path = "apptronik_apollo/scene_hfield.xml"
#   params = mujoco_warp.BenchmarkSuite.params + ("step.euler",)
#   batch_size = 1024
#   nconmax = 700_000
#   njmax = 128


class ApptronikApolloTerrain(mujoco_warp.BenchmarkSuite):
  """Apptronik Apollo locomoting on Isaac-style pyramids made of thousands of boxes."""

  path = "apptronik_apollo/scene_terrain.xml"
  params = mujoco_warp.BenchmarkSuite.params + ("step.euler",)
  batch_size = 8192
  nconmax = 400_000
  njmax = 96


class FrankaEmikaPanda(mujoco_warp.BenchmarkSuite):
  """Franka Emika Panda on an infinite plane."""

  path = "franka_emika_panda/scene.xml"
  params = mujoco_warp.BenchmarkSuite.params + ("step.implicit",)
  batch_size = 32768
  nconmax = 10_000
  njmax = 5


class Humanoid(mujoco_warp.BenchmarkSuite):
  """MuJoCo humanoid on an infinite plane."""

  path = "humanoid/humanoid.xml"
  params = mujoco_warp.BenchmarkSuite.params + ("step.euler",)
  batch_size = 8192
  nconmax = 200_000
  njmax = 64


class ThreeHumanoids(mujoco_warp.BenchmarkSuite):
  """Three MuJoCo humanoids on an infinite plane.
  Ideally, simulation time scales linearly with number of humanoids.
  """

  path = "humanoid/n_humanoid.xml"
  params = mujoco_warp.BenchmarkSuite.params + ("step.euler",)
  # TODO: use batch_size=8192 once performance is fixed
  batch_size = 1024
  nconmax = 100_000
  njmax = 192


# attach a setup_cache to each test for one-time setup of benchmarks
ApptronikApolloFlat.setup_cache = lambda s: mujoco_warp.BenchmarkSuite.setup_cache(s)
ApptronikApolloHfield.setup_cache = lambda s: mujoco_warp.BenchmarkSuite.setup_cache(s)
ApptronikApolloTerrain.setup_cache = lambda s: mujoco_warp.BenchmarkSuite.setup_cache(s)
FrankaEmikaPanda.setup_cache = lambda s: mujoco_warp.BenchmarkSuite.setup_cache(s)
Humanoid.setup_cache = lambda s: mujoco_warp.BenchmarkSuite.setup_cache(s)
ThreeHumanoids.setup_cache = lambda s: mujoco_warp.BenchmarkSuite.setup_cache(s)
