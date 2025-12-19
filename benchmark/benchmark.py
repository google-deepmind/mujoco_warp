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

from mujoco_warp._src import benchmark


class AlohaCloth(benchmark.BenchmarkSuite):
  """Aloha robot with a towel on the workbench."""

  path = "aloha_cloth/scene.xml"
  params = benchmark.BenchmarkSuite.params + ("step.euler",)
  batch_size = 256
  nconmax = 920
  njmax = 3606


class AlohaPot(benchmark.BenchmarkSuite):
  """Aloha robot with a pasta pot on the workbench."""

  path = "aloha_pot/scene.xml"
  params = benchmark.BenchmarkSuite.params + ("step.euler",)
  batch_size = 8192
  nconmax = 24
  njmax = 128
  replay = "lift_pot"


class AlohaSdf(benchmark.BenchmarkSuite):
  """Aloha robot with SDF grippers and an SDF asset."""

  path = "aloha_sdf/scene.xml"
  params = benchmark.BenchmarkSuite.params + ("step.euler",)
  batch_size = 8192
  nconmax = 32
  njmax = 226


class ApptronikApolloFlat(benchmark.BenchmarkSuite):
  """Apptronik Apollo locomoting on an infinite plane."""

  path = "apptronik_apollo/scene_flat.xml"
  params = benchmark.BenchmarkSuite.params + ("step.euler",)
  batch_size = 8192
  nconmax = 16
  njmax = 64


class ApptronikApolloHfield(benchmark.BenchmarkSuite):
  """Apptronik Apollo locomoting on a pyramidal hfield."""

  path = "apptronik_apollo/scene_hfield.xml"
  params = benchmark.BenchmarkSuite.params + ("step.euler",)
  batch_size = 8192
  nconmax = 32
  njmax = 128


class ApptronikApolloTerrain(benchmark.BenchmarkSuite):
  """Apptronik Apollo locomoting on Isaac-style pyramids made of thousands of boxes."""

  path = "apptronik_apollo/scene_terrain.xml"
  params = benchmark.BenchmarkSuite.params + ("step.euler",)
  batch_size = 8192
  nconmax = 48
  njmax = 96


class Cloth(benchmark.BenchmarkSuite):
  """Draping of a cloth over the MuJoCo humanoid."""

  path = "cloth/scene.xml"
  params = benchmark.BenchmarkSuite.params + ("step.euler",)
  batch_size = 256
  nconmax = 200
  njmax = 600


class FrankaEmikaPanda(benchmark.BenchmarkSuite):
  """Franka Emika Panda on an infinite plane."""

  path = "franka_emika_panda/scene.xml"
  params = benchmark.BenchmarkSuite.params + ("step.implicit",)
  batch_size = 32768
  nconmax = 1
  njmax = 5


class Humanoid(benchmark.BenchmarkSuite):
  """MuJoCo humanoid on an infinite plane."""

  path = "humanoid/humanoid.xml"
  params = benchmark.BenchmarkSuite.params + ("step.euler",)
  batch_size = 8192
  nconmax = 24
  njmax = 64


class MyoArm(benchmark.BenchmarkSuite):
  """MyoArm."""

  path = "myo_sim/arm/myoarm.xml"
  params = benchmark.BenchmarkSuite.params + ("step.euler",)
  batch_size = 8192
  nconmax = 16
  njmax = 48
  overrides = {"opt.ccd_iterations": 50}  # default 35 is too low


class ThreeHumanoids(benchmark.BenchmarkSuite):
  """Three MuJoCo humanoids on an infinite plane.

  Ideally, simulation time scales linearly with number of humanoids.
  """

  path = "humanoid/n_humanoid.xml"
  params = benchmark.BenchmarkSuite.params + ("step.euler",)
  # TODO: use batch_size=8192 once performance is fixed
  batch_size = 1024
  nconmax = 100
  njmax = 192


# attach a setup_cache to each test for one-time setup of benchmarks
AlohaPot.setup_cache = lambda s: benchmark.BenchmarkSuite.setup_cache(s)
AlohaSdf.setup_cache = lambda s: benchmark.BenchmarkSuite.setup_cache(s)
ApptronikApolloFlat.setup_cache = lambda s: benchmark.BenchmarkSuite.setup_cache(s)
ApptronikApolloHfield.setup_cache = lambda s: benchmark.BenchmarkSuite.setup_cache(s)
ApptronikApolloTerrain.setup_cache = lambda s: benchmark.BenchmarkSuite.setup_cache(s)
Cloth.setup_cache = lambda s: benchmark.BenchmarkSuite.setup_cache(s)
FrankaEmikaPanda.setup_cache = lambda s: benchmark.BenchmarkSuite.setup_cache(s)
Humanoid.setup_cache = lambda s: benchmark.BenchmarkSuite.setup_cache(s)
MyoArm.setup_cache = lambda s: benchmark.BenchmarkSuite.setup_cache(s)
ThreeHumanoids.setup_cache = lambda s: benchmark.BenchmarkSuite.setup_cache(s)
