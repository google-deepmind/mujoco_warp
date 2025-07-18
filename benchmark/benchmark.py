import mujoco_warp


class AlohaPot(mujoco_warp.BenchmarkSuite):
  """Aloha robot with a pasta pot on the workbench."""

  path = "aloha_pot/scene.xml"
  batch_size = 8192
  nconmax = 200_000
  njmax = 200_000


class ApptronikApolloFlat(mujoco_warp.BenchmarkSuite):
  """Apptronik Apollo locomoting on an infinite plain."""

  path = "apptronik_apollo/scene_flat.xml"
  batch_size = 8192
  nconmax = 200_000
  njmax = 500_000


class ApptronikApolloHfield(mujoco_warp.BenchmarkSuite):
  """Apptronik Apollo locomoting on a pyramidal hfield."""

  path = "apptronik_apollo/scene_hfield.xml"
  batch_size = 1024
  nconmax = 700_000
  njmax = 50_000


class ApptronikApolloTerrain(mujoco_warp.BenchmarkSuite):
  """Apptronik Apollo locomoting on Isaac-style pyramids made of thousands of boxes."""

  path = "apptronik_apollo/scene_terrain.xml"
  batch_size = 8192
  nconmax = 400_000
  njmax = 600_000


class FrankaEmikaPanda(mujoco_warp.BenchmarkSuite):
  """Franka Emika Panda on an infinite plain."""

  path = "franka_emika_panda/scene.xml"
  batch_size = 32768
  nconmax = 10_000
  njmax = 150_000


class Humanoid(mujoco_warp.BenchmarkSuite):
  """MuJoCo humanoid on an infinite plain."""

  path = "humanoid/humanoid.xml"
  batch_size = 8192
  nconmax = 200_000
  njmax = 500_000


class ThreeHumanoids(mujoco_warp.BenchmarkSuite):
  """Three MuJoCo humanoids on an infinite plain.

  Ideally, simulation time scales linearly with number of humanoids.
  """

  repeat = 100
  path = "humanoid/n_humanoid.xml"
  batch_size = 1024
  nconmax = 50_000
  njmax = 150_000
