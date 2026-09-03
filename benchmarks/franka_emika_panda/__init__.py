ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  }
]

BENCHMARKS = [
  {
    "name": "franka_emika_panda",
    "mjcf": "scene.xml",
    "nworld": 32768,
    "nconmax": 1,
    "njmax": 5,
    "assets": [(ASSETS[0], "franka_emika_panda")],
  },
  {
    "name": "franka_emika_pandas",
    "mjcf": "scene_pandas.xml",
    "replay": "pick_objects.npz",
    "nworld": 8192,
    "nconmax": 48,
    "njmax": 192,
    "assets": [(ASSETS[0], "franka_emika_panda")],
  },
]
