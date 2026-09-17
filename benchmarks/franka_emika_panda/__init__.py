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
    "name": "panda_threading",
    "mjcf": "scene_threading.xml",
    "replay": "threading.npz",
    "nworld": 1024,
    "nconmax": 512,
    "njmax": 4096,
    "assets": [(ASSETS[0], "franka_emika_panda")],
  },
]
