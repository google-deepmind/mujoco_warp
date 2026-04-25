ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  }
]

BENCHMARKS = [
  {
    "name": "aloha_pot",
    "mjcf": "scene.xml",
    "nworld": 8192,
    "nconmax": 24,
    "njmax": 128,
    "replay": "lift_pot",
    "assets": [(ASSETS[0], "aloha")],
  },
]
