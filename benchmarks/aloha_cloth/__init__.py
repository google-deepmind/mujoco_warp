ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  }
]

BENCHMARKS = [
  {
    "name": "aloha_cloth",
    "mjcf": "scene.xml",
    "nworld": 32,
    "nconmax": 4096,
    "njmax": 40_000,
    "nstep": 100,
    "assets": [(ASSETS[0], "aloha")],
  }
]
