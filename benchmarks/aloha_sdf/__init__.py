ASSETS = [
  {
    "source": "git@github.com:google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  }
]

BENCHMARKS = [
  {
    "name": "aloha_sdf",
    "mjcf": "scene.xml",
    "nworld": 8192,
    "nconmax": 32,
    "njmax": 226,
    "assets": [(ASSETS[0], "aloha")],
  }
]
