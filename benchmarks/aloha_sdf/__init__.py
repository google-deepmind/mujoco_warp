ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  },
  {
    "source": "https://github.com/google-deepmind/mujoco.git",
    "ref": "4eb987ad2557cf448fc2b61473bb6409b68e50eb",
  },
]

BENCHMARKS = [
  {
    "name": "aloha_sdf",
    "mjcf": "scene.xml",
    "nworld": 8192,
    "nconmax": 32,
    "njmax": 226,
    "assets": [(ASSETS[0], "aloha"), (ASSETS[1], "model/plugin/sdf/asset", "assets")],
  }
]
