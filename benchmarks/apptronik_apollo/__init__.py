ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  }
]

BENCHMARKS = [
  {
    "name": "apptronik_apollo_flat",
    "mjcf": "scene_flat.xml",
    "nworld": 8192,
    "nconmax": 16,
    "njmax": 64,
    "assets": [(ASSETS[0], "apptronik_apollo")],
  },
  {
    "name": "apptronik_apollo_hfield",
    "mjcf": "scene_hfield.xml",
    "nworld": 8192,
    "nconmax": 32,
    "njmax": 128,
    "assets": [(ASSETS[0], "apptronik_apollo")],
  },
  {
    "name": "apptronik_apollo_terrain",
    "mjcf": "scene_terrain.xml",
    "nworld": 8192,
    "nconmax": 48,
    "njmax": 96,
    "assets": [(ASSETS[0], "apptronik_apollo")],
  },
]
