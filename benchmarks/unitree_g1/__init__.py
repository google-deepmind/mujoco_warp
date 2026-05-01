ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  }
]

BENCHMARKS = [
  {
    "name": "unitree_g1_flat",
    "mjcf": "scene_flat.xml",
    "nworld": 8192,
    "nconmax": 48,
    "njmax": 192,
    "replay": "shuffle_dance.npz",
    "assets": [(ASSETS[0], "unitree_g1/assets", "assets")],
  },
  {
    "name": "unitree_g1_hfield",
    "mjcf": "scene_hfield.xml",
    "nworld": 8192,
    "nconmax": 48,
    "njmax": 192,
    "replay": "shuffle_dance.npz",
    "assets": [(ASSETS[0], "unitree_g1/assets", "assets")],
  },
]
