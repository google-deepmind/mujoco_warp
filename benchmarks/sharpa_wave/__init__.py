ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "370c43965323c0944b433b202aac5058b5f907a9",
  },
  {
    "source": "https://github.com/google-deepmind/mujoco.git",
    "ref": "4eb987ad2557cf448fc2b61473bb6409b68e50eb",
  },
]

BENCHMARKS = [
  {
    "name": "sharpa_wave",
    "mjcf": "scene.xml",
    "nworld": 8192,
    "nconmax": 48,
    "nccdmax": 8,
    "njmax": 160,
    "replay": "grasp_mug.npz",
    "assets": [
      (ASSETS[0], "sharpa_wave/assets/right", "assets/right"),
      (ASSETS[1], "model/mug", "assets/mug"),
    ],
  },
]
