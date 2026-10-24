ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  }
]

BENCHMARKS = [
  {
    "name": "franka_threading",
    "mjcf": "scene.xml",
    "nworld": 8192,
    "nconmax": 4096,
    "nccdmax": 512,
    "njmax": 4096,
    "noise_std": 0.0,
    "noise_rate": 1e-6,
    "replay": "threading.npz",
    "assets": [(ASSETS[0], "franka_emika_panda")],
  }
]
