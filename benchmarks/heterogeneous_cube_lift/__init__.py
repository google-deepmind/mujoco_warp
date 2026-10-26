ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  }
]

BENCHMARKS = [
  {
    "name": "franka_cube_lift",
    "mjcf": "franka.xml",
    "nworld": 8192,
    "nconmax": 64,
    "nccdmax": 64,
    "njmax": 256,
    "noise_std": 0.0,
    "noise_rate": 1e-6,
    "replay": "franka.npz",
    "assets": [(ASSETS[0], "franka_emika_panda", "franka_emika_panda")],
  },
  {
    "name": "aloha_cube_lift",
    "mjcf": "aloha.xml",
    "nworld": 8192,
    "nconmax": 64,
    "nccdmax": 64,
    "njmax": 256,
    "noise_std": 0.0,
    "noise_rate": 1e-6,
    "replay": "aloha.npz",
    "assets": [(ASSETS[0], "aloha", "aloha")],
  },
  {
    "name": "so101_cube_lift",
    "mjcf": "so101.xml",
    "nworld": 8192,
    "nconmax": 64,
    "nccdmax": 64,
    "njmax": 256,
    "noise_std": 0.0,
    "noise_rate": 1e-6,
    "replay": "so101.npz",
    "assets": [(ASSETS[0], "robotstudio_so101", "robotstudio_so101")],
  },
]
