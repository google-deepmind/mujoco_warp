ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  },
  {
    "source": "https://github.com/google-deepmind/mujoco.git",
    "ref": "4eb987ad2557cf448fc2b61473bb6409b68e50eb",
  },
  {
    "source": "https://github.com/google-deepmind/aloha_sim.git",
    "ref": "d02904607cca1bf6dfb72f30b522506ac7ca0f91",
  },
]

BENCHMARKS = [
  {
    "name": "aloha_pot",
    "mjcf": "scene_pot.xml",
    "nworld": 8192,
    "nconmax": 24,
    "njmax": 128,
    "replay": "lift_pot.npz",
    "assets": [(ASSETS[0], "aloha")],
  },
  {
    "name": "aloha_sdf",
    "mjcf": "scene_sdf.xml",
    "nworld": 8192,
    "nconmax": 32,
    "njmax": 226,
    "assets": [(ASSETS[0], "aloha"), (ASSETS[1], "model/plugin/sdf/asset", "assets")],
  },
  {
    "name": "aloha_cloth",
    "mjcf": "scene_cloth.xml",
    "nworld": 32,
    "nconmax": 4096,
    "njmax": 40_000,
    "nstep": 100,
    "assets": [(ASSETS[0], "aloha")],
  },
  {
    "name": "aloha_clutter",
    "mjcf": "scene_clutter.xml",
    "nworld": 512,
    "nconmax": 512,
    "njmax": 1024,
    "replay": "pick_clutter.npz",
    "assets": [
      (ASSETS[0], "aloha"),
      (ASSETS[2], "aloha_sim/assets/ycb/*/google_64k", "assets/ycb"),
      (ASSETS[2], "aloha_sim/assets/gso/*", "assets/gso"),
    ],
  },
]
