ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco_menagerie.git",
    "ref": "affef0836947b64cc06c4ab1cbf0152835693374",
  }
]

BENCHMARKS = [
  {
    "name": "franka_emika_panda",
    "mjcf": "scene.xml",
    "nworld": 32768,
    "nconmax": 1,
    "njmax": 5,
    "assets": [(ASSETS[0], "franka_emika_panda")],
  }
]

BENCHMARKS += [
  {
    "name": "panda_nist_assembly",
    "mjcf": "scene_nist_k4.mjb",
    "replay": "nist_k4_replay.npz",
    "nworld": 4096,
    "nstep": 2400,
    "nconmax": 600,
    "njmax": 704,
    "nccdmax": 64,
    "noise_std": 0,
    "noise_rate": 0,
    "override": ["opt.jacobian=sparse"],
    "assets": [(ASSETS[0], "franka_emika_panda")],
    "prepare": [
      "python",
      "{input_dir}/contrib/prepare_sdf.py",
      "{benchmark_dir}/scene_nist_k4.xml",
      "--output={benchmark_dir}/scene_nist_k4.mjb",
      "--depth=8",
    ],
  }
]
