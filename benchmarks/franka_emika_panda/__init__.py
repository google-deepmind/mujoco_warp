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
    "name": f"panda_nist_k4_states_{layout}",
    "mjcf": "scene_nist_k4.mjb",
    "state_profile": "nist_k4_states.npz",
    "nworld": 4096,
    "nstep": 559,
    "nconmax": 600,
    "njmax": 704,
    "nccdmax": 64,
    "noise_std": 0,
    "noise_rate": 0,
    "override": [f"opt.jacobian={layout}"],
    "assets": [(ASSETS[0], "franka_emika_panda")],
    "prepare": [
      "python",
      "{input_dir}/contrib/prepare_sdf.py",
      "{benchmark_dir}/scene_nist_k4.xml",
      "--output={benchmark_dir}/scene_nist_k4.mjb",
      "--depth=8",
    ],
  }
  for layout in ("dense", "sparse")
]
