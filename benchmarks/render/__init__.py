ASSETS = [
  {
    "source": "https://github.com/google-deepmind/mujoco.git",
    "ref": "3f3ff85a59b9ce68cfb9d9a5222bf7e050d966c1",
  }
]

BENCHMARKS = [
  {
    "name": "primitives",
    "mjcf": "primitives.xml",
    "function": "render",
    "nworld": 8192,
    "nconmax": 100,
    "njmax": 256,
    "nstep": 100,
  },
  {
    "name": "mug",
    "mjcf": "mug.xml",
    "function": "render",
    "nworld": 8192,
    "nstep": 100,
    "render_width": 64,
    "render_height": 64,
    "assets": [
      (ASSETS[0], "model/mug", "assets"),
    ],
  },
]
