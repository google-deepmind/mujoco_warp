BENCHMARKS = [
  {
    "name": "cloth",
    "mjcf": "scene.xml",
    "nworld": 32,
    "nconmax": 1024,
    "njmax": 4096,
  },
  {
    "name": "cloth_render",
    "mjcf": "scene.xml",
    "function": "render",
    "nworld": 32,
    "nconmax": 1024,
    "njmax": 4096,
    "nstep": 200,
    "render_width": 64,
    "render_height": 64,
  },
]
