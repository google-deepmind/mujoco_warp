BENCHMARKS = [
  {
    "name": "cloth",
    "mjcf": "scene.xml",
    "nworld": 512,
    "nconmax": 1024,
    "njmax": 4096,
  },
  {
    "name": "cloth_render",
    "mjcf": "scene.xml",
    "function": "render",
    "nworld": 512,
    "nconmax": 1024,
    "njmax": 4096,
    "nstep": 200,
    "render_width": 64,
    "render_height": 64,
  },
]
