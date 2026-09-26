BENCHMARKS = [
  {
    "name": "cloth",
    "mjcf": "scene.xml",
    "nworld": 256,
    "nconmax": 3600,
    "nccdmax": 3600,
    "njmax": 3600,
  },
  {
    "name": "cloth_render",
    "mjcf": "scene.xml",
    "function": "render",
    "nworld": 32,
    "nconmax": 3600,
    "nccdmax": 3600,
    "njmax": 3600,
    "nstep": 200,
    "render_width": 64,
    "render_height": 64,
  },
]
