BENCHMARKS = [
  {
    "name": "cloth",
    "mjcf": "scene.xml",
    "nworld": 32,
    "nconmax": 2200,
    "njmax": 8000,
    "override": "opt.jac_preconditioner=true",
  },
  {
    "name": "cloth_render",
    "mjcf": "scene.xml",
    "function": "render",
    "nworld": 32,
    "nconmax": 2200,
    "njmax": 8000,
    "nstep": 200,
    "render_width": 64,
    "render_height": 64,
  },
]
