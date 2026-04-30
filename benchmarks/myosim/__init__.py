ASSETS = [
  {
    "source": "https://github.com/MyoHub/myo_sim.git",
    "ref": "33f3ded946f55adbdcf963c99999587aadaf975f",
  }
]

BENCHMARKS = [
  {
    "name": "myoarm",
    "mjcf": "myo_sim/arm/myoarm.xml",
    "nworld": 8192,
    "nconmax": 16,
    "njmax": 48,
    "assets": [(ASSETS[0], ".", "myo_sim")],
  },
]
