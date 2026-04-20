import importlib
import os
import re
import sys
from datetime import datetime
from typing import Sequence

from absl import app
from absl import flags
import mujoco
import warp as wp

import mujoco_warp as mjw

_FILTER = flags.DEFINE_string("filter", "", "Filter benchmarks by name (regex)", short_name="f")


def _main(argv: Sequence[str]):
  script_dir = os.path.dirname(os.path.abspath(__file__))

  wp.config.quiet = True
  wp.init()

  # Find all directories in benchmarks/
  for item in os.listdir(script_dir):
    item_path = os.path.join(script_dir, item)
    if not os.path.isdir(item_path) or not os.path.exists(os.path.join(item_path, "__init__.py")):
      continue
    module = importlib.import_module(item)
    if not hasattr(module, "BENCHMARKS"):
      continue
    for bm in module.BENCHMARKS:
      name = bm["name"]

      if _FILTER.value and not re.search(_FILTER.value, name):
        continue

      mjm = mujoco.MjModel.from_xml_path(os.path.join(script_dir, item, bm["mjcf"]))
      mjd = mujoco.MjData(mjm)

      m, d, metrics = mjw.benchmark.init(mjm, mjd, bm["nworld"], bm["nconmax"], bm["njmax"])
      metrics = mjw.benchmark.run(mjw.step, m, d, metrics, bm.get("nstep", 1000), None, True)

      for metric in metrics.to_short().split("\n"):
        print(f"{name}.{metric}")


if __name__ == "__main__":
  app.run(_main)
