# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Global switch controlling whether differentiable (backward-enabled) kernels are compiled.

Modules on the gradient path (forward, smooth, passive, derivative,
collision_smooth) compile their kernels with enable_backward set from this
flag. Backward-enabled kernels generate measurably slower forward code, so
the flag defaults to off: the default forward-only step keeps full
performance, and gradient support is opted into via enable_ad() (called
automatically by make_diff_data / enable_grad).

Toggling the flag changes Warp module options, which triggers a module
recompile on the next kernel launch. Prefer enabling AD once, before the
first step, to avoid recompilation; mixing AD and non-AD stepping in one
process works but pays a recompile each time the mode flips.

The flag can also be preset with the MJWARP_ENABLE_AD environment variable
(any value but '0'/''/'false' enables it).
"""

import os

import warp as wp

_AD_MODULES = (
  "mujoco_warp._src.forward",
  "mujoco_warp._src.smooth",
  "mujoco_warp._src.passive",
  "mujoco_warp._src.derivative",
  "mujoco_warp._src.collision_smooth",
  "mujoco_warp._src.history",
)

_enabled = os.environ.get("MJWARP_ENABLE_AD", "0").lower() not in ("", "0", "false")


def ad_enabled() -> bool:
  """Returns True if differentiable kernels are enabled."""
  return _enabled


def enable_ad() -> None:
  """Compile gradient-path modules with backward kernels enabled.

  Idempotent. If forward-only kernels were already compiled, the affected
  modules are recompiled on the next launch.
  """
  _set_ad(True)


def disable_ad() -> None:
  """Compile gradient-path modules forward-only (full forward performance)."""
  _set_ad(False)


def _set_ad(value: bool) -> None:
  global _enabled
  if _enabled == value:
    return
  _enabled = value
  import sys

  for name in _AD_MODULES:
    mod = sys.modules.get(name)
    if mod is not None:
      wp.set_module_options({"enable_backward": value}, module=mod)
