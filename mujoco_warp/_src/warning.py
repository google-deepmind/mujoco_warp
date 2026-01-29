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
"""Warning utilities for kernel-side overflow detection."""

import sys
import warnings
from typing import List

from . import types

_WARNING_MESSAGES = {
    types.WarningType.NEFC_OVERFLOW: "nefc overflow - increase njmax to {0}",
    types.WarningType.BROADPHASE_OVERFLOW: (
        "broadphase overflow - increase nconmax to {0} or naconmax to {1}"
    ),
    types.WarningType.NARROWPHASE_OVERFLOW: (
        "narrowphase overflow - increase nconmax to {0} or naconmax to {1}"
    ),
    types.WarningType.CONTACT_MATCH_OVERFLOW: (
        "contact match overflow - increase Option.contact_sensor_maxmatch to {0}"
    ),
    types.WarningType.GJK_ITERATIONS: (
        "GJK did not converge - increase opt.ccd_iterations (currently {0})"
    ),
    types.WarningType.EPA_HORIZON: "EPA horizon overflow - horizon size {0} insufficient",
}


def check_warnings(d: types.Data, clear: bool = True) -> List[str]:
  """Check warning flags and emit to stderr.

  This function reads the warning flags set by kernels and emits appropriate
  warning messages to stderr. Warning flags accumulate across simulation steps
  (using atomic_max), so this should be called after graph execution completes.

  Args:
    d: The Data object containing warning flags.
    clear: Whether to clear warning flags after checking. Default True.

  Returns:
    List of warning message strings that were emitted.
  """
  flags = d.warning.numpy()
  info = d.warning_info.numpy()

  emitted = []
  for wtype in types.WarningType:
    if flags[wtype]:
      msg = _WARNING_MESSAGES[wtype].format(info[wtype, 0], info[wtype, 1])
      warnings.warn(msg)
      emitted.append(msg)

  if clear:
    d.warning.zero_()
    d.warning_info.zero_()

  return emitted


def get_warnings(d: types.Data) -> List[str]:
  """Get warning messages without emitting or clearing.

  Args:
    d: The Data object containing warning flags.

  Returns:
    List of warning message strings.
  """
  flags = d.warning.numpy()
  info = d.warning_info.numpy()

  messages = []
  for wtype in types.WarningType:
    if flags[wtype]:
      msg = _WARNING_MESSAGES[wtype].format(info[wtype, 0], info[wtype, 1])
      messages.append(msg)

  return messages


def clear_warnings(d: types.Data) -> None:
  """Clear all warning flags.

  Args:
    d: The Data object containing warning flags.
  """
  d.warning.zero_()
  d.warning_info.zero_()

