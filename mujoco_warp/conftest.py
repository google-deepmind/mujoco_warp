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

import os

import numpy as np
import warp as wp


# Default tolerance for test assertions
TEST_TOLERANCE = 5e-5


def assert_allclose(a, b, name, tolerance_multiplier=10):
  """Assert arrays are close within tolerance.

  Args:
    a: First array to compare.
    b: Second array to compare.
    name: Name of the comparison for error messages.
    tolerance_multiplier: Multiplier applied to TEST_TOLERANCE (default 10).
  """
  tol = TEST_TOLERANCE * tolerance_multiplier
  err_msg = f"mismatch: {name}"
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=tol, rtol=tol)


def pytest_addoption(parser):
  parser.addoption("--cpu", action="store_true", default=False, help="run tests with cpu")
  parser.addoption(
    "--verify_cuda",
    action="store_true",
    default=False,
    help="run tests with cuda error checking",
  )
  parser.addoption("--lineinfo", action="store_true", default=False, help="add lineinfo to warp kernel")
  parser.addoption("--optimization_level", action="store", default=None, type=int, help="set wp.config.optimization_level")
  parser.addoption("--debug_mode", action="store_true", default=False, help="debug mode compilation")
  parser.addoption(
    "--kernel_cache_dir",
    action="store",
    default=None,
    help="directory path for storing compiled warp kernel cache",
  )


def pytest_configure(config):
  kernel_cache_dir = config.getoption("--kernel_cache_dir")
  if kernel_cache_dir:
    wp.config.kernel_cache_dir = os.path.expanduser(kernel_cache_dir)
  if config.getoption("--cpu"):
    wp.set_device("cpu")
  if config.getoption("--verify_cuda"):
    wp.config.verify_cuda = True
  if config.getoption("--lineinfo"):
    wp.config.lineinfo = True
  wp.config.optimization_level = config.getoption("--optimization_level")
  if config.getoption("--debug_mode"):
    wp.config.mode = "debug"
