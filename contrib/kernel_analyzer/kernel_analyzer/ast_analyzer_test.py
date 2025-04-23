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
"""Tests for the kernel analyzer using direct string testing."""

import os
import pathlib
from typing import Any, List, Type

import ast_analyzer
from absl.testing import absltest

# Test code snippets
_DEFAULT_PARAMS_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_default_params(qpos0: int, qvel: int = 0):
    pass
"""

_VARARGS_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_varargs(qpos0: int, *args):
    pass
"""

_KWARGS_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_kwargs(qpos0: int, **kwargs):
    pass
"""

_TYPE_ISSUE_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_type_issue(qpos0: str, qvel):
    pass
"""

_TYPE_MISMATCH_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_type_mismatch(qpos0: array, geom_pos: array2d):
    pass
"""

_MODEL_SUFFIX_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_model_suffix(qpos0_in: int):
    pass
"""

_DATA_SUFFIX_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_data_suffix(qpos_invalid: int):
    pass
"""

_MISSING_COMMENT_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_missing_comment(
    qpos0: int,
    qvel: int,
):
    pass
"""

_WRITE_READONLY_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_write_readonly(qpos0: int, qvel_in: int):
    qpos0 = 1  # Writing to Model field
    qvel_in = 2  # Writing to Data _in field
"""

_ALL_ISSUES_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_all_issues(
    haha,               # No type
    qpos0: str,         # Type mismatch with Model field
    qvel_invalid: int,  # Invalid data field suffix
    geom_pos_in: int,   # Model field with suffix
    custom_param: int,  # Non-model/data in the middle
    act_in: int,        # Data order issue (in after out)
    qvel_out: int,      # Out before in
    qpos: int = 0,      # Default param
    *args,              # Varargs
    **kwargs            # Kwargs
):
    qpos0 = 1  # Writing to Model field
    act_in = 2  # Writing to Data _in field
"""

_NO_ISSUES_CODE = """
import warp as wp
from mujoco_warp.warp_util import kernel

@kernel
def test_no_issues(
    # Model:
    qpos0: wp.array(dtype=wp.float32, ndim=1),
    geom_pos: wp.array(dtype=wp.vec3, ndim=1),
    # Data in:
    qpos_in: wp.array(dtype=wp.float32, ndim=2),
    qvel_in: wp.array(dtype=wp.float32, ndim=2),
    act_in: wp.array(dtype=wp.float32, ndim=2),
    # Data out:
    act_out: wp.array(dtype=wp.float32, ndim=2)
):
    x = qpos0  # Reading Model field is fine
    y = act_in  # Reading Data _in field is fine
    act_out = 1  # Writing to Data _out field is fine
"""

_NON_KERNEL_CODE = """
import warp as wp

def test_non_kernel(qpos0: int = 0, *args, **kwargs):
    qpos0 = 1
"""


def _analyze_str(code_str: str) -> List[Any]:
  full_path = os.path.realpath(__file__)
  path = pathlib.Path(full_path).parent / '../../../mujoco_warp/_src/types.py'
  return ast_analyzer.analyze(code_str, "somefile.py", path.read_text())


def _assert_has_issue(issues, issue_type: Type):
  """Assert that the issues list contains at least one issue of the given type."""
  if not any(isinstance(issue, issue_type) for issue in issues):
    raise AssertionError(
      f"Expected issue of type {issue_type.__name__} not found in issues."
    )


class TestAnalyzer(absltest.TestCase):
  """Tests for the kernel analyzer."""

  def test_default_params_issue(self):
    """Test that default parameters raise an issue."""
    issues = _analyze_str(_DEFAULT_PARAMS_CODE)
    _assert_has_issue(issues, ast_analyzer.DefaultForbidden)

  def test_varargs_issue(self):
    """Test that varargs raise an issue."""
    issues = _analyze_str(_VARARGS_CODE)
    _assert_has_issue(issues, ast_analyzer.VarArgsForbidden)

  def test_kwargs_issue(self):
    """Test that kwargs raise an issue."""
    issues = _analyze_str(_KWARGS_CODE)
    _assert_has_issue(issues, ast_analyzer.KwArgsForbidden)

  def test_type_issue(self):
    """Test that invalid types raise an issue."""
    issues = _analyze_str(_TYPE_ISSUE_CODE)
    _assert_has_issue(issues, ast_analyzer.MissingType)

  def test_type_mismatch_issue(self):
    """Test that type mismatches raise an issue."""
    issues = _analyze_str(_TYPE_MISMATCH_CODE)
    _assert_has_issue(issues, ast_analyzer.TypeMismatch)

  def test_model_field_suffix_issue(self):
    """Test that model fields with suffixes raise an issue."""
    issues = _analyze_str(_MODEL_SUFFIX_CODE)
    _assert_has_issue(issues, ast_analyzer.InvalidSuffix)

  def test_data_field_suffix_issue(self):
    """Test that data fields with invalid suffixes raise an issue."""
    issues = _analyze_str(_DATA_SUFFIX_CODE)
    _assert_has_issue(issues, ast_analyzer.InvalidSuffix)

  def test_missing_comment_issue(self):
    """Test that missing comments raise an issue."""
    issues = _analyze_str(_MISSING_COMMENT_CODE)
    _assert_has_issue(issues, ast_analyzer.MissingComment)

  def test_write_to_readonly_field_issue(self):
    """Test that writing to readonly fields raises an issue."""
    issues = _analyze_str(_WRITE_READONLY_CODE)
    _assert_has_issue(issues, ast_analyzer.InvalidWrite)
    # There should be 2 instances - one for Model and one for Data _in
    write_issues = [i for i in issues if isinstance(i, ast_analyzer.InvalidWrite)]
    self.assertEqual(len(write_issues), 2)

  def test_all_issues(self):
    """Test a function with all issue types."""
    issues = _analyze_str(_ALL_ISSUES_CODE)
    expected_types = [
      ast_analyzer.DefaultForbidden,
      ast_analyzer.VarArgsForbidden,
      ast_analyzer.KwArgsForbidden,
      ast_analyzer.MissingType,
      ast_analyzer.TypeMismatch,
      ast_analyzer.InvalidSuffix,
      ast_analyzer.MissingComment,
      ast_analyzer.InvalidWrite,
    ]
    for issue_type in expected_types:
      _assert_has_issue(issues, issue_type)

  def test_no_issues(self):
    """Test a function with no issues."""
    issues = _analyze_str(_NO_ISSUES_CODE)
    for iss in issues:
      print(f"{iss.node.lineno}:{iss}")

    self.assertEqual(len(issues), 0)

  def test_non_kernel_function(self):
    """Test that non-kernel functions aren't analyzed."""
    issues = _analyze_str(_NON_KERNEL_CODE)
    self.assertEqual(len(issues), 0)  # Not a kernel, so no issues


if __name__ == "__main__":
  absltest.main()
