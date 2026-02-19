# Copyright 2025 DeepMind Technologies Limited
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

"""Tests for package version utilities."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from mujoco_warp._src import util_pkg


class ParseVersionTest(parameterized.TestCase):
  """Tests for version string parsing."""

  @parameterized.parameters(
    ("1.0.0", ((0, 1), (0, 0), (0, 0))),
    ("3.5.0", ((0, 3), (0, 5), (0, 0))),
    ("1.20.0", ((0, 1), (0, 20), (0, 0))),
    ("3.5.0.dev869102767", ((0, 3), (0, 5), (0, 0), (1, "dev869102767"))),
    ("3.5.0-google3", ((0, 3), (0, 5), (0, 0), (1, "google3"))),
    ("1.0.0-alpha", ((0, 1), (0, 0), (0, 0), (1, "alpha"))),
    ("2.0.0-beta.1", ((0, 2), (0, 0), (0, 0), (1, "beta"), (0, 1))),
  )
  def test_parse_version(self, version_str, expected):
    self.assertEqual(util_pkg._parse_version(version_str), expected)


class CheckVersionTest(parameterized.TestCase):
  """Tests for check_version function."""

  @parameterized.parameters(
    # >= operator
    ("pkg>=1.0.0", "1.0.0", True),
    ("pkg>=1.0.0", "1.0.1", True),
    ("pkg>=1.0.0", "2.0.0", True),
    ("pkg>=1.0.0", "0.9.0", False),
    # <= operator
    ("pkg<=1.0.0", "1.0.0", True),
    ("pkg<=1.0.0", "0.9.0", True),
    ("pkg<=1.0.0", "1.0.1", False),
    # > operator
    ("pkg>1.0.0", "1.0.1", True),
    ("pkg>1.0.0", "1.0.0", False),
    # < operator
    ("pkg<1.0.0", "0.9.0", True),
    ("pkg<1.0.0", "1.0.0", False),
    # == operator
    ("pkg==1.0.0", "1.0.0", True),
    ("pkg==1.0.0", "1.0.1", False),
    # != operator
    ("pkg!=1.0.0", "1.0.1", True),
    ("pkg!=1.0.0", "1.0.0", False),
    # With dev/pre-release versions
    ("pkg>=3.5.0", "3.5.0.dev869102767", True),  # dev version > base
    ("pkg>=3.5.0", "3.5.0-google3", True),  # google3 > base
    ("pkg>3.5.0", "3.5.0.dev869102767", True),  # dev version > base
    # Lexicographic ordering: google3 > dev
    ("pkg>=3.5.0-google3", "3.5.0-google3", True),
    ("pkg>3.5.0.dev869102767", "3.5.0-google3", True),
  )
  def test_check_version(self, spec, installed_version, expected):
    with mock.patch("importlib.metadata.version", return_value=installed_version):
      self.assertEqual(util_pkg.check_version(spec), expected)

  @parameterized.parameters(
    "numpy",  # no operator
    ">=1.0.0",  # no package name
    "numpy~=1.0.0",  # unsupported operator
    "",  # empty string
  )
  def test_check_version_invalid_spec(self, spec):
    with self.assertRaises(ValueError):
      util_pkg.check_version(spec)

  def test_check_version_package_not_found(self):
    """Test that PackageNotFoundError is raised for missing packages."""
    import importlib.metadata

    with mock.patch(
      "importlib.metadata.version",
      side_effect=importlib.metadata.PackageNotFoundError("nonexistent"),
    ):
      with self.assertRaises(importlib.metadata.PackageNotFoundError):
        util_pkg.check_version("nonexistent>=1.0.0")


if __name__ == "__main__":
  absltest.main()
