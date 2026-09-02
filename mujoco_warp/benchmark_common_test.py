# Copyright 2026 The Newton Developers
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

"""Tests for the helpers shared by the benchmark scripts."""

import ast
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

from absl.testing import absltest

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
_COMMON = _BENCHMARKS_DIR / "common.py"
_SWEEP = _BENCHMARKS_DIR / "sweep.py"


def _load_common():
  """Import benchmarks/common.py by path, without putting benchmarks/ on sys.path."""
  spec = importlib.util.spec_from_file_location("_benchmarks_common", _COMMON)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _imported_roots(path: Path) -> set:
  """Return the top-level module name of every import in path."""
  roots = set()
  for node in ast.walk(ast.parse(path.read_text())):
    if isinstance(node, ast.Import):
      roots.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
      # a relative import cannot resolve from the installed copy at all, so record it as such
      roots.add(node.module.split(".")[0] if node.level == 0 and node.module else ".")
  return roots


class EnsurePinnedCloneTest(absltest.TestCase):
  """Tests for common.ensure_pinned_clone, which populates the benchmark asset cache."""

  def setUp(self):
    """Build a throwaway git repo to fetch from, and a cache directory to fetch into."""
    super().setUp()
    if not _COMMON.exists():
      self.skipTest("benchmarks/ is not present, mujoco_warp was likely installed without the repo")
    self.common = _load_common()

    base = Path(tempfile.mkdtemp())
    self.addCleanup(lambda: subprocess.run(("rm", "-rf", base.as_posix()), check=False))
    self.source = base / "source"
    self.source.mkdir()
    for args in (
      ("git", "init", "--quiet", "-b", "main"),
      ("git", "config", "user.email", "test@example.com"),
      ("git", "config", "user.name", "test"),
    ):
      subprocess.run(args, cwd=self.source, check=True, capture_output=True)
    (self.source / "asset.xml").write_text("<mujoco/>")
    subprocess.run(("git", "add", "-A"), cwd=self.source, check=True, capture_output=True)
    subprocess.run(("git", "commit", "--quiet", "-m", "init"), cwd=self.source, check=True, capture_output=True)
    self.ref = subprocess.run(("git", "rev-parse", "HEAD"), cwd=self.source, capture_output=True, text=True).stdout.strip()

    self.cache = base / "cache"
    self.dst = self.cache / "repo"

  def _fetch(self, ref=None):
    self.common.ensure_pinned_clone(self.source.as_posix(), ref or self.ref, self.dst)

  def _assert_is_checkout(self):
    self.assertEqual((self.dst / "asset.xml").read_text(), "<mujoco/>")
    self.assertTrue((self.dst / ".git").exists())

  def test_fetches_pinned_ref(self):
    """A missing dst is populated at the requested ref."""
    self._fetch()
    self._assert_is_checkout()

  def test_reuses_existing_checkout(self):
    """A dst that already has .git is left completely alone."""
    self._fetch()
    (self.dst / "marker").write_text("untouched")
    self._fetch()
    self.assertEqual((self.dst / "marker").read_text(), "untouched")

  def test_rebuilds_cache_entry_without_git_dir(self):
    """A directory left behind by an interrupted fetch is rebuilt rather than trusted."""
    self.dst.mkdir(parents=True)
    (self.dst / "partial").write_text("junk")
    self._fetch()
    self._assert_is_checkout()
    self.assertFalse((self.dst / "partial").exists())

  def test_replaces_regular_file_at_dst(self):
    """A regular file at dst is removed; shutil.rmtree alone would leave it and break the rename."""
    self.cache.mkdir(parents=True)
    self.dst.write_text("not a directory")
    self._fetch()
    self._assert_is_checkout()

  def test_replaces_broken_symlink_at_dst(self):
    """A dangling symlink at dst is removed rather than left to break the rename."""
    self.cache.mkdir(parents=True)
    self.dst.symlink_to(self.cache / "nowhere")
    self._fetch()
    self._assert_is_checkout()

  def test_replaces_symlink_without_touching_its_target(self):
    """A symlink at dst is unlinked, and the directory it pointed at is left intact."""
    self.cache.mkdir(parents=True)
    target = self.cache / "elsewhere"
    target.mkdir()
    (target / "keep").write_text("precious")
    self.dst.symlink_to(target)
    self._fetch()
    self._assert_is_checkout()
    self.assertFalse(self.dst.is_symlink())
    self.assertEqual((target / "keep").read_text(), "precious")

  def test_failed_fetch_leaves_no_cache_entry(self):
    """A fetch that fails leaves neither a dst nor a staging directory for later runs to trip on."""
    with self.assertRaises(subprocess.CalledProcessError):
      self._fetch(ref="0" * 40)
    self.assertFalse(self.dst.exists())
    self.assertEqual(sorted(p.name for p in self.cache.iterdir()), [])


class SweepDeploymentTest(absltest.TestCase):
  """Tests the constraint that lets the systemd nightly run sweep.py outside a repo checkout."""

  def setUp(self):
    """Skip when the benchmark scripts are not on disk."""
    super().setUp()
    if not _COMMON.exists() or not _SWEEP.exists():
      self.skipTest("benchmarks/ is not present, mujoco_warp was likely installed without the repo")

  def test_installed_files_only_import_the_standard_library(self):
    """sweep.py and common.py import nothing that has to be installed alongside them.

    contrib/systemd/README.md installs just these two files into ~/.local/share/mjwarp-benchmarks
    and runs sweep.py from there, detached from any checkout. Anything else they import has to be
    added to that install step, or the nightly fails at startup rather than at install time.
    """
    for path in (_SWEEP, _COMMON):
      for root in _imported_roots(path) - {"common"}:
        self.assertIn(root, sys.stdlib_module_names, f"{path.name} imports non-stdlib module {root!r}")

  def test_common_is_the_only_local_import(self):
    """sweep.py's sole non-stdlib import is its installed sibling common.py."""
    siblings = {path.stem for path in _BENCHMARKS_DIR.glob("*.py")} - {"common"}
    self.assertIn("run", siblings, "expected benchmarks/run.py to exist as a sibling module")
    self.assertEqual(_imported_roots(_SWEEP) & (siblings | {"."}), set())


if __name__ == "__main__":
  absltest.main()
