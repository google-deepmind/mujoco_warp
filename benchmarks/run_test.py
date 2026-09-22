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

"""CPU regression checks for benchmark preparation and the portable NIST package."""

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import common
import numpy as np
import pytest
import run
from franka_emika_panda import BENCHMARKS


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
  input_dir = tmp_path / "input checkout"
  module_dir = input_dir / "benchmarks" / "fixture"
  module_dir.mkdir(parents=True)
  (module_dir / "scene.xml").write_text("local scene overlay")
  assets_root = tmp_path / "assembled assets"
  repo = {"source": "https://example.invalid/fake_assets.git", "ref": "pinned_revision"}
  cached = assets_root / "_git" / "fake_assets" / repo["ref"] / "robot"
  cached.mkdir(parents=True)
  (cached / "mesh.obj").write_text("fetched mesh")
  (cached / "scene.xml").write_text("fetched scene replaced by local overlay")
  monkeypatch.setattr(run, "_ARGS", SimpleNamespace(assets_root=assets_root, clear_warp_cache=False))

  def ensure_pinned_clone(source, ref, destination):
    assert (source, ref, destination) == (repo["source"], repo["ref"], cached.parent)

  monkeypatch.setattr(run, "ensure_pinned_clone", ensure_pinned_clone)
  return {
    "name": "fixture",
    "mjcf": "scene.mjb",
    "replay": "action tape.npz",
    "assets": [(repo, "robot")],
    "_dir": module_dir,
    "nworld": 16,
    "prepare": ["python", "{input_dir}/contrib/prepare_sdf.py", "{benchmark_dir}/scene.xml"],
  }


@pytest.mark.parametrize("prepare", [False, True])
def test_assembly_prepares_after_assets_and_local_overlay(benchmark, monkeypatch, prepare):
  if not prepare:
    benchmark.pop("prepare")
  assembled = Path(run._ARGS.assets_root) / benchmark["name"]
  assembled.mkdir()
  (assembled / "stale.mjb").write_text("old build")
  commands = []

  def uv_run(*args, cwd):
    assert (assembled / "mesh.obj").read_text() == "fetched mesh"
    assert (assembled / "scene.xml").read_text() == "local scene overlay"
    assert not (assembled / "stale.mjb").exists()
    commands.append((args, cwd))

  monkeypatch.setattr(run, "uv_run", uv_run)
  run._assemble_benchmark(benchmark)
  assert (assembled / "scene.xml").read_text() == "local scene overlay"
  input_dir = benchmark["_dir"].parents[1]
  expected = ("python", str(input_dir / "contrib/prepare_sdf.py"), str(assembled / "scene.xml"))
  expected_commands = [(expected, input_dir)] if prepare else []
  assert commands == expected_commands


def test_preparation_failure_stops_assembly(benchmark, monkeypatch):
  def fail(*args, **kwargs):
    raise subprocess.CalledProcessError(1, args, stderr="native SDF preparation failed")

  monkeypatch.setattr(run, "uv_run", fail)
  with pytest.raises(subprocess.CalledProcessError, match="non-zero exit status 1"):
    run._assemble_benchmark(benchmark)


@pytest.mark.parametrize("prepare", [False, True])
def test_testspeed_receives_only_runtime_flags(benchmark, monkeypatch, prepare):
  if not prepare:
    benchmark.pop("prepare")
  commands = []

  def uv_run(*args, cwd):
    commands.append((args, cwd))
    return subprocess.CompletedProcess(args, 0, stdout="step 0.01\nsolver 0.003\n")

  monkeypatch.setattr(run, "uv_run", uv_run)
  input_dir = benchmark["_dir"].parents[1]
  assert run._run_benchmark(benchmark, input_dir) == {"step": "0.01", "solver": "0.003"}
  assembled = Path(run._ARGS.assets_root) / benchmark["name"]
  assert commands == [
    (
      (
        "mjwarp-testspeed",
        str(assembled / "scene.mjb"),
        "--clear_warp_cache=False",
        "--format=short",
        "--event_trace=true",
        "--memory=true",
        "--measure_solver=true",
        "--measure_alloc=true",
        f"--replay={assembled / 'action tape.npz'}",
        "--nworld=16",
      ),
      input_dir,
    )
  ]


def test_assembly_package_has_initial_only_control_replay():
  package = Path(__file__).parent / "franka_emika_panda"
  assert {path.relative_to(package).as_posix() for path in package.rglob("*.py")} == {"__init__.py"}
  assert not list(package.rglob("*.mjb")), "Compiled models must remain outside the source package"
  for pattern in ("*.pt", "*.pth", "*.ckpt", "*.usd", "*.usda"):
    assert not list(package.rglob(pattern)), "No policy or source-environment runtime dependencies"
  variants = [item for item in BENCHMARKS if item["name"] == "panda_nist_assembly"]
  assert len(variants) == 1
  for item in variants:
    assert "state_profile" not in item
    assert all(item[key] > 0 for key in ("nworld", "nstep", "nconmax", "njmax", "nccdmax"))
    assert item["noise_std"] == item["noise_rate"] == 0
    assert item["override"] == ["opt.jacobian=sparse"]
    assert item["replay"] == "nist_k4_replay.npz"
  assert not list(package.glob("*_states.npz")), "Runtime replay retains only the initial state"
  with np.load(package / variants[0]["replay"], allow_pickle=False) as replay:
    assert set(replay.files) == {"qpos", "qvel", "ctrl", "times"}
    assert replay["qpos"].shape == (1, 37) and replay["qvel"].shape == (1, 33)
    assert replay["ctrl"].ndim == 2 and replay["ctrl"].shape[1] == 8 and len(replay["ctrl"]) > 0
    assert replay["times"].shape == (len(replay["ctrl"]) + 1,), "Use explicit control interval boundaries"
    assert replay["times"][0] == 0 and np.all(np.diff(replay["times"]) > 0)
    assert variants[0]["nstep"] == round(replay["times"][-1] / 0.00125), "Time the complete control tape"
    for key in ("qpos", "qvel", "ctrl"):
      assert replay[key].dtype in (np.float32, np.float64)
    for key in replay.files:
      assert np.isfinite(replay[key]).all()
  provenance = json.loads((package / "nist_k4_provenance.json").read_text())
  assert provenance["source_row"] == 1549
  assert hashlib.sha256((package / variants[0]["replay"]).read_bytes()).hexdigest() == provenance["replay_sha256"]
  scene = ET.parse(package / "scene_nist_k4.xml").getroot()
  assert len(scene.findall(".//geom[@type='plane']")) == 1
  assert len(scene.findall(".//freejoint")) + len(scene.findall(".//joint[@type='free']")) == 4
  assert not scene.findall("include") and not scene.findall(".//plugin")
  assert len(scene.findall(".//geom[@type='sdf']")) == 8
  for mesh in scene.findall("./asset/mesh"):
    path = Path(mesh.get("file"))
    assert not path.is_absolute() and ".." not in path.parts
  for asset in provenance["assembly_meshes"]:
    assert hashlib.sha256((package / asset["file"]).read_bytes()).hexdigest() == asset["sha256"]

  board = scene.find("./worldbody/body[@name='nist_board']")
  assert board is not None and board.find("geom[@type='box']") is not None
  assert not board.findall("joint") and not board.findall("freejoint")


def test_runner_reuses_common_helpers_without_legacy_aliases():
  assert run.uv_run is common.uv_run
  assert run.ensure_pinned_clone is common.ensure_pinned_clone
  assert not hasattr(run, "_uv_run") and not hasattr(run, "_git")


def test_package_replaces_previous_sparse_contact_case():
  package = Path(__file__).parent / "franka_emika_panda"
  assert not (package / "scene_sparse_contact.xml").exists()
  assert not any("sparse_contact" in item["name"] or "threading" in item["name"] for item in BENCHMARKS)
  assert not list(package.glob("*k3*")) and not (package / "scene_nist_threading.xml").exists()


def test_continuous_replay_has_no_saved_state_runtime():
  root = Path(__file__).parents[1]
  for name in ("benchmarks/run.py", "mujoco_warp/_src/cli.py", "mujoco_warp/testspeed.py"):
    source = (root / name).read_text()
    assert "state_profile" not in source and "_restore_state" not in source
  cli_source = (root / "mujoco_warp/_src/cli.py").read_text()
  assert "reset_data_keyframe" not in cli_source, "Replay initializes once and preserves solver history"
  assert not (root / "mujoco_warp/testspeed_test.py").exists(), "No obsolete saved-state diagnostic tests"
