# Copyright 2026 The MuJoCo Authors.
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

"""CPU tests for the offline native SDF build step."""

import hashlib
import json
import shutil

import mujoco
import numpy as np
import pytest
from prepare_sdf import correct_distances
from prepare_sdf import prepare


@pytest.fixture
def scene(tmp_path):
  assets = tmp_path / "source assets"
  assets.mkdir()
  (assets / "cube.obj").write_text(
    "v -0.1 -0.2 -0.3\nv 0.1 -0.2 -0.3\nv 0.1 0.2 -0.3\nv -0.1 0.2 -0.3\n"
    "v -0.1 -0.2 0.3\nv 0.1 -0.2 0.3\nv 0.1 0.2 0.3\nv -0.1 0.2 0.3\n"
    "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\n"
    "f 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n"
  )
  path = tmp_path / "scene.xml"
  path.write_text(
    '<mujoco><compiler meshdir="source assets"/><asset><mesh name="cube" file="cube.obj"/></asset>'
    '<worldbody><geom type="sdf" mesh="cube"/><body pos="2 0 0"><joint name="j" type="slide"/>'
    '<geom type="sdf" mesh="cube"/></body></worldbody>'
    '<actuator><position joint="j" kp="600" kv="1.0000001192092896"/></actuator></mujoco>'
  )
  return path


def test_correct_distances_preserves_geometry_and_physics(scene):
  spec = mujoco.MjSpec.from_file(str(scene))
  spec.mesh("cube").octree_maxdepth = 3
  model = spec.compile()
  original = {
    name: hashlib.sha256(np.ascontiguousarray(getattr(model, name))).hexdigest()
    for name in dir(model)
    if isinstance(getattr(model, name), np.ndarray) and name != "oct_coeff"
  }
  model.oct_coeff[:] = 42
  evidence = correct_distances(model)
  assert set(evidence) == {"cube"}  # Both geoms share the same octree.
  for name, fingerprint in original.items():
    assert hashlib.sha256(np.ascontiguousarray(getattr(model, name))).hexdigest() == fingerprint, name
  boxes = model.oct_aabb.reshape(-1, 2, 3)
  signs = np.array([[1 if j & (1 << k) else -1 for k in range(3)] for j in range(8)])
  points = boxes[:, None, 0] + boxes[:, None, 1] * signs
  lower, upper = model.mesh_vert.min(axis=0), model.mesh_vert.max(axis=0)
  delta = np.abs(points - (lower + upper) / 2) - (upper - lower) / 2
  expected = np.linalg.norm(np.maximum(delta, 0), axis=-1) + np.minimum(delta.max(axis=-1), 0)
  np.testing.assert_allclose(model.oct_coeff, expected, atol=1e-7, rtol=1e-6)
  assert (expected < -1e-3).any() and (expected > 1e-3).any()


def test_winding_sign_preserves_an_empty_cavity(scene):
  spec = mujoco.MjSpec.from_file(str(scene))
  spec.mesh("cube").octree_maxdepth = 3
  solid = spec.compile()
  vertices, faces = solid.mesh_vert.copy(), solid.mesh_face.copy()
  hollow = mujoco.MjSpec()
  mesh = hollow.add_mesh(name="hollow")
  mesh.uservert = np.concatenate((vertices, vertices * 0.5)).ravel()
  mesh.userface = np.concatenate((faces, faces[:, ::-1] + len(vertices))).ravel()
  mesh.octree_maxdepth = 4
  hollow.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_SDF, meshname="hollow")
  model = hollow.compile()
  correct_distances(model)
  boxes = model.oct_aabb.reshape(-1, 2, 3)
  signs = np.array([[1 if j & (1 << k) else -1 for k in range(3)] for j in range(8)])
  points = boxes[:, None, 0] + boxes[:, None, 1] * signs
  size = np.max(np.abs(model.mesh_vert), axis=0)
  outer_delta, inner_delta = np.abs(points) - size, np.abs(points) - size * 0.5
  outer_sdf = np.linalg.norm(np.maximum(outer_delta, 0), axis=-1) + np.minimum(outer_delta.max(axis=-1), 0)
  inner_sdf = np.linalg.norm(np.maximum(inner_delta, 0), axis=-1) + np.minimum(inner_delta.max(axis=-1), 0)
  expected = np.maximum(outer_sdf, -inner_sdf)
  np.testing.assert_allclose(model.oct_coeff, expected, atol=1e-7, rtol=1e-6)
  cavity = inner_sdf < -1e-3
  assert cavity.any() and (model.oct_coeff[cavity] > 0).all()


def test_prepare_is_relocatable_and_records_inputs(scene, tmp_path):
  relocated = tmp_path / "relocated"
  relocated.mkdir()
  shutil.copytree(scene.parent / "source assets", relocated / "source assets")
  shutil.copy(scene, relocated / scene.name)
  output = tmp_path / "generated.mjb"
  report = prepare(relocated / scene.name, output, depth=3)
  model = mujoco.MjModel.from_binary_path(str(output))
  assert model.oct_depth.max() == 3
  assert model.actuator_gainprm[0, 0] == 600
  assert model.actuator_biasprm[0, 2] == -float(np.float32(1.0000001192092896))
  assert report["source_sha256"] == hashlib.sha256(scene.read_bytes()).hexdigest()
  mesh_sha256 = hashlib.sha256((relocated / "source assets/cube.obj").read_bytes()).hexdigest()
  assert report["mesh_sources"]["cube"]["sha256"] == mesh_sha256
  assert report["model_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
  assert json.loads(output.with_suffix(".json").read_text()) == report
  with pytest.raises(ValueError, match="new .mjb"):
    prepare(scene, output, depth=3)
  output.unlink()
  with pytest.raises(ValueError, match="new .mjb"):
    prepare(scene, output, depth=3)  # Keep a model's provenance immutable even if its MJB is moved.


@pytest.mark.parametrize("depth", [0, 11])
def test_rejects_unbounded_depth(scene, tmp_path, depth):
  with pytest.raises(ValueError, match="depth"):
    prepare(scene, tmp_path / "generated.mjb", depth)


def test_rejects_scene_without_native_sdf(tmp_path):
  source = tmp_path / "scene.xml"
  source.write_text('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
  with pytest.raises(ValueError, match="no native SDF"):
    prepare(source, tmp_path / "generated.mjb")
