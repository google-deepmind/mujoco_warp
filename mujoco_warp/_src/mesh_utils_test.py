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

"""Tests for non-convex mesh splitting utilities."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np
from absl.testing import absltest

from mujoco_warp._src import mesh_utils


def _write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
  mesh_utils._write_binary_stl(path, vertices, faces)


def _triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
  vertices = np.array(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
  )
  faces = np.array([[0, 1, 2]], dtype=np.int32)
  return vertices, faces


class MeshUtilsTest(absltest.TestCase):
  def test_split_nonconvex_meshes_rewrites_assets_and_geoms(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp = Path(tmpdir)
      meshes_dir = tmp / "meshes"
      meshes_dir.mkdir(parents=True, exist_ok=True)

      stl_path = meshes_dir / "tool.stl"
      vertices, faces = _triangle_mesh()
      _write_binary_stl(stl_path, vertices, faces)

      xml = """
<mujoco>
  <asset>
    <mesh name="tool" file="meshes/tool.stl"/>
  </asset>
  <worldbody>
    <body>
      <geom name="g" type="mesh" mesh="tool"/>
    </body>
  </worldbody>
</mujoco>
"""

      calls = {"count": 0}

      class FakeMesh:
        def __init__(self, v, f):
          self.v = v
          self.f = f

      def _fake_run_coacd(mesh, **kwargs):
        del mesh, kwargs
        calls["count"] += 1
        verts0 = np.array(
          [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
          dtype=np.float64,
        )
        faces0 = np.array([[0, 1, 2]], dtype=np.int32)
        verts1 = np.array(
          [[0.0, 0.0, 0.1], [0.5, 0.0, 0.1], [0.0, 0.5, 0.1]],
          dtype=np.float64,
        )
        faces1 = np.array([[0, 1, 2]], dtype=np.int32)
        return [(verts0, faces0), (verts1, faces1)]

      fake_coacd = SimpleNamespace(__version__="test", Mesh=FakeMesh, run_coacd=_fake_run_coacd)

      with mock.patch.dict(sys.modules, {"coacd": fake_coacd}):
        out1 = mesh_utils.split_nonconvex_meshes(
          xml,
          "tool",
          assets_dir=tmp,
          cache_dir=tmp / ".cache",
          input_is_path=False,
          validate_output=False,
        )
        out2 = mesh_utils.split_nonconvex_meshes(
          xml,
          "tool",
          assets_dir=tmp,
          cache_dir=tmp / ".cache",
          input_is_path=False,
          validate_output=False,
        )

      self.assertEqual(calls["count"], 1)
      self.assertEqual(out1, out2)

      root = ET.fromstring(out1)
      mesh_names = [m.get("name") for m in root.findall("./asset/mesh")]
      self.assertEqual(mesh_names, ["tool__cvx_000", "tool__cvx_001"])

      geom_names = [g.get("mesh") for g in root.findall(".//geom")]
      self.assertEqual(geom_names, ["tool__cvx_000", "tool__cvx_001"])

      manifests = list((tmp / ".cache" / "tool").glob("*/manifest.json"))
      self.assertLen(manifests, 1)
      manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
      self.assertEqual(manifest["mesh"], "tool")
      self.assertLen(manifest["parts"], 2)

  def test_split_nonconvex_meshes_missing_mesh_raises(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp = Path(tmpdir)
      xml = """
<mujoco>
  <asset>
    <mesh name="tool" file="meshes/tool.stl"/>
  </asset>
</mujoco>
"""

      with self.assertRaises(mesh_utils.MeshNotFoundError):
        mesh_utils.split_nonconvex_meshes(
          xml,
          "missing",
          assets_dir=tmp,
          input_is_path=False,
          validate_output=False,
        )

  def test_split_nonconvex_meshes_skip_already_convex(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp = Path(tmpdir)
      meshes_dir = tmp / "meshes"
      meshes_dir.mkdir(parents=True, exist_ok=True)
      stl_path = meshes_dir / "tool.stl"
      vertices, faces = _triangle_mesh()
      _write_binary_stl(stl_path, vertices, faces)

      xml = """
<mujoco>
  <asset>
    <mesh name="tool" file="meshes/tool.stl"/>
  </asset>
  <worldbody>
    <body>
      <geom name="g" type="mesh" mesh="tool"/>
    </body>
  </worldbody>
</mujoco>
"""

      class FakeMesh:
        def __init__(self, v, f):
          self.v = v
          self.f = f

      def _fake_run_coacd(mesh, **kwargs):
        del kwargs
        return [(mesh.v, mesh.f)]

      fake_coacd = SimpleNamespace(__version__="test", Mesh=FakeMesh, run_coacd=_fake_run_coacd)

      with mock.patch.dict(sys.modules, {"coacd": fake_coacd}):
        out = mesh_utils.split_nonconvex_meshes(
          xml,
          "tool",
          assets_dir=tmp,
          cache_dir=tmp / ".cache",
          input_is_path=False,
          validate_output=False,
          skip_if_already_convex=True,
        )

      root = ET.fromstring(out)
      mesh_names = [m.get("name") for m in root.findall("./asset/mesh")]
      self.assertEqual(mesh_names, ["tool"])
      geom_meshes = [g.get("mesh") for g in root.findall(".//geom")]
      self.assertEqual(geom_meshes, ["tool"])


if __name__ == "__main__":
  absltest.main()
