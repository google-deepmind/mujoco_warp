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

"""Utilities for splitting non-convex MuJoCo meshes into convex parts."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import Any
from typing import Mapping
from typing import Sequence
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


_HELPER_VERSION = "1"


class MeshSplitError(RuntimeError):
  """Base class for non-convex mesh splitting errors."""


class MeshNotFoundError(MeshSplitError):
  """Raised when a requested mesh name is missing in the asset section."""


class CoacdNotInstalledError(MeshSplitError):
  """Raised when coacd is not installed."""


class InvalidMeshAssetError(MeshSplitError):
  """Raised when a target mesh asset is malformed."""


def split_nonconvex_meshes(
  xml: str,
  mesh_names: Sequence[str] | str,
  coacd_kwargs: Mapping[str, Any] | None = None,
  *,
  assets_dir: str | os.PathLike[str] | None = None,
  cache_dir: str | os.PathLike[str] | None = None,
  input_is_path: bool | None = None,
  skip_if_already_convex: bool = True,
  validate_output: bool = True,
) -> str:
  """Splits non-convex mesh assets and rewrites geoms to convex mesh parts.

  The function accepts an XML string or an XML file path. For each requested
  mesh in ``mesh_names``, it loads the corresponding ``.stl`` file, runs
  ``coacd.run_coacd``, caches the output parts on disk, removes the original
  mesh entry, inserts per-part mesh entries, and duplicates geoms that
  referenced the original mesh.

  Args:
    xml: MuJoCo XML string or XML path.
    mesh_names: One mesh name or a sequence of mesh names.
    coacd_kwargs: Keyword arguments forwarded to ``coacd.run_coacd``.
    assets_dir: Directory used to resolve relative mesh file paths.
    cache_dir: Directory used to cache convex decomposition results.
    input_is_path: If set, forces whether ``xml`` is interpreted as a path.
    skip_if_already_convex: If True, keep original mesh if decomposition
      returns one convex part.
    validate_output: If True, validates output by compiling the transformed XML.

  Returns:
    Transformed MuJoCo XML as a string.

  Raises:
    MeshNotFoundError: If any requested mesh name is missing.
    InvalidMeshAssetError: If target mesh lacks a file or file is not ``.stl``.
    FileNotFoundError: If a target ``.stl`` file does not exist.
    CoacdNotInstalledError: If ``coacd`` is unavailable.
    MeshSplitError: For decomposition failures or invalid transformed output.
  """
  target_meshes = _normalize_mesh_names(mesh_names)
  kwargs = dict(coacd_kwargs or {})

  xml_text, xml_path = _load_xml_input(xml, input_is_path=input_is_path)
  model_dir = xml_path.parent if xml_path else Path.cwd()
  base_assets_dir = Path(assets_dir).resolve() if assets_dir else model_dir
  base_cache_dir = Path(cache_dir).resolve() if cache_dir else base_assets_dir / ".mujoco_mesh_cache"
  base_cache_dir.mkdir(parents=True, exist_ok=True)

  root = ET.fromstring(xml_text)
  asset = root.find("asset")
  if asset is None:
    raise MeshSplitError("XML has no <asset> section")

  mesh_elements = _mesh_elements_by_name(asset)
  replacement_map: dict[str, list[tuple[str, str]]] = {}

  for mesh_name in target_meshes:
    if mesh_name not in mesh_elements:
      raise MeshNotFoundError(f"mesh '{mesh_name}' not found in <asset>")

    mesh_elem = mesh_elements[mesh_name]
    mesh_file = mesh_elem.get("file")
    if not mesh_file:
      raise InvalidMeshAssetError(f"mesh '{mesh_name}' is missing file attribute")

    mesh_file_path = Path(mesh_file)
    stl_path = mesh_file_path if mesh_file_path.is_absolute() else (base_assets_dir / mesh_file_path)
    stl_path = stl_path.resolve()

    if stl_path.suffix.lower() != ".stl":
      raise InvalidMeshAssetError(f"mesh '{mesh_name}' must point to an .stl file: {mesh_file}")
    if not stl_path.exists():
      raise FileNotFoundError(f"stl file for mesh '{mesh_name}' not found: {stl_path}")

    part_paths = _decompose_or_cache(mesh_name=mesh_name, stl_path=stl_path, cache_dir=base_cache_dir, coacd_kwargs=kwargs)

    if len(part_paths) == 1 and skip_if_already_convex:
      continue

    mesh_entries: list[tuple[str, str]] = []
    for idx, part_path in enumerate(part_paths):
      part_name = f"{mesh_name}__cvx_{idx:03d}"
      if mesh_file_path.is_absolute():
        part_file = str(part_path).replace("\\", "/")
      else:
        part_file = os.path.relpath(part_path, base_assets_dir).replace("\\", "/")
      mesh_entries.append((part_name, part_file))

    replacement_map[mesh_name] = mesh_entries

  for old_name, new_parts in replacement_map.items():
    old_mesh_elem = mesh_elements[old_name]
    old_attrs = dict(old_mesh_elem.attrib)
    insert_idx = list(asset).index(old_mesh_elem)
    asset.remove(old_mesh_elem)

    for offset, (new_name, new_file) in enumerate(new_parts):
      attrs = dict(old_attrs)
      attrs["name"] = new_name
      attrs["file"] = new_file
      asset.insert(insert_idx + offset, ET.Element("mesh", attrs))

    _replace_mesh_geoms(root, old_name, [name for name, _ in new_parts])

  tree = ET.ElementTree(root)
  ET.indent(tree, space="  ", level=0)
  result_xml = ET.tostring(root, encoding="unicode")

  if validate_output:
    _validate_transformed_xml(result_xml, base_assets_dir)

  return result_xml


def _normalize_mesh_names(mesh_names: Sequence[str] | str) -> list[str]:
  if isinstance(mesh_names, str):
    names = [mesh_names]
  else:
    names = list(mesh_names)
  if not names:
    raise ValueError("mesh_names must not be empty")
  if len(set(names)) != len(names):
    raise ValueError("mesh_names contains duplicates")
  return names


def _load_xml_input(xml: str, input_is_path: bool | None) -> tuple[str, Path | None]:
  candidate = Path(xml)
  treat_as_path = input_is_path if input_is_path is not None else (candidate.suffix.lower() == ".xml" and candidate.exists())

  if treat_as_path:
    xml_path = candidate.resolve()
    if not xml_path.exists():
      raise FileNotFoundError(f"xml file not found: {xml_path}")
    return xml_path.read_text(encoding="utf-8"), xml_path

  return xml, None


def _mesh_elements_by_name(asset_elem: ET.Element) -> dict[str, ET.Element]:
  mesh_elements: dict[str, ET.Element] = {}
  for mesh_elem in asset_elem.findall("mesh"):
    name = mesh_elem.get("name")
    if name:
      mesh_elements[name] = mesh_elem
  return mesh_elements


def _canonicalize_kwargs(coacd_kwargs: Mapping[str, Any]) -> str:
  return json.dumps(coacd_kwargs, sort_keys=True, separators=(",", ":"), default=str)


def _decomposition_key(stl_path: Path, coacd_kwargs: Mapping[str, Any], coacd_version: str) -> str:
  digest = hashlib.sha256()
  digest.update(_HELPER_VERSION.encode("utf-8"))
  digest.update(coacd_version.encode("utf-8"))
  digest.update(stl_path.read_bytes())
  digest.update(_canonicalize_kwargs(coacd_kwargs).encode("utf-8"))
  return digest.hexdigest()[:16]


def _decompose_or_cache(
  *, mesh_name: str, stl_path: Path, cache_dir: Path, coacd_kwargs: Mapping[str, Any]
) -> list[Path]:
  try:
    coacd = importlib.import_module("coacd")
  except Exception as exc:
    raise CoacdNotInstalledError("coacd is required. Install with `pip install coacd`.") from exc

  coacd_version = getattr(coacd, "__version__", "unknown")
  key = _decomposition_key(stl_path, coacd_kwargs, coacd_version)
  out_dir = cache_dir / mesh_name / key
  out_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = out_dir / "manifest.json"

  if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cached_parts = [out_dir / name for name in manifest.get("parts", [])]
    if cached_parts and all(path.exists() for path in cached_parts):
      return cached_parts

  vertices, faces = _read_stl_mesh(stl_path)
  if faces.shape[0] == 0:
    raise MeshSplitError(f"stl has no triangles: {stl_path}")

  mesh = coacd.Mesh(vertices, faces)
  parts = coacd.run_coacd(mesh, **coacd_kwargs)

  if not parts:
    raise MeshSplitError(f"coacd returned zero parts for: {stl_path}")

  part_paths: list[Path] = []
  for idx, part in enumerate(parts):
    part_vertices = np.asarray(part[0], dtype=np.float64)
    part_faces = np.asarray(part[1], dtype=np.int32)
    part_path = out_dir / f"{mesh_name}__cvx_{idx:03d}.stl"
    _write_binary_stl(part_path, part_vertices, part_faces)
    part_paths.append(part_path)

  manifest = {
    "mesh": mesh_name,
    "source_stl": str(stl_path),
    "key": key,
    "helper_version": _HELPER_VERSION,
    "coacd_version": coacd_version,
    "coacd_kwargs": json.loads(_canonicalize_kwargs(coacd_kwargs)),
    "parts": [path.name for path in part_paths],
  }
  manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

  return part_paths


def _replace_mesh_geoms(root: ET.Element, old_mesh_name: str, new_mesh_names: Sequence[str]) -> int:
  replacements = 0
  if not new_mesh_names:
    return replacements

  for parent in root.iter():
    children = list(parent)
    for child_idx, child in enumerate(children):
      if child.tag != "geom" or child.get("mesh") != old_mesh_name:
        continue

      original_name = child.get("name")
      clones: list[ET.Element] = []
      for idx, new_mesh_name in enumerate(new_mesh_names):
        geom = copy.deepcopy(child)
        geom.set("mesh", new_mesh_name)
        if original_name and idx > 0:
          geom.set("name", f"{original_name}__cvx_{idx:03d}")
        clones.append(geom)

      parent.remove(child)
      for offset, clone in enumerate(clones):
        parent.insert(child_idx + offset, clone)
      replacements += 1

  return replacements


def _validate_transformed_xml(xml_text: str, model_dir: Path) -> None:
  model_dir.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=model_dir, encoding="utf-8", delete=False) as tmp:
    tmp_path = Path(tmp.name)
    tmp.write(xml_text)
  try:
    getattr(mujoco, "MjModel").from_xml_path(str(tmp_path))
  except Exception as exc:
    raise MeshSplitError(f"transformed xml failed to compile: {exc}") from exc
  finally:
    tmp_path.unlink(missing_ok=True)


def _read_stl_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
  data = path.read_bytes()
  if len(data) < 84:
    raise MeshSplitError(f"invalid stl file: {path}")

  tri_count = int.from_bytes(data[80:84], "little")
  expected = 84 + tri_count * 50
  if expected == len(data):
    return _read_binary_stl(data)

  return _read_ascii_stl(data.decode("utf-8", errors="ignore"))


def _read_binary_stl(data: bytes) -> tuple[np.ndarray, np.ndarray]:
  tri_count = int.from_bytes(data[80:84], "little")
  vertices: list[tuple[float, float, float]] = []
  faces: list[tuple[int, int, int]] = []
  vertex_index: dict[tuple[float, float, float], int] = {}

  offset = 84
  for _ in range(tri_count):
    offset += 12
    face_idx = []
    for _ in range(3):
      x, y, z = struct.unpack_from("<fff", data, offset)
      offset += 12
      key = (float(x), float(y), float(z))
      idx = vertex_index.get(key)
      if idx is None:
        idx = len(vertices)
        vertex_index[key] = idx
        vertices.append(key)
      face_idx.append(idx)
    faces.append((face_idx[0], face_idx[1], face_idx[2]))
    offset += 2

  return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _read_ascii_stl(text: str) -> tuple[np.ndarray, np.ndarray]:
  raw_vertices: list[tuple[float, float, float]] = []
  for line in text.splitlines():
    parts = line.strip().split()
    if len(parts) == 4 and parts[0].lower() == "vertex":
      raw_vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

  if len(raw_vertices) % 3 != 0:
    raise MeshSplitError("ascii stl has invalid vertex count")

  vertices: list[tuple[float, float, float]] = []
  faces: list[tuple[int, int, int]] = []
  vertex_index: dict[tuple[float, float, float], int] = {}

  for i in range(0, len(raw_vertices), 3):
    tri = raw_vertices[i : i + 3]
    idxs = []
    for vertex in tri:
      idx = vertex_index.get(vertex)
      if idx is None:
        idx = len(vertices)
        vertex_index[vertex] = idx
        vertices.append(vertex)
      idxs.append(idx)
    faces.append((idxs[0], idxs[1], idxs[2]))

  return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)

  header = b"mujoco_warp convex decomposition"
  header = header[:80].ljust(80, b"\0")

  with path.open("wb") as f:
    f.write(header)
    f.write(struct.pack("<I", int(faces.shape[0])))
    for face in faces:
      i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
      p0 = vertices[i0]
      p1 = vertices[i1]
      p2 = vertices[i2]
      f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
      f.write(struct.pack("<fff", float(p0[0]), float(p0[1]), float(p0[2])))
      f.write(struct.pack("<fff", float(p1[0]), float(p1[1]), float(p1[2])))
      f.write(struct.pack("<fff", float(p2[0]), float(p2[1]), float(p2[2])))
      f.write(struct.pack("<H", 0))
