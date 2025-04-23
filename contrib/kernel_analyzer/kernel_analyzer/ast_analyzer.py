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
"""AST analyzer for kernel functions."""

import ast
import dataclasses
import logging
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass
class Issue:
  node: ast.AST
  kernel: str


@dataclasses.dataclass
class DefaultForbidden(Issue):
  def __str__(self):
    return f'"{self.node.arg}" default not allowed.'


@dataclasses.dataclass
class VarArgsForbidden(Issue):
  def __str__(self):
    return f'"{self.node.arg}" varargs not allowed.'


@dataclasses.dataclass
class KwArgsForbidden(Issue):
  def __str__(self):
    return f'"{self.node.arg}" kwargs not allowed.'


@dataclasses.dataclass
class MissingType(Issue):
  def __str__(self):
    return f'"{self.node.arg}" missing type annotation.'


@dataclasses.dataclass
class TypeMismatch(Issue):
  expected_type: str
  source: Optional[str]

  def __str__(self):
    ret = f'"{self.node.arg}" type mismatch, expected {self.expected_type}'
    ret += "." if self.source is None else f" from {self.source}."
    return ret


@dataclasses.dataclass
class InvalidSuffix(Issue):
  source: Optional[str]

  def __str__(self):
    if self.source == "Model":
      return (
        f'"{self.node.arg}" invalid suffix, _in/_out not allowed on Model parameters.'
      )
    elif self.source == "Data":
      return f'"{self.node.arg}" invalid suffix, _in/_out required on Data parameters.'
    else:
      return f'"{self.node.arg}" invalid suffix, _in/_out required.'


@dataclasses.dataclass
class InvalidParamOrder(Issue):
  expected: List[str]

  def __str__(self):
    return (
      f'"{self.kernel}" invalid parameter order, expected: {", ".join(self.expected)}'
    )


@dataclasses.dataclass
class MissingComment(Issue):
  expected_comment: str

  def __str__(self):
    return f'"{self.node.arg}" should be preceded by comment: {self.expected_comment}'


@dataclasses.dataclass
class InvalidWrite(Issue):
  def __str__(self):
    return f'"{self.node.id}" invalid write: parameter is read-only'


def _get_classes(src: str) -> Dict[str, List[Tuple[str, str]]]:
  """Return classes and fields and annotations."""
  ret = {}
  class_name = None
  in_docstring = False
  field_pattern = re.compile(r"^\s+(\w+)\s*:\s*(.*?)(?:\s*#.*)?$")

  for line in src.splitlines():
    if line.startswith("class "):
      class_name = line[len("class ") : -1]
    elif '"""' in line:
      in_docstring = not in_docstring

    m = field_pattern.match(line)
    if not class_name or not m or in_docstring:
      continue

    ret.setdefault(class_name, []).append((m.group(1), m.group(2)))

  return ret


def analyze(source: str, filename: str, type_source: str) -> List[Issue]:
  """Parses Python code and finds functions with unsorted simple parameters."""
  logging.info(f"Analyzing {filename}...")

  # get class fields and types for Model and Data
  type_classes = _get_classes(type_source)
  field_source = {k: "Model" for k, _ in type_classes["Model"]}
  field_source.update({k: "Data" for k, _ in type_classes["Data"]})
  field_type = {k: v for k, v in type_classes["Model"] + type_classes["Data"]}
  field_order = {k: i for i, (k, _) in enumerate(type_classes["Model"])}
  field_order.update({k: i for i, (k, _) in enumerate(type_classes["Data"])})

  try:
    tree = ast.parse(source, filename=filename)
  except SyntaxError as e:
    logging.error(f"Syntax error in {filename}:{e.lineno}: {e.msg}")
    return []

  issues: List[Issue] = []
  source_lines = source.splitlines()

  for node in ast.walk(tree):
    if not isinstance(node, ast.FunctionDef):
      continue

    is_kernel = False
    for d in node.decorator_list:
      decorator_name = ast.get_source_segment(source, d)
      if decorator_name in ("kernel", "warp_util.kernel", "wp.kernel"):
        is_kernel = True
        break

    if not is_kernel:
      continue

    kernel, args = node.name, node.args

    # defaults not allowed
    if args.defaults:
      for param in args.args[-len(args.defaults) :]:
        issues.append(DefaultForbidden(param, kernel))

    # kw defaults not allowed.
    if args.kwonlyargs:
      for param, default in zip(args.kwonlyargs, args.kw_defaults):
        if default:
          issues.append(DefaultForbidden(param, kernel))

    # varargs not allowed.
    if args.vararg:
      issues.append(VarArgsForbidden(args.vararg, kernel))

    # kwargs not allowed.
    if args.kwarg:
      issues.append(KwArgsForbidden(args.kwarg, kernel))

    param_names = set()
    params_ordering = []
    params_multiline = any(a.lineno != args.args[0].lineno for a in args.args)
    param_groups = set()

    for param in args.args:
      param_name = param.arg
      param_names.add(param_name)

      # params must be type annotated
      if param.annotation is None:
        issues.append(MissingType(param, kernel))
        continue

      param_type = ast.get_source_segment(source, param.annotation)
      has_suffix = param_name.endswith("_in") or param_name.endswith("_out")
      field_name = param_name[: param_name.rfind("_")] if has_suffix else param_name
      param_out = param_name.endswith("_out")
      param_source = field_source.get(field_name)
      param_order = field_order.get(field_name, len(params_ordering))
      expected_type = field_type.get(field_name)

      # if parameters are multi-line, parameters must be grouped by comments of the form
      # "{source} {in | out | ""}:" e.g. "Model:" or "Data in:" or "Out:"
      if params_multiline and (param_out, param_source) not in param_groups:
        param_groups.add((param_out, param_source))
        if param_source == "Model":
          expected_comment = "# Model:"
        elif param_source == "Data":
          expected_comment = f"# Data {'out' if param_out else 'in'}:"
        else:
          expected_comment = f"# {'Out' if param_out else 'In'}:"
        if (
          param.lineno < 2 or source_lines[param.lineno - 2].strip() != expected_comment
        ):
          issues.append(MissingComment(param, kernel, expected_comment))

      # paramater type must match field type (or generic types if no corresponding field)
      if expected_type is None:
        expected_type = ("int", "float", "bool")
        if param_type not in expected_type and not param_type.startswith("wp.array"):
          issues.append(TypeMismatch(param, kernel, "int, float, bool, wp.array", None))
      elif param_type != expected_type:
        issues.append(TypeMismatch(param, kernel, expected_type, param_source))

      # Model params must not have in/out suffix, Data/other params must
      if (param_source == "Model") == has_suffix:
        issues.append(InvalidSuffix(param, kernel, param_source))

      source_order = {"Model": 0, "Data": 1, None: 2}[param_source]
      params_ordering.append((param_out, source_order, param_order, param_name))

    # parameters must follow a specified order:
    # 1) in fields, then out fields
    # 2) Model, then Data, then other
    # 3) matching Model field order or Data field order
    expected_ordering = sorted(params_ordering)
    if params_ordering != expected_ordering:
      expected_names = [p[-1] for p in expected_ordering]
      node.col_offset = node.col_offset + 4
      node.end_lineno = node.lineno
      node.end_col_offset = node.col_offset + len(node.name)
      issues.append(InvalidParamOrder(node, kernel, expected_names))

    # don't allow assignments to in parameters
    for sub_node in ast.walk(node):
      # simple assignments:
      if isinstance(sub_node, ast.Assign):
        for target in sub_node.targets:
          assignee = ast.get_source_segment(source, target)
          if assignee in param_names and not assignee.endswith("_out"):
            issues.append(InvalidWrite(target, kernel))
      # augmented assignments (+=, -=, etc)
      elif isinstance(sub_node, ast.AugAssign):
        assignee = ast.get_source_segment(source, sub_node.target)
        if assignee in param_names and not assignee.endswith("_out"):
          issues.append(InvalidWrite(sub_node.target, kernel))
      # in-place operations like a[i] = value
      elif isinstance(sub_node, ast.Subscript) and isinstance(sub_node.ctx, ast.Store):
        assignee = ast.get_source_segment(source, sub_node.value)
        if assignee in param_names and not assignee.endswith("_out"):
          issues.append(InvalidWrite(sub_node.value, kernel))
      # TODO: atomic_add, atomic_sub

  logging.info(f"Finished analyzing {filename}. Found {len(issues)} issues.")

  return issues
