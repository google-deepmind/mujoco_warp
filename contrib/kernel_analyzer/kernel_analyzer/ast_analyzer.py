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
from typing import Dict, List, Optional, Tuple


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
      return f'"{self.node.arg}" invalid suffix, _in/_out not allowed on Model parameters.'
    elif self.source == "Data":
      return f'"{self.node.arg}" invalid suffix, _in/_out required on Data parameters.'
    else:
      return f'"{self.node.arg}" invalid suffix, _in/_out required.'


@dataclasses.dataclass
class InvalidParamOrder(Issue):
  expected: List[str]
  expected_full: List[str]  # with comments and types
  # function arg range ((start_lineno, start_col), (end_lineno, end_col))
  arg_range: Tuple[Tuple[int, int], Tuple[int, int]]

  def __str__(self):
    return f'"{self.kernel}" invalid parameter order, expected: {", ".join(self.expected)}'


@dataclasses.dataclass
class MissingComment(Issue):
  expected_comment: str

  def __str__(self):
    return f'"{self.node.arg}" should be preceded by comment: {self.expected_comment}'


@dataclasses.dataclass
class InvalidWrite(Issue):
  def __str__(self):
    return f'"{self.node.id}" invalid write: parameter is read-only'


@dataclasses.dataclass
class MissingModuleUnique(Issue):
  def __str__(self):
    return f'"{self.kernel}" nested kernel missing module="unique"'


# TODO(team): add argument order analyzer.
# this one is tricky because just verifying order does not tell you if the arguments
# match the parameter signature.


def _get_classes(src: str) -> Dict[str, List[Tuple[str, str]]]:
  """Return classes and fields and annotations."""
  ret = {}
  class_name = None
  in_docstring = False
  field_pattern = re.compile(r"^\s+(\w+)\s*:\s*(.*?)(?:\s*#.*)?$")

  for line in src.splitlines():
    if line.startswith("class "):
      class_name = line[len("class ") : -1]
    if line.strip().startswith('"""'):
      in_docstring = True
    if line.strip().endswith('"""'):
      in_docstring = False

    m = field_pattern.match(line)
    if not class_name or not m or in_docstring:
      continue

    ret.setdefault(class_name, []).append((m.group(1), m.group(2)))

  return ret


def _get_function_arg_range(node: ast.FunctionDef, source: str) -> tuple[tuple[int, int], tuple[int, int]]:
  """Gets the function arg range."""
  start_line = node.lineno - 1
  start_char = node.col_offset + len(node.name)
  lines = source.splitlines()
  open_paren = lines[start_line].find("(", start_char)
  start_pos = (start_line, open_paren + 1)

  balance = 1
  curr_line, curr_char = start_line, open_paren + 1
  while balance > 0 and curr_line < len(lines):
    search_line = lines[curr_line]
    while curr_char < len(search_line):
      if search_line[curr_char] == "(":
        balance += 1
      elif search_line[curr_char] == ")":
        balance -= 1
        if balance == 0:
          return start_pos, (curr_line, curr_char)
      curr_char += 1
    curr_line += 1
    curr_char = 0
  return start_pos, (node.end_lineno - 1, node.end_col_offset)


def _get_arg_expected_comment(param_source: str, param_out: bool) -> str:
  if param_source == "Model":
    expected_comment = "# Model:"
  elif param_source == "Data":
    expected_comment = f"# Data {'out' if param_out else 'in'}:"
  else:
    expected_comment = f"# {'Out' if param_out else 'In'}:"
  return expected_comment


def analyze(source: str, filename: str, type_source: str) -> List[Issue]:
  """Parses Python code and finds functions with unsorted simple parameters."""
  logging.info(f"Analyzing {filename}...")

  # get class fields and types for Model and Data
  type_classes = _get_classes(type_source)
  field_info = {}

  for field, typ in type_classes["Model"]:
    if field == "opt":
      for sfield, styp in type_classes["Option"]:
        field_info[field + "_" + sfield] = ("Model", styp, len(field_info))
    elif field == "stat":
      for sfield, styp in type_classes["Statistic"]:
        field_info[field + "_" + sfield] = ("Model", styp, len(field_info))
    else:
      field_info[field] = ("Model", typ, len(field_info))

  for field, typ in type_classes["Data"]:
    if field == "efc":
      for sfield, styp in type_classes["Constraint"]:
        field_info[field + "_" + sfield] = ("Data", styp, len(field_info))
    elif field == "contact":
      for sfield, styp in type_classes["Contact"]:
        field_info[field + "_" + sfield] = ("Data", styp, len(field_info))
    else:
      field_info[field] = ("Data", typ, len(field_info))

  try:
    tree = ast.parse(source, filename=filename)
  except SyntaxError as e:
    logging.error(f"Syntax error in {filename}:{e.lineno}: {e.msg}")
    return []

  issues: List[Issue] = []
  source_lines = source.splitlines()

  def _is_kernel_decorator(decorator_name: str) -> bool:
    """Check if a decorator marks a kernel or func."""
    return decorator_name and any(
        decorator_name == name or decorator_name.startswith(name + "(")
        for name in ("wp.kernel", "wp.func")
    )

  def _has_module_unique(decorator_name: str) -> bool:
    """Check if decorator has module='unique' or module=\"unique\"."""
    return 'module="unique"' in decorator_name or "module='unique'" in decorator_name

  def _analyze_function(node: ast.FunctionDef, is_nested: bool):
    """Analyze a function definition for kernel issues."""
    decorator_name = None
    is_kernel = False
    for d in node.decorator_list:
      decorator_name = ast.get_source_segment(source, d)
      if _is_kernel_decorator(decorator_name):
        is_kernel = True
        break

    # Recursively check nested functions first
    for child in ast.iter_child_nodes(node):
      if isinstance(child, ast.FunctionDef):
        _analyze_function(child, is_nested=True)

    if not is_kernel:
      return

    # Check for nested kernel missing module="unique"
    if is_nested and decorator_name and not _has_module_unique(decorator_name):
      issues.append(MissingModuleUnique(node, node.name))

    _analyze_kernel(node)

  def _analyze_kernel(node: ast.FunctionDef):
    """Analyze kernel parameters and body."""
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
    param_types = {}
    param_outs = set()
    param_reftypes = set()
    expected_types = {}

    # TODO(team): we might also want to check return types of wp.func
    for param in args.args:
      param_name = param.arg
      param_names.add(param_name)

      # params must be type annotated
      if param.annotation is None:
        issues.append(MissingType(param, kernel))
        continue

      param_type = ast.get_source_segment(source, param.annotation)
      param_type = param_type.replace("types.", "")  # ignore types module prefix
      param_types[param_name] = param_type
      has_suffix = param_name.endswith("_in") or param_name.endswith("_out")
      field_name = param_name[: param_name.rfind("_")] if has_suffix else param_name
      param_out = param_name.endswith("_out") or param_name in ("res", "out")
      if param_out:
        param_outs.add(param_name)
      if "array" in param_type:
        param_reftypes.add(param_name)
      param_source, expected_type, param_order = field_info.get(field_name, (None, None, None))

      # TODO(team): indicate array slice with suffix and check type?

      # if parameters are multi-line, parameters must be grouped by comments of the form
      # "{source} {in | out | ""}:" e.g. "Model:" or "Data in:" or "Out:"
      if params_multiline and (param_out, param_source) not in param_groups:
        param_groups.add((param_out, param_source))
        expected_comment = _get_arg_expected_comment(param_source, param_out)
        if param.lineno < 2 or source_lines[param.lineno - 2].strip() != expected_comment:
          issues.append(MissingComment(param, kernel, expected_comment))

      # paramater type must match field type (or generic types if no corresponding field)
      if expected_type is None:
        # if the parameter does not correspond to Model/Data fields, it has no expected type
        # still, there are a few type conventions we stick to
        if "wp.int32" in param_type:
          expected_type = "int"
        elif "wp.float32" in param_type:
          expected_type = "float"
        elif "wp.bool" in param_type:
          expected_type = "bool"
      elif expected_type.startswith("array("):
        # array(...) is our custom annotation that we can translate to wp.array
        expected_dtype = expected_type[expected_type.rfind(" ") + 1 : -1]
        expected_ndim = expected_type.count(",")
        if expected_ndim == 1:
          expected_type = f"wp.array(dtype={expected_dtype})"
        else:
          expected_type = f"wp.array{expected_ndim}d(dtype={expected_dtype})"

      expected_types[param_name] = expected_type
      if expected_type and param_type != expected_type:
        issues.append(TypeMismatch(param, kernel, expected_type, param_source))

      # Model params must not have in/out suffix, Data params must, other params optional
      if param_source == "Model" and has_suffix:
        issues.append(InvalidSuffix(param, kernel, param_source))
      elif param_source == "Data" and not has_suffix:
        issues.append(InvalidSuffix(param, kernel, param_source))

      source_order = {"Model": 0, "Data": 1, None: 2}[param_source]
      param_order = len(params_ordering) if param_order is None else param_order
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
      arg_range = _get_function_arg_range(node, source)
      # Build the full arg strings with comments and types.
      expected_full, prev = [], None
      for e in expected_ordering:
        if prev != e[:2]:
          src = {0: "Model", 1: "Data", 2: None}[e[1]]
          expected_full.append(_get_arg_expected_comment(src, e[0]))
        type_ = expected_types[e[-1]] or param_types[e[-1]]
        expected_full.append(e[-1] + ": " + type_ + ",")
        prev = e[:2]
      issues.append(InvalidParamOrder(node, kernel, expected_names, expected_full, arg_range))

    # don't allow assignments to in parameters
    for sub_node in ast.walk(node):
      # simple assignments:
      if isinstance(sub_node, ast.Assign):
        for target in sub_node.targets:
          assignee = ast.get_source_segment(source, target)
          if assignee in param_names and assignee not in param_outs and assignee in param_reftypes:
            issues.append(InvalidWrite(target, kernel))
      # augmented assignments (+=, -=, etc)
      elif isinstance(sub_node, ast.AugAssign):
        assignee = ast.get_source_segment(source, sub_node.target)
        if assignee in param_names and assignee not in param_outs and assignee in param_reftypes:
          issues.append(InvalidWrite(sub_node.target, kernel))
      # in-place operations like a[i] = value
      elif isinstance(sub_node, ast.Subscript) and isinstance(sub_node.ctx, ast.Store):
        assignee = ast.get_source_segment(source, sub_node.value)
        if assignee in param_names and assignee not in param_outs and assignee in param_reftypes:
          issues.append(InvalidWrite(sub_node.value, kernel))
      # TODO: atomic_add, atomic_sub

  # Analyze all top-level function definitions
  for node in ast.iter_child_nodes(tree):
    if isinstance(node, ast.FunctionDef):
      _analyze_function(node, is_nested=False)

  # skip issues in ignored lines
  ignore_lines = set()
  ignoring = False
  for lineno, line in enumerate(source_lines, 1):  # lineno is 1-indexed
    if "# kernel_analyzer: off" in line:
      ignoring = True
    elif "# kernel_analyzer: on" in line:
      ignoring = False
    if "# kernel_analyzer: ignore" in line or ignoring:
      ignore_lines.add(lineno)

  filtered_issues = [i for i in issues if i.node.lineno not in ignore_lines]
  filter_count, ignore_count = len(filtered_issues), len(issues) - len(filtered_issues)

  logging.info(f"Finished analyzing {filename}. Found {filter_count} issues, ignoring {ignore_count} issues.")

  return filtered_issues
