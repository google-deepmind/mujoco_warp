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
"""Command-line parsing for dataclass argument definitions."""

import argparse
import dataclasses
import types
import typing
from dataclasses import MISSING


def _argument_type(annotation):
  args = typing.get_args(annotation)
  if typing.get_origin(annotation) in (typing.Union, types.UnionType):
    annotation = next(arg for arg in args if arg is not type(None))
  return annotation


def _default(spec):
  if spec.default is not MISSING:
    return spec.default
  if spec.default_factory is not MISSING:
    return spec.default_factory()
  return MISSING


def _metadata(args_cls, spec):
  if spec.metadata:
    return spec.metadata
  for base in args_cls.__mro__[1:]:
    if not dataclasses.is_dataclass(base):
      continue
    inherited = next((field for field in dataclasses.fields(base) if field.name == spec.name), None)
    if inherited is not None and inherited.metadata:
      return inherited.metadata
  return {}


def argument_parser(args_cls) -> argparse.ArgumentParser:
  """Creates an argument parser from a dataclass."""
  parser = argparse.ArgumentParser(
    description=args_cls.__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  hints = typing.get_type_hints(args_cls)
  for spec in dataclasses.fields(args_cls):
    default = _default(spec)
    metadata = _metadata(args_cls, spec)
    kwargs = {"help": metadata.get("help", spec.name.replace("_", " "))}
    if default is MISSING:
      kwargs["required"] = True
    else:
      kwargs["default"] = default
    choices = metadata.get("choices")
    if choices:
      kwargs["choices"] = choices
    annotation = _argument_type(hints[spec.name])
    if annotation is bool:
      kwargs["action"] = argparse.BooleanOptionalAction
    elif typing.get_origin(annotation) is tuple:
      item_types = typing.get_args(annotation)
      if not item_types or item_types[-1] is Ellipsis or len(set(item_types)) != 1:
        raise TypeError(f"unsupported command-line tuple annotation: {hints[spec.name]}")
      kwargs["type"] = item_types[0]
      kwargs["nargs"] = len(item_types)
    else:
      kwargs["type"] = annotation
    parser.add_argument(f"--{spec.name.replace('_', '-')}", **kwargs)
  return parser


def parse_args_dataclass(args_cls, argv=None):
  """Parses command-line arguments into a dataclass instance."""
  values = vars(argument_parser(args_cls).parse_args(argv))
  hints = typing.get_type_hints(args_cls)
  for name, value in values.items():
    if typing.get_origin(_argument_type(hints[name])) is tuple and isinstance(value, list):
      values[name] = tuple(value)
  return args_cls(**values)
