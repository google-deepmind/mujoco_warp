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
"""Differentiability API for MuJoCo Warp."""

import dataclasses
import functools
import inspect
import weakref
from typing import Any, Callable, Collection, Sequence

import warp as wp
from warp._src import context as wp_context
from warp._src import types as wp_types

from mujoco_warp._src import types
from mujoco_warp._src import warp_util


class _Untaped:
  """Context manager that swaps and restores the active Warp tape."""

  def __init__(self, tape: wp.Tape | None = None):
    self.tape = tape

  def __enter__(self):
    self._tape, wp_context.runtime.tape = wp_context.runtime.tape, self.tape
    return self.tape

  def __exit__(self, *exc):
    wp_context.runtime.tape = self._tape


def _call_untaped(fn: Callable, *args, **kwargs):
  with _Untaped():
    return fn(*args, **kwargs)


def _is_float(arr: wp.array) -> bool:
  return wp.types.type_is_float(wp_types.type_scalar_type(arr.dtype))


def _inputs(launches: Sequence[Any]) -> set[int]:
  """Returns `id(arr)` for arrays read by a kernel launch before being written."""
  read, written = set(), set()
  for l in launches:
    if isinstance(l, list) and all(l[1]):
      read.update(id(a) for a in l[3] if isinstance(a, wp.array) and id(a) not in written)
      written.update(id(a) for a in l[4] if isinstance(a, wp.array))
  return read


def _fields(cls: type, prefix: str = "") -> tuple[str, ...]:
  out: list[str] = []
  for f in dataclasses.fields(cls):
    path = f"{prefix}.{f.name}" if prefix else f.name
    if dataclasses.is_dataclass(f.type):
      out.extend(_fields(f.type, path))
    elif warp_util.is_array_spec(f.type):
      out.append(path)
  return tuple(out)


_FIELDS: dict[type, tuple[str, ...]] = {cls: _fields(cls) for cls in (types.Model, types.Data, types.RenderContext)}


def get(obj: Any, path: str) -> Any:
  """Resolves a dotted attribute path on a dataclass container."""
  for part in path.split("."):
    obj = vars(obj)[part]
  return obj


@dataclasses.dataclass(frozen=True)
class AdjointRule:
  """Custom analytical backward rule for an API function."""

  backward: Callable[..., None] | None = None
  in_fields: tuple[str, ...] = ()
  out_fields: tuple[str, ...] = ()
  model_fields: tuple[str, ...] = ()
  render_fields: tuple[str, ...] = ()
  array_args: tuple[str, ...] = ()
  supported: Callable[..., Sequence[str] | None] | None = None

  @property
  def fields(self) -> dict[type, tuple[str, ...]]:
    return {
      types.Model: self.model_fields,
      types.Data: (*self.in_fields, *self.out_fields),
      types.RenderContext: self.render_fields,
    }


RULES: dict[str, AdjointRule | None] = {}
_PARAMS: dict[str, tuple[str, ...]] = {}
_PROMOTED: dict[str, weakref.WeakSet] = {}


def _unsupported(arr: Any, supported: bool, exempt: bool = False) -> bool:
  """Returns whether `arr` requires grad without support; unused promoted arrays are exempt."""
  return (
    isinstance(arr, wp.array)
    and arr.requires_grad
    and not (exempt and any(arr in s for k, s in _PROMOTED.items() if RULES[k] is not None))
    and (not _is_float(arr) or not supported)
  )


def _validate(obj: Any, rule: AdjointRule, used: Collection[int]) -> None:
  cls = type(obj)
  allowed = rule.fields[cls]
  for path in _FIELDS[cls]:
    arr = get(obj, path)
    if _unsupported(arr, path in allowed, id(arr) not in used):
      raise NotImplementedError(f"Differentiation of {cls.__name__} field '{path}' is not supported.")


def _device(bound: dict[str, Any]) -> wp.Device:
  """Returns the device of the first array in the bound arguments."""
  for obj in bound.values():
    for arr in (get(obj, p) for p in _FIELDS[type(obj)]) if type(obj) in _FIELDS else (obj,):
      if isinstance(arr, wp.array):
        return arr.device
  return wp.get_device()


def register_adjoint(
  fn: Callable | str,
  backward: Callable[..., None] | None = None,
  *,
  in_fields: Sequence[str] = (),
  out_fields: Sequence[str] | None = None,
  model_fields: Sequence[str] = (),
  render_fields: Sequence[str] = (),
  array_args: Sequence[str] = (),
  supported: Callable[..., Sequence[str] | None] | None = None,
) -> Any:
  """Registers a custom backward rule for `fn` (callable directly or as a decorator)."""
  name = fn if isinstance(fn, str) else fn.__name__
  if name not in RULES:
    raise ValueError(f"Function {name!r} is not registered as differentiable.")
  resolved_out = in_fields if out_fields is None else out_fields
  for seq, valid, label in (
    (in_fields, _FIELDS[types.Data], "Data fields"),
    (resolved_out, _FIELDS[types.Data], "Data fields"),
    (model_fields, _FIELDS[types.Model], "Model fields"),
    (render_fields, _FIELDS[types.RenderContext], "RenderContext fields"),
    (array_args, _PARAMS[name], "array_args"),
  ):
    if isinstance(seq, str) or not set(seq).issubset(valid):
      raise ValueError(f"Invalid {label} for {name!r}: {seq!r}")

  def _register(bwd: Callable[..., None] | None = None) -> Any:
    for arr in _PROMOTED[name]:
      arr.requires_grad = False
    _PROMOTED[name].clear()
    RULES[name] = AdjointRule(
      bwd,
      tuple(in_fields),
      tuple(resolved_out),
      tuple(model_fields),
      tuple(render_fields),
      tuple(array_args),
      supported,
    )
    return bwd

  _register(backward)
  return _register if backward is None else backward


def _wrap(fn: Callable) -> Callable:
  """Wraps an API function for Warp tape validation and custom adjoint dispatch."""
  name = fn.__name__
  params = tuple(inspect.signature(fn).parameters)
  RULES[name] = None
  _PARAMS[name] = params
  _PROMOTED[name] = weakref.WeakSet()

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    tape = wp_context.runtime.tape if wp_context.runtime is not None else None
    if tape is None:
      return fn(*args, **kwargs)

    rule = RULES[name]
    if rule is None:
      raise NotImplementedError(f"Differentiation of '{name}' is not supported.")

    bound = {**dict(zip(params, args)), **kwargs}
    reasons = () if rule.supported is None or _device(bound).is_capturing else rule.supported(*args, **kwargs)
    if reasons:
      raise NotImplementedError(f"Differentiation of '{name}' is not supported: {', '.join(reasons)}.")

    with _Untaped(wp.Tape()) as fwd_tape:
      result = fn(*args, **kwargs)

    used = _inputs(fwd_tape.launches)
    for k, obj in bound.items():
      if type(obj) in _FIELDS:
        _validate(obj, rule, used)
      elif _unsupported(obj, k in rule.array_args):
        raise NotImplementedError(f"Differentiation of '{name}' with array argument is not supported.")

    # Containers by type in signature order; the first Data is the input, the last is the output:
    objs: dict[type, list[Any]] = {cls: [] for cls in _FIELDS}
    for k in params:
      if type(bound.get(k)) in objs:
        objs[type(bound[k])].append(bound[k])
    m, data, rc = objs[types.Model][:1], objs[types.Data], objs[types.RenderContext][:1]
    containers = ((data[:1], rule.in_fields), (data[-1:], rule.out_fields), (rc, rule.render_fields))
    auto = [*(get(c, p) for cs, fs in containers for c in cs for p in fs), *(bound.get(k) for k in rule.array_args)]
    for arr in auto:
      if isinstance(arr, wp.array) and not arr.requires_grad and arr.size > 0 and _is_float(arr):
        arr.requires_grad = True
        _PROMOTED[name].add(arr)

    if rule.backward is None:
      tape.launches.extend(fwd_tape.launches)
      return result

    tracked = (*(get(c, p) for c in m for p in rule.model_fields), *auto)
    arrays = list({id(a): a for a in tracked if isinstance(a, wp.array) and a.size > 0 and a.requires_grad}.values())
    tape.record_func(backward=lambda: _call_untaped(rule.backward, *args, **kwargs), arrays=arrays)
    return result

  return wrapper


def differentiable(namespace: dict[str, Any]) -> None:
  """Wraps the public functions in `namespace` (`mujoco_warp.__init__`)."""
  for name, obj in list(namespace.items()):
    if inspect.isfunction(obj) and not name.startswith("_"):
      namespace[name] = _wrap(obj)
