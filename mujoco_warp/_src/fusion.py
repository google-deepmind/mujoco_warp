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
"""Stages: kernels written as functions, fused into fewer kernels at launch.

A stage is a kernel body written as a `wp.func` whose first parameters are its thread indices:

  @fusion.stage("nworld", "ngeom")
  def geom_xpos(worldid: int, geomid: int, geom_pos: ..., xpos_in: ..., geom_xpos_out: ...):
    ...

The other parameters follow the kernel analyzer naming: Model fields by name (`opt_*` and `stat_*`
for Option and Statistic), and Data fields with an `_in` or `_out` suffix (`efc_*` and `contact_*`
for Constraint and Contact). `launch(m, d, *stages)` binds them by name and runs the stages in
order, fusing them into as few kernels as their data dependencies allow:

- stages that do not touch an array another stage in the kernel writes share it, each over its own
  range of threads
- a stage with the same dims as the stage before it runs in the same thread after it, if every
  array one of them writes and the other touches is only accessed at the thread's own index
"""

import ast
import contextlib
import dataclasses
import hashlib
import inspect
import linecache
import math
import textwrap
from typing import Any

import warp as wp
from warp._src import context as wp_context

# launch one kernel per stage when False, the reference that fused launches must match
FUSE = True


@dataclasses.dataclass(frozen=True, eq=False)
class Stage:
  """A kernel body with its launch dims and the parameters it binds by name."""

  func: wp.Function
  name: str
  dim: tuple[str, ...]
  params: dict[str, Any]  # parameter name -> annotation, after the thread indices
  fields: dict[str, tuple[bool, str, str]]  # parameter name -> (in Data, sub-struct or "", field)
  pointwise: frozenset[str]  # array parameters only accessed at the thread indices


def stage(*dim: str):
  """Declares a kernel body as a stage launched over `dim`, sizes named as in Model or nworld."""

  def decorator(f):
    sig = inspect.signature(f)
    names = list(sig.parameters)
    index, params = names[: len(dim)], names[len(dim) :]
    pointwise = _pointwise(f, index, params)
    return Stage(
      func=wp.func(f),
      name=f.__name__,
      dim=dim,
      params={p: sig.parameters[p].annotation for p in params},
      fields={p: _path(p) for p in params},
      pointwise=pointwise,
    )

  return decorator


def _pointwise(f, index: list[str], params: list[str]) -> frozenset[str]:
  """Returns the parameters only subscripted with the thread indices, in order and unmodified.

  Raises if a parameter is written that is not named `*_out`, since the plan trusts the names.
  """
  tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
  for n in ast.walk(tree):
    written = None
    if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name):
      written = n.value.id
    elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith("atomic_") and n.args:
      written = n.args[0].id if isinstance(n.args[0], ast.Name) else None
    if written in params and not written.endswith("_out"):
      raise ValueError(f"{f.__name__} writes {written!r}, which a stage must name with an _out suffix")
  if any(isinstance(n, ast.Name) and n.id in index and isinstance(n.ctx, ast.Store) for n in ast.walk(tree)):
    return frozenset()
  own = [ast.dump(ast.Name(i, ast.Load())) for i in index]
  subscripts = {id(n.value): n.slice for n in ast.walk(tree) if isinstance(n, ast.Subscript)}
  pointwise = set(params)
  for n in ast.walk(tree):
    if isinstance(n, ast.Name) and n.id in params:
      idx = subscripts.get(id(n))
      elts = idx.elts if isinstance(idx, ast.Tuple) else [idx]
      if idx is None or [ast.dump(e) for e in elts] != own:
        pointwise.discard(n.id)
  return frozenset(pointwise)


def _path(name: str) -> tuple[bool, str, str]:
  """Returns where the Model or Data field a parameter is named after lives."""
  is_data = name.endswith("_in") or name.endswith("_out")
  field = name.rsplit("_", 1)[0] if is_data else name
  for sub in ("efc", "contact") if is_data else ("opt", "stat"):
    if field.startswith(sub + "_"):
      return is_data, sub, field[len(sub) + 1 :]
  return is_data, "", field


def _field(m, d, s: Stage, p: str):
  """Returns the Model or Data field parameter `p` of stage `s` is named after."""
  is_data, sub, field = s.fields[p]
  obj = d if is_data else m
  if sub:
    obj = getattr(obj, sub)
  try:
    return getattr(obj, field)
  except AttributeError:
    raise AttributeError(f"{s.name} parameter {p!r} is not a Model or Data field, pass it to launch()") from None


def _size(m, d, dim: str) -> int:
  return d.nworld if dim == "nworld" else getattr(m, dim)


def _key(a) -> Any:
  """Identifies the memory an argument refers to, so aliases of an array conflict."""
  if isinstance(a, wp.array):
    return a.ptr if a.ptr else id(a)
  return None


@dataclasses.dataclass
class _Launch:
  """A stage with the arguments it was launched with."""

  stage: Stage
  args: dict[str, Any]
  sizes: tuple[int, ...]


@dataclasses.dataclass
class _Group:
  """Launches fused into one kernel: segments of launches sharing threads, and the arrays used."""

  segments: list[list[_Launch]]
  touched: list[set]  # arrays each segment reads or writes
  read: set
  write: set


def _plan(launches: list[_Launch]) -> list[list[list[_Launch]]]:
  """Groups launches, in order, into kernels of segments."""
  groups = []
  for ln in launches:
    s = ln.stage
    r = {_key(a) for p, a in ln.args.items() if not p.endswith("_out")} - {None}
    w = {_key(a) for p, a in ln.args.items() if p.endswith("_out")} - {None}
    g = groups[-1] if groups and FUSE else None
    if g is not None:
      last = g.segments[-1]
      hazard = (w & (g.read | g.write)) | (r & g.write)
      if not hazard:
        # independent of the kernel so far: a segment of its own threads
        g.segments.append([ln])
        g.touched.append(r | w)
      elif (
        s.dim == last[0].stage.dim
        and ln.sizes == last[0].sizes
        and not (hazard & set().union(*g.touched[:-1]))
        and all(p in t.stage.pointwise for t in [*last, ln] for p, a in t.args.items() if _key(a) in hazard)
      ):
        # depends only on the last segment, at the thread's own index: runs after it in its threads
        last.append(ln)
        g.touched[-1] |= r | w
      else:
        g = None
      if g is not None:
        g.read |= r
        g.write |= w
        continue
    groups.append(_Group([[ln]], [r | w], r, w))
  return [g.segments for g in groups]


_KERNELS = {}


def _kernel(structure) -> wp.Kernel:
  """Generates the kernel running each segment of stages over its own range of threads.

  `structure` holds, per segment, its stages with the argument slot of each parameter: launches
  that pass the same array under the same name share a kernel argument.
  """
  kernel = _KERNELS.get(structure)
  if kernel is not None:
    return kernel

  namespace = {"wp": wp, "__name__": __name__}
  slots = {}  # slot -> (argument name, annotation)
  for seg in structure:
    for s, seg_slots in seg:
      for (p, typ), slot in zip(s.params.items(), seg_slots):
        if slot not in slots:
          name = p if all(n != p for n, _ in slots.values()) else f"{p}_{slot}"
          slots[slot] = (name, typ)
  for slot, (name, typ) in slots.items():
    namespace[f"T_{name}"] = typ

  sizes = [f"n{i}_{j}" for i, seg in enumerate(structure) for j in range(len(seg[0][0].dim))]
  args = ", ".join([f"{n}: int" for n in sizes] + [f"{name}: T_{name}" for name, _ in slots.values()])
  name = "fused__" + "__".join(s.name.lstrip("_") for seg in structure for s, _ in seg)
  if len(name) > 96:
    # kernel names end up in cache paths, keep them short and stable
    name = name[:80] + "_" + hashlib.sha1(name.encode()).hexdigest()[:12]
  body = [f"def {name}({args}):", "  tid = wp.tid()", "  start = int(0)"]
  for i, seg in enumerate(structure):
    n = [f"n{i}_{j}" for j in range(len(seg[0][0].dim))]
    body.append(f"  end = start + {' * '.join(n)}")
    body.append("  if tid >= start and tid < end:")
    body.append("    t = tid - start")
    for j in reversed(range(1, len(n))):
      body.append(f"    i{j} = t % {n[j]}")
      body.append(f"    t = t // {n[j]}")
    index = [f"i{j}" for j in range(1, len(n))]
    for k, (s, seg_slots) in enumerate(seg):
      namespace[f"f{i}_{k}"] = s.func
      body.append(f"    f{i}_{k}({', '.join(['t'] + index + [slots[slot][0] for slot in seg_slots])})")
    body.append("  start = end")
  source = "\n".join(body) + "\n"

  filename = f"<mujoco_warp {name}>"
  linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
  exec(compile(source, filename, "exec"), namespace)
  kernel = wp.kernel(namespace[name], module="unique", enable_backward=False)
  _KERNELS[structure] = kernel
  return kernel


def _record(segments: list[list[_Launch]]) -> wp.Launch:
  """Returns the launch command of the kernel running the segments."""
  slots, values = {}, []
  structure = []
  for seg in segments:
    structure.append([])
    for ln in seg:
      seg_slots = []
      for p, a in ln.args.items():
        slot = slots.setdefault((p, _key(a) or id(a)), len(slots))
        if slot == len(values):
          values.append(a)
        seg_slots.append(slot)
      structure[-1].append((ln.stage, tuple(seg_slots)))
  structure = tuple(tuple(seg) for seg in structure)
  sizes = [n for seg in segments for n in seg[0].sizes]
  dim = sum(math.prod(seg[0].sizes) for seg in segments)
  # a recording tape needs the launch itself, so it is not replayed from a command
  return wp.launch(_kernel(structure), dim=dim, inputs=sizes + values, record_cmd=wp_context.runtime.tape is None)


def _arg_key(a) -> Any:
  """Identifies an argument by value, an array by the memory and layout it refers to."""
  if isinstance(a, wp.array):
    return a.ptr, a.shape, a.strides
  return a


# launch commands for each sequence of stages with the same arguments, which hold those arguments
_COMMANDS = {}
_MAX_COMMANDS = 1024


def _launch(launches: list[_Launch]):
  """Launches the stages, recording the fused launch commands the first time they are seen."""
  if wp_context.runtime.tape is not None:
    for segments in _plan(launches):
      _record(segments)
    return
  key = (FUSE, *((ln.stage, ln.sizes, *(_arg_key(a) for a in ln.args.values())) for ln in launches))
  commands = _COMMANDS.get(key)
  if commands is None:
    if len(_COMMANDS) >= _MAX_COMMANDS:
      _COMMANDS.clear()
    # empty launches record no command
    commands = _COMMANDS[key] = [c for c in map(_record, _plan(launches)) if c is not None]
  for cmd in commands:
    cmd.launch()


_QUEUE = None


@contextlib.contextmanager
def fuse():
  """Queues the stages launched inside, and launches them fused when the block ends.

  Launches that are not stages run when called, so any inside the block must come after a
  `flush()` if they use arrays the queued stages write, and must not write arrays they use.
  """
  global _QUEUE
  if _QUEUE is not None:
    yield
    return
  _QUEUE = queue = []
  try:
    yield
  finally:
    _QUEUE = None
    if queue:
      _launch(queue)


def flush():
  """Launches the queued stages."""
  if _QUEUE:
    queue = _QUEUE[:]
    _QUEUE.clear()
    _launch(queue)


def launch(m, d, *stages: Stage, **bindings):
  """Launches the stages in order, fused into as few kernels as their data dependencies allow.

  Args:
    m: The model, which binds parameters named after Model fields and sizes the dims.
    d: The data, which binds `_in` and `_out` parameters and sizes the nworld dim.
    *stages: The stages, in the order they would be launched.
    **bindings: Arguments for parameters that are not Model or Data fields, or that alias a
      different array than the field of their name.
  """
  launches = [
    _Launch(
      s,
      {p: bindings[p] if p in bindings else _field(m, d, s, p) for p in s.params},
      tuple(_size(m, d, n) for n in s.dim),
    )
    for s in stages
  ]
  if _QUEUE is not None:
    _QUEUE.extend(launches)
  else:
    _launch(launches)
