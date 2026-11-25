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
for Constraint and Contact). `launch(m, d, stage)` binds them by name, and arguments that are not
those fields are passed by name.

Inside a `fuse()` block, stages are queued with the other device work of the block (`zero`, `fill`,
`copy`, `launch_kernel`, `launch_tiled`, `call`, `capture_while`, `capture_if`), and launched when
the block ends, fused into as few kernels as their data dependencies allow:

- a stage that does not touch an array another stage in the kernel writes shares it, over its own
  range of threads, and may move ahead of queued work it does not touch either
- a stage with the same sizes as the stage before it runs in the same thread after it, if every
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
from typing import Any, Callable, Optional

import warp as wp
from warp._src import context as wp_context

# "fused": queue and fuse, "unfused": queue and launch one kernel per stage, "immediate": launch
# every stage and operation when called, the reference that queued launches must match
MODE = "fused"

# module options a fused kernel takes from its stages, which only fuse with equal options
_OPTIONS = ("default_grid_stride", "fast_math", "max_unroll", "fuse_fp", "deterministic", "deterministic_max_records")

# kernel argument bytes a fused kernel may use, below the 4 KB CUDA limit of older toolkits
_MAX_ARG_BYTES = 4000


@dataclasses.dataclass(frozen=True, eq=False)
class Stage:
  """A kernel body with its default launch dims and the parameters it binds by name."""

  func: wp.Function
  name: str
  dim: tuple[str, ...]
  params: dict[str, Any]  # parameter name -> annotation, after the thread indices
  fields: dict[str, tuple[bool, str, str]]  # parameter name -> (in Data, sub-struct or "", field)
  pointwise: frozenset[str]  # array parameters only accessed at the thread indices
  options: tuple  # module options of the kernel that runs it
  fingerprint: str  # the values the body closes over, which tell apart stages of one factory


def stage(*dim: str, options: Optional[dict] = None):
  """Declares a kernel body as a stage, launched over `dim` unless `launch` is given `dim`.

  Args:
    *dim: Sizes named after Model or Data fields (or nworld), one per thread index parameter.
    options: Module options of the kernel, which default to those of the module defining it.
  """

  def decorator(f):
    sig = inspect.signature(f)
    names = list(sig.parameters)
    index, params = names[: len(dim)], names[len(dim) :]
    pointwise = _pointwise(f, index, params)
    module = wp.get_module(f.__module__).options
    opts = {k: module.get(k) for k in _OPTIONS}
    opts.update({k: v for k, v in (options or {}).items() if k in _OPTIONS})
    return Stage(
      func=wp.func(f),
      name=f.__name__,
      dim=dim,
      params={p: sig.parameters[p].annotation for p in params},
      fields={p: _path(p) for p in params},
      pointwise=pointwise,
      options=tuple(sorted(opts.items())),
      fingerprint=_fingerprint(f),
    )

  return decorator


def _is_out(name: str) -> bool:
  """Whether a parameter is written, by the kernel analyzer naming."""
  return name.endswith("_out") or name in ("res", "out")


def _fingerprint(f) -> str:
  """Describes the values a function closes over, stable across processes where possible."""

  def describe(v):
    if isinstance(v, (list, tuple)):
      return "(" + ",".join(describe(x) for x in v) + ")"
    if isinstance(v, (wp.Function, Stage)):
      return getattr(v, "key", None) or v.name + v.fingerprint
    if isinstance(v, (bool, int, float, str, type(None))) or hasattr(v, "__members__") or isinstance(v, type):
      return repr(v)
    return f"{type(v).__name__}:{id(v)}"

  return ";".join(f"{k}={describe(v)}" for k, v in sorted(inspect.getclosurevars(f).nonlocals.items()))


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
    if written in params and not _is_out(written):
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
  if dim == "nworld":
    return d.nworld
  return getattr(m, dim) if hasattr(m, dim) else getattr(d, dim)


def _key(a) -> Any:
  """Identifies the memory an argument refers to, so aliases of an array conflict."""
  if isinstance(a, wp.array):
    return a.ptr if a.ptr else id(a)
  return None


def _keys(arrays) -> set:
  return {_key(a) for a in arrays} - {None}


@dataclasses.dataclass
class _Launch:
  """A stage with the arguments it was launched with."""

  stage: Stage
  args: dict[str, Any]
  sizes: tuple[int, ...]
  persistent: bool  # whether every argument is a Model or Data field, so a command can replay it
  read: set
  write: set
  stream: Any  # the stream it was launched on


@dataclasses.dataclass
class _Op:
  """Device work that is not a stage, with the arrays it reads and writes."""

  fn: Callable
  read: set
  write: set
  stream: Any
  barrier: bool = False  # nothing moves ahead of it


@dataclasses.dataclass
class _Group:
  """Launches fused into one kernel: segments of launches sharing threads, and the arrays used."""

  segments: list[list[_Launch]]
  touched: list[set]  # arrays each segment reads or writes
  read: set
  write: set
  nbytes: int
  stream: Any


def _hazard(read, write, other_read, other_write) -> set:
  return (write & (other_read | other_write)) | (read & other_write)


def _nbytes(ln: _Launch) -> int:
  return sum(16 + 8 * a.ndim if isinstance(a, wp.array) else 16 for a in ln.args.values())


def _plan(entries: list) -> list:
  """Orders the entries into kernels of segments (`_Group`) and operations (`_Op`)."""
  schedule = []
  last = None  # index in schedule of the last group
  for e in entries:
    if isinstance(e, _Op):
      schedule.append(e)
      continue
    g = schedule[last] if last is not None and MODE == "fused" else None
    if g is not None:
      later = schedule[last + 1 :]
      seg = g.segments[-1]
      hazard = _hazard(e.read, e.write, g.read, g.write)
      # nothing moves across a stream change or a barrier
      fits = (
        e.stream == g.stream
        and all(op.stream == e.stream and not op.barrier for op in later)
        and e.stage.options == seg[0].stage.options
        and g.nbytes + _nbytes(e) <= _MAX_ARG_BYTES
        and not any(_hazard(e.read, e.write, op.read, op.write) for op in later)
      )
      if fits and not hazard:
        # independent of the kernel so far: a segment of its own threads
        g.segments.append([e])
        g.touched.append(e.read | e.write)
      elif (
        fits
        and not later
        and e.sizes == seg[0].sizes
        and not (hazard & set().union(*g.touched[:-1]))
        and all(p in t.stage.pointwise for t in [*seg, e] for p, a in t.args.items() if _key(a) in hazard)
      ):
        # depends only on the last segment, at the thread's own index: runs after it in its threads
        seg.append(e)
        g.touched[-1] |= e.read | e.write
      else:
        g = None
      if g is not None:
        g.read |= e.read
        g.write |= e.write
        g.nbytes += _nbytes(e)
        continue
    last = len(schedule)
    schedule.append(_Group([[e]], [e.read | e.write], set(e.read), set(e.write), _nbytes(e), e.stream))
  return schedule


_KERNELS = {}


def _kernel(structure, options: tuple) -> wp.Kernel:
  """Generates the kernel running each segment of stages over its own range of threads.

  `structure` holds, per segment, its stages with the argument slot of each parameter and their
  number of dims: launches that pass the same array under the same name share a kernel argument.
  """
  kernel = _KERNELS.get(structure)
  if kernel is not None:
    return kernel

  namespace = {"wp": wp, "__name__": __name__}
  slots = {}  # slot -> (argument name, annotation)
  for seg in structure:
    for s, seg_slots, _ in seg:
      for (p, typ), slot in zip(s.params.items(), seg_slots):
        if slot not in slots:
          name = p if all(n != p for n, _ in slots.values()) else f"{p}_{slot}"
          slots[slot] = (name, typ)
  for name, typ in slots.values():
    namespace[f"T_{name}"] = typ

  sizes = [f"n{i}_{j}" for i, seg in enumerate(structure) for j in range(seg[0][2])]
  args = ", ".join([f"{n}: int" for n in sizes] + [f"{name}: T_{name}" for name, _ in slots.values()])
  name = "fused__" + "__".join(s.name.lstrip("_") for seg in structure for s, _, _ in seg)
  # Warp's module hash leaves out what the stage functions close over, so the name tells apart
  # stages of one factory, and stays short since it ends up in cache paths
  fingerprint = "|".join(s.fingerprint for seg in structure for s, _, _ in seg)
  name = name[:80] + "_" + hashlib.sha1(fingerprint.encode()).hexdigest()[:12]
  body = [f"def {name}({args}):", "  tid = wp.tid()", "  start = int(0)"]
  for i, seg in enumerate(structure):
    n = [f"n{i}_{j}" for j in range(seg[0][2])]
    body.append(f"  end = start + {' * '.join(n)}")
    body.append("  if tid >= start and tid < end:")
    body.append("    t = tid - start")
    for j in reversed(range(1, len(n))):
      body.append(f"    i{j} = t % {n[j]}")
      body.append(f"    t = t // {n[j]}")
    index = [f"i{j}" for j in range(1, len(n))]
    for k, (s, seg_slots, _) in enumerate(seg):
      namespace[f"f{i}_{k}"] = s.func
      body.append(f"    f{i}_{k}({', '.join(['t'] + index + [slots[slot][0] for slot in seg_slots])})")
    body.append("  start = end")
  source = "\n".join(body) + "\n"

  filename = f"<mujoco_warp {name}>"
  linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
  exec(compile(source, filename, "exec"), namespace)
  opts = dict(options)
  grid_stride = opts.pop("default_grid_stride")
  module_options = {k: v for k, v in opts.items() if v is not None}
  kernel = wp.kernel(
    namespace[name], module="unique", module_options=module_options, enable_backward=False, grid_stride=grid_stride
  )
  kernel.stages = [s for seg in structure for s, _, _ in seg]
  _KERNELS[structure] = kernel
  return kernel


def kernel(s) -> wp.Kernel:
  """Returns the kernel a stage runs as on its own (a kernel is returned as is), for occupancy."""
  if isinstance(s, wp.Kernel):
    return s
  slots = tuple(range(len(s.params)))
  return _kernel((((s, slots, len(s.dim)),),), s.options)


def _arg_key(a) -> Any:
  """Identifies an argument by value, an array by the memory and layout it refers to."""
  if isinstance(a, wp.array):
    return a.ptr, a.shape, a.strides
  return a


# launch commands of fused kernels whose arguments are all Model or Data fields, which they hold
_COMMANDS = {}
_MAX_COMMANDS = 4096


def _launch_group(segments: list[list[_Launch]]):
  """Launches the kernel running the segments, replaying its recorded command if it has one."""
  persistent = all(ln.persistent for seg in segments for ln in seg) and wp_context.runtime.tape is None
  if persistent:
    key = tuple((ln.stage, ln.sizes, *(_arg_key(a) for a in ln.args.values())) for seg in segments for ln in seg)
    cmd = _COMMANDS.get(key)
    if cmd is not None:
      cmd.launch()
      return

  slots, values, structure = {}, [], []
  for seg in segments:
    structure.append([])
    for ln in seg:
      seg_slots = []
      for p, a in ln.args.items():
        slot = slots.setdefault((p, _key(a) or id(a)), len(slots))
        if slot == len(values):
          values.append(a)
        seg_slots.append(slot)
      structure[-1].append((ln.stage, tuple(seg_slots), len(ln.sizes)))
  structure = tuple(tuple(seg) for seg in structure)
  sizes = [n for seg in segments for n in seg[0].sizes]
  dim = sum(math.prod(seg[0].sizes) for seg in segments)
  kernel = _kernel(structure, segments[0][0].stage.options)
  # a recorded command is not launched until replayed
  cmd = wp.launch(kernel, dim=dim, inputs=sizes + values, record_cmd=persistent)
  if persistent and cmd is not None:
    if len(_COMMANDS) >= _MAX_COMMANDS:
      _COMMANDS.clear()
    _COMMANDS[key] = cmd
    cmd.launch()


_RUNNING = [0]  # nesting depth of _run, during which device work comes from the queue


def _run(entries: list):
  _RUNNING[0] += 1
  try:
    for item in _plan(entries):
      # each item runs on the stream it was launched on, which waits for the stream of the flush
      scope = wp.ScopedStream(item.stream) if item.stream is not None and item.stream != _stream() else contextlib.nullcontext()
      with scope:
        if isinstance(item, _Op):
          item.fn()
        else:
          _launch_group(item.segments)
  finally:
    _RUNNING[0] -= 1


def pending() -> bool:
  """Whether queued work would be passed by device work launched now, outside this module."""
  return not _RUNNING[0] and any(_QUEUES)


_QUEUES = []  # the queue of the innermost fuse() block or capture body


def _queue() -> Optional[list]:
  if not _QUEUES or MODE == "immediate" or wp_context.runtime.tape is not None:
    return None
  return _QUEUES[-1]


@contextlib.contextmanager
def fuse():
  """Queues the stages and operations of the block, and launches them fused when it ends.

  Device work in the block must go through this module (or follow a `flush()`), so its order
  with the queued stages is kept. A nested block joins the outer one.
  """
  if _QUEUES:
    yield
    return
  _QUEUES.append([])
  try:
    yield
  finally:
    _run(_QUEUES.pop())


def flush():
  """Launches the queued stages and operations."""
  queue = _queue()
  if queue:
    entries = queue[:]
    queue.clear()
    _run(entries)


def _stream():
  device = wp.get_device()
  return wp.get_stream(device) if device.is_cuda else None


def _submit(entry):
  queue = _queue()
  if queue is None:
    _run([entry])
  else:
    queue.append(entry)


def launch(m, d, stage: Stage, dim=None, **bindings):
  """Launches a stage, queued in a `fuse()` block.

  Args:
    m: The model, which binds parameters named after Model fields and sizes named dims.
    d: The data, which binds `_in` and `_out` parameters and sizes named dims.
    stage: The stage.
    dim: Launch dims, instead of the stage's named dims.
    **bindings: Arguments for parameters that are not Model or Data fields, or that alias a
      different array than the field of their name.
  """
  args = {p: bindings[p] if p in bindings else _field(m, d, stage, p) for p in stage.params}
  if dim is None:
    sizes = tuple(_size(m, d, n) for n in stage.dim)
  else:
    sizes = tuple(int(n) for n in (dim if isinstance(dim, (tuple, list)) else (dim,)))
  if len(sizes) != len(stage.dim):
    raise ValueError(f"{stage.name} takes {len(stage.dim)} dims, got {sizes}")
  read = _keys(a for p, a in args.items() if not _is_out(p))
  write = _keys(a for p, a in args.items() if _is_out(p))
  persistent = not any(isinstance(bindings.get(p), wp.array) for p in stage.params)
  _submit(_Launch(stage, args, sizes, persistent, read, write, _stream()))


def call(fn: Callable, *args, reads=(), writes=(), **kwargs):
  """Calls `fn(*args, **kwargs)`, queued in a `fuse()` block, as device work using the arrays."""
  _submit(_Op(lambda: fn(*args, **kwargs), _keys(reads) | _keys(writes), _keys(writes), _stream()))


def wait_stream(stream):
  """Makes the current stream wait for `stream`, as `wp.wait_stream`; nothing moves ahead of it."""
  _submit(_Op(lambda: wp.wait_stream(stream), set(), set(), _stream(), barrier=True))


def zero(a: wp.array):
  """Zeros an array."""
  call(a.zero_, writes=(a,))


def fill(a: wp.array, value):
  """Fills an array with a value."""
  call(a.fill_, value, writes=(a,))


def copy(dest: wp.array, src: wp.array, *args, **kwargs):
  """Copies an array, as `wp.copy`."""
  call(wp.copy, dest, src, *args, reads=(src,), writes=(dest,), **kwargs)


def clone(src: wp.array) -> wp.array:
  """Returns a copy of an array, as `wp.clone`."""
  dest = wp.empty_like(src)
  copy(dest, src)
  return dest


def launch_kernel(kernel, dim, inputs=(), outputs=(), **kwargs):
  """Launches a kernel that is not a stage, as `wp.launch`, outputs read and written."""
  call(wp.launch, kernel, dim=dim, inputs=list(inputs), outputs=list(outputs), reads=inputs, writes=outputs, **kwargs)


def launch_tiled(kernel, dim, inputs=(), outputs=(), **kwargs):
  """Launches a tiled kernel, as `wp.launch_tiled`, outputs read and written."""
  call(wp.launch_tiled, kernel, dim=dim, inputs=list(inputs), outputs=list(outputs), reads=inputs, writes=outputs, **kwargs)


def _body(fn):
  """Runs a capture body with its own queue, launched within the body."""
  if fn is None or not callable(fn):
    return fn

  def body(*args, **kwargs):
    _QUEUES.append([])
    try:
      return fn(*args, **kwargs)
    finally:
      _run(_QUEUES.pop())

  return body


def capture_while(condition: wp.array, while_body, **kwargs):
  """Runs `while_body` while `condition` is nonzero, as `wp.capture_while`."""
  flush()
  wp.capture_while(condition, _body(while_body), **kwargs)


def capture_if(condition: wp.array, on_true=None, on_false=None, **kwargs):
  """Runs `on_true` if `condition` is nonzero, else `on_false`, as `wp.capture_if`."""
  flush()
  wp.capture_if(condition, _body(on_true), _body(on_false), **kwargs)
