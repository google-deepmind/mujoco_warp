"""Shared harness for the diffsim demos: config, gradients, capture/reuse, the optimize loop, main().

A demo subclasses `Example` (viz.py is the render counterpart) and supplies only its model + taped
forward + loss + per-env MuJoCo-C rollout + param plumbing + render; the harness provides CUDA-graph
capture with BackwardContext reuse, the fd/analytic gradient dispatch, the parallel gradient-descent
`optimize()` loop, a typed `Args(CommonArgs)` dataclass bridged to absl.flags, and the `run()` main
flow (device scope -> optimize/benchmark -> render).

The analytic path is always CHECKPOINTED: `chunk_step` defines one physics step (+ optional per-step
loss) and the harness captures/replays ONE chunk over the trajectory (exact via segment recompute;
only chunk+1 Datas held). Terminal terms go in `terminal_loss()`; TBPTT demos set `truncated=True`
to detach the carried state adjoint per chunk while model/schedule leaf grads still accumulate.
Params are host numpy `self.params` (num_envs, k), written in place into datas[0] each iter
(set_params) with the per-env grad read back after the backward (read_grad); the eager (CPU) and
captured paths produce identical gradients.

  import demo                          # demos already `sys.path.insert(0, dirname(__file__))`
  from absl import app

  @dataclass
  class Args(demo.CommonArgs):
    num_envs: int = 4                  # override a default; add demo-specific fields below

  class MyDemo(demo.Example):
    Args = Args
    def build_model(self): ...         # set self.mjm, self.mjd (+ mj_forward)
    def chunk_step(self, i, t): ...    # one mjw.step over datas[i]->datas[i+1] (+ optional per-step loss)
    def terminal_loss(self): ...       # optional: terminal term from the final state datas[0]

  demo.define_flags(Args)
  def main(argv):
    demo.run(MyDemo, demo.parse_args(Args))
  if __name__ == "__main__":
    app.run(main)
"""

import contextlib
import json
import os
import typing
from dataclasses import MISSING
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields

import numpy as np
import warp as wp
import warp.optim
from absl import flags

import mujoco_warp as mjw

# Every demo runs a taped backward, which needs the grad opt-in (forward.step() hard-raises on a taped
# record while grad is off). Enable it here so a demo body stays just model+forward+loss+render.
mjw.enable_grad()


@wp.kernel
def _scatter_ctrl(src: wp.array(dtype=float), nu: int, dst: wp.array2d(dtype=float)):
  """Scatter a flat (num_envs*nu,) control leaf into a Data.ctrl buffer (num_envs, nu). Used by
  Example.bind_ctrl so a schedule demo's per-step control leaf reaches the sim WITHOUT rebinding
  datas[i].ctrl (mjw.step copies ctrl into d_out; a rebind aliases and drops the last-step ctrl grad in
  a multi-step chunk -- the mjw adjoint_test _assign_shared_ctrl pattern). (@wp.kernel stays module-level.)"""
  w, u = wp.tid()
  dst[w, u] = src[w * nu + u]


# ---- Args dataclass <-> absl.flags bridge -------------------------------------------------------


@dataclass
class CommonArgs:
  """Config shared by every diffsim demo; a demo subclasses this with its own fields + defaults.

  `metadata={"help": ..., "choices": [...]}` on a field sets the flag help / makes it an enum. A demo
  overrides a default just by redeclaring the field (e.g. `num_envs: int = 4`) and adds its own."""

  grad: str = field(default="analytic", metadata={"help": "gradient source", "choices": ["analytic", "fd"]})
  num_envs: int = field(default=1, metadata={"help": "parallel envs optimized at once"})
  steps: int = field(default=100, metadata={"help": "gradient-descent steps"})
  lr: float = field(default=0.1, metadata={"help": "gradient-descent learning rate"})
  spread: float = field(default=0.0, metadata={"help": "per-env spread of the init params (visual fan)"})
  device: typing.Optional[str] = field(default=None, metadata={"help": "warp device, e.g. cuda:0 or cpu"})
  capture: typing.Optional[bool] = field(default=None, metadata={"help": "force CUDA graph capture (default: auto)"})
  chunk: int = field(default=1, metadata={"help": "checkpoint/capture unit in steps for the analytic path (>=1): 1 = a single physics step (default; smallest graph + least memory), larger = fewer graph replays. Graph capture on/off is the separate --capture/--nocapture flag."})
  out: typing.Optional[str] = field(default=None, metadata={"help": "output mp4 path (else the demo default)"})
  live: bool = field(default=False, metadata={"help": "open the live MuJoCo viewer instead of an mp4"})
  no_render: bool = field(default=False, metadata={"help": "skip rendering (compute only)"})
  export_usd: bool = field(default=False, metadata={"help": "export the demo's viz trajectory to an animated USD and exit"})
  usd_frames: int = field(default=150, metadata={"help": "USD export: number of animation frames"})
  usd_fps: int = field(default=30, metadata={"help": "USD export: animation playback fps"})
  # Blender field export (--export_usd): replay the OPTIMIZATION PROCESS across an instanced grid (one shared
  # proto per lane). Generic defaults for a small square field; a demo overrides them (cradle: 13x13 @ 2.4).
  usd_envs: int = field(default=25, metadata={"help": "USD field: number of instanced lanes"})
  usd_cols: int = field(default=5, metadata={"help": "USD field: grid columns"})
  usd_xpitch: float = field(default=2.2, metadata={"help": "USD field: column pitch (x)"})
  usd_ypitch: float = field(default=2.2, metadata={"help": "USD field: row pitch (y)"})
  usd_iters: int = field(default=6, metadata={"help": "USD field: optimization iterations sampled across the run"})
  usd_stride: int = field(default=4, metadata={"help": "USD field: timestep subsample within each iteration's rollout"})
  usd_hold: int = field(default=6, metadata={"help": "USD field: frames held at each rollout's settled end"})


_DEFINERS = {bool: flags.DEFINE_bool, int: flags.DEFINE_integer, float: flags.DEFINE_float, str: flags.DEFINE_string}


def _base_type(hint):
  """Strip `Optional[X]` / `X | None` down to X so the flag type dispatches on the underlying type."""
  ga = typing.get_args(hint)
  if ga:
    non_none = [a for a in ga if a is not type(None)]
    if len(non_none) == 1:
      return non_none[0]
  return hint


def define_flags(args_cls, flag_values=None):
  """Register one absl flag per field of an Args dataclass (typed from the annotation), so a demo's
  CLI is just its dataclass. Call at demo module scope; `app.run` parses argv, then `parse_args`
  rebuilds the dataclass. `choices` metadata -> DEFINE_enum; `Optional[bool]` -> tri-state flag."""
  fv = flag_values if flag_values is not None else flags.FLAGS
  hints = typing.get_type_hints(args_cls)
  for f in fields(args_cls):
    default = f.default if f.default is not MISSING else None
    help_ = f.metadata.get("help", f.name)
    choices = f.metadata.get("choices")
    if choices is not None:
      flags.DEFINE_enum(f.name, default, choices, help_, flag_values=fv)
    else:
      _DEFINERS.get(_base_type(hints[f.name]), flags.DEFINE_string)(f.name, default, help_, flag_values=fv)
  return fv


def parse_args(args_cls, flag_values=None):
  """Build an Args dataclass instance from the already-parsed absl flags (call inside `main(argv)`)."""
  fv = flag_values if flag_values is not None else flags.FLAGS
  return args_cls(**{f.name: getattr(fv, f.name) for f in fields(args_cls)})


# ---- capture support ----------------------------------------------------------------------------


def capture_supported(device=None):
  """True iff `device` (default: Warp's current) can CUDA-graph-capture the taped backward: a CUDA
  device with a memory pool (capture-with-alloc needs the mempool; World A, MJPLAN_STYLE §7.2)."""
  device = device or wp.get_device()
  return device.is_cuda and wp.is_mempool_enabled(device)


def resolve_capture(capture, device=None):
  """Normalize `capture`: None auto-detects (capture iff supported); True/False force it.

  Forcing True on an unsupported device raises (fail loud, not a silent fall back to eager)."""
  device = device or wp.get_device()
  can = capture_supported(device)
  if capture and not can:
    raise RuntimeError(f"capture=True needs a CUDA device with mempool enabled (got {device})")
  return can if capture is None else capture


# ---- the demo template --------------------------------------------------------------------------


class Example:
  """Base for a parallel gradient-descent diffsim demo.

  Subclass + set `Args`, then implement the demo hooks (see the NotImplementedError stubs below):
  the model, the taped forward+loss, the per-env MuJoCo-C rollout (for the video + display loss),
  the param plumbing (init/set/read), the per-env FD, and the render. The harness owns capture +
  BackwardContext reuse of the analytic backward, the fd/analytic dispatch, and the descent loop.

  `self.params` is host numpy of shape (num_envs, k) -- the optimized DOFs only; set_params expands
  it into datas[0] (and rollout_env into the MuJoCo-C state). The analytic machinery (m/datas/loss/bc
  + capture) is built only for `grad == "analytic"`; the fd path needs just mjm/mjd."""

  Args = CommonArgs
  log_every = 10
  # Demo-declared shape flags (see HARNESS_MIGRATION.md). Defaults suit a state-leaf, exact-BPTT,
  # capturable demo (bounce/cradle); other styles flip these.
  capturable = True  # False -> FORCE the eager checkpointed path: schedule/time-dependent demos whose
  # per-step control leaf or host step index can't be baked into a single captured chunk-0 graph.
  truncated = False  # True -> DETACH the carried state adjoint at chunk boundaries (truncated BPTT).
  scatter_params = False  # True (Style-D) -> set_params scatters self.param into datas[0] via a
  # differentiable kernel; backward() re-runs it on a mini-tape to land the grad on self.param.grad.
  ckpt_carry = ("qacc_warmstart",)  # Data fields (besides qpos/qvel) that mjw.step carries across steps
  # -- the solver warmstart, plus e.g. "act" for stateful actuators -- restored as DETACHED values so the
  # segment recompute reproduces the forward EXACTLY (without them, many-contact rollouts drift chunk to
  # chunk). Their gradient is not carried (a converged solver's fixed point is warmstart-independent).

  def __init__(self, args):
    self.args = args
    self.build_model()  # DEMO: self.mjm, self.mjd (+ mj_forward)
    self.params = self.init_params()  # DEMO: (num_envs, k) host numpy
    self.m = mjw.put_model(self.mjm)  # the warp Model, reused across iterations (both grad paths)
    self.use_graph = False
    if args.grad == "analytic" and not getattr(args, "classic", False):  # analytic path (skip for the classic no-opt sim)
      assert args.chunk >= 1, f"chunk must be >= 1 (got {args.chunk})"
      self.build_datas()  # DEMO: self.datas (chunk+1 segment buffers), self.loss, datas[0] leaves
      self.use_graph = resolve_capture(args.capture) if self.capturable else False
      self.tape = wp.Tape()  # persistent recorded tape (graph path); replaced per call on the eager path
      # Preallocate the backward's reusable scratch ONCE; active during capture -> the reuse (~1
      # qpos-copy/step instead of a full Data clone/step) is baked into the captured graph.
      self.bc = mjw.create_backward_context(self.m, self.datas[0])
      self.step_offset = wp.zeros(1, dtype=int)  # global-step base per chunk; a time-dep loss kernel can
      # read step_offset[0]+i to stay capture-safe (eager chunk_step just uses the Python t).
      self._build_ckpt()  # analytic ⇒ ALWAYS checkpointed (single-step/chunk capture + gradient checkpointing)

  # ---- checkpointed backward (harness; the ONLY analytic path) ----

  def backward(self):
    """Checkpointed analytic backward. Segmented forward (per-step loss accumulates into self.loss), an
    optional terminal-loss mini-tape (adds its term + seeds the last chunk's incoming state adjoint), the
    reverse segment loop (state adjoint carried across chunks, or DETACHED when `truncated`; a model-param
    leaf's grad accumulated across chunks), then an optional differentiable set_params backprop
    (Style-D scatter: datas[0] adjoint -> self.param.grad)."""
    self._ckpt_fwd_pass()                  # datas[0] = state_T; self.loss = per-step sum (0 if none)
    if hasattr(self, "param"):
      self.param.grad.zero_()              # clean target for terminal-param + scatter backprop
    seed = self._terminal_seed()           # None, or (dL/dqpos_T, dL/dqvel_T) from terminal_loss()
    wp.copy(self.ckpt_loss, self.loss)     # stash the true forward loss (recompute re-accumulates it)
    self._ckpt_bwd_pass(seed)              # reverse -> datas[0].grad = dL/dstate_0 (+ accum_leaf grad)
    wp.copy(self.loss, self.ckpt_loss)     # restore the reported loss
    if self.scatter_params:                # Style-D: differentiate the param -> datas[0] scatter
      tp = wp.Tape()
      with tp:
        self.set_params()
      tp.backward()                        # datas[0].{qpos,qvel}.grad -> self.param.grad (accumulate)

  def _terminal_seed(self):
    """Terminal-loss demos: run terminal_loss() on the final state (datas[0] == state_T after the forward
    pass) on a mini-tape -- it adds its term to self.loss and, via tape.backward, yields dL_terminal/
    dstate_T (which seeds the last chunk's incoming state adjoint) plus any DIRECT param grad (e.g.
    dominos' effort term) into self.param.grad. Returns None when the demo has no terminal loss."""
    if not hasattr(self, "terminal_loss"):
      return None
    d0 = self.datas[0]
    d0.qpos.grad.zero_()
    d0.qvel.grad.zero_()
    tt = wp.Tape()
    with tt:
      self.terminal_loss()
    tt.backward(loss=self.loss)
    return (wp.clone(d0.qpos.grad), wp.clone(d0.qvel.grad))

  # ---- checkpointed analytic path: single-step/chunk graph capture + gradient checkpointing ---------
  # Replaces whole-rollout capture (which chokes past ~1400 steps -> forced eager, host-dispatch-bound)
  # with a per-CHUNK graph replayed over the trajectory: the whole tape is exact via segment recompute,
  # yet the captured graph is ONE chunk (size independent of the horizon), so capture never chokes and
  # only chunk+1 Datas are held. Opt in with args.chunk>0 (1 = a single physics step) + a chunk_step().
  # The loss MUST be per-step-accumulated into self.loss: then d(loss)/d(loss)=1 is a constant seed at
  # every chunk and only the per-step state adjoint (qpos, qvel) crosses chunk boundaries -- no
  # accumulator to checkpoint/carry. (fluid-checkpoint pattern; rewarped's single-step capture.)

  def _set_offset(self, base):
    """Set the global-step base for this chunk (capture-safe time index: a loss kernel can read
    step_offset[0]+i; eager chunk_step just uses the Python t). No-op on the fd path (no step_offset)."""
    so = getattr(self, "step_offset", None)
    if so is not None:
      so.fill_(base)

  def _chunk(self, c):
    """Advance chunk c's `chunk` steps over the fixed segment buffers datas[0..chunk], at global offset
    c*chunk. `chunk_prologue` runs once per chunk (e.g. a model-param reduction that must be re-taped
    every chunk so its adjoint accumulates); `chunk_step` is one physics step + optional per-step loss."""
    base, C = c * self.args.chunk, self.args.chunk
    self.chunk_prologue()
    for i in range(C):
      self.chunk_step(i, base + i)  # DEMO: mjw.step(datas[i], datas[i+1]) + per-step loss at global t

  def bind_ctrl(self, i, ctrl_leaf):
    """Scatter a flat (num_envs*nu,) per-step control leaf into datas[i].ctrl via a kernel (NOT a
    `datas[i].ctrl = leaf` rebind, which drops the last-step ctrl grad in a multi-step chunk -- mjw.step
    copies ctrl into d_out and the rebind aliases). Requires build_datas to set datas[i].ctrl.requires_grad
    = True. Call from chunk_step for schedule demos (dm_suite/go1)."""
    wp.launch(_scatter_ctrl, dim=(self.args.num_envs, self.mjm.nu), inputs=[ctrl_leaf, self.mjm.nu],
              outputs=[self.datas[i].ctrl])

  def _build_ckpt(self):
    """Set up the checkpointed backward: state checkpoints, carried state adjoints, and the three
    per-chunk graphs (forward / recompute / adjoint). Called from __init__ in place of whole-tape capture."""
    C = self.args.chunk
    assert self.nT % C == 0, f"chunk {C} must divide the rollout length nT={self.nT}"
    self.nchunks = self.nT // C
    d0 = self.datas[0]
    self.ckpt_init = (wp.clone(d0.qpos), wp.clone(d0.qvel))  # initial state, restored each forward pass
    self.ckpt_qpos = [wp.zeros_like(d0.qpos) for _ in range(self.nchunks)]  # per-chunk start-state checkpoints
    self.ckpt_qvel = [wp.zeros_like(d0.qvel) for _ in range(self.nchunks)]
    cf = [f for f in self.ckpt_carry if getattr(d0, f, None) is not None and getattr(d0, f).size > 0]
    self._carry = cf  # carried Data fields actually present + non-empty for this model
    self.ckpt_init_carry = {f: wp.clone(getattr(d0, f)) for f in cf}  # initial warmstart etc.
    self.ckpt_carry_bufs = {f: [wp.zeros_like(getattr(d0, f)) for _ in range(self.nchunks)] for f in cf}
    self.ckpt_gq = wp.zeros_like(d0.qpos)  # state adjoint carried across chunk boundaries
    self.ckpt_gv = wp.zeros_like(d0.qvel)
    self.ckpt_loss = wp.zeros_like(self.loss)  # stash the true forward loss (backward recompute re-accumulates it)
    al = getattr(self, "accum_leaf", None)  # model-param leaf whose grad accumulates over ALL chunks
    self._accum_np = np.zeros_like(al.grad.numpy()) if al is not None else None
    self.ckpt_fwd = self.ckpt_rec = self.ckpt_adj = None
    if self.use_graph:
      with mjw.backward_context(self.bc):
        self._chunk(0)  # warm up: JIT + lazy allocations OUTSIDE capture
        tw = wp.Tape()
        with tw:
          self._chunk(0)
        tw.backward()
        wp.synchronize()
        with wp.ScopedCapture() as cap:  # forward: chunk step + per-step loss, no tape
          self._chunk(0)
        self.ckpt_fwd = cap.graph
        with wp.ScopedCapture() as cap:  # recompute: same, records self.tape (built once)
          with self.tape:
            self._chunk(0)
        self.ckpt_rec = cap.graph
        with wp.ScopedCapture() as cap:  # adjoint: propagate from the seeded grads
          self.tape.backward()
        self.ckpt_adj = cap.graph

  def _ckpt_fwd_pass(self):
    """Segment loop: reset datas[0] to the initial state, expand params (set_params), then per chunk
    save the start-state checkpoint, replay the forward graph, carry the last state forward. Leaves
    self.loss = the trajectory-accumulated loss."""
    C = self.args.chunk
    wp.copy(self.datas[0].qpos, self.ckpt_init[0])
    wp.copy(self.datas[0].qvel, self.ckpt_init[1])
    for f in self._carry:
      wp.copy(getattr(self.datas[0], f), self.ckpt_init_carry[f])
    self.set_params()  # DEMO: expand params into datas[0] (leaf) / model field, in place
    self.loss.zero_()
    for c in range(self.nchunks):
      self._set_offset(c * self.args.chunk)
      wp.copy(self.ckpt_qpos[c], self.datas[0].qpos)
      wp.copy(self.ckpt_qvel[c], self.datas[0].qvel)
      for f in self._carry:
        wp.copy(self.ckpt_carry_bufs[f][c], getattr(self.datas[0], f))  # checkpoint the carried state (warmstart)
      if self.ckpt_fwd:
        wp.capture_launch(self.ckpt_fwd)
      else:
        with mjw.backward_context(self.bc):
          self._chunk(c)
      wp.copy(self.datas[0].qpos, self.datas[C].qpos)
      wp.copy(self.datas[0].qvel, self.datas[C].qvel)
      for f in self._carry:
        wp.copy(getattr(self.datas[0], f), getattr(self.datas[C], f))  # carry warmstart etc. across the boundary

  def _ckpt_bwd_pass(self, seed=None):
    """Reverse segment loop: restore each chunk's checkpoint, recompute its forward on the tape, seed
    loss.grad=1 + the carried datas[chunk] state adjoint, replay the adjoint, carry datas[0]'s adjoint to
    the earlier chunk. datas[0]'s grad after chunk 0 is the state-0 gradient (read_grad reads it). `seed`
    (from a terminal loss) initializes the LAST chunk's incoming state adjoint; `truncated` DETACHES the
    carry at every boundary (TBPTT); a model-param `accum_leaf` has its grad summed over all chunks (it
    is used every step, so its gradient accumulates and must survive the per-chunk tape.zero())."""
    C = self.args.chunk
    if seed is None:
      self.ckpt_gq.zero_()  # per-step-only loss: no direct dependence on state beyond the trajectory end
      self.ckpt_gv.zero_()
    else:
      wp.copy(self.ckpt_gq, seed[0])  # terminal-loss adjoint seeds the last chunk's incoming state grad
      wp.copy(self.ckpt_gv, seed[1])
    al = getattr(self, "accum_leaf", None)
    if al is not None:
      self._accum_np.fill(0.0)
    for c in range(self.nchunks - 1, -1, -1):
      self._set_offset(c * C)
      wp.copy(self.datas[0].qpos, self.ckpt_qpos[c])
      wp.copy(self.datas[0].qvel, self.ckpt_qvel[c])
      for f in self._carry:
        wp.copy(getattr(self.datas[0], f), self.ckpt_carry_bufs[f][c])  # restore warmstart so recompute == forward
      if self.ckpt_rec:  # recompute forward (regenerate intermediates + re-run per-step loss on the tape)
        wp.capture_launch(self.ckpt_rec)
      else:
        self.tape = wp.Tape()
        with mjw.backward_context(self.bc):
          with self.tape:
            self._chunk(c)
      self.tape.zero()
      if al is not None:
        al.grad.zero_()  # tape.zero() may miss a MODEL-field leaf -> zero it so += gets THIS chunk's grad only
      self.loss.grad.fill_(1.0)  # per-step loss -> constant seed d(loss)/d(loss)=1 (harmless if loss ∉ chunk tape)
      wp.copy(self.datas[C].qpos.grad, self.ckpt_gq)  # seed AFTER recompute: incoming state adjoint
      wp.copy(self.datas[C].qvel.grad, self.ckpt_gv)
      if self.ckpt_adj:
        wp.capture_launch(self.ckpt_adj)
      else:
        with mjw.backward_context(self.bc):
          self.tape.backward()
      if al is not None:
        self._accum_np += al.grad.numpy()  # accumulate this chunk's model-param grad (survives tape.zero())
      if self.truncated:
        self.ckpt_gq.zero_()  # truncated BPTT: drop the cross-chunk state adjoint (detach)
        self.ckpt_gv.zero_()
      else:
        wp.copy(self.ckpt_gq, self.datas[0].qpos.grad)  # carry state adjoint to the earlier chunk
        wp.copy(self.ckpt_gv, self.datas[0].qvel.grad)
    if al is not None:
      self.accum_grad = self._accum_np.copy()  # model-param demos read this in read_grad()

  # ---- gradient dispatch (harness) ----

  def analytic_grad(self):
    """d(loss)/d(params) per env via the checkpointed taped backward. backward() refreshes datas[0]
    from self.params (via set_params inside the forward pass), so read the leaf grad straight after."""
    self.backward()
    return self.read_grad()

  def grad(self):
    """(num_envs, k) gradient for the current params: 'fd' = per-env central diff, else analytic."""
    if self.args.grad == "fd":
      return np.stack([self.fd_grad(self.params[e]) for e in range(self.args.num_envs)])
    return self.analytic_grad()

  # ---- optimize loop (harness) ----

  def rollout_all(self):
    """Per-env MuJoCo-C rollout of the current params (for the video + display loss). Returns stacked
    (trajs[E, ...], qposs[E, ...], losses[E]); the demo's rollout_env returns one env's (traj, qpos, loss)."""
    outs = [self.rollout_env(self.params[e]) for e in range(self.args.num_envs)]
    trajs = np.array([o[0] for o in outs])
    qposs = np.array([o[1] for o in outs])
    losses = np.array([o[2] for o in outs])
    return trajs, qposs, losses

  def optimize(self):
    """Parallel gradient descent: each iter, MuJoCo-C rollout (video/loss) + one batched grad call +
    per-env host update `params -= lr * g`. Returns (history, best) with best = min mean-loss iter."""
    a = self.args
    history = []
    for it in range(a.steps):
      trajs, qposs, losses = self.rollout_all()
      g = self.grad()
      history.append(self.record(it, losses, trajs, qposs))
      if it % self.log_every == 0:
        print(self.progress(it, losses, g))
      self.params = self.clip_params(self.params - a.lr * g)
    best = int(np.argmin([h["losses"].mean() for h in history]))
    print(self.summary(history, best))
    return history, best

  def optimize_adam(self, betas=(0.9, 0.999)):
    """Style-D optimize (slide/dominos/dm_suite/...): a `wp.array` param leaf `self.param` (built in
    build_datas) that flows into the rollout via a kernel INSIDE the tape, so `tape.backward` lands the
    grad on `self.param.grad`; `warp.optim.Adam` updates it IN PLACE. The video/loss come from the
    taped sim qpos (analytic) or the demo's fd_step (fd). A demo picks this by overriding
    `optimize(self): return self.optimize_adam(betas=...)` and implementing `record(it, L, qpos, pnp)
    -> {"loss": ...}`, `progress(it, L, g, pnp)`, `summary`, and `sim_qpos`/`fd_step`/`project` as
    needed. Grads are read to numpy for a NaN-guard then reinjected (float32 round-trip is identity)."""
    a = self.args
    if not hasattr(self, "param"):  # fd path builds no datas -> create the Adam leaf from the host init
      self.param = wp.array(np.asarray(self.params, dtype=np.float32), dtype=float, requires_grad=True)
    self.opt = warp.optim.Adam([self.param], lr=a.lr, betas=betas)
    history, nan_warned = [], False
    for it in range(a.steps):
      pnp = self.param.numpy().astype(np.float64).copy()
      if a.grad == "analytic":
        self.backward()  # checkpointed forward+backward -> self.param.grad (or accum_grad for model leaves)
        g = np.asarray(self.read_grad(), dtype=np.float64)  # scatter -> param.grad; model -> projected accum_grad
        loss = float(self.loss.numpy()[0])
        qpos = self.sim_qpos()
      else:
        g, loss, qpos = self.fd_step(pnp)  # DEMO: separate MuJoCo-C rollout + fd grad
      if not np.isfinite(g).all():
        if not nan_warned:
          print(f"  WARNING: {a.grad} gradient is NaN/Inf (iter {it}) -- zeroing; optimizer will stall.")
          nan_warned = True
        g = np.nan_to_num(g)
      rec = self.record(it, loss, qpos, pnp, g)
      history.append(rec)
      if it % self.log_every == 0:
        print(self.progress(rec, g))
      self.param.grad = wp.array(g.astype(np.float32), dtype=float)
      self.opt.step([self.param.grad])
      self.project()  # optional in-place projection of self.param (default: no-op)
    best = min(range(len(history)), key=lambda k: history[k]["loss"])
    print(self.summary(history, best))
    return history, best

  def sim_qpos(self):
    """Style-D video source: the taped per-step qpos trajectory (num_envs, len(datas), nq), read from
    the preallocated datas after a backward -- used when the differentiable sim IS the rollout."""
    return np.transpose(np.array([d.qpos.numpy() for d in self.datas]), (1, 0, 2))

  def fd_step(self, pnp):
    """Style-D fd path: return (grad, loss, qpos) from a separate MuJoCo-C rollout (demos with fd)."""
    raise NotImplementedError

  def project(self):
    """Optional post-Adam projection of self.param in place (e.g. clamp/clip). Default: no-op."""

  def clip_params(self, p):
    """Optional projection of the host params after a GD step (e.g. clamp friction mu >= 0). Default: id."""
    return p

  def progress(self, it, losses, g):
    """One-line per-iter status (override for demo-specific fields)."""
    return f"  [{it:4d}] mean_loss={losses.mean():.5f} best={losses.min():.5f}"

  def summary(self, history, best):
    """Final one-line result (override for a demo-specific tag)."""
    ml = [h["losses"].mean() for h in history]
    tag = "analytic (adjoint)" if self.args.grad == "analytic" else "finite diff"
    return (
      f"[{type(self).__name__} / {tag}] {self.args.num_envs} envs: "
      f"mean_loss {ml[0]:.5f} -> best {ml[best]:.5f} (iter {best})"
    )

  def default_out(self):
    """Default output mp4 path (override); used when --out is unset."""
    return os.path.join(os.path.dirname(__file__), "reports", "assets", f"{type(self).__name__}.mp4")

  # ---- USD export (harness) ----

  def write_usd_trajectory(self, viz_model, frames, out_dir, name="traj", fps=30, width=1024, height=768):
    """Play per-frame qpos `frames` on `viz_model` and export a Blender-clean ANIMATED USD to
    <out_dir>/usdpkg/frames/<name>.usdc (returns the path). The `export_usd` hook builds the viz model
    + trajectory and calls this; it's demo-agnostic. Fixes the MuJoCo USDExporter for a clean Blender
    import: metersPerUnit=1 (else Blender's unit conversion shrinks the scene 100x), and per prim drop
    the exporter's valueless xformOp:scale while KEEPING the time-sampled xformOp:transform (else
    Blender's merge_parent_xform collapse reads the transform as identity and dumps every geom at the
    origin). Render it with merge_parent_xform=False + delete the imported MuJoCo lights (see cradle_render_blender.py)."""
    import mujoco
    from mujoco.usd.exporter import USDExporter
    from pxr import UsdGeom  # lazy: demo.py must import in envs without pxr (e.g. the bpy render venv)

    vd = mujoco.MjData(viz_model)
    exp = USDExporter(model=viz_model, height=height, width=width, output_directory="usdpkg",
                      output_directory_root=out_dir, verbose=False)
    for q in frames:
      vd.qpos[:] = q
      mujoco.mj_forward(viz_model, vd)
      exp.update_scene(vd)  # time-samples every geom's pose at this frame
    UsdGeom.SetStageMetersPerUnit(exp.stage, 1.0)
    exp.stage.SetTimeCodesPerSecond(fps)
    exp.stage.SetStartTimeCode(0)
    exp.stage.SetEndTimeCode(max(0, len(frames) - 1))
    for prim in exp.stage.Traverse():
      if prim.IsA(UsdGeom.Xformable):
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        keep = [op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTransform]
        if keep and len(keep) != len(ops):
          xf.SetXformOpOrder(keep)  # drop the valueless scale op; keep the time-sampled transform
      if prim.IsA(UsdGeom.Imageable):
        va = UsdGeom.Imageable(prim).GetVisibilityAttr()
        if va and va.GetNumTimeSamples() > 0:
          va.Clear()
          va.Set(UsdGeom.Tokens.inherited)
    out = os.path.join(out_dir, "usdpkg", "frames", f"{name}.usdc")
    exp.stage.Export(out)
    print(f"[usd] wrote {len(frames)}-frame trajectory -> {out}")
    return out

  def _usd_grid_offsets(self, n, cols, xpitch, ypitch, center_x=0.0):
    """Per-env world offset (n, 3) on a `cols`-wide grid with independent x/y pitch (scenes are often
    much wider in x than deep in y), centered on 0. `center_x` shifts every lane so a scene's visual
    centre (not its origin) lands on the lane centre -- then the render frames on x=0 with no mesh math."""
    rows = int(np.ceil(n / cols))
    off = np.zeros((n, 3))
    for e in range(n):
      r, c = divmod(e, cols)
      off[e, 0] = (c - (cols - 1) / 2.0) * xpitch + center_x
      off[e, 1] = (r - (rows - 1) / 2.0) * ypitch
    return off

  def export_field(self, proto_model, frames, offsets, out_dir, name="traj", fps=30, frame_iters=None,
                   ribbons=None, opt_label=None, env_frames=None, env_ribbons=None, env_protos=None):
    """Blender field export (compute side; no bpy). Animate `frames` (per-frame qpos) on the single-
    instance viz `proto_model` -> a clean proto USD, then a tiny INSTANCED-field USD: one Xform per
    `offsets` row, each an instanceable reference to a proto, so Blender syncs the prototype(s) once.
    `env_frames` (optional): a LIST of K DISTINCT per-env frame-sequences -> a DIVERSE field where each grid
    cell shows one of the K trajectories, spread (deterministic-random) across the grid. K instanceable protos
    are written (same base geometry, K distinct joint trajectories); lanes sharing a proto share its prototype,
    so Blender syncs K prototypes (not N) -- the geometry duplication is only K-fold, not per-cell. Falls back
    to the single (all-identical) field when env_frames is None. Also writes the banner sidecar (`frame_iters`
    = frame->optimization-iteration map; `opt_label` = WHAT the demo optimizes). `ribbons`/`env_ribbons`
    (optional) bake per-frame GROWING trajectory curves into each proto. Returns the field USD path."""
    if env_frames is None:
      env_frames, env_ribbons = [frames], ([ribbons] if ribbons else None)
    k = len(env_frames)
    protos = env_protos if env_protos is not None else [proto_model] * k  # per-env geometry (e.g. slide's target rings) or shared
    proto_paths = []
    for e, ef in enumerate(env_frames):
      pnm = name + "_proto" if k == 1 else f"{name}_proto{e:03d}"
      p = self.write_usd_trajectory(protos[e], ef, out_dir, name=pnm, fps=fps)
      if env_ribbons and env_ribbons[e]:
        self._add_ribbons(p, env_ribbons[e])
      proto_paths.append(p)
    nf = len(env_frames[0])
    out = os.path.join(out_dir, "usdpkg", "frames", f"{name}.usdc")
    self._write_instanced_field(proto_paths, offsets, nf, fps, out)
    self.write_banner(out, nf, fps, frame_iters, opt_label)
    print(f"[usd] field: {k} distinct proto(s) tiled across {len(offsets)} cells")
    return out

  def _add_ribbons(self, proto_path, ribbons):
    """Bake GROWING trajectory ribbons into the proto USD as time-sampled BasisCurves (so they instance
    per-lane with the proto). Each ribbon = {"pts":(NF,3), "iters":(NF,), "rgb":(r,g,b), "width":w}; at
    frame f the curve spans the current iteration's points `pts[iter_start:f+1]` (resets when `iters`
    changes), so each lane draws the trajectory accumulating within the shown iteration."""
    from pxr import Gf, Usd, UsdGeom, Vt

    stage = Usd.Stage.Open(proto_path)
    for i, rib in enumerate(ribbons):
      pts, iters, w = rib["pts"], list(rib["iters"]), float(rib["width"])
      colors = rib.get("colors")   # optional per-frame rgb (e.g. a Bourke loss map) -> per-vertex displayColor
      rgb = rib.get("rgb", (0.6, 0.6, 0.6))
      nf = len(pts)
      starts = [0] * nf
      for f in range(1, nf):
        starts[f] = f if iters[f] != iters[f - 1] else starts[f - 1]
      curve = UsdGeom.BasisCurves.Define(stage, f"/World/ribbon_{i}")
      curve.CreateTypeAttr(UsdGeom.Tokens.linear)
      curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
      pa, vc, wa = curve.CreatePointsAttr(), curve.CreateCurveVertexCountsAttr(), curve.CreateWidthsAttr()
      dca = curve.CreateDisplayColorAttr()
      if colors is not None:  # recolor per frame -> a PER-VERTEX (time-sampled) primvar the render reads
        curve.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.vertex)
      else:
        dca.Set([Gf.Vec3f(*rgb)])
      for f in range(nf):
        seg = pts[starts[f]: f + 1]
        if len(seg) < 2:  # a linear curve needs >= 2 verts; pad with a duplicate first point
          seg = [pts[starts[f]], pts[starts[f]]]
        t = Usd.TimeCode(f)
        pa.Set(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in seg]), t)
        vc.Set(Vt.IntArray([len(seg)]), t)
        wa.Set(Vt.FloatArray([w] * len(seg)), t)
        if colors is not None:
          c = colors[f]
          dca.Set(Vt.Vec3fArray([Gf.Vec3f(float(c[0]), float(c[1]), float(c[2]))] * len(seg)), t)
    stage.GetRootLayer().Save()
    print(f"[usd] baked {len(ribbons)} ribbon(s) into {os.path.basename(proto_path)}")

  def _clean_robot_spec(self, xml, from_string=False, drop_geom=("floor", "ground")):
    """A single robot MjSpec cleaned for viz: floor/ground geoms + lights deleted, unnamed visual geoms
    named by mesh. Shared by strip_to_proto (one robot) and multi_proto (N prefixed copies)."""
    import mujoco

    spec = mujoco.MjSpec.from_string(xml) if from_string else mujoco.MjSpec.from_file(xml)
    for g in [g for g in spec.geoms
              if g.type == mujoco.mjtGeom.mjGEOM_PLANE or any(d in (g.name or "").lower() for d in drop_geom)]:
      spec.delete(g)
    seen = {g.name for g in spec.geoms if g.name}
    for g in spec.geoms:
      if not g.name:  # name unnamed viz geoms so a prefix/regex can target them
        base = (getattr(g, "meshname", "") or "geom").replace(".", "_")
        nm, k = base, 1
        while nm in seen:
          nm, k = f"{base}_{k}", k + 1
        g.name = nm
        seen.add(nm)
    for lt in list(spec.lights):
      spec.delete(lt)
    return spec

  def multi_proto(self, xml, copies, from_string=False, drop_geom=("floor", "ground")):
    """Compile a proto with N prefixed copies of a robot into ONE model, for overlay/side-by-side
    reference renders (e.g. a solid `sim_` robot + a translucent `ghost_`/orange `gt_` reference). `copies`
    = [(prefix, (x,y,z)), ...]; each copy is a cleaned robot (strip floor/lights, name geoms) attached at
    its offset with its prefix, so the render can target `sim_*` vs `ghost_*`/`gt_*` by name. qpos is
    copy-major ([copy0 nq, copy1 nq, ...]) -> concatenate the per-copy trajectories in the same order."""
    import mujoco

    world = mujoco.MjSpec()
    for prefix, off in copies:
      r = self._clean_robot_spec(xml, from_string=from_string, drop_geom=drop_geom)
      fr = world.worldbody.add_frame()
      fr.pos = [float(off[0]), float(off[1]), float(off[2])]
      world.attach(r, prefix=prefix, frame=fr)
    world.visual.global_.offwidth = max(world.visual.global_.offwidth, 1280)
    world.visual.global_.offheight = max(world.visual.global_.offheight, 960)
    return world.compile()

  def strip_to_proto(self, xml, from_string=False, drop_geom=("floor", "ground"), name_geoms=True):
    """Compile a scene XML into a clean single-instance viz proto MjModel for the field render: delete
    the floor/ground worldbody geoms (any PLANE, or a name containing `drop_geom`) + ALL lights. The
    field instances one proto per lane, so a per-lane floor would z-fight and blow up the framing bbox,
    and the exporter's lights would blast the render white -- the render lib adds a single studio
    floor + rig. Deleting geoms/lights leaves every body/joint, so nq/qpos layout is UNCHANGED (the sim
    rollout's qpos frames stay valid).

    Menagerie robots leave their VISUAL mesh geoms UNNAMED (the exporter then emits them as
    `Mesh_Xform_None_id{N}` -> the material regex can't target them), so `name_geoms` fills each unnamed
    kept geom with its mesh name (deduped), giving the render side a stable `<mesh>`-based name to match."""
    import mujoco

    spec = mujoco.MjSpec.from_string(xml) if from_string else mujoco.MjSpec.from_file(xml)
    for g in [g for g in spec.geoms
              if g.type == mujoco.mjtGeom.mjGEOM_PLANE or any(d in (g.name or "").lower() for d in drop_geom)]:
      spec.delete(g)
    if name_geoms:
      seen = {g.name for g in spec.geoms if g.name}
      for g in spec.geoms:
        if not g.name:  # name unnamed viz geoms (by mesh) so the material regex can target them
          base = (getattr(g, "meshname", "") or "geom").replace(".", "_")
          nm, k = base, 1
          while nm in seen:
            nm, k = f"{base}_{k}", k + 1
          g.name = nm
          seen.add(nm)
    for lt in list(spec.lights):
      spec.delete(lt)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 1280)  # exporter needs framebuffer
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 960)  # >= write_usd_trajectory size
    return spec.compile()

  def write_banner(self, usd_path, nf, fps, frame_iters=None, opt_label=None):
    """Write the render-side banner sidecar `<usd_stem>_banner.json` = {nf, fps, frame_iters, opt_label}.
    The render lib (render_blender.load_banner) reads it to stamp the native mp4 banner: demo name (big) +
    `<opt_label> · iter N` subtext. `frame_iters` (len nf) maps each frame to the optimization iteration;
    `opt_label` names WHAT is optimized (e.g. "init velocity", "open-loop ctrl")."""
    fi = None if frame_iters is None else [int(v) for v in frame_iters]
    path = os.path.splitext(usd_path)[0] + "_banner.json"
    with open(path, "w") as f:
      json.dump({"nf": int(nf), "fps": int(fps), "frame_iters": fi, "opt_label": opt_label}, f)
    print(f"[usd] wrote banner sidecar -> {path}  (nf={nf}, fps={fps}, opt={opt_label!r}, "
          f"iters={sorted(set(fi)) if fi else None})")
    return path

  def _write_instanced_field(self, proto_paths, offsets, nframes, fps, out_path):
    """Build the instanced field: /World/lane_i is an Xform with the grid-offset translate holding an
    instanceable child that references a prototype's /World. `proto_paths` = one proto (all lanes identical ->
    ONE shared prototype) OR a LIST of K distinct protos, spread deterministic-randomly across the lanes so the
    field shows K different trajectories; lanes sharing a proto still share its prototype (Blender syncs K, not
    N). metersPerUnit=1 (else Blender shrinks 100x). Returns out_path."""
    import numpy as np
    from pxr import Usd, UsdGeom  # lazy: the bpy render venv has no pxr

    if isinstance(proto_paths, str):
      proto_paths = [proto_paths]
    if os.path.exists(out_path):
      os.remove(out_path)
    stage = Usd.Stage.CreateNew(out_path)  # anchored at frames/ so the relative proto reference resolves
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(0, nframes - 1))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    rels = [os.path.relpath(p, os.path.dirname(out_path)) for p in proto_paths]
    k = len(rels)
    if k == 1:
      assign = np.zeros(len(offsets), int)
    elif k == len(offsets):
      assign = np.arange(len(offsets), dtype=int) % k  # UNTILED (protos == lanes): one DISTINCT proto per lane, in
      # order (else rng.integers duplicates/drops protos -> repeated colors + missing envs in a small few-env view)
    else:
      assign = np.random.default_rng(0).integers(0, k, len(offsets))  # tiled: spread K protos across N>K lanes
    for i, (ox, oy, oz) in enumerate(offsets):
      lane = UsdGeom.Xform.Define(stage, f"/World/lane_{i:04d}")
      lane.AddTranslateOp().Set((float(ox), float(oy), float(oz)))
      inst = stage.DefinePrim(f"/World/lane_{i:04d}/Inst")
      inst.GetReferences().AddReference(rels[int(assign[i])], "/World")
      inst.SetInstanceable(True)  # identical refs to a given proto -> that prototype is shared
    stage.GetRootLayer().Save()
    return out_path

  def export_usd(self, history, best):
    """DEMO hook (--export_usd): replay this optimization's `history` (the same trajectory render()
    animates) to a Blender-clean animated USD via self.write_usd_trajectory. Override per demo."""
    raise NotImplementedError

  # ---- DEMO hooks (subclass implements) ----

  def build_model(self):
    """Set self.mjm, self.mjd (a `mujoco.MjModel`/`MjData`, mj_forward'd). Needed for both grad paths."""
    raise NotImplementedError

  def init_params(self):
    """Return the initial params: host numpy (num_envs, k), the optimized DOFs only."""
    raise NotImplementedError

  def build_datas(self):
    """Analytic path only: set self.datas (preallocated list for num_envs) + self.loss
    (`wp.zeros(1, requires_grad=True)`), with datas' requires_grad flags + datas[0] initial leaves."""
    raise NotImplementedError

  def chunk_step(self, i, t):
    """Analytic path (MANDATORY): one physics step over the segment buffers, `mjw.step(self.m,
    datas[i], datas[i+1])`, plus any per-step loss `wp.atomic_add`-ed into self.loss at global step `t`
    (time-dependent losses index a schedule/reference by `t`; eager gets the real global `t`, so
    schedule/time-dep demos run with capturable=False). Time-INDEPENDENT losses ignore `t` and capture
    fine. A pure-terminal demo just does the step here and puts its loss in terminal_loss()."""
    raise NotImplementedError

  def chunk_prologue(self):
    """Analytic path (optional): per-chunk setup run once at the start of each chunk -- on the recompute
    pass it runs INSIDE the chunk tape, so a model-parameter reduction (e.g. theta -> m.body_inertia)
    placed here is re-recorded every chunk and its adjoint accumulates into the param leaf. Default: no-op."""

  # A demo may also define `terminal_loss(self)` (optional): add a terminal term to self.loss from the
  # FINAL state (datas[0] == state_T after the forward pass) and optionally self.param; the harness runs
  # it on a mini-tape to seed the last chunk's state adjoint (+ any direct param grad). Not a base method
  # so `hasattr(self, "terminal_loss")` detects it.

  def set_params(self):
    """Analytic path: inject the current params into the sim IN PLACE -- datas[0].{qpos,qvel} for a state
    leaf (`.assign`), a differentiable scatter of self.param into datas[0] for Style-D (scatter_params),
    or a model field (`m.<field>`) for a sys-id leaf. Run non-taped in the forward pass; re-taped by the
    harness for the Style-D scatter backprop."""
    raise NotImplementedError

  def read_grad(self):
    """Analytic path: read d(loss)/d(the optimized leaf) -> host numpy. State leaf: datas[0].*.grad;
    Style-D scatter: self.param.grad; model leaf: self.accum_grad (the across-chunk sum), projected onto
    the optimized DOFs (e.g. armature scale, friction mu)."""
    raise NotImplementedError

  def rollout_env(self, p):
    """One env's MuJoCo-C rollout at params `p`: return (traj[T+1, ...], qpos[T+1, nq], loss)."""
    raise NotImplementedError

  def fd_grad(self, p):
    """One env's central-difference d(loss)/d(p) over the MuJoCo-C rollout -> (k,)."""
    raise NotImplementedError

  def record(self, it, losses, trajs, qposs):
    """Return this iteration's history dict (must include "it" + "losses"; add the demo's render keys)."""
    raise NotImplementedError

  def render(self, history, best, out):
    """Animate the optimization history to `out` (mp4) or the live viewer; delegate to viz.py."""
    raise NotImplementedError


def run(example_cls, args):
  """Shared main flow: scope the Warp device, then run the demo's benchmark or optimize + render."""
  ctx = wp.ScopedDevice(args.device) if args.device else contextlib.nullcontext()
  with ctx:
    ex = example_cls(args)
    if getattr(args, "benchmark", False):
      ex.benchmark()
      return
    history, best = ex.optimize()
    if getattr(args, "export_usd", False):  # replay THIS optimization's trajectory to an animated USD
      ex.export_usd(history, best)
      return
    if args.no_render:
      return
    out = args.out or os.environ.get("MJW_RENDER_PATH") or ex.default_out()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ex.render(history, best, out)
