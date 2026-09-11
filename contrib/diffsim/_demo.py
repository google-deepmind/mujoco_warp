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
"""Shared command-line and optimization loop for differentiable simulation demos."""

import dataclasses
import enum
import math
import types
from dataclasses import dataclass
from dataclasses import field

import _args
import _rollout
import _viz
import mujoco
import numpy as np
import warp as wp
import warp.optim

import mujoco_warp as mjw

mjw.enable_grad()

MENAGERIE_VISUAL = """
<headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
<rgba haze="0.15 0.25 0.35 1"/>
<global azimuth="60" elevation="-20"/>
"""

MENAGERIE_ASSETS = """
<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
<texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
         markrgb="0.8 0.8 0.8" width="300" height="300"/>
<material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
"""

MENAGERIE_LIGHTS = """
<light pos="0 0 1"/>
<light pos="0 -0.2 1" dir="0 0.2 -0.8" directional="true"/>
"""

_INTEGRATORS = {
  "euler": mujoco.mjtIntegrator.mjINT_EULER,
  "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
_CONES = {
  "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
  "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
}
_ENV_SAMPLING = ("random", "linspace")


class DemoType(str, enum.Enum):
  """Differentiable simulation workflow."""

  INIT_VAL = "init_val"
  TRAJ_OPT = "traj_opt"
  SYS_ID = "sys_id"
  MPC = "mpc"


class LossType(str, enum.Enum):
  """Trajectory loss reduction."""

  TERMINAL = "terminal"
  PER_STEP = "per_step"


@dataclass
class Args:
  """Arguments shared by the differentiable simulation demos."""

  # Simulation
  sim_dt: float = field(default=0.01, metadata={"help": "physics timestep in seconds"})
  integrator: str = field(
    default="implicitfast",
    metadata={"help": "physics integrator", "choices": tuple(_INTEGRATORS)},
  )
  cone: str = field(
    default="elliptic",
    metadata={"help": "contact friction cone", "choices": tuple(_CONES)},
  )
  impratio: float = field(
    default=1.0,
    metadata={"help": "friction-to-normal constraint impedance ratio"},
  )

  # Optimization
  ctrl_dt: float | None = field(default=None, metadata={"help": "control interval in seconds"})
  iterations: int = field(default=100, metadata={"help": "optimization iterations"})
  lr: float = field(default=0.01, metadata={"help": "learning rate"})
  optimizer: str = field(
    default="adam",
    metadata={"help": "optimizer", "choices": ("adam", "sgd")},
  )
  adam_betas: tuple[float, float] = field(default=(0.7, 0.95), metadata={"help": "Adam first- and second-moment decay"})
  num_envs: int = field(default=8, metadata={"help": "parallel simulation worlds"})
  env_sampling: str = field(
    default="random",
    metadata={"help": "parallel-environment sampling", "choices": _ENV_SAMPLING},
  )
  seed: int = field(default=0, metadata={"help": "parallel-environment random seed"})
  device: str | None = field(default=None, metadata={"help": "Warp device, for example cuda:0 or cpu"})
  graph: bool = field(default=True, metadata={"help": "capture one forward and backward physics step"})
  tbptt: int | None = field(default=None, metadata={"help": "TBPTT window length"})
  log_every: int = field(default=10, metadata={"help": "print every N optimization iterations"})
  metrics: bool = field(default=False, metadata={"help": "record every iteration for offline paper metrics"})

  # Visualization
  viz: str = field(
    default="mjwarp",
    metadata={"help": "visualization mode", "choices": ("mjwarp", "mujoco", "usd", "none")},
  )
  viz_every: int = field(
    default=10,
    metadata={"help": "optimization iterations between saved visualizations; mjwarp always shows every iteration"},
  )
  viz_stride: int = field(default=4, metadata={"help": "visualize every N simulation frames"})
  viz_env: int = field(default=0, metadata={"help": "parallel world for live and MuJoCo visualization"})
  contact_forces: bool = field(default=False, metadata={"help": "visualize contact-force arrows"})
  fps: int = field(default=30, metadata={"help": "live and MuJoCo visualization frame rate"})
  width: int = field(default=1024, metadata={"help": "render width"})
  height: int = field(default=768, metadata={"help": "render height"})
  output: str | None = field(default=None, metadata={"help": "MP4 path or USD output directory"})


def sample_range(num_envs: int, low, high, sampling: str, seed: int = 0) -> np.ndarray:
  """Samples an environment batch randomly or as an evenly spaced sweep."""
  if num_envs < 1:
    raise ValueError("num_envs must be positive")
  low, high = np.broadcast_arrays(np.asarray(low), np.asarray(high))
  if not np.all(low <= high):
    raise ValueError("low must be less than or equal to high")
  if sampling == "linspace":
    return np.linspace(low, high, num_envs, dtype=np.float32)
  if sampling == "random":
    return np.random.default_rng(seed).uniform(low, high, (num_envs, *low.shape)).astype(np.float32)
  raise ValueError(f"unknown environment sampling: {sampling}")


def control_steps(sim_dt: float, ctrl_dt: float) -> int:
  """Returns the integer number of physics steps in one control interval."""
  if sim_dt <= 0.0 or ctrl_dt <= 0.0:
    raise ValueError("sim_dt and ctrl_dt must be positive")
  steps = round(ctrl_dt / sim_dt)
  if steps < 1 or not math.isclose(ctrl_dt, steps * sim_dt, rel_tol=1.0e-9, abs_tol=1.0e-12):
    raise ValueError(f"ctrl_dt={ctrl_dt:g} must be an integer multiple of sim_dt={sim_dt:g}")
  return steps


def set_sim_params(
  mjm: mujoco.MjModel,
  sim_dt: float | None = None,
  integrator: str | None = None,
  cone: str | None = None,
  impratio: float | None = None,
):
  """Validates and applies shared MuJoCo simulation parameters."""
  if sim_dt is not None:
    if sim_dt <= 0.0:
      raise ValueError("sim_dt must be positive")
    mjm.opt.timestep = sim_dt

  if integrator is not None:
    if integrator not in _INTEGRATORS:
      raise ValueError(f"unknown integrator: {integrator}")
    mjm.opt.integrator = _INTEGRATORS[integrator]

  if cone is not None:
    if cone not in _CONES:
      raise ValueError(f"unknown cone: {cone}")
    mjm.opt.cone = _CONES[cone]

  if impratio is not None:
    if impratio <= 0.0:
      raise ValueError("impratio must be positive")
    mjm.opt.impratio = impratio


def capture(function, enabled=True):
  """Captures one invocation on CUDA and returns ``None`` for eager execution."""
  if not enabled or not wp.get_device().is_cuda:
    return None
  with wp.ScopedCapture() as scoped:
    function()
  return scoped.graph


def launch(function, graph):
  """Runs a captured invocation, or its eager equivalent."""
  if graph is None:
    function()
  else:
    wp.capture_launch(graph)


def graph(function, enabled=True):
  """Returns a callable backed by a CUDA graph when enabled."""
  captured = capture(function, enabled)
  return lambda: launch(function, captured)


def value_and_grad(function, loss, context=None, enabled=True):
  """Builds scalar-value and analytic-gradient callables with a captured forward."""
  tape = None
  forward = function

  def backward():
    assert tape is not None
    tape.zero()
    loss.grad.fill_(1.0)
    if context is None:
      tape.backward()
    else:
      with mjw.backward_context(context):
        tape.backward()

  def value():
    forward()

  def gradient():
    nonlocal tape, forward
    if tape is None:
      tape = wp.Tape()
      with tape:
        function()
      backward()
      wp.synchronize()
      forward = graph(function, enabled)
    else:
      value()
      backward()

  return value, gradient


class Demo:
  """Base class for batched differentiable simulation demos."""

  Args = Args
  name = "diffsim"
  demo_type = DemoType.TRAJ_OPT
  loss_type = LossType.PER_STEP
  model_fields = ()
  trace_body = None
  trace_width = 0.01
  trace_alpha = 0.5
  layout = _viz.Layout()
  camera = _viz.Camera()
  data_kwargs = {}

  def __init__(
    self,
    args: Args,
    mjm: mujoco.MjModel,
    horizon: int,
    eval_horizon: int = 0,
    iterations=None,
    lr=None,
    loss_scale=None,
  ):
    self.args = args
    self.mjm = mjm
    set_sim_params(mjm, args.sim_dt, args.integrator, args.cone, args.impratio)
    self.mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, self.mjd)

    batch_sizes = {name: args.num_envs for name in self.model_fields}
    self.m = mjw.put_model(mjm, batch_sizes=batch_sizes)
    self.datas = [mjw.put_data(mjm, self.mjd, nworld=args.num_envs, **self.data_kwargs) for _ in range(2)]
    for d in self.datas:
      d.qpos.requires_grad = True
      d.qvel.requires_grad = True
    self.horizon = horizon
    if loss_scale is None:
      if self.loss_type == LossType.TERMINAL:
        loss_steps = 1
      elif self.loss_type == LossType.PER_STEP:
        loss_steps = horizon
      else:
        raise NotImplementedError(self.loss_type)
      loss_scale = 1.0 / (args.num_envs * loss_steps)
    self.loss_scale = loss_scale
    if eval_horizon < 0:
      raise ValueError("eval_horizon must be nonnegative")
    self.eval_horizon = eval_horizon
    self._eval_state = None
    if args.tbptt is not None and args.tbptt < 1:
      raise ValueError("tbptt must be positive")
    if len(args.adam_betas) != 2 or any(beta < 0.0 or beta >= 1.0 for beta in args.adam_betas):
      raise ValueError("adam_betas must contain two values in [0, 1)")
    self.iterations = args.iterations if iterations is None else iterations
    self.lr = args.lr if lr is None else lr
    self.step_index = wp.zeros(1, dtype=int)
    # placeholders, set by each demo
    self.params = []
    self.model_params = []
    self.init_qpos = None

  def reset(self):
    """Resets the initial simulation state before a rollout."""
    d = self.datas[0]
    d.qpos.assign(self.init_qpos)
    qvel0 = getattr(self, "qvel0", None)
    if qvel0 is None:
      d.qvel.zero_()
    else:
      d.qvel.assign(qvel0)
    if d.act.size:
      d.act.zero_()
    d.ctrl.zero_()
    d.qacc_warmstart.zero_()
    d.time.zero_()

  def prepare_step(self, d):
    """Applies parameters and controls before one physics step."""

  def step_loss(self, d, d_out):
    """Adds this step's contribution to the trajectory loss."""

  def make_optimizer(self):
    if self.args.optimizer == "sgd":
      return warp.optim.SGD(self.params, lr=self.lr)
    if self.args.optimizer == "adam":
      return warp.optim.Adam(self.params, lr=self.lr, betas=self.args.adam_betas)
    raise NotImplementedError(self.args.optimizer)

  def project(self):
    """Projects optimized parameters after an update."""

  def optimizer_step(self, optimizer, gradients, iteration):
    """Applies one optimizer update."""
    del iteration
    optimizer.step(gradients)

  def trajectory(self):
    """Returns qpos with shape (world, time, nq)."""
    return self.rollout.qpos.numpy().transpose(1, 0, 2)

  def trajectory_state(self):
    """Returns the position, velocity, and control trajectories."""
    return types.SimpleNamespace(
      qpos=self.trajectory(),
      qvel=self.rollout.qvel.numpy().transpose(1, 0, 2),
      ctrl=self.rollout.ctrl.numpy().transpose(1, 0, 2),
    )

  def eval_trajectory(self, trajectory):
    """Continues a fitted trajectory through forward-only evaluation steps."""
    if not self.eval_horizon:
      return trajectory

    d = self.datas[0]
    self.rollout.restore(self.horizon, d)
    if self._eval_state is None:
      self._eval_state = (
        wp.empty((self.eval_horizon, *d.qpos.shape), dtype=float),
        wp.empty((self.eval_horizon, *d.qvel.shape), dtype=float),
        wp.empty((self.eval_horizon, *d.ctrl.shape), dtype=float),
      )
    for offset in range(self.eval_horizon):
      self.step_index.fill_(self.horizon + offset)
      self.prepare_step(d)
      mjw.step(self.m, d)
      wp.copy(self._eval_state[0][offset], d.qpos)
      wp.copy(self._eval_state[1][offset], d.qvel)
      if d.ctrl.size:
        wp.copy(self._eval_state[2][offset], d.ctrl)

    evaluation = [value.numpy().transpose(1, 0, 2) for value in self._eval_state]
    return types.SimpleNamespace(
      qpos=np.concatenate((trajectory.qpos, evaluation[0]), axis=1),
      qvel=np.concatenate((trajectory.qvel, evaluation[1]), axis=1),
      ctrl=np.concatenate((trajectory.ctrl, evaluation[2]), axis=1),
    )

  def viz_trajectory(self, trajectory):
    """Maps simulation qpos into the visualization scene."""
    return trajectory

  def viz_mjm(self):
    return self.mjm

  def viz_model_fields(self, mjm):
    """Returns per-environment MJWarp model fields used only for visualization."""
    fields = {}
    for name, value in vars(self.model_state()).items():
      host = getattr(mjm, name, None)
      if host is not None and value.shape == (self.args.num_envs, *host.shape):
        fields[name] = value
    return fields

  def model_state(self):
    """Returns the per-environment MJWarp model fields."""
    return types.SimpleNamespace(**{name: getattr(self.m, name).numpy() for name in self.model_fields})

  def visualizer(self):
    """Builds a visualizer whose models use the runtime simulation parameters."""
    mjm = self.viz_mjm() if self.args.viz != "none" else self.mjm
    if self.args.viz != "none":
      set_sim_params(
        mjm,
        float(self.mjm.opt.timestep),
        self.args.integrator,
        self.args.cone,
        self.args.impratio,
      )
    return _viz.Visualizer(
      mjm,
      self.camera,
      self.layout,
      self.args,
      self.name,
      final_iteration=self.iterations - 1,
      trace_width=self.trace_width,
      trace_alpha=self.trace_alpha,
      arguments=dataclasses.asdict(self.args),
      settings={
        "integrator": self.args.integrator,
        "cone": self.args.cone,
        "optimizer": self.args.optimizer,
      },
      scene={"bodies": self.mjm.nbody, "dofs": self.mjm.nv, "geoms": self.mjm.ngeom},
      model_fields=self.viz_model_fields(mjm) if self.args.viz != "none" else {},
      metric_mjm=self.mjm,
    )

  def _physics_step(self):
    d, d_out = self.datas
    self.prepare_step(d)
    mjw.step(self.m, d, d_out)
    self.step_loss(d, d_out)

  def forward(self, tape):
    self.reset()
    self.rollout.begin(tape)
    with tape:
      for step in range(self.horizon):
        self.rollout.step(tape, step)
    self.rollout.end(tape)

  def optimize(self):
    if not self.params:
      raise ValueError("Demo.params must contain at least one differentiable array")
    if self.iterations < 1:
      raise ValueError("iterations must be positive")

    self.loss = wp.zeros(1, dtype=float, requires_grad=True)
    self.rollout = _rollout.Rollout(self, self.horizon)
    optimizer = self.make_optimizer()
    self.rollout.prepare(self.args.graph)
    grads = [param.grad for param in self.params]
    history = []
    with self.visualizer() as visualizer:
      for iteration in range(self.iterations):
        tape = wp.Tape()
        self.forward(tape)
        tape.backward(loss=self.loss)
        loss = float(self.loss.numpy()[0])
        record = {"iteration": iteration, "loss": loss}

        grad_norm = float(np.sqrt(sum(np.sum(grad.numpy() ** 2) for grad in grads)))
        if not np.isfinite(grad_norm):
          raise FloatingPointError(f"non-finite gradient at iteration {iteration}")
        record["grad_norm"] = grad_norm
        history.append(record)
        if iteration % self.args.log_every == 0 or iteration == self.iterations - 1:
          print(f"[{self.name}:{iteration:4d}] loss={loss:.6g}  |grad|={grad_norm:.3g}")

        # Capture the current model and data before the optimizer step.
        # Metrics retain every iteration; visualization is attached only at its requested cadence.
        render = visualizer.should_update(iteration)
        if self.args.metrics or render:
          data = self.trajectory_state()
          visualization = None
          if render:
            evaluated = self.eval_trajectory(data)
            visualization = types.SimpleNamespace(
              qpos=self.viz_trajectory(evaluated.qpos),
              qvel=evaluated.qvel,
              ctrl=evaluated.ctrl,
            )
          visualizer.record_iteration(
            iteration,
            loss,
            grad_norm,
            self.model_state() if self.args.metrics else types.SimpleNamespace(),
            data,
            visualization,
            self.trace_body,
          )
        self.optimizer_step(optimizer, grads, iteration)
        self.project()
        tape.zero()
        self.loss.zero_()

    print(f"[{self.name}] loss {history[0]['loss']:.6g} -> {history[-1]['loss']:.6g}")
    return history


def run(demo_cls: type[Demo], argv=None):
  """Parses a demo's Args subclass and runs it on the requested Warp device."""
  args = _args.parse_args_dataclass(demo_cls.Args, argv)
  with wp.ScopedDevice(args.device):
    return demo_cls(args).optimize()
