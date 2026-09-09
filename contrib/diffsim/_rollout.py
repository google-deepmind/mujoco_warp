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
"""One-step gradient checkpointing for differentiable simulation demos."""

import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src import forward


@wp.kernel(enable_backward=False)
def _accumulate_grad(
  # In:
  grad: wp.array[float],
  # Out:
  total_out: wp.array[float],
):
  i = wp.tid()
  total_out[i] += grad[i]


class Rollout:
  """Records checkpointed simulation steps as custom Warp tape operations."""

  def __init__(self, demo, horizon):
    self.demo = demo
    d = demo.datas[0]

    def allocate(array, requires_grad=False):
      return wp.empty((horizon + 1, *array.shape), dtype=array.dtype, requires_grad=requires_grad)

    self.qpos = allocate(d.qpos, requires_grad=True)
    self.qvel = allocate(d.qvel, requires_grad=True)
    self.act = allocate(d.act)
    self.ctrl = allocate(d.ctrl)
    self.qacc_warmstart = allocate(d.qacc_warmstart)
    self.time = allocate(d.time)
    self.loss_value = wp.empty_like(demo.loss, requires_grad=False)
    self.state_grads = (
      wp.zeros_like(d.qpos, requires_grad=False),
      wp.zeros_like(d.qvel, requires_grad=False),
    )
    self.param_grads = []
    self.forward_graph = None
    self.backward_graph = None

  def prepare(self, graph):
    self.demo.bc = mjw.create_backward_context(self.demo.m, self.demo.datas[0])
    self.param_grads = [wp.zeros_like(param, requires_grad=False) for param in self.demo.params]
    self.qpos.grad.zero_()
    self.qvel.grad.zero_()
    if graph and wp.get_device().is_cuda:
      self._capture()

  def save(self, step, d):
    wp.copy(self.qpos[step], d.qpos)
    wp.copy(self.qvel[step], d.qvel)
    if d.act.size:
      wp.copy(self.act[step], d.act)
    if d.ctrl.size:
      wp.copy(self.ctrl[step], d.ctrl)
    wp.copy(self.qacc_warmstart[step], d.qacc_warmstart)
    wp.copy(self.time[step], d.time)

  def restore(self, step, d):
    wp.copy(d.qpos, self.qpos[step])
    wp.copy(d.qvel, self.qvel[step])
    if d.act.size:
      wp.copy(d.act, self.act[step])
    if d.ctrl.size:
      wp.copy(d.ctrl, self.ctrl[step])
    wp.copy(d.qacc_warmstart, self.qacc_warmstart[step])
    wp.copy(d.time, self.time[step])

  def begin(self, tape):
    d, d_out = self.demo.datas
    d.qpos.grad.zero_()
    d.qvel.grad.zero_()
    d_out.qpos.grad.zero_()
    d_out.qvel.grad.zero_()
    self.save(0, d)
    tape.record_func(self._finish_backward, self._grad_arrays())

  def step(self, tape, step):
    runtime = wp._src.context.runtime
    active_tape = runtime.tape
    runtime.tape = None
    try:
      self.demo.step_index.fill_(step)
      if self.forward_graph is None:
        self.demo._physics_step()
        forward._copy_state(self.demo.datas[1], self.demo.datas[0])
      else:
        wp.capture_launch(self.forward_graph)
      self.save(step + 1, self.demo.datas[0])
    finally:
      runtime.tape = active_tape
    tape.record_func(lambda step=step: self._backward(step), self._grad_arrays())

  def end(self, tape):
    tape.record_func(self._begin_backward, self._grad_arrays())

  def _grad_arrays(self):
    return [self.qpos, self.qvel, self.demo.loss, *self.demo.params, *self.demo.model_params]

  def _begin_backward(self):
    wp.copy(self.loss_value, self.demo.loss)
    for grad in self.param_grads:
      grad.zero_()

  def _backward(self, step):
    d, d_out = self.demo.datas
    self.demo.step_index.fill_(step)
    self.restore(step, d)
    tbptt = self.demo.args.tbptt
    if tbptt is not None and step + 1 < self.demo.horizon and (step + 1) % tbptt == 0:
      self.state_grads[0].zero_()
      self.state_grads[1].zero_()
    else:
      wp.copy(self.state_grads[0], self.qpos.grad[step + 1])
      wp.copy(self.state_grads[1], self.qvel.grad[step + 1])
    if self.backward_graph is None:
      self._backward_step()
    else:
      wp.capture_launch(self.backward_graph)
    wp.copy(self.qpos.grad[step], d.qpos.grad)
    wp.copy(self.qvel.grad[step], d.qvel.grad)

  def _backward_step(self):
    _, d_out = self.demo.datas
    tape = wp.Tape()
    with mjw.backward_context(self.demo.bc):
      with tape:
        self.demo._physics_step()
      tape.zero()
      for param in (*self.demo.params, *self.demo.model_params):
        param.grad.zero_()
      self.demo.loss.grad.fill_(1.0)
      wp.copy(d_out.qpos.grad, self.state_grads[0])
      wp.copy(d_out.qvel.grad, self.state_grads[1])
      tape.backward()
    for param, grad in zip(self.demo.params, self.param_grads):
      wp.launch(_accumulate_grad, dim=param.size, inputs=[param.grad], outputs=[grad])
    return tape

  def _finish_backward(self):
    for param, grad in zip(self.demo.params, self.param_grads):
      wp.copy(param.grad, grad)
    wp.copy(self.demo.loss, self.loss_value)

  def _capture(self):
    demo = self.demo
    demo.reset()
    demo.loss.zero_()
    demo.step_index.zero_()
    self.state_grads[0].zero_()
    self.state_grads[1].zero_()
    for grad in self.param_grads:
      grad.zero_()
    self._backward_step()
    wp.synchronize()

    demo.reset()
    demo.loss.zero_()
    demo.step_index.zero_()
    with wp.ScopedCapture() as capture:
      demo._physics_step()
      forward._copy_state(demo.datas[1], demo.datas[0])
    self.forward_graph = capture.graph

    demo.reset()
    demo.loss.zero_()
    demo.step_index.zero_()
    self.state_grads[0].zero_()
    self.state_grads[1].zero_()
    for grad in self.param_grads:
      grad.zero_()
    with wp.ScopedCapture() as capture:
      self._step_tape = self._backward_step()
    self.backward_graph = capture.graph

    demo.reset()
    demo.loss.zero_()
    self.qpos.grad.zero_()
    self.qvel.grad.zero_()
    for grad in self.param_grads:
      grad.zero_()
