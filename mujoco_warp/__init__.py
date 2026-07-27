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

"""Public API for MJWarp."""

from importlib import metadata

try:
  __version__ = metadata.version("mujoco_warp")
except metadata.PackageNotFoundError:
  __version__ = "unknown"

# isort: off
from mujoco_warp._src.forward import step as step
from mujoco_warp._src.types import Model as Model
from mujoco_warp._src.types import Data as Data
# isort: on


from mujoco_warp._src.bvh import refit_bvh as refit_bvh
from mujoco_warp._src.collision_driver import collision as collision
from mujoco_warp._src.collision_driver import nxn_broadphase as nxn_broadphase
from mujoco_warp._src.collision_driver import sap_broadphase as sap_broadphase
from mujoco_warp._src.collision_primitive import primitive_narrowphase as primitive_narrowphase
from mujoco_warp._src.collision_sdf import sdf_narrowphase as sdf_narrowphase
from mujoco_warp._src.constraint import make_constraint as make_constraint
from mujoco_warp._src.derivative import deriv_smooth_vel as deriv_smooth_vel
from mujoco_warp._src.forward import euler as euler
from mujoco_warp._src.forward import forward as forward
from mujoco_warp._src.forward import fwd_acceleration as fwd_acceleration
from mujoco_warp._src.forward import fwd_actuation as fwd_actuation
from mujoco_warp._src.forward import fwd_position as fwd_position
from mujoco_warp._src.forward import fwd_velocity as fwd_velocity
from mujoco_warp._src.forward import implicit as implicit
from mujoco_warp._src.forward import rungekutta4 as rungekutta4
from mujoco_warp._src.forward import step1 as step1
from mujoco_warp._src.forward import step2 as step2
from mujoco_warp._src.history import init_ctrl_history as init_ctrl_history
from mujoco_warp._src.history import init_sensor_history as init_sensor_history
from mujoco_warp._src.history import read_ctrl as read_ctrl
from mujoco_warp._src.history import read_sensor as read_sensor
from mujoco_warp._src.inverse import inverse as inverse
from mujoco_warp._src.io import create_render_context as create_render_context
from mujoco_warp._src.io import get_data_into as get_data_into
from mujoco_warp._src.io import make_data as make_data
from mujoco_warp._src.io import put_data as put_data
from mujoco_warp._src.io import put_model as put_model
from mujoco_warp._src.io import reset_data as reset_data
from mujoco_warp._src.io import set_const as set_const
from mujoco_warp._src.io import set_const_0 as set_const_0
from mujoco_warp._src.io import set_const_fixed as set_const_fixed
from mujoco_warp._src.io import set_length_range as set_length_range
from mujoco_warp._src.island import island as island
from mujoco_warp._src.passive import passive as passive
from mujoco_warp._src.ray import ray as ray
from mujoco_warp._src.ray import rays as rays
from mujoco_warp._src.render import render as render
from mujoco_warp._src.render_util import get_depth as get_depth
from mujoco_warp._src.render_util import get_rgb as get_rgb
from mujoco_warp._src.render_util import get_segmentation as get_segmentation
from mujoco_warp._src.sensor import energy_pos as energy_pos
from mujoco_warp._src.sensor import energy_vel as energy_vel
from mujoco_warp._src.sensor import sensor_acc as sensor_acc
from mujoco_warp._src.sensor import sensor_pos as sensor_pos
from mujoco_warp._src.sensor import sensor_vel as sensor_vel
from mujoco_warp._src.smooth import camlight as camlight
from mujoco_warp._src.smooth import com_pos as com_pos
from mujoco_warp._src.smooth import com_vel as com_vel
from mujoco_warp._src.smooth import crb as crb
from mujoco_warp._src.smooth import factor_m as factor_m
from mujoco_warp._src.smooth import flex as flex
from mujoco_warp._src.smooth import kinematics as kinematics
from mujoco_warp._src.smooth import rne as rne
from mujoco_warp._src.smooth import rne_postconstraint as rne_postconstraint
from mujoco_warp._src.smooth import solve_m as solve_m
from mujoco_warp._src.smooth import subtree_vel as subtree_vel
from mujoco_warp._src.smooth import tendon as tendon
from mujoco_warp._src.smooth import transmission as transmission
from mujoco_warp._src.solver import solve as solve
from mujoco_warp._src.support import contact_force as contact_force
from mujoco_warp._src.support import get_state as get_state
from mujoco_warp._src.support import jac as jac
from mujoco_warp._src.support import mul_m as mul_m
from mujoco_warp._src.support import set_state as set_state
from mujoco_warp._src.support import xfrc_accumulate as xfrc_accumulate
from mujoco_warp._src.types import BiasType as BiasType
from mujoco_warp._src.types import BroadphaseFilter as BroadphaseFilter
from mujoco_warp._src.types import BroadphaseType as BroadphaseType
from mujoco_warp._src.types import Callback as Callback
from mujoco_warp._src.types import ConeType as ConeType
from mujoco_warp._src.types import Constraint as Constraint
from mujoco_warp._src.types import Contact as Contact
from mujoco_warp._src.types import DisableBit as DisableBit
from mujoco_warp._src.types import DynType as DynType
from mujoco_warp._src.types import EnableBit as EnableBit
from mujoco_warp._src.types import GainType as GainType
from mujoco_warp._src.types import GeomType as GeomType
from mujoco_warp._src.types import IntegratorType as IntegratorType
from mujoco_warp._src.types import JointType as JointType
from mujoco_warp._src.types import ObjType as ObjType
from mujoco_warp._src.types import Option as Option
from mujoco_warp._src.types import RenderContext as RenderContext
from mujoco_warp._src.types import SolverType as SolverType
from mujoco_warp._src.types import State as State
from mujoco_warp._src.types import Statistic as Statistic
from mujoco_warp._src.types import TrnType as TrnType


# Hip-aware graph capture helper — works with any Warp version on ROCm.
# On HIP, temporarily enables memory pool (required for ScopedCapture),
# then restores pool state after the graph is captured.
# Usage: with mjw.hip_graph_capture() as cap: mjw.step(m, d)
#        wp.capture_launch(cap.graph)  # replays ~1.7x faster on AMD ROCm
import contextlib as _contextlib

@_contextlib.contextmanager
def hip_graph_capture(model, data, device=None, warmup_steps=3):
    """Context manager for HIP/CUDA graph capture of mujoco_warp physics.

    Handles all pre-capture requirements automatically:

    1. Runs warmup_steps eager steps to trigger all lazy allocations
       (solver context, tendon scratch, RK4 buffers, collision structures).
    2. On HIP/ROCm: enables memory pool before capture (required for
       hipGraph) and restores pool state after.
    3. Wraps wp.ScopedCapture — the caller gets the standard graph
       object and uses wp.capture_launch(cap.graph) to replay.

    Usage::

        model = mjw.put_model(mj_model)
        data  = mjw.put_data(mj_model, mj_data, nworld=256)

        with mjw.hip_graph_capture(model, data) as cap:
            mjw.step(model, data)

        # Replay ~1.7x faster on AMD ROCm, ~2x faster on NVIDIA CUDA
        for _ in range(1000):
            wp.capture_launch(cap.graph)

    Args:
        model: mujoco_warp Model (from put_model).
        data:  mujoco_warp Data  (from put_data).
        device: Warp device to capture on. Defaults to current device.
        warmup_steps: Number of eager steps before capture to trigger all
            lazy buffer allocations. Default 3 is sufficient for most models.
    """
    import warp as _wp
    from mujoco_warp._src import forward as _fwd

    dev = device or _wp.get_device()
    is_hip = getattr(dev, 'is_hip', False)
    pool_was_enabled = _wp.is_mempool_enabled(dev) if (dev.is_cuda or is_hip) else False

    try:
        # Step 1: warmup to trigger all lazy allocations in the step() call graph.
        # This includes solver context, tendon scratch, RK4 buffers, constraint
        # structures, collision temporaries — any wp.zeros/wp.empty inside step().
        # After warmup these are cached on Data and will NOT fire during capture.
        for _ in range(warmup_steps):
            _fwd.step(model, data)

        import torch as _torch
        _torch.cuda.synchronize()

        # Step 2: enable mempool on HIP (required for ScopedCapture to allocate
        # the graph node backing store).
        if is_hip and not pool_was_enabled:
            _wp.set_mempool_enabled(dev, True)

        # Step 3: capture — after warmup, step() should fire zero new allocations.
        with _wp.ScopedCapture() as cap:
            yield cap

    finally:
        # Restore mempool state.
        if is_hip and not pool_was_enabled:
            _wp.set_mempool_enabled(dev, False)
