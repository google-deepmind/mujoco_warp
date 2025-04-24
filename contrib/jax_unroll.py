# before running this, you may need to install JAX, most likely
# backed by your local cuda install:
#
# pip install --upgrade "jax[cuda12_local]"

import os
import time

import jax
import mujoco
import numpy as np
import warp as wp
from etils import epath
from jax import numpy as jp
from warp.jax_experimental.ffi import jax_callable

import mujoco_warp as mjwarp

os.environ["XLA_FLAGS"] = "--xla_gpu_graph_min_graph_size=1"

NWORLDS = 8192
UNROLL_LENGTH = 1000

wp.clear_kernel_cache()

path = epath.resource_path("mujoco_warp") / "test_data" / "humanoid/humanoid.xml"
mjm = mujoco.MjModel.from_xml_path(path.as_posix())
mjm.opt.iterations = 1
mjm.opt.ls_iterations = 4
mjd = mujoco.MjData(mjm)
# give the system a little kick to ensure we have non-identity rotations
mjd.qvel = np.random.uniform(-0.01, 0.01, mjm.nv)
mujoco.mj_step(mjm, mjd, 3)  # let dynamics get state significantly non-zero
mujoco.mj_forward(mjm, mjd)
m = mjwarp.put_model(mjm)
d = mjwarp.put_data(mjm, mjd, nworld=NWORLDS, nconmax=131012, njmax=131012 * 4)


def warp_step(
  qpos_in: wp.array(dtype=wp.float32, ndim=2),
  qvel_in: wp.array(dtype=wp.float32, ndim=2),
  qpos_out: wp.array(dtype=wp.float32, ndim=2),
  qvel_out: wp.array(dtype=wp.float32, ndim=2),
):
  wp.copy(d.qpos, qpos_in)
  wp.copy(d.qvel, qvel_in)
  mjwarp.step(m, d)
  wp.copy(qpos_out, d.qpos)
  wp.copy(qvel_out, d.qvel)


warp_step_fn = jax_callable(
  warp_step,
  num_outputs=2,
  output_dims={"qpos_out": (NWORLDS, mjm.nq), "qvel_out": (NWORLDS, mjm.nv)},
)

jax_qpos = jp.tile(jp.array(m.qpos0), (8192, 1))
jax_qvel = jp.zeros((8192, m.nv))


def unroll(qpos, qvel):
  def step(carry, _):
    qpos, qvel = carry
    qpos, qvel = warp_step_fn(qpos, qvel)
    return (qpos, qvel), None

  (qpos, qvel), _ = jax.lax.scan(step, (qpos, qvel), length=UNROLL_LENGTH)

  return qpos, qvel


jax_unroll_fn = jax.jit(unroll).lower(jax_qpos, jax_qvel).compile()

# warm up:
jax.block_until_ready(jax_unroll_fn(jax_qpos, jax_qvel))

beg = time.perf_counter()
final_qpos, final_qvel = jax_unroll_fn(jax_qpos, jax_qvel)
jax.block_until_ready((final_qpos, final_qvel))
end = time.perf_counter()

run_time = end - beg

print(f"Total steps per second: {NWORLDS * UNROLL_LENGTH / run_time:,.0f}")
