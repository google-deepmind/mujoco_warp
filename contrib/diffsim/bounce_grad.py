"""Same bounce optimization video as _scratch_bounce_viz.py, but the gradient comes from
DIFFERENTIABLE mjwarp (adjoint.py via wp.Tape -> tape.backward), NOT finite differences.

Reuses the elastic wall-bounce scene + video machinery from _scratch_bounce_viz; only the
gradient source changes (grad_test._taped_bounce_grad: multi-step wp.Tape over mjw.step).
"""

import os

import mujoco
import numpy as np
import warp as wp

wp.init()
from mujoco_warp._src import adjoint_test as GT  # noqa: E402

import bounce_viz as B  # noqa: E402

OUT_MP4 = os.environ.get("MJW_RENDER_PATH", "/tmp/bounce_optim_grad.mp4")


def optimize_analytic(steps=160, rate=0.05):
  """LOCAL gradient descent driven by the analytic taped grad from adjoint.py."""
  mjm = mujoco.MjModel.from_xml_string(B.bounce_xml())
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  qpos0 = mjd.qpos.copy()
  target = np.array(B.TARGET)
  T = B.T
  qvel = np.array(B.QVEL0, dtype=np.float64)
  history = []
  for it in range(steps):
    xyz, qpos, loss, hf, hw = B.rollout(qvel)  # MuJoCo-C rollout for the trajectory/video
    history.append({"it": it, "qvel": qvel[:3].copy(), "loss": loss, "xyz": xyz, "qpos": qpos})
    # analytic gradient from differentiable mjwarp (wp.Tape over mjw.step + adjoint.py backward)
    _, g = GT._taped_bounce_grad(mjm, mjd, qpos0, qvel, target, T)
    if it % 10 == 0:
      print(f"  [{it:4d}] loss={loss:.4f} floor={hf} wall={hw}  |analytic_g|={np.linalg.norm(g):.3f}")
    qvel[:3] -= rate * g
  best = min(range(len(history)), key=lambda k: history[k]["loss"])
  print(f"[optim-analytic] loss {history[0]['loss']:.3f} -> best {history[best]['loss']:.3f} (iter {best})")
  return history, best


def main():
  history, best = optimize_analytic()
  B.OUT_MP4 = OUT_MP4  # redirect the shared video writer
  B.render_video(history, best, label="ADJOINT")


if __name__ == "__main__":
  main()
