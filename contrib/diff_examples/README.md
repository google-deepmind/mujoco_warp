# Differentiable stepping examples

These examples show that MuJoCo Warp reverse-mode gradients are usable for optimization, not only finite-difference checks.

- `control_trajopt.py` optimizes a 60-step control sequence for a two-link pendulum. The fresh verification run reduced final-state loss by 9.57e8 and reached the target within 3.77e-5 rad.
- `sysid_friction.py` recovered an unknown viscous-friction coefficient to 1.086% relative error in the fresh verification run using one-step prediction error.
- `contact_hopper_torch.py` validates selected action derivatives against finite differences and then trains a short-horizon contact-rich Hopper policy through 32 physics substeps. The rollout starts with two foot contacts and uses pyramidal friction; the learned policy can subsequently leave the ground. It composes one isolated Warp tape per substep through `torch.autograd.Function`, matching the intended SHAC-style integration boundary.

Run the Warp-only examples with the project environment:

```bash
uv run python contrib/diff_examples/control_trajopt.py
uv run python contrib/diff_examples/sysid_friction.py
```

The Hopper example additionally requires a CUDA-enabled PyTorch installation:

```bash
python contrib/diff_examples/contact_hopper_torch.py \
  --output viz_out/contact_hopper_control_run.npz
```

The renderers generate MP4 evidence locally; binary media is intentionally not versioned in this repository. For example:

```bash
MUJOCO_GL=egl uv run python contrib/diff_examples/render_contact_hopper.py \
  viz_out/contact_hopper_control_run.npz viz_out/contact_hopper_control.mp4
uv run python contrib/diff_examples/render_gradient_validation.py \
  contrib/diff_examples/gradient_validation.json viz_out/gradient_validation.mp4
```

Scope:

- Constraint gradients require the Newton solver.
- The validated friction-contact path uses the default pyramidal cone. Elliptic cones require a coupled per-contact Hessian block and are not claimed.
- Scalar active-limit VJPs are currently validated for slide and hinge joints. Ball-joint limits, tendon limits, equality constraints, and independent friction constraints remain outside the demonstrated coverage.
- The Hopper integration intentionally uses one isolated tape per physics substep. This avoids retaining one monolithic Warp tape across the full policy rollout and gives a clean PyTorch autograd node for each substep.
