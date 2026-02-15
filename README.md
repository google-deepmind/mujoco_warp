Disclaimer and Personal Statements of Tech Transfers from USA to PRC!

U.S. EAR restricts AI exports (e.g., model weights >10^25 FLOP, ECCN 4E091) to PRC/Entity List.
Check license needs: bis.doc.gov/ear. Report issues to compliance.

Even if not controlled, I protest unauthorized sharing—protect U.S. security, in accordance with my rights to 1st Amendment.

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_warp/ci.yml?branch=main)](https://github.com/google-deepmind/mujoco_warp/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://readthedocs.org/projects/mujoco/badge/?version=latest)](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)
[![License](https://img.shields.io/github/license/google-deepmind/mujoco_warp)](https://github.com/google-deepmind/mujoco_warp/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://google-deepmind.github.io/mujoco_warp/nightly/)

# MuJoCo Warp (MJWarp)

MJWarp is a GPU-optimized version of the [MuJoCo](https://github.com/google-deepmind/mujoco) physics simulator, designed for NVIDIA hardware.

MJWarp uses [NVIDIA Warp](https://github.com/NVIDIA/warp) to circumvent many of the [sharp bits](https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-the-sharp-bits) in [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html#). MJWarp is integrated into both [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [Newton](https://github.com/newton-physics/newton).

MJWarp is maintained by [Google DeepMind](https://deepmind.google/) and [NVIDIA](https://www.nvidia.com/).

# Getting started

There are a few ways to jump into using MuJoCo Warp:

* For a quick overview of MJWarp's API and design, please see [our colab that introduces the basics](https://colab.research.google.com/github/google-deepmind/mujoco_warp/blob/main/notebooks/tutorial.ipynb).
* For more details and advanced topics on using MJWarp, see the [MuJoCo Warp documentation](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html).

If you would like to train robot policies using MJWarp, consider using a robotics research toolkit that integrates it:

* [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) integrates MJWarp via [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
* [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/feature/newton) integrates MJWarp via [Newton](https://github.com/newton-physics/newton)
* [mjlab](https://github.com/mujocolab/mjlab) integrates MJWarp directly

# Installing

**From PyPI:**

```bash
pip install mujoco-warp
```

**From source:**

```bash
git clone https://github.com/google-deepmind/mujoco_warp.git
cd mujoco_warp
uv sync --all-extras
```

To make sure everything is working:

```bash
uv run pytest -n 8
```

If you plan to write Warp kernels for MJWarp, please use the `kernel_analyzer` vscode plugin located in [`contrib/kernel_analyzer`](https://github.com/google-deepmind/mujoco_warp/tree/main/contrib/kernel_analyzer).
Please see the [README](https://github.com/google-deepmind/mujoco_warp/blob/main/contrib/kernel_analyzer/README.md) there for details on how to install it and use it.  The same kernel analyzer will be run on any PR
you open, so it's important to fix any issues it reports.

# Compatibility

The following features are implemented:

| Category           | Feature                                                                                                 |
| ------------------ | --------------------------------------------------------------------------------------------------------|
| Dynamics           | Forward, Inverse                                                                                        |
| Transmission       | All                                                                                                     |
| Actuator Dynamics  | All except `USER`                                                                                       |
| Actuator Gain      | All except `USER`                                                                                       |
| Actuator Bias      | All except `USER`                                                                                       |
| Geom               | All                                                                                                     |
| Constraint         | All                                                                                                     |
| Equality           | All                                                                                                     |
| Integrator         | All except `IMPLICIT`                                                                                   |
| Cone               | All                                                                                                     |
| Condim             | All                                                                                                     |
| Solver             | All except `PGS`, `noslip`                                                                              |
| Fluid Model        | All                                                                                                     |
| Tendon Wrap        | All                                                                                                     |
| Sensors            | All except `GEOMDIST`, `GEOMNORMAL`, and `GEOMFROMTO` with `BOX`-`BOX`; `PLUGIN`, `USER`                |
| Flex               | `VERTCOLLIDE`, `ELASTICITY`                                                                             |
| Mass matrix format | Sparse and Dense                                                                                        |
| Jacobian format    | `DENSE` only (row-sparse, no islanding yet)                                                             |

[Differentiability via Warp](https://nvidia.github.io/warp/user_guide/differentiability.html) is not currently
available.

# Viewing simulations

Explore MuJoCo Warp simulations using an interactive viewer:

```bash
mjwarp-viewer benchmarks/humanoid/humanoid.xml
```

This will open a window on your local machine that uses the [MuJoCo native visualizer](https://mujoco.readthedocs.io/en/stable/programming/visualization.html).

# Batch Rendering

MJWarp includes a **high-throughput** GPU batch renderer designed for simultaneous rendering of cameras across many parallel simulation worlds. The renderer uses ray-tracing to render MuJoCo primitives using Warp's BVH API.

Key capabilities:
- Mesh rendering
- Texture support
- Heightfield rendering
- Flex deformable rendering
- Heterogeneous multi-camera support (different resolutions/FOV/intrinsics for each camera)
- Lighting and shadow support

# Benchmarking

Benchmark as follows:

```bash
mjwarp-testspeed benchmarks/humanoid/humanoid.xml
```

To get a full trace of the physics steps (e.g. timings of the subcomponents) run the following:

```bash
mjwarp-testspeed benchmarks/humanoid/humanoid.xml --event_trace=True
```

`mjwarp-testspeed` has many configuration options, see ```mjwarp-testspeed --help``` for details.

Benchmark rendering with:

```bash
mjwarp-testspeed benchmarks/primitives.xml --function=render
```
