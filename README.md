<p>
  <a href="https://github.com/google-deepmind/mujoco_warp/actions/workflows/ci.yml?query=branch%3Amain" alt="GitHub Actions">
    <img src="https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_warp/ci.yml?branch=main">
  </a>
  <a href="https://github.com/google-deepmind/mujoco_warp/blob/main/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/google-deepmind/mujoco_warp">
  </a>
</p>

# MuJoCo Warp (MJWarp)

MJWarp is a GPU-optimized version of the [MuJoCo](https://github.com/google-deepmind/mujoco) physics simulator, designed for NVIDIA hardware.

> [!NOTE]
> MJWarp is in Beta and under active development:
> * MJWarp developers will triage and respond to [bug report and feature requests](https://github.com/google-deepmind/mujoco_warp/issues).
> * MJWarp is mostly feature complete but requires performance optimization, documentation, and testing.
> * The intended audience during Beta are physics engine enthusiasts and learning framework integrators.
> * Machine learning / robotics researchers who just want to train policies should wait for the [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) or [Isaac](https://isaac-sim.github.io/IsaacLab/main/index.html)/[Newton](https://github.com/newton-physics/newton) integrations, which are coming soon.

MJWarp uses [NVIDIA Warp](https://github.com/NVIDIA/warp) to circumvent many of the [sharp bits](https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-the-sharp-bits) in [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html#). MJWarp is integrated into both [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and [Newton](https://github.com/newton-physics/newton).

MJWarp is maintained by [Google DeepMind](https://deepmind.google/) and [NVIDIA](https://www.nvidia.com/).

# Installing for development

MuJoCo Warp is currently supported on Windows or Linux on x86-64 architecture (to be expanded to more platforms and architectures soon).

**Linux**
```bash
git clone https://github.com/google-deepmind/mujoco_warp.git
cd mujoco_warp
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install uv
```
**Windows**
(native Python only, not MSYS2 or WSL)

```powershell
git clone https://github.com/google-deepmind/mujoco_warp.git
cd mujoco_warp
python -m venv env
.\env\Scripts\Activate.ps1  # For MSYS2 Python: env\bin\activate
pip install --upgrade pip
pip install uv
```


Then install MJWarp in editable mode for local development:

```
uv pip install -e .[dev,cuda]
```

Now make sure everything is working:

```bash
pytest
```

Should print out something like `XX passed in XX.XXs` at the end!

If you plan to write Warp kernels for MJWarp, please use the `kernel_analyzer` vscode plugin located in `contrib/kernel_analyzer`.
Please see the `README.md` there for details on how to install it and use it.  The same kernel analyzer will be run on any PR
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
| Equality           | All except `FLEX`.                                                                                      |
| Integrator         | All except `IMPLICIT`                                                                                   |
| Cone               | All                                                                                                     |
| Condim             | All                                                                                                     |
| Solver             | All except `PGS`, `noslip`                                                                              |
| Fluid Model        | All                                                                                                     |
| Tendon Wrap        | All                                                                                                     |
| Sensors            | All except `GEOMDIST`, `GEOMNORMAL`, `GEOMFROMTO`, `PLUGIN`, `USER`                                     |
| Flex               | `VERTCOLLIDE`, `ELASTICITY`                                                                             |
| Mass matrix format | Sparse and Dense                                                                                        |
| Jacobian format    | `DENSE` only (row-sparse, no islanding yet)                                                             |

[Differentiability via Warp](https://nvidia.github.io/warp/modules/differentiability.html#differentiability) is not currently
available.

# Viewing simulations

Explore MuJoCo Warp simulations using an interactive viewer:

```bash
mjwarp-viewer benchmark/humanoid/humanoid.xml
```

This will open a window on your local machine that uses the [MuJoCo native visualizer](https://mujoco.readthedocs.io/en/stable/programming/visualization.html).

# Benchmarking

Benchmark as follows:

```bash
mjwarp-testspeed benchmark/humanoid/humanoid.xml
```

To get a full trace of the physics steps (e.g. timings of the subcomponents) run the following:

```bash
mjwarp-testspeed benchmark/humanoid/humanoid.xml --event_trace=True
```

`mjwarp-testspeed` has many configuration options, see ```mjwarp-testspeed --help``` for details.
