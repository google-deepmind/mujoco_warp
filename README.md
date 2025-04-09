# MuJoCo Warp (MJWarp)

MJWarp is a GPU-optimized version of the [MuJoCo](https://github.com/google-deepmind/mujoco) physics simulator, designed for NVIDIA hardware.

> [!WARNING]
> MJWarp is in its Alpha stage, with many features still missing and limited testing so far.

MJWarp uses [NVIDIA Warp](https://github.com/NVIDIA/warp) to circumvent many of the [sharp bits](https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-the-sharp-bits) in [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html#). Once MJWarp exits Alpha, it will be integrated into both MJX and [Newton](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation).

MJWarp is maintained by [Google Deepmind](https://deepmind.google/) and [NVIDIA](https://www.nvidia.com/).

# Installing for development

```bash
git clone https://github.com/google-deepmind/mujoco_warp.git
cd mujoco_warp
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
```

During early development, MJWarp is on the bleeding edge - you should install Warp nightly:

```bash
pip install warp-lang --pre --upgrade -f https://pypi.nvidia.com/warp-lang/
```

Then install MJWarp in editable mode for local development:

```
pip install -e .
```

Now make sure everything is working:

```bash
pytest
```

Should print out something like `XX passed in XX.XXs` at the end!

# Compatibility

The following features are implemented:

| Category          | Feature                            |
| ----------------- | ---------------------------------- |
| Dynamics          | Forward only                       |
| Transmission      | `JOINT`, `JOINTINPARENT`           |
| Actuator Dynamics | `NONE`                             |
| Actuator Gain     | `FIXED`, `AFFINE`                  |
| Actuator Bias     | `NONE`, `AFFINE`                   |
| Geom              | `PLANE`, `SPHERE`, `CAPSULE`       |
| Constraint        | `LIMIT_JOINT`, `CONTACT_PYRAMIDAL` |
|                   | `CONTACT_ELLIPTIC`                 |
| Equality          | Not yet implemented                |
| Integrator        | `EULER`, `IMPLICITFAST`, `RK4`     |
| Cone              | `PYRAMIDAL`, `ELLIPTIC`            |
| Condim            | (1, 3, 4, 6), (1, 3)               |
| Solver            | `CG`, `NEWTON`                     |
| Fluid Model       | None                               |
| Tendons           | Not yet implemented.               |
| Sensors           | `JOINTPOS`, `JOINTVEL`,            |
|                   | `ACTUATORFRC`                      |

# Benchmarking

Benchmark as follows:

```bash
mjwarp-testspeed --function=step --mjcf=test_data/humanoid/humanoid.xml --batch_size=8192
```

To get a full trace of the physics steps (e.g. timings of the subcomponents) run the following:

```bash
mjwarp-testspeed --function=step --mjcf=test_data/humanoid/humanoid.xml --batch_size=8192 --event_trace=True
```

