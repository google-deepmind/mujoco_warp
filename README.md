# MuJoCo Warp (MJWarp)

MJWarp is a GPU-optimized version of the [MuJoCo](https://github.com/google-deepmind/mujoco) physics simulator, designed for NVIDIA hardware.

> [!WARNING]
> MJWarp is in its Alpha stage, with many features still missing and limited testing so far.

MJWarp uses [NVIDIA Warp](https://github.com/NVIDIA/warp) to circumvent many of the [sharp bits](https://mujoco.readthedocs.io/en/stable/mjx.html#mjx-the-sharp-bits) in [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html#). Once MJWarp exits Alpha, it will be integrated into both MJX and [Newton](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation).

MJWarp is maintained by [Google DeepMind](https://deepmind.google/) and [NVIDIA](https://www.nvidia.com/).

# Installing for development

```bash
git clone https://github.com/google-deepmind/mujoco_warp.git
cd mujoco_warp
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
```

During early development, MJWarp is on the bleeding edge - you should install Warp and MuJoCo nightly:

```bash
pip install warp-lang --pre --upgrade -f https://pypi.nvidia.com/warp-lang/
pip install mujoco --pre --upgrade -f https://py.mujoco.org/
```

Then install MJWarp in editable mode for local development:

```
pip install -e .[dev,cuda]
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

| Category          | Feature                                                                                                  |
| ----------------- | ---------------------------------------------------------------------------------------------------------|
| Dynamics          | Forward, Inverse                                                                                         |
| Transmission      | `JOINT`, `JOINTINPARENT`, `TENDON`, `SITE`                                                               |
| Actuator Dynamics | `NONE`, `INTEGRATOR`, `FILTER`, `FILTEREXACT`, `MUSCLE`                                                  |
| Actuator Gain     | `FIXED`, `AFFINE`, `MUSCLE`                                                                              |
| Actuator Bias     | `NONE`, `AFFINE`, `MUSCLE`                                                                               |
| Geom              | `PLANE`, `SPHERE`, `CAPSULE`, `ELLIPSOID`, `CYLINDER`, `BOX`, `HFIELD`, `MESH`, `SDF`                           |
| Constraint        | `FRICTION JOINT`, `FRICTION TENDON`, `LIMIT_BALL`, `LIMIT_JOINT`, `LIMIT_TENDON`,                        |
|                   | `CONTACT_PYRAMIDAL`, `CONTACT_ELLIPTIC`                                                                  |
| Equality          | `CONNECT`, `WELD`, `JOINT`, `TENDON`                                                                     |
| Integrator        | `EULER`, `IMPLICITFAST`, `RK4`                                                                           |
| Cone              | `PYRAMIDAL`, `ELLIPTIC`                                                                                  |
| Condim            | 1, 3, 4, 6                                                                                               |
| Solver            | `CG`, `NEWTON`                                                                                           |
| Fluid Model       | `BOX`                                                                                                    |
| Tendons           | `FIXED`, `SITE`, `SPHERE`, `CYLINDER`, `PULLEY`                                                          |
| Sensors           | `MAGNETOMETER`, `CAMPROJECTION`, `RANGEFINDER`, `JOINTPOS`, `TENDONPOS`, `ACTUATORPOS`, `BALLQUAT`,      |
|                   | `JOINTLIMITPOS`, `TENDONLIMITPOS`, `FRAMEPOS`, `FRAMEXAXIS`, `FRAMEYAXIS`, `FRAMEZAXIS`, `FRAMEQUAT`,    |
|                   | `SUBTREECOM`, `E_POTENTIAL`, `E_KINETIC`, `CLOCK`, `VELOCIMETER`, `GYRO`, `JOINTVEL`, `TENDONVEL`,       |
|                   | `ACTUATORVEL`, `BALLANGVEL`, `JOINTLIMITVEL`, `TENDONLIMITVEL`, `FRAMELINVEL`, `FRAMEANGVEL`,            |
|                   | `SUBTREELINVEL`, `SUBTREEANGMOM`,`TOUCH`, `ACCELEROMETER`, `FORCE`, `TORQUE`, `ACTUATORFRC`,             |
|                   | `TENDONACTFRC`, `JOINTACTFRC`, `JOINTLIMITFRC`, `TENDONLIMITFRC`, `FRAMELINACC`, `FRAMEANGACC`           |

# Benchmarking

Benchmark as follows:

```bash
mjwarp-testspeed --function=step --mjcf=test_data/humanoid/humanoid.xml --batch_size=8192
```

To get a full trace of the physics steps (e.g. timings of the subcomponents) run the following:

```bash
mjwarp-testspeed --function=step --mjcf=test_data/humanoid/humanoid.xml --batch_size=8192 --event_trace=True
```

