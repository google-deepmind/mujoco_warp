# Flex contacts attributed to unrelated rigid sensors

`flex_rigid_sensors.py` is a standalone reproducer, **not a fix**. It is outside
pytest collection and does not change production code or mark failures as expected.

From a development checkout with dependencies installed:

```sh
uv run python -m contrib.repro.flex_rigid_sensors --device cpu --sensor force --flex-collisions off
uv run python -m contrib.repro.flex_rigid_sensors --device cpu --sensor force --flex-collisions on
```

Repeat with `--sensor touch` and `--device cuda:0`. `--sensor both` is the default.
Use `--nworld 2` for heterogeneous worlds with different sphere heights, velocities,
and native touch loads. The control retains the flex but disables its collisions.

Both controls require ordinary native MuJoCo parity at `rtol=atol=1e-5`. The
collision-disabled case should pass; the collision-enabled case currently fails
with `AssertionError` and a nonzero exit code. A failure is evidence of the bug,
not a passing regression test. JSON printed before the assertion includes sensor
values, rigid body forces, contact participants, finiteness, and overflow flags.

The scene has a sphere and a separate small cloth contacting a plane. A spherical
site on the rigid body covers both contact regions, so spatial filtering cannot
hide erroneous touch-body matching. The sphere's binary-exact radius and height
avoid an unrelated near-zero float32 force discrepancy in the passing control.
Only input positions and velocities are copied; native and Warp each execute a
real forward pass. Sensor and body-force outputs are poisoned before Warp forward.

The force consumer `smooth._cfrc_ext_contact` and the ordinary touch consumer
`sensor._sensor_touch` index `geom_bodyid` using a flex contact's `geom == -1` sentinel.
Warp can resolve that index to the last rigid geom's body, silently attributing
cloth loads to the unrelated sphere. Consequently, clean memory checking does
not establish correct physics. To inspect this distinction on CUDA:

```sh
uv run compute-sanitizer --tool memcheck --error-exitcode 99 python -m contrib.repro.flex_rigid_sensors --device cuda:0
```

One possible implementation direction is to resolve each contact participant
from its geom/flex/element/vertex metadata before projecting and accumulating
body forces or matching touch bodies. Element and interpolated flex ownership
must retain the appropriate barycentric/interpolation weights. Simply skipping
all flex contacts would discard genuine loads and is not a complete fix. This
reproducer does not implement or validate that direction; element/interpolated
ownership needs additional regression coverage with any eventual fix.

Related #1422.
