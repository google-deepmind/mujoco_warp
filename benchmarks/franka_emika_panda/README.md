# Franka Emika Panda

## Description

Measures MuJoCo Warp throughput for the Panda in idle and continuous assembly scenes.

### franka_emika_panda

| Property | Value |
|----------|-------|
| Bodies | 12 |
| DoFs | 9 |
| Actuators | 8 |
| Geoms | 23 |
| Timestep | 0.005s |
| Solver | Newton |
| Friction | Pyramidal |
| Integrator | ImplicitFast |
| Matrix Format | Dense |

![franka_emika_panda](rollout.webp)

### panda_nist_assembly

A Panda threads an M8 nut onto a bolt on the NIST board. The scene also contains
an RJ45 connector, large gear and 8 mm rod, already assembled at this point in
the recorded episode. It uses one ground plane, native SDF collision for the
assembly parts, and Panda visuals from the pinned Menagerie assets. Offline
preparation builds depth-8 SDFs into an MJB outside the source package.
The runner fetches the assembly meshes and license notices from a pinned Git snapshot.

The replay starts with the nut partly threaded and contains one initial
position/velocity state followed by 300 policy-recorded actuator targets at 10 ms
intervals. The standard loader holds each target for eight 1.25 ms physics steps.
Timing covers all 2,400 steps (three simulated seconds), with 4,096 parallel worlds.
There is no policy inference, per-step state restoration, or episode-reset logic.

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/run.py -f '^panda_nist_assembly$' --clear_warp_cache false
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/run.py -f '^panda_nist_assembly$' --view
```

The validation and timings in the provenance file were recorded before the rebase,
using MuJoCo 3.12.1.dev968306640, Warp 1.15.0 and the engine revision recorded there.
They are historical results, not measurements of current main.

Replay and asset hashes are recorded in [nist_k4_provenance.json](nist_k4_provenance.json).
See [asset attribution](https://github.com/ooctipus/mujoco_warp/blob/0613dbfa348fddea85cda81a85a31fdec60237db/benchmarks/franka_emika_panda/assets/nist/README.md) for the source designs and license notices.
