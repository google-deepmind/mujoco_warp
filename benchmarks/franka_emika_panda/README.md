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

A Panda manipulates an M8 nut, RJ45 connector, large gear and 8 mm rod on the NIST board.
The scene uses one ground plane, native SDF collision for the assembly parts, and
Panda visuals from the existing pinned Menagerie assets. Offline preparation builds
depth-8 SDFs into an MJB outside the source package.

The replay contains one initial position/velocity state and 3,159 actuator targets
spaced 10 ms apart. The standard replay loader holds each target for eight 1.25 ms
physics steps. Runtime uses continuous native physics with the original solver,
without policy inference, per-step state restoration, or episode resets.

The controls were recorded from a successful native policy rollout, ending at its
first simultaneous assembly success after 31.59 seconds. Fixed-control replay on
this engine drifts and has not reproduced that success. This local draft measures
continuous physics throughput; it is not yet a successful assembly demonstration.

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/run.py -f '^panda_nist_assembly$' --clear_warp_cache false
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/run.py -f '^panda_nist_assembly$' --view
```

Routine timing covers the first 1,000 physics steps (1.25 seconds) at 4,096 worlds.
The viewer uses the complete control tape. Full-episode timing can be requested
with the standard testspeed replay command by omitting `--nstep`.

On an RTX 5090 at 4,096 worlds, the median of three runs gives solver time
2.314 ms dense → 1.496 ms sparse (35.4% shorter), and whole-step time
29.657 → 28.454 ms (4.1% shorter). Native SDF collision takes about 26.1 ms
per step and remains the bottleneck. These are continuous-prefix measurements;
the two layouts follow their own evolving trajectories.

Measurements use MuJoCo 3.12.1.dev968306640 and Warp 1.15.0, with the same
prepared model and recorded controls for both layouts. The repository lock may
select a newer MuJoCo build; reproducing these numbers requires the measured
versions. With the model prepared and kernels cached, each 1,000-step process
took about 40.6 seconds dense or 37.5 seconds sparse, including model loading.

Replay and asset hashes are recorded in [nist_k4_provenance.json](nist_k4_provenance.json).
See [asset attribution](assets/nist/README.md) for the source designs and license notices.
