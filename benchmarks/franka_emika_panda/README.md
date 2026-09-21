# Franka Emika Panda

## Description

Measures MuJoCo Warp throughput for the Panda in idle and saved-state assembly scenes.

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

### panda_nist_k4_states_dense / panda_nist_k4_states_sparse

A Panda assembles an M8 nut, RJ45 connector, large gear and 8 mm rod on the NIST board.
Eight local assembly meshes provide native SDF collision; the board uses its source
box collision and authored visual mesh. Panda visuals use the existing pinned Menagerie assets.
Offline preparation builds depth 8 SDFs into an MJB in the assembled assets directory.

This scene requires the native SDF gradient, line-search and disjoint-bound corrections
included in this branch. Earlier published profiling curves use a separate sparse-optimization
build; see the PR description for exact revisions.

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/run.py -f '^panda_nist_k4_states_' --clear_warp_cache false
```

The benchmark restores 559 preterminal states from a source IsaacLab/Newton policy
episode, sampled every 40 ms through its first geometric success at 22.36 s.
Each sample measures one 1.25 ms native MJWarp step on 4,096 replicated worlds.
State restoration is excluded and warmstart is reset to zero. Both layouts use
600 contact slots, 704 constraint slots and 64 CCD slots; sleeping is disabled.
No policy inference or source collision pipeline is used at benchmark runtime.

These are independent saved-state measurements, not continuous action replay;
`--view` is unavailable for this mode. The source board and one ground at the source
tabletop height support the assemblies. Recorded states, including source-existing
intersections, are preserved. GPU event components exclude state upload; standard
runner throughput also includes host launch overhead.

Source and asset hashes are recorded in [nist_k4_provenance.json](nist_k4_provenance.json).
See [asset attribution](assets/nist/README.md) for the source designs and retained license notices.
