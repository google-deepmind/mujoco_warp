# Franka Emika Panda

## Description

Measures MuJoCo Warp throughput for the Panda in idle and sparse-contact scenes.

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

### franka_emika_panda_sparse_contact

The Panda gripper presses against a fixed box. This variant exercises the
sparse Newton solver for many independent, small-DoF worlds with contact and
substantial contact and constraint capacity headroom. The headroom represents
batched manipulation workloads whose peak capacity is much larger than the
number of constraints active in a typical step. Its 600-contact and
3,000-constraint capacities mirror Factory-style manipulation configurations;
the default rollout averages approximately 5 active contacts and 24 constraint
rows per world.

| Property | Value |
|----------|-------|
| Bodies | 12 |
| DoFs | 9 |
| Actuators | 8 |
| Geoms | 24 |
| Timestep | 0.005s |
| Solver | Newton |
| Friction | Pyramidal |
| Integrator | ImplicitFast |
| Matrix Format | Sparse |
| Parallel Worlds | 4096 |
| Contact Capacity / World | 600 |
| Constraint Capacity / World | 3,000 |
