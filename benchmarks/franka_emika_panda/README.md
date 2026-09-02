# Franka Emika Panda

## Description

Measures MuJoCo Warp throughput for Panda robots in idle and sparse-contact scenes.

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

### franka_emika_pandas_sparse_contact

Five independent Panda grippers press against fixed boxes. Each constraint row
touches one 9-DoF chain, giving the 45-DoF system a naturally block-sparse
constraint topology. The default rollout averages approximately 27 contacts and
119 constraint rows per world (p95: 35 and 155), which fit the 64-contact and
192-constraint capacities.

| Property | Value |
|----------|-------|
| Bodies | 56 |
| DoFs | 45 |
| Actuators | 40 |
| Geoms | 116 |
| Timestep | 0.005s |
| Solver | Newton |
| Friction | Pyramidal |
| Integrator | ImplicitFast |
| Matrix Format | Sparse |
| Parallel Worlds | 4096 |
| Contact Capacity / World | 64 |
| Constraint Capacity / World | 192 |
