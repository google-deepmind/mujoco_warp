# Cloth

## Description

A soft-body simulation benchmark featuring a cloth draped over the MuJoCo humanoid. This tests the performance of [MuJoCo deformable bodies](https://mujoco.readthedocs.io/en/stable/modeling.html#deformable-objects).

### cloth

| Property | Value |
|----------|-------|
| Bodies | 918 |
| DoFs | 2706 |
| Actuators | 0 |
| Geoms | 21 |
| Timestep | 0.005s |
| Solver | CG |
| Friction | Pyramidal |
| Integrator | Euler |
| Matrix Format | Sparse |

![cloth](rollout.webp)

### cloth_render

This benchmark measures rendering performance of deformable meshes and soft-body grids on the GPU.

| Property | Value |
|----------|-------|
| Bodies | 918 |
| DoFs | 2706 |
| Actuators | 0 |
| Geoms | 21 |
| Cameras | 1 |
| Resolution | 64×64 |
| Worlds | 2048 |

![cloth_render](rollout_cloth_render.webp)

