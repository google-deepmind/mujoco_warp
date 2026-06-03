# Render

## Description

GPU ray-traced rendering performance benchmarks. These benchmarks measure rendering speed across both simple primitive layouts and complex high-resolution textured meshes on the GPU.

### primitives

GPU ray-traced rendering of the [primitives](../../mujoco_warp/test_data/primitives.xml) scene. This benchmark measures rendering performance using a 5×5 grid of spheres, capsules, ellipsoids, cylinders, and boxes above a plane.

| Property | Value |
|----------|-------|
| Bodies | 126 |
| DoFs | 750 |
| Geoms | 127 |
| Cameras | 1 |
| Resolution | 64×64 |
| Worlds | 8192 |

![primitives](primitives.webp)

### mug

GPU ray-traced rendering of the official [MuJoCo Mug](assets/mug.xml) scene. This benchmark measures rendering performance of a high-resolution, textured OBJ mesh with complex geometry.

| Property | Value |
|----------|-------|
| Bodies | 2 |
| DoFs | 6 |
| Geoms | 37 |
| Cameras | 1 |
| Resolution | 64×64 |
| Worlds | 8192 |

![mug](mug.webp)
