# Franka Threading

## Description

A [Franka Emika Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/affef0836947b64cc06c4ab1cbf0152835693374/franka_emika_panda) threads an M16×2 nut onto a stationary bolt. The 5.5-second control replay starts engaged and completes one full turn through three grasp, turn, release, and regrasp phases. Convex helical thread meshes exercise contact-rich manipulation with physical finger grasps.

### franka_threading

| Property | Value |
|----------|-------|
| Bodies | 14 |
| DoFs | 15 |
| Actuators | 8 |
| Geoms | 614 |
| Timestep | 0.001s |
| Solver | Newton |
| Friction | Elliptic |
| Integrator | Implicit Fast |
| Matrix Format | Dense |

[![Threading trajectory](https://github.com/StafaH/mujoco_warp/releases/download/benchmark-videos/franka-threading-reviewed.webp)](https://github.com/StafaH/mujoco_warp/releases/download/benchmark-videos/franka-threading-reviewed.mp4)

## Creation steps

1. The Isaac Lab Factory M16×2 [nut](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/factory_nut_m16.usd) and [bolt](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/factory_bolt_m16.usd) USD meshes were converted to OBJ for the original visual model. Those visual-only meshes are omitted here; the collision geometry is also used for display.
2. Collision geometry was generated procedurally with [`mjfit.thread`](https://github.com/younghyopark/mjfit/tree/5833f465896e1e06bc4be4f003be847f5183a4db), rather than decomposing the USD triangles. The parameters are a 16 mm major diameter, 2 mm pitch, 25 mm threaded length, 13 mm nut height, 24 mm nut and bolt-head size, 10 mm bolt-head height, 0.1 mm clearance, 24 segments per turn, 5 µm seam relief, and 1 mm chamfer. The resulting convex thread and shell meshes are embedded in `scene.xml`.
3. The robot is derived from `benchmarks/franka_emika_panda/panda.xml`, with visual meshes fetched from the pinned MuJoCo Menagerie revision in `__init__.py`.
4. The position-control trajectory in `threading.npz` starts from an engaged keyframe and uses three 120° turns, rewinding the wrist between grasps to stay within joint limits. The nut advances approximately 2 mm through finger and thread contact.

## License

[LICENSE](LICENSE) contains the Isaac Lab Factory BSD-3-Clause and mjfit MIT licenses. The Factory-specific [acknowledgments](https://github.com/isaac-sim/IsaacLab/blob/main/docs/licenses/assets/factory-acknowledgments.txt) identify the nut and bolt as custom-designed assets under the Isaac Gym Environments license. Franka meshes retain their upstream [Apache-2.0 license](https://github.com/google-deepmind/mujoco_menagerie/blob/affef0836947b64cc06c4ab1cbf0152835693374/franka_emika_panda/LICENSE).
