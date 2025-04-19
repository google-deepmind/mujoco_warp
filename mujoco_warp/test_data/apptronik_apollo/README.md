# Apptronik Apollo Large Terrain

<p float="left">
  <img src="apollo_terrain.png" width="400">
</p>

## Overview
Collisions-only version of Apollo humanoid.

This box-only terrain is a 10x20 grid of subterrains. The subterrains are roughly 45% upright pyramids, 45% inverted pyramids and 10% flat. The pyramids have random step heights.

## Derivation Steps
Got the apollo XML from `contrib/xml/apptronik_apollo.xml` ([source file](https://github.com/google-deepmind/mujoco_warp/blob/8b26735a3ab3602cee898cfb002f609c38137b36/contrib/xml/apptronik_apollo.xml), but removed visual assets.

Apart from the additional geoms, `scene_terrain.xml` differs from the original `scene.xml` in a few settings, for stable simulation:
1. Use `<flag eulerdamp="disable" />`
2. `iterations`: `2` -> `10`
3. `ls_iterations`: `10` -> `20`
