# NIST assembly meshes

These OBJ files were exported from NIST assembly USD scenes. The eight assembly
meshes use the source collision geometry; the board mesh uses its authored
visual geometry. The benchmark provenance JSON records each USD and OBJ hash.

The original Factory nut, bolt, peg, hole and gear-base models are distributed
under the NVIDIA BSD-3-Clause terms, as stated in the upstream
[Factory acknowledgment](https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/assets/licenses/factory-acknowledgments.txt).
The unchanged terms and acknowledgment are retained in `LICENSE.factory.txt`
and `factory-acknowledgments.txt`. The source peg and hole visuals closely match
those upstream models. The local nut, bolt and gear-base models include changes
whose full intermediate history was not retained. Local gear metadata also
references [FORGE](https://noseworm.github.io/forge/); Factory credits NIST for
its original gear and shaft designs.

The board and RJ45 connectors belong to the
[NIST Assembly Task Board 1 design family](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly).
The asset contributor confirmed that these local meshes use the same
[BSD-3-Clause terms](LICENSE.factory.txt) as the Factory assembly parts.
NIST provides the [board designs](https://www.nist.gov/document/gmclaserplatezip),
[connector-housing STL files](https://www.nist.gov/document/connectorhousingszip),
[component CAD files](https://www.nist.gov/document/task-board-1-cad-zip) and
[component STL files](https://www.nist.gov/document/taskboard1stlzip), subject to
its [copyright and disclaimer information](https://www.nist.gov/copyrights-disclaimers).
These links identify the design sources; the exact upstream archive versions
used for the local board and connector conversions were not retained.

The benchmark exports preserve the supplied geometry, applying USD transforms
and triangulation to write OBJ files in meters. They do not include the source
USD layers, local conversion paths or account metadata. The Factory license
notice is retained for the Factory sources and the local board and RJ45 exports.
