# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Applies physically coupled mass and rotational-inertia scales."""

import numpy as np
import warp as wp

_MODEL_FIELDS = (
  "body_mass",
  "body_inertia",
  "body_subtreemass",
  "dof_invweight0",
  "body_invweight0",
)


def scale(parameters, reference_mass, reference_inertia):
  """Applies per-link log scales to reference mass and principal moments."""
  parameters = np.asarray(parameters)
  if parameters.shape[-1] != 1:
    raise ValueError("each link requires one inertial scale")
  factor = np.exp(parameters[..., 0])
  mass = factor * np.asarray(reference_mass)
  inertia = factor[..., None] * np.asarray(reference_inertia)
  return mass, inertia


@wp.kernel
def _project(
  # In:
  parameters: wp.array[float],
  num_links: int,
  body_link: wp.array[int],
  reference_mass: wp.array[float],
  reference_inertia: wp.array[wp.vec3],
  # Out:
  mass_out: wp.array2d[float],
  inertia_out: wp.array2d[wp.vec3],
):
  worldid, bodyid = wp.tid()
  linkid = body_link[bodyid]
  if linkid < 0:
    mass_out[worldid, bodyid] = reference_mass[bodyid]
    inertia_out[worldid, bodyid] = reference_inertia[bodyid]
    return

  factor = wp.exp(parameters[worldid * num_links + linkid])
  mass_out[worldid, bodyid] = factor * reference_mass[bodyid]
  inertia_out[worldid, bodyid] = factor * reference_inertia[bodyid]


def project(parameters, num_links, body_link, reference_mass, reference_inertia, mass_out, inertia_out):
  """Projects per-link scales into batched MuJoCo mass and inertia arrays."""
  if mass_out.shape != inertia_out.shape[:2]:
    raise ValueError("mass and inertia outputs must have matching world and body dimensions")
  if parameters.size != mass_out.shape[0] * num_links:
    raise ValueError("each link requires one inertial scale")
  wp.launch(
    _project,
    dim=inertia_out.shape,
    inputs=[parameters, num_links, body_link, reference_mass, reference_inertia],
    outputs=[mass_out, inertia_out],
  )
