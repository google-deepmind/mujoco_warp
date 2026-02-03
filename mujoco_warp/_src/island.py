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

import warp as wp

from mujoco_warp._src import types


@wp.kernel
def _find_tree_edges(
  # Model:
  nv: int,
  dof_treeid: wp.array(dtype=int),
  # Data in:
  nefc_in: wp.array(dtype=int),
  efc_type_in: wp.array2d(dtype=int),
  efc_id_in: wp.array2d(dtype=int),
  efc_J_in: wp.array3d(dtype=float),
  njmax_in: int,
  # Out:
  edges_out: wp.array2d(dtype=int),
  nedge_out: wp.array(dtype=int),
):
  worldid, efcid = wp.tid()

  # skip if beyond active constraints
  if efcid >= wp.min(njmax_in, nefc_in[worldid]):
    return

  # skip continuation rows (same constraint type and id as previous)
  # this avoids duplicate edges from multi-row constraints (e.g., 3D contacts)
  if efcid > 0:
    if efc_type_in[worldid, efcid] == efc_type_in[worldid, efcid - 1]:
      if efc_id_in[worldid, efcid] == efc_id_in[worldid, efcid - 1]:
        return

  # collect trees touched by this constraint row
  prev_tree = int(-2)  # -1 is valid (static), so use -2 as sentinel
  first_tree = int(-2)

  for dof in range(nv):
    J_val = efc_J_in[worldid, efcid, dof]

    if J_val != 0.0:
      tree = dof_treeid[dof]

      if first_tree == -2:
        first_tree = tree
        prev_tree = tree
      elif tree != prev_tree and tree >= 0:
        # found a new tree, add edge between prev_tree and tree
        if prev_tree >= 0:
          idx = wp.atomic_add(nedge_out, 0, 1)
          if idx < njmax_in:
            # store as ordered pair for deduplication
            t1 = wp.min(prev_tree, tree)
            t2 = wp.max(prev_tree, tree)
            edges_out[idx, 0] = t1
            edges_out[idx, 1] = t2
        prev_tree = tree

  # add self-edge if only one tree found (constrained to itself)
  if first_tree >= 0 and prev_tree == first_tree:
    idx = wp.atomic_add(nedge_out, 0, 1)
    if idx < njmax_in:
      edges_out[idx, 0] = first_tree
      edges_out[idx, 1] = first_tree


@wp.kernel
def _compute_keys_and_indices(
  ntree: int,
  nedge_in: wp.array(dtype=int),
  edges_in: wp.array2d(dtype=int),
  njmax: int,
  keys_out: wp.array(dtype=int),
  indices_out: wp.array(dtype=int),
):
  """Compute sort keys and initialize indices. Launch: 2 * njmax."""
  i = wp.tid()

  # Always init index
  indices_out[i] = i

  # Only compute keys for first njmax elements
  if i >= njmax:
    return
  if i >= nedge_in[0]:
    keys_out[i] = 2147483647  # sort to end
  else:
    keys_out[i] = edges_in[i, 0] * ntree + edges_in[i, 1]


@wp.kernel
def _deduplicate_edges(
  nedge_in: wp.array(dtype=int),
  sorted_indices_in: wp.array(dtype=int),
  edges_in: wp.array2d(dtype=int),
  edges_out: wp.array2d(dtype=int),
  nedge_out: wp.array(dtype=int),
):
  """Mark unique and compact in one pass using atomics. Launch: njmax."""
  i = wp.tid()
  n = nedge_in[0]

  if i >= n:
    return

  # Check if this edge is unique (different from previous)
  is_unique = int(0)
  if i == 0:
    is_unique = 1
  else:
    idx = sorted_indices_in[i]
    prev_idx = sorted_indices_in[i - 1]
    if edges_in[idx, 0] != edges_in[prev_idx, 0]:
      is_unique = 1
    elif edges_in[idx, 1] != edges_in[prev_idx, 1]:
      is_unique = 1

  if is_unique == 1:
    # Atomic add to get output index
    dst = wp.atomic_add(nedge_out, 0, 1)
    src = sorted_indices_in[i]
    edges_out[dst, 0] = edges_in[src, 0]
    edges_out[dst, 1] = edges_in[src, 1]


def find_tree_edges(
  m: types.Model,
  d: types.Data,
) -> tuple[wp.array, wp.array]:
  """Find tree-tree edges from the constraint Jacobian.

  Args:
    m: The model containing kinematic and dynamic information.
    d: The data object containing the current state and output arrays.

  Returns:
    Tuple of (edges, nedge) arrays on device.
    edges has shape (njmax, 2) where each row is an ordered (t1, t2) pair.
    nedge is a (1,) array with the number of unique edges.
  """
  # allocate outputs
  edges = wp.zeros((d.njmax, 2), dtype=int)
  nedge = wp.zeros(1, dtype=int)

  # find edges
  wp.launch(
    kernel=_find_tree_edges,
    dim=(d.nworld, d.njmax),
    inputs=[
      m.nv,
      m.dof_treeid,
      d.nefc,
      d.efc.type,
      d.efc.id,
      d.efc.J,
      d.njmax,
      edges,
      nedge,
    ],
  )

  # compute sort keys and init indices (fused)
  keys = wp.zeros(2 * d.njmax, dtype=int)
  sorted_indices = wp.empty(2 * d.njmax, dtype=int)
  wp.launch(
    kernel=_compute_keys_and_indices,
    dim=2 * d.njmax,
    inputs=[m.ntree, nedge, edges, d.njmax, keys, sorted_indices],
  )

  # sort by keys using Warp's radix sort
  wp.utils.radix_sort_pairs(keys, sorted_indices, count=d.njmax)
  del keys

  # deduplicate edges (fused, no prefix sum)
  edges_unique = wp.zeros((d.njmax, 2), dtype=int)
  nedge_unique = wp.zeros(1, dtype=int)
  wp.launch(
    kernel=_deduplicate_edges,
    dim=d.njmax,
    inputs=[nedge, sorted_indices, edges, edges_unique, nedge_unique],
  )

  return edges_unique, nedge_unique
