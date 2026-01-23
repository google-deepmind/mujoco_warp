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
def _compute_edge_keys(
  # Model:
  ntree: int,
  # In:
  nedge_in: wp.array(dtype=int),
  edges_in: wp.array2d(dtype=int),
  # Out:
  keys_out: wp.array(dtype=int),
):
  """Compute sort keys for edges: t1 * ntree + t2."""
  i = wp.tid()
  if i >= nedge_in[0]:
    keys_out[i] = 2147483647  # max int, sort to end
    return
  keys_out[i] = edges_in[i, 0] * ntree + edges_in[i, 1]


@wp.kernel
def _init_indices(
  # Out:
  indices_out: wp.array(dtype=int),
):
  """Initialize indices to [0, 1, 2, ...]."""
  i = wp.tid()
  indices_out[i] = i


@wp.kernel
def _mark_unique_edges(
  # In:
  nedge_in: wp.array(dtype=int),
  sorted_indices_in: wp.array(dtype=int),
  edges_in: wp.array2d(dtype=int),
  # Out:
  unique_mask_out: wp.array(dtype=int),
):
  """Mark unique edges after sorting."""
  i = wp.tid()
  n = nedge_in[0]
  if i >= n:
    unique_mask_out[i] = 0
    return

  if i == 0:
    unique_mask_out[i] = 1
    return

  idx = sorted_indices_in[i]
  prev_idx = sorted_indices_in[i - 1]

  # check if different from previous
  if edges_in[idx, 0] != edges_in[prev_idx, 0] or edges_in[idx, 1] != edges_in[prev_idx, 1]:
    unique_mask_out[i] = 1
  else:
    unique_mask_out[i] = 0


@wp.kernel
def _compact_edges(
  # In:
  nedge_in: wp.array(dtype=int),
  sorted_indices_in: wp.array(dtype=int),
  unique_prefix_in: wp.array(dtype=int),
  unique_mask_in: wp.array(dtype=int),
  edges_in: wp.array2d(dtype=int),
  # Out:
  edges_out: wp.array2d(dtype=int),
  nedge_out: wp.array(dtype=int),
):
  """Compact edges using prefix sum of unique mask."""
  i = wp.tid()
  n = nedge_in[0]
  if i >= n:
    return

  if unique_mask_in[i] == 1:
    # exclusive prefix sum gives destination index
    dst = unique_prefix_in[i]
    src = sorted_indices_in[i]
    edges_out[dst, 0] = edges_in[src, 0]
    edges_out[dst, 1] = edges_in[src, 1]

    # last unique element writes the count
    if i == n - 1 or unique_mask_in[i + 1] == 0:
      # count is prefix + 1 for this element
      pass

  # the last thread writes the unique count
  if i == n - 1:
    nedge_out[0] = unique_prefix_in[n - 1] + unique_mask_in[n - 1]


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

  # compute sort keys (need 2x size for radix sort scratch space)
  keys = wp.zeros(2 * d.njmax, dtype=int)
  wp.launch(
    kernel=_compute_edge_keys,
    dim=d.njmax,
    inputs=[m.ntree, nedge, edges, keys],
  )

  # sort by keys using Warp's radix sort
  # radix_sort_pairs sorts (keys, values) pairs - values must be initialized to indices
  sorted_indices = wp.empty(2 * d.njmax, dtype=int)
  wp.launch(
    kernel=_init_indices,
    dim=2 * d.njmax,
    inputs=[sorted_indices],
  )
  wp.utils.radix_sort_pairs(keys, sorted_indices, count=d.njmax)
  del keys

  # mark unique edges
  unique_mask = wp.zeros(d.njmax, dtype=int)
  wp.launch(
    kernel=_mark_unique_edges,
    dim=d.njmax,
    inputs=[nedge, sorted_indices, edges, unique_mask],
  )

  # prefix sum for compaction addresses
  unique_prefix = wp.zeros(d.njmax, dtype=int)
  wp.utils.array_scan(unique_mask, unique_prefix, inclusive=False)

  # compact edges
  edges_unique = wp.zeros((d.njmax, 2), dtype=int)
  nedge_unique = wp.zeros(1, dtype=int)
  wp.launch(
    kernel=_compact_edges,
    dim=d.njmax,
    inputs=[
      nedge,
      sorted_indices,
      unique_prefix,
      unique_mask,
      edges,
      edges_unique,
      nedge_unique,
    ],
  )

  return edges_unique, nedge_unique
