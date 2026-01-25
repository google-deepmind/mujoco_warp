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
from mujoco_warp._src.warp_util import event_scope


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


@wp.kernel
def _prefix_sum_per_world(
  # In:
  n: int,
  # In:
  input_in: wp.array2d(dtype=int),
  # Out:
  output_out: wp.array2d(dtype=int),
):
  """Compute exclusive prefix sum for each world independently."""
  worldid = wp.tid()
  total = int(0)
  for i in range(n):
    output_out[worldid, i] = total
    total = total + input_in[worldid, i]


@wp.kernel
def _count_edge_degrees(
  # Model:
  ntree: int,
  # In:
  nedge_in: wp.array(dtype=int),
  edges_in: wp.array2d(dtype=int),
  # Out:
  rownnz_out: wp.array2d(dtype=int),
):
  """Count edges per tree (degree in symmetric adjacency)."""
  worldid, edgeid = wp.tid()
  if edgeid >= nedge_in[0]:
    return
  t1 = edges_in[edgeid, 0]
  t2 = edges_in[edgeid, 1]
  if t1 >= 0 and t1 < ntree:
    wp.atomic_add(rownnz_out, worldid, t1, 1)
  if t2 != t1 and t2 >= 0 and t2 < ntree:
    wp.atomic_add(rownnz_out, worldid, t2, 1)


@wp.kernel
def _fill_adjacency(
  # Model:
  ntree: int,
  # In:
  nedge_in: wp.array(dtype=int),
  edges_in: wp.array2d(dtype=int),
  rowadr_in: wp.array2d(dtype=int),
  # Out:
  row_cursor_out: wp.array2d(dtype=int),
  colind_out: wp.array2d(dtype=int),
):
  """Fill CSR column indices for symmetric adjacency matrix."""
  worldid, edgeid = wp.tid()
  if edgeid >= nedge_in[0]:
    return
  t1 = edges_in[edgeid, 0]
  t2 = edges_in[edgeid, 1]
  if t1 >= 0 and t1 < ntree:
    slot = wp.atomic_add(row_cursor_out, worldid, t1, 1)
    colind_out[worldid, rowadr_in[worldid, t1] + slot] = t2
  if t2 != t1 and t2 >= 0 and t2 < ntree:
    slot = wp.atomic_add(row_cursor_out, worldid, t2, 1)
    colind_out[worldid, rowadr_in[worldid, t2] + slot] = t1



@wp.kernel
def _flood_fill(
  # Model:
  ntree: int,
  max_stack_size: int,
  # In:
  rownnz_in: wp.array2d(dtype=int),
  rowadr_in: wp.array2d(dtype=int),
  colind_in: wp.array2d(dtype=int),
  # InOut:
  labels_inout: wp.array2d(dtype=int),
  # Out:
  nisland_out: wp.array(dtype=int),
  # Scratch:
  stack_scratch: wp.array2d(dtype=int),
):
  """DFS flood fill to discover islands, matching MuJoCo's mj_floodFill().
  
  Each thread handles one world. For each unvisited tree with edges,
  performs DFS traversal to assign all connected trees the same island ID.
  
  Args:
    ntree: Number of trees
    max_stack_size: Maximum stack size (>= 2*njmax for safety)
    rownnz_in: Number of neighbors per tree  
    rowadr_in: CSR row addresses
    colind_in: CSR column indices (neighbor tree IDs)
    labels_inout: On exit, island ID per tree (-1 if singleton)
    nisland_out: Number of islands discovered
    stack_scratch: Scratch array for DFS stack, shape (nworld, max_stack_size)
  """
  worldid = wp.tid()

  # Initialize island count
  nisland = int(0)

  # Iterate over vertices, discover islands
  for i in range(ntree):
    # Tree already in island or singleton with no edges: skip
    if labels_inout[worldid, i] != -1 or rownnz_in[worldid, i] == 0:
      continue

    # Push i onto stack
    nstack = int(0)
    stack_scratch[worldid, nstack] = i
    nstack = nstack + 1

    # DFS traversal of island
    while nstack > 0:
      # Pop v from stack
      nstack = nstack - 1
      v = stack_scratch[worldid, nstack]

      # If v is already assigned, continue
      if labels_inout[worldid, v] != -1:
        continue

      # Assign v to current island
      labels_inout[worldid, v] = nisland

      # Push adjacent vertices onto stack
      for j in range(rownnz_in[worldid, v]):
        neighbor = colind_in[worldid, rowadr_in[worldid, v] + j]
        # Only push if not already assigned and stack has room
        if labels_inout[worldid, neighbor] == -1 and nstack < max_stack_size:
          stack_scratch[worldid, nstack] = neighbor
          nstack = nstack + 1

    # Island is filled: increment nisland
    nisland = nisland + 1

  # Write island count
  nisland_out[worldid] = nisland



@wp.kernel
def _count_island_trees(
  # Model:
  ntree: int,
  # In:
  labels_in: wp.array2d(dtype=int),
  # Data out:
  island_ntree_out: wp.array2d(dtype=int),
):
  """Count trees per island."""
  worldid, tree = wp.tid()
  if tree >= ntree:
    return
  island = labels_in[worldid, tree]
  if island >= 0:
    wp.atomic_add(island_ntree_out, worldid, island, 1)


@wp.kernel
def _build_tree_mapping(
  # Model:
  ntree: int,
  # Data in:
  island_itreeadr_in: wp.array2d(dtype=int),
  # In:
  labels_in: wp.array2d(dtype=int),
  # Data out:
  map_itree2tree_out: wp.array2d(dtype=int),
  # Out:
  island_cursor_out: wp.array2d(dtype=int),
):
  """Build map from island-local tree index to global tree ID."""
  worldid, tree = wp.tid()
  if tree >= ntree:
    return
  island = labels_in[worldid, tree]
  if island < 0:
    return
  slot = wp.atomic_add(island_cursor_out, worldid, island, 1)
  map_itree2tree_out[worldid, island_itreeadr_in[worldid, island] + slot] = tree


@event_scope
def island(
  m: types.Model,
  d: types.Data,
) -> None:
  """Discover constraint islands via edge extraction and label propagation.
  """
  if m.ntree == 0:
    d.nisland.zero_()
    return

  # Allocate temporary arrays
  rownnz = wp.zeros((d.nworld, m.ntree), dtype=int)
  rowadr = wp.zeros((d.nworld, m.ntree), dtype=int)
  colind = wp.zeros((d.nworld, 2 * d.njmax), dtype=int)
  row_cursor = wp.zeros((d.nworld, m.ntree), dtype=int)
  island_cursor = wp.zeros((d.nworld, types.MJ_MAX_NISLAND), dtype=int)

  # Step 1: Find tree edges from Jacobian
  edges, nedge = find_tree_edges(m, d)

  # Step 2: Build CSR adjacency matrix
  wp.launch(_count_edge_degrees, dim=(d.nworld, d.njmax), inputs=[m.ntree, nedge, edges, rownnz])
  wp.launch(_prefix_sum_per_world, dim=d.nworld, inputs=[m.ntree, rownnz, rowadr])
  wp.launch(_fill_adjacency, dim=(d.nworld, d.njmax), inputs=[m.ntree, nedge, edges, rowadr, row_cursor, colind])

  # Step 3: DFS flood fill to discover islands
  # DFS directly assigns island IDs to tree_island and counts nisland
  # Initialize labels to -1 (labels are stored directly in tree_island)
  d.tree_island.fill_(-1)
  
  # Allocate DFS stack scratch space (2*njmax for safety)
  max_stack_size = 2 * d.njmax
  stack_scratch = wp.zeros((d.nworld, max_stack_size), dtype=int)
  
  wp.launch(
    _flood_fill,
    dim=d.nworld,
    inputs=[m.ntree, max_stack_size, rownnz, rowadr, colind, d.tree_island, d.nisland, stack_scratch],
  )

  # Step 5: Build island-tree mappings
  d.island_ntree.zero_()
  wp.launch(_count_island_trees, dim=(d.nworld, m.ntree), inputs=[m.ntree, d.tree_island, d.island_ntree])
  wp.launch(_prefix_sum_per_world, dim=d.nworld, inputs=[types.MJ_MAX_NISLAND, d.island_ntree, d.island_itreeadr])
  wp.launch(
    _build_tree_mapping,
    dim=(d.nworld, m.ntree),
    inputs=[m.ntree, d.island_itreeadr, d.tree_island, d.map_itree2tree, island_cursor],
  )
