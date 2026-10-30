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

"""Device-side canonical permutations for contacts and constraints."""

from typing import Any, Callable

import warp as wp

from mujoco_warp._src import types
from mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"default_grid_stride": False})


@wp.struct
class _ConstraintKeys:
  primary: wp.array2d[wp.vec4i]
  J: wp.array3d[wp.uint32]
  J_rownnz: wp.array2d[int]
  J_rowadr: wp.array2d[int]
  J_colind: wp.array3d[int]
  pos: wp.array2d[wp.uint32]
  margin: wp.array2d[wp.uint32]
  D: wp.array2d[wp.uint32]
  vel: wp.array2d[wp.uint32]
  aref: wp.array2d[wp.uint32]
  frictionloss: wp.array2d[wp.uint32]
  nv: int


@wp.func
def _float_key(bits: wp.uint32) -> wp.uint32:
  """Total ordering of float bits, including signed zero and NaNs."""
  return wp.where((bits & wp.uint32(0x80000000)) != wp.uint32(0), ~bits, bits ^ wp.uint32(0x80000000))


@wp.func
def _contact_less(keys: wp.array3d[wp.uint32], worldid: int, a: int, b: int) -> bool:
  for j in range(keys.shape[2]):
    ka = keys[worldid, a, j]
    kb = keys[worldid, b, j]
    if ka != kb:
      return ka < kb
  return False


@wp.func
def _constraint_less(keys: _ConstraintKeys, worldid: int, a: int, b: int) -> bool:
  ka = keys.primary[worldid, a]
  kb = keys.primary[worldid, b]
  for i in range(4):
    if ka[i] != kb[i]:
      return ka[i] < kb[i]

  # Equal type/id is not sufficient for flex edges. Compare actual Jacobian rows, not their atomic
  # allocation addresses. Connect/weld and contact components were already distinguished above.
  if keys.J_rownnz.shape[1] > 0:
    na = keys.J_rownnz[worldid, a]
    nb = keys.J_rownnz[worldid, b]
    aa = keys.J_rowadr[worldid, a]
    ab = keys.J_rowadr[worldid, b]
    for i in range(wp.min(na, nb)):
      ca = keys.J_colind[worldid, 0, aa + i]
      cb = keys.J_colind[worldid, 0, ab + i]
      if ca != cb:
        return ca < cb
      va = _float_key(keys.J[worldid, 0, aa + i])
      vb = _float_key(keys.J[worldid, 0, ab + i])
      if va != vb:
        return va < vb
    if na != nb:
      return na < nb
  else:
    for i in range(keys.nv):
      va = _float_key(keys.J[worldid, a, i])
      vb = _float_key(keys.J[worldid, b, i])
      if va != vb:
        return va < vb

  va = _float_key(keys.pos[worldid, a])
  vb = _float_key(keys.pos[worldid, b])
  if va != vb:
    return va < vb
  va = _float_key(keys.margin[worldid, a])
  vb = _float_key(keys.margin[worldid, b])
  if va != vb:
    return va < vb
  va = _float_key(keys.D[worldid, a])
  vb = _float_key(keys.D[worldid, b])
  if va != vb:
    return va < vb
  va = _float_key(keys.vel[worldid, a])
  vb = _float_key(keys.vel[worldid, b])
  if va != vb:
    return va < vb
  va = _float_key(keys.aref[worldid, a])
  vb = _float_key(keys.aref[worldid, b])
  if va != vb:
    return va < vb
  va = _float_key(keys.frictionloss[worldid, a])
  vb = _float_key(keys.frictionloss[worldid, b])
  if va != vb:
    return va < vb
  return False


@wp.func
def _block_less(keys: wp.array2d[int], worldid: int, a: int, b: int) -> bool:
  return keys[worldid, a] < keys[worldid, b]


@wp.kernel
def _identity(order_out: wp.array2d[int]):
  worldid, i = wp.tid()
  order_out[worldid, i] = i


@cache_kernel
def _merge(compare: Callable):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # In:
    count_in: wp.array[int],
    keys_in: Any,
    order_in: wp.array2d[int],
    width: int,
    # Out:
    order_out: wp.array2d[int],
  ):
    worldid, i = wp.tid()
    count = wp.min(count_in[worldid], order_in.shape[1])
    if i >= count:
      order_out[worldid, i] = i
      return
    start = (i // (2 * width)) * (2 * width)
    middle = wp.min(start + width, count)
    end = wp.min(start + 2 * width, count)
    left = i < middle
    lo = wp.where(left, middle, start)
    hi = wp.where(left, end, middle)
    other_start = lo
    source = order_in[worldid, i]
    while lo < hi:
      mid = (lo + hi) // 2
      other = order_in[worldid, mid]
      before = compare(keys_in, worldid, other, source)
      # Stable merge of *identical* keys, not a key based on emission order.
      if not left:
        before = not compare(keys_in, worldid, source, other)
      if before:
        lo = mid + 1
      else:
        hi = mid
    dest = start + (i - wp.where(left, start, middle)) + lo - other_start
    order_out[worldid, dest] = source

  return kernel


@wp.kernel
def _gather(
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  value_in: wp.array3d[Any],
  # Out:
  value_out: wp.array3d[Any],
):
  worldid, row, col = wp.tid()
  if row < count_in[worldid] and row < order_in.shape[1]:
    value_out[worldid, row, col] = value_in[worldid, order_in[worldid, row], col]


@cache_kernel
def _contact_key_field(is_float: bool, feature: bool):
  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # Data in:
    nacon_in: wp.array[int],
    # In:
    geom_in: wp.array[wp.vec2i],
    value_in: wp.array3d[wp.uint32],
    offset: int,
    # Out:
    keys_out: wp.array3d[wp.uint32],
  ):
    i, j = wp.tid()
    if i >= wp.min(nacon_in[0], keys_out.shape[1]):
      return
    bits = value_in[0, i, j]
    if wp.static(feature):
      # Rigid writers do not initialize flex feature fields; ignore those slots.
      if geom_in[i][j] >= 0:
        bits = wp.uint32(0xFFFFFFFF)
    if wp.static(is_float):
      key = _float_key(bits)
    else:
      key = bits ^ wp.uint32(0x80000000)
    keys_out[0, i, offset + j] = key

  return kernel


@wp.kernel
def _constraint_counts(
  # Data in:
  nefc_in: wp.array[int],
  njmax_in: int,
  njmax_nnz_in: int,
  # In:
  nnz_in: wp.array[int],
  # Out:
  count_out: wp.array[int],
):
  worldid = wp.tid()
  count = nefc_in[worldid]
  # Overflow may leave partially assembled groups/rows. Leave them and the original counters alone;
  # the existing overflow checks report the error. Canonical ordering requires sufficient buffers.
  if count > njmax_in:
    count = 0
  if nnz_in.shape[0] > 0:
    if nnz_in[worldid] > njmax_nnz_in:
      count = 0
  count_out[worldid] = count


@wp.kernel
def _constraint_primary(
  # Model:
  eq_type: wp.array[int],
  # In:
  count_in: wp.array[int],
  type_in: wp.array2d[int],
  id_in: wp.array2d[int],
  # Out:
  keys_out: wp.array2d[wp.vec4i],
):
  worldid, row = wp.tid()
  if row < count_in[worldid]:
    row_type = type_in[worldid, row]
    id = id_in[worldid, row]
    subtype = int(0)
    component = int(0)
    if row_type == int(types.ConstraintType.EQUALITY):
      subtype = eq_type[id]
      if subtype == int(types.EqType.CONNECT) or subtype == int(types.EqType.WELD):
        # Assembly emits each group contiguously in XYZ force, then XYZ torque order.
        # Use the offset within that group, not its nondeterministic allocation address.
        nrow = 3 if subtype == int(types.EqType.CONNECT) else 6
        while component + 1 < nrow and row > component:
          previous = row - component - 1
          if type_in[worldid, previous] != row_type or id_in[worldid, previous] != id:
            break
          component += 1
    # Force reconstruction requires all connects before welds, regardless of XML equality IDs.
    keys_out[worldid, row] = wp.vec4i(row_type, subtype, id, component)


@wp.kernel
def _contact_constraint_primary(
  # Data in:
  nacon_in: wp.array[int],
  # In:
  count_in: wp.array[int],
  worldid_in: wp.array[int],
  efc_address_in: wp.array2d[int],
  order_in: wp.array2d[int],
  type_in: wp.array2d[int],
  id_in: wp.array2d[int],
  dim_in: wp.array[int],
  is_pyramidal: bool,
  # Out:
  keys_out: wp.array2d[wp.vec4i],
):
  rank, component = wp.tid()
  if rank >= wp.min(nacon_in[0], order_in.shape[1]):
    return
  conid = order_in[0, rank]
  dim = dim_in[conid]
  nrow = wp.max(1, 2 * (dim - 1)) if is_pyramidal else dim
  if component >= nrow:
    return
  worldid = worldid_in[conid]
  row = efc_address_in[conid, component]
  if row >= 0 and row < count_in[worldid]:
    if type_in[worldid, row] >= int(types.ConstraintType.CONTACT_FRICTIONLESS) and id_in[worldid, row] == conid:
      keys_out[worldid, row] = wp.vec4i(type_in[worldid, row], 0, rank, component)


@wp.kernel
def _inverse(order_in: wp.array2d[int], inverse_out: wp.array2d[int]):
  worldid, row = wp.tid()
  inverse_out[worldid, order_in[worldid, row]] = row


@wp.kernel
def _remap_contacts(
  # Data in:
  nacon_in: wp.array[int],
  # In:
  count_in: wp.array[int],
  worldid_in: wp.array[int],
  inverse_in: wp.array2d[int],
  address_in: wp.array2d[int],
  # Out:
  address_out: wp.array2d[int],
):
  conid, component = wp.tid()
  if conid < wp.min(nacon_in[0], address_in.shape[0]):
    row = address_in[conid, component]
    worldid = worldid_in[conid]
    if row >= 0 and row < count_in[worldid]:
      address_out[conid, component] = inverse_in[worldid, row]


@wp.kernel
def _sparse_offsets(
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  rownnz_in: wp.array2d[int],
  # Out:
  offsets_out: wp.array2d[int],
):
  worldid = wp.tid()
  offset = int(0)
  # Scan assembled rows within each world, not the (possibly interleaved) contact buffer.
  for row in range(count_in[worldid]):
    offsets_out[worldid, row] = offset
    offset += rownnz_in[worldid, order_in[worldid, row]]


@wp.kernel
def _pack_sparse(
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  offsets_in: wp.array2d[int],
  rownnz_in: wp.array2d[int],
  rowadr_in: wp.array2d[int],
  colind_in: wp.array3d[int],
  J_in: wp.array3d[float],
  # Out:
  rownnz_out: wp.array2d[int],
  rowadr_out: wp.array2d[int],
  colind_out: wp.array3d[int],
  J_out: wp.array3d[float],
):
  worldid, row = wp.tid()
  if row >= count_in[worldid]:
    return
  old = order_in[worldid, row]
  nnz = rownnz_in[worldid, old]
  start = rowadr_in[worldid, old]
  adr = offsets_in[worldid, row]
  rownnz_out[worldid, row] = nnz
  rowadr_out[worldid, row] = adr
  for i in range(nnz):
    colind_out[worldid, 0, adr + i] = colind_in[worldid, 0, start + i]
    J_out[worldid, 0, adr + i] = J_in[worldid, 0, start + i]


@wp.kernel
def _remap_blocks(
  # In:
  count_in: wp.array[int],
  nblock_in: wp.array[int],
  inverse_in: wp.array2d[int],
  adr_in: wp.array2d[int],
  nrow_in: wp.array2d[int],
  # Out:
  adr_out: wp.array2d[int],
  nblock_out: wp.array[int],
):
  worldid, block = wp.tid()
  if block == 0:
    nblock_out[worldid] = wp.where(count_in[worldid] > 0, nblock_in[worldid], 0)
  if count_in[worldid] == 0 or block >= nblock_in[worldid]:
    return
  start = adr_in[worldid, block]
  first = inverse_in[worldid, start]
  for i in range(1, nrow_in[worldid, block]):
    first = wp.min(first, inverse_in[worldid, start + i])
  adr_out[worldid, block] = first


def permutation(count: wp.array, capacity: int, keys: wp.array | _ConstraintKeys.cls, compare: Callable) -> wp.array:
  """Stable parallel mergesort, bounded by capacity and compatible with CUDA capture."""
  order = wp.empty((count.size, capacity), dtype=int)
  scratch = wp.empty_like(order)
  wp.launch(_identity, dim=order.shape, outputs=[order])
  width = 1
  while width < capacity:
    wp.launch(_merge(compare), dim=order.shape, inputs=[count, keys, order, width], outputs=[scratch])
    order, scratch = scratch, order
    width *= 2
  return order


def row_view(value: wp.array, nworld: int, capacity: int, dtype=None) -> wp.array:
  """View a contiguous array as scalar rows without moving its storage."""
  scalar = (
    value.dtype._wp_scalar_type_
    if wp.types.type_is_vector(value.dtype) or wp.types.type_is_matrix(value.dtype)
    else value.dtype
  )
  view = value.view(dtype or scalar)
  width = view.size // max(nworld * capacity, 1)
  # Keep the backing allocation alive when a captured gather retains the view.
  return view.reshape((nworld, capacity, width))


def gather(count: wp.array, order: wp.array, value: wp.array, capacity: int):
  """Permute active rows in place, preserving array references and unused padding."""
  source = wp.clone(value)
  source_view = row_view(source, count.size, capacity)
  dest_view = row_view(value, count.size, capacity)
  wp.launch(_gather, dim=dest_view.shape, inputs=[count, order, source_view], outputs=[dest_view])


def contact_permutation(d: types.Data) -> wp.array:
  """Order by world, features, geometry, then material; never by an allocated contact index.

  Equal keys describe interchangeable contacts. This only orders the retained contacts on overflow;
  it cannot recover contacts dropped by collision detection.
  """
  fields = []
  offset = 0
  for value, is_float, feature in (
    (d.contact.worldid, False, False),
    (d.contact.geom, False, False),
    (d.contact.flex, False, True),
    (d.contact.elem, False, True),
    (d.contact.vert, False, True),
    (d.contact.pos, True, False),
    (d.contact.frame, True, False),
    (d.contact.dist, True, False),
    (d.contact.dim, False, False),
    (d.contact.type, False, False),
    (d.contact.includemargin, True, False),
    (d.contact.friction, True, False),
    (d.contact.solref, True, False),
    (d.contact.solreffriction, True, False),
    (d.contact.solimp, True, False),
    (d.contact.adhesion, True, False),
    (d.contact.geomcollisionid, False, False),
  ):
    if not value.size:
      continue
    view = row_view(value, 1, d.naconmax, wp.uint32)
    fields.append((view, offset, is_float, feature))
    offset += view.shape[2]
  keys = wp.empty((1, d.naconmax, offset), dtype=wp.uint32)
  for view, offset, is_float, feature in fields:
    wp.launch(
      _contact_key_field(is_float, feature),
      dim=(d.naconmax, view.shape[2]),
      inputs=[d.nacon, d.contact.geom, view, offset],
      outputs=[keys],
    )
  return permutation(d.nacon, d.naconmax, keys, _contact_less)


def constraints(m: types.Model, d: types.Data, nnz: wp.array | None = None):
  """Sort assembled rows and repair their contact, sparse-J, and JTDAJ references."""
  count = wp.empty_like(d.nefc)
  wp.launch(
    _constraint_counts,
    dim=d.nworld,
    inputs=[d.nefc, d.njmax, d.njmax_nnz, nnz if m.is_sparse and nnz is not None else wp.empty(0, dtype=int)],
    outputs=[count],
  )
  keys = _ConstraintKeys()
  keys.primary = wp.empty((d.nworld, d.njmax), dtype=wp.vec4i)
  keys.J = d.efc.J.view(wp.uint32)
  keys.J_rownnz = d.efc.J_rownnz
  keys.J_rowadr = d.efc.J_rowadr
  keys.J_colind = d.efc.J_colind
  keys.pos = d.efc.pos.view(wp.uint32)
  keys.margin = d.efc.margin.view(wp.uint32)
  keys.D = d.efc.D.view(wp.uint32)
  keys.vel = d.efc.vel.view(wp.uint32)
  keys.aref = d.efc.aref.view(wp.uint32)
  keys.frictionloss = d.efc.frictionloss.view(wp.uint32)
  keys.nv = m.nv
  wp.launch(
    _constraint_primary, dim=keys.primary.shape, inputs=[m.eq_type, count, d.efc.type, d.efc.id], outputs=[keys.primary]
  )
  # CONTACTS may be set with collision disabled or standalone make_constraint calls, so it does
  # not establish that contacts are already in canonical order.
  if d.naconmax:
    contact_order = contact_permutation(d)
  else:
    contact_order = wp.empty((1, d.naconmax), dtype=int)
    wp.launch(_identity, dim=contact_order.shape, outputs=[contact_order])
  wp.launch(
    _contact_constraint_primary,
    dim=d.contact.efc_address.shape,
    inputs=[
      d.nacon,
      count,
      d.contact.worldid,
      d.contact.efc_address,
      contact_order,
      d.efc.type,
      d.efc.id,
      d.contact.dim,
      m.opt.cone == types.ConeType.PYRAMIDAL,
    ],
    outputs=[keys.primary],
  )
  order = permutation(count, d.njmax, keys, _constraint_less)
  inverse = wp.empty_like(order)
  wp.launch(_inverse, dim=order.shape, inputs=[order], outputs=[inverse])
  wp.launch(
    _remap_contacts,
    dim=d.contact.efc_address.shape,
    inputs=[d.nacon, count, d.contact.worldid, inverse, d.contact.efc_address],
    outputs=[d.contact.efc_address],
  )
  if m.is_sparse:
    offsets = wp.empty_like(order)
    wp.launch(_sparse_offsets, dim=d.nworld, inputs=[count, order, d.efc.J_rownnz], outputs=[offsets])
    wp.launch(
      _pack_sparse,
      dim=order.shape,
      inputs=[
        count,
        order,
        offsets,
        wp.clone(d.efc.J_rownnz),
        wp.clone(d.efc.J_rowadr),
        wp.clone(d.efc.J_colind),
        wp.clone(d.efc.J),
      ],
      outputs=[d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J],
    )
    if m.opt.solver == types.SolverType.NEWTON:
      nblock = wp.zeros_like(d.efc.jtdaj_nblock)
      wp.launch(
        _remap_blocks,
        dim=order.shape,
        inputs=[count, d.efc.jtdaj_nblock, inverse, d.efc.jtdaj_adr, d.efc.jtdaj_nrow],
        outputs=[d.efc.jtdaj_adr, nblock],
      )
      blocks = permutation(nblock, d.njmax, d.efc.jtdaj_adr, _block_less)
      gather(nblock, blocks, d.efc.jtdaj_adr, d.njmax)
      gather(nblock, blocks, d.efc.jtdaj_nrow, d.njmax)
  else:
    gather(count, order, d.efc.J, d.efc.J.shape[1])
  for value in (
    d.efc.type,
    d.efc.id,
    d.efc.pos,
    d.efc.margin,
    d.efc.D,
    d.efc.vel,
    d.efc.aref,
    d.efc.frictionloss,
    d.efc.force,
    d.efc.state,
    d.efc.island,
    d.efc.Jqvel,
  ):
    if value.size:
      gather(count, order, value, value.shape[1])
