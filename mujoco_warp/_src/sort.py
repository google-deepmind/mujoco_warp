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
class _ContactKeys:
  worldid: wp.array[int]
  geom: wp.array[wp.vec2i]
  flex: wp.array[wp.vec2i]
  elem: wp.array[wp.vec2i]
  vert: wp.array[wp.vec2i]
  pos: wp.array2d[wp.uint32]
  frame: wp.array3d[wp.uint32]
  dist: wp.array[wp.uint32]
  dim: wp.array[int]
  type: wp.array[int]
  includemargin: wp.array[wp.uint32]
  friction: wp.array2d[wp.uint32]
  solref: wp.array2d[wp.uint32]
  solreffriction: wp.array2d[wp.uint32]
  solimp: wp.array2d[wp.uint32]
  adhesion: wp.array[wp.uint32]
  surfacevel: wp.array2d[wp.uint32]
  geomcollisionid: wp.array[int]


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
def _contact_less(keys: _ContactKeys, worldid: int, a: int, b: int) -> bool:
  if keys.worldid[a] != keys.worldid[b]:
    return keys.worldid[a] < keys.worldid[b]
  ga = keys.geom[a]
  gb = keys.geom[b]
  for i in range(2):
    if ga[i] != gb[i]:
      return ga[i] < gb[i]
  # Rigid writers do not initialize flex feature fields; ignore those slots when geom >= 0.
  if (ga[0] < 0 or ga[1] < 0) and keys.flex.shape[0] > 0:
    for i in range(2):
      if ga[i] < 0 and keys.flex[a][i] != keys.flex[b][i]:
        return keys.flex[a][i] < keys.flex[b][i]
    for i in range(2):
      if ga[i] < 0 and keys.elem[a][i] != keys.elem[b][i]:
        return keys.elem[a][i] < keys.elem[b][i]
    for i in range(2):
      if ga[i] < 0 and keys.vert[a][i] != keys.vert[b][i]:
        return keys.vert[a][i] < keys.vert[b][i]
  for i in range(3):
    if keys.pos[a, i] != keys.pos[b, i]:
      return _float_key(keys.pos[a, i]) < _float_key(keys.pos[b, i])
  for i in range(3):
    for j in range(3):
      if keys.frame[a, i, j] != keys.frame[b, i, j]:
        return _float_key(keys.frame[a, i, j]) < _float_key(keys.frame[b, i, j])
  if keys.dist[a] != keys.dist[b]:
    return _float_key(keys.dist[a]) < _float_key(keys.dist[b])
  if keys.dim[a] != keys.dim[b]:
    return keys.dim[a] < keys.dim[b]
  if keys.type[a] != keys.type[b]:
    return keys.type[a] < keys.type[b]
  if keys.includemargin[a] != keys.includemargin[b]:
    return _float_key(keys.includemargin[a]) < _float_key(keys.includemargin[b])
  for i in range(5):
    if keys.friction[a, i] != keys.friction[b, i]:
      return _float_key(keys.friction[a, i]) < _float_key(keys.friction[b, i])
  for i in range(2):
    if keys.solref[a, i] != keys.solref[b, i]:
      return _float_key(keys.solref[a, i]) < _float_key(keys.solref[b, i])
  for i in range(2):
    if keys.solreffriction[a, i] != keys.solreffriction[b, i]:
      return _float_key(keys.solreffriction[a, i]) < _float_key(keys.solreffriction[b, i])
  for i in range(5):
    if keys.solimp[a, i] != keys.solimp[b, i]:
      return _float_key(keys.solimp[a, i]) < _float_key(keys.solimp[b, i])
  if keys.adhesion[a] != keys.adhesion[b]:
    return _float_key(keys.adhesion[a]) < _float_key(keys.adhesion[b])
  for i in range(3):
    if keys.surfacevel[a, i] != keys.surfacevel[b, i]:
      return _float_key(keys.surfacevel[a, i]) < _float_key(keys.surfacevel[b, i])
  return keys.geomcollisionid[a] < keys.geomcollisionid[b]


@wp.func
def _constraint_less(keys: _ConstraintKeys, worldid: int, a: int, b: int) -> bool:
  ka = keys.primary[worldid, a]
  kb = keys.primary[worldid, b]
  for i in range(4):
    if ka[i] != kb[i]:
      return ka[i] < kb[i]

  # Equal type/id is not sufficient for flex edges. Compare actual Jacobian rows, not their atomic
  # allocation addresses. Equality group and contact components were already distinguished above.
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
      if keys.J[worldid, 0, aa + i] != keys.J[worldid, 0, ab + i]:
        return _float_key(keys.J[worldid, 0, aa + i]) < _float_key(keys.J[worldid, 0, ab + i])
    if na != nb:
      return na < nb
  else:
    for i in range(keys.nv):
      if keys.J[worldid, a, i] != keys.J[worldid, b, i]:
        return _float_key(keys.J[worldid, a, i]) < _float_key(keys.J[worldid, b, i])

  if keys.pos[worldid, a] != keys.pos[worldid, b]:
    return _float_key(keys.pos[worldid, a]) < _float_key(keys.pos[worldid, b])
  if keys.margin[worldid, a] != keys.margin[worldid, b]:
    return _float_key(keys.margin[worldid, a]) < _float_key(keys.margin[worldid, b])
  if keys.D[worldid, a] != keys.D[worldid, b]:
    return _float_key(keys.D[worldid, a]) < _float_key(keys.D[worldid, b])
  if keys.vel[worldid, a] != keys.vel[worldid, b]:
    return _float_key(keys.vel[worldid, a]) < _float_key(keys.vel[worldid, b])
  if keys.aref[worldid, a] != keys.aref[worldid, b]:
    return _float_key(keys.aref[worldid, a]) < _float_key(keys.aref[worldid, b])
  if keys.frictionloss[worldid, a] != keys.frictionloss[worldid, b]:
    return _float_key(keys.frictionloss[worldid, a]) < _float_key(keys.frictionloss[worldid, b])
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
    source = order_in[worldid, i]
    while lo < hi:
      mid = (lo + hi) // 2
      other = order_in[worldid, mid]
      lhs = wp.where(left, other, source)
      rhs = wp.where(left, source, other)
      # Stable merge of *identical* keys, not a key based on emission order.
      if compare(keys_in, worldid, lhs, rhs) == left:
        lo = mid + 1
      else:
        hi = mid
    order_out[worldid, i - middle + lo] = source

  return kernel


@wp.kernel
def _gather(
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  value_in: wp.array3d[wp.uint32],
  # Out:
  value_out: wp.array3d[wp.uint32],
):
  worldid, row, col = wp.tid()
  if row < count_in[worldid] and row < order_in.shape[1]:
    value_out[worldid, row, col] = value_in[worldid, order_in[worldid, row], col]


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
  # Data in:
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  # In:
  count_in: wp.array[int],
  # Out:
  keys_out: wp.array2d[wp.vec4i],
):
  worldid, row = wp.tid()
  if row < count_in[worldid]:
    row_type = efc_type_in[worldid, row]
    id = efc_id_in[worldid, row]
    subtype = int(0)
    component = int(0)
    if row_type == int(types.ConstraintType.EQUALITY):
      subtype = eq_type[id]
      if subtype == int(types.EqType.CONNECT) or subtype == int(types.EqType.WELD) or subtype == int(types.EqType.FLEXSTRAIN):
        # Assembly emits each group contiguously: XYZ force then XYZ torque, or the strain modes of
        # a cell. Use the offset within that group, not its nondeterministic allocation address.
        while row > component:
          previous = row - component - 1
          if efc_type_in[worldid, previous] != row_type or efc_id_in[worldid, previous] != id:
            break
          component += 1
    # Force reconstruction requires all connects before welds, regardless of XML equality IDs.
    keys_out[worldid, row] = wp.vec4i(row_type, subtype, id, component)


@wp.kernel
def _contact_constraint_primary(
  # Model:
  opt_cone: int,
  # Data in:
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  nacon_in: wp.array[int],
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  # Out:
  keys_out: wp.array2d[wp.vec4i],
):
  rank, component = wp.tid()
  if rank >= wp.min(nacon_in[0], order_in.shape[1]):
    return
  conid = order_in[0, rank]
  nrow = contact_dim_in[conid]
  if opt_cone == int(types.ConeType.PYRAMIDAL):
    nrow = wp.max(1, 2 * (nrow - 1))
  if component >= nrow:
    return
  worldid = contact_worldid_in[conid]
  row = contact_efc_address_in[conid, component]
  if row >= 0 and row < count_in[worldid]:
    row_type = efc_type_in[worldid, row]
    if row_type >= int(types.ConstraintType.CONTACT_FRICTIONLESS) and efc_id_in[worldid, row] == conid:
      keys_out[worldid, row] = wp.vec4i(row_type, 0, rank, component)


@wp.kernel
def _inverse(order_in: wp.array2d[int], inverse_out: wp.array2d[int]):
  worldid, row = wp.tid()
  inverse_out[worldid, order_in[worldid, row]] = row


@wp.kernel
def _remap_contacts(
  # Data in:
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  count_in: wp.array[int],
  inverse_in: wp.array2d[int],
  # Data out:
  contact_efc_address_out: wp.array2d[int],
):
  conid, component = wp.tid()
  if conid < wp.min(nacon_in[0], contact_efc_address_in.shape[0]):
    row = contact_efc_address_in[conid, component]
    worldid = contact_worldid_in[conid]
    if row >= 0 and row < count_in[worldid]:
      contact_efc_address_out[conid, component] = inverse_in[worldid, row]


@wp.kernel
def _remap_friction(
  # Data in:
  nf_in: wp.array[int],
  dof_frictionadr_in: wp.array2d[int],
  # In:
  count_in: wp.array[int],
  inverse_in: wp.array2d[int],
  # Data out:
  dof_frictionadr_out: wp.array2d[int],
):
  worldid, dofid = wp.tid()
  # Assembly only writes these addresses, and the solver only reads them, with friction rows.
  if nf_in[worldid] == 0:
    return
  row = dof_frictionadr_in[worldid, dofid]
  if row >= 0 and row < count_in[worldid]:
    dof_frictionadr_out[worldid, dofid] = inverse_in[worldid, row]


@wp.kernel
def _sparse_offsets(
  # Data in:
  efc_J_rownnz_in: wp.array2d[int],
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  # Out:
  offsets_out: wp.array2d[int],
):
  worldid = wp.tid()
  offset = int(0)
  # Scan assembled rows within each world, not the (possibly interleaved) contact buffer.
  for row in range(count_in[worldid]):
    offsets_out[worldid, row] = offset
    offset += efc_J_rownnz_in[worldid, order_in[worldid, row]]


@wp.kernel
def _pack_sparse(
  # Data in:
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  efc_J_colind_in: wp.array3d[int],
  efc_J_in: wp.array3d[float],
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  offsets_in: wp.array2d[int],
  # Data out:
  efc_J_rownnz_out: wp.array2d[int],
  efc_J_rowadr_out: wp.array2d[int],
  efc_J_colind_out: wp.array3d[int],
  efc_J_out: wp.array3d[float],
):
  worldid, row = wp.tid()
  if row >= count_in[worldid]:
    return
  old = order_in[worldid, row]
  nnz = efc_J_rownnz_in[worldid, old]
  start = efc_J_rowadr_in[worldid, old]
  adr = offsets_in[worldid, row]
  efc_J_rownnz_out[worldid, row] = nnz
  efc_J_rowadr_out[worldid, row] = adr
  for i in range(nnz):
    efc_J_colind_out[worldid, 0, adr + i] = efc_J_colind_in[worldid, 0, start + i]
    efc_J_out[worldid, 0, adr + i] = efc_J_in[worldid, 0, start + i]


@wp.kernel
def _remap_blocks(
  # Data in:
  efc_jtdaj_adr_in: wp.array2d[int],
  efc_jtdaj_nblock_in: wp.array[int],
  # In:
  count_in: wp.array[int],
  inverse_in: wp.array2d[int],
  # Data out:
  efc_jtdaj_adr_out: wp.array2d[int],
  # Out:
  nblock_out: wp.array[int],
):
  worldid, block = wp.tid()
  if block == 0:
    nblock_out[worldid] = wp.where(count_in[worldid] > 0, efc_jtdaj_nblock_in[worldid], 0)
  if count_in[worldid] == 0 or block >= efc_jtdaj_nblock_in[worldid]:
    return
  # The rows of a block share a primary key up to their component, so its head row stays first.
  efc_jtdaj_adr_out[worldid, block] = inverse_in[worldid, efc_jtdaj_adr_in[worldid, block]]


def permutation(
  count: wp.array, capacity: int, keys: wp.array | _ContactKeys.cls | _ConstraintKeys.cls, compare: Callable
) -> wp.array:
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


def row_view(value: wp.array, nworld: int, capacity: int) -> wp.array:
  """View a contiguous array as scalar rows without moving its storage."""
  view = value.view(wp.uint32)
  width = view.size // max(nworld * capacity, 1)
  # Keep the backing allocation alive when a captured gather retains the view.
  return view.reshape((nworld, capacity, width))


def gather(count: wp.array, order: wp.array, value: wp.array, capacity: int):
  """Permute active rows in place, preserving array references and unused padding."""
  source = row_view(wp.clone(value), count.size, capacity)
  dest = row_view(value, count.size, capacity)
  wp.launch(_gather, dim=dest.shape, inputs=[count, order, source], outputs=[dest])


def contact_permutation(d: types.Data) -> wp.array:
  """Order by world, features, geometry, then material; never by an allocated contact index.

  Equal keys describe interchangeable contacts. This only orders the retained contacts on overflow;
  it cannot recover contacts dropped by collision detection.
  """
  keys = _ContactKeys()
  keys.worldid = d.contact.worldid
  keys.geom = d.contact.geom
  keys.flex = d.contact.flex
  keys.elem = d.contact.elem
  keys.vert = d.contact.vert
  keys.pos = d.contact.pos.view(wp.uint32)
  keys.frame = d.contact.frame.view(wp.uint32)
  keys.dist = d.contact.dist.view(wp.uint32)
  keys.dim = d.contact.dim
  keys.type = d.contact.type
  keys.includemargin = d.contact.includemargin.view(wp.uint32)
  keys.friction = d.contact.friction.view(wp.uint32)
  keys.solref = d.contact.solref.view(wp.uint32)
  keys.solreffriction = d.contact.solreffriction.view(wp.uint32)
  keys.solimp = d.contact.solimp.view(wp.uint32)
  keys.adhesion = d.contact.adhesion.view(wp.uint32)
  keys.surfacevel = d.contact.surfacevel.view(wp.uint32)
  keys.geomcollisionid = d.contact.geomcollisionid
  return permutation(d.nacon, d.naconmax, keys, _contact_less)


def constraints(m: types.Model, d: types.Data, nnz: wp.array | None = None):
  """Sort assembled rows and repair their contact, friction, sparse-J, and JTDAJ references."""
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
    _constraint_primary, dim=keys.primary.shape, inputs=[m.eq_type, d.efc.type, d.efc.id, count], outputs=[keys.primary]
  )
  # CONTACTS may be set with collision disabled or standalone make_constraint calls, so it does
  # not establish that contacts are already in canonical order.
  contact_order = contact_permutation(d)
  wp.launch(
    _contact_constraint_primary,
    dim=d.contact.efc_address.shape,
    inputs=[
      m.opt.cone,
      d.contact.dim,
      d.contact.efc_address,
      d.contact.worldid,
      d.efc.type,
      d.efc.id,
      d.nacon,
      count,
      contact_order,
    ],
    outputs=[keys.primary],
  )
  order = permutation(count, d.njmax, keys, _constraint_less)
  inverse = wp.empty_like(order)
  wp.launch(_inverse, dim=order.shape, inputs=[order], outputs=[inverse])
  wp.launch(
    _remap_contacts,
    dim=d.contact.efc_address.shape,
    inputs=[d.contact.efc_address, d.contact.worldid, d.nacon, count, inverse],
    outputs=[d.contact.efc_address],
  )
  wp.launch(
    _remap_friction,
    dim=d.dof_frictionadr.shape,
    inputs=[d.nf, d.dof_frictionadr, count, inverse],
    outputs=[d.dof_frictionadr],
  )
  if m.is_sparse:
    offsets = wp.empty_like(order)
    wp.launch(_sparse_offsets, dim=d.nworld, inputs=[d.efc.J_rownnz, count, order], outputs=[offsets])
    wp.launch(
      _pack_sparse,
      dim=order.shape,
      inputs=[
        wp.clone(d.efc.J_rownnz),
        wp.clone(d.efc.J_rowadr),
        wp.clone(d.efc.J_colind),
        wp.clone(d.efc.J),
        count,
        order,
        offsets,
      ],
      outputs=[d.efc.J_rownnz, d.efc.J_rowadr, d.efc.J_colind, d.efc.J],
    )
    if m.opt.solver == types.SolverType.NEWTON:
      nblock = wp.zeros_like(d.efc.jtdaj_nblock)
      wp.launch(
        _remap_blocks,
        dim=order.shape,
        inputs=[d.efc.jtdaj_adr, d.efc.jtdaj_nblock, count, inverse],
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
