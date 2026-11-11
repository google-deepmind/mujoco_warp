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

from typing import Callable

import warp as wp

from mujoco_warp._src import types
from mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"default_grid_stride": False})

# One launch orders blocks of elements by counting, and the merge passes only start at the width of
# a block. Each launch costs more than the comparisons of a small batch, which sorts wide blocks.
# The comparisons grow with the block and with the batch, so a large batch sorts narrow blocks.
_RANK_WIDTH = 64
_RANK_WIDTH_LARGE = 16
_RANK_LARGE = 65536

# Contacts are ranked within per-world buckets while the buffer holds at most this many contacts per
# world. Larger buffers use the merge passes, whose cost does not grow with the square of a bucket.
_BUCKET_NCONMAX = 256

# Contact rows take the order of their contact from one launch that walks the rows of its world,
# while a world holds at most this many rows. Unsorted contacts cost that launch the square of the
# rows of a world, so larger worlds use the merge passes over the contact buffer instead.
_ROW_RANK_NJMAX = 256

# The rows that precede the contact rows of a world are ordered as one block, without merge passes,
# while a world holds at most this many rows. Every row of a world is ordered with them when its
# contact rows are not in place, which costs the square of the rows of the world.
_HEAD_NJMAX = 256

# What `_contact_row_count` knows of a contact, in one value: its rows, whether it is frictionless,
# so that its row precedes those of the other contacts, and whether it is the first of its world.
_ROWS_MASK = 255
_ROWS_FRICTIONLESS = 256
_ROWS_FIRST = 512

# Threads per world that order its leading rows as one block. Each strides over the rows in use: a
# launch with a thread per row of the capacity mostly starts threads that find no row.
_ROW_LANES = 16

# Adjacent columns of a wide row that one thread moves.
_CHUNK = 4

# A Jacobian whose capacity is larger than this many chunks is moved by threads that stride over
# the chunks in use, so that a launch does not start a thread per chunk of the capacity, most of
# which would find nothing to move: this many threads per world for the nonzeros of a sparse
# Jacobian whose worlds hold more chunks, and the threads that move the row fields for the dense
# rows of a batch that holds more chunks.
_LANES = 64

# JTDAJ blocks take their order from one launch that counts the blocks of their world that precede
# them, while a world holds at most this many rows. The count costs the square of the blocks of a
# world, so larger worlds sort their blocks with merge passes.
_BLOCK_RANK_NJMAX = 256

# The nonzero offsets of all worlds come from one scan while its running total fits an int.
_SCAN_MAX = 2**31


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
class _ContactRows:
  dist: wp.array[float]
  pos: wp.array[wp.vec3]
  frame: wp.array[wp.mat33]
  includemargin: wp.array[float]
  friction: wp.array[types.vec5]
  solref: wp.array[wp.vec2]
  solreffriction: wp.array[wp.vec2]
  solimp: wp.array[types.vec5]
  dim: wp.array[int]
  geom: wp.array[wp.vec2i]
  flex: wp.array[wp.vec2i]
  elem: wp.array[wp.vec2i]
  vert: wp.array[wp.vec2i]
  efc_address: wp.array2d[int]
  worldid: wp.array[int]
  type: wp.array[int]
  geomcollisionid: wp.array[int]
  adhesion: wp.array[float]
  surfacevel: wp.array[wp.vec3]


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


@wp.struct
class _ConstraintRows:
  type: wp.array2d[int]
  id: wp.array2d[int]
  pos: wp.array2d[float]
  margin: wp.array2d[float]
  D: wp.array2d[float]
  vel: wp.array2d[float]
  aref: wp.array2d[float]
  frictionloss: wp.array2d[float]
  force: wp.array2d[float]
  state: wp.array2d[int]
  island: wp.array2d[int]
  Jqvel: wp.array2d[float]
  J_rownnz: wp.array2d[int]


@wp.struct
class _BlockKeys:
  adr: wp.array2d[int]
  inverse: wp.array2d[int]


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
def _block_less(keys: _BlockKeys, worldid: int, a: int, b: int) -> bool:
  # The rows of a block share a primary key up to their component, so its head row stays first.
  return keys.inverse[worldid, keys.adr[worldid, a]] < keys.inverse[worldid, keys.adr[worldid, b]]


@cache_kernel
def _rank(compare: Callable):
  key_type = compare.input_types["keys"]

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # In:
    count_in: wp.array[int],
    keys_in: key_type,
    width: int,
    # Out:
    order_out: wp.array2d[int],
    inverse_out: wp.array2d[int],
  ):
    worldid, i = wp.tid()
    count = wp.min(count_in[worldid], order_out.shape[1])
    if i >= count:
      order_out[worldid, i] = i
      if inverse_out.shape[1] > 0:
        inverse_out[worldid, i] = i
      return
    start = (i // width) * width
    end = wp.min(start + width, count)
    rank = start
    # Stable order of *identical* keys, not a key based on emission order: an equal element only
    # precedes this one from a lower index.
    for other in range(start, end):
      if other < i:
        if not compare(keys_in, worldid, i, other):
          rank += 1
      elif other > i:
        if compare(keys_in, worldid, other, i):
          rank += 1
    order_out[worldid, rank] = i
    if inverse_out.shape[1] > 0:
      inverse_out[worldid, i] = rank

  return kernel


@cache_kernel
def _merge(compare: Callable):
  key_type = compare.input_types["keys"]

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
  def kernel(
    # In:
    count_in: wp.array[int],
    keys_in: key_type,
    order_in: wp.array2d[int],
    width: int,
    # Out:
    order_out: wp.array2d[int],
    inverse_out: wp.array2d[int],
  ):
    worldid, i = wp.tid()
    count = wp.min(count_in[worldid], order_in.shape[1])
    if i >= count:
      order_out[worldid, i] = i
      if inverse_out.shape[1] > 0:
        inverse_out[worldid, i] = i
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
    if inverse_out.shape[1] > 0:
      inverse_out[worldid, source] = i - middle + lo

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
  # A few adjacent columns per thread: neighbouring threads then read and write neighbouring memory,
  # which a thread that walks a whole row, next to threads that walk other rows, does not.
  worldid, row, chunk = wp.tid()
  if row < count_in[worldid]:
    source = order_in[worldid, row]
    # Rows that are already in place do not move: with sorted contacts that is most of them.
    if source != row:
      for col in range(_CHUNK * chunk, wp.min(_CHUNK * chunk + _CHUNK, value_in.shape[2])):
        value_out[worldid, row, col] = value_in[worldid, source, col]


@wp.kernel
def _restore(
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  value_in: wp.array3d[wp.uint32],
  # Out:
  value_out: wp.array3d[wp.uint32],
):
  worldid, row, chunk = wp.tid()
  if row < count_in[worldid]:
    if order_in[worldid, row] != row:
      for col in range(_CHUNK * chunk, wp.min(_CHUNK * chunk + _CHUNK, value_in.shape[2])):
        value_out[worldid, row, col] = value_in[worldid, row, col]


@wp.func
def _dense_lane(
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  value_in: wp.array3d[wp.uint32],
  restore: bool,
  worldid: int,
  lane: int,
  stride: int,
  # Out:
  value_out: wp.array3d[wp.uint32],
):
  """Copy the sorted rows out like `_gather`, or restore the copy like `_restore`.

  One of the threads of a world that stride over the chunks of its rows in use: neighbouring threads
  move neighbouring chunks, and no thread is launched for the rows that are not in use.
  """
  nchunk = (value_in.shape[2] + _CHUNK - 1) // _CHUNK
  for slot in range(lane, count_in[worldid] * nchunk, stride):
    row = slot // nchunk
    source = order_in[worldid, row]
    # Rows that are already in place do not move.
    if source != row:
      if restore:
        source = row
      first = _CHUNK * (slot - row * nchunk)
      for col in range(first, wp.min(first + _CHUNK, value_in.shape[2])):
        value_out[worldid, row, col] = value_in[worldid, source, col]


@wp.func
def _move_contact(
  # In:
  rows_in: _ContactRows,
  source: int,
  target: int,
  # Out:
  rows_out: _ContactRows,
):
  """Copy every field of one contact. Loads and stores of whole values preserve their bits."""
  rows_out.dist[target] = rows_in.dist[source]
  rows_out.pos[target] = rows_in.pos[source]
  rows_out.frame[target] = rows_in.frame[source]
  rows_out.includemargin[target] = rows_in.includemargin[source]
  rows_out.friction[target] = rows_in.friction[source]
  rows_out.solref[target] = rows_in.solref[source]
  rows_out.solreffriction[target] = rows_in.solreffriction[source]
  rows_out.solimp[target] = rows_in.solimp[source]
  rows_out.dim[target] = rows_in.dim[source]
  rows_out.geom[target] = rows_in.geom[source]
  # A model without flexes does not allocate the flex feature fields.
  if rows_in.flex.shape[0] > 0:
    rows_out.flex[target] = rows_in.flex[source]
  if rows_in.elem.shape[0] > 0:
    rows_out.elem[target] = rows_in.elem[source]
  if rows_in.vert.shape[0] > 0:
    rows_out.vert[target] = rows_in.vert[source]
  for i in range(rows_in.efc_address.shape[1]):
    rows_out.efc_address[target, i] = rows_in.efc_address[source, i]
  rows_out.worldid[target] = rows_in.worldid[source]
  rows_out.type[target] = rows_in.type[source]
  rows_out.geomcollisionid[target] = rows_in.geomcollisionid[source]
  rows_out.adhesion[target] = rows_in.adhesion[source]
  rows_out.surfacevel[target] = rows_in.surfacevel[source]


@wp.func
def _contact_bucket(worldid: int, nbucket: int) -> int:
  # The outer buckets keep world IDs outside the batch apart from valid worlds.
  return wp.clamp(worldid + 1, 0, nbucket - 1)


@wp.kernel
def _contact_buckets(
  # Data in:
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # Out:
  count_out: wp.array[int],
  slot_out: wp.array[int],
):
  conid = wp.tid()
  if conid >= wp.min(nacon_in[0], slot_out.shape[0]):
    return
  bucket = _contact_bucket(contact_worldid_in[conid], count_out.shape[0])
  # An integer counter: the members of a bucket do not depend on the order in which they arrive,
  # and the slot that a contact takes within its bucket never reaches the result.
  slot_out[conid] = wp.atomic_add(count_out, bucket, 1)  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _contact_group(
  # Data in:
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  offset_in: wp.array[int],
  slot_in: wp.array[int],
  rows_in: _ContactRows,
  # Out:
  source_out: wp.array[int],
  rows_out: _ContactRows,
):
  """Copy the contacts of each world next to each other, in no particular order."""
  conid = wp.tid()
  if conid >= wp.min(nacon_in[0], slot_in.shape[0]):
    return
  target = offset_in[_contact_bucket(contact_worldid_in[conid], offset_in.shape[0])] + slot_in[conid]
  source_out[target] = conid
  _move_contact(rows_in, conid, target, rows_out)


@wp.kernel
def _contact_place(
  # Data in:
  nacon_in: wp.array[int],
  # In:
  keys_in: _ContactKeys,
  offset_in: wp.array[int],
  count_in: wp.array[int],
  source_in: wp.array[int],
  rows_in: _ContactRows,
  # Out:
  rows_out: _ContactRows,
):
  """Move each grouped contact to its canonical position: after the contacts that precede it."""
  slot = wp.tid()
  if slot >= wp.min(nacon_in[0], source_in.shape[0]):
    return
  bucket = _contact_bucket(keys_in.worldid[slot], offset_in.shape[0])
  start = offset_in[bucket]
  rank = start
  conid = source_in[slot]
  # Same stable order as the merge passes: an equal contact only precedes this one from a lower
  # index in the contact buffer, not from a lower slot of the bucket.
  for other in range(start, start + count_in[bucket]):
    if other != slot:
      if source_in[other] < conid:
        if not _contact_less(keys_in, 0, slot, other):
          rank += 1
      elif _contact_less(keys_in, 0, other, slot):
        rank += 1
  _move_contact(rows_in, slot, rank, rows_out)


@wp.func
def _contact_precedes(
  # In:
  keys: _ContactKeys,
  source: wp.array[int],
  geom: wp.array[wp.vec2i],
  pos: wp.array2d[wp.uint32],
  outer: bool,
  a: int,
  b: int,
  gb: wp.vec2i,
) -> bool:
  """Whether the contact in slot a of a bucket precedes the one in slot b in the stable order.

  The grouped copies of the first keys decide most pairs: `gb` is the geom pair of slot b.
  """
  # The contacts of an outer bucket differ in their world IDs, which are not in the grouped copy.
  if not outer:
    ga = geom[a]
    for i in range(2):
      if ga[i] != gb[i]:
        return ga[i] < gb[i]
    # Flex features precede the position of a flex contact.
    if not ((ga[0] < 0 or ga[1] < 0) and keys.flex.shape[0] > 0):
      for i in range(3):
        if pos[a, i] != pos[b, i]:
          return _float_key(pos[a, i]) < _float_key(pos[b, i])
  # Seldom reached: the remaining keys are read where collision detection wrote them, and equal
  # contacts keep the order in which they were written, like the stable merge.
  sa = source[a]
  sb = source[b]
  if _contact_less(keys, 0, sa, sb):
    return True
  if _contact_less(keys, 0, sb, sa):
    return False
  return sa < sb


@wp.kernel
def _contact_group_keys(
  # Data in:
  contact_surfacevel_in: wp.array[wp.vec3],
  nacon_in: wp.array[int],
  # In:
  offset_in: wp.array[int],
  slot_in: wp.array[int],
  worldid_in: wp.array[int],
  geom_in: wp.array[wp.vec2i],
  pos_in: wp.array[wp.vec3],
  # Out:
  source_out: wp.array[int],
  geom_out: wp.array[wp.vec2i],
  pos_out: wp.array[wp.vec3],
  surfacevel_out: wp.array[wp.vec3],
):
  """List the contacts of each world next to each other, with the keys that decide most ranks."""
  conid = wp.tid()
  if conid >= wp.min(nacon_in[0], slot_in.shape[0]):
    return
  target = offset_in[_contact_bucket(worldid_in[conid], offset_in.shape[0])] + slot_in[conid]
  source_out[target] = conid
  geom_out[target] = geom_in[conid]
  pos_out[target] = pos_in[conid]
  # Collision detection wrote no surface velocity: the one in the contact buffer moves with the
  # contact that took its slot, as it does when the contacts are sorted in the contact buffer.
  if contact_surfacevel_in.shape[0] > 0:
    surfacevel_out[conid] = contact_surfacevel_in[conid]


@wp.kernel
def _contact_place_once(
  # Data in:
  nacon_in: wp.array[int],
  # In:
  keys_in: _ContactKeys,
  offset_in: wp.array[int],
  count_in: wp.array[int],
  source_in: wp.array[int],
  geom_in: wp.array[wp.vec2i],
  pos_in: wp.array2d[wp.uint32],
  rows_in: _ContactRows,
  # Out:
  rows_out: _ContactRows,
):
  """Move each contact from where collision detection wrote it to its canonical position."""
  slot = wp.tid()
  if slot >= wp.min(nacon_in[0], source_in.shape[0]):
    return
  conid = source_in[slot]
  nbucket = offset_in.shape[0]
  bucket = _contact_bucket(keys_in.worldid[conid], nbucket)
  outer = bucket == 0 or bucket == nbucket - 1
  start = offset_in[bucket]
  rank = start
  own = geom_in[slot]
  for other in range(start, start + count_in[bucket]):
    if other != slot:
      if _contact_precedes(keys_in, source_in, geom_in, pos_in, outer, other, slot, own):
        rank += 1
  _move_contact(rows_in, conid, rank, rows_out)


@wp.kernel
def _contact_gather(
  # Data in:
  nacon_in: wp.array[int],
  # In:
  order_in: wp.array2d[int],
  rows_in: _ContactRows,
  # Out:
  rows_out: _ContactRows,
):
  conid = wp.tid()
  if conid < wp.min(nacon_in[0], order_in.shape[1]):
    _move_contact(rows_in, order_in[0, conid], conid, rows_out)


@wp.kernel
def _contact_restore(
  # Data in:
  nacon_in: wp.array[int],
  # In:
  rows_in: _ContactRows,
  # Out:
  rows_out: _ContactRows,
):
  conid = wp.tid()
  if conid < wp.min(nacon_in[0], rows_in.worldid.shape[0]):
    _move_contact(rows_in, conid, conid, rows_out)


@wp.func
def _constraint_count(
  # Data in:
  nefc_in: wp.array[int],
  njmax_in: int,
  njmax_nnz_in: int,
  # In:
  nnz_in: wp.array[int],
  worldid: int,
) -> int:
  count = nefc_in[worldid]
  # Overflow may leave partially assembled groups/rows. Leave them and the original counters alone;
  # the existing overflow checks report the error. Canonical ordering requires sufficient buffers.
  if count > njmax_in:
    count = 0
  if nnz_in.shape[0] > 0:
    if nnz_in[worldid] > njmax_nnz_in:
      count = 0
  return count


@wp.kernel
def _contact_unsorted(
  # Data in:
  nacon_in: wp.array[int],
  # In:
  keys_in: _ContactKeys,
  # Out:
  unsorted_out: wp.array[int],
):
  """Count the contacts of a buffer in which some contact precedes its predecessor.

  The count stays zero for a buffer in canonical order, which has no contact left to sort.
  """
  conid = wp.tid()
  if conid > 0 and conid < wp.min(nacon_in[0], keys_in.worldid.shape[0]):
    if _contact_less(keys_in, 0, conid, conid - 1):
      unsorted_out[0] = nacon_in[0]


@wp.func
def _contact_nrow(opt_cone: int, condim: int) -> int:
  """Rows of an active contact."""
  if opt_cone == int(types.ConeType.PYRAMIDAL):
    return wp.max(1, 2 * (condim - 1))
  return condim


@wp.kernel
def _contact_row_count(
  # Model:
  opt_cone: int,
  # Data in:
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_J_rownnz_in: wp.array2d[int],
  nacon_in: wp.array[int],
  # In:
  keys_in: _ContactKeys,
  # Out:
  unsorted_out: wp.array[int],
  rows_out: wp.array[int],
  nnz_out: wp.array[int],
):
  """Rows and nonzeros that each contact allocated, and the count of `_contact_unsorted`."""
  conid = wp.tid()
  if conid >= wp.min(nacon_in[0], rows_out.shape[0]):
    return
  worldid = contact_worldid_in[conid]
  rows = int(0)
  if conid == 0:
    rows = _ROWS_FIRST
  else:
    if _contact_less(keys_in, 0, conid, conid - 1):
      unsorted_out[0] = nacon_in[0]
    if contact_worldid_in[conid - 1] != worldid:
      rows = _ROWS_FIRST
  first = contact_efc_address_in[conid, 0]
  if first >= 0:
    condim = contact_dim_in[conid]
    nrow = _contact_nrow(opt_cone, condim)
    rows += nrow
    if condim == 1:
      rows += _ROWS_FRICTIONLESS
    if nnz_out.shape[0] > 0:
      rownnz = int(0)
      if worldid >= 0 and worldid < efc_J_rownnz_in.shape[0] and first < efc_J_rownnz_in.shape[1]:
        rownnz = efc_J_rownnz_in[worldid, first]
      nnz_out[conid] = nrow * rownnz
  rows_out[conid] = rows


@wp.kernel
def _contact_rows_place(
  # Data in:
  nefc_in: wp.array[int],
  contact_worldid_in: wp.array[int],
  njmax_in: int,
  njmax_nnz_in: int,
  nacon_in: wp.array[int],
  # In:
  nnz_in: wp.array[int],
  nblock_in: wp.array[int],
  unsorted_in: wp.array[int],
  rows_in: wp.array[int],
  contact_nnz_in: wp.array[int],
  # Data out:
  contact_efc_address_out: wp.array2d[int],
  efc_id_out: wp.array2d[int],
  efc_jtdaj_adr_out: wp.array2d[int],
  efc_jtdaj_nrow_out: wp.array2d[int],
  efc_J_rownnz_out: wp.array2d[int],
  efc_J_rowadr_out: wp.array2d[int],
):
  """Give the rows that contacts just allocated, and that are still empty, their canonical address.

  Contacts allocate their rows in the order in which their threads arrive, so the rows would have
  to move once they are filled. In a canonical contact buffer the contacts of a world are
  neighbours: frictionless contacts come first among its rows, then the other contacts, each in the
  order of the buffer. Their nonzeros and their JTDAJ blocks follow that order. The sort that
  follows the assembly still orders every row, and finds these in place.
  """
  conid = wp.tid()
  ncon = wp.min(nacon_in[0], rows_in.shape[0])
  if conid >= ncon or unsorted_in[0] != 0:
    return
  rows = rows_in[conid]
  nrow = rows & _ROWS_MASK
  if nrow == 0:
    return
  worldid = contact_worldid_in[conid]
  if worldid < 0 or worldid >= nefc_in.shape[0]:
    return
  # Rows or nonzeros that did not fit were dropped in the order of arrival: leave such a world.
  nefc = nefc_in[worldid]
  if nefc > njmax_in:
    return
  sparse = contact_nnz_in.shape[0] > 0
  nnz = int(0)
  contact_nnz = int(0)
  if sparse:
    nnz = nnz_in[worldid]
    if nnz > njmax_nnz_in:
      return
    contact_nnz = contact_nnz_in[conid]
  frictionless = (rows & _ROWS_FRICTIONLESS) != 0
  # Rows, nonzeros and contacts with rows that precede this contact, and those of the world.
  before = int(0)
  before_nnz = int(0)
  before_count = int(0)
  total = nrow
  total_nnz = contact_nnz
  total_count = int(1)
  other_rows = rows
  for i in range(conid):
    # The contact that was just counted is the first of the world.
    if (other_rows & _ROWS_FIRST) != 0:
      break
    other = conid - 1 - i
    other_rows = rows_in[other]
    other_nrow = other_rows & _ROWS_MASK
    if other_nrow != 0:
      other_nnz = int(0)
      if sparse:
        other_nnz = contact_nnz_in[other]
      total += other_nrow
      total_nnz += other_nnz
      total_count += 1
      # An earlier contact precedes this one unless it has friction and this one does not.
      if (other_rows & _ROWS_FRICTIONLESS) != 0 or not frictionless:
        before += other_nrow
        before_nnz += other_nnz
        before_count += 1
  for other in range(conid + 1, ncon):
    other_rows = rows_in[other]
    if (other_rows & _ROWS_FIRST) != 0:
      break
    other_nrow = other_rows & _ROWS_MASK
    if other_nrow != 0:
      other_nnz = int(0)
      if sparse:
        other_nnz = contact_nnz_in[other]
      total += other_nrow
      total_nnz += other_nnz
      total_count += 1
      # A later contact only precedes this one when it is frictionless and this one is not.
      if (other_rows & _ROWS_FRICTIONLESS) != 0 and not frictionless:
        before += other_nrow
        before_nnz += other_nnz
        before_count += 1
  # Contacts allocate the last rows, the last nonzeros and the last blocks of their world.
  base = nefc - total
  nnz_base = nnz - total_nnz
  block_base = int(0)
  if nblock_in.shape[0] > 0:
    block_base = nblock_in[worldid] - total_count
  if base < 0 or nnz_base < 0 or block_base < 0:
    return
  target = base + before
  rownnz = contact_nnz / nrow
  for i in range(nrow):
    contact_efc_address_out[conid, i] = target + i
    efc_id_out[worldid, target + i] = conid
    if sparse:
      efc_J_rownnz_out[worldid, target + i] = rownnz
      efc_J_rowadr_out[worldid, target + i] = nnz_base + before_nnz + i * rownnz
  if nblock_in.shape[0] > 0:
    efc_jtdaj_adr_out[worldid, block_base + before_count] = target
    efc_jtdaj_nrow_out[worldid, block_base + before_count] = nrow


@wp.kernel
def _contact_row_rank(
  # Model:
  opt_cone: int,
  # Data in:
  nefc_in: wp.array[int],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  njmax_in: int,
  njmax_nnz_in: int,
  nacon_in: wp.array[int],
  # In:
  keys_in: _ContactKeys,
  nnz_in: wp.array[int],
  unsorted_in: wp.array[int],
  # Out:
  rank_out: wp.array[int],
):
  """Canonical order of each contact among the contacts of its world that have constraint rows."""
  conid = wp.tid()
  if unsorted_in[0] == 0:
    # No contact precedes its predecessor, so the buffer is in canonical order and the index of a
    # contact already is its order: the usual case, when collision detection sorted the contacts.
    # The rows then read the index, not this rank.
    return
  ncon = wp.min(nacon_in[0], rank_out.shape[0])
  if conid >= ncon:
    return
  rank_out[conid] = -1
  worldid = contact_worldid_in[conid]
  if worldid < 0 or worldid >= nefc_in.shape[0]:
    return
  count = _constraint_count(nefc_in, njmax_in, njmax_nnz_in, nnz_in, worldid)
  nrow = contact_dim_in[conid]
  if opt_cone == int(types.ConeType.PYRAMIDAL):
    nrow = wp.max(1, 2 * (nrow - 1))
  rows = int(0)
  for i in range(wp.min(nrow, contact_efc_address_in.shape[1])):
    row = contact_efc_address_in[conid, i]
    if row >= 0 and row < count:
      if efc_type_in[worldid, row] >= int(types.ConstraintType.CONTACT_FRICTIONLESS) and efc_id_in[worldid, row] == conid:
        rows += 1
  if rows == 0:
    return
  # CONTACTS may be set with collision disabled or standalone make_constraint calls, so it does
  # not establish that contacts are already in canonical order: this buffer is not, so compare the
  # contact keys. The rows of the world enumerate its contacts. A contact is counted once per row,
  # which keeps the count increasing along the canonical order, and ties are stable in the contact
  # index like a merge.
  rank = int(0)
  previous = int(-1)
  before = int(0)
  for row in range(count):
    if efc_type_in[worldid, row] >= int(types.ConstraintType.CONTACT_FRICTIONLESS):
      other = efc_id_in[worldid, row]
      if other != previous:
        previous = other
        before = 0
        if other != conid and other >= 0 and other < ncon:
          if contact_worldid_in[other] == worldid:
            if other < conid:
              if not _contact_less(keys_in, worldid, conid, other):
                before = 1
            elif _contact_less(keys_in, worldid, other, conid):
              before = 1
      rank += before
  rank_out[conid] = rank


@wp.kernel
def _constraint_primary(
  # Model:
  opt_cone: int,
  eq_type: wp.array[int],
  # Data in:
  nefc_in: wp.array[int],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_type_in: wp.array2d[int],
  efc_id_in: wp.array2d[int],
  efc_jtdaj_nblock_in: wp.array[int],
  njmax_in: int,
  njmax_nnz_in: int,
  nacon_in: wp.array[int],
  # In:
  nnz_in: wp.array[int],
  unsorted_in: wp.array[int],
  rank_in: wp.array[int],
  # Data out:
  efc_Jqvel_out: wp.array2d[float],
  # Out:
  count_out: wp.array[int],
  nblock_out: wp.array[int],
  keys_out: wp.array2d[wp.vec4i],
  unsorted_out: wp.array[int],
  disorder_out: wp.array[int],
):
  worldid, row = wp.tid()
  count = _constraint_count(nefc_in, njmax_in, njmax_nnz_in, nnz_in, worldid)
  if row == 0:
    count_out[worldid] = count
    # The counts that the next launch raises.
    unsorted_out[worldid] = 0
    disorder_out[worldid] = 0
    if nblock_out.shape[0] > 0:
      nblock_out[worldid] = wp.where(count > 0, efc_jtdaj_nblock_in[worldid], 0)
  if row >= count:
    return
  row_type = efc_type_in[worldid, row]
  id = efc_id_in[worldid, row]
  if row_type < int(types.ConstraintType.CONTACT_FRICTIONLESS):
    efc_Jqvel_out[worldid, row] = 0.0
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
  elif row_type >= int(types.ConstraintType.CONTACT_FRICTIONLESS):
    # A row that its contact addresses is ordered by the rank of that contact and by its component,
    # not by the contact index. Components that the contact does not use cannot claim a row.
    if id >= 0 and id < wp.min(nacon_in[0], rank_in.shape[0]):
      # The index of a contact is its rank in a canonical buffer, which has no rank to read.
      rank = id
      if unsorted_in[0] != 0:
        rank = rank_in[id]
      if contact_worldid_in[id] == worldid and rank >= 0:
        nrow = contact_dim_in[id]
        if opt_cone == int(types.ConeType.PYRAMIDAL):
          nrow = wp.max(1, 2 * (nrow - 1))
        for i in range(wp.min(nrow, contact_efc_address_in.shape[1])):
          if contact_efc_address_in[id, i] == row:
            id = rank
            component = i
            break
  # Force reconstruction requires all connects before welds, regardless of XML equality IDs.
  keys_out[worldid, row] = wp.vec4i(row_type, subtype, id, component)


@wp.kernel
def _constraint_unsorted(
  # In:
  count_in: wp.array[int],
  keys_in: _ConstraintKeys,
  # Out:
  unsorted_out: wp.array[int],
  disorder_out: wp.array[int],
):
  """Count the leading rows of each world that the sort has to order, if any row is out of order.

  Contact rows that end the rows of a world, none of which precedes its predecessor, are in
  canonical order: every other row precedes a contact row. The count stays at the rows that precede
  them, which is every row of a world whose contact rows are not such a tail. A world in which no
  row precedes its predecessor is in canonical order altogether, and is not marked.
  """
  worldid, row = wp.tid()
  count = count_in[worldid]
  if row >= count:
    return
  contact = int(types.ConstraintType.CONTACT_FRICTIONLESS)
  previous_type = int(-1)
  precedes = False
  if row > 0:
    previous_type = keys_in.primary[worldid, row - 1][0]
    precedes = _constraint_less(keys_in, worldid, row, row - 1)
  if precedes:
    disorder_out[worldid] = 1
  # The largest count wins, in whatever order the threads arrive.
  if keys_in.primary[worldid, row][0] < contact:
    # A row that follows a contact row, or a world that does not end in contact rows.
    if previous_type >= contact or row == count - 1:
      wp.atomic_max(unsorted_out, worldid, count)  # kernel_analyzer: ignore[atomic]
  elif previous_type < contact:
    # The first contact row: the rows before it are not known to be in order.
    wp.atomic_max(unsorted_out, worldid, row)  # kernel_analyzer: ignore[atomic]
  elif precedes:
    wp.atomic_max(unsorted_out, worldid, count)  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _constraint_rank(
  # In:
  count_in: wp.array[int],
  disorder_in: wp.array[int],
  keys_in: _ConstraintKeys,
  rows_in: wp.array[int],
  stride: int,
  # Out:
  order_out: wp.array2d[int],
  inverse_out: wp.array2d[int],
  head_out: wp.array[int],
):
  """Order the leading rows of each world as one block, like `_rank` with `_constraint_less`.

  A few threads per world stride over these rows, and each row counts those that precede it. A
  world without a row out of order has no row to order: the count that the later launches read is
  zero. The rows past the count keep their index, up to the rows that the later launches read.
  """
  worldid, lane = wp.tid()
  count = int(0)
  if disorder_in[worldid] != 0:
    count = wp.min(count_in[worldid], order_out.shape[1])
  if lane == 0:
    head_out[worldid] = count
  limit = count
  if rows_in.shape[0] > 0:
    limit = wp.max(count, wp.min(rows_in[worldid], order_out.shape[1]))
  for row in range(lane, limit, stride):
    if row >= count:
      order_out[worldid, row] = row
      inverse_out[worldid, row] = row
    else:
      rank = int(0)
      # Stable order of identical rows: an equal row only precedes this one from a lower index.
      for other in range(count):
        if other < row:
          if not _constraint_less(keys_in, worldid, row, other):
            rank += 1
        elif other > row:
          if _constraint_less(keys_in, worldid, other, row):
            rank += 1
      order_out[worldid, rank] = row
      inverse_out[worldid, row] = rank


@wp.func
def _pack_constraint(
  # In:
  rows_in: _ConstraintRows,
  worldid: int,
  source: int,
  target: int,
  # Out:
  ints_out: wp.array3d[int],
  floats_out: wp.array3d[float],
):
  """Copy every field of one row into a packed copy. Loads and stores of values preserve their bits.

  Fields are the leading dimension of the packed rows, so that the threads of neighbouring rows
  write neighbouring memory.
  """
  ints_out[worldid, 0, target] = rows_in.type[worldid, source]
  ints_out[worldid, 1, target] = rows_in.id[worldid, source]
  ints_out[worldid, 2, target] = rows_in.state[worldid, source]
  ints_out[worldid, 3, target] = rows_in.island[worldid, source]
  # Only a sparse Jacobian has row sizes.
  if rows_in.J_rownnz.shape[1] > 0:
    ints_out[worldid, 4, target] = rows_in.J_rownnz[worldid, source]
  floats_out[worldid, 0, target] = rows_in.pos[worldid, source]
  floats_out[worldid, 1, target] = rows_in.margin[worldid, source]
  floats_out[worldid, 2, target] = rows_in.D[worldid, source]
  floats_out[worldid, 3, target] = rows_in.vel[worldid, source]
  floats_out[worldid, 4, target] = rows_in.aref[worldid, source]
  floats_out[worldid, 5, target] = rows_in.frictionloss[worldid, source]
  floats_out[worldid, 6, target] = rows_in.force[worldid, source]
  floats_out[worldid, 7, target] = rows_in.Jqvel[worldid, source]


@wp.func
def _unpack_constraint(
  # In:
  ints_in: wp.array3d[int],
  floats_in: wp.array3d[float],
  worldid: int,
  row: int,
  # Out:
  rows_out: _ConstraintRows,
):
  rows_out.type[worldid, row] = ints_in[worldid, 0, row]
  rows_out.id[worldid, row] = ints_in[worldid, 1, row]
  rows_out.state[worldid, row] = ints_in[worldid, 2, row]
  rows_out.island[worldid, row] = ints_in[worldid, 3, row]
  if rows_out.J_rownnz.shape[1] > 0:
    rows_out.J_rownnz[worldid, row] = ints_in[worldid, 4, row]
  rows_out.pos[worldid, row] = floats_in[worldid, 0, row]
  rows_out.margin[worldid, row] = floats_in[worldid, 1, row]
  rows_out.D[worldid, row] = floats_in[worldid, 2, row]
  rows_out.vel[worldid, row] = floats_in[worldid, 3, row]
  rows_out.aref[worldid, row] = floats_in[worldid, 4, row]
  rows_out.frictionloss[worldid, row] = floats_in[worldid, 5, row]
  rows_out.force[worldid, row] = floats_in[worldid, 6, row]
  rows_out.Jqvel[worldid, row] = floats_in[worldid, 7, row]


@wp.kernel
def _constraint_gather(
  # Data in:
  efc_J_rownnz_in: wp.array2d[int],
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  rows_in: _ConstraintRows,
  J_in: wp.array3d[wp.uint32],
  # Out:
  ints_out: wp.array3d[int],
  floats_out: wp.array3d[float],
  nnz_out: wp.array2d[int],
  J_out: wp.array3d[wp.uint32],
):
  worldid, row = wp.tid()
  if row < count_in[worldid]:
    source = order_in[worldid, row]
    if nnz_out.shape[1] > 0:
      nnz_out[worldid, row] = efc_J_rownnz_in[worldid, source]
    # Rows that are already in place do not move: with sorted contacts that is most of them.
    if source != row:
      _pack_constraint(rows_in, worldid, source, row, ints_out, floats_out)
  # The threads of a world also stride over the chunks of its dense Jacobian rows.
  if J_in.shape[2] > 0:
    _dense_lane(count_in, order_in, J_in, False, worldid, row, order_in.shape[1], J_out)


@wp.kernel
def _constraint_unpack(
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  ints_in: wp.array3d[int],
  floats_in: wp.array3d[float],
  offsets_in: wp.array2d[int],
  nblock_in: wp.array[int],
  adr_in: wp.array2d[int],
  nrow_in: wp.array2d[int],
  J_in: wp.array3d[wp.uint32],
  # Data out:
  efc_jtdaj_adr_out: wp.array2d[int],
  efc_jtdaj_nrow_out: wp.array2d[int],
  efc_J_rowadr_out: wp.array2d[int],
  # Out:
  rows_out: _ConstraintRows,
  J_out: wp.array3d[wp.uint32],
):
  worldid, row = wp.tid()
  # The threads of a world also stride over the chunks of the dense Jacobian rows that were copied.
  if J_in.shape[2] > 0:
    _dense_lane(count_in, order_in, J_in, True, worldid, row, order_in.shape[1], J_out)
  if row < count_in[worldid]:
    if order_in[worldid, row] != row:
      _unpack_constraint(ints_in, floats_in, worldid, row, rows_out)
    if offsets_in.shape[1] > 0:
      efc_J_rowadr_out[worldid, row] = _sparse_offset(offsets_in, worldid, row)
  if nblock_in.shape[0] > 0:
    if row < nblock_in[worldid]:
      efc_jtdaj_adr_out[worldid, row] = adr_in[worldid, row]
      efc_jtdaj_nrow_out[worldid, row] = nrow_in[worldid, row]


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
  conid = wp.tid()
  if conid < wp.min(nacon_in[0], contact_efc_address_in.shape[0]):
    # Rows past the count did not move, and neither did the rows of a world outside the batch.
    worldid = contact_worldid_in[conid]
    count = int(0)
    if worldid >= 0 and worldid < count_in.shape[0]:
      count = count_in[worldid]
    for component in range(contact_efc_address_in.shape[1]):
      row = contact_efc_address_in[conid, component]
      if row >= 0 and row < count:
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
  nnz_out: wp.array[int],
):
  worldid = wp.tid()
  offset = int(0)
  offsets_out[worldid, 0] = 0
  # Scan assembled rows within each world, not the (possibly interleaved) contact buffer.
  for row in range(count_in[worldid]):
    offsets_out[worldid, row] = offset
    offset += efc_J_rownnz_in[worldid, order_in[worldid, row]]
  nnz_out[worldid] = offset


@wp.func
def _sparse_offset(
  # In:
  offsets_in: wp.array2d[int],
  worldid: int,
  row: int,
) -> int:
  """Address of a sorted row. A scan over all worlds starts each world at its running total."""
  return offsets_in[worldid, row] - offsets_in[worldid, 0]


@wp.func
def _sparse_nnz(
  # In:
  offsets_in: wp.array2d[int],
  nnz_in: wp.array[int],
  worldid: int,
) -> int:
  if nnz_in.shape[0] > 0:
    return nnz_in[worldid]
  # A scan over all worlds continues into the next world, and ends in a row of zeros.
  return offsets_in[worldid + 1, 0] - offsets_in[worldid, 0]


@wp.func
def _sparse_chunk(
  # Data in:
  efc_J_rowadr_in: wp.array2d[int],
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  offsets_in: wp.array2d[int],
  colind_in: wp.array2d[int],
  J_in: wp.array2d[float],
  restore: bool,
  worldid: int,
  first: int,
  last: int,
  # Out:
  colind_out: wp.array2d[int],
  J_out: wp.array2d[float],
):
  """Move the packed elements from first up to last, which all lie within the sorted rows."""
  # Last row that starts at or before the first element: the row that holds it, since the empty
  # rows that share an offset precede it.
  count = count_in[worldid]
  row = int(0)
  hi = count
  while hi - row > 1:
    mid = (row + hi) // 2
    if _sparse_offset(offsets_in, worldid, mid) <= first:
      row = mid
    else:
      hi = mid
  shift = efc_J_rowadr_in[worldid, order_in[worldid, row]] - _sparse_offset(offsets_in, worldid, row)
  for i in range(first, last):
    while row + 1 < count and _sparse_offset(offsets_in, worldid, row + 1) <= i:
      row += 1
      shift = efc_J_rowadr_in[worldid, order_in[worldid, row]] - _sparse_offset(offsets_in, worldid, row)
    # Rows whose nonzeros are already in place do not move: with sorted contacts that is many.
    if shift != 0:
      source = i + wp.where(restore, 0, shift)
      colind_out[worldid, i] = colind_in[worldid, source]
      J_out[worldid, i] = J_in[worldid, source]


@wp.kernel
def _sparse_move(
  # Data in:
  efc_J_rowadr_in: wp.array2d[int],
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  offsets_in: wp.array2d[int],
  nnz_in: wp.array[int],
  colind_in: wp.array2d[int],
  J_in: wp.array2d[float],
  restore: bool,
  # Out:
  colind_out: wp.array2d[int],
  J_out: wp.array2d[float],
):
  """Pack the nonzeros of the sorted rows into a copy, or restore the packed copy.

  One thread per few packed elements rather than per row: neighbouring threads then read and write
  neighbouring memory, and no thread walks a whole row.
  """
  worldid, chunk = wp.tid()
  first = _CHUNK * chunk
  last = wp.min(first + _CHUNK, _sparse_nnz(offsets_in, nnz_in, worldid))
  if first >= last:
    return
  _sparse_chunk(
    efc_J_rowadr_in, count_in, order_in, offsets_in, colind_in, J_in, restore, worldid, first, last, colind_out, J_out
  )


@wp.kernel
def _sparse_unpacked(
  # Data in:
  efc_J_rownnz_in: wp.array2d[int],
  efc_J_rowadr_in: wp.array2d[int],
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  offsets_in: wp.array2d[int],
  # Out:
  unpacked_out: wp.array[int],
):
  """Count the leading packed nonzeros of each world that hold a row that is not in place.

  The nonzeros past that count are where the sorted rows pack them, so they do not move: with
  contact rows in place, those are all but the nonzeros of the few rows that precede them.
  """
  worldid, row = wp.tid()
  if row >= count_in[worldid]:
    return
  source = order_in[worldid, row]
  target = _sparse_offset(offsets_in, worldid, row)
  if efc_J_rowadr_in[worldid, source] != target:
    # The largest count wins, in whatever order the threads arrive.
    wp.atomic_max(unpacked_out, worldid, target + efc_J_rownnz_in[worldid, source])  # kernel_analyzer: ignore[atomic]


@wp.kernel
def _sparse_move_lanes(
  # Data in:
  efc_J_rowadr_in: wp.array2d[int],
  # In:
  count_in: wp.array[int],
  order_in: wp.array2d[int],
  offsets_in: wp.array2d[int],
  nnz_in: wp.array[int],
  colind_in: wp.array2d[int],
  J_in: wp.array2d[float],
  restore: bool,
  stride: int,
  # Out:
  colind_out: wp.array2d[int],
  J_out: wp.array2d[float],
):
  """Pack or restore the nonzeros like `_sparse_move`, with a few threads per world.

  Each thread strides over the chunks of the nonzeros in use: neighbouring threads move neighbouring
  chunks, and no thread is launched for the capacity that is not in use.
  """
  worldid, lane = wp.tid()
  total = _sparse_nnz(offsets_in, nnz_in, worldid)
  for first in range(_CHUNK * lane, total, _CHUNK * stride):
    last = wp.min(first + _CHUNK, total)
    _sparse_chunk(
      efc_J_rowadr_in, count_in, order_in, offsets_in, colind_in, J_in, restore, worldid, first, last, colind_out, J_out
    )


@wp.kernel
def _block_gather(
  # Data in:
  efc_jtdaj_adr_in: wp.array2d[int],
  efc_jtdaj_nrow_in: wp.array2d[int],
  # In:
  nblock_in: wp.array[int],
  order_in: wp.array2d[int],
  inverse_in: wp.array2d[int],
  # Out:
  adr_out: wp.array2d[int],
  nrow_out: wp.array2d[int],
):
  worldid, block = wp.tid()
  if block < nblock_in[worldid]:
    source = order_in[worldid, block]
    adr_out[worldid, block] = inverse_in[worldid, efc_jtdaj_adr_in[worldid, source]]
    nrow_out[worldid, block] = efc_jtdaj_nrow_in[worldid, source]


@wp.kernel
def _block_place(
  # Data in:
  efc_jtdaj_adr_in: wp.array2d[int],
  efc_jtdaj_nrow_in: wp.array2d[int],
  # In:
  nblock_in: wp.array[int],
  inverse_in: wp.array2d[int],
  # Out:
  adr_out: wp.array2d[int],
  nrow_out: wp.array2d[int],
):
  """Order the blocks by the new position of their head row, like merges with `_block_less`."""
  worldid, block = wp.tid()
  nblock = wp.min(nblock_in[worldid], adr_out.shape[1])
  if block >= nblock:
    return
  head = inverse_in[worldid, efc_jtdaj_adr_in[worldid, block]]
  rank = int(0)
  for other in range(nblock):
    other_head = inverse_in[worldid, efc_jtdaj_adr_in[worldid, other]]
    # An equal head only precedes this block from a lower index, like a stable merge.
    if other_head < head:
      rank += 1
    elif other_head == head and other < block:
      rank += 1
  adr_out[worldid, rank] = head
  nrow_out[worldid, rank] = efc_jtdaj_nrow_in[worldid, block]


def _rank_width(nworld: int, capacity: int) -> int:
  """Power of two that one counting launch orders: the whole capacity when it is small."""
  limit = _RANK_WIDTH if nworld * capacity <= _RANK_LARGE else _RANK_WIDTH_LARGE
  width = 1
  while width < capacity and width < limit:
    width *= 2
  return width


def permutation(
  count: wp.array,
  capacity: int,
  keys: wp.array | _ContactKeys.cls | _ConstraintKeys.cls | _BlockKeys.cls,
  compare: Callable,
  inverse: wp.array | None = None,
  width: int | None = None,
) -> wp.array:
  """Stable parallel sort, bounded by capacity and compatible with CUDA capture.

  One launch orders blocks of elements by counting, then merge passes double the sorted width. The
  launches only depend on the capacity. The inverse permutation is written by the last of them.
  The width of the counted blocks is a power of two, chosen from the sizes unless it is given.
  """
  shape = (count.size, capacity)
  order = wp.empty(shape, dtype=int)
  unused = wp.empty((0, 0), dtype=int)
  if width is None:
    width = _rank_width(count.size, capacity)
  last = inverse if inverse is not None and width >= capacity else unused
  wp.launch(_rank(compare), dim=shape, inputs=[count, keys, width], outputs=[order, last])
  if width < capacity:
    scratch = wp.empty_like(order)
    while width < capacity:
      last = inverse if inverse is not None and 2 * width >= capacity else unused
      wp.launch(_merge(compare), dim=shape, inputs=[count, keys, order, width], outputs=[scratch, last])
      order, scratch = scratch, order
      width *= 2
  return order


def row_view(value: wp.array, nworld: int, capacity: int) -> wp.array:
  """View a contiguous array as scalar rows without moving its storage."""
  view = value.view(wp.uint32)
  width = view.size // max(nworld * capacity, 1)
  # Keep the backing allocation alive when a captured gather retains the view.
  return view.reshape((nworld, capacity, width))


def _contact_keys(contact: types.Contact | _ContactRows.cls) -> _ContactKeys.cls:
  keys = _ContactKeys()
  keys.worldid = contact.worldid
  keys.geom = contact.geom
  keys.flex = contact.flex
  keys.elem = contact.elem
  keys.vert = contact.vert
  keys.pos = contact.pos.view(wp.uint32)
  keys.frame = contact.frame.view(wp.uint32)
  keys.dist = contact.dist.view(wp.uint32)
  keys.dim = contact.dim
  keys.type = contact.type
  keys.includemargin = contact.includemargin.view(wp.uint32)
  keys.friction = contact.friction.view(wp.uint32)
  keys.solref = contact.solref.view(wp.uint32)
  keys.solreffriction = contact.solreffriction.view(wp.uint32)
  keys.solimp = contact.solimp.view(wp.uint32)
  keys.adhesion = contact.adhesion.view(wp.uint32)
  keys.surfacevel = contact.surfacevel.view(wp.uint32)
  keys.geomcollisionid = contact.geomcollisionid
  return keys


def _contact_rows(contact: types.Contact) -> _ContactRows.cls:
  rows = _ContactRows()
  rows.dist = contact.dist
  rows.pos = contact.pos
  rows.frame = contact.frame
  rows.includemargin = contact.includemargin
  rows.friction = contact.friction
  rows.solref = contact.solref
  rows.solreffriction = contact.solreffriction
  rows.solimp = contact.solimp
  rows.dim = contact.dim
  rows.geom = contact.geom
  rows.flex = contact.flex
  rows.elem = contact.elem
  rows.vert = contact.vert
  rows.efc_address = contact.efc_address
  rows.worldid = contact.worldid
  rows.type = contact.type
  rows.geomcollisionid = contact.geomcollisionid
  rows.adhesion = contact.adhesion
  rows.surfacevel = contact.surfacevel
  return rows


def _empty_contact_rows(contact: types.Contact) -> _ContactRows.cls:
  rows = _ContactRows()
  rows.dist = wp.empty_like(contact.dist)
  rows.pos = wp.empty_like(contact.pos)
  rows.frame = wp.empty_like(contact.frame)
  rows.includemargin = wp.empty_like(contact.includemargin)
  rows.friction = wp.empty_like(contact.friction)
  rows.solref = wp.empty_like(contact.solref)
  rows.solreffriction = wp.empty_like(contact.solreffriction)
  rows.solimp = wp.empty_like(contact.solimp)
  rows.dim = wp.empty_like(contact.dim)
  rows.geom = wp.empty_like(contact.geom)
  rows.flex = wp.empty_like(contact.flex)
  rows.elem = wp.empty_like(contact.elem)
  rows.vert = wp.empty_like(contact.vert)
  rows.efc_address = wp.empty_like(contact.efc_address)
  rows.worldid = wp.empty_like(contact.worldid)
  rows.type = wp.empty_like(contact.type)
  rows.geomcollisionid = wp.empty_like(contact.geomcollisionid)
  rows.adhesion = wp.empty_like(contact.adhesion)
  rows.surfacevel = wp.empty_like(contact.surfacevel)
  return rows


def contact_permutation(d: types.Data) -> wp.array:
  """Order by world, features, geometry, then material; never by an allocated contact index.

  Equal keys describe interchangeable contacts. This only orders the retained contacts on overflow;
  it cannot recover contacts dropped by collision detection.
  """
  return permutation(d.nacon, d.naconmax, _contact_keys(d.contact), _contact_less)


def contact_buffer(m: types.Model, d: types.Data, incremental: bool = False) -> types.Contact | None:
  """Arrays for collision detection to write its contacts to, instead of `d.contact`.

  `contacts` then places every contact in `d.contact` at its canonical position: each field moves
  once, instead of out of the contact buffer and back. The choice only depends on static sizes
  and options, and the arrays are the same on every replay of a captured step.

  Returns:
    The arrays to pass to the narrowphase and to `contacts`, or None when collision detection has
    to write to `d.contact`.
  """
  if not (m.opt.deterministic & types.DeterminismType.CONTACTS):
    return None
  # Merge passes order the contacts where they are.
  if d.naconmax > _BUCKET_NCONMAX * d.nworld:
    return None
  # An incremental pass appends to the contacts in `d.contact`, and a contact filter reads them
  # there.
  if incremental or m.callback.contactfilter:
    return None
  # Rigid contacts keep the flex features of the slot that they are written to.
  if d.contact.flex.size or d.contact.elem.size or d.contact.vert.size:
    return None
  # Every new contact gets the same addresses (-1), so collision detection writes them in place
  # and they do not move. The flex features are empty arrays.
  return types.Contact(
    dist=wp.empty_like(d.contact.dist),
    pos=wp.empty_like(d.contact.pos),
    frame=wp.empty_like(d.contact.frame),
    includemargin=wp.empty_like(d.contact.includemargin),
    friction=wp.empty_like(d.contact.friction),
    solref=wp.empty_like(d.contact.solref),
    solreffriction=wp.empty_like(d.contact.solreffriction),
    solimp=wp.empty_like(d.contact.solimp),
    dim=wp.empty_like(d.contact.dim),
    geom=wp.empty_like(d.contact.geom),
    flex=d.contact.flex,
    elem=d.contact.elem,
    vert=d.contact.vert,
    efc_address=d.contact.efc_address,
    worldid=wp.empty_like(d.contact.worldid),
    type=wp.empty_like(d.contact.type),
    geomcollisionid=wp.empty_like(d.contact.geomcollisionid),
    adhesion=wp.empty_like(d.contact.adhesion),
    surfacevel=wp.empty_like(d.contact.surfacevel),
  )


def contacts(d: types.Data, source: types.Contact | None = None, surfacevel: bool = True):
  """Sort contacts by `contact_permutation`, preserving references and unused slots.

  Args:
    d: The data that holds the contact buffer.
    source: The arrays of `contact_buffer` that collision detection wrote the contacts to, or
      None to sort the contacts in `d.contact`.
    surfacevel: Whether collision detection wrote `source.surfacevel`. Otherwise the surface
      velocities in `d.contact` move with the contacts that were written to their slots.
  """
  if d.naconmax == 0:
    return
  rows = _contact_rows(d.contact)
  if source is not None:
    # Every contact counts the contacts of its world that precede it and moves from where it was
    # written to its canonical position. The comparisons read a grouped copy of the first keys.
    count = wp.zeros(d.nworld + 2, dtype=int)
    offset = wp.empty(d.nworld + 2, dtype=int)
    slot = wp.empty(d.naconmax, dtype=int)
    order = wp.empty(d.naconmax, dtype=int)
    geom = wp.empty(d.naconmax, dtype=wp.vec2i)
    pos = wp.empty(d.naconmax, dtype=wp.vec3)
    stale = wp.empty(0, dtype=wp.vec3) if surfacevel else d.contact.surfacevel
    source_rows = _contact_rows(source)
    source_rows.efc_address = wp.empty((0, 0), dtype=int)
    wp.launch(_contact_buckets, dim=d.naconmax, inputs=[source.worldid, d.nacon], outputs=[count, slot])
    wp.utils.array_scan(count, offset, False)
    wp.launch(
      _contact_group_keys,
      dim=d.naconmax,
      inputs=[stale, d.nacon, offset, slot, source.worldid, source.geom, source.pos],
      outputs=[order, geom, pos, source.surfacevel],
    )
    wp.launch(
      _contact_place_once,
      dim=d.naconmax,
      inputs=[d.nacon, _contact_keys(source), offset, count, order, geom, pos.view(wp.uint32), source_rows],
      outputs=[rows],
    )
    return
  # Preserve initialized unused-address sentinels along with the other contact fields: every field
  # moves through a copy, with one launch for all of them.
  copy_rows = _empty_contact_rows(d.contact)
  if d.naconmax <= _BUCKET_NCONMAX * d.nworld:
    # Group the contacts of each world, then every contact counts the contacts of its world that
    # precede it: a few launches instead of one merge pass per doubling of the whole buffer, and
    # the comparisons stay within the neighbouring contacts of a world.
    count = wp.zeros(d.nworld + 2, dtype=int)
    offset = wp.empty(d.nworld + 2, dtype=int)
    slot = wp.empty(d.naconmax, dtype=int)
    source = wp.empty(d.naconmax, dtype=int)
    wp.launch(_contact_buckets, dim=d.naconmax, inputs=[d.contact.worldid, d.nacon], outputs=[count, slot])
    wp.utils.array_scan(count, offset, False)
    wp.launch(
      _contact_group,
      dim=d.naconmax,
      inputs=[d.contact.worldid, d.nacon, offset, slot, rows],
      outputs=[source, copy_rows],
    )
    wp.launch(
      _contact_place,
      dim=d.naconmax,
      inputs=[d.nacon, _contact_keys(copy_rows), offset, count, source, copy_rows],
      outputs=[rows],
    )
  else:
    order = contact_permutation(d)
    wp.launch(_contact_gather, dim=d.naconmax, inputs=[d.nacon, order, rows], outputs=[copy_rows])
    wp.launch(_contact_restore, dim=d.naconmax, inputs=[d.nacon, copy_rows], outputs=[rows])


def contact_rows(m: types.Model, d: types.Data, nnz: wp.array | None = None) -> wp.array | None:
  """Move the rows that contacts allocated to their canonical address, before they are filled.

  `constraints` then finds the contact rows, their Jacobian and their nonzeros in place, and only
  has to move the other rows. Call it after the contacts allocated their rows, which must be the
  last rows of each world, and pass the result to `constraints`.

  Returns:
    The count of `_contact_unsorted`, or None when the rows were left where they were allocated.
  """
  if d.njmax == 0 or d.contact.efc_address.size == 0:
    return None
  # Each contact walks the contacts of its world, like the buckets that sort the contacts.
  if d.naconmax > _BUCKET_NCONMAX * d.nworld:
    return None
  # Contacts that collision detection did not sort are seldom in canonical order.
  if not (m.opt.deterministic & types.DeterminismType.CONTACTS):
    return None
  unsorted = wp.zeros(1, dtype=int)
  nnz = nnz if m.is_sparse and nnz is not None else wp.empty(0, dtype=int)
  # Sparse Newton steps keep one JTDAJ block per contact.
  blocks = m.is_sparse and m.opt.solver == types.SolverType.NEWTON
  nblock = d.efc.jtdaj_nblock if blocks else wp.empty(0, dtype=int)
  rows = wp.empty(d.naconmax, dtype=int)
  contact_nnz = wp.empty(d.naconmax if m.is_sparse else 0, dtype=int)
  wp.launch(
    _contact_row_count,
    dim=d.naconmax,
    inputs=[
      m.opt.cone,
      d.contact.dim,
      d.contact.efc_address,
      d.contact.worldid,
      d.efc.J_rownnz,
      d.nacon,
      _contact_keys(d.contact),
    ],
    outputs=[unsorted, rows, contact_nnz],
  )
  wp.launch(
    _contact_rows_place,
    dim=d.naconmax,
    inputs=[d.nefc, d.contact.worldid, d.njmax, d.njmax_nnz, d.nacon, nnz, nblock, unsorted, rows, contact_nnz],
    outputs=[d.contact.efc_address, d.efc.id, d.efc.jtdaj_adr, d.efc.jtdaj_nrow, d.efc.J_rownnz, d.efc.J_rowadr],
  )
  return unsorted


def constraints(m: types.Model, d: types.Data, nnz: wp.array | None = None, unsorted: wp.array | None = None):
  """Sort assembled rows and repair their contact, friction, sparse-J, and JTDAJ references.

  Args:
    m: The model containing kinematic and dynamic information (device).
    d: The data object containing the current state and output arrays (device).
    nnz: Nonzeros that the assembly reserved in each world, for a sparse Jacobian.
    unsorted: The result of `contact_rows`, if it was called for these rows.
  """
  if d.njmax == 0:
    return
  shape = (d.nworld, d.njmax)
  blocks = m.is_sparse and m.opt.solver == types.SolverType.NEWTON
  count = wp.empty_like(d.nefc)
  nblock = wp.empty(d.nworld if blocks else 0, dtype=int)
  nnz = nnz if m.is_sparse and nnz is not None else wp.empty(0, dtype=int)
  rank = wp.empty(d.naconmax, dtype=int)
  if d.contact.efc_address.size:
    contact_keys = _contact_keys(d.contact)
    if unsorted is None:
      unsorted = wp.zeros(1, dtype=int)
      wp.launch(_contact_unsorted, dim=d.naconmax, inputs=[d.nacon, contact_keys], outputs=[unsorted])
    if d.njmax <= _ROW_RANK_NJMAX:
      wp.launch(
        _contact_row_rank,
        dim=d.naconmax,
        inputs=[
          m.opt.cone,
          d.nefc,
          d.contact.dim,
          d.contact.efc_address,
          d.contact.worldid,
          d.efc.type,
          d.efc.id,
          d.njmax,
          d.njmax_nnz,
          d.nacon,
          contact_keys,
          nnz,
          unsorted,
        ],
        outputs=[rank],
      )
    else:
      # The position of each contact in the canonical order of the buffer. The passes sort the
      # unsorted count: none of a buffer in canonical order, whose contacts keep their index.
      permutation(unsorted, d.naconmax, contact_keys, _contact_less, rank.reshape((1, d.naconmax)))
  else:
    # No contact has rows: the rank of every contact is read, and invalid.
    unsorted = wp.ones(1, dtype=int)
    rank.fill_(-1)
  # The rows of each world that have to be ordered, and whether any is out of order: the launch
  # that writes the keys starts both at zero.
  head = wp.empty(d.nworld, dtype=int)
  disorder = wp.empty(d.nworld, dtype=int)
  keys = _ConstraintKeys()
  keys.primary = wp.empty(shape, dtype=wp.vec4i)
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
    _constraint_primary,
    dim=shape,
    inputs=[
      m.opt.cone,
      m.eq_type,
      d.nefc,
      d.contact.dim,
      d.contact.efc_address,
      d.contact.worldid,
      d.efc.type,
      d.efc.id,
      d.efc.jtdaj_nblock,
      d.njmax,
      d.njmax_nnz,
      d.nacon,
      nnz,
      unsorted,
      rank,
    ],
    outputs=[d.efc.Jqvel, count, nblock, keys.primary, head, disorder],
  )
  # With sorted contacts, contact rows usually are in place already and end the rows of their
  # world: the passes then only order the few rows that precede them, and write the identity for
  # the others. Without, nearly every row is ordered and moves, which the launches that are chosen
  # for a few rows per world would do slowly.
  sorted_contacts = bool(m.opt.deterministic & types.DeterminismType.CONTACTS)
  lanes_shape = (d.nworld, min(_ROW_LANES, d.njmax))
  if sorted_contacts:
    wp.launch(_constraint_unsorted, dim=shape, inputs=[count, keys], outputs=[head, disorder])
  else:
    head = count
  inverse = wp.empty(shape, dtype=int)
  if sorted_contacts and d.njmax <= _HEAD_NJMAX:
    # The head usually is a few rows, which one launch orders without merge passes, and none in a
    # world whose rows are all in order.
    order = wp.empty(shape, dtype=int)
    ordered = wp.empty(d.nworld, dtype=int)
    # Sparse rows all take a packed address, so the later launches read all of them.
    read = count if m.is_sparse else wp.empty(0, dtype=int)
    wp.launch(
      _constraint_rank,
      dim=lanes_shape,
      inputs=[head, disorder, keys, read, lanes_shape[1]],
      outputs=[order, inverse, ordered],
    )
    head = ordered
  else:
    # Larger worlds merge wide blocks, whatever the size of the batch.
    order = permutation(head, d.njmax, keys, _constraint_less, inverse, _RANK_WIDTH if sorted_contacts else None)
  # Only the rows of the head have a new address.
  wp.launch(
    _remap_contacts,
    dim=d.naconmax,
    inputs=[d.contact.efc_address, d.contact.worldid, d.nacon, head, inverse],
    outputs=[d.contact.efc_address],
  )
  wp.launch(
    _remap_friction,
    dim=d.dof_frictionadr.shape,
    inputs=[d.nf, d.dof_frictionadr, head, inverse],
    outputs=[d.dof_frictionadr],
  )

  # Every row field moves through one packed copy: one launch out and one launch back, instead of a
  # copy of the whole array and a launch per field.
  rows = _ConstraintRows()
  rows.type = d.efc.type
  rows.id = d.efc.id
  rows.pos = d.efc.pos
  rows.margin = d.efc.margin
  rows.D = d.efc.D
  rows.vel = d.efc.vel
  rows.aref = d.efc.aref
  rows.frictionloss = d.efc.frictionloss
  rows.force = d.efc.force
  rows.state = d.efc.state
  rows.island = d.efc.island
  rows.Jqvel = d.efc.Jqvel
  rows.J_rownnz = d.efc.J_rownnz
  ints = wp.empty((d.nworld, 5, d.njmax), dtype=int)
  floats = wp.empty((d.nworld, 8, d.njmax), dtype=float)
  unused = wp.empty((0, 0), dtype=int)
  offsets, adr, nrow = unused, unused, unused
  # One scan over the nonzeros of the sorted rows of all worlds gives their addresses. Rows past
  # the count of a world hold zero, and so does a last row that ends the scan.
  scan = m.is_sparse and d.nworld * d.njmax_nnz < _SCAN_MAX
  rownnz = wp.zeros((d.nworld + 1, d.njmax), dtype=int) if scan else unused
  # Only the rows that the passes ordered can move. Sparse rows all take a packed address.
  moved = count if m.is_sparse else head
  # The threads that move the row fields of a world also stride over the chunks of its dense
  # Jacobian rows in use, when the batch has more chunks than lanes: two launches with a thread
  # per chunk of the capacity would mostly start threads that find nothing to move, and cost two
  # launches however few chunks a world holds.
  unused_J = wp.empty((0, 0, 0), dtype=wp.uint32)
  J_lanes, J_lanes_sorted = unused_J, unused_J
  dense = not m.is_sparse and d.efc.J.size > 0
  lanes = False
  if dense:
    J_dense = row_view(d.efc.J, d.nworld, d.efc.J.shape[1])
    J_dense_sorted = wp.empty_like(J_dense)
    nchunk = (J_dense.shape[2] + _CHUNK - 1) // _CHUNK
    lanes = sorted_contacts and d.nworld * d.njmax * nchunk > _LANES
    if lanes:
      J_lanes, J_lanes_sorted = J_dense, J_dense_sorted
  wp.launch(
    _constraint_gather,
    dim=shape,
    inputs=[d.efc.J_rownnz, moved, order, rows, J_lanes],
    outputs=[ints, floats, rownnz, J_lanes_sorted],
  )

  if m.is_sparse:
    # The nonzeros are packed in the new row order, so they move through a copy as well.
    total = wp.empty(0, dtype=int)
    if scan:
      offsets = wp.empty_like(rownnz)
      wp.utils.array_scan(rownnz.reshape(-1), offsets.reshape(-1), False)
    else:
      offsets = wp.empty(shape, dtype=int)
      total = wp.empty(d.nworld, dtype=int)
      wp.launch(_sparse_offsets, dim=d.nworld, inputs=[d.efc.J_rownnz, count, order], outputs=[offsets, total])
    nchunk = (d.njmax_nnz + _CHUNK - 1) // _CHUNK
    colind = d.efc.J_colind.reshape((d.nworld, d.njmax_nnz))
    J = d.efc.J.reshape((d.nworld, d.njmax_nnz))
    colind_sorted = wp.empty_like(colind)
    J_sorted = wp.empty_like(J)
    # The threads only look at the packed nonzeros up to the last row that is not in place, which
    # without sorted contacts are nearly all the nonzeros of a world.
    unpacked = total
    if sorted_contacts:
      unpacked = wp.zeros(d.nworld, dtype=int)
      wp.launch(
        _sparse_unpacked,
        dim=shape,
        inputs=[d.efc.J_rownnz, d.efc.J_rowadr, count, order, offsets],
        outputs=[unpacked],
      )
    # The second launch restores the copy before the rows take their new addresses.
    for restore in (False, True):
      inputs = [d.efc.J_rowadr, count, order, offsets, unpacked]
      inputs += [colind_sorted, J_sorted] if restore else [colind, J]
      outputs = [colind, J] if restore else [colind_sorted, J_sorted]
      if sorted_contacts and nchunk > _LANES:
        wp.launch(_sparse_move_lanes, dim=(d.nworld, _LANES), inputs=inputs + [restore, _LANES], outputs=outputs)
      else:
        wp.launch(_sparse_move, dim=(d.nworld, nchunk), inputs=inputs + [restore], outputs=outputs)
    if blocks:
      adr = wp.empty(shape, dtype=int)
      nrow = wp.empty(shape, dtype=int)
      if d.njmax <= _BLOCK_RANK_NJMAX:
        wp.launch(
          _block_place,
          dim=shape,
          inputs=[d.efc.jtdaj_adr, d.efc.jtdaj_nrow, nblock, inverse],
          outputs=[adr, nrow],
        )
      else:
        block_keys = _BlockKeys()
        block_keys.adr = d.efc.jtdaj_adr
        block_keys.inverse = inverse
        block_order = permutation(nblock, d.njmax, block_keys, _block_less)
        wp.launch(
          _block_gather,
          dim=shape,
          inputs=[d.efc.jtdaj_adr, d.efc.jtdaj_nrow, nblock, block_order, inverse],
          outputs=[adr, nrow],
        )
  elif dense and not lanes:
    wp.launch(_gather, dim=(d.nworld, d.njmax, nchunk), inputs=[head, order, J_dense], outputs=[J_dense_sorted])

  wp.launch(
    _constraint_unpack,
    dim=shape,
    inputs=[moved, order, ints, floats, offsets, nblock, adr, nrow, J_lanes_sorted],
    outputs=[d.efc.jtdaj_adr, d.efc.jtdaj_nrow, d.efc.J_rowadr, rows, J_lanes],
  )
  if dense and not lanes:
    wp.launch(_restore, dim=(d.nworld, d.njmax, nchunk), inputs=[head, order, J_dense_sorted], outputs=[J_dense])
