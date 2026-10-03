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
"""Continuous collision detection (CCD) and candidate generation for deformables.

Matches MuJoCo C engine_collision_continuous.c / engine_collision_continuous.h.
Provides GPU distance primitives (point-triangle, segment-segment, static geom features),
ownership predicates (mjc_ipcOwnsFlexFlex, mjc_ipcOwnsFlexGeom), pair geometry (mjc_pairGap,
mjc_pairVerts, mjc_pairBand, mjc_standoff), swept candidate discovery (mjc_candidates),
and conservative advancement ToI bisection (mjc_advance).
"""

from enum import IntEnum
from typing import Any

import warp as wp

from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"enable_backward": False})


class FlexPairType(IntEnum):
  """Pair types for continuous collision."""

  FLEX_VERT_TRI = 0
  FLEX_EDGE_EDGE = 1
  FLEX_VERT_GEOM = 2
  GEOM_CORNER_TRI = 3
  GEOM_EDGE_EDGE = 4


@wp.func
def geom_supported(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  mesh_graphadr: wp.array[int],
  # In:
  worldid: int,
  gi: int,
) -> bool:
  """Returns True if geom gi is a primitive or hulled mesh supported by IPC (mjc_GeomSupported)."""
  gt = geom_type[gi]
  if gt == int(GeomType.PLANE) or gt == int(GeomType.SPHERE) or gt == int(GeomType.CAPSULE) or gt == int(GeomType.BOX):
    return True
  if gt == int(GeomType.MESH) and mesh_graphadr.shape[0] > 0:
    mid = geom_dataid[worldid % geom_dataid.shape[0], gi]
    return mid >= 0 and mesh_graphadr[mid] >= 0
  return False


@wp.func
def ipc_owns_flex_flex(
  # Model:
  flex_dim: wp.array[int],
  # In:
  f1: int,
  f2: int,
) -> bool:
  """Returns True if IPC handles collision between flexes f1 and f2 (mjc_ipcOwnsFlexFlex)."""
  return flex_dim[f1] == 2 and flex_dim[f2] == 2


@wp.func
def ipc_owns_flex_geom(
  # Model:
  body_weldid: wp.array[int],
  geom_type: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_dataid: wp.array2d[int],
  flex_dim: wp.array[int],
  mesh_graphadr: wp.array[int],
  # In:
  worldid: int,
  f: int,
  gi: int,
) -> bool:
  """Returns True if IPC handles collision between flex f and static geom gi."""
  if flex_dim[f] != 2:
    return False
  b = geom_bodyid[gi]
  if b >= 0 and body_weldid[b] != 0:
    return False
  return geom_supported(geom_type, geom_dataid, mesh_graphadr, worldid, gi)


@wp.func
def standoff(band: float, cap: float) -> float:
  """Computes rest standoff distance from activation band and cap (mjc_standoff)."""
  return wp.min(band, cap)


@wp.func
def feat_radius(gt: int, gsz: wp.vec3) -> float:
  """Returns rounding radius for sphere/capsule hull features (featRadius)."""
  return gsz[0] if (gt == int(GeomType.CAPSULE) or gt == int(GeomType.SPHERE)) else 0.0


@wp.func
def _gap_normal(diff: wp.vec3, d_val: float) -> wp.vec3:
  """Returns unit contact normal or zero vector on degenerate gap (gapNormal)."""
  return diff / d_val if d_val >= 1e-15 else wp.vec3(0.0, 0.0, 0.0)


@wp.func
def pair_vert_range(pt: int) -> tuple[int, int]:
  """Returns (start_offset, num_verts) in pair_idx for a FlexPairType (mjc_pairVerts)."""
  if pt == int(FlexPairType.FLEX_VERT_TRI) or pt == int(FlexPairType.FLEX_EDGE_EDGE):
    return 0, 4
  if pt == int(FlexPairType.FLEX_VERT_GEOM):
    return 0, 1
  if pt == int(FlexPairType.GEOM_CORNER_TRI):
    return 1, 3
  if pt == int(FlexPairType.GEOM_EDGE_EDGE):
    return 1, 2
  return 0, 0


@wp.func
def pair_bo_and_band(
  # In:
  pt: int,
  idx: wp.vec4i,
  rad: wp.array[float],
  ghat: float,
) -> wp.vec2:
  """Returns (midsurface_offset bo, activation_band min(ghat, r_min)) (mjc_pairBand)."""
  if pt == int(FlexPairType.FLEX_VERT_TRI):
    r0 = rad[idx[0]]
    r1 = rad[idx[1]]
    return wp.vec2(r0 + r1, wp.min(ghat, wp.min(r0, r1)))
  if pt == int(FlexPairType.FLEX_EDGE_EDGE):
    r0 = rad[idx[0]]
    r1 = rad[idx[2]]
    return wp.vec2(r0 + r1, wp.min(ghat, wp.min(r0, r1)))
  if pt == int(FlexPairType.FLEX_VERT_GEOM):
    r0 = rad[idx[0]]
    return wp.vec2(r0, wp.min(ghat, r0))
  if pt == int(FlexPairType.GEOM_CORNER_TRI) or pt == int(FlexPairType.GEOM_EDGE_EDGE):
    r0 = rad[idx[1]]
    return wp.vec2(r0, wp.min(ghat, r0))
  return wp.vec2(0.0, 0.0)


@wp.func
def _mask_filtered(c1: int, a1: int, c2: int, a2: int) -> bool:
  """Returns True if contype/conaffinity bitmasks filter out the pair (maskFiltered)."""
  return (c1 & a2) == 0 and (c2 & a1) == 0


@wp.func
def _aabb_separated(a_min: wp.vec3, a_max: wp.vec3, b_min: wp.vec3, b_max: wp.vec3, pad: float) -> bool:
  return (
    a_max[0] + pad < b_min[0]
    or a_min[0] - pad > b_max[0]
    or a_max[1] + pad < b_min[1]
    or a_min[1] - pad > b_max[1]
    or a_max[2] + pad < b_min[2]
    or a_min[2] - pad > b_max[2]
  )


@wp.func
def _body_pair_filtered(
  # Model:
  nexclude: int,
  opt_disableflags: int,
  body_parentid: wp.array[int],
  body_weldid: wp.array[int],
  exclude_signature: wp.array[int],
  # In:
  b1: int,
  b2: int,
) -> bool:
  """Returns True if bodies b1 and b2 are collision-filtered by weld, parent, or exclude rules."""
  if b1 < 0 or b2 < 0:
    return True
  w1 = body_weldid[b1]
  w2 = body_weldid[b2]
  if w1 == w2:
    return True
  if (opt_disableflags & int(DisableBit.FILTERPARENT)) == 0:
    if w1 != 0 and w2 != 0:
      wp1 = body_weldid[body_parentid[w1]]
      wp2 = body_weldid[body_parentid[w2]]
      if w1 == wp2 or w2 == wp1:
        return True
  if nexclude > 0:
    lo = wp.min(b1, b2)
    hi = wp.max(b1, b2)
    sig1 = (lo << 16) + hi
    sig2 = (hi << 16) + lo
    for i in range(nexclude):
      s = exclude_signature[i]
      if s == sig1 or s == sig2:
        return True
  return False


@wp.func
def pt_tri_dist(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
  """Computes point-triangle distance, closest point, and barycentric weights (mjc_PtTri)."""
  ab = b - a
  ac = c - a
  ap = p - a

  d1 = wp.dot(ab, ap)
  d2 = wp.dot(ac, ap)
  if d1 <= 0.0 and d2 <= 0.0:
    return wp.length(p - a), a, wp.vec3(1.0, 0.0, 0.0)

  bp = p - b
  d3 = wp.dot(ab, bp)
  d4 = wp.dot(ac, bp)
  if d3 >= 0.0 and d4 <= d3:
    return wp.length(p - b), b, wp.vec3(0.0, 1.0, 0.0)

  vc = d1 * d4 - d3 * d2
  if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
    t = d1 / (d1 - d3)
    cp = a + ab * t
    return wp.length(p - cp), cp, wp.vec3(1.0 - t, t, 0.0)

  cp_v = p - c
  d5 = wp.dot(ab, cp_v)
  d6 = wp.dot(ac, cp_v)
  if d6 >= 0.0 and d5 <= d6:
    return wp.length(p - c), c, wp.vec3(0.0, 0.0, 1.0)

  vb = d5 * d2 - d1 * d6
  if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
    t = d2 / (d2 - d6)
    cp = a + ac * t
    return wp.length(p - cp), cp, wp.vec3(1.0 - t, 0.0, t)

  va = d3 * d6 - d5 * d4
  if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
    t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    cp = b + (c - b) * t
    return wp.length(p - cp), cp, wp.vec3(0.0, 1.0 - t, t)

  den = 1.0 / (va + vb + vc)
  t_b = vb * den
  u_c = vc * den
  w = wp.vec3(1.0 - t_b - u_c, t_b, u_c)
  cp = a * w[0] + b * w[1] + c * w[2]
  return wp.length(p - cp), cp, w


@wp.func
def seg_seg_dist(p1: wp.vec3, p2: wp.vec3, q1: wp.vec3, q2: wp.vec3):
  """Computes segment-segment distance and closest points (mjc_SegSeg)."""
  d1 = p2 - p1
  d2 = q2 - q1
  rr = p1 - q1

  a = wp.dot(d1, d1)
  e = wp.dot(d2, d2)
  fq = wp.dot(d2, rr)
  c = wp.dot(d1, rr)

  s = float(0.0)
  t = float(0.0)
  if a <= 1e-12 and e <= 1e-12:
    s = 0.0
    t = 0.0
  elif a <= 1e-12:
    s = 0.0
    t = wp.clamp(fq / e, 0.0, 1.0)
  elif e <= 1e-12:
    s = wp.clamp(-c / a, 0.0, 1.0)
    t = 0.0
  else:
    b = wp.dot(d1, d2)
    den = a * e - b * b
    if den > 1e-12:
      s = wp.clamp((b * fq - c * e) / den, 0.0, 1.0)
    else:
      s = 0.0
    t = (b * s + fq) / e
    if t < 0.0:
      t = 0.0
      s = wp.clamp(-c / a, 0.0, 1.0)
    elif t > 1.0:
      t = 1.0
      s = wp.clamp((b - c) / a, 0.0, 1.0)

  cp1 = p1 + d1 * s
  cp2 = q1 + d2 * t
  dist = wp.length(cp1 - cp2)
  return dist, cp1, cp2, s, t


@wp.func
def _closest_on_poly(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  mesh_polyvert: wp.array[int],
  # In:
  pl: wp.vec3,
  va: int,
  pv_adr: int,
  nv_p: int,
  pnl: wp.vec3,
) -> wp.vec3:
  """Closest point on a convex polygon face to pl in mesh-local coordinates (closestOnPoly)."""
  a0 = mesh_vert[va + mesh_polyvert[pv_adr]]
  dpl = wp.dot(pnl, pl - a0)
  pp = pl - pnl * dpl

  npos = int(0)
  nneg = int(0)
  for i in range(nv_p):
    a = mesh_vert[va + mesh_polyvert[pv_adr + i]]
    next_i = (i + 1) if (i + 1) < nv_p else 0
    b = mesh_vert[va + mesh_polyvert[pv_adr + next_i]]
    e = b - a
    w = pp - a
    cr = wp.dot(wp.cross(e, w), pnl)
    if cr > 0.0:
      npos += 1
    elif cr < 0.0:
      nneg += 1

  if npos == 0 or nneg == 0:
    return pp

  best = float(1e30)
  out = pp
  for i in range(nv_p):
    a = mesh_vert[va + mesh_polyvert[pv_adr + i]]
    next_i = (i + 1) if (i + 1) < nv_p else 0
    b = mesh_vert[va + mesh_polyvert[pv_adr + next_i]]
    e = b - a
    w = pl - a
    len_sq = wp.dot(e, e) + 1e-18
    t = wp.clamp(wp.dot(e, w) / len_sq, 0.0, 1.0)
    c = a + e * t
    d2 = wp.dot(pl - c, pl - c)
    if d2 < best:
      best = d2
      out = c
  return out


@wp.func
def geom_dist_mesh(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  # In:
  pl: wp.vec3,
  va: int,
  pa: int,
  pn: int,
  distmax: float,
) -> tuple[float, wp.vec3]:
  """Signed distance and outward normal from static mesh to local point pl."""
  if pn <= 0:
    return float(1e30), wp.vec3(0.0, 0.0, 1.0)
  maxd = float(-1e30)
  bestn = mesh_polynormal[pa]
  for p_idx in range(pn):
    pnl = mesh_polynormal[pa + p_idx]
    pv_adr = mesh_polyvertadr[pa + p_idx]
    v0 = mesh_vert[va + mesh_polyvert[pv_adr]]
    dist_p = wp.dot(pnl, pl - v0)
    if dist_p > maxd:
      maxd = dist_p
      bestn = pnl

  if maxd <= 0.0 or maxd > distmax:
    return maxd, bestn

  best = float(1e30)
  bc = wp.vec3(0.0, 0.0, 0.0)
  for p_idx in range(pn):
    pnl = mesh_polynormal[pa + p_idx]
    pv_adr = mesh_polyvertadr[pa + p_idx]
    nv_p = mesh_polyvertnum[pa + p_idx]
    cc = _closest_on_poly(mesh_vert, mesh_polyvert, pl, va, pv_adr, nv_p, pnl)
    d2 = wp.dot(pl - cc, pl - cc)
    if d2 < best:
      best = d2
      bc = cc

  dist = wp.sqrt(best)
  nl = (pl - bc) / dist if dist > 1e-12 else bestn
  return dist, nl


@wp.func
def geom_point_dist_normal(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  # In:
  gt: int,
  gsz: wp.vec3,
  gp: wp.vec3,
  gR: wp.mat33,
  pos: wp.vec3,
  va: int,
  pa: int,
  pn: int,
  distmax: float,
) -> tuple[float, wp.vec3]:
  """Signed distance and outward world normal from static geom to pos (mjc_GeomDist)."""
  if gt == int(GeomType.PLANE):
    n = wp.vec3(gR[0, 2], gR[1, 2], gR[2, 2])
    return wp.dot(pos - gp, n), n
  if gt == int(GeomType.SPHERE):
    diff = pos - gp
    dist = wp.length(diff)
    return dist - gsz[0], (diff / dist if dist >= 1e-12 else wp.vec3(0.0, 0.0, 1.0))
  if gt == int(GeomType.CAPSULE):
    p_loc = wp.transpose(gR) * (pos - gp)
    zc = wp.clamp(p_loc[2], -gsz[1], gsz[1])
    q = wp.vec3(p_loc[0], p_loc[1], p_loc[2] - zc)
    L = wp.length(q)
    nl = wp.vec3(1.0, 0.0, 0.0) if L < 1e-12 else q / L
    return L - gsz[0], gR * nl
  if gt == int(GeomType.BOX):
    loc = wp.transpose(gR) * (pos - gp)
    px = gsz[0] - wp.abs(loc[0])
    py = gsz[1] - wp.abs(loc[1])
    pz = gsz[2] - wp.abs(loc[2])
    if px < 0.0 or py < 0.0 or pz < 0.0:
      cp = wp.vec3(
        wp.clamp(loc[0], -gsz[0], gsz[0]),
        wp.clamp(loc[1], -gsz[1], gsz[1]),
        wp.clamp(loc[2], -gsz[2], gsz[2]),
      )
      diff = loc - cp
      dd = wp.length(diff)
      nl = diff / dd if dd >= 1e-12 else wp.vec3(0.0, 0.0, 1.0)
    elif px <= py and px <= pz:
      dd = -px
      nl = wp.vec3(1.0 if loc[0] >= 0.0 else -1.0, 0.0, 0.0)
    elif py <= pz:
      dd = -py
      nl = wp.vec3(0.0, 1.0 if loc[1] >= 0.0 else -1.0, 0.0)
    else:
      dd = -pz
      nl = wp.vec3(0.0, 0.0, 1.0 if loc[2] >= 0.0 else -1.0)
    return dd, gR * nl
  if gt == int(GeomType.MESH):
    pl = wp.transpose(gR) * (pos - gp)
    d_mesh, n_mesh = geom_dist_mesh(
      mesh_vert,
      mesh_polynormal,
      mesh_polyvertadr,
      mesh_polyvertnum,
      mesh_polyvert,
      pl,
      va,
      pa,
      pn,
      distmax,
    )
    return d_mesh, gR * n_mesh
  return float(1e30), wp.vec3(0.0, 0.0, 1.0)


@wp.func
def pair_gap(
  # Model:
  mesh_vert: wp.array[wp.vec3],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  # In:
  pt: int,
  idx: wp.vec4i,
  gt: int,
  gsz: wp.vec3,
  feat_rad: float,
  gp: wp.vec3,
  gR: wp.mat33,
  corner: wp.vec3,
  eg0: wp.vec3,
  eg1: wp.vec3,
  p0: wp.vec3,
  p1: wp.vec3,
  p2: wp.vec3,
  p3: wp.vec3,
  va: int,
  pa: int,
  pn: int,
) -> tuple[float, wp.vec3, wp.vec4i, wp.vec4, int]:
  """Evaluates midsurface gap, outward normal, participant vertices, and weights (mjc_pairGap)."""
  if pt == int(FlexPairType.FLEX_VERT_TRI):
    d_val, cp, w_bary = pt_tri_dist(p0, p1, p2, p3)
    return (
      d_val,
      _gap_normal(p0 - cp, d_val),
      idx,
      wp.vec4(1.0, -w_bary[0], -w_bary[1], -w_bary[2]),
      4,
    )
  if pt == int(FlexPairType.FLEX_EDGE_EDGE):
    d_val, cp1, cp2, s_val, t_val = seg_seg_dist(p0, p1, p2, p3)
    return (
      d_val,
      _gap_normal(cp1 - cp2, d_val),
      idx,
      wp.vec4(1.0 - s_val, s_val, -(1.0 - t_val), -t_val),
      4,
    )
  if pt == int(FlexPairType.GEOM_CORNER_TRI):
    d_val, cp, w_bary = pt_tri_dist(corner, p0, p1, p2)
    return (
      d_val - feat_rad,
      _gap_normal(corner - cp, d_val),
      wp.vec4i(idx[1], idx[2], idx[3], 0),
      wp.vec4(-w_bary[0], -w_bary[1], -w_bary[2], 0.0),
      3,
    )
  if pt == int(FlexPairType.GEOM_EDGE_EDGE):
    d_val, cp1, cp2, _s_val, t_val = seg_seg_dist(eg0, eg1, p0, p1)
    return (
      d_val - feat_rad,
      _gap_normal(cp1 - cp2, d_val),
      wp.vec4i(idx[1], idx[2], 0, 0),
      wp.vec4(-(1.0 - t_val), -t_val, 0.0, 0.0),
      2,
    )
  if pt == int(FlexPairType.FLEX_VERT_GEOM):
    dd, n = geom_point_dist_normal(
      mesh_vert,
      mesh_polynormal,
      mesh_polyvertadr,
      mesh_polyvertnum,
      mesh_polyvert,
      gt,
      gsz,
      gp,
      gR,
      p0,
      va,
      pa,
      pn,
      float(1e30),
    )
    return dd, n, wp.vec4i(idx[0], 0, 0, 0), wp.vec4(1.0, 0.0, 0.0, 0.0), 1
  return float(1e30), wp.vec3(0.0, 0.0, 0.0), wp.vec4i(0, 0, 0, 0), wp.vec4(0.0, 0.0, 0.0, 0.0), 0


@wp.func
def unpack_pair_geom(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  geom_corners: wp.array2d[wp.vec3],
  geom_edges: wp.array3d[wp.vec3],
  worldid: int,
  pt: int,
  idx: wp.vec4i,
  gi: int,
) -> tuple[int, wp.vec3, float, wp.vec3, wp.mat33, wp.vec3, wp.vec3, wp.vec3, int, int, int]:
  """Unpacks static geom pose, size, feature radius, mesh addresses, and corner/edge endpoints."""
  gt = int(0)
  gsz = wp.vec3(0.0, 0.0, 0.0)
  feat_rad = float(0.0)
  gp = wp.vec3(0.0, 0.0, 0.0)
  gR = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
  va = int(0)
  pa = int(0)
  pn = int(0)
  if gi >= 0:
    gt = geom_type[gi]
    gsz = geom_size[worldid % geom_size.shape[0], gi]
    feat_rad = feat_radius(gt, gsz)
    gp = geom_xpos_in[worldid, gi]
    gR = geom_xmat_in[worldid, gi]
    if gt == int(GeomType.MESH):
      mid = geom_dataid[worldid % geom_dataid.shape[0], gi]
      va = mesh_vertadr[mid]
      pa = mesh_polyadr[mid]
      pn = mesh_polynum[mid]
  corner = geom_corners[worldid, idx[0]] if pt == int(FlexPairType.GEOM_CORNER_TRI) else wp.vec3(0.0, 0.0, 0.0)
  eg0 = geom_edges[worldid, idx[0], 0] if pt == int(FlexPairType.GEOM_EDGE_EDGE) else wp.vec3(0.0, 0.0, 0.0)
  eg1 = geom_edges[worldid, idx[0], 1] if pt == int(FlexPairType.GEOM_EDGE_EDGE) else wp.vec3(0.0, 0.0, 0.0)
  return gt, gsz, feat_rad, gp, gR, corner, eg0, eg1, va, pa, pn


@wp.func
def eval_pair_gap(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  geom_corners: wp.array2d[wp.vec3],
  geom_edges: wp.array3d[wp.vec3],
  worldid: int,
  pt: int,
  idx: wp.vec4i,
  gi: int,
  p0: wp.vec3,
  p1: wp.vec3,
  p2: wp.vec3,
  p3: wp.vec3,
) -> tuple[float, wp.vec3, wp.vec4i, wp.vec4, int]:
  """Unpacks static geom features and evaluates pair_gap at (p0, p1, p2, p3)."""
  gt, gsz, feat_rad, gp, gR, corner, eg0, eg1, va, pa, pn = unpack_pair_geom(
    geom_type,
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_polynum,
    mesh_polyadr,
    geom_xpos_in,
    geom_xmat_in,
    geom_corners,
    geom_edges,
    worldid,
    pt,
    idx,
    gi,
  )
  return pair_gap(
    mesh_vert,
    mesh_polynormal,
    mesh_polyvertadr,
    mesh_polyvertnum,
    mesh_polyvert,
    pt,
    idx,
    gt,
    gsz,
    feat_rad,
    gp,
    gR,
    corner,
    eg0,
    eg1,
    p0,
    p1,
    p2,
    p3,
    va,
    pa,
    pn,
  )


@wp.kernel
def ipc_init_geom_features_kernel(
  # Model:
  ngeom: int,
  nmesh: int,
  body_weldid: wp.array[int],
  geom_type: wp.array[int],
  geom_contype: wp.array[int],
  geom_conaffinity: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_dataid: wp.array2d[int],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  # Out:
  ngv_out: wp.array[int],
  nge_out: wp.array[int],
  geom_corner_loc_out: wp.array2d[wp.vec3],
  corner_geom_out: wp.array2d[int],
  geom_edge_loc_out: wp.array3d[wp.vec3],
  edge_geom_out: wp.array2d[int],
):
  """Extracts local-space corners and edges of static supported geoms once per model."""
  worldid = wp.tid()
  max_c = geom_corner_loc_out.shape[1]
  max_e = geom_edge_loc_out.shape[1]
  ngv = int(0)
  nge = int(0)

  for gi in range(ngeom):
    if body_weldid[geom_bodyid[gi]] != 0:
      continue
    if geom_contype[gi] == 0 and geom_conaffinity[gi] == 0:
      continue

    gt = geom_type[gi]
    if gt == int(GeomType.SPHERE):
      if ngv < max_c:
        geom_corner_loc_out[worldid, ngv] = wp.vec3(0.0, 0.0, 0.0)
        corner_geom_out[worldid, ngv] = gi
      ngv += 1
    elif gt == int(GeomType.CAPSULE):
      if ngv + 1 < max_c:
        geom_corner_loc_out[worldid, ngv] = wp.vec3(0.0, 0.0, -1.0)
        corner_geom_out[worldid, ngv] = gi
        geom_corner_loc_out[worldid, ngv + 1] = wp.vec3(0.0, 0.0, 1.0)
        corner_geom_out[worldid, ngv + 1] = gi
      ngv += 2
      if nge < max_e:
        geom_edge_loc_out[worldid, nge, 0] = wp.vec3(0.0, 0.0, -1.0)
        geom_edge_loc_out[worldid, nge, 1] = wp.vec3(0.0, 0.0, 1.0)
        edge_geom_out[worldid, nge] = gi
      nge += 1
    elif gt == int(GeomType.BOX):
      if ngv + 7 < max_c:
        idx = int(0)
        for ix in range(2):
          sx = -1.0 if ix == 0 else 1.0
          for iy in range(2):
            sy = -1.0 if iy == 0 else 1.0
            for iz in range(2):
              sz = -1.0 if iz == 0 else 1.0
              geom_corner_loc_out[worldid, ngv + idx] = wp.vec3(sx, sy, sz)
              corner_geom_out[worldid, ngv + idx] = gi
              idx += 1
      ngv += 8
      if nge + 11 < max_e:
        eidx = int(0)
        for axis in range(3):
          for i1 in range(2):
            s1 = -1.0 if i1 == 0 else 1.0
            for i2 in range(2):
              s2 = -1.0 if i2 == 0 else 1.0
              lo = wp.vec3(0.0, 0.0, 0.0)
              hi = wp.vec3(0.0, 0.0, 0.0)
              if axis == 0:
                lo = wp.vec3(-1.0, s1, s2)
                hi = wp.vec3(1.0, s1, s2)
              elif axis == 1:
                lo = wp.vec3(s2, -1.0, s1)
                hi = wp.vec3(s2, 1.0, s1)
              else:
                lo = wp.vec3(s1, s2, -1.0)
                hi = wp.vec3(s1, s2, 1.0)
              geom_edge_loc_out[worldid, nge + eidx, 0] = lo
              geom_edge_loc_out[worldid, nge + eidx, 1] = hi
              edge_geom_out[worldid, nge + eidx] = gi
              eidx += 1
      nge += 12
    elif gt == int(GeomType.MESH) and nmesh > 0:
      mid = geom_dataid[worldid % geom_dataid.shape[0], gi]
      if mid < 0 or mesh_graphadr[mid] < 0:
        continue
      mv_num = mesh_vertnum[mid]
      mv_adr = mesh_vertadr[mid]
      for i in range(mv_num):
        if ngv + i < max_c:
          geom_corner_loc_out[worldid, ngv + i] = mesh_vert[mv_adr + i]
          corner_geom_out[worldid, ngv + i] = gi
      ngv += mv_num
      pn = mesh_polynum[mid]
      pa = mesh_polyadr[mid]
      for p in range(pn):
        adr = mesh_polyvertadr[pa + p]
        nvp = mesh_polyvertnum[pa + p]
        for j in range(nvp):
          a = mesh_polyvert[adr + j]
          next_j = (j + 1) if (j + 1) < nvp else 0
          b = mesh_polyvert[adr + next_j]
          if a < b:
            if nge < max_e:
              geom_edge_loc_out[worldid, nge, 0] = mesh_vert[mv_adr + a]
              geom_edge_loc_out[worldid, nge, 1] = mesh_vert[mv_adr + b]
              edge_geom_out[worldid, nge] = gi
            nge += 1

  ngv_out[worldid] = wp.min(ngv, max_c)
  nge_out[worldid] = wp.min(nge, max_e)


@wp.kernel
def ipc_update_geom_features_kernel(
  # Model:
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  ngv: wp.array[int],
  nge: wp.array[int],
  geom_corner_loc: wp.array2d[wp.vec3],
  corner_geom: wp.array2d[int],
  geom_edge_loc: wp.array3d[wp.vec3],
  edge_geom: wp.array2d[int],
  # Out:
  geom_corners_out: wp.array2d[wp.vec3],
  geom_edges_out: wp.array3d[wp.vec3],
):
  """Transforms static geom corners and edges into world space (mjc_GeomVerts, mjc_GeomEdges)."""
  worldid, idx = wp.tid()
  ngv_w = wp.min(ngv[worldid], geom_corners_out.shape[1])
  nge_w = wp.min(nge[worldid], geom_edges_out.shape[1])

  sz_w = worldid % geom_size.shape[0]
  if idx < ngv_w:
    gi = corner_geom[worldid, idx]
    gt = geom_type[gi]
    gp = geom_xpos_in[worldid, gi]
    gR = geom_xmat_in[worldid, gi]
    loc = geom_corner_loc[worldid, idx]
    if gt == int(GeomType.CAPSULE):
      gsz = geom_size[sz_w, gi]
      loc = wp.vec3(0.0, 0.0, loc[2] * gsz[1])
    elif gt == int(GeomType.BOX):
      gsz = geom_size[sz_w, gi]
      loc = wp.vec3(loc[0] * gsz[0], loc[1] * gsz[1], loc[2] * gsz[2])
    geom_corners_out[worldid, idx] = gp + gR * loc

  if idx < nge_w:
    gi = edge_geom[worldid, idx]
    gt = geom_type[gi]
    gp = geom_xpos_in[worldid, gi]
    gR = geom_xmat_in[worldid, gi]
    loc0 = geom_edge_loc[worldid, idx, 0]
    loc1 = geom_edge_loc[worldid, idx, 1]
    if gt == int(GeomType.CAPSULE):
      gsz = geom_size[sz_w, gi]
      loc0 = wp.vec3(0.0, 0.0, loc0[2] * gsz[1])
      loc1 = wp.vec3(0.0, 0.0, loc1[2] * gsz[1])
    elif gt == int(GeomType.BOX):
      gsz = geom_size[sz_w, gi]
      loc0 = wp.vec3(loc0[0] * gsz[0], loc0[1] * gsz[1], loc0[2] * gsz[2])
      loc1 = wp.vec3(loc1[0] * gsz[0], loc1[1] * gsz[1], loc1[2] * gsz[2])
    geom_edges_out[worldid, idx, 0] = gp + gR * loc0
    geom_edges_out[worldid, idx, 1] = gp + gR * loc1


@wp.kernel
def ipc_update_geom_aabbs_kernel(
  # Model:
  geom_aabb: wp.array3d[wp.vec3],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # Out:
  geom_aabb_min_out: wp.array2d[wp.vec3],
  geom_aabb_max_out: wp.array2d[wp.vec3],
):
  """Computes world-space AABBs for all geoms."""
  worldid, gi = wp.tid()
  gp = geom_xpos_in[worldid, gi]
  gR = geom_xmat_in[worldid, gi]
  aabb_w = worldid % geom_aabb.shape[0]
  la_center = geom_aabb[aabb_w, gi, 0]
  la_half = geom_aabb[aabb_w, gi, 1]
  wc = gp + gR * la_center
  wh = wp.vec3(
    wp.abs(gR[0, 0]) * la_half[0] + wp.abs(gR[0, 1]) * la_half[1] + wp.abs(gR[0, 2]) * la_half[2],
    wp.abs(gR[1, 0]) * la_half[0] + wp.abs(gR[1, 1]) * la_half[1] + wp.abs(gR[1, 2]) * la_half[2],
    wp.abs(gR[2, 0]) * la_half[0] + wp.abs(gR[2, 1]) * la_half[1] + wp.abs(gR[2, 2]) * la_half[2],
  )
  geom_aabb_min_out[worldid, gi] = wc - wh
  geom_aabb_max_out[worldid, gi] = wc + wh


def ipc_update_geom_features(m: Model, d: Data, ws: Any):
  """Updates world-space static geom corners, edges, and AABBs across all worlds."""
  wp.launch(
    ipc_update_geom_features_kernel,
    dim=(d.nworld, max(ws.max_ngv, ws.max_nge)),
    inputs=[
      m.geom_type,
      m.geom_size,
      d.geom_xpos,
      d.geom_xmat,
      ws.ngv,
      ws.nge,
      ws.geom_corner_loc,
      ws.corner_geom,
      ws.geom_edge_loc,
      ws.edge_geom,
    ],
    outputs=[
      ws.geom_corners,
      ws.geom_edges,
    ],
  )
  wp.launch(
    ipc_update_geom_aabbs_kernel,
    dim=(d.nworld, m.ngeom),
    inputs=[
      m.geom_aabb,
      d.geom_xpos,
      d.geom_xmat,
    ],
    outputs=[ws.geom_aabb_min, ws.geom_aabb_max],
  )


@wp.kernel
def ipc_compute_fsweep_kernel(
  # Model:
  flex_dim: wp.array[int],
  flex_vertflexid: wp.array[int],
  # In:
  dfrom: wp.array2d[wp.vec3],
  dto: wp.array2d[wp.vec3],
  world_done: wp.array[int],
  # Out:
  dl_out: wp.array2d[float],
  fsweep_out: wp.array2d[float],
):
  """Computes per-vertex travel dl and per-flex maximum travel fsweep along dfrom -> dto."""
  worldid, v = wp.tid()
  if world_done[worldid] != 0:
    return
  f = flex_vertflexid[v]
  if flex_dim[f] != 2:
    dl_out[worldid, v] = 0.0
    return
  dl_v = wp.length(dto[worldid, v] - dfrom[worldid, v])
  dl_out[worldid, v] = dl_v
  wp.atomic_max(fsweep_out, worldid, f, dl_v)


@wp.func
def _emit_cand(
  # In:
  worldid: int,
  pt: int,
  idx: wp.vec4i,
  gi: int,
  dd: float,
  warn_overflow: bool,
  # Data out:
  overflow_out: wp.array[int],
  # Out:
  ncand_out: wp.array[int],
  cand_type_out: wp.array2d[int],
  cand_idx_out: wp.array2d[wp.vec4i],
  cand_geom_out: wp.array2d[int],
  cand_ld0_out: wp.array2d[float],
):
  c = wp.atomic_add(ncand_out, worldid, 1)
  max_cand = cand_type_out.shape[1]
  if c >= max_cand:
    if warn_overflow and c == max_cand:
      wp.printf(
        "IPC candidate overflow - please increase nconmax beyond %u or naconmax beyond %u\n"
        "To disable the print warning: m.opt.warn_overflow &= ~mjw.OverflowType.BROADPHASE (or = 0 for all)\n",
        max_cand,
        max_cand * ncand_out.shape[0],
      )
    wp.atomic_or(overflow_out, worldid, wp.static(OverflowType.BROADPHASE))
    return
  cand_type_out[worldid, c] = pt
  cand_idx_out[worldid, c] = idx
  cand_geom_out[worldid, c] = gi
  cand_ld0_out[worldid, c] = dd


@wp.func
def _add_cand_geom(
  # In:
  worldid: int,
  pt: int,
  idx: wp.vec4i,
  gi: int,
  dd: float,
  bo: float,
  band: float,
  reach: float,
  dl: wp.array2d[float],
  warn_overflow: bool,
  # Data out:
  overflow_out: wp.array[int],
  # Out:
  ncand_out: wp.array[int],
  cand_type_out: wp.array2d[int],
  cand_idx_out: wp.array2d[wp.vec4i],
  cand_geom_out: wp.array2d[int],
  cand_ld0_out: wp.array2d[float],
):
  """Applies mjc_addCand filtering for flex-geom pairs using precomputed vertex travel dl."""
  if dd >= bo + reach:
    return
  off, nv = pair_vert_range(pt)
  bound = dl[worldid, idx[off]]
  if nv > 1:
    bound = wp.max(bound, dl[worldid, idx[off + 1]])
  if nv > 2:
    bound = wp.max(bound, dl[worldid, idx[off + 2]])
  if dd >= bo + band + bound:
    return
  _emit_cand(
    worldid,
    pt,
    idx,
    gi,
    dd,
    warn_overflow,
    overflow_out,
    ncand_out,
    cand_type_out,
    cand_idx_out,
    cand_geom_out,
    cand_ld0_out,
  )


@wp.func
def _add_cand(
  # In:
  worldid: int,
  pt: int,
  idx: wp.vec4i,
  gi: int,
  dd: float,
  bo: float,
  band: float,
  reach: float,
  dfrom: wp.array2d[wp.vec3],
  dto: wp.array2d[wp.vec3],
  dl: wp.array2d[float],
  warn_overflow: bool,
  # Data out:
  overflow_out: wp.array[int],
  # Out:
  ncand_out: wp.array[int],
  cand_type_out: wp.array2d[int],
  cand_idx_out: wp.array2d[wp.vec4i],
  cand_geom_out: wp.array2d[int],
  cand_ld0_out: wp.array2d[float],
):
  """Applies mjc_addCand filtering and emits flex-flex candidate to cand arrays."""
  if dd >= bo + reach or dd <= 0.0:
    return

  v0 = idx[0]
  v1 = idx[1]
  v2 = idx[2]
  v3 = idx[3]
  dl0 = dl[worldid, v0]
  dl1 = dl[worldid, v1]
  dl2 = dl[worldid, v2]
  dl3 = dl[worldid, v3]
  dl_upper = (
    dl0 + wp.max(dl1, wp.max(dl2, dl3)) if pt == int(FlexPairType.FLEX_VERT_TRI) else wp.max(dl0, dl1) + wp.max(dl2, dl3)
  )
  if dd >= bo + band + dl_upper + 1e-6:
    return

  dv0 = dto[worldid, v0] - dfrom[worldid, v0]
  dv1 = dto[worldid, v1] - dfrom[worldid, v1]
  dv2 = dto[worldid, v2] - dfrom[worldid, v2]
  dv3 = dto[worldid, v3] - dfrom[worldid, v3]

  bound = (
    wp.max(wp.length(dv1 - dv0), wp.max(wp.length(dv2 - dv0), wp.length(dv3 - dv0)))
    if pt == int(FlexPairType.FLEX_VERT_TRI)
    else wp.max(wp.max(wp.length(dv0 - dv2), wp.length(dv0 - dv3)), wp.max(wp.length(dv1 - dv2), wp.length(dv1 - dv3)))
  )
  if dd >= bo + band + bound:
    return

  _emit_cand(
    worldid,
    pt,
    idx,
    gi,
    dd,
    warn_overflow,
    overflow_out,
    ncand_out,
    cand_type_out,
    cand_idx_out,
    cand_geom_out,
    cand_ld0_out,
  )


@cache_kernel
def ipc_cand_vert_geom_kernel(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nexclude: int,
    opt_disableflags: int,
    body_parentid: wp.array[int],
    body_weldid: wp.array[int],
    geom_type: wp.array[int],
    geom_contype: wp.array[int],
    geom_conaffinity: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    flex_contype: wp.array[int],
    flex_conaffinity: wp.array[int],
    flex_dim: wp.array[int],
    mesh_vertadr: wp.array[int],
    mesh_graphadr: wp.array[int],
    mesh_vert: wp.array[wp.vec3],
    mesh_polynum: wp.array[int],
    mesh_polyadr: wp.array[int],
    mesh_polynormal: wp.array[wp.vec3],
    mesh_polyvertadr: wp.array[int],
    mesh_polyvertnum: wp.array[int],
    mesh_polyvert: wp.array[int],
    exclude_signature: wp.array[int],
    flex_vertflexid: wp.array[int],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    # In:
    fidx: wp.array[int],
    pin_has_chain: wp.array[bool],
    rad: wp.array[float],
    pbody: wp.array[int],
    geom_aabb_min: wp.array2d[wp.vec3],
    geom_aabb_max: wp.array2d[wp.vec3],
    world_done: wp.array[int],
    x: wp.array2d[wp.vec3],
    dl: wp.array2d[float],
    thresh: float,
    ghat: float,
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    ncand_out: wp.array[int],
    cand_type_out: wp.array2d[int],
    cand_idx_out: wp.array2d[wp.vec4i],
    cand_geom_out: wp.array2d[int],
    cand_ld0_out: wp.array2d[float],
  ):
    """Discovers FLEX_VERT_GEOM candidates between 2D flex vertices and static geoms."""
    worldid, v, gi = wp.tid()
    if world_done[worldid] != 0:
      return
    if fidx[v] < 0 and not pin_has_chain[v]:
      return
    f = flex_vertflexid[v]
    if not ipc_owns_flex_geom(body_weldid, geom_type, geom_bodyid, geom_dataid, flex_dim, mesh_graphadr, worldid, f, gi):
      return
    fc = flex_contype[f]
    fa = flex_conaffinity[f]
    gc = geom_contype[gi]
    ga = geom_conaffinity[gi]
    if _mask_filtered(fc, fa, gc, ga):
      return

    bid_g = geom_bodyid[gi]
    if _body_pair_filtered(nexclude, opt_disableflags, body_parentid, body_weldid, exclude_signature, pbody[v], bid_g):
      return

    gt = geom_type[gi]
    va = int(0)
    pa = int(0)
    pn = int(0)
    if gt == int(GeomType.MESH):
      mid = geom_dataid[worldid % geom_dataid.shape[0], gi]
      va = mesh_vertadr[mid]
      pa = mesh_polyadr[mid]
      pn = mesh_polynum[mid]

    idx = wp.vec4i(v, 0, 0, 0)
    bo_band = pair_bo_and_band(int(FlexPairType.FLEX_VERT_GEOM), idx, rad, ghat)
    bo = bo_band[0]
    band = bo_band[1]
    reach = thresh + dl[worldid, v]
    pv = x[worldid, v]

    if gt != int(GeomType.PLANE):
      if _aabb_separated(pv, pv, geom_aabb_min[worldid, gi], geom_aabb_max[worldid, gi], bo + reach):
        return

    gsz = geom_size[worldid % geom_size.shape[0], gi]
    gp = geom_xpos_in[worldid, gi]
    gR = geom_xmat_in[worldid, gi]
    dd, _n = geom_point_dist_normal(
      mesh_vert,
      mesh_polynormal,
      mesh_polyvertadr,
      mesh_polyvertnum,
      mesh_polyvert,
      gt,
      gsz,
      gp,
      gR,
      pv,
      va,
      pa,
      pn,
      bo + reach,
    )

    _add_cand_geom(
      worldid,
      int(FlexPairType.FLEX_VERT_GEOM),
      idx,
      gi,
      dd,
      bo,
      band,
      reach,
      dl,
      wp.static(bool(warn_overflow & OverflowType.BROADPHASE)),
      overflow_out,
      ncand_out,
      cand_type_out,
      cand_idx_out,
      cand_geom_out,
      cand_ld0_out,
    )

  return kernel


@cache_kernel
def ipc_cand_corner_tri_kernel(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nexclude: int,
    opt_disableflags: int,
    body_parentid: wp.array[int],
    body_weldid: wp.array[int],
    geom_type: wp.array[int],
    geom_contype: wp.array[int],
    geom_conaffinity: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_size: wp.array2d[wp.vec3],
    flex_contype: wp.array[int],
    flex_conaffinity: wp.array[int],
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_elem: wp.array[int],
    exclude_signature: wp.array[int],
    flex_elemflexid: wp.array[int],
    # In:
    rad: wp.array[float],
    pbody: wp.array[int],
    ngv: wp.array[int],
    geom_corners: wp.array2d[wp.vec3],
    corner_geom: wp.array2d[int],
    world_done: wp.array[int],
    x: wp.array2d[wp.vec3],
    dl: wp.array2d[float],
    fsweep: wp.array2d[float],
    thresh_geom: float,
    ghat: float,
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    ncand_out: wp.array[int],
    cand_type_out: wp.array2d[int],
    cand_idx_out: wp.array2d[wp.vec4i],
    cand_geom_out: wp.array2d[int],
    cand_ld0_out: wp.array2d[float],
  ):
    """Discovers GEOM_CORNER_TRI candidates between static geom corners and 2D flex triangles."""
    worldid, c_i, e = wp.tid()
    if world_done[worldid] != 0 or c_i >= wp.min(ngv[worldid], geom_corners.shape[1]):
      return
    f = flex_elemflexid[e]
    if flex_dim[f] != 2:
      return
    gi = corner_geom[worldid, c_i]
    if _mask_filtered(flex_contype[f], flex_conaffinity[f], geom_contype[gi], geom_conaffinity[gi]):
      return

    eadr = flex_elemdataadr[f] + 3 * (e - flex_elemadr[f])
    va = flex_vertadr[f]
    v0 = va + flex_elem[eadr]
    v1 = va + flex_elem[eadr + 1]
    v2 = va + flex_elem[eadr + 2]

    bid_g = geom_bodyid[gi]
    if (
      _body_pair_filtered(nexclude, opt_disableflags, body_parentid, body_weldid, exclude_signature, pbody[v0], bid_g)
      and _body_pair_filtered(nexclude, opt_disableflags, body_parentid, body_weldid, exclude_signature, pbody[v1], bid_g)
      and _body_pair_filtered(nexclude, opt_disableflags, body_parentid, body_weldid, exclude_signature, pbody[v2], bid_g)
    ):
      return

    idx = wp.vec4i(c_i, v0, v1, v2)
    bo_band = pair_bo_and_band(int(FlexPairType.GEOM_CORNER_TRI), idx, rad, ghat)
    bo = bo_band[0]
    band = bo_band[1]
    reach = thresh_geom + fsweep[worldid, f]
    fr = feat_radius(geom_type[gi], geom_size[worldid % geom_size.shape[0], gi])
    corner = geom_corners[worldid, c_i]
    p1 = x[worldid, v0]
    p2 = x[worldid, v1]
    p3 = x[worldid, v2]

    if _aabb_separated(corner, corner, wp.min(wp.min(p1, p2), p3), wp.max(wp.max(p1, p2), p3), bo + reach + fr):
      return

    d_val, _cp, _w = pt_tri_dist(corner, p1, p2, p3)
    dd = d_val - fr

    _add_cand_geom(
      worldid,
      int(FlexPairType.GEOM_CORNER_TRI),
      idx,
      gi,
      dd,
      bo,
      band,
      reach,
      dl,
      wp.static(bool(warn_overflow & OverflowType.BROADPHASE)),
      overflow_out,
      ncand_out,
      cand_type_out,
      cand_idx_out,
      cand_geom_out,
      cand_ld0_out,
    )

  return kernel


@cache_kernel
def ipc_cand_geom_edge_edge_kernel(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    nexclude: int,
    opt_disableflags: int,
    body_parentid: wp.array[int],
    body_weldid: wp.array[int],
    geom_type: wp.array[int],
    geom_contype: wp.array[int],
    geom_conaffinity: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_size: wp.array2d[wp.vec3],
    flex_contype: wp.array[int],
    flex_conaffinity: wp.array[int],
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemnum: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    exclude_signature: wp.array[int],
    flex_edgeflexid: wp.array[int],
    # In:
    rad: wp.array[float],
    pbody: wp.array[int],
    nge: wp.array[int],
    geom_edges: wp.array3d[wp.vec3],
    edge_geom: wp.array2d[int],
    world_done: wp.array[int],
    x: wp.array2d[wp.vec3],
    dl: wp.array2d[float],
    fsweep: wp.array2d[float],
    thresh_geom: float,
    ghat: float,
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    ncand_out: wp.array[int],
    cand_type_out: wp.array2d[int],
    cand_idx_out: wp.array2d[wp.vec4i],
    cand_geom_out: wp.array2d[int],
    cand_ld0_out: wp.array2d[float],
  ):
    """Discovers GEOM_EDGE_EDGE candidates between static geom edges and 2D flex edges."""
    worldid, e_i, fe = wp.tid()
    if world_done[worldid] != 0 or e_i >= wp.min(nge[worldid], geom_edges.shape[1]):
      return
    f = flex_edgeflexid[fe]
    if flex_dim[f] != 2 or flex_elemnum[f] <= 0:
      return
    gi = edge_geom[worldid, e_i]
    if _mask_filtered(flex_contype[f], flex_conaffinity[f], geom_contype[gi], geom_conaffinity[gi]):
      return

    va = flex_vertadr[f]
    ev = flex_edge[fe]
    v0 = va + ev[0]
    v1 = va + ev[1]

    bid_g = geom_bodyid[gi]
    if _body_pair_filtered(
      nexclude, opt_disableflags, body_parentid, body_weldid, exclude_signature, pbody[v0], bid_g
    ) and _body_pair_filtered(nexclude, opt_disableflags, body_parentid, body_weldid, exclude_signature, pbody[v1], bid_g):
      return

    idx = wp.vec4i(e_i, v0, v1, 0)
    bo_band = pair_bo_and_band(int(FlexPairType.GEOM_EDGE_EDGE), idx, rad, ghat)
    bo = bo_band[0]
    band = bo_band[1]
    reach = thresh_geom + fsweep[worldid, f]
    fr = feat_radius(geom_type[gi], geom_size[worldid % geom_size.shape[0], gi])
    eg0 = geom_edges[worldid, e_i, 0]
    eg1 = geom_edges[worldid, e_i, 1]
    p1 = x[worldid, v0]
    p2 = x[worldid, v1]

    if _aabb_separated(wp.min(eg0, eg1), wp.max(eg0, eg1), wp.min(p1, p2), wp.max(p1, p2), bo + reach + fr):
      return

    d_val, _cp1, _cp2, _s, _t = seg_seg_dist(eg0, eg1, p1, p2)
    dd = d_val - fr

    _add_cand_geom(
      worldid,
      int(FlexPairType.GEOM_EDGE_EDGE),
      idx,
      gi,
      dd,
      bo,
      band,
      reach,
      dl,
      wp.static(bool(warn_overflow & OverflowType.BROADPHASE)),
      overflow_out,
      ncand_out,
      cand_type_out,
      cand_idx_out,
      cand_geom_out,
      cand_ld0_out,
    )

  return kernel


@cache_kernel
def ipc_cand_vert_tri_kernel(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    flex_contype: wp.array[int],
    flex_conaffinity: wp.array[int],
    flex_selfcollide: wp.array[int],
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_elemadr: wp.array[int],
    flex_elemdataadr: wp.array[int],
    flex_elem: wp.array[int],
    flex_elemflexid: wp.array[int],
    flex_vertflexid: wp.array[int],
    # In:
    rad: wp.array[float],
    world_done: wp.array[int],
    x: wp.array2d[wp.vec3],
    dfrom: wp.array2d[wp.vec3],
    dto: wp.array2d[wp.vec3],
    dl: wp.array2d[float],
    fsweep: wp.array2d[float],
    ghat: float,
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    ncand_out: wp.array[int],
    cand_type_out: wp.array2d[int],
    cand_idx_out: wp.array2d[wp.vec4i],
    cand_geom_out: wp.array2d[int],
    cand_ld0_out: wp.array2d[float],
  ):
    """Discovers FLEX_VERT_TRI candidates between 2D flex vertices and triangles."""
    worldid, v, e = wp.tid()
    if world_done[worldid] != 0:
      return
    f1 = flex_vertflexid[v]
    f2 = flex_elemflexid[e]
    if not ipc_owns_flex_flex(flex_dim, f1, f2):
      return
    fc1 = flex_contype[f1]
    fa1 = flex_conaffinity[f1]
    if f1 == f2:
      if flex_selfcollide[f1] == 0 or (fc1 & fa1) == 0:
        return
    elif _mask_filtered(fc1, fa1, flex_contype[f2], flex_conaffinity[f2]):
      return

    eadr = flex_elemdataadr[f2] + 3 * (e - flex_elemadr[f2])
    va2 = flex_vertadr[f2]
    v0 = va2 + flex_elem[eadr]
    v1 = va2 + flex_elem[eadr + 1]
    v2 = va2 + flex_elem[eadr + 2]
    if v == v0 or v == v1 or v == v2:
      return

    idx = wp.vec4i(v, v0, v1, v2)
    bo_band = pair_bo_and_band(int(FlexPairType.FLEX_VERT_TRI), idx, rad, ghat)
    bo = bo_band[0]
    band = bo_band[1]
    reach = 3.0 * band + dl[worldid, v] + fsweep[worldid, f2]
    p0 = x[worldid, v]
    p1 = x[worldid, v0]
    p2 = x[worldid, v1]
    p3 = x[worldid, v2]

    if _aabb_separated(p0, p0, wp.min(wp.min(p1, p2), p3), wp.max(wp.max(p1, p2), p3), bo + reach):
      return

    dd, _cp, _w = pt_tri_dist(p0, p1, p2, p3)

    _add_cand(
      worldid,
      int(FlexPairType.FLEX_VERT_TRI),
      idx,
      -1,
      dd,
      bo,
      band,
      reach,
      dfrom,
      dto,
      dl,
      wp.static(bool(warn_overflow & OverflowType.BROADPHASE)),
      overflow_out,
      ncand_out,
      cand_type_out,
      cand_idx_out,
      cand_geom_out,
      cand_ld0_out,
    )

  return kernel


@cache_kernel
def ipc_cand_flex_edge_edge_kernel(warn_overflow: int):
  @wp.kernel(module="unique", enable_backward=False)
  def kernel(
    # Model:
    flex_contype: wp.array[int],
    flex_conaffinity: wp.array[int],
    flex_selfcollide: wp.array[int],
    flex_dim: wp.array[int],
    flex_vertadr: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_edgeflexid: wp.array[int],
    # In:
    rad: wp.array[float],
    world_done: wp.array[int],
    x: wp.array2d[wp.vec3],
    dfrom: wp.array2d[wp.vec3],
    dto: wp.array2d[wp.vec3],
    dl: wp.array2d[float],
    fsweep: wp.array2d[float],
    ghat: float,
    # Data out:
    overflow_out: wp.array[int],
    # Out:
    ncand_out: wp.array[int],
    cand_type_out: wp.array2d[int],
    cand_idx_out: wp.array2d[wp.vec4i],
    cand_geom_out: wp.array2d[int],
    cand_ld0_out: wp.array2d[float],
  ):
    """Discovers FLEX_EDGE_EDGE candidates between non-adjacent 2D flex edges."""
    worldid, p = wp.tid()
    if world_done[worldid] != 0:
      return
    e2 = int((1.0 + wp.sqrt(1.0 + 8.0 * float(p))) * 0.5)
    while e2 * (e2 - 1) // 2 > p:
      e2 -= 1
    while (e2 + 1) * e2 // 2 <= p:
      e2 += 1
    e1 = p - e2 * (e2 - 1) // 2
    f1 = flex_edgeflexid[e1]
    f2 = flex_edgeflexid[e2]
    if not ipc_owns_flex_flex(flex_dim, f1, f2):
      return
    fc1 = flex_contype[f1]
    fa1 = flex_conaffinity[f1]
    if f1 == f2:
      if flex_selfcollide[f1] == 0 or (fc1 & fa1) == 0:
        return
    elif _mask_filtered(fc1, fa1, flex_contype[f2], flex_conaffinity[f2]):
      return

    va1 = flex_vertadr[f1]
    ev1 = flex_edge[e1]
    a1 = va1 + ev1[0]
    b1 = va1 + ev1[1]

    va2 = flex_vertadr[f2]
    ev2 = flex_edge[e2]
    a2 = va2 + ev2[0]
    b2 = va2 + ev2[1]

    if a1 == a2 or a1 == b2 or b1 == a2 or b1 == b2:
      return

    idx = wp.vec4i(a1, b1, a2, b2)
    bo_band = pair_bo_and_band(int(FlexPairType.FLEX_EDGE_EDGE), idx, rad, ghat)
    bo = bo_band[0]
    band = bo_band[1]
    reach = 3.0 * band + wp.max(dl[worldid, a1], dl[worldid, b1]) + fsweep[worldid, f2]
    p0 = x[worldid, a1]
    p1 = x[worldid, b1]
    p2 = x[worldid, a2]
    p3 = x[worldid, b2]

    if _aabb_separated(wp.min(p0, p1), wp.max(p0, p1), wp.min(p2, p3), wp.max(p2, p3), bo + reach):
      return

    dd, _cp1, _cp2, _s, _t = seg_seg_dist(p0, p1, p2, p3)

    _add_cand(
      worldid,
      int(FlexPairType.FLEX_EDGE_EDGE),
      idx,
      -1,
      dd,
      bo,
      band,
      reach,
      dfrom,
      dto,
      dl,
      wp.static(bool(warn_overflow & OverflowType.BROADPHASE)),
      overflow_out,
      ncand_out,
      cand_type_out,
      cand_idx_out,
      cand_geom_out,
      cand_ld0_out,
    )

  return kernel


def ipc_discover_candidates(
  m: Model,
  d: Data,
  ws: Any,
  x: wp.array2d[wp.vec3],
  dfrom: wp.array2d[wp.vec3],
  dto: wp.array2d[wp.vec3],
  thresh: float,
  thresh_geom: float,
  ghat: float = 0.003,
):
  """Launches GPU candidate discovery across all 5 pair families (mjc_candidates)."""
  ws.fsweep.zero_()
  ws.ncand.zero_()
  if bool(m.opt.disableflags & (DisableBit.CONTACT | DisableBit.CONSTRAINT)):
    return

  nworld = d.nworld
  wp.launch(
    ipc_compute_fsweep_kernel,
    dim=(nworld, m.nflexvert),
    inputs=[m.flex_dim, m.flex_vertflexid, dfrom, dto, ws.world_done],
    outputs=[ws.dl, ws.fsweep],
  )

  warn_overflow = int(m.opt.warn_overflow)
  common_outputs = [
    d.overflow,
    ws.ncand,
    ws.cand_type,
    ws.cand_idx,
    ws.cand_geom,
    ws.cand_ld0,
  ]

  wp.launch(
    ipc_cand_vert_geom_kernel(warn_overflow),
    dim=(nworld, m.nflexvert, m.ngeom),
    inputs=[
      m.nexclude,
      m.opt.disableflags,
      m.body_parentid,
      m.body_weldid,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_bodyid,
      m.geom_dataid,
      m.geom_size,
      m.flex_contype,
      m.flex_conaffinity,
      m.flex_dim,
      m.mesh_vertadr,
      m.mesh_graphadr,
      m.mesh_vert,
      m.mesh_polynum,
      m.mesh_polyadr,
      m.mesh_polynormal,
      m.mesh_polyvertadr,
      m.mesh_polyvertnum,
      m.mesh_polyvert,
      m.exclude_signature,
      m.flex_vertflexid,
      d.geom_xpos,
      d.geom_xmat,
      ws.fidx,
      ws.pin_has_chain,
      ws.rad,
      ws.pbody,
      ws.geom_aabb_min,
      ws.geom_aabb_max,
      ws.world_done,
      x,
      ws.dl,
      thresh,
      ghat,
    ],
    outputs=common_outputs,
  )
  wp.launch(
    ipc_cand_corner_tri_kernel(warn_overflow),
    dim=(nworld, ws.max_ngv, m.nflexelem),
    inputs=[
      m.nexclude,
      m.opt.disableflags,
      m.body_parentid,
      m.body_weldid,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_bodyid,
      m.geom_size,
      m.flex_contype,
      m.flex_conaffinity,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_elemadr,
      m.flex_elemdataadr,
      m.flex_elem,
      m.exclude_signature,
      m.flex_elemflexid,
      ws.rad,
      ws.pbody,
      ws.ngv,
      ws.geom_corners,
      ws.corner_geom,
      ws.world_done,
      x,
      ws.dl,
      ws.fsweep,
      thresh_geom,
      ghat,
    ],
    outputs=common_outputs,
  )
  wp.launch(
    ipc_cand_geom_edge_edge_kernel(warn_overflow),
    dim=(nworld, ws.max_nge, m.nflexedge),
    inputs=[
      m.nexclude,
      m.opt.disableflags,
      m.body_parentid,
      m.body_weldid,
      m.geom_type,
      m.geom_contype,
      m.geom_conaffinity,
      m.geom_bodyid,
      m.geom_size,
      m.flex_contype,
      m.flex_conaffinity,
      m.flex_dim,
      m.flex_vertadr,
      m.flex_elemnum,
      m.flex_edge,
      m.exclude_signature,
      m.flex_edgeflexid,
      ws.rad,
      ws.pbody,
      ws.nge,
      ws.geom_edges,
      ws.edge_geom,
      ws.world_done,
      x,
      ws.dl,
      ws.fsweep,
      thresh_geom,
      ghat,
    ],
    outputs=common_outputs,
  )
  if m.has_flex_selfcollide or m.nflex > 1:
    wp.launch(
      ipc_cand_vert_tri_kernel(warn_overflow),
      dim=(nworld, m.nflexvert, m.nflexelem),
      inputs=[
        m.flex_contype,
        m.flex_conaffinity,
        m.flex_selfcollide,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_elemadr,
        m.flex_elemdataadr,
        m.flex_elem,
        m.flex_elemflexid,
        m.flex_vertflexid,
        ws.rad,
        ws.world_done,
        x,
        dfrom,
        dto,
        ws.dl,
        ws.fsweep,
        ghat,
      ],
      outputs=common_outputs,
    )
    wp.launch(
      ipc_cand_flex_edge_edge_kernel(warn_overflow),
      dim=(nworld, m.nflexedge * (m.nflexedge - 1) // 2),
      inputs=[
        m.flex_contype,
        m.flex_conaffinity,
        m.flex_selfcollide,
        m.flex_dim,
        m.flex_vertadr,
        m.flex_edge,
        m.flex_edgeflexid,
        ws.rad,
        ws.world_done,
        x,
        dfrom,
        dto,
        ws.dl,
        ws.fsweep,
        ghat,
      ],
      outputs=common_outputs,
    )


@wp.kernel
def ipc_eval_toi_kernel(
  # Model:
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  flex_vertflexid: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  ncand: wp.array[int],
  cand_type: wp.array2d[int],
  cand_idx: wp.array2d[wp.vec4i],
  cand_geom: wp.array2d[int],
  cand_ld0: wp.array2d[float],
  geom_corners: wp.array2d[wp.vec3],
  geom_edges: wp.array3d[wp.vec3],
  world_done: wp.array[int],
  x: wp.array2d[wp.vec3],
  xfree: wp.array2d[wp.vec3],
  # Out:
  cand_toi_out: wp.array2d[float],
  alpha_min_out: wp.array[float],
):
  """Evaluates conservative advancement time-of-impact across candidates (mjc_advance)."""
  worldid, c = wp.tid()
  if world_done[worldid] != 0 or c >= wp.min(ncand[worldid], cand_type.shape[1]):
    return
  g0 = cand_ld0[worldid, c]
  if g0 >= 1e20:
    cand_toi_out[worldid, c] = -1.0
    return

  pt = cand_type[worldid, c]
  idx = cand_idx[worldid, c]
  off, nv = pair_vert_range(pt)

  v0 = idx[off]
  xf0 = xfree[worldid, v0]
  dx0 = x[worldid, v0] - xf0
  dp0 = dx0

  xf1 = wp.vec3(0.0, 0.0, 0.0)
  dx1 = wp.vec3(0.0, 0.0, 0.0)
  dp1 = wp.vec3(0.0, 0.0, 0.0)
  if nv > 1:
    v1 = idx[off + 1]
    xf1 = xfree[worldid, v1]
    dx1 = x[worldid, v1] - xf1
    dp1 = dx1

  xf2 = wp.vec3(0.0, 0.0, 0.0)
  dx2 = wp.vec3(0.0, 0.0, 0.0)
  dp2 = wp.vec3(0.0, 0.0, 0.0)
  if nv > 2:
    v2 = idx[off + 2]
    xf2 = xfree[worldid, v2]
    dx2 = x[worldid, v2] - xf2
    dp2 = dx2

  xf3 = wp.vec3(0.0, 0.0, 0.0)
  dx3 = wp.vec3(0.0, 0.0, 0.0)
  dp3 = wp.vec3(0.0, 0.0, 0.0)
  if nv > 3:
    v3 = idx[off + 3]
    xf3 = xfree[worldid, v3]
    dx3 = x[worldid, v3] - xf3
    dp3 = dx3

  # Coherent mean displacement removal for true same-flex self-collision
  if pt <= int(FlexPairType.FLEX_EDGE_EDGE):
    other = idx[1] if pt == int(FlexPairType.FLEX_VERT_TRI) else idx[2]
    if flex_vertflexid[idx[0]] == flex_vertflexid[other]:
      mean = (dp0 + dp1 + dp2 + dp3) * 0.25
      dp0 -= mean
      dp1 -= mean
      dp2 -= mean
      dp3 -= mean

  # Bound l on the gap-shrink rate per unit alpha
  l_disp = float(0.0)
  if pt == int(FlexPairType.FLEX_VERT_TRI):
    l_disp = wp.length(dp0) + wp.max(wp.length(dp1), wp.max(wp.length(dp2), wp.length(dp3)))
  elif pt == int(FlexPairType.FLEX_EDGE_EDGE):
    l_disp = wp.max(wp.length(dp0), wp.length(dp1)) + wp.max(wp.length(dp2), wp.length(dp3))
  else:
    l_disp = wp.length(dp0)
    if nv > 1:
      l_disp = wp.max(l_disp, wp.length(dp1))
    if nv > 2:
      l_disp = wp.max(l_disp, wp.length(dp2))

  if l_disp < 1e-12 or g0 <= 0.0 or l_disp <= 0.8 * g0:
    cand_toi_out[worldid, c] = -1.0
    return

  gt, gsz, feat_rad, gp, gR, corner, eg0, eg1, va, pa, pn = unpack_pair_geom(
    geom_type,
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_polynum,
    mesh_polyadr,
    geom_xpos_in,
    geom_xmat_in,
    geom_corners,
    geom_edges,
    worldid,
    pt,
    idx,
    cand_geom[worldid, c],
  )

  gtarget = 0.2 * g0
  t = float(0.0)

  for _it in range(32):
    p0_t = xf0 + dx0 * t
    p1_t = xf1 + dx1 * t
    p2_t = xf2 + dx2 * t
    p3_t = xf3 + dx3 * t
    g, _n, _liv, _lcw, _lniv = pair_gap(
      mesh_vert,
      mesh_polynormal,
      mesh_polyvertadr,
      mesh_polyvertnum,
      mesh_polyvert,
      pt,
      idx,
      gt,
      gsz,
      feat_rad,
      gp,
      gR,
      corner,
      eg0,
      eg1,
      p0_t,
      p1_t,
      p2_t,
      p3_t,
      va,
      pa,
      pn,
    )
    room = g - gtarget
    if room <= 1e-9 * g0:
      break
    t += room / l_disp
    if t >= 1.0:
      t = 1.0
      break

  cand_toi_out[worldid, c] = t
  if t < 1.0:
    wp.atomic_min(alpha_min_out, worldid, t)


def ipc_advance(m: Model, d: Data, ws: Any):
  """Evaluates conservative advancement time-of-impact and reduces alpha_min (mjc_advance)."""
  ws.alpha_min.fill_(1.0)
  wp.launch(
    ipc_eval_toi_kernel,
    dim=(d.nworld, ws.cand_type.shape[1]),
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_size,
      m.mesh_vertadr,
      m.mesh_vert,
      m.mesh_polynum,
      m.mesh_polyadr,
      m.mesh_polynormal,
      m.mesh_polyvertadr,
      m.mesh_polyvertnum,
      m.mesh_polyvert,
      m.flex_vertflexid,
      d.geom_xpos,
      d.geom_xmat,
      ws.ncand,
      ws.cand_type,
      ws.cand_idx,
      ws.cand_geom,
      ws.cand_ld0,
      ws.geom_corners,
      ws.geom_edges,
      ws.world_done,
      ws.x,
      ws.xfree,
    ],
    outputs=[
      ws.cand_toi,
      ws.alpha_min,
    ],
  )
