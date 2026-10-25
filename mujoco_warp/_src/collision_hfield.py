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

"""Height field contacts measured against the height field's surface.

The height field is the union of one prism per triangle of its surface. A prism's side walls and
bottom are not part of that surface, so a contact computed against a single prism can leave through
them. Instead, for each triangle, the contact point is the point of the geom inside the triangle's
vertical column that is lowest along the triangle's normal, and its depth is the distance from that
point to the nearest point of the surface.

The lowest point is computed in the geom's frame, where the column runs along `up`. Candidates are
carried as (point, height along the triangle's normal).
"""

from typing import Tuple

import warp as wp

from mujoco_warp._src.collision_core import Geom
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import GeomType


@wp.func
def _wall(p0: wp.vec3, p1: wp.vec3, up: wp.vec3, x: wp.vec3) -> float:
  """Signed side of x from the wall through p0 p1 along up."""
  return wp.dot(wp.cross(p1 - p0, up), x - p0)


@wp.func
def _in_column(a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, x: wp.vec3) -> bool:
  side = wp.sign(_wall(a, b, up, c))
  tol = -1.0e-7 * wp.abs(_wall(a, b, up, c))
  return side * _wall(a, b, up, x) >= tol and side * _wall(b, c, up, x) >= tol and side * _wall(c, a, up, x) >= tol


@wp.func
def _on_wall(p0: wp.vec3, p1: wp.vec3, up: wp.vec3, x: wp.vec3) -> bool:
  """Whether x, a point on the wall through p0 p1 along up, lies between the wall's edges."""
  e = (p1 - p0) - wp.dot(p1 - p0, up) * up
  s = wp.dot(x - p0, e) / wp.dot(e, e)
  return s >= 0.0 and s <= 1.0


@wp.func
def _lower(n: wp.vec3, x: wp.vec3, low: wp.vec4) -> wp.vec4:
  return wp.where(wp.dot(n, x) < low[3], wp.vec4(x[0], x[1], x[2], wp.dot(n, x)), low)


@wp.func
def _lower_inside(a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, n: wp.vec3, x: wp.vec3, low: wp.vec4) -> wp.vec4:
  return wp.where(_in_column(a, b, c, up, x), _lower(n, x, low), low)


@wp.func
def _lower_on_wall(p0: wp.vec3, p1: wp.vec3, up: wp.vec3, n: wp.vec3, x: wp.vec3, low: wp.vec4) -> wp.vec4:
  return wp.where(_on_wall(p0, p1, up, x), _lower(n, x, low), low)


@wp.func
def _edge_through_wall(v0: wp.vec3, v1: wp.vec3, p0: wp.vec3, p1: wp.vec3, up: wp.vec3, n: wp.vec3, low: wp.vec4) -> wp.vec4:
  f0 = _wall(p0, p1, up, v0)
  f1 = _wall(p0, p1, up, v1)
  crosses = f0 * f1 <= 0.0 and f0 != f1
  x = v0 + (f0 / wp.where(crosses, f0 - f1, 1.0)) * (v1 - v0)
  return wp.where(crosses, _lower_on_wall(p0, p1, up, n, x, low), low)


@wp.func
def _edge_through_walls(
  v0: wp.vec3, v1: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, n: wp.vec3, low: wp.vec4
) -> wp.vec4:
  low = _edge_through_wall(v0, v1, a, b, up, n, low)
  low = _edge_through_wall(v0, v1, b, c, up, n, low)
  return _edge_through_wall(v0, v1, c, a, up, n, low)


@wp.func
def _column_edge(q: wp.vec3, up: wp.vec3, t: float, inside: bool, n: wp.vec3, low: wp.vec4) -> wp.vec4:
  """Lowers low by q + t up, the bottom of the geom on the column's edge through q, if it exists."""
  return wp.where(inside, _lower(n, q + t * up, low), low)


@wp.func
def _box_vert(size: wp.vec3, i: int) -> wp.vec3:
  return wp.vec3(wp.where(i & 1, size[0], -size[0]), wp.where(i & 2, size[1], -size[1]), wp.where(i & 4, size[2], -size[2]))


@wp.func
def _box_lowest(size: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, n: wp.vec3) -> wp.vec4:
  """Box: vertices in the column, edges through the walls, the column's edges through the box."""
  low = wp.vec4(0.0, 0.0, 0.0, MJ_MAXVAL)

  for i in range(8):
    low = _lower_inside(a, b, c, up, n, _box_vert(size, i), low)

  # edges from each vertex to the vertex with one more coordinate positive; repeats of a vertex
  # cross no wall
  for i in range(8):
    for k in range(3):
      low = _edge_through_walls(_box_vert(size, i), _box_vert(size, i | (1 << k)), a, b, c, up, n, low)

  for w in range(3):
    q = wp.where(w == 0, a, wp.where(w == 1, b, c))
    lo = float(-MJ_MAXVAL)
    hi = float(MJ_MAXVAL)
    for j in range(3):
      if wp.abs(up[j]) < MJ_MINVAL:
        if wp.abs(q[j]) > size[j]:
          lo = MJ_MAXVAL
      else:
        t0 = (-size[j] - q[j]) / up[j]
        t1 = (size[j] - q[j]) / up[j]
        lo = wp.max(lo, wp.min(t0, t1))
        hi = wp.min(hi, wp.max(t0, t1))
    low = _column_edge(q, up, lo, lo <= hi, n, low)

  return low


@wp.func
def _mesh_lowest(geom: Geom, a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, n: wp.vec3) -> wp.vec4:
  """Mesh: vertices in the column, polygon edges through the walls, column edges through it."""
  low = wp.vec4(0.0, 0.0, 0.0, MJ_MAXVAL)

  for i in range(geom.vertnum):
    low = _lower_inside(a, b, c, up, n, geom.vert[geom.vertadr + i], low)

  # each polygon edge is visited from both of its polygons; each polygon's half-space pn . x <= pn .
  # v bounds the column's edges a + t up, b + t up and c + t up to t in [lo, hi]
  lo = wp.vec3(-MJ_MAXVAL)
  hi = wp.vec3(MJ_MAXVAL)
  for k in range(geom.mesh_polynum):
    adr = geom.mesh_polyvertadr[geom.mesh_polyadr + k]
    nv = geom.mesh_polyvertnum[geom.mesh_polyadr + k]
    v0 = geom.vert[geom.vertadr + geom.mesh_polyvert[adr + nv - 1]]
    for i in range(nv):
      v1 = geom.vert[geom.vertadr + geom.mesh_polyvert[adr + i]]
      low = _edge_through_walls(v0, v1, a, b, c, up, n, low)
      v0 = v1
    pn = geom.mesh_polynormal[geom.mesh_polyadr + k]
    pnz = wp.dot(pn, up)
    rhs = wp.vec3(wp.dot(pn, v0 - a), wp.dot(pn, v0 - b), wp.dot(pn, v0 - c))
    if wp.abs(pnz) < MJ_MINVAL:
      for j in range(3):
        if rhs[j] < 0.0:
          lo[j] = MJ_MAXVAL
    elif pnz > 0.0:
      hi = wp.min(hi, rhs / pnz)
    else:
      lo = wp.max(lo, rhs / pnz)

  low = _column_edge(a, up, lo[0], lo[0] <= hi[0], n, low)
  low = _column_edge(b, up, lo[1], lo[1] <= hi[1], n, low)
  return _column_edge(c, up, lo[2], lo[2] <= hi[2], n, low)


@wp.func
def _ball_lowest(o: wp.vec3, r: float, a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, n: wp.vec3, low: wp.vec4) -> wp.vec4:
  """Ball: support point in the column, lowest point of each disk a wall cuts, entries of edges."""
  low = _lower_inside(a, b, c, up, n, o - r * n, low)

  for w in range(3):
    p0 = wp.where(w == 0, a, wp.where(w == 1, b, c))
    p1 = wp.where(w == 0, b, wp.where(w == 1, c, a))
    m = wp.normalize(wp.cross(p1 - p0, up))
    h = wp.dot(m, o - p0)
    t = n - wp.dot(n, m) * m
    t -= wp.dot(t, m) * m
    x = o - h * m - wp.sqrt(wp.max(r * r - h * h, 0.0)) * wp.normalize(t)
    low = wp.where(wp.abs(h) <= r, _lower_on_wall(p0, p1, up, n, x, low), low)

  # the column's edges enter the ball at the smaller root of |q + t up - o| = r, as n . up > 0
  for w in range(3):
    q = wp.where(w == 0, a, wp.where(w == 1, b, c))
    bq = wp.dot(q - o, up)
    disc = bq * bq - wp.length_sq(q - o) + r * r
    low = _column_edge(q, up, -bq - wp.sqrt(wp.max(disc, 0.0)), disc >= 0.0, n, low)

  return low


@wp.func
def _disk_lowest(o: wp.vec3, r: float, a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, n: wp.vec3, low: wp.vec4) -> wp.vec4:
  """Disk of radius r around o across z: support point, ends of the wall chords, edge crossings."""
  z = wp.vec3(0.0, 0.0, 1.0)
  t = wp.vec3(n[0], n[1], 0.0)
  low = _lower_inside(a, b, c, up, n, wp.where(wp.length_sq(t) > 0.0, o - r * wp.normalize(t), o), low)

  for w in range(3):
    p0 = wp.where(w == 0, a, wp.where(w == 1, b, c))
    p1 = wp.where(w == 0, b, wp.where(w == 1, c, a))
    m = wp.cross(p1 - p0, up)
    e1 = wp.cross(z, m)
    if wp.length_sq(e1) > 0.0:
      e1 = wp.normalize(e1)
      e2 = wp.cross(z, e1)
      # points o + r (+-sqrt(1 - k^2) e1 + k e2) of the circle on the wall
      k = wp.dot(m, p0 - o) / (r * wp.dot(m, e2))
      if wp.abs(k) <= 1.0:
        s = wp.sqrt(1.0 - k * k)
        low = _lower_on_wall(p0, p1, up, n, o + r * (s * e1 + k * e2), low)
        low = _lower_on_wall(p0, p1, up, n, o + r * (k * e2 - s * e1), low)

  for w in range(3):
    q = wp.where(w == 0, a, wp.where(w == 1, b, c))
    tz = (o[2] - q[2]) / wp.where(wp.abs(up[2]) > MJ_MINVAL, up[2], 1.0)
    x = q + tz * up
    inside = wp.abs(up[2]) > MJ_MINVAL and wp.length_sq(wp.vec3(x[0] - o[0], x[1] - o[1], 0.0)) <= r * r
    low = _column_edge(q, up, tz, inside, n, low)

  return low


@wp.func
def _capsule_lowest(r: float, half: float, a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, n: wp.vec3) -> wp.vec4:
  """Capsule of radius r around the segment from -half to half on z."""
  low = wp.vec4(0.0, 0.0, 0.0, MJ_MAXVAL)
  e0 = wp.vec3(0.0, 0.0, -half) - r * n
  e1 = wp.vec3(0.0, 0.0, half) - r * n

  # the support points, which are the segment shifted by -r n where n is across the axis
  low = _lower_inside(a, b, c, up, n, e0, low)
  low = _lower_inside(a, b, c, up, n, e1, low)
  low = _edge_through_walls(e0, e1, a, b, c, up, n, low)

  # on a wall at distance h(u) = h0 + hz u from the axis point (0, 0, u), the ball cuts a disk whose
  # lowest point is g(u) = n_z u - h(u) n . m - |t| sqrt(r^2 - h(u)^2), convex in u; its stationary
  # point has h = hs
  for w in range(3):
    p0 = wp.where(w == 0, a, wp.where(w == 1, b, c))
    p1 = wp.where(w == 0, b, wp.where(w == 1, c, a))
    m = wp.normalize(wp.cross(p1 - p0, up))
    t = n - wp.dot(n, m) * m
    t -= wp.dot(t, m) * m
    tl = wp.length(t)
    h0 = -wp.dot(m, p0)
    hz = m[2]
    slope = n[2] - hz * wp.dot(n, m)
    if wp.abs(hz) > MJ_MINVAL:
      ulo = wp.max(-half, wp.min((-r - h0) / hz, (r - h0) / hz))
      uhi = wp.min(half, wp.max((-r - h0) / hz, (r - h0) / hz))
      hs = -wp.sign(slope * hz) * wp.abs(slope) * r / wp.sqrt(slope * slope + tl * tl * hz * hz)
      u = wp.clamp((hs - h0) / hz, ulo, uhi)
    else:
      ulo = wp.where(wp.abs(h0) <= r, -half, MJ_MAXVAL)
      uhi = half
      u = -half * wp.sign(slope)
    h = h0 + hz * u
    x = wp.vec3(0.0, 0.0, u) - h * m - wp.sqrt(wp.max(r * r - h * h, 0.0)) * wp.where(tl > 0.0, t / tl, t)
    low = wp.where(ulo <= uhi, _lower_on_wall(p0, p1, up, n, x, low), low)

  # the column's edges enter the capsule through its side, |x_xy| = r with |x_z| <= half, or through
  # an end ball
  for w in range(3):
    q = wp.where(w == 0, a, wp.where(w == 1, b, c))
    qa = up[0] * up[0] + up[1] * up[1]
    qb = q[0] * up[0] + q[1] * up[1]
    disc = qb * qb - qa * (q[0] * q[0] + q[1] * q[1] - r * r)
    ts = (-qb - wp.sqrt(wp.max(disc, 0.0))) / wp.where(qa > MJ_MINVAL, qa, 1.0)
    side = qa > MJ_MINVAL and disc >= 0.0 and wp.abs(q[2] + ts * up[2]) <= half
    low = _column_edge(q, up, ts, side, n, low)
    for e in range(2):
      o = wp.vec3(0.0, 0.0, wp.where(e == 0, -half, half))
      bq = wp.dot(q - o, up)
      disc = bq * bq - wp.length_sq(q - o) + r * r
      low = _column_edge(q, up, -bq - wp.sqrt(wp.max(disc, 0.0)), disc >= 0.0, n, low)

  return low


@wp.func
def _cylinder_lowest(r: float, half: float, a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, n: wp.vec3) -> wp.vec4:
  """Cylinder of radius r around z from -half to half."""
  low = wp.vec4(0.0, 0.0, 0.0, MJ_MAXVAL)

  # caps: their support points, the ends of the wall chords, and the column's edges through them
  low = _disk_lowest(wp.vec3(0.0, 0.0, -half), r, a, b, c, up, n, low)
  low = _disk_lowest(wp.vec3(0.0, 0.0, half), r, a, b, c, up, n, low)

  # a wall m . x = d cuts the side in the points (r cos f, r sin f, (d - r (m_x cos f + m_y sin f))
  # / m_z), on which n . x is lowest at f = atan2(-k_y, -k_x)
  for w in range(3):
    p0 = wp.where(w == 0, a, wp.where(w == 1, b, c))
    p1 = wp.where(w == 0, b, wp.where(w == 1, c, a))
    m = wp.normalize(wp.cross(p1 - p0, up))
    mz = wp.where(wp.abs(m[2]) > MJ_MINVAL, m[2], 1.0)
    kx = n[0] - n[2] * m[0] / mz
    ky = n[1] - n[2] * m[1] / mz
    f = wp.atan2(-ky, -kx)
    x = wp.vec3(r * wp.cos(f), r * wp.sin(f), 0.0)
    x[2] = (wp.dot(m, p0) - m[0] * x[0] - m[1] * x[1]) / mz
    cuts = wp.abs(m[2]) > MJ_MINVAL and kx * kx + ky * ky > 0.0 and wp.abs(x[2]) <= half
    low = wp.where(cuts, _lower_on_wall(p0, p1, up, n, x, low), low)

  # the column's edges enter the cylinder where they are inside both its side and its caps
  for w in range(3):
    q = wp.where(w == 0, a, wp.where(w == 1, b, c))
    lo = float(-MJ_MAXVAL)
    hi = float(MJ_MAXVAL)
    if wp.abs(up[2]) > MJ_MINVAL:
      lo = wp.min((-half - q[2]) / up[2], (half - q[2]) / up[2])
      hi = wp.max((-half - q[2]) / up[2], (half - q[2]) / up[2])
    elif wp.abs(q[2]) > half:
      lo = MJ_MAXVAL
    qa = up[0] * up[0] + up[1] * up[1]
    qb = q[0] * up[0] + q[1] * up[1]
    qc = q[0] * q[0] + q[1] * q[1] - r * r
    if qa > MJ_MINVAL:
      disc = qb * qb - qa * qc
      if disc < 0.0:
        lo = MJ_MAXVAL
      else:
        lo = wp.max(lo, (-qb - wp.sqrt(disc)) / qa)
        hi = wp.min(hi, (-qb + wp.sqrt(disc)) / qa)
    elif qc > 0.0:
      lo = MJ_MAXVAL
    low = _column_edge(q, up, lo, lo <= hi, n, low)

  return low


@wp.func
def lowest_in_column(geom: Geom, geomtype: int, a: wp.vec3, b: wp.vec3, c: wp.vec3, up: wp.vec3, n: wp.vec3) -> wp.vec4:
  """The point of the geom in the column over abc along up that is lowest along n, in its frame.

  Returns (point, n . point); the height is MJ_MAXVAL if the geom does not reach into the column.
  """
  size = geom.size
  if geomtype == GeomType.BOX:
    return _box_lowest(size, a, b, c, up, n)
  if geomtype == GeomType.MESH:
    return _mesh_lowest(geom, a, b, c, up, n)
  if geomtype == GeomType.SPHERE:
    return _ball_lowest(wp.vec3(0.0), size[0], a, b, c, up, n, wp.vec4(0.0, 0.0, 0.0, MJ_MAXVAL))
  if geomtype == GeomType.ELLIPSOID:
    # the ellipsoid is the unit ball scaled by size; the column and n follow the scaling
    upu = wp.normalize(wp.cw_div(up, size))
    nu = wp.normalize(wp.cw_mul(n, size))
    low = _ball_lowest(
      wp.vec3(0.0), 1.0, wp.cw_div(a, size), wp.cw_div(b, size), wp.cw_div(c, size), upu, nu, wp.vec4(0.0, 0.0, 0.0, MJ_MAXVAL)
    )
    x = wp.cw_mul(wp.vec3(low[0], low[1], low[2]), size)
    return wp.vec4(x[0], x[1], x[2], wp.where(low[3] < MJ_MAXVAL, wp.dot(n, x), MJ_MAXVAL))
  if geomtype == GeomType.CAPSULE:
    return _capsule_lowest(size[0], size[1], a, b, c, up, n)
  return _cylinder_lowest(size[0], size[1], a, b, c, up, n)


@wp.func
def _closest_on_triangle(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3:
  """Closest point to p on triangle abc (Ericson, Real-Time Collision Detection, 5.1.5)."""
  ab = b - a
  ac = c - a
  ap = p - a
  d1 = wp.dot(ab, ap)
  d2 = wp.dot(ac, ap)
  if d1 <= 0.0 and d2 <= 0.0:
    return a
  bp = p - b
  d3 = wp.dot(ab, bp)
  d4 = wp.dot(ac, bp)
  if d3 >= 0.0 and d4 <= d3:
    return b
  vc = d1 * d4 - d3 * d2
  if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
    return a + d1 / (d1 - d3) * ab
  cp = p - c
  d5 = wp.dot(ab, cp)
  d6 = wp.dot(ac, cp)
  if d6 >= 0.0 and d5 <= d6:
    return c
  vb = d5 * d2 - d1 * d6
  if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
    return a + d2 / (d2 - d6) * ac
  va = d3 * d6 - d5 * d4
  if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
    return b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)
  denom = 1.0 / (va + vb + vc)
  return a + ab * (vb * denom) + ac * (vc * denom)


@wp.func
def _hfield_vert(data: wp.array[float], adr: int, ncol: int, size: wp.vec4, dx: float, dy: float, r: int, c: int) -> wp.vec3:
  return wp.vec3(dx * float(c) - size[0], dy * float(r) - size[1], data[adr + r * ncol + c] * size[2])


@wp.func
def hfield_triangle(
  data: wp.array[float], adr: int, ncol: int, size: wp.vec4, dx: float, dy: float, r: int, c: int, i: int
) -> Tuple[wp.vec3, wp.vec3, wp.vec3]:
  """Triangle i of the cell between rows r, r + 1 and columns c - 1, c, as MuJoCo's prisms."""
  v00 = _hfield_vert(data, adr, ncol, size, dx, dy, r, c - 1)
  v11 = _hfield_vert(data, adr, ncol, size, dx, dy, r + 1, c)
  if i == 0:
    return _hfield_vert(data, adr, ncol, size, dx, dy, r + 1, c - 1), v00, v11
  return v00, v11, _hfield_vert(data, adr, ncol, size, dx, dy, r, c)


@wp.func
def _hfield_closest(data: wp.array[float], adr: int, nrow: int, ncol: int, size: wp.vec4, p: wp.vec3, radius: float) -> wp.vec3:
  """The point of the height field surface closest to p, among the cells within radius of p."""
  dx = 2.0 * size[0] / float(ncol - 1)
  dy = 2.0 * size[1] / float(nrow - 1)
  cmin = wp.max(1, int(wp.floor((p[0] - radius + size[0]) / dx)))
  cmax = wp.min(ncol - 1, int(wp.ceil((p[0] + radius + size[0]) / dx)))
  rmin = wp.max(0, int(wp.floor((p[1] - radius + size[1]) / dy)))
  rmax = wp.min(nrow - 2, int(wp.ceil((p[1] + radius + size[1]) / dy)))

  closest = p
  best = float(MJ_MAXVAL)
  for r in range(rmin, rmax + 1):
    for c in range(cmin, cmax + 1):
      for i in range(2):
        a, b, t = hfield_triangle(data, adr, ncol, size, dx, dy, r, c, i)
        q = _closest_on_triangle(p, a, b, t)
        if wp.length_sq(q - p) < best:
          best = wp.length_sq(q - p)
          closest = q
  return closest


@wp.func
def hfield_cell_contact(
  # In:
  geom: Geom,
  geomtype: int,
  up: wp.vec3,
  a: wp.vec3,
  b: wp.vec3,
  c: wp.vec3,
  data: wp.array[float],
  adr: int,
  nrow: int,
  ncol: int,
  size: wp.vec4,
) -> Tuple[bool, float, wp.vec3, wp.vec3]:
  """Contact of the geom with the height field over triangle abc, in the height field frame.

  up is the height field's z in the geom's frame. Returns whether the geom reaches into the
  triangle's column, and the contact's distance, position and normal from the height field to the
  geom.
  """
  n = wp.normalize(wp.cross(b - a, c - a))
  n = wp.where(n[2] < 0.0, -n, n)
  rotT = wp.transpose(geom.rot)
  low = lowest_in_column(geom, geomtype, rotT @ (a - geom.pos), rotT @ (b - geom.pos), rotT @ (c - geom.pos), up, rotT @ n)
  if low[3] == MJ_MAXVAL:
    return False, 0.0, wp.vec3(), wp.vec3()

  p = geom.pos + geom.rot @ wp.vec3(low[0], low[1], low[2])
  below = wp.dot(n, a - p)
  if below <= 0.0:
    # above the triangle's plane: the distance along its normal
    return True, -below, p + 0.5 * below * n, n

  # under the surface: the depth is the distance to the surface, which the triangle above p bounds
  q = _hfield_closest(data, adr, nrow, ncol, size, p, below / n[2])
  depth = wp.length(q - p)
  normal = wp.where(depth > MJ_MINVAL, (q - p) / wp.max(depth, MJ_MINVAL), n)
  return True, -depth, 0.5 * (p + q), normal
