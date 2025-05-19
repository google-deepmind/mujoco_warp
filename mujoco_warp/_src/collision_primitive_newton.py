import warp as wp


from .math import closest_segment_point
from .math import closest_segment_to_segment_points
from .math import make_frame
from .math import normalize_with_norm

wp.config.enable_backward = False

wp.clear_kernel_cache()


@wp.struct
class Geom:
  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3
  size: wp.vec3
  vertadr: int
  vertnum: int
  vert: wp.array(dtype=wp.vec3)


@wp.struct
class ContactFrame:
  pos: wp.vec3
  frame: wp.mat33  # The first row of the frame is the normal
  dist: float


@wp.func
def _plane_sphere(plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float):
  dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@wp.func
def plane_sphere(
  # In:
  plane: Geom,
  sphere: Geom,
) -> ContactFrame:
  dist, pos = _plane_sphere(plane.normal, plane.pos, sphere.pos, sphere.size[0])

  # Return contact frame using make_frame helper
  return ContactFrame(pos=pos, frame=make_frame(plane.normal), dist=dist)


@wp.func
def _sphere_sphere(pos1: wp.vec3, radius1: float, pos2: wp.vec3, radius2: float) -> ContactFrame:
  dir = pos2 - pos1
  dist = wp.length(dir)
  if dist == 0.0:
    n = wp.vec3(1.0, 0.0, 0.0)
  else:
    n = dir / dist
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)

  return ContactFrame(pos=pos, frame=make_frame(n), dist=dist)


@wp.func
def sphere_sphere(
  # In:
  sphere1: Geom,
  sphere2: Geom,
) -> ContactFrame:
  return _sphere_sphere(
    sphere1.pos,
    sphere1.size[0],
    sphere2.pos,
    sphere2.size[0],
  )


@wp.func
def sphere_capsule(
  # In:
  sphere: Geom,
  cap: Geom,
) -> ContactFrame:
  """Calculates one contact between a sphere and a capsule."""
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  length = cap.size[1]
  segment = axis * length

  # Find closest point on capsule centerline to sphere center
  pt = closest_segment_point(cap.pos - segment, cap.pos + segment, sphere.pos)

  # Treat as sphere-sphere collision between sphere and closest point
  return _sphere_sphere(
    sphere.pos,
    sphere.size[0],
    pt,
    cap.size[0],
  )


@wp.func
def capsule_capsule(
  # In:
  cap1: Geom,
  cap2: Geom,
) -> ContactFrame:
  """Calculates one contact between two capsules."""
  axis1 = wp.vec3(cap1.rot[0, 2], cap1.rot[1, 2], cap1.rot[2, 2])
  axis2 = wp.vec3(cap2.rot[0, 2], cap2.rot[1, 2], cap2.rot[2, 2])
  length1 = cap1.size[1]
  length2 = cap2.size[1]
  seg1 = axis1 * length1
  seg2 = axis2 * length2

  pt1, pt2 = closest_segment_to_segment_points(
    cap1.pos - seg1,
    cap1.pos + seg1,
    cap2.pos - seg2,
    cap2.pos + seg2,
  )

  return _sphere_sphere(
    pt1,
    cap1.size[0],
    pt2,
    cap2.size[0],
  )


@wp.func
def plane_capsule(
  # In:
  plane: Geom,
  cap: Geom,
):
  """Calculates two contacts between a capsule and a plane."""
  n = plane.normal
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  # align contact frames with capsule axis
  b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))

  if b_norm < 0.5:
    if -0.5 < n[1] and n[1] < 0.5:
      b = wp.vec3(0.0, 1.0, 0.0)
    else:
      b = wp.vec3(0.0, 0.0, 1.0)

  c = wp.cross(n, b)
  frame = wp.mat33(n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2])
  segment = axis * cap.size[1]

  dist1, pos1 = _plane_sphere(n, plane.pos, cap.pos + segment, cap.size[0])
  dist2, pos2 = _plane_sphere(n, plane.pos, cap.pos - segment, cap.size[0])

  return ContactFrame(pos=pos1, frame=frame, dist=dist1), ContactFrame(pos=pos2, frame=frame, dist=dist2)


@wp.func
def _sphere_sphere_ext(
  # In:
  pos1: wp.vec3,
  radius1: float,
  pos2: wp.vec3,
  radius2: float,
  mat1: wp.mat33,
  mat2: wp.mat33,
) -> ContactFrame:
  dir = pos2 - pos1
  dist = wp.length(dir)
  if dist == 0.0:
    # Use cross product of z axes like MuJoCo
    axis1 = wp.vec3(mat1[0, 2], mat1[1, 2], mat1[2, 2])
    axis2 = wp.vec3(mat2[0, 2], mat2[1, 2], mat2[2, 2])
    n = wp.cross(axis1, axis2)
    n = wp.normalize(n)
  else:
    n = dir / dist
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)

  return ContactFrame(pos=pos, frame=make_frame(n), dist=dist)


@wp.func
def sphere_cylinder(
  # In:
  sphere: Geom,
  cylinder: Geom,
) -> ContactFrame:
  axis = wp.vec3(
    cylinder.rot[0, 2],
    cylinder.rot[1, 2],
    cylinder.rot[2, 2],
  )

  vec = sphere.pos - cylinder.pos
  x = wp.dot(vec, axis)

  a_proj = axis * x
  p_proj = vec - a_proj
  p_proj_sqr = wp.dot(p_proj, p_proj)

  collide_side = wp.abs(x) < cylinder.size[1]
  collide_cap = p_proj_sqr < (cylinder.size[0] * cylinder.size[0])

  if collide_side and collide_cap:
    dist_cap = cylinder.size[1] - wp.abs(x)
    dist_radius = cylinder.size[0] - wp.sqrt(p_proj_sqr)

    if dist_cap < dist_radius:
      collide_side = False
    else:
      collide_cap = False

  # Side collision
  if collide_side:
    pos_target = cylinder.pos + a_proj

    return _sphere_sphere_ext(
      sphere.pos,
      sphere.size[0],
      pos_target,
      cylinder.size[0],
      sphere.rot,
      cylinder.rot,
    )

  # Cap collision
  if collide_cap:
    if x > 0.0:
      # top cap
      pos_cap = cylinder.pos + axis * cylinder.size[1]
      plane_normal = axis
    else:
      # bottom cap
      pos_cap = cylinder.pos - axis * cylinder.size[1]
      plane_normal = -axis

    dist, pos_contact = _plane_sphere(plane_normal, pos_cap, sphere.pos, sphere.size[0])
    plane_normal = -plane_normal  # Flip normal after position calculation

    return ContactFrame(pos=pos_contact, frame=make_frame(plane_normal), dist=dist)

  # Corner collision
  inv_len = 1.0 / wp.sqrt(p_proj_sqr)
  p_proj = p_proj * (cylinder.size[0] * inv_len)

  cap_offset = axis * (wp.sign(x) * cylinder.size[1])
  pos_corner = cylinder.pos + cap_offset + p_proj

  return _sphere_sphere_ext(
    sphere.pos,
    sphere.size[0],
    pos_corner,
    0.0,
    sphere.rot,
    cylinder.rot,
  )


@wp.func
def _sphere_box(
  sphere_pos: wp.vec3, sphere_size: float, box_pos: wp.vec3, box_rot: wp.mat33, box_size: wp.vec3, margin: float
) -> ContactFrame:
  center = wp.transpose(box_rot) @ (sphere_pos - box_pos)

  clamped = wp.max(-box_size, wp.min(box_size, center))
  clamped_dir, dist = normalize_with_norm(clamped - center)

  if dist - sphere_size > margin:
    return ContactFrame(pos=wp.vec3(0.0), frame=wp.mat33(1.0), dist=100000.0)

  # sphere center inside box
  if dist <= float(1e-8):
    closest = 2.0 * (box_size[0] + box_size[1] + box_size[2])
    k = wp.int32(0)
    for i in range(6):
      face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i / 2] - center[i / 2])
      if closest > face_dist:
        closest = face_dist
        k = i

    nearest = wp.vec3(0.0)
    nearest[k / 2] = wp.where(k % 2, -1.0, 1.0)
    pos = center + nearest * (sphere_size - closest) / 2.0
    contact_normal = box_rot @ nearest
    contact_dist = -closest - sphere_size

  else:
    deepest = center + clamped_dir * sphere_size
    pos = 0.5 * (clamped + deepest)
    contact_normal = box_rot @ clamped_dir
    contact_dist = dist - sphere_size

  contact_pos = box_pos + box_rot @ pos
  return ContactFrame(pos=contact_pos, frame=make_frame(contact_normal), dist=contact_dist)


@wp.func
def sphere_box(
  # In:
  sphere: Geom,
  box: Geom,
  margin: float,
) -> ContactFrame:
  return _sphere_box(sphere.pos, sphere.size[0], box.pos, box.rot, box.size, margin)


@wp.func
def capsule_box(
  # In:
  cap: Geom,
  box: Geom,
  margin: float,
):
  """Calculates contacts between a capsule and a box."""
  # Based on the mjc implementation
  pos = wp.transpose(box.rot) @ (cap.pos - box.pos)
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  halfaxis = axis * cap.size[1]  # halfaxis is the capsule direction
  axisdir = wp.int32(axis[0] > 0.0) + 2 * wp.int32(axis[1] > 0.0) + 4 * wp.int32(axis[2] > 0.0)

  bestdistmax = margin + 2.0 * (cap.size[0] + cap.size[1] + box.size[0] + box.size[1] + box.size[2])

  # keep track of closest point
  bestdist = wp.float32(bestdistmax)
  bestsegmentpos = wp.float32(-12)

  # cltype: encoded collision configuration
  # cltype / 3 == 0 : lower corner is closest to the capsule
  #            == 2 : upper corner is closest to the capsule
  #            == 1 : middle of the edge is closest to the capsule
  # cltype % 3 == 0 : lower corner is closest to the box
  #            == 2 : upper corner is closest to the box
  #            == 1 : middle of the capsule is closest to the box
  cltype = wp.int32(-4)

  # clface: index of the closest face of the box to the capsule
  # -1: no face is closest (edge or corner is closest)
  # 0, 1, 2: index of the axis perpendicular to the closest face
  clface = wp.int32(-12)

  # first: consider cases where a face of the box is closest
  for i in range(-1, 2, 2):
    axisTip = pos + wp.float32(i) * halfaxis
    boxPoint = wp.vec3(axisTip)

    n_out = wp.int32(0)
    ax_out = wp.int32(-1)

    for j in range(3):
      if boxPoint[j] < -box.size[j]:
        n_out += 1
        ax_out = j
        boxPoint[j] = -box.size[j]
      elif boxPoint[j] > box.size[j]:
        n_out += 1
        ax_out = j
        boxPoint[j] = box.size[j]

    if n_out > 1:
      continue

    dist = wp.length_sq(boxPoint - axisTip)

    if dist < bestdist:
      bestdist = dist
      bestsegmentpos = wp.float32(i)
      cltype = -2 + i
      clface = ax_out

  # second: consider cases where an edge of the box is closest
  clcorner = wp.int32(-123)  # which corner is the closest
  cledge = wp.int32(-123)  # which axis
  bestboxpos = wp.float32(0.0)

  for i in range(8):
    for j in range(3):
      if i & (1 << j) != 0:
        continue

      c2 = wp.int32(-123)

      # box_pt is the starting point (corner) on the box
      box_pt = wp.cw_mul(
        wp.vec3(
          wp.where(i & 1, 1.0, -1.0),
          wp.where(i & 2, 1.0, -1.0),
          wp.where(i & 4, 1.0, -1.0),
        ),
        box.size,
      )
      box_pt[j] = 0.0

      # find closest point between capsule and the edge
      dif = box_pt - pos

      u = -box.size[j] * dif[j]
      v = wp.dot(halfaxis, dif)
      ma = box.size[j] * box.size[j]
      mb = -box.size[j] * halfaxis[j]
      mc = cap.size[1] * cap.size[1]
      det = ma * mc - mb * mb
      if wp.abs(det) < float(1e-8):
        continue

      idet = 1.0 / det
      # sX : X=1 means middle of segment. X=0 or 2 one or the other end

      x1 = wp.float32((mc * u - mb * v) * idet)
      x2 = wp.float32((ma * v - mb * u) * idet)

      s1 = wp.int32(1)
      s2 = wp.int32(1)

      if x1 > 1:
        x1 = 1.0
        s1 = 2
        x2 = (v - mb) / mc
      elif x1 < -1:
        x1 = -1.0
        s1 = 0
        x2 = (v + mb) / mc

      x2_over = x2 > 1.0
      if x2_over or x2 < -1.0:
        if x2_over:
          x2 = 1.0
          s2 = 2
          x1 = (u - mb) / ma
        else:
          x2 = -1.0
          s2 = 0
          x1 = (u + mb) / ma

        if x1 > 1:
          x1 = 1.0
          s1 = 2
        elif x1 < -1:
          x1 = -1.0
          s1 = 0

      dif -= halfaxis * x2
      dif[j] += box.size[j] * x1

      # encode relative positions of the closest points
      ct = s1 * 3 + s2

      dif_sq = wp.length_sq(dif)
      if dif_sq < bestdist - float(1e-8):
        bestdist = dif_sq
        bestsegmentpos = x2
        bestboxpos = x1
        # ct<6 means closest point on box is at lower end or middle of edge
        c2 = ct / 6

        clcorner = i + (1 << j) * c2  # index of closest box corner
        cledge = j  # axis index of closest box edge
        cltype = ct  # encoded collision configuration

  best = wp.float32(0.0)

  p = wp.vec2(pos.x, pos.y)
  dd = wp.vec2(halfaxis.x, halfaxis.y)
  s = wp.vec2(box.size.x, box.size.y)
  secondpos = wp.float32(-4.0)

  uu = dd.x * s.y
  vv = dd.y * s.x
  w_neg = dd.x * p.y - dd.y * p.x < 0

  best = wp.float32(-1.0)

  ee1 = uu - vv
  ee2 = uu + vv

  if wp.abs(ee1) > best:
    best = wp.abs(ee1)
    c1 = wp.where((ee1 < 0) == w_neg, 0, 3)

  if wp.abs(ee2) > best:
    best = wp.abs(ee2)
    c1 = wp.where((ee2 > 0) == w_neg, 1, 2)

  if cltype == -4:  # invalid type
    return ContactFrame(), ContactFrame(), 0

  if cltype >= 0 and cltype / 3 != 1:  # closest to a corner of the box
    c1 = axisdir ^ clcorner
    # Calculate relative orientation between capsule and corner
    # There are two possible configurations:
    # 1. Capsule axis points toward/away from corner
    # 2. Capsule axis aligns with a face or edge
    if c1 != 0 and c1 != 7:  # create second contact point
      if c1 == 1 or c1 == 2 or c1 == 4:
        mul = 1
      else:
        mul = -1
        c1 = 7 - c1

      # "de" and "dp" distance from first closest point on the capsule to both ends of it
      # mul is a direction along the capsule's axis

      if c1 == 1:
        ax = 0
        ax1 = 1
        ax2 = 2
      elif c1 == 2:
        ax = 1
        ax1 = 2
        ax2 = 0
      elif c1 == 4:
        ax = 2
        ax1 = 0
        ax2 = 1

      if axis[ax] * axis[ax] > 0.5:  # second point along the edge of the box
        m = 2.0 * box.size[ax] / wp.abs(halfaxis[ax])
        secondpos = min(1.0 - wp.float32(mul) * bestsegmentpos, m)
      else:  # second point along a face of the box
        # check for overshoot again
        m = 2.0 * min(
          box.size[ax1] / wp.abs(halfaxis[ax1]),
          box.size[ax2] / wp.abs(halfaxis[ax2]),
        )
        secondpos = -min(1.0 + wp.float32(mul) * bestsegmentpos, m)
      secondpos *= wp.float32(mul)

  elif cltype >= 0 and cltype / 3 == 1:  # we are on box's edge
    # Calculate relative orientation between capsule and edge
    # Two possible configurations:
    # - T configuration: c1 = 2^n (no additional contacts)
    # - X configuration: c1 != 2^n (potential additional contacts)
    c1 = axisdir ^ clcorner
    c1 &= 7 - (1 << cledge)  # mask out edge axis to determine configuration

    if c1 == 1 or c1 == 2 or c1 == 4:  # create second contact point
      if cledge == 0:
        ax1 = 1
        ax2 = 2
      if cledge == 1:
        ax1 = 2
        ax2 = 0
      if cledge == 2:
        ax1 = 0
        ax2 = 1
      ax = cledge

      # Then it finds with which face the capsule has a lower angle and switches the axis names
      if wp.abs(axis[ax1]) > wp.abs(axis[ax2]):
        ax1 = ax2
      ax2 = 3 - ax - ax1

      # mul determines direction along capsule axis for second contact point
      if c1 & (1 << ax2):
        mul = 1
        secondpos = 1.0 - bestsegmentpos
      else:
        mul = -1
        secondpos = 1.0 + bestsegmentpos

      # now we have to find out whether we point towards the opposite side or towards one of the
      # sides and also find the farthest point along the capsule that is above the box

      e1 = 2.0 * box.size[ax2] / wp.abs(halfaxis[ax2])
      secondpos = min(e1, secondpos)

      if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):
        e2 = 1.0 - bestboxpos
      else:
        e2 = 1.0 + bestboxpos

      e1 = box.size[ax] * e2 / wp.abs(halfaxis[ax])

      secondpos = min(e1, secondpos)
      secondpos *= wp.float32(mul)

  elif cltype < 0:
    # similarly we handle the case when one capsule's end is closest to a face of the box
    # and find where is the other end pointing to and clamping to the farthest point
    # of the capsule that's above the box
    # if the closest point is inside the box there's no need for a second point

    if clface != -1:  # create second contact point
      mul = wp.where(cltype == -3, 1, -1)
      secondpos = 2.0

      tmp1 = pos - halfaxis * wp.float32(mul)

      for i in range(3):
        if i != clface:
          ha_r = wp.float32(mul) / halfaxis[i]
          e1 = (box.size[i] - tmp1[i]) * ha_r
          if 0 < e1 and e1 < secondpos:
            secondpos = e1

          e1 = (-box.size[i] - tmp1[i]) * ha_r
          if 0 < e1 and e1 < secondpos:
            secondpos = e1

      secondpos *= wp.float32(mul)

  # create sphere in original orientation at first contact point
  s1_pos_l = pos + halfaxis * bestsegmentpos
  s1_pos_g = box.rot @ s1_pos_l + box.pos

  # collide with sphere
  contact = _sphere_box(s1_pos_g, cap.size[0], box.pos, box.rot, box.size, margin)

  if secondpos > -3:  # secondpos was modified
    s2_pos_l = pos + halfaxis * (secondpos + bestsegmentpos)
    s2_pos_g = box.rot @ s2_pos_l + box.pos
    contact2 = _sphere_box(s2_pos_g, cap.size[0], box.pos, box.rot, box.size, margin)

    return contact, contact2, 2

  return contact, ContactFrame(), 1


@wp.func
def plane_box(
  # In:
  plane: Geom,
  box: Geom,
  margin: float,
):
  count = int(0)
  corner = wp.vec3()
  dist = wp.dot(box.pos - plane.pos, plane.normal)

  # Initialize contact frames
  contact1 = ContactFrame(pos=wp.vec3(0.0), frame=wp.mat33(1.0), dist=0.0)
  contact2 = ContactFrame(pos=wp.vec3(0.0), frame=wp.mat33(1.0), dist=0.0)
  contact3 = ContactFrame(pos=wp.vec3(0.0), frame=wp.mat33(1.0), dist=0.0)
  contact4 = ContactFrame(pos=wp.vec3(0.0), frame=wp.mat33(1.0), dist=0.0)
  count = int(0)

  # test all corners, pick bottom 4
  for i in range(8):
    # get corner in local coordinates
    corner.x = wp.where(i & 1, box.size.x, -box.size.x)
    corner.y = wp.where(i & 2, box.size.y, -box.size.y)
    corner.z = wp.where(i & 4, box.size.z, -box.size.z)

    # get corner in global coordinates relative to box center
    corner = box.rot * corner

    # compute distance to plane, skip if too far or pointing up
    ldist = wp.dot(plane.normal, corner)
    if dist + ldist > margin or ldist > 0:
      continue

    cdist = dist + ldist
    frame = make_frame(plane.normal)
    pos = corner + box.pos + (plane.normal * cdist / -2.0)

    if count == 0:
      contact1 = ContactFrame(pos=pos, frame=frame, dist=cdist)
    elif count == 1:
      contact2 = ContactFrame(pos=pos, frame=frame, dist=cdist)
    elif count == 2:
      contact3 = ContactFrame(pos=pos, frame=frame, dist=cdist)
    elif count == 3:
      contact4 = ContactFrame(pos=pos, frame=frame, dist=cdist)

    count += 1
    if count >= 4:
      break

  return contact1, contact2, contact3, contact4, count


_HUGE_VAL = 1e6


@wp.func
def plane_convex(
  # In:
  plane: Geom,
  convex: Geom,
):
  """Calculates contacts between a plane and a convex object."""

  # get points in the convex frame
  plane_pos = wp.transpose(convex.rot) @ (plane.pos - convex.pos)
  n = wp.transpose(convex.rot) @ plane.normal

  # Find support points
  max_support = wp.float32(-_HUGE_VAL)
  for i in range(convex.vertnum):
    support = wp.dot(plane_pos - convex.vert[convex.vertadr + i], n)

    max_support = wp.max(support, max_support)

  threshold = wp.max(0.0, max_support - 1e-3)

  # Store indices in vec4
  indices = wp.vec4i(-1, -1, -1, -1)

  # TODO(team): Explore faster methods like tile_min or even fast pass kernels if the upper bound of vertices in all convexes is small enough such that all vertices fit into shared memory
  # Find point a (first support point)
  a_dist = wp.float32(-_HUGE_VAL)
  for i in range(convex.vertnum):
    support = wp.dot(plane_pos - convex.vert[convex.vertadr + i], n)
    dist = wp.where(support > threshold, 0.0, -_HUGE_VAL)
    if dist > a_dist:
      indices[0] = i
      a_dist = dist
  a = convex.vert[convex.vertadr + indices[0]]

  # Find point b (furthest from a)
  b_dist = wp.float32(-_HUGE_VAL)
  for i in range(convex.vertnum):
    support = wp.dot(plane_pos - convex.vert[convex.vertadr + i], n)
    dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
    dist = wp.length_sq(a - convex.vert[convex.vertadr + i]) + dist_mask
    if dist > b_dist:
      indices[1] = i
      b_dist = dist
  b = convex.vert[convex.vertadr + indices[1]]

  # Find point c (furthest along axis orthogonal to a-b)
  ab = wp.cross(n, a - b)
  c_dist = wp.float32(-_HUGE_VAL)
  for i in range(convex.vertnum):
    support = wp.dot(plane_pos - convex.vert[convex.vertadr + i], n)
    dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
    ap = a - convex.vert[convex.vertadr + i]
    dist = wp.abs(wp.dot(ap, ab)) + dist_mask
    if dist > c_dist:
      indices[2] = i
      c_dist = dist
  c = convex.vert[convex.vertadr + indices[2]]

  # Find point d (furthest from other triangle edges)
  ac = wp.cross(n, a - c)
  bc = wp.cross(n, b - c)
  d_dist = wp.float32(-_HUGE_VAL)
  for i in range(convex.vertnum):
    support = wp.dot(plane_pos - convex.vert[convex.vertadr + i], n)
    dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
    ap = a - convex.vert[convex.vertadr + i]
    bp = b - convex.vert[convex.vertadr + i]
    dist_ap = wp.abs(wp.dot(ap, ac)) + dist_mask
    dist_bp = wp.abs(wp.dot(bp, bc)) + dist_mask
    if dist_ap + dist_bp > d_dist:
      indices[3] = i
      d_dist = dist_ap + dist_bp

  # Prepare contacts
  frame = make_frame(plane.normal)
  contact1 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  contact2 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  contact3 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  contact4 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  count = int(0)

  for i in range(3, -1, -1):
    idx = indices[i]
    unique_count = int(0)
    for j in range(i + 1):
      if indices[j] == idx:
        unique_count = unique_count + 1

    # Check if the index is unique (appears exactly once)
    if unique_count == 1:
      pos = convex.vert[convex.vertadr + idx]
      pos = convex.pos + convex.rot @ pos
      support = wp.dot(plane_pos - convex.vert[convex.vertadr + idx], n)
      dist = -support
      pos = pos - 0.5 * dist * plane.normal

      if count == 0:
        contact1 = ContactFrame(pos=pos, frame=frame, dist=dist)
      elif count == 1:
        contact2 = ContactFrame(pos=pos, frame=frame, dist=dist)
      elif count == 2:
        contact3 = ContactFrame(pos=pos, frame=frame, dist=dist)
      elif count == 3:
        contact4 = ContactFrame(pos=pos, frame=frame, dist=dist)
      count += 1

  return contact1, contact2, contact3, contact4, count


@wp.func
def plane_cylinder(
  # In:
  plane: Geom,
  cylinder: Geom,
  margin: float,
):
  """Calculates contacts between a cylinder and a plane."""
  # Extract plane normal and cylinder axis
  n = plane.normal
  axis = wp.vec3(cylinder.rot[0, 2], cylinder.rot[1, 2], cylinder.rot[2, 2])

  # Project, make sure axis points toward plane
  prjaxis = wp.dot(n, axis)
  if prjaxis > 0:
    axis = -axis
    prjaxis = -prjaxis

  # Compute normal distance from plane to cylinder center
  dist0 = wp.dot(cylinder.pos - plane.pos, n)

  # Remove component of -normal along cylinder axis
  vec = axis * prjaxis - n
  len_sqr = wp.dot(vec, vec)

  # If vector is nondegenerate, normalize and scale by radius
  # Otherwise use cylinder's x-axis scaled by radius
  vec = wp.where(
    len_sqr >= 1e-12,
    vec * (cylinder.size[0] / wp.sqrt(len_sqr)),
    wp.vec3(cylinder.rot[0, 0], cylinder.rot[1, 0], cylinder.rot[2, 0]) * cylinder.size[0],
  )

  # Project scaled vector on normal
  prjvec = wp.dot(vec, n)

  # Scale cylinder axis by half-length
  axis = axis * cylinder.size[1]
  prjaxis = prjaxis * cylinder.size[1]

  frame = make_frame(n)
  count = int(0)

  contact1 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  contact2 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  contact3 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  contact4 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)

  # First contact point (end cap closer to plane)
  dist1 = dist0 + prjaxis + prjvec
  if dist1 <= margin:
    pos1 = cylinder.pos + vec + axis - n * (dist1 * 0.5)
    contact1 = ContactFrame(pos=pos1, frame=frame, dist=dist1)
    count = 1
  else:
    # If nearest point is above margin, no contacts
    return contact1, contact2, contact3, contact4, count

  # Second contact point (end cap farther from plane)
  dist2 = dist0 - prjaxis + prjvec
  if dist2 <= margin:
    pos2 = cylinder.pos + vec - axis - n * (dist2 * 0.5)
    if count == 0:
      contact1 = ContactFrame(pos=pos2, frame=frame, dist=dist2)
    else:
      contact2 = ContactFrame(pos=pos2, frame=frame, dist=dist2)
    count = count + 1

  # Try triangle contact points on side closer to plane
  prjvec1 = -prjvec * 0.5
  dist3 = dist0 + prjaxis + prjvec1
  if dist3 <= margin:
    # Compute sideways vector scaled by radius*sqrt(3)/2
    vec1 = wp.cross(vec, axis)
    vec1 = wp.normalize(vec1) * (cylinder.size[0] * wp.sqrt(3.0) * 0.5)

    # Add contact point A - adjust to closest side
    pos3 = cylinder.pos + vec1 + axis - vec * 0.5 - n * (dist3 * 0.5)
    if count == 0:
      contact1 = ContactFrame(pos=pos3, frame=frame, dist=dist3)
    elif count == 1:
      contact2 = ContactFrame(pos=pos3, frame=frame, dist=dist3)
    elif count:
      contact3 = ContactFrame(pos=pos3, frame=frame, dist=dist3)
    count = count + 1

    # Add contact point B - adjust to closest side
    pos4 = cylinder.pos - vec1 + axis - vec * 0.5 - n * (dist3 * 0.5)
    if count == 0:
      contact1 = ContactFrame(pos=pos4, frame=frame, dist=dist3)
    elif count == 1:
      contact2 = ContactFrame(pos=pos4, frame=frame, dist=dist3)
    elif count == 2:
      contact3 = ContactFrame(pos=pos4, frame=frame, dist=dist3)
    else:
      contact4 = ContactFrame(pos=pos4, frame=frame, dist=dist3)
    count = count + 1

  return contact1, contact2, contact3, contact4, count


class vec8f(wp.types.vector(length=8, dtype=wp.float32)):
  pass


class mat43f(wp.types.matrix(shape=(4, 3), dtype=wp.float32)):
  pass


class mat83f(wp.types.matrix(shape=(8, 3), dtype=wp.float32)):
  pass


@wp.func
def _compute_rotmore(face_idx: int) -> wp.mat33:
  rotmore = wp.mat33(0.0)

  if face_idx == 0:
    rotmore[0, 2] = -1.0
    rotmore[1, 1] = +1.0
    rotmore[2, 0] = +1.0
  elif face_idx == 1:
    rotmore[0, 0] = +1.0
    rotmore[1, 2] = -1.0
    rotmore[2, 1] = +1.0
  elif face_idx == 2:
    rotmore[0, 0] = +1.0
    rotmore[1, 1] = +1.0
    rotmore[2, 2] = +1.0
  elif face_idx == 3:
    rotmore[0, 2] = +1.0
    rotmore[1, 1] = +1.0
    rotmore[2, 0] = -1.0
  elif face_idx == 4:
    rotmore[0, 0] = +1.0
    rotmore[1, 2] = +1.0
    rotmore[2, 1] = -1.0
  elif face_idx == 5:
    rotmore[0, 0] = -1.0
    rotmore[1, 1] = +1.0
    rotmore[2, 2] = -1.0

  return rotmore


@wp.func
def box_box(
  box1: Geom,
  box2: Geom,
  margin: float,
):
  # Compute transforms between box's frames

  # Initialize 8 empty contact frames
  contact1 = ContactFrame()
  contact2 = ContactFrame()
  contact3 = ContactFrame()
  contact4 = ContactFrame()
  contact5 = ContactFrame()
  contact6 = ContactFrame()
  contact7 = ContactFrame()
  contact8 = ContactFrame()

  pos21 = wp.transpose(box1.rot) @ (box2.pos - box1.pos)
  pos12 = wp.transpose(box2.rot) @ (box1.pos - box2.pos)

  rot21 = wp.transpose(box1.rot) @ box2.rot
  rot12 = wp.transpose(rot21)

  rot21abs = wp.matrix_from_rows(wp.abs(rot21[0]), wp.abs(rot21[1]), wp.abs(rot21[2]))
  rot12abs = wp.transpose(rot21abs)

  plen2 = rot21abs @ box2.size
  plen1 = rot12abs @ box1.size

  # Compute axis of maximum separation
  s_sum_3 = 3.0 * (box1.size + box2.size)
  separation = wp.float32(margin + s_sum_3[0] + s_sum_3[1] + s_sum_3[2])
  axis_code = wp.int32(-1)

  # First test: consider boxes' face normals
  for i in range(3):
    c1 = -wp.abs(pos21[i]) + box1.size[i] + plen2[i]

    c2 = -wp.abs(pos12[i]) + box2.size[i] + plen1[i]

    if c1 < -margin or c2 < -margin:
      return contact1, contact2, contact3, contact4, contact5, contact6, contact7, contact8, 0

    if c1 < separation:
      separation = c1
      axis_code = i + 3 * wp.int32(pos21[i] < 0) + 0  # Face of box1
    if c2 < separation:
      separation = c2
      axis_code = i + 3 * wp.int32(pos12[i] < 0) + 6  # Face of box2

  clnorm = wp.vec3(0.0)
  inv = wp.bool(False)
  cle1 = wp.int32(0)
  cle2 = wp.int32(0)

  # Second test: consider cross products of boxes' edges
  for i in range(3):
    for j in range(3):
      # Compute cross product of box edges (potential separating axis)
      if i == 0:
        cross_axis = wp.vec3(0.0, -rot12[j, 2], rot12[j, 1])
      elif i == 1:
        cross_axis = wp.vec3(rot12[j, 2], 0.0, -rot12[j, 0])
      else:
        cross_axis = wp.vec3(-rot12[j, 1], rot12[j, 0], 0.0)

      cross_length = wp.length(cross_axis)
      if cross_length < float(1e-8):
        continue

      cross_axis /= cross_length

      box_dist = wp.dot(pos21, cross_axis)
      c3 = wp.float32(0.0)

      # Project box half-sizes onto the potential separating axis
      for k in range(3):
        if k != i:
          c3 += box1.size[k] * wp.abs(cross_axis[k])
        if k != j:
          c3 += box2.size[k] * rot21abs[i, 3 - k - j] / cross_length

      c3 -= wp.abs(box_dist)

      # Early exit: no collision if separated along this axis
      if c3 < -margin:
        return contact1, contact2, contact3, contact4, contact5, contact6, contact7, contact8, 0

      # Track minimum separation and which edge-edge pair it occurs on
      if c3 < separation * (1.0 - 1e-12):
        separation = c3
        # Determine which corners/edges are closest
        cle1 = 0
        cle2 = 0

        for k in range(3):
          if k != i and (int(cross_axis[k] > 0) ^ int(box_dist < 0)):
            cle1 += 1 << k
          if k != j and (int(rot21[i, 3 - k - j] > 0) ^ int(box_dist < 0) ^ int((k - j + 3) % 3 == 1)):
            cle2 += 1 << k

        axis_code = 12 + i * 3 + j
        clnorm = cross_axis
        inv = box_dist < 0

  # No axis with separation < margin found
  if axis_code == -1:
    return contact1, contact2, contact3, contact4, contact5, contact6, contact7, contact8, 0

  points = mat83f()
  depth = vec8f()
  max_con_pair = 8
  # 8 contacts should suffice for most configurations

  if axis_code < 12:
    # Handle face-vertex collision
    face_idx = axis_code % 6
    box_idx = axis_code / 6
    rotmore = _compute_rotmore(face_idx)

    r = rotmore @ wp.where(box_idx, rot12, rot21)
    p = rotmore @ wp.where(box_idx, pos12, pos21)
    ss = wp.abs(rotmore @ wp.where(box_idx, box2.size, box1.size))
    s = wp.where(box_idx, box1.size, box2.size)
    rt = wp.transpose(r)

    lx, ly, hz = ss[0], ss[1], ss[2]
    p[2] -= hz

    clcorner = wp.int32(0)  # corner of non-face box with least axis separation

    for i in range(3):
      if r[2, i] < 0:
        clcorner += 1 << i

    lp = p
    for i in range(wp.static(3)):
      lp += rt[i] * s[i] * wp.where(clcorner & 1 << i, 1.0, -1.0)

    m = wp.int32(1)
    dirs = wp.int32(0)

    cn1 = wp.vec3(0.0)
    cn2 = wp.vec3(0.0)

    for i in range(3):
      if wp.abs(r[2, i]) < 0.5:
        if not dirs:
          cn1 = rt[i] * s[i] * wp.where(clcorner & (1 << i), -2.0, 2.0)
        else:
          cn2 = rt[i] * s[i] * wp.where(clcorner & (1 << i), -2.0, 2.0)

        dirs += 1

    k = dirs * dirs

    # Find potential contact points

    n = wp.int32(0)

    for i in range(k):
      for q in range(2):
        # lines_a and lines_b (lines between corners) computed on the fly
        lav = lp + wp.where(i < 2, wp.vec3(0.0), wp.where(i == 2, cn1, cn2))
        lbv = wp.where(i == 0 or i == 3, cn1, cn2)

        if wp.abs(lbv[q]) > float(1e-8):
          br = 1.0 / lbv[q]
          for j in range(-1, 2, 2):
            l = ss[q] * wp.float32(j)
            c1 = (l - lav[q]) * br
            if c1 < 0 or c1 > 1:
              continue
            c2 = lav[1 - q] + lbv[1 - q] * c1
            if wp.abs(c2) > ss[1 - q]:
              continue

            points[n] = lav + c1 * lbv
            n += 1

    if dirs == 2:
      ax = cn1[0]
      bx = cn2[0]
      ay = cn1[1]
      by = cn2[1]
      C = 1.0 / (ax * by - bx * ay)

      for i in range(4):
        llx = wp.where(i / 2, lx, -lx)
        lly = wp.where(i % 2, ly, -ly)

        x = llx - lp[0]
        y = lly - lp[1]

        u = (x * by - y * bx) * C
        v = (y * ax - x * ay) * C

        if u > 0 and v > 0 and u < 1 and v < 1:
          points[n] = wp.vec3(llx, lly, lp[2] + u * cn1[2] + v * cn2[2])
          n += 1

    for i in range(1 << dirs):
      tmpv = lp + wp.float32(i & 1) * cn1 + wp.float32((i & 2) != 0) * cn2
      if tmpv[0] > -lx and tmpv[0] < lx and tmpv[1] > -ly and tmpv[1] < ly:
        points[n] = tmpv
        n += 1

    m = n
    n = wp.int32(0)

    for i in range(m):
      if points[i][2] > margin:
        continue
      if i != n:
        points[n] = points[i]

      points[n, 2] *= 0.5
      depth[n] = points[n, 2]
      n += 1

    # Set up contact frame
    rw = wp.where(box_idx, box2.rot, box1.rot) @ wp.transpose(rotmore)
    pw = wp.where(box_idx, box2.pos, box1.pos)
    normal = wp.where(box_idx, -1.0, 1.0) * wp.transpose(rw)[2]

  else:
    # Handle edge-edge collision
    edge1 = (axis_code - 12) / 3
    edge2 = (axis_code - 12) % 3

    # Set up non-contacting edges ax1, ax2 for box2 and pax1, pax2 for box 1
    ax1 = wp.int(1 - (edge2 & 1))
    ax2 = wp.int(2 - (edge2 & 2))

    pax1 = wp.int(1 - (edge1 & 1))
    pax2 = wp.int(2 - (edge1 & 2))

    if rot21abs[edge1, ax1] < rot21abs[edge1, ax2]:
      ax1, ax2 = ax2, ax1

    if rot12abs[edge2, pax1] < rot12abs[edge2, pax2]:
      pax1, pax2 = pax2, pax1

    rotmore = _compute_rotmore(wp.where(cle1 & (1 << pax2), pax2, pax2 + 3))

    # Transform coordinates for edge-edge contact calculation
    p = rotmore @ pos21
    rnorm = rotmore @ clnorm
    r = rotmore @ rot21
    rt = wp.transpose(r)
    s = wp.abs(wp.transpose(rotmore) @ box1.size)

    lx, ly, hz = s[0], s[1], s[2]
    p[2] -= hz

    # Calculate closest box2 face

    points[0] = (
      p
      + rt[ax1] * box2.size[ax1] * wp.where(cle2 & (1 << ax1), 1.0, -1.0)
      + rt[ax2] * box2.size[ax2] * wp.where(cle2 & (1 << ax2), 1.0, -1.0)
    )
    points[1] = points[0] - rt[edge2] * box2.size[edge2]
    points[0] += rt[edge2] * box2.size[edge2]

    points[2] = (
      p
      + rt[ax1] * box2.size[ax1] * wp.where(cle2 & (1 << ax1), -1.0, 1.0)
      + rt[ax2] * box2.size[ax2] * wp.where(cle2 & (1 << ax2), 1.0, -1.0)
    )

    points[3] = points[2] - rt[edge2] * box2.size[edge2]
    points[2] += rt[edge2] * box2.size[edge2]

    n = 4

    # Set up coordinate axes for contact face of box2
    axi_lp = points[0]
    axi_cn1 = points[1] - points[0]
    axi_cn2 = points[2] - points[0]

    # Check if contact normal is valid
    if wp.abs(rnorm[2]) < float(1e-8):
      return contact1, contact2, contact3, contact4, contact5, contact6, contact7, contact8, 0  # Shouldn't happen

    # Calculate inverse normal for projection
    innorm = wp.where(inv, -1.0, 1.0) / rnorm[2]

    pu = mat43f()

    # Project points onto contact plane
    for i in range(4):
      pu[i] = points[i]
      c_scl = points[i, 2] * wp.where(inv, -1.0, 1.0) * innorm
      points[i] -= rnorm * c_scl

    pts_lp = points[0]
    pts_cn1 = points[1] - points[0]
    pts_cn2 = points[2] - points[0]

    n = wp.int32(0)

    for i in range(4):
      for q in range(2):
        la = pts_lp[q] + wp.where(i < 2, 0.0, wp.where(i == 2, pts_cn1[q], pts_cn2[q]))
        lb = wp.where(i == 0 or i == 3, pts_cn1[q], pts_cn2[q])
        lc = pts_lp[1 - q] + wp.where(i < 2, 0.0, wp.where(i == 2, pts_cn1[1 - q], pts_cn2[1 - q]))
        ld = wp.where(i == 0 or i == 3, pts_cn1[1 - q], pts_cn2[1 - q])

        # linesu_a and linesu_b (lines between corners) computed on the fly
        lua = axi_lp + wp.where(i < 2, wp.vec3(0.0), wp.where(i == 2, axi_cn1, axi_cn2))
        lub = wp.where(i == 0 or i == 3, axi_cn1, axi_cn2)

        if wp.abs(lb) > float(1e-8):
          br = 1.0 / lb
          for j in range(-1, 2, 2):
            if n == max_con_pair:
              break
            l = s[q] * wp.float32(j)
            c1 = (l - la) * br
            if c1 < 0 or c1 > 1:
              continue
            c2 = lc + ld * c1
            if wp.abs(c2) > s[1 - q]:
              continue
            if (lua[2] + lub[2] * c1) * innorm > margin:
              continue

            points[n] = lua * 0.5 + c1 * lub * 0.5
            points[n, q] += 0.5 * l
            points[n, 1 - q] += 0.5 * c2
            depth[n] = points[n, 2] * innorm * 2.0
            n += 1

    nl = n

    ax = pts_cn1[0]
    bx = pts_cn2[0]
    ay = pts_cn1[1]
    by = pts_cn2[1]
    C = 1.0 / (ax * by - bx * ay)

    for i in range(4):
      if n == max_con_pair:
        break
      llx = wp.where(i / 2, lx, -lx)
      lly = wp.where(i % 2, ly, -ly)

      x = llx - pts_lp[0]
      y = lly - pts_lp[1]

      u = (x * by - y * bx) * C
      v = (y * ax - x * ay) * C

      if nl == 0:
        if (u < 0 or u > 0) and (v < 0 or v > 1):
          continue
      elif u < 0 or v < 0 or u > 1 or v > 1:
        continue

      u = wp.clamp(u, 0.0, 1.0)
      v = wp.clamp(v, 0.0, 1.0)
      w = 1.0 - u - v
      vtmp = pu[0] * w + pu[1] * u + pu[2] * v

      points[n] = wp.vec3(llx, lly, 0.0)

      vtmp2 = points[n] - vtmp
      tc1 = wp.length_sq(vtmp2)
      if vtmp[2] > 0 and tc1 > margin * margin:
        continue

      points[n] = 0.5 * (points[n] + vtmp)

      depth[n] = wp.sqrt(tc1) * wp.where(vtmp[2] < 0, -1.0, 1.0)
      n += 1

    nf = n

    for i in range(4):
      if n >= max_con_pair:
        break
      x = pu[i, 0]
      y = pu[i, 1]
      if nl == 0 and nf != 0:
        if (x < -lx or x > lx) and (y < -ly or y > ly):
          continue
      elif x < -lx or x > lx or y < -ly or y > ly:
        continue

      c1 = wp.float32(0)

      for j in range(2):
        if pu[i, j] < -s[j]:
          c1 += (pu[i, j] + s[j]) * (pu[i, j] + s[j])
        elif pu[i, j] > s[j]:
          c1 += (pu[i, j] - s[j]) * (pu[i, j] - s[j])

      c1 += pu[i, 2] * innorm * pu[i, 2] * innorm

      if pu[i, 2] > 0 and c1 > margin * margin:
        continue

      tmp_p = wp.vec3(pu[i, 0], pu[i, 1], 0.0)

      for j in range(2):
        if pu[i, j] < -s[j]:
          tmp_p[j] = -s[j] * 0.5
        elif pu[i, j] > s[j]:
          tmp_p[j] = +s[j] * 0.5

      tmp_p += pu[i]
      points[n] = tmp_p * 0.5

      depth[n] = wp.sqrt(c1) * wp.where(pu[i, 2] < 0, -1.0, 1.0)
      n += 1

    # Set up contact data for all points
    rw = box1.rot @ wp.transpose(rotmore)
    pw = box1.pos
    normal = wp.where(inv, -1.0, 1.0) * rw @ rnorm

  frame = make_frame(normal)

  # prepare contacts

  for i in range(min(8, n)):
    points[i, 2] += hz
    pos = rw @ points[i] + pw

    contact = ContactFrame(pos=pos, frame=frame, dist=depth[i])

    if i == 0:
      contact1 = contact
    elif i == 1:
      contact2 = contact
    elif i == 2:
      contact3 = contact
    elif i == 3:
      contact4 = contact
    elif i == 4:
      contact5 = contact
    elif i == 5:
      contact6 = contact
    elif i == 6:
      contact7 = contact
    elif i == 7:
      contact8 = contact

  return contact1, contact2, contact3, contact4, contact5, contact6, contact7, contact8, min(8, n)


# newton/geometry/types.py (or within geometry/__init__.py)

# Shape geometry types
GEO_SPHERE = wp.constant(0)
GEO_BOX = wp.constant(1)
GEO_CAPSULE = wp.constant(2)
GEO_CYLINDER = wp.constant(3)
GEO_CONE = wp.constant(4)
GEO_MESH = wp.constant(5)
GEO_SDF = wp.constant(6)
GEO_PLANE = wp.constant(7)
GEO_NONE = wp.constant(8)
GEO_CONVEX = wp.constant(9)
GEO_ELLIPSOID = wp.constant(10)


@wp.struct
class ShapeGeometry:  # Renamed from ModelShapeGeometry in model.py
  """
  Represents the geometry of a set of shapes
  """

  type: wp.array(dtype=wp.int32)
  is_solid: wp.array(dtype=bool)
  thickness: wp.array(dtype=float)
  source: wp.array(dtype=wp.uint64)  # ID for wp.Mesh or wp.Volume
  scale: wp.array(dtype=wp.vec3)
  filter: wp.array(dtype=int)


@wp.struct
class ContactGeometry:  # This remains the output data structure
  # Soft contacts
  soft_contact_count: wp.array(dtype=int)
  soft_contact_particle: wp.array(dtype=int)
  soft_contact_shape: wp.array(dtype=int)

  # Rigid contacts
  rigid_contact_count: wp.array(dtype=int)
  rigid_contact_point0_world: wp.array(dtype=wp.vec3)
  rigid_contact_point1_world: wp.array(dtype=wp.vec3)
  rigid_contact_normal_world: wp.array(dtype=wp.vec3)
  rigid_contact_thickness: wp.array(dtype=float)
  rigid_contact_shape0_idx: wp.array(dtype=int)
  rigid_contact_shape1_idx: wp.array(dtype=int)


@wp.func
def write_contact_newton(
  # In:
  nconmax_in: int,
  pos0_in: wp.vec3,
  pos1_in: wp.vec3,
  normal_in: wp.vec3,
  thickness_in: float,
  shape0_idx: int,
  shape1_idx: int,
  # Out:
  contact_geometry: ContactGeometry,
):
  cid = wp.atomic_add(contact_geometry.rigid_contact_count, 0, 1)
  if cid < nconmax_in:
    # Set contact points - for now just using pos_in for both points
    contact_geometry.rigid_contact_point0_world[cid] = pos0_in
    contact_geometry.rigid_contact_point1_world[cid] = pos1_in

    # Set normal
    contact_geometry.rigid_contact_normal_world[cid] = normal_in

    # Set thickness
    contact_geometry.rigid_contact_thickness[cid] = thickness_in

    # Set shape indices
    contact_geometry.rigid_contact_shape0_idx[cid] = shape0_idx
    contact_geometry.rigid_contact_shape1_idx[cid] = shape1_idx


@wp.kernel
def _primitive_narrowphase_newton(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  mesh_vertadr: wp.array(dtype=int),
  mesh_vertnum: wp.array(dtype=int),
  # Data in:
  nconmax_in: int,
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  collision_pair_in: wp.array(dtype=wp.vec2i),
  ncollision_in: wp.array(dtype=int),
  # Data out:
  contact_geometry: ContactGeometry,
):
  tid = wp.tid()

  if tid >= ncollision_in[0]:
    return

  geoms = collision_pair_in[tid]
  g1 = geoms[0]
  g2 = geoms[1]

  # Get geometry objects for both shapes
  geom1 = Geom(
    pos=geom_xpos_in[g1],
    rot=geom_xmat_in[g1],
    size=geom_size[g1],
    vertadr=mesh_vertadr[g1],
    vertnum=mesh_vertnum[g1],
  )

  geom2 = Geom(
    pos=geom_xpos_in[g2],
    rot=geom_xmat_in[g2],
    size=geom_size[g2],
    vertadr=mesh_vertadr[g2],
    vertnum=mesh_vertnum[g2],
  )

  type1 = geom_type[g1]
  type2 = geom_type[g2]

  # Handle different collision type pairs
  if type1 == GEO_PLANE and type2 == GEO_SPHERE:
    contact = plane_sphere(geom1, geom2)
    write_contact_newton(
      nconmax_in,
      contact.pos,
      contact.pos,  # For plane-sphere, both contact points are the same
      contact.frame[0],  # Normal is first row of frame
      contact.dist,
      g1,
      g2,
      contact_geometry,
    )

  elif type1 == GEO_SPHERE and type2 == GEO_SPHERE:
    contact = sphere_sphere(geom1, geom2)
    write_contact_newton(
      nconmax_in,
      contact.pos,
      contact.pos,  # For sphere-sphere, both contact points are the same
      contact.frame[0],  # Normal is first row of frame
      contact.dist,
      g1,
      g2,
      contact_geometry,
    )

  elif type1 == GEO_PLANE and type2 == GEO_CAPSULE:
    contact1, contact2 = plane_capsule(geom1, geom2)
    # Write both contacts for plane-capsule
    write_contact_newton(
      nconmax_in,
      contact1.pos,
      contact1.pos,
      contact1.frame[0],
      contact1.dist,
      g1,
      g2,
      contact_geometry,
    )
    write_contact_newton(
      nconmax_in,
      contact2.pos,
      contact2.pos,
      contact2.frame[0],
      contact2.dist,
      g1,
      g2,
      contact_geometry,
    )

  elif type1 == GEO_SPHERE and type2 == GEO_CAPSULE:
    contact = sphere_capsule(geom1, geom2)
    write_contact_newton(
      nconmax_in,
      contact.pos,
      contact.pos,
      contact.frame[0],
      contact.dist,
      g1,
      g2,
      contact_geometry,
    )

  elif type1 == GEO_SPHERE and type2 == GEO_BOX:
    contact = sphere_box(geom1, geom2, 0.0)  # No margin for Newton contacts
    write_contact_newton(
      nconmax_in,
      contact.pos,
      contact.pos,
      contact.frame[0],
      contact.dist,
      g1,
      g2,
      contact_geometry,
    )

  elif type1 == GEO_CAPSULE and type2 == GEO_CAPSULE:
    contact = capsule_capsule(geom1, geom2)
    write_contact_newton(
      nconmax_in,
      contact.pos,
      contact.pos,
      contact.frame[0],
      contact.dist,
      g1,
      g2,
      contact_geometry,
    )

  elif type1 == GEO_PLANE and type2 == GEO_BOX:
    contact1, contact2, contact3, contact4, count = plane_box(geom1, geom2, 0.0)  # No margin for Newton
    for i in range(count):
      contact = contact1
      if i == 1:
        contact = contact2
      elif i == 2:
        contact = contact3
      elif i == 3:
        contact = contact4
      write_contact_newton(
        nconmax_in,
        contact.pos,
        contact.pos,
        contact.frame[0],
        contact.dist,
        g1,
        g2,
        contact_geometry,
      )

  elif type1 == GEO_PLANE and type2 == GEO_MESH:
    contact1, contact2, contact3, contact4, count = plane_convex(geom1, geom2)
    for i in range(count):
      contact = contact1
      if i == 1:
        contact = contact2
      elif i == 2:
        contact = contact3
      elif i == 3:
        contact = contact4
      write_contact_newton(
        nconmax_in,
        contact.pos,
        contact.pos,
        contact.frame[0],
        contact.dist,
        g1,
        g2,
        contact_geometry,
      )

  elif type1 == GEO_SPHERE and type2 == GEO_CYLINDER:
    contact = sphere_cylinder(geom1, geom2)
    write_contact_newton(
      nconmax_in,
      contact.pos,
      contact.pos,
      contact.frame[0],
      contact.dist,
      g1,
      g2,
      contact_geometry,
    )

  elif type1 == GEO_PLANE and type2 == GEO_CYLINDER:
    contact1, contact2, contact3, contact4, count = plane_cylinder(geom1, geom2, 0.0)  # No margin for Newton
    for i in range(count):
      contact = contact1
      if i == 1:
        contact = contact2
      elif i == 2:
        contact = contact3
      elif i == 3:
        contact = contact4
      write_contact_newton(
        nconmax_in,
        contact.pos,
        contact.pos,
        contact.frame[0],
        contact.dist,
        g1,
        g2,
        contact_geometry,
      )

  elif type1 == GEO_BOX and type2 == GEO_BOX:
    contact1, contact2, contact3, contact4, contact5, contact6, contact7, contact8, count = box_box(
      geom1, geom2, 0.0
    )  # No margin for Newton
    for i in range(count):
      contact = contact1
      if i == 1:
        contact = contact2
      elif i == 2:
        contact = contact3
      elif i == 3:
        contact = contact4
      elif i == 4:
        contact = contact5
      elif i == 5:
        contact = contact6
      elif i == 6:
        contact = contact7
      elif i == 7:
        contact = contact8
      write_contact_newton(
        nconmax_in,
        contact.pos,
        contact.pos,
        contact.frame[0],
        contact.dist,
        g1,
        g2,
        contact_geometry,
      )

  elif type1 == GEO_CAPSULE and type2 == GEO_BOX:
    contact1, contact2, count = capsule_box(geom1, geom2, 0.0)  # No margin for Newton
    for i in range(count):
      contact = contact1 if i == 0 else contact2
      write_contact_newton(
        nconmax_in,
        contact.pos,
        contact.pos,
        contact.frame[0],
        contact.dist,
        g1,
        g2,
        contact_geometry,
      )


def primitive_narrowphase_newton(
  # Model:
  shape_geometry: ShapeGeometry,
  convex_vert: wp.array(dtype=wp.vec3),  # Global vertex buffer of convex hulls
  # Data in:
  nconmax_in: int,
  shape_transform: wp.array(dtype=wp.transform),
  collision_pair_in: wp.array(dtype=wp.vec2i),
  ncollision_in: wp.array(dtype=int),
  # Data out:
  contact_geometry: ContactGeometry,
):
  """Launch the Newton collision detection kernel.

  This function handles primitive collision detection for Newton physics.
  It's a simplified version compared to MuJoCo's collision detection,
  focusing only on the core geometric contact computation without
  additional parameters like friction, margins etc.
  """
  wp.launch(
    _primitive_narrowphase_newton,
    dim=nconmax_in,
    inputs=[
      shape_geometry.geom_type,
      shape_geometry.geom_dataid,
      shape_geometry.geom_size,
      shape_geometry.mesh_vertadr,
      shape_geometry.mesh_vertnum,
      convex_vert,
      nconmax_in,
      shape_transform.pos,
      shape_transform.rot,
      collision_pair_in,
      ncollision_in,
    ],
    outputs=[contact_geometry],
  )
