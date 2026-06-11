# Copyright 2025 The Newton Developers
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
"""Smooth (differentiable) collision recomputation for autodifferentiation.

This module provides differentiable replacements for the collision pipeline's
contact geometry and constraint assembly. It runs *after* the discrete pipeline
and overwrites contact.{dist, pos, frame} and efc.{J, pos, D, aref, vel} with
smooth values that Warp can differentiate through.

Supported geom type pairs:
  - plane-sphere, sphere-sphere, sphere-capsule
  - capsule-capsule (2 contacts), plane-capsule (2 contacts)

Unsupported types (box, mesh, convex, etc.) are no-ops that keep discrete
values (zero gradient through those contacts).
"""

from typing import Tuple

import warp as wp

from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.types import MJ_MINVAL
from mujoco_warp._src.types import DisableBit

wp.set_module_options({"enable_backward": True})


# ============================================================================
# Custom types (matching collision_primitive_core.py)
# ============================================================================


class mat23f(wp.types.matrix(shape=(2, 3), dtype=wp.float32)):
  pass


# ============================================================================
# Smooth distance functions
# ============================================================================


@wp.func
def smooth_plane_sphere(
  # In:
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  sphere_pos: wp.vec3,
  sphere_radius: float,
) -> Tuple[float, wp.vec3]:
  """Plane-sphere distance (already smooth)."""
  dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@wp.func
def smooth_sphere_sphere(
  # In:
  pos1: wp.vec3,
  radius1: float,
  pos2: wp.vec3,
  radius2: float,
) -> Tuple[float, wp.vec3, wp.vec3]:
  """Sphere-sphere distance with smooth normalization at coincident centers."""
  dir = pos2 - pos1
  raw_dist = wp.length(dir)
  # Smooth normalization: replaces if dist==0 branch
  n = dir / wp.max(raw_dist, 1e-8)
  dist = raw_dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)
  return dist, pos, n


@wp.func
def smooth_sphere_capsule(
  # In:
  sphere_pos: wp.vec3,
  sphere_radius: float,
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
) -> Tuple[float, wp.vec3, wp.vec3]:
  """Sphere-capsule distance using wp.clamp (subdifferentiable at boundary)."""
  segment = capsule_axis * capsule_half_length
  seg_start = capsule_pos - segment
  seg_end = capsule_pos + segment

  # Closest point on capsule centerline to sphere center
  ab = seg_end - seg_start
  t = wp.dot(sphere_pos - seg_start, ab) / (wp.dot(ab, ab) + 1e-6)
  pt = seg_start + wp.clamp(t, 0.0, 1.0) * ab

  return smooth_sphere_sphere(sphere_pos, sphere_radius, pt, capsule_radius)


@wp.func
def smooth_capsule_capsule(
  # In:
  cap1_pos: wp.vec3,
  cap1_axis: wp.vec3,
  cap1_radius: float,
  cap1_half_length: float,
  cap2_pos: wp.vec3,
  cap2_axis: wp.vec3,
  cap2_radius: float,
  cap2_half_length: float,
  margin: float,
) -> Tuple[wp.vec2, mat23f, mat23f]:
  """Capsule-capsule distance returning 2 contacts, regularized for parallel axes."""
  contact_dist = wp.vec2(wp.inf, wp.inf)
  contact_pos = mat23f()
  contact_normal = mat23f()

  axis1 = cap1_axis * cap1_half_length
  axis2 = cap2_axis * cap2_half_length
  dif = cap1_pos - cap2_pos

  ma = wp.dot(axis1, axis1)
  mb = -wp.dot(axis1, axis2)
  mc = wp.dot(axis2, axis2)
  u = -wp.dot(axis1, dif)
  v = wp.dot(axis2, dif)
  det = ma * mc - mb * mb

  # Regularized determinant: smooth handling of near-parallel axes
  det_abs = wp.abs(det)
  det_sign = wp.where(det >= 0.0, 1.0, -1.0)
  det_reg = det_sign * wp.max(det_abs, 1e-10)

  # Blend: use non-parallel path when |det| > threshold, parallel otherwise
  # Smooth blending factor
  blend_threshold = 1e-8
  alpha = wp.min(det_abs / wp.max(blend_threshold, 1e-15), 1.0)

  # -- Non-parallel path --
  inv_det = 1.0 / det_reg
  x1_np = (mc * u - mb * v) * inv_det
  x2_np = (ma * v - mb * u) * inv_det

  # Clamp with recomputation (smooth via wp.clamp)
  x1_np = wp.clamp(x1_np, -1.0, 1.0)
  x2_np = wp.clamp(x2_np, -1.0, 1.0)

  # Re-clamp for consistency
  x2_np = wp.clamp((v + mb * x1_np) / wp.max(mc, 1e-10), -1.0, 1.0)
  x1_np = wp.clamp((u - mb * x2_np) / wp.max(ma, 1e-10), -1.0, 1.0)

  vec1_np = cap1_pos + axis1 * x1_np
  vec2_np = cap2_pos + axis2 * x2_np
  dist_np, pos_np, normal_np = smooth_sphere_sphere(vec1_np, cap1_radius, vec2_np, cap2_radius)

  # -- Parallel path: test 4 endpoint pairs, keep first 2 --
  # Endpoint 1: x1 = 1
  vec1_a = cap1_pos + axis1
  x2_a = wp.clamp((v - mb) / wp.max(mc, 1e-10), -1.0, 1.0)
  vec2_a = cap2_pos + axis2 * x2_a
  dist_a, pos_a, normal_a = smooth_sphere_sphere(vec1_a, cap1_radius, vec2_a, cap2_radius)

  # Endpoint 2: x1 = -1
  vec1_b = cap1_pos - axis1
  x2_b = wp.clamp((v + mb) / wp.max(mc, 1e-10), -1.0, 1.0)
  vec2_b = cap2_pos + axis2 * x2_b
  dist_b, pos_b, normal_b = smooth_sphere_sphere(vec1_b, cap1_radius, vec2_b, cap2_radius)

  # Endpoint 3: x2 = 1
  vec2_c = cap2_pos + axis2
  x1_c = wp.clamp((u - mb) / wp.max(ma, 1e-10), -1.0, 1.0)
  vec1_c = cap1_pos + axis1 * x1_c
  dist_c, pos_c, normal_c = smooth_sphere_sphere(vec1_c, cap1_radius, vec2_c, cap2_radius)

  # Endpoint 4: x2 = -1
  vec2_d = cap2_pos - axis2
  x1_d = wp.clamp((u + mb) / wp.max(ma, 1e-10), -1.0, 1.0)
  vec1_d = cap1_pos + axis1 * x1_d
  dist_d, pos_d, normal_d = smooth_sphere_sphere(vec1_d, cap1_radius, vec2_d, cap2_radius)

  # Sort 4 endpoints by distance, pick best 2 for parallel contacts
  # Contact 0: best of all 4
  par_dist0 = dist_a
  par_pos0 = pos_a
  par_normal0 = normal_a

  if dist_b < par_dist0:
    par_dist0 = dist_b
    par_pos0 = pos_b
    par_normal0 = normal_b
  if dist_c < par_dist0:
    par_dist0 = dist_c
    par_pos0 = pos_c
    par_normal0 = normal_c
  if dist_d < par_dist0:
    par_dist0 = dist_d
    par_pos0 = pos_d
    par_normal0 = normal_d

  # Contact 1: second best
  par_dist1 = wp.inf
  par_pos1 = wp.vec3(0.0)
  par_normal1 = wp.vec3(1.0, 0.0, 0.0)

  if dist_a <= margin and dist_a != par_dist0:
    par_dist1 = dist_a
    par_pos1 = pos_a
    par_normal1 = normal_a
  if dist_b <= margin and dist_b != par_dist0:
    if dist_b < par_dist1:
      par_dist1 = dist_b
      par_pos1 = pos_b
      par_normal1 = normal_b
  if dist_c <= margin and dist_c != par_dist0:
    if dist_c < par_dist1:
      par_dist1 = dist_c
      par_pos1 = pos_c
      par_normal1 = normal_c
  if dist_d <= margin and dist_d != par_dist0:
    if dist_d < par_dist1:
      par_dist1 = dist_d
      par_pos1 = pos_d
      par_normal1 = normal_d

  # Blend between non-parallel (1 contact) and parallel (2 contacts)
  # Non-parallel: contact 0 = np result, contact 1 = inf
  # Parallel: contact 0, 1 from sorted endpoints
  blend_dist0 = alpha * dist_np + (1.0 - alpha) * par_dist0
  blend_pos0 = alpha * pos_np + (1.0 - alpha) * par_pos0
  blend_normal0 = alpha * normal_np + (1.0 - alpha) * par_normal0
  # Renormalize blended normal
  blend_normal0 = blend_normal0 / wp.max(wp.length(blend_normal0), 1e-8)

  # Contact 1: only from parallel path (non-parallel has 1 contact)
  blend_dist1 = (1.0 - alpha) * par_dist1 + alpha * wp.inf

  if blend_dist0 <= margin:
    contact_dist[0] = blend_dist0
    contact_pos[0] = blend_pos0
    contact_normal[0] = blend_normal0

  if blend_dist1 <= margin:
    contact_dist[1] = blend_dist1
    contact_pos[1] = par_pos1
    contact_normal[1] = par_normal1

  return contact_dist, contact_pos, contact_normal


@wp.func
def smooth_plane_capsule(
  # In:
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
) -> Tuple[wp.vec2, mat23f, wp.mat33]:
  """Plane-capsule distance returning 2 contacts (already smooth)."""
  n = plane_normal
  axis = capsule_axis
  segment = axis * capsule_half_length

  # Build contact frame (smooth version matching collision_primitive_core.py)
  proj = axis - n * wp.dot(n, axis)
  proj_len = wp.length(proj)
  b = proj / wp.max(proj_len, 1e-8)

  # Fallback when capsule axis is nearly parallel to plane normal
  if proj_len < 0.5:
    if -0.5 < n[1] and n[1] < 0.5:
      b = wp.vec3(0.0, 1.0, 0.0)
    else:
      b = wp.vec3(0.0, 0.0, 1.0)

  c = wp.cross(n, b)
  frame = wp.mat33(n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2])

  # Two contacts at capsule endpoints
  dist1, pos1 = smooth_plane_sphere(n, plane_pos, capsule_pos + segment, capsule_radius)
  dist2, pos2 = smooth_plane_sphere(n, plane_pos, capsule_pos - segment, capsule_radius)

  dist = wp.vec2(dist1, dist2)
  pos = mat23f(pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2])

  return dist, pos, frame


@wp.func
def smooth_make_frame(normal: wp.vec3) -> wp.mat33:
  """Construct contact frame from normal with smooth tangent directions."""
  a = normal / wp.max(wp.length(normal), 1e-8)

  # Gram-Schmidt orthogonalization (same as math.orthogonals but using
  # wp.where instead of branching on a[1] for smoother gradients)
  y = wp.vec3(0.0, 1.0, 0.0)
  z = wp.vec3(0.0, 0.0, 1.0)
  b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)
  b = b - a * wp.dot(a, b)
  b_len = wp.length(b)
  b = b / wp.max(b_len, 1e-8)
  c = wp.cross(a, b)

  return wp.mat33(
    a[0],
    a[1],
    a[2],
    b[0],
    b[1],
    b[2],
    c[0],
    c[1],
    c[2],
  )


# ============================================================================
# Smooth contact recomputation kernel
# ============================================================================


@wp.kernel
def _smooth_recompute_kernel(
  # Model:
  geom_type: wp.array(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  geom_size: wp.array2d(dtype=wp.vec3),
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  contact_geom_in: wp.array(dtype=wp.vec2i),
  contact_worldid_in: wp.array(dtype=int),
  contact_geomcollisionid_in: wp.array(dtype=int),
  nacon_in: wp.array(dtype=int),
  # Data out:
  contact_dist_out: wp.array(dtype=float),
  contact_pos_out: wp.array(dtype=wp.vec3),
  contact_frame_out: wp.array(dtype=wp.mat33),
):
  cid = wp.tid()

  if cid >= nacon_in[0]:
    return

  geoms = contact_geom_in[cid]
  g1 = geoms[0]
  g2 = geoms[1]

  # Skip flex contacts (geom id = -1)
  if g1 < 0 or g2 < 0:
    return

  worldid = contact_worldid_in[cid]
  subcid = contact_geomcollisionid_in[cid]
  t1 = geom_type[g1]
  t2 = geom_type[g2]

  # Geom poses (differentiable from Phase 1 kinematics)
  pos1 = geom_xpos_in[worldid, g1]
  pos2 = geom_xpos_in[worldid, g2]
  mat1 = geom_xmat_in[worldid, g1]
  mat2 = geom_xmat_in[worldid, g2]

  # Geom sizes (model constants — use worldid=0 for batched models)
  size_id = worldid % geom_size.shape[0]
  size1 = geom_size[size_id, g1]
  size2 = geom_size[size_id, g2]

  # Dispatch based on geom type pair
  # Geom types: PLANE=0, HFIELD=1, SPHERE=2, CAPSULE=3, ELLIPSOID=4,
  #             CYLINDER=5, BOX=6, MESH=7, SDF=8

  handled = False

  # plane-sphere
  if t1 == 0 and t2 == 2:
    plane_normal = wp.vec3(mat1[0, 2], mat1[1, 2], mat1[2, 2])
    dist, pos = smooth_plane_sphere(plane_normal, pos1, pos2, size2[0])
    frame = smooth_make_frame(plane_normal)
    contact_dist_out[cid] = dist
    contact_pos_out[cid] = pos
    contact_frame_out[cid] = frame
    handled = True

  # sphere-sphere
  if not handled and t1 == 2 and t2 == 2:
    dist, pos, normal = smooth_sphere_sphere(pos1, size1[0], pos2, size2[0])
    frame = smooth_make_frame(normal)
    contact_dist_out[cid] = dist
    contact_pos_out[cid] = pos
    contact_frame_out[cid] = frame
    handled = True

  # sphere-capsule
  if not handled and t1 == 2 and t2 == 3:
    cap_axis = wp.vec3(mat2[0, 2], mat2[1, 2], mat2[2, 2])
    dist, pos, normal = smooth_sphere_capsule(pos1, size1[0], pos2, cap_axis, size2[0], size2[1])
    frame = smooth_make_frame(normal)
    contact_dist_out[cid] = dist
    contact_pos_out[cid] = pos
    contact_frame_out[cid] = frame
    handled = True

  # capsule-capsule (2 contacts via geomcollisionid)
  if not handled and t1 == 3 and t2 == 3:
    cap1_axis = wp.vec3(mat1[0, 2], mat1[1, 2], mat1[2, 2])
    cap2_axis = wp.vec3(mat2[0, 2], mat2[1, 2], mat2[2, 2])
    dists, positions, normals = smooth_capsule_capsule(
      pos1,
      cap1_axis,
      size1[0],
      size1[1],
      pos2,
      cap2_axis,
      size2[0],
      size2[1],
      1e10,  # large margin so we always compute both contacts
    )
    if subcid == 0:
      contact_dist_out[cid] = dists[0]
      contact_pos_out[cid] = wp.vec3(positions[0, 0], positions[0, 1], positions[0, 2])
      normal0 = wp.vec3(normals[0, 0], normals[0, 1], normals[0, 2])
      contact_frame_out[cid] = smooth_make_frame(normal0)
    else:
      contact_dist_out[cid] = dists[1]
      contact_pos_out[cid] = wp.vec3(positions[1, 0], positions[1, 1], positions[1, 2])
      normal1 = wp.vec3(normals[1, 0], normals[1, 1], normals[1, 2])
      contact_frame_out[cid] = smooth_make_frame(normal1)
    handled = True

  # plane-capsule (2 contacts via geomcollisionid)
  if not handled and t1 == 0 and t2 == 3:
    plane_normal = wp.vec3(mat1[0, 2], mat1[1, 2], mat1[2, 2])
    cap_axis = wp.vec3(mat2[0, 2], mat2[1, 2], mat2[2, 2])
    dists, positions, frame = smooth_plane_capsule(plane_normal, pos1, pos2, cap_axis, size2[0], size2[1])
    if subcid == 0:
      contact_dist_out[cid] = dists[0]
      contact_pos_out[cid] = wp.vec3(positions[0, 0], positions[0, 1], positions[0, 2])
    else:
      contact_dist_out[cid] = dists[1]
      contact_pos_out[cid] = wp.vec3(positions[1, 0], positions[1, 1], positions[1, 2])
    contact_frame_out[cid] = frame
    handled = True

  # Unsupported types: no-op (keeps discrete values, zero gradient)


# ============================================================================
# Shared constraint parameter computation
# ============================================================================


@wp.func
def compute_k_imp(
  # Model:
  opt_disableflags: int,
  # In:
  solref: wp.vec2,
  solimp: types.vec5,
  pos: float,
  timestep: float,
) -> wp.vec2:
  """Compute stiffness k and impedance imp from solref/solimp parameters.

  Returns (k, imp) packed as a vec2. Used by both the forward constraint
  assembly and the adjoint gradient kernel.
  """
  timeconst = solref[0]
  dampratio = solref[1]
  dmin = solimp[0]
  dmax = solimp[1]
  width = solimp[2]
  mid = solimp[3]
  power = solimp[4]

  if not (opt_disableflags & DisableBit.REFSAFE):
    timeconst = wp.max(timeconst, 2.0 * timestep)

  dmin = wp.clamp(dmin, types.MJ_MINIMP, types.MJ_MAXIMP)
  dmax = wp.clamp(dmax, types.MJ_MINIMP, types.MJ_MAXIMP)
  width = wp.max(MJ_MINVAL, width)
  mid = wp.clamp(mid, types.MJ_MINIMP, types.MJ_MAXIMP)
  power = wp.max(1.0, power)

  dmax_sq = dmax * dmax
  k = 1.0 / (dmax_sq * timeconst * timeconst * dampratio * dampratio)
  k = wp.where(solref[0] <= 0.0, -solref[0] / dmax_sq, k)

  imp_x = wp.abs(pos) / width
  imp_a = (1.0 / wp.pow(mid, power - 1.0)) * wp.pow(imp_x, power)
  imp_b = 1.0 - (1.0 / wp.pow(1.0 - mid, power - 1.0)) * wp.pow(1.0 - imp_x, power)
  imp_y = wp.where(imp_x < mid, imp_a, imp_b)
  imp = dmin + imp_y * (dmax - dmin)
  imp = wp.clamp(imp, dmin, dmax)
  imp = wp.where(imp_x > 1.0, dmax, imp)

  return wp.vec2(k, imp)


# ============================================================================
# Differentiable constraint assembly kernel
# ============================================================================


@wp.func
def _smooth_efc_row(
  # Model:
  opt_disableflags: int,
  # In:
  worldid: int,
  timestep: float,
  efcid: int,
  pos_aref: float,
  pos_imp: float,
  invweight: float,
  solref: wp.vec2,
  solimp: types.vec5,
  margin: float,
  vel: float,
  # Out:
  pos_out: wp.array2d(dtype=float),
  D_out: wp.array2d(dtype=float),
  aref_out: wp.array2d(dtype=float),
  vel_out: wp.array2d(dtype=float),
):
  """Smooth reimplementation of _efc_row for differentiable constraint params."""
  k_imp = compute_k_imp(opt_disableflags, solref, solimp, pos_imp, timestep)
  k = k_imp[0]
  imp = k_imp[1]

  # Damping coefficient (not shared — only needed by forward, not adjoint)
  dmax = wp.clamp(solimp[1], types.MJ_MINIMP, types.MJ_MAXIMP)
  timeconst = solref[0]
  if not (opt_disableflags & DisableBit.REFSAFE):
    timeconst = wp.max(timeconst, 2.0 * timestep)
  b = 2.0 / (dmax * timeconst)
  b = wp.where(solref[1] <= 0.0, -solref[1] / dmax, b)

  D_out[worldid, efcid] = 1.0 / wp.max(invweight * (1.0 - imp) / imp, MJ_MINVAL)
  vel_out[worldid, efcid] = vel
  aref_out[worldid, efcid] = -k * imp * pos_aref - b * vel
  pos_out[worldid, efcid] = pos_aref + margin


@wp.kernel
def _smooth_contact_to_efc_kernel(
  # Model:
  nv: int,
  opt_timestep: wp.array(dtype=float),
  opt_disableflags: int,
  opt_impratio_invsqrt: wp.array(dtype=float),
  body_parentid: wp.array(dtype=int),
  body_rootid: wp.array(dtype=int),
  body_weldid: wp.array(dtype=int),
  body_dofnum: wp.array(dtype=int),
  body_dofadr: wp.array(dtype=int),
  body_invweight0: wp.array2d(dtype=wp.vec2),
  dof_bodyid: wp.array(dtype=int),
  dof_parentid: wp.array(dtype=int),
  body_isdofancestor: wp.array2d(dtype=int),
  geom_bodyid: wp.array(dtype=int),
  # Data in:
  qvel_in: wp.array2d(dtype=float),
  subtree_com_in: wp.array2d(dtype=wp.vec3),
  cdof_in: wp.array2d(dtype=wp.spatial_vector),
  contact_dist_in: wp.array(dtype=float),
  contact_pos_in: wp.array(dtype=wp.vec3),
  contact_frame_in: wp.array(dtype=wp.mat33),
  contact_includemargin_in: wp.array(dtype=float),
  contact_friction_in: wp.array(dtype=types.vec5),
  contact_solref_in: wp.array(dtype=wp.vec2),
  contact_solimp_in: wp.array(dtype=types.vec5),
  contact_dim_in: wp.array(dtype=int),
  contact_geom_in: wp.array(dtype=wp.vec2i),
  contact_efc_address_in: wp.array2d(dtype=int),
  contact_worldid_in: wp.array(dtype=int),
  contact_type_in: wp.array(dtype=int),
  njmax_in: int,
  nacon_in: wp.array(dtype=int),
  # Data out:
  efc_J_out: wp.array3d(dtype=float),
  efc_pos_out: wp.array2d(dtype=float),
  efc_D_out: wp.array2d(dtype=float),
  efc_vel_out: wp.array2d(dtype=float),
  efc_aref_out: wp.array2d(dtype=float),
):
  conid, dimid = wp.tid()

  if conid >= nacon_in[0]:
    return

  # Only process constraint contacts
  if not (contact_type_in[conid] & 1):  # ContactType.CONSTRAINT = 1
    return

  condim = contact_dim_in[conid]
  if condim == 1 and dimid > 0:
    return
  elif condim > 1 and dimid >= 2 * (condim - 1):
    return

  # Read efc_address — skip if -1 (not active)
  efcid = contact_efc_address_in[conid, dimid]
  if efcid < 0:
    return
  if efcid >= njmax_in:
    return

  worldid = contact_worldid_in[conid]
  timestep = opt_timestep[worldid % opt_timestep.shape[0]]
  impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

  geom = contact_geom_in[conid]
  body1 = geom_bodyid[geom[0]]
  body2 = geom_bodyid[geom[1]]

  con_pos = contact_pos_in[conid]
  frame = contact_frame_in[conid]
  includemargin = contact_includemargin_in[conid]
  pos = contact_dist_in[conid] - includemargin

  # Pyramidal invweight computation
  body_invweight0_id = worldid % body_invweight0.shape[0]
  invweight = body_invweight0[body_invweight0_id, body1][0] + body_invweight0[body_invweight0_id, body2][0]

  fri0 = float(0.0)
  frii = float(0.0)
  dimid2 = int(0)
  if condim > 1:
    dimid2 = dimid / 2 + 1
    friction = contact_friction_in[conid]
    fri0 = friction[0]
    frii = friction[dimid2 - 1]
    invweight = invweight + fri0 * fri0 * invweight
    invweight = invweight * 2.0 * fri0 * fri0 * impratio_invsqrt * impratio_invsqrt

  Jqvel = float(0.0)

  # Skip fixed bodies
  body1 = body_weldid[body1]
  body2 = body_weldid[body2]

  da1 = int(body_dofadr[body1] + body_dofnum[body1] - 1)
  da2 = int(body_dofadr[body2] + body_dofnum[body2] - 1)

  # Dense Jacobian computation (AD requires dense)
  da = wp.max(da1, da2)
  dofid = int(nv - 1)

  while True:
    if dofid < 0:
      break

    if dofid == da:
      jac1p, jac1r = support.jac_dof(
        body_parentid,
        body_rootid,
        dof_bodyid,
        body_isdofancestor,
        subtree_com_in,
        cdof_in,
        con_pos,
        body1,
        dofid,
        worldid,
      )
      jac2p, jac2r = support.jac_dof(
        body_parentid,
        body_rootid,
        dof_bodyid,
        body_isdofancestor,
        subtree_com_in,
        cdof_in,
        con_pos,
        body2,
        dofid,
        worldid,
      )

      J = float(0.0)
      Ji = float(0.0)

      for xyz in range(3):
        jacp_dif = jac2p[xyz] - jac1p[xyz]
        J += frame[0, xyz] * jacp_dif

        if condim > 1:
          if dimid2 < 3:
            Ji += frame[dimid2, xyz] * jacp_dif
          else:
            Ji += frame[dimid2 - 3, xyz] * (jac2r[xyz] - jac1r[xyz])

      if condim > 1:
        if dimid % 2 == 0:
          J += Ji * frii
        else:
          J -= Ji * frii

      efc_J_out[worldid, efcid, dofid] = J
      Jqvel += J * qvel_in[worldid, dofid]

      # Advance tree pointers
      if da1 == da:
        da1 = dof_parentid[da1]
      if da2 == da:
        da2 = dof_parentid[da2]
      da = wp.max(da1, da2)
      dofid -= 1
    else:
      efc_J_out[worldid, efcid, dofid] = 0.0
      dofid -= 1

  # Compute constraint equation row
  _smooth_efc_row(
    opt_disableflags,
    worldid,
    timestep,
    efcid,
    pos,
    pos,
    invweight,
    contact_solref_in[conid],
    contact_solimp_in[conid],
    includemargin,
    Jqvel,
    efc_pos_out,
    efc_D_out,
    efc_aref_out,
    efc_vel_out,
  )


# ============================================================================
# Python launchers
# ============================================================================


def smooth_recompute_contacts(m: types.Model, d: types.Data):
  """Overwrite contact.{dist, pos, frame} with smooth differentiable values."""
  if d.naconmax == 0:
    return

  wp.launch(
    _smooth_recompute_kernel,
    dim=d.naconmax,
    inputs=[
      # Model
      m.geom_type,
      m.geom_bodyid,
      m.geom_size,
      # Data in
      d.geom_xpos,
      d.geom_xmat,
      d.contact.geom,
      d.contact.worldid,
      d.contact.geomcollisionid,
      d.nacon,
    ],
    outputs=[
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
    ],
  )


def smooth_contact_to_efc(m: types.Model, d: types.Data):
  """Overwrite efc.{J, pos, D, aref, vel} with smooth differentiable values."""
  if d.naconmax == 0 or d.njmax == 0:
    return

  wp.launch(
    _smooth_contact_to_efc_kernel,
    dim=(d.naconmax, m.nmaxpyramid),
    inputs=[
      # Model
      m.nv,
      m.opt.timestep,
      m.opt.disableflags,
      m.opt.impratio_invsqrt,
      m.body_parentid,
      m.body_rootid,
      m.body_weldid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_invweight0,
      m.dof_bodyid,
      m.dof_parentid,
      m.body_isdofancestor,
      m.geom_bodyid,
      # Data in
      d.qvel,
      d.subtree_com,
      d.cdof,
      d.contact.dist,
      d.contact.pos,
      d.contact.frame,
      d.contact.includemargin,
      d.contact.friction,
      d.contact.solref,
      d.contact.solimp,
      d.contact.dim,
      d.contact.geom,
      d.contact.efc_address,
      d.contact.worldid,
      d.contact.type,
      d.njmax,
      d.nacon,
    ],
    outputs=[
      d.efc.J,
      d.efc.pos,
      d.efc.D,
      d.efc.vel,
      d.efc.aref,
    ],
  )
