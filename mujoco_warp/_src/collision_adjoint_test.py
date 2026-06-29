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
"""FD gate for the contact-``∂qpos`` analytic VJPs in ``collision_adjoint.py``.

Reuses the MuJoCo-C-validated geom-pair scenes from ``collision_driver_test.CollisionTest._FIXTURES``
(the same XML that gates the FORWARD narrowphase), restricted to the AD-safe primitive pairs that
``collision_adjoint._narrowphase_recompute`` dispatches (``_PAIR``). Two oracles per fixture:

  1. ``test_narrowphase_qpos_vjp`` -- SURGICAL, single-step. Perturb qpos in TANGENT space
     (``mj_integratePos``, so free-joint quaternions are handled), re-run ``mjw.forward``, central-diff
     the contact geometry (``dist`` and ``pos``) the narrowphase produces, and compare to ``contact_qpos_vjp``
     driven with unit/normal seeds. This isolates ``collision_adjoint.py`` from the solver -- the sharpest
     per-branch signal, and the only one that exercises a pair's ORIENTATION ∂qpos directly. (We read the
     per-dof directional derivative straight out of ``res_qpos`` at ``jnt_qposadr + (k - jnt_dofadr)`` --
     exactly the index ``_geom_pose_qpos_vjp`` wrote it to -- so this is a tangent-space comparison.)
  2. ``test_step_qvel_grad`` -- END-TO-END, multi-step BPTT. ``d(‖qvel_T‖²)/d(qvel0)`` analytic (taped
     ``adjoint.py`` backward) vs central FD of ``mjw.step``, so any per-step ∂qpos bias ACCUMULATES (the
     regime a single-step check misses). Runs on the plane-supported subset that rolls out stably.

Differentiable regime applied to every fixture (``_diffify``): EULER (``adjoint._advance_state`` is
semi-implicit Euler), elliptic cone, soft solimp (``dmin=0`` -> unsaturated/smooth contact), eulerdamp off,
timestep 0.004. FD eps is per-regime (geometry is smooth -> tight; the stiff multi-step rollout is looser).

  uv run --with pytest python -m pytest mujoco_warp/_src/collision_adjoint_test.py -q
"""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp._src import adjoint  # noqa: F401  registers the analytic step backward
from mujoco_warp._src import collision_adjoint
from mujoco_warp._src import collision_driver_test as _cdt  # import as module so pytest doesn't re-collect its TestCase
from mujoco_warp._src import support as _support
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import JointType

_EULER = int(mujoco.mjtIntegrator.mjINT_EULER)
_ELLIPTIC = int(mujoco.mjtCone.mjCONE_ELLIPTIC)
_FIXTURES = _cdt.CollisionTest._FIXTURES  # reuse the forward-narrowphase scenes

_PLANE = GeomType.PLANE
_SPHERE = GeomType.SPHERE
_CAPSULE = GeomType.CAPSULE
_ELLIPSOID = GeomType.ELLIPSOID
_CYLINDER = GeomType.CYLINDER
_BOX = GeomType.BOX

# AD-safe primitive pairs collision_adjoint._narrowphase_recompute dispatches, mapped to the driver
# fixtures that exercise them. (geom0, geom1) is ascending geom type = forward narrowphase order.
# Excluded driver fixtures: capsule_capsule_parallel_axes (parallel slot-1 = margin-dependent follow-up),
# capsule_box_* / box_box_* / *mesh* / convex_* / sdf / hfield (NOT AD-safe -- data-dependent feature search).
_PAIR = {
  "plane_sphere": (_PLANE, _SPHERE),
  "plane_ellipsoid": (_PLANE, _ELLIPSOID),
  "plane_capsule": (_PLANE, _CAPSULE),
  "box_plane": (_PLANE, _BOX),  # plane-box
  "plane_cylinder_1": (_PLANE, _CYLINDER),
  "plane_cylinder_2": (_PLANE, _CYLINDER),
  "plane_cylinder_3": (_PLANE, _CYLINDER),
  "sphere_sphere": (_SPHERE, _SPHERE),
  "sphere_capsule": (_SPHERE, _CAPSULE),
  "sphere_cylinder_cap": (_SPHERE, _CYLINDER),
  "sphere_cylinder_side": (_SPHERE, _CYLINDER),
  "sphere_cylinder_corner": (_SPHERE, _CYLINDER),
  "sphere_box_shallow": (_SPHERE, _BOX),
  "sphere_box_deep": (_SPHERE, _BOX),
  "capsule_capsule": (_CAPSULE, _CAPSULE),  # crossed (non-parallel) -> slot 0
}

# End-to-end accumulation gate: plane-supported scenes that settle to a stable contact and SLIDE/ROLL under
# a lateral kick, so ‖qvel_T‖² stays sensitive to qvel0. key -> (settle_steps, rollout T). The ROLLING
# rotation-dependent scenes (plane_ellipsoid, plane_cylinder_1/3) gate the free-joint QUATERNION LIFT in
# _dof_to_qpos (collision_adjoint.py) -- before it they carried a stable ~15-50% rotational-composition
# bias; with it they match FD-of-mjw.step (cos>0.99).
_MULTISTEP = {
  "plane_sphere": (8, 30),
  "plane_ellipsoid": (8, 30),  # rolls -> gates the quaternion lift
  "plane_capsule": (8, 30),
  "box_plane": (10, 30),
  "plane_cylinder_1": (8, 30),  # rolls -> gates the quaternion lift
  "plane_cylinder_2": (8, 30),
  "plane_cylinder_3": (8, 30),  # rolls -> gates the quaternion lift
  "sphere_box_shallow": (8, 30),
}

# Articulated scenes (not in the driver fixtures -- those are single-geom-pair) that gate the free/ball
# quaternion lift beyond a single free body: a BALL joint's OWN quaternion, and an ANCESTOR free joint's
# quaternion (jac_dof already chains the tangent gradient to ancestor dofs; _dof_to_qpos lifts each
# free/ball joint locally). key -> (xml, settle, T, {dof: kick}). Ground-supported -> well-conditioned.
_BALL_GROUND = """
<mujoco>
  <option timestep="0.004" cone="elliptic" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <worldbody>
    <geom type="plane" size="5 5 0.1" condim="3" friction="0.8 0.01 0.01"/>
    <body pos="0 0 0.25"><joint type="ball"/>
      <geom type="ellipsoid" size="0.2 0.12 0.12" mass="1" condim="3" friction="0.8 0.01 0.01" solimp="0 0.95 0.001"/></body>
  </worldbody>
</mujoco>
"""
_FREE_BALL_WORM = """
<mujoco>
  <option timestep="0.004" cone="elliptic" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <default><geom condim="3" friction="0.8 0.01 0.01" solimp="0 0.95 0.001"/></default>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.1" euler="0 0 5"><freejoint/>
      <geom type="capsule" fromto="-0.22 0 0 -0.02 0 0" size="0.1" mass="1"/>
      <body pos="0 0 0"><joint type="ball"/>
        <geom type="capsule" fromto="0.02 0 0 0.22 0 0" size="0.1" mass="1"/></body></body>
  </worldbody>
</mujoco>
"""
_ARTIC = {
  "ball_ground_ellipsoid": (_BALL_GROUND, 8, 30, {0: 0.5, 1: 0.3}),  # BALL own-quaternion (nv=3)
  "free_ball_worm": (_FREE_BALL_WORM, 12, 30, {0: 0.4, 1: 0.25, 5: 0.3}),  # ANCESTOR free + own ball (nv=9)
}


def _diffify(mjm):
  """Apply the differentiable-contact regime the adjoint validates against (in place)."""
  mjm.opt.integrator = _EULER
  mjm.opt.cone = _ELLIPTIC
  mjm.opt.timestep = 0.004
  mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
  mjm.geom_solimp[:, 0] = 0.0  # dmin=0 -> soft/unsaturated contact (smooth regime)
  return mjm


def _load(xml, settle=0):
  mjm = _diffify(mujoco.MjModel.from_xml_string(xml))
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  for _ in range(settle):
    mujoco.mj_step(mjm, mjd)
  mujoco.mj_forward(mjm, mjd)
  return mjm, mjd


def _find_contact(d, type0, type1, geom_type, slot=None):
  """Index of an active contact whose (geom0,geom1) types match (type0,type1); optionally a given slot."""
  nacon = int(d.nacon.numpy()[0])
  geom = d.contact.geom.numpy()
  gcid = d.contact.geomcollisionid.numpy()
  for c in range(nacon):
    g0, g1 = geom[c]
    if g0 < 0 or g1 < 0:
      continue
    if int(geom_type[g0]) == int(type0) and int(geom_type[g1]) == int(type1):
      if slot is None or int(gcid[c]) == slot:
        return c
  return -1


# --- Oracle 1 helpers: surgical single-step narrowphase ∂qpos VJP (tangent space) ---

_FREE = int(JointType.FREE.value)
_BALL = int(JointType.BALL.value)


def _qmul(a, b):  # Hamilton product, w-first (matches mjw math.mul_quat)
  return np.array([
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ])


def _unlift_quat(q, dq):
  """Invert the kernel's lift dq = 2 q⊗[0,g]: g = 1/2 vec(conj(q)⊗dq). Recovers the per-dof angular
  TANGENT gradient from the raw-quaternion gradient so the surgical oracle keeps its clean tangent FD
  comparison (the lift FORMULA itself is gated end-to-end by the multi-step FD-of-mjw.step oracle)."""
  conj = np.array([q[0], -q[1], -q[2], -q[3]])
  return 0.5 * _qmul(conj, dq)[1:]


def _analytic_geom_vjp(mjm, m, d, cid, w, e0, seed_kind, u, qpos0):
  """∂(contact geometry)/∂qpos via contact_qpos_vjp seeded on ONE contact's geometry intermediate.

  seed_kind 'dist': seed res_efc_pos[w,e0]=1 (∂efc_pos/∂dist=1) -> res_qpos = ∂dist/∂qpos.
  seed_kind 'pos':  seed res_contact_pos[cid]=u             -> res_qpos = ∂(pos·u)/∂qpos.
  Returned per-dof in TANGENT space: 1:1 for slide/hinge/free-translation; for free/ball the quaternion
  block is un-lifted back to the angular tangent gradient so it matches the mj_integratePos FD.
  """
  nworld, nq, nv = d.qpos.shape[0], m.nq, m.nv
  res_qpos = wp.zeros((nworld, nq), dtype=float)
  res_subtree_com = wp.zeros((nworld, m.nbody), dtype=wp.vec3)
  res_cdof = wp.zeros((nworld, nv), dtype=wp.spatial_vector)
  res_efc_pos = wp.zeros_like(d.efc.pos)
  res_contact_pos = wp.zeros_like(d.contact.pos)
  res_contact_frame = wp.zeros_like(d.contact.frame)
  if seed_kind == "dist":
    a = np.zeros(d.efc.pos.shape, dtype=np.float32)
    a[w, e0] = 1.0
    res_efc_pos = wp.array(a, dtype=float)
  else:
    a = np.zeros((d.contact.pos.shape[0], 3), dtype=np.float32)
    a[cid] = u
    res_contact_pos = wp.array(a, dtype=wp.vec3)
  collision_adjoint.contact_qpos_vjp(
    m, d, res_contact_pos, res_contact_frame, res_efc_pos, res_subtree_com, res_cdof, res_qpos
  )
  rq = res_qpos.numpy()[w]
  g = np.zeros(nv)
  for j in range(mjm.njnt):
    jt, qadr, dadr = int(mjm.jnt_type[j]), mjm.jnt_qposadr[j], mjm.jnt_dofadr[j]
    if jt == _FREE:
      g[dadr : dadr + 3] = rq[qadr : qadr + 3]  # translation 1:1
      g[dadr + 3 : dadr + 6] = _unlift_quat(qpos0[qadr + 3 : qadr + 7], rq[qadr + 3 : qadr + 7])
    elif jt == _BALL:
      g[dadr : dadr + 3] = _unlift_quat(qpos0[qadr : qadr + 4], rq[qadr : qadr + 4])
    else:  # HINGE / SLIDE
      g[dadr] = rq[qadr]
  return g


def _fd_geom_vjp(mjm, mjd, m, type0, type1, geom_type, slot, seed_kind, u, eps):
  """Tangent-space central FD of the matched contact's dist / (pos·u) wrt each dof (mj_integratePos)."""
  nv = mjm.nv

  def value(qp):
    d = mjw.put_data(mjm, mjd)
    d.qpos = wp.array(qp.reshape(1, -1).astype(np.float32), dtype=float)
    mjw.forward(m, d)
    c = _find_contact(d, type0, type1, geom_type, slot)
    if c < 0:
      return None
    if seed_kind == "dist":
      return float(d.contact.dist.numpy()[c])
    return float(np.dot(d.contact.pos.numpy()[c], u))

  g = np.zeros(nv)
  for k in range(nv):
    e = np.zeros(nv)
    e[k] = 1.0
    qp = mjd.qpos.copy()
    qm = mjd.qpos.copy()
    mujoco.mj_integratePos(mjm, qp, e, eps)
    mujoco.mj_integratePos(mjm, qm, e, -eps)
    vp, vm = value(qp), value(qm)
    g[k] = np.nan if (vp is None or vm is None) else (vp - vm) / (2.0 * eps)
  return g


# --- Oracle 2 helpers: end-to-end multi-step BPTT d(‖qvel_T‖²)/d(qvel0) vs central FD of mjw.step ---


@wp.kernel
def _sumsq_qvel(qvel: wp.array2d[float], loss: wp.array[float]):
  i = wp.tid()
  wp.atomic_add(loss, 0, qvel[0, i] * qvel[0, i])


def _analytic_grad(mjm, mjd, T, qvel0):
  m = mjw.put_model(mjm)
  datas = [mjw.put_data(mjm, mjd) for _ in range(T + 1)]
  for dd in datas:
    dd.qpos.requires_grad = True
    dd.qvel.requires_grad = True
  datas[0].qvel = wp.array(qvel0.reshape(1, -1).astype(np.float32), dtype=float, requires_grad=True)
  loss = wp.zeros(1, dtype=float, requires_grad=True)
  tape = wp.Tape()
  with tape:
    for t in range(T):
      mjw.step(m, datas[t], datas[t + 1])
    wp.launch(_sumsq_qvel, dim=mjm.nv, inputs=[datas[T].qvel, loss])
  tape.backward(loss=loss)
  return np.nan_to_num(datas[0].qvel.grad.numpy()[0].astype(np.float64))


def _fd_grad(mjm, mjd, T, qvel0, eps=1e-4):
  def L(qv):
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)
    d.qvel = wp.array(qv.reshape(1, -1).astype(np.float32), dtype=float)
    for _ in range(T):
      mjw.step(m, d)
    qv_T = d.qvel.numpy()[0]
    return float(np.dot(qv_T, qv_T))

  g = np.zeros(mjm.nv)
  for i in range(mjm.nv):
    vp, vm = qvel0.copy(), qvel0.copy()
    vp[i] += eps
    vm[i] -= eps
    g[i] = (L(vp) - L(vm)) / (2.0 * eps)
  return g


class ContactQposVJPTest(parameterized.TestCase):
  """FD gates for the contact-∂qpos analytic VJPs (collision_adjoint.py)."""

  @parameterized.parameters(*_PAIR.keys())
  def test_narrowphase_qpos_vjp(self, name):
    """Surgical: analytic ∂(contact dist, pos)/∂qpos (contact_qpos_vjp) vs tangent-space FD of mjw.forward."""
    type0, type1 = _PAIR[name]
    mjm, mjd = _load(_FIXTURES[name])
    m, d = mjw.put_model(mjm), mjw.put_data(mjm, mjd)
    mjw.forward(m, d)
    geom_type = mjm.geom_type

    cid = _find_contact(d, type0, type1, geom_type)
    self.assertGreaterEqual(cid, 0, f"{name}: no active {type0}-{type1} contact (scene/dispatch mismatch)")
    w = int(d.contact.worldid.numpy()[cid])
    e0 = int(d.contact.efc_address.numpy()[cid, 0])
    slot = int(d.contact.geomcollisionid.numpy()[cid])
    normal = d.contact.frame.numpy()[cid][0]  # contact normal = frame row 0

    qpos0 = mjd.qpos.copy()
    for seed_kind, u, eps in (("dist", None, 1e-6), ("pos", normal, 1e-6)):
      ana = _analytic_geom_vjp(mjm, m, d, cid, w, e0, seed_kind, u, qpos0)
      fd = np.nan_to_num(_fd_geom_vjp(mjm, mjd, m, type0, type1, geom_type, slot, seed_kind, u, eps))
      na, nf = np.linalg.norm(ana), np.linalg.norm(fd)
      cos = float(ana @ fd / (na * nf)) if na > 1e-12 and nf > 1e-12 else float("nan")
      rel = float(np.linalg.norm(ana - fd) / (nf + 1e-12))
      print(
        f"\n[{name}/{seed_kind}] nv={mjm.nv} slot={slot} cos={cos:+.5f} rel={rel:.4f}"
        f"\n  ana={np.array2string(ana, precision=4, max_line_width=140)}"
        f"\n  fd ={np.array2string(fd, precision=4, max_line_width=140)}"
      )
      if nf < 1e-9:
        continue  # this geometry quantity is qpos-insensitive at this config (e.g. pos·n flat) -> nothing to gate
      self.assertGreater(cos, 0.99, f"{name}/{seed_kind}: ∂qpos DIRECTION off, cos={cos:.4f}")
      self.assertLess(rel, 0.1, f"{name}/{seed_kind}: ∂qpos MAGNITUDE off, rel={rel:.4f}")

  @parameterized.parameters(*_MULTISTEP.keys())
  def test_step_qvel_grad(self, name):
    """End-to-end: analytic d(‖qvel_T‖²)/d(qvel0) (multi-step BPTT) vs central FD of mjw.step."""
    settle, T = _MULTISTEP[name]
    mjm, mjd = _load(_FIXTURES[name], settle)
    qvel0 = mjd.qvel.astype(np.float64).copy()
    qvel0[0] += 0.4  # lateral kick so the body slides -> ‖qvel_T‖² stays sensitive to qvel0
    qvel0[1] += 0.2

    analytic = _analytic_grad(mjm, mjd, T, qvel0)
    fd = _fd_grad(mjm, mjd, T, qvel0)
    na, nf = np.linalg.norm(analytic), np.linalg.norm(fd)
    cos = float(analytic @ fd / (na * nf)) if na > 0 and nf > 0 else float("nan")
    rel = float(np.linalg.norm(analytic - fd) / (nf + 1e-12))
    print(
      f"\n[{name}] T={T} nv={mjm.nv}  cos={cos:+.5f} rel={rel:.4f}"
      f"\n  analytic={np.array2string(analytic, precision=4, max_line_width=140)}"
      f"\n  fd      ={np.array2string(fd, precision=4, max_line_width=140)}"
    )
    self.assertGreater(nf, 1e-6, f"{name}: FD gradient ~0 (loss insensitive to qvel0); scene not exercising contact")
    self.assertGreater(cos, 0.99, f"{name}: gradient DIRECTION off, cos={cos:.4f}")
    self.assertLess(rel, 0.15, f"{name}: gradient MAGNITUDE off, rel={rel:.4f}")

  @parameterized.parameters(*_ARTIC.keys())
  def test_articulated_quat_lift(self, name):
    """Articulated free/ball quaternion ∂qpos lift (_dof_to_qpos): multi-step BPTT d(‖qvel_T‖²)/d(qvel0)
    vs central FD of mjw.step. Gates a BALL joint's own quaternion and an ANCESTOR free-joint quaternion
    (the contacting child's ∂qpos flows through the base free quat + the ball quat)."""
    xml, settle, T, kick = _ARTIC[name]
    mjm, mjd = _load(xml, settle)
    qvel0 = mjd.qvel.astype(np.float64).copy()
    for dof, dv in kick.items():
      qvel0[dof] += dv

    analytic = _analytic_grad(mjm, mjd, T, qvel0)
    fd = _fd_grad(mjm, mjd, T, qvel0)
    na, nf = np.linalg.norm(analytic), np.linalg.norm(fd)
    cos = float(analytic @ fd / (na * nf)) if na > 0 and nf > 0 else float("nan")
    rel = float(np.linalg.norm(analytic - fd) / (nf + 1e-12))
    ang_idx = []  # free/ball ANGULAR dof indices = the ones the quaternion lift touches
    for j in range(mjm.njnt):
      jt, dadr = int(mjm.jnt_type[j]), mjm.jnt_dofadr[j]
      if jt == _FREE:
        ang_idx += [dadr + 3, dadr + 4, dadr + 5]
      elif jt == _BALL:
        ang_idx += [dadr, dadr + 1, dadr + 2]
    angfrac = float(np.linalg.norm(analytic[ang_idx]) / (na + 1e-12))  # rotation must be materially exercised
    print(f"\n[{name}] T={T} nv={mjm.nv}  cos={cos:+.5f} rel={rel:.4f} ang_frac={angfrac:.3f}")
    self.assertGreater(nf, 1e-6, f"{name}: FD gradient ~0; scene not exercising contact")
    self.assertGreater(angfrac, 0.02, f"{name}: rotation barely exercised (ang_frac={angfrac:.3f}); weak lift gate")
    self.assertGreater(cos, 0.99, f"{name}: gradient DIRECTION off, cos={cos:.4f}")
    self.assertLess(rel, 0.15, f"{name}: gradient MAGNITUDE off, rel={rel:.4f}")


# --- Articulated contact-Jacobian ∂qpos: cdof screw commutator + subtree-COM Jacobian (MJPLAN_ADRNE §7) ---
# Surgical gate (§14.2), the decisive isolated test of the cdof/subtree VJP: build a random J-functional
# L = Σ_i w_i·J_p,i (J_p,i = jac_dof column i for the FIXED world contact point on the contacting body), get
# the residual-style seeds res_cdof = ∂L/∂cdof, res_subtree_com = ∂L/∂subtree_com via warp AUTODIFF of
# jac_dof (correct by construction), drive contact_qpos_vjp with the narrowphase seeds ZEROED, and require
# the resulting res_qpos = ∂L/∂q to match the float64 MuJoCo mj_jac central difference. Covers an articulated
# HINGE chain (foot loaded on the floor) and a FREE-base + internal-BALL worm (the full quaternion blocks).

# 3-hinge "hopper" leg, sphere foot LOADED on the floor (joint stiffness/damping hold a stable bent pose).
_HOPPER_HINGE = """
<mujoco>
  <option timestep="0.004" cone="elliptic" gravity="0 0 -9.81"><flag eulerdamp="disable"/></option>
  <default><geom condim="3" friction="1.0 0.05 0.05" solimp="0 0.95 0.001"/><joint damping="3" stiffness="40"/></default>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body pos="0 0 0.71"><joint type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.08 0.06 0.1" mass="3.0"/>
      <body pos="0 0 -0.1"><joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.035" mass="0.6"/>
        <body pos="0 0 -0.3"><joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03" mass="0.4"/>
          <geom type="sphere" size="0.05" pos="0 0 -0.27" mass="0.2"/></body></body></body>
  </worldbody>
</mujoco>
"""
_CDOF_SCENES = {"hopper_hinge": (_HOPPER_HINGE, 40), "free_ball_worm": (_FREE_BALL_WORM, 12)}

_LOSS_KERNELS = {}


def _make_loss_kernel(nv):
  if nv not in _LOSS_KERNELS:
    @wp.kernel(enable_backward=True)
    def _jp_loss(
      body_parentid: wp.array[int], body_rootid: wp.array[int], dof_bodyid: wp.array[int],
      body_isdofancestor: wp.array2d[int], subtree_com_in: wp.array2d[wp.vec3],
      cdof_in: wp.array2d[wp.spatial_vector], point: wp.array[wp.vec3], bodyid: wp.array[int],
      w_in: wp.array2d[wp.vec3], loss: wp.array[float],
    ):
      for i in range(nv):
        jp, _jr = _support.jac_dof(body_parentid, body_rootid, dof_bodyid, body_isdofancestor,
                                   subtree_com_in, cdof_in, point[0], bodyid[0], i, 0)
        wp.atomic_add(loss, 0, wp.dot(jp, w_in[0, i]))
    _LOSS_KERNELS[nv] = _jp_loss
  return _LOSS_KERNELS[nv]


class ArticContactJacQposVJPTest(parameterized.TestCase):
  """Surgical FD gate for the articulated contact-Jacobian ∂qpos VJP (cdof + subtree_com), MJPLAN_ADRNE §7."""

  @parameterized.parameters(*_CDOF_SCENES.keys())
  def test_contact_jacobian_qpos_vjp(self, name):
    xml, settle = _CDOF_SCENES[name]
    mjm, mjd = _load(xml, settle)
    nv = mjm.nv
    m, d = mjw.put_model(mjm), mjw.put_data(mjm, mjd)
    mjw.forward(m, d)
    self.assertGreater(int(d.nacon.numpy()[0]), 0, f"{name}: no contact after settle")

    g0, g1 = d.contact.geom.numpy()[0]
    body = int(max(mjm.geom_bodyid[g0], mjm.geom_bodyid[g1]))
    point = d.contact.pos.numpy()[0].astype(np.float64).copy()
    par = mjm.body_parentid
    chain, b = set(), body
    while b > 0:
      chain.add(b); b = par[b]
    rng = np.random.default_rng(0)
    w = np.zeros((nv, 3), dtype=np.float64)
    for i in range(nv):
      if mjm.dof_bodyid[i] in chain:
        w[i] = rng.standard_normal(3)

    # seeds res_cdof = ∂L/∂cdof, res_subtree_com = ∂L/∂subtree_com via warp autodiff of jac_dof
    d.cdof.requires_grad = True
    d.subtree_com.requires_grad = True
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      wp.launch(_make_loss_kernel(nv), dim=1,
                inputs=[m.body_parentid, m.body_rootid, m.dof_bodyid, m.body_isdofancestor,
                        d.subtree_com, d.cdof, wp.array(point.reshape(1, 3).astype(np.float32), dtype=wp.vec3),
                        wp.array(np.array([body], np.int32), dtype=int),
                        wp.array(w.reshape(1, nv, 3).astype(np.float32), dtype=wp.vec3)],
                outputs=[loss])
    tape.backward(loss=loss)
    res_cdof, res_subtree_com = wp.clone(d.cdof.grad), wp.clone(d.subtree_com.grad)

    # wired VJP with narrowphase seeds zeroed -> res_qpos = ∂L/∂q (cdof + subtree_com only)
    res_qpos = wp.zeros((1, mjm.nq), dtype=float)
    collision_adjoint.contact_qpos_vjp(
      m, d, wp.zeros_like(d.contact.pos), wp.zeros_like(d.contact.frame), wp.zeros_like(d.efc.pos),
      res_subtree_com, res_cdof, res_qpos)
    rq = res_qpos.numpy()[0]
    ana = np.zeros(nv)  # lift res_qpos -> per-dof tangent (un-lift free/ball quaternion blocks)
    for j in range(mjm.njnt):
      jt, qa, da = int(mjm.jnt_type[j]), mjm.jnt_qposadr[j], mjm.jnt_dofadr[j]
      if jt == _FREE:
        ana[da:da + 3] = rq[qa:qa + 3]
        ana[da + 3:da + 6] = _unlift_quat(mjd.qpos[qa + 3:qa + 7], rq[qa + 3:qa + 7])
      elif jt == _BALL:
        ana[da:da + 3] = _unlift_quat(mjd.qpos[qa:qa + 4], rq[qa:qa + 4])
      else:
        ana[da] = rq[qa]

    # FD oracle: L(q) = Σ_i w_i · mj_jac column i at the FIXED world point (float64)
    def Lq(qp):
      dd = mujoco.MjData(mjm); dd.qpos[:] = qp
      mujoco.mj_kinematics(mjm, dd); mujoco.mj_comPos(mjm, dd)
      J = np.zeros((3, nv)); mujoco.mj_jac(mjm, dd, J, None, point, body)
      return float(sum(w[i] @ J[:, i] for i in range(nv)))
    fd = np.zeros(nv)
    for k in range(nv):
      e = np.zeros(nv); e[k] = 1.0
      qp, qm = mjd.qpos.copy(), mjd.qpos.copy()
      mujoco.mj_integratePos(mjm, qp, e, 1e-7); mujoco.mj_integratePos(mjm, qm, e, -1e-7)
      fd[k] = (Lq(qp) - Lq(qm)) / 2e-7

    na, nf = np.linalg.norm(ana), np.linalg.norm(fd)
    cos = float(ana @ fd / (na * nf)) if na > 1e-12 and nf > 1e-12 else float("nan")
    rel = float(np.linalg.norm(ana - fd) / (nf + 1e-12))
    print(f"\n[{name}] nv={nv} cos={cos:+.6f} rel={rel:.2e}"
          f"\n  ana={np.array2string(ana, precision=4, max_line_width=140)}"
          f"\n  fd ={np.array2string(fd, precision=4, max_line_width=140)}")
    self.assertGreater(nf, 1e-6, f"{name}: FD ~0 (scene not exercising the contact Jacobian)")
    self.assertGreater(cos, 0.99, f"{name}: contact-Jacobian ∂qpos DIRECTION off, cos={cos:.4f}")
    self.assertLess(rel, 0.02, f"{name}: contact-Jacobian ∂qpos MAGNITUDE off, rel={rel:.4f}")


if __name__ == "__main__":
  absltest.main()
