"""Physically-consistent rigid-body inertia parameterization (the ../mjlab `dr.pseudo_inertia` tooling,
restricted to the CENTRAL rotational inertia with mass + COM held fixed).

A rotational inertia tensor `I` (3x3, at the COM) is physically realizable by SOME rigid mass distribution
iff its *mass covariance* `Sigma = 1/2*tr(I)*I3 - I` is positive-definite (Wensing et al.; equivalently the
principal moments satisfy the triangle inequality `I_i + I_j >= I_k`). So we parameterize `Sigma = L L^T` via
a log-Cholesky factor (unconstrained `theta in R^6`, diagonal exponentiated -> SPD for ANY theta), recover
`I = tr(Sigma)*I3 - Sigma`, then diagonalize `I` to MuJoCo's storage: principal moments `body_inertia` +
principal-frame quaternion `body_iquat`. This is exactly mjlab's forward/inverse pseudo-inertia transform
(`_reconstruct_pseudo_inertia_J` / `_decompose_pseudo_inertia_J`, Rucker & Wensing RA-L 2022,
https://par.nsf.gov/servlets/purl/10347458) on the 3x3 rotational block -- mass and COM are frozen at GT.

Quaternions here are wxyz (MuJoCo/mjlab convention). Warp's `wp.quat` is xyzw -> convert at the boundary
with `wxyz_to_xyzw` when writing `m.body_iquat`.
"""

import numpy as np


def mat_from_quat_wxyz(q):
  """3x3 rotation from a wxyz quaternion (assumed ~unit)."""
  w, x, y, z = q
  return np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def quat_wxyz_from_mat(R):
  """wxyz quaternion from a 3x3 proper rotation (Shepperd's best-conditioned branch)."""
  t = np.trace(R)
  if t > 0.0:
    s = np.sqrt(t + 1.0) * 2.0
    w = 0.25 * s
    x = (R[2, 1] - R[1, 2]) / s
    y = (R[0, 2] - R[2, 0]) / s
    z = (R[1, 0] - R[0, 1]) / s
  elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
    s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
    w = (R[2, 1] - R[1, 2]) / s
    x = 0.25 * s
    y = (R[0, 1] + R[1, 0]) / s
    z = (R[0, 2] + R[2, 0]) / s
  elif R[1, 1] > R[2, 2]:
    s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
    w = (R[0, 2] - R[2, 0]) / s
    x = (R[0, 1] + R[1, 0]) / s
    y = 0.25 * s
    z = (R[1, 2] + R[2, 1]) / s
  else:
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    w = (R[1, 0] - R[0, 1]) / s
    x = (R[0, 2] + R[2, 0]) / s
    y = (R[1, 2] + R[2, 1]) / s
    z = 0.25 * s
  q = np.array([w, x, y, z])
  return q / np.linalg.norm(q)


def wxyz_to_xyzw(q):
  """wxyz (MuJoCo) -> xyzw (warp wp.quat)."""
  return np.array([q[1], q[2], q[3], q[0]])


def reconstruct_Icom(inertia_diag, iquat_wxyz):
  """MuJoCo (principal moments + principal-frame quat) -> the full 3x3 central inertia tensor `R diag R^T`."""
  R = mat_from_quat_wxyz(iquat_wxyz)
  return R @ np.diag(inertia_diag) @ R.T


def sigma_from_I(I):
  """Mass covariance from the inertia tensor: Sigma = 1/2*tr(I)*I3 - I (SPD iff I is physically realizable)."""
  return 0.5 * np.trace(I) * np.eye(3) - I


def I_from_sigma(S):
  """Inverse: inertia tensor from the mass covariance: I = tr(Sigma)*I3 - Sigma."""
  return np.trace(S) * np.eye(3) - S


def logchol_from_sigma(S):
  """Inverse log-Cholesky: theta6 = [log L00, log L11, log L22, L10, L20, L21] from an SPD Sigma = L L^T."""
  L = np.linalg.cholesky(S)  # lower-triangular
  return np.array([np.log(L[0, 0]), np.log(L[1, 1]), np.log(L[2, 2]), L[1, 0], L[2, 0], L[2, 1]])


def sigma_from_logchol(theta):
  """Forward log-Cholesky: SPD Sigma = L L^T from unconstrained theta6 (diagonal exponentiated)."""
  L = np.array([
    [np.exp(theta[0]), 0.0, 0.0],
    [theta[3], np.exp(theta[1]), 0.0],
    [theta[4], theta[5], np.exp(theta[2])],
  ])
  return L @ L.T


def decompose_I_to_mjc(I, ref_V=None):
  """Diagonalize the central inertia tensor -> (principal moments `body_inertia`, wxyz `body_iquat`).

  Eigenvectors have a per-column sign ambiguity (all sign choices give the SAME tensor, hence the same
  physics). To make the map a CONTINUOUS function of `I` (needed for the finite-difference parameterization
  Jacobian), align each eigenvector column's sign to `ref_V` when given; otherwise fix det(V)=+1 so V is a
  proper rotation. Returns (principal[3], iquat_wxyz[4], V) -- V is the aligned eigenvector matrix (pass it
  back as `ref_V` for nearby `I`)."""
  evals, V = np.linalg.eigh(I)  # ascending eigenvalues; V columns orthonormal
  if ref_V is not None:
    for i in range(3):
      if np.dot(V[:, i], ref_V[:, i]) < 0.0:
        V[:, i] = -V[:, i]
  if np.linalg.det(V) < 0.0:  # ensure a proper rotation (flip the least-aligned / last column)
    V[:, 2] = -V[:, 2]
  return evals, quat_wxyz_from_mat(V), V


def params_to_mjc(theta, ref_V=None):
  """theta6 -> physically-consistent (body_inertia[3], iquat_wxyz[4], V). The full pipeline:
  Sigma = L(theta) L(theta)^T -> I = tr(Sigma)I3 - Sigma -> eigendecompose."""
  S = sigma_from_logchol(theta)
  I = I_from_sigma(S)
  return decompose_I_to_mjc(I, ref_V=ref_V)


# --- diagonal (principal-moment) sub-parameterization: identify the 3 principal moments, orientation fixed ---
# For an arm link the inertia ORIENTATION (iquat) is only weakly observable, so the well-posed sysid is the 3
# PRINCIPAL MOMENTS. Physical realizability of principal moments (I_i) is the triangle inequality
# I_i + I_j >= I_k, i.e. the mass-covariance eigenvalues sigma_i = (sum(I) - 2*I_i)/2 are all >= 0. So we
# parameterize theta3 = log(sigma) (unconstrained -> sigma>0 -> triangle strictly satisfied) and map back with
# I_i = sigma_j + sigma_k. This uses only `body_inertia` (the FD-exact channel) with an analytic Jacobian.


def principal_to_logsigma(I_principal):
  """Inverse: log mass-covariance eigenvalues theta3 from principal moments (requires the triangle inequality)."""
  I0, I1, I2 = I_principal
  sigma = np.array([(-I0 + I1 + I2) / 2, (I0 - I1 + I2) / 2, (I0 + I1 - I2) / 2])
  return np.log(sigma)


def logsigma_to_principal(theta):
  """theta3 (= log mass-covariance eigenvalues) -> principal moments I_i = sigma_j + sigma_k (triangle-safe)."""
  s = np.exp(theta)
  return np.array([s[1] + s[2], s[0] + s[2], s[0] + s[1]])


def principal_jacobian(theta):
  """Analytic d(principal moments)/d(theta3), 3x3. I_i = sum_{k!=i} exp(theta_k) -> dI_i/dtheta_k = s_k (k!=i)."""
  s = np.exp(theta)
  return np.array([[0.0, s[1], s[2]], [s[0], 0.0, s[2]], [s[0], s[1], 0.0]])


def mjc_jacobian(theta, eps=1e-6):
  """Finite-difference Jacobian d(body_inertia[3], iquat_wxyz[4]) / d(theta6), shape (7, 6). The host transform
  is smooth f64 numpy (no float32 cancellation); eigenvector signs are aligned to the base `V` so the columns
  are a continuous function of theta. Returns (J[7,6], inertia0[3], iquat0_wxyz[4], V0)."""
  inertia0, quat0, V0 = params_to_mjc(theta)
  base = np.concatenate([inertia0, quat0])
  J = np.zeros((7, len(theta)))
  for k in range(len(theta)):
    tp, tm = theta.copy(), theta.copy()
    tp[k] += eps
    tm[k] -= eps
    ip, qp, _ = params_to_mjc(tp, ref_V=V0)
    im, qm, _ = params_to_mjc(tm, ref_V=V0)
    J[:, k] = (np.concatenate([ip, qp]) - np.concatenate([im, qm])) / (2.0 * eps)
  return J, inertia0, quat0, V0
