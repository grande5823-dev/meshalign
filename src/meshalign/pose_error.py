"""Pose-recovery error metric: how close does the registration transform
get to the ground-truth applied transform?

Setup: a known rigid transform T_applied is applied to a reference mesh to
create a test mesh (optionally with noise). Registration produces T_align
(an estimate of T_applied^{-1}). The residual transform

    T_residual = T_align @ T_applied

is identity for a perfect recovery. The geometric error is the per-vertex
displacement under T_residual:

    err_i = || (T_residual @ v_i) - v_i ||

in mm. For a part of characteristic size L:
  - a rotation error θ displaces vertices up to L·sin(θ) at the extremes
  - a translation error t displaces all vertices by t
  - mean/max/RMS give the typical and worst-case geometric error

Also decomposes into pure rotation (deg) and pure translation (mm) for
reporting. Zero iff the pose is perfectly recovered. Rotationally
symmetric parts (sphere, cone) can have large rotation error with zero
surface error — that's the C7 claim (pose is unrecoverable; surface fit
is what matters for those).

Pure functions, no server needed. Test with known transforms.
"""
from __future__ import annotations

import numpy as np


def residual_transform(T_align: np.ndarray, T_applied: np.ndarray) -> np.ndarray:
    """The leftover transform after alignment: T_align @ T_applied.

    Identity (up to float precision) = perfect pose recovery.
    Both inputs are 4x4; convention: T_applied maps ref → test,
    T_align maps test → ref.
    """
    return np.asarray(T_align, dtype=np.float64) @ np.asarray(T_applied, dtype=np.float64)


def pose_recovery_error(vertices: np.ndarray, T_align: np.ndarray,
                        T_applied: np.ndarray) -> dict:
    """Geometric pose-recovery error for a mesh under a known applied transform.

    Parameters
    ----------
    vertices : (N, 3) array — the ORIGINAL (untransformed) reference vertices
    T_align : 4x4 — the transform returned by registration (test → ref)
    T_applied : 4x4 — the known transform that was applied (ref → test)

    Returns dict with:
      pose_err_mean_mm, pose_err_max_mm, pose_err_rms_mm
      pose_rot_deg, pose_trans_mm
    """
    T_res = residual_transform(T_align, T_applied)
    v = np.asarray(vertices, dtype=np.float64)
    moved = v @ T_res[:3, :3].T + T_res[:3, 3]
    disp = np.linalg.norm(moved - v, axis=1)
    R = T_res[:3, :3]
    # geodesic rotation angle
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(tr)))
    trans_mm = float(np.linalg.norm(T_res[:3, 3]))
    return {
        "pose_err_mean_mm": round(float(disp.mean()), 6),
        "pose_err_max_mm": round(float(disp.max()), 6),
        "pose_err_rms_mm": round(float(np.sqrt((disp ** 2).mean())), 6),
        "pose_rot_deg": round(rot_deg, 6),
        "pose_trans_mm": round(trans_mm, 6),
    }


def decompose_rigid(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 rigid transform into (rotation 3x3, translation 3)."""
    T = np.asarray(T, dtype=np.float64)
    return T[:3, :3].copy(), T[:3, 3].copy()


def rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic rotation angle of a 3x3 rotation matrix, in degrees."""
    tr = np.clip((np.trace(np.asarray(R)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))
