"""Pose-set diagnostics for hand-eye calibration.

Given a list of arm poses, compute metrics that diagnose whether the pose
distribution is well-conditioned for hand-eye calibration. Catches
ill-conditioned setups (e.g., rotation axes clustered along one direction)
before the calibration runs and produces an actionable feedback string.

References:
- Tsai & Lenz 1989, "A New Technique for Fully Autonomous and Efficient 3D
  Robotics Hand/Eye Calibration" — the "Golden Rules" for pose selection.
- Horn et al. 2023, "User Feedback and Sample Weighting for Ill-Conditioned
  Hand-Eye Calibration", arXiv:2308.06045 — translation condition number and
  rotation-axis density on the unit sphere.
"""

from typing import List, Sequence

import numpy as np


_MIN_ROTATION_RAD = np.deg2rad(0.5)
_TSAI_MIN_MEAN_ANGLE_DEG = 30.0
_CT_WARN_THRESHOLD = 100.0
_DENSITY_RATIO_WARN_THRESHOLD = 3.0


def _axis_angle_from_rotation(R: np.ndarray) -> tuple:
    """Return (axis, angle_rad) for a 3x3 rotation matrix.

    Angle is in [0, π]. Axis is a unit vector. For zero rotations returns a
    placeholder axis (0, 0, 1) and angle 0.
    """
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-9:
        return np.array([0.0, 0.0, 1.0]), 0.0

    sin_theta = np.sin(theta)
    if sin_theta < 1e-6:
        # Near-π rotation: recover axis from the symmetric part of R.
        # The eigenvector of (R + I)/2 with the largest eigenvalue is the axis.
        M = (R + np.eye(3)) / 2.0
        eigvals, eigvecs = np.linalg.eigh(M)
        axis = eigvecs[:, -1]
        return axis / np.linalg.norm(axis), theta

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    ) / (2.0 * sin_theta)
    return axis, theta


def _axis_distance(n1: np.ndarray, n2: np.ndarray) -> float:
    """Distance between two rotation axes treating n and -n as identical.

    Returns the smallest angle (radians, [0, π/2]) needed to align n1 with
    ±n2. 0 means parallel or antiparallel; π/2 means perpendicular.
    """
    return float(np.arccos(np.clip(abs(float(np.dot(n1, n2))), 0.0, 1.0)))


def compute_pose_diversity(
    transforms: Sequence[np.ndarray],
    density_bandwidth_rad: float = 0.3,
) -> dict:
    """Compute diagnostic metrics on a set of arm poses.

    Args:
        transforms: list of 4x4 SE(3) matrices. Typically end-effector poses
            in the robot base frame, but any consistent frame works since the
            diagnostics use only relative motions.
        density_bandwidth_rad: σ of the Gaussian kernel used for rotation-axis
            density estimation on the unit sphere. ~0.3 rad ≈ 17°.

    Returns:
        dict containing diagnostic metrics, a list of warnings, and an
        actionable feedback string. Always includes ``n_poses``, ``n_pairs``,
        ``warnings``, and ``feedback``. When at least two non-trivial inter-pose
        rotations exist, also includes ``translation_condition_number``,
        ``clustered_axis_direction``, axis pair-angle stats, and density stats.
    """
    n = len(transforms)
    if n < 3:
        return {
            "n_poses": n,
            "n_pairs": 0,
            "warnings": [f"need at least 3 poses for diagnostics, got {n}"],
            "feedback": "Provide at least 3 poses to run diagnostics.",
        }

    Ts = [np.asarray(T, dtype=np.float64) for T in transforms]
    rotations = [T[:3, :3] for T in Ts]
    translations = [T[:3, 3] for T in Ts]

    axes_list: List[np.ndarray] = []
    angles_list: List[float] = []
    pair_translations_mm: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            R_rel = rotations[j] @ rotations[i].T
            axis, angle = _axis_angle_from_rotation(R_rel)
            t_rel_norm = float(np.linalg.norm(translations[j] - translations[i]))

            axes_list.append(axis)
            angles_list.append(angle)
            pair_translations_mm.append(t_rel_norm)

    axes = np.array(axes_list)
    angles_rad = np.array(angles_list)
    pair_translations = np.array(pair_translations_mm)

    nonzero = angles_rad > _MIN_ROTATION_RAD
    axes_nz = axes[nonzero]
    angles_nz = angles_rad[nonzero]

    warnings: List[str] = []

    result = {
        "n_poses": n,
        "n_pairs": int(len(angles_rad)),
        "mean_rotation_angle_deg": float(np.degrees(np.mean(angles_rad))),
        "min_rotation_angle_deg": float(np.degrees(np.min(angles_rad))),
        "max_rotation_angle_deg": float(np.degrees(np.max(angles_rad))),
        "mean_pair_translation_mm": float(np.mean(pair_translations)),
        "max_pair_translation_mm": float(np.max(pair_translations)),
    }

    if len(axes_nz) < 2:
        warnings.append("not enough non-trivial rotations to evaluate axis spread")
        result["warnings"] = warnings
        result["feedback"] = (
            "Most pose pairs have near-zero rotation. "
            "Increase angular variation between poses (Tsai's Golden Rule 2)."
        )
        return result

    # Translation condition number c_t.
    # H = Σ 2(1-cos θ) · (I - n n^T) is the 3x3 translation block of the
    # hand-eye normal equations; equivalent to Σ (R_rel - I)^T (R_rel - I).
    # Its smallest eigenvalue's eigenvector points along the axis the
    # rotations are clustered on — i.e. the direction in which translation
    # of X cannot be estimated from this pose set.
    H = np.zeros((3, 3))
    for axis, angle in zip(axes_nz, angles_nz):
        w = 2.0 * (1.0 - np.cos(angle))
        H += w * (np.eye(3) - np.outer(axis, axis))

    eigvals, eigvecs = np.linalg.eigh(H)
    mu_min = float(eigvals[0])
    mu_max = float(eigvals[-1])
    c_t = float(mu_max / mu_min) if mu_min > 1e-12 else float("inf")
    clustered_axis = eigvecs[:, 0]
    # Sign convention: largest-magnitude component positive, for stable display.
    if clustered_axis[int(np.argmax(np.abs(clustered_axis)))] < 0:
        clustered_axis = -clustered_axis

    result["translation_condition_number"] = c_t
    result["clustered_axis_direction"] = clustered_axis.tolist()

    # Pairwise angles between axes (Tsai's Golden Rule 1: maximize spread).
    pair_axis_angles_deg: List[float] = []
    for i in range(len(axes_nz)):
        for j in range(i + 1, len(axes_nz)):
            d = _axis_distance(axes_nz[i], axes_nz[j])
            pair_axis_angles_deg.append(float(np.degrees(d)))
    if pair_axis_angles_deg:
        result["mean_axis_pair_angle_deg"] = float(np.mean(pair_axis_angles_deg))
        result["min_axis_pair_angle_deg"] = float(np.min(pair_axis_angles_deg))
        result["max_axis_pair_angle_deg"] = float(np.max(pair_axis_angles_deg))

    # Rotation-axis density on the sphere (Horn et al. §IV-B).
    sigma = float(density_bandwidth_rad)
    n_axes = len(axes_nz)
    densities = np.zeros(n_axes)
    for i in range(n_axes):
        for j in range(n_axes):
            d = _axis_distance(axes_nz[i], axes_nz[j])
            densities[i] += float(np.exp(-(d * d) / (2.0 * sigma * sigma)))
    mean_density = float(np.mean(densities))
    max_density = float(np.max(densities))
    density_ratio = max_density / mean_density if mean_density > 0 else float("inf")

    result["mean_axis_density"] = mean_density
    result["max_axis_density"] = max_density
    result["axis_density_ratio"] = density_ratio

    feedback_parts: List[str] = []

    if result["mean_rotation_angle_deg"] < _TSAI_MIN_MEAN_ANGLE_DEG:
        warnings.append(
            f"mean inter-pose rotation is {result['mean_rotation_angle_deg']:.1f}° "
            f"— Tsai's Golden Rule 2 wants ≥ {_TSAI_MIN_MEAN_ANGLE_DEG:.0f}°"
        )
        feedback_parts.append(
            "Increase the angular variation between poses (Tsai's Golden Rule 2)."
        )

    if c_t > _CT_WARN_THRESHOLD:
        warnings.append(
            f"translation condition number is {c_t:.1f} (poorly conditioned, > {_CT_WARN_THRESHOLD:.0f})"
        )
        cx, cy, cz = clustered_axis
        feedback_parts.append(
            f"Rotation axes are clustered along [{cx:+.2f}, {cy:+.2f}, {cz:+.2f}] "
            f"in the input frame. Add motions whose rotation axis is perpendicular to this."
        )
    elif density_ratio > _DENSITY_RATIO_WARN_THRESHOLD:
        warnings.append(
            f"max axis density is {density_ratio:.1f}× the mean — some rotation axes over-represented"
        )
        feedback_parts.append(
            "Some rotation axes are over-represented. Add poses with rotations about other axes."
        )

    if not feedback_parts:
        feedback_parts.append("Pose set looks well-conditioned.")

    result["warnings"] = warnings
    result["feedback"] = " ".join(feedback_parts)
    return result
