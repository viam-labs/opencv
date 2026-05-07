"""Tests for src/diagnostics/pose_diversity.py.

Synthetic test cases — pure math, no Viam or Go-binary dependency.
"""

import numpy as np
import pytest

from diagnostics.pose_diversity import (
    _axis_angle_from_rotation,
    _axis_distance,
    compute_pose_diversity,
)


def _rotation_about(axis, angle_rad):
    """Rodrigues' rotation formula. axis must be a unit 3-vector."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def _transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _build_pose_set(rotation_specs, translations=None):
    """Each spec is (axis, angle_rad). Translations default to zero."""
    if translations is None:
        translations = [np.zeros(3)] * len(rotation_specs)
    return [
        _transform(_rotation_about(axis, ang), t)
        for (axis, ang), t in zip(rotation_specs, translations)
    ]


# ---- helper-function tests ----


def test_axis_angle_round_trip():
    axis_in = np.array([0.3, 0.4, 0.866025])
    axis_in /= np.linalg.norm(axis_in)
    angle_in = 0.7
    R = _rotation_about(axis_in, angle_in)
    axis, angle = _axis_angle_from_rotation(R)
    assert angle == pytest.approx(angle_in, abs=1e-9)
    # axis sign may flip; compare absolute alignment
    assert abs(np.dot(axis, axis_in)) == pytest.approx(1.0, abs=1e-9)


def test_axis_angle_identity_returns_zero_angle():
    axis, angle = _axis_angle_from_rotation(np.eye(3))
    assert angle == 0.0


def test_axis_angle_handles_180_degrees():
    R = _rotation_about([1, 0, 0], np.pi)
    axis, angle = _axis_angle_from_rotation(R)
    assert angle == pytest.approx(np.pi, abs=1e-9)
    assert abs(np.dot(axis, [1, 0, 0])) == pytest.approx(1.0, abs=1e-6)


def test_axis_distance_treats_antipodes_as_same():
    n = np.array([0.5, 0.5, np.sqrt(0.5)])
    assert _axis_distance(n, n) == pytest.approx(0.0, abs=1e-12)
    assert _axis_distance(n, -n) == pytest.approx(0.0, abs=1e-12)


def test_axis_distance_perpendicular_is_pi_over_2():
    assert _axis_distance(np.array([1, 0, 0]), np.array([0, 1, 0])) == pytest.approx(
        np.pi / 2, abs=1e-12
    )


# ---- end-to-end diagnostic tests ----


def test_too_few_poses_returns_early():
    out = compute_pose_diversity([np.eye(4), np.eye(4)])
    assert out["n_poses"] == 2
    assert out["n_pairs"] == 0
    assert any("at least 3" in w for w in out["warnings"])


def test_all_identical_poses_warns_about_no_rotation():
    transforms = [np.eye(4) for _ in range(5)]
    out = compute_pose_diversity(transforms)
    assert out["n_poses"] == 5
    assert out["mean_rotation_angle_deg"] == pytest.approx(0.0, abs=1e-9)
    assert any("near-zero rotation" in out["feedback"] or "non-trivial" in w for w in out["warnings"])
    # No condition number computed when there are too few non-trivial rotations.
    assert "translation_condition_number" not in out


def test_single_axis_rotations_have_huge_condition_number():
    # All rotations about z — c_t should blow up; clustered axis should be ẑ.
    angles = np.deg2rad([0, 20, 40, 60, 80, 100])
    transforms = _build_pose_set([([0, 0, 1], a) for a in angles])
    out = compute_pose_diversity(transforms)
    assert out["translation_condition_number"] > 1e3
    clustered = np.array(out["clustered_axis_direction"])
    assert abs(clustered[2]) == pytest.approx(1.0, abs=1e-3)
    assert "clustered" in out["feedback"].lower()


def test_well_distributed_rotations_have_low_condition_number():
    # Rotations about three orthogonal axes, equal magnitudes.
    angles_per_axis = np.deg2rad([0, 40, 80])
    specs = []
    for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        for a in angles_per_axis:
            specs.append((axis, a))
    transforms = _build_pose_set(specs)
    out = compute_pose_diversity(transforms)
    # Three orthogonal axes equally weighted ⇒ H ≈ 2(1-cos θ̄) · 2I ⇒ c_t ≈ 1.
    assert out["translation_condition_number"] < 5.0
    assert "well-conditioned" in out["feedback"].lower()


def test_two_axis_rotations_are_intermediate():
    # Half about x, half about y — c_t finite but worse than uniform.
    # Pairwise rotations cover x, y, and combinations, so translation along x
    # and y are each only missed by half the pair-rotations (those about that
    # axis). Translation along z is always observable. The smallest-eigenvalue
    # direction therefore lies in the xy plane (z component should be small).
    angles = np.deg2rad([20, 40, 60])
    specs = [([1, 0, 0], a) for a in angles] + [([0, 1, 0], a) for a in angles]
    transforms = _build_pose_set(specs)
    out = compute_pose_diversity(transforms)
    c_t = out["translation_condition_number"]
    assert c_t > 1.5  # worse than uniform
    assert c_t < 100.0  # but not catastrophic
    clustered = np.array(out["clustered_axis_direction"])
    assert abs(clustered[2]) < 0.5  # poorly-conditioned direction is in xy plane


def test_small_rotations_trigger_tsai_warning():
    # All rotations are small (< 30° mean).
    angles = np.deg2rad([5, 8, 10, 12])
    specs = [([1, 0, 0], a) for a in angles] + [([0, 1, 0], a) for a in angles]
    transforms = _build_pose_set(specs)
    out = compute_pose_diversity(transforms)
    assert out["mean_rotation_angle_deg"] < 30.0
    assert any("Golden Rule 2" in w or "Tsai" in w for w in out["warnings"])


def test_translation_magnitudes_reported():
    # Same rotations, varying translations.
    specs = [([1, 0, 0], np.deg2rad(30)) for _ in range(4)]
    translations = [
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 0.0, 0.0]),
        np.array([0.0, 200.0, 0.0]),
        np.array([0.0, 0.0, 50.0]),
    ]
    transforms = _build_pose_set(specs, translations=translations)
    out = compute_pose_diversity(transforms)
    assert out["max_pair_translation_mm"] == pytest.approx(
        max(
            np.linalg.norm(translations[i] - translations[j])
            for i in range(4)
            for j in range(i + 1, 4)
        ),
        abs=1e-9,
    )


def test_density_ratio_higher_when_axes_clustered():
    # Compare a clustered set (mostly z-axis) vs a uniform set, on the same
    # number of poses. The clustered set should have a noticeably higher
    # density ratio than the uniform set.
    clustered_specs = [([0, 0, 1], np.deg2rad(a)) for a in [10, 20, 30, 40, 50, 60, 70, 80, 90]]
    clustered_specs.append(([1, 0, 0], np.deg2rad(45)))
    clustered_specs.append(([0, 1, 0], np.deg2rad(45)))
    clustered_out = compute_pose_diversity(_build_pose_set(clustered_specs))

    uniform_specs = []
    for axis in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        for a in [20, 40, 60, 80]:
            uniform_specs.append((axis, np.deg2rad(a)))
    uniform_out = compute_pose_diversity(_build_pose_set(uniform_specs))

    assert clustered_out["axis_density_ratio"] > uniform_out["axis_density_ratio"]
    assert clustered_out["axis_density_ratio"] > 1.2
