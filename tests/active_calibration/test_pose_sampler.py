"""Tests for active_calibration.pose_sampler."""

import numpy as np
import pytest

from active_calibration.pose_sampler import (
    generate_pose_set,
    look_at_rotation,
    sample_transform,
)
from diagnostics.pose_diversity import compute_pose_diversity


def test_look_at_rotation_z_points_at_target():
    position = np.array([0.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0])
    R = look_at_rotation(position, target, roll_rad=0.0)
    z_axis = R[:, 2]
    assert np.allclose(z_axis, [1.0, 0.0, 0.0])


def test_look_at_rotation_is_orthonormal():
    R = look_at_rotation(np.array([3.0, -2.0, 5.0]), np.array([1.0, 1.0, 1.0]), 0.7)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.isclose(float(np.linalg.det(R)), 1.0)


def test_look_at_rotation_roll_rotates_xy_about_z():
    position = np.array([0.0, 0.0, 0.0])
    target = np.array([1.0, 0.0, 0.0])
    R0 = look_at_rotation(position, target, roll_rad=0.0)
    R1 = look_at_rotation(position, target, roll_rad=np.pi / 2)
    # Z axis must be unchanged by roll.
    assert np.allclose(R0[:, 2], R1[:, 2])
    # X and Y axes must differ.
    assert not np.allclose(R0[:, 0], R1[:, 0])


def test_look_at_rotation_degenerate_raises():
    p = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="coincides"):
        look_at_rotation(p, p, roll_rad=0.0)


def test_look_at_handles_z_parallel_axis():
    # Look straight up: target directly above position, so z is parallel to world up.
    position = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])
    R = look_at_rotation(position, target, roll_rad=0.0)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.allclose(R[:, 2], [0.0, 0.0, 1.0])


def test_sample_transform_position_within_bounds():
    rng = np.random.default_rng(42)
    bounds = {
        "x": {"min": 100.0, "max": 200.0},
        "y": {"min": -50.0, "max": 50.0},
        "z": {"min": 300.0, "max": 400.0},
    }
    target = [150.0, 0.0, 0.0]
    for _ in range(50):
        T = sample_transform(bounds, target, rng=rng)
        p = T[:3, 3]
        assert 100.0 <= p[0] <= 200.0
        assert -50.0 <= p[1] <= 50.0
        assert 300.0 <= p[2] <= 400.0


def test_sample_transform_z_axis_points_at_target():
    rng = np.random.default_rng(0)
    bounds = {
        "x": {"min": 100.0, "max": 200.0},
        "y": {"min": -50.0, "max": 50.0},
        "z": {"min": 300.0, "max": 400.0},
    }
    target = np.array([0.0, 0.0, 0.0])
    for _ in range(20):
        T = sample_transform(bounds, target, rng=rng)
        position = T[:3, 3]
        z_axis = T[:3, 2]
        expected = target - position
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(z_axis, expected, atol=1e-10)


def test_sample_transform_invalid_bounds():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="min < max"):
        sample_transform(
            {
                "x": {"min": 200.0, "max": 100.0},
                "y": {"min": -50.0, "max": 50.0},
                "z": {"min": 300.0, "max": 400.0},
            },
            [0.0, 0.0, 0.0],
            rng=rng,
        )
    with pytest.raises(ValueError, match="missing"):
        sample_transform(
            {
                "x": {"min": 100.0, "max": 200.0},
                "y": {"min": -50.0, "max": 50.0},
            },
            [0.0, 0.0, 0.0],
            rng=rng,
        )
    with pytest.raises(ValueError, match="'min' and 'max' keys"):
        sample_transform(
            {
                "x": [100.0, 200.0],
                "y": {"min": -50.0, "max": 50.0},
                "z": {"min": 300.0, "max": 400.0},
            },
            [0.0, 0.0, 0.0],
            rng=rng,
        )


def test_generate_pose_set_with_roll_is_well_conditioned():
    """A pose set generated with random roll about the optical axis should
    have a reasonable translation condition number (rotation axes spread
    across the unit sphere rather than clustered).
    """
    rng = np.random.default_rng(7)
    bounds = {
        "x": {"min": 100.0, "max": 400.0},
        "y": {"min": -150.0, "max": 150.0},
        "z": {"min": 200.0, "max": 500.0},
    }
    target = [250.0, 0.0, 0.0]
    transforms = generate_pose_set(
        n_poses=15,
        workspace_bounds=bounds,
        look_at_point=target,
        roll_range_rad=(-np.pi, np.pi),
        rng=rng,
    )
    diversity = compute_pose_diversity(transforms)
    assert diversity["translation_condition_number"] < 100.0


def test_random_roll_improves_conditioning_vs_zero_roll():
    """The full-roll-range pose set should have a meaningfully lower
    translation condition number than the zero-roll pose set on the same
    workspace and target. This is the core justification for randomizing
    roll about the optical axis — it breaks rotation-axis clustering.
    """
    bounds = {
        "x": {"min": 100.0, "max": 400.0},
        "y": {"min": -150.0, "max": 150.0},
        "z": {"min": 200.0, "max": 500.0},
    }
    target = [250.0, 0.0, 0.0]

    rng_a = np.random.default_rng(7)
    full_roll = generate_pose_set(
        n_poses=15,
        workspace_bounds=bounds,
        look_at_point=target,
        roll_range_rad=(-np.pi, np.pi),
        rng=rng_a,
    )
    rng_b = np.random.default_rng(7)
    no_roll = generate_pose_set(
        n_poses=15,
        workspace_bounds=bounds,
        look_at_point=target,
        roll_range_rad=(0.0, 0.0),
        rng=rng_b,
    )
    c_full = compute_pose_diversity(full_roll)["translation_condition_number"]
    c_none = compute_pose_diversity(no_roll)["translation_condition_number"]
    assert c_full < c_none


def test_generate_pose_set_count():
    rng = np.random.default_rng(0)
    bounds = {
        "x": {"min": 0.0, "max": 100.0},
        "y": {"min": 0.0, "max": 100.0},
        "z": {"min": 0.0, "max": 100.0},
    }
    transforms = generate_pose_set(
        n_poses=7, workspace_bounds=bounds, look_at_point=[-100, -100, -100], rng=rng
    )
    assert len(transforms) == 7
    for T in transforms:
        assert T.shape == (4, 4)


def test_generate_pose_set_n_zero_raises():
    with pytest.raises(ValueError, match=">= 1"):
        generate_pose_set(
            n_poses=0,
            workspace_bounds={
                "x": {"min": 0, "max": 1},
                "y": {"min": 0, "max": 1},
                "z": {"min": 0, "max": 1},
            },
            look_at_point=[-1, -1, -1],
        )


def test_sample_transform_deterministic_with_seed():
    bounds = {
        "x": {"min": 100.0, "max": 200.0},
        "y": {"min": -50.0, "max": 50.0},
        "z": {"min": 300.0, "max": 400.0},
    }
    target = [0.0, 0.0, 0.0]
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    T_a = sample_transform(bounds, target, rng=rng_a)
    T_b = sample_transform(bounds, target, rng=rng_b)
    assert np.allclose(T_a, T_b)
