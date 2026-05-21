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


def test_anchored_roll_reference_keeps_x_consistent_across_target():
    """With an anchored roll reference and roll=0, two positions on opposite
    sides of the target should produce gripper X axes pointing in the same
    world direction (not 180° apart as the per-pose reference would).
    """
    target = np.array([400.0, 0.0, 0.0])
    ref = np.array([1.0, 0.0, 0.0])

    R_west = look_at_rotation(np.array([300.0, 0.0, 400.0]), target, 0.0, roll_reference=ref)
    R_east = look_at_rotation(np.array([500.0, 0.0, 400.0]), target, 0.0, roll_reference=ref)

    x_west = R_west[:, 0]
    x_east = R_east[:, 0]
    # Both should point predominantly in the +X direction (positive dot
    # product with the reference) — the anchored reference does not flip.
    assert float(np.dot(x_west, ref)) > 0.5
    assert float(np.dot(x_east, ref)) > 0.5


def test_per_pose_reference_flips_across_target():
    """Sanity counterpart: the default (per-pose) reference DOES flip
    when the position crosses to the other side of the target. Documents
    the behavior that anchored mode fixes.
    """
    target = np.array([400.0, 0.0, 0.0])
    R_west = look_at_rotation(np.array([300.0, 0.0, 400.0]), target, 0.0)
    R_east = look_at_rotation(np.array([500.0, 0.0, 400.0]), target, 0.0)
    # Per-pose x_ref = world_up × z. Crossing the target along X flips
    # the sign of the horizontal projection of z, which flips x_ref.
    assert float(np.dot(R_west[:, 0], R_east[:, 0])) < -0.5


def test_anchored_roll_reference_parallel_raises():
    """If the reference is parallel to the optical axis, the projection
    is ill-defined and look_at_rotation raises ValueError.
    """
    target = np.array([0.0, 0.0, 0.0])
    position = np.array([100.0, 0.0, 0.0])
    # Optical axis = (-1, 0, 0). Reference along the same direction.
    with pytest.raises(ValueError, match="parallel to the optical axis"):
        look_at_rotation(position, target, 0.0, roll_reference=np.array([1.0, 0.0, 0.0]))


def test_anchored_roll_reference_zero_vector_raises():
    target = np.array([400.0, 0.0, 0.0])
    position = np.array([300.0, 0.0, 400.0])
    with pytest.raises(ValueError, match="non-zero vector"):
        look_at_rotation(position, target, 0.0, roll_reference=np.array([0.0, 0.0, 0.0]))


def test_sample_transform_rejects_parallel_reference_and_retries():
    """If some sampled positions produce optical axes parallel to the
    reference, sample_transform should skip them and still return a
    valid pose when the workspace contains enough non-parallel positions.

    With target at origin, reference=+X, and a workspace centered along +X
    but with substantial Y/Z spread, positions near the X axis produce
    optical axes parallel to the reference and get rejected; positions
    away from the axis succeed.
    """
    bounds = {
        "x": {"min": 600.0, "max": 700.0},
        "y": {"min": -100.0, "max": 100.0},
        "z": {"min": -100.0, "max": 100.0},
    }
    target = [0.0, 0.0, 0.0]
    rng = np.random.default_rng(0)
    T = sample_transform(
        bounds,
        target,
        rng=rng,
        roll_reference=np.array([1.0, 0.0, 0.0]),
        max_retries=200,
    )
    assert T.shape == (4, 4)
    # And generating a whole set should also work.
    transforms = generate_pose_set(
        n_poses=5,
        workspace_bounds=bounds,
        look_at_point=target,
        rng=np.random.default_rng(1),
        roll_reference=np.array([1.0, 0.0, 0.0]),
    )
    assert len(transforms) == 5


def test_anchored_roll_range_is_respected():
    """With an anchored reference and a restricted roll range, the actual
    roll angles (measured against the projected reference) stay within the
    configured range.
    """
    target = np.array([400.0, 0.0, 0.0])
    ref = np.array([1.0, 0.0, 0.0])
    bounds = {
        "x": {"min": 200.0, "max": 600.0},
        "y": {"min": -200.0, "max": 200.0},
        "z": {"min": 200.0, "max": 500.0},
    }
    roll_lo_deg, roll_hi_deg = -30.0, 30.0
    roll_range_rad = (np.deg2rad(roll_lo_deg), np.deg2rad(roll_hi_deg))

    rng = np.random.default_rng(0)
    for _ in range(15):
        T = sample_transform(
            bounds, target, roll_range_rad, rng=rng, roll_reference=ref
        )
        position = T[:3, 3]
        z = target - position
        z = z / np.linalg.norm(z)
        x_ref_expected = ref - float(np.dot(ref, z)) * z
        x_ref_expected = x_ref_expected / np.linalg.norm(x_ref_expected)
        y_ref_expected = np.cross(z, x_ref_expected)

        x_actual = T[:3, 0]
        sin_r = float(np.dot(x_actual, y_ref_expected))
        cos_r = float(np.dot(x_actual, x_ref_expected))
        actual_deg = np.degrees(np.arctan2(sin_r, cos_r))
        assert roll_lo_deg - 0.01 <= actual_deg <= roll_hi_deg + 0.01


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
