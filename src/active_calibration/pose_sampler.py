"""Random pose sampling within a rectangular workspace.

Generates end-effector poses whose Z axis points at a user-supplied look-at
target, with random roll about that axis. Assumes the camera is mounted with
its optical axis roughly aligned with the end-effector's +Z axis (the common
mount); the chessboard tracker is forgiving as long as the board is in the
camera FOV.

The roll variation is the key trick: a position-only random sample with
"look-at" orientation produces rotation axes that all lie in a plane
perpendicular to the look-at direction, which Phase 1's translation
condition number flags as ill-conditioned. Randomizing the roll about the
optical axis breaks that clustering and gives the diverse rotation-axis
distribution the calibration needs.

All vectors are in the robot base frame. Positions are in millimeters,
angles in radians.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


WorkspaceBounds = Dict[str, Sequence[float]]


def _validate_workspace_bounds(workspace_bounds: WorkspaceBounds) -> None:
    for axis in ("x", "y", "z"):
        if axis not in workspace_bounds:
            raise ValueError(f"workspace_bounds missing '{axis}' axis")
        bounds = workspace_bounds[axis]
        if not isinstance(bounds, dict) or "min" not in bounds or "max" not in bounds:
            raise ValueError(
                f"workspace_bounds[{axis!r}] must be a dict with 'min' and 'max' keys"
            )
        lo, hi = float(bounds["min"]), float(bounds["max"])
        if lo >= hi:
            raise ValueError(
                f"workspace_bounds[{axis!r}] must have min < max, got min={lo} max={hi}"
            )


def look_at_rotation(
    position: np.ndarray,
    target: np.ndarray,
    roll_rad: float,
) -> np.ndarray:
    """Build a 3x3 rotation whose third column is the unit vector from
    ``position`` to ``target``, with the rotation about that axis
    parameterized by ``roll_rad``.

    The returned matrix R has columns equal to the end-effector axes
    expressed in the base frame (i.e., R takes vectors in end-effector
    coordinates and produces vectors in base coordinates).

    Raises ``ValueError`` if ``position`` and ``target`` coincide.
    """
    position = np.asarray(position, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)

    z = target - position
    z_norm = float(np.linalg.norm(z))
    if z_norm < 1e-9:
        raise ValueError("sampled position coincides with look-at target")
    z = z / z_norm

    # Reference up vector. Fall back if z is nearly parallel to world +Z.
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z, world_up))) > 0.99:
        world_up = np.array([1.0, 0.0, 0.0])

    x_ref = np.cross(world_up, z)
    x_ref = x_ref / np.linalg.norm(x_ref)
    y_ref = np.cross(z, x_ref)

    c, s = np.cos(roll_rad), np.sin(roll_rad)
    x = c * x_ref + s * y_ref
    y = -s * x_ref + c * y_ref

    return np.column_stack([x, y, z])


def sample_transform(
    workspace_bounds: WorkspaceBounds,
    look_at_point: Sequence[float],
    roll_range_rad: Tuple[float, float] = (-np.pi, np.pi),
    rng: Optional[np.random.Generator] = None,
    max_retries: int = 32,
) -> np.ndarray:
    """Sample a single 4x4 SE(3) transform.

    Position is uniform in ``workspace_bounds``; orientation is built by
    :func:`look_at_rotation` with a roll uniform in ``roll_range_rad``.

    If a sampled position coincides with ``look_at_point`` (degenerate
    look-at), retries up to ``max_retries`` times before raising.
    """
    _validate_workspace_bounds(workspace_bounds)
    if rng is None:
        rng = np.random.default_rng()

    target = np.asarray(look_at_point, dtype=np.float64).reshape(3)
    roll_lo, roll_hi = float(roll_range_rad[0]), float(roll_range_rad[1])
    if roll_lo > roll_hi:
        raise ValueError(f"roll_range_rad min > max: ({roll_lo}, {roll_hi})")

    for _ in range(max_retries):
        position = np.array(
            [
                rng.uniform(workspace_bounds["x"]["min"], workspace_bounds["x"]["max"]),
                rng.uniform(workspace_bounds["y"]["min"], workspace_bounds["y"]["max"]),
                rng.uniform(workspace_bounds["z"]["min"], workspace_bounds["z"]["max"]),
            ]
        )
        if np.linalg.norm(position - target) < 1e-6:
            continue
        roll = rng.uniform(roll_lo, roll_hi)
        R = look_at_rotation(position, target, roll)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        return T

    raise RuntimeError(
        f"could not sample non-degenerate pose after {max_retries} attempts; "
        "look_at_point may be inside the workspace_bounds and the volume is small"
    )


def generate_pose_set(
    n_poses: int,
    workspace_bounds: WorkspaceBounds,
    look_at_point: Sequence[float],
    roll_range_rad: Tuple[float, float] = (-np.pi, np.pi),
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Generate ``n_poses`` independent sample transforms."""
    if n_poses < 1:
        raise ValueError(f"n_poses must be >= 1, got {n_poses}")
    if rng is None:
        rng = np.random.default_rng()
    return [
        sample_transform(workspace_bounds, look_at_point, roll_range_rad, rng)
        for _ in range(n_poses)
    ]
