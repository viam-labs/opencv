"""Tests for src/solvers/reprojection_solver.py.

Synthetic ground-truth scenarios. Build a known true (X, Y) and a set of
stations, project corners to image (optionally with pixel noise), then verify
the solver recovers X within tolerance.
"""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from solvers.reprojection_solver import (
    _se3_inverse,
    bootstrap_T_bw,
    refine_handeye,
)


def _se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _euler_xyz_se3(rx_deg, ry_deg, rz_deg, t):
    R = Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
    return _se3(R, np.asarray(t, dtype=np.float64))


def _board_corners(rows=5, cols=7, square_mm=25.0):
    pts = np.zeros((rows * cols, 3), dtype=np.float64)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    pts *= square_mm
    return pts


def _camera_intrinsics():
    K = np.array(
        [
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    dist = np.zeros(5)
    return K, dist


def _project_to_camera(P_w, T_ce, T_eb, T_bw, K, dist):
    M = T_ce @ T_eb @ T_bw
    rvec, _ = cv2.Rodrigues(M[:3, :3])
    proj, _ = cv2.projectPoints(P_w, rvec, M[:3, 3].reshape(3, 1), K, dist)
    return proj.reshape(-1, 2)


def _build_synthetic_dataset(noise_pixels=0.0, seed=0):
    """Build a noise-free or noisy synthetic dataset.

    Returns dict with keys: T_be_list, corners_2d_list, corners_3d, K, dist,
    X_true, Y_true, T_cw_list.
    """
    rng = np.random.default_rng(seed)
    K, dist = _camera_intrinsics()
    corners_3d = _board_corners()

    # Truth: camera offset 80 mm along gripper +Z, slightly rotated.
    X_true = _euler_xyz_se3(10.0, -5.0, 15.0, [10.0, 5.0, 80.0])
    # Board sits 500 mm in front of base, 100 mm to the right, on a flat table.
    Y_true = _euler_xyz_se3(0.0, 180.0, 0.0, [400.0, 100.0, 0.0])

    # Stations: place gripper above the board with varying orientation so the
    # camera sees it from a good angle.
    rotation_specs = [
        (-150.0, 0.0, 0.0),
        (-160.0, 10.0, 5.0),
        (-150.0, -15.0, 10.0),
        (-145.0, 5.0, -10.0),
        (-155.0, -5.0, 15.0),
        (-160.0, 15.0, -5.0),
        (-145.0, -10.0, 0.0),
        (-150.0, 0.0, 20.0),
    ]
    translation_specs = [
        [400.0, 100.0, 400.0],
        [380.0, 90.0, 420.0],
        [420.0, 110.0, 410.0],
        [410.0, 80.0, 430.0],
        [390.0, 120.0, 400.0],
        [400.0, 100.0, 450.0],
        [430.0, 100.0, 405.0],
        [380.0, 105.0, 415.0],
    ]

    T_be_list = []
    corners_2d_list = []
    T_cw_list = []
    for rs, ts in zip(rotation_specs, translation_specs):
        T_be = _euler_xyz_se3(*rs, ts)
        T_be_list.append(T_be)
        T_eb = _se3_inverse(T_be)
        T_ce = _se3_inverse(X_true)
        u = _project_to_camera(corners_3d, T_ce, T_eb, Y_true, K, dist)
        if noise_pixels > 0:
            u = u + rng.normal(0, noise_pixels, size=u.shape)
        corners_2d_list.append(u)

        # Recover T_cw_k from solvePnP on the (possibly noisy) corners.
        ok, rvec, tvec = cv2.solvePnP(corners_3d, u.astype(np.float64), K, dist)
        assert ok
        R_cw, _ = cv2.Rodrigues(rvec)
        T_cw_list.append(_se3(R_cw, tvec.reshape(3)))

    return {
        "T_be_list": T_be_list,
        "corners_2d_list": corners_2d_list,
        "corners_3d": corners_3d,
        "K": K,
        "dist": dist,
        "X_true": X_true,
        "Y_true": Y_true,
        "T_cw_list": T_cw_list,
    }


def _se3_diff(T_a, T_b):
    """Return (translation_error_mm, rotation_error_deg) between two SE(3) poses."""
    delta_R = T_a[:3, :3] @ T_b[:3, :3].T
    cos_angle = np.clip((np.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(cos_angle)))
    trans_mm = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    return trans_mm, rot_deg


# ---- tests ----


def test_noise_free_recovery():
    data = _build_synthetic_dataset(noise_pixels=0.0)
    # Start from a perturbed X (so we actually have to optimize).
    X_init = data["X_true"] @ _euler_xyz_se3(2.0, -1.5, 1.0, [3.0, -2.0, 4.0])
    out = refine_handeye(
        data["T_be_list"],
        data["corners_2d_list"],
        data["corners_3d"],
        data["K"],
        data["dist"],
        X_init=X_init,
        T_cw_list=data["T_cw_list"],
    )
    trans_err, rot_err = _se3_diff(out["X_refined"], data["X_true"])
    assert trans_err < 0.1, f"translation error {trans_err:.4f} mm too high"
    assert rot_err < 0.05, f"rotation error {rot_err:.4f} deg too high"
    assert out["rmse_pixels"] < 0.5
    assert out["success"]


def test_partial_views_per_station_corners_recovery():
    """ChArUco-style partial views: each station detects a different subset of
    corners, so corners_3d is passed as a per-station list with varying N_k.
    The refiner should still recover X.
    """
    data = _build_synthetic_dataset(noise_pixels=0.0)
    full_3d = data["corners_3d"]
    n_total = full_3d.shape[0]
    rng = np.random.default_rng(7)

    corners_2d_partial = []
    corners_3d_partial = []
    for k, c2d in enumerate(data["corners_2d_list"]):
        # Keep a different ~70% subset of corners at each station.
        keep = np.sort(rng.choice(n_total, size=int(n_total * 0.7), replace=False))
        corners_2d_partial.append(c2d[keep])
        corners_3d_partial.append(full_3d[keep])

    X_init = data["X_true"] @ _euler_xyz_se3(2.0, -1.5, 1.0, [3.0, -2.0, 4.0])
    out = refine_handeye(
        data["T_be_list"],
        corners_2d_partial,
        corners_3d_partial,  # per-station list, variable N_k
        data["K"],
        data["dist"],
        X_init=X_init,
        T_cw_list=data["T_cw_list"],
    )
    trans_err, rot_err = _se3_diff(out["X_refined"], data["X_true"])
    assert trans_err < 0.1, f"translation error {trans_err:.4f} mm too high"
    assert rot_err < 0.05, f"rotation error {rot_err:.4f} deg too high"
    assert len(out["per_pose_rmse_pixels"]) == len(data["T_be_list"])
    assert out["success"]


def test_mismatched_2d_3d_corner_counts_raises():
    data = _build_synthetic_dataset(noise_pixels=0.0)
    bad_3d = [data["corners_3d"][:-1] for _ in data["T_be_list"]]  # drop one 3d point
    X_init = data["X_true"]
    with pytest.raises(ValueError):
        refine_handeye(
            data["T_be_list"],
            data["corners_2d_list"],
            bad_3d,
            data["K"],
            data["dist"],
            X_init=X_init,
            T_cw_list=data["T_cw_list"],
        )


def test_jacobian_shape():
    data = _build_synthetic_dataset(noise_pixels=0.0)
    X_init = data["X_true"] @ _euler_xyz_se3(2.0, -1.5, 1.0, [3.0, -2.0, 4.0])
    out = refine_handeye(
        data["T_be_list"],
        data["corners_2d_list"],
        data["corners_3d"],
        data["K"],
        data["dist"],
        X_init=X_init,
        T_cw_list=data["T_cw_list"],
    )
    n_stations = len(data["T_be_list"])
    n_corners = data["corners_3d"].shape[0]
    expected_rows = 2 * n_stations * n_corners
    assert out["jacobian"].shape == (expected_rows, 12)


def test_per_pose_rmse_one_per_station():
    data = _build_synthetic_dataset(noise_pixels=0.5, seed=1)
    X_init = data["X_true"] @ _euler_xyz_se3(1.0, 0.5, -0.5, [1.0, -1.0, 1.5])
    out = refine_handeye(
        data["T_be_list"],
        data["corners_2d_list"],
        data["corners_3d"],
        data["K"],
        data["dist"],
        X_init=X_init,
        T_cw_list=data["T_cw_list"],
    )
    assert len(out["per_pose_rmse_pixels"]) == len(data["T_be_list"])
    assert all(r > 0 for r in out["per_pose_rmse_pixels"])


def test_noisy_recovery_within_tolerance():
    data = _build_synthetic_dataset(noise_pixels=0.5, seed=42)
    X_init = data["X_true"] @ _euler_xyz_se3(2.0, -1.0, 1.5, [2.0, -2.0, 3.0])
    out = refine_handeye(
        data["T_be_list"],
        data["corners_2d_list"],
        data["corners_3d"],
        data["K"],
        data["dist"],
        X_init=X_init,
        T_cw_list=data["T_cw_list"],
    )
    trans_err, rot_err = _se3_diff(out["X_refined"], data["X_true"])
    assert trans_err < 5.0, f"translation error {trans_err:.3f} mm too high under 0.5px noise"
    assert rot_err < 1.0, f"rotation error {rot_err:.3f} deg too high under 0.5px noise"


def test_refinement_beats_calibratehandeye_bootstrap_under_noise():
    data = _build_synthetic_dataset(noise_pixels=1.0, seed=7)
    K = data["K"]
    dist = data["dist"]

    # Use OpenCV calibrateHandEye for the bootstrap, mirroring the existing
    # service code path. It expects T_gripper2base (R, t) and T_target2cam (R, t).
    R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list = [], [], [], []
    for T_be, T_cw in zip(data["T_be_list"], data["T_cw_list"]):
        R_g2b_list.append(T_be[:3, :3])
        t_g2b_list.append(T_be[:3, 3].reshape(3, 1))
        R_t2c_list.append(T_cw[:3, :3])
        t_t2c_list.append(T_cw[:3, 3].reshape(3, 1))
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list, method=cv2.CALIB_HAND_EYE_TSAI
    )
    X_bootstrap = _se3(R_c2g, t_c2g.reshape(3))

    out = refine_handeye(
        data["T_be_list"],
        data["corners_2d_list"],
        data["corners_3d"],
        K,
        dist,
        X_init=X_bootstrap,
        T_cw_list=data["T_cw_list"],
    )

    bootstrap_trans, bootstrap_rot = _se3_diff(X_bootstrap, data["X_true"])
    refined_trans, refined_rot = _se3_diff(out["X_refined"], data["X_true"])

    # Refinement should at least match the bootstrap, ideally beat it.
    assert refined_trans <= bootstrap_trans + 1e-3, (
        f"refined translation worse than bootstrap: {refined_trans:.4f} vs {bootstrap_trans:.4f}"
    )
    assert refined_rot <= bootstrap_rot + 1e-3, (
        f"refined rotation worse than bootstrap: {refined_rot:.4f} vs {bootstrap_rot:.4f}"
    )


def test_too_few_stations_raises():
    data = _build_synthetic_dataset(noise_pixels=0.0)
    with pytest.raises(ValueError, match="at least 3"):
        refine_handeye(
            data["T_be_list"][:2],
            data["corners_2d_list"][:2],
            data["corners_3d"],
            data["K"],
            data["dist"],
            X_init=data["X_true"],
            T_cw_list=data["T_cw_list"][:2],
        )


def test_bootstrap_T_bw_recovers_truth_with_perfect_inputs():
    data = _build_synthetic_dataset(noise_pixels=0.0)
    Y_recovered = bootstrap_T_bw(
        data["T_be_list"][0], data["X_true"], data["T_cw_list"][0]
    )
    trans_err, rot_err = _se3_diff(Y_recovered, data["Y_true"])
    assert trans_err < 1e-6
    assert rot_err < 1e-6
