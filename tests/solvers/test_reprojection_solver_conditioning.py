"""Regression tests for the ill-conditioning divergence in refine_handeye.

Background
----------
``refine_handeye`` optimizes 12 variables: T_ec (camera-in-end-effector) and
T_bw (board-in-base), each parameterized as (rotvec[rad], translation[mm]). The
rotation components have magnitude ~1 while the translation components have
magnitude ~10^2-10^3. ``scipy.optimize.least_squares(method="trf")`` without
per-variable scaling takes wildly oversized steps along the translation axes,
which can push board points onto/behind the camera plane where
``cv2.projectPoints`` blows up -- the optimizer then diverges (rmse -> 1e6+ px)
and returns a garbage hand-eye pose.

This stayed hidden for a fully-visible chessboard: 30+ corners spanning the
whole board give a strong, well-conditioned Jacobian that converges anyway. It
surfaces with a ChArUco board, whose partial/clustered per-view corner subsets
give a weak, ill-conditioned Jacobian -- exactly the regime where bad variable
scaling tips the solve into divergence.

The fix is ``x_scale="jac"`` in the ``least_squares`` call. These tests build a
synthetic ChArUco-style scenario (partial clustered corner subsets, pixel
noise) that reliably diverges without the fix, and assert that the solver now
recovers the true hand-eye transform.
"""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from solvers.reprojection_solver import _se3_inverse, refine_handeye


# ---- helpers ----


def _se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _euler_xyz_se3(rx_deg, ry_deg, rz_deg, t):
    R = Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
    return _se3(R, np.asarray(t, dtype=np.float64))


def _se3_diff(T_a, T_b):
    delta_R = T_a[:3, :3] @ T_b[:3, :3].T
    cos_angle = np.clip((np.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(cos_angle)))
    trans_mm = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    return trans_mm, rot_deg


# 6x4 interior-corner grid (a 5x7-square ChArUco board), 30 mm spacing.
_ROWS, _COLS, _SQUARE = 6, 4, 30.0
_K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
_DIST = np.zeros(5)
_X_TRUE = _euler_xyz_se3(10.0, -5.0, 15.0, [10.0, 5.0, 80.0])
_Y_TRUE = _euler_xyz_se3(0.0, 180.0, 0.0, [400.0, 100.0, 0.0])


def _full_board():
    pts = np.zeros((_ROWS * _COLS, 3), dtype=np.float64)
    pts[:, :2] = np.mgrid[0:_COLS, 0:_ROWS].T.reshape(-1, 2)
    return pts * _SQUARE


def _build_charuco_like_dataset(seed, n_poses=14, noise_px=1.0):
    """Synthetic eye-in-hand dataset with ChArUco-style partial views.

    Each station sees only a contiguous 3x3 block of interior corners (a partial
    view of one board region), near fronto-parallel, with pixel noise. The
    per-station board-in-camera pose is recovered with the same IPPE solve the
    service uses. This is the weak-Jacobian regime that diverges without
    ``x_scale``.
    """
    rng = np.random.default_rng(seed)
    full = _full_board()

    T_be_list, corners_2d_list, corners_3d_list, T_cw_list = [], [], [], []
    for _ in range(n_poses):
        # Near fronto-parallel (small tilt off straight-down), full roll spread.
        tilt = 2.0
        rs = (
            -180.0 + tilt * rng.uniform(-1, 1),
            tilt * rng.uniform(-1, 1),
            rng.uniform(-180, 180),
        )
        ts = [
            400.0 + rng.uniform(-20, 20),
            100.0 + rng.uniform(-20, 20),
            700.0 + rng.uniform(-20, 20),
        ]
        T_be = _euler_xyz_se3(*rs, ts)
        T_be_list.append(T_be)

        M = _se3_inverse(_X_TRUE) @ _se3_inverse(T_be) @ _Y_TRUE
        rvec, _ = cv2.Rodrigues(M[:3, :3])
        u, _ = cv2.projectPoints(full, rvec, M[:3, 3].reshape(3, 1), _K, _DIST)
        u = u.reshape(-1, 2) + rng.normal(0, noise_px, (full.shape[0], 2))

        # Contiguous 3x3 block of corners -> a clustered partial view.
        c0 = int(rng.integers(0, _COLS - 2))
        r0 = int(rng.integers(0, _ROWS - 2))
        keep = np.array(
            [c * _ROWS + r for c in range(c0, c0 + 3) for r in range(r0, r0 + 3)]
        )
        c2d = u[keep]
        c3d = full[keep]
        corners_2d_list.append(c2d)
        corners_3d_list.append(c3d)

        # Per-station PnP exactly as the service does it (IPPE, best branch).
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            c3d, c2d.astype(np.float64), _K, _DIST, flags=cv2.SOLVEPNP_IPPE
        )
        R_cw, _ = cv2.Rodrigues(rvecs[0])
        T_cw_list.append(_se3(R_cw, tvecs[0].reshape(3)))

    return {
        "T_be_list": T_be_list,
        "corners_2d_list": corners_2d_list,
        "corners_3d_list": corners_3d_list,
        "T_cw_list": T_cw_list,
    }


def _bootstrap_X(data):
    """cv2.calibrateHandEye bootstrap, mirroring the service's reprojection path."""
    R_g2b = [T[:3, :3] for T in data["T_be_list"]]
    t_g2b = [T[:3, 3].reshape(3, 1) for T in data["T_be_list"]]
    R_t2c = [T[:3, :3] for T in data["T_cw_list"]]
    t_t2c = [T[:3, 3].reshape(3, 1) for T in data["T_cw_list"]]
    R, t = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI
    )
    return _se3(R, t.reshape(3))


# Seeds chosen because they reproduce the divergence with the unscaled solver
# (verified: refined X error of hundreds-to-thousands of mm, success=False).
_DIVERGING_SEEDS = [1, 3, 4]


@pytest.mark.parametrize("seed", _DIVERGING_SEEDS)
def test_charuco_partial_views_recover_with_conditioning(seed):
    """The exact scenario that diverged now converges to the true hand-eye pose.

    Per-station poses here are NOT flipped (IPPE picks the correct branch); the
    failure was purely optimizer conditioning. With ``x_scale="jac"`` the solve
    recovers X to within a few mm / sub-degree despite the weak partial-view
    Jacobian.
    """
    data = _build_charuco_like_dataset(seed)
    X_init = _bootstrap_X(data)

    out = refine_handeye(
        data["T_be_list"],
        data["corners_2d_list"],
        data["corners_3d_list"],
        _K,
        _DIST,
        X_init=X_init,
        T_cw_list=data["T_cw_list"],
    )

    trans_err, rot_err = _se3_diff(out["X_refined"], _X_TRUE)
    assert out["rmse_pixels"] < 5.0, f"rmse {out['rmse_pixels']:.3g}px indicates divergence"
    assert trans_err < 25.0, f"translation error {trans_err:.1f} mm too high"
    assert rot_err < 2.0, f"rotation error {rot_err:.2f} deg too high"


def test_unscaled_solver_would_diverge_on_this_data():
    """Guard test: documents that WITHOUT x_scale the same data diverges.

    Re-runs the optimizer on a known-bad seed without per-variable scaling and
    asserts it blows up. If a future refactor drops ``x_scale`` from
    ``refine_handeye``, the recovery test above protects the fix; this test
    pins down *why* the fix is needed so it isn't silently removed.
    """
    from scipy.optimize import least_squares
    from solvers.reprojection_solver import (
        _residual_fn,
        _params_from_se3,
        _se3_from_params,
        bootstrap_T_bw,
    )

    data = _build_charuco_like_dataset(seed=3)
    X_init = _bootstrap_X(data)
    Y_init = bootstrap_T_bw(data["T_be_list"][0], X_init, data["T_cw_list"][0])

    x0 = np.concatenate([_params_from_se3(X_init), _params_from_se3(Y_init)])
    result = least_squares(
        fun=_residual_fn,
        x0=x0,
        args=(
            data["T_be_list"],
            [np.asarray(c, dtype=np.float64) for c in data["corners_2d_list"]],
            [np.asarray(c, dtype=np.float64) for c in data["corners_3d_list"]],
            _K,
            _DIST,
        ),
        method="trf",
        loss="huber",
        f_scale=1.0,
        max_nfev=200,
        # NB: no x_scale -- the unconditioned configuration.
    )
    X_unscaled = _se3_from_params(result.x[:6])
    trans_err, _ = _se3_diff(X_unscaled, _X_TRUE)
    rmse = float(np.sqrt(np.mean(result.fun ** 2)))
    assert trans_err > 100.0 or rmse > 50.0, (
        "expected the unscaled solver to diverge on this data; if it now "
        "converges, the conditioning regression scenario may need to be revisited"
    )
