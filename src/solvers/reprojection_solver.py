"""Reprojection-error refinement for hand-eye calibration.

Given an initial estimate of the camera-in-end-effector transform X (from
``cv2.calibrateHandEye`` or similar), refines X by minimizing per-corner pixel
reprojection error across all stations using ``scipy.optimize.least_squares``.

Eye-in-hand model. The board is fixed in the world. For each station k and
each board corner i,

    P_c,k,i = T_ce · T_eb,k · T_bw · P_w,i
    u_pred  = π(K, dist, P_c,k,i)
    residual = u_observed - u_pred

Optimization variables: T_ec (camera in end-effector, 6 DoF) and T_bw (board
in robot base, 6 DoF). Both parameterized in se(3) as (rotvec, translation).

Why reprojection refinement? OpenCV's analytical hand-eye solvers minimize an
algebraic error on relative-pose pairs, which is statistically the wrong cost
function and leaves accuracy on the table when good intrinsics are available
and a planar target is in view. The reprojection cost uses every detected
corner directly. As a side benefit it produces a Jacobian — useful for later
phases that pick poses to maximize information gain.
"""

from typing import List, Optional, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _se3_inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _se3_from_params(params: np.ndarray) -> np.ndarray:
    rvec = params[:3]
    t = params[3:6]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    T[:3, 3] = t
    return T


def _params_from_se3(T: np.ndarray) -> np.ndarray:
    rvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([rvec, T[:3, 3]])


def _project(P_w: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(P_w, rvec, t.reshape(3, 1), K, dist)
    return projected.reshape(-1, 2)


def _residual_fn(
    params: np.ndarray,
    T_be_list: List[np.ndarray],
    corners_2d_list: List[np.ndarray],
    corners_3d: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    T_ec = _se3_from_params(params[:6])
    T_bw = _se3_from_params(params[6:12])
    T_ce = _se3_inverse(T_ec)

    residuals: List[np.ndarray] = []
    for T_be, corners_obs in zip(T_be_list, corners_2d_list):
        T_eb = _se3_inverse(T_be)
        M = T_ce @ T_eb @ T_bw
        u_pred = _project(corners_3d, M[:3, :3], M[:3, 3], K, dist)
        residuals.append((corners_obs - u_pred).reshape(-1))
    return np.concatenate(residuals)


def bootstrap_T_bw(
    T_be_0: np.ndarray,
    T_ec_init: np.ndarray,
    T_cw_0: np.ndarray,
) -> np.ndarray:
    """Initial estimate of board-in-base from station 0.

    T_cw,0 (board pose in camera at station 0) comes from the per-station
    PnP solve. Using the chain T_cw,0 = T_ce · T_eb,0 · T_bw,
    T_bw = T_be,0 · T_ec · T_cw,0.
    """
    return T_be_0 @ T_ec_init @ T_cw_0


def refine_handeye(
    T_be_list: Sequence[np.ndarray],
    corners_2d_list: Sequence[np.ndarray],
    corners_3d: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    X_init: np.ndarray,
    Y_init: Optional[np.ndarray] = None,
    T_cw_list: Optional[Sequence[np.ndarray]] = None,
    huber_pixels: float = 1.0,
    max_nfev: int = 200,
) -> dict:
    """Refine X (= T_ec) by minimizing reprojection error.

    Args:
        T_be_list: list of 4x4 end-effector-in-base transforms, one per station.
        corners_2d_list: list of (N, 2) detected corner pixel arrays, one per
            station. N must be the same across stations and match corners_3d.
        corners_3d: (N, 3) board corner positions in board frame.
        K: (3, 3) camera intrinsic matrix.
        dist: (5,) distortion coefficients in the (k1, k2, p1, p2, k3) order
            expected by ``cv2.projectPoints``.
        X_init: (4, 4) initial estimate of T_ec from cv2.calibrateHandEye.
        Y_init: (4, 4) initial estimate of T_bw. If None, bootstrapped from
            station 0 using T_cw_list[0]. One of (Y_init, T_cw_list) must be
            provided.
        T_cw_list: list of (4, 4) board-in-camera transforms (one per station)
            from per-station PnP solves. Used only to bootstrap Y_init if not
            provided.
        huber_pixels: f_scale for the Huber loss; pixel residuals below this
            are treated as inliers in the linear regime.
        max_nfev: maximum function evaluations for the optimizer.

    Returns:
        dict with X_refined, Y_refined, rmse_pixels, per_pose_rmse_pixels,
        residuals, jacobian, success, n_iterations, message.
    """
    n_stations = len(T_be_list)
    if n_stations < 3:
        raise ValueError(f"need at least 3 stations, got {n_stations}")
    if len(corners_2d_list) != n_stations:
        raise ValueError("T_be_list and corners_2d_list must be the same length")

    T_be_arr = [np.asarray(T, dtype=np.float64) for T in T_be_list]
    corners_2d_arr = [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in corners_2d_list]
    corners_3d_arr = np.asarray(corners_3d, dtype=np.float64).reshape(-1, 3)
    K_arr = np.asarray(K, dtype=np.float64)
    dist_arr = np.asarray(dist, dtype=np.float64).reshape(-1)

    n_corners = corners_3d_arr.shape[0]
    for i, c in enumerate(corners_2d_arr):
        if c.shape[0] != n_corners:
            raise ValueError(
                f"station {i} has {c.shape[0]} corners but corners_3d has {n_corners}"
            )

    if Y_init is None:
        if T_cw_list is None:
            raise ValueError("must provide either Y_init or T_cw_list to bootstrap Y")
        Y_init = bootstrap_T_bw(T_be_arr[0], np.asarray(X_init, dtype=np.float64), np.asarray(T_cw_list[0], dtype=np.float64))

    x0 = np.concatenate([
        _params_from_se3(np.asarray(X_init, dtype=np.float64)),
        _params_from_se3(np.asarray(Y_init, dtype=np.float64)),
    ])

    result = least_squares(
        fun=_residual_fn,
        x0=x0,
        args=(T_be_arr, corners_2d_arr, corners_3d_arr, K_arr, dist_arr),
        method="trf",
        loss="huber",
        f_scale=huber_pixels,
        max_nfev=max_nfev,
    )

    X_refined = _se3_from_params(result.x[:6])
    Y_refined = _se3_from_params(result.x[6:12])

    residuals = result.fun
    rmse_pixels = float(np.sqrt(np.mean(residuals ** 2)))

    per_pose_rmse: List[float] = []
    residuals_per_station = residuals.reshape(n_stations, n_corners, 2)
    for k in range(n_stations):
        per_pose_rmse.append(float(np.sqrt(np.mean(residuals_per_station[k] ** 2))))

    return {
        "X_refined": X_refined,
        "Y_refined": Y_refined,
        "rmse_pixels": rmse_pixels,
        "per_pose_rmse_pixels": per_pose_rmse,
        "residuals": residuals,
        "jacobian": np.asarray(result.jac),
        "success": bool(result.success),
        "n_iterations": int(result.nfev),
        "message": str(result.message),
    }
