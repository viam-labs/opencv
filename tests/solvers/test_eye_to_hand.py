"""Eye-to-hand round-trip tests for the hand-eye calibration math.

Scenario: a static camera watches a ChArUco/chessboard target rigidly held in
the gripper. The service solves this by feeding INVERTED arm poses
(base-in-gripper) into the same eye-in-hand machinery; the solved X is then
camera-in-base and the refined Y is board-in-gripper.

These tests build a synthetic ground-truth scene (known camera-in-base and
board-in-gripper), generate stations and projected corners, and verify that
the exact call sequence `run_calibration` performs — cv2.calibrateHandEye on
inverted poses to bootstrap, then refine_handeye on inverted poses — recovers
the ground truth. Partial per-station corner subsets mimic gripper occlusion.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from solvers.reprojection_solver import _se3_inverse, refine_handeye


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


def _camera_in_base():
    """Camera on a tripod at [700, 0, 300] mm, optical axis looking along
    base -X toward the workspace. Columns are the camera axes in base:
    x_cam = +Y_base (image right), y_cam = -Z_base (image down, OpenCV
    convention), z_cam = -X_base (view direction). Right-handed:
    x_cam × y_cam = Y × -Z = -X = z_cam."""
    R_bc = np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    return _se3(R_bc, np.array([700.0, 0.0, 300.0]))


def _project_board(corners_3d, T_c_board, K, dist):
    rvec, _ = cv2.Rodrigues(T_c_board[:3, :3])
    proj, _ = cv2.projectPoints(corners_3d, rvec, T_c_board[:3, 3].reshape(3, 1), K, dist)
    return proj.reshape(-1, 2)


def _build_eye_to_hand_dataset(noise_pixels=0.0, seed=0):
    """Returns dict with T_be_list (gripper-in-base), corners, intrinsics,
    T_cw_list (board-in-camera PnP poses), and ground truth T_bc_true
    (camera-in-base) / T_eboard_true (board-in-gripper)."""
    rng = np.random.default_rng(seed)
    K, dist = _camera_intrinsics()
    corners_3d = _board_corners()

    T_bc_true = _camera_in_base()
    # Board bolted 60 mm past the flange, slightly askew — exactly the
    # unknown, constant grasp transform of a gripper-held board. The ~180°
    # y-rotation puts the printed face (board -Z side) toward the camera:
    # OpenCV's planar-PnP convention has the board +Z pointing away from
    # the viewer, and IPPE only represents poses on that side.
    T_eboard_true = _euler_xyz_se3(3.0, 178.0, 8.0, [-75.0, -50.0, 60.0])

    # Stations: gripper near [300, 0, 300] with tool +Z aimed roughly along
    # base +X so the held board faces the camera, with orientation diversity
    # about all three axes (rotation-axis diversity is what conditions the
    # hand-eye problem).
    rotation_specs = [
        (0.0, 90.0, 0.0),
        (30.0, 70.0, 10.0),
        (-28.0, 105.0, -20.0),
        (15.0, 120.0, 35.0),
        (-20.0, 65.0, 40.0),
        (35.0, 95.0, -30.0),
        (-15.0, 75.0, -45.0),
        (25.0, 110.0, 55.0),
    ]
    translation_specs = [
        [300.0, 0.0, 300.0],
        [320.0, 30.0, 280.0],
        [280.0, -30.0, 320.0],
        [310.0, 20.0, 340.0],
        [290.0, -20.0, 260.0],
        [330.0, 10.0, 310.0],
        [270.0, 40.0, 290.0],
        [305.0, -40.0, 330.0],
    ]

    T_be_list = []
    corners_2d_list = []
    T_cw_list = []
    T_cb_true = _se3_inverse(T_bc_true)
    for rs, ts in zip(rotation_specs, translation_specs):
        T_be = _euler_xyz_se3(*rs, ts)
        T_be_list.append(T_be)

        # Board-in-camera: T_c_board = T_cb · T_be · T_eboard
        T_c_board = T_cb_true @ T_be @ T_eboard_true
        # Every corner must be in front of the static camera.
        pts_cam = (T_c_board[:3, :3] @ corners_3d.T).T + T_c_board[:3, 3]
        assert np.all(pts_cam[:, 2] > 0), "synthetic geometry: board behind camera"

        u = _project_board(corners_3d, T_c_board, K, dist)
        if noise_pixels > 0:
            u = u + rng.normal(0, noise_pixels, size=u.shape)
        corners_2d_list.append(u)

        # Per-station PnP exactly as the charuco tracker does it: IPPE with
        # the lowest-reprojection-error branch (planar two-fold ambiguity).
        n_sol, rvecs, tvecs, _errs = cv2.solvePnPGeneric(
            corners_3d,
            u.reshape(-1, 1, 2).astype(np.float64),
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE,
        )
        assert n_sol >= 1
        R_cw, _ = cv2.Rodrigues(rvecs[0])
        T_cw_list.append(_se3(R_cw, tvecs[0].reshape(3)))

    return {
        "T_be_list": T_be_list,
        "corners_2d_list": corners_2d_list,
        "corners_3d": corners_3d,
        "K": K,
        "dist": dist,
        "T_bc_true": T_bc_true,
        "T_eboard_true": T_eboard_true,
        "T_cw_list": T_cw_list,
    }


def _se3_diff(T_a, T_b):
    delta_R = T_a[:3, :3] @ T_b[:3, :3].T
    cos_angle = np.clip((np.trace(delta_R) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(cos_angle)))
    trans_mm = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    return trans_mm, rot_deg


def _bootstrap_camera_in_base(data):
    """The service's opencv-solver path for eye-to-hand: calibrateHandEye on
    inverted arm poses. Output is camera-in-base."""
    T_arm_list = [_se3_inverse(T) for T in data["T_be_list"]]
    R_g2b = [T[:3, :3] for T in T_arm_list]
    t_g2b = [T[:3, 3].reshape(3, 1) for T in T_arm_list]
    R_t2c = [T[:3, :3] for T in data["T_cw_list"]]
    t_t2c = [T[:3, 3].reshape(3, 1) for T in data["T_cw_list"]]
    R, t = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI
    )
    return _se3(R, t.flatten()), T_arm_list


# ---- tests ----


def test_opencv_bootstrap_recovers_camera_in_base():
    data = _build_eye_to_hand_dataset(noise_pixels=0.0)
    X_boot, _ = _bootstrap_camera_in_base(data)
    trans_err, rot_err = _se3_diff(X_boot, data["T_bc_true"])
    assert trans_err < 0.5, f"translation error {trans_err:.4f} mm too high"
    assert rot_err < 0.05, f"rotation error {rot_err:.4f} deg too high"


def test_refinement_recovers_camera_in_base_and_board_in_gripper():
    data = _build_eye_to_hand_dataset(noise_pixels=0.0)
    X_boot, T_arm_list = _bootstrap_camera_in_base(data)
    out = refine_handeye(
        T_arm_list,
        data["corners_2d_list"],
        data["corners_3d"],
        data["K"],
        data["dist"],
        X_init=X_boot,
        T_cw_list=data["T_cw_list"],
    )
    assert out["success"]
    assert out["rmse_pixels"] < 0.5

    trans_err, rot_err = _se3_diff(out["X_refined"], data["T_bc_true"])
    assert trans_err < 0.1, f"camera-in-base translation error {trans_err:.4f} mm"
    assert rot_err < 0.05, f"camera-in-base rotation error {rot_err:.4f} deg"

    trans_err, rot_err = _se3_diff(out["Y_refined"], data["T_eboard_true"])
    assert trans_err < 0.1, f"board-in-gripper translation error {trans_err:.4f} mm"
    assert rot_err < 0.05, f"board-in-gripper rotation error {rot_err:.4f} deg"


def test_partial_views_gripper_occlusion():
    """Each station sees a different subset of corners, as when the gripper
    hides part of the ChArUco board. Recovery should still hold."""
    data = _build_eye_to_hand_dataset(noise_pixels=0.1, seed=3)
    rng = np.random.default_rng(7)

    n_corners = data["corners_3d"].shape[0]
    corners_2d_subsets = []
    corners_3d_subsets = []
    for u in data["corners_2d_list"]:
        keep = rng.choice(n_corners, size=int(n_corners * 0.6), replace=False)
        keep.sort()
        corners_2d_subsets.append(u[keep])
        corners_3d_subsets.append(data["corners_3d"][keep])

    X_boot, T_arm_list = _bootstrap_camera_in_base(data)
    out = refine_handeye(
        T_arm_list,
        corners_2d_subsets,
        corners_3d_subsets,
        data["K"],
        data["dist"],
        X_init=X_boot,
        T_cw_list=data["T_cw_list"],
    )
    assert out["success"]

    trans_err, rot_err = _se3_diff(out["X_refined"], data["T_bc_true"])
    assert trans_err < 1.0, f"camera-in-base translation error {trans_err:.4f} mm"
    assert rot_err < 0.1, f"camera-in-base rotation error {rot_err:.4f} deg"


def test_target_in_gripper_invariant():
    """The residual diagnostic's eye-to-hand invariant: with inverted arm
    poses and the solved camera-in-base, the product in
    _compute_per_pose_residuals is target-in-gripper — constant across
    stations, equal to the true grasp transform."""
    data = _build_eye_to_hand_dataset(noise_pixels=0.0)
    X_boot, T_arm_list = _bootstrap_camera_in_base(data)

    for T_arm, T_cw in zip(T_arm_list, data["T_cw_list"]):
        # Mirrors R_g2b @ R_c2g @ R_t2c with the eye-to-hand substitution.
        T_target_in_gripper = T_arm @ X_boot @ T_cw
        trans_err, rot_err = _se3_diff(T_target_in_gripper, data["T_eboard_true"])
        assert trans_err < 0.5
        assert rot_err < 0.05
