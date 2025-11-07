import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from viam.proto.common import Pose

try:
    from utils.utils import call_go_mat2ov, call_go_ov2mat, call_visualize_pose
except ModuleNotFoundError:
    from ..utils.utils import call_go_mat2ov, call_go_ov2mat, call_visualize_pose


def get_calibration_values_from_chessboard_from_image(
    image,
    chessboard_pattern_size,
    chessboard_square_size,
    camera_matrix,
    dist_coeffs,
):
    """
    Finds chessboard corners in an image and calculates rotation and translation vectors.
    This is an adaptation of HandEyeCalibration.get_calibration_values_from_chessboard.
    """
    pattern_size = (
        int(chessboard_pattern_size[0]),
        int(chessboard_pattern_size[1]),
    )

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size, cv2.CALIB_CB_FAST_CHECK
    )

    if not ret:
        print("No corners found")
        return False, None, None, None

    # Create the object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp = objp * chessboard_square_size

    # Refine the corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Find the rotation and translation vectors.
    (
        success,
        rvec,
        tvec,
    ) = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
    if not success:
        return False, None, None, None

    # Calculate reprojection error
    reprojected_points, _ = cv2.projectPoints(
        objp, rvec, tvec, camera_matrix, dist_coeffs
    )
    error = cv2.norm(corners2, reprojected_points, cv2.NORM_L2) / len(
        reprojected_points
    )

    return True, rvec, tvec, error


def get_pose_transformations(
    pose_info,
    pose_index,
    data_directory,
    camera_matrix,
    dist_coeffs,
    chessboard_pattern_size,
    chessboard_square_size,
    wrist_correction_degrees=0.0,
):
    image_filename = f"image_pose_{pose_index + 1}.jpg"
    image_path = os.path.join(data_directory, "pose_images", image_filename)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    print(f"Processing pose {pose_index + 1} with image {image_filename}...")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    (
        success,
        rvec,
        tvec,
        reprojection_error,
    ) = get_calibration_values_from_chessboard_from_image(
        image,
        chessboard_pattern_size,
        chessboard_square_size,
        camera_matrix,
        dist_coeffs,
    )

    if not success:
        print(f"  Could not find chessboard in pose {pose_index + 1}")
        return None

    print(f"  Successfully processed pose {pose_index + 1}")
    print(f"    Reprojection Error: {reprojection_error:.4f} pixels")

    R_board_to_cam, _ = cv2.Rodrigues(rvec)

    # Apply wrist rotation correction if specified
    if wrist_correction_degrees != 0.0:
        angle_rad = np.radians(wrist_correction_degrees)
        R_correction = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        R_board_to_cam = R_correction @ R_board_to_cam
        tvec = R_correction @ tvec

    raw_pose = pose_info["pose_spec"]
    temp_pose = Pose(
        x=raw_pose["x"],
        y=raw_pose["y"],
        z=raw_pose["z"],
        o_x=raw_pose["o_x"],
        o_y=raw_pose["o_y"],
        o_z=raw_pose["o_z"],
        theta=raw_pose["theta"],
    )

    R_base_to_gripper_raw = call_go_ov2mat(
        temp_pose.o_x,
        temp_pose.o_y,
        temp_pose.o_z,
        temp_pose.theta,
    )
    R_gripper_to_base = R_base_to_gripper_raw.T
    t_gripper_to_base = np.array([[temp_pose.x], [temp_pose.y], [temp_pose.z]])

    orientation_gripper_to_base = call_go_mat2ov(R_gripper_to_base)
    if orientation_gripper_to_base:
        ox, oy, oz, theta = orientation_gripper_to_base
        print("    Viam Pose (Gripper to Base):")
        print(
            f"      Translation: x={t_gripper_to_base[0][0]:.4f}, y={t_gripper_to_base[1][0]:.4f}, z={t_gripper_to_base[2][0]:.4f}"
        )
        print(
            f"      Orientation (ov): ox={ox:.4f}, oy={oy:.4f}, oz={oz:.4f}, theta={theta:.4f}"
        )

    orientation_board_to_cam = call_go_mat2ov(R_board_to_cam)
    if orientation_board_to_cam:
        ox, oy, oz, theta = orientation_board_to_cam
        print("    Viam Pose (Board to Cam):")
        print(f"      Translation: x={tvec[0][0]:.4f}, y={tvec[1][0]:.4f}, z={tvec[2][0]:.4f}")
        print(
            f"      Orientation (ov): ox={ox:.4f}, oy={oy:.4f}, oz={oz:.4f}, theta={theta:.4f}"
        )
        if wrist_correction_degrees != 0.0:
            print(f"      (Applied {wrist_correction_degrees}° wrist correction)")

    return {
        "reprojection_error": reprojection_error,
        "R_gripper_to_base": R_gripper_to_base,
        "t_gripper_to_base": t_gripper_to_base,
        "R_board_to_cam": R_board_to_cam,
        "t_board_to_cam": tvec,
    }


def perform_and_print_calibration(
    R_gripper_to_base_list,
    t_gripper_to_base_list,
    R_board_to_cam_list,
    t_board_to_cam_list,
    label="",
):
    print_label = f" for {label}" if label else ""
    print(f"\n--- Performing Hand-Eye Calibration{print_label} ---")

    try:
        R_cam_to_gripper, t_cam_to_gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper_to_base_list,
            t_gripper2base=t_gripper_to_base_list,
            R_target2cam=R_board_to_cam_list,
            t_target2cam=t_board_to_cam_list,
            method=cv2.CALIB_HAND_EYE_PARK,
        )

        R_gripper2cam = R_cam_to_gripper.T
        orientation_result = call_go_mat2ov(R_gripper2cam)
        if orientation_result is None:
            raise Exception("failed to convert rotation matrix to orientation vector")

        ox, oy, oz, theta = orientation_result
        x, y, z = (
            float(t_cam_to_gripper[0][0]),
            float(t_cam_to_gripper[1][0]),
            float(t_cam_to_gripper[2][0]),
        )

        print(
            json.dumps(
                {
                    "frame": {
                        "translation": {"x": x, "y": y, "z": z},
                        "orientation": {
                            "type": "ov_degrees",
                            "value": {"x": ox, "y": oy, "z": oz, "th": theta},
                        },
                    }
                },
                indent=2,
            )
        )

        # Visualize pose
        call_visualize_pose(ox=ox, oy=oy, oz=oz, theta=theta)
    except Exception as e:
        print(f"Hand-eye calibration failed{print_label}: {e}")


def run_dry_calibration(data_directory: str, wrist_correction_degrees: float = 0.0):
    config_path = os.path.join(data_directory, "calibration_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    camera_matrix = np.array(config["camera_matrix"])
    dist_coeffs = np.array(config["dist_coeffs"])
    chessboard_pattern_size = config["chessboard_size"]
    chessboard_square_size = config["square_size"]

    pose_data_path = os.path.join(data_directory, "pose_data.json")
    with open(pose_data_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    if wrist_correction_degrees != 0.0:
        print(f"\n{'='*80}")
        print(f"APPLYING WRIST CORRECTION: {wrist_correction_degrees}°")
        print(f"{'='*80}\n")

    R_gripper_to_base_list, t_gripper_to_base_list = [], []
    R_board_to_cam_list, t_board_to_cam_list = [], []
    reprojection_errors = []

    for i, pose_info in enumerate(pose_data["poses"]):
        result = get_pose_transformations(
            pose_info,
            i,
            data_directory,
            camera_matrix,
            dist_coeffs,
            chessboard_pattern_size,
            chessboard_square_size,
            wrist_correction_degrees,
        )
        if result:
            reprojection_errors.append(result["reprojection_error"])
            R_gripper_to_base_list.append(result["R_gripper_to_base"])
            t_gripper_to_base_list.append(result["t_gripper_to_base"])
            R_board_to_cam_list.append(result["R_board_to_cam"])
            t_board_to_cam_list.append(result["t_board_to_cam"])

            if len(R_gripper_to_base_list) >= 3:
                label = f"with {len(R_gripper_to_base_list)} poses (up to pose index {i + 1})"
                perform_and_print_calibration(
                    R_gripper_to_base_list,
                    t_gripper_to_base_list,
                    R_board_to_cam_list,
                    t_board_to_cam_list,
                    label=label,
                )

    if reprojection_errors:
        print(
            f"\nAverage Reprojection Error: {np.mean(reprojection_errors):.4f} pixels"
        )


def run_side_by_side_calibration(data_directory1: str, data_directory2: str, wrist_correction_degrees: float = 0.0):
    config_path1 = os.path.join(data_directory1, "calibration_config.json")
    with open(config_path1, "r", encoding="utf-8") as f:
        config1 = json.load(f)
    pose_data_path1 = os.path.join(data_directory1, "pose_data.json")
    with open(pose_data_path1, "r", encoding="utf-8") as f:
        pose_data1 = json.load(f)

    config_path2 = os.path.join(data_directory2, "calibration_config.json")
    with open(config_path2, "r", encoding="utf-8") as f:
        config2 = json.load(f)
    pose_data_path2 = os.path.join(data_directory2, "pose_data.json")
    with open(pose_data_path2, "r", encoding="utf-8") as f:
        pose_data2 = json.load(f)

    if wrist_correction_degrees != 0.0:
        print(f"\n{'='*80}")
        print(f"APPLYING WRIST CORRECTION TO {data_directory1}: {wrist_correction_degrees}°")
        print(f"{'='*80}\n")

    params1 = {
        "camera_matrix": np.array(config1["camera_matrix"]),
        "dist_coeffs": np.array(config1["dist_coeffs"]),
        "chessboard_pattern_size": config1["chessboard_size"],
        "chessboard_square_size": config1["square_size"],
    }
    params2 = {
        "camera_matrix": np.array(config2["camera_matrix"]),
        "dist_coeffs": np.array(config2["dist_coeffs"]),
        "chessboard_pattern_size": config2["chessboard_size"],
        "chessboard_square_size": config2["square_size"],
    }

    lists1 = {
        "R_gripper_to_base": [],
        "t_gripper_to_base": [],
        "R_board_to_cam": [],
        "t_board_to_cam": [],
        "reprojection_errors": [],
    }
    lists2 = {
        "R_gripper_to_base": [],
        "t_gripper_to_base": [],
        "R_board_to_cam": [],
        "t_board_to_cam": [],
        "reprojection_errors": [],
    }

    num_poses = min(len(pose_data1["poses"]), len(pose_data2["poses"]))
    for i in range(num_poses):
        pose_info1 = pose_data1["poses"][i]
        pose_info2 = pose_data2["poses"][i]

        print(f"\n----- POSE INDEX {i + 1} -----")

        print(f"\n--- Processing {data_directory1} ---")
        result1 = get_pose_transformations(
            pose_info1,
            i,
            data_directory1,
            params1["camera_matrix"],
            params1["dist_coeffs"],
            params1["chessboard_pattern_size"],
            params1["chessboard_square_size"],
            wrist_correction_degrees,
        )
        if result1:
            lists1["reprojection_errors"].append(result1["reprojection_error"])
            lists1["R_gripper_to_base"].append(result1["R_gripper_to_base"])
            lists1["t_gripper_to_base"].append(result1["t_gripper_to_base"])
            lists1["R_board_to_cam"].append(result1["R_board_to_cam"])
            lists1["t_board_to_cam"].append(result1["t_board_to_cam"])

        print(f"\n--- Processing {data_directory2} ---")
        result2 = get_pose_transformations(
            pose_info2,
            i,
            data_directory2,
            params2["camera_matrix"],
            params2["dist_coeffs"],
            params2["chessboard_pattern_size"],
            params2["chessboard_square_size"],
        )
        if result2:
            lists2["reprojection_errors"].append(result2["reprojection_error"])
            lists2["R_gripper_to_base"].append(result2["R_gripper_to_base"])
            lists2["t_gripper_to_base"].append(result2["t_gripper_to_base"])
            lists2["R_board_to_cam"].append(result2["R_board_to_cam"])
            lists2["t_board_to_cam"].append(result2["t_board_to_cam"])

    for dir_name, data_lists in [(data_directory1, lists1), (data_directory2, lists2)]:
        if len(data_lists["R_gripper_to_base"]) < 3:
            print(f"Not enough successful poses for {dir_name} to perform calibration.")
            continue

        if data_lists["reprojection_errors"]:
            avg_error = np.mean(data_lists["reprojection_errors"])
            print(
                f"\nAverage Reprojection Error for {dir_name}: {avg_error:.4f} pixels"
            )

        print(
            f"\nPerforming hand-eye calibration for {dir_name} with {len(data_lists['R_gripper_to_base'])} poses..."
        )
        perform_and_print_calibration(
            data_lists["R_gripper_to_base"],
            data_lists["t_gripper_to_base"],
            data_lists["R_board_to_cam"],
            data_lists["t_board_to_cam"],
            label=dir_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hand-eye calibration dry run script."
    )
    parser.add_argument(
        "--data_directories",
        nargs="+",
        type=str,
        required=True,
        help="Path(s) to the data directory(ies) (e.g., nj1 or nj1 nj2).",
    )
    parser.add_argument(
        "--wrist_correction",
        type=float,
        default=0.0,
        help="Wrist rotation correction in degrees to apply to board-to-camera transforms. "
             "Use -26.5 for nj1-hardtop to simulate correct end effector mounting. "
             "This compensates for end effector mounting errors by rotating the observed "
             "chessboard poses around the camera Z-axis.",
    )
    args = parser.parse_args()

    if len(args.data_directories) == 1:
        run_dry_calibration(args.data_directories[0], args.wrist_correction)
    elif len(args.data_directories) == 2:
        run_side_by_side_calibration(
            args.data_directories[0], args.data_directories[1], args.wrist_correction
        )
    else:
        print("Please provide 1 or 2 data directories.")
        parser.print_help()
