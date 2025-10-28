#!/usr/bin/env python3

import argparse
import asyncio
import os
import numpy as np
import cv2
import json
from datetime import datetime
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.app.app_client import AppClient
from viam.components.arm import Arm
from viam.components.pose_tracker import PoseTracker
from viam.services.motion import MotionClient
from viam.proto.common import PoseInFrame, Pose
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import CameraMimeType
from viam.components.camera import Camera
from viam.rpc.dial import DialOptions
from viam.app.viam_client import ViamClient
from typing import Optional

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils.utils import call_go_ov2mat, call_go_mat2ov
    from utils.chessboard_utils import (
        generate_object_points,
    )
except ModuleNotFoundError:
    from ..utils.utils import call_go_ov2mat, call_go_mat2ov
    from ..utils.chessboard_utils import (
        generate_object_points,
    )

DEFAULT_WORLD_FRAME = "world"
DEFAULT_VELOCITY_NORMAL = 25
DEFAULT_VELOCITY_SLOW = 10
DEFAULT_SETTLE_TIME = 5.0

import cv2  # Make sure cv2 is imported

def rotation_error(R1, R2):
    """Compute rotation error in degrees between two rotation matrices."""
    R_error = R1.T @ R2
    angle_rad = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1.0, 1.0))
    return np.degrees(angle_rad)

def analyze_hand_eye_error(T_hand_eye, T_delta_A_world_frame, T_delta_B_camera_frame, 
                          A_0_pose, A_i_pose, pose_num):
    """
    Detailed analysis of hand-eye verification errors.
    """
    print(f"\n{'='*60}")
    print(f"DETAILED ERROR ANALYSIS - POSE {pose_num}")
    print(f"{'='*60}")
    
    # Extract components
    R_X = T_hand_eye[:3, :3]
    t_X = T_hand_eye[:3, 3]
    
    R_A = T_delta_A_world_frame[:3, :3]
    t_A = T_delta_A_world_frame[:3, 3]
    
    R_B = T_delta_B_camera_frame[:3, :3]
    t_B = T_delta_B_camera_frame[:3, 3]
    
    print(f"\n📊 MOTION MAGNITUDES:")
    print(f"  Arm translation:   {np.linalg.norm(t_A):8.2f} mm")
    print(f"  Board translation: {np.linalg.norm(t_B):8.2f} mm")
    print(f"  Ratio (arm/board): {np.linalg.norm(t_A)/np.linalg.norm(t_B):.3f}x")
    
    # Rotation magnitudes
    angle_A = np.degrees(np.arccos(np.clip((np.trace(R_A) - 1) / 2, -1, 1)))
    angle_B = np.degrees(np.arccos(np.clip((np.trace(R_B) - 1) / 2, -1, 1)))
    print(f"  Arm rotation:      {angle_A:8.2f}°")
    print(f"  Board rotation:    {angle_B:8.2f}°")
    
    print(f"\n🔍 HAND-EYE TRANSFORM:")
    print(f"  Translation: [{t_X[0]:7.2f}, {t_X[1]:7.2f}, {t_X[2]:7.2f}] mm")
    angle_X = np.degrees(np.arccos(np.clip((np.trace(R_X) - 1) / 2, -1, 1)))
    print(f"  Rotation magnitude: {angle_X:.2f}°")
    
    # Check rotation axis of hand-eye
    if angle_X > 1.0:  # Only compute axis if rotation is significant
        # Convert rotation matrix to axis-angle
        axis_angle = cv2.Rodrigues(R_X)[0]
        axis = axis_angle.flatten() / (np.linalg.norm(axis_angle) + 1e-10)
        print(f"  Rotation axis: [{axis[0]:6.3f}, {axis[1]:6.3f}, {axis[2]:6.3f}]")
    
    print(f"\n🧮 VERIFICATION EQUATION TESTING:")
    
    # Method 1: Standard X⁻¹BX (your current method)
    X_inv = np.linalg.inv(T_hand_eye)
    T_predicted_1 = X_inv @ T_delta_B_camera_frame @ T_hand_eye
    error_1_rot = rotation_error(T_predicted_1[:3,:3], R_A)
    error_1_trans = np.linalg.norm(T_predicted_1[:3,3] - t_A)
    
    print(f"  Method 1 (X⁻¹BX):   rot={error_1_rot:.3f}°, trans={error_1_trans:.2f}mm")
    
    # Method 2: Try XBX⁻¹
    T_predicted_2 = T_hand_eye @ T_delta_B_camera_frame @ X_inv
    error_2_rot = rotation_error(T_predicted_2[:3,:3], R_A)
    error_2_trans = np.linalg.norm(T_predicted_2[:3,3] - t_A)
    
    print(f"  Method 2 (XBX⁻¹):   rot={error_2_rot:.3f}°, trans={error_2_trans:.2f}mm")
    
    # Method 3: Try XA⁻¹X⁻¹ (predicting B from A)
    T_A_inv = np.linalg.inv(T_delta_A_world_frame)
    T_predicted_3 = T_hand_eye @ T_A_inv @ X_inv
    error_3_rot = rotation_error(T_predicted_3[:3,:3], R_B)
    error_3_trans = np.linalg.norm(T_predicted_3[:3,3] - t_B)
    
    print(f"  Method 3 (XA⁻¹X⁻¹): rot={error_3_rot:.3f}°, trans={error_3_trans:.2f}mm (predicting B)")
    
    # Method 4: Try X⁻¹AX (predicting B from A)
    T_predicted_4 = X_inv @ T_delta_A_world_frame @ T_hand_eye
    error_4_rot = rotation_error(T_predicted_4[:3,:3], R_B)
    error_4_trans = np.linalg.norm(T_predicted_4[:3,3] - t_B)
    
    print(f"  Method 4 (X⁻¹AX):   rot={error_4_rot:.3f}°, trans={error_4_trans:.2f}mm (predicting B)")
    
    print(f"\n🎯 BEST METHOD: ", end="")
    errors = [
        (1, error_1_rot, error_1_trans, "X⁻¹BX (current)"),
        (2, error_2_rot, error_2_trans, "XBX⁻¹"),
        (3, error_3_rot, error_3_trans, "XA⁻¹X⁻¹ (inverse direction)"),
        (4, error_4_rot, error_4_trans, "X⁻¹AX (inverse direction)")
    ]
    best = min(errors, key=lambda x: x[1] + x[2]/10)  # Weight rotation more
    print(f"Method {best[0]} - {best[3]}")
    print(f"           rot={best[1]:.3f}°, trans={best[2]:.2f}mm")
    
    # Check for scale issues
    print(f"\n⚠️  DIAGNOSTIC CHECKS:")
    
    # Translation scale mismatch
    ratio = np.linalg.norm(t_A) / (np.linalg.norm(t_B) + 1e-10)
    if ratio < 0.5 or ratio > 2.0:
        print(f"  ❌ Motion scale mismatch: arm/board = {ratio:.2f}x")
        print(f"     Expected: ratio should be 0.5-2.0x")
        print(f"     Possible causes:")
        print(f"       - Hand-eye calibration is incorrect")
        print(f"       - Units mismatch (mm vs m)")
        print(f"       - Wrong transformation chain")
    else:
        print(f"  ✅ Motion scale OK: arm/board = {ratio:.2f}x")
    
    # Check if rotations are similar
    if abs(angle_A - angle_B) > 10.0:
        print(f"  ⚠️  Rotation mismatch: arm={angle_A:.1f}° vs board={angle_B:.1f}°")
    else:
        print(f"  ✅ Rotation magnitudes match: ~{angle_A:.1f}°")
    
    # Check hand-eye rotation magnitude
    if 170 < angle_X < 190:
        print(f"  ℹ️  Hand-eye has ~180° rotation (camera mounted upside-down/backwards)")
    
    print(f"\n📍 ABSOLUTE POSITIONS:")
    print(f"  A_0: [{A_0_pose.x:7.1f}, {A_0_pose.y:7.1f}, {A_0_pose.z:7.1f}] mm")
    print(f"  A_i: [{A_i_pose.x:7.1f}, {A_i_pose.y:7.1f}, {A_i_pose.z:7.1f}] mm")
    actual_motion = np.sqrt((A_i_pose.x - A_0_pose.x)**2 + 
                           (A_i_pose.y - A_0_pose.y)**2 + 
                           (A_i_pose.z - A_0_pose.z)**2)
    print(f"  Actual arm motion: {actual_motion:.1f} mm")
    
    return best[0]  # Return best method number

def get_aruco_dict_constant(dict_name: str):
    """Convert string aruco dictionary name to cv2.aruco constant."""
    dict_map = {
        '4X4_50': cv2.aruco.DICT_4X4_50,
        '4X4_100': cv2.aruco.DICT_4X4_100,
        '4X4_250': cv2.aruco.DICT_4X4_250,
        '4X4_1000': cv2.aruco.DICT_4X4_1000,
        '5X5_50': cv2.aruco.DICT_5X5_50,
        '5X5_100': cv2.aruco.DICT_5X5_100,
        '5X5_250': cv2.aruco.DICT_5X5_250,
        '5X5_1000': cv2.aruco.DICT_5X5_1000,
        '6X6_50': cv2.aruco.DICT_6X6_50,
        '6X6_100': cv2.aruco.DICT_6X6_100,
        '6X6_250': cv2.aruco.DICT_6X6_250,
        '6X6_1000': cv2.aruco.DICT_6X6_1000,
        '7X7_50': cv2.aruco.DICT_7X7_50,
        '7X7_100': cv2.aruco.DICT_7X7_100,
        '7X7_250': cv2.aruco.DICT_7X7_250,
        '7X7_1000': cv2.aruco.DICT_7X7_1000,
    }
    return dict_map.get(dict_name, cv2.aruco.DICT_6X6_250)


def parse_poses_from_json(json_path: str) -> list:
    """
    Parse poses from a JSON file.

    Args:
        json_path: Path to JSON file containing list of pose objects

    Returns:
        List of dicts with pose attributes: x, y, z, o_x, o_y, o_z, theta

    Example JSON format:
        [
          {"x": 100, "y": 200, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 45},
          {"x": 150, "y": 100, "z": 250, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 90}
        ]
    """
    if json_path is None:
        return None
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Poses file not found: {json_path}")

    with open(json_path, 'r') as f:
        poses = json.load(f)

    if not isinstance(poses, list):
        raise ValueError("JSON file must contain a list of pose objects")

    # Validate each pose has required attributes
    required_attrs = ['x', 'y', 'z', 'o_x', 'o_y', 'o_z', 'theta']
    for i, pose in enumerate(poses):
        for attr in required_attrs:
            if attr not in pose:
                raise ValueError(f"Pose {i} missing required attribute '{attr}'")

    return poses

def _pose_to_matrix(pose: Pose) -> np.ndarray:
    """Convert a Viam Pose to a 4x4 homogeneous transformation matrix."""
    # Get 3x3 rotation matrix from orientation vector
    R = call_go_ov2mat(pose.o_x, pose.o_y, pose.o_z, pose.theta)
    if R is None:
        raise Exception("Failed to convert orientation vector to rotation matrix")

    # Build 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [pose.x, pose.y, pose.z]

    return T

def _matrix_to_pose(T: np.ndarray) -> Pose:
    """Convert a 4x4 homogeneous transformation matrix to a Viam Pose."""
    # Extract rotation matrix (top-left 3x3)
    R = T[0:3, 0:3]

    # Extract translation vector
    t = T[0:3, 3]

    # Convert rotation matrix to orientation vector
    ov = call_go_mat2ov(R)
    if ov is None:
        raise Exception("Failed to convert rotation matrix to orientation vector")

    ox, oy, oz, theta = ov

    return Pose(
        x=float(t[0]),
        y=float(t[1]),
        z=float(t[2]),
        o_x=ox,
        o_y=oy,
        o_z=oz,
        theta=theta
    )

def _invert_pose_rotation_only(pose: Pose) -> Pose:
    """Invert only the rotation of a pose, keeping translation unchanged."""
    # Convert pose to matrix
    T = _pose_to_matrix(pose)

    # Extract rotation and translation
    R = T[0:3, 0:3]
    t = T[0:3, 3]

    # Invert rotation (transpose for rotation matrices)
    R_inv = R.T

    # Build new transformation with inverted rotation but same translation
    T_inv_rot = np.eye(4)
    T_inv_rot[0:3, 0:3] = R_inv
    T_inv_rot[0:3, 3] = t  # Keep original translation

    # Convert back to pose
    return _matrix_to_pose(T_inv_rot)

async def connect():
    load_dotenv()
    opts = RobotClient.Options.with_api_key( 
        api_key=os.getenv('VIAM_MACHINE_API_KEY'),
        api_key_id=os.getenv('VIAM_MACHINE_API_KEY_ID'),
    )
    address = os.getenv('VIAM_MACHINE_ADDRESS')
    if not address:
        raise Exception("VIAM_MACHINE_ADDRESS environment variable not found. Check your .env file.")
    robot = await RobotClient.at_address(address, opts)
    if not robot:
        raise Exception("Failed to create RobotClient")
    dial_options = DialOptions.with_api_key(api_key=os.getenv('VIAM_MACHINE_API_KEY'), api_key_id=os.getenv('VIAM_MACHINE_API_KEY_ID'))
    viam_client = await ViamClient.create_from_dial_options(dial_options)
    if not viam_client:
        raise Exception("Failed to create ViamClient")
    app_client = viam_client.app_client
    if not app_client:
        raise Exception("Failed to create AppClient")
    return app_client, robot

async def _get_current_arm_pose(motion: MotionClient, arm_name: str, arm: Arm) -> Pose:
    pose_in_frame = await arm.get_end_position()
    return pose_in_frame

def frame_config_to_transformation_matrix(frame_config):
    """
    Convert Viam frame configuration to a 4x4 transformation matrix.
    Works with both dictionary and object formats.
    """
    # Extract translation - handle both dict and object formats
    if isinstance(frame_config, dict):
        translation = frame_config.get('translation', {})
        t = np.array([
            translation.get('x', 0),
            translation.get('y', 0), 
            translation.get('z', 0)
        ])
    else:
        t = np.array([frame_config.translation.x, frame_config.translation.y, frame_config.translation.z])
    
    # Extract rotation - handle both dict and object formats
    if isinstance(frame_config, dict):
        orientation = frame_config.get('orientation', {})
        if orientation and 'value' in orientation:
            value = orientation['value']
            th = value.get('th', 0)
            axis = np.array([
                value.get('x', 0),
                value.get('y', 0),
                value.get('z', 0)
            ])
        else:
            axis = np.array([0, 0, 1])
            th = 0
    else:
        if frame_config.orientation and hasattr(frame_config.orientation, 'value'):
            th = frame_config.orientation.value['th']
            axis = np.array([
                frame_config.orientation.value['x'],
                frame_config.orientation.value['y'],
                frame_config.orientation.value['z']
            ])
        else:
            axis = np.array([0, 0, 1])
            th = 0
    
    # Normalize axis
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 0:
        axis = axis / axis_norm
    
    # Build rotation matrix from axis-angle
    th_rad = np.deg2rad(th)
    c = np.cos(th_rad)
    s = np.sin(th_rad)
    t_temp = 1 - c
    
    x, y, z = axis
    R = np.array([
        [t_temp*x*x + c,      t_temp*x*y - s*z,  t_temp*x*z + s*y],
        [t_temp*x*y + s*z,    t_temp*y*y + c,    t_temp*y*z - s*x],
        [t_temp*x*z - s*y,    t_temp*y*z + s*x,  t_temp*z*z + c]
    ])
    
    # Build transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    
    return T


def validate_chessboard_detection(image, corners, rvec, tvec, camera_matrix, dist_coeffs, 
                                   chessboard_size, square_size=30.0, objp=None):
    """
    Validate chessboard detection quality by computing reprojection error and sharpness.
    
    Args:
        objp: If provided, use this object points array. Otherwise generate from chessboard_size.
              This is important when corners have been filtered for outliers!
    
    Returns: (mean_reprojection_error, max_reprojection_error, reprojected_points, sharpness, errors)
    """
    # Generate 3D object points ONLY if not provided
    if objp is None:
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
    
    # CRITICAL FIX: Reshape corners to (N, 2) to match reprojected_points
    # OpenCV returns corners as (N, 1, 2), but we need (N, 2) for calculations
    corners_2d = corners.reshape(-1, 2)
    
    # Verify dimensions match
    if len(corners_2d) != len(objp):
        raise ValueError(f"Dimension mismatch: {len(corners_2d)} corners but {len(objp)} object points")
    
    # Calculate reprojection errors using the same method as camera_calibration.py
    # Project 3D points back to image space
    reprojected_points, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    reprojected_points = reprojected_points.reshape(-1, 2)
    
    # Calculate reprojection error using camera_calibration.py method (OpenCV standard)
    imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(corners, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error = error
    
    # Calculate per-point errors (NOW WITH CORRECT DIMENSIONS!)
    errors = np.linalg.norm(corners_2d - reprojected_points, axis=1)
    
    # Filter outliers using multiple methods
    # Method 1: Remove errors > 3 standard deviations
    mean_error_raw = np.mean(errors)
    std_error = np.std(errors)
    threshold_3std = mean_error_raw + 3 * std_error
    filtered_errors_3std = errors[errors < threshold_3std]
    
    # Method 2: Remove top 5% of errors (percentile-based)
    threshold_95th = np.percentile(errors, 95)
    filtered_errors_95th = errors[errors < threshold_95th]
    
    # Method 3: Remove errors > 2 pixels (tighter absolute threshold for calibration)
    threshold_abs = 2.0
    filtered_errors_abs = errors[errors < threshold_abs]
    
    # Calculate statistics
    mean_error2 = np.mean(errors)
    max_error2 = np.max(errors)
    mean_error_filtered_3std = np.mean(filtered_errors_3std) if len(filtered_errors_3std) > 0 else 0
    mean_error_filtered_95th = np.mean(filtered_errors_95th) if len(filtered_errors_95th) > 0 else 0
    mean_error_filtered_abs = np.mean(filtered_errors_abs) if len(filtered_errors_abs) > 0 else 0
    
    print(f"Reprojection error (OpenCV method): {mean_error:.3f} pixels")
    print(f"Reprojection error (mean): {mean_error2:.3f} pixels")
    print(f"Reprojection error (3σ filtered): {mean_error_filtered_3std:.3f} pixels ({len(filtered_errors_3std)}/{len(errors)} points)")
    print(f"Reprojection error (95th percentile): {mean_error_filtered_95th:.3f} pixels ({len(filtered_errors_95th)}/{len(errors)} points)")
    print(f"Reprojection error (<2px): {mean_error_filtered_abs:.3f} pixels ({len(filtered_errors_abs)}/{len(errors)} points)")
    print(f"Max individual error: {max_error2:.3f} pixels")
    
    # Count outliers with different thresholds
    outliers_1px = np.sum(errors > 1.0)
    outliers_2px = np.sum(errors > 2.0)
    outliers_5px = np.sum(errors > 5.0)
    print(f"Outliers: {outliers_1px} >1px, {outliers_2px} >2px, {outliers_5px} >5px")
    
    # Create histogram of reprojection errors
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        # Create histogram with appropriate bins for sub-pixel errors
        max_bin_value = min(max_error2 * 1.1, 10.0)  # Cap at 10 pixels for better visualization
        bins = np.linspace(0, max_bin_value, 51)
        
        plt.hist(errors, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(mean_error2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error2:.3f}px')
        plt.axvline(threshold_abs, color='green', linestyle='--', linewidth=2, label=f'{threshold_abs}px threshold')
        
        # Add percentile lines
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        plt.axvline(p50, color='purple', linestyle=':', linewidth=1.5, label=f'Median: {p50:.3f}px')
        plt.axvline(p95, color='orange', linestyle=':', linewidth=1.5, label=f'95th %ile: {p95:.3f}px')
        
        plt.xlabel('Reprojection Error (pixels)', fontsize=12)
        plt.ylabel('Number of Corners', fontsize=12)
        plt.title('Distribution of Reprojection Errors\n(After Outlier Filtering)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add comprehensive statistics text
        stats_text = (f'Total points: {len(errors)}\n'
                     f'Mean: {mean_error2:.3f}px\n'
                     f'Median: {p50:.3f}px\n'
                     f'Std: {np.std(errors):.3f}px\n'
                     f'Max: {max_error2:.3f}px\n'
                     f'Outliers >2px: {outliers_2px}')
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        # Save histogram
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        histogram_path = f"reprojection_error_histogram_{timestamp}.png"
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        print(f"Saved reprojection error histogram: {histogram_path}")
        
        plt.close()  # Close to free memory
        
    except ImportError:
        print("Matplotlib not available, skipping histogram")
    except Exception as e:
        print(f"Failed to create histogram: {e}")
    
    return mean_error, max_error2, reprojected_points, errors


def get_chessboard_pose_in_camera_frame(image, camera_matrix, dist_coeffs, chessboard_size, 
                                       square_size=30.0, pnp_method=cv2.SOLVEPNP_IPPE, 
                                       use_sb_detection=True):
    """
    Get chessboard pose in camera frame using PnP with improved outlier filtering.
    
    Note: cv2.solvePnP returns the transformation from chessboard coordinates to camera coordinates.
    This is T_chessboard_to_camera, NOT T_camera_to_chessboard.
    
    Args:
        use_sb_detection: If True, use findChessboardCornersSB (more robust), otherwise use findChessboardCorners
    
    Returns: (success, rotation_vector, translation_vector, corners, marker_info)
        - rvec, tvec represent the chessboard's pose in the camera's coordinate system
        - marker_info contains validation info (reprojection error, sharpness)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Prepare 3D object points (chessboard corners in world coordinates)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objp *= square_size  # Scale by square size (e.g., 25mm)
    
    # Find chessboard corners using selected method
    corners = None
    if use_sb_detection:
        print("Using findChessboardCornersSB (subpixel detection)")
        flags = (cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, flags=flags)
    else:
        print("Using findChessboardCorners (traditional detection)")
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # Enhanced corner refinement for higher accuracy (only for traditional method)
        if not use_sb_detection:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)  # More iterations, stricter convergence
            corners = cv2.cornerSubPix(gray, corners, (15,15), (-1,-1), criteria)  # Larger refinement window

        # Estimate chessboard sharpness (only when detection succeeds)
        sharpness = float('inf')
        try:
            sharpness_result = cv2.estimateChessboardSharpness(image, chessboard_size, corners)
            # The function returns a tuple ((sharpness_value, ...), sharpness_map)
            sharpness = sharpness_result[0][0]  # First element of first tuple
            print(f"Chessboard sharpness: {sharpness:.2f} pixels")
        except Exception as e:
            print(f"Could not estimate sharpness: {e}")
            sharpness = float('inf')  # Mark as unknown
        
        
        # Filter outliers before solvePnP with IMPROVED thresholds
        print(f"Original corners: {len(corners)} points")
        
        # Method 1: Use iterative outlier filtering with tighter threshold
        try:
            # First, try to get an initial pose estimate with all points
            success_init, rvec_init, tvec_init = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)
            
            if success_init:
                # Project points back and calculate errors
                projected_points, _ = cv2.projectPoints(objp, rvec_init, tvec_init, camera_matrix, dist_coeffs)
                projected_points = projected_points.reshape(-1, 2)
                corners_2d = corners.reshape(-1, 2)
                
                # Calculate reprojection errors
                errors = np.linalg.norm(corners_2d - projected_points, axis=1)
                
                # IMPROVED: Use statistical outlier detection
                # Filter using median absolute deviation (more robust than std)
                median_error = np.median(errors)
                mad = np.median(np.abs(errors - median_error))
                
                # Modified z-score using MAD
                modified_z_scores = 0.6745 * (errors - median_error) / (mad + 1e-10)
                
                # Keep points with modified z-score < 3.5 (equivalent to ~3 sigma)
                # OR use absolute threshold of 2 pixels (whichever is stricter)
                threshold_statistical = median_error + 3.5 * mad
                threshold_absolute = 2.0
                threshold = min(threshold_statistical, threshold_absolute)
                
                good_indices = errors < threshold
                
                n_filtered = len(corners) - np.sum(good_indices)
                if n_filtered > 0:
                    print(f"Filtering {n_filtered} outliers (threshold: {threshold:.2f}px)")
                    print(f"  Error range: {np.min(errors):.3f} to {np.max(errors):.3f}px")
                    print(f"  Median: {median_error:.3f}px, MAD: {mad:.3f}px")
                
                if np.sum(good_indices) >= 20:  # Need at least 20 points for reliable PnP
                    filtered_corners = corners[good_indices]
                    filtered_objp = objp[good_indices]
                    print(f"Filtered corners: {len(filtered_corners)}/{len(corners)} points (error < {threshold:.2f}px)")
                    
                    # Use filtered points for final solvePnP
                    success, rvec, tvec = cv2.solvePnP(filtered_objp, filtered_corners, camera_matrix, dist_coeffs, flags=pnp_method)
                    corners = filtered_corners  # Update corners for validation
                    objp = filtered_objp  # Update objp for validation
                else:
                    print(f"Not enough good points ({np.sum(good_indices)}), using all points")
                    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)
            else:
                print("Initial solvePnP failed, using all points")
                success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)
                
        except Exception as e:
            print(f"Outlier filtering failed: {e}, using all points")
            success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)

        if not success:
            print("Failed to solve PnP")
            return False, None, None, None, None

        # Enhanced refinement with VVS
        rvec, tvec = cv2.solvePnPRefineVVS(objp, corners, camera_matrix, dist_coeffs, rvec, tvec)
        
        # Additional refinement with LM method
        rvec, tvec = cv2.solvePnPRefineLM(objp, corners, camera_matrix, dist_coeffs, rvec, tvec)
        
        # Validate detection quality
        mean_error, max_error, reprojected_points, errors = validate_chessboard_detection(
            image, corners, rvec, tvec, camera_matrix, dist_coeffs, chessboard_size, square_size, objp
        )
        
        print(f"Chessboard detection quality: mean={mean_error:.3f}px, max={max_error:.3f}px")
        
        # Return validation info as marker_info
        validation_info = {
            'mean_reprojection_error': mean_error,
            'max_reprojection_error': max_error,
            'reprojected_points': reprojected_points,
            'sharpness': sharpness,
            'errors': errors
        }
        
        return True, rvec, tvec, corners, validation_info
    else:
        return False, None, None, None, None


# [Rest of the file remains the same - just include the complete original file from line 696 onwards]
# I'll mark this with a comment so you know where to append the rest

async def get_camera_image(camera: Camera) -> np.ndarray:
    cam_images = await camera.get_images()
    pil_image = None
    for cam_image in cam_images[0]:
        # Accept any standard image format that viam_to_pil_image can handle
        if cam_image.mime_type in [CameraMimeType.JPEG, CameraMimeType.PNG, CameraMimeType.VIAM_RGBA]:
            pil_image = viam_to_pil_image(cam_image)
            break
    if pil_image is None:
        raise Exception("Could not get latest image from camera")        
    image = np.array(pil_image)
    return image
    
async def get_camera_intrinsics(camera: Camera) -> tuple:
    """Get camera intrinsic parameters"""
    camera_params = await camera.do_command({"get_camera_params": None})
    intrinsics = camera_params["Color"]["intrinsics"]
    dist_params = camera_params["Color"]["distortion"]
    
    K = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist = np.array([dist_params["k1"], dist_params["k2"], dist_params["p1"], dist_params["p2"], dist_params["k3"]], dtype=np.float32)
    
    if K is None or dist is None:
        raise Exception("Could not get camera intrinsic parameters")
    
    return K, dist

def rvec_tvec_to_matrix(rvec, tvec):
    """Convert rotation vector and translation vector to 4x4 transformation matrix"""
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    
    return T

def get_aruco_pose_in_camera_frame(image, camera_matrix, dist_coeffs, marker_id=0, marker_size=300.0, aruco_dict=cv2.aruco.DICT_6X6_250, pnp_method=cv2.SOLVEPNP_IPPE_SQUARE):
    """
    Get ArUco marker pose in camera frame using PnP.
    
    Args:
        image: Input image
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        marker_id: ID of the marker to detect (if None, uses first detected marker)
        marker_size: Size of the marker in mm
        aruco_dict: ArUco dictionary to use
        pnp_method: PnP method to use (cv2.SOLVEPNP_* constant)
        
    Returns: (success, rotation_vector, translation_vector, corners, detected_id)
        - rvec, tvec represent the marker's pose in the camera's coordinate system
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create ArUco detector with optimized parameters
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    parameters = cv2.aruco.DetectorParameters()
    
    # Optimize corner refinement for higher accuracy
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01
    parameters.cornerRefinementWinSize = 7
    parameters.relativeCornerRefinmentWinSize = 0.4
    
    # Optimize detection sensitivity
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 3.0
    parameters.minCornerDistanceRate = 0.03
    
    # Improve error correction
    parameters.errorCorrectionRate = 0.8
    parameters.maxErroneousBitsInBorderRate = 0.2
    
    # Optimize adaptive thresholding
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 25
    parameters.adaptiveThreshWinSizeStep = 8
    parameters.adaptiveThreshConstant = 5
    
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, parameters)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
        # Find the requested marker (or use first one if marker_id is None)
        marker_idx = None
        if marker_id is None:
            marker_idx = 0
            detected_id = ids[0][0]
        else:
            for i, id_val in enumerate(ids):
                if id_val[0] == marker_id:
                    marker_idx = i
                    detected_id = marker_id
                    break
        
        if marker_idx is None:
            print(f"Marker ID {marker_id} not found. Detected IDs: {ids.flatten()}")
            return False, None, None, None, None
        
        # Get corners for the detected marker
        marker_corners = corners[marker_idx][0]  # Shape: (4, 2)
        
        # Define 3D points of the marker (standard ArUco coordinate system)
        # Origin at top-left, X-right, Y-down, Z-out
        objp = np.array([
            [0, 0, 0],                    # Top-left
            [marker_size, 0, 0],          # Top-right
            [marker_size, marker_size, 0], # Bottom-right
            [0, marker_size, 0]           # Bottom-left
        ], dtype=np.float32)
        
        # Solve PnP for this marker using specified method
        success, rvec, tvec = cv2.solvePnP(
            objp, marker_corners, camera_matrix, dist_coeffs,
            flags=pnp_method
        )
        
        if not success:
            print("Failed to solve PnP for ArUco marker")
            return False, None, None, None, None
        
        # Refine with VVS
        rvec, tvec = cv2.solvePnPRefineVVS(objp, marker_corners, camera_matrix, dist_coeffs, rvec, tvec)
        
        return True, rvec, tvec, marker_corners, detected_id
    else:
        return False, None, None, None, None

def get_pnp_method_constant(method_name: str):
    """Convert string PnP method name to cv2 constant."""
    method_map = {
        'IPPE_SQUARE': cv2.SOLVEPNP_IPPE_SQUARE,
        'IPPE': cv2.SOLVEPNP_IPPE,
        'ITERATIVE': cv2.SOLVEPNP_ITERATIVE,
        'SQPNP': cv2.SOLVEPNP_SQPNP,
    }
    return method_map.get(method_name, cv2.SOLVEPNP_IPPE_SQUARE)

def draw_marker_debug(image, rvec, tvec, camera_matrix, dist_coeffs, marker_type='chessboard', 
                     chessboard_size=(11, 8), square_size=30.0, aruco_size=200.0, validation_info=None):
    """
    Draw debug visualization on image showing detected marker and coordinate axes.
    """
    debug_image = image.copy()
    
    if marker_type == 'chessboard':
        # Draw coordinate axes for chessboard
        cv2.drawFrameAxes(debug_image, camera_matrix, dist_coeffs, rvec, tvec, length=50)
        
        # Draw reprojection validation if available
        if validation_info and 'reprojected_points' in validation_info:
            reprojected_points = validation_info['reprojected_points']
            mean_error = validation_info['mean_reprojection_error']
            max_error = validation_info['max_reprojection_error']
            
            # Draw reprojected points in green
            for point in reprojected_points:
                cv2.circle(debug_image, tuple(point.astype(int)), 2, (0, 255, 0), -1)
            
            # Add error text
            cv2.putText(debug_image, f"Reprojection Error: {mean_error:.1f}px", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_image, f"Max Error: {max_error:.1f}px", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add sharpness text with color coding
            if 'sharpness' in validation_info:
                sharpness = validation_info['sharpness']
                if sharpness != float('inf'):
                    sharpness_color = (0, 255, 0) if sharpness < 3.0 else (0, 165, 255) if sharpness < 5.0 else (0, 0, 255)
                    sharpness_text = f"Sharpness: {sharpness:.1f}px"
                    cv2.putText(debug_image, sharpness_text, 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sharpness_color, 2)
    elif marker_type == 'aruco':
        # Draw coordinate axes for ArUco marker
        cv2.drawFrameAxes(debug_image, camera_matrix, dist_coeffs, rvec, tvec, aruco_size/2)
    
    return debug_image

def get_marker_pose_in_camera_frame(image, camera_matrix, dist_coeffs, marker_type='chessboard', 
                                   chessboard_size=(11, 8), square_size=30.0,
                                   aruco_id=0, aruco_size=200.0, aruco_dict='6X6_250', pnp_method='IPPE_SQUARE', use_sb_detection=True):
    """
    Get marker pose in camera frame using PnP.
    Supports both chessboard and ArUco markers.
    
    Returns: (success, rotation_vector, translation_vector, corners, marker_info)
    """
    if marker_type == 'chessboard':
        pnp_method_const = get_pnp_method_constant(pnp_method)
        return get_chessboard_pose_in_camera_frame(image, camera_matrix, dist_coeffs, chessboard_size, square_size, pnp_method=pnp_method_const, use_sb_detection=use_sb_detection)
    elif marker_type == 'aruco':
        aruco_dict_const = get_aruco_dict_constant(aruco_dict)
        pnp_method_const = get_pnp_method_constant(pnp_method)
        return get_aruco_pose_in_camera_frame(image, camera_matrix, dist_coeffs, 
                                            marker_id=aruco_id, marker_size=aruco_size, aruco_dict=aruco_dict_const, pnp_method=pnp_method_const)
    else:
        raise ValueError(f"Unknown marker type: {marker_type}")


async def get_hand_eye_from_machine(app_client: AppClient, camera_name: str):
    """Get the hand-eye transformation from the machine's frame configuration"""
    print(f"\n=== EXTRACTING HAND-EYE TRANSFORMATION ===")

    try:
        organizations = await app_client.list_organizations()
        if not organizations:
            print("Warning: No organizations found")
            return None

        org = organizations[0]
        locations = await app_client.list_locations(org_id=org.id)
        if not locations:
            print("Warning: No locations found")
            return None

        location = locations[0]
        robots = await app_client.list_robots(location_id=location.id)
        if not robots:
            print("Warning: No robots found")
            return None

        robot = robots[0]
        robot_parts = await app_client.get_robot_parts(robot.id)
        if not robot_parts:
            print("Warning: No robot parts found")
            return None

        robot_part_config = await app_client.get_robot_part(robot_parts[0].id)
        if not robot_part_config:
            print("Warning: Could not retrieve robot configuration")
            return None

        robot_config = robot_part_config.robot_config
        if 'components' not in robot_config:
            print("Warning: Robot config does not have 'components' key")
            return None

        # Find the camera component configuration
        camera_config = None
        for component in robot_config['components']:
            if component.get('name') == camera_name:
                camera_config = component
                break

        if not camera_config or 'frame' not in camera_config or not camera_config['frame']:
            print(f"Warning: No frame configuration found for camera '{camera_name}'")
            return None

        frame_config = camera_config['frame']
        if not isinstance(frame_config, dict):
            print(f"Warning: Frame configuration is not a dictionary")
            return None

        print(f"Found camera frame configuration")

        parent = frame_config.get('parent', 'unknown')
        translation = frame_config.get('translation', {})
        print(f"Parent frame: {parent}")
        print(f"Translation: x={translation.get('x', 0):.3f}, y={translation.get('y', 0):.3f}, z={translation.get('z', 0):.3f}")

        T_hand_eye = frame_config_to_transformation_matrix(frame_config)

        print(f"\nHand-Eye Transformation Matrix:")
        for i in range(4):
            print(f"  [{T_hand_eye[i,0]:8.4f} {T_hand_eye[i,1]:8.4f} {T_hand_eye[i,2]:8.4f} {T_hand_eye[i,3]:8.4f}]")

        return T_hand_eye

    except Exception as e:
        print(f"Error retrieving frame configuration: {e}")
        return None


def compute_hand_eye_verification_errors(T_hand_eye, T_delta_A_world_frame, T_delta_B_camera_frame):
    """
    Compute hand-eye calibration verification errors.

    Args:
        T_hand_eye: 4x4 hand-eye transformation matrix (camera to gripper)
        T_delta_A_world_frame: 4x4 actual robot motion in world frame
        T_delta_B_camera_frame: 4x4 camera motion (chessboard motion in camera frame)

    Returns:
        dict with rotation_error (degrees) and translation_error (mm)
    """
    # Predict robot motion from camera motion: A_predicted = X^(-1) @ B @ X
    T_eye_hand = np.linalg.inv(T_hand_eye)
    T_A_predicted = T_hand_eye @ T_delta_B_camera_frame @ T_eye_hand 


    # Extract rotation matrices and translations
    R_A_actual = T_delta_A_world_frame[:3, :3]
    R_A_predicted = T_A_predicted[:3, :3]
    t_A_actual = T_delta_A_world_frame[:3, 3]
    t_A_predicted = T_A_predicted[:3, 3]

    # Calculate rotation error
    R_error = R_A_predicted.T @ R_A_actual
    angle_rad = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1.0, 1.0))
    rotation_error = np.degrees(angle_rad)

    # Calculate translation error
    t_error = t_A_predicted - t_A_actual
    translation_error = np.linalg.norm(t_error)

    return {
        'rotation_error': rotation_error,
        'translation_error': translation_error
    }

async def main(
    arm_name: str,
    pose_tracker_name: str,
    motion_service_name: str,
    camera_name: str,
    poses: list,
    marker_type: str = 'chessboard',
    aruco_id: int = 0,
    aruco_size: float = 200.0,
    aruco_dict: str = '6X6_250',
    pnp_method: str = 'IPPE_SQUARE',
    use_sb_detection: bool = True,
):
    app_client: Optional[AppClient] = None
    machine: Optional[RobotClient] = None
    pt: Optional[PoseTracker] = None
    arm: Optional[Arm] = None
    motion_service: Optional[MotionClient] = None
    camera: Optional[Camera] = None
    
    try:
        app_client, machine = await connect()
        arm = Arm.from_robot(machine, arm_name)
        await arm.do_command({"set_vel": 25})
        camera = Camera.from_robot(machine, camera_name)
        motion_service = MotionClient.from_robot(machine, motion_service_name)
        pt = PoseTracker.from_robot(machine, pose_tracker_name)

        print(f"Connected to robot")

        # Get the hand-eye transformation from camera configuration
        T_hand_eye = await get_hand_eye_from_machine(app_client, camera_name)
        if T_hand_eye is None:
            print("ERROR: Could not retrieve hand-eye transformation")
            return

        # Get initial poses
        A_0_pose_world_frame_raw = await _get_current_arm_pose(motion_service, arm.name, arm)
        # Invert only the rotation, keep translation unchanged
        A_0_pose_world_frame = _invert_pose_rotation_only(A_0_pose_world_frame_raw)
        T_A_0_world_frame = _pose_to_matrix(A_0_pose_world_frame)

        camera_matrix, dist_coeffs = await get_camera_intrinsics(camera)
        image = await get_camera_image(camera)

        success, rvec, tvec, _, marker_info = get_marker_pose_in_camera_frame(
            image, camera_matrix, dist_coeffs, marker_type=marker_type,
            chessboard_size=(11, 8), square_size=30.0,
            aruco_id=aruco_id, aruco_size=aruco_size, aruco_dict=aruco_dict, pnp_method=pnp_method, use_sb_detection=use_sb_detection
        )
        if not success:
            print(f"Failed to detect {marker_type} in reference image")
            return
        
        if marker_type == 'aruco':
            print(f"Detected ArUco marker ID: {marker_info}")
        
        # Convert rvec, tvec to 4x4 transformation matrix (chessboard in camera frame)
        # Don't transpose - solvePnP output is already in OpenCV convention
        T_B_0_camera_frame = rvec_tvec_to_matrix(rvec, tvec)
        
        # Create directory for saving data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = f"calibration_data_{timestamp}"
        os.makedirs(data_dir, exist_ok=True)
        print(f"\n=== SAVING DATA TO: {data_dir} ===")
        
        # Save camera calibration and chessboard config
        calibration_data = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "chessboard_size": [11, 8],  # (width, height)
            "square_size": 30.0,  # mm
            "hand_eye_transform": T_hand_eye.tolist() if T_hand_eye is not None else None,
            "timestamp": timestamp,
            "A_0_pose_raw": {
                "x": A_0_pose_world_frame_raw.x,
                "y": A_0_pose_world_frame_raw.y,
                "z": A_0_pose_world_frame_raw.z,
                "o_x": A_0_pose_world_frame_raw.o_x,
                "o_y": A_0_pose_world_frame_raw.o_y,
                "o_z": A_0_pose_world_frame_raw.o_z,
                "theta": A_0_pose_world_frame_raw.theta
            },
            "A_0_pose_inverted": {
                "x": A_0_pose_world_frame.x,
                "y": A_0_pose_world_frame.y,
                "z": A_0_pose_world_frame.z,
                "o_x": A_0_pose_world_frame.o_x,
                "o_y": A_0_pose_world_frame.o_y,
                "o_z": A_0_pose_world_frame.o_z,
                "theta": A_0_pose_world_frame.theta
            },
            "T_B_0_camera_frame": T_B_0_camera_frame.tolist(),
            "note": "Arm poses have rotation inverted (translation unchanged) before processing"
        }
        with open(os.path.join(data_dir, "calibration_config.json"), "w") as f:
            json.dump(calibration_data, f, indent=2)
        
        # Save reference image
        # Save reference image with debug visualization
        debug_image = draw_marker_debug(image, rvec, tvec, camera_matrix, dist_coeffs, 
                                      marker_type=marker_type, aruco_size=aruco_size, validation_info=marker_info)
        cv2.imwrite(os.path.join(data_dir, "image_reference.jpg"), debug_image)
        print(f"Saved reference image and config")
        
        # List to store all rotation data
        rotation_data = []

        # Test the hand-eye transformation with provided poses
        if poses is None:
            print("No poses provided, using default poses")
            initial_pose = await _get_current_arm_pose(motion_service, arm.name, arm)
            print(f"Initial pose theta: {initial_pose.theta:.1f}°")
            
            poses = []
            for i in range(4):
                new_pose = Pose(
                    x=initial_pose.x,
                    y=initial_pose.y,
                    z=initial_pose.z,
                    o_x=initial_pose.o_x,
                    o_y=initial_pose.o_y,
                    o_z=initial_pose.o_z,
                    theta=i * 90  # Normalize to 0°, 90°, 180°, 270°
                )
                poses.append(new_pose)
            print(f"Created poses at: 0°, 90°, 180°, 270°")
        

        await arm.do_command({"set_vel": DEFAULT_VELOCITY_SLOW})
        print(f"\n=== TESTING {len(poses)} POSES ===")
        for i, pose_spec in enumerate(poses):
            print(f"\n=== POSE {i+1}/{len(poses)} ===")

            # Create target pose from specification
            if isinstance(pose_spec, dict):
                # Pose from file (dictionary)
                target_pose = Pose(
                    x=pose_spec['x'],
                    y=pose_spec['y'],
                    z=pose_spec['z'],
                    o_x=pose_spec['o_x'],
                    o_y=pose_spec['o_y'],
                    o_z=pose_spec['o_z'],
                    theta=pose_spec['theta']
                )
            else:
                # Pose object (default poses)
                target_pose = pose_spec
            print(f"  Position: ({target_pose.x:.1f}, {target_pose.y:.1f}, {target_pose.z:.1f})")
            print(f"  Orientation: ({target_pose.o_x:.3f}, {target_pose.o_y:.3f}, {target_pose.o_z:.3f}) @ {target_pose.theta:.1f}°")

            # Move to target pose
            target_pose_in_frame = PoseInFrame(reference_frame=DEFAULT_WORLD_FRAME, pose=target_pose)
            await motion_service.move(component_name=arm.name, destination=target_pose_in_frame)
            await asyncio.sleep(DEFAULT_SETTLE_TIME)  # Increased settling time to reduce motion blur
            
            image = await get_camera_image(camera)
            success, rvec, tvec, _, marker_info = get_marker_pose_in_camera_frame(
                image, camera_matrix, dist_coeffs, marker_type=marker_type,
                chessboard_size=(11, 8), square_size=30.0,
                aruco_id=aruco_id, aruco_size=aruco_size, aruco_dict=aruco_dict, pnp_method=pnp_method, use_sb_detection=use_sb_detection
            )
            if not success:
                print(f"  Failed to detect {marker_type}, skipping pose {i+1}")
                continue
            
            if marker_type == 'aruco':
                print(f"  Detected ArUco marker ID: {marker_info}")

            # Get current poses
            A_i_pose_world_frame_raw = await _get_current_arm_pose(motion_service, arm.name, arm)
            # Invert only the rotation, keep translation unchanged
            A_i_pose_world_frame = _invert_pose_rotation_only(A_i_pose_world_frame_raw)
            T_A_i_world_frame = _pose_to_matrix(A_i_pose_world_frame)

            # Convert chessboard pose (don't transpose - solvePnP is OpenCV convention)
            T_B_i_camera_frame = rvec_tvec_to_matrix(rvec, tvec)
            
            # Save pose data
            # Convert pose_spec to dictionary if it's a Pose object
            if isinstance(pose_spec, Pose):
                pose_spec_dict = {
                    "x": pose_spec.x,
                    "y": pose_spec.y,
                    "z": pose_spec.z,
                    "o_x": pose_spec.o_x,
                    "o_y": pose_spec.o_y,
                    "o_z": pose_spec.o_z,
                    "theta": pose_spec.theta
                }
            else:
                pose_spec_dict = pose_spec
            
            pose_info = {
                "pose_index": i,
                "pose_spec": pose_spec_dict,
                "A_i_pose_raw": {
                    "x": A_i_pose_world_frame_raw.x,
                    "y": A_i_pose_world_frame_raw.y,
                    "z": A_i_pose_world_frame_raw.z,
                    "o_x": A_i_pose_world_frame_raw.o_x,
                    "o_y": A_i_pose_world_frame_raw.o_y,
                    "o_z": A_i_pose_world_frame_raw.o_z,
                    "theta": A_i_pose_world_frame_raw.theta
                },
                "A_i_pose_inverted": {
                    "x": A_i_pose_world_frame.x,
                    "y": A_i_pose_world_frame.y,
                    "z": A_i_pose_world_frame.z,
                    "o_x": A_i_pose_world_frame.o_x,
                    "o_y": A_i_pose_world_frame.o_y,
                    "o_z": A_i_pose_world_frame.o_z,
                    "theta": A_i_pose_world_frame.theta
                },
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "T_B_i_camera_frame": T_B_i_camera_frame.tolist()
            }
            rotation_data.append(pose_info)

            # Save image for this pose
            # Save pose image with debug visualization
            debug_image = draw_marker_debug(image, rvec, tvec, camera_matrix, dist_coeffs, 
                                          marker_type=marker_type, aruco_size=aruco_size, validation_info=marker_info)
            cv2.imwrite(os.path.join(data_dir, f"image_pose_{i+1}.jpg"), debug_image)

            # Save pose data incrementally (in case of crash)
            with open(os.path.join(data_dir, "pose_data.json"), "w") as f:
                json.dump(rotation_data, f, indent=2)
            print(f"  Saved pose {i+1} data")
            
            # Compute relative transformations
            T_delta_A_world_frame = np.linalg.inv(T_A_i_world_frame) @ T_A_0_world_frame
            T_delta_B_camera_frame = T_B_i_camera_frame @ np.linalg.inv(T_B_0_camera_frame)

            # ADD THIS: Detailed debug analysis
            best_method = analyze_hand_eye_error(
                T_hand_eye, 
                T_delta_A_world_frame, 
                T_delta_B_camera_frame,
                A_0_pose_world_frame_raw,
                A_i_pose_world_frame_raw,
                i+1
            )

            # Compute verification errors (keep your existing code)
            errors = compute_hand_eye_verification_errors(
                T_hand_eye,
                T_delta_A_world_frame,
                T_delta_B_camera_frame
            )

            print(f"\n  📊 CURRENT METHOD ERRORS:")
            print(f"  Rotation error: {errors['rotation_error']:.3f}°")
            print(f"  Translation error: {errors['translation_error']:.3f} mm")

            await asyncio.sleep(1.0)
        
        print(f"\n✅ ALL DATA SAVED TO: {data_dir}")
        print(f"   - calibration_config.json (camera params, hand-eye, reference pose)")
        print(f"   - image_reference.jpg + image_pose_1-{len(poses)}.jpg")
        print(f"   - pose_data.json (all arm poses and chessboard detections)")
        
        # Return to reference pose
        print(f"\n=== RETURNING TO REFERENCE POSE ===")
        A_0_pose_in_frame = PoseInFrame(reference_frame=DEFAULT_WORLD_FRAME, pose=A_0_pose_world_frame_raw)
        await motion_service.move(component_name=arm.name, destination=A_0_pose_in_frame)
        await asyncio.sleep(5.0)  # Increased settling time
        
        print("=" * 60)
        
    except Exception as e:
        print("Caught exception in script main: ")
        raise e
    finally:
        if pt:
            await pt.close()
        if arm:
            await arm.close()
        if motion_service:
            await motion_service.close()
        if machine:
            await machine.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test hand-eye calibration by moving robot and verifying transformations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python pose_test_script.py --camera-name sensing-camera --arm-name myarm \\
    --pose-tracker-name mytracker --poses poses.json

Pose JSON format (list of pose objects):
  [
    {"x": 100, "y": 200, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 45},
    {"x": 150, "y": 100, "z": 250, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 90},
    {"x": 120, "y": 180, "z": 280, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 135}
  ]

All pose objects must have: x, y, z, o_x, o_y, o_z, theta
        """
    )
    parser.add_argument(
        '--camera-name',
        type=str,
        required=True,
        help='Name of the camera component'
    )
    parser.add_argument(
        '--arm-name',
        type=str,
        required=True,
        help='Name of the arm component'
    )
    parser.add_argument(
        '--pose-tracker-name',
        type=str,
        required=True,
        help='Name of the pose tracker resource'
    )
    parser.add_argument(
        '--poses',
        type=str,
        required=False,
        help='Path to JSON file containing list of pose objects'
    )
    parser.add_argument(
        '--marker-type',
        type=str,
        choices=['chessboard', 'aruco'],
        default='chessboard',
        help='Type of marker to use for pose estimation (default: chessboard)'
    )
    parser.add_argument(
        '--aruco-id',
        type=int,
        default=0,
        help='ArUco marker ID to detect (default: 0)'
    )
    parser.add_argument(
        '--aruco-size',
        type=float,
        default=200.0,
        help='ArUco marker size in mm (default: 200.0)'
    )
    parser.add_argument(
        '--aruco-dict',
        type=str,
        default='6X6_250',
        choices=['4X4_50', '4X4_100', '4X4_250', '4X4_1000',
                '5X5_50', '5X5_100', '5X5_250', '5X5_1000',
                '6X6_50', '6X6_100', '6X6_250', '6X6_1000',
                '7X7_50', '7X7_100', '7X7_250', '7X7_1000'],
        help='ArUco dictionary to use (default: 6X6_250)'
    )
    parser.add_argument(
        '--pnp-method',
        type=str,
        default='IPPE_SQUARE',
        choices=['IPPE_SQUARE', 'IPPE', 'ITERATIVE', 'SQPNP'],
        help='PnP method to use for ArUco detection (default: IPPE_SQUARE)'
    )
    parser.add_argument(
        '--use-sb-detection',
        action='store_true',
        help='Use findChessboardCornersSB (subpixel detection) instead of findChessboardCorners for chessboard'
    )

    args = parser.parse_args()

    # Parse poses from JSON file
    try:
        poses = parse_poses_from_json(args.poses)
        if poses is None:
            print("No poses provided, using default poses")
            poses = None
        else:
            print(f"Loaded {len(poses)} poses to test")
    except Exception as e:
        print(f"Error parsing poses: {e}")
        exit(1)

    asyncio.run(main(
        arm_name=args.arm_name,
        pose_tracker_name=args.pose_tracker_name,
        motion_service_name="motion",
        camera_name=args.camera_name,
        poses=poses,
        marker_type=args.marker_type,
        aruco_id=args.aruco_id,
        aruco_size=args.aruco_size,
        aruco_dict=args.aruco_dict,
        pnp_method=args.pnp_method,
        use_sb_detection=args.use_sb_detection,
    ))