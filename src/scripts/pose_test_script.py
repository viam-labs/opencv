#!/usr/bin/env python3

import argparse
import asyncio
import os
import numpy as np
import cv2
import json
import logging
import sys
import shlex
import shutil
import matplotlib.pyplot as plt
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
from typing import Optional, Tuple

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

DEFAULT_VELOCITY_SLOW = 10
DEFAULT_SETTLE_TIME = 5.0

import cv2  # Make sure cv2 is imported

def setup_logging(data_dir: str, command: str = None):
    """Setup logging to both console and file"""
    # Create log file path
    log_file = os.path.join(data_dir, "pose_test_log.txt")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log the command that was executed
    if command:
        # Write directly to log file to ensure it's captured
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {'='*80}\n")
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - COMMAND EXECUTED:\n")
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {command}\n")
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {'='*80}\n")
        # Also print to console
        print("=" * 80)
        print("COMMAND EXECUTED:")
        print(command)
        print("=" * 80)
    
    # Create a custom logger that captures print statements
    class PrintLogger:
        def __init__(self, log_file):
            self.log_file = log_file
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            
        def write(self, message):
            if message.strip():  # Only log non-empty messages
                with open(self.log_file, 'a') as f:
                    # Add newline if message doesn't end with one (only for log file)
                    log_message = message
                    if not log_message.endswith('\n'):
                        log_message = log_message + '\n'
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_message}")
            # Write original message to terminal (preserve original formatting)
            self.original_stdout.write(message)
            
        def flush(self):
            self.original_stdout.flush()
    
    # Redirect stdout to capture print statements
    sys.stdout = PrintLogger(log_file)
    
    return log_file

def rotation_error(R1, R2):
    """Compute rotation error in degrees between two rotation matrices."""
    R_error = R1.T @ R2
    angle_rad = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1.0, 1.0))
    return np.degrees(angle_rad)

def compare_poses(pose1: Pose, pose2: Pose, label: str = "Pose comparison", verbose: bool = False) -> dict:
    """
    Compare two poses and return detailed differences.
    
    Args:
        pose1: First pose (from arm.get_end_position())
        pose2: Second pose (from motion.get_pose())
        label: Label for the comparison (for logging)
    
    Returns:
        Dictionary with comparison metrics
    """
    # Convert poses to matrices for comparison
    T1 = _pose_to_matrix(pose1)
    T2 = _pose_to_matrix(pose2)
    
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    
    # Calculate differences
    translation_diff = t2 - t1
    translation_error = np.linalg.norm(translation_diff)
    
    # Rotation error
    rot_error_deg = rotation_error(R1, R2)
    
    # Position differences
    position_diff = {
        'x': pose2.x - pose1.x,
        'y': pose2.y - pose1.y,
        'z': pose2.z - pose1.z
    }
    
    # Orientation differences
    orientation_diff = {
        'o_x': pose2.o_x - pose1.o_x,
        'o_y': pose2.o_y - pose1.o_y,
        'o_z': pose2.o_z - pose1.o_z,
        'theta': pose2.theta - pose1.theta
    }
    
    comparison = {
        'translation_error_mm': float(translation_error),
        'rotation_error_deg': float(rot_error_deg),
        'position_diff_mm': position_diff,
        'orientation_diff': orientation_diff,
        'pose1': {
            'x': pose1.x, 'y': pose1.y, 'z': pose1.z,
            'o_x': pose1.o_x, 'o_y': pose1.o_y, 'o_z': pose1.o_z, 'theta': pose1.theta
        },
        'pose2': {
            'x': pose2.x, 'y': pose2.y, 'z': pose2.z,
            'o_x': pose2.o_x, 'o_y': pose2.o_y, 'o_z': pose2.o_z, 'theta': pose2.theta
        }
    }
    
    # Log the comparison
    if verbose:
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")
        print(f"Translation error: {translation_error:.3f} mm")
        print(f"Rotation error: {rot_error_deg:.3f}°")
        print(f"\nPosition differences (pose2 - pose1):")
        print(f"  X: {position_diff['x']:+.3f} mm")
        print(f"  Y: {position_diff['y']:+.3f} mm")
        print(f"  Z: {position_diff['z']:+.3f} mm")
        print(f"\nOrientation differences (pose2 - pose1):")
        print(f"  o_x: {orientation_diff['o_x']:+.6f}")
        print(f"  o_y: {orientation_diff['o_y']:+.6f}")
        print(f"  o_z: {orientation_diff['o_z']:+.6f}")
        print(f"  theta: {orientation_diff['theta']:+.3f}°")
        print(f"\nPose 1: ({pose1.x:7.2f}, {pose1.y:7.2f}, {pose1.z:7.2f}) mm")
        print(f"Pose 2: ({pose2.x:7.2f}, {pose2.y:7.2f}, {pose2.z:7.2f}) mm")
        print(f"{'='*60}\n")
    
    return comparison

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
    axis_angle_A = cv2.Rodrigues(R_A)[0]
    axis_A = axis_angle_A.flatten() / (np.linalg.norm(axis_angle_A) + 1e-10)
    axis_angle_B = cv2.Rodrigues(R_B)[0]
    axis_B = axis_angle_B.flatten() / (np.linalg.norm(axis_angle_B) + 1e-10)
    print(f"  Arm rotation axis: [{axis_A[0]:6.3f}, {axis_A[1]:6.3f}, {axis_A[2]:6.3f}]")
    print(f"  Board rotation axis: [{axis_B[0]:6.3f}, {axis_B[1]:6.3f}, {axis_B[2]:6.3f}]")

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
    
    # Method 1: Try XBX⁻¹
    X_inv = np.linalg.inv(T_hand_eye)
    T_predicted_1 = T_hand_eye @ T_delta_B_camera_frame @ X_inv
    error_1_rot = rotation_error(T_predicted_1[:3,:3], R_A)
    error_1_trans = np.linalg.norm(T_predicted_1[:3,3] - t_A)
    
    print(f"  Method 1 (XBX⁻¹):   rot={error_1_rot:.3f}°, trans={error_1_trans:.2f}mm")
    
    # Method 2: Try X⁻¹AX (predicting B from A)
    T_predicted_2 = X_inv @ T_delta_A_world_frame @ T_hand_eye
    error_2_rot = rotation_error(T_predicted_2[:3,:3], R_B)
    error_2_trans = np.linalg.norm(T_predicted_2[:3,3] - t_B)
    
    print(f"  Method 2 (X⁻¹AX):   rot={error_2_rot:.3f}°, trans={error_2_trans:.2f}mm (predicting B)")
    
    print(f"\n🎯 BEST METHOD: ", end="")
    errors = [
        (1, error_1_rot, error_1_trans, "XBX⁻¹ (current)"),
        (2, error_2_rot, error_2_trans, "X⁻¹AX (predicting B)"),
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


def parse_poses_from_json(json_path: str) -> tuple:
    """
    Parse poses from a JSON file.

    Args:
        json_path: Path to JSON file containing poses and optional reference_pose

    Returns:
        Tuple of (poses_list, reference_pose_dict)

    Example JSON format (old):
        [
          {"x": 100, "y": 200, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 45},
          {"x": 150, "y": 100, "z": 250, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 90}
        ]
    
    Example JSON format (new):
        {
          "reference_pose": {"x": 400, "y": 0, "z": -25, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 0},
          "poses": [
            {"x": 100, "y": 200, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 45},
            {"x": 150, "y": 100, "z": 250, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 90}
          ]
        }
    """
    if json_path is None:
        return None, None
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Poses file not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle both old format (list) and new format (dict with reference_pose and poses)
    if isinstance(data, list):
        poses = data
        reference_pose = None
    elif isinstance(data, dict):
        poses = data.get('poses', [])
        reference_pose = data.get('reference_pose', None)
    else:
        raise ValueError("JSON file must contain a list of pose objects or a dict with 'poses' and optional 'reference_pose'")

    # Validate each pose has required attributes
    required_attrs = ['x', 'y', 'z', 'o_x', 'o_y', 'o_z', 'theta']
    for i, pose in enumerate(poses):
        for attr in required_attrs:
            if attr not in pose:
                raise ValueError(f"Pose {i} missing required attribute '{attr}'")

    # Validate reference pose if provided
    if reference_pose is not None:
        for attr in required_attrs:
            if attr not in reference_pose:
                raise ValueError(f"Reference pose missing required attribute '{attr}'")

    return poses, reference_pose

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

async def connect(env_file: str):
    load_dotenv(env_file, override=True)
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
    return viam_client, robot

async def _get_current_arm_pose(motion: MotionClient, arm_name: str, arm: Arm) -> Tuple[Pose, Pose]:
    """
    Get current arm pose from both arm.get_end_position() and motion.get_pose().
    
    Returns:
        Tuple of (pose_from_arm, pose_from_motion_service)
    """
    pose_in_frame = await arm.get_end_position()
    pose_in_frame_motion_service = await motion.get_pose(
        component_name=arm_name,
        destination_frame="world"
    )
    return pose_in_frame, pose_in_frame_motion_service.pose

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
                                   chessboard_size, square_size=30.0, objp=None, data_dir=None, verbose=False):
    """
    Validate chessboard detection quality by computing reprojection error and sharpness.
    
    Args:
        objp: If provided, use this object points array. Otherwise generate from chessboard_size.
              This is important when corners have been filtered for outliers!
        data_dir: Directory to save histogram. If None, saves in current directory.
    
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
    
    # Count outliers with different thresholds (needed for histogram even if not verbose)
    outliers_1px = np.sum(errors > 1.0)
    outliers_2px = np.sum(errors > 2.0)
    outliers_5px = np.sum(errors > 5.0)
    
    if verbose:
        print(f"Reprojection error (OpenCV method): {mean_error:.3f} pixels")
        print(f"Reprojection error (mean): {mean_error2:.3f} pixels")
        print(f"Reprojection error (3σ filtered): {mean_error_filtered_3std:.3f} pixels ({len(filtered_errors_3std)}/{len(errors)} points)")
        print(f"Reprojection error (95th percentile): {mean_error_filtered_95th:.3f} pixels ({len(filtered_errors_95th)}/{len(errors)} points)")
        print(f"Reprojection error (<2px): {mean_error_filtered_abs:.3f} pixels ({len(filtered_errors_abs)}/{len(errors)} points)")
        print(f"Max individual error: {max_error2:.3f} pixels")
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
        histogram_filename = f"reprojection_error_histogram_{timestamp}.png"
        
        if data_dir:
            # Create subdirectory for reprojection error histograms
            histograms_dir = os.path.join(data_dir, "reprojection_error_histograms")
            os.makedirs(histograms_dir, exist_ok=True)
            histogram_path = os.path.join(histograms_dir, histogram_filename)
        else:
            histogram_path = histogram_filename
            
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"Saved reprojection error histogram: {histogram_path}")
        
        plt.close()  # Close to free memory
        
    except ImportError:
        print("Matplotlib not available, skipping histogram")
    except Exception as e:
        print(f"Failed to create histogram: {e}")
    
    return mean_error, max_error2, reprojected_points, errors


def get_chessboard_pose_in_camera_frame(image, camera_matrix, dist_coeffs, chessboard_size, 
                                       square_size=30.0, pnp_method=cv2.SOLVEPNP_IPPE, 
                                       use_sb_detection=True, data_dir=None, verbose=False):
    """
    Get chessboard pose in camera frame using PnP with improved outlier filtering.
    
    Args:
        data_dir: Directory to save histogram. If None, saves in current directory.
    
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
        if verbose:
            print("Using findChessboardCornersSB (subpixel detection)")
        flags = (cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, flags=flags)
    else:
        if verbose:
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
            if verbose:
                print(f"Chessboard sharpness: {sharpness:.2f} pixels")
        except Exception as e:
            if verbose:
                print(f"Could not estimate sharpness: {e}")
            sharpness = float('inf')  # Mark as unknown
        
        
        # Filter outliers before solvePnP with IMPROVED thresholds
        if verbose:
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
                if n_filtered > 0 and verbose:
                    print(f"Filtering {n_filtered} outliers (threshold: {threshold:.2f}px)")
                    print(f"  Error range: {np.min(errors):.3f} to {np.max(errors):.3f}px")
                    print(f"  Median: {median_error:.3f}px, MAD: {mad:.3f}px")
                
                if np.sum(good_indices) >= 20:  # Need at least 20 points for reliable PnP
                    filtered_corners = corners[good_indices]
                    filtered_objp = objp[good_indices]
                    if verbose:
                        print(f"Filtered corners: {len(filtered_corners)}/{len(corners)} points (error < {threshold:.2f}px)")
                    
                    # Use filtered points for final solvePnP
                    success, rvec, tvec = cv2.solvePnP(filtered_objp, filtered_corners, camera_matrix, dist_coeffs, flags=pnp_method)
                    corners = filtered_corners  # Update corners for validation
                    objp = filtered_objp  # Update objp for validation
                else:
                    if verbose:
                        print(f"Not enough good points ({np.sum(good_indices)}), using all points")
                    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)
            else:
                if verbose:
                    print("Initial solvePnP failed, using all points")
                success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)
                
        except Exception as e:
            if verbose:
                print(f"Outlier filtering failed: {e}, using all points")
            success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)

        if not success:
            if verbose:
                print("Failed to solve PnP")
            return False, None, None, None, None

        # Enhanced refinement with VVS
        rvec, tvec = cv2.solvePnPRefineVVS(objp, corners, camera_matrix, dist_coeffs, rvec, tvec)
        
        # Additional refinement with LM method
        rvec, tvec = cv2.solvePnPRefineLM(objp, corners, camera_matrix, dist_coeffs, rvec, tvec)
        
        # Validate detection quality
        mean_error, max_error, reprojected_points, errors = validate_chessboard_detection(
            image, corners, rvec, tvec, camera_matrix, dist_coeffs, chessboard_size, square_size, objp, data_dir, verbose=verbose
        )
        
        if verbose:
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
                                   aruco_id=0, aruco_size=200.0, aruco_dict='6X6_250', pnp_method='IPPE_SQUARE', use_sb_detection=True, data_dir=None, verbose=False):
    """
    Get marker pose in camera frame using PnP.
    Supports both chessboard and ArUco markers.
    
    Args:
        data_dir: Directory to save histogram. If None, saves in current directory.
    
    Returns: (success, rotation_vector, translation_vector, corners, marker_info)
    """
    if marker_type == 'chessboard':
        pnp_method_const = get_pnp_method_constant(pnp_method)
        return get_chessboard_pose_in_camera_frame(image, camera_matrix, dist_coeffs, chessboard_size, square_size, pnp_method=pnp_method_const, use_sb_detection=use_sb_detection, data_dir=data_dir, verbose=verbose)
    elif marker_type == 'aruco':
        aruco_dict_const = get_aruco_dict_constant(aruco_dict)
        pnp_method_const = get_pnp_method_constant(pnp_method)
        return get_aruco_pose_in_camera_frame(image, camera_matrix, dist_coeffs, 
                                            marker_id=aruco_id, marker_size=aruco_size, aruco_dict=aruco_dict_const, pnp_method=pnp_method_const)
    else:
        raise ValueError(f"Unknown marker type: {marker_type}")


def load_hand_eye_transform_from_json(json_path: str) -> np.ndarray:
    """
    Load hand-eye transformation matrix from a JSON file.
    
    The JSON file can contain either:
    1. A direct 4x4 transformation matrix as a list of lists:
       [[1.0, 0.0, 0.0, 80.0],
        [0.0, 1.0, 0.0, -20.2],
        [0.0, 0.0, 1.0, 48.7],
        [0.0, 0.0, 0.0, 1.0]]
    
    2. A frame configuration format (same as Viam frame config):
       {
         "translation": {"x": 80.0, "y": -20.2, "z": 48.7},
         "orientation": {
           "value": {"x": 0.0, "y": 0.0, "z": 1.0, "theta": 0.0},
           "type": "ov_degrees"
         }
       }
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        4x4 numpy array representing the transformation matrix
    """
    import json
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check if it's a direct 4x4 matrix
    if isinstance(data, list) and len(data) == 4:
        if all(isinstance(row, list) and len(row) == 4 for row in data):
            T = np.array(data, dtype=np.float64)
            return T
    
    # Check if it's a frame configuration format
    if isinstance(data, dict):
        if 'translation' in data or 'orientation' in data:
            T = frame_config_to_transformation_matrix(data)
            return T

    
    
    raise ValueError(f"Invalid JSON format in {json_path}. Expected either a 4x4 matrix (list of lists) or a frame configuration dict with 'translation' and 'orientation' keys.")


async def get_hand_eye_from_machine(app_client: AppClient, camera_name: str):
    """Get the hand-eye transformation from the machine's frame configuration"""

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


        parent = frame_config.get('parent', 'unknown')
        translation = frame_config.get('translation', {})
        T_hand_eye = frame_config_to_transformation_matrix(frame_config)
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

async def perform_pose_measurement(camera, camera_matrix, dist_coeffs, marker_type, 
                                 aruco_id, aruco_size, aruco_dict, pnp_method, 
                                 use_sb_detection, data_dir, measurement_num, 
                                 T_hand_eye, T_A_0_world_frame, T_B_0_camera_frame, 
                                 A_0_pose_world_frame_raw, motion_service, arm_name, arm,
                                 chessboard_cols=11, chessboard_rows=8, chessboard_square_size=30.0):
    """
    Perform a single pose measurement and return all relevant data.
    
    Returns:
        dict with all measurement data including errors, sharpness, etc.
    """
    # Get camera image
    image = await get_camera_image(camera)
    
    # Get marker pose (silently, for data collection)
    success, rvec, tvec, corners, marker_info = get_marker_pose_in_camera_frame(
        image, camera_matrix, dist_coeffs, marker_type=marker_type,
        chessboard_size=(chessboard_cols, chessboard_rows), square_size=chessboard_square_size,
        aruco_id=aruco_id, aruco_size=aruco_size, aruco_dict=aruco_dict, 
        pnp_method=pnp_method, use_sb_detection=use_sb_detection, data_dir=data_dir, verbose=False
    )
    
    if not success:
        return None
    
    # Get current arm pose for hand-eye verification
    A_i_pose_world_frame_raw, A_i_pose_world_frame_raw_from_motion_service = await _get_current_arm_pose(motion_service, arm_name, arm)
    
    # Compare poses from arm vs motion service (silently, for data collection)
    pose_comparison = compare_poses(
        A_i_pose_world_frame_raw, 
        A_i_pose_world_frame_raw_from_motion_service,
        label=f"Pose Comparison - Measurement {measurement_num}",
        verbose=False
    )
    
    # Invert only the rotation, keep translation unchanged
    A_i_pose_world_frame = _invert_pose_rotation_only(A_i_pose_world_frame_raw)
    A_i_pose_world_frame_from_motion_service = _invert_pose_rotation_only(A_i_pose_world_frame_raw_from_motion_service)
    T_A_i_world_frame = _pose_to_matrix(A_i_pose_world_frame)
    T_A_i_world_frame_from_motion_service = _pose_to_matrix(A_i_pose_world_frame_from_motion_service)
    
    # Convert chessboard pose (don't transpose - solvePnP is OpenCV convention)
    T_B_i_camera_frame = rvec_tvec_to_matrix(rvec, tvec)
    
    # Compute hand-eye verification errors for this measurement
    T_delta_A_world_frame = np.linalg.inv(T_A_i_world_frame) @ T_A_0_world_frame
    T_delta_B_camera_frame = T_B_i_camera_frame @ np.linalg.inv(T_B_0_camera_frame)
    
    hand_eye_errors = compute_hand_eye_verification_errors(
        T_hand_eye,
        T_delta_A_world_frame,
        T_delta_B_camera_frame
    )
    
    # Extract measurement data
    measurement_data = {
        'measurement_num': measurement_num,
        'success': success,
        'rvec': rvec.tolist() if rvec is not None else None,
        'tvec': tvec.tolist() if tvec is not None else None,
        'corners_count': len(corners) if corners is not None else 0,
        'filtered_corners_count': len(corners) if corners is not None else 0,  # Will be updated by validation
        'hand_eye_errors': hand_eye_errors
    }
    
    # Add validation info if available
    if marker_info:
        measurement_data.update({
            'mean_reprojection_error': float(marker_info.get('mean_reprojection_error', 0.0)),
            'max_reprojection_error': float(marker_info.get('max_reprojection_error', 0.0)),
            'sharpness': float(marker_info.get('sharpness', float('inf'))) if marker_info.get('sharpness') != float('inf') else float('inf')
            # Note: Individual corner errors (one per detected corner) are not stored to reduce file size.
            # Only aggregated statistics (mean, max) are saved.
        })
    
    return measurement_data

def calculate_measurement_statistics(measurements):
    """
    Calculate statistics from multiple measurements of the same pose.
    
    Args:
        measurements: List of measurement dictionaries
        
    Returns:
        dict with averaged statistics
    """
    if not measurements or len(measurements) == 0:
        return None
    
    # Filter out failed measurements
    valid_measurements = [m for m in measurements if m and m.get('success', False)]
    
    if len(valid_measurements) == 0:
        return None
    
    # Calculate averages for numerical values
    stats = {
        'num_measurements': len(valid_measurements),
        'success_rate': len(valid_measurements) / len(measurements),
    }
    
    # Average reprojection errors
    mean_errors = [m.get('mean_reprojection_error', 0) for m in valid_measurements]
    max_errors = [m.get('max_reprojection_error', 0) for m in valid_measurements]
    sharpness_values = [m.get('sharpness', float('inf')) for m in valid_measurements if m.get('sharpness') != float('inf')]
    
    # Hand-eye errors
    rotation_errors = [m.get('hand_eye_errors', {}).get('rotation_error', 0) for m in valid_measurements]
    translation_errors = [m.get('hand_eye_errors', {}).get('translation_error', 0) for m in valid_measurements]
    
    if mean_errors:
        stats.update({
            'mean_reprojection_error_avg': float(np.mean(mean_errors)),
            'mean_reprojection_error_std': float(np.std(mean_errors)),
            'mean_reprojection_error_min': float(np.min(mean_errors)),
            'mean_reprojection_error_max': float(np.max(mean_errors)),
        })
    
    if max_errors:
        stats.update({
            'max_reprojection_error_avg': float(np.mean(max_errors)),
            'max_reprojection_error_std': float(np.std(max_errors)),
            'max_reprojection_error_min': float(np.min(max_errors)),
            'max_reprojection_error_max': float(np.max(max_errors)),
        })
    
    if sharpness_values:
        stats.update({
            'sharpness_avg': float(np.mean(sharpness_values)),
            'sharpness_std': float(np.std(sharpness_values)),
            'sharpness_min': float(np.min(sharpness_values)),
            'sharpness_max': float(np.max(sharpness_values)),
        })
    
    # Average corner counts
    corner_counts = [m.get('corners_count', 0) for m in valid_measurements]
    if corner_counts:
        stats.update({
            'corners_count_avg': float(np.mean(corner_counts)),
            'corners_count_std': float(np.std(corner_counts)),
            'corners_count_min': float(np.min(corner_counts)),
            'corners_count_max': float(np.max(corner_counts)),
        })
    
    # Hand-eye error statistics
    if rotation_errors:
        stats.update({
            'rotation_error_avg': float(np.mean(rotation_errors)),
            'rotation_error_std': float(np.std(rotation_errors)),
            'rotation_error_min': float(np.min(rotation_errors)),
            'rotation_error_max': float(np.max(rotation_errors)),
        })
    
    if translation_errors:
        stats.update({
            'translation_error_avg': float(np.mean(translation_errors)),
            'translation_error_std': float(np.std(translation_errors)),
            'translation_error_min': float(np.min(translation_errors)),
            'translation_error_max': float(np.max(translation_errors)),
        })
    
    return stats

def generate_comprehensive_statistics(rotation_data):
    """
    Generate comprehensive statistics from all pose data.
    
    Note: Uses ALL individual measurements (not per-pose averages).
    If a pose has 3 measurements, all 3 are included in statistics.
    This shows measurement variability and gives more data points.
    
    Args:
        rotation_data: List of pose data dictionaries
        
    Returns:
        dict with comprehensive statistics (aggregated across all individual measurements)
    """
    if not rotation_data:
        return {
            'total_poses': 0,
            'successful_poses': 0,
            'success_rate': 0.0,
            'hand_eye': {},
            'reprojection': {},
            'detection': {}
        }
    
    # Basic counts
    total_poses = len(rotation_data)
    successful_poses = len([p for p in rotation_data if p.get('hand_eye_errors')])
    success_rate = successful_poses / total_poses if total_poses > 0 else 0.0
    
    # Collect all data for statistics
    rotation_errors = []
    translation_errors = []
    mean_reprojection_errors = []
    max_reprojection_errors = []
    sharpness_values = []
    corner_counts = []
    
    # Collect pose comparison errors from compare_poses results
    commanded_vs_arm_translation_errors = []
    commanded_vs_arm_rotation_errors = []
    commanded_vs_motion_translation_errors = []
    commanded_vs_motion_rotation_errors = []
    arm_vs_motion_translation_errors = []
    arm_vs_motion_rotation_errors = []
    
    # Collect temperature data
    rgb_temps = []
    main_board_temps = []
    chip_bottom_temps = []
    ir_left_temps = []
    ir_right_temps = []
    chip_top_temps = []
    cpu_temps = []
    
    for pose_data in rotation_data:
        # Collect data from all individual measurements (not per-pose averages)
        if 'measurements' in pose_data:
            for measurement in pose_data['measurements']:
                if measurement and measurement.get('success', False):
                    # Hand-eye errors from individual measurements
                    if 'hand_eye_errors' in measurement:
                        hand_eye = measurement['hand_eye_errors']
                        rotation_errors.append(hand_eye.get('rotation_error', 0))
                        translation_errors.append(hand_eye.get('translation_error', 0))
                    
                    # Reprojection errors from individual measurements
                    if 'mean_reprojection_error' in measurement:
                        mean_reprojection_errors.append(measurement['mean_reprojection_error'])
                    if 'max_reprojection_error' in measurement:
                        max_reprojection_errors.append(measurement['max_reprojection_error'])
                    
                    # Sharpness from individual measurements
                    if 'sharpness' in measurement and measurement['sharpness'] != float('inf'):
                        sharpness_values.append(measurement['sharpness'])
                    
                    # Corner counts from individual measurements
                    if 'corners_count' in measurement:
                        corner_counts.append(measurement['corners_count'])
        
        # Extract pose comparison data from compare_poses results
        pose_comparisons = pose_data.get('pose_comparisons', {})
        
        if pose_comparisons:
            # Commanded vs Arm comparison
            cmd_vs_arm = pose_comparisons.get('commanded_vs_arm_after_move', {})
            if cmd_vs_arm:
                if 'translation_error_mm' in cmd_vs_arm:
                    commanded_vs_arm_translation_errors.append(cmd_vs_arm['translation_error_mm'])
                if 'rotation_error_deg' in cmd_vs_arm:
                    commanded_vs_arm_rotation_errors.append(cmd_vs_arm['rotation_error_deg'])
            
            # Commanded vs Motion Service comparison
            cmd_vs_motion = pose_comparisons.get('commanded_vs_motion_service', {})
            if cmd_vs_motion:
                if 'translation_error_mm' in cmd_vs_motion:
                    commanded_vs_motion_translation_errors.append(cmd_vs_motion['translation_error_mm'])
                if 'rotation_error_deg' in cmd_vs_motion:
                    commanded_vs_motion_rotation_errors.append(cmd_vs_motion['rotation_error_deg'])
            
            # Arm vs Motion Service comparison
            arm_vs_motion = pose_comparisons.get('arm_after_move_vs_motion_service', {})
            if arm_vs_motion:
                if 'translation_error_mm' in arm_vs_motion:
                    arm_vs_motion_translation_errors.append(arm_vs_motion['translation_error_mm'])
                if 'rotation_error_deg' in arm_vs_motion:
                    arm_vs_motion_rotation_errors.append(arm_vs_motion['rotation_error_deg'])
        
        # Collect temperature data from measurements
        if 'measurements' in pose_data:
            for measurement in pose_data['measurements']:
                if 'camera_temperature' in measurement:
                    temp = measurement['camera_temperature']
                    if 'rgb_temp_c' in temp:
                        rgb_temps.append(temp['rgb_temp_c'])
                    if 'main_board_temp_c' in temp:
                        main_board_temps.append(temp['main_board_temp_c'])
                    if 'chip_bottom_temp_c' in temp:
                        chip_bottom_temps.append(temp['chip_bottom_temp_c'])
                    if 'ir_left_temp_c' in temp:
                        ir_left_temps.append(temp['ir_left_temp_c'])
                    if 'ir_right_temp_c' in temp:
                        ir_right_temps.append(temp['ir_right_temp_c'])
                    if 'chip_top_temp_c' in temp:
                        chip_top_temps.append(temp['chip_top_temp_c'])
                    if 'cpu_temp_c' in temp:
                        cpu_temps.append(temp['cpu_temp_c'])
    
    def calculate_stats(values):
        """Calculate mean, std, min, max for a list of values"""
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    statistics = {
        'total_poses': total_poses,
        'successful_poses': successful_poses,
        'success_rate': success_rate,
        'hand_eye': {
            'rotation_error': calculate_stats(rotation_errors),
            'translation_error': calculate_stats(translation_errors)
        },
        'reprojection': {
            'mean_error': calculate_stats(mean_reprojection_errors),
            'max_error': calculate_stats(max_reprojection_errors)
        },
        'detection': {
            'sharpness': calculate_stats(sharpness_values),
            'corners': calculate_stats(corner_counts)
        },
        'pose_comparisons': {
            'commanded_vs_arm': {
                'translation_error_mm': calculate_stats(commanded_vs_arm_translation_errors),
                'rotation_error_deg': calculate_stats(commanded_vs_arm_rotation_errors)
            },
            'commanded_vs_motion_service': {
                'translation_error_mm': calculate_stats(commanded_vs_motion_translation_errors),
                'rotation_error_deg': calculate_stats(commanded_vs_motion_rotation_errors)
            },
            'arm_vs_motion_service': {
                'translation_error_mm': calculate_stats(arm_vs_motion_translation_errors),
                'rotation_error_deg': calculate_stats(arm_vs_motion_rotation_errors)
            }
        },
        'camera_temperature': {
            'rgb_temp_c': calculate_stats(rgb_temps),
            'main_board_temp_c': calculate_stats(main_board_temps),
            'chip_bottom_temp_c': calculate_stats(chip_bottom_temps),
            'ir_left_temp_c': calculate_stats(ir_left_temps),
            'ir_right_temp_c': calculate_stats(ir_right_temps),
            'chip_top_temp_c': calculate_stats(chip_top_temps),
            'cpu_temp_c': calculate_stats(cpu_temps)
        }
    }
    
    return statistics

def create_comprehensive_statistics_plot(rotation_data, data_dir, tag=None):
    """
    Create comprehensive statistics plots for all pose data.
    
    Args:
        rotation_data: List of pose data dictionaries
        data_dir: Directory to save the plot
    """
    if not rotation_data:
        print("No data available for plotting")
        return
    
    # Collect all measurement data for plotting
    all_rotation_errors = []
    all_translation_errors = []
    all_mean_reprojection_errors = []
    all_max_reprojection_errors = []
    all_sharpness_values = []
    all_corner_counts = []
    pose_indices = []
    
    # Collect pose comparison data from compare_poses results
    all_cmd_vs_arm_translation = []
    all_cmd_vs_arm_rotation = []
    all_cmd_vs_motion_translation = []
    all_cmd_vs_motion_rotation = []
    all_arm_vs_motion_translation = []
    all_arm_vs_motion_rotation = []
    
    for i, pose_data in enumerate(rotation_data):
        if 'measurements' in pose_data:
            pose_indices.append(pose_data.get('pose_index', i))
            
            # Extract pose comparison data
            pose_comparisons = pose_data.get('pose_comparisons', {})
            if pose_comparisons:
                # Commanded vs Arm
                cmd_vs_arm = pose_comparisons.get('commanded_vs_arm_after_move', {})
                if cmd_vs_arm:
                    if 'translation_error_mm' in cmd_vs_arm:
                        all_cmd_vs_arm_translation.append(cmd_vs_arm['translation_error_mm'])
                    if 'rotation_error_deg' in cmd_vs_arm:
                        all_cmd_vs_arm_rotation.append(cmd_vs_arm['rotation_error_deg'])
                
                # Commanded vs Motion Service
                cmd_vs_motion = pose_comparisons.get('commanded_vs_motion_service', {})
                if cmd_vs_motion:
                    if 'translation_error_mm' in cmd_vs_motion:
                        all_cmd_vs_motion_translation.append(cmd_vs_motion['translation_error_mm'])
                    if 'rotation_error_deg' in cmd_vs_motion:
                        all_cmd_vs_motion_rotation.append(cmd_vs_motion['rotation_error_deg'])
                
                # Arm vs Motion Service
                arm_vs_motion = pose_comparisons.get('arm_after_move_vs_motion_service', {})
                if arm_vs_motion:
                    if 'translation_error_mm' in arm_vs_motion:
                        all_arm_vs_motion_translation.append(arm_vs_motion['translation_error_mm'])
                    if 'rotation_error_deg' in arm_vs_motion:
                        all_arm_vs_motion_rotation.append(arm_vs_motion['rotation_error_deg'])
            
            # Collect individual measurement data
            for measurement in pose_data['measurements']:
                if measurement and measurement.get('success', False):
                    if 'hand_eye_errors' in measurement:
                        all_rotation_errors.append(measurement['hand_eye_errors']['rotation_error'])
                        all_translation_errors.append(measurement['hand_eye_errors']['translation_error'])
                    if 'mean_reprojection_error' in measurement:
                        all_mean_reprojection_errors.append(measurement['mean_reprojection_error'])
                    if 'max_reprojection_error' in measurement:
                        all_max_reprojection_errors.append(measurement['max_reprojection_error'])
                    if 'sharpness' in measurement:
                        all_sharpness_values.append(measurement['sharpness'])
                    if 'corners_count' in measurement:
                        all_corner_counts.append(measurement['corners_count'])
    
    if not all_rotation_errors:
        print("No valid measurement data found for plotting")
        return
    
    # Create comprehensive statistics plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    title = 'Comprehensive Pose Test Statistics'
    if tag:
        title += f' - {tag}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Hand-Eye Rotation Errors
    axes[0, 0].hist(all_rotation_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(all_rotation_errors), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(all_rotation_errors):.3f}°')
    axes[0, 0].set_xlabel('Rotation Error (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Hand-Eye Rotation Errors')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Hand-Eye Translation Errors
    axes[0, 1].hist(all_translation_errors, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(np.mean(all_translation_errors), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(all_translation_errors):.3f}mm')
    axes[0, 1].set_xlabel('Translation Error (mm)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Hand-Eye Translation Errors')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean Reprojection Errors
    axes[1, 0].hist(all_mean_reprojection_errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(np.mean(all_mean_reprojection_errors), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(all_mean_reprojection_errors):.3f}px')
    axes[1, 0].set_xlabel('Mean Reprojection Error (pixels)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Mean Reprojection Errors')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Max Reprojection Errors
    axes[1, 1].hist(all_max_reprojection_errors, bins=20, alpha=0.7, color='pink', edgecolor='black')
    axes[1, 1].axvline(np.mean(all_max_reprojection_errors), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(all_max_reprojection_errors):.3f}px')
    axes[1, 1].set_xlabel('Max Reprojection Error (pixels)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Max Reprojection Errors')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Sharpness Values
    axes[2, 0].hist(all_sharpness_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[2, 0].axvline(np.mean(all_sharpness_values), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(all_sharpness_values):.2f}px')
    axes[2, 0].set_xlabel('Sharpness (pixels)')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title('Chessboard Sharpness')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Corner Counts
    axes[2, 1].hist(all_corner_counts, bins=20, alpha=0.7, color='brown', edgecolor='black')
    axes[2, 1].axvline(np.mean(all_corner_counts), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(all_corner_counts):.1f}')
    axes[2, 1].set_xlabel('Corner Count')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].set_title('Detected Corners')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Create pose comparison plots if data is available
    if all_cmd_vs_arm_translation or all_arm_vs_motion_translation:
        # Create a new figure for pose comparison plots
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
        pose_title = 'Pose Comparison Statistics (Commanded vs Arm vs Motion Service)'
        if tag:
            pose_title += f' - {tag}'
        fig2.suptitle(pose_title, fontsize=14, fontweight='bold')
        
        # Row 1: Translation Errors
        # Commanded vs Arm
        if all_cmd_vs_arm_translation:
            axes2[0, 0].hist(all_cmd_vs_arm_translation, bins=20, alpha=0.7, color='darkgreen', edgecolor='black')
            axes2[0, 0].axvline(np.mean(all_cmd_vs_arm_translation), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {np.mean(all_cmd_vs_arm_translation):.3f}mm')
            axes2[0, 0].set_xlabel('Translation Error (mm)')
            axes2[0, 0].set_ylabel('Frequency')
            axes2[0, 0].set_title('Commanded vs Arm (Translation)')
            axes2[0, 0].legend()
            axes2[0, 0].grid(True, alpha=0.3)
        
        # Commanded vs Motion Service
        if all_cmd_vs_motion_translation:
            axes2[0, 1].hist(all_cmd_vs_motion_translation, bins=20, alpha=0.7, color='darkblue', edgecolor='black')
            axes2[0, 1].axvline(np.mean(all_cmd_vs_motion_translation), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {np.mean(all_cmd_vs_motion_translation):.3f}mm')
            axes2[0, 1].set_xlabel('Translation Error (mm)')
            axes2[0, 1].set_ylabel('Frequency')
            axes2[0, 1].set_title('Commanded vs Motion Service (Translation)')
            axes2[0, 1].legend()
            axes2[0, 1].grid(True, alpha=0.3)
        
        # Arm vs Motion Service
        if all_arm_vs_motion_translation:
            axes2[0, 2].hist(all_arm_vs_motion_translation, bins=20, alpha=0.7, color='darkorange', edgecolor='black')
            axes2[0, 2].axvline(np.mean(all_arm_vs_motion_translation), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {np.mean(all_arm_vs_motion_translation):.3f}mm')
            axes2[0, 2].set_xlabel('Translation Error (mm)')
            axes2[0, 2].set_ylabel('Frequency')
            axes2[0, 2].set_title('Arm vs Motion Service (Translation)')
            axes2[0, 2].legend()
            axes2[0, 2].grid(True, alpha=0.3)
        
        # Row 2: Rotation Errors
        # Commanded vs Arm
        if all_cmd_vs_arm_rotation:
            axes2[1, 0].hist(all_cmd_vs_arm_rotation, bins=20, alpha=0.7, color='darkgreen', edgecolor='black')
            axes2[1, 0].axvline(np.mean(all_cmd_vs_arm_rotation), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {np.mean(all_cmd_vs_arm_rotation):.3f}°')
            axes2[1, 0].set_xlabel('Rotation Error (degrees)')
            axes2[1, 0].set_ylabel('Frequency')
            axes2[1, 0].set_title('Commanded vs Arm (Rotation)')
            axes2[1, 0].legend()
            axes2[1, 0].grid(True, alpha=0.3)
        
        # Commanded vs Motion Service
        if all_cmd_vs_motion_rotation:
            axes2[1, 1].hist(all_cmd_vs_motion_rotation, bins=20, alpha=0.7, color='darkblue', edgecolor='black')
            axes2[1, 1].axvline(np.mean(all_cmd_vs_motion_rotation), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {np.mean(all_cmd_vs_motion_rotation):.3f}°')
            axes2[1, 1].set_xlabel('Rotation Error (degrees)')
            axes2[1, 1].set_ylabel('Frequency')
            axes2[1, 1].set_title('Commanded vs Motion Service (Rotation)')
            axes2[1, 1].legend()
            axes2[1, 1].grid(True, alpha=0.3)
        
        # Arm vs Motion Service
        if all_arm_vs_motion_rotation:
            axes2[1, 2].hist(all_arm_vs_motion_rotation, bins=20, alpha=0.7, color='darkorange', edgecolor='black')
            axes2[1, 2].axvline(np.mean(all_arm_vs_motion_rotation), color='red', linestyle='--', linewidth=2,
                            label=f'Mean: {np.mean(all_arm_vs_motion_rotation):.3f}°')
            axes2[1, 2].set_xlabel('Rotation Error (degrees)')
            axes2[1, 2].set_ylabel('Frequency')
            axes2[1, 2].set_title('Arm vs Motion Service (Rotation)')
            axes2[1, 2].legend()
            axes2[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the pose comparison plot
        pose_accuracy_file = os.path.join(data_dir, "pose_accuracy_statistics.png")
        plt.savefig(pose_accuracy_file, dpi=300, bbox_inches='tight')
        print(f"Pose comparison statistics plot saved to: {pose_accuracy_file}")
        
        plt.close(fig2)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(data_dir, "comprehensive_statistics.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comprehensive statistics plot saved to: {plot_file}")
    
    # Also create a summary statistics table plot
    create_summary_table_plot(rotation_data, data_dir, tag)
    
    # Create pose-by-pose error plot
    create_pose_error_plot(rotation_data, data_dir, tag)
    
    plt.close()

def create_pose_error_plot(rotation_data, data_dir, tag=None):
    """
    Create a plot showing rotational and translational errors for each pose index with temperature overlay.
    
    Args:
        rotation_data: List of pose data dictionaries
        data_dir: Directory to save the plot
        tag: Optional tag to add to the plot title
    """
    if not rotation_data:
        print("No valid measurement data found for pose error plotting")
        return
    
    # Extract pose indices, errors, and temperatures
    pose_indices = []
    rotation_errors = []
    translation_errors = []
    rgb_temps = []
    main_board_temps = []

    for i, pose_data in enumerate(rotation_data):
        # Determine index robustly
        pose_index = pose_data.get('pose_index', i)

        # Prefer aggregated stats if present
        rot_avg = None
        trans_avg = None
        stats = pose_data.get('measurement_statistics', {})

        # Newer format uses *_avg keys
        if 'rotation_error_avg' in stats and 'translation_error_avg' in stats:
            rot_avg = stats['rotation_error_avg']
            trans_avg = stats['translation_error_avg']
        # Older/nested format fallback (not currently used, but keep for resiliency)
        elif 'rotation_error' in stats and isinstance(stats['rotation_error'], dict):
            rot_avg = stats['rotation_error'].get('avg')
            trans_avg = stats.get('translation_error', {}).get('avg')
        # Final fallback: compute from per-measurement data
        if rot_avg is None or trans_avg is None:
            if 'measurements' in pose_data and pose_data['measurements']:
                rot_list = []
                trans_list = []
                for m in pose_data['measurements']:
                    he = m.get('hand_eye_errors', {})
                    if 'rotation_error' in he:
                        rot_list.append(he['rotation_error'])
                    if 'translation_error' in he:
                        trans_list.append(he['translation_error'])
                if rot_avg is None and rot_list:
                    rot_avg = float(np.mean(rot_list))
                if trans_avg is None and trans_list:
                    trans_avg = float(np.mean(trans_list))

        # If we have both, record this pose
        if rot_avg is not None and trans_avg is not None:
            pose_indices.append(pose_index)
            rotation_errors.append(rot_avg)
            translation_errors.append(trans_avg)

            # Extract temperature data from measurements
            pose_rgb_temps = []
            pose_main_board_temps = []
            if 'measurements' in pose_data:
                for measurement in pose_data['measurements']:
                    if 'camera_temperature' in measurement:
                        temp = measurement['camera_temperature']
                        if 'rgb_temp_c' in temp:
                            try:
                                pose_rgb_temps.append(float(temp['rgb_temp_c']))
                            except Exception:
                                pass
                        if 'main_board_temp_c' in temp:
                            try:
                                pose_main_board_temps.append(float(temp['main_board_temp_c']))
                            except Exception:
                                pass

            # Use average temperature for this pose
            rgb_temps.append(np.mean(pose_rgb_temps) if pose_rgb_temps else 0.0)
            main_board_temps.append(np.mean(pose_main_board_temps) if pose_main_board_temps else 0.0)
    
    if not pose_indices:
        print("No valid pose error data found for plotting")
        return
    
    # Create figure with triple y-axes
    fig, ax1 = plt.subplots(figsize=(16, 10))
    
    # Plot rotation errors on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Pose Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rotation Error (degrees)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(pose_indices, rotation_errors, 'o-', color=color1, linewidth=2, 
                     markersize=6, label='Rotation Error', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for translation errors
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Translation Error (mm)', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(pose_indices, translation_errors, 's-', color=color2, linewidth=2, 
                     markersize=6, label='Translation Error', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Create third y-axis for temperature (if we have temperature data)
    if any(t > 0 for t in rgb_temps) or any(t > 0 for t in main_board_temps):
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
        
        # Plot RGB temperature
        if any(t > 0 for t in rgb_temps):
            color3 = 'tab:orange'
            ax3.set_ylabel('Temperature (°C)', color=color3, fontsize=12, fontweight='bold')
            line3 = ax3.plot(pose_indices, rgb_temps, '^-', color=color3, linewidth=2, 
                             markersize=5, label='RGB Temperature', alpha=0.7)
            ax3.tick_params(axis='y', labelcolor=color3)
        
        # Plot main board temperature on the same axis
        if any(t > 0 for t in main_board_temps):
            color4 = 'tab:green'
            line4 = ax3.plot(pose_indices, main_board_temps, 'v-', color=color4, linewidth=2, 
                             markersize=5, label='Main Board Temperature', alpha=0.7)
    
    # Add horizontal lines for mean values
    mean_rotation = np.mean(rotation_errors)
    mean_translation = np.mean(translation_errors)
    
    ax1.axhline(y=mean_rotation, color=color1, linestyle='--', alpha=0.7, 
                label=f'Mean Rotation: {mean_rotation:.3f}°')
    ax2.axhline(y=mean_translation, color=color2, linestyle='--', alpha=0.7, 
                label=f'Mean Translation: {mean_translation:.3f}mm')
    
    # Set title
    title = 'Pose-by-Pose Hand-Eye Calibration Errors with Temperature'
    if tag:
        title += f' - {tag}'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Add temperature lines to legend if they exist
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    
    if any(t > 0 for t in rgb_temps) or any(t > 0 for t in main_board_temps):
        lines3, labels3 = ax3.get_legend_handles_labels()
        all_lines.extend(lines3)
        all_labels.extend(labels3)
    
    ax1.legend(all_lines, all_labels, loc='upper right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(data_dir, "pose_error_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Pose error analysis plot saved to: {plot_file}")
    
    plt.close()

def create_summary_table_plot(rotation_data, data_dir, tag=None):
    """
    Create a summary statistics table plot.
    
    Args:
        rotation_data: List of pose data dictionaries
        data_dir: Directory to save the plot
    """
    if not rotation_data:
        return
    
    # Calculate summary statistics
    stats = generate_comprehensive_statistics(rotation_data)
    
    # Create table data
    table_data = [
        ['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
        ['Rotation Error (°)', f"{stats['hand_eye']['rotation_error']['mean']:.3f}", 
         f"{stats['hand_eye']['rotation_error']['std']:.3f}",
         f"{stats['hand_eye']['rotation_error']['min']:.3f}",
         f"{stats['hand_eye']['rotation_error']['max']:.3f}"],
        ['Translation Error (mm)', f"{stats['hand_eye']['translation_error']['mean']:.3f}",
         f"{stats['hand_eye']['translation_error']['std']:.3f}",
         f"{stats['hand_eye']['translation_error']['min']:.3f}",
         f"{stats['hand_eye']['translation_error']['max']:.3f}"],
        ['Mean Reprojection (px)', f"{stats['reprojection']['mean_error']['mean']:.3f}",
         f"{stats['reprojection']['mean_error']['std']:.3f}",
         f"{stats['reprojection']['mean_error']['min']:.3f}",
         f"{stats['reprojection']['mean_error']['max']:.3f}"],
        ['Max Reprojection (px)', f"{stats['reprojection']['max_error']['mean']:.3f}",
         f"{stats['reprojection']['max_error']['std']:.3f}",
         f"{stats['reprojection']['max_error']['min']:.3f}",
         f"{stats['reprojection']['max_error']['max']:.3f}"],
        ['Sharpness (px)', f"{stats['detection']['sharpness']['mean']:.2f}",
         f"{stats['detection']['sharpness']['std']:.2f}",
         f"{stats['detection']['sharpness']['min']:.2f}",
         f"{stats['detection']['sharpness']['max']:.2f}"],
        ['Corner Count', f"{stats['detection']['corners']['mean']:.1f}",
         f"{stats['detection']['corners']['std']:.1f}",
         f"{stats['detection']['corners']['min']:.1f}",
         f"{stats['detection']['corners']['max']:.1f}"]
    ]
    
    # Add pose comparison data if available
    if 'pose_comparisons' in stats:
        comparisons = stats['pose_comparisons']
        
        # Commanded vs Arm
        if 'commanded_vs_arm' in comparisons:
            cmd_arm = comparisons['commanded_vs_arm']
            if 'translation_error_mm' in cmd_arm:
                table_data.append(['Cmd vs Arm Trans (mm)', f"{cmd_arm['translation_error_mm']['mean']:.3f}",
                                 f"{cmd_arm['translation_error_mm']['std']:.3f}",
                                 f"{cmd_arm['translation_error_mm']['min']:.3f}",
                                 f"{cmd_arm['translation_error_mm']['max']:.3f}"])
            if 'rotation_error_deg' in cmd_arm:
                table_data.append(['Cmd vs Arm Rot (°)', f"{cmd_arm['rotation_error_deg']['mean']:.3f}",
                                 f"{cmd_arm['rotation_error_deg']['std']:.3f}",
                                 f"{cmd_arm['rotation_error_deg']['min']:.3f}",
                                 f"{cmd_arm['rotation_error_deg']['max']:.3f}"])
        
        # Commanded vs Motion Service
        if 'commanded_vs_motion_service' in comparisons:
            cmd_motion = comparisons['commanded_vs_motion_service']
            if 'translation_error_mm' in cmd_motion:
                table_data.append(['Cmd vs Motion Trans (mm)', f"{cmd_motion['translation_error_mm']['mean']:.3f}",
                                 f"{cmd_motion['translation_error_mm']['std']:.3f}",
                                 f"{cmd_motion['translation_error_mm']['min']:.3f}",
                                 f"{cmd_motion['translation_error_mm']['max']:.3f}"])
            if 'rotation_error_deg' in cmd_motion:
                table_data.append(['Cmd vs Motion Rot (°)', f"{cmd_motion['rotation_error_deg']['mean']:.3f}",
                                 f"{cmd_motion['rotation_error_deg']['std']:.3f}",
                                 f"{cmd_motion['rotation_error_deg']['min']:.3f}",
                                 f"{cmd_motion['rotation_error_deg']['max']:.3f}"])
        
        # Arm vs Motion Service
        if 'arm_vs_motion_service' in comparisons:
            arm_motion = comparisons['arm_vs_motion_service']
            if 'translation_error_mm' in arm_motion:
                table_data.append(['Arm vs Motion Trans (mm)', f"{arm_motion['translation_error_mm']['mean']:.3f}",
                                 f"{arm_motion['translation_error_mm']['std']:.3f}",
                                 f"{arm_motion['translation_error_mm']['min']:.3f}",
                                 f"{arm_motion['translation_error_mm']['max']:.3f}"])
            if 'rotation_error_deg' in arm_motion:
                table_data.append(['Arm vs Motion Rot (°)', f"{arm_motion['rotation_error_deg']['mean']:.3f}",
                                 f"{arm_motion['rotation_error_deg']['std']:.3f}",
                                 f"{arm_motion['rotation_error_deg']['min']:.3f}",
                                 f"{arm_motion['rotation_error_deg']['max']:.3f}"])
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f1f1f2')
            else:
                table[(i, j)].set_facecolor('white')
    
    title = 'Comprehensive Statistics Summary'
    if tag:
        title += f' - {tag}'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Save the table plot
    table_file = os.path.join(data_dir, "statistics_summary_table.png")
    plt.savefig(table_file, dpi=300, bbox_inches='tight')
    print(f"Statistics summary table saved to: {table_file}")
    
    plt.close()

async def main(
    env_file: str,
    arm_name: str,
    pose_tracker_name: str,
    motion_service_name: str,
    camera_name: str,
    poses: list,
    reference_pose: dict = None,
    marker_type: str = 'chessboard',
    aruco_id: int = 0,
    aruco_size: float = 200.0,
    aruco_dict: str = '6X6_250',
    pnp_method: str = 'IPPE_SQUARE',
    use_sb_detection: bool = True,
    resume_from_pose: int = 1,
    data_dir: str = None,
    tag: str = None,
    hand_eye_transform_file: str = None,
    chessboard_cols: int = 11,
    chessboard_rows: int = 8,
    chessboard_square_size: float = 30.0,
    poses_file_path: str = None,
):
    app_client: Optional[AppClient] = None
    machine: Optional[RobotClient] = None
    pt: Optional[PoseTracker] = None
    arm: Optional[Arm] = None
    motion_service: Optional[MotionClient] = None
    camera: Optional[Camera] = None
    viam_client = None
    
    try:
        viam_client, machine = await connect(env_file)
        app_client = viam_client.app_client
        arm = Arm.from_robot(machine, arm_name)
        await arm.do_command({"set_vel": 25})
        camera = Camera.from_robot(machine, camera_name)
        motion_service = MotionClient.from_robot(machine, motion_service_name)
        pt = PoseTracker.from_robot(machine, pose_tracker_name)


        # Get the hand-eye transformation from camera configuration or manual JSON file
        if hand_eye_transform_file:
            try:
                T_hand_eye = load_hand_eye_transform_from_json(hand_eye_transform_file)
            except Exception as e:
                print(f"ERROR: Failed to load hand-eye transformation from file: {e}")
                return
        else:
            T_hand_eye = await get_hand_eye_from_machine(app_client, camera_name)
            if T_hand_eye is None:
                print("ERROR: Could not retrieve hand-eye transformation")
                return

        # Check if we're resuming (to skip reference pose movement)
        calibration_config_path_check = os.path.join(data_dir, "calibration_config.json") if data_dir else None
        is_resuming_check = resume_from_pose > 1
        has_existing_config_check = calibration_config_path_check and os.path.exists(calibration_config_path_check)
        
        # Get initial poses (skip if resuming and config exists)
        if not (is_resuming_check and has_existing_config_check):
            if reference_pose is not None:
                
                # Convert reference pose dict to Viam Pose
                reference_pose_viam = Pose(
                    x=reference_pose['x'],
                    y=reference_pose['y'], 
                    z=reference_pose['z'],
                    o_x=reference_pose['o_x'],
                    o_y=reference_pose['o_y'],
                    o_z=reference_pose['o_z'],
                    theta=reference_pose['theta']
                )

                # Pause before moving to reference pose
                input("Press Enter to move to the reference/base position...")
                
                # Move to reference pose
                reference_pose_in_frame = PoseInFrame(reference_frame=arm.name  + "_origin", pose=reference_pose_viam)
                await motion_service.move(component_name=arm.name, destination=reference_pose_in_frame)
                await asyncio.sleep(DEFAULT_SETTLE_TIME)  # Settle time
                # print(f"⏸️  PAUSING FOR EVALUATION - Press Enter to continue...")
                # input()  # Pause for user evaluation
                
                # Get the actual pose after movement
                A_0_pose_world_frame_raw, A_0_pose_world_frame_raw_from_motion_service = await _get_current_arm_pose(motion_service, arm.name, arm)
                
                # Compare poses from arm vs motion service
                pose_comparison = compare_poses(
                    A_0_pose_world_frame_raw,
                    A_0_pose_world_frame_raw_from_motion_service,
                    label="Pose Comparison - Reference Pose"
                )
            else:
                raise Exception("No reference pose provided. A reference pose is required for hand-eye calibration testing.")
        else:
            # When resuming, we'll load A_0_pose_world_frame_raw from config later
            # For now, just set it to None as a placeholder
            A_0_pose_world_frame_raw = None
            # Also set these as placeholders, will be loaded from config
            A_0_pose_world_frame = None
            T_A_0_world_frame = None
            T_A_0_world_frame_from_motion_service = None
        
        # Invert only the rotation, keep translation unchanged (only if not resuming)
        if A_0_pose_world_frame_raw is not None:
            A_0_pose_world_frame = _invert_pose_rotation_only(A_0_pose_world_frame_raw)
            T_A_0_world_frame = _pose_to_matrix(A_0_pose_world_frame)
            A_0_pose_world_frame_from_motion_service = _invert_pose_rotation_only(A_0_pose_world_frame_raw_from_motion_service)
            T_A_0_world_frame_from_motion_service = _pose_to_matrix(A_0_pose_world_frame_from_motion_service)
            
            # Compare the inverted poses as well (silently, for data collection)
            inverted_comparison = compare_poses(
                A_0_pose_world_frame,
                A_0_pose_world_frame_from_motion_service,
                label="Inverted Pose Comparison - Reference Pose"
            )

        camera_matrix, dist_coeffs = await get_camera_intrinsics(camera)
        
        # Create or use existing directory for saving data
        if data_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if tag:
                data_dir = f"calibration_data_{timestamp}_{tag}"
            else:
                data_dir = f"calibration_data_{timestamp}"
            os.makedirs(data_dir, exist_ok=True)
        else:
            # Use existing directory, create if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
        
        # Extract timestamp from existing directory name for consistency
        if data_dir.startswith("calibration_data_"):
            timestamp = data_dir.replace("calibration_data_", "")
            if "_" in timestamp:
                timestamp = timestamp.split("_")[0]  # Remove tag if present
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Copy poses JSON file to data directory if provided
        if poses_file_path and os.path.exists(poses_file_path):
            poses_file_dest = os.path.join(data_dir, os.path.basename(poses_file_path))
            try:
                shutil.copy2(poses_file_path, poses_file_dest)
            except Exception as e:
                print(f"Warning: Could not copy poses file: {e}")
        
        # Reconstruct command from sys.argv for logging
        # sys.argv[0] is the script name, sys.argv[1:] are the arguments
        # Use shlex.join to properly quote arguments with spaces (Python 3.8+)
        command_parts = [sys.executable] + sys.argv
        if hasattr(shlex, 'join'):
            command = shlex.join(command_parts)
        else:
            # Fallback for Python < 3.8: quote all parts and join
            command = " ".join(shlex.quote(part) for part in command_parts)
        
        # Setup logging
        log_file = setup_logging(data_dir, command=command)
        
        # Record start time
        start_time = datetime.now()
        
        # Check if we're resuming and if calibration config exists
        calibration_config_path = os.path.join(data_dir, "calibration_config.json")
        is_resuming = resume_from_pose > 1
        has_existing_config = os.path.exists(calibration_config_path)
        
        if is_resuming and has_existing_config:
            # Load existing reference pose data
            try:
                with open(calibration_config_path, 'r') as f:
                    existing_config = json.load(f)
                
                # Load A_0_pose_raw and reconstruct the Pose object
                A_0_raw_dict = existing_config.get('A_0_pose_raw', {})
                A_0_pose_world_frame_raw = Pose(
                    x=A_0_raw_dict['x'],
                    y=A_0_raw_dict['y'],
                    z=A_0_raw_dict['z'],
                    o_x=A_0_raw_dict['o_x'],
                    o_y=A_0_raw_dict['o_y'],
                    o_z=A_0_raw_dict['o_z'],
                    theta=A_0_raw_dict['theta']
                )
                
                # Load A_0_pose_inverted
                A_0_inv_dict = existing_config.get('A_0_pose_inverted', {})
                A_0_pose_world_frame = Pose(
                    x=A_0_inv_dict['x'],
                    y=A_0_inv_dict['y'],
                    z=A_0_inv_dict['z'],
                    o_x=A_0_inv_dict['o_x'],
                    o_y=A_0_inv_dict['o_y'],
                    o_z=A_0_inv_dict['o_z'],
                    theta=A_0_inv_dict['theta']
                )
                
                # Reconstruct T_A_0_world_frame and T_B_0_camera_frame
                T_A_0_world_frame = _pose_to_matrix(A_0_pose_world_frame)
                
                # When resuming, optionally re-fetch both poses for comparison
                # For actual processing, use the saved arm pose
                A_0_pose_world_frame_raw_recheck, A_0_pose_world_frame_raw_from_motion_service_recheck = await _get_current_arm_pose(motion_service, arm.name, arm)
                
                # Compare current poses (though robot may have moved)
                resume_comparison = compare_poses(
                    A_0_pose_world_frame_raw_recheck,
                    A_0_pose_world_frame_raw_from_motion_service_recheck,
                    label="Pose Comparison - Resume (Current Position)"
                )
                
                # Use the saved pose for processing, but note the motion service version for reference
                A_0_pose_world_frame_from_motion_service = _invert_pose_rotation_only(A_0_pose_world_frame_raw_from_motion_service_recheck)
                T_A_0_world_frame_from_motion_service = _pose_to_matrix(A_0_pose_world_frame_from_motion_service)
                
                T_B_0_camera_frame = np.array(existing_config.get('T_B_0_camera_frame', []))
                
                reference_camera_temperature = existing_config.get('reference_camera_temperature', {})
                
                # Reference pose loaded silently (data is logged to file)
                
            except Exception as e:
                print(f"⚠️  Warning: Could not load existing config: {e}")
                print(f"   Will capture new reference pose instead...")
                has_existing_config = False  # Fall through to capture new reference
        
        if not (is_resuming and has_existing_config):
            # Capture new reference pose (either not resuming or config doesn't exist)
            image = await get_camera_image(camera)

            # Capture camera temperature for reference pose
            try:
                temp_response = await camera.do_command({"get_camera_temperature": {}})
                reference_camera_temperature = temp_response.get("get_camera_temperature", {})
                # Temperature captured silently
            except Exception:
                pass  # Temperature capture failed silently
                reference_camera_temperature = {}

            success, rvec, tvec, _, marker_info = get_marker_pose_in_camera_frame(
                image, camera_matrix, dist_coeffs, marker_type=marker_type,
                chessboard_size=(chessboard_cols, chessboard_rows), square_size=chessboard_square_size,
                aruco_id=aruco_id, aruco_size=aruco_size, aruco_dict=aruco_dict, pnp_method=pnp_method, use_sb_detection=use_sb_detection, data_dir=data_dir, verbose=False
            )
            if not success:
                print(f"⚠️  Failed to detect {marker_type} in reference image")
                return
            
            # Show concise reference pose summary
            if marker_info:
                reproj_err = marker_info.get('mean_reprojection_error', 0)
                reproj_max = marker_info.get('max_reprojection_error', 0)
                sharpness = marker_info.get('sharpness', float('inf'))
                reproj_quality = "✅" if reproj_err < 0.5 else "⚠️" if reproj_err < 1.0 else "❌"
                sharpness_quality = "✅" if sharpness < 3.0 else "⚠️" if sharpness < 5.0 else "❌"
                print(f"  📊 Reference: Reproj: {reproj_quality} {reproj_err:.3f}px (max: {reproj_max:.3f}px) | Sharpness: {sharpness_quality} {sharpness:.2f}px")
            
            # Convert rvec, tvec to 4x4 transformation matrix (chessboard in camera frame)
            # Don't transpose - solvePnP output is already in OpenCV convention
            T_B_0_camera_frame = rvec_tvec_to_matrix(rvec, tvec)
            
            # Save camera calibration and chessboard config (only when capturing new reference)
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
                "reference_camera_temperature": reference_camera_temperature,
                "note": "Arm poses have rotation inverted (translation unchanged) before processing"
            }
            with open(os.path.join(data_dir, "calibration_config.json"), "w", encoding='utf-8') as f:
                json.dump(calibration_data, f, indent=2, ensure_ascii=False)
            
            # Save reference image
            # Save reference image with debug visualization
            debug_image = draw_marker_debug(image, rvec, tvec, camera_matrix, dist_coeffs, 
                                          marker_type=marker_type, chessboard_size=(chessboard_cols, chessboard_rows), 
                                          square_size=chessboard_square_size, aruco_size=aruco_size, validation_info=marker_info)
            cv2.imwrite(os.path.join(data_dir, "image_reference.jpg"), debug_image)
            print(f"✅ Reference pose saved")
        
        # List to store all rotation data
        rotation_data = []
        
        # If resuming, try to load existing pose data
        if resume_from_pose > 1 and data_dir is not None:
            pose_data_path = os.path.join(data_dir, "pose_data.json")
            if os.path.exists(pose_data_path):
                try:
                    with open(pose_data_path, 'r') as f:
                        data = json.load(f)
                    # Handle both old format (list) and new format (dict with 'poses' key)
                    if isinstance(data, list):
                        existing_poses = data
                    elif isinstance(data, dict) and 'poses' in data:
                        existing_poses = data['poses']
                    else:
                        print(f"Warning: Unexpected data format in {pose_data_path}, starting fresh")
                        existing_poses = []
                    
                    # Remove poses that will be re-measured (from resume_from_pose onwards)
                    # Keep only poses with pose_index < (resume_from_pose - 1) since pose_index is 0-based
                    rotation_data = [pose for pose in existing_poses 
                                   if pose.get('pose_index', 0) < (resume_from_pose - 1)]
                    
                except Exception as e:
                    print(f"Warning: Could not load existing pose data: {e}")
                    rotation_data = []
            else:
                print(f"Warning: No existing pose data found at {pose_data_path}, starting fresh")

        # Test the hand-eye transformation with provided poses
        if poses is None:
            raise Exception("No poses provided")

        await arm.do_command({"set_vel": DEFAULT_VELOCITY_SLOW})
        
        # Validate resume_from_pose parameter
        if resume_from_pose < 1 or resume_from_pose > len(poses):
            print(f"ERROR: resume_from_pose ({resume_from_pose}) must be between 1 and {len(poses)}")
            return
        
        # Calculate the actual start index (0-based)
        start_index = resume_from_pose - 1
        poses_to_test = poses[start_index:]
        
        
        for i, pose_spec in enumerate(poses_to_test):
            actual_pose_number = start_index + i + 1
            print(f"\nPOSE {actual_pose_number}/{len(poses)}")

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
            # Move to target pose
            target_pose_in_frame = PoseInFrame(reference_frame=arm.name + "_origin", pose=target_pose)
            await motion_service.move(component_name=arm.name, destination=target_pose_in_frame)
            await asyncio.sleep(DEFAULT_SETTLE_TIME)  # Increased settling time to reduce motion blur
            
            # Get actual arm pose after movement and compare with commanded pose
            A_i_pose_world_frame_raw_after_move, A_i_pose_world_frame_raw_from_motion_service_after_move = await _get_current_arm_pose(motion_service, arm.name, arm)
            
            # Compare commanded pose vs actual arm pose (silently, for data collection)
            commanded_vs_arm_comparison = compare_poses(
                target_pose,
                A_i_pose_world_frame_raw_after_move,
                label=f"Commanded vs Actual Pose - Pose {actual_pose_number}",
                verbose=False
            )
            
            # Perform 3 measurements for this pose
            measurements = []
            successful_count = 0
            for measurement_num in range(1, 4):
                # Capture camera temperature (silently)
                try:
                    temp_response = await camera.do_command({"get_camera_temperature": {}})
                    camera_temperature = temp_response.get("get_camera_temperature", {})
                except Exception:
                    camera_temperature = {}
                
                measurement = await perform_pose_measurement(
                    camera, camera_matrix, dist_coeffs, marker_type,
                    aruco_id, aruco_size, aruco_dict, pnp_method,
                    use_sb_detection, data_dir, measurement_num,
                    T_hand_eye, T_A_0_world_frame, T_B_0_camera_frame,
                    A_0_pose_world_frame_raw, motion_service, arm_name, arm,
                    chessboard_cols, chessboard_rows, chessboard_square_size
                )
                
                # Add temperature data to measurement if it succeeded
                if measurement is not None:
                    measurement['camera_temperature'] = camera_temperature
                    measurements.append(measurement)
                    if measurement.get('success', False):
                        successful_count += 1
                else:
                    # Create a failed measurement entry with temperature data
                    failed_measurement = {
                        'measurement_num': measurement_num,
                        'success': False,
                        'camera_temperature': camera_temperature
                    }
                    measurements.append(failed_measurement)
                
                # Small delay between measurements
                if measurement_num < 3:
                    await asyncio.sleep(1.0)
            
            # Calculate statistics from measurements
            measurement_stats = calculate_measurement_statistics(measurements)
            
            if measurement_stats is None or measurement_stats.get('success_rate', 0) == 0:
                print(f"  ❌ All measurements failed")
                continue
            
            # Print concise summary for this pose
            avg_reproj_err = measurement_stats.get('mean_reprojection_error_avg', 0)
            avg_reproj_std = measurement_stats.get('mean_reprojection_error_std', 0)
            avg_sharpness = measurement_stats.get('sharpness_avg', 0)
            avg_sharpness_std = measurement_stats.get('sharpness_std', 0)
            avg_rot_err = measurement_stats.get('rotation_error_avg', 0)
            avg_rot_std = measurement_stats.get('rotation_error_std', 0)
            avg_trans_err = measurement_stats.get('translation_error_avg', 0)
            avg_trans_std = measurement_stats.get('translation_error_std', 0)
            
            # Quality indicators
            reproj_quality = "✅" if avg_reproj_err < 0.5 else "⚠️" if avg_reproj_err < 1.0 else "❌"
            sharpness_quality = "✅" if avg_sharpness < 3.0 else "⚠️" if avg_sharpness < 5.0 else "❌"
            
            print(f"  📊 Results: {successful_count}/3 successful | "
                  f"Reproj: {reproj_quality} {avg_reproj_err:.3f}±{avg_reproj_std:.3f}px | "
                  f"Sharpness: {sharpness_quality} {avg_sharpness:.2f}±{avg_sharpness_std:.2f}px")
            print(f"  🎯 Errors: Rot={avg_rot_err:.3f}±{avg_rot_std:.3f}° | Trans={avg_trans_err:.3f}±{avg_trans_std:.3f}mm")

            # Get current arm pose for pose data (reuse from after movement)
            A_i_pose_world_frame_raw = A_i_pose_world_frame_raw_after_move
            A_i_pose_world_frame_raw_from_motion_service = A_i_pose_world_frame_raw_from_motion_service_after_move
            
            # Compare poses from arm vs motion service (silently, for data collection)
            arm_vs_motion_comparison = compare_poses(
                A_i_pose_world_frame_raw,
                A_i_pose_world_frame_raw_from_motion_service,
                label=f"Arm vs Motion Service - Pose {actual_pose_number}",
                verbose=False
            )
            
            # Also compare commanded vs motion service (silently, for data collection)
            commanded_vs_motion_comparison = compare_poses(
                target_pose,
                A_i_pose_world_frame_raw_from_motion_service,
                label=f"Commanded vs Motion Service - Pose {actual_pose_number}",
                verbose=False
            )
            
            # Invert only the rotation, keep translation unchanged
            A_i_pose_world_frame = _invert_pose_rotation_only(A_i_pose_world_frame_raw)
            A_i_pose_world_frame_from_motion_service = _invert_pose_rotation_only(A_i_pose_world_frame_raw_from_motion_service)

            # Use the first successful measurement for pose data
            successful_measurement = next((m for m in measurements if m and m.get('success', False)), None)
            if successful_measurement is None:
                print(f"  No successful measurements for pose data, skipping pose {actual_pose_number}")
                continue
                
            # Convert chessboard pose (don't transpose - solvePnP is OpenCV convention)
            rvec = np.array(successful_measurement['rvec'])
            tvec = np.array(successful_measurement['tvec'])
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
            
            # Use the average hand-eye errors from all successful measurements
            successful_measurements = [m for m in measurements if m and m.get('success', False)]
            if successful_measurements:
                avg_rotation_error = np.mean([m.get('hand_eye_errors', {}).get('rotation_error', 0) for m in successful_measurements])
                avg_translation_error = np.mean([m.get('hand_eye_errors', {}).get('translation_error', 0) for m in successful_measurements])
                hand_eye_errors = {
                    'rotation_error': float(avg_rotation_error),
                    'translation_error': float(avg_translation_error)
                }
            else:
                hand_eye_errors = {'rotation_error': 0, 'translation_error': 0}
            
            # Create detailed measurement results with individual values and averages
            measurement_results = []
            for measurement in measurements:
                if measurement and measurement.get('success', False):
                    measurement_result = {
                        "measurement_num": measurement['measurement_num'],
                        "success": measurement['success'],
                        "rvec": measurement['rvec'],
                        "tvec": measurement['tvec'],
                        "corners_count": measurement['corners_count'],
                        "filtered_corners_count": measurement['filtered_corners_count'],
                        "hand_eye_errors": measurement['hand_eye_errors'],
                        "mean_reprojection_error": measurement['mean_reprojection_error'],
                        "max_reprojection_error": measurement['max_reprojection_error'],
                        "sharpness": measurement['sharpness'],
                        "camera_temperature": measurement.get('camera_temperature', {})
                    }
                    measurement_results.append(measurement_result)
            
            pose_info = {
                "pose_index": actual_pose_number - 1,  # Store 0-based index for consistency
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
                "A_i_pose_from_motion_service_raw": {
                    "x": A_i_pose_world_frame_raw_from_motion_service.x,
                    "y": A_i_pose_world_frame_raw_from_motion_service.y,
                    "z": A_i_pose_world_frame_raw_from_motion_service.z,
                    "o_x": A_i_pose_world_frame_raw_from_motion_service.o_x,
                    "o_y": A_i_pose_world_frame_raw_from_motion_service.o_y,
                    "o_z": A_i_pose_world_frame_raw_from_motion_service.o_z,
                    "theta": A_i_pose_world_frame_raw_from_motion_service.theta
                },
                "A_i_pose_inverted_from_motion_service": {
                    "x": A_i_pose_world_frame_from_motion_service.x,
                    "y": A_i_pose_world_frame_from_motion_service.y,
                    "z": A_i_pose_world_frame_from_motion_service.z,
                    "o_x": A_i_pose_world_frame_from_motion_service.o_x,
                    "o_y": A_i_pose_world_frame_from_motion_service.o_y,
                    "o_z": A_i_pose_world_frame_from_motion_service.o_z,
                    "theta": A_i_pose_world_frame_from_motion_service.theta
                },
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "T_B_i_camera_frame": T_B_i_camera_frame.tolist(),
                "hand_eye_errors": hand_eye_errors,
                "measurements": measurement_results,
                "measurement_statistics": measurement_stats,
                "pose_comparisons": {
                    "commanded_vs_arm_after_move": commanded_vs_arm_comparison,
                    "commanded_vs_motion_service": commanded_vs_motion_comparison,
                    "arm_after_move_vs_motion_service": arm_vs_motion_comparison
                },
                "summary": {
                    "num_measurements": len(measurement_results),
                    "success_rate": len(measurement_results) / 3.0,
                    "hand_eye_errors": {
                        "rotation_error": f"{hand_eye_errors['rotation_error']:.3f}°",
                        "translation_error": f"{hand_eye_errors['translation_error']:.3f}mm"
                    },
                    "reprojection_errors": {
                        "mean_error": f"{measurement_stats.get('mean_reprojection_error_avg', 0):.3f}±{measurement_stats.get('mean_reprojection_error_std', 0):.3f}px",
                        "max_error": f"{measurement_stats.get('max_reprojection_error_avg', 0):.3f}±{measurement_stats.get('max_reprojection_error_std', 0):.3f}px"
                    },
                    "detection_quality": {
                        "sharpness": f"{measurement_stats.get('sharpness_avg', 0):.2f}±{measurement_stats.get('sharpness_std', 0):.2f}px",
                        "corners": f"{measurement_stats.get('corners_count_avg', 0):.1f}±{measurement_stats.get('corners_count_std', 0):.1f}"
                    }
                }
            }
            rotation_data.append(pose_info)

            # Save image for this pose (use the first successful measurement's image)
            if successful_measurement:
                # Get a fresh image for visualization
                image = await get_camera_image(camera)
                debug_image = draw_marker_debug(image, rvec, tvec, camera_matrix, dist_coeffs, 
                                              marker_type=marker_type, chessboard_size=(chessboard_cols, chessboard_rows), 
                                              square_size=chessboard_square_size, aruco_size=aruco_size, 
                                              validation_info=successful_measurement)
                # Create subdirectory for pose images
                pose_images_dir = os.path.join(data_dir, "pose_images")
                os.makedirs(pose_images_dir, exist_ok=True)
                cv2.imwrite(os.path.join(pose_images_dir, f"image_pose_{actual_pose_number}.jpg"), debug_image)

            # Save pose data incrementally (in case of crash)
            with open(os.path.join(data_dir, "pose_data.json"), "w", encoding='utf-8') as f:
                json.dump(rotation_data, f, indent=2, ensure_ascii=False)

            await asyncio.sleep(1.0)
        
        # Generate comprehensive statistics and add to pose data
        
        # For statistics, we use all poses (rotation_data already contains the cleaned data)
        # rotation_data now contains: poses from before resume_from_pose + new poses from current run
        all_poses_for_statistics = rotation_data.copy()
        statistics = generate_comprehensive_statistics(all_poses_for_statistics)
        
        # Calculate timing statistics
        end_time = datetime.now()
        total_runtime = (end_time - start_time).total_seconds()
        poses_tested = len([p for p in all_poses_for_statistics if p.get('hand_eye_errors')])
        avg_time_per_pose = total_runtime / poses_tested if poses_tested > 0 else 0
        
        # Add statistics to the pose data (includes all poses: previous + current run)
        pose_data_with_stats = {
            "poses": rotation_data,
            "comprehensive_statistics": statistics,
            "timing": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_runtime_seconds": total_runtime,
                "total_runtime_minutes": total_runtime / 60,
                "poses_tested": poses_tested,
                "average_time_per_pose_seconds": avg_time_per_pose
            }
        }
        
        # Save updated pose data with statistics
        pose_data_file = os.path.join(data_dir, "pose_data.json")
        with open(pose_data_file, "w", encoding='utf-8') as f:
            json.dump(pose_data_with_stats, f, indent=2, ensure_ascii=False)
        
        # Create comprehensive statistics plots
        create_comprehensive_statistics_plot(all_poses_for_statistics, data_dir, tag)
        create_summary_table_plot(all_poses_for_statistics, data_dir, tag)
        create_pose_error_plot(all_poses_for_statistics, data_dir, tag)
        
        # Print summary statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        if resume_from_pose > 1:
            print(f"Note: Statistics include all poses from previous runs and current run")
        print(f"Note: Statistics use ALL individual measurements (3 per pose), showing measurement variability")
        print(f"Total poses tested: {statistics['total_poses']}")
        print(f"Successful poses: {statistics['successful_poses']}")
        print(f"Success rate: {statistics['success_rate']:.1%}")
        
        print(f"\n🎯 HAND-EYE VERIFICATION ERRORS (from all individual measurements)t :")
        print(f"Rotation error: {statistics['hand_eye']['rotation_error']['mean']:.3f}° ± {statistics['hand_eye']['rotation_error']['std']:.3f}°")
        print(f"  Range: {statistics['hand_eye']['rotation_error']['min']:.3f}° - {statistics['hand_eye']['rotation_error']['max']:.3f}°")
        print(f"Translation error: {statistics['hand_eye']['translation_error']['mean']:.3f}mm ± {statistics['hand_eye']['translation_error']['std']:.3f}mm")
        print(f"  Range: {statistics['hand_eye']['translation_error']['min']:.3f}mm - {statistics['hand_eye']['translation_error']['max']:.3f}mm")
        
        print(f"\n📐 REPROJECTION ERRORS:")
        print(f"Mean reprojection error: {statistics['reprojection']['mean_error']['mean']:.3f}px ± {statistics['reprojection']['mean_error']['std']:.3f}px")
        print(f"  Range: {statistics['reprojection']['mean_error']['min']:.3f}px - {statistics['reprojection']['mean_error']['max']:.3f}px")
        print(f"Max reprojection error: {statistics['reprojection']['max_error']['mean']:.3f}px ± {statistics['reprojection']['max_error']['std']:.3f}px")
        print(f"  Range: {statistics['reprojection']['max_error']['min']:.3f}px - {statistics['reprojection']['max_error']['max']:.3f}px")
        
        print(f"\n🔍 CHESSBOARD DETECTION QUALITY:")
        print(f"Sharpness: {statistics['detection']['sharpness']['mean']:.2f}px ± {statistics['detection']['sharpness']['std']:.2f}px")
        print(f"  Range: {statistics['detection']['sharpness']['min']:.2f}px - {statistics['detection']['sharpness']['max']:.2f}px")
        print(f"Corners detected: {statistics['detection']['corners']['mean']:.1f} ± {statistics['detection']['corners']['std']:.1f}")
        print(f"  Range: {statistics['detection']['corners']['min']:.0f} - {statistics['detection']['corners']['max']:.0f}")
        
        # Add pose comparison statistics if available
        if 'pose_comparisons' in statistics:
            print(f"\n📐 POSE COMPARISON STATISTICS:")
            comparisons = statistics['pose_comparisons']
            
            # Commanded vs Arm
            if 'commanded_vs_arm' in comparisons:
                cmd_arm = comparisons['commanded_vs_arm']
                print(f"\n  Commanded vs Arm:")
                if 'translation_error_mm' in cmd_arm:
                    t_stats = cmd_arm['translation_error_mm']
                    print(f"    Translation error: {t_stats['mean']:.3f}mm ± {t_stats['std']:.3f}mm")
                    print(f"      Range: {t_stats['min']:.3f}mm - {t_stats['max']:.3f}mm")
                if 'rotation_error_deg' in cmd_arm:
                    r_stats = cmd_arm['rotation_error_deg']
                    print(f"    Rotation error: {r_stats['mean']:.3f}° ± {r_stats['std']:.3f}°")
                    print(f"      Range: {r_stats['min']:.3f}° - {r_stats['max']:.3f}°")
            
            # Commanded vs Motion Service
            if 'commanded_vs_motion_service' in comparisons:
                cmd_motion = comparisons['commanded_vs_motion_service']
                print(f"\n  Commanded vs Motion Service:")
                if 'translation_error_mm' in cmd_motion:
                    t_stats = cmd_motion['translation_error_mm']
                    print(f"    Translation error: {t_stats['mean']:.3f}mm ± {t_stats['std']:.3f}mm")
                    print(f"      Range: {t_stats['min']:.3f}mm - {t_stats['max']:.3f}mm")
                if 'rotation_error_deg' in cmd_motion:
                    r_stats = cmd_motion['rotation_error_deg']
                    print(f"    Rotation error: {r_stats['mean']:.3f}° ± {r_stats['std']:.3f}°")
                    print(f"      Range: {r_stats['min']:.3f}° - {r_stats['max']:.3f}°")
            
            # Arm vs Motion Service
            if 'arm_vs_motion_service' in comparisons:
                arm_motion = comparisons['arm_vs_motion_service']
                print(f"\n  Arm vs Motion Service:")
                if 'translation_error_mm' in arm_motion:
                    t_stats = arm_motion['translation_error_mm']
                    print(f"    Translation error: {t_stats['mean']:.3f}mm ± {t_stats['std']:.3f}mm")
                    print(f"      Range: {t_stats['min']:.3f}mm - {t_stats['max']:.3f}mm")
                if 'rotation_error_deg' in arm_motion:
                    r_stats = arm_motion['rotation_error_deg']
                    print(f"    Rotation error: {r_stats['mean']:.3f}° ± {r_stats['std']:.3f}°")
                    print(f"      Range: {r_stats['min']:.3f}° - {r_stats['max']:.3f}°")
        
        # Add camera temperature statistics if available
        if 'camera_temperature' in statistics:
            print(f"\n🌡️  CAMERA TEMPERATURE:")
            temp_stats = statistics['camera_temperature']
            # Print unconditionally to avoid hiding valid small values
            rgb = temp_stats.get('rgb_temp_c', {})
            mb = temp_stats.get('main_board_temp_c', {})
            cpu = temp_stats.get('cpu_temp_c', {})
            if rgb:
                print(f"RGB sensor: {rgb.get('mean', 0.0):.1f}°C ± {rgb.get('std', 0.0):.1f}°C")
                print(f"  Range: {rgb.get('min', 0.0):.1f}°C - {rgb.get('max', 0.0):.1f}°C")
            if mb:
                print(f"Main board: {mb.get('mean', 0.0):.1f}°C ± {mb.get('std', 0.0):.1f}°C")
                print(f"  Range: {mb.get('min', 0.0):.1f}°C - {mb.get('max', 0.0):.1f}°C")
            if cpu:
                print(f"CPU: {cpu.get('mean', 0.0):.1f}°C ± {cpu.get('std', 0.0):.1f}°C")
                print(f"  Range: {cpu.get('min', 0.0):.1f}°C - {cpu.get('max', 0.0):.1f}°C")
        
        print(f"\n⏱️  TIMING STATISTICS:")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {total_runtime:.1f} seconds ({total_runtime/60:.1f} minutes)")
        print(f"Poses tested: {poses_tested}")
        print(f"Average time per pose: {avg_time_per_pose:.1f} seconds")
        
        
        # Return to reference pose
        A_0_pose_in_frame = PoseInFrame(reference_frame=arm.name + "_origin", pose=A_0_pose_world_frame_raw)
        await motion_service.move(component_name=arm.name, destination=A_0_pose_in_frame)
        await asyncio.sleep(5.0)  # Increased settling time
        
        
    except Exception as e:
        print("Caught exception in script main: ")
        raise e
    finally:
        if viam_client:
            viam_client.close()
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
Examples:
  # Start new test
  python pose_test_script.py --camera-name sensing-camera --arm-name myarm \\
    --pose-tracker-name mytracker --poses poses.json

  # Start new test with custom tag
  python pose_test_script.py --camera-name sensing-camera --arm-name myarm \\
    --pose-tracker-name mytracker --poses poses.json --tag "calibration_v2"

  # Resume from pose 27, continuing to save in existing directory
  python pose_test_script.py --camera-name sensing-camera --arm-name myarm \\
    --pose-tracker-name mytracker --poses poses.json --resume-from-pose 27 \\
    --data-dir calibration_data_20251029_180338

  # Use manual hand-eye transformation from JSON file
  python pose_test_script.py --camera-name sensing-camera --arm-name myarm \\
    --pose-tracker-name mytracker --poses poses.json \\
    --hand-eye-transform hand_eye_transform.json

  # Use custom chessboard configuration
  python pose_test_script.py --camera-name sensing-camera --arm-name myarm \\
    --pose-tracker-name mytracker --poses poses.json \\
    --chessboard-cols 9 --chessboard-rows 6 --chessboard-square-size 25.0

Hand-eye transform JSON format (two options):
  1. Direct 4x4 matrix:
     [[1.0, 0.0, 0.0, 80.0],
      [0.0, 1.0, 0.0, -20.2],
      [0.0, 0.0, 1.0, 48.7],
      [0.0, 0.0, 0.0, 1.0]]
  
  2. Frame configuration format:
     {
       "translation": {"x": 80.0, "y": -20.2, "z": 48.7},
       "orientation": {
         "value": {"x": 0.0, "y": 0.0, "z": 1.0, "theta": 0.0},
         "type": "ov_degrees"
       }
     }

Pose JSON format (list of pose objects):
  [
    {"x": 100, "y": 200, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 45},
    {"x": 150, "y": 100, "z": 250, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 90},
    {"x": 120, "y": 180, "z": 280, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 135}
  ]

All pose objects must have: x, y, z, o_x, o_y, o_z, theta
        """
    )
    parser.add_argument('--env-file', 
        default='.env',
        type=str,
        help='Path to the .env file to use (default: .env)'
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
        required=True,
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
        default='IPPE',
        choices=['IPPE_SQUARE', 'IPPE', 'ITERATIVE', 'SQPNP'],
        help='PnP method to use for ArUco detection (default: IPPE_SQUARE)'
    )
    parser.add_argument(
        '--use-sb-detection',
        action='store_true',
        help='Use findChessboardCornersSB (subpixel detection) instead of findChessboardCorners for chessboard'
    )
    parser.add_argument(
        '--resume-from-pose',
        type=int,
        default=1,
        help='Resume testing from this pose number (1-based indexing, default: 1)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory to save/continue saving data. Use this when resuming to continue saving to the same directory. If not provided, creates new timestamped directory.'
    )
    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        help='Tag to add to directory name and plot titles (e.g., "test1", "calibration_v2")'
    )
    parser.add_argument(
        '--hand-eye-transform',
        type=str,
        default=None,
        help='Path to JSON file containing hand-eye transformation matrix. If provided, this will be used instead of extracting from the machine configuration. The JSON can be either a 4x4 matrix (list of lists) or a frame config format with translation and orientation.'
    )
    parser.add_argument(
        '--chessboard-cols',
        type=int,
        default=11,
        help='Number of columns (inner corners) in the chessboard pattern (default: 11)'
    )
    parser.add_argument(
        '--chessboard-rows',
        type=int,
        default=8,
        help='Number of rows (inner corners) in the chessboard pattern (default: 8)'
    )
    parser.add_argument(
        '--chessboard-square-size',
        type=float,
        default=30.0,
        help='Size of each chessboard square in millimeters (default: 30.0)'
    )

    args = parser.parse_args()

    # Parse poses from JSON file
    try:
        poses, reference_pose = parse_poses_from_json(args.poses)
        if poses is None:
            print("No poses provided, using default poses")
            poses = None
        else:
            if not reference_pose:
                print("No reference pose specified - will use current arm position")
    except Exception as e:
        print(f"Error parsing poses: {e}")
        exit(1)

    asyncio.run(main(
        env_file=args.env_file,
        arm_name=args.arm_name,
        pose_tracker_name=args.pose_tracker_name,
        motion_service_name="motion",
        camera_name=args.camera_name,
        poses=poses,
        reference_pose=reference_pose,
        marker_type=args.marker_type,
        aruco_id=args.aruco_id,
        aruco_size=args.aruco_size,
        aruco_dict=args.aruco_dict,
        pnp_method=args.pnp_method,
        use_sb_detection=args.use_sb_detection,
        resume_from_pose=args.resume_from_pose,
        data_dir=args.data_dir,
        tag=args.tag,
        hand_eye_transform_file=args.hand_eye_transform,
        chessboard_cols=args.chessboard_cols,
        chessboard_rows=args.chessboard_rows,
        chessboard_square_size=args.chessboard_square_size,
        poses_file_path=args.poses,
    ))
