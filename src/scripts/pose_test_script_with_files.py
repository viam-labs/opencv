#!/usr/bin/env python3

"""
Script to test hand-eye calibration from machine against tsai50 dataset.
Gets the hand-eye transformation from the machine's frame configuration
and verifies it against pre-recorded tsai50 poses and images.
"""

import argparse
import asyncio
import os
import numpy as np
import cv2
import json
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.app.app_client import AppClient
from viam.rpc.dial import DialOptions
from viam.app.viam_client import ViamClient
from typing import Optional
import math

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils.utils import call_go_ov2mat, call_go_mat2ov
except ModuleNotFoundError:
    from ..utils.utils import call_go_ov2mat, call_go_mat2ov

# Default configuration
DEFAULT_CAMERA_INTRINSICS = {
    'fx': 1256.13623046875, 'fy': 1256.1561279296875,
    'cx': 955.8938598632812, 'cy': 542.18994140625
}
DEFAULT_DISTORTION_COEFFS = {
    'k1': 0.11422603577375412, 'k2': -0.32854774594306946,
    'p1': -0.00028485868824645877, 'p2': -0.00017464916163589805,
    'k3': 0.2558501362800598
}
DEFAULT_CHESSBOARD_CONFIG = {
  "square_size_mm": 30,
  "pattern_size": (11, 8)  # (width, height) of inner corners
}

async def connect():
    """Connect to the robot and get AppClient"""
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
    
    # Convert axis-angle to rotation matrix
    R = call_go_ov2mat(axis[0], axis[1], axis[2], th)
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def ov_to_transform_matrix(pose: dict) -> np.ndarray:
    """
    Converts a Viam orientation vector pose to a 4x4 homogeneous transformation matrix.
    Uses Viam's spatialmath library via call_go_ov2mat for correct conversion.
    """
    t = np.array([pose['x'], pose['y'], pose['z']])
    
    R = call_go_ov2mat(pose['o_x'], pose['o_y'], pose['o_z'], pose['theta'])

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def load_dataset(dataset_dir: str):
    """Load dataset with poses and images from specified directory"""
    poses_path = os.path.join(dataset_dir, 'poses.json')
    
    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Could not find 'poses.json' in '{dataset_dir}'")
    
    with open(poses_path, 'r') as f:
        poses_data = json.load(f)
    
    if isinstance(poses_data, dict) and 'poses' in poses_data:
        all_poses = poses_data['poses']
    elif isinstance(poses_data, list):
        all_poses = poses_data
    else:
        raise ValueError("Could not find a list of poses in poses.json")
    
    return all_poses

def process_dataset(dataset_dir: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                    chessboard_config: dict, image_prefix: str = None):
    """
    Process dataset to get gripper and target poses.
    
    Args:
        dataset_dir: Path to dataset directory containing poses.json and images
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Camera distortion coefficients
        chessboard_config: Dict with 'pattern_size' and 'square_size_mm'
        image_prefix: Prefix for image files (e.g., 'tsai50_', 'image_'). 
                     If None, will try common patterns.
    
    IMPORTANT: The poses are assumed to be stored as base-to-gripper transforms,
    NOT gripper-to-base. We invert them to get the correct gripper-to-base transforms.
    """
    all_poses = load_dataset(dataset_dir)
    
    gripper_poses = []
    target_poses = []
    
    pattern_size = chessboard_config['pattern_size']
    square_size = chessboard_config['square_size_mm']
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    dataset_name = os.path.basename(dataset_dir)
    print(f"\nProcessing {dataset_name} dataset...")
    num_poses = len(all_poses)
    
    for i in range(num_poses):
        # Try to find the image file with different naming patterns
        img_path = None
        
        # Try provided prefix first
        if image_prefix:
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = os.path.join(dataset_dir, f"{image_prefix}{i}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break
        
        # Try common patterns if not found
        if not img_path:
            for ext in ['.png', '.jpg', '.jpeg']:
                # Try dataset_name_i pattern (e.g., tsai50_0.png)
                test_path = os.path.join(dataset_dir, f"{dataset_name}_{i}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break
                # Try image_i pattern
                test_path = os.path.join(dataset_dir, f"image_{i}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break
                # Try just i pattern
                test_path = os.path.join(dataset_dir, f"{i}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break
        
        if not img_path:
            print(f"  - WARNING: Could not find image for index {i}")
            continue
        
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            success, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
            
            if not success:
                print(f"  - Failed to solve PnP for index {i}")
                continue
            
            # Refine pose estimation
            rvec, tvec = cv2.solvePnPRefineVVS(objp, corners2, camera_matrix, dist_coeffs, rvec, tvec)
            
            # solvePnP returns rvec, tvec that describe the object's position in camera frame
            # This is T_object_to_camera (transforms points from object coords to camera coords)
            R, _ = cv2.Rodrigues(rvec)
            T_target_cam = np.identity(4)
            T_target_cam[:3, :3] = R
            T_target_cam[:3, 3] = tvec.flatten()
            target_poses.append(T_target_cam)
            
            # CRITICAL: Test theory - only invert rotation, keep translation as-is
            # Hypothesis: Rotation is base-to-gripper, but translation is already gripper-in-base
            T_base_gripper = ov_to_transform_matrix(all_poses[i])
            
            # Create gripper-to-base with inverted rotation but original translation
            T_gripper_base = np.eye(4)
            T_gripper_base[:3, :3] = T_base_gripper[:3, :3].T  # Transpose (inverse) of rotation
            T_gripper_base[:3, 3] = T_base_gripper[:3, 3]      # Keep translation as-is
            
            gripper_poses.append(T_gripper_base)
        else:
            print(f"  - Could not find chessboard in image {i}")
    
    print(f"Successfully processed {len(gripper_poses)} / {num_poses} image-pose pairs from {dataset_name}.")
    return gripper_poses, target_poses

async def get_hand_eye_from_machine(app_client: AppClient, camera_name: str):
    """Get the hand-eye transformation from the machine's frame configuration"""
    print(f"\n=== EXTRACTING HAND-EYE TRANSFORMATION FROM MACHINE ===")
    
    try:
        # Get robot configuration from app client
        organizations = await app_client.list_organizations()
        if not organizations:
            print("Warning: No organizations found")
            return None
        
        org = organizations[0]
        org_id = org.id
        locations = await app_client.list_locations(org_id=org_id)
        if not locations:
            print("Warning: No locations found")
            return None
        
        location = locations[0]
        location_id = location.id
        robots = await app_client.list_robots(location_id=location_id)
        if not robots:
            print("Warning: No robots found")
            return None
        
        robot = robots[0]
        robot_id = robot.id
        robot_parts = await app_client.get_robot_parts(robot_id)
        if not robot_parts:
            print("Warning: No robot parts found")
            return None
        
        robot_part = robot_parts[0]
        robot_part_id = robot_part.id
        robot_part_config = await app_client.get_robot_part(robot_part_id)
        
        if not robot_part_config:
            print("Warning: Could not retrieve robot configuration")
            return None
        
        robot_config = robot_part_config.robot_config
        
        if 'components' not in robot_config:
            print("Warning: Robot config does not have 'components' key")
            return None
        
        components = robot_config['components']
        
        # Find the camera component configuration
        camera_config = None
        for component in components:
            if component.get('name') == camera_name:
                camera_config = component
                break
        
        if not camera_config or 'frame' not in camera_config or not camera_config['frame']:
            print(f"Warning: No frame configuration found for camera '{camera_name}'")
            return None
        
        frame_config = camera_config['frame']
        print(f"Found camera frame configuration")
        
        # Handle frame configuration as dictionary
        if isinstance(frame_config, dict):
            parent = frame_config.get('parent', 'unknown')
            translation = frame_config.get('translation', {})
            orientation = frame_config.get('orientation', {})
            
            print(f"Parent frame: {parent}")
            print(f"Translation: x={translation.get('x', 0):.6f}, y={translation.get('y', 0):.6f}, z={translation.get('z', 0):.6f}")
            print(f"Orientation: {orientation}")
            
            # Convert frame configuration directly to transformation matrix
            T_hand_eye = frame_config_to_transformation_matrix(frame_config)
            
            print(f"\nHand-Eye Transformation Matrix (4x4):")
            print(f"T_camera_to_{parent} =")
            for i in range(4):
                print(f"  [{T_hand_eye[i,0]:8.4f} {T_hand_eye[i,1]:8.4f} {T_hand_eye[i,2]:8.4f} {T_hand_eye[i,3]:8.4f}]")
            
            # Extract rotation matrix and translation vector
            R = T_hand_eye[:3, :3]
            t = T_hand_eye[:3, 3]
            print(f"\nRotation Matrix (3x3):")
            for i in range(3):
                print(f"  [{R[i,0]:8.4f} {R[i,1]:8.4f} {R[i,2]:8.4f}]")
            print(f"Translation Vector: [{t[0]:8.4f}, {t[1]:8.4f}, {t[2]:8.4f}]")
            
            return T_hand_eye
        else:
            print(f"Warning: Frame configuration is not a dictionary")
            return None
            
    except Exception as e:
        print(f"Error retrieving frame configuration: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_hand_eye_calibration(X_hand_eye: np.ndarray, gripper_poses, target_poses, dataset_name: str = "dataset"):
    """Verify the hand-eye calibration against the dataset"""
    
    if len(gripper_poses) < 2:
        print("\nWARNING: Not enough data to verify (need at least 2 poses)")
        return
    
    rot_errors = []
    trans_errors = []
    
    print(f"\n=== VERIFICATION AGAINST {dataset_name.upper()} DATASET ===")
    print("Testing hand-eye transformation")
    print("Note: Using same formulation as pose_test_script.py")
    
    # Use pose 0 as reference
    T_g0_base = gripper_poses[0]
    T_t0_cam = target_poses[0]
    
    for i in range(1, len(gripper_poses)):
        # Compute relative gripper motion A (BACKWARDS: from i to 0)
        T_gi_base = gripper_poses[i]
        A = np.linalg.inv(T_gi_base) @ T_g0_base
        
        # Compute relative target motion B (FORWARDS: from 0 to i)
        T_ti_cam = target_poses[i]
        B = T_ti_cam @ np.linalg.inv(T_t0_cam)
        
        # Verify using equation: AX = XB => A_predicted = inv(X) @ B @ X
        X_inv = np.linalg.inv(X_hand_eye)
        A_predicted = X_inv @ B @ X_hand_eye
        
        # Compute rotation error (from error matrix)
        R_A_actual = A[:3, :3]
        R_A_predicted = A_predicted[:3, :3]
        R_error = R_A_predicted.T @ R_A_actual
        angle_rad = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1.0, 1.0))
        rot_errors.append(np.degrees(angle_rad))
        
        # Compute translation error (direct comparison, as in pose_test_script.py)
        t_A_actual = A[:3, 3]
        t_A_predicted = A_predicted[:3, 3]
        t_error = t_A_predicted - t_A_actual
        trans_errors.append(np.linalg.norm(t_error))
    
    mean_rot_err = np.mean(rot_errors)
    std_rot_err = np.std(rot_errors)
    mean_trans_err = np.mean(trans_errors)
    std_trans_err = np.std(trans_errors)
    
    print("\n--- Verification Results ---")
    print(f"Number of poses tested against reference pose 0: {len(rot_errors)}")
    print("\nRotation Error:")
    print(f"  - Mean: {mean_rot_err:.4f} degrees")
    print(f"  - Std Dev: {std_rot_err:.4f} degrees")
    print("\nTranslation Error:")
    print(f"  - Mean: {mean_trans_err:.4f} mm")
    print(f"  - Std Dev: {std_trans_err:.4f} mm")
    
    return mean_rot_err, std_rot_err, mean_trans_err, std_trans_err

async def main(camera_name: str, dataset_dir: str, camera_intrinsics: dict, 
               distortion_coeffs: dict, chessboard_config: dict, image_prefix: str = None):
    """Main function"""
    app_client: Optional[AppClient] = None
    machine: Optional[RobotClient] = None
    
    try:
        # Connect to robot
        app_client, machine = await connect()
        print(f"Connected to robot")
        
        # Get hand-eye transformation from machine
        X_hand_eye = await get_hand_eye_from_machine(app_client, camera_name)
        
        if X_hand_eye is None:
            print("\nERROR: Could not retrieve hand-eye transformation from machine")
            return
        
        # Prepare camera parameters
        cam_mat = np.array([
            [camera_intrinsics['fx'], 0, camera_intrinsics['cx']],
            [0, camera_intrinsics['fy'], camera_intrinsics['cy']],
            [0, 0, 1]
        ])
        dist_coeffs = np.array([
            distortion_coeffs['k1'], distortion_coeffs['k2'],
            distortion_coeffs['p1'], distortion_coeffs['p2'],
            distortion_coeffs['k3']
        ])
        
        # Load and process dataset
        gripper_poses, target_poses = process_dataset(
            dataset_dir, cam_mat, dist_coeffs, chessboard_config, image_prefix
        )
        
        if len(gripper_poses) == 0:
            print(f"\nERROR: No valid poses found in {dataset_dir}")
            return
        
        # Verify the hand-eye calibration
        dataset_name = os.path.basename(dataset_dir)
        verify_hand_eye_calibration(X_hand_eye, gripper_poses, target_poses, dataset_name)
        
    finally:
        if machine:
            await machine.close()
            print("\nDisconnected from robot")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test machine hand-eye calibration against a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test against tsai50 dataset with defaults
  python pose_test_script_with_files.py --dataset-dir ./tsai50
  
  # Test against simple_rotation dataset
  python pose_test_script_with_files.py --dataset-dir ./simple_rotation
  
  # Test against park28 dataset
  python pose_test_script_with_files.py --dataset-dir ./park28
        """
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="sensing-camera",
        help="Name of the camera component (default: sensing-camera)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./tsai50",
        help="Path to dataset directory containing poses.json and images (default: ./tsai50)"
    )
    parser.add_argument(
        "--image-prefix",
        type=str,
        default=None,
        help="Prefix for image files (e.g., 'tsai50_', 'image_'). Auto-detected if not specified."
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(
        camera_name=args.camera_name,
        dataset_dir=args.dataset_dir,
        camera_intrinsics=DEFAULT_CAMERA_INTRINSICS,
        distortion_coeffs=DEFAULT_DISTORTION_COEFFS,
        chessboard_config=DEFAULT_CHESSBOARD_CONFIG,
        image_prefix=args.image_prefix
    ))
