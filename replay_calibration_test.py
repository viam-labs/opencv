#!/usr/bin/env python3
"""
Replay calibration test data without connecting to the robot.
Re-detects chessboard from saved images.
Usage: python replay_calibration_test.py <data_directory>
"""

import argparse
import json
import numpy as np
import cv2
import os
import sys
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from utils.utils import call_go_ov2mat
except ModuleNotFoundError:
    print("Warning: Could not import go utilities, using basic rotation conversion")
    def call_go_ov2mat(ox, oy, oz, theta):
        """Fallback: convert axis-angle to rotation matrix"""
        angle_rad = np.deg2rad(theta)
        axis = np.array([ox, oy, oz])
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
        return R

def rvec_tvec_to_matrix(rvec, tvec):
    """Convert rotation vector and translation vector to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = tvec.flatten()
    return T

def detect_chessboard(image, camera_matrix, dist_coeffs, chessboard_size, square_size):
    """Detect chessboard in image and return pose"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if not success:
        return None, None, None
    
    # Refine corner positions
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Create 3D object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    
    if not success:
        return None, None, None
    
    return rvec, tvec, corners

def pose_dict_to_matrix(pose_dict):
    """Convert pose dictionary to 4x4 transformation matrix"""
    R = call_go_ov2mat(pose_dict["o_x"], pose_dict["o_y"], pose_dict["o_z"], pose_dict["theta"])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [pose_dict["x"], pose_dict["y"], pose_dict["z"]]
    return T

def run_verification(data_dir):
    """Run the verification using saved data"""
    
    print(f"Loading data from: {data_dir}")
    
    # Load calibration config
    with open(os.path.join(data_dir, "calibration_config.json"), "r") as f:
        config = json.load(f)
    
    camera_matrix = np.array(config["camera_matrix"])
    dist_coeffs = np.array(config["dist_coeffs"]).reshape(-1)
    chessboard_size = tuple(config["chessboard_size"])
    square_size = config["square_size"]
    T_hand_eye = np.array(config["hand_eye_transform"])
    T_A_0_world_frame = pose_dict_to_matrix(config["A_0_pose"])
    
    # Try to load rotation data with actual robot poses
    rotation_data = None
    rotation_data_path = os.path.join(data_dir, "rotation_data.json")
    if os.path.exists(rotation_data_path):
        with open(rotation_data_path, "r") as f:
            rotation_data = json.load(f)
        print(f"✓ Loaded rotation data with actual robot poses")
    else:
        print(f"⚠️  rotation_data.json not found - will only show predicted motion")
    
    print(f"\n=== CONFIGURATION ===")
    print(f"Chessboard: {chessboard_size} squares, {square_size}mm")
    print(f"Hand-eye transform loaded")
    
    # Detect chessboard in reference image
    print(f"\nProcessing reference image...")
    ref_image = cv2.imread(os.path.join(data_dir, "image_reference.jpg"))
    rvec_0, tvec_0, corners_0 = detect_chessboard(ref_image, camera_matrix, dist_coeffs, chessboard_size, square_size)
    if rvec_0 is None:
        print("ERROR: Could not detect chessboard in reference image!")
        return
    
    T_B_0_camera_frame = rvec_tvec_to_matrix(rvec_0, tvec_0)
    print(f"✓ Reference chessboard detected")
    
    # Find all rotation images
    rotation_images = sorted(glob.glob(os.path.join(data_dir, "image_rotation_*.jpg")))
    print(f"Found {len(rotation_images)} rotation images")
    
    # Process each rotation
    for idx, img_path in enumerate(rotation_images):
        img_name = os.path.basename(img_path)
        rot_num = img_name.split('_')[-1].split('.')[0]
        
        print(f"\n=== ROTATION {rot_num} ===")
        
        # Detect chessboard
        image = cv2.imread(img_path)
        rvec_i, tvec_i, corners_i = detect_chessboard(image, camera_matrix, dist_coeffs, chessboard_size, square_size)
        
        if rvec_i is None:
            print(f"WARNING: Could not detect chessboard in {img_name}")
            continue
        
        T_B_i_camera_frame = rvec_tvec_to_matrix(rvec_i, tvec_i)
        
        # Camera motion
        T_delta_B_camera_frame = T_B_i_camera_frame @ np.linalg.inv(T_B_0_camera_frame)
        
        # Predicted robot motion using similarity transform
        T_A_predicted = T_hand_eye @ T_delta_B_camera_frame @ np.linalg.inv(T_hand_eye)
        
        # If we have actual robot pose data, compute full errors
        if rotation_data and idx < len(rotation_data):
            rot_data = rotation_data[idx]
            T_A_i_world_frame = pose_dict_to_matrix(rot_data["A_i_pose"])
            
            # Compute actual robot motion
            T_delta_A_world_frame = np.linalg.inv(T_A_i_world_frame) @ T_A_0_world_frame
            
            # Transform to gripper frame for fair comparison
            R_A_0 = T_A_0_world_frame[:3, :3]
            T_delta_A_gripper_frame = np.eye(4)
            T_delta_A_gripper_frame[:3, :3] = R_A_0.T @ T_delta_A_world_frame[:3, :3] @ R_A_0
            T_delta_A_gripper_frame[:3, 3] = R_A_0.T @ T_delta_A_world_frame[:3, 3]
            
            # Extract rotations and translations
            R_A_actual = T_delta_A_gripper_frame[:3, :3]
            R_A_predicted = T_A_predicted[:3, :3]
            t_A_actual = T_delta_A_gripper_frame[:3, 3]
            t_A_predicted = T_A_predicted[:3, 3]
            
            # Convert to axis-angle
            rvec_actual, _ = cv2.Rodrigues(R_A_actual)
            rvec_pred, _ = cv2.Rodrigues(R_A_predicted)
            angle_actual = np.linalg.norm(rvec_actual) * 180 / np.pi
            angle_pred = np.linalg.norm(rvec_pred) * 180 / np.pi
            
            print(f"  Predicted vs Actual robot motion (in gripper frame):")
            if angle_actual > 0.01:
                axis_actual = rvec_actual.flatten() / np.linalg.norm(rvec_actual)
                print(f"    Actual: {angle_actual:.2f}° around axis [{axis_actual[0]:.3f}, {axis_actual[1]:.3f}, {axis_actual[2]:.3f}]")
            if angle_pred > 0.01:
                axis_pred = rvec_pred.flatten() / np.linalg.norm(rvec_pred)
                print(f"    Predicted: {angle_pred:.2f}° around axis [{axis_pred[0]:.3f}, {axis_pred[1]:.3f}, {axis_pred[2]:.3f}]")
            
            # Calculate errors
            R_error = R_A_predicted.T @ R_A_actual
            rvec_error, _ = cv2.Rodrigues(R_error)
            rotation_error = np.linalg.norm(rvec_error) * 180 / np.pi
            
            t_error = t_A_predicted - t_A_actual
            translation_error = np.linalg.norm(t_error)
            
            print(f"\n  ERRORS (Paper's method - Eq. 48):")
            print(f"    Rotation error: {rotation_error:.3f}°")
            print(f"    Translation error: {translation_error:.3f} mm")
        else:
            # No actual robot pose data - only show predictions
            R_A_predicted = T_A_predicted[:3, :3]
            t_A_predicted = T_A_predicted[:3, 3]
            
            rvec_pred, _ = cv2.Rodrigues(R_A_predicted)
            angle_pred = np.linalg.norm(rvec_pred) * 180 / np.pi
            
            print(f"  Camera translation (from chessboard): [{T_delta_B_camera_frame[0,3]:.1f}, {T_delta_B_camera_frame[1,3]:.1f}, {T_delta_B_camera_frame[2,3]:.1f}] mm")
            
            if angle_pred > 0.01:
                axis_pred = rvec_pred.flatten() / np.linalg.norm(rvec_pred)
                print(f"  Predicted robot motion: {angle_pred:.2f}° around axis [{axis_pred[0]:.3f}, {axis_pred[1]:.3f}, {axis_pred[2]:.3f}]")
                print(f"  Predicted translation: [{t_A_predicted[0]:.1f}, {t_A_predicted[1]:.1f}, {t_A_predicted[2]:.1f}] mm")
            else:
                print(f"  Predicted robot motion: ~0° (minimal rotation)")
            
            print(f"  ⚠️  Cannot compute full error without actual robot pose data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay calibration test data")
    parser.add_argument("data_dir", help="Directory containing saved calibration data")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory not found: {args.data_dir}")
        sys.exit(1)
    
    run_verification(args.data_dir)
