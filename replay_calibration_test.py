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

# Import functions from pose_test_script
from scripts.pose_test_script import (
    _pose_to_matrix, 
    rvec_tvec_to_matrix, 
    get_camera_pose_from_chessboard,
    compute_hand_eye_verification_errors
)

def pose_dict_to_matrix(pose_dict):
    """Convert pose dictionary to 4x4 transformation matrix using pose_test_script function"""
    # Create a simple object to pass to _pose_to_matrix
    from types import SimpleNamespace
    pose = SimpleNamespace(
        x=pose_dict["x"],
        y=pose_dict["y"],
        z=pose_dict["z"],
        o_x=pose_dict["o_x"],
        o_y=pose_dict["o_y"],
        o_z=pose_dict["o_z"],
        theta=pose_dict["theta"]
    )
    return _pose_to_matrix(pose)

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
    success, rvec_0, tvec_0, corners_0 = get_camera_pose_from_chessboard(
        ref_image, camera_matrix, dist_coeffs, 
        chessboard_size=chessboard_size, 
        square_size=square_size
    )
    if not success:
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
        success, rvec_i, tvec_i, corners_i = get_camera_pose_from_chessboard(
            image, camera_matrix, dist_coeffs,
            chessboard_size=chessboard_size,
            square_size=square_size
        )
        
        if not success:
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
            
            # Compute errors using modular function
            errors = compute_hand_eye_verification_errors(
                T_hand_eye,
                T_delta_A_gripper_frame,
                T_delta_B_camera_frame
            )
            
            # Print results
            print(f"  Predicted vs Actual robot motion (in gripper frame):")
            if errors['axis_actual'] is not None:
                axis = errors['axis_actual']
                print(f"    Actual: {errors['angle_actual']:.2f}° around axis [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
            if errors['axis_predicted'] is not None:
                axis = errors['axis_predicted']
                print(f"    Predicted: {errors['angle_predicted']:.2f}° around axis [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
            
            print(f"\n  ERRORS (Paper's method - Eq. 48):")
            print(f"    Rotation error: {errors['rotation_error']:.3f}°")
            print(f"    Translation error: {errors['translation_error']:.3f} mm")
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
