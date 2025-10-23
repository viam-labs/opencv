import argparse
import asyncio
import copy
import os
import numpy as np
import cv2
import json
import pickle
from datetime import datetime
from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.app.app_client import AppClient
from viam.components.arm import Arm
from viam.components.pose_tracker import PoseTracker
from viam.services.motion import MotionClient, Constraints
from viam.proto.service.motion import CollisionSpecification
from viam.proto.common import PoseInFrame, Pose
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import CameraMimeType
from viam.components.camera import Camera
from viam.rpc.dial import DialOptions, Credentials
from viam.app.viam_client import ViamClient

from typing import Dict, Optional

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils.utils import call_go_ov2mat, call_go_mat2ov
except ModuleNotFoundError:
    from ..utils.utils import call_go_ov2mat, call_go_mat2ov

# Default values for optional args
DEFAULT_WORLD_FRAME = "world"

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

async def _get_current_arm_pose(motion: MotionClient, arm_name: str) -> Pose:
    pose_in_frame = await motion.get_pose(
        component_name=arm_name,
        destination_frame="world"
    )
    return pose_in_frame.pose

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

def get_camera_pose_from_chessboard(image, camera_matrix, dist_coeffs, chessboard_size, square_size=30.0):
    """
    Get camera pose relative to chessboard using PnP
    Returns: (success, rotation_vector, translation_vector, corners)
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
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # Refine corners to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        # Solve PnP to get camera pose
        success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
        
        if success:
            return True, rvec, tvec, corners
        else:
            return False, None, None, None
    else:
        return False, None, None, None

def compute_hand_eye_verification_errors(T_hand_eye, T_delta_A_gripper_frame, T_delta_B_camera_frame):
    """
    Compute hand-eye calibration verification errors using the paper's method.
    
    Args:
        T_hand_eye: 4x4 hand-eye transformation matrix (camera to gripper)
        T_delta_A_gripper_frame: 4x4 actual robot motion in gripper frame
        T_delta_B_camera_frame: 4x4 camera motion (chessboard motion in camera frame)
    
    Returns:
        dict with keys:
            - R_A_actual: 3x3 actual rotation matrix
            - R_A_predicted: 3x3 predicted rotation matrix
            - t_A_actual: 3x1 actual translation vector
            - t_A_predicted: 3x1 predicted translation vector
            - angle_actual: actual rotation angle in degrees
            - angle_predicted: predicted rotation angle in degrees
            - axis_actual: actual rotation axis (if angle > 0.01)
            - axis_predicted: predicted rotation axis (if angle > 0.01)
            - rotation_error: rotation error in degrees (Equation 48)
            - translation_error: translation error in mm (Equation 48)
            - error_axis: error rotation axis (if error > 0.01)
    """
    # Paper's method (Equation 47): Predict robot motion from camera motion
    # Â_i0 = X B_i0 X^(-1) (similarity transform)
    T_A_predicted = T_hand_eye @ T_delta_B_camera_frame @ np.linalg.inv(T_hand_eye)
    
    # Extract rotation matrices and translations
    R_A_actual = T_delta_A_gripper_frame[:3, :3]
    R_A_predicted = T_A_predicted[:3, :3]
    t_A_actual = T_delta_A_gripper_frame[:3, 3]
    t_A_predicted = T_A_predicted[:3, 3]
    
    # Convert to axis-angle
    rvec_actual, _ = cv2.Rodrigues(R_A_actual)
    rvec_pred, _ = cv2.Rodrigues(R_A_predicted)
    angle_actual = np.linalg.norm(rvec_actual) * 180 / np.pi
    angle_pred = np.linalg.norm(rvec_pred) * 180 / np.pi
    
    # Calculate errors according to paper (Equation 48)
    # R_error = (R̂_A_i0)^T R_A_i0
    R_error = R_A_predicted.T @ R_A_actual
    rvec_error, _ = cv2.Rodrigues(R_error)
    rotation_error = np.linalg.norm(rvec_error) * 180 / np.pi
    
    # t_error = t̂_A_i0 - t_A_i0
    t_error = t_A_predicted - t_A_actual
    translation_error = np.linalg.norm(t_error)
    
    # Build result dictionary
    result = {
        'R_A_actual': R_A_actual,
        'R_A_predicted': R_A_predicted,
        't_A_actual': t_A_actual,
        't_A_predicted': t_A_predicted,
        'angle_actual': angle_actual,
        'angle_predicted': angle_pred,
        'rotation_error': rotation_error,
        'translation_error': translation_error,
    }
    
    # Add rotation axes if angles are significant
    if angle_actual > 0.01:
        result['axis_actual'] = rvec_actual.flatten() / np.linalg.norm(rvec_actual)
    else:
        result['axis_actual'] = None
        
    if angle_pred > 0.01:
        result['axis_predicted'] = rvec_pred.flatten() / np.linalg.norm(rvec_pred)
    else:
        result['axis_predicted'] = None
    
    if rotation_error > 0.01:
        result['error_axis'] = rvec_error.flatten() / np.linalg.norm(rvec_error)
    else:
        result['error_axis'] = None
    
    return result

async def main(
    arm_name: str,
    pose_tracker_name: str,
    motion_service_name: str,
    body_names: list[str],
    camera_name: str,
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

        print(f"Connected to robot: {machine}")
        
        # Get the hand-eye transformation from camera configuration
        print(f"\n=== EXTRACTING HAND-EYE TRANSFORMATION ===")
        try:
            # Get robot configuration from app client
            organizations = await app_client.list_organizations()
            if organizations:
                org = organizations[0]
                org_id = org.id
                locations = await app_client.list_locations(org_id=org_id)
                if locations:
                    location = locations[0]
                    location_id = location.id
                    robots = await app_client.list_robots(location_id=location_id)
                    if robots:
                        robot = robots[0]
                        robot_id = robot.id
                        robot_parts = await app_client.get_robot_parts(robot_id)
                        if robot_parts:
                            robot_part = robot_parts[0]
                            robot_part_id = robot_part.id
                            robot_part_config = await app_client.get_robot_part(robot_part_id)
                            
                            if robot_part_config:
                                robot_config = robot_part_config.robot_config
                                
                                if 'components' in robot_config:
                                    components = robot_config['components']
                                    
                                    # Find the camera component configuration
                                    camera_config = None
                                    for component in components:
                                        if component.get('name') == camera_name:
                                            camera_config = component
                                            break
                                    
                                    if camera_config and 'frame' in camera_config and camera_config['frame']:
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
                                            print(f"\nRotation Matrix (3x3) - as stored in config:")
                                            for i in range(3):
                                                print(f"  [{R[i,0]:8.4f} {R[i,1]:8.4f} {R[i,2]:8.4f}]")
                                            print(f"Translation Vector: [{t[0]:8.4f}, {t[1]:8.4f}, {t[2]:8.4f}]")
                                            
                                            # Config stores T_gripper2cam, but we need T_cam2gripper for verification
                                            # Invert the transformation: T^(-1) = [R^T, -R^T*t; 0, 1]
                                            R_cam2gripper = R.T
                                            t_cam2gripper = -R.T @ t
                                            T_hand_eye = np.eye(4)
                                            T_hand_eye[:3, :3] = R_cam2gripper
                                            T_hand_eye[:3, 3] = t_cam2gripper
                                            
                                            # Analyze what transformation T_hand_eye represents
                                            print(f"\n=== ANALYZING HAND-EYE TRANSFORM ===")
                                            rvec_he, _ = cv2.Rodrigues(R_cam2gripper)
                                            angle_he = np.linalg.norm(rvec_he) * 180 / np.pi
                                            if angle_he > 0.01:
                                                axis_he = rvec_he.flatten() / np.linalg.norm(rvec_he)
                                                print(f"Hand-eye rotation: {angle_he:.2f}° around axis [{axis_he[0]:.3f}, {axis_he[1]:.3f}, {axis_he[2]:.3f}]")
                                            
                                            # Test: Apply similarity transform to a pure Z-axis 90° rotation
                                            R_test_z = np.array([
                                                [0, -1, 0],
                                                [1,  0, 0],
                                                [0,  0, 1]
                                            ], dtype=np.float64)  # 90° around Z
                                            
                                            R_test_transformed = R_cam2gripper @ R_test_z @ R_cam2gripper.T
                                            rvec_test, _ = cv2.Rodrigues(R_test_transformed)
                                            angle_test = np.linalg.norm(rvec_test) * 180 / np.pi
                                            if angle_test > 0.01:
                                                axis_test = rvec_test.flatten() / np.linalg.norm(rvec_test)
                                                print(f"\nTest: Pure Z-axis 90° rotation through similarity transform:")
                                                print(f"  Result: {angle_test:.2f}° around axis [{axis_test[0]:.3f}, {axis_test[1]:.3f}, {axis_test[2]:.3f}]")
                                                print(f"  Expected robot axis: [-0.804, 0.033, -0.594]")
                                                
                                                expected_axis = np.array([-0.804, 0.033, -0.594])
                                                dot_product = np.abs(np.dot(axis_test, expected_axis))
                                                angle_between = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
                                                print(f"  Angle between transformed and expected: {angle_between:.2f}°")
                                                if angle_between < 5:
                                                    print(f"  ✅ Hand-eye transform looks correct!")
                                                else:
                                                    print(f"  ❌ Hand-eye transform NOT producing expected axis")
                                            
                                        else:
                                            print(f"Warning: Frame configuration is not a dictionary")
                                    else:
                                        print(f"Warning: No frame configuration found for camera '{camera_name}'")
                                else:
                                    print(f"Warning: Robot config does not have 'components' key")
                            else:
                                print(f"Warning: Could not retrieve robot configuration")
                        else:
                            print(f"Warning: No robot parts found")
                    else:
                        print(f"Warning: No robots found")
                else:
                    print(f"Warning: No locations found")
            else:
                print(f"Warning: No organizations found")
                
        except Exception as e:
            print(f"Error retrieving frame configuration: {e}")
            print("Continuing without frame configuration...")
            T_hand_eye = None
        
        # Get initial poses
        A_0_pose_world_frame = await _get_current_arm_pose(motion_service, arm.name)
        T_A_0_world_frame = _pose_to_matrix(A_0_pose_world_frame)

        camera_matrix, dist_coeffs = await get_camera_intrinsics(camera)
        image = await get_camera_image(camera)

        success, rvec, tvec, corners = get_camera_pose_from_chessboard(image, camera_matrix, dist_coeffs, chessboard_size=(11, 8), square_size=30.0)
        if not success:
            print("Failed to detect chessboard in reference image")
            return
        
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
            "A_0_pose": {
                "x": A_0_pose_world_frame.x,
                "y": A_0_pose_world_frame.y,
                "z": A_0_pose_world_frame.z,
                "o_x": A_0_pose_world_frame.o_x,
                "o_y": A_0_pose_world_frame.o_y,
                "o_z": A_0_pose_world_frame.o_z,
                "theta": A_0_pose_world_frame.theta
            },
            "T_B_0_camera_frame": T_B_0_camera_frame.tolist()
        }
        with open(os.path.join(data_dir, "calibration_config.json"), "w") as f:
            json.dump(calibration_data, f, indent=2)
        
        # Save reference image
        cv2.imwrite(os.path.join(data_dir, "image_reference.jpg"), image)
        print(f"Saved reference image and config")
        
        # List to store all rotation data
        rotation_data = []
        
        # Test the hand-eye transformation with 4 rotations
        for i in range(4):
            rotation_angle = (i + 1) * 90  # 5°, 10°, 15°, 20°
            print(f"\n=== ROTATION {i+1}/4: {rotation_angle}° ===")
            
            # Calculate target pose (rotate around Z-axis)
            target_pose = copy.deepcopy(A_0_pose_world_frame)
            target_pose.theta = A_0_pose_world_frame.theta + rotation_angle
            
            print(f"Moving to target pose: theta={target_pose.theta:.2f}°")
            
            # Move to target pose
            target_pose_in_frame = PoseInFrame(reference_frame=DEFAULT_WORLD_FRAME, pose=target_pose)
            if(rotation_angle == 360):
                await motion_service.move(component_name=arm.name, destination=PoseInFrame(reference_frame=DEFAULT_WORLD_FRAME, pose=A_0_pose_world_frame))
            else:
                await motion_service.move(component_name=arm.name, destination=target_pose_in_frame)
            await asyncio.sleep(2.0)
            
            image = await get_camera_image(camera)
            success, rvec, tvec, corners = get_camera_pose_from_chessboard(image, camera_matrix, dist_coeffs, chessboard_size=(11, 8), square_size=30.0)
            if not success:
                print(f"Failed to detect chessboard in rotation {i+1}")
                continue
            
            # Debug: Save image with detected corners
            debug_img = image.copy()
            cv2.drawChessboardCorners(debug_img, (11, 8), corners, success)
            # Draw coordinate axes on the chessboard
            cv2.drawFrameAxes(debug_img, camera_matrix, dist_coeffs, rvec, tvec, 50)
            cv2.imwrite(f"debug_rotation_{i+1}.jpg", debug_img)
            print(f"Saved debug image: debug_rotation_{i+1}.jpg")
            
            # Get current poses
            A_i_pose_world_frame = await _get_current_arm_pose(motion_service, arm.name)
            T_A_i_world_frame = _pose_to_matrix(A_i_pose_world_frame)

            # Convert chessboard pose (don't transpose - solvePnP is OpenCV convention)
            T_B_i_camera_frame = rvec_tvec_to_matrix(rvec, tvec)
            
            # Save rotation data
            rotation_info = {
                "rotation_index": i,
                "rotation_angle": rotation_angle,
                "A_i_pose": {
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
            rotation_data.append(rotation_info)
            
            # Save image for this rotation
            cv2.imwrite(os.path.join(data_dir, f"image_rotation_{i+1}.jpg"), image)
            
            # Save rotation data incrementally (in case of crash)
            with open(os.path.join(data_dir, "rotation_data.json"), "w") as f:
                json.dump(rotation_data, f, indent=2)
            print(f"Saved rotation {i+1} data")
            
            # Compute robot motion in WORLD frame
            T_delta_A_world_frame = np.linalg.inv(T_A_i_world_frame) @ T_A_0_world_frame
            
            # Compute robot motion in GRIPPER's LOCAL frame (for proper comparison with similarity transform)
            # A_local = inv(T_A_0) @ T_A_i expressed in gripper_0's frame
            T_delta_A_gripper_frame = np.linalg.inv(T_A_i_world_frame) @ T_A_0_world_frame @ np.linalg.inv(T_A_i_world_frame).T
            
            # Actually, simpler: transform world-frame motion to gripper-frame motion
            R_A_0 = T_A_0_world_frame[:3, :3]
            T_delta_A_gripper_frame = np.eye(4)
            T_delta_A_gripper_frame[:3, :3] = R_A_0.T @ T_delta_A_world_frame[:3, :3] @ R_A_0
            T_delta_A_gripper_frame[:3, 3] = R_A_0.T @ T_delta_A_world_frame[:3, 3]
            
            # B = motion of target as observed in camera frame
            T_delta_B_camera_frame = T_B_i_camera_frame @ np.linalg.inv(T_B_0_camera_frame)
            
            # Debug: Print detailed transformation info
            print(f"\n  DEBUG - Transformations:")
            print(f"  Robot pose 0: ({A_0_pose_world_frame.x:.1f}, {A_0_pose_world_frame.y:.1f}, {A_0_pose_world_frame.z:.1f}) θ={A_0_pose_world_frame.theta:.1f}°")
            print(f"  Robot pose i: ({A_i_pose_world_frame.x:.1f}, {A_i_pose_world_frame.y:.1f}, {A_i_pose_world_frame.z:.1f}) θ={A_i_pose_world_frame.theta:.1f}°")
            print(f"  Robot delta translation: [{T_delta_A_world_frame[0,3]:.2f}, {T_delta_A_world_frame[1,3]:.2f}, {T_delta_A_world_frame[2,3]:.2f}]")
            print(f"  Camera delta translation: [{T_delta_B_camera_frame[0,3]:.2f}, {T_delta_B_camera_frame[1,3]:.2f}, {T_delta_B_camera_frame[2,3]:.2f}]")
            
            # Compute verification errors using modular function
            errors = compute_hand_eye_verification_errors(
                T_hand_eye, 
                T_delta_A_gripper_frame, 
                T_delta_B_camera_frame
            )
            
            # Debug: Print rotation matrices
            print(f"\n  DEBUG - Predicted vs Actual robot motion:")
            print(f"  R_A_actual:")
            for row in errors['R_A_actual']:
                print(f"    [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}]")
            print(f"  R_A_predicted:")
            for row in errors['R_A_predicted']:
                print(f"    [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}]")
            
            # Print axis-angle representation
            if errors['axis_actual'] is not None:
                axis = errors['axis_actual']
                print(f"  Actual: {errors['angle_actual']:.2f}° around axis [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
            if errors['axis_predicted'] is not None:
                axis = errors['axis_predicted']
                print(f"  Predicted: {errors['angle_predicted']:.2f}° around axis [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")

            # Print errors
            print(f"\n  ERRORS (Paper's method - Eq. 48):")
            print(f"  Rotation error: {errors['rotation_error']:.3f}°")
            print(f"  Translation error: {errors['translation_error']:.3f} mm")
            if errors['error_axis'] is not None:
                axis = errors['error_axis']
                print(f"  Error rotation: {errors['rotation_error']:.2f}° around axis [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
            
            # Wait between measurements
            await asyncio.sleep(1.0)
        
        print(f"\n✅ ALL DATA SAVED TO: {data_dir}")
        print(f"   - calibration_config.json (camera params, hand-eye, reference pose)")
        print(f"   - image_reference.jpg + image_rotation_1-4.jpg")
        print(f"   - rotation_data.json (all arm poses and chessboard detections)")
        
        # Return to reference pose
        print(f"\n=== RETURNING TO REFERENCE POSE ===")
        A_0_pose_in_frame = PoseInFrame(reference_frame=DEFAULT_WORLD_FRAME, pose=A_0_pose_world_frame)
        await motion_service.move(component_name=arm.name, destination=A_0_pose_in_frame)
        await asyncio.sleep(2.0)
        
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
    parser = argparse.ArgumentParser(description='Clean hand-eye calibration test')
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
    args = parser.parse_args()
    asyncio.run(main(
        arm_name=args.arm_name,
        pose_tracker_name=args.pose_tracker_name,
        motion_service_name="motion",
        body_names=["corner_45",],
        camera_name=args.camera_name,
    ))