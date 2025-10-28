#!/usr/bin/env python3

"""
Script to test hand-eye calibration by moving the robot arm and comparing
predicted vs actual transformations.
"""

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
except ModuleNotFoundError:
    from ..utils.utils import call_go_ov2mat, call_go_mat2ov

DEFAULT_WORLD_FRAME = "world"

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

def get_chessboard_pose_in_camera_frame(image, camera_matrix, dist_coeffs, chessboard_size, square_size=30.0):
    """
    Get chessboard pose in camera frame using PnP.
    
    Note: cv2.solvePnP returns the transformation from chessboard coordinates to camera coordinates.
    This is T_chessboard_to_camera, NOT T_camera_to_chessboard.
    
    Returns: (success, rotation_vector, translation_vector, corners)
        - rvec, tvec represent the chessboard's pose in the camera's coordinate system
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

        if not success:
            print("Failed to solve PnP")
            return False, None, None, None

        rvec, tvec = cv2.solvePnPRefineVVS(objp, corners, camera_matrix, dist_coeffs, rvec, tvec)
        return True, rvec, tvec, corners
    else:
        return False, None, None, None


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
    T_A_predicted = T_eye_hand @ T_delta_B_camera_frame @ T_hand_eye

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

        success, rvec, tvec, _ = get_chessboard_pose_in_camera_frame(image, camera_matrix, dist_coeffs, chessboard_size=(11, 8), square_size=30.0)
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
        cv2.imwrite(os.path.join(data_dir, "image_reference.jpg"), image)
        print(f"Saved reference image and config")
        
        # List to store all rotation data
        rotation_data = []

        # Test the hand-eye transformation with provided poses
        print(f"\n=== TESTING {len(poses)} POSES ===")
        for i, pose_spec in enumerate(poses):
            print(f"\n=== POSE {i+1}/{len(poses)} ===")

            # Create target pose from specification
            target_pose = Pose(
                x=pose_spec['x'],
                y=pose_spec['y'],
                z=pose_spec['z'],
                o_x=pose_spec['o_x'],
                o_y=pose_spec['o_y'],
                o_z=pose_spec['o_z'],
                theta=pose_spec['theta']
            )
            print(f"  Position: ({target_pose.x:.1f}, {target_pose.y:.1f}, {target_pose.z:.1f})")
            print(f"  Orientation: ({target_pose.o_x:.3f}, {target_pose.o_y:.3f}, {target_pose.o_z:.3f}) @ {target_pose.theta:.1f}°")

            # Move to target pose
            target_pose_in_frame = PoseInFrame(reference_frame=DEFAULT_WORLD_FRAME, pose=target_pose)
            await motion_service.move(component_name=arm.name, destination=target_pose_in_frame)
            await asyncio.sleep(2.0)
            
            image = await get_camera_image(camera)
            success, rvec, tvec, _ = get_chessboard_pose_in_camera_frame(image, camera_matrix, dist_coeffs, chessboard_size=(11, 8), square_size=30.0)
            if not success:
                print(f"  Failed to detect chessboard, skipping pose {i+1}")
                continue

            # Get current poses
            A_i_pose_world_frame_raw = await _get_current_arm_pose(motion_service, arm.name, arm)
            # Invert only the rotation, keep translation unchanged
            A_i_pose_world_frame = _invert_pose_rotation_only(A_i_pose_world_frame_raw)
            T_A_i_world_frame = _pose_to_matrix(A_i_pose_world_frame)

            # Convert chessboard pose (don't transpose - solvePnP is OpenCV convention)
            T_B_i_camera_frame = rvec_tvec_to_matrix(rvec, tvec)
            
            # Save pose data
            pose_info = {
                "pose_index": i,
                "pose_spec": pose_spec,
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
            cv2.imwrite(os.path.join(data_dir, f"image_pose_{i+1}.jpg"), image)

            # Save pose data incrementally (in case of crash)
            with open(os.path.join(data_dir, "pose_data.json"), "w") as f:
                json.dump(rotation_data, f, indent=2)
            print(f"  Saved pose {i+1} data")
            
            # Compute relative transformations
            T_delta_A_world_frame = np.linalg.inv(T_A_i_world_frame) @ T_A_0_world_frame
            T_delta_B_camera_frame = T_B_i_camera_frame @ np.linalg.inv(T_B_0_camera_frame)

            # Compute verification errors
            errors = compute_hand_eye_verification_errors(
                T_hand_eye,
                T_delta_A_world_frame,
                T_delta_B_camera_frame
            )

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
        required=True,
        help='Path to JSON file containing list of pose objects'
    )
    args = parser.parse_args()

    # Parse poses from JSON file
    try:
        poses = parse_poses_from_json(args.poses)
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
    ))