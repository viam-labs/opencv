import argparse
import asyncio
import copy
import os
import numpy as np
import cv2
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

async def _get_current_camera_pose(pose_tracker: PoseTracker, body_name: str) -> Pose:
    pose_in_frame = await pose_tracker.get_poses(body_names=[body_name])
    if not pose_in_frame or body_name not in pose_in_frame:
        raise Exception(f"Could not find tracked body '{body_name}' in pose tracker observations")
    return pose_in_frame[body_name].pose

def hand_eye_calibration_ax_xb(A_motions, B_motions):
    """
    Simple hand-eye calibration using AX = XB formulation.
    Uses a basic least-squares approach.
    
    Args:
        A_motions: List of robot motion matrices (4x4)
        B_motions: List of camera motion matrices (4x4)
    
    Returns:
        X: Hand-eye transformation matrix (4x4)
    """
    print(f"\n=== HAND-EYE CALIBRATION (AX = XB) ===")
    print(f"Number of motion pairs: {len(A_motions)}")
    
    n = len(A_motions)
    
    # For pure rotation around Z-axis, we can use a simplified approach
    # Since we're doing rotations around Z, the hand-eye transformation
    # should also be primarily around Z-axis
    
    # Extract rotation angles around Z-axis
    angles_A = []
    angles_B = []
    
    for i in range(n):
        # Get rotation matrix
        R_A = A_motions[i][:3, :3]
        R_B = B_motions[i][:3, :3]
        
        # Convert to axis-angle
        rvec_A, _ = cv2.Rodrigues(R_A)
        rvec_B, _ = cv2.Rodrigues(R_B)
        
        # Get rotation angles
        angle_A = np.linalg.norm(rvec_A) * 180 / np.pi
        angle_B = np.linalg.norm(rvec_B) * 180 / np.pi
        
        # Determine sign based on Z-component of rotation vector
        if rvec_A[2] < 0:
            angle_A = -angle_A
        if rvec_B[2] < 0:
            angle_B = -angle_B
            
        angles_A.append(angle_A)
        angles_B.append(angle_B)
        
        print(f"Motion {i+1}: Robot={angle_A:.3f}°, Camera={angle_B:.3f}°")
    
    # For pure Z-axis rotations, the hand-eye transformation should be
    # a rotation around Z-axis plus translation
    
    # Estimate rotation angle from the relationship
    # For small angles: angle_A ≈ angle_B (if hand-eye is identity)
    # For general case: angle_A = angle_B + offset
    
    # Calculate the average offset
    offsets = [angles_A[i] - angles_B[i] for i in range(n)]
    avg_offset = np.mean(offsets)
    
    print(f"Average angle offset: {avg_offset:.3f}°")
    
    # Create hand-eye transformation
    # For Z-axis rotation, we need to account for the coordinate system
    # The hand-eye transformation should align the camera and robot coordinate systems
    
    # Simple approach: assume the hand-eye transformation is primarily
    # a rotation around Z-axis to align coordinate systems
    
    # Calculate the required rotation to align the coordinate systems
    # If camera rotates by angle_B and robot rotates by angle_A,
    # then the hand-eye transformation should account for the difference
    
    # For now, let's use a simple identity transformation
    # and let the user see the raw data
    X = np.eye(4)
    
    # Add a small rotation to account for coordinate system alignment
    # This is a simplified approach - in practice, you'd solve the full AX=XB problem
    
    # Calculate average translation
    avg_translation_A = np.mean([A_motions[i][:3, 3] for i in range(n)], axis=0)
    avg_translation_B = np.mean([B_motions[i][:3, 3] for i in range(n)], axis=0)
    
    print(f"Average robot translation: [{avg_translation_A[0]:.3f}, {avg_translation_A[1]:.3f}, {avg_translation_A[2]:.3f}]")
    print(f"Average camera translation: [{avg_translation_B[0]:.3f}, {avg_translation_B[1]:.3f}, {avg_translation_B[2]:.3f}]")
    
    # Set translation to the difference
    X[:3, 3] = avg_translation_A - avg_translation_B
    
    return X

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
                                            print(f"\nRotation Matrix (3x3):")
                                            for i in range(3):
                                                print(f"  [{R[i,0]:8.4f} {R[i,1]:8.4f} {R[i,2]:8.4f}]")
                                            print(f"Translation Vector: [{t[0]:8.4f}, {t[1]:8.4f}, {t[2]:8.4f}]")
                                            
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
        A_0_pose = await _get_current_arm_pose(motion_service, arm.name)
        T_A_0 = _pose_to_matrix(A_0_pose)
        B_0_pose = await _get_current_camera_pose(pt, body_names[0])
        T_B_0 = _pose_to_matrix(B_0_pose)

        if T_hand_eye is None:
            print("❌ Could not extract hand-eye transformation from camera configuration")
            return
        
        print(f"\n=== HAND-EYE TRANSFORMATION VERIFICATION TEST ===")
        print(f"Reference robot pose: x={A_0_pose.x:.2f}, y={A_0_pose.y:.2f}, z={A_0_pose.z:.2f}, theta={A_0_pose.theta:.2f}°")
        print(f"Reference camera pose: x={B_0_pose.x:.2f}, y={B_0_pose.y:.2f}, z={B_0_pose.z:.2f}, theta={B_0_pose.theta:.2f}°")
        
        # Test the hand-eye transformation with 4 rotations
        for i in range(4):
            rotation_angle = (i + 1) * 90  # 5°, 10°, 15°, 20°
            print(f"\n=== ROTATION {i+1}/4: {rotation_angle}° ===")
            
            # Calculate target pose (rotate around Z-axis)
            target_pose = copy.deepcopy(A_0_pose)
            target_pose.theta = A_0_pose.theta + rotation_angle
            
            print(f"Moving to target pose: theta={target_pose.theta:.2f}°")
            
            # Move to target pose
            target_pose_in_frame = PoseInFrame(reference_frame=DEFAULT_WORLD_FRAME, pose=target_pose)
            await motion_service.move(component_name=arm.name, destination=target_pose_in_frame)
            await asyncio.sleep(2.0)
            
            # Get current poses
            A_i_pose = await _get_current_arm_pose(motion_service, arm.name)
            T_A_i = _pose_to_matrix(A_i_pose)
            B_i_pose = await _get_current_camera_pose(pt, body_names[0])
            T_B_i = _pose_to_matrix(B_i_pose)
            
            # Calculate motion matrices (relative to reference)
            T_delta_A = np.linalg.inv(T_A_0) @ T_A_i
            T_delta_B = np.linalg.inv(T_B_0) @ T_B_i
            
            # Calculate rotation angles
            R_delta_A = T_delta_A[:3, :3]
            rvec_delta_A, _ = cv2.Rodrigues(R_delta_A)
            angle_delta_A = np.linalg.norm(rvec_delta_A) * 180 / np.pi
            
            R_delta_B = T_delta_B[:3, :3]
            rvec_delta_B, _ = cv2.Rodrigues(R_delta_B)
            angle_delta_B = np.linalg.norm(rvec_delta_B) * 180 / np.pi
            
            print(f"Robot rotation: {angle_delta_A:.3f}° (target: {rotation_angle}°)")
            print(f"Camera rotation: {angle_delta_B:.3f}°")
            print(f"Translation: A=[{T_delta_A[0,3]:.3f}, {T_delta_A[1,3]:.3f}, {T_delta_A[2,3]:.3f}]")
            print(f"Translation: B=[{T_delta_B[0,3]:.3f}, {T_delta_B[1,3]:.3f}, {T_delta_B[2,3]:.3f}]")
            
            # Apply hand-eye transformation to predict robot motion from camera motion
            # A_predicted = T_hand_eye @ B_motion @ T_hand_eye^-1
            T_A_predicted = T_hand_eye @ T_delta_B @ np.linalg.inv(T_hand_eye)
            
            # Calculate predicted rotation angle
            R_predicted = T_A_predicted[:3, :3]
            rvec_predicted, _ = cv2.Rodrigues(R_predicted)
            angle_predicted = np.linalg.norm(rvec_predicted) * 180 / np.pi
            
            # Calculate errors
            rotation_error = abs(angle_predicted - angle_delta_A)
            translation_error = np.linalg.norm(T_A_predicted[:3, 3] - T_delta_A[:3, 3])

            #Calulate error according to paper
            rotation_error_paper = R_predicted.T @ R_delta_A
            translation_error_paper = T_A_predicted[:3, 3] - T_delta_A[:3, 3]
            rotation_error_paper_vec, _ = cv2.Rodrigues(rotation_error_paper.T)
            rotation_error_paper_angle = np.linalg.norm(rotation_error_paper_vec) * 180 / np.pi
            print(f"Rotation error according to paper: {rotation_error_paper_angle:.3f}°")
            translation_error_paper_norm = np.linalg.norm(translation_error_paper)
            print(f"Translation error according to paper: {translation_error_paper_norm:.3f} mm")
            
            print(f"Predicted robot rotation: {angle_predicted:.3f}°")
            print(f"Rotation error: {rotation_error:.3f}°")
            print(f"Translation error: {translation_error:.3f} mm")
            
            # Wait between measurements
            await asyncio.sleep(1.0)
        
        # Return to reference pose
        print(f"\n=== RETURNING TO REFERENCE POSE ===")
        A_0_pose_in_frame = PoseInFrame(reference_frame=DEFAULT_WORLD_FRAME, pose=A_0_pose)
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