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
    # when running as local module with run.sh
    from ..utils.utils import call_go_ov2mat, call_go_mat2ov

# Default values for optional args
DEFAULT_WORLD_FRAME = "world"

def frame_config_to_transformation_matrix(frame_config):
    """
    Convert Viam frame configuration to a 4x4 transformation matrix.
    
    Args:
        frame_config: Viam frame configuration object
        
    Returns:
        numpy.ndarray: 4x4 transformation matrix
    """
    import numpy as np
    
    # Extract translation
    t = np.array([frame_config.translation.x, frame_config.translation.y, frame_config.translation.z])
    
    # Extract rotation based on orientation type
    print(f"DEBUG: frame_config.orientation = {frame_config.orientation}")
    if frame_config.orientation and hasattr(frame_config.orientation, 'value'):
        print(f"DEBUG: frame_config.orientation.value = {frame_config.orientation.value}")
        print(f"DEBUG: hasattr(frame_config.orientation.value, 'th') = {hasattr(frame_config.orientation.value, 'th')}")
        print(f"DEBUG: hasattr(frame_config.orientation.value, 'w') = {hasattr(frame_config.orientation.value, 'w')}")
        
        if 'th' in frame_config.orientation.value:
            # Axis-angle representation
            th = frame_config.orientation.value['th']
            axis = np.array([frame_config.orientation.value['x'], frame_config.orientation.value['y'], frame_config.orientation.value['z']])
            print(f"DEBUG: Axis-angle values - axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}], th: {th:.6f} degrees")
            # Convert axis-angle to rotation matrix
            R = call_go_ov2mat(axis[0], axis[1], axis[2], th)
            print(f"DEBUG: call_go_ov2mat result:")
            for i in range(3):
                print(f"  [{R[i,0]:.6f}, {R[i,1]:.6f}, {R[i,2]:.6f}]")
        elif 'w' in frame_config.orientation.value:
            # Quaternion representation
            q = np.array([frame_config.orientation.value['w'], frame_config.orientation.value['x'], 
                         frame_config.orientation.value['y'], frame_config.orientation.value['z']])
            print(f"DEBUG: Quaternion values - w: {q[0]:.6f}, x: {q[1]:.6f}, y: {q[2]:.6f}, z: {q[3]:.6f}")
            # Convert quaternion to rotation matrix
            w, x, y, z = q
            R = np.array([
                [1-2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)]
            ])
            print(f"DEBUG: Quaternion to rotation matrix result:")
            for i in range(3):
                print(f"  [{R[i,0]:.6f}, {R[i,1]:.6f}, {R[i,2]:.6f}]")
        else:
            print("DEBUG: No valid orientation found, using identity matrix")
            R = np.eye(3)
    else:
        print("DEBUG: No orientation found, using identity matrix")
        R = np.eye(3)
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

async def connect():
    load_dotenv('.env')  # Specify the .env file path
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



def _pose_to_matrix(pose: Pose) -> np.ndarray:
        """Convert a Viam Pose to a 4x4 homogeneous transformation matrix.

        Args:
            pose: Viam Pose object

        Returns:
            4x4 numpy array representing the homogeneous transformation
        """
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
        """Convert a 4x4 homogeneous transformation matrix to a Viam Pose.

        Args:
            T: 4x4 numpy array representing the homogeneous transformation

        Returns:
            Viam Pose object
        """
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



async def _get_current_camera_pose(pose_tracker: PoseTracker, body_name: str) -> Pose:
        pose_in_frame = await pose_tracker.get_poses(body_names=[body_name])
        if not pose_in_frame or body_name not in pose_in_frame:
            raise Exception(f"Could not find tracked body '{body_name}' in pose tracker observations")
        return pose_in_frame[body_name].pose

async def _get_corner_pose_in_robot_frame(pose_tracker: PoseTracker, machine, arm_name: str, body_name: str) -> Pose:
        """Get corner pose transformed to robot frame"""
        # Get corner pose in camera frame (as PoseInFrame)
        corner_poses = await pose_tracker.get_poses(body_names=[body_name])
        if body_name not in corner_poses:
            raise Exception(f"Could not find tracked body '{body_name}' in pose tracker observations")
        corner_pose_in_frame = corner_poses[body_name]
        
        # Transform to robot frame using machine's transform_pose method
        corner_pose_robot = await machine.transform_pose(
            corner_pose_in_frame,
            arm_name  # Robot's own frame
        )
        
        return corner_pose_robot

async def _get_current_arm_pose(motion: MotionClient, arm_name: str) -> Pose:
        pose_in_frame = await motion.get_pose(
            component_name=arm_name,
            destination_frame="world"
        )
        print(f"_get_current_arm_pose: Pose in frame: {pose_in_frame.pose}")
        return pose_in_frame.pose

async def get_camera_intrinsics(camera: Camera):
    """Get camera intrinsic parameters"""
    camera_params = await camera.do_command({"get_camera_params": None})
    
    # Handle different response structures
    if "Color" in camera_params:
        intrinsics = camera_params["Color"]["intrinsics"]
        dist_params = camera_params["Color"]["distortion"]
    elif "intrinsics" in camera_params:
        intrinsics = camera_params["intrinsics"]
        dist_params = camera_params["distortion"]
    else:
        raise Exception(f"Unexpected camera params structure. Keys found: {list(camera_params.keys())}")
    
    K = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist = np.array([
        dist_params["k1"],
        dist_params["k2"],  
        dist_params["p1"],  
        dist_params["p2"],    
        dist_params["k3"]
    ], dtype=np.float32)
    
    return K, dist

def detect_chessboard_rotation(img1, img2, pattern_size=(8,11), camera_matrix=None, dist_coeffs=None):
    """
    Calculate rotation between two checkerboard images using OpenCV
    
    Args:
        img1, img2: Two images containing checkerboard patterns
        pattern_size: Size of the checkerboard (inner corners)
        camera_matrix: Camera intrinsic matrix (optional, for more accurate results)
        dist_coeffs: Distortion coefficients (optional)
    
    Returns:
        rotation_matrix: 3x3 rotation matrix between the two poses
        translation_vector: 3x1 translation vector
        success: Boolean indicating if detection was successful
    """
    
    # Define checkerboard pattern size (inner corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (3D points of checkerboard corners)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    def detect_pose(img):
        """Detect checkerboard and return pose"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Solve PnP to get pose
            if camera_matrix is not None and dist_coeffs is not None:
                success, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
            else:
                # Use identity camera matrix if not provided (less accurate)
                dummy_camera_matrix = np.array([[1000, 0, img.shape[1]/2], [0, 1000, img.shape[0]/2], [0, 0, 1]], dtype=np.float32)
                success, rvec, tvec = cv2.solvePnP(objp, corners2, dummy_camera_matrix, None)
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                return rotation_matrix, tvec, corners2, True
            else:
                print("detect_pose: Failed to solve PnP")
                return None, None, None, False
        else:
            print("detect_pose: Failed to find corners")
            return None, None, None, False
    
    # Detect poses in both images
    R1, t1, corners1, success1 = detect_pose(img1)
    R2, t2, corners2, success2 = detect_pose(img2)
    
    if not (success1 and success2):
        print("Failed to detect checkerboard in one or both images")
        return None, None, False
    
    # Calculate relative rotation and translation
    # R_rel = R2 * R1^T (rotation from pose1 to pose2)
    R_rel = R2 @ R1.T
    
    # t_rel = t2 - R_rel * t1 (translation from pose1 to pose2)
    t_rel = t2 - R_rel @ t1
    
    return R_rel, t_rel, True

def rotation_matrix_to_axis_angle(R):
    """Convert rotation matrix to axis-angle representation"""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()

def axis_angle_to_degrees(rvec):
    """Convert axis-angle to degrees"""
    angle_rad = np.linalg.norm(rvec)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

async def capture_camera_image(camera: Camera):
    """Capture a single image from the camera"""
    try:
        cam_images = await camera.get_images()
        
        for cam_image in cam_images[0]:
            if cam_image.mime_type in [CameraMimeType.JPEG, CameraMimeType.PNG, CameraMimeType.VIAM_RGBA]:
                pil_image = viam_to_pil_image(cam_image)
                # Convert PIL to OpenCV format
                image = np.array(pil_image)
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                return image
        
        return None
    except Exception as e:
        print(f"Error capturing camera image: {e}")
        return None

async def compare_chessboard_poses(camera: Camera, pattern_size=(9,6)):
    """Capture two images and calculate rotation between them"""
    print("Capturing first image... Press Enter when ready.")
    input()
    
    img1 = await capture_camera_image(camera)
    if img1 is None:
        print("Failed to capture first image")
        return None, None, False
    
    print("Move the robot/chessboard, then press Enter to capture second image...")
    input()
    
    img2 = await capture_camera_image(camera)
    if img2 is None:
        print("Failed to capture second image")
        return None, None, False
    
    # Get camera intrinsics for more accurate results
    try:
        camera_matrix, dist_coeffs = await get_camera_intrinsics(camera)
    except:
        camera_matrix, dist_coeffs = None, None
        print("Could not get camera intrinsics, using dummy values")
    
    # Calculate rotation between images
    R_rel, t_rel, success = detect_chessboard_rotation(img1, img2, pattern_size, camera_matrix, dist_coeffs)
    
    if success:
        # Convert rotation matrix to axis-angle
        rvec = rotation_matrix_to_axis_angle(R_rel)
        angle_deg = axis_angle_to_degrees(rvec)
        
        print(f"Rotation between images:")
        print(f"  Rotation matrix:\n{R_rel}")
        print(f"  Axis-angle (degrees): {angle_deg:.2f}")
        print(f"  Translation: {t_rel.flatten()}")
        
        # Show both images with detected corners
        cv2.imshow("Image 1", img1)
        cv2.imshow("Image 2", img2)
        cv2.waitKey(0)
        
        return R_rel, t_rel, True
    else:
        print("Failed to detect checkerboard rotation")
        return None, None, False

def project_3d_to_2d(point_3d, camera_matrix):
    """Project 3D point to 2D pixel coordinates"""
    # Convert to homogeneous coordinates
    point_3d_homo = np.array([point_3d.x, point_3d.y, point_3d.z, 1.0])
    
    # Project to image plane (assuming no distortion for now)
    point_2d_homo = camera_matrix @ point_3d_homo[:3]
    
    # Convert back to pixel coordinates
    if point_2d_homo[2] != 0:  # Avoid division by zero
        pixel_x = int(point_2d_homo[0] / point_2d_homo[2])
        pixel_y = int(point_2d_homo[1] / point_2d_homo[2])
        return pixel_x, pixel_y
    else:
        return None, None

async def show_camera_overlay(camera: Camera, pt: PoseTracker, body_name: str, window_name: str = "Camera Overlay"):
    """Show camera feed with tracked body position overlaid"""
    try:
        # Get camera images
        cam_images = await camera.get_images()
        pil_image = None
        
        # Find a compatible image format
        for cam_image in cam_images[0]:
            if cam_image.mime_type in [CameraMimeType.JPEG, CameraMimeType.PNG, CameraMimeType.VIAM_RGBA]:
                pil_image = viam_to_pil_image(cam_image)
                break
        
        if pil_image is None:
            print("Could not get camera image")
            return None, None
            
        # Convert PIL to OpenCV format
        image = np.array(pil_image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get camera intrinsics
        camera_matrix, dist_coeffs = await get_camera_intrinsics(camera)
        
        # Get current pose of tracked body
        poses = await pt.get_poses(body_names=[body_name])
        if body_name in poses:
            pose = poses[body_name]
            print(f"Tracked body {body_name} at: x={pose.pose.x:.2f}, y={pose.pose.y:.2f}, z={pose.pose.z:.2f}")
            
            # Project 3D coordinates to 2D pixel coordinates
            pixel_x, pixel_y = project_3d_to_2d(pose.pose, camera_matrix)
            
            if pixel_x is not None and pixel_y is not None:
                # Check if point is within image bounds
                height, width = image.shape[:2]
                if 0 <= pixel_x < width and 0 <= pixel_y < height:
                    # Draw a circle to mark the tracked position
                    cv2.circle(image, (pixel_x, pixel_y), 15, (0, 255, 0), 3)  # Green circle
                    cv2.putText(image, f"{body_name}", (pixel_x + 20, pixel_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw crosshairs for better visibility
                    cv2.line(image, (pixel_x-20, pixel_y), (pixel_x+20, pixel_y), (0, 255, 0), 2)
                    cv2.line(image, (pixel_x, pixel_y-20), (pixel_x, pixel_y+20), (0, 255, 0), 2)
                else:
                    print(f"Projected point ({pixel_x}, {pixel_y}) is outside image bounds ({width}x{height})")
                    # Draw at center if out of bounds
                    center_x, center_y = width // 2, height // 2
                    cv2.circle(image, (center_x, center_y), 15, (0, 0, 255), 3)  # Red circle for out of bounds
                    cv2.putText(image, f"{body_name} (OOB)", (center_x + 20, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add pose info text
            info_text = f"3D: x:{pose.pose.x:.1f} y:{pose.pose.y:.1f} z:{pose.pose.z:.1f}"
            cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if pixel_x is not None and pixel_y is not None:
                info_text2 = f"2D: px:{pixel_x} py:{pixel_y}"
                cv2.putText(image, info_text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            info_text3 = f"theta:{pose.pose.theta:.1f}deg"
            cv2.putText(image, info_text3, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Draw "No tracking" message
            cv2.putText(image, f"No tracking for {body_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the image
        cv2.imshow(window_name, image)
        cv2.waitKey(1)  # Non-blocking wait
        
        return image, poses.get(body_name) if body_name in poses else None
        
    except Exception as e:
        print(f"Error in camera overlay: {e}")
        import traceback
        traceback.print_exc()
        return None, None


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
        await arm.do_command({"set_vel": 25})  # Set a default velocity
        camera = Camera.from_robot(machine, camera_name)
        # Print initial arm pose for utility and sanity check
        motion_service = MotionClient.from_robot(machine, motion_service_name)
        initial_arm_pose = await motion_service.get_pose(
            component_name=arm.name,
            destination_frame=DEFAULT_WORLD_FRAME
        )

        pt = PoseTracker.from_robot(machine, pose_tracker_name)

        # Robot connection is already established via 'machine' variable
        print(f"Connected to robot via machine: {machine}")
        
        # Get the frame configuration for the sensing-camera (Hand-Eye transformation)
        try:
            # Try to get robot configuration from app client using available methods
            robot_config = None
            
            # Method 1: Try get_robot_part to get robot part configuration
            try:
                # First, get the organizations to find the org_id
                organizations = await app_client.list_organizations()
                if organizations:
                    org = organizations[0]  # Get first organization
                    org_id = org.id
                    print(f"Found organization: {org_id}")
                    
                    # Get locations for this organization
                    locations = await app_client.list_locations(org_id=org_id)
                    if locations:
                        location = locations[0]  # Get first location
                        location_id = location.id
                        print(f"Found location: {location_id}")
                        
                        # Get robots for this location
                        robots = await app_client.list_robots(location_id=location_id)
                        if robots:
                            robot = robots[0]  # Get first robot
                            robot_id = robot.id
                            print(f"Found robot: {robot_id}")
                            
                            # Get robot parts
                            robot_parts = await app_client.get_robot_parts(robot_id)
                            if robot_parts:
                                robot_part = robot_parts[0]  # Get first robot part
                                robot_part_id = robot_part.id
                                print(f"Found robot part: {robot_part_id}")
                                
                            # Get robot part configuration
                            robot_part_config = await app_client.get_robot_part(robot_part_id)
                            print(f"Robot part config type: {type(robot_part_config)}")
                            print(f"Robot part config attributes: {[attr for attr in dir(robot_part_config) if not attr.startswith('_')]}")
                            
                            if robot_part_config:
                                # Try different possible attributes for the configuration
                                if hasattr(robot_part_config, 'config'):
                                    robot_config = robot_part_config.config
                                    print(f"Robot part config retrieved successfully via 'config' attribute")
                                elif hasattr(robot_part_config, 'robot_config'):
                                    robot_config = robot_part_config.robot_config
                                    print(f"Robot part config retrieved successfully via 'robot_config' attribute")
                                elif hasattr(robot_part_config, 'configuration'):
                                    robot_config = robot_part_config.configuration
                                    print(f"Robot part config retrieved successfully via 'configuration' attribute")
                                else:
                                    print(f"Robot part config found but no known config attribute")
                                    # Try to use the robot_part_config directly
                                    robot_config = robot_part_config
                                    print(f"Using robot_part_config directly")
                            else:
                                print(f"Robot part config is None")
                        else:
                            print(f"No robot parts found for robot {robot_id}")
            except Exception as e:
                print(f"Error getting robot part config: {e}")
            
            if robot_config:
                print(f"Robot config type: {type(robot_config)}")
                print(f"Robot config keys: {list(robot_config.keys())}")
                
                # Since robot_config is a dictionary, look for components
                if 'components' in robot_config:
                    components = robot_config['components']
                    print(f"Robot config has components: {len(components) if components else 0}")
                    if components:
                        print(f"Component names: {[comp.get('name', 'unnamed') for comp in components]}")
                    
                    # Find the sensing-camera component configuration
                    camera_config = None
                    for component in components:
                        if component.get('name') == camera_name:
                            camera_config = component
                            break
                else:
                    print(f"Robot config does not have 'components' key")
                    camera_config = None
                
                if camera_config and 'frame' in camera_config and camera_config['frame']:
                    frame_config = camera_config['frame']
                    print(f"\n=== SENSING-CAMERA FRAME CONFIGURATION (Hand-Eye Transformation) ===")
                    print(f"Frame config type: {type(frame_config)}")
                    print(f"Frame config: {frame_config}")
                    
                    # Handle frame configuration as dictionary
                    if isinstance(frame_config, dict):
                        parent = frame_config.get('parent', 'unknown')
                        translation = frame_config.get('translation', {})
                        orientation = frame_config.get('orientation', {})
                        
                        print(f"Parent frame: {parent}")
                        if translation:
                            print(f"Translation: x={translation.get('x', 0):.6f}, y={translation.get('y', 0):.6f}, z={translation.get('z', 0):.6f}")
                        
                        if orientation:
                            print(f"Orientation: {orientation}")
                            
                        # Try to convert to transformation matrix
                        try:
                            # Create a mock frame config object for the transformation matrix function
                            class MockFrameConfig:
                                def __init__(self, frame_dict):
                                    self.parent = frame_dict.get('parent', 'unknown')
                                    self.translation = type('Translation', (), frame_dict.get('translation', {}))()
                                    self.orientation = type('Orientation', (), frame_dict.get('orientation', {}))()
                            
                            mock_frame = MockFrameConfig(frame_config)
                            T_hand_eye = frame_config_to_transformation_matrix(mock_frame)
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
                            
                        except Exception as e:
                            print(f"Error converting frame to transformation matrix: {e}")
                    
                    print("=" * 70)
                else:
                    print(f"Warning: No frame configuration found for camera '{camera_name}'")
                    if camera_config:
                        print(f"Camera config keys: {list(camera_config.keys()) if isinstance(camera_config, dict) else 'Not a dict'}")
            else:
                print(f"Warning: Could not retrieve robot configuration from app client")
                
        except Exception as e:
            print(f"Error retrieving frame configuration: {e}")
            print("Continuing without frame configuration...")


        A_0_pose = await _get_current_arm_pose(motion_service, arm.name)
        # Convert to matrix
        T_A_0 = _pose_to_matrix(A_0_pose)
        B_0_pose = await _get_current_camera_pose(pt, body_names[0])
        # Convert to matrix
        T_B_0 = _pose_to_matrix(B_0_pose)


        B_0_pose_world = await machine.transform_pose(
            PoseInFrame(reference_frame=camera_name, pose=B_0_pose), 
            DEFAULT_WORLD_FRAME
        )
        current_pose_index = 0
        while True:
            response = input("Manually Move to next pose, press Enter to continue, type 'c' for chessboard comparison, or type 'q' to quit...")
            if response == "q":
                break
            elif response == "c":
                # Compare chessboard poses using OpenCV
                print("Starting chessboard pose comparison...")
                R_chess, t_chess, success = await compare_chessboard_poses(camera, pattern_size=(9,6))
                if success:
                    rvec_chess = rotation_matrix_to_axis_angle(R_chess)
                    angle_chess = axis_angle_to_degrees(rvec_chess)
                    print(f"OpenCV detected rotation: {angle_chess:.2f} degrees")
                continue
            elif response == "":

                current_pose_index += 1
                print(f"Current pose index: {current_pose_index}")
                
                # Calculate actual 3D rotation from transformation matrix for arm
                # Calculate the difference between the current pose and the starting pose
                A_i_pose = await _get_current_arm_pose(motion_service, arm.name)
                # Convert to matrix
                T_A_i = _pose_to_matrix(A_i_pose)
                T_delta_A = np.linalg.inv(T_A_0) @ T_A_i
                R_delta_A = T_delta_A[:3, :3]  # Extract rotation matrix
                rvec_delta_A, _ = cv2.Rodrigues(R_delta_A)
                angle_delta_A_3d = np.linalg.norm(rvec_delta_A) * 180 / np.pi
                
                print(f"\n=== ROBOT ROTATION DEBUGGING ===")
                print(f"T_A_0 (initial robot pose matrix):")
                print(f"  Translation: [{T_A_0[0,3]:.6f}, {T_A_0[1,3]:.6f}, {T_A_0[2,3]:.6f}]")
                print(f"  Rotation matrix:")
                for i in range(3):
                    print(f"    [{T_A_0[i,0]:.6f}, {T_A_0[i,1]:.6f}, {T_A_0[i,2]:.6f}]")
                
                print(f"\nT_A_i (current robot pose matrix):")
                print(f"  Translation: [{T_A_i[0,3]:.6f}, {T_A_i[1,3]:.6f}, {T_A_i[2,3]:.6f}]")
                print(f"  Rotation matrix:")
                for i in range(3):
                    print(f"    [{T_A_i[i,0]:.6f}, {T_A_i[i,1]:.6f}, {T_A_i[i,2]:.6f}]")
                
                print(f"\nT_delta_A (robot movement matrix):")
                print(f"  Translation: [{T_delta_A[0,3]:.6f}, {T_delta_A[1,3]:.6f}, {T_delta_A[2,3]:.6f}]")
                print(f"  Rotation matrix:")
                for i in range(3):
                    print(f"    [{T_delta_A[i,0]:.6f}, {T_delta_A[i,1]:.6f}, {T_delta_A[i,2]:.6f}]")
                
                print(f"\nR_delta_A (rotation matrix only):")
                for i in range(3):
                    print(f"  [{R_delta_A[i,0]:.6f}, {R_delta_A[i,1]:.6f}, {R_delta_A[i,2]:.6f}]")
                
                print(f"\nArm 3D rotation angle from transformation matrix: {angle_delta_A_3d:.6f} degrees")
                print(f"Arm rotation vector: {rvec_delta_A.flatten()}")
                print(f"Arm rotation vector magnitude: {np.linalg.norm(rvec_delta_A):.6f} radians")
                
                # Also check the pose-based rotation calculation
                A_0_pose = _matrix_to_pose(T_A_0)
                A_i_pose = _matrix_to_pose(T_A_i)
                pose_based_rotation = abs(A_i_pose.theta - A_0_pose.theta)
                print(f"\nPose-based rotation (theta difference): {pose_based_rotation:.6f} degrees")
                print(f"A_0_pose.theta: {A_0_pose.theta:.6f} degrees")
                print(f"A_i_pose.theta: {A_i_pose.theta:.6f} degrees")
                print("=" * 50)

                
                # Calculate actual 3D rotation from transformation matrix
                # T_delta_B contains the full 3D transformation between poses

                # Calculate the difference between the current pose and the starting pose
                B_i_pose = await _get_current_camera_pose(pt, body_names[0])
                # Convert to matrix
                T_B_i = _pose_to_matrix(B_i_pose)
                T_delta_B = np.linalg.inv(T_B_0) @ T_B_i
                R_delta_B = T_delta_B[:3, :3]  # Extract rotation matrix
                rvec_delta_B, _ = cv2.Rodrigues(R_delta_B)
                angle_delta_B_3d = np.linalg.norm(rvec_delta_B) * 180 / np.pi
                print(f"Camera rotation angle: {angle_delta_B_3d:.6f} degrees")
                print(f"Camera rotation vector: {rvec_delta_B.flatten()}")
                print(f"Camera rotation vector magnitude: {np.linalg.norm(rvec_delta_B):.6f} radians")

                # HAND-EYE CALIBRATION PREDICTION (X @ delta_B @ inv(X))
                # Based on: "Robust and Accurate Hand–Eye Calibration Method Based on Schur Matrix Decomposition"
                # https://pmc.ncbi.nlm.nih.gov/articles/PMC6832585/#sec4-sensors-19-04490
                
                print(f"\n=== FRAME REFERENCE DEBUGGING ===")
                print(f"Camera poses are in frame: {camera_name}")
                print(f"Robot poses are in frame: {DEFAULT_WORLD_FRAME}")
                print(f"Hand-Eye transformation: camera -> {arm_name}")
                
                print(f"\n=== TRANSFORMATION MATRICES DEBUG ===")
                print(f"T_hand_eye (camera -> robot):")
                print(f"  Translation: [{T_hand_eye[0,3]:.6f}, {T_hand_eye[1,3]:.6f}, {T_hand_eye[2,3]:.6f}]")
                print(f"  Rotation matrix:")
                for i in range(3):
                    print(f"    [{T_hand_eye[i,0]:.6f}, {T_hand_eye[i,1]:.6f}, {T_hand_eye[i,2]:.6f}]")
                
                print(f"\nT_delta_B (camera movement in camera frame):")
                print(f"  Translation: [{T_delta_B[0,3]:.6f}, {T_delta_B[1,3]:.6f}, {T_delta_B[2,3]:.6f}]")
                print(f"  Rotation matrix:")
                for i in range(3):
                    print(f"    [{T_delta_B[i,0]:.6f}, {T_delta_B[i,1]:.6f}, {T_delta_B[i,2]:.6f}]")
                
                print(f"\nT_delta_A (robot movement in world frame):")
                print(f"  Translation: [{T_delta_A[0,3]:.6f}, {T_delta_A[1,3]:.6f}, {T_delta_A[2,3]:.6f}]")
                print(f"  Rotation matrix:")
                for i in range(3):
                    print(f"    [{T_delta_A[i,0]:.6f}, {T_delta_A[i,1]:.6f}, {T_delta_A[i,2]:.6f}]")
                
                # CRITICAL FIX: Transform camera movement to world frame first
                # Camera movement is in camera frame, but robot movement is in world frame
                # We need to transform camera movement to world frame before applying Hand-Eye transformation
                
                # Transform camera movement from camera frame to world frame
                # T_delta_B_world = T_hand_eye @ T_delta_B @ inv(T_hand_eye)
                T_delta_B_world = T_hand_eye @ T_delta_B @ np.linalg.inv(T_hand_eye)
                
                print(f"\nT_delta_B_world (camera movement transformed to world frame):")
                print(f"  Translation: [{T_delta_B_world[0,3]:.6f}, {T_delta_B_world[1,3]:.6f}, {T_delta_B_world[2,3]:.6f}]")
                print(f"  Rotation matrix:")
                for i in range(3):
                    print(f"    [{T_delta_B_world[i,0]:.6f}, {T_delta_B_world[i,1]:.6f}, {T_delta_B_world[i,2]:.6f}]")
                
                # Now both movements are in world frame - they should be comparable
                # The Hand-Eye transformation should make them equal: T_delta_B_world ≈ T_delta_A
                T_A_predicted = T_delta_B_world  # This is the corrected prediction
                
                print(f"\nT_A_predicted (predicted robot movement):")
                print(f"  Translation: [{T_A_predicted[0,3]:.6f}, {T_A_predicted[1,3]:.6f}, {T_A_predicted[2,3]:.6f}]")
                print(f"  Rotation matrix:")
                for i in range(3):
                    print(f"    [{T_A_predicted[i,0]:.6f}, {T_A_predicted[i,1]:.6f}, {T_A_predicted[i,2]:.6f}]")
                
                # Calculate rotation angles directly from transformation matrices
                # This avoids the pose conversion issue
                R_predicted = T_A_predicted[:3, :3]
                R_actual = T_delta_A[:3, :3]
                
                # Convert rotation matrices to rotation vectors
                rvec_predicted, _ = cv2.Rodrigues(R_predicted)
                rvec_actual, _ = cv2.Rodrigues(R_actual)
                
                # Calculate rotation angles
                angle_predicted = np.linalg.norm(rvec_predicted) * 180 / np.pi
                angle_actual = np.linalg.norm(rvec_actual) * 180 / np.pi
                
                # Calculate prediction errors
                translation_error = np.linalg.norm([
                    T_A_predicted[0,3] - T_delta_A[0,3],
                    T_A_predicted[1,3] - T_delta_A[1,3],
                    T_A_predicted[2,3] - T_delta_A[2,3]
                ])
                rotation_error = abs(angle_predicted - angle_actual)
                
                # Also convert to poses for display
                A_predicted_pose = _matrix_to_pose(T_A_predicted)
                A_actual_pose = _matrix_to_pose(T_delta_A)
                
                print(f"\n=== HAND-EYE CALIBRATION PREDICTION (X @ delta_B @ inv(X)) ===")
                print(f"Camera movement (delta_B) - rotation: {angle_delta_B_3d:.6f} degrees")
                print(f"Hand-Eye transformation (X) retrieved from robot config")
                
                print(f"\nPredicted robot movement (A_predicted):")
                print(f"  Translation: [{T_A_predicted[0,3]:.6f}, {T_A_predicted[1,3]:.6f}, {T_A_predicted[2,3]:.6f}]")
                print(f"  Rotation: {angle_predicted:.6f} degrees (from matrix)")
                print(f"  Rotation (pose): {A_predicted_pose.theta:.6f} degrees (from pose conversion)")
                
                print(f"\nActual robot movement (A_actual):")
                print(f"  Translation: [{T_delta_A[0,3]:.6f}, {T_delta_A[1,3]:.6f}, {T_delta_A[2,3]:.6f}]")
                print(f"  Rotation: {angle_actual:.6f} degrees (from matrix)")
                print(f"  Rotation (pose): {A_actual_pose.theta:.6f} degrees (from pose conversion)")
                
                print(f"\n=== PREDICTION ERRORS ===")
                print(f"Translation error: {translation_error:.6f} mm")
                print(f"Rotation error: {rotation_error:.6f} degrees")
                
                # Interpretation
                if rotation_error < 1.0 and translation_error < 5.0:
                    print(f"✅ Hand-Eye calibration is GOOD (errors < 1° and 5mm)")
                elif rotation_error < 5.0 and translation_error < 20.0:
                    print(f"⚠️  Hand-Eye calibration is ACCEPTABLE (errors < 5° and 20mm)")
                else:
                    print(f"❌ Hand-Eye calibration needs IMPROVEMENT (errors > 5° or 20mm)")
                
                print("=" * 60)
    except Exception as e:
        print("Caught exception in script main: ")
        raise e
    finally:
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
        
        if pt:
            await pt.close()
        if arm:
            await arm.close()
        if motion_service:
            await motion_service.close()
        if machine:
            await machine.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose test script')
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