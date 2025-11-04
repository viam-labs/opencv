import asyncio
import cv2
import numpy as np
from typing import ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm, JointPositions
from viam.components.camera import Camera
from viam.components.pose_tracker import PoseTracker
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import *
from viam.services.motion import Motion
from viam.utils import struct_to_dict, ValueTypes
from viam.media.video import CameraMimeType
from viam.media.utils.pil import viam_to_pil_image

from utils.utils import call_go_ov2mat, call_go_mat2ov

CALIBS = ["eye-in-hand", "eye-to-hand"]
METHODS = [
    "CALIB_HAND_EYE_TSAI",
    "CALIB_HAND_EYE_PARK",
    "CALIB_HAND_EYE_HORAUD",
    "CALIB_HAND_EYE_ANDREFF",
    "CALIB_HAND_EYE_DANIILIDIS"
]

# required attributes
ARM_ATTR = "arm_name"
BODY_NAME_ATTR = "body_name"
USE_INTERNAL_POSE_TRACKER_ATTR = "use_internal_pose_tracker"
CALIB_ATTR = "calibration_type"
CAM_ATTR = "camera_name"
JOINT_POSITIONS_ATTR = "joint_positions"
METHOD_ATTR = "method"
MOTION_ATTR = "motion"
POSE_TRACKER_ATTR = "pose_tracker"
SLEEP_ATTR = "sleep_seconds"
PATTERN_SIZE_ATTR = "pattern_size"
SQUARE_SIZE_MM_ATTR = "square_size_mm"

# Default config attribute values
DEFAULT_SLEEP_SECONDS = 2.0
DEFAULT_METHOD = "CALIB_HAND_EYE_TSAI"


class HandEyeCalibration(Generic, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam", "opencv"), "hand_eye_calibration"
    )

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Generic service.
        The default implementation sets the name from the `config` parameter and then calls `reconfigure`.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both required and optional)

        Returns:
            Self: The resource
        """
        return super().new(config, dependencies)

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any required dependencies or optional dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Tuple[Sequence[str], Sequence[str]]: A tuple where the
                first element is a list of required dependencies and the
                second element is a list of optional dependencies
        """
        attrs = struct_to_dict(config.attributes)

        cam = attrs.get(CAM_ATTR)
        if cam is None:
            raise Exception(f"Missing required {CAM_ATTR} attribute.")
        
        arm = attrs.get(ARM_ATTR)
        if arm is None:
            raise Exception(f"Missing required {ARM_ATTR} attribute.")
        
        if attrs.get(JOINT_POSITIONS_ATTR) is None:
            raise Exception(f"Missing required {JOINT_POSITIONS_ATTR} attribute.")
        
        pose_tracker = attrs.get(POSE_TRACKER_ATTR)
        if pose_tracker is None:
            raise Exception(f"Missing required {POSE_TRACKER_ATTR} attribute.")
        
        calib = attrs.get(CALIB_ATTR)
        if calib is None:
            raise Exception(f"Missing required {CALIB_ATTR} attribute.")
        if calib not in CALIBS:
            raise Exception(f"{calib} is not an available calibration.")
        
        method = attrs.get(METHOD_ATTR)
        if method not in METHODS:
            raise Exception(f"{method} is not an available method for calibration.")

        body_name = attrs.get(BODY_NAME_ATTR)
        if body_name is not None:
            if not isinstance(body_name, str):
                raise Exception(f"'{BODY_NAME_ATTR}' must be a string, got {type(body_name)}")

        return [str(arm), str(cam), str(pose_tracker)], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both required and optional)
        """
        attrs = struct_to_dict(config.attributes)

        camera: str = attrs.get(CAM_ATTR)
        self.camera: Camera = dependencies.get(Camera.get_resource_name(camera))
        
        arm: Arm = attrs.get(ARM_ATTR)
        self.arm: Arm = dependencies.get(Arm.get_resource_name(arm))

        pose_tracker: PoseTracker = attrs.get(POSE_TRACKER_ATTR)
        self.pose_tracker: PoseTracker = dependencies.get(PoseTracker.get_resource_name(pose_tracker))

        motion: Motion = attrs.get(MOTION_ATTR)
        self.motion: Motion = dependencies.get(Motion.get_resource_name(motion))

        self.calib = attrs.get(CALIB_ATTR)
        self.joint_positions = attrs.get(JOINT_POSITIONS_ATTR, [])
        self.method = attrs.get(METHOD_ATTR, DEFAULT_METHOD)
        self.sleep_seconds = attrs.get(SLEEP_ATTR, DEFAULT_SLEEP_SECONDS)
        self.body_names = [attrs.get(BODY_NAME_ATTR)] if attrs.get(BODY_NAME_ATTR) is not None else []
        self.use_internal_pose_tracker = attrs.get(USE_INTERNAL_POSE_TRACKER_ATTR, False)
        self.pattern_size = attrs.get(PATTERN_SIZE_ATTR, [11, 8])
        self.square_size = attrs.get(SQUARE_SIZE_MM_ATTR, 20.0)

        return super().reconfigure(config, dependencies)

    async def get_camera_image(self) -> np.ndarray:
        cam_images = await self.camera.get_images()
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

    async def get_camera_intrinsics(self) -> tuple:
        """Get camera intrinsic parameters"""
        camera_params = await self.camera.do_command({"get_camera_params": None})
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

    async def get_calibration_values_from_chessboard(self):
        chessboard_size = self.pattern_size
        square_size = self.square_size
        image = await self.get_camera_image()
        camera_matrix, dist_coeffs = await self.get_camera_intrinsics()
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
    
        # Prepare 3D object points (chessboard corners in world coordinates)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
        objp *= square_size
    
        # Find chessboard corners using selected method
        corners = None
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
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

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)

        R_cam2target = R.T
        t_cam2target = -R.T @ t

        return R_cam2target, t_cam2target
    
    async def get_calibration_values(self):
        arm_pose = await self.arm.get_end_position()
        self.logger.debug(f"Found end of arm pose: {arm_pose}")

        # Get rotation matrix: base -> gripper
        R_base2gripper = call_go_ov2mat(
            arm_pose.o_x,
            arm_pose.o_y,
            arm_pose.o_z,
            arm_pose.theta
        )
        t_base2gripper = np.array([[arm_pose.x], [arm_pose.y], [arm_pose.z]], dtype=np.float64)

        # Get pose from the tracker (AprilTag, chessboard corner, etc.)
        tracked_poses: Dict[str, PoseInFrame] = await self.pose_tracker.get_poses(body_names=self.body_names)
        if tracked_poses is None or len(tracked_poses) == 0:
            no_bodies_error_msg = "could not find any tracked bodies in camera frame"
            if self.body_names:
                no_bodies_error_msg += f" (looking for: {self.body_names})"
            no_bodies_error_msg += ". Check to make sure a calibration target is in view."
            self.logger.warning(no_bodies_error_msg)
            raise Exception(no_bodies_error_msg)
        if len(tracked_poses.items()) > 1:
            multiple_bodies_error_msg = (
                f"more than 1 body detected was returned from the pose tracker: {tracked_poses.keys()}."
                " Ensure only one calibration target is visible or set the 'body_name' config attribute"
                " to one of the aforementioned body names to filter for a specific tracked body."
            )
            self.logger.warning(multiple_bodies_error_msg)
            raise Exception(multiple_bodies_error_msg)
        # Get rotation matrix: camera -> target
        tracked_pose: Pose = list(tracked_poses.values())[0].pose
        R_cam2target = call_go_ov2mat(
            tracked_pose.o_x,
            tracked_pose.o_y,
            tracked_pose.o_z,
            tracked_pose.theta
        )
        t_cam2target = np.array([[tracked_pose.x], [tracked_pose.y], [tracked_pose.z]], dtype=np.float64)

        return R_base2gripper, t_base2gripper, R_cam2target, t_cam2target

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        resp = {}
        for key, value in command.items():
            match key:
                case "run_calibration":
                    # Initialize data collection lists
                    R_gripper2base_list = []
                    t_gripper2base_list = []
                    R_target2cam_list = []
                    t_target2cam_list = []

                    # TODO: Use motion planning if available
                    if self.motion is None:
                        for i, joints in enumerate(self.joint_positions):
                            R_base2gripper = None
                            t_base2gripper = None
                            R_cam2target = None
                            t_cam2target = None

                            self.logger.debug(f"Moving to pose {i+1}/{len(self.joint_positions)}")

                            # Python SDK requires degrees for joint positions
                            joints_deg = [np.degrees(joint) for joint in joints]
                            jp = JointPositions(values=joints_deg)
                            await self.arm.move_to_joint_positions(jp)
                            while await self.arm.is_moving():
                                await asyncio.sleep(0.05)
                            self.logger.debug(f"Moved arm to position: {jp}")

                            # Sleep for the configured amount of time to allow the arm and camera to settle
                            await asyncio.sleep(self.sleep_seconds)

                            R_base2gripper = None
                            t_base2gripper = None
                            R_cam2target = None
                            t_cam2target = None

                            if(self.use_internal_pose_tracker):
                                R_base2gripper, t_base2gripper, R_cam2target, t_cam2target = await self.get_calibration_values_from_chessboard()
                            else:
                                R_base2gripper, t_base2gripper, R_cam2target, t_cam2target = await self.get_calibration_values()
                            if R_base2gripper is None or t_base2gripper is None or R_cam2target is None or t_cam2target is None:
                                self.logger.warning(f"Could not find calibration values for pose {i+1}/{len(self.joint_positions)}")
                                continue

                            self.logger.info(f"successfully collected calibration data for pose {i+1}/{len(self.joint_positions)}")

                            # TODO: Implement eye-to-hand. This should just be changing the
                            # order/in-frame values

                            # OpenCV calibrateHandEye expects transposed rotations but original translation vectors
                            # Only invert rotation matrices, keep translation vectors as-is
                            R_gripper2base = R_base2gripper.T
                            t_gripper2base = t_base2gripper
                            R_target2cam = R_cam2target.T
                            t_target2cam = t_cam2target
                            
                            R_gripper2base_list.append(R_gripper2base)
                            t_gripper2base_list.append(t_gripper2base)
                            R_target2cam_list.append(R_target2cam)
                            t_target2cam_list.append(t_target2cam)
                    else:
                        # TODO: Implement motion service code here
                        try:
                            success = await self.motion.move(
                                component_name=self.arm.name,
                                destination=None,
                            )
                        except Exception as e:
                            raise Exception(e)

                    # Check if we have enough measurements
                    if len(R_gripper2base_list) < 3:
                        raise Exception(f"not enough valid measurements collected. Got {len(R_gripper2base_list)}, need at least 3. Make sure the pose tracker can see exactly one target in each calibration position.")

                    self.logger.info(f"collected {len(R_gripper2base_list)} measurements, running calibration...")

                    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                        R_gripper2base=R_gripper2base_list,
                        t_gripper2base=t_gripper2base_list,
                        R_target2cam=R_target2cam_list,
                        t_target2cam=t_target2cam_list,
                        method=getattr(cv2, self.method)
                    )
                    if R_cam2gripper is None or t_cam2gripper is None:
                        raise Exception("could not solve calibration")
                    
                    # Convert OpenCV output to frame system format
                    # Transpose rotation but keep translation as-is (consistent with input handling)
                    R_gripper2cam = R_cam2gripper.T
                    t_gripper2cam = t_cam2gripper.reshape(3, 1)
                    
                    # Rotation matrix to orientation vector
                    orientation_result = call_go_mat2ov(R_gripper2cam)
                    if orientation_result is None:
                        raise Exception("failed to convert rotation matrix to orientation vector")
                    ox, oy, oz, theta = orientation_result
                    
                    # Translation
                    x = float(t_gripper2cam[0][0])
                    y = float(t_gripper2cam[1][0])
                    z = float(t_gripper2cam[2][0])

                    viam_pose = Pose(x=x, y=y, z=z, o_x=ox, o_y=oy, o_z=oz, theta=theta)

                    self.logger.info(f"calibration success: {viam_pose}")

                    # Format output to be frame system compatible
                    resp["run_calibration"] = {
                        "frame": {
                            "translation": {
                                "x": x,
                                "y": y,
                                "z": z
                            },
                            "orientation": {
                                "type": "ov_degrees",
                                "value": {
                                    "x": ox,
                                    "y": oy,
                                    "z": oz,
                                    "th": theta
                                }
                            },
                            "parent": self.arm.name
                        }
                    }
                case "move_arm": 
                    raise NotImplementedError("This is not yet implemented")
                case "check_tags":
                    tracked_poses: Dict[str, PoseInFrame] = await self.pose_tracker.get_poses(body_names=self.body_names)
                    if tracked_poses is None or len(tracked_poses) == 0:
                        resp["check_tags"] = "No tracked bodies found in image"
                        break

                    resp["check_tags"] = f"Number of tracked bodies seen: {len(tracked_poses)}"
                case "save_calibration_position":
                    index = int(value)

                    arm_joint_pos = await self.arm.get_joint_positions()
                    if arm_joint_pos is None:
                        resp["save_calibration_position"] = "Could not get joint positions of arm"
                        break

                    if index < 0:
                        self.joint_positions.append(arm_joint_pos)
                        resp["save_calibration_position"] = f"joint position {len(self.joint_positions) - 1} added to config"
                    elif index >= len(self.joint_positions):
                        resp["save_calibration_position"] = f"index {value} is out of range, only {len(self.joint_positions)} are set."
                    else:
                        self.joint_positions[index] = arm_joint_pos
                        resp["save_calibration_position"] = f"joint position {index} updated in config"

                    # TODO: Update config with updated joint positions array
                case "move_arm_to_position":
                    index = int(value)

                    if index >= len(self.joint_positions):
                        resp["move_arm_to_position"] = f"position {index} invalid since there are only {len(self.joint_positions)} positions"

                    joint_pos = self.joint_positions[index]

                    if self.motion is None:
                        jp = JointPositions(joint_pos)
                        arm_joint_pos = self.arm.move_to_joint_positions(jp)

                        # Sleep to allow time for the camera and arm to settle
                        asyncio.sleep(self.sleep_seconds)

                    # TODO: Implement using motion service 

                    tracked_poses: dict = await self.pose_tracker.get_poses(body_names=self.body_names)
                    if tracked_poses is None:
                        resp["move_arm_to_position"] = "No tracked bodies found in image"
                        break

                    resp["move_arm_to_position"] = len(tracked_poses)
                case "delete_calibration_position":
                    index = int(value)
                    if index >= len(self.joint_positions):
                        resp["delete_calibration_position"] = f"index {index} is out of range of list with length {len(self.joint_positions)}"

                    del self.joint_positions[index]

                    resp["delete_calibration_position"] = f"position {index} deleted"
                case "clear_calibration_positions":
                    self.joint_positions = []

                    resp["clear_calibration_positions"] = "all calibration positions removed"
                case "auto_calibrate":
                    raise NotImplementedError("This is not yet implemented")
                case _:
                    resp[key] = "unsupported key"

        if len(resp) == 0:
            return None, "no valid do_command submitted"
        
        return resp
