import asyncio
import cv2
import numpy as np
import uuid
import os
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
from utils.camera_utils import get_camera_image, get_camera_intrinsics
from utils.pose_utils import get_chessboard_pose_in_camera_frame

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
CAMERA_NAME_ATTR = "camera_name"
CALIB_ATTR = "calibration_type"
JOINT_POSITIONS_ATTR = "joint_positions"
POSES_ATTR = "poses"
METHOD_ATTR = "method"
MOTION_ATTR = "motion"
POSE_TRACKER_ATTR = "pose_tracker"
SLEEP_ATTR = "sleep_seconds"
PATTERN_SIZE_ATTR = "pattern_size"
SQUARE_SIZE_MM_ATTR = "square_size_mm"
WEB_APP_RESOURCE_NAME_ATTR = "web_app_resource_name"
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

        arm = attrs.get(ARM_ATTR)
        if arm is None:
            raise Exception(f"Missing required {ARM_ATTR} attribute.")

        # Either joint_positions or poses must be provided
        joint_positions = attrs.get(JOINT_POSITIONS_ATTR)
        poses = attrs.get(POSES_ATTR)
        if joint_positions is None and poses is None:
            raise Exception(f"Must provide either {JOINT_POSITIONS_ATTR} or {POSES_ATTR} attribute.")

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

        use_internal_pose_tracker = attrs.get(USE_INTERNAL_POSE_TRACKER_ATTR, False)
        if use_internal_pose_tracker:
            camera_name = attrs.get(CAMERA_NAME_ATTR)
            if camera_name is None:
                raise Exception(f"When {USE_INTERNAL_POSE_TRACKER_ATTR} is True, {CAMERA_NAME_ATTR} is required.")

        web_app_resource_name = attrs.get(WEB_APP_RESOURCE_NAME_ATTR)
        if web_app_resource_name is not None:
            if not isinstance(web_app_resource_name, str):
                raise Exception(f"'{WEB_APP_RESOURCE_NAME_ATTR}' must be a string, got {type(web_app_resource_name)}")

        motion = attrs.get(MOTION_ATTR)
        optional_deps = []
        if motion is not None:
            optional_deps.append(str(motion))

        required_deps = [str(arm), str(pose_tracker)]
        if use_internal_pose_tracker:
            required_deps.append(str(camera_name))

        return required_deps, optional_deps

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """This method allows you to dynamically update your service when it receives a new `config` object.

        Args:
            config (ComponentConfig): The new configuration
            dependencies (Mapping[ResourceName, ResourceBase]): Any dependencies (both required and optional)
        """
        attrs = struct_to_dict(config.attributes)

        arm = attrs.get(ARM_ATTR)
        self.arm: Arm = dependencies.get(Arm.get_resource_name(arm))

        pose_tracker = attrs.get(POSE_TRACKER_ATTR)
        self.pose_tracker: PoseTracker = dependencies.get(PoseTracker.get_resource_name(pose_tracker))

        motion = attrs.get(MOTION_ATTR)
        self.motion: Optional[Motion] = dependencies.get(Motion.get_resource_name(motion)) if motion else None
        if self.motion is not None:
            self.logger.debug(f"Motion service configured: {self.motion.name}")
        else:
            self.logger.debug("No motion service configured, using direct joint position control")


        # Parse joint positions or poses from config
        self.joint_positions = attrs.get(JOINT_POSITIONS_ATTR, [])
        poses_config = attrs.get(POSES_ATTR, [])
        self.poses = []
        for pose_dict in poses_config:
            pose = Pose(
                x=pose_dict.get("x", 0),
                y=pose_dict.get("y", 0),
                z=pose_dict.get("z", 0),
                o_x=pose_dict.get("o_x", 0),
                o_y=pose_dict.get("o_y", 0),
                o_z=pose_dict.get("o_z", 1),
                theta=pose_dict.get("theta", 0)
            )
            self.poses.append(pose)

        self.method = attrs.get(METHOD_ATTR, DEFAULT_METHOD)
        self.sleep_seconds = attrs.get(SLEEP_ATTR, DEFAULT_SLEEP_SECONDS)
        self.body_names = [attrs.get(BODY_NAME_ATTR)] if attrs.get(BODY_NAME_ATTR) is not None else []
    
        self.use_internal_pose_tracker = attrs.get(USE_INTERNAL_POSE_TRACKER_ATTR, False)
        if self.use_internal_pose_tracker:
            camera_name = attrs.get(CAMERA_NAME_ATTR)
            if camera_name is None:
                raise Exception(f"When {USE_INTERNAL_POSE_TRACKER_ATTR} is True, {CAMERA_NAME_ATTR} is required.")
            pattern_size_raw = attrs.get(PATTERN_SIZE_ATTR, [11, 8])
            # Ensure pattern_size is a list of integers (config might have floats)
            if not isinstance(pattern_size_raw, (list, tuple)) or len(pattern_size_raw) != 2:
                raise Exception(f"{PATTERN_SIZE_ATTR} must be a list or tuple of 2 integers (rows, cols), got: {pattern_size_raw}")
            self.pattern_size = [int(pattern_size_raw[0]), int(pattern_size_raw[1])]
            self.square_size = attrs.get(SQUARE_SIZE_MM_ATTR, 20.0)
            self.camera = dependencies.get(Camera.get_resource_name(camera_name))
            if self.camera is None:
                raise Exception(f"Camera dependency '{camera_name}' not found in dependencies.")

        web_app_resource_name = attrs.get(WEB_APP_RESOURCE_NAME_ATTR)
        self.web_app: EasyResource = dependencies.get(web_app_resource_name) if web_app_resource_name else None
        if self.web_app is not None:
            self.logger.debug(f"Web app service configured: {self.web_app.name}")
        else:
            self.logger.debug("No web app service configured")

        web_app_resource_name = attrs.get(WEB_APP_RESOURCE_NAME_ATTR)
        self.web_app: Optional[Generic] = None
        if web_app_resource_name:
            print(f"web_app_resource_name: {web_app_resource_name}")
            if isinstance(web_app_resource_name, str):
                dependency_name = Generic.get_resource_name(web_app_resource_name)
            elif isinstance(web_app_resource_name, Mapping):
                dependency_name = ResourceName(**web_app_resource_name)
            else:
                raise Exception(
                    f"'{WEB_APP_RESOURCE_NAME_ATTR}' must be a string or resource name mapping, got {type(web_app_resource_name)}"
                )

            dependency_name.type = "service"
            dependency_name.namespace = dependency_name.namespace or "rdk"

            self.web_app = dependencies.get(dependency_name)

        if self.web_app is not None:
            self.logger.debug(f"Web app service configured: {self.web_app.name}")
        else:
            self.logger.debug("No web app service configured")

        return super().reconfigure(config, dependencies)

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

    async def _move_arm_to_position(self, position_data, position_index, total_positions):
        """Move arm to specified position using joint control, direct pose control, or motion planning.

        Args:
            position_data: Either a list of joint positions (radians) or a Pose object
            position_index: Index of the current position
            total_positions: Total number of positions
        """
        self.logger.debug(f"Moving to position {position_index+1}/{total_positions}")

        is_pose = isinstance(position_data, Pose)

        if self.motion is not None and is_pose:
            # hack to get the arm to move relative to the base of the arm, not the end TCP
            pif = PoseInFrame(reference_frame=self.arm.name + "_origin", pose=position_data)

            success = await self.motion.move(
                component_name=self.arm.name,
                destination=pif,
            )
            if not success:
                raise Exception(f"Could not move to pose {position_index+1}/{total_positions}")
            self.logger.debug(f"Moved arm to pose using motion planning")
        elif is_pose:
            # Direct pose control using arm.move_to_position
            try:
                await self.arm.move_to_position(pose=position_data)
            except Exception as e:
                self.logger.error(f"Could not move arm to pose. If the arm does not implement move_to_position, use motion service instead. Error: {e}")
                raise e

            while await self.arm.is_moving():
                await asyncio.sleep(0.05)
            self.logger.debug(f"Moved arm to pose: {position_data}")
        else:
            # Direct joint position control
            joints_deg = [np.degrees(joint) for joint in position_data]
            jp = JointPositions(values=joints_deg)
            await self.arm.move_to_joint_positions(jp)
            while await self.arm.is_moving():
                await asyncio.sleep(0.05)
            self.logger.debug(f"Moved arm to joint position: {jp}")

    async def _collect_calibration_data(self):
        """Collect calibration data for all joint positions or poses."""
        R_gripper2base_list = []
        t_gripper2base_list = []
        R_target2cam_list = []
        t_target2cam_list = []

        # Use poses if available (motion planning mode), otherwise use joint positions
        positions = self.poses if self.poses else self.joint_positions
        total_positions = len(positions)

        for i, position in enumerate(positions):
            try:
                await self._move_arm_to_position(position, i, total_positions)
                await asyncio.sleep(self.sleep_seconds)

                if self.use_internal_pose_tracker:
                    print("Using internal pose tracker")
                    # Get arm pose separately
                    arm_pose = await self.arm.get_end_position()
                    R_base2gripper = call_go_ov2mat(
                        arm_pose.o_x,
                        arm_pose.o_y,
                        arm_pose.o_z,
                        arm_pose.theta
                    )
                    t_base2gripper = np.array([[arm_pose.x], [arm_pose.y], [arm_pose.z]], dtype=np.float64)
                    
                    # Get camera-to-target pose from chessboard
                    image = await get_camera_image(self.camera)
                    camera_matrix, dist_coeffs = await get_camera_intrinsics(self.camera)
                    success, rvec, tvec, R_cam2target, t_cam2target, corners, validation_info = get_chessboard_pose_in_camera_frame(image, camera_matrix, dist_coeffs, self.pattern_size)
                    if not success:
                        self.logger.warning(f"Could not find calibration values from chessboard for position {i+1}/{total_positions}")
                        continue
                else:
                    R_base2gripper, t_base2gripper, R_cam2target, t_cam2target = await self.get_calibration_values()

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

                self.logger.info(f"successfully collected calibration data for position {i+1}/{total_positions}")

            except Exception as e:
                error_msg = str(e)
                # Check if this is a timeout or cancellation error (recoverable)
                # Handle tuple errors like (Status.CANCELLED, 'cbirrt timeout context canceled', None)
                error_str = str(e)
                is_timeout = (
                    "timeout" in error_str.lower() 
                    or "cancelled" in error_str.lower() 
                    or "CANCELLED" in error_str
                    or (isinstance(e, tuple) and len(e) > 0 and "CANCELLED" in str(e[0]))
                )
                
                if is_timeout:
                    self.logger.warning(f"Skipping position {i+1}/{total_positions} due to timeout/cancellation: {error_msg}")
                    self.logger.warning(f"Continuing with remaining positions. Progress: {len(R_gripper2base_list)}/{total_positions} collected so far")
                    continue
                else:
                    # For other errors, log and raise
                    error_msg_full = f"Could not find calibration values for position {i+1}/{total_positions}: {error_msg}"
                    self.logger.error(error_msg_full)
                    raise Exception(error_msg_full)
        
        # Check if we have enough measurements after skipping any positions
        if len(R_gripper2base_list) < 3:
            raise Exception(
                f"Not enough valid measurements collected after processing {total_positions} positions. "
                f"Got {len(R_gripper2base_list)}, need at least 3. "
                f"Some positions may have been skipped due to timeouts or errors. "
                f"Consider adjusting motion planning parameters or checking pose reachability."
            )
        
        return R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list

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
                case "run_simulated_calibration":
                    print("running simulated calibration")
                    calibration_id = str(uuid.uuid4())
                    tracking_dir = await self._resolve_tracking_directory(calibration_id)
                    resp["run_simulated_calibration"] = {
                        "calibration_id": calibration_id,
                        "tracking_directory": tracking_dir
                    }

                    for i in range(10):
                        np.save(os.path.join(tracking_dir, f"data_{i}.npy"), np.random.rand(10, 10))

                case "run_calibration":
                    R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list = await self._collect_calibration_data()

                    # Note: Minimum measurements check is done in _collect_calibration_data()

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
                case "check_bodies":
                    tracked_poses: Dict[str, PoseInFrame] = await self.pose_tracker.get_poses(body_names=self.body_names)
                    if tracked_poses is None or len(tracked_poses) == 0:
                        resp["check_bodies"] = "No tracked bodies found in image"
                        break

                    resp["check_bodies"] = f"Number of tracked bodies seen: {len(tracked_poses)}"
                case "get_current_arm_pose":
                    arm_pose = await self.arm.get_end_position()
                    if arm_pose is None:
                        resp["get_current_arm_pose"] = "Could not get end position of arm"
                        break

                    resp["get_current_arm_pose"] = {
                        "x": arm_pose.x,
                        "y": arm_pose.y,
                        "z": arm_pose.z,
                        "o_x": arm_pose.o_x,
                        "o_y": arm_pose.o_y,
                        "o_z": arm_pose.o_z,
                        "theta": arm_pose.theta
                    }
                case "move_arm_to_position":
                    # Use poses if available (motion planning mode), otherwise use joint positions
                    if self.motion is not None and self.poses:
                        positions = self.poses
                        position_type = "pose"
                    else:
                        positions = self.joint_positions
                        position_type = "joint position"

                    index = int(value)

                    if index >= len(positions):
                        resp["move_arm_to_position"] = f"{position_type} {index} invalid since there are only {len(positions)} {position_type}s"
                        break

                    position = positions[index]

                    await self._move_arm_to_position(position, index, len(positions))

                    # Sleep to allow time for the camera and arm to settle
                    await asyncio.sleep(self.sleep_seconds)

                    tracked_poses: Optional[Dict[str, PoseInFrame]] = await self.pose_tracker.get_poses(body_names=self.body_names)
                    if tracked_poses is None or len(tracked_poses) == 0:
                        resp["move_arm_to_position"] = "No tracked bodies found in image"
                        break

                    resp["move_arm_to_position"] = len(tracked_poses)
                case "auto_calibrate":
                    raise NotImplementedError("This is not yet implemented")
                case _:
                    resp[key] = "unsupported key"

        if len(resp) == 0:
            return None, "no valid do_command submitted"
        
        return resp

    async def _resolve_tracking_directory(self, calibration_id: str) -> str:
        tracking_dir: Optional[str] = None

        if self.web_app is not None:
            try:
                response = await self.web_app.do_command({"command": "get_base_dir"})
                if isinstance(response, Mapping):
                    tracking_dir = response.get("base_dir")
                elif isinstance(response, (str, os.PathLike)):
                    tracking_dir = os.fspath(response)
            except Exception as err:
                self.logger.warning(f"Failed to retrieve base directory from web app: {err}")
        else:
            print("No web app service configured")
        if tracking_dir is None:
            raise Exception("could not get tracking directory")

        tracking_dir = os.path.join(tracking_dir, calibration_id)
        os.makedirs(tracking_dir, exist_ok=True)
        self.logger.info(f"Capturing calibration data to {tracking_dir}")
        return tracking_dir