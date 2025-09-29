import asyncio
import cv2
import numpy as np
from typing import ClassVar, Final, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm, JointPositions
from viam.components.camera import Camera
from viam.components.pose_tracker import PoseTracker
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import *
from viam.services.motion import Motion
from viam.utils import struct_to_dict, ValueTypes

from ..utils.utils import call_go_ov2mat, call_go_mat2ov

# required attributes
arm_attr = "arm_name"
cam_attr = "camera_name"
joint_positions_attr = "joint_positions"
motion_attr = "motion"
pose_tracker_attr = "pose_tracker"
sleep_attr = "sleep_seconds"


class HandEyeCalibration(Generic, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("bradgrigsby", "calibrate"), "hand-eye-calibration"
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

        cam = attrs.get(cam_attr)
        if cam is None:
            raise Exception(f"Missing required {cam_attr} attribute.")
        
        arm = attrs.get(arm_attr)
        if arm is None:
            raise Exception(f"Missing required {arm_attr} attribute.")
        
        if attrs.get(joint_positions_attr) is None:
            raise Exception(f"Missing required {joint_positions_attr} attribute.")
        
        pose_tracker = attrs.get(pose_tracker_attr)
        if pose_tracker is None:
            raise Exception(f"Missing required {pose_tracker_attr} attribute.")

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

        camera: str = attrs.get(cam_attr)
        self.camera: Camera = dependencies.get(Camera.get_resource_name(camera))
        
        arm: Arm = attrs.get(arm_attr)
        self.arm: Arm = dependencies.get(Arm.get_resource_name(arm))

        pose_tracker: PoseTracker = attrs.get(pose_tracker_attr)
        self.pose_tracker: PoseTracker = dependencies.get(PoseTracker.get_resource_name(pose_tracker))

        motion: Motion = attrs.get(motion_attr)
        self.motion: Motion = dependencies.get(Motion.get_resource_name(motion))

        self.joint_positions = attrs.get(joint_positions_attr)
        self.sleep_seconds = attrs.get(sleep_attr, 1.0)

        return super().reconfigure(config, dependencies)

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
                    for i, joints in enumerate(self.joint_positions):
                        R_g2b = None
                        t_g2b = None
                        R_t2c = None
                        t_t2c = None

                        self.logger.debug(f"Moving to pose {i+1}/{len(self.joint_positions)}")

                        jp = JointPositions(values=joints)
                        await self.arm.move_to_joint_positions(jp)
                        while await self.arm.is_moving():
                            await asyncio.sleep(0.05)
                        self.logger.debug(f"Moved arm to position: {jp}")

                        # Sleep for the configured amount of time to allow the arm and camera to settle
                        await asyncio.sleep(self.sleep_seconds)

                        arm_pose = await self.arm.get_end_position()
                        self.logger.debug(f"Found end of are pose: {arm_pose}")

                        R_g2b = call_go_ov2mat(
                            arm_pose.o_x, 
                            arm_pose.o_y, 
                            arm_pose.o_z, 
                            arm_pose.theta
                        )
                        if R_g2b is None:
                            self.logger.warning("Failed to convert arm orientation vector to rotation matrix")
                            continue
                        t_g2b = np.array([arm_pose.x], [arm_pose.y], [arm_pose.z], dtype=np.float64)

                        # Get pose of the tag
                        tag_poses: dict = await self.pose_tracker.get_poses()
                        if tag_poses is None:
                            raise Exception("Could not find any tags in camera frame. Check to make sure there is a tag in view.")
                        if len(tag_poses.items) > 1:
                            raise Exception("More than 1 tag detected in camera frame. Please remove any other tags in view.")
                        
                        tag_pose: Pose = tag_poses.values()[0]
                        R_t2c = call_go_ov2mat(
                            tag_pose.o_x,
                            tag_pose.o_y,
                            tag_pose.o_z,
                            tag_pose.theta
                        )
                        if R_t2c is None:
                            self.logger.warning("Failed to convert tag orientation vector to rotation matrix.")
                        t_t2c = np.array([tag_pose.x], [tag_pose.y], [tag_pose.z], dtype=np.float64)

                        if R_g2b is None or t_g2b is None or R_t2c is None or t_t2c is None:
                            self.logger.warning("Missing data. Skipping this pose.")
                            continue
                        
                        R_gripper2base_list.append(R_g2b.T)
                        t_gripper2base_list.append(t_g2b)
                        R_target2cam_list.append(R_t2c.T)
                        t_target2cam_list.append(t_t2c)

                    pose = cv2.calibrateHandEye(
                        R_gripper2base=R_gripper2base_list,
                        t_gripper2base=t_gripper2base_list,
                        R_target2cam=R_target2cam_list,
                        t_target2cam=t_target2cam_list
                    )
                    if pose is None:
                        raise Exception("Could not solve calibration")
                    
                    viam_pose: Pose = call_go_mat2ov(pose)

                    return {viam_pose}
                case "move_arm": 
                    return {}
                case "check_tags":
                    return {}
                case "save_calibration_position":
                    return {}
                case "move_arm_to_position":
                    return {}
                case "delete_calibration_position":
                    return {}
                case "clear_calibration_positions":
                    return {}
                case "auto_calibrate":
                    return {}
                case "save_guess": 
                    return {}
                case _:
                    resp[key] = "unsupported key"

        if len(resp) == 0:
            return None, "no valid do_command submitted"
        
        return resp
