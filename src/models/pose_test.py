import asyncio
import numpy as np
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from viam.components.arm import Arm
from viam.components.pose_tracker import PoseTracker
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import Motion
from viam.utils import struct_to_dict, ValueTypes

try:
    from utils.utils import call_go_ov2mat, call_go_mat2ov
except ModuleNotFoundError:
    # when running as local module with run.sh
    from ..utils.utils import call_go_ov2mat, call_go_mat2ov


# Required attributes
ARM_NAME_ATTR = "arm_name"
CAMERA_NAME_ATTR = "camera_name"
POSE_TRACKER_ATTR = "pose_tracker"
MOTION_SERVICE_ATTR = "motion_service"
POSES_ATTR = "poses"
BODY_NAME_ATTR = "body_name"


class PoseTest(Generic, EasyResource):
    """Generic service that tests hand-eye calibration accuracy by measuring
    pose estimation errors across multiple arm positions.

    This service moves an arm through predefined poses and compares the predicted
    camera motion (based on hand-eye calibration) against the actual arm motion
    to quantify calibration accuracy.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "opencv"), "pose-test")

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """Validate the configuration object.

        Args:
            config: The configuration for this resource

        Returns:
            Tuple[Sequence[str], Sequence[str]]: A tuple where the
                first element is a list of required dependencies and the
                second element is a list of optional dependencies
        """
        attrs = struct_to_dict(config.attributes)

        arm_name = attrs.get(ARM_NAME_ATTR)
        if arm_name is None:
            raise Exception(f"Missing required {ARM_NAME_ATTR} attribute.")

        camera_name = attrs.get(CAMERA_NAME_ATTR)
        if camera_name is None:
            raise Exception(f"Missing required {CAMERA_NAME_ATTR} attribute.")

        pose_tracker = attrs.get(POSE_TRACKER_ATTR)
        if pose_tracker is None:
            raise Exception(f"Missing required {POSE_TRACKER_ATTR} attribute.")

        motion_service = attrs.get(MOTION_SERVICE_ATTR)
        if motion_service is None:
            raise Exception(f"Missing required {MOTION_SERVICE_ATTR} attribute.")

        poses = attrs.get(POSES_ATTR)
        if poses is None:
            raise Exception(f"Missing required {POSES_ATTR} attribute.")

        if not isinstance(poses, list):
            raise Exception(f"{POSES_ATTR} must be a list.")

        # Validate each pose has required fields
        for i, pose_dict in enumerate(poses):
            required_fields = ["x", "y", "z", "o_x", "o_y", "o_z", "theta"]
            for field in required_fields:
                if field not in pose_dict:
                    raise Exception(f"Pose {i} in {POSES_ATTR} missing required field '{field}'.")

        body_name = attrs.get(BODY_NAME_ATTR)
        if body_name is None:
            raise Exception(f"Missing required {BODY_NAME_ATTR} attribute.")

        return [str(arm_name), str(camera_name), str(pose_tracker), str(motion_service)], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """Dynamically update the service when it receives a new config.

        Args:
            config: The new configuration
            dependencies: Any dependencies (both required and optional)
        """
        self.logger.debug(f"Reconfiguring pose test resource with deps: {dependencies}")
        attrs = struct_to_dict(config.attributes)

        arm_name: str = attrs.get(ARM_NAME_ATTR)
        self.arm: Arm = dependencies.get(Arm.get_resource_name(arm_name))
        self.arm_name = arm_name

        camera_name: str = attrs.get(CAMERA_NAME_ATTR)
        from viam.components.camera import Camera
        self.camera: Camera = dependencies.get(Camera.get_resource_name(camera_name))
        self.camera_name = camera_name

        pose_tracker_name: str = attrs.get(POSE_TRACKER_ATTR)
        self.pose_tracker: PoseTracker = dependencies.get(PoseTracker.get_resource_name(pose_tracker_name))

        motion_service_name: str = attrs.get(MOTION_SERVICE_ATTR)
        self.motion: Motion = dependencies.get(Motion.get_resource_name(motion_service_name))
        if self.motion is None:
            raise Exception(f"Motion service not found: {motion_service_name}")

        # Get body name for pose tracker
        self.body_name: str = attrs.get(BODY_NAME_ATTR)


        # Parse poses from config
        poses_list = attrs.get(POSES_ATTR, [])
        self.poses: List[Pose] = []
        for pose_dict in poses_list:
            self.poses.append(Pose(
                x=float(pose_dict["x"]),
                y=float(pose_dict["y"]),
                z=float(pose_dict["z"]),
                o_x=float(pose_dict["o_x"]),
                o_y=float(pose_dict["o_y"]),
                o_z=float(pose_dict["o_z"]),
                theta=float(pose_dict["theta"])
            ))

        return super().reconfigure(config, dependencies)

    def _pose_to_matrix(self, pose: Pose) -> np.ndarray:
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

    def _matrix_to_pose(self, T: np.ndarray) -> Pose:
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

    async def _get_current_arm_pose(self) -> Pose:
        """Get the current pose of the arm end effector in the world frame.

        Returns:
            Pose: Current arm pose in world frame
        """
        pose_in_frame = await self.motion.get_pose(
            component_name=self.arm_name,
            destination_frame="world"
        )
        return pose_in_frame.pose

    async def _get_current_parent_pose(self) -> Pose:
        """Get the current pose of the arm end effector (parent) in world frame.

        Returns:
            Pose: Current pose in world frame
        """
        pose_in_frame = await self.motion.get_pose(
            component_name=self.arm_name,
            destination_frame="world"
        )
        return pose_in_frame.pose

    def _calculate_translational_error(self, T_estimated: np.ndarray, T_actual: np.ndarray) -> Dict[str, float]:
        """Calculate translational error between two transformation matrices.

        Args:
            T_estimated: Estimated/predicted 4x4 transformation matrix
            T_actual: Actual/ground truth 4x4 transformation matrix

        Returns:
            Dictionary with delta_x, delta_y, delta_z, and magnitude
        """
        # Extract translation vectors
        t_estimated = T_estimated[0:3, 3]
        t_actual = T_actual[0:3, 3]

        # Calculate absolute differences
        delta = np.abs(t_estimated - t_actual)
        delta_x = float(delta[0])
        delta_y = float(delta[1])
        delta_z = float(delta[2])

        # Calculate magnitude
        magnitude = float(np.linalg.norm(t_estimated - t_actual))

        return {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "magnitude": magnitude
        }

    def _calculate_rotational_error(self, T_estimated: np.ndarray, T_actual: np.ndarray) -> float:
        """Calculate rotational error between two transformation matrices.

        The error is computed as the angle of rotation needed to go from the estimated
        orientation to the actual orientation using the proper formula:
        R_error = R_estimated^T * R_actual

        Args:
            T_estimated: Estimated/predicted 4x4 transformation matrix
            T_actual: Actual/ground truth 4x4 transformation matrix

        Returns:
            Rotational error in degrees
        """
        # Extract rotation matrices
        R_estimated = T_estimated[0:3, 0:3]
        R_actual = T_actual[0:3, 0:3]

        # Compute relative rotation: R_error = R_estimated^T * R_actual
        R_error = R_estimated.T @ R_actual

        # Extract the angle from the rotation matrix using the trace
        # trace(R) = 1 + 2*cos(theta)
        # theta = arccos((trace(R) - 1) / 2)
        trace = np.trace(R_error)

        # Clamp trace to valid range to handle numerical errors
        # Valid range for trace is [-1, 3] since trace = 1 + 2*cos(theta) and cos(theta) in [-1, 1]
        trace = np.clip(trace, -1.0, 3.0)

        # Calculate angle
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clamp to [-1, 1] for arccos

        theta_rad = np.arccos(cos_theta)
        theta_deg = float(np.degrees(theta_rad))

        return theta_deg

    async def _run_pose_test(self) -> Dict[str, Any]:
        """Execute the pose test algorithm.

        Returns:
            Dictionary containing test results with per-pose errors and aggregate statistics
        """
        if len(self.poses) < 2:
            raise Exception("Need at least 2 poses to run test (anchor + 1 test pose)")

        self.logger.info(f"Starting pose test with {len(self.poses)} poses")

        # Move to pose_0 (anchor pose)
        self.logger.info("Moving to anchor pose (pose_0)")
        await self.motion.move(
            component_name=self.arm_name,
            destination=PoseInFrame(reference_frame="world", pose=self.poses[0])
        )
        # Wait for arm to settle
        await asyncio.sleep(1.0)

        #########################################################################################################
        # Get A_0 (end effector in world) first
        #########################################################################################################
        A_0_pose = await self._get_current_parent_pose()
        # Convert to matrix
        T_A_0 = self._pose_to_matrix(A_0_pose)

        #########################################################################################################
        # Compute hand-eye transform X robustly from world poses at anchor
        # X should be camera_in_end_effector. We derive: X = inv(A_0) * camera_world_0
        #########################################################################################################
        # Compute X once by querying camera pose in arm frame (gripper parent)
        pose_in_frame = await self.motion.get_pose(
            component_name=self.camera_name,
            destination_frame=self.arm_name
        )
        X_pose = pose_in_frame.pose
        T_X = self._pose_to_matrix(X_pose)

        # minimal; do not use frame system further beyond initial X derivation

        #########################################################################################################
        # Get target observation from pose tracker at anchor pose
        # The pose tracker observes a fixed target and returns its pose in some reference frame
        #########################################################################################################
        tracked_poses_0 = await self.pose_tracker.get_poses(body_names=[self.body_name])
        if not tracked_poses_0:
            raise Exception(f"Could not get any poses from pose tracker at anchor pose")
        if self.body_name not in tracked_poses_0:
            raise Exception(f"Could not find tracked body '{self.body_name}' in pose tracker observations at anchor pose")

        # Extract PoseInFrame and its reference frame at anchor
        target_pif_0 = tracked_poses_0[self.body_name]
        ref_frame_0 = getattr(target_pif_0, "reference_frame", None) or getattr(target_pif_0, "reference_frame_name", None)
        target_in_ref_0 = target_pif_0.pose
        # ref frame required to be camera for algorithm
        if ref_frame_0 != self.camera_name:
            raise Exception("pose tracker must report target in camera frame")
        T_target_in_ref_0 = self._pose_to_matrix(target_in_ref_0)

        # Compute target world anchor purely from arm and X: T_target^W0 = (A0 * X) * T_target^C0
        T_camera_0_in_world = T_A_0 @ T_X
        T_target_in_world = T_camera_0_in_world @ self._pose_to_matrix(target_pif_0.pose)

        #########################################################################################################
        # Test each subsequent pose
        #########################################################################################################
        pose_errors = []
        for i in range(1, len(self.poses)):
            # Move to pose_i
            await self.motion.move(
                component_name=self.arm_name,
                destination=PoseInFrame(reference_frame="world", pose=self.poses[i])
            )

            # Wait for arm to settle
            await asyncio.sleep(1.0)

            # Get A_i
            A_i_pose = await self._get_current_parent_pose()
            # Convert to matrix
            T_A_i = self._pose_to_matrix(A_i_pose)

            # Get target observation from pose tracker at pose i
            tracked_poses_i = await self.pose_tracker.get_poses(body_names=[self.body_name])
            if not tracked_poses_i or self.body_name not in tracked_poses_i:
                self.logger.warning(f"Could not find tracked body '{self.body_name}' at pose {i}, skipping")
                continue

            # Extract PoseInFrame and its reference frame at pose i
            target_pif_i = tracked_poses_i[self.body_name]
            ref_frame_i = getattr(target_pif_i, "reference_frame", None) or getattr(target_pif_i, "reference_frame_name", None)
            target_in_ref_i = target_pif_i.pose
            if ref_frame_i != self.camera_name:
                raise Exception("pose tracker must report target in camera frame")
            T_target_in_ref_i = self._pose_to_matrix(target_in_ref_i)

            # Delta-based predictions per literature (Equation 47)
            # Actual delta_A: motion from anchor to pose i in parent frame
            T_delta_A_actual = np.linalg.inv(T_A_0) @ T_A_i

            # Use tracker deltas: delta_B = inv(B0) * Bi in camera frame
            T_delta_B = np.linalg.inv(T_target_in_ref_0) @ T_target_in_ref_i
            T_delta_A_pred = T_X @ T_delta_B @ np.linalg.inv(T_X)
            translational_error = self._calculate_translational_error(T_delta_A_pred, T_delta_A_actual)
            rotational_error = self._calculate_rotational_error(T_delta_A_pred, T_delta_A_actual)

            pose_errors.append({
                "pose_index": i,
                "translational_error_mm": translational_error,
                "rotational_error_deg": rotational_error
            })

        if len(pose_errors) == 0:
            raise Exception("No valid pose comparisons completed")

        # Calculate aggregate statistics
        mean_trans_error = sum(e["translational_error_mm"]["magnitude"] for e in pose_errors) / len(pose_errors)
        max_trans_error = max(e["translational_error_mm"]["magnitude"] for e in pose_errors)
        mean_rot_error = sum(e["rotational_error_deg"] for e in pose_errors) / len(pose_errors)
        max_rot_error = max(e["rotational_error_deg"] for e in pose_errors)

        return {
            "success": True,
            "num_poses_tested": len(pose_errors),
            "pose_errors": pose_errors,
            "aggregate_errors": {
                "mean_translational_error_mm": mean_trans_error,
                "max_translational_error_mm": max_trans_error,
                "mean_rotational_error_deg": mean_rot_error,
                "max_rotational_error_deg": max_rot_error
            }
        }

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        """Execute custom commands.

        Supported commands:
        - {"get_current_pose": {}}: Get current arm pose in world frame
        - {"validate_setup": {}}: Validate configuration and dependencies
        - {"run_pose_test": {}}: Move the arm and execute the pose test algorithm 
        """
        if "get_current_pose" in command:
            try:
                pose = await self._get_current_arm_pose()
                return {
                    "pose": {
                        "x": pose.x,
                        "y": pose.y,
                        "z": pose.z,
                        "o_x": pose.o_x,
                        "o_y": pose.o_y,
                        "o_z": pose.o_z,
                        "theta": pose.theta
                    }
                }
            except Exception as e:
                self.logger.error(f"Failed to get current pose: {e}")
                return {
                    "error": str(e)
                }

        if "validate_setup" in command:
            warnings = []
            success = True

            # Check number of poses
            if len(self.poses) < 2:
                warnings.append(f"Only {len(self.poses)} pose(s) configured, need at least 2")
                success = False

            # Check arm is reachable
            arm_reachable = self.arm is not None
            if not arm_reachable:
                warnings.append("Arm component not accessible")
                success = False

            # Check camera/pose tracker
            pose_tracker_exists = self.pose_tracker is not None
            if not pose_tracker_exists:
                warnings.append("Pose tracker not accessible")
                success = False

            # Check motion service
            motion_exists = self.motion is not None
            if not motion_exists:
                warnings.append("Motion service not accessible")
                success = False

            # Try to get camera pose to verify frame system
            camera_frame_exists = False
            hand_eye_transform_exists = False
            try:
                camera_pose = await self._get_camera_pose()
                if camera_pose is not None:
                    camera_frame_exists = True
                    hand_eye_transform_exists = True
                else:
                    warnings.append("Cannot get camera pose from frame system")
            except Exception as e:
                warnings.append(f"Frame system error: {e}")

            return {
                "success": success,
                "arm_reachable": arm_reachable,
                "pose_tracker_exists": pose_tracker_exists,
                "motion_exists": motion_exists,
                "camera_frame_exists": camera_frame_exists,
                "hand_eye_transform_exists": hand_eye_transform_exists,
                "num_poses_configured": len(self.poses),
                "warnings": warnings
            }

        if "run_pose_test" in command:
            try:
                result = await self._run_pose_test()
                return result
            except Exception as e:
                self.logger.error(f"Pose test failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        self.logger.error(f"Unknown command: {list(command.keys())}")
        raise NotImplementedError(f"Command not supported: {list(command.keys())}")
