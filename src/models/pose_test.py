import asyncio
import numpy as np
from typing import ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from viam.components.arm import Arm
from viam.components.pose_tracker import PoseTracker
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName
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
WORLD_FRAME_ATTR = "world_frame"

# Default config attribute values
DEFAULT_WORLD_FRAME = "world"


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

        if len(poses) < 2:
            raise Exception(f"{POSES_ATTR} must contain at least 2 poses (anchor + 1 test pose).")

        # Validate each pose has required fields
        for i, pose_dict in enumerate(poses):
            required_fields = ["x", "y", "z", "o_x", "o_y", "o_z", "theta"]
            for field in required_fields:
                if field not in pose_dict:
                    raise Exception(f"Pose {i} in {POSES_ATTR} missing required field '{field}'.")

        return [str(arm_name), str(camera_name), str(pose_tracker), str(motion_service)], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """Dynamically update the service when it receives a new config.

        Args:
            config: The new configuration
            dependencies: Any dependencies (both required and optional)
        """
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

        self.world_frame: str = attrs.get(WORLD_FRAME_ATTR, DEFAULT_WORLD_FRAME)

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
            destination_frame=self.world_frame
        )
        return pose_in_frame.pose

    async def _get_camera_pose(self) -> Optional[Pose]:
        """Get the current pose of the camera in the world frame.

        This uses the frame system which should have the hand-eye calibration
        configured as a transform between the arm and camera.

        Returns:
            Pose: Camera pose in world frame, or None if not available
        """
        try:
            # The camera frame should be accessible through the frame system
            # We get the camera's pose in the world frame
            # Note: This assumes the camera is properly configured in the frame system
            # with the hand-eye calibration transform
            pose_in_frame = await self.motion.get_pose(
                component_name=self.camera_name,
                destination_frame=self.world_frame
            )
            return pose_in_frame.pose
        except Exception as e:
            self.logger.warning(f"Could not get camera pose: {e}")
            return None

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

    async def _run_pose_test_algorithm(self) -> Dict[str, any]:
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
            destination=self.poses[0]
        )

        # Wait for arm to settle
        await asyncio.sleep(1.0)

        # Get A_0 and B_0
        A_0_pose = await self._get_current_arm_pose()
        B_0_pose = await self._get_camera_pose()

        if B_0_pose is None:
            raise Exception("Could not get camera pose. Check frame system configuration and hand-eye calibration.")

        self.logger.info(f"Anchor pose A_0: {A_0_pose}")
        self.logger.info(f"Anchor camera pose B_0: {B_0_pose}")

        # Convert to matrices immediately
        T_A_0 = self._pose_to_matrix(A_0_pose)
        T_B_0 = self._pose_to_matrix(B_0_pose)

        # Get hand-eye transform X from frame system
        # X should be the transform from camera to arm (gripper)
        # X = A_0 * inv(B_0)
        T_X = T_A_0 @ np.linalg.inv(T_B_0)

        self.logger.info(f"Hand-eye transform X: {self._matrix_to_pose(T_X)}")

        # Test each subsequent pose
        pose_errors = []

        for i in range(1, len(self.poses)):
            self.logger.info(f"Testing pose {i}/{len(self.poses)-1}")

            # Move to pose_i
            await self.motion.move(
                component_name=self.arm_name,
                destination=self.poses[i]
            )

            # Wait for arm to settle
            await asyncio.sleep(1.0)

            # Get A_i and B_i
            A_i_pose = await self._get_current_arm_pose()
            B_i_pose = await self._get_camera_pose()

            if B_i_pose is None:
                self.logger.warning(f"Could not get camera pose for pose {i}, skipping")
                continue

            self.logger.debug(f"Pose {i} - A_i: {A_i_pose}")
            self.logger.debug(f"Pose {i} - B_i: {B_i_pose}")

            # Convert to matrices
            T_A_i = self._pose_to_matrix(A_i_pose)
            T_B_i = self._pose_to_matrix(B_i_pose)

            # Calculate delta_B = inv(B_i) * B_0
            T_delta_B = np.linalg.inv(T_B_i) @ T_B_0

            self.logger.debug(f"Pose {i} - delta_B: {self._matrix_to_pose(T_delta_B)}")

            # Calculate delta_A_estimated = X * delta_B * inv(X)
            T_delta_A_estimated = T_X @ T_delta_B @ np.linalg.inv(T_X)

            self.logger.debug(f"Pose {i} - delta_A_estimated: {self._matrix_to_pose(T_delta_A_estimated)}")

            # Calculate delta_A_actual = inv(A_i) * A_0
            T_delta_A_actual = np.linalg.inv(T_A_i) @ T_A_0

            self.logger.debug(f"Pose {i} - delta_A_actual: {self._matrix_to_pose(T_delta_A_actual)}")

            # Calculate errors using matrices
            translational_error = self._calculate_translational_error(T_delta_A_estimated, T_delta_A_actual)
            rotational_error = self._calculate_rotational_error(T_delta_A_estimated, T_delta_A_actual)

            self.logger.info(f"Pose {i} - Translational error: {translational_error['magnitude']:.2f} mm")
            self.logger.info(f"Pose {i} - Rotational error: {rotational_error:.2f} degrees")

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
        - {"run_pose_test": {}}: Execute the pose test algorithm
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
                result = await self._run_pose_test_algorithm()
                return result
            except Exception as e:
                self.logger.error(f"Pose test failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

        self.logger.error(f"Unknown command: {list(command.keys())}")
        raise NotImplementedError(f"Command not supported: {list(command.keys())}")
