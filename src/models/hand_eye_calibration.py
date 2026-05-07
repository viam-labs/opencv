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

from utils.utils import call_go_ov2mat, call_go_mat2ov

try:
    from diagnostics.pose_diversity import compute_pose_diversity
    from solvers.reprojection_solver import refine_handeye
except ModuleNotFoundError:
    # when running as local module with run.sh
    from ..diagnostics.pose_diversity import compute_pose_diversity
    from ..solvers.reprojection_solver import refine_handeye

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
CALIB_ATTR = "calibration_type"
JOINT_POSITIONS_ATTR = "joint_positions"
POSES_ATTR = "poses"
METHOD_ATTR = "method"
MOTION_ATTR = "motion"
POSE_TRACKER_ATTR = "pose_tracker"
SLEEP_ATTR = "sleep_seconds"
SOLVER_ATTR = "solver"
USE_MOTION_SERVICE_FOR_POSES_ATTR = "use_motion_service_for_poses"

# Solver options
SOLVER_OPENCV = "opencv"
SOLVER_HYBRID = "hybrid"
SOLVER_REPROJECTION = "reprojection"
SOLVERS = [SOLVER_OPENCV, SOLVER_HYBRID, SOLVER_REPROJECTION]

# Default config attribute values
DEFAULT_SLEEP_SECONDS = 2.0
DEFAULT_METHOD = "CALIB_HAND_EYE_TSAI"
DEFAULT_SOLVER = SOLVER_OPENCV
DEFAULT_USE_MOTION_SERVICE_FOR_POSES = False


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

        solver = attrs.get(SOLVER_ATTR, DEFAULT_SOLVER)
        if solver not in SOLVERS:
            raise Exception(f"'{solver}' is not a valid {SOLVER_ATTR}; must be one of {SOLVERS}.")

        body_name = attrs.get(BODY_NAME_ATTR)
        if body_name is not None:
            if not isinstance(body_name, str):
                raise Exception(f"'{BODY_NAME_ATTR}' must be a string, got {type(body_name)}")

        motion = attrs.get(MOTION_ATTR)
        optional_deps = []
        if motion is not None:
            optional_deps.append(str(motion))

        # Validate that motion service is configured if use_motion_service_for_poses is true
        use_motion_service_for_poses = attrs.get(USE_MOTION_SERVICE_FOR_POSES_ATTR, DEFAULT_USE_MOTION_SERVICE_FOR_POSES)
        if use_motion_service_for_poses and motion is None:
            raise Exception(f"'{USE_MOTION_SERVICE_FOR_POSES_ATTR}' is set to true but '{MOTION_ATTR}' is not configured. Either set '{USE_MOTION_SERVICE_FOR_POSES_ATTR}' to false or provide a '{MOTION_ATTR}' service name.")

        return [str(arm), str(pose_tracker)], optional_deps

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
        self.arm_name = arm
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
        self.solver = attrs.get(SOLVER_ATTR, DEFAULT_SOLVER)
        self.sleep_seconds = attrs.get(SLEEP_ATTR, DEFAULT_SLEEP_SECONDS)
        self.body_names = [attrs.get(BODY_NAME_ATTR)] if attrs.get(BODY_NAME_ATTR) is not None else []
        self.use_motion_service_for_poses = attrs.get(USE_MOTION_SERVICE_FOR_POSES_ATTR, DEFAULT_USE_MOTION_SERVICE_FOR_POSES)

        return super().reconfigure(config, dependencies)
    
    async def get_calibration_values(self):
        if self.use_motion_service_for_poses and self.motion is not None:
            arm_pose_in_frame = await self.motion.get_pose(
                component_name=self.arm_name,
                destination_frame=self.arm_name + "_origin"
            )
            arm_pose = arm_pose_in_frame.pose
            self.logger.debug(f"Found end of arm pose from motion service: {arm_pose}")
        else:
            arm_pose = await self.arm.get_end_position()
            self.logger.debug(f"Found end of arm pose from arm.get_end_position: {arm_pose}")

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

    def _compute_per_pose_residuals(
        self,
        R_gripper2base_list,
        t_gripper2base_list,
        R_target2cam_list,
        t_target2cam_list,
        R_cam2gripper,
        t_cam2gripper,
    ):
        """Evaluate per-pose calibration residuals to identify outlier measurements.

        For a correct calibration, the target's pose expressed in the base frame
        (T_target2base = T_gripper2base · T_cam2gripper · T_target2cam) should be
        identical across every pose, since the target is physically fixed.
        Deviations from the mean reveal which poses contributed bad measurements.

        Returns a list of per-pose residual dicts plus summary stats.
        """
        n = len(R_gripper2base_list)
        R_target2base = []
        t_target2base = []

        R_c2g = np.asarray(R_cam2gripper)
        t_c2g = np.asarray(t_cam2gripper).reshape(3)

        for i in range(n):
            R_g2b = np.asarray(R_gripper2base_list[i])
            t_g2b = np.asarray(t_gripper2base_list[i]).reshape(3)
            R_t2c = np.asarray(R_target2cam_list[i])
            t_t2c = np.asarray(t_target2cam_list[i]).reshape(3)

            R_t2b = R_g2b @ R_c2g @ R_t2c
            t_t2b = R_g2b @ R_c2g @ t_t2c + R_g2b @ t_c2g + t_g2b

            R_target2base.append(R_t2b)
            t_target2base.append(t_t2b)

        t_array = np.array(t_target2base)
        t_mean = np.mean(t_array, axis=0)

        # Mean rotation: Frobenius average projected back onto SO(3) via SVD
        R_stack = np.stack(R_target2base, axis=0)
        R_avg = np.mean(R_stack, axis=0)
        U, _, Vt = np.linalg.svd(R_avg)
        R_mean = U @ Vt
        if np.linalg.det(R_mean) < 0:
            Vt[-1] *= -1
            R_mean = U @ Vt

        residuals = []
        for i in range(n):
            t_resid = float(np.linalg.norm(t_target2base[i] - t_mean))
            R_diff = R_target2base[i] @ R_mean.T
            cos_angle = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
            rot_resid_deg = float(np.degrees(np.arccos(cos_angle)))
            residuals.append({
                "pose_index": i + 1,
                "translation_residual_mm": t_resid,
                "rotation_residual_deg": rot_resid_deg,
            })

        translation_resids = np.array([r["translation_residual_mm"] for r in residuals])
        rotation_resids = np.array([r["rotation_residual_deg"] for r in residuals])

        summary = {
            "translation_mean_mm": float(np.mean(translation_resids)),
            "translation_max_mm": float(np.max(translation_resids)),
            "rotation_mean_deg": float(np.mean(rotation_resids)),
            "rotation_max_deg": float(np.max(rotation_resids)),
        }

        return residuals, summary

    async def _collect_calibration_data_with_corners(self):
        """Collect end-effector poses and chessboard corner observations.

        Used by the reprojection-based solver path. Returns data in the
        OpenCV-standard convention:
            T_be: gripper-in-base (4x4)
            T_cw: target-in-camera (4x4) — board ≡ world for eye-in-hand

        Note: ``call_go_ov2mat`` returns body-from-parent (R_eb for an arm
        pose), so we transpose to get parent-from-body (R_be).
        """
        positions = self.poses if self.poses else self.joint_positions
        total_positions = len(positions)

        T_be_list = []
        T_cw_list = []
        corners_2d_list = []
        corners_3d = None
        K = None
        dist = None

        for i, position in enumerate(positions):
            try:
                await self._move_arm_to_position(position, i, total_positions)
                await asyncio.sleep(self.sleep_seconds)

                if self.use_motion_service_for_poses and self.motion is not None:
                    arm_pose_in_frame = await self.motion.get_pose(
                        component_name=self.arm_name,
                        destination_frame=self.arm_name + "_origin",
                    )
                    arm_pose = arm_pose_in_frame.pose
                else:
                    arm_pose = await self.arm.get_end_position()

                R_eb = call_go_ov2mat(arm_pose.o_x, arm_pose.o_y, arm_pose.o_z, arm_pose.theta)
                if R_eb is None:
                    raise Exception("could not convert arm orientation to rotation matrix")
                T_be = np.eye(4)
                T_be[:3, :3] = R_eb.T  # R_be: gripper-in-base
                T_be[:3, 3] = [arm_pose.x, arm_pose.y, arm_pose.z]

                obs_resp = await self.pose_tracker.do_command(
                    {"get_chessboard_observation": True}
                )
                obs = obs_resp.get("get_chessboard_observation")
                if obs is None or not isinstance(obs, dict):
                    raise Exception(
                        "pose_tracker.do_command did not return a chessboard observation; "
                        "ensure the pose tracker is the viam:opencv:chessboard model"
                    )

                corners_2d = np.asarray(obs["corners_2d"], dtype=np.float64).reshape(-1, 2)
                if corners_3d is None:
                    corners_3d = np.asarray(obs["corners_3d"], dtype=np.float64).reshape(-1, 3)
                    K = np.asarray(obs["K"], dtype=np.float64)
                    dist = np.asarray(obs["dist"], dtype=np.float64).reshape(-1)

                rvec = np.asarray(obs["rvec"], dtype=np.float64).reshape(3, 1)
                tvec = np.asarray(obs["tvec"], dtype=np.float64).reshape(3)
                R_ct, _ = cv2.Rodrigues(rvec)
                T_cw = np.eye(4)
                T_cw[:3, :3] = R_ct
                T_cw[:3, 3] = tvec

                T_be_list.append(T_be)
                T_cw_list.append(T_cw)
                corners_2d_list.append(corners_2d)

                self.logger.info(
                    f"successfully collected corner observation for position {i+1}/{total_positions}"
                )
            except Exception as e:
                error_msg = (
                    f"could not collect data for position {i+1}/{total_positions}: {e}"
                )
                self.logger.error(error_msg)
                raise Exception(error_msg)

        return T_be_list, T_cw_list, corners_2d_list, corners_3d, K, dist

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
                error_msg = f"Could not find calibration values for position {i+1}/{total_positions}: {e}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
        
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
                case "run_calibration":
                    refinement_info = None
                    if self.solver == SOLVER_OPENCV:
                        R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list = await self._collect_calibration_data()

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
                    else:
                        # Reprojection-based path: collect corner observations and use
                        # OpenCV-standard conventions throughout.
                        T_be_list, T_cw_list, corners_2d_list, corners_3d, K, dist = (
                            await self._collect_calibration_data_with_corners()
                        )
                        if len(T_be_list) < 3:
                            raise Exception(
                                f"not enough valid measurements collected. Got {len(T_be_list)}, need at least 3."
                            )
                        self.logger.info(
                            f"collected {len(T_be_list)} corner observations, "
                            f"bootstrapping with cv2.calibrateHandEye({self.method})"
                        )

                        # Bootstrap with cv2.calibrateHandEye in standard convention.
                        # R_be = T_be[:3,:3] (gripper-in-base), R_ct = T_cw[:3,:3] (target-in-cam).
                        R_g2b = [T[:3, :3] for T in T_be_list]
                        t_g2b = [T[:3, 3].reshape(3, 1) for T in T_be_list]
                        R_t2c = [T[:3, :3] for T in T_cw_list]
                        t_t2c = [T[:3, 3].reshape(3, 1) for T in T_cw_list]
                        R_c2g_init, t_c2g_init = cv2.calibrateHandEye(
                            R_g2b, t_g2b, R_t2c, t_t2c, method=getattr(cv2, self.method)
                        )
                        X_init = np.eye(4)
                        X_init[:3, :3] = R_c2g_init
                        X_init[:3, 3] = t_c2g_init.flatten()

                        self.logger.info("refining with reprojection-error solver...")
                        refinement = refine_handeye(
                            T_be_list,
                            corners_2d_list,
                            corners_3d,
                            K,
                            dist,
                            X_init=X_init,
                            T_cw_list=T_cw_list,
                        )
                        X_refined = refinement["X_refined"]
                        R_cam2gripper = X_refined[:3, :3]
                        t_cam2gripper = X_refined[:3, 3].reshape(3, 1)
                        refinement_info = {
                            "rmse_pixels": refinement["rmse_pixels"],
                            "per_pose_rmse_pixels": refinement["per_pose_rmse_pixels"],
                            "iterations": refinement["n_iterations"],
                            "success": refinement["success"],
                            "message": refinement["message"],
                        }
                        self.logger.info(
                            f"reprojection refinement: rmse={refinement['rmse_pixels']:.3f}px "
                            f"after {refinement['n_iterations']} fn evals "
                            f"(success={refinement['success']})"
                        )
                        median_rmse = float(np.median(refinement["per_pose_rmse_pixels"]))
                        for idx, r in enumerate(refinement["per_pose_rmse_pixels"]):
                            if median_rmse > 0 and r > 3.0 * median_rmse:
                                self.logger.warning(
                                    f"pose {idx+1}: per-pose rmse={r:.3f}px > 3× median ({median_rmse:.3f}px) — possible outlier"
                                )

                        # For the legacy per-pose residual summary, reconstruct the
                        # rotation/translation lists in the convention that helper expects.
                        R_gripper2base_list = R_g2b
                        t_gripper2base_list = t_g2b
                        R_target2cam_list = R_t2c
                        t_target2cam_list = t_t2c

                    # Per-pose residuals to identify outlier poses
                    per_pose_residuals, residual_summary = self._compute_per_pose_residuals(
                        R_gripper2base_list,
                        t_gripper2base_list,
                        R_target2cam_list,
                        t_target2cam_list,
                        R_cam2gripper,
                        t_cam2gripper,
                    )

                    self.logger.info(
                        f"Residual summary: translation mean={residual_summary['translation_mean_mm']:.3f}mm "
                        f"max={residual_summary['translation_max_mm']:.3f}mm; "
                        f"rotation mean={residual_summary['rotation_mean_deg']:.3f}deg "
                        f"max={residual_summary['rotation_max_deg']:.3f}deg"
                    )
                    sorted_resids = sorted(per_pose_residuals, key=lambda r: r["translation_residual_mm"], reverse=True)
                    for r in sorted_resids:
                        self.logger.info(
                            f"  pose {r['pose_index']:>3}: t_resid={r['translation_residual_mm']:.3f}mm "
                            f"r_resid={r['rotation_residual_deg']:.3f}deg"
                        )

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
                    response = {
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
                        },
                        "residuals": {
                            "summary": residual_summary,
                            "per_pose": per_pose_residuals,
                        },
                        "solver": self.solver,
                    }
                    if refinement_info is not None:
                        response["refinement"] = refinement_info
                    resp["run_calibration"] = response
                case "move_arm": 
                    raise NotImplementedError("This is not yet implemented")
                case "check_bodies":
                    tracked_poses: Dict[str, PoseInFrame] = await self.pose_tracker.get_poses(body_names=self.body_names)
                    if tracked_poses is None or len(tracked_poses) == 0:
                        resp["check_bodies"] = "No tracked bodies found in image"
                        break

                    resp["check_bodies"] = f"Number of tracked bodies seen: {len(tracked_poses)}"
                case "get_current_arm_pose":
                    if self.use_motion_service_for_poses:
                        if self.motion is None:
                            resp["get_current_arm_pose"] = f"{USE_MOTION_SERVICE_FOR_POSES_ATTR} is true, but motion service was None."
                            break
                        arm_pose_in_frame = await self.motion.get_pose(
                            component_name=self.arm_name,
                            destination_frame=self.arm_name + "_origin"
                        )
                        arm_pose = arm_pose_in_frame.pose
                    else:
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
                case "compute_pose_diversity":
                    poses_input = None
                    if isinstance(value, dict):
                        poses_input = value.get("poses")

                    if poses_input is not None:
                        poses_for_diag = []
                        for p in poses_input:
                            poses_for_diag.append(Pose(
                                x=p.get("x", 0),
                                y=p.get("y", 0),
                                z=p.get("z", 0),
                                o_x=p.get("o_x", 0),
                                o_y=p.get("o_y", 0),
                                o_z=p.get("o_z", 1),
                                theta=p.get("theta", 0),
                            ))
                    else:
                        poses_for_diag = self.poses

                    if not poses_for_diag:
                        resp["compute_pose_diversity"] = {
                            "error": (
                                f"no poses available. Configure '{POSES_ATTR}' on the service "
                                f"or pass {{'poses': [...]}} in the command. "
                                f"Joint-position-only configurations are not supported by "
                                f"the diagnostic — convert to cartesian poses first."
                            ),
                        }
                        break

                    transforms = []
                    convert_failed = False
                    for pose in poses_for_diag:
                        R = call_go_ov2mat(pose.o_x, pose.o_y, pose.o_z, pose.theta)
                        if R is None:
                            convert_failed = True
                            break
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = [pose.x, pose.y, pose.z]
                        transforms.append(T)

                    if convert_failed:
                        resp["compute_pose_diversity"] = {
                            "error": "could not convert orientation vector to rotation matrix",
                        }
                        break

                    diversity = compute_pose_diversity(transforms)
                    self.logger.info(f"pose-set diagnostics: {diversity['feedback']}")
                    for w in diversity.get("warnings", []):
                        self.logger.warning(f"pose-set diagnostics: {w}")
                    resp["compute_pose_diversity"] = diversity
                case _:
                    resp[key] = "unsupported key"

        if len(resp) == 0:
            return None, "no valid do_command submitted"
        
        return resp
