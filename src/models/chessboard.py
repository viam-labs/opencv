import cv2
import numpy as np

from typing import (Any, ClassVar, Dict, List, Mapping, Optional,
                    Sequence, Tuple)

from typing_extensions import Self
from viam.components.camera import Camera
from viam.components.pose_tracker import *
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import CameraMimeType
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, Pose, PoseInFrame, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict, ValueTypes

try:
    from utils.utils import call_go_mat2ov
    from utils.chessboard_utils import (
        detect_chessboard_corners,
        generate_object_points
    )
except ModuleNotFoundError:
    # when running as local module with run.sh
    from ..utils.utils import call_go_mat2ov
    from ..utils.chessboard_utils import (
        detect_chessboard_corners,
        generate_object_points
    )


# required attributes
cam_attr = "camera_name"
camera_intrinsics = "camera_intrinsics"
pattern_attr = "pattern_size"
square_attr = "square_size_mm"

# Camera intrinsics required keys
K_KEYS = ["fx", "fy", "cx", "cy"]
DIST_KEYS = ["k1", "k2", "k3", "p1", "p2"]



class Chessboard(PoseTracker, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "opencv"), "chessboard")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this PoseTracker component.
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
        if attrs.get(pattern_attr) is None:
            raise Exception(f"Missing required {pattern_attr} attribute.")
        if attrs.get(square_attr) is None:
            raise Exception(f"Missing required {square_attr} attribute.")
        if attrs.get(camera_intrinsics) is not None:
            # Check to make sure the right keys are available
            intrinsics_config = attrs.get(camera_intrinsics)
            if intrinsics_config.get("K") is None:
                raise Exception(f"Missing required K for camera intrinsics.")
            K: dict = intrinsics_config.get("K")
            for K_key in K_KEYS:
                if K.get(K_key) is None:
                    raise Exception(f"Missing required key {K_key} for K camera intrinsics.")
            if intrinsics_config.get("dist") is None:
                raise Exception(f"Missing required dist for camera intrinsics")
            dist: dict = intrinsics_config.get("dist")
            for dist_key in DIST_KEYS:
                if dist.get(dist_key) is None:
                    raise Exception(f"Missing required key {dist_key} for dist camera intrinsics.")

        return [str(cam)], []

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
        if self.camera is None:
            raise Exception(f"Could not find camera resource {camera}")
        
        pattern_list: list = attrs.get(pattern_attr)
        self.pattern_size = [int(x) for x in pattern_list]
        self.square_size = attrs.get(square_attr)

        # Get camera intrinsics provided by user if available
        self.camera_intrinsics = attrs.get(camera_intrinsics)

        return super().reconfigure(config, dependencies)
    
    async def get_camera_intrinsics(self) -> tuple:
        """Get camera intrinsic parameters"""
        if self.camera_intrinsics is None:
            camera_params = await self.camera.do_command({"get_camera_params": None})
            intrinsics = camera_params["Color"]["intrinsics"]
            dist_params = camera_params["Color"]["distortion"]
        else:
            intrinsics = self.camera_intrinsics["K"]
            dist_params = self.camera_intrinsics["dist"]

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
        
        self.logger.debug(f"Camera intrinsics: K shape={K.shape}, dist shape={dist.shape}")
        self.logger.debug(f"Distortion coefficients: {dist}")
        
        return K, dist

    async def get_poses(
        self,
        body_names: List[str],
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, PoseInFrame]:
        
        cam_images = await self.camera.get_images()
        pil_image = None
        for cam_image in cam_images[0]:
            # Accept any standard image format that viam_to_pil_image can handle
            if cam_image.mime_type in [CameraMimeType.JPEG, CameraMimeType.PNG, CameraMimeType.VIAM_RGBA]:
                pil_image = viam_to_pil_image(cam_image)
                self.logger.debug(f"Found {cam_image.mime_type} image from camera")
                break
        if pil_image is None:
            raise Exception("Could not get latest image from camera")        
        image = np.array(pil_image)
        
        K, dist = await self.get_camera_intrinsics()

        # Detect and refine chessboard corners
        corners = detect_chessboard_corners(image, tuple(self.pattern_size))
        if corners is None:
            raise Exception("Could not find chessboard pattern in image")
        self.logger.debug(f"Found chessboard with corners: {corners}")

        # Generate 3D object points for the chessboard
        objp = generate_object_points(tuple(self.pattern_size), self.square_size)
        
        # Solve PnP to get pose
        success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        if not success:
            raise Exception("Could not solve PnP for chessboard")
        self.logger.debug(f"Solved PnP")
        self.logger.debug(f"Rotation vector: {rvec}")
        self.logger.debug(f"Translation vector: {tvec}")
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Transpose needed due to frame convention mismatch:
        # OpenCV solvePnP returns object -> camera transform, but Viam expects camera -> object transform.
        # All corners share the same orientation (they're on the same rigid planar board)
        ox, oy, oz, theta = call_go_mat2ov(R.T)
        self.logger.debug(f"Translated rotation matrix to orientation vector with values ox={ox}, oy={oy}, oz={oz}, theta={theta}")
        
        corner_poses = {}
        
        for i, obj_point in enumerate(objp):
            point_3d = R @ obj_point.reshape(3, 1) + tvec
            
            corner_name = f"corner_{i}"
            corner_poses[corner_name] = PoseInFrame(
                reference_frame=self.camera.name,
                pose=Pose(
                    x=point_3d[0][0],
                    y=point_3d[1][0],
                    z=point_3d[2][0],
                    o_x=ox,
                    o_y=oy,
                    o_z=oz,
                    theta=theta
                )
            )
        
        self.logger.debug(f"generated {len(corner_poses)} corner poses")
        
        if body_names:
            filtered_poses = {}
            for name in body_names:
                if name not in corner_poses:
                    raise Exception(f"requested body name '{name}' not found in detected corners")
                filtered_poses[name] = corner_poses[name]
            return filtered_poses
        
        return corner_poses

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`do_command` is not implemented")
        raise NotImplementedError()

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Sequence[Geometry]:
        self.logger.error("`get_geometries` is not implemented")
        raise NotImplementedError()
