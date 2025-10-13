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

from utils.utils import call_go_mat2ov


# required attributes
cam_attr = "camera_name"
pattern_attr = "pattern_size"
square_attr = "square_size_mm"


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

        return super().reconfigure(config, dependencies)
    
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
        corners = self._detect_chessboard_corners(image)
        if corners is None:
            raise Exception("Could not find chessboard pattern in image")
        self.logger.debug(f"Found chessboard with corners: {corners}")
        
        # Generate 3D object points for the chessboard
        objp = self._generate_object_points()
        
        # Solve PnP to get pose
        success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
        if not success:
            print("Could not solve PnP for chessboard")
            return None, None
        self.logger.debug(f"Solved PnP")
        self.logger.debug(f"Rotation vector: {rvec}")
        self.logger.debug(f"Translation vector: {tvec}")
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Transpose needed due to frame convention mismatch:
        # OpenCV solvePnP returns object -> camera transform, but Viam expects camera -> object transform.
        ox, oy, oz, theta = call_go_mat2ov(R.T)
        self.logger.debug(f"Translated roation matrix to orientation vector with values ox={ox}, oy={oy}, oz={oz}, theta={theta}")
        
        # Convert tvec to column vector (3x1)
        t = tvec.reshape(3, 1)

        pose_in_frame = PoseInFrame(
            reference_frame=self.camera.name,
            pose=Pose(
                x=t[0][0],
                y=t[1][0],
                z=t[2][0],
                o_x=ox,
                o_y=oy,
                o_z=oz,
                theta=theta
            )
        )
        
        return {"pose": pose_in_frame}

    def _detect_chessboard_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and refine chessboard corners in an image.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Refined corner locations as (N, 1, 2) array, or None if not found
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(
            gray,
            tuple(self.pattern_size),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if not found:
            return None
        
        # Refine corner locations to sub-pixel precision
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return corners
    
    def _generate_object_points(self) -> np.ndarray:
        """Generate 3D object points for the chessboard pattern.
        
        Returns:
            Object points as (N, 3) array where N is number of corners
        """
        objp = np.zeros((self.pattern_size[1] * self.pattern_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
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
