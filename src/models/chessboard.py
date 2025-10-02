import asyncio
import base64
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

    async def _detect_chessboard_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
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
    
    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Decode a base64 encoded image string to a numpy array.
        
        Args:
            base64_str: Base64 encoded image string
            
        Returns:
            Image as numpy array
        """
        # Remove data URI prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',', 1)[1]
        
        # Decode base64 string
        img_bytes = base64.b64decode(base64_str)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert from BGR to RGB (OpenCV loads as BGR)
        if image is not None and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    async def _calibrate_camera_from_images(self, base64_images: List[str]) -> Mapping[str, Any]:
        """Calibrate camera using provided chessboard images.
        
        Args:
            base64_images: List of base64 encoded image strings
            
        Returns:
            Dictionary containing calibration results
        """
        object_points = []  # 3D points in real world space
        image_points = []   # 2D points in image plane
        image_size = None
        
        num_images = len(base64_images)
        self.logger.info(f"Starting camera calibration with {num_images} images")
        
        successful_detections = 0
        
        for idx, base64_img in enumerate(base64_images):
            try:
                # Decode base64 image
                image = self._decode_base64_image(base64_img)
                
                if image is None:
                    self.logger.warning(f"Image {idx + 1}: Failed to decode base64 image")
                    continue
                
                # Store image size (width, height)
                if image_size is None:
                    if len(image.shape) == 3:
                        image_size = (image.shape[1], image.shape[0])  # (width, height)
                    else:
                        image_size = (image.shape[1], image.shape[0])
                
                # Detect chessboard corners
                corners = self._detect_chessboard_corners(image)
                
                if corners is None:
                    self.logger.warning(f"Image {idx + 1}: Could not find chessboard pattern")
                    continue
                
                # Add to calibration dataset
                object_points.append(self._generate_object_points())
                image_points.append(corners)
                successful_detections += 1
                
                self.logger.info(f"Successfully processed image {successful_detections}/{num_images} (image {idx + 1})")
                
            except Exception as e:
                self.logger.warning(f"Image {idx + 1}: Error processing image: {e}")
                continue
        
        if successful_detections < 3:
            raise Exception(f"Only found chessboard pattern in {successful_detections}/{num_images} images. Need at least 3 valid images for calibration.")
        
        # Perform camera calibration
        self.logger.info("Running camera calibration...")
        
        # Initialize camera matrix and distortion coefficients
        camera_matrix = np.zeros((3, 3), dtype=np.float32)
        dist_coeffs = np.zeros(5, dtype=np.float32)
        
        # Run calibration
        ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            camera_matrix,
            dist_coeffs,
            flags=cv2.CALIB_RATIONAL_MODEL
        )
        
        self.logger.info(f"Calibration complete with RMS error: {ret}")
        
        # Format results
        result = {
            "success": True,
            "rms_error": float(ret),
            "num_images": successful_detections,
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "camera_matrix": {
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2])
            },
            "distortion_coefficients": {
                "k1": float(dist_coeffs[0]),
                "k2": float(dist_coeffs[1]),
                "p1": float(dist_coeffs[2]),
                "p2": float(dist_coeffs[3]),
                "k3": float(dist_coeffs[4])
            }
        }
        
        return result

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Mapping[str, ValueTypes]:
        """Execute custom commands.
        
        Supported commands:
        - {"calibrate_camera": {"images": [<base64_str>, ...]}}: Calibrate camera intrinsics
        """
        if "calibrate_camera" in command:
            params = command["calibrate_camera"]
            if params is None:
                params = {}
            
            # Extract images from command
            images = params.get("images")
            if images is None or not isinstance(images, list):
                return {
                    "success": False,
                    "error": "Missing required 'images' parameter. Must be a list of base64 encoded image strings."
                }
            
            if len(images) == 0:
                return {
                    "success": False,
                    "error": "At least one image is required for calibration."
                }
            
            try:
                result = await self._calibrate_camera_from_images(images)
                return result
            except Exception as e:
                self.logger.error(f"camera calibration failed: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        self.logger.error(f"unknown command: {list(command.keys())}")
        raise NotImplementedError(f"command not supported: {list(command.keys())}")

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Sequence[Geometry]:
        self.logger.error("`get_geometries` is not implemented")
        raise NotImplementedError()

