import base64
import cv2
import numpy as np

from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict, ValueTypes


# Required attributes
pattern_attr = "pattern_size"
square_attr = "square_size_mm"


class CameraCalibration(Generic, EasyResource):
    """Generic service that provides camera calibration functionality via do_command.

    This service uses chessboard patterns to calibrate camera intrinsic parameters.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "opencv"), "camera-calibration")

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

        if attrs.get(pattern_attr) is None:
            raise Exception(f"Missing required {pattern_attr} attribute.")
        if attrs.get(square_attr) is None:
            raise Exception(f"Missing required {square_attr} attribute.")

        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        """Dynamically update the service when it receives a new config.

        Args:
            config: The new configuration
            dependencies: Any dependencies (both required and optional)
        """
        attrs = struct_to_dict(config.attributes)

        pattern_list: list = attrs.get(pattern_attr)
        self.pattern_size = [int(x) for x in pattern_list]
        self.square_size = attrs.get(square_attr)

        return super().reconfigure(config, dependencies)

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
