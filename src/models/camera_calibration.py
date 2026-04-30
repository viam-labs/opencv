import cv2
import numpy as np

from typing import Any, ClassVar, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict, ValueTypes

try:
    from utils.chessboard_utils import (
        detect_chessboard_corners,
        generate_object_points,
        decode_base64_image
    )
except ModuleNotFoundError:
    # when running as local module with run.sh
    from ..utils.chessboard_utils import (
        detect_chessboard_corners,
        generate_object_points,
        decode_base64_image
    )


# Required attributes
PATTERN_ATTR = "pattern_size"
SQUARE_ATTR = "square_size_mm"


class CameraCalibration(Generic, EasyResource):
    """Generic service that provides camera calibration functionality via do_command.

    This service uses chessboard patterns to calibrate camera intrinsic parameters.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "opencv"), "camera_calibration")

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

        if attrs.get(PATTERN_ATTR) is None:
            raise Exception(f"Missing required {PATTERN_ATTR} attribute.")
        if attrs.get(SQUARE_ATTR) is None:
            raise Exception(f"Missing required {SQUARE_ATTR} attribute.")

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

        pattern_list: list = attrs.get(PATTERN_ATTR)
        self.pattern_size = [int(x) for x in pattern_list]
        self.square_size = attrs.get(SQUARE_ATTR)

        return super().reconfigure(config, dependencies)

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
                image = decode_base64_image(base64_img)

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
                corners = detect_chessboard_corners(image, tuple(self.pattern_size))

                if corners is None:
                    self.logger.warning(f"Image {idx + 1}: Could not find chessboard pattern")
                    continue

                # Add to calibration dataset
                object_points.append(generate_object_points(tuple(self.pattern_size), self.square_size))
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
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            camera_matrix,
            dist_coeffs,
            flags=cv2.CALIB_RATIONAL_MODEL
        )

        # Calculate re-projection error
        mean_error = 0
        for i in range(len(object_points)):
            imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error

        reprojection_error = mean_error / len(object_points)

        self.logger.info(f"Calibration complete with RMS error: {ret:.6f}")
        self.logger.info(f"Mean re-projection error: {reprojection_error:.6f}")

        # Format results
        result = {
            "success": True,
            "rms_error": float(ret),
            "reprojection_error": float(reprojection_error),
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
