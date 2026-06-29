import cv2

from typing import (Any, ClassVar, Dict, Mapping)

from viam.components.pose_tracker import *
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

try:
    from utils.charuco_utils import (
        DEFAULT_DICTIONARY,
        DEFAULT_LEGACY_PATTERN,
        DICTIONARY_ATTR,
        LEGACY_PATTERN_ATTR,
        MARKER_SIZE_ATTR,
        SQUARES_X_ATTR,
        SQUARES_Y_ATTR,
        SQUARE_SIZE_ATTR,
        build_charuco_board_from_attrs,
        detect_charuco_corners,
        match_object_points,
        validate_charuco_attrs,
    )
    from models.base_pose_tracker import BaseTargetTracker, TargetObservation
except ModuleNotFoundError:
    # when running as local module with run.sh
    from ..utils.charuco_utils import (
        DEFAULT_DICTIONARY,
        DEFAULT_LEGACY_PATTERN,
        DICTIONARY_ATTR,
        LEGACY_PATTERN_ATTR,
        MARKER_SIZE_ATTR,
        SQUARES_X_ATTR,
        SQUARES_Y_ATTR,
        SQUARE_SIZE_ATTR,
        build_charuco_board_from_attrs,
        detect_charuco_corners,
        match_object_points,
        validate_charuco_attrs,
    )
    from .base_pose_tracker import BaseTargetTracker, TargetObservation


class Charuco(BaseTargetTracker, PoseTracker, EasyResource):
    """Pose tracker for ChArUco boards (chessboard + ArUco markers).

    Behaves like the ``chessboard`` model -- it reports a ``corner_<id>`` pose
    per detected interior corner and exposes a raw observation via
    ``do_command`` -- but tolerates partial views and labels corners by their
    stable board id rather than sequential detection order.

    To enable debug-level logging, either run viam-server with the --debug
    option, or configure your resource/machine to display debug logs.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "opencv"), "charuco")

    # do_command key downstream services request for a raw observation.
    OBSERVATION_KEY: ClassVar[str] = "get_charuco_observation"

    @classmethod
    def _validate_target_attrs(cls, attrs: Mapping[str, Any]) -> None:
        validate_charuco_attrs(attrs)

    def _reconfigure_target(self, attrs: Mapping[str, Any]) -> None:
        self.squares_x = int(attrs.get(SQUARES_X_ATTR))
        self.squares_y = int(attrs.get(SQUARES_Y_ATTR))
        self.square_size = float(attrs.get(SQUARE_SIZE_ATTR))
        self.marker_size = float(attrs.get(MARKER_SIZE_ATTR))
        self.dictionary = str(attrs.get(DICTIONARY_ATTR, DEFAULT_DICTIONARY))
        self.legacy_pattern = bool(attrs.get(LEGACY_PATTERN_ATTR, DEFAULT_LEGACY_PATTERN))
        self.board = build_charuco_board_from_attrs(attrs)

    def _observation_metadata(self) -> Dict[str, Any]:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_size_mm": self.square_size,
            "marker_size_mm": self.marker_size,
            "dictionary": self.dictionary,
            "legacy_pattern": self.legacy_pattern,
        }

    async def _detect_observation(self) -> TargetObservation:
        """Capture an image and run ChArUco corner detection + PnP."""
        image = await self._capture_image()
        K, dist = await self.get_camera_intrinsics(image.shape)

        detection = detect_charuco_corners(image, self.board)
        if detection is None:
            raise Exception("Could not find ChArUco board in image")
        charuco_corners, charuco_ids = detection
        self.logger.debug(f"Found {len(charuco_ids)} ChArUco corners")

        objp, imgp = match_object_points(self.board, charuco_corners, charuco_ids)

        success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
        if not success:
            raise Exception("Could not solve PnP for ChArUco board")
        self.logger.debug(f"Solved PnP rvec={rvec.flatten()} tvec={tvec.flatten()}")

        R, _ = cv2.Rodrigues(rvec)
        return TargetObservation(
            corners_2d=charuco_corners,
            object_points=objp,
            ids=charuco_ids.flatten().tolist(),
            K=K,
            dist=dist,
            rvec=rvec,
            tvec=tvec,
            R=R,
        )
