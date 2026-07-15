import cv2
import numpy as np

from typing import (Any, ClassVar, Dict, Mapping)

from viam.components.pose_tracker import *
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

try:
    from utils.chessboard_utils import (
        detect_chessboard_corners,
        generate_object_points
    )
    from models.base_pose_tracker import BaseTargetTracker, TargetObservation
except ModuleNotFoundError:
    # when running as local module with run.sh
    from ..utils.chessboard_utils import (
        detect_chessboard_corners,
        generate_object_points
    )
    from .base_pose_tracker import BaseTargetTracker, TargetObservation


# target-specific required attributes
PATTERN_ATTR = "pattern_size"
SQUARE_ATTR = "square_size_mm"


class Chessboard(BaseTargetTracker, PoseTracker, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "opencv"), "chessboard")

    # do_command key downstream services (e.g. hand_eye_calibration) request.
    OBSERVATION_KEY: ClassVar[str] = "get_chessboard_observation"

    @classmethod
    def _validate_target_attrs(cls, attrs: Mapping[str, Any]) -> None:
        if attrs.get(PATTERN_ATTR) is None:
            raise Exception(f"Missing required {PATTERN_ATTR} attribute.")
        if attrs.get(SQUARE_ATTR) is None:
            raise Exception(f"Missing required {SQUARE_ATTR} attribute.")

    def _reconfigure_target(self, attrs: Mapping[str, Any]) -> None:
        pattern_list: list = attrs.get(PATTERN_ATTR)
        self.pattern_size = [int(x) for x in pattern_list]
        self.square_size = attrs.get(SQUARE_ATTR)

    def _observation_metadata(self) -> Dict[str, Any]:
        return {
            "pattern_size": list(self.pattern_size),
            "square_size_mm": float(self.square_size),
        }

    async def _detect_observation(self) -> TargetObservation:
        """Capture an image and run corner detection + PnP."""
        image = await self._capture_image()
        K, dist = await self.get_camera_intrinsics(image.shape)

        corners = detect_chessboard_corners(image, tuple(self.pattern_size))
        if corners is None:
            raise Exception("Could not find chessboard pattern in image")
        self.logger.debug(f"Found chessboard with corners: {corners}")

        objp = generate_object_points(tuple(self.pattern_size), self.square_size)

        # Camera modules don't reliably order distortion parameters the way
        # their model name implies; verify the ordering against this view's
        # corners before trusting it (cached after the first verdict).
        dist = self._resolve_distortion(K, dist, objp, corners)

        # Chessboard corners are coplanar (z=0 in the board frame), so the
        # default SOLVEPNP_ITERATIVE solver is subject to the planar two-fold
        # pose ambiguity and can return the mirror-flipped pose — especially on
        # near-frontal views, where both branches fit the pixels almost equally
        # well. SOLVEPNP_IPPE is purpose-built for coplanar points and returns
        # both candidates sorted by reprojection error; we report the best one
        # here and expose all candidates so downstream consumers (hand-eye
        # calibration) can disambiguate using the arm chain.
        n_solutions, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            objp, corners, K, dist, flags=cv2.SOLVEPNP_IPPE
        )
        if n_solutions < 1:
            raise Exception("Could not solve PnP for chessboard")
        rvec, tvec = rvecs[0], tvecs[0]
        self.logger.debug(
            f"Solved PnP ({n_solutions} candidate(s), best reproj_err="
            f"{float(reproj_errs[0]):.3f}px) rvec={rvec.flatten()} tvec={tvec.flatten()}"
        )

        R, _ = cv2.Rodrigues(rvec)
        # A chessboard is always fully detected and ordered, so corner ids are 0..N-1.
        return TargetObservation(
            corners_2d=corners,
            object_points=objp,
            ids=list(range(len(objp))),
            K=K,
            dist=dist,
            rvec=rvec,
            tvec=tvec,
            R=R,
            rvec_candidates=list(rvecs),
            tvec_candidates=list(tvecs),
            reproj_err_candidates=[float(e) for e in np.asarray(reproj_errs).flatten()],
        )
