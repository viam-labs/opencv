"""Shared base for planar-target pose trackers (chessboard, ChArUco, ...).

A planar calibration target (chessboard, ChArUco board) yields the same
downstream pipeline regardless of how its corners are detected:

    capture image -> detect corners -> solvePnP -> per-corner PoseInFrame

Only the *detection* step differs between target types. This module factors
out everything else so each concrete tracker only has to implement
``_detect_observation`` plus a little config glue.

Concrete trackers inherit ``(BaseTargetTracker, PoseTracker, EasyResource)``.
``BaseTargetTracker`` is a plain mixin (no ``MODEL``) so the Viam
``EasyResource`` machinery never tries to register it on its own.
"""

import abc
from dataclasses import dataclass
from typing import (Any, Dict, List, Mapping, Optional, Sequence, Tuple)

import cv2
import numpy as np

from viam.components.camera import Camera
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import CameraMimeType
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, Pose, PoseInFrame, ResourceName
from viam.resource.base import ResourceBase
from viam.utils import struct_to_dict, ValueTypes

try:
    from utils.utils import call_go_mat2ov
except ModuleNotFoundError:
    # when running as local module with run.sh
    from ..utils.utils import call_go_mat2ov


# Config attributes shared by every planar-target tracker.
CAM_ATTR = "camera_name"
CAMERA_INTRINSICS_ATTR = "camera_intrinsics"

# Camera intrinsics required keys (when supplied via config rather than the camera).
K_KEYS = ["fx", "fy", "cx", "cy"]
DIST_KEYS = ["k1", "k2", "k3", "p1", "p2"]

# Viam distortion models list radial coefficients first, tangential last;
# OpenCV interleaves them as (k1, k2, p1, p2, k3[, k4, k5, k6]). Passing Viam
# parameters to OpenCV verbatim puts a radial coefficient (k3, often ~0.1-0.3)
# in the p1 slot, where a real tangential value is ~1e-3 — a silent, massive
# distortion error that biases every PnP depth by tens of mm. Map from model
# name to the index order that converts Viam's layout to OpenCV's.
_DIST_MODEL_TO_OPENCV_ORDER = {
    # (k1, k2, k3, p1, p2) -> (k1, k2, p1, p2, k3)
    "brown_conrady": [0, 1, 3, 4, 2],
    # (k1, k2, k3, k4, k5, k6, p1, p2) -> (k1, k2, p1, p2, k3, k4, k5, k6)
    "brown_conrady_k6": [0, 1, 6, 7, 2, 3, 4, 5],
}

# Distortion vector lengths cv2.projectPoints / cv2.solvePnP accept.
_OPENCV_DIST_LENGTHS = (4, 5, 8, 12, 14)


def distortion_to_opencv(model: str, params: Sequence[float]) -> np.ndarray:
    """Convert camera-reported distortion parameters to OpenCV's ordering.

    Known Viam models are reordered per ``_DIST_MODEL_TO_OPENCV_ORDER``. An
    empty parameter list means an undistorted/rectified image (zeros). An
    unknown model is passed through only when its length is one OpenCV
    accepts; anything else raises rather than silently corrupting PnP.

    This is only the name-based *prior*: camera modules do not reliably follow
    their model name's documented ordering, so whenever a target is in view the
    trackers verify the ordering empirically (see
    :func:`select_distortion_by_reprojection`) and override this if the pixels
    decisively prefer another interpretation.
    """
    dist = np.asarray(list(params), dtype=np.float32).reshape(-1)
    if dist.size == 0:
        return np.zeros(5, dtype=np.float32)

    order = _DIST_MODEL_TO_OPENCV_ORDER.get((model or "").strip().lower())
    if order is not None:
        if dist.size != len(order):
            raise Exception(
                f"distortion model '{model}' should have {len(order)} "
                f"parameters but the camera reported {dist.size}: {dist.tolist()}"
            )
        return dist[order]

    if dist.size not in _OPENCV_DIST_LENGTHS:
        raise Exception(
            f"unrecognized distortion model '{model}' with {dist.size} "
            f"parameters; cannot map to OpenCV's (k1, k2, p1, p2, k3, ...) "
            f"ordering. Supply 'camera_intrinsics' in the tracker config instead."
        )
    return dist


def distortion_candidates(model: str, params: Sequence[float]) -> List[Tuple[str, np.ndarray]]:
    """Every plausible OpenCV-order interpretation of camera-reported
    distortion parameters, name-based prior first, duplicates removed.

    The Viam camera API carries distortion as an unstructured
    ``(model: str, parameters: [float])`` pair, and in practice third-party
    camera modules emit the same nominal model in different orders (radials
    first per the rdk structs, or already interleaved OpenCV-style). The order
    cannot be trusted from the data alone — it has to be verified against
    pixels. This enumerates the interpretations worth testing.
    """
    raw = np.asarray(list(params), dtype=np.float32).reshape(-1)
    if raw.size == 0:
        return [("rectified (no distortion)", np.zeros(5, dtype=np.float32))]

    labeled_orders: List[Tuple[str, List[int]]] = []
    if raw.size == 5:
        labeled_orders = [
            ("opencv (k1,k2,p1,p2,k3)", [0, 1, 2, 3, 4]),
            ("radial-first (k1,k2,k3,p1,p2)", [0, 1, 3, 4, 2]),
        ]
    elif raw.size == 8:
        labeled_orders = [
            ("opencv (k1,k2,p1,p2,k3..k6)", [0, 1, 2, 3, 4, 5, 6, 7]),
            ("radial-first (k1..k6,p1,p2)", [0, 1, 6, 7, 2, 3, 4, 5]),
            ("radial3-first (k1,k2,k3,p1,p2,k4,k5,k6)", [0, 1, 3, 4, 2, 5, 6, 7]),
        ]

    candidates: List[Tuple[str, np.ndarray]] = []
    prior = distortion_to_opencv(model, params)
    candidates.append((f"prior for model '{model}'", prior))
    for label, order in labeled_orders:
        dist = raw[order]
        if not any(np.array_equal(dist, d) for _, d in candidates):
            candidates.append((label, dist))
    return candidates


def select_distortion_by_reprojection(
    K: np.ndarray,
    candidates: Sequence[Tuple[str, np.ndarray]],
    object_points: np.ndarray,
    image_points: np.ndarray,
    margin: float = 1.2,
) -> Tuple[str, np.ndarray, Dict[str, float]]:
    """Pick the distortion interpretation the pixels actually support.

    Solves a single-view PnP with each candidate and scores it by corner
    reprojection rmse. A wrong ordering puts a radial coefficient in a
    tangential slot and typically misfits by an order of magnitude (observed:
    0.11px correct vs 1.78px wrong on the same frame), so the signal is
    strong exactly when the interpretations differ materially. The first
    candidate (the name-based prior) is kept unless another interpretation
    beats it by more than ``margin`` — when all candidates fit equally well
    the ordering doesn't matter numerically and the prior is as good as any.

    Returns ``(label, dist, scores)`` where scores maps each candidate label
    to its rmse in pixels.
    """
    objp = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    imgp = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    K64 = np.asarray(K, dtype=np.float64)

    scores: Dict[str, float] = {}
    rmses: List[float] = []
    for label, dist in candidates:
        try:
            ok, rvec, tvec = cv2.solvePnP(objp, imgp, K64, np.asarray(dist, dtype=np.float64))
            if not ok:
                raise Exception("solvePnP failed")
            proj, _ = cv2.projectPoints(objp, rvec, tvec, K64, np.asarray(dist, dtype=np.float64))
            rmse = float(np.sqrt(np.mean((proj.reshape(-1, 2) - imgp) ** 2)))
        except Exception:
            rmse = float("inf")
        scores[label] = rmse
        rmses.append(rmse)

    best_i = int(np.argmin(rmses))
    if rmses[0] <= margin * rmses[best_i]:
        best_i = 0  # prior is competitive — keep it
    label, dist = candidates[best_i]
    return label, dist, scores


@dataclass
class TargetObservation:
    """A single detection of a planar target, with its solved pose.

    ``corners_2d`` and ``object_points`` are parallel arrays (same N, same
    order) and ``ids`` gives each corner a *stable identity* across frames.
    For a chessboard the ids are simply ``0..N-1`` (the board is always fully
    visible and ordered). For ChArUco the ids are the board's interior-corner
    indices, so a partially visible board still labels its corners
    consistently from frame to frame.
    """

    corners_2d: np.ndarray            # (N, 1, 2) pixel coordinates
    object_points: np.ndarray         # (N, 3) board-frame coordinates
    ids: List[int]                    # length N, stable per-corner identity
    K: np.ndarray                     # (3, 3) intrinsics
    dist: np.ndarray                  # (5,) distortion
    rvec: np.ndarray                  # (3, 1) board->camera rotation (Rodrigues)
    tvec: np.ndarray                  # (3, 1) board->camera translation
    R: np.ndarray                     # (3, 3) board->camera rotation matrix
    # All PnP candidate poses, sorted by reprojection error (best first).
    # Planar PnP has a two-fold ambiguity; on near-frontal views the two
    # branches have nearly equal reprojection error and the best-error pick is
    # a coin flip. Downstream consumers with more context (e.g. hand-eye
    # calibration, which knows the arm chain) use these to disambiguate.
    rvec_candidates: Optional[List[np.ndarray]] = None   # each (3, 1)
    tvec_candidates: Optional[List[np.ndarray]] = None   # each (3, 1)
    reproj_err_candidates: Optional[List[float]] = None  # px, parallel to the above


def validate_intrinsics_attr(attrs: Mapping[str, Any]) -> None:
    """Validate an optional ``camera_intrinsics`` config block.

    No-op if the block is absent (intrinsics are then fetched from the camera).
    Raises with a specific message naming the first missing key.
    """
    intrinsics_config = attrs.get(CAMERA_INTRINSICS_ATTR)
    if intrinsics_config is None:
        return
    if intrinsics_config.get("K") is None:
        raise Exception("Missing required K for camera intrinsics.")
    K: dict = intrinsics_config.get("K")
    for K_key in K_KEYS:
        if K.get(K_key) is None:
            raise Exception(f"Missing required key {K_key} for K camera intrinsics.")
    if intrinsics_config.get("dist") is None:
        raise Exception("Missing required dist for camera intrinsics")
    dist: dict = intrinsics_config.get("dist")
    for dist_key in DIST_KEYS:
        if dist.get(dist_key) is None:
            raise Exception(f"Missing required key {dist_key} for dist camera intrinsics.")


class BaseTargetTracker(abc.ABC):
    """Mixin holding the detection-agnostic pose-tracker pipeline.

    Subclasses must set the class attribute ``OBSERVATION_KEY`` (the
    ``do_command`` key under which a raw observation is returned) and implement
    the three abstract hooks below.
    """

    # do_command key a downstream service uses to pull a raw observation.
    OBSERVATION_KEY: str = "get_target_observation"

    # ---- subclass hooks -------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def _validate_target_attrs(cls, attrs: Mapping[str, Any]) -> None:
        """Raise if any target-specific required attribute is missing."""

    @abc.abstractmethod
    def _reconfigure_target(self, attrs: Mapping[str, Any]) -> None:
        """Read target-specific attributes off ``attrs`` onto ``self``."""

    @abc.abstractmethod
    async def _detect_observation(self) -> TargetObservation:
        """Capture an image, detect the target, and solve its pose."""

    def _observation_metadata(self) -> Dict[str, Any]:
        """Target-specific fields merged into the ``do_command`` response."""
        return {}

    # ---- shared lifecycle ----------------------------------------------

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        cam = attrs.get(CAM_ATTR)
        if cam is None:
            raise Exception(f"Missing required {CAM_ATTR} attribute.")
        cls._validate_target_attrs(attrs)
        validate_intrinsics_attr(attrs)
        return [str(cam)], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        attrs = struct_to_dict(config.attributes)

        camera: str = attrs.get(CAM_ATTR)
        self.camera: Camera = dependencies.get(Camera.get_resource_name(camera))
        if self.camera is None:
            raise Exception(f"Could not find camera resource {camera}")

        # Optional user-provided intrinsics; falls back to the camera otherwise.
        self.camera_intrinsics = attrs.get(CAMERA_INTRINSICS_ATTR)

        # Camera-reported distortion ordering, resolved empirically on the
        # first detection. Reset on reconfigure: the camera may have changed.
        self._raw_distortion: Optional[Tuple[str, List[float]]] = None
        self._dist_resolution: Optional[Tuple[str, np.ndarray]] = None

        self._reconfigure_target(attrs)

        return super().reconfigure(config, dependencies)

    # ---- shared helpers -------------------------------------------------

    def _scale_K_to_image(self, K, intr_w, intr_h, img_w, img_h):
        """Rescale a camera matrix whose intrinsics were computed at a
        different resolution than the image actually returned by the camera.

        This guards against a common failure with cameras that report
        intrinsics for a subsampled/binned mode while streaming full
        resolution (e.g. a Zivid reporting half-res intrinsics for a full
        2448x2048 image). Using a mismatched K silently produces wrong PnP
        poses -- low reprojection error on a planar target but a badly biased
        pose -- which wrecks downstream hand-eye calibration.

        Only a uniform rescale (same aspect ratio) can be corrected this way;
        a cropped/letterboxed mismatch is raised loudly instead.
        """
        sx = img_w / intr_w
        sy = img_h / intr_h
        if abs(sx - sy) > 0.02 * max(sx, sy):
            raise Exception(
                f"camera intrinsics resolution {intr_w}x{intr_h} does not match "
                f"image {img_w}x{img_h} and the scale is non-uniform "
                f"(sx={sx:.3f}, sy={sy:.3f}); cannot safely rescale K. Configure "
                f"the camera so get_properties and get_images use the same "
                f"resolution."
            )
        K = K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        self.logger.warning(
            f"camera intrinsics are for {intr_w}x{intr_h} but the image is "
            f"{img_w}x{img_h}; auto-scaling K by ({sx:.3f}, {sy:.3f}) as a "
            f"stopgap. Configure the camera to report intrinsics matching the "
            f"streamed image resolution to silence this."
        )
        return K

    async def get_camera_intrinsics(self, image_shape: Optional[tuple] = None) -> tuple:
        """Get camera intrinsic parameters as ``(K, dist)`` numpy arrays.

        If ``image_shape`` (the captured image's ``(H, W, ...)``) is provided
        and the camera reports an intrinsics resolution that differs from it,
        ``K`` is rescaled to the image (see :meth:`_scale_K_to_image`).
        """
        if self.camera_intrinsics is None:
            props = await self.camera.get_properties()
            intrinsics = props.intrinsic_parameters
            dist_params = props.distortion_parameters

            K = np.array([
                [intrinsics.focal_x_px, 0, intrinsics.center_x_px],
                [0, intrinsics.focal_y_px, intrinsics.center_y_px],
                [0, 0, 1]
            ], dtype=np.float32)
            self._raw_distortion = (
                getattr(dist_params, "model", "") or "",
                list(dist_params.parameters),
            )
            if self._dist_resolution is not None:
                # Ordering already verified against pixels on a prior frame.
                dist = self._dist_resolution[1]
            else:
                dist = distortion_to_opencv(*self._raw_distortion)

            if (image_shape is not None
                    and intrinsics.width_px and intrinsics.height_px):
                img_h, img_w = int(image_shape[0]), int(image_shape[1])
                if (int(intrinsics.width_px), int(intrinsics.height_px)) != (img_w, img_h):
                    K = self._scale_K_to_image(
                        K, int(intrinsics.width_px), int(intrinsics.height_px),
                        img_w, img_h)
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

    def _resolve_distortion(
        self, K: np.ndarray, dist: np.ndarray,
        object_points: np.ndarray, image_points: np.ndarray,
    ) -> np.ndarray:
        """Verify the camera-reported distortion ordering against the pixels.

        Camera modules emit ``(model, parameters)`` in whatever order their
        vendor SDK uses, so the name-based mapping is only a guess. With a
        calibration target in view the guess is checkable: solve a single-view
        PnP under every plausible interpretation and keep the one the corners
        support. The verdict is cached until reconfigure. Config-supplied
        intrinsics name each coefficient explicitly and are used as-is.
        """
        if self.camera_intrinsics is not None or self._raw_distortion is None:
            return dist
        if self._dist_resolution is not None:
            return self._dist_resolution[1]

        candidates = distortion_candidates(*self._raw_distortion)
        if len(candidates) == 1:
            self._dist_resolution = candidates[0]
            return candidates[0][1]

        if np.asarray(object_points).reshape(-1, 3).shape[0] < 6:
            # Too few corners to trust a verdict; try again on a richer view.
            return dist

        label, resolved, scores = select_distortion_by_reprojection(
            K, candidates, object_points, image_points
        )
        self._dist_resolution = (label, resolved)
        pretty = ", ".join(f"{lbl}: {rmse:.3f}px" for lbl, rmse in scores.items())
        if np.array_equal(resolved, dist):
            self.logger.info(
                f"distortion ordering verified against pixels ({pretty}); "
                f"keeping '{label}'"
            )
        else:
            self.logger.warning(
                f"camera-reported distortion ordering did not fit the pixels "
                f"({pretty}); overriding to '{label}'. The camera module "
                f"likely emits parameters in a different order than its "
                f"distortion model name implies."
            )
        return resolved

    async def _capture_image(self) -> np.ndarray:
        """Grab the latest color image from the camera as an RGB numpy array."""
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
        return np.array(pil_image)

    # ---- PoseTracker API ------------------------------------------------

    async def get_poses(
        self,
        body_names: List[str],
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, PoseInFrame]:
        obs = await self._detect_observation()

        # Transpose needed due to frame convention mismatch:
        # OpenCV solvePnP returns object -> camera transform, but Viam expects camera -> object transform.
        # All corners share the same orientation (they're on the same rigid planar board)
        ox, oy, oz, theta = call_go_mat2ov(obs.R.T)
        self.logger.debug(f"Translated rotation matrix to orientation vector with values ox={ox}, oy={oy}, oz={oz}, theta={theta}")

        corner_poses = {}
        for obj_point, corner_id in zip(obs.object_points, obs.ids):
            point_3d = obs.R @ obj_point.reshape(3, 1) + obs.tvec

            corner_name = f"corner_{corner_id}"
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
        resp = {}
        for key, _value in command.items():
            if key == self.OBSERVATION_KEY:
                obs = await self._detect_observation()
                observation = {
                    "corners_2d": obs.corners_2d.reshape(-1, 2).tolist(),
                    "corners_3d": obs.object_points.reshape(-1, 3).tolist(),
                    "ids": list(obs.ids),
                    "K": obs.K.tolist(),
                    "dist": obs.dist.tolist(),
                    "rvec": obs.rvec.flatten().tolist(),
                    "tvec": obs.tvec.flatten().tolist(),
                }
                if obs.rvec_candidates is not None and obs.tvec_candidates is not None:
                    observation["rvec_candidates"] = [
                        np.asarray(r).flatten().tolist() for r in obs.rvec_candidates
                    ]
                    observation["tvec_candidates"] = [
                        np.asarray(t).flatten().tolist() for t in obs.tvec_candidates
                    ]
                    if obs.reproj_err_candidates is not None:
                        observation["reproj_err_candidates"] = [
                            float(e) for e in obs.reproj_err_candidates
                        ]
                observation.update(self._observation_metadata())
                resp[key] = observation
            else:
                resp[key] = "unsupported key"
        return resp

    async def get_geometries(
        self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None
    ) -> Sequence[Geometry]:
        self.logger.error("`get_geometries` is not implemented")
        raise NotImplementedError()
