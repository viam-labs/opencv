"""Shared utilities for ChArUco board detection and processing.

A ChArUco board is a chessboard whose white squares carry ArUco markers. The
markers give every interior chessboard corner a unique, identifiable index, so
the board can be detected even when only partially in view -- while the actual
calibration measurements are still the high-accuracy chessboard saddle corners.

This module mirrors ``chessboard_utils`` (build target -> detect corners) but
returns corner *ids* alongside the corners, since a ChArUco detection is
generally a subset of the board's corners rather than the whole grid.

Requires OpenCV >= 4.7 (the ``cv2.aruco.CharucoDetector`` API).
"""

from typing import Any, Mapping, Optional, Tuple

import cv2
import numpy as np

# Minimum interior corners needed to solve a stable PnP pose.
MIN_CHARUCO_CORNERS = 4

# Config attribute names shared by every consumer of a ChArUco board
# (the charuco pose tracker and the camera_calibration service).
SQUARES_X_ATTR = "squares_x"          # number of chessboard squares along X
SQUARES_Y_ATTR = "squares_y"          # number of chessboard squares along Y
SQUARE_SIZE_ATTR = "square_size_mm"   # chessboard square side length
MARKER_SIZE_ATTR = "marker_size_mm"   # ArUco marker side length (< square)
DICTIONARY_ATTR = "dictionary"        # predefined ArUco dictionary name (optional)
DEFAULT_DICTIONARY = "DICT_4X4_50"


def validate_charuco_attrs(attrs: Mapping[str, Any]) -> None:
    """Validate the ChArUco board attributes on a config attribute mapping.

    Raises with a specific message for the first problem found. Shared by the
    charuco pose tracker and the camera_calibration service so the config
    contract is defined in exactly one place.
    """
    for attr in (SQUARES_X_ATTR, SQUARES_Y_ATTR, SQUARE_SIZE_ATTR, MARKER_SIZE_ATTR):
        if attrs.get(attr) is None:
            raise Exception(f"Missing required {attr} attribute.")
    if float(attrs.get(MARKER_SIZE_ATTR)) >= float(attrs.get(SQUARE_SIZE_ATTR)):
        raise Exception(f"{MARKER_SIZE_ATTR} must be smaller than {SQUARE_SIZE_ATTR}.")
    dictionary = attrs.get(DICTIONARY_ATTR, DEFAULT_DICTIONARY)
    if getattr(cv2.aruco, str(dictionary), None) is None:
        raise Exception(f"Unknown ArUco {DICTIONARY_ATTR} '{dictionary}'.")


def build_charuco_board_from_attrs(attrs: Mapping[str, Any]) -> "cv2.aruco.CharucoBoard":
    """Build a ChArUco board from a (validated) config attribute mapping."""
    return build_charuco_board(
        int(attrs.get(SQUARES_X_ATTR)),
        int(attrs.get(SQUARES_Y_ATTR)),
        float(attrs.get(SQUARE_SIZE_ATTR)),
        float(attrs.get(MARKER_SIZE_ATTR)),
        str(attrs.get(DICTIONARY_ATTR, DEFAULT_DICTIONARY)),
    )


def build_charuco_board(
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_size: float,
    dictionary_name: str = "DICT_4X4_50",
) -> "cv2.aruco.CharucoBoard":
    """Construct a ChArUco board definition.

    Args:
        squares_x: Number of chessboard squares along X (columns).
        squares_y: Number of chessboard squares along Y (rows).
        square_size: Side length of a chessboard square (mm or other unit).
        marker_size: Side length of an ArUco marker (same unit, < square_size).
        dictionary_name: Predefined ArUco dictionary, e.g. ``"DICT_4X4_50"``.

    Returns:
        A ``cv2.aruco.CharucoBoard``. Object points are in the same unit as
        ``square_size``, so solved translations come out in that unit.
    """
    if marker_size >= square_size:
        raise ValueError(
            f"marker_size ({marker_size}) must be smaller than square_size ({square_size})"
        )
    dict_id = getattr(cv2.aruco, dictionary_name, None)
    if dict_id is None:
        raise ValueError(f"Unknown ArUco dictionary '{dictionary_name}'")

    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_size, marker_size, dictionary
    )


def detect_charuco_corners(
    image: np.ndarray,
    board: "cv2.aruco.CharucoBoard",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Detect and refine ChArUco corners in an image.

    Args:
        image: Input image (grayscale or color/RGB).
        board: The ChArUco board definition from :func:`build_charuco_board`.

    Returns:
        ``(charuco_corners, charuco_ids)`` where ``charuco_corners`` is an
        ``(N, 1, 2)`` array of refined pixel locations and ``charuco_ids`` is
        an ``(N, 1)`` array of the corresponding interior-corner indices, or
        ``None`` if fewer than ``MIN_CHARUCO_CORNERS`` corners are found.

    Note:
        ``CharucoDetector`` performs the ArUco marker detection, interpolates
        the chessboard corners between markers, and sub-pixel refines them.
        Because each corner is tied to a marker-derived id, the returned ids
        are stable across views even when the board is only partially visible.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
        return None

    return charuco_corners, charuco_ids


def match_object_points(
    board: "cv2.aruco.CharucoBoard",
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair detected ChArUco corners with their board-frame 3D points.

    Thin wrapper over ``board.matchImagePoints`` that returns arrays shaped for
    ``cv2.solvePnP``.

    Returns:
        ``(object_points, image_points)`` as ``(N, 3)`` and ``(N, 2)`` arrays.
    """
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    return (
        object_points.reshape(-1, 3),
        image_points.reshape(-1, 2),
    )
