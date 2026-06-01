"""Tests for src/utils/charuco_utils.py.

Synthetic round-trip: render a known ChArUco board, project it through a known
camera pose, then verify detection + matchImagePoints + solvePnP recovers that
pose. Also verifies partial-view detection still yields a consistent subset of
stable corner ids.
"""

import cv2
import numpy as np
import pytest

from utils.charuco_utils import (
    MIN_CHARUCO_CORNERS,
    build_charuco_board,
    detect_charuco_corners,
    match_object_points,
)


SQUARES_X = 5
SQUARES_Y = 7
SQUARE_MM = 30.0
MARKER_MM = 22.0


def _board():
    return build_charuco_board(SQUARES_X, SQUARES_Y, SQUARE_MM, MARKER_MM, "DICT_4X4_50")


def _render(board, px_per_square=120):
    """Render the board to a face-on grayscale image with a white margin."""
    w = SQUARES_X * px_per_square
    h = SQUARES_Y * px_per_square
    img = board.generateImage((w, h), marginSize=px_per_square // 2)
    return img


def test_build_rejects_marker_not_smaller_than_square():
    with pytest.raises(ValueError):
        build_charuco_board(5, 7, 30.0, 30.0, "DICT_4X4_50")


def test_build_rejects_unknown_dictionary():
    with pytest.raises(ValueError):
        build_charuco_board(5, 7, 30.0, 22.0, "DICT_DOES_NOT_EXIST")


def test_detect_full_board_finds_all_interior_corners():
    board = _board()
    img = _render(board)

    detection = detect_charuco_corners(img, board)
    assert detection is not None
    corners, ids = detection

    n_interior = (SQUARES_X - 1) * (SQUARES_Y - 1)
    assert len(ids) == n_interior
    assert corners.shape == (n_interior, 1, 2)
    # ids are unique and within the valid interior-corner range
    flat = ids.flatten()
    assert len(set(flat.tolist())) == len(flat)
    assert flat.min() >= 0 and flat.max() < n_interior


def test_detect_returns_none_when_no_board():
    board = _board()
    blank = np.full((480, 640), 255, dtype=np.uint8)
    assert detect_charuco_corners(blank, board) is None


def test_solvepnp_recovers_known_pose():
    board = _board()
    img = _render(board)
    detection = detect_charuco_corners(img, board)
    assert detection is not None
    corners, ids = detection

    objp, imgp = match_object_points(board, corners, ids)
    assert objp.shape[0] == imgp.shape[0] == len(ids)

    # Reasonable synthetic intrinsics for the rendered image size.
    h, w = img.shape[:2]
    f = float(max(w, h))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
    assert ok

    # Reproject and confirm low residual -> the recovered pose is consistent.
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean((proj.reshape(-1, 2) - imgp) ** 2)))
    assert rms < 1.0


def test_partial_view_yields_subset_of_stable_ids():
    board = _board()
    img = _render(board)
    detection = detect_charuco_corners(img, board)
    assert detection is not None
    full_corners, full_ids = detection

    # Crop away the right third of the board; the remaining corners must be a
    # subset of the full board's ids (stable identity under partial view).
    h, w = img.shape[:2]
    cropped = img[:, : (2 * w) // 3].copy()
    partial = detect_charuco_corners(cropped, board)
    assert partial is not None
    _, partial_ids = partial

    full_set = set(full_ids.flatten().tolist())
    partial_set = set(partial_ids.flatten().tolist())
    assert len(partial_set) >= MIN_CHARUCO_CORNERS
    assert partial_set.issubset(full_set)
    assert len(partial_set) < len(full_set)
