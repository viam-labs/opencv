"""Tests for ChArUco support in src/models/camera_calibration.py.

Covers the target-detection wiring added for ChArUco. The instance is built
with ``object.__new__`` and the relevant attributes set directly, so the test
exercises ``_detect_target`` without the Viam resource lifecycle.
"""

import numpy as np

from models.camera_calibration import (
    CameraCalibration,
    TARGET_CHARUCO,
)
from utils.charuco_utils import build_charuco_board


SQUARES_X = 5
SQUARES_Y = 7


def _charuco_service():
    board = build_charuco_board(SQUARES_X, SQUARES_Y, 30.0, 22.0, "DICT_4X4_50")
    svc = object.__new__(CameraCalibration)
    svc.target_type = TARGET_CHARUCO
    svc.board = board
    return svc, board


def _render(board, px_per_square=120):
    w = SQUARES_X * px_per_square
    h = SQUARES_Y * px_per_square
    return board.generateImage((w, h), marginSize=px_per_square // 2)


def test_detect_target_charuco_returns_paired_float32_points():
    svc, board = _charuco_service()
    img = _render(board)

    out = svc._detect_target(img)
    assert out is not None
    objp, imgp = out

    # calibrateCamera requires float32 and matched object/image point counts.
    assert objp.dtype == np.float32
    assert imgp.dtype == np.float32
    assert objp.shape[1] == 3
    assert objp.shape[0] == imgp.reshape(-1, 2).shape[0]

    # Object points lie on the board plane (z == 0).
    assert np.allclose(objp[:, 2], 0.0)


def test_detect_target_charuco_returns_none_on_blank():
    svc, _ = _charuco_service()
    blank = np.full((480, 640), 255, dtype=np.uint8)
    assert svc._detect_target(blank) is None


def test_detect_target_charuco_partial_view_is_subset():
    svc, board = _charuco_service()
    img = _render(board)

    full = svc._detect_target(img)
    assert full is not None
    full_objp, _ = full

    # Crop away part of the board -> fewer corners, still valid for calibration.
    h, w = img.shape[:2]
    cropped = img[:, : (2 * w) // 3].copy()
    partial = svc._detect_target(cropped)
    assert partial is not None
    partial_objp, partial_imgp = partial

    assert partial_objp.shape[0] == partial_imgp.reshape(-1, 2).shape[0]
    assert 0 < partial_objp.shape[0] < full_objp.shape[0]
