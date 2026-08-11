"""Tests for the ``target`` attribute added to eye-to-hand auto-mode sampling.

Covers two things: that ``validate_config`` scopes ``target`` to eye-to-hand,
and that ``_sample_and_move_loop`` commands the motion service to move the
``target`` frame instead of the arm when it's configured. The service is
built with ``object.__new__`` and relevant attributes set directly, following
the pattern in ``test_camera_calibration.py`` — no full Viam resource
lifecycle needed.
"""

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose

import models.hand_eye_calibration as hec
from models.hand_eye_calibration import (
    CALIB_EYE_IN_HAND,
    CALIB_EYE_TO_HAND,
    HandEyeCalibration,
)


@pytest.fixture(autouse=True)
def _stub_go_mat2ov(monkeypatch):
    # _transform_to_viam_pose shells out to the compiled go_utils binary for
    # rotation-matrix -> orientation-vector conversion, which isn't built in
    # this environment. These tests only care about which frame gets moved,
    # not the orientation math (covered elsewhere), so stub it out.
    monkeypatch.setattr(hec, "call_go_mat2ov", lambda R: (0.0, 0.0, 1.0, 0.0))


def _config(attrs: dict) -> ComponentConfig:
    struct = Struct()
    struct.update(attrs)
    return ComponentConfig(attributes=struct)


_BASE_ATTRS = {
    "arm_name": "my_arm",
    "pose_tracker": "charuco_tracker",
    "motion": "motion",
    "method": "CALIB_HAND_EYE_TSAI",
    "pose_selection": "auto",
    "pose_sampling": {
        "workspace_bounds": {
            "x": {"min": 200, "max": 450},
            "y": {"min": -150, "max": 150},
            "z": {"min": 200, "max": 450},
        },
        "look_at_point": [700, 0, 300],
        "n_poses": 5,
    },
}


def test_validate_config_rejects_target_for_eye_in_hand():
    attrs = {**_BASE_ATTRS, "calibration_type": CALIB_EYE_IN_HAND, "target": "charuco_target"}
    with pytest.raises(Exception, match="target"):
        HandEyeCalibration.validate_config(_config(attrs))


def test_validate_config_accepts_target_for_eye_to_hand():
    attrs = {**_BASE_ATTRS, "calibration_type": CALIB_EYE_TO_HAND, "target": "charuco_target"}
    required, _optional = HandEyeCalibration.validate_config(_config(attrs))
    assert "my_arm" in required
    assert "charuco_tracker" in required


def test_validate_config_requires_target_for_eye_to_hand_auto():
    attrs = {**_BASE_ATTRS, "calibration_type": CALIB_EYE_TO_HAND}
    assert "target" not in attrs
    with pytest.raises(Exception, match="target"):
        HandEyeCalibration.validate_config(_config(attrs))


def test_validate_config_does_not_require_target_for_eye_to_hand_manual():
    attrs = {
        **_BASE_ATTRS,
        "calibration_type": CALIB_EYE_TO_HAND,
        "pose_selection": "manual",
        "poses": [{"x": 300, "y": 0, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 0}],
    }
    attrs.pop("pose_sampling")
    required, _optional = HandEyeCalibration.validate_config(_config(attrs))
    assert "my_arm" in required


def _service(calibration_type: str, target: str | None):
    """Build a HandEyeCalibration with just enough state for
    _sample_and_move_loop / _move_arm_to_position to run, mocking the motion
    service and arm. ``_capture_measurement`` is mocked directly rather than
    exercised for real, since the real path shells out to the compiled
    go_utils binary (not built in this environment) — out of scope for a
    test that's only checking which frame gets moved."""
    svc = object.__new__(HandEyeCalibration)
    svc.logger = __import__("logging").getLogger("test")

    class _Arm:
        name = "my_arm"

        async def get_end_position(self):
            return Pose(x=300, y=0, z=300, o_x=0, o_y=0, o_z=1, theta=0)

        async def is_moving(self):
            return False

    svc.arm = _Arm()
    svc.arm_name = "my_arm"

    svc.motion = AsyncMock()
    svc.motion.move = AsyncMock(return_value=True)

    svc._capture_measurement = AsyncMock(
        return_value={"arm_pose": Pose(x=300, y=0, z=300, o_x=0, o_y=0, o_z=1, theta=0)}
    )

    svc.calibration_type = calibration_type
    svc.target = target
    svc.solver = "opencv"
    svc.sleep_seconds = 0
    svc.use_motion_service_for_poses = False

    return svc


_SAMPLING = {
    "workspace_bounds": {
        "x": {"min": 200, "max": 450},
        "y": {"min": -150, "max": 150},
        "z": {"min": 200, "max": 450},
    },
    "look_at_point": [700, 0, 300],
    "roll_range_rad": (-np.pi, np.pi),
    "n_poses": 3,
    "max_attempts": 10,
    "seed": 0,
    "roll_reference": None,
}


def test_auto_sampling_moves_target_frame_when_configured():
    svc = _service(CALIB_EYE_TO_HAND, target="charuco_target")

    asyncio.run(svc._sample_and_move_loop(_SAMPLING))

    svc.motion.move.assert_awaited()
    _, kwargs = svc.motion.move.call_args
    assert kwargs["component_name"] == "charuco_target"


def test_auto_sampling_raises_when_eye_to_hand_without_target():
    # target is required for eye-to-hand auto mode; validate_config already
    # rejects this combination, but _sample_and_move_loop guards it too
    # (matching the existing defensive check for a missing motion service).
    svc = _service(CALIB_EYE_TO_HAND, target=None)

    with pytest.raises(Exception, match="target"):
        asyncio.run(svc._sample_and_move_loop(_SAMPLING))

    svc.motion.move.assert_not_awaited()


def test_auto_sampling_moves_arm_when_eye_in_hand_even_with_target_set():
    # target only applies to eye-to-hand; validate_config would reject this
    # combination, but the move loop itself should not use it either.
    svc = _service(CALIB_EYE_IN_HAND, target="charuco_target")

    asyncio.run(svc._sample_and_move_loop(_SAMPLING))

    svc.motion.move.assert_awaited()
    _, kwargs = svc.motion.move.call_args
    assert kwargs["component_name"] == "my_arm"
