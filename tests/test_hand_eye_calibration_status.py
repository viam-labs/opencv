import pytest

from src.models.hand_eye_calibration import (
    HandEyeCalibration,
    STATUS_CAPTURING,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_SAMPLING,
    STATUS_SOLVING,
    _initial_status,
)


def _bare_instance() -> HandEyeCalibration:
    inst = HandEyeCalibration.__new__(HandEyeCalibration)
    inst._status = _initial_status()
    return inst


def test_initial_status_shape():
    s = _initial_status()
    assert s["phase"] == STATUS_IDLE
    assert s["poses_captured"] == 0
    assert s["poses_target"] == 0
    assert s["attempts"] == 0
    assert s["last_updated"] is None
    assert s["last_error"] is None


def test_set_status_updates_named_fields_and_touches_timestamp():
    inst = _bare_instance()
    inst._set_status(phase=STATUS_SAMPLING, poses_target=20)
    assert inst._status["phase"] == STATUS_SAMPLING
    assert inst._status["poses_target"] == 20
    assert inst._status["last_updated"] is not None


def test_set_status_leaves_untouched_fields_alone():
    inst = _bare_instance()
    inst._set_status(phase=STATUS_SAMPLING, poses_target=20, poses_captured=3)
    inst._set_status(attempts=7)
    assert inst._status["poses_target"] == 20
    assert inst._status["poses_captured"] == 3
    assert inst._status["phase"] == STATUS_SAMPLING
    assert inst._status["attempts"] == 7


def test_set_status_records_error_dict():
    inst = _bare_instance()
    inst._set_status(phase=STATUS_ERROR, last_error={"kind": "execution", "message": "arm is stopped"})
    assert inst._status["phase"] == STATUS_ERROR
    assert inst._status["last_error"]["kind"] == "execution"
    assert inst._status["last_error"]["message"] == "arm is stopped"


def test_status_phase_constants_are_distinct():
    phases = {STATUS_IDLE, STATUS_SAMPLING, STATUS_CAPTURING, STATUS_SOLVING, STATUS_DONE, STATUS_ERROR}
    assert len(phases) == 6
