import json
import os
import sys
import types
from unittest.mock import AsyncMock, patch

import pytest

from src.utils.arm_planner import (
    ExecutionError,
    PlanningError,
    _arm_address,
    _binary_path,
    _raise_from_output,
    plan_and_execute,
)


def _fake_arm(path="/tmp/viam.sock"):
    arm = types.SimpleNamespace()
    arm.name = "arm1"
    arm.channel = types.SimpleNamespace(_path=path, _host=None, _port=None)
    return arm


def test_arm_address_unix():
    assert _arm_address(_fake_arm(path="/tmp/viam.sock")) == "unix:///tmp/viam.sock"


def test_arm_address_tcp():
    arm = types.SimpleNamespace(
        name="arm1",
        channel=types.SimpleNamespace(_path=None, _host="10.0.0.1", _port=8080),
    )
    assert _arm_address(arm) == "10.0.0.1:8080"


def test_arm_address_missing_channel_raises():
    arm = types.SimpleNamespace(name="arm1")
    with pytest.raises(ExecutionError):
        _arm_address(arm)


def test_binary_path_dev_mode_uses_repo_bin():
    with patch.object(sys, "frozen", False, create=True):
        path = _binary_path()
    assert path.endswith(os.path.join("bin", "arm-planner"))
    assert os.path.isabs(path)


def test_binary_path_frozen_mode_uses_meipass():
    with patch.object(sys, "frozen", True, create=True), \
         patch.object(sys, "_MEIPASS", "/tmp/fake-meipass", create=True):
        assert _binary_path() == "/tmp/fake-meipass/arm-planner"


def test_raise_from_output_ok_is_silent():
    _raise_from_output(0, b'{"ok":true}', b"")


def test_raise_from_output_planning_kind():
    payload = json.dumps({"error": {"kind": "planning", "message": "no ik"}}).encode()
    with pytest.raises(PlanningError, match="no ik"):
        _raise_from_output(1, payload, b"")


def test_raise_from_output_execution_kind():
    payload = json.dumps({"error": {"kind": "execution", "message": "pstop"}}).encode()
    with pytest.raises(ExecutionError, match="pstop"):
        _raise_from_output(1, payload, b"")


def test_raise_from_output_unknown_kind_treated_as_execution():
    payload = json.dumps({"error": {"kind": "weird", "message": "?"}}).encode()
    with pytest.raises(ExecutionError):
        _raise_from_output(1, payload, b"")


def test_raise_from_output_empty_stdout_is_execution_error():
    with pytest.raises(ExecutionError):
        _raise_from_output(1, b"", b"boom")


def test_raise_from_output_invalid_json_is_execution_error():
    with pytest.raises(ExecutionError, match="invalid JSON"):
        _raise_from_output(1, b"not json", b"")


@pytest.mark.asyncio
async def test_plan_and_execute_rejects_missing_goal():
    with pytest.raises(ValueError):
        await plan_and_execute(arm=_fake_arm())


@pytest.mark.asyncio
async def test_plan_and_execute_rejects_both_goals():
    from viam.proto.common import Pose
    with pytest.raises(ValueError):
        await plan_and_execute(
            arm=_fake_arm(),
            goal_pose=Pose(x=0, y=0, z=0, o_x=0, o_y=0, o_z=1, theta=0),
            goal_joints_deg=[0, 0],
        )


@pytest.mark.asyncio
async def test_plan_and_execute_success_via_mocked_subprocess():
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(b'{"ok":true}', b""))
    proc.returncode = 0

    with patch("src.utils.arm_planner.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        await plan_and_execute(arm=_fake_arm(), goal_joints_deg=[10.0, 20.0])


@pytest.mark.asyncio
async def test_plan_and_execute_planning_error_via_mocked_subprocess():
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(
        b'{"error":{"kind":"planning","message":"no ik solution"}}', b""
    ))
    proc.returncode = 1

    with patch("src.utils.arm_planner.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        with pytest.raises(PlanningError, match="no ik"):
            await plan_and_execute(arm=_fake_arm(), goal_joints_deg=[10.0])
