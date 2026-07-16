import asyncio
import json
import os
import sys
from typing import Optional, Sequence

from google.protobuf.json_format import MessageToJson

from viam.components.arm import Arm
from viam.proto.common import Pose, PoseInFrame


class PlanningError(Exception):
    pass


class ExecutionError(Exception):
    pass


DEFAULT_TIMEOUT_SECONDS = 300.0


def _arm_address(arm: Arm) -> str:
    channel = getattr(arm, "channel", None)
    if channel is None:
        raise ExecutionError(f"arm {arm.name!r} has no gRPC channel")
    path = getattr(channel, "_path", None)
    if path:
        return f"unix://{path}"
    host = getattr(channel, "_host", None)
    port = getattr(channel, "_port", None)
    if host and port:
        return f"{host}:{port}"
    raise ExecutionError(f"cannot derive viam-server address from arm {arm.name!r} channel")


def _binary_path() -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "arm-planner")  # type: ignore[attr-defined]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "bin", "arm-planner")


async def plan_and_execute(
    *,
    arm: Arm,
    goal_pose: Optional[Pose] = None,
    goal_joints_deg: Optional[Sequence[float]] = None,
    reference_frame: str = "world",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    if (goal_pose is None) == (goal_joints_deg is None):
        raise ValueError("exactly one of goal_pose or goal_joints_deg must be set")

    if goal_pose is not None:
        pif = PoseInFrame(reference_frame=reference_frame, pose=goal_pose)
        goal_json = json.dumps({"pose": json.loads(MessageToJson(pif, preserving_proto_field_name=True))})
    else:
        goal_json = json.dumps({"joints_degrees": list(goal_joints_deg)})

    proc = await asyncio.create_subprocess_exec(
        _binary_path(),
        "--arm", arm.name,
        "--parent-addr", _arm_address(arm),
        "--goal", goal_json,
        "--timeout", f"{int(timeout_seconds)}s",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds + 5)
    except asyncio.TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise ExecutionError(f"arm-planner exceeded {timeout_seconds}s wall timeout") from e

    _raise_from_output(proc.returncode or 0, stdout, stderr)


def _raise_from_output(returncode: int, stdout: bytes, stderr: bytes) -> None:
    text = stdout.decode(errors="replace").strip()
    if not text:
        raise ExecutionError(
            f"arm-planner exited {returncode} with empty stdout; stderr: {stderr.decode(errors='replace').strip()[:500]}"
        )
    try:
        result = json.loads(text.splitlines()[-1])
    except json.JSONDecodeError as e:
        raise ExecutionError(f"arm-planner returned invalid JSON: {e}; stdout: {text[:500]}") from e

    if result.get("ok"):
        return

    err = result.get("error") or {}
    kind = err.get("kind", "execution")
    message = err.get("message", "unknown error")
    if kind == "planning":
        raise PlanningError(message)
    raise ExecutionError(message)
