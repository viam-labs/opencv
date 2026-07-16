"""JSON-RPC 2.0 server that hosts the Python calibration resources on behalf
of the Go module entrypoint. Reads framed messages from stdin, dispatches to
resource instances, writes responses to stdout. Logs go to stderr so the Go
side can forward them into the module logger."""

import asyncio
import json
import os
import sys
import traceback
from typing import Any, Awaitable, Callable, Dict

from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

try:
    from models.camera_calibration import CameraCalibration
    from models.charuco import Charuco
    from models.chessboard import Chessboard
    from models.hand_eye_calibration import HandEyeCalibration
except ModuleNotFoundError:
    from .models.camera_calibration import CameraCalibration
    from .models.charuco import Charuco
    from .models.chessboard import Chessboard
    from .models.hand_eye_calibration import HandEyeCalibration


LOG = getLogger(__name__)


MODEL_FACTORIES: Dict[str, Callable[..., Any]] = {
    "chessboard": Chessboard,
    "charuco": Charuco,
    "hand_eye_calibration": HandEyeCalibration,
    "camera_calibration": CameraCalibration,
}


class ResourceRegistry:
    def __init__(self):
        self._resources: Dict[str, Any] = {}

    async def new(self, model: str, name: str, attributes: Dict[str, Any], deps: list) -> None:
        factory = MODEL_FACTORIES.get(model)
        if factory is None:
            raise LookupError(f"unknown model {model!r}")
        cfg = ComponentConfig(name=name, attributes=dict_to_struct(attributes))
        instance = factory(name)
        await instance.reconfigure(cfg, {})
        self._resources[name] = instance

    async def reconfigure(self, model: str, name: str, attributes: Dict[str, Any], deps: list) -> None:
        instance = self._resources.get(name)
        if instance is None:
            await self.new(model, name, attributes, deps)
            return
        cfg = ComponentConfig(name=name, attributes=dict_to_struct(attributes))
        await instance.reconfigure(cfg, {})

    async def call(self, name: str, method: str, args: Dict[str, Any]) -> Any:
        instance = self._resources.get(name)
        if instance is None:
            raise LookupError(f"unknown resource {name!r}")
        fn = getattr(instance, method, None)
        if fn is None:
            raise AttributeError(f"resource {name!r} has no method {method!r}")
        result = fn(**(args or {}))
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def close(self, name: str) -> None:
        instance = self._resources.pop(name, None)
        if instance is not None:
            close_fn = getattr(instance, "close", None)
            if close_fn is not None:
                result = close_fn()
                if asyncio.iscoroutine(result):
                    await result


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    length = -1
    while True:
        line = await reader.readline()
        if not line:
            raise EOFError()
        text = line.decode("ascii").rstrip("\r\n")
        if text == "":
            break
        if text.startswith("Content-Length:"):
            length = int(text.split(":", 1)[1].strip())
    if length < 0:
        raise ValueError("frame missing Content-Length")
    body = await reader.readexactly(length)
    return body


def write_frame(payload: bytes) -> None:
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def send_response(request_id: int, result: Any = None, error: Dict[str, Any] = None) -> None:
    body: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        body["error"] = error
    else:
        body["result"] = result
    write_frame(json.dumps(body, default=_json_default).encode("utf-8"))


def _json_default(o):
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


HANDLERS = {}


def handler(name: str):
    def wrap(fn):
        HANDLERS[name] = fn
        return fn
    return wrap


@handler("resource.new")
async def handle_new(registry: ResourceRegistry, params: Dict[str, Any]) -> Any:
    await registry.new(
        model=params["model"],
        name=params["name"],
        attributes=params.get("config") or {},
        deps=params.get("deps") or [],
    )
    return {}


@handler("resource.reconfigure")
async def handle_reconfigure(registry: ResourceRegistry, params: Dict[str, Any]) -> Any:
    await registry.reconfigure(
        model=params["model"],
        name=params["name"],
        attributes=params.get("config") or {},
        deps=params.get("deps") or [],
    )
    return {}


@handler("resource.call")
async def handle_call(registry: ResourceRegistry, params: Dict[str, Any]) -> Any:
    return await registry.call(
        name=params["name"],
        method=params["method"],
        args=params.get("args") or {},
    )


@handler("resource.close")
async def handle_close(registry: ResourceRegistry, params: Dict[str, Any]) -> Any:
    await registry.close(params["name"])
    return {}


async def serve():
    if "VIAM_MODULE_SOCKET" not in os.environ:
        print("VIAM_MODULE_SOCKET env var required", file=sys.stderr)
        sys.exit(2)

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    registry = ResourceRegistry()
    LOG.info("opencv python module server ready")

    while True:
        try:
            raw = await read_frame(reader)
        except EOFError:
            LOG.info("stdin closed; exiting")
            return

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            LOG.error(f"malformed json frame: {e}")
            continue

        request_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        handler_fn = HANDLERS.get(method)
        if handler_fn is None:
            send_response(request_id, error={"code": -32601, "message": f"unknown method {method!r}"})
            continue

        try:
            result = await handler_fn(registry, params)
            send_response(request_id, result=result)
        except Exception as e:
            LOG.error(f"{method} failed: {e}\n{traceback.format_exc()}")
            send_response(request_id, error={
                "code": -32000,
                "message": str(e),
                "data": {"kind": "python_exception", "type": type(e).__name__},
            })


if __name__ == "__main__":
    asyncio.run(serve())
