"""Private loopback bridge client. No Qt, SDK, shell execution or file browsing."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import uuid

API_VERSION = 1
IMPLEMENTATION_VERSION = "animalstart-mcp-v1"
MAX_REQUEST_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class BridgeError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def session_directory():
    override = os.environ.get("ANIMAL_PEAK_MCP_SESSION_DIR")
    if override:
        return Path(override).resolve()
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home()/".local/share")))
    return base/"SD_BIN_Analysis"/"mcp_sessions"


def _session_path(session_id, directory=None):
    try:
        if str(uuid.UUID(session_id)) != session_id:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise BridgeError("invalid_session", "session_id必须来自list_sessions，不能是路径。") from None
    return (Path(directory) if directory is not None else session_directory())/(session_id+".json")


def list_sessions(directory=None):
    root = Path(directory) if directory is not None else session_directory()
    result = []
    if not root.is_dir():
        return result
    for path in sorted(root.glob("*.json"))[:64]:
        try:
            if path.stat().st_size > 16384:
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if _session_path(row["session_id"], root) != path or row["api_version"] != API_VERSION:
                continue
            public = {key: row[key] for key in ("session_id", "api_version", "implementation_version",
                "pid", "project", "source_name", "created_utc")}
            public["session_kind"] = row.get("session_kind", "animalstart")
            result.append(public)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result  # Tokens/ports are deliberately never exposed to the model.


def call_bridge(session_id, operation, arguments=None, *, expected_revision=None, directory=None, timeout=15):
    path = _session_path(session_id, directory)
    try:
        if path.stat().st_size > 16384:
            raise ValueError("session registry too large")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["api_version"] != API_VERSION or record["session_id"] != session_id:
            raise ValueError("session version/id mismatch")
        port = int(record["port"])
        if not 1 <= port <= 65535 or len(record["token"]) < 32:
            raise ValueError("invalid registry")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BridgeError("session_unavailable", "找不到有效会话；请在目标测试窗口启用MCP控制。") from exc
    request = dict(api_version=API_VERSION, session_id=session_id, token=record["token"],
                   operation=operation, arguments=arguments or {}, expected_revision=expected_revision)
    payload = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")+b"\n"
    if len(payload) > MAX_REQUEST_BYTES:
        raise BridgeError("request_too_large", "请求过大，请减少参数/Trial数量。")
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(payload)
            chunks = bytearray()
            while b"\n" not in chunks:
                chunk = connection.recv(65536)
                if not chunk:
                    raise OSError("connection closed")
                chunks.extend(chunk)
                if len(chunks) > MAX_RESPONSE_BYTES:
                    raise BridgeError("response_too_large", "响应过大，请分页读取。")
        response = json.loads(chunks.split(b"\n", 1)[0])
    except (OSError, ValueError) as exc:
        if isinstance(exc, BridgeError):
            raise
        raise BridgeError("connection_failed", "会话不可用或超时；确认GUI仍运行、控制开关开启且SSH与GUI属于同一账号。") from exc
    if not response.get("ok"):
        error = response.get("error", {})
        raise BridgeError(error.get("code", "operation_failed"), error.get("message", "操作未完成"))
    return response
