"""Opt-in authenticated Qt-main-thread bridge. Binds ONLY IPv4 loopback."""
from __future__ import annotations

import csv
import ctypes
from datetime import datetime, timezone
import hmac
import json
import os
from pathlib import Path
import secrets
import subprocess
import uuid

try:
    from PyQt6 import QtCore, QtNetwork
except ImportError:
    from PyQt5 import QtCore, QtNetwork

from animal_peak_rpc import API_VERSION, IMPLEMENTATION_VERSION, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, session_directory


def protect_private_path(path, *, directory=False):
    """Fail closed if a new token directory cannot be made private."""
    if os.name != "nt":
        path.chmod(0o700 if directory else 0o600)
        return
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    identity = subprocess.check_output(["whoami", "/user", "/fo", "csv", "/nh"],
        creationflags=flags, timeout=5).decode("utf-8", errors="replace").strip()
    sid = next(csv.reader([identity]))[-1]
    if not sid.startswith("S-1-"):
        raise OSError("不能确认当前Windows账号SID，未启用MCP。")
    # Replace the COMPLETE DACL, not only inherited entries. Existing explicit
    # grants to Everyone/other accounts must not survive on a token directory.
    from ctypes import wintypes
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    set_security = advapi.SetFileSecurityW
    set_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    set_security.restype = wintypes.BOOL
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]; kernel.LocalFree.restype = ctypes.c_void_p
    inheritance = "OICI" if directory else ""
    descriptor = ctypes.c_void_p()
    sddl = f"D:P(A;{inheritance};FA;;;{sid})(A;{inheritance};FA;;;SY)(A;{inheritance};FA;;;BA)"
    if not convert(sddl, 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not set_security(str(path), 0x80000004, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(descriptor)


class AnimalPeakBridge(QtCore.QObject):
    def __init__(self, controller, *, directory=None):
        super().__init__(controller.dialog)
        self.controller = controller
        self.session_id = str(uuid.uuid4())
        self.token = secrets.token_hex(32)
        self.registry_path = None
        self.clients = {}
        self.server = QtNetwork.QTcpServer(self)
        if not self.server.listen(QtNetwork.QHostAddress("127.0.0.1"), 0):
            raise OSError("不能建立本机MCP桥接。")
        try:
            root = Path(directory) if directory is not None else session_directory()
            root = root.resolve()
            if root in (Path(root.anchor), Path.home().resolve()):
                raise ValueError("会话目录必须是专用子目录，不能使用磁盘根目录或用户目录。")
            root.mkdir(parents=True, exist_ok=True)
            protect_private_path(root, directory=True)
            path = root/(self.session_id+".json")
            # Directory is already private before the token is written.
            path.write_text(json.dumps(dict(session_id=self.session_id, api_version=API_VERSION,
                implementation_version=IMPLEMENTATION_VERSION, token=self.token, port=self.server.serverPort(),
                pid=os.getpid(), project=str(Path(__file__).resolve().parent),
                source_name=Path(controller.dialog.source.metadata.path).name,
                created_utc=datetime.now(timezone.utc).isoformat()), ensure_ascii=False), encoding="utf-8")
            self.registry_path = path
            protect_private_path(path)
        except Exception:
            self.close()
            raise
        self.server.newConnection.connect(self._accept)
        controller.dialog.destroyed.connect(lambda *_: path.unlink(missing_ok=True))

    def _accept(self):
        while self.server.hasPendingConnections():
            connection = self.server.nextPendingConnection()
            if len(self.clients) >= 8:
                connection.abort(); connection.deleteLater(); continue
            self.clients[connection] = bytearray()
            connection.readyRead.connect(lambda c=connection: self._read(c))
            connection.disconnected.connect(lambda c=connection: self._forget(c))
            timer = QtCore.QTimer(connection); timer.setSingleShot(True)
            timer.timeout.connect(connection.abort); timer.start(15000)

    def _forget(self, connection):
        self.clients.pop(connection, None)
        connection.deleteLater()

    def _read(self, connection):
        pending = self.clients.get(connection)
        if pending is None:
            return
        pending.extend(bytes(connection.readAll()))
        if len(pending) > MAX_REQUEST_BYTES:
            connection.abort(); return
        if b"\n" not in pending:
            return
        try:
            request = json.loads(pending.split(b"\n", 1)[0])
            if (not isinstance(request, dict) or request.get("api_version") != API_VERSION
                    or request.get("session_id") != self.session_id
                    or not isinstance(request.get("token"), str)
                    or not hmac.compare_digest(request["token"], self.token)):
                response = dict(ok=False, error=dict(code="unauthorized", message="会话认证失败。"))
            else:
                response = self.controller.dispatch(request.get("operation"), request.get("arguments", {}),
                                                    request.get("expected_revision"))
                response["session_id"] = self.session_id
        except Exception:
            response = dict(ok=False, error=dict(code="invalid_request", message="无效请求；未执行操作。"))
        data = json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")+b"\n"
        if len(data) > MAX_RESPONSE_BYTES:
            data = b'{"ok":false,"error":{"code":"response_too_large","message":"Use pagination"}}\n'
        self.clients.pop(connection, None)  # One request per connection; no pipelining/replay.
        connection.write(data); connection.disconnectFromHost()

    def close(self):
        for connection in list(self.clients):
            connection.abort()
        self.clients.clear()
        self.server.close()
        if self.registry_path is not None:
            self.registry_path.unlink(missing_ok=True)
