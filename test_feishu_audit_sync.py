import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feishu_audit_sync import FeishuAuditConfig, FeishuAuditSyncService
from qt_gui import OperationAuditStore


class _MockFeishuHandler(BaseHTTPRequestHandler):
    records = {}
    create_count = 0
    update_count = 0
    delete_count = 0

    def log_message(self, _format, *_args):
        return

    def _body(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def _reply(self, payload):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = self._body()
        if self.path.endswith("/auth/v3/tenant_access_token/internal"):
            self._reply({"code": 0, "tenant_access_token": "test-token", "expire": 7200})
            return
        type(self).create_count += 1
        record_id = f"rec-{type(self).create_count}"
        type(self).records[record_id] = body["fields"]
        self._reply({"code": 0, "data": {"record": {"record_id": record_id}}})

    def do_PUT(self):
        body = self._body()
        record_id = self.path.rsplit("/", 1)[-1]
        type(self).update_count += 1
        type(self).records[record_id] = body["fields"]
        self._reply({"code": 0, "data": {"record": {"record_id": record_id}}})

    def do_DELETE(self):
        record_id = self.path.rsplit("/", 1)[-1]
        type(self).delete_count += 1
        type(self).records.pop(record_id, None)
        self._reply({"code": 0, "data": {"deleted": True}})


class FeishuAuditSyncTests(unittest.TestCase):
    def setUp(self):
        _MockFeishuHandler.records = {}
        _MockFeishuHandler.create_count = 0
        _MockFeishuHandler.update_count = 0
        _MockFeishuHandler.delete_count = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _MockFeishuHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "audit.sqlite3"
        self.config = FeishuAuditConfig(
            app_id="app", app_secret="secret", app_token="base", table_id="table",
            api_base=f"http://127.0.0.1:{self.server.server_port}/open-apis",
            request_timeout=2.0,
        )
        self.store = OperationAuditStore(self.database_path, cloud_sync_enabled=True)
        self.service = FeishuAuditSyncService(self.database_path, self.config)
        self.service.start()

    def tearDown(self):
        self.service.stop(timeout=2.0)
        self.server.shutdown()
        self.server.server_close()
        self.temporary.cleanup()

    def _wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.03)
        self.fail("timed out waiting for background synchronization")

    def _row(self):
        with closing(sqlite3.connect(self.database_path)) as database:
            return database.execute(
                "SELECT sync_status, feishu_record_id, summary FROM operation_audit WHERE session_id='s1'"
            ).fetchone()

    def test_one_session_is_created_then_updated_and_discarded(self):
        self.store.append("测试员", "s1", "session_start", summary="程序启动", state={"step": 1})
        self.service.notify()
        self._wait_for(lambda: self._row() and self._row()[0] == "synced")
        self.assertEqual(_MockFeishuHandler.create_count, 1)
        self.assertEqual(len(_MockFeishuHandler.records), 1)

        self.store.append("测试员", "s1", "filter", summary="完成滤波", state={"step": 2})
        self.service.notify()
        self._wait_for(lambda: self._row() and self._row()[0] == "synced" and _MockFeishuHandler.update_count >= 1)
        self.assertEqual(_MockFeishuHandler.create_count, 1)
        self.assertIn("完成滤波", self._row()[2])

        self.store.discard_session("s1")
        self.service.notify()
        self._wait_for(lambda: _MockFeishuHandler.delete_count >= 1)
        self.assertIsNone(self._row())
        self.assertEqual(_MockFeishuHandler.records, {})

    def test_unconfigured_rows_remain_local(self):
        path = Path(self.temporary.name) / "local_only.sqlite3"
        store = OperationAuditStore(path, cloud_sync_enabled=False)
        store.append("离线用户", "offline", "session_start", summary="本地记录")
        with closing(sqlite3.connect(path)) as database:
            status = database.execute(
                "SELECT sync_status FROM operation_audit WHERE session_id='offline'"
            ).fetchone()[0]
        self.assertEqual(status, "not_configured")


if __name__ == "__main__":
    unittest.main()
