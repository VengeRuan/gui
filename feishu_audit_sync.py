"""Background synchronization of GUI operation audits to Feishu Bitable.

The GUI always writes SQLite first.  This module only mirrors committed rows
and never runs network I/O on the Qt event thread.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


FEISHU_FIELD_NAMES = (
    "会话ID", "操作者", "电脑名称", "开始时间", "最后更新时间",
    "会话状态", "数据文件", "操作时间线", "完整状态JSON", "同步来源",
)


@dataclass(frozen=True)
class FeishuAuditConfig:
    app_id: str = ""
    app_secret: str = ""
    app_token: str = ""
    table_id: str = ""
    api_base: str = "https://open.feishu.cn/open-apis"
    request_timeout: float = 12.0

    @classmethod
    def from_environment(cls) -> "FeishuAuditConfig":
        try:
            timeout = float(os.environ.get("SD_GUI_FEISHU_TIMEOUT_SECONDS", "12") or 12)
        except (TypeError, ValueError):
            timeout = 12.0
        return cls(
            app_id=os.environ.get("SD_GUI_FEISHU_APP_ID", "").strip(),
            app_secret=os.environ.get("SD_GUI_FEISHU_APP_SECRET", "").strip(),
            app_token=os.environ.get("SD_GUI_FEISHU_APP_TOKEN", "").strip(),
            table_id=os.environ.get("SD_GUI_FEISHU_TABLE_ID", "").strip(),
            api_base=os.environ.get(
                "SD_GUI_FEISHU_API_BASE", "https://open.feishu.cn/open-apis"
            ).strip().rstrip("/"),
            request_timeout=max(2.0, timeout),
        )

    @property
    def is_configured(self) -> bool:
        return all((self.app_id, self.app_secret, self.app_token, self.table_id))

    @property
    def missing_environment_variables(self) -> tuple[str, ...]:
        pairs = (
            ("SD_GUI_FEISHU_APP_ID", self.app_id),
            ("SD_GUI_FEISHU_APP_SECRET", self.app_secret),
            ("SD_GUI_FEISHU_APP_TOKEN", self.app_token),
            ("SD_GUI_FEISHU_TABLE_ID", self.table_id),
        )
        return tuple(name for name, value in pairs if not value)


class FeishuApiError(RuntimeError):
    pass


class FeishuBitableClient:
    def __init__(self, config: FeishuAuditConfig):
        self.config = config
        self._tenant_token = ""
        self._tenant_token_expires_at = 0.0

    def _request(self, method: str, path: str, payload=None, *, authenticated=True):
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._get_tenant_token()}"
        request = urllib.request.Request(
            f"{self.config.api_base}/{path.lstrip('/')}", data=body,
            headers=headers, method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise FeishuApiError(f"网络请求失败：{exc}") from exc
        try:
            result = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise FeishuApiError("飞书返回了无法解析的响应") from exc
        if int(result.get("code", 0) or 0) != 0:
            raise FeishuApiError(
                f"飞书 API 错误 {result.get('code')}：{result.get('msg') or result.get('message') or '未知错误'}"
            )
        return result

    def _get_tenant_token(self) -> str:
        if self._tenant_token and time.monotonic() < self._tenant_token_expires_at:
            return self._tenant_token
        result = self._request(
            "POST", "auth/v3/tenant_access_token/internal",
            {"app_id": self.config.app_id, "app_secret": self.config.app_secret},
            authenticated=False,
        )
        token = str(result.get("tenant_access_token", ""))
        if not token:
            raise FeishuApiError("飞书未返回 tenant_access_token")
        expire = max(60, int(result.get("expire", 7200) or 7200))
        self._tenant_token = token
        self._tenant_token_expires_at = time.monotonic() + max(30, expire - 120)
        return token

    def _record_path(self, record_id: str = "") -> str:
        app_token = urllib.parse.quote(self.config.app_token, safe="")
        table_id = urllib.parse.quote(self.config.table_id, safe="")
        base = f"bitable/v1/apps/{app_token}/tables/{table_id}/records"
        return f"{base}/{urllib.parse.quote(record_id, safe='')}" if record_id else base

    def create_record(self, fields: dict) -> str:
        result = self._request("POST", self._record_path(), {"fields": fields})
        record_id = str(result.get("data", {}).get("record", {}).get("record_id", ""))
        if not record_id:
            raise FeishuApiError("飞书创建记录成功但未返回 record_id")
        return record_id

    def update_record(self, record_id: str, fields: dict) -> None:
        self._request("PUT", self._record_path(record_id), {"fields": fields})

    def delete_record(self, record_id: str) -> None:
        self._request("DELETE", self._record_path(record_id))


class FeishuAuditSyncService:
    """Single background worker that coalesces SQLite changes by session_id."""

    def __init__(self, database_path: str | Path, config: FeishuAuditConfig):
        self.database_path = Path(database_path)
        self.config = config
        self.client = FeishuBitableClient(config)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="feishu-audit-sync", daemon=True
        )

    def start(self) -> None:
        if self.config.is_configured and not self._thread.is_alive():
            self._thread.start()
            self.notify()

    def notify(self) -> None:
        if self.config.is_configured:
            self._wake.set()

    def stop(self, timeout: float = 4.0) -> None:
        if not self._thread.is_alive():
            return
        self._stop.set()
        self._wake.set()
        self._thread.join(max(0.0, float(timeout)))

    def _connect(self):
        database = sqlite3.connect(str(self.database_path), timeout=30.0)
        database.execute("PRAGMA busy_timeout=30000")
        return database

    @staticmethod
    def _shorten(value, limit: int) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 18)] + "\n…（飞书显示已截断）"

    def _fields_for_row(self, row) -> dict:
        event_time, operator, session_id, action, data_path, summary, state_json = row[:7]
        try:
            document = json.loads(state_json or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            document = {}
        started = str(document.get("started_time", ""))
        return {
            "会话ID": str(session_id),
            "操作者": str(operator),
            "电脑名称": platform.node() or os.environ.get("COMPUTERNAME", ""),
            "开始时间": started,
            "最后更新时间": str(event_time),
            "会话状态": "已关闭并保存" if action == "session_closed" else "操作中",
            "数据文件": self._shorten(data_path, 5000),
            "操作时间线": self._shorten(summary, 90000),
            "完整状态JSON": self._shorten(state_json, 90000),
            "同步来源": "SD EEG Qt GUI / SQLite",
        }

    def _pending_rows(self, limit=30):
        with closing(self._connect()) as database:
            return database.execute(
                """SELECT event_time, operator_name, session_id, action, data_path, summary,
                          state_json, feishu_record_id
                   FROM operation_audit
                   WHERE sync_status IN ('pending', 'failed')
                   ORDER BY event_time ASC LIMIT ?""",
                (int(limit),),
            ).fetchall()

    def _pending_deletions(self, limit=30):
        with closing(self._connect()) as database:
            return database.execute(
                """SELECT session_id, feishu_record_id FROM operation_audit_sync_deletions
                   ORDER BY requested_time ASC LIMIT ?""", (int(limit),)
            ).fetchall()

    def _mark_success(
        self, session_id: str, event_time: str, state_json: str, record_id: str
    ) -> bool:
        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        with closing(self._connect()) as database:
            cursor = database.execute(
                """UPDATE operation_audit SET feishu_record_id=?, sync_status='synced',
                          last_sync_time=?, sync_error='', sync_attempts=0
                   WHERE session_id=? AND event_time=? AND state_json=?""",
                (record_id, now, session_id, event_time, state_json),
            )
            database.commit()
            return cursor.rowcount > 0

    def _mark_failure(self, session_id: str, error: Exception) -> None:
        message = self._shorten(str(error), 2000)
        with closing(self._connect()) as database:
            database.execute(
                """UPDATE operation_audit SET sync_status='failed', sync_error=?,
                          sync_attempts=sync_attempts + 1 WHERE session_id=?""",
                (message, session_id),
            )
            database.commit()

    def _finish_deletion(self, session_id: str) -> None:
        with closing(self._connect()) as database:
            database.execute(
                "DELETE FROM operation_audit_sync_deletions WHERE session_id=?", (session_id,)
            )
            database.commit()

    def _fail_deletion(self, session_id: str, error: Exception) -> None:
        with closing(self._connect()) as database:
            database.execute(
                """UPDATE operation_audit_sync_deletions SET attempts=attempts + 1, error=?
                   WHERE session_id=?""", (self._shorten(str(error), 2000), session_id)
            )
            database.commit()

    def _sync_once(self) -> bool:
        did_work = False
        for session_id, record_id in self._pending_deletions():
            did_work = True
            try:
                self.client.delete_record(str(record_id))
                self._finish_deletion(str(session_id))
            except Exception as exc:
                self._fail_deletion(str(session_id), exc)
                raise
        for row in self._pending_rows():
            did_work = True
            event_time, _operator, session_id, _action, _path, _summary, _json, record_id = row
            created_id = ""
            try:
                fields = self._fields_for_row(row)
                if record_id:
                    self.client.update_record(str(record_id), fields)
                    final_id = str(record_id)
                else:
                    created_id = self.client.create_record(fields)
                    final_id = created_id
                if not self._mark_success(
                    str(session_id), str(event_time), str(_json), final_id
                ) and created_id:
                    # The user selected “do not save” while the POST was in flight.
                    self.client.delete_record(created_id)
            except Exception as exc:
                self._mark_failure(str(session_id), exc)
                raise
        return did_work

    def _run(self) -> None:
        retry_delay = 5.0
        while True:
            self._wake.wait(timeout=retry_delay)
            self._wake.clear()
            try:
                did_work = self._sync_once()
                retry_delay = 1.0 if did_work else 30.0
            except Exception:
                retry_delay = min(60.0, max(5.0, retry_delay * 2.0))
            if self._stop.is_set():
                # One final best-effort drain after the close decision.
                try:
                    self._sync_once()
                except Exception:
                    pass
                return
