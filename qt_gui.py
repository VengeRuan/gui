"""Qt front-end for the SD EEG analysis application.

This Qt interface uses PyQtGraph for every interactive plot.  The HDF5 source
deliberately stays lazy: no full recording is read when a file is opened or a
preview is redrawn.

The legacy ``gui.py`` remains available during migration for processing pages
which have not yet moved to Qt.  This file is a separate entry point so the
two widget toolkits are never placed in one event loop.
"""

from __future__ import annotations

import sys
import json
import html
import re
import os
import sqlite3
import shutil
import warnings
import threading
import uuid
from time import perf_counter
from datetime import datetime, timezone
from copy import deepcopy
from types import SimpleNamespace
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from fractions import Fraction
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import hdf5plugin

# Prefer PyQt6 on computers where it is installed, and transparently fall
# back to PyQt5 elsewhere.  pyqtgraph must be imported only after this choice
# so that the whole process uses one Qt binding.
try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    QT_BINDING = "PyQt6"
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets
    QT_BINDING = "PyQt5"

os.environ["PYQTGRAPH_QT_LIB"] = QT_BINDING
import pyqtgraph as pg

Qt = QtCore.Qt
QRectF, QObject, pyqtSignal, QThread, QTimer = (
    QtCore.QRectF, QtCore.QObject, QtCore.pyqtSignal, QtCore.QThread, QtCore.QTimer,
)
QEvent = QtCore.QEvent
QColor, QFont, QPainter, QPdfWriter = QtGui.QColor, QtGui.QFont, QtGui.QPainter, QtGui.QPdfWriter
from qt_data_model import (
    ArraySource,
    allocate_storage,
    cleanup_process_cache_files,
    CustomChannelH5Source,
    build_task_markers,
    match_behavior_events_to_markers,
    LazyH5Source,
    filter_array,
    parse_alignment_log,
    read_channel_layout,
    read_channel_remap,
    read_channel_remap_for_ids,
    PROCESS_CHUNK_SAMPLES,
    read_behavior_detection_results,
    remap_source_from_excel,
    remap_source_streaming,
)
from compare_bin_storage import BinReader, create_h5, inspect_bin, parse_filename_metadata, read_timing_metadata
QApplication = QtWidgets.QApplication
QAbstractSlider, QAbstractSpinBox = QtWidgets.QAbstractSlider, QtWidgets.QAbstractSpinBox
QButtonGroup, QCheckBox, QComboBox, QFileDialog = (
    QtWidgets.QButtonGroup, QtWidgets.QCheckBox, QtWidgets.QComboBox, QtWidgets.QFileDialog,
)
QGraphicsDropShadowEffect, QDoubleSpinBox = QtWidgets.QGraphicsDropShadowEffect, QtWidgets.QDoubleSpinBox
QDialog, QFormLayout, QFrame = QtWidgets.QDialog, QtWidgets.QFormLayout, QtWidgets.QFrame
QGridLayout, QGroupBox, QHBoxLayout = QtWidgets.QGridLayout, QtWidgets.QGroupBox, QtWidgets.QHBoxLayout
QLabel, QLineEdit, QInputDialog = QtWidgets.QLabel, QtWidgets.QLineEdit, QtWidgets.QInputDialog
QMainWindow, QMessageBox, QPushButton = QtWidgets.QMainWindow, QtWidgets.QMessageBox, QtWidgets.QPushButton
QProgressBar, QPlainTextEdit, QScrollArea = QtWidgets.QProgressBar, QtWidgets.QPlainTextEdit, QtWidgets.QScrollArea
QSizePolicy, QSpinBox, QStatusBar = QtWidgets.QSizePolicy, QtWidgets.QSpinBox, QtWidgets.QStatusBar
QStackedWidget, QSplitter, QTabWidget = QtWidgets.QStackedWidget, QtWidgets.QSplitter, QtWidgets.QTabWidget
QTableWidget, QTableWidgetItem = QtWidgets.QTableWidget, QtWidgets.QTableWidgetItem
QVBoxLayout, QWidget = QtWidgets.QVBoxLayout, QtWidgets.QWidget

if QT_BINDING == "PyQt5":
    # PyQt6 groups enum values into nested classes.  Keep the rest of this
    # module binding-neutral by exposing the same names when running PyQt5.
    class _QtCompat:
        AlignmentFlag = SimpleNamespace(AlignCenter=Qt.AlignCenter, AlignLeft=Qt.AlignLeft,
                                        AlignRight=Qt.AlignRight, AlignTop=Qt.AlignTop,
                                        AlignVCenter=Qt.AlignVCenter)
        CheckState = SimpleNamespace(Checked=Qt.Checked, Unchecked=Qt.Unchecked)
        FindChildOption = SimpleNamespace(FindDirectChildrenOnly=Qt.FindDirectChildrenOnly)
        ItemDataRole = SimpleNamespace(UserRole=Qt.UserRole)
        ItemFlag = SimpleNamespace(
            ItemIsEnabled=Qt.ItemIsEnabled, ItemIsUserCheckable=Qt.ItemIsUserCheckable,
            ItemIsEditable=Qt.ItemIsEditable, ItemIsSelectable=Qt.ItemIsSelectable,
        )
        MouseButton = SimpleNamespace(LeftButton=Qt.LeftButton)
        Orientation = SimpleNamespace(Horizontal=Qt.Horizontal)
        PenStyle = SimpleNamespace(DashLine=Qt.DashLine, SolidLine=Qt.SolidLine)
        ScrollBarPolicy = SimpleNamespace(ScrollBarAlwaysOff=Qt.ScrollBarAlwaysOff)
        TextInteractionFlag = SimpleNamespace(TextSelectableByMouse=Qt.TextSelectableByMouse)
        WidgetAttribute = SimpleNamespace(WA_DeleteOnClose=Qt.WA_DeleteOnClose)
        WindowModality = SimpleNamespace(ApplicationModal=Qt.ApplicationModal)
    Qt = _QtCompat

    class QEvent(QtCore.QEvent):
        Type = SimpleNamespace(Wheel=QtCore.QEvent.Wheel)

    class QDialog(QtWidgets.QDialog):
        DialogCode = SimpleNamespace(Accepted=QtWidgets.QDialog.Accepted)
        def exec(self):
            return self.exec_()

    class QApplication(QtWidgets.QApplication):
        def exec(self):
            return self.exec_()

    class QFrame(QtWidgets.QFrame):
        Shape = SimpleNamespace(NoFrame=QtWidgets.QFrame.NoFrame, StyledPanel=QtWidgets.QFrame.StyledPanel)

    class QSizePolicy(QtWidgets.QSizePolicy):
        Policy = SimpleNamespace(Expanding=QtWidgets.QSizePolicy.Expanding, Fixed=QtWidgets.QSizePolicy.Fixed,
                                 Ignored=QtWidgets.QSizePolicy.Ignored, Maximum=QtWidgets.QSizePolicy.Maximum,
                                 Preferred=QtWidgets.QSizePolicy.Preferred)

    class QMessageBox(QtWidgets.QMessageBox):
        StandardButton = SimpleNamespace(Yes=QtWidgets.QMessageBox.Yes, No=QtWidgets.QMessageBox.No,
                                         Cancel=QtWidgets.QMessageBox.Cancel)
        ButtonRole = SimpleNamespace(AcceptRole=QtWidgets.QMessageBox.AcceptRole,
                                     ActionRole=QtWidgets.QMessageBox.ActionRole)
        Icon = SimpleNamespace(Warning=QtWidgets.QMessageBox.Warning)
from snr_gui import compute_batch_channel_snr, compute_resting_multiband_snr, RESTING_MULTIBAND_DEFINITIONS
from h5_provenance import (
    compact_integer_ranges, compact_provenance_operations, derive_h5_provenance,
    read_h5_provenance, write_h5_provenance,
)
from feishu_audit_sync import (
    FEISHU_FIELD_NAMES, FeishuAuditConfig, FeishuAuditSyncService,
)


class OperationAuditStore:
    """Process-safe audit table with exactly one evolving row per GUI session."""

    def __init__(self, path: str | Path, *, cloud_sync_enabled: bool = False):
        self.path = Path(path)
        self.cloud_sync_enabled = bool(cloud_sync_enabled)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.execute("PRAGMA synchronous=NORMAL")
            # Serialize schema migration because several users can launch the
            # shared GUI at the same time.
            database.execute("BEGIN IMMEDIATE")
            database.execute(
                """CREATE TABLE IF NOT EXISTS operation_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time TEXT NOT NULL,
                    operator_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    data_path TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    state_json TEXT NOT NULL DEFAULT '{}'
                )"""
            )
            database.execute(
                "CREATE INDEX IF NOT EXISTS idx_operation_audit_time ON operation_audit(event_time DESC)"
            )
            database.execute(
                "CREATE INDEX IF NOT EXISTS idx_operation_audit_operator ON operation_audit(operator_name, event_time DESC)"
            )
            existing_columns = {
                str(row[1]) for row in database.execute("PRAGMA table_info(operation_audit)").fetchall()
            }
            sync_columns = {
                "feishu_record_id": "TEXT NOT NULL DEFAULT ''",
                "sync_status": "TEXT NOT NULL DEFAULT 'not_configured'",
                "last_sync_time": "TEXT NOT NULL DEFAULT ''",
                "sync_error": "TEXT NOT NULL DEFAULT ''",
                "sync_attempts": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, declaration in sync_columns.items():
                if column not in existing_columns:
                    database.execute(
                        f"ALTER TABLE operation_audit ADD COLUMN {column} {declaration}"
                    )
            database.execute(
                """CREATE TABLE IF NOT EXISTS operation_audit_sync_deletions (
                    session_id TEXT PRIMARY KEY,
                    feishu_record_id TEXT NOT NULL,
                    requested_time TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT ''
                )"""
            )
            self._merge_legacy_session_rows(database)
            database.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_operation_audit_session ON operation_audit(session_id)"
            )
            database.commit()

    @staticmethod
    def _event(time_value, action, data_path, summary, state):
        return {
            "time": str(time_value), "action": str(action), "data_path": str(data_path or ""),
            "summary": str(summary or ""), "state": state if isinstance(state, dict) else {},
        }

    @classmethod
    def _decode_events(cls, row) -> list[dict]:
        row_id, event_time, _operator, _session, action, data_path, summary, raw_state = row
        try:
            document = json.loads(str(raw_state or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            document = {}
        if isinstance(document, dict) and isinstance(document.get("events"), list):
            return [event for event in document["events"] if isinstance(event, dict)]
        return [cls._event(event_time, action, data_path, summary, document)]

    @staticmethod
    def _summary_from_events(events: list[dict]) -> str:
        lines = []
        for event in events:
            stamp = str(event.get("time", ""))[11:19]
            action = str(event.get("action", ""))
            summary = str(event.get("summary", ""))
            lines.append(f"[{stamp}] {action}" + (f"：{summary}" if summary else ""))
        return "\n".join(lines)

    @classmethod
    def _merge_legacy_session_rows(cls, database) -> None:
        """Collapse rows written by the former event-per-row schema without data loss."""
        rows = database.execute(
            """SELECT id, event_time, operator_name, session_id, action, data_path, summary, state_json
               FROM operation_audit ORDER BY id"""
        ).fetchall()
        sessions = {}
        for row in rows:
            sessions.setdefault(str(row[3]), []).append(row)
        for session_rows in sessions.values():
            if len(session_rows) <= 1:
                continue
            events = []
            for row in session_rows:
                events.extend(cls._decode_events(row))
            latest = session_rows[-1]
            document = {
                "schema": "sd-gui-session-audit", "version": 1,
                "started_time": events[0].get("time", latest[1]) if events else latest[1],
                "updated_time": events[-1].get("time", latest[1]) if events else latest[1],
                "latest_state": events[-1].get("state", {}) if events else {},
                "events": events,
            }
            database.execute(
                """UPDATE operation_audit SET event_time=?, operator_name=?, action=?, data_path=?,
                   summary=?, state_json=? WHERE id=?""",
                (latest[1], latest[2], latest[4], latest[5], cls._summary_from_events(events),
                 json.dumps(document, ensure_ascii=False, sort_keys=True, default=str), latest[0]),
            )
            database.executemany(
                "DELETE FROM operation_audit WHERE id=?", [(row[0],) for row in session_rows[:-1]]
            )

    def _connect(self):
        database = sqlite3.connect(str(self.path), timeout=30.0)
        database.execute("PRAGMA busy_timeout=30000")
        return database

    def append(self, operator_name: str, session_id: str, action: str,
               data_path: str = "", summary: str = "", state=None) -> None:
        now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        state = state if isinstance(state, dict) else {}
        with closing(self._connect()) as database:
            database.execute("BEGIN IMMEDIATE")
            existing = database.execute(
                """SELECT id, event_time, operator_name, session_id, action, data_path, summary, state_json
                   FROM operation_audit WHERE session_id=?""", (str(session_id),)
            ).fetchone()
            events = self._decode_events(existing) if existing else []
            events.append(self._event(now, action, data_path, summary, state))
            document = {
                "schema": "sd-gui-session-audit", "version": 1,
                "started_time": events[0].get("time", now), "updated_time": now,
                "closed_time": now if action == "session_closed" else "",
                "latest_state": state, "events": events,
            }
            payload = json.dumps(document, ensure_ascii=False, sort_keys=True, default=str)
            combined_summary = self._summary_from_events(events)
            if existing:
                database.execute(
                    """UPDATE operation_audit SET event_time=?, operator_name=?, action=?, data_path=?,
                       summary=?, state_json=?, sync_status=?, sync_error='' WHERE session_id=?""",
                    (now, str(operator_name), str(action), str(data_path or existing[5]),
                     combined_summary, payload,
                     "pending" if self.cloud_sync_enabled else "not_configured", str(session_id)),
                )
            else:
                database.execute(
                    """INSERT INTO operation_audit
                       (event_time, operator_name, session_id, action, data_path, summary, state_json, sync_status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (now, str(operator_name), str(session_id), str(action), str(data_path),
                     combined_summary, payload,
                     "pending" if self.cloud_sync_enabled else "not_configured"),
                )
            database.commit()

    def discard_session(self, session_id: str) -> None:
        """Delete only the unsaved audit row belonging to this GUI launch."""
        with closing(self._connect()) as database:
            database.execute("BEGIN IMMEDIATE")
            row = database.execute(
                "SELECT feishu_record_id FROM operation_audit WHERE session_id=?",
                (str(session_id),),
            ).fetchone()
            if row and str(row[0] or ""):
                database.execute(
                    """INSERT INTO operation_audit_sync_deletions
                       (session_id, feishu_record_id, requested_time, attempts, error)
                       VALUES (?, ?, ?, 0, '')
                       ON CONFLICT(session_id) DO UPDATE SET
                         feishu_record_id=excluded.feishu_record_id,
                         requested_time=excluded.requested_time""",
                    (
                        str(session_id), str(row[0]),
                        datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                    ),
                )
            database.execute(
                "DELETE FROM operation_audit WHERE session_id=?", (str(session_id),)
            )
            database.commit()

    def latest(self, limit: int = 2000) -> list[tuple]:
        with closing(self._connect()) as database:
            return database.execute(
                """SELECT event_time, operator_name, action, data_path, summary, state_json,
                          sync_status, last_sync_time, sync_error
                   FROM operation_audit ORDER BY event_time DESC, id DESC LIMIT ?""",
                (max(1, int(limit)),),
            ).fetchall()

    def sync_counts(self) -> dict[str, int]:
        with closing(self._connect()) as database:
            rows = database.execute(
                "SELECT sync_status, COUNT(*) FROM operation_audit GROUP BY sync_status"
            ).fetchall()
            result = {str(status): int(count) for status, count in rows}
            result["pending_deletions"] = int(database.execute(
                "SELECT COUNT(*) FROM operation_audit_sync_deletions"
            ).fetchone()[0])
            return result


class DisableValueControlWheelFilter(QObject):
    """Prevent the mouse wheel from accidentally changing form values.

    Wheel input is reserved for scrolling pages/plots. Spin boxes, sliders,
    and combo boxes must be changed deliberately with the keyboard, buttons,
    dragging, or an opened selector.
    """

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Wheel and isinstance(
            watched, (QAbstractSpinBox, QAbstractSlider, QComboBox)
        ):
            return True
        return super().eventFilter(watched, event)


class PreprocessProgressBar(QWidget):
    """Progress row with a left-aligned bar and separate task controls."""

    pause_clicked = pyqtSignal()
    cancel_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(42)
        controls = QHBoxLayout(self)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(10)
        self.bar = QProgressBar(self)
        self.bar.setMinimumHeight(34)
        self.bar.setTextVisible(True)
        controls.addWidget(self.bar, 1)
        self.pause_button = QPushButton("暂停", self)
        self.cancel_button = QPushButton("取消", self)
        standard_pixmap = getattr(QtWidgets.QStyle, "StandardPixmap", QtWidgets.QStyle)
        self.pause_button.setIcon(
            self.style().standardIcon(standard_pixmap.SP_MediaPause)
        )
        self.cancel_button.setIcon(
            self.style().standardIcon(standard_pixmap.SP_DialogCancelButton)
        )
        for button in (self.pause_button, self.cancel_button):
            button.setMinimumHeight(32)
            button.setMaximumHeight(34)
            button.setMinimumWidth(88)
            button.setMaximumWidth(108)
            button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            button.setEnabled(False)
            button.setStyleSheet(
                "QPushButton { background:rgba(255,255,255,235); color:#17324d;"
                " border:1px solid #8da8c1; border-radius:7px; padding:1px 8px; }"
                "QPushButton:hover { background:#ffffff; border-color:#377ba8; }"
                "QPushButton:disabled { color:#8996a3; background:rgba(245,247,250,190); }"
            )
        controls.addWidget(self.pause_button)
        controls.addWidget(self.cancel_button)
        self.pause_button.clicked.connect(self.pause_clicked)
        self.cancel_button.clicked.connect(self.cancel_clicked)

    def setObjectName(self, name: str) -> None:
        super().setObjectName(name + "Row")
        self.bar.setObjectName(name)

    def setValue(self, value: int) -> None:
        self.bar.setValue(int(value))

    def value(self) -> int:
        return self.bar.value()

    def setRange(self, minimum: int, maximum: int) -> None:
        self.bar.setRange(int(minimum), int(maximum))

    def minimum(self) -> int:
        return self.bar.minimum()

    def maximum(self) -> int:
        return self.bar.maximum()

    def set_running(self, running: bool, paused: bool = False) -> None:
        self.pause_button.setEnabled(bool(running))
        self.cancel_button.setEnabled(bool(running))
        self.pause_button.setText("继续" if paused else "暂停")
        standard_pixmap = getattr(QtWidgets.QStyle, "StandardPixmap", QtWidgets.QStyle)
        self.pause_button.setIcon(self.style().standardIcon(
            standard_pixmap.SP_MediaPlay
            if paused else standard_pixmap.SP_MediaPause
        ))


ANALYSIS_DATA_CACHE_BYTES = 512 * 1024 * 1024


class AnalysisCache:
    """Bounded source-data cache plus lightweight reusable analysis metrics."""

    def __init__(self, data_budget_bytes: int = ANALYSIS_DATA_CACHE_BYTES) -> None:
        self.data_budget_bytes = int(data_budget_bytes)
        self._data_bytes = 0
        self._channel_data: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._saturation_metrics: dict[tuple, dict] = {}
        self._results: OrderedDict[tuple, dict] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def source_key(source) -> tuple:
        meta = source.metadata
        return (
            id(source), id(meta), int(meta.rows), int(meta.channels), float(meta.fs),
            tuple(int(channel) for channel in meta.channel_ids),
        )

    def get_channel_data(self, source, columns) -> np.ndarray:
        columns = np.unique(np.asarray(columns, dtype=np.int64).ravel())
        key = (self.source_key(source), tuple(int(column) for column in columns))
        with self._lock:
            cached = self._channel_data.get(key)
            if cached is not None:
                self._channel_data.move_to_end(key)
                return cached
        if hasattr(source, "data"):
            values = np.asarray(source.data[:, columns], dtype=np.float32)
        else:
            values = np.asarray(source.read(0, source.metadata.rows, columns), dtype=np.float32)
        values = np.ascontiguousarray(values)
        if values.nbytes <= self.data_budget_bytes:
            with self._lock:
                while self._channel_data and self._data_bytes + values.nbytes > self.data_budget_bytes:
                    _, evicted = self._channel_data.popitem(last=False)
                    self._data_bytes -= evicted.nbytes
                self._channel_data[key] = values
                self._data_bytes += values.nbytes
        return values

    def get_saturation_metric(self, source, channel_id: int, width_percent: float):
        with self._lock:
            value = self._saturation_metrics.get(
                (self.source_key(source), int(channel_id), float(width_percent))
            )
            return None if value is None else dict(value)

    def put_saturation_metric(self, source, channel_id: int, width_percent: float, value: dict) -> None:
        with self._lock:
            self._saturation_metrics[
                (self.source_key(source), int(channel_id), float(width_percent))
            ] = dict(value)

    def get_result(self, key):
        with self._lock:
            value = self._results.get(key)
            if value is not None:
                self._results.move_to_end(key)
            return None if value is None else deepcopy(value)

    def put_result(self, key, value: dict) -> None:
        with self._lock:
            self._results[key] = deepcopy(value)
            self._results.move_to_end(key)
            while len(self._results) > 24:
                self._results.popitem(last=False)


def _export_chunk_rows(meta, target_bytes: int = 8 * 1024 * 1024) -> int:
    """Keep HDF5 row chunks useful for both streaming and random previews."""
    bytes_per_row = max(1, int(meta.channels) * np.dtype(np.float32).itemsize)
    by_size = max(1024, target_bytes // bytes_per_row)
    return min(meta.rows, max(1, min(int(round(meta.fs * 10)), by_size)))


def _legacy_h5_compression_kwargs() -> dict:
    """The exact compressed storage contract used by the legacy GUI."""
    return hdf5plugin.Blosc(
        cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE
    )


def _write_legacy_compression_metadata(h5: h5py.File) -> None:
    h5.attrs["compression"] = "Blosc:LZ4+bitshuffle"
    h5.attrs["compression_filter"] = "blosc:lz4"
    h5.attrs["compression_level"] = 5
    h5.attrs["compression_shuffle"] = "bitshuffle"
    h5.attrs["hdf5_compression"] = "Blosc/LZ4 + bitshuffle"
    h5.attrs["hdf5_compression_filter"] = "blosc:lz4"
    h5.attrs["hdf5_compression_level"] = 5
    h5.attrs["hdf5_compression_shuffle"] = "bitshuffle"


def _write_timing_metadata(h5: h5py.File, timing: dict[str, object] | None) -> None:
    """Persist BIN timing in the legacy reloadable HDF5 layout.

    Keep the original deltaT1 value and its unit together.  The normalized
    seconds/milliseconds attributes are redundant by design and make files
    written by this Qt GUI unambiguous to both old and new readers.
    """
    if not timing:
        return
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for key in ("dt1", "dt2", "dt1_date"):
        value = timing.get(key)
        if value not in (None, ""):
            h5.create_dataset(key, data=str(value), dtype=string_dtype)
    for key in ("deltaT1", "deltaT2"):
        value = timing.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            h5.create_dataset(key, data=np.float64(numeric))
    unit = str(timing.get("deltaT1_unit", "")).strip()
    if unit:
        h5.create_dataset("deltaT1_unit", data=unit, dtype=string_dtype)
    for key in ("deltaT1_sec", "deltaT1_ms", "deltaT2_sec", "deltaT2_ms"):
        value = timing.get(key)
        if value is not None and np.isfinite(float(value)):
            h5.attrs[key] = float(value)
    if "deltaT1_sec" in timing:
        h5.attrs["bin_delta_t1_sec"] = float(timing["deltaT1_sec"])
    if "deltaT2_sec" in timing:
        h5.attrs["bin_delta_t2_sec"] = float(timing["deltaT2_sec"])


def _write_output_provenance(
    h5: h5py.File,
    meta,
    *,
    stage: str,
    channel_ids,
    operation: dict,
) -> None:
    """Embed a derived provenance document in every GUI-written HDF5 file."""
    parent = getattr(meta, "provenance", None)
    source = None if parent is not None else {
        "path": str(meta.path), "dataset": str(meta.dataset), "time_offset_sec": float(meta.time_offset),
    }
    document = derive_h5_provenance(
        parent,
        stage=stage,
        fs=float(meta.fs),
        unit="mV",
        channel_ids=channel_ids,
        source=source,
        operation=operation,
    )
    write_h5_provenance(h5, document)


def _provenance_stage(meta, fallback: str) -> str:
    document = getattr(meta, "provenance", None) or {}
    return str(document.get("data", {}).get("stage") or fallback)


def replace_preprocessed_channels(
    raw_path, processed_path, output_path, target_channel_ids, *, mapping_path=None,
    mode="bandpass", low=0.1, high=300.0, highpass_order=3, lowpass_order=5,
    notch=True, notch_frequency=50.0, notch_q=30.0, notch_harmonics=1,
    core_seconds=600.0, overlap_seconds=60.0,
    checkpoint=None, progress=None,
) -> dict:
    """Reprocess raw source channels and replace matching processed columns.

    The destination is always a copy; the existing processed file is never
    modified. Physical-to-destination remapping is resolved before any data
    are written. Chunk overlap is cropped after zero-phase filtering.
    """
    raw_path, processed_path, output_path = map(Path, (raw_path, processed_path, output_path))
    checkpoint = checkpoint if callable(checkpoint) else (lambda: None)
    progress = progress if callable(progress) else (lambda _value, _message: None)
    created_output = False
    checkpoint()
    if raw_path.resolve() == processed_path.resolve() or output_path.resolve() in {
        raw_path.resolve(), processed_path.resolve()
    }:
        raise ValueError("原始、现有预处理和修正版输出必须是三个不同文件。")
    targets = sorted({int(value) for value in target_channel_ids})
    if not targets:
        raise ValueError("至少需要一个目标通道。")
    raw_source = LazyH5Source()
    raw_meta = raw_source.open(raw_path)
    try:
        with h5py.File(processed_path, "r") as current:
            if "rawData512" not in current or "channel_ids" not in current:
                raise ValueError("现有预处理文件缺少 rawData512 或 channel_ids。")
            processed_shape = current["rawData512"].shape
            processed_ids = np.asarray(current["channel_ids"], dtype=np.int64).ravel()
            processed_fs = float(np.asarray(current["FS"]).squeeze())
            if "time_offset_sec" in current:
                processed_offset = float(np.asarray(current["time_offset_sec"][()]).squeeze())
            else:
                processed_offset = float(current.attrs.get("time_offset_sec", 0.0))
            provenance = read_h5_provenance(current) or {}
        if processed_ids.size != processed_shape[1] or np.unique(processed_ids).size != processed_ids.size:
            raise ValueError("预处理文件的 channel_ids 与数据列不一致或存在重复。")
        if (processed_shape[0] != raw_meta.rows or not np.isclose(processed_fs, raw_meta.fs)
                or not np.isclose(processed_offset, raw_meta.time_offset)):
            raise ValueError("原始文件与预处理文件的采样点数、采样率或时间偏移不一致。")
        operations = provenance.get("operations", []) if isinstance(provenance, dict) else []
        operation_names = {str(item.get("name", "")) for item in operations if isinstance(item, dict)}
        unsafe = operation_names.intersection({"ica", "reference"})
        if unsafe:
            raise ValueError("文件包含 ICA/CAR 多通道处理，不能进行独立通道平替：" + ", ".join(sorted(unsafe)))
        remapped = "channel_remap" in operation_names
        raw_ids = np.asarray(raw_meta.channel_ids, dtype=np.int64)
        if remapped:
            remap_operation = next(
                (item for item in reversed(operations)
                 if isinstance(item, dict) and item.get("name") == "channel_remap"), {}
            )
            embedded_sources = np.asarray(remap_operation.get("source_channel_ids", []), dtype=np.int64)
            embedded_destinations = np.asarray(
                remap_operation.get("destination_for_source_channel_ids", []), dtype=np.int64
            )
            if (embedded_sources.size == raw_ids.size == embedded_destinations.size
                    and np.array_equal(embedded_sources, raw_ids)):
                destinations = embedded_destinations
            elif mapping_path:
                destinations = read_channel_remap_for_ids(mapping_path, raw_ids)
            else:
                raise ValueError("检测到通道重映射，但文件未内嵌完整映射；必须提供原处理使用的映射 Excel。")
        else:
            destinations = raw_ids.copy()
        raw_columns_by_destination = {}
        for raw_column, destination_id in enumerate(destinations):
            destination_id = int(destination_id)
            if destination_id in raw_columns_by_destination:
                raw_columns_by_destination[destination_id] = None
            else:
                raw_columns_by_destination[destination_id] = raw_column
        processed_columns = {int(channel): index for index, channel in enumerate(processed_ids)}
        resolved = []
        for target in targets:
            raw_column = raw_columns_by_destination.get(target)
            if raw_column is None or target not in raw_columns_by_destination:
                raise ValueError(f"目标 ch{target} 无法唯一反查到一个原始通道。")
            if target not in processed_columns:
                raise ValueError(f"预处理文件中不存在目标 ch{target}。")
            resolved.append((target, int(raw_column), int(processed_columns[target])))
        if output_path.exists():
            raise FileExistsError(f"修正版输出已存在：{output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        progress(5.0, "正在复制当前预处理文件……")
        shutil.copy2(processed_path, output_path)
        created_output = True
        checkpoint()
        settings = {
            "mode": str(mode), "low": float(low), "high": float(high),
            "hp_order": int(highpass_order), "lp_order": int(lowpass_order),
            "notch": bool(notch), "notch_frequency": float(notch_frequency),
            "notch_q": float(notch_q), "notch_harmonics": int(notch_harmonics),
        }
        core_rows = max(1, int(round(float(core_seconds) * raw_meta.fs)))
        overlap_rows = max(0, int(round(float(overlap_seconds) * raw_meta.fs)))
        with h5py.File(output_path, "r+") as corrected:
            target_data = corrected["rawData512"]
            total_blocks = max(1, len(resolved) * int(np.ceil(raw_meta.rows / core_rows)))
            completed_blocks = 0
            for target, raw_column, processed_column in resolved:
                for first in range(0, raw_meta.rows, core_rows):
                    checkpoint()
                    last = min(raw_meta.rows, first + core_rows)
                    read_first = max(0, first - overlap_rows)
                    read_last = min(raw_meta.rows, last + overlap_rows)
                    values = raw_source.read(read_first, read_last, raw_column)
                    filtered = FilterOverlapTestWorker._filter(values, raw_meta.fs, settings)
                    target_data[first:last, processed_column] = np.asarray(
                        filtered[first - read_first:last - read_first], dtype=np.float32
                    )
                    completed_blocks += 1
                    progress(
                        5.0 + 90.0 * completed_blocks / total_blocks,
                        f"正在重算并替换 ch{target}：{completed_blocks}/{total_blocks} 数据块",
                    )
            checkpoint()
            replacement_operation = {
                "name": "channel_replacement", "channel_ids": targets,
                "source_file": str(raw_path), "mapping_file": str(mapping_path or ""),
                "remapped": remapped, "settings": settings,
                "resolved_channels": [
                    {"target_channel_id": t, "source_channel_id": int(raw_ids[r]),
                     "source_column": r, "destination_column": d}
                    for t, r, d in resolved
                ],
            }
            updated = derive_h5_provenance(
                provenance or None, stage="preprocessed", fs=raw_meta.fs, unit="mV",
                channel_ids=processed_ids, operation=replacement_operation,
            )
            write_h5_provenance(corrected, updated)
            corrected.attrs["channel_replacement_ids"] = np.asarray(targets, dtype=np.int64)
        progress(100.0, "通道替换完成")
        return {"output": str(output_path), "remapped": remapped, "resolved": resolved}
    except Exception:
        if created_output and output_path.exists():
            output_path.unlink()
        raise
    finally:
        # LazyH5Source opens files only for individual reads and owns no
        # persistent HDF5 handle.
        pass


def extract_original_channel_files(raw_path, processed_path, target_channel_ids, output_directory,
                                   *, mapping_path=None) -> list[str]:
    """Export untouched raw signals selected by processed/destination IDs."""
    raw_path, processed_path = Path(raw_path), Path(processed_path)
    output_directory = Path(output_directory)
    source = LazyH5Source(); meta = source.open(raw_path)
    processed_source = LazyH5Source(); processed_meta = processed_source.open(processed_path)
    with h5py.File(processed_path, "r") as h5:
        processed_dataset = h5[processed_meta.dataset]
        if processed_dataset.ndim != 2:
            raise ValueError(
                f"预处理 H5 的数据集 {processed_meta.dataset} 必须是二维的“采样点 × 通道”矩阵。"
            )
        processed_ids = np.asarray(processed_meta.channel_ids, dtype=np.int64)
        processed_rows = int(processed_dataset.shape[0])
        processed_fs = float(processed_meta.fs)
        provenance = read_h5_provenance(h5) or {}
    if processed_rows != meta.rows or not np.isclose(processed_fs, meta.fs):
        raise ValueError("原始文件与预处理文件的采样点数或采样率不一致。")
    operations = provenance.get("operations", []) if isinstance(provenance, dict) else []
    remap_op = next((item for item in reversed(operations)
                     if isinstance(item, dict) and item.get("name") == "channel_remap"), None)
    raw_ids = np.asarray(meta.channel_ids, dtype=np.int64)
    if remap_op is None:
        destinations = raw_ids
    else:
        embedded_sources = np.asarray(remap_op.get("source_channel_ids", []), dtype=np.int64)
        embedded_destinations = np.asarray(remap_op.get("destination_for_source_channel_ids", []), dtype=np.int64)
        if (embedded_sources.size == raw_ids.size == embedded_destinations.size
                and np.array_equal(embedded_sources, raw_ids)):
            destinations = embedded_destinations
        elif mapping_path:
            destinations = read_channel_remap_for_ids(mapping_path, raw_ids)
        else:
            raise ValueError("检测到重映射但没有完整内嵌映射，请选择原映射 Excel。")
    lookup = {}
    for column, destination in enumerate(destinations):
        destination = int(destination)
        lookup[destination] = None if destination in lookup else column
    targets = sorted({int(value) for value in target_channel_ids})
    if not set(targets).issubset(set(map(int, processed_ids))):
        raise ValueError("输入的目标通道不在预处理文件中。")
    output_directory.mkdir(parents=True, exist_ok=True)
    chunk_rows = _export_chunk_rows(meta)
    written = []
    for target in targets:
        column = lookup.get(target)
        if column is None:
            raise ValueError(f"目标 ch{target} 无法唯一反查原始物理通道。")
        source_id = int(raw_ids[column])
        path = output_directory / f".{processed_path.stem}_repair_target_ch{target}_raw_ch{source_id}.h5"
        sequence = 2
        while path.exists():
            path = output_directory / (
                f".{processed_path.stem}_repair_target_ch{target}_raw_ch{source_id}_{sequence}.h5"
            )
            sequence += 1
        with h5py.File(path, "w") as h5:
            # Keep the standard two-dimensional layout so this untouched
            # channel can immediately be imported and previewed by the GUI.
            signal = h5.create_dataset("rawData512", shape=(meta.rows, 1), dtype=np.float32,
                                       chunks=(chunk_rows, 1), **_legacy_h5_compression_kwargs())
            for first in range(0, meta.rows, chunk_rows):
                last = min(meta.rows, first + chunk_rows)
                signal[first:last, 0] = source.read(first, last, int(column))
            h5["FS"] = meta.fs; h5["data_unit"] = "mV"
            h5["channel_ids"] = np.asarray([target], dtype=np.int64)
            h5["target_channel_id"] = np.asarray(target, dtype=np.int64)
            h5["source_physical_channel_id"] = np.asarray(source_id, dtype=np.int64)
            h5["source_column_index"] = np.asarray(column, dtype=np.int64)
            h5.attrs["time_offset_sec"] = float(meta.time_offset)
            h5.attrs["data_stage"] = "raw_channel_extracted"
            h5.attrs["channel_was_remapped"] = bool(remap_op is not None)
        written.append(str(path))
    return written


def merge_repaired_channel_file(processed_path, repaired_channel_path, output_path,
                                *, target_channel_id=None) -> dict:
    """Insert a user-processed single channel into a copied full H5."""
    processed_path, repaired_channel_path, output_path = map(
        Path, (processed_path, repaired_channel_path, output_path)
    )
    if output_path.resolve() in {processed_path.resolve(), repaired_channel_path.resolve()}:
        raise ValueError("修正版必须另存为新文件，不能覆盖输入文件。")
    with h5py.File(repaired_channel_path, "r") as patch:
        key = "signal" if "signal" in patch else "rawData512" if "rawData512" in patch else None
        if key is None:
            raise ValueError("处理好的通道文件缺少 signal 或 rawData512。")
        patch_data = patch[key]
        if patch_data.ndim == 2 and patch_data.shape[1] != 1:
            raise ValueError("处理好的文件必须只包含一个通道。")
        patch_rows = patch_data.shape[0]
        patch_fs = float(np.asarray(patch["FS"]).squeeze())
        patch_provenance = read_h5_provenance(patch) or {}
        stored_target = patch.get("target_channel_id")
        if target_channel_id is None and stored_target is not None:
            target_channel_id = int(np.asarray(stored_target[()]).squeeze())
        if target_channel_id is None:
            raise ValueError("处理好的文件没有目标通道信息，请手动指定目标通道。")
    with h5py.File(processed_path, "r") as base:
        ids = np.asarray(base["channel_ids"], dtype=np.int64).ravel()
        rows = base["rawData512"].shape[0]
        fs = float(np.asarray(base["FS"]).squeeze())
        provenance = read_h5_provenance(base) or {}
    matches = np.flatnonzero(ids == int(target_channel_id))
    if matches.size != 1 or rows != patch_rows or not np.isclose(fs, patch_fs):
        raise ValueError("目标通道不唯一，或处理好通道的长度/采样率与预处理文件不一致。")
    if output_path.exists():
        raise FileExistsError(f"修正版输出已存在：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(processed_path, output_path)
    try:
        def operation_channel_ids(value) -> set[int]:
            if isinstance(value, (list, tuple, np.ndarray)):
                return {int(item) for item in value}
            result = set()
            for part in str(value or "").split(","):
                part = part.strip()
                if not part: continue
                if "-" in part:
                    first, last = (int(item) for item in part.split("-", 1))
                    result.update(range(min(first, last), max(first, last) + 1))
                else:
                    result.add(int(part))
            return result
        patch_processing_operations = []
        for item in patch_provenance.get("operations", []):
            if not isinstance(item, dict) or str(item.get("name", "")).lower() not in {"filter", "reference", "ica"}:
                continue
            scope = operation_channel_ids(item.get("channel_ids", []))
            if not scope or int(target_channel_id) in scope:
                patch_processing_operations.append(item)
        with h5py.File(repaired_channel_path, "r") as patch, h5py.File(output_path, "r+") as result:
            source = patch["signal"] if "signal" in patch else patch["rawData512"]
            destination = result["rawData512"]
            chunk_rows = max(1, int(destination.chunks[0] if destination.chunks else 100000))
            for first in range(0, rows, chunk_rows):
                last = min(rows, first + chunk_rows)
                values = source[first:last]
                destination[first:last, int(matches[0])] = np.asarray(values).reshape(-1)
            updated = derive_h5_provenance(
                provenance or None, stage="preprocessed", fs=fs, unit="mV", channel_ids=ids,
                operation={"name": "merge_repaired_channel", "channel_ids": [int(target_channel_id)],
                           "channel_file": str(repaired_channel_path),
                           "channel_processing_operations": patch_processing_operations},
            )
            write_h5_provenance(result, updated)
            result.attrs["last_merged_channel_id"] = int(target_channel_id)
            raw_state = result.attrs.get("preprocess_state_json", "")
            if isinstance(raw_state, bytes):
                raw_state = raw_state.decode("utf-8", errors="replace")
            try:
                state = json.loads(str(raw_state)) if raw_state else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                state = {}
            if not isinstance(state, dict):
                state = {}
            replacements = {int(value) for value in state.get("replacement_channel_ids", [])}
            replacements.add(int(target_channel_id))
            operation_names = {str(value) for value in state.get("operation_names", [])}
            operation_names.add("merge_repaired_channel")
            channel_processing = state.get("channel_processing", {})
            if not isinstance(channel_processing, dict):
                channel_processing = {}
            channel_processing[str(int(target_channel_id))] = patch_processing_operations
            state.update({
                "schema": "sd-preprocess-state", "version": 1,
                "data_stage": "preprocessed",
                "replacement_channel_ids": sorted(replacements),
                "operation_names": sorted(operation_names),
                "channel_processing": channel_processing,
            })
            result.attrs["preprocess_state_schema"] = "sd-preprocess-state"
            result.attrs["preprocess_state_version"] = 1
            result.attrs["preprocess_state_json"] = json.dumps(
                state, ensure_ascii=False, sort_keys=True
            )
        return {"output": str(output_path), "channel_id": int(target_channel_id)}
    except Exception:
        if output_path.exists(): output_path.unlink()
        raise


def _mat_timing_metadata(values: dict) -> dict[str, object]:
    """MAT counterpart of the old Tk deltaT1 unit-compatibility rule."""
    timing: dict[str, object] = {}
    for key in ("dt1", "dt2", "dt1_date"):
        if key not in values:
            continue
        value = np.asarray(values[key]).squeeze()
        if isinstance(value, np.ndarray):
            value = "".join(str(part) for part in value.ravel())
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        timing[key] = str(value).strip()
    for key in ("deltaT1", "deltaT2"):
        if key in values:
            try:
                timing[key] = float(np.asarray(values[key]).squeeze())
            except (TypeError, ValueError):
                pass
    if "deltaT1_unit" in values:
        unit = np.asarray(values["deltaT1_unit"]).squeeze()
        if isinstance(unit, np.ndarray):
            unit = "".join(str(part) for part in unit.ravel())
        if isinstance(unit, bytes):
            unit = unit.decode("utf-8", errors="replace")
        timing["deltaT1_unit"] = str(unit).strip()
    raw_t1 = float(timing.get("deltaT1", 0.0))
    # Raw MAT/HDF5 deltaT1 mirrors the BIN counter field and is in ms.
    # ``bin_delta_t1_sec`` is the separate, explicitly seconds-valued field.
    is_ms = "deltaT1" in timing
    if is_ms:
        timing["deltaT1_unit"] = "ms"
    if is_ms and abs(raw_t1) >= 10_000.0:
        raw_t1 /= 1000.0
        timing["deltaT1"] = raw_t1
        timing["timing_legacy_ms_scale_repaired"] = True
    timing["deltaT1_sec"] = raw_t1 / 1000.0 if is_ms else raw_t1
    timing["deltaT1_ms"] = timing["deltaT1_sec"] * 1000.0
    if "deltaT2" in timing:
        raw_t2 = float(timing["deltaT2"])
        if is_ms and abs(raw_t2) >= 10_000.0:
            raw_t2 /= 1000.0
            timing["deltaT2"] = raw_t2
        timing["deltaT2_sec"] = raw_t2 / 1000.0 if is_ms else raw_t2
        timing["deltaT2_ms"] = timing["deltaT2_sec"] * 1000.0
    return timing


class _PyQtGraphAxis:
    """Small compatibility facade for the old axis calls, backed by PyQtGraph."""

    def __init__(self, owner):
        self.owner = owner

    @staticmethod
    def _color_with_alpha(color, alpha=1.0):
        value = pg.mkColor(color)
        value.setAlphaF(max(0.0, min(1.0, float(alpha))))
        return value

    def plot(self, x, y, color="#1f77b4", linewidth=1.0, antialiased=True, marker=None, markersize=None, label=None, **kwargs):
        pen = pg.mkPen(color=color, width=max(1, int(round(float(linewidth) * 1.5))))
        opts = {"pen": pen, "antialias": bool(antialiased)}
        if marker:
            opts.update(symbol="o", symbolSize=max(3, int(round(float(markersize or 5)))), symbolBrush=color)
        if label:
            self.owner._legend = self.owner._legend or self.owner.plotItem.addLegend()
            opts["name"] = label
        if kwargs.get("stepMode"):
            opts["stepMode"] = kwargs["stepMode"]
        return self.owner.plot(x, y, **opts)

    def bar(self, x, height, width=.8, color="#1f77b4", **kwargs):
        item = pg.BarGraphItem(
            x=np.asarray(x, dtype=float), height=np.asarray(height, dtype=float),
            width=float(width), brush=color,
        )
        self.owner.addItem(item)
        return item

    def step(self, x, y, where="post", **kwargs):
        return self.plot(x, y, stepMode="left" if where == "post" else "center", **kwargs)

    def axvline(self, value, color="#c62828", alpha=1.0, linewidth=1.0, **kwargs):
        line = pg.InfiniteLine(pos=float(value), angle=90, movable=False,
                               pen=pg.mkPen(color=self._color_with_alpha(color, alpha), width=max(1, int(round(linewidth * 1.5))),
                                            style=Qt.PenStyle.DashLine if kwargs.get("linestyle") == "--" else Qt.PenStyle.SolidLine))
        self.owner.addItem(line)
        return line

    def axhline(self, value, color="#777", alpha=1.0, linewidth=1.0, **kwargs):
        line = pg.InfiniteLine(pos=float(value), angle=0, movable=False,
                               pen=pg.mkPen(color=self._color_with_alpha(color, alpha), width=max(1, int(round(linewidth * 1.5))),
                                            style=Qt.PenStyle.DashLine if kwargs.get("linestyle") == "--" else Qt.PenStyle.SolidLine))
        self.owner.addItem(line)
        return line

    def fill_between(self, x, lower, upper, color="#1f77b4", alpha=.2, linewidth=0, **kwargs):
        """PyQtGraph-native filled band bounded by two curves."""
        x = np.asarray(x, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        edge = None if float(linewidth) <= 0 else pg.mkPen(
            self._color_with_alpha(color, alpha), width=max(1, int(round(float(linewidth))))
        )
        lower_curve = pg.PlotCurveItem(x, lower, pen=edge)
        upper_curve = pg.PlotCurveItem(x, upper, pen=edge)
        band = pg.FillBetweenItem(
            upper_curve, lower_curve,
            brush=pg.mkBrush(self._color_with_alpha(color, alpha)),
        )
        self.owner.addItem(lower_curve); self.owner.addItem(upper_curve); self.owner.addItem(band)
        return band

    def axvspan(self, left, right, color="#2e7d32", alpha=.16, **kwargs):
        """PyQtGraph-native immutable vertical highlighted interval."""
        region = pg.LinearRegionItem(
            values=(float(left), float(right)), orientation="vertical", movable=False,
            brush=pg.mkBrush(self._color_with_alpha(color, alpha)), pen=pg.mkPen(None),
        )
        self.owner.addItem(region)
        return region

    def text(self, x, y, value, *, coordinates="data", color="#4a4a4a", **kwargs):
        """Add a PyQtGraph TextItem in data or normalized plot coordinates."""
        anchor_x = .5 if kwargs.get("ha") == "center" else (1.0 if kwargs.get("ha") == "right" else 0.0)
        anchor_y = .5 if kwargs.get("va") == "center" else (1.0 if kwargs.get("va") == "top" else 0.0)
        item = pg.TextItem(str(value), color=color, anchor=(anchor_x, anchor_y))
        self.owner.addItem(item)
        if coordinates == "axes":
            self.owner.setXRange(0.0, 1.0, padding=0)
            self.owner.setYRange(0.0, 1.0, padding=0)
        item.setPos(float(x), float(y))
        return item

    def set_axis_off(self):
        self.owner.plotItem.hideAxis("bottom")
        self.owner.plotItem.hideAxis("left")
        self.owner.showGrid(x=False, y=False)

    def set_xlabel(self, text): self.owner.setLabel("bottom", str(text))
    def set_ylabel(self, text): self.owner.setLabel("left", str(text))
    def set_title(self, text): self.owner.setTitle(str(text))
    def grid(self, alpha=0.25): self.owner.showGrid(x=True, y=True, alpha=float(alpha))
    def set_xlim(self, low, high): self.owner.setXRange(float(low), float(high), padding=0)
    def set_ylim(self, low, high): self.owner.setYRange(float(low), float(high), padding=0)
    def legend(self, *args, **kwargs):
        if self.owner._legend is None: self.owner._legend = self.owner.plotItem.addLegend()


class _PyQtGraphToolbar(QWidget):
    """Compact navigation bar; plot itself supports wheel zoom and dragging."""
    def __init__(self, plot, parent=None):
        super().__init__(parent)
        self.plot = plot
        row = QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0)
        for text, callback in (("自动范围", plot.autoRange), ("放大", lambda: plot.getPlotItem().vb.scaleBy((.75, .75))),
                               ("缩小", lambda: plot.getPlotItem().vb.scaleBy((1.35, 1.35))), ("保存图片", plot.save_snapshot)):
            button = QPushButton(text); button.clicked.connect(callback); row.addWidget(button)
        row.addStretch(1)


class PyQtGraphPlot(QWidget):
    """Reusable interactive PyQtGraph canvas for all application figures."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plotItem = self._plot_widget.plotItem
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=.18)
        self._legend = None
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar = _PyQtGraphToolbar(self, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self._plot_widget, 1)

    def plot(self, *args, **kwargs): return self._plot_widget.plot(*args, **kwargs)
    def addItem(self, item): self._plot_widget.addItem(item)
    def showGrid(self, *args, **kwargs): self._plot_widget.showGrid(*args, **kwargs)
    def setLabel(self, *args, **kwargs): self._plot_widget.setLabel(*args, **kwargs)
    def setTitle(self, *args, **kwargs): self._plot_widget.setTitle(*args, **kwargs)
    def setXRange(self, *args, **kwargs): self._plot_widget.setXRange(*args, **kwargs)
    def setYRange(self, *args, **kwargs): self._plot_widget.setYRange(*args, **kwargs)
    def autoRange(self): self._plot_widget.autoRange()
    def getPlotItem(self): return self._plot_widget.getPlotItem()
    def clear(self):
        self._plot_widget.clear(); self._legend = None
        self.plotItem.showAxis("bottom"); self.plotItem.showAxis("left")
        self._plot_widget.showGrid(x=True, y=True, alpha=.18)
    def add_subplot(self, *args, **kwargs): return _PyQtGraphAxis(self)
    def draw_idle(self): pass
    def save_snapshot(self, path=None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "保存图像", str(Path.cwd() / "plot.png"), "PNG files (*.png);;PDF files (*.pdf)")
        if not path:
            return
        path = str(path)
        if Path(path).suffix.lower() == ".pdf":
            writer = QPdfWriter(path)
            painter = QPainter(writer)
            self._plot_widget.render(painter)
            painter.end()
        else:
            self._plot_widget.grab().save(path)
    def savefig(self, path, dpi=180): self.save_snapshot(str(path))


class ChannelIdSpinBox(QSpinBox):
    """Spin by local column while displaying the source's real channel ID."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._channel_ids = ()

    def set_channel_ids(self, channel_ids) -> None:
        self._channel_ids = tuple(int(value) for value in channel_ids)
        self.setRange(1, max(1, len(self._channel_ids)))
        self.lineEdit().setText(self.textFromValue(self.value()))

    def textFromValue(self, value: int) -> str:
        index = int(value) - 1
        if 0 <= index < len(self._channel_ids):
            return f"ch {self._channel_ids[index]}"
        return f"ch {int(value)}"

    def valueFromText(self, text: str) -> int:
        match = re.search(r"-?\d+", str(text))
        if match:
            requested = int(match.group())
            if requested in self._channel_ids:
                return self._channel_ids.index(requested) + 1
        return super().valueFromText(text)


def _recording_duration_seconds(meta) -> float:
    return max(0.001, meta.rows / meta.fs)


def _seconds_window_to_samples(
    meta, start_sec: float, end_sec: float, *, full_duration: bool,
) -> tuple[int, int]:
    """Convert a start/end time window in seconds to sample indices."""
    if full_duration:
        return 0, meta.rows
    start = max(0, int(round(float(start_sec) * meta.fs)))
    end = max(start + 1, min(meta.rows, int(round(float(end_sec) * meta.fs))))
    if end <= start:
        end = min(meta.rows, start + 1)
    return start, end


def _slice_duration_from_end(start_sec: float, end_sec: float) -> float:
    """Convert a start/end pair into a non-negative duration for legacy APIs."""
    start = float(start_sec or 0.0)
    end = float(end_sec or 0.0)
    if end <= start:
        raise ValueError("时间窗终点必须大于起点。")
    return end - start


def _single_import_output_directory(
    output_root: str | Path, source_path: str | Path, date_value: str = "",
    animal_value: str = "",
) -> Path:
    """Return the legacy ``root/date/animal`` directory for one-file imports."""
    root = Path(output_root)
    source = Path(source_path)
    compact_date = re.sub(r"[^0-9]", "", str(date_value or "").strip())
    date_token = compact_date if len(compact_date) == 8 else ""
    animal_token = str(animal_value or "").strip()
    if not animal_token.isdigit() or int(animal_token) < 0:
        animal_token = ""

    parsed = parse_filename_metadata(source)
    if not date_token:
        candidate = re.sub(r"[^0-9]", "", str(parsed.get("date", "")))
        if len(candidate) == 8:
            date_token = candidate
    if not animal_token:
        candidate = str(parsed.get("animal", "") or "").strip()
        if candidate.isdigit() and int(candidate) > 0:
            animal_token = candidate

    # Common study folders put the animal/session immediately after an
    # eight-digit acquisition date: .../20260623/1108/source.bin.
    source_parts = list(source.parts)
    for index in range(len(source_parts) - 1):
        possible_date = re.sub(r"[^0-9]", "", source_parts[index])
        possible_animal = source_parts[index + 1].strip()
        if len(possible_date) == 8 and possible_date.isdigit() and possible_animal.isdigit():
            if not date_token:
                date_token = possible_date
            if not animal_token:
                animal_token = possible_animal

    if not date_token:
        return root
    if not animal_token:
        return root if root.name == date_token else root / date_token
    if root.name == animal_token and root.parent.name == date_token:
        return root
    if root.name == date_token:
        return root / animal_token
    return root / date_token / animal_token


def _stream_export_target(source_h5_path: str | Path, timestamp: str) -> tuple[Path, str]:
    """Return its ``processed`` child and a collision-safe streamed basename."""
    source = Path(source_h5_path)
    output_dir = source.parent if source.parent.name.lower() == "processed" else source.parent / "processed"
    return output_dir, f"{source.stem}_{timestamp}"


_PROCESSED_EXPORT_STEM = re.compile(
    r"^(?P<source>.+)_\d{8}_\d{6}(?:_(?:lfp|spike))?_processed$",
    re.IGNORECASE,
)
_PATH_FORMAT_CONTROLS = re.compile("[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


def normalize_h5_path_text(value: str | Path) -> str:
    """Remove invisible clipboard direction markers from a filesystem path."""
    text_value = _PATH_FORMAT_CONTROLS.sub("", str(value)).strip()
    if len(text_value) >= 2 and text_value[0] == text_value[-1] and text_value[0] in {'"', "'"}:
        text_value = text_value[1:-1].strip()
    return text_value


def resolve_original_h5_for_processed(processed_path: str | Path) -> Path:
    """Resolve the raw H5 represented by a timestamped processed export."""
    processed = Path(normalize_h5_path_text(processed_path))
    candidates: list[Path] = []
    try:
        if processed.is_file() and h5py.is_hdf5(processed):
            with h5py.File(processed, "r") as h5:
                provenance = read_h5_provenance(h5) or {}
                for item in reversed(provenance.get("lineage", [])):
                    if not isinstance(item, dict):
                        continue
                    value = str(item.get("path", "")).strip()
                    if value:
                        candidate = Path(value)
                        candidates.append(
                            candidate if candidate.is_absolute()
                            else processed.parent / candidate
                        )
                for key in ("source_h5", "source_file"):
                    value = h5.attrs.get(key, "")
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    if str(value).strip():
                        candidate = Path(str(value).strip())
                        candidates.append(
                            candidate if candidate.is_absolute()
                            else processed.parent / candidate
                        )
    except OSError:
        pass

    match = _PROCESSED_EXPORT_STEM.match(processed.stem)
    if match:
        raw_parent = (
            processed.parent.parent
            if processed.parent.name.lower() == "processed"
            else processed.parent
        )
        # The export naming convention is authoritative for this workflow:
        # it points to the unprocessed H5 beside the ``processed`` folder.
        candidates.insert(0, raw_parent / f"{match.group('source')}{processed.suffix}")
    elif processed.parent.name.lower() == "processed":
        # Some reviewed copies keep the raw basename unchanged and differ
        # only by living inside the ``processed`` child directory.
        candidates.insert(0, processed.parent.parent / processed.name)

    processed_resolved = processed.resolve(strict=False)
    for candidate in candidates:
        try:
            if (candidate.suffix.lower() in {".h5", ".hdf5"}
                    and candidate.resolve(strict=False) != processed_resolved
                    and candidate.is_file()):
                return candidate.resolve()
        except OSError:
            continue
    expected = candidates[0] if candidates else processed.parent
    raise FileNotFoundError(
        f"无法从处理结果追溯原始 H5：{processed}\n"
        f"按目录与文件名规则预期位置：{expected}"
    )


class PyQtGraphPreview(QWidget):
    """Lazy, wheel-zoomable single-channel PyQtGraph preview."""

    window_changed = pyqtSignal(float, float)
    channel_previewed = pyqtSignal(int)

    def __init__(
        self, source: LazyH5Source, parent: QWidget | None = None,
        *, title_prefix: str = "数据通道预览",
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.source = source
        self.title_prefix = str(title_prefix)
        self.figure = PyQtGraphPlot(self)
        self.canvas = self.figure
        self.toolbar = self.figure.toolbar
        self.channel_spin = ChannelIdSpinBox()
        self.channel_spin.setMinimum(1)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setSuffix(" s")
        self.start_spin.setDecimals(3)
        self.start_spin.setRange(0.0, 0.0)
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setSuffix(" s")
        self.end_spin.setDecimals(3)
        self.end_spin.setRange(0.001, 3600.0)
        self.end_spin.setValue(5.0)
        self.full_duration_check = QCheckBox("全时长")
        self.full_duration_check.setChecked(True)
        self.linewidth_spin = QDoubleSpinBox()
        self.linewidth_spin.setRange(0.10, 3.00)
        self.linewidth_spin.setSingleStep(0.05)
        self.linewidth_spin.setDecimals(2)
        self.linewidth_spin.setValue(0.55)
        # Keep the compact preview toolbar on one row.  In the former grid,
        # the line-width column carried the horizontal stretch and therefore
        # expanded across nearly the entire window.
        self.channel_spin.setFixedWidth(84)
        self.start_spin.setFixedWidth(126)
        self.end_spin.setFixedWidth(126)
        self.linewidth_spin.setFixedWidth(86)
        self._channel_cycle_columns: tuple[int, ...] = ()
        self._preview_enabled = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        # Give the user enough time to finish typing a start/end value
        # before issuing another HDF5 read and redraw.
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self.refresh)
        self.status_label = QLabel("请先加载 HDF5 数据。")
        self.status_label.setWordWrap(True)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        for label, control in (
            ("通道", self.channel_spin),
            ("", self.full_duration_check),
            ("起点", self.start_spin),
            ("终点", self.end_spin),
            ("线宽", self.linewidth_spin),
        ):
            if label:
                controls.addWidget(QLabel(label))
            controls.addWidget(control)
        previous = QPushButton("上一通道")
        previous.clicked.connect(lambda: self._step_channel(-1))
        following = QPushButton("下一通道")
        following.clicked.connect(lambda: self._step_channel(1))
        refresh = QPushButton("刷新预览")
        refresh.clicked.connect(self._refresh_now)
        for button in (previous, following, refresh):
            controls.addWidget(button)
        # Plot navigation used to occupy another full row.  Its four buttons
        # now continue on the same compact line as channel/time controls.
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        controls.addWidget(self.toolbar)
        controls.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.status_label)
        layout.addWidget(self.canvas, 1)
        for spin in (self.channel_spin, self.start_spin, self.end_spin, self.linewidth_spin):
            spin.valueChanged.connect(self._schedule_refresh)
            spin.editingFinished.connect(self._schedule_refresh)
        self.start_spin.valueChanged.connect(self._sync_time_window_bounds)
        self.end_spin.valueChanged.connect(self._sync_time_window_bounds)
        self.full_duration_check.toggled.connect(self._toggle_full_duration)
        self._toggle_full_duration(self.full_duration_check.isChecked())

    def _schedule_refresh(self, _value=None) -> None:
        """Refresh after control changes without issuing one read per keystroke."""
        if self._preview_enabled and self.source.metadata is not None:
            self._refresh_timer.start()

    def _refresh_now(self) -> None:
        self._preview_enabled = True
        self._refresh_timer.stop()
        self.refresh()

    def _sync_time_window_bounds(self, _value=None) -> None:
        """Keep start/end spin ranges consistent while editing a custom window."""
        meta = self.source.metadata
        if meta is None or self.full_duration_check.isChecked():
            return
        total_sec = _recording_duration_seconds(meta)
        start_sec = min(max(0.0, self.start_spin.value()), max(0.0, total_sec - 0.001))
        end_sec = min(max(start_sec + 0.001, self.end_spin.value()), total_sec)
        self.start_spin.blockSignals(True)
        self.end_spin.blockSignals(True)
        self.start_spin.setMaximum(max(0.0, end_sec - 0.001))
        self.end_spin.setMinimum(min(total_sec, start_sec + 0.001))
        self.end_spin.setMaximum(total_sec)
        if not np.isclose(self.start_spin.value(), start_sec):
            self.start_spin.setValue(start_sec)
        if not np.isclose(self.end_spin.value(), end_sec):
            self.end_spin.setValue(end_sec)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(False)

    def _toggle_full_duration(self, enabled: bool) -> None:
        """Use the complete recording while checked, without changing data."""
        self.start_spin.setEnabled(not enabled)
        self.end_spin.setEnabled(not enabled)
        if enabled:
            meta = self.source.metadata
            if meta is not None:
                total_sec = _recording_duration_seconds(meta)
                self.start_spin.blockSignals(True)
                self.end_spin.blockSignals(True)
                self.start_spin.setValue(0.0)
                self.end_spin.setValue(total_sec)
                self.start_spin.blockSignals(False)
                self.end_spin.blockSignals(False)
        elif self.source.metadata is not None:
            self._sync_time_window_bounds()
        if self.source.metadata is not None:
            self._schedule_refresh()

    def selected_time_window_seconds(self) -> tuple[float, float, bool]:
        """Return (start_sec, end_sec, full_duration) for the active preview window."""
        meta = self.source.metadata
        if meta is None:
            return 0.0, 5.0, True
        if self.full_duration_check.isChecked():
            total_sec = _recording_duration_seconds(meta)
            return 0.0, total_sec, True
        start_sec = float(self.start_spin.value())
        end_sec = float(self.end_spin.value())
        if end_sec <= start_sec:
            end_sec = min(_recording_duration_seconds(meta), start_sec + 0.001)
        return start_sec, end_sec, False

    def set_channel_cycle_columns(self, columns) -> None:
        """Restrict previous/next navigation to the currently reviewed channels."""
        meta = self.source.metadata
        if meta is None:
            self._channel_cycle_columns = ()
            return
        normalized = np.unique(np.asarray(columns, dtype=np.int64).ravel())
        normalized = normalized[(normalized >= 0) & (normalized < meta.channels)]
        self._channel_cycle_columns = tuple(int(column) for column in normalized)
        if self._channel_cycle_columns and self.channel_spin.value() - 1 not in self._channel_cycle_columns:
            self.channel_spin.setValue(self._channel_cycle_columns[0] + 1)

    def current_channel_id(self) -> int | None:
        meta = self.source.metadata
        if meta is None:
            return None
        column = self.channel_spin.value() - 1
        if not 0 <= column < meta.channels:
            return None
        return int(meta.channel_ids[column])

    def _step_channel(self, delta: int) -> None:
        meta = self.source.metadata
        if meta is None:
            return
        if self._channel_cycle_columns:
            current = self.channel_spin.value() - 1
            try:
                index = self._channel_cycle_columns.index(current)
            except ValueError:
                index = 0 if delta >= 0 else len(self._channel_cycle_columns) - 1
            next_column = self._channel_cycle_columns[(index + delta) % len(self._channel_cycle_columns)]
            self.channel_spin.setValue(next_column + 1)
            return
        self.channel_spin.setValue(
            max(1, min(meta.channels, self.channel_spin.value() + int(delta)))
        )

    def configure_for_source(self) -> None:
        meta = self.source.metadata
        if meta is None:
            return
        self.channel_spin.set_channel_ids(meta.channel_ids)
        total_sec = _recording_duration_seconds(meta)
        self.start_spin.setRange(0.0, max(0.0, total_sec - 0.001))
        self.end_spin.setRange(0.001, total_sec)
        if self.full_duration_check.isChecked():
            self._toggle_full_duration(True)
        else:
            self.end_spin.setValue(min(max(self.end_spin.value(), self.start_spin.value() + 0.001), total_sec))
            self._sync_time_window_bounds()
            self.refresh()

    def set_source(self, source, *, defer_initial_draw: bool = False) -> None:
        self.source = source
        if defer_initial_draw:
            self._preview_enabled = False
            self._refresh_timer.stop()
        self.configure_for_source()
        if defer_initial_draw and self.source.metadata is not None:
            self.figure.clear()
            self.canvas.draw_idle()
            self.status_label.setText("数据已载入；点击“刷新预览”后读取并绘制当前通道。")

    def refresh(self) -> None:
        meta = self.source.metadata
        if meta is None:
            return
        start, end = _seconds_window_to_samples(
            meta,
            self.start_spin.value(),
            self.end_spin.value(),
            full_duration=self.full_duration_check.isChecked(),
        )
        display_step = 1
        if self.full_duration_check.isChecked():
            # Full-duration viewing must remain responsive for long HDF5
            # recordings.  This affects plotted samples only, never data.
            display_step = max(1, int(np.ceil((end - start) / 120_000)))
        channel = self.channel_spin.value() - 1
        if self._channel_cycle_columns and channel not in self._channel_cycle_columns:
            channel = self._channel_cycle_columns[0]
            self.channel_spin.blockSignals(True)
            self.channel_spin.setValue(channel + 1)
            self.channel_spin.blockSignals(False)
        channel_id = meta.channel_ids[channel]
        values = self.source.read(start, end, channel, step=display_step).ravel()
        time_values = meta.time_offset + np.arange(
            start, end, display_step, dtype=np.float64
        )[:values.size] / meta.fs
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.plot(time_values, values, color="#1f77b4", linewidth=self.linewidth_spin.value(), antialiased=True)
        axis.set_title(f"{self.title_prefix} — ch{channel_id}（列 {channel + 1}）")
        axis.set_xlabel("时间（秒）")
        axis.set_ylabel("幅值(μV)")
        axis.grid(alpha=0.25)
        # Fit the freshly refreshed preview once.  Subsequent zoom/pan is
        # left untouched until the next explicit refresh.
        self.figure.autoRange()
        self.canvas.draw_idle()
        self.status_label.setText(
            f"HDF5 懒加载：ch{channel_id}（列 {channel + 1}），{start / meta.fs:.3f}–{end / meta.fs:.3f} 秒，"
            f"显示 {values.size:,}/{end - start:,} 个采样点"
            + (f"（绘图步长 {display_step}）" if display_step > 1 else "")
            + "。"
        )
        self.window_changed.emit(start / meta.fs, end / meta.fs)
        self.channel_previewed.emit(int(channel_id))


class OverviewPageLoadWorker(QThread):
    """Read one already-downsampled overview page without blocking the UI."""

    loaded = pyqtSignal(object, object)
    failed = pyqtSignal(object, str)

    def __init__(self, source, context):
        super().__init__()
        self.source = source
        self.context = dict(context)

    def run(self) -> None:
        try:
            values = self.source.read(
                self.context["start"], self.context["end"],
                self.context["columns"],
                step=self.context["step"],
            )
            self.loaded.emit(self.context["key"], np.asarray(values, dtype=np.float32))
        except Exception as exc:
            self.failed.emit(self.context["key"], str(exc))


class PyQtGraphOverview(QMainWindow):
    """Lazy 10×10 all-channel page using PyQtGraph.

    Every refresh reads only the visible time range and the 100 channels on
    the active page.  HDF5 stride selection keeps the plotted data near 1,200
    points/channel even for long visible windows.
    """

    def __init__(self, source: LazyH5Source, start_seconds: float = 0.0, end_seconds: float = 5.0,
                 *, full_duration: bool = True,
                 selection_mode: bool = False, selected_channel_ids=(), selection_callback=None,
                 selection_label: str = "ICA", display_channel_ids=None,
                 initial_display_channel_ids=None, display_scope_label: str = "",
                 selection_groups=None):
        super().__init__()
        if source.metadata is None:
            raise RuntimeError("Load an HDF5 source before opening the overview.")
        self.source = source
        self.selection_mode = bool(selection_mode)
        self.page = 0
        # Size a page from the available desktop rather than from a design
        # resolution. A 5x5 page is unreadable on a 1366x768 laptop and gets
        # even smaller with Windows DPI scaling.
        screen = QApplication.screenAt(self.frameGeometry().center()) or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        available_width = available.width() if available is not None else 1366
        available_height = available.height() if available is not None else 768
        if available_height < 800 or available_width < 1300:
            self._overview_page_size = 6
        elif available_height < 1000 or available_width < 1700:
            self._overview_page_size = 9
        elif available_height < 1300:
            self._overview_page_size = 12
        else:
            self._overview_page_size = 16
        self._page_plots = []
        self._plot_curves = []
        self._grid_signature = None
        self._detail_dialog = None
        self._page_cache = OrderedDict()
        self._page_workers = {}
        self._page_load_queue = []
        self._page_queue_keys = set()
        self._active_page_key = None
        self._requested_page_key = None
        self._display_context = None
        self._display_values = None
        self._relayout_timer = QTimer(self)
        self._relayout_timer.setSingleShot(True)
        self._relayout_timer.timeout.connect(self._relayout_visible_page)
        source_channel_ids = np.asarray(source.metadata.channel_ids, dtype=np.int64)
        if display_channel_ids is None:
            self._display_columns = np.arange(source.metadata.channels, dtype=np.int64)
        else:
            requested_display_ids = {int(channel) for channel in display_channel_ids}
            self._display_columns = np.flatnonzero(np.isin(source_channel_ids, sorted(requested_display_ids)))
            if not self._display_columns.size:
                raise ValueError("No requested overview channels are available in the current source.")
        # Keep the caller-provided scope as the maximum visible set.  A user
        # can temporarily narrow the overview, but cannot accidentally reveal
        # channels excluded by a specialized selector (for example ICA).
        self._base_display_columns = self._display_columns.copy()
        if initial_display_channel_ids is not None:
            initial_ids = [int(value) for value in initial_display_channel_ids]
            initial_set = set(initial_ids)
            base_ids = {
                int(source_channel_ids[column]): int(column)
                for column in self._base_display_columns
            }
            unavailable = [channel for channel in initial_ids if channel not in base_ids]
            if unavailable:
                raise ValueError(
                    "Initial overview channels are outside the available display scope: "
                    + ", ".join(map(str, unavailable))
                )
            if initial_set:
                self._display_columns = np.asarray(
                    [base_ids[channel] for channel in dict.fromkeys(initial_ids)], dtype=np.int64
                )
        available_channel_ids = {int(value) for value in source_channel_ids}
        self._selected_channel_ids = {
            int(channel) for channel in selected_channel_ids
            if int(channel) in available_channel_ids
        }
        self._selection_groups = {
            str(label): {int(channel) for channel in channel_ids if int(channel) in available_channel_ids}
            for label, channel_ids in (selection_groups or {}).items()
        }
        self._selection_callback = selection_callback
        self.selection_label = str(selection_label or "通道")
        self.display_scope_label = str(display_scope_label or "").strip()
        self.setWindowTitle(f"选择 {self.selection_label} 通道（PyQtGraph）" if self.selection_mode else "全通道总览（PyQtGraph）")
        if available is not None:
            self.resize(min(1600, available_width), min(980, available_height))
        else:
            self.resize(1200, 760)
        self.setMinimumSize(720, 500)
        self._build_ui(start_seconds, end_seconds, full_duration=full_duration)
        self.render_page()

    def _build_ui(self, start_seconds: float, end_seconds: float, *, full_duration: bool = True) -> None:
        outer = QWidget()
        layout = QVBoxLayout(outer)
        controls = QHBoxLayout()
        self.previous_button = QPushButton("上一页")
        self.previous_button.clicked.connect(lambda: self.change_page(-1))
        self.next_button = QPushButton("下一页")
        self.next_button.clicked.connect(lambda: self.change_page(1))
        total_sec = _recording_duration_seconds(self.source.metadata)
        self.full_duration_check = QCheckBox("全时长")
        self.full_duration_check.setChecked(bool(full_duration))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setSuffix(" s")
        self.start_spin.setDecimals(3)
        self.start_spin.setRange(0.0, max(0.0, total_sec - 0.001))
        self.start_spin.setValue(max(0.0, start_seconds))
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setSuffix(" s")
        self.end_spin.setDecimals(3)
        self.end_spin.setRange(0.001, total_sec)
        self.end_spin.setValue(min(max(0.001, end_seconds), total_sec))
        refresh = QPushButton("刷新当前窗口")
        refresh.clicked.connect(lambda: self.render_page(force=True))
        self.page_label = QLabel()
        self.status_label = QLabel()
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        for widget in (
            self.previous_button,
            self.next_button,
            self.full_duration_check,
            QLabel("起点"),
            self.start_spin,
            QLabel("终点"),
            self.end_spin,
            refresh,
            self.page_label,
            self.status_label,
        ):
            controls.addWidget(widget)
        self.full_duration_check.toggled.connect(self._toggle_full_duration)
        for spin in (self.start_spin, self.end_spin):
            spin.valueChanged.connect(self._sync_time_window_bounds)
            spin.editingFinished.connect(lambda: self.render_page(force=True))
        self._toggle_full_duration(self.full_duration_check.isChecked())
        controls.addStretch(1)
        export_controls = QHBoxLayout()
        self.raw_export_channels_edit = QLineEdit()
        self.raw_export_channels_edit.setPlaceholderText("例如：7, 500, 20-26（实际通道号）")
        self.raw_export_channels_edit.setMinimumWidth(260)
        self.raw_export_directory_edit = QLineEdit(str(Path.cwd()))
        self.raw_export_directory_edit.setReadOnly(True)
        choose_directory = QPushButton("选择目录")
        choose_directory.clicked.connect(self.choose_raw_export_directory)
        self.raw_export_button = QPushButton("导出单通道 raw.h5")
        self.raw_export_button.clicked.connect(self.export_selected_raw_channels)
        self.raw_export_status = QLabel("按实际通道号导出；文件可重新导入预处理。")
        self.raw_export_status.setWordWrap(True)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        self.graphics.setMinimumHeight(300)
        # All PlotItems share this GraphicsScene.  Connecting once here avoids
        # accumulating one scene callback per plot on every refresh/page turn.
        self.graphics.scene().sigMouseClicked.connect(self._open_detail_from_scene)
        layout.addLayout(controls)
        display_controls = QHBoxLayout()
        display_controls.addWidget(QLabel("每页显示图数"))
        self.overview_page_size_spin = QSpinBox()
        self.overview_page_size_spin.setRange(1, max(1, int(self._base_display_columns.size)))
        self.overview_page_size_spin.setValue(
            min(self._overview_page_size, max(1, int(self._base_display_columns.size)))
        )
        self.overview_page_size_spin.setToolTip("设置总览每一页同时显示的通道波形数量")
        display_controls.addWidget(self.overview_page_size_spin)
        display_controls.addWidget(QLabel("指定总览通道"))
        self.overview_channels_edit = QLineEdit()
        self.overview_channels_edit.setPlaceholderText("留空=全部；例如：1,5,20-26（实际通道号）")
        self.overview_channels_edit.setMinimumWidth(360)
        self.overview_channels_edit.setToolTip("只显示输入的实际通道；未输入的通道不会出现在总览中")
        if not np.array_equal(self._display_columns, self._base_display_columns):
            channel_ids = np.asarray(self.source.metadata.channel_ids, dtype=np.int64)
            self.overview_channels_edit.setText(
                ",".join(str(int(channel_ids[column])) for column in self._display_columns)
            )
        display_controls.addWidget(self.overview_channels_edit, 1)
        apply_display = QPushButton("应用总览显示")
        apply_display.clicked.connect(self._apply_overview_display_settings)
        self.overview_channels_edit.returnPressed.connect(self._apply_overview_display_settings)
        self.overview_page_size_spin.editingFinished.connect(self._apply_overview_display_settings)
        display_controls.addWidget(apply_display)
        self.overview_display_summary = QLabel()
        display_controls.addWidget(self.overview_display_summary)
        layout.addLayout(display_controls)
        if self.display_scope_label:
            channel_ids = np.asarray(self.source.metadata.channel_ids, dtype=np.int64)
            available_ids = [int(channel_ids[column]) for column in self._base_display_columns]
            compact_ids = compact_integer_ranges(available_ids).replace(",", ", ") or "无"
            self.overview_available_channels_label = QPlainTextEdit()
            self.overview_available_channels_label.setPlainText(
                f"{self.display_scope_label}全部可选通道（共 {len(available_ids)} 个）："
                f"{compact_ids}"
            )
            self.overview_available_channels_label.setReadOnly(True)
            line_wrap_mode = getattr(QPlainTextEdit, "LineWrapMode", QPlainTextEdit)
            self.overview_available_channels_label.setLineWrapMode(
                line_wrap_mode.WidgetWidth
            )
            self.overview_available_channels_label.setMinimumHeight(88)
            self.overview_available_channels_label.setMaximumHeight(170)
            self.overview_available_channels_label.setStyleSheet(
                "QPlainTextEdit { background:#f4f8fc; border:1px solid #c7d5e0; "
                "color:#24465f; padding:5px; }"
            )
            self.overview_available_channels_label.setToolTip(
                "这里列出该类型的全部实际通道号，可复制后填写到“指定总览通道”"
            )
            layout.addWidget(self.overview_available_channels_label)
        self._update_overview_display_summary()
        if self.selection_mode:
            select_controls = QHBoxLayout()
            select_controls.addWidget(QLabel(f"{self.selection_label} 通道（实际通道号）"))
            self.selection_channels_edit = QLineEdit()
            self.selection_channels_edit.setPlaceholderText("例如：7, 500, 20-26；点击小图也可勾选")
            self.selection_channels_edit.setMinimumWidth(420)
            self.selection_channels_edit.textChanged.connect(self._selection_text_changed)
            self.selection_summary = QLabel()
            confirm = QPushButton(
                "确认并开始滤波" if self.selection_label == "滤波"
                else f"确认选择并用于{self.selection_label}"
            )
            confirm.setObjectName("primaryAction")
            confirm.clicked.connect(self._confirm_selection)
            cancel = QPushButton("取消")
            cancel.clicked.connect(self.close)
            for widget in (self.selection_channels_edit, self.selection_summary, confirm, cancel):
                select_controls.addWidget(widget)
            select_controls.addStretch(1)
            layout.addLayout(select_controls)
            if self._selection_groups:
                group_controls = QHBoxLayout()
                group_controls.addWidget(QLabel("按坏道判断快速选择："))
                self.selection_group_buttons = {}
                for label, channel_ids in self._selection_groups.items():
                    add_button = QPushButton(f"加入{label}")
                    add_button.setEnabled(bool(channel_ids))
                    add_button.clicked.connect(
                        lambda _checked=False, ids=channel_ids: self._change_selection_group(ids, add=True)
                    )
                    remove_button = QPushButton(f"移除{label}")
                    remove_button.setEnabled(bool(channel_ids))
                    remove_button.clicked.connect(
                        lambda _checked=False, ids=channel_ids: self._change_selection_group(ids, add=False)
                    )
                    self.selection_group_buttons[label] = (add_button, remove_button)
                    group_controls.addWidget(add_button)
                    group_controls.addWidget(remove_button)
                group_controls.addStretch(1)
                layout.addLayout(group_controls)
            self._sync_selection_text()
        for widget in (
            QLabel("单通道导出"), self.raw_export_channels_edit,
            self.raw_export_directory_edit, choose_directory, self.raw_export_button,
        ):
            export_controls.addWidget(widget)
        layout.addLayout(export_controls)
        layout.addWidget(self.raw_export_status)
        layout.addWidget(self.graphics, 1)
        self.setCentralWidget(outer)

    def resizeEvent(self, event) -> None:
        """Reflow the plot grid after moving to a differently sized screen."""
        super().resizeEvent(event)
        if hasattr(self, "graphics") and hasattr(self, "_relayout_timer"):
            # Coalesce the many resize events emitted while maximizing a
            # window; rebuilding a GraphicsLayout for each event is costly.
            self._relayout_timer.start(120)

    def _relayout_visible_page(self) -> None:
        """Rebuild only the visible grid using the current viewport aspect."""
        if self._display_context is None or self._display_values is None:
            return
        # Force _ensure_plot_grid() to reconsider rows/columns. The sampled
        # page data stay in memory and no HDF5 read is repeated.
        self._grid_signature = None
        self._display_page(self._display_context, self._display_values)

    def _parse_selection_text(self, text: str) -> set[int]:
        requested = set()
        for part in text.replace("，", ",").replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                if "-" in part:
                    first, last = (int(value.strip()) for value in part.split("-", 1))
                    requested.update(range(min(first, last), max(first, last) + 1))
                else:
                    requested.add(int(part))
            except ValueError:
                # Permit an unfinished edit such as a trailing comma without
                # discarding selections already entered in the same field.
                continue
        available = {int(value) for value in self.source.metadata.channel_ids}
        return requested & available

    def _requested_overview_columns(self, text: str) -> np.ndarray:
        """Resolve an ordered actual-channel expression inside the allowed scope."""
        normalized = re.sub(r"(?i)\bch\s*", "", str(text or ""))
        normalized = normalized.replace("，", ",").replace("；", ",").replace(";", ",")
        normalized = normalized.replace("－", "-").replace("—", "-").strip()
        if not normalized:
            return self._base_display_columns.copy()
        requested = []
        for part in normalized.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                if "-" in part:
                    first_text, last_text = part.split("-", 1)
                    first, last = int(first_text.strip()), int(last_text.strip())
                    requested.extend(range(min(first, last), max(first, last) + 1))
                else:
                    requested.append(int(part))
            except ValueError as exc:
                raise ValueError(f"无法识别总览通道“{part}”，请使用 1,5,20-26 这样的格式。") from exc
        requested = list(dict.fromkeys(requested))
        if not requested:
            raise ValueError("没有识别到要显示的通道；留空可恢复显示全部通道。")
        channel_ids = np.asarray(self.source.metadata.channel_ids, dtype=np.int64)
        allowed = {
            int(channel_ids[column]): int(column) for column in self._base_display_columns
        }
        missing = [channel for channel in requested if channel not in allowed]
        if missing:
            raise ValueError(
                "这些实际通道不在当前总览可用范围内：" + ", ".join(map(str, missing))
            )
        return np.asarray([allowed[channel] for channel in requested], dtype=np.int64)

    def _update_overview_display_summary(self) -> None:
        if not hasattr(self, "overview_display_summary"):
            return
        self.overview_display_summary.setText(
            f"当前 {self._display_columns.size} 个通道，每页最多 {self._overview_page_size} 图"
        )

    def _apply_overview_display_settings(self) -> None:
        """Apply page capacity and an optional exact-channel overview filter."""
        try:
            columns = self._requested_overview_columns(self.overview_channels_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "总览通道设置", str(exc))
            return
        self._overview_page_size = max(1, int(self.overview_page_size_spin.value()))
        self._display_columns = columns
        self.page = 0
        self._page_cache.clear()
        self._page_load_queue.clear()
        self._page_queue_keys.clear()
        self._grid_signature = None
        self._update_overview_display_summary()
        self.render_page(force=True)

    def _selection_text_changed(self, text: str) -> None:
        if not self.selection_mode:
            return
        self._selected_channel_ids = self._parse_selection_text(text)
        self._refresh_selection_visuals()

    def _sync_selection_text(self) -> None:
        if not self.selection_mode:
            return
        text = ", ".join(map(str, sorted(self._selected_channel_ids)))
        self.selection_channels_edit.blockSignals(True)
        self.selection_channels_edit.setText(text)
        self.selection_channels_edit.blockSignals(False)
        self._refresh_selection_visuals()

    def _change_selection_group(self, channel_ids, *, add: bool) -> None:
        """Add or remove one QC category from the pending selection only."""
        group = {int(channel) for channel in channel_ids}
        if add:
            self._selected_channel_ids.update(group)
        else:
            self._selected_channel_ids.difference_update(group)
        self._sync_selection_text()

    def _refresh_selection_visuals(self) -> None:
        if not self.selection_mode:
            return
        self.selection_summary.setText(f"已选择 {len(self._selected_channel_ids)} 个通道")
        meta = self.source.metadata
        for plot, column in self._page_plots:
            channel_id = int(meta.channel_ids[column])
            selected = channel_id in self._selected_channel_ids
            plot.setTitle(("✓ " if selected else "") + f"ch{channel_id}", size="9pt", color="#d32f2f" if selected else "#666666")

    def _confirm_selection(self) -> None:
        if not self._selected_channel_ids:
            QMessageBox.information(self, f"未选择{self.selection_label}通道", "请点击波形或输入至少一个实际通道号。")
            return
        if self.selection_label == "ICA" and len(self._selected_channel_ids) < 2:
            QMessageBox.information(self, "ICA 通道不足", "ICA 至少需要选择两个通道。")
            return
        callback = self._selection_callback
        if callable(callback):
            callback(sorted(self._selected_channel_ids))
        self.close()

    @property
    def total_pages(self) -> int:
        return max(1, int(np.ceil(self._display_columns.size / self._overview_page_size)))

    def change_page(self, delta: int) -> None:
        self.page = (self.page + delta) % self.total_pages
        self.render_page()

    def choose_raw_export_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "选择单通道 raw.h5 保存目录", self.raw_export_directory_edit.text() or str(Path.cwd())
        )
        if directory:
            self.raw_export_directory_edit.setText(directory)

    def _raw_export_channel_ids(self) -> list[int]:
        text = self.raw_export_channels_edit.text().replace("，", ",").replace("；", ",").replace(";", ",").strip()
        if not text:
            raise ValueError("请输入要导出的实际通道号，例如 7,500,20-26。")
        requested = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                first, last = (int(value.strip()) for value in part.split("-", 1))
                requested.extend(range(min(first, last), max(first, last) + 1))
            else:
                requested.append(int(part))
        requested = list(dict.fromkeys(requested))
        if not requested:
            raise ValueError("未识别到有效通道号。")
        available = {int(channel) for channel in self.source.metadata.channel_ids}
        missing = [channel for channel in requested if channel not in available]
        if missing:
            raise ValueError("这些实际通道未加载，不能导出：" + ", ".join(map(str, missing)))
        return requested

    def export_selected_raw_channels(self) -> None:
        if getattr(self, "_raw_channel_export_worker", None) is not None and self._raw_channel_export_worker.isRunning():
            return
        try:
            channels = self._raw_export_channel_ids()
            directory = self.raw_export_directory_edit.text().strip()
            if not directory:
                raise ValueError("请先选择保存目录。")
            worker = RawChannelExportWorker(self.source, channels, directory)
        except Exception as exc:
            QMessageBox.warning(self, "导出单通道 raw.h5", str(exc))
            return
        self.raw_export_button.setEnabled(False)
        self.raw_export_status.setText(f"准备导出 {len(channels)} 个单通道 raw.h5…")
        worker.progress.connect(lambda value, text: self.raw_export_status.setText(f"{text}：{value:.1f}%"))
        worker.completed.connect(self._finish_raw_channel_export)
        worker.failed.connect(self._fail_raw_channel_export)
        worker.finished.connect(lambda: self.raw_export_button.setEnabled(True))
        self._raw_channel_export_worker = worker
        worker.start()

    def _finish_raw_channel_export(self, paths) -> None:
        names = [Path(path).name for path in paths]
        self.raw_export_status.setText(
            f"已导出 {len(names)} 个单通道 Blosc/LZ4+bitshuffle raw.h5：" + "、".join(names[:4])
            + (" …" if len(names) > 4 else "")
        )

    def _fail_raw_channel_export(self, error: str) -> None:
        self.raw_export_status.setText(f"单通道 raw.h5 导出失败：{error}")
        QMessageBox.critical(self, "导出单通道 raw.h5", error)

    def _sync_time_window_bounds(self, _value=None) -> None:
        meta = self.source.metadata
        if meta is None or self.full_duration_check.isChecked():
            return
        total_sec = _recording_duration_seconds(meta)
        start_sec = min(max(0.0, self.start_spin.value()), max(0.0, total_sec - 0.001))
        end_sec = min(max(start_sec + 0.001, self.end_spin.value()), total_sec)
        self.start_spin.blockSignals(True)
        self.end_spin.blockSignals(True)
        self.start_spin.setMaximum(max(0.0, end_sec - 0.001))
        self.end_spin.setMinimum(min(total_sec, start_sec + 0.001))
        self.end_spin.setMaximum(total_sec)
        if not np.isclose(self.start_spin.value(), start_sec):
            self.start_spin.setValue(start_sec)
        if not np.isclose(self.end_spin.value(), end_sec):
            self.end_spin.setValue(end_sec)
        self.start_spin.blockSignals(False)
        self.end_spin.blockSignals(False)

    def _toggle_full_duration(self, enabled: bool) -> None:
        self.start_spin.setEnabled(not enabled)
        self.end_spin.setEnabled(not enabled)
        if enabled:
            meta = self.source.metadata
            if meta is not None:
                total_sec = _recording_duration_seconds(meta)
                self.start_spin.blockSignals(True)
                self.end_spin.blockSignals(True)
                self.start_spin.setValue(0.0)
                self.end_spin.setValue(total_sec)
                self.start_spin.blockSignals(False)
                self.end_spin.blockSignals(False)
        elif self.source.metadata is not None:
            self._sync_time_window_bounds()

    def _page_context(self, page=None) -> dict:
        meta = self.source.metadata
        if meta is None:
            raise RuntimeError("No overview source is loaded.")
        page = self.page if page is None else int(page) % self.total_pages
        start, end = _seconds_window_to_samples(
            meta,
            self.start_spin.value(),
            self.end_spin.value(),
            full_duration=self.full_duration_check.isChecked(),
        )
        display_start = page * self._overview_page_size
        display_end = min(self._display_columns.size, display_start + self._overview_page_size)
        columns = self._display_columns[display_start:display_end]
        step = max(1, int(np.ceil((end - start) / 1200)))
        return {
            # Columns are part of the cache key because the user can change
            # the exact overview channel set without reopening the window.
            "key": (page, start, end, step, tuple(map(int, columns))), "page": page,
            "start": start, "end": end, "step": step,
            "display_start": display_start, "display_end": display_end,
            "columns": columns,
        }

    def render_page(self, force: bool = False) -> None:
        meta = self.source.metadata
        if meta is None:
            return
        context = self._page_context()
        key = context["key"]
        self._requested_page_key = key
        if force:
            self._page_cache.pop(key, None)
        values = self._page_cache.get(key)
        if values is not None:
            self._page_cache.move_to_end(key)
            self._display_page(context, values)
            return
        self.page_label.setText(f"第 {self.page + 1}/{self.total_pages} 页")
        self.status_label.setText("正在后台读取当前页显示数据…")
        self._queue_page_load(context, priority=True)

    def _queue_page_load(self, context: dict, priority: bool = False) -> None:
        key = context["key"]
        if key in self._page_cache or key in self._page_workers:
            return
        if key in self._page_queue_keys:
            if priority:
                self._page_load_queue = [item for item in self._page_load_queue if item["key"] != key]
                self._page_load_queue.insert(0, context)
            return
        self._page_queue_keys.add(key)
        if priority:
            self._page_load_queue.insert(0, context)
        else:
            self._page_load_queue.append(context)
        self._start_next_page_load()

    def _start_next_page_load(self) -> None:
        if self._active_page_key is not None or not self._page_load_queue:
            return
        context = self._page_load_queue.pop(0)
        key = context["key"]
        self._page_queue_keys.discard(key)
        worker = OverviewPageLoadWorker(self.source, context)
        worker.loaded.connect(self._finish_page_load)
        worker.failed.connect(self._fail_page_load)
        worker.finished.connect(lambda request_key=key: self._page_worker_finished(request_key))
        self._page_workers[key] = worker
        self._active_page_key = key
        worker.start()

    def _page_worker_finished(self, key) -> None:
        self._page_workers.pop(key, None)
        if self._active_page_key == key:
            self._active_page_key = None
        self._start_next_page_load()

    def _finish_page_load(self, key, values) -> None:
        self._page_cache[key] = values
        self._page_cache.move_to_end(key)
        # Cache exactly the current page and its two neighbours.  These are
        # display samples only, never a second copy of the recording.
        while len(self._page_cache) > 3:
            self._page_cache.popitem(last=False)
        if key == self._requested_page_key:
            context = self._page_context()
            if context["key"] == key:
                self._display_page(context, values)

    def _fail_page_load(self, key, error: str) -> None:
        if key == self._requested_page_key:
            self.status_label.setText(f"总览页读取失败：{error}")

    def _display_page(self, context: dict, values: np.ndarray) -> None:
        meta = self.source.metadata
        if meta is None:
            return
        if values.ndim == 1:
            values = values[:, None]
        self._display_context = dict(context)
        self._display_values = values
        columns = np.asarray(context["columns"], dtype=np.int64)
        visible_channels = columns.size
        if visible_channels <= 1:
            grid_columns = 1
        else:
            # EEG traces benefit from cells that are wider than they are tall.
            viewport_ratio = max(0.5, self.graphics.width() / max(1, self.graphics.height()))
            grid_columns = min(
                visible_channels,
                max(1, int(np.ceil(np.sqrt(visible_channels * viewport_ratio / 1.6)))),
            )
        self._ensure_plot_grid(visible_channels, grid_columns)
        time_values = meta.time_offset + np.arange(context["start"], context["end"], context["step"], dtype=float)[:values.shape[0]] / meta.fs
        self._page_plots = []
        for index, (plot, curve) in enumerate(self._plot_curves):
            curve.setData(time_values, values[:, index])
            if time_values.size:
                plot.setXRange(float(time_values[0]), float(time_values[-1]), padding=0)
            channel_values = np.asarray(values[:, index], dtype=float)
            finite_values = channel_values[np.isfinite(channel_values)]
            if finite_values.size:
                # Do not let a few stimulation/artifact spikes flatten the
                # ordinary EEG into an apparently empty plot.
                if finite_values.size >= 20:
                    y_low, y_high = np.percentile(finite_values, (1.0, 99.0))
                    y_low, y_high = float(y_low), float(y_high)
                else:
                    y_low = float(np.min(finite_values))
                    y_high = float(np.max(finite_values))
                y_padding = max(1e-6, (y_high - y_low) * 0.05)
                if np.isclose(y_low, y_high):
                    y_padding = max(1e-6, abs(y_low) * 0.05)
                plot.setYRange(y_low - y_padding, y_high + y_padding, padding=0)
            source_column = int(columns[index])
            channel_id = int(meta.channel_ids[source_column])
            selected = self.selection_mode and channel_id in self._selected_channel_ids
            title = ("✓ " if selected else "") + f"ch{channel_id}"
            plot.setTitle(title, size="9pt", color="#d32f2f" if selected else "#666666")
            self._page_plots.append((plot, source_column))
        self.page_label.setText(f"第 {context['page'] + 1}/{self.total_pages} 页")
        cache_note = "；正在后台预读相邻页" if self.total_pages > 1 else ""
        self.status_label.setText(
            f"显示 {context['display_start'] + 1}–{context['display_end']}/{self._display_columns.size} 个通道"
            f"（实际通道 {meta.channel_ids[columns[0]]}–{meta.channel_ids[columns[-1]]}）；"
            f"{context['start'] / meta.fs:.3f}–{context['end'] / meta.fs:.3f} 秒；"
            f"{int(np.ceil(visible_channels / grid_columns))}×{grid_columns} 排列；"
            f"HDF5 切片步长 {context['step']}（每通道 {values.shape[0]:,} 点）{cache_note}"
        )
        self._update_overview_display_summary()
        self._prefetch_neighbour_pages(context["page"])

    def _ensure_plot_grid(self, visible_channels: int, grid_columns: int) -> None:
        signature = (visible_channels, grid_columns)
        if signature == self._grid_signature:
            return
        self.graphics.clear()
        self._plot_curves = []
        for index in range(visible_channels):
            row, column = divmod(index, grid_columns)
            plot = self.graphics.addPlot(row=row, col=column)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setLabel("bottom", "时间", units="s")
            # The overview keeps its current numeric values; this corrects
            # only the displayed unit.
            plot.setLabel("left", "幅值", units="μV")
            curve = plot.plot(pen=pg.mkPen("k", width=0.7), antialias=False)
            plot.getAxis("bottom").setStyle(showValues=True)
            plot.getAxis("left").setStyle(showValues=True)
            self._plot_curves.append((plot, curve))
        self._grid_signature = signature

    def _prefetch_neighbour_pages(self, page: int) -> None:
        if self.total_pages <= 1:
            return
        for neighbour in ((page + 1) % self.total_pages, (page - 1) % self.total_pages):
            if neighbour != page:
                self._queue_page_load(self._page_context(neighbour))

    def _open_detail_from_scene(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._detail_dialog is not None and self._detail_dialog.isVisible():
            return
        for item, channel in self._page_plots:
            if item.sceneBoundingRect().contains(event.scenePos()):
                if self.selection_mode:
                    channel_id = int(self.source.metadata.channel_ids[channel])
                    if channel_id in self._selected_channel_ids:
                        self._selected_channel_ids.remove(channel_id)
                    else:
                        self._selected_channel_ids.add(channel_id)
                    self._sync_selection_text()
                    return
                self.open_channel_detail(channel)
                return

    def open_channel_detail(self, channel: int) -> None:
        meta = self.source.metadata
        if meta is None:
            return
        if self._detail_dialog is not None and self._detail_dialog.isVisible():
            self._detail_dialog.raise_()
            self._detail_dialog.activateWindow()
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"通道 ch{meta.channel_ids[channel]} 详细预览")
        dialog.resize(1280, 760)
        layout = QVBoxLayout(dialog)
        plot = pg.PlotWidget()
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "时间", units="s")
        plot.setLabel("left", "幅值", units="μV")
        start, end = _seconds_window_to_samples(
            meta,
            self.start_spin.value(),
            self.end_spin.value(),
            full_duration=self.full_duration_check.isChecked(),
        )
        points = max(1, end - start)
        step = max(1, int(np.ceil(points / 200_000)))
        values = self.source.read(start, end, channel, step=step).ravel()
        times = meta.time_offset + np.arange(start, end, step, dtype=float)[:values.size] / meta.fs
        plot.plot(times, values, pen=pg.mkPen("#1f77b4", width=0.8), antialias=False)
        layout.addWidget(QLabel(f"实际通道 ch{meta.channel_ids[channel]}（列 {channel + 1}）；鼠标滚轮缩放，左键拖动平移；显示步长 {step}"))
        layout.addWidget(plot, 1)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._detail_dialog = dialog
        dialog.finished.connect(self._clear_detail_dialog)
        dialog.show()

    def _clear_detail_dialog(self, _result: int) -> None:
        self._detail_dialog = None


class ChannelSelectionDialog(QDialog):
    """Legacy-style 20×26 physical-channel selector.

    It stores actual channel IDs, never visual positions.  That distinction
    prevents a reloaded custom export (for example only ch500) from being
    previewed or filtered as an unrelated first-column channel.
    """

    def __init__(self, channel_ids, selected_ids=(), parent=None, title="选择通道（20×26）", layout_ids=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1180, 720)
        self._available = {int(value) for value in channel_ids}
        self._selected = {int(value) for value in selected_ids if int(value) in self._available}
        positions = np.asarray(layout_ids if layout_ids is not None else np.arange(1, 521), dtype=np.int64).ravel()
        positions = positions[:520]
        if positions.size < 520:
            positions = np.pad(positions, (0, 520 - positions.size))
        # An incomplete Excel map must not hide selectable channels.  Fill
        # only empty/duplicate positions with IDs absent from the grid.
        seen = set()
        for index, value in enumerate(positions):
            if value > 0 and int(value) not in seen:
                seen.add(int(value))
            else:
                positions[index] = 0
        missing = [value for value in sorted(self._available) if value not in seen]
        for index in np.flatnonzero(positions == 0):
            if not missing:
                break
            positions[index] = missing.pop(0)
        layout = QVBoxLayout(self)
        note = QLabel("红色为可选通道；灰色为当前文件未加载的通道。点击单元格勾选；“全选”再次点击会全部取消。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(20, 26)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.setStyleSheet("QTableWidget::item { padding: 2px; }")
        for index in range(20 * 26):
            row, column = divmod(index, 26)
            channel_id = int(positions[index])
            item = QTableWidgetItem(str(channel_id) if channel_id else "—")
            item.setData(Qt.ItemDataRole.UserRole, channel_id)
            if channel_id in self._available:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if channel_id in self._selected else Qt.CheckState.Unchecked)
                item.setForeground(pg.mkColor("#b71c1c"))
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(pg.mkColor("#9e9e9e"))
            self.table.setItem(row, column, item)
        layout.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        self.select_all_button = QPushButton("全选 / 全不选")
        self.select_all_button.clicked.connect(self._toggle_all)
        buttons.addWidget(self.select_all_button)
        buttons.addStretch(1)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        confirm = QPushButton("确认选择并用于处理")
        confirm.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)

    def _toggle_all(self) -> None:
        available_items = [self.table.item(row, column) for row in range(20) for column in range(26)
                           if self.table.item(row, column) and self.table.item(row, column).isSelected() is False
                           and self.table.item(row, column).flags() & Qt.ItemFlag.ItemIsEnabled]
        all_checked = bool(available_items) and all(item.checkState() == Qt.CheckState.Checked for item in available_items)
        state = Qt.CheckState.Unchecked if all_checked else Qt.CheckState.Checked
        for item in available_items:
            item.setCheckState(state)

    def selected_channel_ids(self) -> list[int]:
        return [
            int(item.data(Qt.ItemDataRole.UserRole))
            for row in range(20) for column in range(26)
            if (item := self.table.item(row, column)) is not None and item.flags() & Qt.ItemFlag.ItemIsEnabled
            and item.checkState() == Qt.CheckState.Checked
        ]


class BinParseWorker(QThread):
    """Qt-thread wrapper around the original BIN → compressed-H5 algorithm."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(str, object)
    failed = pyqtSignal(str)

    def __init__(self, bin_path: str, output_path: str, start_sec: float, duration_sec: float, metadata: dict):
        super().__init__()
        self.bin_path = Path(bin_path)
        self.output_path = Path(output_path)
        self.start_sec = float(start_sec)
        self.duration_sec = float(duration_sec)
        self.metadata = dict(metadata)

    def run(self) -> None:
        try:
            self.progress.emit(0.0, "正在检查 BIN 帧布局…")
            layout = inspect_bin(self.bin_path)
            reader = BinReader(layout, self.start_sec, self.duration_sec, chunk_rows=4096)
            metadata = {
                "source_bin": str(self.bin_path.resolve()),
                "start_sec": self.start_sec,
                "duration_sec": reader.selected_duration_sec,
                **parse_filename_metadata(self.bin_path),
                **read_timing_metadata(self.bin_path),
                **self.metadata,
            }
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            result = create_h5(
                self.output_path, reader, metadata, compression_level=5,
                progress_callback=lambda done, total: self.progress.emit(
                    100.0 * done / max(1, total), f"正在解析 BIN：{done:,}/{total:,} 帧"
                ),
            )
            self.completed.emit(str(self.output_path), result)
        except Exception as exc:
            self.failed.emit(str(exc))


class H5SliceWorker(QThread):
    """Stream a selected HDF5 time window into a self-contained new H5."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(str, object)
    failed = pyqtSignal(str)

    def __init__(self, source_path: str, output_path: str, start_sec: float, duration_sec: float):
        super().__init__()
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.start_sec = float(start_sec)
        self.duration_sec = float(duration_sec)

    @staticmethod
    def _copy_attrs(source, destination) -> None:
        for key, value in source.attrs.items():
            destination.attrs[key] = value

    def run(self) -> None:
        temporary = self.output_path.with_name(self.output_path.name + ".partial")
        try:
            source_model = LazyH5Source()
            meta = source_model.open(self.source_path, 0.0, 0.0)
            start_row = max(0, min(meta.rows, int(round(self.start_sec * meta.fs))))
            end_row = meta.rows if self.duration_sec <= 0 else min(
                meta.rows, start_row + int(round(self.duration_sec * meta.fs))
            )
            if end_row <= start_row:
                raise ValueError("H5 截取时间窗没有包含任何采样点。")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.output_path.exists() or temporary.exists():
                raise FileExistsError(f"输出文件已存在：{self.output_path.name}")

            with h5py.File(self.source_path, "r") as source, h5py.File(temporary, "w") as target:
                self._copy_attrs(source, target)

                def copy_group(source_group, target_group):
                    self._copy_attrs(source_group, target_group)
                    for name, item in source_group.items():
                        if isinstance(item, h5py.Group):
                            child = target_group.create_group(name)
                            copy_group(item, child)
                            continue
                        is_primary = item.name.lstrip("/") == str(meta.dataset).lstrip("/")
                        is_row_aligned = item.ndim >= 1 and int(item.shape[0]) == meta.rows
                        if not (is_primary or is_row_aligned):
                            source.copy(item, target_group, name=name)
                            continue
                        new_shape = (end_row - start_row,) + item.shape[1:]
                        kwargs = {}
                        if item.ndim >= 2 and np.issubdtype(item.dtype, np.number):
                            chunk_rows = min(max(1, int(item.chunks[0] if item.chunks else 100000)), new_shape[0])
                            kwargs.update(chunks=(chunk_rows,) + item.shape[1:])
                            kwargs.update(_legacy_h5_compression_kwargs())
                        dataset = target_group.create_dataset(name, shape=new_shape, dtype=item.dtype, **kwargs)
                        self._copy_attrs(item, dataset)
                        block_rows = max(1, int(dataset.chunks[0] if dataset.chunks else 100000))
                        for output_first in range(0, new_shape[0], block_rows):
                            output_last = min(new_shape[0], output_first + block_rows)
                            dataset[output_first:output_last] = item[
                                start_row + output_first:start_row + output_last
                            ]
                            if is_primary:
                                self.progress.emit(
                                    100.0 * output_last / new_shape[0],
                                    f"正在截取 H5：{output_last:,}/{new_shape[0]:,} 采样点",
                                )

                copy_group(source, target)
                target.attrs["source_h5"] = str(self.source_path.resolve())
                target.attrs["slice_start_sec"] = float(start_row / meta.fs)
                target.attrs["slice_duration_sec"] = float((end_row - start_row) / meta.fs)
                target.attrs["time_offset_sec"] = float(meta.time_offset + start_row / meta.fs)
                provenance = read_h5_provenance(source)
                if provenance:
                    updated = derive_h5_provenance(
                        provenance, stage=_provenance_stage(meta, "time_sliced"), fs=meta.fs,
                        unit="mV", channel_ids=meta.channel_ids,
                        operation={"name": "time_slice", "start_sec": start_row / meta.fs,
                                   "duration_sec": (end_row - start_row) / meta.fs},
                    )
                    write_h5_provenance(target, updated)
            os.replace(temporary, self.output_path)
            self.completed.emit(str(self.output_path), {
                "samples": end_row - start_row,
                "seconds": (end_row - start_row) / meta.fs,
                "start_sec": start_row / meta.fs,
            })
        except Exception as exc:
            if temporary.exists():
                temporary.unlink()
            self.failed.emit(str(exc))


class BatchBinParseWorker(QThread):
    """Recursive legacy batch parser with file and frame weighted progress."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, input_root: str, output_root: str):
        super().__init__()
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)

    def run(self) -> None:
        try:
            files = sorted(self.input_root.rglob("*.bin"))
            if not files:
                raise ValueError("所选文件夹中没有 BIN 文件。")
            successes, failures = [], []
            for index, path in enumerate(files, start=1):
                start_progress = 100.0 * (index - 1) / len(files)
                span = 100.0 / len(files)
                try:
                    info = parse_filename_metadata(path)
                    date = str(info.get("date") or "unknown").replace("-", "")
                    animal = str(info.get("animal") or 0)
                    output = self.output_root / date / animal / f"{path.stem}.h5"
                    self.progress.emit(start_progress, f"批量解析 {index}/{len(files)}：{path.name}")
                    layout = inspect_bin(path); reader = BinReader(layout, 0.0, 0.0, chunk_rows=4096)
                    meta = {**info, **read_timing_metadata(path), "start_sec": 0.0, "duration_sec": reader.selected_duration_sec}
                    output.parent.mkdir(parents=True, exist_ok=True)
                    create_h5(output, reader, meta, compression_level=5, progress_callback=lambda done, total, base=start_progress, width=span, name=path.name: self.progress.emit(base + width * done / max(1, total), f"批量解析 {index}/{len(files)}：{name}（{done:,}/{total:,} 帧）"))
                    successes.append(str(output))
                except Exception as exc:
                    failures.append(f"{path}: {exc}")
                self.progress.emit(100.0 * index / len(files), f"批量解析 {index}/{len(files)} 完成")
            self.completed.emit(successes, failures)
        except Exception as exc:
            self.failed.emit(str(exc))


class RawChannelExportWorker(QThread):
    """Write selected physical channels as reloadable one-channel raw H5 files."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, channel_ids, directory):
        super().__init__()
        self.source = source
        self.channel_ids = tuple(int(channel) for channel in channel_ids)
        self.directory = Path(directory)

    def run(self) -> None:
        try:
            meta = self.source.metadata
            if meta is None:
                raise RuntimeError("No loaded source is available for channel export.")
            columns_by_id = {int(channel): index for index, channel in enumerate(meta.channel_ids)}
            missing = [channel for channel in self.channel_ids if channel not in columns_by_id]
            if missing:
                raise ValueError("Requested physical channels are not loaded: " + ", ".join(map(str, missing)))
            self.directory.mkdir(parents=True, exist_ok=True)
            source_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(meta.path).stem or "recording").strip("._") or "recording"
            paths = [self.directory / f"{source_stem}_ch{channel:03d}_raw.h5" for channel in self.channel_ids]
            existing = [path.name for path in paths if path.exists()]
            if existing:
                raise FileExistsError("Refusing to overwrite existing channel H5: " + ", ".join(existing[:8]))
            chunk_rows = _export_chunk_rows(meta)
            string_dtype = h5py.string_dtype(encoding="utf-8")
            written = []
            total = max(1, len(self.channel_ids) * meta.rows)
            completed_rows = 0
            for position, (channel_id, path) in enumerate(zip(self.channel_ids, paths), start=1):
                column = columns_by_id[channel_id]
                with h5py.File(path, "w") as h5:
                    _write_legacy_compression_metadata(h5)
                    h5.attrs["format"] = "SD_raw_channel_hdf5"
                    h5.attrs["data_stage"] = "raw"
                    h5.attrs["storage_dtype"] = "float32"
                    h5.attrs["hdf5_chunk_shape"] = f"({chunk_rows},)"
                    signal_out = h5.create_dataset("signal", shape=(meta.rows,), dtype=np.float32, chunks=(chunk_rows,), **_legacy_h5_compression_kwargs())
                    time_out = h5.create_dataset("time", shape=(meta.rows,), dtype=np.float64, chunks=(chunk_rows,), **_legacy_h5_compression_kwargs())
                    for first in range(0, meta.rows, chunk_rows):
                        last = min(meta.rows, first + chunk_rows)
                        signal_out[first:last] = self.source.read(first, last, column)
                        time_out[first:last] = meta.time_offset + np.arange(first, last, dtype=np.float64) / meta.fs
                        completed_rows += last - first
                        self.progress.emit(100.0 * completed_rows / total, f"正在导出 ch{channel_id}（{position}/{len(self.channel_ids)}）")
                    h5.create_dataset("FS", data=np.asarray(meta.fs, dtype=np.float64))
                    h5.create_dataset("channel", data=np.asarray(channel_id, dtype=np.int32))
                    h5.create_dataset("physical_channel_id", data=np.asarray(channel_id, dtype=np.int32))
                    h5.create_dataset("channel_ids", data=np.asarray([channel_id], dtype=np.int64))
                    h5.create_dataset("column_index", data=np.asarray(column, dtype=np.int32))
                    h5.create_dataset("source_path", data=str(meta.path), dtype=string_dtype)
                    h5.create_dataset("data_unit", data="mV", dtype=string_dtype)
                    h5.attrs["time_offset_sec"] = float(meta.time_offset)
                    h5.attrs["channel_mapping_version"] = 2
                    _write_timing_metadata(h5, meta.timing_metadata)
                    _write_output_provenance(
                        h5, meta, stage=_provenance_stage(meta, "raw"), channel_ids=[channel_id],
                        operation={"name": "export_channel", "format": "SD_raw_channel_hdf5"},
                    )
                written.append(str(path))
            self.completed.emit(written)
        except Exception as exc:
            self.failed.emit(str(exc))


class CooperativeTaskCancelled(RuntimeError):
    """Raised at a safe checkpoint after a shared task is cancelled."""


class CooperativeWorker(QThread):
    """QThread with shared pause/cancel controls for preprocessing tasks."""

    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._control_condition = threading.Condition()
        self._paused = False
        self._cancel_requested = False

    def set_paused(self, paused: bool) -> None:
        with self._control_condition:
            self._paused = bool(paused)
            if not self._paused:
                self._control_condition.notify_all()

    def cancel(self) -> None:
        with self._control_condition:
            self._cancel_requested = True
            self._paused = False
            self._control_condition.notify_all()

    def _checkpoint(self) -> None:
        with self._control_condition:
            while self._paused and not self._cancel_requested:
                self._control_condition.wait(timeout=0.25)
            if self._cancel_requested:
                raise CooperativeTaskCancelled()


class ChannelReplacementWorker(CooperativeWorker):
    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = dict(kwargs)

    def run(self) -> None:
        try:
            self.completed.emit(replace_preprocessed_channels(
                **self.kwargs,
                checkpoint=self._checkpoint,
                progress=lambda value, message: self.progress.emit(value, message),
            ))
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class PreprocessCancelled(RuntimeError):
    pass


class PreprocessWorker(QThread):
    """Run unavoidable full-record remapping/filtering away from the Qt loop."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object, object, bool, object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, source: LazyH5Source, remap_path: str | None, mode: str, low: float, high: float, **settings):
        super().__init__()
        self.source = source
        self.remap_path = remap_path
        self.mode = mode
        self.low = low
        self.high = high
        self.settings = settings
        self._control_condition = threading.Condition()
        self._paused = False
        self._cancel_requested = False

    def set_paused(self, paused: bool) -> None:
        with self._control_condition:
            self._paused = bool(paused)
            if not self._paused:
                self._control_condition.notify_all()

    def cancel(self) -> None:
        with self._control_condition:
            self._cancel_requested = True
            self._paused = False
            self._control_condition.notify_all()

    def _checkpoint(self) -> None:
        with self._control_condition:
            while self._paused and not self._cancel_requested:
                self._control_condition.wait(timeout=0.25)
            if self._cancel_requested:
                raise PreprocessCancelled()

    def _report_progress(self, value: float, message: str) -> None:
        self._checkpoint()
        self.progress.emit(float(value), str(message))

    def run(self) -> None:
        try:
            self._checkpoint()
            meta = self.source.metadata
            if meta is None:
                raise RuntimeError("请先加载 HDF5 数据。")
            did_remap = bool(self.remap_path) or bool(self.settings.get("already_remapped", False))
            if self.remap_path:
                working_data, working_cache, mapping, ids, remapped_ids = remap_source_from_excel(
                    self.source,
                    self.remap_path,
                    lambda value, message: self._report_progress(value * 65.0, message),
                )
                working_source = ArraySource(
                    working_data, meta.fs, meta.time_offset, label="remapped",
                    storage_path=working_cache,
                    channel_ids=remapped_ids,
                    timing_metadata=meta.timing_metadata,
                    provenance=derive_h5_provenance(
                        getattr(meta, "provenance", None), stage="remapped", fs=meta.fs, unit="mV",
                        channel_ids=remapped_ids,
                        source=None if getattr(meta, "provenance", None) else {"path": str(meta.path), "dataset": str(meta.dataset)},
                        operation={
                            "name": "channel_remap", "mapping_file": str(self.remap_path),
                            "source_channel_ids": ids,
                            "destination_for_source_channel_ids": mapping,
                            "destination_channel_ids": remapped_ids,
                        },
                    ),
                )
                # Channel selection is made against the input's physical IDs.
                # Remapping moves those inputs to destination columns, so the
                # filter must receive those destination columns instead.
                input_columns = self.settings.get("channels")
                if input_columns is None:
                    filter_columns = None
                else:
                    selected_destinations = mapping[np.asarray(input_columns, dtype=np.int64)]
                    output_column_by_id = {
                        int(channel_id): index
                        for index, channel_id in enumerate(remapped_ids)
                    }
                    filter_columns = np.unique([
                        output_column_by_id[int(channel_id)]
                        for channel_id in selected_destinations
                    ])
                filter_progress_start, filter_progress_span = 65.0, 35.0
            else:
                # Filtering is valid without an Excel map.  Materialize only
                # because zero-phase filtering needs complete channel data;
                # retain the original physical channel IDs and their order.
                working_data, working_cache = self.source.materialize(
                    lambda value, message: self._report_progress(value * 25.0, message)
                )
                working_source = ArraySource(
                    working_data, meta.fs, meta.time_offset, label="unmapped",
                    storage_path=working_cache, channel_ids=meta.channel_ids,
                    timing_metadata=meta.timing_metadata,
                    provenance=getattr(meta, "provenance", None),
                )
                filter_columns = self.settings.get("channels")
                filter_progress_start, filter_progress_span = 25.0, 75.0
            baseline_source = self.settings.get("filter_baseline_source")
            filter_active = (
                str(self.mode).strip().lower() != "off"
                or bool(self.settings.get("notch", False))
            )
            filter_channel_input = None
            if baseline_source is not None and filter_active:
                baseline_meta = getattr(baseline_source, "metadata", None)
                working_meta = working_source.metadata
                if baseline_meta is None:
                    raise ValueError("无法读取重新滤波所需的原始基准数据。")
                if baseline_meta.rows != working_meta.rows or not np.isclose(baseline_meta.fs, working_meta.fs):
                    raise ValueError("原始基准数据与当前处理数据的采样点数或采样率不一致。")
                reset_columns = (
                    np.arange(working_meta.channels, dtype=np.int64)
                    if filter_columns is None
                    else np.asarray(filter_columns, dtype=np.int64).ravel()
                )
                baseline_by_id = {
                    int(channel_id): index
                    for index, channel_id in enumerate(baseline_meta.channel_ids)
                }
                reset_ids = [int(working_meta.channel_ids[index]) for index in reset_columns]
                missing_ids = [channel_id for channel_id in reset_ids if channel_id not in baseline_by_id]
                if missing_ids:
                    raise ValueError(
                        "原始基准数据缺少待重新滤波通道："
                        + ", ".join(f"ch{channel_id}" for channel_id in missing_ids[:12])
                    )
                baseline_columns = np.asarray(
                    [baseline_by_id[channel_id] for channel_id in reset_ids], dtype=np.int64,
                )
                filter_channel_input, baseline_cache = allocate_storage(
                    (working_meta.rows, reset_columns.size),
                    np.float32, "filter_baseline_channels",
                )
                chunk_rows = max(1, min(PROCESS_CHUNK_SAMPLES, working_meta.rows))
                for first in range(0, working_meta.rows, chunk_rows):
                    self._checkpoint()
                    last = min(working_meta.rows, first + chunk_rows)
                    baseline_values = baseline_source.read(first, last, baseline_columns)
                    if baseline_values.ndim == 1:
                        baseline_values = baseline_values[:, None]
                    filter_channel_input[first:last] = baseline_values
                if isinstance(filter_channel_input, np.memmap):
                    filter_channel_input.flush()
                working_source = ArraySource(
                    working_data, working_meta.fs, working_meta.time_offset,
                    label="filter-baseline-restored", storage_path=working_cache,
                    channel_ids=working_meta.channel_ids,
                    timing_metadata=working_meta.timing_metadata,
                    provenance=derive_h5_provenance(
                        working_meta.provenance,
                        stage="preprocessed", fs=working_meta.fs, unit="mV",
                        channel_ids=working_meta.channel_ids,
                        operation={
                            "name": "filter_baseline_restore",
                            "channel_ids": reset_ids,
                        },
                    ),
                )
            processed, processed_cache = filter_array(
                working_data,
                meta.fs,
                self.mode,
                self.low,
                self.high,
                lambda value, message: self._report_progress(
                    filter_progress_start + value * filter_progress_span, message
                ),
                highpass_order=self.settings.get("highpass_order", 3),
                lowpass_order=self.settings.get("lowpass_order", 5),
                notch=self.settings.get("notch", False),
                notch_frequency=self.settings.get("notch_frequency", 50.0),
                notch_q=self.settings.get("notch_q", 30.0),
                notch_harmonics=self.settings.get("notch_harmonics", 1),
                channels=filter_columns,
                channel_input=filter_channel_input,
                # Ordinary “执行滤波” uses one extra compute thread while all
                # writes remain on this preprocessing worker.  filter_array
                # automatically falls back to one thread for large records.
                parallel_workers=2,
                parallel_memory_budget_bytes=512 * 1024 * 1024,
            )
            processed_source = (
                working_source
                if processed is working_data
                else ArraySource(
                    processed,
                    meta.fs,
                    meta.time_offset,
                    label="preprocessed",
                    storage_path=processed_cache,
                    channel_ids=working_source.metadata.channel_ids,
                    timing_metadata=meta.timing_metadata,
                    provenance=derive_h5_provenance(
                        working_source.metadata.provenance, stage="preprocessed", fs=meta.fs, unit="mV",
                        channel_ids=working_source.metadata.channel_ids,
                        source=None if working_source.metadata.provenance else {"path": str(meta.path), "dataset": str(meta.dataset)},
                        operation={
                            "name": "filter", "mode": str(self.mode).strip().lower(),
                            "low_hz": float(self.low), "high_hz": float(self.high),
                            "highpass_order": int(self.settings.get("highpass_order", 3)),
                            "lowpass_order": int(self.settings.get("lowpass_order", 5)),
                            "notch_enabled": bool(self.settings.get("notch", False)),
                            "notch_frequency_hz": float(self.settings.get("notch_frequency", 50.0)),
                            "notch_q": float(self.settings.get("notch_q", 30.0)),
                            "notch_harmonics": int(self.settings.get("notch_harmonics", 1)),
                            "channel_ids": [
                                int(working_source.metadata.channel_ids[index])
                                for index in (np.arange(working_source.metadata.channels) if filter_columns is None else filter_columns)
                            ],
                        },
                    ),
                )
            )
            self._report_progress(100.0, "预处理完成")
            filter_active = str(self.mode).strip().lower() != "off" or bool(self.settings.get("notch", False))
            lowpass_mode = str(self.mode).strip().lower() in {"lowpass", "bandpass"}
            filtered_columns = filter_columns
            previous_filtered_ids = {
                int(value) for value in self.settings.get("previous_filtered_channel_ids", ())
            }
            previous_lowpass_ids = {
                int(value) for value in self.settings.get("previous_lowpass_channel_ids", ())
            }
            previous_notched_ids = {
                int(value) for value in self.settings.get("previous_notched_channel_ids", ())
            }
            if filter_active:
                if filtered_columns is None:
                    current_filtered_ids = set(int(value) for value in working_source.metadata.channel_ids)
                else:
                    current_filtered_ids = {
                        int(working_source.metadata.channel_ids[index])
                        for index in np.asarray(filtered_columns, dtype=np.int64)
                        if 0 <= int(index) < working_source.metadata.channels
                    }
                filtered_ids = previous_filtered_ids | current_filtered_ids
            else:
                current_filtered_ids = set()
                filtered_ids = previous_filtered_ids
            if lowpass_mode:
                if filtered_columns is None:
                    current_lowpass_ids = set(int(value) for value in working_source.metadata.channel_ids)
                else:
                    current_lowpass_ids = {
                        int(working_source.metadata.channel_ids[index])
                        for index in np.asarray(filtered_columns, dtype=np.int64)
                        if 0 <= int(index) < working_source.metadata.channels
                    }
            else:
                current_lowpass_ids = set()
            current_notched_ids = (
                current_filtered_ids
                if bool(self.settings.get("notch", False)) else set()
            )
            if baseline_source is not None and filter_active:
                # Selected channels were restored before this run, so their
                # former low-pass/notch status must be replaced by the new mode.
                lowpass_ids = (
                    previous_lowpass_ids - current_filtered_ids
                ) | current_lowpass_ids
                notched_ids = (
                    previous_notched_ids - current_filtered_ids
                ) | current_notched_ids
            else:
                lowpass_ids = previous_lowpass_ids | current_lowpass_ids
                notched_ids = previous_notched_ids | current_notched_ids
            self._checkpoint()
            self.completed.emit(working_source, processed_source, did_remap, {
                "lowpass_channel_ids": lowpass_ids,
                "notched_channel_ids": notched_ids,
                "notch_skipped_channel_ids": set(
                    map(int, self.settings.get("notch_skipped_channel_ids", ()))
                ),
                "filtered_channel_ids": filtered_ids,
                "lowpass_high_hz": float(self.high) if lowpass_mode else None,
            })
        except PreprocessCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class FilterOverlapTestWorker(QThread):
    """Compare overlap-and-crop filtering with a whole-segment filtfilt reference."""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, columns, scenarios):
        super().__init__()
        self.source = source
        self.columns = np.asarray(columns, dtype=np.int64)
        self.scenarios = [dict(item) for item in scenarios]

    @staticmethod
    def _filter(values, fs, settings):
        from scipy import signal

        mode = str(settings["mode"]).lower()
        low, high = float(settings["low"]), float(settings["high"])
        nyquist = fs / 2.0
        filters = []
        if mode == "highpass":
            filters.append(signal.butter(int(settings["hp_order"]), low, btype="highpass", fs=fs, output="sos"))
        elif mode == "lowpass":
            filters.append(signal.butter(int(settings["lp_order"]), high, btype="lowpass", fs=fs, output="sos"))
        elif mode == "bandpass":
            filters.append(signal.butter(int(settings["hp_order"]), low, btype="highpass", fs=fs, output="sos"))
            filters.append(signal.butter(int(settings["lp_order"]), high, btype="lowpass", fs=fs, output="sos"))
        if settings["notch"]:
            for harmonic in range(1, int(settings["notch_harmonics"]) + 1):
                frequency = float(settings["notch_frequency"]) * harmonic
                if frequency >= nyquist:
                    break
                b, a = signal.iirnotch(frequency, float(settings["notch_q"]), fs=fs)
                filters.append(signal.tf2sos(b, a))
        output = np.asarray(values, dtype=np.float64)
        for sos in filters:
            output = signal.sosfiltfilt(sos, output, axis=0)
        return output

    def run(self):
        try:
            meta = self.source.metadata
            fs = float(meta.fs)
            total_duration = meta.rows / fs
            core_sec = min(10.0, total_duration / 3.0)
            if core_sec < 1.0:
                raise ValueError("记录过短；重叠测试至少需要约 3 秒数据。")
            core_samples = max(3, int(round(core_sec * fs)))
            rows, test_ranges, skipped = [], {}, []
            channel_ids = [int(meta.channel_ids[index]) for index in self.columns]
            for settings in self.scenarios:
                name = str(settings["name"])
                if float(settings["high"]) >= fs / 2:
                    raise ValueError(f"{name} 的高频 {settings['high']:g} Hz 不低于奈奎斯特频率 {fs / 2:g} Hz。")
                requested_candidates = np.asarray(settings["overlap_candidates_sec"], dtype=float)
                available_side = max(0.0, (total_duration - core_sec) / 2.0)
                required_side = requested_candidates + np.maximum(5.0, requested_candidates * .25)
                available_mask = required_side <= available_side + 1e-9
                candidates = requested_candidates[available_mask]
                for overlap, side in zip(requested_candidates[~available_mask], required_side[~available_mask]):
                    skipped.append({
                        "branch": name, "overlap_sec": float(overlap),
                        "required_duration_sec": float(core_sec + 2 * side),
                        "available_duration_sec": float(total_duration),
                    })
                if not candidates.size:
                    continue
                reference_margin = max(5.0, float(candidates[-1]) * .25)
                test_duration = min(total_duration, core_sec + 2 * (float(candidates[-1]) + reference_margin))
                first = max(0, int(round((total_duration - test_duration) * .5 * fs)))
                last = min(meta.rows, first + int(round(test_duration * fs)))
                values = self.source.read(first, last, self.columns)
                if values.ndim == 1:
                    values = values[:, None]
                reference = self._filter(values, fs, settings)
                core_first = (values.shape[0] - core_samples) // 2
                core_last = core_first + core_samples
                reference_core = reference[core_first:core_last]
                test_ranges[name] = (float(meta.time_offset + first / fs), float(meta.time_offset + last / fs))
                for overlap_sec in candidates:
                    overlap_samples = int(round(float(overlap_sec) * fs))
                    window_first = max(0, core_first - overlap_samples)
                    window_last = min(values.shape[0], core_last + overlap_samples)
                    cropped = self._filter(values[window_first:window_last], fs, settings)[
                        core_first - window_first:core_last - window_first
                    ]
                    residual = cropped - reference_core
                    for channel_index, channel_id in enumerate(channel_ids):
                        ref = reference_core[:, channel_index]
                        error = residual[:, channel_index]
                        finite = np.isfinite(ref) & np.isfinite(error)
                        if not np.any(finite):
                            rmse = maximum = relative = correlation = np.nan
                        else:
                            rmse = float(np.sqrt(np.mean(error[finite] ** 2)))
                            maximum = float(np.max(np.abs(error[finite])))
                            reference_rms = float(np.sqrt(np.mean(ref[finite] ** 2)))
                            relative = 100.0 * rmse / max(reference_rms, np.finfo(float).eps)
                            correlation = float(np.corrcoef(ref[finite], (ref + error)[finite])[0, 1]) if finite.sum() > 2 else np.nan
                        rows.append({
                            "branch": name, "channel": channel_id, "overlap_sec": float(overlap_sec),
                            "rmse_mv": rmse, "max_abs_residual_mv": maximum,
                            "relative_rmse_percent": relative, "correlation": correlation,
                        })
            if not rows:
                raise ValueError("当前记录过短，没有任何候选重叠长度能够保留独立参考边距。")
            self.completed.emit({
                "rows": rows, "test_ranges": test_ranges,
                "skipped": skipped,
                "channel_ids": channel_ids, "fs": fs, "core_sec": core_sec,
                "scenarios": self.scenarios,
            })
        except Exception as exc:
            self.failed.emit(str(exc))


class DualBranchStreamWorker(CooperativeWorker):
    """Stream one reviewed source into complete LFP and Spike HDF5 products."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, output_dir, base_name, good_ids, qc_snapshot, branch_settings=None, apply_lfp_car=False):
        super().__init__()
        self.source = source
        self.output_dir = Path(output_dir)
        self.base_name = str(base_name)
        self.good_ids = {int(value) for value in good_ids}
        self.qc_snapshot = dict(qc_snapshot)
        self.branch_settings = deepcopy(branch_settings or {})
        self.apply_lfp_car = bool(apply_lfp_car)
        runtime = self.branch_settings.get("runtime", {})
        self.max_filter_workers = max(1, int(runtime.get("max_workers", 24)))
        self.parallel_memory_budget_bytes = max(
            256 * 1024 * 1024,
            int(float(runtime.get("memory_budget_gb", 16.0)) * 1024 ** 3),
        )

    def _branch_settings(self, name):
        if name == "lfp":
            defaults = dict(mode="bandpass", low=.5, high=300.0, hp_order=3, lp_order=5,
                            notch=True, notch_frequency=50.0, notch_q=30.0, notch_harmonics=1,
                            overlap_sec=60.0)
        else:
            defaults = dict(mode="bandpass", low=500.0, high=3000.0, hp_order=3, lp_order=5,
                            notch=False, notch_frequency=50.0, notch_q=30.0, notch_harmonics=1,
                            overlap_sec=.25)
        defaults.update(self.branch_settings.get(name, {}))
        return defaults

    def _create_partial(self, path, branch, settings):
        meta = self.source.metadata
        chunk_rows = _export_chunk_rows(meta)
        h5 = h5py.File(path, "w")
        _write_legacy_compression_metadata(h5)
        channel_chunk = min(8, meta.channels)
        dataset = h5.create_dataset(
            "rawData512", shape=(meta.rows, meta.channels), dtype=np.float32,
            chunks=(chunk_rows, channel_chunk), **_legacy_h5_compression_kwargs(),
        )
        h5.attrs["hdf5_chunk_shape"] = f"({chunk_rows}, {channel_chunk})"
        h5["FS"] = meta.fs
        h5["data_unit"] = "mV"
        h5["time_offset_sec"] = meta.time_offset
        h5.create_dataset("channel_ids", data=np.asarray(meta.channel_ids, dtype=np.int64))
        _write_timing_metadata(h5, meta.timing_metadata)
        h5.attrs["data_stage"] = "preprocessed"
        h5.attrs["processing_branch"] = branch
        h5.attrs["processing_status"] = "partial"
        h5.attrs["processing_record_csv"] = f"{self.base_name}_processing.csv"
        h5.attrs["preprocess_qc_schema"] = self.qc_snapshot.get("schema", "sd-preprocess-qc")
        h5.attrs["preprocess_qc_version"] = int(self.qc_snapshot.get("version", 1))
        h5.attrs["preprocess_qc_json"] = json.dumps(self.qc_snapshot, ensure_ascii=False, sort_keys=True)
        return h5, dataset

    def _write_filtered_branch(self, branch, partial_path, progress_start, progress_span, input_source=None):
        input_source = input_source or self.source
        meta = input_source.metadata
        settings = self._branch_settings(branch)
        if settings["high"] >= meta.fs / 2:
            raise ValueError(f"{branch.upper()} 高频 {settings['high']:g} Hz 不低于奈奎斯特频率 {meta.fs / 2:g} Hz。")
        duration = meta.rows / meta.fs
        core_samples = meta.rows if duration <= 600.0 else max(1, int(round(600.0 * meta.fs)))
        overlap_samples = 0 if core_samples == meta.rows else int(round(settings["overlap_sec"] * meta.fs))
        settings["streaming"] = core_samples < meta.rows
        settings["core_chunk_sec"] = float(core_samples / meta.fs)
        settings["overlap_crop_sec"] = float(overlap_samples / meta.fs)
        cores = [(first, min(meta.rows, first + core_samples)) for first in range(0, meta.rows, core_samples)]
        # HDF5 reads and writes stay on this worker thread.  Only the CPU-heavy
        # scipy filtering runs concurrently; h5py objects are never shared with
        # the pool.  Size channel groups and the pool from a conservative memory
        # estimate so long, high-rate recordings do not create several huge
        # filtfilt work arrays at once.
        max_read_rows = min(meta.rows, core_samples + 2 * overlap_samples)
        bytes_per_value = np.dtype(np.float64).itemsize
        work_array_factor = 3
        target_task_bytes = 96 * 1024 * 1024
        memory_budget_bytes = self.parallel_memory_budget_bytes
        estimated_bytes_per_channel = max(1, max_read_rows * bytes_per_value * work_array_factor)
        channel_group_size = max(1, min(8, target_task_bytes // estimated_bytes_per_channel))
        channel_groups = [
            np.arange(first, min(meta.channels, first + channel_group_size), dtype=np.int64)
            for first in range(0, meta.channels, channel_group_size)
        ]
        estimated_task_bytes = max(1, estimated_bytes_per_channel * channel_group_size)
        parallel_workers = min(
            max(1, os.cpu_count() or 1), self.max_filter_workers, len(channel_groups),
            max(1, memory_budget_bytes // estimated_task_bytes),
        )
        settings["parallel_filter_workers"] = int(parallel_workers)
        settings["max_parallel_workers_requested"] = int(self.max_filter_workers)
        settings["channel_group_size"] = int(channel_group_size)
        settings["parallel_memory_budget_mb"] = int(memory_budget_bytes // (1024 * 1024))
        total_jobs = max(1, len(cores) * len(channel_groups))
        h5, target = self._create_partial(partial_path, branch, settings)
        try:
            job = 0
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                for core_index, (core_first, core_last) in enumerate(cores, start=1):
                    self._checkpoint()
                    read_first = max(0, core_first - overlap_samples)
                    read_last = min(meta.rows, core_last + overlap_samples)
                    # Submit in bounded batches.  This both overlaps serial HDF5
                    # reads with filtering and caps the number of resident arrays.
                    for batch_first in range(0, len(channel_groups), parallel_workers):
                        batch = channel_groups[batch_first:batch_first + parallel_workers]
                        futures = {}
                        for columns in batch:
                            self._checkpoint()
                            values = input_source.read(read_first, read_last, columns)
                            future = executor.submit(
                                FilterOverlapTestWorker._filter, values, float(meta.fs), settings,
                            )
                            futures[future] = columns
                        for future in as_completed(futures):
                            self._checkpoint()
                            columns = futures[future]
                            filtered = future.result()
                            target[core_first:core_last, columns] = np.asarray(
                                filtered[core_first - read_first:core_last - read_first], dtype=np.float32,
                            )
                            job += 1
                            self.progress.emit(
                                progress_start + progress_span * job / total_jobs,
                                f"流式生成 {branch.upper()}：时间块 {core_index}/{len(cores)}，"
                                f"通道至 ch{int(meta.channel_ids[columns[-1]])}，并行 {parallel_workers}",
                            )
            target.flush()
            operation = {
                "name": "filter", "branch": branch, **settings,
                "channel_ids": list(meta.channel_ids),
                "streaming": settings["streaming"],
                "core_chunk_sec": settings["core_chunk_sec"],
                "overlap_crop_sec": settings["overlap_crop_sec"],
            }
            provenance = derive_h5_provenance(
                getattr(meta, "provenance", None), stage="preprocessed", fs=meta.fs, unit="mV",
                channel_ids=meta.channel_ids,
                source=None if getattr(meta, "provenance", None) else {"path": str(meta.path), "dataset": str(meta.dataset)},
                operation=operation,
            )
            write_h5_provenance(h5, provenance)
            h5.attrs["processing_status"] = "filtered"
            h5.flush()
        finally:
            h5.close()
        return settings

    def _write_lfp_car_cache(self, path, progress_start, progress_span):
        meta = self.source.metadata
        good_columns = np.asarray([index for index, channel in enumerate(meta.channel_ids) if int(channel) in self.good_ids], dtype=np.int64)
        if good_columns.size < 2:
            raise ValueError("流式LFP的CAR至少需要两个已确认健康通道。")
        block_rows = max(1, min(50_000, meta.rows))
        with h5py.File(path, "w") as h5:
            chunk_rows = _export_chunk_rows(meta)
            channel_chunk = min(8, meta.channels)
            target = h5.create_dataset(
                "rawData512", shape=(meta.rows, meta.channels), dtype=np.float32,
                chunks=(chunk_rows, channel_chunk), **_legacy_h5_compression_kwargs(),
            )
            h5["FS"] = meta.fs; h5["data_unit"] = "mV"; h5["time_offset_sec"] = meta.time_offset
            h5.create_dataset("channel_ids", data=np.asarray(meta.channel_ids, dtype=np.int64))
            _write_timing_metadata(h5, meta.timing_metadata)
            for first in range(0, meta.rows, block_rows):
                self._checkpoint()
                last = min(meta.rows, first + block_rows)
                block = self.source.read(first, last, slice(None))
                good = block[:, good_columns]
                order = np.argsort(good, axis=1)
                sorted_values = np.take_along_axis(good, order, axis=1)
                remaining = good_columns.size - 1
                if remaining % 2:
                    middle = remaining // 2
                    for local_column, column in enumerate(good_columns):
                        rank = np.argmax(order == local_column, axis=1)
                        reference = np.where(rank <= middle, sorted_values[:, middle + 1], sorted_values[:, middle])
                        block[:, column] -= reference
                else:
                    lower, upper = remaining // 2 - 1, remaining // 2
                    for local_column, column in enumerate(good_columns):
                        rank = np.argmax(order == local_column, axis=1)
                        low_value = np.where(rank <= lower, sorted_values[:, lower + 1], sorted_values[:, lower])
                        high_value = np.where(rank < upper, sorted_values[:, upper + 1], sorted_values[:, upper])
                        block[:, column] -= (low_value + high_value) * .5
                target[first:last] = block
                self.progress.emit(progress_start + progress_span * last / meta.rows, f"LFP CAR：{last:,}/{meta.rows:,} 采样点")
            provenance = read_h5_provenance(h5)
            provenance = derive_h5_provenance(
                getattr(meta, "provenance", None), stage="median_car", fs=meta.fs, unit="mV", channel_ids=meta.channel_ids,
                operation={"name": "reference", "branch": "lfp", "method": "leave_one_out_median_car", "good_channel_ids": sorted(self.good_ids)},
            )
            write_h5_provenance(h5, provenance)

    def run(self):
        partial_paths = []
        car_cache = None
        try:
            self._checkpoint()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            lfp_final = self.output_dir / f"{self.base_name}_lfp_processed.h5"
            spike_final = self.output_dir / f"{self.base_name}_spike_processed.h5"
            lfp_partial = lfp_final.with_suffix(".partial.h5")
            spike_partial = spike_final.with_suffix(".partial.h5")
            partial_paths.extend((lfp_partial, spike_partial))
            if any(path.exists() for path in (lfp_final, spike_final, lfp_partial, spike_partial)):
                raise FileExistsError("目标处理文件已经存在；请重新启动导出以生成新的时间戳目录。")
            if self.apply_lfp_car:
                car_cache = self.output_dir / f".{self.base_name}_lfp_car_cache.partial.h5"
                partial_paths.append(car_cache)
                self._write_lfp_car_cache(car_cache, 0.0, 16.0)
                car_source = LazyH5Source(); car_source.open(car_cache)
                lfp_settings = self._write_filtered_branch("lfp", lfp_partial, 16.0, 39.0, input_source=car_source)
            else:
                lfp_settings = self._write_filtered_branch("lfp", lfp_partial, 0.0, 55.0)
            with h5py.File(lfp_partial, "r+") as h5:
                h5.attrs["processing_status"] = "complete"
            if car_cache is not None:
                car_cache.unlink()
                car_cache = None
            spike_settings = self._write_filtered_branch("spike", spike_partial, 55.0, 44.0)
            with h5py.File(spike_partial, "r+") as h5:
                h5.attrs["processing_status"] = "complete"
            os.replace(lfp_partial, lfp_final)
            os.replace(spike_partial, spike_final)
            provenance = getattr(self.source.metadata, "provenance", None) or {}
            bin_entry = next((
                item for item in reversed(provenance.get("lineage", []))
                if isinstance(item, dict) and str(item.get("kind", "")).lower() == "bin"
            ), {})
            manifest = {
                "schema": "sd-dual-branch-project", "version": 1,
                "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": str(self.source.metadata.path), "source_bin": str(bin_entry.get("path", "")),
                "rows": int(self.source.metadata.rows), "sampling_rate_hz": float(self.source.metadata.fs),
                "time_offset_sec": float(self.source.metadata.time_offset),
                "channel_ids": compact_integer_ranges(self.source.metadata.channel_ids),
                "qc_frozen": bool(self.qc_snapshot.get("completed", False)),
                "qc_skipped_for_test": bool(self.qc_snapshot.get("skipped_for_test", False)),
                "good_channel_ids": compact_integer_ranges(self.good_ids),
                "lfp_car_enabled": self.apply_lfp_car,
                "lfp_file": lfp_final.name, "spike_file": spike_final.name,
                "processing_csv": f"{self.base_name}_processing.csv",
                "lfp_processing": lfp_settings, "spike_processing": spike_settings,
            }
            manifest_path = self.output_dir / f"{self.base_name}_project.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            import csv
            ledger_path = self.output_dir / f"{self.base_name}_processing.csv"
            branch_records = [
                {
                    "branch": "LFP", "file": lfp_final.name, "filter": f"{lfp_settings['low']:g}-{lfp_settings['high']:g} Hz",
                    "notch": (f"{lfp_settings['notch_frequency']:g} Hz; Q={lfp_settings['notch_q']:g}; harmonics={lfp_settings['notch_harmonics']}" if lfp_settings["notch"] else "off"),
                    "reference": "leave-one-out median CAR" if self.apply_lfp_car else "none",
                    "core_chunk_sec": lfp_settings["core_chunk_sec"] if "core_chunk_sec" in lfp_settings else (600 if self.source.metadata.rows / self.source.metadata.fs > 600 else self.source.metadata.rows / self.source.metadata.fs),
                    "overlap_sec": lfp_settings["overlap_sec"],
                    "qc_frozen": bool(self.qc_snapshot.get("completed", False)),
                },
                {
                    "branch": "Spike", "file": spike_final.name, "filter": f"{spike_settings['low']:g}-{spike_settings['high']:g} Hz",
                    "notch": (f"{spike_settings['notch_frequency']:g} Hz; Q={spike_settings['notch_q']:g}; harmonics={spike_settings['notch_harmonics']}" if spike_settings["notch"] else "off"),
                    "reference": "none",
                    "core_chunk_sec": spike_settings["core_chunk_sec"] if "core_chunk_sec" in spike_settings else (600 if self.source.metadata.rows / self.source.metadata.fs > 600 else self.source.metadata.rows / self.source.metadata.fs),
                    "overlap_sec": spike_settings["overlap_sec"],
                    "qc_frozen": bool(self.qc_snapshot.get("completed", False)),
                },
            ]
            exported_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            source_file = str(bin_entry.get("path", "") or self.source.metadata.path)
            all_channel_ids = compact_integer_ranges(self.source.metadata.channel_ids)
            qc_rows = list(self.qc_snapshot.get("channel_rows", []))
            ledger_rows = []
            for branch_record, output_path in zip(branch_records, (lfp_final, spike_final)):
                with h5py.File(output_path, "r") as h5:
                    output_provenance = read_h5_provenance(h5) or {}
                operations = output_provenance.get("operations", [])
                common = {
                    "source_file": source_file,
                    "output_file": output_path.name,
                    "processed_at_utc": exported_utc,
                    "data_stage": "preprocessed",
                    "sampling_rate_hz": float(self.source.metadata.fs),
                    "channel_ids": all_channel_ids,
                    "filtered_channel_ids": all_channel_ids,
                    "processing_operations_json": json.dumps(
                        compact_provenance_operations(operations), ensure_ascii=True,
                        sort_keys=True, separators=(",", ":"),
                    ),
                    **branch_record,
                    "qc_completed": bool(self.qc_snapshot.get("completed", False)),
                }
                if qc_rows:
                    ledger_rows.extend({**common, **row} for row in qc_rows)
                else:
                    ledger_rows.append(common)
            qc_fields = [
                "channel", "final_status", "automatic_status", "automatic_reason",
                "brief_description", "manual_override", "manual_override_applied",
            ]
            fields = list(ledger_rows[0])
            fields.extend(field for field in qc_fields if field not in fields)
            with open(ledger_path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader(); writer.writerows(ledger_rows)
            self.progress.emit(100.0, "LFP、Spike及完整处理记录CSV已生成并完成校验")
            self.completed.emit({"lfp": lfp_final, "spike": spike_final, "manifest": manifest_path, "ledger": ledger_path})
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class LfpWorker(QThread):
    """Compute LFP SNR from the already-preprocessed current data source."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, signal_band: tuple[float, float], noise_band: tuple[float, float], analysis_mode: str = "resting", columns=None):
        super().__init__()
        self.source = source
        self.signal_band = signal_band
        self.noise_band = noise_band
        self.analysis_mode = str(analysis_mode or "resting").strip().lower()
        self.columns = None if columns is None else np.asarray(columns, dtype=np.int64).ravel()

    def run(self) -> None:
        try:
            if not hasattr(self.source, "data"):
                data, _ = self.source.materialize(
                    lambda value, message: self.progress.emit(value * 35.0, message)
                )
            else:
                data = self.source.data
            meta = self.source.metadata
            columns = np.arange(meta.channels, dtype=np.int64) if self.columns is None else self.columns
            if not columns.size or np.any(columns < 0) or np.any(columns >= meta.channels):
                raise ValueError("没有可用于 LFP SNR 的有效通道。")
            data = np.asarray(data[:, columns], dtype=np.float32)
            self.progress.emit(40.0, "正在计算全部通道 LFP SNR…")
            rows = compute_batch_channel_snr(
                data,
                meta.fs,
                mode=self.analysis_mode,
                signal_band=self.signal_band,
                noise_band=self.noise_band,
            )
            for index, row in enumerate(rows):
                row["channel"] = int(meta.channel_ids[columns[index]])
                row["channel_scope"] = "filtered_subset" if columns.size < meta.channels else "all_channels"
            self.progress.emit(100.0, "LFP SNR 计算完成")
            self.completed.emit(rows)
        except Exception as exc:
            self.failed.emit(str(exc))


class RestingMultibandWorker(QThread):
    """Compute resting multi-band SNR without changing the legacy LFP worker."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, columns=None, raw_source=None, settings=None, bad_ids=(), candidate_ids=()):
        super().__init__()
        self.source = source
        self.columns = None if columns is None else np.asarray(columns, dtype=np.int64).ravel()
        self.raw_source = raw_source if raw_source is not source else None
        self.settings = dict(settings or {})
        self.max_workers = max(1, int(self.settings.get("workers", 20)))
        self.bad_ids = {int(value) for value in bad_ids}
        self.candidate_ids = {int(value) for value in candidate_ids}

    @staticmethod
    def _quality_grade(channel_id, result, bad_ids, candidate_ids, settings):
        reasons = []
        line_max = float(result.get("post_line_max_db", np.nan))
        line_fraction = float(result.get("post_line_power_fraction", np.nan))
        valid_bands = [item for item in result["bands"].values() if item.get("available") and np.isfinite(item.get("snr_db", np.nan))]
        available_bands = [item for item in result["bands"].values() if item.get("available")]
        if channel_id in bad_ids:
            reasons.append("已有坏道检查拒绝")
        if channel_id in candidate_ids:
            reasons.append("已有通道质量候选判定")
        if np.isfinite(line_max) and line_max >= float(settings.get("line_reject_db", 10.0)):
            reasons.append(f"工频峰值 {line_max:.1f} dB")
        if np.isfinite(line_fraction) and line_fraction >= float(settings.get("line_reject_fraction", .10)):
            reasons.append(f"工频功率占比 {line_fraction:.1%}")
        if not valid_bands and available_bands:
            reasons.append("没有可用频段 SNR")
        elif len(valid_bands) < len(available_bands):
            reasons.append(f"仅 {len(valid_bands)}/{len(available_bands)} 个频段可用")
        warning_line = (
            np.isfinite(line_max) and line_max >= float(settings.get("line_warning_db", 3.0))
        ) or (
            np.isfinite(line_fraction) and line_fraction >= float(settings.get("line_warning_fraction", .03))
        )
        if warning_line and not any("工频" in reason for reason in reasons):
            reasons.append("工频残留偏高")
        if channel_id in bad_ids or (
            np.isfinite(line_max) and line_max >= float(settings.get("line_reject_db", 10.0))
        ) or (
            np.isfinite(line_fraction) and line_fraction >= float(settings.get("line_reject_fraction", .10))
        ) or (not valid_bands and available_bands):
            grade = "拒绝"
        elif reasons:
            grade = "需复核"
        else:
            grade = "可用"
        return grade, reasons

    @staticmethod
    def _flatten_line_metrics(row, prefix, metrics):
        for key, value in metrics.items():
            if isinstance(value, dict):
                row[f"{prefix}_{key}_power"] = float(value.get("power", np.nan))
                row[f"{prefix}_{key}_db"] = float(value.get("peak_db", np.nan))
                row[f"{prefix}_{key}_fraction"] = float(value.get("fraction", np.nan))
            else:
                row[f"{prefix}_{key}"] = float(value) if isinstance(value, (int, float, np.number)) else value

    def _materialize(self, source, progress_offset=0.0, progress_span=20.0):
        if hasattr(source, "data"):
            return np.asarray(source.data, dtype=np.float32)
        data, _ = source.materialize(
            lambda value, message: self.progress.emit(progress_offset + value * progress_span, message)
        )
        return np.asarray(data, dtype=np.float32)

    def run(self) -> None:
        try:
            meta = self.source.metadata
            if meta is None:
                raise RuntimeError("请先加载用于静息态 SNR 的数据。")
            columns = np.arange(meta.channels, dtype=np.int64) if self.columns is None else self.columns
            if not columns.size or np.any(columns < 0) or np.any(columns >= meta.channels):
                raise ValueError("没有可用于静息态多频段 SNR 的有效通道。")
            data = self._materialize(self.source, 0.0, 20.0)
            raw_data = None
            raw_columns = None
            if self.raw_source is not None and getattr(self.raw_source, "loaded", False):
                raw_meta = self.raw_source.metadata
                selected_ids = np.asarray(meta.channel_ids, dtype=np.int64)[columns]
                raw_columns = np.flatnonzero(np.isin(np.asarray(raw_meta.channel_ids, dtype=np.int64), selected_ids))
                if raw_columns.size == columns.size:
                    raw_data = self._materialize(self.raw_source, 20.0, 15.0)
                else:
                    raw_columns = None
            analysis_high_cutoff = self.settings.get("lowpass_high_hz")
            if analysis_high_cutoff is not None:
                analysis_high_cutoff = float(analysis_high_cutoff)
            line_settings = {
                "target_frequency_resolution": float(self.settings.get("target_frequency_resolution", .5)),
                "line_frequency": float(self.settings.get("line_frequency", 50.0)),
                "line_harmonics": int(self.settings.get("line_harmonics", 3)),
                "line_guard_hz": float(self.settings.get("line_guard_hz", 1.0)),
            }
            total = int(columns.size)
            raw_channel_lookup = (
                {int(channel_id): index for index, channel_id in enumerate(self.raw_source.metadata.channel_ids)}
                if raw_data is not None and raw_columns is not None else {}
            )

            def compute_channel(index: int, column: int):
                channel_id = int(meta.channel_ids[column])
                spectral = compute_resting_multiband_snr(data[:, column], meta.fs, **line_settings)
                row = {
                    "channel": channel_id,
                    "channel_scope": "filtered_subset" if columns.size < meta.channels else "all_channels",
                    "bands": spectral["bands"],
                    "total_power_1_200": spectral["total_power_1_200"],
                    "effective_total_bandwidth_hz": spectral["effective_total_bandwidth_hz"],
                    "nperseg": spectral["nperseg"], "noverlap": spectral["noverlap"],
                    "frequency_resolution_hz": spectral["frequency_resolution_hz"],
                    "line_guard_hz": spectral["line_guard_hz"],
                    "analysis_high_cutoff_hz": analysis_high_cutoff,
                    "quality_reasons": [],
                }
                if analysis_high_cutoff is not None:
                    for band in row["bands"].values():
                        if band["high_hz"] > analysis_high_cutoff + spectral["frequency_resolution_hz"] * .25:
                            band.update({"available": False, "snr_db": np.nan, "band_power": np.nan, "noise_power": np.nan, "effective_bandwidth_hz": 0.0})
                self._flatten_line_metrics(row, "post", spectral["line_metrics"])
                if raw_data is not None and raw_columns is not None:
                    raw_index = int(raw_channel_lookup[channel_id])
                    raw_spectral = compute_resting_multiband_snr(raw_data[:, raw_index], self.raw_source.metadata.fs, **line_settings)
                    self._flatten_line_metrics(row, "pre", raw_spectral["line_metrics"])
                    for key in spectral["line_metrics"]:
                        if key.startswith("line_") and isinstance(spectral["line_metrics"][key], dict):
                            pre_power = float(raw_spectral["line_metrics"][key].get("power", np.nan))
                            post_power = float(spectral["line_metrics"][key].get("power", np.nan))
                            row[f"{key}_suppression_db"] = float(10.0 * np.log10(pre_power / post_power)) if pre_power > 0 and post_power > 0 else np.nan
                    row["line_source_scope"] = "raw_current_paired"
                else:
                    row["line_source_scope"] = "post_only"
                grade, reasons = self._quality_grade(channel_id, row, self.bad_ids, self.candidate_ids, self.settings)
                row["quality_grade"] = grade
                row["quality_reasons"] = reasons
                return index, row

            rows = [None] * total
            workers = min(self.max_workers, total)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(compute_channel, index, int(column))
                    for index, column in enumerate(columns)
                ]
                for completed_count, future in enumerate(as_completed(futures), start=1):
                    index, row = future.result()
                    rows[index] = row
                    self.progress.emit(
                        35.0 + 65.0 * completed_count / max(1, total),
                        f"静息态多频段 SNR：{completed_count}/{total} 通道（{workers} 线程）",
                    )
            # A single high/low physiological band is not a bad-channel rule.
            # Flag only multi-band spectral profiles that are robust outliers
            # within this recording, and downgrade them to manual review.
            profile_outlier_z = float(self.settings.get("band_profile_outlier_z", 3.0))
            outlier_counts = {int(row["channel"]): [] for row in rows}
            for band_name, _, _ in RESTING_MULTIBAND_DEFINITIONS:
                values = np.asarray([row["bands"][band_name].get("snr_db", np.nan) for row in rows], dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size < 3:
                    continue
                center = float(np.nanmedian(finite))
                mad = float(np.nanmedian(np.abs(finite - center)))
                scale = max(1e-6, 1.4826 * mad)
                robust_z = np.abs(values - center) / scale
                for index, value in enumerate(robust_z):
                    if np.isfinite(value) and value >= profile_outlier_z:
                        outlier_counts[int(rows[index]["channel"])].append(band_name)
            for row in rows:
                outliers = outlier_counts[int(row["channel"])]
                if len(outliers) >= 2:
                    row["quality_reasons"].append("多频段谱型偏离：" + ", ".join(outliers))
                    if row.get("quality_grade") == "可用":
                        row["quality_grade"] = "需复核"
            self.completed.emit({
                "rows": rows,
                "channel_ids": [int(meta.channel_ids[column]) for column in columns],
                "bands": RESTING_MULTIBAND_DEFINITIONS,
                "line_settings": line_settings,
                "quality_settings": {key: self.settings.get(key) for key in ("line_warning_db", "line_reject_db", "line_warning_fraction", "line_reject_fraction", "band_profile_outlier_z")},
                "line_source_scope": "raw_current_paired" if raw_data is not None else "post_only",
                "workers": workers,
                "source_signature": (int(meta.rows), float(meta.fs), float(meta.time_offset), tuple(int(value) for value in meta.channel_ids)),
            })
        except Exception as exc:
            self.failed.emit(str(exc))


class SpikeWorker(QThread):
    """Detect negative-threshold spikes in the current data source."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, highpass_hz: float, lowpass_hz: float, threshold_factor: float, refractory_ms: float, columns=None):
        super().__init__()
        self.source = source
        self.highpass_hz = highpass_hz
        self.lowpass_hz = lowpass_hz
        self.threshold_factor = threshold_factor
        self.refractory_ms = refractory_ms
        self.columns = None if columns is None else np.asarray(columns, dtype=np.int64)

    def run(self) -> None:
        try:
            from scipy import signal

            meta = self.source.metadata
            columns = np.arange(meta.channels, dtype=np.int64) if self.columns is None else self.columns
            if not hasattr(self.source, "data"):
                data = self.source.read(0, meta.rows, columns)
                self.progress.emit(25.0, "已读取所选 Spike 通道")
            else:
                data = self.source.data[:, columns]
            if not 0 < self.highpass_hz < self.lowpass_hz < meta.fs / 2:
                raise ValueError("Spike 频段必须满足 0 < 低截止 < 高截止 < 奈奎斯特频率。")
            sos = signal.butter(4, (self.highpass_hz, self.lowpass_hz), btype="bandpass", fs=meta.fs, output="sos")
            refractory = max(1, int(round(self.refractory_ms * meta.fs / 1000.0)))
            rows = []
            for index in range(data.shape[1]):
                raw = np.nan_to_num(np.asarray(data[:, index], dtype=np.float32), nan=0.0)
                try:
                    filtered = signal.sosfiltfilt(sos, raw)
                except ValueError:
                    filtered = raw
                noise_sigma = float(np.median(np.abs(filtered)) / 0.6745)
                threshold = -abs(self.threshold_factor * noise_sigma)
                peaks, _ = signal.find_peaks(-filtered, height=-threshold, distance=refractory)
                rows.append({
                    "channel": int(meta.channel_ids[int(columns[index])]),
                    "spike_count": int(peaks.size),
                    "rate_hz": float(peaks.size / max(1e-12, data.shape[0] / meta.fs)),
                    "threshold_mv": threshold,
                })
                self.progress.emit(25.0 + 75.0 * (index + 1) / data.shape[1], "正在检测 Spike")
            self.completed.emit(rows)
        except Exception as exc:
            self.failed.emit(str(exc))


class DynamicSpikeWorker(QThread):
    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, source, low, high, threshold, refractory):
        super().__init__(); self.source=source; self.low=low; self.high=high; self.threshold=threshold; self.refractory=refractory

    def run(self) -> None:
        try:
            from scipy import signal
            data, _ = self.source.materialize(lambda value, text: self.progress.emit(value * 20, text)) if not hasattr(self.source, "data") else (self.source.data, None)
            meta=self.source.metadata; sos=signal.butter(4,(self.low,self.high),btype="bandpass",fs=meta.fs,output="sos"); refractory=max(1,int(round(self.refractory*meta.fs/1000)))
            windows=max(1,int(np.ceil(meta.rows/(meta.fs*10)))); counts=np.zeros((windows,data.shape[1]),dtype=np.int32)
            for channel in range(data.shape[1]):
                raw=np.nan_to_num(np.asarray(data[:,channel],dtype=np.float32),nan=0.0)
                try: filtered=signal.sosfiltfilt(sos,raw)
                except ValueError: filtered=raw
                sigma=float(np.median(np.abs(filtered))/0.6745); peaks,_=signal.find_peaks(-filtered,height=abs(self.threshold*sigma),distance=refractory)
                bins=np.minimum(windows-1,(peaks/(meta.fs*10)).astype(int)); counts[:,channel]=np.bincount(bins,minlength=windows)
                self.progress.emit(20+80*(channel+1)/data.shape[1],f"动态 Spike：{channel+1}/{data.shape[1]}")
            self.completed.emit(np.arange(windows)*10.0,counts)
        except Exception as exc: self.failed.emit(str(exc))


class IcaWorker(CooperativeWorker):
    """Background port of the legacy selected-channel MNE FastICA routine."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, source, columns, params):
        super().__init__()
        self.source, self.columns, self.params = source, np.asarray(columns, dtype=np.int64), dict(params)

    def run(self) -> None:
        try:
            self._checkpoint()
            import mne
            from mne.preprocessing import ICA
            meta = self.source.metadata
            if self.columns.size < 2:
                raise ValueError("ICA 至少需要选择两个通道。")
            self.progress.emit(5.0, "ICA：正在读取当前处理数据…")
            if hasattr(self.source, "data"):
                base = self.source.data
            else:
                base, _ = self.source.materialize(lambda value, text: self.progress.emit(5.0 + value * 25.0, text))
            self._checkpoint()
            self.progress.emit(35.0, "ICA：正在准备所选通道和平均参考…")
            selected = np.asarray(base[:, self.columns], dtype=np.float64)
            if selected.shape[0] < max(64, self.columns.size * 4):
                raise ValueError("当前数据太短，无法执行 ICA。")
            names = [f"ch_{meta.channel_ids[index]}" for index in self.columns]
            info = mne.create_info(names, sfreq=meta.fs, ch_types=["ecog"] * self.columns.size)
            raw = mne.io.RawArray(selected.T, info, verbose="ERROR")
            raw_car, _ = mne.set_eeg_reference(raw.copy(), ref_channels="average", projection=False, verbose="ERROR")
            fit = raw_car.copy().filter(self.params["low"], self.params["high"], fir_design="firwin", n_jobs=1, verbose="ERROR")
            n_components = min(self.params["components"], self.columns.size, selected.shape[0] - 1)
            ica = ICA(n_components=n_components, method="fastica", random_state=97, max_iter=self.params["max_iter"])
            self.progress.emit(45.0, f"ICA：FastICA 拟合 {n_components} 个成分…")
            # scikit-learn emits ConvergenceWarning when FastICA reaches its
            # iteration cap. Preserve the result for inspection, but make the
            # warning explicit instead of silently presenting it as normal.
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                ica.fit(fit, decim=self.params["decim"], verbose="ERROR")
            self._checkpoint()
            convergence_warnings = [
                str(item.message) for item in caught_warnings
                if item.category.__name__ == "ConvergenceWarning"
                or "converg" in str(item.message).lower()
            ]
            ica.exclude = [value for value in self.params["exclude"] if 0 <= value < n_components]
            clean = raw_car.copy(); ica.apply(clean, verbose="ERROR")
            cleaned = np.asarray(clean.get_data().T, dtype=np.float32)
            self.progress.emit(80.0, "ICA：正在写入清理后的处理数据…")
            target, path = allocate_storage(base.shape, np.float32, "ica_cleaned")
            for first in range(0, meta.rows, 200_000):
                self._checkpoint()
                last = min(meta.rows, first + 200_000)
                target[first:last] = base[first:last]
            target[:, self.columns] = cleaned
            if isinstance(target, np.memmap): target.flush()
            info_dict = {"columns": self.columns.tolist(), "channel_ids": [int(meta.channel_ids[index]) for index in self.columns], "components": int(n_components), "exclude": list(ica.exclude), "low": self.params["low"], "high": self.params["high"], "decim": self.params["decim"], "max_iter": self.params["max_iter"], "convergence_warnings": convergence_warnings}
            self.completed.emit(ArraySource(
                target, meta.fs, meta.time_offset, label="ica_cleaned", storage_path=path,
                channel_ids=meta.channel_ids, timing_metadata=meta.timing_metadata,
                provenance=derive_h5_provenance(
                    getattr(meta, "provenance", None), stage="ica_cleaned", fs=meta.fs, unit="mV",
                    channel_ids=meta.channel_ids,
                    source=None if getattr(meta, "provenance", None) else {"path": str(meta.path), "dataset": str(meta.dataset)},
                    operation={"name": "ica", **info_dict},
                ),
            ), info_dict)
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class BadChannelWorker(CooperativeWorker):
    """Run retained hard bad-channel checks only."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, source, settings, columns=None, analysis_cache: AnalysisCache | None = None):
        super().__init__()
        self.source = source
        self.settings = dict(settings)
        self.columns = None if columns is None else np.asarray(columns, dtype=np.int64).ravel()
        self.analysis_cache = analysis_cache
        self.fast_artifact_rows = []
        self.high_frequency_noise_rows = []
        self.flat_time_rows = []
        self.discrete_level_rows = []
        self.stage_timings: list[tuple[str, float]] = []

    def _run_legacy_full_matrix(self) -> None:
        """Former full-matrix implementation retained for comparison only."""
        try:
            self._checkpoint()
            run_started = perf_counter()
            meta = self.source.metadata
            columns = np.arange(meta.channels, dtype=np.int64) if self.columns is None else self.columns
            if not columns.size or np.any(columns < 0) or np.any(columns >= meta.channels):
                raise ValueError("No valid bad-channel columns were selected.")
            columns = np.unique(columns)
            channel_ids = np.asarray(meta.channel_ids, dtype=np.int64)[columns]

            data_started = perf_counter()
            if self.analysis_cache is not None:
                self.progress.emit(0.0, "坏道检查：正在读取所选通道数据缓存")
                data = self.analysis_cache.get_channel_data(self.source, columns)
            elif hasattr(self.source, "data"):
                data = np.asarray(self.source.data[:, columns], dtype=np.float32)
            else:
                self.progress.emit(0.0, "坏道检查：正在读取所选通道")
                data = np.asarray(self.source.read(0, meta.rows, columns), dtype=np.float32)
            self.stage_timings.append(("读取/取得所选通道数据", perf_counter() - data_started))
            self.progress.emit(20.0, "坏道检查：数据已准备，正在计算保留的坏道指标")

            flat_std = float(self.settings.get("flat_std", 1e-4))
            flat_ratio = float(self.settings.get("flat_ratio", 0.0)) / 100.0
            fast_artifact_only = bool(self.settings.get("fast_artifact_only", False))
            high_frequency_noise_only = bool(self.settings.get("high_frequency_noise_only", False))
            fast_artifact_enabled = bool(self.settings.get("fast_artifact_check", False))
            high_frequency_noise_enabled = bool(self.settings.get("high_frequency_noise_check", False))
            if fast_artifact_only or high_frequency_noise_only:
                fast_artifact_enabled = fast_artifact_only
                high_frequency_noise_enabled = high_frequency_noise_only

            saturation_width_percent = float(self.settings.get("saturation_width_percent", 1.0))
            saturation_ratio_threshold = float(self.settings.get("saturation_ratio_threshold", 45.0)) / 100.0
            if fast_artifact_enabled:
                if not np.isfinite(saturation_width_percent) or not 0.0 < saturation_width_percent <= 100.0:
                    raise ValueError("贴底区间宽度必须在 0 到 100% 之间。")
                if not np.isfinite(saturation_ratio_threshold) or not 0.0 <= saturation_ratio_threshold <= 1.0:
                    raise ValueError("贴底比例阈值必须在 0 到 100% 之间。")

            target = float(self.settings.get("high_frequency_noise_target", 2500.0))
            tolerance = float(self.settings.get("high_frequency_noise_tolerance", 1.0))
            target_ratio_threshold = float(self.settings.get("high_frequency_noise_ratio_threshold", 50.0)) / 100.0
            if high_frequency_noise_enabled:
                if not np.isfinite(target):
                    raise ValueError("2.5 mV 目标值必须是有限数。")
                if not np.isfinite(tolerance) or tolerance <= 0:
                    raise ValueError("2.5 mV 容差必须大于零。")
                if not np.isfinite(target_ratio_threshold) or not 0.0 <= target_ratio_threshold <= 1.0:
                    raise ValueError("2.5 mV 集中比例阈值必须在 0 到 100% 之间。")

            def flat_time_ratio_for_candidate(signal):
                signal = np.asarray(signal, dtype=np.float32)
                if signal.size == 0 or not np.isfinite(signal).any() or meta.fs <= 0:
                    return 1.0 if signal.size == 0 or not np.isfinite(signal).any() else 0.0
                window_size = max(16, int(round(meta.fs)))
                flat_windows = 0
                total_windows = 0
                ptp_threshold = flat_std * 6.0
                for start in range(0, signal.size, window_size):
                    window = signal[start:min(signal.size, start + window_size)]
                    window = window[np.isfinite(window)]
                    if window.size < max(8, window_size // 2):
                        continue
                    total_windows += 1
                    if float(np.ptp(window)) <= ptp_threshold:
                        flat_windows += 1
                return float(flat_windows / total_windows) if total_windows else 0.0

            fast_metrics = []
            fast_bad = np.zeros(data.shape[1], dtype=bool)
            target_metrics = []
            target_bad = np.zeros(data.shape[1], dtype=bool)
            metric_started = perf_counter()
            for index, channel_id in enumerate(channel_ids):
                self._checkpoint()
                signal = np.asarray(data[:, index], dtype=np.float32)
                finite = signal[np.isfinite(signal)]
                if fast_artifact_enabled:
                    metric = self.analysis_cache.get_saturation_metric(
                        self.source, int(channel_id), saturation_width_percent
                    ) if self.analysis_cache is not None else None
                    if metric is None:
                        if finite.size:
                            channel_min = float(np.min(finite))
                            channel_ptp = float(np.ptp(finite))
                            bottom_limit = channel_min + channel_ptp * saturation_width_percent / 100.0
                            bottom_ratio = float(np.count_nonzero(finite <= bottom_limit) / finite.size)
                        else:
                            channel_min = channel_ptp = bottom_limit = bottom_ratio = np.nan
                        metric = {
                            "channel": int(channel_id), "finite_samples": int(finite.size),
                            "channel_min": channel_min, "channel_ptp": channel_ptp,
                            "saturation_width_percent": saturation_width_percent,
                            "bottom_limit": bottom_limit, "bottom_ratio": bottom_ratio,
                        }
                        if self.analysis_cache is not None:
                            self.analysis_cache.put_saturation_metric(
                                self.source, int(channel_id), saturation_width_percent, metric
                            )
                    metric = dict(metric)
                    metric["saturation_ratio_threshold"] = saturation_ratio_threshold
                    metric["bottom_bad"] = bool(np.isfinite(metric["bottom_ratio"]) and metric["bottom_ratio"] >= saturation_ratio_threshold)
                    metric["is_bad"] = metric["bottom_bad"]
                    fast_bad[index] = metric["is_bad"]
                    fast_metrics.append(metric)

                if high_frequency_noise_enabled:
                    finite_count = int(finite.size)
                    near_count = int(np.count_nonzero(np.abs(finite - target) <= tolerance))
                    ratio = float(near_count / finite_count) if finite_count else np.nan
                    metric = {
                        "channel": int(channel_id), "target": target, "tolerance": tolerance,
                        "finite_samples": finite_count, "near_target_samples": near_count,
                        "near_target_ratio": ratio, "threshold": target_ratio_threshold,
                        "is_bad": bool(np.isfinite(ratio) and ratio > target_ratio_threshold),
                    }
                    target_bad[index] = metric["is_bad"]
                    target_metrics.append(metric)
                self.progress.emit(20.0 + 35.0 * (index + 1) / data.shape[1], f"坏道检查：计算指标 {index + 1}/{data.shape[1]}")
            self.stage_timings.append(("坏道指标计算", perf_counter() - metric_started))

            fast_by_channel = {int(row["channel"]): row for row in fast_metrics}
            target_by_channel = {int(row["channel"]): row for row in target_metrics}

            def inspect(index):
                self._checkpoint()
                channel = int(channel_ids[index])
                signal = np.asarray(data[:, index], dtype=np.float32)
                finite = signal[np.isfinite(signal)]
                reasons = []
                if finite.size < 10:
                    reasons.append("too few finite samples")
                elif not (fast_artifact_only or high_frequency_noise_only):
                    std = float(np.std(finite))
                    signal_range = float(np.ptp(finite))
                    global_flat = std <= flat_std or signal_range <= flat_std * 6.0
                    if global_flat:
                        reasons.append(f"flat/dead: std={std:.3g}, range={signal_range:.3g}")
                    levels = np.unique(np.round(finite / max(flat_std, 1e-4))).size
                    low_levels = levels <= 8 or levels / finite.size <= 5e-4
                    if low_levels:
                        reasons.append(f"too few unique levels ({levels})")
                    if (global_flat or low_levels) and flat_ratio > 0:
                        flat_time = flat_time_ratio_for_candidate(signal)
                        if flat_time >= flat_ratio:
                            reasons.append(f"flat time {flat_time * 100:.1f}%")
                if fast_artifact_enabled and fast_bad[index]:
                    metric = fast_by_channel[channel]
                    reasons.append(f"saturation: bottom={metric['bottom_ratio'] * 100:.1f}%")
                if high_frequency_noise_enabled and target_bad[index]:
                    metric = target_by_channel[channel]
                    reasons.append(
                        "坏道：数据集中在2.5mV附近 "
                        f"(center={target:g}, ratio={metric['near_target_ratio'] * 100:.1f}%, band=±{tolerance:g})"
                    )
                return channel, "; ".join(reasons) if reasons else None

            workers = int(self.settings.get("workers", 0))
            if workers <= 0:
                workers = min(max(1, os.cpu_count() or 1), data.shape[1])
            bad = {}
            inspect_started = perf_counter()
            with ThreadPoolExecutor(max_workers=workers if self.settings.get("parallel", False) else 1) as executor:
                futures = [executor.submit(inspect, index) for index in range(data.shape[1])]
                for done, future in enumerate(as_completed(futures), start=1):
                    channel, reason = future.result()
                    if reason:
                        bad[channel] = reason
                    self.progress.emit(55.0 + 45.0 * done / len(futures), f"坏道检查：逐通道判定 {done}/{len(futures)}")
            self.stage_timings.append(("逐通道坏道判定", perf_counter() - inspect_started))
            self.fast_artifact_rows = sorted(fast_metrics, key=lambda row: row["channel"])
            self.high_frequency_noise_rows = sorted(target_metrics, key=lambda row: row["channel"])
            self.stage_timings.append(("总计", perf_counter() - run_started))
            self.progress.emit(100.0, "坏道检查：所有通道判定完成")
            good = [int(channel) for channel in channel_ids if int(channel) not in bad]
            self.completed.emit(good, bad, {})
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


    def run(self) -> None:
        """Evaluate retained rules with bounded memory, independent of recording length."""
        try:
            self._checkpoint()
            run_started = perf_counter()
            meta = self.source.metadata
            columns = np.arange(meta.channels, dtype=np.int64) if self.columns is None else np.unique(self.columns)
            if not columns.size or np.any(columns < 0) or np.any(columns >= meta.channels):
                raise ValueError("No valid bad-channel columns were selected.")
            channel_ids = np.asarray(meta.channel_ids, dtype=np.int64)[columns]
            channel_count = int(columns.size)
            # Retired flat/dead, discrete-level and flat-time rules no longer
            # expose thresholds or participate in normal bad-channel checks.
            flat_std = float(self.settings.get("flat_std", 1e-4))
            flat_ratio = float(self.settings.get("flat_ratio", 0.0)) / 100.0
            fast_only = bool(self.settings.get("fast_artifact_only", False))
            target_only = bool(self.settings.get("high_frequency_noise_only", False))
            fast_enabled = bool(self.settings.get("fast_artifact_check", False))
            target_enabled = bool(self.settings.get("high_frequency_noise_check", False))
            # These retired rules cannot be re-enabled by an old settings
            # file.  The only unconditional guard retained below is too few
            # finite samples; active classification is saturation/2.5 mV.
            global_flat_enabled = False
            discrete_level_enabled = False
            flat_time_enabled = False
            flat_time_all_channels = False
            discrete_level_threshold = int(self.settings.get("discrete_level_threshold", 8))
            level_tracking_limit = int(self.settings.get(
                "discrete_level_tracking_limit", discrete_level_threshold,
            ))
            if not 1 <= discrete_level_threshold <= level_tracking_limit <= 64:
                raise ValueError("离散水平阈值和统计上限必须是 1～64 的整数。")
            if fast_only or target_only:
                fast_enabled, target_enabled = fast_only, target_only
                global_flat_enabled = discrete_level_enabled = flat_time_enabled = False
            width_percent = float(self.settings.get("saturation_width_percent", 1.0))
            saturation_threshold = float(self.settings.get("saturation_ratio_threshold", 45.0)) / 100.0
            target = float(self.settings.get("high_frequency_noise_target", 2500.0))
            tolerance = float(self.settings.get("high_frequency_noise_tolerance", 1.0))
            target_threshold = float(self.settings.get("high_frequency_noise_ratio_threshold", 50.0)) / 100.0
            requested_workers = int(self.settings.get("workers", 0))
            if requested_workers <= 0:
                requested_workers = min(max(1, os.cpu_count() or 1), channel_count)
            channel_workers = (
                min(requested_workers, channel_count)
                if self.settings.get("parallel", False) else 1
            )
            channel_execution_text = (
                f"通道并行 {channel_workers}"
                if channel_workers > 1 else "通道串行"
            )
            if not np.isfinite(flat_std) or flat_std <= 0 or not 0 <= flat_ratio <= 1:
                raise ValueError("平直标准差必须大于0，平直时间比例必须在0到100%之间。")
            if fast_enabled and (not 0 < width_percent <= 100 or not 0 <= saturation_threshold <= 1):
                raise ValueError("贴底区间宽度和贴底比例阈值必须在有效百分比范围内。")
            if target_enabled and (not np.isfinite(target) or tolerance <= 0 or not 0 <= target_threshold <= 1):
                raise ValueError("2.5mV目标、容差或比例阈值无效。")

            finite_count = np.zeros(channel_count, dtype=np.int64)
            totals = np.zeros(channel_count, dtype=np.float64)
            squares = np.zeros(channel_count, dtype=np.float64)
            minima = np.full(channel_count, np.inf, dtype=np.float64)
            maxima = np.full(channel_count, -np.inf, dtype=np.float64)
            near_count = np.zeros(channel_count, dtype=np.int64)
            level_sets = [set() for _ in range(channel_count)]
            # Read a short time slab across as many channels as fit in a
            # bounded ~256 MiB worst-case float64 buffer.  This keeps HDF5
            # access contiguous and much faster than thousands of tiny
            # per-channel reads while remaining independent of total length.
            chunk_rows = max(1, min(int(PROCESS_CHUNK_SAMPLES), 50_000))
            memory_budget = 256 * 1024 * 1024
            block_channels = max(1, min(channel_count, memory_budget // max(1, chunk_rows * 8)))
            total_work = max(1, int(np.ceil(meta.rows / chunk_rows)) * int(np.ceil(channel_count / block_channels)))
            work_done = 0
            first_pass = perf_counter()
            def summarize_channel_chunk(local_and_signal):
                local, signal_values = local_and_signal
                finite = signal_values[np.isfinite(signal_values)]
                if not finite.size:
                    return local, 0, 0.0, 0.0, np.inf, -np.inf, 0, ()
                finite64 = finite.astype(np.float64, copy=False)
                levels = ()
                if discrete_level_enabled or (flat_time_enabled and not flat_time_all_channels):
                    levels = tuple(
                        np.unique(np.round(finite / max(flat_std, 1e-4)))
                        [:level_tracking_limit + 1].tolist()
                    )
                return (
                    local,
                    int(finite.size),
                    float(np.sum(finite64, dtype=np.float64)),
                    float(np.sum(finite64 * finite64, dtype=np.float64)),
                    float(np.min(finite)),
                    float(np.max(finite)),
                    int(np.count_nonzero(np.abs(finite - target) <= tolerance))
                    if target_enabled else 0,
                    levels,
                )

            with ThreadPoolExecutor(max_workers=channel_workers) as executor:
                for start in range(0, meta.rows, chunk_rows):
                    stop = min(meta.rows, start + chunk_rows)
                    for left in range(0, channel_count, block_channels):
                        self._checkpoint()
                        right = min(channel_count, left + block_channels)
                        values = np.asarray(
                            self.source.read(start, stop, columns[left:right]), dtype=np.float32,
                        )
                        if values.ndim == 1:
                            values = values[:, None]
                        channel_inputs = (
                            (local, values[:, local]) for local in range(values.shape[1])
                        )
                        for result in executor.map(summarize_channel_chunk, channel_inputs):
                            local, count, total, square, minimum, maximum, near, levels = result
                            index = left + local
                            if not count:
                                continue
                            finite_count[index] += count
                            totals[index] += total
                            squares[index] += square
                            minima[index] = min(minima[index], minimum)
                            maxima[index] = max(maxima[index], maximum)
                            near_count[index] += near
                            if levels and len(level_sets[index]) <= level_tracking_limit:
                                level_sets[index].update(levels)
                                if len(level_sets[index]) > level_tracking_limit:
                                    level_sets[index] = set(range(level_tracking_limit + 1))
                        work_done += 1
                        self.progress.emit(
                            35.0 * work_done / total_work,
                            f"坏道检查：第一遍读取 {stop}/{meta.rows}（{channel_execution_text}）",
                        )
            self.stage_timings.append((
                f"分块第一遍统计（{channel_execution_text}）",
                perf_counter() - first_pass,
            ))

            bottom_count = np.zeros(channel_count, dtype=np.int64)
            if fast_enabled:
                bottom_limits = minima + (maxima - minima) * width_percent / 100.0
                work_done = 0
                second_pass = perf_counter()
                for start in range(0, meta.rows, chunk_rows):
                    stop = min(meta.rows, start + chunk_rows)
                    for left in range(0, channel_count, block_channels):
                        self._checkpoint()
                        right = min(channel_count, left + block_channels)
                        values = np.asarray(self.source.read(start, stop, columns[left:right]), dtype=np.float32)
                        if values.ndim == 1:
                            values = values[:, None]
                        valid = np.isfinite(values)
                        bottom_count[left:right] += np.sum(
                            valid & (values <= bottom_limits[left:right][None, :]), axis=0, dtype=np.int64
                        )
                        work_done += 1
                        self.progress.emit(35.0 + 30.0 * work_done / total_work, f"坏道检查：贴底比例读取 {stop}/{meta.rows}")
                self.stage_timings.append(("分块第二遍贴底统计", perf_counter() - second_pass))

            means = np.divide(totals, finite_count, out=np.zeros_like(totals), where=finite_count > 0)
            variances = np.divide(squares, finite_count, out=np.zeros_like(squares), where=finite_count > 0) - means * means
            stds = np.sqrt(np.maximum(0.0, variances))
            ranges = maxima - minima
            global_flat = (stds <= flat_std) | (ranges <= flat_std * 6.0)
            level_counts = np.asarray([len(values) for values in level_sets], dtype=np.int64)
            # Retain only enough distinct levels to decide the configured
            # threshold; full unique-level sets grow without bound on long H5s.
            low_levels = level_counts <= discrete_level_threshold
            if flat_time_enabled:
                flat_candidates = (
                    finite_count >= 10
                    if flat_time_all_channels
                    else (global_flat | low_levels) & (finite_count >= 10)
                )
                candidates = np.flatnonzero(flat_candidates)
            else:
                candidates = np.asarray([], dtype=np.int64)
            candidate_set = set(map(int, candidates))
            flat_time_ratios = np.zeros(channel_count, dtype=np.float64)
            if candidates.size and flat_ratio > 0:
                window_rows = max(16, int(round(meta.fs)))
                scan_rows = max(window_rows, (chunk_rows // window_rows) * window_rows)
                flat_windows = np.zeros(candidates.size, dtype=np.int64)
                total_windows = np.zeros(candidates.size, dtype=np.int64)
                candidate_columns = columns[candidates]
                candidate_work = max(1, int(np.ceil(meta.rows / scan_rows)) * int(np.ceil(candidates.size / block_channels)))
                done = 0
                for start in range(0, meta.rows, scan_rows):
                    stop = min(meta.rows, start + scan_rows)
                    full_stop = start + ((stop - start) // window_rows) * window_rows
                    if full_stop <= start:
                        continue
                    for left in range(0, candidates.size, block_channels):
                        self._checkpoint()
                        right = min(candidates.size, left + block_channels)
                        values = np.asarray(
                            self.source.read(start, full_stop, candidate_columns[left:right]),
                            dtype=np.float32,
                        )
                        if values.ndim == 1:
                            values = values[:, None]
                        shaped = values.reshape(-1, window_rows, values.shape[1])
                        finite = np.isfinite(shaped)
                        valid = np.sum(finite, axis=1) >= max(8, window_rows // 2)
                        safe = np.where(finite, shaped, np.nan)
                        with np.errstate(all="ignore"):
                            window_ptp = np.nanmax(safe, axis=1) - np.nanmin(safe, axis=1)
                        total_windows[left:right] += np.sum(valid, axis=0, dtype=np.int64)
                        flat_windows[left:right] += np.sum(valid & (window_ptp <= flat_std * 6.0), axis=0, dtype=np.int64)
                        done += 1
                        self.progress.emit(65.0 + 20.0 * done / candidate_work, f"坏道检查：平直候选分块 {done}/{candidate_work}")
                flat_time_ratios[candidates] = np.divide(
                    flat_windows, total_windows,
                    out=np.zeros(candidates.size, dtype=np.float64), where=total_windows > 0,
                )

            fast_rows, target_rows, bad = [], [], {}
            self.flat_time_rows = [
                {
                    "channel": int(channel_ids[index]),
                    "candidate": bool(index in candidate_set),
                    "flat_time_ratio": float(flat_time_ratios[index]),
                }
                for index in range(channel_count)
            ]
            self.discrete_level_rows = [
                {"channel": int(channel_ids[index]), "level_count": int(level_counts[index])}
                for index in range(channel_count)
            ]
            for index, channel in enumerate(channel_ids):
                self._checkpoint()
                reasons = []
                if finite_count[index] < 10:
                    if self.settings.get("minimum_finite_check", True):
                        reasons.append("too few finite samples")
                elif not (fast_only or target_only):
                    if global_flat_enabled and global_flat[index]:
                        reasons.append(f"flat/dead: std={stds[index]:.3g}, range={ranges[index]:.3g}")
                    if discrete_level_enabled and low_levels[index]:
                        reasons.append(f"too few unique levels ({level_counts[index]})")
                    if (flat_time_enabled and index in candidate_set
                            and flat_ratio > 0 and flat_time_ratios[index] >= flat_ratio):
                        reasons.append(f"flat time {flat_time_ratios[index] * 100:.1f}%")
                if fast_enabled:
                    ratio = bottom_count[index] / finite_count[index] if finite_count[index] else np.nan
                    metric = {
                        "channel": int(channel), "finite_samples": int(finite_count[index]),
                        "channel_min": float(minima[index]), "channel_ptp": float(ranges[index]),
                        "saturation_width_percent": width_percent,
                        "bottom_limit": float(minima[index] + ranges[index] * width_percent / 100.0),
                        "bottom_ratio": float(ratio), "saturation_ratio_threshold": saturation_threshold,
                        "bottom_bad": bool(np.isfinite(ratio) and ratio >= saturation_threshold),
                    }
                    metric["is_bad"] = metric["bottom_bad"]
                    fast_rows.append(metric)
                    if metric["is_bad"]:
                        reasons.append(f"saturation: bottom={ratio * 100:.1f}%")
                if target_enabled:
                    ratio = near_count[index] / finite_count[index] if finite_count[index] else np.nan
                    metric = {
                        "channel": int(channel), "target": target, "tolerance": tolerance,
                        "finite_samples": int(finite_count[index]), "near_target_samples": int(near_count[index]),
                        "near_target_ratio": float(ratio), "threshold": target_threshold,
                        "is_bad": bool(np.isfinite(ratio) and ratio > target_threshold),
                    }
                    target_rows.append(metric)
                    if metric["is_bad"]:
                        reasons.append(f"坏道：数据集中在2.5mV附近 (center={target:g}, ratio={ratio * 100:.1f}%, band=±{tolerance:g})")
                if reasons:
                    bad[int(channel)] = "; ".join(reasons)
                self.progress.emit(85.0 + 15.0 * (index + 1) / channel_count, f"坏道检查：汇总 {index + 1}/{channel_count}")
            self.fast_artifact_rows = fast_rows
            self.high_frequency_noise_rows = target_rows
            self.stage_timings.append(("总计", perf_counter() - run_started))
            good = [int(channel) for channel in channel_ids if int(channel) not in bad]
            self.completed.emit(good, bad, {})
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))



class BatchBadChannelEvaluationWorker(CooperativeWorker):
    """Sequentially evaluate multiple H5 files with one frozen QC setting set."""

    RULE_NAMES = (
        "有效采样不足", "全局平直/低波动", "离散水平过少", "平直时间比例",
        "贴底饱和", "2.5mV附近集中", "其他保留坏道条件",
    )

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, paths, settings):
        super().__init__()
        self.paths = [Path(normalize_h5_path_text(path)) for path in paths]
        self.settings = dict(settings)

    @staticmethod
    def _rule_labels(reason: str) -> list[str]:
        text = str(reason or "")
        lower = text.lower()
        labels = []
        if "too few finite samples" in lower:
            labels.append("有效采样不足")
        if "flat/dead" in lower:
            labels.append("全局平直/低波动")
        if "too few unique levels" in lower:
            labels.append("离散水平过少")
        if "flat time" in lower:
            labels.append("平直时间比例")
        if "saturation" in lower:
            labels.append("贴底饱和")
        if "2.5mv" in lower:
            labels.append("2.5mV附近集中")
        return labels or ["其他保留坏道条件"]

    def run(self) -> None:
        try:
            results = []
            total = max(1, len(self.paths))
            for file_index, path in enumerate(self.paths):
                self._checkpoint()
                file_start = file_index * 100.0 / total
                file_span = 100.0 / total
                self.progress.emit(file_start, f"批量坏道评估：正在打开 {path.name}")
                try:
                    source = LazyH5Source()
                    meta = source.open(path)
                    detector = BadChannelWorker(source, self.settings, analysis_cache=None)
                    # Make pause/cancel checkpoints inside the retained QC
                    # calculation obey this outer batch task's controls.
                    detector._checkpoint = self._checkpoint
                    output = []
                    errors = []
                    detector.completed.connect(lambda *args, target=output: target.append(args))
                    detector.failed.connect(errors.append)
                    detector.progress.connect(
                        lambda value, message, start=file_start, span=file_span:
                        self.progress.emit(start + span * float(value) / 100.0, message)
                    )
                    detector.run()
                    self._checkpoint()
                    if errors:
                        raise RuntimeError(errors[0])
                    if not output:
                        raise RuntimeError("坏道检查没有返回结果。")
                    good_ids, bad_reasons, _unused_candidates = output[0]
                    bad_reasons = {
                        int(channel): str(reason) for channel, reason in bad_reasons.items()
                    }
                    rule_channels: dict[str, list[int]] = {
                        name: [] for name in self.RULE_NAMES
                    }
                    channel_rows = []
                    all_ids = [int(channel) for channel in meta.channel_ids]
                    for channel in all_ids:
                        reason = bad_reasons.get(channel, "")
                        labels = self._rule_labels(reason) if reason else []
                        for label in labels:
                            rule_channels[label].append(channel)
                        channel_rows.append({
                            "channel": channel,
                            "result": "坏道" if reason else "健康",
                            "trigger_count": len(labels),
                            "rules": "；".join(labels),
                            "reason": reason,
                        })
                    bad_count = len(bad_reasons)
                    results.append({
                        "path": str(path.resolve()),
                        "file": path.name,
                        "status": "成功",
                        "rows": int(meta.rows),
                        "channels": int(meta.channels),
                        "sampling_rate_hz": float(meta.fs),
                        "good_count": len(good_ids),
                        "bad_count": bad_count,
                        "bad_ratio": bad_count / max(1, int(meta.channels)),
                        "rule_channels": rule_channels,
                        "overlap_channels": [
                            row["channel"] for row in channel_rows if row["trigger_count"] > 1
                        ],
                        "channel_rows": channel_rows,
                        "error": "",
                    })
                    del detector
                except CooperativeTaskCancelled:
                    raise
                except Exception as exc:
                    results.append({
                        "path": str(path), "file": path.name, "status": "失败",
                        "rows": 0, "channels": 0, "sampling_rate_hz": np.nan,
                        "good_count": 0, "bad_count": 0, "bad_ratio": np.nan,
                        "rule_channels": {}, "overlap_channels": [],
                        "channel_rows": [], "error": str(exc),
                    })
                self.progress.emit(
                    (file_index + 1) * 100.0 / total,
                    f"批量坏道评估：已完成 {file_index + 1}/{len(self.paths)} 个文件",
                )
            self.completed.emit(results)
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class BadChannelParameterSweepWorker(CooperativeWorker):
    """Sweep remapped raw data against labels read from processed exports."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self, paths, base_settings, parameter_key: str, parameter_label: str, values,
        mapping_path: str | Path, raw_paths=None, data_cache: dict | None = None,
    ):
        super().__init__()
        self.paths = [Path(path) for path in paths]
        if raw_paths is None:
            self.raw_paths = [None] * len(self.paths)
        else:
            self.raw_paths = [Path(normalize_h5_path_text(path)) for path in raw_paths]
            if len(self.raw_paths) != len(self.paths):
                raise ValueError("处理结果 H5 与原始 H5 的数量不一致。")
        self.base_settings = dict(base_settings)
        self.parameter_key = str(parameter_key)
        self.parameter_label = str(parameter_label)
        self.values = [float(value) for value in values]
        self.mapping_path = Path(mapping_path)
        self.data_cache = data_cache if data_cache is not None else {}

    @staticmethod
    def _normalize_manual(value: object) -> str:
        text = str(value or "").strip().lower()
        return {"healthy": "good", "健康": "good", "坏道": "bad"}.get(text, text)

    @classmethod
    def _reference_from_snapshot(cls, path: Path, available_ids: set[int]):
        try:
            with h5py.File(path, "r") as h5:
                raw = h5.attrs.get("preprocess_qc_json")
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            snapshot = json.loads(str(raw))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(snapshot, dict) or snapshot.get("completed") is not True:
            return None
        evaluated = {
            int(channel) for channel in snapshot.get("evaluated_channel_ids", [])
            if int(channel) in available_ids
        }
        automatic_bad = {
            int(channel): str(reason)
            for channel, reason in snapshot.get("automatic_bad_reasons", {}).items()
            if int(channel) in evaluated
        }
        manual = {}
        for channel, value in snapshot.get("manual_overrides", {}).items():
            channel = int(channel)
            normalized = cls._normalize_manual(value)
            if channel in evaluated and normalized in {"good", "bad"}:
                manual[channel] = normalized
        reference_bad = set(automatic_bad)
        for channel, result in manual.items():
            if result == "bad":
                reference_bad.add(channel)
            else:
                reference_bad.discard(channel)
        return {
            "evaluated_ids": evaluated,
            "reference_bad_ids": reference_bad,
            "manual_overrides": manual,
            "source": "H5内嵌QC人工复核结果（未载入处理数据）",
        }

    @classmethod
    def _reference_from_csv(cls, path: Path, available_ids: set[int]):
        import csv
        hint = ""
        try:
            with h5py.File(path, "r") as h5:
                hint = h5.attrs.get("processing_record_csv", "")
            if isinstance(hint, bytes):
                hint = hint.decode("utf-8", errors="replace")
        except OSError:
            pass
        candidates = []
        if str(hint).strip():
            candidates.append(path.parent / str(hint).strip())
        candidates.append(path.with_suffix(".csv"))
        stem = path.stem
        for suffix in ("_lfp_processed", "_spike_processed", "_processed"):
            if stem.lower().endswith(suffix):
                candidates.append(path.with_name(stem[:-len(suffix)] + "_processing.csv"))
        candidates.extend(path.parent.glob("*_processing.csv"))
        seen = set()
        for candidate in candidates:
            candidate = Path(candidate)
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            try:
                with open(candidate, newline="", encoding="utf-8-sig") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, UnicodeError, csv.Error):
                continue
            rows = [
                row for row in rows
                if Path(str(row.get("output_file", "")).strip()).name.lower() == path.name.lower()
            ]
            by_channel = {}
            for row in rows:
                try:
                    channel = int(float(str(row.get("channel", "")).strip()))
                except (TypeError, ValueError):
                    continue
                if channel in available_ids:
                    by_channel[channel] = row
            if not by_channel:
                continue
            automatic_bad = set()
            manual = {}
            for channel, row in by_channel.items():
                if str(row.get("automatic_status", "")).strip().lower() in {"bad", "坏道"}:
                    automatic_bad.add(channel)
                applied = str(row.get("manual_override_applied", "")).strip().lower() in {
                    "1", "true", "yes", "y", "是",
                }
                normalized = cls._normalize_manual(row.get("manual_override", ""))
                if applied and normalized in {"good", "bad"}:
                    manual[channel] = normalized
            reference_bad = set(automatic_bad)
            for channel, result in manual.items():
                if result == "bad":
                    reference_bad.add(channel)
                else:
                    reference_bad.discard(channel)
            return {
                "evaluated_ids": set(by_channel),
                "reference_bad_ids": reference_bad,
                "manual_overrides": manual,
                "source": f"CSV人工复核结果:{candidate.name}",
            }
        return None

    @classmethod
    def _load_reference(cls, path: Path, meta):
        available = {int(channel) for channel in meta.channel_ids}
        return (
            cls._reference_from_snapshot(path, available)
            or cls._reference_from_csv(path, available)
        )

    @staticmethod
    def _use_remapped_reference_ids(reference, target_channel_ids):
        """Keep reviewed IDs unchanged because review occurs after remapping."""
        reference = dict(reference)
        available = {int(value) for value in target_channel_ids}
        reference["evaluated_ids"] = {
            int(value) for value in reference.get("evaluated_ids", set())
            if int(value) in available
        }
        reference["reference_bad_ids"] = {
            int(value) for value in reference.get("reference_bad_ids", set())
            if int(value) in available
        }
        reference["manual_overrides"] = {
            int(channel): result
            for channel, result in reference.get("manual_overrides", {}).items()
            if int(channel) in available
        }
        reference["labels_remapped_from_raw"] = False
        reference["coordinate_system"] = "reviewed_remapped_target"
        reference["source"] = (
            str(reference.get("source", "人工复核结果"))
            + "（人工标注为重映射后目标ID，原样使用）"
        )
        return reference

    @staticmethod
    def _metrics(predicted_bad: set[int], reference_bad: set[int], evaluated: set[int]):
        predicted_bad &= evaluated
        reference_bad &= evaluated
        tp = len(predicted_bad & reference_bad)
        fp = len(predicted_bad - reference_bad)
        fn = len(reference_bad - predicted_bad)
        tn = len(evaluated - predicted_bad - reference_bad)
        precision = tp / (tp + fp) if tp + fp else (1.0 if not reference_bad else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        beta2 = 4.0
        f2 = (1 + beta2) * precision * recall / (beta2 * precision + recall) if beta2 * precision + recall else 0.0
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, f2=f2)

    def run(self) -> None:
        try:
            runs = []
            failures = []
            total_runs = max(1, len(self.paths) * len(self.values))
            completed_runs = 0
            threshold_only_keys = {
                "flat_ratio",
                "saturation_ratio_threshold",
                "high_frequency_noise_ratio_threshold",
            }
            factor_settings = dict(self.base_settings)
            factor_settings.update({
                "global_flat_check": False,
                "discrete_level_check": False,
                "flat_time_check": False,
                "flat_time_all_channels": False,
                "fast_artifact_check": False,
                "fast_artifact_only": False,
                "high_frequency_noise_check": False,
                "high_frequency_noise_only": False,
            })
            if self.parameter_key == "flat_std":
                factor_settings["global_flat_check"] = True
            elif self.parameter_key == "flat_ratio":
                factor_settings["flat_time_check"] = True
                factor_settings["flat_time_all_channels"] = True
            elif self.parameter_key.startswith("saturation_"):
                factor_settings["fast_artifact_check"] = True
                factor_settings["fast_artifact_only"] = True
            elif self.parameter_key.startswith("high_frequency_noise_"):
                factor_settings["high_frequency_noise_check"] = True
                factor_settings["high_frequency_noise_only"] = True

            for path, selected_raw_path in zip(self.paths, self.raw_paths):
                self._checkpoint()
                try:
                    raw_path = (
                        selected_raw_path.resolve()
                        if selected_raw_path is not None
                        else resolve_original_h5_for_processed(path)
                    )
                    reviewed_stat = path.stat()
                    raw_stat = raw_path.stat()
                    mapping_stat = self.mapping_path.stat()
                    cache_key = (
                        str(path.resolve()), int(reviewed_stat.st_mtime_ns), int(reviewed_stat.st_size),
                        str(raw_path.resolve()), int(raw_stat.st_mtime_ns), int(raw_stat.st_size),
                        str(self.mapping_path.resolve()), int(mapping_stat.st_mtime_ns), int(mapping_stat.st_size),
                        "reviewed_target_ids_compact_v7_labels_already_remapped",
                    )
                    cached_entry = self.data_cache.get(cache_key)
                    if cached_entry is not None:
                        source = cached_entry["source"]
                        reference = cached_entry["reference"]
                        mapped_ids = set(cached_entry["mapped_ids"])
                        meta = source.metadata
                        self.progress.emit(
                            100.0 * completed_runs / total_runs,
                            f"{path.name}：复用已读取并重映射的数据",
                        )
                    else:
                        raw_source = LazyH5Source()
                        raw_meta = raw_source.open(raw_path)
                        remapped_data, remapped_cache, mapping, raw_ids, remapped_ids = remap_source_from_excel(
                            raw_source,
                            self.mapping_path,
                            lambda _value, message, name=path.name: self.progress.emit(
                                100.0 * completed_runs / total_runs,
                                f"{name}：{message}",
                            ),
                        )
                        source = ArraySource(
                            remapped_data, raw_meta.fs, raw_meta.time_offset,
                            label=f"parameter-sweep-remapped:{raw_path.name}",
                            storage_path=remapped_cache,
                            channel_ids=remapped_ids,
                            timing_metadata=raw_meta.timing_metadata,
                        )
                        meta = source.metadata
                        mapped_ids = {int(value) for value in remapped_ids}
                        reference = self._load_reference(path, meta)
                        if reference is not None:
                            reference = self._use_remapped_reference_ids(
                                reference, meta.channel_ids,
                            )
                        if reference is not None:
                            self.data_cache[cache_key] = {
                                "source": source, "reference": reference,
                                "mapped_ids": mapped_ids,
                            }
                    if reference is None:
                        raise ValueError("未找到可用的历史人工判断结果。")
                    evaluated = set(reference["evaluated_ids"]) & mapped_ids
                    columns = np.flatnonzero(np.isin(meta.channel_ids, sorted(evaluated)))
                    if not columns.size:
                        raise ValueError("历史QC通道与当前H5通道不匹配。")

                    def execute(settings):
                        detector = BadChannelWorker(
                            source, settings, columns=columns, analysis_cache=None,
                        )
                        detector._checkpoint = self._checkpoint
                        output, errors = [], []
                        detector.completed.connect(lambda *args, target=output: target.append(args))
                        detector.failed.connect(errors.append)
                        detector.run()
                        if errors:
                            raise RuntimeError(errors[0])
                        if not output:
                            raise RuntimeError("坏道检查没有返回结果。")
                        return detector, {
                            int(channel): str(reason)
                            for channel, reason in output[0][1].items()
                        }

                    def append_run(value, reasons, previous_bad):
                        predicted_bad = set(reasons)
                        metrics = self._metrics(
                            set(predicted_bad),
                            set(reference["reference_bad_ids"]),
                            set(evaluated),
                        )
                        rule_channels = {
                            name: [] for name in BatchBadChannelEvaluationWorker.RULE_NAMES
                        }
                        details = []
                        for channel in sorted(predicted_bad):
                            labels = BatchBadChannelEvaluationWorker._rule_labels(reasons[channel])
                            for label in labels:
                                rule_channels[label].append(channel)
                            details.append({
                                "channel": channel,
                                "reference_bad": channel in reference["reference_bad_ids"],
                                "rules": "；".join(labels),
                                "reason": reasons[channel],
                            })
                        added = sorted(predicted_bad - previous_bad) if previous_bad is not None else []
                        removed = sorted(previous_bad - predicted_bad) if previous_bad is not None else []
                        runs.append({
                            "file": path.name,
                            "path": str(path.resolve()),
                            "raw_path": str(raw_path.resolve()),
                            "parameter_key": self.parameter_key,
                            "parameter_label": self.parameter_label,
                            "parameter_value": value,
                            "evaluated_count": len(evaluated),
                            "reference_bad_count": len(set(reference["reference_bad_ids"]) & evaluated),
                            "manual_override_count": len(set(reference["manual_overrides"]) & evaluated),
                            "reference_source": reference["source"],
                            "predicted_bad_count": len(predicted_bad),
                            "predicted_bad_ratio": len(predicted_bad) / max(1, len(evaluated)),
                            "predicted_bad_ids": sorted(predicted_bad),
                            "added_vs_previous": added,
                            "removed_vs_previous": removed,
                            "rule_channels": rule_channels,
                            "details": details,
                            **metrics,
                        })
                        return predicted_bad

                    previous_bad = None
                    if self.parameter_key in threshold_only_keys:
                        settings = dict(factor_settings)
                        settings[self.parameter_key] = (
                            1.0 if self.parameter_key == "flat_ratio" else self.values[0]
                        )
                        detector, initial_reasons = execute(settings)

                        def retained_parts(reason):
                            parts = [part.strip() for part in str(reason).split(";") if part.strip()]
                            if self.parameter_key == "flat_ratio":
                                return [part for part in parts if not part.lower().startswith("flat time")]
                            if self.parameter_key == "saturation_ratio_threshold":
                                return [part for part in parts if not part.lower().startswith("saturation:")]
                            return [part for part in parts if "2.5mv" not in part.lower()]

                        base_parts = {
                            channel: retained_parts(initial_reasons.get(channel, ""))
                            for channel in evaluated
                        }
                        if self.parameter_key == "flat_ratio":
                            metric_rows = {
                                int(row["channel"]): row
                                for row in getattr(detector, "flat_time_rows", [])
                            }
                        elif self.parameter_key == "saturation_ratio_threshold":
                            metric_rows = {
                                int(row["channel"]): row
                                for row in getattr(detector, "fast_artifact_rows", [])
                            }
                        else:
                            metric_rows = {
                                int(row["channel"]): row
                                for row in getattr(detector, "high_frequency_noise_rows", [])
                            }

                        for value in self.values:
                            self._checkpoint()
                            derived = {}
                            for channel in evaluated:
                                parts = list(base_parts[channel])
                                metric = metric_rows.get(channel, {})
                                if self.parameter_key == "flat_ratio":
                                    ratio = float(metric.get("flat_time_ratio", 0.0))
                                    if bool(metric.get("candidate", False)) and ratio >= value / 100.0:
                                        parts.append(f"flat time {ratio * 100:.1f}%")
                                elif self.parameter_key == "saturation_ratio_threshold":
                                    ratio = float(metric.get("bottom_ratio", np.nan))
                                    if np.isfinite(ratio) and ratio >= value / 100.0:
                                        parts.append(f"saturation: bottom={ratio * 100:.1f}%")
                                else:
                                    ratio = float(metric.get("near_target_ratio", np.nan))
                                    if np.isfinite(ratio) and ratio > value / 100.0:
                                        target = float(metric.get("target", 2500.0))
                                        tolerance = float(metric.get("tolerance", 1.0))
                                        parts.append(
                                            f"坏道：数据集中在2.5mV附近 "
                                            f"(center={target:g}, ratio={ratio * 100:.1f}%, band=±{tolerance:g})"
                                        )
                                if parts:
                                    derived[channel] = "; ".join(parts)
                            previous_bad = append_run(value, derived, previous_bad)
                            completed_runs += 1
                            self.progress.emit(
                                100.0 * completed_runs / total_runs,
                                f"参数精调：{path.name}，{self.parameter_label}={value:g} "
                                f"({completed_runs}/{total_runs})；底层指标已复用",
                            )
                    else:
                        for value in self.values:
                            self._checkpoint()
                            settings = dict(factor_settings)
                            settings[self.parameter_key] = value
                            _detector, reasons = execute(settings)
                            previous_bad = append_run(value, reasons, previous_bad)
                            completed_runs += 1
                            self.progress.emit(
                                100.0 * completed_runs / total_runs,
                                f"参数精调：{path.name}，{self.parameter_label}={value:g} "
                                f"({completed_runs}/{total_runs})",
                            )
                except CooperativeTaskCancelled:
                    raise
                except Exception as exc:
                    failures.append({"file": path.name, "path": str(path), "error": str(exc)})
                    completed_runs += len(self.values)
                    self.progress.emit(
                        100.0 * completed_runs / total_runs,
                        f"参数精调：跳过 {path.name}（{exc}）",
                    )
            self.completed.emit({
                "parameter_key": self.parameter_key,
                "parameter_label": self.parameter_label,
                "values": list(self.values),
                "evaluation_mode": "single_factor",
                "base_settings": dict(self.base_settings),
                "runs": runs,
                "failures": failures,
            })
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


BAD_CHANNEL_PARAMETER_PAIRS = OrderedDict((
    ("flat", ("平直：标准差 × 平直时间比例", "flat_std", "flat_ratio")),
    ("saturation", ("贴底：区间宽度 × 贴底比例", "saturation_width_percent", "saturation_ratio_threshold")),
    ("near_target", ("2.5mV附近：容差 × 命中比例", "high_frequency_noise_tolerance", "high_frequency_noise_ratio_threshold")),
))
BAD_CHANNEL_PARAMETER_LABELS = {
    "flat_std": "平直标准差阈值（mV）",
    "discrete_level_threshold": "离散水平数上限",
    "flat_ratio": "平直时间比例（%）",
    "saturation_width_percent": "贴底区间宽度（% PTP）",
    "saturation_ratio_threshold": "贴底比例阈值（%）",
    "high_frequency_noise_tolerance": "2.5mV附近容差（当前数据单位）",
    "high_frequency_noise_ratio_threshold": "2.5mV附近比例阈值（%）",
    "high_frequency_noise_target": "2.5mV目标值（当前数据单位）",
}


def paired_bad_channel_settings(base: dict, group: str, first: float, second: float) -> dict:
    """Activate only the complete two-parameter rule being evaluated."""
    settings = dict(base)
    settings.update({
        "global_flat_check": False,
        "discrete_level_check": False,
        "flat_time_check": False,
        "flat_time_all_channels": False,
        "fast_artifact_check": False,
        "fast_artifact_only": False,
        "high_frequency_noise_check": False,
        "high_frequency_noise_only": False,
    })
    if group == "flat":
        settings.update(flat_std=first, flat_ratio=second,
                        flat_time_check=True, flat_time_all_channels=True)
    elif group == "saturation":
        settings.update(saturation_width_percent=first,
                        saturation_ratio_threshold=second, fast_artifact_only=True)
    elif group == "near_target":
        settings.update(high_frequency_noise_tolerance=first,
                        high_frequency_noise_ratio_threshold=second,
                        high_frequency_noise_only=True)
    else:
        raise ValueError(f"未知的坏道参数组：{group}")
    return settings


class BadChannelPairSweepWorker(BadChannelParameterSweepWorker):
    """Evaluate a two-dimensional rule grid against reviewed final labels."""

    def __init__(self, paths, base_settings, group, first_values, second_values,
                 mapping_path, raw_paths, data_cache):
        label, _first_key, _second_key = BAD_CHANNEL_PARAMETER_PAIRS[group]
        super().__init__(paths, base_settings, group, label, [], mapping_path,
                         raw_paths, data_cache)
        self.group = group
        self.first_values = [float(value) for value in first_values]
        self.second_values = [float(value) for value in second_values]

    def _prepared_record(self, reviewed_path: Path, raw_path: Path):
        raw_path = raw_path.resolve()
        reviewed_stat, raw_stat, map_stat = (
            reviewed_path.stat(), raw_path.stat(), self.mapping_path.stat()
        )
        key = (
            str(reviewed_path.resolve()), int(reviewed_stat.st_mtime_ns), int(reviewed_stat.st_size),
            str(raw_path), int(raw_stat.st_mtime_ns), int(raw_stat.st_size),
            str(self.mapping_path.resolve()), int(map_stat.st_mtime_ns), int(map_stat.st_size),
            "reviewed_target_ids_compact_v7_labels_already_remapped",
        )
        cached = self.data_cache.get(key)
        if cached is not None:
            mapped_ids = cached.get("mapped_ids")
            if mapped_ids is None:
                # Cache entries written by the former one-variable worker do
                # not record mapping coverage; rebuild safely once.
                self.data_cache.pop(key, None)
            else:
                return cached["source"], cached["reference"], set(mapped_ids), True
        raw_source = LazyH5Source()
        raw_meta = raw_source.open(raw_path)
        data, storage_path, mapping, ids, remapped_ids = remap_source_from_excel(
            raw_source, self.mapping_path,
            lambda _value, message: self.progress.emit(0, message),
            force_memmap=True,
        )
        source = ArraySource(
            data, raw_meta.fs, raw_meta.time_offset,
            label=f"pair-sweep-remapped:{raw_path.name}", storage_path=storage_path,
            channel_ids=remapped_ids,
            timing_metadata=raw_meta.timing_metadata,
        )
        reference = self._load_reference(reviewed_path, source.metadata)
        if reference is None:
            raise ValueError("未找到可用的历史人工复核结果。")
        reference = self._use_remapped_reference_ids(
            reference, source.metadata.channel_ids,
        )
        mapped_ids = {int(value) for value in remapped_ids}
        self.data_cache[key] = {
            "source": source, "reference": reference, "mapped_ids": mapped_ids,
        }
        return source, reference, mapped_ids, False

    def run(self) -> None:
        try:
            runs, failures = [], []
            total = max(1, len(self.paths) * len(self.first_values) * len(self.second_values))
            done = 0
            for reviewed_path, raw_path in zip(self.paths, self.raw_paths):
                self._checkpoint()
                try:
                    if raw_path is None:
                        raw_path = resolve_original_h5_for_processed(reviewed_path)
                    source, reference, mapped_ids, reused = self._prepared_record(
                        reviewed_path, raw_path,
                    )
                    evaluated = set(reference["evaluated_ids"]) & mapped_ids
                    columns = np.flatnonzero(np.isin(source.metadata.channel_ids, sorted(evaluated)))
                    if not columns.size:
                        raise ValueError("人工复核通道与重映射结果没有交集。")
                    self.progress.emit(100 * done / total,
                        f"{reviewed_path.name}：{'复用已有重映射数据' if reused else '重映射完成'}")
                    for first in self.first_values:
                        self._checkpoint()
                        # The first axis changes the underlying metric.  The
                        # second axis only classifies that metric, so read the
                        # full recording once per first-axis value.
                        settings = paired_bad_channel_settings(
                            self.base_settings, self.group, first,
                            min(self.second_values),
                        )
                        detector = BadChannelWorker(source, settings, columns=columns,
                                                    analysis_cache=None)
                        detector._checkpoint = self._checkpoint
                        output, errors = [], []
                        detector.completed.connect(lambda *args, target=output: target.append(args))
                        detector.failed.connect(errors.append)
                        detector.run()
                        if errors:
                            raise RuntimeError(errors[0])
                        if not output:
                            raise RuntimeError("坏道判断没有返回结果。")
                        initial_reasons = {
                            int(ch): str(reason) for ch, reason in output[0][1].items()
                        }
                        if self.group == "flat":
                            metric_rows = detector.flat_time_rows
                        elif self.group == "saturation":
                            metric_rows = detector.fast_artifact_rows
                        else:
                            metric_rows = detector.high_frequency_noise_rows
                        metric_by_id = {int(row["channel"]): row for row in metric_rows}
                        for second in self.second_values:
                            self._checkpoint()
                            reasons = {}
                            for channel in evaluated:
                                metric = metric_by_id.get(channel, {})
                                # This data-integrity condition is intentionally
                                # independent of the selected signal rule.
                                reason = initial_reasons.get(channel, "")
                                if reason.startswith("too few finite samples"):
                                    reasons[channel] = reason
                                    continue
                                if self.group == "flat":
                                    ratio = float(metric.get("flat_time_ratio", 0.0))
                                    triggered = bool(metric.get("candidate")) and ratio >= second / 100.0
                                    reason = f"flat time {ratio * 100:.1f}%"
                                elif self.group == "saturation":
                                    ratio = float(metric.get("bottom_ratio", np.nan))
                                    triggered = np.isfinite(ratio) and ratio >= second / 100.0
                                    reason = f"saturation: bottom={ratio * 100:.1f}%"
                                else:
                                    ratio = float(metric.get("near_target_ratio", np.nan))
                                    triggered = np.isfinite(ratio) and ratio > second / 100.0
                                    reason = (
                                        f"2.5mV附近集中 (center={float(metric.get('target', 2500)):g}, "
                                        f"ratio={ratio * 100:.1f}%, "
                                        f"band=±{float(metric.get('tolerance', first)):g})"
                                    )
                                if triggered:
                                    reasons[channel] = reason
                            predicted = set(reasons) & evaluated
                            metrics = self._metrics(set(predicted),
                                                    set(reference["reference_bad_ids"]), set(evaluated))
                            runs.append({
                                "file": reviewed_path.name,
                                "path": str(reviewed_path.resolve()),
                                "raw_path": str(raw_path.resolve()),
                                "first": first, "second": second,
                                "evaluated_count": len(evaluated),
                                "evaluated_ids": sorted(evaluated),
                                "excluded_unmapped_count": len(
                                    set(reference["evaluated_ids"]) - mapped_ids
                                ),
                                "reference_bad_count": len(
                                    set(reference["reference_bad_ids"]) & evaluated
                                ),
                                "reference_bad_ids": sorted(
                                    set(reference["reference_bad_ids"]) & evaluated
                                ),
                                "manual_override_count": len(reference["manual_overrides"]),
                                "reference_source": reference["source"],
                                "predicted_bad_ids": sorted(predicted),
                                "predicted_bad_count": len(predicted),
                                "reasons": reasons,
                                **metrics,
                            })
                            done += 1
                            self.progress.emit(100 * done / total,
                                f"二维精调 {reviewed_path.name}：{first:g} × {second:g}（{done}/{total}）")
                except CooperativeTaskCancelled:
                    raise
                except Exception as exc:
                    failures.append({"file": reviewed_path.name, "path": str(reviewed_path),
                                     "error": str(exc)})
                    done += len(self.first_values) * len(self.second_values)
            self.completed.emit({
                "group": self.group,
                "label": BAD_CHANNEL_PARAMETER_PAIRS[self.group][0],
                "first_key": BAD_CHANNEL_PARAMETER_PAIRS[self.group][1],
                "second_key": BAD_CHANNEL_PARAMETER_PAIRS[self.group][2],
                "first_values": self.first_values, "second_values": self.second_values,
                "paths": [str(path.resolve()) for path in self.paths],
                "runs": runs, "failures": failures,
                "base_settings": dict(self.base_settings),
            })
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


def choose_bad_channel_pair_combination(pair_results: dict[str, dict]) -> dict:
    """Find the best three-rule OR over all scanned pair-grid combinations."""
    groups = tuple(BAD_CHANNEL_PARAMETER_PAIRS)
    if set(pair_results) != set(groups):
        raise ValueError("三组二维参数扫描尚未全部完成。")
    paths = tuple(pair_results[groups[0]]["paths"])
    if not paths or any(tuple(pair_results[group]["paths"]) != paths
                        or pair_results[group]["failures"] for group in groups):
        raise ValueError("三组扫描的文件集合不同，或存在失败文件，不能合并寻优。")
    first_rows = {row["path"]: row for row in pair_results[groups[0]]["runs"]}
    if set(first_rows) != set(paths):
        raise ValueError("缺少人工复核参考通道，不能合并寻优。")
    reference_masks = []
    evaluated_counts = []
    for path in paths:
        row = first_rows[path]
        reference_masks.append(sum(1 << (int(ch) - 1)
                                   for ch in row["reference_bad_ids"]))
        evaluated_counts.append(len(row["evaluated_ids"]))
    candidate_groups = []
    for group in groups:
        result = pair_results[group]
        by_pair: dict[tuple[float, float], dict[str, int]] = {}
        for row in result["runs"]:
            pair = float(row["first"]), float(row["second"])
            by_pair.setdefault(pair, {})[row["path"]] = sum(
                1 << (int(ch) - 1) for ch in row["predicted_bad_ids"]
            )
            baseline = first_rows[row["path"]]
            if (row["evaluated_ids"] != baseline["evaluated_ids"]
                    or row["reference_bad_ids"] != baseline["reference_bad_ids"]):
                raise ValueError("三组规则的评估范围或人工标签不一致。")
        candidates = [
            (pair, tuple(by_path[path] for path in paths))
            for pair, by_path in by_pair.items() if set(by_path) == set(paths)
        ]
        if not candidates:
            raise ValueError(f"{group} 没有完整的扫描结果。")
        candidate_groups.append(candidates)

    best, best_key = None, None
    positives = sum(mask.bit_count() for mask in reference_masks)
    evaluated_total = sum(evaluated_counts)
    combinations = 0
    for selected in product(*candidate_groups):
        combinations += 1
        masks = [selected[0][1][index] | selected[1][1][index] | selected[2][1][index]
                 for index in range(len(paths))]
        tp = sum((mask & reference_masks[index]).bit_count()
                 for index, mask in enumerate(masks))
        fp = sum((mask & ~reference_masks[index]).bit_count()
                 for index, mask in enumerate(masks))
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 1.0
        min_file_recall = min(
            (mask & reference_masks[index]).bit_count() / reference_masks[index].bit_count()
            if reference_masks[index] else 1.0
            for index, mask in enumerate(masks)
        )
        qualified = recall >= .98 and min_file_recall >= .95
        f2 = 5 * precision * recall / (4 * precision + recall) \
            if 4 * precision + recall else 0.0
        rank = ((1, precision, -(tp + fp), f2, min_file_recall)
                if qualified else (0, recall, min_file_recall, precision, f2))
        if best_key is None or rank > best_key:
            best_key = rank
            best = {
                "recommendations": {
                    group: selected[index][0] for index, group in enumerate(groups)
                },
                "tp": tp, "fp": fp, "fn": fn,
                "tn": evaluated_total - tp - fp - fn,
                "precision": precision, "recall": recall,
                "min_file_recall": min_file_recall, "f2": f2,
                "qualified": qualified,
            }
    best["combinations_tested"] = combinations
    return best


class BadChannelCombinedValidationWorker(BadChannelPairSweepWorker):
    """Validate the OR of three selected complete rules on the same cohort."""

    def __init__(self, paths, base_settings, pair_results, mapping_path,
                 raw_paths, data_cache):
        super().__init__(paths, base_settings, "flat", [], [], mapping_path,
                         raw_paths, data_cache)
        self.pair_results = dict(pair_results)

    def run(self) -> None:
        try:
            optimization = choose_bad_channel_pair_combination(self.pair_results)
            recommendations = optimization["recommendations"]
            rows, failures = [], []
            for index, (reviewed_path, raw_path) in enumerate(zip(self.paths, self.raw_paths)):
                self._checkpoint()
                try:
                    source, reference, mapped_ids, reused = self._prepared_record(
                        reviewed_path, raw_path,
                    )
                    if not reused:
                        raise RuntimeError("组合验证需要同一批已缓存数据，请先完成三组二维扫描。")
                    evaluated = set(reference["evaluated_ids"]) & mapped_ids
                    columns = np.flatnonzero(np.isin(source.metadata.channel_ids, sorted(evaluated)))
                    rule_bad = {}
                    for group in BAD_CHANNEL_PARAMETER_PAIRS:
                        first, second = recommendations[group]
                        settings = paired_bad_channel_settings(
                            self.pair_results[group]["base_settings"], group, first, second,
                        )
                        detector = BadChannelWorker(source, settings, columns=columns,
                                                    analysis_cache=None)
                        detector._checkpoint = self._checkpoint
                        output, errors = [], []
                        detector.completed.connect(lambda *args, target=output: target.append(args))
                        detector.failed.connect(errors.append)
                        detector.run()
                        if errors:
                            raise RuntimeError(errors[0])
                        if not output:
                            raise RuntimeError(f"{group} 未返回坏道结果。")
                        rule_bad[group] = set(map(int, output[0][1])) & evaluated
                    predicted = set().union(*rule_bad.values())
                    metrics = self._metrics(set(predicted),
                                            set(reference["reference_bad_ids"]),
                                            set(evaluated))
                    rows.append({
                        "file": reviewed_path.name,
                        "raw_path": str(raw_path.resolve()),
                        "evaluated_count": len(evaluated),
                        "excluded_unmapped_count": len(
                            set(reference["evaluated_ids"]) - mapped_ids
                        ),
                        "reference_bad_count": len(
                            set(reference["reference_bad_ids"]) & evaluated
                        ),
                        "predicted_bad_count": len(predicted),
                        "rule_bad_ids": {key: sorted(value) for key, value in rule_bad.items()},
                        "predicted_bad_ids": sorted(predicted),
                        **metrics,
                    })
                except CooperativeTaskCancelled:
                    raise
                except Exception as exc:
                    failures.append({"file": reviewed_path.name, "error": str(exc)})
                self.progress.emit(100 * (index + 1) / max(1, len(self.paths)),
                                   f"组合验证 {index + 1}/{len(self.paths)}")
            if not failures:
                for key in ("tp", "fp", "fn", "tn"):
                    actual = sum(row[key] for row in rows)
                    if actual != optimization[key]:
                        raise RuntimeError(
                            f"联合网格统计与重新判定不一致：{key}={optimization[key]}，"
                            f"重新判定={actual}。请检查规则与缓存。"
                        )
            self.completed.emit({"rows": rows, "failures": failures,
                                 "recommendations": recommendations,
                                 "optimization": optimization})
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


BAD_CHANNEL_SINGLE_PARAMETERS = OrderedDict((
    ("saturation_width_percent", "贴底区间宽度（% PTP）"),
    ("saturation_ratio_threshold", "贴底比例阈值（%）"),
    ("high_frequency_noise_tolerance", "2.5mV附近容差"),
    ("high_frequency_noise_ratio_threshold", "2.5mV附近比例阈值（%）"),
))
BAD_CHANNEL_ABLATION_RULES = OrderedDict((
    ("global_flat_check", "全局平直/低波动"),
    ("discrete_level_check", "离散水平过少"),
    ("flat_time_check", "平直时间比例"),
    ("fast_artifact_check", "贴底饱和"),
    ("high_frequency_noise_check", "2.5mV附近集中"),
))


class BadChannelControlledExperimentWorker(BadChannelPairSweepWorker):
    """Sweep retained saturation/near-target parameters, or run rule ablation."""

    THRESHOLD_ONLY_KEYS = frozenset((
        "saturation_ratio_threshold", "high_frequency_noise_ratio_threshold",
    ))

    def __init__(self, paths, base_settings, mapping_path, raw_paths, data_cache,
                 *, parameter_key=None, values=None, allow_partial_review_scope=False):
        super().__init__(paths, base_settings, "flat", [], [], mapping_path,
                         raw_paths, data_cache)
        self.parameter_key = parameter_key
        self.values = [float(value) for value in (values or [])]
        # Enabled only by “高召回参数精调”. Historical reviewed H5 files may
        # contain labels for only part of today's real remapped target IDs.
        # Evaluate every ID present on both sides without remapping labels.
        self.allow_partial_review_scope = bool(allow_partial_review_scope)

    def _evaluation_scope(self, reference, mapped_ids) -> set[int]:
        evaluated = {
            int(channel) for channel in reference.get("evaluated_ids", set())
        } & {int(channel) for channel in mapped_ids}
        if not evaluated:
            raise ValueError("人工复核通道与原始物理通道映射没有交集。")
        missing_review_ids = {int(channel) for channel in mapped_ids} - evaluated
        if missing_review_ids and not self.allow_partial_review_scope:
            preview = ",".join(map(str, sorted(missing_review_ids)[:20]))
            suffix = "…" if len(missing_review_ids) > 20 else ""
            raise ValueError(
                "人工复核标签没有覆盖全部Excel映射目标通道："
                f"缺少 {preview}{suffix}。为避免少算通道，本文件未纳入统计。"
            )
        return evaluated

    def _detect(self, source, columns, settings, *, return_detector=False):
        detector = BadChannelWorker(source, settings, columns=columns,
                                    analysis_cache=None)
        detector._checkpoint = self._checkpoint
        output, errors = [], []
        detector.completed.connect(lambda *args: output.append(args))
        detector.failed.connect(errors.append)
        detector.progress.connect(
            lambda _value, message: self.progress.emit(
                100.0 * getattr(self, "_current_file_index", 0) / max(1, len(self.paths)),
                f"{getattr(self, '_current_file_name', '')}：{message}",
            )
        )
        detector.run()
        if errors:
            raise RuntimeError(errors[0])
        if not output:
            raise RuntimeError("坏道检查没有返回结果。")
        reasons = {int(channel): str(reason) for channel, reason in output[0][1].items()}
        return (detector, reasons) if return_detector else reasons

    @staticmethod
    def _threshold_predictions(detector, reasons, parameter_key, values, evaluated):
        """Reclassify fixed signal metrics without changing any QC rule."""
        if parameter_key == "saturation_ratio_threshold":
            metric_rows = detector.fast_artifact_rows
            target_part = lambda part: part.startswith("saturation:")
            def triggered(metric, value):
                ratio = float(metric.get("bottom_ratio", np.nan))
                return bool(np.isfinite(ratio) and ratio >= value / 100.0)
        elif parameter_key == "high_frequency_noise_ratio_threshold":
            metric_rows = detector.high_frequency_noise_rows
            target_part = lambda part: "2.5mv附近" in part.lower()
            def triggered(metric, value):
                ratio = float(metric.get("near_target_ratio", np.nan))
                return bool(np.isfinite(ratio) and ratio > value / 100.0)
        else:
            raise ValueError(f"参数不支持阈值复用：{parameter_key}")
        metric_by_id = {int(row["channel"]): row for row in metric_rows}
        other_bad = {
            channel for channel, reason in reasons.items()
            if channel in evaluated and any(
                not target_part(part.strip()) for part in reason.split(";") if part.strip()
            )
        }
        return {
            value: other_bad | {
                channel for channel in evaluated
                if triggered(metric_by_id.get(channel, {}), value)
            }
            for value in values
        }

    @staticmethod
    def _controlled_rule_settings(base_settings, parameter_key, values):
        """Exclude retired flat rules and activate the rule being swept."""
        settings = dict(base_settings)
        settings.update({
            "global_flat_check": False,
            "discrete_level_check": False,
            "flat_time_check": False,
            "flat_time_all_channels": False,
        })
        if parameter_key.startswith("saturation_"):
            settings["fast_artifact_check"] = True
        elif parameter_key.startswith("high_frequency_noise_"):
            settings["high_frequency_noise_check"] = True
        return settings

    def run(self) -> None:
        try:
            runs, failures = [], []
            mode = "single_parameter" if self.parameter_key else "ablation"
            if self.parameter_key and self.parameter_key not in BAD_CHANNEL_SINGLE_PARAMETERS:
                raise ValueError(f"未知的单变量参数：{self.parameter_key}")
            controlled_settings = (
                self._controlled_rule_settings(
                    self.base_settings, self.parameter_key, self.values,
                ) if self.parameter_key else None
            )
            for index, (reviewed_path, raw_path) in enumerate(zip(self.paths, self.raw_paths)):
                self._checkpoint()
                self._current_file_index = index
                self._current_file_name = reviewed_path.name
                try:
                    source, reference, mapped_ids, reused = self._prepared_record(
                        reviewed_path, raw_path,
                    )
                    evaluated = self._evaluation_scope(reference, mapped_ids)
                    columns = np.flatnonzero(np.isin(
                        source.metadata.channel_ids, sorted(evaluated),
                    ))
                    truth = set(reference["reference_bad_ids"]) & evaluated
                    threshold_only = self.parameter_key in self.THRESHOLD_ONLY_KEYS
                    baseline_settings = controlled_settings or self.base_settings
                    if threshold_only:
                        detector, baseline = self._detect(
                            source, columns, baseline_settings, return_detector=True,
                        )
                    else:
                        baseline = self._detect(source, columns, baseline_settings)
                    baseline_bad = set(baseline) & evaluated
                    predictions = {}
                    if threshold_only:
                        metric_detector = detector
                        metric_reasons = baseline
                        baseline_value = float(self.base_settings[self.parameter_key])
                        predictions = self._threshold_predictions(
                            metric_detector, metric_reasons, self.parameter_key,
                            set(self.values) | {baseline_value}, evaluated,
                        )
                        if predictions[baseline_value] != baseline_bad:
                            raise RuntimeError(
                                "阈值复用结果与完整基线判断不一致；已停止该文件以避免输出错误统计。"
                            )
                    scenarios = [("baseline", None, dict(baseline_settings), True)]
                    if self.parameter_key:
                        for value in self.values:
                            settings = dict(baseline_settings)
                            settings[self.parameter_key] = value
                            scenarios.append(("sweep", value, settings, True))
                    else:
                        for rule in BAD_CHANNEL_ABLATION_RULES:
                            active = (rule in {"global_flat_check", "discrete_level_check", "flat_time_check"}
                                      or bool(self.base_settings.get(rule, False)))
                            if active:
                                settings = dict(self.base_settings)
                                settings[rule] = False
                                scenarios.append((rule, None, settings, True))
                            else:
                                scenarios.append((rule, None, {}, False))
                    for kind, value, settings, active in scenarios:
                        self._checkpoint()
                        if kind == "baseline":
                            predicted = set(baseline_bad)
                        elif threshold_only:
                            predicted = set(predictions[value])
                        elif (self.parameter_key is not None and
                              value == float(self.base_settings[self.parameter_key])):
                            predicted = set(baseline_bad)
                        else:
                            reasons = self._detect(source, columns, settings) if active else {}
                            predicted = set(reasons) & evaluated
                        rows = self._metrics(set(predicted), set(truth), set(evaluated)) if active else {}
                        runs.append({
                            "file": reviewed_path.name,
                            "path": str(reviewed_path.resolve()),
                            "raw_path": str(raw_path.resolve()),
                            "scenario": kind, "value": value, "active": active,
                            "evaluated_count": len(evaluated),
                            "excluded_unmapped_count": len(set(reference["evaluated_ids"]) - mapped_ids),
                            "reference_bad_count": len(truth),
                            "reference_source": reference["source"],
                            "predicted_bad_ids": sorted(predicted) if active else [],
                            "tp_lost_vs_baseline_ids": sorted((baseline_bad - predicted) & truth) if active else [],
                            "fp_removed_vs_baseline_ids": sorted((baseline_bad - predicted) - truth) if active else [],
                            "tp_gained_vs_baseline_ids": sorted((predicted - baseline_bad) & truth) if active else [],
                            "fp_added_vs_baseline_ids": sorted((predicted - baseline_bad) - truth) if active else [],
                            "reused_remap": reused,
                            **rows,
                        })
                    self.progress.emit(100 * (index + 1) / max(1, len(self.paths)),
                                       f"{reviewed_path.name}：{mode} {index + 1}/{len(self.paths)}")
                except CooperativeTaskCancelled:
                    raise
                except Exception as exc:
                    failures.append({"file": reviewed_path.name, "path": str(reviewed_path),
                                     "error": str(exc)})
            self.completed.emit({
                "mode": mode, "parameter_key": self.parameter_key,
                "values": self.values, "base_settings": dict(self.base_settings),
                "evaluation_settings": dict(controlled_settings or self.base_settings),
                "paths": [str(path.resolve()) for path in self.paths],
                "runs": runs, "failures": failures,
            })
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class LeaveOneOutMedianCARWorker(CooperativeWorker):
    """Apply leave-one-out median CAR using only reviewed-good channels."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, source, good_channel_ids):
        super().__init__()
        self.source = source
        self.good_channel_ids = {int(value) for value in good_channel_ids}

    def run(self) -> None:
        try:
            self._checkpoint()
            meta = self.source.metadata
            if meta is None:
                raise RuntimeError("请先加载数据。")
            if len(self.good_channel_ids) < 2:
                raise ValueError("Leave-one-out median CAR 至少需要两个合格通道。")
            data, _ = (
                (self.source.data, None)
                if hasattr(self.source, "data")
                else self.source.materialize(lambda value, text: self.progress.emit(value * 35.0, text))
            )
            values = np.asarray(data, dtype=np.float32)
            ids = np.asarray(meta.channel_ids, dtype=np.int64)
            good_columns = np.flatnonzero(np.isin(ids, sorted(self.good_channel_ids)))
            if good_columns.size < 2:
                raise ValueError("当前数据源中可用的合格通道少于两个。")
            target, target_path = allocate_storage(values.shape, values.dtype, "median_car")
            # Reference is computed only for reviewed-good channels, with
            # each channel removed from its own leave-one-out reference.
            total = max(1, values.shape[0])
            for first in range(0, values.shape[0], 20_000):
                self._checkpoint()
                last = min(values.shape[0], first + 20_000)
                block = np.asarray(values[first:last], dtype=np.float32)
                good_block = block[:, good_columns]
                target_block = np.array(block, copy=True)
                # Sort the reference channels once per block.  The rank of an
                # excluded value determines the exact median of the remaining
                # N-1 values, avoiding N repeated nanmedian passes.
                row_fill = np.nanmedian(good_block, axis=1)
                row_fill = np.where(np.isfinite(row_fill), row_fill, 0.0).astype(np.float32)
                good_filled = np.where(np.isfinite(good_block), good_block, row_fill[:, None])
                order = np.argsort(good_filled, axis=1, kind="quicksort")
                sorted_values = np.take_along_axis(good_filled, order, axis=1)
                n_good = int(good_columns.size)
                remaining = n_good - 1
                if remaining % 2:
                    middle = remaining // 2
                    for local_column, column in enumerate(good_columns):
                        rank = np.argmax(order == local_column, axis=1)
                        reference = np.where(rank <= middle, sorted_values[:, middle + 1], sorted_values[:, middle])
                        target_block[:, column] = block[:, column] - reference
                else:
                    lower = remaining // 2 - 1
                    upper = remaining // 2
                    for local_column, column in enumerate(good_columns):
                        rank = np.argmax(order == local_column, axis=1)
                        low_value = np.where(rank <= lower, sorted_values[:, lower + 1], sorted_values[:, lower])
                        high_value = np.where(rank < upper, sorted_values[:, upper + 1], sorted_values[:, upper])
                        reference = (low_value + high_value) * 0.5
                        target_block[:, column] = block[:, column] - reference
                # Bad/noise channels never contribute to the reference and
                # are intentionally left unchanged in the CAR output.
                target[first:last] = target_block
                self.progress.emit(35.0 + 65.0 * last / total, f"重参考：已处理 {last:,}/{values.shape[0]:,} 个采样点")
            if isinstance(target, np.memmap):
                target.flush()
            output = ArraySource(
                target, meta.fs, meta.time_offset, label="median_car",
                storage_path=target_path, channel_ids=meta.channel_ids,
                timing_metadata=meta.timing_metadata,
                provenance=derive_h5_provenance(
                    getattr(meta, "provenance", None), stage="median_car", fs=meta.fs, unit="mV",
                    channel_ids=meta.channel_ids,
                    source=None if getattr(meta, "provenance", None) else {"path": str(meta.path), "dataset": str(meta.dataset)},
                    operation={
                        "name": "reference", "method": "leave_one_out_median_car",
                        "good_channel_ids": sorted(int(channel) for channel in self.good_channel_ids),
                    },
                ),
            )
            self.completed.emit(output, {"good_channel_ids": sorted(self.good_channel_ids), "method": "leave-one-out median CAR"})
        except CooperativeTaskCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


def evaluate_task_trial_window_quality(
    epoch: np.ndarray,
    fs: float,
    epoch_start_ms: float,
    *,
    flat_epsilon: float = 1e-4,
    flat_ratio_percent: float = 30.0,
    flat_ptp: float = 0.01,
    available_ratio: float = 0.995,
    saturation_run_samples: int = 8,
    jump_mad_multiplier: float = 12.0,
    jump_median_multiplier: float = 8.0,
    jump_flat_floor_multiplier: float = 10.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the 5-window trial-quality decision for every channel.

    Task response analysis uses the five 1 s windows from 0.5 to 5.5 s after
    the marker.  This intentionally has no PSD/Welch call: all checks are
    vectorized over ``(window, sample, channel)`` so a large trial set remains
    bounded by the HDF5 slices that are already needed for epoching.
    """
    values = np.asarray(epoch, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Task epoch must be a time x channel array.")
    channels = values.shape[1]
    samples_per_window = max(1, int(round(float(fs))))
    first = int(round((500.0 - float(epoch_start_ms)) * float(fs) / 1000.0))
    # A short task epoch is valid for short-latency responses.  It must not,
    # however, turn the missing tail of the prescribed five-window QC region
    # into false "clean" evidence.  Assess every complete 1 s window that is
    # actually present and mark the remaining windows unavailable.
    bounds = [
        (first + number * samples_per_window, first + (number + 1) * samples_per_window)
        for number in range(5)
    ]
    present = [number for number, (left, right) in enumerate(bounds) if left >= 0 and right <= values.shape[0]]
    clean = np.zeros((5, channels), dtype=bool)
    available = np.zeros((5, channels), dtype=bool)
    saturated = np.zeros((5, channels), dtype=bool)
    flat = np.zeros((5, channels), dtype=bool)
    jump = np.zeros((5, channels), dtype=bool)
    if not present:
        return clean, {
            "clean": clean, "available": available, "saturated": saturated,
            "flat": flat, "jump": jump, "unavailable": ~available,
        }

    windows = np.stack([values[bounds[number][0]:bounds[number][1]] for number in present])
    finite_ratio = np.mean(np.isfinite(windows), axis=1)
    present_available = finite_ratio >= float(available_ratio)
    finite_windows = np.where(np.isfinite(windows), windows, np.nan)
    with np.errstate(all="ignore"):
        ptp = np.nanmax(finite_windows, axis=1) - np.nanmin(finite_windows, axis=1)
        differences = np.diff(finite_windows, axis=1)
        abs_differences = np.abs(differences)
        flat_ratio = np.mean(abs_differences <= abs(float(flat_epsilon)), axis=1) * 100.0
        difference_median = np.nanmedian(abs_differences, axis=1)
        difference_mad = np.nanmedian(
            np.abs(abs_differences - difference_median[:, None, :]), axis=1
        )
        max_difference = np.nanmax(abs_differences, axis=1)

    # A digitizer clipping plateau is represented by a configurable run of
    # identical adjacent samples.  The default of eight avoids flagging an
    # ordinary quantized crossing while still catching a clipped segment.
    identical = np.isfinite(differences) & (np.abs(differences) <= np.finfo(np.float32).eps)
    present_saturated = np.zeros((len(present), channels), dtype=bool)
    run_differences = max(1, int(saturation_run_samples) - 1)
    if identical.shape[1] >= run_differences:
        runs = np.lib.stride_tricks.sliding_window_view(identical, run_differences, axis=1)
        present_saturated = np.any(np.all(runs, axis=-1), axis=1)
    present_flat = (ptp < abs(float(flat_ptp))) | (flat_ratio >= float(flat_ratio_percent))
    # A robust within-window threshold avoids mistaking ordinary high-rate
    # recording noise for a jump, while being insensitive to signal offset.
    jump_threshold = difference_median + np.maximum(
        float(jump_mad_multiplier) * difference_mad,
        np.maximum(
            float(jump_median_multiplier) * difference_median,
            abs(float(flat_epsilon)) * float(jump_flat_floor_multiplier),
        ),
    )
    present_jump = max_difference > jump_threshold
    present_clean = present_available & ~present_saturated & ~present_flat & ~present_jump
    clean[present] = present_clean
    available[present] = present_available
    saturated[present] = present_saturated
    flat[present] = present_flat
    jump[present] = present_jump
    return clean, {
        "clean": clean, "available": available, "saturated": saturated,
        "flat": flat, "jump": jump, "unavailable": ~available,
    }


def compute_task_window_spectral_metrics(
    windows: np.ndarray,
    fs: float,
    target_freq_hz: float,
    neighbor_bins: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate target and local-noise power for independent continuous windows.

    ``windows`` is never a concatenation of good periods.  The tapered
    sine/cosine projections compute only the target and the requested local
    neighbor frequencies, avoiding a full FFT/PSD for every trial and channel.
    """
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[1] < 3:
        raise ValueError("Spectral windows must be window x sample x channel arrays.")
    fs = float(fs)
    target_freq_hz = float(target_freq_hz)
    if not 0 < target_freq_hz < fs / 2:
        raise ValueError("Target frequency must be between 0 and the Nyquist frequency.")
    samples = values.shape[1]
    neighbor_bins = max(1, int(neighbor_bins))
    offsets = np.arange(-neighbor_bins, neighbor_bins + 1, dtype=float)
    frequencies = target_freq_hz + offsets * (fs / samples)
    target_index = int(np.where(offsets == 0)[0][0])
    noise_indices = np.flatnonzero((np.arange(frequencies.size) != target_index) & (frequencies > 0) & (frequencies < fs / 2))
    if not noise_indices.size:
        raise ValueError("Target frequency has no valid local neighbor bins.")
    sample_time = np.arange(samples, dtype=np.float64) / fs
    taper = np.hanning(samples)
    basis = taper[:, None] * np.exp(-2j * np.pi * sample_time[:, None] * frequencies[None, :])
    # The coherent-gain scaling reports sine-wave mean-square power.  Invalid
    # samples are made zero here; their windows are excluded by the QC mask
    # before any aggregate is calculated.
    coefficients = np.einsum(
        "wsc,sf->wfc", np.nan_to_num(values, nan=0.0), basis,
        optimize=True,
    ) * (2.0 / np.sum(taper))
    power = (np.abs(coefficients) ** 2) / 2.0
    return power[:, target_index, :], np.nanmean(power[:, noise_indices, :], axis=1)


def _contiguous_true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open index runs for a short one-dimensional boolean mask."""
    padded = np.r_[False, np.asarray(flags, dtype=bool), False]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(first), int(last)) for first, last in edges.reshape(-1, 2)]


def _continuous_segment_psd_metrics(
    values: np.ndarray,
    fs: float,
    target_freq_hz: float,
    neighbor_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract target and neighboring-bin PSD power from one continuous block."""
    from scipy import signal

    segment = np.asarray(values, dtype=np.float32)
    if segment.ndim == 1:
        segment = segment[:, None]
    if segment.ndim != 2 or segment.shape[0] < 3:
        raise ValueError("PSD requires a continuous time x channel segment.")
    freqs, psd = signal.welch(
        np.nan_to_num(segment, nan=0.0), fs=float(fs), window="hann",
        nperseg=segment.shape[0], noverlap=0, detrend="constant", axis=0,
        scaling="density",
    )
    target_index = int(np.argmin(np.abs(freqs - float(target_freq_hz))))
    neighbor_bins = max(1, int(neighbor_bins))
    left = max(0, target_index - neighbor_bins)
    right = min(freqs.size, target_index + neighbor_bins + 1)
    noise_indices = np.r_[left:target_index, target_index + 1:right]
    if not noise_indices.size:
        return psd[target_index], np.full(psd.shape[1], np.nan, dtype=np.float64)
    return psd[target_index], np.nanmean(psd[noise_indices], axis=0)


def compute_task_continuous_psd_metrics(
    epoch: np.ndarray,
    clean_windows: np.ndarray,
    usable_channels: np.ndarray,
    fs: float,
    epoch_start_ms: float,
    target_freq_hz: float,
    neighbor_bins: int = 4,
    min_run_windows: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute task PSD only from naturally contiguous clean one-second runs.

    A trial can pass the 3/5 quality rule with separated clean windows.  Those
    windows are deliberately never joined.  Instead, every run such as W2-W4
    gets its own Welch PSD, and target/noise powers are duration-weighted over
    the run PSDs.  Channels sharing the same five-window mask are processed
    together, limiting the work to at most 32 mask groups per trial.
    """
    values = np.asarray(epoch, dtype=np.float32)
    clean = np.asarray(clean_windows, dtype=bool) & np.asarray(usable_channels, dtype=bool)[None, :]
    if values.ndim != 2 or clean.shape != (5, values.shape[1]):
        raise ValueError("Task PSD quality mask must be 5 x channel.")
    samples_per_window = max(1, int(round(float(fs))))
    steady_first = int(round((500.0 - float(epoch_start_ms)) * float(fs) / 1000.0))
    channels = values.shape[1]
    target_sum = np.zeros(channels, dtype=np.float64)
    noise_sum = np.zeros(channels, dtype=np.float64)
    sample_counts = np.zeros(channels, dtype=np.int64)
    segment_counts = np.zeros(channels, dtype=np.int64)
    patterns = np.sum(clean * (1 << np.arange(5, dtype=np.int8))[:, None], axis=0)
    for pattern in np.unique(patterns):
        if not pattern:
            continue
        columns = np.flatnonzero(patterns == pattern)
        flags = ((int(pattern) >> np.arange(5)) & 1).astype(bool)
        for first_window, last_window in _contiguous_true_runs(flags):
            if last_window - first_window < max(2, int(min_run_windows)):
                continue
            first = steady_first + first_window * samples_per_window
            last = steady_first + last_window * samples_per_window
            if first < 0 or last > values.shape[0]:
                continue
            target_power, noise_power = _continuous_segment_psd_metrics(
                values[first:last, columns], fs, target_freq_hz, neighbor_bins,
            )
            weight = last - first
            target_sum[columns] += target_power * weight
            noise_sum[columns] += noise_power * weight
            sample_counts[columns] += weight
            segment_counts[columns] += 1
    return (
        np.divide(target_sum, sample_counts, out=np.full(channels, np.nan), where=sample_counts > 0),
        np.divide(noise_sum, sample_counts, out=np.full(channels, np.nan), where=sample_counts > 0),
        sample_counts,
        segment_counts,
    )


def compute_task_psd_metrics_by_run_length(
    epoch: np.ndarray,
    clean_windows: np.ndarray,
    usable_channels: np.ndarray,
    fs: float,
    epoch_start_ms: float,
    target_freq_hz: float,
    neighbor_bins: int = 4,
    run_lengths: tuple[int, ...] = (3, 4, 5),
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return PSD aggregates separately for exact contiguous clean-window lengths.

    This is intentionally narrower than ``compute_task_continuous_psd_metrics``:
    it exists only to pair a task run with an external resting baseline of the
    same duration.  Failed windows are never joined across a gap.
    """
    values = np.asarray(epoch, dtype=np.float32)
    clean = np.asarray(clean_windows, dtype=bool) & np.asarray(usable_channels, dtype=bool)[None, :]
    if values.ndim != 2 or clean.shape != (5, values.shape[1]):
        raise ValueError("Task PSD quality mask must be 5 x channel.")
    lengths = tuple(sorted({int(length) for length in run_lengths if 2 <= int(length) <= 5}))
    channels = values.shape[1]
    sums = {
        length: (
            np.zeros(channels, dtype=np.float64), np.zeros(channels, dtype=np.float64),
            np.zeros(channels, dtype=np.int64), np.zeros(channels, dtype=np.int64),
        )
        for length in lengths
    }
    samples_per_window = max(1, int(round(float(fs))))
    steady_first = int(round((500.0 - float(epoch_start_ms)) * float(fs) / 1000.0))
    patterns = np.sum(clean * (1 << np.arange(5, dtype=np.int8))[:, None], axis=0)
    for pattern in np.unique(patterns):
        if not pattern:
            continue
        columns = np.flatnonzero(patterns == pattern)
        flags = ((int(pattern) >> np.arange(5)) & 1).astype(bool)
        for first_window, last_window in _contiguous_true_runs(flags):
            length = last_window - first_window
            if length not in sums:
                continue
            first = steady_first + first_window * samples_per_window
            last = steady_first + last_window * samples_per_window
            if first < 0 or last > values.shape[0]:
                continue
            target_power, noise_power = _continuous_segment_psd_metrics(
                values[first:last, columns], fs, target_freq_hz, neighbor_bins,
            )
            target_sum, noise_sum, sample_counts, segment_counts = sums[length]
            weight = last - first
            target_sum[columns] += target_power * weight
            noise_sum[columns] += noise_power * weight
            sample_counts[columns] += weight
            segment_counts[columns] += 1
    return {
        length: (
            np.divide(target_sum, sample_counts, out=np.full(channels, np.nan), where=sample_counts > 0),
            np.divide(noise_sum, sample_counts, out=np.full(channels, np.nan), where=sample_counts > 0),
            sample_counts, segment_counts,
        )
        for length, (target_sum, noise_sum, sample_counts, segment_counts) in sums.items()
    }


def _stratified_nonoverlapping_starts(
    rows: int, samples_per_segment: int, candidate_count: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sample one continuous segment from each equal-width stratum."""
    rows = int(rows)
    samples_per_segment = int(samples_per_segment)
    maximum = rows // max(1, samples_per_segment)
    count = min(maximum, max(0, int(candidate_count)))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    edges = np.linspace(0, rows, count + 1, dtype=np.int64)
    starts = np.empty(count, dtype=np.int64)
    for index in range(count):
        low = int(edges[index])
        high = int(edges[index + 1]) - samples_per_segment
        if high < low:
            raise ValueError("Baseline recording is too short for the requested segment count.")
        starts[index] = int(rng.integers(low, high + 1))
    return starts


def compute_external_baseline_reference(
    source,
    baseline_columns: np.ndarray,
    task_channel_ids: list[int],
    settings: dict,
    progress=None,
) -> dict:
    """Build robust, length-matched 3/4/5 s resting-baseline references.

    Each candidate is a natural continuous HDF5 slice.  QC uses the same
    five-window thresholds as the task path, and each valid segment receives
    an independent Welch estimate before median aggregation.
    """
    meta = source.metadata
    columns = np.asarray(baseline_columns, dtype=np.int64)
    if columns.ndim != 1 or not columns.size:
        raise ValueError("No task channels are present in the baseline HDF5.")
    fs = float(meta.fs)
    lengths = (3, 4, 5)
    candidate_count = int(settings.get("baseline_candidate_count", 12))
    minimum_valid = int(settings.get("baseline_min_valid_segments", 5))
    if candidate_count < 1 or minimum_valid < 1:
        raise ValueError("Baseline candidate and minimum-valid counts must be positive.")
    if minimum_valid > candidate_count:
        raise ValueError("Baseline minimum-valid count cannot exceed the candidate count.")
    if meta.rows < int(round(5.0 * fs)):
        raise ValueError("Baseline recording must contain at least one continuous 5 s segment.")
    values = np.asarray(source.read(0, meta.rows, columns), dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if settings.get("analysis_notch") or settings.get("analysis_bandpass"):
        mode = "bandpass" if settings.get("analysis_bandpass") else "off"
        values, _ = filter_array(
            values, fs, mode,
            float(settings.get("analysis_band_low", 0.5)),
            float(settings.get("analysis_band_high", 300.0)),
            notch=bool(settings.get("analysis_notch", False)),
        )
    if settings.get("smooth"):
        width = max(1, int(round(float(settings.get("smooth_window_sec", 0.02)) * fs)))
        if width > 1:
            kernel = np.ones(width, dtype=np.float32) / width
            values = np.apply_along_axis(lambda item: np.convolve(item, kernel, mode="same"), 0, values)

    quality_settings = {
        "flat_epsilon": float(settings["quality_flat_epsilon"]),
        "flat_ratio_percent": float(settings["quality_flat_ratio_percent"]),
        "flat_ptp": float(settings["quality_flat_ptp"]),
        "available_ratio": float(settings["quality_available_ratio"]),
        "saturation_run_samples": int(settings["quality_saturation_run_samples"]),
        "jump_mad_multiplier": float(settings["quality_jump_mad_multiplier"]),
        "jump_median_multiplier": float(settings["quality_jump_median_multiplier"]),
        "jump_flat_floor_multiplier": float(settings["quality_jump_flat_floor_multiplier"]),
    }
    rng = np.random.default_rng(int(settings.get("baseline_random_seed", 20260817)))
    baseline_ids = [int(meta.channel_ids[index]) for index in columns]
    references = {}
    for length_index, length in enumerate(lengths, start=1):
        samples = int(round(length * fs))
        starts = _stratified_nonoverlapping_starts(meta.rows, samples, candidate_count, rng)
        target_values, noise_values = [], []
        valid_counts = np.zeros(columns.size, dtype=np.int64)
        rejected = {name: np.zeros(columns.size, dtype=np.int64) for name in ("unavailable", "saturated", "flat", "jump")}
        for segment_index, start in enumerate(starts, start=1):
            segment = values[start:start + samples]
            clean, quality = evaluate_task_trial_window_quality(segment, fs, 500.0, **quality_settings)
            valid = np.all(clean[:length], axis=0)
            for name in rejected:
                rejected[name] += np.any(quality[name][:length], axis=0)
            target, noise = _continuous_segment_psd_metrics(
                segment, fs, float(settings["task_target_freq_hz"]), int(settings["task_neighbor_bins"]),
            )
            valid &= np.isfinite(target) & np.isfinite(noise)
            target_values.append(np.where(valid, target, np.nan))
            noise_values.append(np.where(valid, noise, np.nan))
            valid_counts += valid
            if progress is not None:
                progress.emit(
                    100.0 * ((length_index - 1) + segment_index / max(1, starts.size)) / len(lengths),
                    f"实验前 baseline：{length} s 候选片段 {segment_index}/{starts.size}",
                )
        target_array = np.asarray(target_values, dtype=np.float64)
        noise_array = np.asarray(noise_values, dtype=np.float64)
        with np.errstate(all="ignore"):
            target_median = np.nanmedian(target_array, axis=0)
            noise_median = np.nanmedian(noise_array, axis=0)
            target_iqr = np.nanpercentile(target_array, 75, axis=0) - np.nanpercentile(target_array, 25, axis=0)
            snr_values = 10.0 * np.log10(target_array / noise_array)
            snr_median = np.nanmedian(snr_values, axis=0)
            snr_iqr = np.nanpercentile(snr_values, 75, axis=0) - np.nanpercentile(snr_values, 25, axis=0)
        eligible = valid_counts >= minimum_valid
        target_median = np.where(eligible, target_median, np.nan)
        noise_median = np.where(eligible, noise_median, np.nan)
        snr_median = np.where(eligible, snr_median, np.nan)
        references[length] = {
            "candidate_starts": starts,
            "target_power": target_median,
            "local_noise_power": noise_median,
            "local_snr_db": snr_median,
            "target_power_iqr": target_iqr,
            "local_snr_iqr_db": snr_iqr,
            "valid_segment_counts": valid_counts,
            "eligible": eligible,
            "rejected": rejected,
        }
    time_reference = None
    if all(name in settings for name in ("epoch_start", "epoch_end", "baseline_start", "baseline_end", "response_start", "response_end")):
        epoch_start = float(settings["epoch_start"])
        epoch_end = float(settings["epoch_end"])
        epoch_samples = int(round((epoch_end - epoch_start) * fs / 1000.0))
        time_ms = np.arange(epoch_samples, dtype=float) / fs * 1000.0 + epoch_start
        baseline_mask = (time_ms >= float(settings["baseline_start"])) & (time_ms <= float(settings["baseline_end"]))
        time_response_start = float(settings.get("external_time_response_start", settings["response_start"]))
        time_response_end = float(settings.get("external_time_response_end", settings["response_end"]))
        response_mask = (time_ms >= time_response_start) & (time_ms <= time_response_end)
        if epoch_samples < 3 or not np.any(baseline_mask) or not np.any(response_mask):
            raise ValueError("External time-domain baseline requires valid epoch, baseline, and response ranges.")
        time_rate_hz = min(fs, max(10.0, float(settings.get("external_time_rate_hz", 250.0))))
        time_step = max(1, int(round(fs / time_rate_hz)))
        response_indices = np.flatnonzero(response_mask)[::time_step]
        starts = _stratified_nonoverlapping_starts(meta.rows, epoch_samples, candidate_count, rng)
        traces = np.full((starts.size, response_indices.size, columns.size), np.nan, dtype=np.float32)
        valid_counts = np.zeros(columns.size, dtype=np.int64)
        rejected = {name: np.zeros(columns.size, dtype=np.int64) for name in ("unavailable", "saturated", "flat", "jump", "response_missing")}
        for segment_index, start in enumerate(starts, start=1):
            epoch = values[start:start + epoch_samples]
            clean, quality = evaluate_task_trial_window_quality(epoch, fs, epoch_start, **quality_settings)
            available_windows = np.sum(quality["available"], axis=0)
            usable = (available_windows > 0) & (np.sum(clean, axis=0) >= np.minimum(3, available_windows))
            finite_response = np.mean(np.isfinite(epoch[response_mask]), axis=0) >= float(quality_settings["available_ratio"])
            finite_baseline = np.mean(np.isfinite(epoch[baseline_mask]), axis=0) >= float(quality_settings["available_ratio"])
            valid = usable & finite_response & finite_baseline
            for name in ("unavailable", "saturated", "flat", "jump"):
                rejected[name] += np.any(quality[name], axis=0)
            rejected["response_missing"] += ~(finite_response & finite_baseline)
            corrected = epoch - np.nanmean(epoch[baseline_mask], axis=0, keepdims=True)
            traces[segment_index - 1] = np.where(valid[None, :], corrected[response_indices], np.nan)
            valid_counts += valid
            if progress is not None:
                progress.emit(
                    100.0 * ((segment_index / max(1, starts.size)) * .2 + .8),
                    f"实验前 baseline：时域伪 epoch {segment_index}/{starts.size}",
                )
        time_reference = {
            "candidate_starts": starts,
            "traces": traces,
            "time_ms": time_ms[response_indices],
            "sample_rate_hz": fs / time_step,
            "valid_segment_counts": valid_counts,
            "eligible": valid_counts >= minimum_valid,
            "rejected": rejected,
        }
    return {
        "channel_ids": baseline_ids,
        "fs": fs,
        "path": str(meta.path),
        "dataset": str(meta.dataset),
        "rows": int(meta.rows),
        "candidate_count": candidate_count,
        "minimum_valid_segments": minimum_valid,
        "random_seed": int(settings.get("baseline_random_seed", 20260817)),
        "references": references,
        "time_reference": time_reference,
    }


class ExternalBaselineWorker(QThread):
    """Build the task-only external baseline reference without blocking the UI."""
    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, baseline_columns, task_channel_ids, settings):
        super().__init__()
        self.source = source
        self.baseline_columns = np.asarray(baseline_columns, dtype=np.int64)
        self.task_channel_ids = [int(channel) for channel in task_channel_ids]
        self.settings = dict(settings)

    def run(self) -> None:
        try:
            self.completed.emit(compute_external_baseline_reference(
                self.source, self.baseline_columns, self.task_channel_ids, self.settings, self.progress,
            ))
        except Exception as exc:
            self.failed.emit(str(exc))


def compute_external_rest_cluster_test(
    task_traces: np.ndarray,
    rest_traces: np.ndarray,
    *,
    permutations: int = 1000,
    cluster_forming_p: float = .01,
    cluster_significance_p: float = .01,
    random_seed: int = 20260709,
) -> dict:
    """Run a two-sided independent temporal cluster permutation test.

    Rows are independent task trials or non-overlapping resting pseudo-epochs.
    Only complete finite rows are admitted, so every time point has exactly
    the same observations.
    """
    from scipy import stats
    from mne.stats import permutation_cluster_test

    task = np.asarray(task_traces, dtype=float)
    rest = np.asarray(rest_traces, dtype=float)
    if task.ndim != 2 or rest.ndim != 2 or task.shape[1:] != rest.shape[1:]:
        raise ValueError("Task and resting traces must be two-dimensional with the same time axis.")
    task = task[np.all(np.isfinite(task), axis=1)]
    rest = rest[np.all(np.isfinite(rest), axis=1)]
    output = {
        "clusters": [],
        "task_trials": int(task.shape[0]),
        "rest_epochs": int(rest.shape[0]),
        "cluster_forming_p": float(cluster_forming_p),
        "cluster_significance_p": float(cluster_significance_p),
        "permutations": int(permutations),
    }
    if task.shape[0] < 3 or rest.shape[0] < 3:
        return {**output, "status": "insufficient_epochs", "t_obs": np.full(task.shape[1], np.nan, dtype=np.float32)}

    threshold = float(stats.t.ppf(
        1.0 - float(cluster_forming_p) / 2.0,
        task.shape[0] + rest.shape[0] - 2,
    ))

    def independent_t(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return stats.ttest_ind(first, second, axis=0, equal_var=False).statistic

    t_obs, clusters, cluster_p_values, _null = permutation_cluster_test(
        [task, rest], threshold=threshold, tail=0, stat_fun=independent_t,
        n_permutations=int(permutations), seed=int(random_seed), out_type="mask", verbose=False,
    )
    entries = []
    for cluster, p_value in zip(clusters, cluster_p_values):
        if isinstance(cluster, slice):
            indices = np.arange(task.shape[1], dtype=np.int64)[cluster]
        elif isinstance(cluster, tuple):
            component = cluster[0]
            if isinstance(component, slice):
                indices = np.arange(task.shape[1], dtype=np.int64)[component]
            else:
                component = np.asarray(component)
                indices = np.flatnonzero(component) if component.dtype == bool else component.astype(np.int64, copy=False)
        else:
            indices = np.flatnonzero(np.asarray(cluster, dtype=bool))
        if not indices.size:
            continue
        mass = float(np.nansum(t_obs[indices]))
        entries.append({
            "first": int(indices[0]), "last": int(indices[-1]), "p_value": float(p_value),
            "significant": bool(p_value < float(cluster_significance_p)),
            "direction": "task>rest" if mass >= 0 else "task<rest", "mass": mass,
        })
    return {**output, "status": "ok", "clusters": entries, "t_obs": np.asarray(t_obs, dtype=np.float32)}


def compute_external_rest_tf_cluster_test(
    task_maps: np.ndarray,
    rest_maps: np.ndarray,
    *,
    permutations: int = 1000,
    cluster_forming_p: float = .01,
    cluster_significance_p: float = .01,
    random_seed: int = 20260710,
) -> dict:
    """Two-sided independent cluster test over frequency x time power maps."""
    from scipy import stats
    from mne.stats import permutation_cluster_test

    task = np.asarray(task_maps, dtype=float)
    rest = np.asarray(rest_maps, dtype=float)
    if task.ndim != 3 or rest.ndim != 3 or task.shape[1:] != rest.shape[1:]:
        raise ValueError("Task and resting time-frequency maps must share frequency and time axes.")
    task = task[np.all(np.isfinite(task), axis=(1, 2))]
    rest = rest[np.all(np.isfinite(rest), axis=(1, 2))]
    shape = task.shape[1:]
    output = {
        "clusters": [], "task_trials": int(task.shape[0]), "rest_epochs": int(rest.shape[0]),
        "cluster_forming_p": float(cluster_forming_p),
        "cluster_significance_p": float(cluster_significance_p), "permutations": int(permutations),
        "significant_mask": np.zeros(shape, dtype=bool),
    }
    if task.shape[0] < 3 or rest.shape[0] < 3:
        return {**output, "status": "insufficient_epochs", "t_obs": np.full(shape, np.nan, dtype=np.float32)}

    threshold = float(stats.t.ppf(1.0 - float(cluster_forming_p) / 2.0, task.shape[0] + rest.shape[0] - 2))

    def independent_t(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return stats.ttest_ind(first, second, axis=0, equal_var=False).statistic

    t_obs, clusters, cluster_p_values, _null = permutation_cluster_test(
        [task, rest], threshold=threshold, tail=0, stat_fun=independent_t,
        n_permutations=int(permutations), seed=int(random_seed), out_type="mask", verbose=False,
    )
    entries = []
    significant_mask = np.zeros(shape, dtype=bool)
    for cluster, p_value in zip(clusters, cluster_p_values):
        if isinstance(cluster, tuple):
            mask = np.zeros(shape, dtype=bool)
            mask[cluster] = True
        else:
            mask = np.asarray(cluster, dtype=bool)
            if mask.shape != shape:
                mask = mask.reshape(shape)
        positions = np.argwhere(mask)
        if not positions.size:
            continue
        mass = float(np.nansum(t_obs[mask]))
        significant = bool(p_value < float(cluster_significance_p))
        if significant:
            significant_mask |= mask
        entries.append({
            "freq_first": int(positions[:, 0].min()), "freq_last": int(positions[:, 0].max()),
            "time_first": int(positions[:, 1].min()), "time_last": int(positions[:, 1].max()),
            "p_value": float(p_value), "significant": significant,
            "direction": "task>rest" if mass >= 0 else "task<rest", "mass": mass,
        })
    return {**output, "status": "ok", "clusters": entries, "significant_mask": significant_mask, "t_obs": np.asarray(t_obs, dtype=np.float32)}


class TaskEpochWorker(QThread):
    """Marker-driven Flash/Letter epoch analysis using legacy task parameters."""
    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, markers, settings):
        super().__init__(); self.source = source; self.markers = np.asarray(markers, dtype=int); self.settings = dict(settings)

    def run(self) -> None:
        try:
            meta = self.source.metadata
            columns = np.asarray(self.settings.get("columns", np.arange(meta.channels)), dtype=np.int64)
            channel_ids = [int(meta.channel_ids[index]) for index in columns]
            requires_full_record = any((
                self.settings.get("analysis_notch"), self.settings.get("analysis_bandpass"),
                self.settings.get("smooth"), self.settings.get("zscore"),
            ))
            data = None
            if requires_full_record:
                self.progress.emit(1, "任务分析：分析参数要求读取完整记录…")
                raw, _ = self.source.materialize(
                    lambda value, text: self.progress.emit(1 + value * 34, text)
                ) if not hasattr(self.source, "data") else (self.source.data, None)
                data = np.asarray(raw[:, columns], dtype=np.float32)
            else:
                # The normal task path mirrors make_epochs() while avoiding a
                # multi-gigabyte materialize just to inspect short epochs.
                self.progress.emit(1, "任务分析：按刺激窗口读取所选通道…")

            def read_epoch(first: int, last: int) -> np.ndarray:
                if data is not None:
                    return np.asarray(data[first:last], dtype=np.float32)
                return np.asarray(self.source.read(first, last, columns), dtype=np.float32)

            # Apply the legacy analysis-preprocess controls before epoching.
            # These are deliberately separate from preprocessing/LFP filters:
            # users can retain the same loaded data and compare task settings.
            if self.settings.get("analysis_notch") or self.settings.get("analysis_bandpass"):
                mode = "bandpass" if self.settings.get("analysis_bandpass") else "off"
                data, _ = filter_array(
                    data, meta.fs, mode,
                    float(self.settings.get("analysis_band_low", 0.5)),
                    float(self.settings.get("analysis_band_high", 300.0)),
                    notch=bool(self.settings.get("analysis_notch", False)),
                )
            if self.settings.get("smooth"):
                width = max(1, int(round(float(self.settings.get("smooth_window_sec", 0.02)) * meta.fs)))
                if width > 1:
                    kernel = np.ones(width, dtype=np.float32) / width
                    data = np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="same"), 0, data)
            if self.settings.get("zscore"):
                average = np.nanmean(data, axis=0, keepdims=True)
                deviation = np.nanstd(data, axis=0, keepdims=True)
                deviation[deviation == 0] = 1.0
                data = (data - average) / deviation
            start = int(round(self.settings["epoch_start"] * meta.fs / 1000)); end = int(round(self.settings["epoch_end"] * meta.fs / 1000))
            if end <= start: raise ValueError("Epoch end must be after epoch start.")
            # Markers are stored as global EEG sample numbers.  A processed
            # export may begin later than the original recording, so this is
            # the exact legacy marker_offset_samples() conversion before any
            # epoch bounds check or HDF5 slice is made.
            marker_offset = int(round(float(meta.time_offset) * meta.fs))
            # Exact port of IntegratedPipelineGUI.selected_stim_tags(): the
            # shared stimtag field is honoured for both Flash and Letter, and
            # nStimTypes limits the final ordered list.  Do not replace a
            # user's Letter tags with a Qt-only automatic default.
            available = [int(tag) for tag in np.unique(self.markers[:, 1])]
            requested_input = [int(tag) for tag in self.settings["tags"]]
            requested = [tag for tag in requested_input if tag in available] if requested_input else available
            n_stim_types = max(0, int(self.settings.get("n_stim_types", len(requested))))
            if n_stim_types > 0:
                requested = requested[:n_stim_types]
            if not requested:
                raise ValueError(f"stimtag 中没有当前 marker。可用标签: {available}；当前 stimtag: {requested_input}")
            result = []
            t_ms = np.arange(start, end) / meta.fs * 1000
            base_mask = (t_ms >= self.settings["baseline_start"]) & (t_ms <= self.settings["baseline_end"])
            baseline_duration_ms = float(np.count_nonzero(base_mask) / meta.fs * 1000.0)
            response_mask = (t_ms >= self.settings["response_start"]) & (t_ms <= self.settings["response_end"])
            external_time_response_start = float(self.settings.get("external_time_response_start", self.settings["response_start"]))
            external_time_response_end = float(self.settings.get("external_time_response_end", self.settings["response_end"]))
            external_time_response_mask = (
                (t_ms >= external_time_response_start) & (t_ms <= external_time_response_end)
            )
            configured_a = max(0, int(self.settings.get("compare_trial_a", 0)))
            configured_b = max(0, int(self.settings.get("compare_trial_b", 0)))
            quality_settings = {
                "flat_epsilon": float(self.settings.get("quality_flat_epsilon", 1e-4)),
                "flat_ratio_percent": float(self.settings.get("quality_flat_ratio_percent", 30.0)),
                "flat_ptp": float(self.settings.get("quality_flat_ptp", 0.01)),
                "available_ratio": float(self.settings.get("quality_available_ratio", .995)),
                "saturation_run_samples": int(self.settings.get("quality_saturation_run_samples", 8)),
                "jump_mad_multiplier": float(self.settings.get("quality_jump_mad_multiplier", 12.0)),
                "jump_median_multiplier": float(self.settings.get("quality_jump_median_multiplier", 8.0)),
                "jump_flat_floor_multiplier": float(self.settings.get("quality_jump_flat_floor_multiplier", 10.0)),
            }
            target_frequency = float(self.settings.get("task_target_freq_hz", 1.0))
            local_neighbor_bins = max(1, int(self.settings.get("task_neighbor_bins", 4)))
            # Direct worker callers are historically LFP task callers;
            # the Spike page explicitly passes False below.
            lfp_task_metrics = bool(self.settings.get("lfp_task_metrics", True))
            quality_only = bool(self.settings.get("quality_only", False))
            precomputed_trial_quality = self.settings.get("precomputed_trial_quality") or {}
            external_baseline_reference = self.settings.get("external_baseline_reference") or None
            external_baseline_by_length = {}
            external_time_reference = None
            external_time_channel_indices = np.full(columns.size, -1, dtype=np.int64)
            external_time_task_indices = np.empty(0, dtype=np.int64)
            if external_baseline_reference is not None:
                reference_ids = [int(channel) for channel in external_baseline_reference.get("channel_ids", [])]
                reference_lookup = {channel: index for index, channel in enumerate(reference_ids)}
                for length, reference in (external_baseline_reference.get("references") or {}).items():
                    target = np.full(columns.size, np.nan, dtype=np.float64)
                    noise = np.full(columns.size, np.nan, dtype=np.float64)
                    valid_count = np.zeros(columns.size, dtype=np.int64)
                    eligible = np.zeros(columns.size, dtype=bool)
                    for channel_index, channel_id in enumerate(channel_ids):
                        baseline_index = reference_lookup.get(int(channel_id))
                        if baseline_index is None:
                            continue
                        target[channel_index] = float(reference["target_power"][baseline_index])
                        noise[channel_index] = float(reference["local_noise_power"][baseline_index])
                        valid_count[channel_index] = int(reference["valid_segment_counts"][baseline_index])
                        eligible[channel_index] = bool(reference["eligible"][baseline_index])
                    external_baseline_by_length[int(length)] = (target, noise, valid_count, eligible)
                candidate_time_reference = external_baseline_reference.get("time_reference")
                if candidate_time_reference is not None and lfp_task_metrics and not quality_only:
                    reference_time_ms = np.asarray(candidate_time_reference.get("time_ms"), dtype=float)
                    reference_rate_hz = float(candidate_time_reference.get("sample_rate_hz", 0.0))
                    time_step = max(1, int(round(meta.fs / reference_rate_hz))) if reference_rate_hz > 0 else 0
                    task_time_indices = np.flatnonzero(external_time_response_mask)[::time_step] if time_step else np.empty(0, dtype=np.int64)
                    if (
                        task_time_indices.size == reference_time_ms.size
                        and task_time_indices.size > 0
                        and np.allclose(t_ms[task_time_indices], reference_time_ms, rtol=0, atol=.51 * 1000.0 / meta.fs)
                    ):
                        external_time_reference = candidate_time_reference
                        external_time_task_indices = task_time_indices
                        for channel_index, channel_id in enumerate(channel_ids):
                            external_time_channel_indices[channel_index] = reference_lookup.get(int(channel_id), -1)
            if lfp_task_metrics and not 0 < target_frequency < meta.fs / 2:
                raise ValueError("任务目标频率必须介于 0 和奈奎斯特频率之间。")
            skipped = []
            trial_quality = {}
            for number, tag in enumerate(requested, start=1):
                candidates = self.markers[self.markers[:, 1] == int(tag)]
                used_samples = []
                if self.settings.get("mode") == "Letter":
                    # 0 means all available, consistently with the task
                    # parameter panel.  A/B only limits collection when a
                    # positive comparison count is actually supplied.
                    limit = max(configured_a, configured_b, int(self.settings["trials_per_stim"]))
                else:
                    limit = self.settings["trials_per_stim"]
                # Mean is the reference default.  Accumulate it one epoch at
                # a time so trialsPerStim=0 remains practical for a large
                # 520-channel recording; the old stacked calculation and
                # this running sum are mathematically identical for Mean.
                streaming_mean = self.settings["aggregate"] != "median"
                epochs = []
                wave_sum = None
                wave_counts = np.zeros(columns.size, dtype=np.int64)
                trial_time_means = []
                processed_count = 0
                accepted_trial_count = 0
                usable_trials = np.zeros(columns.size, dtype=np.int64)
                clean_window_counts = np.zeros(columns.size, dtype=np.int64)
                available_window_counts = np.zeros(columns.size, dtype=np.int64)
                short_epoch_trials = np.zeros(columns.size, dtype=np.int64)
                response_only_trials = np.zeros(columns.size, dtype=np.int64)
                spectral_sample_counts = np.zeros(columns.size, dtype=np.int64)
                spectral_segment_counts = np.zeros(columns.size, dtype=np.int64)
                target_power_sum = np.zeros(columns.size, dtype=np.float64)
                local_noise_power_sum = np.zeros(columns.size, dtype=np.float64)
                baseline_power_sum = np.zeros(columns.size, dtype=np.float64)
                baseline_power_counts = np.zeros(columns.size, dtype=np.int64)
                event_baseline_power_sum = np.zeros(columns.size, dtype=np.float64)
                event_response_power_sum = np.zeros(columns.size, dtype=np.float64)
                event_power_counts = np.zeros(columns.size, dtype=np.int64)
                external_task_target_sum = np.zeros(columns.size, dtype=np.float64)
                external_task_noise_sum = np.zeros(columns.size, dtype=np.float64)
                external_baseline_target_sum = np.zeros(columns.size, dtype=np.float64)
                external_baseline_noise_sum = np.zeros(columns.size, dtype=np.float64)
                external_matched_samples = np.zeros(columns.size, dtype=np.int64)
                external_matched_segments = np.zeros(columns.size, dtype=np.int64)
                external_matched_samples_by_length = {
                    length: np.zeros(columns.size, dtype=np.int64)
                    for length in external_baseline_by_length
                }
                quality_reasons = {
                    name: np.zeros(columns.size, dtype=np.int64)
                    for name in ("unavailable", "saturated", "flat", "jump")
                }
                external_time_task_trials = []
                for marker_number, marker in enumerate(candidates[:limit or None], start=1):
                    center = int(marker[0]) - marker_offset
                    first, last = center + start, center + end
                    if first >= 0 and last <= meta.rows:
                        epoch = read_epoch(first, last)
                        # Frequency metrics stay on CAR-LFP values.  Do not
                        # use the ERP baseline-corrected copy below, because
                        # the two products answer different questions.
                        spectral_epoch = epoch
                        if np.any(base_mask):
                            epoch = epoch - np.nanmean(epoch[base_mask], axis=0, keepdims=True)
                        if lfp_task_metrics:
                            quality_key = (int(tag), int(marker[0]))
                            saved_quality = precomputed_trial_quality.get(quality_key)
                            if saved_quality is None:
                                clean_windows, quality = evaluate_task_trial_window_quality(
                                    epoch, meta.fs, self.settings["epoch_start"], **quality_settings,
                                )
                            else:
                                clean_windows = np.asarray(saved_quality["clean_windows"], dtype=bool)
                                quality = {
                                    name: np.asarray(saved_quality[name], dtype=bool)
                                    for name in (*quality_reasons, "available")
                                }
                            if quality_only:
                                trial_quality[quality_key] = {
                                    "clean_windows": clean_windows,
                                    **quality,
                                }
                            available_windows = np.sum(quality["available"], axis=0)
                            # The full protocol requires three clean windows.
                            # For a deliberately short epoch, apply the same
                            # standard to the windows that actually exist:
                            # two complete clean windows can support a 2 s
                            # PSD, while absent W3-W5 remain unavailable.
                            required_clean_windows = np.minimum(3, available_windows)
                            quality_usable = (available_windows > 0) & (
                                np.sum(clean_windows, axis=0) >= required_clean_windows
                            )
                            # Very short epochs can still contain the
                            # configured baseline and response ranges even
                            # though no full one-second QC window exists.
                            # Keep those waveforms for response comparison;
                            # their PSD fields remain empty because no
                            # continuous two-second segment is available.
                            response_samples = base_mask | response_mask
                            response_usable = (
                                np.mean(np.isfinite(epoch[response_samples]), axis=0) >= .995
                                if np.any(response_samples) else np.ones(columns.size, dtype=bool)
                            )
                            usable = np.where(available_windows > 0, quality_usable, response_usable)
                            if not quality_only:
                                target_power, local_noise_power, psd_samples, psd_segments = compute_task_continuous_psd_metrics(
                                    spectral_epoch, clean_windows, usable, meta.fs,
                                    self.settings["epoch_start"], target_frequency,
                                    local_neighbor_bins,
                                )
                                valid_psd = psd_samples > 0
                                target_power_sum += np.where(valid_psd, target_power * psd_samples, 0.0)
                                local_noise_power_sum += np.where(valid_psd, local_noise_power * psd_samples, 0.0)
                                spectral_sample_counts += psd_samples
                                spectral_segment_counts += psd_segments
                                if external_baseline_by_length:
                                    run_metrics = compute_task_psd_metrics_by_run_length(
                                        spectral_epoch, clean_windows, usable, meta.fs,
                                        self.settings["epoch_start"], target_frequency, local_neighbor_bins,
                                        tuple(external_baseline_by_length),
                                    )
                                    for length, (run_target, run_noise, run_samples, run_segments) in run_metrics.items():
                                        baseline_target, baseline_noise, _valid_count, eligible = external_baseline_by_length[length]
                                        matched = (
                                            (run_samples > 0) & eligible
                                            & np.isfinite(run_target) & np.isfinite(run_noise)
                                            & np.isfinite(baseline_target) & np.isfinite(baseline_noise)
                                        )
                                        external_task_target_sum += np.where(matched, run_target * run_samples, 0.0)
                                        external_task_noise_sum += np.where(matched, run_noise * run_samples, 0.0)
                                        external_baseline_target_sum += np.where(matched, baseline_target * run_samples, 0.0)
                                        external_baseline_noise_sum += np.where(matched, baseline_noise * run_samples, 0.0)
                                        external_matched_samples += np.where(matched, run_samples, 0)
                                        external_matched_segments += np.where(matched, run_segments, 0)
                                        external_matched_samples_by_length[length] += np.where(matched, run_samples, 0)
                                # The existing task baseline fields define a
                                # continuous pre-stim interval.  One complete
                                # target-frequency cycle is required for its PSD.
                                minimum_baseline_samples = max(3, int(np.ceil(meta.fs / target_frequency)))
                                if np.count_nonzero(base_mask) >= minimum_baseline_samples:
                                    baseline_power, _ = _continuous_segment_psd_metrics(
                                        spectral_epoch[base_mask], meta.fs, target_frequency, local_neighbor_bins,
                                    )
                                    valid_baseline = usable & np.isfinite(baseline_power)
                                    baseline_power_sum += np.where(valid_baseline, baseline_power, 0.0)
                                    baseline_power_counts += valid_baseline
                                # Legacy event-locked LFP SNR is a time-domain
                                # power ratio.  It is intentionally distinct
                                # from local tag-SNR, whose denominator is
                                # neighbouring-frequency PSD power.
                                if np.any(base_mask) and np.any(response_mask):
                                    event_baseline_power = np.nanmean(epoch[base_mask] ** 2, axis=0)
                                    event_response_power = np.nanmean(epoch[response_mask] ** 2, axis=0)
                                    event_valid = (
                                        usable
                                        & np.isfinite(event_baseline_power)
                                        & np.isfinite(event_response_power)
                                    )
                                    event_baseline_power_sum += np.where(event_valid, event_baseline_power, 0.0)
                                    event_response_power_sum += np.where(event_valid, event_response_power, 0.0)
                                    event_power_counts += event_valid
                                if external_time_reference is not None:
                                    finite_task_baseline = np.mean(np.isfinite(spectral_epoch[base_mask]), axis=0) >= quality_settings["available_ratio"]
                                    finite_task_response = np.mean(np.isfinite(spectral_epoch[external_time_response_mask]), axis=0) >= quality_settings["available_ratio"]
                                    time_usable = usable & finite_task_baseline & finite_task_response & (external_time_channel_indices >= 0)
                                    corrected_epoch = spectral_epoch - np.nanmean(
                                        spectral_epoch[base_mask], axis=0, keepdims=True,
                                    )
                                    trace = corrected_epoch[external_time_task_indices].astype(np.float32, copy=False)
                                    external_time_task_trials.append(np.where(time_usable[None, :], trace, np.nan))
                        else:
                            # Spike task epochs retain the original marker
                            # response aggregation only.  LFP quality/PSD
                            # criteria never apply to the Spike page.
                            clean_windows = np.zeros((5, columns.size), dtype=bool)
                            available_windows = np.full(columns.size, 5, dtype=np.int64)
                            usable = np.ones(columns.size, dtype=bool)
                            quality = {name: np.zeros((5, columns.size), dtype=bool) for name in quality_reasons}
                        usable_trials += usable
                        clean_window_counts += np.sum(clean_windows, axis=0)
                        available_window_counts += available_windows
                        short_epoch_trials += available_windows < 5
                        response_only_trials += (available_windows == 0) & usable
                        for reason, counts in quality_reasons.items():
                            counts += np.sum(quality[reason], axis=0)
                        accepted_trial_count += int(np.any(usable))
                        masked_epoch = np.where(usable[None, :], epoch, np.nan)
                        if streaming_mean:
                            if wave_sum is None:
                                wave_sum = np.zeros(epoch.shape, dtype=np.float64)
                            wave_sum += np.where(np.isfinite(masked_epoch), masked_epoch, 0.0)
                            wave_counts += usable
                            # Legacy ``axis=(0, 2)`` leaves the epoch time
                            # axis intact: for each trial, average channels
                            # (axis=1), never average time (axis=0).
                            channel_count = int(np.sum(usable))
                            trial_time_means.append(
                                np.sum(np.where(np.isfinite(masked_epoch), masked_epoch, 0.0), axis=1) / channel_count
                                if channel_count else np.full(epoch.shape[0], np.nan, dtype=np.float64)
                            )
                        else:
                            epochs.append(masked_epoch)
                        used_samples.append(int(marker[0])); processed_count += 1
                    if marker_number % 8 == 0:
                        self.progress.emit(2 + 30 * (number - 1 + marker_number / max(1, len(candidates))) / max(1, len(requested)), f"任务分析：tag {tag}，已读取 {processed_count} 个 trial")
                if not processed_count:
                    skipped.append(
                        f"tag {int(tag)}：{len(candidates)} 个 marker 都不在当前数据段的完整 epoch 范围内 "
                        f"(数据全局样本范围 {marker_offset}–{marker_offset + meta.rows - 1})"
                    )
                    continue
                if not np.any(usable_trials) and not quality_only:
                    skipped.append(
                        f"tag {int(tag)}：{processed_count} 个完整 trial 中，没有任何通道在 0.5-5.5 s 的五个窗口内通过至少 3 个干净窗口。"
                    )
                    continue
                array = None if streaming_mean else np.stack(epochs)
                wave = (
                    np.divide(wave_sum, wave_counts[None, :], out=np.full_like(wave_sum, np.nan), where=wave_counts[None, :] > 0)
                    if streaming_mean else np.nanmedian(array, axis=0)
                )
                response = wave[response_mask] if np.any(response_mask) else wave
                finite_response = np.any(np.isfinite(response), axis=0)
                peak_idx = np.argmax(np.where(np.isfinite(response), np.abs(response), -np.inf), axis=0)
                peak_amp = np.where(finite_response, response[peak_idx, np.arange(response.shape[1])], np.nan)
                response_times = t_ms[response_mask] if np.any(response_mask) else t_ms
                peak_time = np.where(finite_response, response_times[peak_idx], np.nan)
                target_power_mean = np.divide(
                    target_power_sum, spectral_sample_counts,
                    out=np.full(columns.size, np.nan), where=spectral_sample_counts > 0,
                )
                local_noise_power_mean = np.divide(
                    local_noise_power_sum, spectral_sample_counts,
                    out=np.full(columns.size, np.nan), where=spectral_sample_counts > 0,
                )
                baseline_power_mean = np.divide(
                    baseline_power_sum, baseline_power_counts,
                    out=np.full(columns.size, np.nan), where=baseline_power_counts > 0,
                )
                event_baseline_power_mean = np.divide(
                    event_baseline_power_sum, event_power_counts,
                    out=np.full(columns.size, np.nan), where=event_power_counts > 0,
                )
                event_response_power_mean = np.divide(
                    event_response_power_sum, event_power_counts,
                    out=np.full(columns.size, np.nan), where=event_power_counts > 0,
                )
                eps = np.finfo(np.float64).tiny
                target_power_change_db = 10.0 * np.log10(
                    np.maximum(target_power_mean, eps) / np.maximum(baseline_power_mean, eps)
                )
                local_tag_snr_db = 10.0 * np.log10(
                    np.maximum(target_power_mean, eps) / np.maximum(local_noise_power_mean, eps)
                )
                event_lfp_snr_db = 10.0 * np.log10(
                    np.maximum(event_response_power_mean, eps) / np.maximum(event_baseline_power_mean, eps)
                )
                external_task_target_power = np.divide(
                    external_task_target_sum, external_matched_samples,
                    out=np.full(columns.size, np.nan), where=external_matched_samples > 0,
                )
                external_task_noise_power = np.divide(
                    external_task_noise_sum, external_matched_samples,
                    out=np.full(columns.size, np.nan), where=external_matched_samples > 0,
                )
                external_baseline_target_power = np.divide(
                    external_baseline_target_sum, external_matched_samples,
                    out=np.full(columns.size, np.nan), where=external_matched_samples > 0,
                )
                external_baseline_noise_power = np.divide(
                    external_baseline_noise_sum, external_matched_samples,
                    out=np.full(columns.size, np.nan), where=external_matched_samples > 0,
                )
                external_baseline_target_power_change_db = 10.0 * np.log10(
                    np.maximum(external_task_target_power, eps) / np.maximum(external_baseline_target_power, eps)
                )
                external_task_snr_db = 10.0 * np.log10(
                    np.maximum(external_task_target_power, eps) / np.maximum(external_task_noise_power, eps)
                )
                external_baseline_snr_db = 10.0 * np.log10(
                    np.maximum(external_baseline_target_power, eps) / np.maximum(external_baseline_noise_power, eps)
                )
                external_baseline_snr_change_db = external_task_snr_db - external_baseline_snr_db
                external_status = []
                external_lengths = []
                external_min_segments = np.zeros(columns.size, dtype=np.int64)
                for channel in range(columns.size):
                    matched_lengths = [
                        length for length, counts in external_matched_samples_by_length.items() if counts[channel] > 0
                    ]
                    external_lengths.append(",".join(f"{length}s" for length in matched_lengths) or "")
                    valid_counts = [
                        int(external_baseline_by_length[length][2][channel]) for length in matched_lengths
                    ]
                    external_min_segments[channel] = min(valid_counts) if valid_counts else 0
                    if external_baseline_reference is None:
                        external_status.append("未启用实验前 baseline")
                    elif matched_lengths:
                        external_status.append("已按 " + "/".join(str(length) for length in matched_lengths) + " s 连续窗配对")
                    elif not any(np.isfinite(item[0][channel]) for item in external_baseline_by_length.values()):
                        external_status.append("无同通道或合格的实验前 baseline")
                    else:
                        external_status.append("无连续 3/4/5 s 任务片段可配对")
                external_time_response = None
                external_time_task_counts = np.zeros(columns.size, dtype=np.int64)
                external_time_rest_counts = np.zeros(columns.size, dtype=np.int64)
                external_time_cluster_counts = np.zeros(columns.size, dtype=np.int64)
                external_time_status = ["未准备时域静息参考"] * columns.size
                if external_time_reference is not None:
                    rest_traces = np.asarray(external_time_reference["traces"], dtype=float)
                    task_traces = np.asarray(external_time_task_trials, dtype=float)
                    channel_response = {}
                    for channel in range(columns.size):
                        baseline_channel = int(external_time_channel_indices[channel])
                        if baseline_channel < 0 or task_traces.ndim != 3:
                            external_time_status[channel] = "无同通道的静息时域参考"
                            continue
                        task_channel_traces = task_traces[:, :, channel]
                        rest_channel_traces = rest_traces[:, :, baseline_channel]
                        comparison = compute_external_rest_cluster_test(
                            task_channel_traces, rest_channel_traces,
                            permutations=1000, cluster_forming_p=.01,
                            cluster_significance_p=.01, random_seed=20260709 + int(tag) * 100_003 + channel,
                        )
                        valid_task = task_channel_traces[np.all(np.isfinite(task_channel_traces), axis=1)]
                        valid_rest = rest_channel_traces[np.all(np.isfinite(rest_channel_traces), axis=1)]
                        external_time_task_counts[channel] = valid_task.shape[0]
                        external_time_rest_counts[channel] = valid_rest.shape[0]
                        significant = [entry for entry in comparison["clusters"] if entry["significant"]]
                        external_time_cluster_counts[channel] = len(significant)
                        if comparison["status"] == "ok":
                            external_time_status[channel] = (
                                f"{len(significant)} 个显著簇（1000 次置换）"
                                if significant else "未发现显著簇（1000 次置换）"
                            )
                        else:
                            external_time_status[channel] = "有效任务或静息伪 epoch 少于 3"
                        channel_response[int(channel_ids[channel])] = {
                            **comparison,
                            "task_mean": np.nanmean(valid_task, axis=0).astype(np.float32) if valid_task.size else np.full(external_time_task_indices.size, np.nan, dtype=np.float32),
                            "task_sem": (np.nanstd(valid_task, axis=0, ddof=1) / np.sqrt(valid_task.shape[0])).astype(np.float32) if valid_task.shape[0] > 1 else np.full(external_time_task_indices.size, np.nan, dtype=np.float32),
                            "rest_mean": np.nanmean(valid_rest, axis=0).astype(np.float32) if valid_rest.size else np.full(external_time_task_indices.size, np.nan, dtype=np.float32),
                            "rest_sem": (np.nanstd(valid_rest, axis=0, ddof=1) / np.sqrt(valid_rest.shape[0])).astype(np.float32) if valid_rest.shape[0] > 1 else np.full(external_time_task_indices.size, np.nan, dtype=np.float32),
                        }
                    external_time_response = {
                        "time_ms": np.asarray(external_time_reference["time_ms"], dtype=np.float32),
                        "sample_rate_hz": float(external_time_reference["sample_rate_hz"]),
                        "channels": channel_response,
                        "cluster_method": "two-sided independent temporal cluster permutation",
                    }
                metrics = [{
                    "tag": int(tag), "trials": processed_count,
                    "usable_trials": int(usable_trials[channel]),
                    "rejected_trials": int(processed_count - usable_trials[channel]),
                    "clean_windows": int(clean_window_counts[channel]),
                    "available_windows": int(available_window_counts[channel]),
                    "short_epoch_trials": int(short_epoch_trials[channel]),
                    "response_only_trials": int(response_only_trials[channel]),
                    "unavailable_windows": int(quality_reasons["unavailable"][channel]),
                    "saturated_windows": int(quality_reasons["saturated"][channel]),
                    "flat_windows": int(quality_reasons["flat"][channel]),
                    "jump_windows": int(quality_reasons["jump"][channel]),
                    "spectral_samples": int(spectral_sample_counts[channel]),
                    "spectral_seconds": float(spectral_sample_counts[channel] / meta.fs),
                    "spectral_segments": int(spectral_segment_counts[channel]),
                    "baseline_trials": int(baseline_power_counts[channel]),
                    "baseline_duration_ms": baseline_duration_ms,
                    "event_lfp_trials": int(event_power_counts[channel]),
                    "event_baseline_power": float(event_baseline_power_mean[channel]),
                    "event_response_power": float(event_response_power_mean[channel]),
                    "event_lfp_snr_db": float(event_lfp_snr_db[channel]),
                    "target_freq_hz": target_frequency,
                    "target_power": float(target_power_mean[channel]),
                    "baseline_target_power": float(baseline_power_mean[channel]),
                    "target_power_change_db": float(target_power_change_db[channel]),
                    "local_tag_snr_db": float(local_tag_snr_db[channel]),
                    "external_baseline_target_power": float(external_baseline_target_power[channel]),
                    "external_baseline_target_power_change_db": float(external_baseline_target_power_change_db[channel]),
                    "external_baseline_local_snr_db": float(external_baseline_snr_db[channel]),
                    "external_baseline_local_snr_change_db": float(external_baseline_snr_change_db[channel]),
                    "external_baseline_matched_seconds": float(external_matched_samples[channel] / meta.fs),
                    "external_baseline_matched_segments": int(external_matched_segments[channel]),
                    "external_baseline_min_valid_segments": int(external_min_segments[channel]),
                    "external_baseline_lengths": external_lengths[channel],
                    "external_baseline_status": external_status[channel],
                    "external_time_task_trials": int(external_time_task_counts[channel]),
                    "external_time_rest_epochs": int(external_time_rest_counts[channel]),
                    "external_time_significant_clusters": int(external_time_cluster_counts[channel]),
                    "external_time_status": external_time_status[channel],
                    "channel": channel_ids[channel], "peak_amp_mv": float(peak_amp[channel]),
                    "peak_latency_ms": float(peak_time[channel]), "aggregate": self.settings["aggregate"],
                } for channel in range(wave.shape[1])]
                item = {
                    "tag": int(tag), "trials": processed_count,
                    "trials_with_usable_channels": accepted_trial_count,
                    "usable_trials_per_channel": usable_trials,
                    "target_freq_hz": target_frequency,
                    "spectral_samples_per_channel": spectral_sample_counts,
                    "spectral_segments_per_channel": spectral_segment_counts,
                    "target_power": target_power_mean,
                    "baseline_target_power": baseline_power_mean,
                    "target_power_change_db": target_power_change_db,
                    "local_tag_snr_db": local_tag_snr_db,
                    "event_lfp_snr_db": event_lfp_snr_db,
                    "external_baseline_target_power": external_baseline_target_power,
                    "external_baseline_target_power_change_db": external_baseline_target_power_change_db,
                    "external_baseline_local_snr_db": external_baseline_snr_db,
                    "external_baseline_local_snr_change_db": external_baseline_snr_change_db,
                    "external_baseline_matched_samples": external_matched_samples,
                    "external_baseline_matched_segments": external_matched_segments,
                    "external_baseline_lengths": external_lengths,
                    "external_time_response": external_time_response,
                    "t_ms": t_ms, "wave": wave, "peak_amp": peak_amp,
                    "peak_latency_ms": peak_time, "metrics": metrics,
                    "used_marker_samples": used_samples,
                }
                if self.settings.get("mode") == "Letter":
                    trial_a = configured_a or processed_count
                    trial_b = configured_b or processed_count
                    # A/B is a larger-versus-smaller comparison.  Preserve
                    # that invariant if only one value was entered.
                    trial_b = max(trial_a, trial_b)
                    if processed_count < trial_a:
                        skipped.append(f"tag {int(tag)}：仅有 {processed_count} 个完整 trial，A/B 对比至少需要 {trial_a} 个。")
                        continue
                    valid_b = min(trial_b, processed_count)
                    aggregate_fn = np.nanmedian if self.settings["aggregate"] == "median" else np.nanmean
                    if streaming_mean:
                        means = np.asarray(trial_time_means, dtype=np.float64)
                        avg_a = np.nanmean(means[:trial_a], axis=0)
                        avg_b = np.nanmean(means[:valid_b], axis=0)
                    else:
                        avg_a = aggregate_fn(array[:trial_a], axis=(0, 2))
                        avg_b = aggregate_fn(array[:valid_b], axis=(0, 2))
                    finite = np.isfinite(avg_a) & np.isfinite(avg_b)
                    corr = float(np.corrcoef(avg_a[finite], avg_b[finite])[0, 1]) if finite.sum() > 2 else np.nan
                    peak_a, peak_b = float(np.nanmax(np.abs(avg_a))), float(np.nanmax(np.abs(avg_b)))
                    item.update(avg_a=avg_a, avg_b=avg_b, diff=avg_b - avg_a, trial_a=trial_a, valid_b=valid_b, corr=corr, peak_a=peak_a, peak_b=peak_b, change=(peak_b - peak_a) / max(abs(peak_a), np.finfo(np.float32).eps) * 100)
                result.append(item)
                self.progress.emit(35 + 65 * number / max(1, len(requested)), f"任务分析：tag {tag}，{processed_count} trials")
            if not result:
                detail = "\n".join(skipped) or "没有 marker 落在当前数据段中。"
                raise ValueError(f"没有可用于任务分析的有效 epoch。\n{detail}")
            # IntegratedPipelineGUI computes its long behavior-linked window
            # only when that browser is explicitly opened.  Do not silently
            # trigger another all-trial, all-channel calculation here.
            behavior_results = []
            matches = self.settings.get("behavior_matches") or {}
            if self.settings.get("calculate_behavior", False) and self.settings.get("mode") == "Letter" and matches:
                # This is the same long-window calculation as the old Tk
                # browser, once for every Letter tag.  Keep only the aggregate
                # and percentile bands here; individual trials are loaded on
                # demand after the user clicks a channel subplot.
                aggregate_fn = np.nanmedian if self.settings["aggregate"] == "median" else np.nanmean
                # Tk's show_long_behavior_linked_window deliberately exposes
                # every available Letter tag, independently of the task
                # result's stimtag/nStimTypes selection.
                behavior_tags = sorted({int(row[1]) for row in self.markers if int(row[1]) in {4, 5, 6, 7}})
                for behavior_index, tag in enumerate(behavior_tags, start=1):
                    tag = int(tag)
                    tag_markers = self.markers[self.markers[:, 1] == tag]
                    valid_latencies = [
                        float(event["animal_latency_sec"])
                        for row in tag_markers
                        if (event := matches.get(int(row[0]))) is not None
                        and event.get("validity") == "Y"
                        and np.isfinite(float(event.get("animal_latency_sec", np.nan)))
                    ]
                    end_sec = max(6.0, max(valid_latencies, default=0.0) + .5)
                    long_start, long_end = int(round(-.5 * meta.fs)), int(round(end_sec * meta.fs))
                    long_epochs, long_samples = [], []
                    for marker in tag_markers[:self.settings["trials_per_stim"] or None]:
                        center = int(marker[0]) - marker_offset
                        first, last = center + long_start, center + long_end
                        if first >= 0 and last <= meta.rows:
                            long_epochs.append(read_epoch(first, last)); long_samples.append(int(marker[0]))
                    if not long_epochs:
                        continue
                    long_array = np.stack(long_epochs)
                    long_t = np.arange(long_start, long_end, dtype=float) / meta.fs * 1000.0
                    long_mask = (long_t >= self.settings["baseline_start"]) & (long_t <= self.settings["baseline_end"])
                    if np.any(long_mask):
                        long_array -= np.nanmean(long_array[:, long_mask], axis=1, keepdims=True)
                    behavior_results.append({
                        "tag": tag, "t_ms": long_t,
                        "wave": aggregate_fn(long_array, axis=0),
                        "q10": np.nanpercentile(long_array, 10, axis=0),
                        "q90": np.nanpercentile(long_array, 90, axis=0),
                        "used_marker_samples": long_samples, "trials": len(long_epochs),
                        "end_sec": end_sec, "animal_count": len(valid_latencies),
                    })
                    self.progress.emit(88 + 12 * behavior_index / max(1, len(behavior_tags)), f"Letter 长行为窗：tag {tag}")
            self.completed.emit({"results": result, "channel_ids": channel_ids, "columns": columns, "fs": float(meta.fs), "aggregate": self.settings["aggregate"], "view_mode": self.settings.get("view_mode", "raw"), "mode": self.settings.get("mode", "Flash"), "lfp_task_metrics": lfp_task_metrics, "quality_only": quality_only, "quality_signature": self.settings.get("quality_signature"), "trial_quality": trial_quality, "source_signature": (int(meta.rows), float(meta.fs), float(meta.time_offset), tuple(int(value) for value in meta.channel_ids)), "source_identity": (id(self.source), id(getattr(self.source, "data", None))), "epoch_start_ms": float(self.settings["epoch_start"]), "epoch_end_ms": float(self.settings["epoch_end"]), "baseline_start_ms": float(self.settings["baseline_start"]), "baseline_end_ms": float(self.settings["baseline_end"]), "task_processing": {name: self.settings.get(name) for name in ("analysis_notch", "analysis_bandpass", "analysis_band_low", "analysis_band_high", "smooth", "smooth_window_sec", "zscore")}, "behavior_result": behavior_results[0] if behavior_results else None, "behavior_results": behavior_results})
        except Exception as exc:
            self.failed.emit(str(exc))


class TrialVepWorker(QThread):
    """Extract and baseline-correct every task epoch without five-window QC."""
    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, task_result, channel_id, output_metadata):
        super().__init__()
        self.source = source
        self.task_result = task_result
        self.channel_id = int(channel_id)
        self.output_metadata = dict(output_metadata)

    def run(self) -> None:
        try:
            meta = self.source.metadata
            channel_lookup = {int(channel): index for index, channel in enumerate(meta.channel_ids)}
            if self.channel_id not in channel_lookup:
                raise ValueError("所选通道不在当前 LFP 数据源中。")
            processing = dict(self.output_metadata.get("task_processing") or {})
            values = np.asarray(self.source.read(0, meta.rows, [channel_lookup[self.channel_id]]), dtype=np.float32)
            if values.ndim == 2:
                values = values[:, 0]
            if processing.get("analysis_notch") or processing.get("analysis_bandpass"):
                filtered, _ = filter_array(
                    values[:, None], meta.fs,
                    "bandpass" if processing.get("analysis_bandpass") else "off",
                    float(processing.get("analysis_band_low", .5)),
                    float(processing.get("analysis_band_high", 300.0)),
                    notch=bool(processing.get("analysis_notch")),
                )
                values = filtered[:, 0]
            if processing.get("smooth"):
                width = max(1, int(round(float(processing.get("smooth_window_sec", .02)) * meta.fs)))
                if width > 1:
                    values = np.convolve(values, np.ones(width) / width, mode="same")
            if processing.get("zscore"):
                deviation = float(np.nanstd(values))
                values = (values - float(np.nanmean(values))) / (deviation if deviation > 0 else 1.0)

            epoch_start = float(self.output_metadata["epoch_start_ms"])
            epoch_end = float(self.output_metadata["epoch_end_ms"])
            baseline_start = float(self.output_metadata["baseline_start_ms"])
            baseline_end = float(self.output_metadata["baseline_end_ms"])
            first_offset = int(round(epoch_start * meta.fs / 1000.0))
            epoch_samples = int(round((epoch_end - epoch_start) * meta.fs / 1000.0))
            t_ms = np.arange(epoch_samples, dtype=float) / meta.fs * 1000.0 + epoch_start
            baseline_mask = (t_ms >= baseline_start) & (t_ms <= baseline_end)
            if epoch_samples < 3 or not np.any(baseline_mask):
                raise ValueError("VEP epoch 或刺激前基线窗无效。")
            marker_offset = int(round(float(meta.time_offset) * meta.fs))
            trials = []
            used_samples = []
            markers = list(self.task_result.get("used_marker_samples") or [])
            for index, marker_sample in enumerate(markers, start=1):
                center = int(marker_sample) - marker_offset
                first = center + first_offset
                last = first + epoch_samples
                if first < 0 or last > values.size:
                    continue
                epoch = values[first:last].astype(np.float32, copy=True)
                with np.errstate(invalid="ignore"):
                    correction = np.nanmean(epoch[baseline_mask])
                if not np.isfinite(correction):
                    continue
                trials.append(epoch - correction)
                used_samples.append(int(marker_sample))
                if index % 8 == 0:
                    self.progress.emit(10 + 80 * index / max(1, len(markers)), f"VEP：已读取 {index}/{len(markers)} 个 trial")
            if not trials:
                raise ValueError("当前 tag 没有落在数据范围内且具有有效刺激前基线的完整 trial。")
            trial_array = np.asarray(trials, dtype=np.float32)
            valid_count = np.sum(np.isfinite(trial_array), axis=0)
            with np.errstate(all="ignore"):
                mean = np.nanmean(trial_array, axis=0)
                std = np.nanstd(trial_array, axis=0, ddof=1)
            sem = np.divide(std, np.sqrt(valid_count), out=np.full(epoch_samples, np.nan), where=valid_count > 1)
            self.completed.emit({
                "tag": int(self.task_result["tag"]), "channel": self.channel_id,
                "time_ms": t_ms.astype(np.float32), "trials": trial_array,
                "mean": mean.astype(np.float32), "sem": sem.astype(np.float32),
                "valid_count": valid_count.astype(np.int32), "used_marker_samples": used_samples,
                "baseline_start_ms": baseline_start, "baseline_end_ms": baseline_end,
                "qc_applied": False,
            })
        except Exception as exc:
            self.failed.emit(str(exc))


class ExternalTimeFrequencyWorker(QThread):
    """Compute one QC-matched task/rest power time-frequency comparison."""
    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, task_source, baseline_source, reference, task_result, channel_id, settings):
        super().__init__()
        self.task_source = task_source
        self.baseline_source = baseline_source
        self.reference = reference
        self.task_result = task_result
        self.channel_id = int(channel_id)
        self.settings = dict(settings)

    def run(self) -> None:
        try:
            from scipy import signal
            from mne.time_frequency import tfr_array_morlet

            task_meta = self.task_source.metadata
            baseline_meta = self.baseline_source.metadata
            if not np.isclose(task_meta.fs, baseline_meta.fs, rtol=0, atol=1e-9):
                raise ValueError("任务与静息数据采样率不一致，不能进行严格时频对照。")
            if self.settings.get("zscore"):
                raise ValueError("独立静息 baseline 时频比较不支持逐记录 z-score。")
            task_lookup = {int(channel): index for index, channel in enumerate(task_meta.channel_ids)}
            baseline_lookup = {int(channel): index for index, channel in enumerate(baseline_meta.channel_ids)}
            if self.channel_id not in task_lookup or self.channel_id not in baseline_lookup:
                raise ValueError("所选通道不同时存在于任务和实验前静息 H5。")
            fs = float(task_meta.fs)

            def conditioned(source, column: int) -> np.ndarray:
                values = np.asarray(source.read(0, source.metadata.rows, [column]), dtype=np.float32)
                if values.ndim == 1:
                    values = values[:, None]
                if self.settings.get("analysis_notch") or self.settings.get("analysis_bandpass"):
                    values, _ = filter_array(
                        values, fs, "bandpass" if self.settings.get("analysis_bandpass") else "off",
                        float(self.settings.get("analysis_band_low", .5)),
                        float(self.settings.get("analysis_band_high", 300.0)),
                        notch=bool(self.settings.get("analysis_notch")),
                    )
                if self.settings.get("smooth"):
                    width = max(1, int(round(float(self.settings.get("smooth_window_sec", .02)) * fs)))
                    if width > 1:
                        values = np.convolve(values[:, 0], np.ones(width) / width, mode="same")[:, None]
                return values[:, 0]

            epoch_start = float(self.settings["epoch_start"])
            epoch_end = float(self.settings["epoch_end"])
            baseline_start = float(self.settings["baseline_start"])
            baseline_end = float(self.settings["baseline_end"])
            response_start = float(self.settings["response_start"])
            response_end = float(self.settings["response_end"])
            epoch_samples = int(round((epoch_end - epoch_start) * fs / 1000.0))
            if epoch_samples < 3:
                raise ValueError("时频 epoch 长度无效。")
            t_ms = np.arange(epoch_samples, dtype=float) / fs * 1000.0 + epoch_start
            baseline_mask = (t_ms >= baseline_start) & (t_ms <= baseline_end)
            response_mask = (t_ms >= response_start) & (t_ms <= response_end)
            if not np.any(baseline_mask) or not np.any(response_mask):
                raise ValueError("时频基线窗或对比窗没有采样点。")
            quality_settings = dict(self.settings["quality_settings"])
            quality_kwargs = {
                "flat_epsilon": float(quality_settings["quality_flat_epsilon"]),
                "flat_ratio_percent": float(quality_settings["quality_flat_ratio_percent"]),
                "flat_ptp": float(quality_settings["quality_flat_ptp"]),
                "available_ratio": float(quality_settings["quality_available_ratio"]),
                "saturation_run_samples": int(quality_settings["quality_saturation_run_samples"]),
                "jump_mad_multiplier": float(quality_settings["quality_jump_mad_multiplier"]),
                "jump_median_multiplier": float(quality_settings["quality_jump_median_multiplier"]),
                "jump_flat_floor_multiplier": float(quality_settings["quality_jump_flat_floor_multiplier"]),
            }

            def valid_epoch(epoch: np.ndarray) -> bool:
                clean, quality = evaluate_task_trial_window_quality(epoch[:, None], fs, epoch_start, **quality_kwargs)
                available = int(np.sum(quality["available"][:, 0]))
                usable = available > 0 and int(np.sum(clean[:, 0])) >= min(3, available)
                finite = (
                    np.mean(np.isfinite(epoch[baseline_mask])) >= quality_kwargs["available_ratio"]
                    and np.mean(np.isfinite(epoch[response_mask])) >= quality_kwargs["available_ratio"]
                )
                return bool(usable and finite)

            self.progress.emit(3, "时频图：读取并应用任务分析预处理…")
            task_values = conditioned(self.task_source, task_lookup[self.channel_id])
            baseline_values = conditioned(self.baseline_source, baseline_lookup[self.channel_id])
            task_offset = int(round(float(task_meta.time_offset) * fs))
            task_epochs = []
            samples = list(self.task_result.get("used_marker_samples") or [])
            for index, sample in enumerate(samples, start=1):
                center = int(sample) - task_offset
                first = center + int(round(epoch_start * fs / 1000.0))
                last = first + epoch_samples
                if first < 0 or last > task_values.size:
                    continue
                epoch = task_values[first:last]
                if valid_epoch(epoch):
                    task_epochs.append(epoch)
                if index % 4 == 0:
                    self.progress.emit(5 + 35 * index / max(1, len(samples)), f"时频图：任务 trial {index}/{len(samples)}")
            rest_epochs = []
            starts = np.asarray((self.reference.get("time_reference") or {}).get("candidate_starts", []), dtype=np.int64)
            for index, first in enumerate(starts, start=1):
                epoch = baseline_values[int(first):int(first) + epoch_samples]
                if epoch.size == epoch_samples and valid_epoch(epoch):
                    rest_epochs.append(epoch)
                self.progress.emit(40 + 25 * index / max(1, starts.size), f"时频图：静息伪 epoch {index}/{starts.size}")
            if len(task_epochs) < 3 or len(rest_epochs) < 3:
                raise ValueError(f"有效任务 trial / 静息伪 epoch 为 {len(task_epochs)} / {len(rest_epochs)}，均至少需要 3。")

            requested_rate = min(fs, max(50.0, float(self.settings["sample_rate_hz"])))
            ratio = Fraction(requested_rate / fs).limit_denominator(1000)
            output_fs = fs * ratio.numerator / ratio.denominator

            def resample(epochs: list[np.ndarray]) -> np.ndarray:
                array = np.asarray(epochs, dtype=np.float32)
                if ratio.numerator != ratio.denominator:
                    array = signal.resample_poly(array, ratio.numerator, ratio.denominator, axis=1).astype(np.float32)
                return array

            task_array, rest_array = resample(task_epochs), resample(rest_epochs)
            output_t_ms = np.arange(task_array.shape[1], dtype=float) / output_fs * 1000.0 + epoch_start
            output_baseline = (output_t_ms >= baseline_start) & (output_t_ms <= baseline_end)
            output_response = (output_t_ms >= response_start) & (output_t_ms <= response_end)
            frequencies = np.arange(
                float(self.settings["freq_low"]), float(self.settings["freq_high"]) + float(self.settings["freq_step"]) * .25,
                float(self.settings["freq_step"]),
            )
            cycles = float(self.settings["cycles"])
            if not frequencies.size or frequencies[-1] >= output_fs / 2:
                raise ValueError("时频最高频率必须低于重采样后的奈奎斯特频率。")
            if np.count_nonzero(output_baseline) / output_fs < cycles / frequencies[0]:
                raise ValueError("刺激前基线窗过短，无法支持当前最低频率和小波周期数。")

            def normalized_power(array: np.ndarray, label: str) -> np.ndarray:
                self.progress.emit(68 if label == "任务" else 80, f"时频图：计算{label}小波功率…")
                power = tfr_array_morlet(
                    array[:, None, :], sfreq=output_fs, freqs=frequencies, n_cycles=cycles,
                    output="power", zero_mean=True, n_jobs=1, verbose=False,
                )[:, 0]
                reference_power = np.nanmean(power[:, :, output_baseline], axis=2, keepdims=True)
                return (10.0 * np.log10(np.maximum(power, np.finfo(float).tiny) / np.maximum(reference_power, np.finfo(float).tiny)))[:, :, output_response]

            task_maps = normalized_power(task_array, "任务")
            rest_maps = normalized_power(rest_array, "静息")
            comparison = compute_external_rest_tf_cluster_test(task_maps, rest_maps)
            response_time_ms = output_t_ms[output_response]
            self.completed.emit({
                "tag": int(self.task_result["tag"]), "channel": self.channel_id,
                "frequencies_hz": frequencies.astype(np.float32), "time_ms": response_time_ms.astype(np.float32),
                "task_mean_db": np.nanmean(task_maps, axis=0).astype(np.float32),
                "rest_mean_db": np.nanmean(rest_maps, axis=0).astype(np.float32),
                "task_minus_rest_db": (np.nanmean(task_maps, axis=0) - np.nanmean(rest_maps, axis=0)).astype(np.float32),
                "comparison": comparison, "sample_rate_hz": output_fs, "cycles": cycles,
            })
        except Exception as exc:
            self.failed.emit(str(exc))


class ItpcWorker(QThread):
    """Stream marker-aligned Morlet phases using the saved page-4 QC mask."""
    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, quality_output: dict, settings: dict):
        super().__init__()
        self.source = source
        self.quality_output = quality_output
        self.settings = settings

    def run(self) -> None:
        try:
            from scipy import signal

            meta = self.source.metadata
            columns = np.asarray(self.quality_output["columns"], dtype=np.int64)
            channel_ids = np.asarray(self.quality_output["channel_ids"], dtype=int)
            cached = self.quality_output["trial_quality"]
            epoch_start = float(self.quality_output["epoch_start_ms"])
            epoch_end = float(self.quality_output["epoch_end_ms"])
            frequencies = np.arange(self.settings["freq_low"], self.settings["freq_high"] + self.settings["freq_step"] * .25, self.settings["freq_step"])
            if not frequencies.size or frequencies[-1] >= meta.fs / 2:
                raise ValueError("ITPC 频率必须位于 0 和奈奎斯特频率之间。")
            if self.settings["time_end"] <= self.settings["time_start"]:
                raise ValueError("ITPC 时间终点必须大于起点。")
            ratio = Fraction(min(float(self.settings["sample_rate"]), float(meta.fs)) / float(meta.fs)).limit_denominator(1000)
            work_fs = float(meta.fs) * ratio.numerator / ratio.denominator
            first = int(round(epoch_start * meta.fs / 1000.0))
            last = int(round(epoch_end * meta.fs / 1000.0))
            output_samples = int(np.ceil((last - first) * ratio.numerator / ratio.denominator))
            t_ms = np.arange(output_samples, dtype=float) / work_fs * 1000.0 + epoch_start
            wanted_time = (t_ms >= self.settings["time_start"]) & (t_ms <= self.settings["time_end"])
            if not np.any(wanted_time):
                raise ValueError("ITPC 时间范围不在当前 epoch 内。")
            t_ms = t_ms[wanted_time]
            marker_offset = int(round(float(meta.time_offset) * meta.fs))
            processing = dict(self.quality_output.get("task_processing") or {})
            data = None
            if any((processing.get("analysis_notch"), processing.get("analysis_bandpass"), processing.get("smooth"), processing.get("zscore"))):
                raw, _ = self.source.materialize() if not hasattr(self.source, "data") else (self.source.data, None)
                data = np.asarray(raw[:, columns], dtype=np.float32)
                if processing.get("analysis_notch") or processing.get("analysis_bandpass"):
                    data, _ = filter_array(data, meta.fs, "bandpass" if processing.get("analysis_bandpass") else "off", float(processing.get("analysis_band_low", .5)), float(processing.get("analysis_band_high", 300.0)), notch=bool(processing.get("analysis_notch")))
                if processing.get("smooth"):
                    width = max(1, int(round(float(processing.get("smooth_window_sec", .02)) * meta.fs)))
                    if width > 1:
                        kernel = np.ones(width, dtype=np.float32) / width
                        data = np.apply_along_axis(lambda values: np.convolve(values, kernel, mode="same"), 0, data)
                if processing.get("zscore"):
                    deviation = np.nanstd(data, axis=0, keepdims=True); deviation[deviation == 0] = 1.0
                    data = (data - np.nanmean(data, axis=0, keepdims=True)) / deviation
            by_tag: dict[int, list[int]] = {}
            for tag, sample in cached:
                by_tag.setdefault(int(tag), []).append(int(sample))
            requested_tags = set(self.settings.get("tags") or by_tag)
            tags = [tag for tag in sorted(by_tag) if tag in requested_tags]
            if not tags:
                raise ValueError("所选 ITPC tag 不在已缓存的五窗质量结果中。")
            max_cells = len(tags) * frequencies.size * t_ms.size * max(1, columns.size)
            if max_cells > 30_000_000:
                raise ValueError("ITPC 输出过大；请缩小 tag、通道、频率或时间范围。")
            results = []
            total = sum(len(by_tag[tag]) for tag in tags) * frequencies.size
            done = 0
            for tag in tags:
                phase_sum = np.zeros((frequencies.size, t_ms.size, columns.size), dtype=np.complex64)
                counts = np.zeros((frequencies.size, t_ms.size, columns.size), dtype=np.int32)
                for sample in by_tag[tag]:
                    center = sample - marker_offset
                    epoch = np.asarray(data[center + first:center + last] if data is not None else self.source.read(center + first, center + last, columns), dtype=np.float32)
                    if epoch.ndim == 1:
                        epoch = epoch[:, None]
                    if ratio.numerator != ratio.denominator:
                        epoch = signal.resample_poly(epoch, ratio.numerator, ratio.denominator, axis=0).astype(np.float32, copy=False)
                    if epoch.shape[0] != output_samples:
                        epoch = epoch[:output_samples]
                    clean = np.asarray(cached[(tag, sample)]["clean_windows"], dtype=bool)
                    for freq_index, frequency in enumerate(frequencies):
                        support_ms = self.settings["cycles"] / (2.0 * frequency) * 1000.0
                        half_width = max(2, int(np.ceil(support_ms / 1000.0 * work_fs)))
                        wave_t = np.arange(-half_width, half_width + 1, dtype=float) / work_fs
                        wavelet = np.exp(2j * np.pi * frequency * wave_t) * np.exp(-wave_t ** 2 / (2.0 * (self.settings["cycles"] / (2.0 * np.pi * frequency)) ** 2))
                        coefficients = signal.fftconvolve(epoch, wavelet[:, None], mode="same", axes=0)
                        for window in range(5):
                            left_ms, right_ms = 500.0 + 1000.0 * window, 1500.0 + 1000.0 * window
                            valid_time = (t_ms >= left_ms + support_ms) & (t_ms < right_ms - support_ms)
                            valid_channels = np.flatnonzero(clean[window])
                            if valid_channels.size and np.any(valid_time):
                                original_positions = np.flatnonzero(wanted_time)[valid_time]
                                values = coefficients[original_positions[:, None], valid_channels[None, :]]
                                target = np.ix_(np.flatnonzero(valid_time), valid_channels)
                                phase_sum[freq_index][target] += values / np.maximum(np.abs(values), np.finfo(np.float32).tiny)
                                counts[freq_index][target] += 1
                        done += 1
                        self.progress.emit(100.0 * done / max(1, total), f"ITPC：tag {tag}，trial {sample}，{frequency:g} Hz")
                itpc = np.abs(phase_sum) / np.maximum(counts, 1)
                itpc[counts < int(self.settings["min_trials"])] = np.nan
                baseline_mask = (t_ms >= self.settings["baseline_start"]) & (t_ms <= self.settings["baseline_end"])
                baseline = np.nanmean(itpc[:, baseline_mask, :], axis=1, keepdims=True) if np.any(baseline_mask) else np.full((frequencies.size, 1, columns.size), np.nan)
                results.append({"tag": tag, "itpc": itpc.astype(np.float32), "itpc_baseline_delta": (itpc - baseline).astype(np.float32), "counts": counts, "trials": len(by_tag[tag])})
            self.completed.emit({"results": results, "frequencies_hz": frequencies, "t_ms": t_ms, "channel_ids": channel_ids, "sample_rate": work_fs, "cycles": self.settings["cycles"], "min_trials": self.settings["min_trials"], "baseline_start": self.settings["baseline_start"], "baseline_end": self.settings["baseline_end"]})
        except Exception as exc:
            self.failed.emit(str(exc))


class ParameterBatchWorker(QThread):
    """Run the legacy parameter-CSV matrix against one loaded data source."""

    progress = pyqtSignal(float, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, rows: list[dict], run_spike: bool = False, columns=None):
        super().__init__()
        self.source = source
        self.rows = rows
        self.run_spike = run_spike
        self.columns = None if columns is None else np.asarray(columns, dtype=np.int64).ravel()

    def run(self) -> None:
        try:
            from scipy import signal
            data, _ = self.source.materialize(lambda value, message: self.progress.emit(value * 25.0, message)) if not hasattr(self.source, "data") else (self.source.data, None)
            meta = self.source.metadata
            columns = np.arange(meta.channels, dtype=np.int64) if self.columns is None else self.columns
            if not columns.size or np.any(columns < 0) or np.any(columns >= meta.channels):
                raise ValueError("没有可用于参数 CSV 批处理的有效 LFP 通道。")
            data = np.asarray(data[:, columns], dtype=np.float32)
            outputs = []
            for index, params in enumerate(self.rows, start=1):
                # LFP batch scans vary analysis bands/thresholds only.  Signal
                # conditioning belongs to the preprocessing page, so a CSV
                # row must never silently apply a second filter here.
                snr_rows = compute_batch_channel_snr(data, meta.fs, mode=str(params.get("mode", "resting")), signal_band=(float(params.get("signal_low", 1) or 1), float(params.get("signal_high", 30) or 30)), noise_band=(float(params.get("noise_low", 1) or 1), float(params.get("noise_high", 200) or 200)))
                for channel_index, row in enumerate(snr_rows):
                    row["channel"] = int(meta.channel_ids[columns[channel_index]]); row["parameter_name"] = params.get("name", f"row_{index}")
                record = {"name": params.get("name", f"row_{index}"), "lfp": snr_rows}
                if self.run_spike and str(params.get("run_spike", "1")).lower() not in {"0", "false", "no"}:
                    hp_low = float(params.get("spike_low", 300) or 300); hp_high = float(params.get("spike_high", 3000) or 3000)
                    sos = signal.butter(3, (hp_low, hp_high), btype="bandpass", fs=meta.fs, output="sos")
                    refractory = max(1, int(round(float(params.get("spike_refractory", 2) or 2) * meta.fs / 1000.0)))
                    spikes = []
                    for channel_index in range(data.shape[1]):
                        raw = np.nan_to_num(np.asarray(data[:, channel_index], dtype=np.float32), nan=0.0)
                        try: filtered = signal.sosfiltfilt(sos, raw)
                        except ValueError: filtered = raw
                        sigma = float(np.median(np.abs(filtered)) / 0.6745); threshold = -abs(float(params.get("spike_threshold", 4.5) or 4.5) * sigma)
                        peaks, _ = signal.find_peaks(-filtered, height=-threshold, distance=refractory)
                        spikes.append({"channel": int(meta.channel_ids[columns[channel_index]]), "spike_count": int(peaks.size), "rate_hz": float(peaks.size / max(1e-12, data.shape[0] / meta.fs)), "threshold_mv": threshold, "parameter_name": params.get("name", f"row_{index}")})
                    record["spike"] = spikes
                outputs.append(record)
                self.progress.emit(25.0 + 75.0 * index / max(1, len(self.rows)), f"批量参数 {index}/{len(self.rows)} 完成")
            self.completed.emit(outputs)
        except Exception as exc:
            self.failed.emit(str(exc))


class QtAnalysisGUI(QMainWindow):
    """Stage-one Qt application shell with a working lazy HDF5 preview page."""

    def __init__(self, operator_name: str | None = None) -> None:
        super().__init__()
        QApplication.instance().setFont(QFont("Microsoft YaHei", 11))
        self.operator_name = str(operator_name or os.environ.get("SD_GUI_OPERATOR", "未登录用户")).strip()
        self.operation_session_id = uuid.uuid4().hex
        self.feishu_audit_config = FeishuAuditConfig.from_environment()
        self.feishu_audit_sync = None
        audit_path = Path(os.environ.get(
            "SD_GUI_AUDIT_DB", str(Path(__file__).resolve().with_name("user_operation_audit.sqlite3"))
        ))
        self.operation_audit_store = None
        self.operation_audit_error = ""
        try:
            self.operation_audit_store = OperationAuditStore(
                audit_path, cloud_sync_enabled=self.feishu_audit_config.is_configured
            )
            if self.feishu_audit_config.is_configured:
                self.feishu_audit_sync = FeishuAuditSyncService(
                    audit_path, self.feishu_audit_config
                )
                self.feishu_audit_sync.start()
        except (OSError, sqlite3.Error) as exc:
            self.operation_audit_error = str(exc)
        self.source = LazyH5Source()
        self.active_source = self.source
        self.remapped_source = None
        self.preprocessed_source = None
        self.bin_timing_metadata: dict[str, object] = {}
        self.lfp_source = None
        self.external_baseline_source = None
        self.external_baseline_reference = None
        self._external_baseline_worker = None
        self._preprocess_filtered_channel_ids: set[int] = set()
        self._preprocess_notched_channel_ids: set[int] = set()
        self.analysis_cache = AnalysisCache()
        self._bad_channel_sweep_data_cache: dict[tuple, dict] = {}
        self._bad_channel_sweep_selected_pairs: tuple[tuple[Path, Path], ...] | None = None
        self._bad_channel_pair_results: dict[str, dict] = {}
        self._bad_channel_sweep_cohort_signature = None
        self.spike_source = None
        self.lfp_selected_ids = set()
        self._lfp_selection_explicit = False
        self.spike_selected_ids = set()
        self.lfp_rows = []
        self.spike_rows = []
        self.manual_channel_overrides = {}
        self.bad_channel_auto_reasons = {}
        self.bad_channel_fast_artifact_rows = []
        self.bad_channel_high_frequency_noise_rows = []
        self.bad_channel_ids = set()
        self.good_channel_ids = set()
        self.bad_channel_candidate_ids = set()
        self.bad_channel_candidate_reasons = {}
        self._bad_channel_result_channel_ids = ()
        self._bad_channel_check_completed = False
        self._bad_channel_fast_only = False
        self._bad_channel_review_brief_mode = False
        self._car_worker = None
        self._preprocess_lowpass_channel_ids = set()
        self._preprocess_lowpass_high_hz = None
        self._task_marker_source = None
        self.channel_layout_ids = None
        self._preprocess_worker = None
        self._last_filter_cache_signature = None
        self._last_filter_cache_value = None
        self.last_bad_channel_stage_timings = []
        self.last_one_click_stage_timings = []
        self._pending_filter_cache_signature = None
        self._remapped_detection_signature = None
        self._pending_remapped_detection_signature = None
        self._pending_detection_remapped_source = None
        self._pending_preserve_channel_review = False
        self._selected_analysis_batch = None
        self._analysis_batch_waiting_for_worker = False
        self._analysis_batch_waiting_for_psd = False
        self._create_legacy_variable_contract()
        # Task steady-state spectral analysis is independent of the older
        # whole-record SSVEP setting (whose legacy default is 10 Hz).
        self.task_target_freq_var = QLineEdit("1")
        self.task_target_freq_var.setObjectName("legacy_task_target_freq_var")
        self.setWindowTitle(f"SD 脑电分析 - 当前用户：{self.operator_name}")
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        if available is not None:
            self.resize(min(1500, available.width()), min(960, available.height()))
        else:
            self.resize(1200, 760)
        self.setMinimumSize(760, 520)
        self._build_ui_legacy_layout()
        audit_menu = self.menuBar().addMenu(f"操作记录（{self.operator_name}）")
        view_audit = audit_menu.addAction("查看多人操作记录")
        view_audit.triggered.connect(self.show_operation_audit_table)
        feishu_status = audit_menu.addAction("飞书同步状态与配置说明")
        feishu_status.triggered.connect(self.show_feishu_audit_status)
        self._audit_event("session_start", "程序启动")

    def _create_legacy_variable_contract(self) -> None:
        """Create every audited Tk controller variable before building pages.

        The audit file is a development/QA artifact rather than a runtime
        requirement.  When present it lets the Qt version retain all old
        variable names and defaults, including options that begin collapsed.
        Existing visible Qt controls replace these placeholders as their page
        is built.
        """
        inventory_path = Path(__file__).with_name("legacy_ui_inventory.json")
        if not inventory_path.is_file():
            return
        try:
            variables = json.loads(inventory_path.read_text(encoding="utf-8"))["main_window"]["variables"]
        except Exception:
            return
        select_values = {
            "snr_mode_var": ("resting", "task"),
            # This was a readonly Tk combobox.  Keeping it as a combo box is
            # important: values such as ``bandpass`` are modes, not numeric
            # cutoffs that should be accepted as arbitrary free text.
            "snr_highpass_var": ("off", "0.5", "1.0", "bandpass"),
            "psd_channel_source_var": ("selected channels", "healthy LFP channels", "all channels"),
            "psd_scale_var": ("dB", "linear"),
            "analysis_view_mode_var": ("raw", "overlay"),
            "task_response_aggregate_var": ("Mean", "Median"),
        }
        # Retired preprocessing controls remain in the historical Tk audit
        # file only; do not recreate invisible compatibility placeholders.
        retired_variables = {
            "bad_flat_std_var", "bad_flat_ratio_var",
            "bad_2s_window_var", "bad_valid_window_var", "bad_ptp_var",
            "bad_linear_drift_check_var", "bad_linear_drift_r2_var",
            "bad_linear_drift_order_var", "bad_linear_drift_net_shift_ratio_var",
            "bad_spectral_flatness_check_var", "bad_spectral_flatness_threshold_var",
            "bad_spectral_flatness_low_hz_var", "bad_spectral_flatness_high_hz_var",
            "bad_repeated_peak_check_var", "bad_repeated_peak_tolerance_var",
            "bad_repeated_peak_ratio_threshold_var", "bad_repeated_peak_window_seconds_var",
            "snr_view_action_var", "snr_export_action_var",
        }
        for name, details in variables.items():
            if name in retired_variables or hasattr(self, name):
                continue
            value = details.get("value", "")
            if name in select_values:
                widget = QComboBox()
                widget.addItems(select_values[name])
                widget.setCurrentText(str(value))
            elif details.get("type") == "BooleanVar":
                widget = QCheckBox()
                widget.setChecked(bool(value))
            else:
                widget = QLineEdit(str(value))
            widget.setObjectName(f"legacy_{name}")
            setattr(self, name, widget)

    @staticmethod
    def _widget_text(widget, default: str = "") -> str:
        """Read a legacy controller value from its real Qt widget.

        The old Tk GUI used ``.get()`` for every controller.  The migration
        deliberately uses the appropriate Qt control (combo box, check box or
        spin box) instead of turning choice parameters into free text.  This
        one adapter keeps calculation code independent of that widget choice.
        """
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QLineEdit):
            return widget.text()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return str(widget.value())
        if isinstance(widget, QCheckBox):
            return "1" if widget.isChecked() else "0"
        return default

    def _legacy_parameter_panel(self, title: str, names: list[str]) -> QGroupBox:
        """Expose formerly hidden Tk parameters without duplicating widgets."""
        compact_labels = {
            "snr_mode_var": "SNR 模式", "snr_skip_var": "跳过 SNR",
            "psd_channel_source_var": "PSD 通道", "psd_page_size_var": "PSD 每页数",
            "psd_welch_sec_var": "Welch 窗(s)", "psd_overlap_var": "重叠(%)",
            "psd_freq_low_var": "PSD 下限", "psd_freq_high_var": "PSD 上限",
            "psd_scale_var": "PSD 标度", "psd_mark_stim_var": "标刺激频率",
            "psd_stim_freq_var": "刺激频率", "psd_stim_harmonics_var": "刺激谐波数",
            "psd_mask_stim_var": "屏蔽刺激频率", "psd_bandwidth_var": "频带宽度",
            "signal_band_low_var": "信号下限", "signal_band_high_var": "信号上限",
            "noise_band_low_var": "噪声下限", "noise_band_high_var": "噪声上限",
            "stim_interval_var": "刺激间隔", "stim_duration_var": "刺激时长",
            "first_onset_var": "首次起点", "stim_freq_var": "刺激 Hz",
            "task_target_freq_var": "任务目标 Hz", "harmonics_var": "谐波数",
            "neighbor_bins_var": "邻频 bin", "fft_len_var": "FFT 长度",
            "state_start_var": "状态起点", "state_duration_var": "状态时长",
            "epoch_start_ms_var": "Epoch 起点", "epoch_end_ms_var": "Epoch 终点",
            "baseline_start_ms_var": "基线起点", "baseline_end_ms_var": "基线终点",
            "response_start_ms_var": "响应起点", "response_end_ms_var": "响应终点",
            "trial_count_var": "Trial 数", "n_stim_types_var": "刺激类型数",
            "trials_per_stim_var": "每刺激 Trial", "stimtag_var": "刺激 Tag",
            "compare_trial_a_var": "对比 Trial A", "compare_trial_b_var": "对比 Trial B",
            "analysis_view_mode_var": "分析视图", "task_response_aggregate_var": "聚合方式",
            "analysis_notch_var": "陷波", "analysis_bandpass_var": "带通",
            "analysis_band_low_var": "带通下限", "analysis_band_high_var": "带通上限",
            "smooth_signal_var": "平滑", "smooth_window_size_var": "平滑窗",
            "zscoredata_var": "Z-score", "resting_page_size_var": "静息每页数",
            "tvep_page_size_var": "任务每页数", "auto_plot_marker_alignment_var": "自动标记对齐",
            "spike_low_var": "Spike 下限", "spike_high_var": "Spike 上限",
            "spike_threshold_var": "阈值系数", "spike_noise_window_var": "噪声窗",
            "spike_step_var": "检测步长", "spike_refractory_var": "不应期",
            "spike_pre_samples_var": "峰前采样", "spike_post_samples_var": "峰后采样",
            "spike_min_count_var": "最少峰数",
        }
        box = QGroupBox(title)
        grid = QGridLayout(box)
        grid.setContentsMargins(8, 8, 8, 7)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        position = 0
        for name in names:
            widget = getattr(self, name, None)
            if not isinstance(widget, QWidget) or widget.parent() is not None:
                continue
            full_name = name.removesuffix("_var").replace("_", " ")
            pretty = compact_labels.get(name, full_name)
            cell = QWidget()
            cell.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            cell_row = QHBoxLayout(cell)
            cell_row.setContentsMargins(0, 0, 0, 0)
            cell_row.setSpacing(5)
            label = QLabel(pretty)
            label.setMinimumWidth(78)
            label.setToolTip(full_name)
            cell_row.addWidget(label)
            if isinstance(widget, QCheckBox):
                widget.setMaximumWidth(32)
            elif isinstance(widget, QComboBox):
                widget.setMinimumWidth(105)
                widget.setMaximumWidth(150)
            else:
                widget.setMinimumWidth(68)
                widget.setMaximumWidth(100)
            cell_row.addWidget(widget)
            row, column = divmod(position, 5)
            grid.addWidget(cell, row, column)
            position += 1
        # Do not leave an empty, apparently expandable panel on a page.  A
        # control already embedded in the compact top row is intentionally
        # not duplicated in this advanced panel.
        box.setVisible(position > 0)
        self._make_collapsible(box, collapsed=True)
        return box

    @staticmethod
    def _make_collapsible(box: QGroupBox, collapsed: bool = True) -> None:
        """Match the old button-driven “显示参数” behavior.

        The Tk page used a separate button, not a checkable group-box title.
        Keeping the group title passive prevents a large checkbox from
        appearing beside every parameter section and avoids accidental
        expansion while the user is operating the page.
        """
        direct_children = [child for child in box.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)]
        def apply(opened: bool) -> None:
            for child in direct_children:
                child.setVisible(opened)
            # A compact height when folded prevents the hidden parameters
            # from consuming a blank block in the scrollable page.
            box.setMaximumHeight(16777215 if opened else 26)
            box.setProperty("parameters_expanded", opened)
        box._set_parameters_expanded = apply
        apply(not collapsed)

    @staticmethod
    def _toggle_parameter_section(box: QGroupBox) -> None:
        setter = getattr(box, "_set_parameters_expanded", None)
        if callable(setter):
            setter(not bool(box.property("parameters_expanded")))

    def _toggle_preprocess_parameter_area(self) -> None:
        """Reveal or hide existing preprocessing parameters and expert actions."""
        advanced = self.preprocess_advanced_parameters
        parameter_panel = getattr(self, "preprocess_parameter_panel", None)
        if advanced.isHidden():
            if parameter_panel is not None:
                parameter_panel.setVisible(True)
            advanced.setVisible(True)
            setter = getattr(advanced, "_set_parameters_expanded", None)
            if callable(setter):
                setter(True)
            advanced.setMinimumHeight(advanced.sizeHint().height())
            expanded = True
        else:
            self._toggle_parameter_section(advanced)
            expanded = bool(advanced.property("parameters_expanded"))
            if not expanded:
                advanced.setMinimumHeight(0)
                advanced.setVisible(False)
                if parameter_panel is not None:
                    parameter_panel.setVisible(False)
        owner = advanced.parentWidget()
        if owner is not None and owner.layout() is not None:
            owner.layout().invalidate()
            owner.layout().activate()
            owner.updateGeometry()
        for button in getattr(self, "_preprocess_expert_action_buttons", ()):
            button.setVisible(expanded)
        for widget in getattr(self, "_preprocess_expert_widgets", ()):
            widget.setVisible(expanded)

    def _build_ui_legacy_layout(self) -> None:
        """Recreate the old workstation sidebar + page stack in Qt."""
        # A calm, high-contrast workstation language shared by every page.
        # The larger control metrics keep the application readable on dense
        # laboratory displays without making the analysis panels decorative.
        self.setStyleSheet(
            "QMainWindow { background:#f5f5f7; }"
            "QFrame#workspace { background:#f5f5f7; }"
            "QLabel { color:#1d1d1f; font-size:13px; }"
            "QGroupBox { background:#ffffff; border:1px solid #d2d2d7; border-radius:10px;"
            " margin-top:16px; padding:15px 13px 13px; color:#1d1d1f; font-size:14px; font-weight:600; }"
            "QGroupBox::title { subcontrol-origin:margin; left:13px; padding:0 6px; background:#f5f5f7; }"
            "QPushButton { background:#ffffff; color:#1d4f79; border:1px solid #c7d5e0; border-radius:10px;"
            " min-height:32px; padding:4px 13px; font-size:13px; font-weight:500; }"
            "QPushButton:hover { background:#f0f7fc; border-color:#6da4ca; }"
            "QPushButton:pressed { background:#e1f0fa; }"
            "QPushButton#primaryAction { background:#0071e3; color:#ffffff; border-color:#0071e3; font-weight:600; }"
            "QPushButton#primaryAction:hover { background:#0077ed; border-color:#0077ed; }"
            "QPushButton#primaryAction:pressed { background:#0068d0; border-color:#0068d0; }"
            "QPushButton#oneClickPreprocess { background:#0071e3; color:#ffffff; border-color:#0071e3;"
            " border-radius:10px; min-height:32px; padding:4px 13px; font-size:13px; font-weight:600; }"
            "QPushButton#oneClickPreprocess:hover { background:#0077ed; border-color:#0077ed; }"
            "QPushButton#oneClickPreprocess:pressed { background:#0068d0; border-color:#0068d0; }"
            "QPushButton#filterAction { background:#138a72; color:#ffffff; border:1px solid #138a72;"
            " border-radius:10px; min-height:32px; padding:4px 13px; font-size:13px; font-weight:600; }"
            "QPushButton#filterAction:hover { background:#169b80; border-color:#169b80; }"
            "QPushButton#filterAction:pressed { background:#0f715e; border-color:#0f715e; }"
            "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background:#ffffff; border:1px solid #c7d5e0;"
            " border-radius:8px; min-height:32px; padding:2px 9px; color:#1d1d1f; font-size:13px; }"
            "QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border:2px solid #0071e3; }"
            "QLabel#fileBanner { background:#eef7ff; border:1px solid #bfd9ed; border-radius:8px; padding:10px; color:#174c73; }"
            "QProgressBar { border:1px solid #d2d2d7; border-radius:7px; min-height:10px; text-align:right; padding-right:6px; }"
            "QProgressBar::chunk { background:#0071e3; border-radius:6px; }"
            "QProgressBar#preprocessProgress { background:#f7f4ff; border:1px solid #c9bdf2;"
            " border-radius:7px; color:#342568; min-height:14px; }"
            "QProgressBar#preprocessProgress::chunk { background:#7258c7; border-radius:6px; }"
            "QPlainTextEdit, QTableWidget { background:#ffffff; border:1px solid #d2d2d7; border-radius:8px; }"
            "QWidget#importPage QPushButton { min-height:48px; padding:7px 20px; font-size:15px; }"
            "QWidget#importPage QPushButton#primaryAction { min-width:154px; font-size:15px; }"
            "QWidget#importPage QLineEdit, QWidget#importPage QComboBox { min-height:44px; font-size:14px; }"
        )
        shell = QWidget()
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        screen = QApplication.primaryScreen()
        desktop_width = screen.availableGeometry().width() if screen is not None else 1920
        sidebar.setFixedWidth(210 if desktop_width < 1500 else (240 if desktop_width < 1900 else 270))
        sidebar.setStyleSheet(
            "QFrame#sidebar { background:#182431; color:#f5f5f7; }"
            "QPushButton { text-align:left; padding:12px 15px; min-height:42px; border:0; border-radius:8px;"
            " color:#ecf3f8; background:transparent; font-size:14px; font-weight:500; }"
            "QPushButton:hover { background:#26394b; }"
            "QPushButton:checked { background:#326787; color:#ffffff; font-weight:600; }"
        )
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 22, 15, 16)
        brand = QLabel("SD\nSignal Lab")
        brand.setStyleSheet("font-size:24px; font-weight:700; color:#ffffff;")
        sidebar_layout.addWidget(brand)
        workspace = QLabel("WORKSPACE")
        workspace.setStyleSheet("margin-top:24px; color:#a9c1d2; font-size:11px; font-weight:700;")
        sidebar_layout.addWidget(workspace)
        self._page_stack = QStackedWidget()
        raw_page_specs = [
            ("1  数据导入\n单个 / 批量 BIN", self._build_import_tab()),
            ("2  预处理\n开始预处理", self._build_preprocess_tab()),
            ("3  时间对齐\n确认 marker 位置", self._build_alignment_tab()),
            ("4  LFP 分析\nLFP SNR / 通道质量", self._build_lfp_tab()),
            ("5  Spike 分析\nSpike 检测 / 结果", self._build_spike_tab()),
            ("6  结果导出\n保存分析结果", self._build_export_tab()),
        ]
        page_specs = [(title, self._make_scroll_page(page)) for title, page in raw_page_specs]
        self._nav_buttons = []
        advanced_navigation = QWidget(sidebar)
        advanced_navigation.setStyleSheet("background:transparent;")
        advanced_navigation_layout = QVBoxLayout(advanced_navigation)
        advanced_navigation_layout.setContentsMargins(0, 0, 0, 0)
        advanced_navigation_layout.setSpacing(2)
        for index, (title, page) in enumerate(page_specs):
            button = QPushButton(title)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, target=index: self._activate_page(target))
            if index < 3:
                sidebar_layout.addWidget(button)
            else:
                advanced_navigation_layout.addWidget(button)
            self._nav_buttons.append(button)
            self._page_stack.addWidget(page)
        self._advanced_navigation = advanced_navigation
        self._advanced_navigation.setVisible(False)
        advanced_toggle_row = QHBoxLayout()
        advanced_toggle_row.setContentsMargins(0, 0, 0, 0)
        advanced_toggle_row.addStretch(1)
        self._advanced_navigation_toggle = QPushButton("▶")
        self._advanced_navigation_toggle.setFixedSize(38, 34)
        self._advanced_navigation_toggle.setStyleSheet(
            "QPushButton { text-align:center; padding:0; background:#26394b; "
            "border:1px solid #496579; border-radius:7px; font-size:16px; }"
            "QPushButton:hover { background:#326787; }"
        )
        self._advanced_navigation_toggle.setToolTip("展开 LFP、Spike 和结果导出")
        self._advanced_navigation_toggle.clicked.connect(self._toggle_advanced_navigation)
        advanced_toggle_row.addWidget(self._advanced_navigation_toggle)
        sidebar_layout.addLayout(advanced_toggle_row)
        sidebar_layout.addWidget(advanced_navigation)
        sidebar_layout.addStretch(1)
        self.workflow_label = QLabel("当前工作流 · 准备开始\n导入 → 预处理 → 对齐 → LFP → Spike → 导出")
        self.workflow_label.setWordWrap(True)
        self.workflow_label.setStyleSheet("color:#c2d3df; font-size:12px; padding:10px 3px;")
        self.workflow_path_var = self.workflow_label
        self.status_var = self.workflow_label
        sidebar_layout.addWidget(self.workflow_label)
        workspace_panel = QFrame()
        workspace_panel.setObjectName("workspace")
        workspace_layout = QVBoxLayout(workspace_panel)
        workspace_layout.setContentsMargins(24, 18, 24, 10)
        header = QLabel("工作台")
        header.setStyleSheet("font-size:22px; font-weight:700; color:#1d1d1f; padding:0 0 3px 2px;")
        self.workflow_title_var = header
        workspace_layout.addWidget(header)
        workspace_layout.addWidget(self._page_stack, 1)
        shell_layout.addWidget(sidebar)
        shell_layout.addWidget(workspace_panel, 1)
        self.setCentralWidget(shell)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("已连接分析流程")
        self._update_file_banners()
        self._activate_page(0)

    def _toggle_advanced_navigation(self) -> None:
        """Show or hide the optional LFP/Spike/export navigation entries."""
        expanded = not self._advanced_navigation.isVisible()
        self._advanced_navigation.setVisible(expanded)
        self._advanced_navigation_toggle.setText("▼" if expanded else "▶")
        self._advanced_navigation_toggle.setToolTip(
            "收起 LFP、Spike 和结果导出" if expanded else "展开 LFP、Spike 和结果导出"
        )

    @staticmethod
    def _make_scroll_page(page: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page.setMinimumWidth(0)
        page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        scroll.setWidget(page)
        return scroll

    def _activate_page(self, index: int) -> None:
        self._page_stack.setCurrentIndex(index)
        for position, button in enumerate(self._nav_buttons):
            button.setChecked(position == index)
        if index == 3:
            self._warn_lfp_lowpass_requirement()

    def _warn_lfp_lowpass_requirement(self) -> None:
        """Prompt once per source when LFP is entered without LFP low-pass."""
        source = self.lfp_source or self.active_source
        if source is None or not source.loaded:
            return
        ids = {int(value) for value in source.metadata.channel_ids}
        processed_ids = set(getattr(self, "_preprocess_lowpass_channel_ids", set()))
        source_key = (str(source.metadata.path), source.metadata.rows, source.metadata.channels, tuple(source.metadata.channel_ids))
        if ids and ids.issubset(processed_ids):
            return
        if getattr(self, "_last_lowpass_warning_source", None) == source_key:
            return
        self._last_lowpass_warning_source = source_key
        QMessageBox.information(
            self,
            "建议先完成低通滤波",
            "当前 LFP 数据未确认已对全部分析通道完成低通或带通低通处理。\n"
            "请先在“预处理”页完成低通/带通滤波，再进入 LFP 分析。\n"
            "该提示不阻止查看页面；已完成低通时不会显示。",
        )

    def _update_file_banners(self) -> None:
        source = self.active_source if getattr(self, "active_source", None) is not None and self.active_source.loaded else None
        eeg = "尚未加载脑电文件" if source is None else f"{source.metadata.path.name}｜{source.metadata.channels} 通道｜FS={source.metadata.fs:g} Hz"
        if hasattr(self, "preprocess_file_banner"):
            if source is None:
                self.preprocess_file_banner.setText("当前预处理数据：尚未加载脑电文件")
            else:
                original = getattr(self, "source", None)
                original_name = (
                    original.metadata.path.name
                    if original is not None and getattr(original, "loaded", False)
                    else source.metadata.path.name
                )
                if source is self.preprocessed_source:
                    stage_text = "预处理后数据"
                elif source is self.remapped_source:
                    stage_text = "重映射后数据"
                else:
                    stage_text = "原始/导入数据"
                current_name = source.metadata.path.name
                self.preprocess_file_banner.setText(
                    f"当前预处理数据：{stage_text}｜{current_name}｜"
                    f"{source.metadata.channels} 通道｜FS={source.metadata.fs:g} Hz\n"
                    f"原始加载文件：{original_name}"
                )
        aligned = f"已完成（{len(self.stim_markers)} markers）" if hasattr(self, "stim_markers") else "尚未完成"
        if hasattr(self, "alignment_banner"):
            event_name = Path(self.alignment_event_edit.text()).name if self.alignment_event_edit.text() else "未选择"
            detection_name = Path(self.alignment_detection_edit.text()).name if self.alignment_detection_edit.text() else "未选择"
            self.alignment_banner.setText(f"当前脑电：{eeg}\n对齐输入：Event CSV={event_name} ｜ 视频检测={detection_name} ｜ 状态={aligned}")
        if hasattr(self, "lfp_banner"):
            lfp = self.lfp_source.metadata.path.name if self.lfp_source is not None else "承接当前处理数据"
            self.lfp_banner.setText(f"当前脑电：{eeg}\n时间对齐：{aligned} ｜ LFP 数据：{lfp} ｜ 独立选择：{len(self.lfp_selected_ids)} 通道")
        if hasattr(self, "spike_banner"):
            spike = self.spike_source.metadata.path.name if self.spike_source is not None else "承接当前处理数据"
            self.spike_banner.setText(f"当前脑电：{eeg}\n时间对齐：{aligned} ｜ Spike 数据：{spike} ｜ 独立选择：{len(self.spike_selected_ids)} 通道")

    def _build_ui_qt(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_import_tab(), "数据导入")
        tabs.addTab(self._build_preprocess_tab(), "数据预处理")
        tabs.addTab(self._build_alignment_tab(), "时间对齐")
        tabs.addTab(self._build_lfp_tab(), "LFP 分析")
        tabs.addTab(self._build_spike_tab(), "Spike 分析")
        self.setCentralWidget(tabs)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Qt 控件框架已启动；所有绘图均使用 PyQtGraph。")

    def _build_preprocess_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        # Keep the currently used data source visible on this page at all
        # times.  Long preprocessing jobs may switch from raw to remapped or
        # filtered in-memory/disk-backed data, so the banner deliberately
        # shows both the working source and the original loaded filename.
        self.preprocess_file_banner = QLabel("当前预处理数据：尚未加载脑电文件")
        self.preprocess_file_banner.setObjectName("fileBanner")
        self.preprocess_file_banner.setWordWrap(True)
        layout.addWidget(self.preprocess_file_banner)
        self.loaded_preprocess_summary = QLabel()
        self.loaded_preprocess_summary.setWordWrap(True)
        self.loaded_preprocess_summary.setStyleSheet(
            "QLabel { background:#fff4d6; border:2px solid #e1a500; color:#5b3a00; "
            "padding:10px; font-weight:600; }"
        )
        self.loaded_preprocess_summary.setVisible(False)
        layout.addWidget(self.loaded_preprocess_summary)
        controls = QGroupBox("通道重映射与可选滤波")
        form = QFormLayout(controls)
        self.remap_edit = QLineEdit()
        remap_row = QHBoxLayout()
        remap_row.addWidget(self.remap_edit, 1)
        browse = QPushButton("选择映射 Excel…")
        browse.clicked.connect(self.choose_remap_file)
        remap_row.addWidget(browse)
        remap_holder = QWidget()
        remap_holder.setLayout(remap_row)
        self.filter_mode = QComboBox()
        # Preserve the legacy display values and their order; the processing
        # code still receives the explicit stable mode in ``userData``.
        self.filter_mode.addItem("关闭", userData="off")
        self.filter_mode.addItem("高通", userData="highpass")
        self.filter_mode.addItem("低通", userData="lowpass")
        self.filter_mode.addItem("带通", userData="bandpass")
        self.filter_mode.setCurrentIndex(self.filter_mode.findData("highpass"))
        # Current LFP production default; the overlap test below evaluates
        # this branch independently from the Spike 500-3000 Hz branch.
        self.preprocess_filter_mode_var = self.filter_mode
        self.preprocess_filter_low_var = QLineEdit("0.1")
        self.preprocess_filter_high_var = QLineEdit("300")
        self.preprocess_filter_hp_order_var = QLineEdit("3")
        self.preprocess_filter_lp_order_var = QLineEdit("5")
        self.preprocess_filter_channels_var = QLineEdit("")
        self.preprocess_filter_channels_var.setPlaceholderText("例如：1,5,20-26；留空表示全部通道")
        self.remap_rule_var = QLineEdit("")
        # Manual filtering and the one-click pipeline intentionally use two
        # independent notch switches.  Keep the legacy attribute as an alias
        # for project/settings compatibility, but show this control directly
        # beside the manual "执行滤波" action.
        self.manual_filter_notch_var = QCheckBox("同时执行 50 Hz 陷波")
        self.manual_filter_notch_var.setChecked(True)
        self.manual_filter_notch_var.setToolTip(
            "仅控制左侧“执行滤波”按钮；不会影响“一键执行已勾选步骤”或流式生成。"
        )
        self.preprocess_filter_notch_var = self.manual_filter_notch_var
        self.preprocess_filter_notch_harmonics_var = QLineEdit("1")
        self.motion_ica_enable_var = QCheckBox("启用 ICA")
        self.motion_ica_enable_var.setChecked(False)
        self.motion_ica_components_var = QLineEdit("32")
        self.motion_ica_exclude_var = QLineEdit("")
        self.motion_ica_exclude_var.setPlaceholderText("例如：0, 1；留空表示不移除成分")
        self.motion_ica_low_var = QLineEdit("1")
        self.motion_ica_high_var = QLineEdit("100")
        self.motion_ica_decim_var = QLineEdit("3")
        self.motion_ica_max_iter_var = QLineEdit("800")
        # Bad-channel checking is a mandatory workflow step. Keep a checked
        # compatibility control for existing internal calls, but do not expose
        # a checkbox that can accidentally disable the whole quality pass.
        self.bad_check_var = QCheckBox()
        self.bad_check_var.setChecked(True)
        self.bad_parallel_var = QCheckBox("启用并行")
        self.bad_parallel_var.setChecked(True)
        self.bad_workers_var = QLineEdit("0")
        self.bad_fast_artifact_check_var = QCheckBox("\u8d34\u5e95\u9971\u548c\u68c0\u67e5")
        self.bad_fast_artifact_check_var.setChecked(True)
        self.bad_saturation_width_percent_var = QLineEdit("1.0")
        self.bad_saturation_ratio_threshold_var = QLineEdit("45")
        self.bad_high_frequency_noise_check_var = QCheckBox("2.5mV附近集中坏道检查")
        self.bad_high_frequency_noise_check_var.setChecked(True)
        self.bad_high_frequency_noise_target_var = QLineEdit("2500")
        self.bad_high_frequency_noise_tolerance_var = QLineEdit("1")
        self.bad_high_frequency_noise_ratio_threshold_var = QLineEdit("50")
        fast_artifact_tip = (
            "\u4ee5\u6bcf\u901a\u9053\u5168\u8bb0\u5f55\u6700\u5c0f\u503c\u548c PTP \u5b9a\u4e49\u8d34\u5e95\u7a84\u533a\u95f4\uff1b"
            "\u6700\u5c0f\u503c\u9644\u8fd1\u7684\u91c7\u6837\u70b9\u6bd4\u4f8b\u8fbe\u5230\u9608\u503c\u65f6\u5224\u4e3a\u9971\u548c\u574f\u9053\u3002"
        )
        self.bad_fast_artifact_check_var.setToolTip(fast_artifact_tip)
        self.bad_saturation_width_percent_var.setToolTip("\u8d34\u5e95\u7a84\u533a\u95f4\u5bbd\u5ea6\uff0c\u5360\u8be5\u901a\u9053\u5168\u8bb0\u5f55 PTP \u7684\u767e\u5206\u6bd4\u3002")
        self.bad_saturation_ratio_threshold_var.setToolTip("\u8d34\u5e95\u91c7\u6837\u70b9\u6bd4\u4f8b\u8fbe\u5230\u6b64\u767e\u5206\u6bd4\u65f6\uff0c\u5224\u4e3a\u9971\u548c\u574f\u9053\u3002")
        # Complete LFP/Spike streaming-product parameters.  These are separate
        # from the interactive preview filter above and are frozen into both
        # output H5 provenance documents and the project manifest.
        self.stream_lfp_low_var = QLineEdit("0.5")
        self.stream_lfp_high_var = QLineEdit("300")
        self.stream_lfp_overlap_var = QLineEdit("60")
        self.stream_lfp_notch_var = QCheckBox("LFP 50 Hz 陷波")
        self.stream_lfp_notch_var.setChecked(True)
        self.stream_notch_frequency_var = QLineEdit("50")
        self.stream_notch_q_var = QLineEdit("30")
        self.stream_notch_harmonics_var = QLineEdit("1")
        self.stream_spike_low_var = QLineEdit("500")
        self.stream_spike_high_var = QLineEdit("3000")
        self.stream_spike_overlap_var = QLineEdit("0.25")
        self.stream_spike_notch_var = QCheckBox("Spike 陷波")
        self.stream_spike_notch_var.setChecked(False)
        self.stream_lfp_car_var = QCheckBox("LFP Leave-one-out median CAR")
        self.stream_lfp_car_var.setChecked(False)
        self.stream_max_workers_var = QLineEdit("24")
        self.stream_max_workers_var.setToolTip("滤波计算任务上限；实际数量还受CPU、通道组数和内存预算限制。")
        self.stream_memory_budget_gb_var = QLineEdit("16")
        self.stream_memory_budget_gb_var.setToolTip("并行滤波临时数组的估算内存预算，不是整个程序的内存上限。")
        advanced = QGroupBox("预处理参数（与旧版一致）")
        self.preprocess_advanced_parameters = advanced
        advanced_grid = QGridLayout(advanced)
        advanced_items = [
            ("映射规则说明", self.remap_rule_var),
            ("ICA 成分", self.motion_ica_components_var),
            ("ICA 排除", self.motion_ica_exclude_var), ("ICA 低频", self.motion_ica_low_var),
            ("ICA 高频", self.motion_ica_high_var), ("ICA decim", self.motion_ica_decim_var),
            ("ICA 最大迭代", self.motion_ica_max_iter_var), ("坏道 workers", self.bad_workers_var),
            ("工频谐波数（1=50 Hz）", self.preprocess_filter_notch_harmonics_var),
            ("\u8d34\u5e95\u533a\u95f4\u5bbd\u5ea6 % PTP", self.bad_saturation_width_percent_var),
            ("\u8d34\u5e95\u6bd4\u4f8b\u9608\u503c %", self.bad_saturation_ratio_threshold_var),
            ("流式 LFP 低截止 Hz", self.stream_lfp_low_var),
            ("流式 LFP 高截止 Hz", self.stream_lfp_high_var),
            ("流式 LFP 单侧重叠秒", self.stream_lfp_overlap_var),
            ("流式 Spike 低截止 Hz", self.stream_spike_low_var),
            ("流式 Spike 高截止 Hz", self.stream_spike_high_var),
            ("流式 Spike 单侧重叠秒", self.stream_spike_overlap_var),
            ("流式陷波频率 Hz", self.stream_notch_frequency_var),
            ("流式陷波 Q", self.stream_notch_q_var),
            ("流式陷波谐波数", self.stream_notch_harmonics_var),
            ("流式最大并行任务", self.stream_max_workers_var),
            ("流式并行内存预算 GB", self.stream_memory_budget_gb_var),
        ]
        advanced_items.extend([
            ("2.5mV附近目标值", self.bad_high_frequency_noise_target_var),
            ("2.5mV附近容差 ±", self.bad_high_frequency_noise_tolerance_var),
            ("2.5mV附近比例阈值 %", self.bad_high_frequency_noise_ratio_threshold_var),
        ])
        # This panel is displayed horizontally above the EEG preview.  Eight
        # columns keep the parameter area shallow enough that the waveform
        # retains useful vertical space on common 1080p/2K screens.
        parameter_columns = 8
        for index, (label, widget) in enumerate(advanced_items):
            row, column = divmod(index, parameter_columns)
            label_widget = QLabel(label)
            label_widget.setWordWrap(True)
            label_widget.setMinimumHeight(30)
            label_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            advanced_grid.addWidget(label_widget, row * 2, column)
            advanced_grid.addWidget(widget, row * 2 + 1, column)
        advanced_grid.setHorizontalSpacing(14)
        advanced_grid.setVerticalSpacing(5)
        for column in range(parameter_columns):
            advanced_grid.setColumnStretch(column, 1)
        check_grid = QGridLayout()
        check_grid.setHorizontalSpacing(18)
        check_grid.setVerticalSpacing(4)
        check_widgets = (
            self.motion_ica_enable_var,
            self.bad_parallel_var,
            self.bad_fast_artifact_check_var,
            self.bad_high_frequency_noise_check_var,
            self.stream_lfp_notch_var, self.stream_spike_notch_var,
            self.stream_lfp_car_var,
        )
        check_columns = 6
        check_start_row = ((len(advanced_items) + parameter_columns - 1) // parameter_columns) * 2 + 1
        for index, widget in enumerate(check_widgets):
            row, column = divmod(index, check_columns)
            check_grid.addWidget(widget, row, column)
        advanced_grid.addLayout(check_grid, check_start_row, 0, 2, parameter_columns)
        self._make_collapsible(advanced, collapsed=True)
        advanced.setVisible(False)
        action_row = QHBoxLayout()
        remap_only = QPushButton("执行通道重映射")
        remap_only.clicked.connect(lambda: self.run_preprocess(force_filter_off=True))
        full_h5 = QPushButton("导入完整 H5 做预处理")
        full_h5.clicked.connect(self.choose_file)
        overview = QPushButton("选择滤波通道（总览）")
        overview.clicked.connect(self.open_preprocess_filter_overview)
        show_preprocess_params = QPushButton("显示参数")
        self.preprocess_params_button_var = show_preprocess_params
        show_preprocess_params.clicked.connect(self._toggle_preprocess_parameter_area)
        # Keep the four primary preprocessing actions on one row and give
        # each the same stretch so they fill all space released by the
        # removed experiment/test buttons.
        action_buttons = (
            full_h5, overview, remap_only, show_preprocess_params,
        )
        self._preprocess_action_buttons = action_buttons
        for button in action_buttons:
            button.setObjectName("oneClickPreprocess")
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            action_row.addWidget(button, 1)
        action_holder = QWidget(); action_holder.setLayout(action_row)
        filter_channels_row = QHBoxLayout()
        filter_channels_row.addWidget(self.preprocess_filter_channels_var, 1)
        filter_all = QPushButton("全部通道")
        filter_all.setToolTip("清空通道选择；下一次滤波将应用到全部通道")
        filter_all.clicked.connect(self.preprocess_filter_channels_var.clear)
        filter_channels_row.addWidget(filter_all)
        filter_channels_holder = QWidget()
        filter_channels_holder.setLayout(filter_channels_row)
        analysis_row = QHBoxLayout()
        apply_ica = QPushButton("应用选中 ICA")
        apply_ica.clicked.connect(self.run_selected_ica)
        clear_ica = QPushButton("清除 ICA")
        clear_ica.clicked.connect(self.clear_ica)
        run_bad_check = QPushButton("执行坏道检查")
        run_bad_check.clicked.connect(self.run_bad_channel_check)
        show_bad_contribution = QPushButton("当前规则触发数")
        show_bad_contribution.setToolTip(
            "仅显示本次各规则触发数量；不能据此判断规则是否必要。"
        )
        show_bad_contribution.clicked.connect(self.show_current_bad_channel_contribution)
        self.show_bad_channel_contribution_button = show_bad_contribution
        run_fast_artifact = QPushButton("\u5355\u72ec\u6267\u884c\u8d34\u5e95\u9971\u548c")
        run_fast_artifact.clicked.connect(self.run_fast_artifact_check)
        run_high_frequency_noise = QPushButton(
            "2.5mV附近集中坏道检查"
        )
        run_high_frequency_noise.clicked.connect(self.run_high_frequency_noise_check)
        mark_good = QPushButton("标记选中为健康")
        mark_good.clicked.connect(lambda: self._set_selected_channel_review("good"))
        mark_bad = QPushButton("标记选中为坏道")
        mark_bad.clicked.connect(lambda: self._set_selected_channel_review("bad"))
        restore_auto = QPushButton("恢复自动判定")
        restore_auto.clicked.connect(self._restore_selected_channel_review)
        apply_car = QPushButton("应用 Leave-one-out median CAR")
        apply_car.setObjectName("primaryAction")
        apply_car.setEnabled(False)
        apply_car.clicked.connect(self.run_leave_one_out_median_car)
        self.apply_car_button = apply_car
        analysis_row.addWidget(apply_ica)
        analysis_row.addWidget(clear_ica)
        analysis_row.addWidget(run_bad_check)
        analysis_row.addWidget(show_bad_contribution)
        analysis_row.addWidget(run_fast_artifact)
        analysis_row.addWidget(run_high_frequency_noise)
        run_selected_analysis = QPushButton("执行已勾选检测")
        run_selected_analysis.setToolTip("使用原始/重映射数据运行当前勾选的质量检测；滤波和 ICA 结果不会作为这些检测的输入。")
        run_selected_analysis.clicked.connect(self.run_selected_analysis)
        self.run_selected_analysis_button = run_selected_analysis
        show_detection_timing = QPushButton("运行耗时")
        show_detection_timing.setToolTip("查看最近一次一键流程或单独质量检测的阶段墙钟耗时")
        show_detection_timing.clicked.connect(self.show_bad_channel_timing)
        self.show_detection_timing_button = show_detection_timing
        analysis_row.addWidget(run_selected_analysis)
        analysis_row.addWidget(show_detection_timing)
        analysis_row.addWidget(apply_car)
        analysis_row.addStretch(1)
        analysis_holder = QWidget(); analysis_holder.setLayout(analysis_row)
        self.preprocess_run_button = QPushButton("执行滤波")
        self.preprocess_run_button.setObjectName("filterAction")
        self.preprocess_run_button.setToolTip(
            "按右侧模式、截止频率和阶数执行普通滤波；是否同时陷波只由本行的“同时执行 50 Hz 陷波”控制。"
        )
        self.preprocess_run_button.clicked.connect(self.run_preprocess)
        self.preprocess_export_button = QPushButton("导出预处理 H5 + CSV")
        self.preprocess_export_button.setToolTip(
            "导出当前完整通道矩阵；已处理通道使用当前结果，其余通道保持原值，"
            "并同时生成同名 CSV 处理记录。"
        )
        self.preprocess_export_button.setEnabled(False)
        self.preprocess_export_button.clicked.connect(self.show_preprocessed_export_workflow)
        self.extract_original_channel_button = QPushButton("提取原始通道")
        self.extract_original_channel_button.setObjectName("filterAction")
        self.extract_original_channel_button.setToolTip(
            "根据重映射关系找回原始物理通道并单独导出；不执行任何处理。"
        )
        self.extract_original_channel_button.clicked.connect(self.extract_original_channels)
        self.merge_repaired_channel_button = QPushButton("并入处理好通道")
        self.merge_repaired_channel_button.setObjectName("filterAction")
        self.merge_repaired_channel_button.setToolTip(
            "把刚才提取并由你处理好的通道，直接并入先前选定的预处理 H5。"
        )
        self.merge_repaired_channel_button.setEnabled(False)
        self.merge_repaired_channel_button.clicked.connect(self.merge_repaired_channel)
        channel_repair_actions = QHBoxLayout()
        channel_repair_actions.setContentsMargins(0, 0, 0, 0)
        channel_repair_actions.setSpacing(12)
        channel_repair_actions.addWidget(self.extract_original_channel_button, 1)
        channel_repair_actions.addWidget(self.merge_repaired_channel_button, 1)
        channel_repair_holder = QWidget()
        channel_repair_holder.setLayout(channel_repair_actions)
        self.one_click_preprocess_check = QCheckBox(
            "一键流程执行 50 Hz 工频陷波（仅控制陷波，不控制重映射或普通滤波）"
        )
        self.one_click_preprocess_check.setToolTip(
            "该勾选只决定一键流程是否执行 50 Hz 工频陷波；重映射由映射文件和流程需要决定，"
            "普通高通/低通请使用上方“执行滤波”。"
        )
        self.one_click_preprocess_check.setChecked(True)
        self.one_click_quality_check = QCheckBox("通道质量检查")
        self.one_click_quality_check.setChecked(True)
        self.one_click_car_check = QCheckBox("自动应用 CAR")
        self.one_click_car_check.toggled.connect(
            lambda checked: self.one_click_quality_check.setChecked(True) if checked else None
        )
        one_click_steps = QHBoxLayout()
        one_click_steps.setContentsMargins(0, 0, 0, 0)
        one_click_steps.setSpacing(18)
        for step_check in (
            self.one_click_preprocess_check,
            self.one_click_quality_check,
            self.one_click_car_check,
        ):
            one_click_steps.addWidget(step_check)
        self.dual_stream_export_button = QPushButton("流式生成 LFP + Spike H5")
        self.dual_stream_export_button.clicked.connect(self.show_dual_stream_export_workflow)
        self.dual_stream_export_button.setEnabled(True)
        self.dual_stream_export_button.setToolTip(
            "从重映射后的原始数据（未重映射时为最初加载数据）独立生成两个完整 H5；"
            "不继承普通滤波、CAR 或 ICA 结果"
        )
        one_click_steps.addStretch(1)
        one_click_steps_holder = QWidget()
        one_click_steps_holder.setLayout(one_click_steps)
        self.one_click_plan_summary = QLabel()
        self.one_click_plan_summary.setObjectName("oneClickPlanSummary")
        self.one_click_plan_summary.setWordWrap(True)
        self.one_click_plan_summary.setStyleSheet(
            "QLabel#oneClickPlanSummary {"
            " background:#eef6ff; border:1px solid #9fc5e8; border-radius:6px;"
            " color:#123c68; padding:8px 10px; font-weight:600;"
            "}"
        )
        self.one_click_plan_summary.setToolTip(
            "根据当前映射文件、执行步骤及显示参数，实时预览一键执行的实际流程。"
        )
        self.one_click_preprocess_button = QPushButton("一键执行已勾选步骤")
        self.one_click_preprocess_button.setObjectName("oneClickPreprocess")
        self.one_click_preprocess_button.clicked.connect(self.run_one_click_preprocess)
        filter_action_row = QHBoxLayout()
        filter_action_row.setContentsMargins(0, 0, 0, 0)
        filter_action_row.setSpacing(10)
        filter_action_row.addWidget(self.preprocess_run_button)
        filter_action_row.addWidget(self.manual_filter_notch_var)
        filter_controls = (
            ("滤波模式", self.filter_mode),
            ("低截止 Hz", self.preprocess_filter_low_var),
            ("高截止 Hz", self.preprocess_filter_high_var),
            ("高通阶数", self.preprocess_filter_hp_order_var),
            ("低通阶数", self.preprocess_filter_lp_order_var),
        )
        for label_text, widget in filter_controls:
            label_widget = QLabel(label_text)
            label_widget.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
            filter_action_row.addWidget(label_widget)
            filter_action_row.addWidget(widget, 1)
            if widget is self.preprocess_filter_low_var:
                self._preprocess_filter_low_label = label_widget
            elif widget is self.preprocess_filter_high_var:
                self._preprocess_filter_high_label = label_widget
        filter_action_holder = QWidget()
        filter_action_holder.setLayout(filter_action_row)
        self.filter_mode.currentIndexChanged.connect(self._update_preprocess_filter_cutoff_visibility)
        self._update_preprocess_filter_cutoff_visibility()
        output_actions = QHBoxLayout()
        output_actions.setContentsMargins(0, 0, 0, 0)
        output_actions.setSpacing(18)
        for button in (self.dual_stream_export_button, self.preprocess_export_button):
            button.setObjectName("oneClickPreprocess")
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            output_actions.addWidget(button, 1)
        output_actions_holder = QWidget()
        output_actions_holder.setLayout(output_actions)
        for button, color in (
            (self.one_click_preprocess_button, QColor(34, 88, 170, 85)),
            (self.preprocess_run_button, QColor(16, 102, 83, 80)),
        ):
            shadow = QGraphicsDropShadowEffect(button)
            shadow.setBlurRadius(16.0)
            shadow.setOffset(0.0, 3.0)
            shadow.setColor(color)
            button.setGraphicsEffect(shadow)
        self._preprocess_expert_action_buttons = ()
        self._preprocess_expert_widgets = (
            analysis_holder, channel_repair_holder,
        )
        for widget in self._preprocess_expert_widgets:
            widget.setVisible(False)
        self.preprocess_progress = PreprocessProgressBar()
        self.preprocess_progress.setObjectName("preprocessProgress")
        self.preprocess_progress.pause_clicked.connect(self._toggle_preprocess_pause)
        self.preprocess_progress.cancel_clicked.connect(self._cancel_preprocess_run)
        progress_shadow = QGraphicsDropShadowEffect(self.preprocess_progress)
        progress_shadow.setBlurRadius(12.0)
        progress_shadow.setOffset(0.0, 2.0)
        progress_shadow.setColor(QColor(70, 49, 130, 55))
        self.preprocess_progress.setGraphicsEffect(progress_shadow)
        self.preprocess_status = QLabel("请先在“数据导入”页加载 HDF5 文件。")
        self.preprocess_status.setWordWrap(True)
        self.preprocess_filter_scope = QLabel("实际滤波范围：尚未执行。滤波通道留空时才会处理全部通道。")
        self.preprocess_filter_scope.setWordWrap(True)
        self.preprocess_filter_scope.setStyleSheet("color:#24516e; padding:2px 0;")
        self.preprocess_data_state = QLabel("数据状态：尚未加载，未进行重映射或滤波。")
        self.preprocess_data_state.setWordWrap(True)
        # Keep this label as a compatibility state holder, but render its text
        # inside one_click_plan_summary so process and data state stay together.
        self._preprocess_data_state_text = self.preprocess_data_state.text()
        self.preprocess_status_var = self.preprocess_status
        self.preprocess_filter_status_var = self.preprocess_status
        self.preprocess_filter_progress_var = self.preprocess_progress
        form.addRow("映射 Excel（H2:H）", remap_holder)
        form.addRow("滤波通道（留空=全部）", filter_channels_holder)
        form.addRow(action_holder)
        form.addRow(analysis_holder)
        form.addRow("执行步骤", one_click_steps_holder)
        form.addRow(self.one_click_plan_summary)
        form.addRow(filter_action_holder)
        form.addRow(self.one_click_preprocess_button)
        form.addRow(output_actions_holder)
        form.addRow(channel_repair_holder)
        form.addRow("进度", self.preprocess_progress)
        form.addRow(self.preprocess_status)
        form.addRow(self.preprocess_filter_scope)
        for signal in (
            self.remap_edit.textChanged,
            self.one_click_preprocess_check.toggled,
            self.one_click_quality_check.toggled,
            self.one_click_car_check.toggled,
            self.motion_ica_enable_var.toggled,
            self.filter_mode.currentIndexChanged,
        ):
            signal.connect(self._update_one_click_plan_summary)
        self._update_one_click_plan_summary()
        layout.addWidget(controls)
        bad_result_box = QGroupBox("坏道检查结果（实际通道号与判定理由）")
        self.bad_channel_result_box = bad_result_box
        bad_result_layout = QVBoxLayout(bad_result_box)
        self.bad_channel_result_summary = QLabel("尚未执行坏道检查。")
        result_filter_row = QHBoxLayout()
        result_filter_row.addWidget(QLabel("显示通道："))
        self.bad_channel_result_filter = QComboBox()
        self.bad_channel_result_filter.addItem("全部", userData="all")
        self.bad_channel_result_filter.addItem("人工覆盖", userData="manual")
        self.bad_channel_result_filter.addItem("坏道", userData="bad")
        self.bad_channel_result_filter.addItem("健康道", userData="good")
        self.bad_channel_result_filter.currentIndexChanged.connect(self._refresh_bad_channel_review_table)
        result_filter_row.addWidget(self.bad_channel_result_filter)
        result_filter_row.addSpacing(12)
        result_filter_row.addWidget(mark_good)
        result_filter_row.addWidget(mark_bad)
        result_filter_row.addWidget(restore_auto)
        result_filter_row.addStretch(1)
        self.bad_channel_reason_table = QTableWidget(0, 3)
        self.bad_channel_reason_table.setHorizontalHeaderLabels(["\u901a\u9053", "\u7ed3\u679c", "\u5224\u5b9a\u8bf4\u660e"])
        self.bad_channel_reason_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.bad_channel_reason_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.bad_channel_reason_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        # Keep the automatically advanced row visibly blue even while one of
        # the review buttons temporarily owns the active-window focus.
        self.bad_channel_reason_table.setStyleSheet("""
            QTableWidget::item:selected,
            QTableWidget::item:selected:!active {
                background-color: #0b74de;
                color: white;
            }
        """)
        self.bad_channel_reason_table.setMinimumHeight(145)
        self.bad_channel_reason_table.verticalHeader().setVisible(False)
        self.bad_channel_reason_table.horizontalHeader().setStretchLastSection(True)
        self.bad_channel_reason_table.setColumnWidth(0, 90)
        self.bad_channel_reason_table.setColumnWidth(1, 80)
        self.bad_channel_reason_table.horizontalHeader().sectionClicked.connect(
            self._toggle_bad_channel_review_description_mode
        )
        bad_result_layout.addWidget(self.bad_channel_result_summary)
        bad_result_layout.addLayout(result_filter_row)
        bad_result_layout.addWidget(self.bad_channel_reason_table)
        self.preprocessed_preview = PyQtGraphPreview(
            self.active_source, title_prefix="处理后数据",
        )
        self.original_preprocess_preview = PyQtGraphPreview(
            self.source, title_prefix="原始数据（处理前）",
        )
        self.preprocess_preview_tabs = QTabWidget()
        self.preprocess_preview_tabs.setObjectName("preprocessPreviewTabs")
        self.preprocess_preview_tabs.addTab(self.preprocessed_preview, "处理后数据")
        self.preprocess_preview_tabs.addTab(
            self.original_preprocess_preview, "原始数据（处理前）",
        )
        # The comparison page appears only after a separate processed source
        # exists. Until then both pages would show the same samples.
        if hasattr(self.preprocess_preview_tabs, "setTabVisible"):
            self.preprocess_preview_tabs.setTabVisible(1, False)
        else:
            self.preprocess_preview_tabs.setTabEnabled(1, False)
        self._connect_preprocess_comparison_controls(
            self.preprocessed_preview, self.original_preprocess_preview,
        )
        self._connect_preprocess_comparison_controls(
            self.original_preprocess_preview, self.preprocessed_preview,
        )
        self.preprocess_preview_tabs.currentChanged.connect(
            self._preprocess_comparison_tab_changed
        )
        # Exact legacy preview parameter names are aliases for the visible Qt
        # controls, so programmatic callers and saved workflows keep working.
        self.preprocessed_preview_channel_var = self.preprocessed_preview.channel_spin
        self.preprocessed_preview_start_var = self.preprocessed_preview.start_spin
        self.preprocessed_preview_end_var = self.preprocessed_preview.end_spin
        # Legacy alias: duration is derived from start/end when a custom window is used.
        self.preprocessed_preview_duration_var = self.preprocessed_preview.end_spin
        self.preprocessed_preview_linewidth_var = self.preprocessed_preview.linewidth_spin
        self.preprocessed_preview_status_var = self.preprocessed_preview.status_label
        self.preprocessed_preview.channel_previewed.connect(
            self._highlight_bad_channel_review_row
        )
        self.preprocessed_preview.channel_previewed.connect(
            self._sync_repair_filter_channel
        )
        self.preprocessed_preview.channel_previewed.connect(
            lambda _channel_id: self._refresh_matching_preprocess_comparison(
                self.preprocessed_preview, self.original_preprocess_preview,
            )
        )
        self.original_preprocess_preview.channel_previewed.connect(
            self._highlight_bad_channel_review_row
        )
        self.original_preprocess_preview.channel_previewed.connect(
            self._sync_repair_filter_channel
        )
        self.original_preprocess_preview.channel_previewed.connect(
            lambda _channel_id: self._refresh_matching_preprocess_comparison(
                self.original_preprocess_preview, self.preprocessed_preview,
            )
        )
        self.bad_channel_reason_table.itemSelectionChanged.connect(
            self._preview_selected_bad_channel_review_row
        )
        workspace = QWidget()
        workspace.setObjectName("preprocessWorkspace")
        self.preprocess_workspace = workspace
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(10)

        parameter_panel = QWidget()
        self.preprocess_parameter_panel = parameter_panel
        parameter_panel.setMinimumWidth(0)
        parameter_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        parameter_layout = QVBoxLayout(parameter_panel)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        parameter_layout.addWidget(advanced)

        # Stack both full-width auxiliary areas above the plot: preprocessing
        # parameters first, bad-channel results directly underneath.  When
        # parameters are hidden, the result panel automatically moves up.
        upper_panels = QWidget()
        upper_panels.setObjectName("preprocessUpperPanels")
        upper_panels.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.preprocess_upper_panels = upper_panels
        upper_panels_layout = QVBoxLayout(upper_panels)
        upper_panels_layout.setContentsMargins(0, 0, 0, 0)
        upper_panels_layout.setSpacing(10)
        upper_panels_layout.addWidget(parameter_panel)
        parameter_panel.setVisible(False)
        bad_result_box.setMinimumWidth(0)
        bad_result_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        upper_panels_layout.addWidget(bad_result_box)
        workspace_layout.addWidget(upper_panels, 0)

        self.preprocess_preview_tabs.setMinimumWidth(0)
        self.preprocess_preview_tabs.setMinimumHeight(390)
        self.preprocess_preview_tabs.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        workspace_layout.addWidget(self.preprocess_preview_tabs, 1)
        layout.addWidget(workspace, 1)
        return page

    def _connect_preprocess_comparison_controls(self, source_preview, target_preview) -> None:
        """Keep both comparison pages on the same channel and time window."""
        for signal in (
            source_preview.channel_spin.valueChanged,
            source_preview.start_spin.valueChanged,
            source_preview.end_spin.valueChanged,
            source_preview.linewidth_spin.valueChanged,
            source_preview.full_duration_check.toggled,
        ):
            signal.connect(
                lambda _value=None, source=source_preview, target=target_preview:
                self._sync_preprocess_comparison_controls(source, target)
            )

    @staticmethod
    def _sync_preprocess_comparison_controls(source_preview, target_preview) -> None:
        """Copy visible preview controls, matching channels by real channel ID."""
        source_meta = getattr(source_preview.source, "metadata", None)
        target_meta = getattr(target_preview.source, "metadata", None)
        if source_meta is None or target_meta is None:
            return

        source_channel_id = source_preview.current_channel_id()
        target_ids = tuple(int(value) for value in target_meta.channel_ids)
        if source_channel_id in target_ids:
            target_column = target_ids.index(source_channel_id) + 1
        else:
            target_column = min(source_preview.channel_spin.value(), target_meta.channels)

        controls = (
            target_preview.channel_spin,
            target_preview.start_spin,
            target_preview.end_spin,
            target_preview.linewidth_spin,
            target_preview.full_duration_check,
        )
        for control in controls:
            control.blockSignals(True)
        try:
            full_duration = source_preview.full_duration_check.isChecked()
            target_preview.channel_spin.setValue(target_column)
            target_preview.full_duration_check.setChecked(full_duration)
            target_preview.start_spin.setEnabled(not full_duration)
            target_preview.end_spin.setEnabled(not full_duration)
            target_preview.start_spin.setValue(source_preview.start_spin.value())
            target_preview.end_spin.setValue(source_preview.end_spin.value())
            target_preview.linewidth_spin.setValue(source_preview.linewidth_spin.value())
        finally:
            for control in controls:
                control.blockSignals(False)

    def _refresh_matching_preprocess_comparison(
        self, source_preview, target_preview,
    ) -> None:
        """Redraw the other comparison page on the same physical channel."""
        if bool(getattr(self, "_preprocess_comparison_refreshing", False)):
            return
        if getattr(target_preview.source, "metadata", None) is None:
            return
        if source_preview.source is target_preview.source:
            return
        # A refresh emits ``channel_previewed``. Guard the reciprocal signal so
        # refreshing the second plot does not bounce forever between the tabs.
        self._preprocess_comparison_refreshing = True
        try:
            self._sync_preprocess_comparison_controls(source_preview, target_preview)
            target_preview._refresh_now()
        finally:
            self._preprocess_comparison_refreshing = False

    def _preprocess_comparison_source(self):
        """Return the matching data immediately before filter/CAR/ICA processing."""
        remapped = getattr(self, "remapped_source", None)
        if remapped is not None and remapped.loaded:
            return remapped
        source = getattr(self, "source", None)
        return source if source is not None and source.loaded else None

    def _update_preprocess_comparison_preview(self, *, defer_initial_draw: bool = True) -> None:
        """Refresh the original-data tab and show it only for a real comparison."""
        original = self._preprocess_comparison_source()
        processed = getattr(self, "preprocessed_source", None)
        comparison_available = (
            original is not None and processed is not None and processed is not original
        )
        if original is not None:
            self.original_preprocess_preview.set_source(
                original, defer_initial_draw=defer_initial_draw,
            )
            self._sync_preprocess_comparison_controls(
                self.preprocessed_preview, self.original_preprocess_preview,
            )
        if hasattr(self.preprocess_preview_tabs, "setTabVisible"):
            self.preprocess_preview_tabs.setTabVisible(1, comparison_available)
        else:
            self.preprocess_preview_tabs.setTabEnabled(1, comparison_available)
        if not comparison_available and self.preprocess_preview_tabs.currentIndex() == 1:
            self.preprocess_preview_tabs.setCurrentIndex(0)

    def _preprocess_comparison_tab_changed(self, index: int) -> None:
        """Synchronize and draw the page selected by the operator."""
        if index not in (0, 1):
            return
        target = (
            self.preprocessed_preview if index == 0
            else self.original_preprocess_preview
        )
        source = (
            self.original_preprocess_preview if index == 0
            else self.preprocessed_preview
        )
        if getattr(target.source, "metadata", None) is None:
            return
        self._sync_preprocess_comparison_controls(source, target)
        target._refresh_now()

    def _update_preprocess_filter_cutoff_visibility(self, _index: int | None = None) -> None:
        """Show only cutoff controls that apply to the selected filter mode."""
        mode = str(self.filter_mode.currentData() or "off").strip().lower()
        show_low = mode in {"highpass", "bandpass"}
        show_high = mode in {"lowpass", "bandpass"}

        self._preprocess_filter_low_label.setText("低截止 Hz")
        self._preprocess_filter_high_label.setText("高截止 Hz")
        self._preprocess_filter_low_label.setVisible(show_low)
        self.preprocess_filter_low_var.setVisible(show_low)
        self._preprocess_filter_high_label.setVisible(show_high)
        self.preprocess_filter_high_var.setVisible(show_high)

    def _build_export_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("结果导出")
        form = QFormLayout(box)
        self.export_path_edit = QLineEdit()
        path_row = QHBoxLayout(); path_row.addWidget(self.export_path_edit, 1)
        browse = QPushButton("选择输出文件…"); browse.clicked.connect(self.choose_export_path); path_row.addWidget(browse)
        holder = QWidget(); holder.setLayout(path_row)
        self.export_button = QPushButton("导出当前数据为 HDF5")
        self.export_button.setObjectName("primaryAction")
        self.export_button.clicked.connect(self.export_current_h5)
        self.export_status = QLabel("导出当前数据源：优先为预处理后的数据。")
        self.export_status.setWordWrap(True)
        form.addRow("输出 HDF5", holder)
        form.addRow(self.export_button)
        form.addRow(self.export_status)
        layout.addWidget(box)
        options = QGroupBox("逐项导出（与旧版一致）")
        grid = QGridLayout(options)
        self.export_lfp_check = QCheckBox("静息态多频段 SNR CSV"); self.export_lfp_check.setChecked(True)
        self.export_spike_check = QCheckBox("Spike 结果 CSV"); self.export_spike_check.setChecked(True)
        self.export_spike_plot_check = QCheckBox("Spike 结果 + 图"); self.export_spike_plot_check.setChecked(True)
        self.export_preprocess_qc_check = QCheckBox("预处理 QC 结果 CSV")
        self.export_selected_h5_check = QCheckBox("当前选择通道 H5")
        self.export_processed_h5_check = QCheckBox("当前处理数据集 H5"); self.export_processed_h5_check.setChecked(True)
        option_rows = [
            (self.export_lfp_check, self.export_resting_multiband_csv),
            (self.export_spike_check, self.export_spike_csv),
            (self.export_spike_plot_check, self.export_spike_plot),
            (self.export_preprocess_qc_check, self.export_preprocess_qc_csv),
            (self.export_selected_h5_check, self.export_processed_channel_h5_files),
            (self.export_processed_h5_check, self.export_current_h5),
        ]
        for row, (check, callback) in enumerate(option_rows):
            button = QPushButton("导出"); button.clicked.connect(callback)
            grid.addWidget(check, row, 0); grid.addWidget(button, row, 1)
        export_selected = QPushButton("导出勾选结果")
        export_selected.setObjectName("primaryAction")
        export_selected.clicked.connect(self.export_checked_results)
        output_dir = QPushButton("选择目录")
        output_dir.clicked.connect(self.choose_output_directory)
        open_dir = QPushButton("打开输出文件夹")
        open_dir.clicked.connect(self.open_output_directory)
        grid.addWidget(export_selected, len(option_rows), 0); grid.addWidget(output_dir, len(option_rows), 1); grid.addWidget(open_dir, len(option_rows), 2)
        layout.addWidget(options)
        layout.addStretch(1)
        return page

    def _build_lfp_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("lfpPage")
        page.setStyleSheet(
            "QWidget#lfpPage QGroupBox { margin-top:10px; padding:8px 8px 7px; }"
            "QWidget#lfpPage QGroupBox::title { left:9px; padding:0 4px; }"
        )
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(5)
        self.lfp_banner = QLabel()
        self.lfp_banner.setObjectName("fileBanner")
        self.lfp_banner.setWordWrap(True)
        self.lfp_banner.setStyleSheet("background:#eef5fa; border:1px solid #b6cddd; padding:8px; color:#18384e;")
        self.lfp_file_banner_var = self.lfp_banner
        layout.addWidget(self.lfp_banner)
        controls = QGroupBox("静息态分析参数与执行")
        form = QFormLayout(controls)
        self.lfp_signal_low = QDoubleSpinBox()
        self.lfp_signal_low.setRange(0.01, 5000.0)
        self.lfp_signal_low.setValue(30.0)
        self.lfp_signal_high = QDoubleSpinBox()
        self.lfp_signal_high.setRange(0.02, 5000.0)
        self.lfp_signal_high.setValue(80.0)
        self.lfp_noise_low = QDoubleSpinBox()
        self.lfp_noise_low.setRange(0.01, 5000.0)
        self.lfp_noise_low.setValue(1.0)
        self.lfp_noise_high = QDoubleSpinBox()
        self.lfp_noise_high.setRange(0.02, 5000.0)
        self.lfp_noise_high.setValue(200.0)
        self.signal_band_low_var = self.lfp_signal_low
        self.signal_band_high_var = self.lfp_signal_high
        self.noise_band_low_var = self.lfp_noise_low
        self.noise_band_high_var = self.lfp_noise_high
        band_row = QHBoxLayout()
        for widget in (
            QLabel("信号"), self.lfp_signal_low, QLabel("–"), self.lfp_signal_high,
            QLabel("噪声"), self.lfp_noise_low, QLabel("–"), self.lfp_noise_high,
            QLabel("Hz"),
        ):
            band_row.addWidget(widget)
        band_holder = QWidget(); band_holder.setLayout(band_row)
        self.lfp_run_button = QPushButton("运行静息态多频段 SNR")
        self.lfp_run_button.setObjectName("primaryAction")
        self.lfp_run_button.clicked.connect(self.run_resting_multiband_snr)
        self.lfp_progress = QProgressBar()
        self.lfp_status = QLabel("使用当前数据源：直接分析预处理后的数据（LFP 页不再二次滤波）。")
        self.lfp_status_var = self.lfp_status
        self.lfp_progress_var = self.lfp_progress
        self.lfp_multiband_definition_label = QLabel(
            "静息态固定频段：Delta 1-4 ｜ Theta 4-8 ｜ Alpha 8-13 ｜ Beta 13-30 ｜ "
            "Low gamma 30-80 ｜ High gamma 80-200 Hz；50/100/150 Hz 工频保护带单独排除。"
        )
        self.lfp_multiband_definition_label.setWordWrap(True)
        self.lfp_multiband_definition_label.setStyleSheet("color:#24516e; padding:2px 0;")
        form.addRow(self.lfp_multiband_definition_label)
        form.addRow(self.lfp_run_button)
        actions = QGroupBox("第 1 步：确认数据并选择 LFP 通道")
        actions_layout = QVBoxLayout(actions)
        choose_channels = QPushButton("选择 LFP 通道（20×26）")
        choose_channels.clicked.connect(lambda: self.choose_stage_channels("lfp"))
        import_channels = QPushButton("单独导入 LFP H5")
        import_channels.setToolTip("选择一份完整LFP H5，或选择多份单通道LFP H5")
        import_channels.clicked.connect(lambda: self.import_stage_data("lfp"))
        psd = QPushButton("查看所选通道 PSD")
        psd.clicked.connect(lambda: self.plot_lfp_psd())
        params = QPushButton("显示 LFP 参数")
        self.lfp_params_button_var = params
        next_step = QPushButton("下一步：进入 Spike 分析")
        next_step.clicked.connect(lambda: self._activate_page(4))
        apply_parameters = QPushButton("应用参数并运行多频段 SNR"); apply_parameters.clicked.connect(self.run_resting_multiband_snr)
        batch_parameters = QPushButton("参数 CSV 批量处理"); batch_parameters.clicked.connect(lambda: self.run_parameter_csv_batch(False))
        task_quality = QPushButton("执行 trial 五窗质量判断")
        task_quality.setObjectName("primaryAction")
        task_quality.clicked.connect(self.run_lfp_trial_quality_check)
        task_browser = QPushButton("查看 trial 五窗波形")
        task_browser.clicked.connect(self.show_lfp_trial_quality_browser)
        task_letter = QPushButton("Letter 任务分析")
        task_letter.setObjectName("primaryAction")
        task_letter.clicked.connect(self.run_lfp_letter_frequency_analysis)
        task_vep = QPushButton("Trial-trial 叠加 VEP")
        task_vep.clicked.connect(self.run_trial_vep_overlay)
        export_vep = QPushButton("导出平均 VEP 数据")
        export_vep.clicked.connect(self.export_trial_vep)
        self.lfp_trial_vep_button = task_vep
        self.lfp_trial_vep_export_button = export_vep
        self.lfp_trial_quality_button = task_quality
        self.lfp_trial_quality_browser_button = task_browser
        self.lfp_letter_task_button = task_letter
        source_row = QHBoxLayout()
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.addWidget(QLabel("数据来源"))
        source_row.addWidget(import_channels)
        source_row.addWidget(choose_channels)
        source_row.addWidget(QLabel("未导入时承接当前预处理结果"))
        source_row.addStretch(1)
        self.lfp_channel_select_action = QComboBox()
        self.lfp_channel_select_action.addItems([
            "选择 SNR > 阈值", "选择全部有效 SNR 通道", "选择预处理健康通道",
            "选择预处理坏道", "反选当前通道", "清空选择",
        ])
        self.lfp_channel_select_threshold = QLineEdit("5")
        self.lfp_channel_select_threshold.setMaximumWidth(90)
        apply_lfp_channel_select = QPushButton("应用通道选择")
        apply_lfp_channel_select.clicked.connect(self.apply_lfp_channel_selection)
        channel_select_row = QHBoxLayout()
        channel_select_row.addWidget(QLabel("依据预处理结果选择"))
        channel_select_row.addWidget(self.lfp_channel_select_action)
        channel_select_row.addWidget(QLabel("SNR阈值 dB"))
        channel_select_row.addWidget(self.lfp_channel_select_threshold)
        channel_select_row.addWidget(apply_lfp_channel_select)
        channel_select_row.addStretch(1)
        task_workflow = QGroupBox("任务分析主流程")
        task_workflow_layout = QVBoxLayout(task_workflow)
        task_hint = QLabel("依赖顺序：时间对齐完成 → trial 五窗 QC → Letter 任务分析。后续 VEP、ITPC 和 baseline 比较均属于扩展分析。")
        task_hint.setWordWrap(True)
        task_hint.setStyleSheet("color:#24516e; padding:2px 0;")
        task_qc_row = QHBoxLayout()
        task_qc_row.addWidget(QLabel("① 质量判断")); task_qc_row.addWidget(task_quality); task_qc_row.addWidget(task_browser); task_qc_row.addStretch(1)
        task_analysis_row = QHBoxLayout()
        task_analysis_row.addWidget(QLabel("② 核心分析")); task_analysis_row.addWidget(task_letter); task_analysis_row.addStretch(1)
        task_extension_row = QHBoxLayout()
        task_extension_row.addWidget(QLabel("③ 扩展分析")); task_extension_row.addWidget(task_vep); task_extension_row.addWidget(export_vep); task_extension_row.addStretch(1)
        task_workflow_layout.addWidget(task_hint)
        task_workflow_layout.addLayout(task_qc_row)
        task_workflow_layout.addLayout(task_analysis_row)
        task_workflow_layout.addLayout(task_extension_row)

        source_row.insertWidget(max(0, source_row.count() - 1), params)
        source_row.insertWidget(max(0, source_row.count() - 1), batch_parameters)
        source_row.insertWidget(max(0, source_row.count() - 1), next_step)

        result_actions = QHBoxLayout()
        result_actions.addWidget(QLabel("辅助查看"))
        result_actions.addWidget(psd)
        result_actions.addStretch(1)
        result_actions_holder = QWidget(); result_actions_holder.setLayout(result_actions)

        actions_layout.insertLayout(0, source_row)
        actions_layout.insertLayout(1, channel_select_row)
        external_baseline = QGroupBox("实验前静息 baseline（按任务连续五窗长度配对）")
        external_baseline_layout = QGridLayout(external_baseline)
        self.external_baseline_file_label = QLabel("尚未导入实验前 baseline H5。")
        self.external_baseline_file_label.setWordWrap(True)
        self.external_baseline_file_label.setStyleSheet("color:#24516e; padding:2px 0;")
        self.external_baseline_confirm_check = QCheckBox("已确认与任务数据使用相同的上游预处理、重参考和单位")
        self.external_baseline_candidate_spin = QSpinBox()
        self.external_baseline_candidate_spin.setRange(1, 100)
        self.external_baseline_candidate_spin.setValue(12)
        self.external_baseline_min_valid_spin = QSpinBox()
        self.external_baseline_min_valid_spin.setRange(1, 100)
        self.external_baseline_min_valid_spin.setValue(5)
        self.external_baseline_seed_spin = QSpinBox()
        self.external_baseline_seed_spin.setRange(0, 2_147_483_647)
        self.external_baseline_seed_spin.setValue(20260817)
        self.external_time_response_start_spin = QDoubleSpinBox()
        self.external_time_response_start_spin.setRange(-60000.0, 60000.0)
        self.external_time_response_start_spin.setDecimals(1)
        self.external_time_response_start_spin.setValue(0.0)
        self.external_time_response_end_spin = QDoubleSpinBox()
        self.external_time_response_end_spin.setRange(-60000.0, 60000.0)
        self.external_time_response_end_spin.setDecimals(1)
        self.external_time_response_end_spin.setValue(5500.0)
        self.external_tf_freq_low_spin = QDoubleSpinBox()
        self.external_tf_freq_low_spin.setRange(.1, 5000.0)
        self.external_tf_freq_low_spin.setValue(15.0)
        self.external_tf_freq_high_spin = QDoubleSpinBox()
        self.external_tf_freq_high_spin.setRange(.1, 5000.0)
        self.external_tf_freq_high_spin.setValue(80.0)
        self.external_tf_freq_step_spin = QDoubleSpinBox()
        self.external_tf_freq_step_spin.setRange(.1, 1000.0)
        self.external_tf_freq_step_spin.setValue(2.0)
        self.external_tf_cycles_spin = QDoubleSpinBox()
        self.external_tf_cycles_spin.setRange(.5, 20.0)
        self.external_tf_cycles_spin.setValue(3.0)
        import_external_baseline = QPushButton("导入 baseline H5")
        import_external_baseline.clicked.connect(self.import_external_baseline_h5)
        prepare_external_baseline = QPushButton("准备 3/4/5 s baseline 参考")
        prepare_external_baseline.setObjectName("primaryAction")
        prepare_external_baseline.clicked.connect(self.prepare_external_baseline_reference)
        export_external_baseline = QPushButton("导出任务-baseline CSV")
        export_external_baseline.clicked.connect(self.export_lfp_task_baseline_csv)
        self.external_baseline_prepare_button = prepare_external_baseline
        external_baseline_layout.addWidget(self.external_baseline_file_label, 0, 0, 1, 8)
        external_baseline_layout.addWidget(import_external_baseline, 1, 0)
        external_baseline_layout.addWidget(self.external_baseline_confirm_check, 1, 1, 1, 3)
        external_baseline_layout.addWidget(QLabel("每长度候选"), 2, 0)
        external_baseline_layout.addWidget(self.external_baseline_candidate_spin, 2, 1)
        external_baseline_layout.addWidget(QLabel("最少合格"), 2, 2)
        external_baseline_layout.addWidget(self.external_baseline_min_valid_spin, 2, 3)
        external_baseline_layout.addWidget(QLabel("随机种子"), 2, 4)
        external_baseline_layout.addWidget(self.external_baseline_seed_spin, 2, 5)
        external_baseline_layout.addWidget(prepare_external_baseline, 2, 6)
        external_baseline_layout.addWidget(export_external_baseline, 2, 7)
        external_baseline_layout.addWidget(QLabel("时域响应窗 ms"), 3, 0)
        external_baseline_layout.addWidget(self.external_time_response_start_spin, 3, 1)
        external_baseline_layout.addWidget(QLabel("至"), 3, 2)
        external_baseline_layout.addWidget(self.external_time_response_end_spin, 3, 3)
        external_baseline_layout.addWidget(QLabel("基线窗沿用任务参数"), 3, 4, 1, 4)
        external_tf = QPushButton("计算任务-静息时频图")
        external_tf.setObjectName("primaryAction")
        external_tf.clicked.connect(self.run_external_time_frequency)
        self.external_time_frequency_button = external_tf
        external_baseline_layout.addWidget(QLabel("时频 Hz"), 4, 0)
        external_baseline_layout.addWidget(self.external_tf_freq_low_spin, 4, 1)
        external_baseline_layout.addWidget(QLabel("至"), 4, 2)
        external_baseline_layout.addWidget(self.external_tf_freq_high_spin, 4, 3)
        external_baseline_layout.addWidget(QLabel("步长"), 4, 4)
        external_baseline_layout.addWidget(self.external_tf_freq_step_spin, 4, 5)
        external_baseline_layout.addWidget(QLabel("周期"), 4, 6)
        external_baseline_layout.addWidget(self.external_tf_cycles_spin, 4, 7)
        external_baseline_layout.addWidget(external_tf, 5, 0, 1, 3)
        external_baseline_layout.addWidget(
            QLabel("每种长度从整段记录的不同分层随机位置抽取连续片段；只与任务中连续 3、4、5 个干净窗分别配对。"),
            5, 3, 1, 5,
        )
        self.external_baseline_qc_summary = QLabel("准备 baseline 参考后，将显示每个通道和长度的 QC 拒绝原因。")
        self.external_baseline_qc_summary.setWordWrap(True)
        self.external_baseline_qc_summary.setStyleSheet("color:#24516e; padding:2px 0;")
        self.external_baseline_qc_table = QTableWidget(0, 10)
        self.external_baseline_qc_table.setHorizontalHeaderLabels([
            "长度", "通道", "实际候选", "合格", "不可用", "饱和", "平坦", "跳变", "响应/基线缺失", "状态",
        ])
        self.external_baseline_qc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.external_baseline_qc_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.external_baseline_qc_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.external_baseline_qc_table.setMinimumHeight(150)
        self.external_baseline_qc_table.setMaximumHeight(240)
        self.external_baseline_qc_table.horizontalHeader().setStretchLastSection(True)
        external_baseline_layout.addWidget(self.external_baseline_qc_summary, 6, 0, 1, 8)
        external_baseline_layout.addWidget(self.external_baseline_qc_table, 7, 0, 1, 8)
        itpc = QGroupBox("ITPC（复用 trial 五窗质量结果）")
        itpc_grid = QGridLayout(itpc)
        self.itpc_tag_edit = QLineEdit()
        self.itpc_tag_edit.setPlaceholderText("tag；留空为全部")
        self.itpc_freq_low = QDoubleSpinBox(); self.itpc_freq_low.setRange(.1, 5000); self.itpc_freq_low.setValue(5.0)
        self.itpc_freq_high = QDoubleSpinBox(); self.itpc_freq_high.setRange(.1, 5000); self.itpc_freq_high.setValue(40.0)
        self.itpc_freq_step = QDoubleSpinBox(); self.itpc_freq_step.setRange(.1, 1000); self.itpc_freq_step.setValue(1.0)
        self.itpc_cycles = QDoubleSpinBox(); self.itpc_cycles.setRange(.5, 20); self.itpc_cycles.setValue(3.0)
        self.itpc_time_start = QDoubleSpinBox(); self.itpc_time_start.setRange(-60000, 60000); self.itpc_time_start.setValue(500.0)
        self.itpc_time_end = QDoubleSpinBox(); self.itpc_time_end.setRange(-60000, 60000); self.itpc_time_end.setValue(5500.0)
        self.itpc_baseline_start = QDoubleSpinBox(); self.itpc_baseline_start.setRange(-60000, 60000); self.itpc_baseline_start.setValue(500.0)
        self.itpc_baseline_end = QDoubleSpinBox(); self.itpc_baseline_end.setRange(-60000, 60000); self.itpc_baseline_end.setValue(1500.0)
        self.itpc_sample_rate = QDoubleSpinBox(); self.itpc_sample_rate.setRange(10, 5000); self.itpc_sample_rate.setValue(250.0)
        self.itpc_min_trials = QSpinBox(); self.itpc_min_trials.setRange(1, 10000); self.itpc_min_trials.setValue(10)
        run_itpc = QPushButton("计算 ITPC"); run_itpc.setObjectName("primaryAction"); run_itpc.clicked.connect(self.run_lfp_itpc)
        view_itpc = QPushButton("查看 ITPC 时频图"); view_itpc.clicked.connect(self.show_lfp_itpc_viewer)
        export_itpc = QPushButton("导出 ITPC NPZ"); export_itpc.clicked.connect(self.export_lfp_itpc)
        itpc_grid.addWidget(QLabel("Tag"), 0, 0); itpc_grid.addWidget(self.itpc_tag_edit, 0, 1)
        itpc_grid.addWidget(QLabel("频率 Hz"), 0, 2); itpc_grid.addWidget(self.itpc_freq_low, 0, 3)
        itpc_grid.addWidget(QLabel("至"), 0, 4); itpc_grid.addWidget(self.itpc_freq_high, 0, 5)
        itpc_grid.addWidget(QLabel("步长"), 0, 6); itpc_grid.addWidget(self.itpc_freq_step, 0, 7)
        itpc_grid.addWidget(QLabel("Cycles"), 0, 8); itpc_grid.addWidget(self.itpc_cycles, 0, 9)
        itpc_grid.addWidget(QLabel("时间 ms"), 1, 0); itpc_grid.addWidget(self.itpc_time_start, 1, 1)
        itpc_grid.addWidget(QLabel("至"), 1, 2); itpc_grid.addWidget(self.itpc_time_end, 1, 3)
        itpc_grid.addWidget(QLabel("计算采样率"), 1, 4); itpc_grid.addWidget(self.itpc_sample_rate, 1, 5)
        itpc_grid.addWidget(QLabel("最少 trial"), 1, 6); itpc_grid.addWidget(self.itpc_min_trials, 1, 7)
        itpc_grid.addWidget(run_itpc, 1, 8); itpc_grid.addWidget(view_itpc, 1, 9); itpc_grid.addWidget(export_itpc, 1, 10)
        itpc_grid.addWidget(QLabel("参考窗 ms"), 2, 0); itpc_grid.addWidget(self.itpc_baseline_start, 2, 1)
        itpc_grid.addWidget(QLabel("至"), 2, 2); itpc_grid.addWidget(self.itpc_baseline_end, 2, 3)
        itpc_grid.addWidget(QLabel("参考窗仅使用五窗 QC 覆盖的时间"), 2, 4, 1, 5)
        self.lfp_qc_available_ratio_var = QLineEdit("0.995")
        self.lfp_qc_saturation_run_var = QLineEdit("8")
        self.lfp_qc_flat_epsilon_var = QLineEdit("1e-4")
        self.lfp_qc_flat_ratio_var = QLineEdit("30")
        self.lfp_qc_flat_ptp_var = QLineEdit("0.01")
        self.lfp_qc_jump_mad_multiplier_var = QLineEdit("12")
        self.lfp_qc_jump_median_multiplier_var = QLineEdit("8")
        self.lfp_qc_jump_floor_multiplier_var = QLineEdit("10")
        self.lfp_parameter_panel = self._legacy_parameter_panel("第 2 步：设置分析参数（高级参数可折叠）", [
            "snr_mode_var", "snr_skip_var",
            "psd_channel_source_var", "psd_page_size_var", "psd_welch_sec_var", "psd_overlap_var", "psd_freq_low_var", "psd_freq_high_var", "psd_scale_var", "psd_mark_stim_var", "psd_stim_freq_var", "psd_stim_harmonics_var", "psd_mask_stim_var", "psd_bandwidth_var",
            "signal_band_low_var", "signal_band_high_var", "noise_band_low_var", "noise_band_high_var", "stim_interval_var", "stim_duration_var", "first_onset_var", "stim_freq_var", "task_target_freq_var", "harmonics_var", "neighbor_bins_var", "fft_len_var", "state_start_var", "state_duration_var", "epoch_start_ms_var", "epoch_end_ms_var", "baseline_start_ms_var", "baseline_end_ms_var", "response_start_ms_var", "response_end_ms_var", "trial_count_var", "n_stim_types_var", "trials_per_stim_var", "stimtag_var", "compare_trial_a_var", "compare_trial_b_var", "analysis_view_mode_var", "task_response_aggregate_var", "analysis_notch_var", "analysis_bandpass_var", "analysis_band_low_var", "analysis_band_high_var", "smooth_signal_var", "smooth_window_size_var", "zscoredata_var", "resting_page_size_var", "tvep_page_size_var", "auto_plot_marker_alignment_var"
        ])
        lfp_grid = self.lfp_parameter_panel.layout()
        lfp_items = []
        for index in range(lfp_grid.count()):
            item = lfp_grid.itemAt(index); row, column, rowspan, colspan = lfp_grid.getItemPosition(index)
            if item.widget() is not None:
                lfp_items.append((item.widget(), row, column, rowspan, colspan))
        for widget, row, column, rowspan, colspan in lfp_items:
            lfp_grid.addWidget(widget, row + 2, column, rowspan, colspan)
        lfp_grid.addWidget(QLabel("SNR 频段"), 0, 0)
        lfp_grid.addWidget(band_holder, 1, 0, 1, 5)
        qc_box = QGroupBox("trial 五窗 QC 阈值")
        qc_form = QGridLayout(qc_box)
        qc_form.setContentsMargins(7, 7, 7, 6)
        qc_form.setHorizontalSpacing(8)
        qc_form.setVerticalSpacing(4)
        qc_fields = (
            ("有限值比例 ≥", self.lfp_qc_available_ratio_var),
            ("饱和连续相同采样点 ≥", self.lfp_qc_saturation_run_var),
            ("平坦：相邻差值 ≤ mV", self.lfp_qc_flat_epsilon_var),
            ("平坦：小差值比例 ≥ %", self.lfp_qc_flat_ratio_var),
            ("平坦：窗口 PTP < mV", self.lfp_qc_flat_ptp_var),
            ("跳变：MAD 系数", self.lfp_qc_jump_mad_multiplier_var),
            ("跳变：中位差值系数", self.lfp_qc_jump_median_multiplier_var),
            ("跳变：平坦阈值系数", self.lfp_qc_jump_floor_multiplier_var),
        )
        for index, (label, widget) in enumerate(qc_fields):
            row, group = divmod(index, 4)
            widget.setMinimumWidth(70)
            widget.setMaximumWidth(105)
            qc_form.addWidget(QLabel(label), row, group * 2)
            qc_form.addWidget(widget, row, group * 2 + 1)
        qc_form.setColumnStretch(8, 1)
        qc_row = max(
            (lfp_grid.getItemPosition(index)[0] + lfp_grid.getItemPosition(index)[2]
             for index in range(lfp_grid.count())),
            default=0,
        )
        lfp_grid.addWidget(qc_box, qc_row, 0, 1, 5)
        self._make_collapsible(self.lfp_parameter_panel, collapsed=True)
        params.clicked.connect(lambda: self._toggle_parameter_section(self.lfp_parameter_panel))
        self.lfp_selected_summary_var = self.lfp_status
        self.lfp_task_status_var = self.lfp_status
        self.lfp_results = QPlainTextEdit()
        self.lfp_results.setReadOnly(True)
        self.lfp_results.setMaximumBlockCount(5000)
        self.lfp_multiband_panel = QGroupBox("静息态多频段 SNR 与通道质量")
        multiband_layout = QVBoxLayout(self.lfp_multiband_panel)
        self.lfp_multiband_summary = QLabel("点击“运行静息态多频段 SNR”后显示六频段结果。")
        self.lfp_multiband_summary.setWordWrap(True)
        multiband_layout.addWidget(self.lfp_multiband_summary)
        multiband_controls = QHBoxLayout()
        self.lfp_multiband_channel_box = QComboBox()
        self.lfp_multiband_metric_box = QComboBox()
        self.lfp_multiband_metric_box.addItems(["六频段 SNR", "工频突出度（dB）", "工频功率占比"])
        multiband_controls.addWidget(QLabel("通道")); multiband_controls.addWidget(self.lfp_multiband_channel_box)
        multiband_controls.addWidget(QLabel("图形")); multiband_controls.addWidget(self.lfp_multiband_metric_box)
        multiband_previous = QPushButton("上一通道")
        multiband_next = QPushButton("下一通道")
        multiband_refresh = QPushButton("刷新图表")
        multiband_export = QPushButton("导出静息态 CSV")
        multiband_previous.clicked.connect(lambda: self._step_multiband_channel(-1))
        multiband_next.clicked.connect(lambda: self._step_multiband_channel(1))
        multiband_refresh.clicked.connect(self._render_multiband_plot)
        multiband_export.clicked.connect(self.export_resting_multiband_csv)
        for button in (multiband_previous, multiband_next, multiband_refresh, multiband_export):
            multiband_controls.addWidget(button)
        multiband_controls.addStretch(1)
        multiband_layout.addLayout(multiband_controls)
        self.lfp_multiband_table = QTableWidget(0, 11)
        self.lfp_multiband_table.setHorizontalHeaderLabels([
            "通道", "Delta", "Theta", "Alpha", "Beta", "Low gamma", "High gamma",
            "50 Hz 峰值", "工频占比", "等级", "原因",
        ])
        self.lfp_multiband_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.lfp_multiband_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.lfp_multiband_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.lfp_multiband_table.setMaximumHeight(260)
        self.lfp_multiband_table.horizontalHeader().setStretchLastSection(True)
        self.lfp_multiband_table.cellClicked.connect(self._select_multiband_table_channel)
        multiband_layout.addWidget(self.lfp_multiband_table)
        self.lfp_line_table = QTableWidget(0, 10)
        self.lfp_line_table.setHorizontalHeaderLabels([
            "通道", "50 Hz", "100 Hz", "150 Hz", "工频峰值", "工频占比",
            "50 变化", "100 变化", "150 变化", "原始/当前",
        ])
        self.lfp_line_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.lfp_line_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.lfp_line_table.setMaximumHeight(190)
        self.lfp_line_table.horizontalHeader().setStretchLastSection(True)
        multiband_layout.addWidget(QLabel("工频污染指标（独立汇总）"))
        multiband_layout.addWidget(self.lfp_line_table)
        self.lfp_multiband_panel.setVisible(False)
        self.lfp_multiband_channel_box.currentIndexChanged.connect(self._render_multiband_plot)
        self.lfp_multiband_metric_box.currentIndexChanged.connect(self._render_multiband_plot)
        self.lfp_task_result_panel = QGroupBox("任务频率结果")
        task_result_layout = QVBoxLayout(self.lfp_task_result_panel)
        self.lfp_task_result_summary = QLabel("完成 Flash/Letter LFP 任务分析后，在此显示目标频率结果。")
        self.lfp_task_result_summary.setWordWrap(True)
        task_result_layout.addWidget(self.lfp_task_result_summary)
        task_result_controls = QHBoxLayout()
        self.lfp_task_result_tag_box = QComboBox()
        self.lfp_task_result_channel_box = QComboBox()
        self.lfp_task_result_metric_box = QComboBox()
        self.lfp_task_result_metric_box.addItem("local tag-SNR（dB）", "local_tag_snr_db")
        self.lfp_task_result_metric_box.addItem("事件锁定 LFP SNR（dB）", "event_lfp_snr_db")
        self.lfp_task_result_metric_box.addItem("相对刺激前 baseline 功率变化（dB）", "target_power_change_db")
        self.lfp_task_result_metric_box.addItem("相对实验前 baseline 功率变化（dB）", "external_baseline_target_power_change_db")
        self.lfp_task_result_metric_box.addItem("相对实验前 baseline local SNR 变化（dB）", "external_baseline_local_snr_change_db")
        self.lfp_task_result_metric_box.addItem("目标功率（mV²/Hz）", "target_power")
        self.lfp_task_result_metric_box.addItem("平均时域响应", "wave")
        self.lfp_task_result_metric_box.addItem("任务响应 vs 实验前静息（聚类检验）", "external_time_response")
        for label, widget in (
            ("Tag", self.lfp_task_result_tag_box),
            ("通道", self.lfp_task_result_channel_box),
            ("显示", self.lfp_task_result_metric_box),
        ):
            task_result_controls.addWidget(QLabel(label))
            task_result_controls.addWidget(widget)
        task_result_controls.addStretch(1)
        task_result_layout.addLayout(task_result_controls)
        self.lfp_task_result_table = QTableWidget(0, 8)
        self.lfp_task_result_table.setHorizontalHeaderLabels([
            "通道", "可用/总 trial", "有效 PSD（s）", "目标功率（mV²/Hz）",
            "刺激前 baseline（dB）", "实验前 baseline（dB）", "local tag-SNR（dB）", "状态 / 原因",
        ])
        self.lfp_task_result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.lfp_task_result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.lfp_task_result_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.lfp_task_result_table.setMinimumHeight(160)
        self.lfp_task_result_table.setMaximumHeight(250)
        self.lfp_task_result_table.horizontalHeader().setStretchLastSection(True)
        task_result_layout.addWidget(self.lfp_task_result_table)
        self.lfp_task_result_panel.setVisible(False)
        self.lfp_task_result_tag_box.currentIndexChanged.connect(self._render_lfp_task_result_view)
        self.lfp_task_result_channel_box.currentIndexChanged.connect(self._render_lfp_task_result_view)
        self.lfp_task_result_metric_box.currentIndexChanged.connect(self._render_lfp_task_result_view)
        self.lfp_task_result_table.cellClicked.connect(self._select_lfp_task_result_table_channel)
        self.lfp_preview_channel_box = QComboBox()
        self.lfp_preview_channel_box.setMinimumWidth(130)
        self.lfp_preview_controls = self._build_channel_preview_controls(
            self.lfp_preview_channel_box, self.refresh_lfp_channel_preview
        )
        self.lfp_figure = PyQtGraphPlot(page)
        self.lfp_canvas = self.lfp_figure
        self.lfp_toolbar = self.lfp_figure.toolbar

        # Present the page as one dependency-ordered workflow.  Resting-state
        # and task analyses are alternatives after the shared source/channel
        # and parameter steps, so they belong in separate tabs rather than in
        # one undifferentiated action row.
        layout.addWidget(actions)
        layout.addWidget(self.lfp_parameter_panel)

        analysis_tabs = QTabWidget()
        analysis_tabs.setObjectName("lfpAnalysisModeTabs")
        resting_page = QWidget(); resting_layout = QVBoxLayout(resting_page)
        resting_layout.setContentsMargins(6, 5, 6, 5); resting_layout.setSpacing(4)
        resting_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        resting_intro = QLabel("静息态流程：确认频段参数 → 运行多频段 SNR → 检查通道及工频结果。")
        resting_intro.setWordWrap(True); resting_intro.setStyleSheet("color:#24516e; padding:4px 0;")
        resting_layout.addWidget(resting_intro)
        resting_layout.addWidget(controls)
        resting_layout.addWidget(self.lfp_multiband_panel)

        task_page = QWidget(); task_layout = QVBoxLayout(task_page)
        task_layout.setContentsMargins(6, 5, 6, 5); task_layout.setSpacing(4)
        task_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        task_layout.addWidget(task_workflow)
        task_layout.addWidget(external_baseline)
        task_layout.addWidget(itpc)
        task_layout.addWidget(self.lfp_task_result_panel)
        resting_scroll = QScrollArea()
        resting_scroll.setFrameShape(QFrame.Shape.NoFrame)
        resting_scroll.setWidgetResizable(True)
        resting_scroll.setWidget(resting_page)
        task_scroll = QScrollArea()
        task_scroll.setFrameShape(QFrame.Shape.NoFrame)
        task_scroll.setWidgetResizable(True)
        task_scroll.setWidget(task_page)
        analysis_tabs.addTab(resting_scroll, "第 3 步 A：静息态分析")
        analysis_tabs.addTab(task_scroll, "第 3 步 B：Flash / Letter 任务分析")

        def resize_analysis_tabs(index: int) -> None:
            content = resting_page if index == 0 else task_page
            desired = content.sizeHint().height() + analysis_tabs.tabBar().sizeHint().height() + 18
            analysis_tabs.setFixedHeight(max(165, min(500, desired)))

        analysis_tabs.currentChanged.connect(resize_analysis_tabs)
        resize_analysis_tabs(0)
        layout.addWidget(analysis_tabs)
        self.lfp_analysis_mode_tabs = analysis_tabs

        run_status = QWidget()
        run_status_layout = QHBoxLayout(run_status)
        run_status_layout.setContentsMargins(5, 2, 5, 2)
        run_status_layout.setSpacing(7)
        run_status_layout.addWidget(QLabel("运行状态"))
        run_status_layout.addWidget(self.lfp_progress, 1)
        run_status_layout.addWidget(self.lfp_status, 2)
        layout.addWidget(run_status)

        results_box = QGroupBox("第 4 步：查看与导出结果")
        results_layout = QVBoxLayout(results_box)
        results_hint = QLabel("静息态六频段结果请在上方结果面板查看和导出；PSD 用于辅助检查所选通道的谱型。")
        results_hint.setWordWrap(True); results_hint.setStyleSheet("color:#24516e; padding:2px 0;")
        results_layout.addWidget(results_hint)
        results_layout.addWidget(result_actions_holder)
        result_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.lfp_results.setMinimumWidth(320)
        self.lfp_results.setMaximumWidth(560)
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(3)
        plot_layout.addWidget(self.lfp_preview_controls)
        plot_layout.addWidget(self.lfp_toolbar)
        plot_layout.addWidget(self.lfp_canvas, 1)
        result_splitter.addWidget(self.lfp_results)
        result_splitter.addWidget(plot_panel)
        result_splitter.setStretchFactor(0, 1)
        result_splitter.setStretchFactor(1, 3)
        result_splitter.setSizes([420, 1100])
        results_layout.addWidget(result_splitter, 1)
        self.lfp_result_splitter = result_splitter
        layout.addWidget(results_box, 1)
        return page

    def _build_alignment_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.alignment_banner = QLabel()
        self.alignment_banner.setObjectName("fileBanner")
        self.alignment_banner.setWordWrap(True)
        self.alignment_banner.setStyleSheet("background:#eef5fa; border:1px solid #b6cddd; padding:8px; color:#18384e;")
        self.alignment_file_banner_var = self.alignment_banner
        layout.addWidget(self.alignment_banner)
        controls = QGroupBox("实验时间对齐（Flash Event CSV）")
        form = QFormLayout(controls)
        self.alignment_log_edit = QLineEdit()
        self.alignment_event_edit = QLineEdit()
        self.alignment_detection_edit = QLineEdit()
        for name, label, callback in (
            ("alignment_log_edit", "log.txt", self.choose_alignment_log),
            ("alignment_event_edit", "Event CSV", self.choose_alignment_events),
            ("alignment_detection_edit", "视频检测结果（可选）", self.choose_alignment_detection),
        ):
            edit = getattr(self, name)
            row = QHBoxLayout(); row.addWidget(edit, 1)
            button = QPushButton("浏览…"); button.clicked.connect(callback); row.addWidget(button)
            holder = QWidget(); holder.setLayout(row)
            form.addRow(label, holder)
        self.alignment_task_mode_var = QComboBox()
        self.alignment_task_mode_var.addItems(["Flash", "Letter"])
        self.alignment_task_mode_var.setCurrentText("Letter")
        # Letter is the default task in this Qt workflow.  Keep its default
        # stimtag list aligned with the Letter marker definitions used by the
        # integrated pipeline; Flash retains its historical list when chosen.
        if isinstance(self.stimtag_var, QLineEdit) and self.stimtag_var.text().strip() in {"", "2,5,10,30"}:
            self.stimtag_var.setText("4,5,6,7")
        # User-facing task defaults: zero means use every complete trial.
        # Keep these assignments beside the task-mode defaults so the visible
        # controls and TaskEpochWorker have the same contract.
        for name in ("trials_per_stim_var", "compare_trial_a_var", "compare_trial_b_var"):
            widget = getattr(self, name, None)
            if isinstance(widget, QLineEdit):
                widget.setText("0")
        self._last_alignment_task_mode = "Letter"
        self.alignment_task_mode_var.currentTextChanged.connect(self._on_alignment_task_mode_changed)
        self.alignment_log_var = self.alignment_log_edit
        self.game_csv_var = self.alignment_event_edit
        self.detection_result_var = self.alignment_detection_edit
        self.align_run_button = QPushButton("计算并刷新对齐")
        self.align_run_button.setObjectName("primaryAction")
        self.align_run_button.clicked.connect(self.run_alignment)
        self.alignment_status = QLabel("加载 log.txt 与 Flash Event CSV 后生成 sample/tag 刺激标记。")
        self.alignment_status.setWordWrap(True)
        form.addRow("任务模式", self.alignment_task_mode_var)
        self.loaded_dt1_label = QLabel("文件 dt1：未加载；文件 deltaT1：未加载")
        self.loaded_dt1_label.setWordWrap(True)
        self.loaded_dt1_label.setStyleSheet("color:#24516e; padding:2px 0;")
        form.addRow("文件 BIN 定时", self.loaded_dt1_label)
        form.addRow(self.align_run_button)
        form.addRow(self.alignment_status)
        layout.addWidget(controls)
        action_row = QHBoxLayout()
        select = QPushButton("选择对齐 / Letter 通道（10×10 波形）")
        select.clicked.connect(self.choose_alignment_channels)
        custom = QPushButton("导入自定义通道数据")
        custom.clicked.connect(self.import_custom_channels)
        flash = QPushButton("Flash 任务")
        # Page 3 retains the legacy task plotting entry points.  They only
        # use aligned markers and never start LFP QC/PSD metrics.
        flash.clicked.connect(lambda: self.run_task_epoch_analysis("lfp", "Flash", lfp_task_metrics=False))
        letter = QPushButton("Letter 任务")
        # Page 3 keeps the original Letter video/marker browser.  It uses
        # only aligned epochs and does not run LFP QC/PSD metrics.
        letter.clicked.connect(self._run_alignment_letter_browser)
        for button in (select, custom, flash, letter): action_row.addWidget(button)
        next_alignment = QPushButton("下一步：检查时间对齐")
        next_alignment.clicked.connect(lambda: self._activate_page(3))
        action_row.addWidget(next_alignment)
        self.task_parameter_panel = self._legacy_parameter_panel("任务参数（与 integrated_pipeline_gui.py 一致）", [
                "epoch_start_ms_var", "epoch_end_ms_var",
                "baseline_start_ms_var", "baseline_end_ms_var",
                "response_start_ms_var", "response_end_ms_var",
                "n_stim_types_var", "trials_per_stim_var", "stimtag_var",
            "compare_trial_a_var", "compare_trial_b_var", "tvep_page_size_var",
            "task_response_aggregate_var", "smooth_signal_var", "smooth_window_size_var", "zscoredata_var",
        ])
        task_params = QPushButton("显示任务参数")
        task_params.clicked.connect(lambda: self._toggle_parameter_section(self.task_parameter_panel))
        action_row.addWidget(task_params)
        action_row.addStretch(1)
        action_holder = QWidget(); action_holder.setLayout(action_row); layout.addWidget(action_holder)
        layout.addWidget(self.task_parameter_panel)
        # Do not show the old GUI's unrelated placeholder fields (dt1_var,
        # start_frame_var, etc.) as editable parameters.  They are absent
        # from integrated_pipeline_gui.py; its real Letter/TXT alignment is
        # reported as read-only results and in the timing table.
        self.alignment_results = QPlainTextEdit(); self.alignment_results.setReadOnly(True)
        self.alignment_preview_channel_box = QComboBox()
        self.alignment_preview_channel_box.setMinimumWidth(130)
        self.alignment_preview_controls = self._build_channel_preview_controls(
            self.alignment_preview_channel_box, self.refresh_alignment_channel_preview
        )
        self.alignment_preview_start_spin = QDoubleSpinBox()
        self.alignment_preview_start_spin.setDecimals(3)
        self.alignment_preview_start_spin.setRange(0.0, 86_400.0)
        self.alignment_preview_start_spin.setValue(0.0)
        self.alignment_preview_start_spin.setSuffix(" s")
        self.alignment_preview_start_spin.setToolTip("预览窗口相对于当前已加载数据段的起点")
        self.alignment_preview_duration_spin = QDoubleSpinBox()
        self.alignment_preview_duration_spin.setDecimals(3)
        self.alignment_preview_duration_spin.setRange(0.001, 86_400.0)
        self.alignment_preview_duration_spin.setValue(5.0)
        self.alignment_preview_duration_spin.setSuffix(" s")
        self.alignment_preview_duration_spin.setToolTip("每次读取并显示的时间长度")
        self.alignment_preview_full_duration_check = QCheckBox("完整时长")
        self.alignment_preview_full_duration_check.setToolTip("显示当前加载数据段的完整时长；长数据会自动抽点绘制")
        preview_row = self.alignment_preview_controls.layout()
        insert_at = max(0, preview_row.count() - 1)
        for widget in (
            QLabel("起点"), self.alignment_preview_start_spin,
            QLabel("时长"), self.alignment_preview_duration_spin,
            self.alignment_preview_full_duration_check,
        ):
            preview_row.insertWidget(insert_at, widget)
            insert_at += 1
        self.alignment_preview_full_duration_check.toggled.connect(
            lambda checked: (
                self.alignment_preview_start_spin.setEnabled(not checked),
                self.alignment_preview_duration_spin.setEnabled(not checked),
            )
        )
        self.alignment_figure = PyQtGraphPlot(page)
        self.alignment_canvas = self.alignment_figure
        self.alignment_toolbar = self.alignment_figure.toolbar
        layout.addWidget(self.alignment_results)
        layout.addWidget(self.alignment_preview_controls)
        layout.addWidget(self.alignment_toolbar)
        layout.addWidget(self.alignment_canvas, 1)
        return page

    @staticmethod
    def _build_channel_preview_controls(channel_box, refresh_callback) -> QWidget:
        """Small, view-only channel switcher shared by the alignment/LFP plots."""
        row = QHBoxLayout()
        row.addWidget(QLabel("通道"))
        row.addWidget(channel_box)
        previous = QPushButton("上一通道")
        following = QPushButton("下一通道")
        refresh = QPushButton("刷新预览")

        def step(delta: int) -> None:
            if channel_box.count() <= 0:
                return
            channel_box.setCurrentIndex(
                max(0, min(channel_box.count() - 1, channel_box.currentIndex() + delta))
            )
            refresh_callback()

        previous.clicked.connect(lambda: step(-1))
        following.clicked.connect(lambda: step(1))
        refresh.clicked.connect(refresh_callback)
        row.addWidget(previous)
        row.addWidget(following)
        row.addWidget(refresh)
        row.addStretch(1)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    @staticmethod
    def _sync_preview_channel_box(channel_box, source) -> None:
        """Populate the view selector with physical IDs without changing data state."""
        meta = getattr(source, "metadata", None)
        if meta is None:
            return
        selected_id = channel_box.currentData()
        channel_box.blockSignals(True)
        channel_box.clear()
        for column, channel_id in enumerate(meta.channel_ids):
            channel_box.addItem(f"ch{int(channel_id)}", column)
        if selected_id is not None:
            previous = channel_box.findData(selected_id)
            if previous >= 0:
                channel_box.setCurrentIndex(previous)
        channel_box.blockSignals(False)

    def refresh_lfp_channel_preview(self) -> None:
        """Render PSD for the selected physical LFP channel; no analysis state changes."""
        source = self.lfp_source or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载 LFP 数据。")
            return
        self._sync_preview_channel_box(self.lfp_preview_channel_box, source)
        column = self.lfp_preview_channel_box.currentData()
        if column is None:
            return
        self.plot_lfp_psd(columns=[int(column)], preview_channel=True)

    def refresh_alignment_channel_preview(self) -> None:
        """Render one aligned channel with already computed marker lines."""
        source = self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载脑电数据。")
            return
        self._sync_preview_channel_box(self.alignment_preview_channel_box, source)
        column = self.alignment_preview_channel_box.currentData()
        if column is None:
            return
        meta = source.metadata
        column = int(column)
        total_seconds = max(0.001, float(meta.rows) / float(meta.fs))
        self.alignment_preview_start_spin.setMaximum(max(0.0, total_seconds - 0.001))
        self.alignment_preview_duration_spin.setMaximum(total_seconds)
        if self.alignment_preview_full_duration_check.isChecked():
            first, last = 0, int(meta.rows)
        else:
            start_seconds = min(float(self.alignment_preview_start_spin.value()), total_seconds)
            duration_seconds = min(float(self.alignment_preview_duration_spin.value()), total_seconds)
            first = min(int(meta.rows) - 1, max(0, int(round(start_seconds * meta.fs))))
            last = min(int(meta.rows), first + max(1, int(round(duration_seconds * meta.fs))))
        # Keep long/full-record previews responsive without changing the
        # selected time interval or any marker locations.
        display_step = max(1, int(np.ceil((last - first) / 20_000)))
        values = source.read(first, last, column, step=display_step).ravel()
        sample_indices = np.arange(first, last, display_step, dtype=np.int64)[:values.size]
        time_values = meta.time_offset + sample_indices / meta.fs
        markers = np.asarray(getattr(self, "stim_markers", np.empty((0, 2), dtype=int)), dtype=int)
        self.alignment_figure.clear()
        axis = self.alignment_figure.add_subplot(111)
        channel_id = int(meta.channel_ids[column])
        axis.plot(time_values, values, color="#1f77b4", linewidth=0.6, label=f"ch{channel_id}")
        marker_offset = int(round(float(meta.time_offset) * meta.fs))
        for sample, tag in markers:
            local_sample = int(sample) - marker_offset
            if first <= local_sample < last:
                axis.axvline(meta.time_offset + local_sample / meta.fs, color="#c62828", alpha=0.45, linewidth=0.8)
        axis.set_xlabel("时间（秒）")
        axis.set_ylabel("幅值（mV）")
        axis.set_title(
            f"时间对齐预览：ch{channel_id} 与刺激标记｜"
            f"{first / meta.fs:.3f}–{last / meta.fs:.3f} s"
        )
        axis.grid(alpha=0.25)
        self.alignment_canvas.draw_idle()

    def _on_alignment_task_mode_changed(self, mode: str) -> None:
        """Switch only untouched default stimtags when Flash/Letter changes."""
        widget = getattr(self, "stimtag_var", None)
        if not isinstance(widget, QLineEdit):
            return
        current = widget.text().strip()
        previous = getattr(self, "_last_alignment_task_mode", "Letter")
        defaults = {"Flash": "2,5,10,30", "Letter": "4,5,6,7"}
        if current in {defaults.get(previous, ""), ""}:
            widget.setText(defaults.get(mode, "4,5,6,7"))
        self._last_alignment_task_mode = str(mode)

    def _build_spike_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.spike_banner = QLabel()
        self.spike_banner.setObjectName("fileBanner")
        self.spike_banner.setWordWrap(True)
        self.spike_banner.setStyleSheet("background:#eef5fa; border:1px solid #b6cddd; padding:8px; color:#18384e;")
        self.spike_file_banner_var = self.spike_banner
        layout.addWidget(self.spike_banner)
        controls = QGroupBox("Spike 阈值检测")
        form = QFormLayout(controls)
        self.spike_highpass = QDoubleSpinBox()
        self.spike_highpass.setRange(1.0, 10000.0)
        self.spike_highpass.setValue(300.0)
        self.spike_lowpass = QDoubleSpinBox()
        self.spike_lowpass.setRange(2.0, 20000.0)
        self.spike_lowpass.setValue(3000.0)
        self.spike_threshold = QDoubleSpinBox()
        self.spike_threshold.setRange(0.1, 20.0)
        self.spike_threshold.setDecimals(2)
        self.spike_threshold.setValue(4.5)
        self.spike_refractory = QDoubleSpinBox()
        self.spike_refractory.setRange(0.01, 100.0)
        self.spike_refractory.setDecimals(2)
        self.spike_refractory.setValue(2.0)
        self.spike_low_var = self.spike_highpass
        self.spike_high_var = self.spike_lowpass
        self.spike_threshold_var = self.spike_threshold
        self.spike_refractory_var = self.spike_refractory
        param_row = QHBoxLayout()
        for widget in (
            QLabel("低截止"), self.spike_highpass, QLabel("高截止"), self.spike_lowpass, QLabel("Hz"),
            QLabel("阈值系数"), self.spike_threshold,
            QLabel("不应期"), self.spike_refractory, QLabel("ms"),
        ):
            param_row.addWidget(widget)
        parameter_holder = QWidget(); parameter_holder.setLayout(param_row)
        self.spike_run_button = QPushButton("运行 Spike 检测 / SNR")
        self.spike_run_button.setObjectName("primaryAction")
        self.spike_run_button.clicked.connect(self.run_spike)
        self.spike_progress = QProgressBar()
        self.spike_status = QLabel("使用当前数据源：预处理完成后自动承接处理后数据。")
        self.spike_status_var = self.spike_status
        form.addRow(self.spike_run_button)
        form.addRow("进度", self.spike_progress)
        form.addRow(self.spike_status)
        layout.addWidget(controls)
        actions = QGroupBox("Spike 通道、任务与导出")
        action_layout = QVBoxLayout(actions)
        choose_channels = QPushButton("选择 Spike 通道（20×26）")
        choose_channels.clicked.connect(lambda: self.choose_stage_channels("spike"))
        import_channels = QPushButton("单独导入 Spike H5")
        import_channels.setToolTip("选择一份完整Spike H5，或选择多份单通道Spike H5")
        import_channels.clicked.connect(lambda: self.import_stage_data("spike"))
        plot = QPushButton("绘制 Spike SNR")
        plot.clicked.connect(self.plot_spike_results)
        dynamic = QPushButton("10 秒动态 Spike SNR")
        dynamic.clicked.connect(self.plot_spike_dynamic)
        export = QPushButton("导出 Spike 结果")
        export.clicked.connect(self.export_spike_csv)
        flash = QPushButton("Flash 任务")
        flash.clicked.connect(lambda: self.run_task_epoch_analysis("spike", "Flash"))
        letter = QPushButton("Letter 任务")
        letter.clicked.connect(lambda: self.run_task_epoch_analysis("spike", "Letter"))
        self.spike_flash_task_button = flash
        self.spike_letter_task_button = letter
        params = QPushButton("显示 Spike 参数")
        self.spike_params_button_var = params
        next_step = QPushButton("下一步：进入结果导出")
        next_step.clicked.connect(lambda: self._activate_page(5))
        apply_detection = QPushButton("应用参数并检测"); apply_detection.clicked.connect(self.run_spike)
        batch_parameters = QPushButton("Spike 参数 CSV 批量处理"); batch_parameters.clicked.connect(lambda: self.run_parameter_csv_batch(True))
        channel_row = QHBoxLayout()
        for button in (choose_channels, import_channels, plot, dynamic, export):
            channel_row.addWidget(button)
        channel_row.addStretch(1)
        task_row = QHBoxLayout()
        for button in (flash, letter, batch_parameters, params):
            task_row.addWidget(button)
        task_row.addStretch(1)
        next_row = QHBoxLayout(); next_row.addWidget(apply_detection); next_row.addStretch(1); next_row.addWidget(next_step)
        action_layout.addLayout(channel_row)
        action_layout.addLayout(task_row)
        action_layout.addLayout(next_row)
        layout.addWidget(actions)
        self.spike_parameter_panel = self._legacy_parameter_panel("Spike 完整参数（旧版折叠区）", [
            "spike_low_var", "spike_high_var", "spike_threshold_var", "spike_noise_window_var", "spike_step_var", "spike_refractory_var", "spike_pre_samples_var", "spike_post_samples_var", "spike_min_count_var"
        ])
        spike_grid = self.spike_parameter_panel.layout()
        spike_items = []
        for index in range(spike_grid.count()):
            item = spike_grid.itemAt(index); row, column, rowspan, colspan = spike_grid.getItemPosition(index)
            if item.widget() is not None:
                spike_items.append((item.widget(), row, column, rowspan, colspan))
        for widget, row, column, rowspan, colspan in spike_items:
            spike_grid.addWidget(widget, row + 2, column, rowspan, colspan)
        spike_grid.addWidget(QLabel("Spike 频段与阈值"), 0, 0)
        spike_grid.addWidget(parameter_holder, 1, 0, 1, 3)
        self._make_collapsible(self.spike_parameter_panel, collapsed=True)
        params.clicked.connect(lambda: self._toggle_parameter_section(self.spike_parameter_panel))
        self.spike_task_status_var = self.spike_status
        self.spike_count_summary_var = self.spike_status
        self.spike_rate_summary_var = self.spike_status
        self.spike_unit_summary_var = self.spike_status
        self.spike_state_summary_var = self.spike_status
        layout.addWidget(self.spike_parameter_panel)
        self.spike_results = QPlainTextEdit()
        self.spike_results.setReadOnly(True)
        self.spike_results.setMaximumBlockCount(1000)
        self.spike_figure = PyQtGraphPlot(page)
        self.spike_canvas = self.spike_figure
        self.spike_toolbar = self.spike_figure.toolbar
        layout.addWidget(self.spike_results)
        layout.addWidget(self.spike_toolbar)
        layout.addWidget(self.spike_canvas, 1)
        return page

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_import_tab(), "数据导入")
        tabs.addTab(self._build_stage_notice("预处理"), "数据预处理")
        tabs.addTab(self._build_stage_notice("时间对齐"), "时间对齐")
        tabs.addTab(self._build_stage_notice("LFP 分析"), "LFP 分析")
        tabs.addTab(self._build_stage_notice("Spike 分析"), "Spike 分析")
        self.setCentralWidget(tabs)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Qt 控件框架已启动；所有绘图均使用 PyQtGraph。")

    def _build_import_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("importPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 16)
        layout.setSpacing(12)
        source_box = QGroupBox("HDF5 数据源")
        source_layout = QGridLayout(source_box)
        source_layout.setHorizontalSpacing(10)
        source_layout.setVerticalSpacing(12)
        source_layout.setColumnStretch(1, 1)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择 rawData512 / raw_data 等 HDF5 数据文件")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self.choose_file)
        load = QPushButton("加载元数据")
        load.setObjectName("primaryAction")
        load.clicked.connect(self.load_file)
        self.metadata_label = QLabel("尚未加载文件。")
        self.metadata_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.sample_rate_summary_var = QLabel("FS：—")
        self.channel_count_summary_var = QLabel("通道：—")
        self.duration_summary_var = QLabel("时长：—")
        self.channel_var = QLabel("预览通道：ch021")
        self.filter_var = QLabel("滤波：0.5 - 250 Hz")
        self.alignment_var = QLabel("对齐：首个 trial 对齐")
        source_summary = QHBoxLayout()
        source_summary.setContentsMargins(2, 5, 2, 2)
        source_summary.setSpacing(18)
        for summary in (self.sample_rate_summary_var, self.channel_count_summary_var, self.duration_summary_var, self.channel_var, self.filter_var, self.alignment_var):
            summary.setStyleSheet("color:#4a6175; font-size:13px;")
            source_summary.addWidget(summary)
        source_summary.addStretch(1)
        source_summary_holder = QWidget(); source_summary_holder.setLayout(source_summary)
        source_layout.addWidget(QLabel("文件"), 0, 0)
        source_layout.addWidget(self.path_edit, 0, 1)
        source_layout.addWidget(browse, 0, 2)
        source_layout.addWidget(load, 0, 3)
        source_layout.addWidget(self.metadata_label, 1, 0, 1, 4)
        source_layout.addWidget(source_summary_holder, 2, 0, 1, 4)
        layout.addWidget(source_box)
        legacy_box = QGroupBox("BIN / H5 数据截取与转换")
        legacy_form = QFormLayout(legacy_box)
        legacy_form.setHorizontalSpacing(14)
        legacy_form.setVerticalSpacing(12)
        legacy_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.bin_path_edit = QLineEdit()
        bin_row = QHBoxLayout(); bin_row.addWidget(self.bin_path_edit, 1)
        bin_choose = QPushButton("选择单个 BIN")
        bin_choose.clicked.connect(self.choose_bin_file)
        bin_row.addWidget(bin_choose)
        batch_choose = QPushButton("批量解析 BIN")
        batch_choose.clicked.connect(self.parse_batch_bins)
        bin_row.addWidget(batch_choose)
        holder = QWidget(); holder.setLayout(bin_row)
        self.output_dir_edit = QLineEdit()
        out_row = QHBoxLayout(); out_row.addWidget(self.output_dir_edit, 1)
        out_button = QPushButton("选择目录")
        out_button.clicked.connect(self.choose_output_directory)
        out_row.addWidget(out_button)
        out_holder = QWidget(); out_holder.setLayout(out_row)
        self.preview_full_duration_check = QCheckBox("全时长")
        self.preview_full_duration_check.setChecked(True)
        self.preview_channels_edit = QLineEdit("1,92,204,325")
        self.preview_start_edit = QLineEdit("0")
        self.preview_end_edit = QLineEdit("5")
        self.path_var = self.path_edit
        self.output_var = self.output_dir_edit
        self.range_var = self.preview_full_duration_check
        self.preview_channels_var = self.preview_channels_edit
        self.preview_start_var = self.preview_start_edit
        self.preview_end_var = self.preview_end_edit
        self.preview_duration_var = self.preview_end_edit
        legacy_form.addRow("BIN 文件", holder)
        legacy_form.addRow("输出目录", out_holder)
        self.import_progress = QProgressBar(); self.import_progress.setValue(0)
        self.import_status = QLabel("BIN 读取器：等待选择文件")
        self.import_progress_var = self.import_progress
        self.import_progress_status_var = self.import_status
        self.batch_status_var = self.import_status
        self.channel_folder_var = QLineEdit()
        self.import_params_button = QPushButton("显示参数")
        self.import_params_button_var = self.import_params_button
        self.next_preprocess_button = QPushButton("下一步：进入预处理")
        self.next_preprocess_button.clicked.connect(lambda: self._activate_page(1))
        legacy_form.addRow("进度", self.import_progress)
        legacy_form.addRow(self.import_status)
        save_h5 = QPushButton("解析/截取并保存总 H5")
        save_h5.setObjectName("primaryAction")
        save_h5.clicked.connect(self.parse_selected_bin)
        self.save_h5_button_var = save_h5
        primary_actions = QHBoxLayout()
        primary_actions.addWidget(save_h5)
        primary_actions.addWidget(self.import_params_button)
        primary_actions.addStretch(1)
        primary_actions.addWidget(self.next_preprocess_button)
        primary_actions_holder = QWidget(); primary_actions_holder.setLayout(primary_actions)
        legacy_form.addRow(primary_actions_holder)
        layout.addWidget(legacy_box)
        # These settings were in the old import page's button-driven hidden
        # parameter frame; keep the main page focused on selecting/reading a
        # file and expose them only after “显示参数”.
        self.import_parameter_panel = QGroupBox("数据导入参数")
        import_parameter_form = QFormLayout(self.import_parameter_panel)
        import_parameter_form.addRow("预览通道", self.preview_channels_edit)
        time_row = QHBoxLayout()
        time_row.addWidget(self.preview_full_duration_check)
        time_row.addWidget(QLabel("起点 s"))
        time_row.addWidget(self.preview_start_edit)
        time_row.addWidget(QLabel("终点 s"))
        time_row.addWidget(self.preview_end_edit)
        time_holder = QWidget(); time_holder.setLayout(time_row); import_parameter_form.addRow("时间窗", time_holder)
        self.preview_full_duration_check.toggled.connect(self._toggle_import_time_window)
        self._toggle_import_time_window(self.preview_full_duration_check.isChecked())
        import_parameter_form.addRow("日期", self.date_var)
        import_parameter_form.addRow("动物", self.animal_var)
        import_parameter_form.addRow("Block", self.block_var)
        import_parameter_form.addRow("通道文件夹", self.channel_folder_var)
        for label, widget in (("当前文件", self.active_file_var), ("文件摘要", self.active_file_summary_var), ("已解析 H5", self.parsed_h5_path_var)):
            if isinstance(widget, QLineEdit):
                widget.setReadOnly(True)
            import_parameter_form.addRow(label, widget)
        self._make_collapsible(self.import_parameter_panel, collapsed=True)
        self.import_params_button.clicked.connect(lambda: self._toggle_parameter_section(self.import_parameter_panel))
        layout.addWidget(self.import_parameter_panel)
        overview = QPushButton("全通道总览（PyQtGraph）")
        overview.clicked.connect(self.open_import_overview)
        overview_row = QHBoxLayout(); overview_row.addStretch(1); overview_row.addWidget(overview)
        overview_holder = QWidget(); overview_holder.setLayout(overview_row)
        layout.addWidget(overview_holder)
        return page

    def _toggle_import_time_window(self, full_duration: bool) -> None:
        """Disable start/end edits while the import page uses the full recording."""
        self.preview_start_edit.setEnabled(not full_duration)
        self.preview_end_edit.setEnabled(not full_duration)

    def _import_time_window(self) -> tuple[float, float]:
        """Return (start_seconds, duration_seconds) for BIN/HDF5 import APIs."""
        if self.preview_full_duration_check.isChecked():
            return 0.0, 0.0
        try:
            start = float(self.preview_start_edit.text() or "0")
            end = float(self.preview_end_edit.text() or "0")
        except ValueError as exc:
            raise ValueError("BIN/HDF5 时间窗起点和终点必须为非负数字。") from exc
        if start < 0.0 or end < 0.0:
            raise ValueError("BIN/HDF5 时间窗起点和终点必须为非负数字。")
        return start, _slice_duration_from_end(start, end)

    def choose_bin_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择单个 BIN", str(Path.cwd()), "BIN files (*.bin);;All files (*.*)")
        if filename:
            # The import page has one active input at a time. Selecting BIN
            # clears a previously selected H5 so the shared action is unambiguous.
            self.path_edit.clear()
            self.bin_path_edit.setText(filename)
            try:
                timing = read_timing_metadata(Path(filename))
                date = str(timing.get("dt1_date", ""))
                if isinstance(self.date_var, QLineEdit) and date:
                    self.date_var.setText(date)
                detail = "；".join(
                    f"{name}={timing[name]:.3f} ms" for name in ("deltaT1_ms", "deltaT2_ms") if name in timing
                )
                self.import_status.setText(f"已选择 BIN：{Path(filename).name}" + (f"；{detail}" if detail else ""))
            except Exception as exc:
                self.import_status.setText(f"已选择 BIN：{Path(filename).name}；定时信息读取失败：{exc}")

    def choose_bin_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择包含 BIN 的目录", str(Path.cwd()))
        if directory:
            self.output_dir_edit.setText(directory)
            files = list(Path(directory).glob("*.bin"))
            self.import_status.setText(f"已扫描 {len(files)} 个 BIN；Qt 会保留旧版批量解析入口，当前数据预览需加载生成的 H5。")

    def parse_batch_bins(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择包含 BIN 文件的根目录", str(Path.cwd()))
        if not directory:
            return
        output_root = Path(self.output_dir_edit.text().strip() or Path(directory) / "parsed_h5")
        if getattr(self, "_batch_bin_worker", None) is not None and self._batch_bin_worker.isRunning():
            return
        self.output_dir_edit.setText(str(output_root)); self.import_progress.setValue(0)
        self.import_status.setText("正在后台扫描并批量解析 BIN…")
        worker = BatchBinParseWorker(directory, str(output_root))
        worker.progress.connect(lambda value, text: (self.import_progress.setValue(int(round(value))), self.import_status.setText(text)))
        worker.completed.connect(self._finish_batch_bin_parse)
        worker.failed.connect(lambda error: self._fail_bin_parse(error))
        self._batch_bin_worker = worker; worker.start()

    def _finish_batch_bin_parse(self, successes, failures) -> None:
        self.import_progress.setValue(100)
        self.import_status.setText(f"批量 BIN 解析完成：成功 {len(successes)}，失败/跳过 {len(failures)}。")
        details = "\n".join(successes[:8] + failures[:8])
        QMessageBox.information(self, "批量 BIN 解析", self.import_status.text() + (f"\n\n{details}" if details else ""))

    def parse_selected_bin(self) -> None:
        h5_path = Path(self.path_edit.text().strip())
        if h5_path.is_file() and h5py.is_hdf5(h5_path):
            self._slice_selected_h5(h5_path)
            return
        path = self.bin_path_edit.text().strip()
        if not Path(path).is_file():
            QMessageBox.information(
                self, "未选择数据文件",
                "请先在上方加载有效的 H5，或者在 BIN 区域选择有效的 BIN 文件。",
            )
            return
        try:
            start, duration = self._import_time_window()
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        date_value = self.date_var.text().strip() if isinstance(self.date_var, QLineEdit) else ""
        animal_value = self.animal_var.text().strip() if isinstance(self.animal_var, QLineEdit) else ""
        output_root = Path(self.output_dir_edit.text().strip() or Path(path).parent / "parsed_h5")
        root = _single_import_output_directory(
            output_root, path, date_value=date_value, animal_value=animal_value,
        )
        output = root / f"{Path(path).stem}.h5"
        if getattr(self, "_bin_worker", None) is not None and self._bin_worker.isRunning():
            return
        metadata = {
            "date": date_value,
            "animal": int(animal_value or 0) if animal_value.isdigit() else 0,
            "block": int(self.block_var.text() or 0) if isinstance(self.block_var, QLineEdit) and self.block_var.text().strip().isdigit() else 0,
        }
        self.import_progress.setValue(0)
        self.import_status.setText(f"正在后台解析 BIN 并保存 Blosc/LZ4 H5… 输出：{output}")
        worker = BinParseWorker(path, str(output), start, duration, metadata)
        worker.progress.connect(lambda value, text: (self.import_progress.setValue(int(round(value))), self.import_status.setText(text)))
        worker.completed.connect(self._finish_bin_parse)
        worker.failed.connect(lambda error: self._fail_bin_parse(error))
        self._bin_worker = worker; worker.start()

    def _slice_selected_h5(self, source_path: Path) -> None:
        """Save the shared import-page time window as a new standalone H5."""
        try:
            start, duration = self._import_time_window()
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        if getattr(self, "_h5_slice_worker", None) is not None and self._h5_slice_worker.isRunning():
            return
        date_value = self.date_var.text().strip() if isinstance(self.date_var, QLineEdit) else ""
        animal_value = self.animal_var.text().strip() if isinstance(self.animal_var, QLineEdit) else ""
        output_root = Path(self.output_dir_edit.text().strip() or source_path.parent / "parsed_h5")
        root = _single_import_output_directory(
            output_root, source_path, date_value=date_value, animal_value=animal_value,
        )
        if duration <= 0:
            suffix = "full"
        else:
            end = start + duration
            suffix = f"slice_{start:g}_{end:g}s".replace(".", "p")
        output = root / f"{source_path.stem}_{suffix}.h5"
        sequence = 2
        while output.exists():
            output = root / f"{source_path.stem}_{suffix}_{sequence}.h5"
            sequence += 1
        self.import_progress.setValue(0)
        self.import_status.setText(
            f"正在后台截取 H5：{source_path.name}；"
            + ("全时长" if duration <= 0 else f"{start:g}–{start + duration:g} 秒")
            + f"；输出：{output}"
        )
        worker = H5SliceWorker(str(source_path), str(output), start, duration)
        worker.progress.connect(
            lambda value, text: (
                self.import_progress.setValue(int(round(value))), self.import_status.setText(text)
            )
        )
        worker.completed.connect(self._finish_h5_slice)
        worker.failed.connect(self._fail_h5_slice)
        self._h5_slice_worker = worker
        worker.start()

    def _finish_bin_parse(self, output: str, result) -> None:
        self.import_progress.setValue(100)
        self.path_edit.setText(output)
        self.parsed_h5_path_var.setText(output)
        self.import_status.setText(f"BIN 解析完成：{Path(output).name}；{result.get('seconds', 0):.1f} 秒。")
        self._load_file_use_full_range_once = True
        self.load_file()

    def _finish_h5_slice(self, output: str, result) -> None:
        self.import_progress.setValue(100)
        self.path_edit.setText(output)
        self.parsed_h5_path_var.setText(output)
        self.import_status.setText(
            f"H5 截取完成：{Path(output).name}；起点 {result.get('start_sec', 0):g} 秒，"
            f"时长 {result.get('seconds', 0):g} 秒。已作为后续预处理数据源加载。"
        )
        # The generated H5 already contains exactly the requested window; do
        # not apply the same start/end a second time while loading it.
        self._load_file_use_full_range_once = True
        self.load_file()

    def _fail_h5_slice(self, error: str) -> None:
        self.import_progress.setValue(0)
        self.import_status.setText(f"H5 截取失败：{error}")
        QMessageBox.critical(self, "H5 截取失败", error)

    def _fail_bin_parse(self, error: str) -> None:
        self.import_progress.setValue(0); self.import_status.setText(f"BIN 解析失败：{error}")
        QMessageBox.critical(self, "BIN 解析失败", error)

    def choose_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", str(Path.cwd()))
        if directory:
            self.output_dir_edit.setText(directory)

    def _show_parameter_summary(self, title: str, names: list[str]) -> None:
        lines = []
        for name in names:
            widget = getattr(self, name, None)
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, QLineEdit):
                value = widget.text()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            else:
                value = ""
            lines.append(f"{name}: {value}")
        QMessageBox.information(self, f"{title}参数", "\n".join(lines) or "当前没有可显示参数。")

    def _build_stage_notice(self, stage_name: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QFrame()
        box.setFrameShape(QFrame.Shape.StyledPanel)
        content = QVBoxLayout(box)
        label = QLabel(
            f"{stage_name}将沿用现有计算逻辑，正在从 Tk 控件层迁移到 Qt。\n"
            "当前 Qt 入口已完成 HDF5 懒加载与 PyQtGraph 预览；"
            "请暂时通过原 gui.py 使用尚未迁移的处理页面。"
        )
        label.setWordWrap(True)
        content.addWidget(label)
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def choose_alignment_log(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择实验 log.txt", str(Path.cwd()), "Text files (*.txt);;All files (*.*)")
        if filename:
            self.alignment_log_edit.setText(filename)

    def choose_alignment_events(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择 Flash Event CSV", str(Path.cwd()), "CSV files (*.csv);;All files (*.*)")
        if filename:
            self.alignment_event_edit.setText(filename)

    def choose_alignment_detection(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择视频检测结果", str(Path.cwd()),
            "Detection results (*.txt);;Text files (*.txt);;All files (*.*)",
        )
        if filename:
            self.alignment_detection_edit.setText(filename)

    def choose_alignment_channels(self) -> bool:
        if not self.active_source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载 HDF5 数据文件。")
            return False
        selector = getattr(self, "_alignment_channel_selector", None)
        if selector is not None:
            try:
                if selector.isVisible() and selector.source is self.active_source:
                    selector.raise_()
                    selector.activateWindow()
                    return False
                if selector.isVisible():
                    selector.close()
            except RuntimeError:
                pass
        selector = PyQtGraphOverview(
            self.active_source,
            0.0,
            min(5.0, self.active_source.metadata.rows / self.active_source.metadata.fs),
            selection_mode=True,
            selected_channel_ids=getattr(self, "alignment_selected_ids", set()),
            selection_callback=self._set_alignment_selected_channels,
            selection_label="对齐 / Letter",
        )
        selector.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        selector.showMaximized()
        self._alignment_channel_selector = selector
        return False

    def _set_alignment_selected_channels(self, channel_ids) -> None:
        self.alignment_selected_ids = {int(channel) for channel in channel_ids}
        self.alignment_status.setText(
            f"已选择 {len(self.alignment_selected_ids)} 个对齐 / Letter 通道；"
            "后续 Flash 和 Letter 任务将使用这组通道。"
        )
        self._update_file_banners()

    def _inherit_alignment_selection_for_lfp(self) -> None:
        """Seed page-4 selection once, without coupling later user choices."""
        if self._lfp_selection_explicit or self.lfp_source is not None:
            return
        source = self.active_source
        if source is None or not source.loaded:
            return
        available = {int(channel) for channel in source.metadata.channel_ids}
        inherited = set(getattr(self, "alignment_selected_ids", set())) & available
        if inherited == self.lfp_selected_ids:
            return
        self.lfp_selected_ids = inherited
        self._invalidate_lfp_channel_dependent_results()

    def run_alignment(self) -> None:
        if not self.active_source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载 HDF5 数据文件。")
            return
        if not getattr(self, "alignment_selected_ids", set()):
            QMessageBox.information(self, "未选择对齐通道", "请先点击“选择对齐通道”，在 10×10 波形窗口中勾选通道并确认。")
            return
        log_path = self.alignment_log_edit.text().strip()
        event_path = self.alignment_event_edit.text().strip()
        if not Path(log_path).is_file() or not Path(event_path).is_file():
            QMessageBox.warning(self, "缺少对齐文件", "请选择有效的 log.txt 和 Flash Event CSV。")
            return
        try:
            meta = self.active_source.metadata
            timing = parse_alignment_log(
                log_path, float(self.bin_timing_metadata.get("deltaT1_sec", 0.0) or 0.0)
            )
            task_mode = self.alignment_task_mode_var.currentText()
            markers = build_task_markers(event_path, timing, meta.fs, task_mode)
            self.timing_info = timing
            self.stim_markers = markers
            self.stim_marker_mode = task_mode
            self._task_marker_source = {
                "source_path": str(meta.path),
                "time_offset_sec": float(meta.time_offset),
                "fs": float(meta.fs),
                "selected_channel_ids": sorted(self.alignment_selected_ids),
            }
            # A new alignment replaces the previous task's state.  In
            # particular, Flash must never inherit a stale Letter TXT match.
            for name in (
                "behavior_matches", "behavior_detection_path",
                "behavior_detection_events", "behavior_detection_frame_rate",
                "behavior_detection_recording_time_sec", "behavior_detection_video_eeg_offset_sec",
            ):
                if hasattr(self, name):
                    delattr(self, name)
            valid = markers[(markers[:, 0] >= 0) & (markers[:, 0] < meta.rows)]
            behavior_matches = {}
            detection_path = self.alignment_detection_edit.text().strip()
            if task_mode == "Letter" and detection_path:
                frame_rate, recording_time, events = read_behavior_detection_results(detection_path)
                behavior_matches = match_behavior_events_to_markers(
                    markers, meta.fs, frame_rate, recording_time, np.nan, events
                )
                self.behavior_matches = behavior_matches
                self.behavior_detection_path = Path(detection_path)
                self.behavior_detection_events = events
                self.behavior_detection_frame_rate = frame_rate
                self.behavior_detection_recording_time_sec = recording_time
                if behavior_matches:
                    first = next(iter(behavior_matches.values()))
                    self.behavior_detection_video_eeg_offset_sec = float(first["video_eeg_offset_sec"])
                    # IntegratedPipelineGUI does not expose editable
                    # start-frame/error controller variables; it reports the
                    # values below and in the timing table instead.
            behavior_text = ""
            if task_mode == "Letter" and behavior_matches:
                first = next(iter(behavior_matches.values()))
                behavior_text = (
                    f"\n视频检测：帧率={frame_rate:g} Hz；匹配={len(behavior_matches)} 个\n"
                    f"首个 Letter trial 校准：marker sample={next(iter(behavior_matches))}，"
                    f"tag={first['tag']}；StartFrame={first['start_frame']}；"
                    f"视频→EEG 偏移={first['video_eeg_offset_sec']:+.6f} s；"
                    f"StartFrame 相对 marker={first['visual_start_latency_sec'] * 1000.0:+.3f} ms\n"
                    "说明：StartFrame 仅用于视觉对齐检查；有效 AnimalStart 仅作为行为参考。"
                )
            self.alignment_results.setPlainText(
                f"{task_mode} 对齐完成\nRecording ON: {timing['rec_start_text']}\n"
                f"PROGRAM_START: {timing['stim_start_text']}\n"
                f"原始延迟：{timing['stim_delay_raw_sec']:.6f} s\n"
                f"deltaT1：{timing['deltaT1_sec']:.6f} s\n"
                f"校正后刺激延迟：{timing['stim_delay_sec']:.6f} s\n"
                f"标记：{markers.shape[0]} 个；记录范围内：{valid.shape[0]} 个；"
                f"视频检测匹配：{len(behavior_matches)} 个\n"
                + behavior_text + "\n\n"
                + "前十个 [sample, tag]：\n" + np.array2string(markers[:10], separator=", ")
            )
            self._sync_preview_channel_box(self.alignment_preview_channel_box, self.active_source)
            ids = self.active_source.metadata.channel_ids
            preferred = next((index for index, channel_id in enumerate(ids) if channel_id in self.alignment_selected_ids), 0)
            self.alignment_preview_channel_box.setCurrentIndex(preferred)
            self.refresh_alignment_channel_preview()
            self.alignment_status.setText(f"对齐完成：{markers.shape[0]} 个刺激标记。")
            self._inherit_alignment_selection_for_lfp()
            self._update_file_banners()
        except Exception as exc:
            self.alignment_status.setText(f"时间对齐失败：{exc}")
            QMessageBox.critical(self, "时间对齐失败", str(exc))

    def _resting_multiband_source_and_columns(self):
        source = self.lfp_source or self.active_source
        if not source.loaded:
            raise RuntimeError("请先加载用于静息态 SNR 的数据。")
        meta = source.metadata
        # A page-4 LFP selection is authoritative whenever it exists.  Keep
        # the old all-channel default only when the user has not selected a
        # page-4 channel set yet.
        columns = self._page4_channel_columns(source)
        # Match the existing LFP SNR scope: a partial preprocessing run may
        # never silently mix filtered and unfiltered channels.
        if self.lfp_source is None:
            conditioned_ids = set(getattr(self, "_preprocess_filtered_channel_ids", set()))
            source_ids = np.asarray(meta.channel_ids, dtype=np.int64)
            if conditioned_ids and len(conditioned_ids) < source_ids.size:
                conditioned_columns = np.flatnonzero(np.isin(source_ids, sorted(conditioned_ids)))
                if columns is not None:
                    missing = np.setdiff1d(columns, conditioned_columns, assume_unique=True)
                    if missing.size:
                        missing_ids = source_ids[missing].tolist()
                        raise ValueError(f"已选择的 LFP 通道尚未完成同样的预处理：{missing_ids}。请先对这些通道滤波，或取消选择。")
                columns = conditioned_columns if columns is None else np.intersect1d(columns, conditioned_columns)
                if not columns.size:
                    raise ValueError("当前预处理结果没有可用于静息态 SNR 的已滤波通道。")
        return source, columns

    def run_resting_multiband_snr(self) -> None:
        try:
            source, columns = self._resting_multiband_source_and_columns()
            settings = {
                "target_frequency_resolution": .5,
                "line_frequency": 50.0,
                "line_harmonics": 3,
                "line_guard_hz": 1.0,
                "line_warning_db": 3.0,
                "line_reject_db": 10.0,
                "line_warning_fraction": .03,
                "line_reject_fraction": .10,
                "band_profile_outlier_z": 3.0,
                "workers": 20,
                "lowpass_high_hz": getattr(self, "_preprocess_lowpass_high_hz", None) if self.lfp_source is None else None,
            }
            raw_source = self.source if self.lfp_source is None and source is not self.source else None
            if getattr(self, "_resting_multiband_worker", None) is not None and self._resting_multiband_worker.isRunning():
                return
            self.lfp_run_button.setEnabled(False)
            self.lfp_progress.setValue(0)
            scope = "已滤波通道子集" if columns is not None else "全部同条件通道"
            self.lfp_status.setText(f"正在并行计算静息态六频段 SNR（{scope}；20 线程；工频保护带已排除）…")
            worker = RestingMultibandWorker(
                source, columns=columns, raw_source=raw_source, settings=settings,
                bad_ids=getattr(self, "bad_channel_ids", set()),
                candidate_ids=getattr(self, "bad_channel_candidate_ids", set()),
            )
            worker.progress.connect(lambda value, message: self._update_lfp_progress(value, message))
            worker.completed.connect(self._finish_resting_multiband_snr)
            worker.failed.connect(self._fail_resting_multiband_snr)
            worker.finished.connect(lambda: self.lfp_run_button.setEnabled(True))
            self._resting_multiband_worker = worker
            worker.start()
        except Exception as exc:
            QMessageBox.warning(self, "静息态多频段 SNR", str(exc))

    def run_lfp(self) -> None:
        """Compatibility entry point: the primary LFP SNR is now multi-band."""
        self.run_resting_multiband_snr()

    def _finish_resting_multiband_snr(self, output: dict) -> None:
        self.lfp_multiband_results = output
        rows = list(output.get("rows") or [])
        # Keep the legacy row contract alive: low-gamma is the former 30-80 Hz
        # LFP SNR and remains usable by existing channel-selection/export code.
        self.lfp_rows = []
        for row in rows:
            low_gamma = row["bands"].get("low_gamma", {})
            self.lfp_rows.append({
                "channel": int(row["channel"]),
                "snr_db": float(low_gamma.get("snr_db", np.nan)),
                "mode": "resting",
                "detail": "low_gamma 30-80 Hz; mains guard bands excluded",
                "channel_scope": row.get("channel_scope", "all_channels"),
                "quality_grade": row.get("quality_grade", "需复核"),
            })
        finite = [row for row in rows if np.isfinite(row["bands"].get("low_gamma", {}).get("snr_db", np.nan))]
        grades = {grade: sum(row.get("quality_grade") == grade for row in rows) for grade in ("可用", "需复核", "拒绝")}
        scope = rows[0].get("channel_scope", "all_channels") if rows else "all_channels"
        line_scope = output.get("line_source_scope", "post_only")
        resolution = float(rows[0].get("frequency_resolution_hz", np.nan)) if rows else np.nan
        self.lfp_results.setPlainText(
            "静息态多频段 SNR 计算完成\n"
            f"范围：{'已滤波通道子集' if scope == 'filtered_subset' else '全部同条件通道'}；通道数：{len(rows)}；"
            f"low-gamma 有效：{len(finite)}；Welch df：{resolution:.3f} Hz\n"
            f"质量分级：可用 {grades['可用']}，需复核 {grades['需复核']}，拒绝 {grades['拒绝']}；"
            f"工频数据：{'原始/当前配对' if line_scope == 'raw_current_paired' else '仅当前数据源残留'}\n\n"
            + "\n".join(
                f"ch{row['channel']}: "
                + ", ".join(f"{name}={row['bands'][name]['snr_db']:.3f} dB" if np.isfinite(row['bands'][name]['snr_db']) else f"{name}=—" for name, _, _ in RESTING_MULTIBAND_DEFINITIONS)
                + f"；等级={row.get('quality_grade', '需复核')}"
                for row in rows
            )
        )
        self._populate_multiband_tables(output)
        self.lfp_progress.setValue(100)
        self._save_lfp_analysis_state(
            "analysis_complete",
            (row.get("channel") for row in rows),
            "resting_multiband_snr",
        )
        self.lfp_status.setText(
            f"静息态多频段 SNR 计算完成（{int(output.get('workers', 1))} 线程）；"
            "原有任务 SNR 与 ITPC 缓存未改变。"
        )

    def _fail_resting_multiband_snr(self, message: str) -> None:
        self.lfp_progress.setValue(0)
        self.lfp_status.setText(f"静息态多频段 SNR 失败：{message}")
        QMessageBox.critical(self, "静息态多频段 SNR 失败", message)

    def _populate_multiband_tables(self, output: dict) -> None:
        rows = list(output.get("rows") or [])
        self.lfp_multiband_panel.setVisible(bool(rows))
        box = self.lfp_multiband_channel_box
        selected = box.currentData()
        box.blockSignals(True); box.clear()
        for row in rows:
            box.addItem(f"ch{int(row['channel'])}", int(row["channel"]))
        if selected is not None:
            index = box.findData(selected)
            if index >= 0:
                box.setCurrentIndex(index)
        box.blockSignals(False)
        table = self.lfp_multiband_table
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [f"ch{int(row['channel'])}"]
            values.extend(
                self._task_metric_number(row["bands"][name].get("snr_db", np.nan))
                for name, _, _ in RESTING_MULTIBAND_DEFINITIONS
            )
            values.extend([
                self._task_metric_number(row.get("post_line_max_db")),
                self._task_metric_number(row.get("post_line_power_fraction"), 4),
                str(row.get("quality_grade", "需复核")),
                "; ".join(row.get("quality_reasons") or []) or "—",
            ])
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, int(row["channel"]))
                table.setItem(row_index, column, item)
        table.resizeColumnsToContents()
        line_table = self.lfp_line_table
        line_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            line_values = [
                f"ch{int(row['channel'])}",
                self._task_metric_number(row.get("post_line_50_hz_db")),
                self._task_metric_number(row.get("post_line_100_hz_db")),
                self._task_metric_number(row.get("post_line_150_hz_db")),
                self._task_metric_number(row.get("post_line_max_db")),
                self._task_metric_number(row.get("post_line_power_fraction"), 4),
                self._task_metric_number(row.get("line_50_hz_suppression_db")),
                self._task_metric_number(row.get("line_100_hz_suppression_db")),
                self._task_metric_number(row.get("line_150_hz_suppression_db")),
                "原始/当前" if row.get("line_source_scope") == "raw_current_paired" else "仅当前",
            ]
            for column, text in enumerate(line_values):
                line_table.setItem(row_index, column, QTableWidgetItem(text))
        line_table.resizeColumnsToContents()
        self._render_multiband_plot()

    def _select_multiband_table_channel(self, row: int, _column: int) -> None:
        item = self.lfp_multiband_table.item(row, 0)
        if item is None:
            return
        index = self.lfp_multiband_channel_box.findData(item.data(Qt.ItemDataRole.UserRole))
        if index >= 0:
            self.lfp_multiband_channel_box.setCurrentIndex(index)

    def _step_multiband_channel(self, delta: int) -> None:
        box = self.lfp_multiband_channel_box
        if box.count():
            box.setCurrentIndex(max(0, min(box.count() - 1, box.currentIndex() + int(delta))))
            self._render_multiband_plot()

    def _render_multiband_plot(self, *_args) -> None:
        output = getattr(self, "lfp_multiband_results", None)
        if not output or not output.get("rows"):
            return
        channel = self.lfp_multiband_channel_box.currentData()
        row = next((item for item in output["rows"] if int(item["channel"]) == int(channel)), output["rows"][0])
        metric = self.lfp_multiband_metric_box.currentIndex()
        self.lfp_figure.clear(); axis = self.lfp_figure.add_subplot(111)
        if metric == 0:
            labels = [name for name, _, _ in RESTING_MULTIBAND_DEFINITIONS]
            values = np.asarray([row["bands"][name].get("snr_db", np.nan) for name in labels], dtype=float)
            valid = np.isfinite(values)
            if np.any(valid):
                axis.bar(np.flatnonzero(valid), values[valid], width=.65, color="#1f77b4")
            axis.axhline(0.0, color="#777777", linestyle="--", linewidth=.7)
            axis.set_xlabel("频段"); axis.set_ylabel("SNR（dB）")
            axis.set_title(f"ch{int(row['channel'])} 静息态六频段 SNR")
        else:
            centers = np.asarray([50.0, 100.0, 150.0])
            values = np.asarray([row.get(f"post_line_{int(center)}_hz_db", np.nan) for center in centers], dtype=float)
            if metric == 2:
                values = np.asarray([row.get(f"post_line_{int(center)}_hz_fraction", np.nan) for center in centers], dtype=float)
            valid = np.isfinite(values)
            if np.any(valid):
                axis.bar(centers[valid], values[valid], width=4.0, color="#c62828")
            axis.set_xlabel("工频及谐波（Hz）")
            axis.set_ylabel("突出度（dB）" if metric == 1 else "各谐波功率占比")
            axis.set_title(f"ch{int(row['channel'])} 工频污染指标")
        axis.grid(alpha=.25)
        self.lfp_canvas.draw_idle()

    def run_lfp_legacy(self) -> None:
        source = self.lfp_source or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载 HDF5 数据文件。")
            return
        signal_band = (self.lfp_signal_low.value(), self.lfp_signal_high.value())
        noise_band = (self.lfp_noise_low.value(), self.lfp_noise_high.value())
        fs = source.metadata.fs
        if not (0 < signal_band[0] < signal_band[1] < fs / 2):
            QMessageBox.warning(self, "频段无效", "信号频段必须位于 0 到奈奎斯特频率之间。")
            return
        if not (0 <= noise_band[0] < noise_band[1] < fs / 2):
            QMessageBox.warning(self, "频段无效", "噪声频段必须位于 0 到奈奎斯特频率之间。")
            return
        if getattr(self, "_lfp_worker", None) is not None and self._lfp_worker.isRunning():
            return
        columns = self._page4_channel_columns(source)
        # A partial page-2 filter leaves the other columns unchanged.  Do not
        # place those two conditioning states in one all-channel SNR result.
        if source is self.active_source and self.preprocessed_source is not None:
            conditioned_ids = set(getattr(self, "_preprocess_filtered_channel_ids", set()))
            source_ids = np.asarray(source.metadata.channel_ids, dtype=int)
            if conditioned_ids and len(conditioned_ids) < source_ids.size:
                conditioned_columns = np.flatnonzero(np.isin(source_ids, sorted(conditioned_ids)))
                if columns is not None:
                    missing = np.setdiff1d(columns, conditioned_columns, assume_unique=True)
                    if missing.size:
                        missing_ids = source_ids[missing].tolist()
                        QMessageBox.warning(self, "通道预处理不一致", f"已选择的 LFP 通道尚未完成同样的预处理：{missing_ids}。请先对这些通道滤波，或取消选择。")
                        return
                columns = conditioned_columns if columns is None else np.intersect1d(columns, conditioned_columns)
                if not columns.size:
                    QMessageBox.warning(self, "没有已处理通道", "当前预处理结果没有可用于一致 LFP SNR 比较的已滤波通道。")
                    return
        self.lfp_run_button.setEnabled(False)
        self.lfp_progress.setValue(0)
        scope = "已滤波通道" if columns is not None else "全部同条件通道"
        self.lfp_status.setText(f"正在后台计算 LFP SNR（{scope}；直接使用当前数据，不二次滤波）…")
        analysis_mode = self._widget_text(self.snr_mode_var, "resting").strip().lower()
        worker = LfpWorker(source, signal_band, noise_band, analysis_mode, columns)
        worker.progress.connect(lambda value, message: self._update_lfp_progress(value, message))
        worker.completed.connect(self._finish_lfp)
        worker.failed.connect(self._fail_lfp)
        worker.finished.connect(lambda: self.lfp_run_button.setEnabled(True))
        self._lfp_worker = worker
        worker.start()

    def _update_lfp_progress(self, value: float, message: str) -> None:
        self.lfp_progress.setValue(max(0, min(100, int(round(value)))))
        self.lfp_status.setText(message)

    def _finish_lfp(self, rows) -> None:
        self.lfp_rows = list(rows)
        finite = [row for row in rows if np.isfinite(row.get("snr_db", np.nan))]
        values = np.asarray([row["snr_db"] for row in finite], dtype=float)
        scope = "已滤波通道子集" if rows and rows[0].get("channel_scope") == "filtered_subset" else "全部同条件通道"
        self.lfp_results.setPlainText(
            "LFP SNR 计算完成\n"
            f"范围：{scope}；通道数：{len(rows)}；有效通道：{len(finite)}；"
            f"中位数：{np.nanmedian(values):.3f} dB；均值：{np.nanmean(values):.3f} dB\n\n"
            + "\n".join(f"ch{row['channel']}: {row['snr_db']:.3f} dB" for row in rows)
        )
        self.lfp_figure.clear()
        axis = self.lfp_figure.add_subplot(111)
        channels = [row["channel"] for row in rows]
        scores = [row["snr_db"] if np.isfinite(row["snr_db"]) else np.nan for row in rows]
        axis.plot(channels, scores, color="#1f77b4", linewidth=0.8, marker=".", markersize=3)
        axis.axhline(0.0, color="#777", linewidth=0.7, linestyle="--")
        axis.set_xlabel("通道")
        axis.set_ylabel("SNR（dB）")
        axis.set_title("全通道 LFP 静息态 SNR")
        axis.grid(alpha=0.25)
        self.lfp_canvas.draw_idle()
        self.lfp_progress.setValue(100)
        self._save_lfp_analysis_state(
            "analysis_complete",
            (row.get("channel") for row in rows),
            "legacy_lfp_snr",
        )
        self.lfp_status.setText("LFP SNR 计算完成。")

    def _fail_lfp(self, message: str) -> None:
        self.lfp_progress.setValue(0)
        self.lfp_status.setText(f"LFP SNR 失败：{message}")
        QMessageBox.critical(self, "LFP SNR 失败", message)

    def choose_stage_channels(self, stage: str) -> None:
        source = (self.lfp_source if stage == "lfp" else self.spike_source) or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载或导入该分析阶段的数据。")
            return
        selected = self.lfp_selected_ids if stage == "lfp" else self.spike_selected_ids
        previous_selection = set(selected)
        dialog = ChannelSelectionDialog(source.metadata.channel_ids, selected, self, f"选择 {stage.upper()} 通道（20×26）", self._layout_for_source(source))
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected.clear()
            selected.update(dialog.selected_channel_ids())
            if stage == "lfp":
                self._lfp_selection_explicit = True
            scope_changed = stage == "lfp" and selected != previous_selection
            if stage == "lfp" and selected != previous_selection:
                self._invalidate_lfp_channel_dependent_results()
            label = self.lfp_status if stage == "lfp" else self.spike_status
            label.setText(
                f"{stage.upper()} 已选择 {len(selected)} 个实际通道号。"
                + (" 通道范围已改变，请重新计算第四页结果。" if scope_changed else "")
            )
            if stage == "spike":
                self._save_spike_analysis_state("channels_selected")
            else:
                self._save_lfp_analysis_state("channels_selected")
            self._update_file_banners()

    def _lfp_analysis_state(self, status: str, completed_channel_ids=(), analysis_kind="") -> dict:
        return {
            "schema": "sd-lfp-analysis-state", "version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": str(status), "analysis_kind": str(analysis_kind),
            "selected_channel_ids": sorted(int(value) for value in self.lfp_selected_ids),
            "completed_channel_ids": sorted({int(value) for value in completed_channel_ids}),
            "parameters": {
                "signal_low_hz": float(self.lfp_signal_low.value()),
                "signal_high_hz": float(self.lfp_signal_high.value()),
                "noise_low_hz": float(self.lfp_noise_low.value()),
                "noise_high_hz": float(self.lfp_noise_high.value()),
                "channel_selection_rule": self.lfp_channel_select_action.currentText(),
                "channel_selection_threshold_db": self.lfp_channel_select_threshold.text().strip(),
                "multiband_definitions": [list(item) for item in RESTING_MULTIBAND_DEFINITIONS],
                "line_frequency_hz": 50.0, "line_harmonics": 3, "line_guard_hz": 1.0,
            },
            "channel_status": {
                str(int(channel)): (
                    "analysis_complete" if int(channel) in {int(v) for v in completed_channel_ids}
                    else "selected"
                ) for channel in self.lfp_selected_ids
            },
        }

    def _save_lfp_analysis_state(self, status: str, completed_channel_ids=(), analysis_kind="") -> bool:
        source = self.lfp_source or self.active_source
        meta = getattr(source, "metadata", None)
        path = Path(meta.path) if meta is not None else None
        if path is None or not path.is_file() or not h5py.is_hdf5(path):
            return False
        state = self._lfp_analysis_state(status, completed_channel_ids, analysis_kind)
        try:
            with h5py.File(path, "r+") as h5:
                h5.attrs["lfp_analysis_state_schema"] = state["schema"]
                h5.attrs["lfp_analysis_state_version"] = state["version"]
                h5.attrs["lfp_analysis_state_json"] = json.dumps(state, ensure_ascii=False, sort_keys=True)
            for manifest_path in path.parent.glob("*_project.json"):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if str(manifest.get("lfp_file", "")) != path.name:
                        continue
                    manifest["lfp_analysis_state"] = state
                    temporary = manifest_path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                    os.replace(temporary, manifest_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
            self._audit_event(
                "lfp_analysis_state", f"LFP 状态：{status}", data_path=str(path), state=state
            )
            return True
        except OSError:
            return False

    def _restore_lfp_analysis_state(self, path: Path, meta) -> bool:
        try:
            with h5py.File(path, "r") as h5:
                raw = h5.attrs.get("lfp_analysis_state_json", "")
            if isinstance(raw, bytes): raw = raw.decode("utf-8", errors="replace")
            state = json.loads(str(raw)) if raw else {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(state, dict) or state.get("schema") != "sd-lfp-analysis-state" or int(state.get("version", -1)) != 1:
            return False
        available = {int(value) for value in meta.channel_ids}
        self.lfp_selected_ids = {int(value) for value in state.get("selected_channel_ids", []) if int(value) in available}
        self._lfp_selection_explicit = bool(self.lfp_selected_ids)
        params = state.get("parameters", {}) if isinstance(state.get("parameters"), dict) else {}
        for widget, key in ((self.lfp_signal_low, "signal_low_hz"), (self.lfp_signal_high, "signal_high_hz"),
                            (self.lfp_noise_low, "noise_low_hz"), (self.lfp_noise_high, "noise_high_hz")):
            if key in params: widget.setValue(float(params[key]))
        if "channel_selection_threshold_db" in params:
            self.lfp_channel_select_threshold.setText(str(params["channel_selection_threshold_db"]))
        rule = str(params.get("channel_selection_rule", ""))
        index = self.lfp_channel_select_action.findText(rule)
        if index >= 0: self.lfp_channel_select_action.setCurrentIndex(index)
        completed = {int(value) for value in state.get("completed_channel_ids", []) if int(value) in available}
        self._loaded_lfp_analysis_state = state
        self.lfp_status.setText(
            f"已恢复 LFP 状态：选择 {len(self.lfp_selected_ids)} 个通道，已完成 {len(completed)} 个通道；"
            f"分析={state.get('analysis_kind') or '未记录'}；参数已同步。"
        )
        return True

    @staticmethod
    def _format_lfp_analysis_state(state: dict) -> str:
        """Build an auditable LFP state summary for the preprocessing data banner."""
        if not isinstance(state, dict):
            return ""
        selected = sorted({int(value) for value in state.get("selected_channel_ids", [])})
        completed = sorted({int(value) for value in state.get("completed_channel_ids", [])})
        params = state.get("parameters", {}) if isinstance(state.get("parameters"), dict) else {}

        def ids_text(values):
            if not values:
                return "无"
            ranges = []
            start = previous = values[0]
            for value in values[1:]:
                if value == previous + 1:
                    previous = value
                    continue
                ranges.append(str(start) if start == previous else f"{start}-{previous}")
                start = previous = value
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            return ",".join(ranges)

        names = {"resting_multiband_snr": "静息态六频段 SNR", "legacy_lfp_snr": "LFP SNR"}
        analysis_key = str(state.get("analysis_kind", ""))
        analysis = names.get(analysis_key, analysis_key or "尚未分析")
        return (
            f"LFP 状态：已选 ch{ids_text(selected)}（{len(selected)} 个）；"
            f"已完成 ch{ids_text(completed)}（{len(completed)} 个）；分析={analysis}；"
            f"信号带={params.get('signal_low_hz', '—')}–{params.get('signal_high_hz', '—')} Hz；"
            f"噪声带={params.get('noise_low_hz', '—')}–{params.get('noise_high_hz', '—')} Hz；"
            f"选道规则={params.get('channel_selection_rule') or '未记录'}；"
            f"阈值={params.get('channel_selection_threshold_db') or '—'} dB；"
            f"工频={params.get('line_frequency_hz', '—')} Hz，谐波={params.get('line_harmonics', '—')}，"
            f"保护带=±{params.get('line_guard_hz', '—')} Hz。"
        )

    def _spike_analysis_state(self, status: str, completed_channel_ids=()) -> dict:
        return {
            "schema": "sd-spike-analysis-state", "version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": str(status),
            "selected_channel_ids": sorted(int(value) for value in self.spike_selected_ids),
            "completed_channel_ids": sorted({int(value) for value in completed_channel_ids}),
            "parameters": {
                "highpass_hz": float(self.spike_highpass.value()),
                "lowpass_hz": float(self.spike_lowpass.value()),
                "threshold_factor": float(self.spike_threshold.value()),
                "refractory_ms": float(self.spike_refractory.value()),
            },
            "channel_status": {
                str(int(channel)): (
                    "analysis_complete" if int(channel) in {int(v) for v in completed_channel_ids}
                    else "selected"
                ) for channel in self.spike_selected_ids
            },
        }

    def _save_spike_analysis_state(self, status: str, completed_channel_ids=()) -> bool:
        source = self.spike_source or self.active_source
        meta = getattr(source, "metadata", None)
        path = Path(meta.path) if meta is not None else None
        if path is None or not path.is_file() or not h5py.is_hdf5(path):
            return False
        state = self._spike_analysis_state(status, completed_channel_ids)
        try:
            with h5py.File(path, "r+") as h5:
                h5.attrs["spike_analysis_state_schema"] = state["schema"]
                h5.attrs["spike_analysis_state_version"] = state["version"]
                h5.attrs["spike_analysis_state_json"] = json.dumps(
                    state, ensure_ascii=False, sort_keys=True
                )
            for manifest_path in path.parent.glob("*_project.json"):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if str(manifest.get("spike_file", "")) != path.name:
                        continue
                    manifest["spike_analysis_state"] = state
                    temporary = manifest_path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                    os.replace(temporary, manifest_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
            self._audit_event(
                "spike_analysis_state", f"Spike 状态：{status}", data_path=str(path), state=state
            )
            return True
        except OSError:
            return False

    def _restore_spike_analysis_state(self, path: Path, meta) -> bool:
        try:
            with h5py.File(path, "r") as h5:
                raw = h5.attrs.get("spike_analysis_state_json", "")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            state = json.loads(str(raw)) if raw else {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(state, dict) or state.get("schema") != "sd-spike-analysis-state" or int(state.get("version", -1)) != 1:
            return False
        available = {int(value) for value in meta.channel_ids}
        self.spike_selected_ids = {
            int(value) for value in state.get("selected_channel_ids", []) if int(value) in available
        }
        params = state.get("parameters", {}) if isinstance(state.get("parameters"), dict) else {}
        for widget, key in ((self.spike_highpass, "highpass_hz"), (self.spike_lowpass, "lowpass_hz"),
                            (self.spike_threshold, "threshold_factor"), (self.spike_refractory, "refractory_ms")):
            if key in params:
                widget.setValue(float(params[key]))
        self._loaded_spike_analysis_state = state
        completed = {
            int(value) for value in state.get("completed_channel_ids", []) if int(value) in available
        }
        self.spike_status.setText(
            f"已恢复 Spike 状态：选择 {len(self.spike_selected_ids)} 个通道，"
            f"已完成 {len(completed)} 个通道；参数已同步。"
        )
        return True

    def import_stage_data(self, stage: str) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self, f"单独导入 {stage.upper()} HDF5", str(Path.cwd()), "HDF5 files (*.h5 *.hdf5);;All files (*.*)"
        )
        if not filenames:
            return
        try:
            source = self._open_custom_h5_source(filenames); meta = source.metadata
            # Stage-specific data is allowed, but its BIN timing must still
            # become the active timing source used by the alignment formula.
            self._apply_loaded_timing(meta)
            if stage == "lfp":
                self.lfp_source = source
                restored_lfp_state = (
                    len(filenames) == 1
                    and self._restore_lfp_analysis_state(Path(filenames[0]), meta)
                )
                if not restored_lfp_state:
                    self.lfp_selected_ids.intersection_update(meta.channel_ids)
                # An independently imported LFP file must not inherit QC from
                # a previously loaded, unrelated preprocessing source.
                self._bad_channel_check_completed = False
                self._sync_dual_stream_export_enabled()
                self._bad_channel_result_channel_ids = ()
                self.good_channel_ids = set()
                self.bad_channel_ids = set()
                if len(filenames) == 1:
                    restored = self._restore_preprocess_qc_from_h5(Path(filenames[0]), meta)
                    if restored:
                        self._refresh_bad_channel_review_table()
                    self._show_loaded_preprocess_summary(meta, restored)
                self._invalidate_lfp_channel_dependent_results()
                if not restored_lfp_state:
                    self.lfp_status.setText(f"LFP 独立数据：{len(filenames)} 个文件，{meta.channels} 通道。")
                else:
                    self._set_preprocess_data_state(
                        "数据状态：已加载 LFP H5；已同步该文件中保存的通道选择、完成状态和分析参数。 "
                        + self._format_lfp_analysis_state(self._loaded_lfp_analysis_state)
                    )
            else:
                self.spike_source = source
                restored_spike_state = (
                    len(filenames) == 1
                    and self._restore_spike_analysis_state(Path(filenames[0]), meta)
                )
                if not restored_spike_state:
                    self.spike_selected_ids.intersection_update(meta.channel_ids)
                    self.spike_status.setText(f"Spike 独立数据：{len(filenames)} 个文件，{meta.channels} 通道。")
            self._update_file_banners()
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))

    def apply_lfp_channel_selection(self) -> None:
        """Select LFP channels from the current processed source and its QC."""
        source = self.lfp_source or self.active_source
        if source is None or not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先导入LFP H5或完成预处理。")
            return
        available = {int(channel) for channel in source.metadata.channel_ids}
        action = self.lfp_channel_select_action.currentText()
        previous = set(self.lfp_selected_ids)
        try:
            if action == "选择 SNR > 阈值":
                if not self.lfp_rows:
                    raise ValueError("请先运行LFP SNR，再按SNR阈值选择通道。")
                threshold = float(self.lfp_channel_select_threshold.text() or 5)
                selected = {
                    int(row["channel"]) for row in self.lfp_rows
                    if int(row["channel"]) in available
                    and np.isfinite(float(row.get("snr_db", np.nan)))
                    and float(row.get("snr_db", np.nan)) > threshold
                }
            elif action == "选择全部有效 SNR 通道":
                if not self.lfp_rows:
                    raise ValueError("请先运行LFP SNR，再选择全部有效SNR通道。")
                selected = {
                    int(row["channel"]) for row in self.lfp_rows
                    if int(row["channel"]) in available
                    and np.isfinite(float(row.get("snr_db", np.nan)))
                }
            elif action == "选择预处理健康通道":
                if not self._bad_channel_check_completed:
                    raise ValueError("当前LFP数据没有可恢复的已完成坏道判断，请先完成坏道检查。")
                selected = available & {int(channel) for channel in self.good_channel_ids}
            elif action == "选择预处理坏道":
                if not self._bad_channel_check_completed:
                    raise ValueError("当前LFP数据没有可恢复的已完成坏道判断，请先完成坏道检查。")
                selected = available & {int(channel) for channel in self.bad_channel_ids}
            elif action == "反选当前通道":
                selected = available - previous
            else:
                selected = set()
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "LFP通道选择", str(exc))
            return
        self._lfp_selection_explicit = True
        self.lfp_selected_ids = selected
        if selected != previous:
            self._invalidate_lfp_channel_dependent_results()
        self.lfp_status.setText(
            f"LFP 已按“{action}”选择 {len(selected)} 个通道。"
            + (" 通道范围已改变，请重新运行相关LFP分析。" if selected != previous else "")
        )
        self._save_lfp_analysis_state("channels_selected")
        self._update_file_banners()

    def _invalidate_lfp_channel_dependent_results(self) -> None:
        """Drop cached page-4 products when its channel scope changes."""
        self.lfp_rows = []
        self.lfp_multiband_results = None
        self.lfp_trial_quality_output = None
        self.lfp_itpc_output = None
        self.task_results = None
        panel = getattr(self, "lfp_multiband_panel", None)
        if panel is not None:
            panel.setVisible(False)

    def _page4_channel_columns(self, source, *, fallback_alignment: bool = False):
        """Return the page-4 channel scope, or None for the legacy all-channel default."""
        ids = np.asarray(source.metadata.channel_ids, dtype=np.int64)
        selected = set(getattr(self, "lfp_selected_ids", set()))
        if not selected and fallback_alignment:
            selected = set(getattr(self, "alignment_selected_ids", set()))
        if not selected:
            return None
        columns = np.flatnonzero(np.isin(ids, sorted(int(value) for value in selected)))
        if not columns.size:
            raise ValueError("已选择的 LFP 通道不在当前数据源中，请重新选择。")
        return columns

    def _page4_rows_for_selection(self, rows):
        """Limit cached page-4 rows to the current LFP selection."""
        selected = {int(value) for value in getattr(self, "lfp_selected_ids", set())}
        rows = list(rows or [])
        if not selected:
            return rows
        return [row for row in rows if int(row.get("channel", -1)) in selected]

    def plot_lfp_psd(self, columns=None, preview_channel: bool = False) -> None:
        source = self.lfp_source or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载数据。")
            return
        try:
            from scipy import signal
            meta = source.metadata; ids = list(meta.channel_ids)
            page4_selected = {
                int(channel) for channel in getattr(self, "lfp_selected_ids", set())
                if int(channel) in ids
            }
            if columns is not None:
                chosen = [int(column) for column in columns if 0 <= int(column) < meta.channels]
                if not chosen:
                    raise ValueError("所选预览通道不在当前 LFP 数据中。")
            else: chosen = [ids.index(channel) for channel in page4_selected] or [0]
            # Once the user has selected page-4 channels, even the legacy PSD
            # source selector cannot expand the calculation to all channels.
            if page4_selected:
                allowed = {ids.index(channel) for channel in page4_selected}
                chosen = [column for column in chosen if column in allowed]
                if not chosen:
                    raise ValueError("当前 PSD 预览通道不在已选择的 LFP 通道集合中。")
            elif columns is None:
                source_mode = self._widget_text(self.psd_channel_source_var, "selected channels").strip().lower()
                if source_mode == "all channels": chosen = list(range(meta.channels))
                elif source_mode == "healthy lfp channels" and hasattr(self, "good_channel_ids"):
                    chosen = [index for index, channel in enumerate(ids) if channel in self.good_channel_ids] or [0]
            welch_sec = max(.1, float(self.psd_welch_sec_var.text() or 2)); overlap_pct = min(95, max(0, float(self.psd_overlap_var.text() or 50)))
            freq_low = max(0, float(self.psd_freq_low_var.text() or 0)); freq_high = min(meta.fs/2, float(self.psd_freq_high_var.text() or 300))
            duration = min(meta.rows, max(1, int(round(max(10.0, welch_sec * 4) * meta.fs))))
            values = source.read(0, duration, chosen)
            if values.ndim == 1: values = values[:, None]
            nperseg=min(values.shape[0], max(8, int(round(welch_sec*meta.fs)))); noverlap=min(nperseg-1,int(round(nperseg*overlap_pct/100)))
            freqs, power = signal.welch(values, fs=meta.fs, axis=0, nperseg=nperseg, noverlap=noverlap)
            self.lfp_figure.clear(); axis = self.lfp_figure.add_subplot(111)
            average = np.nanmean(power, axis=1)
            scale = self._widget_text(self.psd_scale_var, "dB").strip().lower()
            y = 10*np.log10(np.maximum(average,np.finfo(float).tiny)) if scale == "db" else average
            if isinstance(self.psd_mask_stim_var,QCheckBox) and self.psd_mask_stim_var.isChecked():
                stim=float(self.psd_stim_freq_var.text() or 10); harmonics=int(self.psd_stim_harmonics_var.text() or 5); width=float(self.psd_bandwidth_var.text() or .25)
                for harmonic in range(1,harmonics+1): y[np.abs(freqs-stim*harmonic)<=width]=np.nan
            axis.plot(freqs, y, color="#1f77b4")
            if isinstance(self.psd_mark_stim_var,QCheckBox) and self.psd_mark_stim_var.isChecked():
                stim=float(self.psd_stim_freq_var.text() or 10); harmonics=int(self.psd_stim_harmonics_var.text() or 5)
                for harmonic in range(1,harmonics+1): axis.axvline(stim*harmonic,color="#c62828",alpha=.25,linewidth=.7)
            title = (
                f"LFP PSD（ch{int(ids[chosen[0]])}）" if preview_channel
                else f"LFP PSD（{len(chosen)} 通道均值）"
            )
            axis.set_xlim(freq_low, freq_high); axis.set_xlabel("频率（Hz）"); axis.set_ylabel("功率（dB）" if scale=="db" else "功率"); axis.set_title(title); axis.grid(alpha=.25)
            self.lfp_canvas.draw_idle()
        except Exception as exc:
            QMessageBox.critical(self, "PSD 失败", str(exc))

    @staticmethod
    def _write_records_csv(path: Path, records: list[dict]) -> None:
        import csv
        if not records:
            return
        fields = sorted({key for record in records for key in record})
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    def export_resting_multiband_csv(self) -> None:
        output = getattr(self, "lfp_multiband_results", None)
        rows = self._page4_rows_for_selection((output or {}).get("rows") or [])
        if not rows:
            QMessageBox.information(self, "尚无静息态结果", "请先运行静息态多频段 SNR。")
            return
        directory = QFileDialog.getExistingDirectory(self, "选择静息态结果输出目录", str(Path.cwd()))
        if not directory:
            return
        try:
            root = Path(directory)
            wide, long_records, line_records, quality = [], [], [], []
            for row in rows:
                common = {
                    "channel": int(row["channel"]),
                    "channel_scope": row.get("channel_scope", ""),
                    "total_power_1_200": row.get("total_power_1_200", np.nan),
                    "effective_total_bandwidth_hz": row.get("effective_total_bandwidth_hz", np.nan),
                    "nperseg": row.get("nperseg", ""),
                    "frequency_resolution_hz": row.get("frequency_resolution_hz", np.nan),
                    "quality_grade": row.get("quality_grade", ""),
                    "quality_reasons": "; ".join(row.get("quality_reasons") or []),
                }
                wide_row = dict(common)
                for name, low, high in RESTING_MULTIBAND_DEFINITIONS:
                    band = row["bands"][name]
                    wide_row.update({
                        f"{name}_low_hz": low, f"{name}_high_hz": high,
                        f"{name}_snr_db": band.get("snr_db", np.nan),
                        f"{name}_power": band.get("band_power", np.nan),
                        f"{name}_noise_power": band.get("noise_power", np.nan),
                        f"{name}_effective_bandwidth_hz": band.get("effective_bandwidth_hz", np.nan),
                        f"{name}_available": band.get("available", False),
                    })
                    long_records.append({
                        **common, "band": name, "low_hz": low, "high_hz": high,
                        "snr_db": band.get("snr_db", np.nan), "band_power": band.get("band_power", np.nan),
                        "noise_power": band.get("noise_power", np.nan),
                        "effective_bandwidth_hz": band.get("effective_bandwidth_hz", np.nan),
                        "available": band.get("available", False),
                    })
                wide.append(wide_row)
                line_row = {
                    "channel": int(row["channel"]),
                    "line_source_scope": row.get("line_source_scope", "post_only"),
                    "line_50_hz_db": row.get("post_line_50_hz_db", np.nan),
                    "line_100_hz_db": row.get("post_line_100_hz_db", np.nan),
                    "line_150_hz_db": row.get("post_line_150_hz_db", np.nan),
                    "line_max_db": row.get("post_line_max_db", np.nan),
                    "line_power_fraction": row.get("post_line_power_fraction", np.nan),
                    "line_50_hz_suppression_db": row.get("line_50_hz_suppression_db", np.nan),
                    "line_100_hz_suppression_db": row.get("line_100_hz_suppression_db", np.nan),
                    "line_150_hz_suppression_db": row.get("line_150_hz_suppression_db", np.nan),
                }
                line_records.append(line_row)
                quality.append({
                    "channel": int(row["channel"]), "quality_grade": row.get("quality_grade", ""),
                    "quality_reasons": "; ".join(row.get("quality_reasons") or []),
                    "line_max_db": row.get("post_line_max_db", np.nan),
                    "line_power_fraction": row.get("post_line_power_fraction", np.nan),
                })
            self._write_records_csv(root / "resting_multiband_snr_wide.csv", wide)
            self._write_records_csv(root / "resting_multiband_snr_long.csv", long_records)
            self._write_records_csv(root / "mains_contamination.csv", line_records)
            self._write_records_csv(root / "resting_channel_quality.csv", quality)
            self.lfp_status.setText(f"静息态多频段结果已导出：{root}")
        except Exception as exc:
            QMessageBox.critical(self, "静息态结果导出失败", str(exc))

    def run_parameter_csv_batch(self, include_spike: bool) -> None:
        source = (self.spike_source if include_spike else self.lfp_source) or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载数据后再运行参数 CSV 批处理。")
            return
        template = "batch_snr_params_with_spike_template.csv" if include_spike else "batch_snr_params_template.csv"
        filename, _ = QFileDialog.getOpenFileName(self, "选择参数 CSV", str(Path.cwd() / template), "CSV files (*.csv)")
        if not filename:
            return
        directory = QFileDialog.getExistingDirectory(self, "选择批处理结果目录", str(Path(filename).parent))
        if not directory:
            return
        try:
            import csv
            with open(filename, newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                raise ValueError("参数 CSV 没有数据行。")
            columns = None
            if not include_spike:
                columns = self._page4_channel_columns(source)
            worker = ParameterBatchWorker(source, rows, include_spike, columns=columns)
            worker.progress.connect(lambda value, text: self._update_parameter_batch_progress(include_spike, value, text))
            worker.completed.connect(lambda result, target=Path(directory), spike=include_spike: self._finish_parameter_csv_batch(result, target, spike))
            worker.failed.connect(lambda error, spike=include_spike: self._fail_parameter_csv_batch(error, spike))
            if include_spike: self._spike_batch_worker = worker
            else: self._lfp_batch_worker = worker
            self._update_parameter_batch_progress(include_spike, 0, "正在读取数据并执行参数 CSV 批处理…")
            worker.start()
        except Exception as exc:
            self._fail_parameter_csv_batch(str(exc), include_spike)

    def _update_parameter_batch_progress(self, include_spike: bool, value: float, message: str) -> None:
        progress = self.spike_progress if include_spike else self.lfp_progress
        status = self.spike_status if include_spike else self.lfp_status
        progress.setValue(max(0, min(100, int(round(value))))); status.setText(message)

    def _finish_parameter_csv_batch(self, results, directory: Path, include_spike: bool) -> None:
        import csv
        directory.mkdir(parents=True, exist_ok=True)
        for result in results:
            name = re.sub(r"[^\w.-]+", "_", str(result["name"]))
            for kind in ("lfp", "spike"):
                rows = result.get(kind)
                if not rows:
                    continue
                with open(directory / f"{name}_{kind}.csv", "w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
        self._update_parameter_batch_progress(include_spike, 100, f"参数 CSV 批处理完成：{len(results)} 组，结果目录：{directory}")

    def _fail_parameter_csv_batch(self, error: str, include_spike: bool) -> None:
        self._update_parameter_batch_progress(include_spike, 0, f"参数 CSV 批处理失败：{error}")
        QMessageBox.critical(self, "参数 CSV 批处理失败", error)

    def run_spike(self) -> None:
        source = self.spike_source or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载 HDF5 数据文件。")
            return
        fs = source.metadata.fs
        if not 0 < self.spike_highpass.value() < self.spike_lowpass.value() < fs / 2:
            QMessageBox.warning(self, "参数无效", "Spike 频段必须满足 0 < 低截止 < 高截止 < 奈奎斯特频率。")
            return
        if getattr(self, "_spike_worker", None) is not None and self._spike_worker.isRunning():
            return
        self.spike_run_button.setEnabled(False)
        self.spike_progress.setValue(0)
        self.spike_status.setText("正在后台检测 Spike…")
        source_ids = np.asarray(source.metadata.channel_ids, dtype=np.int64)
        selected_columns = (
            np.flatnonzero(np.isin(source_ids, sorted(self.spike_selected_ids)))
            if self.spike_selected_ids else np.arange(source.metadata.channels, dtype=np.int64)
        )
        worker = SpikeWorker(
            source,
            self.spike_highpass.value(),
            self.spike_lowpass.value(),
            self.spike_threshold.value(),
            self.spike_refractory.value(),
            columns=selected_columns,
        )
        worker.progress.connect(lambda value, message: self._update_spike_progress(value, message))
        worker.completed.connect(self._finish_spike)
        worker.failed.connect(self._fail_spike)
        worker.finished.connect(lambda: self.spike_run_button.setEnabled(True))
        self._spike_worker = worker
        worker.start()

    def _update_spike_progress(self, value: float, message: str) -> None:
        self.spike_progress.setValue(max(0, min(100, int(round(value)))))
        self.spike_status.setText(message)

    def _finish_spike(self, rows) -> None:
        self.spike_rows = list(rows)
        rates = np.asarray([row["rate_hz"] for row in rows], dtype=float)
        total_spikes = sum(row["spike_count"] for row in rows)
        self.spike_results.setPlainText(
            "Spike 阈值检测完成\n"
            f"通道数：{len(rows)}；总 Spike 数：{total_spikes:,}；"
            f"中位发放率：{np.nanmedian(rates):.3f} Hz\n\n"
            + "\n".join(
                f"ch{row['channel']}: {row['spike_count']} 个，{row['rate_hz']:.3f} Hz，"
                f"阈值 {row['threshold_mv']:.4g} mV"
                for row in rows
            )
        )
        self.spike_figure.clear()
        axis = self.spike_figure.add_subplot(111)
        axis.plot([row["channel"] for row in rows], rates, color="#c62828", linewidth=0.8, marker=".", markersize=3)
        axis.set_xlabel("通道")
        axis.set_ylabel("Spike 发放率（Hz）")
        axis.set_title("全通道 Spike 阈值检测")
        axis.grid(alpha=0.25)
        self.spike_canvas.draw_idle()
        self.spike_progress.setValue(100)
        self.spike_status.setText("Spike 检测完成。")
        self._save_spike_analysis_state(
            "analysis_complete", [int(row["channel"]) for row in self.spike_rows]
        )

    def _fail_spike(self, message: str) -> None:
        self.spike_progress.setValue(0)
        self.spike_status.setText(f"Spike 检测失败：{message}")
        QMessageBox.critical(self, "Spike 检测失败", message)

    def plot_spike_results(self) -> None:
        if not self.spike_rows:
            QMessageBox.information(self, "尚无结果", "请先运行 Spike 检测。")
            return
        self._finish_spike(self.spike_rows)

    def plot_spike_dynamic(self) -> None:
        source = self.spike_source or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载数据。")
            return
        worker = DynamicSpikeWorker(source, self.spike_highpass.value(), self.spike_lowpass.value(), self.spike_threshold.value(), self.spike_refractory.value())
        worker.progress.connect(lambda value, text: self._update_spike_progress(value, text))
        worker.completed.connect(self._finish_dynamic_spike)
        worker.failed.connect(self._fail_spike)
        self._dynamic_spike_worker = worker; worker.start()

    def _finish_dynamic_spike(self, seconds, counts) -> None:
        self.spike_figure.clear(); axis = self.spike_figure.add_subplot(111)
        axis.step(seconds, np.mean(counts, axis=1), where="post", color="#c62828")
        axis.set_xlabel("10 秒窗口起点（s）"); axis.set_ylabel("每通道平均 Spike 数")
        axis.set_title("真实 10 秒动态 Spike 检测"); axis.grid(alpha=.25); self.spike_canvas.draw_idle()
        self._update_spike_progress(100, f"动态 Spike 完成：{counts.shape[0]} 个 10 秒窗口。")

    def export_spike_csv(self) -> None:
        if not self.spike_rows:
            QMessageBox.information(self, "尚无结果", "请先运行 Spike 检测。")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "导出 Spike 结果 CSV", str(Path.cwd() / "spike_results.csv"), "CSV files (*.csv)")
        if not filename:
            return
        try:
            import csv
            with open(filename, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(self.spike_rows[0].keys()))
                writer.writeheader(); writer.writerows(self.spike_rows)
            self.spike_status.setText(f"已导出 Spike 结果：{filename}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _show_task_notice(self, task: str) -> None:
        marker_count = 0 if not hasattr(self, "stim_markers") else len(self.stim_markers)
        QMessageBox.information(self, f"{task} 分析", f"当前对齐标记：{marker_count} 个。任务窗参数沿用 LFP/Spike 页中已显示的旧版参数。")

    def _open_lfp_task_analysis(self, mode: str) -> None:
        """Carry aligned markers to page 4 without analysing an EEG epoch."""
        if not hasattr(self, "stim_markers"):
            QMessageBox.information(self, "尚未对齐", "请先在本页计算并刷新对齐，生成任务 marker。")
            return
        aligned_mode = getattr(self, "stim_marker_mode", None)
        if aligned_mode and aligned_mode != mode:
            QMessageBox.warning(
                self, "任务模式不一致",
                f"当前保存的是 {aligned_mode} markers。请先选择 {mode} 并重新计算时间对齐。",
            )
            return
        self.pending_lfp_task_mode = str(mode)
        self._activate_page(3)
        self.lfp_status.setText(
            f"已接收 {mode} 对齐 marker（{len(self.stim_markers)} 个）。请先点击“执行 trial 五窗质量判断”；Letter 对齐完成后可继续进行 Letter 任务分析。"
        )

    def _run_alignment_letter_browser(self) -> None:
        """Restore page-3 Letter plotting without LFP frequency metrics."""
        self.run_task_epoch_analysis("lfp", "Letter", lfp_task_metrics=False)

    def _lfp_quality_signature(self, mode: str, markers: np.ndarray, settings: dict) -> tuple:
        """Identify the marker/channel/threshold state that a QC cache belongs to."""
        source = self.lfp_source or self.active_source
        meta = source.metadata
        return (
            str(mode), tuple(np.asarray(markers, dtype=int).ravel().tolist()),
            int(meta.rows), int(meta.channels), float(meta.fs), float(meta.time_offset),
            tuple(int(value) for value in meta.channel_ids),
            tuple(np.asarray(settings["columns"], dtype=int).tolist()),
            float(settings["epoch_start"]), float(settings["epoch_end"]),
            int(settings["trials_per_stim"]),
            float(settings["quality_flat_epsilon"]),
            float(settings["quality_flat_ratio_percent"]),
            float(settings["quality_flat_ptp"]),
            float(settings["quality_available_ratio"]),
            int(settings["quality_saturation_run_samples"]),
            float(settings["quality_jump_mad_multiplier"]),
            float(settings["quality_jump_median_multiplier"]),
            float(settings["quality_jump_flat_floor_multiplier"]),
            bool(settings.get("analysis_notch")), bool(settings.get("analysis_bandpass")),
            float(settings.get("analysis_band_low", .5)), float(settings.get("analysis_band_high", 300)),
            bool(settings.get("smooth")), float(settings.get("smooth_window_sec", .02)), bool(settings.get("zscore")),
        )

    def _external_baseline_task_signature(self, source, channel_ids, settings: dict) -> tuple:
        """Fingerprint the task-side conditions required for a valid comparison."""
        meta = source.metadata
        return (
            str(meta.path), str(meta.dataset), int(meta.rows), float(meta.fs), float(meta.scale_to_mv),
            tuple(int(channel) for channel in channel_ids),
            float(settings["task_target_freq_hz"]), int(settings["task_neighbor_bins"]),
            bool(settings.get("analysis_notch")), bool(settings.get("analysis_bandpass")),
            float(settings.get("analysis_band_low", .5)), float(settings.get("analysis_band_high", 300.0)),
            bool(settings.get("smooth")), float(settings.get("smooth_window_sec", .02)), bool(settings.get("zscore")),
            float(settings["quality_flat_epsilon"]), float(settings["quality_flat_ratio_percent"]),
            float(settings["quality_flat_ptp"]), float(settings["quality_available_ratio"]),
            int(settings["quality_saturation_run_samples"]),
            float(settings["quality_jump_mad_multiplier"]), float(settings["quality_jump_median_multiplier"]),
            float(settings["quality_jump_flat_floor_multiplier"]),
            float(settings["epoch_start"]), float(settings["epoch_end"]),
            float(settings["baseline_start"]), float(settings["baseline_end"]),
            float(settings["response_start"]), float(settings["response_end"]),
            float(settings.get("external_time_response_start", 0.0)),
            float(settings.get("external_time_response_end", 5500.0)),
            float(settings.get("external_time_rate_hz", 250.0)),
        )

    def _external_baseline_analysis_settings(self) -> dict:
        """Read the task settings that must be reproduced on external baseline data."""
        get = lambda name, default: float(self._widget_text(getattr(self, name), str(default)) or default)
        settings = {
            "analysis_notch": isinstance(self.analysis_notch_var, QCheckBox) and self.analysis_notch_var.isChecked(),
            "analysis_bandpass": isinstance(self.analysis_bandpass_var, QCheckBox) and self.analysis_bandpass_var.isChecked(),
            "analysis_band_low": get("analysis_band_low_var", .5),
            "analysis_band_high": get("analysis_band_high_var", 300),
            "smooth": isinstance(self.smooth_signal_var, QCheckBox) and self.smooth_signal_var.isChecked(),
            "smooth_window_sec": get("smooth_window_size_var", .02),
            "zscore": isinstance(self.zscoredata_var, QCheckBox) and self.zscoredata_var.isChecked(),
            "task_target_freq_hz": get("task_target_freq_var", 1.0),
            "task_neighbor_bins": int(get("neighbor_bins_var", 4)),
            "epoch_start": get("epoch_start_ms_var", -500.0),
            "epoch_end": get("epoch_end_ms_var", 5500.0),
            "baseline_start": get("baseline_start_ms_var", 500.0),
            "baseline_end": get("baseline_end_ms_var", 1500.0),
            "response_start": get("response_start_ms_var", 1500.0),
            "response_end": get("response_end_ms_var", 5500.0),
            "external_time_response_start": self.external_time_response_start_spin.value(),
            "external_time_response_end": self.external_time_response_end_spin.value(),
            "external_time_rate_hz": 250.0,
            **self._read_lfp_trial_quality_settings(),
        }
        if settings["zscore"]:
            raise ValueError("启用逐记录 z-score 时，任务与独立 baseline 的幅值标尺不可严格配对；请关闭 z-score 后准备 baseline。")
        if not (
            settings["epoch_start"] < settings["baseline_start"] <= settings["baseline_end"] <= settings["external_time_response_start"] < settings["external_time_response_end"] <= settings["epoch_end"]
        ):
            raise ValueError("严格 baseline 配对要求 epoch、刺激前基线和响应时间窗严格递增且均在 epoch 内。")
        return settings

    def import_external_baseline_h5(self) -> None:
        """Load a separate pre-task resting H5 without changing any LFP source."""
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "导入实验前 baseline HDF5", str(Path.cwd()), "HDF5 files (*.h5 *.hdf5);;All files (*.*)",
        )
        if not filenames:
            return
        try:
            source = self._open_custom_h5_source(filenames)
            self.external_baseline_source = source
            self.external_baseline_reference = None
            meta = source.metadata
            self.external_baseline_file_label.setText(
                f"已导入：{Path(meta.path).name} | {meta.rows / meta.fs:.1f} s | {meta.channels} 通道 | {meta.fs:g} Hz；尚未准备参考。"
            )
        except Exception as exc:
            QMessageBox.critical(self, "导入 baseline 失败", str(exc))

    def prepare_external_baseline_reference(self) -> bool:
        """Prepare independently QC'd 3/4/5 s references for page-4 task data."""
        baseline = self.external_baseline_source
        task_source = self.lfp_source or self.active_source
        if baseline is None or not baseline.loaded:
            QMessageBox.information(self, "尚未导入 baseline", "请先导入实验前静息 baseline H5。")
            return False
        if not task_source.loaded:
            QMessageBox.information(self, "尚未加载任务数据", "请先加载用于 LFP 任务分析的数据。")
            return False
        if not self.external_baseline_confirm_check.isChecked():
            QMessageBox.warning(self, "需要确认预处理", "请确认 baseline 与任务数据的上游预处理、重参考和单位一致。")
            return False
        try:
            settings = self._external_baseline_analysis_settings()
            if not 0 < settings["task_target_freq_hz"] < task_source.metadata.fs / 2:
                raise ValueError("任务目标频率必须介于 0 和任务数据奈奎斯特频率之间。")
            if not np.isclose(float(baseline.metadata.fs), float(task_source.metadata.fs), rtol=0, atol=1e-9):
                raise ValueError(
                    f"严格配对要求采样率一致：任务 {task_source.metadata.fs:g} Hz，baseline {baseline.metadata.fs:g} Hz。"
                )
            if self.external_baseline_min_valid_spin.value() > self.external_baseline_candidate_spin.value():
                raise ValueError("最少合格片段数不能大于每长度候选片段数。")
            selected_ids = set(getattr(self, "lfp_selected_ids", set())) or set(getattr(self, "alignment_selected_ids", set()))
            task_columns = np.flatnonzero(np.isin(task_source.metadata.channel_ids, sorted(selected_ids)))
            if not task_columns.size:
                raise ValueError("请先选择至少一个用于 LFP 任务分析的通道。")
            task_channel_ids = [int(task_source.metadata.channel_ids[index]) for index in task_columns]
            baseline_lookup = {int(channel): index for index, channel in enumerate(baseline.metadata.channel_ids)}
            baseline_columns = np.asarray(
                [baseline_lookup[channel] for channel in task_channel_ids if channel in baseline_lookup], dtype=np.int64,
            )
            if not baseline_columns.size:
                raise ValueError("baseline H5 没有任何与当前 LFP 任务通道 ID 匹配的通道。")
            settings.update({
                "baseline_candidate_count": self.external_baseline_candidate_spin.value(),
                "baseline_min_valid_segments": self.external_baseline_min_valid_spin.value(),
                "baseline_random_seed": self.external_baseline_seed_spin.value(),
            })
            signature = self._external_baseline_task_signature(task_source, task_channel_ids, settings)
        except Exception as exc:
            QMessageBox.warning(self, "baseline 参数无效", str(exc))
            return False
        worker = ExternalBaselineWorker(baseline, baseline_columns, task_channel_ids, settings)
        worker.progress.connect(lambda value, text: self._update_lfp_progress(value, text))
        worker.completed.connect(
            lambda reference, task_signature=signature: self._finish_external_baseline_reference(reference, task_signature)
        )
        worker.failed.connect(lambda error: self._fail_external_baseline_reference(error))
        self._external_baseline_worker = worker
        self.external_baseline_prepare_button.setEnabled(False)
        self._update_lfp_progress(0, "正在准备实验前 baseline 的 3/4/5 s 参考…")
        worker.start()
        return True

    def _finish_external_baseline_reference(self, reference: dict, task_signature: tuple) -> None:
        reference["task_signature"] = task_signature
        self.external_baseline_reference = reference
        self.external_baseline_prepare_button.setEnabled(True)
        counts = []
        for length, item in reference["references"].items():
            eligible = int(np.count_nonzero(item["eligible"]))
            actual_candidates = len(item["candidate_starts"])
            counts.append(f"{length}s: {eligible}/{len(reference['channel_ids'])} 通道达标（实际 {actual_candidates} 段）")
        self.external_baseline_file_label.setText(
            f"已准备：{Path(reference['path']).name} | 分层随机候选 {reference['candidate_count']} 段/长度 | "
            + "；".join(counts)
        )
        self._render_external_baseline_qc_details(reference)
        self._update_lfp_progress(100, "实验前 baseline 参考已准备；运行 Letter 任务分析后将自动按连续 3/4/5 s 配对。")
        if getattr(self, "_resume_external_time_frequency_after_baseline", False):
            self._resume_external_time_frequency_after_baseline = False
            QTimer.singleShot(0, self.run_external_time_frequency)

    def _render_external_baseline_qc_details(self, reference: dict) -> None:
        """Show the per-length, per-channel QC evidence behind baseline eligibility."""
        table = self.external_baseline_qc_table
        rows = []
        minimum_valid = int(reference["minimum_valid_segments"])
        for length, item in sorted(reference["references"].items()):
            actual_candidates = len(item["candidate_starts"])
            for index, channel in enumerate(reference["channel_ids"]):
                valid = int(item["valid_segment_counts"][index])
                if bool(item["eligible"][index]):
                    status = "达标"
                elif actual_candidates < minimum_valid:
                    status = "候选片段不足"
                else:
                    status = f"合格片段不足（需 {minimum_valid}）"
                rows.append((
                    f"{int(length)} s", f"ch{int(channel)}", str(actual_candidates), str(valid),
                    str(int(item["rejected"]["unavailable"][index])),
                    str(int(item["rejected"]["saturated"][index])),
                    str(int(item["rejected"]["flat"][index])),
                    str(int(item["rejected"]["jump"][index])), "0", status,
                ))
        time_reference = reference.get("time_reference")
        if time_reference is not None:
            actual_candidates = len(time_reference["candidate_starts"])
            for index, channel in enumerate(reference["channel_ids"]):
                valid = int(time_reference["valid_segment_counts"][index])
                status = "达标" if bool(time_reference["eligible"][index]) else f"合格伪 epoch 不足（需 {minimum_valid}）"
                rows.append((
                    "时域", f"ch{int(channel)}", str(actual_candidates), str(valid),
                    str(int(time_reference["rejected"]["unavailable"][index])),
                    str(int(time_reference["rejected"]["saturated"][index])),
                    str(int(time_reference["rejected"]["flat"][index])),
                    str(int(time_reference["rejected"]["jump"][index])),
                    str(int(time_reference["rejected"]["response_missing"][index])), status,
                ))
        table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                table.setItem(row_index, column, QTableWidgetItem(value))
        table.resizeColumnsToContents()
        self.external_baseline_qc_summary.setText(
            "拒绝计数按候选片段或时域伪 epoch 统计：一个候选片段只要包含至少一个对应失败的 1 s 窗口，"
            "就在该原因列计 1；同一片段可同时出现在多个原因列。时域行另列出基线或响应窗有限值不足的计数。"
        )

    def _fail_external_baseline_reference(self, error: str) -> None:
        self.external_baseline_prepare_button.setEnabled(True)
        self._resume_external_time_frequency_after_baseline = False
        self._update_lfp_progress(0, f"实验前 baseline 准备失败：{error}")
        QMessageBox.warning(self, "baseline 准备失败", error)

    def export_lfp_task_baseline_csv(self) -> None:
        output = getattr(self, "lfp_task_result_output", None)
        if not output:
            QMessageBox.information(self, "尚无任务结果", "请先完成 LFP Letter 任务分析。")
            return
        rows = []
        for result in output.get("results", []):
            for metric in result.get("metrics", []):
                row = dict(metric)
                row["tag"] = int(result.get("tag", row.get("tag", -1)))
                rows.append(row)
        if not rows or not any("external_baseline_target_power_change_db" in row for row in rows):
            QMessageBox.information(self, "无外部 baseline 对照", "当前任务结果没有可导出的实验前 baseline 配对指标。")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出任务-baseline CSV", str(Path.cwd() / "lfp_task_external_baseline.csv"), "CSV files (*.csv)",
        )
        if not filename:
            return
        try:
            import csv
            keys = sorted({key for row in rows for key in row})
            with open(filename, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            cluster_rows = []
            for result in output.get("results", []):
                time_response = result.get("external_time_response") or {}
                time_ms = np.asarray(time_response.get("time_ms", []), dtype=float)
                for channel, comparison in (time_response.get("channels") or {}).items():
                    for cluster in comparison.get("clusters", []):
                        first, last = int(cluster["first"]), int(cluster["last"])
                        cluster_rows.append({
                            "tag": int(result.get("tag", -1)), "channel": int(channel),
                            "start_ms": float(time_ms[first]) if first < time_ms.size else np.nan,
                            "end_ms": float(time_ms[last]) if last < time_ms.size else np.nan,
                            "p_value": cluster["p_value"], "significant": cluster["significant"],
                            "direction": cluster["direction"], "mass": cluster["mass"],
                            "task_trials": comparison.get("task_trials", 0),
                            "rest_epochs": comparison.get("rest_epochs", 0),
                            "cluster_forming_p": comparison.get("cluster_forming_p", .01),
                            "cluster_significance_p": comparison.get("cluster_significance_p", .01),
                            "permutations": comparison.get("permutations", 1000),
                        })
            cluster_filename = str(Path(filename).with_name(Path(filename).stem + "_clusters.csv"))
            cluster_keys = [
                "tag", "channel", "start_ms", "end_ms", "p_value", "significant", "direction", "mass",
                "task_trials", "rest_epochs", "cluster_forming_p", "cluster_significance_p", "permutations",
            ]
            with open(cluster_filename, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=cluster_keys)
                writer.writeheader()
                writer.writerows(cluster_rows)
            self.lfp_status.setText(f"已导出任务-实验前 baseline 对照：{filename}；时域聚类明细：{cluster_filename}")
        except Exception as exc:
            QMessageBox.critical(self, "导出 baseline 对照失败", str(exc))

    def run_external_time_frequency(self) -> None:
        """Calculate one selected channel's task/rest power time-frequency map."""
        output = getattr(self, "lfp_task_result_output", None)
        reference = getattr(self, "external_baseline_reference", None)
        task_source = self.lfp_source or self.active_source
        baseline_source = self.external_baseline_source
        if not output or not output.get("results"):
            QMessageBox.information(self, "尚无任务结果", "请先完成 Letter 任务分析，再计算任务-静息时频图。")
            return
        if baseline_source is None:
            QMessageBox.information(self, "尚无静息 baseline", "请先导入实验前 baseline H5。")
            return
        if reference is None or reference.get("time_reference") is None:
            if not self.external_baseline_confirm_check.isChecked():
                QMessageBox.information(self, "尚无静息时域参考", "请确认上游重参考和单位一致后，程序才能自动准备 baseline。")
                return
            self._resume_external_time_frequency_after_baseline = True
            self._update_lfp_progress(0, "尚无当前任务对应的 baseline 参考，正在自动准备…")
            if not self.prepare_external_baseline_reference():
                self._resume_external_time_frequency_after_baseline = False
            return
        tag_index = self.lfp_task_result_tag_box.currentData()
        channel = self.lfp_task_result_channel_box.currentData()
        if tag_index is None or channel is None:
            QMessageBox.warning(self, "缺少选择", "请先在任务结果中选择 tag 和通道。")
            return
        try:
            if output.get("quality_signature") is not None:
                current_quality_signature = self._current_lfp_quality_signature()
                if output.get("quality_signature") != current_quality_signature:
                    raise ValueError("当前 marker、通道或五窗 QC 参数已改变，请重新运行任务分析后再计算时频图。")
            task_result = output["results"][int(tag_index)]
            if int(channel) not in {int(value) for value in output.get("channel_ids", [])}:
                raise ValueError("所选通道不在当前任务结果中。")
            current_baseline_settings = self._external_baseline_analysis_settings()
            task_channel_ids = [int(value) for value in output.get("channel_ids", [])]
            current_signature = self._external_baseline_task_signature(task_source, task_channel_ids, current_baseline_settings)
            if reference.get("task_signature") != current_signature:
                if not self.external_baseline_confirm_check.isChecked():
                    raise ValueError("baseline 参考需要重新按当前任务处理准备；请先确认上游重参考和单位一致。")
                self._resume_external_time_frequency_after_baseline = True
                self._update_lfp_progress(0, "检测到 baseline 参考与当前任务处理不一致，正在按当前参数自动重建…")
                if not self.prepare_external_baseline_reference():
                    self._resume_external_time_frequency_after_baseline = False
                return
            settings = {
                "epoch_start": float(output["epoch_start_ms"]), "epoch_end": float(output["epoch_end_ms"]),
                "baseline_start": float(output["baseline_start_ms"]), "baseline_end": float(output["baseline_end_ms"]),
                "response_start": self.external_time_response_start_spin.value(),
                "response_end": self.external_time_response_end_spin.value(),
                "analysis_notch": bool((output.get("task_processing") or {}).get("analysis_notch")),
                "analysis_bandpass": bool((output.get("task_processing") or {}).get("analysis_bandpass")),
                "analysis_band_low": float((output.get("task_processing") or {}).get("analysis_band_low", .5)),
                "analysis_band_high": float((output.get("task_processing") or {}).get("analysis_band_high", 300.0)),
                "smooth": bool((output.get("task_processing") or {}).get("smooth")),
                "smooth_window_sec": float((output.get("task_processing") or {}).get("smooth_window_sec", .02)),
                "zscore": bool((output.get("task_processing") or {}).get("zscore")),
                "quality_settings": self._read_lfp_trial_quality_settings(),
                "freq_low": self.external_tf_freq_low_spin.value(), "freq_high": self.external_tf_freq_high_spin.value(),
                "freq_step": self.external_tf_freq_step_spin.value(), "cycles": self.external_tf_cycles_spin.value(),
                "sample_rate_hz": 250.0,
            }
            if settings["response_end"] <= settings["response_start"]:
                raise ValueError("时频对比窗终点必须大于起点。")
            if not (settings["epoch_start"] <= settings["response_start"] < settings["response_end"] <= settings["epoch_end"]):
                raise ValueError("时频对比窗必须位于当前任务 epoch 内。")
            if settings["freq_high"] < settings["freq_low"]:
                raise ValueError("时频最高频率不能低于最低频率。")
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "时频参数无效", str(exc))
            return
        worker = ExternalTimeFrequencyWorker(task_source, baseline_source, reference, task_result, int(channel), settings)
        worker.progress.connect(self._update_lfp_progress)
        worker.completed.connect(self._finish_external_time_frequency)
        worker.failed.connect(self._fail_external_time_frequency)
        self._external_time_frequency_worker = worker
        self.external_time_frequency_button.setEnabled(False)
        self._update_lfp_progress(0, "正在计算任务-静息时频图…")
        worker.start()

    def run_trial_vep_overlay(self) -> None:
        """Plot all complete marker epochs; this path deliberately skips five-window QC."""
        output = getattr(self, "lfp_task_result_output", None)
        source = self.lfp_source or self.active_source
        if not output or not output.get("results"):
            QMessageBox.information(self, "尚无任务结果", "请先完成任务分析，再绘制 Trial-trial VEP。")
            return
        tag_index = self.lfp_task_result_tag_box.currentData()
        channel = self.lfp_task_result_channel_box.currentData()
        if tag_index is None or channel is None:
            QMessageBox.warning(self, "缺少选择", "请先在任务结果中选择 tag 和通道。")
            return
        try:
            task_result = output["results"][int(tag_index)]
            if not source.loaded:
                raise ValueError("当前 LFP 数据源尚未加载。")
        except (IndexError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "VEP 参数无效", str(exc))
            return
        worker = TrialVepWorker(source, task_result, int(channel), output)
        worker.progress.connect(self._update_lfp_progress)
        worker.completed.connect(self._finish_trial_vep)
        worker.failed.connect(self._fail_trial_vep)
        self._trial_vep_worker = worker
        self.lfp_trial_vep_button.setEnabled(False)
        self._update_lfp_progress(0, "正在提取未经过五窗 QC 切割的 Trial-trial VEP…")
        worker.start()

    def _finish_trial_vep(self, output: dict) -> None:
        self.lfp_trial_vep_output = output
        self.lfp_trial_vep_button.setEnabled(True)
        self._update_lfp_progress(100, f"Trial-trial VEP 完成：tag {output['tag']}，ch{output['channel']}，{len(output['trials'])} 个完整 trial。")
        self.show_trial_vep_overlay()

    def _fail_trial_vep(self, error: str) -> None:
        self.lfp_trial_vep_button.setEnabled(True)
        self._update_lfp_progress(0, f"Trial-trial VEP 失败：{error}")
        QMessageBox.warning(self, "Trial-trial VEP 失败", error)

    def show_trial_vep_overlay(self) -> None:
        initial_output = getattr(self, "lfp_trial_vep_output", None)
        task_output = getattr(self, "lfp_task_result_output", None)
        if not initial_output or not task_output:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Trial-trial 叠加 VEP")
        dialog.resize(1000, 680)
        layout = QVBoxLayout(dialog)
        controls = QHBoxLayout()
        tag_box = QComboBox()
        channel_box = QComboBox()
        for index, result in enumerate(task_output["results"]):
            tag_box.addItem(f"tag {int(result['tag'])}", index)
        for channel in task_output["channel_ids"]:
            channel_box.addItem(f"ch{int(channel)}", int(channel))
        tag_box.setCurrentIndex(max(0, tag_box.findText(f"tag {int(initial_output['tag'])}")))
        channel_box.setCurrentIndex(max(0, channel_box.findData(int(initial_output["channel"]))))
        calculate = QPushButton("计算所选 VEP")
        controls.addWidget(QLabel("Tag")); controls.addWidget(tag_box)
        controls.addWidget(QLabel("通道")); controls.addWidget(channel_box)
        controls.addWidget(calculate); controls.addStretch(1)
        layout.addLayout(controls)
        status = QLabel()
        status.setWordWrap(True)
        layout.addWidget(status)
        plot = pg.PlotWidget(); plot.setBackground("w")
        plot.setLabel("bottom", "相对刺激时间（ms）")
        plot.setLabel("left", "基线校正幅值（mV）")
        layout.addWidget(plot, 1)

        def render(output: dict) -> None:
            time_ms = np.asarray(output["time_ms"], dtype=float)
            trials = np.asarray(output["trials"], dtype=float)
            mean = np.asarray(output["mean"], dtype=float)
            sem = np.asarray(output["sem"], dtype=float)
            plot.clear(); plot.addLegend(offset=(12, 12))
            for trial in trials:
                plot.plot(time_ms, trial, pen=pg.mkPen((110, 110, 110, 55), width=1))
            plot.plot(time_ms, mean, pen=pg.mkPen("#d62728", width=2), name="平均响应")
            finite_sem = np.isfinite(mean) & np.isfinite(sem)
            if np.any(finite_sem):
                upper = pg.PlotCurveItem(time_ms, mean + sem, pen=pg.mkPen((214, 39, 40, 0)))
                lower = pg.PlotCurveItem(time_ms, mean - sem, pen=pg.mkPen((214, 39, 40, 0)))
                plot.addItem(upper); plot.addItem(lower)
                plot.addItem(pg.FillBetweenItem(upper, lower, brush=pg.mkBrush(214, 39, 40, 45)))
            plot.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#777", style=Qt.PenStyle.DashLine)))
            plot.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#777", style=Qt.PenStyle.DashLine)))
            dialog.setWindowTitle(f"Trial-trial 叠加 VEP：tag {output['tag']}，ch{output['channel']}")
            status.setText(
                f"完整 trial 数：{trials.shape[0]}；平均响应为所有完整 trial 的逐时间点均值；"
                f"红线为平均响应，红色半透明带为 SEM；本图未执行五窗切割、unavailable/saturated/flat/jump 拒绝。"
            )

        def calculate_selected() -> None:
            index = tag_box.currentData()
            channel = channel_box.currentData()
            if index is None or channel is None:
                return
            source = self.lfp_source or self.active_source
            worker = TrialVepWorker(source, task_output["results"][int(index)], int(channel), task_output)
            calculate.setEnabled(False)
            status.setText("正在计算所选 tag 和通道的 VEP…")
            worker.progress.connect(lambda _value, text: status.setText(text))

            def completed(new_output: dict) -> None:
                self.lfp_trial_vep_output = new_output
                calculate.setEnabled(True)
                render(new_output)

            def failed(error: str) -> None:
                calculate.setEnabled(True)
                status.setText(f"计算失败：{error}")

            worker.completed.connect(completed)
            worker.failed.connect(failed)
            self._trial_vep_dialog_worker = worker
            worker.start()

        calculate.clicked.connect(calculate_selected)
        render(initial_output)
        dialog.exec()

    def export_trial_vep(self) -> None:
        output = getattr(self, "lfp_trial_vep_output", None)
        if not output:
            QMessageBox.information(self, "尚无 VEP 数据", "请先计算 Trial-trial 叠加 VEP。")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出平均 VEP 数据", str(Path.cwd() / f"tag_{output['tag']}_ch{output['channel']}_vep.npz"),
            "NumPy archive (*.npz)",
        )
        if not filename:
            return
        try:
            np.savez_compressed(
                filename, tag=int(output["tag"]), channel=int(output["channel"]),
                time_ms=output["time_ms"], trials=output["trials"], mean=output["mean"],
                sem=output["sem"], valid_count=output["valid_count"],
                used_marker_samples=np.asarray(output["used_marker_samples"], dtype=np.int64),
                baseline_start_ms=float(output["baseline_start_ms"]),
                baseline_end_ms=float(output["baseline_end_ms"]), qc_applied=False,
            )
            self.lfp_status.setText(f"平均 VEP 数据已导出：{filename}")
        except Exception as exc:
            QMessageBox.critical(self, "导出 VEP 数据失败", str(exc))

    def _finish_external_time_frequency(self, output: dict) -> None:
        self.lfp_external_time_frequency_output = output
        self.external_time_frequency_button.setEnabled(True)
        comparison = output["comparison"]
        significant = sum(1 for item in comparison["clusters"] if item["significant"])
        self._update_lfp_progress(
            100,
            f"时频图完成：tag {output['tag']}，ch{output['channel']}；显著二维簇 {significant} 个。",
        )
        self.show_external_time_frequency()

    def _fail_external_time_frequency(self, error: str) -> None:
        self.external_time_frequency_button.setEnabled(True)
        self._update_lfp_progress(0, f"时频图失败：{error}")
        QMessageBox.warning(self, "任务-静息时频图失败", error)

    def show_external_time_frequency(self) -> None:
        output = getattr(self, "lfp_external_time_frequency_output", None)
        if not output:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"任务-静息功率时频图：tag {output['tag']}，ch{output['channel']}")
        dialog.resize(900, 680)
        layout = QVBoxLayout(dialog)
        controls = QHBoxLayout()
        metric = QComboBox()
        metric.addItem("任务相对自身基线（dB）", "task_mean_db")
        metric.addItem("静息相对自身基线（dB）", "rest_mean_db")
        metric.addItem("任务 - 静息（dB，叠加显著簇）", "task_minus_rest_db")
        controls.addWidget(QLabel("显示")); controls.addWidget(metric); controls.addStretch(1)
        layout.addLayout(controls)
        status = QLabel(); status.setWordWrap(True); layout.addWidget(status)
        plot = pg.PlotWidget(); plot.setBackground("w")
        plot.setLabel("bottom", "相对刺激时间（ms)")
        plot.setLabel("left", "频率（Hz)")
        layout.addWidget(plot, 1)
        time_ms = np.asarray(output["time_ms"], dtype=float)
        frequencies = np.asarray(output["frequencies_hz"], dtype=float)

        def render() -> None:
            key = metric.currentData()
            values = np.asarray(output[key], dtype=float)
            plot.clear()
            image = pg.ImageItem(values.T)
            image.setRect(QRectF(
                float(time_ms[0]), float(frequencies[0]),
                max(.001, float(time_ms[-1] - time_ms[0])), max(.001, float(frequencies[-1] - frequencies[0])),
            ))
            finite = values[np.isfinite(values)]
            if finite.size:
                bound = float(np.nanpercentile(np.abs(finite), 98))
                image.setLevels((-bound, bound) if key == "task_minus_rest_db" else (float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))))
            plot.addItem(image)
            if key == "task_minus_rest_db":
                mask = np.asarray(output["comparison"]["significant_mask"], dtype=bool)
                if np.any(mask):
                    rgba = np.zeros((*mask.T.shape, 4), dtype=np.ubyte)
                    rgba[mask.T] = (22, 101, 52, 115)
                    overlay = pg.ImageItem(rgba)
                    overlay.setRect(image.rect())
                    plot.addItem(overlay)
            plot.setXRange(float(time_ms[0]), float(time_ms[-1]), padding=0)
            plot.setYRange(float(frequencies[0]), float(frequencies[-1]), padding=0)
            comparison = output["comparison"]
            significant = sum(1 for item in comparison["clusters"] if item["significant"])
            status.setText(
                f"任务有效 trial {comparison['task_trials']}；静息有效伪 epoch {comparison['rest_epochs']}；"
                f"Morlet {output['cycles']:g} cycles；二维双侧簇置换 1000 次，簇形成及显著性均为 p<0.01；"
                f"绿色半透明区域为最终显著簇（{significant} 个）。"
            )

        metric.currentIndexChanged.connect(render)
        render()
        dialog.exec()

    def _read_lfp_trial_quality_settings(self) -> dict:
        """Read and validate the editable page-4 five-window QC thresholds."""
        get = lambda name, default: float(self._widget_text(getattr(self, name), str(default)) or default)
        settings = {
            "quality_available_ratio": get("lfp_qc_available_ratio_var", .995),
            "quality_saturation_run_samples": int(get("lfp_qc_saturation_run_var", 8)),
            "quality_flat_epsilon": get("lfp_qc_flat_epsilon_var", 1e-4),
            "quality_flat_ratio_percent": get("lfp_qc_flat_ratio_var", 30.0),
            "quality_flat_ptp": get("lfp_qc_flat_ptp_var", .01),
            "quality_jump_mad_multiplier": get("lfp_qc_jump_mad_multiplier_var", 12.0),
            "quality_jump_median_multiplier": get("lfp_qc_jump_median_multiplier_var", 8.0),
            "quality_jump_flat_floor_multiplier": get("lfp_qc_jump_floor_multiplier_var", 10.0),
        }
        if not 0 < settings["quality_available_ratio"] <= 1:
            raise ValueError("五窗 QC 的有限值比例必须在 0 到 1 之间。")
        if settings["quality_saturation_run_samples"] < 2:
            raise ValueError("五窗 QC 的饱和连续采样点数至少为 2。")
        if settings["quality_flat_epsilon"] < 0 or settings["quality_flat_ptp"] < 0:
            raise ValueError("五窗 QC 的平坦阈值不能为负数。")
        if not 0 <= settings["quality_flat_ratio_percent"] <= 100:
            raise ValueError("五窗 QC 的平坦比例必须在 0 到 100 之间。")
        if any(settings[name] < 0 for name in (
            "quality_jump_mad_multiplier", "quality_jump_median_multiplier",
            "quality_jump_flat_floor_multiplier",
        )):
            raise ValueError("五窗 QC 的跳变系数不能为负数。")
        return settings

    def _current_lfp_quality_signature(self) -> tuple:
        """Build the current UI fingerprint required to reuse a QC cache."""
        if not hasattr(self, "stim_markers"):
            raise ValueError("请先在时间对齐页生成 marker。")
        mode = str(getattr(self, "stim_marker_mode", "") or "")
        if mode not in {"Flash", "Letter"}:
            raise ValueError("当前 marker 没有有效的 Flash 或 Letter 模式。")
        source = self.lfp_source or self.active_source
        alignment_ids = set(getattr(self, "lfp_selected_ids", set())) or set(getattr(self, "alignment_selected_ids", set()))
        if not alignment_ids:
            raise ValueError("请先在时间对齐页选择至少一个对齐通道。")
        columns = np.flatnonzero(np.isin(source.metadata.channel_ids, sorted(alignment_ids)))
        if not columns.size:
            raise ValueError("当前对齐通道不在 LFP 数据源中。")
        get = lambda name, default: float(self._widget_text(getattr(self, name), str(default)) or default)
        settings = {
            "columns": columns,
            "epoch_start": get("epoch_start_ms_var", -500), "epoch_end": get("epoch_end_ms_var", 6000),
            "trials_per_stim": int(get("trials_per_stim_var", 30)),
            **self._read_lfp_trial_quality_settings(),
            "analysis_notch": isinstance(self.analysis_notch_var, QCheckBox) and self.analysis_notch_var.isChecked(),
            "analysis_bandpass": isinstance(self.analysis_bandpass_var, QCheckBox) and self.analysis_bandpass_var.isChecked(),
            "analysis_band_low": get("analysis_band_low_var", .5), "analysis_band_high": get("analysis_band_high_var", 300),
            "smooth": isinstance(self.smooth_signal_var, QCheckBox) and self.smooth_signal_var.isChecked(),
            "smooth_window_sec": get("smooth_window_size_var", .02),
            "zscore": isinstance(self.zscoredata_var, QCheckBox) and self.zscoredata_var.isChecked(),
        }
        return self._lfp_quality_signature(mode, np.asarray(self.stim_markers, dtype=int), settings)

    def _validate_lfp_quality_cache(self, output: dict) -> str | None:
        """Return a user-facing reason when cached page-4 QC is stale."""
        source = self.lfp_source or self.active_source
        meta = source.metadata
        expected_source = (int(meta.rows), float(meta.fs), float(meta.time_offset), tuple(int(value) for value in meta.channel_ids))
        if output.get("source_signature") != expected_source:
            return "当前 LFP 数据源已改变。"
        expected_identity = (id(source), id(getattr(source, "data", None)))
        if output.get("source_identity") != expected_identity:
            return "当前 LFP 数据对象或预处理版本已改变。"
        try:
            current_signature = self._current_lfp_quality_signature()
        except ValueError as exc:
            return str(exc)
        if output.get("quality_signature") != current_signature:
            return "marker、对齐通道、epoch、质量阈值或任务处理参数已改变。"
        return None

    def run_lfp_trial_quality_check(self) -> None:
        """Run the page-4 five-window QC before any LFP task frequency analysis."""
        mode = str(getattr(self, "stim_marker_mode", "") or "")
        if mode not in {"Flash", "Letter"}:
            QMessageBox.information(self, "尚未对齐", "请先在时间对齐页生成 Flash 或 Letter markers。")
            return
        self.run_task_epoch_analysis("lfp", mode, quality_only=True)

    def run_lfp_letter_frequency_analysis(self) -> None:
        """Use the saved five-window QC cache for Letter PSD/task metrics."""
        if getattr(self, "stim_marker_mode", None) != "Letter":
            QMessageBox.information(self, "需要 Letter 对齐", "请先在时间对齐页选择 Letter 并生成对应 markers。")
            return
        if not getattr(self, "lfp_trial_quality_output", None):
            QMessageBox.information(self, "尚未完成五窗判断", "请先点击“执行 trial 五窗质量判断”。")
            return
        self.run_task_epoch_analysis("lfp", "Letter", reuse_cached_quality=True)

    def run_task_epoch_analysis(
        self, stage: str, mode: str, *, lfp_task_metrics: bool | None = None,
        quality_only: bool = False, reuse_cached_quality: bool = False,
    ) -> None:
        if not hasattr(self, "stim_markers"):
            QMessageBox.information(self, "尚未对齐", "请先在时间对齐页生成 Flash 或 Letter markers。")
            return
        aligned_mode = getattr(self, "stim_marker_mode", None)
        if aligned_mode and aligned_mode != mode:
            QMessageBox.warning(
                self, "任务模式不一致",
                f"当前保存的是 {aligned_mode} markers。请在时间对齐页选择 {mode}，然后点击“计算并刷新对齐”。",
            )
            return
        source = (self.lfp_source if stage == "lfp" else self.spike_source) or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载数据。")
            return
        try:
            get = lambda name, default: float(self._widget_text(getattr(self, name), str(default)) or default)
            tags_text = self._widget_text(self.stimtag_var, "")
            tags = [int(float(value.strip())) for value in tags_text.replace(";", ",").split(",") if value.strip()]
            # A page-4 LFP selection is authoritative for every LFP task
            # operation.  If it is empty, retain the alignment-page selection
            # as the compatibility fallback.
            alignment_ids = set(getattr(self, "lfp_selected_ids", set())) or set(getattr(self, "alignment_selected_ids", set()))
            if not alignment_ids:
                raise ValueError("请先在时间对齐页选择并确认至少一个对齐 / Letter 通道。")
            columns = np.flatnonzero(np.isin(source.metadata.channel_ids, sorted(alignment_ids)))
            if not columns.size:
                raise ValueError("已选对齐通道不在当前任务数据源中；请重新选择或导入对应通道数据。")
            settings = {
                "mode": mode,
                "epoch_start": get("epoch_start_ms_var", -500), "epoch_end": get("epoch_end_ms_var", 6000),
                "baseline_start": get("baseline_start_ms_var", -200), "baseline_end": get("baseline_end_ms_var", 0),
                "response_start": get("response_start_ms_var", 0), "response_end": get("response_end_ms_var", 300),
                "external_time_response_start": self.external_time_response_start_spin.value() if stage == "lfp" else 0.0,
                "external_time_response_end": self.external_time_response_end_spin.value() if stage == "lfp" else 5500.0,
                "external_time_rate_hz": 250.0,
                "trials_per_stim": int(get("trials_per_stim_var", 30)),
                "n_stim_types": int(get("n_stim_types_var", 4)), "tags": tags,
                "aggregate": self._widget_text(self.task_response_aggregate_var, "Mean").strip().lower(), "columns": columns,
                "analysis_notch": isinstance(self.analysis_notch_var, QCheckBox) and self.analysis_notch_var.isChecked(),
                "analysis_bandpass": isinstance(self.analysis_bandpass_var, QCheckBox) and self.analysis_bandpass_var.isChecked(),
                "analysis_band_low": get("analysis_band_low_var", .5), "analysis_band_high": get("analysis_band_high_var", 300),
                "smooth": isinstance(self.smooth_signal_var, QCheckBox) and self.smooth_signal_var.isChecked(),
                "smooth_window_sec": get("smooth_window_size_var", .02),
                "zscore": isinstance(self.zscoredata_var, QCheckBox) and self.zscoredata_var.isChecked(),
                "task_target_freq_hz": get("task_target_freq_var", 1.0),
                "task_neighbor_bins": int(get("neighbor_bins_var", 4)),
                "lfp_task_metrics": stage == "lfp" if lfp_task_metrics is None else bool(lfp_task_metrics),
                "view_mode": self._widget_text(self.analysis_view_mode_var, "raw").lower(),
                "compare_trial_a": int(get("compare_trial_a_var", 30)),
                "compare_trial_b": int(get("compare_trial_b_var", 60)),
                # Page 4 owns its five-window QC thresholds independently
                # from the preprocessing-page bad-channel check.
                **self._read_lfp_trial_quality_settings(),
                "behavior_matches": dict(getattr(self, "behavior_matches", {})) if mode == "Letter" else {},
                "quality_only": bool(quality_only),
            }
        except Exception as exc:
            QMessageBox.warning(self, "任务参数无效", str(exc)); return
        markers = np.asarray(self.stim_markers, dtype=int)
        if quality_only:
            # QC belongs to the aligned marker set itself, not to a later
            # display subset selected through stimtag/nStimTypes.
            settings["tags"] = []
            settings["n_stim_types"] = 0
        if stage == "lfp":
            settings["quality_signature"] = self._lfp_quality_signature(mode, markers, settings)
            external_reference = getattr(self, "external_baseline_reference", None)
            task_channel_ids = [int(source.metadata.channel_ids[index]) for index in settings["columns"]]
            if external_reference is not None and external_reference.get("task_signature") == self._external_baseline_task_signature(
                source, task_channel_ids, settings,
            ):
                settings["external_baseline_reference"] = external_reference
            if reuse_cached_quality:
                cached = self.lfp_trial_quality_output
                if cached.get("quality_signature") != settings["quality_signature"]:
                    raise ValueError("当前 marker、通道、epoch 或质量阈值已改变；请先重新执行 trial 五窗质量判断。")
                settings["precomputed_trial_quality"] = cached.get("trial_quality", {})
        if mode == "Letter" and not np.any(np.isin(markers[:, 1], [4, 5, 6, 7])):
            QMessageBox.warning(
                self, "没有 Letter markers",
                "当前对齐结果没有 4/5/6/7 标签。请确认时间对齐页选择了 Letter，并使用对应的 Event CSV 后重新计算。",
            )
            return
        active_worker = getattr(self, "_lfp_task_worker" if stage == "lfp" else "_spike_task_worker", None)
        if active_worker is not None and active_worker.isRunning():
            QMessageBox.information(self, "任务正在运行", "当前任务仍在读取或分析数据，请等待完成。")
            return
        worker = TaskEpochWorker(source, markers, settings)
        worker.progress.connect(lambda value, text: self._update_task_progress(stage, value, text))
        worker.completed.connect(lambda output, kind=stage, name=mode: self._finish_task_epoch_analysis(kind, name, output))
        worker.failed.connect(lambda error, kind=stage: self._fail_task_epoch_analysis(kind, error))
        if stage == "lfp": self._lfp_task_worker = worker
        else: self._spike_task_worker = worker
        self._set_task_buttons_enabled(stage, False)
        self._update_task_progress(stage, 0, f"{mode} 任务：正在启动…")
        worker.start()

    def _set_task_buttons_enabled(self, stage: str, enabled: bool) -> None:
        prefix = "lfp" if stage == "lfp" else "spike"
        names = [f"{prefix}_flash_task_button", f"{prefix}_letter_task_button"]
        if stage == "lfp":
            names.append("lfp_trial_quality_button")
        for name in names:
            button = getattr(self, name, None)
            if button is not None:
                button.setEnabled(enabled)

    def _update_task_progress(self, stage, value, text) -> None:
        (self.lfp_progress if stage == "lfp" else self.spike_progress).setValue(int(round(value)))
        (self.lfp_status if stage == "lfp" else self.spike_status).setText(text)

    def _finish_task_epoch_analysis(self, stage, mode, output) -> None:
        self._set_task_buttons_enabled(stage, True)
        try:
            self.task_results = output
            figure = self.lfp_figure if stage == "lfp" else self.spike_figure
            canvas = self.lfp_canvas if stage == "lfp" else self.spike_canvas
            figure.clear(); result = output["results"][0]; ids = output["channel_ids"]
            if stage == "lfp" and output.get("quality_only", False):
                self.lfp_trial_quality_output = output
                self._render_lfp_trial_quality_summary(output)
                self._update_task_progress(stage, 100, f"trial 五窗质量判断完成：{len(output['results'])} 个 tag；现在可进行 Letter 任务分析。")
                return
            if mode == "Letter" and not output.get("lfp_task_metrics", False):
                self._show_letter_behavior_browser(stage, output)
                self._update_task_progress(stage, 100, f"Letter 任务完成：{len(output['results'])} 个 tag；已打开 Letter 视频关联通道浏览。")
                return
            if mode == "Letter" and stage == "lfp" and output.get("lfp_task_metrics", False):
                self._render_lfp_task_quality_summary(mode, output)
                self._update_task_progress(stage, 100, f"Letter 五窗任务分析完成：{len(output['results'])} 个 tag；已显示质量控制和连续 PSD 结果。")
                return
            axis = figure.add_subplot(111)
            for index in range(min(12, result["wave"].shape[1])):
                axis.plot(result["t_ms"], result["wave"][:, index], linewidth=.65, label=f"ch{ids[index]}")
            axis.axvline(0, color="#c62828", linestyle="--", linewidth=.8); axis.set_xlabel("相对刺激时间（ms）"); axis.set_ylabel("幅值(μV)")
            axis.set_title(f"{mode} {stage.upper()} task：tag {result['tag']}，{result['trials']} trials，{output['aggregate']}"); axis.grid(alpha=.25); axis.legend(ncol=4, fontsize=7)
            canvas.draw_idle(); self._update_task_progress(stage, 100, f"{mode} 任务分析完成：{len(output['results'])} 个 tag。")
        except Exception as exc:
            self._fail_task_epoch_analysis(stage, f"结果绘制失败：{exc}")

    def _render_lfp_trial_quality_summary(self, output: dict) -> None:
        """Render the saved page-4 trial QC result before PSD is requested."""
        results = list(output.get("results") or [])
        if not results:
            raise ValueError("没有可显示的 trial 五窗质量判断结果。")
        summary_lines = [
            "trial 五窗质量判断完成（尚未计算 PSD 或频率指标）",
            "稳态区间：0.5-5.5 s；五个连续 1 s 窗口为 W1 0.5-1.5、W2 1.5-2.5、W3 2.5-3.5、W4 3.5-4.5、W5 4.5-5.5 s。",
            "完整五窗 trial 需至少 3 个干净窗；短 epoch 只评估实际完整存在的窗口，缺失窗口明确记为不可用。连续至少 2 个干净窗才能计算 PSD。",
            "窗口检查项：数据可用性、饱和、平坦、突变；下方按通道列出累计结果。",
        ]
        channel_rates: dict[int, list[float]] = {}
        for result in results:
            metrics = list(result.get("metrics") or [])
            trial_count = max(1, int(result["trials"]))
            qualified_channels = sum(int(row["usable_trials"]) > 0 for row in metrics)
            summary_lines.append(
                f"\ntag {result['tag']}：完整 trial {result['trials']}，有至少一个可用 trial 的通道 {qualified_channels}/{len(metrics)}。"
            )
            for row in metrics:
                channel = int(row["channel"])
                usable = int(row["usable_trials"])
                channel_rates.setdefault(channel, []).append(usable / trial_count * 100.0)
                summary_lines.append(
                    f"  ch{channel}: 可用/拒绝 trial {usable}/{row['rejected_trials']}，"
                    f"完整可用/干净窗 {row.get('available_windows', 0)}/{row['clean_windows']}，短 epoch/仅响应 trial {row.get('short_epoch_trials', 0)}/{row.get('response_only_trials', 0)}，不可用/饱和/平坦/跳变窗 "
                    f"{row['unavailable_windows']}/{row['saturated_windows']}/{row['flat_windows']}/{row['jump_windows']}。"
                )
        self.lfp_results.setPlainText("\n".join(summary_lines))

        self.lfp_figure.clear()
        axis = self.lfp_figure.add_subplot(111)
        channels = np.asarray(sorted(channel_rates), dtype=float)
        rates = np.asarray([np.mean(channel_rates[int(channel)]) for channel in channels], dtype=float)
        axis.plot(channels, rates, color="#b71c1c", linewidth=.8, marker=".", markersize=5)
        axis.set_xlabel("通道号")
        axis.set_ylabel("可用 trial 比例 (%)")
        axis.set_title("trial 五窗质量判断：各通道可用于后续 Letter 分析的 trial 比例")
        axis.set_ylim(0, 100)
        axis.grid(alpha=.25)
        self.lfp_canvas.draw_idle()

    def show_lfp_trial_quality_browser(self) -> None:
        """Let users inspect a cached QC decision against its source epoch."""
        output = getattr(self, "lfp_trial_quality_output", None)
        if not output or not output.get("trial_quality"):
            QMessageBox.information(self, "尚无五窗判断", "请先执行 trial 五窗质量判断。")
            return
        source = self.lfp_source or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载用于 LFP 分析的数据。")
            return
        stale_reason = self._validate_lfp_quality_cache(output)
        if stale_reason:
            QMessageBox.warning(self, "五窗缓存已失效", f"{stale_reason} 请重新执行 trial 五窗质量判断。")
            return
        quality_rows = output["trial_quality"]
        keys = sorted((int(tag), int(sample)) for tag, sample in quality_rows)
        channel_ids = [int(value) for value in output.get("channel_ids", [])]
        if not keys or not channel_ids:
            QMessageBox.information(self, "没有可查看的数据", "缓存中没有完整的 trial 或通道。")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("trial 五窗质量浏览")
        dialog.resize(980, 720)
        layout = QVBoxLayout(dialog)
        controls = QHBoxLayout()
        tag_box = QComboBox()
        for tag in sorted({tag for tag, _sample in keys}):
            tag_box.addItem(f"tag {tag}", tag)
        trial_box = QComboBox()
        channel_box = QComboBox()
        for index, channel in enumerate(channel_ids):
            channel_box.addItem(f"ch{channel}", index)
        controls.addWidget(QLabel("标签")); controls.addWidget(tag_box)
        controls.addWidget(QLabel("trial")); controls.addWidget(trial_box)
        controls.addWidget(QLabel("通道")); controls.addWidget(channel_box)
        controls.addStretch(1)
        layout.addLayout(controls)
        status = QLabel()
        status.setWordWrap(True)
        layout.addWidget(status)
        plot = PyQtGraphPlot(dialog)
        layout.addWidget(plot, 1)
        table = QTableWidget(5, 4, dialog)
        table.setHorizontalHeaderLabels(["窗口", "时间（相对刺激）", "判断", "原因"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)

        def populate_trials() -> None:
            tag = int(tag_box.currentData())
            trial_box.blockSignals(True)
            trial_box.clear()
            for key_tag, sample in keys:
                if key_tag == tag:
                    trial_box.addItem(f"样本 {sample}", (key_tag, sample))
            trial_box.blockSignals(False)
            render()

        def render() -> None:
            key = trial_box.currentData()
            channel_index = channel_box.currentData()
            if key is None or channel_index is None:
                return
            tag, sample = (int(key[0]), int(key[1]))
            channel_index = int(channel_index)
            saved = quality_rows[(tag, sample)]
            meta = source.metadata
            epoch_start_ms = float(output["epoch_start_ms"])
            epoch_end_ms = float(output["epoch_end_ms"])
            start = int(round(epoch_start_ms * meta.fs / 1000.0))
            end = int(round(epoch_end_ms * meta.fs / 1000.0))
            marker_offset = int(round(float(meta.time_offset) * meta.fs))
            center = sample - marker_offset
            values = np.asarray(source.read(center + start, center + end, int(output["columns"][channel_index])), dtype=float).ravel()
            t_ms = np.arange(values.size, dtype=float) / meta.fs * 1000.0 + epoch_start_ms
            baseline = (t_ms >= float(output["baseline_start_ms"])) & (t_ms <= float(output["baseline_end_ms"]))
            if np.any(baseline):
                values = values - np.nanmean(values[baseline])

            clean = np.asarray(saved["clean_windows"], dtype=bool)[:, channel_index]
            available = np.asarray(saved["available"], dtype=bool)[:, channel_index]
            plot.clear()
            plot.plot(t_ms, values, pen=pg.mkPen("#9aa8b1", width=1))
            color_by_reason = {"saturated": "#c62828", "flat": "#ef6c00", "jump": "#7b1fa2"}
            reasons_by_window = []
            for number in range(5):
                left, right = 500.0 + number * 1000.0, 1500.0 + number * 1000.0
                plot.addItem(pg.InfiniteLine(pos=left, angle=90, movable=False, pen=pg.mkPen("#78909c", width=1)))
                in_window = (t_ms >= left) & (t_ms < right)
                if not available[number]:
                    judgment, reason, color = "不可用", "epoch 未完整覆盖或数据缺失", "#757575"
                elif clean[number]:
                    judgment, reason, color = "干净", "通过", "#2e7d32"
                else:
                    failed = [name for name in ("saturated", "flat", "jump") if np.asarray(saved[name], dtype=bool)[number, channel_index]]
                    judgment, reason = "拒绝", "、".join({"saturated": "饱和", "flat": "平坦", "jump": "突变"}[name] for name in failed)
                    color = color_by_reason.get(failed[0], "#c62828") if failed else "#c62828"
                reasons_by_window.append((judgment, reason))
                if np.any(in_window):
                    plot.plot(t_ms[in_window], values[in_window], pen=pg.mkPen(color, width=2))
                table.setItem(number, 0, QTableWidgetItem(f"W{number + 1}"))
                table.setItem(number, 1, QTableWidgetItem(f"{left / 1000:.1f}–{right / 1000:.1f} s"))
                table.setItem(number, 2, QTableWidgetItem(judgment))
                table.setItem(number, 3, QTableWidgetItem(reason))
            plot.addItem(pg.InfiniteLine(pos=5500.0, angle=90, movable=False, pen=pg.mkPen("#78909c", width=1)))
            plot.setLabel("bottom", "相对刺激时间（ms）")
            plot.setLabel("left", "幅值（当前 LFP 数据源，baseline 校正）")
            plot.setTitle(f"tag {tag}，样本 {sample}，ch{channel_ids[channel_index]}：绿=干净，红/橙/紫=拒绝，灰=不可用")
            plot.setXRange(epoch_start_ms, epoch_end_ms, padding=0)
            status.setText(
                f"该图使用当前 LFP 数据源的单个 epoch；五窗判断来自已缓存的 QC。"
                f"完整可用窗 {int(np.sum(available))}/5，干净窗 {int(np.sum(clean))}/5。"
            )

        tag_box.currentIndexChanged.connect(populate_trials)
        trial_box.currentIndexChanged.connect(render)
        channel_box.currentIndexChanged.connect(render)
        populate_trials()
        dialog.exec()

    def run_lfp_itpc(self) -> None:
        quality = getattr(self, "lfp_trial_quality_output", None)
        if not quality or not quality.get("trial_quality"):
            QMessageBox.information(self, "尚无五窗判断", "请先执行 trial 五窗质量判断，再计算 ITPC。")
            return
        stale_reason = self._validate_lfp_quality_cache(quality)
        if stale_reason:
            QMessageBox.warning(self, "五窗缓存已失效", f"{stale_reason} 请重新执行 trial 五窗质量判断。")
            return
        try:
            tags = [int(value.strip()) for value in self.itpc_tag_edit.text().replace(";", ",").split(",") if value.strip()]
            settings = {
                "tags": tags,
                "freq_low": self.itpc_freq_low.value(), "freq_high": self.itpc_freq_high.value(),
                "freq_step": self.itpc_freq_step.value(), "cycles": self.itpc_cycles.value(),
                "time_start": self.itpc_time_start.value(), "time_end": self.itpc_time_end.value(),
                "baseline_start": self.itpc_baseline_start.value(), "baseline_end": self.itpc_baseline_end.value(),
                "sample_rate": self.itpc_sample_rate.value(), "min_trials": self.itpc_min_trials.value(),
            }
            if settings["freq_low"] > settings["freq_high"]:
                raise ValueError("ITPC 起始频率不能高于终止频率。")
            if settings["baseline_end"] <= settings["baseline_start"]:
                raise ValueError("ITPC 参考窗终点必须大于起点。")
        except ValueError as exc:
            QMessageBox.warning(self, "ITPC 参数无效", str(exc)); return
        source = self.lfp_source or self.active_source
        worker = ItpcWorker(source, quality, settings)
        worker.progress.connect(lambda value, text: self._update_lfp_progress(value, text))
        worker.completed.connect(self._finish_lfp_itpc)
        worker.failed.connect(self._fail_lfp_itpc)
        self._lfp_itpc_worker = worker
        self.lfp_status.setText("正在计算 marker 对齐 ITPC（复用五窗 QC）…")
        worker.start()

    def _finish_lfp_itpc(self, output: dict) -> None:
        self.lfp_itpc_output = output
        tags = ", ".join(str(item["tag"]) for item in output["results"])
        self.lfp_status.setText(f"ITPC 计算完成：tag {tags}；{len(output['frequencies_hz'])} 个频率，采样率 {output['sample_rate']:g} Hz。")
        self.show_lfp_itpc_viewer()

    def _fail_lfp_itpc(self, error: str) -> None:
        self.lfp_status.setText(f"ITPC 计算失败：{error}")
        QMessageBox.critical(self, "ITPC 计算失败", error)

    def show_lfp_itpc_viewer(self) -> None:
        output = getattr(self, "lfp_itpc_output", None)
        if not output or not output.get("results"):
            QMessageBox.information(self, "尚无 ITPC", "请先计算 ITPC。")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("ITPC 时频图")
        dialog.resize(920, 680)
        layout = QVBoxLayout(dialog)
        controls = QHBoxLayout()
        tag_box = QComboBox(); channel_box = QComboBox(); metric_box = QComboBox()
        metric_box.addItems(["ITPC", "相对参考窗变化"])
        for index, result in enumerate(output["results"]):
            tag_box.addItem(f"tag {result['tag']}", index)
        for index, channel in enumerate(output["channel_ids"]):
            channel_box.addItem(f"ch{int(channel)}", index)
        controls.addWidget(QLabel("Tag")); controls.addWidget(tag_box)
        controls.addWidget(QLabel("通道")); controls.addWidget(channel_box)
        controls.addWidget(QLabel("显示")); controls.addWidget(metric_box); controls.addStretch(1)
        layout.addLayout(controls)
        status = QLabel(); layout.addWidget(status)
        numeric_button = QPushButton("ITPC 数值表")
        numeric_button.setToolTip("查看当前 ITPC 矩阵的轴信息、指定时间点数值和有效 trial 数")
        numeric_button.clicked.connect(lambda: self.show_lfp_itpc_numeric_table(output))
        layout.addWidget(numeric_button)
        plot = pg.PlotWidget(); plot.setBackground("w")
        plot.setLabel("bottom", "相对刺激时间（ms）")
        plot.setLabel("left", "频率（Hz）")
        layout.addWidget(plot, 1)

        def render() -> None:
            result = output["results"][int(tag_box.currentData())]
            channel = int(channel_box.currentData())
            field = "itpc" if metric_box.currentIndex() == 0 else "itpc_baseline_delta"
            values = np.asarray(result[field][:, :, channel], dtype=float)
            plot.clear()
            image = pg.ImageItem(values.T)
            t = np.asarray(output["t_ms"], dtype=float); f = np.asarray(output["frequencies_hz"], dtype=float)
            image.setRect(QRectF(float(t[0]), float(f[0]), max(.001, float(t[-1] - t[0])), max(.001, float(f[-1] - f[0]))))
            plot.addItem(image); plot.setXRange(float(t[0]), float(t[-1]), padding=0); plot.setYRange(float(f[0]), float(f[-1]), padding=0)
            count = np.asarray(result["counts"][:, :, channel])
            status.setText(f"ITPC 范围 0–1；参考窗 {output['baseline_start']:g}–{output['baseline_end']:g} ms。该 tag 有 {result['trials']} 个 QC 缓存 trial，显示点要求至少 {output['min_trials']} 个有效 trial。有效 trial 数：{int(np.nanmin(count))}–{int(np.nanmax(count))}。")

        tag_box.currentIndexChanged.connect(render); channel_box.currentIndexChanged.connect(render); metric_box.currentIndexChanged.connect(render)
        render(); dialog.exec()

    def show_lfp_itpc_numeric_table(self, output: dict) -> None:
        """Show inspectable ITPC array values without expanding a huge matrix."""
        if not output or not output.get("results"):
            QMessageBox.information(self, "尚无 ITPC", "请先计算 ITPC。")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("ITPC 数值表")
        dialog.resize(900, 700)
        layout = QVBoxLayout(dialog)
        controls = QHBoxLayout()
        tag_box = QComboBox(); channel_box = QComboBox(); metric_box = QComboBox()
        metric_box.addItems(["ITPC", "相对参考窗变化"])
        for index, result in enumerate(output["results"]):
            tag_box.addItem(f"tag {result['tag']}", index)
        for index, channel in enumerate(output["channel_ids"]):
            channel_box.addItem(f"ch{int(channel)}", index)
        controls.addWidget(QLabel("Tag")); controls.addWidget(tag_box)
        controls.addWidget(QLabel("通道")); controls.addWidget(channel_box)
        controls.addWidget(QLabel("显示")); controls.addWidget(metric_box)
        controls.addStretch(1)
        layout.addLayout(controls)

        metadata_table = QTableWidget(0, 4)
        metadata_table.setHorizontalHeaderLabels(["数组", "形状/数量", "范围", "说明"])
        metadata_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        metadata_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        metadata_table.setMaximumHeight(180)
        layout.addWidget(metadata_table)

        point_controls = QHBoxLayout()
        point_controls.addWidget(QLabel("时间点 ms"))
        time_box = QComboBox(); time_box.setMinimumWidth(140)
        times = np.asarray(output["t_ms"], dtype=float)
        for value in times:
            time_box.addItem(f"{value:.3f}", float(value))
        point_controls.addWidget(time_box)
        point_controls.addStretch(1)
        layout.addLayout(point_controls)

        value_table = QTableWidget(0, 3)
        value_table.setHorizontalHeaderLabels(["频率 Hz", "ITPC", "有效 trial"])
        value_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        value_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        value_table.setAlternatingRowColors(True)
        layout.addWidget(value_table, 1)
        status = QLabel(); status.setWordWrap(True); layout.addWidget(status)

        def format_range(values: np.ndarray, integer: bool = False) -> str:
            array = np.asarray(values)
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                return "无有效值"
            if integer:
                return f"{int(np.min(finite))}–{int(np.max(finite))}"
            return f"{float(np.min(finite)):.6g}–{float(np.max(finite)):.6g}"

        def render() -> None:
            result = output["results"][int(tag_box.currentData())]
            channel_index = int(channel_box.currentData())
            frequencies = np.asarray(output["frequencies_hz"], dtype=float)
            full_itpc = np.asarray(result["itpc"], dtype=float)
            full_counts = np.asarray(result["counts"], dtype=float)
            field = "itpc" if metric_box.currentIndex() == 0 else "itpc_baseline_delta"
            values = np.asarray(result[field][:, :, channel_index], dtype=float)
            counts = np.asarray(full_counts[:, :, channel_index], dtype=float)
            metadata = [
                (f"tag_{result['tag']}_itpc", str(tuple(full_itpc.shape)), format_range(values), "完整 ITPC 数组；当前表显示所选通道"),
                (f"tag_{result['tag']}_valid_trials", str(tuple(full_counts.shape)), format_range(counts, integer=True), "每个频率–时间点的有效 trial 数"),
                ("frequencies_hz", str(frequencies.size), format_range(frequencies), "ITPC 数组第 1 维"),
                ("t_ms", str(times.size), format_range(times), "ITPC 数组第 2 维；相对刺激时间"),
            ]
            metadata_table.setRowCount(len(metadata))
            for row, row_values in enumerate(metadata):
                for column, text in enumerate(row_values):
                    metadata_table.setItem(row, column, QTableWidgetItem(str(text)))
            metadata_table.resizeColumnsToContents()

            selected_time = float(time_box.currentData()) if time_box.currentData() is not None else float(times[0])
            time_index = int(np.argmin(np.abs(times - selected_time)))
            value_table.setHorizontalHeaderLabels([
                "频率 Hz",
                "ITPC" if metric_box.currentIndex() == 0 else "相对参考窗变化",
                "有效 trial",
            ])
            value_table.setRowCount(frequencies.size)
            for row, frequency in enumerate(frequencies):
                value = values[row, time_index]
                value_table.setItem(row, 0, QTableWidgetItem(f"{frequency:.6g}"))
                value_table.setItem(row, 1, QTableWidgetItem("—" if not np.isfinite(value) else f"{value:.6f}"))
                value_table.setItem(row, 2, QTableWidgetItem(str(int(counts[row, time_index]))))
            value_table.resizeColumnsToContents()
            status.setText(
                f"当前 tag {result['tag']}、通道 ch{int(output['channel_ids'][channel_index])}，时间 {times[time_index]:.3f} ms；"
                f"完整数组维度为频率 × 时间 × 通道 = {tuple(full_itpc.shape)}。"
            )

        tag_box.currentIndexChanged.connect(render)
        channel_box.currentIndexChanged.connect(render)
        metric_box.currentIndexChanged.connect(render)
        time_box.currentIndexChanged.connect(render)
        render()
        dialog.exec()

    def export_lfp_itpc(self) -> None:
        output = getattr(self, "lfp_itpc_output", None)
        if not output or not output.get("results"):
            QMessageBox.information(self, "尚无 ITPC", "请先计算 ITPC。")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "导出 ITPC NPZ", str(Path.cwd() / "lfp_itpc.npz"), "NumPy archive (*.npz)")
        if not filename:
            return
        try:
            payload = {"frequencies_hz": output["frequencies_hz"], "t_ms": output["t_ms"], "channel_ids": output["channel_ids"], "sample_rate": output["sample_rate"], "cycles": output["cycles"], "min_trials": output["min_trials"]}
            for result in output["results"]:
                payload[f"tag_{result['tag']}_itpc"] = result["itpc"]
                payload[f"tag_{result['tag']}_itpc_baseline_delta"] = result["itpc_baseline_delta"]
                payload[f"tag_{result['tag']}_valid_trials"] = result["counts"]
            np.savez_compressed(filename, **payload)
            self.lfp_status.setText(f"ITPC 已导出：{filename}")
        except Exception as exc:
            QMessageBox.critical(self, "ITPC 导出失败", str(exc))

    @staticmethod
    def _task_metric_number(value, digits: int = 3) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        return f"{number:.{digits}f}" if np.isfinite(number) else "—"

    def _populate_lfp_task_result_controls(self, output: dict) -> None:
        """Load task tags/channels into the result panel without changing metrics."""
        self.lfp_task_result_output = output
        results = list(output.get("results") or [])
        channel_ids = [int(channel) for channel in output.get("channel_ids", [])]
        for box in (self.lfp_task_result_tag_box, self.lfp_task_result_channel_box):
            box.blockSignals(True)
            box.clear()
        for index, result in enumerate(results):
            self.lfp_task_result_tag_box.addItem(f"tag {int(result['tag'])}", index)
        for channel in channel_ids:
            self.lfp_task_result_channel_box.addItem(f"ch{channel}", channel)
        for box in (self.lfp_task_result_tag_box, self.lfp_task_result_channel_box):
            box.blockSignals(False)
        self.lfp_task_result_panel.setVisible(bool(results and channel_ids))

    def _select_lfp_task_result_table_channel(self, row: int, _column: int) -> None:
        item = self.lfp_task_result_table.item(row, 0)
        if item is None:
            return
        channel = item.data(Qt.ItemDataRole.UserRole)
        index = self.lfp_task_result_channel_box.findData(channel)
        if index >= 0:
            self.lfp_task_result_channel_box.setCurrentIndex(index)

    def _render_lfp_task_result_view(self, *_args) -> None:
        """Render one task tag as a table plus a deliberately single-purpose plot."""
        output = getattr(self, "lfp_task_result_output", None)
        if not output:
            return
        results = list(output.get("results") or [])
        tag_index = self.lfp_task_result_tag_box.currentData()
        if not results or tag_index is None or not (0 <= int(tag_index) < len(results)):
            return
        result = results[int(tag_index)]
        metrics = list(result.get("metrics") or [])
        selected_channel = self.lfp_task_result_channel_box.currentData()
        if selected_channel is None and metrics:
            selected_channel = int(metrics[0]["channel"])
        target_hz = float(result.get("target_freq_hz", np.nan))
        valid_count = sum(np.isfinite(float(row.get("local_tag_snr_db", np.nan))) for row in metrics)
        self.lfp_task_result_summary.setText(
            f"tag {int(result['tag'])} | 目标频率 {target_hz:g} Hz | 完整 trial {int(result.get('trials', 0))} | "
            f"至少一通道合格 trial {int(result.get('trials_with_usable_channels', 0))} | "
            f"有效 local tag-SNR 通道 {valid_count}/{len(metrics)}"
        )

        table = self.lfp_task_result_table
        table.setRowCount(len(metrics))
        for row_index, metric in enumerate(metrics):
            channel = int(metric["channel"])
            usable = int(metric.get("usable_trials", 0))
            total = int(metric.get("trials", result.get("trials", 0)))
            seconds = float(metric.get("spectral_seconds", np.nan))
            snr = float(metric.get("local_tag_snr_db", np.nan))
            if not np.isfinite(seconds) or seconds <= 0:
                state = "无连续干净片段可计算 PSD"
            elif not np.isfinite(snr):
                state = "目标频率或邻频噪声无有效值"
            else:
                state = "可用"
            values = (
                f"ch{channel}", f"{usable}/{total}", self._task_metric_number(seconds, 2),
                self._task_metric_number(metric.get("target_power")),
                self._task_metric_number(metric.get("target_power_change_db")),
                self._task_metric_number(metric.get("external_baseline_target_power_change_db")),
                self._task_metric_number(snr),
                f"{state}；{metric.get('external_baseline_status', '未启用实验前 baseline')}；"
                f"时域：{metric.get('external_time_status', '未准备时域静息参考')}",
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, channel)
                table.setItem(row_index, column, item)
        table.resizeColumnsToContents()
        selected_row = next((index for index, row in enumerate(metrics) if int(row["channel"]) == int(selected_channel)), -1)
        if selected_row >= 0:
            table.selectRow(selected_row)

        self.lfp_figure.clear()
        axis = self.lfp_figure.add_subplot(111)
        metric_key = self.lfp_task_result_metric_box.currentData() or "local_tag_snr_db"
        if metric_key == "wave":
            channel_ids = [int(channel) for channel in output.get("channel_ids", [])]
            try:
                channel_index = channel_ids.index(int(selected_channel))
            except ValueError:
                return
            wave = np.asarray(result.get("wave"), dtype=float)
            t_ms = np.asarray(result.get("t_ms"), dtype=float)
            if wave.ndim != 2 or channel_index >= wave.shape[1] or not t_ms.size:
                return
            axis.plot(t_ms, wave[:, channel_index], color="#1f77b4", linewidth=.9)
            axis.axvline(0, color="#777777", linestyle="--", linewidth=.7)
            axis.set_xlabel("相对刺激时间（ms）")
            axis.set_ylabel("幅值（mV）")
            axis.set_title(f"tag {int(result['tag'])}：ch{int(selected_channel)} 平均时域响应")
        elif metric_key == "external_time_response":
            response = (result.get("external_time_response") or {}).get("channels", {}).get(int(selected_channel))
            time_ms = np.asarray((result.get("external_time_response") or {}).get("time_ms", []), dtype=float)
            if response is None or not time_ms.size:
                axis.text(
                    .5, .5, "当前 tag/通道没有可用的实验前静息时域对照",
                    ha="center", va="center", coordinates="axes",
                )
                axis.set_axis_off()
            else:
                task_mean = np.asarray(response.get("task_mean"), dtype=float)
                task_sem = np.asarray(response.get("task_sem"), dtype=float)
                rest_mean = np.asarray(response.get("rest_mean"), dtype=float)
                rest_sem = np.asarray(response.get("rest_sem"), dtype=float)
                axis.plot(time_ms, task_mean, color="#1f77b4", linewidth=1.1, label=f"任务（n={response.get('task_trials', 0)}）")
                axis.fill_between(time_ms, task_mean - task_sem, task_mean + task_sem, color="#1f77b4", alpha=.16, linewidth=0)
                axis.plot(time_ms, rest_mean, color="#d84315", linewidth=1.1, label=f"静息伪 epoch（n={response.get('rest_epochs', 0)}）")
                axis.fill_between(time_ms, rest_mean - rest_sem, rest_mean + rest_sem, color="#d84315", alpha=.16, linewidth=0)
                for cluster in response.get("clusters", []):
                    if cluster.get("significant"):
                        left = time_ms[int(cluster["first"])]
                        right = time_ms[int(cluster["last"])]
                        axis.axvspan(left, right, color="#2e7d32", alpha=.16)
                axis.axhline(0, color="#777777", linestyle="--", linewidth=.7)
                axis.axvline(0, color="#777777", linestyle="--", linewidth=.7)
                axis.set_xlabel("相对刺激时间（ms）")
                axis.set_ylabel("基线校正幅值（mV）")
                axis.set_title(
                    f"tag {int(result['tag'])}：ch{int(selected_channel)} 任务 vs 实验前静息 "
                    f"({response.get('status', 'unknown')}；簇形成/显著性 p<0.01)"
                )
                axis.legend(loc="best", fontsize=8)
        else:
            labels = {
                "local_tag_snr_db": "local tag-SNR（dB）",
                "event_lfp_snr_db": "事件锁定 LFP SNR（dB）",
                "target_power_change_db": "相对刺激前 baseline 目标功率变化（dB）",
                "external_baseline_target_power_change_db": "相对实验前 baseline 目标功率变化（dB）",
                "external_baseline_local_snr_change_db": "相对实验前 baseline local SNR 变化（dB）",
                "target_power": "目标功率（mV²/Hz）",
            }
            channels = np.asarray([int(row["channel"]) for row in metrics], dtype=float)
            values = np.asarray([float(row.get(metric_key, np.nan)) for row in metrics], dtype=float)
            axis.plot(channels, values, color="#1f77b4", linewidth=.8, marker=".", markersize=5)
            selected = np.flatnonzero(channels == int(selected_channel))
            if selected.size and np.isfinite(values[selected[0]]):
                axis.plot([channels[selected[0]]], [values[selected[0]]], color="#d84315", marker="o", markersize=7)
            if metric_key in {
                "local_tag_snr_db", "event_lfp_snr_db", "target_power_change_db",
                "external_baseline_target_power_change_db", "external_baseline_local_snr_change_db",
            }:
                axis.axhline(0.0, color="#777777", linestyle="--", linewidth=.7)
            axis.set_xlabel("通道号")
            axis.set_ylabel(labels[metric_key])
            axis.set_title(f"tag {int(result['tag'])}：{labels[metric_key]}")
        axis.grid(alpha=.25)
        self.lfp_canvas.draw_idle()

    def _render_lfp_task_quality_summary(self, mode: str, output: dict) -> None:
        """Show page-4 LFP task QC/PSD results without Letter marker browser lines."""
        results = list(output.get("results") or [])
        channel_ids = np.asarray(output.get("channel_ids", []), dtype=int)
        if not results or channel_ids.size == 0:
            raise ValueError("没有可显示的 LFP 五窗任务结果。")

        summary_lines = [
            f"{mode} LFP 五窗任务分析完成",
            "质量控制：稳态区间 0.5-5.5 s 被划分为 5 个连续 1 s 窗口：W1 0.5-1.5、W2 1.5-2.5、W3 2.5-3.5、W4 3.5-4.5、W5 4.5-5.5 s。",
            "纳入规则：每个 通道 x trial 至少 3 个干净窗口才进入任务响应聚合。",
            "PSD 规则：仅对连续且长度不少于 2 s 的干净窗口段计算 Welch PSD；不拼接不连续窗口。",
        ]
        plot_channels, plot_snr = [], []
        for result in results:
            metrics = list(result.get("metrics") or [])
            target_hz = float(result.get("target_freq_hz", np.nan))
            usable = np.asarray(result.get("usable_trials_per_channel", []), dtype=int)
            spectral_seconds = np.asarray(result.get("spectral_samples_per_channel", []), dtype=float) / float(output["fs"])
            valid_snr = [row for row in metrics if np.isfinite(row.get("local_tag_snr_db", np.nan))]
            summary_lines.append(
                f"\ntag {result['tag']}：完整 trial {result['trials']}，至少一个合格通道的 trial {result.get('trials_with_usable_channels', 0)}，"
                f"目标频率 {target_hz:g} Hz，有效 tag-SNR 通道 {len(valid_snr)}/{len(metrics)}。"
            )
            for index, row in enumerate(metrics):
                channel = int(row["channel"])
                snr = float(row.get("local_tag_snr_db", np.nan))
                change = float(row.get("target_power_change_db", np.nan))
                event_snr = float(row.get("event_lfp_snr_db", np.nan))
                summary_lines.append(
                    f"  ch{channel}: 可用/拒绝 trial {int(usable[index]) if index < usable.size else row['usable_trials']}/{row['rejected_trials']}，"
                    f"完整可用/干净窗 {row.get('available_windows', 0)}/{row['clean_windows']}，短 epoch/仅响应 trial {row.get('short_epoch_trials', 0)}/{row.get('response_only_trials', 0)}"
                    f"（不可用/饱和/平坦/跳变：{row['unavailable_windows']}/{row['saturated_windows']}/{row['flat_windows']}/{row['jump_windows']}），"
                    f"连续 PSD {float(spectral_seconds[index]) if index < spectral_seconds.size else row['spectral_seconds']:.2f} s，"
                    f"功率变化 {change:.3f} dB，local tag-SNR {snr:.3f} dB，"
                    f"事件锁定 LFP SNR {event_snr:.3f} dB；"
                    f"时域任务/静息 {row.get('external_time_task_trials', 0)}/{row.get('external_time_rest_epochs', 0)}，"
                    f"显著簇 {row.get('external_time_significant_clusters', 0)}（{row.get('external_time_status', '未准备时域静息参考')}）"
                )
                if np.isfinite(snr):
                    plot_channels.append(channel)
                    plot_snr.append(snr)
        self.lfp_results.setPlainText("\n".join(summary_lines))

        self._populate_lfp_task_result_controls(output)
        self._render_lfp_task_result_view()

    def _show_letter_behavior_browser(self, stage, output) -> None:
        """Letter channel browser: old long-window markers plus EndFrame."""
        results = list(output.get("results") or [])
        ids = [int(value) for value in output["channel_ids"]]
        matches = getattr(self, "behavior_matches", {})
        # The old GUI obtains the Letter tag selector from *all detected
        # markers*, rather than from the subset for which the short task
        # epoch happened to fit inside the loaded recording.  In particular,
        # a final tag-7 marker can be detected correctly but have no complete
        # -500--6000 ms epoch; it must still be visible to the user.
        marker_rows = np.asarray(getattr(self, "stim_markers", []), dtype=int)
        available_tags = sorted({
            int(row[1]) for row in marker_rows
            if len(row) >= 2 and int(row[1]) in {4, 5, 6, 7}
        })
        result_by_tag = {int(item["tag"]): item for item in results}
        if not available_tags:
            # Keep this fallback for manually constructed task output, while
            # normal operation always uses stim_markers above.
            available_tags = sorted(result_by_tag)
        if not available_tags:
            QMessageBox.information(self, "没有 Letter tag", "当前 marker 中没有检测到 4、5、6 或 7 tag。")
            return
        window = QMainWindow(self)
        window.setWindowTitle("Letter 视频检测关联通道浏览")
        window.resize(1350, 900)
        root = QWidget(); layout = QVBoxLayout(root); window.setCentralWidget(root)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("tag"))
        tag_combo = QComboBox(window)
        for tag in available_tags:
            trial_count = sum(1 for row in marker_rows if len(row) >= 2 and int(row[1]) == tag)
            suffix = "" if tag in result_by_tag else "（无完整 epoch）"
            tag_combo.addItem(f"{tag} tag（检测到 {trial_count} 个）{suffix}", tag)
        controls.addWidget(tag_combo)
        previous = QPushButton("上一页"); following = QPushButton("下一页")
        page_label = QLabel(); controls.addWidget(previous); controls.addWidget(following); controls.addWidget(page_label, 1)
        timing_button = QPushButton("显示 marker / 视频检测时间表")
        controls.addWidget(timing_button); layout.addLayout(controls)
        frame_rate = getattr(self, "behavior_detection_frame_rate", np.nan)
        offset = getattr(self, "behavior_detection_video_eeg_offset_sec", np.nan)
        info = QLabel(f"红虚线=Letter onset；绿点线=StartFrame；橙虚线=EndFrame；紫点划线=有效 AnimalStart。首个 Letter trial 校准；帧率={frame_rate:g} Hz；视频-EEG 偏移={offset:+.6f} s")
        info.setWordWrap(True); layout.addWidget(info)
        grid_holder = QWidget(window); grid = QGridLayout(grid_holder); grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(5)
        layout.addWidget(grid_holder, 1)
        state = {"tag": int(available_tags[0]), "page": 0}
        page_size = max(1, int(float(self._widget_text(self.tvep_page_size_var, "12") or 12)))

        # Creating PlotWidget/OpenGL scene objects is substantially more
        # expensive than replacing their data.  Keep one fixed set of page
        # slots and reuse it for tag/page changes.  This preserves the exact
        # plots and layout while avoiding repeated QObject/scene teardown.
        page_plots = []
        for local in range(page_size):
            plot = PyQtGraphPlot(grid_holder)
            plot.setMinimumSize(280, 190)
            plot._letter_channel_index = None
            plot._plot_widget.scene().sigMouseClicked.connect(
                lambda event, item=plot: (
                    show_trials(int(item._letter_channel_index), result_by_tag.get(int(state["tag"])))
                    if event.button() == Qt.MouseButton.LeftButton
                    and item._letter_channel_index is not None
                    and result_by_tag.get(int(state["tag"])) is not None
                    else None
                )
            )
            grid.addWidget(plot, local // min(4, page_size), local % min(4, page_size))
            page_plots.append(plot)
        window._letter_page_plots = page_plots
        window._letter_page_state = state

        def line_values(result):
            visual, end_values, animal = [], [], []
            t_sec = np.asarray(result["t_ms"], dtype=float) / 1000.0
            for sample in result["used_marker_samples"]:
                event = matches.get(int(sample))
                if not event:
                    continue
                value = float(event.get("visual_start_latency_sec", np.nan))
                if np.isfinite(value) and t_sec[0] <= value <= t_sec[-1]: visual.append(value)
                value = float(event.get("end_latency_sec", np.nan))
                if np.isfinite(value) and t_sec[0] <= value <= t_sec[-1]: end_values.append(value)
                value = float(event.get("animal_latency_sec", np.nan))
                if event.get("validity") == "Y" and np.isfinite(value) and t_sec[0] <= value <= t_sec[-1]: animal.append(value)
            return visual, end_values, animal

        def show_trials(channel_index: int, result):
            self._show_letter_behavior_trial_grid(stage, output, result, channel_index, matches)

        def render():
            grid_holder.setUpdatesEnabled(False)
            for plot in page_plots:
                plot.hide()
                plot._letter_channel_index = None
            tag = int(state["tag"])
            result = result_by_tag.get(tag)
            if result is None:
                marker_count = sum(1 for row in marker_rows if len(row) >= 2 and int(row[1]) == tag)
                missing = QLabel(
                    f"检测到了 {tag} tag（{marker_count} 个 marker），但当前数据段内没有可用于\n"
                    f"任务 epoch 的完整 trial，因此不能绘制平均波形。\n"
                    "这通常发生在 marker 靠近已加载数据段的开头或结尾；tag 没有丢失。"
                )
                missing.setAlignment(Qt.AlignmentFlag.AlignCenter)
                missing.setWordWrap(True)
                if not hasattr(window, "_missing_label"):
                    window._missing_label = missing
                    grid.addWidget(window._missing_label, 0, 0, 1, min(4, page_size))
                else:
                    window._missing_label.setText(missing.text())
                window._missing_label.show()
                page_label.setText(f"tag {tag} | 无完整 epoch（检测到 {marker_count} 个 marker）")
                previous.setEnabled(False); following.setEnabled(False)
                grid_holder.setUpdatesEnabled(True)
                return
            if hasattr(window, "_missing_label"):
                window._missing_label.hide()
            total_pages = max(1, int(np.ceil(len(ids) / page_size)))
            state["page"] = max(0, min(state["page"], total_pages - 1))
            start = state["page"] * page_size; end = min(len(ids), start + page_size); count = end - start
            columns = min(4, max(1, count)); visual, end_values, animal = line_values(result)
            t_sec = np.asarray(result["t_ms"], dtype=float) / 1000.0
            for local, channel_index in enumerate(range(start, end)):
                plot = page_plots[local]
                grid.addWidget(plot, local // columns, local % columns)
                plot.clear()
                plot._letter_channel_index = channel_index
                axis = plot.add_subplot(111)
                axis.plot(t_sec, result["wave"][:, channel_index], color="#111111", linewidth=1.0)
                axis.axvline(0, color="#d62728", linestyle="--", linewidth=1.2)
                for value in visual: axis.axvline(value, color="#2ca02c", linewidth=1.0)
                for value in end_values: axis.axvline(value, color="#ff7f0e", linestyle="--", linewidth=1.0)
                for value in animal: axis.axvline(value, color="#9467bd", linewidth=1.1)
                axis.set_title(f"ch{ids[channel_index]}"); axis.set_xlabel("相对 Letter 时间 (s)"); axis.set_ylabel("幅值(μV)"); axis.grid(alpha=.2)
                plot.show()
            page_label.setText(f"tag {result['tag']} | 第 {state['page'] + 1}/{total_pages} 页，通道 {start + 1}-{end}/{len(ids)}。红=Letter，绿=StartFrame，橙=EndFrame，紫=有效 AnimalStart；点击通道查看逐 trial 图。")
            previous.setEnabled(state["page"] > 0); following.setEnabled(state["page"] < total_pages - 1)
            grid_holder.setUpdatesEnabled(True)

        def show_timing_table():
            self._show_letter_behavior_timing_table(window, output, matches, frame_rate)

        previous.clicked.connect(lambda: (state.update(page=state["page"] - 1), render()))
        following.clicked.connect(lambda: (state.update(page=state["page"] + 1), render()))
        tag_combo.currentIndexChanged.connect(lambda index: (state.update(tag=int(tag_combo.itemData(index)), page=0), render()))
        timing_button.clicked.connect(show_timing_table)
        window._letter_render = render
        render()
        window.show(); self._letter_behavior_window = window

    def _load_letter_channel_trials(self, source, output, result, channel_index):
        """Load and LRU-cache one channel's baseline-corrected Letter epochs."""
        meta = source.metadata; fs = float(meta.fs)
        source_column = int(np.asarray(output.get("columns", []), dtype=int)[channel_index])
        marker_offset = int(round(float(meta.time_offset) * fs))
        t_sec = np.asarray(result["t_ms"], dtype=float) / 1000.0
        start = int(round(float(result["t_ms"][0]) * fs / 1000.0))
        end = start + int(t_sec.size)
        baseline_start = float(self._widget_text(self.baseline_start_ms_var, "-200"))
        baseline_end = float(self._widget_text(self.baseline_end_ms_var, "0"))
        baseline = (t_sec * 1000 >= baseline_start) & (t_sec * 1000 <= baseline_end)
        cache_key = (
            id(source), id(getattr(source, "data", None)), int(meta.rows), fs,
            float(meta.time_offset), source_column, int(result["tag"]), start, end,
            baseline_start, baseline_end,
            tuple(int(sample) for sample in result["used_marker_samples"]),
        )
        cache = getattr(self, "_letter_trial_cache", None)
        if cache is None:
            cache = self._letter_trial_cache = OrderedDict()
            self._letter_trial_cache_bytes = 0
        cached = cache.pop(cache_key, None)
        if cached is not None:
            cache[cache_key] = cached
            return t_sec, cached[0], cached[1]

        ranges, samples = [], []
        for sample in result["used_marker_samples"]:
            center = int(sample) - marker_offset
            first, last = center + start, center + end
            if first < 0 or last > meta.rows:
                continue
            ranges.append((first, last)); samples.append(int(sample))
        if hasattr(source, "read_many"):
            loaded = source.read_many(ranges, source_column)
        else:
            loaded = [source.read(first, last, source_column) for first, last in ranges]
        trials = []
        for values in loaded:
            values = np.asarray(values, dtype=np.float32).ravel()
            if np.any(baseline):
                values = values - np.nanmean(values[baseline])
            trials.append(values)
        trial_array = np.stack(trials) if trials else np.empty((0, t_sec.size), dtype=np.float32)
        entry_bytes = int(trial_array.nbytes)
        if entry_bytes > 128 * 1024 * 1024:
            return t_sec, trial_array, tuple(samples)
        cache[cache_key] = (trial_array, tuple(samples), entry_bytes)
        self._letter_trial_cache_bytes += entry_bytes
        # A strict byte cap prevents high-rate recordings from retaining an
        # unbounded number of channel x trial arrays.
        while len(cache) > 12 or self._letter_trial_cache_bytes > 128 * 1024 * 1024:
            _old_key, old_value = cache.popitem(last=False)
            self._letter_trial_cache_bytes -= int(old_value[2])
        return t_sec, trial_array, tuple(samples)

    def _show_letter_behavior_trial_grid(self, stage, output, result, channel_index, matches) -> None:
        """On-demand, one-channel counterpart of Tk's long trial grid."""
        source = (self.lfp_source if stage == "lfp" else self.spike_source) or self.active_source
        ids = output["channel_ids"]
        t_sec, trials, samples = self._load_letter_channel_trials(source, output, result, channel_index)
        if trials.shape[0] == 0:
            QMessageBox.warning(self, "无完整 trial", "此通道没有可显示的完整长行为 epoch。")
            return
        window = QMainWindow(self); window.setWindowTitle(f"ch{ids[channel_index]} 单 trial - tag {result['tag']}"); window.resize(1350, 900)
        root = QWidget(); layout = QVBoxLayout(root); window.setCentralWidget(root)
        layout.addWidget(QLabel("每格一个 trial：红=Letter onset，绿=StartFrame，橙=EndFrame，紫=有效 AnimalStart。"))
        pager = QHBoxLayout(); previous = QPushButton("上一页"); following = QPushButton("下一页"); page_label = QLabel()
        pager.addWidget(previous); pager.addWidget(following); pager.addWidget(page_label); pager.addStretch(1); layout.addLayout(pager)
        holder = QWidget(window); grid = QGridLayout(holder); grid.setContentsMargins(0, 0, 0, 0); grid.setSpacing(5); layout.addWidget(holder, 1)
        page_size = 20; columns = 5; state = {"page": 0}
        plots = []
        for local in range(min(page_size, trials.shape[0])):
            plot = PyQtGraphPlot(holder); plot.setMinimumSize(225, 155)
            grid.addWidget(plot, local // columns, local % columns); plots.append(plot)
        window._letter_trial_plots = plots
        window._letter_trial_state = state

        def render_trials():
            holder.setUpdatesEnabled(False)
            total_pages = max(1, int(np.ceil(trials.shape[0] / page_size)))
            state["page"] = max(0, min(state["page"], total_pages - 1))
            first = state["page"] * page_size; last = min(trials.shape[0], first + page_size)
            for plot in plots: plot.hide()
            for local, index in enumerate(range(first, last)):
                plot = plots[local]; plot.clear(); axis = plot.add_subplot(111)
                axis.plot(t_sec, trials[index], color="#111111", linewidth=.7); axis.axvline(0, color="#d62728", linestyle="--", linewidth=1.1)
                event = matches.get(samples[index], {})
                visual = float(event.get("visual_start_latency_sec", np.nan)); end_latency = float(event.get("end_latency_sec", np.nan)); animal = float(event.get("animal_latency_sec", np.nan))
                if np.isfinite(visual): axis.axvline(visual, color="#2ca02c", linewidth=1.0)
                if np.isfinite(end_latency): axis.axvline(end_latency, color="#ff7f0e", linestyle="--", linewidth=1.0)
                if event.get("validity") == "Y" and np.isfinite(animal): axis.axvline(animal, color="#9467bd", linewidth=1.1)
                axis.set_title(f"trial {index + 1}"); axis.set_xlabel("s"); axis.set_ylabel("幅值(μV)"); axis.grid(alpha=.2); plot.show()
            page_label.setText(f"第 {state['page'] + 1}/{total_pages} 页，trial {first + 1}-{last}/{trials.shape[0]}")
            previous.setEnabled(state["page"] > 0); following.setEnabled(state["page"] < total_pages - 1)
            holder.setUpdatesEnabled(True)

        previous.clicked.connect(lambda: (state.update(page=state["page"] - 1), render_trials()))
        following.clicked.connect(lambda: (state.update(page=state["page"] + 1), render_trials()))
        window._letter_trial_render = render_trials
        render_trials()
        window.show(); self._letter_trial_window = window

    def _show_letter_behavior_timing_table(self, parent, output, matches, frame_rate) -> None:
        """Timing table and alignment summary from Tk's behavior browser."""
        markers = sorted((row for row in np.asarray(self.stim_markers, dtype=int) if int(row[1]) in {4, 5, 6, 7}), key=lambda row: int(row[0]))
        window = QMainWindow(parent); window.setWindowTitle("Letter marker / 视频检测时间表"); window.resize(1320, 650)
        root = QWidget(); layout = QVBoxLayout(root); window.setCentralWidget(root)
        visual_offsets = [float(event.get("visual_start_latency_sec", np.nan)) * 1000 for event in matches.values()]
        finite = np.asarray([value for value in visual_offsets if np.isfinite(value)], dtype=float)
        matched_count = sum(int(row[0]) in matches for row in markers); rate = matched_count / max(1, len(markers))
        if finite.size < 3 or rate < .8: verdict = "对齐检查：数据不足，暂不判断。"
        elif np.std(finite) <= 50 and abs(np.mean(finite)) <= 50: verdict = "对齐检查：良好，StartFrame 相对 Letter marker 接近且稳定。"
        elif np.std(finite) <= 50: verdict = f"对齐检查：稳定但有系统偏移，约 {np.mean(finite):+.1f} ms。"
        else: verdict = f"对齐检查：需检查抖动，StartFrame 的 SD 为 {np.std(finite):.1f} ms。"
        layout.addWidget(QLabel(f"帧率：{frame_rate:g} Hz | Letter markers：{len(markers)} | 已匹配：{matched_count}\nStartFrame 相对 Letter marker：n={finite.size}，均值={np.mean(finite) if finite.size else np.nan:.2f} ms，SD={np.std(finite) if finite.size else np.nan:.2f} ms\n{verdict}"))
        table = QTableWidget(len(markers), 11, window); table.setHorizontalHeaderLabels(["全局 trial", "tag 内 trial", "tag", "marker sample", "marker s", "StartFrame", "StartFrame 偏移 ms", "EndFrame", "EndFrame 延迟 ms", "有效性", "AnimalStart 延迟 ms"])
        table.setUpdatesEnabled(False)
        table.setSortingEnabled(False)
        tag_trials = {}
        for row_index, marker in enumerate(markers):
            sample, tag = int(marker[0]), int(marker[1]); tag_trials[tag] = tag_trials.get(tag, 0) + 1; event = matches.get(sample, {})
            values = [row_index + 1, tag_trials[tag], tag, sample, f"{sample / float(output.get('fs', self.active_source.metadata.fs)):.6f}", event.get("start_frame", "NA"), f"{float(event.get('visual_start_latency_sec', np.nan)) * 1000:.3f}", event.get("end_frame", "NA"), f"{float(event.get('end_latency_sec', np.nan)) * 1000:.3f}", event.get("validity", "unmatched"), f"{float(event.get('animal_latency_sec', np.nan)) * 1000:.3f}"]
            for column, value in enumerate(values): table.setItem(row_index, column, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents(); table.setUpdatesEnabled(True); layout.addWidget(table, 1)
        window.show(); self._letter_timing_window = window

    def _fail_task_epoch_analysis(self, stage, error) -> None:
        self._set_task_buttons_enabled(stage, True)
        self._update_task_progress(stage, 0, f"任务分析失败：{error}"); QMessageBox.critical(self, "任务分析失败", error)

    def choose_remap_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择通道映射 Excel",
            self.remap_edit.text() or str(Path.cwd()),
            "Excel files (*.xlsx *.xls);;All files (*.*)",
        )
        if filename:
            self.remap_edit.setText(filename)

    def _selected_preprocess_columns(self, source=None):
        """Turn comma/range channel IDs into current-source column indexes."""
        text = self.preprocess_filter_channels_var.text().strip()
        if not text:
            return None
        try:
            requested = set()
            for part in text.replace("，", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    first, last = (int(value.strip()) for value in part.split("-", 1))
                    requested.update(range(min(first, last), max(first, last) + 1))
                else:
                    requested.add(int(part))
            source = source or (self.active_source if self.active_source.loaded else self.source)
            ids = np.asarray(source.metadata.channel_ids, dtype=np.int64)
            columns = np.flatnonzero(np.isin(ids, sorted(requested)))
            if not columns.size:
                raise ValueError("指定通道不在当前已加载数据内。")
            return columns
        except ValueError as exc:
            raise ValueError("指定通道请使用如 1,5,20-26 的格式。") from exc

    def _selected_bad_check_columns(self, source):
        """Resolve the explicit analysis scope against an unfiltered source."""
        return self._selected_preprocess_columns(source)

    def _detection_source(self):
        """Return the raw/remapped source used by channel-quality metrics.

        Filter, ICA and CAR outputs deliberately never become an implicit
        input to baseline, raw-PSD or saturation checks.
        """
        remap_path = self.remap_edit.text().strip() if hasattr(self, "remap_edit") else ""
        current_remap_signature = self._remap_signature(remap_path) if remap_path else None
        if (
            remap_path and self.remapped_source is not None and self.remapped_source.loaded
            and current_remap_signature == self._remapped_detection_signature
        ):
            return self.remapped_source
        if self.source is not None and self.source.loaded:
            return self.source
        return None

    @staticmethod
    def _remap_signature(remap_path):
        path = Path(remap_path)
        stat = path.stat()
        return str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)

    @staticmethod
    def _filter_cache_signature(source, remap_path, mode, low, high, hp_order, lp_order, notch, notch_harmonics, columns):
        """Identify the one reusable user-requested filter result."""
        remap_signature = None
        if remap_path:
            remap_signature = QtAnalysisGUI._remap_signature(remap_path)
        selected = None if columns is None else tuple(int(value) for value in np.asarray(columns, dtype=np.int64))
        return (
            "filter_result", AnalysisCache.source_key(source), remap_signature,
            str(mode), float(low), float(high), int(hp_order), int(lp_order),
            bool(notch), int(notch_harmonics), selected,
        )

    @staticmethod
    def _bad_channel_result_cache_key(source, columns, settings: dict) -> tuple:
        meta = source.metadata
        selected = (
            np.arange(meta.channels, dtype=np.int64)
            if columns is None else np.unique(np.asarray(columns, dtype=np.int64).ravel())
        )
        channel_ids = tuple(int(np.asarray(meta.channel_ids, dtype=np.int64)[index]) for index in selected)
        return (
            "bad_channel_result",
            AnalysisCache.source_key(source),
            channel_ids,
            tuple(sorted(
                (str(key), str(value)) for key, value in settings.items()
                if key not in {"parallel", "workers"}
            )),
        )

    @staticmethod
    def _format_physical_channel_ids(channel_ids) -> str:
        """Compact an ordered physical-channel set without hiding any IDs."""
        values = sorted({int(value) for value in channel_ids})
        if not values:
            return "无"
        groups = []
        first = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue
            groups.append(str(first) if first == previous else f"{first}-{previous}")
            first = previous = value
        groups.append(str(first) if first == previous else f"{first}-{previous}")
        return ", ".join(groups)

    def _preprocess_filter_scope_text(self, channel_ids, total_channels: int, *, completed: bool = False) -> str:
        ids = sorted({int(value) for value in channel_ids})
        total = max(0, int(total_channels))
        prefix = "实际已滤波通道" if completed else "本次将滤波通道"
        if len(ids) == total and total > 0:
            return f"{prefix}：全部 {total} 个物理通道（{self._format_physical_channel_ids(ids)}）。"
        return (
            f"{prefix}：{len(ids)}/{total} 个物理通道（{self._format_physical_channel_ids(ids)}）。"
            " 未列出的通道保持原始值，未执行本次滤波。"
        )

    def run_preprocess_all_channels(self) -> None:
        previous = self.preprocess_filter_channels_var.text()
        self.preprocess_filter_channels_var.clear()
        self.run_preprocess()
        # The worker has already copied settings, so restoring the previous
        # selection only affects the next explicit user action.
        self.preprocess_filter_channels_var.setText(previous)

    def _sync_dual_stream_export_enabled(self) -> None:
        """Disable streaming only while its worker is running."""
        worker = getattr(self, "_dual_branch_stream_worker", None)
        running = worker is not None and worker.isRunning()
        if hasattr(self, "dual_stream_export_button"):
            self.dual_stream_export_button.setEnabled(not running)
            self.dual_stream_export_button.setToolTip(
                "从重映射后的原始数据（未重映射时为最初加载数据）按流式专用参数生成 "
                "LFP/Spike H5；不继承普通滤波、CAR 或 ICA 结果。"
            )
    def _has_reusable_one_click_remap(self, remap_path: str) -> bool:
        """Return whether the current working data already has this remap."""
        if not remap_path:
            return False
        remapped_source = getattr(self, "remapped_source", None)
        if remapped_source is not None and remapped_source.loaded:
            try:
                return self._remapped_detection_signature == self._remap_signature(remap_path)
            except OSError:
                return False
        if remapped_source is None:
            # A reloaded processed H5 has no separate in-memory raw/remapped
            # cache. Its provenance is authoritative.
            source = getattr(self, "source", None)
            loaded_processed = getattr(self, "preprocessed_source", None) is source
            return bool(
                loaded_processed
                and source is not None
                and source.metadata is not None
                and "channel_remap" in self._preprocess_operation_names(source.metadata)
            )
        return False

    def _update_one_click_plan_summary(self, *_args) -> None:
        """Show the one-click execution plan implied by the live controls."""
        label = getattr(self, "one_click_plan_summary", None)
        if label is None:
            return

        flow = []
        remap_path = self.remap_edit.text().strip()
        has_any_processing = any((
            self.one_click_preprocess_check.isChecked(),
            self.one_click_quality_check.isChecked(),
            self.one_click_car_check.isChecked(),
            self.motion_ica_enable_var.isChecked(),
        ))
        if remap_path and has_any_processing:
            flow.append(
                "复用已有重映射"
                if self._has_reusable_one_click_remap(remap_path)
                else "重映射"
            )
        if self.one_click_quality_check.isChecked():
            flow.append("坏道检查")
        if self.one_click_car_check.isChecked():
            flow.append("CAR")
        if self.one_click_preprocess_check.isChecked():
            flow.append("50 Hz 工频陷波（仅未处理通道）")
        if self.motion_ica_enable_var.isChecked():
            flow.append("ICA")

        flow_text = " → ".join(flow) if flow else "未选择执行步骤"
        data_state = getattr(
            self,
            "_preprocess_data_state_text",
            "数据状态：尚未加载，未进行重映射或滤波。",
        ).strip()
        label.setText(
            f"本次流程：{flow_text}\n"
            f"CAR：{'开启' if self.one_click_car_check.isChecked() else '关闭'}｜"
            f"50 Hz 工频陷波：{'开启（自动跳过已处理通道）' if self.one_click_preprocess_check.isChecked() else '关闭'}｜"
            "普通高通/低通：不由一键流程执行（请使用上方“执行滤波”）｜"
            f"ICA：{'开启' if self.motion_ica_enable_var.isChecked() else '关闭'}\n"
            f"{data_state}"
        )

    def run_one_click_preprocess(self) -> None:
        """Run the selected novice-facing preprocessing steps in sequence."""
        if getattr(self, "_one_click_pipeline", None) is not None:
            return
        if not self.source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先在“数据导入”页加载 HDF5 数据文件。")
            return
        if self.one_click_car_check.isChecked():
            self.one_click_quality_check.setChecked(True)
        steps = []
        remap_path = self.remap_edit.text().strip()
        # Remapping defines physical channel positions and must precede every
        # automatic quality classification.  Reuse an already completed
        # remap, however: a later quality-only run must not replace the
        # current filtered source with a newly remapped, unfiltered source.
        reusable_remap = self._has_reusable_one_click_remap(remap_path)
        if remap_path and any((
            self.one_click_preprocess_check.isChecked(),
            self.one_click_quality_check.isChecked(),
            self.one_click_car_check.isChecked(),
            self.motion_ica_enable_var.isChecked(),
        )) and not reusable_remap:
            steps.append("remap")
        if self.one_click_quality_check.isChecked():
            quality_enabled = any((
                self.bad_check_var.isChecked(),
                self.bad_fast_artifact_check_var.isChecked(),
                self.bad_high_frequency_noise_check_var.isChecked(),
            ))
            if not quality_enabled:
                QMessageBox.warning(self, "未选择检测", "请在“显示参数”中至少启用一项通道质量检查。")
                return
            steps.append("quality")
        if self.one_click_car_check.isChecked():
            steps.append("car")
        if self.one_click_preprocess_check.isChecked():
            steps.append("preprocess")
        if self.motion_ica_enable_var.isChecked():
            steps.append("ica")
        if not steps:
            QMessageBox.information(self, "未选择步骤", "请至少勾选一个要执行的处理步骤。")
            return
        requested = set()
        if "remap" in steps:
            requested.add("channel_remap")
        if "preprocess" in steps:
            requested.add("filter")
        if "car" in steps:
            requested.add("reference")
        if "ica" in steps:
            requested.add("ica")
        if not self._confirm_repeated_preprocessing(requested, "一键预处理"):
            return
        self._one_click_pipeline = {
            "steps": steps,
            "requested_steps": tuple(steps),
            "waiting": None,
            "started_at": perf_counter(),
            "step_started_at": None,
            "timings": [],
        }
        self.one_click_preprocess_button.setEnabled(False)
        self._continue_one_click_preprocess()

    def _continue_one_click_preprocess(self) -> None:
        pipeline = getattr(self, "_one_click_pipeline", None)
        if pipeline is None or pipeline["waiting"] is not None:
            return
        if not pipeline["steps"]:
            self._finish_one_click_preprocess()
            return
        step = pipeline["steps"].pop(0)
        pipeline["waiting"] = step
        pipeline["step_started_at"] = perf_counter()
        if step == "remap":
            self.run_preprocess(force_filter_off=True)
            if (
                self._one_click_pipeline is pipeline
                and pipeline["waiting"] == step
                and (self._preprocess_worker is None or not self._preprocess_worker.isRunning())
            ):
                self._stop_one_click_preprocess()
        elif step == "quality":
            self.run_selected_analysis()
            if (
                self._one_click_pipeline is pipeline
                and pipeline["waiting"] == step
                and self._selected_analysis_batch is None
            ):
                self._stop_one_click_preprocess()
        elif step == "car":
            if len(getattr(self, "good_channel_ids", set())) < 2:
                self.preprocess_status.setText("自动 CAR 已跳过：合格通道不足两个。")
                self._advance_one_click_preprocess("car")
                return
            self.run_leave_one_out_median_car()
            if (
                self._one_click_pipeline is pipeline
                and pipeline["waiting"] == step
                and (self._car_worker is None or not self._car_worker.isRunning())
            ):
                self._stop_one_click_preprocess()
        elif step == "ica":
            source = self.preprocessed_source or self.remapped_source or self.active_source
            eligible_ids = sorted(self._ica_eligible_channel_ids(source))
            if len(eligible_ids) < 2:
                self.preprocess_status.setText("自动 ICA 已跳过：排除坏道后可用通道不足两个。")
                self._advance_one_click_preprocess("ica")
                return
            self._run_ica_for_channel_ids(source, eligible_ids)
            if (
                self._one_click_pipeline is pipeline
                and pipeline["waiting"] == step
                and (
                    getattr(self, "_ica_worker", None) is None
                    or not self._ica_worker.isRunning()
                )
            ):
                self._stop_one_click_preprocess()
    def _advance_one_click_preprocess(self, completed_step: str) -> None:
        pipeline = getattr(self, "_one_click_pipeline", None)
        if pipeline is None or pipeline.get("waiting") != completed_step:
            return
        step_started_at = pipeline.get("step_started_at")
        if step_started_at is not None:
            labels = {
                "remap": "通道重映射",
                "quality": "通道质量/坏道检查",
                "car": "Leave-one-out median CAR",
                "preprocess": "50 Hz 工频陷波",
                "ica": "ICA",
            }
            pipeline["timings"].append((
                labels.get(completed_step, completed_step),
                perf_counter() - step_started_at,
            ))
            if completed_step == "quality":
                for name, elapsed in getattr(self, "last_bad_channel_stage_timings", []):
                    if name != "总计":
                        pipeline["timings"].append((f"  - {name}", elapsed))
        pipeline["step_started_at"] = None
        pipeline["waiting"] = None
        QTimer.singleShot(0, self._continue_one_click_preprocess)

    def _finish_one_click_preprocess(self) -> None:
        pipeline = self._one_click_pipeline
        if pipeline is not None:
            timings = list(pipeline.get("timings", []))
            completed_steps = {
                name for name, _ in timings if not name.startswith("  - ")
            }
            for name in ("通道重映射", "通道质量/坏道检查", "Leave-one-out median CAR", "50 Hz 工频陷波", "ICA"):
                if name not in completed_steps:
                    timings.append((f"{name}（未执行）", 0.0))
            timings.append(("一键流程总计", perf_counter() - pipeline["started_at"]))
            self.last_one_click_stage_timings = timings
        self._one_click_pipeline = None
        self.one_click_preprocess_button.setEnabled(True)
        self.statusBar().showMessage("已完成一键预处理。")

    def _stop_one_click_preprocess(self) -> None:
        if getattr(self, "_one_click_pipeline", None) is None:
            return
        self._one_click_pipeline = None
        self.one_click_preprocess_button.setEnabled(True)

    def open_remapped_overview(self) -> None:
        source = self.remapped_source or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未处理", "请先完成通道重映射。")
            return
        start_sec, end_sec, full_duration = self.preprocessed_preview.selected_time_window_seconds()
        self._show_overview(
            source,
            start_sec,
            end_sec,
            full_duration=full_duration,
            window_attr="_remapped_overview_window",
        )

    def run_bad_channel_check(self) -> None:
        self._run_bad_channel_worker()

    def run_fast_artifact_check(self) -> None:
        """Run only the full-record bottom saturation check."""
        self._run_bad_channel_worker(fast_artifact_only=True)

    def run_high_frequency_noise_check(self) -> None:
        """Run only the inexpensive concentration check near the 2.5mV level."""
        self._run_bad_channel_worker(high_frequency_noise_only=True, allow_when_disabled=True)

    def run_selected_analysis(self) -> None:
        """Run all enabled quality checks from one raw/remapped channel cache."""
        if self._selected_analysis_batch is not None:
            return
        source = self._detection_source()
        if source is None:
            QMessageBox.information(self, "尚未加载数据", "请先加载 HDF5 数据。")
            return
        try:
            columns = self._selected_bad_check_columns(source)
        except ValueError as exc:
            QMessageBox.warning(self, "通道范围无效", str(exc))
            return

        tasks = []
        if self.bad_check_var.isChecked():
            tasks.append("bad")
        else:
            if self.bad_fast_artifact_check_var.isChecked():
                tasks.append("saturation")
            if self.bad_high_frequency_noise_check_var.isChecked():
                tasks.append("high_frequency_noise")
        if not tasks:
            QMessageBox.information(
                self, "未选择检测", "请启用坏道检查、贴底饱和或 2.5mV 附近集中检查。"
            )
            return

        self._selected_analysis_batch = {
            "source": source,
            "columns": columns,
            "tasks": tasks,
            "completed": [],
            "current_task": None,
            "review_channel_ids": (),
            "review_auto_reasons": {},
            "review_candidate_reasons": {},
        }
        scope = source.metadata.channels if columns is None else int(columns.size)
        self.preprocess_status.setText(
            f"已开始批量检测：{len(tasks)} 项，{scope} 个通道；检测使用原始/重映射数据。"
        )
        self._continue_selected_analysis_batch()

    def _continue_selected_analysis_batch(self) -> None:
        batch = self._selected_analysis_batch
        if batch is None or self._analysis_batch_waiting_for_worker or self._analysis_batch_waiting_for_psd:
            return
        if not batch["tasks"]:
            completed = "、".join(batch["completed"])
            self.preprocess_status.setText(
                f"批量检测完成：{completed}。原始/重映射通道数据和独立指标已缓存。"
            )
            self._selected_analysis_batch = None
            self._advance_one_click_preprocess("quality")
            return

        task = batch["tasks"].pop(0)
        batch["current_task"] = task
        source = batch["source"]
        columns = batch["columns"]
        if task == "bad":
            batch["completed"].append("全记录坏道检查")
            self._analysis_batch_waiting_for_worker = True
            self._run_bad_channel_worker(source=source, selected_columns=columns)
        elif task == "saturation":
            batch["completed"].append("\u8d34\u5e95\u9971\u548c")
            self._analysis_batch_waiting_for_worker = True
            self._run_bad_channel_worker(
                fast_artifact_only=True, source=source, selected_columns=columns,
                allow_when_disabled=True,
            )
        elif task == "high_frequency_noise":
            batch["completed"].append("2.5mV附近集中坏道")
            self._analysis_batch_waiting_for_worker = True
            self._run_bad_channel_worker(
                high_frequency_noise_only=True, source=source, selected_columns=columns,
                allow_when_disabled=True,
            )
        else:
            self._selected_analysis_batch = None

    def _run_bad_channel_worker(
        self,
        *,
        fast_artifact_only: bool = False,
        high_frequency_noise_only: bool = False,
        source=None,
        selected_columns=None,
        allow_when_disabled: bool = False,
    ) -> None:
        if getattr(self, "_one_click_pipeline", None) is None:
            self.last_one_click_stage_timings = []
        source = source or self._detection_source()
        if source is None or not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载或完成预处理。")
            return
        if not allow_when_disabled and not self.bad_check_var.isChecked():
            self.bad_channel_result_summary.setText("坏道检查当前已关闭；勾选“启用坏道检查”后再执行。")
            if self._selected_analysis_batch is None:
                self.bad_channel_reason_table.setRowCount(0)
            self.preprocess_status.setText("坏道检查已关闭，未执行判定。")
            self._bad_channel_check_completed = False
            self.apply_car_button.setEnabled(False)
            return
        try:
            self._bad_channel_fast_only = bool(fast_artifact_only)
            settings = {
                "global_flat_check": False,
                "discrete_level_check": False,
                "flat_time_check": False,
                "flat_time_all_channels": False,
                "parallel": self.bad_parallel_var.isChecked(),
                "workers": self.bad_workers_var.text() or "0",
                "fast_artifact_only": fast_artifact_only,
                "fast_artifact_check": self.bad_fast_artifact_check_var.isChecked(),
                "saturation_width_percent": self.bad_saturation_width_percent_var.text() or "1.0",
                "saturation_ratio_threshold": self.bad_saturation_ratio_threshold_var.text() or "45",
                "high_frequency_noise_check": self.bad_high_frequency_noise_check_var.isChecked(),
                "high_frequency_noise_only": high_frequency_noise_only,
                "high_frequency_noise_target": self.bad_high_frequency_noise_target_var.text() or "2500",
                "high_frequency_noise_tolerance": self.bad_high_frequency_noise_tolerance_var.text() or "1",
                "high_frequency_noise_ratio_threshold": self.bad_high_frequency_noise_ratio_threshold_var.text() or "50",
            }
            if selected_columns is None:
                selected_columns = self._selected_bad_check_columns(source)
            else:
                selected_columns = np.unique(np.asarray(selected_columns, dtype=np.int64).ravel())
            result_cache_key = self._bad_channel_result_cache_key(source, selected_columns, settings)
            cached_result = self.analysis_cache.get_result(result_cache_key)
            if cached_result is not None:
                self._bad_channel_worker = SimpleNamespace(
                    source=source,
                    columns=selected_columns,
                    settings=dict(settings),
                    fast_artifact_rows=cached_result["fast_artifact_rows"],
                    high_frequency_noise_rows=cached_result.get("high_frequency_noise_rows", []),
                    stage_timings=[("坏道检查结果缓存（未重新计算）", 0.0)],
                )
                self._finish_bad_channel_check(
                    cached_result["good_ids"], cached_result["reasons"], cached_result["candidates"]
                )
                self.preprocess_status.setText("坏道检查：使用缓存结果。")
                return
            worker = BadChannelWorker(
                source, settings, columns=selected_columns, analysis_cache=self.analysis_cache
            )
            worker.settings = dict(settings)
            worker.result_cache_key = result_cache_key
            worker.progress.connect(self._update_preprocess_progress)
            worker.completed.connect(self._finish_bad_channel_check)
            worker.failed.connect(lambda error: self._fail_bad_channel_check(error))
            self._bad_channel_worker = worker
            self.bad_channel_reason_table.setRowCount(0)
            self._bad_channel_check_completed = False
            self.apply_car_button.setEnabled(False)
            operation = "贴底饱和检查" if fast_artifact_only else "2.5mV附近集中检查" if high_frequency_noise_only else "全记录坏道检查"
            self._begin_shared_preprocess_task(worker, operation)
            self.bad_channel_result_summary.setText(f"正在执行{operation}…")
            self.preprocess_progress.setValue(0); self.preprocess_status.setText(f"正在执行{operation}…"); worker.start()
            if selected_columns is not None:
                self.bad_channel_result_summary.setText(
                    f"{operation}：正在检查 {selected_columns.size} 个选定通道…"
                )
        except Exception as exc:
            QMessageBox.critical(self, "坏道检查失败", str(exc))

    def show_bad_channel_timing(self) -> None:
        """Show the latest completed one-click or standalone check timings."""
        one_click_timings = list(getattr(self, "last_one_click_stage_timings", []))
        timings = one_click_timings or list(getattr(self, "last_bad_channel_stage_timings", []))
        if not timings:
            QMessageBox.information(self, "运行耗时", "尚无一键流程或坏道/质量检测耗时记录。")
            return
        title = "最近一次一键流程的阶段耗时：" if one_click_timings else "最近一次坏道/质量检测的阶段耗时："
        lines = [title]
        lines.extend(f"{name}: {elapsed:.2f} 秒" for name, elapsed in timings)
        lines.append("")
        lines.append("并行阶段显示的是实际等待的墙钟耗时，不会把各线程时间相加。")
        QMessageBox.information(self, "运行耗时", "\n".join(lines))

    def _finish_bad_channel_check(self, good_ids, reasons, candidates=None) -> None:
        candidates = candidates or {}
        worker = getattr(self, "_bad_channel_worker", None)
        self._channel_review_source = getattr(worker, "source", None)
        self.last_bad_channel_stage_timings = list(getattr(worker, "stage_timings", []))
        result_cache_key = getattr(worker, "result_cache_key", None)
        if result_cache_key is not None:
            self.analysis_cache.put_result(result_cache_key, {
                "good_ids": list(good_ids),
                "reasons": dict(reasons),
                "candidates": dict(candidates),
                "fast_artifact_rows": list(getattr(worker, "fast_artifact_rows", [])),
                "high_frequency_noise_rows": list(getattr(worker, "high_frequency_noise_rows", [])),
            })
        preserve_batch_metrics = self._selected_analysis_batch is not None
        saturation_rows = list(getattr(worker, "fast_artifact_rows", []))
        high_frequency_noise_rows = list(getattr(worker, "high_frequency_noise_rows", []))
        if saturation_rows or not preserve_batch_metrics:
            self.bad_channel_fast_artifact_rows = saturation_rows
        if high_frequency_noise_rows or not preserve_batch_metrics:
            self.bad_channel_high_frequency_noise_rows = high_frequency_noise_rows
        result_channel_ids = (
            {int(channel) for channel in good_ids}
            | {int(channel) for channel in reasons}
            | {int(channel) for channel in candidates}
        )
        if self._selected_analysis_batch is not None:
            self._merge_batch_review_results(result_channel_ids, reasons, candidates)
        else:
            self._bad_channel_result_channel_ids = tuple(sorted(result_channel_ids))
            self.bad_channel_auto_reasons = {int(channel): str(reason) for channel, reason in reasons.items()}
            self.bad_channel_candidate_reasons = {int(channel): str(reason) for channel, reason in candidates.items()}
            self.manual_channel_overrides = {
                int(channel): result for channel, result in self.manual_channel_overrides.items()
                if int(channel) in self._bad_channel_result_channel_ids
            }
            self._apply_channel_review_overrides()
        self._bad_channel_check_completed = True
        self._sync_dual_stream_export_enabled()
        if hasattr(self, "apply_car_button"):
            self.apply_car_button.setEnabled(len(self.good_channel_ids) >= 2)
        self.preprocess_progress.setValue(100)
        self._refresh_bad_channel_review_table()
        summary = "; ".join(f"ch{channel}: {reason}" for channel, reason in list(reasons.items())[:12])
        self.preprocess_status.setText(
            f"全记录坏道检查完成：健康 {len(self.good_channel_ids)}，坏道 {len(self.bad_channel_ids)}。{summary}"
        )
        if self._analysis_batch_waiting_for_worker:
            self._analysis_batch_waiting_for_worker = False
            self._continue_selected_analysis_batch()

    def _apply_channel_review_overrides(self) -> None:
        all_ids = set(self._bad_channel_result_channel_ids)
        bad_ids = set(self.bad_channel_auto_reasons)
        for channel, result in self.manual_channel_overrides.items():
            if result == "good":
                bad_ids.discard(channel)
            elif result == "bad":
                bad_ids.add(channel)
        self.bad_channel_ids = bad_ids
        self.good_channel_ids = all_ids - bad_ids

    def _completed_qc_matches_source(self, source) -> bool:
        """Whether stored completed QC rows still identify this data source."""
        if not self._bad_channel_check_completed:
            return False
        meta = getattr(source, "metadata", None)
        if meta is None:
            return False
        evaluated = {int(channel) for channel in self._bad_channel_result_channel_ids}
        available = {int(channel) for channel in meta.channel_ids}
        # A partial-channel QC remains exportable after filtering too; every
        # evaluated channel only needs to retain the same real channel ID.
        return bool(evaluated) and evaluated.issubset(available)

    def _remap_channel_review_ids(self, source, remap_path: str) -> bool:
        """Move existing review results to their remapped destination IDs."""
        if (
            not self._bad_channel_check_completed
            or getattr(self, "_channel_review_source", None) is not source
        ):
            return False
        source_ids = np.asarray(source.metadata.channel_ids, dtype=np.int64)
        destinations = read_channel_remap_for_ids(remap_path, source_ids)
        id_map = {
            int(old): int(new) for old, new in zip(source_ids, destinations)
            if int(new) >= 1
        }

        def map_ids(values):
            return {id_map[value] for value in values if value in id_map}

        def map_reasons(values):
            return {
                id_map[channel]: reason for channel, reason in values.items()
                if channel in id_map
            }

        def map_rows(rows):
            mapped = []
            for row in rows:
                old_channel = int(row.get("channel", -1))
                if old_channel not in id_map:
                    continue
                copied = dict(row)
                copied["channel"] = id_map[old_channel]
                mapped.append(copied)
            return mapped

        self._bad_channel_result_channel_ids = tuple(sorted(map_ids(self._bad_channel_result_channel_ids)))
        self.bad_channel_auto_reasons = map_reasons(self.bad_channel_auto_reasons)
        self.bad_channel_candidate_reasons = map_reasons(self.bad_channel_candidate_reasons)
        self.manual_channel_overrides = map_reasons(self.manual_channel_overrides)
        self.good_channel_ids = map_ids(self.good_channel_ids)
        self.bad_channel_ids = map_ids(self.bad_channel_ids)
        self.lfp_selected_ids = map_ids(self.lfp_selected_ids)
        self.spike_selected_ids = map_ids(self.spike_selected_ids)
        for attribute in (
            "bad_channel_fast_artifact_rows",
            "bad_channel_high_frequency_noise_rows",
        ):
            setattr(self, attribute, map_rows(getattr(self, attribute, [])))
        self._channel_review_source = self.remapped_source
        return True

    def _merge_batch_review_results(self, channel_ids, automatic_reasons, candidate_reasons) -> None:
        """Accumulate independent checked-rule outcomes into one review table."""
        batch = self._selected_analysis_batch
        if batch is None:
            return
        batch["review_channel_ids"] = tuple(sorted(
            set(batch.get("review_channel_ids", ())) | {int(channel) for channel in channel_ids}
        ))

        def merge(target, incoming) -> None:
            for channel, reason in incoming.items():
                channel = int(channel)
                reason = str(reason)
                previous = target.get(channel)
                if not previous:
                    target[channel] = reason
                elif reason not in previous:
                    target[channel] = f"{previous}; {reason}"

        merge(batch["review_auto_reasons"], automatic_reasons)
        merge(batch["review_candidate_reasons"], candidate_reasons)
        self._bad_channel_fast_only = False
        self._bad_channel_result_channel_ids = batch["review_channel_ids"]
        self.bad_channel_auto_reasons = dict(batch["review_auto_reasons"])
        self.bad_channel_candidate_reasons = dict(batch["review_candidate_reasons"])
        self.manual_channel_overrides = {
            int(channel): result for channel, result in self.manual_channel_overrides.items()
            if int(channel) in self._bad_channel_result_channel_ids
        }
        self._apply_channel_review_overrides()

    def _channel_review_category(self, channel: int) -> str:
        manual = self.manual_channel_overrides.get(channel)
        if manual == "bad":
            return "bad"
        if manual == "good":
            return "good"
        if channel in self.bad_channel_auto_reasons:
            return "bad"
        return "good"

    def _set_preprocess_preview_review_scope(self, channel_ids) -> None:
        """Make preview navigation follow the channel rows currently displayed."""
        preview = getattr(self, "preprocessed_preview", None)
        if preview is None or preview.source.metadata is None:
            return
        columns_by_id = {
            int(channel_id): index
            for index, channel_id in enumerate(preview.source.metadata.channel_ids)
        }
        columns = [columns_by_id[int(channel)] for channel in channel_ids if int(channel) in columns_by_id]
        preview.set_channel_cycle_columns(columns)
        original_preview = getattr(self, "original_preprocess_preview", None)
        original_meta = getattr(getattr(original_preview, "source", None), "metadata", None)
        if original_preview is not None and original_meta is not None:
            original_columns_by_id = {
                int(channel_id): index
                for index, channel_id in enumerate(original_meta.channel_ids)
            }
            original_preview.set_channel_cycle_columns([
                original_columns_by_id[int(channel)]
                for channel in channel_ids
                if int(channel) in original_columns_by_id
            ])
        channel_id = preview.current_channel_id()
        if channel_id is not None:
            self._highlight_bad_channel_review_row(channel_id)

    def _highlight_bad_channel_review_row(self, channel_id: int) -> None:
        """Keep the result row visibly selected when the preview changes channel."""
        table = getattr(self, "bad_channel_reason_table", None)
        if table is None:
            return
        target_row = None
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == int(channel_id):
                target_row = row
                break
        if target_row is None:
            return
        table.blockSignals(True)
        table.clearSelection()
        table.setCurrentCell(target_row, 0)
        # setCurrentCell can reduce a preceding row selection to one current
        # cell. Select the whole row last so selectedRows() remains populated
        # for the next rapid review-button click.
        table.selectRow(target_row)
        table.blockSignals(False)
        item = table.item(target_row, 0)
        if item is not None:
            table.scrollToItem(item)

    def _preview_selected_bad_channel_review_row(self) -> None:
        """Open the selected review-row channel in the right-hand preview."""
        table = getattr(self, "bad_channel_reason_table", None)
        preview = getattr(self, "preprocessed_preview", None)
        if table is None or preview is None or preview.source.metadata is None:
            return
        row = table.currentRow()
        item = table.item(row, 0) if row >= 0 else None
        if item is None:
            return
        channel_id = item.data(Qt.ItemDataRole.UserRole)
        if channel_id is None:
            return
        columns = np.flatnonzero(
            np.asarray(preview.source.metadata.channel_ids, dtype=np.int64) == int(channel_id)
        )
        if not columns.size:
            return
        preview.channel_spin.setValue(int(columns[0]) + 1)
        preview._refresh_now()

    @staticmethod
    def _brief_channel_review_description(detail: str, result: str = "") -> str:
        """Translate verbose QC reasons into scan-friendly labels."""
        source_text = str(detail or "")
        text = source_text.lower()
        labels = []
        if "人工健康" in source_text:
            labels.append("人工标记为健康")
        elif "人工坏道" in source_text:
            labels.append("人工标记为坏道")
        if "2.5mv" in text or "2.5v" in text or "高频噪声/伪迹" in source_text:
            labels.append("数据集中在2.5mV")
        if "saturation" in text or "贴底" in source_text or "饱和" in source_text:
            labels.append("贴底饱和")
        if (
            "flat/dead" in text or "flat time" in text
            or "unique levels" in text or "平坦" in source_text or "死道" in source_text
        ):
            labels.append("平坦/死道")
        if "too few finite" in text or "有效采样不足" in source_text:
            labels.append("有效采样不足")
        if not labels:
            return "自动检查未发现异常" if result in {"健康", "healthy"} else "未发现明确异常原因"
        return "；".join(dict.fromkeys(labels))

    def _toggle_bad_channel_review_description_mode(self, section: int) -> None:
        """Clicking the description header switches compact/full explanations."""
        if int(section) != 2:
            return
        self._bad_channel_review_brief_mode = not bool(
            getattr(self, "_bad_channel_review_brief_mode", False)
        )
        self._refresh_bad_channel_review_table()

    @staticmethod
    def _matches_bad_channel_review_filter(
        channel: int, filter_name: str, manual_overrides: dict, category: str,
    ) -> bool:
        if filter_name == "all":
            return True
        if filter_name == "manual":
            return int(channel) in manual_overrides
        return category == filter_name

    def _selected_bad_channel_result_ids(self) -> list[int]:
        rows = sorted({index.row() for index in self.bad_channel_reason_table.selectionModel().selectedRows()})
        return [
            int(self.bad_channel_reason_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
            for row in rows if self.bad_channel_reason_table.item(row, 0) is not None
        ]

    @staticmethod
    def _next_review_channel_id(visible_channel_ids, selected_rows) -> int | None:
        """Return the row immediately after the last selected visible row."""
        channels = [int(channel) for channel in visible_channel_ids]
        rows = sorted({int(row) for row in selected_rows if int(row) >= 0})
        if not rows:
            return None
        next_row = rows[-1] + 1
        return channels[next_row] if next_row < len(channels) else None

    def _next_selected_bad_channel_review_id(self) -> int | None:
        table = self.bad_channel_reason_table
        visible_ids = []
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None:
                visible_ids.append(int(item.data(Qt.ItemDataRole.UserRole)))
        selected_rows = [index.row() for index in table.selectionModel().selectedRows()]
        return self._next_review_channel_id(visible_ids, selected_rows)

    def _focus_bad_channel_review(self, channel_id: int | None) -> None:
        table = getattr(self, "bad_channel_reason_table", None)
        if channel_id is None:
            if table is not None:
                table.blockSignals(True)
                table.clearSelection()
                table.setCurrentCell(-1, -1)
                table.blockSignals(False)
            return
        self._highlight_bad_channel_review_row(int(channel_id))
        # _highlight blocks table signals to avoid a selection/preview loop;
        # explicitly refresh the preview once after the automatic advance.
        self._preview_selected_bad_channel_review_row()
        if table is not None:
            table.setFocus()

            # The preview refresh can emit a queued channel-preview signal.
            # Reassert the same current row after that event so rapid repeated
            # review-button clicks always operate on a real selected row.
            def keep_next_row_selected(expected_channel=int(channel_id), review_table=table):
                if review_table is not getattr(self, "bad_channel_reason_table", None):
                    return
                self._highlight_bad_channel_review_row(expected_channel)
                review_table.setFocus()

            QTimer.singleShot(0, keep_next_row_selected)

    def _set_selected_channel_review(self, result: str) -> None:
        channels = self._selected_bad_channel_result_ids()
        if not channels:
            QMessageBox.information(self, "未选择通道", "请先在坏道结果表中选择一个或多个通道。")
            return
        next_channel = self._next_selected_bad_channel_review_id()
        self.manual_channel_overrides.update({channel: result for channel in channels})
        self._apply_channel_review_overrides()
        if hasattr(self, "apply_car_button"):
            self.apply_car_button.setEnabled(self._bad_channel_check_completed and len(self.good_channel_ids) >= 2)
        if result == "good":
            self.lfp_selected_ids.update(channels)
            self.spike_selected_ids.update(channels)
            label = "健康"
        else:
            self.lfp_selected_ids.difference_update(channels)
            self.spike_selected_ids.difference_update(channels)
            label = "坏道"
        self._refresh_bad_channel_review_table()
        self._focus_bad_channel_review(next_channel)
        advance_text = f"；已自动切换到 ch{next_channel}" if next_channel is not None else "；已到当前列表末尾"
        self.preprocess_status.setText(
            f"已将 {len(channels)} 个选中通道人工标记为{label}；后续通道选择已同步更新{advance_text}。"
        )
        self._update_file_banners()

    def run_leave_one_out_median_car(self) -> None:
        """Apply CAR only after bad-channel review has produced good IDs."""
        if not self._confirm_repeated_preprocessing({"reference"}, "重参考（CAR）"):
            return
        if not self._bad_channel_check_completed:
            QMessageBox.information(self, "尚未完成坏道判断", "请先执行坏道检查，确认合格通道后再应用重参考。")
            return
        source = self.preprocessed_source or self.remapped_source or self.active_source
        if source is None or not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载或完成预处理。")
            return
        good_ids = set(getattr(self, "good_channel_ids", set()))
        if len(good_ids) < 2:
            QMessageBox.warning(self, "合格通道不足", "Leave-one-out median CAR 至少需要两个合格通道。")
            return
        if self._car_worker is not None and self._car_worker.isRunning():
            return
        self.preprocess_progress.setValue(0)
        self.preprocess_status.setText(f"正在后台应用 Leave-one-out median CAR（参考通道 {len(good_ids)} 个）…")
        self._set_preprocess_data_state("数据状态：正在应用 Leave-one-out median CAR；坏道不参与参考，也不改写。")
        worker = LeaveOneOutMedianCARWorker(source, good_ids)
        worker.progress.connect(self._update_preprocess_progress)
        worker.completed.connect(self._finish_leave_one_out_median_car)
        worker.failed.connect(self._fail_leave_one_out_median_car)
        self._car_worker = worker
        self._begin_shared_preprocess_task(worker, "Leave-one-out median CAR")
        worker.start()

    def _finish_leave_one_out_median_car(self, source, info) -> None:
        self.preprocessed_source = source
        self.active_source = source
        self.preprocess_export_button.setEnabled(True)
        self.preprocessed_preview.set_source(source)
        self._update_preprocess_comparison_preview()
        self.car_info = info
        self.preprocess_progress.setValue(100)
        self.preprocess_status.setText(
            f"Leave-one-out median CAR 已完成：{len(info['good_channel_ids'])} 个合格通道参与参考；坏道未参与且保持原值。"
        )
        self._set_preprocess_data_state(
            self._live_preprocess_state_text(
                source, "当前预览和后续分析使用 CAR 后数据；坏道保持原值。"
            )
        )
        self._update_file_banners()
        self._advance_one_click_preprocess("car")

    def _fail_leave_one_out_median_car(self, error: str) -> None:
        self.preprocess_progress.setValue(0)
        self.preprocess_status.setText(f"Leave-one-out median CAR 失败：{error}")
        self._stop_one_click_preprocess()
        QMessageBox.critical(self, "重参考失败", error)

    def _restore_selected_channel_review(self) -> None:
        channels = self._selected_bad_channel_result_ids()
        if not channels:
            QMessageBox.information(self, "未选择通道", "请先在坏道结果表中选择一个或多个通道。")
            return
        next_channel = self._next_selected_bad_channel_review_id()
        for channel in channels:
            self.manual_channel_overrides.pop(channel, None)
        self._apply_channel_review_overrides()
        if hasattr(self, "apply_car_button"):
            self.apply_car_button.setEnabled(self._bad_channel_check_completed and len(self.good_channel_ids) >= 2)
        self._refresh_bad_channel_review_table()
        self._focus_bad_channel_review(next_channel)
        advance_text = f"；已自动切换到 ch{next_channel}" if next_channel is not None else "；已到当前列表末尾"
        self.preprocess_status.setText(
            f"已恢复 {len(channels)} 个通道的自动坏道判定{advance_text}。"
        )
        self._update_file_banners()

    def _fail_bad_channel_check(self, error: str) -> None:
        self._analysis_batch_waiting_for_worker = False
        self._selected_analysis_batch = None
        self.preprocess_progress.setValue(0); self.preprocess_status.setText(f"坏道检查失败：{error}")
        self._stop_one_click_preprocess()
        QMessageBox.critical(self, "坏道检查失败", error)

    def _ica_eligible_channel_ids(self, source) -> set[int]:
        """Return ICA channels after excluding only confirmed bad channels."""
        available = {int(channel) for channel in source.metadata.channel_ids}
        if not self._bad_channel_check_completed:
            return available
        return available - {int(channel) for channel in self.bad_channel_ids}

    def run_selected_ica(self) -> None:
        if not self._confirm_repeated_preprocessing({"ica"}, "ICA"):
            return
        source = self.preprocessed_source or self.remapped_source or self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载并完成预处理。")
            return
        eligible_ids = self._ica_eligible_channel_ids(source)
        if len(eligible_ids) < 2:
            QMessageBox.warning(
                self, "ICA 通道不足",
                "坏道已排除后，可用于 ICA 的通道不足两个。",
            )
            return
        selector = getattr(self, "_ica_channel_selector", None)
        if selector is not None:
            try:
                if selector.isVisible() and selector.source is source:
                    selector.raise_()
                    selector.activateWindow()
                    return
                if selector.isVisible():
                    selector.close()
            except RuntimeError:
                pass
        selector = PyQtGraphOverview(
            source,
            0.0,
            min(5.0, source.metadata.rows / source.metadata.fs),
            selection_mode=True,
            selected_channel_ids=set(getattr(self, "ica_selected_ids", set())) & eligible_ids,
            selection_callback=lambda channel_ids, selected_source=source: self._run_ica_for_channel_ids(selected_source, channel_ids),
            display_channel_ids=eligible_ids,
        )
        selector.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        selector.showMaximized()
        self._ica_channel_selector = selector

    def _run_ica_for_channel_ids(self, source, channel_ids) -> None:
        """Run ICA after the dedicated overview selector has been confirmed."""
        available_ids = np.asarray(source.metadata.channel_ids, dtype=np.int64)
        selected_ids = {int(channel) for channel in channel_ids}
        eligible_ids = self._ica_eligible_channel_ids(source)
        excluded_ids = selected_ids - eligible_ids
        selected_ids &= eligible_ids
        columns = np.flatnonzero(np.isin(available_ids, sorted(selected_ids)))
        if columns.size < 2:
            suffix = "；坏道已自动排除" if excluded_ids else ""
            QMessageBox.information(self, "ICA 通道不足", f"ICA 至少需要选择两个当前已加载的可用通道{suffix}。")
            return
        self.ica_selected_ids = {int(available_ids[index]) for index in columns}
        # Clicking the explicit ICA action is itself an unambiguous request to
        # enable it.  The legacy check box remains a visible advanced setting,
        # but it no longer prevents the channel selector from opening.
        self.motion_ica_enable_var.setChecked(True)
        try:
            params = {
                "components": max(1, int(self.motion_ica_components_var.text() or 32)),
                "exclude": [int(value) for value in re.split(r"[,;\s]+", self.motion_ica_exclude_var.text().strip()) if value],
                "low": float(self.motion_ica_low_var.text() or 1),
                "high": float(self.motion_ica_high_var.text() or 100),
                "decim": max(1, int(self.motion_ica_decim_var.text() or 3)),
                "max_iter": max(1, int(self.motion_ica_max_iter_var.text() or 800)),
            }
            if not 0 < params["low"] < params["high"] < source.metadata.fs / 2:
                raise ValueError("ICA 频段必须满足 0 < 低频 < 高频 < 奈奎斯特频率。")
        except ValueError as exc:
            QMessageBox.warning(self, "ICA 参数无效", str(exc)); return
        actual_components = min(params["components"], int(columns.size))
        if params["components"] > columns.size:
            QMessageBox.information(
                self,
                "ICA 成分数已调整",
                f"当前选择了 {columns.size} 个通道，但 ICA 成分设为 {params['components']}。\n"
                f"独立成分数不能超过选中通道数，本次实际分解 {actual_components} 个成分。",
            )
        if getattr(self, "_ica_worker", None) is not None and self._ica_worker.isRunning():
            return
        self.preprocess_progress.setValue(0)
        excluded_note = f"；已排除 {len(excluded_ids)} 个坏道" if excluded_ids else ""
        self.preprocess_status.setText(
            f"ICA 正在后台运行：{len(self.ica_selected_ids)} 个已选通道{excluded_note}…"
        )
        worker = IcaWorker(source, columns, params)
        worker.progress.connect(self._update_preprocess_progress)
        worker.completed.connect(self._finish_ica)
        worker.failed.connect(lambda error: self._fail_ica(error))
        self._begin_shared_preprocess_task(worker, "ICA")
        self._ica_worker = worker; worker.start()

    def _finish_ica(self, source, info) -> None:
        self._before_ica_source = self.preprocessed_source or self.remapped_source or self.active_source
        self.preprocessed_source = source; self.active_source = source
        self.preprocess_export_button.setEnabled(True)
        self.preprocessed_preview.set_source(source)
        self._update_preprocess_comparison_preview()
        self.ica_info = info
        self.preprocess_progress.setValue(100)
        convergence_warnings = info.get("convergence_warnings", [])
        if convergence_warnings:
            warning_text = "；".join(convergence_warnings)
            self.preprocess_status.setText(
                f"ICA 已应用，但 FastICA 未在 {info['max_iter']} 次迭代内收敛：{warning_text}"
            )
            QMessageBox.warning(
                self,
                "ICA 未收敛",
                "ICA 已生成结果，但 FastICA 在最大迭代次数内未收敛。\n"
                "请谨慎使用本次结果；可减少 ICA 成分、排除明显噪声/坏道，或调整数据与参数后重试。\n\n"
                f"详细信息：{warning_text}",
            )
        else:
            self.preprocess_status.setText(
                f"ICA 已应用：{len(info['channel_ids'])} 通道，{info['components']} 成分，"
                f"排除 IC {info['exclude'] or '无'}。"
            )
        self._set_preprocess_data_state(
            self._live_preprocess_state_text(
                source, "当前预览和后续分析使用 ICA 处理后的数据。"
            )
        )
        self._update_file_banners()
        self._advance_one_click_preprocess("ica")

    def _fail_ica(self, error: str) -> None:
        self.preprocess_progress.setValue(0); self.preprocess_status.setText(f"ICA 失败：{error}")
        self._stop_one_click_preprocess()
        QMessageBox.critical(self, "ICA 失败", error)

    def run_ica_snr(self) -> None:
        if not hasattr(self, "ica_info"):
            QMessageBox.information(self, "ICA SNR", "请先对选中通道应用 ICA。")
            return
        self.lfp_source = self.preprocessed_source
        self.lfp_selected_ids = set(self.ica_info["channel_ids"])
        self.run_lfp()

    def clear_ica(self) -> None:
        previous = getattr(self, "_before_ica_source", self.remapped_source)
        if previous is not None:
            self.active_source = previous; self.preprocessed_source = previous
            self.preprocessed_preview.set_source(previous)
            self._update_preprocess_comparison_preview()
        self.__dict__.pop("ica_info", None)
        self.preprocess_status.setText("已清除 ICA 结果，恢复到 ICA 前的数据源。")
        self._set_preprocess_data_state("数据状态：已恢复到 ICA 前的预处理数据。")

    def _refresh_bad_channel_review_table(self) -> None:
        """Render the current two-class (healthy/bad) review state."""
        all_ids = tuple(int(value) for value in self._bad_channel_result_channel_ids)
        filter_name = self.bad_channel_result_filter.currentData() if hasattr(self, "bad_channel_result_filter") else "all"
        visible = [
            channel for channel in all_ids
            if self._matches_bad_channel_review_filter(
                channel, str(filter_name), self.manual_channel_overrides,
                "bad" if channel in self.bad_channel_ids else "good",
            )
        ]
        saturation = {
            int(row["channel"]): row for row in getattr(self, "bad_channel_fast_artifact_rows", [])
        }
        target_rows = {
            int(row["channel"]): row for row in getattr(self, "bad_channel_high_frequency_noise_rows", [])
        }
        brief = bool(getattr(self, "_bad_channel_review_brief_mode", False))
        self.bad_channel_reason_table.setColumnCount(3)
        self.bad_channel_reason_table.setHorizontalHeaderLabels([
            "通道", "结果", "简要说明（点击查看完整）" if brief else "判定说明（点击查看简要）",
        ])
        self.bad_channel_reason_table.setRowCount(len(visible))
        for row_index, channel in enumerate(visible):
            automatic = self.bad_channel_auto_reasons.get(channel, "")
            manual = self.manual_channel_overrides.get(channel)
            if manual == "good":
                result = "人工健康"
                detail = "人工健康覆盖" + (f"；自动：{automatic}" if automatic else "")
            elif manual == "bad":
                result = "人工坏道"
                detail = "人工坏道覆盖" + (f"；自动：{automatic}" if automatic else "")
            elif automatic:
                result, detail = "坏道", automatic
            else:
                result, detail = "健康", "自动检查未发现坏道理由"
            metric_details = []
            metric = saturation.get(channel)
            if metric:
                metric_details.append(
                    f"贴底比例={float(metric.get('bottom_ratio', np.nan)) * 100:.2f}%，"
                    f"阈值={float(metric.get('saturation_ratio_threshold', np.nan)) * 100:.2f}%"
                )
            metric = target_rows.get(channel)
            if metric:
                metric_details.append(
                    f"2.5mV附近比例={float(metric.get('near_target_ratio', np.nan)) * 100:.2f}%，"
                    f"阈值={float(metric.get('threshold', np.nan)) * 100:.2f}%"
                )
            full_detail = detail + ("；" + "；".join(metric_details) if metric_details else "")
            channel_item = QTableWidgetItem(f"ch{channel}")
            channel_item.setData(Qt.ItemDataRole.UserRole, channel)
            self.bad_channel_reason_table.setItem(row_index, 0, channel_item)
            self.bad_channel_reason_table.setItem(row_index, 1, QTableWidgetItem(result))
            description = self._brief_channel_review_description(detail, result) if brief else full_detail
            description_item = QTableWidgetItem(description)
            description_item.setToolTip(full_detail)
            self.bad_channel_reason_table.setItem(row_index, 2, description_item)
        self._set_preprocess_preview_review_scope(visible)
        self.bad_channel_reason_table.resizeColumnsToContents()
        self.bad_channel_reason_table.horizontalHeader().setStretchLastSection(True)
        self.bad_channel_result_summary.setText(
            f"检查结果：健康 {len(self.good_channel_ids)}，坏道 {len(self.bad_channel_ids)}，"
            f"人工覆盖 {len(self.manual_channel_overrides)}。当前筛选显示 {len(visible)}/{len(all_ids)} 个通道。"
        )

    def _bad_channel_experiment_settings(self) -> dict:
        """Freeze the retained bad-channel rules for repeatable experiments."""
        return {
            "global_flat_check": False,
            "discrete_level_check": False,
            "flat_time_check": False,
            "flat_time_all_channels": False,
            "parallel": self.bad_parallel_var.isChecked(),
            "workers": self.bad_workers_var.text() or "0",
            "fast_artifact_check": self.bad_fast_artifact_check_var.isChecked(),
            "saturation_width_percent": self.bad_saturation_width_percent_var.text() or "1.0",
            "saturation_ratio_threshold": self.bad_saturation_ratio_threshold_var.text() or "45",
            "high_frequency_noise_check": self.bad_high_frequency_noise_check_var.isChecked(),
            "high_frequency_noise_target": self.bad_high_frequency_noise_target_var.text() or "2500",
            "high_frequency_noise_tolerance": self.bad_high_frequency_noise_tolerance_var.text() or "1",
            "high_frequency_noise_ratio_threshold": self.bad_high_frequency_noise_ratio_threshold_var.text() or "50",
        }

    def _prompt_bad_channel_sweep_fixed_settings(
        self, parameter_key: str, settings: dict,
    ) -> dict | None:
        """Let the operator edit all values that stay fixed during a sweep."""
        dialog = QDialog(self)
        dialog.setWindowTitle("设置本轮固定参数")
        dialog.setMinimumWidth(620)
        layout = QVBoxLayout(dialog)
        note = QLabel(
            f"本轮测试变量：{BAD_CHANNEL_SINGLE_PARAMETERS[parameter_key]}。\n"
            "该变量由下一步的起点、终点和步长控制；以下参数在整轮测试中保持固定。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        parameter_box = QGroupBox("固定数值参数")
        form = QFormLayout(parameter_box)
        numeric_specs = OrderedDict((
            ("saturation_width_percent", ("贴底区间宽度（% PTP）", "float", 0.0, 100.0)),
            ("saturation_ratio_threshold", ("贴底比例阈值（%）", "float", 0.0, 100.0)),
            ("high_frequency_noise_target", ("2.5mV目标值（当前数据单位）", "float", None, None)),
            ("high_frequency_noise_tolerance", ("2.5mV附近容差（当前数据单位）", "float", 0.0, None)),
            ("high_frequency_noise_ratio_threshold", ("2.5mV附近比例阈值（%）", "float", 0.0, 100.0)),
        ))
        editors = {}
        for key, (label, _kind, _minimum, _maximum) in numeric_specs.items():
            editor = QLineEdit()
            if key == parameter_key:
                editor.setText("由下一步扫描范围决定")
                editor.setReadOnly(True)
                editor.setEnabled(False)
            else:
                editor.setText(str(settings.get(key, "")))
                editors[key] = editor
            form.addRow(label + "：", editor)
        layout.addWidget(parameter_box)

        rule_box = QGroupBox("参与判定的规则")
        rule_layout = QGridLayout(rule_box)
        rule_specs = OrderedDict((
            ("fast_artifact_check", "贴底饱和"),
            ("high_frequency_noise_check", "2.5mV附近集中"),
        ))
        required_rule = (
            "fast_artifact_check" if parameter_key.startswith("saturation_") else
            "high_frequency_noise_check"
        )
        rule_checks = {}
        for index, (key, label) in enumerate(rule_specs.items()):
            check = QCheckBox(label)
            check.setChecked(True if key == required_rule else bool(settings.get(key, False)))
            if key == required_rule:
                check.setEnabled(False)
                check.setToolTip("测试变量所属规则必须启用。")
            rule_checks[key] = check
            rule_layout.addWidget(check, index // 2, index % 2)
        layout.addWidget(rule_box)

        runtime_box = QGroupBox("执行设置（不影响坏道判定）")
        runtime_form = QFormLayout(runtime_box)
        parallel_check = QCheckBox("启用并行")
        parallel_check.setChecked(bool(settings.get("parallel", False)))
        workers_edit = QLineEdit(str(settings.get("workers", 0)))
        runtime_form.addRow("并行计算：", parallel_check)
        runtime_form.addRow("工作线程数（0=自动）：", workers_edit)
        layout.addWidget(runtime_box)

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel = QPushButton("取消")
        confirm = QPushButton("确认固定参数并设置扫描范围")
        confirm.setDefault(True)
        actions.addWidget(cancel)
        actions.addWidget(confirm)
        layout.addLayout(actions)
        cancel.clicked.connect(dialog.reject)

        def validate_and_accept():
            updated = dict(settings)
            for key, editor in editors.items():
                label, kind, minimum, maximum = numeric_specs[key]
                text_value = editor.text().strip()
                try:
                    value = int(text_value) if kind == "int" else float(text_value)
                except ValueError:
                    QMessageBox.warning(dialog, "参数无效", f"{label} 必须是有效数值。")
                    editor.setFocus()
                    return
                if not np.isfinite(value):
                    QMessageBox.warning(dialog, "参数无效", f"{label} 必须是有限数值。")
                    editor.setFocus()
                    return
                if minimum is not None and value < minimum:
                    QMessageBox.warning(dialog, "参数超出范围", f"{label} 不能小于 {minimum:g}。")
                    editor.setFocus()
                    return
                if maximum is not None and value > maximum:
                    QMessageBox.warning(dialog, "参数超出范围", f"{label} 不能大于 {maximum:g}。")
                    editor.setFocus()
                    return
                updated[key] = value
            try:
                workers = int(workers_edit.text().strip())
            except ValueError:
                QMessageBox.warning(dialog, "线程数无效", "工作线程数必须是 0～256 的整数。")
                workers_edit.setFocus()
                return
            if not 0 <= workers <= 256:
                QMessageBox.warning(dialog, "线程数无效", "工作线程数必须是 0～256 的整数。")
                workers_edit.setFocus()
                return
            for key, check in rule_checks.items():
                updated[key] = check.isChecked() or key == required_rule
            updated.update({
                "global_flat_check": False,
                "discrete_level_check": False,
                "flat_time_check": False,
                "flat_time_all_channels": False,
            })
            updated["parallel"] = parallel_check.isChecked()
            updated["workers"] = workers
            dialog.fixed_settings = updated
            dialog.accept()

        confirm.clicked.connect(validate_and_accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.fixed_settings

    @staticmethod
    def _format_id_list(values) -> str:
        return ",".join(str(int(value)) for value in sorted({int(v) for v in values}))

    def show_current_bad_channel_contribution(self) -> None:
        """Recalculate rule shares and manual-review burden from the frozen QC result."""
        if not self._bad_channel_check_completed:
            QMessageBox.information(self, "当前坏道贡献", "请先执行坏道检查。")
            return
        automatic = {int(channel): str(reason) for channel, reason in self.bad_channel_auto_reasons.items()}
        counts = {name: 0 for name in BatchBadChannelEvaluationWorker.RULE_NAMES}
        channels = {name: [] for name in counts}
        for channel, reason in automatic.items():
            for name in BatchBadChannelEvaluationWorker._rule_labels(reason):
                counts[name] += 1
                channels[name].append(channel)
        auto_count = len(automatic)
        manual = {
            int(channel): value for channel, value in self.manual_channel_overrides.items()
            if value in {"good", "bad"}
        }
        changed = sum(
            (value == "good" and channel in automatic)
            or (value == "bad" and channel not in automatic)
            for channel, value in manual.items()
        )
        lines = ["以下仅为触发次数，规则可能重叠；不能据此判断是否必要。",
                 f"自动判定坏道：{auto_count}；最终坏道：{len(self.bad_channel_ids)}"]
        for name in BatchBadChannelEvaluationWorker.RULE_NAMES:
            count = counts[name]
            if count:
                lines.append(
                    f"{name}：{count} 个，占自动坏道 {count / max(1, auto_count) * 100:.1f}%"
                )
        total = len(self._bad_channel_result_channel_ids)
        lines.extend([
            "",
            f"人工覆盖：{len(manual)} 个，占已检查通道 {len(manual) / max(1, total) * 100:.1f}%",
            f"实际改变自动结论：{changed} 个，占已检查通道 {changed / max(1, total) * 100:.1f}%",
            f"自动坏道→人工健康：{sum(v == 'good' and c in automatic for c, v in manual.items())} 个",
            f"自动健康→人工坏道：{sum(v == 'bad' and c not in automatic for c, v in manual.items())} 个",
        ])
        QMessageBox.information(self, "当前规则触发数（实时重算）", "\n".join(lines))

    def run_batch_bad_channel_evaluation(self) -> None:
        mapping_path = self.remap_edit.text().strip()
        if not mapping_path or not Path(mapping_path).is_file():
            QMessageBox.warning(self, "缺少映射文件", "请先选择有效的通道映射 Excel。")
            return
        reusable_pairs = self._bad_channel_sweep_selected_pairs
        if reusable_pairs is None:
            filenames, _ = QFileDialog.getOpenFileNames(
                self, "选择人工复核 H5（也可取消并在下一步读取路径清单）", str(Path.cwd()),
                "HDF5 files (*.h5 *.hdf5);;All files (*.*)",
            )
            selected_pairs = self._prompt_parameter_sweep_raw_paths(filenames)
            if selected_pairs is None:
                return
            reviewed_paths, raw_paths = selected_pairs
            self._bad_channel_sweep_selected_pairs = tuple(zip(reviewed_paths, raw_paths))
        else:
            reviewed_paths = [pair[0] for pair in reusable_pairs]
            raw_paths = [pair[1] for pair in reusable_pairs]
        default_name = reviewed_paths[0].parent / f"坏道规则去除贡献_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        output, _ = QFileDialog.getSaveFileName(
            self, "保存坏道贡献工作簿", str(default_name), "Excel workbook (*.xlsx)",
        )
        if not output:
            return
        worker = BadChannelControlledExperimentWorker(
            reviewed_paths, self._bad_channel_experiment_settings(), mapping_path,
            raw_paths, self._bad_channel_sweep_data_cache,
        )
        worker.progress.connect(self._update_preprocess_progress)
        worker.completed.connect(lambda result, path=output: self._finish_controlled_bad_channel_ablation(path, result))
        worker.failed.connect(self._fail_bad_channel_experiment)
        self._batch_bad_channel_worker = worker
        self._begin_shared_preprocess_task(worker, "批量 H5 坏道贡献")
        self.preprocess_status.setText(f"正在比较完整判断与逐规则去除：{len(reviewed_paths)} 个 H5…")
        worker.start()

    @staticmethod
    def _safe_sheet_title(text: str, used: set[str]) -> str:
        title = re.sub(r"[\\/*?:\[\]]", "_", str(text))[:31] or "结果"
        base = title
        number = 2
        while title in used:
            suffix = f"_{number}"
            title = base[:31 - len(suffix)] + suffix
            number += 1
        used.add(title)
        return title

    @staticmethod
    def _style_bad_channel_workbook(workbook) -> None:
        """Apply a compact, readable scientific-results layout."""
        from openpyxl.styles import Alignment, Font, PatternFill
        header_fill = PatternFill("solid", fgColor="1F4E78")
        header_font = Font(color="FFFFFF", bold=True)
        percent_headers = {
            "坏道比例", "Precision", "Recall", "NPV", "F1", "F2", "单文件最低Recall",
        }
        for sheet in workbook.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            headers = {cell.column: str(cell.value or "") for cell in sheet[1]}
            for column, header in headers.items():
                if header in percent_headers:
                    for row in range(2, sheet.max_row + 1):
                        sheet.cell(row, column).number_format = "0.0%"
            for column_cells in sheet.columns:
                values = list(column_cells)
                width = max(
                    (len(str(cell.value)) for cell in values[:201] if cell.value is not None),
                    default=8,
                )
                sheet.column_dimensions[values[0].column_letter].width = min(48, max(10, width + 2))
            sheet.row_dimensions[1].height = 28

    @staticmethod
    def _controlled_metric_summary(rows: list[dict]) -> dict:
        tp = sum(row["tp"] for row in rows)
        fp = sum(row["fp"] for row in rows)
        fn = sum(row["fn"] for row in rows)
        tn = sum(row["tn"] for row in rows)
        precision = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        npv = tn / (tn + fn) if tn + fn else 1.0
        return dict(files=len(rows), tp=tp, fp=fp, fn=fn, tn=tn,
                    precision=precision, recall=recall, npv=npv,
                    min_file_recall=min((row["recall"] for row in rows), default=0.0))

    @classmethod
    def _controlled_sweep_choice(cls, result: dict):
        paths = set(result["paths"])
        if result["failures"] or not paths:
            return None, "有文件未完成；不从部分数据推荐阈值。"
        summaries = []
        baseline_rows = [row for row in result["runs"] if row["scenario"] == "baseline"]
        if {row["path"] for row in baseline_rows} != paths:
            return None, "基线结果不完整。"
        baseline = cls._controlled_metric_summary(baseline_rows)
        for value in result["values"]:
            rows = [row for row in result["runs"]
                    if row["scenario"] == "sweep" and row["value"] == value]
            if {row["path"] for row in rows} != paths:
                return None, "扫描结果不完整。"
            summaries.append(dict(value=value, **cls._controlled_metric_summary(rows)))
        qualified = [row for row in summaries
                     if row["recall"] >= .98 and row["min_file_recall"] >= .95]
        if all((row["tp"], row["fp"], row["fn"]) ==
               (baseline["tp"], baseline["fp"], baseline["fn"]) for row in summaries):
            return None, "扫描值对本批最终判定无可观察影响；保留原设置，不能声称该数值有独立贡献。"
        if not qualified:
            return None, "本次扫描范围没有数值同时达到总体 Recall≥98% 与每文件 Recall≥95%；不推荐达标参数。"
        baseline_value = float(result["base_settings"][result["parameter_key"]])
        chosen = max(qualified, key=lambda row: (
            row["precision"], -row["fp"], row["npv"],
            -abs(row["value"] - baseline_value),
        ))
        return chosen, (
            f"固定其他设置后，相对原值：TP {chosen['tp']-baseline['tp']:+d}，"
            f"FP {chosen['fp']-baseline['fp']:+d}，FN {chosen['fn']-baseline['fn']:+d}；"
            "先满足召回约束，再优先减少误报。"
        )

    def _finish_controlled_bad_channel_sweep(self, output: str, result: dict) -> None:
        from openpyxl import Workbook
        choice, rationale = self._controlled_sweep_choice(result)
        baseline_rows = [row for row in result["runs"] if row["scenario"] == "baseline"]
        baseline = self._controlled_metric_summary(baseline_rows)
        try:
            workbook = Workbook()
            summary = workbook.active
            summary.title = "单变量结论"
            summary.append(["项目", "内容"])
            summary.append(["唯一变化参数", BAD_CHANNEL_SINGLE_PARAMETERS[result["parameter_key"]]])
            active_rules = [
                label for key, label in BAD_CHANNEL_ABLATION_RULES.items()
                if result["evaluation_settings"].get(key, False)
            ]
            summary.append(["本轮参与判坏规则", "、".join(active_rules)])
            summary.append(["原设置", float(result["base_settings"][result["parameter_key"]])])
            summary.append(["达标建议值", choice["value"] if choice else "无"])
            summary.append(["选择依据", rationale])
            summary.append(["对照设计", "同一批文件、映射、人工标签及其他规则固定；每轮只改变一个参数数值。"])
            summary.append(["已排除规则", "全局平直/低波动、离散水平过少、平直时间比例不参与本轮基线或扫描判定。"])
            summary.append(["证据范围", "同批数据回顾性调参；不是独立验证，不能直接证明对新文件同样有效。"])
            summary.append(["无预测坏道时", "若有人工坏道，Precision 按 0 记录；若无人工坏道则按 1 记录，必须结合 TP/FP/FN 解读。"])
            summary.append(["基线 TP", baseline["tp"]])
            summary.append(["基线 FP", baseline["fp"]])
            summary.append(["基线 FN", baseline["fn"]])
            table = workbook.create_sheet("参数汇总")
            table.append(["参数值", "文件数", "TP", "FP", "FN", "TN", "Precision", "Recall", "NPV",
                          "单文件最低Recall", "相对基线TP变化", "相对基线FP变化", "相对基线FN变化", "是否达标"])
            for value in result["values"]:
                rows = [row for row in result["runs"] if row["scenario"] == "sweep" and row["value"] == value]
                score = self._controlled_metric_summary(rows)
                table.append([value, score["files"], score["tp"], score["fp"], score["fn"], score["tn"],
                              score["precision"], score["recall"], score["npv"], score["min_file_recall"],
                              score["tp"]-baseline["tp"], score["fp"]-baseline["fp"],
                              score["fn"]-baseline["fn"],
                              "是" if score["recall"] >= .98 and score["min_file_recall"] >= .95 else "否"])
            detail = workbook.create_sheet("逐文件对照")
            detail.append(["人工复核H5", "原始H5", "场景", "参数值", "评估通道",
                           "人工坏道", "TP", "FP", "FN", "TN", "Precision", "Recall", "预测坏道ID",
                           "相对基线漏掉的真坏道ID", "相对基线减少的误报ID",
                           "相对基线新增的真坏道ID", "相对基线新增的误报ID"])
            for row in result["runs"]:
                detail.append([row["file"], row["raw_path"], row["scenario"], row["value"],
                               row["evaluated_count"],
                               row["reference_bad_count"], row["tp"], row["fp"], row["fn"], row["tn"],
                               row["precision"], row["recall"], self._format_id_list(row["predicted_bad_ids"]),
                               self._format_id_list(row["tp_lost_vs_baseline_ids"]),
                               self._format_id_list(row["fp_removed_vs_baseline_ids"]),
                               self._format_id_list(row["tp_gained_vs_baseline_ids"]),
                               self._format_id_list(row["fp_added_vs_baseline_ids"])])
            fixed = workbook.create_sheet("实际测试设置")
            fixed.append(["参数", "基线测试值"])
            for key, value in result["evaluation_settings"].items():
                if key in {"flat_std", "discrete_level_threshold", "flat_ratio"}:
                    continue
                fixed.append([key, value])
            failure = workbook.create_sheet("失败文件")
            failure.append(["文件", "路径", "错误"])
            for row in result["failures"]:
                failure.append([row["file"], row["path"], row["error"]])
            self._style_bad_channel_workbook(workbook)
            workbook.save(output)
        except Exception as exc:
            self._fail_bad_channel_experiment(str(exc))
            return
        self.preprocess_progress.setValue(100)
        self.preprocess_status.setText(f"单变量参数精调完成：{output}")
        QMessageBox.information(self, "单变量参数精调完成",
                                (f"{rationale}\n建议值：{choice['value']:g}" if choice else rationale)
                                + f"\n\n工作簿：\n{output}")

    def _finish_controlled_bad_channel_ablation(self, output: str, result: dict) -> None:
        from openpyxl import Workbook
        try:
            workbook = Workbook()
            summary = workbook.active
            summary.title = "规则贡献结论"
            summary.append(["规则", "状态", "基线TP", "基线FP", "去除后TP", "去除后FP",
                            "丢失真坏道", "减少误报", "判断"])
            baseline_rows = [row for row in result["runs"] if row["scenario"] == "baseline"]
            baseline = self._controlled_metric_summary(baseline_rows)
            complete = not result["failures"] and len(baseline_rows) == len(result["paths"])
            for rule, label in BAD_CHANNEL_ABLATION_RULES.items():
                rows = [row for row in result["runs"] if row["scenario"] == rule]
                active = bool(rows) and all(row["active"] for row in rows)
                if not active:
                    status, judgement, score = "原本未启用", "未参与基线判定，不评价必要性", {}
                elif not complete or len(rows) != len(result["paths"]):
                    status, judgement, score = "数据不完整", "不能据此判定贡献", {}
                else:
                    status = "已测试"
                    score = self._controlled_metric_summary(rows)
                    lost = sum(len(row["tp_lost_vs_baseline_ids"]) for row in rows)
                    removed = sum(len(row["fp_removed_vs_baseline_ids"]) for row in rows)
                    unexpected = sum(len(row["tp_gained_vs_baseline_ids"]) +
                                     len(row["fp_added_vs_baseline_ids"]) for row in rows)
                    if unexpected:
                        judgement = "关闭规则后出现新增预测，存在规则交互；不可按简单去除贡献解释"
                    elif lost > 0:
                        judgement = "有独立真坏道增益；是否保留须权衡同时带来的误报"
                    elif removed > 0:
                        judgement = "本批无独立真坏道增益，去除可减少误报；建议验证后考虑停用"
                    else:
                        judgement = "本批去除后最终预测不变；可能冗余，不能证明对新文件无用"
                summary.append([label, status, baseline["tp"], baseline["fp"],
                                score.get("tp"), score.get("fp"),
                                lost if score else None,
                                removed if score else None, judgement])
            note = workbook.create_sheet("方法说明", 0)
            note.append(["项目", "说明"])
            note.append(["方法", "固定全部数值参数与文件，分别关闭一项规则并重跑完整坏道判断；与同文件基线比较。"])
            note.append(["含义", "这是规则在当前参数、当前数据上的边际贡献；不能直接归因到规则内部某一个数值参数。"])
            note.append(["有效采样不足", "数据完整性保护条件始终保留，不作为可删除的坏道规则。"])
            note.append(["局限", "同批人工复核数据的回顾性结论；无独立增益不等于以后永远没用。"])
            detail = workbook.create_sheet("逐文件去除对照")
            detail.append(["人工复核H5", "原始H5", "场景", "是否启用", "评估通道", "未纳入评估通道数",
                           "TP", "FP", "FN", "TN", "预测坏道ID", "去除后丢失真坏道ID", "去除后减少误报ID",
                           "去除后新增真坏道ID", "去除后新增误报ID"])
            for row in result["runs"]:
                detail.append([row["file"], row["raw_path"], row["scenario"], row["active"],
                               row["evaluated_count"], row["excluded_unmapped_count"],
                               row.get("tp"), row.get("fp"), row.get("fn"), row.get("tn"),
                               self._format_id_list(row["predicted_bad_ids"]),
                               self._format_id_list(row["tp_lost_vs_baseline_ids"]),
                               self._format_id_list(row["fp_removed_vs_baseline_ids"]),
                               self._format_id_list(row["tp_gained_vs_baseline_ids"]),
                               self._format_id_list(row["fp_added_vs_baseline_ids"])])
            fixed = workbook.create_sheet("固定参数")
            fixed.append(["参数", "固定值"])
            for key, value in result["base_settings"].items():
                fixed.append([key, value])
            failure = workbook.create_sheet("失败文件")
            failure.append(["文件", "路径", "错误"])
            for row in result["failures"]:
                failure.append([row["file"], row["path"], row["error"]])
            self._style_bad_channel_workbook(workbook)
            workbook.save(output)
        except Exception as exc:
            self._fail_bad_channel_experiment(str(exc))
            return
        self.preprocess_progress.setValue(100)
        self.preprocess_status.setText(f"坏道规则贡献完成：{output}")
        QMessageBox.information(self, "坏道规则贡献完成",
                                f"已完成逐项去除对照；请结合丢失真坏道与减少误报解读。\n\n工作簿：\n{output}")

    def _write_batch_bad_channel_workbook(self, output: str, results: list[dict]) -> None:
        from openpyxl import Workbook
        workbook = Workbook()
        overview = workbook.active
        overview.title = "总览"
        overview.append(["文件", "状态", "通道数", "健康数", "坏道数", "坏道比例", "错误"])
        for item in results:
            overview.append([
                item["file"], item["status"], item["channels"], item["good_count"],
                item["bad_count"], item["bad_ratio"], item["error"],
            ])
        settings_sheet = workbook.create_sheet("本次参数")
        settings_sheet.append(["参数", "值"])
        for key, value in self._bad_channel_experiment_settings().items():
            settings_sheet.append([key, value])
        used = {sheet.title for sheet in workbook.worksheets}
        for item in results:
            sheet = workbook.create_sheet(self._safe_sheet_title(Path(item["file"]).stem, used))
            sheet.append(["通道", "结果", "触发条件数", "触发条件", "完整理由"])
            for row in item.get("channel_rows", []):
                sheet.append([row["channel"], row["result"], row["trigger_count"], row["rules"], row["reason"]])
        self._style_bad_channel_workbook(workbook)
        workbook.save(output)

    def _finish_batch_bad_channel_evaluation(self, output: str, results: list[dict]) -> None:
        try:
            self._write_batch_bad_channel_workbook(output, results)
        except Exception as exc:
            self._fail_bad_channel_experiment(str(exc))
            return
        success = sum(item.get("status") == "成功" for item in results)
        self.preprocess_progress.setValue(100)
        self.preprocess_status.setText(f"批量坏道贡献完成：{success}/{len(results)} 个成功；{output}")
        QMessageBox.information(self, "批量坏道贡献完成", f"结果已保存：\n{output}")

    def _fail_bad_channel_experiment(self, error: str) -> None:
        self.preprocess_progress.setValue(0)
        self.preprocess_status.setText(f"坏道实验失败：{error}")
        QMessageBox.critical(self, "坏道实验失败", error)

    def clear_bad_channel_parameter_sweep_data(self) -> None:
        """Forget the reusable sweep selection and its remapped data."""
        self._bad_channel_sweep_selected_pairs = None
        self._bad_channel_sweep_data_cache.clear()
        self._bad_channel_pair_results.clear()
        self._bad_channel_sweep_cohort_signature = None
        self.preprocess_status.setText("参数精调数据已清除；下次运行时将重新选择并重映射 H5。")

    def _prompt_parameter_sweep_raw_paths(
        self, filenames,
    ) -> tuple[list[Path], list[Path]] | None:
        """Edit, order, save, and restore reviewed/raw H5 path pairs."""
        initial_processed_paths = [Path(normalize_h5_path_text(name)) for name in filenames]
        dialog = QDialog(self)
        dialog.setWindowTitle("人工复核 H5 与原始 H5 配对清单")
        dialog.resize(1350, max(420, min(800, 245 + len(initial_processed_paths) * 42)))
        layout = QVBoxLayout(dialog)
        note = QLabel(
            "只需填写人工复核结果 H5 的绝对路径；原始 H5 将按既定目录和文件名规则自动追溯。"
            "可添加、删除和调整顺序，确认后严格按表格从上到下读取。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        toolbar = QHBoxLayout()
        add_row_button = QPushButton("添加一组")
        delete_row_button = QPushButton("删除选中")
        move_up_button = QPushButton("上移")
        move_down_button = QPushButton("下移")
        save_list_button = QPushButton("保存路径清单…")
        load_list_button = QPushButton("读取路径清单…")
        for button in (
            add_row_button, delete_row_button, move_up_button, move_down_button,
            save_list_button, load_list_button,
        ):
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        table = QTableWidget(0, 3, dialog)
        table.setHorizontalHeaderLabels([
            "人工复核结果 H5 绝对路径", "选择", "自动识别的原始 H5（只读）",
        ])
        table.verticalHeader().setVisible(False)
        table.setColumnWidth(1, 84)
        table.horizontalHeader().setStretchLastSection(False)
        stretch_mode = (
            QtWidgets.QHeaderView.ResizeMode.Stretch
            if QT_BINDING == "PyQt6" else QtWidgets.QHeaderView.Stretch
        )
        table.horizontalHeader().setSectionResizeMode(0, stretch_mode)
        table.horizontalHeader().setSectionResizeMode(2, stretch_mode)

        def recognized_raw_path(processed_text: str) -> str:
            try:
                processed_path = Path(normalize_h5_path_text(processed_text))
                if not processed_path.is_absolute():
                    return ""
                return str(resolve_original_h5_for_processed(processed_path))
            except (OSError, ValueError):
                return ""

        def refresh_raw_path(row: int):
            if row < 0 or row >= table.rowCount():
                return
            processed_item = table.item(row, 0)
            raw_item = table.item(row, 2)
            processed_text = processed_item.text() if processed_item is not None else ""
            cleaned_processed_text = normalize_h5_path_text(processed_text)
            if processed_item is not None and cleaned_processed_text != processed_text:
                processed_item.setText(cleaned_processed_text)
            if raw_item is None:
                raw_item = QTableWidgetItem("")
                raw_item.setFlags(raw_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row, 2, raw_item)
            raw_item.setText(recognized_raw_path(cleaned_processed_text))

        def choose_h5(button):
            row = table.indexAt(button.pos()).row()
            if row < 0:
                return
            item = table.item(row, 0)
            current = item.text().strip() if item is not None else ""
            start = current if current else str(Path.cwd())
            chosen, _ = QFileDialog.getOpenFileName(
                dialog, "选择人工复核结果 H5", start,
                "HDF5 files (*.h5 *.hdf5);;All files (*.*)",
            )
            if chosen:
                table.item(row, 0).setText(str(Path(chosen).resolve()))
                refresh_raw_path(row)

        def append_pair(processed="", raw=""):
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(str(processed)))
            raw_item = QTableWidgetItem(recognized_raw_path(processed))
            raw_item.setFlags(raw_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 2, raw_item)
            processed_browse = QPushButton("浏览…")
            processed_browse.clicked.connect(
                lambda _checked=False, button=processed_browse:
                choose_h5(button)
            )
            table.setCellWidget(row, 1, processed_browse)

        def table_pairs() -> list[tuple[str, str]]:
            return [
                (
                    table.item(row, 0).text().strip() if table.item(row, 0) else "",
                    table.item(row, 2).text().strip() if table.item(row, 2) else "",
                )
                for row in range(table.rowCount())
            ]

        def replace_pairs(pairs):
            table.setRowCount(0)
            for processed, raw in pairs:
                append_pair(processed, raw)
            if table.rowCount():
                table.setCurrentCell(0, 0)

        initial_pairs = []
        for processed_path in initial_processed_paths:
            initial_pairs.append((str(processed_path.resolve(strict=False)), ""))
        replace_pairs(initial_pairs)
        table.itemChanged.connect(
            lambda item: refresh_raw_path(item.row()) if item.column() == 0 else None
        )
        layout.addWidget(table, 1)

        add_row_button.clicked.connect(lambda: append_pair())

        def delete_selected_rows():
            selected = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
            if not selected and table.currentRow() >= 0:
                selected = [table.currentRow()]
            for row in selected:
                table.removeRow(row)

        delete_row_button.clicked.connect(delete_selected_rows)

        def move_current(delta: int):
            row = table.currentRow()
            target = row + delta
            pairs = table_pairs()
            if row < 0 or target < 0 or target >= len(pairs):
                return
            pairs[row], pairs[target] = pairs[target], pairs[row]
            replace_pairs(pairs)
            table.setCurrentCell(target, 0)

        move_up_button.clicked.connect(lambda: move_current(-1))
        move_down_button.clicked.connect(lambda: move_current(1))

        def save_path_list():
            default_dir = initial_processed_paths[0].parent if initial_processed_paths else Path.cwd()
            filename, _ = QFileDialog.getSaveFileName(
                dialog, "保存 H5 路径清单", str(default_dir / "坏道H5路径清单.json"),
                "JSON files (*.json)",
            )
            if not filename:
                return
            document = {
                "schema": "sd-bad-channel-h5-path-list", "version": 2,
                "reviewed_h5_paths": [
                    normalize_h5_path_text(processed)
                    for processed, _raw in table_pairs()
                ],
            }
            try:
                Path(filename).write_text(
                    json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8",
                )
            except OSError as exc:
                QMessageBox.critical(dialog, "保存失败", str(exc))

        def load_path_list():
            filename, _ = QFileDialog.getOpenFileName(
                dialog, "读取 H5 路径清单", str(Path.cwd()),
                "JSON files (*.json);;All files (*.*)",
            )
            if not filename:
                return
            try:
                document = json.loads(Path(filename).read_text(encoding="utf-8-sig"))
                if (not isinstance(document, dict)
                        or document.get("schema") != "sd-bad-channel-h5-path-list"):
                    raise ValueError("不是有效的坏道 H5 路径清单。")
                version = int(document.get("version", -1))
                if version == 2 and isinstance(document.get("reviewed_h5_paths"), list):
                    pairs = [
                        (str(path), "") for path in document["reviewed_h5_paths"]
                    ]
                elif version == 1 and isinstance(document.get("pairs"), list):
                    # Read path lists saved by the immediately preceding UI;
                    # raw paths are intentionally re-derived using today's rule.
                    pairs = [
                        (str(item.get("reviewed_h5", "")), "")
                        for item in document["pairs"] if isinstance(item, dict)
                    ]
                else:
                    raise ValueError("不支持的路径清单版本。")
                if not pairs:
                    raise ValueError("路径清单中没有文件配对。")
                replace_pairs(pairs)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                QMessageBox.critical(dialog, "读取失败", str(exc))

        save_list_button.clicked.connect(save_path_list)
        load_list_button.clicked.connect(load_path_list)

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel = QPushButton("取消")
        confirm = QPushButton("确认并按顺序执行")
        confirm.setDefault(True)
        actions.addWidget(cancel)
        actions.addWidget(confirm)
        layout.addLayout(actions)
        cancel.clicked.connect(dialog.reject)

        def validate_h5_path(row: int, column: int, label: str) -> Path | None:
            item = table.item(row, column)
            text_value = normalize_h5_path_text(item.text()) if item is not None else ""
            path = Path(text_value) if text_value else None
            if path is None or not path.is_absolute():
                QMessageBox.warning(
                    dialog, f"{label}路径无效",
                    f"第 {row + 1} 行必须填写{label}的绝对路径。",
                )
                table.setCurrentCell(row, column)
                return None
            if path.suffix.lower() not in {".h5", ".hdf5"} or not path.is_file():
                QMessageBox.warning(
                    dialog, f"{label}不存在",
                    f"第 {row + 1} 行不是有效的 H5 文件：\n{path}",
                )
                table.setCurrentCell(row, column)
                return None
            if not h5py.is_hdf5(path):
                QMessageBox.warning(
                    dialog, "H5 文件无效",
                    f"第 {row + 1} 行文件不是有效的 HDF5：\n{path}",
                )
                table.setCurrentCell(row, column)
                return None
            return path.resolve()

        def validate_and_accept():
            if table.rowCount() == 0:
                QMessageBox.warning(dialog, "清单为空", "请至少添加一组 H5 文件。")
                return
            resolved_pairs = []
            for row in range(table.rowCount()):
                processed_path = validate_h5_path(row, 0, "人工复核结果 H5")
                if processed_path is None:
                    return
                try:
                    raw_path = resolve_original_h5_for_processed(processed_path)
                except (OSError, ValueError) as exc:
                    refresh_raw_path(row)
                    QMessageBox.warning(
                        dialog, "无法追溯原始 H5",
                        f"第 {row + 1} 行无法按既定规则找到原始 H5：\n{exc}",
                    )
                    table.setCurrentCell(row, 0)
                    return
                table.item(row, 2).setText(str(raw_path))
                resolved_pairs.append((processed_path, raw_path))
            dialog.resolved_pairs = resolved_pairs
            dialog.accept()

        confirm.clicked.connect(validate_and_accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return (
            [pair[0] for pair in dialog.resolved_pairs],
            [pair[1] for pair in dialog.resolved_pairs],
        )

    def _prompt_bad_channel_grid_axis(
        self, label: str, current: float, *, percent: bool = False,
        integer: bool = False,
    ) -> list[float] | None:
        if integer:
            start, accepted = QInputDialog.getInt(
                self, "扫描起点", f"{label} 起点：", int(current), 1, 64,
            )
            if not accepted:
                return None
            end, accepted = QInputDialog.getInt(
                self, "扫描终点", f"{label} 终点：", int(current), 1, 64,
            )
            if not accepted:
                return None
            step, accepted = QInputDialog.getInt(
                self, "扫描步长", f"{label} 步长：", 1, 1, 64,
            )
            if not accepted:
                return None
            low, high = sorted((start, end))
            values = list(range(low, high + 1, step))
            if values[-1] != high:
                values.append(high)
            return values
        minimum, maximum = (0.01, 100.0) if percent else (1e-12, 1e9)
        start, accepted = QInputDialog.getDouble(
            self, "二维扫描起点", f"{label} 起点：", current,
            minimum, maximum, 8,
        )
        if not accepted:
            return None
        end, accepted = QInputDialog.getDouble(
            self, "二维扫描终点", f"{label} 终点：", current,
            minimum, maximum, 8,
        )
        if not accepted:
            return None
        suggested = abs(end - start) / 4 if end != start else max(current * .1, .01)
        step, accepted = QInputDialog.getDouble(
            self, "二维扫描步长", f"{label} 步长：", suggested,
            1e-12, maximum, 8,
        )
        if not accepted:
            return None
        low, high = sorted((start, end))
        values = [low + index * step for index in range(
            int(np.floor((high - low) / step + 1e-9)) + 1
        )]
        if not values or values[-1] < high - step * 1e-8:
            values.append(high)
        return [float(round(value, 10)) for value in values]

    def run_bad_channel_parameter_sweep(self) -> None:
        if getattr(self, "_bad_channel_parameter_sweep_worker", None) is not None \
                and self._bad_channel_parameter_sweep_worker.isRunning():
            return
        mapping_path = self.remap_edit.text().strip()
        if not mapping_path or not Path(mapping_path).is_file():
            QMessageBox.warning(
                self, "缺少映射文件",
                "批量阈值测试会回溯原始 H5，并在测试前重映射一次。"
                "请先选择有效的通道映射 Excel。",
            )
            return
        keys = list(BAD_CHANNEL_SINGLE_PARAMETERS)
        labels = list(BAD_CHANNEL_SINGLE_PARAMETERS.values())
        label, accepted = QInputDialog.getItem(
            self, "单规则参数精调", "选择本轮单独测试的坏道条件及参数：", labels, 0, False,
        )
        if not accepted:
            return
        parameter_key = keys[labels.index(label)]
        current_settings = self._bad_channel_experiment_settings()
        current_settings = self._prompt_bad_channel_sweep_fixed_settings(
            parameter_key, current_settings,
        )
        if current_settings is None:
            return
        required_toggle = (
            "fast_artifact_check" if parameter_key.startswith("saturation_") else
            "high_frequency_noise_check" if parameter_key.startswith("high_frequency_noise_") else None
        )
        if required_toggle and not current_settings.get(required_toggle):
            QMessageBox.warning(self, "对应规则未启用",
                                f"{label} 所属规则在当前设置中未启用。请先启用该规则，再做单变量扫描。")
            return
        values = self._prompt_bad_channel_grid_axis(
            label, float(current_settings[parameter_key]),
            integer=parameter_key == "discrete_level_threshold",
            percent=parameter_key in {
                "flat_ratio", "saturation_width_percent", "saturation_ratio_threshold",
                "high_frequency_noise_ratio_threshold",
            },
        )
        if values is None:
            return
        baseline_value = float(current_settings[parameter_key])
        if not any(np.isclose(value, baseline_value, rtol=1e-9, atol=1e-12) for value in values):
            values.append(baseline_value)
            values.sort()
        if len(values) > 100:
            QMessageBox.warning(self, "数值过多", "一次最多允许 100 个参数值（包括当前基线值），请增大步长。")
            return
        reusable_pairs = self._bad_channel_sweep_selected_pairs
        if reusable_pairs is None:
            filenames, _ = QFileDialog.getOpenFileNames(
                self, "选择人工复核 H5（也可取消并在下一步读取路径清单）", str(Path.cwd()),
                "HDF5 files (*.h5 *.hdf5);;All files (*.*)",
            )
            selected_pairs = self._prompt_parameter_sweep_raw_paths(filenames)
            if selected_pairs is None:
                return
            reviewed_paths, raw_paths = selected_pairs
            self._bad_channel_sweep_selected_pairs = tuple(zip(reviewed_paths, raw_paths))
        else:
            reviewed_paths = [pair[0] for pair in reusable_pairs]
            raw_paths = [pair[1] for pair in reusable_pairs]
            self.preprocess_status.setText(
                f"参数精调：复用已选择的 {len(reviewed_paths)} 组 H5；不会重新选择文件。"
            )
        filenames = [str(path) for path in reviewed_paths]
        default_name = Path(filenames[0]).parent / f"坏道单变量精调_{parameter_key}_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        output, _ = QFileDialog.getSaveFileName(
            self, "保存参数精调工作簿", str(default_name), "Excel workbook (*.xlsx)",
        )
        if not output:
            return
        worker = BadChannelControlledExperimentWorker(
            filenames, current_settings, mapping_path, raw_paths,
            self._bad_channel_sweep_data_cache,
            parameter_key=parameter_key, values=values,
            allow_partial_review_scope=True,
        )
        worker.progress.connect(self._update_preprocess_progress)
        worker.completed.connect(lambda result, path=output: self._finish_controlled_bad_channel_sweep(path, result))
        worker.failed.connect(self._fail_bad_channel_experiment)
        self._bad_channel_parameter_sweep_worker = worker
        self._begin_shared_preprocess_task(worker, "高召回参数精调")
        self.preprocess_status.setText(
            f"正在控制单变量 {label}：{len(filenames)} 个 H5 × {len(values)} 个值…"
        )
        worker.start()

    @staticmethod
    def _pair_sweep_summary(result: dict) -> list[dict]:
        summaries = []
        for first in result["first_values"]:
            for second in result["second_values"]:
                runs = [row for row in result["runs"]
                        if row["first"] == first and row["second"] == second]
                if not runs:
                    continue
                tp, fp, fn, tn = (
                    sum(row[key] for row in runs) for key in ("tp", "fp", "fn", "tn")
                )
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 1.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
                summaries.append({
                    "first": first, "second": second, "files": len(runs),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": precision, "recall": recall, "f1": f1, "f2": f2,
                    "min_file_recall": min(row["recall"] for row in runs),
                    "predicted_bad_count": tp + fp,
                })
        return summaries

    @classmethod
    def _recommend_bad_channel_pair(cls, result: dict):
        summaries = cls._pair_sweep_summary(result)
        if not summaries:
            return None, "没有成功的二维参数组合。"
        successful_files = len({row["path"] for row in result["runs"]})
        if result["failures"] or successful_files < len(result.get("paths", [])):
            return None, "存在未完成的文件；不在部分数据上推荐参数。"
        best = max(summaries, key=lambda row: (
            row["f2"], row["recall"], row["precision"], -row["predicted_bad_count"],
        ))
        return best, "按单规则 F2 选出的探索性候选；最终参数以三组完整规则联合寻优为准。"

    def _write_bad_channel_pair_workbook(self, output: str, result: dict) -> None:
        from openpyxl import Workbook
        workbook = Workbook()
        recommendation, rationale = self._recommend_bad_channel_pair(result)
        main = workbook.active
        main.title = "单组候选"
        main.append(["二维规则", result["label"]])
        main.append(["第一参数", result["first_key"]])
        main.append(["第二参数", result["second_key"]])
        main.append(["单组候选第一参数", recommendation["first"] if recommendation else ""])
        main.append(["单组候选第二参数", recommendation["second"] if recommendation else ""])
        main.append(["状态", rationale])
        main.append(["定义", "仅使用本组成对参数判定；单组候选不是最终参数。三组规则按或（OR）合并后在完整网格中寻优并验证。"])
        main.append(["平直定义", "约1秒窗口PTP≤6×标准差参数；平直窗口占比≥时间比例参数。"])
        main.append(["评估范围", "只评估有原始物理通道映射到的目标通道；排除重映射填零空位。"])
        if result["group"] == "near_target":
            main.append(["附近目标数值", result["base_settings"].get("high_frequency_noise_target", "")])
            main.append(["单位核对", "重映射信号按H5元信息转换为mV；请确认目标数值与数据量级一致。"])
        summary = workbook.create_sheet("参数组合汇总")
        summary.append(["第一参数", "第二参数", "文件数", "TP", "FP", "FN", "TN",
                        "Precision", "Recall", "F1", "F2", "单文件最低Recall", "待复核坏道数"])
        for row in self._pair_sweep_summary(result):
            summary.append([row[key] for key in (
                "first", "second", "files", "tp", "fp", "fn", "tn",
                "precision", "recall", "f1", "f2", "min_file_recall",
                "predicted_bad_count",
            )])
        per_file = workbook.create_sheet("逐文件结果")
        per_file.append(["人工复核H5", "原始H5", "第一参数", "第二参数", "评估通道", "未纳入评估通道数",
                         "历史坏道", "预测坏道", "TP", "FP", "FN", "TN",
                         "Precision", "Recall", "F1", "F2", "预测坏道ID"])
        for row in result["runs"]:
            per_file.append([row["file"], row["raw_path"], row["first"], row["second"],
                             row["evaluated_count"], row["excluded_unmapped_count"],
                             row["reference_bad_count"],
                             row["predicted_bad_count"], row["tp"], row["fp"],
                             row["fn"], row["tn"], row["precision"], row["recall"],
                             row["f1"], row["f2"], self._format_id_list(row["predicted_bad_ids"])])
        detail = workbook.create_sheet("坏道明细")
        detail.append(["文件", "第一参数", "第二参数", "通道", "完整理由"])
        for row in result["runs"]:
            for channel, reason in sorted(row["reasons"].items()):
                detail.append([row["file"], row["first"], row["second"], channel, reason])
        failure = workbook.create_sheet("失败文件")
        failure.append(["文件", "路径", "错误"])
        for row in result["failures"]:
            failure.append([row["file"], row["path"], row["error"]])
        self._style_bad_channel_workbook(workbook)
        workbook.save(output)

    def _finish_bad_channel_pair_sweep(self, output: str, result: dict) -> None:
        try:
            self._write_bad_channel_pair_workbook(output, result)
        except Exception as exc:
            self._fail_bad_channel_experiment(str(exc))
            return
        recommendation, rationale = self._recommend_bad_channel_pair(result)
        self.preprocess_progress.setValue(100)
        self.preprocess_status.setText(f"二维参数精调完成：{output}")
        if recommendation is None:
            QMessageBox.warning(self, "二维精调未完成", f"{rationale}\n\n工作簿：\n{output}")
            return
        cohort = tuple((str(reviewed.resolve()), str(raw.resolve()),
                        reviewed.stat().st_mtime_ns, raw.stat().st_mtime_ns)
                       for reviewed, raw in self._bad_channel_sweep_selected_pairs)
        signature = (cohort, self._remap_signature(self.remap_edit.text().strip()))
        if signature != self._bad_channel_sweep_cohort_signature:
            self._bad_channel_pair_results.clear()
            self._bad_channel_sweep_cohort_signature = signature
        self._bad_channel_pair_results[result["group"]] = result
        summary_text = (
            f"{result['label']}：{recommendation['first']:g} × {recommendation['second']:g}\n"
            f"Recall={recommendation['recall']:.1%}，Precision={recommendation['precision']:.1%}\n"
            f"{rationale}\n\n工作簿：\n{output}"
        )
        if len(self._bad_channel_pair_results) < len(BAD_CHANNEL_PARAMETER_PAIRS):
            QMessageBox.information(self, "二维精调完成", summary_text)
            return
        worker = BadChannelCombinedValidationWorker(
            [pair[0] for pair in self._bad_channel_sweep_selected_pairs],
            self._bad_channel_experiment_settings(),
            dict(self._bad_channel_pair_results),
            self.remap_edit.text().strip(),
            [pair[1] for pair in self._bad_channel_sweep_selected_pairs],
            self._bad_channel_sweep_data_cache,
        )
        worker.progress.connect(self._update_preprocess_progress)
        worker.completed.connect(
            lambda combined, path=output, message=summary_text:
            self._finish_bad_channel_combined_validation(path, message, combined)
        )
        worker.failed.connect(self._fail_bad_channel_experiment)
        self._bad_channel_combined_worker = worker
        self._begin_shared_preprocess_task(worker, "三组坏道规则组合验证")
        worker.start()

    def _finish_bad_channel_combined_validation(
        self, output: str, summary_text: str, combined: dict,
    ) -> None:
        from openpyxl import load_workbook
        try:
            workbook = load_workbook(output)
            optimization = combined["optimization"]
            chosen = workbook.create_sheet("三组联合最优参数", 0)
            chosen.append(["范围", "本次已扫描网格中的三组组合，不代表连续参数空间全局最优"])
            chosen.append(["组合数", optimization["combinations_tested"]])
            chosen.append(["高召回条件", "总体 Recall≥98%，每文件 Recall≥95%"])
            chosen.append(["是否达标", "是" if optimization["qualified"] else "否；仅为当前网格相对最优"])
            chosen.append(["联合 TP", optimization["tp"]])
            chosen.append(["联合 FP", optimization["fp"]])
            chosen.append(["联合 FN", optimization["fn"]])
            chosen.append(["联合 Precision", optimization["precision"]])
            chosen.append(["联合 Recall", optimization["recall"]])
            chosen.append(["最低单文件 Recall", optimization["min_file_recall"]])
            chosen.append(["规则组", "第一参数", "第一参数值", "第二参数", "第二参数值"])
            for group, (first, second) in combined["recommendations"].items():
                _label, first_key, second_key = BAD_CHANNEL_PARAMETER_PAIRS[group]
                chosen.append([_label, BAD_CHANNEL_PARAMETER_LABELS[first_key], first,
                               BAD_CHANNEL_PARAMETER_LABELS[second_key], second])
            chosen.append(["2.5mV目标数值", self._bad_channel_pair_results["near_target"]["base_settings"].get(
                "high_frequency_noise_target", "")])
            sheet = workbook.create_sheet("三组规则合并验证")
            sheet.append(["文件", "原始H5", "评估通道", "未纳入评估通道数", "人工坏道", "合并预测坏道",
                          "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "F2",
                          "平直坏道ID", "贴底坏道ID", "2.5mV坏道ID", "合并坏道ID"])
            for row in combined["rows"]:
                sheet.append([row["file"], row["raw_path"], row["evaluated_count"],
                              row["excluded_unmapped_count"],
                              row["reference_bad_count"], row["predicted_bad_count"],
                              row["tp"], row["fp"], row["fn"], row["tn"],
                              row["precision"], row["recall"], row["f1"], row["f2"],
                              self._format_id_list(row["rule_bad_ids"]["flat"]),
                              self._format_id_list(row["rule_bad_ids"]["saturation"]),
                              self._format_id_list(row["rule_bad_ids"]["near_target"]),
                              self._format_id_list(row["predicted_bad_ids"])])
            if combined["failures"]:
                errors = workbook.create_sheet("组合验证失败")
                errors.append(["文件", "错误"])
                for row in combined["failures"]:
                    errors.append([row["file"], row["error"]])
            self._style_bad_channel_workbook(workbook)
            workbook.save(output)
        except Exception as exc:
            self._fail_bad_channel_experiment(str(exc))
            return
        tp, fp, fn = (sum(row[key] for row in combined["rows"])
                      for key in ("tp", "fp", "fn"))
        recall = tp / (tp + fn) if tp + fn else 1.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        self.preprocess_status.setText(f"三组规则合并验证完成：{output}")
        choices = "\n".join(
            f"{BAD_CHANNEL_PARAMETER_PAIRS[group][0]}：{first:g} × {second:g}"
            for group, (first, second) in combined["recommendations"].items()
        )
        qualified = combined["optimization"]["qualified"] and not combined["failures"]
        QMessageBox.information(
            self, "二维精调与组合验证完成",
            f"三组规则在已扫描网格内联合寻优：\n{choices}\n\n"
            f"组合验证：Recall={recall:.1%}，Precision={precision:.1%}，FN={fn}。\n"
            f"高召回条件：{'达标' if qualified else '未达标；仅为当前网格相对最优'}。\n"
            f"工作簿：\n{output}"
            + (f"\n组合验证失败文件：{len(combined['failures'])}" if combined["failures"] else ""),
        )

    @staticmethod
    def _parameter_sweep_summary(result: dict) -> list[dict]:
        summaries = []
        for value in result["values"]:
            runs = [row for row in result["runs"] if np.isclose(row["parameter_value"], value)]
            if not runs:
                continue
            tp = sum(row["tp"] for row in runs)
            fp = sum(row["fp"] for row in runs)
            fn = sum(row["fn"] for row in runs)
            tn = sum(row["tn"] for row in runs)
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 1.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
            summaries.append({
                "value": value, "files": len(runs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": recall, "f1": f1, "f2": f2,
                "min_file_recall": min(row["recall"] for row in runs),
                "predicted_bad_count": sum(row["predicted_bad_count"] for row in runs),
            })
        return summaries

    @classmethod
    def _recommend_parameter_sweep(cls, result: dict):
        summaries = cls._parameter_sweep_summary(result)
        strict = [row for row in summaries if row["recall"] >= .98 and row["min_file_recall"] >= .95]
        if strict:
            return max(strict, key=lambda row: (row["precision"], -row["predicted_bad_count"], row["f2"])), "满足总体召回≥98%、单文件最低召回≥95%，再优先精确率与较少复核量"
        if not summaries:
            return None, "没有可比较的成功结果"
        return max(summaries, key=lambda row: (row["recall"], row["precision"], row["f2"])), "没有参数同时达到召回约束，暂按召回率优先推荐"

    def _write_bad_channel_parameter_sweep_workbook(self, output: str, result: dict) -> None:
        from openpyxl import Workbook
        workbook = Workbook()
        recommendation_sheet = workbook.active
        recommendation_sheet.title = "推荐结果"
        recommendation, rationale = self._recommend_parameter_sweep(result)
        recommendation_sheet.append(["扫描参数", result["parameter_label"]])
        recommendation_sheet.append(["判断模式", "单因素：仅所选条件参与坏道判定"])
        recommendation_sheet.append(["推荐参数值", "" if recommendation is None else recommendation["value"]])
        recommendation_sheet.append(["推荐逻辑", rationale])
        recommendation_sheet.append(["推荐门槛", "总体 Recall≥98% 且每个文件 Recall≥95%；未达到时按 Recall、Precision、F2 依次排序"])
        recommendation_sheet.append(["目标", "高召回前提下提高精确率，减少在健康通道中的人工查询"])
        summary_sheet = workbook.create_sheet("参数汇总")
        summary_sheet.append(["参数值", "文件数", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "F2", "单文件最低Recall", "待复核坏道总数"])
        for row in self._parameter_sweep_summary(result):
            summary_sheet.append([row[key] for key in (
                "value", "files", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "f2",
                "min_file_recall", "predicted_bad_count",
            )])
        run_sheet = workbook.create_sheet("逐文件结果")
        run_sheet.append(["处理结果文件", "追溯原始H5", "参数值", "参考来源", "评估通道", "历史坏道", "人工覆盖", "预测坏道", "Precision", "Recall", "F1", "F2", "TP", "FP", "FN", "TN", "预测坏道ID", "较前值新增ID", "较前值移除ID"])
        for row in result["runs"]:
            run_sheet.append([
                row["file"], row.get("raw_path", ""), row["parameter_value"], row["reference_source"], row["evaluated_count"],
                row["reference_bad_count"], row["manual_override_count"], row["predicted_bad_count"],
                row["precision"], row["recall"], row["f1"], row["f2"], row["tp"], row["fp"], row["fn"], row["tn"],
                self._format_id_list(row["predicted_bad_ids"]), self._format_id_list(row["added_vs_previous"]),
                self._format_id_list(row["removed_vs_previous"]),
            ])
        detail_sheet = workbook.create_sheet("坏道明细")
        detail_sheet.append(["文件", "参数值", "通道", "历史最终坏道", "触发条件", "完整理由"])
        for run in result["runs"]:
            for row in run["details"]:
                detail_sheet.append([run["file"], run["parameter_value"], row["channel"], row["reference_bad"], row["rules"], row["reason"]])
        failure_sheet = workbook.create_sheet("失败文件")
        failure_sheet.append(["文件", "路径", "错误"])
        for row in result["failures"]:
            failure_sheet.append([row["file"], row["path"], row["error"]])
        settings_sheet = workbook.create_sheet("固定参数")
        settings_sheet.append(["参数", "值"])
        for key, value in result["base_settings"].items():
            settings_sheet.append([key, value])
        self._style_bad_channel_workbook(workbook)
        workbook.save(output)

    def _finish_bad_channel_parameter_sweep(self, output: str, result: dict) -> None:
        try:
            self._write_bad_channel_parameter_sweep_workbook(output, result)
        except Exception as exc:
            self._fail_bad_channel_experiment(str(exc))
            return
        recommendation, rationale = self._recommend_parameter_sweep(result)
        text = "无可推荐结果" if recommendation is None else (
            f"推荐 {result['parameter_label']} = {recommendation['value']:g}\n"
            f"Precision={recommendation['precision']:.1%}，Recall={recommendation['recall']:.1%}，"
            f"单文件最低Recall={recommendation['min_file_recall']:.1%}\n{rationale}"
        )
        self.preprocess_progress.setValue(100)
        self.preprocess_status.setText(f"参数精调完成；结果：{output}")
        QMessageBox.information(self, "高召回参数精调完成", f"{text}\n\n工作簿：\n{output}")

    def _layout_for_source(self, source):
        if self.channel_layout_ids is not None:
            return self.channel_layout_ids
        # Before a mapping spreadsheet is selected, retain the historic
        # default 20×26 physical/FPC order rather than compacting a partial
        # custom data file into the first cells.
        return np.arange(1, 521, dtype=np.int64)

    def import_custom_channels(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "导入已处理的自定义通道 HDF5", str(Path.cwd()), "HDF5 files (*.h5 *.hdf5);;All files (*.*)"
        )
        if not filenames:
            return
        try:
            imported = self._open_custom_h5_source(filenames)
            meta = imported.metadata
            # A reload of individual ``*_raw.h5`` files becomes the actual
            # preprocessing input.  Its physical_channel_id metadata is kept
            # by CustomChannelH5Source and is used by the sparse remapper.
            self.source = imported
            self.remapped_source = None
            self.preprocessed_source = imported
            self.active_source = imported
            self._preprocess_lowpass_channel_ids = set()
            self._preprocess_lowpass_high_hz = None
            self._preprocess_filtered_channel_ids = set()
            self._preprocess_notched_channel_ids = set()
            self._apply_loaded_timing(meta)
            self.preprocessed_preview.set_source(imported, defer_initial_draw=True)
            self._update_preprocess_comparison_preview()
            self.preprocess_status.setText(
                f"已承接自定义通道数据：{len(filenames)} 个文件；{meta.channels} 通道，"
                f"实际通道号示例：{', '.join(map(str, meta.channel_ids[:8]))}"
            )
            self._set_preprocess_data_state(
                f"数据状态：已导入 {meta.channels} 个自定义原始通道；保留实际通道号，尚未重新映射。"
            )
            self.statusBar().showMessage("已导入自定义通道数据；通道号由 HDF5 channel_ids 保持。")
            self._update_file_banners()
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))

    @staticmethod
    def _open_custom_h5_source(filenames):
        """Open total H5 or the legacy selected per-channel H5 collection."""
        if len(filenames) == 1:
            source = LazyH5Source()
            try:
                source.open(filenames[0])
                return source
            except ValueError:
                pass
        return CustomChannelH5Source(filenames)

    def run_filter_overlap_test(self) -> None:
        """Run a read-only overlap-size experiment with current filter settings."""
        source = self.remapped_source if self.remapped_source is not None and self.remapped_source.loaded else self.source
        if source is None or not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载数据，再测试重叠裁剪误差。")
            return
        try:
            selected = self._selected_preprocess_columns(source)
            available = np.arange(source.metadata.channels, dtype=np.int64) if selected is None else np.asarray(selected, dtype=np.int64)
            if not available.size:
                raise ValueError("没有可用于测试的通道。")
            representative_indices = np.unique(np.linspace(0, available.size - 1, min(3, available.size), dtype=int))
            columns = available[representative_indices]
            scenarios = [
                {
                    "name": "LFP 0.5-300 Hz + 50 Hz陷波", "mode": "bandpass",
                    "low": .5, "high": 300.0, "hp_order": 3, "lp_order": 5,
                    "notch": True, "notch_frequency": 50.0, "notch_q": 30.0,
                    "notch_harmonics": 1,
                    "overlap_candidates_sec": [1, 2, 5, 10, 20, 30, 40, 60, 120],
                },
                {
                    "name": "Spike 500-3000 Hz", "mode": "bandpass",
                    "low": 500.0, "high": 3000.0, "hp_order": 3, "lp_order": 5,
                    "notch": False, "notch_frequency": 50.0, "notch_q": 30.0,
                    "notch_harmonics": 1,
                    "overlap_candidates_sec": [.05, .1, .25, .5, 1, 2, 5, 10],
                },
            ]
        except ValueError as exc:
            QMessageBox.warning(self, "测试参数无效", str(exc))
            return
        if getattr(self, "_filter_overlap_test_worker", None) is not None and self._filter_overlap_test_worker.isRunning():
            return
        worker = FilterOverlapTestWorker(source, columns, scenarios)
        worker.completed.connect(self._finish_filter_overlap_test)
        worker.failed.connect(self._fail_filter_overlap_test)
        worker.finished.connect(lambda: self.filter_overlap_test_button.setEnabled(True))
        self._filter_overlap_test_worker = worker
        self.filter_overlap_test_button.setEnabled(False)
        self.preprocess_status.setText("正在分别测试 LFP 和 Spike 的重叠裁剪误差；0.5 Hz LFP 测试可能需要较长时间…")
        worker.start()

    def _fail_filter_overlap_test(self, error: str) -> None:
        self.preprocess_status.setText(f"重叠裁剪测试失败：{error}")
        QMessageBox.warning(self, "重叠裁剪测试失败", error)

    def _finish_filter_overlap_test(self, output: dict) -> None:
        self.filter_overlap_test_output = output
        self.preprocess_status.setText(
            f"LFP/Spike 重叠裁剪测试完成：核心比较区 {output['core_sec']:.3f} s。"
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("重叠裁剪 vs 整段 filtfilt")
        dialog.resize(1220, 840)
        layout = QVBoxLayout(dialog)
        ranges = "；".join(f"{name}：{bounds[0]:.3f}–{bounds[1]:.3f} s" for name, bounds in output["test_ranges"].items())
        skipped_text = ""
        if output.get("skipped"):
            skipped_text = "\n未执行（数据时长不足）：" + "；".join(
                f"{item['branch']} {item['overlap_sec']:g} s需至少{item['required_duration_sec']:.1f} s"
                for item in output["skipped"]
            )
        info = QLabel(
            f"参考：测试区间整体 sosfiltfilt；比较：核心 {output['core_sec']:g} s 加左右重叠后滤波并裁边。\n"
            f"代表通道：{self._format_channel_scope(output['channel_ids'])}。\n测试区间：{ranges}。\n"
            "LFP：0.5–300 Hz（高通3阶、低通5阶）+ 50 Hz陷波（Q=30）；"
            "Spike：500–3000 Hz（高通3阶、低通5阶）。"
            f"{skipped_text}"
        )
        info.setWordWrap(True); layout.addWidget(info)
        table = QTableWidget(len(output["rows"]), 7, dialog)
        table.setHorizontalHeaderLabels(["分支", "通道", "重叠（s）", "RMSE（mV）", "最大残差（mV）", "相对 RMSE（%）", "相关系数"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row_index, row in enumerate(output["rows"]):
            values = (
                row["branch"], f"ch{row['channel']}", f"{row['overlap_sec']:g}", f"{row['rmse_mv']:.9g}",
                f"{row['max_abs_residual_mv']:.9g}", f"{row['relative_rmse_percent']:.6g}",
                f"{row['correlation']:.9g}",
            )
            for column, value in enumerate(values):
                table.setItem(row_index, column, QTableWidgetItem(value))
        table.resizeColumnsToContents(); layout.addWidget(table, 1)
        plot = pg.PlotWidget(dialog); plot.setBackground("w"); plot.addLegend()
        branch_names = list(output["test_ranges"])
        for color_index, branch in enumerate(branch_names):
            branch_rows = [row for row in output["rows"] if row["branch"] == branch]
            overlaps = sorted({float(row["overlap_sec"]) for row in branch_rows})
            worst = [max(float(row["relative_rmse_percent"]) for row in branch_rows if float(row["overlap_sec"]) == overlap) for overlap in overlaps]
            plot.plot(overlaps, worst, pen=pg.mkPen(pg.intColor(color_index, hues=len(branch_names)), width=2), symbol="o", name=branch)
        plot.setLogMode(x=True, y=True)
        plot.setLabel("bottom", "单侧重叠", units="s")
        plot.setLabel("left", "代表通道中最差相对 RMSE", units="%")
        layout.addWidget(plot, 1)
        controls = QHBoxLayout(); export = QPushButton("导出测试 CSV"); close = QPushButton("关闭")
        controls.addWidget(export); controls.addStretch(1); controls.addWidget(close); layout.addLayout(controls)

        def export_csv():
            filename, _ = QFileDialog.getSaveFileName(
                dialog, "导出重叠裁剪测试", str(Path.cwd() / "filter_overlap_test.csv"), "CSV files (*.csv)"
            )
            if not filename:
                return
            import csv
            export_rows = [{
                "test_status": "tested",
                "core_sec": output["core_sec"],
                "test_start_sec": output["test_ranges"][row["branch"]][0],
                "test_end_sec": output["test_ranges"][row["branch"]][1],
                "required_duration_sec": "",
                "available_duration_sec": "",
                **row,
            } for row in output["rows"]]
            export_rows.extend({
                "test_status": "skipped_insufficient_duration",
                "core_sec": output["core_sec"], "test_start_sec": "", "test_end_sec": "",
                "branch": item["branch"], "channel": "", "overlap_sec": item["overlap_sec"],
                "rmse_mv": "", "max_abs_residual_mv": "", "relative_rmse_percent": "", "correlation": "",
                "required_duration_sec": item["required_duration_sec"],
                "available_duration_sec": item["available_duration_sec"],
            } for item in output.get("skipped", []))
            with open(filename, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(export_rows[0]))
                writer.writeheader(); writer.writerows(export_rows)

        export.clicked.connect(export_csv); close.clicked.connect(dialog.accept)
        dialog.exec()

    def _dual_branch_stream_settings(self, fs: float) -> tuple[dict, bool]:
        """Validate and freeze the visible streaming-product parameters."""
        try:
            lfp_low = float(self.stream_lfp_low_var.text())
            lfp_high = float(self.stream_lfp_high_var.text())
            lfp_overlap = float(self.stream_lfp_overlap_var.text())
            spike_low = float(self.stream_spike_low_var.text())
            spike_high = float(self.stream_spike_high_var.text())
            spike_overlap = float(self.stream_spike_overlap_var.text())
            notch_frequency = float(self.stream_notch_frequency_var.text())
            notch_q = float(self.stream_notch_q_var.text())
            notch_harmonics = int(self.stream_notch_harmonics_var.text())
            max_workers = int(self.stream_max_workers_var.text())
            memory_budget_gb = float(self.stream_memory_budget_gb_var.text())
        except ValueError as exc:
            raise ValueError("流式滤波参数必须是有效数字。") from exc
        nyquist = float(fs) / 2.0
        if not (0 < lfp_low < lfp_high < nyquist):
            raise ValueError(f"LFP频段必须满足 0 < 低截止 < 高截止 < {nyquist:g} Hz。")
        if not (0 < spike_low < spike_high < nyquist):
            raise ValueError(f"Spike频段必须满足 0 < 低截止 < 高截止 < {nyquist:g} Hz。")
        if lfp_overlap < 0 or spike_overlap < 0:
            raise ValueError("流式单侧重叠时间不能小于0秒。")
        if notch_frequency <= 0 or notch_q <= 0 or notch_harmonics < 1:
            raise ValueError("陷波频率、Q和谐波数必须大于0。")
        if max_workers < 1 or max_workers > 64:
            raise ValueError("流式最大并行任务必须在1–64之间。")
        if memory_budget_gb < 0.25 or memory_budget_gb > 80:
            raise ValueError("流式并行内存预算必须在0.25–80 GB之间。")
        common_notch = {
            "notch_frequency": notch_frequency,
            "notch_q": notch_q,
            "notch_harmonics": notch_harmonics,
        }
        return {
            "lfp": {
                "low": lfp_low, "high": lfp_high, "overlap_sec": lfp_overlap,
                "notch": self.stream_lfp_notch_var.isChecked(), **common_notch,
            },
            "spike": {
                "low": spike_low, "high": spike_high, "overlap_sec": spike_overlap,
                "notch": self.stream_spike_notch_var.isChecked(), **common_notch,
            },
            "runtime": {
                "max_workers": max_workers,
                "memory_budget_gb": memory_budget_gb,
            },
        }, self.stream_lfp_car_var.isChecked()

    @staticmethod
    def _workflow_window_flags():
        window_type = QtCore.Qt.WindowType if QT_BINDING == "PyQt6" else QtCore.Qt
        return (
            window_type.Window
            | window_type.WindowTitleHint
            | window_type.WindowSystemMenuHint
            | window_type.WindowMinimizeButtonHint
            | window_type.WindowCloseButtonHint
        )

    def _source_processing_descriptions(self, source) -> list[str]:
        meta = getattr(source, "metadata", None)
        provenance = getattr(meta, "provenance", None) or {}
        descriptions = []
        for operation in provenance.get("operations", []):
            if not isinstance(operation, dict):
                continue
            description = self._describe_preprocess_operation(operation)
            descriptions.append(description or str(operation.get("name", "未命名处理")))
        return descriptions

    def _show_generation_workflow_dialog(
        self, *, title: str, heading: str, source, steps: list[tuple[str, str]],
        after_text: str, action_text: str, action_callback, dialog_attribute: str,
    ) -> None:
        """Show an inspectable before/during/after processing chain before export."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setWindowFlags(self._workflow_window_flags())
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.resize(1120, 760)
        layout = QVBoxLayout(dialog)

        banner = QLabel(heading)
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setStyleSheet(
            "background:#087be5; color:white; border-radius:9px; padding:11px 14px;"
            "font-size:19px; font-weight:700;"
        )
        layout.addWidget(banner)

        meta = getattr(source, "metadata", None)
        input_steps = self._source_processing_descriptions(source)
        input_history = "".join(
            f"<div style='margin-top:5px;'><b>{index}.</b> {html.escape(text)}</div>"
            for index, text in enumerate(input_steps, 1)
        )
        if not input_history:
            input_history = "<div style='margin-top:5px;'>未记录额外处理步骤（按当前数据源状态使用）。</div>"
        if meta is not None:
            source_file = html.escape(Path(str(meta.path)).name)
            source_detail = (
                f"<div style='margin-top:7px;'><b>文件：</b>{source_file}</div>"
                f"<div style='margin-top:6px;'><b>数据尺寸：</b>{meta.rows:,} 采样点 × {meta.channels} 通道</div>"
                f"<div style='margin-top:6px;'><b>采样率：</b>{meta.fs:g} Hz"
                f"　　<b>时长：</b>{meta.rows / meta.fs:.3f} 秒</div>"
            )
        else:
            source_detail = "<div style='margin-top:7px;'>没有可用的数据源元信息。</div>"
        before = QLabel(
            "<div style='font-size:18px; font-weight:700; color:#124a7a;'>生成前：实际输入数据</div>"
            + source_detail
            + "<div style='margin-top:12px; font-size:16px; font-weight:700;'>已有处理链</div>"
            + input_history
        )
        before.setWordWrap(True)
        before.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        before.setStyleSheet(
            "QLabel { background:#eef6ff; border:2px solid #75aee0; border-radius:10px;"
            "padding:14px 16px; font-size:15px; color:#17212b; }"
        )
        layout.addWidget(before)

        step_label = QLabel("【本次生成：逐步执行内容】")
        step_label.setStyleSheet("font-size:17px; font-weight:700; margin-top:4px;")
        layout.addWidget(step_label)
        table = QTableWidget(len(steps), 3, dialog)
        table.setHorizontalHeaderLabels(["顺序", "阶段", "当前实际设置与处理参数"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        for row, (stage, processing) in enumerate(steps):
            for column, value in enumerate((str(row + 1), stage, processing)):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(row, column, item)
        table.setColumnWidth(0, 60)
        table.setColumnWidth(1, 190)
        table.horizontalHeader().setStretchLastSection(True)
        table.setStyleSheet(
            "QTableWidget { font-size:14px; gridline-color:#c7d3df; }"
            "QHeaderView::section { font-size:14px; font-weight:700; padding:7px;"
            "background:#e7f0f8; border:1px solid #c7d3df; }"
        )
        table.resizeRowsToContents()
        layout.addWidget(table, 1)

        after = QLabel("【生成后：文件与后续处理】\n" + after_text)
        after.setWordWrap(True)
        after.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        after.setStyleSheet(
            "background:#f3fff5; border:2px solid #8bc798; border-radius:10px;"
            "padding:12px 14px; font-size:14px; color:#17212b;"
        )
        layout.addWidget(after)

        cancel_button = QPushButton("关闭")
        action_button = QPushButton(action_text)
        action_button.setDefault(True)
        cancel_button.clicked.connect(dialog.close)

        def continue_action() -> None:
            dialog.close()
            QTimer.singleShot(0, action_callback)

        action_button.clicked.connect(continue_action)
        controls = QHBoxLayout()
        controls.addStretch(1); controls.addWidget(cancel_button); controls.addWidget(action_button)
        layout.addLayout(controls)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        setattr(self, dialog_attribute, dialog)

    def _bad_channel_workflow_steps(self) -> list[tuple[str, str]]:
        """Describe every visible QC choice using the values currently in the UI."""
        checked = lambda widget: "☑ 已勾选" if widget.isChecked() else "☐ 未勾选"
        return [
            (
                "坏道判断｜贴底饱和",
                f"{checked(self.bad_fast_artifact_check_var)}；贴底区间宽度="
                f"PTP 的 {self.bad_saturation_width_percent_var.text() or '1.0'}%，"
                f"贴底采样点比例阈值={self.bad_saturation_ratio_threshold_var.text() or '45'}%。",
            ),
            (
                "坏道判断｜2.5mV附近集中",
                f"{checked(self.bad_high_frequency_noise_check_var)}；目标值="
                f"{self.bad_high_frequency_noise_target_var.text() or '2500'}，"
                f"容差=±{self.bad_high_frequency_noise_tolerance_var.text() or '1'}，"
                f"命中比例 > {self.bad_high_frequency_noise_ratio_threshold_var.text() or '50'}% 时直接判为坏道。",
            ),
            (
                "坏道判断｜计算方式",
                f"{checked(self.bad_parallel_var)}并行；workers="
                f"{self.bad_workers_var.text() or '0'}（0 表示自动选择）。",
            ),
        ]

    def _dual_stream_input_source(self):
        """Use unfiltered input so interactive preprocessing never leaks into streaming."""
        # Remapping only changes channel identity/order and is therefore kept.
        # Ordinary filtering, CAR and ICA all live in ``preprocessed_source``;
        # deliberately skip it so each stream branch starts from original
        # samples and applies only its own frozen LFP/Spike settings.
        for source in (self.remapped_source, self.source):
            if source is not None and source.loaded:
                return source
        return None

    def _dual_stream_input_description(self, source) -> str:
        if source is self.remapped_source:
            return (
                "使用重映射后的原始数据作为共同输入；忽略“执行滤波”的模式、截止频率及其结果，"
                "也不继承普通流程中的 CAR/ICA；LFP 与 Spike 将分别按流式专用参数重新处理。"
            )
        return (
            "使用最初加载的数据作为共同输入；忽略“执行滤波”的模式、截止频率及其结果，"
            "也不继承普通流程中的 CAR/ICA；LFP 与 Spike 将分别按流式专用参数重新处理。"
        )

    def show_dual_stream_export_workflow(self) -> None:
        source = self._dual_stream_input_source()
        if source is None or not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载原始 HDF5 数据。")
            return
        try:
            settings, apply_lfp_car = self._dual_branch_stream_settings(source.metadata.fs)
        except ValueError as exc:
            QMessageBox.warning(self, "流式处理参数无效", str(exc))
            return
        lfp, spike, runtime = settings["lfp"], settings["spike"], settings["runtime"]
        notch = lambda branch: (
            f"{branch['notch_frequency']:g} Hz 陷波，Q={branch['notch_q']:g}，"
            f"谐波数={branch['notch_harmonics']}" if branch["notch"] else "不做陷波"
        )
        steps = [
            ("确定输入", self._dual_stream_input_description(source)),
            ("当前 QC 结果", f"检查={'已完成' if self._bad_channel_check_completed else '未完成'}；健康 {len(self.good_channel_ids)}、坏道 {len(self.bad_channel_ids)}、人工覆盖 {len(self.manual_channel_overrides)}。"),
            *self._bad_channel_workflow_steps(),
            ("LFP 重参考", "☑ 已勾选；对健康通道执行 Leave-one-out median CAR，坏道不参与。" if apply_lfp_car else "☐ 未勾选；不执行 LFP CAR。"),
            ("LFP 滤波", f"{lfp['low']:g}–{lfp['high']:g} Hz 带通；{notch(lfp)}；单侧重叠 {lfp['overlap_sec']:g} 秒。"),
            ("Spike 滤波", f"{spike['low']:g}–{spike['high']:g} Hz 带通；{notch(spike)}；单侧重叠 {spike['overlap_sec']:g} 秒。"),
            ("流式计算", f"并行任务上限={runtime['max_workers']}；临时数组内存预算={runtime['memory_budget_gb']:g} GB。"),
        ]
        after_text = (
            "1. 在已加载 H5 同级的 processed 目录生成 LFP H5、Spike H5、项目 JSON 和处理记录 CSV。\n"
            "2. LFP 文件自动接入通用预览与 LFP 分析；Spike 文件自动接入 Spike 分析。\n"
            "3. 两份 H5 均为懒加载，后续只读取所需时间窗和通道，不会一次性占满内存。\n"
            "4. 原始文件不会被修改；若中途失败，未完成文件保留为 .partial.h5 便于识别。"
        )
        self._show_generation_workflow_dialog(
            title="流式生成 LFP + Spike H5：处理流程",
            heading="流式生成 LFP + Spike H5",
            source=source, steps=steps, after_text=after_text,
            action_text="开始生成", action_callback=self.run_dual_branch_stream_export,
            dialog_attribute="_dual_stream_workflow_dialog",
        )

    def show_preprocessed_export_workflow(self) -> None:
        source = self.preprocessed_source
        if source is None or not source.loaded:
            QMessageBox.information(self, "尚无预处理结果", "请先完成滤波、重参考或 ICA 后再导出。")
            return
        steps = [
            ("确定输入", "使用当前 preprocessed_source 的完整通道矩阵，保持当前采样点、通道顺序和采样率。"),
            ("当前 QC 结果", f"检查={'已完成' if self._bad_channel_check_completed else '未完成'}；健康 {len(self.good_channel_ids)}、坏道 {len(self.bad_channel_ids)}、人工覆盖 {len(self.manual_channel_overrides)}。"),
            *self._bad_channel_workflow_steps(),
            ("写入数据", "分块写入当前预处理结果；已处理通道写入处理值，其余通道保持当前原值。"),
            ("写入元数据", "保存采样率、单位、时间偏移、通道号、计时信息和完整 provenance 处理链。"),
            ("生成处理记录", "把来源、输出文件、处理时间、处理步骤、参数和通道质量结果写入同名 CSV。"),
            ("完成校验", "关闭并刷新输出文件；原始文件与当前内存数据均不修改。"),
        ]
        after_text = (
            "1. 继续后先选择 H5 保存位置；程序同时生成同名的处理记录 CSV。\n"
            "2. 导出只是保存当前结果，不再额外执行滤波、CAR 或 ICA。\n"
            "3. 导出完成后，当前预览和分析数据源保持不变，可继续分析或再次另存。\n"
            "4. 新 H5 可稍后重新导入；程序会读取其中的处理链并提示避免重复预处理。"
        )
        self._show_generation_workflow_dialog(
            title="导出预处理 H5 + CSV：处理流程",
            heading="导出预处理 H5 + CSV",
            source=source, steps=steps, after_text=after_text,
            action_text="继续选择保存位置", action_callback=self.export_preprocessed_h5,
            dialog_attribute="_preprocessed_export_workflow_dialog",
        )

    def run_dual_branch_stream_export(self) -> None:
        """Freeze reviewed channels and stream complete LFP/Spike HDF5 branches."""
        source = self._dual_stream_input_source()
        if source is None or not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载原始 HDF5 数据。")
            return
        try:
            branch_settings, apply_lfp_car = self._dual_branch_stream_settings(source.metadata.fs)
        except ValueError as exc:
            QMessageBox.warning(self, "流式处理参数无效", str(exc))
            return
        available = {int(value) for value in source.metadata.channel_ids}
        reviewed = {int(value) for value in self._bad_channel_result_channel_ids}
        qc_ready = self._bad_channel_check_completed and reviewed == available
        skip_qc_for_test = False
        if not qc_ready:
            reason = (
                "尚未完成坏道检查和人工通道确认。" if not self._bad_channel_check_completed else
                "坏道检查结果没有完整覆盖当前数据通道。"
            )
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Icon.Warning)
            dialog.setWindowTitle("坏道判断尚未就绪")
            dialog.setText(reason)
            dialog.setInformativeText(
                "正式处理建议先完成坏道判断。测试流程可选择跳过；跳过后不会根据QC排除任何通道，"
                "输出文件会明确记录“坏道检查已跳过（测试）”。"
            )
            check_button = dialog.addButton("返回完成检查", QMessageBox.ButtonRole.ActionRole)
            skip_button = dialog.addButton("跳过并继续（测试）", QMessageBox.ButtonRole.AcceptRole)
            dialog.addButton(QMessageBox.StandardButton.Cancel)
            dialog.setDefaultButton(check_button)
            dialog.exec()
            if dialog.clickedButton() is not skip_button:
                return
            skip_qc_for_test = True
        export_good_ids = set(available) if skip_qc_for_test else set(self.good_channel_ids)
        if apply_lfp_car and len(export_good_ids) < 2:
            QMessageBox.warning(self, "健康通道不足", "LFP流式CAR至少需要两个确认健康的通道。")
            return
        if getattr(self, "_dual_branch_stream_worker", None) is not None and self._dual_branch_stream_worker.isRunning():
            return
        original_source_path = Path(self.source.metadata.path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Keep every streamed derivative under ``processed`` beside the H5
        # that the operator actually loaded. Provenance may point to an older
        # BIN in a different tree, but it must not redirect new output there.
        output_dir, base_name = _stream_export_target(original_source_path, timestamp)
        qc_snapshot = self._preprocess_qc_snapshot(source)
        if skip_qc_for_test:
            qc_snapshot.update({
                "completed": False,
                "skipped_for_test": True,
                "skip_reason": (
                    "用户选择跳过坏道检查；全部可用通道暂按健康通道参与LFP CAR。" if apply_lfp_car else
                    "用户选择跳过坏道检查；本次流式处理未根据QC排除通道，且LFP CAR未启用。"
                ),
                "assumed_good_channel_ids": sorted(export_good_ids),
            })
        worker = DualBranchStreamWorker(
            source, output_dir, base_name, export_good_ids, qc_snapshot,
            branch_settings=branch_settings, apply_lfp_car=apply_lfp_car,
        )
        worker.progress.connect(self._update_preprocess_progress)
        worker.completed.connect(self._finish_dual_branch_stream_export)
        worker.failed.connect(self._fail_dual_branch_stream_export)
        worker.finished.connect(self._sync_dual_stream_export_enabled)
        self._dual_branch_stream_worker = worker
        self._begin_shared_preprocess_task(worker, "流式生成 LFP + Spike")
        self.dual_stream_export_button.setEnabled(False)
        self.preprocess_progress.setValue(0)
        if skip_qc_for_test:
            self.preprocess_status.setText(
                f"测试模式：已跳过坏道检查，全部 {len(export_good_ids)} 个通道均不作QC排除；"
                f"LFP CAR={'开启' if apply_lfp_car else '关闭'}。"
                f"并行上限 {branch_settings['runtime']['max_workers']}，内存预算 "
                f"{branch_settings['runtime']['memory_budget_gb']:g} GB。"
                f"正在流式生成LFP与Spike完整H5；输出目录：{output_dir}"
            )
        else:
            self.preprocess_status.setText(
                f"已冻结QC：健康 {len(export_good_ids)}、坏道 {len(self.bad_channel_ids)}。"
                f"LFP CAR={'开启' if apply_lfp_car else '关闭'}。"
                f"并行上限 {branch_settings['runtime']['max_workers']}，内存预算 "
                f"{branch_settings['runtime']['memory_budget_gb']:g} GB。"
                f"正在流式生成LFP与Spike完整H5；输出目录：{output_dir}"
            )
        worker.start()

    def _finish_dual_branch_stream_export(self, output: dict) -> None:
        self.preprocess_progress.setValue(100)
        self.lfp_source = LazyH5Source(); self.lfp_source.open(output["lfp"])
        self.spike_source = LazyH5Source(); self.spike_source.open(output["spike"])
        self.active_source = self.lfp_source
        self.preprocess_status.setText(
            f"双分支生成完成。LFP：{Path(output['lfp']).name}；Spike：{Path(output['spike']).name}；"
            f"项目清单：{Path(output['manifest']).name}；完整处理记录：{Path(output['ledger']).name}。"
            "两份数据已按懒加载方式接入对应分析页。"
        )
        self._set_preprocess_data_state(
            f"双分支已加载（懒加载）：通用预览与LFP分析使用 {Path(output['lfp']).name}；"
            f"Spike分析使用 {Path(output['spike']).name}。仅按当前时间窗和通道读取，不会同时载入两份完整数据。"
        )
        self.preprocess_filter_scope.setText("实际滤波范围：LFP与Spike均已按项目参数完成滤波；详情见上方已处理数据摘要及项目清单。")
        self._update_file_banners()
        self.statusBar().showMessage(f"LFP/Spike处理项目已保存：{Path(output['manifest']).parent}")

    def _fail_dual_branch_stream_export(self, error: str) -> None:
        self.preprocess_progress.setValue(0)
        self.preprocess_status.setText(f"LFP/Spike流式生成失败：{error}；未完成文件保留为.partial.h5。")
        QMessageBox.critical(self, "流式生成失败", error)

    def load_dual_branch_project(self) -> None:
        """Open a paired project manifest while keeping both HDF5 branches lazy."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "导入LFP + Spike处理项目", str(Path.cwd()), "Project JSON (*_project.json);;JSON files (*.json)"
        )
        if not filename:
            return
        try:
            manifest_path = Path(filename)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != "sd-dual-branch-project" or int(manifest.get("version", -1)) != 1:
                raise ValueError("不是受支持的LFP/Spike项目清单。")
            if manifest.get("status") != "complete":
                raise ValueError("项目尚未完整生成，不能作为正式数据加载。")
            lfp_path = manifest_path.parent / str(manifest["lfp_file"])
            spike_path = manifest_path.parent / str(manifest["spike_file"])
            lfp = LazyH5Source(); lfp_meta = lfp.open(lfp_path)
            spike = LazyH5Source(); spike_meta = spike.open(spike_path)
            mismatches = []
            if lfp_meta.rows != spike_meta.rows: mismatches.append("采样点数")
            if not np.isclose(lfp_meta.fs, spike_meta.fs): mismatches.append("采样率")
            if lfp_meta.channel_ids != spike_meta.channel_ids: mismatches.append("通道编号/顺序")
            if not np.isclose(lfp_meta.time_offset, spike_meta.time_offset): mismatches.append("时间偏移")
            if mismatches:
                raise ValueError("LFP与Spike文件不一致：" + "、".join(mismatches))
            self.lfp_source, self.spike_source = lfp, spike
            self.active_source = lfp
            lfp_state_restored = self._restore_lfp_analysis_state(lfp_path, lfp_meta)
            spike_state_restored = self._restore_spike_analysis_state(spike_path, spike_meta)
            restored = self._restore_preprocess_qc_from_h5(lfp_path, lfp_meta)
            if restored:
                self._refresh_bad_channel_review_table()
            self._show_loaded_preprocess_summary(lfp_meta, restored)
            self.preprocess_status.setText(
                f"已导入双分支项目：LFP={lfp_path.name}；Spike={spike_path.name}。"
                "当前仅加载元数据，两个分支均按时间窗和通道懒读取。"
            )
            if spike_state_restored:
                self.spike_status.setText(
                    f"已从 {spike_path.name} 恢复 Spike 通道选择、处理状态和参数。"
                )
            self._set_preprocess_data_state(
                f"双分支已加载（懒加载）：通用预览与LFP分析使用 {lfp_path.name}；"
                f"Spike分析使用 {spike_path.name}。仅按当前时间窗和通道读取，不会同时载入两份完整数据。"
            )
            if lfp_state_restored:
                self._set_preprocess_data_state(
                    self.preprocess_data_state.text()
                    + " " + self._format_lfp_analysis_state(self._loaded_lfp_analysis_state)
                )
            else:
                self._set_preprocess_data_state(
                    self.preprocess_data_state.text() + " LFP 状态：该文件尚未保存分析状态。"
                )
            lfp_settings = manifest.get("lfp_processing", {})
            spike_settings = manifest.get("spike_processing", {})
            def setting_number(settings, name, default=0.0):
                value = settings.get(name, default) if isinstance(settings, dict) else default
                try:
                    return float(default if value is None else value)
                except (TypeError, ValueError):
                    return float(default)
            self.preprocess_filter_scope.setText(
                "实际滤波范围："
                f"LFP {setting_number(lfp_settings, 'low'):g}–{setting_number(lfp_settings, 'high'):g} Hz；"
                f"Spike {setting_number(spike_settings, 'low'):g}–{setting_number(spike_settings, 'high'):g} Hz。"
            )
            self._update_file_banners()
        except Exception as exc:
            QMessageBox.critical(self, "项目导入失败", str(exc))

    def _sync_repair_filter_channel(self, channel_id: int) -> None:
        """Bind repair filtering to the currently previewed destination ID."""
        context = getattr(self, "_channel_repair_context", None)
        if not context:
            return
        channel_id = int(channel_id)
        if channel_id not in {int(value) for value in context.get("target_channel_ids", [])}:
            return
        self.preprocess_filter_channels_var.setText(str(channel_id))
        completed = {int(value) for value in context.get("processed_channel_ids", [])}
        state = "已处理，可用新参数再次处理" if channel_id in completed else "尚未处理"
        self.preprocess_filter_channels_var.setToolTip(
            f"当前修复通道 ch{channel_id}（{state}）；执行滤波只处理当前这一通道。"
        )

    def run_preprocess(
        self,
        force_filter_off: bool = False,
        use_existing_remap: bool = False,
        preserve_channel_review: bool = False,
        notch_only: bool = False,
    ) -> None:
        repair_source = (
            getattr(self, "_extracted_channel_preview_source", None)
            if getattr(self, "_channel_repair_context", None) else None
        )
        repair_mode = repair_source is not None and repair_source.loaded
        if not self.source.loaded and not repair_mode:
            QMessageBox.information(self, "尚未加载数据", "请先加载 HDF5 数据文件。")
            return
        remap_path = "" if repair_mode else self.remap_edit.text().strip()
        requested_operations = set()
        if remap_path and not use_existing_remap:
            requested_operations.add("channel_remap")
        planned_filter_mode = "off" if notch_only else str(self.preprocess_filter_mode_var.currentData() or "off")
        planned_notch = (
            False if force_filter_off
            else True if notch_only
            else self.manual_filter_notch_var.isChecked()
        )
        if not force_filter_off and (planned_filter_mode != "off" or planned_notch):
            requested_operations.add("filter")
        if not repair_mode and not self._confirm_repeated_preprocessing(requested_operations, "重映射或滤波"):
            return
        # Do not invalidate a completed QC decision merely because a signal
        # processing worker is about to run. Filtering, CAR and ICA are a
        # separate data pipeline and do not change channel identity. A truly
        # incompatible fresh remap is handled only after it succeeds, so a
        # failed/cancelled filter can never make an exportable QC result vanish.
        if remap_path and not Path(remap_path).is_file():
            QMessageBox.information(self, "映射文件无效", "请选择有效的 H2:H 通道映射 Excel 文件，或清空该项后直接滤波。")
            return
        # Once remapping has completed, channel choices in the preprocessing
        # overview refer to remapped destination IDs. Reuse that exact source
        # for a subsequent filter run instead of interpreting those IDs
        # against the original input and remapping a second time.
        if (
            not force_filter_off
            and not use_existing_remap
            and remap_path
            and self.remapped_source is not None
            and self.remapped_source.loaded
            and self._remapped_detection_signature == self._remap_signature(remap_path)
        ):
            use_existing_remap = True
        filter_baseline_source = None
        if repair_mode:
            processing_source = repair_source
            effective_remap_path = ""
        elif use_existing_remap:
            if self.remapped_source is None or not self.remapped_source.loaded:
                QMessageBox.warning(self, "尚无重映射结果", "请先完成通道重映射后再在重映射数据上滤波。")
                return
            # In the one-click order CAR has already produced the current
            # preprocessing source. Filter that result, while retaining the
            # unreferenced remap separately for later quality checks.
            processing_source = self.preprocessed_source or self.remapped_source
            effective_remap_path = ""
            if not notch_only and processing_source is not self.remapped_source:
                filter_baseline_source = self.remapped_source
        else:
            processing_source = self.source
            effective_remap_path = remap_path
            # With no remap, retain previously processed *other* channels but
            # restore this run's selected channels from the initially loaded
            # source before applying the new filter settings.
            current_processed = getattr(self, "preprocessed_source", None)
            current_meta = getattr(current_processed, "metadata", None)
            source_meta = getattr(self.source, "metadata", None)
            same_channel_matrix = (
                current_processed is not None
                and current_processed is not self.source
                and getattr(current_processed, "loaded", False)
                and current_meta is not None
                and source_meta is not None
                and current_meta.rows == source_meta.rows
                and np.isclose(current_meta.fs, source_meta.fs)
                and tuple(current_meta.channel_ids) == tuple(source_meta.channel_ids)
            )
            if not remap_path and same_channel_matrix:
                processing_source = current_processed
                if not notch_only:
                    filter_baseline_source = self.source
        did_remap = bool(remap_path)
        if force_filter_off and not did_remap:
            QMessageBox.information(self, "缺少映射文件", "“执行通道重映射”需要选择 H2:H 通道映射 Excel 文件。若只需滤波，请使用“滤波并应用到指定通道”。")
            return
        if did_remap and not use_existing_remap:
            self.channel_layout_ids = read_channel_layout(remap_path, self.source.metadata.channels)
        self._pending_remapped_detection_signature = (
            self._remap_signature(remap_path) if did_remap else None
        )
        if self._preprocess_worker is not None and self._preprocess_worker.isRunning():
            return
        mode = "off" if force_filter_off or notch_only else str(self.preprocess_filter_mode_var.currentData() or "off")
        notch_enabled = (
            False if force_filter_off
            else True if notch_only
            else self.manual_filter_notch_var.isChecked()
        )
        try:
            low = float(self.preprocess_filter_low_var.text().strip())
            high = float(self.preprocess_filter_high_var.text().strip())
        except ValueError:
            QMessageBox.warning(self, "参数无效", "预处理滤波截止频率必须是数字。")
            return
        self.preprocess_progress.setValue(0)
        try:
            if repair_mode:
                current_id = self.preprocessed_preview.current_channel_id()
                source_ids = np.asarray(processing_source.metadata.channel_ids, dtype=np.int64)
                selected_columns = np.flatnonzero(source_ids == int(current_id))
                if selected_columns.size != 1:
                    raise ValueError("当前预览通道无法在本次提取数据中唯一定位。")
            else:
                selected_columns = self._selected_preprocess_columns(processing_source)
            hp_order = int(self.preprocess_filter_hp_order_var.text().strip())
            lp_order = int(self.preprocess_filter_lp_order_var.text().strip())
            notch_harmonics = int(self.preprocess_filter_notch_harmonics_var.text().strip() or "1")
            if hp_order < 1 or lp_order < 1 or notch_harmonics < 1:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "参数无效", "滤波阶数和工频谐波数必须是大于 0 的整数；指定通道格式为 1,5,20-26。")
            return
        skipped_notched_ids = set()
        if notch_only:
            source_ids = np.asarray(processing_source.metadata.channel_ids, dtype=np.int64)
            candidate_columns = (
                np.arange(source_ids.size, dtype=np.int64)
                if selected_columns is None else np.asarray(selected_columns, dtype=np.int64)
            )
            already_notched = set(getattr(self, "_preprocess_notched_channel_ids", set()))
            candidate_ids = source_ids[candidate_columns]
            already_notched_mask = np.isin(candidate_ids, sorted(already_notched))
            skipped_notched_ids = set(map(int, candidate_ids[already_notched_mask]))
            selected_columns = candidate_columns[~already_notched_mask]
            if selected_columns.size == 0:
                skipped = candidate_columns.size
                self.preprocess_progress.setValue(100)
                self.preprocess_status.setText(
                    f"一键 50 Hz 工频陷波已跳过：目标 {skipped} 个通道均已完成该陷波。"
                )
                self._advance_one_click_preprocess("preprocess")
                return
        filter_cache_signature = self._filter_cache_signature(
            processing_source, effective_remap_path, mode, low, high, hp_order, lp_order,
            notch_enabled,
            notch_harmonics, selected_columns,
        )
        self._pending_preserve_channel_review = bool(preserve_channel_review)
        self._pending_channel_repair_filter = bool(repair_mode)
        self._pending_detection_remapped_source = (
            self.remapped_source if use_existing_remap else None
        )
        cached_filter = self._last_filter_cache_value
        if self._last_filter_cache_signature == filter_cache_signature and cached_filter is not None:
            cached_remapped, cached_processed, cached_did_remap, cached_info = cached_filter
            self._finish_preprocess(
                cached_remapped, cached_processed, cached_did_remap, dict(cached_info)
            )
            self.preprocess_status.setText("预处理：复用完全相同参数的滤波结果缓存。")
            return
        filtering_enabled = mode != "off" or notch_enabled
        source_ids = np.asarray(processing_source.metadata.channel_ids, dtype=np.int64)
        planned_ids = source_ids if selected_columns is None else source_ids[selected_columns]
        planned_scope = self._preprocess_filter_scope_text(
            planned_ids, processing_source.metadata.channels,
        )
        if not filtering_enabled:
            planned_scope = "本次未启用滤波；数据仅执行重映射或保持当前值。"
        if use_existing_remap:
            operation = "在已重映射数据上滤波"
        elif did_remap:
            operation = "重映射" if not filtering_enabled else "读取 HDF5、重映射并滤波"
        else:
            operation = "按已选通道滤波（未重映射，保持原始通道顺序）" if filtering_enabled else "读取当前数据（未重映射）"
        self.preprocess_status.setText(f"正在后台{operation}…")
        self.preprocess_filter_scope.setText(planned_scope)
        self._set_preprocess_data_state(
            "数据状态：正在后台处理已重映射数据；当前预览仍使用上一次有效数据。"
            if use_existing_remap else (
                "数据状态：正在后台重映射；当前预览仍使用上一次有效数据。"
                if did_remap else f"数据状态：正在后台滤波；未重映射，保持原始通道号与顺序。{planned_scope}"
            )
        )
        self.preprocess_run_button.setEnabled(False)
        worker = PreprocessWorker(
            processing_source,
            effective_remap_path,
            mode,
            low,
            high,
            highpass_order=hp_order,
            lowpass_order=lp_order,
            notch=notch_enabled,
            notch_harmonics=notch_harmonics,
            channels=selected_columns,
            already_remapped=use_existing_remap,
            filter_baseline_source=filter_baseline_source,
            previous_filtered_channel_ids=getattr(
                self, "_preprocess_filtered_channel_ids", set(),
            ),
            previous_lowpass_channel_ids=getattr(
                self, "_preprocess_lowpass_channel_ids", set(),
            ),
            previous_notched_channel_ids=getattr(
                self, "_preprocess_notched_channel_ids", set(),
            ) if not (did_remap and not use_existing_remap) else set(),
            notch_skipped_channel_ids=skipped_notched_ids,
        )
        worker.progress.connect(self._update_preprocess_progress)
        worker.completed.connect(self._finish_preprocess)
        worker.failed.connect(self._fail_preprocess)
        worker.cancelled.connect(self._preprocess_cancelled)
        worker.finished.connect(self._preprocess_worker_finished)
        self._preprocess_worker = worker
        self._preprocess_paused = False
        self._begin_shared_preprocess_task(worker, "通道重映射/滤波")
        self._pending_filter_cache_signature = filter_cache_signature
        worker.start()

    def _update_preprocess_progress(self, value: float, message: str) -> None:
        self.preprocess_progress.setValue(max(0, min(100, int(round(value)))))
        self.preprocess_status.setText(message)

    def _begin_shared_preprocess_task(self, worker, label: str) -> None:
        """Make the shared progress controls operate on this worker."""
        tasks = list(getattr(self, "_shared_preprocess_tasks", []))
        tasks = [item for item in tasks if item[0] is not worker and item[0].isRunning()]
        tasks.append((worker, str(label)))
        self._shared_preprocess_tasks = tasks
        self._active_preprocess_task_worker = worker
        self._active_preprocess_task_label = str(label)
        worker.finished.connect(lambda w=worker: self._end_shared_preprocess_task(w))
        if hasattr(worker, "cancelled"):
            worker.cancelled.connect(
                lambda w=worker, task_label=str(label):
                self._shared_preprocess_task_cancelled(w, task_label)
            )
        self.preprocess_progress.set_running(True, False)

    def _end_shared_preprocess_task(self, worker) -> None:
        tasks = [
            item for item in getattr(self, "_shared_preprocess_tasks", [])
            if item[0] is not worker and item[0].isRunning()
        ]
        self._shared_preprocess_tasks = tasks
        if tasks:
            active, label = tasks[-1]
            self._active_preprocess_task_worker = active
            self._active_preprocess_task_label = label
            self.preprocess_progress.set_running(
                True, bool(getattr(active, "_paused", False)),
            )
        else:
            self._active_preprocess_task_worker = None
            self._active_preprocess_task_label = ""
            self.preprocess_progress.set_running(False)

    def _current_shared_preprocess_task(self):
        worker = getattr(self, "_active_preprocess_task_worker", None)
        if worker is not None and worker.isRunning():
            return worker
        tasks = [
            item for item in getattr(self, "_shared_preprocess_tasks", [])
            if item[0].isRunning()
        ]
        self._shared_preprocess_tasks = tasks
        if not tasks:
            return None
        worker, label = tasks[-1]
        self._active_preprocess_task_worker = worker
        self._active_preprocess_task_label = label
        return worker

    def _shared_preprocess_task_cancelled(self, worker, label: str) -> None:
        if worker is not getattr(self, "_active_preprocess_task_worker", None):
            return
        self._analysis_batch_waiting_for_worker = False
        self._analysis_batch_waiting_for_psd = False
        self._selected_analysis_batch = None
        self._stop_one_click_preprocess()
        self.preprocess_progress.setRange(0, 100)
        self.preprocess_progress.setValue(0)
        suffix = "；已生成的 partial 文件保留" if isinstance(worker, DualBranchStreamWorker) else ""
        self.preprocess_status.setText(f"{label}已安全取消{suffix}。")

    def _toggle_preprocess_pause(self) -> None:
        worker = self._current_shared_preprocess_task()
        if worker is None or not hasattr(worker, "set_paused"):
            return
        paused = not bool(getattr(worker, "_paused", False))
        worker.set_paused(paused)
        self.preprocess_progress.set_running(True, paused)
        label = getattr(self, "_active_preprocess_task_label", "当前任务") or "当前任务"
        self.preprocess_status.setText(
            f"{label}已暂停；点击“继续”恢复运行。" if paused else f"{label}已继续运行。"
        )

    def _cancel_preprocess_run(self) -> None:
        worker = self._current_shared_preprocess_task()
        if worker is None or not hasattr(worker, "cancel"):
            return
        self.preprocess_progress.cancel_button.setEnabled(False)
        self.preprocess_progress.pause_button.setEnabled(False)
        label = getattr(self, "_active_preprocess_task_label", "当前任务") or "当前任务"
        self.preprocess_status.setText(f"正在取消{label}，请等待当前计算块安全结束……")
        worker.cancel()

    def _preprocess_cancelled(self) -> None:
        self._pending_filter_cache_signature = None
        self._pending_channel_repair_filter = False
        self.preprocess_progress.setRange(0, 100)
        self.preprocess_progress.setValue(0)
        self.preprocess_status.setText("预处理已取消；原始数据和上一次有效预览均未改变。")

    def _preprocess_worker_finished(self) -> None:
        self._preprocess_paused = False
        self.preprocess_progress.set_running(False)
        self.preprocess_run_button.setEnabled(True)

    def _finish_preprocess(self, remapped_source: ArraySource, processed_source: ArraySource, did_remap: bool = True, filter_info=None) -> None:
        if bool(getattr(self, "_pending_channel_repair_filter", False)):
            self._pending_channel_repair_filter = False
            self._pending_filter_cache_signature = None
            context = getattr(self, "_channel_repair_context", None) or {}
            channel_files = context.get("channel_files", [])
            if not channel_files:
                self._fail_preprocess("本次修复的临时通道文件已丢失，请重新提取。")
                return
            target = int(self.preprocessed_preview.current_channel_id())
            try:
                for column, channel_path in enumerate(map(Path, channel_files)):
                    with h5py.File(channel_path, "r+") as h5:
                        dataset = h5["rawData512"]
                        chunk_rows = max(1, int(dataset.chunks[0] if dataset.chunks else 100000))
                        for first in range(0, processed_source.metadata.rows, chunk_rows):
                            last = min(processed_source.metadata.rows, first + chunk_rows)
                            dataset[first:last, 0] = processed_source.read(first, last, column).reshape(-1)
                        provenance = getattr(processed_source.metadata, "provenance", None)
                        if provenance:
                            write_h5_provenance(h5, provenance)
                        h5.attrs["data_stage"] = "user_repair_processed_channel"
                refreshed = CustomChannelH5Source(channel_files)
                self._extracted_channel_preview_source = refreshed
                self.preprocessed_preview.set_source(refreshed)
                target_index = list(map(int, context["target_channel_ids"])).index(target)
                self.preprocessed_preview.channel_spin.setValue(target_index + 1)
                self.preprocessed_preview._refresh_now()
                targets = [int(value) for value in context["target_channel_ids"]]
                with h5py.File(channel_files[target_index], "r") as h5:
                    source_id = int(np.asarray(h5["source_physical_channel_id"][()]).squeeze())
                completed = {int(value) for value in context.get("processed_channel_ids", [])}
                completed.add(target)
                context["processed_channel_ids"] = sorted(completed)
                operations = (getattr(processed_source.metadata, "provenance", None) or {}).get("operations", [])
                last_filter = next((dict(item) for item in reversed(operations)
                                    if isinstance(item, dict) and item.get("name") == "filter"), {})
                context.setdefault("processing_by_channel", {})[str(target)] = last_filter
                self.preprocessed_preview.figure.setTitle(
                    f"修复通道处理后预览 — 原始 ch{source_id} → 目标 ch{target}"
                )
                self.preprocess_progress.setValue(100)
                self.preprocess_status.setText(
                    f"目标 ch{target} 处理完成，预览已更新；已处理 {len(completed)}/{len(targets)} 个修复通道。"
                )
                base_state = getattr(self, "_channel_repair_base_state_text", "")
                self._set_preprocess_data_state(
                    f"{base_state} 临时修复状态：已处理通道 {','.join(map(str, sorted(completed)))}；"
                    f"待处理通道 {','.join(map(str, sorted(set(targets) - completed))) or '无'}。"
                )
            except Exception as exc:
                self._fail_preprocess(str(exc))
            return
        preserve_channel_review = bool(getattr(self, "_pending_preserve_channel_review", False))
        had_completed_review = bool(self._bad_channel_check_completed)
        self._pending_preserve_channel_review = False
        detection_remapped_source = getattr(self, "_pending_detection_remapped_source", None)
        self._pending_detection_remapped_source = None
        if self._pending_filter_cache_signature is not None:
            self._last_filter_cache_signature = self._pending_filter_cache_signature
            self._last_filter_cache_value = (
                remapped_source, processed_source, bool(did_remap), dict(filter_info or {})
            )
            self._pending_filter_cache_signature = None
        self.remapped_source = (
            detection_remapped_source
            if detection_remapped_source is not None
            else (remapped_source if did_remap else None)
        )
        review_was_remapped = False
        fresh_remap = bool(did_remap and detection_remapped_source is None)
        if fresh_remap and self.remapped_source is not None:
            review_was_remapped = self._remap_channel_review_ids(
                self.source, self.remap_edit.text().strip(),
            )
        self._remapped_detection_signature = (
            self._pending_remapped_detection_signature if did_remap else None
        )
        self._pending_remapped_detection_signature = None
        self.preprocessed_source = processed_source
        self.active_source = processed_source
        self.preprocess_export_button.setEnabled(True)
        self._preprocess_lowpass_channel_ids = set((filter_info or {}).get("lowpass_channel_ids", set()))
        self._preprocess_notched_channel_ids = set((filter_info or {}).get("notched_channel_ids", set()))
        self._preprocess_lowpass_high_hz = (filter_info or {}).get("lowpass_high_hz")
        self._preprocess_filtered_channel_ids = set((filter_info or {}).get("filtered_channel_ids", set()))
        review_still_compatible = self._completed_qc_matches_source(processed_source)
        if had_completed_review and (
            not review_still_compatible
            or (fresh_remap and not review_was_remapped)
        ):
            # A new mapping changes what each destination channel means. If
            # the previous QC source cannot be translated, keeping its labels
            # would silently attach decisions to the wrong electrodes.
            self._bad_channel_check_completed = False
            if hasattr(self, "apply_car_button"):
                self.apply_car_button.setEnabled(False)
        elif had_completed_review and review_still_compatible:
            # Ordinary filtering (including cached filtering), CAR and ICA do
            # not alter the independently stored QC decision or its CSV data.
            self._bad_channel_check_completed = True
            if hasattr(self, "apply_car_button"):
                self.apply_car_button.setEnabled(len(self.good_channel_ids) >= 2)
        self.preprocessed_preview.set_source(processed_source)
        self._update_preprocess_comparison_preview()
        if review_was_remapped:
            self._refresh_bad_channel_review_table()
            if hasattr(self, "apply_car_button"):
                self.apply_car_button.setEnabled(len(self.good_channel_ids) >= 2)
        filter_enabled = processed_source is not remapped_source
        self.preprocess_progress.setValue(100)
        filtered_ids = self._preprocess_filtered_channel_ids
        scope_text = self._preprocess_filter_scope_text(
            filtered_ids, processed_source.metadata.channels, completed=True,
        ) if filtered_ids else "实际已滤波通道：无（本次滤波关闭）。"
        self.preprocess_filter_scope.setText(scope_text)
        if did_remap:
            status = f"重映射和滤波已完成。{scope_text}" if filter_enabled else "通道重映射已完成（滤波关闭）。"
        else:
            status = f"滤波已完成（未重映射，保留原始通道号与顺序）。{scope_text}" if filter_enabled else "未重映射且滤波关闭，当前数据未改变。"
        if had_completed_review and self._bad_channel_check_completed:
            status += " 此前坏道判定已独立保留，可继续导出 QC CSV。"
        elif had_completed_review:
            status += " 本次新重映射无法对应旧通道，旧坏道判定已失效，请重新检查。"
        skipped_notched = set((filter_info or {}).get("notch_skipped_channel_ids", set()))
        if skipped_notched:
            status += f" 已自动跳过 {len(skipped_notched)} 个已完成 50 Hz 陷波的通道。"
        self.preprocess_status.setText(status)
        self._set_preprocess_data_state(
            self._live_preprocess_state_text(
                processed_source,
                f"数据总通道数 {processed_source.metadata.channels}；当前预览、总览和后续分析均使用处理后数据。",
            )
        )
        self._update_file_banners()
        self._update_one_click_plan_summary()
        self.statusBar().showMessage("处理后数据已成为当前预览与全通道总览的数据源。")
        self._advance_one_click_preprocess("remap")
        self._advance_one_click_preprocess("preprocess")

    def _fail_preprocess(self, message: str) -> None:
        self._pending_channel_repair_filter = False
        self._pending_filter_cache_signature = None
        self._pending_remapped_detection_signature = None
        self._pending_detection_remapped_source = None
        self._pending_preserve_channel_review = False
        self.preprocess_progress.setValue(0)
        self.preprocess_status.setText(f"预处理失败：{message}")
        self._stop_one_click_preprocess()
        QMessageBox.critical(self, "预处理失败", message)

    def _current_audit_data_path(self) -> str:
        for source in (
            getattr(self, "preprocessed_source", None), getattr(self, "active_source", None),
            getattr(self, "source", None),
        ):
            meta = getattr(source, "metadata", None)
            if meta is not None and getattr(meta, "path", None):
                return str(Path(meta.path).resolve())
        return ""

    def _audit_state_snapshot(self) -> dict:
        state = {
            "good_channel_ids": sorted(map(int, getattr(self, "good_channel_ids", set()))),
            "bad_channel_ids": sorted(map(int, getattr(self, "bad_channel_ids", set()))),
            "filtered_channel_ids": sorted(map(int, getattr(self, "_preprocess_filtered_channel_ids", set()))),
            "lfp_selected_channel_ids": sorted(map(int, getattr(self, "lfp_selected_ids", set()))),
            "spike_selected_channel_ids": sorted(map(int, getattr(self, "spike_selected_ids", set()))),
        }
        if hasattr(self, "filter_mode"):
            state["preprocess_parameters"] = {
                "filter_mode": str(self.filter_mode.currentData() or "off"),
                "low_hz": self.preprocess_filter_low_var.text(),
                "high_hz": self.preprocess_filter_high_var.text(),
                "highpass_order": self.preprocess_filter_hp_order_var.text(),
                "lowpass_order": self.preprocess_filter_lp_order_var.text(),
                "notch": self.manual_filter_notch_var.isChecked(),
                "notch_harmonics": self.preprocess_filter_notch_harmonics_var.text(),
                "filter_channels": self.preprocess_filter_channels_var.text(),
            }
        return state

    def _audit_event(self, action: str, summary: str = "", *, data_path: str = "", state=None) -> None:
        store = getattr(self, "operation_audit_store", None)
        if store is None:
            return
        try:
            store.append(
                self.operator_name, self.operation_session_id, action,
                data_path or self._current_audit_data_path(), summary,
                self._audit_state_snapshot() if state is None else state,
            )
            sync_service = getattr(self, "feishu_audit_sync", None)
            if sync_service is not None:
                sync_service.notify()
        except (OSError, sqlite3.Error) as exc:
            self.operation_audit_error = str(exc)

    def show_feishu_audit_status(self) -> None:
        config = self.feishu_audit_config
        store = getattr(self, "operation_audit_store", None)
        counts = {}
        if store is not None:
            try:
                counts = store.sync_counts()
            except (OSError, sqlite3.Error):
                counts = {}
        if config.is_configured:
            heading = "飞书同步：已启用（SQLite 本地先写，后台自动同步）"
        else:
            heading = "飞书同步：尚未启用"
        count_text = "，".join(f"{key}={value}" for key, value in sorted(counts.items())) or "暂无"
        missing = "、".join(config.missing_environment_variables) or "无"
        QMessageBox.information(
            self,
            "飞书多维表格同步",
            f"{heading}\n\n"
            f"本地数据库：{getattr(store, 'path', '不可用')}\n"
            f"同步队列：{count_text}\n"
            f"缺少配置：{missing}\n\n"
            "飞书数据表需建立以下文本字段：\n"
            + "、".join(FEISHU_FIELD_NAMES)
            + "\n\n通过系统环境变量配置 App ID/App Secret，程序不会把密钥写入 SQLite。",
        )

    def show_operation_audit_table(self) -> None:
        store = getattr(self, "operation_audit_store", None)
        if store is None:
            QMessageBox.warning(self, "操作记录不可用", self.operation_audit_error or "操作记录数据库无法打开。")
            return
        try:
            rows = store.latest(2000)
        except (OSError, sqlite3.Error) as exc:
            QMessageBox.critical(self, "读取操作记录失败", str(exc))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"多人操作记录 — 当前用户：{self.operator_name}")
        dialog.resize(1350, 720)
        layout = QVBoxLayout(dialog)
        location = QLabel(f"共享日志：{store.path}　共显示 {len(rows)} 次 GUI 会话（每次启动一行）")
        location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(location)
        table = QTableWidget(len(rows), 9, dialog)
        table.setHorizontalHeaderLabels(
            ["最后更新时间", "操作者", "最后动作", "最后数据文件", "本次会话操作时间线",
             "完整会话 JSON", "飞书同步", "同步时间", "同步错误"]
        )
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                text = str(value)
                item = QTableWidgetItem(text if column not in (5, 8) or len(text) <= 180 else text[:180] + "…")
                if column in (5, 8):
                    item.setToolTip(text)
                table.setItem(row_index, column, item)
        table.setColumnWidth(0, 190)
        table.setColumnWidth(1, 110)
        table.setColumnWidth(2, 150)
        table.setColumnWidth(3, 330)
        table.setColumnWidth(4, 360)
        table.setColumnWidth(6, 110)
        table.setColumnWidth(7, 190)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table, 1)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        button_row = QHBoxLayout(); button_row.addStretch(1); button_row.addWidget(close_button)
        layout.addLayout(button_row)
        dialog.show()
        self._operation_audit_dialog = dialog

    def _set_preprocess_data_state(self, text: str) -> None:
        self._preprocess_data_state_text = text
        if hasattr(self, "preprocess_data_state"):
            self.preprocess_data_state.setText(text)
        self._update_one_click_plan_summary()
        self._audit_event("preprocess_data_state", text)

    def choose_export_path(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前数据为 HDF5",
            self.export_path_edit.text() or str(Path.cwd() / "processed_data.h5"),
            "HDF5 files (*.h5 *.hdf5)",
        )
        if filename:
            if not Path(filename).suffix:
                filename += ".h5"
            self.export_path_edit.setText(filename)

    def _original_bin_path(self) -> Path | None:
        """Return the original BIN recorded in the source provenance, if any."""
        source_meta = getattr(getattr(self, "source", None), "metadata", None)
        provenance = getattr(source_meta, "provenance", None) if source_meta is not None else None
        if isinstance(provenance, dict):
            for entry in reversed(provenance.get("lineage", [])):
                if not isinstance(entry, dict) or str(entry.get("kind", "")).lower() != "bin":
                    continue
                raw_path = str(entry.get("path", "")).strip()
                if raw_path:
                    return Path(raw_path)
        # Compatibility with converted files written before provenance was
        # introduced, where source_bin was stored as a root HDF5 attribute.
        source_path = Path(source_meta.path) if source_meta is not None else None
        try:
            if source_path is not None and h5py.is_hdf5(source_path):
                with h5py.File(source_path, "r") as h5:
                    raw_path = h5.attrs.get("source_bin", "")
                if isinstance(raw_path, bytes):
                    raw_path = raw_path.decode("utf-8", errors="replace")
                if str(raw_path).strip():
                    return Path(str(raw_path).strip())
        except OSError:
            pass
        return None

    def _preprocessed_export_suggestion(self) -> Path:
        """Place a timestamped preprocessing product beside the original BIN."""
        source_meta = getattr(getattr(self, "source", None), "metadata", None)
        bin_path = self._original_bin_path()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if bin_path is not None:
            return bin_path.parent / "processed" / f"{bin_path.stem}_{timestamp}_processed.h5"
        source_path = Path(source_meta.path) if source_meta is not None else Path()
        if str(source_path).startswith("<"):
            return Path.cwd() / "processed" / f"processed_data_{timestamp}_processed.h5"
        stem = source_path.stem
        if stem.lower().endswith("_raw"):
            stem = stem[:-4]
        return source_path.parent / "processed" / f"{stem}_{timestamp}_processed.h5"

    def export_preprocessed_h5(self) -> None:
        """Export the current preprocessing result and its paired ledger CSV."""
        source = self.preprocessed_source
        if source is None or not source.loaded:
            QMessageBox.information(self, "尚无预处理结果", "请先完成滤波、重参考或 ICA 后再导出。")
            return
        suggested = self._preprocessed_export_suggestion()
        suggested.parent.mkdir(parents=True, exist_ok=True)
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "导出预处理数据和处理记录",
            str(suggested),
            "HDF5 files (*.h5 *.hdf5)",
        )
        if not filename:
            return
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".h5")
        try:
            record_path = self._write_preprocessed_h5(path, source)
            self.preprocess_status.setText(
                f"已导出预处理数据：{path.name}；处理记录：{record_path.name}"
            )
            self.statusBar().showMessage(f"已导出预处理数据和处理记录：{path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "预处理数据导出失败", str(exc))

    def open_channel_replacement(self) -> None:
        """Collect and launch a safe raw-to-preprocessed channel repair."""
        raw_default = ""
        if getattr(self.source, "metadata", None) is not None:
            raw_default = str(self.source.metadata.path)
        raw_name, _ = QFileDialog.getOpenFileName(
            self, "选择原始 H5", raw_default or str(Path.cwd()), "HDF5 files (*.h5 *.hdf5)"
        )
        if not raw_name:
            return
        processed_default = ""
        if getattr(self.preprocessed_source, "metadata", None) is not None:
            processed_default = str(self.preprocessed_source.metadata.path)
        processed_name, _ = QFileDialog.getOpenFileName(
            self, "选择需要修正的预处理 H5", processed_default or str(Path(raw_name).parent),
            "HDF5 files (*.h5 *.hdf5)",
        )
        if not processed_name:
            return
        channel_text, accepted = QInputDialog.getText(
            self, "修复/平替指定通道", "输入预处理文件中的目标通道号（例如 4,27,100-103）："
        )
        if not accepted:
            return
        try:
            target_ids = set()
            for part in channel_text.replace("，", ",").replace(";", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    first, last = (int(value.strip()) for value in part.split("-", 1))
                    target_ids.update(range(min(first, last), max(first, last) + 1))
                else:
                    target_ids.add(int(part))
            if not target_ids:
                raise ValueError("没有输入有效通道号。")
            with h5py.File(processed_name, "r") as h5:
                provenance = read_h5_provenance(h5) or {}
            operations = provenance.get("operations", []) if isinstance(provenance, dict) else []
            remap_operation = next((
                item for item in reversed(operations)
                if isinstance(item, dict) and item.get("name") == "channel_remap"
            ), None)
            mapping_name = None
            if remap_operation is not None:
                embedded_sources = remap_operation.get("source_channel_ids", [])
                embedded_destinations = remap_operation.get("destination_for_source_channel_ids", [])
                if not embedded_sources or not embedded_destinations:
                    suggested = str(remap_operation.get("mapping_file", ""))
                    mapping_name, _ = QFileDialog.getOpenFileName(
                        self, "该文件已重映射：选择原处理使用的映射 Excel",
                        suggested or str(Path(processed_name).parent), "Excel files (*.xlsx *.xls)"
                    )
                    if not mapping_name:
                        raise ValueError("已取消：重映射文件缺少内嵌映射，不能安全反查原始通道。")
            output_default = Path(processed_name).with_name(Path(processed_name).stem + "_repaired.h5")
            output_name, _ = QFileDialog.getSaveFileName(
                self, "另存修正版预处理 H5", str(output_default), "HDF5 files (*.h5 *.hdf5)"
            )
            if not output_name:
                return
            if not Path(output_name).suffix:
                output_name += ".h5"
            kwargs = {
                "raw_path": raw_name, "processed_path": processed_name,
                "output_path": output_name, "target_channel_ids": sorted(target_ids),
                "mapping_path": mapping_name,
                "mode": str(self.preprocess_filter_mode_var.currentData() or "off"),
                "low": float(self.preprocess_filter_low_var.text()),
                "high": float(self.preprocess_filter_high_var.text()),
                "highpass_order": int(self.preprocess_filter_hp_order_var.text()),
                "lowpass_order": int(self.preprocess_filter_lp_order_var.text()),
                "notch": self.manual_filter_notch_var.isChecked(),
                "notch_harmonics": int(self.preprocess_filter_notch_harmonics_var.text()),
            }
        except Exception as exc:
            QMessageBox.warning(self, "无法开始通道平替", str(exc))
            return
        self.channel_replacement_button.setEnabled(False)
        self.preprocess_progress.setRange(0, 0)
        self.preprocess_status.setText("正在从原始 H5 分块重算并写入修正版文件……")
        worker = ChannelReplacementWorker(kwargs)
        worker.completed.connect(self._finish_channel_replacement)
        worker.failed.connect(self._fail_channel_replacement)
        worker.finished.connect(lambda: self.channel_replacement_button.setEnabled(True))
        self._channel_replacement_worker = worker
        worker.progress.connect(self._update_preprocess_progress)
        self._begin_shared_preprocess_task(worker, "通道替换")
        worker.start()

    @staticmethod
    def _parse_channel_id_text(text: str) -> list[int]:
        ids = set()
        for part in str(text).replace("，", ",").replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                first, last = (int(value.strip()) for value in part.split("-", 1))
                ids.update(range(min(first, last), max(first, last) + 1))
            else:
                ids.add(int(part))
        if not ids:
            raise ValueError("没有输入有效通道号。")
        return sorted(ids)

    def extract_original_channels(self) -> None:
        """Export untouched raw channels; the application does no filtering."""
        current_source = self.preprocessed_source or self.active_source
        current_meta = getattr(current_source, "metadata", None)
        if current_meta is None:
            QMessageBox.information(self, "尚未加载预处理 H5", "请先加载需要修复的预处理 H5。")
            return
        processed_path = Path(current_meta.path)
        if not processed_path.is_file() or not h5py.is_hdf5(processed_path):
            QMessageBox.information(
                self, "当前结果尚未保存",
                "当前预处理结果还不是磁盘 H5，请先导出并重新加载该预处理 H5。",
            )
            return
        if not self._is_preprocessed_h5(processed_path, current_meta):
            QMessageBox.information(self, "当前文件不是预处理 H5", "请先加载需要修复的预处理 H5。")
            return
        raw_name, _ = QFileDialog.getOpenFileName(
            self, "选择最初的原始 H5", str(Path.cwd()), "HDF5 files (*.h5 *.hdf5)"
        )
        if not raw_name: return
        processed_name = str(processed_path)
        text, accepted = QInputDialog.getText(
            self, "提取原始通道", "输入预处理文件中显示的目标通道号（例如 20 或 20,26-28）："
        )
        if not accepted: return
        try:
            targets = self._parse_channel_id_text(text)
            mapping_name = None
            with h5py.File(processed_name, "r") as h5:
                provenance = read_h5_provenance(h5) or {}
            operations = provenance.get("operations", []) if isinstance(provenance, dict) else []
            remap = next((item for item in reversed(operations)
                          if isinstance(item, dict) and item.get("name") == "channel_remap"), None)
            if remap is not None and not (
                remap.get("source_channel_ids") and remap.get("destination_for_source_channel_ids")
            ):
                mapping_name, _ = QFileDialog.getOpenFileName(
                    self, "旧文件未内嵌映射：选择原映射 Excel",
                    str(remap.get("mapping_file", "")) or str(Path(processed_name).parent),
                    "Excel files (*.xlsx *.xls)",
                )
                if not mapping_name: return
            # Extracted channels are disposable repair-session files.  Keep
            # them beside the currently loaded processed H5 so the user does
            # not need to choose or manage a separate output directory.
            directory = str(Path(processed_name).parent)
            paths = extract_original_channel_files(
                raw_name, processed_name, targets, directory, mapping_path=mapping_name
            )
            mappings = []
            for path in paths:
                with h5py.File(path, "r") as extracted:
                    target_id = int(np.asarray(extracted["target_channel_id"][()]).squeeze())
                    source_id = int(np.asarray(extracted["source_physical_channel_id"][()]).squeeze())
                mappings.append(f"目标 ch{target_id} ← 原始 ch{source_id}")
            if not getattr(self, "_channel_repair_context", None):
                self._channel_repair_base_state_text = self.preprocess_data_state.text()
            self._channel_repair_context = {
                "processed_path": str(Path(processed_name).resolve()),
                "channel_files": [str(Path(path).resolve()) for path in paths],
                "target_channel_ids": list(targets),
                "processed_channel_ids": [],
                "processing_by_channel": {},
            }
            if not hasattr(self, "_channel_repair_base_filter_channels_text"):
                self._channel_repair_base_filter_channels_text = self.preprocess_filter_channels_var.text()
            self.preprocess_filter_channels_var.setText(str(targets[0]))
            self.preprocess_filter_channels_var.setReadOnly(True)
            self.preprocess_filter_channels_var.setToolTip(
                f"当前修复通道 ch{targets[0]}；切换预览通道后会自动更新，执行滤波只处理当前通道。"
            )
            self.merge_repaired_channel_button.setEnabled(True)
            base_state = getattr(
                self, "_channel_repair_base_state_text", self.preprocess_data_state.text()
            )
            self._set_preprocess_data_state(
                f"{base_state} 临时修复状态：已提取 {'；'.join(mappings)}，等待你处理后并入。"
            )
            # Immediately show the actual untouched source signal, rather
            # than leaving the preview on the same-numbered processed column.
            preview_source = CustomChannelH5Source(paths)
            self._extracted_channel_preview_source = preview_source
            self.preprocessed_preview.set_source(preview_source)
            self.preprocessed_preview._refresh_now()
            with h5py.File(paths[0], "r") as extracted:
                shown_target = int(np.asarray(extracted["target_channel_id"][()]).squeeze())
                shown_source = int(np.asarray(extracted["source_physical_channel_id"][()]).squeeze())
            self.preprocessed_preview.figure.setTitle(
                f"原始通道预览 — 原始 ch{shown_source} → 目标 ch{shown_target}"
            )
            self.preprocessed_preview.status_label.setText(
                f"当前显示未经处理的原始物理 ch{shown_source}（将并入目标 ch{shown_target}）。"
            )
            self.preprocess_status.setText(
                f"已提取 {len(paths)} 个未经处理的原始通道。请处理并保存原文件，完成后直接点击“并入处理好通道”。"
            )
        except Exception as exc:
            QMessageBox.critical(self, "提取原始通道失败", str(exc))

    def merge_repaired_channel(self) -> None:
        """Merge the files remembered by the preceding extraction step."""
        context = getattr(self, "_channel_repair_context", None)
        if not context:
            QMessageBox.information(self, "尚未提取通道", "请先点击“提取原始通道”完成本次修复选择。")
            return
        processed_path = Path(context["processed_path"])
        channel_files = [Path(path) for path in context["channel_files"]]
        target_ids = [int(value) for value in context["target_channel_ids"]]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = processed_path.with_name(
            f"{processed_path.stem}_replace_channel{processed_path.suffix}"
        )
        sequence = 2
        while output_path.exists():
            output_path = processed_path.with_name(
                f"{processed_path.stem}_replace_channel_{sequence}{processed_path.suffix}"
            )
            sequence += 1
        temporary_paths = []
        try:
            if not channel_files or len(channel_files) != len(target_ids):
                raise ValueError("本次提取会话的通道文件与目标通道记录不完整，请重新提取。")
            if not processed_path.is_file():
                raise FileNotFoundError(f"先前选择的预处理 H5 不存在：{processed_path}")
            missing = [str(path) for path in channel_files if not path.is_file()]
            if missing:
                raise FileNotFoundError("提取的通道文件不存在：" + ", ".join(missing))
            current_input = processed_path
            for index, (channel_file, target) in enumerate(zip(channel_files, target_ids), start=1):
                temporary = output_path if index == len(channel_files) else processed_path.with_name(
                    f".{processed_path.stem}.channel_merge_{timestamp}_{index}.tmp{processed_path.suffix}"
                )
                temporary_paths.append(temporary)
                merge_repaired_channel_file(
                    current_input, channel_file, temporary, target_channel_id=target
                )
                if current_input != processed_path and current_input.exists():
                    current_input.unlink()
                current_input = temporary
            temporary_paths = []
            self._channel_repair_context = None
            self.merge_repaired_channel_button.setEnabled(False)
            previous_filter_text = getattr(self, "_channel_repair_base_filter_channels_text", "")
            self.preprocess_filter_channels_var.setReadOnly(False)
            self.preprocess_filter_channels_var.setText(previous_filter_text)
            self.preprocess_filter_channels_var.setToolTip("")
            self.__dict__.pop("_channel_repair_base_filter_channels_text", None)
            corrected_source = LazyH5Source()
            corrected_meta = corrected_source.open(output_path)
            self.source = corrected_source
            self.active_source = corrected_source
            self.preprocessed_source = corrected_source
            self.remapped_source = None
            restored_qc = self._restore_preprocess_qc_from_h5(output_path, corrected_meta)
            self._show_loaded_preprocess_summary(corrected_meta, restored_qc)
            self._restore_preprocess_processing_state(output_path, corrected_meta)
            self.preprocessed_preview.set_source(corrected_source)
            self._update_preprocess_comparison_preview()
            target_column = int(np.flatnonzero(
                np.asarray(corrected_meta.channel_ids, dtype=np.int64) == target_ids[0]
            )[0])
            self.preprocessed_preview.channel_spin.setValue(target_column + 1)
            self.preprocessed_preview._refresh_now()
            # The preview previously pointed at these temporary single-channel
            # files.  Switch it to the merged H5 and complete one refresh before
            # deleting them, otherwise an already queued refresh can reopen a
            # path that has just been removed.
            removed_channel_files = []
            for channel_file in channel_files:
                if channel_file.exists():
                    channel_file.unlink()
                    removed_channel_files.append(channel_file.name)
            self.__dict__.pop("_channel_repair_base_state_text", None)
            self.__dict__.pop("_extracted_channel_preview_source", None)
            self.preprocess_status.setText(
                f"已并入通道 {','.join(map(str, target_ids))}；原文件未修改，新文件：{output_path.name}；"
                f"已删除 {len(removed_channel_files)} 个临时单通道 H5。"
            )
        except Exception as exc:
            for path in temporary_paths:
                if path.exists():
                    path.unlink()
            QMessageBox.critical(self, "并入处理好通道失败", str(exc))

    def _finish_channel_replacement(self, result: dict) -> None:
        self.preprocess_progress.setRange(0, 100)
        self.preprocess_progress.setValue(100)
        mapping_note = "已按重映射关系反查原始通道" if result.get("remapped") else "通道未重映射"
        self.preprocess_status.setText(
            f"通道平替完成（{mapping_note}）：{Path(result['output']).name}"
        )

    def _fail_channel_replacement(self, message: str) -> None:
        self.preprocess_progress.setRange(0, 100)
        self.preprocess_progress.setValue(0)
        self.preprocess_status.setText(f"通道平替失败：{message}")
        QMessageBox.critical(self, "通道平替失败", message)

    @staticmethod
    def _preprocess_qc_json_default(value):
        """Convert the small QC snapshot into plain JSON values."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Not JSON serializable: {type(value).__name__}")

    def _preprocess_qc_snapshot(self, source) -> dict:
        """Return the reloadable, user-facing outcome of preprocessing QC."""
        meta = source.metadata
        available_ids = {int(channel) for channel in meta.channel_ids}
        completed = bool(self._bad_channel_check_completed)
        evaluated_ids = tuple(sorted(
            int(channel) for channel in self._bad_channel_result_channel_ids
            if completed and int(channel) in available_ids
        ))
        evaluated_set = set(evaluated_ids)
        automatic_bad = {
            str(int(channel)): str(reason)
            for channel, reason in self.bad_channel_auto_reasons.items()
            if int(channel) in evaluated_set
        }
        manual_overrides = {
            str(int(channel)): str(result)
            for channel, result in self.manual_channel_overrides.items()
            if int(channel) in evaluated_set and result in {"good", "bad"}
        }
        rows = []
        for channel in evaluated_ids:
            manual = manual_overrides.get(str(channel), "")
            automatic_status = (
                "bad" if str(channel) in automatic_bad else "healthy"
            )
            final_status = (
                "healthy" if manual == "good" else
                "bad" if manual == "bad" else
                "bad" if channel in self.bad_channel_ids else "healthy"
            )
            automatic_reason = automatic_bad.get(str(channel), "")
            review_detail = (
                ("人工健康覆盖；" if manual == "good" else "人工坏道覆盖；" if manual == "bad" else "")
                + automatic_reason
            )
            rows.append({
                "channel": channel,
                "final_status": final_status,
                "automatic_status": automatic_status,
                "automatic_reason": automatic_reason,
                "brief_description": self._brief_channel_review_description(review_detail, final_status),
                "manual_override": manual,
                "manual_override_applied": bool(manual),
            })

        def detail_rows(name: str) -> list[dict]:
            return [
                dict(row) for row in getattr(self, name, [])
                if isinstance(row, dict) and int(row.get("channel", -1)) in evaluated_set
            ]

        worker = getattr(self, "_bad_channel_worker", None)
        return {
            "schema": "sd-preprocess-qc",
            "version": 1,
            "completed": completed,
            "source_channel_ids": [int(channel) for channel in meta.channel_ids],
            "evaluated_channel_ids": list(evaluated_ids),
            "automatic_bad_reasons": automatic_bad,
            "manual_overrides": manual_overrides,
            "channel_rows": rows,
            "settings": dict(getattr(worker, "settings", {})),
            "metric_settings": {},
            "detail_rows": {
                "fast_artifact": detail_rows("bad_channel_fast_artifact_rows"),
                "high_frequency_noise": detail_rows("bad_channel_high_frequency_noise_rows"),
            },
        }

    def _restore_preprocess_qc_from_processing_csv(self, path: Path, meta) -> bool:
        """Restore two-class QC from the exact CSV row set for this H5."""
        import csv
        hint = ""
        try:
            if h5py.is_hdf5(path):
                with h5py.File(path, "r") as h5:
                    hint = h5.attrs.get("processing_record_csv", "")
                if isinstance(hint, bytes):
                    hint = hint.decode("utf-8", errors="replace")
        except OSError:
            pass
        candidates = []
        if str(hint).strip():
            candidates.append(path.parent / str(hint).strip())
        candidates.append(path.with_suffix(".csv"))
        stem = path.stem
        for suffix in ("_lfp_processed", "_spike_processed", "_processed"):
            if stem.lower().endswith(suffix):
                candidates.append(path.with_name(stem[:-len(suffix)] + "_processing.csv"))
        candidates.extend(path.parent.glob("*_processing.csv"))
        available = {int(channel) for channel in meta.channel_ids}
        seen = set()
        for candidate in candidates:
            candidate = Path(candidate)
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            try:
                with open(candidate, newline="", encoding="utf-8-sig") as handle:
                    rows = list(csv.DictReader(handle))
            except (OSError, UnicodeError, csv.Error):
                continue
            matched = [
                row for row in rows
                if Path(str(row.get("output_file", "")).strip()).name.lower() == path.name.lower()
            ]
            by_channel = {}
            for row in matched:
                try:
                    channel = int(float(str(row.get("channel", "")).strip()))
                except (TypeError, ValueError):
                    continue
                if channel in available:
                    by_channel[channel] = row
            if not by_channel:
                continue
            automatic_bad = {}
            manual = {}
            for channel, row in by_channel.items():
                automatic_status = str(row.get("automatic_status", "")).strip().lower()
                if automatic_status in {"bad", "坏道"}:
                    automatic_bad[channel] = str(row.get("automatic_reason", "") or "历史CSV自动坏道")
                applied = str(row.get("manual_override_applied", "")).strip().lower() in {
                    "1", "true", "yes", "y", "是",
                }
                normalized = {"healthy": "good", "健康": "good", "坏道": "bad"}.get(
                    str(row.get("manual_override", "")).strip().lower(),
                    str(row.get("manual_override", "")).strip().lower(),
                )
                if applied and normalized in {"good", "bad"}:
                    manual[channel] = normalized
            self._bad_channel_result_channel_ids = tuple(sorted(by_channel))
            self.bad_channel_auto_reasons = automatic_bad
            self.bad_channel_candidate_reasons = {}
            self.manual_channel_overrides = manual
            self.bad_channel_fast_artifact_rows = []
            self.bad_channel_high_frequency_noise_rows = []
            self._bad_channel_check_completed = True
            self._bad_channel_fast_only = False
            self._restored_qc_record_source = str(candidate)
            self._restored_qc_settings = {}
            self._apply_channel_review_overrides()
            if hasattr(self, "apply_car_button"):
                self.apply_car_button.setEnabled(len(self.good_channel_ids) >= 2)
            return True
        return False

    def _restore_preprocess_qc_from_h5(self, path: Path, meta) -> bool:
        """Restore a completed QC decision from an exported preprocessing H5.

        The H5 attribute is intentionally the source of truth.  The adjacent
        CSV is a readable ledger and is not required for a reliable reload.
        """
        self._loaded_qc_skipped_for_test = False
        try:
            if not h5py.is_hdf5(path):
                return self._restore_preprocess_qc_from_processing_csv(path, meta)
            with h5py.File(path, "r") as h5:
                raw_snapshot = h5.attrs.get("preprocess_qc_json")
            if raw_snapshot is None:
                return self._restore_preprocess_qc_from_processing_csv(path, meta)
            if isinstance(raw_snapshot, bytes):
                raw_snapshot = raw_snapshot.decode("utf-8", errors="replace")
            snapshot = json.loads(str(raw_snapshot))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return self._restore_preprocess_qc_from_processing_csv(path, meta)
        if not isinstance(snapshot, dict):
            return self._restore_preprocess_qc_from_processing_csv(path, meta)
        if snapshot.get("schema") != "sd-preprocess-qc" or snapshot.get("version") != 1:
            return self._restore_preprocess_qc_from_processing_csv(path, meta)
        self._loaded_qc_skipped_for_test = bool(snapshot.get("skipped_for_test", False))
        if snapshot.get("completed") is not True:
            self._sync_dual_stream_export_enabled()
            return False
        self._sync_dual_stream_export_enabled()

        available_ids = {int(channel) for channel in meta.channel_ids}

        def valid_id(value) -> int | None:
            try:
                channel = int(value)
            except (TypeError, ValueError):
                return None
            return channel if channel in available_ids else None

        raw_evaluated = snapshot.get("evaluated_channel_ids", [])
        if not isinstance(raw_evaluated, list):
            return False
        evaluated_ids = tuple(sorted({
            channel for value in raw_evaluated
            if (channel := valid_id(value)) is not None
        }))
        evaluated_set = set(evaluated_ids)

        def restore_reasons(name: str) -> dict[int, str]:
            values = snapshot.get(name, {})
            if not isinstance(values, dict):
                return {}
            restored = {}
            for raw_channel, reason in values.items():
                channel = valid_id(raw_channel)
                if channel in evaluated_set and isinstance(reason, (str, int, float)):
                    restored[channel] = str(reason)
            return restored

        automatic_bad = restore_reasons("automatic_bad_reasons")
        raw_overrides = snapshot.get("manual_overrides", {})
        manual_overrides = {}
        if isinstance(raw_overrides, dict):
            for raw_channel, value in raw_overrides.items():
                channel = valid_id(raw_channel)
                normalized_value = {"healthy": "good"}.get(str(value), str(value))
                if channel in evaluated_set and normalized_value in {"good", "bad"}:
                    manual_overrides[channel] = normalized_value

        detail_rows = snapshot.get("detail_rows", {})
        if not isinstance(detail_rows, dict):
            detail_rows = {}

        def restore_detail(name: str) -> list[dict]:
            rows = detail_rows.get(name, [])
            if not isinstance(rows, list):
                return []
            return [
                dict(row) for row in rows
                if isinstance(row, dict) and valid_id(row.get("channel")) in evaluated_set
            ]

        self._bad_channel_result_channel_ids = evaluated_ids
        self.bad_channel_auto_reasons = automatic_bad
        self.bad_channel_candidate_reasons = {}
        self.manual_channel_overrides = manual_overrides
        self.bad_channel_fast_artifact_rows = restore_detail("fast_artifact")
        self.bad_channel_high_frequency_noise_rows = restore_detail("high_frequency_noise")
        self._bad_channel_check_completed = True
        self._bad_channel_fast_only = False
        self._restored_qc_record_source = "H5内嵌QC快照"
        self._restored_qc_settings = dict(snapshot.get("settings", {})) if isinstance(snapshot.get("settings"), dict) else {}
        self._apply_channel_review_overrides()
        if hasattr(self, "apply_car_button"):
            self.apply_car_button.setEnabled(len(self.good_channel_ids) >= 2)
        return True

    @staticmethod
    def _preprocess_operation_names(meta) -> set[str]:
        provenance = getattr(meta, "provenance", None) or {}
        return {
            str(item.get("name", "")).strip().lower()
            for item in provenance.get("operations", [])
            if isinstance(item, dict) and item.get("name")
        }

    @staticmethod
    def _provenance_integer_ids(values) -> list[int]:
        """Decode both legacy JSON arrays and compact ``1-10,15`` strings."""
        if values is None:
            return []
        if not isinstance(values, str):
            return sorted({int(value) for value in values})
        output = set()
        for part in values.replace("；", ",").replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                first, last = (int(value.strip()) for value in part.split("-", 1))
                output.update(range(min(first, last), max(first, last) + 1))
            else:
                output.add(int(part))
        return sorted(output)

    @classmethod
    def _format_channel_scope(cls, channel_ids) -> str:
        ids = cls._provenance_integer_ids(channel_ids)
        if not ids:
            return "未记录通道"
        if ids == list(range(ids[0], ids[-1] + 1)):
            return f"ch{ids[0]}–{ids[-1]}" if len(ids) > 1 else f"ch{ids[0]}"
        preview = "、".join(f"ch{value}" for value in ids[:8])
        return preview + (f" 等 {len(ids)} 个通道" if len(ids) > 8 else "")

    def _describe_preprocess_operation(self, operation: dict) -> str | None:
        name = str(operation.get("name", "")).strip().lower()
        if name == "channel_remap":
            mapping = Path(str(operation.get("mapping_file", ""))).name
            return "通道重映射" + (f"（映射表：{mapping}）" if mapping else "")
        if name == "filter_baseline_restore":
            scope = self._format_channel_scope(operation.get("channel_ids", []))
            return f"{scope} 恢复处理前数据后重新滤波"
        if name == "filter":
            mode = str(operation.get("mode", "off")).lower()
            scope = self._format_channel_scope(operation.get("channel_ids", []))
            # Normal preprocessing provenance uses low_hz/high_hz while the
            # streaming branch settings historically used low/high.  Accept
            # both, including explicit JSON null values from older files.
            low_value = operation.get("low_hz")
            high_value = operation.get("high_hz")
            low = float(operation.get("low", 0) if low_value is None else low_value)
            high = float(operation.get("high", 0) if high_value is None else high_value)
            hp_order = operation.get("highpass_order")
            lp_order = operation.get("lowpass_order")
            hp_order = int(operation.get("hp_order", 3) if hp_order is None else hp_order)
            lp_order = int(operation.get("lp_order", 5) if lp_order is None else lp_order)
            if mode == "bandpass":
                text = (
                    f"{scope} 带通滤波 {low:g}–{high:g} Hz"
                    f"（高通 {hp_order} 阶，低通 {lp_order} 阶）"
                )
            elif mode == "highpass":
                text = f"{scope} 高通滤波 {low:g} Hz（{hp_order} 阶）"
            elif mode == "lowpass":
                text = f"{scope} 低通滤波 {high:g} Hz（{lp_order} 阶）"
            else:
                text = f"{scope} 未启用高通/低通/带通"
            notch_enabled = operation.get("notch_enabled")
            notch_enabled = operation.get("notch", False) if notch_enabled is None else notch_enabled
            if notch_enabled:
                notch_frequency = operation.get("notch_frequency_hz")
                notch_frequency = operation.get("notch_frequency", 50) if notch_frequency is None else notch_frequency
                text += (
                    f"；{float(notch_frequency):g} Hz 陷波"
                    f"（Q={float(operation.get('notch_q', 30)):g}，"
                    f"谐波数={int(operation.get('notch_harmonics', 1))}）"
                )
            return text
        if name == "reference":
            count = len(self._provenance_integer_ids(operation.get("good_channel_ids", [])))
            return f"Leave-one-out median CAR（{count} 个健康通道参与参考）"
        if name == "ica":
            excluded = operation.get("exclude", [])
            suffix = "、".join(f"IC{int(value)}" for value in excluded) or "无"
            scope = self._format_channel_scope(operation.get("channel_ids", []))
            return (
                f"{scope} ICA（{operation.get('components', '—')} 个成分，移除 {suffix}，"
                f"拟合频段 {float(operation.get('low', 0)):g}–{float(operation.get('high', 0)):g} Hz，"
                f"decim={operation.get('decim', '—')}，最大迭代={operation.get('max_iter', '—')}）"
            )
        if name in {"merge_repaired_channel", "channel_replacement"}:
            channels = self._format_channel_scope(operation.get("channel_ids", []))
            channel_ids = self._provenance_integer_ids(operation.get("channel_ids", []))
            nested = [
                text for item in operation.get("channel_processing_operations", [])
                if isinstance(item, dict)
                and (
                    not self._provenance_integer_ids(item.get("channel_ids", []))
                    or bool(set(channel_ids).intersection(
                        self._provenance_integer_ids(item.get("channel_ids", []))
                    ))
                )
                and (text := self._describe_preprocess_operation(item))
            ]
            detail = " → ".join(nested)
            return f"替换 {channels}" + (f"（通道处理：{detail}）" if detail else "")
        return None

    def _live_preprocess_state_text(self, source, suffix: str = "") -> str:
        """Build the data-state banner from the current in-memory lineage."""
        meta = getattr(source, "metadata", None)
        provenance = getattr(meta, "provenance", None) or {}
        steps = [
            text for operation in provenance.get("operations", [])
            if isinstance(operation, dict)
            and (text := self._describe_preprocess_operation(operation))
        ]
        history = " → ".join(steps) if steps else "未记录处理步骤"
        time_summary = self._source_time_summary(meta)
        return (
            f"数据状态：已处理。\n{time_summary}\n处理步骤：{history}"
            + (f"\n{suffix}" if suffix else "")
        )

    @staticmethod
    def _source_time_summary(meta) -> str:
        """Describe the actual BIN segment represented by the current source."""
        provenance = getattr(meta, "provenance", None) or {}
        bin_entry = next((
            item for item in reversed(provenance.get("lineage", []))
            if isinstance(item, dict) and str(item.get("kind", "")).lower() == "bin"
        ), None)
        duration = float(meta.rows) / float(meta.fs)
        relative_offset = float(getattr(meta, "time_offset", 0.0) or 0.0)
        if bin_entry is not None:
            bin_name = Path(str(bin_entry.get("path", ""))).name or "未命名 BIN"
            original_start = float(bin_entry.get("selected_start_sec", 0.0) or 0.0)
            start = original_start + relative_offset
            source_text = f"原始 BIN：{bin_name}；"
        else:
            start = relative_offset
            source_text = f"数据文件：{Path(str(meta.path)).name}；"
        end = start + duration
        return (
            f"{source_text}当前数据范围 {start:.3f}–{end:.3f} s；"
            f"截取时长 {duration:.3f} s（{int(meta.rows):,} 采样点，FS={float(meta.fs):g} Hz）"
        )

    def _show_loaded_preprocess_summary(self, meta, qc_restored: bool) -> None:
        provenance = getattr(meta, "provenance", None) or {}
        provenance_operations = [
            item for item in provenance.get("operations", []) if isinstance(item, dict)
        ]
        operations = [
            text for item in provenance_operations
            if (text := self._describe_preprocess_operation(item))
        ]
        self._loaded_preprocessed_operations = self._preprocess_operation_names(meta)
        self._repeat_preprocessing_acknowledged = set()
        filter_operations = [item for item in provenance_operations if str(item.get("name", "")).lower() == "filter"]
        filtered_ids = {
            int(channel) for item in filter_operations
            for channel in self._provenance_integer_ids(item.get("channel_ids", []))
            if str(item.get("mode", "off")).lower() != "off" or bool(item.get("notch_enabled"))
        }
        self._preprocess_filtered_channel_ids = filtered_ids
        self._preprocess_notched_channel_ids = {
            int(channel) for item in filter_operations
            if bool(item.get("notch_enabled", item.get("notch", False)))
            and np.isclose(float(item.get("notch_frequency_hz", item.get("notch_frequency", 50.0))), 50.0)
            for channel in self._provenance_integer_ids(item.get("channel_ids", []))
        }
        lowpass_operations = [
            item for item in filter_operations if str(item.get("mode", "")).lower() in {"lowpass", "bandpass"}
        ]
        self._preprocess_lowpass_channel_ids = {
            int(channel) for item in lowpass_operations
            for channel in self._provenance_integer_ids(item.get("channel_ids", []))
        }
        if lowpass_operations:
            last_filter = lowpass_operations[-1]
            high_value = last_filter.get("high_hz")
            if high_value is None:
                high_value = last_filter.get("high")
            try:
                self._preprocess_lowpass_high_hz = float(high_value) if high_value is not None else None
            except (TypeError, ValueError):
                self._preprocess_lowpass_high_hz = None
        else:
            self._preprocess_lowpass_high_hz = None
        if not hasattr(self, "loaded_preprocess_summary"):
            return
        if _provenance_stage(meta, "raw") != "preprocessed" and not operations:
            self.loaded_preprocess_summary.clear()
            self.loaded_preprocess_summary.setVisible(False)
            return
        steps = " → ".join(operations) if operations else "文件未记录详细处理步骤"
        if qc_restored:
            channel_state = f"健康 {len(self.good_channel_ids)}｜坏道 {len(self.bad_channel_ids)}"
        elif getattr(self, "_loaded_qc_skipped_for_test", False):
            channel_state = "坏道检查已跳过（测试数据）；全部通道曾暂按健康通道处理"
        else:
            channel_state = "未记录可恢复的通道 QC 注释"
        self.loaded_preprocess_summary.setText(
            "【已加载预处理数据】\n"
            f"{self._source_time_summary(meta)}\n"
            f"处理步骤：{steps}\n"
            f"通道状态：{channel_state}\n"
            "数据已处理；再次滤波、重参考或 ICA 可能造成重复处理，请核对后再执行。"
        )
        self.loaded_preprocess_summary.setVisible(True)

    def _restore_preprocess_processing_state(self, path: Path, meta) -> dict:
        """Restore the complete processing state embedded in a preprocessed H5."""
        state = {}
        try:
            with h5py.File(path, "r") as h5:
                raw = h5.attrs.get("preprocess_state_json", "")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            candidate = json.loads(str(raw)) if raw else {}
            if (isinstance(candidate, dict)
                    and candidate.get("schema") == "sd-preprocess-state"
                    and int(candidate.get("version", -1)) == 1):
                state = candidate
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            state = {}
        provenance = getattr(meta, "provenance", None) or {}
        operations = [item for item in provenance.get("operations", []) if isinstance(item, dict)]
        names = {str(item.get("name", "")).lower() for item in operations}
        if not state:
            replacement_ids = {
                int(channel) for item in operations
                if str(item.get("name", "")).lower() in {"channel_replacement", "merge_repaired_channel"}
                for channel in self._provenance_integer_ids(item.get("channel_ids", []))
            }
            state = {
                "schema": "sd-preprocess-state", "version": 1,
                "data_stage": _provenance_stage(meta, "preprocessed"),
                "remapped": "channel_remap" in names,
                "car_applied": "reference" in names,
                "ica_applied": "ica" in names,
                "filtered_channel_ids": sorted(self._preprocess_filtered_channel_ids),
                "notched_channel_ids": sorted(self._preprocess_notched_channel_ids),
                "lowpass_channel_ids": sorted(self._preprocess_lowpass_channel_ids),
                "lowpass_high_hz": self._preprocess_lowpass_high_hz,
                "replacement_channel_ids": sorted(replacement_ids),
                "operation_names": sorted(names),
            }
        channel_processing = state.get("channel_processing", {})
        if not isinstance(channel_processing, dict):
            channel_processing = {}
        # Backward compatibility: replacement files created before the
        # unified state field still carry the per-channel chain in provenance.
        for item in operations:
            if str(item.get("name", "")).lower() not in {"merge_repaired_channel", "channel_replacement"}:
                continue
            nested = item.get("channel_processing_operations", [])
            for channel in self._provenance_integer_ids(item.get("channel_ids", [])):
                if str(int(channel)) not in channel_processing and isinstance(nested, list):
                    channel_processing[str(int(channel))] = [
                        entry for entry in nested if isinstance(entry, dict)
                        and (
                            not self._provenance_integer_ids(entry.get("channel_ids", []))
                            or int(channel) in self._provenance_integer_ids(entry.get("channel_ids", []))
                        )
                    ]
        state["channel_processing"] = channel_processing
        available = {int(channel) for channel in meta.channel_ids}
        self._preprocess_filtered_channel_ids = {
            int(channel) for channel in state.get("filtered_channel_ids", []) if int(channel) in available
        }
        self._preprocess_lowpass_channel_ids = {
            int(channel) for channel in state.get("lowpass_channel_ids", []) if int(channel) in available
        }
        if "notched_channel_ids" in state:
            self._preprocess_notched_channel_ids = {
                int(channel) for channel in state.get("notched_channel_ids", []) if int(channel) in available
            }
        else:
            self._preprocess_notched_channel_ids &= available
        try:
            value = state.get("lowpass_high_hz")
            self._preprocess_lowpass_high_hz = None if value is None else float(value)
        except (TypeError, ValueError):
            self._preprocess_lowpass_high_hz = None
        self._loaded_replacement_channel_ids = {
            int(channel) for channel in state.get("replacement_channel_ids", []) if int(channel) in available
        }
        self._loaded_preprocess_state = dict(state)
        labels = []
        if state.get("remapped"): labels.append("已重映射")
        if self._preprocess_filtered_channel_ids:
            labels.append(f"已滤波 {len(self._preprocess_filtered_channel_ids)} 通道")
        if state.get("car_applied"): labels.append("已应用 CAR")
        if state.get("ica_applied"): labels.append("已应用 ICA")
        if self._loaded_replacement_channel_ids:
            labels.append("已替换通道 " + ",".join(map(str, sorted(self._loaded_replacement_channel_ids))))
        replacement_details = []
        for channel in sorted(self._loaded_replacement_channel_ids):
            descriptions = [
                text for item in channel_processing.get(str(channel), [])
                if isinstance(item, dict)
                and (
                    not self._provenance_integer_ids(item.get("channel_ids", []))
                    or channel in self._provenance_integer_ids(item.get("channel_ids", []))
                )
                and (text := self._describe_preprocess_operation(item))
            ]
            replacement_details.append(
                f"ch{channel}：" + (" → ".join(descriptions) if descriptions else "未记录具体通道处理参数")
            )
        if self._bad_channel_check_completed:
            labels.append(
                f"QC：健康 {len(self.good_channel_ids)}、坏道 {len(self.bad_channel_ids)}"
            )
        detail = "；".join(labels) if labels else "未记录详细处理状态"
        self._set_preprocess_data_state(
            f"数据状态：已加载预处理 H5；{detail}。"
            + (f" 替换通道处理追溯：{'；'.join(replacement_details)}。" if replacement_details else "")
            + " 当前预览和后续分析使用该完整处理后数据。"
        )
        self.filter_var.setText(
            f"滤波：已处理 {len(self._preprocess_filtered_channel_ids)} 个通道"
            if self._preprocess_filtered_channel_ids else "滤波：文件未记录已滤波通道"
        )
        return state

    def _confirm_repeated_preprocessing(self, operation_names: set[str], action_label: str) -> bool:
        loaded = set(getattr(self, "_loaded_preprocessed_operations", set()))
        acknowledged = set(getattr(self, "_repeat_preprocessing_acknowledged", set()))
        repeated = loaded.intersection(operation_names) - acknowledged
        if not repeated:
            return True
        labels = {"channel_remap": "通道重映射", "filter": "滤波", "reference": "CAR 重参考", "ica": "ICA"}
        names = "、".join(labels.get(name, name) for name in sorted(repeated))
        answer = QMessageBox.question(
            self, "可能重复预处理",
            f"导入文件的处理记录显示已经执行过：{names}。\n再次执行{action_label}可能改变数据并造成重复处理。\n\n仍要继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self._repeat_preprocessing_acknowledged = acknowledged | repeated
        return True

    @staticmethod
    def _is_preprocessed_h5(path: Path, meta) -> bool:
        if _provenance_stage(meta, "raw") == "preprocessed":
            return True
        try:
            if h5py.is_hdf5(path):
                with h5py.File(path, "r") as h5:
                    value = h5.attrs.get("data_stage", "")
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                return str(value).strip().lower() == "preprocessed"
        except OSError:
            pass
        return False

    def _write_preprocessed_h5(self, path: Path, source) -> Path:
        """Write a complete processed matrix and a same-basename ledger CSV."""
        path = Path(path)
        record_path = path.with_suffix(".csv")
        meta = source.metadata
        source_meta = getattr(getattr(self, "source", None), "metadata", None)
        filtered_ids = sorted(int(channel) for channel in self._preprocess_filtered_channel_ids)
        provenance = getattr(meta, "provenance", None) or {}
        exported_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        qc_snapshot = self._preprocess_qc_snapshot(source)
        qc_json = json.dumps(
            qc_snapshot, ensure_ascii=False, sort_keys=True,
            default=self._preprocess_qc_json_default,
        )
        original_bin = self._original_bin_path()
        operation_names = {
            str(item.get("name", "")).lower() for item in provenance.get("operations", [])
            if isinstance(item, dict)
        }
        channel_processing = {}
        for item in provenance.get("operations", []):
            if not isinstance(item, dict) or str(item.get("name", "")).lower() not in {
                "channel_replacement", "merge_repaired_channel"
            }:
                continue
            nested = item.get("channel_processing_operations", [])
            for channel in self._provenance_integer_ids(item.get("channel_ids", [])):
                channel_processing[str(int(channel))] = [
                    entry for entry in nested if isinstance(entry, dict)
                ] if isinstance(nested, list) else []
        preprocess_state = {
            "schema": "sd-preprocess-state", "version": 1,
            "data_stage": "preprocessed",
            "remapped": "channel_remap" in operation_names,
            "car_applied": "reference" in operation_names,
            "ica_applied": "ica" in operation_names,
            "filtered_channel_ids": filtered_ids,
            "notched_channel_ids": sorted(int(channel) for channel in self._preprocess_notched_channel_ids),
            "lowpass_channel_ids": sorted(int(channel) for channel in self._preprocess_lowpass_channel_ids),
            "lowpass_high_hz": self._preprocess_lowpass_high_hz,
            "replacement_channel_ids": sorted({
                int(channel) for item in provenance.get("operations", []) if isinstance(item, dict)
                and str(item.get("name", "")).lower() in {"channel_replacement", "merge_repaired_channel"}
                for channel in self._provenance_integer_ids(item.get("channel_ids", []))
            }),
            "qc_completed": bool(qc_snapshot.get("completed")),
            "operation_names": sorted(operation_names),
            "channel_processing": channel_processing,
        }
        self._write_source_h5(
            path,
            source,
            stage="preprocessed",
            operation={
                "name": "export_hdf5",
                "scope": "all_channels",
                "data_product": "preprocessed",
                "processing_record_csv": record_path.name,
                "filtered_channel_ids": filtered_ids,
            },
            attributes={
                "data_stage": "preprocessed",
                "processing_record_csv": record_path.name,
                "preprocessing_exported_utc": exported_utc,
                "filtered_channel_ids": np.asarray(filtered_ids, dtype=np.int64),
                "preprocess_qc_schema": qc_snapshot["schema"],
                "preprocess_qc_version": qc_snapshot["version"],
                "preprocess_qc_json": qc_json,
                "preprocess_state_schema": "sd-preprocess-state",
                "preprocess_state_version": 1,
                "preprocess_state_json": json.dumps(
                    preprocess_state, ensure_ascii=False, sort_keys=True,
                    default=self._preprocess_qc_json_default,
                ),
            },
        )
        import csv

        operations = provenance.get("operations", []) if isinstance(provenance, dict) else []
        record = {
            "source_file": str(original_bin or (source_meta.path if source_meta is not None else "")),
            "output_file": path.name,
            "processed_at_utc": exported_utc,
            "data_stage": "preprocessed",
            "sampling_rate_hz": float(meta.fs),
            "channel_ids": ",".join(str(int(channel)) for channel in meta.channel_ids),
            "filtered_channel_ids": ",".join(str(channel) for channel in filtered_ids),
            "processing_operations_json": json.dumps(
                compact_provenance_operations(operations), ensure_ascii=True, sort_keys=True,
                separators=(",", ":"),
            ),
        }
        qc_rows = list(qc_snapshot["channel_rows"])
        fields = [
            *record,
            "qc_completed", "channel", "final_status", "automatic_status",
            "automatic_reason", "brief_description", "manual_override", "manual_override_applied",
        ]
        with open(record_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            if qc_rows:
                for qc_row in qc_rows:
                    writer.writerow({
                        **record,
                        "qc_completed": bool(qc_snapshot["completed"]),
                        **qc_row,
                    })
            else:
                writer.writerow({**record, "qc_completed": bool(qc_snapshot["completed"])})
        return record_path

    def export_current_h5(self) -> None:
        if not self.active_source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载数据后再导出。")
            return
        path_text = self.export_path_edit.text().strip()
        if not path_text:
            self.choose_export_path()
            path_text = self.export_path_edit.text().strip()
        if not path_text:
            return
        try:
            path = Path(path_text)
            meta = self.active_source.metadata
            chunk_rows = _export_chunk_rows(meta)
            with h5py.File(path, "w") as h5:
                _write_legacy_compression_metadata(h5)
                h5.attrs["storage_dtype"] = "float32"
                h5.attrs["hdf5_chunk_shape"] = f"({chunk_rows}, {meta.channels})"
                output = h5.create_dataset(
                    "rawData512",
                    shape=(meta.rows, meta.channels),
                    dtype=np.float32,
                    chunks=(chunk_rows, meta.channels),
                    **_legacy_h5_compression_kwargs(),
                )
                for first in range(0, meta.rows, chunk_rows):
                    last = min(meta.rows, first + chunk_rows)
                    output[first:last] = self.active_source.read(first, last, slice(None))
                h5["FS"] = meta.fs
                h5["data_unit"] = "mV"
                h5["time_offset_sec"] = meta.time_offset
                h5.attrs["time_offset_sec"] = meta.time_offset
                # Preserve physical/FPC channel identity for a later custom
                # import.  Position 1 in a 1-channel export can therefore
                # still be correctly shown as e.g. channel 500.
                h5.create_dataset("channel_ids", data=np.asarray(meta.channel_ids, dtype=np.int64))
                _write_timing_metadata(h5, meta.timing_metadata or self.bin_timing_metadata)
                _write_output_provenance(
                    h5, meta, stage=_provenance_stage(meta, "exported"), channel_ids=meta.channel_ids,
                    operation={"name": "export_hdf5", "scope": "all_channels"},
                )
                if hasattr(self, "timing_info"):
                    h5.attrs["timing_info"] = str(self.timing_info)
            self.export_status.setText(f"已导出：{path}")
            self.statusBar().showMessage(f"已导出当前数据：{path.name}")
        except Exception as exc:
            self.export_status.setText(f"导出失败：{exc}")
            QMessageBox.critical(self, "导出失败", str(exc))

    def export_selected_channels_h5(self) -> None:
        source = self.active_source
        selected = self.lfp_selected_ids or self.spike_selected_ids
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载数据后再导出。")
            return
        if not selected:
            QMessageBox.information(self, "尚未选择通道", "请先在 LFP、Spike 或预处理页选择通道。")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "导出当前选择通道 H5", str(Path.cwd() / "selected_channels.h5"), "HDF5 files (*.h5)")
        if not filename:
            return
        try:
            meta = source.metadata
            columns = np.flatnonzero(np.isin(meta.channel_ids, list(selected)))
            if not columns.size:
                raise ValueError("选择的通道不在当前数据源中。")
            chunk_rows = _export_chunk_rows(meta)
            with h5py.File(filename, "w") as h5:
                _write_legacy_compression_metadata(h5)
                h5.attrs["storage_dtype"] = "float32"
                h5.attrs["hdf5_chunk_shape"] = f"({chunk_rows}, {columns.size})"
                output = h5.create_dataset(
                    "rawData512", shape=(meta.rows, columns.size), dtype=np.float32,
                    chunks=(chunk_rows, columns.size), **_legacy_h5_compression_kwargs(),
                )
                for first in range(0, meta.rows, chunk_rows):
                    output[first:min(meta.rows, first + chunk_rows)] = source.read(first, min(meta.rows, first + chunk_rows), columns)
                h5["FS"] = meta.fs; h5["data_unit"] = "mV"; h5["time_offset_sec"] = meta.time_offset; h5.attrs["time_offset_sec"] = meta.time_offset
                h5.create_dataset("channel_ids", data=np.asarray(meta.channel_ids)[columns])
                _write_timing_metadata(h5, meta.timing_metadata or self.bin_timing_metadata)
                _write_output_provenance(
                    h5, meta, stage=_provenance_stage(meta, "exported"), channel_ids=np.asarray(meta.channel_ids)[columns],
                    operation={"name": "export_hdf5", "scope": "selected_channels"},
                )
            self.export_status.setText(f"已导出 {columns.size} 个选择通道（保留 channel_ids）：{filename}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def export_processed_channel_h5_files(self) -> None:
        """Write the legacy one-channel ``SD_processed_channel_hdf5`` files."""
        source = self.active_source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载数据后再导出。")
            return
        selected = self.lfp_selected_ids or self.spike_selected_ids
        if not selected:
            try:
                selected = set(np.asarray(source.metadata.channel_ids)[self._selected_preprocess_columns()])
            except (TypeError, ValueError):
                selected = set(source.metadata.channel_ids)
        directory = QFileDialog.getExistingDirectory(self, "选择处理后通道 H5 保存目录", str(Path.cwd()))
        if not directory:
            return
        try:
            meta = source.metadata
            columns = np.flatnonzero(np.isin(meta.channel_ids, sorted(selected)))
            if not columns.size:
                raise ValueError("选择的通道不在当前数据源中。")
            source_name = meta.path.stem if not str(meta.path).startswith("<") else "processed_data"
            date_token = self.date_var.text().replace("-", "") if isinstance(self.date_var, QLineEdit) else ""
            date_token = date_token if len(date_token) == 8 and date_token.isdigit() else "processed"
            animal = self.animal_var.text().strip() if isinstance(self.animal_var, QLineEdit) else "0"
            block = self.block_var.text().strip() if isinstance(self.block_var, QLineEdit) else "0"
            root = Path(directory) / "channel_exports" / "processed" / date_token / (animal or "0") / source_name
            root.mkdir(parents=True, exist_ok=True)
            chunk_rows = _export_chunk_rows(meta)
            string_dtype = h5py.string_dtype(encoding="utf-8")
            for column in columns:
                channel_id = int(meta.channel_ids[column])
                path = root / f"{date_token}_{animal or 0}_block{block or 0}_ch{channel_id:03d}.h5"
                with h5py.File(path, "w") as h5:
                    _write_legacy_compression_metadata(h5)
                    h5.attrs["format"] = "SD_processed_channel_hdf5"
                    h5.attrs["data_stage"] = "preprocessed"
                    h5.attrs["storage_dtype"] = "float32"
                    h5.attrs["hdf5_chunk_shape"] = f"({chunk_rows},)"
                    signal_out = h5.create_dataset("signal", shape=(meta.rows,), dtype=np.float32, chunks=(chunk_rows,), **_legacy_h5_compression_kwargs())
                    time_out = h5.create_dataset("time", shape=(meta.rows,), dtype=np.float64, chunks=(chunk_rows,), **_legacy_h5_compression_kwargs())
                    for first in range(0, meta.rows, chunk_rows):
                        last = min(meta.rows, first + chunk_rows)
                        signal_out[first:last] = source.read(first, last, int(column))
                        time_out[first:last] = meta.time_offset + np.arange(first, last, dtype=np.float64) / meta.fs
                    h5.create_dataset("FS", data=np.asarray(meta.fs, dtype=np.float64))
                    h5.create_dataset("channel", data=np.asarray(channel_id, dtype=np.int32))
                    h5.create_dataset("physical_channel_id", data=np.asarray(channel_id, dtype=np.int32))
                    h5.create_dataset("column_index", data=np.asarray(int(column), dtype=np.int32))
                    h5.attrs["channel_mapping_version"] = 2
                    h5.create_dataset("source_bin", data=str(meta.path), dtype=string_dtype)
                    h5.create_dataset("date", data=date_token, dtype=string_dtype)
                    h5.create_dataset("animal", data=np.asarray(int(animal or 0), dtype=np.int32))
                    h5.create_dataset("block", data=np.asarray(int(block or 0), dtype=np.int32))
                    h5.create_dataset("data_unit", data="mV", dtype=string_dtype)
                    h5.create_dataset("data_stage", data="preprocessed", dtype=string_dtype)
                    _write_timing_metadata(h5, meta.timing_metadata or self.bin_timing_metadata)
                    _write_output_provenance(
                        h5, meta, stage=_provenance_stage(meta, "preprocessed"), channel_ids=[channel_id],
                        operation={"name": "export_channel", "format": "SD_processed_channel_hdf5"},
                    )
            self.export_status.setText(f"已按旧版逐通道格式导出 {len(columns)} 个 H5：{root}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def export_spike_plot(self) -> None:
        if not self.spike_rows:
            QMessageBox.information(self, "尚无结果", "请先运行 Spike 检测。")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "导出 Spike 图", str(Path.cwd() / "spike_results.png"), "PNG files (*.png);;PDF files (*.pdf)")
        if filename:
            try:
                self.spike_figure.savefig(filename, dpi=180)
                self.export_status.setText(f"已导出 Spike 图：{filename}")
            except Exception as exc:
                QMessageBox.critical(self, "导出失败", str(exc))

    def _write_preprocess_qc_csvs(self, root: Path) -> list[str]:
        """Write the completed bad-channel/QC metrics without recalculating them."""
        if not self._bad_channel_check_completed:
            raise ValueError("请先在预处理页完成坏道检查，再导出预处理 QC 结果。")

        import csv

        def write_records(filename: str, fields: list[str], records: list[dict]) -> str:
            path = root / filename
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)
            return path.name

        bad_reasons = {int(channel): str(reason) for channel, reason in self.bad_channel_auto_reasons.items()}
        summary_records = []
        for channel in sorted(int(value) for value in self._bad_channel_result_channel_ids):
            manual = self.manual_channel_overrides.get(channel)
            if manual == "good":
                final_status, manual_override = "healthy", "healthy"
            elif manual == "bad":
                final_status, manual_override = "bad", "bad"
            elif channel in bad_reasons:
                final_status, manual_override = "bad", ""
            else:
                final_status, manual_override = "healthy", ""
            automatic_status = "bad" if channel in bad_reasons else "healthy"
            automatic_reason = bad_reasons.get(channel, "")
            review_detail = (
                ("人工健康覆盖；" if manual == "good" else "人工坏道覆盖；" if manual == "bad" else "")
                + automatic_reason
            )
            summary_records.append({
                "channel": channel,
                "final_status": final_status,
                "automatic_status": automatic_status,
                "automatic_reason": automatic_reason,
                "brief_description": self._brief_channel_review_description(review_detail, final_status),
                "manual_override": manual_override,
                "manual_override_applied": bool(manual_override),
            })

        saturation_fields = [
            "channel", "finite_samples", "channel_min", "channel_ptp",
            "saturation_width_percent", "bottom_limit", "bottom_ratio",
            "saturation_ratio_threshold", "bottom_bad", "is_bad",
        ]
        target_fields = [
            "channel", "target", "tolerance", "finite_samples", "near_target_samples",
            "near_target_ratio", "threshold", "is_bad",
        ]
        saturation_records = sorted(
            (dict(row) for row in getattr(self, "bad_channel_fast_artifact_rows", [])),
            key=lambda row: int(row["channel"]),
        )
        target_records = sorted(
            (dict(row) for row in getattr(self, "bad_channel_high_frequency_noise_rows", [])),
            key=lambda row: int(row["channel"]),
        )
        saturation_by_channel = {int(row["channel"]): row for row in saturation_records}
        target_by_channel = {int(row["channel"]): row for row in target_records}
        channel_metric_fields = [
            "channel", "final_status", "automatic_status", "automatic_reason", "brief_description",
            "manual_override", "manual_override_applied",
            *(f"saturation_{field}" for field in saturation_fields[1:]),
            *(f"target_2_5mv_{field}" for field in target_fields[1:]),
        ]
        channel_metrics = []
        for summary in summary_records:
            channel = int(summary["channel"])
            record = dict(summary)
            for field in saturation_fields[1:]:
                record[f"saturation_{field}"] = saturation_by_channel.get(channel, {}).get(field, "")
            for field in target_fields[1:]:
                record[f"target_2_5mv_{field}"] = target_by_channel.get(channel, {}).get(field, "")
            channel_metrics.append(record)

        def describe_source(source) -> dict | None:
            if source is None or not getattr(source, "loaded", False):
                return None
            meta = source.metadata
            return {
                "path": str(meta.path), "dataset": str(meta.dataset),
                "rows": int(meta.rows), "channels": int(meta.channels),
                "sampling_rate_hz": float(meta.fs), "time_offset_sec": float(meta.time_offset),
                "channel_ids": [int(value) for value in meta.channel_ids],
                "provenance": getattr(meta, "provenance", None),
            }

        worker = getattr(self, "_bad_channel_worker", None)
        qc_source = getattr(worker, "source", None) or self._detection_source()
        current_source = self.active_source if getattr(self.active_source, "loaded", False) else None
        from datetime import datetime, timezone
        run_meta = {
            "schema": "sd-preprocess-qc-export",
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "application": {"entrypoint": "qt_gui.py", "version": "unversioned"},
            "quality_control": {
                "settings": dict(getattr(worker, "settings", {})),
                "evaluated_channel_ids": [int(row["channel"]) for row in summary_records],
                "automatic_bad_channel_ids": sorted(bad_reasons),
                "manual_overrides": {
                    str(channel): ("healthy" if value == "good" else value)
                    for channel, value in sorted(self.manual_channel_overrides.items())
                },
            },
            "qc_source": describe_source(qc_source),
            "current_output_source": describe_source(current_source),
            "files": {
                "channel_metrics": "channel_metrics.csv",
                "saturation_detail": "preprocess_saturation.csv",
                "target_2_5mv_detail": "preprocess_2_5mv_concentration.csv",
            },
        }
        (root / "run_meta.json").write_text(
            json.dumps(
                run_meta, ensure_ascii=False, indent=2, sort_keys=True,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
            ),
            encoding="utf-8",
        )
        return [
            "run_meta.json",
            write_records("channel_metrics.csv", channel_metric_fields, channel_metrics),
            write_records("preprocess_saturation.csv", saturation_fields, saturation_records),
            write_records("preprocess_2_5mv_concentration.csv", target_fields, target_records),
        ]

    def export_preprocess_qc_csv(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择预处理 QC 结果输出目录", str(Path.cwd()))
        if not directory:
            return
        try:
            written = self._write_preprocess_qc_csvs(Path(directory))
            self.export_status.setText("已导出预处理 QC 结果：" + "、".join(written))
        except Exception as exc:
            QMessageBox.critical(self, "预处理 QC 导出失败", str(exc))

    def export_checked_results(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择勾选结果的导出目录", str(Path.cwd()))
        if not directory:
            return
        try:
            import csv
            root = Path(directory); root.mkdir(parents=True, exist_ok=True); written = []
            if self.export_lfp_check.isChecked() and self.lfp_rows:
                path = root / "lfp_snr.csv"
                with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=sorted(self.lfp_rows[0])); writer.writeheader(); writer.writerows(self.lfp_rows)
                written.append(path.name)
            if self.export_spike_check.isChecked() and self.spike_rows:
                path = root / "spike_results.csv"
                with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=sorted(self.spike_rows[0])); writer.writeheader(); writer.writerows(self.spike_rows)
                written.append(path.name)
            if self.export_spike_plot_check.isChecked() and self.spike_rows:
                path = root / "spike_results.png"; self.spike_figure.savefig(path, dpi=180); written.append(path.name)
            if hasattr(self, "task_results"):
                rows = [metric for result in self.task_results.get("results", []) for metric in result.get("metrics", [])]
                if rows:
                    path = root / "task_response_metrics.csv"
                    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
                    written.append(path.name)
            if self.export_preprocess_qc_check.isChecked():
                written.extend(self._write_preprocess_qc_csvs(root))
            if self.export_processed_h5_check.isChecked() and self.active_source.loaded:
                self._write_source_h5(root / "processed_data.h5", self.active_source); written.append("processed_data.h5")
            if self.export_selected_h5_check.isChecked() and (self.lfp_selected_ids or self.spike_selected_ids):
                self._write_selected_h5(root / "selected_channels.h5", self.active_source, self.lfp_selected_ids or self.spike_selected_ids); written.append("selected_channels.h5")
            self.export_status.setText("已导出勾选结果：" + ("、".join(written) if written else "没有可导出的已完成结果"))
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _write_source_h5(
        self,
        path: Path,
        source,
        *,
        stage: str | None = None,
        operation: dict | None = None,
        attributes: dict | None = None,
    ) -> None:
        meta = source.metadata; chunk_rows = _export_chunk_rows(meta)
        with h5py.File(path, "w") as h5:
            _write_legacy_compression_metadata(h5); h5.attrs["storage_dtype"] = "float32"; h5.attrs["hdf5_chunk_shape"] = f"({chunk_rows}, {meta.channels})"
            target = h5.create_dataset("rawData512", shape=(meta.rows, meta.channels), dtype=np.float32, chunks=(chunk_rows, meta.channels), **_legacy_h5_compression_kwargs())
            for first in range(0, meta.rows, chunk_rows): target[first:min(meta.rows, first + chunk_rows)] = source.read(first, min(meta.rows, first + chunk_rows), slice(None))
            h5["FS"] = meta.fs; h5["data_unit"] = "mV"; h5["time_offset_sec"] = meta.time_offset; h5.create_dataset("channel_ids", data=np.asarray(meta.channel_ids, dtype=np.int64))
            _write_timing_metadata(h5, meta.timing_metadata or self.bin_timing_metadata)
            for key, value in (attributes or {}).items():
                h5.attrs[str(key)] = value
            _write_output_provenance(
                h5, meta, stage=stage or _provenance_stage(meta, "exported"), channel_ids=meta.channel_ids,
                operation=operation or {"name": "export_hdf5", "scope": "all_channels"},
            )

    def _write_selected_h5(self, path: Path, source, selected) -> None:
        meta = source.metadata; columns = np.flatnonzero(np.isin(meta.channel_ids, list(selected)))
        if not columns.size: return
        chunk_rows = _export_chunk_rows(meta)
        with h5py.File(path, "w") as h5:
            _write_legacy_compression_metadata(h5); target = h5.create_dataset("rawData512", shape=(meta.rows, columns.size), dtype=np.float32, chunks=(chunk_rows, columns.size), **_legacy_h5_compression_kwargs())
            for first in range(0, meta.rows, chunk_rows): target[first:min(meta.rows, first + chunk_rows)] = source.read(first, min(meta.rows, first + chunk_rows), columns)
            h5["FS"] = meta.fs; h5["data_unit"] = "mV"; h5["time_offset_sec"] = meta.time_offset; h5.create_dataset("channel_ids", data=np.asarray(meta.channel_ids)[columns])
            _write_timing_metadata(h5, meta.timing_metadata or self.bin_timing_metadata)
            _write_output_provenance(
                h5, meta, stage=_provenance_stage(meta, "exported"), channel_ids=np.asarray(meta.channel_ids)[columns],
                operation={"name": "export_hdf5", "scope": "selected_channels"},
            )

    def open_output_directory(self) -> None:
        import os
        directory = self.output_dir_edit.text().strip() if hasattr(self, "output_dir_edit") else ""
        if not directory:
            directory = str(Path(self.export_path_edit.text()).parent) if self.export_path_edit.text() else str(Path.cwd())
        if Path(directory).is_dir():
            os.startfile(directory) if sys.platform.startswith("win") else None

    def choose_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择脑电 HDF5 / MAT 文件",
            self.path_edit.text() or str(Path.cwd()),
            "EEG files (*.h5 *.hdf5 *.mat);;HDF5 files (*.h5 *.hdf5);;MAT files (*.mat);;All files (*.*)",
        )
        if filename:
            # Selecting H5/MAT makes it the active import source instead of a
            # stale BIN path retained from an earlier operation.
            self.bin_path_edit.clear()
            self.path_edit.setText(filename)
            self.load_file()

    def load_file(self) -> None:
        try:
            filename = Path(self.path_edit.text().strip())
            if filename.suffix.lower() == ".mat" and not h5py.is_hdf5(filename):
                self.source = self._load_mat_source(filename)
                meta = self.source.metadata
            else:
                if bool(getattr(self, "_load_file_use_full_range_once", False)):
                    start_seconds, duration_seconds = 0.0, 0.0
                    self._load_file_use_full_range_once = False
                else:
                    try:
                        start_seconds, duration_seconds = self._import_time_window()
                    except ValueError as exc:
                        raise ValueError(str(exc)) from exc
                self.source = LazyH5Source()
                meta = self.source.open(filename, start_seconds, duration_seconds)
            self.active_source = self.source
            # A pending extracted-channel repair belongs to the previously
            # loaded preprocessing H5 and must never carry across files.
            self._channel_repair_context = None
            self.__dict__.pop("_channel_repair_base_state_text", None)
            self.__dict__.pop("_extracted_channel_preview_source", None)
            if hasattr(self, "preprocess_filter_channels_var"):
                previous_filter_text = getattr(self, "_channel_repair_base_filter_channels_text", None)
                if previous_filter_text is not None:
                    self.preprocess_filter_channels_var.setText(previous_filter_text)
                self.preprocess_filter_channels_var.setReadOnly(False)
                self.preprocess_filter_channels_var.setToolTip("")
            self.__dict__.pop("_channel_repair_base_filter_channels_text", None)
            if hasattr(self, "merge_repaired_channel_button"):
                self.merge_repaired_channel_button.setEnabled(False)
            self.remapped_source = None
            self.preprocessed_source = None
            self.analysis_cache = AnalysisCache()
            self._last_filter_cache_signature = None
            self._last_filter_cache_value = None
            self._pending_filter_cache_signature = None
            self._remapped_detection_signature = None
            self._pending_remapped_detection_signature = None
            self._selected_analysis_batch = None
            self._preprocess_lowpass_channel_ids = set()
            self._preprocess_lowpass_high_hz = None
            self._preprocess_filtered_channel_ids = set()
            self._preprocess_notched_channel_ids = set()
            self.__dict__.pop("_last_lowpass_warning_source", None)
            self._bad_channel_check_completed = False
            self._sync_dual_stream_export_enabled()
            self.bad_channel_auto_reasons = {}
            self.bad_channel_candidate_reasons = {}
            self.bad_channel_fast_artifact_rows = []
            self.bad_channel_high_frequency_noise_rows = []
            self.bad_channel_ids = set()
            self.good_channel_ids = set()
            self.bad_channel_candidate_ids = set()
            self._bad_channel_result_channel_ids = ()
            self.manual_channel_overrides = {}
            self._bad_channel_fast_only = False
            if hasattr(self, "apply_car_button"):
                self.apply_car_button.setEnabled(False)
            is_preprocessed_h5 = self._is_preprocessed_h5(filename, meta)
            restored_preprocess_qc = self._restore_preprocess_qc_from_h5(filename, meta)
            if is_preprocessed_h5:
                self.preprocessed_source = self.source
                if hasattr(self, "preprocess_export_button"):
                    self.preprocess_export_button.setEnabled(True)
            elif hasattr(self, "preprocess_export_button"):
                self.preprocess_export_button.setEnabled(False)
            self._apply_loaded_timing(meta)
            self.metadata_label.setText(
                f"数据集：{meta.dataset}　尺寸：{meta.rows:,} 采样点 × {meta.channels} 通道　"
                f"采样率：{meta.fs:g} Hz　存储：{meta.chunks or 'MAT/连续存储'}"
            )
            self.sample_rate_summary_var.setText(f"FS：{meta.fs:g} Hz")
            self.channel_count_summary_var.setText(f"通道：{meta.channels}")
            self.duration_summary_var.setText(f"时长：{meta.rows / meta.fs:.3f} s")
            self.channel_var.setText(f"预览通道：ch{meta.channel_ids[0]}")
            self.filter_var.setText("滤波：当前未应用")
            self.alignment_var.setText("对齐：尚未完成")
            self._set_preprocess_data_state("数据状态：原始数据已加载，尚未进行重映射或滤波。")
            self.preprocessed_preview.set_source(self.source, defer_initial_draw=True)
            self._update_preprocess_comparison_preview()
            if is_preprocessed_h5:
                self._set_preprocess_data_state("数据状态：已加载预处理 H5；当前预览和后续分析均使用该处理后的完整数据。")
            if restored_preprocess_qc:
                self._refresh_bad_channel_review_table()
                self.preprocess_status.setText(
                    f"已恢复历史坏道判定：健康 {len(self.good_channel_ids)}，坏道 {len(self.bad_channel_ids)}；"
                    f"来源：{getattr(self, '_restored_qc_record_source', 'H5内嵌QC快照')}。"
                )
            self._show_loaded_preprocess_summary(meta, restored_preprocess_qc)
            if is_preprocessed_h5:
                self._restore_preprocess_processing_state(filename, meta)
            restored_lfp_state = self._restore_lfp_analysis_state(filename, meta)
            self._restore_spike_analysis_state(filename, meta)
            if restored_lfp_state:
                self.lfp_source = self.source
                self._set_preprocess_data_state(
                    self.preprocess_data_state.text()
                    + " " + self._format_lfp_analysis_state(self._loaded_lfp_analysis_state)
                )
            self.statusBar().showMessage("已加载 HDF5 元数据；预览仅按当前时间窗/通道切片读取。")
            self._update_file_banners()
            self._update_one_click_plan_summary()
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))

    def _apply_loaded_timing(self, meta) -> None:
        """Restore legacy BIN timing when any reloadable data source is opened."""
        self.bin_timing_metadata = dict(getattr(meta, "timing_metadata", {}) or {})
        delta_sec = float(self.bin_timing_metadata.get("deltaT1_sec", 0.0) or 0.0)
        if not np.isfinite(delta_sec):
            delta_sec = 0.0
        self.bin_timing_metadata["deltaT1_sec"] = delta_sec
        date = str(self.bin_timing_metadata.get("dt1_date", "")).strip()
        if date and isinstance(getattr(self, "date_var", None), QLineEdit):
            self.date_var.setText(date)
        if hasattr(self, "alignment_status"):
            if self.bin_timing_metadata:
                self.alignment_status.setText(
                    f"已自动读取 BIN 定时：deltaT1={delta_sec * 1000.0:.4f} ms"
                    + (f"；dt1 日期={date}" if date else "")
                    + "。时间对齐将自动使用该值。"
                )
            else:
                self.alignment_status.setText("未检测到 BIN deltaT1；时间对齐将按 0 s 校正。")
        if hasattr(self, "loaded_dt1_label"):
            raw = self.bin_timing_metadata.get("deltaT1")
            unit = str(self.bin_timing_metadata.get("deltaT1_unit", "")).strip() or "s"
            dt1 = str(self.bin_timing_metadata.get("dt1", "")).strip() or "未提供"
            raw_text = f"{float(raw):.6f} {unit}" if raw is not None and np.isfinite(float(raw)) else "未提供"
            self.loaded_dt1_label.setText(
                f"文件 dt1：{dt1}；文件 deltaT1：{raw_text}；对齐使用：{delta_sec:.9f} s（{delta_sec * 1000.0:.4f} ms）"
            )

    @staticmethod
    def _load_mat_source(path: Path) -> ArraySource:
        """Compatibility path for classic MATLAB files (non-v7.3 MAT)."""
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise ImportError("读取 MAT 需要 scipy。") from exc
        values = loadmat(path, squeeze_me=True)
        candidates = []
        for key in ("rawData512", "raw_data", "data", "eeg", "signal"):
            value = values.get(key)
            if isinstance(value, np.ndarray) and value.ndim == 2 and np.issubdtype(value.dtype, np.number):
                candidates.append((key, value))
        if not candidates:
            candidates = [
                (key, value) for key, value in values.items()
                if isinstance(value, np.ndarray) and value.ndim == 2 and np.issubdtype(value.dtype, np.number)
            ]
        if not candidates:
            raise ValueError("MAT 文件中没有二维数值脑电矩阵。")
        key, data = max(candidates, key=lambda item: item[1].size)
        if data.shape[0] < data.shape[1] and data.shape[0] <= 520:
            data = data.T
        fs_value = values.get("FS", values.get("fs", 6490.0))
        fs = float(np.asarray(fs_value).squeeze())
        if not np.isfinite(fs) or fs <= 0:
            fs = 6490.0
        ids_value = values.get("channel_ids", values.get("channel_numbers"))
        ids = None if ids_value is None else np.asarray(ids_value).reshape(-1)
        timing = _mat_timing_metadata(values)
        source = ArraySource(np.asarray(data, dtype=np.float32), fs, label=f"MAT:{key}", channel_ids=ids, timing_metadata=timing)
        source.metadata = source.metadata.__class__(
            path=path, dataset=key, rows=source.metadata.rows, channels=source.metadata.channels, fs=source.metadata.fs,
            chunks=None, scale_to_mv=1.0, time_offset=0.0, channel_ids=source.metadata.channel_ids,
            timing_metadata=timing,
        )
        return source

    def open_import_overview(self) -> None:
        """Open a read-only raw-data overview for the import page."""
        if not self.source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载 HDF5 数据文件。")
            return
        try:
            start_seconds, duration_seconds = self._import_time_window()
        except ValueError as exc:
            QMessageBox.warning(self, "时间窗无效", str(exc))
            return
        end_seconds = (
            self.source.metadata.rows / self.source.metadata.fs
            if self.preview_full_duration_check.isChecked()
            else start_seconds + duration_seconds
        )
        self._show_overview(
            self.source,
            start_seconds,
            end_seconds,
            full_duration=self.preview_full_duration_check.isChecked(),
            window_attr="_import_overview_window",
        )

    def open_overview(self) -> None:
        """Legacy public entry point for the import-page read-only overview."""
        self.open_import_overview()

    def open_preprocess_filter_overview(self) -> None:
        """Open the filter-channel selector on the current remapped source."""
        source = self.remapped_source if (
            self.remapped_source is not None and self.remapped_source.loaded
        ) else self.source
        if not source.loaded:
            QMessageBox.information(self, "尚未加载数据", "请先加载 HDF5 数据文件。")
            return
        try:
            selected_columns = self._selected_preprocess_columns(source)
        except ValueError as exc:
            QMessageBox.warning(self, "指定通道", str(exc))
            return
        selected_ids = () if selected_columns is None else np.asarray(
            source.metadata.channel_ids, dtype=np.int64
        )[selected_columns]
        selection_groups = None
        if self._bad_channel_check_completed:
            selection_groups = {
                "健康道": set(self.good_channel_ids),
                "坏道": set(self.bad_channel_ids),
            }
        overview = getattr(self, "_preprocess_filter_selector", None)
        if overview is not None:
            try:
                if overview.isVisible() and overview.source is source:
                    overview.raise_()
                    overview.activateWindow()
                    return
                if overview.isVisible():
                    overview.close()
            except RuntimeError:
                pass
        overview = PyQtGraphOverview(
            source,
            0.0,
            _recording_duration_seconds(source.metadata),
            full_duration=True,
            selection_mode=True,
            selected_channel_ids=selected_ids,
            selection_callback=self._set_preprocess_filter_channel_ids,
            display_channel_ids=None,
            selection_label="滤波",
            selection_groups=selection_groups,
        )
        overview.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        overview.showMaximized()
        self._preprocess_filter_selector = overview

    def _set_preprocess_filter_channel_ids(self, channel_ids) -> None:
        """Save the filter-channel selection without starting preprocessing."""
        selected_ids = sorted({int(channel) for channel in channel_ids})
        source = self.remapped_source if (
            self.remapped_source is not None and self.remapped_source.loaded
        ) else self.source
        available_ids = {int(channel) for channel in source.metadata.channel_ids}
        selected_ids = [channel for channel in selected_ids if channel in available_ids]
        if not selected_ids:
            QMessageBox.warning(self, "滤波通道", "所选通道不在当前导入数据中。")
            return
        self.preprocess_filter_channels_var.setText(",".join(map(str, selected_ids)))
        self.preprocess_status.setText(
            f"已选择 {len(selected_ids)} 个滤波通道；请点击“执行滤波”开始处理。"
        )

    def _show_overview(
        self, source: LazyH5Source, start_seconds: float, end_seconds: float,
        *, full_duration: bool = True, window_attr: str = "_overview_window",
    ) -> None:
        overview = getattr(self, window_attr, None)
        if overview is not None:
            try:
                if overview.isVisible() and overview.source is source:
                    overview.raise_()
                    overview.activateWindow()
                    return
                if overview.isVisible():
                    overview.close()
            except RuntimeError:
                # The Qt object was deleted after its previous window closed.
                pass
        overview = PyQtGraphOverview(
            source, start_seconds, end_seconds, full_duration=full_duration,
        )
        overview.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        overview.showMaximized()
        # Retain the Python reference after this slot returns.
        setattr(self, window_attr, overview)

    def _cleanup_processing_cache_on_close(self) -> None:
        """Stop background users of memmaps, then remove process/stale caches."""
        def is_running(worker) -> bool:
            try:
                return bool(worker.isRunning())
            except RuntimeError:
                return False

        workers = {}
        for value in self.__dict__.values():
            if isinstance(value, QThread):
                workers[id(value)] = value
        for worker, _label in getattr(self, "_shared_preprocess_tasks", []):
            if isinstance(worker, QThread):
                workers[id(worker)] = worker
        running = [worker for worker in workers.values() if is_running(worker)]
        for worker in running:
            if hasattr(worker, "set_paused"):
                try:
                    worker.set_paused(False)
                except (RuntimeError, TypeError):
                    pass
            if hasattr(worker, "cancel"):
                try:
                    worker.cancel()
                except (RuntimeError, TypeError):
                    pass
            try:
                worker.requestInterruption()
            except RuntimeError:
                pass
        deadline = perf_counter() + 15.0
        for worker in running:
            remaining_ms = max(0, int((deadline - perf_counter()) * 1000.0))
            if remaining_ms <= 0:
                break
            try:
                worker.wait(remaining_ms)
            except RuntimeError:
                pass
        # Cooperative preprocessing workers normally stop above. If an
        # unrelated analysis thread is still finishing, the atexit hook in
        # qt_data_model retries deletion after process teardown.
        if not any(is_running(worker) for worker in workers.values()):
            cleanup_process_cache_files(include_stale=True)

    def closeEvent(self, event) -> None:
        """Let the operator save or discard this launch's single audit row."""
        if getattr(self, "_audit_session_closed", False):
            super().closeEvent(event)
            return
        answer = QMessageBox.question(
            self,
            "保存本次操作记录",
            f"是否保存操作者“{self.operator_name}”本次 GUI 的全部操作记录？\n\n"
            "选择“是”：保留本次会话的一行记录。\n"
            "选择“否”：删除本次会话记录。\n"
            "选择“取消”：返回工作台，不关闭 GUI。",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        store = getattr(self, "operation_audit_store", None)
        try:
            if answer == QMessageBox.StandardButton.Yes:
                if store is None:
                    raise sqlite3.OperationalError(
                        self.operation_audit_error or "操作记录数据库不可用"
                    )
                store.append(
                    self.operator_name,
                    self.operation_session_id,
                    "session_closed",
                    self._current_audit_data_path(),
                    "程序关闭，本次 GUI 操作会话已保存",
                    self._audit_state_snapshot(),
                )
            elif store is not None:
                store.discard_session(self.operation_session_id)
            sync_service = getattr(self, "feishu_audit_sync", None)
            if sync_service is not None:
                sync_service.notify()
        except (OSError, sqlite3.Error) as exc:
            QMessageBox.critical(
                self,
                "操作记录处理失败",
                f"无法{'保存' if answer == QMessageBox.StandardButton.Yes else '删除'}本次操作记录：\n{exc}\n\nGUI 暂未关闭，请重试。",
            )
            event.ignore()
            return
        sync_service = getattr(self, "feishu_audit_sync", None)
        if sync_service is not None:
            # Give the daemon a short final opportunity; an offline item stays
            # in SQLite and is retried automatically at the next launch.
            sync_service.stop(timeout=4.0)
        self._cleanup_processing_cache_on_close()
        self._audit_session_closed = True
        super().closeEvent(event)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    # Keep a live reference: QApplication does not take Python ownership of
    # an installed event filter, so a local-only object could be collected.
    app._disable_value_control_wheel_filter = DisableValueControlWheelFilter(app)
    app.installEventFilter(app._disable_value_control_wheel_filter)
    operator_name = ""
    while not operator_name:
        entered, accepted = QInputDialog.getText(
            None, "操作者登录", "请输入你的姓名（将写入数据操作记录）："
        )
        if not accepted:
            return 0
        operator_name = str(entered).strip()
        if not operator_name:
            QMessageBox.information(None, "姓名不能为空", "请输入姓名后再进入工作台。")
    window = QtAnalysisGUI(operator_name=operator_name)
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
