"""Opt-in page 2/3 controller; no shell, eval, synthetic clicks or new algorithms.

The stdio adapter and private loopback transport are shared with AnimalStart.
Every action runs on the GUI thread and calls the same handlers as the GUI.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np
try:
    from PyQt6 import QtCore, QtWidgets
except ImportError:
    from PyQt5 import QtCore, QtWidgets

from animal_peak_rpc import BridgeError
from animal_peak_control import safe_json, _only, _integer


def pipeline_message_box(base):
    """Keep human dialogs unchanged; remote job questions always fail closed."""
    class MessageBox(base):
        @staticmethod
        def _message(kind, parent, title, text, *args, **kwargs):
            messages = getattr(parent, "_pipeline_mcp_messages", None)
            if messages is None:
                return getattr(base, kind)(parent, title, text, *args, **kwargs)
            messages.append(dict(level=kind, title=str(title), message=str(text)))
            return base.StandardButton.No

        @staticmethod
        def information(parent, title, text, *args, **kwargs):
            return MessageBox._message("information", parent, title, text, *args, **kwargs)

        @staticmethod
        def warning(parent, title, text, *args, **kwargs):
            return MessageBox._message("warning", parent, title, text, *args, **kwargs)

        @staticmethod
        def critical(parent, title, text, *args, **kwargs):
            return MessageBox._message("critical", parent, title, text, *args, **kwargs)

        @staticmethod
        def question(parent, title, text, *args, **kwargs):
            return MessageBox._message("question", parent, title, text, *args, **kwargs)
    return MessageBox


# (widget attribute, input type, inclusive minimum, inclusive maximum).
def field(widget, kind="float", low=-1e9, high=1e9):
    return (widget, kind, low, high)


FIELDS = {
    "preprocess": {
        "remap_path": field("remap_edit", "path"),
        "channels": field("preprocess_filter_channels_var", "channels"),
        "filter_mode": field("filter_mode", "enum"),
        "low_hz": field("preprocess_filter_low_var", low=0),
        "high_hz": field("preprocess_filter_high_var", low=0),
        "hp_order": field("preprocess_filter_hp_order_var", "int", 1, 20),
        "lp_order": field("preprocess_filter_lp_order_var", "int", 1, 20),
        "notch": field("manual_filter_notch_var", "bool"),
        "notch_harmonics": field("preprocess_filter_notch_harmonics_var", "int", 1, 100),
        "one_click_notch": field("one_click_preprocess_check", "bool"),
        "one_click_quality": field("one_click_quality_check", "bool"),
        "one_click_car": field("one_click_car_check", "bool"),
    },
    "quality": {
        "parallel": field("bad_parallel_var", "bool"),
        "workers": field("bad_workers_var", "int", 0, 256),
        "saturation": field("bad_fast_artifact_check_var", "bool"),
        "saturation_width_percent": field("bad_saturation_width_percent_var", low=0, high=100),
        "saturation_ratio_percent": field("bad_saturation_ratio_threshold_var", low=0, high=100),
        "concentration": field("bad_high_frequency_noise_check_var", "bool"),
        "concentration_target": field("bad_high_frequency_noise_target_var"),
        "concentration_tolerance": field("bad_high_frequency_noise_tolerance_var", low=0),
        "concentration_ratio_percent": field("bad_high_frequency_noise_ratio_threshold_var", low=0, high=100),
    },
    "ica": {
        "enabled": field("motion_ica_enable_var", "bool"),
        "components": field("motion_ica_components_var", "int", 1, 4096),
        "exclude": field("motion_ica_exclude_var", "indices"),
        "low_hz": field("motion_ica_low_var", low=0),
        "high_hz": field("motion_ica_high_var", low=0),
        "decim": field("motion_ica_decim_var", "int", 1, 10000),
        "max_iter": field("motion_ica_max_iter_var", "int", 1, 100000),
    },
    "stream": {
        **{f"{branch}_{key}": field(f"stream_{branch}_{key}_var", low=0)
           for branch in ("lfp", "spike") for key in ("low", "high", "overlap")},
        **{f"{branch}_notch": field(f"stream_{branch}_notch_var", "bool") for branch in ("lfp", "spike")},
        "lfp_car": field("stream_lfp_car_var", "bool"),
        "notch_frequency": field("stream_notch_frequency_var", low=.001),
        "notch_q": field("stream_notch_q_var", low=.001),
        "notch_harmonics": field("stream_notch_harmonics_var", "int", 1, 100),
        "max_workers": field("stream_max_workers_var", "int", 1, 256),
        "memory_budget_gb": field("stream_memory_budget_gb_var", low=.001, high=1024),
    },
    "alignment": {
        "log_path": field("alignment_log_edit", "path"),
        "events_path": field("alignment_event_edit", "path"),
        "detection_path": field("alignment_detection_edit", "path"),
        "task_mode": field("alignment_task_mode_var", "enum"),
    },
    "task": {
        **{name: field(f"{name}_var") for name in (
            "epoch_start_ms", "epoch_end_ms", "baseline_start_ms", "baseline_end_ms",
            "response_start_ms", "response_end_ms")},
        **{name: field(f"{name}_var", "int", 0, 100000) for name in (
            "n_stim_types", "trials_per_stim", "compare_trial_a", "compare_trial_b")},
        "stimtags": field("stimtag_var", "indices"),
        "page_size": field("tvep_page_size_var", "int", 1, 1000),
        "aggregate": field("task_response_aggregate_var", "enum"),
        "smooth": field("smooth_signal_var", "bool"),
        "smooth_window_sec": field("smooth_window_size_var", low=.000001),
        "zscore": field("zscoredata_var", "bool"),
    },
    "preview": {
        "start_sec": field("alignment_preview_start_spin", low=0, high=86400),
        "duration_sec": field("alignment_preview_duration_spin", low=.001, high=86400),
        "full_duration": field("alignment_preview_full_duration_check", "bool"),
    },
}

ACTIONS = {
    "load_h5": ("path", "confirm"),
    "remap": ("confirm", "repeat_confirm"),
    "filter": ("confirm", "repeat_confirm"),
    "one_click": ("confirm", "repeat_confirm"),
    "quality_check": ("confirm",),
    "saturation_check": ("confirm",),
    "concentration_check": ("confirm",),
    "car": ("confirm", "repeat_confirm"),
    "ica": ("channels", "confirm", "repeat_confirm"),
    "clear_ica": ("confirm",),
    "review_channels": ("channels", "classification", "confirm"),
    "autofill_alignment": (),
    "align": ("segment_index", "confirm"),
    "refresh_alignment": (),
    "refresh_preprocess": ("channel", "start_sec", "end_sec", "full_duration", "original"),
    "open_letter": (),
    "open_flash": (),
    "open_animalstart": (),
    "export_preprocessed": ("path", "confirm"),
    "stream_export": ("confirm",),
    "copy_plot": ("plot",),
    "pause": ("job_id",),
    "resume": ("job_id",),
    "cancel": ("job_id",),
}
CONFIRMED = {name for name, options in ACTIONS.items() if "confirm" in options}
WORKER_ACTIONS = {"remap", "filter", "one_click", "quality_check", "saturation_check",
                  "concentration_check", "car", "ica", "stream_export"}


class PipelineController(QtCore.QObject):
    def __init__(self, window):
        super().__init__(window)
        self.dialog = window  # shared transport owner
        self.revision = 0
        self._fingerprint = None
        self.jobs = {}
        self.active_job_id = None
        self._workers = []
        self._settled_workers = set()
        self.repeat_confirmed = False
        self.command_active = False
        self.alignment_signature = None
        self.alignment_preview_signature = None
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._poll_job)

    def session_metadata(self):
        meta = getattr(self.dialog.active_source, "metadata", None)
        return dict(session_kind="pipeline", source_name=Path(meta.path).name if meta else "未加载数据")

    def _widget(self, spec):
        return getattr(self.dialog, spec[0])

    def _read(self, spec):
        w = self._widget(spec)
        if isinstance(w, QtWidgets.QComboBox):
            return w.currentData() if w.currentData() is not None else w.currentText()
        if isinstance(w, QtWidgets.QAbstractButton):
            return w.isChecked()
        if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            return w.value()
        text = w.text()
        try:
            if spec[1] == "int": return int(text)
            if spec[1] == "float": return float(text)
        except ValueError:
            pass  # Human edits may be incomplete; report rather than mutate.
        return text

    def _controls(self):
        return {group: {key: self._read(spec) for key, spec in fields.items()}
                for group, fields in FIELDS.items()}

    @staticmethod
    def _source(source):
        meta = getattr(source, "metadata", None)
        if meta is None: return None
        return dict(path=str(meta.path), rows=int(meta.rows), fs=float(meta.fs),
                    channel_ids=list(map(int, meta.channel_ids)), time_offset_sec=float(meta.time_offset))

    def _running_workers(self):
        return [v for v in vars(self.dialog).values() if isinstance(v, QtCore.QThread) and v.isRunning()]

    def _busy(self):
        return bool(self.active_job_id or self._running_workers() or
                    getattr(self.dialog, "_one_click_pipeline", None) or
                    getattr(self.dialog, "_selected_analysis_batch", None))

    def alignment_key(self):
        d = self.dialog
        stamps = []
        for edit in d._alignment_input_edits().values():
            try:
                stat = Path(edit.text()).stat()
                stamps.append((stat.st_size, stat.st_mtime_ns))
            except OSError:
                stamps.append(None)
        return (id(d.active_source), json.dumps(self._controls()["alignment"], sort_keys=True),
                tuple(sorted(getattr(d, "alignment_selected_ids", []))),
                bool(getattr(d, "_alignment_use_all_healthy", True)),
                json.dumps(safe_json(d.bin_timing_metadata), sort_keys=True), stamps,
                bool(d._bad_channel_check_completed), tuple(sorted(d.good_channel_ids)), tuple(sorted(d.bad_channel_ids)))

    def mark_alignment_applied(self):
        self.alignment_signature = self.alignment_key()

    def _alignment_stale(self):
        return self.alignment_signature != self.alignment_key()

    def alignment_preview_key(self):
        return (id(self.dialog.active_source), id(getattr(self.dialog, "stim_markers", None)),
                self._preview_channel(), json.dumps(self._controls()["preview"], sort_keys=True))

    @staticmethod
    def preprocess_preview_key(preview):
        return (id(preview.source), preview.channel_spin.value(), preview.start_spin.value(),
                preview.end_spin.value(), preview.full_duration_check.isChecked(), preview.linewidth_spin.value())

    def state(self):
        d = self.dialog
        markers = np.asarray(getattr(d, "stim_markers", []))
        result = dict(session_kind="pipeline", pages=["preprocess", "alignment"],
            source=self._source(d.active_source), original_source=self._source(d.source),
            controls=self._controls(), busy=self._busy(), active_job_id=self.active_job_id,
            page="alignment" if d._page_stack.currentIndex() == 2 else
                 "preprocess" if d._page_stack.currentIndex() == 1 else "other",
            preprocess=dict(status=d.preprocess_status.text(), progress=d.preprocess_progress.value(),
                plan=d.one_click_plan_summary.text(), data_state=d.preprocess_data_state.text(),
                filtered_channel_ids=sorted(d._preprocess_filtered_channel_ids),
                notched_channel_ids=sorted(d._preprocess_notched_channel_ids),
                qc_completed=d._bad_channel_check_completed, good_channel_ids=sorted(d.good_channel_ids),
                bad_channel_ids=sorted(d.bad_channel_ids), manual_overrides=d.manual_channel_overrides),
            alignment=dict(status=d.alignment_status.text(), report=d.alignment_results.toPlainText(),
                stale=self._alignment_stale(), marker_count=len(markers),
                task_mode=getattr(d, "stim_marker_mode", None),
                timing=getattr(d, "timing_info", {}),
                selected_channel_ids=sorted(getattr(d, "alignment_selected_ids", [])),
                use_all_healthy=getattr(d, "_alignment_use_all_healthy", True),
                autofill=getattr(d, "_alignment_autofill_result", None),
                manual_inputs=sorted(d._alignment_manual_inputs)),
            preview_channel=self._preview_channel(),
            previews={"alignment": dict(stale=self.alignment_preview_signature != self.alignment_preview_key()),
                      **{name: dict(channel=int(p.source.metadata.channel_ids[p.channel_spin.value()-1]) if p.source.loaded else None,
                                    start_sec=p.start_spin.value(), end_sec=p.end_spin.value(),
                                    full_duration=p.full_duration_check.isChecked(),
                                    stale=getattr(p, "_mcp_render_key", None) != self.preprocess_preview_key(p))
                         for name, p in (("preprocess", d.preprocessed_preview), ("original", d.original_preprocess_preview))}},
            last_jobs=[{k: v for k, v in j.items() if k not in ("messages", "options", "input_key")}
                       for j in list(self.jobs.values())[-5:]])
        # Do not bump revision simply for polling progress. Also detect human
        # changes to source identity, marker contents, QC and controls.
        identity = [result["controls"], id(d.active_source), id(d.preprocessed_source),
            result["page"], result["preview_channel"], result["preprocess"]["good_channel_ids"],
            result["previews"],
            result["preprocess"]["bad_channel_ids"], d.manual_channel_overrides,
            result["alignment"]["selected_channel_ids"], self.alignment_signature, self.alignment_key(),
            hashlib.sha256(markers.tobytes()).hexdigest(),
            [(j["job_id"], j["status"]) for j in self.jobs.values()]]
        fingerprint = json.dumps(safe_json(identity), sort_keys=True, allow_nan=False)
        if fingerprint != self._fingerprint:
            self.revision += 1
            self._fingerprint = fingerprint
        return safe_json(result)

    def _preview_channel(self):
        d = self.dialog
        column = d.alignment_preview_channel_box.currentData()
        meta = getattr(d.active_source, "metadata", None)
        return (int(meta.channel_ids[column]) if meta and isinstance(column, int)
                and 0 <= column < len(meta.channel_ids) else None)

    def describe(self):
        groups = {}
        for group, fields in FIELDS.items():
            groups[group] = {}
            for name, spec in fields.items():
                w = self._widget(spec)
                details = dict(type=spec[1], enabled=w.isEnabled())
                if spec[1] in ("float", "int"): details.update(min=spec[2], max=spec[3])
                if isinstance(w, QtWidgets.QComboBox):
                    details["choices"] = [w.itemData(i) if w.itemData(i) is not None else w.itemText(i)
                                          for i in range(w.count())]
                groups[group][name] = details
        return dict(session_kind="pipeline", groups=groups, actions=ACTIONS,
            confirmation_required=sorted(CONFIRMED), tables=["markers", "annotations", "segments", "qc"],
            plots=["alignment", "preprocess", "original"],
            navigation=dict(page=["preprocess", "alignment"], channel="actual physical ID",
                            channels="alignment healthy IDs; null=all healthy"),
            notes=["set_parameters only stages inputs; execute explicitly applies them.",
                   "align.segment_index is ZERO-based; inspect segments before choosing.",
                   "No raw-file overwrite. Export path must not exist; paired CSV must not exist.",
                   "stream_export uses the existing processed directory beside loaded H5; complete QC required.",
                   "repeat_confirm=true separately authorizes repeating operations recorded in imported H5.",
                   "MCP checkbox revocation does not cancel already started jobs."])

    def dispatch(self, operation, arguments=None, expected_revision=None):
        try:
            if not self.dialog.pipeline_mcp_check.isChecked():
                raise BridgeError("disabled", "请在主窗口开启“允许 MCP 控制第 2 / 3 页”。")
            if not isinstance(arguments, dict): raise ValueError("arguments必须是对象。")
            self.state()
            reads = {"get_state", "describe", "get_jobs", "read_table", "get_image", "read_waveform"}
            writes = {"set_parameters", "navigate", "execute"}
            if operation not in reads | writes: raise ValueError("主窗口不支持此操作；请先 describe_controls。")
            if operation in writes:
                if type(expected_revision) is not int or expected_revision != self.revision:
                    raise BridgeError("stale_revision", "界面已变化；重新读取 get_state 后再提交，不要自动重试旧命令。")
                if self.command_active: raise BridgeError("busy", "正在提交另一操作。")
                if QtWidgets.QApplication.activeModalWidget() is not None:
                    raise BridgeError("modal_dialog", "请先处理当前 GUI 模态对话框。")
                management = operation == "execute" and arguments.get("action") in {"pause", "resume", "cancel"}
                if self._busy() and not management: raise BridgeError("busy", "已有任务执行中；请查询 get_jobs。")
            self.command_active = True
            try:
                if operation == "get_state":
                    _only(arguments, ()); result = self.state()
                elif operation == "describe":
                    _only(arguments, ()); result = self.describe()
                else:
                    handler = {"set_parameters": self.set_parameters, "navigate": self.navigate,
                        "execute": self.execute, "get_jobs": self.get_jobs, "read_table": self.read_table,
                        "get_image": self.get_image, "read_waveform": self.read_waveform}[operation]
                    result = handler(arguments)
            finally:
                self.command_active = False
            self.state()
            return dict(ok=True, revision=self.revision, result=safe_json(result))
        except Exception as exc:
            return dict(ok=False, revision=self.revision,
                        error=dict(code=getattr(exc, "code", "invalid_parameters"), message=str(exc)))

    def _validate(self, spec, value):
        w, kind = self._widget(spec), spec[1]
        if kind == "bool":
            if type(value) is not bool: raise ValueError("必须是布尔值。")
        elif kind == "enum":
            choices = [w.itemData(i) if w.itemData(i) is not None else w.itemText(i) for i in range(w.count())]
            if value not in choices: raise ValueError(f"枚举必须是{choices}。")
        elif kind in ("float", "int"):
            if type(value) not in (int, float) or not math.isfinite(value) or not spec[2] <= value <= spec[3]:
                raise ValueError(f"数字必须在{spec[2]}～{spec[3]}之间。")
            if kind == "int" and type(value) is not int: raise ValueError("必须是整数。")
            if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                if not w.minimum() <= value <= w.maximum(): raise ValueError("超出当前数据允许的控件范围。")
                if round(value, w.decimals()) != value: raise ValueError("小数精度超出控件显示精度。")
        elif kind == "path":
            if not isinstance(value, str) or len(value) > 4096: raise ValueError("路径必须是字符串。")
            if value and (not Path(value).is_absolute() or not Path(value).is_file()):
                raise ValueError("输入必须是工作站上的现有文件绝对路径；留空可清除。")
        elif kind in ("channels", "indices"):
            if value is None and kind == "channels": return ""
            if not isinstance(value, list) or len(value) > 4096 or any(type(v) is not int or v < 0 for v in value):
                raise ValueError("请传入非负整数列表；channels=null表示全部。")
            if len(set(value)) != len(value): raise ValueError("通道/索引不能重复。")
            if kind == "channels": self._channel_ids(value)
            return ",".join(map(str, value))
        return value

    def _assign(self, spec, value):
        w = self._widget(spec)
        if isinstance(w, QtWidgets.QComboBox):
            index = w.findData(value)
            w.setCurrentIndex(index if index >= 0 else w.findText(value))
        elif isinstance(w, QtWidgets.QAbstractButton): w.setChecked(value)
        elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)): w.setValue(value)
        else: w.setText(str(value))

    def set_parameters(self, args):
        _only(args, ("group", "values"))
        group, values = args.get("group"), args.get("values")
        if group not in FIELDS or not isinstance(values, dict) or not values: raise ValueError("未知参数组或空参数。")
        _only(values, FIELDS[group])
        validated = {key: self._validate(FIELDS[group][key], value) for key, value in values.items()}
        merged = {**self._controls()[group], **validated}
        if group == "preprocess":
            mode = merged["filter_mode"]
            if mode in ("highpass", "bandpass") and merged["low_hz"] <= 0: raise ValueError("低截止必须>0。")
            if mode in ("lowpass", "bandpass") and merged["high_hz"] <= 0: raise ValueError("高截止必须>0。")
            if mode == "bandpass" and merged["low_hz"] >= merged["high_hz"]: raise ValueError("低截止必须小于高截止。")
        if group in ("task", "ica"):
            pairs = [("low_hz", "high_hz")] if group == "ica" else [
                ("epoch_start_ms", "epoch_end_ms"), ("baseline_start_ms", "baseline_end_ms"),
                ("response_start_ms", "response_end_ms")]
            for low, high in pairs:
                if merged[low] >= merged[high]: raise ValueError(f"{low}必须小于{high}。")
        for key, value in validated.items():
            self._assign(FIELDS[group][key], value)
            if group == "alignment" and key.endswith("_path"):
                self.dialog._mark_alignment_input_manual({"log_path": "log", "events_path": "events",
                                                          "detection_path": "detection"}[key])
        return dict(controls=self._controls()[group], applied=False)

    def _channel_ids(self, ids, *, healthy=False, allow_empty=True):
        if not isinstance(ids, list) or len(ids) > 4096 or any(type(i) is not int for i in ids):
            raise ValueError("channels必须是物理通道整数列表。")
        if len(set(ids)) != len(ids) or (not allow_empty and not ids): raise ValueError("通道列表为空或重复。")
        meta = getattr(self.dialog.active_source, "metadata", None)
        available = set(map(int, meta.channel_ids)) if meta else set()
        if healthy: available = self.dialog._alignment_healthy_channel_ids(self.dialog.active_source)
        if set(ids) - available: raise ValueError(f"这些通道不在当前可用范围：{sorted(set(ids)-available)}")
        return ids

    def navigate(self, args):
        _only(args, ("page", "channel", "channels"))
        d = self.dialog
        page = args.get("page")
        if page is not None and page not in ("preprocess", "alignment"): raise ValueError("page只能是preprocess/alignment。")
        if "channels" in args:
            ids = args["channels"]
            if ids is None: ids = sorted(d._alignment_healthy_channel_ids(d.active_source))
            self._channel_ids(ids, healthy=True, allow_empty=False)
        if "channel" in args: self._channel_ids([args["channel"]], healthy=True)
        if page is not None: d._activate_page(1 if page == "preprocess" else 2)
        if "channels" in args: d._set_alignment_selected_channels(ids, use_all_healthy=args["channels"] is None)
        if "channel" in args:
            d._sync_preview_channel_box(d.alignment_preview_channel_box, d.active_source,
                                       allowed_channel_ids=d._alignment_healthy_channel_ids(d.active_source))
            column = list(d.active_source.metadata.channel_ids).index(args["channel"])
            d.alignment_preview_channel_box.setCurrentIndex(d.alignment_preview_channel_box.findData(column))
        return dict(page=page, channel=self._preview_channel(),
                    channels=sorted(getattr(d, "alignment_selected_ids", [])))

    def execute(self, args):
        action = args.get("action")
        if action not in ACTIONS: raise ValueError("未知主窗口动作；请先 describe_controls。")
        _only(args, ("action", *ACTIONS[action]))
        if action in CONFIRMED and args.get("confirm") is not True:
            raise BridgeError("confirmation_required", "此动作会替换/处理/导出数据或QC；须用户授权并传入confirm=true。")
        for key in ("confirm", "repeat_confirm"):
            if key in args and type(args[key]) is not bool: raise ValueError(key+"必须是布尔值。")
        if action in {"pause", "resume", "cancel"}: return self._manage(action, args)
        if action == "load_h5":
            path = self._input_path(args.get("path"), (".h5", ".hdf5"))
            from qt_data_model import LazyH5Source
            LazyH5Source().open(path)  # validate metadata before replacing the GUI source
        if action == "load_h5":
            child = getattr(self.dialog, "_animal_peak_test_dialog", None)
            if child is not None and child.isVisible():
                raise ValueError("请先保存并关闭现有AnimalStart测试窗口，再替换它的数据源。")
        if action == "export_preprocessed": self._export_path(args.get("path"))
        if action not in {"load_h5", "autofill_alignment"} and not self.dialog.active_source.loaded:
            raise ValueError("请先载入数据。")
        if action == "align":
            self.dialog._alignment_channel_ids(self.dialog.active_source)
            self._input_path(self.dialog.alignment_log_edit.text())
            self._input_path(self.dialog.alignment_event_edit.text())
            if "segment_index" in args: _integer(args["segment_index"], 0, 10000, "segment_index")
        if action in {"open_letter", "open_flash", "open_animalstart"} and self._alignment_stale():
            raise ValueError("对齐尚未计算或输入已变化；先重新align。")
        if action == "ica": self._channel_ids(args.get("channels"), allow_empty=False)
        if action == "review_channels":
            self._channel_ids(args.get("channels"), allow_empty=False)
            if args.get("classification") not in ("good", "bad", "auto"): raise ValueError("classification=good/bad/auto。")
            if not self.dialog._bad_channel_check_completed or set(args["channels"])-set(self.dialog._bad_channel_result_channel_ids):
                raise ValueError("先完成所选通道的QC，不能凭空标记未检查通道。")
        if action == "stream_export":
            d = self.dialog
            available = set(map(int, d._dual_stream_input_source().metadata.channel_ids))
            if not d._bad_channel_check_completed or set(d._bad_channel_result_channel_ids) != available:
                raise ValueError("流式导出需完整QC；MCP不允许静默跳过QC。")
        job = self._new_job(action, args, origin="mcp")
        job["input_key"] = self._input_key()
        self.dialog._pipeline_mcp_messages = job["messages"]
        self.repeat_confirmed = args.get("repeat_confirm", False)
        QtCore.QTimer.singleShot(0, lambda: self._run_action(job))
        return dict(job_id=job["job_id"], status="running")

    def _input_key(self):
        return json.dumps(safe_json([self._controls(), id(self.dialog.active_source), self.alignment_key()]), sort_keys=True)

    @staticmethod
    def _input_path(value, suffixes=None):
        if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_file():
            raise ValueError("需要工作站上现有文件的绝对路径。")
        if suffixes and Path(value).suffix.lower() not in suffixes: raise ValueError("文件类型不正确。")
        return Path(value).resolve()

    def _export_path(self, value):
        if not isinstance(value, str) or not Path(value).is_absolute(): raise ValueError("需要新的导出绝对路径。")
        path = Path(value).resolve()
        if path.suffix.lower() not in (".h5", ".hdf5"): raise ValueError("导出扩展名需为.h5/.hdf5。")
        if not path.parent.is_dir(): raise ValueError("导出父目录必须已存在。")
        if path.exists() or path.with_suffix(".csv").exists(): raise ValueError("目标H5或配对CSV已存在；MCP不覆盖。")
        for source in (self.dialog.source, self.dialog.active_source, self.dialog.preprocessed_source):
            meta = getattr(source, "metadata", None)
            if meta and path == Path(meta.path).resolve(): raise ValueError("禁止覆盖数据源。")
        return path

    def _new_job(self, action, options, *, origin):
        job = dict(job_id=str(uuid.uuid4()), action=action, status="running", origin=origin,
                   progress=0, messages=[], error=None, result={}, options=dict(options), started=False)
        self.jobs[job["job_id"]] = job
        self.active_job_id = job["job_id"]
        self._workers = []
        self._settled_workers = set()
        while len(self.jobs) > 100: self.jobs.pop(next(iter(self.jobs)))
        self.timer.start()
        return job

    def track_worker(self, worker, label):
        if self.active_job_id is None: self._new_job(label, {}, origin="gui")["started"] = True
        if worker in self._workers: return
        job = self.jobs[self.active_job_id]
        self._workers.append(worker)
        worker.finished.connect(lambda w=worker: self._settled_workers.add(id(w)))
        if hasattr(worker, "failed"):
            worker.failed.connect(lambda message, j=job: j.update(error=str(message)))
        if hasattr(worker, "cancelled"):
            worker.cancelled.connect(lambda j=job: j.update(cancelled=True))
        if hasattr(worker, "completed"):
            worker.completed.connect(lambda *values, j=job: self._worker_result(j, values))
        if hasattr(worker, "progress"):
            worker.progress.connect(lambda value, text, j=job: j.update(progress=float(value), message=str(text)))

    @staticmethod
    def _worker_result(job, values):
        # Only export paths, not full arrays or potentially huge analysis outputs.
        for value in values:
            if isinstance(value, dict):
                paths = {key: str(v) for key, v in value.items()
                         if key in ("lfp", "spike", "manifest", "output_directory") and isinstance(v, (str, Path))}
                job["result"].update(paths)

    def _run_action(self, job):
        d, args, action = self.dialog, job["options"], job["action"]
        job["started"] = True
        try:
            if not d.pipeline_mcp_check.isChecked(): raise ValueError("提交后MCP授权已撤销；未开始执行。")
            if self._input_key() != job["input_key"]: raise ValueError("提交后界面输入已改变；本次未执行，请重新读取状态。")
            if action == "load_h5":
                self._input_path(args["path"], (".h5", ".hdf5"))
                d.bin_path_edit.clear(); d.path_edit.setText(args["path"])
                d._load_file_use_full_range_once = True
                try: d.load_file()
                finally: d._load_file_use_full_range_once = False
                self.alignment_signature = None
            elif action == "remap": d.run_preprocess(force_filter_off=True)
            elif action == "filter": d.run_preprocess()
            elif action == "one_click":
                # A wholly cached chain may finish synchronously with no worker.
                d._mcp_one_click_completed = False
                d.run_one_click_preprocess()
            elif action == "quality_check": d.run_selected_analysis()
            elif action == "saturation_check": d.run_fast_artifact_check()
            elif action == "concentration_check": d.run_high_frequency_noise_check()
            elif action == "car": d.run_leave_one_out_median_car()
            elif action == "ica":
                if not d._confirm_repeated_preprocessing({"ica"}, "ICA"): raise ValueError("未确认重复ICA。")
                d._run_ica_for_channel_ids(d.preprocessed_source or d.remapped_source or d.active_source, args["channels"])
            elif action == "clear_ica": d.clear_ica()
            elif action == "review_channels":
                for channel in args["channels"]:
                    if args["classification"] == "auto": d.manual_channel_overrides.pop(channel, None)
                    else: d.manual_channel_overrides[channel] = args["classification"]
                if args["classification"] == "good":
                    d.lfp_selected_ids.update(args["channels"]); d.spike_selected_ids.update(args["channels"])
                elif args["classification"] == "bad":
                    d.lfp_selected_ids.difference_update(args["channels"]); d.spike_selected_ids.difference_update(args["channels"])
                d._apply_channel_review_overrides(); d._refresh_bad_channel_review_table()
                d.apply_car_button.setEnabled(len(d.good_channel_ids) >= 2)
                d._update_file_banners()
            elif action == "autofill_alignment": d._retry_alignment_autofill()
            elif action == "align":
                names = ("alignment_selected_ids", "letter_csv_annotations", "timing_info", "stim_markers",
                    "stim_marker_mode", "_task_marker_source", "behavior_matches", "behavior_detection_path",
                    "behavior_detection_events", "behavior_detection_frame_rate", "behavior_detection_recording_time_sec",
                    "behavior_detection_video_eeg_offset_sec")
                previous = {name: getattr(d, name) for name in names if hasattr(d, name)}
                signature = self.alignment_signature
                d.run_alignment(segment_index=args.get("segment_index"))
                if self._alignment_stale() or any(m["level"] == "critical" for m in job["messages"]):
                    for name in names:
                        if name in previous: setattr(d, name, previous[name])
                        elif hasattr(d, name): delattr(d, name)
                    self.alignment_signature = signature
                    raise ValueError(d.alignment_status.text())
            elif action == "refresh_alignment": d.refresh_alignment_channel_preview()
            elif action == "refresh_preprocess": self._refresh_preprocess(args)
            elif action in ("open_letter", "open_flash"):
                if action == "open_letter": d._run_alignment_letter_browser()
                else: d.run_task_epoch_analysis("lfp", "Flash", lfp_task_metrics=False)
                worker = getattr(d, "_lfp_task_worker", None)
                if worker not in self._workers: raise ValueError("未启动任务分析；检查任务模式与参数。")
            elif action == "open_animalstart": d.open_animal_peak_test()
            elif action == "export_preprocessed":
                path = self._export_path(args["path"])
                source = d.preprocessed_source
                if source is None or not source.loaded: raise ValueError("没有可导出的预处理结果。")
                csv = d._write_preprocessed_h5(path, source)
                job["result"] = dict(h5=str(path), csv=str(csv))
            elif action == "stream_export": d.run_dual_branch_stream_export()
            elif action == "copy_plot": QtWidgets.QApplication.clipboard().setPixmap(self._capture(args.get("plot", "alignment")))
            if action in WORKER_ACTIONS and not self._workers:
                # Cache hits are valid only if the handler explicitly reports completion.
                text = d.preprocess_status.text()
                cached_chain = action == "one_click" and getattr(d, "_mcp_one_click_completed", False)
                if "缓存" not in text and not cached_chain:
                    raise ValueError(job["messages"][-1]["message"] if job["messages"] else "未启动处理任务；"+text)
            if not self._workers and job["messages"]:
                # A handler may return after a validation dialog. Never report that as success.
                if action != "align":
                    raise ValueError(job["messages"][-1]["message"])
        except Exception as exc:
            job["error"] = str(exc)
        self._poll_job()

    def _poll_job(self):
        if self.active_job_id is None: self.timer.stop(); return
        job = self.jobs[self.active_job_id]
        if not job["started"]: return
        d = self.dialog
        if job["action"] not in ("open_letter", "open_flash"):
            job["progress"] = d.preprocess_progress.value()
        if any(w.isRunning() or id(w) not in self._settled_workers for w in self._workers) or self._running_workers(): return
        if not job["error"] and (getattr(d, "_one_click_pipeline", None) or
                                  getattr(d, "_selected_analysis_batch", None)): return
        failures = [m["message"] for m in job["messages"] if m["level"] in ("critical", "question")]
        job["error"] = job["error"] or (failures[-1] if failures else None)
        if (job["origin"] == "mcp" and job["action"] == "one_click" and not job.get("cancelled")
                and not getattr(d, "_mcp_one_click_completed", False)):
            job["error"] = job["error"] or "一键流程提前停止；" + d.preprocess_status.text()
        job["status"] = "failed" if job["error"] else "cancelled" if job.get("cancelled") else "completed"
        job["progress"] = 100 if job["status"] == "completed" else job["progress"]
        job["preprocess_status"] = d.preprocess_status.text()
        job["alignment_status"] = d.alignment_status.text()
        if job["origin"] == "mcp": d._pipeline_mcp_messages = None
        self.repeat_confirmed = False
        self.active_job_id = None
        self._workers = []
        self._settled_workers = set()
        self.timer.stop()

    def _manage(self, action, args):
        if args.get("job_id") != self.active_job_id or not self.active_job_id: raise ValueError("需要当前运行任务的job_id。")
        worker = self.dialog._current_shared_preprocess_task()
        if worker is None and self.jobs[self.active_job_id]["action"] in ("open_letter", "open_flash"):
            worker = next((w for w in self._workers if w.isRunning()), None)
            if worker is not None and action == "cancel":
                self.dialog._mark_lfp_cancel_requested(worker)
                worker.cancel()
                return dict(job_id=self.active_job_id, requested=action)
        if worker is None: raise ValueError("当前任务不支持暂停/取消。")
        if action == "cancel":
            if not hasattr(worker, "cancel"): raise ValueError("任务不支持取消。")
            self.dialog._cancel_preprocess_run()
        else:
            if not hasattr(worker, "set_paused"): raise ValueError("任务不支持暂停。")
            desired = action == "pause"
            if bool(getattr(worker, "_paused", False)) != desired: self.dialog._toggle_preprocess_pause()
        return dict(job_id=self.active_job_id, requested=action)

    def get_jobs(self, args):
        _only(args, ("job_id",))
        self._poll_job()
        def public(job): return {k: v for k, v in job.items() if k not in ("options", "started", "input_key")}
        if "job_id" in args:
            if args["job_id"] not in self.jobs: raise ValueError("未知job_id。")
            return public(self.jobs[args["job_id"]])
        return dict(jobs=[public(j) for j in self.jobs.values()])

    def read_table(self, args):
        _only(args, ("table", "offset", "limit"))
        offset = _integer(args.get("offset", 0), 0, 10**9, "offset")
        limit = _integer(args.get("limit", 100), 1, 500, "limit")
        table, d = args.get("table"), self.dialog
        if table == "markers":
            rows = np.asarray(getattr(d, "stim_markers", np.empty((0, 2), dtype=int)))
            data = [dict(sample=int(s), tag=int(t), time_sec=float(s)/d.active_source.metadata.fs) for s, t in rows[offset:offset+limit]]
            total = len(rows)
        elif table == "annotations":
            annotations = getattr(d, "letter_csv_annotations", {})
            behavior = getattr(d, "behavior_matches", {})
            keys = sorted(set(annotations) | set(behavior))
            data = [dict(sample=int(k), csv=annotations.get(k), behavior=behavior.get(k)) for k in keys[offset:offset+limit]]
            total = len(keys)
        elif table == "segments":
            from qt_data_model import alignment_log_segments
            path = self._input_path(d.alignment_log_edit.text())
            rows = alignment_log_segments(path, float(d.bin_timing_metadata.get("deltaT1_sec", 0) or 0))
            data = [dict(segment_index=i, **row) for i, row in enumerate(rows) if offset <= i < offset+limit]
            total = len(rows)
        elif table == "qc":
            ids = sorted(map(int, d._bad_channel_result_channel_ids))
            data = [dict(channel=ch, classification="bad" if ch in d.bad_channel_ids else "good",
                         override=d.manual_channel_overrides.get(ch), auto_reasons=d.bad_channel_auto_reasons.get(ch, []))
                    for ch in ids[offset:offset+limit]]
            total = len(ids)
        else: raise ValueError("table=markers/annotations/segments/qc。")
        return dict(table=table, total=total, offset=offset, rows=data,
                    stale=self._alignment_stale() if table in ("markers", "annotations") else False)

    def _refresh_preprocess(self, args):
        d = self.dialog
        original = args.get("original", False)
        if type(original) is not bool: raise ValueError("original必须是布尔值。")
        preview = d.original_preprocess_preview if original else d.preprocessed_preview
        meta = preview.source.metadata
        channel = args.get("channel", int(meta.channel_ids[preview.channel_spin.value()-1]))
        self._channel_ids([channel])
        start, end = args.get("start_sec", preview.start_spin.value()), args.get("end_sec", preview.end_spin.value())
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= meta.rows/meta.fs:
            raise ValueError("预览范围必须在当前数据时长内，且起点<终点。")
        full = args.get("full_duration", preview.full_duration_check.isChecked())
        if type(full) is not bool: raise ValueError("full_duration必须是布尔值。")
        d._activate_page(1)
        d.preprocess_preview_tabs.setCurrentIndex(1 if original else 0)
        preview.channel_spin.setValue(list(meta.channel_ids).index(channel)+1)
        if "channel" in args: preview.set_channel_cycle_columns(())
        preview.full_duration_check.setChecked(full)
        preview.start_spin.setValue(start); preview.end_spin.setValue(end)
        preview.refresh()

    def _plot(self, name):
        d = self.dialog
        if name == "alignment":
            if self.alignment_preview_signature != self.alignment_preview_key():
                raise ValueError("对齐预览参数/数据已改变，请先刷新refresh_alignment。")
            return d.alignment_figure._plot_widget
        if name in ("preprocess", "original"):
            preview = d.preprocessed_preview if name == "preprocess" else d.original_preprocess_preview
            if getattr(preview, "_mcp_render_key", None) != self.preprocess_preview_key(preview):
                raise ValueError("预处理预览参数/数据已改变，请先刷新refresh_preprocess。")
            return preview.figure._plot_widget
        raise ValueError("plot=alignment/preprocess/original。")

    def _capture(self, name):
        if self._running_workers(): raise ValueError("任务运行中，请等完成后截图。")
        widget = self._plot(name)
        if not widget.listDataItems(): raise ValueError("没有预览曲线；请先刷新相应预览。")
        pixmap = widget.grab()
        if pixmap.isNull(): raise ValueError("无法获取图像。")
        return pixmap

    def get_image(self, args):
        _only(args, ("plot", "index"))
        if args.get("index", 0) != 0: raise ValueError("主窗口图像index只能为0。")
        pixmap = self._capture(args.get("plot", "alignment"))
        data = QtCore.QByteArray(); buffer = QtCore.QBuffer(data)
        mode = getattr(QtCore.QIODevice, "OpenModeFlag", QtCore.QIODevice)
        buffer.open(mode.WriteOnly)
        if not pixmap.save(buffer, "PNG"): raise ValueError("PNG生成失败。")
        buffer.close()
        if len(data) > 8*1024*1024: raise ValueError("图像过大。")
        return dict(mime_type="image/png", data=base64.b64encode(bytes(data)).decode("ascii"),
                    width=pixmap.width(), height=pixmap.height(), includes_title=True,
                    source="current_display", stale=self._alignment_stale() if args.get("plot", "alignment") == "alignment" else False)

    def read_waveform(self, args):
        _only(args, ("plot", "index", "curve", "offset", "limit"))
        if args.get("index", 0) != 0: raise ValueError("index只能为0。")
        curves = self._plot(args.get("plot", "alignment")).listDataItems()
        curve = _integer(args.get("curve", 0), 0, max(0, len(curves)-1), "curve")
        if not curves: raise ValueError("无曲线，请先刷新预览。")
        offset = _integer(args.get("offset", 0), 0, 10**9, "offset")
        limit = _integer(args.get("limit", 1000), 1, 4096, "limit")
        x, y = curves[curve].getData()
        return dict(x=x[offset:offset+limit], y=y[offset:offset+limit], offset=offset, total=len(x),
                    curve_count=len(curves), units="mV" if args.get("plot", "alignment") == "alignment" else "μV",
                    x_units="seconds", source="display_decimated", unit_convention="GUI label, not physical recalibration")
