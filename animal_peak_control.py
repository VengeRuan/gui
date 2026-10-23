"""Explicit, versioned operations for ONE existing AnimalPeakDialog.

All entry points execute on the Qt GUI thread. Algorithms stay in existing
workers; this module never evaluates code or simulates mouse/keyboard input.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
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

from animal_peak_rpc import API_VERSION, IMPLEMENTATION_VERSION, BridgeError


CONTROLS = {
    "extraction": dict(search_ms="search_spin", search_after_ms="search_after_spin", pre_ms="pre_spin",
        post_ms="post_spin", channels="channel_edit", tags="tag_edit", trial_limit="limit_spin",
        localization_mode="localization_box", despike_window_ms="despike_spin",
        despike_mad_threshold="mad_spin", trend_window_ms="trend_spin", use_task_processing="processing_check"),
    "event": dict(alignment="letter_vep_alignment", start_ms="letter_vep_start", end_ms="letter_vep_end",
        baseline_enabled="letter_vep_baseline", baseline_start_ms="letter_vep_base_start",
        baseline_end_ms="letter_vep_base_end", duration_ms="letter_vep_duration"),
    "similarity": dict(before_ms="judge_before_spin", after_ms="judge_after_spin", r_threshold="judge_r_spin",
        required_fraction_percent="judge_fraction_spin", min_window_coverage_percent="judge_coverage_spin",
        min_reference_coverage_percent="judge_reference_coverage_spin"),
    "filter": dict(channels="secondary_channels", mode="secondary_mode", frequency_hz="secondary_frequency",
        q="secondary_q", low_hz="secondary_low", high_hz="secondary_high", order="secondary_order"),
    "view": dict(global_y_min="y_min_spin", global_y_max="y_max_spin", event_y_min="letter_vep_y_min",
        event_y_max="letter_vep_y_max", event_width_px="letter_vep_width", letter_before_ms="letter_view_spin",
        trial_range_mode="trial_range_box", details_per_page="details_per_page_box", detail_curve="detail_curve_box",
        manual_average_visible="manual_average_check", manual_pick_enabled="manual_pick_check",
        parameters_expanded="parameter_toggle", individual_png="individual_check"),
    "manual": dict(relative_ms="manual_time_spin"),
}
TABS = {"average": 0, "histogram": 1, "details": 2, "records": 3,
        "animal_reference": 4, "similarity": 5, "event": 6}
READ_OPERATIONS = {"get_state", "describe", "read_table", "read_records", "read_waveform", "get_image", "get_jobs"}
WRITE_OPERATIONS = {"set_parameters", "navigate", "set_trial_selection", "execute"}
ACTIONS = ("extract", "refresh_average", "refresh_event", "compute_similarity", "apply_manual", "restore_manual",
    "filter", "restore_filter", "apply_global_y", "auto_global_y", "apply_event_y", "auto_event_y", "apply_width",
    "auto_width", "apply_letter_range", "show_average", "clear_manual_preview", "reset_view", "copy_plot",
    "export_plot", "export", "cancel", "close")


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.size > 4096:
            raise ValueError("数组过大，请使用分页波形接口。")
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("不支持的结果类型。")


def _only(arguments, names):
    unknown = set(arguments)-set(names)
    if unknown:
        raise ValueError("未知参数："+", ".join(sorted(unknown)))


def _integer(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name}必须是{low}～{high}的整数。")
    return value


def _value(widget):
    if isinstance(widget, QtWidgets.QComboBox):
        return widget.currentData()
    if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        return widget.value()
    if isinstance(widget, QtWidgets.QAbstractButton):
        return widget.isChecked()
    return widget.text()


def _assign(widget, value):
    if isinstance(widget, QtWidgets.QComboBox):
        widget.setCurrentIndex(widget.findData(value))
    elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
        widget.setValue(value)
    elif isinstance(widget, QtWidgets.QAbstractButton):
        widget.setChecked(value)
    else:
        widget.setText(value)


class AnimalPeakController:
    def __init__(self, dialog):
        self.dialog = dialog
        self.revision = 0
        self._fingerprint = None
        self.command_active = False
        self.jobs = {}
        self.active_job_id = None
        self.audit = []

    def controls(self):
        return {group: {name: _value(getattr(self.dialog, attr)) for name, attr in fields.items()}
                for group, fields in CONTROLS.items()}

    def _sync_revision(self):
        d = self.dialog
        signature = safe_json(dict(controls=self.controls(), output=id(d.output), source=id(d.source),
            busy=d._busy, pair=d._current_pair(), page=d._detail_page, tab=d.tabs.currentIndex(),
            trial=d.manual_trial_box.currentData(), branch=d.manual_branch_box.currentData(),
            reference=d.similarity_reference_box.currentData(), ranked=d.similarity_trial_box.currentData(),
            similarity_tab=d.similarity_views.currentIndex(), applied_display=(d._fixed_y_limits,
                d._event_y_override, d._event_plot_width, d._letter_before_ms, d._trial_range_mode),
            exported=(d._exported_manual_actions, d._exported_window_resizes, d._exported_average_selections,
                      d._exported_secondary_filters, d._exported_letter_revision)))
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).digest()
        if fingerprint != self._fingerprint:
            self.revision += 1; self._fingerprint = fingerprint
        return self.revision

    def dispatch(self, operation, arguments, expected_revision=None):
        d = self.dialog
        if QtCore.QThread.currentThread() is not d.thread():
            return dict(ok=False, error=dict(code="wrong_thread", message="控制请求必须在Qt主线程执行。"))
        try:
            if operation not in READ_OPERATIONS | WRITE_OPERATIONS or not isinstance(arguments, dict):
                raise BridgeError("invalid_operation", "操作不在允许清单中。")
            self._sync_revision()
            if operation in WRITE_OPERATIONS:
                if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision != self.revision:
                    raise BridgeError("stale_revision", "窗口状态已变化或缺少expected_revision，请重新get_state后再操作。")
                if QtWidgets.QApplication.activeModalWidget() is not None:
                    raise BridgeError("modal_dialog_open", "窗口存在人工确认对话框，请先完成或关闭该对话框。")
                if d._busy and not (operation == "execute" and arguments.get("action") == "cancel"):
                    raise BridgeError("busy", "当前任务未结束；可查询进度或取消，不并发修改状态。")
            handlers = {"get_state": self.get_state, "describe": self.describe, "set_parameters": self.set_parameters,
                "navigate": self.navigate, "set_trial_selection": self.set_trial_selection, "execute": self.execute,
                "read_table": self.read_table, "read_records": self.read_records, "read_waveform": self.read_waveform,
                "get_image": self.get_image, "get_jobs": self.get_jobs}
            self.command_active = operation in WRITE_OPERATIONS
            result = handlers[operation](arguments)
            self._sync_revision()
            if operation in WRITE_OPERATIONS:
                self.audit.append(dict(operation=operation, arguments=safe_json(arguments), revision=self.revision,
                    timestamp_utc=datetime.now(timezone.utc).isoformat()))
                self.audit = self.audit[-200:]
            return dict(ok=True, revision=self.revision, result=safe_json(result))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            return dict(ok=False, revision=self._sync_revision(), error=dict(
                code=exc.code if isinstance(exc, BridgeError) else "invalid_request", message=str(exc)))
        except Exception:
            return dict(ok=False, revision=self._sync_revision(), error=dict(code="internal_error",
                message="操作未完成，请读取窗口状态与任务错误；不会自动重试写操作。"))
        finally:
            self.command_active = False

    def get_state(self, args):
        _only(args, ())
        d = self.dialog; meta = d.source.metadata; output = d.output
        self._sync_revision()
        event = output.get("letter_vep", {}) if output else {}
        group = event.get("cache")
        return dict(api_version=API_VERSION, implementation_version=IMPLEMENTATION_VERSION, revision=self.revision,
            source=dict(path=str(meta.path), fs=float(meta.fs), rows=int(meta.rows),
                aligned_channel_ids=list(d.defaults["channel_ids"])), controls=self.controls(),
            read_only_defaults=d.defaults,
            selectors=dict(channels=[d.result_channel_box.itemData(i) for i in range(d.result_channel_box.count())],
                tags=[d.result_tag_box.itemData(i) for i in range(d.result_tag_box.count())]),
            busy=d._busy, progress=d.progress.value(), status=d.status.text(), active_job_id=self.active_job_id,
            current=dict(pair=d._current_pair(), tab=next(k for k,v in TABS.items() if v == d.tabs.currentIndex()),
                detail_page=d._detail_page, trial_id=d.manual_trial_box.currentData(), branch=d.manual_branch_box.currentData()),
            has_results=output is not None, unsaved_edits=d._has_unsaved_edits(),
            parameters_dirty=d._parameters_dirty, display_settings=output.get("display_settings", {}) if output else {},
            result_summary=None if output is None else dict(record_count=len(output["records"]),
                group_count=len(output["groups"]), units=output["units"], settings=output["settings"],
                manual_actions=len(output.get("manual_alignment_history", [])),
                event_stale=event.get("stale", False), event_parameters=event.get("parameters"),
                event_selected_count=len(group["selected_indices"]) if group else None,
                event_selections=event.get("selections", []), peak_selections=output.get("average_trial_selections", [])),
            export_root=str(Path(d.mcp_export_root.text()).resolve()), recent_operations=self.audit[-10:])

    def describe(self, args):
        _only(args, ())
        result = {}
        for group, fields in CONTROLS.items():
            result[group] = {}
            for name, attr in fields.items():
                widget = getattr(self.dialog, attr)
                item = dict(value=_value(widget), enabled=widget.isEnabled(), tooltip=widget.toolTip())
                if isinstance(widget, QtWidgets.QComboBox):
                    item.update(type="enum", options=[widget.itemData(i) for i in range(widget.count())])
                elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    item.update(type="integer" if isinstance(widget, QtWidgets.QSpinBox) else "number",
                        minimum=widget.minimum(), maximum=widget.maximum(),
                        decimals=widget.decimals() if isinstance(widget, QtWidgets.QDoubleSpinBox) else 0)
                elif isinstance(widget, QtWidgets.QAbstractButton):
                    item.update(type="boolean")
                else:
                    item.update(type="string", read_only=widget.isReadOnly())
                result[group][name] = item
        return dict(groups=result, tabs=TABS, actions=ACTIONS,
            plot_targets=["average", "histogram", "details", "animal_reference", "similarity", "event", "manual_average"],
            table_targets=["records", "event", "similarity", "rankings"],
            rules="写操作必须指定session_id和最新expected_revision。参数只修改控件，需执行对应apply/refresh动作。"
                "事件平均名单与峰谷名单独立；原Trial编号不是Tag内序号。filter/export/丢弃未保存结果需confirm=true。")

    def _validate_value(self, group, name, value):
        widget = getattr(self.dialog, CONTROLS[group][name])
        if isinstance(widget, QtWidgets.QComboBox):
            if not any(type(value) is type(widget.itemData(i)) and value == widget.itemData(i) for i in range(widget.count())):
                raise ValueError(f"{group}.{name}不在允许选项中。")
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(name+"必须为有限数字。")
            low, high = widget.minimum(), widget.maximum()
            if group == "event" and name in ("start_ms", "end_ms", "duration_ms"):
                low, high = (-60000, 60000) if name != "duration_ms" else (.1, 120000)
            if not low <= value <= high:
                raise ValueError(f"{name}超出范围[{low}, {high}]。")
            decimals = widget.decimals() if isinstance(widget, QtWidgets.QDoubleSpinBox) else 0
            if abs(value-round(value, decimals)) > 1e-8:
                raise ValueError(f"{name}最多支持{decimals}位小数，不会静默截断。")
            if isinstance(widget, QtWidgets.QSpinBox) and not isinstance(value, int):
                raise ValueError(name+"必须为整数。")
        elif isinstance(widget, QtWidgets.QAbstractButton):
            if not isinstance(value, bool):
                raise ValueError(name+"必须为布尔值。")
        else:
            if widget.isReadOnly():
                raise ValueError(name+"来自已完成对齐，是只读通道范围。")
            if not isinstance(value, str) or len(value) > 16384:
                raise ValueError(name+"必须为长度受限的文本。")
            if name in ("channels", "tags"):
                parsed = [int(x.strip()) for x in value.replace(";", ",").replace("，", ",").split(",") if x.strip()]
                allowed = set(self.dialog.source.metadata.channel_ids) if name == "channels" else {4,5,6,7}
                if not set(parsed) <= allowed:
                    raise ValueError(name+"含未载入通道或非Letter Tag。")

    def set_parameters(self, args):
        _only(args, ("group", "values"))
        group, values = args.get("group"), args.get("values")
        if group not in CONTROLS or not isinstance(values, dict) or not values:
            raise ValueError("请指定describe中的参数组及非空values。")
        _only(values, CONTROLS[group])
        for name, value in values.items():
            self._validate_value(group, name, value)
        d = self.dialog
        if group == "event":
            from animal_peak_letter_vep import parameters
            current = self.controls()["event"]
            cfg = {k:v for k,v in current.items() if k != "duration_ms"}
            if "alignment" in values and values["alignment"] != current["alignment"]:
                cfg.update(alignment=values["alignment"], start_ms=-current["duration_ms"] if values["alignment"]=="animal" else 0,
                           end_ms=0 if values["alignment"]=="animal" else current["duration_ms"])
            cfg.update({k:v for k,v in values.items() if k != "duration_ms"})
            if "duration_ms" in values:
                if set(values) & {"start_ms", "end_ms"}:
                    raise ValueError("duration_ms不能与显式start_ms/end_ms同时设置。")
                if cfg["alignment"] == "animal": cfg["start_ms"] = cfg["end_ms"]-values["duration_ms"]
                else: cfg["end_ms"] = cfg["start_ms"]+values["duration_ms"]
            cfg = parameters(cfg)
            if not (-60000 <= cfg["start_ms"] <= 60000 and -60000 <= cfg["end_ms"] <= 60000):
                raise ValueError("事件窗超出界面可设置范围。")
            values = {**cfg, "duration_ms":cfg["end_ms"]-cfg["start_ms"]}
        # Validate the whole patch BEFORE modifying any widget. Block signals
        # so intermediate endpoint/mode changes never clamp valid final values.
        widgets = [getattr(d, attr) for attr in CONTROLS[group].values()]
        blockers = [QtCore.QSignalBlocker(widget) for widget in widgets]
        if group == "event":
            d.letter_vep_end.setMaximum(0 if values["alignment"] == "animal" else 60000)
            d.letter_vep_duration.setMaximum(120000)
        for name, value in values.items():
            _assign(getattr(d, CONTROLS[group][name]), value)
        del blockers
        if group == "extraction":
            d._localization_changed(mark_changed=False); d.parameters_changed()
        elif group == "event":
            d._letter_vep_parameters_changed()
        elif group == "similarity":
            d._similarity_parameters_changed()
        elif group == "filter":
            d._secondary_mode_changed()
        elif group == "manual":
            d._preview_manual_point()
        elif group == "view":
            if set(values) & {"trial_range_mode", "letter_before_ms"}: d._trial_range_pending()
            if "details_per_page" in values: d._detail_layout_changed()
            elif "detail_curve" in values: d.render_details()
            if "parameters_expanded" in values: d._toggle_parameters(values["parameters_expanded"])
            d.manual_average_box.setVisible(d.manual_average_check.isChecked())
            d._update_page_geometry()
        return dict(controls=self.controls()[group], note="输入已更新；需执行对应刷新/应用动作才改变计算或显示范围。")

    def navigate(self, args):
        _only(args, ("tab", "channel", "tag", "trial_id", "branch", "detail_page", "reference_trial_id", "rank_trial_id", "similarity_tab"))
        d = self.dialog
        if "tab" in args and args["tab"] not in TABS:
            raise ValueError("未知标签页。")
        if "branch" in args and args["branch"] not in ("peak", "trough"):
            raise ValueError("branch应为peak或trough。")
        tab = args.get("tab", next(k for k,v in TABS.items() if v == d.tabs.currentIndex()))
        pair = d._current_pair()
        if "channel" in args or "tag" in args:
            if d.output is None:
                raise ValueError("请先运行提取。")
            channel, tag = args.get("channel", pair[0] if pair else None), args.get("tag", pair[1] if pair else None)
            if not isinstance(channel, int) or isinstance(channel, bool) or not isinstance(tag, int) or isinstance(tag, bool):
                raise ValueError("channel/tag必须为实际整数编号。")
            if tab == "event":
                from animal_peak_letter_vep import available_pairs
                from animal_peak_letter_vep import new_state
                allowed = available_pairs(d.output, d.output.get("letter_vep") or new_state(d.markers,d.matches))
            else:
                allowed = d._pairs
            if (channel, tag) not in allowed:
                raise ValueError("此通道/Tag不属于目标页可用结果，不增加计算范围。")
        else:
            channel, tag = pair if pair else (None,None)
        if "trial_id" in args:
            _integer(args["trial_id"],1,10000000,"trial_id")
            if d.output is None: raise ValueError("没有Trial结果。")
            trials = sorted({r["trial_id"] for r in d.output["records"] if
                (r.get("channel"),r.get("tag")) == (channel,tag) and "extreme_local_sample" in r})
            if args["trial_id"] not in trials: raise ValueError("该原Trial没有可调整点位。")
        for key in ("detail_page", "similarity_tab"):
            if key in args: _integer(args[key], 0, 1000000 if key=="detail_page" else 2, key)
        if "reference_trial_id" in args or "rank_trial_id" in args:
            similarity = (d.output or {}).get("judgment_similarity")
            group = next((g for g in similarity["groups"] if (g["channel"],g["tag"]) == (channel,tag)), None) if similarity and not similarity.get("stale") and not d._parameters_dirty else None
            if group is None: raise ValueError("当前组尚无有效判断窗相似度。")
            for key in ("reference_trial_id", "rank_trial_id"):
                if key not in args: continue
                value = args[key]
                if key == "reference_trial_id" and value is None: continue
                _integer(value,1,10000000,key)
                if not any(t["trial_id"] == value and (key == "rank_trial_id" or not t.get("reason")) for t in group["trials"]):
                    raise ValueError(key+"不属于可用Trial。")
        d.tabs.setCurrentIndex(TABS[tab])
        if channel is not None:
            d.result_tag_box.setCurrentIndex(d.result_tag_box.findData(tag))
            index = d.result_channel_box.findData(channel)
            if index < 0: raise ValueError("通道不在当前选择列表中。")
            d.result_channel_box.setCurrentIndex(index)
        if "detail_page" in args:
            d._detail_page = args["detail_page"]; d.render_details()
        if "trial_id" in args:
            d._detail_page = trials.index(args["trial_id"])//int(d.details_per_page_box.currentData())
            d.render_details()
            d.manual_trial_box.setCurrentIndex(d.manual_trial_box.findData(args["trial_id"]))
        if "branch" in args: d.manual_branch_box.setCurrentIndex(d.manual_branch_box.findData(args["branch"]))
        for key, widget in (("reference_trial_id",d.similarity_reference_box),("rank_trial_id",d.similarity_trial_box)):
            if key in args:
                index = widget.findData(args[key])
                if index < 0 or not widget.isEnabled(): raise ValueError(key+"尚无有效相似度结果。")
                widget.setCurrentIndex(index)
        if "similarity_tab" in args: d.similarity_views.setCurrentIndex(args["similarity_tab"])
        return dict(current_pair=d._current_pair(), detail_page=d._detail_page, trial_id=d.manual_trial_box.currentData())

    def set_trial_selection(self, args):
        _only(args, ("target", "trial_ids"))
        d = self.dialog
        if d.output is None or d._current_pair() is None: raise ValueError("没有当前组结果。")
        ids = args.get("trial_ids")
        if ids is not None and (not isinstance(ids,list) or any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in ids)):
            raise ValueError("trial_ids为原Trial编号列表，null=全部，[]=全部不选。")
        if args.get("target") == "peaks":
            from animal_peak_trial_selection import set_average_trial_selection
            d.output = set_average_trial_selection(d.output,*d._current_pair(),ids)
            d.render_pair(reset_page=False); d.export_button.setEnabled(True)
            return dict(selection=ids, note="峰谷/旧AnimalStart对照名单已更新；事件名单独立。")
        if args.get("target") == "event":
            state = d.output.get("letter_vep", {})
            if not d._event_plot_copy_ready() or d._letter_vep_config()!=state.get("parameters"):
                raise ValueError("请先刷新当前事件组及参数，再修改事件共用名单。")
            d.refresh_letter_vep(trial_ids=ids)
            return dict(job_id=self.active_job_id)
        raise ValueError("target必须为peaks或event。")

    def execute(self, args):
        _only(args, ("action", "confirm", "job_id", "plot", "index", "x_range", "y_range", "directory", "filename"))
        action = args.get("action"); d = self.dialog
        if action not in ACTIONS: raise ValueError("未知动作。")
        confirmed = args.get("confirm") is True
        if action in ("filter", "restore_filter", "export", "export_plot") and not confirmed:
            raise BridgeError("confirmation_required", "整段滤波/导出需要明确confirm=true。")
        if action in ("extract", "close") and d._has_unsaved_edits() and not confirmed:
            raise BridgeError("confirmation_required", "操作会丢弃未保存修改；请先导出或明确confirm=true。")
        no_output_actions = {"extract", "apply_global_y", "auto_global_y", "apply_event_y", "auto_event_y",
            "apply_width", "auto_width", "apply_letter_range", "cancel", "close"}
        if d.output is None and action not in no_output_actions: raise ValueError("请先提取结果。")
        if action in ("apply_manual", "restore_manual") and (d._parameters_dirty or d._selected_manual_row() is None):
            raise ValueError("需先应用/恢复提取参数并选择有效Trial与分支。")
        if action == "extract": d.start_analysis(confirm_discard=confirmed)
        elif action == "filter": d.start_secondary_filter(confirmed=True)
        elif action == "restore_filter": d.start_secondary_filter(True, confirmed=True)
        elif action == "export":
            directory = self._export_path(args.get("directory", "."))
            directory.mkdir(parents=True, exist_ok=True)
            d.start_export(directory=str(directory), confirmed=True)
        elif action in ("apply_manual", "restore_manual"): d.start_manual_adjustment(action=="restore_manual")
        elif action == "refresh_event": d.refresh_letter_vep()
        elif action == "refresh_average": d.refresh_average_plot()
        elif action == "compute_similarity": d.start_similarity()
        elif action == "cancel":
            job_id = args.get("job_id")
            if job_id != self.active_job_id or job_id is None: raise ValueError("请指定当前正在运行的job_id。")
            d.cancel()
        elif action == "close":
            d._mcp_discard_close = confirmed
            QtCore.QTimer.singleShot(100, d.close)
        elif action in ("copy_plot", "export_plot"):
            pixmap = self._capture(args.get("plot", "event"), args.get("index", 0))
            if action == "copy_plot": QtWidgets.QApplication.clipboard().setPixmap(pixmap)
            else:
                filename = args.get("filename")
                if not isinstance(filename,str) or Path(filename).suffix.lower() != ".png": raise ValueError("需指定新的.png文件名。")
                path = self._export_path(filename)
                if path.exists(): raise ValueError("目标文件已存在，不覆盖。")
                path.parent.mkdir(parents=True, exist_ok=True)
                if not pixmap.save(str(path), "PNG"): raise OSError("保存PNG失败。")
                return dict(path=str(path))
        elif action == "reset_view": self._reset_view(args)
        else:
            {"apply_global_y":d.apply_fixed_y_axis, "auto_global_y":d.restore_auto_y_axis,
             "apply_event_y":d.apply_event_y_axis, "auto_event_y":d.restore_event_auto_y_axis,
             "apply_width":d.apply_event_plot_width, "auto_width":d.restore_event_auto_width,
             "apply_letter_range":d.apply_letter_display_range, "show_average":d.show_latest_average,
             "clear_manual_preview":d._clear_manual_preview}[action]()
        return dict(action=action, job_id=self.active_job_id, status=d.status.text())

    def _export_path(self, name):
        if not isinstance(name,str) or not name: raise ValueError("导出路径必须为文本。")
        root = Path(self.dialog.mcp_export_root.text()).resolve()
        path = (root/Path(name)).resolve()
        if not path.is_relative_to(root): raise ValueError("导出路径超出窗口已授权的根目录。")
        return path

    def _plot_widget(self, name, index=0):
        d = self.dialog
        mapping = {"average":d.average_plots, "histogram":d.histogram_plots,
            "details":[d.detail_canvas], "animal_reference":d.animal_reference_plots,
            "similarity":[d.similarity_canvas], "event":[d.letter_vep_plot], "manual_average":d.manual_average_plots}
        if name not in mapping: raise ValueError("未知图形目标。")
        _integer(index,0,len(mapping[name])-1,"index")
        if d.output is None: raise ValueError("请先提取结果。")
        if name == "event" and not d._event_plot_copy_ready(): raise ValueError("当前事件组尚未刷新。")
        if name == "similarity" and d._current_similarity_group() is None: raise ValueError("当前相似度尚未计算或已失效。")
        return mapping[name][index]

    def _capture(self, name, index):
        if self.dialog._busy: raise ValueError("计算中不读取可能残留的旧图。")
        widget = self._plot_widget(name,index)
        if name == "event": self.dialog._sync_event_plot_size()
        if name in ("details", "similarity"): widget.draw()
        pixmap = widget.grab()
        if pixmap.isNull(): raise ValueError("图像尚未生成。")
        return pixmap

    def get_image(self, args):
        _only(args, ("plot", "index"))
        pixmap = self._capture(args.get("plot","event"),args.get("index",0))
        data = QtCore.QByteArray(); buffer = QtCore.QBuffer(data)
        buffer.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        if not pixmap.save(buffer, "PNG"): raise OSError("生成PNG失败。")
        buffer.close()
        if len(data) > 8*1024*1024: raise ValueError("图像过大，请缩小图框或每页显示图数。")
        return dict(mime_type="image/png", data=base64.b64encode(bytes(data)).decode("ascii"),
            width=pixmap.width(), height=pixmap.height(), includes_title=True, source="current_display")

    def _reset_view(self, args):
        widget = self._plot_widget(args.get("plot","event"),args.get("index",0))
        ranges = {}
        for key in ("x_range","y_range"):
            value = args.get(key)
            if value is not None:
                if not isinstance(value,list) or len(value)!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in value) or value[0]>=value[1]:
                    raise ValueError(key+"必须是两个递增有限数字。")
                ranges[key] = value
        if hasattr(widget,"getViewBox"):
            if not ranges: widget.autoRange()
            if "x_range" in ranges: widget.setXRange(*ranges["x_range"],padding=0)
            if "y_range" in ranges: widget.setYRange(*ranges["y_range"],padding=0)
            fixed = self.dialog._fixed_y_limits is not None
            if widget is self.dialog.letter_vep_plot and self.dialog._event_y_override is not None:
                fixed = self.dialog._event_y_override["y_axis_mode"] == "fixed"
            if fixed: self.dialog._apply_waveform_y_ranges([widget])
        else:
            for axis in widget.figure.axes:
                if not ranges: axis.relim(); axis.autoscale_view()
                if "x_range" in ranges: axis.set_xlim(*ranges["x_range"])
                if "y_range" in ranges: axis.set_ylim(*ranges["y_range"])
            widget.draw_idle()

    def read_table(self, args):
        _only(args, ("table","offset","limit"))
        d = self.dialog
        tables = {"records":d.table,"event":d.letter_vep_table,"similarity":d.similarity_table,"rankings":d.similarity_rank_table}
        if args.get("table") not in tables: raise ValueError("未知表格。")
        table = tables[args["table"]]
        offset = _integer(args.get("offset",0),0,10000000,"offset")
        limit = _integer(args.get("limit",100),1,500,"limit")
        return dict(total=table.rowCount(), offset=offset,
            columns=[table.horizontalHeaderItem(i).text() for i in range(table.columnCount())],
            rows=[[table.item(i,j).text() if table.item(i,j) else "" for j in range(table.columnCount())]
                for i in range(offset,min(offset+limit,table.rowCount()))])

    def read_records(self, args):
        _only(args, ("channel","tag","trial_id","offset","limit"))
        offset = _integer(args.get("offset",0),0,10000000,"offset")
        limit = _integer(args.get("limit",100),1,500,"limit")
        for key in ("channel","tag","trial_id"):
            if key in args: _integer(args[key],0,10000000,key)
        rows = self.dialog.output["records"] if self.dialog.output else []
        rows = [r for r in rows if all(r.get(k)==args[k] for k in ("channel","tag","trial_id") if k in args)]
        return dict(total=len(rows), offset=offset, rows=rows[offset:offset+limit])

    def read_waveform(self, args):
        _only(args, ("plot","index","curve","offset","limit"))
        widget = self._plot_widget(args.get("plot","event"),args.get("index",0))
        curves = widget.listDataItems() if hasattr(widget,"listDataItems") else [line for ax in widget.figure.axes for line in ax.lines]
        if self.dialog._busy: raise ValueError("计算中不读取旧波形，请先等待任务结束。")
        if not curves: raise ValueError("当前图没有波形曲线。")
        curve = _integer(args.get("curve",0),0,len(curves)-1,"curve")
        item = curves[curve]
        x,y = (item.xData,item.yData) if hasattr(item,"xData") else item.get_data()
        x,y = np.asarray(x),np.asarray(y)
        offset = _integer(args.get("offset",0),0,100000000,"offset")
        limit = _integer(args.get("limit",1000),1,4096,"limit")
        return dict(curve_count=len(curves),curve=curve,total=len(x),offset=offset,
                    x=x[offset:offset+limit], y=y[offset:offset+limit], meaning="当前绘图数值，不是重新读取/计算的原始数据")

    def worker_started(self, worker):
        job_id = str(uuid.uuid4()); self.active_job_id = job_id
        label = next((name for name in ("letter","secondary","resize","similarity","manual","export")
            if getattr(worker,name,None) is not None), "extract")
        job = dict(job_id=job_id, action=label, status="running", progress=0, origin="mcp" if self.command_active else "gui",
            created_utc=datetime.now(timezone.utc).isoformat(), error=None)
        self.jobs[job_id] = job
        if self.command_active: worker._mcp_job_id = job_id
        for old in list(self.jobs)[:-100]: del self.jobs[old]
        worker.progress.connect(lambda value,_text: job.update(progress=float(value)))
        outcome = {}
        def completed(value):
            outcome["status"] = "cancelled" if getattr(worker,"_ui_cancel_requested",False) or worker.isInterruptionRequested() else "completed"
            if isinstance(value,Path): job["export_directory"] = str(value)
        worker.completed.connect(completed)
        worker.failed.connect(lambda message: outcome.update(status="failed",error=message))
        worker.cancelled.connect(lambda message: outcome.update(status="cancelled",error=message))
        def finished():
            # _worker_finished is connected before this slot. Do not advertise
            # completion until GUI callbacks have committed/rendered the result.
            job.update(status=outcome.get("status","cancelled"), error=outcome.get("error"))
            job["finished_utc"] = datetime.now(timezone.utc).isoformat()
            if self.active_job_id == job_id: self.active_job_id = None
        worker.finished.connect(finished)

    def get_jobs(self, args):
        _only(args, ("job_id",))
        if "job_id" in args:
            if args["job_id"] not in self.jobs: raise ValueError("未知job_id。")
            return self.jobs[args["job_id"]]
        return dict(active_job_id=self.active_job_id, jobs=list(self.jobs.values())[-20:])
