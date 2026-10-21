"""Non-modal experimental AnimalStart peak/trough viewer; no main-page mutations."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
try:
    from PyQt6 import QtCore, QtWidgets
except ImportError:
    from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from animal_peak_plotting import PLOT_STYLE, draw_trial_detail, draw_similarity_heatmap, waveform_display
from animal_peak_manual import adjust_animal_peak, resize_animal_peak_windows
from animal_peak_similarity import calculate_similarity, ranked_matches, summarize_group
from animal_peak_trial_selection import average_trial_ids, set_average_trial_selection, selected_average_group
from animal_peak_filter import apply_secondary_filter, filter_config, SecondaryFilteredSource
import animal_peak_letter_vep as letter_vep

from animal_peak_analysis import (
    AnalysisCancelled, analyze_animal_peaks, export_animal_peak_results, trial_detail_trace,
    _condition_channel, trial_detail_localization,
)


class AnimalPeakWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal(str)

    def __init__(self, source, markers, matches, settings, parent=None, export=None, manual=None, similarity=None, resize=None, secondary=None, letter=None):
        super().__init__(parent)
        self.source, self.markers, self.matches, self.settings = source, markers, matches, settings
        self.export = export
        self.manual = manual
        self.similarity = similarity
        self.resize = resize
        self.secondary = secondary
        self.letter = letter
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True
        self.requestInterruption()

    def _is_cancelled(self):
        return self._cancel_requested or self.isInterruptionRequested()

    def run(self):
        try:
            if self._is_cancelled():
                raise AnalysisCancelled("已取消测试操作")
            if self.letter is not None:
                output, pair, trial_ids = self.letter
                value = letter_vep.refresh(self.source, output, self.markers, self.matches, pair, self.settings,
                    trial_ids=trial_ids, progress=self.progress.emit, cancel=self._is_cancelled)
            elif self.secondary is not None:
                output, channels, config = self.secondary
                value = apply_secondary_filter(self.source, output, channels, config,
                                               self.progress.emit, self._is_cancelled)
            elif self.resize is not None:
                value = resize_animal_peak_windows(self.source, self.resize, **self.settings,
                                                   progress=self.progress.emit, cancel=self._is_cancelled)
            elif self.similarity is not None:
                value = calculate_similarity(self.source, self.similarity, self.settings,
                                             self.progress.emit, self._is_cancelled)
            elif self.manual is not None:
                output, request = self.manual
                value = adjust_animal_peak(self.source, output, **request,
                                           progress=self.progress.emit, cancel=self._is_cancelled)
            elif self.export is None:
                value = analyze_animal_peaks(self.source, self.markers, self.matches, self.settings,
                                            self.progress.emit, self._is_cancelled)
            else:
                output, directory, individual = self.export
                value = export_animal_peak_results(output, self.source, directory, individual,
                                                  self.progress.emit, self._is_cancelled)
            if self._is_cancelled():
                raise AnalysisCancelled("已取消测试操作")
            self.completed.emit(value)
        except AnalysisCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))


class AverageTrialSelectionDialog(QtWidgets.QDialog):
    """A reversible per-channel/Tag choice, separate from Trial eligibility."""
    def __init__(self, parent, output, pair, letter_group=None):
        super().__init__(parent)
        self.setWindowTitle(f"选择参与叠加平均的 Trial：ch{pair[0]} / Tag {pair[1]}")
        self.resize(950, 620)
        flags = QtCore.Qt.WindowType if hasattr(QtCore.Qt, "WindowType") else QtCore.Qt
        self.setWindowFlags(self.windowFlags() | flags.WindowMaximizeButtonHint | flags.WindowMinimizeButtonHint)
        layout = QtWidgets.QVBoxLayout(self)
        self._letter_selection = letter_group is not None
        hint = QtWidgets.QLabel("LetterStart、Dot和AnimalStart事件对齐共用此名单，与峰谷页独立。需有效AnimalStart；当前模式截取、基线或Dot时间不完整时不纳入，原因保留，但选择保留供其他模式使用。全不选不会自动恢复全选。" if self._letter_selection else "本组峰、谷及 AnimalStart 对照使用同一份选择，只影响平均与导出；不删除 Trial、不改变点位、不改变全组判断窗相似度。这里只列已有定位点的 Trial，缺少 AnimalStart/定位点的不能参与平均。全不选时不绘制平均线，不会自动恢复全选。")
        hint.setWordWrap(True); layout.addWidget(hint)
        toolbar = QtWidgets.QHBoxLayout()
        self.select_all_button = QtWidgets.QPushButton("全选")
        self.select_none_button = QtWidgets.QPushButton("全不选")
        self.invert_button = QtWidgets.QPushButton("反选")
        for button in (self.select_all_button, self.select_none_button, self.invert_button):
            toolbar.addWidget(button)
        self.count_label = QtWidgets.QLabel(); toolbar.addWidget(self.count_label, 1); layout.addLayout(toolbar)
        records = {}
        for row in (letter_group["records"] if self._letter_selection else output["records"]):
            if self._letter_selection:
                records[row["trial_id"]] = {"letter": row}
                continue
            if (row.get("channel"), row.get("tag")) == pair and "extreme_local_sample" in row:
                records.setdefault(row["trial_id"], {})[row["branch"]] = row
        ids = letter_vep.selected_ids(output["letter_vep"], pair) if self._letter_selection else average_trial_ids(output, *pair)
        chosen = set(records) if ids is None else set(ids)
        self.table = QtWidgets.QTableWidget(len(records), 6)
        self.table.setHorizontalHeaderLabels(["参与平均", "原 Trial", "Tag 内 Trial", "对齐事件时间 s", "状态/排除原因", "基线均值"] if self._letter_selection else ["参与平均", "原 Trial", "Tag 内 Trial", "峰分支", "谷分支", "人工点位"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        role = QtCore.Qt.ItemDataRole.UserRole
        for index, (trial, branches) in enumerate(sorted(records.items())):
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setData(role, int(trial))
            if self._letter_selection and not branches["letter"]["animal_valid"]:
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable)
                chosen.discard(trial)
            item.setCheckState(QtCore.Qt.CheckState.Checked if trial in chosen else QtCore.Qt.CheckState.Unchecked)
            self.table.setItem(index, 0, item)
            first = next(iter(branches.values()))
            texts = [trial, first["tag_trial_id"]]
            if self._letter_selection:
                texts.extend([f"{first['alignment_sec']:.6f}", first["reason"] or "可用", f"{first['baseline_correction']:.6g}"])
            else:
                for branch in ("peak", "trough"):
                    row = branches.get(branch)
                    texts.append("可用" if row and row["status"] == "included" else row.get("reason", "无完整片段") if row else "缺少定位点")
                texts.append("、".join("峰" if branch == "peak" else "谷" for branch, row in branches.items()
                                      if row.get("alignment_origin") == "manual") or "无")
            for column, value in enumerate(texts, 1):
                cell = QtWidgets.QTableWidgetItem(str(value)); cell.setToolTip(str(value)); self.table.setItem(index, column, cell)
        self.table.resizeColumnsToContents()
        for column in (3,4):
            self.table.setColumnWidth(column, 220)
        layout.addWidget(self.table, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("应用选择并更新平均")
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)
        self.select_all_button.clicked.connect(lambda: self._check_all(True))
        self.select_none_button.clicked.connect(lambda: self._check_all(False))
        self.invert_button.clicked.connect(lambda: self._check_all(None))
        self.table.itemChanged.connect(self._update_count)
        self._update_count()

    def selected_ids(self):
        return [int(self.table.item(row,0).data(QtCore.Qt.ItemDataRole.UserRole)) for row in range(self.table.rowCount())
                if self.table.item(row,0).checkState() == QtCore.Qt.CheckState.Checked]

    def _check_all(self, checked):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            item = self.table.item(row,0)
            if self._letter_selection and not item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
                continue
            state = item.checkState() != QtCore.Qt.CheckState.Checked if checked is None else checked
            item.setCheckState(QtCore.Qt.CheckState.Checked if state else QtCore.Qt.CheckState.Unchecked)
        self.table.blockSignals(False); self._update_count()

    def _update_count(self, *_):
        self.count_label.setText(f"已选择 {len(self.selected_ids())}/{self.table.rowCount()} 个 Trial；"
            + ("仅平均AnimalStart有效、截取和基线完整的所选项；名单与峰谷页独立。" if self._letter_selection else "各分支仅平均有完整片段的所选项。"))


class CurrentPageTabs(QtWidgets.QTabWidget):
    """Let the outer page scroll, without a hidden tall tab squeezing this one."""
    def _page_hint(self, minimum=False):
        page = self.currentWidget()
        if page is None:
            return super().minimumSizeHint() if minimum else super().sizeHint()
        size = page.minimumSizeHint() if minimum else page.sizeHint()
        size = size.expandedTo(page.minimumSize())
        return QtCore.QSize(max(0, size.width()) + 8,
                            max(360, size.height()) + self.tabBar().sizeHint().height() + 8)

    def minimumSizeHint(self):
        return self._page_hint(minimum=True)

    def sizeHint(self):
        return self._page_hint()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        page = self.currentWidget()
        if page is None:
            return self.sizeHint().height()
        height = page.heightForWidth(max(0, width-8)) if page.hasHeightForWidth() else page.sizeHint().height()
        return max(360, height, page.minimumSizeHint().height()) + self.tabBar().sizeHint().height() + 8


class AnimalPeakDialog(QtWidgets.QDialog):
    def __init__(self, parent, source, markers, matches, defaults):
        super().__init__(parent)
        self.setWindowTitle("测试：AnimalStart 前后最大峰 / 最小谷对齐平均")
        flags = QtCore.Qt.WindowType if hasattr(QtCore.Qt, "WindowType") else QtCore.Qt
        self.setWindowFlags(flags.Window | flags.WindowTitleHint | flags.WindowSystemMenuHint
                            | flags.WindowMinimizeButtonHint | flags.WindowMaximizeButtonHint
                            | flags.WindowCloseButtonHint)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(800, 600)
        self.resize(1300, 900)
        screen = self.screen()
        if screen is not None:
            area = screen.availableGeometry()
            self.resize(min(1300, max(800, area.width() - 40)), min(900, max(600, area.height() - 60)))
        self.source = source
        self._base_source = source
        self._secondary_history = []
        self.markers = np.asarray(markers).copy()
        self.matches = {int(key): dict(value) for key, value in matches.items()}
        self.defaults = dict(defaults)
        self.output = None
        self.worker = None
        self._busy = False
        self._close_requested = False
        self._detail_page = 0
        self._rendering = False
        self._detail_conditioned = None
        self._detail_conditioned_channel = None
        self._fixed_y_limits = None
        self._event_y_override = None
        self._event_plot_width = None
        self._letter_before_ms = min(60000, max(0, -float(defaults["epoch_start_ms"])))
        self._parameters_dirty = False
        self._other_parameters_dirty = False
        self._window_parameters_dirty = False
        self._extraction_state = None
        self._manual_axes = {}
        self._manual_preview = None
        self._exported_manual_actions = 0
        self._exported_window_resizes = 0
        self._exported_average_selections = 0
        self._exported_secondary_filters = 0
        self._exported_letter_revision = 0
        self._updating_similarity = False
        window_layout = QtWidgets.QVBoxLayout(self)
        window_layout.setContentsMargins(8, 8, 8, 8); window_layout.setSpacing(6)
        self.page_scroll = QtWidgets.QScrollArea()
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.page_content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.page_content)
        layout.setContentsMargins(2, 2, 2, 2); layout.setSpacing(6)
        layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.page_scroll.setWidget(self.page_content)
        window_layout.addWidget(self.page_scroll, 1)
        self.parameter_toggle = QtWidgets.QPushButton("收起参数区，扩大绘图区")
        self.parameter_toggle.setCheckable(True); self.parameter_toggle.setChecked(True)
        self.parameter_toggle.setMaximumWidth(230)
        layout.addWidget(self.parameter_toggle)
        self.parameter_panel = QtWidgets.QWidget()
        parameter_layout = QtWidgets.QVBoxLayout(self.parameter_panel)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        parameter_layout.setSpacing(4)
        intro = QtWidgets.QLabel(
            "测试功能，与原 Letter/SNR 计算独立。每个有效 trial 同时取搜索区最大值和最小值；"
            "不按相似度筛选、不反转负谷、不归一化幅值。没有有效 AnimalStart 的 trial 不生成子图。"
            "强制极值对齐的平均峰不能单独证明存在稳定 VEP。"
        )
        intro.setWordWrap(True); parameter_layout.addWidget(intro)
        metadata = source.metadata
        channel_scope = "第三页已完成对齐" if defaults.get("channel_selection_source") == "completed_alignment" else "当前已选"
        source_label = QtWidgets.QLabel(
            f"数据：{Path(metadata.path).name} | {metadata.fs:g} Hz | {channel_scope} {len(defaults['channel_ids'])} 通道\n"
            f"沿用 Letter Epoch {defaults['epoch_start_ms']:g}～{defaults['epoch_end_ms']:g} ms；"
            f"基线 {defaults['baseline_start_ms']:g}～{defaults['baseline_end_ms']:g} ms（相对 Letter）。"
            "截取片段可跨 AnimalStart，并可按需读取旧 Epoch 外仍存在的源数据。"
        )
        source_label.setWordWrap(True); parameter_layout.addWidget(source_label)
        self.parameter_box = QtWidgets.QGroupBox("测试参数（不回写原任务参数）")
        form = QtWidgets.QGridLayout(self.parameter_box)
        form.setContentsMargins(8, 6, 8, 6); form.setHorizontalSpacing(6); form.setVerticalSpacing(4)
        self.search_spin = self._spin(300, .1, 60000)
        self.search_spin.setToolTip("向前搜索若超出本 Trial 的 Letter Epoch，自动从 Epoch 起点开始，不因此排除 Trial；例如 Epoch 起点为 -500 ms 时，仍包含 Letter 开始前 500 ms。")
        self.search_after_spin = self._spin(float(defaults.get("search_after_ms", 0)), 0, 60000)
        self.search_after_spin.setToolTip("将找最大峰/最小谷的范围扩展到 AnimalStart 之后；0 表示只搜索之前。不是特征点后的截取长度。")
        self.pre_spin = self._spin(50, 0, 60000)
        self.post_spin = self._spin(50, 0, 60000)
        self.channel_edit = QtWidgets.QLineEdit()
        self.channel_edit.setPlaceholderText("留空使用当前已选 LFP 通道；或输入实际通道号：1,3,22")
        aligned_channels = defaults.get("channel_selection_source") == "completed_alignment"
        if aligned_channels:
            self.channel_edit.setText(",".join(map(str, defaults["channel_ids"])))
            self.channel_edit.setReadOnly(True)
            self.channel_edit.setToolTip("来自第三页最近一次完成的时间对齐。要修改通道，请在第三页重新选择并计算对齐，再重新打开测试窗口；不读取 LFP 页的独立通道选择。")
        self.tag_edit = QtWidgets.QLineEdit(",".join(str(x) for x in defaults.get("tags", [])))
        self.tag_edit.setPlaceholderText("留空=全部 Letter Tag 4/5/6/7")
        self.limit_spin = QtWidgets.QSpinBox(); self.limit_spin.setRange(0, 1000000)
        self.limit_spin.setMaximumWidth(145)
        self.limit_spin.setValue(0)
        self.limit_spin.setSpecialValueText("全部")
        self.limit_spin.setToolTip("测试默认使用全部 Trial，不继承 Letter 的每刺激 Trial 和对比 A/B 上限；可手动设置正整数。")
        for column, (label, widget) in enumerate((("AnimalStart 前搜索 ms", self.search_spin),
                                                ("AnimalStart 后搜索 ms", self.search_after_spin),
                                                ("特征点前截取 ms", self.pre_spin), ("特征点后截取 ms", self.post_spin))):
            form.addWidget(QtWidgets.QLabel(label), 1, column*2); form.addWidget(widget, 1, column*2+1)
        form.addWidget(QtWidgets.QLabel("对齐通道（第三页）" if aligned_channels else "实际通道号"), 3, 0); form.addWidget(self.channel_edit, 3, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("Tag"), 3, 4); form.addWidget(self.tag_edit, 3, 5, 1, 3)
        form.addWidget(QtWidgets.QLabel("每 Tag Trial 上限"), 4, 0); form.addWidget(self.limit_spin, 4, 1)
        self.processing_check = QtWidgets.QCheckBox("沿用任务额外分析处理（陷波/带通/平滑/Z-score；可能需读整条通道）")
        self.processing_check.setChecked(True); form.addWidget(self.processing_check, 4, 2, 1, 6)
        self.localization_box = QtWidgets.QComboBox()
        self.localization_box.addItem("原波形极值定位（旧方式）", "raw")
        self.localization_box.addItem("去毛刺＋趋势线定位（平均载入波形）", "trend")
        self.despike_spin = self._spin(5, 0, 1000)
        self.mad_spin = self._spin(6, .1, 100)
        self.trend_spin = self._spin(20, .1, 1000)
        form.addWidget(QtWidgets.QLabel("定位方式"), 0, 0)
        form.addWidget(self.localization_box, 0, 1, 1, 7)
        for column, (text, widget) in enumerate((("去毛刺窗 ms（0关闭）", self.despike_spin),
                                               ("异常阈值 MAD 倍数", self.mad_spin),
                                               ("趋势平滑窗 ms", self.trend_spin))):
            form.addWidget(QtWidgets.QLabel(text), 2, column * 2)
            form.addWidget(widget, 2, column * 2 + 1)
        self.localization_hint = QtWidgets.QLabel()
        self.localization_hint.setWordWrap(True); form.addWidget(self.localization_hint, 5, 0, 1, 8)
        parameter_layout.addWidget(self.parameter_box)
        layout.addWidget(self.parameter_panel)
        self.parameter_toggle.toggled.connect(self._toggle_parameters)
        self.secondary_box = QtWidgets.QGroupBox("选定通道二次滤波（当前 H5 完整时长；不覆盖源文件）")
        secondary_layout = QtWidgets.QGridLayout(self.secondary_box)
        self.secondary_channels = QtWidgets.QLineEdit()
        self.secondary_channels.setPlaceholderText("留空=当前结果通道；多个实际通道号用逗号分隔")
        self.secondary_mode = QtWidgets.QComboBox()
        for text, mode in (("自定义陷波", "notch"), ("高通", "highpass"), ("低通", "lowpass"), ("带通", "bandpass")):
            self.secondary_mode.addItem(text, mode)
        max_hz = max(.01, float(metadata.fs)/2-.001)
        self.secondary_frequency = self._spin(1, .001, max_hz); self.secondary_frequency.setDecimals(3)
        self.secondary_q = self._spin(30, .1, 10000); self.secondary_q.setDecimals(2)
        self.secondary_low = self._spin(1, .001, max_hz); self.secondary_low.setDecimals(3)
        self.secondary_high = self._spin(min(300, max_hz*.8), .001, max_hz); self.secondary_high.setDecimals(3)
        self.secondary_order = QtWidgets.QSpinBox(); self.secondary_order.setRange(1, 10); self.secondary_order.setValue(3)
        secondary_layout.addWidget(QtWidgets.QLabel("实际通道"), 0, 0)
        secondary_layout.addWidget(self.secondary_channels, 0, 1, 1, 5)
        secondary_layout.addWidget(QtWidgets.QLabel("方式"), 0, 6); secondary_layout.addWidget(self.secondary_mode, 0, 7)
        for column, (text, widget) in enumerate((("陷波 Hz", self.secondary_frequency), ("陷波 Q", self.secondary_q),
                                               ("高通/带通下限 Hz", self.secondary_low), ("低通/带通上限 Hz", self.secondary_high))):
            secondary_layout.addWidget(QtWidgets.QLabel(text), 1, column*2)
            secondary_layout.addWidget(widget, 1, column*2+1)
        self.secondary_apply = QtWidgets.QPushButton("应用二次滤波并更新结果")
        self.secondary_restore = QtWidgets.QPushButton("恢复选定通道滤波前数据")
        self.secondary_apply.setEnabled(False); self.secondary_restore.setEnabled(False)
        secondary_layout.addWidget(QtWidgets.QLabel("高/低/带通阶数"), 2, 0)
        secondary_layout.addWidget(self.secondary_order, 2, 1)
        secondary_layout.addWidget(self.secondary_apply, 2, 2, 1, 3)
        secondary_layout.addWidget(self.secondary_restore, 2, 5, 1, 3)
        self.secondary_status = QtWidgets.QLabel("尚未启用。先提取结果，再选通道应用；同一通道的所有 Tag/Trial 一起更新，已确认点位保留。1 Hz 陷波只抑制该频率附近，1 Hz 高通衰减低频成分。")
        self.secondary_status.setWordWrap(True); secondary_layout.addWidget(self.secondary_status, 3, 0, 1, 8)
        layout.addWidget(self.secondary_box)
        self.secondary_apply.clicked.connect(lambda: self.start_secondary_filter(False))
        self.secondary_restore.clicked.connect(lambda: self.start_secondary_filter(True))
        self.secondary_mode.currentIndexChanged.connect(self._secondary_mode_changed)
        self._secondary_mode_changed()
        row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("提取最大峰 / 最小谷并平均")
        self.run_button.setObjectName("primaryAction")
        self.cancel_button = QtWidgets.QPushButton("取消当前操作"); self.cancel_button.setEnabled(False)
        self.cancel_button.setMinimumWidth(120); self.cancel_button.setMinimumHeight(32)
        self.export_button = QtWidgets.QPushButton("保存完整结果（图 + 明细 + 数据）"); self.export_button.setEnabled(False)
        self.individual_check = QtWidgets.QCheckBox("同时保存每个 trial 的独立 PNG"); self.individual_check.setChecked(True)
        for widget in (self.run_button, self.export_button, self.individual_check):
            row.addWidget(widget)
        layout.addLayout(row)
        progress_row = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        progress_row.addWidget(self.progress, 1); progress_row.addWidget(self.cancel_button)
        window_layout.addLayout(progress_row)
        self.status = QtWidgets.QLabel("设置长度后点击提取；最大值、最小值分开平均，相关系数仅供查看。")
        self.status.setWordWrap(True); window_layout.addWidget(self.status)
        selection = QtWidgets.QHBoxLayout()
        self.result_tag_box = QtWidgets.QComboBox()
        self.result_channel_box = QtWidgets.QComboBox()
        self.result_channel_box.setEditable(True)
        self.result_channel_box.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.result_channel_box.lineEdit().setPlaceholderText("通道号 / ch22")
        self.result_channel_box.setToolTip("只切换本次已计算的结果；支持输入实际通道号。要增加计算通道，请在第三页重新选择并计算对齐，再重新运行测试。")
        self._selected_result_channel = None
        self.result_tag_box.setEnabled(False); self.result_channel_box.setEnabled(False)
        self.result_tag_label = QtWidgets.QLabel("结果 Tag")
        self.result_channel_label = QtWidgets.QLabel("结果通道 ch")
        for label in (self.result_tag_label, self.result_channel_label):
            label.setStyleSheet("color: #c62828; font-weight: bold;")
        selector_style = """
            QComboBox { color: #c62828; font-weight: bold;
                background: #fff7f7; border: 1px solid #e57373; border-radius: 5px; }
            QComboBox:focus { border: 2px solid #c62828; }
            QComboBox:disabled { color: #a87575; background: #f5f5f5; border-color: #d8baba; }
            QComboBox QLineEdit { color: #c62828; font-weight: bold; background: transparent; border: none; }
            QComboBox QAbstractItemView { color: #c62828;
                selection-color: white; selection-background-color: #c62828; }
        """
        for box, width in ((self.result_tag_box, 140), (self.result_channel_box, 180)):
            box.setFixedWidth(width); box.setMinimumHeight(32); box.setStyleSheet(selector_style)
        selection.addWidget(self.result_tag_label); selection.addWidget(self.result_tag_box)
        selection.addSpacing(14)
        selection.addWidget(self.result_channel_label); selection.addWidget(self.result_channel_box)
        selection.addStretch(1)
        self.summary = QtWidgets.QLabel(); self.summary.setWordWrap(True)
        self.y_axis_box = QtWidgets.QGroupBox("统一波形 Y 轴（只影响显示与导出图，不改变数据）")
        y_row = QtWidgets.QHBoxLayout(self.y_axis_box)
        self.y_min_spin = QtWidgets.QDoubleSpinBox()
        self.y_max_spin = QtWidgets.QDoubleSpinBox()
        for spin, value in ((self.y_min_spin, -100000), (self.y_max_spin, 100000)):
            spin.setRange(-1e12, 1e12); spin.setDecimals(4); spin.setValue(value)
            spin.setMinimumWidth(125)
        self.y_unit_label = QtWidgets.QLabel("μV")
        self.y_fixed_button = QtWidgets.QPushButton("应用固定 Y 轴（所有通道）")
        self.y_auto_button = QtWidgets.QPushButton("恢复自适应")
        self.y_axis_status = QtWidgets.QLabel("当前：自适应")
        for widget in (QtWidgets.QLabel("下限"), self.y_min_spin, QtWidgets.QLabel("上限"), self.y_max_spin,
                       self.y_unit_label, self.y_fixed_button, self.y_auto_button, self.y_axis_status):
            y_row.addWidget(widget)
        layout.addWidget(self.y_axis_box)
        self.y_fixed_button.clicked.connect(self.apply_fixed_y_axis)
        self.y_auto_button.clicked.connect(self.restore_auto_y_axis)
        self.letter_view_box = QtWidgets.QGroupBox("Letter 红线前的查看范围（不改变搜索或平均）")
        letter_view_row = QtWidgets.QHBoxLayout(self.letter_view_box)
        self.letter_view_spin = self._spin(self._letter_before_ms, 0, 60000)
        self.letter_view_spin.setToolTip("所有通道、Tag、Trial 统一从 Letter 开始前指定毫秒数显示；可读取 Epoch 之前仍存在的数据。仅用于查看和导出，不重找点、不改变基线/平均/相似度。")
        self.letter_view_button = QtWidgets.QPushButton("刷新所有 Trial 显示范围")
        self.letter_view_status = QtWidgets.QLabel(f"当前：Letter 前 {self._letter_before_ms:g} ms；源数据不足时显示到数据起点。")
        self.letter_view_status.setWordWrap(True)
        letter_view_row.addWidget(QtWidgets.QLabel("Letter 开始前显示 ms"))
        letter_view_row.addWidget(self.letter_view_spin)
        letter_view_row.addWidget(self.letter_view_button)
        letter_view_row.addWidget(self.letter_view_status, 1)
        layout.addWidget(self.letter_view_box)
        layout.addLayout(selection)
        layout.addWidget(self.summary)
        self.letter_view_button.clicked.connect(self.apply_letter_display_range)
        self.letter_view_spin.valueChanged.connect(lambda *_: self.letter_view_status.setText(
            f"待刷新：输入 {self.letter_view_spin.value():g} ms；当前已应用 Letter 前 {self._letter_before_ms:g} ms。"))
        self.tabs = CurrentPageTabs(); layout.addWidget(self.tabs, 1)
        self.average_plots = self._plot_tab("峰 / 谷叠加平均", 1, 2)
        average_layout = self.tabs.widget(0).layout()
        for column, plot in enumerate(self.average_plots):
            average_layout.removeWidget(plot); average_layout.addWidget(plot, 1, column)
        average_toolbar = QtWidgets.QHBoxLayout()
        self.average_refresh_button = QtWidgets.QPushButton("刷新叠加平均图")
        self.average_refresh_button.setEnabled(False)
        self.average_refresh_button.setToolTip("保留已应用点位刷新叠加平均。若只修改特征点前/后截取长度，按新长度重取原波形；不重新找点、不清除人工调整。橙色待确认点需先应用。")
        self.average_refresh_status = QtWidgets.QLabel("尚未提取结果。")
        self.average_refresh_status.setWordWrap(True)
        average_toolbar.addWidget(self.average_refresh_button)
        self.average_trial_button = QtWidgets.QPushButton("选择参与平均的 Trial…")
        self.average_trial_button.setEnabled(False)
        average_toolbar.addWidget(self.average_trial_button)
        average_toolbar.addWidget(self.average_refresh_status, 1)
        average_layout.addLayout(average_toolbar, 0, 0, 1, 2)
        self.histogram_plots = self._plot_tab("原始峰时 / 谷时分布", 1, 2)
        details = QtWidgets.QWidget(); details_layout = QtWidgets.QVBoxLayout(details)
        details_layout.setContentsMargins(6, 6, 6, 6); details_layout.setSpacing(6)
        navigation = QtWidgets.QHBoxLayout()
        self.previous_button = QtWidgets.QPushButton("上一页")
        self.next_button = QtWidgets.QPushButton("下一页")
        self.page_label = QtWidgets.QLabel()
        self.page_label.setWordWrap(True)
        self.details_per_page_box = QtWidgets.QComboBox()
        self.details_per_page_box.addItem("每页 4 图（与 PDF 一致）", 4)
        self.details_per_page_box.addItem("每页 1 图（放大查看）", 1)
        self.detail_curve_box = QtWidgets.QComboBox()
        self.detail_curve_box.setEnabled(False)
        for text, value in (("原波形＋定位趋势", "overlay"), ("仅原波形", "raw"),
                            ("仅定位趋势（无原始尖刺）", "trend"), ("仅去毛刺曲线", "despiked")):
            self.detail_curve_box.addItem(text, value)
        navigation.addWidget(self.previous_button); navigation.addWidget(self.next_button)
        navigation.addWidget(self.details_per_page_box); navigation.addWidget(self.page_label, 1)
        details_layout.addLayout(navigation)
        curve_row = QtWidgets.QHBoxLayout()
        curve_row.addWidget(QtWidgets.QLabel("显示曲线")); curve_row.addWidget(self.detail_curve_box)
        curve_row.addStretch(1); details_layout.addLayout(curve_row)
        self.manual_box = QtWidgets.QGroupBox("人工调整对齐点（只改变截取时间，不改变原波形）")
        manual_layout = QtWidgets.QGridLayout(self.manual_box)
        self.manual_pick_check = QtWidgets.QCheckBox("图上点选（需确认）")
        self.manual_trial_box = QtWidgets.QComboBox()
        self.manual_branch_box = QtWidgets.QComboBox()
        self.manual_branch_box.addItem("峰分支", "peak"); self.manual_branch_box.addItem("谷分支", "trough")
        self.manual_time_spin = QtWidgets.QDoubleSpinBox()
        self.manual_time_spin.setRange(-1000000, 1000000); self.manual_time_spin.setDecimals(3)
        self.manual_time_spin.setSuffix(" ms"); self.manual_time_spin.setMinimumWidth(130)
        self.manual_apply_button = QtWidgets.QPushButton("应用点位并更新平均")
        self.manual_restore_button = QtWidgets.QPushButton("恢复该点自动位置")
        self.manual_average_check = QtWidgets.QCheckBox("显示最新平均响应线（已应用点位）")
        self.manual_average_check.setChecked(True)
        self.manual_average_view_button = QtWidgets.QPushButton("放大查看最新平均响应")
        self.manual_average_view_button.setToolTip("切换到峰/谷叠加平均页；直接显示已应用的最新结果，不重新自动找点，不清除人工点位。橙色待确认点须先点击应用。")
        self.manual_hint = QtWidgets.QLabel("选择 Trial 和峰/谷分支，再点图或输入相对 AnimalStart 的时间；橙线仅为待确认位置，不自动吸附极值。")
        self.manual_hint.setWordWrap(True)
        for column, widget in enumerate((self.manual_pick_check, self.manual_trial_box, self.manual_branch_box,
                                         self.manual_time_spin, self.manual_apply_button, self.manual_restore_button)):
            manual_layout.addWidget(widget, 0, column)
        manual_layout.addWidget(self.manual_hint, 1, 0, 1, 6)
        manual_layout.addWidget(self.manual_average_check, 2, 0, 1, 3)
        manual_layout.addWidget(self.manual_average_view_button, 2, 3, 1, 3)
        details_layout.addWidget(self.manual_box)
        self.manual_box.setEnabled(False)
        with rc_context(PLOT_STYLE):
            self.detail_figure = Figure(figsize=(12, 10), layout="constrained")
        self.detail_canvas = FigureCanvasQTAgg(self.detail_figure)
        self.detail_canvas.mpl_connect("button_press_event", self._pick_manual_point)
        self.detail_canvas.setMinimumWidth(600); self.detail_canvas.setFixedHeight(680)
        self.detail_axes = []
        self.detail_toolbar = NavigationToolbar2QT(self.detail_canvas, details)
        self.detail_toolbar.setToolTip("Home：恢复视图；Pan：拖动；Zoom：框选放大；Save：保存当前显示页。")
        details_layout.addWidget(self.detail_toolbar)
        details_layout.addWidget(self.detail_canvas)
        self.manual_average_box = QtWidgets.QGroupBox("最新平均响应（峰、谷分别对齐；只使用已应用点位）")
        preview_layout = QtWidgets.QVBoxLayout(self.manual_average_box)
        self.manual_average_summary = QtWidgets.QLabel("应用点位后，这里的平均响应线将自动刷新；无需重新提取。")
        self.manual_average_summary.setWordWrap(True); preview_layout.addWidget(self.manual_average_summary)
        preview_plots = QtWidgets.QHBoxLayout()
        self.manual_average_plots = []
        for _ in range(2):
            plot = self._plot(); plot.setFixedHeight(220)
            preview_plots.addWidget(plot, 1); self.manual_average_plots.append(plot)
        preview_layout.addLayout(preview_plots)
        details_layout.addWidget(self.manual_average_box)
        self.tabs.addTab(details, "逐 Trial 找波与截取详情")
        self.table = QtWidgets.QTableWidget(0, 32)
        self.table.setHorizontalHeaderLabels(["原Trial", "Tag内Trial", "Tag", "通道", "峰/谷", "状态", "AnimalStart s", "极值相对 ms", "幅值", "旧原波形模板r（不筛选）", "标记", "具体排除原因",
                                             "validity", "StartFrame", "AnimalStartFrame", "AnimalStart原值", "AnimalStart相对Letter ms", "检测TXT行号", "排除代码",
                                             "定位方式", "定位曲线幅值", "原极值相对ms（仅对照）", "搜索区去毛刺点数",
                                             "点位来源", "自动初始位置ms", "人工操作次数",
                                             "Dot CSV M列 s", "Dot 持续 s", "Tag100 起点相对Letter ms",
                                             "Tag100 终点相对Letter ms", "Tag100 状态/提示", "Tag100 CSV行号"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers if hasattr(QtWidgets.QAbstractItemView, "EditTrigger") else QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.table, "全部 Trial 明细 / 排除原因")
        self.animal_reference_plots = self._plot_tab("AnimalStart 直接对齐对照", 1, 1)
        self._build_similarity_tab()
        self._build_letter_vep_tab()
        self.run_button.clicked.connect(self.start_analysis)
        self.cancel_button.clicked.connect(self.cancel)
        self.export_button.clicked.connect(self.start_export)
        self.result_tag_box.currentIndexChanged.connect(self._result_tag_changed)
        self.result_channel_box.currentIndexChanged.connect(self._result_channel_changed)
        self.result_channel_box.lineEdit().editingFinished.connect(self._commit_result_channel)
        self.tabs.currentChanged.connect(self._result_tab_changed)
        self.previous_button.clicked.connect(lambda: self._change_page(-1))
        self.next_button.clicked.connect(lambda: self._change_page(1))
        self.details_per_page_box.currentIndexChanged.connect(self._detail_layout_changed)
        self.detail_curve_box.currentIndexChanged.connect(lambda *_: self.render_details())
        self.manual_trial_box.currentIndexChanged.connect(self._manual_selection_changed)
        self.manual_branch_box.currentIndexChanged.connect(self._manual_selection_changed)
        self.manual_time_spin.valueChanged.connect(self._preview_manual_point)
        self.manual_apply_button.clicked.connect(lambda: self.start_manual_adjustment(False))
        self.manual_restore_button.clicked.connect(lambda: self.start_manual_adjustment(True))
        self.manual_average_check.toggled.connect(self.manual_average_box.setVisible)
        self.manual_average_check.toggled.connect(self._update_page_geometry)
        self.manual_average_view_button.clicked.connect(self.show_latest_average)
        self.average_refresh_button.clicked.connect(self.refresh_average_plot)
        self.average_trial_button.clicked.connect(self.choose_average_trials)
        self.localization_box.currentIndexChanged.connect(self._localization_changed)
        for widget in (self.search_spin, self.search_after_spin, self.pre_spin, self.post_spin, self.limit_spin,
                       self.despike_spin, self.mad_spin, self.trend_spin):
            widget.valueChanged.connect(self.parameters_changed)
        for widget in (self.channel_edit, self.tag_edit):
            widget.textChanged.connect(self.parameters_changed)
        self.processing_check.toggled.connect(self.parameters_changed)
        self._localization_changed()

    @staticmethod
    def _spin(value, low, high):
        spin = QtWidgets.QDoubleSpinBox(); spin.setRange(low, high); spin.setDecimals(1); spin.setValue(value)
        spin.setMinimumWidth(85); spin.setMaximumWidth(145)
        return spin

    @staticmethod
    def _plot():
        plot = pg.PlotWidget(); plot.setBackground("w"); plot.showGrid(x=True, y=True, alpha=.15)
        plot.setMinimumHeight(180)
        return plot

    def _sync_y_axis_display(self):
        unit = "μV" if self.output is None or self.output["units"] == "mV" else "z"
        self.y_unit_label.setText(unit)
        display = {"y_axis_mode": "auto", "y_unit": unit, "letter_before_ms": self._letter_before_ms}
        if self._fixed_y_limits is not None:
            low, high = self._fixed_y_limits
            display.update(y_axis_mode="fixed", y_min=low, y_max=high)
            self.y_axis_status.setText(f"当前：固定 [{low:g}, {high:g}] {unit}")
        else:
            self.y_axis_status.setText("当前：自适应")
        if self._event_y_override is not None:
            display["event_vep_y_axis"] = {**self._event_y_override, "y_unit": unit}
        if self._event_plot_width is not None:
            display["event_vep_plot"] = self._event_plot_display_settings()
        if hasattr(self, "letter_vep_y_status"):
            event_display = display.get("event_vep_y_axis", display)
            self.letter_vep_y_unit.setText(unit)
            self.letter_vep_y_status.setText(
                f"当前：固定 [{event_display['y_min']:g}, {event_display['y_max']:g}] {unit}"
                if event_display["y_axis_mode"] == "fixed" else "当前：自适应")
        if self.output is not None:
            self.output["display_settings"] = display

    def _apply_waveform_y_ranges(self, plots=None):
        if plots is None:
            plots = self.average_plots + self.manual_average_plots + self.animal_reference_plots + [self.letter_vep_plot]
        for plot in plots:
            view = plot.getViewBox()
            limits = self._fixed_y_limits
            if plot is self.letter_vep_plot and self._event_y_override is not None:
                limits = ((self._event_y_override["y_min"], self._event_y_override["y_max"])
                          if self._event_y_override["y_axis_mode"] == "fixed" else None)
            if limits is None:
                view.setLimits(yMin=None, yMax=None, minYRange=None, maxYRange=None)
                view.setMouseEnabled(y=True); view.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
                view.updateAutoRange()
            else:
                low, high = limits
                view.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
                view.setLimits(yMin=low, yMax=high, minYRange=high-low, maxYRange=high-low)
                view.setMouseEnabled(y=False); view.setYRange(low, high, padding=0)

    def apply_letter_display_range(self):
        if self._busy:
            return
        self._letter_before_ms = self.letter_view_spin.value()
        self._sync_y_axis_display()
        self.letter_view_status.setText(f"当前：Letter 前 {self._letter_before_ms:g} ms，所有 Trial 统一；源数据不足时显示到数据起点。")
        if self.output is not None and self._current_pair() is not None:
            pending = self.manual_time_spin.value() if self._manual_preview is not None else None
            self.render_details()
            if pending is not None:
                self.manual_time_spin.setValue(pending)
                self._preview_manual_point()
        self._sync_event_plot_size()

    def apply_fixed_y_axis(self):
        if self._busy:
            return
        low, high = self.y_min_spin.value(), self.y_max_spin.value()
        if low >= high:
            QtWidgets.QMessageBox.warning(self, "Y 轴范围无效", "下限必须小于上限；尚未改变当前显示范围。")
            return
        self._fixed_y_limits = (low, high)
        self._refresh_y_axis_display()

    def restore_auto_y_axis(self):
        if self._busy:
            return
        self._fixed_y_limits = None
        self._refresh_y_axis_display()

    def apply_event_y_axis(self):
        if self._busy:
            return
        low, high = self.letter_vep_y_min.value(), self.letter_vep_y_max.value()
        if low >= high:
            QtWidgets.QMessageBox.warning(self, "Y 轴范围无效", "下限必须小于上限；尚未改变当前显示范围。")
            return
        self._event_y_override = dict(y_axis_mode="fixed", y_min=low, y_max=high)
        self._sync_y_axis_display()
        self._apply_waveform_y_ranges([self.letter_vep_plot])
        self._sync_event_plot_size()

    def restore_event_auto_y_axis(self):
        if self._busy:
            return
        self._event_y_override = dict(y_axis_mode="auto")
        self._sync_y_axis_display()
        self._apply_waveform_y_ranges([self.letter_vep_plot])
        self._sync_event_plot_size()

    def _event_plot_display_settings(self):
        return dict(width_mode="manual", requested_width_px=self._event_plot_width,
                    width_px=self.letter_vep_plot.width(), height_px=self.letter_vep_plot.height())

    def apply_event_plot_width(self):
        if self._busy:
            return
        self._event_plot_width = self.letter_vep_width.value()
        self._sync_event_plot_size()
        self._sync_y_axis_display()

    def restore_event_auto_width(self):
        if self._busy:
            return
        self._event_plot_width = None
        self._sync_event_plot_size()
        self._sync_y_axis_display()

    def _refresh_y_axis_display(self):
        self._sync_y_axis_display()
        self._apply_waveform_y_ranges()
        if self.output is not None and self._current_pair() is not None:
            self.render_details()

    def _plot_tab(self, title, rows, columns):
        page = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(page); plots = []
        for index in range(rows*columns):
            plot = self._plot(); grid.addWidget(plot, index//columns, index%columns); plots.append(plot)
        self.tabs.addTab(page, title)
        return plots

    def _secondary_mode_changed(self, *_):
        mode = self.secondary_mode.currentData()
        self.secondary_frequency.setEnabled(mode == "notch"); self.secondary_q.setEnabled(mode == "notch")
        self.secondary_low.setEnabled(mode in ("highpass", "bandpass"))
        self.secondary_high.setEnabled(mode in ("lowpass", "bandpass"))
        self.secondary_order.setEnabled(mode != "notch")

    def _build_letter_vep_tab(self):
        self.letter_vep_page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.letter_vep_page)
        self.letter_vep_parameters_box = QtWidgets.QGroupBox("事件对齐参数（独立于峰谷页）")
        grid = QtWidgets.QGridLayout(self.letter_vep_parameters_box)
        self.letter_vep_start = self._spin(0, -60000, 60000)
        self.letter_vep_end = self._spin(6000, -60000, 60000)
        self.letter_vep_baseline = QtWidgets.QCheckBox("启用基线校正"); self.letter_vep_baseline.setChecked(True)
        self.letter_vep_base_start = self._spin(-1000, -60000, 0)
        self.letter_vep_base_end = self._spin(0, -60000, 0)
        for column, (label, widget) in enumerate((("截取起点 ms", self.letter_vep_start), ("截取终点 ms", self.letter_vep_end),
                ("基线起点 ms", self.letter_vep_base_start), ("基线终点 ms", self.letter_vep_base_end))):
            grid.addWidget(QtWidgets.QLabel(label), 0, column*2); grid.addWidget(widget, 0, column*2+1)
        grid.addWidget(self.letter_vep_baseline, 1, 0, 1, 2)
        self.letter_vep_hint = QtWidgets.QLabel("红线LetterStart=0；基线相对LetterStart；要求有效AnimalStart；三种对齐共用Trial名单，不使用峰谷点或趋势线。")
        self.letter_vep_hint.setWordWrap(True); grid.addWidget(self.letter_vep_hint, 1, 2, 1, 6)
        layout.addWidget(self.letter_vep_parameters_box)
        toolbar = QtWidgets.QHBoxLayout()
        self.letter_vep_refresh = QtWidgets.QPushButton("刷新事件对齐平均图")
        self.letter_vep_trials = QtWidgets.QPushButton("选择参与平均的 Trial…")
        self.letter_vep_refresh.setEnabled(False); self.letter_vep_trials.setEnabled(False)
        toolbar.addWidget(self.letter_vep_refresh); toolbar.addWidget(self.letter_vep_trials)
        toolbar.addWidget(QtWidgets.QLabel("对齐方式"))
        self.letter_vep_alignment = QtWidgets.QComboBox()
        for key, text in (("letter", "LetterStart（之后）"), ("dot", "Dot（之后）"), ("animal", "AnimalStart（之前）")):
            self.letter_vep_alignment.addItem(text, key)
        self.letter_vep_alignment.setToolTip("切换时按当前总长度设为事件后0～长度，或AnimalStart前−长度～0；点击刷新计算。Dot仅使用CSV同行M列。三种方式共用Trial名单。")
        toolbar.addWidget(self.letter_vep_alignment)
        toolbar.addWidget(QtWidgets.QLabel("响应总长度 ms"))
        self.letter_vep_duration = self._spin(6000, .1, 60000)
        self.letter_vep_duration.lineEdit().setPlaceholderText("起止范围无效")
        self.letter_vep_duration.setToolTip("总长度=截取终点−起点。修改时保持起点不变、同步终点；点击刷新后重新截取并平均，不只是缩放坐标轴。保留Trial名单，不改变峰谷页。")
        toolbar.addWidget(self.letter_vep_duration)
        self.letter_vep_status = QtWidgets.QLabel("先提取测试结果，再刷新；Trial名单与峰谷页独立。")
        self.letter_vep_status.setWordWrap(True); toolbar.addWidget(self.letter_vep_status, 1)
        layout.addLayout(toolbar)
        self.letter_vep_y_box = QtWidgets.QWidget()
        y_row = QtWidgets.QHBoxLayout(self.letter_vep_y_box)
        y_row.setContentsMargins(0, 0, 0, 0)
        y_row.addWidget(QtWidgets.QLabel("纵轴显示范围"))
        self.letter_vep_y_min = self._spin(-100, -1e12, 1e12)
        self.letter_vep_y_max = self._spin(100, -1e12, 1e12)
        self.letter_vep_y_unit = QtWidgets.QLabel("μV")
        self.letter_vep_y_apply = QtWidgets.QPushButton("应用固定纵轴")
        self.letter_vep_y_auto = QtWidgets.QPushButton("恢复自适应")
        self.letter_vep_y_status = QtWidgets.QLabel("当前：自适应")
        for widget in (QtWidgets.QLabel("下限"), self.letter_vep_y_min, QtWidgets.QLabel("上限"),
                       self.letter_vep_y_max, self.letter_vep_y_unit, self.letter_vep_y_apply,
                       self.letter_vep_y_auto, self.letter_vep_y_status):
            y_row.addWidget(widget)
        y_row.addStretch(1)
        self.letter_vep_y_box.setToolTip("默认自适应；未单独设置时沿用全局纵轴。此处应用后仅影响事件对齐页及其导出图，三种对齐及各通道/Tag共用；不会裁剪或重新计算数据。")
        self.letter_vep_y_apply.clicked.connect(self.apply_event_y_axis)
        self.letter_vep_y_auto.clicked.connect(self.restore_event_auto_y_axis)
        layout.addWidget(self.letter_vep_y_box)
        self.letter_vep_width_box = QtWidgets.QWidget()
        width_row = QtWidgets.QHBoxLayout(self.letter_vep_width_box)
        width_row.setContentsMargins(0, 0, 0, 0)
        width_row.addWidget(QtWidgets.QLabel("手动图框宽度 px"))
        self.letter_vep_width = QtWidgets.QSpinBox()
        self.letter_vep_width.setRange(260, 6000); self.letter_vep_width.setValue(600)
        self.letter_vep_width.setMaximumWidth(145)
        self.letter_vep_width_apply = QtWidgets.QPushButton("应用图框宽度")
        self.letter_vep_width_auto = QtWidgets.QPushButton("恢复自动参考宽度")
        for widget in (self.letter_vep_width, self.letter_vep_width_apply, self.letter_vep_width_auto):
            width_row.addWidget(widget)
        width_row.addStretch(1)
        self.letter_vep_width_box.setToolTip("默认参考逐Trial自动计算宽度；手动输入后点击应用。宽度按逻辑像素设置，受页面可用宽度及文字最小可读宽度限制；只改变图框，不改变横轴时间范围、纵轴范围和平均数据。")
        self.letter_vep_width_apply.clicked.connect(self.apply_event_plot_width)
        self.letter_vep_width_auto.clicked.connect(self.restore_event_auto_width)
        layout.addWidget(self.letter_vep_width_box)
        self.letter_vep_plot = self._plot()
        self.letter_vep_plot.setFixedSize(480, 340)
        self.letter_vep_plot.getAxis("bottom").enableAutoSIPrefix(False)
        plot_row = QtWidgets.QHBoxLayout()
        plot_row.addWidget(self.letter_vep_plot, 0, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        plot_row.addStretch(1); layout.addLayout(plot_row)
        self.letter_vep_size_status = QtWidgets.QLabel()
        self.letter_vep_size_status.setWordWrap(True); layout.addWidget(self.letter_vep_size_status)
        self._event_plot_resize_timer = QtCore.QTimer(self)
        self._event_plot_resize_timer.setSingleShot(True)
        self._event_plot_resize_timer.timeout.connect(self._sync_event_plot_size)
        self.letter_vep_table = QtWidgets.QTableWidget(0, 11)
        self.letter_vep_table.setHorizontalHeaderLabels(["原Trial", "Tag内Trial", "选择", "实际纳入", "Letter时间 s", "AnimalStart相对Letter ms", "基线均值", "排除原因", "对齐方式", "对齐事件时间 s", "基线参考事件"])
        self.letter_vep_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.letter_vep_table.setMinimumHeight(180); layout.addWidget(self.letter_vep_table)
        self.tabs.addTab(self.letter_vep_page, "事件对齐 VEP（Letter / Dot / Animal）")
        self._letter_selector_active = False
        for widget in (self.letter_vep_start, self.letter_vep_end, self.letter_vep_base_start, self.letter_vep_base_end):
            widget.valueChanged.connect(self._letter_vep_parameters_changed)
        self.letter_vep_baseline.toggled.connect(self._letter_vep_parameters_changed)
        self.letter_vep_alignment.currentIndexChanged.connect(self._letter_vep_alignment_changed)
        self.letter_vep_duration.valueChanged.connect(self._letter_vep_duration_changed)
        self.letter_vep_duration.editingFinished.connect(lambda: self._letter_vep_duration_changed(self.letter_vep_duration.value()))
        self.letter_vep_refresh.clicked.connect(self.refresh_letter_vep)
        self.letter_vep_trials.clicked.connect(self.choose_letter_vep_trials)

    def _letter_vep_config(self):
        return letter_vep.parameters(dict(alignment=self.letter_vep_alignment.currentData(), start_ms=self.letter_vep_start.value(), end_ms=self.letter_vep_end.value(),
            baseline_enabled=self.letter_vep_baseline.isChecked(), baseline_start_ms=self.letter_vep_base_start.value(),
            baseline_end_ms=self.letter_vep_base_end.value()))

    def _letter_vep_parameters_changed(self, *_):
        mode = self.letter_vep_alignment.currentData()
        self.letter_vep_duration.blockSignals(True)
        maximum = (self.letter_vep_end.value()-self.letter_vep_start.minimum()) if mode == "animal" else (self.letter_vep_end.maximum()-self.letter_vep_start.value())
        self.letter_vep_duration.setMaximum(max(.1, maximum))
        width = self.letter_vep_end.value()-self.letter_vep_start.value()
        if width > 0:
            self.letter_vep_duration.setValue(width)
        else:
            self.letter_vep_duration.setValue(.1)
            self.letter_vep_duration.clear()
        self.letter_vep_duration.setEnabled(maximum >= .1)
        self.letter_vep_duration.blockSignals(False)
        enabled = self.letter_vep_baseline.isChecked()
        self.letter_vep_base_start.setEnabled(enabled); self.letter_vep_base_end.setEnabled(enabled)
        reference = letter_vep.baseline_reference({"alignment": mode})
        self.letter_vep_baseline.setText(f"启用基线校正（相对{reference}）")
        self.letter_vep_hint.setText(f"零点={letter_vep.ALIGNMENTS[mode]}；基线相对{reference}；要求有效AnimalStart；三种对齐共用名单，不使用峰谷点或趋势线。")
        self.letter_vep_duration.setToolTip("总长度=终点−起点；AnimalStart保持终点、向前调整起点，其余保持起点、向后调整终点。点击刷新重新截取并平均，保留共用Trial名单，不改变峰谷页。")
        self._render_letter_vep()

    def _letter_vep_duration_changed(self, value):
        if self.letter_vep_alignment.currentData() == "animal":
            self.letter_vep_start.setValue(self.letter_vep_end.value()-value)
        else:
            self.letter_vep_end.setValue(self.letter_vep_start.value()+value)

    def _letter_vep_alignment_changed(self, *_):
        length = self.letter_vep_duration.value()
        animal = self.letter_vep_alignment.currentData() == "animal"
        self.letter_vep_start.blockSignals(True); self.letter_vep_end.blockSignals(True)
        self.letter_vep_start.setMaximum(0 if animal else 60000)
        self.letter_vep_end.setMaximum(0 if animal else 60000)
        self.letter_vep_start.setValue(-length if animal else 0)
        self.letter_vep_end.setValue(0 if animal else length)
        self.letter_vep_start.blockSignals(False); self.letter_vep_end.blockSignals(False)
        self._letter_vep_parameters_changed()

    def _selector_pairs(self):
        if self.output is not None and self.tabs.currentWidget() is self.letter_vep_page:
            state = self.output.get("letter_vep") or letter_vep.new_state(self.markers, self.matches)
            return dict.fromkeys(letter_vep.available_pairs(self.output, state))
        return getattr(self, "_pairs", {})

    def refresh_letter_vep(self, *_, trial_ids="unchanged"):
        if self._busy or self.output is None or self._current_pair() is None:
            return
        try:
            config = self._letter_vep_config()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "事件对齐参数无效", str(exc)); return
        self._set_busy(True); self.progress.setValue(0)
        self.status.setText(f"正在按{letter_vep.ALIGNMENTS[config['alignment']]}截取并平均；保留全部峰谷点位与共用名单，取消不改变旧结果。")
        self.worker = AnimalPeakWorker(self.source, self.markers, self.matches, config, self,
            letter=(self.output, self._current_pair(), trial_ids))
        self._wire_worker(self.worker)
        self.worker.completed.connect(lambda output, worker=self.worker: self._accept_completion(worker, self._letter_vep_finished, output))
        self.worker.start()

    def _letter_vep_finished(self, output):
        self.output = output
        self._render_letter_vep()
        self.export_button.setEnabled(True); self.progress.setValue(100)
        self.status.setText("事件对齐VEP已更新；Letter/Dot/Animal共用名单，峰谷点位、选择和相似度不变。请保存完整结果。")

    def choose_letter_vep_trials(self):
        if self._busy or self.output is None:
            return
        state = self.output.get("letter_vep") or {}
        group = state.get("cache")
        if group is None or (group["channel"], group["tag"]) != self._current_pair():
            QtWidgets.QMessageBox.information(self, "请先刷新", "请先刷新当前通道/Tag/对齐方式的VEP，再选择共用Trial名单。")
            return
        try:
            if self._letter_vep_config() != state["parameters"]:
                raise ValueError("参数或对齐方式已改，请先刷新事件对齐VEP，再选择Trial。")
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "请先刷新", str(exc)); return
        dialog = AverageTrialSelectionDialog(self, self.output, self._current_pair(), letter_group=group)
        try:
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self.refresh_letter_vep(trial_ids=dialog.selected_ids())
        finally:
            dialog.deleteLater()

    def _render_letter_vep(self):
        if not hasattr(self, "letter_vep_plot"):
            return
        plot = self.letter_vep_plot; plot.clear(); self.letter_vep_table.setRowCount(0)
        state = self.output.get("letter_vep") if self.output is not None else None
        group = state.get("cache") if state else None
        if group is None or state.get("stale") or (group["channel"], group["tag"]) != self._current_pair() or group["parameters"].get("alignment", "letter") != self.letter_vep_alignment.currentData():
            plot.setTitle("当前通道/Tag/对齐方式尚未计算或信号已更新，请刷新事件对齐VEP")
            self.letter_vep_status.setText("点击刷新读取当前通道/Tag；各组选择独立保留，保存时计算全部通道/Tag。")
            self._sync_event_plot_size()
            return
        scale, unit = waveform_display(self.output)
        indices = group["selected_indices"]; t = group["time_ms"]
        for index in indices:
            plot.plot(t, group["waves"][index]*scale, pen=pg.mkPen((120,130,140,65), width=.7))
        if indices: plot.plot(t, group["mean"]*scale, pen=pg.mkPen("#b32025", width=2.2))
        plot.addItem(pg.InfiniteLine(0, pen=pg.mkPen("#c62828", width=1.2)))
        plot.setXRange(float(t[0]), float(t[-1]), padding=0)
        name = letter_vep.ALIGNMENTS[group["parameters"].get("alignment", "letter")]
        plot.setLabel("bottom", f"相对 {name}", units="ms"); plot.setLabel("left", unit)
        plot.setTitle(f"ch{group['channel']} | Tag {group['tag']} | n={len(indices)}<br>{name} 对齐 VEP" + ("（无所选可用Trial）" if not indices else ""))
        try: pending = self._letter_vep_config() != state["parameters"]
        except ValueError: pending = True
        cfg = group["parameters"]
        baseline = f"基线相对{letter_vep.baseline_reference(cfg)} {cfg['baseline_start_ms']:g}～{cfg['baseline_end_ms']:g} ms" if cfg["baseline_enabled"] else "基线校正关闭"
        self.letter_vep_status.setText(f"截取 {cfg['start_ms']:g}～{cfg['end_ms']:g} ms；{baseline}；实际平均 {len(indices)}/{len(group['rows'])} 个可用Trial。"
            + ("参数待刷新；当前显示上一次完成的结果。" if pending else "三种对齐共用名单，与峰谷页独立；灰线=参与Trial，红线=等权平均。"))
        rows = group["records"]; self.letter_vep_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = [row["trial_id"], row["tag_trial_id"], "是" if row["average_selected"] else "否",
                "是" if row["average_included"] else "否", f"{row['marker_sec']:.6f}", self._number(row["animal_latency_ms"]),
                self._number(row["baseline_correction"]*scale), row["reason"], name,
                self._number(row["alignment_sec"],6), row["baseline_reference"]]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value)); item.setToolTip(str(value)); self.letter_vep_table.setItem(index,column,item)
        self.letter_vep_table.resizeColumnsToContents(); self._apply_waveform_y_ranges()
        self._sync_event_plot_size()

    def _sync_event_plot_size(self):
        if not hasattr(self, "letter_vep_plot"):
            return
        per_page = int(self.details_per_page_box.currentData())
        columns = 2 if per_page == 4 else 1
        available = max(280, self.page_scroll.viewport().width()-32)
        cell_width = max(260, (available-12)/columns)
        cell_height = 340 if per_page == 4 else 400
        span, reference_trial = 6500., None
        duration = self.letter_vep_duration.value()
        pair = self._current_pair()
        if self.output is not None and pair is not None:
            span, reference_trial = letter_vep.reference_span_ms(self.source, self.output, pair, self.manual_trial_box.currentData())
            # If the reference subplot was rendered, use its real x range,
            # including the user's zoom. Keep frame geometry independent of
            # Matplotlib's asynchronous constrained-layout passes.
            for axis, rows in self._manual_axes.items():
                if rows and (rows[0]["channel"],rows[0]["tag"],rows[0]["trial_id"]) == (*pair,reference_trial):
                    span = abs(float(np.diff(axis.get_xlim())[0]))
                    break
            group = self.output.get("letter_vep", {}).get("cache")
            if group is not None and not self.output["letter_vep"].get("stale") and (group["channel"],group["tag"]) == pair and group["parameters"].get("alignment","letter") == self.letter_vep_alignment.currentData():
                duration = float(np.ptp(group["time_ms"]))
        self.letter_vep_plot.plotItem.layout.invalidate()
        self.letter_vep_plot.plotItem.layout.activate()
        readable_width = np.ceil(self.letter_vep_plot.centralWidget.effectiveSizeHint(QtCore.Qt.SizeHint.MinimumSize).width())
        width, height = letter_vep.compact_plot_size(duration, span, cell_width, cell_height,
                                                  minimum_width=max(260, readable_width))
        if self._event_plot_width is not None:
            width = min(available, max(260, readable_width, self._event_plot_width))
        width, height = int(round(width)), int(round(height))
        if self.letter_vep_plot.size() != QtCore.QSize(width,height):
            self.letter_vep_plot.setFixedSize(width,height)
        # Title changes can temporarily retain the previous scene minimum size.
        # Reflow before fitting the scene so a compact widget does not clip its
        # right-hand data, title, or time axis.
        self.letter_vep_plot.plotItem.layout.invalidate()
        self.letter_vep_plot.plotItem.layout.activate()
        self.letter_vep_plot.centralWidget.setGeometry(QtCore.QRectF(self.letter_vep_plot.rect()))
        reference = f"原Trial {reference_trial}" if reference_trial is not None else "逐Trial单图布局"
        if self._event_plot_width is None:
            description = f"自动：图框参考{reference}中约{duration:g} ms的显示宽度：{width}×{height}；短窗保留最小可读宽度。"
        else:
            description = f"手动：请求宽度{self._event_plot_width} px，实际图框{width}×{height}；受页面可用宽度及最小可读宽度限制。"
        self.letter_vep_size_status.setText(description + "仅调整图框，时间范围、计算片段、基线及Trial名单不变。")
        if self.output is not None:
            display = self.output.setdefault("display_settings", {})
            if self._event_plot_width is None:
                display.pop("event_vep_plot", None)
            else:
                display["event_vep_plot"] = self._event_plot_display_settings()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self,"_event_plot_resize_timer"):
            self._event_plot_resize_timer.start(0)

    def start_secondary_filter(self, restore=False):
        if self._busy or self.output is None:
            return
        if self._parameters_dirty:
            QtWidgets.QMessageBox.warning(self, "提取参数待刷新", "请先应用/恢复待修改的提取参数，再进行二次滤波；不会重新自动找点。")
            return
        try:
            text = self.secondary_channels.text().replace("，", ",").replace(";", ",")
            channels = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
            if not channels and self._current_pair() is not None:
                channels = [self._current_pair()[0]]
            if not channels or not set(channels) <= set(self.output["settings"]["channel_ids"]):
                raise ValueError("请输入本次已提取的实际通道号；留空使用当前结果通道。")
            config = None if restore else {
                "mode": self.secondary_mode.currentData(), "frequency_hz": self.secondary_frequency.value(),
                "q": self.secondary_q.value(), "low_hz": self.secondary_low.value(),
                "high_hz": self.secondary_high.value(), "order": self.secondary_order.value()}
            if config is not None:
                config, _sos = filter_config(config, self.source.metadata.fs)
            elif not isinstance(self.source, SecondaryFilteredSource) or not any(ch in self.source.channels for ch in channels):
                raise ValueError("选定通道尚未进行二次滤波，无需恢复。")
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "二次滤波参数无效", str(exc)); return
        cache_mb = self.source.metadata.rows*4*len(channels)/1024**2
        action = "恢复载入时的数据" if restore else "从载入时的数据重新滤波（不叠加上次二次滤波）"
        answer = QtWidgets.QMessageBox.question(self, "确认完整时长通道处理",
            f"通道：{','.join(map(str,channels))}\n范围：当前 H5 全部 {self.source.metadata.rows/self.source.metadata.fs:.3f} 秒，各 Tag/Trial 一起更新。\n"
            f"将{action}，保留已有自动/人工点位和平均Trial选择，重新计算基线及平均；判断窗相似度需重算。\n"
            + ("" if restore else f"临时通道缓存约 {cache_mb:.1f} MB，另需一条通道的双向计算工作文件；恢复/关闭后自动释放。\n")
            + "不覆盖源 H5，不改变主页面数据。是否继续？",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._secondary_view_state = (self._current_pair(), self._detail_page, self.manual_trial_box.currentData())
        self._set_busy(True); self.progress.setValue(0)
        self.status.setText("正在逐通道完整时长处理；取消时保留上一份结果。")
        self.worker = AnimalPeakWorker(self.source, None, None, {}, self, secondary=(self.output, channels, config))
        self._wire_worker(self.worker)
        self.worker.completed.connect(lambda value, worker=self.worker:
            self._accept_completion(worker, self._secondary_finished, value))
        self.worker.start()

    def _secondary_finished(self, value):
        pair, page, trial = self._secondary_view_state
        self.source, output = value
        self._secondary_history = list(output.get("secondary_filter_history", []))
        self._detail_conditioned = None; self._detail_conditioned_channel = None
        self.analysis_finished(output)
        self._restore_result_pair(pair); self._detail_page = page; self.render_details()
        index = self.manual_trial_box.findData(trial)
        if index >= 0:
            self.manual_trial_box.setCurrentIndex(index)
        self.status.setText("整段通道处理完成，点位及Trial选择保留，基线/曲线/平均已更新。自动点仍是原定位时间，不重新取极值；判断窗相似度需重算。请保存完整结果。")

    def _sync_secondary_status(self):
        state = self.source.state() if isinstance(self.source, SecondaryFilteredSource) else []
        if self.output is not None:
            self.output["secondary_filter_state"] = state
            self.output["secondary_filter_history"] = list(self._secondary_history)
        descriptions = []
        for item in state:
            mode = item["mode"]
            value = (f"陷波 {item['frequency_hz']:g} Hz / Q={item['q']:g}" if mode == "notch"
                     else f"高通 {item['low_hz']:g} Hz" if mode == "highpass"
                     else f"低通 {item['high_hz']:g} Hz" if mode == "lowpass"
                     else f"带通 {item['low_hz']:g}～{item['high_hz']:g} Hz")
            descriptions.append(f"ch{item['channel']}：{value}")
        short = sum(item["short_finite_spans_unchanged"] for item in state)
        self.secondary_status.setText(("当前完整时长二次滤波：" + "；".join(descriptions) if descriptions else "当前未启用二次滤波，使用载入的数据。")
                                     + (f"；缺失值分段中有 {short} 个过短有效段保持原值（已记录）。" if short else ""))

    def _build_similarity_tab(self):
        page = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(page)
        self.similarity_parameter_box = QtWidgets.QGroupBox("响应判断窗（独立于平均截取窗；只比较趋势形状，不筛选平均）")
        grid = QtWidgets.QGridLayout(self.similarity_parameter_box)
        self.judge_before_spin = self._spin(50, 0, 60000)
        self.judge_after_spin = self._spin(50, 0, 60000)
        self.judge_r_spin = self._spin(.7, 0, 1); self.judge_r_spin.setDecimals(3); self.judge_r_spin.setSingleStep(.05)
        self.judge_fraction_spin = self._spin(60, .1, 100)
        self.judge_coverage_spin = self._spin(80, .1, 100)
        self.judge_reference_coverage_spin = self._spin(80, .1, 100)
        fields = (("较早点前延伸 ms", self.judge_before_spin), ("较晚点后延伸 ms", self.judge_after_spin),
                  ("形状阈值 r ≥", self.judge_r_spin), ("参考一致率 ≥ %", self.judge_fraction_spin),
                  ("两窗最低覆盖率 %", self.judge_coverage_spin), ("参考最低有效比较率 %", self.judge_reference_coverage_spin))
        for index, (label, widget) in enumerate(fields):
            grid.addWidget(QtWidgets.QLabel(label), index//3, index%3*2)
            grid.addWidget(widget, index//3, index%3*2+1)
            widget.valueChanged.connect(self._similarity_parameters_changed)
        self.similarity_button = QtWidgets.QPushButton("计算所有通道 / Tag 的判断窗相似度")
        self.similarity_button.setObjectName("primaryAction"); self.similarity_button.setEnabled(False)
        grid.addWidget(self.similarity_button, 2, 0, 1, 6)
        self.similarity_button.clicked.connect(self.start_similarity)
        layout.addWidget(self.similarity_parameter_box)
        hint = QtWidgets.QLabel("较早点对齐后按真实毫秒比较共同数据；不拉伸、不滑动优化、不取绝对值。参考本人不计入分母；不可比较项单列。自动参考来自本组，仅为探索性形状重复性，不证明刺激锁时或幅值稳定。")
        hint.setWordWrap(True); layout.addWidget(hint)
        self.similarity_summary = QtWidgets.QLabel("请先使用趋势定位提取峰谷，可人工调整两点，再点击计算。")
        self.similarity_summary.setWordWrap(True); layout.addWidget(self.similarity_summary)
        choice = QtWidgets.QHBoxLayout()
        self.similarity_reference_box = QtWidgets.QComboBox(); self.similarity_reference_box.setEnabled(False)
        self.similarity_trial_box = QtWidgets.QComboBox(); self.similarity_trial_box.setEnabled(False)
        choice.addWidget(QtWidgets.QLabel("本组参考")); choice.addWidget(self.similarity_reference_box, 1)
        choice.addWidget(QtWidgets.QLabel("查看匹配排名的 Trial")); choice.addWidget(self.similarity_trial_box, 1)
        layout.addLayout(choice)
        self.similarity_views = QtWidgets.QTabWidget(); layout.addWidget(self.similarity_views, 1)
        self.similarity_table = QtWidgets.QTableWidget(0, 10)
        self.similarity_table.setHorizontalHeaderLabels(["原Trial", "Tag内Trial", "相似数", "可比较数", "相似比例 %", "中位数r", "最佳匹配Trial", "最高r", "与参考r", "不可比较原因"])
        self.similarity_rank_table = QtWidgets.QTableWidget(0, 8)
        self.similarity_rank_table.setHorizontalHeaderLabels(["排名", "对方原Trial", "对方Tag内Trial", "趋势r", "达形状阈值", "有效长度ms", "最低覆盖率%", "不可比较原因"])
        for table in (self.similarity_table, self.similarity_rank_table):
            table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.similarity_views.addTab(self.similarity_table, "逐 Trial 汇总")
        self.similarity_views.addTab(self.similarity_rank_table, "所选 Trial 完整排名")
        with rc_context(PLOT_STYLE):
            self.similarity_figure = Figure(figsize=(7, 5), layout="constrained")
        self.similarity_canvas = FigureCanvasQTAgg(self.similarity_figure)
        self.similarity_views.addTab(self.similarity_canvas, "两两相似度热图")
        self.similarity_reference_box.currentIndexChanged.connect(self._similarity_reference_changed)
        self.similarity_trial_box.currentIndexChanged.connect(self._render_similarity_rankings)
        self.similarity_table.cellClicked.connect(self._similarity_row_clicked)
        self.tabs.addTab(page, "响应判断窗相似度")

    def _similarity_parameters(self):
        return {"before_ms": self.judge_before_spin.value(), "after_ms": self.judge_after_spin.value(),
                "r_threshold": self.judge_r_spin.value(), "required_fraction": self.judge_fraction_spin.value()/100,
                "min_window_coverage": self.judge_coverage_spin.value()/100,
                "min_reference_coverage": self.judge_reference_coverage_spin.value()/100}

    def _similarity_parameters_changed(self, *_):
        if self.output is not None and not self._busy:
            current = self.output.get("judgment_similarity")
            if current:
                self.output = {**self.output, "judgment_similarity": {**current, "stale": True}}
            self._render_similarity()
            if self.tabs.currentIndex() == 2:
                self.render_details()

    def start_similarity(self):
        if self._busy or self.output is None:
            return
        if self._parameters_dirty or self.output["settings"].get("localization_mode") != "trend":
            QtWidgets.QMessageBox.warning(self, "需要趋势提取结果", "请先选择趋势定位并重新提取峰谷，再计算判断窗相似度。")
            return
        self._set_busy(True); self.progress.setValue(0)
        self.status.setText("正在计算判断窗趋势线两两相似度；不修改平均，不筛选 Trial。")
        self.worker = AnimalPeakWorker(self.source, None, None, self._similarity_parameters(), self, similarity=self.output)
        self._wire_worker(self.worker)
        self.worker.completed.connect(lambda output, worker=self.worker:
                                      self._accept_completion(worker, self._similarity_finished, output))
        self.worker.start()

    def _similarity_finished(self, output):
        self.output = output
        self.render_pair()
        self.progress.setValue(100)
        self.status.setText("判断窗相似度已更新；保存完整结果将同时导出逐 Trial 汇总、完整排名、矩阵、热图及 HTML 报告。")

    def _current_similarity_group(self):
        result = self.output.get("judgment_similarity") if self.output else None
        if not result or result.get("stale") or self._parameters_dirty:
            return None
        return next((g for g in result["groups"] if (g["channel"], g["tag"]) == self._current_pair()), None)

    @staticmethod
    def _fill_table(table, rows):
        table.setRowCount(len(rows))
        for i, values in enumerate(rows):
            for j, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value)); item.setToolTip(str(value))
                table.setItem(i, j, item)
        table.resizeColumnsToContents()

    def _render_similarity(self):
        self._updating_similarity = True
        try:
            previous = self.similarity_trial_box.currentData()
            self.similarity_trial_box.clear(); self.similarity_reference_box.clear()
            group = self._current_similarity_group()
            self.similarity_trial_box.setEnabled(group is not None and not self._busy)
            self.similarity_reference_box.setEnabled(group is not None and not self._busy)
            if group is None:
                current = self.output.get("judgment_similarity") if self.output else None
                self.similarity_summary.setText("判断窗相似度待重算：参数或点位已变化；旧分数不显示，也不作为当前结果导出。"
                                                if current else "尚未计算本组的判断窗相似度；先使用趋势定位提取峰谷，再点击计算。")
                self.similarity_table.setRowCount(0); self.similarity_rank_table.setRowCount(0)
                self.similarity_figure.clear(); self.similarity_canvas.draw_idle()
                return
            result = self.output["judgment_similarity"]
            params = result["parameters"]
            percent = lambda value: self._number(value*100, 1)
            count = len(group["trials"])
            criterion = "达到当前参考一致率规则（不等于全部Trial彼此一致）" if group["reference_threshold_met"] else "未达到当前参考一致率规则或有效比较不足"
            self.similarity_summary.setText(
                f"参考原 Trial {group['reference_trial_id']}（{'人工指定' if group['reference_mode']=='manual' else '自动推荐'}）；"
                f"参考一致率 {group['similar_count']}/{group['comparable_count']}＝{percent(group['reference_fraction'])}%；"
                f"有效比较率 {group['comparable_count']}/{max(0,count-1)}＝{percent(group['reference_coverage'])}%；"
                f"不可比较 {max(0,count-1-group['comparable_count'])} 个。\n"
                f"形状阈值 r≥{params['r_threshold']:g}；一致率要求≥{params['required_fraction']*100:g}%；"
                f"最低有效比较率≥{params['min_reference_coverage']*100:g}%。{criterion}。\n"
                f"匹配组含参考 {len(group['matched_trial_ids'])}/{count} 个 Trial（{percent(group['matched_group_fraction'])}%）；"
                f"匹配组（含参考）内部中位数 r={self._number(group['internal_median_r'])}；"
                f"内部达标配对比例 {percent(group['internal_similar_pair_fraction'])}%；"
                f"内部有效配对覆盖率 {percent(group['internal_comparison_coverage'])}%。")
            self.similarity_reference_box.addItem("自动推荐（相似伙伴最多）", None)
            for trial in group["trials"]:
                label = f"原 Trial {trial['trial_id']} / Tag 内 {trial['tag_trial_id']}"
                self.similarity_trial_box.addItem(label, trial["trial_id"])
                if not trial.get("reason"):
                    self.similarity_reference_box.addItem(label, trial["trial_id"])
            if group["reference_mode"] == "manual":
                self.similarity_reference_box.setCurrentIndex(self.similarity_reference_box.findData(group["reference_trial_id"]))
            index = self.similarity_trial_box.findData(previous)
            self.similarity_trial_box.setCurrentIndex(index if index >= 0 else 0)
            ref = next((i for i, t in enumerate(group["trials"]) if t["trial_id"] == group["reference_trial_id"]), None)
            self._fill_table(self.similarity_table, [(
                row["trial_id"], row["tag_trial_id"], row["similar_count"], row["comparable_count"],
                percent(row["similar_fraction"]), self._number(row["median_r"]), row["best_trial_id"], self._number(row["best_r"]),
                "参考本人，不评分" if ref == i else self._number(group["matrix"][i, ref]) if ref is not None else "—", row["reason"])
                for i, row in enumerate(group["trial_summaries"])])
            draw_similarity_heatmap(self.similarity_figure, group)
            self.similarity_canvas.draw_idle()
        finally:
            self._updating_similarity = False
        self._render_similarity_rankings()

    def _render_similarity_rankings(self, *_):
        if self._updating_similarity:
            return
        group = self._current_similarity_group()
        trial_id = self.similarity_trial_box.currentData()
        if group is None or trial_id is None:
            return
        threshold = self.output["judgment_similarity"]["parameters"]["r_threshold"]
        self._fill_table(self.similarity_rank_table, [(
            row["rank"], row["other_trial_id"], row["other_tag_trial_id"], self._number(row["r"]),
            "—" if not np.isfinite(row["r"]) else "是" if row["r"] >= threshold else "否",
            self._number(row["effective_ms"], 2), self._number(row["minimum_window_coverage"]*100, 1), row["reason"])
            for row in ranked_matches(group, trial_id, self.output["fs"])])

    def _similarity_row_clicked(self, row, _column):
        item = self.similarity_table.item(row, 0)
        if item is not None:
            self.similarity_trial_box.setCurrentIndex(self.similarity_trial_box.findData(int(item.text())))
            self.similarity_views.setCurrentIndex(1)

    def _similarity_reference_changed(self, *_):
        if self._updating_similarity or self._busy:
            return
        group = self._current_similarity_group()
        if group is None:
            return
        result = self.output["judgment_similarity"]
        changed = summarize_group(group, result["parameters"], self.similarity_reference_box.currentData())
        self.output = {**self.output, "judgment_similarity": {**result,
            "groups": [changed if g is group else g for g in result["groups"]]}}
        self._render_similarity()
        self.render_pair()

    def _extraction_control_state(self):
        """Changes other than pre/post excerpts require an explicit new extraction."""
        return (self.search_spin.value(), self.search_after_spin.value(), self.channel_edit.text(),
                self.tag_edit.text(), self.limit_spin.value(), self.localization_box.currentData(),
                self.despike_spin.value(), self.mad_spin.value(), self.trend_spin.value(),
                self.processing_check.isChecked())

    def _has_unsaved_edits(self):
        return self.output is not None and (
            len(self.output.get("manual_alignment_history", [])) > self._exported_manual_actions
            or len(self.output.get("window_resize_history", [])) > self._exported_window_resizes
            or len(self.output.get("average_selection_history", [])) > self._exported_average_selections
            or len(self.output.get("secondary_filter_history", [])) > self._exported_secondary_filters
            or self.output.get("letter_vep", {}).get("revision", 0) > self._exported_letter_revision)

    def parameters_changed(self, *_):
        if self.output is not None and not self._busy:
            self._other_parameters_dirty = self._extraction_state != self._extraction_control_state()
            self._window_parameters_dirty = (
                self.pre_spin.value() != self.output["settings"]["pre_ms"]
                or self.post_spin.value() != self.output["settings"]["post_ms"])
            self._parameters_dirty = self._other_parameters_dirty or self._window_parameters_dirty
            has_edits = bool(self.output.get("manual_alignment_history") or self.output.get("window_resize_history") or self.output.get("average_selection_history") or self.output.get("secondary_filter_history") or self.output.get("letter_vep"))
            self.export_button.setEnabled(not self._parameters_dirty or has_edits)
            self.secondary_apply.setEnabled(not self._parameters_dirty)
            self.secondary_restore.setEnabled(not self._parameters_dirty and isinstance(self.source, SecondaryFilteredSource))
            self.manual_box.setEnabled(not self._parameters_dirty and self.manual_trial_box.count() > 0)
            if self._other_parameters_dirty:
                self.status.setText("搜索/定位/处理等参数已修改；保留点位刷新只支持修改特征点前/后截取长度。请恢复其他参数，或先保存后重新提取（重新找点会重置人工位置）。")
            elif self._window_parameters_dirty:
                self.status.setText("截取参数已修改（特征点前/后长度）。请到“峰 / 谷叠加平均”点击“刷新叠加平均图”，保留全部已确认点位，按新长度更新原波形平均；当前仍显示上一份结果。")
            else:
                self.status.setText("参数已恢复到当前结果，可继续人工调整或刷新图。")
            self.similarity_button.setEnabled(not self._parameters_dirty and self.output["settings"].get("localization_mode") == "trend")
            self._render_similarity()
            self._render_manual_average_preview()
            self._manual_selection_changed()

    def _toggle_parameters(self, expanded):
        self.parameter_panel.setVisible(expanded)
        self.parameter_toggle.setText("收起参数区，扩大绘图区" if expanded else "展开参数区")
        self._update_page_geometry()

    def _update_page_geometry(self, *_):
        self.tabs.updateGeometry()
        self.page_content.layout().activate()
        self.page_content.updateGeometry()

    def _result_tab_changed(self, index):
        active = self.tabs.currentWidget() is self.letter_vep_page
        if active != self._letter_selector_active and self.output is not None:
            self._letter_selector_active = active
            pairs = self._selector_pairs(); old_tag = self.result_tag_box.currentData()
            self.result_tag_box.blockSignals(True); self.result_tag_box.clear()
            for tag in sorted({pair[1] for pair in pairs}): self.result_tag_box.addItem(f"Tag {tag}", tag)
            selected = self.result_tag_box.findData(old_tag)
            if selected >= 0: self.result_tag_box.setCurrentIndex(selected)
            self.result_tag_box.blockSignals(False); self._refresh_result_channels()
            self.result_tag_box.setEnabled(bool(pairs) and not self._busy)
            self.result_channel_box.setEnabled(bool(pairs) and not self._busy)
            self.render_pair()
        if index == 2:
            self.render_details()
        if self.tabs.currentWidget() is self.letter_vep_page:
            self._event_plot_resize_timer.start(0)
        self._update_page_geometry()

    def _localization_changed(self, *_, mark_changed=True):
        trend = self.localization_box.currentData() == "trend"
        for widget in (self.despike_spin, self.mad_spin, self.trend_spin):
            widget.setEnabled(trend)
        self.processing_check.setEnabled(not trend)
        self.localization_hint.setText(
            "趋势定位：局部中位数/MAD 去毛刺，再以居中二阶 Savitzky–Golay 平滑找正峰/负谷；仅定位，不改变载入波形及平均。5/6/20 为试用初值，不代表最优参数。"
            if trend else "旧方式保持原有计算；选择趋势定位可减少孤立尖刺对极值位置的影响。")
        if mark_changed:
            self.parameters_changed()

    def _detail_layout_changed(self, *_):
        self._detail_page = 0
        self.detail_canvas.setFixedHeight(680 if self.details_per_page_box.currentData() == 4 else 400)
        self.render_details()
        self._sync_event_plot_size()
        self._update_page_geometry()

    def _set_busy(self, busy):
        self._busy = busy
        self.y_axis_box.setEnabled(not busy)
        self.letter_view_box.setEnabled(not busy)
        self.secondary_box.setEnabled(not busy)
        self.secondary_apply.setEnabled(not busy and self.output is not None and not self._parameters_dirty)
        self.secondary_restore.setEnabled(not busy and self.output is not None and not self._parameters_dirty
                                          and isinstance(self.source, SecondaryFilteredSource))
        self.parameter_box.setEnabled(not busy); self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy); self.export_button.setEnabled(
            not busy and self.output is not None
            and (not self._parameters_dirty or bool(self.output.get("manual_alignment_history") or self.output.get("window_resize_history") or self.output.get("average_selection_history") or self.output.get("secondary_filter_history") or self.output.get("letter_vep"))))
        self.cancel_button.setText("取消当前操作")
        has_results = bool(self._selector_pairs())
        self.result_tag_box.setEnabled(not busy and has_results)
        self.result_channel_box.setEnabled(not busy and has_results)
        self.average_refresh_button.setEnabled(not busy and self.output is not None and has_results)
        self.average_trial_button.setEnabled(not busy and self.output is not None and has_results)
        self.letter_vep_parameters_box.setEnabled(not busy)
        self.letter_vep_y_box.setEnabled(not busy)
        self.letter_vep_width_box.setEnabled(not busy)
        self.letter_vep_refresh.setEnabled(not busy and self.output is not None and has_results)
        self.letter_vep_trials.setEnabled(not busy and self.output is not None and has_results)
        self.tabs.setEnabled(not busy)
        self.similarity_parameter_box.setEnabled(not busy)
        self.similarity_button.setEnabled(not busy and self.output is not None and not self._parameters_dirty
                                         and self.output["settings"].get("localization_mode") == "trend")
        has_similarity = self._current_similarity_group() is not None
        self.similarity_reference_box.setEnabled(not busy and has_similarity)
        self.similarity_trial_box.setEnabled(not busy and has_similarity)
        if not busy:
            self._localization_changed(mark_changed=False)
        self.manual_box.setEnabled(not busy and self.output is not None and not self._parameters_dirty
                                   and self.manual_trial_box.count() > 0)
        if not busy:
            self._manual_selection_changed()

    def start_analysis(self):
        if self._busy:
            return
        if self._has_unsaved_edits():
            answer = QtWidgets.QMessageBox.question(
                self, "修改尚未导出", "重新提取将清除本窗口的人工点位、截取窗、Trial平均选择及LetterStart VEP，请先保存完整结果。仍要重新提取吗？",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            parse = lambda value: list(dict.fromkeys(int(part.strip()) for part in value.replace(";", ",").replace("，", ",").split(",") if part.strip()))
            settings = {**self.defaults, "search_ms": self.search_spin.value(), "pre_ms": self.pre_spin.value(), "post_ms": self.post_spin.value(),
                        "search_after_ms": self.search_after_spin.value(),
                        "localization_mode": self.localization_box.currentData(),
                        "despike_window_ms": self.despike_spin.value(), "despike_mad_threshold": self.mad_spin.value(),
                        "trend_window_ms": self.trend_spin.value(),
                        "channel_ids": list(self.defaults["channel_ids"])
                        if self.defaults.get("channel_selection_source") == "completed_alignment"
                        else parse(self.channel_edit.text()) or list(self.defaults["channel_ids"]),
                        "tags": parse(self.tag_edit.text()), "trial_limit": self.limit_spin.value(),
                        "processing": dict(self.defaults.get("processing") or {})
                        if self.processing_check.isChecked() and self.localization_box.currentData() == "raw" else {}}
            if any(tag not in (4, 5, 6, 7) for tag in settings["tags"]):
                raise ValueError("此测试只支持 Letter Tag 4/5/6/7。")
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数无效", str(exc)); return
        self.output = None  # Release the previous waveform set before computing another.
        self._exported_manual_actions = 0
        self._exported_window_resizes = 0
        self._exported_average_selections = 0
        self._exported_letter_revision = 0
        self._pairs = {}
        self._detail_conditioned = None
        self._detail_conditioned_channel = None
        for plot in self.average_plots + self.histogram_plots + self.animal_reference_plots + self.manual_average_plots + [self.letter_vep_plot]:
            plot.clear()
        self.manual_average_summary.setText("正在重新提取；完成后显示新结果。旧人工点位不用于本次自动提取。")
        self.detail_figure.clear(); self.detail_axes = []
        self.detail_canvas.draw_idle()
        self.table.setRowCount(0)
        self._set_busy(True); self.progress.setValue(0)
        self.status.setText("正在逐通道、逐 trial 提取；不读取整份 H5 到内存。")
        self.worker = AnimalPeakWorker(self.source, self.markers, self.matches, settings, self)
        self._wire_worker(self.worker)
        self.worker.completed.connect(lambda output, worker=self.worker:
                                      self._accept_completion(worker, self.analysis_finished, output))
        self.worker.start()

    def _wire_worker(self, worker):
        # The main window's existing shutdown cleanup discovers this handle.
        self.parent()._animal_peak_test_worker = worker
        if hasattr(self.parent(), "_track_lfp_operation"):
            label = "事件对齐 VEP" if worker.letter is not None else "完整时长二次滤波/恢复" if worker.secondary is not None else "保留点位刷新截取窗" if worker.resize is not None else "判断窗相似度" if worker.similarity is not None else "人工点位更新平均" if worker.manual is not None else "峰谷结果导出" if worker.export is not None else "AnimalStart 峰谷测试"
            self.parent()._track_lfp_operation(worker, label, cancel_notify=self._show_cancel_request)
        worker.progress.connect(self.update_progress)
        worker.failed.connect(self.failed)
        worker.cancelled.connect(self.cancelled)
        worker.finished.connect(self._worker_finished)

    def update_progress(self, value, text):
        if self.worker is not None and (getattr(self.worker, "_ui_cancel_requested", False)
                                        or self.worker.isInterruptionRequested()):
            return
        self.progress.setValue(int(value)); self.status.setText(text)

    def _accept_completion(self, worker, callback, value):
        if getattr(worker, "_ui_cancel_requested", False) or worker.isInterruptionRequested():
            self.cancelled("已取消测试操作；未显示本次未完成结果。")
            return
        callback(value)

    def analysis_finished(self, output):
        self.output = output
        self._sync_y_axis_display()
        self._sync_secondary_status()
        self._parameters_dirty = False
        self._other_parameters_dirty = False
        self._window_parameters_dirty = False
        self._extraction_state = self._extraction_control_state()
        self.secondary_apply.setEnabled(not self._busy)
        self.secondary_restore.setEnabled(not self._busy and isinstance(self.source, SecondaryFilteredSource))
        self._pairs = defaultdict(dict)
        for group in output["groups"]:
            self._pairs[(group["channel"], group["tag"])][group["branch"]] = group
        for group in output.get("animal_reference_groups", []):
            self._pairs.setdefault((group["channel"], group["tag"]), {})
        if output.get("window_resize_history"):
            for row in output["records"]:
                if "extreme_local_sample" in row:
                    self._pairs.setdefault((row["channel"], row["tag"]), {})
        self.detail_curve_box.setEnabled(output["settings"].get("localization_mode", "raw") == "trend")
        self.result_tag_box.blockSignals(True); self.result_tag_box.clear()
        for tag in sorted({pair[1] for pair in self._selector_pairs()}):
            self.result_tag_box.addItem(f"Tag {tag}", tag)
        self.result_tag_box.blockSignals(False)
        self._refresh_result_channels()
        self.result_tag_box.setEnabled(bool(self._selector_pairs()) and not self._busy)
        self.result_channel_box.setEnabled(bool(self._selector_pairs()) and not self._busy)
        self.letter_vep_refresh.setEnabled(bool(self._selector_pairs()) and not self._busy)
        self.letter_vep_trials.setEnabled(bool(self._selector_pairs()) and not self._busy)
        self.table.setRowCount(len(output["records"]))
        scale, unit = waveform_display(output)
        for index, row in enumerate(output["records"]):
            values = (row["trial_id"], row["tag_trial_id"], row["tag"], row["channel"], row["branch"], row["status"],
                      self._number(row.get("animal_sec"), 6), self._number(row.get("extreme_relative_ms"), 2),
                      self._number(row.get("amplitude", np.nan)*scale) + " " + unit,
                      self._number(row.get("similarity_r"), 3), row.get("flags", ""), row.get("reason", ""),
                      row.get("validity", ""), row.get("start_frame", ""), row.get("animal_start_frame", ""),
                      row.get("animal_start_raw", ""), self._number(row.get("animal_latency_ms"), 3),
                      row.get("detection_line_number", ""), row.get("reason_code", ""), row.get("localization_mode", ""),
                      self._number(row.get("localization_amplitude", np.nan) * scale),
                      self._number(row.get("raw_extreme_relative_ms")), row.get("despike_count_search", ""),
                      "人工" if row.get("alignment_origin") == "manual" else "自动" if "extreme_local_sample" in row else "未定位",
                      self._number(row.get("automatic_relative_ms", row.get("extreme_relative_ms"))), row.get("manual_edit_count", 0),
                      row.get("tag100_dot_offset_sec", ""), row.get("tag100_duration_sec", ""),
                      self._number(float(row.get("tag100_start_latency_sec", np.nan))*1000),
                      self._number(float(row.get("tag100_end_latency_sec", np.nan))*1000),
                      row.get("tag100_message", "需重新计算对齐"), row.get("tag100_csv_row", ""))
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(11, min(720, max(420, self.table.columnWidth(11))))
        self.table.resizeRowsToContents()
        self._detail_page = 0; self.render_pair()
        included = sum(row["status"] == "included" for row in output["records"])
        self.status.setText(f"测试完成：{len(output['eligible_trials'])} 个有效 AnimalStart trial；峰谷分支纳入 {included} 条记录，详情及排除原因已保留。")
        self.progress.setValue(100)
        self.similarity_button.setEnabled(not self._busy and output["settings"].get("localization_mode") == "trend")

    @staticmethod
    def _number(value, digits=3):
        return f"{float(value):.{digits}f}" if value is not None and np.isfinite(value) else "—"

    def _current_pair(self):
        tag = self.result_tag_box.currentData()
        return (self._selected_result_channel, tag) if tag is not None and self._selected_result_channel is not None else None

    def _refresh_result_channels(self, preferred=None):
        tag = self.result_tag_box.currentData()
        channels = sorted(channel for channel, result_tag in self._selector_pairs() if result_tag == tag)
        previous = self._selected_result_channel if preferred is None else preferred
        self.result_channel_box.blockSignals(True)
        self.result_channel_box.clear()
        for channel in channels:
            self.result_channel_box.addItem(f"ch{channel}", channel)
        selected = channels.index(previous) if previous in channels else 0
        self.result_channel_box.setCurrentIndex(selected if channels else -1)
        self._selected_result_channel = channels[selected] if channels else None
        self.result_channel_box.blockSignals(False)

    def _result_tag_changed(self, *_):
        self._refresh_result_channels()
        self.render_pair()

    def _result_channel_changed(self, index):
        self._selected_result_channel = self.result_channel_box.itemData(index) if index >= 0 else None
        self.render_pair()

    def _commit_result_channel(self):
        if self._busy or not self._selector_pairs():
            return
        text = self.result_channel_box.currentText().strip().lower()
        if text.startswith("ch"):
            text = text[2:].strip()
        channel = int(text) if text.isdecimal() else None
        index = self.result_channel_box.findData(channel) if channel is not None else -1
        if index < 0:
            self.result_channel_box.setEditText(f"ch{self._selected_result_channel}")
            QtWidgets.QMessageBox.warning(self, "通道无结果", "请输入当前 Tag 下已有结果的实际通道号。未计算的通道不能直接切换；请在第三页重新选择并计算对齐后，再重新运行测试。")
            return
        previous = self._selected_result_channel
        self.result_channel_box.blockSignals(True)
        self.result_channel_box.setCurrentIndex(index)
        self.result_channel_box.setEditText(f"ch{channel}")
        self.result_channel_box.blockSignals(False)
        self._selected_result_channel = channel
        if channel != previous:
            self.render_pair()

    def _restore_result_pair(self, pair):
        if pair is None:
            return
        tag_index = self.result_tag_box.findData(pair[1])
        if tag_index >= 0:
            self.result_tag_box.blockSignals(True)
            self.result_tag_box.setCurrentIndex(tag_index)
            self.result_tag_box.blockSignals(False)
            self._refresh_result_channels(preferred=pair[0])
            self.render_pair()

    def show_latest_average(self):
        """Navigation only: never rerun detection or discard manual anchors."""
        if self._busy or self.output is None:
            return
        self.tabs.setCurrentIndex(0)

    def choose_average_trials(self):
        if self._busy or self.output is None or self._current_pair() is None:
            return
        pair = self._current_pair()
        dialog = AverageTrialSelectionDialog(self, self.output, pair)
        try:
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            self.output = set_average_trial_selection(self.output, *pair, dialog.selected_ids())
            self.render_pair(reset_page=False)
            self.export_button.setEnabled(True)
            self.status.setText("已更新本组所选 Trial 的峰/谷平均、SEM及AnimalStart平均；点位和全部明细保留，全组判断窗相似度统计不变。请保存完整结果。")
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Trial选择无效", str(exc))
        finally:
            dialog.deleteLater()

    def _average_selection_text(self, pair):
        ids = average_trial_ids(self.output, *pair) if pair is not None else None
        if ids is None:
            return "平均选择：全部 Trial"
        if not ids:
            return "平均选择：0 个 Trial（不显示平均线）"
        shown = ",".join(map(str, ids[:12])) + ("…" if len(ids)>12 else "")
        return f"平均选择：{len(ids)} 个 Trial，原编号 {shown}"

    def refresh_average_plot(self):
        """Resize around existing anchors if needed; otherwise only redraw."""
        if self._busy or self.output is None or self._current_pair() is None:
            return
        if self._other_parameters_dirty:
            QtWidgets.QMessageBox.warning(self, "其他提取参数已修改", "保留人工点位刷新目前只支持“特征点前/后截取 ms”。请先恢复搜索、定位方式、趋势、通道/Tag及额外处理等其他修改，再刷新；或保存当前人工结果后重新提取。")
            return
        if self._window_parameters_dirty:
            self._resize_view_state = (self._current_pair(), self._detail_page, self.manual_trial_box.currentData())
            self._set_busy(True); self.progress.setValue(0)
            self.status.setText("正在保留全部人工/自动点位，按新前后截取长度重新读取并更新平均；取消不修改上一份结果。")
            self.worker = AnimalPeakWorker(self.source, None, None,
                {"pre_ms": self.pre_spin.value(), "post_ms": self.post_spin.value()}, self, resize=self.output)
            self._wire_worker(self.worker)
            self.worker.completed.connect(lambda output, worker=self.worker:
                self._accept_completion(worker, self._window_refresh_finished, output))
            self.worker.start()
            return
        self._pairs = defaultdict(dict)
        for group in self.output["groups"]:
            self._pairs[(group["channel"], group["tag"])][group["branch"]] = group
        for group in self.output.get("animal_reference_groups", []):
            self._pairs.setdefault((group["channel"], group["tag"]), {})
        if self.output.get("window_resize_history"):
            for row in self.output["records"]:
                if "extreme_local_sample" in row:
                    self._pairs.setdefault((row["channel"], row["tag"]), {})
        self.render_pair(reset_page=False)
        for plot in self.average_plots:
            plot.autoRange()
        self._apply_waveform_y_ranges()
        self.average_refresh_status.setText("已刷新。" + self.average_refresh_status.text())

    def _window_refresh_finished(self, output):
        pair, page, trial = self._resize_view_state
        self.analysis_finished(output)
        self._restore_result_pair(pair)
        self._detail_page = page
        self.render_details()
        index = self.manual_trial_box.findData(trial)
        if index >= 0:
            self.manual_trial_box.setCurrentIndex(index)
        for plot in self.average_plots:
            plot.autoRange()
        self._apply_waveform_y_ranges()
        excluded = sum(row.get("status") == "excluded" for row in output["records"] if "extreme_local_sample" in row)
        self.average_refresh_status.setText("已按新截取窗刷新。" + self.average_refresh_status.text())
        self.status.setText(f"已保留全部点位，更新截取窗：前 {output['settings']['pre_ms']:g} / 后 {output['settings']['post_ms']:g} ms。"
                            f"超出源数据边界的峰/谷分支 {excluded} 个，点位仍保留，详情见明细。"
                            + ("判断窗相似度待重算。" if output.get("judgment_similarity") else "") + "请保存完整结果。")

    def _render_manual_average_preview(self):
        pair = self._current_pair()
        scale, unit = waveform_display(self.output)
        labels = []
        for plot, branch, name in zip(self.manual_average_plots, ("peak", "trough"), ("峰分支", "谷分支")):
            plot.clear()
            group = selected_average_group(self.output, self._pairs.get(pair, {}).get(branch))
            if group is None or not group["rows"]:
                plot.setTitle(name + ("：未选中可用 Trial" if group is not None else "：无完整片段"))
                labels.append(name + "：0 个 Trial"); continue
            manual = sum(row.get("alignment_origin") == "manual" for row in group["rows"])
            count = len(group["rows"])
            plot.plot(group["time_ms"], group["mean"]*scale,
                      pen=pg.mkPen("#b32025" if branch == "peak" else "#16885b", width=2.5),
                      name="最新原波形平均")
            plot.addItem(pg.InfiniteLine(0, pen=pg.mkPen("#777777", style=QtCore.Qt.PenStyle.DashLine if hasattr(QtCore.Qt, "PenStyle") else QtCore.Qt.DashLine)))
            plot.setTitle(f"ch{pair[0]} | Tag {pair[1]} | {name}平均 | n={count}（人工{manual} / 自动{count-manual}）")
            plot.setLabel("bottom", "相对最终对齐点", units="ms"); plot.setLabel("left", unit)
            labels.append(f"{name}：{count} 个 Trial，人工点位 {manual} 个")
        revisions = len(self.output.get("manual_alignment_history", []))
        pending = "原提取参数已改，当前仍显示上一次已应用结果。" if self._parameters_dirty else ""
        self.manual_average_summary.setText("；".join(labels) + f"。人工操作版本 {revisions}。"
                                            f"截取窗：前 {self.output['settings']['pre_ms']:g} / 后 {self.output['settings']['post_ms']:g} ms。"
                                            + self._average_selection_text(pair) + "。曲线为所选可用 Trial 按最终点位截取的原波形等权平均；橙色待确认点不计入。" + pending)

    def render_pair(self, *_, reset_page=True):
        if not self.output or not hasattr(self, "_pairs"):
            return
        pair = self._current_pair()
        if reset_page:
            self._detail_page = 0
        scale, unit = waveform_display(self.output)
        counts = []
        for index, branch in enumerate(("peak", "trough")):
            plot, histogram = self.average_plots[index], self.histogram_plots[index]
            plot.clear(); histogram.clear()
            group = selected_average_group(self.output, self._pairs.get(pair, {}).get(branch))
            name = "最大峰" if branch == "peak" else "最小谷"
            if group is None or not group["rows"]:
                plot.setTitle(name + ("：未选中可用 Trial" if group is not None else "：无完整片段"))
                histogram.setTitle(name + "：无所选可用 Trial")
                counts.append(f"{name} n=0"); continue
            if any(row.get("alignment_origin") == "manual" for row in group["rows"]):
                name = ("峰分支" if branch == "peak" else "谷分支") + "（含人工锚点）"
            # Explicit user selection only; no hidden similarity rejection.
            for wave in group["waves"]:
                plot.plot(group["time_ms"], wave*scale, pen=pg.mkPen((120, 130, 140, 65), width=.7))
            plot.plot(group["time_ms"], group["mean"]*scale, pen=pg.mkPen("#b32025", width=2.2))
            plot.addItem(pg.InfiniteLine(0, pen=pg.mkPen("#333333", style=QtCore.Qt.PenStyle.DashLine if hasattr(QtCore.Qt, "PenStyle") else QtCore.Qt.DashLine)))
            plot.setTitle(f"ch{pair[0]} | Tag {pair[1]} | {name}对齐 | n={len(group['rows'])}")
            plot.setLabel("bottom", "相对最终对齐点", units="ms"); plot.setLabel("left", unit)
            latency = [row["extreme_relative_ms"] for row in group["rows"]]
            y, x = np.histogram(latency, bins=min(20, max(1, len(latency))))
            histogram.plot(x, y, stepMode="center", fillLevel=0, brush=(65, 119, 163, 110), pen="#4177a3")
            histogram.setTitle(name + "相对 AnimalStart 的时间分布")
            histogram.setLabel("bottom", "相对 AnimalStart", units="ms"); histogram.setLabel("left", "Trial 数")
            counts.append(f"{name} n={len(group['rows'])}")
        self.summary.setText("；".join(counts) + "。灰线=所选可用Trial，红线=等权平均；不按相似度自动筛选。")
        chosen = average_trial_ids(self.output, *pair) if pair is not None else None
        manually_aligned = sum(row.get("alignment_origin") == "manual" for row in self.output["records"]
                               if (row.get("channel"), row.get("tag")) == pair and row.get("status") == "included"
                               and (chosen is None or row["trial_id"] in chosen))
        if manually_aligned:
            self.summary.setText(self.summary.text() + f" 已采用 {manually_aligned} 个人工点位。")
        revision = len(self.output.get("manual_alignment_history", []))
        self.average_refresh_button.setEnabled(not self._busy and pair is not None)
        self.average_trial_button.setEnabled(not self._busy and pair is not None)
        self.average_refresh_status.setText(
            f"人工操作版本 {revision}，当前通道/Tag 已采用 {manually_aligned} 个人工点位。"
            f"截取窗前 {self.output['settings']['pre_ms']:g} / 后 {self.output['settings']['post_ms']:g} ms。"
            + self._average_selection_text(pair) + "。"
            "刷新显示已应用的结果，未确认的橙色预览不计入。"
            + ("原提取参数已改，此处仍为上一次完成的结果。" if self._parameters_dirty else ""))
        reference_plot = self.animal_reference_plots[0]
        reference_plot.clear()
        reference = selected_average_group(self.output, next((group for group in self.output.get("animal_reference_groups", [])
                          if (group["channel"], group["tag"]) == pair), None))
        if reference is None:
            reference_plot.setTitle("AnimalStart 直接对齐对照：选择趋势定位并重新提取后生成")
        elif not reference["rows"]:
            reference_plot.setTitle("AnimalStart 直接对齐对照：未选中可用 Trial")
        else:
            for wave in reference["waves"]:
                reference_plot.plot(reference["time_ms"], wave * scale, pen=pg.mkPen((120, 130, 140, 65), width=.7))
            reference_plot.plot(reference["time_ms"], reference["mean"] * scale, pen=pg.mkPen("#2679b5", width=2.2))
            reference_plot.setTitle(f"ch{pair[0]} | Tag {pair[1]} | AnimalStart 直接对齐（未去毛刺的载入波形） | n={len(reference['rows'])}")
            reference_plot.setLabel("bottom", "相对 AnimalStart", units="ms"); reference_plot.setLabel("left", unit)
        if self.output["settings"].get("localization_mode", "raw") == "trend":
            self.summary.setText(self.summary.text() + " 当前用趋势定位、载入波形平均；蓝色趋势线不参与平均。")
        self._render_manual_average_preview()
        self._apply_waveform_y_ranges()
        self._render_similarity()
        self._render_letter_vep()
        if self.tabs.currentIndex() == 2:
            self.render_details()

    def _change_page(self, step):
        self._detail_page = max(0, self._detail_page + step); self.render_details()

    def _selected_manual_row(self):
        if self.output is None:
            return None
        pair, trial, branch = self._current_pair(), self.manual_trial_box.currentData(), self.manual_branch_box.currentData()
        return next((row for row in self.output["records"]
                     if (row.get("channel"), row.get("tag")) == pair and row.get("trial_id") == trial
                     and row.get("branch") == branch and "extreme_local_sample" in row), None)

    def _manual_selection_changed(self, *_):
        row = self._selected_manual_row()
        ready = row is not None and not self._busy and not self._parameters_dirty
        self.manual_apply_button.setEnabled(ready)
        self.manual_restore_button.setEnabled(ready and row.get("alignment_origin") == "manual")
        if row is not None:
            self.manual_time_spin.blockSignals(True)
            self.manual_time_spin.setValue(row["extreme_relative_ms"])
            self.manual_time_spin.blockSignals(False)
            origin = "人工" if row.get("alignment_origin") == "manual" else "自动"
            self.manual_hint.setText(f"当前{origin}点位 {row['extreme_relative_ms']:.3f} ms；自动初值 {row.get('automatic_relative_ms', row['extreme_relative_ms']):.3f} ms。"
                                     "点图或输入时间后确认；橙线为待确认位置，不吸附极值。先关闭工具栏的 Pan/Zoom 再点选。")
        self._clear_manual_preview()

    def _clear_manual_preview(self):
        if self._manual_preview is not None:
            try:
                self._manual_preview.remove()
            except (ValueError, NotImplementedError):
                pass
            self._manual_preview = None
            self.detail_canvas.draw_idle()

    def _preview_manual_point(self, *_):
        self._clear_manual_preview()
        row = self._selected_manual_row()
        if row is None or self._busy or self._parameters_dirty:
            return
        for axis, rows in self._manual_axes.items():
            if rows[0]["trial_id"] == row["trial_id"]:
                self._manual_preview = axis.axvline(self.manual_time_spin.value(), color="#d77b16", linestyle="--", linewidth=1.2)
                self.detail_canvas.draw_idle()
                break

    def _pick_manual_point(self, event):
        if not self.manual_pick_check.isChecked() or self._busy or self._parameters_dirty:
            return
        if event.button != 1 or event.inaxes not in self._manual_axes or event.xdata is None:
            return
        if self.detail_toolbar.mode:
            self.manual_hint.setText("请先关闭工具栏 Pan/Zoom，再点击人工位置；本次未修改点位。")
            return
        rows = self._manual_axes[event.inaxes]
        row = rows[0]
        self.manual_trial_box.setCurrentIndex(self.manual_trial_box.findData(row["trial_id"]))
        position = row["animal_local_sample"] + int(round(event.xdata * self.output["fs"] / 1000))
        if not row["search_first"] <= position < row["search_last"]:
            self.manual_hint.setText("该点击位置超出本次搜索区，未应用。请在搜索阴影范围内选择。")
            return
        self.manual_time_spin.setValue(float(event.xdata))
        self._preview_manual_point()

    def start_manual_adjustment(self, restore=False):
        if self._busy or self.output is None or self._parameters_dirty:
            return
        row = self._selected_manual_row()
        if row is None:
            return
        request = {"channel": row["channel"], "tag": row["tag"], "trial_id": row["trial_id"],
                   "branch": row["branch"], "relative_ms": self.manual_time_spin.value(), "restore": restore}
        self._manual_view_state = (self._current_pair(), self._detail_page, row["trial_id"])
        self._set_busy(True); self.progress.setValue(0)
        self.status.setText("正在按确认的点位重新截取原波形并更新平均；完成前保留原结果。")
        self.worker = AnimalPeakWorker(self.source, None, None, {}, self, manual=(self.output, request))
        self._wire_worker(self.worker)
        self.worker.completed.connect(lambda output, worker=self.worker:
                                      self._accept_completion(worker, self._manual_finished, output))
        self.worker.start()

    def _manual_finished(self, output):
        pair, page, trial_id = self._manual_view_state
        self.analysis_finished(output)
        self._restore_result_pair(pair)
        self._detail_page = page
        self.render_details()
        trial_index = self.manual_trial_box.findData(trial_id)
        if trial_index >= 0:
            self.manual_trial_box.setCurrentIndex(trial_index)
        change = output["manual_alignment_history"][-1]
        action = "恢复自动位置" if change["action"] == "restore_automatic" else "人工调整"
        self.status.setText(f"{action}完成：Trial {trial_id} / {change['branch']}，"
                            f"{change['previous_relative_ms']:.3f} → {change['new_relative_ms']:.3f} ms。"
                            "下方最新平均响应线、SEM和明细已更新；可放大查看，无需重新自动提取。"
                            + ("判断窗相似度待重算。" if output.get("judgment_similarity") else "")
                            + "请保存完整结果。")

    def render_details(self):
        if self._rendering or not self.output:
            return
        self._rendering = True
        try:
            pair = self._current_pair()
            details = defaultdict(list)
            for row in self.output["records"]:
                if (row["channel"], row["tag"]) == pair and "extreme_local_sample" in row:
                    details[row["trial_id"]].append(row)
            trials = sorted(details)
            per_page = int(self.details_per_page_box.currentData())
            pages = max(1, (len(trials) + per_page - 1) // per_page)
            self._detail_page = min(self._detail_page, pages-1)
            shown = trials[self._detail_page * per_page:(self._detail_page + 1) * per_page]
            selected_trial = self.manual_trial_box.currentData()
            self.manual_trial_box.blockSignals(True); self.manual_trial_box.clear()
            for trial_id in shown:
                self.manual_trial_box.addItem(f"原Trial {trial_id} / Tag内 {details[trial_id][0]['tag_trial_id']}", trial_id)
            selected_index = self.manual_trial_box.findData(selected_trial)
            self.manual_trial_box.setCurrentIndex(selected_index if selected_index >= 0 else 0)
            self.manual_trial_box.blockSignals(False)
            self._manual_preview = None
            self._manual_axes = {}
            with rc_context(PLOT_STYLE):
                self.detail_figure.clear()
                self.detail_axes = np.atleast_1d(self.detail_figure.subplots(
                    2 if per_page == 4 else 1, 2 if per_page == 4 else 1,
                )).ravel().tolist()
                self.detail_figure.suptitle(
                    f"{Path(self.output['source']['path']).name} | 逐 Trial 找波与截取详情", fontsize=12,
                )
                for index, axis in enumerate(self.detail_axes):
                    axis.set_visible(index < len(shown))
                    if index >= len(shown):
                        continue
                    rows = details[shown[index]]
                    self._manual_axes[axis] = rows
                    channel = rows[0]["channel"]
                    if self._detail_conditioned_channel != channel:
                        self._detail_conditioned = _condition_channel(self.source, rows[0]["column"], self.output["settings"])
                        self._detail_conditioned_channel = channel
                    t, values = trial_detail_trace(self.source, self.output, rows, self._detail_conditioned)
                    localization = trial_detail_localization(self.source, self.output, rows)
                    draw_trial_detail(axis, self.output, rows, t, values, localization, self.detail_curve_box.currentData())
            self.detail_toolbar.update()  # Discard zoom history from the previous page/tag.
            self.manual_box.setEnabled(not self._busy and not self._parameters_dirty and bool(shown))
            self._manual_selection_changed()
            self.detail_canvas.draw_idle()
            self.page_label.setText(f"第 {self._detail_page+1}/{pages} 页 | {len(trials)} 个 trial；红=峰，绿=谷，紫=AnimalStart")
            self.previous_button.setEnabled(self._detail_page > 0); self.next_button.setEnabled(self._detail_page+1 < pages)
        except Exception as exc:
            self.status.setText("详情读取失败：" + str(exc))
        finally:
            self._rendering = False

    def start_export(self):
        if self._busy or self.output is None:
            return
        state = self.output.get("letter_vep")
        if state and state.get("stale"):
            QtWidgets.QMessageBox.warning(self, "事件对齐 VEP待刷新", "二次滤波/恢复后信号已变化，请先刷新事件对齐VEP，再保存完整结果。")
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择完整测试结果的保存目录")
        if not directory:
            return
        details = {(row["channel"], row["tag"], row["trial_id"]) for row in self.output["records"] if "extreme_local_sample" in row}
        pending = "\n当前控件参数尚未重新计算：本次导出上一次完成的结果、人工点位和该结果的原参数。" if self._parameters_dirty else ""
        if state:
            try: letter_pending = self._letter_vep_config() != state["parameters"]
            except ValueError: letter_pending = True
            pending += "\n包含已刷新过的各事件对齐方式、全部通道/Tag、共用Trial名单及数据。"
            if letter_pending: pending += "\n事件对齐参数/方式待刷新：将导出已应用过的模式及其参数，不使用尚未刷新的输入值。"
        answer = QtWidgets.QMessageBox.question(self, "确认详细导出", f"将创建独立结果目录，保存 {len(details)} 个 trial×通道的找波详情、分页PDF、平均图、CSV、NPZ及参数。\n独立trial PNG：{'开启' if self.individual_check.isChecked() else '关闭'}。{pending}\n可能耗时并占用较多空间，是否继续？")
        yes = QtWidgets.QMessageBox.StandardButton.Yes if hasattr(QtWidgets.QMessageBox, "StandardButton") else QtWidgets.QMessageBox.Yes
        if answer != yes:
            return
        self._set_busy(True); self.progress.setValue(0)
        self.worker = AnimalPeakWorker(self.source, None, None, {}, self,
                                       export=(self.output, directory, self.individual_check.isChecked()))
        self._wire_worker(self.worker)
        self.worker.completed.connect(lambda folder, worker=self.worker: self._accept_completion(
            worker, self._export_finished, folder))
        self.worker.start()

    def _export_finished(self, folder):
        self._exported_manual_actions = len(self.output.get("manual_alignment_history", []))
        self._exported_window_resizes = len(self.output.get("window_resize_history", []))
        self._exported_average_selections = len(self.output.get("average_selection_history", []))
        self._exported_secondary_filters = len(self.output.get("secondary_filter_history", []))
        self._exported_letter_revision = self.output.get("letter_vep", {}).get("revision", 0)
        self.status.setText(f"完整测试结果已保存：{folder}")

    def cancel(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker._ui_cancel_requested = True
            if hasattr(self.parent(), "_mark_lfp_cancel_requested"):
                self.parent()._mark_lfp_cancel_requested(self.worker)
            self.worker.cancel()
            self._show_cancel_request()

    def _show_cancel_request(self):
        self.status.setText("正在取消，等待当前片段/当前页安全结束…")
        self.cancel_button.setText("取消中…"); self.cancel_button.setEnabled(False)

    def failed(self, message):
        if self.worker is not None and (getattr(self.worker, "_ui_cancel_requested", False)
                                        or self.worker.isInterruptionRequested()):
            self.cancelled(message)
            return
        self.status.setText(message)
        if not self._close_requested:
            QtWidgets.QMessageBox.warning(self, "测试操作未完成", message)

    def cancelled(self, message="已取消测试操作"):
        self.progress.setValue(0)
        self.status.setText(message + "\n未修改原数据及原任务结果。")

    def _worker_finished(self):
        worker = self.worker
        if getattr(self.parent(), "_animal_peak_test_worker", None) is worker:
            self.parent()._animal_peak_test_worker = None
        self.worker = None
        if worker is not None:
            if worker.secondary is not None:
                # The committed view owns any live caches now. Do not let a
                # queued QObject deletion retain replaced/restored channels.
                worker.source = None
                worker.secondary = None
            worker.deleteLater()
        self._set_busy(False)
        if self._close_requested:
            self.close()

    def closeEvent(self, event):
        if self._busy:
            self._close_requested = True; self.cancel(); event.ignore()
        else:
            if self._has_unsaved_edits():
                answer = QtWidgets.QMessageBox.question(
                    self, "修改尚未导出", "存在尚未保存的人工点位、截取窗、Trial平均选择、二次滤波或LetterStart VEP结果。关闭后会释放滤波缓存且不能直接恢复本窗口，是否仍要关闭？",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No)
                if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                    self._close_requested = False
                    event.ignore(); return
            self.source = self._base_source
            self.output = None  # Release private filter caches/arrays on an accepted close.
            self._detail_conditioned = None
            super().closeEvent(event)
