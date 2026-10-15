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

from animal_peak_analysis import (
    AnalysisCancelled, analyze_animal_peaks, export_animal_peak_results, trial_detail_trace,
    _condition_channel,
)


class AnimalPeakWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, source, markers, matches, settings, parent=None, export=None):
        super().__init__(parent)
        self.source, self.markers, self.matches, self.settings = source, markers, matches, settings
        self.export = export

    def run(self):
        try:
            if self.export is None:
                value = analyze_animal_peaks(self.source, self.markers, self.matches, self.settings,
                                            self.progress.emit, self.isInterruptionRequested)
            else:
                output, directory, individual = self.export
                value = export_animal_peak_results(output, self.source, directory, individual,
                                                  self.progress.emit, self.isInterruptionRequested)
            self.completed.emit(value)
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class AnimalPeakDialog(QtWidgets.QDialog):
    def __init__(self, parent, source, markers, matches, defaults):
        super().__init__(parent)
        self.setWindowTitle("测试：AnimalStart 前最大峰 / 最小谷对齐平均")
        self.resize(1300, 900)
        self.source = source
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
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "测试功能，与原 Letter/SNR 计算独立。每个有效 trial 同时取搜索区最大值和最小值；"
            "不按相似度筛选、不反转负谷、不归一化幅值。没有有效 AnimalStart 的 trial 不生成子图。"
            "强制极值对齐的平均峰不能单独证明存在稳定 VEP。"
        )
        intro.setWordWrap(True); layout.addWidget(intro)
        metadata = source.metadata
        source_label = QtWidgets.QLabel(
            f"数据：{Path(metadata.path).name} | {metadata.fs:g} Hz | 当前已选 {len(defaults['channel_ids'])} 通道\n"
            f"沿用 Letter Epoch {defaults['epoch_start_ms']:g}～{defaults['epoch_end_ms']:g} ms；"
            f"基线 {defaults['baseline_start_ms']:g}～{defaults['baseline_end_ms']:g} ms（相对 Letter）。"
            "截取片段可跨 AnimalStart，并可按需读取旧 Epoch 外仍存在的源数据。"
        )
        source_label.setWordWrap(True); layout.addWidget(source_label)
        self.parameter_box = QtWidgets.QGroupBox("测试参数（不回写原任务参数）")
        form = QtWidgets.QGridLayout(self.parameter_box)
        self.search_spin = self._spin(300, .1, 60000)
        self.pre_spin = self._spin(50, 0, 60000)
        self.post_spin = self._spin(50, 0, 60000)
        self.channel_edit = QtWidgets.QLineEdit()
        self.channel_edit.setPlaceholderText("留空使用当前已选 LFP 通道；或输入实际通道号：1,3,22")
        self.tag_edit = QtWidgets.QLineEdit(",".join(str(x) for x in defaults.get("tags", [])))
        self.tag_edit.setPlaceholderText("留空=全部 Letter Tag 4/5/6/7")
        self.limit_spin = QtWidgets.QSpinBox(); self.limit_spin.setRange(0, 1000000)
        self.limit_spin.setMaximumWidth(145)
        self.limit_spin.setValue(int(defaults.get("trial_limit", 0)))
        self.limit_spin.setSpecialValueText("全部")
        for column, (label, widget) in enumerate((("AnimalStart 前搜索 ms", self.search_spin),
                                                ("特征点前截取 ms", self.pre_spin), ("特征点后截取 ms", self.post_spin))):
            form.addWidget(QtWidgets.QLabel(label), 0, column*2); form.addWidget(widget, 0, column*2+1)
        form.addWidget(QtWidgets.QLabel("实际通道号"), 1, 0); form.addWidget(self.channel_edit, 1, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("Tag"), 1, 4); form.addWidget(self.tag_edit, 1, 5)
        form.addWidget(QtWidgets.QLabel("每 Tag Trial 上限"), 2, 0); form.addWidget(self.limit_spin, 2, 1)
        self.processing_check = QtWidgets.QCheckBox("沿用任务额外分析处理（陷波/带通/平滑/Z-score；可能需读整条通道）")
        self.processing_check.setChecked(True); form.addWidget(self.processing_check, 2, 2, 1, 4)
        layout.addWidget(self.parameter_box)
        row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("提取最大峰 / 最小谷并平均")
        self.run_button.setObjectName("primaryAction")
        self.cancel_button = QtWidgets.QPushButton("取消当前操作"); self.cancel_button.setEnabled(False)
        self.export_button = QtWidgets.QPushButton("保存完整结果（图 + 明细 + 数据）"); self.export_button.setEnabled(False)
        self.individual_check = QtWidgets.QCheckBox("同时保存每个 trial 的独立 PNG"); self.individual_check.setChecked(True)
        for widget in (self.run_button, self.cancel_button, self.export_button, self.individual_check):
            row.addWidget(widget)
        layout.addLayout(row)
        self.progress = QtWidgets.QProgressBar(); layout.addWidget(self.progress)
        self.status = QtWidgets.QLabel("设置长度后点击提取；最大值、最小值分开平均，相关系数仅供查看。")
        self.status.setWordWrap(True); layout.addWidget(self.status)
        selection = QtWidgets.QHBoxLayout()
        self.pair_box = QtWidgets.QComboBox(); selection.addWidget(QtWidgets.QLabel("结果通道 / Tag")); selection.addWidget(self.pair_box, 1)
        self.summary = QtWidgets.QLabel(); self.summary.setWordWrap(True); layout.addLayout(selection); layout.addWidget(self.summary)
        self.tabs = QtWidgets.QTabWidget(); layout.addWidget(self.tabs, 1)
        self.average_plots = self._plot_tab("峰 / 谷叠加平均", 1, 2)
        self.histogram_plots = self._plot_tab("原始峰时 / 谷时分布", 1, 2)
        details = QtWidgets.QWidget(); details_layout = QtWidgets.QVBoxLayout(details)
        navigation = QtWidgets.QHBoxLayout()
        self.previous_button = QtWidgets.QPushButton("上一页")
        self.next_button = QtWidgets.QPushButton("下一页")
        self.page_label = QtWidgets.QLabel()
        navigation.addWidget(self.previous_button); navigation.addWidget(self.next_button); navigation.addWidget(self.page_label); navigation.addStretch(1)
        details_layout.addLayout(navigation)
        grid = QtWidgets.QGridLayout(); self.detail_plots = []; self.detail_labels = []; self.detail_slots = []
        for index in range(4):
            slot = QtWidgets.QWidget(); body = QtWidgets.QVBoxLayout(slot); body.setContentsMargins(0, 0, 0, 0)
            label = QtWidgets.QLabel(); label.setWordWrap(True); label.setMinimumHeight(55)
            label.setStyleSheet("color:#202832; font-size:12px;")
            plot = self._plot(); body.addWidget(label); body.addWidget(plot, 1)
            grid.addWidget(slot, index//2, index%2)
            self.detail_plots.append(plot); self.detail_labels.append(label); self.detail_slots.append(slot)
        details_layout.addLayout(grid, 1); self.tabs.addTab(details, "逐 Trial 找波与截取详情")
        self.table = QtWidgets.QTableWidget(0, 12)
        self.table.setHorizontalHeaderLabels(["原Trial", "Tag内Trial", "Tag", "通道", "峰/谷", "状态", "AnimalStart s", "极值相对 ms", "幅值", "相似度r（不筛选）", "标记", "排除原因"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers if hasattr(QtWidgets.QAbstractItemView, "EditTrigger") else QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.table, "全部 Trial 明细 / 排除原因")
        self.run_button.clicked.connect(self.start_analysis)
        self.cancel_button.clicked.connect(self.cancel)
        self.export_button.clicked.connect(self.start_export)
        self.pair_box.currentIndexChanged.connect(self.render_pair)
        self.tabs.currentChanged.connect(lambda *_: self.render_details() if self.tabs.currentIndex() == 2 else None)
        self.previous_button.clicked.connect(lambda: self._change_page(-1))
        self.next_button.clicked.connect(lambda: self._change_page(1))
        for widget in (self.search_spin, self.pre_spin, self.post_spin, self.limit_spin):
            widget.valueChanged.connect(self.parameters_changed)
        for widget in (self.channel_edit, self.tag_edit):
            widget.textChanged.connect(self.parameters_changed)
        self.processing_check.toggled.connect(self.parameters_changed)

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

    def _plot_tab(self, title, rows, columns):
        page = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(page); plots = []
        for index in range(rows*columns):
            plot = self._plot(); grid.addWidget(plot, index//columns, index%columns); plots.append(plot)
        self.tabs.addTab(page, title)
        return plots

    def parameters_changed(self, *_):
        if self.output is not None and not self._busy:
            self.export_button.setEnabled(False)
            self.status.setText("参数已修改；当前图仍为上次计算结果，请重新提取后导出。")

    def _set_busy(self, busy):
        self._busy = busy
        self.parameter_box.setEnabled(not busy); self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy); self.export_button.setEnabled(not busy and self.output is not None)
        self.pair_box.setEnabled(not busy); self.tabs.setEnabled(not busy)

    def start_analysis(self):
        if self._busy:
            return
        try:
            parse = lambda value: list(dict.fromkeys(int(part.strip()) for part in value.replace(";", ",").replace("，", ",").split(",") if part.strip()))
            settings = {**self.defaults, "search_ms": self.search_spin.value(), "pre_ms": self.pre_spin.value(), "post_ms": self.post_spin.value(),
                        "channel_ids": parse(self.channel_edit.text()) or list(self.defaults["channel_ids"]),
                        "tags": parse(self.tag_edit.text()), "trial_limit": self.limit_spin.value(),
                        "processing": dict(self.defaults.get("processing") or {}) if self.processing_check.isChecked() else {}}
            if any(tag not in (4, 5, 6, 7) for tag in settings["tags"]):
                raise ValueError("此测试只支持 Letter Tag 4/5/6/7。")
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数无效", str(exc)); return
        self.output = None  # Release the previous waveform set before computing another.
        self._pairs = {}
        self._detail_conditioned = None
        self._detail_conditioned_channel = None
        for plot in self.average_plots + self.histogram_plots + self.detail_plots:
            plot.clear()
        self.table.setRowCount(0)
        self._set_busy(True); self.progress.setValue(0)
        self.status.setText("正在逐通道、逐 trial 提取；不读取整份 H5 到内存。")
        self.worker = AnimalPeakWorker(self.source, self.markers, self.matches, settings, self)
        self._wire_worker(self.worker)
        self.worker.completed.connect(self.analysis_finished)
        self.worker.start()

    def _wire_worker(self, worker):
        # The main window's existing shutdown cleanup discovers this handle.
        self.parent()._animal_peak_test_worker = worker
        worker.progress.connect(self.update_progress)
        worker.failed.connect(self.failed)
        worker.cancelled.connect(self.cancelled)
        worker.finished.connect(self._worker_finished)

    def update_progress(self, value, text):
        self.progress.setValue(int(value)); self.status.setText(text)

    def analysis_finished(self, output):
        self.output = output
        self._pairs = defaultdict(dict)
        for group in output["groups"]:
            self._pairs[(group["channel"], group["tag"])][group["branch"]] = group
        self.pair_box.blockSignals(True); self.pair_box.clear()
        for pair in sorted(self._pairs):
            self.pair_box.addItem(f"ch{pair[0]} / Tag {pair[1]}", pair)
        self.pair_box.blockSignals(False)
        self.table.setRowCount(len(output["records"]))
        scale = 1000 if output["units"] == "mV" else 1
        unit = "μV" if scale == 1000 else "z"
        for index, row in enumerate(output["records"]):
            values = (row["trial_id"], row["tag_trial_id"], row["tag"], row["channel"], row["branch"], row["status"],
                      self._number(row.get("animal_sec"), 6), self._number(row.get("extreme_relative_ms"), 2),
                      self._number(row.get("amplitude", np.nan)*scale) + " " + unit,
                      self._number(row.get("similarity_r"), 3), row.get("flags", ""), row.get("reason", ""))
            for column, value in enumerate(values):
                self.table.setItem(index, column, QtWidgets.QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
        self._detail_page = 0; self.render_pair()
        included = sum(row["status"] == "included" for row in output["records"])
        self.status.setText(f"测试完成：{len(output['eligible_trials'])} 个有效 AnimalStart trial；峰谷分支纳入 {included} 条记录，详情及排除原因已保留。")
        self.progress.setValue(100)

    @staticmethod
    def _number(value, digits=3):
        return f"{float(value):.{digits}f}" if value is not None and np.isfinite(value) else "—"

    def render_pair(self, *_):
        if not self.output or not hasattr(self, "_pairs"):
            return
        pair = self.pair_box.currentData()
        self._detail_page = 0
        scale = 1000 if self.output["units"] == "mV" else 1
        unit = "μV" if scale == 1000 else "z"
        counts = []
        for index, branch in enumerate(("peak", "trough")):
            plot, histogram = self.average_plots[index], self.histogram_plots[index]
            plot.clear(); histogram.clear()
            group = self._pairs.get(pair, {}).get(branch)
            name = "最大峰" if branch == "peak" else "最小谷"
            if group is None:
                plot.setTitle(name + "：无完整片段"); continue
            # Every trial is displayed; no hidden decimation across trials.
            for wave in group["waves"]:
                plot.plot(group["time_ms"], wave*scale, pen=pg.mkPen((120, 130, 140, 65), width=.7))
            plot.plot(group["time_ms"], group["mean"]*scale, pen=pg.mkPen("#b32025", width=2.2))
            plot.addItem(pg.InfiniteLine(0, pen=pg.mkPen("#333333", style=QtCore.Qt.PenStyle.DashLine if hasattr(QtCore.Qt, "PenStyle") else QtCore.Qt.DashLine)))
            plot.setTitle(f"ch{pair[0]} | Tag {pair[1]} | {name}对齐 | n={len(group['rows'])}")
            plot.setLabel("bottom", "相对选中极值", units="ms"); plot.setLabel("left", unit)
            latency = [row["extreme_relative_ms"] for row in group["rows"]]
            y, x = np.histogram(latency, bins=min(20, max(1, len(latency))))
            histogram.plot(x, y, stepMode="center", fillLevel=0, brush=(65, 119, 163, 110), pen="#4177a3")
            histogram.setTitle(name + "相对 AnimalStart 的时间分布")
            histogram.setLabel("bottom", "相对 AnimalStart", units="ms"); histogram.setLabel("left", "Trial 数")
            counts.append(f"{name} n={len(group['rows'])}")
        self.summary.setText("；".join(counts) + "。灰线=全部片段，红线=等权平均；相似度仅在明细中报告，不作筛选。")
        if self.tabs.currentIndex() == 2:
            self.render_details()

    def _change_page(self, step):
        self._detail_page = max(0, self._detail_page + step); self.render_details()

    def render_details(self):
        if self._rendering or not self.output:
            return
        self._rendering = True
        try:
            pair = self.pair_box.currentData()
            details = defaultdict(list)
            for row in self.output["records"]:
                if (row["channel"], row["tag"]) == pair and "extreme_local_sample" in row:
                    details[row["trial_id"]].append(row)
            trials = sorted(details)
            pages = max(1, (len(trials)+3)//4)
            self._detail_page = min(self._detail_page, pages-1)
            shown = trials[self._detail_page*4:(self._detail_page+1)*4]
            scale = 1000 if self.output["units"] == "mV" else 1
            for index, plot in enumerate(self.detail_plots):
                plot.clear(); self.detail_slots[index].setVisible(index < len(shown))
                if index >= len(shown):
                    continue
                rows = details[shown[index]]
                channel = rows[0]["channel"]
                if self._detail_conditioned_channel != channel:
                    self._detail_conditioned = _condition_channel(self.source, rows[0]["column"], self.output["settings"])
                    self._detail_conditioned_channel = channel
                t, values = trial_detail_trace(self.source, self.output, rows, self._detail_conditioned)
                plot.plot(t, values*scale, pen=pg.mkPen("#30343a", width=.8))
                animal = pg.InfiniteLine(0, pen="#9467bd", label="AnimalStart"); plot.addItem(animal)
                search = pg.LinearRegionItem([-self.output["settings"]["search_ms"], 0], movable=False, brush=(100, 160, 210, 25))
                plot.addItem(search)
                subtitles = []
                for row in rows:
                    color = "#d44436" if row["branch"] == "peak" else "#16885b"
                    name = "峰" if row["branch"] == "peak" else "谷"
                    position = row["extreme_relative_ms"]
                    plot.plot([position], [row["amplitude"]*scale], pen=None, symbol="o", symbolBrush=color)
                    region = pg.LinearRegionItem([position-self.output["settings"]["pre_ms"], position+self.output["settings"]["post_ms"]], movable=False, brush=pg.mkBrush(color))
                    region.setBrush(pg.mkBrush(210, 100, 80, 12) if row["branch"] == "peak" else pg.mkBrush(50, 160, 90, 12)); plot.addItem(region)
                    subtitles.append(f"{name} {position:.2f}ms, r={self._number(row['similarity_r'], 2)}")
                flags = "；".join(dict.fromkeys(filter(None, (row["flags"] or row["reason"] for row in rows))))
                self.detail_labels[index].setText(f"ch{rows[0]['channel']} | Tag {rows[0]['tag']} | 原Trial {rows[0]['trial_id']} | Tag内 {rows[0]['tag_trial_id']}\nAnimalStart {rows[0]['animal_sec']:.6f}s | {' / '.join(subtitles)}\n{flags or '完整截取；相似度不筛选'}")
                plot.setLabel("bottom", "相对 AnimalStart", units="ms"); plot.setLabel("left", "μV" if scale == 1000 else "z")
            self.page_label.setText(f"第 {self._detail_page+1}/{pages} 页 | {len(trials)} 个 trial；红=峰，绿=谷，紫=AnimalStart")
            self.previous_button.setEnabled(self._detail_page > 0); self.next_button.setEnabled(self._detail_page+1 < pages)
        except Exception as exc:
            self.status.setText("详情读取失败：" + str(exc))
        finally:
            self._rendering = False

    def start_export(self):
        if self._busy or self.output is None:
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择完整测试结果的保存目录")
        if not directory:
            return
        details = {(row["channel"], row["tag"], row["trial_id"]) for row in self.output["records"] if "extreme_local_sample" in row}
        answer = QtWidgets.QMessageBox.question(self, "确认详细导出", f"将创建独立结果目录，保存 {len(details)} 个 trial×通道的找波详情、分页PDF、平均图、CSV、NPZ及参数。\n独立trial PNG：{'开启' if self.individual_check.isChecked() else '关闭'}。可能耗时并占用较多空间，是否继续？")
        yes = QtWidgets.QMessageBox.StandardButton.Yes if hasattr(QtWidgets.QMessageBox, "StandardButton") else QtWidgets.QMessageBox.Yes
        if answer != yes:
            return
        self._set_busy(True); self.progress.setValue(0)
        self.worker = AnimalPeakWorker(self.source, None, None, {}, self,
                                       export=(self.output, directory, self.individual_check.isChecked()))
        self._wire_worker(self.worker)
        self.worker.completed.connect(lambda folder: self.status.setText(f"完整测试结果已保存：{folder}"))
        self.worker.start()

    def cancel(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption(); self.status.setText("已请求取消，正在完成当前片段/当前页…")

    def failed(self, message):
        self.status.setText(message)
        if not self._close_requested:
            QtWidgets.QMessageBox.warning(self, "测试操作未完成", message)

    def cancelled(self):
        self.status.setText("已取消测试计算；未修改原数据及原任务结果。")

    def _worker_finished(self):
        worker = self.worker
        if getattr(self.parent(), "_animal_peak_test_worker", None) is worker:
            self.parent()._animal_peak_test_worker = None
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)
        if self._close_requested:
            self.close()

    def closeEvent(self, event):
        if self._busy:
            self._close_requested = True; self.cancel(); event.ignore()
        else:
            super().closeEvent(event)
