"""Qt integration checks for the new, isolated experimental button/window."""
import os
import csv
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import rc_context
from matplotlib.figure import Figure

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qt_gui import QApplication, QtAnalysisGUI
from animal_peak_dialog import AnimalPeakDialog, AverageTrialSelectionDialog, QtCore, QtWidgets
from animal_peak_trial_selection import average_trial_ids
from animal_peak_analysis import analyze_animal_peaks, trial_detail_trace
from animal_peak_analysis import AnalysisCancelled
from animal_peak_plotting import PLOT_STYLE, draw_trial_detail
from test_animal_peak_analysis import fixture
from test_animal_peak_localization import trend_fixture


class AnimalPeakDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        with patch("qt_gui.OperationAuditStore"), patch(
            "qt_gui.FeishuAuditConfig.from_environment", return_value=SimpleNamespace(is_configured=False),
        ):
            self.window = QtAnalysisGUI("UI test")
        self.source, self.markers, self.matches, self.settings = fixture()
        self.dialog = AnimalPeakDialog(self.window, self.source, self.markers, self.matches, self.settings)

    def tearDown(self):
        if self.dialog.worker is not None and self.dialog.worker.isRunning():
            self.dialog.worker.requestInterruption()
            self.dialog.worker.wait(15000)
        self.app.processEvents()
        self.dialog.deleteLater(); self.window.deleteLater(); self.app.processEvents()

    def wait_for_worker(self):
        deadline = time.monotonic()+20
        while self.dialog._busy and time.monotonic() < deadline:
            self.app.processEvents()
            if self.dialog.worker is not None:
                self.dialog.worker.wait(5)
        self.app.processEvents()
        self.assertFalse(self.dialog._busy, "background operation did not finish")

    def configure_completed_alignment(self, source, channel_ids, markers=None, matches=None):
        window = self.window
        window.active_source = source
        window.stim_markers = self.markers if markers is None else markers
        window.stim_marker_mode = "Letter"
        window.behavior_matches = self.matches if matches is None else matches
        window._bad_channel_check_completed = True
        window.good_channel_ids = set(map(int, source.metadata.channel_ids))
        window.bad_channel_ids = set()
        window._task_marker_source = {
            "source_path": str(source.metadata.path), "fs": float(source.metadata.fs),
            "time_offset_sec": float(source.metadata.time_offset), "selected_channel_ids": channel_ids,
        }

    def test_threaded_analysis_details_and_parameter_changes(self):
        with patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.dialog.channel_edit.setText("7")
            self.dialog.start_analysis()
            self.wait_for_worker()
            warning.assert_not_called()
            self.assertIsNotNone(self.dialog.output)
            self.assertEqual(self.dialog.result_tag_box.count(), 1)
            self.assertEqual(self.dialog.result_channel_box.count(), 1)
            self.assertTrue(self.dialog.export_button.isEnabled())
            self.assertIsNone(self.window._animal_peak_test_worker)
            self.dialog.tabs.setCurrentIndex(2)
            self.app.processEvents()
            self.assertIn("2 个 trial", self.dialog.page_label.text())
            self.assertIn("Tag 4", self.dialog.detail_axes[0].get_title())
            self.assertIn("原 Trial 1", self.dialog.detail_axes[0].get_title())
            self.dialog.post_spin.setValue(70)
            self.assertFalse(self.dialog.export_button.isEnabled())
            self.assertIn("参数已修改", self.dialog.status.text())

    def test_after_search_parameter_reaches_worker_and_detail_highlight(self):
        self.assertEqual(self.dialog.search_after_spin.value(), 0)
        self.dialog.channel_edit.setText("7")
        self.dialog.search_after_spin.setValue(200)
        self.source.data[3080, 0] = 15
        self.dialog.start_analysis()
        self.wait_for_worker()
        self.assertEqual(self.dialog.output["settings"]["search_after_ms"], 200)
        peak = next(group for group in self.dialog.output["groups"] if group["branch"] == "peak")
        self.assertEqual(peak["rows"][0]["extreme_relative_ms"], 80)
        self.dialog.tabs.setCurrentIndex(2)
        self.app.processEvents()
        region = self.dialog.detail_axes[0].patches[0]
        self.assertAlmostEqual(region.get_x(), -300)
        self.assertAlmostEqual(region.get_width(), 500)
        self.assertTrue(self.dialog.export_button.isEnabled())
        self.dialog.search_after_spin.setValue(250)
        self.assertFalse(self.dialog.export_button.isEnabled())
        self.assertIn("参数已修改", self.dialog.status.text())

    def test_oversized_pre_search_keeps_trials_and_shows_clipped_epoch_range(self):
        self.dialog.channel_edit.setText("7")
        self.dialog.search_spin.setValue(5000)
        self.dialog.start_analysis()
        self.wait_for_worker()
        output = self.dialog.output
        self.assertEqual([row["trial_id"] for row in output["eligible_trials"]], [1, 2])
        self.assertFalse(any(row["reason_code"] == "search_before_epoch" for row in output["records"]))
        self.dialog.tabs.setCurrentIndex(2)
        self.app.processEvents()
        axis = self.dialog.detail_axes[0]
        self.assertAlmostEqual(axis.patches[0].get_x(), -1500)
        self.assertAlmostEqual(axis.patches[0].get_width(), 1500)
        self.assertGreater(axis.get_xlim()[0], -2000)
        self.assertTrue(next(line for line in axis.lines if line.get_gid() == "letter_start").get_visible())

    def show_details(self):
        output = analyze_animal_peaks(self.source, self.markers, self.matches,
                                      {**self.settings, "channel_ids": [7]})
        self.dialog.analysis_finished(output)
        self.dialog.tabs.setCurrentIndex(2)
        self.dialog.show()
        self.app.processEvents()
        return output

    def test_window_controls_and_parameter_collapse_leave_progress_available(self):
        flags = QtCore.Qt.WindowType if hasattr(QtCore.Qt, "WindowType") else QtCore.Qt
        self.assertTrue(self.dialog.windowFlags() & flags.WindowMinimizeButtonHint)
        self.assertTrue(self.dialog.windowFlags() & flags.WindowMaximizeButtonHint)
        self.assertTrue(self.dialog.isSizeGripEnabled())
        self.dialog.show(); self.app.processEvents()
        self.dialog.showMaximized(); self.app.processEvents()
        self.assertTrue(self.dialog.isMaximized())
        self.dialog.showMinimized(); self.app.processEvents()
        self.assertTrue(self.dialog.isMinimized())
        self.dialog.showNormal(); self.app.processEvents()
        self.dialog.resize(950, 650); self.app.processEvents()
        self.assertLessEqual(self.dialog.width(), 1000)
        self.dialog.parameter_toggle.click(); self.app.processEvents()
        self.assertTrue(self.dialog.parameter_panel.isHidden())
        self.assertFalse(self.dialog.progress.isHidden())
        self.assertFalse(self.dialog.cancel_button.isHidden())
        self.dialog.parameter_toggle.click()
        self.assertFalse(self.dialog.parameter_panel.isHidden())

    def test_result_selectors_are_immediately_below_letter_display_range(self):
        layout = self.dialog.page_content.layout()
        position = layout.indexOf(self.dialog.letter_view_box)
        self.assertGreaterEqual(position, 0)
        selection = layout.itemAt(position + 1).layout()
        self.assertIsNotNone(selection)
        self.assertGreaterEqual(selection.indexOf(self.dialog.result_tag_box), 0)
        self.assertGreaterEqual(selection.indexOf(self.dialog.result_channel_box), 0)
        self.assertIs(layout.itemAt(position + 2).widget(), self.dialog.summary)
        self.assertIs(layout.itemAt(position + 3).widget(), self.dialog.tabs)

    def test_result_selectors_are_compact_red_and_stay_left_aligned_on_resize(self):
        d = self.dialog
        d.analysis_finished(analyze_animal_peaks(self.source, self.markers, self.matches, self.settings))
        d.parameter_toggle.setChecked(False)
        d.resize(1600, 900); d.show(); self.app.processEvents()
        for box, width in ((d.result_tag_box, 140), (d.result_channel_box, 180)):
            self.assertEqual(box.width(), width)
            self.assertEqual(box.minimumWidth(), box.maximumWidth())
            self.assertEqual(box.palette().color(box.foregroundRole()).name(), "#c62828")
            self.assertTrue(box.font().bold())
        for label in (d.result_tag_label, d.result_channel_label):
            self.assertEqual(label.palette().color(label.foregroundRole()).name(), "#c62828")
            self.assertTrue(label.font().bold())
        self.assertEqual(d.result_channel_box.lineEdit().palette().color(d.result_channel_box.lineEdit().foregroundRole()).name(), "#c62828")
        self.assertTrue(d.result_channel_box.isEditable())
        self.assertLess(d.result_channel_box.geometry().right(), 600)
        d.resize(1000, 900); self.app.processEvents()
        self.assertEqual(d.result_tag_box.width(), 140)
        self.assertEqual(d.result_channel_box.width(), 180)

    def test_layout_has_one_whole_page_scroll_and_no_nested_parameter_or_plot_scroll(self):
        self.dialog.resize(1300, 900); self.dialog.show(); self.app.processEvents()
        self.assertEqual(self.dialog.findChildren(QtWidgets.QScrollArea), [self.dialog.page_scroll])
        self.assertIs(self.dialog.parameter_panel.parentWidget(), self.dialog.page_content)
        self.assertGreater(self.dialog.parameter_panel.maximumHeight(), 280)
        self.assertGreaterEqual(self.dialog.parameter_panel.height(), self.dialog.parameter_panel.minimumSizeHint().height())
        self.assertEqual(self.dialog.page_scroll.horizontalScrollBar().maximum(), 0)

    def test_whole_page_scrolls_full_details_and_keeps_cancel_progress_visible(self):
        self.show_details()
        self.dialog.resize(1300, 900); self.app.processEvents()
        self.assertEqual(self.dialog.detail_canvas.height(), 680)
        bar = self.dialog.page_scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        bar.setValue(bar.maximum()); self.app.processEvents()
        self.assertTrue(self.dialog.manual_average_box.isVisible())
        for widget in (self.dialog.progress, self.dialog.cancel_button):
            position = widget.mapTo(self.dialog, QtCore.QPoint(0, 0))
            self.assertTrue(self.dialog.rect().contains(position))
            self.assertTrue(self.dialog.rect().contains(position + QtCore.QPoint(widget.width()-1, widget.height()-1)))
        detail_height = self.dialog.tabs.height()
        self.dialog.tabs.setCurrentIndex(0); self.app.processEvents()
        self.assertLess(self.dialog.tabs.height(), detail_height)

    def test_one_trial_layout_and_parameter_collapse_reduce_whole_page_height(self):
        self.show_details()
        self.dialog.resize(1300, 900); self.app.processEvents()
        initial = self.dialog.tabs.minimumSizeHint().height()
        self.dialog.details_per_page_box.setCurrentIndex(1); self.app.processEvents()
        self.assertEqual(self.dialog.detail_canvas.height(), 400)
        self.assertLess(self.dialog.tabs.minimumSizeHint().height(), initial)
        full_height = self.dialog.page_content.minimumHeight()
        self.dialog.parameter_toggle.click(); self.app.processEvents()
        self.assertLess(self.dialog.page_content.minimumHeight(), full_height)
        self.dialog.parameter_toggle.click(); self.app.processEvents()
        self.assertFalse(self.dialog.parameter_panel.isHidden())

    def test_trial_detail_matches_export_renderer_and_resets_zoom(self):
        output = self.show_details()
        ui_axis = self.dialog.detail_axes[0]
        rows = [row for row in output["records"] if row["trial_id"] == 1 and row.get("channel") == 7]
        time_ms, values = trial_detail_trace(self.source, output, rows)
        with rc_context(PLOT_STYLE):
            figure = Figure()
            exported_axis = figure.subplots()
            draw_trial_detail(exported_axis, output, rows, time_ms, values)
        self.assertEqual(ui_axis.get_title(), exported_axis.get_title())
        self.assertIn("AnimalStart=4.250000 s", ui_axis.get_title())
        self.assertIn("相对 Letter 开始=+1000.000 ms", ui_axis.get_title())
        np.testing.assert_allclose(ui_axis.get_xlim(), exported_axis.get_xlim())
        np.testing.assert_allclose(ui_axis.get_ylim(), exported_axis.get_ylim())
        for ui_line, exported_line in zip(ui_axis.lines, exported_axis.lines):
            np.testing.assert_array_equal(ui_line.get_xdata(), exported_line.get_xdata())
            np.testing.assert_array_equal(ui_line.get_ydata(), exported_line.get_ydata())
        self.assertIsNone(ui_axis.get_legend())
        self.assertIsNone(exported_axis.get_legend())
        self.assertEqual(sum(axis.get_visible() for axis in self.dialog.detail_axes), 2)
        expected_limits = ui_axis.get_xlim()
        ui_axis.set_xlim(-2, 2)
        self.dialog.render_details()
        np.testing.assert_allclose(self.dialog.detail_axes[0].get_xlim(), expected_limits)

    def test_detail_and_amplitude_table_display_source_numbers_without_voltage_conversion(self):
        output = self.show_details()
        rows = [row for row in output["records"] if row.get("channel") == 7 and row["trial_id"] == 1]
        _time, values = trial_detail_trace(self.source, output, rows)
        np.testing.assert_array_equal(self.dialog.detail_axes[0].lines[0].get_ydata(), values)
        self.assertEqual(self.dialog.detail_axes[0].get_ylabel(), "μV")
        index = next(index for index, row in enumerate(output["records"])
                     if row.get("channel") == 7 and row["trial_id"] == 1 and row["branch"] == "peak")
        self.assertEqual(self.dialog.table.item(index, 8).text(), "5.000 μV")
        self.assertEqual(output["records"][index]["amplitude"], 5)
        self.assertEqual(output["units"], "mV")  # Source/numerical metadata is not rewritten.

    def test_letter_onset_marker_is_identical_in_viewer_and_export_renderer(self):
        self.dialog.search_spin.setValue(1400)
        output = analyze_animal_peaks(self.source, self.markers[:1], self.matches,
                                      {**self.settings, "search_ms": 1400, "channel_ids": [7]})
        self.dialog.analysis_finished(output)
        self.dialog.tabs.setCurrentIndex(2)
        self.app.processEvents()
        rows = [row for row in output["records"] if row.get("channel") == 7]
        time_ms, values = trial_detail_trace(self.source, output, rows)
        exported = Figure().subplots()
        draw_trial_detail(exported, output, rows, time_ms, values)
        for axis in (self.dialog.detail_axes[0], exported):
            line = next(line for line in axis.lines if line.get_gid() == "letter_start")
            self.assertTrue(line.get_visible())
            np.testing.assert_allclose(line.get_xdata(), [-1000, -1000])
            self.assertEqual(line.get_color(), "#c62828")
        np.testing.assert_allclose(self.dialog.detail_axes[0].get_xlim(), exported.get_xlim())

    def test_fixed_y_axis_defaults_auto_and_applies_to_all_channels_without_changing_data(self):
        output = analyze_animal_peaks(self.source, self.markers, self.matches, self.settings)
        original = self.source.data.copy()
        waves = [group["waves"].copy() for group in output["groups"]]
        self.dialog.analysis_finished(output)
        self.dialog.tabs.setCurrentIndex(2)
        self.assertIsNone(self.dialog._fixed_y_limits)
        self.assertEqual(output["display_settings"]["y_axis_mode"], "auto")
        self.dialog.y_min_spin.setValue(-120)
        self.dialog.y_max_spin.setValue(340)
        self.assertEqual(output["display_settings"]["y_axis_mode"], "auto")  # Not applied yet.
        self.dialog.y_fixed_button.click()
        for channel in (7, 22):
            self.dialog.result_channel_box.setCurrentIndex(self.dialog.result_channel_box.findData(channel))
            for axis in self.dialog.detail_axes:
                if axis.get_visible():
                    np.testing.assert_allclose(axis.get_ylim(), [-120, 340])
                    axis.set_ylim(-1000, 1000)  # Vertical toolbar navigation cannot undo the lock.
                    np.testing.assert_allclose(axis.get_ylim(), [-120, 340])
            for plot in self.dialog.average_plots + self.dialog.manual_average_plots + self.dialog.animal_reference_plots:
                np.testing.assert_allclose(plot.getViewBox().viewRange()[1], [-120, 340])
            self.dialog.refresh_average_plot()
            np.testing.assert_allclose(self.dialog.average_plots[0].getViewBox().viewRange()[1], [-120, 340])
        self.assertFalse(self.dialog._parameters_dirty)
        np.testing.assert_array_equal(self.source.data, original)
        for group, before in zip(output["groups"], waves):
            np.testing.assert_array_equal(group["waves"], before)

    def test_fixed_y_axis_invalid_range_keeps_previous_limits_and_auto_restores_data_range(self):
        output = self.show_details()
        self.dialog.y_min_spin.setValue(-1)
        self.dialog.y_max_spin.setValue(1)
        self.dialog.y_fixed_button.click()
        self.dialog.y_min_spin.setValue(2)
        with patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.dialog.y_fixed_button.click()
        warning.assert_called_once()
        self.assertEqual(self.dialog._fixed_y_limits, (-1, 1))
        self.assertEqual(output["display_settings"]["y_min"], -1)
        self.dialog.y_auto_button.click()
        self.app.processEvents()
        self.assertIsNone(self.dialog._fixed_y_limits)
        self.assertEqual(output["display_settings"]["y_axis_mode"], "auto")
        self.assertGreater(self.dialog.detail_axes[0].get_ylim()[1], 1)
        self.assertGreater(self.dialog.average_plots[0].getViewBox().viewRange()[1][1], 1)
        self.assertTrue(self.dialog.average_plots[0].getViewBox().autoRangeEnabled()[1])

    def test_fixed_y_axis_survives_new_results_and_reports_z_units(self):
        self.dialog.y_min_spin.setValue(-2)
        self.dialog.y_max_spin.setValue(3)
        self.dialog.y_fixed_button.click()  # Can configure before extracting.
        output = analyze_animal_peaks(self.source, self.markers, self.matches, self.settings)
        output["units"] = "z"
        self.dialog.analysis_finished(output)
        self.assertEqual(self.dialog.y_unit_label.text(), "z")
        self.assertEqual(output["display_settings"], {"y_axis_mode": "fixed", "y_unit": "z", "y_min": -2, "y_max": 3, "letter_before_ms": 500})
        self.dialog._set_busy(True)
        self.assertFalse(self.dialog.y_axis_box.isEnabled())
        self.dialog.restore_auto_y_axis()
        self.assertEqual(self.dialog._fixed_y_limits, (-2, 3))
        self.dialog._set_busy(False)

    def test_letter_display_refresh_is_independent_of_search_and_preserves_manual_result(self):
        from animal_peak_manual import adjust_animal_peak
        from animal_peak_trial_selection import set_average_trial_selection

        output = analyze_animal_peaks(self.source, self.markers, self.matches, self.settings)
        output = adjust_animal_peak(self.source, output, channel=7, tag=4, trial_id=1,
                                   branch="peak", relative_ms=-100)
        output = set_average_trial_selection(output, 7, 4, [1])
        output["judgment_similarity"] = {"stale": True, "groups": []}
        self.dialog.analysis_finished(output)
        self.dialog.tabs.setCurrentIndex(2)
        waves = [group["waves"].copy() for group in output["groups"]]
        history = list(output["manual_alignment_history"])
        self.dialog.manual_time_spin.setValue(-110)  # Unconfirmed orange preview is kept too.
        self.dialog.letter_view_spin.setValue(1000)
        self.assertFalse(self.dialog._parameters_dirty)
        self.assertEqual(output["display_settings"]["letter_before_ms"], 500)
        with patch("animal_peak_dialog.analyze_animal_peaks", side_effect=AssertionError("must not re-detect")):
            self.dialog.letter_view_button.click()
        self.assertIs(self.dialog.output, output)
        self.assertEqual(output["display_settings"]["letter_before_ms"], 1000)
        self.assertEqual(output["manual_alignment_history"], history)
        self.assertEqual(average_trial_ids(output, 7, 4), (1,))
        self.assertEqual(self.dialog.manual_time_spin.value(), -110)
        self.assertIsNotNone(self.dialog._manual_preview)
        self.assertEqual(self.dialog.detail_axes[0].get_xlim()[0], -2000)
        for group, original in zip(output["groups"], waves):
            np.testing.assert_array_equal(group["waves"], original)
        for channel in (22, 7):
            self.dialog.result_channel_box.setCurrentIndex(self.dialog.result_channel_box.findData(channel))
            self.assertEqual(self.dialog.detail_axes[0].get_xlim()[0], -2000)
        self.dialog.y_min_spin.setValue(-50); self.dialog.y_max_spin.setValue(50)
        self.dialog.y_fixed_button.click()
        self.assertEqual(output["display_settings"]["letter_before_ms"], 1000)
        self.assertEqual(self.dialog.detail_axes[0].get_xlim()[0], -2000)
        self.dialog._set_busy(True)
        self.assertFalse(self.dialog.letter_view_box.isEnabled())
        self.dialog._set_busy(False)

    def test_full_record_secondary_filter_and_restore_keep_manual_points_and_trial_selection(self):
        from animal_peak_manual import adjust_animal_peak
        from animal_peak_trial_selection import set_average_trial_selection
        from animal_peak_filter import SecondaryFilteredSource

        self.assertFalse(self.dialog.secondary_apply.isEnabled())
        self.assertFalse(self.dialog.secondary_restore.isEnabled())
        self.assertEqual(self.dialog.secondary_frequency.value(),1)
        output=analyze_animal_peaks(self.source,self.markers,self.matches,self.settings)
        output=adjust_animal_peak(self.source,output,7,4,1,"peak",relative_ms=-100)
        output=set_average_trial_selection(output,7,4,[1])
        self.window.active_source=self.source
        self.dialog.analysis_finished(output)
        self.dialog.tabs.setCurrentIndex(2)
        original=self.source.data.copy()
        self.assertTrue(self.dialog.secondary_apply.isEnabled())
        self.assertEqual(self.dialog._current_pair(),(7,4))
        self.dialog.secondary_mode.setCurrentIndex(self.dialog.secondary_mode.findData("highpass"))
        self.assertFalse(self.dialog.secondary_frequency.isEnabled())
        self.assertTrue(self.dialog.secondary_low.isEnabled())
        self.dialog.secondary_low.setValue(1)
        with patch.object(QtWidgets.QMessageBox,"question",return_value=QtWidgets.QMessageBox.StandardButton.Yes), patch.object(QtWidgets.QMessageBox,"warning") as warning:
            self.dialog.secondary_apply.click(); self.wait_for_worker()
            warning.assert_not_called()
            self.assertIsInstance(self.dialog.source,SecondaryFilteredSource)
            self.assertEqual(set(self.dialog.source.channels),{7})
            self.assertTrue(self.dialog.secondary_restore.isEnabled())
            self.assertEqual(average_trial_ids(self.dialog.output,7,4),(1,))
            row=next(r for r in self.dialog.output["records"] if r.get("channel")==7 and r["trial_id"]==1 and r["branch"]=="peak")
            self.assertEqual(row["extreme_relative_ms"],-100)
            self.assertEqual(row["alignment_origin"],"manual")
            self.assertIs(self.window.active_source,self.source)
            once=self.dialog.source.read(0,self.source.metadata.rows,0).copy()
            self.dialog.secondary_apply.click(); self.wait_for_worker()
            np.testing.assert_array_equal(self.dialog.source.read(0,self.source.metadata.rows,0),once)
            self.dialog.secondary_restore.click(); self.wait_for_worker()
            warning.assert_not_called()
        self.assertIs(self.dialog.source,self.source)
        self.assertEqual(self.dialog.output["secondary_filter_state"],[])
        self.assertEqual(len(self.dialog.output["secondary_filter_history"]),3)
        self.assertTrue(self.dialog._has_unsaved_edits())
        self.assertEqual(average_trial_ids(self.dialog.output,7,4),(1,))
        np.testing.assert_array_equal(self.source.data,original)

    def test_secondary_filter_cancel_keeps_committed_source_and_result(self):
        output=self.show_details()
        entered=threading.Event()
        def blocked(source,result,channels,config,progress,cancel):
            entered.set()
            while not cancel():
                time.sleep(.005)
            raise AnalysisCancelled("取消二次滤波测试")
        with patch("animal_peak_dialog.apply_secondary_filter",side_effect=blocked), patch.object(
            QtWidgets.QMessageBox,"question",return_value=QtWidgets.QMessageBox.StandardButton.Yes,
        ), patch.object(QtWidgets.QMessageBox,"warning") as warning:
            self.dialog.secondary_apply.click()
            self.assertTrue(entered.wait(5))
            self.assertTrue(self.dialog.cancel_button.isEnabled())
            self.assertFalse(self.dialog.secondary_box.isEnabled())
            self.dialog.cancel_button.click(); self.wait_for_worker()
            warning.assert_not_called()
        self.assertIs(self.dialog.source,self.source)
        self.assertIs(self.dialog.output,output)
        self.assertEqual(output["secondary_filter_state"],[])
        self.assertTrue(self.dialog.secondary_apply.isEnabled())

    def test_secondary_filter_invalid_channel_or_pending_parameters_do_not_start_worker(self):
        self.show_details()
        self.dialog.secondary_channels.setText("9999")
        with patch.object(QtWidgets.QMessageBox,"warning") as warning, patch.object(QtWidgets.QMessageBox,"question") as question:
            self.dialog.start_secondary_filter()
            warning.assert_called_once(); question.assert_not_called()
        self.assertIsNone(self.dialog.worker)
        self.dialog.search_spin.setValue(500)
        self.assertFalse(self.dialog.secondary_apply.isEnabled())
        self.assertFalse(self.dialog.secondary_restore.isEnabled())

    def test_secondary_cache_is_released_on_accepted_window_close(self):
        import gc
        self.show_details()
        with patch.object(QtWidgets.QMessageBox,"question",return_value=QtWidgets.QMessageBox.StandardButton.Yes):
            self.dialog.secondary_apply.click(); self.wait_for_worker()
            path=self.dialog.source.channels[7].path
            self.assertTrue(path.is_file())
            self.assertTrue(self.dialog.close())
        self.app.processEvents(); gc.collect()
        self.assertFalse(path.exists())
        self.assertIs(self.dialog.source,self.source)

    def test_trial_legends_removed_without_removing_waveforms_markers_or_trial_title(self):
        output = self.show_details()
        axis = self.dialog.detail_axes[0]
        self.assertIsNone(axis.get_legend())
        self.assertTrue(axis.lines)
        self.assertTrue(axis.collections)  # Peak/trough markers remain.
        self.assertIn("Tag 4", axis.get_title())
        self.assertIn("原 Trial 1", axis.get_title())
        rows = [row for row in output["records"] if row.get("channel") == 7 and row["trial_id"] == 1]
        time_ms, values = trial_detail_trace(self.source, output, rows)
        figure = Figure(); exported = figure.subplots()
        exported.plot([], [], label="旧图例"); exported.legend()
        self.assertIsNotNone(exported.get_legend())
        draw_trial_detail(exported, output, rows, time_ms, values)
        self.assertIsNone(exported.get_legend())

    def test_single_trial_paging_preserves_labels_and_pdfs_stay_four_per_page(self):
        self.show_details()
        self.dialog.details_per_page_box.setCurrentIndex(1); self.app.processEvents()
        self.assertEqual(len(self.dialog.detail_axes), 1)
        self.assertIn("第 1/2 页", self.dialog.page_label.text())
        self.assertIn("原 Trial 1", self.dialog.detail_axes[0].get_title())
        self.dialog.next_button.click(); self.app.processEvents()
        self.assertIn("第 2/2 页", self.dialog.page_label.text())
        self.assertIn("原 Trial 2", self.dialog.detail_axes[0].get_title())
        self.dialog.details_per_page_box.setCurrentIndex(0); self.app.processEvents()
        self.assertEqual(len(self.dialog.detail_axes), 4)
        self.assertIn("第 1/1 页", self.dialog.page_label.text())
        self.assertEqual(self.dialog.detail_canvas.minimumHeight(), 680)

    def test_full_export_runs_in_background_and_saves_detail_labels(self):
        self.dialog.channel_edit.setText("7")
        self.dialog.limit_spin.setValue(1)
        self.dialog.start_analysis(); self.wait_for_worker()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            QtWidgets.QFileDialog, "getExistingDirectory", return_value=directory,
        ), patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.StandardButton.Yes), patch.object(
            QtWidgets.QMessageBox, "warning",
        ) as warning:
            self.dialog.start_export(); self.wait_for_worker()
            warning.assert_not_called()
            folders = list(Path(directory).glob("animal_peak_test_*"))
            self.assertEqual(len(folders), 1)
            self.assertTrue((folders[0]/"ch7_tag4_trial00001_detail.png").is_file())

    def test_main_button_opens_independent_parameters_without_mutation(self):
        window = self.window
        self.configure_completed_alignment(self.source, [7, 22])
        window.lfp_source = self.source
        window.stim_markers = self.markers
        window.stim_marker_mode = "Letter"
        window.behavior_matches = self.matches
        original_baseline = window.baseline_start_ms_var.text()
        window.trials_per_stim_var.setText("2")
        window.compare_trial_a_var.setText("3")
        window.compare_trial_b_var.setText("4")
        window.lfp_task_result_output = {"sentinel": "original Letter result"}
        window.lfp_selected_ids = {7}
        with patch.object(window, "_task_analysis_channel_ids", side_effect=AssertionError("must not read page 4")):
            window.lfp_animal_peak_test_button.click()
        opened = window._animal_peak_test_dialog
        self.assertIsInstance(opened, AnimalPeakDialog)
        self.assertEqual(opened.limit_spin.value(), 0)
        self.assertEqual(opened.limit_spin.text(), "全部")
        self.assertEqual(opened.defaults["trial_limit"], 0)
        self.assertEqual(opened.defaults["channel_ids"], [7, 22])
        self.assertEqual(opened.channel_edit.text(), "7,22")
        self.assertTrue(opened.channel_edit.isReadOnly())
        opened.search_spin.setValue(500)
        opened.post_spin.setValue(200)
        self.assertEqual(window.baseline_start_ms_var.text(), original_baseline)
        self.assertEqual(window.lfp_task_result_output, {"sentinel": "original Letter result"})
        self.assertEqual(window.stim_markers.tolist(), self.markers.tolist())
        opened.close(); self.app.processEvents()

    def test_alignment_snapshot_ignores_pending_selection_and_page4_source(self):
        self.configure_completed_alignment(self.source, [7, 22])
        window = self.window
        window.alignment_selected_ids = {7}  # Not recalculated yet.
        window.lfp_selected_ids = {7}
        window.lfp_source = trend_fixture()[0]  # Only channel 7, not the aligned source.
        source, channels, _key = window._animal_peak_alignment_context()
        self.assertIs(source, self.source)
        self.assertEqual(channels, [7, 22])

    def test_analysis_results_cover_all_completed_alignment_channels(self):
        self.configure_completed_alignment(self.source, [7, 22])
        self.window.lfp_selected_ids = {7}
        self.window.epoch_start_ms_var.setText("-500")
        self.window.epoch_end_ms_var.setText("2000")
        self.window.open_animal_peak_test()
        self.dialog.deleteLater(); self.app.processEvents()
        self.dialog = self.window._animal_peak_test_dialog
        # Read-only UI and calculation both enforce the alignment snapshot.
        self.dialog.channel_edit.setText("7")
        self.dialog.start_analysis(); self.wait_for_worker()
        self.assertEqual(self.dialog.output["settings"]["channel_ids"], [7, 22])
        self.assertEqual({group["channel"] for group in self.dialog.output["groups"]}, {7, 22})
        self.assertEqual({self.dialog.result_channel_box.itemData(index)
                          for index in range(self.dialog.result_channel_box.count())}, {7, 22})

    def multi_tag_output(self):
        matches = {**self.matches, 9250: {"validity": "Y", "animal_latency_sec": 1}}
        return analyze_animal_peaks(self.source, self.markers, matches, self.settings)

    def prepare_similarity(self):
        source, markers, matches, settings = trend_fixture()
        self.dialog.deleteLater(); self.app.processEvents()
        self.dialog = AnimalPeakDialog(self.window, source, markers, matches, settings)
        self.dialog.localization_box.setCurrentIndex(1)
        self.dialog.analysis_finished(analyze_animal_peaks(source, markers, matches, settings))
        self.dialog._set_busy(False)
        return self.dialog

    def test_similarity_defaults_worker_summary_and_no_average_changes(self):
        dialog = self.prepare_similarity()
        self.assertEqual(dialog.judge_r_spin.value(), .7)
        self.assertEqual(dialog.judge_fraction_spin.value(), 60)
        old_groups = dialog.output["groups"]
        old_means = [g["mean"].copy() for g in old_groups]
        self.assertTrue(dialog.similarity_button.isEnabled())
        dialog.start_similarity(); self.wait_for_worker()
        self.assertEqual(dialog.similarity_table.rowCount(), 2)
        self.assertEqual(dialog.similarity_rank_table.rowCount(), 1)
        self.assertIn("参考一致率 1/1＝100.0%", dialog.similarity_summary.text())
        self.assertIn("内部", dialog.similarity_summary.text())
        self.assertIs(dialog.output["groups"], old_groups)
        for group, mean in zip(old_groups, old_means):
            np.testing.assert_array_equal(group["mean"], mean)
        dialog.tabs.setCurrentIndex(2); self.app.processEvents()
        self.assertIn("判断窗趋势", dialog.detail_axes[0].get_title())
        self.assertIn("响应判断窗（趋势相似度）", dialog.detail_axes[0].get_legend_handles_labels()[1])
        self.assertNotIn("旧原波形模板r", " ".join(dialog.detail_axes[0].get_legend_handles_labels()[1]))

    def test_similarity_threshold_change_clears_old_scores_and_reuses_windows(self):
        dialog = self.prepare_similarity()
        dialog.start_similarity(); self.wait_for_worker()
        dialog.judge_r_spin.setValue(.9)
        self.assertTrue(dialog.output["judgment_similarity"]["stale"])
        self.assertEqual(dialog.similarity_table.rowCount(), 0)
        self.assertIn("待重算", dialog.similarity_summary.text())
        self.assertFalse(dialog._parameters_dirty)  # Does not require redoing peaks/averages.
        with patch.object(dialog.source, "read", side_effect=AssertionError("reuse cached windows")):
            dialog.start_similarity(); self.wait_for_worker()
        self.assertFalse(dialog.output["judgment_similarity"]["stale"])
        self.assertEqual(dialog.output["judgment_similarity"]["parameters"]["r_threshold"], .9)

    def test_similarity_reference_and_rank_trial_can_be_selected_without_io(self):
        dialog = self.prepare_similarity()
        dialog.start_similarity(); self.wait_for_worker()
        with patch.object(dialog.source, "read", side_effect=AssertionError("no H5 reread")):
            dialog.similarity_reference_box.setCurrentIndex(dialog.similarity_reference_box.findData(2))
            dialog.similarity_trial_box.setCurrentIndex(dialog.similarity_trial_box.findData(2))
        group = dialog._current_similarity_group()
        self.assertEqual(group["reference_trial_id"], 2)
        self.assertEqual(group["reference_mode"], "manual")
        self.assertEqual(dialog.similarity_rank_table.item(0, 1).text(), "1")
        self.assertEqual(dialog.similarity_table.item(1, 8).text(), "参考本人，不评分")

    def test_manual_adjustment_marks_similarity_stale_until_recomputed(self):
        dialog = self.prepare_similarity()
        dialog.start_similarity(); self.wait_for_worker()
        dialog.tabs.setCurrentIndex(2); self.app.processEvents()
        dialog.manual_time_spin.setValue(-60)
        dialog.start_manual_adjustment(); self.wait_for_worker()
        self.assertTrue(dialog.output["judgment_similarity"]["stale"])
        self.assertEqual(dialog.similarity_table.rowCount(), 0)
        self.assertIn("待重算", dialog.detail_axes[0].get_title())
        dialog.start_similarity(); self.wait_for_worker()
        self.assertFalse(dialog.output["judgment_similarity"]["stale"])
        self.assertEqual(dialog._current_similarity_group()["trials"][0]["peak_ms"], -60)

    def test_similarity_cancellation_keeps_previous_completed_result(self):
        dialog = self.prepare_similarity()
        dialog.start_similarity(); self.wait_for_worker()
        old = dialog.output
        started = threading.Event()
        def slow(source, output, settings, progress, cancel):
            started.set()
            while not cancel():
                time.sleep(.005)
            raise AnalysisCancelled("已取消判断窗测试")
        with patch("animal_peak_dialog.calculate_similarity", side_effect=slow):
            dialog.start_similarity()
            self.assertTrue(started.wait(2))
            self.assertFalse(dialog.similarity_button.isEnabled())
            dialog.cancel(); self.wait_for_worker()
        self.assertIs(dialog.output, old)
        self.assertTrue(dialog.similarity_button.isEnabled())
        self.assertIsNone(self.window._animal_peak_test_worker)

    def test_complete_export_includes_similarity_report_and_tables(self):
        dialog = self.prepare_similarity()
        dialog.start_similarity(); self.wait_for_worker()
        dialog.individual_check.setChecked(False)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            QtWidgets.QFileDialog, "getExistingDirectory", return_value=directory,
        ), patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.StandardButton.Yes), patch.object(
            QtWidgets.QMessageBox, "warning",
        ) as warning:
            dialog.start_export(); self.wait_for_worker()
            warning.assert_not_called()
            folder = next(Path(directory).glob("animal_peak_test_*"))
            self.assertTrue((folder/"response_similarity"/"report.html").is_file())
            self.assertTrue((folder/"response_similarity"/"group_summary.csv").is_file())
            self.assertTrue((folder/"response_similarity"/"all_rankings.csv").is_file())

    def test_main_parameter_edit_disables_similarity_without_discarding_manual_work(self):
        dialog = self.prepare_similarity()
        dialog.start_similarity(); self.wait_for_worker()
        dialog.trend_spin.setValue(35)
        self.assertTrue(dialog._parameters_dirty)
        self.assertFalse(dialog.similarity_button.isEnabled())
        self.assertIsNone(dialog._current_similarity_group())
        self.assertEqual(dialog.similarity_table.rowCount(), 0)

    def test_separate_tag_and_channel_dropdowns_preserve_channel_on_tag_change(self):
        self.dialog.analysis_finished(self.multi_tag_output())
        self.assertEqual(self.dialog.result_tag_box.count(), 2)
        self.assertEqual(self.dialog.result_channel_box.count(), 2)
        self.assertEqual(self.dialog._current_pair(), (7, 4))
        self.dialog.result_channel_box.setCurrentIndex(self.dialog.result_channel_box.findData(22))
        self.assertEqual(self.dialog._current_pair(), (22, 4))
        self.dialog.result_tag_box.setCurrentIndex(self.dialog.result_tag_box.findData(5))
        self.assertEqual(self.dialog._current_pair(), (22, 5))
        self.assertIn("ch22", self.dialog.average_plots[0].plotItem.titleLabel.text)
        self.assertIn("Tag 5", self.dialog.average_plots[0].plotItem.titleLabel.text)
        self.dialog.tabs.setCurrentIndex(2); self.app.processEvents()
        self.assertIn("ch22", self.dialog.detail_axes[0].get_title())
        self.assertIn("Tag 5", self.dialog.detail_axes[0].get_title())

    def test_channel_manual_entry_accepts_number_and_ch_prefix_on_enter(self):
        try:
            from PyQt6.QtTest import QTest
        except ImportError:
            from PyQt5.QtTest import QTest
        self.dialog.analysis_finished(self.multi_tag_output())
        output = self.dialog.output
        self.dialog.result_tag_box.setCurrentIndex(self.dialog.result_tag_box.findData(5))
        for text, expected in (("22", 22), ("ch7", 7), ("CH 22", 22)):
            self.dialog.result_channel_box.setEditText(text)
            key = QtCore.Qt.Key.Key_Return if hasattr(QtCore.Qt, "Key") else QtCore.Qt.Key_Return
            QTest.keyClick(self.dialog.result_channel_box.lineEdit(), key)
            self.app.processEvents()
            self.assertEqual(self.dialog._current_pair(), (expected, 5))
            self.assertEqual(self.dialog.result_channel_box.currentText(), f"ch{expected}")
        self.assertIs(self.dialog.output, output)  # Viewing never recalculates results.

    def test_invalid_manual_channel_keeps_previous_view_and_does_not_add_channels(self):
        self.dialog.analysis_finished(self.multi_tag_output())
        with patch.object(QtWidgets.QMessageBox, "warning") as warning:
            for text in ("999", "wrong", "", "7.5"):
                self.dialog.result_channel_box.setEditText(text)
                self.dialog._commit_result_channel()
                self.assertEqual(self.dialog._current_pair(), (7, 4))
                self.assertEqual(self.dialog.result_channel_box.currentText(), "ch7")
            self.assertEqual(warning.call_count, 4)
        self.assertEqual(self.dialog.result_channel_box.count(), 2)

    def test_tag_without_previous_channel_selects_existing_result_only(self):
        output = self.multi_tag_output()
        output["groups"] = [group for group in output["groups"]
                            if (group["channel"], group["tag"]) != (22, 5)]
        self.dialog.analysis_finished(output)
        self.dialog.result_channel_box.setCurrentIndex(self.dialog.result_channel_box.findData(22))
        self.dialog.result_tag_box.setCurrentIndex(self.dialog.result_tag_box.findData(5))
        self.assertEqual(self.dialog._current_pair(), (7, 5))
        self.assertEqual(self.dialog.result_channel_box.count(), 1)

    def test_manual_update_preserves_both_result_selections(self):
        self.dialog.analysis_finished(self.multi_tag_output())
        self.dialog._restore_result_pair((22, 5))
        self.dialog.tabs.setCurrentIndex(2); self.app.processEvents()
        self.dialog.manual_time_spin.setValue(-50)
        self.dialog.start_manual_adjustment(); self.wait_for_worker()
        self.assertEqual(self.dialog._current_pair(), (22, 5))
        self.assertEqual(self.dialog.output["manual_alignment_history"][-1]["channel"], 22)
        self.assertEqual(self.dialog.output["manual_alignment_history"][-1]["tag"], 5)

    def test_missing_completed_alignment_requires_recalculation(self):
        self.configure_completed_alignment(self.source, [7])
        self.window._task_marker_source = None
        with patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.window.open_animal_peak_test()
        self.assertIn("计算并刷新对齐", warning.call_args.args[2])
        self.assertIsNone(getattr(self.window, "_animal_peak_test_dialog", None))

    def test_missing_or_newly_bad_aligned_channels_do_not_silently_shrink(self):
        self.configure_completed_alignment(self.source, [7, 22, 99])
        with self.assertRaisesRegex(ValueError, "99"):
            self.window._animal_peak_alignment_context()
        self.window._task_marker_source["selected_channel_ids"] = [7, 22]
        self.window.bad_channel_ids = {22}
        with self.assertRaisesRegex(ValueError, "22"):
            self.window._animal_peak_alignment_context()

    def test_changed_alignment_source_requires_recalculation(self):
        self.configure_completed_alignment(self.source, [7])
        self.window._task_marker_source["source_path"] = "different_recording.h5"
        with self.assertRaisesRegex(ValueError, "数据源不一致"):
            self.window._animal_peak_alignment_context()

    def test_visible_test_window_is_not_silently_reused_after_alignment_change(self):
        self.configure_completed_alignment(self.source, [7])
        self.window.open_animal_peak_test()
        opened = self.window._animal_peak_test_dialog
        self.window._task_marker_source["selected_channel_ids"] = [7, 22]
        with patch.object(QtWidgets.QMessageBox, "information") as information:
            self.window.open_animal_peak_test()
        self.assertIn("对齐数据或通道已更新", information.call_args.args[2])
        self.assertEqual(opened.defaults["channel_ids"], [7])
        opened.close(); self.app.processEvents()

    def test_detailed_exclusion_table_contains_evidence_and_full_tooltip(self):
        output = analyze_animal_peaks(self.source, self.markers, self.matches, self.settings)
        self.dialog.analysis_finished(output)
        index = next(i for i, row in enumerate(output["records"]) if row.get("reason_code") == "validity_not_y")
        reason = self.dialog.table.item(index, 11)
        self.assertIn("'N'", reason.text())
        self.assertEqual(reason.toolTip(), reason.text())
        self.assertEqual(self.dialog.table.item(index, 12).text(), "N")
        self.assertEqual(self.dialog.table.item(index, 18).text(), "validity_not_y")

    def test_trend_controls_and_display_switch_do_not_change_raw_average(self):
        source, markers, matches, defaults = trend_fixture()
        self.dialog.deleteLater(); self.app.processEvents()
        self.dialog = AnimalPeakDialog(self.window, source, markers, matches, defaults)
        self.assertEqual(self.dialog.localization_box.currentData(), "raw")
        self.assertFalse(self.dialog.trend_spin.isEnabled())
        self.dialog.localization_box.setCurrentIndex(1)
        self.dialog.despike_spin.setValue(9); self.dialog.trend_spin.setValue(31)
        self.assertFalse(self.dialog.processing_check.isEnabled())
        self.assertTrue(self.dialog.trend_spin.isEnabled())
        self.dialog.start_analysis(); self.wait_for_worker()
        self.assertTrue(self.dialog.export_button.isEnabled())
        self.assertEqual(self.dialog.output["settings"]["localization_mode"], "trend")
        self.assertEqual(self.dialog.output["settings"]["processing"], {})
        means = [group["mean"].copy() for group in self.dialog.output["groups"]]
        self.dialog.tabs.setCurrentIndex(2); self.app.processEvents()
        self.assertTrue(self.dialog.detail_curve_box.isEnabled())
        labels = self.dialog.detail_axes[0].get_legend_handles_labels()[1]
        self.assertIn("定位趋势线（不用于平均）", labels)
        self.assertIn("载入波形（参与平均）", labels)
        self.dialog.detail_curve_box.setCurrentIndex(2); self.app.processEvents()
        labels = self.dialog.detail_axes[0].get_legend_handles_labels()[1]
        self.assertNotIn("载入波形（参与平均）", labels)
        self.assertTrue(self.dialog.export_button.isEnabled())  # Viewing is not a parameter edit.
        for group, previous in zip(self.dialog.output["groups"], means):
            np.testing.assert_array_equal(group["mean"], previous)
        reference = self.dialog.animal_reference_plots[0]
        self.assertTrue(reference.listDataItems())
        self.dialog.mad_spin.setValue(7)
        self.assertFalse(self.dialog.export_button.isEnabled())

    def test_switching_back_to_old_mode_restores_legacy_controls(self):
        self.dialog.localization_box.setCurrentIndex(1)
        self.assertFalse(self.dialog.processing_check.isEnabled())
        self.dialog.localization_box.setCurrentIndex(0)
        self.assertTrue(self.dialog.processing_check.isEnabled())
        self.assertFalse(self.dialog.despike_spin.isEnabled())
        self.assertFalse(self.dialog.trend_spin.isEnabled())

    def test_main_gui_button_runs_new_trend_mode_end_to_end(self):
        source, markers, matches, _ = trend_fixture()
        window = self.window
        self.configure_completed_alignment(source, [7], markers, matches)
        window.lfp_source = source
        window.stim_markers = markers
        window.stim_marker_mode = "Letter"
        window.behavior_matches = matches
        window.epoch_start_ms_var.setText("-500")
        window.epoch_end_ms_var.setText("2000")
        window.baseline_start_ms_var.setText("-200")
        window.baseline_end_ms_var.setText("0")
        window.lfp_animal_peak_test_button.click()
        opened = window._animal_peak_test_dialog
        self.dialog.deleteLater(); self.app.processEvents(); self.dialog = opened
        self.assertIs(opened.source, source)
        opened.localization_box.setCurrentIndex(1)
        opened.search_after_spin.setValue(200)
        opened.despike_spin.setValue(9)
        opened.trend_spin.setValue(31)
        opened.post_spin.setValue(150)
        opened.start_analysis(); self.wait_for_worker()
        self.assertEqual(opened.output["settings"]["localization_mode"], "trend")
        self.assertEqual(len(opened.output["eligible_trials"]), 2)
        peak = next(group for group in opened.output["groups"] if group["branch"] == "peak")
        self.assertLessEqual(abs(peak["rows"][0]["extreme_relative_ms"] + 120), 3)
        self.assertGreater(peak["mean"].max(), 15)
        self.assertTrue(opened.export_button.isEnabled())
        self.assertIsNone(window._animal_peak_test_worker)

    def test_manual_mouse_preview_apply_restore_and_average(self):
        self.show_details()
        old_mean = self.dialog._pairs[(7, 4)]["peak"]["mean"].copy()
        self.dialog.manual_pick_check.setChecked(True)
        self.dialog._pick_manual_point(SimpleNamespace(button=1, inaxes=self.dialog.detail_axes[0], xdata=-100))
        self.assertEqual(self.dialog.manual_time_spin.value(), -100)
        self.assertIsNotNone(self.dialog._manual_preview)
        self.assertNotIn("manual_alignment_history", self.dialog.output)
        self.dialog.manual_apply_button.click(); self.wait_for_worker()
        peak = self.dialog._pairs[(7, 4)]["peak"]
        self.assertEqual(peak["rows"][0]["extreme_relative_ms"], -100)
        self.assertEqual(peak["rows"][0]["alignment_origin"], "manual")
        self.assertEqual(peak["rows"][0]["automatic_relative_ms"], -280)
        self.assertFalse(np.array_equal(old_mean, peak["mean"]))
        self.assertTrue(self.dialog.manual_restore_button.isEnabled())
        self.assertTrue(self.dialog.export_button.isEnabled())
        self.assertTrue(self.dialog.detail_axes[0].get_title().find("人工定位") >= 0)
        self.dialog.manual_restore_button.click(); self.wait_for_worker()
        np.testing.assert_array_equal(self.dialog._pairs[(7, 4)]["peak"]["mean"], old_mean)
        self.assertFalse(self.dialog.manual_restore_button.isEnabled())
        self.assertEqual(len(self.dialog.output["manual_alignment_history"]), 2)

    def test_live_average_preview_uses_applied_raw_mean_and_updates_after_manual_edit(self):
        self.show_details()
        plot = self.dialog.manual_average_plots[0]
        old = self.dialog._pairs[(7, 4)]["peak"]["mean"].copy()
        old_trough = self.dialog._pairs[(7, 4)]["trough"]["mean"].copy()
        np.testing.assert_allclose(plot.listDataItems()[0].getData()[1], old)
        self.dialog.manual_time_spin.setValue(-100)
        # Merely selecting a point is not a committed data change.
        np.testing.assert_allclose(plot.listDataItems()[0].getData()[1], old)
        self.dialog.manual_apply_button.click(); self.wait_for_worker()
        group = self.dialog._pairs[(7, 4)]["peak"]
        expected = np.mean(group["waves"], axis=0)
        self.assertFalse(np.array_equal(expected, old))
        np.testing.assert_array_equal(group["mean"], expected)
        np.testing.assert_allclose(plot.listDataItems()[0].getData()[1], expected)
        np.testing.assert_array_equal(self.dialog._pairs[(7, 4)]["trough"]["mean"], old_trough)
        self.assertIn("人工1 / 自动1", plot.plotItem.titleLabel.text)
        self.assertIn("人工操作版本 1", self.dialog.manual_average_summary.text())
        self.assertEqual(self.dialog.tabs.currentIndex(), 2)  # Continue editing in place.
        self.dialog.manual_restore_button.click(); self.wait_for_worker()
        np.testing.assert_allclose(plot.listDataItems()[0].getData()[1], old)
        self.assertIn("人工0 / 自动2", plot.plotItem.titleLabel.text)

    def test_average_navigation_preserves_manual_point_trial_and_page(self):
        self.show_details()
        self.dialog.details_per_page_box.setCurrentIndex(1)
        self.dialog._change_page(1)
        self.dialog.manual_time_spin.setValue(-100)
        self.dialog.manual_apply_button.click(); self.wait_for_worker()
        self.assertEqual(self.dialog.manual_trial_box.currentData(), 2)
        output = self.dialog.output
        self.dialog.manual_average_view_button.click()
        self.assertEqual(self.dialog.tabs.currentIndex(), 0)
        self.assertIs(self.dialog.output, output)
        np.testing.assert_allclose(self.dialog.average_plots[0].listDataItems()[-1].getData()[1],
                                   self.dialog._pairs[(7, 4)]["peak"]["mean"])
        self.dialog.tabs.setCurrentIndex(2); self.app.processEvents()
        self.assertEqual(self.dialog._detail_page, 1)
        self.assertEqual(self.dialog.manual_trial_box.currentData(), 2)
        self.assertEqual(self.dialog._selected_manual_row()["extreme_relative_ms"], -100)

    def test_average_preview_can_be_hidden_and_uses_current_channel_and_tag(self):
        self.dialog.analysis_finished(self.multi_tag_output())
        self.dialog._restore_result_pair((22, 5))
        self.assertIn("ch22", self.dialog.manual_average_plots[0].plotItem.titleLabel.text)
        self.assertIn("Tag 5", self.dialog.manual_average_plots[0].plotItem.titleLabel.text)
        self.dialog.manual_average_check.setChecked(False)
        self.assertTrue(self.dialog.manual_average_box.isHidden())
        self.dialog.manual_average_check.setChecked(True)
        self.assertFalse(self.dialog.manual_average_box.isHidden())

    def test_average_trial_picker_checkboxes_all_none_invert_and_identifiers(self):
        output=self.show_details()
        picker=AverageTrialSelectionDialog(self.dialog,output,(7,4))
        self.assertEqual(picker.selected_ids(),[1,2])
        self.assertEqual(picker.table.item(0,1).text(),"1")
        self.assertEqual(picker.table.item(0,2).text(),"1")
        picker.select_none_button.click();self.assertEqual(picker.selected_ids(),[])
        picker.invert_button.click();self.assertEqual(picker.selected_ids(),[1,2])
        picker.table.item(0,0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.assertEqual(picker.selected_ids(),[2])
        self.assertIn("1/2",picker.count_label.text())
        picker.select_all_button.click();self.assertEqual(picker.selected_ids(),[1,2])
        picker.deleteLater();self.app.processEvents()

    def test_select_average_trials_redraws_subset_without_changing_points_or_reading_source(self):
        self.show_details()
        self.dialog.manual_time_spin.setValue(-100)
        self.dialog.manual_apply_button.click();self.wait_for_worker()
        full=self.dialog.output
        self.dialog.tabs.setCurrentIndex(0)
        with patch.object(AverageTrialSelectionDialog,"exec",return_value=QtWidgets.QDialog.DialogCode.Accepted),patch.object(
            AverageTrialSelectionDialog,"selected_ids",return_value=[1]),patch.object(self.source,"read",side_effect=AssertionError("no IO")):
            self.dialog.average_trial_button.click()
        self.assertEqual(average_trial_ids(self.dialog.output,7,4),(1,))
        self.assertIs(self.dialog.output["records"],full["records"])
        self.assertIs(self.dialog.output["groups"],full["groups"])
        self.assertEqual(len(self.dialog._pairs[(7,4)]["peak"]["rows"]),2)
        for index,branch in enumerate(("peak","trough")):
            expected=self.dialog._pairs[(7,4)][branch]["waves"][0]
            lines=self.dialog.average_plots[index].listDataItems()
            self.assertEqual(len(lines),2)
            np.testing.assert_allclose(lines[-1].getData()[1],expected)
            np.testing.assert_allclose(self.dialog.manual_average_plots[index].listDataItems()[0].getData()[1],expected)
        self.assertIn("平均选择：1 个",self.dialog.average_refresh_status.text())
        self.assertTrue(self.dialog._has_unsaved_edits())

    def test_cancel_average_trial_picker_keeps_output_and_empty_choice_is_not_all(self):
        self.show_details();self.dialog.tabs.setCurrentIndex(0)
        old=self.dialog.output
        with patch.object(AverageTrialSelectionDialog,"exec",return_value=QtWidgets.QDialog.DialogCode.Rejected):
            self.dialog.choose_average_trials()
        self.assertIs(self.dialog.output,old)
        with patch.object(AverageTrialSelectionDialog,"exec",return_value=QtWidgets.QDialog.DialogCode.Accepted),patch.object(
            AverageTrialSelectionDialog,"selected_ids",return_value=[]):
            self.dialog.choose_average_trials()
        self.assertEqual(average_trial_ids(self.dialog.output,7,4),())
        self.assertFalse(self.dialog.average_plots[0].listDataItems())
        self.assertFalse(self.dialog.manual_average_plots[0].listDataItems())
        self.assertIn("未选中可用",self.dialog.average_plots[0].plotItem.titleLabel.text)
        self.assertTrue(self.dialog.average_trial_button.isEnabled())
        with patch.object(AverageTrialSelectionDialog,"exec",return_value=QtWidgets.QDialog.DialogCode.Accepted),patch.object(
            AverageTrialSelectionDialog,"selected_ids",return_value=[1,2]):
            self.dialog.choose_average_trials()
        self.assertIsNone(average_trial_ids(self.dialog.output,7,4))
        self.assertEqual(len(self.dialog.average_plots[0].listDataItems()),3)

    def test_average_trial_choice_survives_manual_edits_and_window_refresh(self):
        original=self.show_details();self.dialog.tabs.setCurrentIndex(0)
        with patch.object(AverageTrialSelectionDialog,"exec",return_value=QtWidgets.QDialog.DialogCode.Accepted),patch.object(
            AverageTrialSelectionDialog,"selected_ids",return_value=[2]):
            self.dialog.choose_average_trials()
        self.dialog.tabs.setCurrentIndex(2);self.app.processEvents()
        self.dialog.manual_time_spin.setValue(-100)
        self.dialog.manual_apply_button.click();self.wait_for_worker()
        self.assertEqual(average_trial_ids(self.dialog.output,7,4),(2,))
        self.dialog.pre_spin.setValue(200);self.dialog.post_spin.setValue(100)
        self.dialog.refresh_average_plot();self.wait_for_worker()
        self.assertEqual(average_trial_ids(self.dialog.output,7,4),(2,))
        self.assertEqual(self.dialog.manual_average_plots[0].listDataItems()[0].getData()[0].size,301)
        self.assertEqual(len(self.dialog.output["records"]),len(original["records"]))

    def test_refresh_average_button_redraws_latest_committed_overlays_without_io(self):
        self.show_details()
        stale_pairs = self.dialog._pairs
        self.dialog.manual_time_spin.setValue(-100)
        self.dialog.manual_apply_button.click(); self.wait_for_worker()
        output = self.dialog.output
        latest = next(g for g in output["groups"] if g["branch"] == "peak")
        self.dialog.manual_average_view_button.click()
        self.dialog._pairs = stale_pairs  # Refresh must use output, not a stale display index.
        self.dialog.average_plots[0].clear()
        with patch.object(self.source, "read", side_effect=AssertionError("no reread")), patch(
            "animal_peak_dialog.analyze_animal_peaks", side_effect=AssertionError("no detection"),
        ), patch("animal_peak_dialog.adjust_animal_peak", side_effect=AssertionError("no new point update")):
            self.dialog.average_refresh_button.click()
        self.assertIs(self.dialog.output, output)
        self.assertEqual(len(output["manual_alignment_history"]), 1)
        lines = self.dialog.average_plots[0].listDataItems()
        self.assertEqual(len(lines), len(latest["rows"])+1)
        for line, wave in zip(lines[:-1], latest["waves"]):
            np.testing.assert_allclose(line.getData()[1], wave)
        np.testing.assert_allclose(lines[-1].getData()[1], latest["mean"])
        self.assertIn("已刷新", self.dialog.average_refresh_status.text())
        self.assertIn("人工操作版本 1", self.dialog.average_refresh_status.text())

    def test_refresh_keeps_pending_point_unapplied_and_preserves_trial_page(self):
        self.show_details()
        self.dialog.details_per_page_box.setCurrentIndex(1)
        self.dialog._change_page(1)
        self.dialog.manual_time_spin.setValue(-100)
        self.dialog.manual_apply_button.click(); self.wait_for_worker()
        group = self.dialog._pairs[(7, 4)]["peak"]
        self.dialog.manual_time_spin.setValue(-60)  # Preview only.
        self.dialog.manual_average_view_button.click()
        self.dialog.average_refresh_button.click()
        np.testing.assert_allclose(self.dialog.average_plots[0].listDataItems()[-1].getData()[1], group["mean"])
        self.assertEqual(self.dialog._detail_page, 1)
        self.assertEqual(self.dialog.manual_trial_box.currentData(), 2)
        self.assertEqual(self.dialog._selected_manual_row()["extreme_relative_ms"], -100)
        self.assertEqual(len(self.dialog.output["manual_alignment_history"]), 1)

    def test_refresh_average_button_disabled_before_results_and_during_work(self):
        self.assertFalse(self.dialog.average_refresh_button.isEnabled())
        self.dialog.refresh_average_plot()
        self.show_details()
        self.assertTrue(self.dialog.average_refresh_button.isEnabled())
        old = self.dialog.output
        self.dialog._set_busy(True)
        self.assertFalse(self.dialog.average_refresh_button.isEnabled())
        with patch.object(self.dialog, "render_pair") as render:
            self.dialog.refresh_average_plot()
            render.assert_not_called()
        self.assertIs(self.dialog.output, old)
        self.dialog._set_busy(False)
        self.assertTrue(self.dialog.average_refresh_button.isEnabled())

    def test_refresh_after_excerpt_parameter_edit_preserves_manual_points_and_updates_plots(self):
        self.show_details()
        self.dialog.manual_time_spin.setValue(-100)
        self.dialog.manual_apply_button.click(); self.wait_for_worker()
        anchors = [(r["trial_id"],r["branch"],r["extreme_local_sample"],r.get("alignment_origin"))
                   for r in self.dialog.output["records"] if "extreme_local_sample" in r]
        self.dialog.pre_spin.setValue(200); self.dialog.post_spin.setValue(100)
        self.assertTrue(self.dialog._window_parameters_dirty)
        self.assertFalse(self.dialog._other_parameters_dirty)
        self.dialog.tabs.setCurrentIndex(0)
        with patch("animal_peak_dialog.analyze_animal_peaks", side_effect=AssertionError("must not redetect")):
            self.dialog.average_refresh_button.click(); self.wait_for_worker()
        output = self.dialog.output
        self.assertFalse(self.dialog._parameters_dirty)
        self.assertEqual(output["settings"]["pre_ms"],200)
        self.assertEqual(output["settings"]["post_ms"],100)
        self.assertEqual(anchors,[(r["trial_id"],r["branch"],r["extreme_local_sample"],r.get("alignment_origin"))
                                 for r in output["records"] if "extreme_local_sample" in r])
        group = self.dialog._pairs[(7,4)]["peak"]
        self.assertEqual(group["waves"].shape[1],301)
        line = self.dialog.average_plots[0].listDataItems()[-1]
        np.testing.assert_array_equal(line.getData()[0],np.arange(-200,101))
        np.testing.assert_allclose(line.getData()[1],group["mean"])
        self.assertIn("前 200 / 后 100",self.dialog.average_refresh_status.text())
        self.assertEqual(len(output["manual_alignment_history"]),1)
        self.assertEqual(len(output["window_resize_history"]),1)

    def test_refresh_rejects_other_parameter_changes_and_allows_reverting_them(self):
        self.show_details()
        original = self.dialog.output
        old_search = self.dialog.search_spin.value()
        self.dialog.pre_spin.setValue(200)
        self.dialog.search_spin.setValue(old_search+100)
        with patch.object(QtWidgets.QMessageBox,"warning") as warning:
            self.dialog.refresh_average_plot()
        warning.assert_called_once()
        self.assertIs(self.dialog.output,original)
        self.assertIsNone(self.dialog.worker)
        self.dialog.search_spin.setValue(old_search)
        self.assertFalse(self.dialog._other_parameters_dirty)
        self.dialog.refresh_average_plot();self.wait_for_worker()
        self.assertEqual(self.dialog.output["settings"]["pre_ms"],200)

    def test_resize_cancellation_keeps_old_averages_anchors_and_pending_parameters(self):
        self.show_details()
        original = self.dialog.output
        entered = threading.Event()
        def slow(source,output,**kwargs):
            entered.set()
            while not kwargs["cancel"]():
                time.sleep(.005)
            raise AnalysisCancelled("已取消截取窗刷新")
        self.dialog.pre_spin.setValue(200)
        with patch("animal_peak_dialog.resize_animal_peak_windows",side_effect=slow):
            self.dialog.refresh_average_plot()
            self.assertTrue(entered.wait(2))
            self.dialog.cancel();self.wait_for_worker()
        self.assertIs(self.dialog.output,original)
        self.assertEqual(original["settings"]["pre_ms"],50)
        self.assertTrue(self.dialog._window_parameters_dirty)
        self.assertTrue(self.dialog.average_refresh_button.isEnabled())

    def test_excluded_all_by_long_window_keeps_channel_and_can_recover_by_refresh(self):
        self.show_details()
        self.dialog.pre_spin.setValue(30000)
        self.dialog.refresh_average_plot();self.wait_for_worker()
        self.assertEqual(self.dialog._current_pair(),(7,4))
        self.assertFalse(self.dialog.output["groups"])
        self.assertTrue(self.dialog.average_refresh_button.isEnabled())
        self.dialog.pre_spin.setValue(50)
        self.dialog.refresh_average_plot();self.wait_for_worker()
        self.assertTrue(self.dialog.output["groups"])
        self.assertEqual(len(self.dialog.output["window_resize_history"]),2)

    def test_window_resize_history_is_protected_and_exported_with_current_mean(self):
        self.show_details()
        self.dialog.pre_spin.setValue(200);self.dialog.post_spin.setValue(100)
        self.dialog.refresh_average_plot();self.wait_for_worker()
        self.assertTrue(self.dialog._has_unsaved_edits())
        with patch.object(QtWidgets.QMessageBox,"question",return_value=QtWidgets.QMessageBox.StandardButton.No):
            self.assertFalse(self.dialog.close())
        self.dialog.individual_check.setChecked(False)
        with tempfile.TemporaryDirectory() as directory,patch.object(
            QtWidgets.QFileDialog,"getExistingDirectory",return_value=directory),patch.object(
            QtWidgets.QMessageBox,"question",return_value=QtWidgets.QMessageBox.StandardButton.Yes):
            self.dialog.start_export();self.wait_for_worker()
            folder=next(Path(directory).glob("animal_peak_test_*"))
            import json
            manifest=json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["window_resize_history"]),1)
            with (folder/"ch7_tag4_peak_average.csv").open(encoding="utf-8-sig") as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),301)
            self.assertEqual(rows[0]["window_resize_revision"],"1")
            self.assertFalse(self.dialog._has_unsaved_edits())
    def test_manual_average_csv_matches_latest_numeric_result(self):
        self.show_details()
        self.dialog.manual_time_spin.setValue(-100)
        self.dialog.manual_apply_button.click(); self.wait_for_worker()
        self.dialog.individual_check.setChecked(False)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            QtWidgets.QFileDialog, "getExistingDirectory", return_value=directory,
        ), patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.StandardButton.Yes), patch.object(
            QtWidgets.QMessageBox, "warning",
        ) as warning:
            self.dialog.start_export(); self.wait_for_worker()
            warning.assert_not_called()
            folder = next(Path(directory).glob("animal_peak_test_*"))
            with (folder/"ch7_tag4_peak_average.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            group = self.dialog._pairs[(7, 4)]["peak"]
            np.testing.assert_allclose([float(row["mean"]) for row in rows], group["mean"])
            np.testing.assert_allclose([float(row["time_ms"]) for row in rows], group["time_ms"])
            self.assertEqual(rows[0]["unit"], "mV")
            self.assertEqual(rows[0]["manual_anchor_count"], "1")
            self.assertEqual(rows[0]["manual_revision"], "1")

    def test_manual_click_selects_trial_and_does_not_conflict_with_pan(self):
        self.show_details()
        self.dialog.manual_pick_check.setChecked(True)
        self.dialog.manual_branch_box.setCurrentIndex(1)
        event = SimpleNamespace(button=1, inaxes=self.dialog.detail_axes[1], xdata=-210)
        with patch.object(self.dialog.detail_toolbar, "mode", "pan/zoom"):
            self.dialog._pick_manual_point(event)
        self.assertEqual(self.dialog.manual_trial_box.currentData(), 1)
        self.dialog._pick_manual_point(event)
        self.assertEqual(self.dialog.manual_trial_box.currentData(), 2)
        self.assertEqual(self.dialog.manual_time_spin.value(), -210)
        self.dialog.manual_apply_button.click(); self.wait_for_worker()
        trough = self.dialog._pairs[(7, 4)]["trough"]
        self.assertEqual(trough["rows"][1]["extreme_relative_ms"], -210)
        self.assertEqual(self.dialog.manual_trial_box.currentData(), 2)
        self.assertEqual(self.dialog.manual_branch_box.currentData(), "trough")

    def test_parameter_change_blocks_manual_updates_until_reextraction(self):
        self.show_details()
        self.dialog.search_spin.setValue(500)
        self.assertFalse(self.dialog.manual_box.isEnabled())
        old = self.dialog.output
        self.dialog.start_manual_adjustment()
        self.assertIs(self.dialog.output, old)
        self.assertIsNone(self.dialog.worker)

    def test_cancel_manual_recompute_preserves_previous_output(self):
        self.show_details()
        old = self.dialog.output
        entered, release = threading.Event(), threading.Event()
        def blocked(source, output, **kwargs):
            entered.set(); release.wait(5)
            if kwargs["cancel"]():
                raise AnalysisCancelled("已取消人工点位更新")
            return output
        with patch("animal_peak_dialog.adjust_animal_peak", side_effect=blocked), patch.object(
            QtWidgets.QMessageBox, "warning",
        ) as warning:
            try:
                self.dialog.manual_time_spin.setValue(-100)
                self.dialog.start_manual_adjustment()
                self.assertTrue(entered.wait(2))
                self.window.lfp_cancel_button.click()
            finally:
                release.set()
            self.wait_for_worker()
            self.assertIs(self.dialog.output, old)
            self.assertNotIn("manual_alignment_history", old)
            self.assertTrue(self.dialog.manual_apply_button.isEnabled())
            warning.assert_not_called()

    def test_unsaved_manual_work_can_be_exported_and_is_protected_from_reset(self):
        self.show_details()
        self.dialog.manual_time_spin.setValue(-100)
        self.dialog.start_manual_adjustment(); self.wait_for_worker()
        old = self.dialog.output
        self.dialog.search_spin.setValue(500)
        self.assertTrue(self.dialog.export_button.isEnabled())
        self.assertFalse(self.dialog.manual_box.isEnabled())
        with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.StandardButton.No):
            self.dialog.start_analysis()
            self.assertIs(self.dialog.output, old)
            self.assertFalse(self.dialog.close())
            self.assertTrue(self.dialog.isVisible())


if __name__ == "__main__":
    unittest.main()
