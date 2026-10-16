"""Qt integration checks for the new, isolated experimental button/window."""
import os
import tempfile
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
from animal_peak_dialog import AnimalPeakDialog, QtCore, QtWidgets
from animal_peak_analysis import analyze_animal_peaks, trial_detail_trace
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

    def test_threaded_analysis_details_and_parameter_changes(self):
        with patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.dialog.channel_edit.setText("7")
            self.dialog.start_analysis()
            self.wait_for_worker()
            warning.assert_not_called()
            self.assertIsNotNone(self.dialog.output)
            self.assertEqual(self.dialog.pair_box.count(), 1)
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
        self.assertTrue(self.dialog.parameter_scroll.isHidden())
        self.assertFalse(self.dialog.progress.isHidden())
        self.assertFalse(self.dialog.cancel_button.isHidden())
        self.dialog.parameter_toggle.click()
        self.assertFalse(self.dialog.parameter_scroll.isHidden())

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
        np.testing.assert_allclose(ui_axis.get_xlim(), exported_axis.get_xlim())
        np.testing.assert_allclose(ui_axis.get_ylim(), exported_axis.get_ylim())
        for ui_line, exported_line in zip(ui_axis.lines, exported_axis.lines):
            np.testing.assert_array_equal(ui_line.get_xdata(), exported_line.get_xdata())
            np.testing.assert_array_equal(ui_line.get_ydata(), exported_line.get_ydata())
        self.assertEqual([text.get_text() for text in ui_axis.get_legend().texts],
                         [text.get_text() for text in exported_axis.get_legend().texts])
        self.assertEqual(sum(axis.get_visible() for axis in self.dialog.detail_axes), 2)
        expected_limits = ui_axis.get_xlim()
        ui_axis.set_xlim(-2, 2)
        self.dialog.render_details()
        np.testing.assert_allclose(self.dialog.detail_axes[0].get_xlim(), expected_limits)

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
        window.lfp_source = self.source
        window.stim_markers = self.markers
        window.stim_marker_mode = "Letter"
        window.behavior_matches = self.matches
        original_baseline = window.baseline_start_ms_var.text()
        window.trials_per_stim_var.setText("2")
        window.compare_trial_a_var.setText("3")
        window.compare_trial_b_var.setText("4")
        window.lfp_task_result_output = {"sentinel": "original Letter result"}
        with patch.object(window, "_task_analysis_channel_ids", return_value={7}):
            window.lfp_animal_peak_test_button.click()
        opened = window._animal_peak_test_dialog
        self.assertIsInstance(opened, AnimalPeakDialog)
        self.assertEqual(opened.limit_spin.value(), 0)
        self.assertEqual(opened.limit_spin.text(), "全部")
        self.assertEqual(opened.defaults["trial_limit"], 0)
        opened.search_spin.setValue(500)
        opened.post_spin.setValue(200)
        self.assertEqual(window.baseline_start_ms_var.text(), original_baseline)
        self.assertEqual(window.lfp_task_result_output, {"sentinel": "original Letter result"})
        self.assertEqual(window.stim_markers.tolist(), self.markers.tolist())
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
        window.lfp_source = source
        window.stim_markers = markers
        window.stim_marker_mode = "Letter"
        window.behavior_matches = matches
        window.epoch_start_ms_var.setText("-500")
        window.epoch_end_ms_var.setText("2000")
        window.baseline_start_ms_var.setText("-200")
        window.baseline_end_ms_var.setText("0")
        with patch.object(window, "_task_analysis_channel_ids", return_value={7}):
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


if __name__ == "__main__":
    unittest.main()
