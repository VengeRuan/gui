"""Qt integration checks for the new, isolated experimental button/window."""
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qt_gui import QApplication, QtAnalysisGUI
from animal_peak_dialog import AnimalPeakDialog, QtWidgets
from test_animal_peak_analysis import fixture


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
            self.assertIn("Tag 4", self.dialog.detail_labels[0].text())
            self.assertIn("原Trial 1", self.dialog.detail_labels[0].text())
            self.dialog.post_spin.setValue(70)
            self.assertFalse(self.dialog.export_button.isEnabled())
            self.assertIn("参数已修改", self.dialog.status.text())

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
        window.lfp_task_result_output = {"sentinel": "original Letter result"}
        with patch.object(window, "_task_analysis_channel_ids", return_value={7}):
            window.lfp_animal_peak_test_button.click()
        opened = window._animal_peak_test_dialog
        self.assertIsInstance(opened, AnimalPeakDialog)
        opened.search_spin.setValue(500)
        opened.post_spin.setValue(200)
        self.assertEqual(window.baseline_start_ms_var.text(), original_baseline)
        self.assertEqual(window.lfp_task_result_output, {"sentinel": "original Letter result"})
        self.assertEqual(window.stim_markers.tolist(), self.markers.tolist())
        opened.close(); self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
