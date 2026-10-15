"""Cooperative cancellation and late-signal regression tests for LFP task jobs."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qt_gui import (
    QApplication, QtAnalysisGUI, TaskEpochWorker, TrialVepWorker,
    ItpcWorker, ExternalBaselineWorker, ExternalTimeFrequencyWorker,
)
from animal_peak_dialog import AnimalPeakDialog, AnimalPeakWorker
from animal_peak_analysis import AnalysisCancelled, analyze_animal_peaks, export_animal_peak_results
from test_animal_peak_analysis import fixture


class GateSource:
    """Pause the first read to click Cancel deterministically during real work."""
    def __init__(self, source):
        self.source = source
        self.metadata = source.metadata
        self.loaded = True
        self.entered = threading.Event()
        self.release = threading.Event()
        self.reads = 0

    def read(self, *args):
        self.reads += 1
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("test read gate was not released")
        return self.source.read(*args)


class LfpCancellationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        with patch("qt_gui.OperationAuditStore"), patch(
            "qt_gui.FeishuAuditConfig.from_environment", return_value=SimpleNamespace(is_configured=False),
        ):
            self.window = QtAnalysisGUI("Cancellation test")
        self.source, self.markers, self.matches, self.settings = fixture()
        self.dialog = None
        self.gate = None

    def tearDown(self):
        if self.gate:
            self.gate.release.set()
        for entry in list(self.window._lfp_cancellable_operations.values()):
            worker = entry["worker"]
            worker.cancel()
            worker.wait(5000)
        self.app.processEvents()
        if self.dialog:
            self.dialog.deleteLater()
        self.window.deleteLater()
        self.app.processEvents()

    def drain(self, worker):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.processEvents()
            if not self.window._lfp_cancellable_operations:
                break
            # Peak workers are deleteLater'd by their dialog; never use them
            # after finished has been handled by processEvents().
            worker.wait(5)
        self.app.processEvents()
        self.assertFalse(self.window._lfp_cancellable_operations)

    def test_pre_cancel_every_worker_does_not_read_or_publish(self):
        workers = [
            TaskEpochWorker(self.source, self.markers, {}),
            TrialVepWorker(self.source, {}, 7, {}),
            ItpcWorker(self.source, {}, {}),
            ExternalBaselineWorker(self.source, [0], [7], {}),
            ExternalTimeFrequencyWorker(self.source, self.source, {}, {}, 7, {}),
            AnimalPeakWorker(self.source, self.markers, self.matches, self.settings),
        ]
        with patch.object(self.source, "read") as read:
            for worker in workers:
                with self.subTest(worker=type(worker).__name__):
                    cancelled, completed, failed = [], [], []
                    worker.cancelled.connect(lambda *args: cancelled.append(args))
                    worker.completed.connect(completed.append)
                    worker.failed.connect(failed.append)
                    worker.cancel()
                    worker.run()
                    self.assertEqual(len(cancelled), 1)
                    self.assertEqual(completed, [])
                    self.assertEqual(failed, [])
            read.assert_not_called()

    def test_cancel_button_is_immediately_right_of_progress(self):
        for progress, cancel in [(self.window.lfp_progress, self.window.lfp_cancel_button)]:
            row = progress.parentWidget().layout()
            # The shared row is nested in a form: search it recursively.
            def adjacent(layout):
                for index in range(layout.count()):
                    item = layout.itemAt(index)
                    if item.widget() is progress:
                        return index + 1 < layout.count() and layout.itemAt(index + 1).widget() is cancel
                    if item.layout() and adjacent(item.layout()):
                        return True
                return False
            self.assertTrue(adjacent(row))
            self.assertGreaterEqual(cancel.minimumWidth(), 90)
            self.assertFalse(cancel.isEnabled())

    def test_real_letter_cancel_stops_after_current_read_and_can_restart(self):
        window = self.window
        self.gate = GateSource(self.source)
        window.lfp_source = self.gate
        window.stim_markers = self.markers
        window.stim_marker_mode = "Letter"
        previous = {"sentinel": "previous completed task"}
        window.lfp_task_result_output = previous
        original = self.source.data.copy()
        with patch.object(window, "_task_analysis_channel_ids", return_value={7}), patch(
            "qt_gui.QMessageBox.critical",
        ) as critical:
            window.run_task_epoch_analysis("lfp", "Letter")
            worker = window._lfp_task_worker
            self.assertTrue(self.gate.entered.wait(2))
            self.app.processEvents()
            self.assertTrue(window.lfp_cancel_button.isEnabled())
            window.lfp_cancel_button.click()
            self.assertIn("取消中", window.lfp_cancel_button.text())
            # Queued progress cannot overwrite the waiting-for-cancel status.
            worker.progress.emit(80, "stale progress")
            self.app.processEvents()
            self.assertIn("正在取消", window.lfp_status.text())
            self.gate.release.set()
            self.drain(worker)
            critical.assert_not_called()
            self.assertEqual(self.gate.reads, 1)
            self.assertIs(window.lfp_task_result_output, previous)
            self.assertIn("已取消", window.lfp_status.text())
            self.assertFalse(window.lfp_cancel_button.isEnabled())
            np.testing.assert_array_equal(original, self.source.data)
            self.assertTrue(window.lfp_letter_task_button.isEnabled())
            # A fresh worker is not poisoned by the previous cancellation flag.
            with patch.object(window, "_finish_task_epoch_analysis") as finish:
                window.run_task_epoch_analysis("lfp", "Letter")
                self.drain(window._lfp_task_worker)
                finish.assert_called_once()

    def test_late_completion_or_failure_after_click_is_ignored(self):
        worker = TaskEpochWorker(self.source, self.markers, {})
        completed, failed, restored = [], [], []
        self.window._connect_lfp_operation(worker, "Letter", completed.append, failed.append,
                                           restore=lambda: restored.append(True))
        self.window._resume_external_time_frequency_after_baseline = True
        self.window.lfp_cancel_button.click()
        worker.completed.emit({"late": True})
        worker.failed.emit("cancel must not open an error dialog")
        worker.finished.emit()
        self.app.processEvents()
        self.assertEqual(completed, [])
        self.assertEqual(failed, [])
        self.assertEqual(restored, [True])
        self.assertFalse(self.window._resume_external_time_frequency_after_baseline)
        self.assertFalse(self.window.lfp_cancel_button.isEnabled())

    def test_success_and_real_failure_still_reach_original_handlers(self):
        for success in (True, False):
            worker = TaskEpochWorker(self.source, self.markers, {})
            completed, failed = [], []
            self.window._connect_lfp_operation(worker, "task", completed.append, failed.append)
            if success:
                worker.completed.emit({"ok": 1})
            else:
                worker.failed.emit("real failure")
            worker.finished.emit()
            self.app.processEvents()
            self.assertEqual(completed, [{"ok": 1}] if success else [])
            self.assertEqual(failed, [] if success else ["real failure"])

    def test_peak_test_cancels_from_either_progress_bar(self):
        for from_main in (False, True):
            with self.subTest(from_main=from_main):
                self.gate = GateSource(self.source)
                dialog = self.dialog = AnimalPeakDialog(self.window, self.gate, self.markers,
                                                       self.matches, self.settings)
                layout = dialog.layout()
                row = next(layout.itemAt(i).layout() for i in range(layout.count())
                           if layout.itemAt(i).layout() and layout.itemAt(i).layout().indexOf(dialog.progress) >= 0)
                self.assertIs(row.itemAt(row.indexOf(dialog.progress) + 1).widget(), dialog.cancel_button)
                with patch("animal_peak_dialog.QtWidgets.QMessageBox.warning") as warning:
                    dialog.start_analysis()
                    worker = dialog.worker
                    self.assertTrue(self.gate.entered.wait(2))
                    button = self.window.lfp_cancel_button if from_main else dialog.cancel_button
                    button.click()
                    self.assertIn("取消中", dialog.cancel_button.text())
                    self.gate.release.set()
                    self.drain(worker)
                    warning.assert_not_called()
                    self.assertFalse(dialog._busy)
                    self.assertIsNone(dialog.output)
                    self.assertIsNone(self.window._animal_peak_test_worker)
                    self.assertIn("取消", dialog.status.text())
                    self.assertTrue(dialog.run_button.isEnabled())
                    self.assertFalse(dialog.cancel_button.isEnabled())
                    self.assertEqual(self.gate.reads, 1)
                dialog.deleteLater(); self.app.processEvents(); self.dialog = None

    def test_cancel_during_vep_read_never_publishes(self):
        self.gate = GateSource(self.source)
        worker = TrialVepWorker(self.gate, {}, 7, {})
        completed, failed = [], []
        self.window._connect_lfp_operation(worker, "VEP", completed.append, failed.append)
        worker.start()
        self.assertTrue(self.gate.entered.wait(2))
        self.window.lfp_cancel_button.click()
        self.gate.release.set()
        self.drain(worker)
        self.assertFalse(completed or failed)
        self.assertEqual(self.gate.reads, 1)

    def test_export_cancel_keeps_marked_partial_files_without_success(self):
        output = analyze_animal_peaks(self.source, self.markers, self.matches, self.settings)
        cancelled = False
        def progress(value, text):
            nonlocal cancelled
            cancelled = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AnalysisCancelled) as caught:
                export_animal_peak_results(output, self.source, directory, False,
                                           progress, lambda: cancelled)
            folders = list(Path(directory).glob("animal_peak_test_*"))
            self.assertEqual(len(folders), 1)
            self.assertIn(str(folders[0]), str(caught.exception))
            manifest = json.loads((folders[0] / "parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "cancelled")
            self.assertTrue((folders[0] / "trial_details.csv").exists())


if __name__ == "__main__":
    unittest.main()
