"""Conservative companion discovery and H5-import integration regressions."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np

from alignment_autofill import CSV_MIN_BYTES, CSV_MAX_BYTES, discover_alignment_files


class CompanionFixture:
    def make_fixture(self, root):
        self.root = Path(root)
        self.logs = self.root / "logfiles" / "20260707" / "1101"
        self.detection = self.root / "processed_data" / "行为学数据处理" / "chars_detected" / "20260707"
        self.logs.mkdir(parents=True); self.detection.mkdir(parents=True)
        self.h5 = self.root / "processed_data" / "20260707" / "1101" / "processed" / "20260707_1101_block2_20260929_100817_lfp_processed.h5"
        self.h5.parent.mkdir(parents=True)
        self.log = self.logs / "20260707_1101_block2.txt"
        self.csv = self.logs / "1101_Letter_experiment_data_20260707_154929_block1.csv"
        self.video = self.detection / "2026-07-07 15-34-08_1101_detection_results.txt"
        self.log.write_text("log fixture", encoding="utf-8")
        self.csv.write_bytes(b"x" * (9 * 1024))
        self.video.write_text("video fixture", encoding="utf-8")

    def discover(self, filename=None):
        return discover_alignment_files(filename or self.h5, self.root)


class AlignmentDiscoveryTests(CompanionFixture, unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.make_fixture(self.temp.name)

    def test_exact_example_ignores_export_timestamp_and_csv_block(self):
        result = self.discover()
        self.assertEqual(result["identity"], {"date": "20260707", "animal": "1101", "block": 2})
        self.assertEqual(result["files"]["log"]["selected"], self.log)
        self.assertEqual(result["files"]["events"]["selected"], self.csv)
        self.assertEqual(result["files"]["detection"]["selected"], self.video)

    def test_csv_size_boundaries_are_inclusive(self):
        for size in (CSV_MIN_BYTES, CSV_MAX_BYTES):
            self.csv.write_bytes(b"x"*size)
            self.assertEqual(self.discover()["files"]["events"]["selected"], self.csv)
        for size in (CSV_MIN_BYTES-1, CSV_MAX_BYTES+1):
            self.csv.write_bytes(b"x"*size)
            result = self.discover()["files"]["events"]
            self.assertIsNone(result["selected"])
            self.assertEqual(result["status"], "missing")
            self.assertIn("大小不符 1", result["message"])

    def test_csv_is_never_loaded_from_processed_folder(self):
        self.csv.rename(self.h5.parent / self.csv.name)
        self.assertIsNone(self.discover()["files"]["events"]["selected"])

    def test_date_animal_and_letter_filename_must_match_exactly(self):
        self.csv.rename(self.logs / self.csv.name.replace("1101_", "11010_"))
        for name in ("1101_Letter_experiment_data_20260708_154929_block1.csv", "1101_processing.csv"):
            (self.logs/name).write_bytes(b"x"*9216)
        self.video.rename(self.detection / self.video.name.replace("_1101_", "_11010_"))
        self.assertIsNone(self.discover()["files"]["events"]["selected"])
        self.assertIsNone(self.discover()["files"]["detection"]["selected"])

    def test_multiple_matches_are_not_resolved_by_time_or_block(self):
        (self.logs / "1101_Letter_experiment_data_20260707_170000_block2.csv").write_bytes(b"x"*9216)
        (self.detection / "2026-07-07 18-00-00_1101_detection_results.txt").write_text("x")
        for kind in ("events", "detection"):
            item = self.discover()["files"][kind]
            self.assertEqual(item["status"], "ambiguous")
            self.assertIsNone(item["selected"])
            self.assertEqual(len(item["candidates"]), 2)

    def test_log_still_requires_h5_block_and_accepts_block_underscore(self):
        self.log.rename(self.logs / "20260707_1101_block_2.txt")
        self.assertEqual(self.discover()["files"]["log"]["status"], "found")
        result = self.discover(self.h5.with_name("20260707_1101_block3.h5"))
        self.assertIsNone(result["files"]["log"]["selected"])
        self.assertEqual(result["files"]["events"]["selected"], self.csv)

    def test_unknown_name_does_not_guess_and_missing_folders_are_safe(self):
        result = self.discover(self.h5.with_name("unknown.h5"))
        self.assertTrue(all(item["status"] == "unrecognized" for item in result["files"].values()))
        result = self.discover(self.h5.with_name("20260230_1101_block2.h5"))
        self.assertIsNone(result["identity"])
        result = self.discover(self.h5.with_name("20260708_1101_block2.h5"))
        self.assertTrue(all(item["status"] == "unavailable" for item in result["files"].values()))

    def test_no_block_can_still_match_csv_and_video(self):
        result = self.discover(self.h5.with_name("20260707_1101_lfp.h5"))
        self.assertIsNone(result["files"]["log"]["selected"])
        self.assertEqual(result["files"]["events"]["selected"], self.csv)
        self.assertEqual(result["files"]["detection"]["selected"], self.video)


class AlignmentAutofillGuiTests(CompanionFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qt_gui import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from qt_gui import QtAnalysisGUI
        self.temp = tempfile.TemporaryDirectory()
        self.make_fixture(self.temp.name)
        with patch("qt_gui.OperationAuditStore"), patch("qt_gui.FeishuAuditConfig.from_environment", return_value=SimpleNamespace(is_configured=False)):
            self.window = QtAnalysisGUI("Autofill UI test")
        self.window._alignment_autofill_root = self.root

    def tearDown(self):
        self.window.deleteLater(); self.app.processEvents()
        self.temp.cleanup()

    def test_fill_is_path_only_and_does_not_compute_alignment(self):
        window = self.window
        with patch.object(window, "run_alignment") as run, patch("qt_gui.parse_alignment_log") as parse:
            window._autofill_alignment_inputs(self.h5)
            run.assert_not_called(); parse.assert_not_called()
        self.assertEqual(window.alignment_log_edit.text(), self.log.as_posix())
        self.assertEqual(window.alignment_event_edit.text(), self.csv.as_posix())
        self.assertEqual(window.alignment_detection_edit.text(), self.video.as_posix())
        self.assertIn("已自动填入", window.alignment_autofill_status.text())

    def test_browser_and_typed_overrides_survive_same_file_retry(self):
        window = self.window
        window._autofill_alignment_inputs(self.h5)
        custom = str(self.root / "manual.csv")
        with patch("qt_gui.QFileDialog.getOpenFileName", return_value=(custom, "")):
            window.choose_alignment_events()
        window.alignment_log_edit.clear()
        window.alignment_log_edit.textEdited.emit("")
        window.alignment_autofill_button.click()
        self.assertEqual(window.alignment_event_edit.text(), custom)
        self.assertEqual(window.alignment_log_edit.text(), "")
        self.assertIn("CSV：手动指定", window.alignment_autofill_status.text())
        self.assertIn("log：手动留空", window.alignment_autofill_status.text())

    def test_new_h5_clears_both_manual_and_automatic_stale_paths(self):
        window = self.window
        window._autofill_alignment_inputs(self.h5)
        window._set_manual_alignment_input("events", "old_manual.csv")
        window._autofill_alignment_inputs(self.h5.with_name("20260708_1101_block2.h5"))
        self.assertTrue(all(edit.text() == "" for edit in window._alignment_input_edits().values()))
        self.assertFalse(window._alignment_manual_inputs)

    def test_new_ambiguity_clears_old_automatic_selection(self):
        window = self.window
        window._autofill_alignment_inputs(self.h5)
        (self.logs / "1101_Letter_experiment_data_20260707_180000_block9.csv").write_bytes(b"x"*9216)
        window._retry_alignment_autofill()
        self.assertEqual(window.alignment_event_edit.text(), "")
        self.assertIn("多个候选", window.alignment_autofill_status.text())
        self.assertIn("block9", window.alignment_event_edit.toolTip())

    def test_lookup_failure_is_not_an_h5_import_failure(self):
        with patch("alignment_autofill.discover_alignment_files", side_effect=PermissionError("test folder unavailable")):
            self.window._autofill_alignment_inputs(self.h5)
        self.assertIn("目录不可用", self.window.alignment_autofill_status.text())

    def test_real_h5_load_and_stage_import_trigger_matching_without_signal_matrix_read(self):
        window = self.window
        with h5py.File(self.h5, "w") as handle:
            handle.create_dataset("raw_data", data=np.zeros((1000, 2), dtype=np.float32))
            handle.attrs["fs"] = 1000.0
        window.path_edit.setText(str(self.h5))
        with patch("qt_gui.QMessageBox.critical") as critical, patch("qt_data_model.LazyH5Source.read", side_effect=AssertionError("auto-fill must not read the H5 signal")):
            window.load_file()
            critical.assert_not_called()
        self.assertEqual(window.alignment_event_edit.text(), self.csv.as_posix())
        window._alignment_autofill_context = None
        with patch("qt_gui.QFileDialog.getOpenFileNames", return_value=([str(self.h5)], "")), patch("qt_gui.QMessageBox.critical") as critical:
            window.import_stage_data("lfp")
            critical.assert_not_called()
        self.assertEqual(window.alignment_detection_edit.text(), self.video.as_posix())


if __name__ == "__main__":
    unittest.main()
