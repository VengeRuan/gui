"""A log with several task runs must be explicitly selected, never last-wins."""
import csv
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qt_data_model import (alignment_log_segments, parse_alignment_log,
                           AlignmentSegmentSelectionRequired, ArraySource)


TWO_RUNS = """[19:58:15.934] 2026-09-03 19:58:15.838651
Recording ON Wbuf addr=c0000000 size=256 KB
[2026-09-03 19:58:34.158551]-> Testv: PROGRAM_START:2026-09-03 19:58:33.921131 Task ON
[2026-09-03 19:59:47.229702]-> Testv: PROGRAM_END:2026-09-03 19:59:47.228237 Task ON
[2026-09-03 20:00:24.388364]-> Testv: PROGRAM_START:2026-09-03 20:00:24.279620 Task ON
[2026-09-03 20:40:38.188547]-> Testv: PROGRAM_END:2026-09-03 20:40:38.187176 Task ON
"""


class AlignmentLogSegmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.log = Path(self.temp.name)/"log.txt"

    def tearDown(self):
        self.temp.cleanup()

    def test_two_complete_runs_require_explicit_choice_and_keep_real_marker_time(self):
        self.log.write_text(TWO_RUNS, encoding="utf-8")
        with self.assertRaises(AlignmentSegmentSelectionRequired) as caught:
            parse_alignment_log(self.log, .25)
        self.assertEqual(len(caught.exception.segments), 2)
        first = parse_alignment_log(self.log, .25, segment_index=0)
        second = parse_alignment_log(self.log, .25, segment_index=1)
        self.assertEqual(first["stim_start_text"], "19:58:33.921131")
        self.assertEqual(first["stim_end_text"], "19:59:47.228237")
        self.assertAlmostEqual(first["stim_delay_raw_sec"], 18.082480, places=6)
        self.assertAlmostEqual(first["stim_delay_sec"], 18.332480, places=6)
        self.assertAlmostEqual(first["program_duration_sec"], 73.307106, places=6)
        self.assertEqual(second["stim_start_text"], "20:00:24.279620")
        self.assertAlmostEqual(second["stim_delay_raw_sec"], 128.440969, places=6)
        self.assertAlmostEqual(second["program_duration_sec"], 2413.907556, places=6)
        self.assertEqual((first["program_segment_index"], second["program_segment_index"]), (1, 2))
        self.assertTrue(first["program_segment_complete"])
        self.assertEqual(first["program_segment_count"], 2)

    def test_single_start_only_log_remains_supported(self):
        self.log.write_text("12:00:00.000\nRecording ON\n12:00:05.000 PROGRAM_START\n", encoding="utf-8")
        timing = parse_alignment_log(self.log, -.5)
        self.assertEqual(timing["stim_delay_sec"], 4.5)
        self.assertIsNone(timing["stim_end_text"])
        self.assertFalse(timing["program_segment_complete"])

    def test_repeated_start_and_unpaired_end_do_not_cross_pair(self):
        self.log.write_text("12:00:00 Recording ON\n12:00:01 PROGRAM_END\n"
            "12:00:02 PROGRAM_START\n12:00:03 PROGRAM_START\n12:00:04 PROGRAM_END\n"
            "12:00:05 PROGRAM_END\n12:00:06 PROGRAM_START\n", encoding="utf-8")
        runs = alignment_log_segments(self.log)
        self.assertEqual([r["stim_end_text"] for r in runs], [None, "12:00:04", None])
        self.assertEqual(runs[1]["program_duration_sec"], 1)

    def test_each_run_uses_its_own_preceding_recording_on(self):
        self.log.write_text("11:59:00 old line\n12:00:00 Recording ON\n12:00:01 PROGRAM_START\n"
            "12:00:02 PROGRAM_END\n12:05:00 Recording ON\n12:05:03 PROGRAM_START\n12:05:04 PROGRAM_END\n", encoding="utf-8")
        runs = alignment_log_segments(self.log)
        self.assertEqual([r["stim_delay_sec"] for r in runs], [1, 3])
        self.assertEqual([r["rec_start_text"] for r in runs], ["12:00:00", "12:05:00"])

    def test_new_recording_never_closes_an_old_unfinished_task(self):
        self.log.write_text("12:00:00 Recording ON\n12:00:01 PROGRAM_START\n"
            "12:05:00 Recording ON\n12:05:01 PROGRAM_END\n12:05:02 PROGRAM_START\n12:05:05 PROGRAM_END\n", encoding="utf-8")
        runs = alignment_log_segments(self.log)
        self.assertIsNone(runs[0]["stim_end_text"])
        self.assertEqual(runs[1]["stim_delay_sec"], 2)
        self.assertEqual(runs[1]["program_duration_sec"], 3)

    def test_midnight_rollover_preserves_delay_and_duration(self):
        self.log.write_text("23:59:59 Recording ON\n00:00:01 PROGRAM_START\n00:00:05 PROGRAM_END\n", encoding="utf-8")
        timing = parse_alignment_log(self.log)
        self.assertEqual(timing["stim_delay_sec"], 2)
        self.assertEqual(timing["program_duration_sec"], 4)

    def test_invalid_indices_and_missing_recording_do_not_silently_choose(self):
        self.log.write_text(TWO_RUNS, encoding="utf-8")
        for index in (-1, 2, True, "1"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                parse_alignment_log(self.log, segment_index=index)
        self.log.write_text("12:00:01 PROGRAM_START\n12:00:02 Recording ON\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Recording ON"):
            parse_alignment_log(self.log)


class AlignmentLogSegmentQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qt_gui import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from qt_gui import QtAnalysisGUI
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.log, self.csv = root/"log.txt", root/"events.csv"
        self.log.write_text(TWO_RUNS, encoding="utf-8")
        with self.csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle); writer.writerow([f"c{i}" for i in range(15)])
            for dot, letter in ((2, 5), (100, 105)):
                row = [""]*15; row[1], row[12], row[13] = "o", dot, letter; writer.writerow(row)
        with patch("qt_gui.OperationAuditStore"), patch("qt_gui.FeishuAuditConfig.from_environment",
                return_value=SimpleNamespace(is_configured=False)):
            self.window = QtAnalysisGUI("Alignment segment test")
        w = self.window
        w.active_source = ArraySource(np.zeros((400000, 1), dtype=np.float32), 1000, channel_ids=[7])
        w._bad_channel_check_completed = True; w.good_channel_ids = {7}; w.bad_channel_ids = set()
        w.alignment_selected_ids = {7}; w._alignment_use_all_healthy = False
        w.bin_timing_metadata = {"deltaT1_sec": .25}
        w.alignment_log_edit.setText(str(self.log)); w.alignment_event_edit.setText(str(self.csv))
        w.alignment_detection_edit.clear()

    def tearDown(self):
        self.window.deleteLater(); self.app.processEvents(); self.temp.cleanup()

    def run_choice(self, index=None, cancelled=False):
        from qt_gui import QMessageBox
        def select(*args):
            self.labels = args[3]
            return (self.labels[index or 0], not cancelled)
        with patch("qt_gui.QInputDialog.getItem", side_effect=select) as dialog, \
             patch.object(self.window, "refresh_alignment_channel_preview"), \
             patch.object(self.window, "_inherit_alignment_selection_for_lfp"), \
             patch.object(self.window, "_update_file_banners"), \
             patch.object(QMessageBox, "critical") as critical:
            self.window.align_run_button.click()
        critical.assert_not_called()
        return dialog

    def test_each_choice_controls_markers_dot_and_audit_without_end_cropping(self):
        w = self.window
        for index, delay in ((0, 18.332480), (1, 128.690969)):
            with self.subTest(index=index):
                dialog = self.run_choice(index)
                dialog.assert_called_once()
                self.assertIn("19:58:33.921131", self.labels[0]); self.assertIn("19:59:47.228237", self.labels[0])
                self.assertIn("20:40:38.187176", self.labels[1])
                letters = w.stim_markers[np.isin(w.stim_markers[:, 1], [4, 5, 6, 7])]
                np.testing.assert_array_equal(letters[:, 0], np.rint((np.array([5, 105])+delay)*1000).astype(int))
                self.assertEqual(len(letters), 2)  # END never clips CSV offsets.
                self.assertEqual(w.timing_info["program_segment_index"], index+1)
                self.assertIn(f"第 {index+1}/2 段", w.alignment_results.toPlainText())
                self.assertAlmostEqual(w.letter_csv_annotations[int(letters[0, 0])]["tag100_start_eeg_sec"], 2+delay)

    def test_cancel_preserves_previous_markers_matches_timing_and_channel_selection(self):
        w = self.window
        old = np.asarray([[10, 4]])
        w.stim_markers = old; w.timing_info = {"previous": True}; w.behavior_matches = {10: {"previous": True}}
        w.letter_csv_annotations = {"previous": True}
        self.run_choice(cancelled=True)
        self.assertIs(w.stim_markers, old)
        self.assertEqual(w.timing_info, {"previous": True})
        self.assertEqual(w.behavior_matches, {10: {"previous": True}})
        self.assertEqual(w.letter_csv_annotations, {"previous": True})
        self.assertEqual(w.alignment_selected_ids, {7})
        self.assertIn("原结果保留", w.alignment_status.text())

    def test_single_run_does_not_open_selector(self):
        self.log.write_text("12:00:00 Recording ON\n12:00:01 PROGRAM_START\n12:02:00 PROGRAM_END\n", encoding="utf-8")
        dialog = self.run_choice()
        dialog.assert_not_called()
        self.assertEqual(self.window.timing_info["program_segment_count"], 1)


if __name__ == "__main__":
    unittest.main()
