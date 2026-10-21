"""Detailed exclusion diagnostics must not change numerical eligibility rules."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from animal_peak_analysis import analyze_animal_peaks
from qt_data_model import read_behavior_detection_results, match_behavior_events_to_markers
from test_animal_peak_analysis import fixture


class ExclusionReasonTests(unittest.TestCase):
    def test_each_behavior_failure_has_its_own_code_and_specific_evidence(self):
        source, markers, _, settings = fixture()
        cases = [
            ({"validity": "N", "animal_latency_sec": 1}, "validity_not_y", "'N'"),
            ({"validity": "", "animal_latency_sec": 1}, "validity_missing", "第5列"),
            ({"validity": "Y", "animal_latency_sec": np.nan, "animal_start_frame": None,
              "animal_start_raw": "NA", "animal_start_status": "missing"}, "animal_frame_missing", "'NA'"),
            ({"validity": "Y", "animal_latency_sec": np.nan, "animal_start_frame": None,
              "animal_start_raw": "bad", "animal_start_status": "unparseable"}, "animal_frame_unparseable", "'bad'"),
            ({"validity": "Y", "animal_latency_sec": np.nan, "animal_start_frame": 100,
              "start_frame": 100}, "animal_frame_equal_start", "AnimalStartFrame=100"),
            ({"validity": "Y", "animal_latency_sec": np.nan, "animal_start_frame": 90,
              "start_frame": 100}, "animal_frame_before_start", "AnimalStartFrame=90"),
            ({"validity": "Y", "animal_latency_sec": np.inf, "animal_start_frame": 110,
              "start_frame": 100}, "animal_latency_nonfinite", "inf"),
        ]
        for event, code, evidence in cases:
            with self.subTest(code=code):
                output = analyze_animal_peaks(source, markers[:1], {3250: event}, settings)
                self.assertFalse(output["groups"])
                record = output["records"][0]
                self.assertEqual(record["reason_code"], code)
                self.assertIn(evidence, record["reason"])
                self.assertNotIn("或", record["reason"])
                self.assertEqual(record["validity"], event["validity"])

    def test_epoch_and_search_diagnostics_identify_the_exact_boundary(self):
        source, _, _, settings = fixture()
        cases = [
            (1250, 1, {}, "epoch_before_data", "[-500, 2000)"),
            (20250, 1, {}, "epoch_after_data", "[18500, 21000)"),
            (3250, -1, {}, "animal_before_epoch", "-1000.000 ms"),
            (3250, 2, {}, "animal_at_epoch_end", "正好位于"),
            (3250, 2.2, {}, "animal_after_epoch", "2200.000 ms"),
            (3250, 1.9, {"search_after_ms": 200}, "search_after_epoch", "2100.000 ms"),
        ]
        for sample, latency, extra, code, evidence in cases:
            with self.subTest(code=code):
                event = {"validity": "Y", "animal_latency_sec": latency}
                output = analyze_animal_peaks(source, [[sample, 4]], {sample: event}, {**settings, **extra})
                record = output["records"][0]
                self.assertEqual(record["reason_code"], code)
                self.assertIn(evidence, record["reason"])
                self.assertNotIn("或", record["reason"])

    def test_missing_behavior_record_identifies_marker_and_tag(self):
        source, markers, _, settings = fixture()
        output = analyze_animal_peaks(source, markers[:1], {}, settings)
        record = output["records"][0]
        self.assertEqual(record["reason_code"], "behavior_unmatched")
        self.assertIn("marker_sample=3250", record["reason"])
        self.assertIn("Tag=4", record["reason"])

    def test_baseline_and_search_numerical_failures_are_not_combined(self):
        for first, last, code in [(1800, 2001, "baseline_no_finite"), (2700, 3000, "search_no_finite")]:
            with self.subTest(code=code):
                source, markers, matches, settings = fixture()
                source.data[first:last, 0] = np.nan
                output = analyze_animal_peaks(source, markers[:1], matches, settings)
                record = next(row for row in output["records"] if row.get("channel") == 7)
                self.assertEqual(record["reason_code"], code)
                self.assertIn(f"[{first}, {last})", record["reason"])
                self.assertNotIn("或", record["reason"])
                self.assertTrue(any(group["channel"] == 22 for group in output["groups"]))

    def test_excerpt_failure_is_branch_specific_and_reports_bounds(self):
        source, markers, matches, settings = fixture()
        for extra, code in [({"pre_ms": 4000}, "excerpt_before_data"),
                            ({"post_ms": 20000}, "excerpt_after_data")]:
            with self.subTest(code=code):
                output = analyze_animal_peaks(source, markers[:1], matches, {**settings, **extra})
                self.assertTrue(output["eligible_trials"])
                for row in output["records"]:
                    self.assertEqual(row["reason_code"], code)
                    self.assertIn("[0, 20000)", row["reason"])
                    self.assertIn("仅排除本峰谷分支", row["reason"])
                    self.assertIn("extreme_local_sample", row)

    def test_detection_raw_values_and_line_numbers_survive_matching(self):
        text = "FrameRate: 30\n\n4, 1, 80, NA, Y\n4, 91, 180, oops, Y\n4, 181, 270, 181, Y\n4, 271, 360, 280, N\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detection.txt"
            path.write_text(text, encoding="utf-8")
            fps, recording, events = read_behavior_detection_results(path)
        self.assertEqual(events[0]["detection_line_number"], 3)
        self.assertEqual(events[1]["animal_start_status"], "unparseable")
        markers = np.asarray([[3250, 4], [6250, 4], [9250, 4], [12250, 4]])
        matches = match_behavior_events_to_markers(markers, 1000, fps, recording, np.nan, events)
        self.assertEqual(matches[3250]["animal_start_raw"], "NA")
        self.assertEqual(matches[6250]["animal_start_raw"], "oops")
        self.assertEqual(matches[6250]["detection_line_number"], 4)
        source, _, _, settings = fixture()
        output = analyze_animal_peaks(source, markers, matches, settings)
        self.assertEqual([row["reason_code"] for row in output["records"]], [
            "animal_frame_missing", "animal_frame_unparseable", "animal_frame_equal_start", "validity_not_y",
        ])


if __name__ == "__main__":
    unittest.main()
