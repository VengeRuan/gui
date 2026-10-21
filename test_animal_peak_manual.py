"""Manual anchors re-extract original samples and never mutate prior results."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from animal_peak_analysis import AnalysisCancelled, analyze_animal_peaks, export_animal_peak_results
from animal_peak_manual import adjust_animal_peak
from test_animal_peak_localization import trend_fixture


class ManualAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.source, self.markers, self.matches, self.settings = trend_fixture()
        self.output = analyze_animal_peaks(self.source, self.markers, self.matches, self.settings)

    def edit(self, output=None, **kwargs):
        return adjust_animal_peak(self.source, self.output if output is None else output,
                                  7, 4, 1, "peak", **kwargs)

    @staticmethod
    def group(output, branch="peak"):
        return next(group for group in output["groups"] if group["branch"] == branch)

    def test_manual_position_does_not_snap_and_updates_original_sample_average(self):
        original = self.source.data.copy()
        old_group = self.group(self.output)
        old_waves, old_mean = old_group["waves"].copy(), old_group["mean"].copy()
        old_rows = [dict(row) for row in old_group["rows"]]
        result = self.edit(relative_ms=-170)
        group = self.group(result)
        row = group["rows"][0]
        self.assertEqual(row["extreme_local_sample"], 1830)
        self.assertEqual(row["extreme_relative_ms"], -170)
        self.assertEqual(row["automatic_relative_ms"], -120)
        self.assertEqual(row["alignment_origin"], "manual")
        np.testing.assert_array_equal(group["waves"][0], self.source.data[1780:1981, 0] - row["baseline_correction"])
        np.testing.assert_array_equal(group["mean"], np.mean(group["waves"], axis=0))
        np.testing.assert_array_equal(group["waves"][1], old_waves[1])
        self.assertIs(self.group(result, "trough")["waves"], self.group(self.output, "trough")["waves"])
        self.assertIs(result["animal_reference_groups"], self.output["animal_reference_groups"])
        np.testing.assert_array_equal(self.source.data, original)
        np.testing.assert_array_equal(old_group["waves"], old_waves)
        np.testing.assert_array_equal(old_group["mean"], old_mean)
        self.assertEqual(old_group["rows"], old_rows)
        self.assertNotIn("manual_alignment_history", self.output)
        for other in group["rows"]:
            self.assertIs(other, next(item for item in result["records"] if item["trial_id"] == other["trial_id"] and item["branch"] == "peak"))

    def test_repeated_edits_preserve_auto_anchor_and_restore_exact_mean(self):
        edited = self.edit(relative_ms=-170)
        edited_twice = self.edit(edited, relative_ms=-220)
        restored = self.edit(edited_twice, restore=True)
        np.testing.assert_array_equal(self.group(restored)["waves"], self.group(self.output)["waves"])
        np.testing.assert_array_equal(self.group(restored)["mean"], self.group(self.output)["mean"])
        row = self.group(restored)["rows"][0]
        self.assertEqual(row["alignment_origin"], "automatic")
        self.assertEqual(row["automatic_relative_ms"], -120)
        self.assertEqual(row["manual_edit_count"], 3)
        self.assertEqual(len(restored["manual_alignment_history"]), 3)
        self.assertEqual(restored["manual_alignment_history"][-1]["action"], "restore_automatic")
        self.assertEqual(len(edited["manual_alignment_history"]), 1)

    def test_invalid_edits_do_not_modify_last_successful_result(self):
        previous = self.group(self.output)["mean"].copy()
        for value in (-301, 200, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.edit(relative_ms=value)
        with self.assertRaises(ValueError):
            adjust_animal_peak(self.source, self.output, 7, 4, 999, "peak", relative_ms=-100)
        np.testing.assert_array_equal(self.group(self.output)["mean"], previous)

    def test_cancel_during_recompute_is_transactional(self):
        cancelled = False
        def progress(value, text):
            nonlocal cancelled
            cancelled = value >= 10
        with self.assertRaises(AnalysisCancelled):
            self.edit(relative_ms=-170, progress=progress, cancel=lambda: cancelled)
        self.assertNotIn("manual_alignment_history", self.output)
        self.assertEqual(self.group(self.output)["rows"][0]["extreme_relative_ms"], -120)

    def test_excluded_excerpt_can_be_recovered_and_restored_without_other_trials_changing(self):
        settings = {**self.settings, "pre_ms": 2000}
        output = analyze_animal_peaks(self.source, self.markers, self.matches, settings)
        self.assertEqual(len(self.group(output)["rows"]), 1)
        result = self.edit(output, relative_ms=150)
        self.assertEqual(len(self.group(result)["rows"]), 2)
        restored = self.edit(result, restore=True)
        self.assertEqual(len(self.group(restored)["rows"]), 1)
        row = next(row for row in restored["records"] if row["trial_id"] == 1 and row["branch"] == "peak")
        self.assertEqual(row["status"], "excluded")
        self.assertEqual(row["reason_code"], "excerpt_before_data")
        with self.assertRaisesRegex(ValueError, "超出源数据"):
            self.edit(output, relative_ms=-200)

    def test_missing_raw_sample_is_not_manually_filled(self):
        self.source.data[1900, 0] = np.nan
        output = analyze_animal_peaks(self.source, self.markers, self.matches, self.settings)
        with self.assertRaisesRegex(ValueError, "有效数值"):
            self.edit(output, relative_ms=-100)

    def test_only_selected_trough_changes_and_timing_is_quantized(self):
        result = adjust_animal_peak(self.source, self.output, 7, 4, 2, "trough", relative_ms=-159.6)
        group = self.group(result, "trough")
        row = group["rows"][1]
        self.assertEqual(row["extreme_relative_ms"], -160)
        self.assertEqual(row["extreme_sample"], 4840)
        self.assertEqual(result["manual_alignment_history"][-1]["requested_relative_ms"], -159.6)
        self.assertIs(self.group(result)["waves"], self.group(self.output)["waves"])

    def test_export_contains_manual_positions_history_and_recomputed_mean(self):
        result = self.edit(relative_ms=-170)
        with tempfile.TemporaryDirectory() as directory:
            folder = export_animal_peak_results(result, self.source, directory, individual_png=False)
            audit = json.loads((folder / "manual_alignment.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["history"][0]["new_relative_ms"], -170)
            self.assertEqual(audit["anchors"][0]["automatic_alignment"]["extreme_relative_ms"], -120)
            with (folder / "trial_details.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            row = next(row for row in rows if row["trial_id"] == "1" and row["branch"] == "peak")
            self.assertEqual(row["alignment_origin"], "manual")
            self.assertEqual(float(row["automatic_relative_ms"]), -120)
            with np.load(folder / "ch7_tag4_peak.npz") as saved:
                np.testing.assert_array_equal(saved["mean"], self.group(result)["mean"])
                self.assertEqual(saved["alignment_origin"][0], "manual")
                self.assertEqual(saved["anchor_global_samples"][0], 1830)


if __name__ == "__main__":
    unittest.main()
