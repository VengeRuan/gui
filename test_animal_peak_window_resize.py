"""Changing excerpt lengths must preserve confirmed anchors, not redetect peaks."""
import unittest
from unittest.mock import patch

import numpy as np

from animal_peak_analysis import AnalysisCancelled, analyze_animal_peaks
from animal_peak_manual import adjust_animal_peak, resize_animal_peak_windows
from animal_peak_similarity import calculate_similarity
from test_animal_peak_localization import trend_fixture


class WindowResizeTests(unittest.TestCase):
    def setUp(self):
        self.source, markers, matches, settings = trend_fixture()
        self.automatic = analyze_animal_peaks(self.source, markers, matches, settings)
        self.output = adjust_animal_peak(self.source, self.automatic, 7, 4, 1, "peak", -170)

    @staticmethod
    def anchors(output):
        return [(r["channel"], r["tag"], r["trial_id"], r["branch"], r["extreme_local_sample"],
                 r["extreme_relative_ms"], r.get("alignment_origin"), r.get("manual_edit_count"), r["amplitude"])
                for r in output["records"] if "extreme_local_sample" in r]

    def test_resize_preserves_all_anchors_and_recomputes_raw_averages_and_references(self):
        old_data = self.source.data.copy()
        with patch("animal_peak_analysis.analyze_animal_peaks", side_effect=AssertionError("no detection")):
            result = resize_animal_peak_windows(self.source, self.output, 200, 100)
        self.assertEqual(self.anchors(result), self.anchors(self.output))
        self.assertIs(result["manual_alignment_history"], self.output["manual_alignment_history"])
        self.assertNotIn("window_resize_history", self.output)
        self.assertEqual(result["settings"]["pre_ms"], 200)
        self.assertEqual(result["settings"]["post_ms"], 100)
        self.assertEqual(self.output["settings"]["pre_ms"], 50)
        for group in result["groups"]+result["animal_reference_groups"]:
            self.assertEqual(group["waves"].shape[1], 301)
            np.testing.assert_array_equal(group["time_ms"], np.arange(-200,101))
            for row, wave in zip(group["rows"], group["waves"]):
                center = row.get("extreme_local_sample", row["animal_local_sample"])
                self.assertEqual(row["excerpt_first"], center-200)
                self.assertEqual(row["excerpt_last"], center+101)
                np.testing.assert_array_equal(wave, self.source.data[center-200:center+101, row["column"]]-row["baseline_correction"])
            np.testing.assert_array_equal(group["mean"], np.mean(group["waves"], axis=0))
        np.testing.assert_array_equal(self.source.data, old_data)

    def test_restoring_automatic_point_after_resize_uses_current_lengths(self):
        resized = resize_animal_peak_windows(self.source, self.output, 200, 100)
        restored = adjust_animal_peak(self.source, resized, 7, 4, 1, "peak", restore=True)
        group = next(g for g in restored["groups"] if g["branch"] == "peak")
        row = group["rows"][0]
        self.assertEqual(row["extreme_relative_ms"], -120)
        self.assertEqual(row["alignment_origin"], "automatic")
        self.assertEqual(row["excerpt_last"]-row["excerpt_first"], 301)
        self.assertEqual(group["waves"].shape[1], 301)
        np.testing.assert_array_equal(group["waves"][0], self.source.data[row["excerpt_first"]:row["excerpt_last"],0]-row["baseline_correction"])

    def test_long_windows_keep_points_and_shortening_recovers_excluded_branches(self):
        long = resize_animal_peak_windows(self.source, self.output, 10000, 10000)
        self.assertEqual(self.anchors(long), self.anchors(self.output))
        self.assertFalse(long["groups"])
        self.assertTrue(all(r["status"] == "excluded" and "保留定位点" in r["reason"]
                            for r in long["records"] if "extreme_local_sample" in r))
        recovered = resize_animal_peak_windows(self.source, long, 50, 150)
        self.assertEqual(self.anchors(recovered), self.anchors(self.output))
        self.assertEqual(len(recovered["groups"]), len(self.output["groups"]))
        for new, original in zip(recovered["groups"], self.output["groups"]):
            np.testing.assert_array_equal(new["mean"], original["mean"])

    def test_similarity_scores_invalidated_and_trend_cache_not_reused_after_resize(self):
        scored = calculate_similarity(self.source, self.output)
        resized = resize_animal_peak_windows(self.source, scored, 200, 100)
        self.assertTrue(resized["judgment_similarity"]["stale"])
        self.assertFalse(scored["judgment_similarity"]["stale"])
        with patch.object(self.source, "read", wraps=self.source.read) as read:
            refreshed = calculate_similarity(self.source, resized)
        self.assertGreater(read.call_count, 0)
        self.assertEqual(refreshed["judgment_similarity"]["window_resize_revision"], 1)

    def test_cancel_after_io_is_atomic_and_failure_keeps_old_result(self):
        reads = []
        def progress(value, message):
            reads.append(value)
        with self.assertRaises(AnalysisCancelled):
            resize_animal_peak_windows(self.source, self.output, 200, 100,
                                       progress=progress, cancel=lambda: bool(reads))
        self.assertEqual(self.output["settings"]["pre_ms"], 50)
        self.assertNotIn("window_resize_history", self.output)
        for invalid in ((-1,100),(0,0),(float("nan"),100)):
            with self.assertRaises(ValueError):
                resize_animal_peak_windows(self.source, self.output, *invalid)
        with self.assertRaisesRegex(ValueError, "预算"):
            resize_animal_peak_windows(self.source, {**self.output,"settings":{**self.output["settings"],"memory_budget_bytes":1}}, 200,100)
        with self.assertRaisesRegex(ValueError, "数据源"):
            resize_animal_peak_windows(self.source, {**self.output,"source_identity":()},200,100)


if __name__ == "__main__":
    unittest.main()
