"""Window geometry, reproducibility denominators, isolation and report tests."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from animal_peak_analysis import AnalysisCancelled, analyze_animal_peaks, trial_detail_localization
from animal_peak_manual import adjust_animal_peak
from animal_peak_similarity import (calculate_similarity, compare_windows, export_similarity_report,
                                    ranked_matches, similarity_config, summarize_group)
from test_animal_peak_localization import trend_fixture
from test_animal_peak_analysis import fixture


class SimilarityTests(unittest.TestCase):
    def setUp(self):
        self.source, markers, matches, settings = trend_fixture()
        self.output = analyze_animal_peaks(self.source, markers, matches, settings)

    def test_defaults_and_invalid_parameters(self):
        config = similarity_config()
        self.assertEqual(config["r_threshold"], .7)
        self.assertEqual(config["required_fraction"], .6)
        for invalid in ({"r_threshold": 1.1}, {"before_ms": -1}, {"required_fraction": 0},
                        {"min_window_coverage": np.nan}):
            with self.assertRaises(ValueError):
                similarity_config(invalid)

    def test_correlation_ignores_offset_scale_but_not_polarity(self):
        wave = np.sin(np.linspace(0, 4, 100))
        self.assertAlmostEqual(compare_windows(wave, wave*3+20)[0], 1)
        self.assertAlmostEqual(compare_windows(wave, -wave)[0], -1)
        self.assertTrue(np.isnan(compare_windows(np.ones(20), np.ones(20))[0]))

    def test_no_stretch_or_lag_search_and_invalid_pairs_are_not_zero(self):
        wave = np.sin(np.linspace(0, 5, 100))
        r, count, coverage, code = compare_windows(wave, wave[:40])
        self.assertTrue(np.isnan(r)); self.assertEqual(count, 40)
        self.assertAlmostEqual(coverage, .4); self.assertEqual(code, 3)
        missing = wave.copy(); missing[:30] = np.nan
        self.assertEqual(compare_windows(wave, missing)[3], 3)
        self.assertEqual(compare_windows(wave[:2], wave[:2])[3], 2)

    def test_windows_use_both_final_anchors_and_independent_extensions(self):
        result = calculate_similarity(self.source, self.output, {"before_ms": 30, "after_ms": 80})
        group = result["judgment_similarity"]["groups"][0]
        for trial in group["trials"]:
            anchors = [row["extreme_local_sample"] for row in self.output["records"]
                       if row["trial_id"] == trial["trial_id"] and "extreme_local_sample" in row]
            self.assertEqual(trial["first"], min(anchors)-30)
            self.assertEqual(trial["last"], max(anchors)+80+1)
        self.assertTrue(np.isnan(group["matrix"].diagonal()).all())
        self.assertGreater(group["matrix"][0, 1], .99)
        self.assertEqual(group["reference_trial_id"], 1)
        self.assertEqual(group["comparable_count"], 1)  # Never count reference itself.
        self.assertEqual(group["reference_fraction"], 1)
        self.assertTrue(group["reference_threshold_met"])
        self.assertNotIn("judgment_similarity", self.output)
        self.assertIs(result["groups"], self.output["groups"])
        for original, unchanged in zip(self.output["groups"], result["groups"]):
            self.assertIs(original["mean"], unchanged["mean"])

    def test_judgment_trend_matches_display_even_when_window_expands(self):
        result = calculate_similarity(self.source, self.output, {"before_ms": 600, "after_ms": 500})
        group = result["judgment_similarity"]["groups"][0]
        trial = group["trials"][0]
        rows = [row for row in result["records"] if row["trial_id"] == trial["trial_id"]]
        trace = trial_detail_localization(self.source, result, rows)
        animal = rows[0]["animal_local_sample"]
        first_ms = (trial["first"]-animal)/result["fs"]*1000
        start = int(round((first_ms-trace["time_ms"][0])*result["fs"]/1000))
        np.testing.assert_array_equal(trial["trend"], trace["trend"][start:start+len(trial["trend"])] )

    def test_threshold_changes_reuse_window_reads(self):
        result = calculate_similarity(self.source, self.output)
        with patch.object(self.source, "read", side_effect=AssertionError("must reuse trend windows")):
            changed = calculate_similarity(self.source, result, {"r_threshold": .95})
        self.assertEqual(changed["judgment_similarity"]["parameters"]["r_threshold"], .95)
        self.assertIs(changed["judgment_similarity"]["groups"][0]["trials"],
                      result["judgment_similarity"]["groups"][0]["trials"])

    def test_manual_edit_invalidates_scores_and_uses_new_point(self):
        result = calculate_similarity(self.source, self.output)
        changed = adjust_animal_peak(self.source, result, 7, 4, 1, "peak", -50)
        self.assertTrue(changed["judgment_similarity"]["stale"])
        self.assertFalse(result["judgment_similarity"]["stale"])
        recalculated = calculate_similarity(self.source, changed)
        self.assertEqual(recalculated["judgment_similarity"]["anchor_revision"], 1)
        trial = recalculated["judgment_similarity"]["groups"][0]["trials"][0]
        self.assertEqual(trial["peak_ms"], -50)
        self.assertEqual(trial["peak_origin"], "manual")

    def test_missing_and_out_of_bounds_windows_are_explicit_not_trial_deletions(self):
        output = {**self.output, "records": [row for row in self.output["records"]
                  if not(row["trial_id"] == 1 and row["branch"] == "trough")]}
        result = calculate_similarity(self.source, output)
        group = result["judgment_similarity"]["groups"][0]
        self.assertIn("缺少", group["trials"][0]["reason"])
        self.assertEqual(len(group["trials"]), 2)
        self.assertEqual(group["comparable_count"], 0)
        self.assertFalse(group["reference_threshold_met"])
        self.assertIn("缺少", list(ranked_matches(group, 1, 1000))[0]["reason"])
        out = calculate_similarity(self.source, self.output, {"before_ms": 10000})
        self.assertIn("超出源数据", out["judgment_similarity"]["groups"][0]["trials"][0]["reason"])

    @staticmethod
    def numerical_group(matrix):
        n = len(matrix)
        return {"channel": 7, "tag": 4, "trials": [{"trial_id": i+1, "tag_trial_id": i+1, "reason": ""} for i in range(n)],
                "matrix": matrix, "counts": np.ones((n, n), dtype=int)*100,
                "coverage": np.ones((n, n)), "reason_codes": np.zeros((n, n), dtype=int)}

    def test_exact_threshold_inclusive_ranking_and_manual_reference(self):
        matrix = np.asarray([[np.nan, .7, .9], [.7, np.nan, -.8], [.9, -.8, np.nan]])
        group = summarize_group(self.numerical_group(matrix), {})
        self.assertEqual(group["reference_trial_id"], 1)
        self.assertEqual(group["similar_count"], 2)
        self.assertEqual([row["other_trial_id"] for row in ranked_matches(group, 1, 1000)], [3, 2])
        changed = summarize_group(group, {}, reference_trial=2)
        self.assertEqual(changed["reference_mode"], "manual")
        self.assertEqual(changed["reference_fraction"], .5)
        self.assertFalse(changed["reference_threshold_met"])
        self.assertEqual(group["reference_fraction"], 1)

    def test_reference_consistency_is_not_all_pair_median_or_subgroup_fraction(self):
        matrix = np.zeros((20, 20)); matrix[:12, :12] = .9; np.fill_diagonal(matrix, np.nan)
        group = summarize_group(self.numerical_group(matrix), {})
        self.assertEqual(group["similar_count"], 11)
        self.assertAlmostEqual(group["reference_fraction"], 11/19)
        self.assertEqual(group["matched_group_fraction"], .6)
        self.assertEqual(group["internal_median_r"], .9)
        self.assertFalse(group["reference_threshold_met"])

    def test_sparse_comparisons_cannot_produce_misleading_pass(self):
        matrix = np.full((6, 6), np.nan); matrix[0, 1] = matrix[1, 0] = .95
        group = summarize_group(self.numerical_group(matrix), {})
        self.assertEqual(group["reference_fraction"], 1)
        self.assertEqual(group["reference_coverage"], .2)
        self.assertFalse(group["reference_threshold_met"])

    def test_reference_neighborhood_internal_consistency_is_separately_reported(self):
        matrix = np.eye(4); matrix[0, :] = .8; matrix[:, 0] = .8; np.fill_diagonal(matrix, np.nan)
        group = summarize_group(self.numerical_group(matrix), {})
        self.assertEqual(group["reference_fraction"], 1)
        self.assertEqual(group["internal_median_r"], .4)
        self.assertEqual(group["internal_similar_pair_fraction"], .5)

    def test_guard_raw_mode_source_mismatch_memory_and_cancellation(self):
        raw = {**self.output, "settings": {**self.output["settings"], "localization_mode": "raw"}}
        with self.assertRaisesRegex(ValueError, "趋势"):
            calculate_similarity(self.source, raw)
        with self.assertRaisesRegex(ValueError, "数据源"):
            calculate_similarity(self.source, {**self.output, "source_identity": ()})
        with self.assertRaisesRegex(ValueError, "预算"):
            calculate_similarity(self.source, {**self.output, "settings": {**self.output["settings"], "memory_budget_bytes": 1}})
        with self.assertRaises(AnalysisCancelled):
            calculate_similarity(self.source, self.output, cancel=lambda: True)
        self.assertNotIn("judgment_similarity", self.output)

    def test_export_tables_rankings_numeric_trends_and_stale_skip(self):
        result = calculate_similarity(self.source, self.output)
        with tempfile.TemporaryDirectory() as directory:
            folder = export_similarity_report(result, directory)
            self.assertTrue((folder/"report.html").is_file())
            self.assertTrue((folder/"ch7_tag4_heatmap.png").is_file())
            config = json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(config["parameters"]["r_threshold"], .7)
            with (folder/"all_rankings.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["trial_id"] != row["other_trial_id"] for row in rows))
            with (folder/"judgment_windows.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            with np.load(folder/"ch7_tag4_data.npz") as data:
                np.testing.assert_array_equal(data["matrix"], result["judgment_similarity"]["groups"][0]["matrix"])
                self.assertIn("trial_1_trend", data.files)
            result["judgment_similarity"]["stale"] = True
            self.assertIsNone(export_similarity_report(result, Path(directory)/"stale"))
            self.assertFalse((Path(directory)/"stale").exists())

    def test_multiple_channel_tag_reports_preserve_global_exclusion_counts(self):
        source, markers, matches, settings = fixture()
        settings = {**settings, "localization_mode": "trend"}
        output = analyze_animal_peaks(source, markers, matches, settings)
        result = calculate_similarity(source, output)
        self.assertEqual(len(result["judgment_similarity"]["groups"]), 4)
        invalid = next(g for g in result["judgment_similarity"]["groups"] if g["channel"] == 22 and g["tag"] == 5)
        self.assertEqual(len(invalid["trials"]), 3)
        self.assertEqual(invalid["comparable_count"], 0)
        with tempfile.TemporaryDirectory() as directory:
            folder = export_similarity_report(result, directory)
            report = (folder/"report.html").read_text(encoding="utf-8")
            self.assertIn("ch22 / Tag 5", report)
            self.assertIn("匹配组内部中位数", report)
            with (folder/"group_summary.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)


if __name__ == "__main__":
    unittest.main()
