"""Headless numerical and export checks for the isolated AnimalStart experiment."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from animal_peak_analysis import (
    AnalysisCancelled, _summarize, analyze_animal_peaks,
    export_animal_peak_results, trial_detail_trace,
)
from qt_data_model import ArraySource


def fixture():
    data = np.zeros((20000, 2), dtype=np.float32)
    for animal in (3000, 6000):
        data[animal-280, 0] = 5
        data[animal-10, 0] = 2
        data[animal-150, 0] = -7
        data[animal-5, 0] = -1
        data[animal-10, 1] = 8
        data[animal-100, 1] = -9
    source = ArraySource(data, 1000, time_offset=1.25, channel_ids=[7, 22])
    markers = np.asarray([[3250, 4], [6250, 4], [9250, 5], [12250, 5], [15250, 5]])
    matches = {3250: {"validity": "Y", "animal_latency_sec": 1},
               6250: {"validity": "Y", "animal_latency_sec": 1},
               12250: {"validity": "N", "animal_latency_sec": 1},
               15250: {"validity": "Y", "animal_latency_sec": 3}}
    settings = {"search_ms": 300, "pre_ms": 50, "post_ms": 50,
                "epoch_start_ms": -500, "epoch_end_ms": 2000,
                "baseline_start_ms": -200, "baseline_end_ms": 0,
                "channel_ids": [7, 22], "tags": [4, 5], "trial_limit": 0, "processing": {}}
    return source, markers, matches, settings


class AnimalPeakAnalysisTests(unittest.TestCase):
    def test_largest_extrema_per_channel_not_nearest_and_no_mutation(self):
        source, markers, matches, settings = fixture()
        original = source.data.copy()
        output = analyze_animal_peaks(source, markers, matches, settings)
        self.assertEqual(len(output["groups"]), 4)
        peak = next(g for g in output["groups"] if g["channel"] == 7 and g["branch"] == "peak")
        trough = next(g for g in output["groups"] if g["channel"] == 7 and g["branch"] == "trough")
        self.assertEqual(peak["rows"][0]["extreme_relative_ms"], -280)
        self.assertEqual(trough["rows"][0]["extreme_relative_ms"], -150)
        self.assertEqual(peak["rows"][0]["amplitude"], 5)
        self.assertEqual(trough["rows"][0]["amplitude"], -7)
        other = next(g for g in output["groups"] if g["channel"] == 22 and g["branch"] == "peak")
        self.assertEqual(other["rows"][0]["extreme_relative_ms"], -10)
        self.assertTrue(other["rows"][0]["crosses_animal_start"])
        self.assertEqual(other["rows"][0]["animal_sec"], 4.25)
        self.assertEqual(other["rows"][0]["extreme_sample"], 4240)
        np.testing.assert_array_equal(peak["mean"], peak["waves"].mean(axis=0))
        np.testing.assert_array_equal(original, source.data)
        t, trace = trial_detail_trace(source, output, [other["rows"][0]])
        self.assertTrue(np.max(t) > 0)
        self.assertEqual(trace.size, t.size)

    def test_missing_invalid_or_outside_epoch_animal_is_not_in_plots_or_average(self):
        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers, matches, settings)
        self.assertEqual([row["trial_id"] for row in output["eligible_trials"]], [1, 2])
        exclusions = {row["trial_id"]: row["reason"] for row in output["records"] if row["status"] == "excluded"}
        self.assertIn("行为记录", exclusions[3])
        self.assertIn("validity", exclusions[4])
        self.assertIn("Epoch 内", exclusions[5])
        self.assertTrue(all(row["trial_id"] in (1, 2) for g in output["groups"] for row in g["rows"]))

    def test_flat_boundary_and_equal_extrema_still_participate(self):
        source, markers, matches, settings = fixture()
        source.data[:] = 0
        output = analyze_animal_peaks(source, markers, matches, settings)
        self.assertEqual(sum(len(g["rows"]) for g in output["groups"]), 8)
        for group in output["groups"]:
            for row in group["rows"]:
                self.assertIn("平直", row["flags"])
                self.assertIn("边界", row["flags"])
                self.assertEqual(row["extreme_relative_ms"], -1)
                self.assertTrue(np.isnan(row["similarity_r"]))

    def test_similarity_is_descriptive_not_filtering(self):
        waves = np.asarray([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=np.float32)
        mean, sem, count, correlations = _summarize(waves)
        self.assertTrue(all(r < .6 for r in correlations))
        np.testing.assert_allclose(mean, waves.mean(axis=0))
        self.assertTrue(np.all(count == 2))

    def test_search_must_be_covered_but_excerpt_can_extend_outside_old_epoch(self):
        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers, matches, {**settings, "search_ms": 2000})
        self.assertFalse(output["groups"])
        self.assertTrue(any("搜索范围" in row["reason"] for row in output["records"]))
        output = analyze_animal_peaks(source, markers[:1], matches, {**settings, "post_ms": 1500})
        group = next(g for g in output["groups"] if g["channel"] == 22 and g["branch"] == "peak")
        self.assertGreater(group["rows"][0]["excerpt_last"], group["rows"][0]["epoch_last"])
        self.assertEqual(group["waves"].shape[1], 1551)

    def test_source_boundary_exclusion_is_branch_specific_and_preserves_found_peak(self):
        source, _, _, settings = fixture()
        markers = np.asarray([[18250, 4]])  # local marker 17000; epoch ends 19000.
        matches = {18250: {"validity": "Y", "animal_latency_sec": 1}}
        output = analyze_animal_peaks(source, markers, matches, {**settings, "post_ms": 3000})
        self.assertFalse(output["groups"])
        self.assertEqual(len(output["eligible_trials"]), 1)
        for row in output["records"]:
            self.assertIn("extreme_sample", row)
            self.assertIn("源数据边界", row["reason"])

    def test_ordinary_path_only_reads_short_slices_and_has_memory_guard(self):
        source, markers, matches, settings = fixture()
        read = source.read
        sizes = []
        def short_read(first, last, columns):
            sizes.append(last-first)
            return read(first, last, columns)
        source.read = short_read
        analyze_animal_peaks(source, markers, matches, settings)
        self.assertLessEqual(max(sizes), 300)
        with self.assertRaisesRegex(ValueError, "内存|预算"):
            analyze_animal_peaks(source, markers, matches, {**settings, "memory_budget_bytes": 1})
        with self.assertRaises(AnalysisCancelled):
            analyze_animal_peaks(source, markers, matches, settings, cancel=lambda: True)

    def test_source_identity_and_trial_limit(self):
        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers, matches, {**settings, "trial_limit": 1})
        self.assertEqual([r["trial_id"] for r in output["eligible_trials"]], [1])
        different = ArraySource(source.data.copy(), 1000, time_offset=1.25, channel_ids=[7, 22])
        with self.assertRaisesRegex(ValueError, "改变"):
            trial_detail_trace(different, output, [output["groups"][0]["rows"][0]])

    def test_export_includes_tag_trial_labels_figures_data_and_manifest(self):
        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers[:1], matches, {**settings, "channel_ids": [7]})
        with tempfile.TemporaryDirectory() as directory:
            folder = export_animal_peak_results(output, source, directory, individual_png=True)
            manifest = json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue((folder/"ch7_tag4_trial00001_detail.png").is_file())
            self.assertTrue((folder/"ch7_tag4_summary.png").is_file())
            self.assertTrue((folder/"ch7_tag4_summary.pdf").is_file())
            with np.load(folder/"ch7_tag4_trial_traces_part001.npz") as trace:
                self.assertEqual(trace["trial_ids"].tolist(), [1])
                self.assertEqual(trace["tag"], 4)
                self.assertEqual(trace["offsets"].tolist(), [0, trace["values"].size])
            self.assertGreater((folder/"all_trial_details.pdf").stat().st_size, 1000)
            with (folder/"trial_details.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["branch"] for row in rows}, {"peak", "trough"})
            self.assertTrue(all(row["trial_id"] == "1" and row["tag"] == "4" for row in rows))
            with np.load(folder/"ch7_tag4_peak.npz") as saved:
                self.assertEqual(saved["tag"], 4)
                self.assertEqual(saved["trial_ids"].tolist(), [1])
                np.testing.assert_allclose(saved["mean"], output["groups"][0]["mean"])


if __name__ == "__main__":
    unittest.main()
