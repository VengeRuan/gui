"""Headless numerical and export checks for the isolated AnimalStart experiment."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
    def test_letter_display_duration_reads_before_epoch_without_changing_analysis(self):
        from matplotlib.figure import Figure
        from animal_peak_plotting import draw_trial_detail

        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers[:2], matches, settings)
        waves = [group["waves"].copy() for group in output["groups"]]
        bounds = [(row["search_first"], row["search_last"], row["extreme_local_sample"])
                  for row in output["records"]]
        output["display_settings"] = {"letter_before_ms": 1000}
        for trial_id in (1, 2):
            rows = [row for row in output["records"] if row.get("channel") == 7 and row["trial_id"] == trial_id]
            time_ms, values = trial_detail_trace(source, output, rows)
            first = rows[0]["marker_sample"] - 1250 - 1000
            self.assertLess(first, rows[0]["epoch_first"])
            self.assertEqual(time_ms[0], -2000)
            np.testing.assert_array_equal(values[:10], source.data[first:first+10, 0] - rows[0]["baseline_correction"])
            axis = Figure().subplots()
            draw_trial_detail(axis, output, rows, time_ms, values)
            self.assertEqual(axis.get_xlim()[0], -2000)
        for group, original in zip(output["groups"], waves):
            np.testing.assert_array_equal(group["waves"], original)
        self.assertEqual(bounds, [(row["search_first"], row["search_last"], row["extreme_local_sample"])
                                  for row in output["records"]])

    def test_letter_display_zero_source_boundary_and_invalid_duration(self):
        from matplotlib.figure import Figure
        from animal_peak_plotting import draw_trial_detail

        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers[:1], matches, settings)
        rows = [row for row in output["records"] if row.get("channel") == 7]
        for duration, expected_first in ((0, -1000), (3000, -3000)):
            output["display_settings"] = {"letter_before_ms": duration}
            time_ms, values = trial_detail_trace(source, output, rows)
            self.assertEqual(time_ms[0], expected_first)
            axis = Figure().subplots()
            draw_trial_detail(axis, output, rows, time_ms, values)
            self.assertEqual(axis.get_xlim()[0], expected_first)
            self.assertTrue(next(line for line in axis.lines if line.get_gid() == "letter_start").get_visible())
        for invalid in (-1, np.inf, np.nan):
            output["display_settings"] = {"letter_before_ms": invalid}
            with self.assertRaises(ValueError):
                trial_detail_trace(source, output, rows)

    def test_export_keeps_requested_letter_display_context_in_figures_and_trace_data(self):
        from matplotlib.figure import Figure

        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers[:1], matches, {**settings, "channel_ids": [7]})
        output["display_settings"] = {"letter_before_ms": 1000}
        left_limits = []
        def capture(figure, *_args, **_kwargs):
            for axis in figure.axes:
                if any(line.get_gid() == "letter_start" for line in axis.lines):
                    left_limits.append(axis.get_xlim()[0])
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", capture):
            folder = export_animal_peak_results(output, source, directory)
            manifest = json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["display_settings"]["letter_before_ms"], 1000)
            with np.load(folder/"ch7_tag4_trial_traces_part001.npz") as saved:
                self.assertEqual(saved["time_ms"][0], -2000)
            with np.load(folder/"ch7_tag4_peak.npz") as saved:
                np.testing.assert_array_equal(saved["trials"], output["groups"][0]["waves"])
        self.assertTrue(left_limits)
        self.assertTrue(all(left == -2000 for left in left_limits))

    def test_letter_onset_red_line_uses_trial_latency_not_epoch_crop_or_source_offset(self):
        from matplotlib.figure import Figure
        from animal_peak_plotting import draw_trial_detail

        source, markers, matches, settings = fixture()
        matches[6250] = {"validity": "Y", "animal_latency_sec": 1.234567}
        output = analyze_animal_peaks(source, markers[:2], matches, {**settings, "search_ms": 1400})
        for trial_id, expected_ms in ((1, -1000), (2, -1234.567)):
            rows = [row for row in output["records"] if row.get("channel") == 7 and row["trial_id"] == trial_id]
            time_ms, values = trial_detail_trace(source, output, rows)
            axis = Figure().subplots()
            draw_trial_detail(axis, output, rows, time_ms, values)
            line = next(line for line in axis.lines if line.get_gid() == "letter_start")
            self.assertTrue(line.get_visible())
            np.testing.assert_allclose(line.get_xdata(), [expected_ms, expected_ms])
            self.assertEqual(line.get_color(), "#c62828")
            self.assertEqual(line.get_linestyle(), "-")
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), values)
            self.assertIsNone(axis.get_legend())

    def test_offscreen_letter_onset_does_not_expand_view_and_tracks_zoom(self):
        from matplotlib.figure import Figure
        from animal_peak_plotting import draw_trial_detail

        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers[:1], matches, settings)
        rows = [row for row in output["records"] if row.get("channel") == 7]
        time_ms, values = trial_detail_trace(source, output, rows)
        axis = Figure().subplots()
        draw_trial_detail(axis, output, rows, time_ms, values)
        line = next(line for line in axis.lines if line.get_gid() == "letter_start")
        self.assertFalse(line.get_visible())
        self.assertGreater(axis.get_xlim()[0], -1000)
        axis.set_xlim(-1200, 100)
        self.assertTrue(line.get_visible())
        axis.set_xlim(-1000, 0)  # Boundary included.
        self.assertTrue(line.get_visible())
        axis.set_xlim(0, -1200)  # Reversed axis also works.
        self.assertTrue(line.get_visible())
        axis.set_xlim(-300, 100)
        self.assertFalse(line.get_visible())

    def test_fixed_display_range_exports_all_waveform_axes_without_clipping_saved_samples(self):
        from matplotlib.figure import Figure

        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers[:1], matches, {**settings, "channel_ids": [7]})
        output["display_settings"] = {"y_axis_mode": "fixed", "y_min": -1, "y_max": 2, "y_unit": "μV"}
        limits_seen = []
        def capture(figure, *_args, **_kwargs):
            for axis in figure.axes:
                if axis.get_ylabel() == "μV":
                    limits_seen.append(axis.get_ylim())
                    for line in axis.lines:
                        if line.get_label() == "所选Trial等权平均":
                            branch = "peak" if "最大峰" in axis.get_title() else "trough"
                            group = next(group for group in output["groups"] if group["branch"] == branch)
                            np.testing.assert_array_equal(line.get_ydata(), group["mean"])
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", capture):
            folder = export_animal_peak_results(output, source, directory, individual_png=True)
            manifest = json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["display_settings"], output["display_settings"])
            self.assertEqual(manifest["waveform_display"]["amplitude_multiplier"], 1)
            self.assertEqual(manifest["waveform_display"]["label_unit"], "μV")
            self.assertEqual(manifest["units"], output["units"])
            with np.load(folder/"ch7_tag4_peak.npz") as saved:
                np.testing.assert_array_equal(saved["trials"], output["groups"][0]["waves"])
                self.assertGreater(np.max(saved["trials"]), 2)
        self.assertTrue(limits_seen)
        for limits in limits_seen:
            np.testing.assert_allclose(limits, [-1, 2])

    def test_trial_title_shows_animal_time_relative_to_its_letter_not_epoch_boundary(self):
        from matplotlib.figure import Figure
        from animal_peak_plotting import draw_trial_detail

        source, markers, matches, settings = fixture()
        # Different per-trial latencies, including a sub-sample fraction.
        matches[6250] = {"validity": "Y", "animal_latency_sec": 1.234567}
        output = analyze_animal_peaks(source, markers[:2], matches, settings)
        for trial_id, absolute, relative in ((1, "4.250000", "+1000.000"),
                                             (2, "7.484567", "+1234.567")):
            with self.subTest(trial_id=trial_id):
                rows = [row for row in output["records"]
                        if row.get("channel") == 7 and row["trial_id"] == trial_id]
                time_ms, values = trial_detail_trace(source, output, rows)
                axis = Figure().subplots()
                draw_trial_detail(axis, output, rows, time_ms, values)
                self.assertIn(f"AnimalStart={absolute} s", axis.get_title())
                self.assertIn(f"相对 Letter 开始={relative} ms", axis.get_title())
                # Epoch starts at -500 ms, but the requested origin is Letter=0.
                self.assertNotIn("+1500.000 ms", axis.get_title())

    def test_after_search_finds_post_event_extrema_only_inside_requested_range(self):
        source, markers, matches, settings = fixture()
        original = source.data.copy()
        source.data[3080, 0] = 15
        source.data[3120, 0] = -17
        source.data[3200, 0] = 99  # Right endpoint is excluded.
        before = analyze_animal_peaks(source, markers[:1], matches, settings)
        self.assertTrue(all(row["extreme_relative_ms"] < 0 for g in before["groups"] for row in g["rows"]))
        self.assertEqual(before["settings"]["search_after_ms"], 0)
        data_before_run = source.data.copy()
        output = analyze_animal_peaks(source, markers[:1], matches, {**settings, "search_after_ms": 200})
        peak = next(g for g in output["groups"] if g["channel"] == 7 and g["branch"] == "peak")
        trough = next(g for g in output["groups"] if g["channel"] == 7 and g["branch"] == "trough")
        self.assertEqual(peak["rows"][0]["extreme_relative_ms"], 80)
        self.assertEqual(trough["rows"][0]["extreme_relative_ms"], 120)
        self.assertEqual(peak["rows"][0]["search_last"], 3200)
        np.testing.assert_array_equal(source.data, data_before_run)
        self.assertEqual(peak["waves"].shape[1], 101)  # Excerpt size unchanged.
        time_ms, _ = trial_detail_trace(source, output, [peak["rows"][0]])
        self.assertEqual(time_ms.max(), 199)
        self.assertFalse(np.array_equal(original, source.data))  # Fixture edits only.

    def test_after_search_requires_full_epoch_coverage_and_valid_parameter(self):
        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers[:1], matches, {**settings, "search_after_ms": 1000})
        self.assertTrue(output["groups"])  # Search ends exactly at Epoch's exclusive end.
        output = analyze_animal_peaks(source, markers[:1], matches, {**settings, "search_after_ms": 1001})
        self.assertFalse(output["groups"])
        self.assertIn("完整搜索范围", output["records"][0]["reason"])
        for invalid in (-1, np.nan, np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                analyze_animal_peaks(source, markers, matches, {**settings, "search_after_ms": invalid})

    def test_equal_extrema_in_two_sided_search_choose_nearest_then_latest(self):
        source, markers, matches, settings = fixture()
        source.data[:] = 0
        source.data[[2900, 3010, 3150], 0] = 5
        source.data[[2980, 3020, 3170], 0] = -7
        output = analyze_animal_peaks(source, markers[:1], matches,
                                      {**settings, "channel_ids": [7], "search_after_ms": 200})
        relative = {group["branch"]: group["rows"][0]["extreme_relative_ms"] for group in output["groups"]}
        self.assertEqual(relative, {"peak": 10, "trough": 20})

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

    def test_search_start_is_clipped_and_excerpt_can_extend_outside_old_epoch(self):
        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers, matches, {**settings, "search_ms": 2000})
        self.assertTrue(output["groups"])
        self.assertEqual([row["trial_id"] for row in output["eligible_trials"]], [1, 2])
        for group in output["groups"]:
            for row in group["rows"]:
                self.assertTrue(row["search_start_clipped"])
                self.assertEqual(row["search_first"], row["epoch_first"])
                self.assertLess(row["requested_search_first"], row["search_first"])
                self.assertEqual(row["reason"], "")
        output = analyze_animal_peaks(source, markers[:1], matches, {**settings, "post_ms": 1500})
        group = next(g for g in output["groups"] if g["channel"] == 22 and g["branch"] == "peak")
        self.assertGreater(group["rows"][0]["excerpt_last"], group["rows"][0]["epoch_last"])
        self.assertEqual(group["waves"].shape[1], 1551)

    def test_clipped_search_keeps_pre_letter_data_and_plot_highlight_uses_actual_range(self):
        from matplotlib.figure import Figure
        from animal_peak_plotting import draw_trial_detail

        for mode in ("raw", "trend"):
            with self.subTest(mode=mode):
                source, markers, matches, settings = fixture()
                source.data[1400:1450, 0] = 10000  # Outside Epoch: must not become an anchor.
                source.data[1700:1750, 0] = 20  # Before Letter=2000, inside Epoch starting at 1500.
                output = analyze_animal_peaks(source, markers[:1], matches,
                    {**settings, "channel_ids": [7], "search_ms": 5000, "localization_mode": mode,
                     "despike_window_ms": 0, "trend_window_ms": 5})
                self.assertEqual(len(output["eligible_trials"]), 1)
                rows = [row for row in output["records"] if row.get("channel") == 7]
                peak = next(row for row in rows if row["branch"] == "peak")
                self.assertEqual(peak["search_first"], 1500)
                self.assertTrue(1500 <= peak["extreme_local_sample"] < 2000)
                self.assertEqual(peak["reason"], "")
                self.assertIn("搜索起点已截至 Epoch 起点", peak["flags"])
                time_ms, values = trial_detail_trace(source, output, rows)
                axis = Figure().subplots()
                draw_trial_detail(axis, output, rows, time_ms, values)
                self.assertAlmostEqual(axis.patches[0].get_x(), -1500)
                self.assertAlmostEqual(axis.patches[0].get_width(), 1500)
                self.assertGreater(axis.get_xlim()[0], -2000)  # Not nominal -5000 ms.
                line = next(line for line in axis.lines if line.get_gid() == "letter_start")
                self.assertTrue(line.get_visible())

    def test_search_clipping_does_not_hide_other_errors_or_fail_on_empty_search(self):
        source, markers, matches, settings = fixture()
        output = analyze_animal_peaks(source, markers[:1], {3250: {"validity": "Y", "animal_latency_sec": -.4}}, settings)
        self.assertTrue(output["groups"])
        self.assertEqual(output["eligible_trials"][0]["search_first"], 1500)
        output = analyze_animal_peaks(source, markers[:1], {3250: {"validity": "Y", "animal_latency_sec": -.5}}, settings)
        self.assertTrue(output["eligible_trials"])
        self.assertFalse(output["groups"])
        self.assertTrue(all(row["reason_code"] == "search_no_finite" for row in output["records"]))

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
        source.data[3050, 0] = 15
        output = analyze_animal_peaks(source, markers[:1], matches,
                                      {**settings, "channel_ids": [7], "search_after_ms": 100})
        with tempfile.TemporaryDirectory() as directory:
            folder = export_animal_peak_results(output, source, directory, individual_png=True)
            manifest = json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["settings"]["search_after_ms"], 100)
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
            self.assertEqual(float(next(row for row in rows if row["branch"] == "peak")["extreme_relative_ms"]), 50)
            self.assertTrue(all(row["trial_id"] == "1" and row["tag"] == "4" for row in rows))
            with np.load(folder/"ch7_tag4_peak.npz") as saved:
                self.assertEqual(saved["tag"], 4)
                self.assertEqual(saved["trial_ids"].tolist(), [1])
                np.testing.assert_allclose(saved["mean"], output["groups"][0]["mean"])


if __name__ == "__main__":
    unittest.main()
