"""The localization copy can change, but the averaged loaded data must not."""
import csv
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
from matplotlib.figure import Figure

from animal_peak_analysis import (
    AnalysisCancelled, analyze_animal_peaks, trial_detail_localization,
    trial_detail_trace, export_animal_peak_results,
)
from animal_peak_localization import localization_config, trial_localization
from animal_peak_plotting import draw_trial_detail
from qt_data_model import ArraySource


def trend_fixture():
    fs = 1000
    values = np.zeros((7500, 1), dtype=np.float32)
    x = np.arange(values.shape[0])
    for animal in (2000, 5000):
        values[:, 0] += (2 * np.exp(-.5 * ((x - (animal - 120)) / 25) ** 2)
                         - 1.5 * np.exp(-.5 * ((x - (animal + 50)) / 30) ** 2)).astype(np.float32)
        values[animal - 30, 0] = 20
        values[animal + 100, 0] = -25
    source = ArraySource(values, fs, channel_ids=[7])
    markers = np.asarray([[1000, 4], [4000, 4]])
    matches = {int(sample): {"validity": "Y", "animal_latency_sec": 1} for sample, _ in markers}
    settings = dict(channel_ids=[7], tags=[4], trial_limit=0, processing={},
                    epoch_start_ms=-500, epoch_end_ms=2000, baseline_start_ms=-200, baseline_end_ms=0,
                    search_ms=300, search_after_ms=200, pre_ms=50, post_ms=150,
                    localization_mode="trend", despike_window_ms=9, despike_mad_threshold=6, trend_window_ms=31)
    return source, markers, matches, settings


class LocalizationTests(unittest.TestCase):
    def test_extra_letter_display_context_preserves_original_locator_and_raw_data(self):
        source, markers, matches, settings = trend_fixture()
        output = analyze_animal_peaks(source, markers[:1], matches, settings)
        rows = output["records"]
        original = trial_detail_localization(source, output, rows)
        original_data = source.data.copy()
        output["display_settings"] = {"letter_before_ms": 800}
        time_ms, raw = trial_detail_trace(source, output, rows)
        extended = trial_detail_localization(source, output, rows)
        self.assertEqual(time_ms[0], -1800)
        np.testing.assert_array_equal(time_ms, extended["time_ms"])
        common = np.searchsorted(time_ms, original["time_ms"])
        for key in ("trend", "despiked", "despike_mask"):
            np.testing.assert_array_equal(extended[key][common], original[key])
        for row in rows:
            index = np.argmin(abs(time_ms - row["extreme_relative_ms"]))
            self.assertAlmostEqual(float(extended["trend"][index]), row["localization_amplitude"], places=6)
        np.testing.assert_array_equal(source.data, original_data)
        self.assertEqual(len(raw), len(extended["trend"]))

    def test_spikes_do_not_win_but_original_samples_are_averaged(self):
        source, markers, matches, settings = trend_fixture()
        original = source.data.copy()
        raw = analyze_animal_peaks(source, markers, matches, {**settings, "localization_mode": "raw"})
        self.assertEqual({g["branch"]: g["rows"][0]["extreme_relative_ms"] for g in raw["groups"]},
                         {"peak": -30, "trough": 100})
        output = analyze_animal_peaks(source, markers, matches, settings)
        for group in output["groups"]:
            expected = -120 if group["branch"] == "peak" else 50
            for row, wave in zip(group["rows"], group["waves"]):
                self.assertLessEqual(abs(row["extreme_relative_ms"] - expected), 3)
                manual = source.data[row["excerpt_first"]:row["excerpt_last"], 0] - row["baseline_correction"]
                np.testing.assert_array_equal(wave, manual)
                self.assertEqual(row["amplitude"], float(source.data[row["extreme_local_sample"], 0] - row["baseline_correction"]))
                self.assertGreaterEqual(row["despike_count_search"], 2)
                self.assertEqual(row["raw_extreme_relative_ms"], -30 if group["branch"] == "peak" else 100)
            np.testing.assert_array_equal(group["mean"], np.mean(group["waves"], axis=0))
        peak = next(g for g in output["groups"] if g["branch"] == "peak")
        self.assertGreater(peak["mean"].max(), 15)  # The original spike is NOT smoothed out of the average.
        np.testing.assert_array_equal(source.data, original)

    def test_trend_mode_never_applies_legacy_conditioning_to_average(self):
        source, markers, matches, settings = trend_fixture()
        settings["processing"] = {"smooth": True, "smooth_window_sec": .1, "zscore": True}
        output = analyze_animal_peaks(source, markers, matches, settings)
        self.assertEqual(output["settings"]["processing"], {})
        self.assertEqual(output["units"], "mV")
        self.assertEqual(output["settings"]["average_waveform"], "loaded_source_baseline_corrected")
        reference = output["animal_reference_groups"][0]
        for row, wave in zip(reference["rows"], reference["waves"]):
            animal = row["animal_local_sample"]
            np.testing.assert_array_equal(wave, source.data[animal - 50:animal + 151, 0] - row["baseline_correction"])
        self.assertEqual(len(output["animal_reference_records"]), 2)

    def test_reproduced_curve_uses_same_samples_for_alignment_and_export(self):
        source, markers, matches, settings = trend_fixture()
        output = analyze_animal_peaks(source, markers, matches, settings)
        rows = [row for row in output["records"] if row["trial_id"] == 1]
        curve = trial_detail_localization(source, output, rows)
        t, raw = trial_detail_trace(source, output, rows)
        np.testing.assert_array_equal(t, curve["time_ms"])
        self.assertEqual(raw.size, curve["trend"].size)
        for row in rows:
            index = np.argmin(abs(t - row["extreme_relative_ms"]))
            self.assertAlmostEqual(float(curve["trend"][index]), row["localization_amplitude"], places=6)
        figure = Figure()
        raw_axis, trend_axis = figure.subplots(1, 2)
        draw_trial_detail(raw_axis, output, rows, t, raw, curve, "raw")
        draw_trial_detail(trend_axis, output, rows, t, raw, curve, "trend")
        self.assertGreater(np.ptp(raw_axis.get_ylim()), np.ptp(trend_axis.get_ylim()) * 5)
        np.testing.assert_array_equal(raw_axis.lines[0].get_ydata(), raw)
        np.testing.assert_array_equal(trend_axis.lines[0].get_ydata(), curve["trend"])

    def test_nonfinite_samples_are_not_candidates_and_raw_gaps_are_preserved(self):
        source, markers, matches, settings = trend_fixture()
        source.data[1877:1883, 0] = np.nan
        output = analyze_animal_peaks(source, markers[:1], matches, settings)
        self.assertTrue(output["groups"])
        for group in output["groups"]:
            row = group["rows"][0]
            self.assertTrue(np.isfinite(row["amplitude"]))
            self.assertFalse(1877 <= row["extreme_local_sample"] < 1883)
        rows = output["records"]
        curve = trial_detail_localization(source, output, rows)
        t, raw = trial_detail_trace(source, output, rows)
        self.assertTrue(np.isnan(raw).any())
        self.assertTrue(np.all(np.isnan(curve["trend"])[np.isnan(raw)]))

    def test_parameters_padding_memory_and_cancel(self):
        source, _, _, settings = trend_fixture()
        for extra in ({"trend_window_ms": 0}, {"despike_mad_threshold": 0},
                      {"despike_window_ms": -1}, {"trend_window_ms": np.nan}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                localization_config({**settings, **extra}, source.metadata.fs)
        trial = {"search_first": 0, "search_last": 200}
        sizes = []
        def read(first, last):
            sizes.append(last - first)
            return source.data[first:last, 0]
        curve = trial_localization(read, trial, source.metadata.rows, 1000, settings)
        self.assertEqual(curve["first"], 0)
        self.assertLess(max(sizes), source.metadata.rows)
        self.assertTrue(np.isfinite(curve["trend"]).all())
        with self.assertRaisesRegex(ValueError, "内存预算"):
            trial_localization(read, trial, source.metadata.rows, 1000, {**settings, "memory_budget_bytes": 1})
        with self.assertRaises(AnalysisCancelled):
            trial_localization(read, trial, source.metadata.rows, 1000, settings,
                               check=lambda: (_ for _ in ()).throw(AnalysisCancelled()))

    def test_time_offset_changes_global_samples_not_selected_raw_segments(self):
        source, markers, matches, settings = trend_fixture()
        output = analyze_animal_peaks(source, markers, matches, settings)
        shifted = ArraySource(source.data.copy(), 1000, time_offset=1.25, channel_ids=[7])
        shifted_markers = markers.copy(); shifted_markers[:, 0] += 1250
        shifted_matches = {sample + 1250: value for sample, value in matches.items()}
        result = analyze_animal_peaks(shifted, shifted_markers, shifted_matches, settings)
        for actual, expected in zip(result["groups"], output["groups"]):
            np.testing.assert_array_equal(actual["waves"], expected["waves"])
            self.assertEqual(actual["rows"][0]["extreme_sample"], expected["rows"][0]["extreme_sample"] + 1250)
            self.assertEqual(actual["rows"][0]["extreme_relative_ms"], expected["rows"][0]["extreme_relative_ms"])

    def test_cancel_inside_chunked_despiking_leaves_input_untouched(self):
        source, _, _, settings = trend_fixture()
        settings["despike_window_ms"] = 500
        calls = 0
        original = source.data.copy()
        def checkpoint():
            nonlocal calls
            calls += 1
            if calls == 5:
                raise AnalysisCancelled("cancel within median/MAD blocks")
        with self.assertRaises(AnalysisCancelled):
            trial_localization(lambda first, last: source.data[first:last, 0],
                               {"search_first": 1700, "search_last": 2200}, 7500, 1000, settings, checkpoint)
        np.testing.assert_array_equal(source.data, original)

    def test_full_export_preserves_raw_and_localization_arrays_and_reference(self):
        source, markers, matches, settings = trend_fixture()
        output = analyze_animal_peaks(source, markers[:1], matches, settings)
        with tempfile.TemporaryDirectory() as directory:
            folder = export_animal_peak_results(output, source, directory, individual_png=False)
            manifest = json.loads((folder / "parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["settings"]["localization_mode"], "trend")
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue((folder / "ch7_tag4_animalstart_reference.png").exists())
            self.assertTrue((folder / "animalstart_reference_details.csv").exists())
            rows = [row for row in output["records"] if row["trial_id"] == 1]
            curve = trial_detail_localization(source, output, rows)
            _, raw = trial_detail_trace(source, output, rows)
            with np.load(folder / "ch7_tag4_trial_traces_part001.npz") as saved:
                np.testing.assert_array_equal(saved["values"], raw)
                np.testing.assert_array_equal(saved["trend_values"], curve["trend"])
                np.testing.assert_array_equal(saved["despiked_values"], curve["despiked"])
                np.testing.assert_array_equal(saved["despike_mask"], curve["despike_mask"])
            with (folder / "trial_details.csv").open(encoding="utf-8-sig") as handle:
                record = next(csv.DictReader(handle))
            self.assertEqual(record["localization_mode"], "trend")
            self.assertGreater(int(record["despike_count_search"]), 0)


if __name__ == "__main__":
    unittest.main()
