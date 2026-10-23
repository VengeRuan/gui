"""Dot minus 2 s to AnimalStart is a display crop, never an analysis window."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from animal_peak_analysis import (analyze_animal_peaks, trial_detail_trace,
    trial_detail_localization, export_animal_peak_results, TrialDisplayRangeError)
from animal_peak_plotting import draw_trial_detail
from test_animal_peak_analysis import fixture
from test_animal_peak_localization import trend_fixture


def dots(markers, matches, fs=1000, before=.7):
    return {sample: {**event, "tag100_source": "csv", "tag100_status": "csv",
        "tag100_start_eeg_sec": sample/fs-before+.0004,
        "tag100_start_latency_sec": -before+.0004,
        "tag100_end_latency_sec": -before+2+.0004}
        for sample, event in matches.items()}


class DotDisplayTests(unittest.TestCase):
    def prepare(self):
        source, markers, matches, settings = fixture()
        matches = dots(markers, matches)
        output = analyze_animal_peaks(source, markers[:2], matches, settings)
        output["display_settings"] = {"trial_range_mode": "dot_animal", "letter_before_ms": 4000}
        rows = [r for r in output["records"] if r.get("channel") == 7 and r["trial_id"] == 1]
        return source, output, rows

    def test_exact_csv_samples_source_offset_endpoint_and_overlay_clipping(self):
        source, output, rows = self.prepare()
        original = [g["waves"].copy() for g in output["groups"]]
        bounds = [(r["search_first"], r["search_last"], r["excerpt_first"], r["excerpt_last"])
                  for r in rows]
        t, values = trial_detail_trace(source, output, rows)
        np.testing.assert_array_equal(t[[0, -1]], [-3000, 0])
        np.testing.assert_array_equal(values, source.data[:3001, 0]-rows[0]["baseline_correction"])
        axis = Figure().subplots()
        draw_trial_detail(axis, output, rows, t, values)
        np.testing.assert_array_equal(axis.get_xlim(), [-3000, 0])
        self.assertEqual(next(l for l in axis.lines if l.get_gid() == "letter_start").get_xdata()[0], -1000)
        self.assertTrue(any(getattr(p, "_tag100_interval", None) for p in axis.patches))
        interval = next(p._tag100_interval for p in axis.patches if hasattr(p, "_tag100_interval"))
        # The actual Dot shading must not move two seconds to the left.
        self.assertAlmostEqual(interval[0], -1699.6)
        for group, saved in zip(output["groups"], original):
            np.testing.assert_array_equal(group["waves"], saved)
        self.assertEqual(bounds, [(r["search_first"], r["search_last"], r["excerpt_first"], r["excerpt_last"]) for r in rows])

    def test_display_does_not_expand_for_search_after_or_manual_anchor(self):
        source, output, rows = self.prepare()
        rows = [{**r, "search_last": 3300, "excerpt_last": 3400,
                 "extreme_relative_ms": 200, "amplitude": 1e6} for r in rows]
        t, values = trial_detail_trace(source, output, rows)
        axis = Figure().subplots()
        draw_trial_detail(axis, output, rows, t, values)
        self.assertEqual(t[-1], 0)
        np.testing.assert_array_equal(axis.get_xlim(), [-3000, 0])
        self.assertLess(axis.get_ylim()[1], 1e6)

    def test_unclipped_trial_starts_exactly_two_seconds_before_rounded_csv_dot(self):
        source, output, _ = self.prepare()
        rows = [r for r in output["records"] if r.get("channel") == 7 and r["trial_id"] == 2]
        t, values = trial_detail_trace(source, output, rows)
        np.testing.assert_array_equal(t[[0, -1]], [-3700, 0])
        np.testing.assert_array_equal(values, source.data[2300:6001, 0]-rows[0]["baseline_correction"])
        self.assertEqual(len(values), 3701)

    def test_source_start_clipping_and_invalid_csv_are_explicit(self):
        source, output, rows = self.prepare()
        clipped = [{**r, "tag100_start_eeg_sec": 1.} for r in rows]
        t, _ = trial_detail_trace(source, output, clipped)
        np.testing.assert_array_equal(t[[0, -1]], [-3000, 0])
        for patch_values in ({"tag100_start_eeg_sec": np.nan}, {"tag100_source": "txt"},
                             {"tag100_status": "missing"}, {"tag100_start_eeg_sec": 4.25},
                             {"tag100_start_eeg_sec": 5.}):
            with self.subTest(patch=patch_values), self.assertRaises(TrialDisplayRangeError):
                trial_detail_trace(source, output, [{**r, **patch_values} for r in rows])
        output["settings"]["memory_budget_bytes"] = 1
        with self.assertRaisesRegex(ValueError, "内存预算"):
            trial_detail_trace(source, output, rows)

    def test_trend_is_cropped_without_recomputing_original_locator(self):
        source, markers, matches, settings = trend_fixture()
        output = analyze_animal_peaks(source, markers[:1], dots(markers, matches), settings)
        rows = output["records"]
        original = trial_detail_localization(source, output, rows)
        output["display_settings"] = {"trial_range_mode": "dot_animal"}
        detail = trial_detail_localization(source, output, rows)
        t, _ = trial_detail_trace(source, output, rows)
        np.testing.assert_array_equal(detail["time_ms"], t)
        self.assertEqual(t[-1], 0)
        mask = original["time_ms"] <= 0
        indices = np.searchsorted(t, original["time_ms"][mask])
        np.testing.assert_array_equal(detail["trend"][indices], original["trend"][mask])

    def test_pdf_png_npz_and_manifest_use_the_same_crop(self):
        source, output, rows = self.prepare()
        limits = []
        def capture(figure, *_args, **_kwargs):
            limits.extend(a.get_xlim() for a in figure.axes
                          if any(l.get_gid() == "letter_start" for l in a.lines))
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", capture):
            folder = export_animal_peak_results(output, source, directory)
            with np.load(folder/"ch7_tag4_trial_traces_part001.npz") as saved:
                np.testing.assert_array_equal(saved["time_ms"][[0, -1]], [-3000, 0])
            manifest = json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["display_settings"]["trial_range_mode"], "dot_animal")
        self.assertTrue(limits)
        for lo, hi in limits:
            self.assertIn((lo, hi), [(-3000, 0), (-3700, 0)])


class DotDisplayQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from animal_peak_dialog import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from animal_peak_dialog import AnimalPeakDialog
        self.source, self.markers, matches, self.settings = fixture()
        self.matches = dots(self.markers, matches)
        self.dialog = AnimalPeakDialog(None, self.source, self.markers, self.matches, self.settings)

    def tearDown(self):
        self.dialog.deleteLater(); self.app.processEvents()

    def test_default_mode_switch_refresh_and_calculations_are_independent(self):
        d = self.dialog
        self.assertEqual(d.trial_range_box.currentData(), "dot_animal")
        self.assertFalse(d.letter_view_spin.isEnabled())
        output = analyze_animal_peaks(self.source, self.markers[:2], self.matches, self.settings)
        d.analysis_finished(output); d.tabs.setCurrentIndex(2)
        np.testing.assert_array_equal(d.detail_axes[0].get_xlim(), [-3000, 0])
        self.assertEqual(output["display_settings"]["trial_dot_before_ms"], 2000)
        self.assertIn("Dot 前 2 秒", d.trial_range_box.currentText())
        original = [g["waves"].copy() for g in output["groups"]]
        d.trial_range_box.setCurrentIndex(d.trial_range_box.findData("letter"))
        d.letter_view_spin.setValue(1000)
        self.assertEqual(output["display_settings"]["trial_range_mode"], "dot_animal")
        self.assertTrue(d.letter_view_spin.isEnabled())
        with patch("animal_peak_dialog.analyze_animal_peaks", side_effect=AssertionError("must not re-detect")):
            d.letter_view_button.click()
        self.assertEqual(output["display_settings"]["trial_range_mode"], "letter")
        self.assertEqual(d.detail_axes[0].get_xlim()[0], -2000)
        d.trial_range_box.setCurrentIndex(0); d.letter_view_button.click()
        for ch in (22, 7):
            d.result_channel_box.setCurrentIndex(d.result_channel_box.findData(ch))
            np.testing.assert_array_equal(d.detail_axes[0].get_xlim(), [-3000, 0])
        for group, saved in zip(output["groups"], original):
            np.testing.assert_array_equal(group["waves"], saved)
        self.assertFalse(d._parameters_dirty)

    def test_missing_dot_does_not_hide_other_trials_or_guess_an_anchor(self):
        self.matches[int(self.markers[0, 0])]["tag100_status"] = "missing"
        output = analyze_animal_peaks(self.source, self.markers[:2], self.matches, self.settings)
        d = self.dialog; d.analysis_finished(output); d.tabs.setCurrentIndex(2)
        self.assertTrue(any("CSV Dot" in t.get_text() for t in d.detail_axes[0].texts))
        self.assertFalse(d.detail_axes[0].lines)
        np.testing.assert_array_equal(d.detail_axes[1].get_xlim(), [-3700, 0])
        self.assertEqual(len(d._manual_axes), 1)

    def test_mcp_can_set_and_apply_the_same_selector(self):
        from animal_peak_control import AnimalPeakController
        controller = AnimalPeakController(self.dialog)
        rev = controller.dispatch("get_state", {})["revision"]
        result = controller.dispatch("set_parameters", {"group": "view", "values": {"trial_range_mode": "letter", "letter_before_ms": 4000}}, rev)
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.dialog._trial_range_mode, "dot_animal")
        rev = controller.dispatch("get_state", {})["revision"]
        result = controller.dispatch("execute", {"action": "apply_letter_range"}, rev)
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.dialog._trial_range_mode, "letter")
        self.assertTrue(self.dialog.letter_view_spin.isEnabled())


if __name__ == "__main__":
    unittest.main()
