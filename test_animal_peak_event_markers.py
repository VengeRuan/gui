"""Per-Trial event timing annotations never change an event-aligned average."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import animal_peak_letter_vep as vep
from animal_peak_analysis import analyze_animal_peaks
from qt_data_model import ArraySource
from test_animal_peak_analysis import fixture


def prepared():
    _, _, _, settings = fixture()
    values = np.sin(np.arange(30000)/50).astype(np.float32)[:, None]
    source = ArraySource(values, 1000, time_offset=1.25, channel_ids=[7])
    markers = np.asarray([[10250, 4], [19250, 4]])
    matches = {}
    for sample, latency, dot_before in ((10250, 1., .5), (19250, 1.5, .8)):
        matches[sample] = dict(validity="Y", animal_latency_sec=latency,
            tag100_source="csv", tag100_status="csv", tag100_start_eeg_sec=sample/1000-dot_before+.0004,
            tag100_start_latency_sec=-dot_before+.0004, tag100_end_latency_sec=2-dot_before+.0004)
    output = analyze_animal_peaks(source, markers, matches, {**settings, "channel_ids": [7], "tags": [4]})
    return source, markers, matches, output


def refresh(source, markers, matches, output, mode, trial_ids="unchanged", **window):
    cfg = dict(alignment=mode, start_ms=-6000 if mode == "animal" else 0,
               end_ms=0 if mode == "animal" else 4000 if mode == "dot" else 6000)
    cfg.update(window)
    return vep.refresh(source, output, markers, matches, (7, 4), cfg, trial_ids=trial_ids)


def coordinates(markers):
    return sorted((m["trial_id"], m["event"], m["relative_ms"]) for m in markers)


EXPECTED = {
    "animal": [(1, "dot", -1500.), (1, "letter", -1000.), (2, "dot", -2300.), (2, "letter", -1500.)],
    "letter": [(1, "animal", 1000.), (2, "animal", 1500.)],
    "dot": [(1, "letter", 500.), (1, "animal", 1500.), (2, "letter", 800.), (2, "animal", 2300.)],
}


class EventTimeMarkerTests(unittest.TestCase):
    def test_three_alignments_use_each_trial_and_global_sample_differences(self):
        source, markers, matches, output = prepared()
        for mode in EXPECTED:
            with self.subTest(mode=mode):
                revised = refresh(source, markers, matches, output, mode)
                group = revised["letter_vep"]["cache"]
                waves, mean = group["waves"].copy(), group["mean"].copy()
                with patch.object(source, "read", side_effect=AssertionError("annotations must not read signals")):
                    actual = vep.event_time_markers(group, revised["fs"])
                self.assertEqual(coordinates(actual), sorted(EXPECTED[mode]))
                self.assertFalse(any(m["event"] == mode for m in actual))
                np.testing.assert_array_equal(group["waves"], waves)
                np.testing.assert_array_equal(group["mean"], mean)

    def test_only_actually_included_trials_and_empty_selection(self):
        source, markers, matches, output = prepared()
        selected = refresh(source, markers, matches, output, "dot", trial_ids=[2])
        self.assertEqual(coordinates(vep.event_time_markers(selected["letter_vep"]["cache"], 1000)),
                         sorted(m for m in EXPECTED["dot"] if m[0] == 2))
        empty = refresh(source, markers, matches, selected, "dot", trial_ids=[])
        self.assertEqual(vep.event_time_markers(empty["letter_vep"]["cache"], 1000), [])
        # Trial 1 is selected but invalidated by NaN during the actual read.
        source.data[10000, 0] = np.nan
        excluded = refresh(source, markers, matches, output, "letter")
        group = excluded["letter_vep"]["cache"]
        self.assertEqual(group["records"][0]["status"], "excluded")
        self.assertEqual(coordinates(vep.event_time_markers(group, 1000)), [(2, "animal", 1500.)])

    def test_window_boundaries_inclusive_and_outside_events_omitted(self):
        source, markers, matches, output = prepared()
        revised = refresh(source, markers, matches, output, "letter", start_ms=-500, end_ms=1000)
        group = revised["letter_vep"]["cache"]
        self.assertEqual(coordinates(vep.event_time_markers(group, 1000)),
                         [(1, "animal", 1000.), (1, "dot", -500.)])
        axis = Figure().subplots(); vep.draw_group(axis, revised, group)
        np.testing.assert_array_equal(axis.get_xlim(), [-500, 1000])

    def test_invalid_csv_dot_has_no_txt_fallback_but_other_events_remain(self):
        for updates in ({"tag100_source": "txt"}, {"tag100_status": "missing"},
                        {"tag100_start_eeg_sec": np.nan}, {"tag100_start_eeg_sec": None}):
            with self.subTest(updates=updates):
                source, markers, matches, output = prepared()
                matches[10250].update(updates)
                revised = refresh(source, markers, matches, output, "animal")
                actual = coordinates(vep.event_time_markers(revised["letter_vep"]["cache"], 1000))
                self.assertEqual(actual, sorted(m for m in EXPECTED["animal"] if m[:2] != (1, "dot")))

    def test_saved_artists_match_helper_have_thin_lines_and_one_zero(self):
        source, markers, matches, output = prepared()
        for mode in EXPECTED:
            revised = refresh(source, markers, matches, output, mode)
            group = revised["letter_vep"]["cache"]
            axis = Figure().subplots(); vep.draw_group(axis, revised, group)
            lines = [l for l in axis.lines if hasattr(l, "_vep_event_marker")]
            self.assertEqual(coordinates(l._vep_event_marker for l in lines), sorted(EXPECTED[mode]))
            for line in lines:
                self.assertLessEqual(line.get_linewidth(), .7)
                self.assertLess(line.get_alpha(), 1)
                self.assertEqual(line.get_color(), vep.EVENT_COLORS[line._vep_event_marker["event"]])
            self.assertEqual(len([l for l in axis.lines if l.get_gid() == "event_alignment_zero"]), 1)
            np.testing.assert_array_equal(axis.get_xlim(), group["time_ms"][[0, -1]])

    def test_all_saved_modes_and_shared_trial_selection_keep_annotations(self):
        source, markers, matches, output = prepared()
        for mode in EXPECTED:
            output = refresh(source, markers, matches, output, mode, trial_ids=[2])
        seen = []
        def capture(fig, *_args, **_kwargs):
            for axis in fig.axes:
                drawn = [l._vep_event_marker for l in axis.lines if hasattr(l, "_vep_event_marker")]
                if drawn:
                    seen.extend(drawn)
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", capture), \
             patch.object(PdfPages, "savefig", lambda pdf, fig=None, **kwargs: capture(fig)):
            vep.export_results(source, output, directory)
            for mode in EXPECTED:
                path = Path(directory)/"letter_start_vep"/f"ch7_tag4_{mode}_start.npz"
                with np.load(path) as data:
                    np.testing.assert_array_equal(data["selected_trial_ids"], [2])
        self.assertTrue(seen)
        self.assertEqual({m["trial_id"] for m in seen}, {2})
        self.assertEqual({(m["event"], m["relative_ms"]) for m in seen},
                         {(e, t) for values in EXPECTED.values() for trial, e, t in values if trial == 2})


class EventTimeMarkerQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from animal_peak_dialog import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        from animal_peak_dialog import AnimalPeakDialog
        self.source, self.markers, self.matches, self.output = prepared()
        self.dialog = AnimalPeakDialog(None, self.source, self.markers, self.matches, self.output["settings"])

    def tearDown(self):
        self.dialog.deleteLater(); self.app.processEvents()

    def show_mode(self, mode, trial_ids="unchanged"):
        d = self.dialog
        self.output = refresh(self.source, self.markers, self.matches, self.output, mode, trial_ids=trial_ids)
        d.letter_vep_alignment.setCurrentIndex(d.letter_vep_alignment.findData(mode))
        d.analysis_finished(self.output); self.app.processEvents()
        return self.output["letter_vep"]["cache"]

    def test_qt_export_parity_and_markers_are_not_waveform_curves(self):
        from animal_peak_control import AnimalPeakController
        d = self.dialog
        for mode in EXPECTED:
            group = self.show_mode(mode)
            markers = [item._vep_event_marker for item in d.letter_vep_plot.items() if hasattr(item, "_vep_event_marker")]
            self.assertEqual(coordinates(markers), sorted(EXPECTED[mode]))
            self.assertEqual(len(d.letter_vep_plot.listDataItems()), len(group["selected_indices"])+1)
            controller = AnimalPeakController(d)
            response = controller.dispatch("read_waveform", {"plot": "event", "curve": len(group["selected_indices"]), "limit": 10})
            self.assertTrue(response["ok"], response)
            np.testing.assert_array_equal(response["result"]["y"], group["mean"][:10])
            np.testing.assert_allclose(d.letter_vep_plot.viewRange()[0], group["time_ms"][[0, -1]])
            title = d.letter_vep_plot.plotItem.titleLabel
            self.assertGreaterEqual(title.geometry().height(), title.item.boundingRect().height())

    def test_copy_includes_annotations_without_recomputing_and_refresh_clears_them(self):
        from animal_peak_dialog import QtWidgets
        group = self.show_mode("dot", trial_ids=[2]); d = self.dialog
        before = group["mean"].copy(); clipboard = Mock()
        with patch.object(QtWidgets.QApplication, "clipboard", return_value=clipboard), \
             patch.object(self.source, "read", side_effect=AssertionError("copy must not reread")):
            d.copy_event_plot()
        clipboard.setPixmap.assert_called_once()
        self.assertFalse(clipboard.setPixmap.call_args.args[0].isNull())
        np.testing.assert_array_equal(group["mean"], before)
        self.show_mode("dot", trial_ids=[])
        self.assertFalse(any(hasattr(i, "_vep_event_marker") for i in d.letter_vep_plot.items()))
        self.assertFalse(d.letter_vep_plot.listDataItems())


if __name__ == "__main__":
    unittest.main()
