"""Unavailable AnimalStart blocks only dependent plots, not Letter/Dot VEP."""
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import animal_peak_letter_vep as vep
from animal_peak_analysis import analyze_animal_peaks, export_animal_peak_results
from test_animal_peak_analysis import fixture
from test_animal_peak_event_vep import with_dots


def prepared():
    source, markers, matches, settings = fixture()
    markers = markers[:2]
    cues = {int(sample): {k: v for k, v in with_dots(matches)[int(sample)].items() if k.startswith("tag100_")}
            for sample, _ in markers}
    settings = {**settings, "channel_ids": [7], "tags": [4]}
    output = analyze_animal_peaks(source, markers, cues, settings)
    return source, markers, cues, output


def refresh(source, markers, matches, output, mode, **kwargs):
    cfg = dict(alignment=mode, start_ms=-1500 if mode == "animal" else 0,
               end_ms=0 if mode == "animal" else 2000)
    return vep.refresh(source, output, markers, matches, (7, 4), cfg, **kwargs)


class MissingAnimalTests(unittest.TestCase):
    def test_csv_only_letter_and_dot_still_average_without_inventing_animal(self):
        s, m, cues, o = prepared()
        self.assertFalse(o["groups"])
        for mode in ("letter", "dot"):
            revised = refresh(s, m, cues, o, mode)
            group = revised["letter_vep"]["cache"]
            self.assertEqual(len(group["selected_indices"]), 2)
            for row, wave in zip(group["rows"], group["waves"]):
                self.assertTrue(row["animal_missing"])
                self.assertFalse(row["animal_valid"])
                self.assertTrue(row["average_included"])
                self.assertTrue(np.isnan(row["animal_latency_ms"]))
                correction = float(np.mean(s.data[row["baseline_first"]:row["baseline_last"], 0], dtype=np.float64))
                np.testing.assert_array_equal(wave, s.data[row["first"]:row["last"], 0]-correction)
            np.testing.assert_allclose(group["mean"], np.mean(group["waves"], axis=0))
            times = vep.event_time_markers(group, o["fs"])
            self.assertFalse(any(t["event"] == "animal" for t in times))
            if mode == "dot":
                self.assertEqual([t["event"] for t in times], ["letter", "letter"])
        animal = refresh(s, m, cues, o, "animal")["letter_vep"]["cache"]
        self.assertFalse(animal["selected_indices"])
        self.assertTrue(vep.animal_plot_unavailable(animal))

    def test_without_any_behavior_or_dot_metadata_only_letter_remains_available(self):
        s, m, _, o = prepared()
        letter = refresh(s, m, {}, o, "letter")["letter_vep"]["cache"]
        self.assertEqual(len(letter["selected_indices"]), 2)
        dot = refresh(s, m, {}, o, "dot")["letter_vep"]["cache"]
        self.assertFalse(dot["selected_indices"])
        self.assertTrue(all(r["reason_code"] == "dot_time_missing" for r in dot["records"]))

    def test_known_invalid_trials_remain_excluded(self):
        s, m, cues, o = prepared()
        cues[6250].update(validity="N", animal_latency_sec=np.nan)
        group = refresh(s, m, cues, o, "letter")["letter_vep"]["cache"]
        self.assertEqual([r["trial_id"] for r in group["rows"]], [1])
        self.assertEqual(group["records"][1]["reason_code"], "validity_not_y")

    def test_partial_missing_and_shared_selection_survive_mode_switches(self):
        s, m, cues, o = prepared()
        cues[3250].update(validity="Y", animal_latency_sec=1.)
        for mode, n in (("letter", 2), ("dot", 2), ("animal", 1)):
            group = refresh(s, m, cues, o, mode)["letter_vep"]["cache"]
            self.assertEqual(len(group["selected_indices"]), n)
        revised = refresh(s, m, cues, o, "dot", trial_ids=[2])
        revised = refresh(s, m, cues, revised, "animal")
        self.assertTrue(vep.animal_plot_unavailable(revised["letter_vep"]["cache"]))
        revised = refresh(s, m, cues, revised, "letter")
        self.assertEqual(vep.selected_ids(revised["letter_vep"], (7, 4)), (2,))
        group = revised["letter_vep"]["cache"]
        self.assertEqual([group["rows"][i]["trial_id"] for i in group["selected_indices"]], [2])

    def test_complete_save_omits_animal_images_but_keeps_remaining_images_and_audit(self):
        s, m, cues, o = prepared()
        for mode in ("letter", "dot", "animal"):
            o = refresh(s, m, cues, o, mode)
        names, titles = [], []
        def capture(fig, filename=None, **kwargs):
            if filename is not None:
                names.append(Path(filename).name)
            titles.extend(a.get_title() for a in fig.axes)
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", capture), \
             patch.object(PdfPages, "savefig", lambda pdf, fig=None, **kwargs: capture(fig)):
            folder = export_animal_peak_results(o, s, directory)
            with np.load(folder/"letter_start_vep/ch7_tag4_animal_start.npz") as data:
                self.assertEqual(len(data["selected_trial_ids"]), 0)
            self.assertTrue((folder/"letter_start_vep/trial_details.csv").is_file())
        self.assertIn("ch7_tag4_letter_start.png", names)
        self.assertIn("ch7_tag4_dot_start.png", names)
        self.assertNotIn("ch7_tag4_animal_start.png", names)
        self.assertNotIn("ch7_tag4_animal_start.pdf", names)
        self.assertFalse(any("AnimalStart对齐 VEP" in title for title in titles))


class MissingAnimalQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qt_gui import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from qt_gui import QtAnalysisGUI
        with patch("qt_gui.OperationAuditStore"), patch("qt_gui.FeishuAuditConfig.from_environment",
                return_value=SimpleNamespace(is_configured=False)):
            self.window = QtAnalysisGUI("Missing AnimalStart test")
        self.source, self.markers, self.cues, self.output = prepared()
        w = self.window
        w.active_source = self.source; w.stim_markers = self.markers; w.stim_marker_mode = "Letter"
        w.letter_csv_annotations = self.cues
        if hasattr(w, "behavior_matches"):
            delattr(w, "behavior_matches")
        w._bad_channel_check_completed = True; w.good_channel_ids = {7, 22}; w.bad_channel_ids = set()
        w._task_marker_source = dict(source_path=str(self.source.metadata.path), fs=1000.,
                                    time_offset_sec=1.25, selected_channel_ids=[7])

    def tearDown(self):
        d = getattr(self.window, "_animal_peak_test_dialog", None)
        if d is not None:
            d.deleteLater()
        self.window.deleteLater(); self.app.processEvents()

    def wait(self, d):
        deadline = time.monotonic()+20
        while d._busy and time.monotonic() < deadline:
            self.app.processEvents()
            if d.worker is not None:
                d.worker.wait(5)
        self.app.processEvents(); self.assertFalse(d._busy)

    def test_open_without_txt_prepare_refresh_choose_and_mcp_read_remain_available(self):
        from qt_gui import QMessageBox
        from animal_peak_dialog import AverageTrialSelectionDialog
        from animal_peak_control import AnimalPeakController
        with patch.object(QMessageBox, "information") as info, patch.object(QMessageBox, "warning") as warning:
            self.window.open_animal_peak_test()
            info.assert_not_called(); warning.assert_not_called()
        d = self.window._animal_peak_test_dialog
        self.assertIs(d.tabs.currentWidget(), d.letter_vep_page)
        self.assertEqual(d.matches[3250]["tag100_source"], "csv")
        self.assertNotIn("animal_latency_sec", d.matches[3250])
        d.run_button.click(); self.wait(d)
        self.assertFalse(d.output["groups"])
        self.assertTrue(d.letter_vep_refresh.isEnabled())
        for mode, count in (("letter", 2), ("dot", 2), ("animal", 0), ("letter", 2)):
            d.letter_vep_alignment.setCurrentIndex(d.letter_vep_alignment.findData(mode))
            d.letter_vep_duration.setValue(1500)
            d.letter_vep_refresh.click(); self.wait(d)
            group = d.output["letter_vep"]["cache"]
            self.assertEqual(len(group["selected_indices"]), count)
            self.assertEqual(d.letter_vep_copy.isEnabled(), mode != "animal")
            if mode == "animal":
                self.assertIn("跳过", d.letter_vep_status.text())
        picker = AverageTrialSelectionDialog(d, d.output, (7, 4), letter_group=group)
        self.assertEqual(picker.selected_ids(), [1, 2]); picker.deleteLater()
        controller = AnimalPeakController(d)
        result = controller.dispatch("read_waveform", {"plot": "event", "curve": 2, "limit": 3})
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["result"]["curve_count"], 3)
        with patch.object(QMessageBox, "information") as info:
            self.window.open_animal_peak_test()
            info.assert_not_called()  # Stable alignment context reuses this window.
        self.assertIs(self.window._animal_peak_test_dialog, d)


if __name__ == "__main__":
    unittest.main()
