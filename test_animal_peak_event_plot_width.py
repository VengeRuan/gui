"""Manual event-frame widths are display-only and survive normal refreshes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

import animal_peak_letter_vep as vep
import test_animal_peak_event_plot_layout as layout
import test_animal_peak_letter_vep as legacy


class EventPlotWidthQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        layout.CompactPlotQtTests.setUpClass()
        cls.app = layout.CompactPlotQtTests.app

    setUp = layout.CompactPlotQtTests.setUp
    tearDown = layout.CompactPlotQtTests.tearDown
    wait_worker = layout.CompactPlotQtTests.wait_worker

    def apply_width(self, width=600):
        self.dialog.letter_vep_width.setValue(width)
        self.dialog.letter_vep_width_apply.click()

    def test_default_auto_pending_input_then_manual_changes_geometry_only(self):
        d = self.dialog
        group = d.output["letter_vep"]["cache"]
        waves = group["waves"].copy()
        mean = group["mean"].copy()
        initial = d.letter_vep_plot.width()
        self.assertIsNone(d._event_plot_width)
        d.letter_vep_width.setValue(600)
        d._sync_event_plot_size()
        self.assertEqual(d.letter_vep_plot.width(), initial)
        d.letter_vep_y_min.setValue(-30); d.letter_vep_y_max.setValue(40)
        d.apply_event_y_axis()
        with patch.object(self.source, "read", side_effect=AssertionError("width must not read signal")):
            self.apply_width()
        self.assertEqual(d.letter_vep_plot.width(), 600)
        self.assertEqual(d.letter_vep_plot.height(), 340)
        self.assertLessEqual(d.letter_vep_plot.centralWidget.geometry().right(), 601)
        np.testing.assert_allclose(d.letter_vep_plot.viewRange(), [[0, 1000], [-30, 40]])
        self.assertIs(d.output["letter_vep"]["cache"], group)
        np.testing.assert_array_equal(group["waves"], waves)
        np.testing.assert_array_equal(group["mean"], mean)
        self.assertEqual(d.output["display_settings"]["event_vep_plot"]["width_px"], 600)

    def test_large_requested_width_fits_page_and_grows_back_when_window_grows(self):
        d = self.dialog
        self.apply_width(2000)
        self.assertEqual(d._event_plot_width, 2000)
        self.assertLess(d.letter_vep_plot.width(), 2000)
        self.assertLessEqual(d.letter_vep_plot.width(), d.page_scroll.viewport().width()-32)
        d.resize(2600, 960); self.app.processEvents(); d._sync_event_plot_size()
        self.assertEqual(d.letter_vep_plot.width(), 2000)
        self.apply_width(260)
        self.assertGreaterEqual(d.letter_vep_plot.width(), 260)
        self.assertLessEqual(d.letter_vep_plot.centralWidget.geometry().right(), d.letter_vep_plot.width()+1)

    def test_refresh_alignment_channel_layout_and_other_display_changes_keep_width(self):
        d = self.dialog
        self.apply_width()
        d.letter_vep_alignment.setCurrentIndex(d.letter_vep_alignment.findData("animal"))
        d.refresh_letter_vep(); self.wait_worker()
        self.assertEqual(d.letter_vep_plot.width(), 600)
        d.result_channel_box.setCurrentIndex(1)
        d.refresh_letter_vep(); self.wait_worker()
        self.assertEqual(d.letter_vep_plot.width(), 600)
        d.details_per_page_box.setCurrentIndex(1); self.app.processEvents()
        self.assertEqual(d.letter_vep_plot.width(), 600)
        self.assertEqual(d.letter_vep_plot.height(), 400)
        d.restore_event_auto_y_axis()
        self.assertEqual(d.output["display_settings"]["event_vep_plot"]["width_px"], 600)
        self.assertEqual(d.output["display_settings"]["event_vep_plot"]["height_px"], 400)

    def test_restore_auto_and_busy_guards_keep_cache_and_pending_input_is_not_applied(self):
        d = self.dialog
        self.apply_width()
        group = d.output["letter_vep"]["cache"]
        d.letter_vep_width.setValue(900)
        d._sync_event_plot_size()
        self.assertEqual(d.letter_vep_plot.width(), 600)
        d._set_busy(True)
        self.assertFalse(d.letter_vep_width_box.isEnabled())
        d.restore_event_auto_width(); d.apply_event_plot_width()
        self.assertEqual(d._event_plot_width, 600)
        d._set_busy(False)
        with patch.object(self.source, "read", side_effect=AssertionError("auto width must not read signal")):
            d.letter_vep_width_auto.click()
        self.assertIsNone(d._event_plot_width)
        self.assertLess(d.letter_vep_plot.width(), 600)
        self.assertNotIn("event_vep_plot", d.output["display_settings"])
        self.assertIs(d.output["letter_vep"]["cache"], group)


class EventPlotWidthExportTests(unittest.TestCase):
    def test_export_matches_applied_frame_aspect_and_keeps_numerical_results(self):
        source, markers, matches, output = legacy.prepared()
        output = legacy.calculate(source, markers, matches, output, config=dict(end_ms=1000))
        group = output["letter_vep"]["cache"]
        output["display_settings"] = {"event_vep_plot": dict(width_mode="manual",
            requested_width_px=900, width_px=720, height_px=340)}
        sizes = []
        def capture(figure, *args, **kwargs):
            sizes.append(tuple(figure.get_size_inches()))
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", capture):
            vep.export_results(source, output, directory)
            with np.load(Path(directory)/"letter_start_vep/ch7_tag4_letter_start.npz") as data:
                np.testing.assert_array_equal(data["mean"], group["mean"])
                np.testing.assert_array_equal(data["trials"], group["waves"])
                np.testing.assert_array_equal(data["time_ms"], group["time_ms"])
        self.assertTrue(sizes)
        for size in sizes:
            np.testing.assert_allclose(size, [720/96, 340/96])


if __name__ == "__main__":
    unittest.main()
