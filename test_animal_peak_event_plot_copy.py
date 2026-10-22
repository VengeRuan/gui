"""Copy the full displayed event plot, without recomputing or touching files."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

import test_animal_peak_event_vep as events
from animal_peak_dialog import QtWidgets


class EventPlotCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        events.EventVepQtTests.setUpClass()
        cls.app = events.EventVepQtTests.app

    def setUp(self):
        events.EventVepQtTests.setUp(self)
        d = self.dialog
        d.resize(1360, 960); d.parameter_toggle.setChecked(False); d.show()
        self.app.processEvents()
        d.letter_vep_duration.setValue(1000)

    tearDown = events.EventVepQtTests.tearDown
    wait_worker = events.EventVepQtTests.wait_worker
    mode = events.EventVepQtTests.mode

    def refresh(self, **kwargs):
        self.dialog.refresh_letter_vep(**kwargs)
        self.wait_worker()

    def test_copy_contains_title_and_axes_full_widget_even_when_page_is_scrolled(self):
        d = self.dialog
        self.refresh()
        d.letter_vep_width.setValue(600); d.apply_event_plot_width()
        d.page_scroll.verticalScrollBar().setValue(d.page_scroll.verticalScrollBar().maximum())
        self.app.processEvents()
        self.assertTrue(d.letter_vep_copy.isEnabled())
        clipboard = Mock()
        group = d.output["letter_vep"]["cache"]
        before = group["mean"].copy()
        with patch.object(QtWidgets.QApplication, "clipboard", return_value=clipboard), \
             patch.object(self.source, "read", side_effect=AssertionError("copy must not reread signal")):
            d.letter_vep_copy.click()
        clipboard.setPixmap.assert_called_once()
        pixmap = clipboard.setPixmap.call_args.args[0]
        self.assertFalse(pixmap.isNull())
        ratio = pixmap.devicePixelRatio()
        self.assertAlmostEqual(pixmap.width()/ratio, d.letter_vep_plot.width(), delta=1)
        self.assertAlmostEqual(pixmap.height()/ratio, d.letter_vep_plot.height(), delta=1)
        self.assertGreater(pixmap.height()/ratio, d.letter_vep_plot.plotItem.vb.height())
        self.assertIn("LetterStart", d.letter_vep_plot.plotItem.titleLabel.text)
        self.assertIn("Tag 4", d.letter_vep_plot.plotItem.titleLabel.text)
        # Actual title pixels, not just a data-only ViewBox screenshot.
        image = pixmap.toImage()
        self.assertTrue(any(image.pixelColor(x, y).lightness() < 220
            for y in range(0, min(image.height(), int(25*ratio)), 2)
            for x in range(0, image.width(), 2)))
        self.assertIn("Ctrl+V", d.status.text())
        self.assertIs(d.output["letter_vep"]["cache"], group)
        np.testing.assert_array_equal(group["mean"], before)

    def test_pending_parameters_copy_displayed_result_and_keep_zoom_y_and_data(self):
        d = self.dialog
        self.refresh()
        d.letter_vep_y_min.setValue(-30); d.letter_vep_y_max.setValue(40); d.apply_event_y_axis()
        group = d.output["letter_vep"]["cache"]
        d.letter_vep_duration.setValue(2000)
        self.assertIn("待刷新", d.letter_vep_status.text())
        d.letter_vep_plot.setXRange(200, 700, padding=0)
        clipboard = Mock()
        with patch.object(QtWidgets.QApplication, "clipboard", return_value=clipboard), \
             patch.object(self.source, "read", side_effect=AssertionError("copy must not recompute")):
            d.copy_event_plot()
        clipboard.setPixmap.assert_called_once()
        np.testing.assert_allclose(d.letter_vep_plot.viewRange(), [[200, 700], [-30, 40]])
        self.assertIs(d.output["letter_vep"]["cache"], group)
        self.assertEqual(group["waves"].shape, (2, 1001))

    def test_no_current_result_never_overwrites_clipboard_or_copies_old_mode(self):
        d = self.dialog
        self.assertFalse(d.letter_vep_copy.isEnabled())
        with patch.object(QtWidgets.QApplication, "clipboard") as clipboard, \
             patch.object(QtWidgets.QMessageBox, "warning") as warning:
            d.copy_event_plot()
        clipboard.assert_not_called(); warning.assert_called_once()
        self.refresh()
        self.mode("dot")
        self.assertFalse(d.letter_vep_copy.isEnabled())
        with patch.object(QtWidgets.QApplication, "clipboard") as clipboard, \
             patch.object(QtWidgets.QMessageBox, "warning") as warning:
            d.copy_event_plot()
        clipboard.assert_not_called(); warning.assert_called_once()

    def test_busy_stale_or_different_channel_cannot_copy_old_result(self):
        d = self.dialog
        self.refresh()
        d._set_busy(True)
        self.assertFalse(d.letter_vep_copy.isEnabled())
        with patch.object(QtWidgets.QApplication, "clipboard") as clipboard:
            d.copy_event_plot()
        clipboard.assert_not_called()
        d._set_busy(False)
        self.assertTrue(d.letter_vep_copy.isEnabled())
        d.result_channel_box.setCurrentIndex(1)
        self.assertFalse(d.letter_vep_copy.isEnabled())
        self.refresh()
        d.output["letter_vep"]["stale"] = True
        d._render_letter_vep()
        self.assertFalse(d.letter_vep_copy.isEnabled())
        with patch.object(QtWidgets.QApplication, "clipboard") as clipboard, \
             patch.object(QtWidgets.QMessageBox, "warning"):
            d.copy_event_plot()
        clipboard.assert_not_called()

    def test_all_three_alignment_titles_and_empty_selection_can_be_copied(self):
        d = self.dialog
        for mode, title in (("letter", "LetterStart"), ("dot", "Dot"), ("animal", "AnimalStart")):
            self.mode(mode); self.refresh(trial_ids=[])
            self.assertTrue(d.letter_vep_copy.isEnabled())
            clipboard = Mock()
            with patch.object(QtWidgets.QApplication, "clipboard", return_value=clipboard):
                d.copy_event_plot()
            clipboard.setPixmap.assert_called_once()
            self.assertIn(title, d.letter_vep_plot.plotItem.titleLabel.text)
            self.assertIn("n=0", d.letter_vep_plot.plotItem.titleLabel.text)

    def test_failed_capture_does_not_overwrite_existing_clipboard(self):
        d = self.dialog
        self.refresh()
        with patch.object(d.letter_vep_plot, "grab", return_value=Mock(isNull=lambda: True)), \
             patch.object(QtWidgets.QApplication, "clipboard") as clipboard, \
             patch.object(QtWidgets.QMessageBox, "warning") as warning:
            d.copy_event_plot()
        clipboard.assert_not_called(); warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
