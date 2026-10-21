"""Event-only display limits must never change numerical VEP results."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

import animal_peak_letter_vep as vep
import test_animal_peak_letter_vep as legacy


class EventYAxisQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        legacy.LetterVepQtTests.setUpClass()
        cls.app = legacy.LetterVepQtTests.app

    def setUp(self):
        legacy.LetterVepQtTests.setUp(self)
        self.dialog.letter_vep_duration.setValue(1000)
        self.dialog.refresh_letter_vep()
        self.wait_worker()

    tearDown = legacy.LetterVepQtTests.tearDown
    wait_worker = legacy.LetterVepQtTests.wait_worker

    def fix(self, low=-25, high=30):
        d = self.dialog
        d.letter_vep_y_min.setValue(low)
        d.letter_vep_y_max.setValue(high)
        d.letter_vep_y_apply.click()

    def test_default_auto_pending_inputs_and_apply_do_not_touch_data_or_peak_page(self):
        d = self.dialog
        group = d.output["letter_vep"]["cache"]
        before = group["waves"].copy()
        self.assertTrue(d.letter_vep_plot.getViewBox().autoRangeEnabled()[1])
        d.letter_vep_y_min.setValue(-25)
        self.assertIsNone(d._event_y_override)
        d.average_plots[0].setYRange(-200, 300, padding=0)
        peak_view = d.average_plots[0].viewRange()[1]
        with patch.object(self.source, "read", side_effect=AssertionError("display must not read signal")):
            self.fix()
        np.testing.assert_allclose(d.letter_vep_plot.viewRange()[1], [-25, 30])
        self.assertIsNone(d._fixed_y_limits)
        np.testing.assert_allclose(d.average_plots[0].viewRange()[1], peak_view)
        self.assertEqual(d.output["display_settings"]["y_axis_mode"], "auto")
        self.assertEqual(d.output["display_settings"]["event_vep_y_axis"]["y_min"], -25)
        self.assertIs(d.output["letter_vep"]["cache"], group)
        np.testing.assert_array_equal(group["waves"], before)
        d.letter_vep_plot.setYRange(-200, 200)
        np.testing.assert_allclose(d.letter_vep_plot.viewRange()[1], [-25, 30])

    def test_invalid_input_and_busy_actions_preserve_previous_range(self):
        from animal_peak_dialog import QtWidgets
        d = self.dialog
        self.fix()
        with patch.object(QtWidgets.QMessageBox, "warning") as warning:
            self.fix(30, -25)
        warning.assert_called_once()
        np.testing.assert_allclose(d.letter_vep_plot.viewRange()[1], [-25, 30])
        d._set_busy(True)
        self.assertFalse(d.letter_vep_y_box.isEnabled())
        d.restore_event_auto_y_axis()
        self.assertEqual(d._event_y_override["y_axis_mode"], "fixed")
        d._set_busy(False)

    def test_auto_reset_is_independent_from_global_and_refresh_keeps_applied_range(self):
        d = self.dialog
        self.fix()
        d.refresh_letter_vep(); self.wait_worker()
        np.testing.assert_allclose(d.letter_vep_plot.viewRange()[1], [-25, 30])
        d.y_min_spin.setValue(-5); d.y_max_spin.setValue(5); d.apply_fixed_y_axis()
        np.testing.assert_allclose(d.letter_vep_plot.viewRange()[1], [-25, 30])
        with patch.object(self.source, "read", side_effect=AssertionError("auto must not read signal")):
            d.letter_vep_y_auto.click()
        self.assertTrue(d.letter_vep_plot.getViewBox().autoRangeEnabled()[1])
        self.assertEqual(d._fixed_y_limits, (-5, 5))
        self.assertIsNone(d.letter_vep_plot.getViewBox().state["limits"]["yLimits"][0])
        self.assertEqual(d.output["display_settings"]["event_vep_y_axis"]["y_axis_mode"], "auto")

    def test_switch_alignment_and_channel_keep_event_range_and_unit_matches_data(self):
        d = self.dialog
        self.fix(-2, 3)
        d.letter_vep_alignment.setCurrentIndex(d.letter_vep_alignment.findData("animal"))
        d.refresh_letter_vep(); self.wait_worker()
        np.testing.assert_allclose(d.letter_vep_plot.viewRange()[1], [-2, 3])
        d.result_channel_box.setCurrentIndex(1)
        d.refresh_letter_vep(); self.wait_worker()
        np.testing.assert_allclose(d.letter_vep_plot.viewRange()[1], [-2, 3])
        d.output["units"] = "z"
        d._sync_y_axis_display()
        self.assertEqual(d.letter_vep_y_unit.text(), "z")
        self.assertEqual(d.output["display_settings"]["event_vep_y_axis"]["y_unit"], "z")


class EventYAxisExportTests(unittest.TestCase):
    def test_export_uses_event_limits_without_clipping_npz_values(self):
        source, markers, matches, output = legacy.prepared()
        output = legacy.calculate(source, markers, matches, output, config=dict(end_ms=1000))
        group = output["letter_vep"]["cache"]
        mean = group["mean"].copy()
        output["display_settings"] = dict(y_axis_mode="fixed", y_min=-100, y_max=100, y_unit="μV",
            event_vep_y_axis=dict(y_axis_mode="fixed", y_min=-1, y_max=1, y_unit="μV"))
        limits = []
        def capture(figure, *args, **kwargs):
            limits.extend(axis.get_ylim() for axis in figure.axes)
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, "savefig", capture):
            vep.export_results(source, output, directory)
            with np.load(Path(directory)/"letter_start_vep/ch7_tag4_letter_start.npz") as data:
                np.testing.assert_array_equal(data["mean"], mean)
                np.testing.assert_array_equal(data["trials"], group["waves"])
        self.assertTrue(limits)
        for limits_pair in limits:
            np.testing.assert_allclose(limits_pair, [-1, 1])

    def test_event_auto_overrides_global_fixed_but_old_outputs_keep_global_limits(self):
        source, markers, matches, output = legacy.prepared()
        output = legacy.calculate(source, markers, matches, output, config=dict(end_ms=1000))
        display = dict(y_axis_mode="fixed", y_min=-.01, y_max=.01, y_unit="μV")
        output["display_settings"] = display
        fig = Figure()
        axis = fig.subplots()
        vep.draw_group(axis, output, output["letter_vep"]["cache"])
        np.testing.assert_allclose(axis.get_ylim(), [-.01, .01])
        display["event_vep_y_axis"] = dict(y_axis_mode="auto", y_unit="μV")
        fig = Figure()
        axis = fig.subplots()
        vep.draw_group(axis, output, output["letter_vep"]["cache"])
        self.assertTrue(axis.get_autoscaley_on())
        self.assertGreater(axis.get_ylim()[1]-axis.get_ylim()[0], .02)


if __name__ == "__main__":
    unittest.main()
