"""Regression checks for grouped LFP metric parameters and shared controllers."""

import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qt_data_model import ArraySource
from qt_gui import QApplication, QLabel, QPushButton, QtAnalysisGUI, TaskEpochWorker, ItpcWorker


class LfpMetricParametersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # UI tests must not create operator audit files or start cloud services.
        with patch("qt_gui.OperationAuditStore"), patch(
            "qt_gui.FeishuAuditConfig.from_environment",
            return_value=SimpleNamespace(is_configured=False),
        ):
            cls.window = QtAnalysisGUI(operator_name="UI test")

    @classmethod
    def tearDownClass(cls):
        cls.window.deleteLater()
        cls.app.processEvents()

    def test_controls_are_beside_the_metrics_they_affect(self):
        window = self.window
        expected = {
            "local_tag_snr_db": {"task_target_freq_var", "neighbor_bins_var"},
            "target_power_change_db": {
                "task_target_freq_var", "baseline_start_ms_var", "baseline_end_ms_var",
            },
            "event_lfp_snr_db": {
                "baseline_start_ms_var", "baseline_end_ms_var",
                "response_start_ms_var", "response_end_ms_var",
            },
        }
        for key, names in expected.items():
            group = window.lfp_metric_parameter_groups[key]
            self.assertIn(key, group.title())
            actual = {
                name for name, editors in window.lfp_metric_parameter_editors.items()
                if any(group.isAncestorOf(editor) for editor in editors)
            }
            self.assertEqual(actual, names)
            self.assertTrue(any("10·log" in label.text() for label in group.findChildren(QLabel)))

    def test_shared_frequency_and_baseline_edits_reach_original_controllers(self):
        window = self.window
        for name in ("task_target_freq_var", "baseline_start_ms_var", "baseline_end_ms_var"):
            source = getattr(window, name)
            original = source.text()
            editors = window.lfp_metric_parameter_editors[name]
            self.assertEqual(len(editors), 2)
            try:
                editors[-1].setText("12.5")
                self.assertEqual(source.text(), "12.5")
                self.assertTrue(all(editor.text() == "12.5" for editor in editors))
                source.setText("23.5")
                self.assertTrue(all(editor.text() == "23.5" for editor in editors))
            finally:
                source.setText(original)
        # Alignment keeps its original controls; the LFP panel did not steal them.
        for name in ("baseline_start_ms_var", "baseline_end_ms_var", "response_start_ms_var"):
            self.assertTrue(window.task_parameter_panel.isAncestorOf(getattr(window, name)))

    def test_shared_processing_and_waveform_choices_sync_both_ways(self):
        window = self.window
        source = window.smooth_signal_var
        editor = window.lfp_metric_parameter_editors["smooth_signal_var"][0]
        original = source.isChecked()
        try:
            editor.setChecked(not original)
            self.assertEqual(source.isChecked(), not original)
            source.setChecked(original)
            self.assertEqual(editor.isChecked(), original)
        finally:
            source.setChecked(original)
        source = window.task_response_aggregate_var
        editor = window.lfp_metric_parameter_editors["task_response_aggregate_var"][0]
        original = source.currentIndex()
        try:
            editor.setCurrentIndex(1)
            self.assertEqual(source.currentText(), "Median")
            source.setCurrentIndex(0)
            self.assertEqual(editor.currentText(), "Mean")
        finally:
            source.setCurrentIndex(original)

    def test_baseline_warning_updates_without_changing_defaults(self):
        window = self.window
        names = ("task_target_freq_var", "baseline_start_ms_var", "baseline_end_ms_var", "epoch_start_ms_var")
        originals = {name: getattr(window, name).text() for name in names}
        try:
            window.epoch_start_ms_var.setText("-1500")
            window.task_target_freq_var.setText("1")
            window.baseline_start_ms_var.setText("-200")
            window.baseline_end_ms_var.setText("0")
            self.assertIn("NaN/空值", window.lfp_metric_baseline_warning.text())
            self.assertIn("1000 ms", window.lfp_metric_baseline_warning.text())
            window.lfp_metric_parameter_editors["baseline_start_ms_var"][-1].setText("-1200")
            self.assertIn("已满足", window.lfp_metric_baseline_warning.text())
            window.task_target_freq_var.setText("")
            self.assertIn("请填写有效", window.lfp_metric_baseline_warning.text())
        finally:
            for name, value in originals.items():
                getattr(window, name).setText(value)
        self.assertEqual({name: getattr(window, name).text() for name in names}, originals)

    def test_parameter_toggle_keeps_metric_groups_and_folded_auxiliary_controls(self):
        panel = self.window.lfp_parameter_panel
        self.assertFalse(panel.property("parameters_expanded"))
        self.window._toggle_parameter_section(panel)
        try:
            for key in ("local_tag_snr_db", "target_power_change_db", "event_lfp_snr_db"):
                self.assertFalse(self.window.lfp_metric_parameter_groups[key].isHidden())
            auxiliary = [box for box in panel.findChildren(type(panel)) if "不直接控制" in box.title() or "旧版分析" in box.title()]
            self.assertEqual(len(auxiliary), 2)
            self.assertTrue(all(not box.property("parameters_expanded") for box in auxiliary))
            for box in auxiliary:
                self.window._toggle_parameter_section(box)
                self.assertTrue(box.property("parameters_expanded"))
                self.window._toggle_parameter_section(box)
        finally:
            self.window._toggle_parameter_section(panel)

    def test_documented_direct_dependencies_match_worker_results(self):
        fs = 250.0
        times = np.arange(2000) / fs
        rng = np.random.default_rng(27)
        amplitude = np.where(times < 2.0, .1, .4)
        values = amplitude * np.sin(2 * np.pi * 10 * times) + rng.normal(0, .02, times.size)
        source = ArraySource(values.astype(np.float32)[:, None], fs, channel_ids=[7])
        settings = {
            "mode": "Letter", "epoch_start": -1500, "epoch_end": 6000,
            "baseline_start": -1200, "baseline_end": -200,
            "response_start": 0, "response_end": 300,
            "trials_per_stim": 0, "tags": [4], "aggregate": "mean", "columns": [0],
            "task_target_freq_hz": 10.0, "task_neighbor_bins": 4,
        }

        def calculate(**changes):
            worker = TaskEpochWorker(source, np.asarray([[500, 4]]), {**settings, **changes})
            results, errors = [], []
            worker.completed.connect(results.append)
            worker.failed.connect(errors.append)
            worker.run()
            self.assertFalse(errors, errors)
            return results[0]["results"][0]["metrics"][0]

        original = calculate()
        for key in ("local_tag_snr_db", "target_power_change_db", "event_lfp_snr_db"):
            self.assertTrue(np.isfinite(original[key]), key)
        neighbor = calculate(task_neighbor_bins=1)
        self.assertNotAlmostEqual(original["local_tag_snr_db"], neighbor["local_tag_snr_db"])
        for key in ("target_power_change_db", "event_lfp_snr_db"):
            self.assertAlmostEqual(original[key], neighbor[key])
        response = calculate(response_start=500, response_end=1500)
        self.assertNotAlmostEqual(original["event_lfp_snr_db"], response["event_lfp_snr_db"])
        for key in ("local_tag_snr_db", "target_power_change_db"):
            self.assertAlmostEqual(original[key], response[key])
        short_baseline = calculate(baseline_start=-20, baseline_end=0)
        self.assertTrue(np.isnan(short_baseline["target_power_change_db"]))
        self.assertTrue(np.isfinite(short_baseline["event_lfp_snr_db"]))
        self.assertAlmostEqual(original["local_tag_snr_db"], short_baseline["local_tag_snr_db"])

    def test_five_window_controls_and_callbacks_are_removed(self):
        window = self.window
        self.assertFalse(any("五窗" in button.text() for button in window.findChildren(QPushButton)))
        for name in ("run_lfp_trial_quality_check", "show_lfp_trial_quality_browser",
                     "_read_lfp_trial_quality_settings", "lfp_trial_quality_output",
                     "lfp_qc_available_ratio_var", "lfp_qc_saturation_run_var"):
            self.assertFalse(hasattr(window, name), name)
        self.assertFalse(any(name.startswith("quality_") for name in window._external_baseline_analysis_settings()))

    def test_letter_and_itpc_start_directly_without_quality_results(self):
        window = self.window
        names = ("lfp_source", "stim_markers", "stim_marker_mode", "_lfp_task_worker", "_lfp_itpc_worker")
        sentinel = object()
        previous = {name: getattr(window, name, sentinel) for name in names}
        try:
            window.lfp_source = ArraySource(np.zeros((2000, 1), dtype=np.float32), 250, channel_ids=[7])
            window.stim_markers = np.asarray([[500, 4]])
            window.stim_marker_mode = "Letter"
            with patch.object(window, "_task_analysis_channel_ids", return_value={7}), patch("qt_gui.TaskEpochWorker.start") as start:
                window.run_lfp_letter_frequency_analysis()
                start.assert_called_once()
            worker = window._lfp_task_worker
            self.assertTrue(worker.settings["lfp_task_metrics"])
            self.assertFalse(any(key.startswith("quality_") or "trial_quality" in key for key in worker.settings))
            with patch.object(window, "_task_analysis_channel_ids", return_value={7}), patch("qt_gui.ItpcWorker.start") as start:
                window.run_lfp_itpc()
                start.assert_called_once()
            context = window._lfp_itpc_worker.task_context
            np.testing.assert_array_equal(context["markers"], window.stim_markers)
            self.assertNotIn("trial_quality", context)
        finally:
            for name, value in previous.items():
                if value is sentinel:
                    if hasattr(window, name):
                        delattr(window, name)
                else:
                    setattr(window, name, value)
            window._set_task_buttons_enabled("lfp", True)

    def test_flat_clipped_and_jump_channels_are_not_filtered_out(self):
        rng = np.random.default_rng(10)
        data = rng.normal(0, .05, (2000, 4)).astype(np.float32)
        data[:, 1] = 0
        for first in range(625, 1875, 250):
            data[first:first + 12, 2] = .75
            data[first + 100, 3] = 30
        worker = TaskEpochWorker(ArraySource(data, 250, channel_ids=[7, 8, 9, 10]), np.asarray([[500, 4]]), {
            "mode": "Letter", "epoch_start": -500, "epoch_end": 6000,
            "baseline_start": -200, "baseline_end": 0, "response_start": 0, "response_end": 300,
            "trials_per_stim": 0, "tags": [4], "aggregate": "mean", "columns": [0, 1, 2, 3],
            "task_target_freq_hz": 10.0, "task_neighbor_bins": 4,
        })
        outputs, errors = [], []
        worker.completed.connect(outputs.append)
        worker.failed.connect(errors.append)
        worker.run()
        self.assertFalse(errors, errors)
        for metric in outputs[0]["results"][0]["metrics"]:
            self.assertEqual(metric["usable_trials"], 1)
            self.assertEqual(metric["rejected_trials"], 0)
            self.assertEqual(metric["spectral_seconds"], 5.0)
            self.assertNotIn("clean_windows", metric)
        # Check the unchanged Welch formula against the complete, unsplit interval.
        from scipy import signal
        freq, psd = signal.welch(data[625:1875], fs=250, window="hann", nperseg=1250,
                                 noverlap=0, detrend="constant", axis=0, scaling="density")
        index = np.argmin(np.abs(freq - 10))
        np.testing.assert_allclose(outputs[0]["results"][0]["target_power"], psd[index], rtol=1e-6)

    def test_itpc_has_no_artificial_one_second_window_gaps(self):
        fs = 100
        source = ArraySource(np.sin(2 * np.pi * 10 * np.arange(8000) / fs).astype(np.float32)[:, None], fs, channel_ids=[7])
        markers = np.asarray([[1000 + 1000 * index, 4] for index in range(6)])
        worker = ItpcWorker(source, {
            "columns": [0], "channel_ids": [7], "markers": markers,
            "epoch_start_ms": -500, "epoch_end_ms": 6000,
        }, {
            "tags": [4], "freq_low": 10, "freq_high": 10, "freq_step": 1,
            "cycles": 1, "time_start": 500, "time_end": 5500,
            "baseline_start": 500, "baseline_end": 1500, "sample_rate": 100, "min_trials": 4,
        })
        outputs, errors = [], []
        worker.completed.connect(outputs.append)
        worker.failed.connect(errors.append)
        worker.run()
        self.assertFalse(errors, errors)
        result = outputs[0]["results"][0]
        self.assertTrue(np.all(result["counts"] == 6))
        self.assertTrue(np.all(result["itpc"] > .99))


if __name__ == "__main__":
    unittest.main()
