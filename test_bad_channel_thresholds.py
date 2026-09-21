import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qt_data_model import ArraySource  # noqa: E402
from qt_gui import BadChannelWorker  # noqa: E402


def classify_25mv_ratio(percent: float):
    sample_count = 1000
    near_count = int(round(sample_count * percent / 100.0))
    values = np.zeros(sample_count, dtype=np.float32)
    values[:near_count] = 2500.0
    source = ArraySource(values[:, None], 1000.0, channel_ids=[1])
    worker = BadChannelWorker(source, {
        "flat_std": "1e-4",
        "flat_ratio": "30",
        "parallel": False,
        "workers": "1",
        "high_frequency_noise_only": True,
        "high_frequency_noise_check": True,
        "high_frequency_noise_target": "2500",
        "high_frequency_noise_tolerance": "1",
        "high_frequency_noise_ratio_threshold": "50",
    })
    output, errors = [], []
    worker.completed.connect(lambda *args: output.append(args))
    worker.failed.connect(errors.append)
    worker.run()
    if errors:
        raise AssertionError(errors)
    return output[0]


class BadChannelThresholdTests(unittest.TestCase):
    def test_removed_flat_rules_cannot_mark_a_channel_bad(self):
        source = ArraySource(np.zeros((1000, 1), dtype=np.float32), 1000.0, channel_ids=[1])
        worker = BadChannelWorker(source, {
            "flat_std": "0.01", "flat_ratio": "1",
            "global_flat_check": True, "discrete_level_check": True,
            "flat_time_check": True, "parallel": False, "workers": "1",
        })
        output, errors = [], []
        worker.completed.connect(lambda *args: output.append(args))
        worker.failed.connect(errors.append)
        worker.run()
        self.assertFalse(errors)
        self.assertEqual(output[0][0], [1])
        self.assertEqual(output[0][1], {})

    def test_minimum_finite_sample_guard_remains(self):
        source = ArraySource(np.zeros((9, 1), dtype=np.float32), 1000.0, channel_ids=[1])
        worker = BadChannelWorker(source, {"parallel": False, "workers": "1"})
        output, errors = [], []
        worker.completed.connect(lambda *args: output.append(args))
        worker.failed.connect(errors.append)
        worker.run()
        self.assertFalse(errors)
        self.assertIn("too few finite samples", output[0][1][1])

    def test_saturation_still_detects_constant_signal(self):
        source = ArraySource(np.zeros((1000, 1), dtype=np.float32), 1000.0, channel_ids=[1])
        worker = BadChannelWorker(source, {
            "fast_artifact_check": True, "saturation_width_percent": "1",
            "saturation_ratio_threshold": "50", "parallel": False, "workers": "1",
        })
        output, errors = [], []
        worker.completed.connect(lambda *args: output.append(args))
        worker.failed.connect(errors.append)
        worker.run()
        self.assertFalse(errors)
        self.assertIn("saturation: bottom=100.0%", output[0][1][1])

    def test_default_saturation_ratio_is_49_percent(self):
        values = np.concatenate((
            np.full(495, -1.0, dtype=np.float32),
            np.full(505, 1.0, dtype=np.float32),
        ))
        source = ArraySource(values[:, None], 1000.0, channel_ids=[1])
        worker = BadChannelWorker(source, {
            "fast_artifact_check": True,
            "parallel": False, "workers": "1",
        })
        output, errors = [], []
        worker.completed.connect(lambda *args: output.append(args))
        worker.failed.connect(errors.append)
        worker.run()
        self.assertFalse(errors)
        self.assertIn("saturation: bottom=49.5%", output[0][1][1])
        self.assertEqual(worker.fast_artifact_rows[0]["saturation_ratio_threshold"], 0.49)

    def test_below_50_percent_does_not_trigger_rule(self):
        good, bad, noise = classify_25mv_ratio(49.9)
        self.assertEqual(good, [1])
        self.assertFalse(bad)
        self.assertNotIn(1, noise)

    def test_exactly_50_percent_does_not_trigger_rule(self):
        good, bad, noise = classify_25mv_ratio(50.0)
        self.assertEqual(good, [1])
        self.assertFalse(bad)
        self.assertNotIn(1, noise)

    def test_above_50_percent_is_bad(self):
        good, bad, noise = classify_25mv_ratio(50.1)
        self.assertNotIn(1, good)
        self.assertIn(1, bad)
        self.assertNotIn(1, noise)

    def test_full_concentration_is_bad_not_noise(self):
        good, bad, noise = classify_25mv_ratio(100.0)
        self.assertNotIn(1, good)
        self.assertIn(1, bad)
        self.assertNotIn(1, noise)


if __name__ == "__main__":
    unittest.main()
