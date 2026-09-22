import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qt_data_model import filter_array  # noqa: E402


class ParallelFilterTests(unittest.TestCase):
    def test_two_thread_result_matches_single_thread(self):
        rng = np.random.default_rng(20260922)
        values = rng.normal(size=(4000, 24)).astype(np.float32)
        serial, _ = filter_array(
            values, 1000.0, "bandpass", 1.0, 200.0,
            notch=True, parallel_workers=1,
        )
        parallel, _ = filter_array(
            values, 1000.0, "bandpass", 1.0, 200.0,
            notch=True, parallel_workers=2,
        )
        np.testing.assert_array_equal(parallel, serial)

    def test_parallel_selected_channels_leave_others_unchanged(self):
        rng = np.random.default_rng(22)
        values = rng.normal(size=(2000, 20)).astype(np.float32)
        selected = np.arange(2, 19, dtype=np.int64)
        result, _ = filter_array(
            values, 1000.0, "lowpass", high=100.0,
            channels=selected, parallel_workers=2,
        )
        np.testing.assert_array_equal(result[:, :2], values[:, :2])
        np.testing.assert_array_equal(result[:, 19:], values[:, 19:])
        self.assertFalse(np.array_equal(result[:, selected], values[:, selected]))

    def test_memory_limit_falls_back_without_changing_result(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(1000, 16)).astype(np.float32)
        messages = []
        result, _ = filter_array(
            values, 1000.0, "highpass", low=1.0,
            progress=lambda _value, message: messages.append(message),
            parallel_workers=2, parallel_memory_budget_bytes=1,
        )
        self.assertEqual(result.shape, values.shape)
        self.assertTrue(messages)
        self.assertTrue(all("单线程" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
