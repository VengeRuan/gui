"""Regression checks for source tracing and first-512-column remapping."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qt_gui
from qt_data_model import ArraySource, electrode_remap_output_limit, remap_source_streaming


class ProcessedSourceTraceTests(unittest.TestCase):
    def test_timestamped_processed_h5_resolves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            animal = Path(folder) / "20260622" / "1102"
            processed_dir = animal / "processed"
            processed_dir.mkdir(parents=True)
            raw = animal / "20260622_1102_block_1.h5"
            reviewed = processed_dir / "20260622_1102_block_1_20260917_115545_lfp_processed.h5"
            with h5py.File(raw, "w") as h5:
                h5.create_dataset("rawData512", data=np.zeros((4, 2), dtype=np.float32))
            with h5py.File(reviewed, "w"):
                pass
            self.assertEqual(qt_gui.resolve_original_h5_for_processed(reviewed), raw.resolve())

    def test_520_raw_columns_remap_only_first_512_to_targets_up_to_520(self):
        values = np.tile(np.arange(1, 521, dtype=np.float32), (4, 1))
        source = ArraySource(values, 1000.0, channel_ids=np.arange(1, 521))
        mapping = np.concatenate((np.arange(1, 505), np.arange(513, 521)))
        original_reader = qt_gui.read_channel_remap
        requests = []

        def read_mapping(_path, count):
            requests.append(count)
            return mapping

        qt_gui.read_channel_remap = read_mapping
        try:
            columns, selected_mapping = qt_gui.electrode_remap_selection(
                "mapping.xlsx", source.metadata,
            )
            self.assertEqual(requests, [512])
            np.testing.assert_array_equal(columns, np.arange(512))
            remapped, _storage_path = remap_source_streaming(
                source, selected_mapping,
                max_output_channels=electrode_remap_output_limit(len(columns)),
                source_columns=columns,
            )
            self.assertEqual(remapped.shape, (4, 520))
            np.testing.assert_array_equal(remapped[:, 504:512], 0)
            np.testing.assert_array_equal(remapped[0, 512:520], np.arange(505, 513))
        finally:
            qt_gui.read_channel_remap = original_reader

    def test_preprocess_worker_accepts_selected_source_columns(self):
        values = np.tile(np.arange(1, 521, dtype=np.float32), (4, 1))
        source = ArraySource(values, 1000.0, channel_ids=np.arange(1, 521))
        mapping = np.concatenate((np.arange(1, 512), [520]))
        original_reader = qt_gui.read_channel_remap
        qt_gui.read_channel_remap = lambda _path, _count: mapping
        try:
            worker = qt_gui.PreprocessWorker(source, "mapping.xlsx", "off", 0.1, 300.0)
            completed, errors = [], []
            worker.completed.connect(lambda *args: completed.append(args))
            worker.failed.connect(errors.append)
            worker.run()
            self.assertFalse(errors, errors)
            self.assertTrue(completed[0][2])
            remapped = completed[0][0]
            self.assertEqual(remapped.metadata.channels, 520)
            np.testing.assert_array_equal(remapped.data[:, 519], values[:, 511])
        finally:
            qt_gui.read_channel_remap = original_reader


if __name__ == "__main__":
    unittest.main()
