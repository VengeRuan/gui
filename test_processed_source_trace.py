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
from qt_data_model import ArraySource, remap_source_streaming


class ProcessedSourceTraceTests(unittest.TestCase):
    def test_high_recall_accepts_any_overlapping_real_channel_ids(self):
        worker = qt_gui.BadChannelControlledExperimentWorker(
            [], {}, "mapping.xlsx", [], {},
            parameter_key="saturation_width_percent", values=[1.0],
            allow_partial_review_scope=True,
        )
        evaluated = worker._evaluation_scope(
            {"evaluated_ids": {3, 520, 10007, 20000}},
            {1, 3, 520, 8000, 10007},
        )
        self.assertEqual(evaluated, {3, 520, 10007})

    def test_other_controlled_experiments_remain_strict(self):
        worker = qt_gui.BadChannelControlledExperimentWorker(
            [], {}, "mapping.xlsx", [], {},
        )
        with self.assertRaisesRegex(ValueError, "没有覆盖全部"):
            worker._evaluation_scope(
                {"evaluated_ids": {3, 520}}, {3, 520, 10007},
            )

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

    def test_512_raw_columns_keep_sparse_targets_up_to_520_compactly(self):
        values = np.tile(np.arange(1, 513, dtype=np.float32), (4, 1))
        source = ArraySource(values, 1000.0, channel_ids=np.arange(1, 513))
        mapping = np.concatenate((np.arange(1, 505), np.arange(513, 521)))
        remapped, _storage_path = remap_source_streaming(
            source, mapping, max_output_channels=512,
        )
        self.assertEqual(remapped.shape, (4, 512))
        np.testing.assert_array_equal(remapped, values)

    def test_preprocess_worker_accepts_selected_source_columns(self):
        from openpyxl import Workbook

        values = np.tile(np.arange(1, 513, dtype=np.float32), (4, 1))
        source = ArraySource(values, 1000.0, channel_ids=np.arange(1, 513))
        mapping = np.concatenate((np.arange(1, 512), [520]))
        with tempfile.TemporaryDirectory() as folder:
            mapping_path = Path(folder) / "mapping.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet["H1"] = "target"
            for row, target in enumerate(mapping, 2):
                sheet.cell(row, 8, int(target))
            workbook.save(mapping_path)
            worker = qt_gui.PreprocessWorker(source, str(mapping_path), "off", 0.1, 300.0)
            completed, errors = [], []
            worker.completed.connect(lambda *args: completed.append(args))
            worker.failed.connect(errors.append)
            worker.run()
            self.assertFalse(errors, errors)
            self.assertTrue(completed[0][2])
            remapped = completed[0][0]
            self.assertEqual(remapped.metadata.channels, 512)
            self.assertEqual(remapped.metadata.channel_ids[-1], 520)
            np.testing.assert_array_equal(remapped.data[:, 511], values[:, 511])

    def test_dual_stream_h5_preserves_compact_large_channel_ids(self):
        values = np.zeros((8, 3), dtype=np.float32)
        source = ArraySource(values, 10000.0, channel_ids=[3, 520, 10007])
        with tempfile.TemporaryDirectory() as folder:
            worker = qt_gui.DualBranchStreamWorker(
                source, folder, "compact", {3, 520, 10007},
                {"schema": "sd-preprocess-qc", "version": 1, "completed": True},
            )
            path = Path(folder) / "branch.partial.h5"
            h5, dataset = worker._create_partial(path, "lfp", {})
            try:
                self.assertEqual(dataset.shape, (8, 3))
                np.testing.assert_array_equal(h5["channel_ids"][()], [3, 520, 10007])
            finally:
                h5.close()


if __name__ == "__main__":
    unittest.main()
