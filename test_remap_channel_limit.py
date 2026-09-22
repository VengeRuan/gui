import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qt_data_model import (  # noqa: E402
    ArraySource,
    remap_array,
    remap_source_from_excel,
    remap_source_streaming,
)


class RemapChannelLimitTests(unittest.TestCase):
    def setUp(self):
        self.values = np.arange(3 * 520, dtype=np.float32).reshape(3, 520)
        self.mapping = np.arange(1, 521, dtype=np.int64)

    def test_array_remap_keeps_all_destinations_compactly(self):
        result, _cache = remap_array(
            self.values,
            self.mapping,
            max_output_channels=512,
        )
        self.assertEqual(result.shape, (3, 520))
        np.testing.assert_array_equal(result, self.values)

    def test_streaming_remap_keeps_all_destinations_compactly(self):
        source = ArraySource(self.values, fs=1000, channel_ids=np.arange(1, 521))
        result, _cache = remap_source_streaming(
            source,
            self.mapping,
            max_output_channels=512,
        )
        self.assertEqual(result.shape, (3, 520))
        np.testing.assert_array_equal(result, self.values)

    def test_permutation_keeps_sources_that_map_into_1_to_512(self):
        mapping = np.roll(self.mapping, 8)
        result, _cache = remap_array(
            self.values,
            mapping,
            max_output_channels=512,
        )
        self.assertEqual(result.shape, (3, 520))
        np.testing.assert_array_equal(result, self.values[:, np.argsort(mapping)])

    def test_512_source_columns_keep_destinations_513_to_520(self):
        values = self.values[:, :512]
        mapping = np.concatenate((np.arange(1, 512), [520])).astype(np.int64)
        source = ArraySource(values, fs=1000, channel_ids=np.arange(1, 513))
        result, _cache = remap_source_streaming(
            source, mapping,
            max_output_channels=512,
        )
        self.assertEqual(result.shape, (3, 512))
        np.testing.assert_array_equal(result[:, :511], values[:, :511])
        np.testing.assert_array_equal(result[:, 511], values[:, 511])

    def test_520_input_uses_only_first_512_columns_for_520_destinations(self):
        mapping = np.concatenate((np.arange(1, 512), [520])).astype(np.int64)
        source = ArraySource(self.values, fs=1000, channel_ids=np.arange(1, 521))
        result, _cache = remap_source_streaming(
            source, mapping, source_columns=np.arange(512),
            max_output_channels=512,
        )
        self.assertEqual(result.shape, (3, 512))
        np.testing.assert_array_equal(result[:, 511], self.values[:, 511])
        self.assertNotIn(self.values[0, 512], result[0])

    def test_arbitrarily_large_target_ids_are_not_truncated_or_padded(self):
        values = np.asarray([[11, 22, 33]], dtype=np.float32)
        mapping = np.asarray([10007, 3, 800], dtype=np.int64)
        source = ArraySource(values, fs=1000, channel_ids=[1, 2, 3])
        result, _cache = remap_source_streaming(
            source, mapping, max_output_channels=512,
        )
        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_array_equal(result[0], [22, 33, 11])
        self.assertFalse(np.any(result == 0))

    def test_button_remap_compacts_missing_target_ids_without_zero_columns(self):
        from openpyxl import Workbook

        values = np.asarray([[10, 20, 30, 40]], dtype=np.float32)
        source = ArraySource(values, fs=1000, channel_ids=np.arange(1, 5))
        # Source 1->target 5, source 2->target 1, source 3->target 8,
        # source 4->target 3. Targets 2,4,6,7 do not become zero columns.
        mapping = [5, 1, 8, 3]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "mapping.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet["H1"] = "target"
            for row, target in enumerate(mapping, 2):
                sheet.cell(row, 8, target)
            workbook.save(path)
            result, _cache, actual, source_ids, target_ids = remap_source_from_excel(
                source, path,
            )

        self.assertEqual(result.shape, (1, 4))
        np.testing.assert_array_equal(actual, mapping)
        np.testing.assert_array_equal(source_ids, [1, 2, 3, 4])
        np.testing.assert_array_equal(target_ids, [1, 3, 5, 8])
        np.testing.assert_array_equal(result[0], [20, 40, 10, 30])
        self.assertFalse(np.any(result == 0))


if __name__ == "__main__":
    unittest.main()
