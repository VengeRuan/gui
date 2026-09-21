import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qt_data_model import (  # noqa: E402
    ArraySource,
    REMAPPED_ELECTRODE_CHANNELS,
    electrode_remap_output_limit,
    remap_array,
    remap_source_streaming,
)


class RemapChannelLimitTests(unittest.TestCase):
    def setUp(self):
        self.values = np.arange(3 * 520, dtype=np.float32).reshape(3, 520)
        self.mapping = np.arange(1, 521, dtype=np.int64)

    def test_array_remap_drops_auxiliary_destinations(self):
        result, _cache = remap_array(
            self.values,
            self.mapping,
            max_output_channels=REMAPPED_ELECTRODE_CHANNELS,
        )
        self.assertEqual(result.shape, (3, 512))
        np.testing.assert_array_equal(result, self.values[:, :512])

    def test_streaming_remap_drops_auxiliary_destinations(self):
        source = ArraySource(self.values, fs=1000, channel_ids=np.arange(1, 521))
        result, _cache = remap_source_streaming(
            source,
            self.mapping,
            max_output_channels=REMAPPED_ELECTRODE_CHANNELS,
        )
        self.assertEqual(result.shape, (3, 512))
        np.testing.assert_array_equal(result, self.values[:, :512])

    def test_permutation_keeps_sources_that_map_into_1_to_512(self):
        mapping = np.roll(self.mapping, 8)
        result, _cache = remap_array(
            self.values,
            mapping,
            max_output_channels=REMAPPED_ELECTRODE_CHANNELS,
        )
        self.assertEqual(result.shape, (3, 512))
        inverse = np.argsort(mapping)
        np.testing.assert_array_equal(result, self.values[:, inverse[:512]])

    def test_512_source_columns_keep_destinations_513_to_520(self):
        values = self.values[:, :512]
        mapping = np.concatenate((np.arange(1, 512), [520])).astype(np.int64)
        source = ArraySource(values, fs=1000, channel_ids=np.arange(1, 513))
        result, _cache = remap_source_streaming(
            source, mapping,
            max_output_channels=electrode_remap_output_limit(source.metadata.channels),
        )
        self.assertEqual(result.shape, (3, 520))
        np.testing.assert_array_equal(result[:, :511], values[:, :511])
        np.testing.assert_array_equal(result[:, 519], values[:, 511])
        self.assertTrue(np.all(result[:, 511:519] == 0))

    def test_520_input_uses_only_first_512_columns_for_520_destinations(self):
        mapping = np.concatenate((np.arange(1, 512), [520])).astype(np.int64)
        source = ArraySource(self.values, fs=1000, channel_ids=np.arange(1, 521))
        result, _cache = remap_source_streaming(
            source, mapping, source_columns=np.arange(512),
            max_output_channels=electrode_remap_output_limit(512),
        )
        self.assertEqual(result.shape, (3, 520))
        np.testing.assert_array_equal(result[:, 519], self.values[:, 511])
        self.assertTrue(np.all(result[:, 511:519] == 0))
        self.assertNotIn(self.values[0, 512], result[0])


if __name__ == "__main__":
    unittest.main()
