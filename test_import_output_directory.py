import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qt_gui import QtAnalysisGUI, _single_import_output_directory, _stream_export_target


class SingleImportOutputDirectoryTests(unittest.TestCase):
    def test_date_and_animal_are_recovered_from_source_folders(self):
        root = Path(r"D:\data\PilotStudy2\正式实验\rawbin\20260623\1108\processed")
        source = Path(r"D:\data\PilotStudy2\正式实验\rawbin\20260623\1108\D0001.bin")
        self.assertEqual(
            _single_import_output_directory(root, source),
            root / "20260623" / "1108",
        )

    def test_explicit_import_parameters_take_priority(self):
        root = Path(r"D:\data\processed")
        source = Path(r"D:\incoming\recording.h5")
        self.assertEqual(
            _single_import_output_directory(root, source, "2026-06-23", "1108"),
            root / "20260623" / "1108",
        )

    def test_already_expanded_directory_is_not_duplicated(self):
        root = Path(r"D:\data\processed\20260623\1108")
        source = Path(r"D:\incoming\recording.bin")
        self.assertEqual(
            _single_import_output_directory(root, source, "20260623", "1108"), root,
        )

    def test_stream_export_follows_loaded_h5_directory(self):
        source = Path(
            r"D:\data\PilotStudy2\正式实验\processed_data\20260608\1102\20260608_1102_block_1.h5"
        )
        output_dir, base_name = _stream_export_target(source, "20260915_160000")
        self.assertEqual(output_dir, source.parent / "processed")
        self.assertEqual(base_name, "20260608_1102_block_1_20260915_160000")

    def test_stream_export_does_not_duplicate_processed_folder(self):
        source = Path(r"D:\data\20260608\1102\processed\input.h5")
        output_dir, _base_name = _stream_export_target(source, "20260915_160000")
        self.assertEqual(output_dir, source.parent)

    def test_stream_export_ignores_interactive_filtered_source(self):
        raw = SimpleNamespace(loaded=True)
        remapped = SimpleNamespace(loaded=True)
        filtered = SimpleNamespace(loaded=True)
        window = SimpleNamespace(
            source=raw, remapped_source=remapped, preprocessed_source=filtered,
        )
        self.assertIs(QtAnalysisGUI._dual_stream_input_source(window), remapped)

        window.remapped_source = None
        self.assertIs(QtAnalysisGUI._dual_stream_input_source(window), raw)

    def test_completed_qc_remains_compatible_after_filtering(self):
        window = SimpleNamespace(
            _bad_channel_check_completed=True,
            _bad_channel_result_channel_ids=(1, 2, 5),
        )
        filtered_same_channels = SimpleNamespace(
            metadata=SimpleNamespace(channel_ids=(1, 2, 3, 4, 5)),
        )
        incompatible_channels = SimpleNamespace(
            metadata=SimpleNamespace(channel_ids=(101, 102, 103)),
        )
        matcher = QtAnalysisGUI._completed_qc_matches_source
        self.assertTrue(matcher(window, filtered_same_channels))
        self.assertFalse(matcher(window, incompatible_channels))

    def test_manual_override_filter_only_matches_overridden_channels(self):
        overrides = {7: "good", 12: "bad", 20: "noise"}
        matcher = QtAnalysisGUI._matches_bad_channel_review_filter
        self.assertTrue(matcher(7, "manual", overrides, "good"))
        self.assertTrue(matcher(12, "manual", overrides, "bad"))
        self.assertFalse(matcher(8, "manual", overrides, "bad"))
        self.assertTrue(matcher(8, "bad", overrides, "bad"))
        self.assertTrue(matcher(8, "all", overrides, "healthy"))

    def test_manual_review_advances_after_last_selected_row(self):
        channels = [296, 297, 298, 299, 300]
        next_id = QtAnalysisGUI._next_review_channel_id(channels, [1])
        self.assertEqual(next_id, 298)
        self.assertEqual(
            QtAnalysisGUI._next_review_channel_id(channels, [1, 2]), 299
        )
        self.assertIsNone(
            QtAnalysisGUI._next_review_channel_id(channels, [4])
        )


if __name__ == "__main__":
    unittest.main()
