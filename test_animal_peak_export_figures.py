"""Complete-save omits peak/trough graphics, not their numeric audit data."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from animal_peak_analysis import analyze_animal_peaks, export_animal_peak_results
from animal_peak_letter_vep import refresh
from test_animal_peak_analysis import fixture
from test_animal_peak_dot_display import dots


class ExportFigureSelectionTests(unittest.TestCase):
    def prepare(self):
        source, markers, matches, settings = fixture()
        markers = markers[:2]; matches = dots(markers, matches)
        output = analyze_animal_peaks(source, markers, matches,
                                     {**settings, "channel_ids": [7], "localization_mode": "trend"})
        output["display_settings"] = {"trial_range_mode": "dot_animal", "letter_before_ms": 500}
        return source, markers, matches, output

    def export(self, source, output, directory, individual_png):
        files, pages = [], []
        def figure_info(fig):
            return [fig._suptitle.get_text() if fig._suptitle else "", *[a.get_title() for a in fig.axes]]
        def file_capture(fig, path, *args, **kwargs):
            files.append((Path(path).name, figure_info(fig)))
        def page_capture(pdf, fig=None, **kwargs):
            pages.append(figure_info(fig))
        with patch.object(Figure, "savefig", file_capture), patch.object(PdfPages, "savefig", page_capture), \
             patch.object(Axes, "hist", side_effect=AssertionError("latency histogram must not be drawn")):
            folder = export_animal_peak_results(output, source, directory, individual_png=individual_png)
        names = [name for name, _ in files]
        self.assertFalse(any("_summary." in name for name in names))
        titles = [text for _, ts in files for text in ts] + [text for ts in pages for text in ts]
        for title in titles:
            self.assertNotIn("最大峰 / 最小谷对齐平均", title)
            self.assertNotIn("保留原始峰时/谷时分布", title)
        return folder, names, pages

    def test_both_png_choices_keep_trial_reference_figures_and_peak_trough_data(self):
        for individual in (True, False):
            with self.subTest(individual=individual), tempfile.TemporaryDirectory() as directory:
                source, _, _, output = self.prepare()
                originals = [(g["waves"].copy(), g["mean"].copy()) for g in output["groups"]]
                folder, names, pages = self.export(source, output, directory, individual)
                self.assertIn("overview.png", names)
                self.assertIn("ch7_tag4_animalstart_reference.png", names)
                self.assertIn("ch7_tag4_animalstart_reference.pdf", names)
                self.assertEqual(any(name.endswith("_detail.png") for name in names), individual)
                self.assertTrue(any(any("逐 Trial 找波与截取详情" in t for t in ts) for ts in pages))
                for group, (waves, mean) in zip(output["groups"], originals):
                    name = f"ch7_tag4_{group['branch']}"
                    with np.load(folder/(name+".npz")) as data:
                        np.testing.assert_array_equal(data["trials"], waves)
                        np.testing.assert_array_equal(data["mean"], mean)
                    with (folder/(name+"_average.csv")).open(encoding="utf-8-sig") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), len(group["time_ms"]))
                    np.testing.assert_array_equal(group["waves"], waves)
                self.assertTrue((folder/"trial_details.csv").is_file())
                self.assertTrue(list(folder.glob("*_trial_traces_part*.npz")))
                manifest = json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "complete")
                self.assertTrue(manifest["figure_exports"]["peak_trough_numeric_data"])

    def test_all_refreshed_event_modes_remain_in_files_and_combined_pdf(self):
        source, markers, matches, output = self.prepare()
        for mode in ("letter", "dot", "animal"):
            cfg = {"alignment": mode, "start_ms": -500 if mode == "animal" else 0,
                   "end_ms": 0 if mode == "animal" else 500}
            output = refresh(source, output, markers, matches, (7, 4), cfg)
        with tempfile.TemporaryDirectory() as directory:
            folder, names, pages = self.export(source, output, directory, False)
            titles = [title for page in pages for title in page]
            for mode, display_name in (("letter", "LetterStart"), ("dot", "Dot"), ("animal", "AnimalStart")):
                self.assertTrue(any(display_name+"对齐 VEP" in title for title in titles), (mode, titles))
            self.assertGreaterEqual(len(list((folder/"letter_start_vep").glob("*.npz"))), 3)


if __name__ == "__main__":
    unittest.main()
