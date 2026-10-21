"""Average Trial choices are explicit, scoped, reversible and export-consistent."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from animal_peak_analysis import AnalysisCancelled, analyze_animal_peaks, export_animal_peak_results
from animal_peak_manual import adjust_animal_peak, resize_animal_peak_windows
from animal_peak_similarity import calculate_similarity
from animal_peak_trial_selection import average_trial_ids, selected_average_group, set_average_trial_selection
from test_animal_peak_analysis import fixture
from test_animal_peak_localization import trend_fixture


class TrialSelectionTests(unittest.TestCase):
    def setUp(self):
        self.source, markers, matches, settings = fixture()
        self.output = analyze_animal_peaks(self.source, markers, matches, settings)

    def group(self, output=None, channel=7, branch="peak"):
        return next(g for g in (self.output if output is None else output)["groups"]
                    if (g["channel"],g["tag"],g["branch"]) == (channel,4,branch))

    def test_default_all_returns_original_group_without_recomputation(self):
        self.assertIsNone(average_trial_ids(self.output,7,4))
        group=self.group()
        self.assertIs(selected_average_group(self.output,group),group)
        self.assertIs(set_average_trial_selection(self.output,7,4,[1,2]),self.output)

    def test_subset_updates_both_branches_but_preserves_full_groups_and_other_channels(self):
        result=set_average_trial_selection(self.output,7,4,[2])
        self.assertEqual(average_trial_ids(result,7,4),(2,))
        self.assertIs(result["groups"],self.output["groups"])
        self.assertIs(result["records"],self.output["records"])
        for branch in ("peak","trough"):
            original=self.group(branch=branch)
            selected=selected_average_group(result,original)
            self.assertEqual([r["trial_id"] for r in selected["rows"]],[2])
            np.testing.assert_array_equal(selected["mean"],original["waves"][1])
            self.assertTrue(np.isnan(selected["sem"]).all())
            np.testing.assert_array_equal(selected["valid_count"],np.ones(selected["mean"].shape))
        other=self.group(channel=22)
        self.assertIs(selected_average_group(result,other),other)
        self.assertIsNone(average_trial_ids(self.output,7,4))

    def test_empty_does_not_fall_back_to_all_and_reset_restores_default(self):
        empty=set_average_trial_selection(self.output,7,4,[])
        self.assertEqual(average_trial_ids(empty,7,4),())
        group=selected_average_group(empty,self.group())
        self.assertEqual(group["waves"].shape,(0,self.group()["waves"].shape[1]))
        self.assertFalse(group["rows"])
        self.assertTrue(np.isnan(group["mean"]).all())
        self.assertTrue((group["valid_count"]==0).all())
        restored=set_average_trial_selection(empty,7,4,None)
        self.assertIsNone(average_trial_ids(restored,7,4))
        self.assertIs(selected_average_group(restored,self.group()),self.group())
        self.assertEqual(len(restored["average_selection_history"]),2)

    def test_invalid_trial_or_pair_does_not_change_selection(self):
        for channel,tag,ids in ((7,4,[999]),(7,5,[1]),(999,4,[1])):
            with self.assertRaises(ValueError):
                set_average_trial_selection(self.output,channel,tag,ids)
        self.assertNotIn("average_trial_selections",self.output)

    def test_manual_edit_and_resize_preserve_choice_and_use_new_committed_waves(self):
        selected=set_average_trial_selection(self.output,7,4,[1])
        edited=adjust_animal_peak(self.source,selected,7,4,1,"peak",-100)
        resized=resize_animal_peak_windows(self.source,edited,200,100)
        self.assertEqual(average_trial_ids(resized,7,4),(1,))
        result=selected_average_group(resized,self.group(resized))
        self.assertEqual(result["rows"][0]["alignment_origin"],"manual")
        self.assertEqual(result["waves"].shape,(1,301))
        np.testing.assert_array_equal(result["mean"],self.group(resized)["waves"][0])
        self.assertEqual(len(self.group(resized)["rows"]),2)

    def test_similarity_remains_all_trials_and_does_not_become_stale(self):
        source,markers,matches,settings=trend_fixture()
        scored=calculate_similarity(source,analyze_animal_peaks(source,markers,matches,settings))
        result=set_average_trial_selection(scored,7,4,[1])
        self.assertIs(result["judgment_similarity"],scored["judgment_similarity"])
        self.assertFalse(result["judgment_similarity"]["stale"])
        self.assertEqual(len(result["judgment_similarity"]["groups"][0]["trials"]),2)
        with self.assertRaises(AnalysisCancelled):
            selected_average_group(result,result["groups"][0],cancel=lambda:True)

    def test_export_uses_selected_averages_and_preserves_unselected_details(self):
        result=set_average_trial_selection(self.output,7,4,[2])
        with tempfile.TemporaryDirectory() as directory:
            folder=export_animal_peak_results(result,self.source,directory,individual_png=False)
            with np.load(folder/"ch7_tag4_peak.npz") as data:
                np.testing.assert_array_equal(data["trial_ids"],[2])
                np.testing.assert_array_equal(data["all_available_trial_ids"],[1,2])
                np.testing.assert_array_equal(data["mean"],self.group()["waves"][1])
            manifest=json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["average_trial_selections"][0]["trial_ids"],[2])
            with (folder/"trial_details.csv").open(encoding="utf-8-sig") as handle:
                rows=list(csv.DictReader(handle))
            first=next(r for r in rows if r["channel"]=="7" and r["trial_id"]=="1" and r["branch"]=="peak")
            second=next(r for r in rows if r["channel"]=="7" and r["trial_id"]=="2" and r["branch"]=="peak")
            self.assertEqual(first["average_selected"],"False")
            self.assertEqual(second["average_included"],"True")
            self.assertEqual(len(rows),len(self.output["records"]))
            with (folder/"ch7_tag4_peak_average.csv").open(encoding="utf-8-sig") as handle:
                self.assertTrue(all(row["trial_count"]=="1" for row in csv.DictReader(handle)))

    def test_export_empty_selection_saves_zero_trials_not_default_average(self):
        output=set_average_trial_selection(self.output,7,4,[])
        with tempfile.TemporaryDirectory() as directory:
            folder=export_animal_peak_results(output,self.source,directory,individual_png=False)
            with np.load(folder/"ch7_tag4_peak.npz") as data:
                self.assertEqual(data["trial_ids"].size,0)
                self.assertTrue(np.isnan(data["mean"]).all())


if __name__=="__main__":
    unittest.main()
