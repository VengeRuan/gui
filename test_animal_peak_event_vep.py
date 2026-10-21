"""Three event anchors, explicit baseline references, shared choices and exports."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

import animal_peak_letter_vep as vep
import test_animal_peak_letter_vep as legacy
from animal_peak_analysis import AnalysisCancelled, analyze_animal_peaks
from animal_peak_filter import apply_secondary_filter
from animal_peak_trial_selection import set_average_trial_selection
from test_animal_peak_analysis import fixture


def with_dots(matches):
    return {sample: {**event, "tag100_source":"csv", "tag100_status":"csv",
        "tag100_start_eeg_sec":sample/1000-.5+.0004, "tag100_dot_offset_sec":sample/1000-.5+.0004,
        "tag100_duration_sec":2., "tag100_csv_row":index+2}
        for index,(sample,event) in enumerate(matches.items())}


def prepared():
    source,markers,matches,settings=fixture()
    # Different Dot-, Letter-, and Animal-preceding means make wrong baseline
    # anchoring numerically detectable, not just a label check.
    source.data[500:1000,0]+=10
    source.data[1000:2000,0]+=20
    source.data[2000:3000,0]+=30
    matches=with_dots(matches)
    return source,markers,matches,analyze_animal_peaks(source,markers,matches,settings)


def run(source,markers,matches,output,mode,**kwargs):
    config={"alignment":mode, **({"start_ms":-2000,"end_ms":0} if mode=="animal" else {"start_ms":0,"end_ms":500})}
    config.update(kwargs.pop("config",{}))
    return vep.refresh(source,output,markers,matches,(7,4),config,**kwargs)


class EventVepTests(unittest.TestCase):
    def test_dot_uses_exact_csv_time_rounded_to_samples_and_its_own_baseline(self):
        source,markers,matches,output=prepared()
        revised=run(source,markers,matches,output,"dot")
        group=revised["letter_vep"]["cache"]; row=group["rows"][0]
        self.assertEqual(row["alignment_sample"],2750)
        self.assertAlmostEqual(row["alignment_event_sec"],2.7504)
        self.assertAlmostEqual(row["alignment_sec"],2.75)
        self.assertEqual(row["baseline_reference"],"Dot")
        self.assertEqual((row["baseline_first"],row["baseline_last"]),(500,1501))
        baseline=float(np.mean(source.data[500:1501,0],dtype=np.float64))
        np.testing.assert_array_equal(group["waves"][0],source.data[1500:2001,0]-baseline)
        self.assertNotAlmostEqual(baseline,float(np.mean(source.data[1000:2001,0],dtype=np.float64)))
        self.assertEqual(set(revised["letter_vep"]["configurations"]),{"dot"})

    def test_animal_reads_only_before_anchor_but_reuses_letter_baseline(self):
        source,markers,matches,output=prepared()
        group=run(source,markers,matches,output,"animal")["letter_vep"]["cache"]
        row=group["rows"][0]
        np.testing.assert_allclose(group["time_ms"][[0,-1]],[-2000,0])
        self.assertEqual((row["alignment_sample"],row["baseline_anchor_sample"]),(4250,3250))
        self.assertEqual(row["baseline_reference"],"LetterStart")
        self.assertEqual((row["baseline_first"],row["baseline_last"]),(1000,2001))
        baseline=float(np.mean(source.data[1000:2001,0],dtype=np.float64))
        np.testing.assert_array_equal(group["waves"][0],source.data[1000:3001,0]-baseline)
        self.assertNotAlmostEqual(baseline,float(np.mean(source.data[2000:3001,0],dtype=np.float64)))
        self.assertEqual(row["last"]-1,row["alignment_sample"]-1250)

    def test_defaults_direction_validation_and_missing_csv_never_use_txt(self):
        self.assertEqual(vep.parameters({"alignment":"animal"})["start_ms"],-6000)
        self.assertEqual(vep.parameters({"alignment":"animal"})["end_ms"],0)
        self.assertEqual(vep.parameters({"alignment":"dot"})["end_ms"],6000)
        with self.assertRaises(ValueError): vep.parameters({"alignment":"unknown"})
        with self.assertRaisesRegex(ValueError,"只看事件前"): vep.parameters({"alignment":"animal","end_ms":1})
        source,markers,matches,output=prepared()
        for replacement in ({}, {"tag100_source":"txt","tag100_status":"matched","tag100_start_eeg_sec":2.75},
                            {"tag100_source":"csv","tag100_status":"invalid_csv_dot","tag100_start_eeg_sec":2.75}):
            changed={key:{name:value for name,value in event.items() if not name.startswith("tag100_")} for key,event in matches.items()}
            changed[3250].update(replacement)
            # A TXT interval cannot substitute for the missing CSV timestamp.
            changed[3250].update(tag100_start_frame=1,tag100_end_frame=61)
            dot=run(source,markers,changed,output,"dot")["letter_vep"]["cache"]
            self.assertEqual(dot["records"][0]["reason_code"],"dot_time_missing")
            letter=run(source,markers,changed,output,"letter")["letter_vep"]["cache"]
            self.assertEqual(len(letter["selected_indices"]),2)

    def test_shared_choices_survive_missing_dot_and_do_not_change_peak_choices(self):
        source,markers,matches,output=prepared()
        output=set_average_trial_selection(output,7,4,[2])
        output=run(source,markers,matches,output,"letter",trial_ids=[1])
        output=run(source,markers,matches,output,"dot")
        self.assertEqual(vep.selected_ids(output["letter_vep"],(7,4)),(1,))
        self.assertEqual(output["letter_vep"]["cache"]["selected_indices"],[0])
        output=run(source,markers,matches,output,"animal",trial_ids=[2])
        output=run(source,markers,matches,output,"letter")
        self.assertEqual(output["letter_vep"]["cache"]["selected_indices"],[1])
        self.assertEqual(output["average_trial_selections"][0]["trial_ids"],[2])
        output=run(source,markers,matches,output,"dot",trial_ids=[])
        for mode in ("animal","letter"):
            output=run(source,markers,matches,output,mode)
            self.assertFalse(output["letter_vep"]["cache"]["selected_indices"])
        # Missing Dot suppresses only its current-mode waveform, not shared choice.
        output=run(source,markers,matches,output,"letter",trial_ids=[2])
        state=output["letter_vep"]
        changed={sample:dict(event) for sample,event in state["matches"].items()}
        changed[6250]["tag100_status"]="invalid_csv_dot"
        output={**output,"letter_vep":{**state,"matches":changed}}
        output=run(source,markers,matches,output,"dot")
        self.assertFalse(output["letter_vep"]["cache"]["selected_indices"])
        self.assertEqual(vep.selected_ids(output["letter_vep"],(7,4)),(2,))
        output=run(source,markers,matches,output,"animal")
        self.assertEqual(output["letter_vep"]["cache"]["selected_indices"],[1])

    def test_exports_all_refreshed_modes_with_latest_shared_selection_and_anchor_metadata(self):
        source,markers,matches,output=prepared()
        for mode in ("letter","dot","animal"):
            output=run(source,markers,matches,output,mode,trial_ids=[2])
        labels=[]
        def capture(figure,*args,**kwargs):
            for axis in figure.axes: labels.append((axis.get_title(),axis.get_xlabel()))
        with tempfile.TemporaryDirectory() as directory,patch.object(Figure,"savefig",capture):
            vep.export_results(source,output,directory)
            folder=Path(directory)/"letter_start_vep"
            self.assertEqual(len(list(folder.glob("*.npz"))),12)
            for mode,anchor,reference in (("letter",6250,"LetterStart"),("dot",5750,"Dot"),("animal",7250,"LetterStart")):
                with np.load(folder/f"ch7_tag4_{mode}_start.npz") as data:
                    self.assertEqual(str(data["alignment"]),mode)
                    self.assertEqual(str(data["baseline_reference"]),reference)
                    np.testing.assert_array_equal(data["selected_trial_ids"],[2])
                    self.assertEqual(data["alignment_samples"][1],anchor)
                    self.assertEqual(data["baseline_anchor_samples"][1],5750 if mode=="dot" else 6250)
            with (folder/"trial_details.csv").open(encoding="utf-8-sig") as handle: rows=list(csv.DictReader(handle))
            self.assertEqual({row["alignment"] for row in rows},{"letter","dot","animal"})
            self.assertEqual(len(rows),30)
        self.assertTrue(any("相对 AnimalStart" in label and "基线相对LetterStart" in title for title,label in labels))
        self.assertTrue(any("相对 Dot" in label and "基线相对Dot" in title for title,label in labels))
        manifest=vep.manifest(output["letter_vep"])
        self.assertEqual(set(manifest["configurations"]),{"letter","dot","animal"})

    def test_filter_refresh_baseline_anchor_and_cancellation_are_transactional(self):
        source,markers,matches,output=prepared()
        output=run(source,markers,matches,output,"dot",trial_ids=[2])
        output=run(source,markers,matches,output,"animal")
        with self.assertRaises(AnalysisCancelled): run(source,markers,matches,output,"letter",cancel=lambda:True)
        self.assertNotIn("letter",output["letter_vep"]["configurations"])
        with tempfile.TemporaryDirectory() as directory:
            filtered,revised=apply_secondary_filter(source,output,[7],dict(mode="highpass",low_hz=1),cache_dir=directory)
            self.assertTrue(revised["letter_vep"]["stale"])
            self.assertEqual(set(revised["letter_vep"]["configurations"]),{"dot","animal"})
            revised=run(filtered,markers,matches,revised,"animal")
            row=revised["letter_vep"]["cache"]["rows"][1]
            expected=np.mean(np.asarray(filtered.read(4000,5001,[0]),dtype=np.float32),dtype=np.float64)
            self.assertAlmostEqual(row["baseline_correction"],expected)
            self.assertEqual(vep.selected_ids(revised["letter_vep"],(7,4)),(2,))


class EventVepQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        legacy.LetterVepQtTests.setUpClass()
        cls.app=legacy.LetterVepQtTests.app

    def setUp(self):
        legacy.LetterVepQtTests.setUp(self)
        self.dialog.matches=with_dots(self.matches)

    tearDown=legacy.LetterVepQtTests.tearDown
    wait_worker=legacy.LetterVepQtTests.wait_worker

    def mode(self, name):
        d=self.dialog
        d.letter_vep_alignment.setCurrentIndex(d.letter_vep_alignment.findData(name))
        self.app.processEvents()

    def test_selector_direction_duration_baseline_reference_and_no_old_mode_plot(self):
        d=self.dialog
        self.assertEqual(d.letter_vep_alignment.count(),3)
        d.letter_vep_duration.setValue(2000); d.refresh_letter_vep(); self.wait_worker()
        self.mode("dot")
        self.assertEqual(len(d.letter_vep_plot.listDataItems()),0)
        self.assertEqual((d.letter_vep_start.value(),d.letter_vep_end.value()),(0,2000))
        self.assertIn("相对Dot",d.letter_vep_baseline.text())
        d.refresh_letter_vep(); self.wait_worker()
        self.assertEqual(d.output["letter_vep"]["cache"]["parameters"]["alignment"],"dot")
        self.mode("animal")
        self.assertEqual((d.letter_vep_start.value(),d.letter_vep_end.value()),(-2000,0))
        self.assertIn("相对LetterStart",d.letter_vep_baseline.text())
        d.letter_vep_duration.setValue(1500)
        self.assertEqual((d.letter_vep_start.value(),d.letter_vep_end.value()),(-1500,0))
        d.letter_vep_end.setValue(100); self.assertEqual(d.letter_vep_end.value(),0)
        d.refresh_letter_vep(); self.wait_worker()
        self.assertIn("AnimalStart",d.letter_vep_plot.getAxis("bottom").labelText)
        self.assertEqual(d.letter_vep_table.item(0,10).text(),"LetterStart")
        self.mode("letter")
        self.assertEqual((d.letter_vep_start.value(),d.letter_vep_end.value()),(0,1500))

    def test_trial_dialog_controls_and_selection_are_shared_across_modes(self):
        from animal_peak_dialog import AverageTrialSelectionDialog,QtWidgets
        d=self.dialog; d.letter_vep_duration.setValue(1000)
        d.refresh_letter_vep(); self.wait_worker()
        with patch.object(AverageTrialSelectionDialog,"exec",return_value=QtWidgets.QDialog.DialogCode.Accepted),patch.object(AverageTrialSelectionDialog,"selected_ids",return_value=[2]):
            d.choose_letter_vep_trials(); self.wait_worker()
        for mode in ("dot","animal","letter"):
            self.mode(mode); d.refresh_letter_vep(); self.wait_worker()
            group=d.output["letter_vep"]["cache"]
            self.assertEqual(group["selected_indices"],[1])
            self.assertEqual(vep.selected_ids(d.output["letter_vep"],(7,4)),(2,))
        self.assertNotIn("average_trial_selections",d.output)
        d._set_busy(True); self.assertFalse(d.letter_vep_alignment.isEnabled()); d._set_busy(False)

    def test_missing_dot_report_does_not_uncheck_shared_trial(self):
        d=self.dialog; d.matches[3250]["tag100_status"]="invalid_csv_dot"
        self.mode("dot"); d.letter_vep_duration.setValue(500); d.refresh_letter_vep(); self.wait_worker()
        self.assertIn("CSV Dot",d.letter_vep_table.item(0,7).text())
        self.assertEqual(d.letter_vep_table.item(0,2).text(),"是")
        self.assertEqual(d.letter_vep_table.item(0,3).text(),"否")
        self.mode("letter"); d.refresh_letter_vep(); self.wait_worker()
        self.assertEqual(d.letter_vep_table.item(0,3).text(),"是")


if __name__=="__main__": unittest.main()
