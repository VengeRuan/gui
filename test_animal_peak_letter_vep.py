"""LetterStart VEP numerical, independence, export, cancellation and Qt tests."""
import csv
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import animal_peak_letter_vep as vep
from animal_peak_analysis import AnalysisCancelled, analyze_animal_peaks, export_animal_peak_results
from animal_peak_filter import apply_secondary_filter
from animal_peak_manual import adjust_animal_peak, resize_animal_peak_windows
from animal_peak_trial_selection import set_average_trial_selection
from test_animal_peak_analysis import fixture


def prepared():
    source, markers, matches, settings = fixture()
    output = analyze_animal_peaks(source, markers, matches, settings)
    return source, markers, matches, output


def calculate(source, markers, matches, output, pair=(7,4), **kwargs):
    return vep.refresh(source, output, markers, matches, pair, kwargs.pop("config", {}), **kwargs)


class LetterVepTests(unittest.TestCase):
    def test_each_trial_uses_its_own_one_second_baseline_not_peak_baseline(self):
        source,markers,matches,settings=fixture()
        source.data[1000:1800,0]=10
        source.data[4000:4800,0]=20
        output=analyze_animal_peaks(source,markers,matches,settings)
        original_peak=output["groups"][0]["mean"].copy()
        revised=calculate(source,markers,matches,output)
        group=revised["letter_vep"]["cache"]
        np.testing.assert_allclose([row["baseline_correction"] for row in group["rows"]],[8000/1001,16000/1001])
        self.assertEqual(next(row for row in output["records"] if row["status"]=="included")["baseline_correction"],0)
        np.testing.assert_array_equal(revised["groups"][0]["mean"],original_peak)
        raw=calculate(source,markers,matches,revised,config=dict(baseline_enabled=False))["letter_vep"]["cache"]
        np.testing.assert_allclose(raw["waves"][0]-group["waves"][0],np.full(6001,8000/1001),rtol=1e-6)

    def test_defaults_exact_letter_samples_baseline_and_actual_channel_ids(self):
        source, markers, matches, output = prepared()
        original = [group["waves"].copy() for group in output["groups"]]
        result = calculate(source, markers, matches, output)
        group = result["letter_vep"]["cache"]
        self.assertEqual(group["parameters"], vep.DEFAULTS)
        self.assertEqual(group["waves"].shape, (2,6001))
        np.testing.assert_allclose(group["time_ms"][[0,-1]], [0,6000])
        for index, center in enumerate((2000,5000)):
            expected = source.data[center:center+6001,0]-np.mean(source.data[center-1000:center+1,0],dtype=np.float64)
            np.testing.assert_array_equal(group["waves"][index],expected)
        np.testing.assert_array_equal(group["mean"],np.mean(group["waves"],axis=0))
        other = calculate(source, markers, matches, result, pair=(22,4))["letter_vep"]["cache"]
        np.testing.assert_array_equal(other["waves"][0],source.data[2000:8001,1])
        for old,new in zip(original,result["groups"]): np.testing.assert_array_equal(old,new["waves"])
        self.assertNotIn("letter_vep",output)

    def test_negative_view_and_baseline_off_do_not_use_peak_epoch_crop(self):
        source, markers, matches, output = prepared()
        output = calculate(source,markers,matches,output,config=dict(start_ms=-1500,end_ms=2000,baseline_enabled=False))
        group=output["letter_vep"]["cache"]
        self.assertEqual(len(group["rows"]),2)
        np.testing.assert_array_equal(group["waves"][0],source.data[500:4001,0])
        self.assertTrue(all(row["baseline_correction"]==0 for row in group["rows"]))
        self.assertEqual(output["settings"]["baseline_start_ms"],-200)

    def test_animal_validity_is_required_but_peak_search_success_is_not(self):
        source,markers,matches,output=prepared()
        output=calculate(source,markers,matches,output,pair=(7,5))
        rows=output["letter_vep"]["cache"]["records"]
        self.assertEqual([row["trial_id"] for row in rows],[3,4,5])
        self.assertEqual([row["reason_code"] for row in rows],["behavior_unmatched","validity_not_y","animal_outside_epoch"])
        self.assertTrue(np.isnan(output["letter_vep"]["cache"]["mean"]).all())
        # Remove every located peak; Letter response remains available.
        output["groups"]=[]; output["records"]=[]
        result=calculate(source,markers,matches,output)
        self.assertEqual(len(result["letter_vep"]["cache"]["selected_indices"]),2)

    def test_incomplete_baseline_wave_and_nonfinite_samples_are_explicit(self):
        source,markers,matches,output=prepared()
        state=vep.new_state(markers,matches)
        group=vep.calculate_group(source,output,state,(7,4),dict(baseline_start_ms=-3000))
        self.assertEqual(group["records"][0]["reason_code"],"letter_baseline_outside_source")
        group=vep.calculate_group(source,output,state,(7,4),dict(end_ms=16000))
        self.assertEqual(group["records"][1]["reason_code"],"letter_window_outside_source")
        source.data[1000,0]=np.nan
        group=vep.calculate_group(source,output,state,(7,4))
        self.assertEqual(group["records"][0]["reason_code"],"baseline_nonfinite")
        source.data[1000,0]=0; source.data[2500,0]=np.inf
        group=vep.calculate_group(source,output,state,(7,4))
        self.assertEqual(group["records"][0]["reason_code"],"wave_nonfinite")

    def test_selection_empty_all_channel_tag_independence_and_manual_anchor_independence(self):
        source,markers,matches,output=prepared()
        output=set_average_trial_selection(output,7,4,[2])
        output=calculate(source,markers,matches,output,trial_ids=[1])
        self.assertEqual(vep.selected_ids(output["letter_vep"],(7,4)),(1,))
        self.assertEqual(output["average_trial_selections"][0]["trial_ids"],[2])
        self.assertIsNone(vep.selected_ids(output["letter_vep"],(22,4)))
        before=output["letter_vep"]["cache"]["mean"].copy()
        output=adjust_animal_peak(source,output,7,4,1,"peak",relative_ms=-100)
        output=resize_animal_peak_windows(source,output,100,100)
        output=calculate(source,markers,matches,output)
        np.testing.assert_array_equal(output["letter_vep"]["cache"]["mean"],before)
        output=calculate(source,markers,matches,output,trial_ids=[])
        self.assertEqual(vep.selected_ids(output["letter_vep"],(7,4)),())
        self.assertFalse(output["letter_vep"]["cache"]["selected_indices"])
        self.assertTrue(np.isnan(output["letter_vep"]["cache"]["mean"]).all())
        output=calculate(source,markers,matches,output,trial_ids=[1,2],config=dict(end_ms=100))
        self.assertIsNone(vep.selected_ids(output["letter_vep"],(7,4)))
        self.assertEqual(output["letter_vep"]["cache"]["waves"].shape,(2,101))
        self.assertEqual(output["average_trial_selections"][0]["trial_ids"],[2])
        with self.assertRaises(ValueError): calculate(source,markers,matches,output,trial_ids=[999])

    def test_secondary_filter_invalidation_refresh_and_restore_preserve_own_choices(self):
        source,markers,matches,output=prepared()
        output=calculate(source,markers,matches,output,trial_ids=[2])
        original=output["letter_vep"]["cache"]["waves"].copy()
        with tempfile.TemporaryDirectory() as directory:
            filtered,revised=apply_secondary_filter(source,output,[7],dict(mode="highpass",low_hz=1),cache_dir=directory)
            self.assertTrue(revised["letter_vep"]["stale"])
            self.assertIsNone(revised["letter_vep"]["cache"])
            revised=calculate(filtered,markers,matches,revised)
            group=revised["letter_vep"]["cache"]
            center=5000; baseline=np.mean(filtered.read(center-1000,center+1,[0]),dtype=np.float64)
            expected=np.asarray(filtered.read(center,center+6001,[0]),dtype=np.float32).reshape(-1)-float(baseline)
            np.testing.assert_array_equal(group["waves"][1],expected)
            self.assertEqual(vep.selected_ids(revised["letter_vep"],(7,4)),(2,))
            restored,revised=apply_secondary_filter(filtered,revised,[7],cache_dir=directory)
            revised=calculate(restored,markers,matches,revised)
            np.testing.assert_array_equal(revised["letter_vep"]["cache"]["waves"],original)

    def test_validation_memory_cancellation_and_no_full_record_read(self):
        source,markers,matches,output=prepared()
        for cfg in (dict(start_ms=5,end_ms=1),dict(end_ms=np.nan),dict(baseline_end_ms=100),dict(baseline_start_ms=0)):
            with self.assertRaises(ValueError): calculate(source,markers,matches,output,config=cfg)
        with self.assertRaises(AnalysisCancelled): calculate(source,markers,matches,output,cancel=lambda:True)
        checks=[0]
        def cancel_midway():
            checks[0]+=1
            return checks[0]>=3
        with self.assertRaises(AnalysisCancelled): calculate(source,markers,matches,output,cancel=cancel_midway)
        tiny={**output,"settings":{**output["settings"],"memory_budget_bytes":100}}
        with self.assertRaisesRegex(ValueError,"内存"): calculate(source,markers,matches,tiny)
        with patch.object(source,"read",wraps=source.read) as read:
            calculate(source,markers,matches,output)
            self.assertTrue(all(call.args[1]-call.args[0] < source.metadata.rows for call in read.call_args_list))
        self.assertNotIn("letter_vep",output)

    def test_signal_change_during_read_is_rejected_and_trend_is_never_averaged(self):
        source,markers,matches,settings=fixture()
        settings.update(localization_mode="trend",despike_window_ms=5,despike_mad_threshold=6,trend_window_ms=20)
        output=analyze_animal_peaks(source,markers,matches,settings)
        group=calculate(source,markers,matches,output)["letter_vep"]["cache"]
        np.testing.assert_array_equal(group["waves"][0],source.data[2000:8001,0])
        read=source.read
        original_data=source.data  # Keep the old array alive: IDs must not be recycled.
        def changing(*args,**kwargs):
            value=read(*args,**kwargs)
            source.data=source.data.copy()
            return value
        with patch.object(source,"read",side_effect=changing),self.assertRaisesRegex(ValueError,"数据源改变"):
            calculate(source,markers,matches,output)
        self.assertIsNot(source.data,original_data)

    def test_exports_all_groups_actual_channels_trial_ids_selected_mean_and_manifest(self):
        source,markers,matches,output=prepared()
        output=calculate(source,markers,matches,output,trial_ids=[2],config=dict(end_ms=100))
        group=output["letter_vep"]["cache"]
        with tempfile.TemporaryDirectory() as directory:
            vep.export_results(source,output,directory)
            folder=Path(directory)/"letter_start_vep"
            self.assertEqual(len(list(folder.glob("*.npz"))),4)
            with np.load(folder/"ch7_tag4_letter_start.npz") as data:
                np.testing.assert_array_equal(data["trial_ids"],[1,2])
                np.testing.assert_array_equal(data["selected_trial_ids"],[2])
                np.testing.assert_array_equal(data["mean"],group["mean"])
                np.testing.assert_array_equal(data["marker_samples"],[3250,6250])
                self.assertEqual(int(data["channel"]),7)
            with np.load(folder/"ch22_tag4_letter_start.npz") as data:
                np.testing.assert_array_equal(data["selected_trial_ids"],[1,2])
                self.assertEqual(int(data["channel"]),22)
            with (folder/"trial_details.csv").open(encoding="utf-8-sig") as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual(len(rows),10)
            self.assertEqual(sum(row["average_included"]=="True" for row in rows),3)
            self.assertGreater((folder/"all_letter_start_vep.pdf").stat().st_size,500)
            self.assertTrue((folder/"ch7_tag4_letter_start.png").is_file())
            # Main complete export includes the new section without touching originals.
            with patch.object(Figure,"savefig"):
                full=export_animal_peak_results(output,source,directory,individual_png=False)
            manifest=json.loads((full/"parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"],"complete")
            self.assertEqual(manifest["letter_start_vep"]["parameters"]["end_ms"],100)
            self.assertEqual(manifest["letter_start_vep"]["selections"][0]["trial_ids"],[2])
            self.assertTrue((full/"letter_start_vep"/"ch22_tag4_letter_start.npz").is_file())


class LetterVepQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qt_gui import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        from qt_gui import QtAnalysisGUI
        from animal_peak_dialog import AnimalPeakDialog
        with patch("qt_gui.OperationAuditStore"),patch("qt_gui.FeishuAuditConfig.from_environment",return_value=SimpleNamespace(is_configured=False)):
            self.window=QtAnalysisGUI("Letter VEP test")
        self.source,self.markers,self.matches,output=prepared()
        self.dialog=AnimalPeakDialog(self.window,self.source,self.markers,self.matches,output["settings"])
        self.dialog.analysis_finished(output)
        self.dialog.tabs.setCurrentWidget(self.dialog.letter_vep_page); self.app.processEvents()

    def tearDown(self):
        if self.dialog.worker is not None:
            self.dialog.worker.cancel(); self.dialog.worker.wait(15000)
        self.app.processEvents(); self.dialog.deleteLater(); self.window.deleteLater(); self.app.processEvents()

    def wait_worker(self):
        deadline=time.monotonic()+15
        while self.dialog._busy and time.monotonic()<deadline:
            self.app.processEvents(); time.sleep(.005)
        self.app.processEvents(); self.assertFalse(self.dialog._busy)

    def test_default_tab_refresh_fixed_y_and_selection_dialog_same_controls(self):
        from animal_peak_dialog import AverageTrialSelectionDialog
        d=self.dialog
        self.assertEqual(d.tabs.indexOf(d.letter_vep_page),6)
        self.assertEqual(d.letter_vep_end.value(),6000)
        self.assertEqual(d.letter_vep_base_start.value(),-1000)
        self.assertTrue(d.letter_vep_baseline.isChecked())
        self.assertFalse(d.letter_vep_plot.getAxis("bottom").autoSIPrefix)
        d.letter_vep_refresh.click(); self.wait_worker()
        group=d.output["letter_vep"]["cache"]
        np.testing.assert_array_equal(d.letter_vep_plot.listDataItems()[-1].yData,group["mean"])
        d.y_min_spin.setValue(-20); d.y_max_spin.setValue(20); d.apply_fixed_y_axis()
        np.testing.assert_allclose(d.letter_vep_plot.viewRange()[1],[-20,20])
        selection=AverageTrialSelectionDialog(d,d.output,(7,4),letter_group=group)
        selection.select_none_button.click(); self.assertEqual(selection.selected_ids(),[])
        selection.invert_button.click(); self.assertEqual(selection.selected_ids(),[1,2])
        selection.deleteLater()
        self.assertTrue(d._has_unsaved_edits())
        d._export_finished(Path("test-export")); self.assertFalse(d._has_unsaved_edits())

    def test_invalid_animal_tag_still_has_specific_table_and_peak_selector_unchanged(self):
        d=self.dialog
        d.result_tag_box.setCurrentIndex(d.result_tag_box.findData(5))
        d.refresh_letter_vep(); self.wait_worker()
        self.assertEqual(d.letter_vep_table.rowCount(),3)
        self.assertIn("无对应",d.letter_vep_table.item(0,7).text())
        self.assertIn("不是 Y",d.letter_vep_table.item(1,7).text())
        d.tabs.setCurrentIndex(0)
        self.assertEqual(d.result_tag_box.count(),1)
        self.assertEqual(d.result_tag_box.currentData(),4)

    def test_duration_control_is_next_to_refresh_and_updates_window_without_changing_selections(self):
        d=self.dialog
        self.assertEqual(d.letter_vep_duration.value(),6000)
        toolbar=d.letter_vep_page.layout().itemAt(1).layout()
        self.assertGreaterEqual(toolbar.indexOf(d.letter_vep_refresh),0)
        self.assertGreaterEqual(toolbar.indexOf(d.letter_vep_duration),0)
        d.output=calculate(self.source,self.markers,self.matches,d.output,trial_ids=[2])
        d._render_letter_vep()
        peak=d.output["groups"][0]["mean"].copy()
        d.letter_vep_duration.setValue(1200)
        self.assertEqual(d.letter_vep_end.value(),1200)
        self.assertEqual(d.output["letter_vep"]["cache"]["waves"].shape,(2,6001))
        self.assertIn("待刷新",d.letter_vep_status.text())
        d.letter_vep_refresh.click(); self.wait_worker()
        self.assertEqual(d.output["letter_vep"]["cache"]["waves"].shape,(2,1201))
        np.testing.assert_allclose(d.output["letter_vep"]["cache"]["time_ms"][[0,-1]],[0,1200])
        self.assertEqual(vep.selected_ids(d.output["letter_vep"],(7,4)),(2,))
        np.testing.assert_array_equal(d.output["groups"][0]["mean"],peak)

    def test_duration_syncs_both_directions_with_nonzero_start_and_stays_disabled_while_busy(self):
        d=self.dialog
        d.letter_vep_start.setValue(-500)
        self.assertEqual(d.letter_vep_duration.value(),6500)
        d.letter_vep_duration.setValue(2000)
        self.assertEqual(d.letter_vep_end.value(),1500)
        d.letter_vep_end.setValue(2500)
        self.assertEqual(d.letter_vep_duration.value(),3000)
        d._set_busy(True); self.assertFalse(d.letter_vep_duration.isEnabled())
        d._set_busy(False); self.assertTrue(d.letter_vep_duration.isEnabled())
        d.letter_vep_start.setValue(59000)
        self.assertEqual(d.letter_vep_duration.maximum(),1000)
        d.letter_vep_duration.setValue(1200)
        self.assertEqual(d.letter_vep_end.value(),60000)

    def test_own_selection_parameters_refresh_and_baseline_toggle_do_not_touch_peak_page(self):
        from animal_peak_dialog import AverageTrialSelectionDialog, QtWidgets
        d=self.dialog; d.refresh_letter_vep(); self.wait_worker()
        peak=d.output["groups"][0]["mean"].copy()
        with patch.object(AverageTrialSelectionDialog,"exec",return_value=QtWidgets.QDialog.DialogCode.Accepted),patch.object(AverageTrialSelectionDialog,"selected_ids",return_value=[2]):
            d.choose_letter_vep_trials(); self.wait_worker()
        self.assertEqual(vep.selected_ids(d.output["letter_vep"],(7,4)),(2,))
        self.assertNotIn("average_trial_selections",d.output)
        d.letter_vep_end.setValue(500); d.letter_vep_baseline.setChecked(False)
        self.assertIn("待刷新",d.letter_vep_status.text())
        self.assertFalse(d.letter_vep_base_start.isEnabled())
        d.refresh_letter_vep(); self.wait_worker()
        self.assertEqual(d.output["letter_vep"]["cache"]["waves"].shape,(2,501))
        self.assertEqual(vep.selected_ids(d.output["letter_vep"],(7,4)),(2,))
        np.testing.assert_array_equal(d.output["groups"][0]["mean"],peak)
        self.assertFalse(d._parameters_dirty)

    def test_worker_pre_cancel_does_not_commit_and_stale_export_is_blocked(self):
        from animal_peak_dialog import AnimalPeakWorker, QtWidgets
        d=self.dialog; d.refresh_letter_vep(); self.wait_worker()
        before=d.output
        worker=AnimalPeakWorker(self.source,self.markers,self.matches,{},letter=(before,(7,4),"unchanged"))
        values=[]; cancelled=[]; worker.completed.connect(values.append); worker.cancelled.connect(cancelled.append)
        worker.cancel(); worker.run()
        self.assertFalse(values); self.assertTrue(cancelled); self.assertIs(d.output,before)
        d.output=vep.invalidate(d.output,"test"); d._render_letter_vep()
        self.assertEqual(len(d.letter_vep_plot.listDataItems()),0)
        with patch.object(QtWidgets.QMessageBox,"warning") as warning,patch.object(QtWidgets.QFileDialog,"getExistingDirectory") as directory:
            d.start_export(); warning.assert_called_once(); directory.assert_not_called()


if __name__=="__main__":
    unittest.main()
