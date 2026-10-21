"""CSV same-row Dot is a display-only 2 s cue; TXT Tag100 is never used."""
import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qt_data_model import ArraySource, build_task_markers, match_behavior_events_to_markers
from animal_peak_analysis import analyze_animal_peaks, trial_detail_trace, export_animal_peak_results
from animal_peak_plotting import draw_trial_detail, draw_tag100_interval
from animal_peak_manual import adjust_animal_peak, resize_animal_peak_windows
from animal_peak_filter import apply_secondary_filter


def csv_markers(rows, fs=1000, delay=0):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"events.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"column_{index}" for index in range(15)])
            for symbol, dot, letter in rows:
                row = [""]*15
                row[1], row[12], row[13] = symbol, dot, letter
                writer.writerow(row)
        annotations = {}
        markers = build_task_markers(path, {"stim_delay_sec": delay}, fs, "Letter",
                                     letter_annotations=annotations)
        original = build_task_markers(path, {"stim_delay_sec": delay}, fs, "Letter")
        np.testing.assert_array_equal(markers, original)
    return markers, annotations


def annotation_fixture():
    events = []
    markers = []
    for index, tag in enumerate((4,5,6,7)):
        start = 151 + index*150
        for kind, a, b, animal, validity in ((100,start-30,start-12,None,"NA"),
                                            (0,start-10,start-2,None,"NA"),
                                            (tag,start,start+90,start+30,"Y")):
            events.append(dict(tag=kind,start_frame=a,end_frame=b,animal_start_frame=animal,
                               validity=validity,detection_line_number=len(events)+3))
        markers.append((5000+index*5000,tag))
    markers=np.asarray(markers)
    _,annotations=csv_markers([(symbol,2+index*5,5+index*5) for index,symbol in enumerate(("|","闪烁光点","o","-"))])
    matches=match_behavior_events_to_markers(markers,1000,30,np.nan,np.nan,events,letter_annotations=annotations)
    t=np.arange(30000)/1000
    source=ArraySource(np.column_stack((np.sin(2*np.pi*3*t),np.cos(2*np.pi*5*t))).astype(np.float32),
                       1000,time_offset=1.25,channel_ids=[7,22])
    settings=dict(search_ms=300,search_after_ms=50,pre_ms=50,post_ms=50,epoch_start_ms=-500,
                  epoch_end_ms=2000,baseline_start_ms=-200,baseline_end_ms=0,channel_ids=[7,22],
                  tags=[4,5,6,7],trial_limit=0,processing={})
    return source,markers,events,matches,settings


class Tag100AnnotationTests(unittest.TestCase):
    def test_csv_m_start_fixed_two_seconds_and_same_alignment_offset(self):
        markers,annotations=csv_markers([("o",2.0725,5.3121)], fs=6490, delay=1.12345)
        sample=next(int(sample) for sample,tag in markers if tag==6)
        cue=annotations[sample]
        self.assertEqual(cue["tag100_source"],"csv")
        self.assertEqual(cue["tag100_csv_row"],2)
        self.assertAlmostEqual(cue["tag100_start_eeg_sec"],3.19595)
        self.assertAlmostEqual(cue["tag100_end_eeg_sec"],5.19595)
        self.assertAlmostEqual(cue["tag100_start_latency_sec"],3.19595-sample/6490)
        self.assertAlmostEqual(cue["tag100_end_latency_sec"]-cue["tag100_start_latency_sec"],2)

    def test_same_row_not_nearest_dot_and_repeated_tags_keep_row_identity(self):
        _,annotations=csv_markers([("o",2,10),("o",8,6),("|",30,35)])
        self.assertEqual(annotations[10000]["tag100_dot_offset_sec"],2)
        self.assertEqual(annotations[6000]["tag100_dot_offset_sec"],8)
        self.assertEqual(annotations[6000]["tag100_csv_row"],3)
        self.assertEqual(annotations[35000]["tag100_csv_row"],4)
        _,duplicates=csv_markers([("o",1,5),("|",2,5)])
        self.assertEqual(duplicates[5000]["tag100_status"],"ambiguous_csv_row")

    def test_missing_or_invalid_csv_dot_does_not_exclude_or_change_any_trial(self):
        source,markers,events,matches,settings=annotation_fixture()
        original=analyze_animal_peaks(source,markers,matches,settings)
        for bad in ("", "bad", -1, "inf"):
            _,annotations=csv_markers([(symbol,bad,5+index*5) for index,symbol in enumerate(("|","闪烁光点","o","-"))])
            revised=match_behavior_events_to_markers(markers,1000,30,np.nan,np.nan,events,letter_annotations=annotations)
            for event in revised.values():
                self.assertEqual(event["tag100_status"],"invalid_csv_dot")
                self.assertIn("CSV Dot 时间缺失或无效",event["tag100_message"])
                self.assertNotIn("tag100_start_latency_sec",event)
                self.assertNotIn("tag100_start_frame",event)
            output=analyze_animal_peaks(source,markers,revised,settings)
            self.assertEqual(len(output["eligible_trials"]),4)
            for before,after in zip(original["groups"],output["groups"]):
                for key in ("waves","mean","sem","valid_count"):
                    np.testing.assert_array_equal(before[key],after[key])
            for before,after in zip(original["records"],output["records"]):
                self.assertEqual(before["status"],after["status"])
                self.assertEqual(before["reason"],after["reason"])
                self.assertEqual(before["extreme_local_sample"],after["extreme_local_sample"])
            rows=[row for row in output["records"] if row.get("channel")==7 and row["trial_id"]==1]
            times,values=trial_detail_trace(source,output,rows)
            axis=Figure().subplots(); draw_trial_detail(axis,output,rows,times,values)
            self.assertIn("CSV Dot 时间缺失或无效",axis.get_title())
            self.assertFalse(any(region.get_gid()=="tag100_interval" for region in axis.patches))

    def test_hint_metadata_does_not_change_existing_marker_or_animal_matching(self):
        _,markers,events,with_cues,_=annotation_fixture()
        without=match_behavior_events_to_markers(markers,1000,30,np.nan,np.nan,
                                               [event for event in events if event["tag"]!=100])
        self.assertEqual(set(with_cues),set(without))
        for sample in without:
            self.assertFalse(any(key.startswith("tag100_") for key in without[sample]))
            self.assertEqual({k:v for k,v in with_cues[sample].items() if not k.startswith("tag100_")},
                             {k:v for k,v in without[sample].items() if not k.startswith("tag100_")})

    def test_txt_tag100_changes_or_removal_never_change_csv_annotation(self):
        _,markers,events,matches,_=annotation_fixture()
        annotations={sample:{key:value for key,value in event.items() if key.startswith("tag100_")} for sample,event in matches.items()}
        # Grossly wrong TXT cue boundaries must be irrelevant.
        changed=[{**event,"start_frame":9999,"end_frame":-5} if event["tag"]==100 else event for event in events]
        without=[event for event in events if event["tag"]!=100]
        for alternate in (changed,without):
            actual=match_behavior_events_to_markers(markers,1000,30,np.nan,np.nan,alternate,letter_annotations=annotations)
            for sample in matches:
                self.assertEqual(actual[sample],matches[sample])
        event=matches[10000]
        axis=Figure().subplots(); axis.set_xlim(-4500,0)
        region=draw_tag100_interval(axis,event,axis.get_xlim(),origin_sec=event["animal_latency_sec"],time_scale=1000)
        np.testing.assert_allclose(region._tag100_interval,[-4000,-2000])

    def test_peak_shading_uses_animalstart_ms_without_altering_curves_or_limits(self):
        source,markers,_,matches,settings=annotation_fixture()
        output=analyze_animal_peaks(source,markers[:1],matches,settings)
        output["display_settings"]={"letter_before_ms":3500}
        rows=[row for row in output["records"] if row.get("channel")==7]
        times,values=trial_detail_trace(source,output,rows)
        axis=Figure().subplots(); draw_trial_detail(axis,output,rows,times,values)
        region=next(region for region in axis.patches if region.get_gid()=="tag100_interval")
        self.assertAlmostEqual(region.get_x(),-4000)
        self.assertAlmostEqual(region.get_width(),2000)
        self.assertEqual(region.get_alpha(),.18)
        self.assertEqual(region.get_facecolor()[:3],(0,0,0))
        self.assertLess(region.get_zorder(),axis.lines[0].get_zorder())
        np.testing.assert_array_equal(axis.lines[0].get_ydata(),values)
        self.assertIsNone(axis.get_legend())
        no_cue={**rows[0],"tag100_status":"missing"}
        other=Figure().subplots(); draw_trial_detail(other,output,[no_cue,*rows[1:]],times,values)
        np.testing.assert_array_equal(axis.get_xlim(),other.get_xlim())
        np.testing.assert_array_equal(axis.get_ylim(),other.get_ylim())

    def test_clipped_or_hidden_annotation_never_expands_view(self):
        event={"tag100_source":"csv","tag100_status":"csv","tag100_start_latency_sec":-2.4,"tag100_end_latency_sec":-.4}
        axis=Figure().subplots(); axis.plot([-.5,1],[2,3]); axis.set_xlim(-.5,1)
        region=draw_tag100_interval(axis,event,axis.get_xlim())
        self.assertAlmostEqual(region.get_x(),-.5)
        self.assertAlmostEqual(region.get_width(),.1)
        np.testing.assert_allclose(axis.get_xlim(),[-.5,1])
        self.assertIsNone(draw_tag100_interval(axis,event,(0,1)))
        self.assertIsNone(draw_tag100_interval(axis,{**event,"tag100_status":"invalid_interval"},(-2,1)))
        self.assertIsNone(draw_tag100_interval(axis,{},(-2,1)))
        self.assertIsNone(draw_tag100_interval(axis,{**event,"tag100_status":"matched"},(-2,1)))  # No legacy TXT fallback.

    def test_export_figures_and_csv_keep_same_row_dot_times_and_source(self):
        source,markers,_,matches,settings=annotation_fixture()
        output=analyze_animal_peaks(source,markers[:1],matches,{**settings,"channel_ids":[7]})
        output["display_settings"]={"letter_before_ms":3500}
        regions=[]
        def capture(figure,*_a,**_kw):
            for axis in figure.axes:
                regions.extend(region._tag100_interval for region in axis.patches if region.get_gid()=="tag100_interval")
        with tempfile.TemporaryDirectory() as directory, patch.object(Figure,"savefig",capture):
            folder=export_animal_peak_results(output,source,directory)
            with (folder/"trial_details.csv").open(encoding="utf-8-sig") as handle:
                rows=list(csv.DictReader(handle))
            self.assertTrue(all(row["tag100_dot_offset_sec"]=="2.0" and row["tag100_duration_sec"]=="2.0" for row in rows))
            self.assertTrue(all(row["tag100_csv_row"]=="2" and row["tag100_source"]=="csv" for row in rows))
            self.assertTrue(all("tag100_start_frame" not in row for row in rows))
            manifest=json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
            self.assertIn("tag100_annotation",manifest)
        self.assertTrue(regions)
        np.testing.assert_allclose(regions,[[-4000,-2000]]*len(regions))

    def test_metadata_survives_manual_resize_and_secondary_filter(self):
        source,markers,_,matches,settings=annotation_fixture()
        output=analyze_animal_peaks(source,markers[:1],matches,{**settings,"channel_ids":[7]})
        output=adjust_animal_peak(source,output,7,4,1,"peak",relative_ms=-100)
        output=resize_animal_peak_windows(source,output,100,150)
        with tempfile.TemporaryDirectory() as directory:
            filtered,output=apply_secondary_filter(source,output,[7],{"mode":"highpass","low_hz":1},cache_dir=directory)
            for row in output["records"]+output["animal_reference_records"]:
                self.assertEqual(row["tag100_csv_row"],2)
                self.assertAlmostEqual(row["tag100_start_latency_sec"],-3)


class Tag100QtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qt_gui import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        from qt_gui import QtAnalysisGUI
        with patch("qt_gui.OperationAuditStore"),patch("qt_gui.FeishuAuditConfig.from_environment",return_value=SimpleNamespace(is_configured=False)):
            self.window=QtAnalysisGUI("Tag100 UI test")
        self.source,self.markers,_,self.matches,self.settings=annotation_fixture()

    def tearDown(self):
        self.window.deleteLater(); self.app.processEvents()

    def test_peak_viewer_and_export_renderer_show_same_region_and_audit_columns(self):
        from animal_peak_dialog import AnimalPeakDialog
        dialog=AnimalPeakDialog(self.window,self.source,self.markers,self.matches,self.settings)
        output=analyze_animal_peaks(self.source,self.markers[:1],self.matches,{**self.settings,"channel_ids":[7]})
        dialog.analysis_finished(output)
        dialog.letter_view_spin.setValue(3500); dialog.letter_view_button.click()
        dialog.tabs.setCurrentIndex(2); self.app.processEvents()
        rows=[row for row in output["records"] if row.get("channel")==7]
        times,values=trial_detail_trace(self.source,output,rows)
        exported=Figure().subplots(); draw_trial_detail(exported,output,rows,times,values)
        for axis in (dialog.detail_axes[0],exported):
            region=next(region for region in axis.patches if region.get_gid()=="tag100_interval")
            np.testing.assert_allclose([region.get_x(),region.get_width()],[-4000,2000])
        self.assertEqual(dialog.table.item(0,26).text(),"2.0")
        self.assertEqual(dialog.table.item(0,27).text(),"2.0")
        self.assertEqual(dialog.table.item(0,31).text(),"2")
        dialog.deleteLater(); self.app.processEvents()

    def test_regular_letter_trial_grid_and_its_png_pdf_include_hint_without_new_reads(self):
        self.window.lfp_source=self.source
        self.window.active_source=self.source
        output={"channel_ids":[7],"columns":np.asarray([0]),"fs":1000}
        result={"tag":4,"t_ms":np.arange(-3500,2001),"used_marker_samples":[5000]}
        self.window._show_letter_behavior_trial_grid("lfp",output,result,0,self.matches)
        self.app.processEvents()
        plot=self.window._letter_trial_window._letter_trial_plots[0]
        region=next(item for item in plot.plotItem.items if hasattr(item,"_tag100_interval"))
        np.testing.assert_allclose(region.getRegion(),[-3,-1])
        np.testing.assert_allclose(region._tag100_interval,[-3,-1])
        self.assertLess(region.zValue(),0)
        before=plot.plotItem.listDataItems()[0].getData()[1].copy()
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("png","pdf"):
                path=Path(directory)/f"letter_trial.{suffix}"
                plot.save_snapshot(path)
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size,500)
        self.window._letter_trial_window._letter_trial_render(); self.app.processEvents()
        np.testing.assert_array_equal(plot.plotItem.listDataItems()[0].getData()[1],before)
        self.assertEqual(len([item for item in plot.plotItem.items if hasattr(item,"_tag100_interval")]),1)
        self.window._letter_trial_window.close()

    def test_regular_letter_without_txt_still_uses_csv_and_timing_table_has_csv_audit(self):
        from qt_gui import QtWidgets
        self.window.lfp_source=self.source; self.window.active_source=self.source
        self.window.stim_markers=self.markers
        self.window.letter_csv_annotations={sample:{key:value for key,value in event.items() if key.startswith("tag100_")}
                                            for sample,event in self.matches.items()}
        output={"channel_ids":[7],"columns":np.asarray([0]),"fs":1000}
        result={"tag":4,"t_ms":np.arange(-3500,2001),"used_marker_samples":[5000]}
        self.window._show_letter_behavior_trial_grid("lfp",output,result,0,{})
        plot=self.window._letter_trial_window._letter_trial_plots[0]
        region=next(item for item in plot.plotItem.items if hasattr(item,"_tag100_interval"))
        np.testing.assert_allclose(region._tag100_interval,[-3,-1])
        self.window._show_letter_behavior_timing_table(self.window,output,{},30)
        table=self.window._letter_timing_window.findChild(QtWidgets.QTableWidget)
        self.assertEqual(table.item(0,11).text(),"2.0")
        self.assertEqual(table.item(0,12).text(),"2.0")
        self.assertEqual(table.item(0,16).text(),"2")
        self.assertIn("CSV",table.horizontalHeaderItem(16).text())
        self.window._letter_trial_window.close(); self.window._letter_timing_window.close()

    def test_real_alignment_passes_csv_to_matches_warns_nonfatally_and_clears_on_flash(self):
        from qt_gui import QMessageBox
        window=self.window; window.active_source=self.source
        window._bad_channel_check_completed=True
        window.good_channel_ids={7,22}; window.bad_channel_ids=set()
        window.alignment_selected_ids={7}; window._alignment_use_all_healthy=False
        window.bin_timing_metadata={"deltaT1_sec":0}
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            log=directory/"log.txt"; log.write_text("12:00:00.000\nRecording ON\n12:00:00.000 PROGRAM_START\n",encoding="utf-8")
            csv_path=directory/"events.csv"
            with csv_path.open("w",encoding="utf-8-sig",newline="") as handle:
                writer=csv.writer(handle); writer.writerow([f"c{i}" for i in range(15)])
                for symbol,dot,letter in (("o",2.0725,5.3121),("|","",10)):
                    row=[""]*15; row[1],row[12],row[13]=symbol,dot,letter; writer.writerow(row)
            txt=directory/"video.txt"
            txt.write_text("FrameRate: 30\n100,1,60,NA,NA\n6,151,241,181,Y\n100,260,290,NA,NA\n4,301,391,331,Y\n",encoding="utf-8")
            window.alignment_task_mode_var.setCurrentText("Letter")
            window.alignment_log_edit.setText(str(log)); window.alignment_event_edit.setText(str(csv_path))
            window.alignment_detection_edit.setText(str(txt))
            with patch.object(window,"refresh_alignment_channel_preview"),patch.object(window,"_inherit_alignment_selection_for_lfp"),patch.object(window,"_update_file_banners"),patch.object(QMessageBox,"warning") as warning,patch.object(QMessageBox,"critical") as critical:
                window.run_alignment()
                critical.assert_not_called(); warning.assert_called_once()
                self.assertIn("CSV第3行",warning.call_args.args[2])
                self.assertIn("对齐完成",window.alignment_status.text())
                self.assertEqual(set(window.behavior_matches),{5312,10000})
                self.assertAlmostEqual(window.behavior_matches[5312]["tag100_start_eeg_sec"],2.0725)
                self.assertEqual(window.behavior_matches[10000]["tag100_status"],"invalid_csv_dot")
                self.assertNotIn("tag100_start_latency_sec",window.behavior_matches[10000])
                # TXT still supplies AnimalStart, not Dot; no synthetic matched trials.
                self.assertEqual(window.behavior_matches[10000]["animal_start_frame"],331)
                self.assertEqual(window._task_marker_source["selected_channel_ids"],[7])
                flash=directory/"flash.csv"
                flash.write_text("a,b,c,d,e,f\n0,10,0,1,0,500\n",encoding="utf-8")
                window.alignment_task_mode_var.setCurrentText("Flash")
                window.alignment_event_edit.setText(str(flash)); window.run_alignment()
                critical.assert_not_called()
                self.assertEqual(window.letter_csv_annotations,{})
                self.assertFalse(hasattr(window,"behavior_matches"))


if __name__=="__main__":
    unittest.main()
