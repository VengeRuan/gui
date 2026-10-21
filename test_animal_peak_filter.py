"""Continuous filtering, reversibility, isolation, and bounded temporary IO."""
import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import signal
from matplotlib.figure import Figure

from animal_peak_analysis import AnalysisCancelled, analyze_animal_peaks, export_animal_peak_results, trial_detail_trace
from animal_peak_filter import apply_secondary_filter, filter_config, _finite_runs
from animal_peak_manual import adjust_animal_peak, resize_animal_peak_windows
from animal_peak_similarity import calculate_similarity
from animal_peak_trial_selection import set_average_trial_selection, average_trial_ids
from qt_data_model import ArraySource


def filter_fixture(trend=False):
    fs = 200
    t = np.arange(20000)/fs
    values = np.column_stack((3*np.sin(2*np.pi*t) + np.sin(2*np.pi*10*t) + .4*np.sin(2*np.pi*.2*t),
                              2*np.sin(2*np.pi*3*t) + np.sin(2*np.pi*20*t))).astype(np.float32)
    source = ArraySource(values, fs, time_offset=1.25, channel_ids=[7,22])
    markers = np.array([[2250,4],[5250,4],[9250,5]])
    matches = {int(sample): {"validity":"Y", "animal_latency_sec":1} for sample,_ in markers}
    settings = dict(channel_ids=[7,22], tags=[4,5], trial_limit=0, processing={}, epoch_start_ms=-500,
                    epoch_end_ms=2000, baseline_start_ms=-200, baseline_end_ms=0,
                    search_ms=300, search_after_ms=100, pre_ms=100, post_ms=150,
                    localization_mode="trend" if trend else "raw", despike_window_ms=0, trend_window_ms=25)
    return source, markers, matches, analyze_animal_peaks(source, markers, matches, settings)


class SecondaryFilterTests(unittest.TestCase):
    def test_streaming_matches_full_record_scipy_all_modes_and_preserves_channel_mapping(self):
        for mode in ("notch","highpass","lowpass","bandpass"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                base,_,_,output = filter_fixture()
                original = base.data.copy()
                cfg,sos = filter_config(dict(mode=mode,frequency_hz=1,q=5,low_hz=2,high_hz=30,order=3),200)
                source,result = apply_secondary_filter(base,output,[7],cfg,cache_dir=directory,chunk_samples=137)
                expected = signal.sosfiltfilt(sos,original[:,0])
                np.testing.assert_allclose(source.read(0,20000,[0])[:,0],expected,rtol=2e-6,atol=2e-6)
                np.testing.assert_array_equal(source.read(0,20000,1),original[:,1])
                ordered = source.read(17,121,[1,0,1],step=3)
                np.testing.assert_array_equal(ordered[:,0],original[17:121:3,1])
                np.testing.assert_allclose(ordered[:,1],expected[17:121:3],rtol=2e-6,atol=2e-6)
                np.testing.assert_array_equal(ordered[:,2],original[17:121:3,1])
                np.testing.assert_array_equal(base.data,original)
                self.assertEqual(result["secondary_filter_state"][0]["sample_count"],20000)
                self.assertTrue(all(not list(Path(directory).rglob("forward.dat")) for _ in [0]))
                del source
                gc.collect()
                self.assertEqual(list(Path(directory).iterdir()),[])

    def test_notch_reduces_one_hz_without_removing_ten_hz(self):
        base,_,_,output = filter_fixture()
        with tempfile.TemporaryDirectory() as directory:
            source,result = apply_secondary_filter(base,output,[7],dict(mode="notch",frequency_hz=1,q=5),cache_dir=directory,chunk_samples=233)
            values = source.read(5000,15000,0)
            t = np.arange(5000,15000)/200
            amplitude = lambda f: abs(2*np.mean(values*np.exp(-2j*np.pi*f*t)))
            self.assertLess(amplitude(1), .15)
            self.assertGreater(amplitude(10), .95)
            self.assertEqual(result["secondary_filter_state"][0]["channel"],7)

    def test_reapply_starts_from_loaded_source_and_other_filtered_channel_is_unchanged(self):
        base,_,_,output = filter_fixture()
        with tempfile.TemporaryDirectory() as directory:
            source,one = apply_secondary_filter(base,output,[7],dict(mode="highpass",low_hz=2),cache_dir=directory,chunk_samples=101)
            again,two = apply_secondary_filter(source,one,[7],dict(mode="highpass",low_hz=2),cache_dir=directory,chunk_samples=127)
            np.testing.assert_array_equal(source.read(0,20000,0),again.read(0,20000,0))
            latest,three = apply_secondary_filter(again,two,[22],dict(mode="lowpass",high_hz=12),cache_dir=directory)
            self.assertIs(latest.channels[7],again.channels[7])
            np.testing.assert_array_equal(latest.read(0,20000,0),again.read(0,20000,0))
            self.assertEqual(len(three["secondary_filter_state"]),2)
            reverted,four = apply_secondary_filter(latest,three,[22],None,cache_dir=directory)
            self.assertIn(7,reverted.channels)
            self.assertNotIn(22,reverted.channels)
            np.testing.assert_array_equal(reverted.read(0,20000,1),base.data[:,1])
            restored,five = apply_secondary_filter(reverted,four,[7],None,cache_dir=directory)
            self.assertIs(restored,base)
            self.assertEqual(five["secondary_filter_state"],[])

    def test_manual_points_trials_reference_and_baselines_refresh_and_restore(self):
        base,_,_,output = filter_fixture(trend=True)
        output = adjust_animal_peak(base,output,7,4,1,"peak",relative_ms=-100)
        output = set_average_trial_selection(output,7,4,[1])
        output = calculate_similarity(base,output)
        anchors = [(r["channel"],r["tag"],r["trial_id"],r["branch"],r["extreme_local_sample"]) for r in output["records"]]
        with tempfile.TemporaryDirectory() as directory:
            source,result = apply_secondary_filter(base,output,[7],dict(mode="highpass",low_hz=2),cache_dir=directory)
            self.assertTrue(result["judgment_similarity"]["stale"])
            self.assertEqual(result["manual_alignment_history"],output["manual_alignment_history"])
            self.assertEqual(average_trial_ids(result,7,4),(1,))
            self.assertEqual(anchors,[(r["channel"],r["tag"],r["trial_id"],r["branch"],r["extreme_local_sample"]) for r in result["records"]])
            for group in result["groups"]+result["animal_reference_groups"]:
                for row,wave in zip(group["rows"],group["waves"]):
                    np.testing.assert_array_equal(wave, source.read(row["excerpt_first"],row["excerpt_last"],row["column"])-row["baseline_correction"])
                    if row["channel"]==7:
                        center = row["marker_sample"]-250
                        self.assertEqual(row["baseline_correction"],float(np.mean(source.read(center-40,center+1,0))))
            for old in output["groups"]:
                if old["channel"]==22:
                    new = next(g for g in result["groups"] if g["channel"]==22 and g["tag"]==old["tag"] and g["branch"]==old["branch"])
                    self.assertIs(new["waves"],old["waves"])
            result = adjust_animal_peak(source,result,7,4,1,"peak",relative_ms=-50)
            result = resize_animal_peak_windows(source,result,125,175)
            restored,recovery = apply_secondary_filter(source,result,[7],None,cache_dir=directory)
            self.assertIs(restored,base)
            row = next(r for r in recovery["records"] if r["channel"]==7 and r["trial_id"]==1 and r["branch"]=="peak")
            self.assertEqual(row["extreme_relative_ms"],-50)
            self.assertEqual(row["alignment_origin"],"manual")
            self.assertEqual(average_trial_ids(recovery,7,4),(1,))
            for group in recovery["groups"]:
                if group["channel"]==7:
                    for row,wave in zip(group["rows"],group["waves"]):
                        np.testing.assert_array_equal(wave,base.data[row["excerpt_first"]:row["excerpt_last"],0]-row["baseline_correction"])

    def test_nan_runs_are_preserved_and_short_islands_are_reported(self):
        for chunk in (1,3,7,10):
            values = np.array([np.nan,1,2,3,np.nan,4,np.nan,5,6,7,8,np.nan])
            self.assertEqual(list(_finite_runs(values,chunk,None)),[(1,4),(5,6),(7,11)])
        base,_,_,output = filter_fixture()
        base.data[15000:15010,0]=np.nan
        base.data[15012:15020,0]=np.nan  # Two-sample finite island: left unchanged explicitly.
        with tempfile.TemporaryDirectory() as directory:
            source,result = apply_secondary_filter(base,output,[7],dict(mode="lowpass",high_hz=15),cache_dir=directory,chunk_samples=101)
            values=source.read(0,20000,0)
            np.testing.assert_array_equal(np.isnan(values),np.isnan(base.data[:,0]))
            np.testing.assert_array_equal(values[15010:15012],base.data[15010:15012,0])
            self.assertEqual(result["secondary_filter_state"][0]["short_finite_spans_unchanged"],1)

    def test_cancel_and_rebuild_failure_keep_previous_result_and_clean_new_files(self):
        base,_,_,output=filter_fixture()
        with tempfile.TemporaryDirectory() as directory:
            stop=False
            def progress(value,text):
                nonlocal stop
                stop=value>1
            with self.assertRaises(AnalysisCancelled):
                apply_secondary_filter(base,output,[7],dict(mode="highpass",low_hz=1),progress,lambda:stop,cache_dir=directory,chunk_samples=101)
            self.assertEqual(list(Path(directory).iterdir()),[])
            source,result=apply_secondary_filter(base,output,[7],dict(mode="highpass",low_hz=1),cache_dir=directory)
            path=source.channels[7].path
            committed=source.read(0,100,0).copy()
            with patch("animal_peak_filter._refresh_at_anchors",side_effect=ValueError("synthetic failure")), self.assertRaises(ValueError):
                apply_secondary_filter(source,result,[7],dict(mode="highpass",low_hz=2),cache_dir=directory)
            self.assertTrue(path.is_file())
            self.assertEqual(len(list(Path(directory).iterdir())),1)
            np.testing.assert_array_equal(source.read(0,100,0),committed)

    def test_reads_are_single_channel_chunks_and_export_contains_filtered_values_and_history(self):
        base,_,_,output=filter_fixture()
        read=base.read
        calls=[]
        def bounded(first,last,columns,step=1):
            calls.append((last-first,len(np.atleast_1d(columns))))
            return read(first,last,columns,step)
        base.read=bounded
        with tempfile.TemporaryDirectory() as directory:
            source,result=apply_secondary_filter(base,output,[7],dict(mode="notch",frequency_hz=1,q=5),cache_dir=directory,chunk_samples=123)
            self.assertLessEqual(max(size for size,_ in calls),123)
            self.assertTrue(all(columns==1 for _,columns in calls))
            with patch.object(Figure,"savefig",lambda *_a,**_kw:None):
                folder=export_animal_peak_results(result,source,directory,individual_png=False)
            manifest=json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["secondary_filter_state"][0]["frequency_hz"],1)
            self.assertEqual(len(manifest["secondary_filter_history"]),1)
            group=next(g for g in result["groups"] if g["channel"]==7 and g["tag"]==4 and g["branch"]=="peak")
            with np.load(folder/"ch7_tag4_peak.npz") as saved:
                np.testing.assert_array_equal(saved["trials"],group["waves"])
            rows=[r for r in result["records"] if r["channel"]==7 and r["trial_id"]==1]
            _,trace=trial_detail_trace(source,result,rows)
            self.assertTrue(np.isfinite(trace).all())

    def test_invalid_parameters_or_unselected_channel_do_not_create_caches(self):
        base,_,_,output=filter_fixture()
        with tempfile.TemporaryDirectory() as directory:
            for channels,cfg in (([99],dict(mode="notch")),([7],dict(mode="notch",frequency_hz=100)),
                                  ([7],dict(mode="bandpass",low_hz=5,high_hz=1)),([7],dict(mode="highpass",low_hz=-1))):
                with self.assertRaises(ValueError):
                    apply_secondary_filter(base,output,channels,cfg,cache_dir=directory)
            self.assertEqual(list(Path(directory).iterdir()),[])


if __name__=="__main__":
    unittest.main()
