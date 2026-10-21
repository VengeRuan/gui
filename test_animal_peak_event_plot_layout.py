"""Compact event-average frames must not change any computed waveform."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

import animal_peak_letter_vep as vep
import test_animal_peak_letter_vep as legacy


class CompactPlotTests(unittest.TestCase):
    def test_duration_scales_width_but_is_readable_and_never_exceeds_one_trial_cell(self):
        short=vep.compact_plot_size(1000,4000,1000,340)
        long=vep.compact_plot_size(2000,4000,1000,340)
        self.assertAlmostEqual(short[0],85+915/4)
        self.assertGreater(long[0],short[0]); self.assertEqual(short[1],340)
        self.assertEqual(vep.compact_plot_size(10,4000,1000,340)[0],260)
        self.assertEqual(vep.compact_plot_size(10000,4000,1000,340)[0],1000)
        self.assertEqual(vep.compact_plot_size(1000,4000,200,340)[0],200)

    def test_reference_is_display_only_uses_requested_trial_and_never_reads_signal(self):
        source,markers,matches,output=legacy.prepared()
        output["display_settings"]={"letter_before_ms":5000}
        with patch.object(source,"read",side_effect=AssertionError("layout must not read H5")):
            first,trial=vep.reference_span_ms(source,output,(7,4),1)
            second,other=vep.reference_span_ms(source,output,(7,4),2)
            fallback,missing=vep.reference_span_ms(source,output,(7,5))
        self.assertEqual((trial,other),(1,2)); self.assertGreater(second,first)
        self.assertGreater(fallback,0); self.assertIsNone(missing)

    def test_compact_export_preserves_full_calculated_time_axis_and_values(self):
        source,markers,matches,output=legacy.prepared()
        output=legacy.calculate(source,markers,matches,output,config=dict(end_ms=1000))
        before=output["letter_vep"]["cache"]["mean"].copy()
        sizes=[]
        def capture(figure,*args,**kwargs): sizes.append(tuple(figure.get_size_inches()))
        with tempfile.TemporaryDirectory() as directory,patch.object(Figure,"savefig",capture):
            vep.export_results(source,output,directory)
            with np.load(Path(directory)/"letter_start_vep/ch7_tag4_letter_start.npz") as data:
                np.testing.assert_array_equal(data["mean"],before)
                self.assertEqual(data["trials"].shape,(2,1001))
                np.testing.assert_array_equal(data["time_ms"][[0,-1]],[0,1000])
        self.assertTrue(sizes)
        self.assertTrue(all(2.8<=width<=6 and height==5 for width,height in sizes))


class CompactPlotQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        legacy.LetterVepQtTests.setUpClass(); cls.app=legacy.LetterVepQtTests.app

    def setUp(self):
        legacy.LetterVepQtTests.setUp(self)
        d=self.dialog
        d.resize(1360,960); d.parameter_toggle.setChecked(False); d.show(); self.app.processEvents()
        d.letter_view_spin.setValue(2000); d.apply_letter_display_range()
        d.letter_vep_duration.setValue(1000); d.refresh_letter_vep(); self.wait_worker()
        self.app.processEvents(); d._sync_event_plot_size()

    tearDown=legacy.LetterVepQtTests.tearDown
    wait_worker=legacy.LetterVepQtTests.wait_worker

    def test_plot_is_left_aligned_one_cell_and_does_not_recompute_on_resize(self):
        d=self.dialog; group=d.output["letter_vep"]["cache"]
        mean=group["mean"].copy(); waves=group["waves"].copy()
        self.assertEqual(d.letter_vep_plot.height(),340)
        self.assertLess(d.letter_vep_plot.width(),d.page_scroll.viewport().width()/2)
        self.assertEqual(d.letter_vep_plot.minimumWidth(),d.letter_vep_plot.maximumWidth())
        self.assertLessEqual(d.letter_vep_plot.centralWidget.geometry().right(),d.letter_vep_plot.width()+1)
        with patch.object(self.source,"read",side_effect=AssertionError("resize must not read signal")):
            d.resize(1100,880); self.app.processEvents(); d._sync_event_plot_size()
        self.assertIs(d.output["letter_vep"]["cache"],group)
        np.testing.assert_array_equal(group["mean"],mean); np.testing.assert_array_equal(group["waves"],waves)
        np.testing.assert_allclose(d.letter_vep_plot.viewRange()[0],[0,1000])
        self.assertEqual(d.letter_vep_plot.parentWidget(),d.letter_vep_page)
        self.assertLessEqual(d.letter_vep_plot.centralWidget.geometry().right(),d.letter_vep_plot.width()+1)

    def test_one_trial_reference_layout_and_pending_duration_use_committed_window(self):
        d=self.dialog; group=d.output["letter_vep"]["cache"]
        d.details_per_page_box.setCurrentIndex(1); self.app.processEvents()
        self.assertEqual(d.letter_vep_plot.height(),400)
        before=d.letter_vep_plot.width()
        d.letter_vep_duration.setValue(2000)
        self.app.processEvents()
        self.assertEqual(d.letter_vep_plot.width(),before)
        self.assertEqual(group["waves"].shape,(2,1001))
        d.refresh_letter_vep(); self.wait_worker(); d._sync_event_plot_size()
        self.assertGreater(d.letter_vep_plot.width(),before)
        self.assertEqual(d.output["letter_vep"]["cache"]["waves"].shape,(2,2001))
        self.assertEqual(d.output["settings"]["pre_ms"],50)

    def test_reference_zoom_changes_size_only_and_fixed_y_is_preserved(self):
        d=self.dialog; group=d.output["letter_vep"]["cache"]
        before=d.letter_vep_plot.width()
        axis=next(iter(d._manual_axes))
        axis.set_xlim(-500,500); d._sync_event_plot_size()
        self.assertGreaterEqual(d.letter_vep_plot.width(),before)
        d.y_min_spin.setValue(-20); d.y_max_spin.setValue(20); d.apply_fixed_y_axis()
        d._sync_event_plot_size()
        np.testing.assert_allclose(d.letter_vep_plot.viewRange()[1],[-20,20])
        self.assertIs(d.output["letter_vep"]["cache"],group)
        self.assertEqual(len(group["time_ms"]),1001)


if __name__=="__main__": unittest.main()
