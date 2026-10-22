"""Window-level MCP parity and safety checks using synthetic EEG only."""
import base64
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

import test_animal_peak_event_vep as events
from animal_peak_control import AnimalPeakController, CONTROLS, ACTIONS
from animal_peak_dialog import QtCore, QtWidgets


class McpControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        events.EventVepQtTests.setUpClass(); cls.app = events.EventVepQtTests.app

    def setUp(self):
        events.EventVepQtTests.setUp(self)
        self.controller = AnimalPeakController(self.dialog)
        self.dialog._mcp_controller = self.controller
        self.temp = tempfile.TemporaryDirectory(prefix="animal_mcp_test_")
        self.dialog.mcp_export_root.setText(self.temp.name)
        self.dialog.resize(1360,960); self.dialog.parameter_toggle.setChecked(False)
        self.dialog.show(); self.app.processEvents()

    def tearDown(self):
        events.EventVepQtTests.tearDown(self)
        self.temp.cleanup()

    def call(self, operation, args=None, *, ok=True, revision="current"):
        if revision == "current": revision = self.controller.dispatch("get_state",{})["revision"]
        with patch.object(QtWidgets.QMessageBox,"warning", side_effect=AssertionError("unexpected modal warning")), \
             patch.object(QtWidgets.QMessageBox,"question", side_effect=AssertionError("unexpected modal question")):
            response = self.controller.dispatch(operation,args or {},revision)
        self.assertEqual(response["ok"],ok,response)
        return response

    def params(self, group, **values):
        return self.call("set_parameters",dict(group=group,values=values))

    def action(self, action, **options):
        return self.call("execute",dict(action=action,**options))

    def wait_job(self, response):
        job_id = response["result"]["job_id"]
        self.assertIsNotNone(job_id,response)
        deadline = time.monotonic()+30
        while self.dialog._busy and time.monotonic()<deadline:
            self.app.processEvents(); time.sleep(.003)
        self.app.processEvents()
        self.assertFalse(self.dialog._busy)
        job = self.call("get_jobs",dict(job_id=job_id))["result"]
        self.assertEqual(job["status"],"completed",job)
        return job

    def event(self, mode="letter"):
        self.params("event",alignment=mode,duration_ms=1000)
        self.wait_job(self.action("refresh_event"))

    def test_disabled_by_default_and_complete_discovery(self):
        d = self.dialog
        self.assertFalse(d.mcp_enabled_check.isChecked()); self.assertIsNone(d._mcp_bridge)
        state = self.call("get_state")["result"]
        self.assertEqual(state["source"]["aligned_channel_ids"],[7,22])
        description = self.call("describe")["result"]
        self.assertEqual(set(description["groups"]),set(CONTROLS))
        self.assertEqual(set(description["actions"]),set(ACTIONS))
        self.assertEqual(len(description["tabs"]),7)

    def test_revision_detects_human_edit_and_rejects_missing_and_stale_writes(self):
        revision = self.call("get_state")["revision"]
        self.dialog.pre_spin.setValue(60)
        response = self.call("set_parameters",dict(group="extraction",values=dict(post_ms=70)),revision=revision,ok=False)
        self.assertEqual(response["error"]["code"],"stale_revision")
        self.assertEqual(self.dialog.post_spin.value(),50)
        self.call("execute",dict(action="refresh_average"),revision=None,ok=False)

    def test_invalid_patch_is_atomic_unknown_or_code_operations_rejected(self):
        old = self.dialog.pre_spin.value()
        self.call("set_parameters",dict(group="extraction",values=dict(pre_ms=100,search_ms=-2)),ok=False)
        self.assertEqual(self.dialog.pre_spin.value(),old)
        self.call("set_parameters",dict(group="view",values=dict(__dict__="bad")),ok=False)
        self.call("eval",dict(code="raise Exception()"),ok=False)
        self.call("execute",dict(action="__getattribute__"),ok=False)
        self.call("set_parameters",dict(group="event",values=dict(duration_ms=float("nan"))),ok=False)

    def test_event_modes_baselines_patch_and_actual_numerical_mean_match_gui(self):
        for mode in ("letter","dot","animal"):
            self.event(mode)
            group = self.dialog.output["letter_vep"]["cache"]
            self.assertEqual(group["parameters"]["alignment"],mode)
            np.testing.assert_allclose(group["time_ms"][[0,-1]],[-1000,0] if mode=="animal" else [0,1000])
            np.testing.assert_allclose(group["mean"],np.mean(group["waves"],axis=0))
            self.assertEqual(group["rows"][0]["baseline_reference"],"Dot" if mode=="dot" else "LetterStart")
        self.call("set_parameters",dict(group="event",values=dict(start_ms=0,end_ms=1000)),ok=False)

    def test_selections_are_independent_and_event_list_shared_across_modes(self):
        self.call("set_trial_selection",dict(target="peaks",trial_ids=[1]))
        self.event()
        self.wait_job(self.call("set_trial_selection",dict(target="event",trial_ids=[2])))
        for mode in ("animal","dot"):
            self.event(mode)
            self.assertEqual(self.dialog.output["letter_vep"]["cache"]["selected_indices"],[1])
        self.assertEqual(self.dialog.output["average_trial_selections"][0]["trial_ids"],[1])
        self.call("set_trial_selection",dict(target="peaks",trial_ids=[]))
        self.call("set_trial_selection",dict(target="peaks",trial_ids=None))
        self.assertEqual(self.dialog.output["average_trial_selections"],[])
        self.call("set_trial_selection",dict(target="peaks",trial_ids=[999]),ok=False)

    def test_manual_preview_apply_restore_and_resize_preserve_original_waveform(self):
        self.call("navigate",dict(tab="details",channel=7,tag=4,trial_id=1,branch="peak"))
        row = self.dialog._selected_manual_row(); original = row["extreme_relative_ms"]
        self.params("manual",relative_ms=round(original-1,3))
        self.assertIsNotNone(self.dialog._manual_preview)
        self.wait_job(self.action("apply_manual"))
        self.assertEqual(self.dialog._selected_manual_row()["alignment_origin"],"manual")
        point = self.dialog._selected_manual_row()["extreme_local_sample"]
        self.params("extraction",pre_ms=80,post_ms=90)
        self.wait_job(self.action("refresh_average"))
        self.assertEqual(self.dialog._selected_manual_row()["extreme_local_sample"],point)
        self.wait_job(self.action("restore_manual"))
        self.assertAlmostEqual(self.dialog._selected_manual_row()["extreme_relative_ms"],original)

    def test_applied_width_y_and_letter_view_are_display_only_and_png_includes_title(self):
        self.event()
        group = self.dialog.output["letter_vep"]["cache"]
        before = group["mean"].copy()
        self.params("view",event_width_px=600,event_y_min=-30,event_y_max=40,letter_before_ms=2000)
        with patch.object(self.source,"read",side_effect=AssertionError("display unexpectedly reread")):
            self.action("apply_width"); self.action("apply_event_y")
            image = self.call("get_image",dict(plot="event"))["result"]
        self.assertTrue(base64.b64decode(image["data"]).startswith(b"\x89PNG"))
        self.assertTrue(image["includes_title"])
        self.assertEqual(self.dialog.letter_vep_plot.width(),600)
        np.testing.assert_allclose(self.dialog.letter_vep_plot.viewRange()[1],[-30,40])
        self.assertIs(self.dialog.output["letter_vep"]["cache"],group)
        np.testing.assert_array_equal(group["mean"],before)
        self.action("apply_letter_range")
        self.assertEqual(self.dialog.output["display_settings"]["letter_before_ms"],2000)
        self.action("auto_width"); self.action("auto_event_y")

    def test_all_plot_targets_tables_and_bounded_numeric_reads(self):
        self.event()
        for plot in ("event","average","histogram","details","animal_reference","manual_average"):
            self.call("get_image",dict(plot=plot))
        for table in ("records","event","similarity","rankings"):
            response = self.call("read_table",dict(table=table,offset=0,limit=1))["result"]
            self.assertLessEqual(len(response["rows"]),1)
        records = self.call("read_records",dict(channel=7,tag=4,trial_id=1,limit=1))["result"]
        self.assertEqual(len(records["rows"]),1)
        wave = self.call("read_waveform",dict(plot="event",offset=20,limit=30))["result"]
        self.assertEqual(len(wave["y"]),30)
        self.call("read_waveform",dict(limit=100000),ok=False)
        self.call("get_image",dict(plot="similarity"),ok=False)

    def test_filters_require_confirmation_and_preserve_points_with_restore(self):
        before = [r["extreme_local_sample"] for r in self.dialog.output["records"] if "extreme_local_sample" in r]
        self.params("filter",channels="7",mode="highpass",low_hz=1,order=3)
        response = self.call("execute",dict(action="filter"),ok=False)
        self.assertEqual(response["error"]["code"],"confirmation_required")
        self.wait_job(self.action("filter",confirm=True))
        self.assertEqual(self.dialog.output["secondary_filter_state"][0]["channel"],7)
        after = [r["extreme_local_sample"] for r in self.dialog.output["records"] if "extreme_local_sample" in r]
        self.assertEqual(before,after)
        self.wait_job(self.action("restore_filter",confirm=True))
        self.assertEqual(self.dialog.output["secondary_filter_state"],[])

    def test_reextract_requires_confirmation_and_similarity_tools_work(self):
        self.event()
        self.call("execute",dict(action="extract"),ok=False)
        self.params("extraction",localization_mode="trend",search_ms=1500)
        self.wait_job(self.action("extract",confirm=True))
        self.params("similarity",before_ms=40,after_ms=60,r_threshold=.5)
        self.wait_job(self.action("compute_similarity"))
        self.call("navigate",dict(tab="similarity",reference_trial_id=None,rank_trial_id=1,similarity_tab=2))
        self.call("get_image",dict(plot="similarity"))

    def test_exports_confined_to_authorized_root_and_do_not_overwrite(self):
        self.event()
        self.call("execute",dict(action="export_plot",filename="one.png"),ok=False)
        self.call("execute",dict(action="export_plot",filename="../escape.png",confirm=True),ok=False)
        self.action("export_plot",filename="one.png",confirm=True)
        self.assertTrue((Path(self.temp.name)/"one.png").is_file())
        self.call("execute",dict(action="export_plot",filename="one.png",confirm=True),ok=False)
        with patch.object(Figure,"savefig"):
            job = self.wait_job(self.action("export",directory="reports",confirm=True))
        folder = Path(job["export_directory"])
        self.assertTrue(folder.is_relative_to(Path(self.temp.name).resolve()))
        self.assertTrue((folder/"parameters.json").is_file())
        manifest = json.loads((folder/"parameters.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["mcp_control_history"])

    def test_ui_modal_blocks_write_and_aligned_channels_remain_read_only(self):
        with patch.object(QtWidgets.QApplication,"activeModalWidget",return_value=object()):
            response = self.call("execute",dict(action="auto_width"),ok=False)
        self.assertEqual(response["error"]["code"],"modal_dialog_open")
        self.dialog.channel_edit.setReadOnly(True)
        self.call("set_parameters",dict(group="extraction",values=dict(channels="7")),ok=False)


if __name__ == "__main__": unittest.main()
