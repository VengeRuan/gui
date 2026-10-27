"""Page 2/3 integration tests use only temporary synthetic recordings."""
import base64
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import h5py
import numpy as np
from qt_gui import QApplication, QtAnalysisGUI
from qt_data_model import ArraySource
from animal_peak_rpc import call_bridge, list_sessions, BridgeError


class PipelineMcpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"SD_GUI_AUDIT_DB": str(self.root/"audit.sqlite3"),
            "ANIMAL_PEAK_MCP_SESSION_DIR": str(self.root/"sessions")})
        self.env.start()
        self.w = QtAnalysisGUI("MCP synthetic test")
        self.c = self.w._pipeline_mcp_controller
        self.w.pipeline_mcp_check.setChecked(True)
        self.assertIsNotNone(self.w._pipeline_mcp_bridge, self.w.pipeline_mcp_status.text())
        self.source = ArraySource(np.random.default_rng(2).normal(size=(20000, 3)).astype("float32"),
                                  1000, channel_ids=[7, 22, 82])
        self.w.source = self.w.active_source = self.source
        self.w.preprocessed_preview.set_source(self.source, defer_initial_draw=True)
        self.w.original_preprocess_preview.set_source(self.source, defer_initial_draw=True)
        self.log = self.root/"log.txt"
        self.log.write_text("12:00:00 Recording ON\n12:00:01 PROGRAM_START\n12:00:10 PROGRAM_END\n", encoding="utf-8")
        self.events = self.root/"events.csv"
        with self.events.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"column_{i}" for i in range(15)])
            for symbol, dot, letter in (("|", 2, 5), ("o", 9, 12)):
                row = [""]*15; row[1], row[12], row[13] = symbol, dot, letter
                writer.writerow(row)

    def tearDown(self):
        deadline = time.monotonic()+15
        while self.c._running_workers() and time.monotonic() < deadline:
            self.app.processEvents(); time.sleep(.01)
        self.w.pipeline_mcp_check.setChecked(False)
        self.w._audit_session_closed = True
        self.w._cleanup_processing_cache_on_close()
        self.w.close(); self.w.deleteLater(); self.app.processEvents()
        self.env.stop(); self.temp.cleanup()

    def read(self, op="get_state", **args):
        result = self.c.dispatch(op, args)
        self.assertTrue(result["ok"], result)
        return result

    def write(self, op, **args):
        revision = self.read()["revision"]
        result = self.c.dispatch(op, args, revision)
        self.assertTrue(result["ok"], result)
        return result

    def run_action(self, action, **options):
        response = self.write("execute", action=action, **options)
        job_id = response["result"]["job_id"]
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            self.app.processEvents()
            job = self.read("get_jobs", job_id=job_id)["result"]
            if job["status"] != "running": return job
            time.sleep(.005)
        self.fail(f"job timeout: {job}")

    def good_channels(self):
        self.w._bad_channel_check_completed = True
        self.w._bad_channel_result_channel_ids = (7, 22, 82)
        self.w.good_channel_ids = {7, 22, 82}

    def configure_alignment(self):
        self.good_channels()
        self.write("set_parameters", group="alignment", values=dict(
            log_path=str(self.log), events_path=str(self.events), detection_path="", task_mode="Letter"))
        self.write("navigate", page="alignment", channels=[7, 82])

    def test_registry_consent_and_real_loopback(self):
        sessions = list_sessions(self.root/"sessions")
        self.assertEqual(sessions[0]["session_kind"], "pipeline")
        self.assertNotIn("token", sessions[0]); self.assertNotIn("port", sessions[0])
        session = self.w._pipeline_mcp_bridge.session_id
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            future = executor.submit(call_bridge, session, "get_state", directory=self.root/"sessions", timeout=4)
            while not future.done(): self.app.processEvents(); time.sleep(.002)
            self.assertEqual(future.result()["result"]["session_kind"], "pipeline")
        self.w.pipeline_mcp_check.setChecked(False)
        self.assertEqual(list_sessions(self.root/"sessions"), [])
        self.assertFalse(self.c.dispatch("get_state", {})["ok"])

    def test_describe_all_widgets_and_atomic_validation_and_human_revision(self):
        controls = self.read("describe")["result"]
        self.assertIn("preprocess", controls["groups"])
        before = self.w.preprocess_filter_low_var.text()
        revision = self.read()["revision"]
        error = self.c.dispatch("set_parameters", dict(group="preprocess", values={"low_hz": 5, "hp_order": 0}), revision)
        self.assertFalse(error["ok"]); self.assertEqual(self.w.preprocess_filter_low_var.text(), before)
        self.w.preprocess_filter_low_var.setText("2")
        stale = self.c.dispatch("set_parameters", dict(group="preprocess", values={"low_hz": 3}), revision)
        self.assertEqual(stale["error"]["code"], "stale_revision")
        self.write("set_parameters", group="preprocess", values=dict(channels=[7, 82], filter_mode="bandpass", low_hz=1, high_hz=100))
        self.assertEqual(self.w.preprocess_filter_channels_var.text(), "7,82")
        error = self.c.dispatch("set_parameters", dict(group="preprocess", values={"channels": [7, 999]}), self.read()["revision"])
        self.assertFalse(error["ok"]); self.assertEqual(self.w.preprocess_filter_channels_var.text(), "7,82")

    def test_real_filter_job_busy_confirmation_and_waveform(self):
        revision = self.read()["revision"]
        no = self.c.dispatch("execute", dict(action="filter"), revision)
        self.assertEqual(no["error"]["code"], "confirmation_required")
        self.write("set_parameters", group="preprocess", values=dict(channels=[7], filter_mode="highpass", low_hz=1, notch=False))
        queued = self.write("execute", action="filter", confirm=True)
        blocked = self.c.dispatch("navigate", dict(page="alignment"), self.read()["revision"])
        self.assertEqual(blocked["error"]["code"], "busy")
        job_id = queued["result"]["job_id"]
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            self.app.processEvents(); job = self.read("get_jobs", job_id=job_id)["result"]
            if job["status"] != "running": break
            time.sleep(.005)
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(self.w._preprocess_filtered_channel_ids, {7})
        np.testing.assert_array_equal(self.w.active_source.read(0, 100, [1]), self.source.read(0, 100, [1]))
        job = self.run_action("refresh_preprocess", channel=7, start_sec=0, end_sec=2, full_duration=False)
        self.assertEqual(job["status"], "completed", job)
        wave = self.read("read_waveform", plot="preprocess", limit=10)["result"]
        self.assertEqual(len(wave["y"]), 10)
        image = self.read("get_image", plot="preprocess")["result"]
        self.assertTrue(base64.b64decode(image["data"]).startswith(b"\x89PNG"))

    def test_alignment_markers_csv_and_stale_inputs(self):
        self.configure_alignment()
        job = self.run_action("align", confirm=True)
        self.assertEqual(job["status"], "completed", job)
        self.assertFalse(self.read()["result"]["alignment"]["stale"])
        rows = self.read("read_table", table="markers", limit=1)["result"]
        self.assertGreaterEqual(rows["total"], 2)
        annotations = self.read("read_table", table="annotations")["result"]["rows"]
        self.assertAlmostEqual(annotations[0]["csv"]["tag100_start_eeg_sec"], 3)
        self.assertTrue(self.read("get_image", plot="alignment")["result"]["includes_title"])
        self.events.write_text(self.events.read_text(encoding="utf-8-sig")+"\n", encoding="utf-8-sig")
        self.assertTrue(self.read()["result"]["alignment"]["stale"])

    def test_multisegment_requires_explicit_selection_no_modal_and_preserves_previous(self):
        self.configure_alignment()
        self.assertEqual(self.run_action("align", confirm=True)["status"], "completed")
        previous = self.w.stim_markers
        self.log.write_text(self.log.read_text()+"12:00:11 PROGRAM_START\n12:00:19 PROGRAM_END\n", encoding="utf-8")
        rows = self.read("read_table", table="segments")["result"]["rows"]
        self.assertEqual([row["segment_index"] for row in rows], [0, 1])
        job = self.run_action("align", confirm=True)
        self.assertEqual(job["status"], "failed", job)
        self.assertIs(self.w.stim_markers, previous)
        self.assertIn("segment_index", job["error"])
        job = self.run_action("align", confirm=True, segment_index=1)
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(self.w.timing_info["program_segment_index"], 2)

    def test_bad_detection_does_not_partially_replace_alignment(self):
        self.configure_alignment()
        self.assertEqual(self.run_action("align", confirm=True)["status"], "completed")
        old = self.w.stim_markers
        bad = self.root/"bad.txt"; bad.write_text("not behavior data", encoding="utf-8")
        self.write("set_parameters", group="alignment", values=dict(detection_path=str(bad)))
        job = self.run_action("align", confirm=True)
        self.assertEqual(job["status"], "failed", job)
        self.assertIs(self.w.stim_markers, old)

    def test_export_refuses_existing_h5_csv_and_input(self):
        self.w.preprocessed_source = self.source
        path = self.root/"out.h5"
        path.with_suffix(".csv").write_text("keep me", encoding="utf-8")
        no = self.c.dispatch("execute", dict(action="export_preprocessed", path=str(path), confirm=True), self.read()["revision"])
        self.assertFalse(no["ok"])
        self.assertEqual(path.with_suffix(".csv").read_text(), "keep me")
        output = self.root/"new.h5"
        job = self.run_action("export_preprocessed", path=str(output), confirm=True)
        self.assertEqual(job["status"], "completed", job)
        self.assertTrue(output.is_file()); self.assertTrue(output.with_suffix(".csv").is_file())

    def test_review_health_scope_and_disallow_qc_skip(self):
        self.good_channels()
        self.assertEqual(self.run_action("review_channels", channels=[22], classification="bad", confirm=True)["status"], "completed")
        self.assertEqual(self.w.bad_channel_ids, {22})
        rows = self.read("read_table", table="qc")["result"]["rows"]
        self.assertEqual(rows[1]["override"], "bad")
        revision = self.read()["revision"]
        bad = self.c.dispatch("navigate", dict(channels=[22]), revision)
        self.assertFalse(bad["ok"])
        self.w._bad_channel_check_completed = False
        bad = self.c.dispatch("execute", dict(action="stream_export", confirm=True), self.read()["revision"])
        self.assertFalse(bad["ok"]); self.assertIn("QC", bad["error"]["message"])

    def test_queued_action_rechecks_human_changes_and_revocation(self):
        result = self.write("execute", action="filter", confirm=True)
        self.w.preprocess_filter_low_var.setText("2")
        self.app.processEvents()
        job = self.read("get_jobs", job_id=result["result"]["job_id"])["result"]
        self.assertEqual(job["status"], "failed")
        self.assertIn("已改变", job["error"])
        result = self.write("execute", action="filter", confirm=True)
        self.w.pipeline_mcp_check.setChecked(False)
        self.app.processEvents()
        self.assertIn("撤销", self.c.jobs[result["result"]["job_id"]]["error"])

    def test_one_click_quality_car_notch_and_repeat_authorization(self):
        self.write("set_parameters", group="preprocess", values=dict(one_click_quality=True,
            one_click_car=True, one_click_notch=True))
        job = self.run_action("one_click", confirm=True)
        self.assertEqual(job["status"], "completed", job)
        self.assertTrue(self.w._bad_channel_check_completed)
        self.assertEqual(set(self.w.good_channel_ids), {7, 22, 82})
        self.assertIsNotNone(self.w.preprocessed_source)
        self.assertEqual(self.w._preprocess_notched_channel_ids, {7, 22, 82})
        self.w._loaded_preprocessed_operations = {"filter"}
        self.w._repeat_preprocessing_acknowledged = set()
        job = self.run_action("filter", confirm=True)
        self.assertEqual(job["status"], "failed", job)
        self.assertIn("repeat_confirm", job["error"])
        self.write("set_parameters", group="preprocess", values=dict(low_hz=1, filter_mode="highpass"))
        job = self.run_action("filter", confirm=True, repeat_confirm=True)
        self.assertEqual(job["status"], "completed", job)

    def test_worker_failure_and_disable_reenable_retains_history(self):
        from qt_gui import CooperativeWorker
        from PyQt6 import QtCore
        import threading
        release = threading.Event()
        class FakeWorker(CooperativeWorker):
            completed = QtCore.pyqtSignal(object)
            failed = QtCore.pyqtSignal(str)
            def run(self):
                release.wait(3)
                self.failed.emit("synthetic failure")
        worker = FakeWorker()
        def start():
            self.w._preprocess_worker = worker
            self.w._begin_shared_preprocess_task(worker, "synthetic")
            worker.start()
        with patch.object(self.w, "run_preprocess", side_effect=start):
            response = self.write("execute", action="filter", confirm=True)
            self.app.processEvents()
            job_id = response["result"]["job_id"]
            self.w.pipeline_mcp_check.setChecked(False)
            self.assertTrue(worker.isRunning())
            self.w.pipeline_mcp_check.setChecked(True)
            release.set()
            deadline = time.monotonic()+5
            while worker.isRunning() and time.monotonic() < deadline:
                self.app.processEvents(); time.sleep(.005)
            self.app.processEvents()
            job = self.read("get_jobs", job_id=job_id)["result"]
            self.assertEqual(job["status"], "failed", job)
            self.assertEqual(job["error"], "synthetic failure")

    def test_stdio_protocol_pipeline_round_trip(self):
        import asyncio
        import sys
        from datetime import timedelta
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        self.configure_alignment()
        async def flow():
            parameters = StdioServerParameters(command=sys.executable,
                args=[str(Path(__file__).with_name("animal_peak_mcp_server.py"))],
                env={**os.environ, "ANIMAL_PEAK_MCP_SESSION_DIR": str(self.root/"sessions")})
            async with stdio_client(parameters) as (reader, writer):
                async with ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=10)) as client:
                    await client.initialize()
                    async def call(name, **args):
                        response = await client.call_tool(name, args)
                        self.assertFalse(response.isError, response)
                        return json.loads(response.content[0].text)
                    sessions = (await call("list_sessions"))["sessions"]
                    session = next(s["session_id"] for s in sessions if s["session_kind"] == "pipeline")
                    state = await call("get_state", session_id=session)
                    accepted = await call("execute", session_id=session, expected_revision=state["revision"],
                                          action="align", options=dict(confirm=True))
                    for _ in range(100):
                        job = await call("get_jobs", session_id=session, job_id=accepted["result"]["job_id"])
                        if job["result"]["status"] != "running": break
                        await asyncio.sleep(.02)
                    self.assertEqual(job["result"]["status"], "completed", job)
                    self.assertGreater((await call("read_table", session_id=session, table="markers"))["result"]["total"], 0)
                    image = await client.call_tool("get_plot_image", dict(session_id=session, plot="alignment"))
                    self.assertFalse(image.isError, image)
                    self.assertTrue(any(b.type == "image" for b in image.content))
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            future = executor.submit(lambda: asyncio.run(flow()))
            deadline = time.monotonic()+40
            while not future.done() and time.monotonic() < deadline:
                self.app.processEvents(); time.sleep(.003)
            future.result(timeout=1)

    def test_load_full_h5_and_reject_invalid_before_replacing_source(self):
        invalid = self.root/"invalid.h5"; invalid.write_text("not hdf5", encoding="utf-8")
        old = self.w.source
        no = self.c.dispatch("execute", dict(action="load_h5", path=str(invalid), confirm=True), self.read()["revision"])
        self.assertFalse(no["ok"]); self.assertIs(self.w.source, old)
        path = self.root/"synthetic.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("raw_data", data=self.source.data)
            h5.create_dataset("channel_ids", data=[7, 22, 82])
            h5.attrs["fs"] = 1000.
        job = self.run_action("load_h5", path=str(path), confirm=True)
        self.assertEqual(job["status"], "completed", job)
        state = self.read()["result"]
        self.assertEqual(state["source"]["rows"], 20000)
        self.assertEqual(state["source"]["channel_ids"], [7, 22, 82])
        self.assertEqual(state["source"]["path"], str(path))

    def test_pause_resume_cancel_at_worker_checkpoints(self):
        from qt_gui import CooperativeWorker, CooperativeTaskCancelled
        from PyQt6 import QtCore
        class SlowWorker(CooperativeWorker):
            completed = QtCore.pyqtSignal(object)
            failed = QtCore.pyqtSignal(str)
            def run(self):
                try:
                    for _ in range(1000):
                        self._checkpoint(); time.sleep(.005)
                    self.completed.emit({})
                except CooperativeTaskCancelled:
                    self.cancelled.emit()
        worker = SlowWorker()
        def start():
            self.w._preprocess_worker = worker
            self.w._begin_shared_preprocess_task(worker, "synthetic cancellable")
            worker.start()
        with patch.object(self.w, "run_preprocess", side_effect=start):
            response = self.write("execute", action="filter", confirm=True)
            job_id = response["result"]["job_id"]
            self.app.processEvents()
            self.write("execute", action="pause", job_id=job_id)
            self.assertTrue(worker._paused)
            self.write("execute", action="resume", job_id=job_id)
            self.assertFalse(worker._paused)
            self.write("execute", action="cancel", job_id=job_id)
            deadline = time.monotonic()+5
            while worker.isRunning() and time.monotonic() < deadline:
                self.app.processEvents(); time.sleep(.005)
            self.app.processEvents()
            job = self.read("get_jobs", job_id=job_id)["result"]
            self.assertEqual(job["status"], "cancelled", job)

    def test_changed_preview_refuses_old_image_until_refresh(self):
        self.configure_alignment()
        self.assertEqual(self.run_action("align", confirm=True)["status"], "completed")
        self.write("set_parameters", group="preview", values=dict(duration_sec=2, full_duration=False))
        no = self.c.dispatch("get_image", dict(plot="alignment"))
        self.assertFalse(no["ok"])
        self.assertEqual(self.run_action("refresh_alignment")["status"], "completed")
        self.read("get_image", plot="alignment")


if __name__ == "__main__": unittest.main()
