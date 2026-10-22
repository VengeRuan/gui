"""Real loopback authentication, registry privacy, disconnect and lifecycle."""
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import test_animal_peak_mcp_control as control_tests
from animal_peak_bridge import protect_private_path
from animal_peak_rpc import call_bridge, list_sessions, BridgeError, API_VERSION, MAX_REQUEST_BYTES
from animal_peak_dialog import QtWidgets


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        control_tests.McpControlTests.setUpClass(); cls.app = control_tests.McpControlTests.app

    def setUp(self):
        control_tests.McpControlTests.setUp(self)
        self.registry = Path(self.temp.name)/"sessions"
        self.environment = patch.dict(os.environ, {"ANIMAL_PEAK_MCP_SESSION_DIR":str(self.registry)})
        self.environment.start()
        self.dialog.mcp_enabled_check.setChecked(True)
        self.assertIsNotNone(self.dialog._mcp_bridge,self.dialog.mcp_status.text())
        self.bridge = self.dialog._mcp_bridge
        self.session = self.bridge.session_id

    def tearDown(self):
        self.dialog.mcp_enabled_check.setChecked(False)
        self.environment.stop()
        control_tests.McpControlTests.tearDown(self)

    def threaded(self, function):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(function)
            deadline = time.monotonic()+20
            while not future.done() and time.monotonic()<deadline:
                self.app.processEvents(); time.sleep(.002)
            return future.result(timeout=1)

    def rpc(self, operation, arguments=None, revision=None):
        return self.threaded(lambda:call_bridge(self.session,operation,arguments,
            expected_revision=revision,directory=self.registry,timeout=3))

    def test_registry_omits_credentials_and_authenticated_rpc_uses_gui_thread(self):
        sessions = list_sessions(self.registry)
        self.assertEqual(len(sessions),1)
        self.assertNotIn("token",sessions[0]); self.assertNotIn("port",sessions[0])
        self.assertEqual(self.bridge.server.serverAddress().toString(),"127.0.0.1")
        state = self.rpc("get_state")
        response = self.rpc("set_parameters",dict(group="extraction",values=dict(pre_ms=60)),state["revision"])
        self.assertTrue(response["ok"])
        self.assertEqual(self.dialog.pre_spin.value(),60)

    def test_wrong_token_cannot_read_or_change_window(self):
        request = dict(api_version=API_VERSION,session_id=self.session,token="wrong",operation="set_parameters",
            arguments=dict(group="extraction",values=dict(pre_ms=99)),expected_revision=1)
        def send():
            with socket.create_connection(("127.0.0.1",self.bridge.server.serverPort()),timeout=3) as connection:
                connection.sendall(json.dumps(request).encode()+b"\n")
                return json.loads(connection.recv(4096))
        response = self.threaded(send)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"],"unauthorized")
        self.assertEqual(self.dialog.pre_spin.value(),50)

    def test_wrong_version_malformed_and_oversize_requests_do_not_crash_gui(self):
        port = self.bridge.server.serverPort()
        def send(payload):
            with socket.create_connection(("127.0.0.1",port),timeout=3) as connection:
                connection.sendall(payload)
                try: return connection.recv(4096)
                except ConnectionResetError: return b""
        self.assertIn(b"invalid_request",self.threaded(lambda:send(b"not json\n")))
        request = dict(api_version=999,session_id=self.session,token=self.bridge.token,operation="get_state")
        self.assertIn(b"unauthorized",self.threaded(lambda:send(json.dumps(request).encode()+b"\n")))
        self.assertEqual(self.threaded(lambda:send(b"x"*(MAX_REQUEST_BYTES+1))),b"")
        self.assertTrue(self.rpc("get_state")["ok"])

    def test_invalid_sessions_paths_and_failed_connections_do_not_leak_tokens(self):
        for session in ("../other","x",True,None):
            with self.assertRaises(BridgeError): call_bridge(session,"get_state",directory=self.registry)
        registry = self.bridge.registry_path
        self.dialog.mcp_enabled_check.setChecked(False)
        self.assertFalse(registry.exists())
        with self.assertRaises(BridgeError): call_bridge(self.session,"get_state",directory=self.registry)
        self.assertEqual(list_sessions(self.registry),[])

    def test_disabling_bridge_does_not_cancel_job_and_reenable_retains_history(self):
        captured = self.dialog.output
        release = threading.Event()
        def slow_analysis(_source,_markers,_matches,_settings,progress=None,cancel=None):
            while not release.wait(.01):
                if cancel(): raise ValueError("unexpected cancellation")
            return captured
        with patch("animal_peak_dialog.analyze_animal_peaks",side_effect=slow_analysis):
            state = self.rpc("get_state")
            response = self.rpc("execute",dict(action="extract"),state["revision"])
            job_id = response["result"]["job_id"]
            self.assertTrue(self.dialog._busy)
            self.dialog.mcp_enabled_check.setChecked(False)
            self.assertTrue(self.dialog._busy)
            self.dialog.mcp_enabled_check.setChecked(True)
            self.session = self.dialog._mcp_bridge.session_id
            release.set()
            deadline = time.monotonic()+10
            while self.dialog._busy and time.monotonic()<deadline:
                self.app.processEvents(); time.sleep(.005)
            self.app.processEvents()
        job = self.rpc("get_jobs",dict(job_id=job_id))["result"]
        self.assertEqual(job["status"],"completed")

    def test_async_errors_are_returned_as_jobs_without_modal_dialog(self):
        with patch("animal_peak_dialog.analyze_animal_peaks",side_effect=ValueError("synthetic failure")), \
             patch.object(QtWidgets.QMessageBox,"warning",side_effect=AssertionError("unexpected modal")):
            state = self.rpc("get_state")
            response = self.rpc("execute",dict(action="extract"),state["revision"])
            deadline = time.monotonic()+10
            while self.dialog._busy and time.monotonic()<deadline:
                self.app.processEvents(); time.sleep(.005)
            self.app.processEvents()
        job = self.rpc("get_jobs",dict(job_id=response["result"]["job_id"]))["result"]
        self.assertEqual(job["status"],"failed")
        self.assertIn("synthetic failure",job["error"])

    def test_cancel_running_job_rejects_concurrent_mutations(self):
        from animal_peak_analysis import AnalysisCancelled
        def slow_analysis(_source,_markers,_matches,_settings,progress=None,cancel=None):
            while not cancel(): time.sleep(.005)
            raise AnalysisCancelled("synthetic cancelled")
        with patch("animal_peak_dialog.analyze_animal_peaks",side_effect=slow_analysis):
            state = self.rpc("get_state")
            response = self.rpc("execute",dict(action="extract"),state["revision"])
            state = self.rpc("get_state")
            with self.assertRaises(BridgeError) as error:
                self.rpc("set_parameters",dict(group="view",values=dict(event_width_px=600)),state["revision"])
            self.assertEqual(error.exception.code,"busy")
            self.rpc("execute",dict(action="cancel",job_id=response["result"]["job_id"]),state["revision"])
            deadline=time.monotonic()+10
            while self.dialog._busy and time.monotonic()<deadline:
                self.app.processEvents(); time.sleep(.005)
            self.app.processEvents()
        self.assertEqual(self.rpc("get_jobs",dict(job_id=response["result"]["job_id"]))["result"]["status"],"cancelled")

    @unittest.skipUnless(os.name=="nt","Windows ACL only")
    def test_private_acl_removes_explicit_everyone_grants(self):
        import win32security
        path=Path(self.temp.name)/"acl_test.json"
        path.write_text("test",encoding="utf-8")
        security=win32security.GetFileSecurity(str(path),win32security.DACL_SECURITY_INFORMATION)
        acl=security.GetSecurityDescriptorDacl()
        acl.AddAccessAllowedAce(win32security.ACL_REVISION,0x1f01ff,win32security.ConvertStringSidToSid("S-1-1-0"))
        security.SetSecurityDescriptorDacl(True,acl,False)
        win32security.SetFileSecurity(str(path),win32security.DACL_SECURITY_INFORMATION,security)
        protect_private_path(path)
        acl=win32security.GetFileSecurity(str(path),win32security.DACL_SECURITY_INFORMATION).GetSecurityDescriptorDacl()
        sids=[win32security.ConvertSidToStringSid(acl.GetAce(i)[2]) for i in range(acl.GetAceCount())]
        self.assertEqual(len(sids),3)
        self.assertNotIn("S-1-1-0",sids)


if __name__ == "__main__": unittest.main()
